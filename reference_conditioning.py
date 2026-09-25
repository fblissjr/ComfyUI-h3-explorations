"""Typed, ordered reference conditioning for MiniMax H3.

The core node accepts four parallel socket groups and reconstructs ordering
and video/audio pairing from their names.  These nodes instead carry one
copy-on-append tuple.  A video record owns its metadata and optional
soundtrack, and the compiler walks the tuple exactly once to build both the
Qwen presentation and the DiT payload.

Still images retain their per-record ``match``/``max`` policy. Reference video
preparation is selected once at the compiler boundary: ``comfy`` keeps core's
no-upscale/shared-frame behaviour, and ``release`` both puts the VAE view on
the release canvas and uses the release processor settings. The two release
stages are one policy because enabling the upscale alone overshoots Qwen's
long-clip budget. A third policy, ``encoder``, read a contract stamped by the
AWQ adapter; that lane closed and the policy went with it on 2026-09-13
(`docs/wiki/decisions.md`).

**Every still is read twice.** The video VAE encodes one copy into reference
latent rows the DiT attends on every sampling step; Qwen3-VL reads a copy as
vision tokens placed in the text segment ahead of the prompt. `size_policy`
sizes the first, `qwen_view` the second, and `MiniMaxH3ReferenceReport`
(`reference_report.py`) prices both before anything is encoded.

This is local release-parity handling. It does not change native ComfyUI's
``MiniMaxH3ReferenceToVideo`` node or close either upstream gap.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import functools
import logging
import math
from typing import Any

import node_helpers
import torch
from comfy_api.latest import io, ui
from comfy_extras.nodes_minimax_h3 import (
    CANVAS_MULTIPLE,
    FPS,
    REF_IMAGE_SHORT_EDGE,
    _empty_av_latent,
    _encode_ref_audio,
    _resize,
    adapt_canvas,
)

from .h3_rules import (REF_QWEN_SHORT_EDGE, aspect_in_range,
                       describe_aspect_range)
from .reference_order import AudioRef, ImageRef, VideoRef, assign_labels
from .reference_geometry import (
    IMAGE_POLICIES,
    SIZE_POLICIES,
    fit_reference_image,
    latent_rows,
    qwen_image_settings,
    qwen_image_size,
    snap_to_multiple,
)
from .vendor_config import (
    image_pixel_bounds,
    patch_geometry,
    video_patch_geometry,
    video_pixel_bounds,
)
logger = logging.getLogger(__name__)

H3References = io.Custom("MINIMAX_H3_REFERENCES")
VHSVideoInfo = io.Custom("VHS_VIDEOINFO")
VIDEO_POLICIES = ("comfy", "release")


@dataclass(frozen=True)
class RuntimeImageReference:
    """One still reference and the stage-one sizing decision made for it.

    `short_edge` and `allow_upscale` ride the record rather than being applied
    upstream so that exactly one resize happens, at the compiler, where the
    canvas is also known. Their defaults reproduce ComfyUI's own sizing, which
    is what a saved graph built before they existed must keep getting.
    """

    image: Any
    size_policy: str
    short_edge: int = REF_IMAGE_SHORT_EDGE
    allow_upscale: bool = False
    # A separate view for the encoder. 0 means Qwen sees the VAE view, which
    # is what every serving implementation does and the node default since
    # 2026-09-13. N scales the SOURCE so its shorter side reaches N, for the
    # conditioner alone; the VAE view is unchanged.
    # `docs/h3_conditioning_end_to_end.md` section 1b is why the two branches
    # need not share a geometry.
    qwen_short_edge: int = 0


@dataclass(frozen=True)
class RuntimeVideoReference:
    frames: Any
    loaded_fps: float
    soundtrack: Any | None


@dataclass(frozen=True)
class RuntimeAudioReference:
    audio: Any


RuntimeReference = RuntimeImageReference | RuntimeVideoReference | RuntimeAudioReference


def _reference_tuple(references) -> tuple[RuntimeReference, ...]:
    """Validate the runtime value instead of accepting a list-shaped impostor."""
    if references is None:
        return ()
    if not isinstance(references, tuple):
        raise TypeError(
            "MINIMAX_H3_REFERENCES must come from an H3 append-reference "
            f"node; got {type(references).__name__} instead of its tuple"
        )
    allowed = (RuntimeImageReference, RuntimeVideoReference, RuntimeAudioReference)
    for index, record in enumerate(references):
        if not isinstance(record, allowed):
            raise TypeError(
                f"MINIMAX_H3_REFERENCES item {index} is not an H3 reference "
                f"record: {type(record).__name__}"
            )
    return references


def _image_shape(image, field: str) -> tuple[int, int, int]:
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError(
            f"{field} must be an IMAGE tensor [frames, height, width, channels]"
        )
    count, height, width = image.shape[:3]
    if count < 1 or height < 1 or width < 1:
        raise ValueError(f"{field} is empty: shape={tuple(image.shape)}")
    return int(count), int(height), int(width)


def _audio_shape(audio, field: str) -> tuple[torch.Tensor, int]:
    # VHS returns LazyAudioMap, a Mapping that decodes through ffmpeg on first
    # access. Core LoadAudio returns a plain dict. AUDIO is the public socket
    # type for both, so rejecting everything but dict makes a schema-valid VHS
    # soundtrack fail only at execution time.
    if not isinstance(audio, Mapping):
        raise ValueError(f"{field} must be a Comfy AUDIO value")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError(f"{field}.waveform must have shape [batch, channels, samples]")
    if waveform.shape[0] < 1 or waveform.shape[1] < 1 or waveform.shape[2] < 1:
        raise ValueError(f"{field}.waveform is empty: shape={tuple(waveform.shape)}")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)):
        raise ValueError(f"{field}.sample_rate must be a positive number")
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"{field}.sample_rate must be positive, got {sample_rate}")
    return waveform, sample_rate


def _loaded_fps(video_info, frame_count: int, height: int, width: int) -> float:
    if not isinstance(video_info, dict):
        raise ValueError("video_info must be the VHS_VIDEOINFO from the frames' loader")
    loaded_fps = video_info.get("loaded_fps")
    if isinstance(loaded_fps, bool) or not isinstance(loaded_fps, (int, float)):
        raise ValueError("VHS_VIDEOINFO.loaded_fps must be a positive number")
    loaded_fps = float(loaded_fps)
    if not math.isfinite(loaded_fps) or loaded_fps <= 0:
        raise ValueError(
            f"VHS_VIDEOINFO.loaded_fps must be finite and positive, got {loaded_fps}"
        )

    # These are redundant with the static same-loader/slot check on a saved
    # graph, intentionally. API callers can execute a node without that graph
    # context, so the runtime value must defend its own ownership claim too.
    expected = {
        "loaded_frame_count": frame_count,
        "loaded_height": height,
        "loaded_width": width,
    }
    for key, actual in expected.items():
        value = video_info.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"VHS_VIDEOINFO.{key} is missing or not numeric")
        if int(value) != actual:
            raise ValueError(
                f"VHS_VIDEOINFO.{key} says {value}, but the wired frames say "
                f"{actual}; frames and metadata must describe one decode"
            )
    return loaded_fps


def _resample_video_to_24fps(frames, loaded_fps: float):
    """Nearest-timestamp normalization with the first source frame at t=0.

    VHS already returns 24 fps when ``force_rate=24``.  That path returns the
    original tensor without an allocation.  Other rates get a deterministic
    timestamp map before either Qwen or the VAE sees the material, so the two
    encoders cannot disagree about which clip the metadata described.
    """
    if math.isclose(loaded_fps, FPS, rel_tol=0.0, abs_tol=1e-6):
        return frames
    source_count = int(frames.shape[0])
    # Frames are samples at 0, 1/fps, ..., (N-1)/fps.  Include every 24 fps
    # target timestamp that lies within that closed source interval.
    target_count = int(math.floor((source_count - 1) * FPS / loaded_fps + 1e-9)) + 1
    target_count = max(1, target_count)
    positions = torch.arange(
        target_count, device=frames.device, dtype=torch.float64
    ) * (loaded_fps / FPS)
    indices = positions.round().to(torch.long).clamp_(0, source_count - 1)
    return frames.index_select(0, indices)


def _prepare_audio(audio, duration: float, field: str):
    """Return stereo audio capped at the aligned target duration."""
    waveform, sample_rate = _audio_shape(audio, field)
    channels = int(waveform.shape[1])
    if channels == 1:
        waveform = waveform.repeat(1, 2, 1)
    elif channels != 2:
        raise ValueError(
            f"{field} must be mono or stereo; got {channels} channels. "
            "Choose the intended pair before H3 reference conditioning."
        )
    sample_count = min(
        int(waveform.shape[-1]), int(round(duration * sample_rate))
    )
    if sample_count < 1:
        raise ValueError(f"{field} has no samples inside the target duration")
    # A fresh dict prevents normalization from mutating a value used by a
    # second graph branch. The slice itself is a view, which is sufficient:
    # the audio encoder reads it and never writes it.
    return {"waveform": waveform[..., :sample_count], "sample_rate": sample_rate}


def _encode_ref_audio_aligned(audio_vae, audio):
    """Encode a reference waveform with its END zero-padded to the VAE's hop.

    Core's `_encode_ref_audio` hands the waveform to `VAE.encode`, whose generic
    crop (`comfy/sd.py::vae_encode_crop_pixels`) narrows the SAMPLE axis to a
    multiple of `spacial_compression_encode()` and takes the remainder off both
    ends, so the leading samples go and a soundtrack drifts early against its
    own frames: gap 16 in `docs/comfyui_vendor_gaps.md`, measured by
    `bench/audit_ref_audio_crop.py`.

    The release pads the end instead. Its audio VAE's `preprocess`
    right-pads with zeros to a whole number of hops, the hop being the product
    of `encoder_rates` (`coderef/MiniMax-H3/Ref2VA/audio_vae/dac_audio_vae.py`,
    `preprocess`), and sglang's port does the same
    (`minimax_h3_audio_vae/audio_vae.py::preprocess`). Credit where it is due:
    the behaviour is MiniMax's, and silveroxides/ComfyUI-UtilsCollection
    (`image_helpers.py`, its H3 reference-video audio path) is where this repo
    saw it done in a ComfyUI pack first (2026-09-10).

    Padding here, at the VAE's own rate and to the VAE's own ratio -- the same
    `spacial_compression_encode()` the crop reads -- makes the crop a no-op, so
    no sample is dropped and the last partial hop becomes one more latent step,
    as the release encodes it. Resampling first is the same call core makes
    (`audio_resample.py`); core then sees the VAE's rate and does not
    resample again.
    Core's own H3 nodes still take the crop until Comfy-Org/ComfyUI#15972 or an
    equivalent lands; this is our reference path only.
    """
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    vae_rate = getattr(audio_vae, "audio_sample_rate", 32000)  # core's fallback
    if sample_rate != vae_rate:
        from .audio_resample import resample
        waveform = resample(waveform, sample_rate, vae_rate)
    hop = int(audio_vae.spacial_compression_encode())
    right = -int(waveform.shape[-1]) % hop
    if right:
        waveform = torch.nn.functional.pad(waveform, (0, right))
    return _encode_ref_audio(audio_vae, {"waveform": waveform, "sample_rate": vae_rate})


def _prepare_reference_video(frames, loaded_fps: float, frame_count: int):
    frames = _resample_video_to_24fps(frames, loaded_fps)
    if frames.shape[0] > frame_count:
        frames = frames[:frame_count]
    n = int(frames.shape[0])
    if n < 5:
        raise ValueError(
            "MiniMax H3 reference videos need at least 5 frames after 24 fps "
            "normalization (~0.2 seconds)"
        )
    while n % 17 != 5:
        n -= 1
    return frames[:n]


def _qwen_image_settings(image_policy: str) -> tuple[tuple[int, int], dict]:
    """Delegate to `reference_geometry`, which the static readers share."""
    return qwen_image_settings(image_policy)


def _configured_qwen_image_size(
    width: int, height: int, image_policy: str,
) -> tuple[int, int]:
    """Delegate to `reference_geometry`, which the static readers share."""
    return qwen_image_size(width, height, image_policy)


def _qwen_video_settings(video_policy: str) -> tuple[tuple[int, int], dict]:
    """Return the settings owned by the selected Qwen preprocessing policy.

    Only `release` has one: the release's own video processor declaration,
    read through `vendor_config`. `comfy` runs core's per-pair processor
    inside the encoder and pre-applies nothing here.
    """
    if video_policy == "release":
        return video_pixel_bounds(), video_patch_geometry()
    raise ValueError(f"no configured Qwen processor for policy {video_policy!r}")


def _configured_qwen_video_size(
    sampled_count: int, width: int, height: int, video_policy: str,
) -> tuple[int, int]:
    """Return a configured processor's Qwen view as ``(width, height)``.

    The clip-wide pixel budget is divided by the RAW 2 fps sample count. The
    odd repeat pad belongs to Comfy's later temporal-block presentation and
    must not be passed here: at 31 versus 32 samples it changes the spatial
    grid despite leaving the temporal-block count unchanged.

    ``smart_resize`` is imported from the installed processor rather than
    copied. Bounds and patch geometry come from the selected policy's config,
    never from literals or Comfy's shared Qwen defaults.
    """
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize

    (min_pixels, max_pixels), geometry = _qwen_video_settings(video_policy)
    # `smart_resize` needs a full temporal patch, not merely one frame: it
    # raises `t:1 must be larger than temporal_factor:2` from inside
    # transformers, which names neither the reference nor the policy. The bound
    # is read from the release geometry rather than written as 2, because it is
    # the same constant `smart_resize` is about to enforce.
    #
    # This is error handling, NOT support for shorter clips. The release
    # deliberately requires two 2 fps samples. Reference lengths are snapped to
    # 17n+5 by `_prepare_reference_video`, and `sample_indices` steps by 12, so
    # the legal counts are 5, 22, 39, ... and 5 yields exactly one sample.
    # Release mode's practical minimum is therefore **22 prepared frames**;
    # comfy mode still accepts 5, which is why this refuses here rather than
    # tightening `_prepare_reference_video` for everyone.
    temporal_factor = int(geometry["temporal_patch_size"])
    if sampled_count < temporal_factor:
        raise ValueError(
            f"{video_policy} video policy needs at least {temporal_factor} sampled "
            f"frames at 2 fps, got {sampled_count}. Reference lengths snap to "
            f"17n+5, so the shortest clip this policy can take is 22 prepared "
            f"frames (~0.9s); 5 frames samples to one. Use video_policy="
            f"'comfy' for a reference this short."
        )
    target_h, target_w = smart_resize(
        num_frames=sampled_count,
        height=height,
        width=width,
        temporal_factor=int(geometry["temporal_patch_size"]),
        factor=int(geometry["patch_size"]) * int(geometry["merge_size"]),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return int(target_w), int(target_h)


def _release_qwen_video_size(
    sampled_count: int, width: int, height: int
) -> tuple[int, int]:
    """Compatibility wrapper for the release-policy sizing seam."""
    return _configured_qwen_video_size(
        sampled_count, width, height, video_policy="release"
    )


def _configured_qwen_video_frames(frames, video_policy: str):
    """Resize one raw sampled clip with the policy's bicubic processor.

    Only the sampled Qwen view reaches this function. The full-rate frames for
    the video VAE remain on the release canvas, matching the serving path's
    two independently prepared views.
    """
    sampled_count, height, width = _image_shape(frames, "sampled Qwen video")
    target_w, target_h = _configured_qwen_video_size(
        sampled_count, width, height, video_policy
    )
    if (target_w, target_h) == (width, height):
        return frames

    # The processor's resize method is the pixel authority as well as
    # `smart_resize` being the geometry authority: it preserves the release's
    # bicubic kernel instead of substituting Comfy's bilinear video-block path.
    processor = _qwen_video_processor(video_policy)
    # The released serving path decodes media into uint8 and the HF processor
    # resizes those pixels before its 1/255 rescale. Comfy IMAGE values arrive
    # as floats in [0, 1], so temporarily restore that uint8 boundary; running
    # bicubic directly on floats can retain overshoot values the release clips.
    source_dtype = frames.dtype
    if frames.is_floating_point():
        processor_pixels = frames.mul(255).round().clamp_(0, 255).to(torch.uint8)
    else:
        processor_pixels = frames.to(torch.uint8)
    channel_first = processor_pixels.permute(0, 3, 1, 2).unsqueeze(0)
    resized = processor.resize(
        videos=channel_first,
        size=processor.size,
        resample=processor.resample,
        factor=int(processor.patch_size) * int(processor.merge_size),
        temporal_factor=int(processor.temporal_patch_size),
    )
    if tuple(resized.shape[-2:]) != (target_h, target_w):
        raise RuntimeError(
            f"{video_policy} Qwen processor disagreed with its smart_resize result: "
            f"planned {target_w}x{target_h}, produced "
            f"{int(resized.shape[-1])}x{int(resized.shape[-2])}"
        )
    resized = resized[0].permute(0, 2, 3, 1).contiguous()
    if source_dtype.is_floating_point:
        resized = resized.to(source_dtype).div_(255)
    return resized


def _release_qwen_video_frames(frames):
    """Resize with the local release-parity processor configuration."""
    return _configured_qwen_video_frames(frames, video_policy="release")


def _qwen_video_processor(video_policy: str):
    """A configured processor, built once per distinct configuration."""
    (min_pixels, max_pixels), geometry = _qwen_video_settings(video_policy)
    frozen = tuple(sorted(
        (key, tuple(value) if isinstance(value, list) else value)
        for key, value in geometry.items()
    ))
    return _build_qwen_video_processor(min_pixels, max_pixels, frozen)


@functools.lru_cache(maxsize=4)
def _build_qwen_video_processor(min_pixels: int, max_pixels: int, frozen: tuple):
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        Qwen3VLVideoProcessor,
    )

    geometry = {key: list(value) if isinstance(value, tuple) else value
                for key, value in frozen}
    return Qwen3VLVideoProcessor(
        size={"shortest_edge": min_pixels, "longest_edge": max_pixels},
        **geometry,
    )


def _order_records(records) -> list[ImageRef | VideoRef | AudioRef]:
    """Project runtime payloads onto the pure ordering authority."""
    ordered = []
    for index, record in enumerate(records):
        name = str(index)
        if isinstance(record, RuntimeImageReference):
            ordered.append(ImageRef(name=name))
        elif isinstance(record, RuntimeVideoReference):
            ordered.append(VideoRef(name=name, has_soundtrack=record.soundtrack is not None))
        elif isinstance(record, RuntimeAudioReference):
            ordered.append(AudioRef(name=name))
        else:  # `_reference_tuple` should make this unreachable.
            raise TypeError(f"not a runtime reference record: {record!r}")
    return ordered


def _view_or_source(image, source_w: int, source_h: int, w: int, h: int):
    """One resize, or the source sliced to RGB when the size is unchanged.

    The identity guard. `comfy/utils.py::lanczos` has no short-circuit and
    round-trips through PIL uint8 unconditionally, so a no-op resize is not
    free: it costs a full resample and a float32 -> uint8 -> float32
    quantization for nothing. `[..., :3]` is NOT optional: `_resize` does it
    on every other path, so a bare `image[:1]` would hand `vae.encode` a
    4-channel RGBA reference that every pre-fold render had been sliced to RGB.
    """
    if (w, h) == (source_w, source_h):
        return image[:1, ..., :3]
    return _resize(image[:1], w, h, "disabled")


def qwen_view_size(source_w: int, source_h: int, qwen_short_edge: int) -> tuple[int, int]:
    """The encoder-only view: scale the SOURCE so its shorter side reaches N.

    From the source, not from the stage-one output, so the view is one
    resample from the pixels rather than two. Nearest 32 via the same snap the
    stage-one policies use. Shared with `bench/preflight_graph.py`.

    **Bidirectional, and deliberately so.** There is no `min(1.0, ...)` here,
    unlike the stage-one `max` policy, so N below the source shrinks the view.
    The knob was built to ENLARGE what the encoder sees without paying DiT
    reference rows; under the v2 encoder the useful direction is usually the
    other one. Reference tokens land in the text segment ahead of the prompt,
    so a large view does not merely cost tokens -- it costs the prompt its
    share of the segment, and the prompt is where the subject-to-speaker
    binding lives. Lowering N is how you keep full-resolution references for
    the DiT and still leave the prompt legible to the encoder.
    """
    scale = qwen_short_edge / min(source_w, source_h)
    return snap_to_multiple(source_w, scale), snap_to_multiple(source_h, scale)


def _reference_views(image, source_w, source_h, role_w, role_h, record,
                     image_policy):
    """Return `(vae_view, qwen_view, info)` for one still reference.

    With `qwen_short_edge == 0` one tensor is sized at stage one and, under
    `release`, pre-clamped to the stage-two bounds so both towers encode one
    size. This is what every serving implementation does.

    With `qwen_short_edge == N` the two branches part: the VAE encodes the
    stage-one tensor unclamped, and the encoder is shown a second view of the
    source at an N short edge, with the stage-two bounds pre-applied to that
    view alone under `release`. Section 1b of
    `docs/h3_conditioning_end_to_end.md` is why nothing indexes a Qwen token
    against a latent patch, so this breaks no contract; what it changes is a
    quality question the blind comparison owns.
    """
    info = {"role": (role_w, role_h), "separate": bool(record.qwen_short_edge)}
    if not record.qwen_short_edge:
        target_w, target_h = role_w, role_h
        if image_policy != "comfy":
            target_w, target_h = _configured_qwen_image_size(
                role_w, role_h, image_policy)
        shared = _view_or_source(image, source_w, source_h, target_w, target_h)
        info.update(vae=(target_w, target_h), qwen=(target_w, target_h))
        return shared, shared, info

    vae_view = _view_or_source(image, source_w, source_h, role_w, role_h)
    qwen_w, qwen_h = qwen_view_size(source_w, source_h, record.qwen_short_edge)
    if image_policy != "comfy":
        qwen_w, qwen_h = _configured_qwen_image_size(
            qwen_w, qwen_h, image_policy)
    qwen_view = _view_or_source(image, source_w, source_h, qwen_w, qwen_h)
    info.update(vae=(role_w, role_h), qwen=(qwen_w, qwen_h))
    return vae_view, qwen_view, info


def _compile_reference_records(
    records, vae, audio_vae, width, height, frame_count,
    video_policy="comfy", image_policy="comfy",
):
    """Compile one ordered record list into Qwen items and DiT blocks.

    `vae` and `audio_vae` may each be `None`, mirroring core since ComfyUI
    PR 16065 (merged 2026-09-03, commit `1aec3a13`): a reference whose VAE
    is absent is presented to Qwen exactly as before -- same item, same
    label -- and contributes no DiT block. Core's gates are reproduced
    rather than improved: a sounded video needs the video VAE before its
    soundtrack is even considered, and a video whose soundtrack has no
    audio VAE keeps its `<Audio>` label and becomes a silent `video` block.
    `bench/check_reference_runtime.py::encoder_only_references_skip_the_dit_rows`
    holds the two nodes to the same three cells.
    """
    if video_policy not in VIDEO_POLICIES:
        raise ValueError(
            f"unknown reference video policy {video_policy!r}; "
            f"expected one of {VIDEO_POLICIES}"
        )
    if image_policy not in IMAGE_POLICIES:
        raise ValueError(
            f"unknown reference image policy {image_policy!r}; "
            f"expected one of {IMAGE_POLICIES}"
        )
    ref_items = []
    ref_blocks = []
    duration = frame_count / FPS

    for index, record in enumerate(records):
        if isinstance(record, RuntimeImageReference):
            image = record.image
            _, source_h, source_w = _image_shape(image, f"reference image {index + 1}")
            # Stage one: upstream role sizing. Shared with the calibration
            # builder rather than reimplemented here; see reference_geometry.
            role_w, role_h = fit_reference_image(
                source_w, source_h,
                size_policy=record.size_policy,
                short_edge=record.short_edge,
                allow_upscale=record.allow_upscale,
                canvas_w=width, canvas_h=height,
            )
            # Stage two, and the branch split. With no Qwen view of its own
            # the selected still policy is applied BEFORE the VAE so both
            # towers encode one size; `comfy` declines to have an opinion,
            # which is what core does and what every graph built before this
            # input existed must keep getting. With a Qwen view, the VAE keeps
            # the stage-one tensor and only the encoder's view is shaped.
            vae_view, qwen_view, views = _reference_views(
                image, source_w, source_h, role_w, role_h, record,
                image_policy)
            target_w, target_h = views["vae"]
            if not views["separate"] and (target_w, target_h) != (role_w, role_h):
                logger.info(
                    "[h3] reference %d: %s still policy moved %dx%d to "
                    "%dx%d before the VAE, so the VAE and Qwen stay on one "
                    "size.", index + 1, image_policy, role_w, role_h,
                    target_w, target_h)
            qwen_w, qwen_h = views["qwen"]
            logger.info(
                "[h3] reference %d: %dx%d source -> %dx%d role (%s, "
                "short_edge=%d, allow_upscale=%s) -> VAE %dx%d, %d latent rows; "
                "Qwen view %dx%d (%s; image_policy=%s), about %d merged tokens "
                "before the encoder's own processor",
                index + 1, source_w, source_h, role_w, role_h,
                record.size_policy, record.short_edge, record.allow_upscale,
                target_w, target_h, latent_rows(target_w, target_h),
                qwen_w, qwen_h,
                (f"qwen_short_edge={record.qwen_short_edge}" if views["separate"]
                 else "same tensor as the VAE view"),
                image_policy, (qwen_w // 32) * (qwen_h // 32))
            ref_items.append({"type": "image", "data": qwen_view})
            if vae is None:
                # Encoder-only still: Qwen sees it, the DiT gets no rows.
                continue
            latent = vae.encode(vae_view)
            # Read the grid off the tensor the VAE returned, not off the pixel
            # size. `target_*` is a multiple of 32 only because both installed
            # processor configs declare patch_size 16 / merge_size 2; stage two
            # derives its factor from whichever config is loaded, and an
            # artifact declaring patch_size 14 (the Qwen2-VL value, and the
            # reason `smart_resize`'s own default factor is 28) would make
            # `target_w // 16` disagree with the latent silently, for one
            # reference. This repo's "an assumption that has only ever met one
            # implementation is not a tested assumption" case.
            latent_h, latent_w = int(latent.shape[-2]), int(latent.shape[-1])
            if (latent_h, latent_w) != (target_h // 16, target_w // 16):
                logger.warning(
                    "[h3] reference %d: the VAE returned a %dx%d latent grid "
                    "where %dx%d pixels implies %dx%d. Using the VAE's. Check "
                    "the loaded processor's patch/merge geometry.",
                    index + 1, latent_h, latent_w, target_w, target_h,
                    target_w // 16, target_h // 16)
            ref_blocks.append({
                "kind": "image",
                "latent_h": latent_h,
                "latent_w": latent_w,
                "latent": latent,
            })
            continue

        if isinstance(record, RuntimeVideoReference):
            frames = _prepare_reference_video(
                record.frames, record.loaded_fps, frame_count
            )
            _, source_h, source_w = _image_shape(frames, f"reference video {index + 1}")
            canvas_w, canvas_h = adapt_canvas(source_w, source_h)
            if (video_policy == "comfy"
                    and source_w * source_h < canvas_w * canvas_h):
                # Core never upscales a reference video.
                canvas_w = max(
                    CANVAS_MULTIPLE,
                    round(source_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
                )
                canvas_h = max(
                    CANVAS_MULTIPLE,
                    round(source_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
                )
            frames = _resize(frames, canvas_w, canvas_h, "disabled")

            soundtrack = None
            if record.soundtrack is not None:
                soundtrack = _prepare_audio(
                    record.soundtrack, duration, f"reference video {index + 1} soundtrack"
                )
                # The label is emitted whether or not an audio latent follows;
                # core does the same, and the two nodes must present one
                # prompt to the encoder.
                ref_items.append({"type": "audio"})

            sample_indices = list(range(0, int(frames.shape[0]), FPS // 2))
            qwen_frames = frames[sample_indices]
            if video_policy == "release":
                qwen_frames = _release_qwen_video_frames(qwen_frames)
            ref_items.append({
                "type": "video",
                "data": qwen_frames,
                "timestamps": [i / 2.0 for i in range(len(sample_indices))],
            })
            if vae is None:
                # Encoder-only video: the whole block goes, soundtrack
                # included, which is core's gate (the audio latent is built
                # after the video one and never without it).
                logger.info(
                    "[h3] reference video %d policy=%s: encoder only, no video "
                    "VAE wired; %d raw frame(s) reach Qwen and no DiT rows",
                    index + 1, video_policy, int(qwen_frames.shape[0]))
                continue
            latent = vae.encode(frames)
            audio_latent, ref_audio_t = None, 0
            if soundtrack is not None and audio_vae is not None:
                audio_latent, ref_audio_t = _encode_ref_audio_aligned(audio_vae,soundtrack)
            ref_blocks.append({
                "kind": "video_audio" if ref_audio_t else "video",
                "latent_t": latent.shape[2],
                "latent_h": canvas_h // 16,
                "latent_w": canvas_w // 16,
                "ref_audio_t": ref_audio_t,
                "latent": latent,
                "audio_latent": audio_latent,
            })
            logger.info(
                "[h3] reference video %d policy=%s: source %dx%d, VAE "
                "%dx%d, Qwen sampled view %dx%d x%d raw frame(s)",
                index + 1, video_policy, source_w, source_h,
                canvas_w, canvas_h, int(qwen_frames.shape[2]),
                int(qwen_frames.shape[1]), int(qwen_frames.shape[0]),
            )
            continue

        if isinstance(record, RuntimeAudioReference):
            audio = _prepare_audio(
                record.audio, duration, f"standalone reference audio {index + 1}"
            )
            ref_items.append({"type": "audio"})
            if audio_vae is None:
                # Encoder-only audio is a bare label: Qwen is never handed
                # the waveform, so nothing but "<Audio j>" reaches either
                # model. Kept for parity with core, and worth knowing before
                # wiring it as an experiment.
                continue
            audio_latent, ref_audio_t = _encode_ref_audio_aligned(audio_vae,audio)
            ref_blocks.append({
                "kind": "audio",
                "ref_audio_t": ref_audio_t,
                "audio_latent": audio_latent,
            })
            continue

        raise TypeError(f"not a runtime reference record: {record!r}")

    return ref_items, ref_blocks


class MiniMaxH3AppendRefImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AppendRefImage",
            display_name="Append MiniMax H3 Image Reference",
            category="MiniMaxH3/references",
            description=(
                "Append one still to an ordered H3 reference list. Its list "
                "position, not its socket, decides its <Picture N> label.\n\n"
                "Every still is read twice. The VIDEO MODEL encodes a copy "
                "through the video VAE into reference rows attended on every "
                "sampling step; size_policy sizes that copy. The TEXT ENCODER "
                "(Qwen3-VL) reads a copy as vision tokens placed in the text "
                "segment ahead of your prompt; qwen_view sizes that one. "
                "Wire the chain into MiniMax H3 Reference Report to see both "
                "copies and what they cost before you queue."
            ),
            inputs=[
                io.Image.Input("image"),
                H3References.Input("references", optional=True),
                # A DynamicCombo, not four flat widgets: `dit_short_edge` and
                # `allow_upscale` are read ONLY under `max`, and nesting them
                # under the branch that reads them makes the inert state
                # unreachable rather than warned about (owner, 2026-08-27).
                io.DynamicCombo.Input(
                    "size_policy",
                    options=[
                        io.DynamicCombo.Option("max", [
                            io.Int.Input(
                                # `dit_short_edge`, not `short_edge`: it sits
                                # beside `qwen_short_edge` and the pair size two
                                # DIFFERENT copies, this one the video model's.
                                "dit_short_edge", default=REF_IMAGE_SHORT_EDGE,
                                min=CANVAS_MULTIPLE, max=4096, step=32,
                                tooltip=(
                                    "Shorter side, in pixels, of the VIDEO "
                                    "MODEL's copy, rounded to 32.\n\n"
                                    "With allow_upscale on this is a target: "
                                    "every still reaches it. With allow_upscale "
                                    "off it is a ceiling: a source already "
                                    "smaller passes through untouched and this "
                                    "value does nothing for it.\n\n"
                                    "DiT reference rows grow with the square of "
                                    "this value and are attended on every "
                                    "sampling step. The default is the release "
                                    "pipeline's own constant "
                                    "(comfy_extras REF_IMAGE_SHORT_EDGE)."
                                ),
                            ),
                            io.Boolean.Input(
                                "allow_upscale", default=True,
                                tooltip=(
                                    "Enlarge a still whose shorter side is "
                                    "below dit_short_edge, as well as shrinking "
                                    "a larger one.\n\n"
                                    "On (default): every still reaches "
                                    "dit_short_edge. This is what sglang, "
                                    "diffusers and DiffSynth do.\n\n"
                                    "Off: shrink only, which is core ComfyUI's "
                                    "behaviour. Cheaper for a small source, by "
                                    "the square of the scale it skips.\n\n"
                                    "Upscaling adds rows, not detail the source "
                                    "lacked. Whether it improves identity is "
                                    "what the reference-view ablation measures "
                                    "(docs/h3_references.md)."
                                ),
                            ),
                        ]),
                        io.DynamicCombo.Option("match", []),
                    ],
                    tooltip=(
                        "How the VIDEO MODEL's copy is sized. This decides the "
                        "DiT reference rows paid on every sampling step.\n\n"
                        "max (default): scale so the shorter side is "
                        "dit_short_edge, rounded to 32. The output canvas is "
                        "ignored. This is the release pipeline's rule.\n\n"
                        "match: cap the still at the output canvas's pixel "
                        "area. Never enlarges. dit_short_edge and "
                        "allow_upscale do not apply."
                    ),
                ),
                # A DynamicCombo since 2026-08-31; it was an Int whose 0 meant
                # "no separate view", a number quietly selecting a mode.
                # `shared` first since 2026-09-13, so the default is what every
                # serving implementation does and what an API prompt that omits
                # the input gets.
                io.DynamicCombo.Input(
                    "qwen_view",
                    options=[
                        io.DynamicCombo.Option("shared", []),
                        io.DynamicCombo.Option("separate", [
                            io.Int.Input(
                                "qwen_short_edge",
                                default=REF_QWEN_SHORT_EDGE,
                                min=CANVAS_MULTIPLE, max=4096, step=32,
                                tooltip=(
                                    "Shorter side, in pixels, of the TEXT "
                                    "ENCODER's copy, rounded to 32. Scaled from "
                                    "the source in both directions: below the "
                                    "source it shrinks, above it enlarges.\n\n"
                                    "Vision tokens grow with the square of this "
                                    "value. The pre-filled value is a starting "
                                    "point, not a measured optimum "
                                    "(h3_rules.REF_QWEN_SHORT_EDGE)."
                                ),
                            ),
                        ]),
                    ],
                    tooltip=(
                        "How the TEXT ENCODER's copy is sized. Its vision "
                        "tokens sit in the text segment ahead of your prompt, "
                        "and their hidden states ride the DiT's text segment "
                        "on every step.\n\n"
                        "shared (default): the same copy the video model gets, "
                        "sized by size_policy. One prepared image feeds both, "
                        "which is what every serving implementation does.\n\n"
                        "separate: the encoder gets its own copy scaled to "
                        "qwen_short_edge from the source, while the video "
                        "model keeps the size_policy copy. Use it to change "
                        "what the encoder sees without changing DiT rows: a "
                        "smaller copy leaves the prompt a larger share of what "
                        "the encoder reads, a larger one gives the encoder "
                        "more detail. Neither direction is measured yet; the "
                        "reference-view ablation is the arm that would."
                    ),
                ),
            ],
            outputs=[H3References.Output(display_name="references")],
        )

    @classmethod
    # `qwen_view` before `references` because it is REQUIRED in the schema
    # and `references` is optional; a signature default on a required input
    # is a UI-versus-API split (2026-08-31). `None` is what an API prompt
    # omitting a DynamicCombo yields; it takes the schema's FIRST option, so
    # the schema order above is the default and there is no second copy of it.
    def execute(cls, image, size_policy, qwen_view, references=None):
        # A DynamicCombo arrives as ONE nested dict: the selected key under the
        # input's own id, and the chosen option's inputs alongside it. NOT as
        # flattened kwargs. `MiniMaxH3Resolution.execute` carries the scar from
        # getting this wrong.
        policy = (size_policy if isinstance(size_policy, str)
                  else size_policy["size_policy"])
        if policy not in SIZE_POLICIES:
            raise ValueError(f"unknown image size policy {policy!r}")
        if policy == "max" and not isinstance(size_policy, str):
            short_edge = int(size_policy["dit_short_edge"])
            allow_upscale = bool(size_policy["allow_upscale"])
        elif policy == "max":
            # The bare selection: the schema's own defaults.
            short_edge, allow_upscale = REF_IMAGE_SHORT_EDGE, True
        else:
            # `match` reads neither, and they are not reachable under it.
            short_edge, allow_upscale = REF_IMAGE_SHORT_EDGE, False
        size_policy = policy
        count, source_h, source_w = _image_shape(image, "image")
        if count > 1:
            # `_compile_reference_records` keeps only the first. Saying so is
            # the whole fix: wiring a video loader's IMAGE output here dropped
            # frames 2..N with no message at all.
            logger.warning(
                "[h3] image reference carries %d images; using the first. "
                "Append one node per reference, or use the video reference "
                "node for a clip.", count)
        # The aspect gate, at the node rather than at the compiler, so a
        # reference that cannot be conditioned fails before the VAE is touched.
        if not aspect_in_range(source_w, source_h):
            raise RuntimeError(
                f"A MiniMax H3 reference image must be within "
                f"{describe_aspect_range()}; this one is {source_w}x{source_h} "
                f"({source_w / source_h:.3g}). Crop it before referencing it.")
        if short_edge < CANVAS_MULTIPLE:
            raise ValueError(
                f"short_edge must be at least {CANVAS_MULTIPLE}, got {short_edge}")
        view = (qwen_view if isinstance(qwen_view, str)
                else qwen_view["qwen_view"])
        if view not in ("separate", "shared"):
            raise ValueError(f"unknown qwen_view {view!r}")
        if view == "shared":
            # 0 remains the INTERNAL representation of "one shared view" on
            # `RuntimeImageReference`, a derived value nobody types.
            qwen_short_edge = 0
        elif isinstance(qwen_view, str):
            qwen_short_edge = REF_QWEN_SHORT_EDGE
        else:
            qwen_short_edge = int(qwen_view["qwen_short_edge"])
            if qwen_short_edge < CANVAS_MULTIPLE:
                raise ValueError(
                    f"qwen_short_edge must be at least {CANVAS_MULTIPLE}, "
                    f"got {qwen_short_edge}")
        return io.NodeOutput(
            _reference_tuple(references)
            + (RuntimeImageReference(
                image=image, size_policy=size_policy,
                short_edge=int(short_edge), allow_upscale=bool(allow_upscale),
                qwen_short_edge=qwen_short_edge,
            ),)
        )


class MiniMaxH3AppendRefVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AppendRefVideo",
            display_name="Append MiniMax H3 Video Reference",
            category="MiniMaxH3/references",
            description=(
                "Append one video and its optional soundtrack. Frames and "
                "VHS metadata must come from the same loader; loaded_fps is "
                "owned by the record and normalized to 24 fps at compilation."
            ),
            inputs=[
                io.Image.Input("frames"),
                VHSVideoInfo.Input("video_info"),
                io.Audio.Input("soundtrack", optional=True),
                H3References.Input("references", optional=True),
            ],
            outputs=[H3References.Output(display_name="references")],
        )

    @classmethod
    def execute(cls, frames, video_info, soundtrack=None, references=None):
        frame_count, height, width = _image_shape(frames, "frames")
        loaded_fps = _loaded_fps(video_info, frame_count, height, width)
        if soundtrack is not None:
            _audio_shape(soundtrack, "soundtrack")
        return io.NodeOutput(
            _reference_tuple(references)
            + (RuntimeVideoReference(
                frames=frames, loaded_fps=loaded_fps, soundtrack=soundtrack
            ),)
        )


class MiniMaxH3AppendRefAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AppendRefAudio",
            display_name="Append MiniMax H3 Audio Reference",
            category="MiniMaxH3/references",
            description="Append one standalone audio reference in list order.",
            inputs=[
                io.Audio.Input("audio"),
                H3References.Input("references", optional=True),
            ],
            outputs=[H3References.Output(display_name="references")],
        )

    @classmethod
    def execute(cls, audio, references=None):
        _audio_shape(audio, "audio")
        return io.NodeOutput(
            _reference_tuple(references) + (RuntimeAudioReference(audio=audio),)
        )


class MiniMaxH3ReferenceConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceConditioning",
            display_name="MiniMax H3 Reference Conditioning (Ordered)",
            category="MiniMaxH3",
            description=(
                "Compile an ordered MINIMAX_H3_REFERENCES list into the Qwen "
                "presentation and DiT reference payload. Works on the H3 "
                "reference checkpoint and on fl2va; this node does not infer "
                "checkpoint task identity. "
                "The node's preview shows what each reference cost once it "
                "has run; MiniMax H3 Reference Report shows it beforehand."
            ),
            inputs=[
                io.Clip.Input("clip"),
                # Both optional since 0.99.33, mirroring core's
                # MiniMaxH3ReferenceToVideo after ComfyUI PR 16065. Absent,
                # a reference of that kind conditions the text encoder only.
                io.Vae.Input(
                    "vae", optional=True,
                    tooltip=(
                        "Video VAE. Leave it unwired and reference stills and "
                        "videos reach the text encoder only: same labels, "
                        "same Qwen view, no reference latents for the DiT."
                    ),
                ),
                io.Vae.Input(
                    "audio_vae", optional=True,
                    tooltip=(
                        "Audio VAE. Leave it unwired and reference audio is "
                        "only its <Audio j> label: the encoder never hears "
                        "audio, so without this VAE an audio reference "
                        "carries nothing. A sounded video keeps its label "
                        "and becomes a silent video block."
                    ),
                ),
                H3References.Input("references"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Combo.Input(
                    "video_policy", options=list(VIDEO_POLICIES),
                    default="comfy", optional=True,
                    tooltip=(
                        "Reference VIDEO preparation only.\n\n"
                        "comfy (default): core ComfyUI's behaviour. The video "
                        "model's copy is never enlarged, and the text encoder "
                        "reads the 2 fps samples through core's per-pair "
                        "processor.\n\n"
                        "release: the release pipeline's rule. The video "
                        "model's copy is scaled to the release canvas AND the "
                        "2 fps samples go through the release's duration-aware "
                        "processor. The two stages are one policy because the "
                        "upscale alone overshoots the encoder's long-clip "
                        "budget.\n\n"
                        "Neither changes native ComfyUI nodes."
                    ),
                ),
                io.Combo.Input(
                    "image_policy", options=list(IMAGE_POLICIES),
                    default="comfy", optional=True,
                    tooltip=(
                        "Reference STILL preparation, after each still's own "
                        "size_policy.\n\n"
                        "comfy (default): the still reaches the text encoder "
                        "exactly as core hands it, and the encoder's own "
                        "processor resizes it afterwards if its bounds bind, "
                        "after the VAE has already encoded.\n\n"
                        "release: pre-apply the release's declared pixel floor "
                        "and ceiling before the VAE, so both readers encode "
                        "one size.\n\n"
                        "On the shipped encoder the two produce the same "
                        "geometry at every legal short edge; release differs "
                        "only for a still under the release's floor. The "
                        "Reference Report shows which bounds moved a copy."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(
        cls, clip, references, prompt, width=1344,
        height=768, length=124, video_policy="comfy",
        image_policy="comfy", vae=None, audio_vae=None,
    ):
        records = _reference_tuple(references)
        if not records:
            raise ValueError(
                "MiniMaxH3ReferenceConditioning needs at least one appended reference"
            )
        if not prompt or not prompt.strip():
            raise ValueError(
                "MiniMaxH3ReferenceConditioning needs a prompt; empty prompts "
                "condition on a pad token in core and are refused here"
            )
        latent, frame_count = _empty_av_latent(width, height, length)
        ref_items, ref_blocks = _compile_reference_records(
            records, vae, audio_vae, width, height, frame_count,
            video_policy=video_policy, image_policy=image_policy,
        )
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            # Absent, not []: `model_base` builds the text-only layout for a
            # missing key and a refs=[] layout for an empty one, and core
            # sends the former, so the encoder-only arm of the two nodes
            # must reach the DiT by the same door.
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"minimax_refs": ref_blocks}
            )

        labels = assign_labels(_order_records(records))
        logger.info(
            "[h3] ordered reference conditioning: %dx%d, %d frames, %d "
            "record(s), presentation=%s, DiT reference block(s)=%d%s, "
            "video_policy=%s, image_policy=%s",
            width, height, frame_count, len(records), labels,
            len(ref_blocks),
            "" if ref_blocks else " (encoder only: no VAE wired)",
            video_policy, image_policy,
        )
        # The same pricing the report node shows beforehand, on the node
        # itself once it has run. Geometry only; a failure here must never
        # fail the render, so it is logged and the preview is skipped.
        preview = None
        try:
            from .reference_report import format_report, price_references
            preview = format_report(price_references(
                records, width, height, length, image_policy=image_policy,
                video_policy=video_policy, clip=clip, prompt=prompt))
        except Exception as exc:  # pragma: no cover - reporting must not gate
            logger.warning("[h3] reference report skipped: %s", exc)
        return io.NodeOutput(conditioning, latent,
                             ui=ui.PreviewText(preview) if preview else None)
