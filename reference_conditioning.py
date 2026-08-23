"""Typed, ordered reference conditioning for MiniMax H3.

The core node accepts four parallel socket groups and reconstructs ordering
and video/audio pairing from their names.  These nodes instead carry one
copy-on-append tuple.  A video record owns its metadata and optional
soundtrack, and the compiler walks the tuple exactly once to build both the
Qwen presentation and the DiT payload.

Still images retain their per-record ``match``/``max`` policy. Reference video
preparation is selected once at the compiler boundary: ``comfy`` keeps core's
no-upscale/shared-frame behaviour, ``encoder`` keeps that VAE behaviour while
using the selected encoder artifact's snapshotted Qwen processor settings, and
``release`` both puts the VAE view on the release canvas and uses the release
processor settings. The two release stages are one policy because enabling the
upscale alone overshoots Qwen's long-clip budget.

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
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import (
    CANVAS_MULTIPLE,
    FPS,
    REF_IMAGE_SHORT_EDGE,
    _empty_av_latent,
    _encode_ref_audio,
    _resize,
    adapt_canvas,
)

from .reference_order import AudioRef, ImageRef, VideoRef, assign_labels
from .h3_awq_encoder import (
    source_video_patch_geometry,
    source_video_pixel_bounds,
)
from .vendor_config import video_patch_geometry, video_pixel_bounds
from .vendor_tokens import clip_with_vendor_tokens

logger = logging.getLogger(__name__)

H3References = io.Custom("MINIMAX_H3_REFERENCES")
VHSVideoInfo = io.Custom("VHS_VIDEOINFO")
VIDEO_POLICIES = ("comfy", "release", "encoder")


@dataclass(frozen=True)
class RuntimeImageReference:
    image: Any
    size_policy: str


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


def _qwen_video_settings(video_policy: str) -> tuple[tuple[int, int], dict]:
    """Return the settings owned by the selected Qwen preprocessing policy."""
    if video_policy == "release":
        return video_pixel_bounds(), video_patch_geometry()
    if video_policy == "encoder":
        return source_video_pixel_bounds(), source_video_patch_geometry()
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


def _encoder_qwen_video_size(
    sampled_count: int, width: int, height: int
) -> tuple[int, int]:
    """Size with the selected encoder artifact's snapshotted settings."""
    return _configured_qwen_video_size(
        sampled_count, width, height, video_policy="encoder"
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


def _encoder_qwen_video_frames(frames):
    """Resize with the selected encoder artifact's processor configuration."""
    return _configured_qwen_video_frames(frames, video_policy="encoder")


@functools.lru_cache(maxsize=2)
def _qwen_video_processor(video_policy: str):
    """Construct a configured processor once, and only when its policy runs."""
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        Qwen3VLVideoProcessor,
    )

    (min_pixels, max_pixels), geometry = _qwen_video_settings(video_policy)
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


def _compile_reference_records(
    records, vae, audio_vae, width, height, frame_count,
    video_policy="encoder",
):
    """Compile one ordered record list into Qwen items and DiT blocks."""
    if video_policy not in VIDEO_POLICIES:
        raise ValueError(
            f"unknown reference video policy {video_policy!r}; "
            f"expected one of {VIDEO_POLICIES}"
        )
    ref_items = []
    ref_blocks = []
    duration = frame_count / FPS

    for index, record in enumerate(records):
        if isinstance(record, RuntimeImageReference):
            image = record.image
            _, source_h, source_w = _image_shape(image, f"reference image {index + 1}")
            if record.size_policy == "match":
                scale = min(1.0, math.sqrt((width * height) / (source_w * source_h)))
            elif record.size_policy == "max":
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(source_w, source_h))
            else:
                raise ValueError(f"unknown image size policy {record.size_policy!r}")
            target_w = max(
                CANVAS_MULTIPLE,
                round(source_w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
            )
            target_h = max(
                CANVAS_MULTIPLE,
                round(source_h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
            )
            resized = _resize(image[:1], target_w, target_h, "disabled")
            latent = vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({
                "kind": "image",
                "latent_h": target_h // 16,
                "latent_w": target_w // 16,
                "latent": latent,
            })
            continue

        if isinstance(record, RuntimeVideoReference):
            frames = _prepare_reference_video(
                record.frames, record.loaded_fps, frame_count
            )
            _, source_h, source_w = _image_shape(frames, f"reference video {index + 1}")
            canvas_w, canvas_h = adapt_canvas(source_w, source_h)
            if (video_policy in ("comfy", "encoder")
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
            latent = vae.encode(frames)

            audio_latent, ref_audio_t = None, 0
            if record.soundtrack is not None:
                soundtrack = _prepare_audio(
                    record.soundtrack, duration, f"reference video {index + 1} soundtrack"
                )
                audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, soundtrack)
                ref_items.append({"type": "audio"})

            sample_indices = list(range(0, int(frames.shape[0]), FPS // 2))
            qwen_frames = frames[sample_indices]
            if video_policy == "release":
                qwen_frames = _release_qwen_video_frames(qwen_frames)
            elif video_policy == "encoder":
                qwen_frames = _encoder_qwen_video_frames(qwen_frames)
            ref_items.append({
                "type": "video",
                "data": qwen_frames,
                "timestamps": [i / 2.0 for i in range(len(sample_indices))],
            })
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
            audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
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
                "Append one image to an ordered H3 reference list. Its list "
                "position, not its socket group, determines presentation order."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input(
                    "size_policy", options=["match", "max"], default="match",
                    tooltip=(
                        "Core-compatible sizing for this image. match caps it "
                        "at target pixel area; max caps the short edge at 2048. "
                        "Neither policy upscales."
                    ),
                ),
                H3References.Input("references", optional=True),
            ],
            outputs=[H3References.Output(display_name="references")],
        )

    @classmethod
    def execute(cls, image, size_policy="match", references=None):
        _image_shape(image, "image")
        if size_policy not in ("match", "max"):
            raise ValueError(f"unknown image size policy {size_policy!r}")
        return io.NodeOutput(
            _reference_tuple(references)
            + (RuntimeImageReference(image=image, size_policy=size_policy),)
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
                "presentation and DiT reference payload. Use an H3 reference "
                "checkpoint; this node does not infer checkpoint task identity."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                H3References.Input("references"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Boolean.Input(
                    "vendor_tokens", default=True,
                    tooltip=(
                        "Compatibility for ComfyUI versions before the seven "
                        "H3 tokens landed upstream. No-ops when native tokens "
                        "are already present."
                    ),
                ),
                # APPENDED: saved UI graphs map widgets positionally. Keep new
                # controls after every established widget.
                io.Combo.Input(
                    "video_policy", options=list(VIDEO_POLICIES),
                    default="encoder", optional=True,
                    tooltip=(
                        "Reference VIDEO preparation only. encoder (default) "
                        "keeps ComfyUI's cheaper no-upscale VAE view but runs "
                        "the raw 2 fps Qwen samples through the encoder's "
                        "duration-aware config. comfy is native ComfyUI "
                        "preprocessing. release is a full local parity policy: "
                        "it upscales the VAE view to the release canvas AND "
                        "runs the raw 2 fps Qwen samples through the release's "
                        "duration-aware processor. The two stages are atomic "
                        "because upscale alone overshoots long-clip Qwen rows. "
                        "This does not modify or fix native ComfyUI."
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
        cls, clip, vae, audio_vae, references, prompt, width=1344,
        height=768, length=124, vendor_tokens=True, video_policy="encoder",
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
        if vendor_tokens:
            clip = clip_with_vendor_tokens(clip, strict=True)

        latent, frame_count = _empty_av_latent(width, height, length)
        ref_items, ref_blocks = _compile_reference_records(
            records, vae, audio_vae, width, height, frame_count,
            video_policy=video_policy,
        )
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning = node_helpers.conditioning_set_values(
            conditioning, {"minimax_refs": ref_blocks}
        )

        labels = assign_labels(_order_records(records))
        logger.info(
            "[h3] ordered reference conditioning: %dx%d, %d frames, %d "
            "record(s), presentation=%s, video_policy=%s%s",
            width, height, frame_count, len(records), labels,
            video_policy,
            "" if vendor_tokens else ", vendor tokens OFF",
        )
        return io.NodeOutput(conditioning, latent)
