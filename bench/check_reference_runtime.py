#!/usr/bin/env python3
"""CPU acceptance checks for the typed MiniMax H3 reference runtime.

No server, weights, or CUDA are used.  ComfyUI is imported in CPU mode so the
real custom-type/schema API and the installed H3 preparation helpers remain
the authority; video and audio VAEs are small recording stubs.

This check covers the boundary the pure graph resolver cannot see: append
nodes must be copy-on-add, VHS metadata must describe the wired frames, video
must be normalized from the owned loaded rate to 24 fps, mono must become
stereo, audio must stop at the aligned target duration, the opt-in release
video policy must keep its VAE and duration-aware Qwen views distinct, and one
ordered walk must produce Qwen items and DiT blocks with the sounded-video 2:1
shape.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_COMFY = Path.home() / "ComfyUI"
sys.path.insert(0, str(_COMFY))
sys.path.insert(0, str(_REPO.parent))

import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True

R = importlib.import_module(f"{_REPO.name}.reference_conditioning")


def _audio(seconds=2.0, channels=1, sample_rate=32000):
    import torch
    return {
        "waveform": torch.zeros(1, channels, round(seconds * sample_rate)),
        "sample_rate": sample_rate,
    }


class _LazyAudio(Mapping):
    """VHS-shaped AUDIO: a Mapping that realizes on first value access."""

    def __init__(self, value):
        self.value = value
        self.realized = False

    def __getitem__(self, key):
        self.realized = True
        return self.value[key]

    def __iter__(self):
        self.realized = True
        return iter(self.value)

    def __len__(self):
        self.realized = True
        return len(self.value)


def _frames(count=30, height=64, width=96):
    import torch
    # The source index is recoverable after resampling.
    return torch.arange(count).view(count, 1, 1, 1).expand(
        count, height, width, 3
    ).float()


def _video_info(frames, loaded_fps=30.0):
    return {
        "loaded_fps": loaded_fps,
        "loaded_frame_count": int(frames.shape[0]),
        "loaded_height": int(frames.shape[1]),
        "loaded_width": int(frames.shape[2]),
    }


class _VideoVae:
    def __init__(self):
        self.inputs = []

    def encode(self, frames):
        import torch
        self.inputs.append(frames)
        return torch.zeros(
            1, 24, max(1, int(frames.shape[0])),
            int(frames.shape[1]) // 16, int(frames.shape[2]) // 16,
        )


class _AudioVae:
    audio_sample_rate = 32000

    def __init__(self):
        self.inputs = []

    def encode(self, waveform):
        import torch
        self.inputs.append(waveform)
        return torch.zeros(1, 32, 2, max(1, int(waveform.shape[1]) // 800))


class _Clip:
    def __init__(self):
        self.ref_items = None

    def tokenize(self, _prompt, **kwargs):
        self.ref_items = kwargs.get("minimax_ref_items")
        return {"stub": []}

    def encode_from_tokens_scheduled(self, _tokens):
        import torch
        return [[torch.zeros(1, 1, 1), {}]]


def append_is_copy_on_add_and_ordered():
    """An append returns a new tuple and never rewrites its input plan."""
    audio_out = R.MiniMaxH3AppendRefAudio.execute(_audio()).args[0]
    before = tuple(audio_out)
    image_out = R.MiniMaxH3AppendRefImage.execute(
        _frames(1), references=audio_out
    ).args[0]
    assert audio_out == before and len(audio_out) == 1, audio_out
    assert len(image_out) == 2 and image_out[:1] == audio_out, image_out
    labels = R.assign_labels(R._order_records(image_out))
    assert labels == ["<Audio 1>", "<Picture 1>"], labels


def video_metadata_is_owned():
    """The runtime value refuses metadata that describes a different decode."""
    frames = _frames()
    out = R.MiniMaxH3AppendRefVideo.execute(
        frames, _video_info(frames, 25.0)
    ).args[0]
    assert math.isclose(out[0].loaded_fps, 25.0), out

    bad = _video_info(frames, 25.0)
    bad["loaded_width"] += 32
    try:
        R.MiniMaxH3AppendRefVideo.execute(frames, bad)
    except ValueError as exc:
        assert "one decode" in str(exc), exc
    else:
        raise AssertionError("frames accepted metadata for a different width")


def video_is_normalized_to_24fps():
    """30 samples at 30 fps become the 24 target timestamps in their span."""
    frames = _frames(30)
    got = R._resample_video_to_24fps(frames, 30.0)
    assert got.shape[0] == 24, got.shape
    assert got[:, 0, 0, 0].tolist() == [
        0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 9.0,
        10.0, 11.0, 12.0, 14.0, 15.0, 16.0, 18.0, 19.0,
        20.0, 21.0, 22.0, 24.0, 25.0, 26.0, 28.0, 29.0,
    ], got[:, 0, 0, 0]
    same = R._resample_video_to_24fps(frames, 24.0)
    assert same is frames, "the already-normalized path allocated a second clip"


def audio_is_stereo_and_target_bounded():
    """Mono is doubled and the aligned duration is the sole trim clock."""
    audio = _audio(seconds=2.0, channels=1)
    got = R._prepare_audio(audio, 22 / 24, "test audio")
    assert tuple(got["waveform"].shape) == (1, 2, round(22 / 24 * 32000)), (
        got["waveform"].shape
    )
    assert audio["waveform"].shape[1] == 1, "the source branch was mutated"
    try:
        R._prepare_audio(_audio(channels=3), 1.0, "three-channel audio")
    except ValueError as exc:
        assert "mono or stereo" in str(exc), exc
    else:
        raise AssertionError("three-channel audio was silently reduced")


def vhs_lazy_audio_mapping_is_accepted():
    """The AUDIO socket accepts VHS LazyAudioMap, not only core's dict."""
    frames = _frames()
    lazy = _LazyAudio(_audio(seconds=2.0, channels=2))
    records = R.MiniMaxH3AppendRefVideo.execute(
        frames, _video_info(frames, 24.0), soundtrack=lazy
    ).args[0]
    assert records[0].soundtrack is lazy
    assert lazy.realized, "the mapping was accepted without validating its payload"
    got = R._prepare_audio(lazy, 1.0, "VHS soundtrack")
    assert tuple(got["waveform"].shape) == (1, 2, 32000)


def compiler_preserves_one_order_for_both_lists():
    """Arbitrary order survives; a sounded video is two items, one block."""
    frames = _frames()
    records = (
        R.RuntimeAudioReference(_audio()),
        R.RuntimeVideoReference(frames, 30.0, _audio()),
        R.RuntimeImageReference(_frames(1, 64, 64), "match"),
    )
    audio_vae = _AudioVae()
    items, blocks = R._compile_reference_records(
        records, _VideoVae(), audio_vae, width=64, height=64, frame_count=22
    )
    assert [item["type"] for item in items] == [
        "audio", "audio", "video", "image"
    ], [item["type"] for item in items]
    assert [block["kind"] for block in blocks] == [
        "audio", "video_audio", "image"
    ], [block["kind"] for block in blocks]
    assert R.assign_labels(R._order_records(records)) == [
        "<Audio 1>", "<Audio 2>", "<Video 1>", "<Picture 1>"
    ]
    assert len(audio_vae.inputs) == 2
    for encoded in audio_vae.inputs:
        # _encode_ref_audio presents [batch, samples, channels] to the VAE.
        assert tuple(encoded.shape) == (1, round(22 / 24 * 32000), 2), encoded.shape


def release_video_policy_is_opt_in_and_two_stage():
    """Release mode atomically upscales VAE and processes raw Qwen samples."""
    # The real release processor must see the raw odd count. Repeat-padding 31
    # to 32 preserves 16 temporal blocks but changes the spatial result.
    assert R._release_qwen_video_size(31, 1344, 768) == (1184, 672)
    assert R._release_qwen_video_size(32, 1344, 768) == (1152, 640)

    frames = _frames(22, 32, 32)
    records = (R.RuntimeVideoReference(frames, 24.0, None),)
    original_adapt_canvas = R.adapt_canvas
    try:
        # First isolate the VAE stage. Native-compatible mode keeps a small
        # source small; release mode accepts the canvas upscale.
        R.adapt_canvas = lambda _w, _h: (64, 64)
        comfy_vae = _VideoVae()
        comfy_items, _ = R._compile_reference_records(
            records, comfy_vae, _AudioVae(), 64, 64, 22,
            video_policy="comfy",
        )
        release_vae = _VideoVae()
        release_items, _ = R._compile_reference_records(
            records, release_vae, _AudioVae(), 64, 64, 22,
            video_policy="release",
        )
        assert tuple(comfy_vae.inputs[0].shape[1:3]) == (32, 32)
        assert tuple(release_vae.inputs[0].shape[1:3]) == (64, 64)
        assert tuple(comfy_items[0]["data"].shape[1:3]) == (32, 32)
        assert tuple(release_items[0]["data"].shape[1:3]) == (64, 64)

        # The encoder policy keeps the native-compatible VAE size while using
        # the source-config Qwen stage. It is the shipped custom-loader default.
        encoder_vae = _VideoVae()
        encoder_items, _ = R._compile_reference_records(
            records, encoder_vae, _AudioVae(), 64, 64, 22,
            video_policy="encoder",
        )
        assert tuple(encoder_vae.inputs[0].shape[1:3]) == (32, 32)
        assert tuple(encoder_items[0]["data"].shape[1:3]) == (64, 64)

        # Then isolate the Qwen stage. With an identity VAE canvas, the
        # release processor's clip floor moves only the sampled Qwen view.
        R.adapt_canvas = lambda w, h: (w, h)
        split_vae = _VideoVae()
        split_items, _ = R._compile_reference_records(
            records, split_vae, _AudioVae(), 32, 32, 22,
            video_policy="release",
        )
        assert tuple(split_vae.inputs[0].shape[1:3]) == (32, 32)
        assert tuple(split_items[0]["data"].shape[1:3]) == (64, 64)
    finally:
        R.adapt_canvas = original_adapt_canvas

    try:
        R._compile_reference_records(
            records, _VideoVae(), _AudioVae(), 32, 32, 22,
            video_policy="upscale_only",
        )
    except ValueError as exc:
        assert "unknown reference video policy" in str(exc), exc
    else:
        raise AssertionError("an unowned partial video policy was accepted")


def release_policy_floor_is_two_sampled_frames():
    """The shortest clip release mode can take is 22 prepared frames, not 5.

    `_prepare_reference_video` snaps to 17n+5 and the Qwen sampler steps by 12,
    so the legal prepared counts are 5, 22, 39, ... A 5-frame reference yields
    exactly ONE 2 fps sample, and the release's `smart_resize` requires a full
    temporal patch. Before 2026-08-23 that surfaced as
    `ValueError: t:1 must be larger than temporal_factor:2` raised inside
    transformers, naming neither the reference nor the policy.

    Both ends are asserted on purpose. The floor alone would be satisfied by a
    policy that refused everything, so 22 must pass in the same breath that 5
    fails -- and 5 must still be accepted by comfy mode, because this is error
    handling for a release requirement, not a new minimum for every reference.
    """
    boundary = int(R.video_patch_geometry()["temporal_patch_size"])

    # RED at the floor: one sampled frame, refused by us with our own message.
    try:
        R._release_qwen_video_frames(_frames(1, 544, 960))
    except ValueError as exc:
        assert "release video policy needs at least" in str(exc), exc
        assert "22 prepared frames" in str(exc), (
            "the message must name the real minimum a caller can act on")
    else:
        raise AssertionError("one sampled frame was accepted by release sizing")

    # GREEN one frame later: the boundary is where it is claimed to be.
    assert R._release_qwen_video_frames(_frames(boundary, 544, 960)) is not None

    # And through the compiler, which is what a graph actually reaches.
    short = (R.RuntimeVideoReference(_frames(5, 32, 32), 24.0, None),)
    try:
        R._compile_reference_records(
            short, _VideoVae(), _AudioVae(), 32, 32, 5,
            video_policy="release",
        )
    except ValueError as exc:
        assert "release video policy needs at least" in str(exc), exc
    else:
        raise AssertionError("a 5-frame reference passed the release compiler")

    # The SAME reference under comfy mode still works. Without this, tightening
    # `_prepare_reference_video` for everyone would satisfy the case above.
    items, _ = R._compile_reference_records(
        short, _VideoVae(), _AudioVae(), 32, 32, 5,
        video_policy="comfy",
    )
    assert items, "comfy mode must still accept a 5-frame reference"

    # 22 is the first legal length release mode accepts end to end.
    ok = (R.RuntimeVideoReference(_frames(22, 32, 32), 24.0, None),)
    items, _ = R._compile_reference_records(
        ok, _VideoVae(), _AudioVae(), 32, 32, 22,
        video_policy="release",
    )
    assert items, "22 prepared frames must pass release mode"


def encoder_policy_reads_encoder_config():
    """Encoder policy is governed by the encoder snapshot, not release data.

    The two checked-in processors happen to agree today. Make their temporal
    factors disagree in memory so this case proves which authority is read;
    comparing their current values would go green even if the wrong one won.
    """
    original_source_geometry = R.source_video_patch_geometry
    original_release_geometry = R.video_patch_geometry
    try:
        source = dict(original_source_geometry())
        release = dict(original_release_geometry())
        source["temporal_patch_size"] = 4
        release["temporal_patch_size"] = 2
        R.source_video_patch_geometry = lambda: source
        R.video_patch_geometry = lambda: release

        try:
            R._encoder_qwen_video_size(2, 960, 544)
        except ValueError as exc:
            assert "encoder video policy needs at least 4" in str(exc), exc
        else:
            raise AssertionError("encoder policy ignored its source temporal factor")

        assert R._release_qwen_video_size(2, 960, 544), (
            "release policy incorrectly inherited the encoder's test geometry")
    finally:
        R.source_video_patch_geometry = original_source_geometry
        R.video_patch_geometry = original_release_geometry


def conditioning_node_assembles_the_real_payload_shape():
    """The registered node attaches minimax_refs and an H3 nested latent."""
    records = (
        R.RuntimeImageReference(_frames(1, 64, 64), "match"),
        R.RuntimeAudioReference(_audio(seconds=1.0, channels=1)),
    )
    clip = _Clip()
    output = R.MiniMaxH3ReferenceConditioning.execute(
        clip=clip, vae=_VideoVae(), audio_vae=_AudioVae(),
        references=records, prompt="use <Picture 1> and <Audio 1>",
        width=64, height=64, length=22, vendor_tokens=False,
    )
    conditioning, latent = output.args
    assert [item["type"] for item in clip.ref_items] == ["image", "audio"]
    assert [block["kind"] for block in conditioning[0][1]["minimax_refs"]] == [
        "image", "audio"
    ]
    samples = latent["samples"]
    assert samples.is_nested and len(samples.tensors) == 2

    for bad_refs, bad_prompt in (((), "prompt"), (records, "   ")):
        try:
            R.MiniMaxH3ReferenceConditioning.execute(
                clip=clip, vae=_VideoVae(), audio_vae=_AudioVae(),
                references=bad_refs, prompt=bad_prompt,
                width=64, height=64, length=22, vendor_tokens=False,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("empty references or prompt reached compilation")


CHECKS = (
    append_is_copy_on_add_and_ordered,
    video_metadata_is_owned,
    video_is_normalized_to_24fps,
    audio_is_stereo_and_target_bounded,
    vhs_lazy_audio_mapping_is_accepted,
    compiler_preserves_one_order_for_both_lists,
    release_video_policy_is_opt_in_and_two_stage,
    release_policy_floor_is_two_sampled_frames,
    encoder_policy_reads_encoder_config,
    conditioning_node_assembles_the_real_payload_shape,
)


def all_hold():
    for check in CHECKS:
        check()
    return True


def main():
    failures = []
    print("typed MiniMax H3 reference runtime (CPU only)\n")
    for check in CHECKS:
        try:
            check()
        except Exception as exc:
            failures.append(check.__name__)
            print(f"  FAIL  {check.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {check.__name__}")
    if failures:
        print(f"\n{len(failures)} failure(s): {', '.join(failures)}")
        return 1
    print(f"\nall {len(CHECKS)} runtime contracts hold; no CUDA/server/model used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
