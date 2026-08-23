#!/usr/bin/env python3
"""CPU acceptance checks for the typed MiniMax H3 reference runtime.

No server, weights, or CUDA are used.  ComfyUI is imported in CPU mode so the
real custom-type/schema API and the installed H3 preparation helpers remain
the authority; video and audio VAEs are small recording stubs.

This check covers the boundary the pure graph resolver cannot see: append
nodes must be copy-on-add, VHS metadata must describe the wired frames, video
must be normalized from the owned loaded rate to 24 fps, mono must become
stereo, audio must stop at the aligned target duration, and one ordered walk
must produce Qwen items and DiT blocks with the sounded-video 2:1 shape.
"""

from __future__ import annotations

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
    def encode(self, frames):
        import torch
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
    compiler_preserves_one_order_for_both_lists,
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
