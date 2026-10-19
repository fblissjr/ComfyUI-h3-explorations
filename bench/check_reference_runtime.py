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

Two policies exist at each stage, `comfy` (the default: what core does, and
nothing pre-applied) and `release` (the release's own processor declaration,
read through `vendor_config`). The cases here hold them apart on one input,
which is the only way a selector that quietly collapses to one branch goes
red. A third policy, `encoder`, bound to a contract the AWQ adapter stamped on
its CLIP; that lane closed on 2026-09-13 (`docs/state/roadmap.md` "Closed lanes")
and its arms left with it. The guarded loader `h3_encoder_loader.py` still
records core's bounds, and the static preflight prices against them.
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
G = importlib.import_module(f"{_REPO.name}.reference_geometry")
L = importlib.import_module(f"{_REPO.name}.h3_encoder_loader")


def _preflight():
    """The static reader, loaded from its path the way its own callers do."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "preflight_graph", _REPO / "bench" / "preflight_graph.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    def spacial_compression_encode(self):
        # The real H3 audio VAE's `downscale_ratio` (`comfy/sd.py`, its audio
        # branch), which is what the generic crop and our end-pad both read.
        return 800

    def encode(self, waveform):
        import torch
        self.inputs.append(waveform)
        return torch.zeros(1, 32, 2, max(1, int(waveform.shape[1]) // 800))


class _Clip:
    def __init__(self):
        self.ref_items = None

    def tokenize(self, _prompt, **kwargs):
        # The node tokenizes twice since the report preview landed on it
        # (`reference_report.py::_count_text_tokens` counts the bare prompt), so
        # only the call that carries the reference items records them; the
        # counting call must not erase what the conditioning call presented.
        if "minimax_ref_items" in kwargs:
            self.ref_items = kwargs["minimax_ref_items"]
        return {"stub": []}

    def encode_from_tokens_scheduled(self, _tokens):
        import torch
        return [[torch.zeros(1, 1, 1), {}]]


def _max_policy(short_edge=None, allow_upscale=False):
    """The nested shape a DynamicCombo actually delivers to `execute`.

    NOT a flattened kwarg. `MiniMaxH3Resolution.execute` carries the scar from
    a test that invented its own caller: it passed flattened kwargs, the node
    read the nested form, every selection fell through to one branch, and the
    test agreed with the bug. So these call sites build the dict the executor
    sends rather than the arguments that happen to be convenient.
    """
    return {"size_policy": "max",
            "dit_short_edge": (R.REF_IMAGE_SHORT_EDGE if short_edge is None
                           else short_edge),
            "allow_upscale": allow_upscale}


def append_is_copy_on_add_and_ordered():
    """An append returns a new tuple and never rewrites its input plan."""
    audio_out = R.MiniMaxH3AppendRefAudio.execute(_audio()).args[0]
    before = tuple(audio_out)
    image_out = R.MiniMaxH3AppendRefImage.execute(
        _frames(1), {"size_policy": "match"}, "shared", references=audio_out
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


def ref_audio_end_padded_to_the_hop():
    """The waveform the audio VAE sees ends on a whole hop, padded, not cropped.

    Gap 16: core's generic crop trims a non-aligned waveform from BOTH ends,
    dropping leading samples. The release right-pads to a whole hop instead
    (`reference_conditioning.py::_encode_ref_audio_aligned` cites both), so
    what reaches `encode` must be the prepared waveform, unchanged at its
    start, followed by zeros up to the next multiple of the VAE's ratio.
    Before 2026-09-10 `encode` received the bare trim, and this would fail on
    the length.
    """
    import torch
    audio = _audio(seconds=2.0, channels=2)
    prepared = R._prepare_audio(audio, 22 / 24, "test audio")
    n = int(prepared["waveform"].shape[-1])
    assert n % 800, f"the case must start unaligned to test anything; {n} samples"
    vae = _AudioVae()
    _, t = R._encode_ref_audio_aligned(vae, prepared)
    seen = vae.inputs[0]                       # [1, samples, channels]
    want = -(-n // 800) * 800
    assert seen.shape[1] == want, (seen.shape, want)
    assert t == want // 800, (t, want)
    assert torch.equal(seen[0, :n, :], prepared["waveform"][0].movedim(0, -1)), (
        "the leading samples moved: the pad must be at the end only")
    assert not seen[0, n:, :].any(), "the pad is not silence"


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
        # The trim, then the end-pad to the audio VAE's hop that
        # `_encode_ref_audio_aligned` adds (2026-09-10; it was the bare trim
        # before, which is what core's crop then cut from both ends).
        trimmed = round(22 / 24 * 32000)
        assert tuple(encoded.shape) == (1, -(-trimmed // 800) * 800, 2), encoded.shape


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
        width=64, height=64, length=22,
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
                width=64, height=64, length=22,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("empty references or prompt reached compilation")


def encoder_only_references_skip_the_dit_rows():
    """Without a VAE a reference reaches the text encoder and nothing else.

    Mirrors core since ComfyUI PR 16065 (merged 2026-09-03, commit
    `1aec3a13`): `vae` and `audio_vae` are optional, a reference whose VAE is
    absent is presented to Qwen exactly as before -- same item, same label --
    and contributes no DiT rows. Three cells, each read off the two lists the
    compiler returns and the payload the node attaches:

      neither VAE       every item present, no block, `minimax_refs` absent
      video VAE only    still and video blocks; a sounded video becomes a
                        silent `video` block and its `<Audio>` label stays
                        (core's own quirk, kept so the two nodes agree)
      audio VAE only    the standalone audio block alone; the sounded video
                        loses its whole block, since core gates the audio
                        latent behind the video VAE
    """
    frames = _frames()
    records = (
        R.RuntimeImageReference(_frames(1, 64, 64), "match"),
        R.RuntimeVideoReference(frames, 30.0, _audio()),
        R.RuntimeAudioReference(_audio()),
    )
    items_with_both, _ = R._compile_reference_records(
        records, _VideoVae(), _AudioVae(), width=64, height=64, frame_count=22
    )
    kinds_with_both = [item["type"] for item in items_with_both]
    assert kinds_with_both == ["image", "audio", "video", "audio"], kinds_with_both

    for vae, audio_vae, expect in (
        (None, None, []),
        (_VideoVae(), None, ["image", "video"]),
        (None, _AudioVae(), ["audio"]),
    ):
        items, blocks = R._compile_reference_records(
            records, vae, audio_vae, width=64, height=64, frame_count=22
        )
        assert [item["type"] for item in items] == kinds_with_both, (
            f"vae={vae is not None} audio_vae={audio_vae is not None}: the "
            f"encoder presentation moved: {[item['type'] for item in items]}")
        assert [block["kind"] for block in blocks] == expect, (
            f"vae={vae is not None} audio_vae={audio_vae is not None}: "
            f"{[block['kind'] for block in blocks]} != {expect}")
        if vae is not None:
            silent = blocks[1]
            assert silent["ref_audio_t"] == 0 and silent["audio_latent"] is None

    clip = _Clip()
    output = R.MiniMaxH3ReferenceConditioning.execute(
        clip=clip, references=records, prompt="<Picture 1> <Audio 1> <Video 1> <Audio 2>",
        width=64, height=64, length=22,
    )
    conditioning, _latent = output.args
    assert [item["type"] for item in clip.ref_items] == kinds_with_both
    assert "minimax_refs" not in conditioning[0][1], (
        "an empty reference payload must be absent, not [], so the DiT builds "
        "the same layout core builds for a text-only pass")


def a_still_can_turn_its_own_vae_copy_off():
    """`use_vae` on the append node is `vae` unwired, for one reference.

    Two stills, the second with `use_vae` off: both reach the text encoder,
    in order and unchanged, and only the first gets a DiT block. RED CONTROL:
    with it on, both do. The node's default and the record's default are on,
    the input is the last one, and the report prices no rows for the still
    that turned it off.
    """
    import torch
    first, second = _frames(1, 64, 64), _frames(1, 64, 64)
    chain = R.MiniMaxH3AppendRefImage.execute(first, "match", "shared").args[0]
    off = R.MiniMaxH3AppendRefImage.execute(second, "match", "shared", references=chain, use_vae=False)
    on = R.MiniMaxH3AppendRefImage.execute(second, "match", "shared", references=chain)
    assert [r.use_vae for r in off.args[0]] == [True, False]
    assert [r.use_vae for r in on.args[0]] == [True, True], "the default must be on"
    schema = R.MiniMaxH3AppendRefImage.define_schema()
    last = schema.inputs[-1]
    assert last.id == "use_vae" and last.optional and last.default is True, last.id

    items_off, blocks_off = R._compile_reference_records(
        off.args[0], _VideoVae(), None, width=64, height=64, frame_count=22)
    items_on, blocks_on = R._compile_reference_records(
        on.args[0], _VideoVae(), None, width=64, height=64, frame_count=22)
    assert [i["type"] for i in items_off] == [i["type"] for i in items_on] == ["image", "image"]
    assert all(torch.equal(a["data"], b["data"]) for a, b in zip(items_off, items_on)), (
        "turning the VAE copy off must not change what the text encoder is shown")
    assert len(blocks_on) == 2, "red control: with use_vae on, both stills get a block"
    assert len(blocks_off) == 1, f"the still with use_vae off still got a block: {len(blocks_off)}"

    reference_report = importlib.import_module(f"{_REPO.name}.reference_report")
    priced = reference_report.price_references(off.args[0], 64, 64, 22)
    assert [item.dit_rows > 0 for item in priced.items] == [True, False], (
        [item.dit_rows for item in priced.items])


def a_video_can_turn_its_own_vae_copy_off():
    """`use_vae` on the video append node is `vae` unwired, for one video.

    A still and a sounded video, the video with `use_vae` off: both reach the
    text encoder in order and unchanged (the soundtrack keeps its label), and
    only the still gets a DiT block; the video's soundtrack gets no audio
    block either, since core builds the audio latent only behind the video
    one. RED CONTROL: with it on, the video gets its `video_audio` block. The
    input is last and optional, the default is on, and the report prices no
    rows for the video that turned it off.
    """
    import torch
    frames = _frames()
    still = R.MiniMaxH3AppendRefImage.execute(_frames(1, 64, 64), "match", "shared").args[0]
    info = _video_info(frames, loaded_fps=30.0)
    off = R.MiniMaxH3AppendRefVideo.execute(frames, info, soundtrack=_audio(), references=still, use_vae=False)
    on = R.MiniMaxH3AppendRefVideo.execute(frames, info, soundtrack=_audio(), references=still)
    assert [getattr(r, "use_vae", None) for r in off.args[0]] == [True, False]
    assert [getattr(r, "use_vae", None) for r in on.args[0]] == [True, True], "the default must be on"
    schema = R.MiniMaxH3AppendRefVideo.define_schema()
    last = schema.inputs[-1]
    assert last.id == "use_vae" and last.optional and last.default is True, last.id

    items_off, blocks_off = R._compile_reference_records(
        off.args[0], _VideoVae(), _AudioVae(), width=64, height=64, frame_count=22)
    items_on, blocks_on = R._compile_reference_records(
        on.args[0], _VideoVae(), _AudioVae(), width=64, height=64, frame_count=22)
    kinds = [i["type"] for i in items_on]
    assert kinds == ["image", "audio", "video"], kinds
    assert [i["type"] for i in items_off] == kinds, "turning the VAE copy off must not change the presentation"
    assert all(torch.equal(a["data"], b["data"]) for a, b in zip(items_off, items_on) if "data" in a), (
        "turning the VAE copy off must not change what the text encoder is shown")
    assert [b["kind"] for b in blocks_on] == ["image", "video_audio"], "red control: with use_vae on, the video gets its block"
    assert [b["kind"] for b in blocks_off] == ["image"], f"the video with use_vae off still got a block: {[b['kind'] for b in blocks_off]}"

    reference_report = importlib.import_module(f"{_REPO.name}.reference_report")
    priced = reference_report.price_references(off.args[0], 64, 64, 22)
    rows = [getattr(item, "dit_rows", 0) for item in priced.items]
    assert rows[0] > 0 and rows[1] == 0, rows
    assert priced.items[1].audio_rows == 0, priced.items[1].audio_rows


def append_sizing_reaches_the_encoded_geometry():
    """`short_edge` and `allow_upscale` on the append change what the VAE gets.

    The gap this closes: nothing asserted that the two inputs folded onto
    `MiniMaxH3AppendRefImage` on 2026-08-24 are read by
    `_compile_reference_records` at all. Replacing `record.short_edge` /
    `record.allow_upscale` with the function defaults left every other control
    green -- `check_node_ids` compares schemas, `check_typed_reference_consumers`
    compares the static preflight adapter, and the two `image_policy` contracts
    exercise stage two only -- while every shipped graph silently lost its
    upscale, a 4x change in sequence length. It is the knob with the largest
    blast radius in that change and it had no runtime control.

    Asserted through the registered node on the real compiler, against the
    LATENT GRID the DiT is handed, not against an intermediate. A 64x64 source
    with `short_edge=256, allow_upscale=True` must reach 256x256; the same
    record with the defaults must stay at 64x64.
    """
    def encoded(**record_kwargs):
        records = (R.RuntimeImageReference(_frames(1, 64, 64), "max",
                                           **record_kwargs),)
        output = R.MiniMaxH3ReferenceConditioning.execute(
            clip=_Clip(), vae=_VideoVae(), audio_vae=_AudioVae(),
            references=records, prompt="use <Picture 1>",
            width=64, height=64, length=22,
        )
        block = output.args[0][0][1]["minimax_refs"][0]
        return block["latent_w"] * 16, block["latent_h"] * 16

    default = encoded()
    assert default == (64, 64), (
        f"a 64x64 reference under the append defaults should stay 64x64, got "
        f"{default}; core never upscales")

    upscaled = encoded(short_edge=256, allow_upscale=True)
    assert upscaled == (256, 256), (
        f"short_edge=256 with allow_upscale did not reach the encoder: the DiT "
        f"was handed {upscaled}. The append's sizing is not being read.")

    # short_edge alone, without the upscale flag, must NOT enlarge -- otherwise
    # the assertion above would pass on a build that ignored allow_upscale.
    clamped = encoded(short_edge=256, allow_upscale=False)
    assert clamped == (64, 64), (
        f"allow_upscale=False still enlarged to {clamped}; the flag is not "
        f"being read independently of short_edge")


def image_policy_is_opt_in_and_the_two_differ():
    """`comfy` changes nothing, and `release` is genuinely different from it.

    The failure this exists for is a policy selector that silently collapses to
    one branch. Asserting each policy against its own declared bounds cannot
    catch that -- a branch reading the wrong config would still agree with
    it. So this asserts the two DISAGREE on one input, which is only true if
    the selection is real, and it does so through the compiler the node runs,
    not only through the sizing function.

    The input: a 224x224 still sits under the release's floor
    (`vendor_config.image_pixel_bounds()`), so `release` must ENLARGE it
    before the VAE -- both towers then encode one size -- where `comfy`
    hands the same still through untouched. A 16:9 reference at the
    release's 2048 short edge sits inside the release ceiling and must not
    move, so a policy that only enforced a floor fails the same case.
    """
    role = (3648, 2048)
    release = R._configured_qwen_image_size(*role, "release")
    assert release == role, (
        f"the release still policy resized a reference inside its own "
        f"ceiling: {role} -> {release}")

    # The floor, in the other direction. A policy that only clamps a ceiling
    # would pass everything above and go green here.
    small = (224, 224)
    floor = R._configured_qwen_image_size(*small, "release")
    assert floor[0] * floor[1] > small[0] * small[1], (
        f"release still policy left {small} below its own floor: {floor}")

    # Through the compiler, where the selection actually happens. `max` with
    # no upscale keeps stage one at the source, so what moves is stage two.
    still = (R.RuntimeImageReference(_frames(1, *small), "max"),)

    def both_views(image_policy):
        vae = _VideoVae()
        items, _ = R._compile_reference_records(
            still, vae, _AudioVae(), 64, 64, 22, image_policy=image_policy)
        return (tuple(vae.inputs[0].shape[1:3]),
                tuple(items[0]["data"].shape[1:3]))

    comfy_vae, comfy_qwen = both_views("comfy")
    assert comfy_vae == small and comfy_qwen == small, (
        f"comfy still policy moved a {small} still: VAE {comfy_vae}, Qwen "
        f"{comfy_qwen}; core applies nothing and neither may this")
    release_vae, release_qwen = both_views("release")
    assert release_vae == release_qwen == (floor[1], floor[0]), (
        f"release still policy did not put both towers on its floor: VAE "
        f"{release_vae}, Qwen {release_qwen}, floor {floor}")
    assert release_qwen != comfy_qwen, (
        "comfy and release still policies agreed on a still under the "
        "release floor -- the selector is not selecting")

    # `comfy` has no configured processor and must refuse to invent one rather
    # than quietly returning somebody else's bounds.
    for settings in (R._qwen_image_settings, G.qwen_image_settings):
        try:
            settings("comfy")
        except ValueError as exc:
            assert "no configured processor" in str(exc), exc
        else:
            raise AssertionError("comfy still policy returned processor settings")


def append_node_defaults_are_the_serving_defaults():
    """What a fresh node in the UI is pre-filled with, read off the schema.

    The video model's copy is what sglang, diffusers and DiffSynth prepare
    (since 2026-09-13): `size_policy=max` at the release's 2048 short edge
    WITH upscale. The encoder's copy is NOT theirs since 2026-10-03: the
    owner made `qwen_view = separate` at `h3_rules.REF_QWEN_SHORT_EDGE` the
    default, on two clips (`docs/state/decisions.md`); the serving
    implementations share one copy, which is the second option. A
    DynamicCombo's default is its FIRST option: that is what the UI
    pre-selects, and there is no second copy of it to compare against, so
    the order of `options` is the observable, and the nested inputs'
    `default` attributes are the rest. Typed here would be a cache of the
    schema; this reads it.

    The title's "omits every input" is the UI's fresh node, not an API
    prompt. Core substitutes nothing for a DynamicCombo an API prompt leaves
    out: its expansion drops the input from the required set, validation
    passes, and `execute` fails for want of the argument (driven through
    `comfy_api.latest._io.get_finalized_class_inputs`, 2026-10-04). This
    case does not drive that; the generator always writes both selections.
    """
    schema = R.MiniMaxH3AppendRefImage.define_schema()
    by_id = {spec.id: spec for spec in schema.inputs}
    size_policy, qwen_view = by_id["size_policy"], by_id["qwen_view"]

    assert size_policy.options[0].key == "max", (
        [option.key for option in size_policy.options])
    nested = {spec.id: spec.default for spec in size_policy.options[0].inputs}
    assert nested == {"dit_short_edge": 2048, "allow_upscale": True}, nested
    assert R.REF_IMAGE_SHORT_EDGE == 2048, (
        "the schema default is the release constant; if the constant moved, "
        "this case and the node's tooltip both need to say so")

    assert [option.key for option in qwen_view.options] == ["separate", "shared"], (
        [option.key for option in qwen_view.options])
    by_key = {option.key: option for option in qwen_view.options}
    assert by_key["shared"].inputs == [], "shared carries no size member"
    nested = {spec.id: spec.default for spec in by_key["separate"].inputs}
    h3_rules = importlib.import_module(f"{_REPO.name}.h3_rules")
    assert nested == {"qwen_short_edge": h3_rules.REF_QWEN_SHORT_EDGE}, nested

    # And the executor agrees with the schema: a bare selection with no
    # nested members is the "schema's own defaults" branch of `execute`, and
    # it must land where the schema says, not on a second copy of the values.
    record = R.MiniMaxH3AppendRefImage.execute(
        _frames(1, 64, 64), "max", "separate").args[0][-1]
    assert (record.size_policy, record.short_edge, record.allow_upscale,
            record.qwen_short_edge) == ("max", 2048, True, h3_rules.REF_QWEN_SHORT_EDGE), record
    shared = R.MiniMaxH3AppendRefImage.execute(
        _frames(1, 64, 64), "max", "shared").args[0][-1]
    assert shared.qwen_short_edge == 0, shared


def qwen_view_is_separate_from_the_vae_view():
    """`qwen_short_edge` gives the encoder its own view; the VAE keeps stage one.

    Arms on one 640x480 source, `size_policy=max`, no upscale, so the stage-one
    role size is the source:

    1. `qwen_short_edge=0`: one tensor, both consumers, the same object.
    2. `qwen_short_edge=960`: the VAE encodes 640x480; the Qwen item is
       1280x960 (scaled from the source, nearest 32).
    3. Under `image_policy=release`, whose ceiling admits 1280x960, stage
       two shapes the Qwen view only; the VAE view is untouched.
    4. Under `release` with a Qwen view below the release FLOOR, the Qwen
       view is raised to the floor and the VAE view still is not: the knob's
       loud caveat -- the policy can override the requested view -- asserted
       rather than described.

    The red harness feeds the Qwen view to the VAE (M9); arm 2's VAE-shape
    assertion is what goes red.
    """
    source = _frames(1, 480, 640)

    def compile_one(qwen_short_edge, image_policy="comfy"):
        vae = _VideoVae()
        record = R.RuntimeImageReference(source, "max", qwen_short_edge=qwen_short_edge)
        items, blocks = R._compile_reference_records(
            (record,), vae, _AudioVae(), 64, 64, 22,
            image_policy=image_policy,
        )
        return vae.inputs[0], items[0]["data"], blocks[0]

    vae_in, qwen_in, block = compile_one(0)
    assert vae_in is qwen_in, "with qwen_short_edge=0 the two consumers must share one tensor"
    assert tuple(vae_in.shape[1:3]) == (480, 640)

    vae_in, qwen_in, block = compile_one(960)
    assert tuple(vae_in.shape[1:3]) == (480, 640), (
        f"the VAE received the Qwen view: {tuple(vae_in.shape[1:3])}")
    assert tuple(qwen_in.shape[1:3]) == (960, 1280), (
        f"the Qwen view is not the 960 short-edge view: {tuple(qwen_in.shape[1:3])}")
    assert (block["latent_h"], block["latent_w"]) == (480 // 16, 640 // 16), (
        "the reference-latent grid does not follow the VAE view")

    (release_floor, release_ceiling), _ = G.qwen_image_settings("release")
    assert release_floor < 1280 * 960 <= release_ceiling, (
        "arm 3 needs a Qwen view the release admits; the release bounds moved")
    vae_in, qwen_in, _ = compile_one(960, "release")
    assert tuple(vae_in.shape[1:3]) == (480, 640), (
        "release policy resized the VAE view although a Qwen view exists")
    assert tuple(qwen_in.shape[1:3]) == (960, 1280)

    # A 64 short edge scales the source to a view far under the release floor.
    assert 64 * 96 < release_floor, "arm 4 needs a view under the release floor"
    vae_in, qwen_in, _ = compile_one(64, "release")
    assert tuple(vae_in.shape[1:3]) == (480, 640), (
        "release policy resized the VAE view while raising the Qwen view")
    qh, qw = qwen_in.shape[1:3]
    assert qw * qh >= release_floor, (
        f"the release floor did not raise the Qwen view: {qw}x{qh}")

    # `qwen_view` is a DynamicCombo since 2026-08-31; the size arrives nested
    # under the `separate` option, not as a flat kwarg.
    def _sep(n):
        return {"qwen_view": "separate", "qwen_short_edge": n}

    # The node refuses a sub-grid value and records the field.
    try:
        R.MiniMaxH3AppendRefImage.execute(source, _max_policy(), _sep(16))
    except ValueError as exc:
        assert "qwen_short_edge" in str(exc), exc
    else:
        raise AssertionError("a sub-grid qwen_short_edge was accepted")
    records = R.MiniMaxH3AppendRefImage.execute(
        source, _max_policy(), _sep(960)).args[0]
    assert records[-1].qwen_short_edge == 960
    # `shared` is now the only way to reach the one-view path, and it is named
    # rather than typed as a zero.
    #
    # **This case used to assert that OMITTING the input yielded 0**, which
    # encoded the defect fixed on 2026-08-31: the schema said 512 and
    # `execute`'s signature said 0, and ComfyUI does not inject a schema
    # default for an omitted API input, so the two paths rendered differently.
    # Omission is no longer expressible -- `qwen_view` is required.
    assert R.MiniMaxH3AppendRefImage.execute(
        source, _max_policy(), "shared").args[0][-1].qwen_short_edge == 0


def preflight_prices_the_two_views():
    """The static reader prices reference-latent rows and Qwen tokens apart.

    One 640x480 reference under the guarded loader: the latent rows follow
    the VAE view, the Qwen tokens follow the Qwen view, and a contract whose
    ceiling is under that view reports it clamped -- the knob's caveat, in
    the report. The narrow contract is synthetic (no shipped encoder declares
    one), which is the point: it proves the pricer reads the contract it is
    handed rather than a module default.
    """
    P = _preflight()

    def graph(qwen_short_edge):
        return {
            "1": {"class_type": "MiniMaxH3EncoderLoader",
                  "inputs": {"encoder_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"}},
            "2": {"class_type": "MiniMaxH3AppendRefImage",
                  "inputs": {"image": ["9", 0], "size_policy": "max",
                             "allow_upscale": False, "dit_short_edge": 2048,
                             "qwen_short_edge": qwen_short_edge}},
            "3": {"class_type": "MiniMaxH3ReferenceConditioning",
                  "inputs": {"clip": ["1", 0], "references": ["2", 0]}},
        }
    for edge in (0, 960):
        media, policies, typed = P._reference_media(graph(edge)["3"]["inputs"], graph(edge))
        assert typed and list(policies.values())[0]["qwen_short_edge"] == edge, policies

    assert P._qwen_view_size(640, 480, 960) == (1280, 960)
    native_contract = L.native_encoder_contract()
    narrow = dict(native_contract, image_bounds=(1024, 4096), source="test-narrow")
    priced = P._qwen_tokens(1280, 960, narrow)
    assert priced is not None
    pw, ph, tokens, owner = priced
    assert pw * ph <= 4096 and "encoder contract (test-narrow)" in owner, priced
    wide = dict(native_contract, image_bounds=(65536, 16777216))
    assert P._qwen_tokens(1280, 960, wide)[:3] == (1280, 960, 1200)
    native = P._qwen_tokens(1280, 960, None)
    assert native is not None and native[3] == "native ComfyUI", native
    # And the contract the guarded loader records prices the same as core's
    # own defaults, which is what "records core's bounds" has to mean.
    assert P._qwen_tokens(1280, 960, native_contract)[:3] == native[:3]


def preflight_reads_the_vae_gate_off_the_graph():
    """An encoder-only conditioner is priced at zero DiT reference rows.

    The gate is the presence of the `vae` key on the conditioner's inputs,
    on either node, which is exactly what the executor hands the node as
    None. Pricing rows a graph will never build is the failure this guards:
    the first encoder-only arm was priced at the full row count on
    2026-09-03 before this read existed.
    """
    P = _preflight()
    assert P.reference_rows_reach_the_dit({"vae": ["3", 0], "references": ["2", 0]})
    assert not P.reference_rows_reach_the_dit({"references": ["2", 0]})
    assert not P.reference_rows_reach_the_dit({"ref_images.ref_image_0": ["9", 0]})
    assert P.reference_rows_reach_the_dit({"vae": ["3", 0], "ref_images.ref_image_0": ["9", 0]})


def preflight_resolves_the_contract_from_the_loader_node():
    """The static reader reads the graph's loader, as the runtime reads its CLIP.

    Same conditioner inputs, two loaders: core's `CLIPLoader` yields no
    contract and the reason; the guarded loader yields what it records at
    runtime, core's own bounds read out of core (`native_encoder_contract`).
    An unlinked `clip` yields none, never a guess.
    """
    P = _preflight()

    def graph(loader_type, **loader_inputs):
        return {
            "1": {"class_type": loader_type, "inputs": loader_inputs},
            "2": {"class_type": "MiniMaxH3ReferenceConditioning",
                  "inputs": {"clip": ["1", 0]}},
        }

    native = graph("CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    contract, note = P._encoder_contract_for(native["2"]["inputs"], native)
    assert contract is None and "CLIPLoader" in note, (contract, note)

    guarded = graph("MiniMaxH3EncoderLoader",
                    encoder_name="qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
    contract, note = P._encoder_contract_for(guarded["2"]["inputs"], guarded)
    assert contract == L.native_encoder_contract(), (contract, note)
    assert L.CONTRACT_SOURCE in note, note

    unlinked = {"2": {"class_type": "MiniMaxH3ReferenceConditioning", "inputs": {}}}
    contract, note = P._encoder_contract_for(unlinked["2"]["inputs"], unlinked)
    assert contract is None and "not linked" in note, (contract, note)


class _CoarserVideoVae(_VideoVae):
    """A VAE that compresses 32x where the installed one compresses 16x."""

    def encode(self, frames):
        import torch
        self.inputs.append(frames)
        return torch.zeros(
            1, 24, max(1, int(frames.shape[0])),
            int(frames.shape[1]) // 32, int(frames.shape[2]) // 32,
        )


def video_grid_is_read_off_the_latent():
    """The reference video block carries the grid the VAE returned, not `canvas // 16`.

    Control first: the installed VAE's 16x compression gives the same grid both
    ways, so the shipped path is unchanged. Then a VAE that compresses 32x: the
    block must carry its 2x2 grid for a 64x64 canvas (a canvas-derived 4x4
    would describe tokens that are not in the latent) and the disagreement is
    logged, as it is for a still.
    """
    import logging
    frames = _frames(22, 64, 64)
    records = (R.RuntimeVideoReference(frames, 24.0, None),)
    original_adapt_canvas = R.adapt_canvas
    R.adapt_canvas = lambda _w, _h: (64, 64)
    try:
        _, blocks = R._compile_reference_records(
            records, _VideoVae(), _AudioVae(), 64, 64, 22, video_policy="release")
        assert (blocks[0]["latent_h"], blocks[0]["latent_w"]) == (4, 4), blocks[0]

        class _Capture(logging.Handler):
            def __init__(self):
                super().__init__()
                self.messages = []

            def emit(self, record):
                self.messages.append(record.getMessage())

        handler = _Capture()
        logger = logging.getLogger(R.__name__)
        logger.addHandler(handler)
        try:
            _, blocks = R._compile_reference_records(
                records, _CoarserVideoVae(), _AudioVae(), 64, 64, 22,
                video_policy="release")
        finally:
            logger.removeHandler(handler)
        assert (blocks[0]["latent_h"], blocks[0]["latent_w"]) == (2, 2), blocks[0]
        assert tuple(blocks[0]["latent"].shape[-2:]) == (2, 2)
        assert any("latent grid" in m for m in handler.messages), handler.messages
    finally:
        R.adapt_canvas = original_adapt_canvas


CHECKS = (
    append_is_copy_on_add_and_ordered,
    video_metadata_is_owned,
    video_is_normalized_to_24fps,
    audio_is_stereo_and_target_bounded,
    ref_audio_end_padded_to_the_hop,
    vhs_lazy_audio_mapping_is_accepted,
    compiler_preserves_one_order_for_both_lists,
    release_video_policy_is_opt_in_and_two_stage,
    release_policy_floor_is_two_sampled_frames,
    append_sizing_reaches_the_encoded_geometry,
    image_policy_is_opt_in_and_the_two_differ,
    append_node_defaults_are_the_serving_defaults,
    qwen_view_is_separate_from_the_vae_view,
    preflight_prices_the_two_views,
    preflight_resolves_the_contract_from_the_loader_node,
    conditioning_node_assembles_the_real_payload_shape,
    encoder_only_references_skip_the_dit_rows,
    a_still_can_turn_its_own_vae_copy_off,
    a_video_can_turn_its_own_vae_copy_off,
    preflight_reads_the_vae_gate_off_the_graph,
    video_grid_is_read_off_the_latent,
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
