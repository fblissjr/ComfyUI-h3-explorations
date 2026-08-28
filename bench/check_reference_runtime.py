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
G = importlib.import_module(f"{_REPO.name}.reference_geometry")
A = importlib.import_module(f"{_REPO.name}.h3_awq_encoder")


def _v1_contract():
    """The current W4 artifact's declaration, read from its checked-in snapshot."""
    return A.snapshot_contract(A.CONFIG_DIR)


def _stub_clip(contract=None):
    """A CLIP-shaped object carrying (or not carrying) a stamped contract.

    Shaped like what the loader stamps: `clip.cond_stage_model.qwen3vl_32b
    .transformer._h3_encoder_contract`. `None` builds the transformer with no
    such attribute at all, which is what core's `CLIPLoader` produces.
    """
    from types import SimpleNamespace
    transformer = SimpleNamespace()
    if contract is not None:
        transformer._h3_encoder_contract = contract
    return SimpleNamespace(cond_stage_model=SimpleNamespace(
        qwen3vl_32b=SimpleNamespace(transformer=transformer)))


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
        _frames(1), {"size_policy": "match"}, references=audio_out
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
            video_policy="encoder", contract=_v1_contract(),
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
    """Encoder policy is governed by the loaded encoder's contract, not release data.

    The checked-in snapshot and the release happen to agree today. Make the
    contract's temporal factor disagree so this case proves which authority
    is read; comparing their current values would go green even if the wrong
    one won.
    """
    contract = _v1_contract()
    contract["video_geometry"] = dict(contract["video_geometry"],
                                      temporal_patch_size=4)
    try:
        R._encoder_qwen_video_size(2, 960, 544, contract)
    except ValueError as exc:
        assert "encoder video policy needs at least 4" in str(exc), exc
    else:
        raise AssertionError("encoder policy ignored its contract's temporal factor")

    assert R._release_qwen_video_size(2, 960, 544), (
        "release policy incorrectly inherited the encoder's test geometry")

    # And with no contract at all the encoder settings are refused, not
    # defaulted: the substitution to native happens in effective_policy, in
    # one place, where it is logged.
    try:
        R._qwen_video_settings("encoder")
    except ValueError as exc:
        assert "loaded encoder's contract" in str(exc), exc
    else:
        raise AssertionError("encoder video settings were invented without a contract")


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


def image_policy_is_opt_in_and_the_three_differ():
    """`comfy` changes nothing, and the other two are genuinely different.

    The failure this exists for is a policy selector that silently collapses to
    one branch. Asserting each policy against its own declared bounds cannot
    catch that -- every branch reading the same config would still agree with
    it. So this asserts the three DISAGREE on one input, which is only true if
    the selection is real.

    The input is chosen to make all three differ: a 16:9 reference prepared at
    the release's 2048 short edge sits under the release ceiling untouched and
    far above the current encoder artifact's, and a 224x224 reference sits
    under BOTH floors, which is the half `keep_towers_matched` never modelled.
    """
    role = (3648, 2048)
    contract = _v1_contract()
    release = R._configured_qwen_image_size(*role, "release")
    encoder = R._configured_qwen_image_size(*role, "encoder", contract)
    assert release == role, (
        f"the release still policy resized a reference inside its own "
        f"ceiling: {role} -> {release}")
    assert encoder != role and encoder[0] * encoder[1] < role[0] * role[1], (
        f"the encoder still policy left {role} alone; its declared budget is "
        f"far below that and it must shrink it")
    assert release != encoder, (
        "release and encoder still policies agreed on an input where their "
        "declared budgets differ by orders of magnitude -- the selector is "
        "not selecting")

    # The floor, in the other direction. A policy that only clamps a ceiling
    # would pass everything above and go green here.
    small = (224, 224)
    for policy in ("release", "encoder"):
        out = R._configured_qwen_image_size(*small, policy, contract)
        assert out[0] * out[1] > small[0] * small[1], (
            f"{policy} still policy left {small} below its own floor: {out}")

    # `comfy` has no configured processor and must refuse to invent one rather
    # than quietly returning somebody else's bounds.
    try:
        R._qwen_image_settings("comfy")
    except ValueError as exc:
        assert "no configured processor" in str(exc), exc
    else:
        raise AssertionError("comfy still policy returned processor settings")


def image_policy_reads_encoder_config():
    """The encoder still policy is governed by the contract it is handed.

    The checked-in still processor and the release agree on patch geometry
    today, so comparing their current values would go green even if the wrong
    authority won. Hand in a contract with a ceiling nothing else declares
    and assert the bounds each policy applies come from its own source.
    """
    contract = _v1_contract()
    contract["image_bounds"] = (1024, 4096)
    out = G.qwen_image_size(3648, 2048, "encoder", contract)
    assert out[0] * out[1] <= 4096, (
        f"encoder still policy ignored the contract's ceiling: {out}")
    assert G.qwen_image_size(3648, 2048, "release") == (3648, 2048), (
        "release still policy inherited the contract's test bounds")
    try:
        G.qwen_image_settings("encoder")
    except ValueError as exc:
        assert "loaded encoder's contract" in str(exc), exc
    else:
        raise AssertionError("encoder still settings were invented without a contract")


def encoder_policy_binds_to_the_loaded_clip():
    """`encoder` is whatever the CLIP this node was handed declares.

    Three arms on one 640x640 reference, which the current W4 snapshot's
    301,056-pixel ceiling shrinks and the native path leaves alone:

    1. a CLIP with no stamped contract (core's `CLIPLoader`) resolves
       `encoder` to native for both stages, so the VAE sees 640x640 and the
       Qwen video item is the untouched sample;
    2. a CLIP stamped with the W4 contract shrinks the still to its ceiling;
    3. a CLIP stamped with a different ceiling shrinks it differently.

    The third arm is the one that matters: it is only true if the resolver
    reads the INSTANCE it was given. Reading a module default passes the
    first two.
    """
    still = (R.RuntimeImageReference(_frames(1, 640, 640), "max"),)

    def encoded_still_size(contract):
        vae = _VideoVae()
        R._compile_reference_records(
            still, vae, _AudioVae(), 64, 64, 22,
            video_policy="encoder", image_policy="encoder", contract=contract,
        )
        return tuple(vae.inputs[0].shape[1:3])

    # Through the names the NODE binds (`R.`), so a mutation of the node's
    # resolver reaches this check; the geometry module is the implementation,
    # not the seam under test.
    assert R.encoder_contract_from_clip(_stub_clip(None)) is None, (
        "a CLIP with no stamped contract reported one")
    assert R.effective_policy("encoder", None) == "comfy"
    assert R.effective_policy("release", None) == "release"
    try:
        native = encoded_still_size(R.encoder_contract_from_clip(_stub_clip(None)))
    except ValueError as exc:
        raise AssertionError(
            f"encoder on a CLIP that declares nothing raised instead of "
            f"resolving to the native path: {exc}") from exc
    assert native == (640, 640), (
        f"encoder on a CLIP that declares nothing resized the still: {native}")

    v1 = _v1_contract()
    stamped = R.encoder_contract_from_clip(_stub_clip(v1))
    assert stamped == v1, "the stamped contract did not come back intact"
    w4 = encoded_still_size(stamped)
    assert w4[0] * w4[1] <= v1["image_bounds"][1], (
        f"encoder on the W4 contract left {w4} above its ceiling")

    other = dict(v1, image_bounds=(1024, 4096), source="test-artifact")
    tiny = encoded_still_size(R.encoder_contract_from_clip(_stub_clip(other)))
    assert tiny[0] * tiny[1] <= 4096 and tiny != w4, (
        f"a different stamped contract produced the W4 result {tiny}: the "
        "resolver is reading a module, not the CLIP")

    # A partial stamp is refused, not partially applied.
    partial = {"source": "broken", "image_bounds": (1, 2)}
    try:
        R.encoder_contract_from_clip(_stub_clip(partial))
    except ValueError as exc:
        assert "missing" in str(exc), exc
    else:
        raise AssertionError("a partial contract was accepted")

    # The video stage binds to the same contract: a contract whose temporal
    # factor cannot be met makes the encoder video stage refuse, where the
    # native resolution runs the raw sample through untouched.
    frames = _frames(22, 32, 32)
    video = (R.RuntimeVideoReference(frames, 24.0, None),)
    original_adapt_canvas = R.adapt_canvas
    try:
        R.adapt_canvas = lambda w, h: (w, h)
        try:
            items, _ = R._compile_reference_records(
                video, _VideoVae(), _AudioVae(), 32, 32, 22,
                video_policy="encoder", contract=None,
            )
        except ValueError as exc:
            raise AssertionError(
                f"encoder video on a native CLIP raised instead of resolving "
                f"to the native path: {exc}") from exc
        assert tuple(items[0]["data"].shape[1:3]) == (32, 32), (
            "encoder on a native CLIP resized the Qwen video sample")
        strict = dict(v1, video_geometry=dict(v1["video_geometry"],
                                              temporal_patch_size=8))
        try:
            R._compile_reference_records(
                video, _VideoVae(), _AudioVae(), 32, 32, 22,
                video_policy="encoder", contract=strict,
            )
        except ValueError as exc:
            assert "encoder video policy needs at least 8" in str(exc), exc
        else:
            raise AssertionError("the video stage did not read the stamped contract")
    finally:
        R.adapt_canvas = original_adapt_canvas


def qwen_view_is_separate_from_the_vae_view():
    """`qwen_short_edge` gives the encoder its own view; the VAE keeps stage one.

    Arms on one 640x480 source, `size_policy=max`, no upscale, so the stage-one
    role size is the source:

    1. `qwen_short_edge=0`: one tensor, both consumers, the same object.
    2. `qwen_short_edge=960`: the VAE encodes 640x480; the Qwen item is
       1280x960 (scaled from the source, nearest 32).
    3. Under `image_policy=encoder` with a contract whose ceiling admits it,
       stage two shapes the Qwen view only; the VAE view is unclamped.
    4. Under a contract whose ceiling does not admit it (the v1 snapshot's),
       the Qwen view is clamped back and the VAE view still is not: the
       knob's loud caveat, asserted rather than described.

    The red harness feeds the Qwen view to the VAE (M9); arm 2's VAE-shape
    assertion is what goes red.
    """
    source = _frames(1, 480, 640)

    def compile_one(qwen_short_edge, image_policy="comfy", contract=None):
        vae = _VideoVae()
        record = R.RuntimeImageReference(source, "max", qwen_short_edge=qwen_short_edge)
        items, blocks = R._compile_reference_records(
            (record,), vae, _AudioVae(), 64, 64, 22,
            image_policy=image_policy, contract=contract,
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

    wide = dict(_v1_contract(), image_bounds=(65536, 16777216), source="test-wide")
    vae_in, qwen_in, _ = compile_one(960, "encoder", wide)
    assert tuple(vae_in.shape[1:3]) == (480, 640), (
        "encoder policy clamped the VAE view although a Qwen view exists")
    assert tuple(qwen_in.shape[1:3]) == (960, 1280)

    v1 = _v1_contract()
    vae_in, qwen_in, _ = compile_one(960, "encoder", v1)
    assert tuple(vae_in.shape[1:3]) == (480, 640)
    qh, qw = qwen_in.shape[1:3]
    assert qw * qh <= v1["image_bounds"][1], (
        f"the v1 contract's ceiling did not clamp the Qwen view: {qw}x{qh}")

    # The node refuses a sub-grid value and records the field.
    try:
        R.MiniMaxH3AppendRefImage.execute(source, _max_policy(), qwen_short_edge=16)
    except ValueError as exc:
        assert "qwen_short_edge" in str(exc), exc
    else:
        raise AssertionError("a sub-grid qwen_short_edge was accepted")
    records = R.MiniMaxH3AppendRefImage.execute(source, _max_policy(), qwen_short_edge=960).args[0]
    assert records[-1].qwen_short_edge == 960
    assert R.MiniMaxH3AppendRefImage.execute(
        source, _max_policy()).args[0][-1].qwen_short_edge == 0


def preflight_prices_the_two_views():
    """The static reader prices reference-latent rows and Qwen tokens apart.

    One 640x480 reference under the AWQ loader: the latent rows follow the
    VAE view, the Qwen tokens follow the Qwen view, and under the v1 contract
    the Qwen view is reported clamped -- the knob's caveat, in the report.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "preflight_graph", _REPO / "bench" / "preflight_graph.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)

    def graph(qwen_short_edge):
        return {
            "1": {"class_type": "MiniMaxH3AWQEncoderLoader",
                  "inputs": {"encoder_name": "qwen3vl_32b_minimax_h3_w4a16_awq.safetensors"}},
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
    v1 = _v1_contract()
    priced = P._qwen_tokens(1280, 960, v1)
    assert priced is not None
    pw, ph, tokens, owner = priced
    assert pw * ph <= v1["image_bounds"][1] and "encoder contract" in owner, priced
    wide = dict(v1, image_bounds=(65536, 16777216))
    assert P._qwen_tokens(1280, 960, wide)[:3] == (1280, 960, 1200)
    native = P._qwen_tokens(1280, 960, None)
    assert native is not None and native[3] == "native ComfyUI", native


def preflight_resolves_encoder_from_the_loader_node():
    """The static reader binds `encoder` to the graph's loader, as the node does.

    Same conditioner inputs, two loaders: core's `CLIPLoader` yields no
    contract and the reason; the adapter's loader yields the artifact's
    contract. An unknown artifact name yields none, never a guess.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "preflight_graph", _REPO / "bench" / "preflight_graph.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)

    def graph(loader_type, **loader_inputs):
        return {
            "1": {"class_type": loader_type, "inputs": loader_inputs},
            "2": {"class_type": "MiniMaxH3ReferenceConditioning",
                  "inputs": {"clip": ["1", 0]}},
        }

    native = graph("CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    contract, note = P._encoder_contract_for(native["2"]["inputs"], native)
    assert contract is None and "CLIPLoader" in note, (contract, note)

    awq = graph("MiniMaxH3AWQEncoderLoader",
                encoder_name="qwen3vl_32b_minimax_h3_w4a16_awq.safetensors")
    contract, note = P._encoder_contract_for(awq["2"]["inputs"], awq)
    assert contract == _v1_contract(), (contract, note)

    unknown = graph("MiniMaxH3AWQEncoderLoader", encoder_name="nobody.safetensors")
    contract, note = P._encoder_contract_for(unknown["2"]["inputs"], unknown)
    assert contract is None and "not an artifact" in note, (contract, note)

    unlinked = {"2": {"class_type": "MiniMaxH3ReferenceConditioning", "inputs": {}}}
    contract, note = P._encoder_contract_for(unlinked["2"]["inputs"], unlinked)
    assert contract is None and "not linked" in note, (contract, note)


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
    append_sizing_reaches_the_encoded_geometry,
    image_policy_is_opt_in_and_the_three_differ,
    image_policy_reads_encoder_config,
    encoder_policy_binds_to_the_loaded_clip,
    qwen_view_is_separate_from_the_vae_view,
    preflight_prices_the_two_views,
    preflight_resolves_encoder_from_the_loader_node,
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
