#!/usr/bin/env python3
"""Prove the guarded H3 encoder loader guards what it claims to.

`h3_encoder_loader.py` adds three things to core's own load, and each is only
worth having if it can go red. Two of them are cheap to state and impossible to
check by reading, because the answer depends on ComfyUI's behaviour rather than
on ours:

- does a CORRECT checkpoint pass? A guard that rejects the shipped artifact is
  worse than no guard, and nothing about `strict=False` says what a good file
  reports.
- does a BROKEN one fail, and at which of the two failure shapes? Core does not
  reject an incomplete H3 file. It DETECTS A DIFFERENT ARCHITECTURE from the
  tensors present, so a missing key that happens to be one detection reads
  fails somewhere else entirely -- measured 2026-08-29, dropping
  `visual.deepstack_merger_list.0.norm.weight` sends the load into
  `comfy/text_encoders/flux.py` and dies parsing a Mistral tokenizer.

So the mutations here are not ceremony: before running them it was not known
which tensors fail as "missing" and which fail as "a different model", and the
answer changed the loader (`require_h3` and the construction wrapper exist
because of this file).

CPU only, no server, no CUDA. It does load the real encoder, which is
mmap-backed and returns in about a second once warm.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parent.parent

# ComfyUI's root ahead of the repo's PARENT, and the repo itself never on the
# path: this repo has its own `nodes.py`, so a bare `import nodes` inside
# `comfy_extras` finds ours and dies on a relative import. `docs/comfy_notes.md`
# carries the trap; the repo's modules are reached as a package instead.
sys.path.insert(0, str(REPO.parent))
sys.path.insert(0, str(COMFY))

import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True

import comfy.utils  # noqa: E402

loader = importlib.import_module(f"{REPO.name}.h3_encoder_loader")
vendor_config = importlib.import_module(f"{REPO.name}.vendor_config")

ENCODERS_DIR = COMFY / "models" / "text_encoders"


def _core_loaded_encoders() -> list:
    """Every encoder this node is FOR, read from `h3_config` rather than named.

    **Derived because naming one was the wrong population.** This file swept
    `int8_convrot` alone until 2026-08-29 and passed -- but the node's real
    population is `CORE_LOADED_ENCODERS`, which is also `nvfp4_awq`, and that
    one had never been through the guards. It happened to pass. An instrument
    that covers one member of a set it does not consult reads as coverage
    without being it, and a third core-loadable encoder would have joined the
    menu with nothing checking it. Loaded by path so nothing joins `sys.path`.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_h3_config_population", REPO / "workflows" / "h3_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [ENCODERS_DIR / name
            for name in sorted(getattr(module, "CORE_LOADED_ENCODERS", ()))
            if (ENCODERS_DIR / name).exists()]


#: The one the mutations run against. Which member does not matter -- they
#: exercise the guard, not the artifact -- so this is the file the shipped
#: graphs name, and it is asserted to be in the population above.
ENCODER = ENCODERS_DIR / "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"


def _with_incomplete_checkpoint(mutate):
    """Run one load as if the checkpoint on disk had been mutated.

    `mutate` is applied to BOTH the state dict core reads and the header the
    inventory guard reads, because a real incomplete file changes both and the
    guard compares one against the other. Mutating only the in-memory state
    dict simulates nothing that can happen on disk -- and the guard correctly
    stayed green when this harness did that, which is how the mistake was
    found rather than shipped.

    The alternative, writing a mutated 25 GiB copy, buys nothing this does not.
    """
    read_file, read_header = comfy.utils.load_torch_file, loader._file_tensors

    def reading(*args, **kwargs):
        out = read_file(*args, **kwargs)
        # `return_metadata=True` makes this a (state_dict, metadata) pair.
        mutate(out[0] if isinstance(out, tuple) else out)
        return out

    def heading(path):
        header = read_header(path)
        mutate(header)
        return header

    comfy.utils.load_torch_file = reading
    loader._file_tensors = heading
    try:
        return loader.load_guarded_clip(str(ENCODER), None)
    finally:
        comfy.utils.load_torch_file = read_file
        loader._file_tensors = read_header


def contract_is_derived_from_comfy_not_declared():
    """Every contract value is read out of ComfyUI, and agrees with the release."""
    contract = loader.native_encoder_contract()
    # The keys the readers consume: `bench/preflight_graph.py` prices the
    # still and video Qwen views off the bounds and geometry, and
    # `reference_report.py` names the source. Listed here because there is
    # no longer a shared constant (the geometry module's left with the AWQ
    # adapter on 2026-09-13); a key the loader drops goes red in the reader.
    assert set(contract) == {"source", "image_bounds", "image_geometry",
                             "video_bounds", "video_geometry"}, sorted(contract)
    assert contract["source"] == loader.CONTRACT_SOURCE

    from comfy.text_encoders import minimax
    from comfy.text_encoders.qwen_vl import process_qwen2vl_images

    for key, function in (("image_bounds", process_qwen2vl_images),
                          ("video_bounds", minimax.process_video_block)):
        pixels = loader._signature_defaults(function, ("min_pixels", "max_pixels"))
        assert contract[key] == (pixels["min_pixels"], pixels["max_pixels"]), key

    # The geometry the reference nodes price against has to be the release's,
    # or the token counts they compute are about a model nobody is running.
    declared = vendor_config.patch_geometry()
    for key, value in contract["image_geometry"].items():
        assert declared[key] == value, (key, value, declared[key])


def token_ids_refuse_a_malformed_declaration():
    """The shared id arithmetic rejects what it cannot be sure about."""
    tokens = vendor_config.additional_special_tokens()
    ids = loader.expected_special_token_ids(tokens)
    markers = vendor_config.h3_markers()
    # The seven H3 markers are the tail run, and the DiT was trained on these
    # exact ids; `docs/research/official_weights_metadata.md` owns why.
    assert [ids[token] for token in markers] == list(range(151669, 151676)), ids
    for bad, label in (([], "empty"), (tokens[:19] + [tokens[0]], "duplicate")):
        try:
            loader.expected_special_token_ids(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted a {label} special-token declaration")


def every_core_loadable_encoder_passes_every_guard():
    """The control, over the whole population. A guard that refuses a real
    artifact is worse than none, and "a real artifact" is every file this node
    is for -- not the one that happened to get written into this test."""
    population = _core_loaded_encoders()
    assert population, "no core-loadable H3 encoder is installed"
    assert ENCODER in population, (
        f"{ENCODER.name} is not in CORE_LOADED_ENCODERS; the mutation arm "
        "runs against a file this node is not for")
    started = time.monotonic()
    for path in population:
        clip = loader.load_guarded_clip(str(path), None)
        model = loader.require_h3(clip, path.name)
        # **A native CLIP must NOT claim a processor contract**, and this is the
        # assertion that keeps it that way. See `install_native_contract`: the
        # contract's `video_bounds` would be applied clip-wide where core's
        # budget is per-block, so stamping core's number into a clip-wide
        # field would misdescribe core. The policy that read the stamp is
        # gone (2026-09-13); the attribute staying absent is what keeps a
        # reader from bringing it back by accident.
        assert not hasattr(model, "_h3_encoder_contract"), (
            f"{path.name}: a native CLIP stamped a contract; nothing reads it "
            "and core's per-block budget is not the clip-wide field it would fill")
        # What IS recorded is the still bound, for `reference_report.py`, and
        # it must be the same derivation `native_encoder_contract` returns.
        assert model._h3_image_bounds == loader.native_encoder_contract()["image_bounds"], (
            f"{path.name}: _h3_image_bounds {model._h3_image_bounds!r} is not "
            "what native_encoder_contract derives")
        del clip, model
    names = ", ".join(path.name.replace("qwen3vl_32b_minimax_h3_", "") for path in population)
    return f"{len(population)} encoders ({names}) in {time.monotonic() - started:.1f}s"


def stamping_a_native_contract_would_shrink_reference_video():
    """The measurement behind not stamping. Prove the harm is real, not feared.

    `install_native_contract` withdrew the contract stamp on the strength of
    this. If a future change makes core's video budget clip-wide, or makes the
    node apply a clip-wide budget per block, these two stop disagreeing and the
    decision should be revisited rather than inherited.

    Reframed 2026-09-13, when the `encoder` video policy that applied a
    stamped contract clip-wide was deleted with the AWQ lane. The harm is
    still expressible: `release` is the one clip-wide budget the compiler
    has left, and it reads its bounds through `video_pixel_bounds`, so the
    case substitutes core's per-block number for the release's and drives
    the same `smart_resize` path. That is exactly "core's number in a
    clip-wide field", which is what stamping would have made live.
    """
    conditioning = importlib.import_module(f"{REPO.name}.reference_conditioning")
    contract = loader.native_encoder_contract()
    release_bounds = conditioning.video_pixel_bounds
    conditioning.video_pixel_bounds = lambda: contract["video_bounds"]
    shrunk = []
    try:
        for width, height in ((960, 544), (1344, 768)):
            for sampled in (31, 62):
                got = conditioning._configured_qwen_video_size(
                    sampled, width, height, "release")
                if got != (width, height):
                    shrunk.append((f"{width}x{height}", sampled, f"{got[0]}x{got[1]}"))
    finally:
        conditioning.video_pixel_bounds = release_bounds
    assert shrunk, (
        "core's per-block budget applied clip-wide no longer shrinks reference "
        "video. Either core's video budget became clip-wide or the compiler "
        "stopped applying a configured budget clip-wide; re-read "
        "install_native_contract's reasoning before trusting either.")
    return f"{len(shrunk)} of 4 cases shrink, e.g. {shrunk[0][0]}@{shrunk[0][1]} -> {shrunk[0][2]}"


def an_incomplete_checkpoint_cannot_load_quietly():
    """Both failure shapes, because core has two and they read nothing alike.

    Core reports `unexpected` at DEBUG, so on a normal server the last case is
    invisible without this loader.
    """
    cases = (
        ("drops a detection key",
         lambda sd: sd.pop("visual.deepstack_merger_list.0.norm.weight"),
         "could not be constructed"),
        ("drops a layer-49 linear",
         lambda sd: sd.pop("model.layers.49.self_attn.q_proj.weight"),
         "could not be constructed"),
        ("drops a mid-stack layernorm",
         lambda sd: sd.pop("model.layers.25.input_layernorm.weight"),
         "does not exactly populate"),
        # The final-norm trap, raised by the `dit` session 2026-08-29 and
        # verified here. ComfyUI builds the stack with `final_norm=False`, so
        # "layer 50's output" is the last hidden state with NO norm after it;
        # diffusers refuses a <=50-layer stack outright for this reason, and
        # DiffSynth sets `language_model.norm = Identity()`. A truncated
        # quantized artifact that kept its final norm would feed the DiT
        # post-norm conditioning. Nothing had to be added to catch it: the
        # inventory guard compares against a model that has no `model.norm.*`,
        # so the norm arrives as an unexpected tensor.
        ("keeps the final norm a truncated artifact must drop",
         lambda sd: sd.__setitem__("model.norm.weight",
                                   __import__("torch").zeros(5120)),
         "does not exactly populate"),
        ("carries an unexpected tensor",
         lambda sd: sd.__setitem__("model.layers.0.not_a_real_tensor",
                                   __import__("torch").zeros(4)),
         "does not exactly populate"),
        # Red at CONSTRUCTION, not at the inventory guard: core refuses to load
        # a [7] tensor into a [5120] parameter by itself. Kept because it
        # records which of the two shapes this case takes -- the inventory
        # guard's shape comparison is defence for a mismatch core tolerates,
        # and this says core does not tolerate this one.
        ("carries a wrongly shaped tensor",
         lambda sd: sd.__setitem__("model.layers.25.input_layernorm.weight",
                                   __import__("torch").zeros(7)),
         "could not be constructed"),
    )
    for label, mutate, expected in cases:
        try:
            _with_incomplete_checkpoint(mutate)
        except ValueError as exc:
            assert expected in str(exc), (label, str(exc)[:200])
        else:
            raise AssertionError(f"a checkpoint that {label} loaded green")
    return f"{len(cases)} mutations red, each with its own message"


def _core_clip_loader():
    """Core's own `CLIPLoader`, imported from ComfyUI's `nodes.py` by path.

    By path under its own name, because this repo has a `nodes.py` too
    (`docs/comfy_notes.md`, the `import nodes` trap).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_comfy_core_nodes", COMFY / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CLIPLoader


def device_is_cores_and_survives_a_rebuild():
    """The `device` input offers what core's `CLIPLoader` offers, hands core the
    same `model_options` core's loader builds, and a rebuild keeps them.

    Both of core's halves are read out of core, not typed here: the option
    list from its `INPUT_TYPES`, the mapping by calling its `load_clip` with
    `comfy.sd.load_clip` spied. The rebuild is the real one core performs for
    `clone(disable_dynamic=True)` and for `SelectCLIPDevice` to cpu, through
    the `cached_patcher_init` this loader registers. This harness runs with
    `--cpu`, so "default" also lands on the CPU and the devices alone cannot
    tell the options apart; the recorded `model_options` are the observable.
    """
    import comfy.sd

    core = _core_clip_loader()
    core_devices = tuple(core.INPUT_TYPES()["optional"]["device"][0])
    assert core_devices == loader.DEVICES, (core_devices, loader.DEVICES)
    schema = loader.MiniMaxH3EncoderLoader.define_schema()
    offered = next(i for i in schema.inputs if i.id == "device")
    assert tuple(offered.options) == core_devices, offered.options
    # Optional as core's is, so a graph written before the input existed runs.
    assert offered.optional and offered.default == "default", (
        offered.optional, offered.default)

    real_load_clip = comfy.sd.load_clip
    calls = []

    def spying(*args, **kwargs):
        calls.append(dict(kwargs.get("model_options") or {}))
        return real_load_clip(*args, **kwargs)

    def refusing(*args, **kwargs):
        calls.append(dict(kwargs.get("model_options") or {}))
        raise _Stop

    class _Stop(Exception):
        pass

    comfy.sd.load_clip = refusing
    try:
        for device in core_devices:
            calls.clear()
            try:
                core().load_clip(ENCODER.name, type="minimax", device=device)
            except _Stop:
                pass
            assert calls == [loader.model_options_for(device)], (device, calls)

        comfy.sd.load_clip = spying
        calls.clear()
        clip = loader.MiniMaxH3EncoderLoader.execute(ENCODER.name, device="cpu").result[0]
        cpu = loader.model_options_for("cpu")
        assert calls == [cpu], calls
        factory, factory_args = clip.patcher.cached_patcher_init[:2]
        assert factory is loader.load_guarded_model_patcher, factory
        rebuilt = factory(*factory_args, disable_dynamic=True)
        assert calls == [cpu, cpu], calls
        assert (rebuilt.load_device, rebuilt.offload_device) == (
            cpu["load_device"], cpu["offload_device"]), (
            rebuilt.load_device, rebuilt.offload_device)
        del clip, rebuilt
    finally:
        comfy.sd.load_clip = real_load_clip
    return f"core offers {list(core_devices)}; cpu reached core on load and on rebuild"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    if not ENCODER.exists():
        print(f"SKIP  {ENCODER.name} is not installed")
        return 2

    cases = [
        ("contract derived from comfy", contract_is_derived_from_comfy_not_declared),
        ("special-token arithmetic", token_ids_refuse_a_malformed_declaration),
        ("every core-loadable encoder passes", every_core_loadable_encoder_passes_every_guard),
        ("native contract stays unstamped",
         stamping_a_native_contract_would_shrink_reference_video),
        ("incomplete checkpoints red", an_incomplete_checkpoint_cannot_load_quietly),
        ("device is core's and survives a rebuild",
         device_is_cores_and_survives_a_rebuild),
    ]
    ok = True
    for label, case in cases:
        try:
            note = case()
        except Exception as exc:
            ok = False
            print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {label}" + (f": {note}" if note else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
