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
geometry = importlib.import_module(f"{REPO.name}.reference_geometry")
vendor_config = importlib.import_module(f"{REPO.name}.vendor_config")

ENCODER = COMFY / "models" / "text_encoders" / "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"


def _with_mutated_state_dict(mutate):
    """Run one load with `mutate` applied to the state dict core just read."""
    original = comfy.utils.load_torch_file

    def reading(*args, **kwargs):
        out = original(*args, **kwargs)
        # `return_metadata=True` makes this a (state_dict, metadata) pair.
        mutate(out[0] if isinstance(out, tuple) else out)
        return out

    comfy.utils.load_torch_file = reading
    try:
        return loader.load_guarded_clip(str(ENCODER), None)
    finally:
        comfy.utils.load_torch_file = original


def contract_is_derived_from_comfy_not_declared():
    """Every contract value is read out of ComfyUI, and agrees with the release."""
    contract = loader.native_encoder_contract()
    assert set(contract) == set(geometry.ENCODER_CONTRACT_KEYS), sorted(contract)
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


def the_shipped_encoder_passes_every_guard():
    """The control. A guard that refuses the real artifact is worse than none."""
    started = time.monotonic()
    clip = loader.load_guarded_clip(str(ENCODER), None)
    contract = geometry.encoder_contract_from_clip(clip)
    assert contract is not None, "the loader did not stamp a contract"
    assert contract == loader.native_encoder_contract(), contract
    # The point of stamping: `encoder` stops being silently downgraded.
    assert geometry.effective_policy("encoder", contract) == "encoder"
    assert geometry.effective_policy("encoder", None) == "comfy"
    return f"loaded and verified in {time.monotonic() - started:.1f}s"


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
        ("carries an unexpected tensor",
         lambda sd: sd.__setitem__("model.layers.0.not_a_real_tensor",
                                   __import__("torch").zeros(4)),
         "does not exactly populate"),
    )
    for label, mutate, expected in cases:
        try:
            _with_mutated_state_dict(mutate)
        except ValueError as exc:
            assert expected in str(exc), (label, str(exc)[:200])
        else:
            raise AssertionError(f"a checkpoint that {label} loaded green")
    return f"{len(cases)} mutations red, each with its own message"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    if not ENCODER.exists():
        print(f"SKIP  {ENCODER.name} is not installed")
        return 2

    cases = [
        ("contract derived from comfy", contract_is_derived_from_comfy_not_declared),
        ("special-token arithmetic", token_ids_refuse_a_malformed_declaration),
        ("shipped encoder passes", the_shipped_encoder_passes_every_guard),
        ("incomplete checkpoints red", an_incomplete_checkpoint_cannot_load_quietly),
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
