#!/usr/bin/env python3
"""The PDD sidecar's bake contract: what a stripped sidecar refuses, what a
full one refuses, and that the probe can tell the two checkpoints apart.

`docs/h3_pdd.md` ("What a backbone bake pins") names two mismatches that both
render normally: a full sidecar on a checkpoint whose backbone is already
baked in applies the backbone TWICE, and a stripped sidecar on the unbaked
base applies it NEVER. Neither is visible in a file's keys, and a filename is
a convention. The contract is therefore by content: the converter stores a
slice of the base's own int8 weight (`h3_pdd.backbone_probe`) and the node
compares it with the loaded module before anything is patched.

This grades the free functions that decide it, in `pdd_lora.py` and
`bench/convert_pdd_lora.py`, on synthetic inputs -- and, when the shipped
artifacts are on disk, on the real key set and the real probe slice. It
imports the production functions rather than restating them, so removing a
refusal from the node turns a case here red.

What it does NOT cover: the node's `execute` end to end. That needs a loaded
H3 and a baked checkpoint, and the second does not exist yet. The functions
graded here are the whole of the decision; `execute` only routes their
inputs, and `docs/checks.md` records that as the gap.

    uv run --active --no-sync python bench/check_pdd_sidecar_contract.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))          # ComfyUI root
sys.path.insert(0, str(HERE.parent))              # this repo

import pdd_lora as P                              # noqa: E402
import convert_pdd_lora as C                      # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def raises(fn, needle: str = "") -> tuple[bool, str]:
    """(refused, message). `needle` must appear in the message when given."""
    try:
        fn()
    except (RuntimeError, SystemExit) as e:
        msg = str(e)
        return (needle in msg if needle else True), msg
    return False, "did not raise"


def synthetic_sidecar(n_blocks=50, n_refiner=2, kinds=("attn.qkv_proj",
                                                       "attn.out_proj",
                                                       "mlp.fc1", "mlp.fc2")):
    """The key shape of a converted file. Values are irrelevant to stripping."""
    out = {}
    for i in range(n_blocks):
        for k in kinds:
            for slot in ("lora_A.weight", "lora_B.weight", "alpha"):
                out[f"diffusion_model.blocks.{i}.{k}.{slot}"] = None
    for i in range(n_refiner):
        for k in kinds:
            for slot in ("lora_A.weight", "lora_B.weight", "alpha"):
                out[f"diffusion_model.token_refiner.blocks.{i}.{k}.{slot}"] = None
    for i in range(n_blocks):
        out[f"h3_pdd.adaln_baked.blocks.{i}.diff"] = None
        out[f"h3_pdd.adaln_baked.blocks.{i}.diff_b"] = None
    for k in ("h3_pdd.bank.video.weight", "h3_pdd.bank.video.bias",
              "h3_pdd.bank.audio.weight", "h3_pdd.bank.audio.bias",
              "h3_pdd.base_video_out", "h3_pdd.adaln_table",
              "h3_pdd.backbone_probe"):
        out[k] = None
    return out


class _Quantised:
    """The one attribute of `QuantizedTensor` the node reads."""

    def __init__(self, qdata):
        self._qdata = qdata


class _Node:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def main() -> int:
    print("PDD sidecar contract")

    # --- kind -------------------------------------------------------------
    check("absent metadata is a full sidecar",
          P.sidecar_backbone_kind({}) == "full")
    check("declared stripped is stripped",
          P.sidecar_backbone_kind({"h3_pdd_backbone": "stripped"}) == "stripped")
    ok, msg = raises(lambda: P.sidecar_backbone_kind({"h3_pdd_backbone": "half"}),
                     "not one of")
    check("an unknown kind is refused, not defaulted", ok, msg[:60])

    # --- stripped contract, before the model is touched -------------------
    meta_s = {"h3_pdd_backbone": "stripped",
              "h3_pdd_backbone_strength_baked": "1.0"}
    P.check_stripped_contract("stripped", meta_s, 1.0, "", "x")
    check("stripped at the baked strength with no un-merge passes", True)
    P.check_stripped_contract("full", {}, 0.5, "0-49", "x")
    check("full sidecar is unconstrained by the stripped contract", True)
    ok, msg = raises(lambda: P.check_stripped_contract(
        "stripped", meta_s, 0.5, "", "x"), "strength")
    check("stripped at another strength is refused", ok, msg[:70])
    ok, msg = raises(lambda: P.check_stripped_contract(
        "stripped", meta_s, 1.0, "0-2", "x"), "unmerged_blocks")
    check("stripped with unmerged_blocks is refused", ok, msg[:70])
    ok, msg = raises(lambda: P.check_stripped_contract(
        "stripped", {"h3_pdd_backbone": "stripped"}, 1.0, "", "x"),
        "strength_baked")
    check("stripped with no declared baked strength is refused", ok, msg[:70])

    # --- probe distance -----------------------------------------------------
    g = torch.Generator().manual_seed(0)
    probe = torch.randint(-127, 128, (64, 4096), generator=g,
                          dtype=torch.int8)
    live = torch.cat([probe, torch.zeros(8, 4096, dtype=torch.int8)])
    check("same int8 weight is exactly 0.0",
          P.backbone_probe_distance(live, probe) == 0.0)
    check("a QuantizedTensor-shaped weight is read through _qdata",
          P.backbone_probe_distance(_Quantised(live), probe) == 0.0)
    # A bake at strength 1.0 moves the probed module by its own update, ~0.044
    # relative. Built at that size so the separation graded is the one that
    # exists, not a convenient one.
    rms = probe.to(torch.float32).pow(2).mean().sqrt()
    noise = torch.randn(probe.shape, generator=g) * 0.044 * rms
    baked = (probe.to(torch.float32) + noise).round().clamp(-128, 127)
    baked = baked.to(torch.int8)
    d_bake = P.backbone_probe_distance(baked, probe)
    check("a bake-sized perturbation sits well above the tolerance",
          d_bake > 10 * P.BACKBONE_PROBE_TOLERANCE, f"{d_bake:.4f}")
    check("an fp8/other-dtype weight is undecidable, not a match",
          P.backbone_probe_distance(live.to(torch.float16), probe) is None)
    check("a differently shaped weight is undecidable",
          P.backbone_probe_distance(live[:, :2048], probe) is None)
    check("a weight shorter than the probe is undecidable",
          P.backbone_probe_distance(live[:32], probe) is None)

    tree = _Node(blocks=[_Node(mlp=_Node(fc2=_Node(weight="w0"))),
                         _Node(mlp=_Node(fc2=_Node(weight="w1")))])
    check("the probe key walks the module tree",
          P.live_module_weight(tree, "blocks.1.mlp.fc2.weight") == "w1")

    # --- pairing decisions -------------------------------------------------
    check("full sidecar on its base passes",
          P.check_backbone_pairing("full", 0.0, "x") is None)
    check("stripped sidecar on a bake passes",
          P.check_backbone_pairing("stripped", d_bake, "x") is None)
    ok, msg = raises(lambda: P.check_backbone_pairing("full", d_bake, "x"),
                     "SECOND time")
    check("full sidecar on a bake is refused as a double apply", ok, msg[:70])
    ok, msg = raises(lambda: P.check_backbone_pairing("stripped", 0.0, "x"),
                     "unbaked base")
    check("stripped sidecar on the base is refused as a never-apply", ok,
          msg[:70])
    ok, msg = raises(lambda: P.check_backbone_pairing("stripped", None, "x"),
                     "refuses")
    check("stripped sidecar on an undecidable weight is refused", ok, msg[:70])
    note = P.check_backbone_pairing("full", None, "x")
    check("full sidecar on an undecidable weight warns and proceeds",
          isinstance(note, str) and "unchecked" in note)
    # The tolerance's other side: a value at the tolerance is a match. The
    # premise is that the same int8 file compares at exactly 0.0, so anything
    # nonzero but tiny would be a loader transform this repo has not seen.
    check("tolerance boundary: at the tolerance is still a match",
          P.check_backbone_pairing("full", P.BACKBONE_PROBE_TOLERANCE, "x")
          is None)

    # --- what load_lora resolved ----------------------------------------------
    refiner_keys = [f"diffusion_model.token_refiner.blocks.{i}.{k}.weight"
                    for i in range(2) for k in ("attn.qkv_proj", "attn.out_proj",
                                                "mlp.fc1", "mlp.fc2")]
    adaln_keys = [f"diffusion_model.blocks.{i}.adaln_proj.linear.{s}"
                  for i in range(50) for s in ("weight", "bias")]
    P.check_stripped_targets("stripped", refiner_keys + adaln_keys, 8, "x")
    check("stripped: refiner plus adaln resolved is the expected shape", True)
    ok, msg = raises(lambda: P.check_stripped_targets(
        "stripped", refiner_keys + adaln_keys
        + ["diffusion_model.blocks.3.mlp.fc1.weight"], 8, "x"), "twice")
    check("stripped: one backbone target resolved is refused", ok, msg[:70])
    ok, msg = raises(lambda: P.check_stripped_targets(
        "stripped", refiner_keys[:7] + adaln_keys, 8, "x"), "refiner")
    check("stripped: a short refiner match is refused", ok, msg[:70])
    P.check_stripped_targets("full", ["diffusion_model.blocks.3.mlp.fc1.weight"],
                             8, "x")
    check("full: backbone targets are the merged path's business", True)
    sd = synthetic_sidecar()
    check("refiner modules are counted off the file's keys, not metadata",
          P.refiner_modules_in(sd) == 8)
    check("adaln keys are classified apart from the backbone",
          P.loaded_targets(adaln_keys[:2] + refiner_keys[:1]
                           + ["diffusion_model.blocks.0.mlp.fc2.weight"])
          == {"backbone": ["diffusion_model.blocks.0.mlp.fc2.weight"],
              "refiner": refiner_keys[:1], "adaln": adaln_keys[:2],
              "other": []})

    # --- the converter's strip ---------------------------------------------
    sd = synthetic_sidecar()
    before = len(sd)
    removed = C.strip_backbone(sd, 200, 8)
    check("strip removes exactly the 600 backbone tensors",
          removed == 600 and len(sd) == before - 600)
    check("strip leaves the refiner's 24 and every h3_pdd.* tensor",
          sum(k.startswith("diffusion_model.") for k in sd) == 24
          and all(k in sd for k in synthetic_sidecar() if k.startswith("h3_pdd.")))
    check("a stripped dict has no backbone left to count",
          P.loaded_targets([k.rsplit(".", 2)[0] + ".weight" for k in sd
                            if k.endswith(".lora_A.weight")])["backbone"] == [])
    short = synthetic_sidecar()
    del short["diffusion_model.blocks.7.mlp.fc2.alpha"]
    ok, msg = raises(lambda: C.strip_backbone(short, 200, 8), "expected")
    check("strip refuses a dict already short a tensor", ok, msg[:70])
    stray = synthetic_sidecar()
    stray["diffusion_model.final_layer.video_out.lora_A.weight"] = None
    ok, msg = raises(lambda: C.strip_backbone(stray, 200, 8), "remain")
    check("strip refuses a stray diffusion_model.* tensor it did not expect",
          ok, msg[:70])

    # --- real artifacts, when present ---------------------------------------
    models = HERE.parents[2] / "models"
    pruned = models / "diffusion_models" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    base = models / "diffusion_models" / "minimax_h3_fl2va_int8_convrot.safetensors"
    sidecar = models / "loras" / "h3" / "minimax_h3_fl2va_pdd_8step_comfy.safetensors"
    if pruned.exists() and base.exists():
        a = C.backbone_probe(pruned)
        b = C.backbone_probe(base)
        check("real: pruned and unpruned int8 agree bit-for-bit on the probe "
              "slice", torch.equal(a, b), f"{tuple(a.shape)} {a.dtype}")
        check("real: the probe against its own checkpoint is exactly 0.0",
              P.backbone_probe_distance(a, a) == 0.0)
    else:
        print("  SKIP  real probe cases: int8 checkpoints not on disk")
    if sidecar.exists():
        from safetensors import safe_open
        with safe_open(sidecar, framework="pt") as f:
            keys = {k: None for k in f.keys()}
        n = sum(k.startswith("diffusion_model.blocks.") for k in keys)
        removed = C.strip_backbone(keys, 200, 8)
        check("real: the shipped fl2va sidecar strips to the refiner's 24",
              removed == n == 600
              and sum(k.startswith("diffusion_model.") for k in keys) == 24)
        check("real: the shipped sidecar carries 8 refiner modules",
              P.refiner_modules_in(keys) == 8)
    else:
        print("  SKIP  real sidecar case: shipped fl2va sidecar not on disk")

    if FAILURES:
        print(f"RED: {len(FAILURES)} failed -- {', '.join(FAILURES)}")
        return 1
    print("  ok    every case passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
