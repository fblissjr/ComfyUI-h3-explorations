#!/usr/bin/env python3
"""The PDD sidecar's bake contract: exact pairing and the fixed population.

`docs/h3_pdd.md` ("What a backbone bake pins") names mismatches that all
render normally: a full sidecar on a checkpoint whose backbone is already
baked in applies the backbone TWICE; a stripped sidecar on the unbaked base
applies it NEVER; a stripped sidecar on any bake but its own -- another
strength, another LoRA -- applies the WRONG backbone. None is visible in a
file's keys, and a filename is a convention. The contract is therefore by
content and by identity: the converter stores a slice of the paired
checkpoint's int8 codes AND fp32 row scales (`h3_pdd.backbone_probe`,
`h3_pdd.backbone_probe_scale`), and the node requires the loaded module to
EQUAL both before anything is patched.

The population is fixed by the release, not read off the input: 50 blocks
and 2 refiner blocks from `vendor_config.transformer_depth()`, times four
backbone kinds. Every count the converter and the node assert is against
that, never against what they were handed.

This grades the free functions that decide it, in `pdd_lora.py` and
`bench/convert_pdd_lora.py`, on synthetic inputs -- and, when the shipped
artifacts are on disk, on the real key set and the real probe slices. It
imports the production functions rather than restating them, so removing a
refusal from the node turns a case here red.

## History, because the red controls here are the audit's

The first version (2026-09-02) compared codes only and accepted any distance
above a tolerance on a stripped file, and derived its population from the
input: a strength-0.5 bake, a scale-only change, a 49-block source and a
seven-module refiner all passed. Codex's 2026-09-03 audit listed each as a
required red control; each is a named case below. Separately, this file went
red on its very first run against a real defect in `strip_backbone`'s shape
arithmetic, which is why its counts are now parameters from the release.

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
import vendor_config                              # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def raises(fn, needle: str = "") -> tuple[bool, str]:
    """(refused with the needle in the message, message)."""
    try:
        fn()
    except (RuntimeError, SystemExit, ValueError) as e:
        msg = str(e)
        return (needle in msg if needle else True), msg
    return False, "did not raise"


KINDS = P.BACKBONE_KINDS
SLOTS = ("lora_A.weight", "lora_B.weight", "alpha")


def synthetic_sidecar(n_blocks=50, n_refiner=2, kinds=KINDS, stripped=False):
    """The key shape of a converted file. Values are irrelevant to shape."""
    out = {}
    if not stripped:
        for i in range(n_blocks):
            for k in kinds:
                for slot in SLOTS:
                    out[f"diffusion_model.blocks.{i}.{k}.{slot}"] = None
    for i in range(n_refiner):
        for k in kinds:
            for slot in SLOTS:
                out[f"diffusion_model.token_refiner.blocks.{i}.{k}.{slot}"] = None
    for i in range(50):
        out[f"h3_pdd.adaln_baked.blocks.{i}.diff"] = None
        out[f"h3_pdd.adaln_baked.blocks.{i}.diff_b"] = None
    for k in ("h3_pdd.bank.video.weight", "h3_pdd.bank.video.bias",
              "h3_pdd.bank.audio.weight", "h3_pdd.bank.audio.bias",
              "h3_pdd.base_video_out", "h3_pdd.adaln_table",
              "h3_pdd.backbone_probe", "h3_pdd.backbone_probe_scale"):
        out[k] = None
    return out


class _Params:
    def __init__(self, scale):
        self.scale = scale


class _Quantised:
    """The two attributes of `QuantizedTensor` the node reads."""

    def __init__(self, qdata, scale):
        self._qdata = qdata
        self._params = _Params(scale)


class _Node:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def main() -> int:
    print("PDD sidecar contract")

    # --- the release's population, from vendor_config -----------------------
    depth, refiner_depth = vendor_config.transformer_depth()
    pop = P.expected_population()
    check("depth and refiner depth come from the release and agree across "
          "partitions", depth > 0 and refiner_depth > 0
          and pop["blocks"] == depth and pop["refiner_blocks"] == refiner_depth,
          f"{depth} blocks, {refiner_depth} refiner")
    check("node and converter agree on kinds per block",
          len(P.BACKBONE_KINDS) == C.BACKBONE_KINDS_PER_BLOCK
          and C.expected_population() == pop)
    check("population is depth times kinds, not a literal",
          pop["block_modules"] == depth * len(KINDS)
          and pop["refiner_modules"] == refiner_depth * len(KINDS))

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

    # --- file population, as sets against the release ------------------------
    full = synthetic_sidecar()
    check("full: the whole release population passes",
          P.check_file_population("full", full, {"backbone_modules": "208"}, "x")
          == (pop["block_modules"], pop["refiner_modules"]))
    stripped = synthetic_sidecar(stripped=True)
    check("stripped: no backbone and the whole refiner passes",
          P.check_file_population("stripped", stripped,
                                  {"backbone_modules": "8"}, "x")
          == (0, pop["refiner_modules"]))
    ok, msg = raises(lambda: P.check_file_population(
        "full", synthetic_sidecar(n_blocks=49), {}, "x"), "backbone")
    check("49 whole blocks are refused", ok, msg[:70])
    short = synthetic_sidecar()
    for slot in SLOTS:
        del short[f"diffusion_model.token_refiner.blocks.1.mlp.fc2.{slot}"]
    ok, msg = raises(lambda: P.check_file_population("full", short, {}, "x"),
                     "refiner")
    check("one whole refiner module absent is refused", ok, msg[:70])
    ok, msg = raises(lambda: P.check_file_population(
        "full", synthetic_sidecar(kinds=KINDS[:-1]), {}, "x"), "backbone")
    check("one complete kind absent across every block is refused", ok, msg[:70])
    extra = synthetic_sidecar()
    for slot in SLOTS:
        extra[f"diffusion_model.blocks.50.mlp.fc2.{slot}"] = None
    ok, msg = raises(lambda: P.check_file_population("full", extra, {}, "x"),
                     "unexpected")
    check("one extra module is refused", ok, msg[:70])
    swapped = synthetic_sidecar(n_blocks=49)
    for slot in SLOTS:
        for k in KINDS:
            swapped[f"diffusion_model.blocks.50.{k}.{slot}"] = None
    ok, msg = raises(lambda: P.check_file_population("full", swapped, {}, "x"),
                     "backbone")
    check("a missing block replaced by a stray block is refused (count alone "
          "would pass)", ok, msg[:70])
    ok, msg = raises(lambda: P.check_file_population(
        "full", full, {"backbone_modules": "200"}, "x"), "disagree")
    check("metadata disagreeing with the file is refused", ok, msg[:70])
    ok, msg = raises(lambda: P.check_file_population(
        "stripped", full, {}, "x"), "backbone")
    check("a file declared stripped that still carries the backbone is refused",
          ok, msg[:70])

    # --- probe match: exact, codes and scales -----------------------------
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(-127, 128, (64, 4096), generator=g, dtype=torch.int8)
    scale = torch.rand(64, 1, generator=g, dtype=torch.float32) * 0.01
    live_codes = torch.cat([codes, torch.zeros(8, 4096, dtype=torch.int8)])
    live_scale = torch.cat([scale, torch.ones(8, 1)])
    same = P.probe_match(_Quantised(live_codes, live_scale), codes, scale)
    check("same codes and scales match on both",
          same == {"codes": True, "scale": True})
    # A bake at strength 1.0 moves the probed module by its own update, ~0.044
    # relative; built at that size so the separation graded is the real one.
    rms = codes.to(torch.float32).pow(2).mean().sqrt()
    noise = torch.randn(codes.shape, generator=g) * 0.044 * rms
    baked_codes = (codes.to(torch.float32) + noise).round().clamp(-128, 127).to(torch.int8)
    baked_scale = scale * 1.02
    other = P.probe_match(_Quantised(baked_codes, baked_scale), codes, scale)
    check("a bake differs on both codes and scales",
          other == {"codes": False, "scale": False})
    scale_only = P.probe_match(_Quantised(live_codes, live_scale * 1.0001),
                               codes, scale)
    check("scale-only change: codes equal, scale not",
          scale_only == {"codes": True, "scale": False})
    codes_only = P.probe_match(_Quantised(baked_codes, live_scale), codes, scale)
    check("codes-only change: scale equal, codes not",
          codes_only == {"codes": False, "scale": True})
    check("a plain (unquantised) weight is undecidable, not a match",
          P.probe_match(live_codes.to(torch.float16), codes, scale) is None)
    check("a quantised weight with no scale is undecidable",
          P.probe_match(_Quantised(live_codes, None), codes, scale) is None)
    check("a differently shaped weight is undecidable",
          P.probe_match(_Quantised(live_codes[:, :2048], live_scale), codes,
                        scale) is None)
    tree = _Node(blocks=[_Node(mlp=_Node(fc2=_Node(weight="w0"))),
                         _Node(mlp=_Node(fc2=_Node(weight="w1")))])
    check("the probe key walks the module tree",
          P.live_module_weight(tree, "blocks.1.mlp.fc2.weight") == "w1")

    # --- identity decisions: Codex's red controls ---------------------------
    check("full sidecar on its exact base passes",
          P.check_backbone_identity("full", same, "x") is None)
    check("stripped sidecar on its exact bake passes",
          P.check_backbone_identity("stripped", same, "x", base_match=other) is None)
    ok, msg = raises(lambda: P.check_backbone_identity("full", other, "x"),
                     "SECOND time")
    check("full sidecar on a bake is refused as a double apply", ok, msg[:70])
    ok, msg = raises(lambda: P.check_backbone_identity("full", scale_only, "x"),
                     "row scales differ")
    check("full sidecar: codes identical, scale changed, refused and named",
          ok, msg[:70])
    ok, msg = raises(lambda: P.check_backbone_identity("full", codes_only, "x"),
                     "codes differ")
    check("full sidecar: scale identical, codes changed, refused and named",
          ok, msg[:70])
    # Stripped on the base: the paired (bake) probe does not match, the base
    # probe does -- the message must say "base", not "not your bake".
    ok, msg = raises(lambda: P.check_backbone_identity(
        "stripped", other, "x", base_match=same), "unbaked base")
    check("stripped sidecar on the base is refused as the unbaked base", ok,
          msg[:70])
    # Stripped on a strength-0.5 bake, another LoRA's bake, or an unrelated
    # checkpoint: neither probe matches. One case, because to the node they
    # are the same observation.
    ok, msg = raises(lambda: P.check_backbone_identity(
        "stripped", other, "x", base_match=other), "another")
    check("stripped sidecar on a different bake (other strength, other LoRA, "
          "other checkpoint) is refused", ok, msg[:70])
    ok, msg = raises(lambda: P.check_backbone_identity(
        "stripped", scale_only, "x", base_match=other), "row scales differ")
    check("stripped sidecar: codes identical to its bake, scale changed, "
          "refused", ok, msg[:70])
    ok, msg = raises(lambda: P.check_backbone_identity(
        "stripped", codes_only, "x", base_match=other), "codes differ")
    check("stripped sidecar: scale identical to its bake, codes changed, "
          "refused", ok, msg[:70])
    ok, msg = raises(lambda: P.check_backbone_identity("stripped", None, "x"),
                     "refuses")
    check("stripped sidecar on an undecidable weight is refused", ok, msg[:70])
    note = P.check_backbone_identity("full", None, "x")
    check("full sidecar on an undecidable weight warns and proceeds",
          isinstance(note, str) and "unchecked" in note)
    check("the probe a kind must carry is fixed: full=base, stripped=baked",
          P.PROBE_OF_BY_KIND == {"full": "base", "stripped": "baked"})

    # --- what load_lora resolved ----------------------------------------------
    refiner_keys = [f"diffusion_model.token_refiner.blocks.{i}.{k}.weight"
                    for i in range(refiner_depth) for k in KINDS]
    adaln_keys = [f"diffusion_model.blocks.{i}.adaln_proj.linear.{s}"
                  for i in range(depth) for s in ("weight", "bias")]
    P.check_stripped_targets("stripped", refiner_keys + adaln_keys,
                             pop["refiner_modules"], "x")
    check("stripped: refiner plus adaln resolved is the expected shape", True)
    ok, msg = raises(lambda: P.check_stripped_targets(
        "stripped", refiner_keys + adaln_keys
        + ["diffusion_model.blocks.3.mlp.fc1.weight"], pop["refiner_modules"],
        "x"), "twice")
    check("stripped: one backbone target resolved is refused", ok, msg[:70])
    ok, msg = raises(lambda: P.check_stripped_targets(
        "stripped", refiner_keys[:-1] + adaln_keys, pop["refiner_modules"],
        "x"), "refiner")
    check("stripped: seven refiner modules resolved against the release's "
          "eight is refused", ok, msg[:70])
    P.check_stripped_targets("full", ["diffusion_model.blocks.3.mlp.fc1.weight"],
                             pop["refiner_modules"], "x")
    check("full: backbone targets are the merged path's business", True)
    check("adaln keys are classified apart from the backbone",
          P.loaded_targets(adaln_keys[:2] + refiner_keys[:1]
                           + ["diffusion_model.blocks.0.mlp.fc2.weight"])
          == {"backbone": ["diffusion_model.blocks.0.mlp.fc2.weight"],
              "refiner": refiner_keys[:1], "adaln": adaln_keys[:2],
              "other": []})

    # --- the converter: population and strip against the release -------------
    C.assert_population(depth, refiner_depth, pop["block_modules"],
                        pop["refiner_modules"])
    check("converter accepts the release population", True)
    ok, msg = raises(lambda: C.assert_population(
        49, refiner_depth, 49 * len(KINDS), pop["refiner_modules"]), "blocks 49")
    check("converter refuses a coherent 49-block source (196 modules)", ok,
          msg[:70])
    ok, msg = raises(lambda: C.assert_population(
        depth, refiner_depth, pop["block_modules"], pop["refiner_modules"] - 1),
        "refiner_modules")
    check("converter refuses seven refiner modules", ok, msg[:70])
    sd = synthetic_sidecar()
    before = len(sd)
    removed = C.strip_backbone(sd, pop["block_modules"], pop["refiner_modules"])
    check("strip removes exactly the backbone tensors, three per module",
          removed == pop["block_modules"] * 3 and len(sd) == before - removed)
    check("strip leaves the refiner's tensors and every h3_pdd.* tensor",
          sum(k.startswith("diffusion_model.") for k in sd)
          == pop["refiner_modules"] * 3
          and all(k in sd for k in synthetic_sidecar() if k.startswith("h3_pdd.")))
    short49 = synthetic_sidecar(n_blocks=49)
    ok, msg = raises(lambda: C.strip_backbone(
        short49, pop["block_modules"], pop["refiner_modules"]), "expected")
    check("strip against the release's count refuses a 49-block dict", ok,
          msg[:70])
    stray = synthetic_sidecar()
    stray["diffusion_model.final_layer.video_out.lora_A.weight"] = None
    ok, msg = raises(lambda: C.strip_backbone(
        stray, pop["block_modules"], pop["refiner_modules"]), "remain")
    check("strip refuses a stray diffusion_model.* tensor", ok, msg[:70])

    # --- real artifacts, when present ---------------------------------------
    models = HERE.parents[2] / "models"
    pruned = models / "diffusion_models" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    base = models / "diffusion_models" / "minimax_h3_fl2va_int8_convrot.safetensors"
    sidecar = models / "loras" / "h3" / "minimax_h3_fl2va_pdd_8step_comfy.safetensors"
    if pruned.exists() and base.exists():
        pc, ps = C.backbone_probe(pruned)
        bc, bs = C.backbone_probe(base)
        check("real: pruned and unpruned int8 agree bit-for-bit on codes AND "
              "scales of the probe slice",
              torch.equal(pc, bc) and torch.equal(ps, bs),
              f"{tuple(pc.shape)} {pc.dtype}, {tuple(ps.shape)} {ps.dtype}")
        check("real: the probe against its own checkpoint matches on both",
              P.probe_match(_Quantised(pc, ps), pc, ps)
              == {"codes": True, "scale": True})
        check("real: the same codes with the scale nudged by one ulp do not",
              P.probe_match(_Quantised(pc, torch.nextafter(ps, ps + 1)), pc, ps)
              == {"codes": True, "scale": False})
    else:
        print("  SKIP  real probe cases: int8 checkpoints not on disk")
    if sidecar.exists():
        from safetensors import safe_open
        with safe_open(sidecar, framework="pt") as f:
            keys = {k: None for k in f.keys()}
            meta = f.metadata() or {}
        check("real: the shipped fl2va sidecar is the release population, "
              "and its metadata agrees",
              P.check_file_population("full", keys, meta, sidecar.name)
              == (pop["block_modules"], pop["refiner_modules"]))
        n = sum(k.startswith("diffusion_model.blocks.") for k in keys)
        removed = C.strip_backbone(keys, pop["block_modules"],
                                   pop["refiner_modules"])
        check("real: the shipped sidecar strips to the refiner's tensors",
              removed == n == pop["block_modules"] * 3
              and sum(k.startswith("diffusion_model.") for k in keys)
              == pop["refiner_modules"] * 3)
    else:
        print("  SKIP  real sidecar case: shipped fl2va sidecar not on disk")

    # --- the real baked pair, when it is on disk ------------------------------
    # The first bake was cut 2026-09-05 (docs/research/pdd/2026-09-05_bake_plan.md).
    # The checkpoint lives outside the repo; it is reached through the
    # gitignored `internal/h3_bakes` link until it is registered under
    # models/diffusion_models, and either location counts here.
    bake_name = "minimax_h3_fl2va_pruned_int8_convrot_pdd8_baked_s1.safetensors"
    bake_candidates = [models / "diffusion_models" / bake_name,
                       HERE.parent / "internal" / "h3_bakes" / bake_name]
    bake = next((p for p in bake_candidates if p.exists()), None)
    stripped_side = models / "loras" / "h3" / "minimax_h3_fl2va_pdd_8step_stripped_comfy.safetensors"
    if bake is not None and stripped_side.exists() and pruned.exists() and sidecar.exists():
        from safetensors import safe_open
        with safe_open(stripped_side, framework="pt") as f:
            s_keys = {k: None for k in f.keys()}
            s_meta = f.metadata() or {}
            s_probe = {k: f.get_tensor(k) for k in s_keys
                       if k.startswith("h3_pdd.backbone_probe")}
        with safe_open(sidecar, framework="pt") as f:
            f_probe = {k: f.get_tensor(k) for k in f.keys()
                       if k.startswith("h3_pdd.backbone_probe")}
        kc, ks_ = C.backbone_probe(bake)
        pc, ps = C.backbone_probe(pruned)
        live_bake, live_base = _Quantised(kc, ks_), _Quantised(pc, ps)
        check("real bake: the stripped sidecar declares stripped, strength 1.0, "
              "and names its bake",
              P.sidecar_backbone_kind(s_meta) == "stripped"
              and s_meta.get("h3_pdd_backbone_strength_baked") == "1.0"
              and s_meta.get("h3_pdd_baked_checkpoint") == bake.name)
        check("real bake: the stripped sidecar is refiner-only at the release "
              "population",
              P.check_file_population("stripped", s_keys, s_meta, stripped_side.name)
              == (0, pop["refiner_modules"]))
        pm = lambda live, t, pref: P.probe_match(live, t[pref], t[pref + "_scale"])  # noqa: E731
        check("real bake: the stripped probe IS the bake and is NOT the base",
              pm(live_bake, s_probe, "h3_pdd.backbone_probe") == {"codes": True, "scale": True}
              and pm(live_base, s_probe, "h3_pdd.backbone_probe") == {"codes": False, "scale": False})
        check("real bake: the stripped file's base probe IS the base and is NOT the bake",
              pm(live_base, s_probe, "h3_pdd.backbone_probe_base") == {"codes": True, "scale": True}
              and pm(live_bake, s_probe, "h3_pdd.backbone_probe_base") == {"codes": False, "scale": False})
        check("real bake: the full sidecar's probe is NOT the bake",
              pm(live_bake, f_probe, "h3_pdd.backbone_probe") == {"codes": False, "scale": False})
        check("real bake: stripped on its bake passes identity",
              P.check_backbone_identity(
                  "stripped", pm(live_bake, s_probe, "h3_pdd.backbone_probe"), "s",
                  base_match=pm(live_bake, s_probe, "h3_pdd.backbone_probe_base")) is None)
        ok, msg = raises(lambda: P.check_backbone_identity(
            "stripped", pm(live_base, s_probe, "h3_pdd.backbone_probe"), "s",
            base_match=pm(live_base, s_probe, "h3_pdd.backbone_probe_base")), "unbaked base")
        check("real bake: stripped on the base is refused as the unbaked base", ok, msg[:70])
        ok, msg = raises(lambda: P.check_backbone_identity(
            "full", pm(live_bake, f_probe, "h3_pdd.backbone_probe"), "f"), "SECOND time")
        check("real bake: full sidecar on the bake is refused as a double apply", ok, msg[:70])
        with safe_open(bake, framework="pt") as f:
            b_meta = f.metadata() or {}
            n_i8 = sum(1 for k in f.keys()
                       if k.endswith(".weight") and f.get_slice(k).get_dtype() == "I8")
        check("real bake: the checkpoint carries exactly the release's int8 "
              "backbone population and names its LoRA, strength and rounding",
              n_i8 == pop["block_modules"]
              and b_meta.get("h3_bake_lora") == sidecar.name
              and b_meta.get("h3_bake_strength") == "1.0"
              and b_meta.get("h3_bake_rounding") == "round_to_nearest",
              f"{n_i8} int8 modules")
    else:
        print("  SKIP  real baked-pair cases: bake or stripped sidecar not on disk")

    if FAILURES:
        print(f"RED: {len(FAILURES)} failed -- {', '.join(FAILURES)}")
        return 1
    print("  ok    every case passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
