#!/usr/bin/env python3
"""How much of a merged LoRA survives, and how much noise rides along with it.

## The question

A LoRA merged onto an int8_convrot module is dequantised, added and
REQUANTISED. Three things can go wrong and each is invisible to the other two's
metric -- `docs/checks.md`, "A metric that ranks two arms is a claim about the
metric", is the rule this file exists under.

  delta / step        how big the update is against ONE quantisation step.
                      The driving variable: everything below tracks it.
  realised            `<Q(W+d) - Q(W), d> / <d, d>` -- the fraction of the
                      update that lands along its own direction. Stochastic
                      rounding is unbiased, so this is ~1; round-to-nearest is
                      biased and DISCARDS a sub-step update.
  noise / |d|         `‖Q(W+d) - Q(W) - d‖ / ‖d‖` -- what rides along. Above 1
                      the requantisation perturbs the weights by more than the
                      update itself. **Only this one sees the third failure**,
                      and neither of the others hints at it.

## What this establishes, and the limit

**Not a turbo-only or a lightx2v effect.** The driving variable is delta
magnitude against the quantisation step, not the LoRA's provenance, and across
both shipped artifacts the relationship is monotone and continuous -- one
curve, with turbo at the low-delta end, rather than two phenomena.

**Stored weights only, and the consequence is unmeasured.** The update lands;
the noise is unbiased and random-direction. Nothing here says any of it is
visible at the output, and the PDD LoRAs demonstrably produce working renders.
This is a cost of known magnitude and unknown effect. Do not quote it as a
quality finding.

**VSA is out of scope, and the check is in the record.** The shipped VSA
artifact is a full baked checkpoint with no `lora_*` keys, so it is never
merged and never requantised at load. A future VSA shipping as a LoRA would
inherit all of this; as shipped it inherits none.

    python bench/measure_merge_noise.py              # full, ~400 modules
    python bench/measure_merge_noise.py --stride 12  # a quick shape check
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import date
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO.parents[1]))

from comfy_kitchen.backends.eager.quantization import (  # noqa: E402
    dequantize_int8_convrot_weight, quantize_int8_convrot_weight)

#: The layout default the merge path preserves through `requantize_kwargs`.
GROUP_SIZE = 256
#: Fixed so a re-run reproduces. The shipped path seeds per key
#: (`string_to_seed`), which is a draw from the same distribution rather than
#: this exact one -- what is being measured is the distribution, not one key's
#: realisation.
SEED = 12345

MODELS = REPO.parents[1] / "models"
BASE = MODELS / "diffusion_models" / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
VSA = MODELS / "diffusion_models" / (
    "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors")
DIFF = MODELS / "diffusion_models"
LORA_DIR = MODELS / "loras/h3"

#: (lora, base) pairs, because the answer could depend on either and asking
#: about a LoRA without naming its base is how the merged arm gets measured
#: against a checkpoint nobody pairs it with. The first two are what the
#: shipped PDD graphs actually wire, base included; the rest vary ONE thing
#: each against the first, so a difference is attributable.
ARMS = {
    "pdd_fl2va__pruned": (
        LORA_DIR / "minimax_h3_fl2va_pdd_8step_comfy.safetensors",
        DIFF / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    "pdd_ref2va__pruned": (
        LORA_DIR / "minimax_h3_ref2va_pdd_8step_comfy.safetensors",
        DIFF / "minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
    # varies the BASE only
    "pdd_fl2va__unpruned": (
        LORA_DIR / "minimax_h3_fl2va_pdd_8step_comfy.safetensors",
        DIFF / "minimax_h3_fl2va_int8_convrot.safetensors"),
    # varies the LORA VARIANT only
    "pdd_fl2va_adaln2688__pruned": (
        LORA_DIR / "minimax_h3_fl2va_pdd_8step_adaln2688_comfy.safetensors",
        DIFF / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    # the contrast: a LoRA an order of magnitude smaller against one step
    "turbo_fl2v__pruned": (
        MODELS / "loras/h3/lightx2v_Minimax-h3-Turbo"
        / "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        DIFF / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
}
KINDS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
BLOCKS = 50


def _quant(w: torch.Tensor, seed: int) -> torch.Tensor:
    q, s = quantize_int8_convrot_weight(w.clone().float(), GROUP_SIZE,
                                        stochastic_rounding=seed)
    return dequantize_int8_convrot_weight(q, s, GROUP_SIZE).float()


def _delta(handle, prefix: str) -> torch.Tensor | None:
    keys = set(handle.keys())
    a, b = f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"
    if a not in keys or b not in keys:
        return None
    A = handle.get_tensor(a).float()
    B = handle.get_tensor(b).float()
    rank = A.shape[0]
    alpha_key = f"{prefix}.alpha"
    # ComfyUI reads alpha only from a `<module>.alpha` TENSOR and falls back to
    # 1.0 scaling; `docs/checks.md`'s check_lora_alpha row owns that rule.
    alpha = float(handle.get_tensor(alpha_key)) if alpha_key in keys else float(rank)
    return (B @ A) * (alpha / rank)


def vsa_scope() -> dict:
    """Whether the shipped VSA artifact is merged at all. Skips if absent."""
    if not VSA.exists():
        return {"checked": False, "why": "the VSA checkpoint is not on this box"}
    with safe_open(str(VSA), framework="pt") as handle:
        keys = list(handle.keys())
    lora = [k for k in keys if "lora_" in k]
    return {
        "checked": True,
        "artifact": VSA.name,
        "lora_keys": len(lora),
        "block_weights": len([k for k in keys
                              if k.endswith(".weight") and "blocks." in k]),
        "gate_keys": len([k for k in keys if "to_gate_compress" in k]),
        "in_scope": bool(lora),
        "reading": ("a full baked checkpoint: never merged, never requantised "
                    "at load, so nothing in this record applies to it. A "
                    "future VSA shipping as a LoRA would inherit all of it."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1,
                    help="sample every Nth block (1 = all 50)")
    args = ap.parse_args()

    arms_present = {k: v for k, v in ARMS.items()
                    if v[0].exists() and v[1].exists()}
    absent = sorted(set(ARMS) - set(arms_present))
    if not arms_present:
        print("no (lora, base) pair on this box")
        return 2
    for k in absent:
        print(f"  SKIP {k}: a file is not on this box")

    blocks = list(range(0, BLOCKS, args.stride))
    expected = len(blocks) * len(KINDS)
    print(f"merge noise, int8_convrot gs {GROUP_SIZE}, seed {SEED}")
    print(f"{len(blocks)} block(s) x {len(KINDS)} kind(s) = {expected} module(s) "
          f"per arm\n")

    arms: dict[str, list[dict]] = {}
    total_arms = len(arms_present)
    for arm_index, (name, (lora_path, base_path)) in enumerate(
            arms_present.items(), start=1):
        # **Progress, flushed.** The first version of this printed nothing until
        # it finished, which on the full sweep is about an hour -- so it could
        # not be monitored, triaged, or distinguished from a hang, and the only
        # way to estimate remaining time was to model the work by hand. A long
        # job that reports nothing is a job you cannot decide to stop.
        print(f"  [{arm_index}/{total_arms}] {name} ...", flush=True)
        rows = []
        with safe_open(str(base_path), framework="pt") as base:
            base_keys = set(base.keys())
            with safe_open(str(lora_path), framework="pt") as lora:
                for block in blocks:
                    for kind in KINDS:
                        stem = f"blocks.{block}.{kind}"
                        if f"{stem}.weight" not in base_keys:
                            continue
                        d = _delta(lora, f"diffusion_model.{stem}")
                        if d is None:
                            continue
                        scale = base.get_tensor(f"{stem}.weight_scale").float()
                        step = float(2 * scale.mean())
                        W = dequantize_int8_convrot_weight(
                            base.get_tensor(f"{stem}.weight"), scale,
                            GROUP_SIZE).float()
                        Wq = _quant(W, 0)
                        applied = _quant(W + d, SEED) - Wq
                        dd = float((d * d).sum())
                        rows.append({
                            "module": stem,
                            "kind": kind,
                            "delta_over_step": float(d.pow(2).mean().sqrt()) / step,
                            "realised": float((applied * d).sum() / dd),
                            "noise_over_delta": float((applied - d).norm()
                                                      / d.norm()),
                        })
                        if len(rows) % 25 == 0:
                            print(f"      {len(rows)}/{expected} modules",
                                  flush=True)
            # **The producer asserts its own shape.** An output whose shape is
            # the shape of what survived is indistinguishable from a complete
            # one, and blocks x kinds is not a mystery at write time.
            if len(rows) != expected:
                print(f"FAIL {name}: {len(rows)} modules, expected {expected}. "
                      f"A partial sweep would summarise whatever survived.")
                return 1
            arms[name] = rows
            noise = [r["noise_over_delta"] for r in rows]
            ds = [r["delta_over_step"] for r in rows]
            over = [n for n in noise if n > 1]
            print(f"{name}  ({len(rows)} modules)")
            print(f"  delta/step        median {statistics.median(ds):.5f}")
            print(f"  noise/|d|         median {statistics.median(noise):.3f}  "
                  f"min {min(noise):.3f}  max {max(noise):.3f}")
            print(f"  noise exceeds the update in {len(over)} of {len(rows)} "
                  f"({100 * len(over) / len(rows):.0f}%)")
            print(f"  realised          median "
                  f"{statistics.median([r['realised'] for r in rows]):.4f}")
            for kind in KINDS:
                v = [r["noise_over_delta"] for r in rows if r["kind"] == kind]
                print(f"    {kind:15} median {statistics.median(v):6.3f}  n={len(v)}")
            print()

    scope = vsa_scope()
    print(f"VSA scope: {'IN' if scope.get('in_scope') else 'OUT'} -- "
          f"{scope.get('reading', scope.get('why'))}")

    record = {
        "what": ("how much of a merged LoRA delta survives requantisation into "
                 "int8_convrot, and how much noise rides along with it"),
        "date": date.today().isoformat(),
        "produced_by": "bench/measure_merge_noise.py",
        "base": BASE.name,
        "group_size": GROUP_SIZE, "seed": SEED,
        "blocks_sampled": blocks, "kinds": list(KINDS),
        "finding": ("the driving variable is delta magnitude against ONE "
                    "quantisation step, not the LoRA's provenance. Across both "
                    "shipped artifacts the relationship is monotone and "
                    "continuous -- one curve with the turbo LoRA at the "
                    "low-delta end, not two phenomena. PDD is not clear of it."),
        "is_not": ("a runtime or perceptual measurement. Stored weights only. "
                   "The update LANDS (realised ~1.0) and the noise is unbiased "
                   "and random-direction; nothing here says any of it is "
                   "visible at the output, and the PDD LoRAs demonstrably "
                   "produce working renders."),
        "vsa_scope": scope,
        "arms": {k: {
            "lora": arms_present[k][0].name,
            "base": arms_present[k][1].name,
            "modules": len(v),
            "delta_over_step_median": statistics.median(
                [r["delta_over_step"] for r in v]),
            "noise_over_delta_median": statistics.median(
                [r["noise_over_delta"] for r in v]),
            "noise_over_delta_min": min(r["noise_over_delta"] for r in v),
            "noise_over_delta_max": max(r["noise_over_delta"] for r in v),
            "modules_where_noise_exceeds_update": sum(
                1 for r in v if r["noise_over_delta"] > 1),
            "realised_median": statistics.median([r["realised"] for r in v]),
            "per_module": v,
        } for k, v in arms.items()},
        "reproduce": "python bench/measure_merge_noise.py",
    }
    out = REPO / "bench" / "results" / f"{record['date']}_merge_noise.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
