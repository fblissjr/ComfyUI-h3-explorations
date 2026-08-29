#!/usr/bin/env python3
"""How far apart the fl2va and ref2va partitions actually are, component by component.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). Needs no server and
no GPU; streams safetensors a tensor at a time and never holds two checkpoints.

**Not a check.** It asserts nothing and grades nothing. It answers one question
`docs/research/official_weights_metadata.md` raises but does not measure: that
document records the partition split as something the release *declares* -- two
`model_index.json` blocks with different task lists -- and this measures what
the split is worth in the weights.

## Why the answer is not one number

The partitions are close at the output projection and far in what was learned
on top of it, and reporting either alone is misleading:

  * `final_layer.video_out` -- the tensor the PDD partition guard fingerprints
    -- sits a few percent apart. That is the whole basis of the guard, and it
    is an order of magnitude above what a dtype cast moves.
  * The PDD LoRA's own backbone deltas are near ORTHOGONAL between partitions.
    Cosine near zero, relative distance above 1. Two distillations that found
    unrelated corrections for two nearly-identical trunks.

So a wrong-partition load is not graceful degradation. The heads are nearly
right, which is why the render looks structurally normal; the learned
correction underneath is a full-magnitude vector pointing somewhere else.

**The guard measures the wrong thing and works anyway**, which is worth saying
out loud: it fingerprints the component that differs LEAST, as a proxy for a
swap whose damage is in the components that differ MOST. That is fine -- they
are perfectly correlated, since both follow from the file being the other
partition -- but nobody should read the fingerprint distance as the size of the
error it prevents.

## What it does not measure

Quantized tensors are skipped. `int8_convrot` stores a rotated representation
and comparing two files through it needs the rotation undone; every conclusion
here rests on the unquantized components, which is where the interesting
structure is anyway. The set skipped is reported so the omission is visible.

Usage:

    python bench/compare_h3_partitions.py
    python bench/compare_h3_partitions.py --out bench/results/DATE_h3_partition_distance.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

COMFY = Path.home() / "ComfyUI"
DIT = COMFY / "models" / "diffusion_models"
LORA = COMFY / "models" / "loras" / "h3"

#: The two artifacts per partition this compares. Named rather than globbed so
#: a new file in the directory cannot silently change what was measured.
PAIRS = {
    "base_checkpoint": (DIT / "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                        DIT / "minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
    "pdd_lora": (LORA / "minimax_h3_fl2va_pdd_8step_comfy.safetensors",
                 LORA / "minimax_h3_ref2va_pdd_8step_comfy.safetensors"),
}

#: Quantized payloads and their sidecars. Skipped, and reported as skipped --
#: see the module docstring.
QUANT_SUFFIX = ("weight_scale", "comfy_quant")


def is_quantized(handle, key: str) -> bool:
    return key.endswith(QUANT_SUFFIX) or handle.get_slice(key).get_dtype() in ("I8", "U8")


def distance(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Relative Frobenius distance and cosine, both in float64.

    Cosine is the one that separates "smaller version of the same thing" from
    "pointing somewhere else", and it is the reason this reports both: a
    relative distance near 1.4 with cosine near 0 is two orthogonal vectors of
    similar norm, which reads as catastrophic on distance alone and is better
    described as unrelated.
    """
    a = a.to(torch.float64).flatten()
    b = b.to(torch.float64).flatten()
    nb = float(b.norm())
    return {
        "relative": float((a - b).norm() / nb) if nb else None,
        "cosine": float((a @ b) / (a.norm() * b.norm())) if nb and float(a.norm()) else None,
        "norm_fl2va": float(a.norm()),
        "norm_ref2va": nb,
    }


def collapse(key: str) -> str:
    return re.sub(r"\.\d+\.", ".N.", key)


def compare(fl: Path, rf: Path) -> dict:
    """Every shared unquantized tensor, plus the key-set relationship."""
    with safe_open(fl, framework="pt") as a, safe_open(rf, framework="pt") as b:
        ka, kb = set(a.keys()), set(b.keys())
        shared = sorted(ka & kb)
        per_key, skipped = {}, []
        for k in shared:
            if is_quantized(a, k) or is_quantized(b, k):
                skipped.append(k)
                continue
            if a.get_slice(k).get_shape() != b.get_slice(k).get_shape():
                per_key[k] = {"shape_mismatch": [a.get_slice(k).get_shape(),
                                                 b.get_slice(k).get_shape()]}
                continue
            per_key[k] = distance(a.get_tensor(k), b.get_tensor(k))
        return {
            "files": {"fl2va": fl.name, "ref2va": rf.name},
            "keys": {
                "fl2va_total": len(ka),
                "ref2va_total": len(kb),
                "shared": len(shared),
                "only_in_fl2va": sorted(ka - kb),
                "only_in_ref2va": sorted(kb - ka),
                "identical_key_sets": ka == kb,
            },
            "compared": len(per_key),
            "skipped_quantized": len(skipped),
            "skipped_patterns": sorted({collapse(k) for k in skipped}),
            "per_key": per_key,
        }


def summarise(per_key: dict) -> dict:
    """Group the per-key numbers into the families a reader reasons about."""
    fams = {
        "backbone_lora_A": lambda k: k.startswith("diffusion_model.") and k.endswith("lora_A.weight"),
        "backbone_lora_B": lambda k: k.startswith("diffusion_model.") and k.endswith("lora_B.weight"),
        "adaln_baked_diff": lambda k: ".adaln_baked." in k and k.endswith(".diff"),
        "adaln_baked_bias": lambda k: ".adaln_baked." in k and k.endswith(".diff_b"),
        "head_bank": lambda k: k.startswith("h3_pdd.bank."),
        "partition_fingerprint": lambda k: k == "h3_pdd.base_video_out",
        "adaln_curve_table": lambda k: k.endswith("adaln_t_table"),
        "norms": lambda k: k.endswith("norm.weight") or ".norm1." in k or ".norm2." in k,
        "output_heads": lambda k: k.startswith("final_layer.") and ("_out." in k),
        "input_projections": lambda k: k.endswith(("patch_proj.weight", "patch_proj.bias"))
        or k.startswith("condition_proj."),
    }
    out = {}
    for name, pred in fams.items():
        vals = [v for k, v in per_key.items()
                if pred(k) and isinstance(v, dict) and v.get("relative") is not None]
        if not vals:
            continue
        rels = sorted(v["relative"] for v in vals)
        coss = sorted(v["cosine"] for v in vals if v["cosine"] is not None)
        out[name] = {
            "n": len(vals),
            "relative": {"min": rels[0], "median": rels[len(rels) // 2], "max": rels[-1]},
            "cosine": ({"min": coss[0], "median": coss[len(coss) // 2], "max": coss[-1]}
                       if coss else None),
        }
    return out


def by_depth(per_key: dict) -> dict:
    """Backbone delta similarity per block, because depth is where it varies.

    Reported separately from the family summary: the median hides that the last
    block is markedly less orthogonal than the middle ones, and that is the one
    structural pattern in the backbone comparison.
    """
    rows = {}
    for k, v in per_key.items():
        m = re.match(r"diffusion_model\.blocks\.(\d+)\.attn\.qkv_proj\.lora_B\.weight$", k)
        if m and isinstance(v, dict) and v.get("cosine") is not None:
            rows[int(m.group(1))] = {"relative": v["relative"], "cosine": v["cosine"]}
    return dict(sorted(rows.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="How far apart fl2va and ref2va are, component by component.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    record = {
        "question": "how far apart are the fl2va and ref2va partitions, per component",
        "method": "relative Frobenius distance and cosine in float64 over every shared "
                  "unquantized tensor; quantized payloads skipped because int8_convrot "
                  "stores a rotated representation",
        "pairs": {},
    }
    for label, (fl, rf) in PAIRS.items():
        if not (fl.exists() and rf.exists()):
            record["pairs"][label] = {"absent": [p.name for p in (fl, rf) if not p.exists()]}
            print(f"  {label}: SKIPPED, file(s) absent")
            continue
        cmp = compare(fl, rf)
        cmp["summary"] = summarise(cmp["per_key"])
        depth = by_depth(cmp["per_key"])
        if depth:
            cmp["backbone_by_block"] = depth
        record["pairs"][label] = cmp

        print(f"\n  {label}: {cmp['files']['fl2va']}")
        print(f"           vs {cmp['files']['ref2va']}")
        print(f"     key sets identical: {cmp['keys']['identical_key_sets']}   "
              f"compared {cmp['compared']}, skipped {cmp['skipped_quantized']} quantized")
        for fam, s in cmp["summary"].items():
            c = s["cosine"]
            cs = f"cos {c['median']:+.4f}" if c else "cos n/a"
            print(f"     {fam:24s} n={s['n']:4d}  rel {s['relative']['median']:.4f}  {cs}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
