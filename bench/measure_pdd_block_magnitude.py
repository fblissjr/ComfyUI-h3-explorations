"""How much does the PDD backbone LoRA move each block, and how does that
compare to what int8 quantisation already moved there?

The question this exists for: **can PDD be applied per block, weighted by a
block's quantisation sensitivity?** That plan needs two things to be true and
this measures both.

1. PDD's update must be large enough at a block for scaling it to matter.
   Reported as `pdd_rel`: `||alpha/rank * B @ A||_F / ||W||_F` against the
   dequantised checkpoint weight the patch lands on.
2. The per-block quantisation error must actually VARY, or there is nothing
   to key a per-block schedule off. Read from
   `2026-08-21_quant_delta_*.json`'s `int8_vs_bf16`, which is the same
   normalisation, so the two columns are directly comparable.

Both numbers are STORED-WEIGHT quantities. Neither is an activation-level
sensitivity and neither says what reaches the output -- the propagation
profile is `2026-08-29_block_propagation.json`'s and is a different axis.

Needs `comfy_kitchen` (the un-rotation) and torch on CPU. No GPU, no server.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from analyze_checkpoint_delta import header  # noqa: E402
from analyze_quant_delta import weight_in_compute_space  # noqa: E402

KINDS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
N_BLOCKS = 50


def lora_delta(f, prefix: str) -> tuple[np.ndarray, float, int]:
    """`alpha/rank * B @ A`, exactly what ComfyUI's generic LoRA applies."""
    a = f.get_tensor(prefix + ".lora_A.weight").to(torch.float32)
    b = f.get_tensor(prefix + ".lora_B.weight").to(torch.float32)
    alpha = float(f.get_tensor(prefix + ".alpha").item())
    rank = a.shape[0]
    return ((alpha / rank) * (b @ a)).numpy(), alpha, rank


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="fl2va", choices=("fl2va", "ref2va"))
    ap.add_argument("--lora", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--quant-delta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=N_BLOCKS)
    args = ap.parse_args()

    hdr, off = header(args.base)
    quant = json.load(open(args.quant_delta))
    qmap = {(m["block"], m["kind"]): m["int8_vs_bf16"]["rel_delta"]
            for m in quant["modules"]}

    rows = []
    with safe_open(args.lora, "pt") as f:
        for blk in range(args.blocks):
            for kind in KINDS:
                mod = f"blocks.{blk}.{kind}"
                pre = f"diffusion_model.{mod}"
                d, alpha, rank = lora_delta(f, pre)
                w = weight_in_compute_space(args.base, hdr, off, mod)
                if d.shape != w.shape:
                    raise SystemExit(f"shape {mod}: lora {d.shape} base {w.shape}")
                wn = float(np.linalg.norm(w.astype(np.float64)))
                dn = float(np.linalg.norm(d.astype(np.float64)))
                # cosine between the update and the weight it lands on: an
                # update aligned with W is a rescale, one orthogonal to it is
                # new structure. Different things to perturb.
                cos = float((w.astype(np.float64) * d.astype(np.float64)).sum()
                            / (wn * dn)) if wn and dn else 0.0
                rows.append({
                    "block": blk, "kind": kind,
                    "pdd_rel": dn / wn,
                    "pdd_norm": dn, "base_norm": wn,
                    "cos_with_base": cos,
                    "alpha": alpha, "rank": rank,
                    "int8_vs_bf16_rel": qmap.get((blk, kind)),
                })

    by_block = {}
    for r in rows:
        by_block.setdefault(r["block"], []).append(r)
    block_pdd = {b: sum(x["pdd_rel"] for x in v) / len(v)
                 for b, v in by_block.items()}
    block_q = {b: sum(x["int8_vs_bf16_rel"] for x in v) / len(v)
               for b, v in by_block.items()}

    bs = sorted(block_pdd)
    p = np.array([block_pdd[b] for b in bs])
    q = np.array([block_q[b] for b in bs])
    corr = float(np.corrcoef(p, q)[0, 1])

    out = {
        "measured": "2026-08-30",
        "produced_by": "bench/measure_pdd_block_magnitude.py",
        "what": ("per-block magnitude of the PDD backbone LoRA update, "
                 "relative to the dequantised weight it patches, beside the "
                 "int8-vs-bf16 stored-weight error at the same module"),
        "checkpoint": args.checkpoint,
        "lora": Path(args.lora).name,
        "base": Path(args.base).name,
        "quant_source": Path(args.quant_delta).name,
        "is_not": ("an activation-level sensitivity, and not a statement "
                   "about what reaches the output; both columns are "
                   "stored-weight norms"),
        "pdd_rel": {
            "min": float(p.min()), "max": float(p.max()),
            "mean": float(p.mean()),
            "spread_max_over_min": float(p.max() / p.min()),
        },
        "int8_rel": {
            "min": float(q.min()), "max": float(q.max()),
            "mean": float(q.mean()),
            "spread_max_over_min": float(q.max() / q.min()),
        },
        "pdd_over_int8_mean": float(p.mean() / q.mean()),
        "corr_block_pdd_vs_block_int8": corr,
        "by_block": {str(b): {"pdd_rel": block_pdd[b], "int8_rel": block_q[b]}
                     for b in bs},
        "by_kind": {
            k: {
                "pdd_rel_mean": float(np.mean([r["pdd_rel"] for r in rows
                                               if r["kind"] == k])),
                "pdd_rel_min": float(np.min([r["pdd_rel"] for r in rows
                                             if r["kind"] == k])),
                "pdd_rel_max": float(np.max([r["pdd_rel"] for r in rows
                                             if r["kind"] == k])),
                "cos_with_base_mean": float(np.mean([r["cos_with_base"]
                                                     for r in rows
                                                     if r["kind"] == k])),
                "int8_rel_mean": float(np.mean([r["int8_vs_bf16_rel"]
                                                for r in rows
                                                if r["kind"] == k])),
            } for k in KINDS},
        "modules": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("modules", "by_block")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
