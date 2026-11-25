#!/usr/bin/env python3
"""Timestep conditioning from the unpruned weights, and what each conversion lost.

Written 2026-09-26 for the owner's #4 on FastH3 V2. `compare_adaln_modulation.py`
found FastH3's modulation about 5% off fl2va's, measured on the two PRUNED
files. Each pruned file is a conversion (Comfy-Org's for fl2va, FastVideo's
for FastH3) that replaces the time embedder and full-width AdaLN with a
1025-row table on a small basis, so that 5% mixes what training changed with
what each conversion lost. The unpruned diffusers-form transformers separate
them. CPU only; one block resident at a time.

Unpruned modulation, as core computes it on a non-curve checkpoint
(`comfy/ldm/minimax/model.py`, `TimeEmbedder` then `AdalnProj` with SiLU):
    m(t) = W_adaln @ silu(linear_2(silu(linear_1(sinusoid(t))))) + b
with t the clean time in [0, 1], 256 frequencies, cos before sin. Pruned:
    m(t) = W_curve @ lerp(adaln_t_table, t) + b_curve
evaluated on the table's own 1025 rows.

Reports, per block and the final layer, the median over t of ||m1 - m2|| / ||m2||
for:
    train     unpruned other  vs unpruned base   what the fine-tune changed
    conv_base pruned base     vs unpruned base   the base conversion's error
    conv_oth  pruned other    vs unpruned other  the other conversion's error
    pruned    pruned other    vs pruned base     what compare_adaln_modulation saw
    temb_only base projection on the other's time embedding, vs base
    proj_only the other's projection on the base's time embedding, vs base
The last two split `train` between the time embedder (one MLP every block
shares) and the per-block projections.

    CUDA_VISIBLE_DEVICES= python bench/compare_adaln_unpruned.py \\
        --base-dir BASE/transformer --other-dir OTHER/transformer \\
        --base-pruned B.safetensors --other-pruned O.safetensors [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


class Sharded:
    """Tensor reads from a diffusers sharded directory by its index."""

    def __init__(self, d: str):
        self.d = Path(d)
        idx = next(self.d.glob("*.safetensors.index.json"))
        self.where = json.loads(idx.read_text())["weight_map"]
        self.open = {}

    def get(self, key: str) -> torch.Tensor:
        shard = self.where[key]
        if shard not in self.open:
            self.open[shard] = safe_open(str(self.d / shard), "pt")
        return self.open[shard].get_tensor(key).to(torch.float32)


def sinusoid(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64) / half)
    args = t.to(torch.float64)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def unpruned_temb(sh: Sharded, t: torch.Tensor) -> torch.Tensor:
    w1 = sh.get("time_embedder.linear_1.weight").double()
    emb = sinusoid(t, w1.shape[1])
    h = torch.nn.functional.silu(emb @ w1.T + sh.get("time_embedder.linear_1.bias").double())
    h = h @ sh.get("time_embedder.linear_2.weight").double().T + sh.get("time_embedder.linear_2.bias").double()
    return torch.nn.functional.silu(h)                                   # AdalnProj's SiLU


def unpruned_mod(sh: Sharded, key: str, temb: torch.Tensor) -> torch.Tensor:
    return temb @ sh.get(key + ".weight").double().T + sh.get(key + ".bias").double()


def pruned_mod(f, key: str, table: torch.Tensor) -> torch.Tensor:
    return table @ f.get_tensor(key + ".weight").double().T + f.get_tensor(key + ".bias").double()


def rel(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return ((a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--other-dir", required=True)
    ap.add_argument("--base-pruned", required=True)
    ap.add_argument("--other-pruned", required=True)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    base, other = Sharded(args.base_dir), Sharded(args.other_dir)
    rows = {}
    with safe_open(args.base_pruned, "pt") as pb, safe_open(args.other_pruned, "pt") as po:
        tb = pb.get_tensor("adaln_t_table").double()
        to = po.get_tensor("adaln_t_table").double()
        t = torch.linspace(0.0, 1.0, tb.shape[0], dtype=torch.float64)
        eb, eo = unpruned_temb(base, t), unpruned_temb(other, t)
        depth = sum(1 for k in base.where if k.endswith("adaln_proj.linear.weight"))
        pairs = [(f"transformer_blocks.{i}.adaln_proj.linear", f"blocks.{i}.adaln_proj.linear")
                 for i in range(depth)] + [("norm_out.linear", "final_layer.adaln_proj.linear")]
        for ukey, pkey in pairs:
            ub, uo = unpruned_mod(base, ukey, eb), unpruned_mod(other, ukey, eo)
            qb, qo = pruned_mod(pb, pkey, tb), pruned_mod(po, pkey, to)
            r = {"train": rel(uo, ub), "conv_base": rel(qb, ub), "conv_oth": rel(qo, uo), "pruned": rel(qo, qb),
                 "temb_only": rel(unpruned_mod(base, ukey, eo), ub),
                 "proj_only": rel(unpruned_mod(other, ukey, eb), ub)}
            rows[pkey] = {k: {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
                              "t_of_max": float(t[int(v.argmax())])} for k, v in r.items()}
            print(f"  {pkey:<32} " + "  ".join(f"{k} {rows[pkey][k]['median']:.4f}" for k in r), flush=True)
            base.open.clear()
            other.open.clear()

    blocks = [k for k in rows if k.startswith("blocks.")]
    summary = {k: float(np.median([rows[b][k]["median"] for b in blocks]))
               for k in ("train", "conv_base", "conv_oth", "pruned", "temb_only", "proj_only")}
    print("block medians:", {k: round(v, 5) for k, v in summary.items()})
    if args.json:
        args.json.write_text(json.dumps({
            "measured_by": "bench/compare_adaln_unpruned.py",
            "base_dir": Path(args.base_dir).parent.name + "/" + Path(args.base_dir).name,
            "other_dir": Path(args.other_dir).parent.name + "/" + Path(args.other_dir).name,
            "base_pruned": Path(args.base_pruned).name, "other_pruned": Path(args.other_pruned).name,
            "t_grid": int(tb.shape[0]), "block_median": summary, "per_module": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
