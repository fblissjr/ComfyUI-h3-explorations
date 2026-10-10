#!/usr/bin/env python3
"""Is ref2va's timestep conditioning fl2va's on a warped time axis?

`data/results/2026-09-29_partition_delta_map.jsonl` finds the two partitions
differ by a few percent in every tensor, spread like noise, except the timestep
path: `time_embedder.proj_in` (98% of its delta in 16 directions), `proj_out`,
and every block's `adaln_proj` (about two thirds in 16 directions). This runs the
release's own `TimeEmbedder` (`comfy/ldm/minimax/model.py`: cos/sin of t times
log-spaced frequencies, `proj_in`, SiLU, `proj_out`) on both partitions' weights
over t in [0, 1] and asks:

- `embed_rel`: ||e_ref(t) - e_fl(t)|| / ||e_fl(t)|| at each t;
- the best warp: for each t, the t' on a fine grid that makes e_fl(t') closest to
  e_ref(t), and how much of the mismatch that removes (`embed_rel_warped`);
- the same at the modulation level for chosen blocks: `adaln_proj(silu(e(t)))`
  of ref against fl at t and at the warped t' (`mod_rel`, `mod_rel_warped`),
  on the time-varying part (the mean over t removed) and on the whole vector;
- how the embedding difference is built: the share of its squared norm that is a
  vector constant in t (`delta_constant_share`), and how few directions the rest
  needs (`delta_centered_top_shares`);
- per chunk (3 modalities x 6 params of 5376), the relative delta of the
  time-varying part of the modulation, the quantity `bench/analyze_checkpoint_delta.py`
  records for the pruned files (`mod_tv_rel_by_chunk`), and each chunk's share of
  that time-varying norm (`mod_tv_norm_share_by_chunk`);
- the best flow-shift warp, t' = s t / (1 + (s - 1) t), and the best affine
  warp, by least squares over s and (a, b) on the embedding.

A warp that removes most of the mismatch says the two partitions were trained
on different time axes; one that removes little says the difference is
something else. CPU only.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/analyze_time_warp.py \\
        --release <MiniMaxAI_MiniMax-H3 dir> --blocks 0,25,48 --record data/results/<date>_<name>.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

FREQ_DIM = 256


def shard_map(directory: Path) -> dict[str, Path]:
    where = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            for k in f.keys():
                where[k] = shard
    return where


def get(where: dict, key: str) -> torch.Tensor:
    with safe_open(str(where[key]), "pt") as f:
        return f.get_tensor(key).float()


def embed(w: dict, t: torch.Tensor) -> torch.Tensor:
    half = FREQ_DIM // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None] * freqs[None]
    x = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    x = torch.nn.functional.silu(x @ w["proj_in.weight"].T + w["proj_in.bias"])
    return x @ w["proj_out.weight"].T + w["proj_out.bias"]


def flow_shift(t: torch.Tensor, s: float) -> torch.Tensor:
    return s * t / (1 + (s - 1) * t)


def rel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).norm(dim=-1) / b.norm(dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--blocks", default="0,25,48")
    ap.add_argument("--record", type=Path, required=True)
    args = ap.parse_args()
    where = {n: shard_map(args.release / s) for n, s in {"fl": "FL2VA/transformer", "ref": "Ref2VA/transformer"}.items()}
    te = {n: {k: get(where[n], f"time_embedder.{k}") for k in ("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias")} for n in where}

    grid = torch.linspace(0, 1, 101)
    fine = torch.linspace(0, 1, 4001)
    e_fl_fine = embed(te["fl"], fine)
    e_fl, e_ref = embed(te["fl"], grid), embed(te["ref"], grid)
    embed_rel = rel(e_ref, e_fl)
    d = torch.cdist(e_ref, e_fl_fine)
    warp = fine[d.argmin(dim=1)]
    e_warped = embed(te["fl"], warp)
    embed_rel_warped = rel(e_ref, e_warped)

    best_shift = min(((float(rel(e_ref, embed(te["fl"], flow_shift(grid, s))).pow(2).mean()), s)
                      for s in torch.linspace(0.3, 3.0, 271).tolist()))
    aff = min(((float(rel(e_ref, embed(te["fl"], (a * grid + b).clamp(0, 1))).pow(2).mean()), a, b)
               for a in torch.linspace(0.8, 1.2, 41).tolist() for b in torch.linspace(-0.1, 0.1, 41).tolist()))

    delta = e_ref - e_fl
    centered = torch.linalg.svdvals(delta - delta.mean(dim=0, keepdim=True)) ** 2
    rec = {"t": grid.tolist(),
           "delta_constant_share": float(delta.mean(dim=0).pow(2).sum() * len(grid) / delta.pow(2).sum()),
           "delta_centered_top_shares": {str(k): float(centered[:k].sum() / centered.sum()) for k in (1, 2, 3, 5)}, "embed_rel": embed_rel.tolist(), "warp": warp.tolist(), "embed_rel_warped": embed_rel_warped.tolist(),
           "best_flow_shift": {"s": best_shift[1], "rms_rel": best_shift[0] ** 0.5},
           "best_affine": {"a": aff[1], "b": aff[2], "rms_rel": aff[0] ** 0.5},
           "unwarped_rms_rel": float(embed_rel.pow(2).mean().sqrt()), "warped_rms_rel": float(embed_rel_warped.pow(2).mean().sqrt()),
           "blocks": {}}
    for b in (int(x) for x in args.blocks.split(",")):
        m = {}
        for n in where:
            w, bias = get(where[n], f"blocks.{b}.adaln_proj.linear.weight"), get(where[n], f"blocks.{b}.adaln_proj.linear.bias")
            m[n] = (w, bias)
        def mod(name, e):
            w, bias = m[name]
            return torch.nn.functional.silu(e) @ w.T + bias
        mr, mf = mod("ref", e_ref), mod("fl", e_fl)
        mfw = mod("fl", e_warped)
        tv = lambda x: x - x.mean(dim=0, keepdim=True)
        by_chunk = [float((tv(mr)[:, i * 5376:(i + 1) * 5376] - tv(mf)[:, i * 5376:(i + 1) * 5376]).norm()
                          / tv(mf)[:, i * 5376:(i + 1) * 5376].norm()) for i in range(18)]
        share = [float(tv(mf)[:, i * 5376:(i + 1) * 5376].norm() / tv(mf).norm()) for i in range(18)]
        rec["blocks"][b] = {"mod_tv_rel_by_chunk": by_chunk, "mod_tv_norm_share_by_chunk": share,
                            "mod_rel": float(rel(mr, mf).pow(2).mean().sqrt()),
                            "mod_rel_warped": float(rel(mr, mfw).pow(2).mean().sqrt()),
                            "mod_tv_rel": float((tv(mr) - tv(mf)).norm() / tv(mf).norm()),
                            "mod_tv_rel_warped": float((tv(mr) - tv(mfw)).norm() / tv(mfw).norm())}
        print("block", b, {k: round(v, 4) for k, v in rec["blocks"][b].items() if not isinstance(v, list)})
    print("embed rms rel unwarped", round(rec["unwarped_rms_rel"], 4), "best warp", round(rec["warped_rms_rel"], 4),
          "flow-shift fit", rec["best_flow_shift"], "affine fit", rec["best_affine"])
    print("warp at t=0,.25,.5,.75,1:", [round(warp[i].item(), 3) for i in (0, 25, 50, 75, 100)])
    args.record.write_text(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
