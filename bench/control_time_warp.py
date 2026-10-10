#!/usr/bin/env python3
"""Can `bench/analyze_time_warp.py` see a warp at all? A control for its "not a time warp" claim.

`2026-09-29_ref2va_partition_delta.md` reads Ref2VA's time embedder as FL2VA's
plus a near-constant offset, not FL2VA's evaluated at a warped time. The test
that says so (a nonparametric nearest-t warp, the best flow shift and the best
affine warp of the embedder input) had never been shown to be able to fail.
This gives the script's own statistic (its `embed`, `flow_shift`, `rel`,
`shard_map` and `get`, unchanged) curves built from known warps:

- A: FL2VA's embedder itself, evaluated at a known warp (an affine a*t, a flow
  shift, an offset). The statistic should recover it.
- B: the real Ref2VA embedder, evaluated at a known warp. This is the case
  the real comparison is in: a warp on top of the real offset.

CPU only.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/control_time_warp.py \\
        --release <MiniMaxAI_MiniMax-H3 dir> --record data/results/<date>_<name>.json
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--release", type=Path, required=True)
ap.add_argument("--record", type=Path, required=True)
args = ap.parse_args()
spec = importlib.util.spec_from_file_location("atw", Path(__file__).resolve().parent / "analyze_time_warp.py")
atw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atw)
R = args.release
where = {n: atw.shard_map(R / s) for n, s in {"fl": "FL2VA/transformer", "ref": "Ref2VA/transformer"}.items()}
te = {n: {k: atw.get(where[n], f"time_embedder.{k}") for k in ("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias")} for n in where}
embed, rel, fs = atw.embed, atw.rel, atw.flow_shift
grid = torch.linspace(0, 1, 101)
fine = torch.linspace(0, 1, 4001)
e_fl_fine = embed(te["fl"], fine)
e_fl = embed(te["fl"], grid)

def stat(e_ref):
    """the script's statistic, unchanged."""
    embed_rel = rel(e_ref, e_fl)
    warp = fine[torch.cdist(e_ref, e_fl_fine).argmin(dim=1)]
    embed_rel_w = rel(e_ref, embed(te["fl"], warp))
    bs = min(((float(rel(e_ref, embed(te["fl"], fs(grid, s))).pow(2).mean()), s) for s in torch.linspace(0.3, 3.0, 271).tolist()))
    aff = min(((float(rel(e_ref, embed(te["fl"], (a * grid + b).clamp(0, 1))).pow(2).mean()), a, b)
               for a in torch.linspace(0.8, 1.2, 41).tolist() for b in torch.linspace(-0.1, 0.1, 41).tolist()))
    return dict(unwarped=float(embed_rel.pow(2).mean().sqrt()), nonparam=float(embed_rel_w.pow(2).mean().sqrt()),
                shift_s=bs[1], shift_rms=bs[0] ** .5, aff_a=aff[1], aff_b=aff[2], aff_rms=aff[0] ** .5,
                warp_mid=float(warp[50]), warp_end=float(warp[100]))

out = {"real": stat(embed(te["ref"], grid))}
# A: pure warp of fl (no real delta): ref_fake(t) = e_fl(w(t)); w = affine a*t (a<1 stays in [0,1], so the fine grid covers it)
out["A_pure"] = {}
for a in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0):
    out["A_pure"][f"a={a}"] = stat(embed(te["fl"], a * grid))
for s in (0.5, 0.8, 0.9, 1.1, 1.25, 2.0):
    out["A_pure"][f"flowshift s={s}"] = stat(embed(te["fl"], fs(grid, s)))
for b in (0.02, 0.05, -0.05):
    out["A_pure"][f"shift b={b}"] = stat(embed(te["fl"], (grid + b).clamp(0, 1)))
# B: real Ref2VA embedder, additionally warped by a known factor (a<1 keeps inside [0,1])
out["B_real_plus_warp"] = {}
for a in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0):
    out["B_real_plus_warp"][f"a={a}"] = stat(embed(te["ref"], a * grid))
for s in (0.8, 0.9, 1.1, 1.25, 2.0):
    out["B_real_plus_warp"][f"flowshift s={s}"] = stat(embed(te["ref"], fs(grid, s)))
for b in (0.02, 0.05, -0.05):
    out["B_real_plus_warp"][f"shift b={b}"] = stat(embed(te["ref"], (grid + b).clamp(0, 1)))
args.record.write_text(json.dumps(out, indent=1))
for grp, d in out.items():
    if grp == "real": print("real", {k: round(v, 4) for k, v in d.items()}); continue
    print(grp)
    for k, v in d.items():
        print(f"  {k:20s}", {kk: round(vv, 4) for kk, vv in v.items()})
