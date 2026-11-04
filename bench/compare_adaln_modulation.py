#!/usr/bin/env python3
"""Does a distilled checkpoint change H3's timestep conditioning, and where in t?

Written 2026-09-26 for the owner's #4 on FastH3 V2 (the full-weights plan in
`bench/results/2026-09-26_fasth3_weights_predictions.md`, P1). CPU only.

A pruned (curve-form) checkpoint stores the time embedding as `adaln_t_table`,
1025 rows over t in [0, 1] of a small basis, and each block's AdaLN projection
as a weight on that basis. Core evaluates a block's modulation as
`adaln_proj(lerp(table, t))` with no SiLU (`comfy/ldm/minimax/model.py`, the
`use_adaln_curves` path). Two conversions can pick different bases, so the
tables and weights are not comparable directly (they differ by several times
their own norm between the fl2va and FastH3 files, measured by
`bench/checkpoint_delta_map.py --identity`). The modulation OUTPUT is: this
evaluates it at every table row for both files and compares.

The same table serves the video and the audio stream at their own t, so a
student distilled on fixed rungs was supervised at two sets of t: the rungs
through the video shift and through the audio shift. `--rungs` and the two
`--shift-*` flags name them (FastH3 V2's were the defaults until the
`FASTH3_*` constants were retired on 2026-11-04), and the report compares the
change at those t with the change between them.

    CUDA_VISIBLE_DEVICES= python bench/compare_adaln_modulation.py BASE.safetensors OTHER.safetensors \
        --rungs 0.999,0.874,... --shift-video S --shift-audio S [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent


def shifted(u, s):
    return s * u / (1.0 + (s - 1.0) * u)


def modulation(f, prefix, table):
    w = f.get_tensor(prefix + ".weight").to(torch.float64)
    b = f.get_tensor(prefix + ".bias").to(torch.float64)
    return table @ w.T + b                                       # [T, out]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("other")
    # Defaulted to FastH3 V2's rungs and 10/3 shifts (`h3_config.FASTH3_*`, gone)
    # until those constants were retired on 2026-11-04; the student's own
    # values are named here now.
    ap.add_argument("--rungs", required=True,
                    help="the trained positions u in (0, 1], comma-separated")
    ap.add_argument("--shift-video", type=float, required=True)
    ap.add_argument("--shift-audio", type=float, required=True)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    rungs = [float(x) for x in args.rungs.split(",")]
    with safe_open(args.base, "pt") as a, safe_open(args.other, "pt") as b:
        ta = a.get_tensor("adaln_t_table").to(torch.float64)
        tb = b.get_tensor("adaln_t_table").to(torch.float64)
        assert ta.shape[0] == tb.shape[0], "the two tables sample t differently"
        n = ta.shape[0]
        t = np.linspace(0.0, 1.0, n)
        prefixes = sorted({k.rsplit(".", 1)[0] for k in a.keys() if "adaln_proj.linear" in k},
                          key=lambda p: (not p.startswith("blocks."), int(p.split(".")[1]) if p.startswith("blocks.") else 0))
        rel, remap = {}, {}
        for p in prefixes:
            ma, mb = modulation(a, p, ta), modulation(b, p, tb)
            r = ((mb - ma).norm(dim=1) / ma.norm(dim=1).clamp_min(1e-30)).numpy()
            rel[p] = r
            # A time remap would make the other model's modulation at t equal
            # the base's at some t' != t. For each t, the nearest base t' and
            # how much of the change moving there removes.
            d = torch.cdist(mb, ma)                                  # [T other, T base]
            best = d.argmin(dim=1).numpy()
            resid = (d.min(dim=1).values / ma.norm(dim=1).clamp_min(1e-30)).numpy()
            remap[p] = (best, resid)
            print(f"  {p:<34} rel change median {np.median(r):.4f}  min {r.min():.4f} at t={t[r.argmin()]:.3f}  "
                  f"max {r.max():.4f} at t={t[r.argmax()]:.3f}", flush=True)

    blocks = [p for p in prefixes if p.startswith("blocks.")]
    curve = np.mean([rel[p] for p in blocks], axis=0)
    best_t = np.median([t[remap[p][0]] for p in blocks], axis=0)      # per t, median over blocks
    resid = np.mean([remap[p][1] for p in blocks], axis=0)

    def at(ts):
        return [float(np.interp(x, t, curve)) for x in ts]

    video_t = [shifted(u, args.shift_video) for u in rungs]
    audio_t = [shifted(u, args.shift_audio) for u in rungs]
    # Between: midpoints in u between rungs, through each shift.
    mids = [(x + y) / 2 for x, y in zip(rungs, rungs[1:])]
    between_v = [shifted(u, args.shift_video) for u in mids]
    between_a = [shifted(u, args.shift_audio) for u in mids]
    rec = {
        "measured_by": "bench/compare_adaln_modulation.py", "base": Path(args.base).name,
        "other": Path(args.other).name, "rungs_u": rungs,
        "shift_video": args.shift_video, "shift_audio": args.shift_audio,
        "block_mean_rel_change": {
            "video_rungs": dict(zip([round(x, 5) for x in video_t], at(video_t))),
            "video_between": dict(zip([round(x, 5) for x in between_v], at(between_v))),
            "audio_rungs": dict(zip([round(x, 5) for x in audio_t], at(audio_t))),
            "audio_between": dict(zip([round(x, 5) for x in between_a], at(between_a))),
            "t_argmin": float(t[curve.argmin()]), "t_argmax": float(t[curve.argmax()]),
            "median_all_t": float(np.median(curve)),
        },
        "per_module_median": {p: float(np.median(r)) for p, r in rel.items()},
        # Remap: at each t, the base t' whose modulation is nearest (median over
        # blocks), and the change left after moving there as a share of the
        # change at t' = t. Near 1 everywhere: no remap explains the change.
        "remap": {
            "t_sample": [round(float(x), 3) for x in t[::64]],
            "nearest_base_t": [round(float(x), 4) for x in best_t[::64]],
            "residual_share": [round(float(r / c), 4) for r, c in zip(resid[::64], curve[::64])],
            "residual_share_median": float(np.median(resid / curve)),
        },
        # The full block-mean curve, 1025 points, for plotting.
        "curve_t": [round(float(x), 6) for x in t],
        "curve_block_mean": [round(float(x), 6) for x in curve],
    }
    s = rec["block_mean_rel_change"]
    for k in ("video_rungs", "video_between", "audio_rungs", "audio_between"):
        v = list(s[k].values())
        print(f"{k:<14} mean {np.mean(v):.4f}  " + " ".join(f"{t_:.3f}:{x:.4f}" for t_, x in s[k].items()))
    print(f"block-mean change: median over t {s['median_all_t']:.4f}, min at t={s['t_argmin']:.3f}, "
          f"max at t={s['t_argmax']:.3f}")
    m = rec["remap"]
    print("remap, t -> nearest base t' (residual share):")
    print("  " + "  ".join(f"{a_:.2f}->{b_:.3f}({c_:.2f})" for a_, b_, c_ in
                           zip(m["t_sample"], m["nearest_base_t"], m["residual_share"])))
    print(f"  residual share median {m['residual_share_median']:.3f}")
    if args.json:
        args.json.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
