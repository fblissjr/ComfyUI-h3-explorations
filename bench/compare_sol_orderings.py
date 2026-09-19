"""Read `bench/sweep_sol_orderings_on_capture.py` outputs at raster's shipped point, side by side.

    python bench/compare_sol_orderings.py LABEL=path.json [LABEL=path.json ...] [--tau 1.0] [--json out]

For every cell, raster's (density, error) at `--tau`, and for each other
ordering: its error at raster's density ("err") and the density it needs to
reach raster's error ("dens"), both interpolated linearly in log-log along the
ordering's own curve, as percentages of raster's. Negative is better. "n/a"
means the point is outside the ordering's swept range.

Cells are matched across files by name with the capture's route tag dropped
(`b40_s15_ksol` and `b40_s15` are the same cell), so a set taken before the tag
existed lines up with one taken after.

Written 2026-09-19 for `bench/results/2026-09-19_sol_orderings_old_vs_new.md`.
The 2026-09-17 record states the method in words but not its code; this
re-implementation reproduces that record's published deltas to within a few
tenths of a point, which is the tolerance to quote against it.
"""

from __future__ import annotations

import argparse
import json
import math


def _loglog(xs, ys, x):
    pts = sorted(zip(xs, ys))
    lx = [math.log(a) for a, _ in pts]
    ly = [math.log(b) for _, b in pts]
    t = math.log(x)
    for i in range(len(pts) - 1):
        lo, hi = lx[i], lx[i + 1]
        if lo <= t <= hi:
            f = (t - lo) / (hi - lo) if hi != lo else 0.0
            return math.exp(ly[i] + f * (ly[i + 1] - ly[i]))
    return None


def summarize(path, tau=1.0):
    cells = json.load(open(path))["cells"]
    out = {}
    for cell, curves in cells.items():
        ref = next(p for p in curves["raster"] if p["tau"] == tau)
        d0, e0 = ref["density"], ref["rel_l2"]
        row = {"raster": {"density": d0, "rel_l2": e0}}
        for name, curve in curves.items():
            if name == "raster":
                continue
            dens = [p["density"] for p in curve]
            err = [p["rel_l2"] for p in curve]
            e_at = _loglog(dens, err, d0)
            d_at = _loglog(err, dens, e0)
            row[name] = {"err_pct": None if e_at is None else 100 * (e_at / e0 - 1),
                         "dens_pct": None if d_at is None else 100 * (d_at / d0 - 1)}
        out[cell.replace("_ksol", "").replace("_ksage", "")] = row
    return out


def main():
    ap = argparse.ArgumentParser(description="Sol ordering sweeps at raster's shipped point, side by side.")
    ap.add_argument("inputs", nargs="+", help="LABEL=path.json")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    sets = {}
    for item in args.inputs:
        label, _, path = item.partition("=")
        sets[label] = summarize(path, args.tau)
    cells = list(dict.fromkeys(c for s in sets.values() for c in s))
    pct = lambda v: "n/a" if v is None else f"{v:+.1f}%"
    for cell in cells:
        print(cell)
        for label, s in sets.items():
            row = s.get(cell)
            if row is None:
                continue
            r = row["raster"]
            others = " | ".join(f"{k} err {pct(v['err_pct'])} dens {pct(v['dens_pct'])}"
                                for k, v in row.items() if k != "raster")
            print(f"  {label:10s} raster {r['density']:.3f}/{r['rel_l2']:.4f} | {others}")
    if args.json:
        json.dump({"tau": args.tau, "sets": sets}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
