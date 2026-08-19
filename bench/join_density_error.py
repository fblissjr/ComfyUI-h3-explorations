#!/usr/bin/env python3
"""Does a head's routed density predict its Sol error? The join that prices a per-head tau.

Two per-head records exist and neither answers the question alone.
`bench/sweep_routing_density.py` measures what each head COSTS -- the fraction
of key blocks it keeps exact. `bench/analyze_sol_error.py --heads 0` measures
what each head PAYS -- relative L2 against dense attention, split into the
sparsity term and the INT8 quantization term. A per-head tau is worth setting
only if those two are mismatched: if every head sits on one shared cost/damage
curve, a single tau is already the efficient operating point and per-head
tuning moves along the curve rather than toward it.

So this reports, per (transformer block, sampling step):

- the rank correlation between per-head density and per-head sparsity error.
  Strongly negative would mean density and error trace one shared curve, so a
  head's damage is inferable from its spend. It is not the efficient value and
  this does not report distance from one: heads differ in how compressible their
  attention is, and a genuinely diffuse head can rightly carry both high density
  and high error.
- the residual spread: heads whose error is far from what their density
  predicts. Those are the heads a per-head tau would move, and their count and
  size is the size of the prize.

## What this cannot tell you

**It is not a counterfactual.** Pricing "set head h's tau so its error matches
the cell median" needs error measured at more than one tau per head, and the
error record carries one. What this bounds is whether such a counterfactual is
worth measuring at all.

**Ranks, not magnitudes.** Per-head sparsity error spans two orders of
magnitude in a single cell, so a Pearson correlation is dominated by whichever
head is worst. `analyze_sol_error.py`'s own docstring makes the same point
about ratios needing magnitudes beside them, which is why both are printed.

**Both inputs' caveats survive the join.** The density side is the eager
routing rule rather than the CUDA kernel's INT8 arithmetic; the error side is
measured against this repo's eager Sol, gated against the vendored oracle. A
row here is only as good as the two rows it multiplies.

## Running it

    python bench/join_density_error.py \\
        --density bench/results/2026-08-19_routing_density_per_head.json \\
        --error bench/results/2026-08-19_sol_error_per_head.json \\
        --out bench/results/2026-08-19_density_error_join.json

No GPU, no capture, no server: it reads two records.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path


def spearman(pairs):
    """Rank correlation over [(x, y), ...]. None if fewer than two points."""
    n = len(pairs)
    if n < 2:
        return None
    xs = sorted(range(n), key=lambda i: pairs[i][0])
    ys = sorted(range(n), key=lambda i: pairs[i][1])
    rx = {i: r for r, i in enumerate(xs)}
    ry = {i: r for r, i in enumerate(ys)}
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return round(1 - 6 * d2 / (n * (n * n - 1)), 4)


def residuals(pairs):
    """How far each head's error sits from the density-implied trend, in ranks.

    A rank residual rather than a fitted one, because the density/error shape is
    not known to be linear and assuming it is would put the conclusion in the
    model rather than the data. A head whose density rank is 5 and whose error
    rank is 50 is doing badly for what it spends, whatever the curve.
    """
    n = len(pairs)
    xs = sorted(range(n), key=lambda i: pairs[i][0])
    ys = sorted(range(n), key=lambda i: pairs[i][1])
    rx = {i: r for r, i in enumerate(xs)}
    ry = {i: r for r, i in enumerate(ys)}
    # Density and error are expected to run opposite ways, so the null is
    # rank(error) == n - 1 - rank(density).
    return [(i, ry[i] - (n - 1 - rx[i])) for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--density", required=True)
    ap.add_argument("--error", required=True)
    ap.add_argument("--tau", type=float, default=1.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dens = json.loads(Path(args.density).read_text())
    err = json.loads(Path(args.error).read_text())

    if dens.get("capture") != err.get("capture"):
        raise SystemExit(
            f"these records are of different captures ({dens.get('capture')} vs "
            f"{err.get('capture')}). Joining them per head would silently "
            f"compare two workloads.")
    if abs(float(err.get("tau", -1)) - args.tau) > 1e-9:
        raise SystemExit(
            f"the error record is at tau {err.get('tau')}, not {args.tau}. "
            f"Density and error must be read at one operating point.")

    dmap = {}
    for r in dens["rows"]:
        if abs(r["tau"] - args.tau) < 1e-9:
            dmap[(r["block"], r["step"], r["head"])] = r["kernel_density_pct"]

    cells, flagged = [], []
    for row in err["rows"]:
        block, step = row["block"], row["step"]
        if row["heads_measured"] != row["heads_total"]:
            raise SystemExit(
                f"the error record measured {row['heads_measured']} of "
                f"{row['heads_total']} heads at b{block} s{step}. That slice is "
                f"a PREFIX, not a sample, so its head indices do not name the "
                f"same heads the density record does. Re-run with --heads 0.")
        pairs, heads = [], []
        for h, sparsity in enumerate(row["per_head_sparsity"]):
            key = (block, step, h)
            if key not in dmap:
                continue
            pairs.append((dmap[key], sparsity))
            heads.append(h)
        if len(pairs) < 2:
            continue

        rho_sparsity = spearman(pairs)
        rho_quant = spearman([(dmap[(block, step, h)], row["per_head_quant"][h])
                              for h in heads])
        res = residuals(pairs)
        worst = sorted(res, key=lambda t: -abs(t[1]))[:5]
        cells.append({
            "block": block, "step": step,
            "rho_density_vs_sparsity_error": rho_sparsity,
            "rho_density_vs_quant_error": rho_quant,
            "rank_residual_median_abs": st.median(abs(d) for _, d in res),
            "rank_residual_max_abs": max(abs(d) for _, d in res),
            "worst_heads": [{"head": heads[i], "rank_residual": d,
                             "density_pct": dmap[(block, step, heads[i])],
                             "sparsity_error": row["per_head_sparsity"][heads[i]]}
                            for i, d in worst],
        })
        print(f"b{block:<3} s{step:<3}  rho(density, sparsity err) {rho_sparsity:+.3f}"
              f"   rho(density, quant err) {rho_quant:+.3f}"
              f"   rank residual |med| {st.median(abs(d) for _, d in res):.1f}"
              f"  |max| {max(abs(d) for _, d in res)}")

    if not cells:
        raise SystemExit("nothing joined: no (block, step, head) key is in both records")

    rs = [c["rho_density_vs_sparsity_error"] for c in cells]
    print(f"\nrho(density, sparsity error) across {len(cells)} cells: "
          f"{min(rs):+.3f} .. {st.median(rs):+.3f} .. {max(rs):+.3f}")
    print("Read the sign, not a target. Strongly negative would mean density and\n"
          "error trace one shared curve, so a head's error is inferable from its\n"
          "spend; weak coupling means it is not, and a per-head schedule has to\n"
          "come from an error measurement. It does NOT follow that -1 is the\n"
          "efficient value: heads differ in how compressible their attention is,\n"
          "and a genuinely diffuse head can rightly carry both high density and\n"
          "high error. This bounds what can be inferred, not what is optimal.")

    record = {
        "measured": "2026-08-19",
        "produced_by": "bench/join_density_error.py",
        "what": "per-head routed density joined to per-head Sol error, one operating point",
        "capture": dens["capture"],
        "tau": args.tau,
        "density_record": Path(args.density).name,
        "error_record": Path(args.error).name,
        "argv": sys.argv[1:],
        "cells": cells,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
        print(f"\nwrote {args.out}")
    else:
        json.dump(record, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
