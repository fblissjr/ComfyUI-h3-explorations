"""Per-block tables from the depth-profile outputs: Sol's error, the dense kernels' error, and whether the ranking holds across scenes.

    python bench/depth_profile_tables.py \\
        --sol market=bench/results/2026-09-19_sol_orderings_depth_covered_market.json \\
        --sol noodle=bench/results/2026-09-19_sol_orderings_depth_noodle_bar_107f.json \\
        --dense market=bench/results/2026-09-19_dense_kernels_covered_market_sage_chain_depth.json \\
        --dense market=bench/results/2026-09-19_dense_kernels_covered_market_sage_chain_steps.json \\
        --dense noodle=bench/results/2026-09-19_dense_kernels_noodle_bar_sage_chain_107f.json \\
        [--json out.json]

Inputs are `bench/sweep_sol_orderings_on_capture.py` outputs (Sol, read at
raster order and tau 1.0 through `bench/compare_sol_orderings.py`) and
`bench/grade_dense_kernels_on_captures.py` outputs. Several `--dense` files may
share a label; their rows are pooled.

What it prints, per step that two or more scenes share:

- each block's Sol error (raster, tau 1.0) and routed density per scene, with
  its rank within the scene (1 = worst);
- Spearman's rank correlation of the per-block Sol error between each pair of
  scenes, over the blocks both have. This is the stability check that
  `internal/2026-09-19_question_review.md` section D asks for before any block
  ranking is used: a ranking that does not hold across scenes is noise;
- the dense kernels' error per block, and the same correlation for each.

It sorts and correlates; it does not choose `dense_blocks`. Error against
exact attention on a capture has disagreed with the owner's eye in this pack
(the 2026-09-18 reorder panel), so a ranking here nominates candidates for a
render, never a default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_sol_orderings import summarize  # noqa: E402


def _cell(name):
    m = re.match(r"b(\d+)_s(\d+)", name)
    if m is None:
        raise SystemExit(f"cell name {name!r} is not b<block>_s<step>")
    return int(m.group(1)), int(m.group(2))


def _ranks(values):
    """Average ranks, 1 = smallest; ties share the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else None


def _pairs(labels):
    return [(a, b) for i, a in enumerate(labels) for b in labels[i + 1:]]


def main():
    ap = argparse.ArgumentParser(description="Per-block Sol and dense-kernel tables across scenes.")
    ap.add_argument("--sol", action="append", default=[], help="LABEL=orderings sweep json")
    ap.add_argument("--dense", action="append", default=[], help="LABEL=dense kernels json (repeatable per label)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    sol = defaultdict(dict)          # label -> (block, step) -> (density, error)
    for item in args.sol:
        label, _, path = item.partition("=")
        for cell, row in summarize(path).items():
            sol[label][_cell(cell)] = (row["raster"]["density"], row["raster"]["rel_l2"])
    dense = defaultdict(dict)        # label -> (block, step) -> {kernel: error}
    kernels = []
    for item in args.dense:
        label, _, path = item.partition("=")
        for r in json.load(open(path))["rows"]:
            ks = {k: v for k, v in r.items() if k not in ("block", "step")}
            for k in ks:
                if k not in kernels:
                    kernels.append(k)
            dense[label][(r["block"], r["step"])] = ks

    out = {"sol": {}, "dense": {}}
    labels = list(sol)
    steps = sorted({s for lab in labels for (_, s) in sol[lab]})
    for step in steps:
        have = [lab for lab in labels if any(s == step for (_, s) in sol[lab])]
        blocks = sorted(set.intersection(*[{b for (b, s) in sol[lab] if s == step} for lab in have]))
        print(f"\nSol, raster, tau 1.0, step {step}: routed density / error (rank, 1 = worst)")
        ranks = {}
        for lab in have:
            errs = [sol[lab][(b, step)][1] for b in blocks]
            r = _ranks([-e for e in errs])
            ranks[lab] = dict(zip(blocks, r))
        print("  block  " + "  ".join(f"{lab:>24s}" for lab in have))
        for b in blocks:
            cells = [f"{sol[lab][(b, step)][0]:.3f} / {sol[lab][(b, step)][1]:.4f} ({ranks[lab][b]:>4.1f})"
                     for lab in have]
            print(f"  {b:5d}  " + "  ".join(f"{c:>24s}" for c in cells))
        corr = {}
        for a, c in _pairs(have):
            rho = spearman([sol[a][(b, step)][1] for b in blocks], [sol[c][(b, step)][1] for b in blocks])
            corr[f"{a}~{c}"] = rho
            print(f"  Spearman {a} against {c}: {rho:+.2f} over {len(blocks)} blocks")
        out["sol"][step] = {"blocks": blocks, "cells": {lab: {b: sol[lab][(b, step)] for b in blocks} for lab in have},
                            "spearman": corr}

    dlabels = list(dense)
    dsteps = sorted({s for lab in dlabels for (_, s) in dense[lab]})
    for step in dsteps:
        have = [lab for lab in dlabels if any(s == step for (_, s) in dense[lab])]
        blocks = sorted(set.intersection(*[{b for (b, s) in dense[lab] if s == step} for lab in have]))
        print(f"\nDense kernels, step {step}: error against fp32 per block")
        print("  block  " + "  ".join(f"{lab}:{k:>12s}" for lab in have for k in kernels))
        for b in blocks:
            print(f"  {b:5d}  " + "  ".join(f"{dense[lab][(b, step)].get(k, float('nan')):>{len(lab) + 13}.4f}"
                                             for lab in have for k in kernels))
        corr = {}
        for a, c in _pairs(have):
            for k in kernels:
                rho = spearman([dense[a][(b, step)][k] for b in blocks], [dense[c][(b, step)][k] for b in blocks])
                corr[f"{a}~{c}:{k}"] = rho
                if rho is not None:
                    print(f"  Spearman {a} against {c}, {k}: {rho:+.2f} over {len(blocks)} blocks")
        out["dense"][step] = {"blocks": blocks, "cells": {lab: {b: dense[lab][(b, step)] for b in blocks} for lab in have},
                              "spearman": corr}
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
