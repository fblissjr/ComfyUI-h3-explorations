#!/usr/bin/env python3
"""Rank blocks by the error `dense_blocks` actually REMOVES, not by Sol's error.

## Why the obvious ranking is the wrong one

The tempting way to choose `dense_blocks` is to rank blocks by Sol's error and
keep the worst ones dense. That is wrong here for a reason the error table
cannot show: **a "dense" block does not run dense attention.**
`vendor/sol_attn_minimax.py::make_override`'s `dense()` hands the call to
`previous`, and on every shipped graph `previous` is SAGE. So forcing a block
dense swaps Sol's approximation for sage's, and the quantity that matters is
the DIFFERENCE:

    removed(block) = sol_total_rel_l2(block) - sage_rel_l2(block)

That distinction is not academic. Both kernels are worst at block 49, so the
block with near the highest Sol error is also the block where its replacement is
least trustworthy, and ranking on Sol alone overstates what keeping it dense
buys.

## What this does and does not establish

Arithmetic over two existing measured records, not a new measurement:

  * `bench/results/2026-08-19_sol_error_per_head_tau1.0.json`
    (`analyze_sol_error.py`) -- Sol at the SHIPPED tau, all 56 heads,
    production S.
  * `bench/results/2026-08-18_sage_accuracy_on_capture.json`
    (`grade_sage_on_capture.py`) -- sage in `auto`, the shipped mode, against a
    float64 reference.

**The two come from different captures and different harnesses, so the
subtraction is approximate.** Both are 362 frames at 1024x768 with three
references and both report S = 98,498, which is what makes them comparable at
all; that equality is asserted below rather than assumed. Sol's figure is
per-head over the sequence, sage's is 256 stratified rows against an exact
reference. Read the RANKING, which spans 2-3x and survives that slop; do not
quote a single `removed` value as a measured quantity.

**It says nothing about propagation.** Error at a block is not impact at the
output: block 0's travels through 49 more blocks, block 49's lands on the
output head. Nothing here or anywhere in this repo measures that, and it is the
one thing that could overturn the ranking. See `docs/SOLATTN.md`.

    python bench/rank_dense_blocks.py [--write]

No GPU, no server, no capture -- it reads two JSON files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench" / "results"

SOL = RESULTS / "2026-08-19_sol_error_per_head_tau1.0.json"
SAGE = RESULTS / "2026-08-18_sage_accuracy_on_capture.json"
OUT = RESULTS / "2026-08-29_dense_block_ranking.json"

# What the shipped configs keep dense, so the ranking can be read against them
# rather than beside them. Imported rather than retyped.
sys.path.insert(0, str(REPO / "workflows"))


def _mean(xs):
    return sum(xs) / len(xs)


def load(sol_path=SOL, sage_path=SAGE):
    """Per-block Sol total and sage rel L2, or raise saying why they cannot be compared."""
    sol = json.loads(sol_path.read_text())
    sage = json.loads(sage_path.read_text())

    sol_by, sage_by = {}, {}
    sol_lens, sage_lens = set(), set()
    for r in sol["rows"]:
        sol_by.setdefault(r["block"], []).append(r)
        sol_lens.add(r["seq_len"])
    for p in sage["points"]:
        # `auto` is the shipped sage mode (h3_config.SAGE_NODE). The file's own
        # summary records that every fp8 variant agrees to 4 decimals, so the
        # choice among them cannot move this ranking -- but pin it explicitly
        # rather than taking whichever key sorts first.
        sage_by.setdefault(p["block"], []).append(p["modes"]["auto"]["rel_l2_mean"])
        sage_lens.add(p["sequence"])

    # The guard that makes the subtraction legitimate. Sol's error is strongly
    # length-dependent (routed density is a variance over every block centroid
    # in the sequence), so two records at different S are two operating points
    # and subtracting them is meaningless. Refuse rather than print.
    if sol_lens != sage_lens or len(sol_lens) != 1:
        raise SystemExit(
            f"refusing to subtract: Sol rows are at S={sorted(sol_lens)} and sage "
            f"points at S={sorted(sage_lens)}. Sol's error is length-dependent, so "
            f"records at different lengths are different operating points.")

    blocks = sorted(set(sol_by) & set(sage_by))
    if not blocks:
        raise SystemExit("refusing: the two records share no block")
    rows = []
    for b in blocks:
        sol_total = _mean([r["total_l2"] for r in sol_by[b]])
        sol_sparsity = _mean([r["sparsity_l2"] for r in sol_by[b]])
        sol_quant = _mean([r["quant_l2"] for r in sol_by[b]])
        sage_l2 = _mean(sage_by[b])
        rows.append({
            "block": b,
            "sol_total_rel_l2": sol_total,
            "sol_sparsity_rel_l2": sol_sparsity,
            "sol_quant_rel_l2": sol_quant,
            "sage_rel_l2": sage_l2,
            "removed_by_dense": sol_total - sage_l2,
            "sol_over_sage": sol_total / sage_l2,
        })
    return rows, sol, sage, sorted(sol_lens)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help=f"write {OUT.name}")
    ap.add_argument("--sol", type=Path, default=SOL)
    ap.add_argument("--sage", type=Path, default=SAGE)
    args = ap.parse_args()

    rows, sol, sage, seq = load(args.sol, args.sage)

    import h3_config
    shipped = {
        "SOL_RECOMMENDED_CUDA": h3_config.SOL_RECOMMENDED_CUDA["dense_blocks"],
        "SOL_PDD_CUDA": h3_config.SOL_PDD_CUDA["dense_blocks"],
    }

    print(f"Sol (tau {sol['tau']}) against sage (auto), S = {seq}")
    print(f"  sol  {sol['capture']}")
    print(f"  sage {sage['points'][0]['capture'].split('/')[-2]}")
    print()
    print("  block   sol_total   sage    REMOVED   sol/sage")
    for r in sorted(rows, key=lambda r: -r["removed_by_dense"]):
        print(f"    {r['block']:2d}      {r['sol_total_rel_l2']:.4f}   "
              f"{r['sage_rel_l2']:.4f}   {r['removed_by_dense']:.4f}    "
              f"{r['sol_over_sage']:5.1f}x")
    print()
    best = max(rows, key=lambda r: r["removed_by_dense"])
    worst = min(rows, key=lambda r: r["removed_by_dense"])
    print(f"  best measured candidate: block {best['block']} "
          f"({best['removed_by_dense']:.4f} removed)")
    print(f"  worst measured candidate: block {worst['block']} "
          f"({worst['removed_by_dense']:.4f} removed)")
    for name, spec in shipped.items():
        print(f"  {name} keeps {spec!r} dense")
    print("\n  measured blocks only -- 1, 2 and 48 are in no capture, and "
          "nothing here measures propagation to the output.")

    if args.write:
        OUT.write_text(json.dumps({
            "measured": "2026-08-29",
            "produced_by": "bench/rank_dense_blocks.py",
            "what": "error that dense_blocks REMOVES per block: Sol's total minus "
                    "sage's, because a dense block runs sage rather than exact "
                    "attention",
            "inputs": [args.sol.name, args.sage.name],
            "sol_tau": sol["tau"],
            "sol_capture": sol["capture"],
            "sage_mode": "auto",
            "sequence": seq,
            "shipped_dense_blocks": shipped,
            "rows": rows,
            "caveats": [
                "Arithmetic over two existing records, not a new measurement.",
                "The two come from different captures and different harnesses; "
                "read the ranking, not a single value.",
                "Measured blocks only. 1, 2 and 48 appear in no capture.",
                "Says nothing about propagation: error AT a block is not impact "
                "ON the output, and an early block's error travels through the "
                "rest of the model while block 49's does not.",
                "sage's own cos_min goes NEGATIVE at block 49 (-0.04 to -0.11), "
                "so the replacement is at its own worst exactly there. The "
                "subtraction still favours dense, but by less than the Sol "
                "column alone suggests.",
            ],
        }, indent=1) + "\n")
        print(f"\n  wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
