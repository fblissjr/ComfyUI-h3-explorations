#!/usr/bin/env python3
"""What each legal PDD step count actually buys, on the shift-12 sigma grid.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). Needs no server,
no GPU and no checkpoint -- it is a function of `pdd_math` and the node's own
partition rule, nothing else.

**Not a check.** It grades nothing. It exists because the sidecar README tells
a stranger which step count to use, and that guidance was briefly wrong in a
way worth recording: it was derived from the FINAL block alone, and the final
block is the one statistic on which 5, 6, 7 and 8 are exactly tied.

## The result, and why it is not the obvious one

Under shift 12 the sigma grid is severely non-uniform -- the last block spans
most of the range and the early blocks span almost none. So evaluations added
at the FRONT of the partition buy almost nothing, and the ladder is flat:

    4 evaluations   is ~50% coarser than 5 on summed squared step
    5 through 8     lie within 2% of each other on every statistic here

That is the opposite of the intuition that more evaluations are steadily
better, and it is the reason the shipped examples stop at three counts rather
than one per legal value.

**Four is not a choice.** `[8,8,8,8]` is the only partition of the 32-point
grid into four blocks legal under the trained envelope, so its 80% final step
is forced. Every count from five up can keep a width-4 block last.

## What this is not

A perceptual claim. These are properties of the SCHEDULE. `CLAUDE.md`'s
standing rule is that a rendered clip cannot A/B a numerical change -- two
counts produce different samples, not better and worse versions of one -- so
nothing here says how any of them look. It says what the integrator is asked
to do.

Usage:

    python bench/measure_pdd_step_ladder.py
    python bench/measure_pdd_step_ladder.py --out bench/results/DATE_pdd_step_ladder.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))          # ComfyUI root
sys.path.insert(0, str(HERE.parent))              # this repo

import pdd_lora as P    # noqa: E402
import pdd_math as M    # noqa: E402

#: The grid the heads are replicated over, and the width they were distilled
#: at. Both are properties of the alibaba-pai artifacts, not choices here.
GRID, TRAINED, SHIFT = 32, 4, 12.0


def widths_for(nfe: int, grid: int, trained: int):
    """The partition the node would emit, by ITS rule rather than half of it.

    `resolve_emit_steps` accepts a count by either of two routes and this has
    to mirror both. A first version used `envelope_partition` alone and
    reported 16 and 32 as illegal -- they are divisors, so they tile uniformly
    and never reach the envelope branch at all. The envelope only covers the
    non-divisors 5, 6 and 7.

    Returns the widths, plus which route admitted the count, because the route
    is the interesting part: a divisor below the trained width fuses blocks
    NARROWER than anything the bank was distilled over.
    """
    if grid % nfe == 0:
        return [grid // nfe] * nfe, "divisor"
    w = P.envelope_partition(grid, nfe, trained)
    return (w, "envelope") if w is not None else (None, None)


def ladder(grid: int = GRID, trained: int = TRAINED, shift: float = SHIFT) -> dict:
    """Every count from 1 to `grid`, with the partition the node would emit."""
    rows = {}
    for nfe in range(1, grid + 1):
        widths, route = widths_for(nfe, grid, trained)
        if widths is None:
            rows[nfe] = {"legal": False}
            continue
        b = M.partition_bounds(shift, grid, widths)
        span = float(b[0] - b[-1])
        steps = [float(b[i] - b[i + 1]) / span for i in range(len(b) - 1)]
        rows[nfe] = {
            "legal": True,
            "route": route,
            "within_trained_envelope": all(trained <= x <= 2 * trained for x in widths),
            "widths": list(widths),
            "step_fractions": steps,
            "final": steps[-1],
            "worst": max(steps),
            "sum_squared": sum(x * x for x in steps),
        }
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="What each legal PDD step count buys on the shift-12 grid.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = ladder()
    legal = {k: v for k, v in rows.items() if v["legal"]}

    print(f"  {GRID}-point grid, shift {SHIFT}, trained block width {TRAINED}\n")
    print(f"  {'nfe':>4}  {'widths':30s} {'route':9s} {'env':4s} "
          f"{'final':>7} {'worst':>7} {'sum sq':>8}")
    for nfe, r in legal.items():
        w = str(r['widths'])
        if len(w) > 30:
            w = w[:27] + "..."
        print(f"  {nfe:4d}  {w:30s} {r['route']:9s} "
              f"{'yes' if r['within_trained_envelope'] else 'NO':4s} "
              f"{r['final']*100:6.1f}% {r['worst']*100:6.1f}% {r['sum_squared']:8.4f}")

    ref = legal[8]["sum_squared"]
    print(f"\n  against 8 evaluations:")
    for nfe, r in legal.items():
        print(f"    {nfe:2d}: {r['sum_squared']/ref:6.3f}x")
    print(f"\n  illegal: {sorted(k for k, v in rows.items() if not v['legal'])}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "question": "what does each legal PDD step count buy on the shift-12 grid",
            "method": "partition_bounds over the node's own envelope_partition; "
                      "per-step sigma fractions of the full range. Schedule "
                      "geometry only -- nothing rendered, no perceptual claim.",
            "grid": GRID, "trained_width": TRAINED, "shift": SHIFT,
            "ladder": rows,
        }, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
