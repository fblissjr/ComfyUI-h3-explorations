#!/usr/bin/env python3
"""What the calibration gate would find at production length, and why chunking will not get it there.

`bench/analyze_sol_error.py::calibrate_against_oracle` compares that file's
`eager_sol_reference` against the vendored upstream oracle at t <= 2001 and
infers agreement at production S = 98498. Its docstring named chunking the
oracle as "the one change that would make this a real production gate", and
`docs/roadmap.md` carried that as the highest-value remaining item on the
error-decomposition line.

This probe was written to price that change and refuted it instead. Two results,
both reproducible by running this file.

**The gate was never limited to t <= 2001.** The oracle refuses above a score-
matrix budget rather than at a length, so it runs to roughly 384 blocks
untouched -- twelve times the length the gate actually uses. Nothing had to be
chunked to find that out; the ceiling was a chosen constant read as a limit.

**Run there, the gate goes red, and not for a defect.** Agreement holds to
~3e-04 out to 192 blocks and jumps to ~1e-02 at 256. The jump is not numerical
drift accumulating with length: it is a handful of whole query blocks, always an
exact multiple of 64 rows, whose routing decision lands on opposite sides of the
threshold in two float32 reduction orders. Their identity is reseeded by the
input -- 2-4 blocks of 257 across seeds, never the same ones -- which is the
signature of a tie broken differently, not of two algorithms disagreeing.

So a chunked oracle at S = 98498 would compare 1539 blocks instead of 257, make
flips a certainty, and report red while both implementations are correct. That
is this repo's worst category of check, and the pressure it would create is to
raise `--tol`, which the gate's own refusal text forbids in as many words.

**What would actually close the gap** is a different instrument, not a longer
one: compare the two routing masks rather than the two outputs. At production S
that is a 1539x1539 boolean per head, needs no chunking at all, and separates a
tie flip from a real disagreement by reporting each flipped block's margin --
which output relative L2 cannot do at any length.

Read this beside what the gate still cannot see even so. It runs at head
dimension 64 with one head on `torch.randn`; production is head dimension 128,
56 heads, real activations. Length was never the only axis inferred across, and
closing it alone would have made the gate feel like a production gate without
being one.

## Running it

    python bench/probe_oracle_gate_scaling.py --out bench/results/2026-08-19_oracle_gate_scaling.json

No GPU needed for the oracle side and no capture: the inputs are synthetic, the
same as the gate's own. Peak host memory is set by the largest length attempted.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
BLOCK = 64


def load():
    spec = importlib.util.spec_from_file_location(
        "_ase", REPO / "bench" / "analyze_sol_error.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.insert(0, str(REPO / "bench"))
    import _sol_attn_reference as oracle
    return mod, oracle


def compare(mod, oracle, t_len, tau, seed, head_dim):
    """(rel_l2, flipped query blocks, rows above 1%) or None if the oracle refuses."""
    torch.manual_seed(seed)
    q, k, v = (torch.randn(1, t_len, 1, head_dim) for _ in range(3))
    qh, kh, vh = (x.permute(0, 2, 1, 3).contiguous() for x in (q, k, v))
    try:
        ora = oracle.sol_attn(q, k, v, tau=tau, centroid_tail=True)
    except RuntimeError as exc:
        return None, str(exc)[:120]
    ora = ora.permute(0, 2, 1, 3).float()
    mine = mod.eager_sol_reference(qh, kh, vh, tau=tau).float()

    rel_l2 = mod.rel_l2_error(mine, ora)
    row = (mine - ora)[0, 0].norm(dim=-1) / (ora[0, 0].norm(dim=-1) + 1e-12)
    n = (t_len + BLOCK - 1) // BLOCK
    flipped = [i for i in range(n)
               if row[i * BLOCK:(i + 1) * BLOCK].numel()
               and row[i * BLOCK:(i + 1) * BLOCK].max() > 0.01]
    return {
        "t": t_len, "blocks": n, "ragged_tail": t_len % BLOCK, "seed": seed,
        "rel_l2": round(rel_l2, 8),
        "flipped_query_blocks": flipped,
        "rows_above_1pct": int((row > 0.01).sum()),
        # A tie flip takes whole query blocks. If this is not an exact multiple
        # of the block size, the divergence is NOT a routing flip and the
        # reading below does not apply.
        "rows_equal_blocks_times_64": int((row > 0.01).sum()) == len(flipped) * BLOCK,
    }, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lengths", default="2001,4098,8194,12290,16386,20482,24578,32770",
                    help="the last one is expected to be refused; that is the point")
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--flip-length", type=int, default=16386,
                    help="length at which to run every seed, for the flip signature")
    ap.add_argument("--tau", type=float, default=1.3)
    ap.add_argument("--head-dim", type=int, default=64,
                    help="the gate's own value. Production is 128; this probe does "
                         "not close that axis and says so.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mod, oracle = load()
    lengths = [int(x) for x in args.lengths.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print(f"SCALING at seed {seeds[0]}, tau {args.tau}, head dim {args.head_dim}\n")
    print(f"  {'t':>8}{'blocks':>8}{'rel_l2':>12}{'flipped':>9}")
    scaling, refusal = [], None
    for t_len in lengths:
        row, err = compare(mod, oracle, t_len, args.tau, seeds[0], args.head_dim)
        if row is None:
            refusal = {"t": t_len, "message": err}
            print(f"  {t_len:>8}  oracle refuses: {err}")
            break
        scaling.append(row)
        print(f"  {t_len:>8}{row['blocks']:>8}{row['rel_l2']:>12.6f}"
              f"{len(row['flipped_query_blocks']):>9}")

    print(f"\nFLIP SIGNATURE at t={args.flip_length}, every seed\n")
    print(f"  {'seed':>5}{'rel_l2':>12}{'flipped blocks':>16}   which")
    flips = []
    for seed in seeds:
        row, err = compare(mod, oracle, args.flip_length, args.tau, seed, args.head_dim)
        if row is None:
            continue
        flips.append(row)
        print(f"  {seed:>5}{row['rel_l2']:>12.6f}"
              f"{len(row['flipped_query_blocks']):>16}   {row['flipped_query_blocks']}")

    whole = all(f["rows_equal_blocks_times_64"] for f in flips)
    ident = {tuple(f["flipped_query_blocks"]) for f in flips}
    print(f"\n  every divergence is whole query blocks: {whole}")
    print(f"  the same blocks every seed: {len(ident) == 1}")
    if whole and len(ident) > 1:
        print("  -> reseeded tie flips, not an algorithmic divergence. Output relative\n"
              "     L2 cannot tell those apart, which is why length is the wrong axis\n"
              "     to extend this gate along.")

    record = {
        "measured": "2026-08-19",
        "produced_by": "bench/probe_oracle_gate_scaling.py",
        "what": "how analyze_sol_error's calibration gate behaves beyond the length it runs at",
        "tau": args.tau,
        "head_dim": args.head_dim,
        "axes_not_closed": ["head dim 128", "56 heads", "real activations",
                            "production S = 98498"],
        "argv": sys.argv[1:],
        "oracle_refusal": refusal,
        "scaling": scaling,
        "flip_signature": flips,
        "divergence_is_whole_query_blocks": whole,
        "flipped_blocks_identical_across_seeds": len(ident) == 1,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
        print(f"\nwrote {args.out}")
    else:
        json.dump(record, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
