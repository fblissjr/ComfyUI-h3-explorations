#!/usr/bin/env python3
"""Is token_aug's run-to-run variation a selection flip or accumulation order?

Stage 2 of the token routing plan found that with bit-identical inputs, the
`token_aug` arm's output on one captured block varies between runs while the
other four captured blocks and every plain arm are bitwise identical
(`data/results/2026-09-08_token_aug_stage2_reproduction.json`). That says the
variation belongs to the `token_aug` code path. It does NOT say what varies,
and two very different mechanisms produce the same symptom:

  a SELECTION flip     the set of extra tokens admitted changes between
                       launches. comfy_kitchen's `sol_attn` docstring says
                       this cannot happen -- "whole histogram bins only, so
                       the set never depends on scheduling" -- so if it is
                       this, that claim is wrong as written.

  ACCUMULATION order   the set is fixed and the exactly-attended contributions
                       are summed in a different order, which float addition
                       does not commute over. Ordinary, contradicts nothing,
                       and present in many CUDA kernels by design.

## The discriminator

Not the number of distinct outputs. That was the first version of this file
and it is the wrong statistic: hashing the whole tensor makes ANY differing
element a distinct output, so one flipped value and a global drift both read
as "30 distinct over 30 runs". It classified a case with a max delta of 423
as reduction noise.

The statistic is the DISTRIBUTION of the deltas:

  accumulation order   many elements differ, each by a reduction-scale amount.
                       Summing the same contributions in a different order
                       cannot produce a large change in any one output.

  a changed SET        few elements differ, and some differ enormously,
                       because a different token contributing means a
                       different answer rather than a rounder one.

So: what fraction of rows moved at all, what fraction moved by more than the
output's own scale, and how the largest delta compares to the mean magnitude.

## What this cannot settle

It cannot prove a selection flip, only make one much more or less likely: a
kernel could in principle vary its accumulation order across a small fixed set
of schedules and produce clustering too. Reading a cluster count as a set
count is the error to avoid. And it says nothing about which set is right --
`bench/measure_sol_exact_variants.py` grades accuracy, this only asks whether
the answer is stable.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def load_cell(path):
    """q, k, v as the kernel takes them: capture writes [B, H, S, D]."""
    import torch
    d = torch.load(path, map_location="cuda", mmap=False)
    q, k, v = (d[n].permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
               for n in ("q", "k", "v"))
    return q, k, v, {"block": d.get("block"), "step": d.get("step"),
                     "sigma": d.get("sigma"), "seq_len": q.shape[1],
                     "heads": q.shape[2]}


def run_many(q, k, v, repeats, token_aug, tau):
    """Delta distribution over `repeats` identical calls, against the first.

    Reduced per token row rather than kept whole: the tensors are ~1.5 GiB in
    bf16 at H3's real sequence length and holding two of them plus a float32
    difference is an OOM on a 24 GiB card with a server resident.
    """
    import comfy_kitchen as ck
    import torch
    seen, order = collections.Counter(), []
    first = None
    max_delta = 0.0
    rows_diff = rows_big = 0
    scale = 0.0
    for _ in range(repeats):
        kw = {"tau": tau}
        if token_aug:
            kw["token_aug"] = token_aug
        out = ck.sol_attn(q, k, v, **kw)
        raw = out.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()
        digest = hashlib.sha256(raw).hexdigest()[:16]
        if digest not in seen:
            order.append(digest)
        seen[digest] += 1
        if first is None:
            first = out.clone()
            scale = float(out.abs().float().mean())
        else:
            per_row = (out - first).abs().amax(dim=(0, 2, 3))
            max_delta = max(max_delta, float(per_row.max()))
            rows_diff = max(rows_diff, int((per_row > 0).sum()))
            rows_big = max(rows_big, int((per_row > scale).sum()))
            del per_row
        del out
    n_rows = first.shape[1]
    return seen, order, {
        "max_abs_delta": max_delta,
        "mean_abs_output": scale,
        "rows_differing": rows_diff,
        "rows_differing_by_more_than_mean_output": rows_big,
        "token_rows": n_rows,
        "frac_rows_differing": rows_diff / n_rows,
    }


def verdict(stats):
    """Classify on the delta distribution, not on how many hashes differed."""
    if stats["rows_differing"] == 0:
        return "deterministic", "no element moved over any run"
    big = stats["max_abs_delta"] / max(stats["mean_abs_output"], 1e-9)
    if big < 0.01:
        return ("accumulation-shaped",
                "every delta is small against the output's own scale, which is "
                "what summing the same contributions in a different order "
                "produces")
    return ("NOT accumulation order",
            f"the largest delta is {big:.1f}x the mean output magnitude on "
            f"{stats['frac_rows_differing']:.3%} of rows. Reordering a sum "
            f"cannot do that; a different set of contributions can. This does "
            f"NOT identify what changed -- the admitted set is not observable "
            f"from here -- it rules out the benign explanation")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, help="one qkv_*.pt from a capture")
    ap.add_argument("--control-cell", help="a cell expected to be deterministic")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--token-aug", type=int, default=64)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--out")
    args = ap.parse_args()

    arms = {}
    q, k, v, meta = load_cell(args.cell)
    print(f"subject cell: block {meta['block']} step {meta['step']} "
          f"seq {meta['seq_len']} heads {meta['heads']}")

    # The plain arm first: if THIS varies, the subject proves nothing about
    # token_aug, because the variation would not be the knob's.
    for name, aug in (("plain_token_aug_0", 0), (f"token_aug_{args.token_aug}", args.token_aug)):
        counts, order, stats = run_many(q, k, v, args.repeats, aug, args.tau)
        shape, why = verdict(stats)
        arms[name] = {"repeats": args.repeats, "distinct_outputs": len(counts),
                      **stats, "shape": shape, "reading": why}
        print(f"  {name:22s} distinct {len(counts):3d}/{args.repeats}  "
              f"rows moved {stats['rows_differing']:6d}/{stats['token_rows']}  "
              f"max|d| {stats['max_abs_delta']:.3g} vs mean|out| "
              f"{stats['mean_abs_output']:.3g}  -> {shape}")

    if args.control_cell:
        cq, ck_, cv, cmeta = load_cell(args.control_cell)
        counts, order, stats = run_many(cq, ck_, cv, args.repeats, args.token_aug, args.tau)
        shape, why = verdict(stats)
        arms["control_other_block"] = {
            "block": cmeta["block"], "repeats": args.repeats,
            "distinct_outputs": len(counts), **stats, "shape": shape,
            "reading": "a block stage 2 found bitwise reproducible. If this is "
                       "not deterministic here, the harness is measuring "
                       "something other than what stage 2 measured."}
        print(f"  control block {cmeta['block']:<9} distinct {len(counts):3d}/{args.repeats}  "
              f"rows moved {stats['rows_differing']}  -> {shape}")

    record = {
        "what": "whether token_aug's run-to-run variation looks like a selection "
                "flip or accumulation order, by counting distinct outputs over "
                "identical repeated calls",
        "produced_by": "bench/probe_token_aug_determinism.py",
        "subject_cell": Path(args.cell).name, "subject": meta,
        "tau": args.tau, "arms": arms,
        "cannot_settle": [
            "proof of a selection flip. Clustering is consistent with a small "
            "candidate space and also with a kernel that varies accumulation "
            "across a few fixed schedules; reading a cluster count as a set "
            "count is the error to avoid.",
            "which set is correct. This asks only whether the answer is stable.",
        ],
    }
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
