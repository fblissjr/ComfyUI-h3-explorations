#!/usr/bin/env python3
"""How small can the `token_aug` nondeterminism repro get?

`data/results/2026-09-08_token_aug_selection_structure.json` shows one captured
DiT block whose `sol_attn` output varies between identical calls when
`token_aug` is on. The capture cell that carries it is gigabytes, so nobody
outside this machine can run it. This asks what the smallest artifact is that
still reproduces, in the order that would make a repro most useful to someone
who does not have our captures:

  random tensors      best case. Nothing to ship at all -- the repro is a
                      script. Expected to fail: the instability should need a
                      particular score distribution, and random q/k does not
                      have one.

  one head            the selection is per (batch, head), and the split count
                      the kernel picks is a function of sequence length only
                      (`sol_token_splits` voids B and H), so slicing to one
                      head should not change what that head does.

  one head, shortened not attempted here. Sequence length changes the tile
                      count, the split count and the histogram's bin
                      population, so a shortened sequence is a different
                      question rather than a smaller version of this one.

Every arm reports whether it reproduced, so a negative result is data: "random
does not reproduce" is worth knowing, because it says the trigger is in the
activations rather than in the code path alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def kernel_build():
    import importlib.metadata
    try:
        return importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        return "<no dist metadata>"


def varies(q, k, v, repeats, token_aug, tau):
    """Does the output move over identical calls? Returns the peak delta.

    Reduced per token row against the first launch, which is enough here: this
    file asks whether a slice still reproduces, not what shape the movement
    has. `probe_token_aug_selection_structure.py` owns the shape.
    """
    import comfy_kitchen as ck
    import torch
    first = None
    peak = 0.0
    rows = 0
    scale = 0.0
    for _ in range(repeats):
        out = ck.sol_attn(q, k, v, tau=tau, **({"token_aug": token_aug} if token_aug else {}))
        if first is None:
            first = out.clone()
            scale = float(out.abs().float().mean())
        else:
            per_row = (out.float() - first.float()).abs().amax(dim=(0, 2, 3))
            peak = max(peak, float(per_row.max()))
            rows = max(rows, int((per_row > 0).sum()))
            del per_row
        del out
        torch.cuda.empty_cache()
    return {"max_abs_delta": peak, "rows_differing": rows, "launches": repeats,
            "mean_abs_output": scale, "reproduces": rows > 0,
            "reading": ("moved" if rows else
                        f"not seen in {repeats} launches, which is not the same "
                        f"as stable: an intermittent race can sit out a run")}


def unstable_heads(q, k, v, token_aug, tau):
    """Which heads move at all, so the slice is taken from one that does."""
    import comfy_kitchen as ck
    import torch
    kw = {"tau": tau, "token_aug": token_aug}
    a = ck.sol_attn(q, k, v, **kw)
    first = a.clone()
    del a
    moved = set()
    for _ in range(4):
        out = ck.sol_attn(q, k, v, **kw)
        H = out.shape[2]
        for h0 in range(0, H, 4):
            h1 = min(H, h0 + 4)
            d = (out[0, :, h0:h1, :].float() - first[0, :, h0:h1, :].float()).abs().amax(dim=(0, 2))
            moved |= {h0 + i for i, m in enumerate(d.tolist()) if m > 0}
            del d
        del out
        torch.cuda.empty_cache()
    del first
    torch.cuda.empty_cache()
    return sorted(moved)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--token-aug", type=int, default=64)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--save-slice", help="write the reproducing slice here. Prefer "
                    "verify_token_aug_repro_shapes.py: a slice that moves during "
                    "this in-process sweep need not move in a cold process")
    ap.add_argument("--out")
    args = ap.parse_args()

    import torch
    from probe_token_aug_selection_structure import load_cell

    q, k, v, meta = load_cell(args.cell)
    print(f"cell: block {meta['block']} step {meta['step']} seq {meta['seq_len']} "
          f"heads {meta['heads']}\n")
    arms = {}

    # Random first: if this reproduces, the repro needs no data at all.
    g = torch.Generator(device="cuda").manual_seed(0)
    rq, rk, rv = (torch.randn(q.shape, generator=g, device="cuda", dtype=torch.bfloat16)
                  for _ in range(3))
    arms["random_full_shape"] = varies(rq, rk, rv, args.repeats, args.token_aug, args.tau)
    print(f"  random, full shape      reproduces={arms['random_full_shape']['reproduces']}")
    del rq, rk, rv
    torch.cuda.empty_cache()

    heads = unstable_heads(q, k, v, args.token_aug, args.tau)
    arms["unstable_heads"] = {"count": len(heads), "of": meta["heads"],
                              "first_few": heads[:8]}
    print(f"  heads that move         {len(heads)} of {meta['heads']}")
    if not heads:
        print("  nothing moved; no slice to take")
        return

    # Head-count sweep, and it is a control as much as a minimisation. The
    # selection is per (batch, head) and `sol_token_splits` voids B and H, so a
    # slice does not change what any one head computes -- it changes how many
    # CTAs are resident. If the instability needs a head count, it needs
    # concurrency, and needing concurrency is the signature of a scheduling
    # race rather than of a wrong-but-deterministic computation.
    H = meta["heads"]
    keep = heads[0]
    others = [i for i in range(H) if i != keep]
    counts = [n for n in (1, 2, 4, 8, 16, 32, H) if n <= H]
    sweep = {}
    smallest = None
    for n in counts:
        idx = torch.tensor([keep] + others[:n - 1], device="cuda")
        sq, sk, sv = (t.index_select(2, idx).contiguous() for t in (q, k, v))
        r = varies(sq, sk, sv, args.repeats, args.token_aug, args.tau)
        r["unstable_heads_included"] = len([i for i in [keep] + others[:n - 1] if i in heads])
        sweep[f"heads_{n}"] = r
        print(f"  {n:2d} head(s)               reproduces={r['reproduces']}  "
              f"rows {r['rows_differing']}  "
              f"(unstable heads in slice: {r['unstable_heads_included']})")
        if r["reproduces"] and smallest is None:
            smallest = (n, sq, sk, sv)
        else:
            del sq, sk, sv
        torch.cuda.empty_cache()
    arms["head_count_sweep"] = sweep
    arms["smallest_reproducing_head_count"] = smallest[0] if smallest else None
    del q, k, v
    torch.cuda.empty_cache()
    if smallest is None:
        print("\n  nothing below the full tensor reproduced")
        sq = sk = sv = None
    else:
        _, sq, sk, sv = smallest
        h = keep

    if sq is not None:
        # Random at the reproducing slice's shape: separates "this many heads
        # is enough" from "any tensor of this shape is unstable", which would
        # be a different bug.
        g2 = torch.Generator(device="cuda").manual_seed(1)
        rq, rk, rv = (torch.randn(sq.shape, generator=g2, device="cuda", dtype=torch.bfloat16)
                      for _ in range(3))
        arms["random_slice_shape"] = varies(rq, rk, rv, args.repeats, args.token_aug, args.tau)
        print(f"  random, slice shape     reproduces={arms['random_slice_shape']['reproduces']}")
        del rq, rk, rv
        torch.cuda.empty_cache()

    if sq is not None:
        # Length sweep, to see whether the artifact can get smaller still.
        # Truncating changes the tile count, the split count and the histogram
        # population, so a shorter sequence is a DIFFERENT input rather than a
        # smaller version of this one; a negative says nothing about the
        # original and a positive is just a cheaper repro that shares a cause.
        lens = [n for n in (65536, 32768, 16384, 8192) if n < sq.shape[1]]
        lsweep = {}
        for n in lens:
            tq, tk, tv = (t[:, :n, :, :].contiguous() for t in (sq, sk, sv))
            r = varies(tq, tk, tv, args.repeats, args.token_aug, args.tau)
            lsweep[f"len_{n}"] = r
            print(f"  len {n:6d}              reproduces={r['reproduces']}  "
                  f"rows {r['rows_differing']}")
            if r["reproduces"]:
                sq, sk, sv = tq, tk, tv
            else:
                del tq, tk, tv
            torch.cuda.empty_cache()
        arms["length_sweep"] = lsweep
        arms["saved_slice_shape"] = list(sq.shape)

    if args.save_slice and sq is not None:
        payload = {"q": sq.cpu(), "k": sk.cpu(), "v": sv.cpu(),
                   "token_aug": args.token_aug, "tau": args.tau,
                   "source_block": meta["block"], "source_step": meta["step"],
                   "source_head": h}
        torch.save(payload, args.save_slice)
        size = Path(args.save_slice).stat().st_size
        arms["saved_slice"] = {"bytes": size, "path_is_the_owner's": True}
        print(f"\n  wrote slice: {size / 1e6:.0f} MB")

    record = {
        "what": "the smallest artifact that still reproduces token_aug's "
                "run-to-run nondeterminism, so the defect can be reported "
                "without shipping a capture cell",
        "produced_by": "bench/minimise_token_aug_repro.py",
        "kernel_build": kernel_build(),
        "source_cell": Path(args.cell).name,
        "token_aug": args.token_aug, "tau": args.tau, "repeats": args.repeats,
        "arms": arms,
        "superseded_for_the_artifact_by": "data/results/2026-09-08_token_aug_repro_shapes.json",
        "cannot_settle": [
            "that a non-reproducing arm proves the trigger absent. Each arm is "
            "a fixed number of launches; instability that is rarer than that "
            "reads as absent here.",
            "WHICH SHAPE TO SHIP AS A REPRO. Every arm here runs inside ONE "
            "process, after other launches, with a warm allocator. The effect "
            "is an occupancy-sensitive race, and the first slice this file "
            "picked failed when run cold. bench/verify_token_aug_repro_shapes.py "
            "re-tests candidates in fresh processes and owns the artifact; "
            "what stands here is the head sweep and the random arms.",
        ],
    }
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
