#!/usr/bin/env python3
"""What does a dense block actually COST, in seconds, on this box?

Every figure this repo had for `dense_blocks` was arithmetic over the
2026-08-16 dense pair, assuming uniform per-block cost -- an assumption the
block-0 routing outlier refutes on its face (block 0 keeps ~29.8% of blocks
exact against ~11% deeper, so forcing it dense buys back far less time than an
average block would). This measures it instead.

**Wall clock is one of the few things that can be compared arm to arm here.**
CLAUDE.md's different-sample rule is about what a clip looks like: two arms
that differ numerically draw different samples, so comparing their pixels
answers nothing. It says nothing against comparing how long they took, which is
a property of the work done rather than of the sample drawn.

## What is controlled

Same graph, same seed, same server process, same step count; the ONLY thing
that varies is the `dense_blocks` string on the Sol node. Arms run
back-to-back and the model stays resident, so no arm pays a load the others
did not. A warm-up arm runs first and is discarded, because the first render
after a model load pays staging the others do not -- that is the defect that
invalidated `D_start0`'s timing on 2026-08-27
(`docs/research/pdd/queued_arms.md`).

## What it cannot tell you

Nothing about quality. A cheaper arm is not a better one, and the whole reason
to spend these seconds is a fidelity question this cannot see -- see
`bench/probe_block_propagation.py` for that half.

    python bench/time_dense_blocks.py --specs '' 0-1 0-2,32 0-5,48-49
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))

from probe_block_propagation import BASE_GRAPH, submit          # noqa: E402

OUT = REPO / "bench" / "results" / "2026-08-29_dense_block_cost.json"
PREFIX = "latents/h3_dense_block_cost/arm"


def build(base: dict, spec: str, steps: int, seed: int, tag: str) -> tuple[dict, str]:
    g = json.loads(json.dumps(base))

    def one(*classes):
        """The single node matching any of `classes`.

        Takes alternatives because the Sol node id changed on
        2026-08-30 and this looked for the vendored name alone until
        2026-08-31 -- against a regenerated graph it found zero and
        exited, which at least failed loudly rather than silently.
        """
        hits = [k for k, v in g.items()
                if v.get("class_type") in classes]
        if len(hits) != 1:
            raise SystemExit(f"expected one of {classes}, found {len(hits)}")
        return hits[0]

    sol, sched, sampler, noise = (one("MiniMaxH3SolAttn", "SolAttnMiniMax"), one("BasicScheduler"),
                                  one("KSamplerSelect"), one("RandomNoise"))
    g[sched]["inputs"]["steps"] = steps
    g[sampler]["inputs"]["sampler_name"] = "euler"
    g[noise]["inputs"]["noise_seed"] = seed
    g[sol]["inputs"]["dense_blocks"] = spec
    # The SHIPPED PDD window, not the propagation probe's single-step one: this
    # is a cost question about the configuration that actually runs, where Sol
    # takes 2 of 4 steps and a dense block is charged on both.
    g[sol]["inputs"]["start_percent"] = 0.2
    g[sol]["inputs"]["end_percent"] = 0.74

    sca = one("SamplerCustomAdvanced")
    for cls in ("VAEDecode", "VAEDecodeAudio", "VHS_VideoCombine"):
        for k in [k for k, v in g.items() if v.get("class_type") == cls]:
            del g[k]
    g["900"] = {"class_type": "LTXVSeparateAVLatent",
                "inputs": {"av_latent": [sca, 0]}}
    g["901"] = {"class_type": "SaveLatent",
                "inputs": {"samples": ["900", 0],
                           "filename_prefix": f"{PREFIX}_{tag}_video"}}
    g["902"] = {"class_type": "SaveLatent",
                "inputs": {"samples": ["900", 1],
                           "filename_prefix": f"{PREFIX}_{tag}_audio"}}
    return g, sca


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", default=["", "0-1", "0-2,32", "0-5,48-49"],
                    help="dense_blocks strings to time; '' is Sol everywhere")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=730451892)
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--repeats", type=int, default=1,
                    help="times through the whole arm list; >1 gives a spread "
                         "per arm rather than one draw")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    base = json.loads(BASE_GRAPH.read_text())
    times: dict[str, list[float]] = {s: [] for s in args.specs}

    # Discard the first render: the model load is paid by whoever goes first,
    # and charging it to an arm invalidates the comparison.
    #
    # **On its own SEED, not `specs[0]`'s.** ComfyUI caches by node inputs, so a
    # warm-up identical to the first timed arm makes that arm a cache hit --
    # which is exactly what happened on the first run of this script: the
    # reference came back in 3.0 s and the per-block cost was off by more than
    # an order of magnitude. Every arm below likewise gets its own seed per
    # repeat, so no repeat can serve a cached copy of the one before.
    print("  warm-up (discarded) ...", end="", flush=True)
    warm_g, sca = build(base, args.specs[0], args.steps, args.seed - 1, "warmup")
    _, meta = submit(warm_g, args.host, args.timeout, cache_watch=sca)
    print(f" {(meta['server_ms'] or 0) / 1000:6.1f}s")

    for rep in range(args.repeats):
        seed = args.seed + rep
        for spec in args.specs:
            tag = (spec or "none").replace(",", "_").replace("-", "to")
            print(f"  rep {rep + 1}  dense_blocks={spec or '(none)':<12s} ...",
                  end="", flush=True)
            arm_g, sca = build(base, spec, args.steps, seed, tag)
            _, meta = submit(arm_g, args.host, args.timeout, cache_watch=sca)
            if meta["server_ms"] is None:
                raise SystemExit("\nthe server reported no execution timestamps; "
                                 "refusing to time against this loop's clock, "
                                 "which is quantised to the poll interval")
            dt = meta["server_ms"] / 1000.0
            times[spec].append(dt)
            # A cached arm is not a measurement. Refuse rather than average it
            # in -- silently fast is the failure mode this script already had.
            if meta["cached"]:
                raise SystemExit(
                    f"\narm {spec!r} was served from ComfyUI's cache "
                    f"({dt:.1f}s). Its inputs matched an earlier run, so this "
                    f"is not a render. Vary the seed.")
            print(f" {dt:6.1f}s")

    print(f"\n  {args.steps} steps, Sol window 0.2-0.74 (the shipped PDD window, "
          f"2 sparse steps of {args.steps})\n")
    ref_spec = args.specs[0]
    ref_blocks = len(_expand(ref_spec))
    print(f"    dense_blocks    n  best (s)   vs {ref_spec or '(none)'}")
    ref = min(times[ref_spec]) if times[ref_spec] else None
    rows = []
    for spec in args.specs:
        ts = times[spec]
        best = min(ts)
        n = len(_expand(spec))
        delta = best - ref if ref is not None else 0.0
        rows.append({"dense_blocks": spec, "blocks": n, "seconds": ts,
                     "best": best, "delta_vs_first": delta})
        print(f"    {spec or '(none)':<14s} {n:2d}  {best:7.1f}   {delta:+6.1f}")

    # Per-block marginal cost, which is the number the recipe argument needs.
    #
    # Divided by the block count DIFFERENCE against the reference arm, not by
    # the widest arm's absolute count. Those coincide only when `--specs[0]`
    # names zero blocks, which is the default and was the only shape this was
    # ever run in -- so with any custom `--specs` whose first entry has blocks,
    # the old form understated the cost silently (0.75 s/block instead of 1.01
    # on `--specs 0-1 0-2,32 0-5,48-49`).
    span = [r for r in rows if r["blocks"] > ref_blocks]
    if span and ref is not None:
        widest = max(span, key=lambda r: r["blocks"])
        extra = widest["blocks"] - ref_blocks
        print(f"\n    ~{widest['delta_vs_first'] / extra:.2f} s per dense block "
              f"at {args.steps} steps, from {widest['dense_blocks']!r} "
              f"({widest['blocks']}) against {ref_spec or '(none)'} "
              f"({ref_blocks}) -- {extra} block(s) apart")
    elif ref is not None:
        print("\n    no arm has more blocks than the reference, so there is no "
              "marginal cost to report")

    if args.write:
        OUT.write_text(json.dumps({
            "measured": "2026-08-29",
            "produced_by": "bench/time_dense_blocks.py",
            "what": "wall-clock cost of dense_blocks, same graph and seed, only "
                    "that string varying",
            "base_graph": BASE_GRAPH.name,
            "steps": args.steps, "seed": args.seed,
            "sol_window": {"start_percent": 0.2, "end_percent": 0.74},
            "reference_arm": {"dense_blocks": ref_spec, "blocks": ref_blocks,
                              "note": "delta_vs_first is measured against THIS "
                                      "arm, not against zero blocks"},
            "rows": rows,
            "caveats": [
                "Wall clock only. Says nothing about quality.",
                "First render discarded as warm-up; the model stays resident "
                "across arms.",
                "One box, one canvas, one length. Per-block cost is not "
                "uniform -- block 0 routes ~29.8% exact against ~11% deeper, "
                "so it is the cheapest block to force dense.",
            ],
        }, indent=1) + "\n")
        print(f"\n  wrote {OUT.relative_to(REPO)}")
    return 0


def _expand(spec: str) -> set[int]:
    """Block count for a spec, via the same parser the node uses.

    Shared rather than reimplemented -- a local "count the commas" would be a
    second implementation of a syntax where `-1` means 49 and `0-2` means three
    blocks, and the two would disagree the first time someone used a range.
    """
    if not spec.strip():
        return set()
    # `parse_blocks` left the Sol node for `block_spec.py`; it was reached
    # through `vendor/sol_attn_minimax.py` until 2026-08-31, which is not
    # the running node and carries its own older copy.
    from _live_sol import block_spec
    return set(block_spec().parse_blocks(spec, 50))


if __name__ == "__main__":
    raise SystemExit(main())
