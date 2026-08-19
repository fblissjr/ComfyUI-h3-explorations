#!/usr/bin/env python3
"""Per-head routed density across the capture grid: where the routing structure is.

`bench/results/2026-08-18_routing_density.json` established that the per-block
and per-step axes are flat -- 28-30% kernel density at tau 1.3 everywhere but
block 0 -- so a tau schedule along either axis has nothing to exploit. It named
the per-head axis as the remaining structure and could not measure it, because
`bench/analyze_routing.py` averages over its sampled heads by construction
(`densities()` returns two scalars).

This sweeps the same grid and keeps the head axis. It answers one question:
**is there a head-shaped operating point that a single tau is leaving on the
table**, and how much.

## Why the head axis specifically

Two independent reasons, and the second only became true on 2026-08-19.

`bench/analyze_sol_error.py` found per-head quant/sparsity ratios spanning 24%
to 1695% within one block at one step, with one head's sparsity error above 1.0
while its neighbours sat near 0.1. That is the largest spread in any axis this
repo has measured -- but it is an error spread, and error is not density. This
script measures the density side of the same axis, so the two can be joined.

And a per-head threshold is no longer hypothetical. comfy-kitchen's `sol_attn`
takes `tau: float`, one scalar for the call, so a per-head operating point
cannot be expressed there at all. NVLabs' SM89 kernel takes a `(B, blocks, H)`
threshold tensor and indexes it per head in the mainloop
(`coderef/Sana/techniques/sparse_backends/sol_attn/preprocess.py::_compute_exact_threshold`,
consumed at `coderef/Sana/techniques/sparse_backends/sol_attn/sm89/mainloop.py:260`).
So on that kernel a per-head tau is a preprocess change, not an ABI change.

## What this is and is not

Every caveat in `analyze_routing.py`'s docstring applies unchanged, because the
routing arithmetic here IS that module's -- `block_stats`, `exact_mask` and
`forced_masks` are imported, not reimplemented. In particular this is the eager
routing rule rather than the CUDA kernel's INT8 arithmetic, it is not a speed
measurement, and it is not valid at a length the capture was not taken at.

**Raster ordering only.** The ordering question is `analyze_routing.py`'s and it
is answered there; mixing it in here would produce a four-way table whose head
axis nobody reads. The aggregate rows this emits are directly comparable to that
tool's raster row on the same capture, head set and tau, which is how the reuse
is verified -- see `--verify`.

**A spread is not headroom, and the record says which.** A per-head density
spread inside one cell is the router already responding to per-head content --
the threshold is derived from per-head statistics, so heads *should* differ.
What would make a per-head tau exploitable is that the spread **persists**: that
a head which routes densely at one point routes densely at the next. So the
record carries rank correlations of the per-head ordering between cells, split
two ways -- same transformer block across sampling steps, and across transformer
blocks at one step -- because those two answered differently on first run and
only one of them supports a static per-head table.

**The step axis is as wide as the capture, which is narrow.** A capture holds
the steps it was taken at, so "across sampling steps" is a correlation over
those pairs and nothing more -- two steps gives one pair per block. Read the
`pairs` count in the record before saying a result holds along the trajectory;
what it supports until a denser capture exists is that it holds *between the
captured steps*.

**The head set is the tool's, not a fresh sample.** Head order comes from
`torch.randperm` under `manual_seed(0)`, the same construction
`analyze_routing.py` uses, so `--heads 8` here is the same eight heads it
samples. At `--heads 0` every head is measured and the sampling question does
not arise.

## Running it

    python bench/sweep_routing_density.py \\
        --capture $H3_CAPTURE_ROOT/2026-08-18_ref3_362f_1024x768_ref2va \\
        --canvas 1024x768 --length 362 --heads 0 \\
        --out bench/results/2026-08-19_routing_density_per_head.json

No GPU and no server. Needs the capture and an installed `comfy_kitchen`
carrying `sol_attn`, for the same reason `analyze_routing.py` does.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
BLOCK = 64
_NAME = re.compile(r"_b(\d+)_s(\d+)\.pt$")


def load_analyze_routing():
    """The routing arithmetic, imported rather than restated.

    A second transcription of the threshold rule is a second thing to keep in
    step with `sol_attn_preprocess.cu`, and the whole point of the aggregate
    cross-check below is that this script and `analyze_routing.py` are running
    one implementation.
    """
    spec = importlib.util.spec_from_file_location(
        "_ar", REPO / "bench" / "analyze_routing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def per_head_densities(ar, q, k, order, start, stop, tau, heads, pool,
                       sink_kv, sink_q):
    """[(head, ordering-effect density, kernel density)], one row per head.

    The body is `analyze_routing.densities()` with the averaging removed. Every
    call it makes is that module's.
    """
    rows = []
    for head in heads:
        centroid, kcc, kcvar, n, D = ar.block_stats(
            q, k, order, start, stop, head, pool)
        thresholded = ar.exact_mask(centroid, kcc, kcvar, tau, D ** -0.5)
        forced, free = ar.forced_masks(n, sink_kv, sink_q)
        rows.append((
            int(head),
            float((thresholded & free).sum()) / float(free.sum()),
            float((thresholded | forced).sum()) / float(n * n),
        ))
    return rows


def summarise(rows):
    eff = sorted(r[1] for r in rows)
    ker = sorted(r[2] for r in rows)
    mid = len(ker) // 2
    return {
        "ordering_effect_density_pct": round(100 * sum(eff) / len(eff), 4),
        "kernel_density_pct": round(100 * sum(ker) / len(ker), 4),
        "kernel_density_min_pct": round(100 * ker[0], 4),
        "kernel_density_median_pct": round(100 * ker[mid], 4),
        "kernel_density_max_pct": round(100 * ker[-1], 4),
        "kernel_density_spread_x": round(ker[-1] / ker[0], 4) if ker[0] else None,
    }


def spearman(a, b, keys):
    """Rank correlation between two {key: value} maps over `keys`.

    Rank rather than value, because the question is whether the per-head
    ORDERING survives, not whether the magnitudes match: a cell where every
    head routes 3 points denser has the same operating structure.
    """
    ra = {h: i for i, h in enumerate(sorted(keys, key=lambda h: a[h]))}
    rb = {h: i for i, h in enumerate(sorted(keys, key=lambda h: b[h]))}
    n = len(keys)
    if n < 2:
        return None
    d2 = sum((ra[h] - rb[h]) ** 2 for h in keys)
    return round(1 - 6 * d2 / (n * (n * n - 1)), 4)


def persistence(rows, tau):
    """Does the per-head density ordering survive a step change? A block change?

    Two separate questions with different answers, which is why they are not
    one number. Cells are keyed (transformer block, sampling step).
    """
    sel = [r for r in rows if r["tau"] == tau]
    if not sel:
        return None
    heads = sorted({r["head"] for r in sel})
    cell = {}
    for r in sel:
        cell.setdefault((r["block"], r["step"]), {})[r["head"]] = r["kernel_density_pct"]
    blocks = sorted({b for b, _ in cell})
    steps = sorted({s for _, s in cell})

    across_steps, across_blocks = [], []
    for b in blocks:
        present = [s for s in steps if (b, s) in cell]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                rho = spearman(cell[(b, present[i])], cell[(b, present[j])], heads)
                if rho is not None:
                    across_steps.append({"block": b, "steps": [present[i], present[j]],
                                         "rho": rho})
    for s in steps:
        present = [b for b in blocks if (b, s) in cell]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                rho = spearman(cell[(present[i], s)], cell[(present[j], s)], heads)
                if rho is not None:
                    across_blocks.append({"step": s, "blocks": [present[i], present[j]],
                                          "rho": rho})

    def band(pairs):
        if not pairs:
            return None
        v = sorted(p["rho"] for p in pairs)
        return {"pairs": len(v), "min": v[0], "median": v[len(v) // 2], "max": v[-1]}

    means = {h: round(sum(c[h] for c in cell.values()) / len(cell), 4) for h in heads}
    scope = (f"steps compared: {steps}; blocks compared: {blocks}. The "
             f"across-steps figure is over the captured steps only.")
    ordered = sorted(heads, key=lambda h: means[h])
    return {
        "tau": tau,
        "scope": scope,
        "same_block_across_steps": band(across_steps),
        "across_blocks_same_step": band(across_blocks),
        "per_head_mean_kernel_density_pct": means,
        "lowest_heads": ordered[:6],
        "highest_heads": ordered[-6:],
        "per_head_mean_spread_x": round(means[ordered[-1]] / means[ordered[0]], 4),
        "detail": {"same_block_across_steps": across_steps,
                   "across_blocks_same_step": across_blocks},
    }


def scrub_argv(argv, capture):
    """argv with the capture path rewritten to $H3_CAPTURE_ROOT/<dir>.

    Records are tracked content and the capture store is outside the repo, so
    the literal path is a leak. It is also the less useful of the two: the
    store's location is per-box, the capture's name is not. The 2026-08-18
    workload-grid and ER-SDE-ODE records shipped absolute scratch paths for
    want of this, and the pre-commit gate now refuses them.
    """
    out = []
    for arg in argv:
        if arg.endswith(capture.name) and "/" in arg:
            out.append(f"$H3_CAPTURE_ROOT/{capture.name}")
        else:
            out.append(arg)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", required=True,
                    help="capture DIRECTORY; every qkv_*_b<N>_s<M>.pt in it is swept")
    ap.add_argument("--canvas", required=True)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--taus", default="1.0,1.3")
    ap.add_argument("--heads", type=int, default=0,
                    help="0 = every head. Otherwise the first n of analyze_routing's "
                         "seeded permutation, which is the set that tool samples.")
    ap.add_argument("--audio-start", type=int, default=None)
    ap.add_argument("--out", default=None, help="write the record here")
    ap.add_argument("--verify", action="store_true",
                    help="sweep one cell only and print the aggregate, for comparing "
                         "against analyze_routing.py's raster row on the same input")
    args = ap.parse_args()

    ar = load_analyze_routing()
    pool = ar.load_eager()
    ac = ar.load_capture_tools()

    capture = Path(args.capture).expanduser()
    files = sorted(
        (p for p in capture.glob("qkv_*.pt") if _NAME.search(p.name)),
        key=lambda p: tuple(int(x) for x in _NAME.search(p.name).groups()),
    )
    if not files:
        raise SystemExit(f"no qkv_*_b<N>_s<M>.pt under {capture}")
    if args.verify:
        files = files[:1]

    taus = [float(t) for t in args.taus.split(",")]
    w, h = (int(v) for v in args.canvas.lower().split("x"))

    manifest = capture / "manifest.json"
    provenance = None
    if manifest.exists():
        provenance = json.loads(manifest.read_text()).get("provenance")

    rows, cells, heads_used = [], [], None
    for path in files:
        block, step = (int(x) for x in _NAME.search(path.name).groups())
        d = torch.load(path, map_location="cpu", weights_only=True)
        q, k = d["q"][0], d["k"][0]
        H, S, _ = k.shape
        start, stop, grid = ac.video_span(S, (w, h), args.length)

        g = torch.Generator().manual_seed(0)
        n_heads = H if args.heads == 0 else args.heads
        heads = torch.randperm(H, generator=g)[:n_heads].tolist()
        heads_used = heads

        sink_kv = (start + BLOCK - 1) // BLOCK
        sink_q = 0 if args.audio_start is None else args.audio_start // BLOCK
        order = torch.arange(grid[0] * grid[1] * grid[2])

        for tau in taus:
            per_head = per_head_densities(
                ar, q, k, order, start, stop, tau, heads, pool, sink_kv, sink_q)
            agg = summarise(per_head)
            cells.append({"block": block, "step": step, "tau": tau,
                          "sequence": int(S), "blocks": int(S // BLOCK), **agg})
            for head, eff, ker in per_head:
                rows.append({"block": block, "step": step, "tau": tau,
                             "head": head,
                             "ordering_effect_density_pct": round(100 * eff, 4),
                             "kernel_density_pct": round(100 * ker, 4)})
            print(f"b{block:<3} s{step:<3} tau {tau}  "
                  f"eff {agg['ordering_effect_density_pct']:6.2f}%  "
                  f"kernel {agg['kernel_density_pct']:6.2f}%  "
                  f"per-head kernel {agg['kernel_density_min_pct']:.2f}"
                  f"-{agg['kernel_density_max_pct']:.2f}% "
                  f"({agg['kernel_density_spread_x']}x)", flush=True)

    if args.verify:
        print("\nCompare the aggregate above against analyze_routing.py's raster row\n"
              "on the same capture file, --heads and --tau. They must agree exactly:\n"
              "both call the same block_stats/exact_mask/forced_masks.")
        return 0

    persist = [p for p in (persistence(rows, t) for t in taus) if p]
    for p in persist:
        ab = p["across_blocks_same_step"]
        sb = p["same_block_across_steps"]
        print(f"\nPER-HEAD PERSISTENCE at tau {p['tau']} (Spearman rho of the head ordering)")
        if sb:
            print(f"  same block, across steps : {sb['pairs']} pairs, "
                  f"{sb['min']:+.3f} .. {sb['median']:+.3f} .. {sb['max']:+.3f}")
        if ab:
            print(f"  across blocks, same step : {ab['pairs']} pairs, "
                  f"{ab['min']:+.3f} .. {ab['median']:+.3f} .. {ab['max']:+.3f}")
        print(f"  per-head means over all cells span {p['per_head_mean_spread_x']}x")

    record = {
        "measured": "2026-08-19",
        "produced_by": "bench/sweep_routing_density.py",
        "what": "per-head Sol routing density, raster ordering, every capture cell",
        "capture": capture.name,
        "capture_provenance": provenance,
        "argv": scrub_argv(sys.argv[1:], capture),
        "heads_measured": len(heads_used or []),
        "head_ids": heads_used,
        "ordering": "raster",
        "cells": cells,
        "persistence": persist,
        "rows": rows,
    }
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(record, indent=1) + "\n")
        print(f"\nwrote {out}")
    else:
        json.dump(record, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
