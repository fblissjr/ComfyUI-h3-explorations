#!/usr/bin/env python3
"""How many blocks does Sol's router keep exact, and does the token ordering move it?

`docs/open_experiments.md` #18. This is the missing denominator under every
curve comparison in this repo: every Morton or Hilbert A/B ever run here held
`tau` fixed and believed that held the operating point fixed. It does not. The
routing threshold is

    thr = tau * sqrt(sum_d centroid_d^2 * kcvar_d * log2s^2)

and `kcvar` is the variance **across the block centroids**, which the
permutation defines (`sol_attn_preprocess.cu:107-123`, applied at `:199`). So
reordering moves the threshold, and a fixed-`tau` arm compares two operating
points rather than two orderings.

## Two different numbers, and they answer different questions

Emitting one and labelling it "routed density" is how the prototype behind this
script mislabelled its own output. Both are printed:

**ordering-effect density** -- forced-exact pairs dropped from numerator *and*
denominator. Diagonal and neighbour blocks are always exact, and every pair
touching a conditioning block is exact under `exact_kv`, regardless of
ordering. Including them dilutes exactly the quantity under test. **This is the
number to compare orderings with.**

**kernel density** -- what the kernel actually routes, forced pairs included.
Higher, and it is the number for anything about cost, and the only one that may
be compared against `routed_cap_percent`.

## What this is not

**Not the CUDA kernel's arithmetic.** The routing rule is transcribed from
upstream's own eager reference, whose docstring says it "defines the algorithm,
not the CUDA kernel's arithmetic (full precision here vs INT8 there)". The
pooling is not transcribed at all -- `_pool` is imported from that module. What
is skipped is the INT8 quantization of pooled keys and query centroids, so
expect the kernel to disagree in the last places. Nothing here is a speed
measurement and nothing here is output quality.

**Not valid at a length the capture was not taken at.** `kcvar` is a variance
over every block centroid in the sequence, so the block *count* is part of the
measurement: 591 blocks at 124 frames against roughly 1,700 at 362. Routed
density is the most length-sensitive quantity this repo has, and it is
length-sensitive for that reason and not because of the ~60k token speed floor.
Read the capture's own length before quoting a number from it.

## Running it

    python bench/analyze_routing.py $H3_CAPTURE_ROOT/<dir>/qkv_*_b24_s1.pt \\
        --canvas 1344x768 --length 124

No GPU, no model, no server. Needs the capture and `coderef/comfy-kitchen-sol`
for the eager reference; it refuses rather than falling back if that is absent,
because a locally reimplemented pooling is the thing this script exists to
avoid.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
BLOCK = 64
LOG2E = 1.4426950408889634


def load_eager():
    """Upstream's eager Sol-Attn, imported as a package for its relative imports.

    Imported rather than reimplemented: it is upstream's statement of what the
    algorithm IS, so `_pool` here is not this script's idea of pooling.
    """
    root = REPO / "coderef" / "comfy-kitchen-sol"
    if not root.exists():
        raise SystemExit(
            f"{root} is missing. It is a gitignored symlink to the sister "
            "checkout; see coderef/ in CLAUDE.md. This script will not "
            "substitute its own pooling for upstream's.")
    sys.path.insert(0, str(root))
    from comfy_kitchen.backends.eager.sol_attn import _pool  # noqa: E402
    return _pool


def load_capture_tools():
    spec = importlib.util.spec_from_file_location(
        "_ac", REPO / "bench" / "analyze_capture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def orderings(grid, video_start, vendor, curves):
    """{label: permutation of the video span}, including the identity."""
    pad = (-int(video_start)) % BLOCK
    out = {"raster": torch.arange(grid[0] * grid[1] * grid[2])}
    for curve in curves:
        if curve in ("3d", "2d_frame"):
            perm, _ = vendor.morton_perm(grid, "cpu", curve)
        elif curve == "hilbert":
            sys.path.insert(0, str(REPO))
            import sol_curves
            perm, _ = sol_curves.hilbert_perm(grid, "cpu")
        else:
            raise SystemExit(f"unknown curve {curve!r}")
        out[curve] = torch.roll(perm, pad) if pad else perm
    return out


def block_stats(q, k, order, start, stop, head, pool):
    """(query centroids, centred pooled keys, kcvar) for one head, one ordering.

    **The population is the whole packed sequence**, conditioning rows included,
    because `kcvar` is a variance over every centroid the kernel pools. Blocking
    only the video span measures a different threshold than the one that runs.
    """
    S, D = k.shape[1], k.shape[2]
    rows = torch.arange(S)
    rows[start:stop] = order + start
    n = S // BLOCK
    t = n * BLOCK
    fk = k[head].index_select(0, rows)[:t].float().unsqueeze(0).unsqueeze(2)
    fq = q[head].index_select(0, rows)[:t].float().unsqueeze(0).unsqueeze(2)
    kc = pool(fk, n, "mean")[0, :, 0, :]                  # (N, D), upstream's pooling
    centroid = pool(fq, n, "mean")[0, :, 0, :]            # (N, D)
    kcc = kc - kc.mean(0, keepdim=True)
    return centroid, kcc, kcc.pow(2).mean(0), n, D


def exact_mask(centroid, kcc, kcvar, tau, scale):
    """The router's exact/approximate decision, per (query block, key block).

    Transcribed from the eager reference at `sol_attn.py:105-142`. The
    per-token form there reduces to this: colmean is a mean of dot products
    over the query block's rows, and the mean of dot products is the dot
    product of the mean, so the block centroid may be used directly.
    """
    log2s = scale * LOG2E
    var = (centroid.pow(2) * kcvar).sum(-1)
    thr = tau * torch.sqrt(var * log2s * log2s + 1e-6)
    colmean = (centroid @ kcc.T) * log2s
    return colmean > thr.unsqueeze(-1)


def _thr_naive(centroid, kcc, kcvar, tau, scale):
    """Second implementation, written differently, for the cross-check.

    Deliberately a Python loop over query blocks rather than a batched matmul.
    If this disagrees with `exact_mask`, every number below describes a rule
    nobody runs. Same failure mode `analyze_morton.py` guards with
    `_independent_perm`.
    """
    log2s = scale * LOG2E
    out = torch.zeros(centroid.shape[0], kcc.shape[0], dtype=torch.bool)
    for i in range(centroid.shape[0]):
        c = centroid[i]
        thr = tau * float(torch.sqrt((c.pow(2) * kcvar).sum() * log2s ** 2 + 1e-6))
        for j in range(kcc.shape[0]):
            out[i, j] = float(torch.dot(c, kcc[j])) * log2s > thr
    return out


def forced_masks(n, sink_kv_blocks, sink_q_blocks):
    """(always-exact pairs, pairs to drop from the ordering-effect density).

    Forced-exact is ordering-invariant: the diagonal band is positional, and
    conditioning rows are never permuted by any curve here. So dropping them
    compares like with like.
    """
    idx = torch.arange(n)
    band = (idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1
    kv = (idx < sink_kv_blocks).view(1, -1).expand(n, n)
    qq = (idx < sink_q_blocks).view(-1, 1).expand(n, n)
    forced = band | kv | qq
    return forced, ~forced


def densities(q, k, order, start, stop, tau, heads, pool, sink_kv, sink_q, scale=None):
    """(ordering-effect density, kernel density), averaged over sampled heads."""
    eff, ker = [], []
    for head in heads:
        centroid, kcc, kcvar, n, D = block_stats(q, k, order, start, stop, head, pool)
        s = D ** -0.5 if scale is None else scale
        thresholded = exact_mask(centroid, kcc, kcvar, tau, s)
        forced, free = forced_masks(n, sink_kv, sink_q)
        eff.append(float((thresholded & free).sum()) / float(free.sum()))
        ker.append(float((thresholded | forced).sum()) / float(n * n))
    return sum(eff) / len(eff), sum(ker) / len(ker)


def compensating_tau(q, k, order, start, stop, target, heads, pool, sink_kv, sink_q):
    """tau reproducing `target` ordering-effect density. Density falls with tau."""
    lo, hi = 0.25, 6.0
    for _ in range(20):
        mid = (lo + hi) / 2
        d, _ = densities(q, k, order, start, stop, mid, heads, pool, sink_kv, sink_q)
        if d > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run_controls(q, k, orders, start, stop, tau, heads, pool, sink_kv, sink_q):
    """Every one of these has been shown to fail for the right reason.

    A control that has only ever been green is a control nobody has tested.
    """
    print("CONTROLS")
    ok = True

    for label, order in orders.items():
        if sorted(order.tolist()) != list(range(stop - start)):
            print(f"  FAIL  {label} is not a permutation of the video span")
            ok = False
    print(f"  pass  all {len(orders)} orderings are permutations")

    # An ordering that shuffles WITHIN each 64-token block cannot change block
    # membership, so every centroid, kcvar and threshold is unchanged and the
    # density must be identical. Unlike comparing the identity against raster
    # -- which is comparing `arange` to `arange` and cannot fail -- this is a
    # non-trivial permutation with a forced answer, so it catches rows not
    # actually being reordered, a wrong span, and an off-by-one block boundary.
    #
    # Written after the first version of this control stayed green under a
    # deliberate mutation, on 2026-08-16. It was tautological, which is the
    # defect `verify_adjacency` had and the reason `docs/checks.md` gap #6
    # exists.
    g = torch.Generator().manual_seed(11)
    within = torch.arange(stop - start)
    head_off = (-int(start)) % BLOCK          # first whole block inside the span
    for b0 in range(head_off, (stop - start) - BLOCK + 1, BLOCK):
        within[b0:b0 + BLOCK] = within[b0:b0 + BLOCK][torch.randperm(BLOCK, generator=g)]
    a = densities(q, k, within, start, stop, tau, heads[:1], pool, sink_kv, sink_q)
    b = densities(q, k, orders["raster"], start, stop, tau, heads[:1], pool, sink_kv, sink_q)
    if a != b:
        print(f"  FAIL  a within-block shuffle changed the density ({a} vs {b}); "
              "block boundaries or the span are wrong")
        ok = False
    else:
        print("  pass  a within-block shuffle leaves density unchanged (membership invariant)")

    # ...and the converse, so the pair cannot both pass by nothing happening.
    moved = [l for l, o in orders.items() if l != "raster"
             and densities(q, k, o, start, stop, tau, heads[:1], pool, sink_kv, sink_q) != b]
    if not moved:
        print("  FAIL  no curve changed the density at all; the ordering is not being applied")
        ok = False
    else:
        print(f"  pass  {len(moved)} of {len(orders) - 1} curves do move it ({', '.join(moved)})")

    head = heads[0]
    centroid, kcc, kcvar, n, D = block_stats(
        q, k, orders["raster"], start, stop, head, pool)

    # `kcvar` must be the variance over EVERY block centroid the kernel pools,
    # conditioning rows included -- `open_experiments.md` #18 makes this a
    # requirement because dropping the conditioning blocks moves the threshold
    # for every query block. Added 2026-08-16 after a mutation that computed
    # kcvar over the video span alone passed every other control here.
    expect_n = k.shape[1] // BLOCK
    if kcc.shape[0] != expect_n:
        print(f"  FAIL  {kcc.shape[0]} centroids pooled, expected {expect_n} "
              f"(= sequence // {BLOCK}); the population is not the whole sequence")
        ok = False
    elif not torch.allclose(kcvar, kcc.pow(2).mean(0), rtol=1e-5, atol=1e-8):
        print("  FAIL  kcvar is not the variance of the pooled centroids it was "
              "derived from; some blocks were dropped from the population")
        ok = False
    else:
        print(f"  pass  kcvar spans all {expect_n} block centroids, conditioning included")
    m = 24                                    # the naive form is O(N^2) in Python
    fast = exact_mask(centroid[:m], kcc[:m], kcvar, tau, D ** -0.5)
    slow = _thr_naive(centroid[:m], kcc[:m], kcvar, tau, D ** -0.5)
    if not torch.equal(fast, slow):
        print(f"  FAIL  batched and naive threshold disagree on {int((fast != slow).sum())} pairs")
        ok = False
    else:
        print(f"  pass  batched threshold matches an independent naive transcription ({m}x{m})")

    lo, _ = densities(q, k, orders["raster"], start, stop, tau * 0.9, heads[:1],
                      pool, sink_kv, sink_q)
    hi, _ = densities(q, k, orders["raster"], start, stop, tau * 1.1, heads[:1],
                      pool, sink_kv, sink_q)
    if not lo > hi:
        print(f"  FAIL  density does not fall with tau ({lo:.4f} at 0.9x, {hi:.4f} at 1.1x); "
              "the threshold is not reaching the decision")
        ok = False
    else:
        print(f"  pass  density falls with tau ({lo:.2%} -> {hi:.2%} over 0.9x..1.1x)")
    print()
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capture", help="a qkv_*.pt written by h3_capture.py")
    ap.add_argument("--canvas", required=True)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--tau", type=float, default=1.3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--curves", default="2d_frame,3d,hilbert")
    ap.add_argument("--audio-start", type=int, default=None,
                    help="row where the target audio segment begins. Enables the "
                         "exact_kv_and_rows dense-query range; without it only "
                         "the exact_kv sink is modelled.")
    args = ap.parse_args()

    pool = load_eager()
    ac = load_capture_tools()
    vendor = ac.load_shipped_morton()

    d = torch.load(args.capture, map_location="cpu", weights_only=True)
    q, k = d["q"][0], d["k"][0]
    H, S, _ = k.shape
    w, h = (int(v) for v in args.canvas.lower().split("x"))
    start, stop, grid = ac.video_span(S, (w, h), args.length)
    n_blocks = S // BLOCK

    sink_kv = (start + BLOCK - 1) // BLOCK
    sink_q = 0 if args.audio_start is None else args.audio_start // BLOCK

    g = torch.Generator().manual_seed(0)
    heads = torch.randperm(H, generator=g)[:args.heads].tolist()

    print(f"capture {Path(args.capture).name}")
    print(f"  heads {args.heads} of {H}, sequence {S:,}, {n_blocks} blocks, grid {grid}")
    print(f"  video span [{start:,}, {stop:,}), conditioning {start:,} rows "
          f"({start / S:.1%})")
    print(f"  sink: {sink_kv} exact-KV blocks, {sink_q} dense-query blocks, tau {args.tau}")
    if S < 60_000:
        print(f"  NOTE: {S:,} tokens. kcvar is a variance over all {n_blocks} block "
              f"centroids, so\n        density here is NOT comparable to a capture "
              f"at a different length.")
    print()

    orders = orderings(grid, start, vendor, args.curves.split(","))
    if not run_controls(q, k, orders, start, stop, args.tau, heads, pool, sink_kv, sink_q):
        raise SystemExit("a control failed; the numbers below would not mean anything")

    print(f"ROUTED DENSITY at tau={args.tau}\n")
    print(f"  {'ordering':<14}{'ordering-effect':>17}{'vs raster':>11}{'kernel':>10}")
    base = None
    results = {}
    for label, order in orders.items():
        eff, ker = densities(q, k, order, start, stop, args.tau, heads, pool,
                             sink_kv, sink_q)
        results[label] = eff
        if base is None:
            base = eff
        print(f"  {label:<14}{eff:>16.2%}{eff / base:>10.3f}x{ker:>9.2%}")

    print(f"\nCOMPENSATING TAU -- reproduces raster's ordering-effect density\n")
    print(f"  {'ordering':<14}{'tau':>8}")
    for label, order in orders.items():
        t = compensating_tau(q, k, order, start, stop, base, heads, pool, sink_kv, sink_q)
        print(f"  {label:<14}{t:>8.3f}")

    print("\nReading it. The ordering-effect column is the one to compare curves\n"
          "with; the kernel column is the one to size routed_cap_percent against.\n"
          "A compensating tau far from the base means a fixed-tau A/B of that\n"
          "curve compared two operating points, not two orderings.")


if __name__ == "__main__":
    main()
