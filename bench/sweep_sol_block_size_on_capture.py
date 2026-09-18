#!/usr/bin/env python3
"""Is 64 tokens per block the limit, or does ordering already buy the granularity? Float reference, on captures.

    python bench/sweep_sol_block_size_on_capture.py <capture.pt> [...] --out bench/results/<date>_sol_block_size.json

Sol-Attn routes whole 64-token blocks: all 64 queries of a block share one
routing decision, and a key block is attended exactly or pooled, never split.
On 2026-09-17 the one lever that removed a visible morph was the `3d` token
reorder, which changes block MEMBERSHIP and nothing else, and that points at
blocks holding unlike tokens. Two ways to make a block more alike: order the
tokens better (done), or make the block smaller (a kernel rewrite, `BLOCK = 64`
is baked into the CUDA layout). This measures the second before anyone builds
it.

For each ordering (raster, `3d`) and each block size (64, 32, 16), sweep tau
and record (routed density, error against exact attention). Density is cost.
Read it the way `sweep_sol_orderings_on_capture.py` is read: a configuration is
better only where its curve lies BELOW another's. If `3d` at 64 already sits on
raster at 16, ordering has bought the granularity and smaller blocks are not
worth a kernel; if 16 is far below everything at equal density, the ceiling is
real.

The arithmetic is the algorithm in fp32, NOT the CUDA kernel: this pack's
`analyze_sol_error.eager_sol_reference`, which is query-chunked so it runs at
full length and is calibrated against the vendored oracle. It and the oracle
read a module-level block size; this script sets it (`block_size`), and
re-runs that calibration at every size before trusting a number. So the
INT8 term is absent by construction and the result is a ceiling on what finer
blocks could buy, not a forecast of a kernel. `qk_balance` and `rotate` are
identities in exact arithmetic and are not modelled.

Density is computed here, not by the reference (it returns outputs only): the
query-block mean of the block scores is linear in the queries, so it is the
query-block centroid against the centred block keys, one N x N product per
head. `check_density` pins that against kitchen's eager reference, which
reports its own routed counts, at every block size on a short sequence, up to
the pairs that sit exactly on the threshold (see its docstring).

Sinks are the node's for the capture's layout, in tokens: the node gives them
in 64-token blocks and they are rescaled to the block size under test, so the
same rows stay exact. The forced diagonal (a query block's own and adjacent key
blocks) is part of the algorithm at each size and counts toward density.

Limits: captured blocks and steps only; a head prefix; routing cost, which
grows with the square of the block count, is not in "density"; error against
exact attention is not a verdict on a clip. Captures must have been taken with
reordering OFF.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import _sol_attn_reference as oracle  # noqa: E402
import analyze_sol_error as ase  # noqa: E402
from _live_sol import live_sol  # noqa: E402

BLOCK_SIZES = (64, 32, 16)
ORDERINGS = ("raster", "3d")
NODE_BLOCK = 64                      # the unit `_sink_blocks` and `_perm_for` speak in


def _kitchen_eager():
    # By module name: the package re-exports the function `sol_attn`, which
    # shadows the submodule of the same name under attribute access.
    import importlib
    importlib.import_module("comfy_kitchen.backends.eager.sol_attn")
    return sys.modules["comfy_kitchen.backends.eager.sol_attn"]


@contextlib.contextmanager
def block_size(size):
    """The float reference and its oracle at another block size. Both read a module global."""
    kitchen_eager = _kitchen_eager()
    mods = (oracle, ase, kitchen_eager)
    old = [m.BLOCK for m in mods]
    for m in mods:
        m.BLOCK = size
    try:
        yield
    finally:
        for m, o in zip(mods, old):
            m.BLOCK = o


def routed_counts(q, k, tau, sink_kv, sink_q, size, device):
    """Exact key blocks per query block, (H, N) int64, from fp32 q, k of shape [1, H, T, D]."""
    _b, h, t, d = q.shape
    n = (t + size - 1) // size
    log2s = (d ** -0.5) * ase._LOG2E
    lengths = torch.full((n,), float(size), device=device)
    if n * size - t:
        lengths[-1] = float(t - (n - 1) * size)
    idx = torch.arange(n, device=device)
    forced = (idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1
    forced |= ((idx >= sink_kv[0]) & (idx < sink_kv[1])).view(1, n)
    forced |= ((idx >= sink_q[0]) & (idx < sink_q[1])).view(n, 1)
    out = []
    for head in range(h):
        fq = q[0, head].to(device=device, dtype=torch.float32)
        fk = k[0, head].to(device=device, dtype=torch.float32)
        pad = n * size - t
        if pad:
            fq = torch.nn.functional.pad(fq, (0, 0, 0, pad))
            fk = torch.nn.functional.pad(fk, (0, 0, 0, pad))
        centroid = fq.view(n, size, d).sum(1) / lengths.view(n, 1)
        kc = fk.view(n, size, d).sum(1) / lengths.view(n, 1)
        kcc = kc - kc.mean(0, keepdim=True)
        kc_var = kcc.pow(2).mean(0)
        colmean = (centroid @ kcc.T) * log2s
        thr = tau * torch.sqrt((centroid.pow(2) * kc_var).sum(-1) * log2s * log2s + 1e-6)
        exact = (colmean > thr.view(n, 1)) | forced
        out.append(exact.sum(-1))
    return torch.stack(out)


def check_density(device):
    """`routed_counts` against kitchen's eager reference, which reports its own counts.

    Returns (largest per-block difference, largest relative difference in the
    total routed count). The two compute the same block score in a different
    order of summation (the mean of per-token scores there, the score of the
    mean query here), so a pair sitting exactly on the threshold can fall
    either way: on the GPU one query block in the test differs by one pair, on
    the CPU none does. A logic error would not look like that, so the gate is
    "no block off by more than one, and the totals within a tenth of a percent".
    """
    kitchen_eager = _kitchen_eager()
    g = torch.Generator().manual_seed(11)
    t, h, d = 1500, 3, 128                                 # ragged at every size under test
    q = torch.randn(1, h, t, d, generator=g) + 0.5
    k = torch.randn(1, h, t, d, generator=g) * torch.linspace(0.2, 2.0, d)
    v = torch.randn(1, h, t, d, generator=g)
    worst_block, worst_total = 0, 0.0
    for size in BLOCK_SIZES:
        n = (t + size - 1) // size
        sink_kv, sink_q = (0, 2 * NODE_BLOCK // size), (1 * NODE_BLOCK // size, 3 * NODE_BLOCK // size)
        with block_size(size):
            cnt = torch.zeros(1, h, n, dtype=torch.int32, device=device)
            to = lambda x: x.permute(0, 2, 1, 3).contiguous().to(device)   # noqa: E731
            kitchen_eager.sol_attn(to(q), to(k), to(v), tau=1.0, sink_blocks=list(sink_kv),
                                   sink_q=list(sink_q), blk_cnt=cnt)
        mine = routed_counts(q, k, 1.0, sink_kv, sink_q, size, device)
        theirs = cnt[0].long()
        block = int((mine - theirs).abs().max())
        total = abs(int(mine.sum()) - int(theirs.sum())) / max(int(theirs.sum()), 1)
        worst_block, worst_total = max(worst_block, block), max(worst_total, total)
        print(f"  density check, block {size}: largest per-block difference {block}, "
              f"total routed pairs differ by {total:.3%}")
    return worst_block, worst_total


def error_at(curve, density):
    """Error of one curve at a given routed density, log-linear between its two nearest points; None outside it."""
    import math
    pts = sorted((c["density"], c["rel_l2"]) for c in curve)
    for (d0, e0), (d1, e1) in zip(pts, pts[1:]):
        if d0 <= density <= d1:
            w = (density - d0) / (d1 - d0) if d1 > d0 else 0.0
            return math.exp(math.log(e0) * (1 - w) + math.log(e1) * w)
    return None


def summarize(path, tau=1.0):
    """Every configuration's error at ONE density per cell: what raster at block 64 routes at `tau`.

    Curves are compared at equal cost, never at equal tau (a reorder and a
    block size both move the threshold). Printed as error and as a ratio to
    raster at block 64; a dash is a curve that does not reach that density.
    """
    import orjson
    data = orjson.loads(Path(path).read_bytes())
    names = [f"{o}_b{b}" for o in ORDERINGS for b in BLOCK_SIZES]
    print(f"{path}: error at the density raster_b64 routes at tau {tau}, and its ratio to raster_b64")
    print("| cell | density | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for cell, rows in data["cells"].items():
        base = next(c for c in rows["raster_b64"] if c["tau"] == tau)
        cols = []
        for name in names:
            e = error_at(rows[name], base["density"])
            cols.append("-" if e is None else f"{e:.4f} ({e / base['rel_l2']:.2f}x)")
        print(f"| {cell} | {base['density']:.3f} | " + " | ".join(cols) + " |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="*")
    ap.add_argument("--heads", type=int, default=8, help="head PREFIX")
    ap.add_argument("--taus", default="0.6,0.8,1.0,1.3,2.0")
    ap.add_argument("--grid", default="102,24,42", help="token T,H,W of the video segment")
    ap.add_argument("--audio-span", default="395,1545")
    ap.add_argument("--video-start", type=int, default=1545)
    ap.add_argument("--check-only", action="store_true", help="run the two instrument checks and stop")
    ap.add_argument("--out")
    ap.add_argument("--summarize", metavar="RESULT.json", nargs="+",
                    help="print the equal-density table for finished result files and stop (no GPU)")
    ap.add_argument("--at-tau", type=float, default=1.0, help="with --summarize: whose density to compare at")
    args = ap.parse_args()
    if args.summarize:
        for path in args.summarize:
            summarize(path, args.at_tau)
        return 0

    import orjson
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("instrument checks:")
    calibration = {}
    for size in BLOCK_SIZES:
        with block_size(size):
            drift, at = ase.calibrate_against_oracle(tau=1.0)
        calibration[size] = drift
        print(f"  chunked reference against the oracle, block {size}: worst rel_l2 {drift:.2e} (at t={at})")
    density_bad = check_density(device)
    # The control: if setting the block size did nothing, every line above would
    # still pass, at 64 three times. The reference at 32 must differ from itself at 64.
    torch.manual_seed(5)
    probe = [torch.randn(1, 1, 1000, 64) for _ in range(3)]
    outs = []
    for size in (64, 32):
        with block_size(size):
            outs.append(ase.eager_sol_reference(*probe, tau=1.0))
    moved = float((outs[0] - outs[1]).norm() / outs[0].norm())
    print(f"  control, block 32 against block 64 on one input: rel_l2 {moved:.2e} (must not be zero)")
    if moved < 1e-4:
        print("FAIL: changing the block size did not change the reference")
        return 1
    if max(calibration.values()) > 1e-3 or density_bad[0] > 1 or density_bad[1] > 1e-3:
        print("FAIL: the instrument does not reproduce its oracle at every block size")
        return 1
    if args.check_only:
        return 0
    if not args.captures or not args.out:
        ap.error("captures and --out are required unless --check-only")

    node = live_sol()
    sys.path.insert(0, str(REPO / "workflows"))
    import h3_config
    recipe = h3_config.SOL_RECOMMENDED_CUDA
    grid = tuple(int(x) for x in args.grid.split(","))
    taus = [float(x) for x in args.taus.split(",")]
    start = args.video_start
    a0, a1 = (int(x) for x in args.audio_span.split(","))

    # A sweep that died keeps what it finished (the file is rewritten after every
    # capture); a rerun with the same --out picks up after the last finished cell.
    cells = {}
    if Path(args.out).exists():
        cells = orjson.loads(Path(args.out).read_bytes()).get("cells", {})
        print(f"resuming: {sorted(cells)} already in {args.out}")
    for cap in args.captures:
        m = re.search(r"_b(\d+)_s(\d+)", Path(cap).name)
        if (f"b{m.group(1)}_s{m.group(2)}" if m else Path(cap).stem) in cells:
            continue
        q, k, v = ase.load_capture(cap)
        q, k, v = (x[:, :args.heads] for x in (q, k, v))
        tokens = q.shape[2]
        if start + grid[0] * grid[1] * grid[2] != tokens:
            raise SystemExit(f"{cap}: layout says video starts at {start} with grid {grid}, capture has {tokens} rows")
        m = re.search(r"_b(\d+)_s(\d+)", Path(cap).name)
        cell = f"b{m.group(1)}_s{m.group(2)}" if m else Path(cap).stem
        ref = ase.dense_reference(q, k, v)                   # [1, H, S, D] fp32, raster order
        ref_norm = float(ref.norm())
        layout = {"sol_h3_video_span": (start, tokens), "sol_h3_audio_span": (a0, a1)}
        sink_kv64, sink_q64 = node._sink_blocks(layout, tokens, recipe["sink_conditioning"])
        rows = {}
        for name in ORDERINGS:
            if name == "raster":
                index = torch.arange(tokens)
            else:
                perm, _inv = node._perm_for(grid, name, "cpu", start)
                index = torch.cat([torch.arange(start), start + perm.cpu()])
            assert torch.equal(torch.sort(index).values, torch.arange(tokens)), f"{name}: not a bijection"
            qs, ks, vs, ref_p = (x[:, :, index] for x in (q, k, v, ref))
            for size in BLOCK_SIZES:
                scale = NODE_BLOCK // size
                sink_kv = (sink_kv64[0] * scale, sink_kv64[1] * scale)
                sink_q = (sink_q64[0] * scale, sink_q64[1] * scale)
                n = (tokens + size - 1) // size
                curve = []
                for tau in taus:
                    with block_size(size):
                        out = ase.eager_sol_reference(qs, ks, vs, tau=tau, sink_blocks=list(sink_kv),
                                                      sink_q=list(sink_q),
                                                      centroid_tail=bool(recipe["pooled_tail"]))
                    err = float((out - ref_p).norm()) / ref_norm
                    density = float(routed_counts(qs, ks, tau, sink_kv, sink_q, size, device).float().mean()) / n
                    curve.append({"tau": tau, "density": round(density, 5), "rel_l2": round(err, 5)})
                    del out
                rows[f"{name}_b{size}"] = curve
                print(f"{cell} {name:6s} block {size:2d} "
                      + "  ".join(f"tau {c['tau']}: d {c['density']:.3f} e {c['rel_l2']:.4f}" for c in curve), flush=True)
            del qs, ks, vs, ref_p
        cells[cell] = rows
        del ref, q, k, v
        # written after every capture: a long sweep that dies keeps what it finished
        Path(args.out).write_bytes(orjson.dumps({
            "what": "Sol-Attn ALGORITHM in fp32 (no INT8 term): error against fp32 dense attention versus routed "
                    "density, per token ordering and per block size, tau swept; only the video rows are permuted, "
                    "by the node's own _perm_for; sinks rescaled so the same rows stay exact at every block size",
            "model": "MiniMax H3, int8 convrot checkpoint, base 16-step t2v captures",
            "conditions": {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                           "heads": args.heads, "grid_thw": grid, "video_start": start, "audio_span": [a0, a1],
                           "sink_conditioning": recipe["sink_conditioning"], "pooled_tail": recipe["pooled_tail"],
                           "captures": [Path(c).name for c in args.captures],
                           "capture_set": Path(args.captures[0]).resolve().parent.name},
            "instrument": {"chunked_reference_vs_oracle_worst_rel_l2": {str(s): calibration[s] for s in BLOCK_SIZES},
                           "density_vs_kitchen_eager": {"largest_per_block_difference": density_bad[0],
                                                       "largest_relative_total_difference": density_bad[1]}},
            "how_to_read": "a configuration beats another only where its (density, rel_l2) curve lies below; "
                           "density is the share of key blocks attended exactly, forced diagonal and sinks included; "
                           "routing cost, which grows with the square of the block count, is not in it",
            "cells": cells}, option=orjson.OPT_INDENT_2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
