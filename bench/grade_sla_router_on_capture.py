#!/usr/bin/env python3
"""Gate the vendored SLA router + Triton kernel on real activations before anything renders through it.

`vendor/sla_sparse_triton.py` is LightX2V's sparse top-k router and forward
kernel, copied. A kernel that is subtly wrong renders a plausible clip, so this
runs first, on captured q/k/v, against the float64 row reference that
`bench/grade_sage_on_capture.py` established, and refuses to print a PASS
unless the violations below go red.

## Arms, per (block, head), on `--rows` sampled query rows

  dense_limit   router at sparsity 0 (every key block kept). The kernel is then
                a bf16 flash kernel (P rounded to bf16 before PV, blocks in
                torch.topk's unsorted order), so it agrees with dense attention
                to bf16 accumulation, not exactly. The bar is the sage band on
                the same block: PASS if rel_l2_mean <= 2x the `auto` (fp8)
                mode's rel_l2_mean for that block in
                bench/results/2026-08-18_sage_accuracy_on_capture.json -- a
                record taken on the 2026-08-17 capture (fl2va + ref LoRA), the
                same block, 256 rows, same float64 reference. Named because a
                different capture is a different input.
  router        the release's sparsity (0.85): the sparsity error of the LoRA's
                training attention, on the base model's activations.
  lut_zeroed    VIOLATION 1: every query block reads key block 0 `topk` times.
                Must be worse than `router` and at least 10x worse than
                `dense_limit`, or the kernel is not reading the table. (Not a
                multiple of the router's error: relative L2 saturates near 1.0,
                so on a head whose router error is already ~0.4 no multiple
                is reachable -- the first run failed on exactly that.)
  lut_permuted  VIOLATION 2: the router's lut with its rows permuted across
                query blocks -- the same density, the wrong blocks. Must be
                worse than `router`, or the router is not routing. The margin
                is also the interesting number: what the router buys over a
                random mask at equal density.
  sol_tau_*     the eager Sol-Attn reference (`bench/analyze_sol_error.py`) at
                each `--sol-tau`, scored on the SAME rows and head, so the two
                sparse methods sit on one scale. fp32 algorithm, no INT8.

Captures are [1, H, S, D] bf16; one head is moved to the card at a time. The
record scrubs the capture path to `$H3_CAPTURE_ROOT/<dir>`.

    python bench/grade_sla_router_on_capture.py \\
        --capture $H3_CAPTURE_ROOT/2026-08-20_ref3_362f_1024x768_fl2va \\
        --cells 24:0,49:47,49:8 --step 3 --rows 256 \\
        --out bench/results/2026-08-20_sla_router_gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from grade_sage_on_capture import sample_rows, reference_rows, _err  # noqa: E402
from analyze_sol_error import eager_sol_reference  # noqa: E402
from vendor.sla_sparse_triton import get_block_map, sparse_attn_forward, SOURCE_COMMIT  # noqa: E402

SAGE_RECORD = REPO / "bench" / "results" / "2026-08-18_sage_accuracy_on_capture.json"
BAND_FACTOR = 2.0


def sage_band(block: int, step: int):
    doc = json.loads(SAGE_RECORD.read_text())
    for p in doc["points"]:
        if p["block"] == block and p["step"] == step:
            return {m: v["rel_l2_mean"] for m, v in p["modes"].items()}
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", required=True)
    ap.add_argument("--cells", default="24:0,49:47", help="block:head pairs")
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--sparsity", type=float, default=0.85)
    ap.add_argument("--sol-tau", default="1.0,1.3")
    ap.add_argument("--blocks", default="64,64", help="BLOCK_M,BLOCK_N")
    ap.add_argument("--out", required=True)
    ap.add_argument("--measured", default=date.today().isoformat())
    args = ap.parse_args()

    cap = Path(args.capture).expanduser().resolve()
    if not cap.is_dir():
        sys.exit(f"refuse: {cap} is not a directory")
    bm, bn = (int(x) for x in args.blocks.split(","))
    taus = [float(t) for t in args.sol_tau.split(",") if t]
    dev = "cuda"
    torch.manual_seed(0)

    out = {"measured": args.measured, "produced_by": "bench/grade_sla_router_on_capture.py",
           "what": ("vendored SLA top-k router + Triton forward kernel graded on captured "
                    "activations against a float64 row reference, with two lut violations "
                    "and the eager Sol reference on the same rows"),
           "capture": f"$H3_CAPTURE_ROOT/{cap.name}", "step": args.step, "rows": args.rows,
           "sparsity_ratio": args.sparsity, "blocks_mn": [bm, bn],
           "vendor_source_commit": SOURCE_COMMIT, "band_factor": BAND_FACTOR,
           "sage_record": str(SAGE_RECORD.relative_to(REPO)), "cells": []}
    all_pass = True
    for cell in args.cells.split(","):
        blk, head = (int(x) for x in cell.split(":"))
        f = cap / f"qkv_L98498_S98498_b{blk}_s{args.step}.pt"
        if not f.exists():
            cands = sorted(cap.glob(f"qkv_*_b{blk}_s{args.step}.pt"))
            if len(cands) != 1:
                sys.exit(f"refuse: {cap.name} has {len(cands)} files for b{blk} s{args.step}")
            f = cands[0]
        d = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
        q = d["q"][:, head:head + 1].to(dev).contiguous()   # [1,1,S,D] bf16
        k = d["k"][:, head:head + 1].to(dev).contiguous()
        v = d["v"][:, head:head + 1].to(dev).contiguous()
        S = q.shape[2]
        rows = sample_rows(S, args.rows, device=dev)
        ref = reference_rows(q, k, v, rows)                      # float64 [1,1,n,D]
        cellrec = {"block": blk, "head": head, "seq_len": S, "arms": {}}

        def score(name, full):
            got = full.to(rows.device).index_select(2, rows)
            e = _err(got, ref)
            cellrec["arms"][name] = e
            return e["rel_l2_mean"]

        # dense limit: every key block
        t0 = time.time()
        _, lut_all, topk_all = get_block_map(q, k, topk_ratio=1.0, BLKQ=bm, BLKK=bn)
        dense_lim = score("dense_limit", sparse_attn_forward(q, k, v, lut_all.contiguous(), topk_all, bm, bn))
        torch.cuda.synchronize(); t_dense = time.time() - t0
        # the release's sparsity
        _, lut, topk = get_block_map(q, k, topk_ratio=1.0 - args.sparsity, BLKQ=bm, BLKK=bn)
        lut = lut.contiguous()
        t0 = time.time()
        router = score("router", sparse_attn_forward(q, k, v, lut, topk, bm, bn))
        torch.cuda.synchronize(); t_router = time.time() - t0
        # violation 1: zeroed lut
        zeroed = score("lut_zeroed", sparse_attn_forward(q, k, v, torch.zeros_like(lut), topk, bm, bn))
        # violation 2: rows permuted across query blocks
        g = torch.Generator(device="cpu").manual_seed(1)
        perm = torch.randperm(lut.shape[2], generator=g).to(dev)
        permuted = score("lut_permuted", sparse_attn_forward(q, k, v, lut[:, :, perm].contiguous(), topk, bm, bn))
        # Sol on the same rows
        for tau in taus:
            sol = eager_sol_reference(q, k, v, tau=tau)
            score(f"sol_tau_{tau:g}", sol)
        cellrec["kernel_seconds"] = {"dense_limit": round(t_dense, 3), "router": round(t_router, 3)}
        cellrec["topk_blocks"] = {"dense_limit": topk_all, "router": topk}

        band = sage_band(blk, args.step)
        cellrec["sage_band_rel_l2_mean"] = band
        bar = (band["auto"] * BAND_FACTOR) if band and "auto" in band else None
        # Violation 1 is "the kernel reads the table": zeroed must be worse than
        # the router AND far worse than the dense limit. The first draft asked
        # for 3x the router's error, which a head whose router error is already
        # ~0.4 cannot satisfy -- relative L2 saturates near 1.0 (the error of
        # emitting zeros), so the factor was measuring the head, not the kernel.
        v1 = zeroed > router and zeroed > 10.0 * dense_lim
        v2 = permuted > router
        v0 = (bar is not None and dense_lim <= bar)
        cellrec["verdict"] = {"dense_limit_within_band": v0, "bar": bar,
                              "lut_zeroed_red": v1, "lut_permuted_red": v2}
        ok = v0 and v1 and v2
        all_pass &= ok
        out["cells"].append(cellrec)
        print(f"b{blk} h{head}: dense_limit {dense_lim:.4f} (bar {bar if bar is None else round(bar, 4)}) "
              f"router@{args.sparsity:g} {router:.4f} | zeroed {zeroed:.4f} permuted {permuted:.4f} | "
              + " ".join(f"sol@{t:g} {cellrec['arms'][f'sol_tau_{t:g}']['rel_l2_mean']:.4f}" for t in taus)
              + f" | {'PASS' if ok else 'FAIL'}", flush=True)
        del q, k, v, ref
        torch.cuda.empty_cache()

    out["all_pass"] = all_pass
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("written", args.out, "--", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
