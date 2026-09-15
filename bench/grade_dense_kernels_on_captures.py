#!/usr/bin/env python3
"""The three dense-step kernels on every captured cell: error against fp32 attention, per block and step.

Every block-49 grade so far is at the last step. Composition is decided on
the early steps, and the coins controls (2026-09-15) put a visible flaw on
the kitchen dense kernel's early steps that its last-step grade could not
see. This runs the three kernels a graph can put on the dense steps over
every cell of a capture set, same inputs, same fp32 referent:

    python bench/grade_dense_kernels_on_captures.py <capture dir> [--heads 8] [--json out]

Kernels: comfy_kitchen.int8_attention (Model Attention Backend), sage fp8++
(sageattn_qk_int8_pv_fp8_cuda, pv fp32+fp16, as the Sage node ships it),
and bf16 SDPA on the same bf16 inputs (the "dense" arm; its error is the
bf16 output rounding plus torch's own accumulation, the floor). Sol is not
here: it is the routed-step kernel, graded elsewhere.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sol_error import dense_reference, load_capture, rel_l2_against  # noqa: E402


def kitchen_int8(q, k, v):
    import comfy_kitchen as ck
    return ck.int8_attention(q.cuda(), k.cuda(), v.cuda()).float().cpu()


def sage_fp8pp(q, k, v):
    import sageattention
    return sageattention.sageattn_qk_int8_pv_fp8_cuda(
        q.cuda(), k.cuda(), v.cuda(), tensor_layout="HND", is_causal=False,
        pv_accum_dtype="fp32+fp16", smooth_k=False).float().cpu()


def bf16_sdpa(q, k, v):
    outs = []
    for hh in range(q.shape[1]):
        outs.append(torch.nn.functional.scaled_dot_product_attention(
            q[:, hh:hh + 1].cuda(), k[:, hh:hh + 1].cuda(), v[:, hh:hh + 1].cuda()).float().cpu())
    return torch.cat(outs, dim=1)


KERNELS = {"kitchen_int8": kitchen_int8, "sage_fp8pp": sage_fp8pp, "bf16_sdpa": bf16_sdpa}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture_dir")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--steps", default="", help="comma-separated steps to grade (default all)")
    ap.add_argument("--blocks", default="", help="comma-separated blocks to grade (default all)")
    args = ap.parse_args()
    want_steps = {int(x) for x in args.steps.split(",") if x}
    want_blocks = {int(x) for x in args.blocks.split(",") if x}
    cells = sorted(Path(args.capture_dir).glob("qkv_*_b*_s*.pt"),
                   key=lambda p: tuple(int(x) for x in re.search(r"_b(\d+)_s(\d+)", p.name).groups()))
    rows = []
    print(f"{'cell':14s}" + "".join(f"{k:>14s}" for k in KERNELS))
    for cell in cells:
        m = re.search(r"_b(\d+)_s(\d+)", cell.name)
        block, step = int(m.group(1)), int(m.group(2))
        if (want_steps and step not in want_steps) or (want_blocks and block not in want_blocks):
            continue
        q, k, v = load_capture(str(cell))
        if 0 < args.heads < q.shape[1]:
            q, k, v = q[:, :args.heads], k[:, :args.heads], v[:, :args.heads]
        dense = dense_reference(q, k, v); dn = dense.float().norm().item()
        row = {"block": block, "step": step}
        for name, fn in KERNELS.items():
            try:
                row[name] = rel_l2_against(fn(q, k, v), dense, dn)
            except Exception as exc:  # a kernel absent from this venv is a hole, not a failure
                row[name] = None
                row[name + "_error"] = f"{type(exc).__name__}: {exc}"[:120]
        rows.append(row)
        print(f"b{block:<3d}s{step:<9d}" + "".join(f"{row[k]:>14.4f}" if row[k] is not None else f"{'-':>14s}" for k in KERNELS))
        del q, k, v, dense
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "capture_dir": Path(args.capture_dir).name, "heads_measured": args.heads,
            "referent": "fp32 dense attention on the same bf16 inputs (analyze_sol_error.dense_reference)",
            "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
