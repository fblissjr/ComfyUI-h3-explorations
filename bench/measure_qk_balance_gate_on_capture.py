#!/usr/bin/env python3
"""Which heads open the kitchen Sol kernel's qk_balance gate, per captured block.

The gate (`comfy_kitchen/backends/cuda/sage_attention/sol_attn_preprocess.cu`,
`prep_balance_factor`; mirrored in the eager reference's `qk_balance_factor`):
per (batch, head), take K's per-channel sum of squares over the live rows,
UNCENTRED; the head's factor is applied only when its `QK_BALANCE_TOP` (4)
loudest channels hold at least `QK_BALANCE_MIN_SHARE` (0.2) of the total.
Otherwise the head's codes are bit-identical to the plain path.

Written for the Sol redesign's test 1 (2026-09-27,
`docs/research/2026-09-27_sol_node_redesign.md`): the audit had inferred from
`docs/attention/block49_quant_error.md` ("blocks 45, 48 and 49 open on their own",
a statement about the q/k norm WEIGHTS) that the gate stays shut everywhere
else, so that qk_balance is inert behind the shipped dense tail. Test 1's
renders say it is not inert. This reads the gate off real activations.

CPU only: K is read through a memory map in sequence chunks. The constants
are imported from the installed kitchen's eager reference, never retyped.

    python bench/measure_qk_balance_gate_on_capture.py DIR [DIR ...] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

from comfy_kitchen.backends.eager.sol_attn import QK_BALANCE_MIN_SHARE, QK_BALANCE_TOP

NAME = re.compile(r"^qkv_L\w+_S(\d+)_b(\d+)_s(\d+)(?:_k\w+?)?(?:_r(\d+))?\.pt$")


def head_shares(k: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """(H,) top-QK_BALANCE_TOP energy share per head. k is (1, H, S, D)."""
    _, h, s, d = k.shape
    e = torch.zeros(h, d, dtype=torch.float64)
    for i in range(0, s, chunk):
        e += k[0, :, i:i + chunk].float().pow(2).sum(1, dtype=torch.float64)
    top = torch.topk(e, QK_BALANCE_TOP, dim=-1).values.sum(-1)
    return (top / e.sum(-1)).float()


def main() -> int:
    ap = argparse.ArgumentParser(description="qk_balance gate per captured block")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = []
    for d in args.dirs:
        seen = set()
        for f in sorted(os.listdir(d)):
            m = NAME.match(f)
            if not m or (m.group(4) not in (None, "0")):
                continue          # first render of each cell only
            block, step = int(m.group(2)), int(m.group(3))
            if (block, step) in seen:
                continue
            seen.add((block, step))
            cap = torch.load(os.path.join(d, f), map_location="cpu", mmap=True, weights_only=True)
            sh = head_shares(cap["k"])
            n_open = int((sh >= QK_BALANCE_MIN_SHARE).sum())
            row = {"capture": os.path.basename(d.rstrip("/")), "block": block, "step": step,
                   "heads": int(sh.numel()), "heads_open": n_open,
                   "share_max": round(float(sh.max()), 4), "share_median": round(float(sh.median()), 4)}
            rows.append(row)
            print(f"{row['capture']:45s} b{block:>2} s{step:>2}  open {n_open:>2}/{row['heads']}  "
                  f"max {row['share_max']:.3f}  median {row['share_median']:.3f}", flush=True)
            del cap
    if args.json:
        json.dump({"what": "qk_balance gate per captured block: heads whose top-"
                           f"{QK_BALANCE_TOP} K-channel energy share >= {QK_BALANCE_MIN_SHARE}",
                   "tool": "bench/measure_qk_balance_gate_on_capture.py", "rows": rows},
                  open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
