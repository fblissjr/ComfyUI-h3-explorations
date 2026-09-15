#!/usr/bin/env python3
"""Time comfy_kitchen.sol_attn on a capture: plain vs qk_balance vs rotate, CUDA events, warm.

    python bench/time_sol_options_on_capture.py <capture.pt> [--heads 0=all] [--iters 5]

Reports the median call time per arm and the ratio to plain. Kernel time
only: what a render pays is this times the calls Sol takes, and nothing here
says how many that is. Needs a quiet card.
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sol_error import load_capture, load_cuda_kernel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--heads", type=int, default=0)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--token-aug", type=int, default=0, help="token routing budget applied to every arm (0 off)")
    args = ap.parse_args()
    q, k, v = load_capture(args.capture)
    if 0 < args.heads < q.shape[1]:
        q, k, v = q[:, :args.heads], k[:, :args.heads], v[:, :args.heads]
    to = dict(device="cuda", dtype=torch.bfloat16)
    qs, ks, vs = (x.to(**to).permute(0, 2, 1, 3).contiguous() for x in (q, k, v))
    del q, k, v
    sol = load_cuda_kernel(); params = inspect.signature(sol).parameters
    arms = {"plain": {}}
    if "qk_balance" in params: arms["balanced"] = {"qk_balance": True}
    if "rotate" in params:
        arms["rotated"] = {"rotate": True}
        if "qk_balance" in params: arms["both"] = {"rotate": True, "qk_balance": True}
    m = re.search(r"_b(\d+)_s(\d+)", Path(args.capture).name)
    print(f"timing: {Path(args.capture).name}  S={qs.shape[1]} heads {qs.shape[2]}  tau {args.tau}  token_aug {args.token_aug}  iters {args.iters}")
    def call(kw):
        return sol(qs, ks, vs, tau=args.tau, scale=None, sink_blocks=[0, 0], sink_q=[0, 0], topk_ratio=0.0, tail=True, token_aug=args.token_aug, **kw)
    for kw in arms.values():   # warm every arm before any timing
        call(kw)
    torch.cuda.synchronize()
    # Round-robin over the arms each iteration, so clock drift and allocator
    # state land on every arm alike rather than on whichever ran last.
    times = {name: [] for name in arms}
    for _ in range(args.iters):
        for name, kw in arms.items():
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); call(kw); e.record(); torch.cuda.synchronize()
            times[name].append(s.elapsed_time(e))
    base = sorted(times["plain"])[args.iters // 2]
    for name, ts in times.items():
        med = sorted(ts)[len(ts) // 2]
        print(f"  {name:10s} median {med:8.1f} ms  min {min(ts):8.1f}  max {max(ts):8.1f}   x{med / base:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
