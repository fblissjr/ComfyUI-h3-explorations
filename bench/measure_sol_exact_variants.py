#!/usr/bin/env python3
"""Measure the INSTALLED Sol kernel's exact branch, so two builds can be compared.

Two builds of comfy-kitchen carry the same routing and a different exact
stage: upstream main's (`0.2.31+sol.24908e1`, our `blk_cnt` branch) quantizes
each key block's probabilities against the running maximum of every block
walked so far; kijai's `sol_attn_continued` quantizes against the block's own
maximum, floored twenty doublings under the running one, so a block far below
the running max is not squeezed into the bottom of the 8-bit range. His test
pins the effect at all-routed as relative L2 under 1.6% against about 2%.

This script measures that on whichever build is installed and writes a record
naming it, so running it once per wheel gives a like-for-like comparison at
the call level -- the only level at which a numerical kernel change can be
compared (CLAUDE.md: a rendered clip cannot A/B a numerical change).

Arms, all on the same seeded inputs:

  exact_all_routed      tau -1e9 (every block exact): relative L2 and cosine
                        against fp32 dense attention. kijai's own metric.
  tau_1p0_vs_eager      cosine against the eager reference at the shipped
                        tau, the bar check_solattn_correctness.py uses.
  tau_1p3_vs_eager      the same at the tau that page's older tables used.
  blk_cnt_forced_pairs  the routed count at tau +1e9 equals the closed form,
                        proving the count still comes from the same launch
                        on this build (identical routing is the premise).
  kernel_ms             median of a warm isolated call at a mid shape.
                        ISOLATED KERNEL TIME, not a render time; comparable
                        only between two runs of this script on the same box.

    python bench/measure_sol_exact_variants.py --out bench/results/<date>_sol_exact_<tag>.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--timing-shape", default="1,16384,8", help="B,T,H for the isolated timing")
    args = ap.parse_args()
    try:
        import importlib.metadata as md
        import torch
        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.sol_attn import sol_attn as eager
    except Exception as exc:                          # noqa: BLE001
        print(f"SKIP: {exc}")
        return 2
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 2
    version = md.version("comfy_kitchen")
    has_cnt = "blk_cnt" in inspect.signature(ck.sol_attn).parameters
    rec = {"produced_by": "bench/measure_sol_exact_variants.py", "comfy_kitchen": version,
           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "arms": {}}
    print(f"installed {version} on {rec['device']}; blk_cnt {'present' if has_cnt else 'ABSENT'}\n")

    def qkv(b, t, h, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        mk = lambda s: torch.randn(b, t, h, 128, device="cuda", dtype=torch.bfloat16, generator=g) * s  # noqa: E731
        return mk(0.5), mk(0.5), mk(1.0)

    def dense(q, k, v):
        qq, kk, vv = (x.permute(0, 2, 1, 3).float() for x in (q, k, v))
        return torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, scale=128 ** -0.5).permute(0, 2, 1, 3)

    def cos(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        return float(a @ b / (a.norm() * b.norm()))

    def rel(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm())

    # kijai's metric, his shape and seed convention
    q, k, v = qkv(1, 4096, 8, 0)
    out = ck.sol_attn(q, k, v, tau=-1e9)
    ref = dense(q, k, v)
    rec["arms"]["exact_all_routed"] = {"shape": [1, 4096, 8], "rel_l2_vs_dense": rel(out, ref), "cos_vs_dense": cos(out, ref)}
    print(f"  exact, all routed (tau -1e9), T=4096 H=8: rel L2 {rel(out, ref):.5f}, cos {cos(out, ref):.6f}")

    for tau in (1.0, 1.3):
        got = ck.sol_attn(q, k, v, tau=tau)
        want = eager(q, k, v, tau=tau)
        rec["arms"][f"tau_{tau}_vs_eager"] = {"shape": [1, 4096, 8], "cos": cos(got, want), "rel_l2": rel(got, want)}
        print(f"  tau {tau} vs eager reference: cos {cos(got, want):.6f}, rel L2 {rel(got, want):.5f}")

    if has_cnt:
        t, h, n = 320, 2, 5
        q2, k2, v2 = qkv(1, t, h, 7)
        cnt = torch.empty(1, h, n, dtype=torch.int32, device="cuda")
        ck.sol_attn(q2, k2, v2, tau=1e9, sink_blocks=[0, 2], sink_q=[3, 4], blk_cnt=cnt)
        ok = cnt[0, 0].tolist() == [2, 3, 4, 5, 4] and cnt[0, 1].tolist() == [2, 3, 4, 5, 4]
        rec["arms"]["blk_cnt_forced_pairs"] = {"ok": ok, "got": cnt[0, 0].tolist(), "want": [2, 3, 4, 5, 4]}
        print(f"  blk_cnt forced pairs: {'ok' if ok else 'FAIL'} {cnt[0, 0].tolist()}")
    else:
        rec["arms"]["blk_cnt_forced_pairs"] = {"ok": None, "why": "this build has no blk_cnt"}

    b, t, h = (int(x) for x in args.timing_shape.split(","))
    q3, k3, v3 = qkv(b, t, h, 11)
    for _ in range(3):
        ck.sol_attn(q3, k3, v3, tau=1.0)
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); ck.sol_attn(q3, k3, v3, tau=1.0); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    rec["arms"]["kernel_ms"] = {"shape": [b, t, h], "tau": 1.0, "median_ms": times[len(times) // 2],
                                "min_ms": times[0], "note": "isolated warm kernel call, not a render time"}
    print(f"  isolated kernel call, B={b} T={t} H={h}, tau 1.0: median {times[len(times) // 2]:.2f} ms (min {times[0]:.2f})")

    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
        print(f"\nrecord written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
