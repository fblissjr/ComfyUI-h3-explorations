#!/usr/bin/env python3
"""Grade Sol's ROUTE, not its output: how far the INT8 kernel's routed-block
count per query block sits from the fp32 route, plain vs qk_balance vs rotate.

Sol decides which key blocks a query block attends exactly by comparing INT8
centroid-vs-pooled-key scores against a threshold. The output grades
(`grade_channel_balance.py`) fold that decision into one number with the
quantization of the exact branch; this one isolates the decision. The fp32
route is recomputed here from the same captured q/k with the eager
reference's rule (query-block centroid vs centred pooled keys, column mean
over the query block, threshold tau * sigma from the centroid and the pooled
keys' per-channel variance), block-level only, so it fits at H3 length. The
kernel's route comes back through `blk_cnt` (counts per query block: sink
range + diagonal + routed set), so the comparison is on COUNTS: a query
block that swapped one routed block for another at equal count reads as
agreement here. That is the instrument's limit and is said in the output.

    python bench/grade_sol_route_on_capture.py <capture.pt> [--tau 1.0] [--heads 8] [--json out]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_sol_error import load_capture, load_cuda_kernel  # noqa: E402

BLOCK = 64   # Sol's routing granularity in tokens (sol_layout.cuh)

_LOG2E = 1.4426950408889634


def fp32_route_counts(q, k, tau, scale=None):
    """[B, H, S, D] -> int32 [B, H, NQ]: the eager rule's exact-block count per
    query block (diagonal +-1 forced, no sinks), fp32, block level."""
    b, h, t, d = q.shape
    assert b == 1, "the reference fills batch 0 only; captures are B=1"
    # Every row is live here (no block_len), so the query block's column mean
    # equals centroid . kcc exactly; with dead rows it would not.
    n = (t + BLOCK - 1) // BLOCK
    scale = d ** -0.5 if scale is None else scale
    log2s = scale * _LOG2E
    dev = torch.device("cuda")
    counts = torch.zeros(b, h, n, dtype=torch.int32)
    idx = torch.arange(n, device=dev)
    diag = ((idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1)
    pad = n * BLOCK - t
    for hh in range(h):
        qf = q[0, hh].to(dev, torch.float32)
        kf = k[0, hh].to(dev, torch.float32)
        if pad:
            qf = torch.cat([qf, qf.new_zeros(pad, d)]); kf = torch.cat([kf, kf.new_zeros(pad, d)])
        lengths = torch.full((n,), float(BLOCK), device=dev)
        if pad:
            lengths[-1] = float(BLOCK - pad)
        kc = kf.view(n, BLOCK, d).sum(1) / lengths[:, None]           # pooled keys (N, D)
        kmean = kc.mean(0, keepdim=True)
        kcc = kc - kmean
        kc_var = kcc.pow(2).mean(0)                                    # (D,)
        cen = qf.view(n, BLOCK, d).sum(1) / lengths[:, None]           # centroids (N, D)
        thr = tau * torch.sqrt((cen.pow(2) * kc_var).sum(-1) * log2s * log2s + 1e-6)   # (N,)
        # column mean of q.kcc over each query block == centroid . kcc
        colmean = (cen @ kcc.T) * log2s                                # (N, N)
        exact = colmean > thr[:, None]
        exact |= diag
        counts[0, hh] = exact.sum(-1).to(torch.int32).cpu()
    return counts


def kernel_route_counts(q, k, v, tau, **extra):
    sol = load_cuda_kernel()
    to = dict(device="cuda", dtype=torch.bfloat16)
    qs, ks, vs = (x.to(**to).permute(0, 2, 1, 3).contiguous() for x in (q, k, v))
    b, t, h, _ = qs.shape
    cnt = torch.empty(b, h, (t + BLOCK - 1) // BLOCK, dtype=torch.int32, device="cuda")
    sol(qs, ks, vs, tau=tau, scale=None, sink_blocks=[0, 0], sink_q=[0, 0], topk_ratio=0.0, tail=True,
        blk_cnt=cnt, **extra)
    return cnt.cpu()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()
    m = re.search(r"_b(\d+)_s(\d+)", Path(args.capture).name)
    block, step = (int(m.group(1)), int(m.group(2))) if m else (-1, -1)
    q, k, v = load_capture(args.capture)
    if 0 < args.heads < q.shape[1]:
        q, k, v = q[:, :args.heads], k[:, :args.heads], v[:, :args.heads]
    import inspect
    params = inspect.signature(load_cuda_kernel()).parameters
    arms = {"plain": {}}
    if "qk_balance" in params:
        arms["balanced"] = {"qk_balance": True}
    if "rotate" in params:
        arms["rotated"] = {"rotate": True}
        if "qk_balance" in params:
            arms["both"] = {"rotate": True, "qk_balance": True}
    ref = fp32_route_counts(q, k, args.tau)
    n = ref.shape[-1]
    rows = {}
    for name, kw in arms.items():
        got = kernel_route_counts(q, k, v, args.tau, **kw)
        diff = (got - ref).abs().float()
        rows[name] = {
            "mean_abs_count_diff": diff.mean().item(),
            "mean_signed_count_diff": (got - ref).float().mean().item(),   # kernel minus fp32: + adds blocks, - drops
            "query_blocks_with_equal_count": (diff == 0).float().mean().item(),
            "routed_density_kernel": got.float().mean().item() / n,
            "routed_density_fp32": ref.float().mean().item() / n,
        }
    print(f"route grade: block {block} step {step}  S={q.shape[2]}  heads {q.shape[1]}  tau {args.tau}  "
          f"({n} key blocks per query block; counts only, a swap at equal count reads as agreement)")
    print(f"{'arm':10s}{'mean |dcount|':>15s}{'signed':>9s}{'equal count':>13s}{'density':>10s}")
    for name, r in rows.items():
        print(f"{name:10s}{r['mean_abs_count_diff']:>15.3f}{r['mean_signed_count_diff']:>+9.3f}{r['query_blocks_with_equal_count']:>13.1%}{r['routed_density_kernel']:>10.3%}")
    print(f"fp32 route density {rows['plain']['routed_density_fp32']:.3%}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "capture": Path(args.capture).name, "block": block, "step": step, "tau": args.tau,
            "heads_measured": int(q.shape[1]), "key_blocks": n, "instrument": "blk_cnt counts vs an fp32 block-level route recomputed with the eager rule; counts only",
            "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
