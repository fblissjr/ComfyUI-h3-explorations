#!/usr/bin/env python3
"""Does per-channel q/k balancing lower Sol's INT8 term on a captured cell?

Step 3 of the experiment in `docs/SOLATTN.md`, "The defaults, re-read against
the sage-side error records, 2026-09-14". The sage fork measured that
rebalancing K's channels before INT8 quantization cuts *sage's* error at block
49 by about a fifth. Whether *Sol's* CUDA kernel benefits is a separate
question, because it depends on how that kernel scales K, and this script is
the only thing that answers it: same capture, same decomposition as
`analyze_sol_error.py`, plain against balanced.

The balancing is exact for the attention math. For any per-channel factor s,
`q . k == (q * s) . (k / s)`, and Sol's routing threshold is invariant under the
same rescale in exact arithmetic (`c_d` scales by `s_d`, `kcvar_d` by
`1/s_d^2`). So the eager Sol reference must agree with itself plain and
balanced up to fp32 rounding -- that is checked and printed first, and if it
does not hold, nothing below it is a quantization result.

Two factor sources, both pair-equal within H3's split-half RoPE pairs
(i, i+48) so they fold into `q_norm.weight` / `k_norm.weight`:

  weights  from the checkpoint's own norm weights, no capture needed. What
           `MiniMaxH3ChannelBalance` computes at load.
  capture  from the captured q/k channel rms, shared across heads. The
           calibrated form; needs the very capture being graded.

Usage:
  python bench/grade_channel_balance.py --capture <cell.pt> --unet <unet.safetensors>
      [--tau 1.0] [--heads 8] [--alpha 0.5] [--source weights] [--json out.json]

`--heads` is a PREFIX like `analyze_sol_error.py`'s, not a sample. The sage
fp8++ column is included so the same cell and heads carry both kernels'
answers in one record.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sol_error import (  # noqa: E402
    dense_reference,
    eager_sol_reference,
    load_capture,
    load_cuda_kernel,
    rel_l2_against,
)

ROT, HALF = 96, 48


def pair_equal_unit(s: torch.Tensor) -> torch.Tensor:
    g = (s[:HALF] * s[HALF:ROT]).sqrt()
    s = torch.cat([g, g, s[ROT:]])
    return s / torch.exp(torch.log(s).mean())


def factor_from_weights(unet: str, block: int, alpha: float) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(unet, framework="pt", device="cpu") as sf:
        kw = sf.get_tensor(f"blocks.{block}.attn.k_norm.weight").float().abs().clamp(min=1e-6)
        qw = sf.get_tensor(f"blocks.{block}.attn.q_norm.weight").float().abs().clamp(min=1e-6)
    return pair_equal_unit(kw.pow(alpha) / qw.pow(1 - alpha))


def factor_from_capture(q: torch.Tensor, k: torch.Tensor, alpha: float) -> torch.Tensor:
    # [B, H, S, D]: channel rms over heads and tokens, a head subsample keeps
    # the fp32 temporaries small.
    sub = list(range(0, q.shape[1], max(1, q.shape[1] // 8)))[:8]
    rk = k[:, sub].float().pow(2).mean(dim=(0, 1, 2)).sqrt().clamp(min=1e-6)
    rq = q[:, sub].float().pow(2).mean(dim=(0, 1, 2)).sqrt().clamp(min=1e-6)
    return pair_equal_unit(rk.pow(alpha) / rq.pow(1 - alpha))


def kernel_takes_qk_balance() -> bool:
    """Whether the installed kernel carries the in-quantizer balance (the
    owner's fork, h3-build from 2026-09-15); a stock wheel does not."""
    import inspect

    return "qk_balance" in inspect.signature(load_cuda_kernel()).parameters


def cuda_sol(q, k, v, tau, **extra):
    """comfy_kitchen.sol_attn as `sol_attn_h3._run` calls it on the served build.

    `analyze_sol_error.cuda_sol_kernel` still passes `centroid_tail=`, which
    the installed kernel no longer accepts (comfy-kitchen#117 made that form
    unconditional), so the call is made here with the node's own argument
    set: no sinks, pooled tail on, adaptive tau. Inputs [B, H, S, D]; returns
    [B, H, S, D] fp32 on CPU, matching the other references.
    """
    sol_fn = load_cuda_kernel()
    to = dict(device="cuda", dtype=torch.bfloat16)
    out = sol_fn(q.to(**to).permute(0, 2, 1, 3).contiguous(), k.to(**to).permute(0, 2, 1, 3).contiguous(),
                 v.to(**to).permute(0, 2, 1, 3).contiguous(),
                 tau=tau, scale=None, sink_blocks=[0, 0], sink_q=[0, 0], topk_ratio=0.0, tail=True,
                 **extra)
    return out.permute(0, 2, 1, 3).float().cpu()


def sage_fp8pp(q, k, v, qk_balance=False):
    import sageattention

    # `qk_balance` is the fork's per-head factor inside its own quantizer (its
    # v0.7.19); forwarded only when asked so a build without it still grades
    # the plain column.
    extra = {"qk_balance": True} if qk_balance else {}
    return sageattention.sageattn_qk_int8_pv_fp8_cuda(
        q.cuda(), k.cuda(), v.cuda(), tensor_layout="HND", is_causal=False,
        pv_accum_dtype="fp32+fp16", smooth_k=False, **extra,
    ).float().cpu()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--unet", required=True, help="unet safetensors the capture was rendered with")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--heads", type=int, default=8, help="head PREFIX (0 = all)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--source", choices=("weights", "capture"), default="weights")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    m = re.search(r"_b(\d+)_s(\d+)", Path(args.capture).name)
    if m is None:
        raise SystemExit(f"cannot read block/step from the capture name: {Path(args.capture).name}")
    block, step = int(m.group(1)), int(m.group(2))
    q, k, v = load_capture(args.capture)
    if 0 < args.heads < q.shape[1]:
        q, k, v = q[:, : args.heads], k[:, : args.heads], v[:, : args.heads]
    s = factor_from_weights(args.unet, block, args.alpha) if args.source == "weights" \
        else factor_from_capture(q, k, args.alpha)
    sc = s.to(q.dtype).view(1, 1, 1, -1)
    qb, kb = (q * sc).contiguous(), (k / sc).contiguous()
    print(f"cell block {block} step {step}  S={q.shape[2]} heads {q.shape[1]} of capture  tau {args.tau}  "
          f"factor {args.source} a={args.alpha}  factor range {s.min():.3f}..{s.max():.3f}")

    # Balancing re-rounds q and k to bf16, and on a block with a hundred-logit
    # range that re-rounding alone moves exact attention. So each arm gets
    # its own exact referent (dense on the same bf16 inputs) and the
    # quantization term is kernel-vs-eager on the same inputs, which isolates
    # the INT8 arithmetic from the input rounding. The rounding effect is
    # printed on its own line so it is not read as a routing change.
    dense = dense_reference(q, k, v)
    dense_b = dense_reference(qb, kb, v)
    dn = dense.float().norm().item()
    rounding = rel_l2_against(dense_b, dense, dn)
    eager_p = eager_sol_reference(q, k, v, tau=args.tau)
    eager_b = eager_sol_reference(qb, kb, v, tau=args.tau)
    routing = rel_l2_against(eager_b, eager_p, dn)
    print(f"bf16 re-rounding of the balanced inputs moves exact attention by rel L2 {rounding:.2e}; "
          f"eager Sol balanced vs plain {routing:.2e} (routing invariant if these are of one size)")

    cuda_p = cuda_sol(q, k, v, args.tau)
    cuda_b = cuda_sol(qb, kb, v, args.tau)
    rows = {
        "plain": dict(sparsity_l2=rel_l2_against(eager_p, dense, dn), quant_l2=rel_l2_against(cuda_p, eager_p, dn),
                      total_l2=rel_l2_against(cuda_p, dense, dn)),
        "balanced": dict(sparsity_l2=rel_l2_against(eager_b, dense_b, dn), quant_l2=rel_l2_against(cuda_b, eager_b, dn),
                         total_l2=rel_l2_against(cuda_b, dense_b, dn)),
    }
    # The kernel's own per-head factor (`qk_balance=True`), on the UNMODIFIED
    # inputs: no re-rounding, no weights fold, the same referents as `plain`.
    # This is the arm the Sol node's `qk_balance` widget turns on; the two
    # rows above are the weights fold the ChannelBalance node applies.
    if kernel_takes_qk_balance():
        cuda_k = cuda_sol(q, k, v, args.tau, qk_balance=True)
        rows["kernel"] = dict(sparsity_l2=rows["plain"]["sparsity_l2"], quant_l2=rel_l2_against(cuda_k, eager_p, dn),
                              total_l2=rel_l2_against(cuda_k, dense, dn))
    else:
        print("(kernel row skipped: the installed comfy_kitchen.sol_attn has no qk_balance)")
    try:
        sage_p, sage_b = sage_fp8pp(q, k, v), sage_fp8pp(qb, kb, v)
        rows["plain"]["sage_fp8pp_l2"] = rel_l2_against(sage_p, dense, dn)
        rows["balanced"]["sage_fp8pp_l2"] = rel_l2_against(sage_b, dense_b, dn)
    except Exception as exc:  # sage absent from this venv is not a Sol result
        print(f"(sage column skipped: {type(exc).__name__}: {exc})")
    # Does the fold add anything once sage balances per head itself? Row
    # `plain` is sage's own factor alone; row `balanced` is the fold under it.
    # If the second is not lower, the node is redundant on a sage chain that
    # runs the balanced mode, and the chain needs one node, not two.
    try:
        rows["plain"]["sage_balanced_l2"] = rel_l2_against(sage_fp8pp(q, k, v, qk_balance=True), dense, dn)
        rows["balanced"]["sage_balanced_l2"] = rel_l2_against(sage_fp8pp(qb, kb, v, qk_balance=True), dense_b, dn)
    except Exception as exc:
        print(f"(sage balanced column skipped: {type(exc).__name__}: {exc})")

    cols = ["sparsity_l2", "quant_l2", "total_l2", "sage_fp8pp_l2", "sage_balanced_l2"]
    print(f"{'arm':10s}" + "".join(f"{c:>17s}" for c in cols))
    for arm, r in rows.items():
        print(f"{arm:10s}" + "".join(f"{r[c]:>17.4f}" if c in r else f"{'-':>17s}" for c in cols))
    qp, qb_ = rows["plain"]["quant_l2"], rows["balanced"]["quant_l2"]
    print(f"Sol INT8 term, weights fold: {qp:.4f} -> {qb_:.4f} ({100 * (qb_ - qp) / qp:+.1f}%)")
    if "kernel" in rows:
        qk_ = rows["kernel"]["quant_l2"]
        print(f"Sol INT8 term, kernel qk_balance: {qp:.4f} -> {qk_:.4f} ({100 * (qk_ - qp) / qp:+.1f}%)")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "capture": Path(args.capture).name, "block": block, "step": step, "tau": args.tau,
            "heads_measured": int(q.shape[1]), "alpha": args.alpha, "source": args.source,
            "input_rounding_rel_l2": rounding, "eager_balanced_vs_plain_rel_l2": routing, "rows": rows,
        }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
