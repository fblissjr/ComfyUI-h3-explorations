#!/usr/bin/env python3
"""Sol's four quantizers on captured cells, every head: error against fp32 dense.

Written for the Sol redesign (2026-09-27, `docs/wiki/archive/2026-09-27_sol_node_redesign.md`).
Test 1 showed `quantizer` balanced and plain render differently in every mode,
and `bench/measure_qk_balance_gate_on_capture.py` showed why: the balance
gate opens on some heads of every captured block, not only on the blocks the
shipped `dense_blocks` sends dense. Whether that change helps is numerical,
so it is graded here on captures, not on renders.

For each cell it runs `comfy_kitchen.sol_attn` under plain, balanced
(`qk_balance`), rotated (`rotate`) and both, at the node's fixed settings
(`topk_ratio` 0, the pooled tail on, no token routing) and the given tau, and
reports two rel L2 terms, both over the fp32 dense norm: `total` against fp32
dense attention (SDPA in fp32), and `quant` against the unquantized Sol
reference (`analyze_sol_error.eager_sol_reference`, the decomposition the
2026-09-15 records use). Sparsity dominates `total`, so `quant` is the term
that decides a quantizer. Each is pooled over all heads, and over heads whose
balance gate is open and heads whose gate is shut. On shut heads, balanced must equal plain bit for bit; the script
asserts it, and a failure means the gate here is not the kernel's gate.

The sink is off (`sink_blocks` [0, 0]), as in
`grade_sol_token_aug_x_options.py`; the sink moves routing, not quantization.

    python bench/grade_sol_quantizer_on_capture.py CELL.pt [CELL.pt ...] [--tau 1.0] [--json out.json]
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sol_error import eager_sol_reference  # noqa: E402

import comfy_kitchen
from comfy_kitchen.backends.eager.sol_attn import QK_BALANCE_MIN_SHARE, QK_BALANCE_TOP

QUANTIZERS = {"plain": {}, "balanced": {"qk_balance": True},
              "rotated": {"rotate": True}, "balanced+rotated": {"qk_balance": True, "rotate": True}}
CELL = re.compile(r"_b(\d+)_s(\d+)")
GROUP = 8  # heads per pass; bounds GPU memory at 104k tokens


def gate_open(k: torch.Tensor) -> torch.Tensor:
    """(H,) bool: the kernel's per-head balance gate. k is (1, H, S, D) on the GPU."""
    e = k[0].float().pow(2).sum(1)
    share = torch.topk(e, QK_BALANCE_TOP, dim=-1).values.sum(-1) / e.sum(-1).clamp(min=1e-30)
    return share >= QK_BALANCE_MIN_SHARE


def grade(path: str, tau: float) -> dict:
    cap = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    q, k, v = cap["q"], cap["k"], cap["v"]
    h = q.shape[1]
    err = {name: torch.zeros(h, dtype=torch.float64) for name in QUANTIZERS}
    qerr = {name: torch.zeros(h, dtype=torch.float64) for name in QUANTIZERS}
    ref = torch.zeros(h, dtype=torch.float64)
    opened = torch.zeros(h, dtype=torch.bool)
    for g in range(0, h, GROUP):
        qg, kg, vg = (x[:, g:g + GROUP].cuda() for x in (q, k, v))
        opened[g:g + GROUP] = gate_open(kg).cpu()
        dense = F.scaled_dot_product_attention(qg.float(), kg.float(), vg.float())
        ref[g:g + GROUP] = dense.pow(2).sum((0, 2, 3)).double().cpu()
        eager = eager_sol_reference(q[:, g:g + GROUP], k[:, g:g + GROUP], v[:, g:g + GROUP], tau=tau).cuda()
        args = [x.permute(0, 2, 1, 3).contiguous() for x in (qg, kg, vg)]
        outs = {}
        for name, kw in QUANTIZERS.items():
            o = comfy_kitchen.sol_attn(*args, tau=tau, scale=None, sink_blocks=[0, 0], sink_q=[0, 0],
                                       topk_ratio=0.0, tail=True, token_aug=0, **kw)
            outs[name] = o
            d = o.permute(0, 2, 1, 3).float() - dense
            err[name][g:g + GROUP] = d.pow(2).sum((0, 2, 3)).double().cpu()
            d = o.permute(0, 2, 1, 3).float() - eager
            qerr[name][g:g + GROUP] = d.pow(2).sum((0, 2, 3)).double().cpu()
        shut = ~opened[g:g + GROUP].cuda()
        if shut.any() and not torch.equal(outs["plain"][:, :, shut], outs["balanced"][:, :, shut]):
            raise SystemExit(f"{path}: balanced differs from plain on a head whose gate is shut")
        del qg, kg, vg, dense, eager, args, outs
        torch.cuda.empty_cache()

    def pooled(mask: torch.Tensor) -> dict:
        if not mask.any():
            return {}
        den = ref[mask].sum().sqrt()
        return {term: {name: round(float(e[mask].sum().sqrt() / den), 7) for name, e in errs.items()}
                for term, errs in (("total", err), ("quant", qerr))}

    m = CELL.search(path)
    assert m, path
    return {"cell": path.rsplit("/", 2)[-2] + "/" + path.rsplit("/", 1)[-1],
            "block": int(m.group(1)), "step": int(m.group(2)), "heads": h,
            "heads_open": int(opened.sum()), "tau": tau,
            "all_heads": pooled(torch.ones(h, dtype=torch.bool)),
            "open_heads": pooled(opened), "shut_heads": pooled(~opened)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Sol quantizers on captured cells, every head")
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = []
    for p in args.cells:
        r = grade(p, args.tau)
        rows.append(r)
        a = r["all_heads"]["quant"]
        print(f"b{r['block']:>2} s{r['step']:>2} open {r['heads_open']:>2}/{r['heads']}  quant  "
              + "  ".join(f"{n} {a[n]:.6f}" for n in QUANTIZERS), flush=True)
        if args.json:
            json.dump({"what": "Sol quantizers: rel L2 over the fp32 dense norm, total (vs dense) and quant (vs the unquantized Sol reference), pooled over heads",
                       "tool": "bench/grade_sol_quantizer_on_capture.py",
                       "kernel": importlib.metadata.version("comfy-kitchen"), "rows": rows},
                      open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
