"""Token routing (token_aug=64) on top of plain / balanced / rotated / both, on a capture: total error vs fp32 dense attention.
The eager reference ignores token_aug, so the comparable referent is dense attention (total_l2), and the routed-only eager
gives the block-route sparsity term for scale."""
import sys, json, re
sys.path.insert(0, "bench")
from analyze_sol_error import load_capture, dense_reference, eager_sol_reference, rel_l2_against, load_cuda_kernel
import torch
cap = sys.argv[1]; m = re.search(r"_b(\d+)_s(\d+)", cap); block, step = int(m.group(1)), int(m.group(2))
q, k, v = load_capture(cap); q, k, v = q[:, :8], k[:, :8], v[:, :8]
dense = dense_reference(q, k, v); dn = dense.float().norm().item()
eager = eager_sol_reference(q, k, v, tau=1.0)
sol = load_cuda_kernel(); to = dict(device="cuda", dtype=torch.bfloat16)
qs, ks, vs = (x.to(**to).permute(0, 2, 1, 3).contiguous() for x in (q, k, v))
def run(**kw):
    return sol(qs, ks, vs, tau=1.0, scale=None, sink_blocks=[0, 0], sink_q=[0, 0], topk_ratio=0.0, tail=True, **kw).permute(0, 2, 1, 3).float().cpu()
rows = {"eager_block_route_only": {"total_l2": rel_l2_against(eager, dense, dn)}}
for name, kw in (("plain", {}), ("balanced", {"qk_balance": True}), ("rotated", {"rotate": True}), ("both", {"rotate": True, "qk_balance": True})):
    for aug in (0, 64):
        out = run(token_aug=aug, **kw)
        rows[f"{name}_tok{aug}"] = {"total_l2": rel_l2_against(out, dense, dn), "quant_vs_eager_l2": rel_l2_against(out, eager, dn)}
print(json.dumps({"block": block, "step": step, "heads": 8, "tau": 1.0, "rows": rows}, indent=2))
