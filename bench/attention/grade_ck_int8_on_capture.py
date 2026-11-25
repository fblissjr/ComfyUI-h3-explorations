"""Kitchen's own int8_attention (the --use-ck-attention path) on the block-49 capture: plain vs the weights fold."""
import sys, json, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grade_channel_balance import factor_from_weights
from analyze_sol_error import load_capture, dense_reference, rel_l2_against
import comfy_kitchen as ck
cap, unet, heads = sys.argv[1], sys.argv[2], 8
import re; m = re.search(r"_b(\d+)_s(\d+)", cap); block, step = int(m.group(1)), int(m.group(2))
q, k, v = load_capture(cap); q, k, v = q[:, :heads], k[:, :heads], v[:, :heads]
s = factor_from_weights(unet, block, 0.5).to(q.dtype).view(1, 1, 1, -1)
qb, kb = (q * s).contiguous(), (k / s).contiguous()
dense = dense_reference(q, k, v); dense_b = dense_reference(qb, kb, v); dn = dense.float().norm().item()
def ck_int8(q, k, v):
    return ck.int8_attention(q.cuda(), k.cuda(), v.cuda()).float().cpu()   # BHND, as core hands it
out = {"kernel": "comfy_kitchen.int8_attention (stock design, served wheel)", "block": block, "step": step, "heads": heads,
       "plain_rel_l2": rel_l2_against(ck_int8(q, k, v), dense, dn),
       "weights_fold_rel_l2": rel_l2_against(ck_int8(qb, kb, v), dense_b, dn)}
print(json.dumps(out, indent=2))
