#!/usr/bin/env python3
"""Time comfy_kitchen.int8_attention (the dense kernel) on captured q/k/v cells.

    <comfy venv python> bench/time_dense_on_capture.py <capture.pt> [...] --out OUT.json

The companion to `bench/profile_sol_stages.py`: the same tensors through the
dense kernel, so Sol's exact stage and the dense kernel can be compared per
attended pair (exact-stage ms over routed density, against dense ms). Events
around five calls after two warmups. Needs a quiet card.
"""
import argparse
import json
import torch
import comfy_kitchen as ck

ap = argparse.ArgumentParser()
ap.add_argument("captures", nargs="+")
ap.add_argument("--out", required=True)
args = ap.parse_args()
out = {}
for path in args.captures:
    d = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = (d[n].cuda() for n in ("q", "k", "v"))   # [1, H, S, D] bf16
    for _ in range(2):
        o = ck.int8_attention(q, k, v)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(5):
        o = ck.int8_attention(q, k, v)
    e1.record(); torch.cuda.synchronize()
    out[path.split("/")[-1]] = round(e0.elapsed_time(e1) / 5, 2)
    print(path.split("/")[-1], out[path.split("/")[-1]], "ms dense int8_attention", flush=True)
    del q, k, v, o; torch.cuda.empty_cache()
json.dump(out, open(args.out, "w"), indent=1)
