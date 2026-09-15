"""Bias vs noise: per kernel, the norm of the MEAN error vector over rows against the norm of the mean output, per head.
Noise averages out over a hundred thousand rows; a consistent direction does not, and it is what would accumulate over steps."""
import sys, json, re
sys.path.insert(0, "bench")
import torch
from analyze_sol_error import load_capture, dense_reference
from grade_dense_kernels_on_captures import KERNELS
cap = sys.argv[1]; m = re.search(r"_b(\d+)_s(\d+)", cap)
q, k, v = load_capture(cap); q, k, v = q[:, :8], k[:, :8], v[:, :8]
ref = dense_reference(q, k, v).float()
rows = {}
for name, fn in KERNELS.items():
    err = fn(q, k, v).float() - ref                               # [1, H, T, D]
    bias = err.mean(dim=2).norm(dim=-1) / ref.mean(dim=2).norm(dim=-1)   # per head: |mean error| / |mean output|
    noise = err.norm(dim=-1).mean(dim=2) / ref.norm(dim=-1).mean(dim=2)   # per head: mean row error / mean row norm
    rows[name] = {"bias_over_mean_output_per_head": [round(float(x), 5) for x in bias[0]],
                  "mean_row_rel_err_per_head": [round(float(x), 5) for x in noise[0]],
                  "bias_mean": float(bias.mean()), "noise_mean": float(noise.mean())}
    print(f"{name:14s} bias {float(bias.mean()):.5f}  noise {float(noise.mean()):.5f}  ratio bias/noise {float(bias.mean()/noise.mean()):.3f}")
print(json.dumps({"block": int(m.group(1)), "step": int(m.group(2)), "rows": rows}))
