#!/usr/bin/env python3
"""FL2VA against Ref2VA, weight by weight, on a path independent of `bench/map_partition_delta.py`.

`2026-09-29_ref2va_partition_delta.md` says the two partitions differ by a few
percent in every tensor except a structured timestep path (time embedder,
adaln). `map_partition_delta.py` computes its norms in float32 and its top-16
energy by a randomised SVD. This reads the release's shards itself, computes in
float64, and takes the exact top-16 energy from the eigenvalues of the Gram
matrix, for a chosen set of tensors, against a Gaussian-noise control of the
same norm. It also gives the per-block relative delta of qkv, fc1 and adaln
(min and max over the 50 blocks, and where the max is).

CPU only; the adaln matrices are large and the Gram step takes minutes.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/verify_partition_delta_float64.py \\
        --release <MiniMaxAI_MiniMax-H3 dir> --record data/results/<date>_<name>.json
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--release", type=Path, required=True)
ap.add_argument("--record", type=Path, required=True)
args = ap.parse_args()
R = args.release
def idx(d):
    m = {}
    for p in sorted((R / d / "transformer").glob("*.safetensors")):
        with safe_open(p, "pt") as f:
            for k in f.keys(): m[k] = p
    return m
A, B = idx("FL2VA"), idx("Ref2VA")
print("keys equal:", set(A) == set(B), len(A))
def load(m, k):
    with safe_open(m[k], "pt") as f: return f.get_tensor(k)
torch.manual_seed(0)
def top16(d):
    d = d.double()
    g = d.T @ d if d.shape[0] >= d.shape[1] else d @ d.T
    ev = torch.linalg.eigvalsh(g).flip(0)
    return float(ev[:16].sum() / ev.sum())
out = {}
keys = ["time_embedder.proj_in.weight", "time_embedder.proj_out.weight", "blocks.0.adaln_proj.linear.weight", "blocks.25.adaln_proj.linear.weight",
        "blocks.0.attn.qkv_proj.weight", "blocks.25.attn.qkv_proj.weight", "blocks.25.mlp.fc1.weight", "blocks.10.attn.k_norm.weight", "blocks.10.norm1.weight",
        "final_layer.audio_out.weight", "final_layer.video_out.weight", "final_layer.adaln_proj.linear.weight", "video_patch_proj.weight", "condition_proj.weight"]
for k in keys:
    a, b = load(A, k).double(), load(B, k).double()
    d = b - a
    r = {"rel": float(d.norm() / a.norm())}
    if a.ndim == 2 and min(a.shape) > 32:
        r["top16"] = top16(d)
        n = torch.randn(d.shape, dtype=torch.float64) * float(d.norm()) / d.numel() ** .5
        r["noise_top16"] = top16(n)
    out[k] = r
    print(k, {x: round(y, 5) for x, y in r.items()}, flush=True)
# per-block rel of qkv & adaln across all 50 blocks, min/max
for suf in ["attn.qkv_proj.weight", "mlp.fc1.weight", "adaln_proj.linear.weight"]:
    rs = []
    for b in range(50):
        k = f"blocks.{b}.{suf}"
        a, c = load(A, k).float(), load(B, k).float()
        rs.append(float((c - a).double().norm() / a.double().norm()))
    print(suf, "min %.4f max %.4f" % (min(rs), max(rs)), "argmax", rs.index(max(rs)), flush=True)
args.record.write_text(json.dumps(out, indent=1) + "\n")
