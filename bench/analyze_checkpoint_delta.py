#!/usr/bin/env python3
"""What is inside the fl2va / ref2va checkpoints, measured, not inferred.

Streams tensors out of the two int8_convrot checkpoints by safetensors
header offsets (no GPU, no safetensors dependency), dequantizes the int8
linears against their per-channel scales, and measures per block: relative
Frobenius delta and cosine for the four quantized linears, the AdaLN
projection, and the norms; plus the global tensors and the per-block AdaLN
delta-energy distribution that the hybrid checkpoints slice.

Findings from the 2026-08-18 run (bench/results/2026-08-18_dit_internals.json,
which this script reproduces):

  - **AdaLN is replaced, not adjusted, in EVERY block**: weight rel-delta
    1.84-1.92 with NEGATIVE cosine (-0.71..-0.80) at every depth, and
    `final_layer.adaln_proj` likewise (rel 1.90). The reference-conditioning
    mechanism lives in the modulation pathway, which is also why the
    extracted ref LoRA carries full-rank adaln diffs and why the HF hybrids
    are adaln swaps.
  - **AdaLN delta energy is FLAT across depth** (~2% per block; blocks 0-14
    hold 8.9%). The hybrids' suffix cutoffs are therefore a linear dial on
    "how much ref2va modulation" -- b15-49 carries 91.1% of the delta
    energy, b20 88.4%, b25 85.5%, b30 81.3% -- not a targeting of blocks
    that matter more. None of them swap `final_layer.adaln_proj` or blocks
    0-14, and none carry the linear deltas below.
  - **The int8 linears differ ~3.1-3.3% relative, uniformly** (cosine
    0.9994-0.9998 at every block); the output heads differ more
    (video_out 5.0%, audio_out 7.9%); condition/patch projections ~2.2-2.4%.
  - **Quantization stress is identical** between the two checkpoints
    (qkv per-channel scale mean/max match to 4 decimals per block), so
    neither partition is the harder int8 target.

The checkpoint directory is resolved from this repo's own location inside
`custom_nodes` (../../models/diffusion_models), overridable with
--models-dir for a checkout that lives elsewhere.

Usage:
    bench/analyze_checkpoint_delta.py out.json [--models-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

_DT = {"F32": np.float32, "F16": np.float16, "I8": np.int8, "U8": np.uint8}

FL_NAME = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
RE_NAME = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
HYBRID_CUTS = (15, 20, 25, 30)

INT8_MODS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
GLOBAL_TENSORS = (
    "condition_proj.weight", "video_patch_proj.weight",
    "audio_patch_proj.weight", "final_layer.adaln_proj.linear.weight",
    "final_layer.video_out.weight", "final_layer.audio_out.weight",
    "adaln_t_table")


def header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def read(path: str, hdr: dict, base: int, name: str) -> np.ndarray:
    info = hdr[name]
    o0, o1 = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + o0)
        buf = f.read(o1 - o0)
    dt = info["dtype"]
    if dt == "BF16":
        arr = (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
               ).view(np.float32)
    else:
        arr = np.frombuffer(buf, dtype=_DT[dt]).astype(np.float32)
    return arr.reshape(info["shape"])


def deq(path: str, hdr: dict, base: int, mod: str) -> np.ndarray:
    return (read(path, hdr, base, mod + ".weight")
            * read(path, hdr, base, mod + ".weight_scale"))


def rel_delta(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt((d * d).sum())) / max(float(np.sqrt((a * a).sum())),
                                               1e-12)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum()) / max(
        float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", help="JSON output path")
    ap.add_argument("--models-dir", default=None,
                    help="ComfyUI diffusion_models dir; default resolves "
                         "../../models/diffusion_models from this repo")
    args = ap.parse_args()

    models = Path(args.models_dir) if args.models_dir else (
        Path(__file__).resolve().parents[3] / "models" / "diffusion_models")
    fl = str(models / FL_NAME)
    re_ = str(models / RE_NAME)

    fl_h, fl_b = header(fl)
    re_h, re_b = header(re_)
    out: dict[str, Any] = {
        "files": {"fl2va": FL_NAME, "ref2va": RE_NAME},
        "per_block": [], "scales": [], "global": {}}

    adaln_e: list[float] = []
    for blk in range(50):
        row: dict[str, Any] = {"block": blk}
        for mod in INT8_MODS:
            name = f"blocks.{blk}.{mod}"
            a = deq(fl, fl_h, fl_b, name)
            b = deq(re_, re_h, re_b, name)
            row[mod] = {"rel_delta": round(rel_delta(b, a), 5),
                        "cos": round(cos(a, b), 6)}
            if mod == "attn.qkv_proj":
                sa = read(fl, fl_h, fl_b, name + ".weight_scale").ravel()
                sb = read(re_, re_h, re_b, name + ".weight_scale").ravel()
                out["scales"].append({
                    "block": blk,
                    "fl_qkv_scale_mean": float(sa.mean()),
                    "fl_qkv_scale_max": float(sa.max()),
                    "re_qkv_scale_mean": float(sb.mean()),
                    "re_qkv_scale_max": float(sb.max())})
            del a, b
        wa = read(fl, fl_h, fl_b, f"blocks.{blk}.adaln_proj.linear.weight")
        wb = read(re_, re_h, re_b, f"blocks.{blk}.adaln_proj.linear.weight")
        ba = read(fl, fl_h, fl_b, f"blocks.{blk}.adaln_proj.linear.bias")
        bb = read(re_, re_h, re_b, f"blocks.{blk}.adaln_proj.linear.bias")
        row["adaln"] = {"w_rel_delta": round(rel_delta(wb, wa), 5),
                        "b_rel_delta": round(rel_delta(bb, ba), 5),
                        "w_cos": round(cos(wa, wb), 6)}
        dd = wb - wa
        adaln_e.append(float((dd * dd).sum()))
        norms: dict[str, float] = {}
        for nm in ("norm1", "norm2", "attn.q_norm", "attn.k_norm"):
            na = read(fl, fl_h, fl_b, f"blocks.{blk}.{nm}.weight")
            nb = read(re_, re_h, re_b, f"blocks.{blk}.{nm}.weight")
            norms[nm] = round(rel_delta(nb, na), 5)
        row["norms_rel_delta"] = norms
        out["per_block"].append(row)
        print(f"block {blk} done", flush=True)

    for nm in GLOBAL_TENSORS:
        a = read(fl, fl_h, fl_b, nm)
        b = read(re_, re_h, re_b, nm)
        out["global"][nm] = {"rel_delta": round(rel_delta(b, a), 5),
                             "cos": round(cos(a, b), 6)}

    tot = sum(adaln_e)
    out["hybrid_adaln_delta_energy_fraction"] = {
        f"b{c}-49": round(sum(adaln_e[c:]) / tot, 4) for c in HYBRID_CUTS}
    out["adaln_delta_energy_by_block"] = [round(e / tot, 5) for e in adaln_e]

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("written", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
