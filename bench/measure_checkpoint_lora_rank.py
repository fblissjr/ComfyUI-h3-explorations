#!/usr/bin/env python3
"""How well would a LoRA capture one checkpoint's difference from another?

Written 2026-09-26 for the owner's question "you can extract a lora from
fasth3 cant you?". FastH3 V2 is a full fine-tune sharing the fl2va base's
quantization (each tensor's `weight_scale` is identical in the two files), so
its difference from the base, per linear, is `dW = W_fasth3 - W_base` after
dequantizing (int8 * weight_scale). A rank-r LoRA can hold at most the top r
singular directions of `dW`. This reports, per layer:

- `rel_delta`: ||dW|| / ||W_base||, how far the weights moved;
- `energy@r`: the share of ||dW||^2 held by the top r singular values, for r
  in `RANKS`. Near 1 at a small r means the change is low-rank and a LoRA
  of that rank reproduces it; spread out means it does not.

Only the four quantized linears per block (qkv_proj, out_proj, mlp.fc1,
mlp.fc2). The AdaLN projection is stored as a curve basis times
coefficients, and two checkpoints' bases can differ in sign, so comparing
coefficients directly is an artifact (`bench/analyze_checkpoint_delta.py`,
the 2026-08-20 retraction). Extracting it needs the modulation output, and it
is out of scope here.

    python bench/measure_checkpoint_lora_rank.py --other OTHER.safetensors [--blocks 0,24,49] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO.parents[1] / "models" / "diffusion_models"
sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402

RANKS = (16, 64, 128, 256)
KINDS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


def dequant(f, key):
    w = f.get_tensor(key + ".weight").float()
    s = f.get_tensor(key + ".weight_scale").float()
    return w * s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=h3_config.MODELS["unet_fl2va"])
    # Defaulted to FastH3 V2 (`MODELS["unet_fasth3_v2"]`) until that key was
    # retired on 2026-11-04; name the checkpoint to compare.
    ap.add_argument("--other", required=True,
                    help="the other checkpoint, a file under models/diffusion_models")
    ap.add_argument("--blocks", default="0,24,49")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    rows = {}
    with safe_open(str(MODELS_DIR / args.base), "pt") as a, safe_open(str(MODELS_DIR / args.other), "pt") as b:
        for blk in args.blocks.split(","):
            for kind in KINDS:
                key = f"blocks.{blk}.{kind}"
                wa, wb = dequant(a, key), dequant(b, key)
                d = wb - wa
                sv = torch.linalg.svdvals(d)
                e = (sv ** 2)
                tot = float(e.sum())
                row = {"shape": list(d.shape), "rel_delta": round(float(d.norm() / wa.norm()), 6)}
                for r in RANKS:
                    row[f"energy@{r}"] = round(float(e[:r].sum()) / tot, 4) if tot > 0 else None
                rows[key] = row
                print(f"{key:<26} {str(tuple(d.shape)):<16} rel {row['rel_delta']:.5f}  " +
                      "  ".join(f"@{r} {row[f'energy@{r}']:.3f}" for r in RANKS), flush=True)
    med = {k: round(statistics.median(r[k] for r in rows.values()), 4)
           for k in ["rel_delta"] + [f"energy@{r}" for r in RANKS]}
    print("median:", med)
    if args.json:
        args.json.write_text(json.dumps({"measured_by": "bench/measure_checkpoint_lora_rank.py",
                                         "base": args.base, "other": args.other,
                                         "layers": rows, "median": med}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
