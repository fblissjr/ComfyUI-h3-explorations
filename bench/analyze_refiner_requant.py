#!/usr/bin/env python3
"""Is FastH3's int8 token refiner fl2va's bf16 refiner requantized, or something learned?

fl2va stores its token refiner in bf16 and FastH3 in int8 convrot
(`bench/results/2026-09-29_fasth3_overlay_size.md`), so an overlay carries the
refiner whole. If FastH3's refiner is fl2va's under the same quantizer, that
piece holds no learned change, "no refiner" is not a meaningful arm, and it
could be regenerated from the base instead of shipped. This asks per tensor:

- `codes_equal_fraction`, `max_abs_code_diff`, `scale_max_rel_diff`: fl2va's
  bf16 weight quantized with the format FastH3's file declares (int8 tensorwise
  per-channel, convrot, group 256, deterministic) against FastH3's codes and
  row scales;
- `err_requant`, `err_fasth3`: relative L2 error of each, dequantized, against
  the bf16 weight;
- `err_between`: relative L2 distance between the two dequantized weights;
- the control: FastH3's own weight dequantized to bf16, then requantized the
  same way, against FastH3's codes (`control_codes_equal_fraction`) and weight
  (`err_control`). It is what this recipe does to a weight that is already on
  its grid, through a bf16 round trip, so `err_between` is only a difference
  in the weights if it is well above `err_control`.

CPU only. Reads the refiner tensors of both files.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/analyze_refiner_requant.py \\
        --base <models>/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \\
        --target <models>/diffusion_models/fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors \\
        --record data/results/<date>_<name>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO.parents[1]))

from comfy_kitchen.tensor import TensorWiseINT8Layout as Int8  # noqa: E402

FORMAT = dict(is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256, stochastic_rounding=0)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


def dequant(codes: torch.Tensor, scale: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    params = Int8.Params(scale=scale, orig_dtype=like.dtype, orig_shape=tuple(like.shape),
                         is_weight=True, convrot=True, convrot_groupsize=256)
    return Int8.dequantize(codes, params)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--record", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    with safe_open(str(args.base), "pt") as fb, safe_open(str(args.target), "pt") as ft:
        for key in sorted(k for k in ft.keys() if k.startswith("token_refiner.") and k.endswith(".weight")
                          and ft.get_slice(k).get_dtype() == "I8"):
            w = fb.get_tensor(key)
            codes, scale = ft.get_tensor(key), ft.get_tensor(key + "_scale")
            q, params = Int8.quantize(w, **FORMAT)
            deq_target = dequant(codes, scale, w)
            q_again, again = Int8.quantize(deq_target, **FORMAT)
            diff = (q.int() - codes.int()).abs()
            rows.append({
                "key": key, "base_dtype": str(w.dtype).removeprefix("torch."),
                "codes_equal_fraction": float((diff == 0).float().mean()), "max_abs_code_diff": int(diff.max()),
                "scale_max_rel_diff": float((params.scale / scale - 1).abs().max()),
                "scale_median_rel_diff": float((params.scale / scale - 1).abs().median()),
                "err_requant": rel(dequant(q, params.scale, w), w), "err_fasth3": rel(deq_target, w),
                "err_between": rel(deq_target, dequant(q, params.scale, w)),
                "err_control": rel(dequant(q_again, again.scale, w), deq_target),
                "control_codes_equal_fraction": float((q_again == codes).float().mean()),
            })
            print(json.dumps(rows[-1]))
    args.record.write_text(json.dumps({"base": args.base.name, "target": args.target.name,
                                       "format": FORMAT, "per_tensor": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
