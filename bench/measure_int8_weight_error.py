#!/usr/bin/env python3
"""How much of each linear's weight the shipped int8 convrot file loses, block by block.

There is no sensitivity measurement of the int8 linears here
(`docs/h3_quant_policy.md`, "policy table (linears)"). This is its cheapest
half: for every quantized linear of the shipped int8 file, the relative L2
error of its dequantized weight against the bf16 file it was made from
(`err_shipped`). It is a WEIGHT error: it says which layers the int8 grid
hurts most, not what an activation-weighted output error or a render shows.

The recipe check: for a few blocks (`--recipe-blocks`) the bf16 weight is also
quantized here with the format the file declares (int8 tensorwise per-channel,
convrot, group 256, deterministic) and its codes compared with the shipped
codes (`recipe_codes_equal_fraction`). Near 1 says the file was made this way;
far from 1 and `err_shipped` still stands, the comparison to this recipe does
not.

CPU only. Reads both files, one tensor at a time, and writes a JSONL row per
tensor as it goes, so a stopped run keeps what it measured.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/measure_int8_weight_error.py \\
        --bf16 <fl2va bf16 pruned> --int8 <fl2va int8 pruned> --out data/results/<date>_<name>.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bf16", type=Path, required=True)
    ap.add_argument("--int8", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recipe-blocks", default="0,25,49")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    recipe = {int(b) for b in args.recipe_blocks.split(",")}

    with safe_open(str(args.bf16), "pt") as fb, safe_open(str(args.int8), "pt") as fi, open(args.out, "w") as out:
        keys = [k for k in fi.keys() if k.endswith(".weight") and fi.get_slice(k).get_dtype() == "I8"
                and k in set(fb.keys())]
        for key in sorted(keys, key=lambda k: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", k)]):
            w, codes, scale = fb.get_tensor(key), fi.get_tensor(key), fi.get_tensor(key + "_scale")
            params = Int8.Params(scale=scale, orig_dtype=w.dtype, orig_shape=tuple(w.shape),
                                 is_weight=True, convrot=True, convrot_groupsize=256)
            row = {"key": key, "err_shipped": rel(Int8.dequantize(codes, params), w)}
            block = re.match(r"blocks\.(\d+)\.", key)
            if block and int(block.group(1)) in recipe:
                q, _ = Int8.quantize(w, **FORMAT)
                row["recipe_codes_equal_fraction"] = float((q == codes).float().mean())
            out.write(json.dumps(row) + "\n")
            out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
