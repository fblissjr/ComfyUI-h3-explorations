#!/usr/bin/env python3
"""Add FastH3's VSA gates to a checkpoint, or take them out. CPU only.

Written 2026-09-27 for `docs/state/open_experiments.md` #35: the conditioning swap
(`bench/results/2026-09-27_fasth3_swap.md`) left FastH3's look with its
weights and gates, and cannot say which. The gates are the 50
`blocks.N.attn.to_gate_compress` linears, int8 with a per-row scale and a
`comfy_quant` tag, 150 tensors. Core builds the slot only when block 0's gate
weight is in the file (`comfy/model_detection.py`), the dense forward never
calls it, and core's `BlockSparseAttention` in VSA mode uses it for the gated
coarse branch; with no gates it warns and runs the fine stage alone. So under
FastH3's contract (VSA on) the four cells are:

    fl2va weights, no gates     the renders labelled fl2va_contract (exist)
    fl2va weights + gates       --add: this script
    FastH3 weights, no gates    --drop: this script
    FastH3 weights + gates      FastH3 V2 as released (exists)

`--add` streams `--base` byte for byte and appends every gate tensor from
`--donor`, byte for byte. `--drop` streams `--base` without its gate
tensors. Nothing is requantised. The output is reopened and every gate
tensor, and a spread of the rest, is compared with its source.

    CUDA_VISIBLE_DEVICES= python bench/build_gate_transplant.py --add \\
        --base FL2VA.safetensors --donor FASTH3.safetensors --out OUT.safetensors
    CUDA_VISIBLE_DEVICES= python bench/build_gate_transplant.py --drop \\
        --base FASTH3.safetensors --out OUT.safetensors
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_checkpoint_delta import header  # noqa: E402
from bake_pdd_checkpoint import git_commit, plan_header, raw_bytes  # noqa: E402
from build_adaln_swap import copy_range  # noqa: E402

# 50 gates x (weight, weight_scale, comfy_quant) in FastH3 V2's pruned int8
# file (measured on its header, 2026-09-27).
GATE_TENSORS = 150


def is_gate(name: str) -> bool:
    return ".attn.to_gate_compress." in name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--add", action="store_true", help="append --donor's gates to --base")
    mode.add_argument("--drop", action="store_true", help="write --base without its gates")
    ap.add_argument("--base", required=True)
    ap.add_argument("--donor")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.add and not args.donor:
        ap.error("--add needs --donor")

    hb_, bb = header(args.base)
    hb_.pop("__metadata__", None)
    base_gates = sorted(n for n in hb_ if is_gate(n))
    if args.add:
        hd_, db = header(args.donor)
        hd_.pop("__metadata__", None)
        gates = sorted(n for n in hd_ if is_gate(n))
        assert len(gates) == GATE_TENSORS, f"--donor carries {len(gates)} gate tensors, not {GATE_TENSORS}"
        assert not base_gates, f"--base already carries {len(base_gates)} gate tensors"
        # The gates read the block's input, so the hidden size must agree.
        assert hd_["blocks.0.attn.to_gate_compress.weight"]["shape"][1] == hb_["blocks.0.attn.qkv_proj.weight"]["shape"][1]
        plan = {n: hb_[n] for n in hb_} | {n: hd_[n] for n in gates}
        source = {n: (args.base, bb, hb_[n]) for n in hb_} | {n: (args.donor, db, hd_[n]) for n in gates}
        meta = {"h3_gate_transplant": "add", "h3_gate_base": Path(args.base).name,
                "h3_gate_donor": Path(args.donor).name, "h3_gate_tensors": str(len(gates))}
    else:
        assert len(base_gates) == GATE_TENSORS, f"--base carries {len(base_gates)} gate tensors, not {GATE_TENSORS}"
        gates = base_gates
        plan = {n: hb_[n] for n in hb_ if not is_gate(n)}
        source = {n: (args.base, bb, hb_[n]) for n in plan}
        meta = {"h3_gate_transplant": "drop", "h3_gate_base": Path(args.base).name,
                "h3_gate_tensors": str(len(gates))}
    meta["h3_gate_by"] = f"bench/build_gate_transplant.py @ {git_commit()}"

    hdr, _ = plan_header(plan, meta)
    tmp = args.out.with_suffix(args.out.suffix + ".partial")
    with open(tmp, "wb") as w:
        w.write(struct.pack("<Q", len(hdr)))
        w.write(hdr)
        for name in sorted(plan):
            src, base, info = source[name]
            copy_range(src, base, info, w)
    tmp.rename(args.out)

    h2, b2 = header(str(args.out))
    h2.pop("__metadata__", None)
    assert set(h2) == set(plan), "the output's tensor names are not the plan's"
    rest = [n for n in sorted(plan) if not is_gate(n)]
    check = [n for n in plan if is_gate(n)] + rest[:: max(1, len(rest) // 40)]
    bad = [n for n in check if raw_bytes(str(args.out), b2, h2[n]) != raw_bytes(source[n][0], source[n][1], source[n][2])]
    assert not bad, f"reopened output differs at {bad[:5]}"
    print(json.dumps({"out": args.out.name, "tensors": len(h2),
                      "gates_in_output": sum(is_gate(n) for n in h2), "verified": len(check), **meta}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
