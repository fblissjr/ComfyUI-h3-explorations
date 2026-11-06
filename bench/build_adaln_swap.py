#!/usr/bin/env python3
"""Swap the timestep-conditioning set between two pruned H3 checkpoints. CPU only.

Written 2026-09-26 for the owner's #3 on FastH3 V2
(`bench/results/2026-09-26_fasth3_weights.md`): FastH3 barely moves the
backbone linears and moves every block's modulation by about 5%, so the
transplant that can show where it lives is the conditioning, not the blocks.

In a curve-form checkpoint the conditioning is a SET: `adaln_t_table` (the
time basis) plus every `adaln_proj.linear` weight and bias expressed on it.
Two conversions choose different bases, so the set moves whole or not at all.
This streams `--backbone` byte for byte and takes the set from `--donor`
instead. No tensor is requantised or recomputed; everything outside the set,
including FastH3's VSA gates when FastH3 is the backbone, is the backbone
file's own bytes.

    CUDA_VISIBLE_DEVICES= python bench/build_adaln_swap.py --backbone B.safetensors \\
        --donor D.safetensors --out OUT.safetensors
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_checkpoint_delta import CHUNK, git_commit, header, plan_header, raw_bytes  # noqa: E402


def is_conditioning(name: str) -> bool:
    return name == "adaln_t_table" or ".adaln_proj.linear." in name


def copy_range(src: str, base: int, info: dict, w) -> None:
    o0, o1 = info["data_offsets"]
    with open(src, "rb") as f:
        f.seek(base + o0)
        left = o1 - o0
        while left:
            buf = f.read(min(CHUNK, left))
            w.write(buf)
            left -= len(buf)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    hb_, bb = header(args.backbone)
    hd_, db = header(args.donor)
    hb_.pop("__metadata__", None)
    hd_.pop("__metadata__", None)
    swap = sorted(n for n in hb_ if is_conditioning(n))
    assert swap and "adaln_t_table" in swap, "the backbone file has no curve-form conditioning"
    assert set(swap) == {n for n in hd_ if is_conditioning(n)}, "the two files' conditioning sets differ"
    for n in swap:
        a, b = hb_[n], hd_[n]
        assert (a["dtype"], a["shape"]) == (b["dtype"], b["shape"]), f"{n}: {a} against {b}"

    meta = {"h3_adaln_swap_backbone": Path(args.backbone).name,
            "h3_adaln_swap_donor": Path(args.donor).name,
            "h3_adaln_swap_tensors": str(len(swap)),
            "h3_adaln_swap_by": f"bench/build_adaln_swap.py @ {git_commit()}"}
    hb, _ = plan_header(hb_, meta)
    tmp = args.out.with_suffix(args.out.suffix + ".partial")
    with open(tmp, "wb") as w:
        w.write(struct.pack("<Q", len(hb)))
        w.write(hb)
        for name in sorted(hb_):
            if name in swap:
                copy_range(args.donor, db, hd_[name], w)
            else:
                copy_range(args.backbone, bb, hb_[name], w)
    tmp.rename(args.out)

    # Reopen and compare every swapped tensor with the donor, and a spread of
    # the rest with the backbone, byte for byte.
    h2, b2 = header(str(args.out))
    h2.pop("__metadata__", None)
    assert set(h2) == set(hb_), "the output's tensor names are not the backbone's"
    rest = [n for n in sorted(hb_) if n not in swap]
    sample = rest[:: max(1, len(rest) // 40)]
    bad = [n for n in swap if raw_bytes(str(args.out), b2, h2[n]) != raw_bytes(args.donor, db, hd_[n])]
    bad += [n for n in sample if raw_bytes(str(args.out), b2, h2[n]) != raw_bytes(args.backbone, bb, hb_[n])]
    assert not bad, f"reopened output differs at {bad[:5]}"
    print(json.dumps({"out": args.out.name, "tensors": len(h2), "swapped": len(swap),
                      "verified_swapped": len(swap), "verified_rest_sample": len(sample), **meta}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
