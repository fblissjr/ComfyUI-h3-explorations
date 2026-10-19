#!/usr/bin/env python3
"""Which base each timestep tensor of an fl2va/ref2va hybrid came from. CPU only.

In a pruned curve-form checkpoint the timestep path is a SET: `adaln_t_table` (a 1025x8
curve basis) and every `adaln_proj.linear` weight and bias expressed on it. There are no
`time_embedder` tensors in these files; the release's embedder is folded into the table.
This byte-compares that set in a hybrid against fl2va's and ref2va's pruned files, and a
spread of every other tensor, and prints the tally. Written 2026-09-29 for
`docs/state/open_experiments.md` #50.

    CUDA_VISIBLE_DEVICES= python bench/compare_hybrid_timestep_set.py HYBRID.safetensors \\
        --fl2va FL2VA.safetensors --ref2va REF2VA.safetensors --out RECORD.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_checkpoint_delta import header  # noqa: E402
from bake_pdd_checkpoint import raw_bytes  # noqa: E402


def in_set(name: str) -> bool:
    return name == "adaln_t_table" or ".adaln_proj.linear." in name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hybrid")
    ap.add_argument("--fl2va", required=True)
    ap.add_argument("--ref2va", required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    files = {}
    for label, path in (("hybrid", args.hybrid), ("fl2va", args.fl2va), ("ref2va", args.ref2va)):
        h, base = header(path)
        h.pop("__metadata__", None)
        files[label] = (h, base, path)
    names = sorted(files["fl2va"][0])
    assert set(files["ref2va"][0]) == set(names) == set(files["hybrid"][0]), "the three files differ in tensor names"
    timestep = [n for n in names if in_set(n)]
    rest = [n for n in names if not in_set(n)]
    sample = rest[:: max(1, len(rest) // 60)]

    def data(label, name):
        h, base, path = files[label]
        return raw_bytes(path, base, h[name])

    def tally(group):
        out = {"from_fl2va": 0, "from_ref2va": 0, "from_both_identical": 0, "from_neither": 0}
        for n in group:
            hy, fl, rf = data("hybrid", n), data("fl2va", n), data("ref2va", n)
            if fl == rf and hy == fl:
                out["from_both_identical"] += 1
            elif hy == rf:
                out["from_ref2va"] += 1
            elif hy == fl:
                out["from_fl2va"] += 1
            else:
                out["from_neither"] += 1
        return out

    rec = {"hybrid": Path(args.hybrid).name, "timestep_set": {"tensors": len(timestep), **tally(timestep)},
           "adaln_t_table": tally(["adaln_t_table"]),
           "other_sample": {"tensors": len(sample), **tally(sample)}}
    print(json.dumps(rec, indent=1))
    if args.out:
        args.out.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
