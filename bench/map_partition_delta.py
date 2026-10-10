#!/usr/bin/env python3
"""What differs between the release's FL2VA and Ref2VA weights, tensor by tensor.

Reads the two original checkpoints (`<release>/FL2VA/transformer`,
`<release>/Ref2VA/transformer`, bf16, native key names) and writes one JSONL row
per tensor as it goes:

- `rel_delta`: ||ref - fl|| / ||fl||, and `cos` between the two tensors, from the three norms,
  all in float64 (float32 norms drift by a few percent on the 260M-element adaln weights);
- `identical`, `only_in`: byte-equal tensors, and keys one partition lacks;
- `fl_norm_zero`, `ref_norm_zero`: an all-zero tensor in either;
- for 2-D weights with `--lowrank Q`: `top_energy`, the share of the delta's
  squared Frobenius norm in its top Q singular directions (a randomized SVD,
  `torch.svd_lowrank`, seeded, and still approximate), and `noise_top_energy`, the same for a Gaussian matrix
  of the same shape and norm, the calibration for "concentrated" versus
  "spread like noise".

Nothing here says which difference is a defect. It maps where the two
partitions differ, and where a difference is absent, degenerate or unusually
concentrated. CPU only, one tensor at a time.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/map_partition_delta.py \\
        --release <MiniMaxAI_MiniMax-H3 dir> --out data/results/<date>_<name>.jsonl [--lowrank 16]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open


def shard_map(directory: Path) -> dict[str, Path]:
    where = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            for k in f.keys():
                where[k] = shard
    return where


def get(where: dict, key: str) -> torch.Tensor:
    with safe_open(str(where[key]), "pt") as f:
        return f.get_tensor(key)


def top_energy(m: torch.Tensor, q: int) -> float:
    _, s, _ = torch.svd_lowrank(m, q=q, niter=2)
    return float((s ** 2).sum() / m.pow(2).sum())


def order(key: str):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", key)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lowrank", type=int, default=0, help="Q for the top-Q energy of each 2-D delta; 0 skips it")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    fl = shard_map(args.release / "FL2VA" / "transformer")
    ref = shard_map(args.release / "Ref2VA" / "transformer")
    with open(args.out, "w") as out:
        for key in sorted(set(fl) | set(ref), key=order):
            if key not in fl or key not in ref:
                row = {"key": key, "only_in": "FL2VA" if key in fl else "Ref2VA"}
            else:
                a, b = get(fl, key), get(ref, key)
                if a.shape != b.shape or a.dtype != b.dtype:
                    row = {"key": key, "shape_or_dtype_differs": [str(a.dtype), list(a.shape), str(b.dtype), list(b.shape)]}
                else:
                    af, bf = a.double(), b.double()      # float32 norms of 260M-element tensors come out several percent low
                    d = bf - af
                    an, bn, dn = float(af.norm()), float(bf.norm()), float(d.norm())
                    row = {"key": key, "shape": list(a.shape), "dtype": str(a.dtype).removeprefix("torch."),
                           "identical": bool(torch.equal(a, b)), "fl_norm": an, "ref_norm": bn,
                           "rel_delta": dn / an if an else None,
                           "cos": (an * an + bn * bn - dn * dn) / (2 * an * bn) if an and bn else None}
                    if args.lowrank and a.ndim == 2 and min(a.shape) > args.lowrank * 2 and not row["identical"]:
                        torch.manual_seed(0)      # the randomized SVD moves by a few hundredths between unseeded runs
                        d = d.float()
                        noise = torch.randn(a.shape) * (dn / (a.numel() ** 0.5))
                        row["top_energy"] = top_energy(d, args.lowrank)
                        row["noise_top_energy"] = top_energy(noise, args.lowrank)
            out.write(json.dumps(row) + "\n")
            out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
