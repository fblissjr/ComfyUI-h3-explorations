#!/usr/bin/env python3
"""Is block 49's oddity in the original MiniMax weights, before any conversion?

Two facts about the Comfy conversions are recorded (`docs/h3_block49_quant_error.md`,
`bench/analyze_checkpoint_delta.py`): four K channels carry most of block 49's
`k_norm` energy, and block 49's text-row modulation chunks are exactly zero. Every
file that was scanned was made from the release by us or Comfy-Org. This reads
the release itself, its four transformer directories:

- the top-4 share of the K norm's (`attn.norm_k.weight` in the diffusers directories,
  `attn.k_norm.weight` in the original `FL2VA/` and `Ref2VA/`) energy (squared) per block, the scan of
  `data/results/2026-09-14_block49_checkpoint_scan_and_targets.txt`;
- per block, which of the 18 chunks of `adaln_proj.linear` (3 modalities x 6
  params, `analyze_checkpoint_delta.py`) are exactly zero in weight and bias.

CPU only. Block 49's `adaln_proj` weights and a few contrast blocks' are read; the
K-norm scan reads all 50 small vectors.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/scan_original_block49.py \\
        --release <MiniMaxAI_MiniMax-H3 dir> --record data/results/<date>_<name>.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

CHUNK_W = 5376
N_CHUNKS = 18
CONTRAST_BLOCKS = (0, 25, 45, 48, 49)


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--record", type=Path, required=True)
    args = ap.parse_args()

    dirs = {"transformer": "transformer", "transformer_ref": "transformer_ref",
            "FL2VA": "FL2VA/transformer", "Ref2VA": "Ref2VA/transformer"}
    rec = {}
    for name, sub in dirs.items():
        where = shard_map(args.release / sub)
        prefix, knorm = (("transformer_blocks", "norm_k") if any(k.startswith("transformer_blocks.") for k in where)
                         else ("blocks", "k_norm"))
        blocks = sorted({int(m.group(1)) for k in where if (m := re.match(rf"{prefix}\.(\d+)\.", k))})
        loud = {}
        for b in blocks:
            e = get(where, f"{prefix}.{b}.attn.{knorm}.weight").float() ** 2
            top = e.topk(4)
            loud[b] = {"top4_share": float(top.values.sum() / e.sum()), "channels": top.indices.tolist()}
        zero = {}
        for b in (x for x in CONTRAST_BLOCKS if x in blocks):
            w = get(where, f"{prefix}.{b}.adaln_proj.linear.weight")
            bias = get(where, f"{prefix}.{b}.adaln_proj.linear.bias")
            zero[b] = [c for c in range(N_CHUNKS)
                       if not w[c * CHUNK_W:(c + 1) * CHUNK_W].any() and not bias[c * CHUNK_W:(c + 1) * CHUNK_W].any()]
        rec[name] = {"blocks": len(blocks), "k_norm_top4": loud, "zero_adaln_chunks": zero}
        top = sorted(loud, key=lambda b: -loud[b]["top4_share"])[:4]
        print(name, len(blocks), "loudest:", [(b, round(loud[b]["top4_share"], 3), loud[b]["channels"]) for b in top],
              "zero chunks:", zero)
    args.record.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
