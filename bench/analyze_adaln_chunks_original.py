#!/usr/bin/env python3
"""Where FL2VA and Ref2VA differ in the per-block modulation, by modality, in the release's own weights.

`bench/analyze_checkpoint_delta.py` compared the modulation of the pruned int8
files, and found its largest per-chunk differences on the audio chunks, with
nothing interpreting it (`data/results/2026-08-20_dit_internals.json`). The
pruned files factor the modulation through a shared basis, whose sign flips
withdrew one reading (`docs/method/evidence.md`). The release's own `adaln_proj.linear`
has no factoring, so its weight and bias compare directly.

Each block's `adaln_proj.linear` output is 18 chunks of 5376: 3 modalities
(video 0, text 1, audio 2) x 6 params (shift_msa, scale_msa, gate_msa,
shift_mlp, scale_mlp, gate_mlp). Per block and chunk this writes the relative
delta of weight rows and of bias, ||ref - fl|| / ||fl|| (`weight_rel`,
`bias_rel`), and the chunk's norm in FL2VA (`weight_norm`, `bias_norm`).
Modality summaries are in the printed table. CPU only.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/analyze_adaln_chunks_original.py \\
        --release <MiniMaxAI_MiniMax-H3 dir> --record data/results/<date>_<name>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

CHUNK_W, N_CHUNKS = 5376, 18
PARAMS = ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
MODALITY = ("video", "text", "audio")


def shard_map(directory: Path) -> dict[str, Path]:
    where = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            for k in f.keys():
                where[k] = shard
    return where


def get(where: dict, key: str) -> torch.Tensor:
    with safe_open(str(where[key]), "pt") as f:
        return f.get_tensor(key).float()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--record", type=Path, required=True)
    args = ap.parse_args()
    fl = shard_map(args.release / "FL2VA" / "transformer")
    ref = shard_map(args.release / "Ref2VA" / "transformer")

    rec = {}
    for block in range(50):
        kw, kb = f"blocks.{block}.adaln_proj.linear.weight", f"blocks.{block}.adaln_proj.linear.bias"
        wa, wb, ba, bb = get(fl, kw), get(ref, kw), get(fl, kb), get(ref, kb)
        rows = []
        for c in range(N_CHUNKS):
            s = slice(c * CHUNK_W, (c + 1) * CHUNK_W)
            wn, bn = float(wa[s].norm()), float(ba[s].norm())
            rows.append({"chunk": c, "modality": MODALITY[c // 6], "param": PARAMS[c % 6],
                         "weight_norm": wn, "bias_norm": bn,
                         "weight_rel": float((wb[s] - wa[s]).norm() / wn) if wn else None,
                         "bias_rel": float((bb[s] - ba[s]).norm() / bn) if bn else None})
        rec[block] = rows
        by_mod = {m: [r["weight_rel"] for r in rows if r["modality"] == m and r["weight_rel"] is not None] for m in MODALITY}
        print(block, {m: round(sum(v) / len(v), 3) if v else None for m, v in by_mod.items()}, flush=True)
    args.record.write_text(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
