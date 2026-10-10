#!/usr/bin/env python3
"""K's INT8 stress by sequence segment, on the captured t2v and ref2va cells.

`bench/results/2026-09-27_sol_redesign_test2.md` finds ref2va's block 49 the
hardest cell for every int8 kernel, and the weights do not differ between the
checkpoints there (`2026-09-29_original_block49_scan.json`). What differs is the
capture: ref2va's carries thousands more text rows and two reference images
beside the same number of video rows. This asks, per segment of a captured
block's K, how badly one shared scale per row would quantize it:

- `err_raw`, `err_smooth`: relative L2 error of K quantized to int8 with one
  absmax scale per (head, row) of 128 channels, the way unrotated sage does,
  without and with sage's smoothing (K minus its per-head channel mean over the
  sequence);
- `crest`: mean row absmax over row rms, how spiky a row is;
- `top4_share`: the four largest channels' share of K's energy in the segment.

It is K's quantization stress, a proxy: the attention output's error also
depends on which keys are attended and how peaky the softmax is. Rows are
subsampled evenly per segment. CPU only; reads the K of each capture file.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/analyze_k_by_segment.py \\
        --capture <t2v capture dir> --capture <ref2va capture dir> \\
        --blocks 24,49 --step 2 --record data/results/<date>_<name>.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch

ROWS_PER_SEGMENT = 24000


def quant_error(k: torch.Tensor) -> tuple[float, float]:
    """(mean rel L2 error, mean absmax / rms) of per-row absmax int8 over [heads, rows, 128]."""
    scale = k.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127.0
    deq = (k / scale).round().clamp(-127, 127) * scale
    err = (deq - k).norm(dim=-1) / k.norm(dim=-1).clamp_min(1e-8)
    crest = k.abs().amax(-1) / (k.pow(2).mean(-1).sqrt().clamp_min(1e-8))
    return float(err.mean()), float(crest.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", action="append", type=Path, required=True)
    ap.add_argument("--blocks", default="24,49")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--record", type=Path, required=True)
    args = ap.parse_args()

    rec = {}
    for cap in args.capture:
        manifest = json.loads((cap / "manifest.json").read_text())
        segs_of = {t["filename"]: t["segments"] for t in manifest["captured_tensors"]}
        for block in (int(b) for b in args.blocks.split(",")):
            path = sorted(glob.glob(str(cap / f"qkv_*_b{block}_s{args.step}_*.pt")))
            assert len(path) == 1, (cap, block, path)
            d = torch.load(path[0], map_location="cpu", weights_only=True, mmap=True)
            k = d["k"][0]                                    # [heads, seq, 128]
            mean = k.float().mean(dim=1, keepdim=True)       # per head, channel
            out = {}
            for start, stop, kind in segs_of[Path(path[0]).name]:
                idx = torch.linspace(start, stop - 1, min(ROWS_PER_SEGMENT, stop - start)).long()
                rows = k[:, idx].float()
                err_raw, crest = quant_error(rows)
                err_smooth, _ = quant_error(rows - mean)
                energy = rows.pow(2).sum(dim=(0, 1))
                name = kind if kind not in out else f"{kind}_{sum(x.startswith(kind) for x in out) + 1}"
                out[name] = {"rows": stop - start, "err_raw": err_raw, "err_smooth": err_smooth, "crest": crest,
                             "top4_share": float(energy.topk(4).values.sum() / energy.sum()),
                             "top4_channels": energy.topk(4).indices.tolist()}
            rec[f"{cap.name}:b{block}:s{args.step}"] = out
            print(f"{cap.name} b{block}")
            for name, v in out.items():
                print(f"  {name:8s} rows={v['rows']:6d} err_raw={v['err_raw']:.4f} err_smooth={v['err_smooth']:.4f} "
                      f"crest={v['crest']:.2f} top4={v['top4_share']:.3f} {v['top4_channels']}")
    args.record.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
