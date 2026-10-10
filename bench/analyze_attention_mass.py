#!/usr/bin/env python3
"""Where a captured block's video queries put their attention, by key segment.

`bench/analyze_k_by_segment.py` found K's INT8 stress at block 49 the same in
the t2v and ref2va cells, so it does not explain ref2va's harder cell. This asks
about the softmax instead: for sampled video query rows, exact float32 attention
over every key of the captured sequence, then per head

- `eff_keys`: exp(entropy) of the attention row, how many keys share the mass
  (small means peaky, where one rounded K score can move the output);
- `mass`: the row's probability mass on each key segment (text, reference images,
  audio, video), as the mean over rows.

Reported as the median over heads of each head's mean, and the share of heads
whose mean `eff_keys` is under 20. Queries are sampled evenly from the video
segment. The captured q and k are the kernel's inputs (after RoPE). CPU only.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/analyze_attention_mass.py \\
        --capture <dir> [--capture <dir>] --blocks 24,49 --step 2 --queries 384 \\
        --record data/results/<date>_<name>.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", action="append", type=Path, required=True)
    ap.add_argument("--blocks", default="24,49")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--queries", type=int, default=384)
    ap.add_argument("--record", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(16)

    rec = {}
    for cap in args.capture:
        manifest = json.loads((cap / "manifest.json").read_text())
        segs_of = {t["filename"]: t["segments"] for t in manifest["captured_tensors"]}
        for block in (int(b) for b in args.blocks.split(",")):
            path = sorted(glob.glob(str(cap / f"qkv_*_b{block}_s{args.step}_*.pt")))
            assert len(path) == 1, (cap, block, path)
            d = torch.load(path[0], map_location="cpu", weights_only=True, mmap=True)
            q, k = d["q"][0], d["k"][0]                      # [heads, seq, 128]
            segs = segs_of[Path(path[0]).name]
            video = next((a, b) for a, b, kind in segs if kind == "video")
            rows = torch.linspace(video[0], video[1] - 1, args.queries).long()
            kf = k.float()
            names = [kind if [x[2] for x in segs].count(kind) == 1 else f"{kind}_{i}" for i, (_, _, kind) in enumerate(segs)]
            eff, mass = [], {n: [] for n in names}
            for h in range(q.shape[0]):
                p = torch.softmax((q[h, rows].float() @ kf[h].T) / (q.shape[-1] ** 0.5), dim=-1)   # [queries, seq]
                ent = -(p * p.clamp_min(1e-30).log()).sum(-1)
                eff.append(float(ent.exp().mean()))
                for name, (a, b, _) in zip(names, segs):
                    mass[name].append(float(p[:, a:b].sum(-1).mean()))
            out = {"eff_keys_median_head": statistics.median(eff), "eff_keys_min_head": min(eff),
                   "heads_eff_under_20": sum(e < 20 for e in eff) / len(eff),
                   "mass_median_head": {kind: statistics.median(v) for kind, v in mass.items()},
                   "mass_mean_head": {kind: statistics.fmean(v) for kind, v in mass.items()}}
            rec[f"{cap.name}:b{block}:s{args.step}"] = out
            print(cap.name, f"b{block}", json.dumps({k2: (round(v, 4) if not isinstance(v, dict) else {a: round(b, 4) for a, b in v.items()}) for k2, v in out.items()}))
    args.record.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
