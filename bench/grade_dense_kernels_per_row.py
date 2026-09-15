#!/usr/bin/env python3
"""Per-row error of the dense-step kernels on a capture, split by segment and by attention breadth.

The aggregate relative-L2 grade could not see what the market clips showed
(a kernel with the lowest aggregate error on every cell losing a hand action
every INT8 arm shares). This looks where an aggregate cannot: per query row,
error against fp32 attention for each kernel, grouped by the packed
sequence's segments (text | audio | video rows) and by how broad the row's
fp32 attention is (effective keys = 1 / sum p^2). It also reports, per
breadth bin, the fraction of a row's attention mass sitting below 1/255 of
the row's maximum weight: the mass an unsigned-INT8 P with a per-row max
scale cannot represent. A kernel whose error tracks that fraction is losing
the attention tail, a bias against weak long-range attention that moves the
aggregate little.

    python bench/grade_dense_kernels_per_row.py <capture.pt> [--heads 8] [--json out]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sol_error import load_capture  # noqa: E402
from grade_dense_kernels_on_captures import KERNELS  # noqa: E402

TEXT, AUDIO = 395, 1150   # this capture set's packed layout: [text | audio | video]


def fp32_rows(q, k, v, chunk=256):
    """fp32 attention output plus, per row: effective keys and the P mass below pmax/255."""
    b, h, t, d = q.shape
    out = torch.empty(b, h, t, d, dtype=torch.float32)
    eff = torch.empty(h, t); tail = torch.empty(h, t)
    scale = d ** -0.5
    for hh in range(h):
        kf = k[0, hh].cuda().float(); vf = v[0, hh].cuda().float()
        for s0 in range(0, t, chunk):
            qf = q[0, hh, s0:s0 + chunk].cuda().float()
            p = torch.softmax(qf @ kf.T * scale, dim=-1)
            out[0, hh, s0:s0 + chunk] = (p @ vf).cpu()
            eff[hh, s0:s0 + chunk] = (1.0 / p.pow(2).sum(-1)).cpu()
            pmax = p.max(-1, keepdim=True).values
            tail[hh, s0:s0 + chunk] = (p * (p < pmax / 255.0)).sum(-1).cpu()
    return out, eff, tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()
    m = re.search(r"_b(\d+)_s(\d+)", Path(args.capture).name)
    block, step = int(m.group(1)), int(m.group(2))
    q, k, v = load_capture(args.capture)
    if 0 < args.heads < q.shape[1]:
        q, k, v = q[:, :args.heads], k[:, :args.heads], v[:, :args.heads]
    ref, eff, tail = fp32_rows(q, k, v)
    rn = ref.norm(dim=-1).clamp_min(1e-6)                              # [B, H, T]
    seg = torch.zeros(q.shape[2], dtype=torch.long); seg[TEXT:TEXT + AUDIO] = 1; seg[TEXT + AUDIO:] = 2
    names = ("text", "audio", "video")
    # breadth bins by quartile of effective keys over all rows and heads
    qs = torch.quantile(eff.flatten(), torch.tensor([0.25, 0.5, 0.75]))
    bins = torch.bucketize(eff, qs)                                    # [H, T] in 0..3
    rows = {"block": block, "step": step, "heads": int(q.shape[1]),
            "breadth_bins_effective_keys_edges": [float(x) for x in qs],
            "tail_mass_below_pmax_over_255_by_bin": [float(tail[bins == bnum].mean()) for bnum in range(4)],
            "kernels": {}}
    print(f"per-row grade: block {block} step {step} heads {q.shape[1]}; breadth quartile edges (effective keys) {[round(float(x)) for x in qs]}")
    print(f"tail mass below pmax/255 by breadth bin: {[round(float(tail[bins == b].mean()), 4) for b in range(4)]}")
    print(f"{'kernel':14s}{'text':>9s}{'audio':>9s}{'video':>9s} | {'narrow':>8s}{'q2':>8s}{'q3':>8s}{'broad':>8s} | {'p99 row':>8s}")
    for name, fn in KERNELS.items():
        out = fn(q, k, v)
        err = (out - ref).norm(dim=-1) / rn                            # [B, H, T]
        e = err[0]                                                      # [H, T]
        by_seg = [float(e[:, seg == i].mean()) for i in range(3)]
        by_bin = [float(e[bins == bnum].mean()) for bnum in range(4)]
        p99 = float(torch.quantile(e.flatten(), 0.99))
        rows["kernels"][name] = {"by_segment": dict(zip(names, by_seg)), "by_breadth_bin": by_bin, "p99_row_rel_err": p99,
                                 "mean_row_rel_err": float(e.mean())}
        print(f"{name:14s}" + "".join(f"{x:>9.4f}" for x in by_seg) + " | " + "".join(f"{x:>8.4f}" for x in by_bin) + f" | {p99:>8.4f}")
        del out, err, e
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
