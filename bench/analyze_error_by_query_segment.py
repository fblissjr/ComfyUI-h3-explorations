#!/usr/bin/env python3
"""Whose attention error a whole-cell INT8 grade counts: the query rows' segments.

Two records above this one leave ref2va's block 49 unexplained: K's INT8 stress
(`bench/analyze_k_by_segment.py`) and softmax peakiness
(`bench/analyze_attention_mass.py`) are alike in the t2v and ref2va cells. A
grade of a captured cell over ALL rows also counts the text rows as queries, and
at block 49 the text rows' modulation is exactly zero, so what attention writes
to them is never read (`docs/state/decisions.md`, 2026-09-29). ref2va's cell has
many times more text rows. This measures, per query segment, the error of an
attention whose K is quantized to int8 with one absmax scale per (head, row) (the
unrotated stress) against exact float32 attention, on the captured q, k, v:

- `rel_err`: mean over sampled queries of the output error's norm over the exact
  output's norm, per segment;
- `share_of_sq_error`: the segment's share of the cell's squared output error,
  its rows counted in full (`rows` from the capture's segments), sampled queries
  standing for the rest.

K only: V is not quantized here, and the kernels under test also round Q, P
and V, so this is the K part of the error and not a reproduction of the test
2 numbers. Queries are sampled evenly per segment. CPU only.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/analyze_error_by_query_segment.py \\
        --capture <dir> [--capture <dir>] --blocks 24,49 --step 2 --queries 96 \\
        --record data/results/<date>_<name>.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", action="append", type=Path, required=True)
    ap.add_argument("--blocks", default="24,49")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--queries", type=int, default=96)
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
            q, k, v = d["q"][0], d["k"][0].float(), d["v"][0].float()
            scale = k.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127.0
            kq = (k / scale).round().clamp(-127, 127) * scale
            segs = segs_of[Path(path[0]).name]
            names = [kind if [x[2] for x in segs].count(kind) == 1 else f"{kind}_{i}" for i, (_, _, kind) in enumerate(segs)]
            out = {}
            for name, (a, b, _) in zip(names, segs):
                rows = torch.linspace(a, b - 1, min(args.queries, b - a)).long()
                rel, sq_err, sq_out = [], 0.0, 0.0
                for h in range(q.shape[0]):
                    qh = q[h, rows].float() * (q.shape[-1] ** -0.5)
                    exact = torch.softmax(qh @ k[h].T, dim=-1) @ v[h]
                    quant = torch.softmax(qh @ kq[h].T, dim=-1) @ v[h]
                    diff = quant - exact
                    rel.append(float((diff.norm(dim=-1) / exact.norm(dim=-1).clamp_min(1e-8)).mean()))
                    sq_err += float(diff.pow(2).sum(-1).mean())
                    sq_out += float(exact.pow(2).sum(-1).mean())
                out[name] = {"rows": b - a, "rel_err": sum(rel) / len(rel), "sq_err_per_row": sq_err, "sq_out_per_row": sq_out}
            total = sum(v_["rows"] * v_["sq_err_per_row"] for v_ in out.values())
            total_out = sum(v_["rows"] * v_["sq_out_per_row"] for v_ in out.values())
            for v_ in out.values():
                v_["share_of_sq_error"] = v_["rows"] * v_["sq_err_per_row"] / total
            out["cell_rel_err_all_rows"] = (total / total_out) ** 0.5
            rec[f"{cap.name}:b{block}:s{args.step}"] = out
            print(cap.name, f"b{block}", {n: (round(x["rel_err"], 4), round(x["share_of_sq_error"], 3)) if isinstance(x, dict) else round(x, 4) for n, x in out.items()}, flush=True)
    args.record.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
