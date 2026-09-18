#!/usr/bin/env python3
"""Where in the clip does Sol-Attn's error sit? Per-token error on a capture, by latent frame and in a named region.

    python bench/map_sol_error_on_capture.py <capture.pt> [...] --video-start N --audio-span A,B \\
        --region T0:T1,H0:H1,W0:W1 --out bench/results/<date>_sol_error_map.json

Every capture metric this pack has used is one number per call: the error of
the whole output against exact attention. On 2026-09-17 such numbers and the
owner's eye disagreed about which lever mattered for a visible morph, and one
reason they could is that a whole-call error says nothing about WHERE the error
is. A morph is local: a few latent frames, part of the picture. This asks the
local question of a capture taken from a render where the morph is known to
happen, and where and when it happens was read off the clip.

For each capture and each token ordering (plain raster, `3d`), one call of the
real CUDA kernel at the shipped recipe and the given tau, and then per video
token: the norm of (Sol output - exact fp32 output) over the head prefix,
divided by the MEAN per-token norm of the exact output, so a token's number is
comparable to any other token's and the mean over tokens is a relative error.
Reported per ordering:

  whole          mean over all video tokens
  region         mean over the tokens inside --region (token-grid indices,
                 half-open, T over latent frames, H and W over 2x2 patches)
  region/whole   above 1 means the error concentrates in the region
  by_frame       the mean per latent frame, so a peak in time is visible
  top_frames     the latent frames with the highest mean

What would make this informative: under plain order the region (or its latent
frames) stands out from the rest, and under `3d` it does not. What would make
it uninformative: the region looks like everywhere else under both orderings,
which says the whole-call error of a single attention call is the wrong
instrument for this artifact, however it is sliced. Both are findings.

Limits: a head prefix; the captured blocks and steps; one tau; error of ONE
attention call against exact attention on the same inputs, not the error the
trajectory accumulates over fifty blocks and twelve Sol steps; the routed
densities of the two orderings differ slightly at a fixed tau and are reported.
The capture must have been taken with reordering OFF.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from analyze_sol_error import dense_reference, load_capture, load_cuda_kernel  # noqa: E402
from _live_sol import live_sol  # noqa: E402

ORDERINGS = ("raster", "3d")


def _span(text):
    a, b = text.split(":")
    return int(a), int(b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--heads", type=int, default=8, help="head PREFIX")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--grid", default="102,24,42", help="token T,H,W of the video segment")
    ap.add_argument("--audio-span", required=True)
    ap.add_argument("--video-start", type=int, required=True)
    ap.add_argument("--region", required=True, help="T0:T1,H0:H1,W0:W1 in token-grid indices, half-open")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import orjson
    node = live_sol()
    sys.path.insert(0, str(REPO / "workflows"))
    import h3_config
    recipe = h3_config.SOL_RECOMMENDED_CUDA
    sol = load_cuda_kernel()
    grid = tuple(int(x) for x in args.grid.split(","))
    (t0, t1), (h0, h1), (w0, w1) = (_span(x) for x in args.region.split(","))
    start = args.video_start
    a0, a1 = (int(x) for x in args.audio_span.split(","))
    inside = torch.zeros(grid, dtype=torch.bool)
    inside[t0:t1, h0:h1, w0:w1] = True

    cells = {}
    for cap in args.captures:
        q, k, v = load_capture(cap)
        q, k, v = (x[:, :args.heads] for x in (q, k, v))
        tokens = q.shape[2]
        n_video = grid[0] * grid[1] * grid[2]
        if start + n_video != tokens:
            raise SystemExit(f"{cap}: layout says video is rows {start}..{start + n_video}, capture has {tokens}")
        m = re.search(r"_b(\d+)_s(\d+)", Path(cap).name)
        cell = f"b{m.group(1)}_s{m.group(2)}" if m else Path(cap).stem
        ref = dense_reference(q, k, v)                                  # [1, H, S, D] fp32, raster order
        ref_tok = ref[0, :, start:].permute(1, 0, 2).reshape(n_video, -1)
        scale = float(ref_tok.norm(dim=-1).mean())
        layout = {"sol_h3_video_span": (start, tokens), "sol_h3_audio_span": (a0, a1)}
        sink_kv, sink_q = node._sink_blocks(layout, tokens, recipe["sink_conditioning"])
        nb = (tokens + 63) // 64
        rows = {}
        for name in ORDERINGS:
            if name == "raster":
                index = torch.arange(tokens)
            else:
                perm, _ = node._perm_for(grid, name, "cpu", start)
                index = torch.cat([torch.arange(start), start + perm.cpu()])
            back = torch.argsort(index)
            to = dict(device="cuda", dtype=torch.bfloat16)
            qs, ks, vs = (x[:, :, index].to(**to).permute(0, 2, 1, 3).contiguous() for x in (q, k, v))
            cnt = torch.zeros((1, qs.shape[2], nb), dtype=torch.int32, device="cuda")
            out = sol(qs, ks, vs, tau=args.tau, scale=None, sink_blocks=list(sink_kv), sink_q=list(sink_q),
                      topk_ratio=0.0, tail=bool(recipe["pooled_tail"]), qk_balance=bool(recipe["qk_balance"]),
                      blk_cnt=cnt)
            torch.cuda.synchronize()
            out = out.permute(0, 2, 1, 3).float().cpu()[:, :, back]      # back to raster order
            out_tok = out[0, :, start:].permute(1, 0, 2).reshape(n_video, -1)
            err = ((out_tok - ref_tok).norm(dim=-1) / scale).reshape(grid)
            by_frame = err.mean(dim=(1, 2))
            top = torch.topk(by_frame, 8).indices.tolist()
            rows[name] = {
                "density": round(float(cnt.float().mean()) / nb, 5),
                "whole": round(float(err.mean()), 5),
                "region": round(float(err[inside].mean()), 5),
                "region_over_whole": round(float(err[inside].mean() / err.mean()), 3),
                "region_frames_over_whole": round(float(by_frame[t0:t1].mean() / err.mean()), 3),
                "top_frames": sorted(top),
                "by_frame": [round(float(x), 5) for x in by_frame],
            }
            r = rows[name]
            print(f"{cell} {name:6s} density {r['density']:.3f}  whole {r['whole']:.4f}  region {r['region']:.4f} "
                  f"({r['region_over_whole']:.2f}x)  region's frames {r['region_frames_over_whole']:.2f}x  "
                  f"top frames {r['top_frames']}", flush=True)
            del qs, ks, vs, out, cnt
            torch.cuda.empty_cache()
        cells[cell] = rows
        del ref, q, k, v

    import importlib.metadata
    Path(args.out).write_bytes(orjson.dumps({
        "what": "per-token Sol-Attn error against fp32 dense attention on one call, CUDA kernel, by token ordering: "
                "mean over the video, mean inside a named region, and the mean per latent frame",
        "model": "MiniMax H3, int8 convrot checkpoint, base 16-step t2v capture",
        "conditions": {"gpu": torch.cuda.get_device_name(0), "comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
                       "heads": args.heads, "tau": args.tau, "grid_thw": grid, "video_start": start,
                       "audio_span": [a0, a1], "region_thw": [[t0, t1], [h0, h1], [w0, w1]],
                       "sink_conditioning": recipe["sink_conditioning"], "pooled_tail": recipe["pooled_tail"],
                       "qk_balance": recipe["qk_balance"], "capture_set": Path(args.captures[0]).resolve().parent.name,
                       "captures": [Path(c).name for c in args.captures]},
        "how_to_read": "a token's error is the norm of its output difference over the head prefix divided by the mean "
                       "per-token norm of the exact output; region_over_whole above 1 means error concentrates there",
        "cells": cells}, option=orjson.OPT_INDENT_2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
