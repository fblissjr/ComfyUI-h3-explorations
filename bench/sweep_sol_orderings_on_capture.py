#!/usr/bin/env python3
"""Does any token ordering beat raster for Sol-Attn? Error against routed density, per ordering, on captures.

    python bench/sweep_sol_orderings_on_capture.py <capture.pt> [...] --out bench/results/<date>_sol_orderings.json

`docs/morton.md` concluded in August that a controlled ordering A/B was
unbuildable: a reorder moves the routing threshold, the sign of that move
changes with block and sigma, and no knob holds routed density fixed across an
ordering change. What was missing then and exists now: full-length captures,
the real kernel's routed-block count from the call itself (`blk_cnt`), and the
measurement that a Sol call's time is proportional to routed density
(`bench/results/2026-09-17_sol_stage_profile.md`).

So the comparison needs no matched tau. For each ordering, sweep tau and
record (routed density, error against exact attention). Density is cost. An
ordering is better than raster only if its curve lies BELOW raster's: lower
error at equal density, which is the same as equal error for less time. A
curve that merely sits elsewhere along the same line has bought nothing that
tau could not.

What is permuted and how: exactly what the node does. Only the video rows
move, by `sol_attn_h3._perm_for`, which includes the roll that realigns the
curve to the 64-row block grid. Captures are post-RoPE, so permuting the rows
of q, k and v is the whole effect. The reference is fp32 dense attention on
the raster inputs, permuted the same way; attention is permutation-equivariant,
so that is the exact answer for every ordering up to fp32 summation order.
The call is the node's: its own sink ranges for the capture's layout, the
shipped recipe otherwise.

Limits: captured blocks and steps only; a head prefix, not all heads, because
the fp32 reference is the slow part; error against exact attention is not a
verdict on a clip, and this pack has a standing example of the two disagreeing.
The capture must have been taken with reordering OFF; its payload cannot say,
so that is on the caller.
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

ORDERINGS = ("raster", "3d", "2d_frame", "hilbert")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--heads", type=int, default=8, help="head PREFIX (the fp32 reference is the slow part)")
    ap.add_argument("--taus", default="0.6,0.8,1.0,1.2,1.5,2.0")
    ap.add_argument("--grid", default="102,24,42", help="latent T,H,W of the video segment")
    ap.add_argument("--audio-span", default="395,1545")
    ap.add_argument("--video-start", type=int, default=1545)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import orjson
    node = live_sol()
    sys.path.insert(0, str(REPO / "workflows"))
    import h3_config
    recipe = h3_config.SOL_RECOMMENDED_CUDA
    sol = load_cuda_kernel()
    grid = tuple(int(x) for x in args.grid.split(","))
    taus = [float(x) for x in args.taus.split(",")]
    start = args.video_start
    a0, a1 = (int(x) for x in args.audio_span.split(","))

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
        ref = dense_reference(q, k, v)                       # [B, H, S, D] fp32, raster order
        ref_norm = float(ref.norm())
        layout = {"sol_h3_video_span": (start, tokens), "sol_h3_audio_span": (a0, a1)}
        sink_kv, sink_q = node._sink_blocks(layout, tokens, recipe["sink_conditioning"])
        nb = (tokens + 63) // 64
        rows = {}
        for name in ORDERINGS:
            if name == "raster":
                index = torch.arange(tokens)
            else:
                perm, _inv = node._perm_for(grid, name, "cpu", start)
                index = torch.cat([torch.arange(start), start + perm.cpu()])
            assert torch.equal(torch.sort(index).values, torch.arange(tokens)), f"{name}: not a bijection"
            to = dict(device="cuda", dtype=torch.bfloat16)
            qs, ks, vs = (x[:, :, index].to(**to).permute(0, 2, 1, 3).contiguous() for x in (q, k, v))
            ref_p = ref[:, :, index].to("cuda")
            curve = []
            for tau in taus:
                cnt = torch.zeros((1, qs.shape[2], nb), dtype=torch.int32, device="cuda")
                kw = dict(tau=tau, scale=None, sink_blocks=list(sink_kv), sink_q=list(sink_q),
                          topk_ratio=0.0, tail=bool(recipe["pooled_tail"]), qk_balance=bool(recipe["qk_balance"]))
                out = sol(qs, ks, vs, blk_cnt=cnt, **kw)
                torch.cuda.synchronize()
                err = float((out.permute(0, 2, 1, 3).float() - ref_p).norm()) / ref_norm
                density = float(cnt.float().mean()) / nb
                times = []
                for _ in range(args.iters):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record(); sol(qs, ks, vs, **kw); e.record(); torch.cuda.synchronize()
                    times.append(s.elapsed_time(e))
                curve.append({"tau": tau, "density": round(density, 5), "rel_l2": round(err, 5),
                              "call_ms": round(sorted(times)[len(times) // 2], 2)})
                del out, cnt
            rows[name] = curve
            print(f"{cell} {name:9s} " + "  ".join(f"tau {c['tau']}: d {c['density']:.3f} e {c['rel_l2']:.4f}" for c in curve))
            del qs, ks, vs, ref_p
            torch.cuda.empty_cache()
        cells[cell] = rows
        del ref, q, k, v

    import importlib.metadata
    Path(args.out).write_bytes(orjson.dumps({
        "what": "Sol-Attn error against fp32 dense attention versus routed density, per token ordering, tau swept; "
                "only the video rows are permuted, by the node's own _perm_for",
        "model": "MiniMax H3, int8 convrot checkpoint, base 16-step t2v captures",
        "conditions": {"gpu": torch.cuda.get_device_name(0), "comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
                       "heads": args.heads, "grid_thw": grid, "video_start": start, "audio_span": [a0, a1],
                       "sink_conditioning": recipe["sink_conditioning"], "pooled_tail": recipe["pooled_tail"],
                       "qk_balance": recipe["qk_balance"], "captures": [Path(c).name for c in args.captures]},
        "how_to_read": "an ordering beats raster only where its (density, rel_l2) curve lies below raster's; "
                       "density is cost (call time is linear in it)",
        "cells": cells}, option=orjson.OPT_INDENT_2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
