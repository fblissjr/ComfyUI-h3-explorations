#!/usr/bin/env python3
"""Where a `comfy_kitchen.sol_attn` call spends its device time, by stage, on captured q/k/v.

    python bench/profile_sol_stages.py <capture.pt> [<capture.pt> ...] --out bench/results/<date>_sol_stages.json

The kernel is four stages over one workspace (`sage_attention/sol_attn.cu`):
`preprocess` (quantize q/k/v, pool per block, the routing threshold),
`vtranspose` (INT8 V^T for the exact stage), `route` (choose the routed blocks,
the pooled tail) and `exact` (walk each routed list). Token routing adds a
fifth. Until 2026-09-17 nothing had timed them apart, so every speed idea was
aimed at a stage whose share nobody knew. This file was a scaffold for an
fp16-PV question from 2026-08-16 that was never run; the question here is
simpler and comes first: **which stage dominates, per block, and what do
`qk_balance`, `rotate` and token routing add, and to which stage.**

## What makes it representative, and what does not

- The inputs are real: full-length packed q/k/v captured from a render, so
  shapes, routed density and the data the quantizers see are the render's.
- The call is the node's: the sink ranges come from `sol_attn_h3._sink_blocks`
  on the capture's layout under the shipped `sink_conditioning`, and tau, tail
  and the options are `h3_config.SOL_RECOMMENDED_CUDA`'s. The older timing
  script passed no sinks at all, which is a different call.
- What a replay cannot see: memory pressure from the other resident models,
  weight streaming between blocks, and whatever else shares the card inside a
  render. `--live-check` exists for that: a render's per-call time for the
  same block, from the node's own timing, belongs beside the replay total in
  the record, and a gap between them is the part this file cannot attribute.
- Five captured blocks are not fifty. Routed density differs by block, so the
  record carries each cell's density and a render-level figure has to weight
  by the live route record (`sol_observe`), not average these cells.

## How the split is measured, and how it could be wrong

Device time per kernel comes from the profiler's CUDA-device events only,
grouped by kernel name. Operator rows are never read, so the nested
op-row/kernel-row double count cannot occur. The control is `coverage`: the
summed kernel time against the same call timed by CUDA events with no
profiler attached. Near one, the stages account for the call. Well under
one, the call has host gaps or synchronisation the stages do not own, and the
shares are shares of less than the whole; well over, kernels overlap or the
profiler inflates them. Either way the number says so instead of hiding it.
A kernel name that matches no stage is listed by name, not dropped.

Needs a quiet card.
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from analyze_sol_error import load_capture, load_cuda_kernel  # noqa: E402

# Kernel-name prefixes, from the launchers in sage_attention/sol_attn_*.cu.
STAGES = (
    ("preprocess", ("prep_",)),
    ("vtranspose", ("vquant_transpose",)),
    ("route", ("sol_route_kernel",)),
    ("exact", ("sol_exact_kernel",)),
    ("token", ("sol_token_",)),
    ("producer", ("sol_producer_kernel",)),
)


def stage_of(kernel_name: str) -> str | None:
    # Names arrive as `void (anonymous namespace)::prep_q<__nv_bfloat16>(...)`,
    # so the prefix is searched for after a `::` or a space, not at the start.
    for stage, prefixes in STAGES:
        if any(re.search(r"(?:^|::|\s)" + re.escape(p), kernel_name) for p in prefixes):
            return stage
    return None


def _node():
    """The pack's Sol node module, imported the way the checks import it."""
    sys.path.insert(0, str(REPO.parent))
    sys.path.insert(0, str(REPO.parents[1]))
    return __import__(f"{REPO.name}.sol_attn_h3", fromlist=["_sink_blocks"])


def _recipe():
    sys.path.insert(0, str(REPO / "workflows"))
    import h3_config
    return h3_config.SOL_RECOMMENDED_CUDA


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--heads", type=int, default=0, help="0 = every head, which is the call a render makes")
    ap.add_argument("--iters", type=int, default=5)
    # The capture set's packed layout, [text | audio | video]; the captures
    # carry tensors only, so the spans are stated here and recorded.
    ap.add_argument("--audio-span", default="395,1545")
    ap.add_argument("--video-start", type=int, default=1545)
    ap.add_argument("--token-aug", type=int, default=64, help="budget for the token-routing arm")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import orjson
    from torch.profiler import ProfilerActivity, profile

    node, recipe = _node(), _recipe()
    sol = load_cuda_kernel()
    params = inspect.signature(sol).parameters
    arms = {"shipped": {"qk_balance": bool(recipe["qk_balance"]), "rotate": bool(recipe["rotate"])}}
    arms["plain"] = {"qk_balance": False, "rotate": False}
    arms["balance+rotate"] = {"qk_balance": True, "rotate": True}
    arms[f"balance+rotate+token{args.token_aug}"] = {"qk_balance": True, "rotate": True, "token_aug": args.token_aug}
    arms = {n: {k: v for k, v in kw.items() if k in params and v} for n, kw in arms.items()}

    cells = {}
    for cap in args.captures:
        q, k, v = load_capture(cap)
        if 0 < args.heads < q.shape[1]:
            q, k, v = q[:, :args.heads], k[:, :args.heads], v[:, :args.heads]
        to = dict(device="cuda", dtype=torch.bfloat16)
        qs, ks, vs = (x.to(**to).permute(0, 2, 1, 3).contiguous() for x in (q, k, v))
        del q, k, v
        tokens = qs.shape[1]
        a0, a1 = (int(x) for x in args.audio_span.split(","))
        layout = {"sol_h3_video_span": (args.video_start, tokens), "sol_h3_audio_span": (a0, a1)}
        sink_kv, sink_q = node._sink_blocks(layout, tokens, recipe["sink_conditioning"])
        m = re.search(r"_b(\d+)_s(\d+)", Path(cap).name)
        cell = f"b{m.group(1)}_s{m.group(2)}" if m else Path(cap).stem
        print(f"{cell}: S={tokens} heads {qs.shape[2]} sinks kv={sink_kv} q={sink_q}")

        def call(kw, cnt=None):
            extra = dict(kw)
            if cnt is not None:
                extra["blk_cnt"] = cnt
            return sol(qs, ks, vs, tau=float(recipe["tau"]), scale=None,
                       sink_blocks=list(sink_kv), sink_q=list(sink_q),
                       topk_ratio=0.0, tail=bool(recipe["pooled_tail"]), **extra)

        rows = {}
        for name, kw in arms.items():
            call(kw); call(kw)
            torch.cuda.synchronize()
            # 1. the call with nothing attached: the number the stages must add up to
            plain = []
            for _ in range(args.iters):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record(); call(kw); e.record(); torch.cuda.synchronize()
                plain.append(s.elapsed_time(e))
            call_ms = sorted(plain)[len(plain) // 2]
            # 2. the same calls under the profiler, device events only
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                for _ in range(args.iters):
                    call(kw)
                torch.cuda.synchronize()
            stage_us, other = {}, {}
            for ev in prof.events():
                if ev.device_type != torch.autograd.DeviceType.CUDA:
                    continue
                us = float(ev.device_time_total if hasattr(ev, "device_time_total") else ev.cuda_time_total)
                st = stage_of(ev.name)
                if st is None:
                    other[ev.name[:60]] = other.get(ev.name[:60], 0.0) + us
                else:
                    stage_us[st] = stage_us.get(st, 0.0) + us
            stage_ms = {s: us / 1000.0 / args.iters for s, us in stage_us.items()}
            other_ms = {n: us / 1000.0 / args.iters for n, us in other.items()}
            kernel_ms = sum(stage_ms.values()) + sum(other_ms.values())
            row = {"call_ms": round(call_ms, 2), "kernel_ms": round(kernel_ms, 2),
                   "coverage": round(kernel_ms / call_ms, 3),
                   "stage_ms": {s: round(x, 2) for s, x in sorted(stage_ms.items())},
                   "stage_share_of_kernels": {s: round(x / kernel_ms, 3) for s, x in sorted(stage_ms.items())},
                   "unmapped_ms": {n: round(x, 3) for n, x in sorted(other_ms.items(), key=lambda t: -t[1])[:8]}}
            if "blk_cnt" in params:
                nb = (tokens + 63) // 64
                cnt = torch.zeros((qs.shape[0], qs.shape[2], nb), dtype=torch.int32, device="cuda")
                call(kw, cnt); torch.cuda.synchronize()
                row["routed_density"] = round(float(cnt.float().mean().item()) / nb, 4)
            rows[name] = row
            shares = "  ".join(f"{s} {row['stage_share_of_kernels'][s]:.0%}" for s in row["stage_ms"])
            print(f"  {name:28s} call {call_ms:7.1f} ms  kernels {kernel_ms:7.1f}  coverage {row['coverage']:.2f}  | {shares}")
        cells[cell] = rows
        del qs, ks, vs
        torch.cuda.empty_cache()

    import importlib.metadata
    record = {
        "what": "comfy_kitchen.sol_attn device time by stage on captured q/k/v, the node's own sinks and the shipped recipe",
        "model": "MiniMax H3, int8 convrot checkpoint, captures from a base 16-step t2v render",
        "conditions": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                       "comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
                       "heads": args.heads or "all", "iters": args.iters,
                       "tau": recipe["tau"], "sink_conditioning": recipe["sink_conditioning"],
                       "pooled_tail": recipe["pooled_tail"],
                       "layout": {"audio_span": args.audio_span, "video_start": args.video_start},
                       "captures": [Path(c).name for c in args.captures]},
        "how_to_read": "stage_ms is device time per call from CUDA-device profiler events grouped by kernel name; "
                       "call_ms is the same call timed by CUDA events with no profiler; coverage = kernel_ms / call_ms "
                       "and says how much of the call the stages account for",
        "cells": cells,
    }
    Path(args.out).write_bytes(orjson.dumps(record, option=orjson.OPT_INDENT_2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
