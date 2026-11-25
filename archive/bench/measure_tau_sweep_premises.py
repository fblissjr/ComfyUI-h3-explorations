#!/usr/bin/env python3
"""Premises of the live tau sweep (`sol_tau_sweep.py`), measured on capture cells.

    <comfy venv python> bench/measure_tau_sweep_premises.py <video_start> <out.json> <capture.pt> [...]

Needs a quiet card.

1. Row independence: Sol's output on the video rows with the conditioning rows
   exact (`sink_q` = the sink) against the same call with no exact rows
   (`sink_q` = (0, 0)). The sweep runs the second and prices the first.
2. Time of one call: dense INT8, stock bf16 SDPA, Sol per tau with rows routed.
3. The INT8 floor: dense INT8 against stock SDPA, per head, and Sol per tau
   against both references, video rows.
"""
import json
import sys
import time

import torch
import torch.nn.functional as F
import comfy_kitchen as ck

TAUS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
BLOCK = 64


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1e3


def head_sums(a, r, lo, hi, chunk=4):
    """Per head sum |a - r|^2 over rows [lo, hi); a, r are [1, H, S, D]. r None: sum |a|^2."""
    H = a.shape[1]
    out = torch.zeros(H, dtype=torch.float64, device=a.device)
    for h in range(0, H, chunk):
        x = a[:, h:h + chunk, lo:hi].float()
        if r is not None:
            x = x - r[:, h:h + chunk, lo:hi].float()
        out[h:h + chunk] = (x * x).sum(dim=(0, 2, 3), dtype=torch.float64)
    return out


def main():
    video_start, out_path = int(sys.argv[1]), sys.argv[2]
    import importlib.metadata
    record = {"kitchen": importlib.metadata.version("comfy-kitchen"), "taus": TAUS, "video_start": video_start, "cells": {}}
    for path in sys.argv[3:]:
        name = path.split("/")[-1]
        d = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        q, k, v = (d[n].to("cuda", torch.bfloat16) for n in ("q", "k", "v"))        # [1, H, S, D]
        S = q.shape[2]
        nq = (S + BLOCK - 1) // BLOCK
        sink = [0, (video_start + BLOCK - 1) // BLOCK]
        vrow = sink[1] * BLOCK                                                      # first row of a whole video block
        cell = {"tokens": S, "heads": q.shape[1], "sink": sink, "ms": {}, "peak_gib": None}
        torch.cuda.reset_peak_memory_stats()
        ck.int8_attention(q, k, v)                                                  # warm
        dense, cell["ms"]["dense_int8"] = timed(lambda: ck.int8_attention(q, k, v))
        stock, cell["ms"]["stock_sdpa"] = timed(lambda: F.scaled_dot_product_attention(q, k, v))
        den_dense = head_sums(dense, None, vrow, S)
        den_stock = head_sums(stock, None, vrow, S)
        floor = head_sums(dense, stock, vrow, S)
        cell["floor_dense_vs_stock"] = (floor / den_stock).sqrt().tolist()
        cell["floor_pooled"] = float((floor.sum() / den_stock.sum()).sqrt())
        qs, ks, vs = (x.permute(0, 2, 1, 3).contiguous() for x in (q, k, v))        # [1, S, H, D]
        del q, k, v

        def sol(tau, sink_q, cnt=None):
            extra = {} if cnt is None else {"blk_cnt": cnt}
            return ck.sol_attn(qs, ks, vs, tau=tau, scale=None, sink_blocks=sink, sink_q=sink_q,
                               topk_ratio=0.0, tail=True, rotate=True, **extra).permute(0, 2, 1, 3)

        a = sol(1.0, sink)
        b = sol(1.0, [0, 0])
        cell["video_rows_bit_equal"] = bool(torch.equal(a[:, :, vrow:], b[:, :, vrow:]))
        cell["conditioning_rows_bit_equal"] = bool(torch.equal(a[:, :, :vrow], b[:, :, :vrow]))
        cell["video_rows_max_abs_diff"] = float((a[:, :, vrow:].float() - b[:, :, vrow:].float()).abs().max())
        del a, b
        _, cell["ms"]["sol_tau1_rows_exact"] = timed(lambda: sol(1.0, sink))
        cell["per_tau"] = {}
        for tau in TAUS:
            cnt = torch.zeros((1, cell["heads"], nq), dtype=torch.int32, device="cuda")
            o, ms = timed(lambda: sol(tau, [0, 0], cnt))
            nd, ns = head_sums(o, dense, vrow, S), head_sums(o, stock, vrow, S)
            cell["per_tau"][str(tau)] = {
                "ms_rows_routed": ms,
                "err_vs_dense_pooled": float((nd.sum() / den_dense.sum()).sqrt()),
                "err_vs_stock_pooled": float((ns.sum() / den_stock.sum()).sqrt()),
                "err_vs_dense": (nd / den_dense).sqrt().tolist(),
                "err_vs_stock": (ns / den_stock).sqrt().tolist(),
                "video_share": float(cnt[0, :, sink[1]:].double().mean() / nq),
            }
            del o, cnt
        cell["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        record["cells"][name] = cell
        print(name, "video rows bit-equal:", cell["video_rows_bit_equal"],
              "| cond rows equal:", cell["conditioning_rows_bit_equal"],
              "| ms", {k: round(x, 1) for k, x in cell["ms"].items()},
              "| floor pooled", round(cell["floor_pooled"], 5), "| peak GiB", round(cell["peak_gib"], 1), flush=True)
        for tau in TAUS:
            p = cell["per_tau"][str(tau)]
            print(f"   tau {tau}: {p['ms_rows_routed']:.0f} ms, vs dense {p['err_vs_dense_pooled']:.4f}, "
                  f"vs stock {p['err_vs_stock_pooled']:.4f}, video share {p['video_share']:.3f}", flush=True)
        del dense, stock, qs, ks, vs
        torch.cuda.empty_cache()
        with open(out_path, "w") as f:
            json.dump(record, f, indent=1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
