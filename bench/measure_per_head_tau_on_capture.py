#!/usr/bin/env python3
"""What a tau per head would buy, on captured q/k/v: Sol's error and routed share per head, per tau.

    <comfy venv python> bench/measure_per_head_tau_on_capture.py <capture.pt> [...] \\
        --video-start N --out data/results/<date>_per_head_tau_<what>.json

`comfy_kitchen.sol_attn` takes one tau per call, but its routing threshold is
stored per head and query block (`sol_attn_preprocess.cu`, `thr_of`), and a
head's routing does not read another head's. So one call at tau t gives every
head's result at t, and a per-head assignment can be priced from a sweep
without a kernel change: for each head, take its error and routed share from
the call at the tau assigned to it.

Per cell and tau, per head: the squared error against the dense kitchen kernel
on the same tensors (numerator and denominator kept apart so heads pool), and
the routed share of pairs from the kernel's own `blk_cnt`. The call is the
node's shipped one (rotated quantizer, pooled tail, the conditioning prefix as
exact keys and exact rows).

Then the question: at the pooled error a single tau gives, how much routed
share does the best per-head assignment save, and at the same routed share,
how much error? A Lagrangian sweep over the discrete taus; an upper bound on
what a calibrated table gives, since it is fitted and scored on the same cell.

The reference is the dense INT8 kernel, not exact attention: this prices
Sol's routing against the fallback the node already runs. Needs a quiet card.
"""
import argparse
import json
import math
import re
from pathlib import Path

import torch
import comfy_kitchen as ck

TAUS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
BLOCK = 64


def sweep(path, video_start):
    d = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    q, k, v = (d[n].to("cuda", torch.bfloat16) for n in ("q", "k", "v"))    # [1, H, S, D]
    dense = ck.int8_attention(q, k, v).float()                              # [1, H, S, D]
    den = dense.pow(2).sum(dim=(0, 2, 3))                                    # [H]
    qs, ks, vs = (x.permute(0, 2, 1, 3).contiguous() for x in (q, k, v))    # [1, S, H, D]
    del q, k, v
    H, S = dense.shape[1], dense.shape[2]
    nq = (S + BLOCK - 1) // BLOCK
    sink = [0, (video_start + BLOCK - 1) // BLOCK]
    rows = {}
    for tau in TAUS:
        cnt = torch.zeros((1, H, nq), dtype=torch.int32, device="cuda")
        out = ck.sol_attn(qs, ks, vs, tau=tau, scale=None, sink_blocks=sink, sink_q=sink,
                          topk_ratio=0.0, tail=True, rotate=True, blk_cnt=cnt)
        num = (out.permute(0, 2, 1, 3).float() - dense).pow(2).sum(dim=(0, 2, 3))
        share = cnt[0].double().sum(dim=1) / float(nq * nq)
        rows[tau] = {"num": num.tolist(), "share": share.tolist()}
        del out, cnt
    out = {"tokens": S, "heads": H, "den": den.tolist(), "taus": rows}
    del dense, qs, ks, vs
    torch.cuda.empty_cache()
    return out


def assign(cell, target_num=None, target_share=None):
    """Best per-head tau by a Lagrangian sweep: minimise share + lam * num."""
    H = cell["heads"]
    taus = sorted(cell["taus"])
    num = [[cell["taus"][t]["num"][h] for t in taus] for h in range(H)]
    share = [[cell["taus"][t]["share"][h] for t in taus] for h in range(H)]
    best = None
    scale = sum(cell["den"])
    for e in range(-60, 61):
        lam = (10.0 ** (e / 6.0)) / scale
        pick = [min(range(len(taus)), key=lambda i: share[h][i] + lam * num[h][i] * H) for h in range(H)]
        tn = sum(num[h][pick[h]] for h in range(H))
        ts = sum(share[h][pick[h]] for h in range(H)) / H
        ok = (target_num is not None and tn <= target_num) or (target_share is not None and ts <= target_share)
        if not ok:
            continue
        key = ts if target_num is not None else tn
        if best is None or key < best[0]:
            best = (key, tn, ts, [taus[i] for i in pick])
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--video-start", type=int, required=True)
    ap.add_argument("--base-tau", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    record = {"kitchen": getattr(ck, "__version__", None), "taus": TAUS, "base_tau": args.base_tau,
              "video_start": args.video_start, "cells": {}, "summary": {}}
    for path in args.captures:
        m = re.search(r"_b(\d+)_s(\d+)", path)
        cellname = m.group(0)[1:] if m else Path(path).stem
        with torch.no_grad():
            cell = sweep(path, args.video_start)
        record["cells"][cellname] = cell
        H = cell["heads"]
        b = cell["taus"][args.base_tau]
        num0, share0 = sum(b["num"]), sum(b["share"]) / H
        den = sum(cell["den"])
        eq_err = assign(cell, target_num=num0)
        eq_share = assign(cell, target_share=share0)
        if eq_err is None or eq_share is None:   # the base tau is itself in the sweep, so this cannot happen
            raise SystemExit(f"{cellname}: no assignment meets the base tau's own error or share")
        s = {"base_err": math.sqrt(num0 / den), "base_share": share0,
             "equal_err_share": eq_err[2], "equal_err_err": math.sqrt(eq_err[1] / den),
             "equal_share_err": math.sqrt(eq_share[1] / den), "equal_share_share": eq_share[2],
             "taus_at_equal_err": eq_err[3]}
        record["summary"][cellname] = s
        picks = eq_err[3]
        hist = {t: picks.count(t) for t in sorted(set(picks))}
        print(f"{cellname}: tau {args.base_tau} err {s['base_err']:.4f} share {share0:.4f} | "
              f"same error: share {s['equal_err_share']:.4f} ({100 * (s['equal_err_share'] / share0 - 1):+.1f}%) | "
              f"same share: err {s['equal_share_err']:.4f} ({100 * (s['equal_share_err'] / s['base_err'] - 1):+.1f}%) | "
              f"heads per tau {hist}", flush=True)
        Path(args.out).write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
