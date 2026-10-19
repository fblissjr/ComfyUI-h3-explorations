#!/usr/bin/env python3
"""#45: core's kitchen VSA against a FastVideo-exact reference, on captured FastH3 attention. GPU.

`docs/state/open_experiments.md` #45 asks whether part of FastH3's over-polish is the
kitchen's VSA selection rather than what FastH3 was trained with. Core's VSA
(`comfy_extras/nodes_sparse_attention.py`, via `ck.sol_attn_chunked`) departs
from FastVideo's training in three ways (`bench/results/2026-09-27_attention_parity.md`):
it keeps `round(0.2n)` video tiles plus a tie back-off where FastVideo keeps
`ceil`, it forces the +-1 diagonal exact, and it quantises q, k and v to int8.

Input: a capture from `core_sparse_capture.py` (`qkvpre_*.pt` and `gate_*.pt`
per cell, `pre=` armed). Per cell this grades, all against exact attention:

  kitchen_full   core's own call replayed: `ck.sol_attn_chunked` on the captured
                 projection in core's tile order and chunk size, core's tile plan
                 and rope, sinks (0, n_prefix), tail off, with the coarse gate.
                 Statistics are fresh (the first call's): the real step-2 and
                 step-6 calls used the previous step's pooled K mean and V scale,
                 which a capture does not hold.
  kitchen_fine   the same call without the coarse gate (the fine term alone).
  ref_full       FastVideo's `video_sparse_attn_h3` semantics in torch: fp32
                 tile-mean pooled q and k, scores q.k/sqrt(d), video key tiles by
                 top-k with k = ceil(0.2 n_video), prefix key tiles always kept,
                 prefix query tiles dense, no forced diagonal; block-sparse
                 softmax attention in fp32 over the kept tiles (FastVideo's kernel
                 takes bf16 with fp32 accumulation; fp32 here is the stricter
                 reference); plus softmax(scores) @ pooled v, broadcast per tile
                 and times the gate.
  ref_fine       the same without the coarse term.
  exact          dense fp32 softmax attention on the rebuilt q, k, v
                 (`measure_sol_exact_variants.dense_fp32_chunked`).

and kitchen against reference, fine and full. q, k and v are rebuilt with the
kitchen's own `rms_rope_split_half_` on the captured projection, as
`grade_sol_impl_on_capture.py` does.

Control, on the first cell: the reference with every tile kept must equal
exact attention to fp32 rounding. It proves the tile gather and the pad mask.
A reference that fails it grades nothing.

What it does not show: which tiles the kitchen kept (the kernel does not
report it). A kitchen output that sits farther from exact than the reference,
or far from the reference itself, is the observable. The owner's eye on a
reference render is what would say whether the difference is visible.

    python bench/grade_vsa_selection_on_capture.py --capture DIR [--out OUT.json] [--limit K]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parents[1]
TILE = 64
SPARSITY = 0.8          # FastH3 V2's contract (`h3_config.FASTH3_CONTRACT_VSA` keep 20%)
CELL = re.compile(r"qkvpre_L\d+_S\d+_b(\d+)_s(\d+)(?:_r(\d+))?\.pt$")


def ref_vsa(qt, kt, vt, gate_t, vbs, n_prefix, sparsity, batch_tiles=1, only_tiles=None):
    """FastVideo H3 VSA in torch, tile order. qt/kt/vt: [n, H, D] bf16 with pad
    rows zero; gate_t: [n, H, D] or None; vbs: [T] valid rows per tile.
    Returns (fine, coarse) as [n, H, D] bf16, each computed in fp32 (coarse
    None without a gate). `only_tiles` limits the query tiles computed (the
    control); rows of other tiles are left zero."""
    import torch
    n, H, D = qt.shape
    T = n // TILE
    scale = D ** -0.5
    valid = (torch.arange(TILE, device=qt.device)[None, :] < vbs[:, None])          # [T, 64]
    div = vbs.to(torch.float32).clamp_min(1)[:, None, None]
    qp = qt.view(T, TILE, H, D).sum(1, dtype=torch.float32) / div                    # [T, H, D]
    kp = kt.view(T, TILE, H, D).sum(1, dtype=torch.float32) / div
    scores = torch.einsum("thd,shd->hts", qp, kp) * scale                            # [H, T, T]
    n_video = T - n_prefix
    k_vid = max(1, min(math.ceil((1 - sparsity) * n_video), n_video))
    kh = kt.view(T, TILE, H, D).permute(2, 0, 1, 3)                                  # [H, T, 64, D]
    vh = vt.view(T, TILE, H, D).permute(2, 0, 1, 3)
    qh = qt.view(T, TILE, H, D).permute(2, 0, 1, 3)
    fine = torch.zeros(H, T, TILE, D, dtype=torch.bfloat16, device=qt.device)
    want = None if only_tiles is None else set(int(x) for x in only_tiles)
    hidx = torch.arange(H, device=qt.device)[:, None, None]

    # prefix query tiles are dense: every valid key row. One head at a time:
    # all heads' keys in fp32 at once is 6 GB at 104k rows and ran out of
    # memory on the first real cell.
    vmask = valid.reshape(-1)
    pre_tiles = [t0 for t0 in range(n_prefix) if want is None or t0 in want]
    if pre_tiles:
        for h in range(H):
            kk = kh[h].reshape(T * TILE, D).float()
            vv = vh[h].reshape(T * TILE, D).float()
            for t0 in pre_tiles:
                s = (qh[h, t0].float() @ kk.T) * scale                               # [64, n]
                s = s.masked_fill(~vmask[None, :], float("-inf"))
                fine[h, t0] = (torch.softmax(s, -1) @ vv).to(torch.bfloat16)
            del kk, vv

    video_tiles = list(range(n_prefix, T)) if want is None else sorted(x for x in want if x >= n_prefix)
    for i0 in range(0, len(video_tiles), batch_tiles):
        tq = torch.tensor(video_tiles[i0:i0 + batch_tiles], device=qt.device)
        vid = scores[:, tq, n_prefix:]                                               # [H, B, n_video]
        top = vid.topk(k_vid, dim=-1).indices + n_prefix                             # [H, B, k_vid]
        pre = torch.arange(n_prefix, device=qt.device).expand(H, len(tq), n_prefix)
        sel = torch.cat([pre, top], dim=-1)                                          # [H, B, K]
        K = sel.shape[-1]
        keys = kh[hidx, sel].reshape(H, len(tq), K * TILE, D).float()
        vals = vh[hidx, sel].reshape(H, len(tq), K * TILE, D).float()
        km = valid[sel].reshape(H, len(tq), K * TILE)
        s = (qh[:, tq].float() @ keys.transpose(-1, -2)) * scale                     # [H, B, 64, K*64]
        s = s.masked_fill(~km[:, :, None, :], float("-inf"))
        fine[:, tq] = (torch.softmax(s, -1) @ vals).to(torch.bfloat16)
        del keys, vals, s
    fine = fine.permute(1, 2, 0, 3).reshape(n, H, D)

    coarse = None
    if gate_t is not None:
        vp = vt.view(T, TILE, H, D).sum(1, dtype=torch.float32) / div                # [T, H, D]
        out_c = torch.softmax(scores, -1) @ vp.permute(1, 0, 2)                      # [H, T, D]
        coarse = (out_c.permute(1, 0, 2)[:, None] * gate_t.view(T, TILE, H, D).float()).to(torch.bfloat16).reshape(n, H, D)
    return fine, coarse, {"n_tiles": T, "n_prefix": n_prefix, "n_video": n_video, "k_vid_ref": k_vid,
                          "k_vid_kitchen_round": int(round((1 - sparsity) * n_video))}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--no-control", dest="control", action="store_false")
    args = ap.parse_args()

    sys.path.insert(0, str(COMFY))
    sys.path.insert(0, str(HERE))
    try:
        import torch
        import comfy.quant_ops
        ck = comfy.quant_ops.ck
        import comfy_extras.nodes_sparse_attention as core_sparse
        import measure_sol_exact_variants as msev
        from grade_sol_impl_on_capture import segment_stats
    except Exception as exc:                          # noqa: BLE001
        print(f"SKIP: {type(exc).__name__}: {exc}")
        return 2
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 2

    cells = []
    for p in sorted(glob.glob(os.path.join(args.capture, "qkvpre_*.pt"))):
        m = CELL.search(os.path.basename(p))
        if m:
            cells.append((int(m.group(3) or 0), int(m.group(1)), int(m.group(2)), p))
    cells.sort()
    if args.limit:
        cells = cells[:args.limit]
    if not cells:
        print(f"no qkvpre_*.pt cells in {args.capture}")
        return 1

    def gate_of(path):
        g = path.replace("qkvpre_", "gate_")
        return g if os.path.exists(g) else None

    def metrics(a, b, segments):
        r, c = msev.rel_cos_lean(a, b)
        return {"rel_l2": r, "cos": c, "segments": segment_stats(a, b, segments)}

    rows, dev = [], "cuda"
    patch = core_sparse.SparseAttnPatch(
        tau=1.3, topk_ratio=1 - SPARSITY, vsa=True, sigma_start=float("inf"), sigma_end=float("-inf"),
        min_tokens=0, dense_blocks=set(), sink_conditioning="exact_kv_and_rows", extra_tokens=0, verbose=False)
    with torch.inference_mode():
        for ci, (render, block, step, path) in enumerate(cells):
            d = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            H, D = int(d["heads"]), int(d["head_dim"])
            S = int(d["seq_len"])
            segments = [tuple(x) for x in (d.get("segments") or [])]
            layout = types.SimpleNamespace(signature=tuple(d["layout_signature"]), segments=segments, seq_len=S)
            plan = patch.vsa_plan(layout, dev)
            n, n_prefix = plan["n"], plan["n_prefix"]
            src, inv = plan["src"], plan["inv"]
            live = src >= 0
            rope = d["rope_freqs"].to(dev)
            qw, kw = d["q_norm_weight"].to(dev), d["k_norm_weight"].to(dev)
            eps, rot = float(d["rope_eps"]), int(d["rot_dim"])
            gp = gate_of(path)
            gate = torch.load(gp, map_location="cpu", weights_only=True, mmap=True)["gate"].to(dev, torch.bfloat16) if gp else None
            row = {"file": os.path.basename(path), "gate_file": os.path.basename(gp) if gp else None,
                   "render": render, "block": block, "step": step, "sigma": d.get("sigma"), "seq_len": S,
                   "chunk_check": d.get("chunk_check")}

            # -- kitchen: core's call, replayed ------------------------------
            qkv = d["qkv"].to(dev, torch.bfloat16)
            qkv_t = torch.zeros(n, qkv.shape[1], dtype=qkv.dtype, device=dev)
            qkv_t[live] = qkv[src[live]]
            freqs = patch.vsa_rope_freqs(rope if rope.dim() > 1 and rope.shape[0] == 1 else rope.unsqueeze(0), plan)

            def kitchen(with_gate):
                extra = {"tail": False, "block_len": plan["block_len"]}
                if with_gate and gate is not None:
                    gb = torch.zeros(n, H * D, dtype=torch.bfloat16, device=dev)
                    gb[live] = gate[src[live]]
                    extra["coarse_gate"] = gb.view(1, n, H, D)

                def chunks():
                    for i in range(0, n, core_sparse.PRODUCER_CHUNK):
                        yield qkv_t[i:i + core_sparse.PRODUCER_CHUNK].clone()
                out, _, _ = ck.sol_attn_chunked(
                    chunks, n, H, freqs, (qw, kw), kmean=None, vscale=None,
                    tau=patch.tau, topk_ratio=patch.topk_ratio, token_aug=0,
                    sink_blocks=[0, n_prefix], sink_q=[0, n_prefix], rope_eps=eps, **extra)
                return out.view(n, H * D)[inv].view(1, S, H, D)
            k_full = kitchen(True)
            k_fine = kitchen(False) if gate is not None else k_full
            del qkv_t

            # -- rebuild q, k, v as the forward hands them -------------------
            buf = qkv.clone()
            del qkv
            q, k, v = buf.split(H * D, dim=-1)
            q, k, v = (t.view(1, S, H, D) for t in (q, k, v))
            ck.rms_rope_split_half_(q, k, rope, qw, kw, epsilon=eps, rot_dim=rot)

            exact = msev.dense_fp32_chunked(q, k, v, args.chunk)

            # -- reference, tile order ---------------------------------------
            def tiled(t):
                o = torch.zeros(n, H, D, dtype=t.dtype, device=dev)
                o[live] = t[0, src[live]]
                return o
            qt, kt, vt = tiled(q), tiled(k), tiled(v)
            del q, k, v, buf
            gt = None
            if gate is not None:
                gt = torch.zeros(n, H, D, dtype=torch.bfloat16, device=dev)
                gt[live] = gate.view(S, H, D)[src[live]]
            gate = None

            if args.control and ci == 0:
                # Every tile kept, on a sample of query tiles (one prefix, several
                # video, spread along the sequence): must equal exact attention.
                T = n // TILE
                gen = torch.Generator().manual_seed(0)
                vids = torch.randperm(T - n_prefix, generator=gen)[:6] + n_prefix
                tiles = sorted([0] + vids.tolist())
                cf, _, _ = ref_vsa(qt, kt, vt, None, plan["block_len"], n_prefix, 0.0, only_tiles=tiles)
                rows_t = torch.cat([torch.arange(x * TILE, (x + 1) * TILE, device=dev) for x in tiles])
                keep = src[rows_t] >= 0
                a = cf[rows_t[keep]].float()
                b = exact[0, src[rows_t[keep]]]
                cr = float((a - b).norm() / b.norm())
                row["control_ref_all_tiles_vs_exact"] = {"rel_l2": cr, "tiles": tiles}
                print(f"  control: reference with every tile kept vs exact, {len(tiles)} query tiles: rel_l2 {cr:.2e}")
                del cf
                if cr > 1e-2:
                    print("FAIL: the reference does not reproduce exact attention with every tile kept; nothing graded")
                    return 1

            fine, coarse, sel = ref_vsa(qt, kt, vt, gt, plan["block_len"], n_prefix, SPARSITY)
            row["selection"] = sel
            r_fine = fine[inv].view(1, S, H, D)
            if coarse is not None:
                full = fine.float().add_(coarse.float())
                row["coarse_share_ref"] = float(coarse.float().norm() / full.norm())
                r_full = full.to(torch.bfloat16)[inv].view(1, S, H, D)
                del full
            else:
                r_full = r_fine
            del fine, coarse, qt, kt, vt, gt

            row["vs_exact"] = {
                "kitchen_full": metrics(k_full, exact, segments), "kitchen_fine": metrics(k_fine, exact, segments),
                "ref_full": metrics(r_full, exact, segments), "ref_fine": metrics(r_fine, exact, segments)}
            row["kitchen_vs_ref"] = {"full": metrics(k_full, r_full, segments),
                                     "fine": metrics(k_fine, r_fine, segments)}
            ve = row["vs_exact"]
            print(f"  b{block:>2} s{step} sigma {row['sigma']}: vs exact  kitchen {ve['kitchen_full']['rel_l2']:.4f} "
                  f"(fine {ve['kitchen_fine']['rel_l2']:.4f})  ref {ve['ref_full']['rel_l2']:.4f} "
                  f"(fine {ve['ref_fine']['rel_l2']:.4f})  kitchen vs ref {row['kitchen_vs_ref']['full']['rel_l2']:.4f}  "
                  f"k_vid ref {sel['k_vid_ref']} / round {sel['k_vid_kitchen_round']} of {sel['n_video']}")
            rows.append(row)
            del exact, k_full, k_fine, r_fine, r_full, gate
            torch.cuda.empty_cache()

    import comfy_kitchen
    out = {"measured_by": "bench/grade_vsa_selection_on_capture.py", "capture": os.path.basename(os.path.normpath(args.capture)),
           "sparsity": SPARSITY, "comfy_kitchen": getattr(comfy_kitchen, "__version__", None),
           "torch": torch.__version__, "cells": rows}
    path = args.out or str(REPO / "data" / "results" / f"{os.path.basename(os.path.normpath(args.capture))}_vsa_grade.json")
    Path(path).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
