#!/usr/bin/env python3
"""Measure the INSTALLED Sol kernel's exact branch, so two builds can be compared.

Two builds of comfy-kitchen carry the same routing and a different exact
stage: upstream main's (`0.2.31+sol.24908e1`, our `blk_cnt` branch) quantizes
each key block's probabilities against the running maximum of every block
walked so far; kijai's `sol_attn_continued` quantizes against the block's own
maximum, floored twenty doublings under the running one, so a block far below
the running max is not squeezed into the bottom of the 8-bit range. His test
pins the effect at all-routed as relative L2 under 1.6% against about 2%.

This script measures that on whichever build is installed and writes a record
naming it, so running it once per wheel gives a like-for-like comparison at
the call level -- the only level at which a numerical kernel change can be
compared (CLAUDE.md: a rendered clip cannot A/B a numerical change).

Arms, all on the same seeded inputs:

  exact_all_routed      tau -1e9 (every block exact): relative L2 and cosine
                        against fp32 dense attention. kijai's own metric.
  tau_1p0_vs_eager      cosine against the eager reference at the shipped
                        tau, the bar check_solattn_correctness.py uses.
  tau_1p3_vs_eager      the same at the tau that page's older tables used.
  blk_cnt_forced_pairs  the routed count at tau +1e9 equals the closed form,
                        proving the count still comes from the same launch
                        on this build (identical routing is the premise).
  kernel_ms             median of a warm isolated call at a mid shape.
                        ISOLATED KERNEL TIME, not a render time; comparable
                        only between two runs of this script on the same box.

    python bench/measure_sol_exact_variants.py --out bench/results/<date>_sol_exact_<tag>.json

Capture mode, added 2026-09-03 because the random-input arms above measured
kijai's change as a wash (0.00909 against 0.00922) while his PR
(Comfy-Org/comfy-kitchen#150) reports 1.96% -> 1.40% on an H3 capture. Random
q/k/v have no attention sinks and no heavy-tailed rows, which is exactly the
regime a block-max P scale is meant to help, so the two setups disagree on
input, not on the kernel. `--capture DIR` grades every `qkv_*.pt` that
`h3_capture.py` wrote (bf16 `[B, H, S, D]`, block and step as top-level keys)
on the installed build, with an fp32 chunked dense reference computed here:

  exact_all_routed   tau -1e9: relative L2 and cosine against fp32 dense
  topk_0.10          topk_ratio 0.10, kijai's "10% keep": per-head cosine
                     against fp32 dense, mean and worst over heads
  tau_1.0_no_sinks   the shipped tau, but NOT the shipped call: the node
                     passes sink ranges derived from the segment table, and
                     a capture taken with Sol absent carries no table (the
                     Sol rope hook is what publishes it). An UNSUNK
                     DIAGNOSTIC, not a bound: forced-pair counts are
                     monotone in the sink ranges, but relative L2 and
                     cosine need not be, because errors can cancel. Codex
                     caught the mislabel, then the "upper bound" that
                     replaced it, on 2026-09-03.
  sage_<mode>        every sage mode `attention.py::MODES` builds, on the
                     SAME q/k/v, every row, the SAME fp32 reference -- the
                     shipped fallback a `dense_blocks` entry actually runs.
                     The first record graded sage with a separate script at
                     512 sampled rows against float64 and called that "one
                     footing"; it was not, and Codex said so. This is.

Every arm carries whole-tensor relative L2 and cosine, per-head cosine
(mean and worst), and per-ROW relative L2 (mean, p99) and cosine (mean,
min). Whole-tensor relative L2 is norm-weighted, so a block whose error
sits in a few high-norm rows reads far worse under it than per row; block
49 on the first record is that case, and the two must be quoted together.

Same seeded inputs are replaced by the SAME FILES, so two records from two
builds are like-for-like exactly as the random arms are. Run once per wheel
(`PYTHONPATH=<target-install>` selects one without touching the venv).

    python bench/measure_sol_exact_variants.py --capture /path/to/captures --out ...
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

try:
    import torch
except Exception:  # noqa: BLE001  -- main() reports the SKIP
    torch = None
_attn = None


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--timing-shape", default="1,16384,8", help="B,T,H for the isolated timing")
    ap.add_argument("--capture", help="directory of h3_capture.py qkv_*.pt files; grades those instead of random inputs")
    ap.add_argument("--topk", type=float, default=0.10, help="topk_ratio for the keep arm in capture mode")
    ap.add_argument("--chunk", type=int, default=2048, help="query rows per fp32 dense chunk in capture mode")
    ap.add_argument("--no-sage", dest="sage", action="store_false",
                    help="capture mode: skip the sage arms (the shipped fallback on the same cells)")
    args = ap.parse_args()
    try:
        import importlib.metadata as md
        import torch
        import comfy_kitchen as ck
        from comfy_kitchen.backends.eager.sol_attn import sol_attn as eager
    except Exception as exc:                          # noqa: BLE001
        print(f"SKIP: {exc}")
        return 2
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 2
    version = md.version("comfy_kitchen")
    has_cnt = "blk_cnt" in inspect.signature(ck.sol_attn).parameters
    rec = {"produced_by": "bench/measure_sol_exact_variants.py", "comfy_kitchen": version,
           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "arms": {}}
    print(f"installed {version} on {rec['device']}; blk_cnt {'present' if has_cnt else 'ABSENT'}\n")

    def qkv(b, t, h, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        mk = lambda s: torch.randn(b, t, h, 128, device="cuda", dtype=torch.bfloat16, generator=g) * s  # noqa: E731
        return mk(0.5), mk(0.5), mk(1.0)

    def dense(q, k, v):
        qq, kk, vv = (x.permute(0, 2, 1, 3).float() for x in (q, k, v))
        return torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, scale=128 ** -0.5).permute(0, 2, 1, 3)

    def cos(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        return float(a @ b / (a.norm() * b.norm()))

    def rel(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm())

    if args.capture:
        return grade_capture(args, ck, rec, cos, rel)

    # kijai's metric, his shape and seed convention
    q, k, v = qkv(1, 4096, 8, 0)
    out = ck.sol_attn(q, k, v, tau=-1e9)
    ref = dense(q, k, v)
    rec["arms"]["exact_all_routed"] = {"shape": [1, 4096, 8], "rel_l2_vs_dense": rel(out, ref), "cos_vs_dense": cos(out, ref)}
    print(f"  exact, all routed (tau -1e9), T=4096 H=8: rel L2 {rel(out, ref):.5f}, cos {cos(out, ref):.6f}")

    for tau in (1.0, 1.3):
        got = ck.sol_attn(q, k, v, tau=tau)
        want = eager(q, k, v, tau=tau)
        rec["arms"][f"tau_{tau}_vs_eager"] = {"shape": [1, 4096, 8], "cos": cos(got, want), "rel_l2": rel(got, want)}
        print(f"  tau {tau} vs eager reference: cos {cos(got, want):.6f}, rel L2 {rel(got, want):.5f}")

    if has_cnt:
        t, h, n = 320, 2, 5
        q2, k2, v2 = qkv(1, t, h, 7)
        cnt = torch.empty(1, h, n, dtype=torch.int32, device="cuda")
        ck.sol_attn(q2, k2, v2, tau=1e9, sink_blocks=[0, 2], sink_q=[3, 4], blk_cnt=cnt)
        ok = cnt[0, 0].tolist() == [2, 3, 4, 5, 4] and cnt[0, 1].tolist() == [2, 3, 4, 5, 4]
        rec["arms"]["blk_cnt_forced_pairs"] = {"ok": ok, "got": cnt[0, 0].tolist(), "want": [2, 3, 4, 5, 4]}
        print(f"  blk_cnt forced pairs: {'ok' if ok else 'FAIL'} {cnt[0, 0].tolist()}")
    else:
        rec["arms"]["blk_cnt_forced_pairs"] = {"ok": None, "why": "this build has no blk_cnt"}

    b, t, h = (int(x) for x in args.timing_shape.split(","))
    q3, k3, v3 = qkv(b, t, h, 11)
    for _ in range(3):
        ck.sol_attn(q3, k3, v3, tau=1.0)
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); ck.sol_attn(q3, k3, v3, tau=1.0); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    rec["arms"]["kernel_ms"] = {"shape": [b, t, h], "tau": 1.0, "median_ms": times[len(times) // 2],
                                "min_ms": times[0], "note": "isolated warm kernel call, not a render time"}
    print(f"  isolated kernel call, B={b} T={t} H={h}, tau 1.0: median {times[len(times) // 2]:.2f} ms (min {times[0]:.2f})")

    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
        print(f"\nrecord written to {args.out}")
    return 0


def dense_fp32_chunked(q, k, v, chunk):
    """Exact softmax attention in fp32, one head at a time, `chunk` query rows
    per pass, so a 100k-token capture never allocates an S x S score matrix.
    q/k/v are BTHD; the result is BTHD fp32."""
    import torch
    b, t, h, d = q.shape
    scale = d ** -0.5
    out = torch.empty(b, t, h, d, dtype=torch.float32, device=q.device)
    for bi in range(b):
        for hi in range(h):
            kk = k[bi, :, hi].float()
            vv = v[bi, :, hi].float()
            for s0 in range(0, t, chunk):
                qq = q[bi, s0:s0 + chunk, hi].float()
                p = torch.softmax(qq @ kk.T * scale, dim=-1)
                out[bi, s0:s0 + chunk, hi] = p @ vv
    return out


def per_head_cos(a, b):
    """Cosine per head over (B, T, D); a and b are BTHD. One head at a time:
    a whole-tensor fp32 copy of a 104k x 56 x 128 capture is 3 GiB, and the
    first run of this mode OOMed on exactly that with the server resident."""
    out = []
    for hi in range(a.shape[2]):
        x = a[:, :, hi].float().flatten()
        y = b[:, :, hi].float().flatten()
        out.append(float(x @ y / (x.norm() * y.norm())))
    return out


def row_stats(a, b):
    """Per-ROW relative L2 and cosine (a row is one (b, t, h) vector of D), the
    statistic `bench/grade_sage_on_capture.py` reports for sage, so Sol and the
    shipped fallback can be read on one footing. Mean and a tail per arm;
    accumulated per head for the same memory reason as above."""
    rels, coss = [], []
    for hi in range(a.shape[2]):
        x = a[:, :, hi].float()
        y = b[:, :, hi].float()
        rels.append(((x - y).norm(dim=-1) / y.norm(dim=-1).clamp_min(1e-12)).flatten())
        coss.append(torch.nn.functional.cosine_similarity(x, y, dim=-1).flatten())
    rel = torch.cat(rels); cs = torch.cat(coss)
    return {"rel_l2_row_mean": float(rel.mean()), "rel_l2_row_p99": float(rel.quantile(0.99)),
            "cos_row_mean": float(cs.mean()), "cos_row_min": float(cs.min())}


def rel_cos_lean(a, b):
    """Whole-tensor relative L2 and cosine, accumulated per head (same reason)."""
    import math
    diff2 = ref2 = dot = a2 = 0.0
    for hi in range(a.shape[2]):
        x = a[:, :, hi].float().flatten()
        y = b[:, :, hi].float().flatten()
        diff2 += float(((x - y) ** 2).sum()); ref2 += float((y ** 2).sum())
        dot += float(x @ y); a2 += float((x ** 2).sum())
    return math.sqrt(diff2 / ref2), dot / math.sqrt(a2 * ref2)


def grade_capture(args, ck, rec, cos, rel):
    import glob
    import os
    import torch
    global _attn
    if args.sage:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import attention as _attn  # noqa: E402  -- the pack's sage modes, built as the node builds them
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.capture), "qkv_*.pt")))
    if not files:
        print(f"no qkv_*.pt under {args.capture}")
        return 1
    rec["capture"] = {"dir": os.path.basename(os.path.normpath(args.capture)), "files": len(files),
                      "topk_ratio": args.topk, "reference": "fp32 chunked softmax attention, per head"}
    rec["arms"]["per_file"] = []
    print(f"grading {len(files)} capture files, reference fp32 dense, chunk {args.chunk}\n")
    for f in files:
        d = torch.load(f, map_location="cuda", mmap=False)
        # h3_capture writes [B, H, S, D]; the kernel takes [B, S, H, D]
        q, k, v = (d[n].permute(0, 2, 1, 3).contiguous().to(torch.bfloat16) for n in ("q", "k", "v"))
        b, t, h, _ = q.shape
        ref = dense_fp32_chunked(q, k, v, args.chunk)
        row = {"file": os.path.basename(f), "block": d.get("block"), "step": d.get("step"),
               "sigma": d.get("sigma"), "seq_len": t, "heads": h, "kernel_captured_under": d.get("kernel")}
        out = ck.sol_attn(q, k, v, tau=-1e9)
        r, c = rel_cos_lean(out, ref)
        row["exact_all_routed"] = {"rel_l2": r, "cos": c, **row_stats(out, ref)}
        del out
        out = ck.sol_attn(q, k, v, topk_ratio=args.topk)
        ph = per_head_cos(out, ref)
        rs = row_stats(out, ref)
        del out
        row[f"topk_{args.topk:.2f}"] = {"cos_mean": sum(ph) / len(ph), "cos_worst": min(ph), "per_head": ph, **rs}
        out = ck.sol_attn(q, k, v, tau=1.0)
        ph = per_head_cos(out, ref)
        rs = row_stats(out, ref)
        r, c = rel_cos_lean(out, ref)
        del out
        row["tau_1.0_no_sinks"] = {"rel_l2": r, "cos": c, "cos_mean": sum(ph) / len(ph),
                                   "cos_worst": min(ph), "per_head": ph, **rs,
                                   "note": "no sink ranges: an unsunk diagnostic, not the shipped call and not a bound on its error (errors can cancel)"}
        if args.sage:
            for mode in _attn.MODES:
                try:
                    fn, kw = _attn.build_kernel(mode)
                except Exception as exc:                          # noqa: BLE001
                    row[f"sage_{mode}"] = {"unavailable": str(exc)}
                    continue
                hnd = [x.permute(0, 2, 1, 3).contiguous() for x in (q, k, v)]
                try:
                    out = fn(hnd, **dict(kw, tensor_layout="HND")).permute(0, 2, 1, 3)
                except Exception as exc:                          # noqa: BLE001
                    row[f"sage_{mode}"] = {"raised": f"{type(exc).__name__}: {exc}"}
                    del hnd
                    continue
                ph = per_head_cos(out, ref)
                rs = row_stats(out, ref)
                r, c = rel_cos_lean(out, ref)
                row[f"sage_{mode}"] = {"rel_l2": r, "cos": c, "cos_mean": sum(ph) / len(ph),
                                       "cos_worst": min(ph), "per_head": ph, **rs}
                del out, hnd
                torch.cuda.empty_cache()
        rec["arms"]["per_file"].append(row)
        print(f"  b{row['block']:>2} s{row['step']:>2} S={t}: all-routed rel L2 {row['exact_all_routed']['rel_l2']:.5f}  "
              f"topk {args.topk:.2f} cos {row[f'topk_{args.topk:.2f}']['cos_mean']:.4f}/{row[f'topk_{args.topk:.2f}']['cos_worst']:.4f}  "
              f"tau 1.0 (no sinks) cos {row['tau_1.0_no_sinks']['cos_mean']:.4f}/{row['tau_1.0_no_sinks']['cos_worst']:.4f}"
              + ("  sage auto row rel L2 %.4f" % row["sage_auto"]["rel_l2_row_mean"] if "sage_auto" in row and "rel_l2_row_mean" in row["sage_auto"] else ""))
        del d, q, k, v, ref
        torch.cuda.empty_cache()
    rows = rec["arms"]["per_file"]
    n = len(rows)
    rec["arms"]["aggregate"] = {
        "files": n,
        "all_routed_rel_l2_mean": sum(r["exact_all_routed"]["rel_l2"] for r in rows) / n,
        f"topk_{args.topk:.2f}_cos_mean": sum(r[f"topk_{args.topk:.2f}"]["cos_mean"] for r in rows) / n,
        f"topk_{args.topk:.2f}_cos_worst": min(r[f"topk_{args.topk:.2f}"]["cos_worst"] for r in rows),
        "tau_1.0_no_sinks_cos_mean": sum(r["tau_1.0_no_sinks"]["cos_mean"] for r in rows) / n,
        "tau_1.0_no_sinks_cos_worst": min(r["tau_1.0_no_sinks"]["cos_worst"] for r in rows),
        "all_routed_rel_l2_row_mean": sum(r["exact_all_routed"]["rel_l2_row_mean"] for r in rows) / n,
        f"topk_{args.topk:.2f}_rel_l2_row_mean": sum(r[f"topk_{args.topk:.2f}"]["rel_l2_row_mean"] for r in rows) / n,
        "tau_1.0_no_sinks_rel_l2_row_mean": sum(r["tau_1.0_no_sinks"]["rel_l2_row_mean"] for r in rows) / n,
        **{f"{m}_rel_l2_mean": sum(r[m]["rel_l2"] for r in rows) / n
           for m in rows[0] if m.startswith("sage_") and all("rel_l2" in r.get(m, {}) for r in rows)},
        **{f"{m}_rel_l2_row_mean": sum(r[m]["rel_l2_row_mean"] for r in rows) / n
           for m in rows[0] if m.startswith("sage_") and all("rel_l2_row_mean" in r.get(m, {}) for r in rows)},
        "note": "means over files weight every (block, step) equally; worst is the single worst head anywhere",
    }
    a = rec["arms"]["aggregate"]
    print(f"\naggregate over {n} files: all-routed rel L2 {a['all_routed_rel_l2_mean']:.5f}; "
          f"topk {args.topk:.2f} cos {a[f'topk_{args.topk:.2f}_cos_mean']:.4f}/{a[f'topk_{args.topk:.2f}_cos_worst']:.4f}; "
          f"tau 1.0 (no sinks) cos {a['tau_1.0_no_sinks_cos_mean']:.4f}/{a['tau_1.0_no_sinks_cos_worst']:.4f}; "
          + "; ".join(f"{k} {v:.5f}" for k, v in a.items() if k.startswith("sage_")))
    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
        print(f"record written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
