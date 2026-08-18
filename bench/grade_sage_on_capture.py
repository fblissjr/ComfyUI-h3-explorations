#!/usr/bin/env python3
"""Grade each sage kernel mode against a REAL captured activation, not randn.

This is the run `h3_capture.py` was written to enable and that its own
docstring records as never having happened: "no kernel has yet been graded
against the ones this script produced". `docs/evidence.md` withdrew every
fp8-vs-fp16 accuracy ratio this repo carried, on provenance rather than size --
the surviving sweep measured `torch.randn`, which is not H3's input
distribution, and the competing real-activation figure came from an uncommitted
script across a repo boundary. The withdrawal names exactly one thing that
restores it: grade a kernel against a capture. That is this file.

**What is being decided.** `workflows/h3_config.py` ships
`mode="fp16 (most accurate)"`, the one mode with no `sageattn_consume` entry
point, and `check_bench_matches_shipped.py` records its cost against `auto`.
After the withdrawal, the *numeric* half of that decision is gone and only the
owner's perceptual verdict remains. This measures the numeric half again on
inputs the model actually produces.

## Why a sampled reference, and why that is not a compromise

An exact O(S^2) fp32 reference cannot run at capture length -- `docs/evidence.md`
already records that as the blocker on the Sol diagnostic. But nothing requires
scoring *every* query row. Attention rows are independent: row i's output
depends on q[i] and on all of k/v, and on no other query. So the exact fp32
output for a SAMPLE of rows, computed against the FULL key set, is exact for
those rows -- not an approximation of them. Sampling costs coverage, not
fidelity, and coverage is reported.

Rows are sampled per segment rather than uniformly. H3 packs
`[text | refs | audio | video]` into one sequence and video is the large
majority, so a uniform sample is a video sample wearing a general label. The
segments have different statistics and the reference rows are the ones pinned
across every step, which is where a quantisation error would compound.

## The control

A hand-rolled reference that is wrong makes every mode look wrong together, and
the ranking would still look plausible. So `--self-test` runs the same reference
against `torch.nn.functional.scaled_dot_product_attention` in fp32 on a slice
small enough for both, and refuses to report if they disagree. Run it once on
any new box; it needs no capture.

## Usage

    python bench/grade_sage_on_capture.py --self-test
    python bench/grade_sage_on_capture.py <capture.pt> [--rows 256]

Needs CUDA and the Ada sage fork. Reads the capture read-only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import attention as _attn  # noqa: E402  (after sys.path)


# Segment boundaries are not recorded in the capture, and inventing them would
# be a fabrication. What IS recoverable is that video is the trailing span and
# dominates; `MiniMaxH3Preflight` prints the real split at render time. So rows
# are sampled in equal-count strata across the sequence and reported by
# position, which is honest about what it knows.
def sample_rows(seq_len, n_rows, strata=8, device="cuda"):
    """Evenly-spaced strata, deterministic, no RNG."""
    per = max(1, n_rows // strata)
    idx = []
    for s in range(strata):
        lo = s * seq_len // strata
        hi = (s + 1) * seq_len // strata
        if hi <= lo:
            continue
        step = max(1, (hi - lo) // per)
        idx.extend(range(lo, min(hi, lo + per * step), step))
    return torch.tensor(sorted(set(idx))[:n_rows], device=device, dtype=torch.long)


def reference_rows(q, k, v, rows, head_chunk=8):
    """Exact fp32 attention for `rows`, against every key. Layout [B,H,S,D].

    Chunked over heads for memory only; the maths is unchunked per head.
    """
    b, h, s, d = q.shape
    assert b == 1, "capture is single-batch"
    scale = 1.0 / math.sqrt(d)
    out = torch.empty((1, h, rows.numel(), d), dtype=torch.float32, device=q.device)
    for h0 in range(0, h, head_chunk):
        h1 = min(h, h0 + head_chunk)
        qs = q[0, h0:h1].index_select(1, rows).float()      # [hc, n, d]
        ks = k[0, h0:h1].float()                            # [hc, S, d]
        vs = v[0, h0:h1].float()
        scores = torch.bmm(qs, ks.transpose(1, 2)) * scale  # [hc, n, S]
        probs = torch.softmax(scores, dim=-1)
        out[0, h0:h1] = torch.bmm(probs, vs)
        del qs, ks, vs, scores, probs
    return out


def _err(got, ref):
    """Relative L2 per row, and cosine. Both over the head_dim axis."""
    got = got.float()
    num = torch.linalg.vector_norm(got - ref, dim=-1)
    den = torch.linalg.vector_norm(ref, dim=-1).clamp_min(1e-12)
    rel = (num / den)
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
    return {
        "rel_l2_mean": rel.mean().item(),
        "rel_l2_p99": rel.flatten().quantile(0.99).item(),
        "cos_mean": cos.mean().item(),
        "cos_min": cos.min().item(),
    }


def self_test(device="cuda"):
    """Refuse to report unless the hand-rolled reference matches torch's."""
    torch.manual_seed(0)
    b, h, s, d = 1, 4, 2048, 128
    q = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    k = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    v = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    rows = sample_rows(s, 128, device=device)
    mine = reference_rows(q, k, v, rows)
    # The math backend is the one that is plain fp32 matmul+softmax; flash and
    # mem-efficient would be comparing our reference against another
    # approximation. API moved in torch 2.x, so try the current spelling first.
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        ctx = sdpa_kernel(SDPBackend.MATH)
    except Exception:
        ctx = torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True)
    with ctx:
        full = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    theirs = full.index_select(2, rows)
    e = _err(mine, theirs)
    ok = e["rel_l2_mean"] < 1e-5 and e["cos_min"] > 1 - 1e-6
    print(f"  self-test vs torch SDPA (math backend, fp32): "
          f"rel_l2_mean {e['rel_l2_mean']:.3e}  cos_min {e['cos_min']:.9f}")
    print(f"  {'ok' if ok else 'FAIL'}: the reference {'agrees' if ok else 'DISAGREES'} "
          f"with torch's own attention")
    return ok


def grade(path, n_rows, device="cuda"):
    # weights_only: these are our own tensors, but a capture is a file on a
    # share that other processes write, and nothing here needs pickle.
    blob = torch.load(path, map_location="cpu", weights_only=True)
    q_c, k_c, v_c = blob["q"], blob["k"], blob["v"]
    b, h, s, d = q_c.shape
    print(f"  capture {Path(path).name}")
    print(f"  shape [B={b}, H={h}, S={s}, D={d}] {q_c.dtype}")

    q = q_c.to(device); k = k_c.to(device); v = v_c.to(device)
    rows = sample_rows(s, n_rows, device=device)
    print(f"  scoring {rows.numel()} query rows against all {s} keys, fp32 reference")
    ref = reference_rows(q, k, v, rows)

    results = {}
    for mode in _attn.MODES:
        try:
            fn, kw = _attn.build_kernel(mode)
        except Exception as exc:
            print(f"    {mode:24} unavailable: {exc}")
            continue
        qq, kk, vv = q.clone(), k.clone(), v.clone()
        try:
            out = fn([qq, kk, vv], **dict(kw, tensor_layout="HND"))
        except Exception as exc:
            print(f"    {mode:24} raised: {type(exc).__name__}: {exc}")
            continue
        got = out.index_select(2, rows)
        e = _err(got, ref)
        results[mode] = e
        print(f"    {mode:24} rel_l2 {e['rel_l2_mean']:.4f}  p99 {e['rel_l2_p99']:.4f}  "
              f"cos {e['cos_mean']:.6f}  cos_min {e['cos_min']:.6f}")
        del qq, kk, vv, out, got
        torch.cuda.empty_cache()
    return {"capture": str(path), "seq": s, "heads": h, "head_dim": d,
            "rows_scored": rows.numel(), "modes": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", nargs="*", help="qkv_*.pt from h3_capture.py")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("  needs CUDA; skipped")
        return 2

    print("control:")
    if not self_test():
        print("  refusing to grade: the reference is not trustworthy on this box")
        return 1
    if a.self_test and not a.capture:
        return 0
    if not a.capture:
        print("\nusage: python bench/grade_sage_on_capture.py <capture.pt> [--rows N]")
        return 0

    out = []
    for p in a.capture:
        print("")
        out.append(grade(p, a.rows))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
