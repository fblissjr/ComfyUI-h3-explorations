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


def reference_rows(q, k, v, rows, head_chunk=8, dtype=torch.float64):
    """Attention for `rows` against every key, in `dtype`. Layout [B,H,S,D].

    **float64, not fp32, and that is the whole point.** An fp32 reference and
    fp32 SDPA sit the SAME distance from a float64 arbiter -- measured here
    2026-08-18 at 4.18e-04 each on random input -- while differing from each
    other by 5.09e-04. So two fp32 paths disagree by more than either one's
    actual error, purely from accumulation order, and an fp32 reference cannot
    resolve a kernel difference smaller than that. Attention output is an
    average over the whole key set, so cancellation amplifies relative error;
    this is not a shape where fp32 is "obviously fine".

    float64 is affordable because only sampled rows are scored: the score
    matrix is [head_chunk, rows, S], not [H, S, S].

    Chunked over heads for memory only; the maths is unchunked per head.
    """
    b, h, s, d = q.shape
    assert b == 1, "capture is single-batch"
    scale = 1.0 / math.sqrt(d)
    out = torch.empty((1, h, rows.numel(), d), dtype=dtype, device=q.device)
    for h0 in range(0, h, head_chunk):
        h1 = min(h, h0 + head_chunk)
        qs = q[0, h0:h1].index_select(1, rows).to(dtype)    # [hc, n, d]
        ks = k[0, h0:h1].to(dtype)                          # [hc, S, d]
        vs = v[0, h0:h1].to(dtype)
        scores = torch.bmm(qs, ks.transpose(1, 2)) * scale  # [hc, n, S]
        probs = torch.softmax(scores, dim=-1)
        out[0, h0:h1] = torch.bmm(probs, vs)
        del qs, ks, vs, scores, probs
    return out


def _err(got, ref):
    """Relative L2 per row, and cosine. Both over the head_dim axis."""
    ref = ref.double()
    got = got.double()
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
    """Refuse to report unless the float64 reference tracks torch's own attention,
    AND a deliberately wrong reference is shown to FAIL the same comparison.

    The second half is the part that makes this a control rather than a
    formality. `CLAUDE.md`: a check whose input already satisfies the expected
    outcome cannot fail. Agreement alone would pass just as happily if the
    threshold were loose enough to admit anything, so the negative arm perturbs
    the softmax scale -- the single most likely thing to get wrong in a
    hand-rolled attention -- and requires that it be caught.
    """
    torch.manual_seed(0)
    b, h, s, d = 1, 4, 2048, 128
    q = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    k = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    v = torch.randn(b, h, s, d, device=device, dtype=torch.float32)
    rows = sample_rows(s, 128, device=device)

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        ctx = sdpa_kernel(SDPBackend.MATH)
    except Exception:
        ctx = torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True)
    with ctx:
        theirs = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    theirs = theirs.index_select(2, rows)

    exact = reference_rows(q, k, v, rows)                    # float64
    agree = _err(theirs, exact)["rel_l2_mean"]

    # Negative arm: same code path, scale wrong by 1%. If this passes, the
    # tolerance is not discriminating and no result below is trustworthy.
    d_wrong = d
    saved = math.sqrt
    try:
        math.sqrt = lambda x, _s=saved: _s(x) * 1.01     # perturb 1/sqrt(d)
        wrong = reference_rows(q, k, v, rows)
    finally:
        math.sqrt = saved
    disagree = _err(theirs, wrong)["rel_l2_mean"]

    TOL = 2e-3          # fp32 SDPA against an exact reference, at this shape
    ok = agree < TOL and disagree > TOL
    print(f"  torch fp32 SDPA vs our float64 reference : {agree:.3e}   (tolerance {TOL:.0e})")
    print(f"  same, with the softmax scale 1% wrong    : {disagree:.3e}   (must exceed it)")
    print(f"  {'ok' if ok else 'FAIL'}: the reference tracks torch "
          f"{'and the check can fail' if ok else '-- or the negative arm did not fail'}")
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
    print(f"  scoring {rows.numel()} query rows against all {s} keys, float64 reference")
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
