#!/usr/bin/env python3
"""SCAFFOLDING -- NOT IMPLEMENTED. Gate B0b: decompose Sol-Attn's error.

**Nothing here runs.** `main()` raises. This file fixes the design before the
measurement, because the measurement is the one that can close Track B without
a kernel being written. See
`internal/plan_2026-08-16_sol_fp16_and_triton_retirement.md`, gate B0b.

## The question

Is a 16-bit PV matmul in `sol_attn_exact.cu` worth building? That turns on how
Sol's total error splits:

    total error  =  sparsity error  +  quantization error
                    (the algorithm)    (the kernel's INT8 arithmetic)

A 16-bit PV can only shrink the second term. **If quantization error is small
against sparsity error, the change buys nothing measurable and Track B closes.**
That is the cheap answer, and it costs hours of Python against days of CUDA.

## Why this is not already answered

`bench/check_solattn_correctness.py` computes both quantities, and says so in
its own output -- but at T=512 on `torch.randn`, where it prints DOUBLY
PESSIMISTIC, DO NOT QUOTE and names two reasons:

1. `torch.randn` gives a near-uniform softmax, so attention mass does not
   concentrate and there is nothing for a block router to find. **The premise
   of the method is absent**, so the sparsity term is measured where it cannot
   work.
2. 8 blocks is a different regime from production's ~1,700 -- **derived from
   the token grid, not measured**: 362 frames at 1344x768 is 107,856 video
   tokens on the 17n+5 grid, over 64. The only measured sequence, 104,277
   tokens, was taken at 345 frames, which is a legal count but not the
   ceiling. Either way it is a different regime, not a small version of one.

This script is that check re-run somewhere the premise holds. It is the same
decomposition on real captured activations.

## Inputs, and the gap that matters

`~/Storage/h3_captures/2026-08-15_dense_124f_1344x768/` holds blocks 0/24/49 at
`[1, 56, 37826, 128]` bf16, step 1, captured after the fused RMSNorm+RoPE.

**Those captures are t2v with no references, on the fl2va model.** The owner's
actual work is ref2va with references -- 20 of 35 shipped API graphs are
ref2va. That matters here specifically, not in general:

- `docs/SOLATTN.md`: reference rows are pinned exact by
  `sink_conditioning="exact_kv_and_rows"`, so **reference-heavy is where Sol has
  the LEAST room**. A t2v decomposition therefore measures the workload where
  Sol looks BEST, and the answer will not transfer to the one being run.
- More rows pinned exact means a larger exact branch, which is exactly the
  branch a 16-bit PV makes more expensive. So the t2v number understates the
  cost of Track B on the real workload.

**So this needs a ref2va capture as well.** `h3_capture.py` is env-driven and
graph-agnostic, so that is one render with `H3_CAPTURE` set against
`h3_probe_sol_on_refs_api.json`, not a code change. Run both and report both;
a single-workload answer here would be the "measurement stated at a scope wider
than it was taken at" failure the 2026-08-15 postmortem records three times.

## Design decisions to make before writing this

**The reference cannot be the eager Sol implementation at full length.** It is
O(T^2) and refuses past 4 GiB (`comfy_kitchen/backends/eager/sol_attn.py`), and
S=37,826 needs far more. Three options, none free:

  a. chunked dense fp32 attention as the dense baseline, computed per head in
     tiles. Feasible; this is a flash-style loop and it is exact.
  b. subset the heads and/or the query rows. Cheap, and it samples rather than
     measures -- `docs/morton.md`'s mass-concentration test already does this
     (4 heads of 56) and says so in its caveats.
  c. run the eager reference on a contiguous slice of the sequence. **Wrong**,
     and worth stating: slicing changes which keys exist, so it changes the
     routing decision. It measures a different problem.

Pick (a) for the dense baseline and (b) for coverage, and print the sampling
in the output rather than in a comment.

**The metric has to be stated inside the result.** `docs/SOLATTN.md` records
that a cosine and an rtol from different harnesses fail in opposite directions
and cannot be compared. Whatever is chosen, print its name and definition
beside every number.

**The three quantities must be graded at the same tau**, except the one that
must not be. Kernel-against-reference at the same tau cancels the approximation
and measures fidelity; reference-against-dense measures the approximation
alone; kernel-against-dense measures both. Mixing them up is the error two
sessions nearly published on 2026-08-14.

## The control this needs

A decomposition that reports "quantization error is negligible" is a null
result, and a null result is exactly what a broken harness produces. Before
trusting it: **run the same decomposition with the kernel deliberately
degraded** -- e.g. tau driven high so the sparsity term dominates by
construction, and separately a fully-exact tau where the sparsity term must go
to zero. If the split does not move the way those two forced cases demand, the
instrument cannot see what it claims to measure.

    python bench/analyze_sol_error.py --capture <dir> --blocks 0,24,49
    python bench/analyze_sol_error.py --capture <dir> --control
"""

from __future__ import annotations

import argparse
import sys

_NOT_IMPLEMENTED = (
    "analyze_sol_error is scaffolding. See "
    "internal/plan_2026-08-16_sol_fp16_and_triton_retirement.md, gate B0b."
)


def load_capture(path, block):
    """Load one `qkv_L*_S*_b{block}_s{step}.pt` as (q, k, v), each [B,H,S,D].

    TODO(scaffolding): the captures are [1, 56, 37826, 128] BHSD, while
    `comfy_kitchen.sol_attn` takes (B, T, H, 128) BTHD and requires only the
    last dim contiguous. A permute is a view and goes in without a copy -- but
    at 1.52 GiB per file, confirm that rather than assume it.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def dense_reference(q, k, v, scale=None, chunk=4096):
    """Exact fp32 attention, chunked over queries so it fits.

    TODO(scaffolding): option (a) in the module docstring. Must be exact, not
    SDPA in bf16 -- the whole point is a baseline that is not itself an
    approximation. Verify it against `F.scaled_dot_product_attention` on a
    small shape before using it on a large one.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def decompose(q, k, v, tau, heads=None):
    """Return the three quantities, each labelled with what it means.

    sparsity_error       eager Sol (full precision) against dense
    quantization_error   CUDA Sol against eager Sol, SAME tau
    total_error          CUDA Sol against dense

    TODO(scaffolding): the eager arm is the O(T^2) problem. Decide whether it
    runs on a head subset, and print the subset.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def control(q, k, v):
    """Forced cases that must move the split, or the instrument is inert.

    TODO(scaffolding):
      - tau at the dense limit: sparsity_error must go to ~0 and total must
        collapse onto quantization
      - tau very high: sparsity_error must dominate
    A split that does not respond to either is not measuring what it says.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--capture", required=True,
                    help="capture directory, e.g. "
                         "~/Storage/h3_captures/2026-08-15_dense_124f_1344x768/")
    ap.add_argument("--blocks", default="0,24,49")
    ap.add_argument("--tau", type=float, default=1.3)
    ap.add_argument("--heads", type=int, default=0,
                    help="0 = all 56. A subset samples rather than measures, "
                         "and the count is printed in the result either way.")
    ap.add_argument("--control", action="store_true",
                    help="run the forced cases instead of the measurement")
    ap.parse_args()
    raise NotImplementedError(_NOT_IMPLEMENTED)


if __name__ == "__main__":
    sys.exit(main())
