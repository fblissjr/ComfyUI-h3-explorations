#!/usr/bin/env python3
"""SCAFFOLDING -- NOT IMPLEMENTED. Gate B0a: profile the CUDA Sol stages.

**Nothing here runs.** `main()` raises. See
`internal/plan_2026-08-16_sol_fp16_and_triton_retirement.md`, gate B0a.

## The question

Is `sol_attn_exact.cu` MMA-issue-bound or staging-bound?

That single fact decides how much of Track B's cost estimate to believe. The
estimate is arithmetic over a measured instruction rate: QK is 32 int8 MMA per
warp per key block and PV is 32, so a 16-bit PV at 64 MMA and half the issue
rate predicts **2.5x on the exact branch** with f32 accumulate. `bench/mma_rate.cu`
measured the rates on this box; the 2.5x is inference on top of them.

**It is an upper bound on the MMA term alone.** The same change doubles the V
tile (`LDV` 64 to 128 bytes, smem 32768 B to 49152 B per block), so if the
kernel is bound by cp.async staging rather than by MMA issue, the real figure
sits somewhere else for a reason the arithmetic cannot see. Nothing has ever
profiled this path per stage.

## What to collect

The pipeline is four kernels (`sage_attention/sol_attn.cu:19-24`): `preprocess`,
`vtranspose`, `route`, `exact`. Per kernel:

  - device time, and its share of the four
  - achieved occupancy, and the limiter (registers, smem, or blocks)
  - `smsp__inst_executed_pipe_tensor` against elapsed cycles -- the MMA
    issue-rate question, directly
  - dram throughput and l2 hit rate -- the staging question
  - **routed density**, which nothing here measures today. `blk_cnt` is
    written per (b, h, query block) by `sol_attn_route.cu`; getting at it means
    either the kernel exposing it or the probe of Track A2.

## Two hazards specific to profiling this

**`ncu` serializes kernels and needs the card to itself.** A concurrent render
both wrecks the numbers and is wrecked by them. Free the GPU first (`POST
/free` with `unload_models`, or stop ComfyUI), and note that `docs/checks.md`
already warns the CUDA checks OOM against a live render and read as regressions.

**Sol runs only inside the sigma window.** At the shipped `0.2 / 0.9`, 16
steps, `shift_video=12.0`, that is 11 of 16 steps sparse -- so a profile that
captures the whole render mixes 5 dense sage steps into the average. Either
filter by kernel name or profile a bounded range of launches, and say which.

## What would change the plan

- **MMA-bound** -> the 2.5x estimate roughly holds, and Track B is expensive.
  The f16-accumulate variant (1.5x by the same arithmetic) becomes the
  interesting one, with sage's `_inst_buf` kernels as the precedent for making
  f16 accumulation safe over a long key list.
- **Staging-bound** -> the 16-bit PV costs less than the arithmetic says on
  the MMA side and more on the memory side, since the V tile doubles. Either
  way the estimate is wrong and needs replacing with a measurement.
- **Exact branch is a small share of the four kernels** -> Track B matters
  less than assumed, and routing is where to look instead.

    python bench/profile_sol_stages.py --workflow h3_probe_sol_on_api.json
    python bench/profile_sol_stages.py --workflow h3_probe_sol_on_refs_api.json
"""

from __future__ import annotations

import argparse
import sys

_NOT_IMPLEMENTED = (
    "profile_sol_stages is scaffolding. See "
    "internal/plan_2026-08-16_sol_fp16_and_triton_retirement.md, gate B0a."
)

# The four kernels, in launch order. Names must match the cubin symbols, which
# are `sol_exact_kernel`, `sol_route_kernel` / `sol_route_perrow_kernel`, the
# five `prep_*` kernels and `vquant_transpose`. Verify against `ncu` output
# rather than trusting this list -- they are in anonymous namespaces and the
# mangled names are what appear.
STAGES = ("preprocess", "vtranspose", "route", "exact")


def ensure_card_is_free():
    """Refuse to profile against a live render.

    TODO(scaffolding): check for a running ComfyUI with work queued, and for
    non-trivial GPU memory in use. Refusing is the right behaviour -- a profile
    taken beside a render is not a slow profile, it is a wrong one, and it
    looks like a finding.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def run_ncu(workflow, launch_skip=0, launch_count=0):
    """Submit the graph under `ncu` and return the parsed per-kernel rows.

    TODO(scaffolding): decide how to bound the capture. Profiling every launch
    of a full-length render is not viable -- 50 DiT blocks x 11 sparse steps is
    550 exact-kernel launches. Bound it, and print what was skipped: a silent
    cap reads as "covered everything".
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def summarize(rows):
    """Per-stage table plus the bound verdict.

    Must print the verdict as a claim with its evidence, not a label: which
    metric decided it and what the threshold was. "Staging-bound" with no
    number behind it is the kind of sentence that gets quoted.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--workflow", default="h3_probe_sol_on_api.json",
                    help="t2v by default. Run the refs graph too -- reference "
                         "rows are pinned exact, so they change the exact "
                         "branch's share, which is the quantity being measured.")
    ap.add_argument("--launch-skip", type=int, default=0)
    ap.add_argument("--launch-count", type=int, default=0,
                    help="0 = all. Any cap is printed in the result.")
    ap.parse_args()
    raise NotImplementedError(_NOT_IMPLEMENTED)


if __name__ == "__main__":
    sys.exit(main())
