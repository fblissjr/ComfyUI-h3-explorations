#!/usr/bin/env python3
"""Check Sol-Attn's CUDA kernel against the algorithm's own eager reference.

The kernel had no independent correctness check before this file. Every
judgement about it -- ours and upstream's -- was "the render looks right",
which cannot separate a kernel bug from a sparsity setting that was always
going to soften the output.

kijai/comfy-kitchen's unmerged `sol_attn` branch ships a pure-PyTorch eager
implementation of the same algorithm, written by its author. It is O(T^2) and
slow, but it is a *second implementation*, which is the thing we were missing.
`bench/_sol_attn_reference.py` vendors it.

**Scope narrowed 2026-08-16: this grades the CUDA kernel only.** It used to
grade the Triton kernel too, and the Triton arms were removed with the pack
(see `internal/plan_2026-08-16_sol_fp16_and_triton_retirement.md`, Track A1).
Two reasons. The Triton arms graded a kernel no graph has wired since
2026-08-14, so a regression there could not reach a render. And the CUDA arm
never needed them: the script loaded Triton first and returned 2 on failure,
so an absent pack silently disabled the only correctness check on the kernel
that does run. That coupling was an accident of control flow, not of method.

What each case claims, i.e. what breaks if it is deleted:

  reference == SDPA at tau -inf
      Calibration of the oracle itself. With the routing threshold driven to
      negative infinity every block is exact, so Sol-Attn degenerates to dense
      attention and MUST reproduce scaled_dot_product_attention. If this
      fails, the reference is wrong or wired up wrong and every other number
      in this file is meaningless.

  cuda == reference at tau -inf
      Same degeneration, through the kernel. Isolates plumbing -- layout,
      scale, sink handling -- from the sparsity approximation, because at this
      tau there is no approximation left to blame.

  cuda == reference at real tau
      The measurement. Both implementations route the same blocks by the same
      rule, so they should agree closely; the residual is INT8 against full
      precision, not algorithm. Upstream's own CUDA-vs-eager tests assert
      cos > 0.998, which is the bar used here.

  mismatched tau DISAGREES
      The red control. Compares the kernel at one tau against the reference at
      a very different one. If this passes, the metric cannot see a routing
      difference and every case above proves nothing. A check that cannot fail
      is decoration.

  tail mode (diagnostic, not a case)
      `centroid_tail` shares one pooled tail across a query block instead of
      computing it per row, and it is worth ~5e-4 cosine -- well inside the
      0.998 bar, so a kernel graded against the wrong mode still passes. So it
      is MEASURED (agreement against the reference in both modes, better one
      wins) and the graded cases use the matching oracle. Grading cross-mode
      was a live defect here until 2026-08-14; see `docs/checks.md`.

Exit codes: 0 all graded cases passed, 1 a case failed, 2 nothing was graded
(no CUDA, or the installed comfy_kitchen has no `sol_attn` -- the expected
state on a machine that has not built the fork). A skipped run must not read
as a passing one.

Needs CUDA and a fork build of comfy_kitchen. Small shapes only -- the
reference materialises the full score matrix and refuses past 4 GiB, so it can
never run at H3's real sequence length. That is a real limit on what this
establishes: it checks the kernel's arithmetic, not its behaviour at 40k
tokens. The run that would close that gap is gate B0b in the plan above.

    python bench/check_solattn_correctness.py
    python bench/check_solattn_correctness.py --tokens 1024 --heads 8
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _sol_attn_reference import sol_attn as reference  # noqa: E402

def load_cuda_kernel():
    """`comfy_kitchen.sol_attn`, or (None, why) if this build has no such thing.

    Present only in a local build of kijai/comfy-kitchen's `sol_attn` branch.
    The stock PyPI wheel is version-identical and has no `sol_attn`, which is
    the whole reason `bench/check_sol_kernel.py` exists.
    """
    try:
        ck = importlib.import_module("comfy_kitchen")
    except Exception as exc:
        return None, f"comfy_kitchen is not importable: {exc}"
    if not hasattr(ck, "sol_attn"):
        return None, ("the installed comfy_kitchen has no sol_attn (stock PyPI "
                      "wheel); build kijai's `sol_attn` branch to grade it")
    cuda = importlib.import_module("comfy_kitchen.backends.cuda")
    if not hasattr(cuda, "sol_attn"):
        return None, ("only the eager reference is present, which is the oracle "
                      "itself -- grading it against itself proves nothing")
    return ck.sol_attn, None

# Driven far enough negative that every block clears the routing threshold.
# Not -inf: the threshold is tau * sqrt(...), and -inf would make the
# comparison nan rather than always-true.
DENSE_TAU = -1e9


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128,
                    help="128 is what H3 uses and what the CUDA backend requires")
    ap.add_argument("--tau", type=float, default=1.3,
                    help="the shipped value, from workflows/h3_config.py")
    ap.add_argument("--bar", type=float, default=0.998,
                    help="upstream's own CUDA-vs-eager tolerance")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA; the kernel cannot run. Nothing checked.")
        return 2

    # Exit 2, not 1. An absent kernel is the EXPECTED state on a machine that
    # has not built kijai's fork -- the same third case check_sol_kernel.py
    # gates on. A missing dependency is not a failing kernel.
    cuda_sol, why = load_cuda_kernel()
    if cuda_sol is None:
        print(f"no CUDA Sol-Attn kernel to grade: {why}")
        return 2

    torch.manual_seed(args.seed)
    b, t, h, d = 1, args.tokens, args.heads, args.head_dim
    shape = (b, t, h, d)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))

    print(f"Sol-Attn CUDA kernel vs the algorithm's eager reference")
    print(f"  B={b} T={t} H={h} D={d} bf16, tau={args.tau}, bar cos>{args.bar}\n")

    failures = []

    def report(name, got, bar, want_pass=True):
        ok = (got > bar) if want_pass else (got < bar)
        verb = f"cos {got:.6f}"
        if want_pass:
            line = f"  {'ok  ' if ok else 'FAIL'}  {name:<44s} {verb} (> {bar})"
        else:
            line = f"  {'ok  ' if ok else 'FAIL'}  {name:<44s} {verb} (< {bar}, must differ)"
        print(line)
        if not ok:
            failures.append(name)

    # 1. Calibrate the oracle against something independent of Sol-Attn.
    dense = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    ).permute(0, 2, 1, 3)
    ref_dense = reference(q, k, v, tau=DENSE_TAU)
    report("reference == SDPA at tau -inf", cosine(ref_dense, dense), 0.9999)

    # 2. Plumbing, with the approximation switched off. At this tau every
    #    block is exact, so layout / scale / sink handling are isolated from
    #    the sparsity approximation -- there is none left to blame.
    report("cuda == reference at tau -inf",
           cosine(cuda_sol(q, k, v, tau=DENSE_TAU), ref_dense), args.bar)

    # 3. Which tail mode is the kernel on? MEASURED, not assumed, and it runs
    #    before the graded cases because it decides which oracle they use.
    #    `centroid_tail` is worth ~5e-4 cosine -- well inside the 0.998 bar --
    #    so a kernel graded against the wrong mode passes anyway. That defect
    #    was live in this file until 2026-08-14; see docs/checks.md.
    ref_tau = reference(q, k, v, tau=args.tau, centroid_tail=False)
    ref_centroid = reference(q, k, v, tau=args.tau, centroid_tail=True)
    mode_gap = cosine(ref_tau, ref_centroid)
    cuda_tau = cuda_sol(q, k, v, tau=args.tau)

    c_centroid = cosine(cuda_tau, ref_centroid)
    c_perrow = cosine(cuda_tau, ref_tau)
    cuda_centroid = c_centroid >= c_perrow
    cuda_ref = ref_centroid if cuda_centroid else ref_tau
    print(f"\n  tail mode measured (the modes differ by cos {mode_gap:.6f}, "
          f"and the bar is {args.bar}):")
    print(f"    cuda  centroid_tail={'True ' if cuda_centroid else 'False'} "
          f"(centroid {c_centroid:.6f} vs per-row {c_perrow:.6f})\n")

    # 4. The measurement, at the tau we ship, against the matching mode.
    report(f"cuda == reference at tau {args.tau}",
           cosine(cuda_tau, cuda_ref), args.bar)

    # 5. Red control. A far larger tau routes far fewer blocks; if the metric
    #    cannot tell that apart from the real thing, cases 2 and 4 prove
    #    nothing. Graded in the kernel's own tail mode for the same reason as
    #    case 4 -- a cross-mode control would fail for the wrong reason and
    #    still look like it worked.
    cuda_wrong = reference(q, k, v, tau=args.tau * 20, centroid_tail=cuda_centroid)
    report(f"cuda tau {args.tau} vs reference at tau {args.tau * 20}",
           cosine(cuda_tau, cuda_wrong), args.bar, want_pass=False)

    # Not a case: distance from DENSE attention, i.e. kernel error and the
    # block-sparse approximation together.
    #
    # The distinction from every graded case above is the whole point. Those
    # compare kernel against reference AT THE SAME TAU, so the approximation
    # sits on both sides and cancels: they measure "does this kernel implement
    # Sol faithfully". This measures "how far is Sol's output from exact
    # attention". Only the second is comparable to a dense kernel's accuracy
    # figure, and quoting the first against one flatters Sol by the size of its
    # own approximation -- the error two sessions nearly published on
    # 2026-08-14.
    nblk = (t + 63) // 64
    print(f"\n  vs DENSE attention -- total error, approximation included."
          f"\n  ON SYNTHETIC INPUT AT T={t} ({nblk} blocks of 64). "
          f"DOUBLY PESSIMISTIC, DO NOT QUOTE:")
    print(f"    cuda   {cosine(cuda_tau, dense):.6f}")
    print(f"  Two reasons this is a floor, not an estimate. (1) torch.randn "
          f"gives a\n  near-uniform softmax, so attention mass does not "
          f"concentrate and there is\n  nothing for a block router to find "
          f"-- the premise of the method is absent.\n  (2) {nblk} blocks is "
          f"a different regime from production's ~1,626 at 345\n  frames, "
          f"not a small version of it. Re-run on captured activations at\n"
          f"  production S before this number means anything. This IS the "
          f"quantity\n  comparable to a dense kernel's accuracy figure -- "
          f"but only once measured\n  somewhere the method's premise holds. "
          f"That run is gate B0b in\n  internal/plan_2026-08-16_sol_fp16_and_triton_retirement.md.")

    print(f"\n  distance from the algorithm, in the kernel's own tail mode: "
          f"{cosine(cuda_tau, cuda_ref):.6f}")
    print("  Same algorithm on both sides, so this is kernel arithmetic (INT8 "
          "vs full\n  precision), not a quality ranking of a render. It cannot "
          "see behaviour at\n  40k tokens.")


    # How much sparsity is even active here: if the shipped tau routes
    # everything exact at this size, cases 3 and 4 are testing nothing.
    spread = cosine(ref_tau, ref_dense)
    print(f"\n  reference at tau {args.tau} vs its own dense limit: cos {spread:.6f}")
    if spread > 0.99999:
        print("  WARNING: the shipped tau is not sparsifying at this shape, so the "
              "\n           agreement above is the dense case twice. Raise --tokens.")

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}")
        return 1
    print("\nall ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
