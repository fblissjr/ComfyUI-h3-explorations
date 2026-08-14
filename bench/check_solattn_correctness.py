#!/usr/bin/env python3
"""Check Sol-Attn's Triton and CUDA kernels against the algorithm's own reference.

Neither kernel has ever had an independent correctness check. Every judgement
about them so far -- ours and upstream's -- has been "the render looks right",
which cannot separate a kernel bug from a sparsity setting that was always
going to soften the output.

kijai/comfy-kitchen's unmerged `sol_attn` branch ships a pure-PyTorch eager
implementation of the same algorithm, written by its author. It is O(T^2)
and slow, but it is a *second implementation*, which is the thing we have
been missing. `bench/_sol_attn_reference.py` vendors it.

Two kernels are graded against it: the Triton one in ComfyUI-SolAttn_triton,
and the CUDA one in a local build of the branch (`comfy_kitchen.sol_attn`,
absent from the stock PyPI wheel -- `bench/check_sol_kernel.py` is what tells
those apart). Upstream reports the CUDA path received correctness fixes the
Triton path did not, and reports it as the higher-quality of the two. That is
the author's own report, not something measured here; these cases are what
would let this repo say anything of its own about it.

What each case claims, i.e. what breaks if it is deleted:

  reference == SDPA at tau -inf
      Calibration of the oracle itself. With the routing threshold driven
      to negative infinity every block is exact, so Sol-Attn degenerates to
      dense attention and MUST reproduce scaled_dot_product_attention. If
      this fails, the reference is wrong or wired up wrong, and every other
      number in this file is meaningless.

  triton == reference at tau -inf
      Same degeneration, through the kernel. Isolates plumbing -- layout,
      scale, sink handling -- from the sparsity approximation, because at
      this tau there is no approximation left to blame.

  triton == reference at real tau
      The measurement. Both implementations route the same blocks by the
      same rule, so they should agree closely; the residual is INT8 vs
      full precision, not algorithm. Upstream's own CUDA-vs-eager tests
      assert cos > 0.998, which is the bar used here.

  mismatched tau DISAGREES
      The red control. Compares the kernel at one tau against the reference
      at a very different one. If this still passes, the metric cannot see
      a routing difference and the three cases above prove nothing. A check
      that cannot fail is decoration.

  cuda == reference at tau -inf / at real tau
      The same two cases through `comfy_kitchen.sol_attn`. Separate from the
      Triton arms rather than folded into them: they are different kernels
      with different reported histories, and a single pass/fail over both
      would let one carry the other.

  cuda red control
      The CUDA arm gets its own mismatched-tau control. Reusing Triton's
      would prove nothing about a kernel it never called.

  tail mode (diagnostic, not a case)
      `centroid_tail` shares one pooled tail across a query block instead of
      computing it per row, and upstream puts it at ~5e-4 cosine -- well
      inside this file's 0.998 bar, so a kernel on the opposite mode from the
      reference still passes. The Triton kernel has no such parameter and its
      mode is not documented, so it is MEASURED here (agreement against the
      reference in both modes, better one wins) rather than asserted from
      reading the kernel. Printed, never graded: if the two modes ever land
      far enough apart to matter, that is a finding to act on, not a failure.

Exit codes: 0 all graded cases passed, 1 a case failed, 2 nothing was graded
or an arm was skipped for cause (no CUDA, no Triton, no `sol_attn` in the
installed comfy_kitchen). A skipped arm must not read as a passing one.

Needs CUDA and Triton. Small shapes only -- the reference materialises the
full score matrix and refuses past 4 GiB, so it can never run at H3's real
sequence length. That is a real limit on what this establishes: it checks
the kernel's arithmetic, not its behaviour at 40k tokens.

    python bench/check_solattn_correctness.py
    python bench/check_solattn_correctness.py --tokens 1024 --heads 8
"""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _sol_attn_reference import sol_attn as reference  # noqa: E402

SOLATTN_DIR = HERE.parents[1] / "ComfyUI-SolAttn_triton"


def load_triton_kernels():
    """Import Sol-Attn's kernel modules without running its node registration.

    Two obstacles. The modules use relative imports (`from ._autotune_log
    import ...`), so they must load as a package, not as loose files. And the
    directory is `ComfyUI-SolAttn_triton`, whose hyphen is not a legal module
    name, so it cannot simply go on sys.path.

    Binding a synthetic package with `__path__` pointed at the directory
    solves both, and deliberately never executes the real `__init__.py` --
    that file registers ComfyUI nodes and pulls in comfy_api. We want the
    kernels, not the side effects.
    """
    name = "_solattn_triton_pkg"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(SOLATTN_DIR)]
        sys.modules[name] = pkg
    int8 = importlib.import_module(f"{name}._int8_fwd")
    bf16 = importlib.import_module(f"{name}._tri_fwd")
    return int8.sol_attn_int8, bf16.sol_attn


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
        print("no CUDA; the Triton kernels cannot run. Nothing checked.")
        return 2

    try:
        sol_attn_int8, sol_attn_bf16 = load_triton_kernels()
    except Exception as exc:
        print(f"could not import the Triton kernels from "
              f"ComfyUI-SolAttn_triton: {type(exc).__name__}: {exc}")
        return 2

    torch.manual_seed(args.seed)
    b, t, h, d = 1, args.tokens, args.heads, args.head_dim
    shape = (b, t, h, d)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))

    print(f"Sol-Attn Triton vs the algorithm's eager reference")
    print(f"  B={b} T={t} H={h} D={d} bf16, tau={args.tau}, bar cos>{args.bar}\n")

    failures = []
    skipped = []

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

    # 2. Plumbing, with the approximation switched off.
    for label, fn, kw in (("bf16", sol_attn_bf16, {}),
                          ("int8", sol_attn_int8, {"int8_pv": True})):
        out = fn(q, k, v, tau=DENSE_TAU, **kw)
        report(f"triton {label} == reference at tau -inf",
               cosine(out, ref_dense), args.bar)

    # 3. The measurement, at the tau we actually ship.
    #
    #    Against the PER-ROW reference, because that is the mode the Triton
    #    kernel is on -- measured in case 5, not assumed. This mattered: the
    #    reference gained `centroid_tail` (default True) when it was
    #    re-vendored on 2026-08-14, which silently made these arms cross-mode.
    #    They still passed, because the two modes differ by cos 0.9988 and the
    #    bar is 0.998 -- looser than the discrepancy it was meant to catch.
    #    A bar that cannot see a whole-algorithm difference is not a bar.
    ref_tau = reference(q, k, v, tau=args.tau, centroid_tail=False)
    outs = {}
    for label, fn, kw in (("bf16", sol_attn_bf16, {}),
                          ("int8", sol_attn_int8, {"int8_pv": True}),
                          ("int8 (pv off)", sol_attn_int8, {"int8_pv": False})):
        outs[label] = fn(q, k, v, tau=args.tau, **kw)
        report(f"triton {label} == reference at tau {args.tau}",
               cosine(outs[label], ref_tau), args.bar)

    # 4. Red control. A far larger tau routes far fewer blocks; if the
    #    comparison cannot tell that apart from the real thing, nothing above
    #    is evidence.
    ref_wrong = reference(q, k, v, tau=args.tau * 20, centroid_tail=False)
    report(f"tau {args.tau} vs reference at tau {args.tau * 20}",
           cosine(outs["int8"], ref_wrong), args.bar, want_pass=False)

    # 5. Which tail mode is each kernel on? Not graded -- see the docstring.
    #    Neither kernel documents it and Triton has no such parameter, so it
    #    is measured. Every arm above and below grades against the matching
    #    mode; without this they are silently cross-mode.
    ref_centroid = reference(q, k, v, tau=args.tau, centroid_tail=True)
    mode_gap = cosine(ref_tau, ref_centroid)

    def tail_mode(out, label):
        c_centroid, c_perrow = cosine(out, ref_centroid), cosine(out, ref_tau)
        picked = c_centroid >= c_perrow
        print(f"  {label:<12s} centroid_tail={'True ' if picked else 'False'} "
              f"(centroid {c_centroid:.6f} vs per-row {c_perrow:.6f})")
        return picked

    print(f"\n  tail mode measured (the modes differ by cos {mode_gap:.6f}, "
          f"and the bar is {args.bar}):")
    tail_mode(outs["bf16"], "triton bf16")

    # 6. The CUDA kernel, graded separately against the same oracle, in its
    #    own tail mode. Folding it into the Triton arms would let one kernel's
    #    result carry the other's.
    cuda_sol, why = load_cuda_kernel()
    print()
    if cuda_sol is None:
        print(f"  SKIP  CUDA arm: {why}")
        skipped.append("cuda")
    else:
        cuda_tau = cuda_sol(q, k, v, tau=args.tau)
        cuda_centroid = tail_mode(cuda_tau, "cuda")
        cuda_ref = ref_centroid if cuda_centroid else ref_tau
        cuda_wrong = (reference(q, k, v, tau=args.tau * 20, centroid_tail=True)
                      if cuda_centroid else ref_wrong)
        print()
        report("cuda == reference at tau -inf",
               cosine(cuda_sol(q, k, v, tau=DENSE_TAU), ref_dense), args.bar)
        report(f"cuda == reference at tau {args.tau}",
               cosine(cuda_tau, cuda_ref), args.bar)
        report(f"cuda tau {args.tau} vs reference at tau {args.tau * 20}",
               cosine(cuda_tau, cuda_wrong), args.bar, want_pass=False)

        # Not a case: the two kernels' distance from the algorithm, each in
        # its own tail mode. Comparing them in a single mode would score one
        # of them against an algorithm it is not implementing, which is worth
        # about 1.2e-3 of apparent quality -- larger than the gap being
        # measured, so it would invent a winner.
        # NOT the same quantity as the cases above, and the distinction is
        # the whole point. Every graded case compares a kernel against the
        # reference AT THE SAME TAU, so the block-sparse approximation is on
        # both sides and cancels: they measure "does this kernel implement Sol
        # faithfully". This line measures "how far is Sol's output from exact
        # attention" -- kernel error and approximation error together.
        #
        # Only this second number is comparable to an accuracy figure from a
        # dense kernel (e.g. sage's mean_rtol against SDPA). Quoting the
        # fidelity number against one of those compares different referents
        # and flatters Sol by roughly the size of its own approximation, which
        # is the error two sessions nearly published on 2026-08-14.
        nblk = (t + 63) // 64
        print(f"\n  vs DENSE attention -- total error, approximation included."
              f"\n  ON SYNTHETIC INPUT AT T={t} ({nblk} blocks of 64). "
              f"DOUBLY PESSIMISTIC, DO NOT QUOTE:")
        print(f"    cuda   {cosine(cuda_tau, dense):.6f}")
        print(f"    triton {cosine(outs['int8'], dense):.6f}")
        print(f"  Two reasons this is a floor, not an estimate. (1) torch.randn "
              f"gives a\n  near-uniform softmax, so attention mass does not "
              f"concentrate and there is\n  nothing for a block router to find "
              f"-- the premise of the method is absent.\n  (2) {nblk} blocks is "
              f"a different regime from production's ~1,626 at 345\n  frames, "
              f"not a small version of it. Re-run on captured activations at\n"
              f"  production S before this number means anything. This IS the "
              f"quantity\n  comparable to a dense kernel's accuracy figure -- "
              f"but only once measured\n  somewhere the method's premise holds.")

        print(f"\n  distance from the algorithm, each in its own tail mode:")
        print(f"    cuda        {cosine(cuda_tau, cuda_ref):.6f}")
        print(f"    triton int8 {cosine(outs['int8'], ref_tau):.6f}")
        print(f"    triton bf16 {cosine(outs['bf16'], ref_tau):.6f}")
        print("  Same algorithm, so this is kernel arithmetic (INT8 vs full "
              "precision), not\n  a quality ranking of a render. It cannot see "
              "behaviour at 40k tokens.")

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
    if skipped:
        # Exit 2, not 0. An arm that did not run must not read as one that
        # passed -- docs/checks.md gap 5.
        print(f"\nINCOMPLETE: {len(skipped)} arm(s) skipped: {', '.join(skipped)}")
        return 2
    print("\nall ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
