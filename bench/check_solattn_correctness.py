#!/usr/bin/env python3
"""Check Sol-Attn's Triton kernels against the algorithm's own reference.

The Triton kernels in ComfyUI-SolAttn_triton have never had an independent
correctness check. Every judgement about them so far -- ours and upstream's
-- has been "the render looks right", which cannot separate a kernel bug
from a sparsity setting that was always going to soften the output.

kijai/comfy-kitchen's unmerged `sol_attn` branch ships a pure-PyTorch eager
implementation of the same algorithm, written by its author. It is O(T^2)
and slow, but it is a *second implementation*, which is the thing we have
been missing. `bench/_sol_attn_reference.py` vendors it.

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
    ref_tau = reference(q, k, v, tau=args.tau)
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
    ref_wrong = reference(q, k, v, tau=args.tau * 20)
    report(f"tau {args.tau} vs reference at tau {args.tau * 20}",
           cosine(outs["int8"], ref_wrong), args.bar, want_pass=False)

    # How much sparsity is even active here: if the shipped tau routes
    # everything exact at this size, cases 3 and 4 are testing nothing.
    spread = cosine(ref_tau, ref_dense)
    print(f"\n  reference at tau {args.tau} vs its own dense limit: cos {spread:.6f}")
    if spread > 0.99999:
        print("  WARNING: the shipped tau is not sparsifying at this shape, so the "
              "\n           agreement above is the dense case twice. Raise --tokens.")

    print(f"\n{len(failures)} failure(s): {failures}" if failures else "\nall ok")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
