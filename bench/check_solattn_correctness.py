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

  tail=False, topk + tail=False, block_len, coarse_gate
      **Added 2026-08-30, and every one of them was unreachable before
      comfy-kitchen#117 merged.** That is the class no existing case could
      have covered: until the merge there was no argument to pass. Each is
      reachable from a node this repo now ships, so a silent disagreement
      reaches a render. The tail=False cases are graded at a looser bar for
      upstream's stated reason -- with no pooled term, every int8-vs-fp32
      routing flip shows in full.

  dead rows do not reach the live output
      An exact-equality claim under `block_len`, not a cosine one. Perturbing
      the padding must not move a live row; if it does, a dead row is leaking
      into a key or a block mean, which is the failure that would corrupt VSA
      cube tiling and nothing else here would catch.

  coarse_gate changes the output / cuda tail=True vs reference tail=False
      Red controls for the two new branches. The first says the metric can see
      the coarse term at all; the second says it can see the pooled tail --
      the paper's actual contribution. If either passes, the cases they guard
      are decoration.

  tail mode (RETIRED 2026-08-30)
      `centroid_tail` used to be a toggle, and grading against the wrong mode
      passed anyway because the modes differ by less than the bar -- so this
      file MEASURED which mode the kernel was in and picked the matching
      oracle. The merged kernel evaluates the tail at the query block's
      centroid unconditionally and the argument is gone from both sides, so
      there is no mode left to discover. Recorded rather than deleted because
      the defect it fixed (grading cross-mode, live here until 2026-08-14) is
      in `docs/checks.md`.

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
    ap.add_argument("--notail-bar", type=float, default=0.99,
                    help="tolerance for tail=False cases. Looser on purpose, "
                         "and it is upstream's own number: without the pooled "
                         "term every int8-vs-fp32 routing flip shows in full "
                         "(tests/test_sol_attn.py::test_no_tail_matches_eager)")
    ap.add_argument("--topk", type=float, default=0.2,
                    help="keep-fraction for the SLA selection cases")
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

    # 3. The measurement, at the tau we ship. Both sides route the same blocks
    #    by the same rule, so the residual is INT8 against full precision, not
    #    algorithm.
    #
    #    **This replaced a tail-mode probe on 2026-08-30.** That probe ran the
    #    reference twice, at `centroid_tail` True and False, to DISCOVER which
    #    form the kernel implemented, because grading against the wrong one
    #    passed anyway (the modes differ by less than the bar). The merged
    #    kernel removed the argument -- it always evaluates the tail at the
    #    query block's centroid -- so there is no longer a mode to discover,
    #    and the reference has no such parameter to run twice.
    cuda_tau = cuda_sol(q, k, v, tau=args.tau)
    ref_tau = reference(q, k, v, tau=args.tau)
    report(f"cuda == reference at tau {args.tau}",
           cosine(cuda_tau, ref_tau), args.bar)

    # 4-7 grade paths that were DEAD before comfy-kitchen#117 and are live
    # after it, which is the one class no existing case can cover by
    # construction: until the merge there was nothing to call. Each is
    # reachable from `MiniMaxH3SolAttn` or from the VSA node, so a silent
    # disagreement here reaches a render.

    # 4. `tail=False` -- upstream's tests call it "the SLA / VSA fine stage".
    #    Softmax over the routed blocks only, no pooled correction.
    #
    #    Graded at a LOOSER bar, and the reason is upstream's own: "without the
    #    tail every int8-vs-fp32 block-selection flip shows in full"
    #    (`tests/test_sol_attn.py::test_no_tail_matches_eager`, which asserts
    #    cos > 0.99). The pooled term normally absorbs a block the two
    #    implementations disagree about routing; remove it and the whole block
    #    is gone from one side. Holding this to the tail=True bar would be
    #    grading a different quantity against the same number.
    cuda_notail = cuda_sol(q, k, v, tau=args.tau, tail=False)
    ref_notail = reference(q, k, v, tau=args.tau, tail=False)
    report(f"cuda == reference at tau {args.tau}, tail=False",
           cosine(cuda_notail, ref_notail), args.notail_bar)

    # 5. Top-k selection with no tail. Not an arbitrary pair: it is what SLA
    #    IS, and it is the exact call `MiniMaxH3SolAttn` makes for its
    #    "top-k (SLA)" selection once `tail` is off.
    cuda_sla = cuda_sol(q, k, v, topk_ratio=args.topk, tail=False)
    ref_sla = reference(q, k, v, topk_ratio=args.topk, tail=False)
    report(f"cuda == reference at topk {args.topk}, tail=False",
           cosine(cuda_sla, ref_sla), args.notail_bar)

    # 6. `block_len`: zero-padded tiles, where only the first N rows of each
    #    64-row block are live. Nothing in H3's own packing needs it -- the
    #    packed sequence is contiguous -- but VSA's cube tiling does, one cube
    #    per block, and that is the only way this repo will ever call it.
    #    Dead rows must be excluded from keys, from values and from the pooled
    #    means; the second assertion is upstream's own test for exactly that.
    n_blocks = (t + 63) // 64
    gen = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    block_len = torch.randint(1, 65, (n_blocks,), device="cuda",
                              generator=gen).to(torch.int32)
    lengths = block_len.clone()
    if t % 64:
        lengths[-1] = min(int(lengths[-1]), t % 64)
    live = (torch.arange(t, device="cuda") % 64) < lengths.repeat_interleave(64)[:t]
    cuda_pad = cuda_sol(q, k, v, tau=args.tau, block_len=block_len)
    ref_pad = reference(q, k, v, tau=args.tau, block_len=block_len)
    report(f"cuda == reference at tau {args.tau}, block_len",
           cosine(cuda_pad[:, live], ref_pad[:, live]), args.bar)

    #    Dead rows are really dead: perturb them and nothing live may move.
    #    An exact-equality claim, not a cosine one -- if a dead row leaks into
    #    a key or a block mean, the live output changes and this goes red.
    qd, kd, vd = (x.clone() for x in (q, k, v))
    for x in (qd, kd, vd):
        x[:, ~live] = torch.randn_like(x[:, ~live]) * 3
    leaked = not torch.equal(
        cuda_sol(qd, kd, vd, tau=args.tau, block_len=block_len)[:, live],
        cuda_pad[:, live])
    report("dead rows do not reach the live output",
           0.0 if leaked else 1.0, 0.5)

    # 7. `coarse_gate`: VSA's gated coarse branch, `gate * softmax(q_mean
    #    k_mean^T) v_mean` added per block. Graded with every query block in
    #    sink_q so selection cannot blur what the gate did, which is upstream's
    #    own construction, and at a tight bar because with routing pinned the
    #    only thing left between the two sides is the coarse arithmetic.
    gate = torch.randn(q.shape, device="cuda", dtype=torch.bfloat16) * 0.5
    coarse_kw = dict(tau=args.tau, tail=False, block_len=block_len,
                     sink_q=[0, n_blocks], coarse_gate=gate)
    cuda_coarse = cuda_sol(q, k, v, **coarse_kw)
    ref_coarse = reference(q, k, v, **coarse_kw)
    report("cuda == reference with coarse_gate (VSA branch)",
           cosine(cuda_coarse[:, live], ref_coarse[:, live]), 0.999)

    # 8. Red control for the gate: with every block routed, the coarse term is
    #    the ONLY difference from masked dense attention. If this passes, cases
    #    7 and the whole VSA path are being graded by a metric that cannot see
    #    the branch they exist to add.
    no_gate = cuda_sol(q, k, v, tau=args.tau, tail=False, block_len=block_len,
                       sink_q=[0, n_blocks])
    report("coarse_gate changes the output",
           cosine(cuda_coarse[:, live], no_gate[:, live]), 0.999, want_pass=False)

    # 9. Red control for the tail. The pooled term is what separates Sol-Attn
    #    from plain block-sparse attention -- the paper's contribution, not a
    #    side knob -- so a metric that cannot tell the two apart cannot
    #    adjudicate any case above.
    report(f"cuda tail=True vs reference tail=False",
           cosine(cuda_tau, ref_notail), args.bar, want_pass=False)

    # 10. Red control. A far larger tau routes far fewer blocks; if the metric
    #    cannot tell that apart from the real thing, every graded case proves
    #    nothing.
    cuda_wrong = reference(q, k, v, tau=args.tau * 20)
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
          f"a different regime from production's ~1,700 --\n  DERIVED from the "
          f"token grid, not measured -- and not a small version of\n  it. "
          f"Re-run on captured activations at\n"
          f"  production S before this number means anything. This IS the "
          f"quantity\n  comparable to a dense kernel's accuracy figure -- "
          f"but only once measured\n  somewhere the method's premise holds. "
          f"That run is gate B0b in\n  internal/plan_2026-08-16_sol_fp16_and_triton_retirement.md.")

    print(f"\n  distance from the algorithm, at the shipped tail: "
          f"{cosine(cuda_tau, ref_tau):.6f}")
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
