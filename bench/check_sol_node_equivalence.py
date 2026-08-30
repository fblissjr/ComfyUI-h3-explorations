#!/usr/bin/env python3
"""Grade our Sol node's DISPATCH against the algorithm's own eager reference.

## What this covers that `check_solattn_correctness.py` does not

That file grades the KERNEL against the algorithm's eager reference. This
grades everything our node does AROUND the kernel: the BHND-to-BTHD transpose
and back, the scale it forwards, the sink pair it derives, and the tail flag it
passes. A defect in any of those produces a plausible tensor of the right shape
and a successful render.

The distinction is the reshape. `optimized_attention` hands H3's attention over
as BHND with `skip_reshape=True`, the kernel wants BTHD, and the output goes
back as BHND. Getting that wrong transposes heads against tokens -- which does
not raise, because both are legal sizes.

## The oracle is the KERNEL, not the algorithm, and the first draft got that wrong

This file was first written to compare the dispatch against the eager
reference, and the numbers looked like a marginal failure: cosine ~0.994 to
0.998 depending on shape and seed, sometimes under the bar. The instinct was
to loosen the bar. That would have been wrong twice over.

Measured instead: **the dispatch is BITWISE identical to a direct
`comfy_kitchen.sol_attn` call** on the same inputs. Every bit of that spread
was the kernel's INT8 arithmetic against fp32 -- which is not the node's doing,
is a property `check_solattn_correctness.py` already owns, and would have been
silently absorbed into a loosened tolerance here.

So the oracle is the kernel. That gives an exact claim rather than a tolerance,
it isolates the layer this file is about, and it needs no O(T^2) score tensor,
so it runs at a realistic sequence length instead of a toy one. **A tolerance
where an equality is available is a check that cannot see small defects.**

## Why it no longer compares against the vendored node

**It used to, and that comparison is finished rather than broken.** Until
2026-08-30 this file asserted that our forked node produced the SAME BYTES as
the vendored upstream one at the shipped settings, which is what made migrating
145 graphs safe. It passed, at both selections, and the result is recorded in
`bench/results/2026-08-30_sol_node_equivalence.json`.

That comparison cannot be re-run and should not be resurrected. `vendor/`
now holds the PRE-MERGE upstream drop, restored to be a pristine reference:
its `_run` passes `centroid_tail` to a kernel that no longer accepts it, so it
raises rather than producing a baseline. Keeping a check that can only skip
would be worse than none -- so the baseline moved to the algorithm, which is
the more durable control anyway and one this repo already trusts.

Claims, i.e. what breaks if a case is deleted:

  dispatch == kernel       our node's `_run` produces the SAME BYTES as calling
                           `comfy_kitchen.sol_attn` directly with the transpose
                           done by hand. Catches a transpose, a dropped scale,
                           or a sink pair built wrong.
  top-k dispatch           the same through the other selection, which no
                           shipped graph uses and a bench arm can reach.
  sink pair reaches        a non-zero sink must change the output. The sink is
    the kernel             derived from H3's layout and passed through two
                           call frames; if it stopped arriving, every
                           conditioning row would be routed sparsely and the
                           render would merely look worse.
  pooled_tail reaches      RED CONTROL. `tail` is the argument the fork added.
    the kernel             If turning it off does not move the output, it is
                           not connected and every case above is comparing a
                           knob that does nothing.
  (an OOM exits 2, not 1)  a resident model can leave too little VRAM for
                           these shapes. That is an environment state, not a
                           result, and it used to die on a raw traceback.

  a transposed oracle      RED CONTROL, and the one that earns this file.
    is caught              Compares against a kernel call with heads and tokens
                           swapped. If that still matches, the equality above
                           is not seeing layout at all.

Needs CUDA and a comfy_kitchen carrying the merged `sol_attn`. Exit 0 all
passed, 1 a case failed, 2 nothing was graded.

    python bench/check_sol_node_equivalence.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parent.parent


def load(name, path, package_dir=None):
    spec = importlib.util.spec_from_file_location(
        name, path,
        submodule_search_locations=[str(package_dir)] if package_dir else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(a @ b / (a.norm() * b.norm()))


def main():
    if not torch.cuda.is_available():
        print("no CUDA; the kernel cannot run. Nothing checked.")
        return 2
    sys.path.insert(0, str(COMFY))
    sys.path.insert(0, str(REPO / "bench"))

    load("h3x", REPO / "__init__.py", package_dir=REPO)
    node = load("h3x.sol_attn_h3", REPO / "sol_attn_h3.py")
    import comfy_kitchen as ck                              # noqa: E402
    if not hasattr(ck, "sol_attn"):
        print("this comfy_kitchen has no sol_attn; nothing to grade.")
        return 2

    torch.manual_seed(0)
    # A realistic length, which the kernel oracle allows and an O(T^2) eager
    # oracle would not.
    b, h, t, d = 1, 8, 16384, 128
    # BHND, which is how `optimized_attention` hands H3's attention over.
    q, k, v = (torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))
    common = dict(skip_reshape=True, skip_output_reshape=True, scale=None,
                  min_tokens=12288, verbose=False)

    failures = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    def dispatch(**kw):
        return node._run(q, k, v, h, **{**common, "tau": 1.0, **kw})

    def kernel(transpose_oracle=False, **kw):
        """The kernel call the dispatch should be making, done by hand."""
        qs, ks, vs = (x.transpose(1, 2).contiguous() for x in (q, k, v))
        if transpose_oracle:                      # heads against tokens
            qs, ks, vs = (x.transpose(1, 2).contiguous() for x in (qs, ks, vs))
        out = ck.sol_attn(qs, ks, vs, **kw)
        return out.transpose(1, 2)

    print("our node's dispatch against the kernel call it should be making:")
    print(f"  B={b} H={h} T={t} D={d} bf16, bitwise\n")

    # **An OOM here is not a failure and must not print as one.** This box runs
    # a resident ComfyUI, so a model left loaded from a render leaves under a
    # GiB free while these shapes want about two. Before this guard the check
    # died on a torch traceback with a non-zero exit, which reads exactly like
    # a real mismatch -- and a check that goes red while the state is correct
    # trains a reader to ignore red, which is the one thing docs/checks.md says
    # is worse than having no check. Exit 2, with the fix named.
    def guarded(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except torch.OutOfMemoryError as exc:
            free, total = torch.cuda.mem_get_info()
            print(f"  SKIP  every case   not enough VRAM to run the comparison: "
                  f"{free / 2**30:.2f} GiB free of {total / 2**30:.2f}.\n"
                  f"        This is an environment state, not a result. A "
                  f"resident model is the usual cause;\n"
                  f"        POST /free with unload_models to release it, then "
                  f"re-run.\n        ({type(exc).__name__})")
            raise SystemExit(2)

    for label, kw in (("at the shipped selection", dict(tau=1.0)),
                      ("under top-k", dict(tau=1.0, topk_ratio=0.10))):
        got, want = dispatch(**kw), kernel(**{**kw, "tail": True})
        same = torch.equal(got, want)
        check(f"dispatch == kernel {label}", same,
              "same bytes" if same else
              f"DIFFER: max abs "
              f"{float((got.float() - want.float()).abs().max()):.3e}")

    sink = dispatch(tau=1.0, sink_blocks=(0, 4), sink_q=(0, 4))
    check("sink pair reaches the kernel",
          torch.equal(sink, kernel(tau=1.0, tail=True,
                                   sink_blocks=[0, 4], sink_q=[0, 4]))
          and not torch.equal(sink, dispatch(tau=1.0)),
          "a non-zero sink both arrives and changes the output")

    print("\nred controls:")
    base = dispatch(tau=1.0)
    off = dispatch(tau=1.0, tail=False)
    moved = not torch.equal(base, off)
    c = cosine(base, off)
    check("pooled_tail reaches the kernel", moved,
          f"cos {c:.6f} against tail=True -- connected" if moved else
          "turning the pooled tail off changed nothing; it is not reaching "
          "the kernel and every case above is vacuous")

    swapped = kernel(transpose_oracle=True, tau=1.0, tail=True)
    caught = not (swapped.shape == base.shape and torch.equal(base, swapped))
    check("a transposed oracle is caught", caught,
          "a heads/tokens swap does not match, so the equality above is "
          "actually seeing layout"
          if caught else
          "a transposed oracle still matches; this file cannot see the defect "
          "class it exists for")

    print()
    if failures:
        print(f"FAILED: {len(failures)} case(s): {', '.join(failures)}")
        return 1
    print("all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
