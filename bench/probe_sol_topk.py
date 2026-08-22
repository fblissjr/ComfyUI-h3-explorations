#!/usr/bin/env python3
"""Does the CUDA kernel's `topk_ratio` agree with the algorithm it implements?

`topk_ratio` arrived with comfy-kitchen `0.2.31+sol.23d1a66` (kijai's PR 117,
2026-08-22) and selects exact key blocks by per-query-block top-k instead of
the tau threshold. **No graph here ships it**, so nothing else in this repo
would ever execute the path; this is the probe that says whether it works at
all, and what it costs against its own reference.

Call-level, against `bench/_sol_attn_reference.py` -- the only controlled
comparison this repo can make about a numerical knob, because a rendered clip
cannot A/B one (CLAUDE.md). Small shapes only: the oracle is O(T^2).

Not a check and not in the check suite. It asserts nothing and has no exit
code to grade, because there is no threshold anyone has agreed to hold this
path to while nothing renders under it.

READ THE CAVEAT BEFORE QUOTING THE NUMBERS. `torch.randn` gives a near-uniform
softmax, so attention mass does not concentrate and there is nothing for a
block router to find -- the premise of the method is absent, exactly as
`check_solattn_correctness.py` says of its own dense comparison. The number
that means something here is the RATIO between the two selections at the same
shape, not either one alone.
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _sol_attn_reference import sol_attn as reference  # noqa: E402

from comfy_kitchen.backends import cuda as ck  # noqa: E402

B, T, H, D = 1, 1024, 8, 128
RATIOS = (0.10, 0.15, 0.50)


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def main() -> int:
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))
    n = (T + 63) // 64

    print(f"B={B} T={T} H={H} D={D}, {n} blocks of 64\n")
    print("kernel vs the vendored oracle, same selection on both sides:")
    for r in RATIOS:
        kk = max(1, min(n - 1, round(r * n)))
        got = ck.sol_attn(q, k, v, tau=1.0, topk_ratio=r)
        want = reference(q, k, v, tau=1.0, topk_ratio=r)
        print(f"  topk_ratio {r:<5} (keeps {kk} of {n} blocks)  cos {cos(got, want):.6f}")

    tau_cuda = ck.sol_attn(q, k, v, tau=1.0)
    print(f"  tau 1.0  -- the shipped selection, same shape   cos "
          f"{cos(tau_cuda, reference(q, k, v, tau=1.0)):.6f}")

    # Without this the run above cannot tell "agrees closely" from "the
    # argument was ignored": an ignored topk_ratio makes every arm the tau
    # arm, and every cos above would be the tau arm's own.
    print("\ncontrol -- topk_ratio must not be a no-op:")
    t_lo = ck.sol_attn(q, k, v, tau=1.0, topk_ratio=RATIOS[0])
    t_hi = ck.sol_attn(q, k, v, tau=1.0, topk_ratio=RATIOS[-1])
    print(f"  tau vs topk {RATIOS[0]}   cos {cos(tau_cuda, t_lo):.6f}  (must differ)")
    print(f"  topk {RATIOS[0]} vs {RATIOS[-1]}  cos {cos(t_lo, t_hi):.6f}  (must differ)")
    print("\nEqual cos for two ratios is not a defect on its own: `kk` is "
          f"rounded to whole blocks, and at {n} blocks nearby ratios collapse "
          "onto the same k.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
