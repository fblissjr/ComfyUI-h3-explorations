#!/usr/bin/env python3
"""Does ComfyUI's audio change-of-variable degrade with PDD's block width?

## The question

ComfyUI carries H3's audio latent on the VIDEO schedule and converts velocities
back and forth around every forward (`comfy/ldm/minimax/model.py:530-551`):

    out_audio = (1 - s) * x_a + (1 + (s - 1) * sigma_a) * v_a      s = shift_v/shift_a

That is a change of variable on an INSTANTANEOUS derivative, evaluated at the
step's own sigma. A PDD fused head does not return an instantaneous velocity --
it returns the block's MEAN velocity over `[t_n, t_{n+L}]`. Applying an
instantaneous transform to a block-averaged quantity is exact only as the block
narrows.

Video has no such transform, because video is the reference stream the carry is
defined against. So if this is real, the error is audio-only and grows with L.

## Why this can REFUTE rather than merely agree

It compares:

  applied   = f(mean over block of v, sigma at the block START)   what the code does
  exact     = mean over block of f(v(sigma), sigma)               what it approximates

**The first version of this file asserted the two coincide at L=1 and treated a
non-zero value there as a malformed comparison. That was wrong, and the check
caught it.** A block of L=1 is one grid INTERVAL, not a point, and `sigma_a`
varies across it — so `f` evaluated at the interval's start already differs from
`f` averaged over the interval. L=1 is a FLOOR, not a zero, and the floor is
whatever the sampler's own discretisation already carries.

So the refutable claim is about GROWTH above that floor, not about a zero:
**if the gap does not rise with L, the mechanism in
`docs/research/pdd/audio_under_pdd.md` is wrong and should be withdrawn.**

Monotonicity is checked only across the UNIFORM widths. A non-uniform partition
places its widest block somewhere specific on the curve, so its gap depends on
where the block sits as well as how wide it is, and it is reported without being
required to fall in line.

The velocity field is unknown, so this cannot predict a magnitude. It measures
the GEOMETRY of the transform -- how much `f` varies across a block -- which is
what the block-mean approximation throws away. A field that happened to be
constant would make the gap zero regardless; that is why the smooth-field
assumption is stated rather than hidden, and why the output is a shape and not
a number to quote.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pdd_math import pdd_time_grid  # noqa: E402

SHIFT_V, SHIFT_A, N = 12.0, 3.0, 32
S = SHIFT_V / SHIFT_A          # comfy's `audio_scale`, 4.0


def sigma_a_of(sigma_v):
    """`time_shift_sigma(sigma_v, shift_v, shift_a)`: undo one shift, apply the other."""
    base = sigma_v / (SHIFT_V + sigma_v * (1.0 - SHIFT_V))
    return SHIFT_A * base / (1.0 + (SHIFT_A - 1.0) * base)


def transform(v, sigma_a, x_a=1.0):
    """The line comfy applies to the model's audio output."""
    return (1.0 - S) * x_a + (1.0 + (S - 1.0) * sigma_a) * v


def main() -> int:
    grid_v = (1.0 - pdd_time_grid(SHIFT_V, N)).double()      # sigma_v at each knot
    print("Audio change-of-variable against PDD block width")
    print(f"  s = shift_v/shift_a = {S}   grid {N} points, shift {SHIFT_V}/{SHIFT_A}")
    print()
    print(f"  {'L':>3}{'blocks':>8}{'max |applied-exact|':>22}{'rel to spread of f':>21}")
    print("  " + "-" * 52)
    rows = []
    for L in (1, 2, 4, 8, 16, 28):
        if N % L and L != 28:
            continue
        worst, worst_rel = 0.0, 0.0
        nblocks = 0
        for start in range(0, N, L):
            stop = min(start + L, N)
            if stop - start < 1:
                continue
            nblocks += 1
            # Sample the block densely in sigma_v, mapped to sigma_a.
            sv = torch.linspace(float(grid_v[start]), float(grid_v[stop]), 64,
                                dtype=torch.float64)
            sa = sigma_a_of(sv)
            # A unit velocity field: this measures the transform's own geometry,
            # not the model's. See the docstring -- a constant field would give
            # zero by construction, which is the assumption being made explicit.
            v = torch.ones_like(sa)
            applied = transform(v.mean(), sa[0])            # block-mean v, sigma at START
            exact = transform(v, sa).mean()                 # mean of the transform
            gap = abs(float(applied - exact))
            spread = float(transform(v, sa).max() - transform(v, sa).min())
            worst = max(worst, gap)
            worst_rel = max(worst_rel, gap / spread if spread else 0.0)
        rows.append((L, nblocks, worst, worst_rel))
        print(f"  {L:>3}{nblocks:>8}{worst:>22.6e}{worst_rel:>20.3f}")
    print()
    uniform = [r for r in rows if N % r[0] == 0]
    grew = all(uniform[i][2] <= uniform[i + 1][2] + 1e-15
               for i in range(len(uniform) - 1))
    floor = uniform[0][2]
    print(f"  monotone across uniform widths: {grew}")
    print(f"  floor at L=1: {floor:.4f}   widest uniform (L=16): "
          f"{uniform[-1][2]:.4f}   growth {uniform[-1][2] / floor:.1f}x")
    print("  L=1 is a FLOOR, not zero: one grid interval still spans sigma.")
    if not grew:
        print("\n  REFUTED: the gap does not grow with block width. The audio "
              "mechanism in docs/research/pdd/audio_under_pdd.md does not hold "
              "and should be withdrawn.")
        return 1
    print("\n  CONSISTENT: rising monotonically above the L=1 floor, and "
          "audio-only by construction -- video has no such transform.")
    print("  This is the transform's geometry, NOT a predicted render error.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
