"""Schedule arithmetic shared by the PDD converter and the PDD node.

One copy, because the two consumers must agree exactly or the node selects a
head the converter did not fuse and nothing errors. `bench/convert_pdd_lora.py`
bakes `fuse_heads` output into a file; `pdd_lora.py` reads that file and uses
`block_bounds` to decide which entry a sampling step wants. A drift between
them is a silent wrong-head, which is the failure mode this module exists to
make impossible.

No ComfyUI imports here on purpose: the converter runs without a server, and
this is the part worth testing without one.

The algebra restates the vendor's `minimax_h3_pdd.py`, which ships beside the
published weights and is not installable. Restated, not imported, and the
docstrings say which parts are theirs.
"""

from __future__ import annotations

import math

import torch


def shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    """MiniMax-H3's flow shift. Matches `comfy/ldm/minimax/model.py`'s algebra."""
    return shift * sigma / (1 + (shift - 1) * sigma)


def pdd_time_grid(shift: float, num_steps: int) -> torch.Tensor:
    """Ascending `0 = t_0 < ... < t_N = 1` for one H3 schedule.

    Vendor equivalent: `minimax_h3_pdd.pdd_time_grid`.
    """
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shifted_sigma(shift, sigma)


def block_bounds(shift: float, num_steps: int, block_size: int) -> torch.Tensor:
    """The `nfe + 1` block boundaries a sampling run actually lands on.

    These are exactly the plain `nfe`-step shifted schedule, not an
    approximation of it: `linspace(1, 0, num_steps + 1)[::block_size]` IS
    `linspace(1, 0, nfe + 1)` when `block_size` divides `num_steps`, and
    `shifted_sigma` is pointwise. Verified to bit equality at 32/4 and shift
    12. That identity is why a PDD arm moves no sigma-shift node: the model is
    evaluated at the same times a normal 8-step render would use.
    """
    return pdd_time_grid(shift, num_steps)[::block_size].contiguous()


def fusion_plan(step_sizes: torch.Tensor, start: int,
                block_size: int) -> torch.Tensor:
    """Step-size weights over `[start, start + block_size)`, summing to 1.

    Vendor equivalent: `minimax_h3_pdd.pdd_sampling_plan`, minus the leading
    direction axis, which is always length 1 for a first-order solver.
    """
    plan = torch.zeros(step_sizes.shape[0], dtype=torch.float64)
    span = step_sizes[start:start + block_size].sum()
    plan[start:start + block_size] = step_sizes[start:start + block_size] / span
    return plan


def fuse_heads(stack: torch.Tensor, shift: float, num_steps: int,
               block_size: int) -> torch.Tensor:
    """`[num_steps, ...] -> [nfe, ...]`: one fused head per sampling step.

    `MiniMaxH3ParallelHead.forward` fuses WEIGHTS, not outputs, and its plan is
    a function of `(shift, num_steps, block_size, step)` alone. So collapsing
    the 32-head stack to `nfe` entries here is the same arithmetic, not an
    approximation -- there are only `nfe` distinct fusions a run can ever ask
    for.

    Computed in float64. The vendor casts its plan to the weight dtype (bf16)
    before the einsum, which moves the fused head by ~1.7e-3 relative; our
    output heads are ComfyUI's fp32 island, so we keep the precision and are
    deliberately NOT bit-identical to the reference.
    """
    steps = pdd_time_grid(shift, num_steps).diff()
    src = stack.to(torch.float64)
    return torch.stack([
        torch.tensordot(fusion_plan(steps, k * block_size, block_size), src,
                        dims=([0], [0]))
        for k in range(num_steps // block_size)
    ]).to(torch.float32)


def silu_temb_grid(proj_in_w: torch.Tensor, proj_in_b: torch.Tensor,
                   proj_out_w: torch.Tensor, proj_out_b: torch.Tensor,
                   rows: int, apply_silu: bool = True) -> torch.Tensor:
    """`silu(TimeEmbedder(t))` over `linspace(0, 1, rows)`, from raw tensors.

    Reimplements `comfy/ldm/minimax/model.py::TimeEmbedder.forward` (cos before
    sin, fp32 throughout) without instantiating the module, so it runs on a
    safetensors handle with no CUDA and no model. `apply_silu=False` returns
    the pre-silu embedding, which is what `t_emb` holds on an unpruned
    checkpoint and therefore what a nearest-row lookup must compare against.

    `freq_dim` comes from `proj_in_w.shape[1]` rather than a constant: a
    checkpoint disagreeing with the release's 256 would otherwise be silently
    mis-embedded rather than caught.
    """
    freq_dim = proj_in_w.shape[1]
    half = freq_dim // 2
    t = torch.linspace(0.0, 1.0, rows, dtype=torch.float32)
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    h = torch.nn.functional.linear(emb, proj_in_w.float(), proj_in_b.float())
    o = torch.nn.functional.linear(torch.nn.functional.silu(h),
                                   proj_out_w.float(), proj_out_b.float())
    return torch.nn.functional.silu(o) if apply_silu else o


def step_for_t(t: float, bounds: torch.Tensor, nfe: int,
               snap: float = 0.0) -> int:
    """Which fused head the time `t` belongs to.

    Two regimes, and the first one is not an optimisation -- it is the fix for
    a real off-by-one.

    **On schedule** (`|t - nearest boundary| <= snap`): the model is being
    evaluated AT a block boundary, so boundary `k` means head `k`. Snap to it.
    Plain interval membership is wrong here, because `t` does not arrive
    exact: it is recovered by a nearest-row lookup against a 1025-row curve
    table, which quantises to ~1e-3. A `t` sitting exactly on a boundary comes
    back a fraction BELOW it half the time, and membership then returns the
    previous block. Measured 2026-08-26 against the real `adaln_t_table` and
    the real 8-step shift-12 schedule: heads came out
    `[0, 0, 2, 3, 4, 5, 6, 6]` instead of `[0..7]` -- wrong at step 1 and at
    step 7, which is the largest jump in the schedule and where the fused
    heads differ most from each other.

    That failure is silent in every direction. The recovered `t` is correct to
    4e-5, so `boundary_residual` reports ~0 and no warning fires; the render
    completes; only the output is wrong. Snapping is what makes the residual
    check and the selection agree about what "on schedule" means.

    **Off schedule**: fall back to interval membership, which picks the
    interval containing `t` -- the closest available head -- rather than an
    arbitrary one. The caller is expected to warn; see `boundary_residual`.

    `snap=0.0` keeps pure membership, which is what the unit tests of the
    membership branch want.

    Deriving the step from `t` at all -- rather than from a call counter, as
    the vendor's forward hook does -- is what makes this recoverable. A counter
    cannot be checked against anything.
    """
    if snap > 0.0:
        d = (bounds - t).abs()
        j = int(d.argmin())
        if float(d[j]) <= snap:
            return max(0, min(j, nfe - 1))
    k = int(torch.searchsorted(bounds, torch.tensor(t, dtype=bounds.dtype),
                               right=True)) - 1
    return max(0, min(k, nfe - 1))


def boundary_residual(t: float, bounds: torch.Tensor) -> float:
    """Distance from `t` to the nearest block boundary.

    A run on the schedule the heads were fused for evaluates the model AT a
    boundary, so this is ~0 every step. A graph whose sigma shift or step count
    was changed without reconverting produces a large residual on the first
    step, which is the only cheap signal that the heads no longer match the
    trajectory. Nothing else about such a run looks wrong.
    """
    return float((bounds - t).abs().min())
