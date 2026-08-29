"""Schedule arithmetic shared by the PDD converter and the PDD node.

One copy, because the two consumers must agree exactly about what a block is
or the node decodes intervals the converter never meant, and nothing errors.
`bench/convert_pdd_lora.py` ships the 32-interval bank verbatim;
`pdd_lora.py` turns the sampler's sigma schedule into grid indices with
`schedule_knots` and fuses each span with `fuse_block`. A drift between them is
a silent wrong-head, which is the failure mode this module exists to make
impossible.

**Corrected 2026-08-27.** This said the converter bakes `fuse_heads` output into
the file and the node uses `block_bounds` to pick an entry. Neither is true now:
collapsing the stack pinned a step count into the artifact, and the node has no
step count at patch time. `block_bounds` survives as the closed form for the
uniform case. The checks assert it and `schedule_knots` agree to `torch.equal`
at every divisor, which is what keeps it honest as a reference rather than a
second answer -- but they are NOT its only consumer: `pdd_lora.emit_sigmas` is
`1.0 - block_bounds(...)`, so the node's SIGMAS output goes through it too.

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


def partition_bounds(shift: float, num_steps: int, widths) -> torch.Tensor:
    """`block_bounds` for a partition that is not uniform.

    `block_bounds` slices the grid at a fixed stride, which can only express a
    block size that divides `num_steps`. This indexes the same grid at the
    cumulative sums of `widths`, so any tiling is expressible -- including the
    ones a divisor cannot reach, of which six blocks of 32 is the one that
    matters.

    It is the same grid and the same `shifted_sigma`, so the identity
    `block_bounds` documents still holds: every boundary is a point the 32-grid
    already contains, and the model is evaluated at times it was fused for.
    Uniform `widths` reproduce `block_bounds` exactly, which
    `bench/check_pdd_sigmas.py` asserts rather than assumes.
    """
    if sum(widths) != num_steps:
        raise ValueError(
            f"partition {list(widths)} sums to {sum(widths)}, not the grid's "
            f"{num_steps}; the blocks would not tile it.")
    knots, acc = [0], 0
    for w in widths:
        acc += int(w)
        knots.append(acc)
    return pdd_time_grid(shift, num_steps)[knots].contiguous()


def base_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """Undo the flow shift: which point of the UNSHIFTED grid a sigma came from.

    The exact inverse of `shifted_sigma`, and the same algebra as the first line
    of `comfy/ldm/minimax/model.py::time_shift_sigma`, which inverts one shift
    before applying another.

    This is what makes a grid position readable from a sampler's sigma. The
    published 32-point grid is uniform in BASE sigma, not in `t` and not in the
    shifted sigma, so `1 - base_sigma(s)` scaled by `num_steps` is the grid
    index directly -- and it is the same index for both streams, because
    undoing each stream's own shift lands on the shared grid. Only the fusion
    weights below are per-stream.
    """
    return sigma / (shift + sigma * (1.0 - shift))


def schedule_knots(sample_sigmas, shift: float, num_steps: int) -> list[int]:
    """Which grid points the sampler's own schedule lands on.

    `sample_sigmas` is what `comfy/samplers.py` puts in `transformer_options`:
    the shifted sigmas this render will actually be evaluated at, `steps + 1`
    of them. Mapping each back to a grid index says which intervals each step
    spans, which is the whole of what a PDD run needs to know about the
    schedule -- and it is knowable only at RUN time, because the scheduler sits
    downstream of every model-patch node in the graph.

    Returns strictly increasing indices. Deduped rather than asserted unique:
    a schedule finer than the grid puts two steps on one grid point, which is a
    legal thing to ask for and means those steps share a head.

    On a schedule whose step count divides the grid this is exactly
    `range(0, num_steps + 1, block_size)`, so it reproduces `block_bounds`
    rather than approximating it. On one that does not, the blocks come out
    uneven -- 5 steps over 32 intervals gives `[0, 6, 13, 19, 26, 32]` -- which
    is a real answer and not an error, but is off the distribution the heads
    were distilled on.

    A `denoise` below 1.0 truncates the schedule, so the first knot is not 0.
    That is correct and is the reason this reads the sigmas rather than
    assuming a full trajectory.
    """
    s = torch.as_tensor(sample_sigmas, dtype=torch.float64).flatten()
    idx = torch.round((1.0 - base_sigma(s.clamp(0.0, 1.0), shift)) * num_steps)
    out: list[int] = []
    for k in idx.to(torch.int64).clamp(0, num_steps).tolist():
        if not out or k > out[-1]:
            out.append(k)
    return out


def fusion_plan(step_sizes: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """Step-size weights over `[start, stop)`, summing to 1.

    The paper's `D_k = (t_{k+1} - t_k) / (t_{stop} - t_start)` over the block,
    zero elsewhere. Vendor equivalent: `minimax_h3_pdd.pdd_sampling_plan`, minus
    the leading direction axis, which is always length 1 for a first-order
    solver.

    Takes an END index rather than a length as of 2026-08-27, because a block
    derived from the sampler's own schedule is a span between two knots and is
    not always the same width as its neighbours.
    """
    plan = torch.zeros(step_sizes.shape[0], dtype=torch.float64)
    span = step_sizes[start:stop].sum()
    plan[start:stop] = step_sizes[start:stop] / span
    return plan


def fuse_block(stack: torch.Tensor, shift: float, num_steps: int,
               start: int, stop: int) -> torch.Tensor:
    """One fused head: the block `[start, stop)`'s mean velocity.

    Slices the bank before weighting, so the fp64 intermediate is the block
    rather than the whole 32-head stack -- which matters when this is called
    inside a sampling step rather than once at load.

    **The plan follows the stack's device, and the stack decides.** The grid is
    derived arithmetic and is born wherever torch defaults to, while the bank
    is a buffer ComfyUI owns and moves; fusing on the bank's device keeps a
    42 MiB round trip off the wire every time the two differ.

    That line is here because of a real failure rather than a precaution. Until
    2af7f0b the bank lived in a closure and never left the CPU, so both
    operands were CPU by construction and this could not have gone wrong.
    Handing the bank to ComfyUI made it a managed model, ComfyUI moved it to
    cuda, and the first render after that raised `Expected all tensors to be on
    the same device` from inside this tensordot. The refactor was right and it
    made a previously unreachable device split reachable -- `CLAUDE.md`'s rule
    about the paths a fix brings to life.
    """
    steps = pdd_time_grid(shift, num_steps).diff()
    plan = fusion_plan(steps, start, stop)[start:stop].to(stack.device)
    return torch.tensordot(plan, stack[start:stop].to(torch.float64),
                           dims=([0], [0])).to(torch.float32)


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

    The uniform case, kept because the converter and three checks want the
    whole stack at once. A render goes through `fuse_block` per block instead,
    since a schedule-derived block is not always the same width as its
    neighbours -- both call the same `fusion_plan`.
    """
    return torch.stack([
        fuse_block(stack, shift, num_steps, k * block_size,
                   (k + 1) * block_size)
        for k in range(num_steps // block_size)
    ])


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


# `step_for_t` and `boundary_residual` lived here until 2026-08-26. They
# recovered a `t` from the curve table and bucketed it into a block, which is
# how a `t` sitting exactly on a boundary selected the previous one -- fixed by
# snapping, then made unnecessary altogether when `pdd_lora._StepTracker`
# started matching the block-boundary embeddings directly. Deleted rather than
# left: two selectors for one question is how the wrong one gets called.
