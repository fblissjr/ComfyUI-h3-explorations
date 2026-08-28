"""Apply a converted MiniMax-H3 Parallel Decoding Distillation LoRA.

PDD (arXiv 2607.26004, released for H3 by alibaba-pai) reaches the model on
three surfaces at once, and only one of them is a LoRA in the sense
`LoraLoaderModelOnly` understands. `bench/convert_pdd_lora.py` does everything
that can be done offline; this node does the rest, which is the part that needs
the loaded checkpoint in front of it.

    backbone   208 modules   weight patch, through comfy.lora
    adaln       50 modules   weight patch on an unpruned base; a weight patch
                             in the curve basis on a pruned one when the file
                             carries a bake; runtime injection otherwise
    heads       per block   per-step swap of final_layer's two output
                            linears; how many blocks is the schedule's
                            business, not this file's

## Why the head swap is keyed on the timestep

The vendor arms its parallel heads from a `register_forward_hook` that
increments a counter once per forward and wraps at `nfe`. Nothing ties that
counter to the schedule. One extra evaluation -- a CFG uncond pass, a warmup, a
shape probe, `torch.compile` tracing, an offload dry run -- desyncs it from the
trajectory for the rest of the render, and the wrap-around hides it instead of
raising.

We derive the step from `t_emb`, which is what the model was actually called
with, so it cannot desync. `final_layer.forward` receives separate rows for the
video and audio streams, and PDD runs those on separate schedules (shift 12 and
3), so the per-stream split falls out of the model's own signature rather than
having to be threaded through.

## Where the step COUNT comes from, which is a different question

Selecting a head needs two things: where on the grid this step starts, and how
far it goes. The start is in `t_emb`. The extent is not -- it lives in the
sampler's schedule and nowhere else, and at patch time that schedule does not
exist yet, because `BasicScheduler` sits downstream of every model-patch node
in the graph.

This node used to close that gap with an `nfe` widget the person had to keep
equal to `BasicScheduler.steps` by hand, backed by a warning that fires after
sampling has already started. A requirement with a warning behind it is not a
control, and the warning has never fired in a real render.

It now reads `transformer_options["sample_sigmas"]` -- the shifted sigmas this
render will actually be evaluated at, put there by `comfy/samplers.py` -- and
maps each back through `pdd_math.base_sigma` to a grid index. Those knots ARE
the block boundaries. On a step count that divides the grid they reproduce
`block_bounds` exactly; on one that does not they come out uneven and say so;
under `denoise < 1.0` they start partway down the trajectory, which is correct
and which a widget could not have expressed at all.

Reaching that dict needs one more patch point than the head swap does, so
`diffusion_model.forward` is patched to observe it and delegate. Sol-Attn
composes with `.forward` patches whose owner segment contains "attn"
(`vendor/sol_attn_minimax.py`), so it skips this one and the `final_layer` one
alike -- checked, not assumed. The patch chains onto whatever forward is
already installed rather than replacing it.

The `nfe` input survives as an override for deliberately off-schedule arms. At
its default of 0 nothing has to be entered and nothing can disagree.

## And then the direction was inverted, which is the real fix

Reading the schedule correctly still leaves every way of setting it WRONG in
place, because `scheduler` and `steps` live on a node this one sits above. So
this node also EMITS the schedule: `SIGMAS` is the second output, the shipped
non-split PDD graphs wire it straight into `SamplerCustomAdvanced`, and there
is no `BasicScheduler` in them at all. Off-grid stops being expressible rather
than being detected after the fact.

`1 - pdd_time_grid` is `shifted_sigma` over `linspace(1, 0, nfe + 1)` -- the
plain shifted schedule for the block count -- so this is bit-identical to
`BasicScheduler(simple, N)` at 2, 4 and 8 steps and moves no render.
`bench/check_pdd_sigmas.py` grades that against ComfyUI's own
`calculate_sigmas`, and grades that the graphs consume it.

The observe path above is NOT retired: it still drives head selection, so a
graph that leaves SIGMAS unwired behaves exactly as before -- which is what
`denoise < 1.0` and any deliberately off-grid arm need.

## Three traps this node is shaped around

**Patch `.forward` attributes; never wrap a module.** A wrapper `nn.Module`
holding the original under `.base` injects `.base.linear.weight` into the
parameter tree. ComfyUI's dynamic-VRAM streaming loader records every such path
in its backup and restores it by that path on unload -- by which time the
object patch has reverted the module and the path no longer resolves. Pattern
and diagnosis borrowed from `ComfyUI-MiniMax-H3-Turbo` (`_make_adaln_forward`,
its issue #4); the code here is ours.

**LoRA tensors are captured locals, never registered.** Same reason, plus it
keeps them off the offload bookkeeping entirely. They are cast to the
activation's device and dtype per call, which is also what makes the
CPU-offload case work while the projection runs on GPU.

**The adaln path is partition-specific, and usually is not an injection at
all.** On a pruned checkpoint the 2688-dim space the adaln delta lives in has
been replaced by an 8-column curve. `bench/convert_pdd_lora.py --pruned`
pre-solves the delta into that same basis, so it becomes an ordinary weight
patch and this node installs no adaln forward patches whatever -- no per-forward
`cdist`, no per-call casts, and strength composes through ComfyUI's own `diff`
path instead of a closure. The delta's time curve fits that basis to about
1e-4, measured per block at conversion time and refused there if it does not.

The runtime injection below is the fallback for a file converted without
`--pruned`, or one whose bake was solved against a different table. It is
partition-specific for the same reason the bake is: the fl2va and ref2va time
curves differ by 7.8% relative, which is why our grid is derived from the same
checkpoint that supplies the partition fingerprint rather than bundled once.
The turbo pack ships a single fl2va grid; reusing it for ref2va would feed the
injection a 7.8%-wrong input and render without complaint.

## Strength, and how it composes

Every surface scales through the path that owns it, so nothing here reimplements
weighting that ComfyUI already does:

  backbone       `add_patches(..., strength)`  -- native
  adaln, baked   `diff` / `diff_b` in the same dict -- native
  adaln, injected  captured in the closure and applied to the delta
  heads          `base + strength * (fused - base)`

`strength=0.0` installs nothing on any of them, so it is exactly the base
model. That is deliberately a different control from the one
`workflows/h3_config.py` recommends for plain LoRAs, where 0.0 short-circuits
the dequantise/add/requantise round trip and 0.01 is the like-for-like
baseline -- use 0.01 here to price the backbone's numerical cost, 0.0 to get
the base model back.

## What the timestep keying buys downstream

Deriving the step from `t_emb` rather than counting forwards is not only
robustness against an extra evaluation. It is what lets this node sit in the
graphs the repo already ships:

  * **the step cache** skips forwards on reused steps. A counter would drift by
    exactly the number of skips and never recover; a time lookup simply reads
    whatever step the next real forward is at.
  * **`split_at` two-pass sampling** runs part of the trajectory on one model
    and part on another. A pass covering only the tail still selects the heads
    for the tail, because the heads are indexed by time and not by how many
    calls this particular model object has seen.
  * **CFG**, if a graph ever runs it, doubles the forwards per step without
    moving time at all.

None of those needed special handling. They are the reason not to port the
vendor's forward-hook counter, stated as capabilities rather than as hazards.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
from comfy_api.latest import io

try:                                     # loaded as a package by ComfyUI
    from .pdd_math import (block_bounds, fuse_block, pdd_time_grid,
                           schedule_knots, silu_temb_grid)
except ImportError:                      # loaded as a bare module by a script
    from pdd_math import (block_bounds, fuse_block, pdd_time_grid,
                          schedule_knots, silu_temb_grid)

logger = logging.getLogger(__name__)

# Dtypes, and why each is what it is. They are not interchangeable here.
#
#   fused heads, baked adaln diffs   fp32 on disk. `final_layer`'s two output
#       projections are the checkpoint's fp32 island -- `comfy/ldm/minimax/
#       model.py` builds them with an explicit `dtype=torch.float32` while the
#       rest of the block is model dtype -- and the vendor's own fusion loses
#       ~1.7e-3 by casting its plan to bf16 before the einsum. Storing fp32
#       keeps the precision the island exists for.
#   backbone / adaln LoRA pairs      bf16, as published. `calculate_weight`
#       casts them itself, so nothing is gained by widening on disk.
#   curve table and time grid        fp32, and compared in fp32. The step
#       lookup is a `cdist` against 1025 rows spanning [0, 1]; in bf16 the row
#       spacing would be at the edge of representable and the recovered `t`
#       would quantise further than the 1e-3 the boundary snap already has to
#       absorb.

#: Warn once per patch when the timestep embedding sits this far from every
#: block boundary, in EMBEDDING distance rather than in `t`. Selection no
#: longer needs a tolerance -- the nearest boundary is the answer -- so this
#: guards one thing only: whether the render is on the schedule the heads were
#: fused for at all. On schedule the distance is ~0; a step count the file was
#: not fused for puts it orders of magnitude above this.
BOUNDARY_TOLERANCE = 1e-2

#: `H3_PDD_TRACE=1` logs one line per sampling step naming the fused head each
#: stream selected and how far the timestep embedding sat from its boundary.
#:
#: Off by default because a normal render does not want eight extra lines. On
#: when you are asking whether the mechanism is doing what it claims, which is
#: exactly the window the node was previously silent through: everything it
#: reports lands BEFORE sampling starts, and the sampling loop -- the longest
#: part of the render -- shows a progress bar and nothing else.
#:
#: That silence is not hypothetical. Two of this node's three shipped defects
#: were per-step behaviour: a head index that walked wrong at two of eight
#: steps, and 50 modulation patches that were never installed. Neither was
#: visible while a render was running.
TRACE = os.environ.get("H3_PDD_TRACE", "") not in ("", "0")

#: Relative-Frobenius distance allowed between the loaded checkpoint's
#: `final_layer.video_out` and the one the LoRA was converted against. The
#: fl2va and ref2va partitions sit ~0.05 apart (measured 2026-08-26 across the
#: six H3 checkpoints on this box) and a dtype cast on load moves it a few
#: thousandths, so anything in between is unambiguous. Set an order of
#: magnitude below the partition gap and an order above a cast.
PARTITION_TOLERANCE = 0.015

#: Relative distance allowed between the loaded checkpoint's `adaln_t_table`
#: and the one a bake was solved against. A partition's `int8_convrot` and
#: `fp8_scaled` builds carry byte-identical tables (verified 2026-08-26), so
#: this only has to separate a dtype cast from a different partition's basis.
#: Measured the same day: a bf16 cast of a table is 0.00164 away, and the fl2va
#: and ref2va tables are 0.01835 apart. This sits between them -- three times
#: above the cast, three and a half below the gap.
#:
#: It was 1e-3 for about an hour, which is BELOW the cast. That would have
#: rejected every correct bake the moment the loader cast the buffer and fallen
#: back to the injection -- slower, and silent, because the fallback is correct.
#: The same shape as hashing `video_out`: a tolerance under the noise it has to
#: tolerate.
#:
#: A mismatch FALLS BACK rather than raising. The injection is correct on any
#: pruned base, so refusing the render would be worse than taking the slow path.
#: What it must not do is bake anyway: a bake solved against the wrong basis is
#: 0.0205 wrong at runtime against 0.0001 for the right one, and nothing
#: downstream would say so.
TABLE_TOLERANCE = 5e-3


def resolve_emit_steps(steps, file_nfe: int, num_steps: int) -> int:
    """Evaluations for the SIGMAS output: what `steps` asked for, or the file's.

    Lifted out of `MiniMaxH3PDDLoRA.execute` so it can be driven without a
    loaded model -- `bench/check_pdd_sigmas.py` calls it directly, and a case
    that restated this condition instead of calling it could not have failed.

    Two behaviours, and the asymmetry is the point:

      * **0 never refuses.** It means "the file's own count", which keeps this
        input inert for any graph that does not consume SIGMAS -- including a
        deliberately off-grid arm driving `BasicScheduler` at a count that does
        not tile the grid. That arm stays legal, and the MODEL path still
        reports it at run time. An earlier version of this raised
        unconditionally and would have refused a 6-step render in flight at
        the time it was written.
      * **A non-zero request MUST tile the grid.** At such a count no on-grid
        schedule exists, so there is nothing honest to emit, and raising is the
        only answer that is not silently off it.
    """
    asked = int(steps)
    if not asked:
        return int(file_nfe)
    if num_steps % asked:
        legal = sorted(n for n in range(1, num_steps + 1) if num_steps % n == 0)
        raise RuntimeError(
            f"steps={asked} does not divide the file's {num_steps}-point "
            f"grid, so the blocks cannot tile it and the SIGMAS output would "
            f"step somewhere these heads were never fused for. Legal here: "
            f"{legal}. This raises rather than warning because the failure is "
            f"otherwise silent: the render completes and is merely wrong.")
    return asked


def _is_minimax_h3(diffusion_model) -> bool:
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
    except ImportError:
        return False
    return isinstance(diffusion_model, MiniMaxH3Model)


def _row_index(row) -> int:
    """The `t_emb` row a stream's segment refers to.

    `final_layer` is handed an int normally, and a per-token LongTensor when a
    denoise mask puts rows at different strengths. In that case the stream's
    own timestep is the SMALLEST row value: masked rows are pinned toward the
    conditioning timestep (`rows_t = (1 - m * sigma).clamp(max=t_pin)`), so the
    fully-denoised rows -- the ones on the sampler's actual trajectory -- carry
    the minimum. Taking the max or the mean would read a pinned conditioning
    row and select a head for a time the sampler never visits.
    """
    if torch.is_tensor(row):
        return int(row.min().item())
    return int(row)


class _StepTracker:
    """Which block of the grid each stream's step spans.

    Two independent questions, and they have different answers.

    **Where the step starts** comes from `t_emb`, matched against the block
    boundary embeddings. That is the whole selector: the nearest boundary is
    the block, directly, and it cannot desync from the trajectory because it is
    read off what the model was called with.

    An earlier version recovered a `t` by nearest row of the 1025-row curve
    table and then bucketed it, which is how a `t` sitting exactly ON a boundary
    came back a fraction below it and selected the previous block -- wrong at
    two of eight steps, silent, and it took a deliberate drive against real
    inputs to find. It was fixed by snapping, i.e. by a tolerance that then had
    to be justified against the table's own quantisation. Matching the
    boundaries directly deletes that problem rather than guarding it: no
    intermediate `t` to quantise, exactly `nfe` answers to fall between, no
    tolerance needed.

    **How far it goes** cannot come from `t_emb`. It is a property of the
    schedule, which `observe` reads from `transformer_options["sample_sigmas"]`
    at run time. Until 2026-08-27 it came from an `nfe` widget instead, which
    made the schedule and the heads two facts that had to be typed to agree.

    The boundary tolerance survives, guarding what is now a genuinely different
    question. It no longer asks "is the step count right" -- the step count is
    taken from the sampler, so it is right by construction. It asks whether the
    model is being evaluated at a time the schedule said it would be, which is
    what goes wrong under a sampler that evaluates off its own grid.
    """

    def __init__(self, grid_emb_v, grid_emb_a, grid_t_v, grid_t_a,
                 shift_v, num_steps, forced_nfe, fallback_nfe, label):
        # Embeddings at EVERY grid point, not just the ones one step count
        # visits: which subset is a boundary is not known until a schedule is.
        self.grid_emb_v = grid_emb_v         # [num_steps+1, D]
        self.grid_emb_a = grid_emb_a
        self.grid_t_v = grid_t_v
        self.grid_t_a = grid_t_a
        self.shift_v = shift_v
        self.num_steps = num_steps
        self.forced_nfe = forced_nfe
        self.label = label
        self.warned = False
        self._key = object()             # never equal to a real schedule key
        # A schedule is always observed before the first `final_layer` call --
        # the capture patch is on the model's own forward, which strictly
        # precedes it. This is only so the attributes exist.
        self._adopt(self._uniform(forced_nfe or fallback_nfe),
                    "the file's own step count; no schedule seen yet")

    def _uniform(self, nfe: int) -> list[int]:
        block = max(1, self.num_steps // max(1, nfe))
        return list(range(0, self.num_steps + 1, block))

    def _adopt(self, knots: list[int], why: str) -> None:
        self.knots = knots
        self.nfe = len(knots) - 1
        self.emb_v = self.grid_emb_v[knots]
        self.emb_a = self.grid_emb_a[knots]
        self.block_v = (knots[0], knots[1])
        self.block_a = (knots[0], knots[1])
        widths = [b - a for a, b in zip(knots, knots[1:])]
        uniform = len(set(widths)) == 1
        logger.info(
            "[h3-pdd] %s: %d evaluations over the %d-point grid, %s. Blocks %s.",
            self.label, self.nfe, self.num_steps, why,
            f"uniform, width {widths[0]}" if uniform else f"UNEVEN {widths}")
        if not uniform:
            logger.warning(
                "[h3-pdd] %s: this step count does not divide the %d-point "
                "grid, so the blocks are uneven (%s). Each step still decodes "
                "the mean velocity of the interval it spans, but a partial "
                "block is off the distribution the heads were distilled on. "
                "A count that divides the grid is exact: %s.",
                self.label, self.num_steps, widths,
                sorted(n for n in range(1, self.num_steps + 1)
                       if self.num_steps % n == 0))

    def observe(self, sample_sigmas) -> None:
        """Adopt the sampler's schedule, once per distinct schedule.

        Called from the model's own forward, so it runs before every
        `final_layer` call and re-runs for free if a graph samples twice with
        different schedules -- `split_at` two-pass sampling, or a second
        `SamplerCustomAdvanced` on the same patched model.
        """
        if sample_sigmas is None:
            key = None
        else:
            # The WHOLE vector, not a summary of it. `(len, first, last)` was
            # the first version and it collides: at 8 steps and shift 12,
            # `simple`, `beta`, `kl_optimal` and `linear_quadratic` all produce
            # (9, 1.0, 0.0) while deriving four different knot sets --
            # `simple` [0,4,8,...] against `kl_optimal` [0,24,28,30,31,32].
            #
            # That is not academic. `BasicScheduler` is DOWNSTREAM of this node,
            # so changing its scheduler does not re-execute the node: the cached
            # ModelPatcher keeps this tracker, `observe` sees the new sigmas,
            # computes the same key and returns early, and every block after the
            # first decodes an interval the sampler never steps over. Silent,
            # and the boundary warning names the wrong cause when it fires.
            #
            # A sigma vector is at most a few dozen values and `observe` runs
            # once per forward, so comparing all of them costs nothing.
            s = torch.as_tensor(sample_sigmas).detach().flatten()
            key = tuple(s.tolist())
        if key == self._key:
            return
        self._key = key
        if self.forced_nfe:
            self._adopt(self._uniform(self.forced_nfe),
                        f"FORCED by the node's nfe={self.forced_nfe}, ignoring "
                        f"the sampler's schedule")
            return
        if key is None:
            logger.warning(
                "[h3-pdd] %s: the sampler put no `sample_sigmas` in "
                "transformer_options, so the block extents cannot be derived. "
                "Falling back to uniform blocks at the file's own step count. "
                "Set the node's `nfe` if this render uses a different one.",
                self.label)
            return
        knots = schedule_knots(sample_sigmas, self.shift_v, self.num_steps)
        if len(knots) < 2:
            logger.warning(
                "[h3-pdd] %s: this schedule lands on fewer than two grid "
                "points, so it names no block. Keeping the previous blocks.",
                self.label)
            return
        self._adopt(knots, "derived from the sampler's own sigma schedule")

    def _pick(self, t_emb, row, table):
        e = t_emb[_row_index(row)].detach().float().reshape(1, -1)
        d = torch.cdist(e, table.to(e.device, torch.float32))[0]
        j = int(d.argmin())
        return min(j, self.nfe - 1), float(d[j])

    def update(self, t_emb, video_seg, audio_seg) -> None:
        jv, dv = self._pick(t_emb, video_seg[2], self.emb_v)
        ja, da = self._pick(t_emb, audio_seg[2], self.emb_a)
        self.block_v = (self.knots[jv], self.knots[jv + 1])
        self.block_a = (self.knots[ja], self.knots[ja + 1])
        if TRACE:
            # The ordinal comes from the block, not from a call counter. A
            # counter reported "step 5/4" on the second render of a session:
            # ComfyUI caches the patched model, so this closure outlives one
            # sampling run, and a running count is a fact about call history
            # rather than about the trajectory. Same class of mistake as the
            # vendor's forward-hook step index, in the diagnostic instead of in
            # the selection.
            logger.info(
                "[h3-pdd] step %d/%d: video block %s (%.5f from its boundary), "
                "audio block %s (%.5f)", jv + 1, self.nfe,
                self.block_v, dv, self.block_a, da)
        if not self.warned and max(dv, da) > BOUNDARY_TOLERANCE:
            self.warned = True
            logger.warning(
                "[h3-pdd] %s: the model is being evaluated at a time this "
                "render's schedule does not contain. The timestep embedding "
                "sits %.4f (video) and %.4f (audio) from the nearest of the %d "
                "block boundaries, tolerance %.4f. Blocks are taken from "
                "`sample_sigmas`, so this is not a step-count mismatch -- it is "
                "a sampler evaluating off its own grid, or an `nfe` override "
                "that does not match it. Sampling continues on the nearest "
                "boundary.",
                self.label, dv, da, self.nfe + 1, BOUNDARY_TOLERANCE)


class _FusedHeads:
    """The block heads this render actually asks for, fused once each.

    The paper's section 3.1 says to hold one fused linear per block rather than
    an enlarged final layer, and that is what this is -- it just cannot know
    WHICH blocks until the schedule is known, so it fuses on first use instead
    of at load. A render visits at most `nfe` distinct blocks, so this settles
    after the first pass and the sampling loop is a dict lookup thereafter.

    Two levels: fp32 masters on CPU keyed by the block, and one cast copy per
    `(block, device, dtype)`. The cast cache is the same per-device pattern as
    before, borrowed from `ComfyUI-MiniMaxH3-PDD-Mamad8::PDDHeads.for_device`;
    the master cache is what makes the fusion once-per-block rather than
    once-per-step. Held in a closure and never registered on a module, so the
    streaming loader's backup bookkeeping never sees them.

    `strength` interpolates toward the fused head, so 0.0 would be the base head
    exactly -- but the node installs nothing at all at 0.0, so that path is the
    checkpoint's own `operations.Linear` rather than this one.
    """

    def __init__(self, bank_w, bank_b, base_w, base_b, shift, num_steps,
                 strength):
        self.bank_w = bank_w
        self.bank_b = bank_b
        self.base_w = base_w
        self.base_b = base_b
        self.shift = shift
        self.num_steps = num_steps
        self.strength = strength
        self._master: dict[tuple, tuple] = {}
        self._cast: dict[tuple, tuple] = {}

    def get(self, block, device, dtype):
        key = (block, str(device), dtype)
        hit = self._cast.get(key)
        if hit is not None:
            return hit
        master = self._master.get(block)
        if master is None:
            w = fuse_block(self.bank_w, self.shift, self.num_steps, *block)
            b = fuse_block(self.bank_b, self.shift, self.num_steps, *block)
            master = (self.base_w + self.strength * (w - self.base_w),
                      self.base_b + self.strength * (b - self.base_b))
            self._master[block] = master
        hit = (master[0].to(device, dtype), master[1].to(device, dtype))
        self._cast[key] = hit
        return hit


def boundary_embeddings(bounds, table, time_embedder=None, rows=1025):
    """The `t_emb` the model will produce AT each block boundary.

    Built the same two ways the model builds `t_emb`, chosen by the same
    observable the rest of this node branches on -- a curve table when the
    checkpoint is pruned, the time embedder when it is not -- so the thing being
    matched against is constructed by the model's own arithmetic rather than
    approximated.
    """
    out = []
    for t in bounds.tolist():
        if time_embedder is None:
            pos = min(max(float(t), 0.0), 1.0) * (table.shape[0] - 1)
            i0 = min(int(pos), table.shape[0] - 2)
            out.append(torch.lerp(table[i0].float(), table[i0 + 1].float(),
                                  pos - i0))
        else:
            out.append(table[min(int(round(float(t) * (rows - 1))), rows - 1)])
    return torch.stack(out)


def _make_final_layer_forward(base_forward, tracker):
    """Bookkeeping only, then the stock forward.

    Deliberately does not reimplement the modulation maths: the two head swaps
    below are separate patches on the output linears, so `FinalLayer.forward`
    stays upstream's and a change to it does not silently diverge here.

    **The trailing `*args` is load-bearing and is not defensive padding.**
    Comfy-Org/ComfyUI#15908 (open 2026-08-27, `comfy/ldm/minimax/model.py`
    only) adds PDD support to core and widens this exact signature to
    `(x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts)`. An object
    patch replaces the method, so the four names we bind are the four we read
    and everything after them has to reach `base_forward` untouched or the
    stock forward loses arguments it now requires. Pinned to four, this node
    raises a TypeError on the first sampling step the day that merges.

    Widening costs nothing here and is not a claim that the rest of the node
    survives core learning PDD. It does not: core keys off
    `video_out.weight.shape[0] // out_features`, our converted file leaves that
    weight its original size, so core takes its `n == 1` path and our two
    output-linear patches still own the head swap. That is correct, and it is
    also two implementations of one mechanism sitting in one process. See
    `docs/h3_pdd.md`.
    """
    def forward(x, t_emb, video_seg, audio_seg, *args, **kwargs):
        tracker.update(t_emb, video_seg, audio_seg)
        return base_forward(x, t_emb, video_seg, audio_seg, *args, **kwargs)
    return forward


#: The three keys this node takes when it installs the head swap. Anything
#: already holding one of them owns H3's output projections, and two owners is
#: a silently wrong render.
HEAD_PATCH_KEYS = ("diffusion_model.final_layer.forward",
                   "diffusion_model.final_layer.video_out.forward",
                   "diffusion_model.final_layer.audio_out.forward")


def head_patch_clash(object_patches) -> list[str]:
    """Which of this node's head-patch keys are already taken.

    A free function so the guard can be graded without a loaded H3 -- the
    predicate is the whole of it, and the call site is one `if`. Takes the
    patch mapping rather than a ModelPatcher for the same reason.
    """
    return [k for k in HEAD_PATCH_KEYS if k in object_patches]


def _make_capture_forward(base_forward, tracker):
    """Observe the sampler's schedule, then the stock forward.

    `transformer_options` reaches the model's own forward and stops there --
    `final_layer` never sees it, and neither do the output linears. This is the
    nearest patch point that does, and it is the only reason the node touches
    `diffusion_model.forward` at all.

    Chains: `base_forward` is whatever forward was installed before this,
    so a pack that has already patched the model's forward keeps working.
    Sol-Attn composes only with `.forward` patches whose owner segment contains
    "attn" (`vendor/sol_attn_minimax.py`), and this one's owner is
    `diffusion_model`, so it is left alone rather than gated behind Sol's sigma
    window -- which would have made the capture run only inside that window.
    """
    def forward(*args, **kwargs):
        opts = kwargs.get("transformer_options")
        if opts is None and len(args) > 3:
            opts = args[3]                       # positional in the signature
        tracker.observe((opts or {}).get("sample_sigmas"))
        return base_forward(*args, **kwargs)
    return forward


def _make_head_forward(heads, tracker, stream):
    """Replace one output linear with the fused head for the current block.

    The block is a `(start, stop)` span of the published grid rather than an
    index into a precomputed stack, because its width is a property of the
    schedule and is not known when this patch is installed. `_FusedHeads` fuses
    each span once and caches it; this is a dict lookup after the first pass.
    """
    def forward(inp):
        block = tracker.block_v if stream == "video" else tracker.block_a
        w, b = heads.get(block, inp.device, inp.dtype)
        return F.linear(inp, w, b)
    return forward


def _make_adaln_forward(base, a, b, grid, table, strength):
    """Curve-mode adaln injection: add `strength * B @ A @ silu(t_emb)`.

    On a pruned checkpoint `t_emb` is an 8-column curve coordinate and the
    adaln linear consumes it directly (`apply_silu` is False). The LoRA update
    lives in the 2688-dim `silu(t_emb)` space that was collapsed away, so it is
    recovered per row: nearest row of the model's own `adaln_t_table`, then the
    matching row of the grid this file was converted with.

    Approach borrowed from `ComfyUI-MiniMax-H3-Turbo::_make_adaln_forward`,
    which solved this for the v4 turbo pack. Reimplemented rather than imported:
    that package bundles an fl2va-only grid, and ref2va is the arm this exists
    for.

    `a` and `b` are cast per call and deliberately NOT cached per device, unlike
    the output heads. Caching would pin roughly 25 MB of fp32 per block across
    50 blocks -- over a gigabyte of VRAM standing next to a 24 GB checkpoint on
    a 24 GB card, to save transfers that cost about a second across a whole
    render. The heads are cached because there are two of them, not a hundred.
    """
    def forward(t_emb):
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        tb = table.to(t_emb.device, torch.float32)
        idx = torch.cdist(t_emb.detach().float(), tb).argmin(dim=1)
        st = grid.to(x.device, x.dtype)[idx]                   # [M, 2688]
        av = a.to(x.device, x.dtype)
        bv = b.to(x.device, x.dtype)
        x = x + strength * (bv @ (av @ st.T)).T
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)
    return forward


class MiniMaxH3PDDLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths
        return io.Schema(
            node_id="MiniMaxH3PDDLoRA",
            display_name="MiniMax H3 PDD LoRA (parallel decoding)",
            category="loaders/minimax",
            description=(
                "Loads a Parallel Decoding Distillation LoRA converted by "
                "bench/convert_pdd_lora.py. PDD is not step distillation: the "
                "trajectory is a 32-point grid whose final output head is "
                "replicated per interval, and each sampling step decodes a "
                "block of those heads as one mean velocity. The block "
                "boundaries are exactly the plain 8-step shifted schedule, so "
                "a PDD arm changes the sampler's step count and NOTHING else "
                "-- the shift stays at the checkpoint's own 12/3. Place where "
                "a LoRA loader goes: before MiniMaxH3SigmaShift and before the "
                "attention nodes."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "lora_name",
                    options=[n for n in folder_paths.get_filename_list("loras")
                             if "pdd" in n.lower()],
                    tooltip=(
                        "A CONVERTED PDD file. The published alibaba-pai "
                        "weights do not load here or anywhere else in "
                        "ComfyUI: their keys are diffusers-side and their "
                        "suffixes are bare `lora_down`/`lora_up`, which no "
                        "ComfyUI adapter matches, so every one of the 728 "
                        "tensors would be skipped with a log line and the "
                        "render would come out as an undistilled 8-step pass. "
                        "Run bench/convert_pdd_lora.py first."
                    ),
                ),
                io.Float.Input(
                    "strength", default=1.0, min=-10.0, max=10.0, step=0.01,
                    tooltip=(
                        "Scales all three mechanisms together, and each one "
                        "through the path that owns it: the backbone and the "
                        "adaln update are ordinary ComfyUI weight patches, so "
                        "they take strength natively, and each fused head is "
                        "base + strength * (fused - base) against the "
                        "checkpoint's own head. 1.0 is the vendor's default "
                        "and what their published clips were rendered at.\n\n"
                        "0.0 installs nothing at all and is exactly the base "
                        "model, heads included. Note that is a DIFFERENT "
                        "control from the one h3_config recommends for plain "
                        "LoRAs: there 0.0 short-circuits the "
                        "dequantise/add/requantise round trip and 0.01 is the "
                        "like-for-like baseline. Use 0.01 to isolate the "
                        "backbone's numerical cost, 0.0 to get the base model."
                    ),
                ),
                # APPENDED, not inserted. Saved graphs match widget values by
                # index, so a new widget ahead of `strength` would land an old
                # graph's float on this boolean. Same rule as the head_chunks
                # input on MiniMaxH3SageAttention.
                io.Boolean.Input(
                    "patch_heads", default=True, optional=True,
                    tooltip=(
                        "Off runs the backbone and adaln updates against the "
                        "checkpoint's OWN output heads -- PDD's whole "
                        "mechanism disabled, everything else applied. That is "
                        "the control for whether the per-interval heads earn "
                        "their complexity: measured against the base head the "
                        "fused heads sit 0.005 apart early and 0.015 apart at "
                        "the last step, so if this arm is indistinguishable "
                        "the head machinery is not what is doing the work. "
                        "On by default; turning it off is an experiment, not "
                        "a fallback."
                    ),
                ),
                # APPENDED. See the note on patch_heads.
                io.Int.Input(
                    "nfe", default=0, min=0, max=64, optional=True,
                    tooltip=(
                        "LEAVE THIS AT 0. The step count is read from the "
                        "sampler's own sigma schedule at run time, so it "
                        "cannot disagree with BasicScheduler and there is "
                        "nothing to keep in sync.\n\n"
                        "The published grid is 32 points, so any divisor is a "
                        "legal arm from the same weights -- 8 is what the file "
                        "records, 4 is the other count the vendor's README "
                        "reports rendering at -- and every divisor lands "
                        "exactly on the plain shifted schedule for its own "
                        "count. Change the SAMPLER's steps; this input does "
                        "not need to follow.\n\n"
                        "Non-zero forces uniform blocks at that count and "
                        "IGNORES the schedule, which is only useful for "
                        "deliberately decoding one grid partition while "
                        "stepping another. It logs loudly that it is doing "
                        "so.\n\n"
                        "**`nfe` and `steps` are allowed to disagree, and that "
                        "disagreement IS this input's purpose.** `steps` picks "
                        "the schedule the SIGMAS output emits; `nfe` picks the "
                        "blocks the heads are fused into. Setting them to "
                        "different values is exactly \"decode one partition "
                        "while stepping another\", so it is not an error and "
                        "is not refused. It is announced instead: the tracker "
                        "logs `FORCED by the node's nfe=N, ignoring` on every "
                        "run where this is non-zero, which is the only thing "
                        "separating a deliberate arm from a forgotten widget."
                    ),
                ),
                # APPENDED. See the note on patch_heads.
                io.Int.Input(
                    "steps", default=0, min=0, max=64, optional=True,
                    tooltip=(
                        "Evaluations for the SIGMAS output, and the whole "
                        "reason that output exists.\n\n"
                        "Wire SIGMAS into SamplerCustomAdvanced instead of a "
                        "BasicScheduler and the sampler steps at exactly the "
                        "block boundaries these heads were fused for. There is "
                        "then no scheduler to pick wrong, no step count to "
                        "keep in sync, and off-grid sampling is not "
                        "expressible. On the 32-point grid at 2, 4 and 8 "
                        "steps this output is bit-identical to "
                        "`BasicScheduler(simple, N)`, so it changes no "
                        "existing render -- it removes the ways to get one "
                        "wrong. At 16 it differs by ~2e-3 because `simple` "
                        "reads a 1,000-entry table and 1000 % 16 != 0; the "
                        "closed form here is the more correct of the two.\n\n"
                        "0 emits the file's own count and never refuses, so "
                        "a graph that leaves SIGMAS unwired is untouched by "
                        "this input -- including a deliberately off-grid arm "
                        "driving BasicScheduler at a count that does not tile "
                        "the grid, which stays legal and still reports itself "
                        "at run time. Set it non-zero and it MUST divide the "
                        "grid: you have asked for a partition, and at a "
                        "non-dividing count no on-grid schedule exists, so "
                        "this raises rather than quietly emitting something "
                        "off it.\n\n"
                        "Leave SIGMAS unwired for denoise < 1.0, which this "
                        "output does not express; the MODEL output still "
                        "derives its blocks from whatever the sampler "
                        "publishes and handles a partial trajectory."
                    ),
                ),
            ],
            outputs=[io.Model.Output(), io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, model, lora_name, strength=1.0,
                patch_heads=True, nfe=0, steps=0) -> io.NodeOutput:
        import comfy.lora
        import comfy.utils
        import folder_paths

        # Coerced, not assumed. The schema declares FLOAT and BOOLEAN and the
        # generator writes 1.0 / True, but an API prompt is JSON a person can
        # hand-write, and `strength` reaches tensor arithmetic and an equality
        # against 0.0 that decides whether anything is installed at all.
        strength = float(strength)
        patch_heads = bool(patch_heads)

        dm = model.get_model_object("diffusion_model")
        if not _is_minimax_h3(dm):
            raise RuntimeError(
                f"This node only patches MiniMax H3; got {type(dm).__name__}.")

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd, meta = comfy.utils.load_torch_file(path, return_metadata=True)
        meta = meta or {}
        if "h3_pdd_converter_version" not in meta:
            raise RuntimeError(
                f"{lora_name} was not produced by bench/convert_pdd_lora.py "
                f"(no h3_pdd_converter_version in its metadata). The published "
                f"alibaba-pai files must be converted first; loading one "
                f"directly applies nothing at all.")

        num_steps = int(meta["pdd_num_steps"])
        # The step count is DERIVED at run time from the sampler's own sigmas,
        # so nothing here has to agree with `BasicScheduler` by hand. `nfe` is
        # an override for a deliberately off-schedule arm, and the file's own
        # count is the fallback for a sampler that publishes no schedule.
        file_nfe = int(meta["pdd_nfe"])
        forced_nfe = int(nfe)
        if forced_nfe and num_steps % forced_nfe:
            raise RuntimeError(
                f"nfe={forced_nfe} does not divide the file's {num_steps}-point "
                f"grid. As an OVERRIDE it forces uniform blocks, so it has to "
                f"tile the grid exactly; leave it at 0 to take the blocks from "
                f"the sampler's schedule, which handles an uneven count. Legal "
                f"here: {sorted(n for n in range(1, num_steps + 1) if num_steps % n == 0)}.")
        shift_v = float(meta["pdd_shift_video"])
        shift_a = float(meta["pdd_shift_audio"])

        # The SIGMAS output, and why it is worth a second output rather than a
        # note telling people to set BasicScheduler correctly.
        #
        # Every knob that could put this render off its own grid lives in a
        # node DOWNSTREAM of this one -- `BasicScheduler` sits below every
        # model-patch node -- so this node can only ever observe the schedule
        # after the fact and report. Emitting the schedule inverts that: the
        # sampler consumes the boundaries these heads were fused for, and
        # "off-grid" stops being a thing a graph can express. Same move as
        # `ComfyUI-UtilsCollection`'s PDD node, whose off-grid error says to
        # use its SIGMAS output; ours makes that the wiring rather than the
        # advice.
        #
        # `1 - pdd_time_grid` is `shifted_sigma` over `linspace(1, 0, N+1)`,
        # which is the plain shifted schedule for the block count -- so this is
        # the closed form of what `simple` approximates, not a second opinion
        # about it. Bit-identical to `BasicScheduler(simple, N)` at 2, 4 and 8
        # steps on shift 12 and shift 6; ~2e-3 apart at 16, where `simple`
        # quantises against its 1,000-entry table because 1000 % 16 != 0 and
        # this is the more correct of the two. Graded in
        # `bench/check_pdd_sigmas.py` against ComfyUI's own scheduler.
        # 0 means "the file's own count". That keeps this input inert for any
        # graph that does not consume SIGMAS -- including a deliberately
        # off-grid arm at a count that does not tile the grid, which the MODEL
        # path still supports and reports. Only an explicit request is graded.
        emit_steps = resolve_emit_steps(steps, file_nfe, num_steps)
        block_w = num_steps // emit_steps
        # The trained envelope, warned rather than refused. The file records
        # the width it was distilled at; `ComfyUI-UtilsCollection` hard-refuses
        # anything past twice that, on the reasoning that a block averaged from
        # too many heads is a long way from what the distillation produced.
        # This repo deliberately renders the 2x arm (4 steps against a
        # block-4 file) and has an open question about whether it holds up, so
        # refusing would break a live experiment. Naming it is the compromise:
        # nothing said so before, at any step count.
        trained_w = int(meta.get("pdd_block_size", block_w))
        if block_w > 2 * trained_w:
            logger.warning(
                "[h3-pdd] steps=%d gives block width %d against a file "
                "distilled at width %d. That is past the 2x envelope an "
                "independent implementation refuses outright, and nothing "
                "here has measured whether it holds up. Rendering anyway.",
                emit_steps, block_w, trained_w)
        sigmas = (1.0 - block_bounds(shift_v, num_steps, block_w)).to(
            torch.float32)

        # Partition check. fl2va and ref2va ship identical key sets, so a
        # mismatched pair loads with zero unmatched keys and renders -- the
        # silent-success failure docs/h3_ref2v_distillation.md records.
        #
        # Compared BY DISTANCE, not by hash. The first version hashed this
        # tensor and fired on the first real render against the correct
        # checkpoint: ComfyUI casts on load, and a cast changes every bit while
        # moving the value a fraction of a percent. The two partitions are 5%
        # apart and a cast is a few tenths of one, so the separation is an
        # order of magnitude and the threshold does not need to be delicate --
        # but an exact hash had no separation at all.
        ref = sd.get("h3_pdd.base_video_out")
        if ref is not None:
            live = dm.final_layer.video_out.weight.detach().to(torch.float32).cpu()
            # Shape first, because a differently-SHAPED head is a different
            # failure from a different partition and the subtraction below
            # raises an unreadable broadcast error on it.
            #
            # Observed 2026-08-27 under Comfy-Org/ComfyUI#15908: a graph that
            # ran Kijai's PDD file left `video_out.weight` resident at
            # [32*out, in], and the next graph through this node read the
            # enlarged tensor off the cached model. That is what an
            # enlarging-patch approach costs -- the shape change outlives the
            # graph that asked for it -- and it is the reason ours patches the
            # projection's forward instead of resizing its weight.
            if live.shape != ref.shape:
                raise RuntimeError(
                    f"{lora_name}: this model's final_layer.video_out is "
                    f"{tuple(live.shape)}, not the {tuple(ref.shape)} it was "
                    f"converted against. A head that is an exact multiple of "
                    f"the expected rows is an ENLARGED PDD head bank left "
                    f"resident by another implementation -- core reads "
                    f"`weight.shape[0] // out_features` as its interval count, "
                    f"so the tensor survives on the cached model after the "
                    f"graph that installed it. Restart ComfyUI, or run that "
                    f"graph and this one in separate sessions.")
            dist = float((live - ref.to(torch.float32)).norm() / ref.norm())
            if dist > PARTITION_TOLERANCE:
                raise RuntimeError(
                    f"{lora_name} was converted against a different checkpoint "
                    f"partition than the one loaded: final_layer.video_out is "
                    f"{dist:.4f} away from the one it was built against "
                    f"({meta.get('h3_pdd_base', '?')}), tolerance "
                    f"{PARTITION_TOLERANCE}. The fl2va and ref2va partitions "
                    f"sit about 0.05 apart and a dtype cast moves this a few "
                    f"thousandths, so this is a partition mismatch and not a "
                    f"loader artifact. Their key sets are identical, so this "
                    f"would otherwise render without one unmatched key and "
                    f"merely be wrong.")
            logger.info("[h3-pdd] partition check ok: final_layer.video_out is "
                        "%.5f from %s", dist, meta.get("h3_pdd_base", "?"))

        pruned = bool(getattr(dm, "use_adaln_curves", False))

        backbone = {k: v for k, v in sd.items() if k.startswith("diffusion_model.")}
        adaln = {k: v for k, v in sd.items() if k.startswith("h3_pdd.adaln.")}
        # From the converter's own count, not from a key prefix. Counting
        # `h3_pdd.adaln.` missed every `h3_pdd.adaln_baked.` key -- the prefix
        # is not a prefix of the other -- so a file carrying only the baked
        # form reported 0 modules, the install loop never ran, and four arms
        # rendered with the backbone and heads but NO adaln update. The node
        # logged "0 adaln" and completed.
        n_adaln = int(meta.get("adaln_modules") or 0) or len(
            {k.rsplit(".", 1)[0] for k in adaln})

        # Which of the three adaln paths this checkpoint gets. Branching on
        # `use_adaln_curves` and on the live table -- observables of the loaded
        # model -- rather than on anything in the filename.
        adaln_installed = 0
        baked = None
        if pruned and "h3_pdd.adaln_table" in sd:
            live_table = dm.adaln_t_table.detach().to(torch.float32).cpu()
            ref_table = sd["h3_pdd.adaln_table"].to(torch.float32)
            if live_table.shape == ref_table.shape:
                d = float((live_table - ref_table).norm() / ref_table.norm())
                if d <= TABLE_TOLERANCE:
                    baked = d
                else:
                    logger.warning(
                        "[h3-pdd] %s was baked against a different curve table "
                        "(%.5f away, tolerance %g); falling back to the runtime "
                        "adaln injection, which is correct on any pruned base.",
                        lora_name, d, TABLE_TOLERANCE)

        if not pruned:
            # Unpruned base: the adaln update is an ordinary weight patch in the
            # 2688-dim time space, so hand it to comfy.lora with the rest.
            # The baked tensors are NOT offered here -- they are 8 columns wide
            # and `calculate_weight` only WARNS on a shape mismatch before
            # skipping, so a wrong-width diff would drop 50 modules with a log
            # line rather than an error.
            for k, v in adaln.items():
                i = k.split(".")[3]
                slot = "lora_A" if k.endswith("lora_A") else "lora_B"
                backbone[f"diffusion_model.blocks.{i}.adaln_proj.linear."
                         f"{slot}.weight"] = v
            adaln_installed = len({k.split(".")[3] for k in adaln})
        elif baked is not None:
            # Pruned, with a bake solved against this checkpoint's own basis.
            # `diff` / `diff_b` are comfy.lora's own patch kinds and take
            # strength like any other patch.
            for i in range(n_adaln):
                base_key = f"diffusion_model.blocks.{i}.adaln_proj.linear"
                backbone[f"{base_key}.diff"] = \
                    sd[f"h3_pdd.adaln_baked.blocks.{i}.diff"]
                backbone[f"{base_key}.diff_b"] = \
                    sd[f"h3_pdd.adaln_baked.blocks.{i}.diff_b"]
                adaln_installed += 1

        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        loaded = comfy.lora.load_lora(backbone, key_map, log_missing=True)
        if not loaded:
            raise RuntimeError(
                f"{lora_name} matched no module on this model. Expected "
                f"ComfyUI generic-LoRA keys under `diffusion_model.`; the "
                f"conversion may predate a checkpoint layout change.")

        m = model.clone()
        # `add_patches` returns only the keys it found in the model's state
        # dict, so this separates "matched nothing" -- already caught above --
        # from "matched SOME", which is the case `docs/h3_pdd.md` listed under
        # Enforced by nothing until 2026-08-27. A checkpoint layout change that
        # renames a subset leaves the rest applied and the render plausible.
        #
        # Adopted from `silveroxides/ComfyUI-UtilsCollection`, which had it and
        # we did not. Their count is over the source keys; ours is over what
        # `load_lora` resolved, which is the same question one step later and
        # does not need to know which suffixes the converter emitted.
        applied = m.add_patches(loaded, strength)
        if len(applied) != len(loaded):
            missing = sorted(set(loaded) - set(applied))
            raise RuntimeError(
                f"{lora_name} matched {len(applied)} of {len(loaded)} patch "
                f"keys on this model. A partial match renders and looks "
                f"entirely normal, with whichever modules did not match left "
                f"at their base weights. First unmatched: {missing[:3]}.")



        # --- the runtime surfaces -------------------------------------------
        # The step tracker needs a table in whatever space `t_emb` lives in,
        # and that follows `pruned` alone -- NOT whether the adaln update was
        # baked. Those two were briefly conflated while adding the bake, which
        # left `step_table` unset on the pruned-and-baked path, i.e. on the
        # default configuration.
        if pruned:
            table = dm.adaln_t_table
            step_table = table
        else:
            te = dm.time_embedder
            step_table = silu_temb_grid(
                te.proj_in.weight, te.proj_in.bias,
                te.proj_out.weight, te.proj_out.bias,
                rows=int(meta.get("pdd_grid_rows", 1025)), apply_silu=False)

        if pruned and baked is None:
            # Fallback only: a file with no bake, or one solved against another
            # table. Installs 50 forward patches the baked path does not need.
            grid = sd["h3_pdd.silu_temb_grid"]
            for i in range(n_adaln):
                base = m.get_model_object(f"diffusion_model.blocks.{i}.adaln_proj")
                m.add_object_patch(
                    f"diffusion_model.blocks.{i}.adaln_proj.forward",
                    _make_adaln_forward(
                        base,
                        sd[f"h3_pdd.adaln.blocks.{i}.lora_A"],
                        sd[f"h3_pdd.adaln.blocks.{i}.lora_B"],
                        grid, table, strength))
                adaln_installed += 1

        # What the file declares against what actually reached the model.
        # These were never compared, and a prefix that matched neither form
        # reported 0 modules while the install loop quietly did not run: four
        # arms rendered with the backbone and heads and NO modulation update,
        # which looks like a plausible render and is a different experiment.
        declared_adaln = int(meta.get("adaln_modules") or 0)
        if declared_adaln and adaln_installed != declared_adaln:
            raise RuntimeError(
                f"{lora_name} declares {declared_adaln} adaln modules but "
                f"{adaln_installed} reached the model. A PDD arm without its "
                f"modulation update renders and looks entirely normal.")

        # Embeddings at EVERY point of the published grid, built once at load
        # from the model's own arithmetic. Which subset are block boundaries is
        # a property of the schedule and is not known yet; the tracker indexes
        # this by the knots it derives. The two streams run different shifts, so
        # they get different times and therefore different embeddings -- but the
        # same grid INDICES, because undoing each shift lands on the one grid.
        grid_t_v = pdd_time_grid(shift_v, num_steps)
        grid_t_a = pdd_time_grid(shift_a, num_steps)
        tracker = _StepTracker(
            boundary_embeddings(grid_t_v, step_table,
                                None if pruned else dm.time_embedder),
            boundary_embeddings(grid_t_a, step_table,
                                None if pruned else dm.time_embedder),
            grid_t_v, grid_t_a, shift_v, num_steps, forced_nfe, file_nfe,
            lora_name)

        final_layer = m.get_model_object("diffusion_model.final_layer")
        # strength 0 installs NOTHING on the head path. Interpolating to the
        # base head would be arithmetically identical, but it would still route
        # the projection through this module's `F.linear` instead of the
        # checkpoint's own `operations.Linear`, which owns the casting and
        # offload handling. "Exactly the base model" has to mean the base
        # model's own code, or the claim is only nearly true and the control
        # arm is only nearly a control.
        if patch_heads and strength != 0.0:
            # Refuse to stack, rather than clobber. `add_object_patch` is
            # last-writer-wins per key, and the head swaps live on separate
            # keys from the bookkeeping wrapper, so two implementations both
            # replacing the output projections produce a plausible wrong render
            # with nothing said. Chaining would not help: two things cannot both
            # own `video_out`.
            #
            # The always-reachable case is two of THIS node in one chain, which
            # needs no other pack installed. Beyond that, at least two other
            # ComfyUI implementations patch the same attribute for their own
            # PDD artifact families -- `ComfyUI-MiniMaxH3-PDD-Mamad8` and
            # `silveroxides/ComfyUI-UtilsCollection` -- so the collision is a
            # property of the patch point rather than of what happens to be in
            # `custom_nodes/` today. Guard adopted from the latter.
            taken = head_patch_clash(m.object_patches)
            if taken:
                raise RuntimeError(
                    f"{lora_name}: something upstream in this graph has already "
                    f"patched {', '.join(taken)}. Two things cannot own H3's "
                    f"output heads -- the second silently wins and the render "
                    f"looks entirely normal. Remove the other PDD or head-swap "
                    f"node, or set patch_heads=False here to run this one as "
                    f"the backbone-and-adaln arm.")
            m.add_object_patch(
                "diffusion_model.final_layer.forward",
                _make_final_layer_forward(final_layer.forward, tracker))
            # The one patch that is not about the heads: `transformer_options`
            # reaches the model's forward and no further, and the block extents
            # live in it. Chained onto whatever forward is already installed.
            m.add_object_patch(
                "diffusion_model.forward",
                _make_capture_forward(
                    m.get_model_object("diffusion_model.forward"), tracker))
        for stream, out_name in (
                (("video", "video_out"), ("audio", "audio_out"))
                if (patch_heads and strength != 0.0) else ()):
            live = getattr(final_layer, out_name)
            base_w = live.weight.detach().to(torch.float32).cpu()
            base_b = live.bias.detach().to(torch.float32).cpu()
            # The bank goes in whole; fusion happens per block on first use.
            # The paper's section 3.1 asks for one fused linear per block rather
            # than an enlarged final layer, which this satisfies -- it just
            # cannot know WHICH blocks until the sampler names them, and a
            # render visits at most `nfe` of them.
            bank_w = sd.get(f"h3_pdd.bank.{stream}.weight")
            if bank_w is None:
                raise RuntimeError(
                    f"{lora_name} carries no per-interval head bank. It was "
                    f"converted before the bank was stored; reconvert with "
                    f"bench/convert_pdd_lora.py.")
            m.add_object_patch(
                f"diffusion_model.final_layer.{out_name}.forward",
                _make_head_forward(
                    _FusedHeads(bank_w, sd[f"h3_pdd.bank.{stream}.bias"],
                                base_w, base_b,
                                shift_v if stream == "video" else shift_a,
                                num_steps, strength),
                    tracker, stream))

        # The step count is deliberately NOT in this line. It is not known here
        # -- the scheduler is downstream -- and printing the file's own count
        # would read as a statement about this render. `_StepTracker._adopt`
        # logs it once, with the block widths, when the schedule arrives.
        logger.info(
            "[h3-pdd] %s at strength %.3f: %d weight patches, "
            "%d adaln %s, %s (%d-point grid, shifts %g/%g). Base is %s.",
            lora_name, strength, len(loaded), adaln_installed,
            ("baked into the curve basis, applied as weight patches"
             if baked is not None else
             "re-injected at run time (pruned base, no bake in this file)"
             if pruned else
             "applied as weight patches (unpruned base)"),
            # Branches on the INSTALL condition, not on `patch_heads` alone.
            # At strength 0.0 nothing is installed -- that is the documented
            # "exactly the base model" control -- and this line was claiming the
            # heads were patched, while being the only runtime evidence of which
            # arm ran.
            "heads fused per block from the schedule"
            if (patch_heads and strength != 0.0) else
            "HEADS NOT PATCHED (control arm: the checkpoint's own heads)"
            if patch_heads else
            "HEADS NOT PATCHED (patch_heads off)",
            num_steps, shift_v, shift_a,
            "pruned/curve-form" if pruned else "full-width")
        return io.NodeOutput(m, sigmas)
