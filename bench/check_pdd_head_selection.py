#!/usr/bin/env python3
"""The PDD node's runtime guards do what they claim, on real artifacts.

Two subjects, both about `pdd_lora.py` deciding something at load or step time
that nothing downstream would contradict.

## 1. A PDD render selects the fused head its sampling step was fused for.

## The escaped defect that earns this check

On 2026-08-26 four PDD arms rendered at 1344x768 with every gate green --
`check_distill_settings` graded the shift and step count, `check_lora_alpha`
graded the scale, `check_attention_defaults` graded the chain, the node logged
208 backbone modules patched and 50 adaln re-injected, and its own
boundary-residual warning stayed silent through all four. The clips were
wrong anyway.

`_StepTracker` recovers the sampling time by a nearest-row lookup against the
1025-row `adaln_t_table`, which quantises `t` to about 1e-3. A `t` sitting
exactly ON a block boundary comes back a fraction BELOW it, and the interval
membership that followed then returned the previous block. Measured against
the real table and the real 8-step shift-12 schedule, the heads came out

    video  [0, 0, 2, 3, 4, 5, 6, 6]   instead of   [0, 1, 2, 3, 4, 5, 6, 7]
    audio  [0, 1, 1, 3, 4, 5, 6, 7]

Wrong at step 1 and at step 7 -- step 7 being the largest jump in this
schedule and the place the fused heads differ most from each other.

**Nothing could have caught it.** The recovered `t` was correct to 4e-5, so
the residual warning is silent by construction; the render completes; the
output is merely wrong. It is the shape `docs/checks.md` calls a
silent-success, and the only thing that found it was driving the tracker with
real inputs instead of reading the code.

**The snap tolerance the escape above describes is gone**, and this section
described it until 2026-08-27. `_StepTracker` now matches the `nfe + 1`
boundary embeddings directly, so there is no recovered `t` to quantise and
selection needs no tolerance at all; the two cases about snapping and about
`step_for_t`'s membership fallback went with the code they graded. What
replaced them is wider, not narrower -- every step count the grid divides by,
rather than the one the bug was found at.

## What this asserts, i.e. what breaks if a case is deleted

  every legal nfe      the tracker, the REAL `adaln_t_table` read off a
                       shipped checkpoint, and the real boundaries select
                       heads 0..n-1 in order at 16, 8, 4 and 2 evaluations,
                       for video AND audio. The range is the point: one file
                       serves every divisor by fusing at load, and a selector
                       only ever seen at one block size is not evidence about
                       the others. Audio matters separately because it runs
                       its own shift, so a fix that only lines the video
                       stream up would pass a video-only case
  off schedule warns   a run at a step count the file was not fused for sets
                       `warned`. Selection degrades to the nearest boundary by
                       design, so this warning is the only thing separating a
                       deliberate arm from a misconfigured one
  both tolerances      `TABLE_TOLERANCE` and `PARTITION_TOLERANCE` each sit
                       between the noise below them (a bf16 cast) and the
                       signal above (the other partition). Both were set by
                       hand and one shipped wrong
  arity transparent    the `final_layer.forward` object patch accepts today's
                       four arguments and the seven Comfy-Org/ComfyUI#15908
                       introduces, forwarding the extras verbatim. An object
                       patch REPLACES the method, so a pinned parameter list
                       is a TypeError on step 1 the day core widens it

Needs a checkpoint on disk and ComfyUI importable for `safetensors` for the
first three; the arity cases need neither and always run. No CUDA, no server,
no model load -- it reads one small buffer out of a header.

Exit codes: 0 all cases passed, 1 a case failed, 2 passed but a control was
skipped (no checkpoint on disk to read a real table from).

    python bench/check_pdd_head_selection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))          # ComfyUI root
sys.path.insert(0, str(HERE.parent))              # this repo

import pdd_math as M                              # noqa: E402
import pdd_lora as P                              # noqa: E402

#: Any pruned checkpoint carries the curve table; they are what a PDD arm runs
#: against. Named rather than globbed so a missing file SKIPS loudly instead of
#: quietly grading nothing.
CHECKPOINTS = ("minimax_h3_ref2va_pruned_int8_convrot.safetensors",
               "minimax_h3_fl2va_pruned_int8_convrot.safetensors")

failures: list[str] = []
skipped: list[str] = []


def check(name, fn):
    try:
        detail = fn()
    except AssertionError as exc:
        failures.append(name)
        print(f"  FAIL  {name}: {exc}")
    else:
        print(f"  ok    {name}" + (f"   {detail}" if detail else ""))


def real_table():
    """`adaln_t_table` from a checkpoint on disk, or None."""
    from safetensors import safe_open
    root = HERE.parents[2] / "models" / "diffusion_models"
    for name in CHECKPOINTS:
        for candidate in (root / name, root / "diffusion_models" / name):
            if not candidate.exists():
                continue
            with safe_open(candidate, framework="pt") as f:
                if "adaln_t_table" in set(f.keys()):
                    return f.get_tensor("adaln_t_table").float(), name
    return None, None


def time_shift_sigma(sigma, from_shift, to_shift):
    """`comfy/ldm/minimax/model.py::time_shift_sigma`, restated.

    The audio stream's time is the video sigma carried to the audio shift, so
    the audio boundaries only line up if this matches upstream. Restated rather
    than imported because importing the model module pulls in CUDA-adjacent
    machinery this check deliberately does without.
    """
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def sampler_sigmas(steps, shift=None):
    """The sigmas `BasicScheduler` hands the sampler, restated.

    A flow schedule is uniform in base sigma with the model's shift applied
    pointwise, which is `1 - pdd_time_grid`. This is the input a render puts in
    `transformer_options["sample_sigmas"]`, and driving the tracker with it is
    what makes these cases exercise the DERIVATION rather than a step count
    handed in by the test.
    """
    return 1.0 - M.pdd_time_grid(SHIFT_V if shift is None else shift, steps)


def drive(table, steps, sample_sigmas=None, forced_nfe=0):
    """Run the real tracker over one schedule. Returns (blocks_v, blocks_a, tracker).

    Goes in through `observe`, so the block boundaries come out of the sigma
    schedule the same way they do in a render. Nothing here tells the tracker
    how many steps there are.
    """
    grid_t_v = M.pdd_time_grid(SHIFT_V, NUM_STEPS)
    grid_t_a = M.pdd_time_grid(SHIFT_A, NUM_STEPS)
    tracker = P._StepTracker(
        P.boundary_embeddings(grid_t_v, table),
        P.boundary_embeddings(grid_t_a, table),
        grid_t_v, grid_t_a, SHIFT_V, NUM_STEPS, forced_nfe, NFE,
        f"steps={steps}")
    tracker.observe(sampler_sigmas(steps) if sample_sigmas is None
                    else sample_sigmas)
    rows = table.shape[0]

    def emb(t):
        pos = min(max(t, 0.0), 1.0) * (rows - 1)
        i0 = min(int(pos), rows - 2)
        return torch.lerp(table[i0], table[i0 + 1], pos - i0)

    video, audio = [], []
    for k in tracker.knots[:-1]:
        t_v = float(grid_t_v[k])
        t_a = 1.0 - time_shift_sigma(1.0 - t_v, SHIFT_V, SHIFT_A)
        # The audio grid is the video grid carried to the audio shift. If these
        # ever disagree the audio boundaries are being built two different ways.
        assert abs(t_a - float(grid_t_a[k])) < 1e-9, (
            f"audio time at grid point {k}: {t_a} from the video sigma, "
            f"{float(grid_t_a[k])} from pdd_time_grid")
        tracker.update(torch.stack([emb(t_v), emb(t_a)]), (0, 1, 0), (1, 2, 1))
        video.append(tracker.block_v)
        audio.append(tracker.block_a)
    return video, audio, tracker


SHIFT_V, SHIFT_A = 12.0, 3.0
NUM_STEPS, BLOCK = 32, 4
NFE = NUM_STEPS // BLOCK

print("PDD fused-head selection over a real schedule")

table, source = real_table()
if table is None:
    skipped.append("no checkpoint with an adaln_t_table on disk")
    print("  SKIP  no pruned H3 checkpoint found; the real table is the whole "
          "point of this check and a synthetic one would grade itself")
else:
    bounds_v = M.block_bounds(SHIFT_V, NUM_STEPS, BLOCK)
    bounds_a = M.block_bounds(SHIFT_A, NUM_STEPS, BLOCK)
    want = list(range(NFE))

    def every_legal_nfe():
        """One file, every step count the 32-point grid divides by.

        This is the parallel-decoding claim in the form the node has to get
        right: the same weights decoded at 8, 4, 2 or 16 evaluations, each
        landing on its own block boundaries. The vendor's README reports
        rendering at 8 and 4; the other two come free from the same
        divisibility and are here because a selector that only ever sees one
        block size is not evidence about the others.

        Nothing tells the tracker the step count. It comes out of the sigma
        schedule, so this grades the derivation and the selection together.

        **This is a statement about the SELECTOR, not about the arm.** Passing
        at 2 evaluations means the tracker picks the blocks that partition names;
        it does not mean a 16-interval block is a sensible thing to average into
        one velocity. `silveroxides/ComfyUI-UtilsCollection` restricts block
        sizes to the trained width and twice it for exactly that reason, and
        nothing here has measured who is right. Do not cite this case as
        evidence that every divisor is a usable arm.
        """
        rows = []
        for steps in (n for n in (16, 8, 4, 2) if NUM_STEPS % n == 0):
            video, audio, tracker = drive(table, steps)
            block = NUM_STEPS // steps
            want = [(k * block, (k + 1) * block) for k in range(steps)]
            assert tracker.nfe == steps, (
                f"steps={steps}: derived nfe {tracker.nfe} from a schedule of "
                f"{steps} steps. The step count is read from `sample_sigmas`; "
                f"getting it wrong means every block is wrong.")
            assert video == want, f"steps={steps}: video blocks {video}, want {want}"
            assert audio == want, (
                f"steps={steps}: audio blocks {audio}, want {want}. Audio runs "
                f"shift {SHIFT_A} and is selected independently, so a "
                f"video-only fix passes a video-only case.")
            assert not tracker.warned, f"steps={steps}: on-schedule run warned"
            rows.append(str(steps))
        return f"{', '.join(rows)} steps each tile the grid exactly, both streams"

    def derived_matches_the_old_widget():
        """The schedule-derived knots ARE `block_bounds`, not an approximation.

        The control for replacing a hand-entered `nfe` with a derivation: for
        every count that divides the grid, what the sampler's sigmas produce
        has to equal what the widget produced, or this change moved the render.
        """
        for steps in (16, 8, 4, 2):
            knots = M.schedule_knots(sampler_sigmas(steps), SHIFT_V, NUM_STEPS)
            want = list(range(0, NUM_STEPS + 1, NUM_STEPS // steps))
            assert knots == want, f"steps={steps}: derived {knots}, want {want}"
            # and the times those knots name are the boundaries themselves
            grid = M.pdd_time_grid(SHIFT_V, NUM_STEPS)
            assert torch.equal(grid[knots],
                               M.block_bounds(SHIFT_V, NUM_STEPS, NUM_STEPS // steps)), (
                f"steps={steps}: the derived boundary TIMES differ from "
                f"block_bounds, so the embeddings matched against would differ")
        return "16, 8, 4, 2 derive exactly the boundaries the widget computed"

    def uneven_schedule_is_reported():
        """A step count that does not divide the grid still names blocks.

        5 steps over 32 intervals cannot tile it. The old node refused at load;
        this one takes the spans the sampler actually asks for, which is what
        makes `nfe` deletable -- but a partial block is off the distribution the
        heads were distilled on, so it has to say so rather than proceed
        quietly.
        """
        video, audio, tracker = drive(table, 5)
        widths = [b - a for a, b in zip(tracker.knots, tracker.knots[1:])]
        assert tracker.knots[0] == 0 and tracker.knots[-1] == NUM_STEPS, (
            f"an uneven schedule must still span the whole grid: {tracker.knots}")
        assert len(set(widths)) > 1, (
            f"5 steps over {NUM_STEPS} intervals cannot be uniform; got {widths}")
        assert video == list(zip(tracker.knots, tracker.knots[1:])), (
            f"uneven blocks selected out of order: {video}")
        assert sum(widths) == NUM_STEPS, (
            f"the blocks do not tile the grid: {widths} sums to {sum(widths)}")
        return f"5 steps -> knots {tracker.knots}, widths {widths}, contiguous"

    def truncated_schedule_starts_late():
        """`denoise < 1.0` starts partway down the trajectory.

        The widget could not express this at all: it named a COUNT, and the
        node assumed a full trajectory from it. Reading the sigmas means a
        partial-denoise render selects the heads for the part it actually
        runs.
        """
        full = sampler_sigmas(NFE)
        partial = full[len(full) // 2:]          # the tail half of the schedule
        knots = M.schedule_knots(partial, SHIFT_V, NUM_STEPS)
        assert knots[0] > 0, (
            f"a truncated schedule must not start at grid point 0: {knots}")
        assert knots[-1] == NUM_STEPS, (
            f"a truncated schedule still ends at the trajectory's end: {knots}")
        return f"a half schedule derives knots {knots}, starting at {knots[0]}"

    def off_schedule_warns():
        """Evaluated at a time this render's schedule does not contain."""
        _, _, tracker = drive(table, NFE)
        rows = table.shape[0]
        # Times halfway between this schedule's boundaries: on the trajectory,
        # but not points the sampler said it would visit.
        for a, b in zip(tracker.knots, tracker.knots[1:]):
            t = float(M.pdd_time_grid(SHIFT_V, NUM_STEPS)[(a + b) // 2])
            pos = min(max(t, 0.0), 1.0) * (rows - 1)
            i0 = min(int(pos), rows - 2)
            e = torch.lerp(table[i0], table[i0 + 1], pos - i0)
            tracker.update(torch.stack([e, e]), (0, 1, 0), (1, 2, 1))
        assert tracker.warned, (
            "a forward at a time the schedule does not contain did not warn. "
            "Selection degrades to the nearest boundary by design, so this "
            "warning is the only thing that distinguishes a sampler evaluating "
            "off its own grid from one on it.")
        return "a forward between two scheduled sigmas is reported"

    def override_ignores_the_schedule():
        """`nfe` non-zero forces uniform blocks whatever the sampler says.

        The escape hatch has to actually override, or it is a widget that
        silently does nothing -- which is worse than not having one.
        """
        video, _, tracker = drive(table, 8, forced_nfe=4)
        assert tracker.nfe == 4, (
            f"nfe=4 against an 8-step schedule derived {tracker.nfe}; the "
            f"override did not take")
        assert video[0] == (0, 8), (
            f"forced blocks should be width 8; first is {video[0]}")
        return "nfe=4 against an 8-step schedule forces width-8 blocks"

    def a_new_schedule_is_re_derived():
        """Changing the SCHEDULER re-derives the blocks, not just changing steps.

        The escaped defect this earns: `observe` cached on
        `(len, first, last)`, and at 8 steps `simple`, `beta`, `kl_optimal` and
        `linear_quadratic` all produce `(9, 1.0, 0.0)` while deriving four
        different knot sets. `BasicScheduler` is DOWNSTREAM of the node, so
        changing its scheduler does not re-execute it -- the cached ModelPatcher
        keeps this tracker, `observe` sees new sigmas, matches the stale key and
        returns, and every block decodes an interval the sampler never visits.

        Driven with two schedules that COLLIDE under the old key, so this is red
        against the version it replaced rather than merely green against the new
        one. Uses hand-built sigma vectors rather than `comfy.samplers` so the
        case needs no model-sampling object.
        """
        grid_t_v = M.pdd_time_grid(SHIFT_V, NUM_STEPS)
        grid_t_a = M.pdd_time_grid(SHIFT_A, NUM_STEPS)
        tracker = P._StepTracker(
            P.boundary_embeddings(grid_t_v, table),
            P.boundary_embeddings(grid_t_a, table),
            grid_t_v, grid_t_a, SHIFT_V, NUM_STEPS, 0, NFE, "collide")
        a = sampler_sigmas(8)                       # the uniform 8-step schedule
        # Same length, same endpoints, different interior -- the shape that
        # collided. Knots land on [0, 2, 6, 11, ...] rather than [0, 4, 8, ...].
        b = torch.tensor([1.0] + [float(x) for x in
                                  M.shifted_sigma(SHIFT_V, torch.tensor(
                                      [0.94, 0.82, 0.66, 0.5, 0.34, 0.18, 0.06],
                                      dtype=torch.float64))] + [0.0])
        assert (a.numel(), float(a[0]), float(a[-1])) == \
               (b.numel(), float(b[0]), float(b[-1])), (
            "this case is only meaningful if the two schedules collide under "
            "the retired key; they no longer do, so rebuild the pair")
        tracker.observe(a)
        first = list(tracker.knots)
        tracker.observe(b)
        second = list(tracker.knots)
        assert first != second, (
            f"both schedules derived {first}. `observe` did not re-derive on a "
            f"schedule it had already seen the shape of, which is the caching "
            f"bug: same length and endpoints, different interior.")
        return f"{first} then {second}, re-derived on a colliding shape"

    check("every legal step count tiles the grid", every_legal_nfe)
    check("a new schedule is re-derived", a_new_schedule_is_re_derived)
    check("derived knots equal the widget's boundaries", derived_matches_the_old_widget)
    check("an uneven schedule is taken and reported", uneven_schedule_is_reported)
    check("a truncated schedule starts partway down", truncated_schedule_starts_late)
    check("evaluating off the schedule warns", off_schedule_warns)
    check("the nfe override overrides", override_ignores_the_schedule)


# --- 2. the two tolerances, against the noise each has to straddle ---------
# Both were set by hand from a measurement taken once, and one of them was set
# WRONG: TABLE_TOLERANCE started at 1e-3, below the 1.6e-3 a bf16 cast of the
# table costs, so every correct bake would have been rejected into the slow
# injection path -- silently, because that path is correct. The other,
# PARTITION_TOLERANCE, replaced a sha256 that fired on the RIGHT checkpoint for
# the same reason: an exact test against a value the loader is allowed to cast.
#
# A tolerance is only meaningful between two numbers. These cases pin both.
def _tolerances():
    import torch as _t
    from safetensors import safe_open
    root = HERE.parents[2] / "models" / "diffusion_models"
    want = ("minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    got = {}
    for name in want:
        for cand in (root / name, root / "diffusion_models" / name):
            if cand.exists():
                with safe_open(cand, framework="pt") as f:
                    got[name] = (f.get_tensor("adaln_t_table").float(),
                                 f.get_tensor("final_layer.video_out.weight").float())
                break
    return got


_ck = _tolerances()
if len(_ck) < 2:
    skipped.append("both pruned partitions needed to bound the tolerances")
    print("  SKIP  need both partitions on disk; a tolerance graded against "
          "only one side is not graded")
else:
    import torch as _t
    (_ref_tab, _ref_vo), (_fl_tab, _fl_vo) = (_ck[k] for k in _ck)

    def _rel(a, b):
        return float((a - b).norm() / b.norm())

    def table_tolerance_straddles():
        cast = _rel(_ref_tab.to(_t.bfloat16).float(), _ref_tab)
        cross = _rel(_fl_tab, _ref_tab)
        assert cast <= P.TABLE_TOLERANCE, (
            f"a bf16 cast of the curve table is {cast:.5f} away, above "
            f"TABLE_TOLERANCE={P.TABLE_TOLERANCE}. Every correct bake would be "
            f"rejected into the runtime injection -- slower, and silent, "
            f"because the injection is correct. This is the value it shipped "
            f"with for an hour on 2026-08-26.")
        assert cross > P.TABLE_TOLERANCE, (
            f"the other partition's table is {cross:.5f} away, within "
            f"TABLE_TOLERANCE={P.TABLE_TOLERANCE}. A bake solved against the "
            f"wrong basis would be accepted, and it is 0.02 wrong at runtime "
            f"against 0.0001 for the right one -- which a fit residual cannot "
            f"detect, so this comparison is the only thing standing there.")
        return f"cast {cast:.5f} < {P.TABLE_TOLERANCE} < cross-partition {cross:.5f}"

    def partition_tolerance_straddles():
        cast = _rel(_ref_vo.to(_t.bfloat16).float(), _ref_vo)
        cross = _rel(_fl_vo, _ref_vo)
        assert cast <= P.PARTITION_TOLERANCE, (
            f"a bf16 cast of final_layer.video_out is {cast:.5f} away, above "
            f"PARTITION_TOLERANCE={P.PARTITION_TOLERANCE}. The loader casts on "
            f"load, so this rejects the CORRECT checkpoint -- which is what the "
            f"sha256 this replaced actually did, on the first real render.")
        assert cross > P.PARTITION_TOLERANCE, (
            f"the other partition is {cross:.5f} away, within "
            f"PARTITION_TOLERANCE={P.PARTITION_TOLERANCE}. fl2va and ref2va "
            f"ship identical key sets, so a mismatched pair renders without one "
            f"unmatched key and is merely wrong.")
        return f"cast {cast:.5f} < {P.PARTITION_TOLERANCE} < cross-partition {cross:.5f}"

    check("table tolerance sits between a cast and a partition swap",
          table_tolerance_straddles)
    check("partition tolerance sits between a cast and a partition swap",
          partition_tolerance_straddles)

# --- 3. the final_layer patch survives core widening the signature ---------
# Comfy-Org/ComfyUI#15908 (open 2026-08-27) teaches core the same mechanism and
# widens `FinalLayer.forward` to seven parameters. Our object patch REPLACES
# that method, so a four-parameter replacement drops the three core now passes
# and raises TypeError on the first sampling step of every PDD render.
#
# This is the cheap half of a forward-compatibility question whose expensive
# half cannot be run here: nothing on this box can merge that PR and render.
# What it does assert is that the patch is arity-transparent in BOTH
# directions, which is the part that is ours to get right. Delete the `*args`
# from `_make_final_layer_forward` and the second case goes red.
print()
print("the final_layer patch is arity-transparent")


def _arity():
    seen = {}

    class _Tracker:
        def update(self, t_emb, video_seg, audio_seg):
            seen["update"] = (t_emb, video_seg, audio_seg)

    def base(x, t_emb, video_seg, audio_seg, *extra, **kw):
        seen["extra"] = extra
        seen["kw"] = kw
        return "stock output"

    fwd = P._make_final_layer_forward(base, _Tracker())
    return fwd, seen


def _call(fwd, *args):
    """Call the patch, turning an arity mismatch into a FAIL rather than a crash.

    The failure this section exists to catch IS a `TypeError`, and `check()`
    catches only `AssertionError` -- so without this the deliberate violation
    aborts the run with a traceback instead of reporting a named case, and the
    summary and exit code never happen. Found by running the violation, which
    is the only reason it is here.
    """
    try:
        return fwd(*args)
    except TypeError as exc:
        raise AssertionError(
            f"the patched final_layer rejected a {len(args)}-argument call: "
            f"{exc}. An object patch replaces the method outright, so its "
            f"parameter list has to accept whatever `comfy/ldm/minimax/"
            f"model.py` passes today AND whatever #15908 makes it pass, and "
            f"forward the rest untouched.") from None


def todays_core():
    """Four positional arguments, which is what `model.py` passes today."""
    fwd, seen = _arity()
    out = _call(fwd, "x", "t_emb", (0, 1, 0), (1, 2, 0))
    assert out == "stock output", f"the stock forward's return was not passed through: {out!r}"
    assert seen["update"] == ("t_emb", (0, 1, 0), (1, 2, 0)), (
        f"the tracker saw {seen['update']!r}; it must see the model's own "
        f"t_emb and both segments, because that is the whole of the selection")
    assert seen["extra"] == (), f"invented arguments for the stock forward: {seen['extra']!r}"
    return "4 args reach the tracker and the stock forward unchanged"


def post_pr_core():
    """Seven, which is what #15908 makes `model.py` pass.

    The three extra are `sigma`, `sample_sigmas` and `shifts`. This does not
    check what they mean -- core owns that -- only that they arrive at the
    stock forward verbatim rather than being dropped or reordered by us.
    """
    fwd, seen = _arity()
    extra = ("sigma", "sample_sigmas", (12.0, 3.0))
    out = _call(fwd, "x", "t_emb", (0, 1, 0), (1, 2, 0), *extra)
    assert out == "stock output", f"the stock forward's return was not passed through: {out!r}"
    assert seen["extra"] == extra, (
        f"core's three new arguments arrived as {seen['extra']!r}, not "
        f"{extra!r}. A patch that drops them leaves the stock forward without "
        f"the sigma schedule it now requires, and the render dies on step 1.")
    assert seen["update"] == ("t_emb", (0, 1, 0), (1, 2, 0)), (
        "widening the signature changed which values the tracker reads")
    return "7 args forward verbatim; selection still reads only the first 4"


def refuses_to_stack():
    """A second owner of the output heads is refused, not silently clobbered.

    `add_object_patch` is last-writer-wins per key and the two head swaps live
    on their own keys, so chaining the bookkeeping wrapper would not save it --
    two things cannot both own `video_out`.

    The case needing no other pack installed is two of THIS node in one chain.
    At least two other ComfyUI implementations patch the same attribute for
    their own PDD artifact families, so the collision is a property of the patch
    point rather than of what is in `custom_nodes/` on any given day.
    """
    assert P.head_patch_clash({}) == [], (
        "a clean model reported a clash, which would refuse every render")
    assert P.head_patch_clash({"diffusion_model.blocks.0.attn.forward": 1}) == [], (
        "an unrelated forward patch reported a clash. Sage and Sol patch block "
        "attention on every shipped graph, so this would refuse all of them.")
    for key in P.HEAD_PATCH_KEYS:
        assert P.head_patch_clash({key: 1}) == [key], (
            f"{key} was already taken and went unreported")
    both = P.head_patch_clash({k: 1 for k in P.HEAD_PATCH_KEYS})
    assert both == list(P.HEAD_PATCH_KEYS), f"partial report: {both}"
    return f"{len(P.HEAD_PATCH_KEYS)} head keys reported, attention patches ignored"




# --- the tracker's state cannot outlive the schedule it describes ------------

#: `_key` is the ONLY attribute allowed to survive `_adopt`, and it has to.
#: It is the schedule's identity, the thing `observe` compares against to decide
#: whether to re-adopt at all. Reset it in `_adopt` and every forward re-adopts.
#: Named with its reason rather than pattern-matched, so an attribute that stops
#: needing the exemption fails instead of being quietly covered.
STATE_EXEMPT = {"_key": "the schedule identity observe compares against"}


def no_state_outlives_its_schedule():
    """Every per-render field on `_StepTracker` is reset when the schedule is.

    ## The two escaped instances that earn this, both on 2026-08-28

    `_StepTracker` is a mutable object held by the ModelPatcher, and ComfyUI's
    execution cache keeps that across prompts in a session. So any attribute
    carrying per-render state silently persists into the NEXT render, and both
    times it did, the symptom was a guard going quiet rather than a wrong
    picture -- which this repo already calls the worst shape a check can take.

      `_shift_checked`   a one-shot latch on the shift guard. A passing
                         shift-12 render set it; the shift-6 graph queued next
                         reused the cached patcher, skipped the check entirely
                         and rendered to completion. Caught by driving the two
                         orderings on the card, not by any gate.
      `warned`           the boundary warning's latch, set in `__init__` and
                         never reset. Two arms with a bit-identical truncated
                         sigma vector, queued back to back: the first warned,
                         the second was silent.

    Nothing in the suite could have caught either. What catches them is the
    shape rather than the instance: an attribute assigned somewhere other than
    `__init__` is per-render by construction, and `_adopt` is the one place a
    new schedule is taken. So the invariant is mechanical -- assigned outside
    `__init__` implies assigned in `_adopt` -- and this asserts it statically,
    with no model, no CUDA and no server.

    **Deliberately NOT a runtime probe.** Driving two schedules through a live
    tracker would grade the fields that exist today; parsing the class catches
    the field somebody adds next year, which is the one that will actually bite.

    What breaks if this case is deleted: a new per-render attribute lands on the
    tracker without a reset, and the next graph in the same server session
    inherits it. There is no symptom at the time -- the render completes.
    """
    import ast
    src = (HERE.parent / "pdd_lora.py").read_text()
    cls = next(n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.ClassDef) and n.name == "_StepTracker")
    where: dict[str, set[str]] = {}
    for fn in [n for n in cls.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for node in ast.walk(fn):
            # AugAssign and NamedExpr are collected too. Walking only
            # Assign/AnnAssign made `self.calls += 1` invisible -- and a
            # per-render COUNTER is the single most likely field somebody adds
            # to this class, which is the case this whole guard exists for.
            # `for self.x in ...` targets come through `ast.For` the same way.
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For)):
                targets = [node.target]
            elif isinstance(node, ast.NamedExpr):
                targets = [node.target]
            else:
                targets = []
            for tgt in targets:
                if (isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"):
                    where.setdefault(tgt.attr, set()).add(fn.name)
    assert where, "parsed no attributes off _StepTracker; this case is inert"
    assert "_adopt" in {m for ms in where.values() for m in ms}, (
        "_StepTracker has no `_adopt`; this case is keyed on it and has lost "
        "its subject rather than passing")

    leaked = sorted(
        a for a, ms in where.items()
        if ms != {"__init__"} and "_adopt" not in ms and a not in STATE_EXEMPT)
    assert not leaked, (
        f"per-render state on _StepTracker that `_adopt` does not reset: "
        f"{leaked}. The tracker lives in the ModelPatcher and ComfyUI's cache "
        f"keeps it across prompts, so each of these carries into the next "
        f"render in the same session. Reset it in `_adopt`, or add it to "
        f"STATE_EXEMPT with the reason it must survive.")

    stale = sorted(a for a in STATE_EXEMPT
                   if a not in where or "_adopt" in where.get(a, ()))
    assert not stale, (
        f"STATE_EXEMPT names {stale}, which no longer needs the exemption -- "
        f"either gone from the class or now reset by `_adopt`. An exemption "
        f"that is not necessary covers the next real leak.")
    per_render = sorted(a for a, ms in where.items() if "_adopt" in ms)
    return (f"{len(per_render)} per-render field(s) reset by _adopt, "
            f"{len(STATE_EXEMPT)} exempt and still necessary")


check("today's core signature", todays_core)
check("the signature #15908 introduces", post_pr_core)
check("a second owner of the heads is refused", refuses_to_stack)
check("no state outlives its schedule", no_state_outlives_its_schedule)

print()
if failures:
    print(f"{len(failures)} failure(s): {', '.join(failures)}")
    sys.exit(1)
if skipped:
    print(f"passed, but SKIPPED: {'; '.join(skipped)}")
    sys.exit(2)
print("all ok -- every sampling step decodes the interval it was fused for")
