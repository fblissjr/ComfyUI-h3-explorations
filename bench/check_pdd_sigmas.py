#!/usr/bin/env python3
"""The PDD node's SIGMAS output is the block boundaries, and the graphs use it.

## Why this exists

`MiniMaxH3PDDLoRA` gained a SIGMAS output so the sampler consumes the schedule
the heads were fused for, instead of the node reconstructing whatever schedule
`BasicScheduler` happened to produce. Two claims came with that rewiring, both
load-bearing and neither previously assertable:

  1. **It is numerically inert.** The emitted sigmas are bit-identical to
     `BasicScheduler(simple, N)` at every step count the shipped graphs run, so
     the rewiring removes ways to be wrong without moving any render.
  2. **Off-grid stops being expressible.** Feed the output back through the
     node's own `schedule_knots` and the knots are the uniform block boundaries
     -- which is the property the head selection actually depends on.

`docs/checks.md` records that a requirement with no assertion behind it is not
a control. Both claims above are prose in `pdd_lora.py` and
`workflows/build_workflows.py` until this file grades them.

## The control, and why it is a control

Claim 1 is graded against **ComfyUI's own `comfy.samplers.calculate_sigmas`
over `ModelSamplingAV`** -- the object a graph actually samples through, built
the same two lines `MiniMaxH3SigmaShift.execute` runs. Not against a value
computed here. If our closed form and ComfyUI's discrete table ever disagree at
a shipped step count, that is the thing worth knowing, and neither side is
allowed to be the other's definition of correct.

**Where they legitimately differ, stated rather than tolerated.** `simple`
reads a 1,000-entry table at truncated indices, so it reproduces the closed
form EXACTLY when `1000 % steps == 0` and quantises otherwise. `EXACT_STEPS`
below is that set intersected with the divisors of the 32-point grid; at 16
steps the two sit ~2e-3 apart and the closed form is the more correct of the
two. `exactness_regime_holds` asserts that precondition instead of assuming it,
the same shape `check_distill_grid.py::divisor_regime_holds` uses -- so if a
graph ever ships at a count outside the exact regime, this goes red rather than
quietly widening.

## What this asserts, i.e. what breaks if a case is deleted

  emitted is simple      at 2, 4 and 8 steps on shift 12 and shift 6, the
                         emitted vector equals ComfyUI's `simple` to
                         `torch.equal`. This is the whole inertness claim
  exactness regime       every step count graded above really is in `simple`'s
                         exact regime, so the equality above is a fact about
                         our arithmetic and not about a shared rounding
  knots round trip       feeding the emitted vector to `schedule_knots` returns
                         the uniform boundaries. The sampler stepping where the
                         heads were fused is the ONLY thing the head selection
                         needs, and it is a different claim from matching
                         `simple`
  non-divisor raises     an explicit `steps` that does not tile the grid is
                         refused. At such a count no on-grid schedule exists,
                         so emitting anything would be emitting something off it
  zero is inert          `steps=0` never raises and falls back to the file's
                         own count, whatever it is asked for. This is what
                         keeps a deliberately off-grid arm -- one driving
                         BasicScheduler at a non-dividing count with SIGMAS
                         unwired -- working exactly as before
  graphs consume it      every shipped non-split PDD graph wires its sampler's
                         `sigmas` from the PDD node, carries no BasicScheduler,
                         and names a step count that divides the grid. Without
                         this the node could emit perfect sigmas that nothing
                         reads

Needs ComfyUI importable for `comfy.samplers`; no CUDA, no server, no model
load, no checkpoint on disk. The arithmetic cases need no graphs and the graph
case needs no ComfyUI, so a partial environment still grades something.

Exit codes: 0 all cases passed, 1 a case failed.

    python bench/check_pdd_sigmas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
COMFY = HERE.parents[2]
sys.path.insert(0, str(COMFY))                    # ComfyUI root
sys.path.insert(0, str(HERE.parent))              # this repo
sys.path.insert(0, str(HERE.parent / "workflows"))

import pdd_math as M                              # noqa: E402
from h3_config import (graph_paths, graph_schedule,  # noqa: E402
                       resolve_widget)

WORKFLOWS = HERE.parent / "workflows"

#: The published grid every PDD file declares. A CONSTANT -- nothing here reads
#: `pdd_num_steps`, and an earlier comment claimed it did. A file converted at
#: another grid would be graded against 32 regardless, which is safe only
#: because every artifact this repo ships declares 32.
NUM_STEPS = 32

#: The block width the shipped banks were distilled at, and the floor of the
#: envelope `envelope_partition` tiles within. Both artifacts declare 4 in
#: `pdd_block_size`; a file declaring otherwise would tile differently and this
#: check would grade the wrong envelope, which is why the node reads it from
#: metadata rather than from here.
TRAINED = 4

#: The video shift the emitted schedule is built on.
SHIFT_VIDEO = 12.0

#: Step counts where `simple` reproduces the closed form exactly: divisors of
#: the 32-point grid that also divide `simple`'s 1,000-entry table. 16 is a
#: grid divisor and is deliberately NOT here -- 1000 % 16 != 0.
EXACT_STEPS = tuple(n for n in (1, 2, 4, 8, 16, 32)
                    if NUM_STEPS % n == 0 and 1000 % n == 0)

#: Both shifts the shipped graphs run. 12/3 is the checkpoint's own; 11 graphs
#: run 6 for the turbo arms and the identity must not depend on the value.
SHIFTS = (12.0, 6.0)

failures: list[str] = []


def check(name, fn):
    try:
        detail = fn()
    except AssertionError as exc:
        failures.append(name)
        print(f"  FAIL  {name}: {exc}")
    else:
        print(f"  ok    {name}" + (f"   {detail}" if detail else ""))


def emitted(shift: float, steps: int) -> torch.Tensor:
    """What `MiniMaxH3PDDLoRA` puts on its SIGMAS output.

    CALLS the node, rather than restating it. The first version of this
    duplicated the expression, and a review showed the whole file stayed green
    with the `1.0 -` dropped from `pdd_lora` -- the check was grading its own
    copy. `emit_sigmas` was lifted out of `execute` so this can drive the real
    thing, exactly as `resolve_emit_steps` was for the refusal cases.
    """
    from pdd_lora import emit_sigmas
    return emit_sigmas(shift, NUM_STEPS, NUM_STEPS // steps)


def comfy_simple(shift: float, steps: int) -> torch.Tensor:
    """`BasicScheduler(simple, steps)` on a shift-`shift` H3 model.

    The same two lines `MiniMaxH3SigmaShift.execute` runs, so this is the
    object a graph samples through rather than a reimplementation of it.
    """
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import comfy.model_sampling as model_sampling
    import comfy.samplers

    class _Config:
        sampling_settings = {"shift": shift, "audio_shift": 3.0}

    class _ModelSamplingAdvanced(model_sampling.ModelSamplingAV,
                                 model_sampling.CONST):
        pass

    ms = _ModelSamplingAdvanced(_Config())
    ms.set_parameters(shift=shift, audio_shift=3.0)
    return comfy.samplers.calculate_sigmas(ms, "simple", steps).to(torch.float32)


def case_emitted_is_simple():
    worst = []
    for shift in SHIFTS:
        for steps in EXACT_STEPS:
            if steps < 2:
                continue
            got, want = emitted(shift, steps), comfy_simple(shift, steps)
            assert got.shape == want.shape, (
                f"shift {shift} at {steps} steps: emitted {tuple(got.shape)} "
                f"against simple's {tuple(want.shape)}")
            assert torch.equal(got, want), (
                f"shift {shift} at {steps} steps: emitted sigmas are not "
                f"`simple`'s, max {float((got - want).abs().max()):.3e}. The "
                f"rewiring is only safe because these agree; if this is a "
                f"deliberate change, every PDD render moves.")
            worst.append(steps)
    return f"bit-identical at {sorted(set(worst))} steps, shifts {list(SHIFTS)}"


def case_exactness_regime_holds():
    """The equality above must not be resting on a shared rounding.

    If a step count outside `simple`'s exact regime ever reached
    `EXACT_STEPS`, the case above would be comparing two quantised vectors and
    would pass for the wrong reason.
    """
    for steps in EXACT_STEPS:
        assert 1000 % steps == 0, (
            f"{steps} is graded as exact but 1000 % {steps} != 0, so `simple` "
            f"quantises there and the equality case proves nothing")
        assert NUM_STEPS % steps == 0, (
            f"{steps} does not divide the {NUM_STEPS}-point grid")
    # And the excluded one really is excluded, so this is a real partition.
    assert 16 not in EXACT_STEPS, (
        "16 divides the grid but not 1,000; including it would force a "
        "tolerance and destroy the exactness claim for every other count")
    return f"{list(EXACT_STEPS)} exact; 16 correctly partitioned out"


def case_knots_round_trip():
    """Emitted -> `schedule_knots` -> the uniform boundaries.

    This is the property head selection actually depends on, and it is not the
    same claim as matching `simple`: a schedule could match `simple` and still
    be read back as something else if the inverse shift were wrong.
    """
    seen = []
    for shift in SHIFTS:
        for steps in (2, 4, 8, 16):
            width = NUM_STEPS // steps
            knots = M.schedule_knots(emitted(shift, steps), shift, NUM_STEPS)
            want = list(range(0, NUM_STEPS + 1, width))
            assert knots == want, (
                f"shift {shift} at {steps} steps: emitted sigmas read back as "
                f"knots {knots}, not the uniform boundaries {want}. The "
                f"sampler would step somewhere the heads were not fused for.")
            seen.append(steps)
    return f"uniform boundaries recovered at {sorted(set(seen))} steps"


def case_untileable_raises():
    """An explicit `steps` that tiles the grid by NEITHER route is refused.

    Drives `pdd_lora.resolve_emit_steps` itself. The first version of this case
    recomputed the condition and asserted its own arithmetic, which could not
    have failed however the node behaved -- the shape `docs/checks.md` calls a
    check whose input already satisfies the outcome.

    **Narrowed 2026-08-29, and the old form was over-broad.** It asserted that
    every non-divisor is refused, which was the behaviour but not the
    requirement: a non-divisor can still tile the grid unevenly inside the
    trained envelope, and 5, 6 and 7 do. The requirement is that a count
    reaching the grid by neither route is refused, and 3 and 9 are the cases --
    too few blocks to reach 32 at the envelope's ceiling, and too many to reach
    it at the floor. 12 is now covered by `case_envelope_tiling_is_legal`
    instead, which asserts the boundary from the other side.
    """
    from pdd_lora import resolve_emit_steps, envelope_partition
    for bad in (3, 9, 11):
        assert NUM_STEPS % bad, f"{bad} divides {NUM_STEPS}; not a case"
        assert envelope_partition(NUM_STEPS, bad, TRAINED) is None, (
            f"{bad} HAS an envelope tiling; it is not an untileable case")
        try:
            got = resolve_emit_steps(bad, 8, NUM_STEPS, TRAINED)
        except RuntimeError:
            continue
        raise AssertionError(
            f"steps={bad} tiles the {NUM_STEPS}-point grid by neither a "
            f"divisor nor the trained envelope, but was accepted, returning "
            f"{got}. The SIGMAS output would be silently off the grid.")
    return "3, 9, 11 reach the grid by neither route and are all refused"


def case_envelope_tiling_is_legal():
    """A non-divisor the envelope CAN tile is accepted, and tiles exactly.

    The counterpart to the case above, and the reason it had to narrow. Six
    evaluations is the one that matters: no uniform partition of 32 exists, the
    node refused it until 2026-08-29, and the owner had been supplying exactly
    the partition it now emits by hand through `ManualSigmas` the whole time.

    Asserts three things, because accepting the count is the weakest of them:
    the widths tile the grid, every width is inside the trained envelope, and
    the emitted sigmas are the ones the hand-written partition produced.
    """
    import torch
    from pdd_lora import resolve_emit_steps, envelope_partition
    from pdd_math import partition_bounds
    for nfe in (5, 6, 7):
        assert NUM_STEPS % nfe, f"{nfe} divides {NUM_STEPS}; not a case"
        assert resolve_emit_steps(nfe, 8, NUM_STEPS, TRAINED) == nfe
        w = envelope_partition(NUM_STEPS, nfe, TRAINED)
        assert w is not None and sum(w) == NUM_STEPS, f"{nfe}: {w} does not tile"
        assert all(TRAINED <= x <= 2 * TRAINED for x in w), f"{nfe}: {w} leaves the envelope"
        assert len(w) == nfe, f"{nfe}: {w} is not {nfe} blocks"
    six = (1.0 - partition_bounds(SHIFT_VIDEO, NUM_STEPS,
                                  envelope_partition(NUM_STEPS, 6, TRAINED)))
    # IMPORTED, not restated. This used to hold its own literal copy of the
    # vector, so `h3_config.PDD_MANUAL_SIGMAS` could have drifted and this
    # would still have passed -- the "helper the check defines rather than
    # imports" trap in CLAUDE.md, one level out. The config string is what the
    # shipped `..._manual_sigmas` graph actually renders, so that is the thing
    # worth grading.
    from h3_config import PDD_MANUAL_SIGMAS
    hand = torch.tensor([float(x) for x in PDD_MANUAL_SIGMAS.split(",")])
    d = float((six - hand).abs().max())
    assert d < 5e-6, (
        f"the emitted six-block schedule is {d:.2e} from the hand-written "
        f"partition this repo has been rendering: {six.tolist()}")

    # **The 6dp rounding is load-bearing; do not "simplify" it away.** The
    # config string is a 6-decimal rendering of the derivation, not the
    # derivation: they differ by 2.2e-07 on 0.878049 against
    # 0.8780487804878049, which is 3-4 ULP at float32. Substituting the raw
    # derived value would change the emitted sigmas bitwise, and by CLAUDE.md's
    # different-sample rule every `..._manual_sigmas` arm rendered before that
    # substitution would stop being comparable with every one after it. So the
    # config keeps a string and this asserts the string is exactly what the
    # derivation rounds to. `repr(round(v, 6))` is the formatter that
    # round-trips: `%g` renders the 1.0 and 0.0 endpoints as "1" and "0".
    #
    # It is asserted here rather than derived in `h3_config` because that
    # module imports without torch and several checks rely on it -- and
    # `partition_bounds` returns a tensor. Re-deriving the maths in pure Python
    # there would be a second copy of the maths instead of a second copy of the
    # vector, which is not an improvement.
    rendered = ", ".join(repr(round(float(v), 6)) for v in six.tolist())
    assert rendered == PDD_MANUAL_SIGMAS, (
        f"h3_config.PDD_MANUAL_SIGMAS is no longer what the six-block "
        f"partition rounds to.\n  config:  {PDD_MANUAL_SIGMAS}\n  derived: "
        f"{rendered}")
    return ("5, 6, 7 tile inside the envelope; 6 reproduces "
            "h3_config.PDD_MANUAL_SIGMAS exactly at 6dp")


def case_zero_is_inert():
    """`steps=0` never refuses, whatever the file's own count is.

    This is what keeps a deliberately off-grid arm working: one driving
    BasicScheduler at a non-dividing count with SIGMAS unwired must be
    untouched by this input. An earlier version of the node raised
    unconditionally and would have refused a 6-step render that was in flight.
    """
    from pdd_lora import resolve_emit_steps
    for file_nfe in (1, 2, 4, 8, 16, 32):
        got = resolve_emit_steps(0, file_nfe, NUM_STEPS)
        assert got == file_nfe, (
            f"steps=0 with a file declaring pdd_nfe={file_nfe} returned "
            f"{got}; it must defer to the file and never refuse")
    # And the off-grid arm specifically: a non-dividing count on the sampler
    # while this input stays 0 must not raise here at all.
    assert resolve_emit_steps(0, 8, NUM_STEPS) == 8, (
        "steps=0 must be inert even while the sampler runs a count that does "
        "not tile the grid -- that arm is legal and was rendering when this "
        "was written")
    return "0 defers to the file's own count at every legal nfe"


def case_graphs_consume_it():
    """Every shipped non-split PDD graph samples from the PDD node.

    The arithmetic above is worth nothing if the graphs still read a
    BasicScheduler. Walks `graph_paths` rather than globbing, which
    `check_graph_discovery.py` requires.
    """
    graded, problems, manual = [], [], []
    for path in graph_paths(WORKFLOWS):
        if not path.name.endswith("_api.json"):
            continue
        try:
            g = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            problems.append(f"{path.name}: unreadable ({exc})")
            continue
        pdd = [nid for nid, n in g.items()
               if isinstance(n, dict) and n.get("class_type") == "MiniMaxH3PDDLoRA"]
        if not pdd:
            continue
        # Anything that legitimately sits between the node and the sampler.
        # `SplitSigmas` is the two-stage split (keeps BasicScheduler by design);
        # `SplitSigmasDenoise` is what the node's OWN `steps` tooltip prescribes
        # for denoise < 1.0, so a graph following that advice must not go red
        # here for doing what it was told.
        if any(isinstance(n, dict)
               and n.get("class_type") in ("SplitSigmas", "SplitSigmasDenoise")
               for n in g.values()):
            continue
        # A ManualSigmas arm names an explicit NON-UNIFORM partition, which the
        # node's SIGMAS output cannot express -- it emits uniform blocks, so it
        # only reaches counts that divide the grid, and [8,8,4,4,4,4] is six.
        # This is not an exemption for its own sake: `check_distill_settings`
        # grades that arm's vector against the grid instead, asserting every
        # knot lands on a grid point and every width is inside the trained
        # envelope, which is the property this case protects on the others.
        # Without that second grader the skip would be a hole.
        if any(isinstance(n, dict) and n.get("class_type") == "ManualSigmas"
               for n in g.values()):
            manual.append(path.name)
            continue
        graded.append(path.name)
        nid = pdd[0]
        # Through the resolver, not a literal read: the shipped graphs WIRE
        # `steps` from a PrimitiveInt so the count is one visible number on the
        # canvas, and a literal read sees `["61", 0]` and calls a correct graph
        # broken. This is the case `h3_config.resolve_widget` exists for.
        steps = resolve_widget(g, g[nid], "steps").value
        if not isinstance(steps, int) or steps <= 0:
            problems.append(f"{path.name}: PDD node steps={steps!r}, "
                            f"so its SIGMAS output is the file's count, "
                            f"not the arm's")
        elif NUM_STEPS % steps:
            problems.append(f"{path.name}: steps={steps} does not tile the "
                            f"{NUM_STEPS}-point grid")
        sampler = [n for n in g.values()
                   if isinstance(n, dict)
                   and n.get("class_type") == "SamplerCustomAdvanced"]
        for s in sampler:
            src = s["inputs"].get("sigmas")
            if not (isinstance(src, list) and src[0] == nid and src[1] == 1):
                problems.append(f"{path.name}: sampler reads sigmas from "
                                f"{src!r}, not the PDD node's SIGMAS output")
        leftover = [n.get("class_type") for n in g.values()
                    if isinstance(n, dict)
                    and n.get("class_type") == "BasicScheduler"]
        if leftover:
            problems.append(f"{path.name}: still carries a BasicScheduler, so "
                            f"a scheduler and a step count are settable again")
    assert graded, (
        "no PDD graph was graded. Either none ship or discovery missed them; "
        "either way this case is asserting nothing and must not read green.")
    assert not problems, "; ".join(problems[:4])
    tail = (f", {len(manual)} on an explicit ManualSigmas partition "
            f"(graded by check_distill_settings)" if manual else "")
    return f"{len(graded)} PDD graph(s) sample from the node{tail}"


def _checkpoint_default_shift() -> float:
    """The video shift a graph runs with no `MiniMaxH3SigmaShift` in it.

    Read from ComfyUI's own model config, which is where the DiT, the sampler
    and `pdd_lora.py::check_shift` all end up when the node is absent -- rather
    than from `h3_config.SIGMA_SHIFT`, which is this repo's second copy of the
    same numbers and would agree with itself forever.
    """
    from comfy.supported_models import MiniMaxH3
    return float(MiniMaxH3.sampling_settings["shift"])


def case_graph_shift_matches_file():
    """Each PDD graph's shift equals its file's own shift.

    **This is the gap the rewiring opened, and it is real.** The PDD node sits
    UPSTREAM of `MiniMaxH3SigmaShift`, so the schedule it emits is built from
    the shift recorded in its own file, not from the graph's shift widget. When
    `BasicScheduler` owned the schedule it read the shift off the patched model
    and followed the widget instead. Those two answers agree on every shipped
    graph today and would diverge silently the moment someone set a PDD graph
    to a different shift -- the sampler stepping one curve while the model
    integrates another.

    Asserted rather than documented, because a note would not have caught it.
    Also grades the emitted vector against ComfyUI's `simple` at each graph's
    OWN step count and OWN shift, which is what closes the distance between
    the fixed-shift arithmetic above and the graphs that actually ship.
    """
    from safetensors import safe_open
    loras = COMFY / "models" / "loras"
    graded, problems, skipped = [], [], []
    for path in graph_paths(WORKFLOWS):
        if not path.name.endswith("_api.json"):
            continue
        g = json.loads(path.read_text())
        pdd = [n for n in g.values()
               if isinstance(n, dict)
               and n.get("class_type") == "MiniMaxH3PDDLoRA"]
        if not pdd:
            continue
        steps = resolve_widget(g, pdd[0], "steps").value   # see note above
        if not isinstance(steps, int) or steps <= 0:
            continue                       # graded by `graphs consume it`
        shifts = [n["inputs"] for n in g.values()
                  if isinstance(n, dict)
                  and n.get("class_type") == "MiniMaxH3SigmaShift"]
        if not shifts:
            # ABSENT IS NOT MISSING. Since 2026-08-31 the PDD graphs ship
            # without a shift node, because at the checkpoint's own 12/3 it
            # patched the model into what it already was. So the shift the
            # render runs is the checkpoint's class default, and that is what
            # the file must match. Read it off ComfyUI rather than retyping
            # it: `h3_config.SIGMA_SHIFT` is this repo's copy of the same
            # numbers, and a check that compares one copy against another
            # would go green on a core change that moved the real one.
            # `pdd_lora.py::check_shift` takes the same fallback at run time.
            gshift = float(_checkpoint_default_shift())
        else:
            gshift = float(shifts[0]["shift_video"])
        f = loras / pdd[0]["inputs"]["lora_name"]
        if not f.exists():
            skipped.append(path.name)
            continue
        with safe_open(f, framework="pt") as fh:
            meta = fh.metadata() or {}
        fshift = float(meta["pdd_shift_video"])
        if gshift != fshift:
            problems.append(
                f"{path.name}: graph shift_video {gshift} against the file's "
                f"{fshift}. The node emits the FILE's schedule, so the sampler "
                f"would step one curve while the model integrates another")
            continue
        got = emitted(gshift, steps)
        want = comfy_simple(gshift, steps)
        if not (got.shape == want.shape and torch.equal(got, want)):
            problems.append(
                f"{path.name}: emitted sigmas at shift {gshift}, {steps} steps "
                f"are not `simple`'s")
        graded.append(path.name)
    assert not problems, "; ".join(problems[:4])
    # NOT `assert graded`. This module's contract says it needs "no checkpoint
    # on disk", and every LoRA being absent is a correct state on a fresh
    # checkout -- failing there would be red while nothing is wrong, which
    # `docs/checks.md` calls worse than no check. An empty result is reported
    # as a skip; what stays a failure is a graph whose file IS present and
    # disagrees, which `problems` above carries.
    if not graded:
        return (f"SKIPPED: no PDD LoRA on disk ({len(skipped)} graph(s) "
                f"unreachable), so no shift was compared")
    note = f"{len(graded)} graph(s) agree with their file's shift"
    return note + (f"; {len(skipped)} skipped (file absent)" if skipped else "")


def case_ui_and_api_agree():
    """The UI form's POSITIONAL read of `steps` equals the API form's NAMED one.

    `h3_config.graph_schedule` reads `steps` off a UI graph by widget index,
    because a UI graph carries no names -- and the schema gained BOTH an input
    and an output this round, which is exactly when a positional read drifts.
    `check_node_ids.py` guards against an input being INSERTED rather than
    appended, but nothing tied that guarantee to this reader.

    This is the control the repo's own rule asks for: compare against a
    different implementation rather than against a number computed here. The
    API form names its inputs, so the two forms disagreeing means the index is
    wrong -- and it fails loudly instead of silently reading a neighbouring
    int, which is what would happen if `nfe` and `steps` ever swapped places.
    """
    pairs, problems = 0, []
    for ui in graph_paths(WORKFLOWS):
        if ui.name.endswith("_api.json"):
            continue
        api = ui.with_name(ui.name[:-5] + "_api.json")
        if not api.exists():
            continue
        gu = json.loads(ui.read_text())
        if not any(isinstance(n, dict)
                   and n.get("type") == "MiniMaxH3PDDLoRA"
                   for n in gu.get("nodes", [])):
            continue
        su, _ = graph_schedule(gu)
        sa, _ = graph_schedule(json.loads(api.read_text()))
        pairs += 1
        if su != sa:
            problems.append(f"{ui.name}: UI reads steps={su}, API reads {sa}")
    assert not problems, "; ".join(problems[:4])
    assert pairs, "no PDD UI/API pair was compared; this case asserts nothing"
    return f"{pairs} PDD UI/API pair(s) agree on the step count"


print("PDD SIGMAS output: the schedule the heads were fused for")
check("emitted is simple", case_emitted_is_simple)
def case_pdd_end_percent_keeps_last_step_dense():
    """`SOL_PDD_CUDA`'s flat `end_percent` leaves the FINAL evaluation dense,
    at every step count a PDD graph can legally run.

    That property is the entire argument for one constant where
    `SOL_END_PERCENT_BY_STEPS` needs a row per count, and until 2026-08-29
    nothing asserted it. It is also newly load-bearing: `resolve_emit_steps`
    refused every non-divisor until that day, so 4 and 8 were the only counts
    reachable and a constant tuned on them could not be wrong anywhere else.
    Since 1b8b54d the envelope route admits 5, 6 and 7 as well -- three step
    counts nobody has ever checked this against.

    Losing the dense final step is the defect the step table was written for:
    at shift 12 that evaluation covers the largest jump in the schedule and is
    where PDD's fused heads deviate most from the base, so a sparse one stacks
    two approximations on the step that can least afford either.

    Both shifts, because a graph may carry either and the answer must not
    depend on which.
    """
    from pdd_lora import resolve_emit_steps, envelope_partition
    from h3_config import SOL_PDD_CUDA
    import comfy.model_sampling
    import comfy.samplers

    checked = []
    for shift in SHIFTS:
        sampling = comfy.model_sampling.ModelSamplingDiscreteFlow(None)
        sampling.set_parameters(shift=shift)
        hi = float(sampling.percent_to_sigma(SOL_PDD_CUDA["start_percent"]))
        lo = float(sampling.percent_to_sigma(SOL_PDD_CUDA["end_percent"]))
        for steps in range(1, 9):
            # Only counts the node will actually emit. Asking about an illegal
            # one would assert a property of a schedule no graph can run.
            legal = (NUM_STEPS % steps == 0
                     or envelope_partition(NUM_STEPS, steps, TRAINED) is not None)
            if not legal:
                continue
            # ...and only counts within 2x the trained block width. 1 and 2
            # evaluations tile the grid and the node emits them, but at widths
            # 32 and 16 against a trained 4 they are 8x and 4x out, past the
            # edge this repo already calls 4 "2x, the edge". No graph runs
            # them.
            #
            # **This exclusion is load-bearing and was found by this check
            # going red, not assumed.** At 2 evaluations the flat 0.74 DOES
            # leave the final evaluation sparse: the schedule is 1.0 and
            # 0.9231, and 0.9231 sits inside the band. So the property below
            # is true over the usable range and false just outside it -- worth
            # knowing before anyone reaches for a 2-step PDD arm.
            if NUM_STEPS // steps > 2 * TRAINED:
                continue
            assert resolve_emit_steps(steps, 8, NUM_STEPS, TRAINED) == steps
            sigmas = [float(x) for x in comfy_simple(shift, steps)][:-1]
            last = sigmas[-1]
            # OUTSIDE the band, either side. Above `sigma_start` is dense too,
            # and asserting only `last < lo` called a 1-step schedule broken
            # when its single evaluation sits at 1.0, above the warm-up edge.
            assert not (lo <= last <= hi), (
                f"shift {shift}, {steps} steps: the final evaluation sits at "
                f"sigma {last:.4f}, INSIDE Sol's band [{lo:.4f}, {hi:.4f}] "
                f"(end_percent {SOL_PDD_CUDA['end_percent']}, start_percent "
                f"{SOL_PDD_CUDA['start_percent']}). The last step would run "
                f"SPARSE, which is the defect SOL_END_PERCENT_BY_STEPS exists "
                f"to prevent -- at shift 12 that evaluation covers the largest "
                f"jump in the schedule and is where PDD's fused heads deviate "
                f"most from the base.")
            sparse = sum(1 for x in sigmas if lo <= x <= hi)
            checked.append(f"{steps}@{shift:g}:{sparse}/{steps}")
    assert checked, "no legal step count was exercised"
    return "sparse steps " + " ".join(checked)


check("exactness regime holds", case_exactness_regime_holds)
check("knots round trip", case_knots_round_trip)
check("untileable raises", case_untileable_raises)
check("envelope tiling legal", case_envelope_tiling_is_legal)
check("zero is inert", case_zero_is_inert)
check("graphs consume it", case_graphs_consume_it)
check("graph shift matches file", case_graph_shift_matches_file)
check("ui and api agree", case_ui_and_api_agree)
check("pdd end_percent keeps last step dense",
      case_pdd_end_percent_keeps_last_step_dense)

if failures:
    print(f"\nFAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("\nall ok -- the sampler steps where the heads were fused")
