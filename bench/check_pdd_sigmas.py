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


def case_non_divisor_raises():
    """An explicit `steps` that does not tile the grid is refused.

    Drives `pdd_lora.resolve_emit_steps` itself. The first version of this case
    recomputed the condition and asserted its own arithmetic, which could not
    have failed however the node behaved -- the shape `docs/checks.md` calls a
    check whose input already satisfies the outcome.
    """
    from pdd_lora import resolve_emit_steps
    for bad in (3, 5, 6, 7, 12):
        assert NUM_STEPS % bad, f"{bad} divides {NUM_STEPS}; not a case"
        try:
            got = resolve_emit_steps(bad, 8, NUM_STEPS)
        except RuntimeError:
            continue
        raise AssertionError(
            f"steps={bad} does not tile the {NUM_STEPS}-point grid but was "
            f"accepted, returning {got}. No on-grid schedule exists at that "
            f"count, so the SIGMAS output would be silently off it.")
    return "3, 5, 6, 7, 12 all refused as explicit requests"


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
    graded, problems = [], []
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
    return f"{len(graded)} PDD graph(s) sample from the node"


def case_graph_shift_matches_file():
    """Each PDD graph's `MiniMaxH3SigmaShift` equals its file's own shift.

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
            problems.append(f"{path.name}: no MiniMaxH3SigmaShift to compare")
            continue
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
check("exactness regime holds", case_exactness_regime_holds)
check("knots round trip", case_knots_round_trip)
check("non-divisor raises", case_non_divisor_raises)
check("zero is inert", case_zero_is_inert)
check("graphs consume it", case_graphs_consume_it)
check("graph shift matches file", case_graph_shift_matches_file)
check("ui and api agree", case_ui_and_api_agree)

if failures:
    print(f"\nFAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("\nall ok -- the sampler steps where the heads were fused")
