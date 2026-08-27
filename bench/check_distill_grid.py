#!/usr/bin/env python3
"""Check every distilled graph samples ON the sigma grid its LoRA was distilled at.

`check_distill_settings.py` grades the shift and the step count. Both can be
right while the sampler still evaluates somewhere else, because the scheduler
is what turns (shift, steps) into actual sigmas. A turbo LoRA distilled at
NFE=4 saw four specific sigmas; a scheduler that puts its four steps elsewhere
is running a distilled model off its own grid, and nothing errors.

`workflows/h3_config.py` has asserted since before this file existed that
`simple` is the only scheduler reproducing a distilled LoRA's own grid. That
was prose. This is the control for it.

**What this file does NOT grade, deliberately.** It takes a graph's shift and
step count as the DEFINITION of the target grid and asks only whether the
scheduler lands on it. Whether that LoRA belongs at that shift and that step
count is `check_distill_settings.py`'s subject, graded there against the
vendor's own table. Asserting it here too would be a second copy of one
judgement, and the two would drift. The seam is pinned from the other side:
`bench/red/show_red_distill_grid.py` carries a wrong shift and a wrong step
count as NEAR_MISS cases, so if this file ever starts grading them, they go red
and somebody has to decide which check owns it.

The vendor publishes the grid in two independent places, and this file reads
both rather than computing its own expected values:

  * `coderef/Minimax-H3-Turbo/README.md`, "Note on shift": the rule
    `q_i = (N - i) / N` and, for NFE=4 at shift 12/3, the literal resulting
    sigmas. Parsed out of the README, never retyped here.
  * ComfyUI itself: `comfy.samplers.calculate_sigmas` over `ModelSamplingAV`,
    and `comfy.ldm.minimax.model.time_shift_sigma` for the audio schedule the
    DiT derives from the video one.

DiffSynth is a third implementation of the same rule
(`coderef/DiffSynth-Studio/diffsynth/diffusion/flow_match.py::set_timesteps_minimax_h3`
builds `linspace(1, 0, N+1)[:-1]`, which is `q_i`), at a different shift. The
rule is not in dispute; only the constant is. See `docs/comfyui_vendor_gaps.md`.

**Why exactness rather than a tolerance.** `simple` reads the model's discrete
1,000-entry sigma table at truncated indices, so it reproduces the closed form
EXACTLY when `1000 % steps == 0` and quantizes otherwise (measured: 0 at 4, 5,
8, 10 and 20 steps; ~0.002 at 12, 16 and 24). Every distilled graph this repo
ships runs at 4 or 8 steps, so every one of them is in the exact regime and no
tolerance is needed -- among the arms still graded that way. `divisor_regime_holds`
asserts that precondition instead of assuming it.

**Owner-recipe arms are partitioned OUT rather than tolerated.** Since
2026-08-23 the 768p arm renders at 6 steps, which does not divide 1,000, so
`simple` quantizes there and the exactness claim does not apply. Loosening
EXACT would have destroyed the vendor claim for every arm at once; rewriting
the vendor row to say 6 would have been worse, because the row records what the
student was distilled to do. So those arms go down their own path with a weaker
claim, stated as weaker: the deviation must be DECLARED in
`check_distill_settings.OWNER_RECIPE`, and `simple` must still be strictly the
nearest scheduler at the arm's own step count -- comparative, so it needs no
invented tolerance. What is deliberately not claimed about them is that they
sit on the vendor's distillation grid. They do not. The 16-step BASE graphs are deliberately out of
scope: the base checkpoint was not distilled to a step grid, so the vendor rule
does not bind them, and `check_distill_settings.py` already holds them at 12/3.

Claims, i.e. what breaks if a case is deleted:
  vendor grid agrees     the README's own rule reproduces the README's own
                         published sigmas, and ComfyUI's `simple` reproduces
                         both. Three implementations, none of them this file.
                         Delete it and the remaining cases grade graphs against
                         numbers with no external anchor. Also asserts the two
                         routes to the AUDIO grid agree -- applying the rule at
                         the audio shift, and inverting the video sigma through
                         the DiT's own `time_shift_sigma` -- because the DiT
                         takes the second route and the README publishes the
                         first
  simple is the only one every other scheduler MISSES the grid at the distilled
                         step counts. This is the falsifiable half: it asserts a
                         DISAGREEMENT, so it cannot pass by everything happening
                         to agree. Without it, a change making all schedulers
                         identical would leave every other case green
  divisor regime holds   every VENDOR-GRID arm runs at a step count dividing
                         1,000, which is what makes exactness the right
                         assertion there. Fails if that population empties --
                         if every arm became a recipe arm, this case passing
                         would say nothing
  recipe arms declared   an arm off its LoRA's distilled step count must be
                         declared in OWNER_RECIPE, so a recipe cannot arrive by
                         somebody editing a widget; and `simple` must still be
                         strictly nearest at its own step count, which is the
                         part of the grid claim that survives the recipe
  graphs on grid         every shipped graph loading a turbo LoRA reproduces its
                         own (shift, steps) grid exactly. Both graph forms, and
                         they are read through `check_distill_settings`'s
                         readers rather than a second walk of the same JSON
  exemptions necessary   an exempt graph that stops deviating is a FAILURE, not
                         a pass. Exemption implies coverage; a stale one covers
                         a graph nobody is reading anymore

Needs ComfyUI importable (CPU only -- no CUDA, no model, no server) and
`coderef/`. Neither absence can produce a pass: a missing README SKIPS and
exits 2, and an unreachable ComfyUI FAILS.

Exit codes: 0 all cases passed, 1 a case failed, 2 passed but a control was
skipped (coderef/ absent).

    python bench/check_distill_grid.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# custom_nodes/<this pack>/bench -> the ComfyUI root. Derived from where this
# file actually sits rather than from the home directory, so a checkout
# installed anywhere bootstraps itself and this is not one more script that
# needs PYTHONPATH set before it will run.
COMFY = REPO.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workflows"))
sys.path.insert(0, str(HERE))
sys.path.append(str(COMFY))

WORKFLOWS = REPO / "workflows"
VENDOR_README = REPO / "coderef" / "Minimax-H3-Turbo" / "README.md"

from h3_config import graph_paths  # noqa: E402
from pdd_math import block_bounds  # noqa: E402
from check_distill_settings import (  # noqa: E402
    LEGAL, OWNER_RECIPE, PACK_STEPS, classify, classify_pack, classify_pdd,
    pdd_grid, pdd_nfe, is_turbo,
    read_api, read_ui,
)

#: The closed form is exact only where the discrete table lands on the step
#: boundaries. See the module docstring.
TRAIN_TIMESTEPS = 1000
EXACT = 1e-6

#: {graph stem: reason}. A graph whose scheduler deliberately leaves the
#: distillation grid. A reason naming a preference without a mechanism is not a
#: reason; a reason naming a file is not a reason either.
GRID_EXEMPT_STEMS = {
    "h3_probe_turbo_768p_owner":
        "the owner's own working recipe (euler + beta at 4 steps, strength "
        "0.75), shipped as a graph rather than as remembered widget values so "
        "the vendor-recipe arm has something with a sha to be judged against. "
        "Its deviation from the distilled grid IS one of the things that arm "
        "is measuring, and h3_config already prices beta's effect on Sol's "
        "sparse-step window",
}


def vendor_rule(steps: int, shift: float) -> list[float]:
    """The README's rule: q_i = (N - i) / N, then the flow shift, then 0.

    Same algebra as `comfy.model_sampling.time_snr_shift` and as DiffSynth's
    `set_timesteps_minimax_h3`. Written out because the point of this file is
    to compare implementations, and importing the one under test would make the
    comparison vacuous.
    """
    out = []
    for i in range(steps):
        q = (steps - i) / steps
        out.append(shift * q / (1.0 + (shift - 1.0) * q))
    return out + [0.0]


def parse_vendor_grid(text: str):
    """Pull the published NFE, shifts and sigma lists out of "Note on shift".

    Returns (nfe, shift_video, shift_audio, q, video_sigma, audio_sigma) or
    None if the section is not shaped as expected. Returning None rather than
    raising keeps "the vendor reworded their README" distinguishable from "the
    numbers moved", which are different problems with different fixes.
    """
    m = re.search(r"###\s*Note on shift(.*?)(?=\n###|\Z)", text, re.S)
    if not m:
        return None
    # The published example wraps across source lines mid-sentence.
    body = " ".join(m.group(1).split())

    def nums(pattern):
        hit = re.search(pattern, body)
        if not hit:
            return None
        return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", hit.group(1))]

    def scalar(pattern):
        hit = re.search(pattern, body)
        return None if hit is None else float(hit.group(1))

    nfe = scalar(r"NFE\s*=\s*(\d+)`")
    sv = scalar(r"video shift\s*=\s*(\d+(?:\.\d+)?)")
    sa = scalar(r"audio shift\s*=\s*(\d+(?:\.\d+)?)")
    q = nums(r"`q\s*=\s*\[([^\]]*)\]`")
    vid = nums(r"video sigma\s*`\[([^\]]*)\]\s*->\s*0`")
    aud = nums(r"audio sigma\s*`\[([^\]]*)\]\s*->\s*0`")
    if nfe is None or sv is None or sa is None:
        return None
    if q is None or vid is None or aud is None:
        return None
    return int(nfe), sv, sa, q, vid, aud


def comfy_grid(shift_video: float, shift_audio: float, scheduler: str,
               steps: int):
    """ComfyUI's own sigmas for this arm, plus the DiT's derived audio grid.

    Imported lazily and in CPU mode so this file states what it could not reach
    rather than dying at import; an unreachable ComfyUI is a FAILURE here, never
    a skip, because the vendor half alone proves nothing about what we run.
    """
    try:
        import comfy.cli_args
        comfy.cli_args.args.cpu = True
        import comfy.model_sampling as model_sampling
        import comfy.samplers
        from comfy.ldm.minimax.model import time_shift_sigma
    except ImportError as exc:
        raise AssertionError(
            f"ComfyUI is not importable from {COMFY}, so the half of this "
            f"check that grades what WE run could not be reached. The vendor "
            f"README alone proves nothing about our graphs, so this is a "
            f"failure and not a skip. ({exc})") from exc

    class _Config:
        sampling_settings = {"shift": shift_video, "audio_shift": shift_audio}

    class _ModelSamplingAdvanced(model_sampling.ModelSamplingAV,
                                 model_sampling.CONST):
        pass

    # The same two lines `MiniMaxH3SigmaShift.execute` runs, so this grades the
    # object a graph actually samples through rather than a reimplementation.
    ms = _ModelSamplingAdvanced(_Config())
    ms.set_parameters(shift=shift_video, audio_shift=shift_audio)

    video = [float(s) for s in comfy.samplers.calculate_sigmas(ms, scheduler, steps)]
    audio = [float(time_shift_sigma(s, shift_video, shift_audio)) if s > 0 else 0.0
             for s in video]
    return video, audio


def deviation(got: list[float], want: list[float]) -> float:
    """Max absolute difference, or inf when the lengths disagree.

    A scheduler returning a different number of sigmas has not reproduced the
    grid; `ddim_uniform` does exactly that, and zipping would silently compare
    the shorter prefix and call it close.
    """
    if len(got) != len(want):
        return float("inf")
    return max(abs(a - b) for a, b in zip(got, want))


def grade_shift_nodes(shifts) -> list[str]:
    """Problems with a graph's set of shift nodes. Empty means they agree.

    Split graphs carry two. `build_workflows.py::_plain_model_chain` states
    they must be identical -- both halves read sigmas from ONE `BasicScheduler`,
    so two shifts would have them integrating different curves -- and nothing
    asserted it. A collector so the red harness can drive it; the three shipped
    split graphs agree today, which is precisely why the assertion has to exist
    rather than be inferred from their agreeing.
    """
    distinct = sorted(set(shifts))
    if len(distinct) > 1:
        return [f"{len(shifts)} shift nodes disagree: {distinct}"]
    return []


def grade_arm(shift_video: float, shift_audio: float, scheduler: str,
              steps: int) -> list[str]:
    """Problems with one (shift, scheduler, steps) arm. Empty means on-grid.

    A collector rather than a comparator: `bench/red/show_red_distill_grid.py`
    drives this directly with synthetic arms, so the mutation reaches the same
    code a graph does. A harness that could only feed the reporter would pass a
    grader that returned its own baseline.
    """
    problems = []
    want_v, want_a = vendor_rule(steps, shift_video), vendor_rule(steps, shift_audio)
    got_v, got_a = comfy_grid(shift_video, shift_audio, scheduler, steps)
    dv, da = deviation(got_v, want_v), deviation(got_a, want_a)
    if dv >= EXACT:
        problems.append(
            f"{scheduler} at {steps} steps, shift {shift_video} is off the "
            f"distilled video grid by {dv:.4f}. Got "
            f"{[round(x, 4) for x in got_v]}, distilled at "
            f"{[round(x, 4) for x in want_v]}")
    if da >= EXACT:
        problems.append(
            f"audio grid off by {da:.4f} at shift {shift_audio}")
    return problems


def grade_published(published) -> list[str]:
    """Problems with the vendor's published grid. Empty means all three agree.

    Takes the parsed tuple rather than reading the README, so the red harness
    can hand it mutated numbers without writing to `coderef/`, which is a
    gitignored checkout of somebody else's repo.
    """
    problems = []
    nfe, sv, sa, q, vid, aud = published
    if q != [(nfe - i) / nfe for i in range(nfe)]:
        problems.append(f"published q {q} is not q_i = (N - i) / N at NFE={nfe}")
    rule_v, rule_a = vendor_rule(nfe, sv), vendor_rule(nfe, sa)
    # Published to 4 decimal places, so compare at that precision.
    if deviation(rule_v, vid + [0.0]) >= 5e-5:
        problems.append(
            f"the rule at shift {sv} gives {[round(x, 4) for x in rule_v]}, "
            f"the README publishes {vid} -> 0")
    if deviation(rule_a, aud + [0.0]) >= 5e-5:
        problems.append(
            f"the rule at shift {sa} gives {[round(x, 4) for x in rule_a]}, "
            f"the README publishes {aud} -> 0")
    # ComfyUI against the published numbers, not against our rule: if the
    # vendor moved, this is what tells us what WE run no longer matches them.
    got_v, got_a = comfy_grid(sv, sa, "simple", nfe)
    if deviation(got_v, vid + [0.0]) >= 5e-5:
        problems.append(
            f"ComfyUI simple at shift {sv}, {nfe} steps gives "
            f"{[round(x, 4) for x in got_v]}, the README publishes {vid} -> 0")
    # The DiT inverts the video sigma to the base grid and re-applies the audio
    # shift; the README applies the rule at the audio shift directly. Two routes
    # to the same schedule, and the model takes the second one.
    if deviation(got_a, aud + [0.0]) >= 5e-5:
        problems.append(
            f"time_shift_sigma gives {[round(x, 4) for x in got_a]}, the "
            f"README publishes audio {aud} -> 0")
    return problems


def main() -> int:
    failures: list[str] = []
    skipped: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {exc}")

    print("distilled graphs sample on the grid they were distilled at\n")

    published = None
    if VENDOR_README.is_file():
        published = parse_vendor_grid(VENDOR_README.read_text())

    def vendor_grid_agrees():
        assert published is not None, (
            f"{VENDOR_README.relative_to(REPO)} has no parseable "
            f'"Note on shift" section')
        problems = grade_published(published)
        assert not problems, "; ".join(problems)

    def simple_is_the_only_one():
        assert published is not None, "needs the vendor README"
        nfe, sv, sa, _q, _v, _a = published
        want = vendor_rule(nfe, sv)
        others = {}
        for scheduler in ("beta", "normal", "sgm_uniform", "ddim_uniform",
                          "karras", "exponential"):
            try:
                got, _ = comfy_grid(sv, sa, scheduler, nfe)
            except Exception:
                continue  # a scheduler this build does not offer
            others[scheduler] = deviation(got, want)
        assert others, "no comparison scheduler was reachable"
        agreeing = [s for s, d in others.items() if d < EXACT]
        assert not agreeing, (
            f"h3_config says simple is the only scheduler on the distilled "
            f"grid, but {agreeing} also reproduce it at {nfe} steps. Either "
            f"that claim is now wrong or this check lost its subject.")
        worst = min(others.items(), key=lambda kv: kv[1])
        print(f"        (nearest miss: {worst[0]} off by {worst[1]:.4f} at "
              f"{nfe} steps; simple is exact)")

    # --- graph population -------------------------------------------------
    graded = []           # (path, stem, shift, scheduler, steps)
    unreadable = []       # graphs whose arm could not be resolved statically
    split_disagree = []   # graphs whose two shift nodes do not match
    for path in graph_paths(WORKFLOWS):
        doc = json.loads(path.read_text())
        found = read_ui(doc) if isinstance(doc.get("nodes"), list) else read_api(doc)
        if not any(is_turbo(name) for name in found.loras):
            continue
        if found.shift is None or found.steps is None or found.scheduler is None:
            # Collected, not printed here: printing a FAIL under a case name
            # that `check()` later prints `ok` for makes the log contradict
            # itself. This gets its own case below.
            unreadable.append(
                f"{path.relative_to(REPO)} loads a turbo LoRA but its shift, "
                f"steps or scheduler could not be read (linked widget?)")
            continue
        for why in grade_shift_nodes(found.shifts):
            split_disagree.append(f"{path.relative_to(REPO)} {why}")
        key = classify(next(n for n in found.loras if is_turbo(n)))
        graded.append((path, path.stem[:-4] if path.stem.endswith("_api")
                       else path.stem, found.shift, found.scheduler,
                       found.steps, key))

    # Partition. A vendor-grid arm runs the step count its LoRA was distilled
    # to; a recipe arm runs a declared owner recipe instead. An arm at neither
    # is an undeclared deviation and fails below -- which is the whole point of
    # partitioning rather than widening the tolerance until everything fits.
    vendor_arms, recipe_arms, undeclared_arms = [], [], []
    for row in graded:
        _p, _stem, _sh, _sc, steps, key = row
        recipe = OWNER_RECIPE.get(key)
        if key is None:
            # The third-party pack family (`turbo_v<n>_step<ckpt>_ema`) has no
            # LEGAL row: its README documents a step RANGE, not one NFE, and
            # `check_distill_settings` grades it that way. Treated as a vendor
            # arm at any documented count, so it keeps its exact-grid grading
            # rather than being read as an undeclared deviation.
            lo, hi = PACK_STEPS
            (vendor_arms if lo <= steps <= hi else undeclared_arms).append(row)
            continue
        distilled = LEGAL[key].steps if key in LEGAL else frozenset()
        if recipe is not None and steps == recipe["steps"]:
            recipe_arms.append(row)
        elif steps in distilled:
            vendor_arms.append(row)
        else:
            undeclared_arms.append(row)

    def graphs_are_readable():
        assert not unreadable, "; ".join(unreadable)

    def split_arms_share_one_shift():
        """A split graph's two shift nodes must agree.

        `build_workflows.py::_plain_model_chain` states this as a must -- both
        halves read sigmas from ONE `BasicScheduler`, so two different shifts
        would have the halves integrating different curves and the handoff
        would be meaningless -- and nothing asserted it. Three shipped graphs
        carry two nodes (`h3_probe_split_base_first`, `..._last`, and
        `h3_probe_ref2v_split_turbo_pack`). They agree today, which is exactly
        why this needs stating: without it, the grid case grades whichever node
        the reader happened to see last and passes for a reason it never checks.
        """
        assert not split_disagree, "; ".join(split_disagree)

    def divisor_regime_holds():
        assert vendor_arms, (
            "no graph is a vendor-grid arm any more -- every one now runs an "
            "owner recipe. The exactness claim has lost its subject, and this "
            "case passing would say nothing.")
        bad = [(p.relative_to(REPO), n) for p, _s, _sh, _sc, n, _k in vendor_arms
               if TRAIN_TIMESTEPS % n]
        assert not bad, (
            f"exactness is only the right assertion where the step count "
            f"divides {TRAIN_TIMESTEPS}; these do not: {bad}. Grade them "
            f"against the table's quantization instead of loosening EXACT.")

    def graphs_on_grid():
        problems = []
        for path, stem, (sv, sa), scheduler, steps, _key in vendor_arms:
            if stem in GRID_EXEMPT_STEMS:
                continue
            for why in grade_arm(sv, sa, scheduler, steps):
                problems.append(f"{path.relative_to(REPO)}: {why}")
        assert not problems, "; ".join(problems)

    def exemptions_necessary():
        seen = {stem for _p, stem, _sh, _sc, _n, _k in graded}
        stale = sorted(set(GRID_EXEMPT_STEMS) - seen)
        assert not stale, (
            f"exempted graphs that no longer load a turbo LoRA (or no longer "
            f"exist): {stale}. Remove the exemption or the graph.")
        for path, stem, (sv, sa), scheduler, steps, _key in graded:
            if stem not in GRID_EXEMPT_STEMS:
                continue
            assert grade_arm(sv, sa, scheduler, steps), (
                f"{path.relative_to(REPO)} is exempted from the grid rule "
                f"({GRID_EXEMPT_STEMS[stem]}) but now sits ON the grid. The "
                f"exemption is stale -- remove it, do not leave both.")

    if published is None:
        for name in ("vendor grid agrees", "simple is the only one"):
            skipped.append(name)
            print(f"  SKIP  {name}: {VENDOR_README.relative_to(REPO)} is "
                  f"unreadable (coderef/ is gitignored). The graph cases below "
                  f"still run, but against the rule alone with no vendor "
                  f"anchor confirming it.")
    else:
        check("vendor grid agrees", vendor_grid_agrees)
        check("simple is the only one", simple_is_the_only_one)

    def recipe_arms_are_declared_and_simple_is_nearest():
        """Owner-recipe arms are NOT vendor-grid arms, and are not graded as one.

        Six steps does not divide the 1,000-step training grid, so `simple`
        quantizes there and the exactness claim that makes `graphs on grid`
        mean anything simply does not apply. Loosening EXACT to cover it would
        have destroyed the vendor claim for every arm at once; rewriting the
        vendor row to say 6 would have been worse, because the row is what the
        student was actually distilled to do.

        So these arms get a weaker claim, stated as such. What is still
        asserted:

          * the deviation is declared -- an arm off its distilled step count
            with no `OWNER_RECIPE` entry fails, so a recipe cannot arrive by
            somebody editing a widget;
          * `simple` is STRICTLY the nearest scheduler to the closed form at
            the arm's own step count. Comparative, so it needs no invented
            tolerance, and it is the part of the original claim that survives:
            the steps are the owner's, the placement of them is still the one
            scheduler that tracks the flow curve.

        What is deliberately NOT asserted: that these arms are on the vendor's
        distillation grid. They are not, by construction, and nothing here
        should be readable as saying they are.
        """
        assert not undeclared_arms, (
            "these graphs run a step count that is neither their LoRA's "
            "distilled NFE nor a declared OWNER_RECIPE: "
            + "; ".join(f"{p.relative_to(REPO)} at {n} steps ({k})"
                        for p, _s, _sh, _sc, n, k in undeclared_arms))
        if not recipe_arms:
            return
        compared = 0
        for path, stem, (sv, sa), scheduler, steps, key in recipe_arms:
            if stem in GRID_EXEMPT_STEMS:
                # An arm whose scheduler deviation IS its subject. Grading it
                # on "is simple nearest" would contradict the exemption that
                # already covers it, and `exemptions_necessary` keeps that
                # exemption honest from the other side.
                continue
            compared += 1
            mine = deviation(comfy_grid(sv, sa, scheduler, steps)[0],
                             vendor_rule(steps, sv))
            worse = {}
            # The arm's OWN scheduler is not a comparison against itself.
            for other in ("simple", "beta", "normal", "sgm_uniform",
                          "ddim_uniform"):
                if other == scheduler:
                    continue
                try:
                    worse[other] = deviation(
                        comfy_grid(sv, sa, other, steps)[0],
                        vendor_rule(steps, sv))
                except Exception:
                    continue
            assert worse, f"{path.relative_to(REPO)}: no comparison scheduler ran"
            closer = sorted(o for o, d in worse.items() if d <= mine)
            assert not closer, (
                f"{path.relative_to(REPO)}: this arm runs the owner recipe at "
                f"{steps} steps, where `{scheduler}` is off the closed form by "
                f"{mine:.4f} and {closer} are no further. `simple` being the "
                f"nearest scheduler is the only grid claim these arms carry; "
                f"if it is no longer true, the recipe needs re-deciding.")
        print(f"        ({len(recipe_arms)} owner-recipe arm(s) NOT graded as "
              f"vendor-grid arms, {compared} of them scheduler-compared; "
              f"{len(vendor_arms)} vendor-grid arm(s))")

    check("graphs are readable", graphs_are_readable)
    check("recipe arms are declared, simple still nearest",
          recipe_arms_are_declared_and_simple_is_nearest)
    check("split arms share one shift", split_arms_share_one_shift)
    check("divisor regime holds", divisor_regime_holds)
    def pdd_graphs_on_their_fused_grid():
        """A PDD graph samples exactly where its fused heads were built.

        Graded against ANALYTIC ground truth, not a vendor table. A PDD file
        records the shifts and the grid its heads were fused at, and
        `pdd_math.block_bounds` turns those into the exact times the sampler
        must land on -- so unlike the turbo rows above there is no published
        list to parse and no tolerance to negotiate. Off those boundaries the
        fused output heads decode intervals the sampler never visits, and the
        node's own runtime warning is the only other thing that would say so.

        Not covered by the cases above: `is_turbo` is false for a PDD
        filename, so every one of these graphs was skipped there.
        """
        bad, seen = [], 0
        for path in graph_paths(WORKFLOWS):
            doc = json.loads(path.read_text())
            found = read_ui(doc) if isinstance(doc.get("nodes"), list) else read_api(doc)
            names = [n for n in found.loras if classify_pdd(n)]
            if not names:
                continue
            seen += 1
            rel = path.relative_to(REPO)
            if found.shift is None or found.steps is None or found.scheduler is None:
                bad.append(f"{rel}: loads a PDD LoRA but its shift, steps or "
                           f"scheduler could not be read")
                continue
            # The graph's nfe when it sets one; the file's is only a default
            # now that the heads are fused at load.
            # **The sampler's step count is the evaluation count.** Since
            # 2026-08-27 the node derives the block boundaries from
            # `sample_sigmas`, so the boundaries to compare against are the
            # ones THIS GRAPH's step count lands on, not the file's default.
            # An `nfe` override wins when set, because it makes the node
            # ignore the schedule and fuse uniform blocks at that count.
            nfe = found.pdd_nfe or found.steps
            grid = pdd_grid(names[0])
            if grid is None:
                bad.append(f"{rel}: could not read `pdd_num_steps` from "
                           f"{names[0]}, so there is no grid to grade against")
                continue
            if nfe < 1 or grid % nfe:
                bad.append(f"{rel}: {nfe} evaluations do not divide the "
                           f"{grid}-point grid, so the blocks do not tile it")
                continue
            sv, sa = found.shift
            video, audio = comfy_grid(sv, sa, found.scheduler, found.steps)
            for label, got, shift in (("video", video, sv), ("audio", audio, sa)):
                want = [1.0 - float(t) for t in
                        block_bounds(shift, grid, grid // nfe).tolist()]
                dev = deviation(got, want)
                if dev > 1e-6:
                    bad.append(
                        f"{rel}: {label} sigmas deviate {dev:.5f} from the "
                        f"boundaries its heads were fused at "
                        f"(scheduler={found.scheduler}, steps={found.steps}, "
                        f"shift={shift})")
        assert not bad, "\n         ".join(bad)
        # A case whose input is empty passes for the wrong reason. PDD arms are
        # shipped, so zero here means the scanner stopped recognising the
        # loader -- which is how `is_turbo` silently excluded these graphs from
        # every case above until 2026-08-26.
        assert seen, ("no graph was recognised as loading a PDD LoRA, so this "
                      "case graded nothing and passed. Check that read_api / "
                      "read_ui still see MiniMaxH3PDDLoRA.")
        return f"{seen} PDD graph(s), exact on both streams"

    check("pdd graphs on their fused grid", pdd_graphs_on_their_fused_grid)

    check("graphs on grid", graphs_on_grid)
    check("exemptions necessary", exemptions_necessary)
    print(f"        ({len(graded)} distilled graph(s), "
          f"{len(GRID_EXEMPT_STEMS)} exempt)")

    if failures:
        print(f"\n{len(sorted(set(failures)))} case(s) FAILED: "
              f"{', '.join(sorted(set(failures)))}")
        return 1
    if skipped:
        # A skipped control must not read as a clean pass to anything keying on
        # the exit code.
        print(f"\n{len(skipped)} case(s) SKIPPED: {', '.join(skipped)}. "
              f"Not a clean pass.")
        return 2
    print("\nall ok -- every distilled graph is on its own sigma grid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
