#!/usr/bin/env python3
"""Check every distilled graph samples ON the sigma grid its LoRA was distilled at.

`check_distill_settings.py` grades the shift and the step count. Both can be
right while the sampler still evaluates somewhere else, because the scheduler
is what turns (shift, steps) into actual sigmas. A distill fitted at four
evaluations saw four specific sigmas; a scheduler that puts its four steps
elsewhere is running a distilled model off its own grid, and nothing errors.
Since 2026-09-26 the graded population is PDD; the lightx2v
turbo arms this file was written for are retired (`docs/wiki/decisions.md`),
and their cases went with them. The vendor README they shipped with stays the
external anchor for the flow-shift rule itself.

`workflows/h3_config.py` has asserted since before this file existed that
`simple` is the only scheduler reproducing a distilled LoRA's own grid. That
was prose. This is the control for it.

**What this file does NOT grade, deliberately.** It takes a graph's shift and
step count as the DEFINITION of the target grid and asks only whether the
scheduler lands on it. Whether that LoRA belongs at that shift and that step
count is `check_distill_settings.py`'s subject, graded there against the
vendor's own table. Asserting it here too would be a second copy of one
judgement, and the two would drift. The seam is pinned from the other side:
the red harness (removed 2026-09-11) carries a wrong shift and a wrong step
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
8, 10 and 20 steps; ~0.002 at 12, 16 and 24). The 16-step BASE graphs are
deliberately out of scope: the base checkpoint was not distilled to a step
grid, and `check_distill_settings.py` already holds them at 12/3.

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
  pdd graphs on grid     every PDD graph samples exactly the block boundaries
                         its fused heads were built at, both streams

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
import h3_config  # noqa: E402
from pdd_math import block_bounds, partition_bounds  # noqa: E402
from pdd_lora import envelope_partition  # noqa: E402
from check_distill_settings import (  # noqa: E402
    classify_pdd, pdd_grid, pdd_nfe, pdd_block_size, read_api,
)

#: The closed form is exact only where the discrete table lands on the step
#: boundaries. See the module docstring.
TRAIN_TIMESTEPS = 1000
EXACT = 1e-6

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


def comfy_on_cpu() -> None:
    """Tell ComfyUI this process has no card, before anything of core's that
    picks a device is imported. Every import of core in this file goes through
    here: until 2026-10-10 only `comfy_grid` set the flag, so with the vendor
    README absent (its two cases skip and `comfy_grid` is not reached first)
    the import of `comfy.supported_models` below raised core's "No CUDA GPUs
    are available" on a masked card, which is not an ImportError and was not
    caught.
    """
    import comfy.cli_args
    comfy.cli_args.args.cpu = True


def comfy_grid(shift_video: float, shift_audio: float, scheduler: str,
               steps: int):
    """ComfyUI's own sigmas for this arm, plus the DiT's derived audio grid.

    Imported lazily and in CPU mode so this file states what it could not reach
    rather than dying at import; an unreachable ComfyUI is a FAILURE here, never
    a skip, because the vendor half alone proves nothing about what we run.
    """
    try:
        comfy_on_cpu()
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

    def pdd_graphs_on_their_fused_grid():
        """A PDD graph samples exactly where its fused heads were built.

        Graded against ANALYTIC ground truth, not a vendor table. A PDD file
        records the shifts and the grid its heads were fused at, and
        `pdd_math.block_bounds` turns those into the exact times the sampler
        must land on -- so there is no published list to parse and no
        tolerance to negotiate. Off those boundaries the
        fused output heads decode intervals the sampler never visits, and the
        node's own runtime warning is the only other thing that would say so.

        """
        bad, seen = [], 0
        for path in graph_paths(WORKFLOWS, "*_api.json"):
            doc = json.loads(path.read_text())
            found = read_api(doc)
            names = [n for n in found.loras if classify_pdd(n)]
            if not names:
                continue
            seen += 1
            rel = path.relative_to(REPO)
            # The shift is deliberately NOT required here. Since 2026-08-31 the
            # PDD graphs ship with no `MiniMaxH3SigmaShift` -- at the
            # checkpoint's own 12/3 it patched the model into what it already
            # was -- so `found.shift` is None on every one of them, and nothing
            # in this case reads it: the grid is graded from `steps`, `pdd_nfe`
            # and the file's `pdd_num_steps`. Requiring it here would fail all
            # 20 for a value the case does not use. The absent shift IS graded,
            # by `check_pdd_sigmas.py` against ComfyUI's own model config and
            # by `check_distill_settings.py` against the base pair.
            if found.steps is None or found.scheduler is None:
                bad.append(f"{rel}: loads a PDD LoRA but its steps or "
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
            _ss = path.stem.removesuffix("_api").removesuffix("_savelat")
            _manual = [n["inputs"]["sigmas"] for n in doc.values()
                       if isinstance(n, dict) and n.get("class_type") == "ManualSigmas"]
            _pass1 = {p1 for p1, _ in (*h3_config.STEP_SWITCH_REV.values(),
                                       *h3_config.STEP_SWITCH_BASE.values())}
            if (any(isinstance(n, dict) and n.get("class_type") == "DisableNoise" for n in doc.values())
                    and any(m in _pass1 for m in _manual)):
                # The reverse switch: PDD runs FIRST, on PDD8's own schedule cut
                # at a knot (h3_config.STEP_SWITCH_REV, or STEP_SWITCH_BASE when
                # the base finishes). Every point it samples must be one of
                # PDD8's knots, 0, 4, 8, ... Recognised by its structure (a
                # DisableNoise second pass and a declared pass-1 schedule) since
                # 2026-09-27, when the t2v switch shipped under a user-facing
                # name; the filename prefix this keyed on missed it, the
                # name-keyed failure the 2026-09-26 postmortem's item 4.2 names.
                # The exact pair is graded by check_distill_settings.py.
                import pdd_math as _pm
                pts = [float(x) for x in next(m for m in _manual if m in _pass1).split(",")]
                knots = _pm.schedule_knots(pts, 12.0, 32)
                if knots != list(range(0, 4 * len(knots), 4)):
                    bad.append(f"{rel}: the reverse switch's PDD pass lands on {knots}, "
                               "not PDD8's own knots")
                continue
            if _ss == "h3_probe_t2v_step_switch_flashgen_pdd8":
                # Route 3's PDD pass is a tail, not a trajectory: it starts at
                # FlashGen's own endpoint, 0.888889, off the grid by design so
                # no noise mismatch is handed over, and its sigmas are asserted
                # exactly by check_distill_settings.py against
                # h3_config.STEP_SWITCH_PASS2_SIGMAS. What must hold here is that
                # its tail lands on PDD8's own knots.
                import pdd_math as _pm
                pts = [float(x) for x in h3_config.STEP_SWITCH_PASS2_SIGMAS.split(",")]
                knots = _pm.schedule_knots(pts, 12.0, 32)
                if knots[1:] != [24, 28, 32]:
                    bad.append(f"{rel}: step switch's PDD tail lands on {knots}, "
                               "not PDD8's own knots 24, 28, 32")
                continue
            grid = pdd_grid(names[0])
            if grid is None:
                bad.append(f"{rel}: could not read `pdd_num_steps` from "
                           f"{names[0]}, so there is no grid to grade against")
                continue
            # TWO routes tile the grid, and this graded only the first until
            # 2026-08-31. A divisor tiles it uniformly; a non-divisor may still
            # tile it UNEVENLY inside the trained envelope, which is what
            # `resolve_emit_steps` accepts and what the shipped
            # `..._manual_sigmas` graphs run at six. Grading only divisibility
            # made this check RED on a graph the project ships on purpose --
            # `docs/checks.md`'s rule that a red on correct state is worse than
            # no check. Both `envelope_partition` and `partition_bounds` are
            # imported from the node and `pdd_math` rather than restated here,
            # so deleting either reddens this rather than silently agreeing.
            widths = None
            if nfe < 1:
                bad.append(f"{rel}: {nfe} evaluations is not a step count")
                continue
            if grid % nfe:
                trained = pdd_block_size(names[0])
                widths = (envelope_partition(grid, nfe, int(trained))
                          if trained else None)
                if widths is None:
                    bad.append(
                        f"{rel}: {nfe} evaluations neither divide the "
                        f"{grid}-point grid nor tile it inside the trained "
                        f"envelope, so no on-grid schedule exists")
                    continue
            # None means no `MiniMaxH3SigmaShift` in the graph, which is how
            # every PDD graph ships since 2026-08-31. The render then runs the
            # checkpoint's class default, so that is the pair the fused
            # boundaries must be graded against. Read from ComfyUI's own model
            # config rather than retyped, for the same reason `comfy_grid`
            # above refuses to guess when ComfyUI is unreachable: a literal
            # here would agree with itself after core moved the real value.
            if found.shift is None:
                try:
                    comfy_on_cpu()
                    from comfy.supported_models import MiniMaxH3
                except ImportError as exc:
                    raise AssertionError(
                        f"{rel} carries no MiniMaxH3SigmaShift, so its shift is "
                        f"the checkpoint's own default -- and ComfyUI is not "
                        f"importable from {COMFY} to read it: {exc}") from exc
                sv = float(MiniMaxH3.sampling_settings["shift"])
                sa = float(MiniMaxH3.sampling_settings["audio_shift"])
            else:
                sv, sa = found.shift
            if found.scheduler == "manual":
                # No scheduler produces this vector -- the graph STATES it, so
                # grading it through `calculate_sigmas` is a category error and
                # raised `invalid scheduler manual` until 2026-08-31. The
                # honest comparison is the vector the graph carries against the
                # boundaries the heads were fused at, which is exactly what
                # `partition_bounds` computes for an uneven tiling.
                stated = h3_config.manual_sigmas(json.loads(path.read_text()))
                if stated is None or widths is None:
                    bad.append(f"{rel}: scheduler is `manual` but its vector "
                               f"or its envelope partition could not be read")
                    continue
                want_v = [1.0 - float(t)
                          for t in partition_bounds(sv, grid, widths).tolist()]
                dev = deviation(stated, want_v)
                if dev > 1e-5:
                    bad.append(
                        f"{rel}: its ManualSigmas vector deviates {dev:.6f} "
                        f"from the boundaries its heads were fused at "
                        f"(partition {widths}, shift {sv})")
                continue
            # A graph that samples the node's own SIGMAS output at a count
            # `simple` cannot reach exactly (1000 % steps != 0, i.e. 16 and 32;
            # `check_pdd_sigmas.EXACT_STEPS`) samples the node's emitted grid,
            # not `simple`'s. Grade what it samples: the node's vector, on
            # video. Audio's sigma follows the same base time inside the model.
            sig = doc.get("10", {}).get("inputs", {}).get("sigmas")
            node_sourced = (isinstance(sig, list) and doc.get(str(sig[0]), {}).get("class_type")
                            == "MiniMaxH3PDDLoRA")
            if node_sourced and 1000 % found.steps != 0:
                from pdd_lora import emit_sigmas
                got = [float(x) for x in emit_sigmas(sv, grid, grid // nfe)]
                want = [1.0 - float(t) for t in block_bounds(sv, grid, grid // nfe).tolist()]
                dev = deviation(got, want)
                if dev > 1e-6:
                    bad.append(f"{rel}: the node's emitted sigmas deviate {dev:.5f} "
                               f"from its fused boundaries at {found.steps} steps")
                continue
            video, audio = comfy_grid(sv, sa, found.scheduler, found.steps)
            for label, got, shift in (("video", video, sv), ("audio", audio, sa)):
                bounds = (partition_bounds(shift, grid, widths) if widths
                          else block_bounds(shift, grid, grid // nfe))
                want = [1.0 - float(t) for t in bounds.tolist()]
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
        # loader.
        assert seen, ("no API graph was recognised as loading a PDD LoRA, so "
                      "this case graded nothing and passed. Check that the "
                      "walk still matches `*_api.json` and that read_api "
                      "still sees MiniMaxH3PDDLoRA.")
        return f"{seen} PDD graph(s), exact on both streams"

    check("pdd graphs on their fused grid", pdd_graphs_on_their_fused_grid)

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
