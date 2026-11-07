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
  pika graphs on grid    every Pika H3 Flex graph samples its shift-12 ManualSigmas
                         grid and derives the matching shift-3 audio sigmas
                         through `time_shift_sigma`, both streams

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
from check_distill_settings import (  # noqa: E402
    classify_ph3_image, classify_pika, read_api,
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

    def pika_graphs_on_their_distilled_grid():
        """A Pika H3 Flex graph samples its shift-12 ManualSigmas grid and derives
        the matching shift-3 audio sigmas through `time_shift_sigma`."""
        try:
            comfy_on_cpu()
            from comfy.ldm.minimax.model import time_shift_sigma
            from comfy.supported_models import MiniMaxH3
        except ImportError as exc:
            raise AssertionError(
                f"ComfyUI is not importable from {COMFY}: {exc}") from exc
        want_v = [float(x) for x in h3_config.PIKA_MANUAL_SIGMAS.split(",") if x.strip()]
        bad, seen = [], 0
        for path in graph_paths(WORKFLOWS, "*_api.json"):
            doc = json.loads(path.read_text())
            found = read_api(doc)
            names = [n for n in found.loras if classify_pika(n)]
            if not names:
                continue
            seen += 1
            rel = path.relative_to(REPO)
            if found.steps != h3_config.PIKA_STEPS or found.scheduler != "manual":
                bad.append(f"{rel}: loads a Pika LoRA but its scheduler/steps are "
                           f"{found.scheduler!r}/{found.steps}")
                continue
            sv, sa = (found.shift if found.shift is not None else (
                float(MiniMaxH3.sampling_settings["shift"]),
                float(MiniMaxH3.sampling_settings["audio_shift"])))
            stated = h3_config.manual_sigmas(doc)
            if stated is None:
                bad.append(f"{rel}: ManualSigmas vector could not be read")
                continue
            dev_v = deviation(stated, want_v)
            if dev_v > 1e-6:
                bad.append(f"{rel}: ManualSigmas deviates {dev_v:.6f} from PIKA_MANUAL_SIGMAS")
                continue
            got_a = [float(time_shift_sigma(s, sv, sa)) if s > 0 else 0.0 for s in stated]
            want_a = []
            for s in stated:
                if s <= 0:
                    want_a.append(0.0)
                else:
                    q = s / (sv - (sv - 1.0) * s)
                    want_a.append(sa * q / (1.0 + (sa - 1.0) * q))
            dev_a = deviation(got_a, want_a)
            if dev_a > 1e-6:
                bad.append(f"{rel}: audio sigmas deviate {dev_a:.6f} from inverted shift-{sa} grid")
        assert not bad, "\n         ".join(bad)
        assert seen, ("no API graph was recognised as loading a Pika LoRA, so "
                      "this case graded nothing and passed.")
        return f"{seen} Pika graph(s), exact on both streams"

    check("pika graphs on their distilled grid", pika_graphs_on_their_distilled_grid)

    def ph3_image_graphs_on_their_distilled_grid():
        """Every P-H3-Image graph samples its 8-step non-uniform ManualSigmas
        grid at shift=(1.0, 1.0) so `time_shift_sigma` is the identity."""
        try:
            comfy_on_cpu()
            from comfy.ldm.minimax.model import time_shift_sigma
        except ImportError as exc:
            raise AssertionError(
                f"ComfyUI is not importable from {COMFY}: {exc}") from exc
        want_v = list(h3_config.PH3_IMAGE_SIGMA_KNOTS)
        bad, seen = [], 0
        for path in graph_paths(WORKFLOWS, "*_api.json"):
            doc = json.loads(path.read_text())
            found = read_api(doc)
            if not classify_ph3_image(doc, found):
                continue
            seen += 1
            rel = path.relative_to(REPO)
            if found.steps != h3_config.PH3_IMAGE_STEPS or found.scheduler != "manual":
                bad.append(f"{rel}: P-H3-Image scheduler/steps are "
                           f"{found.scheduler!r}/{found.steps}")
                continue
            if found.shift != h3_config.PH3_IMAGE_SHIFT:
                bad.append(f"{rel}: P-H3-Image shift is {found.shift}, "
                           f"want {h3_config.PH3_IMAGE_SHIFT}")
                continue
            stated = h3_config.manual_sigmas(doc)
            if stated is None:
                bad.append(f"{rel}: ManualSigmas vector could not be read")
                continue
            dev_v = deviation(stated, want_v)
            if dev_v > 1e-6:
                bad.append(f"{rel}: ManualSigmas deviates {dev_v:.6f} from "
                           f"PH3_IMAGE_SIGMA_KNOTS")
                continue
            sv, sa = found.shift
            got_a = [float(time_shift_sigma(s, sv, sa)) if s > 0 else 0.0 for s in stated]
            dev_a = deviation(got_a, want_v)
            if dev_a > 1e-6:
                bad.append(f"{rel}: time_shift_sigma at shift=({sv}, {sa}) deviated "
                           f"{dev_a:.6f} from identity")
        assert not bad, "\n         ".join(bad)
        assert seen, ("no API graph was recognised as P-H3-Image, so this case "
                      "graded nothing and passed.")
        return f"{seen} P-H3-Image graph(s), exact on the 8-step grid"

    check("p-h3-image graphs on their distilled grid", ph3_image_graphs_on_their_distilled_grid)

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
