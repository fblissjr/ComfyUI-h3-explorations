#!/usr/bin/env python3
"""Check every ref graph's prompt declares exactly the labels it wires.

A `ref2va` prompt refers to its references by label -- `<Picture i>`,
`<Video k>`, `<Audio j>` -- and `MiniMaxH3Tokenizer` decides what those labels
ARE from the sockets that are wired, not from the prompt. So the two can
disagree silently: a prompt can name `<Audio 1>` on a graph that wires no
audio, or number a standalone clip `<Audio 1>` when the tokenizer has already
spent that ordinal on a video's soundtrack.

Native socket emission order is fixed (`comfy/text_encoders/minimax.py`):
images, then videos with each soundtrack's `<Audio j>` immediately BEFORE its
`<Video k>`, then standalone audio. This repo's typed surface uses list order
and video ownership; the generator deliberately emits that same sequence so
existing prompts keep their ordinals. Both use separate 1-based counters per
type. One sounded video plus one standalone clip is therefore `<Audio 1>`,
`<Video 1>`, `<Audio 2>` in the shipped ordering.

Claims, i.e. what breaks if a case is deleted:
  no undeclared label   the prompt never names a label the graph does not
                        wire. This is the failure that reads as a model
                        problem: the render succeeds and quietly ignores an
                        instruction about something that is not there
  no unused reference   every wired reference is named at least once. A
                        reference nothing refers to still costs its rows on
                        every sampling step, which is the most expensive way
                        to say nothing
  ordinals match        the numbering follows the tokenizer's per-type
                        counters, including a video soundtrack taking <Audio 1>
                        ahead of a standalone clip
  every ref graph seen  the walk found the shipped ref graphs. Without it a
                        rename turns this into a silent pass over nothing
  discovery is total    no directory under workflows/ holds graphs that
                        discovery cannot see. The case above only asserts the
                        set is non-empty, which a PARTIAL walk also satisfies:
                        demonstrated 2026-08-16, where a stale GRAPH_DIRS made
                        this file and check_prompt_guide_conformance both exit
                        0 while covering 20 ref graphs instead of 28
  no baked drift        every shipped prompt is one the generator can still
                        produce, or a declared authored constant. Catches a
                        hand-edited graph AND a generator constant changed
                        without a rebuild, since it compares by value
  subjects resolve      no <Subject N> is used without being defined
  subjects act          every DEFINED subject is also cited in
                        detailed_description, where guide 5.3 puts the
                        requirement. Added because the dangling-direction case
                        above passes clean on a prompt that defines two
                        subjects and cites neither
  environments subtract generic environment definitions assert no attributes.
                        The forbidden-phrase list is an artifact list, not a
                        style rule: chalet/timber/veranda are the parts of the
                        house that appeared on a lake. **The render behind it
                        is not in `bench/results/`** -- treat it as a
                        regression guard on text, not as a measured mechanism
  force_rate is 24      every VHS_LoadVideo feeding a reference socket resamples
                        onto 24. ComfyUI's node has no fps input and assumes 24
                        twice -- the DiT's temporal clock and the
                        "<T.T seconds>" labels -- so a source at another rate is
                        conditioned at the wrong speed with nothing said.
                        MEASURED on trimmed 6.00s clips: at force_rate=0 a
                        25 fps source is read as 4.2% longer than it is, and a
                        30 fps source as 7.292s instead of 6.000s, a 25%
                        stretch whose last conditioner label reads
                        "<7.0 seconds>" against the correct "<5.2 seconds>".
                        A 24 fps source is unaffected either way, which is why
                        testing on one proves nothing

Reads the shipped API graphs. No CUDA, no model, no ComfyUI import.

    uv run --active --no-sync python bench/check_ref_prompt_labels.py
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / "workflows"

# Graph discovery comes from h3_config, not from a `WORKFLOWS.glob("*.json")`
# here. That glob is non-recursive, so when the image graphs moved into
# `workflows/image/` on 2026-08-16 every walker that kept it would have gone on
# passing over a set that no longer contained them -- and this file's own
# "every ref graph seen" case would still have found the video ones and
# reported ok. Correctly-absent and silently-dropped look identical from a
# green run, which is the failure this repo keeps naming.
sys.path.insert(0, str(REPO / "workflows"))
sys.path.insert(0, str(REPO))
from h3_config import GRAPH_DIRS, graph_paths  # noqa: E402

REF_NODES = ("MiniMaxH3ReferenceToVideo", "MiniMaxH3ReferenceConditioning")


def wired_labels(inputs, graph=None):
    """The labels the tokenizer will emit, in its own order and numbering.

    **A thin front end over `reference_order` since 2026-08-22**, not a second
    authority. It used to compute the answer itself, and computed one case
    wrong: soundtracks were reduced to a COUNT and paired `k < n_vaud`, which
    cannot represent a sparse socket set. Core pairs by suffix
    (`comfy_extras/nodes_minimax_h3.py:313-314`), so videos 0 and 1 with only
    `ref_video_audio_1` wired put the track on the SECOND clip, and this
    function put it on the first. `bench/check_reference_order.py` drives core
    to establish that, and both consumers now share one label function so the
    disagreement cannot recur.

    `graph` is optional and selects the model:

      absent, or a node wiring `ref_*` sockets -> the legacy grouped plan
      a node wiring `references`               -> the typed ordered chain

    Both produce the same kind of record list and go through the same
    `assign_labels`, which is the point: an ordered graph and a socket graph
    are two ways to build a plan, not two labelling rules.
    """
    from reference_order import assign_labels, plan_for
    return assign_labels(plan_for(inputs, graph))


def main():
    failures, seen = [], 0

    def check(name, fn):
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {exc}")

    print("ref graph prompts against the labels they wire")

    graphs = []
    for path in graph_paths(WORKFLOWS, "*_api.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for node in doc.values():
            if isinstance(node, dict) and node.get("class_type") in REF_NODES:
                # the whole doc rides along: an ordered graph's plan lives
                # in the append chain, not in this node's inputs
                graphs.append((path.name, node["inputs"], doc))

    def every_ref_graph_seen():
        assert graphs, f"no shipped graph carries one of {REF_NODES}"
        print(f"        ({len(graphs)} ref graph(s))")

    def no_graph_directory_is_invisible():
        """Nothing under `workflows/` holds graphs that discovery cannot see.

        **This case exists because the failure it catches was demonstrated,
        not imagined.** On 2026-08-16 the image graphs moved into
        `workflows/image/`. Run against a `GRAPH_DIRS` that still said `("",)`,
        this file and `check_prompt_guide_conformance` both **exited 0 while
        covering 20 ref graphs instead of 28** -- no error, no warning, a
        smaller number printed in a line nobody has a prior for.

        `assert graphs` above cannot catch that: the video graphs are still
        there, so the set is non-empty and every case passes over the subset it
        can see. The only thing that can catch it is comparing the discovered
        set against what is actually on disk.

        `bench/` and `archive/` are excluded ON PURPOSE and are named here so
        the exclusion is visible: the stamped bench graphs read another pack's
        internals and are expected to break, and the archive is history.
        Neither should be graded against the live schema. `archive` is retained
        for a `workflows/archive/` that does not currently exist -- the parked
        image graphs went to the repo-root `archive/workflows/image/` on
        2026-08-27, which is outside `WORKFLOWS` and so outside this walk
        already.
        """
        excluded = {"bench", "archive", "__pycache__"}
        on_disk = set()
        for p in WORKFLOWS.rglob("*.json"):
            rel = p.relative_to(WORKFLOWS)
            if set(rel.parts) & excluded:
                continue
            # GRAPH_DIRS spells the root as "", where `relative_to` gives "."
            on_disk.add("" if rel.parent == Path(".")
                        else rel.parent.as_posix())
        seen = set(GRAPH_DIRS)
        missed = sorted(on_disk - seen)
        assert not missed, (
            f"{missed} under workflows/ hold graphs that no check walks. Add "
            f"them to h3_config.GRAPH_DIRS or to this case's `excluded` set -- "
            f"discovery currently covers {sorted(seen)}, and a directory "
            f"missing from it fails SILENTLY, with every case still green")

    def labels_agree():
        bad = []
        for name, inputs, doc in graphs:
            prompt = inputs.get("prompt", "")
            if not isinstance(prompt, str):
                continue
            want = wired_labels(inputs, doc)
            used = set(re.findall(r"<(?:Picture|Video|Audio) \d+>", prompt))
            undeclared = sorted(used - set(want))
            unused = sorted(set(want) - used)
            if undeclared:
                bad.append(f"{name}: prompt names {undeclared} but the graph "
                           f"wires {want}")
            if unused:
                bad.append(f"{name}: wires {unused} and the prompt never "
                           f"refers to them; they cost rows on every step")
        assert not bad, "\n         ".join(bad)

    def subjects_resolve():
        # <Subject N> is prompt-internal, not a tokenizer label, but a subject
        # defined and never used (or used and never defined) is the same class
        # of dangling reference.
        bad = []
        for name, inputs, _doc in graphs:
            prompt = inputs.get("prompt", "")
            if not isinstance(prompt, str) or "subject_definitions:" not in prompt:
                continue
            defined = set(re.findall(r"^(<Subject \d+>) is", prompt, re.M))
            used = set(re.findall(r"<Subject \d+>", prompt))
            if used - defined:
                bad.append(f"{name}: uses undefined {sorted(used - defined)}")
        assert not bad, "\n         ".join(bad)

    def subjects_cited_where_they_act():
        """Every defined <Subject N> is named in detailed_description.

        Guide 5.3 asks for each label at its first real appearance AND where
        its role applies, not merely once in `subject_definitions`. A subject
        defined and then described only as "he" or "the likeness" leaves the
        binding between the reference and the action to inference.

        **This case exists because it was MISSED, not predicted.** The version
        above checks that every subject USED is defined -- the dangling
        direction -- and passes clean on a prompt that defines two subjects and
        cites neither. Six of the eight image graphs shipped that way on
        2026-08-16 and every check here was green; `bench/preflight_graph.py`,
        written in another session, caught it on first contact. That is
        CLAUDE.md's second-reader finding, and the fix is to make the check
        able to see it rather than to rely on the second reader.

        Scoped to `detailed_description` because that is where the guide puts
        the requirement, and skipped for prompts with no such section -- the
        flat structure probe has no sections by construction, and grading it
        here would delete the experiment.
        """
        bad = []
        for name, inputs, _doc in graphs:
            prompt = inputs.get("prompt", "")
            if not isinstance(prompt, str) or "detailed_description:" not in prompt:
                continue
            defined = set(re.findall(r"^(<Subject \d+>) is", prompt, re.M))
            body = prompt.split("detailed_description:", 1)[1]
            # stop at the next section header, if any
            body = re.split(r"^\w+:", body, maxsplit=1, flags=re.M)[0]
            missing = sorted(d for d in defined if d not in body)
            if missing:
                bad.append(f"{name}: defines {missing} but never cites "
                           f"{'it' if len(missing) == 1 else 'them'} in "
                           f"detailed_description, where the role applies")
        assert not bad, "\n         ".join(bad)

    def force_rate_is_24():
        bad = []
        for path in graph_paths(WORKFLOWS, "*_api.json"):
            doc = json.loads(path.read_text(encoding="utf-8"))
            # which loaders actually feed a reference socket
            feeding = set()
            for node in doc.values():
                if (not isinstance(node, dict)
                        or node.get("class_type") not in REF_NODES):
                    continue
                # The typed surface derives `loaded_fps` from its owned
                # VHS_VIDEOINFO and normalizes before either encoder. A
                # force_rate widget is therefore no longer its clock; this
                # legacy gate remains only for the socket node that has no
                # metadata input at all.
                if node.get("class_type") == "MiniMaxH3ReferenceConditioning":
                    continue
                for key, val in (node.get("inputs") or {}).items():
                    if key.startswith(("ref_videos.", "ref_video_audios.")) \
                            and isinstance(val, list) and val:
                        feeding.add(str(val[0]))
            for nid in feeding:
                loader = doc.get(nid, {})
                if loader.get("class_type") != "VHS_LoadVideo":
                    continue
                rate = loader.get("inputs", {}).get("force_rate")
                if rate != 24 and rate != 24.0:
                    bad.append(f"{path.name} node {nid}: force_rate={rate!r}, "
                               "so a non-24fps source is conditioned at the "
                               "wrong speed")
        assert not bad, "\n         ".join(bad)

    def prompts_match_the_generator():
        """A baked prompt that no longer matches its generator is drift.

        The graphs carry their prompt inline, which is what makes them
        editable -- and what lets a hand-edit diverge from `_ref_prompt`
        silently. Rebuild every prompt the GRAPHS table declares and assert
        each shipped graph carries one of them verbatim.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_bw_for_check", REPO / "workflows" / "build_workflows.py")
        if spec is None or spec.loader is None:
            raise AssertionError("could not load build_workflows to compare")
        # build_workflows imports h3_config as a bare name, so its own
        # directory has to be importable; ComfyUI's root has to come FIRST so
        # a later bare `import nodes` finds ComfyUI's and not this repo's.
        import sys as _sys
        for extra in (str(REPO / "workflows"), str(REPO.parents[1])):
            if extra not in _sys.path:
                _sys.path.insert(0, extra)
        try:
            import nodes  # noqa: F401
        except Exception:
            pass
        bw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bw)

        # every prompt `_ref_prompt` can produce for the roles in use
        #
        # `images` is enumerated over ROLE TUPLES, not over (True, False).
        # Until 2026-08-16 it was the bool pair, which could not express a role
        # tuple at all, so the first graph declaring three roles was reported
        # as a hand-edit -- a false positive that accused the graph of the
        # check's own blind spot. The comment below already said a hardcoded
        # copy stops covering the generator "the moment a role is added"; it
        # was right about the risk and still missed it, because what changed
        # was the SHAPE of the argument rather than its value.
        #
        # Built from `IMAGE_ROLES` and `_REF_IMAGE_NODES` so neither the role
        # list nor the socket count is duplicated here.
        image_arms = [False]
        for n in range(1, len(bw._REF_IMAGE_NODES) + 1):
            image_arms += list(itertools.product(bw.IMAGE_ROLES, repeat=n))
        legal = set()
        for imgs in image_arms:
            for vid in (True, False):
                for vaud in (True, False):
                    for aud in (True, False):
                        # Imported, never repeated. A hardcoded copy here
                        # stops covering the generator the moment a role is
                        # added, and reports it as a hand-edit.
                        for vrole in bw.VIDEO_ROLES:
                            for arole in bw.AUDIO_ROLES:
                                legal.add(bw._ref_prompt(
                                    images=imgs, video=vid, video_audio=vaud,
                                    audio=aud, video_role=vrole, audio_role=arole))

        # The single-frame image prompts USED to be enumerated here from
        # `bw._IMAGE_SCENES` x `bw.IMAGE_FORMATS`. Dropped 2026-08-27 with the
        # lane: `build_workflows.py` no longer calls `_image_graphs()`, so no
        # shipped graph can legitimately carry one of those prompts, and adding
        # them to `legal` would widen the accept set to cover text that cannot
        # appear -- an image prompt pasted into a video graph would pass as
        # generated. The generator still defines the tables, so restoring this
        # is un-parking the lane plus these three lines. See
        # `docs/h3_image_editing.md`.
        # `_ref_prompt` also takes `scene=`, and this loop does not enumerate
        # it. That is deliberate and it is the same call as the image prompts
        # above: no call site in `build_workflows.py` passes `scene`, so no
        # shipped graph can legitimately carry one, and adding
        # `REF_SCENE_SHOTS` to `legal` would widen the accept set to cover text
        # that cannot appear. **The cost is a cry-wolf red**: the first graph to
        # wire a scene arm gets accused of being a hand-edit. If you are that
        # person, the fix is to enumerate `bw.REF_SCENE_SHOTS` here in the same
        # breath as wiring it -- not to add your prompt to the list below.
        #
        # This is the 2026-08-16 `images` failure one parameter over: the
        # comment there says a hardcoded copy stops covering the generator "the
        # moment a role is added" and it was right about the class while
        # missing this instance, because again what changed was the SHAPE of
        # the signature rather than a value in it.
        # Authored ref2v prompts, named one at a time on purpose.
        #
        # This case exists to catch a GENERATED graph whose prompt was
        # hand-edited away from `_ref_prompt`. A prompt that is a module-level
        # constant cannot drift from `_ref_prompt` because it never came from
        # it -- but it is also not something the loop above can enumerate, so
        # it has to be declared. Listed individually rather than swept up by a
        # `*_PROMPT` glob: a glob would make every future constant legal
        # silently, and the point of this list is that adding one is a
        # deliberate act somebody can see in a diff.
        for authored in (bw.DIALOGUE_REF2V_PROMPT,
                         bw.MARKET_REF2V_PROMPT):
            legal.add(authored)
        bad = [name for name, inputs, _doc in graphs
               if isinstance(inputs.get("prompt"), str)
               and inputs["prompt"] not in legal]
        assert not bad, (
            "these graphs carry a prompt `_ref_prompt` cannot produce, so a "
            "hand-edit has diverged from the generator: " + ", ".join(bad))

    def no_attribute_assertions_in_environment_templates():
        """Environment templates must be purely subtractive.

        Asserting specific attributes ('architecture', 'palette, and lighting',
        'chalet', 'timber', 'veranda', 'steady interior room tone') in generic
        environment definitions causes the DiT cross-attention to hallucinate
        structures on natural landscapes (e.g., building a house on an alpine lake).
        """
        bad = []
        forbidden_phrases = [
            "architecture", "palette, and lighting", "palette and lighting",
            "timber", "chalet", "veranda", "steady interior room tone",
            "soft directional lighting", "ambient daylight", "harsh sunlight",
            "interior room tone",
        ]
        for name, inputs, _doc in graphs:
            prompt = inputs.get("prompt", "")
            if not isinstance(prompt, str) or "subject_definitions:" not in prompt:
                continue
            for line in prompt.splitlines():
                if "environment in <Picture" in line or "environment shown in <Picture" in line:
                    for phrase in forbidden_phrases:
                        if phrase in line.lower():
                            bad.append(f"{name}: environment definition asserts attribute {phrase!r}: {line!r}")
                if "steady interior" in line.lower():
                    bad.append(f"{name}: soundscape asserts interior room tone: {line!r}")
        assert not bad, "\n         ".join(bad)

    check("every ref graph seen", every_ref_graph_seen)
    check("no graph directory is invisible to discovery",
          no_graph_directory_is_invisible)
    check("baked prompts match the generator", prompts_match_the_generator)
    check("reference videos are resampled to 24 fps", force_rate_is_24)
    check("prompt labels match the wired references", labels_agree)
    check("no undefined subject labels", subjects_resolve)
    check("every defined subject is cited where it acts",
          subjects_cited_where_they_act)
    check("no attribute assertions in environment templates",
          no_attribute_assertions_in_environment_templates)

    print(f"\n{len(failures)} failure(s)" if failures else
          "\nall ok -- every ref prompt names exactly what its graph wires")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
