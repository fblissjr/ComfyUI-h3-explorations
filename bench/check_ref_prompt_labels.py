#!/usr/bin/env python3
"""Check every ref graph's prompt declares exactly the labels it wires.

A `ref2va` prompt refers to its references by label -- `<Picture i>`,
`<Video k>`, `<Audio j>` -- and `MiniMaxH3Tokenizer` decides what those labels
ARE from the sockets that are wired, not from the prompt. So the two can
disagree silently: a prompt can name `<Audio 1>` on a graph that wires no
audio, or number a standalone clip `<Audio 1>` when the tokenizer has already
spent that ordinal on a video's soundtrack.

The emission order is fixed (`comfy/text_encoders/minimax.py`): images, then
videos with each soundtrack's `<Audio j>` immediately BEFORE its `<Video k>`,
then standalone audio -- with a separate 1-based counter per type. One video
with sound plus one standalone clip is therefore `<Audio 1>`, `<Video 1>`,
`<Audio 2>`.

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

    python bench/check_ref_prompt_labels.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / "workflows"

REF_NODE = "MiniMaxH3ReferenceToVideo"


def wired_labels(inputs):
    """The labels the tokenizer will emit, in its own order and numbering."""
    def count(prefix):
        return sum(1 for k, v in inputs.items()
                   if k.startswith(prefix) and v is not None)

    n_img = count("ref_images.ref_image_")
    n_vid = count("ref_videos.ref_video_")
    n_vaud = count("ref_video_audios.ref_video_audio_")
    n_aud = count("ref_audios.ref_audio_")

    labels, audio_n = [], 0
    labels += [f"<Picture {i + 1}>" for i in range(n_img)]
    for k in range(n_vid):
        # a soundtrack's label is emitted BEFORE its video, and only the
        # index-paired one counts
        if k < n_vaud:
            audio_n += 1
            labels.append(f"<Audio {audio_n}>")
        labels.append(f"<Video {k + 1}>")
    for _ in range(n_aud):
        audio_n += 1
        labels.append(f"<Audio {audio_n}>")
    return labels


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
    for path in sorted(WORKFLOWS.glob("*_api.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for node in doc.values():
            if isinstance(node, dict) and node.get("class_type") == REF_NODE:
                graphs.append((path.name, node["inputs"]))

    def every_ref_graph_seen():
        assert graphs, "no shipped graph carries a MiniMaxH3ReferenceToVideo"
        print(f"        ({len(graphs)} ref graph(s))")

    def labels_agree():
        bad = []
        for name, inputs in graphs:
            prompt = inputs.get("prompt", "")
            if not isinstance(prompt, str):
                continue
            want = wired_labels(inputs)
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
        for name, inputs in graphs:
            prompt = inputs.get("prompt", "")
            if not isinstance(prompt, str) or "subject_definitions:" not in prompt:
                continue
            defined = set(re.findall(r"^(<Subject \d+>) is", prompt, re.M))
            used = set(re.findall(r"<Subject \d+>", prompt))
            if used - defined:
                bad.append(f"{name}: uses undefined {sorted(used - defined)}")
        assert not bad, "\n         ".join(bad)

    def force_rate_is_24():
        bad = []
        for path in sorted(WORKFLOWS.glob("*_api.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            # which loaders actually feed a reference socket
            feeding = set()
            for node in doc.values():
                if not isinstance(node, dict) or node.get("class_type") != REF_NODE:
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
        legal = set()
        for imgs in (True, False):
            for vid in (True, False):
                for vaud in (True, False):
                    for aud in (True, False):
                        for vrole in ("structure", "edit", "continue", "motion"):
                            for arole in ("music", "voice", "copy"):
                                legal.add(bw._ref_prompt(
                                    images=imgs, video=vid, video_audio=vaud,
                                    audio=aud, video_role=vrole, audio_role=arole))
        bad = [name for name, inputs in graphs
               if isinstance(inputs.get("prompt"), str)
               and inputs["prompt"] not in legal]
        assert not bad, (
            "these graphs carry a prompt `_ref_prompt` cannot produce, so a "
            "hand-edit has diverged from the generator: " + ", ".join(bad))

    check("every ref graph seen", every_ref_graph_seen)
    check("baked prompts match the generator", prompts_match_the_generator)
    check("reference videos are resampled to 24 fps", force_rate_is_24)
    check("prompt labels match the wired references", labels_agree)
    check("no undefined subject labels", subjects_resolve)

    print(f"\n{len(failures)} failure(s)" if failures else
          "\nall ok -- every ref prompt names exactly what its graph wires")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
