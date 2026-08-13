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

    check("every ref graph seen", every_ref_graph_seen)
    check("prompt labels match the wired references", labels_agree)
    check("no undefined subject labels", subjects_resolve)

    print(f"\n{len(failures)} failure(s)" if failures else
          "\nall ok -- every ref prompt names exactly what its graph wires")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
