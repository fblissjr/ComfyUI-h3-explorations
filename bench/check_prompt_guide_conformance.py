#!/usr/bin/env python3
"""Check every shipped ref prompt against the OFFICIAL guide, not against us.

`check_ref_prompt_labels.py` already rebuilds every prompt `_ref_prompt` can
produce and asserts each shipped graph carries one verbatim. That guards
hand-edits, and it is worth having, but it cannot catch a generator that is
confidently wrong: the drift guard compares the generator to itself, so when
`_ref_prompt` emitted a hardcoded `[reference generation]` on all fourteen
arms -- including the two edits and the continuation -- it passed clean. The
axis those arms exist to vary had collapsed and every check was green.

So this one takes its vocabulary from
`internal/official_prompt_guides/*_ref_en.md` by parsing the guide's own
tables. When the guide and the generator disagree, the guide wins, and no
number in here is computed by this file.

Claims, i.e. what breaks if a case is deleted:
  guide parsed        the four tables were actually found. Without this the
                      whole file degrades into asserting membership in empty
                      sets, which passes for everything
  six sections        all six sections appear, in the guide's order, with no
                      invented ones. A rewrite missing a section is not a
                      ref2va prompt, whatever else it says
  legal task types    the `[...]` prefix uses only types from section 3.2's
                      table, combined with ` + `, with no type repeated --
                      the guide states both rules explicitly
  keyframe type       `keyframe completion` appears only in a graph that can
                      honour it. It is legal vocabulary that used to be
                      structurally inert: MiniMaxH3ReferenceToVideo has no
                      keyframe socket, and the reference's `select_block`
                      returns "ref2va" whenever references are present and
                      silently DROPS image / last_image.
                      **That reasoning expired on 2026-08-13.** ComfyUI added
                      `MiniMaxH3AddGuide`, a separate conditioning-stage node
                      that appends to `minimax_keyframes`, and
                      `comfy/model_base.py` merges those with `minimax_refs`
                      additively rather than choosing between them. So the
                      mechanism now exists and the guide's combined
                      `[video continuation + keyframe completion]` is
                      buildable. The case therefore checks the GRAPH, not the
                      vocabulary: claiming the task type without wiring a node
                      that can deliver it is still asking for nothing
  markers stay in set visual labels take only 4.1's markers and <Audio N>
                      takes only 4.2's. The two sets share `weak_reference`
                      and nothing else, so a crossed marker is otherwise a
                      plausible-looking string that means nothing
  dialogue placement  `<d>` appears only inside detailed_description. Guide
                      section 6: "Write complete dialogue and lyrics only
                      inside `<d>` in `detailed_description`; do not repeat
                      them in these two sections." The voice arm put its only
                      spoken line in overall_soundscape, where nothing
                      anchors it in time

What this does NOT catch, stated so nobody reads a green run as more than it
is: whether a legal marker is on the RIGHT entity. `attribute_transfer` on
the recipient of a transfer instead of its source is set-legal and backwards,
and only reading the sentence finds it.

Reads the shipped API graphs and the guide. No CUDA, no model, no ComfyUI
import. Exits 2 when the guide is absent, so a skip cannot be mistaken for a
pass.

    python bench/check_prompt_guide_conformance.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / "workflows"
GUIDE = (REPO / "internal" / "official_prompt_guides"
         / "minimax-h3-official-VIDEO_PROMPT_WRITING_GUIDE_ref_en.md")

REF_NODE = "MiniMaxH3ReferenceToVideo"

# The node that makes `keyframe completion` real. Added to ComfyUI on
# 2026-08-13; before it, the task type was legal vocabulary with no mechanism
# behind it on the reference path.
GUIDE_NODE = "MiniMaxH3AddGuide"

# Graphs whose prompt is UNSTRUCTURED ON PURPOSE, and only those.
#
# h3_probe_prompt_concise is the twin of h3_ref_video_swap: identical clip,
# image, seed, canvas and length, differing in nothing but whether the prompt
# is six sections or one paragraph. It exists to measure whether the format
# earns its tokens, so failing it for lacking the format would delete the
# experiment.
#
# The waiver is deliberately narrow. It suppresses ONLY the two structural
# cases -- section presence/order, and the task-type prefix. Marker sets,
# dialogue placement and (in check_ref_prompt_labels) label agreement still
# apply, because none of those are what the probe is varying: an
# unstructured prompt still may not name a reference the graph does not wire.
#
# Every run prints what it waived. An exemption nobody sees is an exemption
# that grows.
_STRUCTURE_PROBES = {"h3_probe_prompt_concise_api"}

# A markdown row of the form `| `value` | prose |`, which is how every table
# in the guide names its vocabulary.
_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|")


def _table_after(lines: list[str], anchor: str) -> list[str]:
    """The `value` column of the first markdown table at/after `anchor`."""
    for i, line in enumerate(lines):
        if anchor in line:
            out = []
            for row in lines[i:]:
                m = _ROW.match(row)
                if m:
                    out.append(m.group(1))
                elif out:
                    break          # table ended
            return out
    return []


def parse_guide(text: str):
    lines = text.splitlines()
    return {
        "sections": _table_after(lines, "| Section | Purpose |"),
        "task_types": _table_after(lines, "| Task type | When to use it |"),
        "visual": _table_after(
            lines, "`<Subject N>`, `<Picture N>`, and `<Video N>` use the following"),
        "audio": _table_after(lines, "`<Audio N>` uses the following"),
    }


def graphs_with_guide() -> set:
    """Graph stems that wire a keyframe-guide node.

    Read per graph rather than assumed globally: the point of the case is that
    a prompt may claim `keyframe completion` only when its own graph can
    deliver it, and a repo-wide "the node exists now" would answer a different
    question.
    """
    out = set()
    for path in sorted(WORKFLOWS.glob("*_api.json")):
        wf = json.loads(path.read_text())
        if any(n.get("class_type") == GUIDE_NODE for n in wf.values()):
            out.add(path.stem)
    return out


def ref_prompts() -> dict[str, str]:
    """{graph name: baked prompt} for every shipped API graph wiring refs."""
    out = {}
    for path in sorted(WORKFLOWS.glob("*_api.json")):
        wf = json.loads(path.read_text())
        if not any(n.get("class_type") == REF_NODE for n in wf.values()):
            continue
        # Taken from the reference node's own `prompt`, NOT by sniffing for
        # "subject_definitions:" in any string. Content sniffing looks more
        # robust and is the opposite: the one graph that matters most here --
        # the deliberately unstructured probe -- has no section headers at
        # all, so a content match skipped it silently and reported a clean
        # pass over a graph it never read.
        for node in wf.values():
            if node.get("class_type") != REF_NODE:
                continue
            value = (node.get("inputs") or {}).get("prompt")
            if isinstance(value, str):
                out[path.stem] = value
    return out


def split_sections(prompt: str, names: list[str]) -> dict[str, str]:
    """{section: body}, splitting on the `name:` headers the guide defines."""
    idx = []
    for name in names:
        m = re.search(rf"^{re.escape(name)}:", prompt, re.M)
        if m:
            idx.append((m.start(), m.end(), name))
    idx.sort()
    body = {}
    for i, (_s, e, name) in enumerate(idx):
        end = idx[i + 1][0] if i + 1 < len(idx) else len(prompt)
        body[name] = prompt[e:end].strip()
    return body


def main() -> int:
    if not GUIDE.exists():
        print(f"SKIP: official guide not found at {GUIDE}")
        print("      This check has no opinion of its own -- its vocabulary IS")
        print("      the guide -- so without it there is nothing to assert.")
        return 2

    g = parse_guide(GUIDE.read_text())
    prompts = ref_prompts()
    fails: list[str] = []

    def ok(label, bad):
        if bad:
            fails.append(label)
            print(f"  FAIL  {label}")
            for b in bad[:8]:
                print(f"          {b}")
        else:
            print(f"  ok    {label}")

    print(f"ref prompts against {GUIDE.name}")
    print(f"        ({len(prompts)} ref graph(s), "
          f"{len(g['task_types'])} task types, "
          f"{len(g['visual'])} visual / {len(g['audio'])} audio markers)")

    # 0. The guide actually parsed. Everything below is set membership, and
    #    membership in an empty set fails open, so this case is what stops a
    #    reformatted guide from turning the whole file into a no-op.
    missing = [k for k, v in g.items() if not v]
    ok("guide parsed", [f"empty table: {k}" for k in missing])
    if missing or not prompts:
        if not prompts:
            fails.append("ref graphs found")
            print("  FAIL  ref graphs found\n          no ref graph carried a prompt")
        return 1

    sections = g["sections"]
    types_ok, visual_ok, audio_ok = (set(g["task_types"]), set(g["visual"]),
                                     set(g["audio"]))

    bad_sections, bad_types, bad_keyframe = [], [], []
    bad_marker, bad_dialogue = [], []

    with_guide = graphs_with_guide()
    waived = sorted(_STRUCTURE_PROBES & set(prompts))
    for name, prompt in sorted(prompts.items()):
        body = split_sections(prompt, sections)
        structural = name not in _STRUCTURE_PROBES
        has_guide = name in with_guide

        # 1. Six sections, in the guide's order.
        found = [s for s in sections if s in body]
        if structural and found != sections:
            bad_sections.append(f"{name}: {found or 'none'}")

        # 2. The `[...]` task-type prefix.
        m = re.match(r"\s*\[([^\]]*)\]", body.get("summary", ""))
        if not m:
            if structural:
                bad_types.append(f"{name}: summary has no [task type] prefix")
        else:
            declared = [t.strip() for t in m.group(1).split("+")]
            for t in declared:
                if t not in types_ok:
                    bad_types.append(f"{name}: {t!r} is not a guide task type")
            if len(declared) != len(set(declared)):
                bad_types.append(f"{name}: repeats a type -- {m.group(1)!r}")
            if "keyframe completion" in declared and not has_guide:
                bad_keyframe.append(
                    f"{name}: claims `keyframe completion` but wires no "
                    f"{GUIDE_NODE}. The reference node alone cannot honour it, "
                    "so the prompt asks for something the graph cannot do")

        # 3. Markers never cross their set.
        for line in body.get("retention_analysis", "").splitlines():
            lm = re.match(r"\s*<(Subject|Picture|Video|Audio)\s*\d+>[^:]*:\s*(\w+)", line)
            if not lm:
                continue
            kind, marker = lm.group(1), lm.group(2)
            allowed = audio_ok if kind == "Audio" else visual_ok
            if marker not in allowed:
                which = "audio" if kind == "Audio" else "visual"
                bad_marker.append(f"{name}: <{kind}> takes {marker!r}, not a {which} marker")

        # 4. Dialogue lives in detailed_description and nowhere else.
        for sec, text in body.items():
            if "<d>" in text and sec != "detailed_description":
                bad_dialogue.append(f"{name}: <d> in {sec}")

    # Name the waiver on every run, pass or fail. A silent exemption is how a
    # check quietly stops covering the thing it was written for.
    note = f"  ({len(waived)} structure probe waived: {', '.join(waived)})" if waived else ""
    ok("six sections, in order" + note, bad_sections)
    ok("legal task types" + note, bad_types)
    ok("keyframe completion only where a guide node is wired", bad_keyframe)
    ok("markers stay in their set", bad_marker)
    ok("dialogue only in detailed_description", bad_dialogue)

    print()
    if fails:
        print(f"FAIL -- {len(fails)} case(s): {', '.join(fails)}")
        return 1
    print("all ok -- every shipped ref prompt conforms to the official guide")
    return 0


if __name__ == "__main__":
    sys.exit(main())
