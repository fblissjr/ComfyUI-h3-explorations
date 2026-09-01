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
`vendor_guides/ref_en.md` by parsing the guide's own
tables. When the guide and the generator disagree, the guide wins, and no
number in here is computed by this file.

Claims, i.e. what breaks if a case is deleted:
  guide parsed        the four tables were actually found. Without this the
                      whole file degrades into asserting membership in empty
                      sets, which passes for everything
  six sections        every section the guide requires OF THAT GRAPH appears,
                      in the guide's order, with no invented ones. A rewrite
                      missing a section is not a ref2va prompt, whatever else
                      it says.
                      Two qualifications, both narrower than they sound.
                      (1) `overall_soundscape` and `non_diegetic_music` are
                      not required of a graph with no `VAEDecodeAudio`,
                      because they describe a track that structurally cannot
                      exist there. Read off the graph, not off a list of
                      names -- which is why the rule survives its only
                      population (the single-frame image path) being parked
                      on 2026-08-27. No shipped graph claims it today and the
                      run prints that.
                      (2) ORDER is now actually checked. Until 2026-08-16 the
                      case was named "in order" and compared against a list
                      built by iterating the guide's own sections, so it could
                      only ever detect a missing one
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
GUIDE = (REPO / "vendor_guides"
         / "ref_en.md")

# THE SECOND GUIDE. The release ships two and they do not share a section list;
# this file graded every prompt against the six-section one and reached only the
# graphs wiring `MiniMaxH3ReferenceToVideo`, so the base-format population was
# not wrong -- it was invisible.
BASE_GUIDE = (REPO / "vendor_guides"
              / "base_en.md")

# Non-recursive `WORKFLOWS.glob` would have stopped seeing the image graphs
# when they moved to `workflows/image/` on 2026-08-16, and this file's counts
# would still have printed a plausible number. See h3_config.GRAPH_DIRS.
sys.path.insert(0, str(REPO / "workflows"))
from h3_config import graph_paths  # noqa: E402

REF_NODES = ("MiniMaxH3ReferenceToVideo",
             "MiniMaxH3ReferenceConditioning")

# Every node carrying a prompt. `MiniMaxH3Conditioning` is this repo's own and
# the fl2va path moved onto it on 2026-08-21.
PROMPT_NODES = REF_NODES + ("MiniMaxH3Conditioning",)

# Full-reference mode is the mode that wires reference labels, so the guide a
# graph is graded against is read off its sockets, never off its filename.
REF_SOCKET_PREFIXES = ("ref_images.", "ref_videos.", "ref_audios.",
                       "ref_video_audios.")


def guide_of(inputs: dict) -> str:
    return "ref" if ("references" in inputs or
                     any(k.startswith(REF_SOCKET_PREFIXES) for k in inputs)) \
        else "base"

# The node that decodes the audio half of the packed AV latent. A graph
# without it has no audio track at all. The single-frame image path used to be
# the case that mattered -- one frame is 0.04s of nothing to decode -- and it
# is parked (2026-08-27), so this currently exempts nothing. Kept because it is
# read off the graph rather than off a name: a video graph that lost its audio
# decoder is the same situation and would still be handled.
AUDIO_DECODE_NODE = "VAEDecodeAudio"

# The two of the guide's six sections that describe the audio track. Spelled
# out rather than sliced off the end of the parsed table, so a guide that
# reorders its sections cannot silently make this exempt something else.
_AUDIO_SECTIONS = {"overall_soundscape", "non_diegetic_music"}

# The node that makes `keyframe completion` real. Added to ComfyUI on
# 2026-08-13; before it, the task type was legal vocabulary with no mechanism
# behind it on the reference path.
GUIDE_NODE = "MiniMaxH3AddGuide"

# Graphs whose prompt is UNSTRUCTURED ON PURPOSE, and only those.
#
# h3_ref_video_swap_concise is the twin of h3_ref_video_swap: identical clip,
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
#
# **Currently empty, and that is a state rather than an oversight.** The only
# member was `h3_image_probe_format_flat_api`, the same experiment on the image
# path, and the single-frame lane was parked on 2026-08-27
# (`docs/h3_image_editing.md`); `h3_ref_video_swap_concise` was retired before
# it. So every shipped prompt is graded structurally, with nothing waived --
# which the run prints, because "waived nothing" and "forgot to waive" are the
# pair this file exists to keep apart.
#
# An entry naming a graph that no longer ships is rot, not a safe leftover: it
# waives nothing while reading as coverage. `probes_are_necessary` below asserts
# that, so restoring one and then dropping its graph goes red.
_STRUCTURE_PROBES: set[str] = set()


def _audio_sections_optional(wf: dict) -> bool:
    """True when a graph has no audio track for the audio sections to describe.

    `overall_soundscape` and `non_diegetic_music` are two of the guide's six
    required sections, and on a graph with no `VAEDecodeAudio` they describe
    something that structurally cannot exist: there is no audio output to
    condition. Requiring them would mean writing a soundscape for a still image
    to satisfy a checker, which is worse than not conforming. The single-frame
    image path was the population; it is parked, so this returns False for
    every shipped graph today.

    **Read off the GRAPH, not off a list of graph names**, because "has an
    audio decoder" is the actual reason and a name is a proxy for it that goes
    stale. A video graph that somehow lost its audio decoder would be granted
    this too -- and would deserve the question that raises.

    The other four sections are still required of these graphs. They are not
    audio-specific and a still frame has subjects, a summary, retention and a
    description exactly like a clip does.
    """
    return not any(n.get("class_type") == AUDIO_DECODE_NODE
                   for n in wf.values())

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


def parse_base_guide(text: str) -> list[str]:
    """base-en's three core fields, in order, parsed TWO independent ways.

    The base guide carries no markdown table, so `_table_after` finds nothing
    in it. The fields are read from the fenced example block under "Three Core
    Fields" and, separately, from the bulleted definitions beneath it. Both
    must agree: a guide reformatted so that only one of them still matches
    would otherwise silently narrow this check's vocabulary, and an empty or
    short list fails open the same way case 0 exists to prevent.
    """
    lines = text.splitlines()
    fenced: list[str] = []
    for i, line in enumerate(lines):
        if "Three Core Fields" not in line:
            continue
        for row in lines[i:]:
            m = re.match(r"^([a-z_]+):", row)
            if m:
                fenced.append(m.group(1))
            elif fenced and row.strip().startswith("```"):
                break
        break
    bullets = re.findall(r"^- \*\*([a-z_]+)\*\*:", text, re.M)
    return fenced if fenced and fenced == bullets[:len(fenced)] else []


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
    for path in graph_paths(WORKFLOWS, "*_api.json"):
        wf = json.loads(path.read_text())
        if any(n.get("class_type") == GUIDE_NODE for n in wf.values()):
            out.add(path.stem)
    return out


def graphs_without_audio() -> set:
    """Graph stems with no audio decoder, so no audio layer to describe."""
    out = set()
    for path in graph_paths(WORKFLOWS, "*_api.json"):
        if _audio_sections_optional(json.loads(path.read_text())):
            out.add(path.stem)
    return out


def ref_prompts() -> dict[str, tuple[str, str]]:
    """{graph name: (baked prompt, guide)} for every shipped API graph.

    Was ref-node-only, and returned a bare prompt. Both widened 2026-08-21: the
    graphs this repo ships now carry their prompt on a node this file did not
    look for, and which guide applies is a property of the graph that the
    caller cannot recover from the prompt text alone.
    """
    out = {}
    for path in graph_paths(WORKFLOWS, "*_api.json"):
        wf = json.loads(path.read_text())
        if not any(n.get("class_type") in PROMPT_NODES for n in wf.values()):
            continue
        # Taken from the reference node's own `prompt`, NOT by sniffing for
        # "subject_definitions:" in any string. Content sniffing looks more
        # robust and is the opposite: the one graph that matters most here --
        # the deliberately unstructured probe -- has no section headers at
        # all, so a content match skipped it silently and reported a clean
        # pass over a graph it never read.
        for node in wf.values():
            if node.get("class_type") not in PROMPT_NODES:
                continue
            ins = node.get("inputs") or {}
            value = ins.get("prompt")
            if isinstance(value, str):
                out[path.stem] = (value, guide_of(ins))
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
    base_sections = (parse_base_guide(BASE_GUIDE.read_text())
                     if BASE_GUIDE.exists() else [])
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

    n_base = sum(1 for _p, gu in prompts.values() if gu == "base")
    print(f"shipped prompts against {GUIDE.name} and {BASE_GUIDE.name}")
    print(f"        ({len(prompts) - n_base} full-reference graph(s) on "
          f"ref-en's {len(g['sections'])} sections, {n_base} on base-en's "
          f"{len(base_sections)}, {len(g['task_types'])} task types, "
          f"{len(g['visual'])} visual / {len(g['audio'])} audio markers)")

    # 0. The guide actually parsed. Everything below is set membership, and
    #    membership in an empty set fails open, so this case is what stops a
    #    reformatted guide from turning the whole file into a no-op.
    missing = [f"empty table: {k}" for k, v in g.items() if not v]
    if not base_sections:
        missing.append(
            "base-en's core fields did not parse -- either the file is absent "
            "or its two extractions disagreed, and an empty vocabulary would "
            "grade every base prompt as conforming")
    ok("both guides parsed", missing)
    if missing or not prompts:
        if not prompts:
            fails.append("ref graphs found")
            print("  FAIL  ref graphs found\n          no ref graph carried a prompt")
        return 1

    sections_for = {"ref": g["sections"], "base": base_sections}
    # ref-en.txt:229 names the pair: "Main field |
    # integrated_multimodal_description | detailed_description".
    main_field = {"ref": "detailed_description",
                  "base": base_sections[0]}
    types_ok, visual_ok, audio_ok = (set(g["task_types"]), set(g["visual"]),
                                     set(g["audio"]))

    bad_sections, bad_types, bad_keyframe = [], [], []
    bad_marker, bad_dialogue = [], []

    with_guide = graphs_with_guide()
    no_audio = graphs_without_audio()
    waived = sorted(_STRUCTURE_PROBES & set(prompts))
    base_skipped = []
    for name, (prompt, guide) in sorted(prompts.items()):
        sections = sections_for[guide]
        body = split_sections(prompt, sections)
        structural = name not in _STRUCTURE_PROBES
        has_guide = name in with_guide
        if guide == "base":
            # Cases 2 and 3 read `summary` and `retention_analysis`, which are
            # ref-en constructs a base prompt correctly does not have. Skipping
            # them is recorded and printed, never silent: an exemption nobody
            # sees is an exemption that grows.
            base_skipped.append(name)

        # 1. Every section the guide requires OF THIS GRAPH, in guide order.
        #
        # `split_sections` keys its result by where each header actually
        # appears, so `list(body)` is the prompt's own order. Comparing
        # against a list built by iterating `sections` -- which the earlier
        # version did -- could only ever detect a MISSING section, never a
        # misordered one, while the case was named "in order". It now checks
        # both.
        required = [s for s in sections
                    if not (name in no_audio and s in _AUDIO_SECTIONS)]
        present = list(body)
        if structural:
            missing = [s for s in required if s not in present]
            if missing:
                bad_sections.append(f"{name}: missing {missing}")
            elif present != [s for s in sections if s in present]:
                bad_sections.append(f"{name}: out of guide order -- {present}")

        # 2. The `[...]` task-type prefix.
        if guide == "base":
            # No `summary` section exists to carry one, and no marker table
            # applies. Case 4 below still runs, against this guide's own main
            # field.
            for sec, text in body.items():
                if "<d>" in text and sec != main_field[guide]:
                    bad_dialogue.append(f"{name}: <d> in {sec}")
            continue
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

        # 4. Dialogue lives in the guide's main field and nowhere else.
        for sec, text in body.items():
            if "<d>" in text and sec != main_field[guide]:
                bad_dialogue.append(f"{name}: <d> in {sec}")

    # Name the waiver on every run, pass or fail, INCLUDING when it is empty.
    # A silent exemption is how a check quietly stops covering the thing it was
    # written for, and an exemption printed only when non-empty makes "nothing
    # waived" indistinguishable from "the waiver list was never consulted".
    note = (f"  ({len(waived)} structure probe waived: {', '.join(waived)})"
            if waived else "  (no structure probe waived)")
    audio_note = (f"  ({len(no_audio)} graph(s) with no audio decoder: the "
                  f"two audio sections are not required of them)" if no_audio
                  else "  (every shipped graph decodes audio: no section waived)")
    stale = sorted(_STRUCTURE_PROBES - set(prompts))
    ok("every structure probe still ships",
       [f"_STRUCTURE_PROBES names {s!r}, which matches no shipped prompt -- "
        f"remove the entry, it waives nothing while reading as coverage"
        for s in stale])
    base_note = (f"  ({len(base_skipped)} base-format graph(s) exempt: the "
                 f"section does not exist in base-en)" if base_skipped else "")
    ok("each prompt has its guide's sections, in order" + note + audio_note,
       bad_sections)
    ok("legal task types" + note + base_note, bad_types)
    ok("keyframe completion only where a guide node is wired" + base_note,
       bad_keyframe)
    ok("markers stay in their set" + base_note, bad_marker)
    ok("dialogue only in the guide's main field", bad_dialogue)

    print()
    if fails:
        print(f"FAIL -- {len(fails)} case(s): {', '.join(fails)}")
        return 1
    print("all ok -- every shipped prompt conforms to the guide that applies "
          "to its graph")
    return 0


if __name__ == "__main__":
    sys.exit(main())
