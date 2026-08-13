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
  no keyframe type    `keyframe completion` never appears. It is legal
                      vocabulary and structurally inert HERE:
                      MiniMaxH3ReferenceToVideo has no keyframe socket, and
                      the reference's `select_block` returns "ref2va" whenever
                      references are present and silently DROPS image /
                      last_image. A prompt asking for it is asking for
                      nothing, with no error to say so
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


def ref_prompts() -> dict[str, str]:
    """{graph name: baked prompt} for every shipped API graph wiring refs."""
    out = {}
    for path in sorted(WORKFLOWS.glob("*_api.json")):
        wf = json.loads(path.read_text())
        if not any(n.get("class_type") == REF_NODE for n in wf.values()):
            continue
        # Found by content, not by node name. Which node owns `prompt` has
        # already moved once, and a name lookup that stops matching reports
        # "0 graphs" as a pass rather than as the breakage it is.
        for node in wf.values():
            for value in (node.get("inputs") or {}).values():
                if isinstance(value, str) and "subject_definitions:" in value:
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

    for name, prompt in sorted(prompts.items()):
        body = split_sections(prompt, sections)

        # 1. Six sections, in the guide's order.
        found = [s for s in sections if s in body]
        if found != sections:
            bad_sections.append(f"{name}: {found or 'none'}")

        # 2. The `[...]` task-type prefix.
        m = re.match(r"\s*\[([^\]]*)\]", body.get("summary", ""))
        if not m:
            bad_types.append(f"{name}: summary has no [task type] prefix")
        else:
            declared = [t.strip() for t in m.group(1).split("+")]
            for t in declared:
                if t not in types_ok:
                    bad_types.append(f"{name}: {t!r} is not a guide task type")
            if len(declared) != len(set(declared)):
                bad_types.append(f"{name}: repeats a type -- {m.group(1)!r}")
            if "keyframe completion" in declared:
                bad_keyframe.append(f"{name}: no keyframe socket on {REF_NODE}")

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

    ok("six sections, in order", bad_sections)
    ok("legal task types", bad_types)
    ok("no keyframe completion", bad_keyframe)
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
