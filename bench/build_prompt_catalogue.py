#!/usr/bin/env python
"""Generate `docs/prompt_catalogue.md`: every prompt this repo renders.

## Why this is generated

The prompts live as constants in `workflows/build_workflows.py` and are baked
into graphs on rebuild. A hand-written catalogue would be a second copy of text
that changes when the generator changes, and the failure mode is not
hypothetical here: on 2026-08-28 a generator comment described the opposite of
the graph it emitted, and nothing noticed. A catalogue that quietly diverges
from what actually renders is worse than no catalogue, because it invites
someone to debug a scene they are not looking at.

So the FACTS are derived -- text, scene name, which graphs carry it -- and the
JUDGEMENTS live next door in `docs/prompt_audit.md`, hand-written and keyed by
the scene names this file emits. Facts that can drift are generated; opinions
that cannot be computed are written.

## What it reads

The graphs, not the constants, are the ground truth for what renders: a
constant nothing references is not a shipped prompt. So this walks
`h3_config.graph_paths()`, pulls every prompt string a conditioner node
actually carries, deduplicates by exact text, and then names each one by
matching it back to a module-level constant in `workflows/build_workflows.py`
via `ast`.

**Most ref2va prompts have no constant to match**, because `_ref_prompt()`
composes them at build time from parts. Those are named `derived:<graph>` from
the family that carries them, and the prefix is there so nobody greps for a
constant that was never written. The first version of this file called them
`(unnamed)`, which read as a defect rather than as how that path works.

## Deliberately not asserted

**Whether a prompt is any good.** This file counts shots, speakers and markers
because those are mechanical. Whether the camera vocabulary is in the guide's
table, whether a speaker is introduced where they first appear, whether the
scene is worth having -- all judgement, all in the audit, none of it here.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "workflows"))
OUT = REPO / "docs" / "prompt_catalogue.md"

CONDITIONERS = ("MiniMaxH3Conditioning", "MiniMaxH3ReferenceConditioning",
                "MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo")

# Markers whose presence is a mechanical fact worth recording per scene.
MARKERS = ("<d>", "<|lyrics_start|>", "<|caption_start|>", "<|cutoff|>",
           "<scenetrans>")

SHOT = re.compile(r"\[Shot\s+(\d+)")
SPEAKER = re.compile(r"\(S(\d+)\)")


def constant_names() -> dict[str, str]:
    """Map prompt text -> constant name, from the generator's own source."""
    src = (REPO / "workflows" / "build_workflows.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    out: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if not isinstance(tgt, ast.Name):
                continue
            v = node.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out.setdefault(v.value, tgt.id)
            elif isinstance(v, ast.Dict):
                # T2V_SCENES / REF_SCENE_* hold several prompts under one name.
                for k, item in zip(v.keys, v.values):
                    if isinstance(item, ast.Constant) and isinstance(item.value, str):
                        key = k.value if isinstance(k, ast.Constant) else "?"
                        out.setdefault(item.value, f"{tgt.id}[{key!r}]")
    return out


def scan_graphs() -> dict[str, set[str]]:
    """prompt text -> the set of graph stems carrying it."""
    import h3_config
    found: dict[str, set[str]] = defaultdict(set)
    for path in h3_config.graph_paths(REPO / "workflows", include_bench=True):
        try:
            graph = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        nodes = graph.values() if isinstance(graph, dict) else []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            ct = node.get("class_type")
            inputs = node.get("inputs")
            if ct in CONDITIONERS and isinstance(inputs, dict):
                p = inputs.get("prompt")
                if isinstance(p, str) and p.strip():
                    found[p].add(Path(path).stem)
    return found


def facts(text: str) -> dict:
    shots = sorted({int(n) for n in SHOT.findall(text)})
    speakers = sorted({int(n) for n in SPEAKER.findall(text)})
    return {
        "words": len(text.split()),
        "shots": len(shots),
        "speakers": len(speakers),
        "markers": [m for m in MARKERS if m in text],
        "sections": [ln.rstrip(":") for ln in text.splitlines()
                     if ln.endswith(":") and not ln.startswith("[")],
    }


def anchor(name: str) -> str:
    """Markdown anchor for a heading, matching how renderers derive one.

    Punctuation is dropped rather than replaced -- `derived:h3_x` anchors as
    `derivedh3-x`. Written out because the first version kept the colon and
    every nav link in the table pointed nowhere.
    """
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in " _-":
            out.append("-")
    return "".join(out)


def render(scenes) -> str:
    o: list[str] = []
    w = o.append
    w("# Every prompt this repo renders")
    w("")
    w("**Generated by `bench/build_prompt_catalogue.py`. Do not hand-edit — "
      "rerun it.** The prompts are constants in "
      "[`build_workflows.py`](../workflows/build_workflows.py) baked into graphs "
      "on rebuild, so a hand-maintained copy would drift from what actually "
      "renders; this is derived from the graphs themselves.")
    w("")
    w("**This file judges nothing.** Shot and speaker counts are mechanical. "
      "Whether a prompt follows the official guides — and what to do about it — "
      "is [`prompt_audit.md`](prompt_audit.md), which is hand-written and keyed "
      "to the scene names below.")
    w("")
    w("Scenes are ordered by how many graphs carry them, so the defaults that "
      "reach the most renders come first.")
    w("")
    w("| scene | graphs | words | shots | speakers | markers |")
    w("|---|---|---|---|---|---|")
    for name, text, graphs, f in scenes:
        mk = ", ".join(f"`{m}`" for m in f["markers"]) or "—"
        w(f"| [`{name}`](#{anchor(name)}) "
          f"| {len(graphs)} | {f['words']} | {f['shots']} | {f['speakers']} | {mk} |")
    w("")
    w("---")
    w("")
    for name, text, graphs, f in scenes:
        w(f"## {name}")
        w("")
        w(f"Carried by **{len(graphs)}** graph(s). Sections: "
          + (", ".join(f"`{s}`" for s in f["sections"]) or "none") + ".")
        w("")
        w("<details><summary>graphs</summary>")
        w("")
        for g in sorted(graphs):
            w(f"- `{g}`")
        w("")
        w("</details>")
        w("")
        w("```text")
        w(text.rstrip())
        w("```")
        w("")
    return "\n".join(o) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if the generated file is stale")
    args = ap.parse_args()

    names = constant_names()
    found = scan_graphs()
    if not found:
        # Same refusal as the wiki generator: emitting an empty catalogue would
        # read as "this repo renders no prompts" rather than "the scan broke".
        print("FAIL  no prompts found in any graph -- scan or graph shape changed")
        return 2

    scenes = []
    for text, graphs in found.items():
        # Constant name where one holds the text. Most ref2va prompts are
        # COMPOSED by `_ref_prompt()` at build time, so no constant ever holds
        # them and matching cannot name them -- derive from the graph family
        # instead, marked so nobody mistakes it for a constant that exists.
        name = names.get(text)
        if name is None:
            stems = sorted(g[:-4] if g.endswith("_api") else g for g in graphs)
            name = f"derived:{stems[0]}"
        scenes.append((name, text, graphs, facts(text)))
    scenes.sort(key=lambda s: (-len(s[2]), s[0]))

    out = render(scenes)
    if args.check:
        cur = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if cur != out:
            print("FAIL  docs/prompt_catalogue.md is stale -- rerun "
                  "bench/build_prompt_catalogue.py")
            return 1
        print("ok    docs/prompt_catalogue.md is current")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(out, encoding="utf-8")
    unnamed = sum(1 for s in scenes if s[0].startswith("derived:"))
    print(f"ok    wrote {OUT.relative_to(REPO)}")
    print(f"ok    {len(scenes)} distinct prompt(s) across "
          f"{len({g for s in scenes for g in s[2]})} graph(s)")
    if unnamed:
        print(f"      {unnamed} prompt(s) are composed at build time and named "
              "from their graph family, not from a constant")
    return 0


if __name__ == "__main__":
    sys.exit(main())
