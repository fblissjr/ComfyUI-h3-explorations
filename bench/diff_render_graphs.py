#!/usr/bin/env python3
"""What a SAVED render actually ran, read from its own embedded graph.

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). Needs no server
and no GPU; opens files read-only and allocates nothing.

**Not a check.** It asserts nothing. It answers one question that nothing else
here answers: *what did this finished render actually execute* -- from the
graph ComfyUI embedded in the output, rather than from the workflow JSON in the
repo, the UI's current state, or anybody's memory of it.

Those four disagree routinely, and two instances on 2026-08-29 are why this
exists:

  - A graph was read by eye and reported as driving the sampler from
    `ManualSigmas`, because that node was present and its literal values
    matched what was expected. The live link ran from the PDD node's SIGMAS
    output through a visualizer; `ManualSigmas` fed a second visualizer into a
    preview and touched nothing. Reading node *presence* instead of node
    *reachability* inverted the conclusion, and every downstream number -- step
    count, head coverage, which steps a percent-band node covered -- was
    computed against the wrong schedule.
  - Two renders were recalled as differing in prompt, length and resolution.
    Their embedded graphs differ by one node.

`bench/record_render_substrate.py` could not have caught either: it reads
`/history` from a LIVE server and records the conditions a render ran under, so
it says nothing about a file on disk once the server has moved on. That is the
gap. `bench/preflight_graph.py` grades a graph before a render; this reads one
back afterwards.

**Provenance, not presence.** The `sampler-input provenance` block is the part
that answers the failure above: it resolves each sampler input to the node that
really feeds it, following single-input passthroughs so a visualizer or a
reroute cannot hide the origin.

Read that block, not the node list. Reachability marking is deliberately weak
and says less than it looks like it does: a preview node is a terminal too, so
a node feeding only a preview still counts as reachable. That is exactly the
case that misled a reader once already -- `ManualSigmas` wired into a preview
is reachable and irrelevant at the same time. Only the provenance block
distinguishes them.

Usage:

    python bench/diff_render_graphs.py --dump  RENDER.png
    python bench/diff_render_graphs.py --diff  A.png B.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Inputs whose real source is worth naming outright. These are the ones that
#: decide the trajectory, and the ones a visualizer or a passthrough node can
#: silently sit in front of.
TRACED = ("sigmas", "sampler", "guider", "noise", "latent_image", "model")


def load_graph(path: Path) -> dict:
    """The API-format graph ComfyUI embeds in its own output."""
    from PIL import Image

    with Image.open(path) as im:
        raw = im.info.get("prompt")
    if not raw:
        raise SystemExit(
            f"{path.name} carries no embedded `prompt` graph. ComfyUI writes it "
            f"only when the node that saved the file had metadata enabled.")
    return json.loads(raw)


def reachable(graph: dict) -> set[str]:
    """Node ids the output nodes can reach, walking links backwards.

    Presence is not execution: a node wired only into a preview branch runs,
    but does not touch the render. Callers want the render.
    """
    consumed = {b[0] for v in graph.values()
                for b in v.get("inputs", {}).values()
                if isinstance(b, list) and b}
    roots = [k for k in graph if k not in consumed]
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        nid = stack.pop()
        if nid in seen or nid not in graph:
            continue
        seen.add(nid)
        for b in graph[nid].get("inputs", {}).values():
            if isinstance(b, list) and b:
                stack.append(b[0])
    return seen


def resolve(graph: dict, nid: str, field: str) -> str:
    """`field`'s source, following single-input passthroughs to the origin."""
    link = graph.get(nid, {}).get("inputs", {}).get(field)
    if not isinstance(link, list) or not link:
        return "(literal)"
    chain = []
    src, slot = link[0], link[1] if len(link) > 1 else 0
    for _ in range(16):
        cls = graph.get(src, {}).get("class_type", "?")
        chain.append(f"{src}:{cls}[{slot}]")
        ups = [b for b in graph.get(src, {}).get("inputs", {}).values()
               if isinstance(b, list) and b]
        # only follow through a node with exactly one upstream link, which is
        # what a visualizer or passthrough looks like
        if len(ups) != 1:
            break
        src, slot = ups[0][0], ups[0][1] if len(ups[0]) > 1 else 0
    return " <- ".join(chain)


def scalars(node: dict) -> dict:
    return {a: b for a, b in node.get("inputs", {}).items()
            if not isinstance(b, list)}


def cmd_dump(path: Path) -> int:
    graph = load_graph(path)
    live = reachable(graph)
    print(f"# {path.name}: {len(graph)} node(s), {len(live)} reachable from the output\n")
    for nid in sorted(graph, key=lambda k: int(k) if k.isdigit() else 0):
        node = graph[nid]
        mark = " " if nid in live else "~"
        body = json.dumps(scalars(node))
        if len(body) > 150:
            body = body[:150] + "...}"
        print(f"{mark}{nid:>5} {node.get('class_type',''):<32} {body}")
    print("\n(`~` = unreachable from every terminal. NOTE: a preview is a terminal,\n so an unmarked node may still be preview-only -- read the provenance below.)\n")
    print("sampler-input provenance:")
    for nid, node in graph.items():
        if node.get("class_type", "").startswith(("SamplerCustom", "KSampler")):
            for field in TRACED:
                if field in node.get("inputs", {}):
                    print(f"  {nid}:{node['class_type']}.{field} <- {resolve(graph, nid, field)}")
    return 0


def cmd_diff(a: Path, b: Path) -> int:
    ga, gb = load_graph(a), load_graph(b)
    la, lb = reachable(ga), reachable(gb)
    print(f"# {a.name}  vs  {b.name}\n")
    only_a = sorted(la - lb, key=lambda k: int(k) if k.isdigit() else 0)
    only_b = sorted(lb - la, key=lambda k: int(k) if k.isdigit() else 0)
    for label, ids, g in (("only in A", only_a, ga), ("only in B", only_b, gb)):
        for nid in ids:
            print(f"  {label}: {nid} {g[nid].get('class_type','')}")
    diffs = 0
    for nid in sorted(la & lb, key=lambda k: int(k) if k.isdigit() else 0):
        ia, ib = scalars(ga[nid]), scalars(gb[nid])
        for field in sorted(set(ia) | set(ib)):
            if ia.get(field) != ib.get(field):
                diffs += 1
                va, vb = ia.get(field), ib.get(field)
                for name, val in (("A", va), ("B", vb)):
                    text = str(val)
                    if len(text) > 90:
                        text = f"{text[:90]}... (len {len(str(val))})"
                    print(f"  {nid}.{field:<22} {name}= {text}")
    if not (only_a or only_b or diffs):
        print("  reachable graphs are identical")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="What a saved render actually ran, from its embedded graph.")
    ap.add_argument("--dump", metavar="RENDER")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"))
    args = ap.parse_args(argv)
    if args.dump:
        return cmd_dump(Path(args.dump))
    if args.diff:
        return cmd_diff(Path(args.diff[0]), Path(args.diff[1]))
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
