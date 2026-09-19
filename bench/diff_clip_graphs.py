#!/usr/bin/env python3
"""What actually differs between two rendered clips? Read from the graphs embedded in them.

    python bench/diff_clip_graphs.py <clip_a.mp4> <clip_b.mp4> [--expect NODE.FIELD ...]

Every clip this pack renders carries the API graph it was rendered from in its
`comment` tag. A pair is only a comparison of ONE thing if the two graphs
differ in that one thing, and until 2026-09-19 nothing checked: a reorder
panel pair and a "sage against kitchen" pair were both built on a 2026-09-15
clip that a batch record called the default chain and whose own metadata says
sage `auto`, so one verdict on token order and the owner's one audio
preference between chains were about something else. This prints every input
that differs, by node class, ignoring output filename prefixes, and exits
non-zero when `--expect` is given and the differences are not exactly the
expected fields. Run it before building any stack.

Limits: it compares what the graph SAYS, not what the code did with it (a mode
named `auto` changed meaning on 2026-09-19); node ids may differ between
graphs, so nodes are matched by class and order of appearance. No GPU.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import orjson

IGNORED = {("VHS_VideoCombine", "filename_prefix"), ("SaveImage", "filename_prefix")}


def graph_of(clip: str) -> dict:
    raw = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment",
                          "-of", "default=nw=1:nk=1", clip], capture_output=True, check=True).stdout
    data = orjson.loads(raw)
    prompt = data["prompt"]
    return prompt if isinstance(prompt, dict) else orjson.loads(prompt)


def flat(graph: dict) -> dict:
    """{(class, nth of that class, field): value} for literal inputs; links become the class they point at."""
    seen: dict[str, int] = {}
    out = {}
    for nid in sorted(graph, key=lambda k: (len(k), k)):
        node = graph[nid]
        cls = node["class_type"]
        n = seen.get(cls, 0)
        seen[cls] = n + 1
        for field, value in node.get("inputs", {}).items():
            if (cls, field) in IGNORED:
                continue
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
                value = f"<- {graph[str(value[0])]['class_type']}[{value[1]}]"
            out[(cls, n, field)] = value
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip_a")
    ap.add_argument("clip_b")
    ap.add_argument("--expect", nargs="*", default=None, metavar="CLASS.FIELD",
                    help="the only fields allowed to differ; anything else, or a missing one, fails")
    args = ap.parse_args()
    a, b = flat(graph_of(args.clip_a)), flat(graph_of(args.clip_b))
    diffs = []
    for key in sorted(set(a) | set(b), key=str):
        va, vb = a.get(key, "<absent>"), b.get(key, "<absent>")
        if va != vb:
            diffs.append((key, va, vb))
    for (cls, n, field), va, vb in diffs:
        show = lambda v: (repr(v)[:70] + "...") if len(repr(v)) > 73 else repr(v)   # noqa: E731
        print(f"  {cls}[{n}].{field}: {show(va)}  ->  {show(vb)}")
    if not diffs:
        print("  the two graphs are identical apart from output filenames")
    if args.expect is not None:
        got = {f"{cls}.{field}" for (cls, _n, field), _a, _b in diffs}
        want = set(args.expect)
        if got != want:
            print(f"FAIL: expected exactly {sorted(want)} to differ, found {sorted(got)}")
            return 1
        print("ok: only the expected fields differ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
