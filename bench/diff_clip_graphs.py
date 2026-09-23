#!/usr/bin/env python3
"""What actually differs between two rendered clips? Read from the graphs they were rendered from.

    python bench/diff_clip_graphs.py <clip_a.mp4> <clip_b.mp4> [--expect NODE.FIELD ...]

**Where the graph lives.** Every clip this pack renders has a first-frame PNG
beside it carrying the API graph in its `prompt` text chunk: VHS's
`VideoCombine` writes `<prefix>_NNNNN.png` next to `<prefix>_NNNNN.mp4` and
`<prefix>_NNNNN-audio.mp4`, and `loop_output.write_metadata_png` does the same
for the song node. `graph_of` reads that PNG first. The pairing is by name,
and it is sound because one save writes both files under one counter; a clip
copied away from its PNG falls through to the container, then refuses, and
never guesses. The video container itself carries no graph since 2026-09-23,
when the owner's VHS fork and `loop_output` stopped writing metadata into
video files. Older clips may carry it in the container instead, either as one
`comment` tag holding `{"prompt": ...}` (seen on files on disk) or as a bare
`prompt` tag (upstream VHS's layout at its 4d907be), so both are read as a
fallback. You can also pass a PNG directly.

A pair is only a comparison of ONE thing if the two graphs
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
from pathlib import Path

import orjson

IGNORED = {("VHS_VideoCombine", "filename_prefix"), ("SaveImage", "filename_prefix")}


def sidecar_png(clip) -> Path:
    """The first-frame PNG saved with a clip: `x_00001.png` for `x_00001.mp4` or `x_00001-audio.mp4`."""
    path = Path(clip)
    stem = path.stem[:-len("-audio")] if path.stem.endswith("-audio") else path.stem
    return path.with_name(stem + ".png")


def _as_graph(prompt) -> dict:
    return prompt if isinstance(prompt, dict) else orjson.loads(prompt)


def graph_of(clip: str) -> dict:
    """The API graph `clip` was rendered from: its sidecar PNG, else its container tags. Raises when neither has one."""
    path = Path(clip)
    png = path if path.suffix.lower() == ".png" else sidecar_png(path)
    if png.is_file():
        from PIL import Image
        with Image.open(png) as im:
            raw = im.info.get("prompt")
        if raw:
            return _as_graph(raw)
    if path.suffix.lower() != ".png":
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json",
                              str(path)], capture_output=True, check=True).stdout
        tags = orjson.loads(out).get("format", {}).get("tags", {})
        if "prompt" in tags:
            return _as_graph(tags["prompt"])
        if "comment" in tags:
            return _as_graph(orjson.loads(tags["comment"])["prompt"])
    raise ValueError(f"{path.name}: no graph. {png.name} is missing or carries no `prompt` chunk, "
                     f"and the container carries no `prompt` or `comment` tag")


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
