#!/usr/bin/env python3
"""One plain page, a row a render of a video in frame order, with the owner's verdict lines under each: what was rendered, from what, and what the owner said of it.

    <python> bench/renders_page.py --renders data/<stem>/poc/renders.json --verdicts data/<stem>/verdicts/verdicts.jsonl \\
        [--items data/<stem>/review/items.json] [--clips Video/<stem>/review_clips] --out <folder on the share>

**What it buys.** The owner asked (2026-10-12) for one page on the share where every render of the video stands in
frame order with their own verdicts under it, so that "what has been rendered on this stretch and what did I say
of it" is one look and not a search through sessions' folders. Both facts exist: the renders inventory (`renders.json`,
read from the graph saved inside each render) and the verdicts record (`verdicts.jsonl`, the owner's clip lines
copied whole by `bench/page_answers_to_verdicts.py`). This page lays them side by side, moves no file, writes nothing
but its own folder, and is static HTML so it opens from the share as a file.

**What a row holds.** The render's source frames; its session and name and the day it was written; what its graph
asked for (whole subject or the wired parts, the part phrases, whether a motion video was wired, the stills it was
shown, which is how the subject is known: the still's own name, nothing inferred); whether the joined clip uses it;
links, relative to the page, to the render file and to its stacked clip where `--clips` holds one named
`<first>_<last>__<session>__<name>.mp4`; and every verdict line on it, a button and the typed words each with its
time, byte for byte the record's. A verdict whose render is in no inventory row is not dropped: it is listed in a
last section under its row id, with the review item's title and clip where `--items` names them.

**What it does not do.** It judges nothing, orders nothing but by frame, and reads no render's pixels. A render the
inventory lacks (a file with no readable graph; `renders.json` counts them under `not_listed`) is not on it.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

CSS = """:root{color-scheme:dark}body{font:17px/1.45 system-ui,sans-serif;margin:16px;background:#14161a;color:#e6e6e6}
table{border-collapse:collapse;width:100%}td,th{border-top:1px solid #333;padding:10px;vertical-align:top;text-align:left}
th{background:#1e2127;position:sticky;top:0}.f{font-family:ui-monospace,monospace;white-space:nowrap;font-size:18px}
.no{color:#8a8f99}a{color:#8ab4f8}.how{font-size:14px;color:#9aa3b2}.w{font-size:16px;color:#ddd;white-space:pre-line;margin-bottom:6px}
.b{display:inline-block;padding:1px 8px;border:1px solid #555;border-radius:4px;background:#23272e;margin-right:6px}
h2{margin-top:32px}"""


def relative(target: str, page_dir: str) -> str:
    """A link from the page's folder to a path, both written from the same root (the share's output folder)."""
    t, p = Path(target).parts, Path(page_dir).parts
    k = 0
    while k < min(len(t), len(p)) and t[k] == p[k]:
        k += 1
    return "/".join([".."] * (len(p) - k) + list(t[k:]))


def verdict_lines(verdicts: list[dict], render: str) -> list[dict]:
    """The record's lines on one render, by its file, in the record's own order."""
    return [v for v in verdicts if v.get("render") == render]


def one_verdict(v: dict) -> str:
    at = html.escape(str(v.get("at", ""))[:16])
    if v.get("what") == "choice":
        return f'<div class="w"><span class="b">{html.escape(str(v.get("chosen", "")))}</span><span class="how">{at}</span></div>'
    return f'<div class="w">{html.escape(str(v.get("words", "")))} <span class="how">{at}</span></div>'


def page(inv: dict, verdicts: list[dict], items: list[dict], clips: str | None, page_dir: str, root: Path | None = None) -> str:
    """`clips` and `page_dir` are paths from the share's output root; `root` is that root on this machine, for the is-it-there check."""
    renders = sorted(inv["renders"], key=lambda r: (int(r["first"]), int(r["last"]), str(r.get("written", ""))))
    rows, named = [], set()
    for r in renders:
        g = r.get("as_the_graph_says", {})
        asked = (f'{html.escape(str(g.get("replace", "")))}'
                 + (f' ({html.escape(str(g.get("part_phrases", "")))})' if g.get("replace") == "the wired parts" and g.get("part_phrases") else "")
                 + f'; motion video {"wired" if r.get("motion_wired") else "not wired"}'
                 + (f'; continued from {html.escape(str(r["continued_from"]))}' if r.get("continued_from") else ""))
        stills = ", ".join(html.escape(s) for s in (r.get("stills") or [])) or "none"
        links = [f'<a href="{html.escape(relative(r["file"], page_dir))}">the render</a>']
        if clips:
            c = f'{int(r["first"]):04d}_{int(r["last"]):04d}__{r["session"]}__{r["name"]}.mp4'
            there = root is not None and (root / clips / c).is_file()
            links.append(f'<a href="{html.escape(relative(clips.rstrip("/") + "/" + c, page_dir))}">stacked clip</a>' if there
                         else '<span class="no">no stacked clip on the share</span>')
        vs = verdict_lines(verdicts, render=r["file"])
        named.update(v.get("row") for v in vs)
        rows.append(f'<tr><td class="f">{int(r["first"])}-{int(r["last"])}<br><span class="how">{int(r["frames"])} f</span></td>'
                    f'<td><b>{html.escape(r["name"])}</b> <span class="how">{html.escape(r["session"])}, {html.escape(str(r.get("written", ""))[:10])}</span>'
                    f'<div class="how">{asked}</div><div class="how">stills shown: {stills}</div>'
                    f'<div class="how">{"in the joined clip" if r.get("in_the_joined_clip") else "not in the joined clip"}</div>'
                    f'<div>{" · ".join(links)}</div></td>'
                    f'<td>{"".join(one_verdict(v) for v in vs) or "<span class=no>none yet</span>"}</td></tr>')
    # verdicts on rows that name no inventory render: listed, never dropped
    rest = {}
    for v in verdicts:
        if v.get("render") and any(v["render"] == r["file"] for r in renders):
            continue
        rest.setdefault(str(v.get("row", "")), []).append(v)
    by_id = {i["id"]: i for i in items}
    tail = ""
    if rest:
        trs = []
        for row_id, vs in sorted(rest.items(), key=lambda kv: (int(kv[1][0].get("first") or 0), kv[0])):
            it = by_id.get(row_id, {})
            fr = f'{int(vs[0]["first"])}-{int(vs[0]["first"]) + int(vs[0]["frames"]) - 1}' if vs[0].get("first") is not None and vs[0].get("frames") else ""
            where = html.escape(str(it.get("path") or vs[0].get("render") or ""))
            trs.append(f'<tr><td class="f">{fr}</td><td><b>{html.escape(row_id)}</b><div class="how">{html.escape(str(it.get("title", vs[0].get("title", ""))))}</div>'
                       f'<div class="how">{where}</div></td><td>{"".join(one_verdict(v) for v in vs)}</td></tr>')
        tail = ('<h2>Your lines on clips that name no render of the inventory</h2><p class="how">Review items whose clip was not read '
                'into the inventory (a stacked clip with no graph of its own, or a render whose graph could not be read). Nothing is dropped.</p>'
                f'<table><tr><th>frames</th><th>row</th><th>your lines</th></tr>{"".join(trs)}</table>')
    head = (f'<h1>Renders of {html.escape(str(inv.get("video", "")))}, in frame order</h1>'
            f'<p>{len(renders)} renders read from the graph saved in each file ({html.escape(str(inv.get("read_from", "")))}; '
            f'inventory made {html.escape(str(inv.get("made_at", ""))[:16])} by {html.escape(str(inv.get("made_by", "")))}). '
            f'{len(verdicts)} of your lines from the verdicts record. A still\'s name says who was rendered; nothing here infers it. '
            f'No file was moved to make this page.</p>')
    return (f'<!doctype html><meta charset="utf-8"><title>renders of {html.escape(str(inv.get("video", "")))}</title><style>{CSS}</style>'
            f'{head}<table><tr><th>frames</th><th>the render</th><th>your lines on it</th></tr>{"".join(rows)}</table>{tail}')


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--renders", required=True, help="the renders inventory (data/<stem>/poc/renders.json)")
    ap.add_argument("--verdicts", required=True, help="the owner's lines on renders (bench/page_answers_to_verdicts.py's record)")
    ap.add_argument("--items", help="the review page's items file, for the title and clip of a row that names no inventory render")
    ap.add_argument("--clips", help="the folder of stacked clips named <first>_<last>__<session>__<name>.mp4, as a path from the share's output root")
    ap.add_argument("--root", default="/mnt/hub/ai/img/output", help="the share's output root, which the inventory's file paths and --clips are written from")
    ap.add_argument("--out", required=True, help="the page's folder, under that root")
    a = ap.parse_args()
    inv = json.load(open(a.renders))
    verdicts = [json.loads(t) for t in Path(a.verdicts).read_text().splitlines() if t.strip()]
    items = json.load(open(a.items)).get("items", []) if a.items else []
    out = Path(a.out).resolve()
    root = Path(a.root).resolve()
    try:
        page_dir = str(out.relative_to(root))
    except ValueError:
        raise SystemExit(f"--out {out} is not under --root {root}: the page's links are relative to that root")
    out.mkdir(parents=True, exist_ok=True)
    text = page(inv, verdicts, items, a.clips, page_dir, root)
    (out / "index.html").write_text(text)
    print(f"written: {out / 'index.html'}; {len(inv['renders'])} render(s); {len(verdicts)} verdict line(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
