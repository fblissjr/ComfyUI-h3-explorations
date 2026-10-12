#!/usr/bin/env python3
"""The owner's verdicts and notes on the review page's clips, copied from the page's raw file into the video's record, one line each, with the render they are about.

    <python> bench/page_answers_to_verdicts.py take --answers data/<stem>/poc/answers.jsonl --items data/<stem>/review/items.json \\
        [--items data/<stem>/review/items_removed_<date>.json ...] --out data/<stem>/verdicts/verdicts.jsonl [--dry-run]
    <python> bench/page_answers_to_verdicts.py show --verdicts data/<stem>/verdicts/verdicts.jsonl [--render NAME]

**What it buys.** On 2026-10-11 the owner wrote 147 lines of typed words on the review page in one evening,
about fifty of them on single clips ("this is the best one", "do not use this one", "was that intentional?",
"is it the same as the other two?"). They were saved, one raw line a click or a box, in a file the page
owns, keyed by the page's own item ids; no record held them and nothing read them back beside the render
they were about. The owner's rule for that file (Open List row g01, 2026-10-11): it is raw input, and a
session structures it. This is that structuring for the clip rows, the way `page_answers_to_cast.py` is it
for the who-is-who rows: a copy a line into a file of the record, with the render's own path on it, so a
reader of a render finds what the owner said of it without the page.

**How a line is read.** Only lines whose `who` is `owner` and whose `row` is an item of the review list given
(`--items`, repeatable, so that items taken off the page are still known) are taken. A `choice` line keeps
its `chosen`; a words line keeps its `words`; `frame` is kept when the line was saved at a frame. Nothing is
reworded, merged or judged: a later line that contradicts an earlier one is a later line, and `show` prints
them in order. A line whose row is not an item (the Open List's rows, the who-is-who rows) is counted and
left to its own tool.

**What is written** (`VERDICTS_SCHEMA`): `at` (the page's own time), `row` (the item's id), `title`, `render`
(the render file the item names, relative to the share's output folder, or `null` for a view that is not a
render), `first` and `frames` (the item's source frames), `frame`, `what`, `chosen`, `words`, `source` (the
answers file's name and line). A line already in the record, by source and line, is not written again.

What it does not do: read the words, decide which verdict stands, or know a render the review list does not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VERDICTS_SCHEMA = "h3_verdicts/1"
FIELDS = ("schema", "at", "row", "title", "render", "first", "frames", "frame", "what", "chosen", "words", "source")


def read_items(paths: list[str]) -> dict[str, dict]:
    out = {}
    for p in paths:
        d = json.load(open(p))
        for it in (d["items"] if isinstance(d, dict) else d):
            out.setdefault(it["id"], it)
    return out


def read_lines(path) -> list[tuple[int, dict]]:
    out = []
    for n, text in enumerate(Path(path).read_text().splitlines(), 1):
        if text.strip():
            out.append((n, json.loads(text)))
    return out


def read_verdicts(path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    for n, text in enumerate(p.read_text().splitlines(), 1):
        d = json.loads(text)
        if d.get("schema") != VERDICTS_SCHEMA:
            raise SystemExit(f"{path} line {n}: schema {d.get('schema')!r}, not {VERDICTS_SCHEMA!r}")
        out.append(d)
    return out


def verdict(n: int, d: dict, item: dict, source_name: str) -> dict | None:
    what = str(d.get("what", ""))
    if what == "choice" and not str(d.get("chosen", "")).strip():
        return None
    if what.startswith("words") and not str(d.get("words", "")).strip():
        return None
    if what != "choice" and not what.startswith("words"):
        return None
    return {"schema": VERDICTS_SCHEMA, "at": str(d.get("at", "")), "row": item["id"], "title": str(item.get("title", "")),
            "render": item.get("render"), "first": item.get("first"), "frames": item.get("frames"),
            "frame": d.get("frame"), "what": what, "chosen": d.get("chosen") if what == "choice" else None,
            "words": d.get("words") if what.startswith("words") else None, "source": f"{source_name}:{n}"}


def take(a) -> int:
    items = read_items(a.items)
    have = {v["source"] for v in read_verdicts(a.out)}
    name = Path(a.answers).name
    new, already, not_items, others = [], 0, 0, 0
    for n, d in read_lines(a.answers):
        if d.get("who") != "owner":
            others += 1
            continue
        row = str(d.get("row", ""))
        if row not in items:
            not_items += 1
            continue
        v = verdict(n, d, items[row], name)
        if v is None:
            continue
        if v["source"] in have:
            already += 1
            continue
        new.append(v)
    for v in new:
        where = f"frame {v['frame']}" if v.get("frame") not in (None, "") else "the clip"
        said = v["chosen"] if v["what"] == "choice" else (v["words"] or "").strip().replace("\n", " / ")
        print(f"{v['at'][:16]}  {v['row']}  ({v['render'] or 'not a render'})  {where}: {said}")
    if not a.dry_run and new:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a") as f:
            for v in new:
                f.write(json.dumps({k: v.get(k) for k in FIELDS}, sort_keys=True) + "\n")
    print(f"\n{'would write' if a.dry_run else 'written'} {len(new)} line(s); already in the record {already}; "
          f"the owner's lines on rows that are not clips (the list's and the who-is-who rows, theirs to other tools) {not_items}; "
          f"not the owner's {others}; record {a.out}")
    return 0


def show(a) -> int:
    rows = read_verdicts(a.verdicts)
    if a.render:
        rows = [v for v in rows if v.get("render") and Path(v["render"]).stem.startswith(a.render) or v.get("row") == a.render]
    rows.sort(key=lambda v: (str(v.get("render") or ""), v["at"]))
    last = None
    for v in rows:
        key = v.get("render") or v["row"]
        if key != last:
            print(f"\n{key}  ({v['row']}; source frames {v.get('first')} + {v.get('frames')})")
            last = key
        where = f"frame {v['frame']}" if v.get("frame") not in (None, "") else "clip"
        said = v["chosen"] if v["what"] == "choice" else (v["words"] or "").strip().replace("\n", " / ")
        print(f"  {v['at'][:16]}  {where:>11}: {said}")
    if not rows:
        print("no line" + (f" for {a.render}" if a.render else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("take", help="copy the owner's clip lines from the page's file into the record")
    t.add_argument("--answers", required=True)
    t.add_argument("--items", action="append", required=True, help="the review list; repeat for items taken off the page")
    t.add_argument("--out", required=True)
    t.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("show", help="the owner's lines, a render at a time, in order")
    s.add_argument("--verdicts", required=True)
    s.add_argument("--render", help="a render's name (the file's stem, or its prefix) or an item id")
    a = ap.parse_args()
    return take(a) if a.cmd == "take" else show(a)


if __name__ == "__main__":
    sys.exit(main())
