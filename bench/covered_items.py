#!/usr/bin/env python3
"""The review items a cut the owner kept already answers, derived from the verdicts and the kept cut's own table.

    <python> bench/covered_items.py --items data/<stem>/review/items.json --verdicts data/<stem>/verdicts/verdicts.jsonl \\
        --tables data/<date>_<stem>_delivery_tables [--tables ...] --out data/<stem>/review/covered.json

**What it buys.** The owner, 2026-10-15, reviewing on the interactive page: items a cut they kept the day before
already answers still read as waiting. A cut is kept when the owner's last choice on its joined item is a keep
(`delivery_table.KEEP`). What it answers is read from the table the cut was built from (`<tables>/t_<its name>.json`,
the file `bench/shot_review_clips.py` writes), never from a list a session types:

- **covered by a cut you kept**: an item that is a piece the kept cut lays (its id or `name` is a row's render);
- **superseded**: a joined item of exactly the kept cut's frames built before it, and a joined item built before
  it whose frames hold the kept cut's frames (an older, longer join shows older pieces there).

A joined item built after a kept cut, and a piece the kept cut does not lay, are not touched: they are new to the
owner (2026-10-15: lead_fp_0751b landed after the 604-1009 cut was kept, and stays waiting). An item the owner has a
line on after the keep stays theirs too. A kept cut whose table is not found is said, and answers nothing.

`--out` is a JSON file the page reads: {"made", "kept": [...], "covered": [{"id", "why", "by"}]}. Nothing is typed.

**What it cannot see.** Whether the owner looked at a piece inside the kept cut on every frame it lays; that a kept
cut is good on every frame. It reads the keep as the owner's word on the whole cut, as the pages do (the lead's
ruling of 2026-10-14: a keep on a joined clip counts on each piece it lays).
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STAMP = re.compile(r"_joined_(\d{4}-\d{2}-\d{2})(T\d{4}Z)?")


def built_at(item_id: str) -> str | None:
    """When a joined item was built, from its name: its stamp, or a date alone (the joins of 2026-10-13 carry one) as that day's start."""
    m = STAMP.search(item_id)
    return (m.group(1) + (m.group(2) or "T0000Z")) if m else None


def covered(items: list[dict], lines: list[dict], tables: list[Path]) -> dict:
    import delivery_table
    by_row: dict[str, list[dict]] = {}
    for v in sorted(lines, key=lambda v: v.get("at") or ""):
        by_row.setdefault(str(v.get("row")), []).append(v)
    last_choice = lambda row: next((v for v in reversed(by_row.get(row, [])) if v.get("what") == "choice"), None)  # noqa: E731
    out, kept, said = [], [], []
    for it in items:
        if "_joined_" not in it["id"]:
            continue
        c = last_choice(it["id"])
        if not c or c.get("chosen") not in delivery_table.KEEP:
            continue
        name = it["id"].removesuffix("_stacked")
        table = next((t / f"t_{name}.json" for t in tables if (t / f"t_{name}.json").is_file()), None)
        if table is None:
            said.append(f"{it['id']} is kept and its table t_{name}.json is under no --tables: it answers nothing here")
            continue
        T = json.loads(table.read_text())
        first, last = int(it["first"]), int(it["first"]) + int(it["frames"]) - 1
        kept.append({"id": it["id"], "at": c["at"], "frames": [first, last], "table": str(table),
                     "lays": sorted({r["name"] for r in T["rows"] if r["first"] <= last and r["last"] >= first})})
    for k in kept:
        when, first, last = built_at(k["id"]), *k["frames"]
        for it in items:
            if it["id"] == k["id"] or any(x["id"] == it["id"] for x in out):
                continue
            later = [v for v in by_row.get(it["id"], []) if (v.get("at") or "") > k["at"]]
            if later:
                continue                                       # the owner has said something of it since the keep: theirs
            piece = it.get("name") or it["id"]
            if "_joined_" not in it["id"] and piece in k["lays"]:
                out.append({"id": it["id"], "why": f"laid in the cut you kept, {k['id']} ({k['at'][:16].replace('T', ' ')}Z)", "by": k["id"]})
                continue
            b = built_at(it["id"])
            if "_joined_" in it["id"] and b and when and b < when and it.get("first") is not None:
                f0, f1 = int(it["first"]), int(it["first"]) + int(it["frames"]) - 1
                if f0 <= first and f1 >= last:
                    same = (f0, f1) == (first, last)
                    out.append({"id": it["id"], "why": (f"an earlier build of the cut you kept, {k['id']}" if same else
                                                        f"older than the cut you kept on {first}-{last} ({k['id']}), which its frames hold"),
                                "by": k["id"]})
    return {"made": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "kept": kept, "covered": out, "said": said}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--items", required=True)
    p.add_argument("--verdicts", required=True)
    p.add_argument("--tables", action="append", default=[], required=True, help="a folder of the joined clips' tables; repeatable")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    items = json.loads(Path(a.items).read_text())["items"]
    lines = [json.loads(x) for x in Path(a.verdicts).read_text().splitlines() if x.strip()]
    got = covered(items, lines, [Path(t) for t in a.tables])
    for k in got["kept"]:
        print(f"kept: {k['id']} on {k['frames'][0]}-{k['frames'][1]} at {k['at']}, laying {', '.join(k['lays'])}")
    for x in got["covered"]:
        print(f"  {x['id']}: {x['why']}")
    for s in got["said"]:
        print(s)
    Path(a.out).write_text(json.dumps(got, indent=1) + "\n")
    print(f"wrote {a.out}: {len(got['covered'])} item(s) covered or superseded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
