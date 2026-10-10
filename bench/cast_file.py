#!/usr/bin/env python3
"""The cast file: what a person answered about who is on a frame, kept so it is typed once.

The least thing that stores a typed answer. A person looks at one tile of a preview's shot table, on which
everybody on that frame is outlined and numbered, and says for a named subject "person 2" or "nobody". This
file keeps that as one line: the subject, the frame of the source copy, and either the person's BOX on that
frame or "not on this frame". Nothing here compares anybody with anybody; there is no likeness in it and it
decides nothing.

Four rules, each a fault of 2026-10-10 that this file exists to not repeat:

- AN ANSWER IS OF A FRAME, NOT OF A SHOT. A tile shows one frame. "Nobody" on it is stored as `not on this
  frame`; a subject who walks out half way through a shot is not "not in the shot". Whoever needs the stronger
  statement makes it knowingly (`seeds`' `not_on_frame` has no default).
- THE NUMBER BECOMES A BOX WHEN THE ANSWER IS GIVEN, from the table the tile was drawn from, and the line names
  that table. No tile number is stored, so a tile made again cannot renumber an old answer.
- THE TILE MUST SHOW EVERYBODY. `answer` refuses a table whose shot says another tracker's people were left out
  on the shown frame: those numbers are not everybody. A preview with one subject and no `others` is such a
  pass, and the job builder already makes it.
- ONE WRITER. `append` writes one whole line under a lock and nothing is ever rewritten; taking an answer back
  is another line. The latest line for a subject and a frame is the answer.

    <python> bench/cast_file.py answer --cast CAST --table SHOTS.json --first-frame N --shot K --subject lead --person 2 --by NAME
    <python> bench/cast_file.py answer --cast CAST --table SHOTS.json --first-frame N --shot K --subject lead --nobody --by NAME
    <python> bench/cast_file.py withdraw --cast CAST --subject lead --frame F --by NAME
    <python> bench/cast_file.py seeds --cast CAST --subject lead --table LOAD_SHOTS.json --first-frame N --not-on-frame leave|empty
    <python> bench/cast_file.py show --cast CAST

`--first-frame` is the frame of the source copy that the table's frame 0 is on. `seeds` prints the text the
Subject Track's `seeds` input takes for one load (`subject_track.py::parse_seeds`), one line a shot that has an
answer inside it, and says what it did with every answer of that subject: used, unused and why, or outside the
load. An answer outside the load is never moved to a frame inside it. A box is scaled only to a picture of the
same shape; another shape is refused, since the loader crops before it scales.

A cast file lives under the untracked `data/`, one a source video. No model, no card.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "h3_cast/1"
HERE, NOT_ON_FRAME, WITHDRAWN = "here", "not on this frame", "withdrawn"
ANSWERS = (HERE, NOT_ON_FRAME, WITHDRAWN)
#: What `seeds` may do with a `not on this frame` answer. No default: one frame does not say a subject is out of
#: a shot, and the caller says in words which reading it takes.
LEAVE, EMPTY = "leave", "empty"
NOT_ON_FRAME_MEANS = {LEAVE: "the shot is left unseeded, for the tracker to judge",
                      EMPTY: "the shot is emptied from that one frame (`frame k: none`)"}
#: A subject's label, as the job builder's (`workflows/masked_job_graph.py::NAME`): it goes into file names.
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]*\Z")


def entry(subject: str, frame: int, answer: str, *, by: str, box=None, size=None, table: str = "", at: str | None = None) -> dict:
    """One line of the file, checked. `frame` is a frame of the source copy; `box` is (left, top, width, height) at `size`."""
    if not NAME.match(str(subject)):
        raise ValueError(f"cast: `{subject}` is not a subject's label (letters, digits and _)")
    if answer not in ANSWERS:
        raise ValueError(f"cast: an answer is one of {ANSWERS}, not `{answer}`")
    if int(frame) < 0:
        raise ValueError(f"cast: frame {frame}")
    if not str(by).strip():
        raise ValueError("cast: an answer says who gave it (`by`)")
    if answer == HERE:
        if box is None or size is None or len(box) != 4 or len(size) != 2:
            raise ValueError("cast: `here` needs the person's box (left, top, width, height) and the picture size it was read at")
        x, y, w, h = (float(v) for v in box)
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > size[0] or y + h > size[1]:
            raise ValueError(f"cast: box {list(box)} is not inside a {size[0]}x{size[1]} picture")
        if not str(table).strip():
            raise ValueError("cast: `here` names the table file the tile was drawn from")
    elif box is not None:
        raise ValueError(f"cast: `{answer}` has no box")
    return {"schema": SCHEMA, "subject": str(subject), "frame": int(frame), "answer": answer,
            "box": None if box is None else [float(v) for v in box], "size": None if size is None else [int(v) for v in size],
            "table": str(table), "by": str(by).strip(),
            "at": at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def append(path, line: dict) -> None:
    """Add one line. The only way anything is written: one whole line, in one write, under a lock.

    A file whose last line was cut short (a writer that died) is refused, so a new line is never glued to it."""
    data = (json.dumps(entry(**{k: line[k] for k in ("subject", "frame", "answer", "by", "box", "size", "table", "at")}),
                       sort_keys=True) + "\n").encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        size = os.fstat(fd).st_size
        if size and os.pread(fd, 1, size - 1) != b"\n":
            raise ValueError(f"cast: {path} does not end with a whole line; mend it by hand before anything is added")
        if os.write(fd, data) != len(data):
            raise OSError(f"cast: a short write to {path}")
        os.fsync(fd)
    finally:
        os.close(fd)             # closing lets the lock go


def read(path) -> list[dict]:
    """Every line, oldest first. A line that cannot be read is an error that names it; none is skipped."""
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for number, text in enumerate(path.read_text().splitlines(), 1):
        try:
            line = json.loads(text)
            if line.get("schema") != SCHEMA:
                raise ValueError(f"schema {line.get('schema')!r}")
            out.append(entry(**{k: line.get(k) for k in ("subject", "frame", "answer", "by", "box", "size", "table", "at")}))
        except (ValueError, TypeError, AttributeError) as e:
            raise ValueError(f"cast: {path} line {number} cannot be read: {e}") from None
    return out


def current(lines: list[dict]) -> dict[tuple[str, int], dict]:
    """The answer that stands for each (subject, frame): the latest line, and none where that line takes it back."""
    latest = {(line["subject"], line["frame"]): line for line in lines}
    return {key: line for key, line in latest.items() if line["answer"] != WITHDRAWN}


def from_table(table: dict, table_name: str, first: int, shot: int, subject: str, person: int | None, by: str) -> dict:
    """A person's answer about one tile, as a line: the tile's frame and, for a number, that person's box.

    `person` is the number on the tile, or None for nobody. `first` is the source-copy frame of the table's frame 0."""
    found = next((s for s in table.get("shots", []) if int(s["shot"]) == int(shot)), None)
    if found is None:
        raise ValueError(f"cast: {table_name} has no shot {shot}")
    left = (found.get("others_left_out") or {}).get("on_shown_frame", 0)
    if left:
        raise ValueError(f"cast: {table_name}, shot {shot}: {left} detection(s) were left out of this tile as another tracker's "
                         "people, so its numbers are not everybody on the frame. Answer from a preview with one subject and no `others`.")
    frame = int(first) + int(found["shown_frame"])
    if person is None:
        return entry(subject, frame, NOT_ON_FRAME, by=by, table=table_name)
    one = next((p for p in found.get("people", []) if int(p["person"]) == int(person)), None)
    if one is None:
        raise ValueError(f"cast: {table_name}, shot {shot}: the tile has no person {person}; it shows {len(found.get('people', []))}")
    return entry(subject, frame, HERE, by=by, box=one["box"], size=table["size"], table=table_name)


def _number(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.2f}"


def seeds(lines: list[dict], subject: str, first: int, size, shots: list[tuple[int, int]], not_on_frame: str) -> dict:
    """The Subject Track's `seeds` text for one load of one subject, and what became of every answer.

    `first` is the source-copy frame of the load's frame 0, `size` its picture (width, height), `shots` its
    (first frame, one past the last) per shot in the load's own frames. `not_on_frame` is `LEAVE` or `EMPTY`.
    In a shot with several answers a box is used before a `not on this frame`, and among boxes the latest given.
    That is a tie-break and not a judgement: each box is true of its own frame, a shot takes one seed, and the newer
    answer did not correct the older. What the caller chose for `not_on_frame` is in the result, in words."""
    if not_on_frame not in NOT_ON_FRAME_MEANS:
        raise ValueError("cast: say what a `not on this frame` answer does to its shot: "
                         + "; ".join(f"`{k}`: {v}" for k, v in NOT_ON_FRAME_MEANS.items()))
    last = shots[-1][1] if shots else 0
    mine = [line for (who, _), line in current(lines).items() if who == subject]
    inside = [line for line in mine if 0 <= line["frame"] - first < last]
    outside = sorted(line["frame"] for line in mine if not 0 <= line["frame"] - first < last)
    for line in inside:
        if line["answer"] == HERE and line["size"][0] * size[1] != line["size"][1] * size[0]:
            raise ValueError(f"cast: {subject}'s answer on frame {line['frame']} was read on a {line['size'][0]}x{line['size'][1]} picture "
                             f"and this load is {size[0]}x{size[1]}: another shape. A box is not carried across a crop.")
    order = {id(line): n for n, line in enumerate(lines)}
    rows, text, unused = [], [], []
    for number, (start, end) in enumerate(shots, 1):
        here = sorted((line for line in inside if start <= line["frame"] - first < end),
                      key=lambda line: (line["answer"] == HERE, line["at"], order.get(id(line), 0)))
        row = {"shot": number, "frames": [int(start), int(end) - 1], "seed": None, "from_frame": None, "why": "no answer inside this shot"}
        if here:
            used = here[-1]
            k = used["frame"] - first
            if used["answer"] == HERE:
                factor = size[0] / used["size"][0]
                row["seed"] = f"frame {k}: " + ", ".join(_number(v * factor) for v in used["box"])
                row["why"] = f"{used['by']}'s answer of {used['at']}"
            elif not_on_frame == EMPTY:
                row["seed"] = f"frame {k}: none"
                row["why"] = f"{used['by']} said not on frame {used['frame']}; the caller asked that this empty the shot"
            else:
                row["why"] = f"{used['by']} said not on frame {used['frame']}, which does not say not in the shot; left unseeded"
            row["from_frame"] = used["frame"]
            for other in here[:-1]:
                unused.append({"frame": other["frame"], "answer": other["answer"],
                               "why": f"shot {number} of this load has another answer, on frame {used['frame']}, and a shot takes "
                                      "one seed; this one is not wrong, only not used"})
            if row["seed"]:
                text.append(row["seed"])
        rows.append(row)
    return {"subject": subject, "first_source_frame": int(first), "size": [int(size[0]), int(size[1])], "text": "\n".join(text),
            "not_on_frame": not_on_frame, "not_on_frame_means": NOT_ON_FRAME_MEANS[not_on_frame], "shots": rows, "unused": unused, "outside_this_load": outside}


def _table(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> int:
    p = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("answer", help="store what a person said about one tile")
    a.add_argument("--table", required=True)
    a.add_argument("--first-frame", type=int, required=True)
    a.add_argument("--shot", type=int, required=True)
    who = a.add_mutually_exclusive_group(required=True)
    who.add_argument("--person", type=int)
    who.add_argument("--nobody", action="store_true")
    w = sub.add_parser("withdraw", help="take an answer back")
    w.add_argument("--frame", type=int, required=True)
    s = sub.add_parser("seeds", help="the Subject Track's seeds text for one load")
    s.add_argument("--table", required=True, help="a shot table of the load: its shots and its picture size are read")
    s.add_argument("--first-frame", type=int, required=True)
    s.add_argument("--not-on-frame", required=True, choices=sorted(NOT_ON_FRAME_MEANS))
    show = sub.add_parser("show", help="the answers that stand")
    for one in (a, w, s, show):
        one.add_argument("--cast", required=True)
    for one in (a, w, s):
        one.add_argument("--subject", required=True)
    for one in (a, w):
        one.add_argument("--by", required=True)
    args = p.parse_args()
    if args.cmd == "answer":
        line = from_table(_table(args.table), args.table, args.first_frame, args.shot, args.subject,
                          None if args.nobody else args.person, args.by)
        append(args.cast, line)
        print(json.dumps(line, sort_keys=True))
    elif args.cmd == "withdraw":
        if (args.subject, args.frame) not in current(read(args.cast)):
            raise SystemExit(f"cast: no answer stands for {args.subject} on frame {args.frame}")
        append(args.cast, entry(args.subject, args.frame, WITHDRAWN, by=args.by))
    elif args.cmd == "seeds":
        table = _table(args.table)
        out = seeds(read(args.cast), args.subject, args.first_frame, table["size"],
                    [(int(x["first_frame"]), int(x["last_frame"]) + 1) for x in table["shots"]], args.not_on_frame)
        print(out["text"])
        print(f"# a `not on this frame` answer: {out['not_on_frame_means']} (--not-on-frame {out['not_on_frame']})", file=sys.stderr)
        for row in out["shots"]:
            print(f"# shot {row['shot']} (frames {row['frames'][0]}-{row['frames'][1]}): {row['seed'] or 'no seed'}; {row['why']}", file=sys.stderr)
        for one in out["unused"]:
            print(f"# UNUSED: the answer on frame {one['frame']} ({one['answer']}): {one['why']}", file=sys.stderr)
        if out["outside_this_load"]:
            print(f"# outside this load, not moved: {args.subject}'s answers on frames {out['outside_this_load']}", file=sys.stderr)
    else:
        for (subject, frame), line in sorted(current(read(args.cast)).items()):
            print(f"{subject:12s} frame {frame:6d}  {line['answer']:18s} {'' if line['box'] is None else line['box']}  by {line['by']} {line['at']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
