#!/usr/bin/env python3
"""Join two shots of a video's shots record that are one shot, or cut one that is two, keeping every answer whole.

    <python> bench/shots_record.py merge --shots data/<stem>/shots.json --first FIRST_OF_A --second FIRST_OF_B \\
        --why "what was looked at and what it showed" --by SESSION [--stands "SUBJECT=yes:why"] [--dry-run]
    <python> bench/shots_record.py split --shots data/<stem>/shots.json --shot FIRST --at FRAME --keeps first|second \\
        --other "SUBJECT=unsure:what is known of them on the other cut" [--other ...] [--note-other "words"] \\
        --why "..." --by SESSION [--dry-run]

**What it buys.** The record's cuts are where every later tool takes its shots from (a preview is of a shot, a load
of a shot, a `source` line of the delivery table of a shot), and a cut can be wrong both ways. On 2026-10-13 two
were found on one video: a cut the record had and the picture did not, where two working copies of the source met
inside one shot, so half of a continuous shot was replaced and half was not; and a shot that was two, found by a
session's look at every frame, whose second cut no answer of the owner's was on. Until then no tool joined or cut a
shot; a record's shots were changed by a session's script.

**merge.** Two shots that are neighbours become one. For each subject: `present` is the two halves' word where they
agree; where they do not it is REFUSED unless `--stands` says which stands and why (a person's reading, written into
the entry). `how` is both halves' words whole, each headed by the frames it was said of; `by` both; `frames_on` and
`frames_boxed` the two lists joined. The shot's `note` is both notes whole and a line saying when, by whom and why.
The two shots AS THEY WERE are kept whole under the new shot's `merged_from`, so nothing they held is lost.

**split.** One shot becomes two at `--at`, the first frame of the second. The caller says which half keeps the
shot's entries whole (`--keeps`: the half the owner's answers are on) and what the OTHER half says of every subject
(`--other`, one for each: usually `unsure`, since no answer is on it). REFUSED when a frame an entry names
(`frames_on`, `frames_boxed`, or a `frame N` in its `how`) lies on the other half: the entries then belong to both
and a person divides them by hand. The other half's entries say they came of a split and what the whole shot said.
The shot AS IT WAS is kept under both halves' `split_from`.

A dry run prints the shots before and after and writes nothing. A real run writes the record whole, as its other
writers do, and refuses if the file changed while it ran. It does not make a copy of the record: the session that
runs it does, first.

**What it cannot see.** Whether the picture has a cut there: that is the `--why`, a person's or a session's look.
What a later answer of the owner's should be on a half nobody answered: it is `unsure` until they do. Nothing that
was made of the old shots (a preview, a load, a batch of picks) is touched or told.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "h3_shots/1"
PRESENT = ("yes", "no", "unsure")


class Refused(Exception):
    pass


def frames(shot: dict) -> str:
    return f"{shot['first']}-{shot['last']}"


def said(text: str, kind: str) -> tuple[str, str, str]:
    """`SUBJECT=present:words` as (subject, present, words)."""
    try:
        subject, rest = text.split("=", 1)
        present, words = rest.split(":", 1)
        if present not in PRESENT or not subject.strip() or not words.strip():
            raise ValueError
    except ValueError:
        raise Refused(f"{kind} {text!r} is not `SUBJECT=yes|no|unsure:words`") from None
    return subject.strip(), present, " ".join(words.split())


def merged(a: dict, b: dict, subjects: list[str], why: str, by: str, now: str, stands: dict | None = None) -> dict:
    """The one shot of two neighbours, `a` then `b`."""
    if a["last"] + 1 != b["first"]:
        raise Refused(f"{frames(a)} and {frames(b)} are not neighbours: a merge joins a shot with the one after it")
    stands = stands or {}
    who = {}
    for subject in subjects:
        x, y = a["who"][subject], b["who"][subject]
        if x["present"] == y["present"]:
            present, ruled = x["present"], ""
        elif subject in stands:
            present, ruled = stands[subject][0], f"; the two halves differ ({x['present']} and {y['present']}) and `{stands[subject][0]}` stands: {stands[subject][1]} ({by})"
        else:
            raise Refused(f"{subject} is `{x['present']}` on {frames(a)} and `{y['present']}` on {frames(b)}: say which stands for the whole shot, "
                          f"--stands \"{subject}=yes|no|unsure:why\"")
        entry = {"present": present, "how": f"{frames(a)}: {x.get('how', '')}; {frames(b)}: {y.get('how', '')}" + ruled,
                 "by": x.get("by", "") if x.get("by", "") == y.get("by", "") else f"{frames(a)}: {x.get('by', '')}; {frames(b)}: {y.get('by', '')}"}
        for key in ("frames_on", "frames_boxed"):
            if key in x or key in y:
                entry[key] = sorted(set(x.get(key) or []) | set(y.get(key) or []))
        who[subject] = entry
    notes = [f"{frames(s)}: {s['note']}" for s in (a, b) if s.get("note")]
    notes.append(f"MERGED {now} by {by} from the record's two shots {frames(a)} and {frames(b)}: {why}")
    return {"first": a["first"], "last": b["last"], "who": who, "note": " || ".join(notes), "merged_from": [copy.deepcopy(a), copy.deepcopy(b)]}


def named_frames(entry: dict) -> set[int]:
    """Every frame an entry names: its frame fields, and each `frame N` of its words."""
    return set(entry.get("frames_on") or []) | set(entry.get("frames_boxed") or []) | {int(n) for n in re.findall(r"\bframe (\d+)", str(entry.get("how", "")))}


def split(shot: dict, at: int, keeps: str, other: dict, subjects: list[str], why: str, by: str, now: str, note_other: str = "") -> tuple[dict, dict]:
    """The two shots of one, cut so that `at` is the second's first frame. `keeps` is the half the entries stay whole on."""
    if not shot["first"] < at <= shot["last"]:
        raise Refused(f"--at {at} is not inside {frames(shot)} past its first frame: it is the first frame of the second shot")
    if keeps not in ("first", "second"):
        raise Refused("--keeps is `first` or `second`: the half the shot's entries stay whole on")
    lo, hi = (shot["first"], at - 1) if keeps == "first" else (at, shot["last"])
    missing = [s for s in subjects if s not in other]
    if missing:
        raise Refused(f"the other half needs a word for every subject: none was given for {', '.join(missing)} (--other \"SUBJECT=unsure:words\")")
    for subject in subjects:
        astray = sorted(n for n in named_frames(shot["who"][subject]) if shot["first"] <= n <= shot["last"] and not lo <= n <= hi)
        if astray:
            raise Refused(f"{subject}'s entry names frame(s) {astray}, which lie on the other half ({'second' if keeps == 'first' else 'first'}): its "
                          "answers are on both cuts and are divided by a person, not by this tool")
    line = f"SPLIT {now} by {by} from the record's shot {frames(shot)} at {at}: {why}"
    kept = {k: copy.deepcopy(v) for k, v in shot.items() if k not in ("first", "last", "note")}
    kept.update(note=" || ".join(x for x in (shot.get("note", ""), line) if x), split_from=copy.deepcopy(shot))
    who = {}
    for subject in subjects:
        was = shot["who"][subject]
        present, words = other[subject]
        who[subject] = {"present": present, "how": f"{words}; the whole shot {frames(shot)} was `{was['present']}` ({was.get('how', '')}), said on the other cut",
                        "by": by, "frames_on": [], "frames_boxed": []}
    made = {"who": who, "note": " || ".join(x for x in (line, note_other) if x), "split_from": copy.deepcopy(shot)}
    first_half, second_half = (kept, made) if keeps == "first" else (made, kept)
    return ({"first": shot["first"], "last": at - 1, **first_half}, {"first": at, "last": shot["last"], **second_half})


def checked(record: dict) -> None:
    """The record still covers its span with no gap and no overlap, and every shot has every subject."""
    last = None
    for s in record["shots"]:
        if s["first"] > s["last"] or (last is not None and s["first"] != last + 1):
            raise Refused(f"after the change shot {frames(s)} does not follow {last} with no gap and no overlap")
        last = s["last"]
        for subject in record["subjects"]:
            if s["who"].get(subject, {}).get("present") not in PRESENT:
                raise Refused(f"after the change shot {frames(s)} has no `present` for {subject}")
    if record["shots"][0]["first"] != record["span"][0] or last != record["span"][1]:
        raise Refused("after the change the shots no longer cover the record's span")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="what", required=True)
    for name in ("merge", "split"):
        p = sub.add_parser(name)
        p.add_argument("--shots", required=True)
        p.add_argument("--why", required=True, help="what was looked at and what it showed; written into the shot's note")
        p.add_argument("--by", required=True, help="the session")
        p.add_argument("--dry-run", action="store_true")
        if name == "merge":
            p.add_argument("--first", type=int, required=True, help="the first frame of the first of the two shots")
            p.add_argument("--second", type=int, required=True, help="the first frame of the second")
            p.add_argument("--stands", action="append", default=[], metavar="SUBJECT=yes|no|unsure:why", help="where the two halves differ for a subject: which stands, and why")
        else:
            p.add_argument("--shot", type=int, required=True, help="the first frame of the shot to cut")
            p.add_argument("--at", type=int, required=True, help="the first frame of the second shot")
            p.add_argument("--keeps", required=True, choices=("first", "second"), help="the half the shot's entries stay whole on: the one the answers are on")
            p.add_argument("--other", action="append", default=[], metavar="SUBJECT=yes|no|unsure:words", help="what the other half says of a subject; one for each")
            p.add_argument("--note-other", default="", help="words for the other half's note: a session's reading of it, said as that")
    a = ap.parse_args()
    text = Path(a.shots).read_text()
    record = json.loads(text)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        if record.get("schema") != SCHEMA:
            raise Refused(f"{a.shots}: schema is {record.get('schema')!r}, not {SCHEMA!r}")
        by_first = {s["first"]: n for n, s in enumerate(record["shots"])}
        if a.what == "merge":
            if a.first not in by_first or a.second not in by_first:
                raise Refused(f"{a.first} and {a.second} are not both the first frame of a shot of the record")
            i = by_first[a.first]
            if i + 1 >= len(record["shots"]) or record["shots"][i + 1]["first"] != a.second:
                raise Refused(f"the shot after {frames(record['shots'][i])} does not begin on {a.second}: a merge joins a shot with the one after it")
            before = record["shots"][i:i + 2]
            after = [merged(before[0], before[1], record["subjects"], " ".join(a.why.split()), a.by, now,
                            {s: (p, w) for s, p, w in (said(t, "--stands") for t in a.stands)})]
            record["shots"][i:i + 2] = after
            did = f"shots {frames(before[0])} and {frames(before[1])} merged into {frames(after[0])}"
        else:
            if a.shot not in by_first:
                raise Refused(f"{a.shot} is not the first frame of a shot of the record")
            i = by_first[a.shot]
            before = record["shots"][i:i + 1]
            after = list(split(before[0], a.at, a.keeps, {s: (p, w) for s, p, w in (said(t, "--other") for t in a.other)}, record["subjects"],
                               " ".join(a.why.split()), a.by, now, " ".join(a.note_other.split())))
            record["shots"][i:i + 1] = after
            did = f"shot {frames(before[0])} split into {frames(after[0])} and {frames(after[1])}"
        checked(record)
    except Refused as e:
        print(f"REFUSED, nothing written: {e}")
        return 5
    line = f"{did} by bench/shots_record.py for {a.by} at {now}: {' '.join(a.why.split())}"
    record["written_by"] = f"{record['written_by']}; {line}" if record.get("written_by") else line
    show = lambda shots: json.dumps([{k: v for k, v in s.items() if k not in ("merged_from", "split_from")} for s in shots], indent=1)      # noqa: E731
    print(f"BEFORE:\n{show(before)}\n\nAFTER (each also keeps the shot or shots as they were, whole, under `merged_from` or `split_from`):\n{show(after)}\n")
    print(f"the record: {len(record['shots'])} shots; " + ("DRY: nothing written" if a.dry_run else f"{did}; written to {a.shots}"))
    if a.dry_run:
        return 0
    if Path(a.shots).read_text() != text:
        print(f"REFUSED, nothing written: {a.shots} changed while this ran")
        return 5
    # one space of indent and no last newline: how the record's other writers write it
    Path(a.shots).write_text(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
