#!/usr/bin/env python3
"""Join two shots of a video's shots record that are one shot, or cut one that is two, keeping every answer whole.

    <python> bench/shots_record.py merge --shots data/<stem>/shots.json --first FIRST_OF_A --second FIRST_OF_B \\
        --why "what was looked at and what it showed" --by SESSION [--stands "SUBJECT=yes:why"] [--dry-run]
    <python> bench/shots_record.py split --shots data/<stem>/shots.json --shot FIRST --at FRAME --keeps first|second \\
        --other "SUBJECT=unsure:what is known of them on the other cut" [--other ...] [--note-other "words"] \\
        --why "..." --by SESSION [--answers-on-both "who divides them, and how"] [--dry-run]
    <python> bench/shots_record.py move --shots data/<stem>/shots.json --first FIRST_OF_A --second FIRST_OF_B --to FRAME \\
        --why "..." --by SESSION [--dry-run]
    <python> bench/shots_record.py note --shots data/<stem>/shots.json --shot FIRST --add "words" --by SESSION [--dry-run]
    <python> bench/shots_record.py answer --shots data/<stem>/shots.json --shot FIRST --answers data/<stem>/owner_answers.jsonl \\
        --lines N[,N...] --why "..." --by SESSION [--dry-run]

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
The shot AS IT WAS is kept under both halves' `split_from`. Where the owner DID answer on both cuts
(`--answers-on-both`, with who divides them and how), the split goes through: the kept half keeps its entries
whole, its note says which of the frames they name are now the other shot's, and the other half says what the
caller gives, which is then the owner's own answer on its frames, quoted with its lines.

**move.** The cut between two neighbours moves to `--to`, the second's new first frame: a cut found a frame or two
from where the picture has it. Each shot keeps its own entries whole. REFUSED when a frame either shot's entries
name would change sides: that is a merge and a split, a person's to divide. Both shots as they were are kept
under `moved_from`.

**note.** Words added to a shot's note, with when and by whom: what a look found there that changes no cut.

**answer.** Who is in ONE shot, set from the owner's own answer lines, named by their numbers in the page's
answers file (2026-10-13: a shot cut out of another said `unsure` for everybody, the owner answered on its
numbered picture, and the step that carries a plain frame's answers into this record does not carry a numbered
picture's). Each named line must be the owner's, a choice, on a frame inside the shot. A subject with a line that
names a person or says "not outlined" is `yes`; one whose lines all say "not on frame" is `no`, as a plain
frame's answers have always been read; a subject with no line among those named is left exactly as it was. The
value is READ from the lines and never typed. The entries as they were are kept under `answered_over`.

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


def split(shot: dict, at: int, keeps: str, other: dict, subjects: list[str], why: str, by: str, now: str, note_other: str = "",
          on_both: str = "") -> tuple[dict, dict]:
    """The two shots of one, cut so that `at` is the second's first frame. `keeps` is the half the entries stay whole on."""
    if not shot["first"] < at <= shot["last"]:
        raise Refused(f"--at {at} is not inside {frames(shot)} past its first frame: it is the first frame of the second shot")
    if keeps not in ("first", "second"):
        raise Refused("--keeps is `first` or `second`: the half the shot's entries stay whole on")
    lo, hi = (shot["first"], at - 1) if keeps == "first" else (at, shot["last"])
    missing = [s for s in subjects if s not in other]
    if missing:
        raise Refused(f"the other half needs a word for every subject: none was given for {', '.join(missing)} (--other \"SUBJECT=unsure:words\")")
    both = set()
    for subject in subjects:
        astray = sorted(n for n in named_frames(shot["who"][subject]) if shot["first"] <= n <= shot["last"] and not lo <= n <= hi)
        both.update(astray)
        if astray and not on_both:
            raise Refused(f"{subject}'s entry names frame(s) {astray}, which lie on the other half ({'second' if keeps == 'first' else 'first'}): its "
                          "answers are on both cuts and are divided by a person, not by this tool (--answers-on-both \"who divides them, and how\")")
    line = f"SPLIT {now} by {by} from the record's shot {frames(shot)} at {at}: {why}"
    if both:
        line += (f" THE OWNER'S ANSWERS WERE ON BOTH CUTS: the entries here are kept whole and also name frame(s) {sorted(both)}, which are now the other "
                 f"shot's; divided by {by}: {on_both}")
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


def moved(a: dict, b: dict, to: int, subjects: list[str], why: str, by: str, now: str) -> tuple[dict, dict]:
    """Two neighbours with the cut between them moved to `to`, the second's new first frame; each keeps its entries whole."""
    if a["last"] + 1 != b["first"]:
        raise Refused(f"{frames(a)} and {frames(b)} are not neighbours: a cut is moved between a shot and the one after it")
    if not a["first"] < to <= b["last"] or to == b["first"]:
        raise Refused(f"--to {to} is not a new first frame for the second shot: it is past {a['first']}, no later than {b['last']}, and not {b['first']}, where the cut is")
    for subject in subjects:
        gone = sorted(n for n in named_frames(a["who"][subject]) if to <= n <= b["last"]) + sorted(n for n in named_frames(b["who"][subject]) if a["first"] <= n < to)
        if gone:
            raise Refused(f"{subject}'s answers name frame(s) {gone}, which would change shots with the cut at {to}: that is a merge and a split, divided by a person")
    line = f"CUT MOVED {now} by {by} from {b['first']} to {to} (the shots were {frames(a)} and {frames(b)}): {why}"
    was = [copy.deepcopy(a), copy.deepcopy(b)]
    out = []
    for shot, first, last in ((a, a["first"], to - 1), (b, to, b["last"])):
        new = copy.deepcopy(shot)
        new.update(first=first, last=last, note=" || ".join(x for x in (shot.get("note", ""), line) if x), moved_from=was)
        out.append(new)
    return out[0], out[1]


def answered(shot: dict, lines: list[tuple[int, dict]], subjects: list[str], why: str, by: str, now: str, answers_name: str) -> dict:
    """One shot with `present` set for every subject the owner's named answer lines speak of. `lines` is (line number, line)."""
    said: dict[str, list] = {}
    for n, d in lines:
        frame = d.get("source_frame")
        if d.get("who") != "owner" or d.get("what") != "choice" or d.get("subject") not in subjects or not str(frame).isdigit():
            raise Refused(f"line {n} of {answers_name} is not a choice of the owner's for a subject of the record on a frame: {json.dumps(d)[:200]}")
        if not shot["first"] <= int(frame) <= shot["last"]:
            raise Refused(f"line {n} of {answers_name} is on frame {frame}, which is not in the shot {frames(shot)}")
        said.setdefault(d["subject"], []).append((n, int(frame), str(d.get("chosen")), d))
    if not said:
        raise Refused("no answer line was named")
    new = copy.deepcopy(shot)
    new["answered_over"] = (shot.get("answered_over") or []) + [{"at": now, "by": by, "who": copy.deepcopy(shot["who"])}]
    told = []
    for subject, rows in said.items():
        rows = {(frame, d.get("table") or ""): (n, frame, chosen, d) for n, frame, chosen, d in rows}.values()      # the last line on a picture is the answer
        here = [r for r in rows if r[2] != "not on frame"]
        words = []
        for n, frame, chosen, d in rows:
            where = f"the numbered picture {d['table']}" if d.get("table") else "the source itself"
            what = f"person {chosen}" if chosen.isdigit() else "on this frame, not outlined" if chosen == "not outlined" else "not on this frame"
            words.append(f"frame {frame} ({where}): {what}")
        ns = sorted(n for n, *_ in rows)
        at = max(str(d.get("at", "")) for *_, d in rows)
        new["who"][subject] = {"present": "yes" if here else "no", "how": "; ".join(words),
                               "by": f"owner, on the who page at {at} ({answers_name} lines {', '.join(map(str, ns))}), entered by shots_record.py answer for {by}"}
        told.append(f"{subject} {'yes' if here else 'no'} (was {shot['who'][subject]['present']})")
    new["note"] = " || ".join(x for x in (shot.get("note", ""), f"ANSWERED {now}, entered by {by}: {', '.join(told)}, by the owner's own lines. {why}") if x)
    return new


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
    for name in ("merge", "split", "move", "note", "answer"):
        p = sub.add_parser(name)
        p.add_argument("--shots", required=True)
        if name != "note":
            p.add_argument("--why", required=True, help="what was looked at and what it showed; written into the shot's note")
        p.add_argument("--by", required=True, help="the session")
        p.add_argument("--dry-run", action="store_true")
        if name == "move":
            p.add_argument("--first", type=int, required=True, help="the first frame of the first of the two shots")
            p.add_argument("--second", type=int, required=True, help="the first frame of the second, where the cut is now")
            p.add_argument("--to", type=int, required=True, help="the second shot's new first frame")
        elif name == "note":
            p.add_argument("--shot", type=int, required=True, help="the first frame of the shot")
            p.add_argument("--add", required=True, help="the words to add to its note")
        elif name == "answer":
            p.add_argument("--shot", type=int, required=True, help="the first frame of the shot")
            p.add_argument("--answers", required=True, help="the page's answers file the lines are in")
            p.add_argument("--lines", required=True, metavar="N[,N...]", help="the owner's answer lines, by their numbers in that file, counted from 1")
        elif name == "merge":
            p.add_argument("--first", type=int, required=True, help="the first frame of the first of the two shots")
            p.add_argument("--second", type=int, required=True, help="the first frame of the second")
            p.add_argument("--stands", action="append", default=[], metavar="SUBJECT=yes|no|unsure:why", help="where the two halves differ for a subject: which stands, and why")
        else:
            p.add_argument("--shot", type=int, required=True, help="the first frame of the shot to cut")
            p.add_argument("--at", type=int, required=True, help="the first frame of the second shot")
            p.add_argument("--keeps", required=True, choices=("first", "second"), help="the half the shot's entries stay whole on: the one the answers are on")
            p.add_argument("--other", action="append", default=[], metavar="SUBJECT=yes|no|unsure:words", help="what the other half says of a subject; one for each")
            p.add_argument("--note-other", default="", help="words for the other half's note: a session's reading of it, said as that")
            p.add_argument("--answers-on-both", default="", metavar="WORDS", help="the owner answered on frames of both cuts: who divides the answers, and how; "
                           "without it such a split is refused")
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
        elif a.what == "move":
            if a.first not in by_first or a.second not in by_first:
                raise Refused(f"{a.first} and {a.second} are not both the first frame of a shot of the record")
            i = by_first[a.first]
            if i + 1 >= len(record["shots"]) or record["shots"][i + 1]["first"] != a.second:
                raise Refused(f"the shot after {frames(record['shots'][i])} does not begin on {a.second}")
            before = record["shots"][i:i + 2]
            after = list(moved(before[0], before[1], a.to, record["subjects"], " ".join(a.why.split()), a.by, now))
            record["shots"][i:i + 2] = after
            did = f"the cut between {frames(before[0])} and {frames(before[1])} moved to {a.to}: now {frames(after[0])} and {frames(after[1])}"
        elif a.what == "answer":
            if a.shot not in by_first:
                raise Refused(f"{a.shot} is not the first frame of a shot of the record")
            every = Path(a.answers).read_text().splitlines()
            try:
                wanted = [int(x) for x in a.lines.split(",")]
                named = [(n, json.loads(every[n - 1])) for n in wanted]
            except (ValueError, IndexError):
                raise Refused(f"--lines {a.lines!r}: line numbers of {a.answers}, counted from 1, with commas ({len(every)} lines there)") from None
            i = by_first[a.shot]
            before = record["shots"][i:i + 1]
            after = [answered(before[0], named, record["subjects"], " ".join(a.why.split()), a.by, now, Path(a.answers).name)]
            record["shots"][i:i + 1] = after
            did = f"who is in shot {frames(after[0])} set from the owner's answer lines {a.lines}"
        elif a.what == "note":
            if a.shot not in by_first or not a.add.strip():
                raise Refused(f"{a.shot} is not the first frame of a shot of the record, or --add has no words")
            i = by_first[a.shot]
            before = record["shots"][i:i + 1]
            new = copy.deepcopy(before[0])
            new["note"] = " || ".join(x for x in (new.get("note", ""), f"NOTE {now} by {a.by}: {' '.join(a.add.split())}") if x)
            after = [new]
            record["shots"][i:i + 1] = after
            did = f"a note added to shot {frames(new)}"
        else:
            if a.shot not in by_first:
                raise Refused(f"{a.shot} is not the first frame of a shot of the record")
            i = by_first[a.shot]
            before = record["shots"][i:i + 1]
            after = list(split(before[0], a.at, a.keeps, {s: (p, w) for s, p, w in (said(t, "--other") for t in a.other)}, record["subjects"],
                               " ".join(a.why.split()), a.by, now, " ".join(a.note_other.split()), " ".join(a.answers_on_both.split())))
            record["shots"][i:i + 1] = after
            did = f"shot {frames(before[0])} split into {frames(after[0])} and {frames(after[1])}"
        checked(record)
    except Refused as e:
        print(f"REFUSED, nothing written: {e}")
        return 5
    line = f"{did} by bench/shots_record.py for {a.by} at {now}: {' '.join((a.add if a.what == 'note' else a.why).split())}"
    record["written_by"] = f"{record['written_by']}; {line}" if record.get("written_by") else line
    show = lambda shots: json.dumps([{k: v for k, v in s.items() if k not in ("merged_from", "split_from", "moved_from", "answered_over")} for s in shots], indent=1)      # noqa: E731
    print(f"BEFORE:\n{show(before)}\n\nAFTER (each also keeps the shot or shots as they were, whole, under `merged_from`, `split_from` or `moved_from`):\n{show(after)}\n")
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
