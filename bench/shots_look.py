#!/usr/bin/env python3
"""A person's look at which parts of a subject are seen on frames of a shot, written into the shots record.

    <python> bench/shots_look.py add --shots data/<stem>/shots.json --shot FIRST --subject LABEL --part PART \\
        [--frames A-B] --is "not seen"|"seen" --why "what was looked at, and what was seen" --by "who looked" --entered-by SESSION [--dry]

**What it is for.** The owner's word of 2026-10-14 (about 17:05Z): the records say a subject is IN a shot even
where nothing of them is made, and which of their parts are seen there, so that a later plan (another part of them
replaced) re-derives its loads from the same records. The part model reads slivers (the lead's 5175-5244: her
part on 41 of 70 frames, each frame's largest piece 7 px tall at the median, and no face of hers on any frame by a
look), so a model's reading alone never says a part is not there: a person's line does (the lead's ruling, that day).

**What it writes.** One line under `who[LABEL].seen_looks` of the shot that begins on `--shot`: the part, the source
frames (the whole shot when `--frames` is left out), `is`, why, who looked and when it was entered and by whom.
`bench/shot_coverage.py::nothing_to_make` reads the lines: where they leave every frame of the shot `not seen` for
every part the plan replaces of the subject, the subject is "present, nothing to make under this plan", in the
coverage table, the delivery table and `bench/piece_gaps.py`, and no load is made there. A later `seen` line takes
back an earlier `not seen` on its frames: nothing is ever removed.

**What it refuses.** A shot that does not begin on `--shot`; a subject the record does not have; a subject not
present `yes` there (presence is the owner's, and is answered first); a part that is neither a part name nor a class
of the part model (`sapiens2_parts.PARTS`, `CLASS_NAMES`); frames outside the shot; no why or no by. The record is
replaced in one move, and not when somebody wrote it since it was read; a copy of it before is kept beside it.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shot_coverage  # noqa: E402


class Refused(Exception):
    pass


def part_names() -> tuple[str, ...]:
    """The words a look may name: the part model's part names and its classes (`sapiens2_parts`)."""
    import capture_masked_run as cap
    sp = cap._pack("sapiens2_parts")
    return tuple(sp.PARTS) + tuple(c for c in sp.CLASS_NAMES if c != "Background")


def look(record: dict, shot: int, subject: str, part: str, frames: tuple[int, int] | None, is_: str, why: str, by: str, entered_by: str, now: str,
         names: tuple[str, ...]) -> dict:
    """The line, added to the record in place; refused before anything changes."""
    s = next((x for x in record["shots"] if x["first"] == shot), None)
    if s is None:
        raise Refused(f"no shot of the record begins on {shot}")
    if subject not in record["subjects"]:
        raise Refused(f"{subject!r} is not a subject of the record ({', '.join(record['subjects'])})")
    who = s["who"][subject]
    if who["present"] != "yes":
        raise Refused(f"the record says {subject} is {who['present']!r} on {s['first']}-{s['last']}: presence is the owner's and is answered first")
    if part not in names:
        raise Refused(f"{part!r} is neither a part nor a class of the part model: one of {', '.join(names)}")
    if is_ not in shot_coverage.SEEN_IS:
        raise Refused(f"--is is one of {shot_coverage.SEEN_IS}")
    a, b = frames or (s["first"], s["last"])
    if not (s["first"] <= a <= b <= s["last"]):
        raise Refused(f"frames {a}-{b} are not inside the shot {s['first']}-{s['last']}")
    why, by = " ".join(str(why or "").split()), " ".join(str(by or "").split())
    if not why or not by or not str(entered_by or "").strip():
        raise Refused("a look says what was looked at and what was seen (--why), who looked (--by), and who entered it (--entered-by)")
    line = {"part": part, "first": a, "last": b, "is": is_, "why": why, "by": by, "at": now, "entered_by": entered_by}
    who.setdefault("seen_looks", []).append(line)
    return line


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="step", required=True)
    a = sub.add_parser("add", help="write one look into the shots record")
    a.add_argument("--shots", required=True)
    a.add_argument("--shot", type=int, required=True, help="the first frame of the shot, as the record has it")
    a.add_argument("--subject", required=True)
    a.add_argument("--part", required=True, help="a part name (face_and_neck, hair, lower_clothing, ...) or a class of the part model (Lower_Clothing, ...)")
    a.add_argument("--frames", help="A-B, source frames inside the shot; the whole shot when left out")
    a.add_argument("--is", dest="is_", required=True, choices=shot_coverage.SEEN_IS)
    a.add_argument("--why", required=True, help="what was looked at, and what was seen, in a person's words")
    a.add_argument("--by", required=True, help="who looked")
    a.add_argument("--entered-by", required=True, help="the session writing it in")
    a.add_argument("--dry", action="store_true", help="say the line and write nothing")
    v = ap.parse_args(argv)
    path = Path(v.shots)
    read_as = path.read_text()
    record = json.loads(read_as)
    shot_coverage.load_shots(str(path))                                  # the record as the tools read it, before anything is added
    frames = None
    if v.frames:
        try:
            first, last = (int(x) for x in v.frames.split("-"))
            frames = (first, last)
        except ValueError:
            print(f"REFUSED: --frames {v.frames!r} is not A-B")
            return 5
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        line = look(record, v.shot, v.subject, v.part, frames, v.is_, v.why, v.by, v.entered_by, now, part_names())
    except Refused as e:
        print(f"REFUSED: {e}")
        return 5
    print("the look:", json.dumps(line))
    if v.dry:
        print("dry: nothing written")
        return 0
    if path.read_text() != read_as:
        print(f"REFUSED: {path} changed while this ran; nothing written. Run again.")
        return 5
    copy = path.with_name(f"{path.stem}.before_look_{now.replace(':', '')}.json")
    shutil.copy(path, copy)
    tmp = path.with_name(path.name + ".shots_look.tmp")
    tmp.write_text(json.dumps(record, indent=1) + ("\n" if read_as.endswith("\n") else ""))
    os.replace(tmp, path)
    shot_coverage.load_shots(str(path))                                  # read back as the tools read it
    print(f"wrote {path}; the record before: {copy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
