#!/usr/bin/env python3
"""The cast file: what a person answered about who the outlined people on a frame are, kept so it is typed once.

The least thing that stores a typed answer. A person looks at one tile of a preview's shot table, on which
every detection on that frame is outlined and numbered, and says WHO THE NUMBERED PEOPLE ARE: "2 is the lead",
or "nobody of ours". This file keeps that as one line a named subject: the subject, the frame of the source
copy, and either that person's BOX on the frame or the words `none of the outlined people`. Nothing here
compares anybody with anybody; there is no likeness in it and it decides nothing.

The question is asked of the PEOPLE, never of the subjects, and nothing a tracker concluded is shown or
pre-filled. A default drawn from the trackers' takes would have been wrong on exactly the two shots of
2026-10-10 where a wrong take hid a subject (the lead's tracker holding the second subject; two trackers
holding the only two outlines of a shot the lead was in).

Rules, each a fault of that day this file exists to not repeat:

- AN ANSWER IS OF A FRAME, NOT OF A SHOT. A tile shows one frame, and a subject who walks out half way through a
  shot is not "not in the shot".
- AN UNNAMED SUBJECT IS STORED AS WHAT THE PERSON SAID: `none of the outlined people` on this frame is them. It is
  NOT "not on this frame": a subject small in an inset, turned away or half hidden may be on the frame with no
  outline, and the person was only shown outlines. So `seeds` leaves such a shot unseeded unless the caller
  takes the strong reading in words (`none_outlined`, no default).
- THE NUMBER BECOMES A BOX WHEN THE ANSWER IS GIVEN, from the table the tile was drawn from, and the line names
  that table. No tile number is stored, so a tile made again cannot renumber an old answer.
- THE TILE MUST SHOW EVERYBODY THE DETECTOR FOUND. `answer` refuses a table that says another tracker's people
  were left out on the shown frame, and one that does not say. A preview with one subject and no `others` (a
  census) is such a pass, and the job builder already makes it. A detection with no head is still numbered on
  it, and may be named: a subject seen from behind is one.
- A TILE IS STORED WHOLE OR NOT AT ALL, by one writer: every line of an answer is checked, then all are written
  in one write under a lock. Nothing is ever rewritten; taking an answer back is another line. The latest line
  in the file for a subject and a frame is the answer.

    <python> bench/cast_file.py answer --cast CAST --table CENSUS_SHOTS.json --first-frame N --shot K \\
        --subjects lead,second,third --is 2=lead [--is 5=third] --by NAME
    <python> bench/cast_file.py answer ... --subjects lead,second,third --nobody-of-ours --by NAME
    <python> bench/cast_file.py withdraw --cast CAST --subject lead --frame F --by NAME
    <python> bench/cast_file.py box --cast CAST --subject second --encloses head|body --box F:X,Y,W,H [--box F:X,Y,W,H] \\
        --sheet SHEET.webp --by NAME
    <python> bench/cast_file.py seeds --cast CAST --subject lead --table LOAD_SHOTS.json --first-frame N --none-outlined leave|empty
    <python> bench/cast_file.py show --cast CAST

`--subjects` is every subject the job names: each one not given a number is stored as `none of the outlined
people`. `--first-frame` is the frame of the source copy that the table's frame 0 is on. It is TYPED, and a shot
table holds nothing to check it against: a wrong value stores a true box on a wrong frame and nothing refuses.
A session has to remember to take it from the job that made the table (`job.first`).

`box` enters boxes the owner DREW on a session's sheet (2026-10-23: no tool took one, and the lead was typing them
in by hand): each `--box FRAME:X,Y,W,H` is one `here` line for the subject on that source frame, in the cast file's
own picture size (the one size its boxes are at; a file with none or with two is refused), with `table` naming the
sheet, `by` saying who drew it there, `encloses` saying whether the box is round her head or hair (`head`) or only
her body where her head is hidden (`body`), which is required (`seeds` writes a body box as one, and the tracker seeds
her person track from it and never her head track), and `drawn: approximate`: the owner's boxes are "just to give you a
rough idea" (2026-10-23), so a drawn box says which person, never an outline (`seeds` marks it `drawn`, and the tracker
names the one person lying mostly inside it, `subject_track.py::name_drawn`). A tile's box is a detection's own and
is read as before. The sheet must be an image file that exists. A frame given twice, a subject
the file has no line for, a box outside the picture, and a frame on which the subject already has a standing box (a
box that was wrong is withdrawn first, as for `answer`) are refused, and nothing is written unless every box can be.

`seeds` prints the text the Subject Track's `seeds` input takes for one load (`subject_track.py::parse_seeds`):
one line for EVERY BOX inside the load (2026-10-23, the owner's go: every owner box is a seed on its frame; the
tracker keeps, frame by frame, the track its own score holds best) and one for a shot's `none` read in the strong
reading, and says what it did with every answer of that subject: used, unused and why, in no shot of the table, or
outside the load. An answer outside the load is never moved to a frame inside it. A box is scaled only to a picture
of the same shape; another shape is refused, since the loader crops before it scales. On every frame the subject has
a box on, the OTHER subjects' boxes there are written too, ending in `other`: not seeds, but whom this subject's tracker
must not take on that frame (2026-10-23, the blind read: the first tracker of a job knew nobody else).

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
HERE, NONE_OUTLINED, WITHDRAWN = "here", "none of the outlined people", "withdrawn"
ANSWERS = (HERE, NONE_OUTLINED, WITHDRAWN)
#: What `seeds` may do with a `none of the outlined people` answer. No default: the person was shown outlines on
#: one frame, which says neither that the subject is off that frame nor that they are out of the shot, and the
#: caller says in words which reading it takes. `leave` is the safe one.
LEAVE, EMPTY = "leave", "empty"
NONE_OUTLINED_MEANS = {
    LEAVE: "the shot is left unseeded, for the tracker to judge",
    EMPTY: "the shot is emptied (`frame k: none`): the STRONG reading, that a subject who is none of the people outlined on "
           "one frame is not in the shot at all"}
#: A subject's label, as the job builder's (`workflows/masked_job_graph.py::NAME`): it goes into file names.
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]*\Z")
FIELDS = ("subject", "frame", "answer", "by", "box", "size", "table", "at", "encloses", "drawn")
#: Fields a line may leave out: written only when given, so a line without one reads as it always did.
OPTIONAL = ("encloses", "drawn")
#: What `box` writes as a drawn box's `drawn`: the owner's own word for the boxes on the sheets (2026-10-23, "the
#: yellow/red/blue boxes i drew were approximates!!! they were just to give you a rough idea"). A drawn box says which
#: person is whom, roughly, and is never an outline: `seeds` marks its seed line `drawn`, and the tracker names the one
#: person lying mostly inside it (`subject_track.py::name_drawn`) instead of asking a detection's box to agree with it.
APPROXIMATE = "approximate"
#: What a box the owner drew encloses (`box --encloses`, 2026-10-23, the lead's reading of the owner's sheets): her head
#: or hair, or only her BODY (an arm at the table, where her head is hidden). A body box is written to the tracker as
#: one (`seeds`: its seed line ends in `body`), which seeds her person track and never her head track
#: (`subject_track.py::SEED_BODY`). A box from a tile (`answer`) is of a whole detection and says nothing of this.
HEAD, BODY = "head", "body"
ENCLOSES = (HEAD, BODY)


def entry(subject: str, frame: int, answer: str, *, by: str, box=None, size=None, table: str = "", at: str | None = None,
          encloses: str | None = None, drawn: str | None = None) -> dict:
    """One line of the file, checked. `frame` is a frame of the source copy; `box` is (left, top, width, height) at `size`;
    `encloses`, for a drawn box, says whether it is round her head or only her body (`ENCLOSES`)."""
    if encloses is not None and (encloses not in ENCLOSES or answer != HERE):
        raise ValueError(f"cast: `encloses` is one of {ENCLOSES}, on a `here` line; not {encloses!r} on `{answer}`")
    if drawn is not None and (drawn != APPROXIMATE or answer != HERE):
        raise ValueError(f"cast: `drawn` is `{APPROXIMATE}`, on a `here` line; not {drawn!r} on `{answer}`")
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
            "at": at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **({} if encloses is None else {"encloses": encloses}), **({} if drawn is None else {"drawn": drawn})}


def append(path, lines: list[dict]) -> None:
    """Add the lines of one answer. The only way anything is written: every line checked, then all in one write, under a lock.

    A tile's lines are stored whole or not at all. A file whose last line was cut short (a writer that died) is
    refused, so new lines are never glued to it.

    A STANDING BOX IS NOT REPLACED BY `none of the outlined people`. A person's confirmed box is the strongest
    thing this file holds, and a second preview of the same frame that happens not to outline the subject would
    otherwise erase it by passing over (mrcorn's cold read, by running it: the shown frame is chosen by rule, so a
    census made again lands on the same frames). The answer is refused whole; a box that was wrong is withdrawn
    first, which is two commands a person means. A box replacing a box, and anything replacing a `none`, go in."""
    if not lines:
        raise ValueError("cast: nothing to add")
    data = "".join(json.dumps(entry(**{k: line.get(k) if k in OPTIONAL else line[k] for k in FIELDS}), sort_keys=True) + "\n"
                   for line in lines).encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        size = os.fstat(fd).st_size
        if size and os.pread(fd, 1, size - 1) != b"\n":
            raise ValueError(f"cast: {path} does not end with a whole line; mend it by hand before anything is added")
        standing = current(read(path))                  # read under the lock: no other writer is between this and the write
        for line in lines:
            held = standing.get((line["subject"], int(line["frame"])))
            if line["answer"] == NONE_OUTLINED and held is not None and held["answer"] == HERE:
                raise ValueError(f"cast: {line['subject']} has a confirmed box on frame {line['frame']} ({held['by']}, {held['at']}, from "
                                 f"{held['table']}), and this answer would replace it with `{NONE_OUTLINED}`. Nothing is added. A tile "
                                 f"that does not outline {line['subject']} does not take a box back: if the box was wrong, withdraw it "
                                 "first (`withdraw`), then answer.")
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
            out.append(entry(**{k: line.get(k) for k in FIELDS}))
        except (ValueError, TypeError, AttributeError) as e:
            raise ValueError(f"cast: {path} line {number} cannot be read: {e}") from None
    return out


def current(lines: list[dict]) -> dict[tuple[str, int], dict]:
    """The answer that stands for each (subject, frame): the latest line in the file, and none where that line takes it back."""
    latest = {(line["subject"], line["frame"]): line for line in lines}
    return {key: line for key, line in latest.items() if line["answer"] != WITHDRAWN}


def from_tile(table: dict, table_name: str, first: int, shot: int, subjects: list[str], named: list[tuple[int, str]], by: str) -> list[dict]:
    """A person's answer about one tile, as the lines to store: who each numbered person they named is.

    `subjects` is every subject the job names; `named` is (number on the tile, subject) for each person the answer
    names, and empty for "nobody of ours". Each named subject gets `here` with that person's box; every other
    subject gets `none of the outlined people`. All on the tile's frame, with one time. `first` is the source-copy
    frame of the table's frame 0. Nothing is returned unless every line can be."""
    found = next((s for s in table.get("shots", []) if int(s["shot"]) == int(shot)), None)
    if found is None:
        raise ValueError(f"cast: {table_name} has no shot {shot}")
    _shows_everybody(table, found, table_name, shot)
    subjects = [str(s) for s in subjects]
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError(f"cast: `subjects` is every subject the job names, each once: {subjects}")
    numbers, people = [int(n) for n, _ in named], [str(s) for _, s in named]
    on_tile = {int(p["person"]): p for p in found.get("people", [])}
    for n in numbers:
        if n not in on_tile:
            raise ValueError(f"cast: {table_name}, shot {shot}: the tile has no person {n}; it shows {len(on_tile)}")
    for s in people:
        if s not in subjects:
            raise ValueError(f"cast: `{s}` is not one of the job's subjects, {subjects}")
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"cast: shot {shot}: one numbered person is given to two subjects: {named}")
    if len(set(people)) != len(people):
        raise ValueError(f"cast: shot {shot}: one subject is given two numbered people: {named}")
    frame = int(first) + int(found["shown_frame"])
    at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    is_person = dict(zip(people, numbers))
    return [entry(s, frame, HERE, by=by, box=on_tile[is_person[s]]["box"], size=table["size"], table=table_name, at=at)
            if s in is_person else entry(s, frame, NONE_OUTLINED, by=by, table=table_name, at=at) for s in subjects]


def _shows_everybody(table: dict, found: dict, table_name: str, shot: int) -> None:
    """Refuse a tile whose numbers are not everybody the detector found on its frame, or a table that does not say.

    A version 2 table lists everybody by construction (`shot_table.py::TABLE_VERSION`). A version 1 table has
    three cases, and only one of them is "everybody" (mrcorn's cold read: of forty real tables, twenty-three carry
    the field on no shot, and a missing field was being read as nobody left out): None is what that table wrote
    when the tracker had nobody to leave out; a count is how many it left out; ABSENT is a table that does not
    say. A version this tool does not know is refused: read as complete, it would be the same hole one release on."""
    version = table.get("version")
    if version == 2:
        return
    if version != 1:
        raise ValueError(f"cast: {table_name} is a shot table of version {version!r}, which this tool does not know; it cannot say "
                         "whether its tiles show everybody")
    if "others_left_out" not in found:
        raise ValueError(f"cast: {table_name}, shot {shot}: this table does not say whether anybody was left out of its tiles "
                         "(it was made before the table recorded that), so its numbers cannot be trusted to be everybody. "
                         "Answer from a new preview with one subject and no `others`.")
    left = (found["others_left_out"] or {}).get("on_shown_frame")
    if found["others_left_out"] is not None and left is None:
        raise ValueError(f"cast: {table_name}, shot {shot}: `others_left_out` has no `on_shown_frame`")
    if left:
        raise ValueError(f"cast: {table_name}, shot {shot}: {left} detection(s) were left out of this tile as another tracker's "
                         "people, so its numbers are not everybody on the frame. Answer from a preview with one subject and no `others`.")


#: The files a sheet the owner drew boxes on may be: a picture a person opens to read the boxes off.
SHEET_SUFFIXES = (".webp", ".png", ".jpg", ".jpeg")


def picture_size(lines: list[dict]) -> list[int]:
    """The one picture size the cast file's boxes are read at: its coordinate space. Refused with none, or with two."""
    sizes = sorted({tuple(int(v) for v in line["size"]) for line in lines if line["answer"] == HERE and line.get("size")})
    if len(sizes) != 1:
        raise ValueError(f"cast: the file's boxes are at {len(sizes)} picture size(s) ({', '.join(f'{w}x{h}' for w, h in sizes) or 'none'}), "
                         "so a drawn box has no one coordinate space to be entered in")
    return list(sizes[0])


def drawn(lines: list[dict], subject: str, boxes: list[tuple[int, tuple]], sheet, by: str, encloses: str | None,
          at: str | None = None) -> list[dict]:
    """The lines for boxes the owner DREW on a sheet (2026-10-23): one `here` line a box, checked whole, none written.

    `boxes` is (source frame, (left, top, width, height) in the cast file's own picture size, `picture_size`). Each line
    names the sheet as its `table` and in its `by`, and keeps `encloses` (`ENCLOSES`, required: a box round her body
    alone must not seed her head). Refused, in words: `encloses` not one of them; a sheet that is not an image file that exists; no
    box; a frame given twice; a subject the file has no line for (a typo would make a new subject); a box outside the
    picture or with no area; and a frame on which the subject already has a standing box, which is withdrawn first
    (`withdraw`), as `append` asks of a `none` over a box: a hand-typed frame on top of a confirmed box is likelier a
    slip than a correction."""
    if encloses not in ENCLOSES:
        raise ValueError(f"cast: say what the drawn box encloses, one of {ENCLOSES} (`--encloses`): `{BODY}` for a box round her "
                         "arm or body where her head is hidden, which seeds her person track and never her head track")
    sheet = Path(sheet)
    if not sheet.is_file():
        raise ValueError(f"cast: the sheet {sheet} is not a file: a drawn box cites the sheet it was drawn on, and it must be there")
    if sheet.suffix.lower() not in SHEET_SUFFIXES:
        raise ValueError(f"cast: the sheet {sheet} is not a picture ({', '.join(SHEET_SUFFIXES)}): a drawn box is read off a picture")
    if not str(by).strip():
        raise ValueError("cast: a drawn box says who drew it (`--by`)")
    if not boxes:
        raise ValueError("cast: no box given: `--box FRAME:LEFT,TOP,WIDTH,HEIGHT`, once a box")
    if subject not in {line["subject"] for line in lines}:
        raise ValueError(f"cast: no line of the file names `{subject}`; a drawn box is entered for a subject the file knows "
                         f"({', '.join(sorted({line['subject'] for line in lines})) or 'none yet'})")
    frames = [int(f) for f, _ in boxes]
    twice = sorted({f for f in frames if frames.count(f) > 1})
    if twice:
        raise ValueError(f"cast: frame(s) {twice} given twice for {subject}: the file holds one box a subject a frame")
    standing = current(lines)
    over = [f for f in frames if (standing.get((subject, f)) or {}).get("answer") == HERE]
    if over:
        raise ValueError(f"cast: {subject} already has a standing box on frame(s) {over}: nothing is added. If that box is wrong, "
                         "withdraw it first (`withdraw`), then enter the drawn one")
    size = picture_size(lines)
    at = at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [entry(subject, int(f), HERE, by=f"{' '.join(str(by).split())}; a box drawn on the sheet {sheet}", box=[float(v) for v in box],
                  size=size, table=str(sheet), at=at, encloses=encloses, drawn=APPROXIMATE) for f, box in boxes]


def _drawn_box(text: str) -> tuple[int, tuple]:
    frame, _, numbers = text.partition(":")
    parts = [x.strip() for x in numbers.split(",")]
    try:
        if not frame.strip().isdigit() or len(parts) != 4:
            raise ValueError
        return int(frame), tuple(float(x) for x in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"`{text}`: write FRAME:LEFT,TOP,WIDTH,HEIGHT, as in 4690:512,180,64,72 (a source frame, "
                                         "and the box in the cast file's picture size)") from None


def _number(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.2f}"


def seeds(lines: list[dict], subject: str, first: int, size, shots: list[tuple[int, int]], none_outlined: str) -> dict:
    """The Subject Track's `seeds` text for one load of one subject, and what became of every answer.

    `first` is the source-copy frame of the load's frame 0, `size` its picture (width, height), `shots` its
    (first frame, one past the last) per shot in the load's own frames. `none_outlined` is `LEAVE` or `EMPTY`.
    EVERY BOX INSIDE A SHOT IS USED, each a seed line on its own frame, in frame order (2026-10-23, the owner's go;
    until then a shot took the box latest in the file and listed the rest as unused, and on 1419 the box on 1423 went
    unused and the tracker lost her on 1419-1427). The tracker tracks from each and keeps, frame by frame, the track
    its own score holds best (`subject_track.py::merge_tracks`). A `none of the outlined people` beside a box in the
    same shot is not used, and is listed; in a shot with no box the latest of them in the file is the one read (the
    file's order, as `current` reads it; `at` is never sorted on). Each row's `seeds` is every seed of its shot with
    the answer it came from, and its `seed` their lines joined. What the caller chose for `none_outlined` is in the
    result, in words."""
    if none_outlined not in NONE_OUTLINED_MEANS:
        raise ValueError("cast: say what a `none of the outlined people` answer does to its shot: "
                         + "; ".join(f"`{k}`: {v}" for k, v in NONE_OUTLINED_MEANS.items()))
    last = shots[-1][1] if shots else 0
    mine = [line for (who, _), line in current(lines).items() if who == subject]
    inside = [line for line in mine if 0 <= line["frame"] - first < last]
    outside = sorted(line["frame"] for line in mine if not 0 <= line["frame"] - first < last)
    for line in inside:
        if line["answer"] == HERE and line["size"][0] * size[1] != line["size"][1] * size[0]:
            raise ValueError(f"cast: {subject}'s answer on frame {line['frame']} was read on a {line['size'][0]}x{line['size'][1]} picture "
                             f"and this load is {size[0]}x{size[1]}: another shape. A box is not carried across a crop.")
    order = {id(line): n for n, line in enumerate(lines)}
    # a table's shots meet end to end, so this is empty on a real one; it is here so that every answer is in a list
    in_no_shot = sorted(line["frame"] for line in inside if not any(start <= line["frame"] - first < end for start, end in shots))
    rows, text, unused = [], [], []
    for number, (start, end) in enumerate(shots, 1):
        here = sorted((line for line in inside if start <= line["frame"] - first < end), key=lambda line: order[id(line)])
        row = {"shot": number, "frames": [int(start), int(end) - 1], "seed": None, "seeds": [], "from_frame": None, "why": "no answer inside this shot"}
        boxes = sorted((line for line in here if line["answer"] == HERE), key=lambda line: line["frame"])
        nones = [line for line in here if line["answer"] != HERE]
        for used in boxes:
            factor = size[0] / used["size"][0]
            row["seeds"].append({"seed": f"frame {used['frame'] - first}: " + ", ".join(_number(v * factor) for v in used["box"])
                                         + (" drawn" if used.get("drawn") == APPROXIMATE else "")
                                         + (" body" if used.get("encloses") == BODY else ""),
                                 "from_frame": used["frame"], "why": f"{used['by']}'s answer of {used['at']}"})
        if boxes:
            for other in nones:
                unused.append({"frame": other["frame"], "answer": other["answer"],
                               "why": f"shot {number} of this load has the box(es) on frame(s) {', '.join(str(b['frame']) for b in boxes)}, and a "
                                      "`none of the outlined people` beside a box is no seed; this one is not wrong, only not used"})
            row["why"] = "; ".join(s["why"] for s in row["seeds"])
        elif nones:
            used = nones[-1]
            if none_outlined == EMPTY:
                row["seeds"].append({"seed": f"frame {used['frame'] - first}: none", "from_frame": used["frame"],
                                     "why": (f"{used['by']} said none of the people outlined on frame {used['frame']} is {subject}; the caller took "
                                             "the strong reading, that this empties the shot")})
                row["why"] = row["seeds"][0]["why"]
            else:
                row["why"] = (f"{used['by']} said none of the people outlined on frame {used['frame']} is {subject}, which does not say "
                              f"{subject} is not in the shot; left unseeded")
            row["from_frame"] = used["frame"]
            for other in nones[:-1]:
                unused.append({"frame": other["frame"], "answer": other["answer"],
                               "why": f"shot {number} of this load has a later `none of the outlined people` in the file, on frame "
                                      f"{used['frame']}; this one is not wrong, only not used"})
        if row["seeds"]:
            row["seed"] = "\n".join(s["seed"] for s in row["seeds"])
            row["from_frame"] = row["seeds"][0]["from_frame"]
            text.extend(s["seed"] for s in row["seeds"])
        rows.append(row)
    # THE OTHER SUBJECTS' BOXES on every frame this subject has a box on, marked `other`: not seeds, but whom this tracker
    # must not take there (`subject_track.py::parse_other_boxes`), so the first tracker in a job's order, which is given no
    # earlier tracker's mask, knows them too (2026-10-23, the blind read). Scaled and marked `drawn` as a seed would be
    boxed = {line["frame"] for line in inside if line["answer"] == HERE and any(start <= line["frame"] - first < end for start, end in shots)}
    others = []
    for (who, frame), line in sorted(current(lines).items(), key=lambda item: (item[0][1], item[0][0])):
        if who == subject or frame not in boxed or line["answer"] != HERE:
            continue
        if line["size"][0] * size[1] != line["size"][1] * size[0]:
            raise ValueError(f"cast: {who}'s answer on frame {frame} was read on a {line['size'][0]}x{line['size'][1]} picture "
                             f"and this load is {size[0]}x{size[1]}: another shape. A box is not carried across a crop.")
        factor = size[0] / line["size"][0]
        others.append({"subject": who, "from_frame": frame,
                       "seed": f"frame {frame - first}: " + ", ".join(_number(v * factor) for v in line["box"])
                               + (" drawn" if line.get("drawn") == APPROXIMATE else "") + " other"})
    text.extend(x["seed"] for x in others)
    return {"subject": subject, "first_source_frame": int(first), "size": [int(size[0]), int(size[1])], "text": "\n".join(text),
            "none_outlined": none_outlined, "none_outlined_means": NONE_OUTLINED_MEANS[none_outlined], "shots": rows, "unused": unused,
            "outside_this_load": outside, "inside_the_load_in_no_shot": in_no_shot, "others": others}


def _table(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _pair(text: str) -> tuple[int, str]:
    number, _, subject = text.partition("=")
    if not number.strip().isdigit() or not subject.strip():
        raise argparse.ArgumentTypeError(f"`{text}`: write NUMBER=SUBJECT, as in 2=lead")
    return int(number), subject.strip()


def main() -> int:
    p = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("answer", help="store who the numbered people on one tile are")
    a.add_argument("--table", required=True)
    a.add_argument("--first-frame", type=int, required=True)
    a.add_argument("--shot", type=int, required=True)
    a.add_argument("--subjects", required=True, help="every subject the job names, with commas")
    who = a.add_mutually_exclusive_group(required=True)
    who.add_argument("--is", dest="named", action="append", type=_pair, metavar="NUMBER=SUBJECT")
    who.add_argument("--nobody-of-ours", action="store_true")
    w = sub.add_parser("withdraw", help="take an answer back")
    w.add_argument("--frame", type=int, required=True)
    d = sub.add_parser("box", help="enter boxes the owner drew on a sheet, one line a box. A drawn box is APPROXIMATE (the owner, "
                                   "2026-10-23): it says roughly which person is whom, never an outline, and the line says so "
                                   "(`drawn: approximate`); the tracker takes the one person lying mostly inside it")
    d.add_argument("--box", dest="boxes", action="append", type=_drawn_box, required=True, metavar="FRAME:LEFT,TOP,WIDTH,HEIGHT",
                   help="a source frame and the box on it, in the cast file's picture size, as roughly as it was drawn; once a box")
    d.add_argument("--sheet", required=True, help="the picture the owner drew the boxes on; it must be there")
    d.add_argument("--encloses", required=True, choices=ENCLOSES,
                   help="what every box of this call encloses: `head` (her head or hair) or `body` (only her arm or body, her head "
                        "hidden), which seeds her person track and never her head track")
    s = sub.add_parser("seeds", help="the Subject Track's seeds text for one load")
    s.add_argument("--table", required=True, help="a shot table of the load: its shots and its picture size are read")
    s.add_argument("--first-frame", type=int, required=True)
    s.add_argument("--none-outlined", required=True, choices=sorted(NONE_OUTLINED_MEANS))
    show = sub.add_parser("show", help="the answers that stand")
    for one in (a, w, d, s, show):
        one.add_argument("--cast", required=True)
    for one in (w, d, s):
        one.add_argument("--subject", required=True)
    for one in (a, w, d):
        one.add_argument("--by", required=True)
    args = p.parse_args()
    try:
        return _run(args)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def _run(args) -> int:
    if args.cmd == "answer":
        lines = from_tile(_table(args.table), args.table, args.first_frame, args.shot,
                          [x.strip() for x in args.subjects.split(",") if x.strip()], args.named or [], args.by)
        append(args.cast, lines)
        for line in lines:
            print(json.dumps(line, sort_keys=True))
    elif args.cmd == "box":
        lines = drawn(read(args.cast), args.subject, args.boxes, args.sheet, args.by, args.encloses)
        append(args.cast, lines)
        for line in lines:
            print(json.dumps(line, sort_keys=True))
    elif args.cmd == "withdraw":
        if (args.subject, args.frame) not in current(read(args.cast)):
            raise SystemExit(f"cast: no answer stands for {args.subject} on frame {args.frame}")
        append(args.cast, [entry(args.subject, args.frame, WITHDRAWN, by=args.by)])
    elif args.cmd == "seeds":
        table = _table(args.table)
        out = seeds(read(args.cast), args.subject, args.first_frame, table["size"],
                    [(int(x["first_frame"]), int(x["last_frame"]) + 1) for x in table["shots"]], args.none_outlined)
        print(out["text"])
        print(f"# a `{NONE_OUTLINED}` answer: {out['none_outlined_means']} (--none-outlined {out['none_outlined']})", file=sys.stderr)
        for row in out["shots"]:
            print(f"# shot {row['shot']} (frames {row['frames'][0]}-{row['frames'][1]}): {'; '.join(x['seed'] for x in row['seeds']) or 'no seed'}; "
                  f"{row['why']}", file=sys.stderr)
        for one in out["unused"]:
            print(f"# UNUSED: the answer on frame {one['frame']} ({one['answer']}): {one['why']}", file=sys.stderr)
        for one in out["others"]:
            print(f"# {one['subject']}'s box on frame {one['from_frame']}, carried as another subject's: {one['seed']}", file=sys.stderr)
        if out["inside_the_load_in_no_shot"]:
            print(f"# IN NO SHOT of this table, not used: {args.subject}'s answers on frames {out['inside_the_load_in_no_shot']}", file=sys.stderr)
        if out["outside_this_load"]:
            print(f"# outside this load, not moved: {args.subject}'s answers on frames {out['outside_this_load']}", file=sys.stderr)
    else:
        for (subject, frame), line in sorted(current(read(args.cast)).items()):
            print(f"{subject:12s} frame {frame:6d}  {line['answer']:28s} {'' if line['box'] is None else line['box']}"
                  + (f" ({line['encloses']})" if line.get("encloses") else "") + (f" ({line['drawn']})" if line.get("drawn") else "")
                  + f"  by {line['by']} {line['at']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
