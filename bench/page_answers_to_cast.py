#!/usr/bin/env python3
"""The owner's answers on the review page's numbered pictures, taken into the video's cast file through `bench/cast_file.py`.

    <python> bench/page_answers_to_cast.py --answers data/<stem>/poc/answers.jsonl --cast data/<stem>/cast/cast.jsonl \\
        [--preview JOB.json=TABLES_DIR ...] [--shots data/<stem>/shots.json ...] [--dry-run]

A numbered picture's tile goes into the cast file; a plain frame's answer goes into a shots record (`--shots`).

**What it buys.** The page saves one raw line a click (`answers.jsonl`): a table's file name, the frame of the
source copy the tile shows, a person's number and a button. The owner's rule (2026-10-11, Open List row g01):
those lines are raw input, saved twice, changed a second later or answered two ways, and a session turns them
into the record's structured data and asks where two disagree. On 2026-10-11 eighty-one such lines were saved
and none reached the cast file, so the next preview was seeded by corrections typed by hand. This is the one
way a line gets from the page into the cast, and it goes through `cast_file.from_tile`, whose rules are the
cast's: a tile is stored whole or not at all, a number becomes a box, a subject not named is `none of the
outlined people`, the latest line stands.

**How a line is read.** Only lines whose `who` is `owner` and whose `row` is a who-is-who row (`WHO_ROWS`) are
taken. `who` is the page's "Answering as" box as it stood when the click was posted, not a proof of who clicked:
the page server does not know, so a session testing the page sets the box to its own name first. Two shapes of click: the first page's (a numbered PERSON and a button naming who they are, `BUTTONS`), and
the who page's of 2026-10-12 (a SUBJECT and a button: their number on the tile, `not outlined`, or `not in
shot` (the page's label until 2026-10-12 14:45Z) or `not on frame`; all three store as `none of the outlined people`, the cast's one word, and `seeds` decides per
load how strongly to read it). Otherwise: a session's test clicks are listed as left out. Lines are grouped by tile (the table's file name and the
source frame it shows); on a tile, the LATEST click for each numbered person is the answer (`taken back` leaves
that person unanswered). WHICH SUBJECTS GET A LINE depends on what the tile was asked: of the numbered people (the
first page), naming them says of every other subject of the job that they are none of the people outlined, so
every subject of the job gets a line; of the subjects one by one (the who page), a subject nobody clicked for was
not answered, so only the subjects clicked get one, and the others are printed as NOT ANSWERED (2026-10-13: until
then a click for one subject of a two-subject tile stored `none of the outlined people` for the other, which the
owner had not said, and a tile with no number on it, only `on this frame, not outlined`, stored nothing at all,
though that is the answer for a subject the tracker did not outline). A button is read through `BUTTONS` to a subject's label or to `none of these`; a `not
sure` that stands on any person means the tile is NOT stored and is printed as waiting on the owner, since
storing it would write `none of the outlined people` for a subject the owner had not decided on. The owner's
typed words on the tile are printed beside it, whole, for whoever runs this to read: words count over a button
(the owner, 2026-10-11), and this tool does not read them. A tile whose words and buttons disagree is for a
person to settle, by a `withdraw` and an `answer` on `cast_file.py`.

**Which table and which shot.** A `--preview` names a preview's job file and the folder its shot tables are in:
the job gives the subjects every tile of that preview asks about and the source frame its table frame 0 is on
(`job.first`); the tables are `<job name>_<label>_shots_00001_shots.json` in the folder. A line's tile is the
shot of its table whose shown frame is the line's source frame; a line naming a table no preview holds, or a
frame no shot shows, is refused by name. Every tile is read before any is written, so such a refusal leaves the
cast file as it was (until 2026-10-13 it came after the tiles before it had been appended). A click naming a
subject the tile's job does not track (a page made before `who_page.py` asked only about a job's subjects) is
not a reason to stop: that tile is printed as NOT STORED, whole, and the run goes on.

**A plain frame** (the who page's tile of the source itself, where no tracker outlined anybody, and a long shot's
quarter frames: table `plain`, no person's number). The owner says of each subject `on this frame, not outlined`
or `not on this frame`. NOTHING of it goes to the cast file (the lead's ruling, 2026-10-12): a cast line says
which outlined number is which person and seeds a tracker, and `none of the outlined people` on a frame with no
outlines says nothing true about the cast and would let `seeds` empty a shot the owner said the subject is in.
It is a `present` entry of the shots record: a `--shots` file is the video's `h3_shots/1` table, and a subject's
`present` in the shot that holds the frame becomes `yes` when any answered frame of the shot has them on it (a
plain frame's `on this frame, not outlined`, or a box that stands in the cast file inside the shot, which is
read and not written), else `no` when every answered plain frame says `not on this frame`; `how` names each
frame and what was said on it, and `by` the time and the lines. That `no` is of the frames named and is written
as the shot's, as the record's rows made by hand already were; a subject with no plain answer in a shot is left
as it was. A plain tile is taken only when a `--shots` file holds it: the same `video`, and a shot whose first
and last frame are the ones the page showed the tile under (a table cut again since the page was made is refused
by name); with none it is listed and nothing of it is stored. An entry of the record that somebody else wrote
and that is not `unsure` is never replaced: the same word is left as it is, another word is printed as DISAGREES
for a person to settle. The owner's words on a shot's frames are printed beside it, whole, with their line, and
are copied nowhere: the page reads them from its own file, and they count over a button, so a shot printed with
words is for a person to read. A run that wrote entries says so at the end of the record's `written_by`, after
whatever was there: when, from which lines, and on how many shots the owner's words were left unread. With no
`--preview` at all, the tiles of numbered pictures are counted and left for a run that names their previews.

**Nothing is written twice.** A tile whose answer already stands in the cast file, line for line, is listed as
already there. A tile whose answer differs is appended (the latest line stands, nothing is rewritten), and the
printout says what it replaced. One change the cast file refuses, and this tool passes the refusal on: a
subject with a confirmed box on that frame un-named by a later click (`cast_file.append`: a box is withdrawn
first, by a person, then answered). A shots file is written once, at the end, whole, and only when an entry
changed, and refused when the file is no longer the one this run read. `--dry-run` prints everything and writes nothing.

What it does not do: judge an answer, read the words, or seed a load (`cast_file.py seeds` does that, per load).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
WHO_ROWS = ("b20", "who")
#: The page's button texts and what each means to the cast file. A label is a subject of the job; None is
#: "none of these" (that numbered person is nobody the job names); "unsure" and "taken back" are states.
BUTTONS = {"the lead": "lead", "the second subject": "second", "the third subject": "third",
           "none of these": None, "not sure": "unsure", "taken back": "taken back"}
#: The table name the who page gives a tile of the source itself (`who_page.py::plain_tile`).
PLAIN = "plain"
#: The plain tile's buttons: the shots record's `present` for that frame, and the words the page shows on the button.
#: `not in shot` was the second button's label until 2026-10-12 14:45Z.
PLAIN_BUTTONS = {"not outlined": ("yes", "on this frame, not outlined"), "not on frame": ("no", "not on this frame"),
                 "not in shot": ("no", "not on this frame")}
SHOTS_SCHEMA = "h3_shots/1"
#: How an entry of the shots record says this tool wrote it; such an entry may be written again by a later run.
ENTERED = "entered by page_answers_to_cast"


def _cast_file():
    spec = importlib.util.spec_from_file_location("cast_file", HERE / "cast_file.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def previews(specs: list[str]) -> dict[str, dict]:
    """{table file name: {path, first, subjects, job name}} for every table of every `JOB.json=DIR` given."""
    out = {}
    for spec in specs:
        job_path, _, folder = spec.partition("=")
        if not folder:
            raise SystemExit(f"--preview {spec!r}: write JOB.json=TABLES_DIR")
        job = json.load(open(job_path))["job"]
        labels = [s["label"] for s in job["subjects"]]
        for label in labels:
            name = f"{job['name']}_{label}_shots_00001_shots.json"
            path = Path(folder) / name
            if not path.is_file():
                raise SystemExit(f"{spec}: no table {path}")
            out[name] = {"path": path, "first": int(job["first"]), "subjects": labels, "job": job["name"]}
    return out


def tiles(lines: list[tuple[int, dict]]) -> dict[tuple[str, int], dict]:
    """The owner's who-is-who lines grouped by tile: the latest button per person, and every words line, in order."""
    out: dict[tuple[str, int], dict] = defaultdict(lambda: {"buttons": {}, "by_subject": {}, "words": [], "lines": []})
    for n, d in lines:
        if d.get("who") != "owner" or d.get("row") not in WHO_ROWS or not d.get("table"):
            continue
        key = (str(d["table"]), int(d["source_frame"]))
        t = out[key]
        t["lines"].append(n)
        t["video"], t["shot_frames"] = d.get("video"), d.get("frames")      # the latest line's: what the page showed it under
        if d.get("what") == "words" and d.get("words"):
            t["words"].append((n, str(d["words"])))
        elif d.get("what") == "choice" and d.get("subject"):
            # the who page's lines (2026-10-12): one click names a SUBJECT's number, or says they are not outlined or not in the shot
            t["by_subject"][str(d["subject"])] = (n, str(d.get("chosen", "")), str(d.get("at", "")))
        elif d.get("what") == "choice" and str(d.get("person", "")).isdigit():
            t["buttons"][int(d["person"])] = (n, str(d.get("chosen", "")), str(d.get("at", "")))
    return dict(out)


def read_answers(path) -> list[tuple[int, dict]]:
    out = []
    for n, text in enumerate(Path(path).read_text().splitlines(), 1):
        if text.strip():
            out.append((n, json.loads(text)))
    return out


def plan(tile_key, tile, table_info, table: dict) -> dict:
    """What one tile says, as `from_tile` wants it, or why it cannot be stored."""
    name, frame = tile_key
    shot = next((s for s in table["shots"] if table_info["first"] + int(s["shown_frame"]) == frame), None)
    if shot is None:
        raise SystemExit(f"{name}: no shot shows source frame {frame} (first frame {table_info['first']})")
    # Both shapes of click can be on one tile (the first page answered by number, the who page by subject). The
    # latest click about a subject wins, whichever shape it came in; a number that two subjects still claim is refused.
    claims: dict[str, tuple[str, int, str]] = {}      # subject -> (at, number, chosen)
    unsure, none = [], []
    of_people, times = False, []                      # was any numbered PERSON answered for; every standing click's time
    for person, (n, chosen, at) in sorted(tile["buttons"].items()):
        if chosen not in BUTTONS:
            raise SystemExit(f"{name} frame {frame}: line {n} has a button this tool does not know: {chosen!r}")
        meaning = BUTTONS[chosen]
        if meaning == "taken back":
            continue
        of_people = True
        times.append(at)
        if meaning == "unsure":
            unsure.append(person)
        elif meaning is None:
            none.append(person)
        elif at >= claims.get(meaning, ("", 0, ""))[0]:
            claims[meaning] = (at, person, chosen)
    for subject, (n, chosen, at) in sorted(tile["by_subject"].items()):
        times.append(at)
        if chosen.isdigit():
            if at >= claims.get(subject, ("", 0, ""))[0]:
                claims[subject] = (at, int(chosen), chosen)
        elif chosen in ("not outlined", "not in shot", "not on frame"):
            if at >= claims.get(subject, ("", 0, ""))[0]:
                claims.pop(subject, None)          # the later word: they are none of the outlined people on this frame
                claims[subject + "\x00none"] = (at, 0, chosen)
        else:
            raise SystemExit(f"{name} frame {frame}: line {n} has a button this tool does not know: {chosen!r}")
    named = sorted((num, sub) for sub, (at, num, _c) in claims.items() if "\x00" not in sub)
    by_number: dict[int, list[str]] = {}
    for num, sub in named:
        by_number.setdefault(num, []).append(sub)
    clash = {k: v for k, v in by_number.items() if len(v) > 1}
    if clash:
        raise SystemExit(f"{name} frame {frame}: one number is claimed for two subjects by the owner's latest clicks: {clash}; "
                         f"a person settles it with cast_file.py withdraw and answer")
    clicked = sorted({sub.partition("\x00")[0] for sub in claims})
    # THE SUBJECTS A TILE'S LINES ARE WRITTEN FOR. Asked of the numbered people (the first page), naming them says of every
    # subject of the job not named that they are none of the people outlined: all the job's subjects. Asked of the
    # subjects one by one (the who page), a subject nobody clicked for was not answered, and gets no line: only the
    # subjects clicked. A tile holding both shapes is read as the first.
    answered = list(table_info["subjects"]) if of_people else [sub for sub in table_info["subjects"] if sub in clicked]
    return {"table": name, "frame": frame, "shot": int(shot["shot"]), "named": named, "none_of_these": none,
            "none_said": sorted((sub.partition("\x00")[0], chosen) for sub, (_at, _num, chosen) in claims.items() if "\x00" in sub),
            "not_the_jobs": [sub for sub in clicked if sub not in table_info["subjects"]],
            "answered": answered, "not_answered": [sub for sub in table_info["subjects"] if sub not in answered],
            "unsure": unsure, "words": tile["words"], "lines": tile["lines"], "last_at": max(times, default="")}


def shot_files(paths: list[str]) -> list[dict]:
    """Each `--shots` file: its path, its text as read, and the record."""
    out = []
    for path in paths:
        text = Path(path).read_text()
        record = json.loads(text)
        if record.get("schema") != SHOTS_SCHEMA:
            raise SystemExit(f"{path}: schema is {record.get('schema')!r}, not {SHOTS_SCHEMA!r}")
        out.append({"path": path, "text": text, "record": record})
    return out


def shot_holding(files: list[dict], video, frame: int, shown_under, where: str):
    """The (file, shot) a plain tile's frame is in, or None when no `--shots` file of that video spans it."""
    for f in files:
        rec = f["record"]
        if rec.get("video") != video or not rec["span"][0] <= frame <= rec["span"][1]:
            continue
        shot = next((s for s in rec["shots"] if s["first"] <= frame <= s["last"]), None)
        if shot is None or [shot["first"], shot["last"]] != list(shown_under or []):
            has = "no shot there" if shot is None else f"shot {shot['first']}-{shot['last']}"
            raise SystemExit(f"{where}: the page showed source frame {frame} under shot {shown_under}, and {f['path']} has {has}; "
                             "the table was cut again since the page was made, so the answer is of a shot this file does not have")
        return f, shot
    return None


def plain_plan(frame: int, tile: dict) -> dict:
    """What a plain tile says: for each subject clicked, on the frame or not, by the latest click."""
    said = {}
    for subject, (n, chosen, at) in sorted(tile["by_subject"].items()):
        if chosen not in PLAIN_BUTTONS:
            raise SystemExit(f"plain frame {frame}: line {n} has a button a plain frame does not have: {chosen!r}")
        present, label = PLAIN_BUTTONS[chosen]
        said[subject] = {"present": present, "label": label, "line": n, "at": at}
    return {"frame": frame, "said": said}


def shot_entries(shot: dict, plains: list[dict], standing: dict, name: str) -> dict:
    """A shot's `who` entries from every plain frame answered in it and every box that stands in the cast file inside it."""
    out = {}
    for subject in sorted({s for p in plains for s in p["said"]}):
        frames = sorted((p["frame"], p["said"][subject]) for p in plains if subject in p["said"])
        boxes = sorted(f for (who, f), line in standing.items()
                       if who == subject and shot["first"] <= f <= shot["last"] and line["box"] is not None)
        how = "; ".join(f"frame {f} (the source itself): {v['label']}" for f, v in frames)
        how += "".join(f"; frame {f}: a box in the cast file ({standing[(subject, f)]['table']})" for f in boxes)
        lines = [v["line"] for _, v in frames]
        # the frames themselves, as fields beside the prose (2026-10-13): a tool that needs a frame the subject was marked
        # ON reads `frames_on`, never a sentence of `how`
        out[subject] = {"present": "yes" if boxes or any(v["present"] == "yes" for _, v in frames) else "no", "how": how,
                        "frames_on": [f for f, v in frames if v["present"] == "yes"], "frames_boxed": boxes,
                        "by": f"owner, on the who page at {max(v['at'] for _, v in frames)} ({name} lines {min(lines)}-{max(lines)}), {ENTERED}"}
    return out


def write_whole(path: str, text: str, read_as: str) -> None:
    """Replace a shots file in one move, unless somebody wrote it since this run read it."""
    if Path(path).read_text() != read_as:
        raise SystemExit(f"{path} changed while this ran; nothing of it is written (the cast file's lines are in). Run again.")
    tmp = Path(str(path) + ".page_answers_to_cast.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def same(stored: dict | None, line: dict) -> bool:
    return bool(stored) and stored["answer"] == line["answer"] and stored["box"] == line["box"] and stored["table"] == line["table"]


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--answers", required=True)
    ap.add_argument("--cast", required=True)
    ap.add_argument("--preview", action="append", default=[], metavar="JOB.json=TABLES_DIR")
    ap.add_argument("--shots", action="append", default=[], metavar="SHOTS.json",
                    help="the video's shots record; a plain frame's answers are taken into it and into the cast file")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.preview and not a.shots:
        ap.error("give a --preview (the numbered pictures' tiles), a --shots (the plain frames'), or both")
    cf = _cast_file()
    tables = previews(a.preview)
    records = shot_files(a.shots)
    answers_name = Path(a.answers).name
    loaded = {name: json.loads(info["path"].read_text()) for name, info in tables.items()}
    answers = read_answers(a.answers)
    standing = cf.current(cf.read(a.cast))
    left_out = [(n, d.get("who")) for n, d in answers if d.get("row") in WHO_ROWS and d.get("who") != "owner"]
    if left_out:
        print(f"left out, not the owner's: {len(left_out)} line(s) by " + ", ".join(sorted({str(w) for _, w in left_out})))
    stored = waiting = already = refused = plain = taken = left = outside = 0
    left_tables: set[str] = set()
    by_shot: dict[tuple[int, int], dict] = {}        # (which --shots file, the shot's first frame) -> its plain frames
    grouped = tiles(answers)
    # EVERYTHING IS READ BEFORE ANYTHING IS WRITTEN. A tile that cannot be read (a table no preview holds, a frame no
    # shot shows, a button this tool does not know, a tile the cast file cannot take) stops the run in this loop, with
    # the cast file as it was. Until 2026-10-13 it stopped the run after the tiles before it had been appended.
    work = []
    for key in sorted(grouped, key=lambda k: (k[1], k[0])):
        tile = grouped[key]
        name, frame = key
        if name == PLAIN:
            # the who page's plain frame (nobody outlined, or a long shot's quarter): no table and no box
            head = f"plain frame {frame}, lines {tile['lines'][0]}-{tile['lines'][-1]}"
            work.append({"kind": "plain", "head": head, "pp": plain_plan(frame, tile),
                         "held": shot_holding(records, tile["video"], frame, tile["shot_frames"], head)})
            continue
        if not a.preview:
            work.append({"kind": "left", "name": name})
            continue
        if name not in tables:
            raise SystemExit(f"{name}: no --preview holds this table (line {tile['lines'][0]} of {a.answers})")
        p = plan(key, tile, tables[name], loaded[name])
        one = {"kind": "tile", "name": name, "p": p, "lines": None,
               "head": f"{name} shot {p['shot']} (source frame {frame}), lines {p['lines'][0]}-{p['lines'][-1]}",
               "words": "".join(f"\n    words (line {n}): {w.strip()}" for n, w in p["words"])}
        if not p["not_the_jobs"] and not p["unsure"] and (p["named"] or p["none_of_these"] or p["none_said"]):
            # an answer with no time on it is said so: until 2026-10-13 the line then read "on the page at  (", a hole
            when = f"at {p['last_at']}" if p["last_at"] else "at a time the answers do not give"
            by = f"owner, on the page {when} ({Path(a.answers).name} lines {p['lines'][0]}-{p['lines'][-1]}), entered by page_answers_to_cast"
            try:
                one["lines"] = cf.from_tile(loaded[name], name, tables[name]["first"], p["shot"], p["answered"], p["named"], by)
            except ValueError as e:
                raise SystemExit(f"{one['head']}: {e}")
        work.append(one)
    for one in work:
        if one["kind"] == "plain":
            pp, head, held = one["pp"], one["head"], one["held"]
            if held is None:
                plain += 1
                said = ", ".join(f"{sub}: {v['label']}" for sub, v in sorted(pp["said"].items()))
                print(f"PLAIN FRAME, no --shots file holds it, nothing stored: {head}: {said}")
                continue
            if not pp["said"]:
                continue
            f, shot = held
            taken += 1
            by_shot.setdefault((records.index(f), shot["first"]), {"file": f, "shot": shot, "plains": []})["plains"].append(pp)
            continue
        if one["kind"] == "left":
            left += 1
            left_tables.add(one["name"])
            continue
        name, p, head, words, lines = one["name"], one["p"], one["head"], one["words"], one["lines"]
        if p["not_the_jobs"]:
            # a tile is stored whole or not at all, so the clicks on it that do name the job's subjects wait with it
            outside += 1
            print(f"NOT STORED, a click names a subject this tile's job does not track: {head}: {', '.join(p['not_the_jobs'])}; "
                  f"the job ({tables[name]['job']}) tracks {', '.join(tables[name]['subjects'])}; nothing of the tile is stored{words}")
            continue
        if p["unsure"]:
            waiting += 1
            print(f"WAITING on the owner: {head}: `not sure` stands on person {p['unsure']}; nothing stored{words}")
            continue
        if lines is None:
            print(f"no button stands: {head}; nothing stored{words}")
            continue
        said = ", ".join([f"{n}={s}" for n, s in p["named"]]
                         + [f"{sub}: {PLAIN_BUTTONS[chosen][1]} (stored as `{cf.NONE_OUTLINED}`)" for sub, chosen in p["none_said"]]) or "nobody of ours"
        if p["none_of_these"]:
            said += f"; none of these: {p['none_of_these']}"
        if p["not_answered"]:
            said += f"; NOT ANSWERED, no line written: {', '.join(p['not_answered'])}"
        if all(same(standing.get((ln["subject"], ln["frame"])), ln) for ln in lines):
            already += 1
            print(f"already there: {head}: {said}{words}")
            continue
        replaced = [f"{ln['subject']} was {standing[(ln['subject'], ln['frame'])]['answer']}"
                    for ln in lines if (ln["subject"], ln["frame"]) in standing]
        if not a.dry_run:
            try:
                cf.append(a.cast, lines)
            except ValueError as e:
                refused += 1
                print(f"REFUSED by the cast file: {head}: {said}{words}\n    {e}")
                continue
            for ln in lines:
                standing[(ln["subject"], ln["frame"])] = ln
        print(f"{'would store' if a.dry_run else 'stored'}: {head}: {said}"
              + (f" (replaces: {'; '.join(replaced)})" if replaced else "") + words)
        stored += 1
    if left:
        print(f"left for a run that names their previews (no --preview was given): {left} tile(s) of numbered pictures, on "
              + ", ".join(sorted(left_tables)))

    # The shots record: each shot a plain frame was answered in, from every such frame and the boxes that stand in it.
    # Nothing of a plain frame is written to the cast file; `standing` is read for the boxes.
    word_lines = [(n, d) for n, d in answers if d.get("who") == "owner" and d.get("row") in WHO_ROWS and d.get("what") == "words"
                  and str(d.get("words", "")).strip() and str(d.get("source_frame", "")).lstrip("-").isdigit()]
    counts = {"yes": 0, "no": 0}
    wrote = same_word = disagrees = there = with_words = 0
    changed: dict[int, list[int]] = {}               # which --shots file -> the answers lines its new entries came from
    worded: dict[int, int] = {}                      # which --shots file -> its shots printed with the owner's words
    for (k, _first), item in sorted(by_shot.items()):
        f, shot = item["file"], item["shot"]
        words = [(n, str(d["words"])) for n, d in word_lines
                 if d.get("video") == f["record"].get("video") and shot["first"] <= int(d["source_frame"]) <= shot["last"]]
        entries = shot_entries(shot, item["plains"], standing, answers_name)
        head = f"{f['path']} shot {shot['first']}-{shot['last']}"
        for subject, new in sorted(entries.items()):
            if subject not in f["record"]["subjects"]:
                raise SystemExit(f"{head}: the owner answered for `{subject}`, and the record's subjects are {f['record']['subjects']}")
            old = shot["who"].get(subject) or {}
            if old == new:
                there += 1
            elif old.get("present", "unsure") != "unsure" and ENTERED not in str(old.get("by", "")):
                if old["present"] == new["present"]:
                    same_word += 1
                    print(f"left as it is, another hand's and the same word: {head}: {subject}: {old['present']} ({old.get('by', '')})")
                else:
                    disagrees += 1
                    print(f"DISAGREES, not written, for a person to settle: {head}: {subject} is `{old['present']}` in the record "
                          f"({old.get('how', '')}; {old.get('by', '')}) and `{new['present']}` by these answers ({new['how']})")
            else:
                wrote += 1
                counts[new["present"]] += 1
                changed.setdefault(k, []).extend(p["said"][subject]["line"] for p in item["plains"] if subject in p["said"])
                print(f"{'would write' if a.dry_run else 'written'}: {head}: {subject}: {new['present']}; {new['how']}"
                      + (f" (was {old['present']})" if old.get("present") else ""))
                shot["who"][subject] = new
        with_words += bool(words)
        worded[k] = worded.get(k, 0) + bool(words)
        for n, w in words:
            print(f"    words on {shot['first']}-{shot['last']} (line {n}): {w.strip()}")
    if not a.dry_run:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for k, lines in sorted(changed.items()):
            rec = records[k]["record"]
            did = (f"`present` entries written by bench/page_answers_to_cast.py at {now} from the owner's plain-frame answers in "
                   f"{a.answers} (lines {min(lines)}-{max(lines)}); the owner's words on {worded.get(k, 0)} shot(s) stay in that file and were "
                   "not read against the buttons")
            rec["written_by"] = f"{rec['written_by']}; {did}" if rec.get("written_by") else did
            # one space of indent and no last newline: how both of the first video's records are written (2026-10-12)
            write_whole(records[k]["path"], json.dumps(rec, indent=1), records[k]["text"])

    print(f"\n{'would store' if a.dry_run else 'stored'} {stored} tile(s); already there {already}; waiting on the owner {waiting}; "
          f"refused by the cast file {refused}; naming a subject their job does not track, not stored {outside}; "
          f"plain frames no --shots file holds {plain}; cast file {a.cast}")
    if records:
        print(f"shots record: {'would write' if a.dry_run else 'written'} {wrote} entr{'y' if wrote == 1 else 'ies'} "
              f"(yes {counts['yes']}, no {counts['no']}) from {taken} plain frame(s) in {len(by_shot)} shot(s); already there {there}; "
              f"another hand's with the same word, left {same_word}; DISAGREES, not written {disagrees}; shots printed with the owner's words, for a person to read {with_words}; "
              + ", ".join(r["path"] for r in records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
