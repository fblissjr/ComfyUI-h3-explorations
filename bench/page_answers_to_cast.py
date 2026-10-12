#!/usr/bin/env python3
"""The owner's answers on the review page's numbered pictures, taken into the video's cast file through `bench/cast_file.py`.

    <python> bench/page_answers_to_cast.py --answers data/<stem>/poc/answers.jsonl --cast data/<stem>/cast/cast.jsonl \\
        --preview JOB.json=TABLES_DIR [--preview ...] [--dry-run]

**What it buys.** The page saves one raw line a click (`answers.jsonl`): a table's file name, the frame of the
source copy the tile shows, a person's number and a button. The owner's rule (2026-10-11, Open List row g01):
those lines are raw input, saved twice, changed a second later or answered two ways, and a session turns them
into the record's structured data and asks where two disagree. On 2026-10-11 eighty-one such lines were saved
and none reached the cast file, so the next preview was seeded by corrections typed by hand. This is the one
way a line gets from the page into the cast, and it goes through `cast_file.from_tile`, whose rules are the
cast's: a tile is stored whole or not at all, a number becomes a box, a subject not named is `none of the
outlined people`, the latest line stands.

**How a line is read.** Only lines whose `who` is `owner` and whose `row` is a who-is-who row (`WHO_ROWS`) are
taken. Two shapes of click: the first page's (a numbered PERSON and a button naming who they are, `BUTTONS`), and
the who page's of 2026-10-12 (a SUBJECT and a button: their number on the tile, `not outlined`, or `not in
shot`; the last two both store as `none of the outlined people`, the cast's one word, and `seeds` decides per
load how strongly to read it). Otherwise: a session's test clicks are listed as left out. Lines are grouped by tile (the table's file name and the
source frame it shows); on a tile, the LATEST click for each numbered person is the answer (`taken back` leaves
that person unanswered). A button is read through `BUTTONS` to a subject's label or to `none of these`; a `not
sure` that stands on any person means the tile is NOT stored and is printed as waiting on the owner, since
storing it would write `none of the outlined people` for a subject the owner had not decided on. The owner's
typed words on the tile are printed beside it, whole, for whoever runs this to read: words count over a button
(the owner, 2026-10-11), and this tool does not read them. A tile whose words and buttons disagree is for a
person to settle, by a `withdraw` and an `answer` on `cast_file.py`.

**Which table and which shot.** A `--preview` names a preview's job file and the folder its shot tables are in:
the job gives the subjects every tile of that preview asks about and the source frame its table frame 0 is on
(`job.first`); the tables are `<job name>_<label>_shots_00001_shots.json` in the folder. A line's tile is the
shot of its table whose shown frame is the line's source frame; a line naming a table no preview holds, or a
frame no shot shows, is refused by name.

**Nothing is written twice.** A tile whose answer already stands in the cast file, line for line, is listed as
already there. A tile whose answer differs is appended (the latest line stands, nothing is rewritten), and the
printout says what it replaced. One change the cast file refuses, and this tool passes the refusal on: a
subject with a confirmed box on that frame un-named by a later click (`cast_file.append`: a box is withdrawn
first, by a person, then answered). `--dry-run` prints everything and writes nothing.

What it does not do: judge an answer, read the words, or seed a load (`cast_file.py seeds` does that, per load).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
WHO_ROWS = ("b20", "who")
#: The page's button texts and what each means to the cast file. A label is a subject of the job; None is
#: "none of these" (that numbered person is nobody the job names); "unsure" and "taken back" are states.
BUTTONS = {"the lead": "lead", "the second subject": "second", "the third subject": "third",
           "none of these": None, "not sure": "unsure", "taken back": "taken back"}


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
    named, unsure, none = [], [], []
    for person, (n, chosen, _at) in sorted(tile["buttons"].items()):
        if chosen not in BUTTONS:
            raise SystemExit(f"{name} frame {frame}: line {n} has a button this tool does not know: {chosen!r}")
        meaning = BUTTONS[chosen]
        if meaning == "taken back":
            continue
        if meaning == "unsure":
            unsure.append(person)
        elif meaning is None:
            none.append(person)
        else:
            named.append((person, meaning))
    for subject, (n, chosen, _at) in sorted(tile["by_subject"].items()):
        if chosen.isdigit():
            named.append((int(chosen), subject))
        elif chosen not in ("not outlined", "not in shot"):
            raise SystemExit(f"{name} frame {frame}: line {n} has a button this tool does not know: {chosen!r}")
        # `not outlined` and `not in shot` both leave the subject unnamed: the cast file stores `none of the outlined
        # people` for them (its one word for it); `not in shot` is the strong reading, which `seeds` takes per load
    return {"table": name, "frame": frame, "shot": int(shot["shot"]), "named": named, "none_of_these": none,
            "unsure": unsure, "words": tile["words"], "lines": tile["lines"],
            "last_at": max((at for _, _, at in tile["buttons"].values()), default="")}


def same(stored: dict | None, line: dict) -> bool:
    return bool(stored) and stored["answer"] == line["answer"] and stored["box"] == line["box"] and stored["table"] == line["table"]


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--answers", required=True)
    ap.add_argument("--cast", required=True)
    ap.add_argument("--preview", action="append", required=True, metavar="JOB.json=TABLES_DIR")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cf = _cast_file()
    tables = previews(a.preview)
    loaded = {name: json.loads(info["path"].read_text()) for name, info in tables.items()}
    answers = read_answers(a.answers)
    standing = cf.current(cf.read(a.cast))
    left_out = [(n, d.get("who")) for n, d in answers if d.get("row") in WHO_ROWS and d.get("who") != "owner"]
    if left_out:
        print(f"left out, not the owner's: {len(left_out)} line(s) by " + ", ".join(sorted({str(w) for _, w in left_out})))
    stored = waiting = already = refused = 0
    grouped = tiles(answers)
    for key in sorted(grouped, key=lambda k: (k[1], k[0])):
        tile = grouped[key]
        name, frame = key
        if name not in tables:
            raise SystemExit(f"{name}: no --preview holds this table (line {tile['lines'][0]} of {a.answers})")
        p = plan(key, tile, tables[name], loaded[name])
        head = f"{name} shot {p['shot']} (source frame {frame}), lines {p['lines'][0]}-{p['lines'][-1]}"
        words = "".join(f"\n    words (line {n}): {w.strip()}" for n, w in p["words"])
        if p["unsure"]:
            waiting += 1
            print(f"WAITING on the owner: {head}: `not sure` stands on person {p['unsure']}; nothing stored{words}")
            continue
        if not p["named"] and not p["none_of_these"]:
            print(f"no button stands: {head}; nothing stored{words}")
            continue
        by = f"owner, on the page at {p['last_at']} ({Path(a.answers).name} lines {p['lines'][0]}-{p['lines'][-1]}), entered by page_answers_to_cast"
        try:
            lines = cf.from_tile(loaded[name], name, tables[name]["first"], p["shot"], tables[name]["subjects"], p["named"], by)
        except ValueError as e:
            raise SystemExit(f"{head}: {e}")
        said = ", ".join(f"{n}={s}" for n, s in p["named"]) or "nobody of ours"
        if p["none_of_these"]:
            said += f"; none of these: {p['none_of_these']}"
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
    print(f"\n{'would store' if a.dry_run else 'stored'} {stored} tile(s); already there {already}; waiting on the owner {waiting}; "
          f"refused by the cast file {refused}; cast file {a.cast}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
