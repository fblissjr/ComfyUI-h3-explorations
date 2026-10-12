#!/usr/bin/env python3
"""Who is who across shots: one row a shot of what each tracker called, whose word it is, and the question where it is nobody's.

After a preview with several named subjects, each Subject Track has written its own shot table. This reads the
tables of ONE load together and writes one file, a row a shot, for the gate before a render
(`bench/capture_masked_run.py`). It compares nobody with anybody and decides nothing:

- a shot is `answered` when every tracker's call on it is a PERSON'S WORD (a pick, a typed correction, a seed),
  or when nothing with a head was detected in it;
- otherwise it is `ask`: at least one call is the tracker's own, and the row carries, per such tracker, the
  question to put to a person in that tracker's tile's own numbers.

A tracker's own take is one vote with nothing to agree with. Until 2026-10-10 this tool also carried a sum over
all subjects at once and an "evidence total" with a zero for nobody, and a row agreed when three readings said
the same. They were the tracker's own likeness arranged twice more, not second and third votes: where they agreed
it was one clue agreeing with itself, a row could agree with a wrong take, and on five shots of one clip the two
totals failed in opposite directions. The owner cut them (`docs/wiki/decisions.md`, 2026-10-10, "What the owner
cut from the tools around the nodes"); what was measured with them is the record,
`bench/results/2026-10-10_who_is_who_across_shots.md`, and the code that made its tables is in git before this
commit. A second vote of ANOTHER kind (SAM's own memory seeded on a confirmed frame, the record's last
measurement) is not in this tool.

What a row can still say by itself, with no person: two trackers have ONE person on one tile frame (`ask`, even
when both are a person's word, since two masks are then on one person); a tracker took a detection in a shot
where nothing with a head was found; a tracker has its subject on no shot of the load (`not_picked`, the first
line a reader should see).

The tables must be version 2 (`shot_table.py::TABLE_VERSION`): every tracker lists everybody on a frame in one
numbering, so a person's number means the same on every tracker's tile of that frame. An older table is refused.

    <python> bench/who_is_who_across_shots.py corrections --table lead=<_shots.json> --table second=<_shots.json> ... \\
        [--first-frame N] [--frames COUNT] --out FILE

`--first-frame` is the frame of the source copy the load's frame 0 is on; it is written into the file for the
gate and, without `--frames`, changes nothing else.

**Tables of two previews** (2026-10-12: the first load of two subjects took each subject's track from its own
preview, one of 457 frames and one of 406 from the same first frame, and this tool refused them as not one
load). With `--frames COUNT` each table is cut to COUNT frames from `--first-frame`: its shots that overlap
them, clipped, in the cut's own frame numbers. A table that begins on another source frame says so as
`--table LABEL=PATH@FIRST`. Refused by name: a table that does not cover those frames, tables of two picture
sizes, and cuts INSIDE those frames that differ between tables (the rows are a shot of each table side by side,
so the shots must be the same ones). A tracker's tile frame outside the cut is kept and marked. What it cannot
know then: a person's number is the K-th detection on a tile's frame as THAT preview detected it, so the test
for two trackers on one person (one number on one tile frame) is only as good as the two previews' detections
of that frame being the same; the capture's own columns of how far two masks overlap are what show two masks on
one person. No model, no card.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ANSWERED, ASK = "answered", "ask"
#: The shot table version this reads: the one in which every tracker's `people` is everybody on the frame.
TABLES = 2


def headed(person: dict) -> bool:
    """Whether this detection counts as a person: everything but one the table says has NO head (`has_head` false).

    Unknown (null: a frame the tracker compared nobody on) counts as a person, so the shot asks and is never
    called "only things" on a guess. Every version 2 table carries the field (`shot_table.py::build`)."""
    return person.get("has_head") is not False


def word(shot: dict) -> str:
    """Whose word this tracker's call on this shot is, when it is somebody's: a pick, a typed correction, a seed."""
    return ("picked" if shot["subject"]["state"] == "picked" else "corrected by hand" if shot.get("corrected") else
            "seeded" if shot.get("seeded") else "")


def cut_to(label: str, table: dict, table_first: int, first: int, frames: int) -> dict:
    """A table cut to `frames` frames from source frame `first`: its shots that overlap them, clipped, renumbered from the cut's frame 0.

    `table_first` is the source frame the table's own frame 0 is on. A shot keeps its number in its own table
    (a correction is typed into that tracker by that number) and its tile's frame, marked when it lies outside
    the cut. A table that does not hold every frame asked for is refused."""
    lead = int(first) - int(table_first)
    if lead < 0 or lead + frames > int(table["frames"]):
        raise SystemExit(f"{label}'s table holds source frames {table_first}-{table_first + int(table['frames']) - 1} and the frames asked for "
                         f"are {first}-{first + frames - 1}: it does not cover them")
    shots = []
    for shot in table["shots"]:
        a, b = int(shot["first_frame"]) - lead, int(shot["last_frame"]) - lead
        if b < 0 or a >= frames:
            continue
        tile = int(shot["shown_frame"]) - lead
        shots.append({**shot, "first_frame": max(a, 0), "last_frame": min(b, frames - 1), "shown_frame": tile,
                      "tile_outside_the_frames": not 0 <= tile < frames})
    return {**table, "frames": int(frames), "cuts": [int(c) - lead for c in table["cuts"] if 0 < int(c) - lead < frames], "shots": shots}


def read_rows(tables: dict[str, dict], first_source_frame: int | None = None, frames: int | None = None,
              firsts: dict[str, int] | None = None) -> dict:
    """The file for the gate, from the trackers' tables of one load, {label: table} in the job's order.

    With `frames`, each table is first cut to that many frames from `first_source_frame` (`cut_to`); `firsts` is
    {label: the source frame its table's frame 0 is on} for a table that begins elsewhere."""
    labels = list(tables)
    for label, t in tables.items():
        if t.get("version") != TABLES:
            raise SystemExit(f"{label}: this shot table is version {t.get('version')!r}; this tool reads version {TABLES}, in which every "
                             "tracker lists everybody on a frame. Run the preview again")
    if frames is not None:
        if first_source_frame is None:
            raise SystemExit("--frames cuts each table to that many frames from --first-frame: give --first-frame")
        tables = {label: cut_to(label, t, (firsts or {}).get(label, first_source_frame), first_source_frame, int(frames)) for label, t in tables.items()}
    first = tables[labels[0]]
    for label, t in tables.items():
        if list(t.get("size", [])) != list(first.get("size", [])):
            raise SystemExit(f"{label} and {labels[0]} are tables of two picture sizes ({t.get('size')} and {first.get('size')}): "
                             "not tables of one load")
        if frames is not None and list(t["cuts"]) != list(first["cuts"]):
            at = lambda cuts: [first_source_frame + c for c in cuts]  # noqa: E731
            raise SystemExit(f"inside source frames {first_source_frame}-{first_source_frame + frames - 1} the cuts differ: {labels[0]}'s table "
                             f"has them at {at(first['cuts'])} and {label}'s at {at(t['cuts'])} (source frames). A row is one shot of each "
                             "table side by side, so the shots must be the same: run the previews on the same cuts")
        if t["frames"] != first["frames"] or list(t["cuts"]) != list(first["cuts"]):
            raise SystemExit(f"{label} and {labels[0]} are not tables of one load (frames, size or cuts differ): each tracker "
                             "must be given the same frames. For tables of two previews that both hold the load's frames, give "
                             "--frames COUNT with --first-frame: each is cut to them")
    # a tracker with its subject on no shot is a finding and the first line, not a reason to stop (the first
    # three-tracker run, 2026-10-10: an earlier tracker's wrong take held a later subject on her own pick frame)
    not_picked = {}
    for label, t in tables.items():
        if not any(shot["subject"]["person"] is not None for shot in t["shots"]):
            held = sum(1 for shot in t["shots"] for p in shot["people"] if p.get("held_by_another_tracker"))
            not_picked[label] = (f"{label}'s tracker has its subject on no shot of this load"
                                 + (f"; {held} of the people on its tiles are held by a tracker that runs before it, and its subject "
                                    "may be one of them: a typed correction may name a held person" if held else ""))
    rows = []
    for n in range(len(first["shots"])):
        number = first["shots"][n]["shot"]
        cells, everybody = {}, []
        for label in labels:
            shot = tables[label]["shots"][n]
            everybody += list(shot["people"]) + [p for look in shot.get("looks", []) for p in look["people"]]
            cells[label] = {"called": shot["subject"]["state"], "called_person": shot["subject"]["person"], "answered_by": word(shot),
                            "typed_in_this_table": str(shot.get("corrected", "") or ""), "tile_frame": int(shot["shown_frame"]),
                            "numbers_on_its_tile": [p["person"] for p in shot["people"]], "ask": ""}
            if frames is not None:                    # a cut table: the shot's number in its own table, and a tile outside the cut
                cells[label].update(shot_in_its_table=int(shot["shot"]), tile_outside_the_frames=bool(shot["tile_outside_the_frames"]))
        # one numbering on every tracker's tile of a frame: the same number on the same frame is the same person
        same = [(a, b) for i, a in enumerate(labels) for b in labels[i + 1:]
                if cells[a]["called_person"] is not None and cells[a]["called_person"] == cells[b]["called_person"]
                and cells[a]["tile_frame"] == cells[b]["tile_frame"]]
        nobody = not everybody
        only_things = bool(everybody) and not any(headed(p) for p in everybody)
        unanswered = [label for label, c in cells.items() if not c["answered_by"]]
        status, why, note, asked = ANSWERED, "", "", []
        if same:
            # the pair, and with them every tracker whose call is nobody's word: one round of questions, not two
            status, asked = ASK, sorted({x for pair in same for x in pair} | set(unanswered), key=labels.index)
            others = [label for label in unanswered if not any(label in pair for pair in same)]
            why = (f"{same[0][0]} and {same[0][1]} have one person on one frame: two trackers on one person"
                   + ("" if not others else "; and nobody has said who " + ", ".join(others) + (" is" if len(others) == 1 else " are") + " in this shot"))
        elif nobody or only_things:
            note = ("no table lists anybody in this shot" if nobody else "every detection in this shot has no head: things, not people")
            asked = [label for label in unanswered if cells[label]["called_person"] is not None]
            if asked:
                status, why = ASK, f"{', '.join(asked)} took a detection in a shot where nothing with a head was found"
        elif unanswered:
            status, asked = ASK, unanswered
            why = "nobody has said who " + ", ".join(unanswered) + (" is" if len(unanswered) == 1 else " are") + " in this shot"
        for label in asked:
            c = cells[label]
            numbers = ", ".join(str(k) for k in c["numbers_on_its_tile"]) or "nobody is numbered"
            took = "nobody" if c["called_person"] is None else f"person {c['called_person']}"
            # a reader may show only the questions: the clash is said in each of the pair's own, not in `why` alone
            clash = next((f"TWO TRACKERS ON ONE PERSON: {a} and {b} both have person {c['called_person']} on frame {c['tile_frame']}. "
                          for a, b in same if label in (a, b)), "")
            own = c.get("shot_in_its_table", number)      # a correction is typed into a tracker by the shot's number in ITS table
            c["ask"] = (clash + f"shot {own}, {label}'s tile (frame {c['tile_frame']}): which numbered person is {label}? {numbers}, or none. "
                        f"Its tracker has {took} ({c['called']}). Type `shot {own}: person K` or `shot {own}: none` into {label}'s tracker")
        answered = {label: c["answered_by"] for label, c in cells.items() if c["answered_by"]}
        if answered and not note:
            note = "a person's word: " + ", ".join(f"{label} ({how})" for label, how in answered.items())
        rows.append({"shot": number, "frames": [first["shots"][n]["first_frame"], first["shots"][n]["last_frame"]],
                     "status": status, "why": why, "note": note, "trackers": cells})
    return {"what": "one row a shot of one load: what each tracker called, whose word it is, and the question where it is nobody's; "
                    "a file for the gate, not a job",
            "status_is": f"{ANSWERED}: every tracker's call on the shot is a person's word (a pick, a typed correction, a seed), or nothing "
                         f"with a head was detected in it. {ASK}: at least one call is the tracker's own, or two trackers have one "
                         "person, and a person is asked. A row is answered once its corrections are typed and the preview run again",
            "frames_are": "the load's own frame numbers" + ("" if first_source_frame is None else f"; the load's first frame is the source's {int(first_source_frame)}, as given"),
            "first_source_frame": first_source_frame, "not_picked": not_picked, "tables": labels, "rows": rows,
            **({} if frames is None else {
                "cut_to": {"frames": int(frames), "from_source_frame": int(first_source_frame),
                           "why": "the tables were of different frames (two previews), and each was cut to these"},
                "across_previews": "a person's number is the K-th detection on a tile's frame as that table's own preview detected it: "
                                   "one number on one tile frame in two tables is the same person only if the two previews detected "
                                   "that frame alike. Two masks on one person is shown by the capture's own mask-overlap columns"})}


def render(out: dict) -> None:
    for why in out["not_picked"].values():
        print(f"FIRST: {why}")
    print(f"{out['what']}\n{out['status_is']}\n{out['frames_are']}\n")
    for row in out["rows"]:
        cells = "; ".join(f"{label}: {c['called']}" + ("" if c["called_person"] is None else f" (person {c['called_person']})")
                          + (f" [{c['answered_by']}]" if c["answered_by"] else "") for label, c in row["trackers"].items())
        print(f"shot {row['shot']} ({row['frames'][0]}-{row['frames'][1]}): {row['status']}. {cells}. {row['why'] or row['note']}")
        for c in row["trackers"].values():
            if c["ask"]:
                print(f"    ASK: {c['ask']}")
    asks = sum(1 for r in out["rows"] if r["status"] == ASK)
    print(f"\n{asks} of {len(out['rows'])} shot(s) ask; {sum(1 for r in out['rows'] for c in r['trackers'].values() if c['ask'])} question(s)")


def main() -> int:
    p = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corrections", help="the rows for one load, from its trackers' shot tables")
    c.add_argument("--table", action="append", required=True, metavar="LABEL=PATH[@FIRST]",
                   help="one a tracker, in the job's order; @FIRST when its table begins on another source frame than --first-frame")
    c.add_argument("--first-frame", type=int, help="the source-copy frame the load's frame 0 is on")
    c.add_argument("--frames", type=int, help="cut every table to this many frames from --first-frame: for tables of two previews")
    c.add_argument("--out", required=True)
    a = p.parse_args()
    tables, firsts, names = {}, {}, {}
    for pair in a.table:
        label, _, path = pair.partition("=")
        path, at, begins = path.rpartition("@") if path.rpartition("@")[2].isdigit() else (path, "", "")
        if not label or not path or label in tables:
            raise SystemExit(f"--table takes LABEL=PATH or LABEL=PATH@FIRST, each label once: `{pair}`")
        tables[label], names[label] = json.loads(Path(path).read_text()), Path(path).name
        if at:
            firsts[label] = int(begins)
    if firsts and a.frames is None:
        raise SystemExit("--table LABEL=PATH@FIRST says where a table begins so that it can be cut: give --frames too")
    out = read_rows(tables, a.first_frame, a.frames, firsts)
    out["files"] = names
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    render(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
