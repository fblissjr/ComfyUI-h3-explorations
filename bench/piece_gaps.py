#!/usr/bin/env python3
"""The frames no generated piece covers, sorted and loud; then the frames a piece covers and leaves as the original; then a subject in a shot with no piece of their own.

    <python> bench/piece_gaps.py --table data/<...>/t_<...>.json [--plan data/<stem>/plan.json] [--ledger LEDGER.json]
        [--record CLIP.mp4.check.json] [--up-to FRAME] [--md OUT.md] [--json OUT.json]

**What it buys.** The owner watched a joined clip and saw a person turn back into the original, and asked for
the plain thing (2026-10-13): THE FRAMES NO GENERATED CLIP COVERS, read from what is ledgered of each render,
sorted, loud, and printed by itself whenever a session does the common thing it does to find out what to build
next. `bench/delivery_table.py` prints `loud()` at the end of every table. Nothing here is typed and nothing is
measured on a picture.

**A, the frames inside no standing piece.** Every frame from the table's first to its last frame with any piece
(`--up-to` moves the end) that no row of the table lays a piece on, as runs sorted by frame. Who is in the shot is
the SECOND column, not the definition: for each record shot a run crosses, the subjects the shots record has in
it (the owner's answers, a shot at a time and never a frame), `unsure` said as unsure, and "nobody of ours" where
it has none. Beside each run: the piece before it and the piece after it, each with the day it was rendered on
the owner's clock.

**B, the frames a piece covers and leaves as the original.** Inside a standing piece's own frames, the frames on
which its capture emptied the subject's part, so the node was given nothing to replace there. Read from the
ledger through the table's reading (`leaves_as_the_original`: the capture's manifest); in no ledger before
2026-10-13. With `--ledger`, the load's own words where its job file gives a reason for frames it left. With
`--record`, the assembler's record of a clip built from the table: its `piece_changes_nothing` flags are held to
the ledger's frames, and where the two disagree both are printed. EVERY ROW SAYS WHAT IT WAS READ FROM, and a
piece neither says anything of is NOT KNOWN, never none: a table whose ledger predates the field, with no record
handed in, prints that B is not known (the tool's first real run printed "0 frames" beside a file that listed
six runs from the record, one of them the stretch the owner had seen).

**C, a subject in the shot with no piece of their own, on frames another's piece covers.** Not in A, since a
piece is laid there; the subject is the source's own on those frames all the same.

**D, present with nothing to make under this plan.** The frames of a shot whose every subject in it has, by a
person's look in the shots record (`seen_looks`, `bench/shots_look.py`; the table carries it as `nothing_to_make`),
none of the plan's parts on any frame. Not a gap, so not in A or C: its own list, with the look whole (the owner's
word of 2026-10-14: a later plan of another part re-derives its loads from the same records).

**A recorded look is said, not hidden.** Where the plan's `departures` hold a `NO LOAD` note for a subject over a
run's frames (somebody looked and found no face to replace), the row says so with the note: it stays in the list.

**What it cannot see.** The picture: whether a face shows on a frame, and whether a frame a piece replaces looks
right. A subject the shots record does not name. A frame the capture did not empty and the node left alone for a
reason of its own. A piece that stands but should not: the table's rule is the table's.

No verdict, no color: frames, runs, whose, and the reason on record.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
#: The source's frames a second, for a run's place on the clip's clock. The release's own rate (24000/1001); a clip
#: at another rate wants its own, which no table carries today.
RATE = 24000 / 1001
#: The owner's clock, for the day a piece was rendered (the owner's word of 2026-10-13: a record keeps its UTC
#: stamp; a day said to them is a day on their clock).
OWNERS_CLOCK = ZoneInfo("America/Chicago")
CANNOT = ("the picture: whether a face shows on a frame, and whether a replaced frame looks right; a subject the shots record does not name; a frame the "
          "capture did not empty and the node left alone; a piece that stands and should not")


def clock(frame: int) -> str:
    s = frame / RATE
    return f"{int(s // 60)}:{s % 60:04.1f}"


def runs(frames) -> list[list[int]]:
    out: list[list[int]] = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def day(written: str | None) -> str | None:
    """The day a render was written, on the owner's clock, as month/day."""
    if not written:
        return None
    return datetime.datetime.fromisoformat(written.replace("Z", "+00:00")).astimezone(OWNERS_CLOCK).strftime("%m/%d")


def gaps_of(table: dict, plan: dict | None = None, ledger: dict | None = None, record: dict | None = None, up_to: int | None = None, repo: Path | None = None) -> dict:
    """The three lists, from a delivery table's data (`reading` and `rows`)."""
    reading = table["reading"]
    # the record's shots as the table has them: a reading entry for a subject in a shot (or one the record is unsure of), and a
    # "not in the shot" line for a subject it says is not; a shot with nobody of ours has only the second kind
    who = {}                                                         # shot -> {subject: present}
    for e in table.get("not_in_the_shot") or []:
        who.setdefault((e["first"], e["last"]), {})[e["subject"]] = "no"
    nothing = {}                                                     # shot -> {subject: a person's lines that nothing is to be made}
    for e in reading:
        who.setdefault((e["first"], e["last"]), {})[e["subject"]] = e["present"]
        if e.get("nothing_to_make"):
            nothing.setdefault((e["first"], e["last"]), {})[e["subject"]] = e["nothing_to_make"]
    shots = sorted(who)
    # D: the frames of a shot whose every subject in it has nothing to make under this plan, by a person's look: not a gap
    # (the owner's word and the lead's ruling, 2026-10-14), its own list
    nothing_frames = {x for (lo, hi), here in who.items()
                      if [k for k, v in here.items() if v == "yes"] and not [k for k, v in here.items() if v == "unsure"]
                      and all(k in nothing.get((lo, hi), {}) for k, v in here.items() if v == "yes") for x in range(lo, hi + 1)}
    pieces = sorted(({"name": s["render"], "subject": e["subject"], "first": a, "last": b, "written": s.get("written"), "day": day(s.get("written"))}
                     for e in reading for s in e["stands"] for a, b in s["frames"]), key=lambda p: (p["first"], p["last"]))
    covered = {x for p in pieces for x in range(p["first"], p["last"] + 1)}
    first = shots[0][0] if shots else 0
    last_piece = max(covered) if covered else None
    end = up_to if up_to is not None else last_piece
    departures = [d for d in (plan or {}).get("departures", []) if str(d.get("note", "")).startswith("NO LOAD")]

    def looked(subject: str, a: int, b: int) -> str | None:
        d = next((d for d in departures if d.get("subject") == subject and d["first"] <= a and b <= d["last"]), None)
        return None if d is None else " ".join(str(d["note"]).split())

    def either_side(a: int, b: int, subject: str | None = None) -> dict:
        mine = [p for p in pieces if subject is None or p["subject"] == subject]
        before = max((p for p in mine if p["last"] < a), key=lambda p: p["last"], default=None)
        after = min((p for p in mine if p["first"] > b), key=lambda p: p["first"], default=None)
        return {"the_piece_before": before and {k: before[k] for k in ("name", "subject", "last", "day", "written")},
                "the_piece_after": after and {k: after[k] for k in ("name", "subject", "first", "day", "written")}}

    # A: inside no standing piece, as plain runs; whose each stretch is, a record shot at a time
    part_a = []
    if end is not None:
        for a, b in runs(x for x in range(first, end + 1) if x not in covered and x not in nothing_frames):
            crossed = []
            for lo, hi in shots:
                if lo <= b and hi >= a:
                    x, y = max(a, lo), min(b, hi)
                    here = who[(lo, hi)]
                    names = [k for k, v in here.items() if v == "yes"]
                    unsure = [k for k, v in here.items() if v == "unsure"]
                    crossed.append({"frames": [x, y], "shot": [lo, hi], "in_the_shot": names, "unsure": unsure, "nobody_of_ours": not names and not unsure,
                                    "a_recorded_look": {k: looked(k, x, y) for k in names if looked(k, x, y)}})
            part_a.append({"first": a, "last": b, "frames": b - a + 1, "clip": f"{clock(a)}-{clock(b + 1)}", "by_shot": crossed, **either_side(a, b)})
    part_d = []
    if end is not None:
        for a, b in runs(x for x in range(first, end + 1) if x not in covered and x in nothing_frames):
            part_d.append({"first": a, "last": b, "frames": b - a + 1, "clip": f"{clock(a)}-{clock(b + 1)}",
                           "by_shot": [{"frames": [max(a, lo), min(b, hi)], "shot": [lo, hi], "looks": nothing[(lo, hi)]}
                                       for lo, hi in shots if lo <= b and hi >= a and (lo, hi) in nothing]})
    # B: inside a piece's own frames, what its load left as the original
    by_name = {r["name"]: r for r in (ledger or {}).get("renders", {}).values()}
    flagged = {}
    for f in (record or {}).get("flags", []):
        if f.get("rule") == "piece_changes_nothing":
            flagged.setdefault(f["run"], set()).update(x for a, b in f["source_frames"] for x in range(a, b + 1))

    def the_loads_words(name: str, a: int, b: int) -> str | None:
        rec = ((by_name.get(name) or {}).get("records") or [{}])[-1]
        job_file = rec.get("job_file")
        if not job_file:
            return None
        path = Path(job_file) if Path(job_file).is_absolute() else (repo or HERE.parent) / job_file
        try:
            left = (json.loads(path.read_text()).get("job") or {}).get("left_out") or []
        except (OSError, ValueError):
            return None
        said = [" ".join(str(x.get("why", "")).split()) for x in left if x["first"] <= b and a <= x["first"] + x["frames"] - 1]
        return " | ".join(said) or None
    part_b, not_known = [], []
    for e in reading:
        for s in e["stands"]:
            mine = {x for a, b in s["frames"] for x in range(a, b + 1)}
            if s.get("leaves_as_the_original") is None and record is None:
                not_known.append(s["render"])              # the ledger does not say and no record was handed in
            ledgered = {x for a, b in s.get("leaves_as_the_original") or [] for x in range(a, b + 1)}
            seen = flagged.get(s["render"], set()) & mine
            for a, b in runs(ledgered | seen):
                span = set(range(a, b + 1))
                row = {"first": a, "last": b, "frames": b - a + 1, "clip": f"{clock(a)}-{clock(b + 1)}", "piece": s["render"], "subject": e["subject"],
                       "shot": [e["first"], e["last"]], "day": day(s.get("written")), "written": s.get("written"),
                       "read_from": ("the ledger (its capture's record) and the assembler's record" if span <= ledgered and span <= seen else
                                     "the ledger: its capture's record of where it emptied the part" if span <= ledgered else
                                     "the assembler's record alone (the piece is the original there); the ledger does not hold these frames"
                                     + (", and predates the field" if s.get("leaves_as_the_original") is None else "") if not span & ledgered else
                                     "the ledger for some of these frames and the assembler's record for others: see the difference"),
                       "why": ("its capture emptied this subject's part on these frames, so the node was given nothing to replace there" if span & ledgered else
                               "the assembler found the piece to be the original on these frames; why is in no file read here"),
                       "the_loads_own_words": the_loads_words(s["render"], a, b) if ledger else None}
                if record is not None:
                    row["against_the_assemblers_record"] = ("the same frames" if span <= ledgered and span <= seen else
                                                           {"in_the_ledger_and_not_the_record": runs(span & (ledgered - seen)), "in_the_record_and_not_the_ledger": runs(span & (seen - ledgered))})
                part_b.append(row)
    # C: a subject in the shot with no piece of their own, on frames another's piece covers
    part_c = []
    for e in reading:
        if e["present"] not in ("yes", "unsure") or e.get("nothing_to_make"):
            continue
        taken = {x for s in e["stands"] for a, b in s["frames"] for x in range(a, b + 1)}
        for a, b in runs(x for x in range(e["first"], e["last"] + 1) if x not in taken and x in covered and (end is None or x <= end)):
            under = sorted({p["name"] for p in pieces if p["first"] <= b and p["last"] >= a and p["subject"] != e["subject"]})
            part_c.append({"first": a, "last": b, "frames": b - a + 1, "clip": f"{clock(a)}-{clock(b + 1)}", "subject": e["subject"], "present": e["present"],
                           "shot": [e["first"], e["last"]], "the_pieces_laid_there": under, "a_recorded_look": looked(e["subject"], a, b), **either_side(a, b, e["subject"])})
    for part in (part_b, part_c):
        part.sort(key=lambda r: (r["first"], r["last"], r["subject"]))
    past = None
    if last_piece is not None and shots and shots[-1][1] > (end if end is not None else last_piece):
        past = {"from_frame": (end if end is not None else last_piece) + 1, "to_frame": shots[-1][1]}
    return {"up_to": end, "up_to_was_given": up_to is not None, "the_last_frame_with_any_piece": last_piece, "first": first, "pieces": len(pieces),
            "pieces_named": len({p["name"] for p in pieces}),
            "no_piece": part_a, "nothing_to_make": part_d, "left_by_a_piece": part_b, "left_by_a_piece_not_known_for": sorted(set(not_known)), "another_subject_with_none": part_c, "nothing_is_made_past": past, "cannot_see": CANNOT}


def _whose(seg: dict) -> str:
    if seg["nobody_of_ours"]:
        return "nobody of ours"
    looked = seg.get("a_recorded_look") or {}
    names = [k + (" (a recorded look: no load)" if k in looked else "") for k in seg["in_the_shot"]] + [f"{k} (the record is unsure)" for k in seg["unsure"]]
    return ", ".join(names)


def loud(table: dict, plan: dict | None = None, up_to: int | None = None, ledger: dict | None = None, record: dict | None = None) -> list[str]:
    """The lines a table prints at its end, under capitals: the three lists, short. The same lists as the files: what
    is handed to `gaps_of` for the files is handed here."""
    g = gaps_of(table, plan, ledger, record, up_to)
    if g["up_to"] is None:
        return ["NO PIECE STANDS IN THIS TABLE: every frame of it is the original."]
    a, b, c = g["no_piece"], g["left_by_a_piece"], g["another_subject_with_none"]
    lines = [f"FRAMES NO GENERATED PIECE COVERS, frames {g['first']} to {g['up_to']} "
             + (f"(as asked; the last frame with any piece is {g['the_last_frame_with_any_piece']})" if g["up_to_was_given"] else "(the last frame with any piece)") + ": "
             f"{sum(r['frames'] for r in a)} FRAME(S) IN {len(a)} RUN(S). Sorted; whose each is, a shot at a time, from the shots record:"]
    for r in a:
        by = "; ".join((f"{s['frames'][0]}-{s['frames'][1]}: " if len(r["by_shot"]) > 1 else "") + _whose(s) for s in r["by_shot"])
        lines.append(f"  {r['first']}-{r['last']}  ({r['frames']} frame(s), {r['clip']})  {by}")
    d = g["nothing_to_make"]
    if d:
        lines.append(f"PRESENT, NOTHING TO MAKE UNDER THIS PLAN, by a person's look (not a gap): {sum(r['frames'] for r in d)} FRAME(S) IN {len(d)} RUN(S):")
        lines += [f"  {r['first']}-{r['last']}  ({r['frames']} frame(s), {r['clip']})  "
                  + "; ".join(f"{k}: {' | '.join(v)}" for s in r["by_shot"] for k, v in s["looks"].items()) for r in d]
    unknown = g["left_by_a_piece_not_known_for"]
    if unknown and len(unknown) == g["pieces_named"]:
        lines.append("FRAMES A PIECE COVERS AND LEAVES AS THE ORIGINAL: NOT KNOWN. This table's ledger does not say it of any piece (it predates the field) and no "
                     "assembler's record was handed in. NOT KNOWN IS NOT NONE: make the ledger and the table again, or hand in a clip's record (--record).")
    else:
        lines.append(f"FRAMES A PIECE COVERS AND LEAVES AS THE ORIGINAL: {sum(r['frames'] for r in b)} FRAME(S) IN {len(b)} RUN(S)"
                     + (f"; NOT KNOWN for {len(unknown)} piece(s) whose capture does not say: {', '.join(unknown)}" if unknown else "") + ":")
        lines += [f"  {r['first']}-{r['last']}  ({r['frames']} frame(s), {r['clip']})  {r['subject']}, inside {r['piece']} (rendered {r['day']}); read from {r['read_from']}" for r in b]
    lines.append(f"A SUBJECT IN THE SHOT WITH NO PIECE OF THEIR OWN, on frames another's piece covers: {sum(r['frames'] for r in c)} FRAME(S) IN {len(c)} RUN(S):")
    lines += [f"  {r['first']}-{r['last']}  ({r['frames']} frame(s), {r['clip']})  {r['subject']}" + (" (the record is unsure)" if r["present"] == "unsure" else "")
              + (" (a recorded look: no load)" if r["a_recorded_look"] else "") + f", under {', '.join(r['the_pieces_laid_there'])}" for r in c]
    unsure = [r for r in c if r["present"] == "unsure"]
    if unsure:
        lines.append(f"  OF THESE, {sum(r['frames'] for r in unsure)} frame(s) in {len(unsure)} run(s) are of a subject THE RECORD IS UNSURE OF in the shot: the owner's to "
                     "answer on the who page before anything is built there.")
    if g["nothing_is_made_past"]:
        p = g["nothing_is_made_past"]
        lines.append(f"NOTHING IS MADE FROM FRAME {p['from_frame']} ON (to {p['to_frame']}).")
    lines.append(f"Read from the table's own rows and the ledger's read of each capture; nothing typed. It cannot see {CANNOT}. The whole list with the reasons on "
                 "record: bench/piece_gaps.py --table <this table's .json>.")
    return lines


def markdown(g: dict, read_from: str) -> str:
    side = lambda p, key: "none" if not p else f"{p['name']} ({p['subject']}, to {p['last']}, rendered {p['day']})" if key == "before" else f"{p['name']} ({p['subject']}, from {p['first']}, rendered {p['day']})"   # noqa: E731
    out = [f"# The frames no generated piece covers, frames {g['first']} to {g['up_to']}", "",
           f"Read from `{read_from}` by `bench/piece_gaps.py`; nothing typed, nothing measured on a picture. It cannot see {g['cannot_see']}.", "",
           f"## A. Inside no standing piece: {sum(r['frames'] for r in g['no_piece'])} frame(s) in {len(g['no_piece'])} run(s)", "",
           "| frames | clip time | n | whose, a shot at a time | the piece before | the piece after |", "|---|---|---|---|---|---|"]
    for r in g["no_piece"]:
        whose = "; ".join(f"{s['frames'][0]}-{s['frames'][1]} (shot {s['shot'][0]}-{s['shot'][1]}): {_whose(s)}"
                          + "".join(f" [{k}: {v[:200]}]" for k, v in (s.get("a_recorded_look") or {}).items()) for s in r["by_shot"])
        out.append(f"| {r['first']}-{r['last']} | {r['clip']} | {r['frames']} | {whose} | {side(r['the_piece_before'], 'before')} | {side(r['the_piece_after'], 'after')} |")
    out += ["", f"## D. Present, nothing to make under this plan, by a person's look (not a gap): {sum(r['frames'] for r in g['nothing_to_make'])} frame(s) "
            f"in {len(g['nothing_to_make'])} run(s)", "", "| frames | clip time | n | the look, whole |", "|---|---|---|---|"]
    out += [f"| {r['first']}-{r['last']} | {r['clip']} | {r['frames']} | "
            + "; ".join(f"{k}: {' | '.join(v)}" for s in r["by_shot"] for k, v in s["looks"].items()) + " |" for r in g["nothing_to_make"]]
    out += ["", f"## B. Inside a piece, left as the original by its load: {sum(r['frames'] for r in g['left_by_a_piece'])} frame(s) in {len(g['left_by_a_piece'])} run(s)", ""]
    if g["left_by_a_piece_not_known_for"]:
        out += [f"NOT KNOWN for {len(g['left_by_a_piece_not_known_for'])} of the {g['pieces_named']} piece(s), whose ledger entry does not say it and for which no assembler's record "
                f"was handed in: {', '.join(g['left_by_a_piece_not_known_for'])}. Not known is not none.", ""]
    out += [
            "| frames | clip time | n | subject | the piece, and the day it was rendered | the reason on record | against the assembler's record |", "|---|---|---|---|---|---|---|"]
    for r in g["left_by_a_piece"]:
        words = f"READ FROM {r['read_from']}. " + r["why"] + (f". The load's own words: {r['the_loads_own_words'][:400]}" if r.get("the_loads_own_words") else "")
        out.append(f"| {r['first']}-{r['last']} | {r['clip']} | {r['frames']} | {r['subject']} | {r['piece']}, {r['day']} | {words} | "
                   f"{json.dumps(r['against_the_assemblers_record']) if 'against_the_assemblers_record' in r else 'no record was handed in'} |")
    out += ["", f"## C. A subject in the shot with no piece of their own, where another's piece is laid: {sum(r['frames'] for r in g['another_subject_with_none'])} frame(s) "
            f"in {len(g['another_subject_with_none'])} run(s)", "", "| frames | clip time | n | subject | the shot | the pieces laid there | a recorded look |", "|---|---|---|---|---|---|---|"]
    for r in g["another_subject_with_none"]:
        out.append(f"| {r['first']}-{r['last']} | {r['clip']} | {r['frames']} | {r['subject']}{' (the record is unsure)' if r['present'] == 'unsure' else ''} | "
                   f"{r['shot'][0]}-{r['shot'][1]} | {', '.join(r['the_pieces_laid_there'])} | {(r['a_recorded_look'] or 'none')[:300]} |")
    if g["nothing_is_made_past"]:
        out += ["", f"Nothing is made from frame {g['nothing_is_made_past']['from_frame']} on (to {g['nothing_is_made_past']['to_frame']})."]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=str(__doc__).splitlines()[0])
    ap.add_argument("--table", required=True, help="a delivery table's .json (bench/delivery_table.py --json)")
    ap.add_argument("--plan", help="the video's plan file: its departures are the recorded looks")
    ap.add_argument("--ledger", help="the render ledger: with it, each load's own words on frames it left")
    ap.add_argument("--record", help="the assembler's record of a clip built from this table: held to the ledger's frames")
    ap.add_argument("--up-to", type=int, help="the last frame looked at; the last frame with any piece when left out")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text()) if p else None                       # noqa: E731
    table = load(a.table)
    g = gaps_of(table, load(a.plan), load(a.ledger), load(a.record), a.up_to)
    for line in loud(table, load(a.plan), a.up_to, load(a.ledger), load(a.record)):
        print(line)
    if a.md:
        Path(a.md).write_text(markdown(g, a.table))
    if a.json:
        Path(a.json).write_text(json.dumps({"read_from": {"table": a.table, "plan": a.plan, "ledger": a.ledger, "record": a.record}, **g}, indent=1) + "\n")
    if a.md or a.json:
        print("written:", *(x for x in (a.md, a.json) if x))
    return 0


if __name__ == "__main__":
    sys.exit(main())
