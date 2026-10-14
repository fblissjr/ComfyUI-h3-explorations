#!/usr/bin/env python3
"""`bench/piece_gaps.py` on a made table: each case is a way a frame no piece covers could be missed, joined to the wrong shot's people, or hidden.

  no_piece_runs_are_plain_and_sorted    the frames no row lays a piece on, from the table's first frame to its last frame with any piece, as
                                        runs sorted by frame; a run that crosses a cut says whose each stretch is, a shot at a time, with
                                        "nobody of ours" where the record has nobody and `unsure` said as unsure
  the_pieces_either_side_have_their_day each run names the piece before it and the piece after it with the day each was rendered on the
                                        owner's clock, not the record's
  a_piece_that_leaves_frames            frames inside a piece's own span that its load left as the original are their own list, with the
                                        piece and the subject; with the assembler's record, where the two disagree both are printed
  not_known_is_not_none                 THE FAULT OF THE FIRST REAL RUN: on a table whose ledger predates the field, with the assembler's
                                        record handed in, the loud print listed NO frame a piece leaves while the file beside it listed
                                        six. The print's list is the file's: a run known from the record alone is printed and says what
                                        it was read from, in the print and in the file; with neither the field nor a record the print
                                        says NOT KNOWN and never a count of none; a piece whose capture does not say is named
  a_subject_with_none_under_a_piece     a subject in the shot with no piece of their own, on frames another's piece covers, is listed
                                        with the pieces laid there, and is not in the first list
  a_recorded_look_is_said_not_hidden    a `NO LOAD` departure of the plan over a run is printed with the run; the run stays in the list
  up_to_moves_the_end                   with --up-to, frames past the last piece are listed and the header says the end was asked for;
                                        without it one line says nothing is made from the next frame on
  loud_is_capitals_and_counts           the lines a table prints: the three headings in capitals with their counts, what it cannot see;
                                        a table with no piece says so in one line
  the_table_prints_it_by_itself         `bench/delivery_table.py` ends its own output with those lines
  the_files                             --md and --json are written, the json with what it was read from

    <comfy venv python> bench/check_piece_gaps.py

The standard library only; no model, no card.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import delivery_table as dt  # noqa: E402
import piece_gaps as pg  # noqa: E402

TOOL = Path(__file__).resolve().parent / "piece_gaps.py"


def entry(first, last, subject, present, stands=()):
    return {"first": first, "last": last, "subject": subject, "present": present, "passed_over": [], "left_as_the_source": [],
            "stands": [{"render": n, "frames": [list(f) for f in frames], "written": w, "leaves_as_the_original": [list(x) for x in left]} for n, frames, w, left in stands]}


def table() -> dict:
    """0-9 nobody; 10-29 a (p1 on 12-25, leaving 15-16) and b (none); 30-49 a unsure; 50-69 a (p2) and b (q on 55-60); 70-99 a, nothing made."""
    no = lambda a, b, who: {"first": a, "last": b, "subject": who, "why": "the shots file says this subject is not in the shot"}       # noqa: E731
    return {"not_in_the_shot": [no(0, 9, "a"), no(0, 9, "b"), no(30, 49, "b"), no(70, 99, "b")], "rows": [],
            "reading": [entry(10, 29, "a", "yes", [("p1", [(12, 25)], "2026-01-02T03:00:00Z", [(15, 16)])]), entry(10, 29, "b", "yes"),
                        entry(30, 49, "a", "unsure"),
                        entry(50, 69, "a", "yes", [("p2", [(50, 69)], "2026-01-03T18:00:00Z", [])]), entry(50, 69, "b", "yes", [("q", [(55, 60)], "2026-01-03T19:00:00Z", [])]),
                        entry(70, 99, "a", "yes")]}


PLAN = {"departures": [{"first": 10, "last": 29, "subject": "b", "note": "NO LOAD, it stays the source: the back of a head"},
                       {"first": 50, "last": 69, "subject": "b", "note": "a note that is not a look"}]}


def no_piece_runs():
    g = pg.gaps_of(table())
    a = g["no_piece"]
    assert [(r["first"], r["last"], r["frames"]) for r in a] == [(0, 11, 12), (26, 49, 24)] and (g["first"], g["up_to"], g["the_last_frame_with_any_piece"]) == (0, 69, 69), a
    first, second = a[0]["by_shot"], a[1]["by_shot"]
    assert [(s["frames"], s["shot"], s["in_the_shot"], s["unsure"], s["nobody_of_ours"]) for s in first] == [([0, 9], [0, 9], [], [], True), ([10, 11], [10, 29], ["a", "b"], [], False)], first
    assert [(s["frames"], s["in_the_shot"], s["unsure"], s["nobody_of_ours"]) for s in second] == [([26, 29], ["a", "b"], [], False), ([30, 49], [], ["a"], False)], second
    assert a[0]["clip"] == "0:00.0-0:00.5", a[0]["clip"]
    assert g["nothing_is_made_past"] == {"from_frame": 70, "to_frame": 99}, g["nothing_is_made_past"]
    return "two runs, sorted, each cut at the record's cuts for whose it is; nobody of ours and unsure said as that"


def either_side():
    a = pg.gaps_of(table())["no_piece"]
    assert a[0]["the_piece_before"] is None and a[0]["the_piece_after"]["name"] == "p1" and a[0]["the_piece_after"]["first"] == 12, a[0]
    b, c = a[1]["the_piece_before"], a[1]["the_piece_after"]
    # p1 was written at 03:00 UTC on the 2nd, which is the 1st on the owner's clock; p2 at 18:00 UTC on the 3rd is the 3rd there
    assert (b["name"], b["last"], b["day"], b["written"]) == ("p1", 25, "01/01", "2026-01-02T03:00:00Z") and (c["name"], c["first"], c["day"]) == ("p2", 50, "01/03"), (b, c)
    assert pg.day("2026-07-02T03:00:00Z") == "07/01" and pg.day(None) is None
    return "the piece before and after each run, with the day on the owner's clock (a render of 03:00 UTC is the day before there)"


def a_piece_that_leaves():
    g = pg.gaps_of(table())
    b = g["left_by_a_piece"]
    assert [(r["first"], r["last"], r["piece"], r["subject"], r["shot"], r["day"]) for r in b] == [(15, 16, "p1", "a", [10, 29], "01/01")] and "against_the_assemblers_record" not in b[0], b
    assert not any(15 <= r["first"] <= 16 for r in g["no_piece"]), "frames a piece covers are in the list of frames no piece covers"
    same = pg.gaps_of(table(), record={"flags": [{"rule": "piece_changes_nothing", "run": "p1", "source_frames": [[15, 16]]}, {"rule": "another", "run": "p1", "source_frames": [[20, 21]]}]})
    assert [r["against_the_assemblers_record"] for r in same["left_by_a_piece"]] == ["the same frames"], same["left_by_a_piece"]
    more = pg.gaps_of(table(), record={"flags": [{"rule": "piece_changes_nothing", "run": "p1", "source_frames": [[16, 18]]}]})["left_by_a_piece"]
    assert [(r["first"], r["last"]) for r in more] == [(15, 18)] and more[0]["against_the_assemblers_record"] == {
        "in_the_ledger_and_not_the_record": [[15, 15]], "in_the_record_and_not_the_ledger": [[17, 18]]}, more
    # the load's own words, from the job file the ledger names for the piece
    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / "p1.json"
        job.write_text(json.dumps({"job": {"left_out": [{"subject": "a", "first": 14, "frames": 4, "why": "her  arm is across her face"}, {"subject": "a", "first": 40, "frames": 2, "why": "elsewhere"}]}}))
        ledger = {"renders": {"f": {"name": "p1", "records": [{"job_file": str(job)}]}}}
        told = pg.gaps_of(table(), ledger=ledger)["left_by_a_piece"][0]
        assert told["the_loads_own_words"] == "her arm is across her face", told
        assert pg.gaps_of(table(), ledger={"renders": {"f": {"name": "p1", "records": [{"job_file": str(Path(tmp) / "gone.json")}]}}})["left_by_a_piece"][0]["the_loads_own_words"] is None
    return "frames a piece leaves are their own list; the assembler's record is held to the ledger's frames and a difference is printed both ways"


def not_known_is_not_none():
    old = table()                                           # a table made before the ledger carried the field: no stand has the key
    for e in old["reading"]:
        for s_ in e["stands"]:
            del s_["leaves_as_the_original"]
    record = {"flags": [{"rule": "piece_changes_nothing", "run": "p1", "source_frames": [[15, 16]]}]}
    said = "\n".join(pg.loud(old, record=record))
    g = pg.gaps_of(old, record=record)
    assert [(r["first"], r["last"], r["piece"]) for r in g["left_by_a_piece"]] == [(15, 16, "p1")] and g["left_by_a_piece_not_known_for"] == [], g["left_by_a_piece"]
    row = g["left_by_a_piece"][0]
    assert row["read_from"].startswith("the assembler's record alone") and "predates the field" in row["read_from"] and "capture emptied" not in row["why"], row
    assert "LEAVES AS THE ORIGINAL: 2 FRAME(S) IN 1 RUN(S):" in said and "  15-16  (2 frame(s)" in said and "read from the assembler's record alone" in said, said
    md = pg.markdown(g, "t.json")
    assert "| 15-16 |" in md and "READ FROM the assembler's record alone" in md and "its capture emptied this subject's part" not in md, md
    # neither the field nor a record: NOT KNOWN, never a count
    bare = "\n".join(pg.loud(old))
    assert "FRAMES A PIECE COVERS AND LEAVES AS THE ORIGINAL: NOT KNOWN." in bare and "NOT KNOWN IS NOT NONE" in bare and "0 FRAME(S) IN 0 RUN(S)" not in bare.split("A SUBJECT IN THE SHOT")[0].split("LEAVES AS THE ORIGINAL")[1], bare
    assert pg.gaps_of(old)["left_by_a_piece_not_known_for"] == ["p1", "p2", "q"] and "Not known is not none" in pg.markdown(pg.gaps_of(old), "t.json")
    # one piece whose capture does not say, beside pieces that do: it is named, and the others are counted
    part = table()
    part["reading"][3]["stands"][0]["leaves_as_the_original"] = None
    some = "\n".join(pg.loud(part))
    assert "LEAVES AS THE ORIGINAL: 2 FRAME(S) IN 1 RUN(S); NOT KNOWN for 1 piece(s) whose capture does not say: p2:" in some, some
    # a subject the record is unsure of, under another's piece: counted in a line of its own under list C
    unsure = table()
    unsure["reading"][1]["present"] = "unsure"
    line = [x for x in pg.loud(unsure) if x.startswith("  OF THESE")]
    assert line and "14 frame(s) in 1 run(s) are of a subject THE RECORD IS UNSURE OF" in line[0] and "the who page" in line[0], pg.loud(unsure)
    return "a run known from the record alone is printed and says so; neither says NOT KNOWN; a piece that does not say is named; the unsure are counted"


def under_a_piece():
    c = pg.gaps_of(table())["another_subject_with_none"]
    assert [(r["first"], r["last"], r["subject"], r["the_pieces_laid_there"]) for r in c] == [(12, 25, "b", ["p1"]), (50, 54, "b", ["p2"]), (61, 69, "b", ["p2"])], c
    assert c[1]["the_piece_after"]["name"] == "q" and c[2]["the_piece_before"]["name"] == "q", c[1:]
    return "b under a's pieces on three runs, with b's own piece either side"


def a_recorded_look():
    g = pg.gaps_of(table(), PLAN)
    c = g["another_subject_with_none"]
    assert c[0]["a_recorded_look"].startswith("NO LOAD, it stays the source") and c[1]["a_recorded_look"] is None, c[:2]
    seg = g["no_piece"][0]["by_shot"][1]
    assert seg["a_recorded_look"] == {"b": "NO LOAD, it stays the source: the back of a head"} and seg["in_the_shot"] == ["a", "b"], seg
    assert "b (a recorded look: no load)" in "\n".join(pg.loud(table(), PLAN)) and len(pg.gaps_of(table(), PLAN)["no_piece"]) == 2
    return "a NO LOAD departure is printed with its run, and the run stays"


def up_to_moves():
    g = pg.gaps_of(table(), up_to=80)
    assert [(r["first"], r["last"]) for r in g["no_piece"]][-1] == (70, 80) and g["nothing_is_made_past"] == {"from_frame": 81, "to_frame": 99}, g["no_piece"][-1]
    head = pg.loud(table(), up_to=80)[0]
    assert "frames 0 to 80 (as asked; the last frame with any piece is 69)" in head and "47 FRAME(S) IN 3 RUN(S)" in head, head
    assert "(the last frame with any piece)" in pg.loud(table())[0] and "NOTHING IS MADE FROM FRAME 70 ON (to 99)." in pg.loud(table()), pg.loud(table())[0]


def loud_lines():
    lines = pg.loud(table())
    text = "\n".join(lines)
    assert lines[0].startswith("FRAMES NO GENERATED PIECE COVERS, frames 0 to 69") and "36 FRAME(S) IN 2 RUN(S)" in lines[0], lines[0]
    assert "  0-11  (12 frame(s), 0:00.0-0:00.5)  0-9: nobody of ours; 10-11: a, b" in text and "  26-49  (24 frame(s)" in text and "30-49: a (the record is unsure)" in text, text
    assert "FRAMES A PIECE COVERS AND LEAVES AS THE ORIGINAL: 2 FRAME(S) IN 1 RUN(S):" in text \
        and "a, inside p1 (rendered 01/01); read from the ledger: its capture's record of where it emptied the part" in text, text
    assert "OF THESE, " not in text, "nobody in list C is unsure and a line says somebody is"
    assert "A SUBJECT IN THE SHOT WITH NO PIECE OF THEIR OWN, on frames another's piece covers: 28 FRAME(S) IN 3 RUN(S):" in text and "b, under p1" in text, text
    assert "It cannot see the picture" in text and "nothing typed" in text
    bare = table()
    for e in bare["reading"]:
        e["stands"] = []
    assert pg.loud(bare) == ["NO PIECE STANDS IN THIS TABLE: every frame of it is the original."]
    return "three headings in capitals with their counts; a table with no piece says so"


def the_table_prints_it():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        shots = {"schema": "h3_shots/1", "video": "v.mp4", "subjects": ["a"], "span": [0, 19], "shots": [{"first": 0, "last": 9, "who": {"a": {"present": "yes"}}}, {"first": 10, "last": 19, "who": {"a": {"present": "no", "by": "x"}}}]}
        render = {"file": "Video/s/p/p_00001.mp4", "name": "p", "written": "2026-01-05T00:00:00Z", "span": [2, 7], "making": dt.AGREES, "subjects": ["a"], "records": [],
                  "leaves_as_the_original": {"source_frames": [[4, 5]]}, "owner": [],
                  "form": {"region": "face", "part_named": None, "reference": {"has": {"cut_to_the_head": True, "face_points": True, "lid_marks": False}, "says": "s"}}}
        # a second piece whose ledger entry does NOT say what its load left (no capture record of it): not known, and the table must not write none
        silent = dict(render, file="Video/s/n/n_00001.mp4", name="n", span=[8, 9], leaves_as_the_original=None)
        for name, obj in (("ledger.json", {"schema": dt.LEDGER_SCHEMA, "share": "/share", "renders": {render["file"]: render, silent["file"]: silent}}), ("shots.json", shots),
                          ("plan.json", {"schema": "h3_plan/1", "video": "v.mp4", "subjects": {"a": {"region": "face", "parts": ["face_and_neck"]}}})):
            (d / name).write_text(json.dumps(obj))
        r = subprocess.run([sys.executable, str(TOOL.with_name("delivery_table.py")), "--ledger", str(d / "ledger.json"), "--shots", str(d / "shots.json"), "--plan", str(d / "plan.json"),
                            "--table", str(d / "t.txt"), "--reading", str(d / "r.md"), "--json", str(d / "t.json")], capture_output=True, text=True)
        assert r.returncode == 0, (r.stdout + r.stderr)[-900:]
        tail = r.stdout[r.stdout.index("wrote "):]
        assert "FRAMES NO GENERATED PIECE COVERS, frames 0 to 9 (the last frame with any piece): 2 FRAME(S) IN 1 RUN(S)" in tail and "  0-1  (2 frame(s)" in tail, tail
        assert "LEAVES AS THE ORIGINAL: 2 FRAME(S) IN 1 RUN(S); NOT KNOWN for 1 piece(s) whose capture does not say: n:" in tail and "  4-5  (2 frame(s)" in tail \
            and "inside p (rendered 01/04)" in tail, tail
        assert "NOTHING IS MADE FROM FRAME 10 ON (to 19)." in tail, tail
        kept = {s_["render"]: s_["leaves_as_the_original"] for s_ in json.loads((d / "t.json").read_text())["reading"][0]["stands"]}
        assert [list(x) for x in kept["p"]] == [[4, 5]] and kept["n"] is None, kept
        # THE FILES, from that table's own json
        f = subprocess.run([sys.executable, str(TOOL), "--table", str(d / "t.json"), "--plan", str(d / "plan.json"), "--md", str(d / "g.md"), "--json", str(d / "g.json")], capture_output=True, text=True)
        assert f.returncode == 0 and f.stdout.startswith("FRAMES NO GENERATED PIECE COVERS"), f.stdout + f.stderr
        g, md = json.loads((d / "g.json").read_text()), (d / "g.md").read_text()
        assert g["read_from"]["table"] == str(d / "t.json") and [(r_["first"], r_["last"]) for r_ in g["no_piece"]] == [(0, 1)] and g["left_by_a_piece"][0]["piece"] == "p", g
        assert "## A. Inside no standing piece: 2 frame(s) in 1 run(s)" in md and "| 4-5 |" in md and "no record was handed in" in md and "It cannot see the picture" in md, md
    return "the table's own print ends with the three lists, and its json feeds the tool"


case("no-piece runs are plain and sorted", no_piece_runs)
case("the pieces either side have their day", either_side)
case("a piece that leaves frames", a_piece_that_leaves)
case("not known is not none, and the print is the file's list", not_known_is_not_none)
case("a subject with none under a piece", under_a_piece)
case("a recorded look is said, not hidden", a_recorded_look)
case("--up-to moves the end", up_to_moves)
case("loud is capitals and counts", loud_lines)
case("the table prints it by itself, and the files", the_table_prints_it)

if __name__ == "__main__":
    sys.exit(finish())
