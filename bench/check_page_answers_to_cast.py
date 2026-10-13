#!/usr/bin/env python3
"""`bench/page_answers_to_cast.py` takes the owner's clicks on the review page into the cast file; each case is a way a click could be stored wrongly, twice, or when the owner had not decided.

Why it exists: the page saves one raw line a click, saved twice, changed a second later or answered two ways
(the owner's own description, 2026-10-11), and eighty-one such lines sat in no cast file for a day. The tool is
only worth having if what it stores is what the owner last said, once, through the cast file's own rules.

  the_latest_button_per_person_stands        an earlier click on the same person is not stored
  a_tile_is_stored_through_from_tile         a named person becomes a box; an unnamed subject is `none of the outlined people`
  not_sure_stores_nothing                    the tile is printed as waiting on the owner
  taken_back_leaves_the_person_unanswered    and a tile with no standing button stores nothing
  a_second_run_appends_nothing               the same answers read as already there
  a_changed_answer_is_appended_not_rewritten the file keeps both lines and the latest stands
  un_naming_a_boxed_subject_is_refused       the cast file's own refusal is passed on, nothing written, the tool exits 0
  words_are_printed_and_not_read             the owner's typed words appear beside the tile, unparsed
  another_persons_clicks_are_left_out        a session's test click stores nothing and is listed
  a_frame_no_shot_shows_is_refused           by table name and frame
  a_subjects_click_names_their_number        the who page's line shape: a subject and their number, or not outlined / not in shot
  dry_run_writes_nothing
  a_click_outside_the_job_stores_nothing_of_its_tile   a subject the tile's job does not track: that tile is printed as NOT
                                             STORED, whole, and the tiles after it are stored
  a_tile_that_cannot_be_read_leaves_the_cast_as_it_was every tile is read before any is written: a refusal on a later tile
                                             stores nothing of an earlier one
  a_span_leaves_the_tiles_outside_it         with --span, a tile of an older preview outside it is counted and left where it would
                                             stop the run; inside the span a table no preview holds is still refused
  not_outlined_alone_is_stored_for_that_subject  on a numbered picture, `on this frame, not outlined` with no number clicked is a
                                             cast line `none of the outlined people` with no box, and `not on this frame` the same
  a_subject_nobody_clicked_for_gets_no_line  on a tile asked of the subjects, the job's other subject is printed NOT ANSWERED
  a_subjects_answer_carries_its_time         the line's `by` has the click's time, where it read "at a time the answers do not give"
  not_outlined_does_not_take_a_box_back      the cast file's refusal, passed on: the box stands

A plain frame (the who page's tile of the source itself: table `plain`, no number), into a `--shots` record and
never into the cast file:

  a_plain_frame_reaches_the_record_and_not_the_cast   `present` is yes or no, `how` names the frame, `by` the lines; no cast line
  the_latest_plain_click_stands                       a subject clicked twice on one frame is read as the later click
  a_subject_on_any_answered_frame_is_yes              on one frame and not on another of the same shot is `yes`, both frames named
  a_subject_not_answered_is_left                      a subject no plain click names keeps the entry it had
  a_plain_run_needs_no_preview                        and the numbered pictures' tiles are counted and left, not refused
  a_plain_dry_run_writes_neither_file                 the record's bytes are the ones it had and no cast file is made
  a_run_that_wrote_says_so_in_written_by              after the words that were there: the lines, and the shots whose words were not read
  a_second_plain_run_writes_nothing                   the record is not written again, so `written_by` is not added to twice
  a_changed_plain_button_rewrites_the_tools_own_entry an entry this tool wrote is written again by the later click
  a_shot_the_record_does_not_have_is_refused          the page's shot and the record's differ in a frame: by name, nothing written
  another_hands_entry_is_not_replaced                 the same word is left, another word is printed DISAGREES, neither is written
  a_standing_box_in_the_shot_makes_it_yes             a box in the cast file inside the shot outweighs `not on this frame` elsewhere in it
  a_plain_frame_with_no_record_stores_nothing         with `--preview` only, the plain tiles are counted in one line and nothing is stored
  the_owners_words_are_printed_beside_the_shot        whole, with their line, and copied nowhere

    <python> bench/check_page_answers_to_cast.py

No model, no card; the cast file module is imported from its file, no ComfyUI.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "page_answers_to_cast.py"


def table() -> dict:
    return {"version": 2, "size": [1024, 768], "shots": [
        {"shot": 1, "first_frame": 0, "last_frame": 99, "shown_frame": 10,
         "people": [{"person": 1, "box": [10, 10, 50, 80]}, {"person": 2, "box": [200, 20, 60, 90]}]},
        {"shot": 2, "first_frame": 100, "last_frame": 199, "shown_frame": 150,
         "people": [{"person": 1, "box": [300, 30, 40, 70]}]}]}


def line(n, **kw):
    d = {"at": f"2026-10-12T10:00:{n:02d}Z", "who": "owner", "row": "who", "what": "choice", "table": "pv_lead_shots_00001_shots.json"}
    d.update(kw)
    return d


def setup(folder: Path, lines: list[dict]) -> dict:
    tables = folder / "tables"
    tables.mkdir()
    (tables / "pv_lead_shots_00001_shots.json").write_text(json.dumps(table()))
    (tables / "pv_second_shots_00001_shots.json").write_text(json.dumps(table()))
    job = folder / "job.json"
    job.write_text(json.dumps({"job": {"name": "pv", "first": 1000, "subjects": [{"label": "lead"}, {"label": "second"}]}}))
    ans = folder / "answers.jsonl"
    ans.write_text("".join(json.dumps(d) + "\n" for d in lines))
    return {"answers": ans, "cast": folder / "cast.jsonl", "preview": f"{job}={tables}"}


def run(p: dict, dry=False, *more: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(TOOL), "--answers", str(p["answers"]), "--cast", str(p["cast"]), "--preview", p["preview"], *more]
    if dry:
        cmd.append("--dry-run")
    return subprocess.run(cmd, capture_output=True, text=True)


def record(entries: dict | None = None) -> dict:
    """A shots record of two shots, every `present` unsure unless `entries` gives {(first, subject): entry}."""
    unsure = {"present": "unsure", "how": "nobody has looked", "by": "nobody"}
    shots = [{"first": first, "last": last, "who": {sub: dict((entries or {}).get((first, sub), unsure)) for sub in ("lead", "second")}}
             for first, last in ((2000, 2099), (2100, 2299))]
    return {"schema": "h3_shots/1", "video": "v.mp4", "span": [2000, 2299], "subjects": ["lead", "second"], "written_by": "a hand, earlier",
            "shots": shots}


def plain(n, frame, subject, chosen, shot=(2000, 2099), **kw):
    return line(n, table="plain", source_frame=frame, subject=subject, chosen=chosen, person=None, table_frame=0, shot=0,
                frames=list(shot), video="v.mp4", **kw)


def setup_plain(folder: Path, lines: list[dict], entries: dict | None = None) -> dict:
    p = setup(folder, lines)
    p["shots"] = folder / "shots.json"
    p["shots"].write_text(json.dumps(record(entries), indent=1))
    return p


def run_plain(p: dict, dry=False, preview=False) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(TOOL), "--answers", str(p["answers"]), "--cast", str(p["cast"]), "--shots", str(p["shots"])]
    cmd += ["--preview", p["preview"]] if preview else []
    return subprocess.run(cmd + (["--dry-run"] if dry else []), capture_output=True, text=True)


def who(p: dict, first: int, subject: str) -> dict:
    return next(s for s in json.loads(Path(p["shots"]).read_text())["shots"] if s["first"] == first)["who"][subject]


def cast_lines(p: dict) -> list[dict]:
    path = Path(p["cast"])
    return [json.loads(t) for t in path.read_text().splitlines()] if path.is_file() else []


def main() -> int:
    fails = 0

    def case(name, ok, why=""):
        nonlocal fails
        print(("ok   " if ok else "FAIL ") + name + ("" if ok else f": {why}"))
        fails += not ok

    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        p = setup(folder, [
            line(1, source_frame=1010, person=1, chosen="the second subject"),
            line(2, source_frame=1010, person=1, chosen="the lead"),           # the latest stands
            line(3, source_frame=1010, person=2, chosen="none of these"),
            line(4, source_frame=1010, what="words", words="1 is the singer, 2 is a passer-by"),
            line(5, source_frame=1150, person=1, chosen="not sure"),
            line(6, source_frame=1010, who="mrx (test of the page)", person=2, chosen="the lead"),
        ])
        r = run(p, dry=True)
        case("dry_run_writes_nothing", r.returncode == 0 and not Path(p["cast"]).exists() and "would store" in r.stdout, r.stdout + r.stderr)
        r = run(p)
        got = cast_lines(p)
        lead = [x for x in got if x["subject"] == "lead"]
        second = [x for x in got if x["subject"] == "second"]
        case("the_latest_button_per_person_stands", len(lead) == 1 and lead[0]["frame"] == 1010 and lead[0]["answer"] == "here"
             and lead[0]["box"] == [10.0, 10.0, 50.0, 80.0], json.dumps(got))
        case("a_tile_is_stored_through_from_tile", len(second) == 1 and second[0]["answer"] == "none of the outlined people"
             and lead[0]["table"] == "pv_lead_shots_00001_shots.json", json.dumps(got))
        # who said so and when, with no hole: an answer that carries a time is given it; one that carries none says so
        case("the_line_says_when_the_page_was_answered", "on the page at 2026-10-12T10:00:" in lead[0]["by"] and "at  (" not in lead[0]["by"], lead[0]["by"])
        with tempfile.TemporaryDirectory() as tmp_untimed:
            untimed = setup(Path(tmp_untimed), [line(1, source_frame=1010, person=1, chosen="the lead", at="")])
            r_untimed = run(untimed)
            by = (cast_lines(untimed) or [{"by": "nothing was stored: " + r_untimed.stdout + r_untimed.stderr}])[0]["by"]
            case("an_answer_with_no_time_is_said_so_and_leaves_no_hole", "on the page at a time the answers do not give (" in by and "at  (" not in by, by)
        case("not_sure_stores_nothing", "WAITING on the owner" in r.stdout and "frame 1150" in r.stdout
             and not any(x["frame"] == 1150 for x in got), r.stdout)
        case("words_are_printed_and_not_read", "1 is the singer, 2 is a passer-by" in r.stdout, r.stdout)
        case("another_persons_clicks_are_left_out", "left out, not the owner's: 1 line(s) by mrx (test of the page)" in r.stdout
             and len(got) == 2, r.stdout)
        r2 = run(p)
        case("a_second_run_appends_nothing", len(cast_lines(p)) == 2 and "already there" in r2.stdout and "stored 0 tile(s)" in r2.stdout,
             r2.stdout)

        with open(p["answers"], "a") as f:
            f.write(json.dumps(line(7, source_frame=1010, person=2, chosen="the second subject")) + "\n")
        r3 = run(p)
        got = cast_lines(p)
        second = [x for x in got if x["subject"] == "second"]
        case("a_changed_answer_is_appended_not_rewritten", len(got) == 4 and len(second) == 2 and second[-1]["answer"] == "here"
             and second[-1]["box"] == [200.0, 20.0, 60.0, 90.0] and "replaces: lead was here; second was none" in r3.stdout,
             r3.stdout + json.dumps(got))
        with open(p["answers"], "a") as f:
            f.write(json.dumps(line(8, source_frame=1010, person=1, chosen="none of these")) + "\n")
        r4 = run(p)
        case("un_naming_a_boxed_subject_is_refused", r4.returncode == 0 and "REFUSED by the cast file" in r4.stdout
             and "withdraw it" in r4.stdout and len(cast_lines(p)) == 4 and "refused by the cast file 1" in r4.stdout, r4.stdout + r4.stderr)

        folder2 = Path(tempfile.mkdtemp(dir=d))
        p2 = setup(folder2, [line(1, source_frame=1010, person=1, chosen="the lead"),
                             line(2, source_frame=1010, person=1, chosen="taken back")])
        r = run(p2)
        case("taken_back_leaves_the_person_unanswered", r.returncode == 0 and "no button stands" in r.stdout and not cast_lines(p2), r.stdout)

        folder6 = Path(tempfile.mkdtemp(dir=d))
        p6 = setup(folder6, [line(1, source_frame=1010, subject="lead", chosen="2"),
                             line(2, source_frame=1010, subject="second", chosen="not outlined"),
                             line(3, source_frame=1150, subject="lead", chosen="not in shot"),
                             line(4, source_frame=1150, subject="second", chosen="1")])
        r = run(p6)
        got6 = cast_lines(p6)
        lead6 = {x["frame"]: x for x in got6 if x["subject"] == "lead"}
        case("a_subjects_click_names_their_number", r.returncode == 0 and lead6[1010]["answer"] == "here" and lead6[1010]["box"] == [200.0, 20.0, 60.0, 90.0]
             and lead6[1150]["answer"] == "none of the outlined people"
             and [x for x in got6 if x["subject"] == "second" and x["frame"] == 1150][0]["answer"] == "here", r.stdout + r.stderr + json.dumps(got6))

        folder3 = Path(tempfile.mkdtemp(dir=d))
        p3 = setup(folder3, [line(1, source_frame=1042, person=1, chosen="the lead")])
        r = run(p3)
        case("a_frame_no_shot_shows_is_refused", r.returncode != 0 and "no shot shows source frame 1042" in r.stdout + r.stderr, r.stdout + r.stderr)

        f4 = Path(tempfile.mkdtemp(dir=d))
        p4 = setup(f4, [line(1, source_frame=1010, subject="lead", chosen="1"),
                        line(2, source_frame=1010, subject="third", chosen="2"),      # the job names lead and second
                        line(3, source_frame=1150, subject="lead", chosen="1")])
        r = run(p4)
        got4 = cast_lines(p4)
        case("a_click_outside_the_job_stores_nothing_of_its_tile", r.returncode == 0 and "NOT STORED, a click names a subject" in r.stdout
             and "(source frame 1010), lines 1-2: third; the job (pv) tracks lead, second" in r.stdout and "not stored 1;" in r.stdout
             and got4 and {x["frame"] for x in got4} == {1150}, r.stdout + r.stderr + json.dumps(got4))

        f5 = Path(tempfile.mkdtemp(dir=d))
        p5 = setup(f5, [line(1, source_frame=1010, subject="lead", chosen="1"),
                        line(2, source_frame=1042, subject="lead", chosen="1")])      # no shot shows 1042, and it sorts after 1010
        r = run(p5)
        case("a_tile_that_cannot_be_read_leaves_the_cast_as_it_was", r.returncode != 0 and "no shot shows source frame 1042" in r.stdout + r.stderr
             and not cast_lines(p5), r.stdout + r.stderr + json.dumps(cast_lines(p5)))

        f14 = Path(tempfile.mkdtemp(dir=d))
        older = "older_lead_shots_00001_shots.json"                      # a table of a preview this run does not name
        p14 = setup(f14, [line(1, source_frame=500, subject="lead", chosen="1", table=older),
                          line(2, source_frame=1010, subject="lead", chosen="1")])
        r_all = run(p14)
        r_in = run(p14, False, "--span", "1000-1199")
        with open(p14["answers"], "a") as f:
            f.write(json.dumps(line(3, source_frame=1150, subject="lead", chosen="1", table=older)) + "\n")
        r_bad = run(p14, False, "--span", "1000-1199")
        case("a_span_leaves_the_tiles_outside_it", r_all.returncode != 0 and "no --preview holds this table" in r_all.stdout + r_all.stderr
             and r_in.returncode == 0 and "outside --span 1000-1199, not read: 1 tile(s)" in r_in.stdout and "stored 1 tile(s)" in r_in.stdout
             and [x["frame"] for x in cast_lines(p14)] == [1010]
             and r_bad.returncode != 0 and f"{older}: no --preview holds this table" in r_bad.stdout + r_bad.stderr,
             r_all.stdout + r_all.stderr + r_in.stdout + r_in.stderr + r_bad.stdout + r_bad.stderr)

        f12 = Path(tempfile.mkdtemp(dir=d))
        p12 = setup(f12, [line(1, source_frame=1010, subject="second", chosen="not outlined"),
                          line(2, source_frame=1150, subject="lead", chosen="not on frame")])
        r = run(p12)
        got12 = {(x["subject"], x["frame"]): x for x in cast_lines(p12)}
        case("not_outlined_alone_is_stored_for_that_subject", r.returncode == 0 and set(got12) == {("second", 1010), ("lead", 1150)}
             and all(x["answer"] == "none of the outlined people" and x["box"] is None for x in got12.values())
             and "second: on this frame, not outlined (stored as `none of the outlined people`)" in r.stdout and "stored 2 tile(s)" in r.stdout,
             r.stdout + r.stderr + json.dumps(cast_lines(p12)))
        case("a_subject_nobody_clicked_for_gets_no_line", ("lead", 1010) not in got12 and ("second", 1150) not in got12
             and "frame 1010), lines 1-1: second: on this frame, not outlined (stored as `none of the outlined people`); NOT ANSWERED, no line written: lead" in r.stdout,
             r.stdout)
        case("a_subjects_answer_carries_its_time", got12 and "on the page at 2026-10-12T10:00:01Z (" in got12[("second", 1010)]["by"], json.dumps(cast_lines(p12)))
        f13 = Path(tempfile.mkdtemp(dir=d))
        p13 = setup(f13, [line(1, source_frame=1010, subject="lead", chosen="1")])
        run(p13)
        with open(p13["answers"], "a") as f:
            f.write(json.dumps(line(2, source_frame=1010, subject="lead", chosen="not outlined")) + "\n")
        r = run(p13)
        case("not_outlined_does_not_take_a_box_back", r.returncode == 0 and "REFUSED by the cast file" in r.stdout
             and [(x["subject"], x["answer"]) for x in cast_lines(p13)] == [("lead", "here")], r.stdout + r.stderr + json.dumps(cast_lines(p13)))

        # --- a plain frame: the cast file and the shots record -------------------------------------------------------
        f7 = Path(tempfile.mkdtemp(dir=d))
        p7 = setup_plain(f7, [
            plain(1, 2050, "lead", "not on frame"),
            plain(2, 2050, "lead", "not outlined"),                          # the later click on the same frame stands
            plain(3, 2050, "second", "not on frame"),
            plain(4, 2150, "lead", "not outlined", shot=(2100, 2299)),
            plain(5, 2250, "lead", "not on frame", shot=(2100, 2299)),       # on one answered frame of the shot, not on another
            line(6, source_frame=1010, person=1, chosen="the lead"),         # a numbered picture's tile, and no --preview
            {"at": "2026-10-12T10:00:07Z", "who": "owner", "row": "who", "what": "words", "words": "she is small, on the left",
             "source_frame": 2100, "frames": [2100, 2299], "video": "v.mp4"},
        ])
        before = Path(p7["shots"]).read_bytes()
        r = run_plain(p7, dry=True)
        case("a_plain_dry_run_writes_neither_file", r.returncode == 0 and Path(p7["shots"]).read_bytes() == before
             and not Path(p7["cast"]).exists() and "would write 3 entries (yes 2, no 1) from 3 plain frame(s) in 2 shot(s)" in r.stdout,
             r.stdout + r.stderr)
        r = run_plain(p7)
        case("a_plain_frame_reaches_the_record_and_not_the_cast", r.returncode == 0 and not cast_lines(p7)
             and who(p7, 2000, "lead")["present"] == "yes" and who(p7, 2000, "second")["present"] == "no"
             and who(p7, 2000, "second")["how"] == "frame 2050 (the source itself): not on this frame"
             and "answers.jsonl lines 3-3" in who(p7, 2000, "second")["by"] and "entered by page_answers_to_cast" in who(p7, 2000, "second")["by"],
             r.stdout + r.stderr + json.dumps(cast_lines(p7)) + json.dumps(who(p7, 2000, "second")))
        case("the_latest_plain_click_stands", who(p7, 2000, "lead")["how"] == "frame 2050 (the source itself): on this frame, not outlined",
             json.dumps(who(p7, 2000, "lead")))
        # the frames as fields beside the prose: marked ON by a click is in `frames_on`; marked not on is not
        case("a_frame_marked_on_is_a_field", who(p7, 2000, "lead")["frames_on"] == [2050] and who(p7, 2000, "lead")["frames_boxed"] == []
             and who(p7, 2000, "second")["frames_on"] == [], json.dumps(who(p7, 2000, "lead")) + json.dumps(who(p7, 2000, "second")))
        case("a_subject_on_any_answered_frame_is_yes", who(p7, 2100, "lead") == {
            "present": "yes", "how": "frame 2150 (the source itself): on this frame, not outlined; frame 2250 (the source itself): not on this frame",
            "frames_on": [2150], "frames_boxed": [],
            "by": "owner, on the who page at 2026-10-12T10:00:05Z (answers.jsonl lines 4-5), entered by page_answers_to_cast"},
            json.dumps(who(p7, 2100, "lead")))
        case("a_subject_not_answered_is_left", who(p7, 2100, "second") == {"present": "unsure", "how": "nobody has looked", "by": "nobody"},
             json.dumps(who(p7, 2100, "second")))
        case("a_plain_run_needs_no_preview", "left for a run that names their previews (no --preview was given): 1 tile(s)" in r.stdout, r.stdout)
        case("the_owners_words_are_printed_beside_the_shot", "words on 2100-2299 (line 7): she is small, on the left" in r.stdout
             and "she is small" not in Path(p7["shots"]).read_text(), r.stdout)
        said = json.loads(Path(p7["shots"]).read_text())["written_by"]
        case("a_run_that_wrote_says_so_in_written_by", said.startswith("a hand, earlier; `present` entries written by bench/page_answers_to_cast.py at 20")
             and f"plain-frame answers in {p7['answers']} (lines 2-5); the owner's words on 1 shot(s) stay in that file" in said, said)
        written = Path(p7["shots"]).read_bytes()
        stamp = Path(p7["shots"]).stat().st_mtime_ns
        r2 = run_plain(p7)
        case("a_second_plain_run_writes_nothing", r2.returncode == 0 and not cast_lines(p7) and Path(p7["shots"]).read_bytes() == written
             and Path(p7["shots"]).stat().st_mtime_ns == stamp and "written 0 entries" in r2.stdout and "already there 3" in r2.stdout,
             r2.stdout + r2.stderr)
        with open(p7["answers"], "a") as f:
            f.write(json.dumps(plain(8, 2050, "second", "not outlined")) + "\n")
        r3 = run_plain(p7)
        case("a_changed_plain_button_rewrites_the_tools_own_entry", r3.returncode == 0 and not cast_lines(p7)
             and who(p7, 2000, "second")["present"] == "yes" and "second: yes" in r3.stdout and "(was no)" in r3.stdout
             and "written 1 entry" in r3.stdout, r3.stdout + r3.stderr)

        f8 = Path(tempfile.mkdtemp(dir=d))
        p8 = setup_plain(f8, [plain(1, 2050, "lead", "not outlined", shot=(2000, 2120))])     # the page's shot ended at 2120
        before = Path(p8["shots"]).read_bytes()
        r = run_plain(p8)
        case("a_shot_the_record_does_not_have_is_refused", r.returncode != 0 and "under shot [2000, 2120]" in r.stdout + r.stderr
             and "has shot 2000-2099" in r.stdout + r.stderr and Path(p8["shots"]).read_bytes() == before and not cast_lines(p8),
             r.stdout + r.stderr)

        f9 = Path(tempfile.mkdtemp(dir=d))
        hand = {(2000, "lead"): {"present": "no", "how": "frame 2050: not on this frame", "by": "owner, by hand"},
                (2000, "second"): {"present": "yes", "how": "frame 2010: 1 is the second subject", "by": "owner, by hand"}}
        p9 = setup_plain(f9, [plain(1, 2050, "lead", "not on frame"), plain(2, 2050, "second", "not on frame")], hand)
        before = Path(p9["shots"]).read_bytes()
        r = run_plain(p9)
        case("another_hands_entry_is_not_replaced", r.returncode == 0 and Path(p9["shots"]).read_bytes() == before
             and "left as it is, another hand's and the same word" in r.stdout and "lead: no (owner, by hand)" in r.stdout
             and "DISAGREES, not written, for a person to settle" in r.stdout and "second is `yes` in the record" in r.stdout
             and "left 1; DISAGREES, not written 1" in r.stdout, r.stdout + r.stderr)

        f10 = Path(tempfile.mkdtemp(dir=d))
        p10 = setup_plain(f10, [line(1, source_frame=1010, person=1, chosen="the lead"),
                                plain(2, 1050, "lead", "not on frame", shot=(1000, 1099)), plain(3, 1050, "second", "not on frame", shot=(1000, 1099))])
        rec = record()
        rec["span"], rec["shots"][0]["first"], rec["shots"][0]["last"] = [1000, 2299], 1000, 1099
        rec["shots"][1]["first"] = 1100
        Path(p10["shots"]).write_text(json.dumps(rec, indent=1))
        r = run_plain(p10, preview=True)
        lead10 = who(p10, 1000, "lead")
        case("a_standing_box_in_the_shot_makes_it_yes", r.returncode == 0 and lead10["present"] == "yes"
             and lead10["how"] == "frame 1050 (the source itself): not on this frame; frame 1010: a box in the cast file (pv_lead_shots_00001_shots.json)"
             and who(p10, 1000, "second")["present"] == "no" and len(cast_lines(p10)) == 2, r.stdout + r.stderr + json.dumps(lead10))
        # the frames as fields: marked ON by a click, and boxed in the cast file; a frame marked not-on is in neither
        case("a_boxed_frame_is_a_field_and_a_frame_marked_not_on_is_in_neither", (lead10["frames_on"], lead10["frames_boxed"]) == ([], [1010]), json.dumps(lead10))

        f11 = Path(tempfile.mkdtemp(dir=d))
        p11 = setup(f11, [plain(1, 2050, "lead", "not outlined")])
        r = run(p11)
        case("a_plain_frame_with_no_record_stores_nothing", r.returncode == 0 and not cast_lines(p11)
             and "left for a run that names a --shots record: 1 tile(s); nothing of them is stored" in r.stdout
             and "PLAIN FRAME" not in r.stdout and "plain frames no --shots file holds 1" in r.stdout, r.stdout + r.stderr)

    print(f"{'FAIL' if fails else 'ok'}: {fails} failing case(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
