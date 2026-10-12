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
  dry_run_writes_nothing

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


def run(p: dict, dry=False) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(TOOL), "--answers", str(p["answers"]), "--cast", str(p["cast"]), "--preview", p["preview"]]
    if dry:
        cmd.append("--dry-run")
    return subprocess.run(cmd, capture_output=True, text=True)


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

        folder3 = Path(tempfile.mkdtemp(dir=d))
        p3 = setup(folder3, [line(1, source_frame=1042, person=1, chosen="the lead")])
        r = run(p3)
        case("a_frame_no_shot_shows_is_refused", r.returncode != 0 and "no shot shows source frame 1042" in r.stdout + r.stderr, r.stdout + r.stderr)

    print(f"{'FAIL' if fails else 'ok'}: {fails} failing case(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
