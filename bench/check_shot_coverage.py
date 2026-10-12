#!/usr/bin/env python3
"""`bench/shot_coverage.py` lays a plan, a delivery table and a renders inventory side by side, a shot and a subject at a time; each case is a way a row could say the wrong thing.

Why it exists: on 2026-10-11 the state of the music video lived in a hand-typed delivery table and in the heads of
six sessions, and the day's renders went to one shot while others had none. The tool's rows are only worth
reading if each state means one thing about the files. The fixture is a small video of its own: three shots, two
subjects, a table and an inventory made here, so the expected rows are known by construction.

  a_piece_of_the_plans_kind_is_as_planned        parts where the plan says face and hair
  the_other_kind_is_not_as_planned               a whole-subject piece where the plan says face
  a_piece_the_inventory_lacks_has_no_region      the state says so instead of guessing
  a_piece_on_some_frames_is_partly               and the row says which frames
  a_source_line_is_said_why                      and its words are on the row
  nothing_on_a_present_subject_is_no_line        the gap the table is silent on
  unsure_presence_plans_nothing                  the row is `presence unsure` whatever the table says
  absent_subjects_have_no_row                    and the frames by state add up to the present shots only
  a_row_outside_the_span_is_refused              by name
  a_plan_for_a_subject_the_shots_lack_is_refused and a missing plan likewise
  a_render_not_in_the_table_is_listed            under the row, with its region

    <python> bench/check_shot_coverage.py

No model, no card, no ComfyUI import: the tool imports the delivery tool's two readers from its file, so this
check needs cv2 and numpy as that file does.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "shot_coverage.py"


def write(folder: Path, name: str, obj) -> Path:
    p = folder / name
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))
    return p


def fixture(folder: Path, *, shots=None, plan=None, table=None, renders=True) -> dict:
    shots = shots or {
        "schema": "h3_shots/1", "video": "t.mp4", "source_frames": 300, "span": [0, 99], "subjects": ["a", "b"],
        "shots": [
            {"first": 0, "last": 29, "who": {"a": {"present": "yes"}, "b": {"present": "no"}}},
            {"first": 30, "last": 59, "who": {"a": {"present": "yes"}, "b": {"present": "yes"}}},
            {"first": 60, "last": 99, "who": {"a": {"present": "unsure"}, "b": {"present": "yes"}}}]}
    plan = plan or {"schema": "h3_plan/1", "subjects": {"a": {"region": "face and hair"}, "b": {"region": "face"}},
                    "load_prefixes": {"a": ["a_"], "b": ["b_"]}, "departures": []}
    table = table if table is not None else "\n".join([
        "0-29 /share/Video/x/a_one/a_one_00001.mp4 0",                 # a: parts, as planned
        "30-59 /share/Video/x/b_two/b_two_00001.mp4 30",               # b: whole, not as planned
        "30-44 /share/Video/x/a_three/a_three_00001.mp4 30",           # a: partly (30-44 of 30-59)
        "60-99 /share/Video/x/a_four/a_four_00001.mp4 60",             # a unsure: plans nothing
        "source 60-99 subject=b too small to track",                   # b: said why
        ""])
    inv = {"renders": [
        {"name": "a_one", "file": "Video/x/a_one/a_one_00001.mp4", "first": 0, "last": 29, "motion_wired": False,
         "as_the_graph_says": {"replace": "the wired parts"}},
        {"name": "b_two", "file": "Video/x/b_two/b_two_00001.mp4", "first": 30, "last": 59, "motion_wired": True,
         "as_the_graph_says": {"replace": "whole subject"}},
        {"name": "a_spare", "file": "Video/x/a_spare/a_spare_00001.mp4", "first": 0, "last": 29, "motion_wired": False,
         "as_the_graph_says": {"replace": "whole subject"}}]}
    paths = {"shots": write(folder, "shots.json", shots), "plan": write(folder, "plan.json", plan),
             "table": write(folder, "t.txt", table)}
    if renders:
        paths["renders"] = write(folder, "renders.json", inv)
    return paths


def run(paths: dict, out: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(TOOL), "--shots", str(paths["shots"]), "--plan", str(paths["plan"]), "--table", str(paths["table"])]
    if "renders" in paths:
        cmd += ["--renders", str(paths["renders"])]
    if out:
        cmd += ["--json", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True)


def rows_of(paths: dict, folder: Path) -> dict:
    out = folder / "out.json"
    r = run(paths, out)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.load(open(out))


def row(out, first, subject):
    return next(r for r in out["rows"] if r["first"] == first and r["subject"] == subject)


def main() -> int:
    fails = 0

    def case(name, ok, why=""):
        nonlocal fails
        print(("ok   " if ok else "FAIL ") + name + ("" if ok else f": {why}"))
        fails += not ok

    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        out = rows_of(fixture(folder), folder)
        case("a_piece_of_the_plans_kind_is_as_planned", row(out, 0, "a")["state"] == "as planned", row(out, 0, "a")["state"])
        case("the_other_kind_is_not_as_planned", row(out, 30, "b")["state"] == "not as planned", row(out, 30, "b")["state"])
        r = row(out, 30, "a")
        case("a_piece_on_some_frames_is_partly", r["state"] == "partly" and r["laid"][0]["frames"] == [30, 44], json.dumps(r["laid"]))
        r = row(out, 60, "b")
        case("a_source_line_is_said_why", r["state"] == "source, said why" and r["left"][0]["why"] == "too small to track", r["state"])
        case("unsure_presence_plans_nothing", row(out, 60, "a")["state"] == "presence unsure", row(out, 60, "a")["state"])
        case("absent_subjects_have_no_row", not any(r["first"] == 0 and r["subject"] == "b" for r in out["rows"])
             and out["frames_by_state"]["b"] == {"not as planned": 30, "source, said why": 40}, json.dumps(out["frames_by_state"]))
        others = [o for o in row(out, 0, "a")["others"] if not o["in_table"]]
        case("a_render_not_in_the_table_is_listed", [o["name"] for o in others] == ["a_spare"] and others[0]["region"] == "whole subject",
             json.dumps(others))

        folder2 = Path(tempfile.mkdtemp(dir=d))
        out = rows_of(fixture(folder2, renders=False), folder2)
        case("a_piece_the_inventory_lacks_has_no_region", row(out, 0, "a")["state"] == "region not readable", row(out, 0, "a")["state"])

        folder3 = Path(tempfile.mkdtemp(dir=d))
        out = rows_of(fixture(folder3, table="source 60-99 subject=b too small to track\n"), folder3)
        case("nothing_on_a_present_subject_is_no_line", row(out, 0, "a")["state"] == "source, no line", row(out, 0, "a")["state"])

        folder4 = Path(tempfile.mkdtemp(dir=d))
        r = run(fixture(folder4, table="0-120 /share/Video/x/a_one/a_one_00001.mp4 0\n"))
        case("a_row_outside_the_span_is_refused", r.returncode != 0 and "outside the shots file's span" in r.stdout + r.stderr, r.stdout + r.stderr)

        folder5 = Path(tempfile.mkdtemp(dir=d))
        bad = {"schema": "h3_plan/1", "subjects": {"a": {"region": "face"}, "c": {"region": "face"}}}
        r = run(fixture(folder5, plan=bad))
        folder6 = Path(tempfile.mkdtemp(dir=d))
        r2 = run(fixture(folder6, plan={"schema": "h3_plan/1", "subjects": {"a": {"region": "face"}}}))
        case("a_plan_for_a_subject_the_shots_lack_is_refused",
             r.returncode != 0 and "`c` is not a subject" in r.stdout + r.stderr and r2.returncode != 0 and "no plan for `b`" in r2.stdout + r2.stderr,
             (r.stdout + r.stderr + r2.stdout + r2.stderr)[-300:])

    print(f"{'FAIL' if fails else 'ok'}: {fails} failing case(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
