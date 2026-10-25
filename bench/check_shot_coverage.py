#!/usr/bin/env python3
"""`bench/shot_coverage.py` lays a plan, a delivery table and a renders inventory side by side, a shot and a subject at a time; each case is a way a row could say the wrong thing.

Why it exists: on 2026-10-11 the state of the music video lived in a hand-typed delivery table and in the heads of
six sessions, and the day's renders went to one shot while others had none. The tool's rows are only worth
reading if each state means one thing about the files. The fixture is a small video of its own: three shots, two
subjects, a table and an inventory made here, so the expected rows are known by construction.

  a_piece_of_the_plans_kind_is_as_planned        parts where the plan says face and hair
  the_other_kind_is_not_as_planned               a whole-subject piece where the plan says face
  a_piece_the_inventory_lacks_has_no_region      the state says so instead of guessing
  a_piece_on_some_frames_is_partly               and the row says which frames, and still says the piece's kind
  a_source_line_is_said_why                      and its words are on the row
  nothing_on_a_present_subject_is_no_line        the gap the table is silent on
  a_persons_look_is_nothing_to_make              a look that every plan part is not seen on every frame: "present, nothing to make under this plan";
                                                 a look on one of two parts is not
  nothing_to_make_is_printed_with_the_look       the look's words under the row
  a_look_outside_its_shot_is_refused             by the shots file's reader
  unsure_presence_plans_nothing                  the row is `presence unsure` whatever the table says
  absent_subjects_have_no_row                    and the frames by state add up to the present shots only
  a_row_outside_the_span_is_refused              by name
  a_plan_for_a_subject_the_shots_lack_is_refused and a missing plan likewise
  a_render_not_in_the_table_is_listed            under the row, with its region
  the_ledger_names_the_region_itself             with --ledger, a face piece where the plan says face and hair is NOT as planned (the kind
                                                 alone called it planned); a render whose region the ledger does not place is held to its
                                                 graph's kind; a render with no span is not walked; the report says which reading it is
  two_inventories_are_refused                    --ledger with --renders, and a --ledger that is not the ledger's JSON
  the_gap_check_is_read_and_verified             reads the table's gap check into each row and prints the worst-first remake
                                                 and capture lists, the other run kinds, and failures; refuses a gap check
                                                 whose `table_sha256`, span, or `clip_check` is another table's

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

import hashlib

HERE = Path(__file__).resolve().parent
TOOL = HERE / "shot_coverage.py"


def write(folder: Path, name: str, obj) -> Path:
    p = folder / name
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))
    return p


def sync_gap_check(folder: Path, *, span=(0, 99), by_shot=None, runs=None, failures=None,
                   cmd_out: str | None = "default", rows_override=None, reading_override=None,
                   clip_check_override=None) -> tuple[Path, Path, Path]:
    clip_check = folder / "joined.mp4.check.json"
    if not clip_check.exists():
        clip_check.write_text(json.dumps({"ok": True}))
    cov_file = folder / "frame_coverage_t.json"
    t_json = folder / "t.json"
    t_txt = folder / "t.txt"
    t_rows = []
    if rows_override is not None:
        t_rows = rows_override
    elif t_txt.exists():
        for ln in t_txt.read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith(("#", "source ")):
                sp, piece = ln.split()[:2]
                lo, hi = map(int, sp.split("-"))
                t_rows.append({"first": lo, "last": hi, "file": piece})
    out_flag = f" --out {clip_check}" if cmd_out == "default" else (f" --out {cmd_out}" if cmd_out else "")
    t_json.write_text(json.dumps({
        "reading": reading_override if reading_override is not None else [{"first": span[0], "last": span[1]}],
        "rows": t_rows,
        "assembly": {
            "command": f"python bench/assemble_delivery.py --table {t_txt}{out_flag}",
            "stack": {"coverage_json": str(cov_file)},
        },
    }))
    gap_data = {
        "table": str(t_json),
        "table_sha256": hashlib.sha256(t_json.read_bytes()).hexdigest(),
        "clip_check": clip_check_override if clip_check_override is not None else {
            "file": str(clip_check), "sha256": hashlib.sha256(clip_check.read_bytes()).hexdigest()
        },
        "span": list(span),
        "failures": list(failures or []),
        "by_shot": by_shot if by_shot is not None else {"remake": [], "capture": []},
        "runs": list(runs or []),
    }
    cov_file.write_text(json.dumps(gap_data))
    return t_json, cov_file, clip_check


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
         "as_the_graph_says": {"replace": "whole subject"}},
        {"name": "a_three", "file": "Video/x/a_three/a_three_00001.mp4", "first": 30, "last": 44, "motion_wired": False,
         "as_the_graph_says": {"replace": "the wired parts"}}]}
    paths = {"shots": write(folder, "shots.json", shots), "plan": write(folder, "plan.json", plan),
             "table": write(folder, "t.txt", table)}
    if renders:
        paths["renders"] = write(folder, "renders.json", inv)
    sync_gap_check(folder, span=shots["span"])
    return paths


def run(paths: dict, out: Path | None = None, more: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(TOOL), "--shots", str(paths["shots"]), "--plan", str(paths["plan"]), "--table", str(paths["table"])]
    if "renders" in paths:
        cmd += ["--renders", str(paths["renders"])]
    if "ledger" in paths:
        cmd += ["--ledger", str(paths["ledger"])]
    if out:
        cmd += ["--json", str(out)]
    if more:
        cmd += list(more)
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
        case("a_piece_on_some_frames_is_partly", r["state"] == "partly, as planned" and r["laid"][0]["frames"] == [30, 44], json.dumps(r["laid"]))
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

        # PRESENT, NOTHING TO MAKE UNDER THIS PLAN (the owner's word of 2026-10-14; a person's line, the lead's ruling): with no piece laid
        folder8 = Path(tempfile.mkdtemp(dir=d))
        looked = json.loads(fixture(folder8, table="")["shots"].read_text())
        plan8 = {"schema": "h3_plan/1", "subjects": {"a": {"region": "face and hair", "parts": ["hair", "face_and_neck"]}, "b": {"region": "face", "parts": ["face_and_neck"]}},
                 "load_prefixes": {"a": ["a_"], "b": ["b_"]}, "departures": []}
        look = {"first": 0, "last": 29, "is": "not seen", "why": "A LOOK: HER LEGS ONLY", "by": "a reader", "at": "2026-01-02T00:00:00Z"}
        looked["shots"][0]["who"]["a"]["seen_looks"] = [dict(look, part="hair"), dict(look, part="face_and_neck")]
        looked["shots"][1]["who"]["a"]["seen_looks"] = [dict(look, part="hair", first=30, last=59)]           # one part of two: not enough
        out = rows_of(fixture(folder8, shots=looked, plan=plan8, table=""), folder8)
        r = row(out, 0, "a")
        case("a_persons_look_is_nothing_to_make", r["state"] == "present, nothing to make under this plan" and "A LOOK: HER LEGS ONLY" in r["nothing_to_make"][0]
             and row(out, 30, "a")["state"] == "source, no line", (r["state"], row(out, 30, "a")["state"]))
        printed = run(fixture(folder8, shots=looked, plan=plan8, table=""))
        case("nothing_to_make_is_printed_with_the_look", "NOTHING TO MAKE UNDER THIS PLAN, a person's look: hair not seen on 0-29: A LOOK" in printed.stdout, printed.stdout[-800:])
        bad = json.loads(json.dumps(looked))
        bad["shots"][0]["who"]["a"]["seen_looks"] = [dict(look, part="hair", last=40)]                       # frames outside the shot
        refused = run(fixture(Path(tempfile.mkdtemp(dir=d)), shots=bad, plan=plan8, table=""))
        case("a_look_outside_its_shot_is_refused", refused.returncode != 0 and "a look on `a` is not" in refused.stdout + refused.stderr, refused.stdout + refused.stderr)

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

        # the render ledger as the inventory: the fixture's four renders, with the region where the ledger could place it
        folder7 = Path(tempfile.mkdtemp(dir=d))
        paths = fixture(folder7, renders=False)

        def led(name, first, last, replace, region, motion="none"):
            return {"file": f"Video/x/{name}/{name}_00001.mp4", "name": name, "span": [first, last], "graph": {"replace": replace, "motion_reference": motion},
                    "form": {"region": region}}
        ledger = {"schema": "h3_render_ledger/2", "renders": {r["file"]: r for r in [
            led("a_one", 0, 29, "the wired parts", "face"),                       # the plan says face and hair for a: parts, and not the plan's parts
            led("b_two", 30, 59, "the wired parts", "face"),                      # the plan says face for b: as planned
            led("a_three", 30, 44, "the wired parts", "a part"),                  # not placed by the ledger: held to the graph's kind
            led("a_spare", 0, 29, "whole subject", "whole", "a video I wire, zoomed in"),
            {"file": "Video/x/a_lost/a_lost_00001.mp4", "name": "a_lost", "span": None, "graph": {}, "form": {}}]}}
        paths["ledger"] = write(folder7, "ledger.json", ledger)
        got = rows_of(paths, folder7)
        text = run(paths).stdout
        a0, b30, a30 = row(got, 0, "a"), row(got, 30, "b"), row(got, 30, "a")
        case("the_ledger_names_the_region_itself",
             a0["state"] == "not as planned" and a0["laid"][0]["region_exact"] == "face" and b30["state"] == "as planned"
             and a30["state"] == "partly, as planned" and a30["laid"][0]["region_exact"] is None and a30["laid"][0]["region"] == "the wired parts"
             and [(o["name"], o["region"], o["motion"]) for o in a0["others"] if not o["in_table"]] == [("a_spare", "whole", True)]
             and "a_lost" not in json.dumps(got) and "holds a piece to the plan's region itself where the ledger names it" in text
             and "region: face  subject" in text and got["read_from"]["ledger"].endswith("ledger.json"),
             json.dumps([a0["state"], a0["laid"], b30["state"], a30["state"], a30["laid"], a0["others"]])[:900])
        both = run({**paths, "renders": write(folder7, "renders.json", {"renders": []})})
        wrong = run({**paths, "ledger": write(folder7, "not_a_ledger.json", {"schema": "h3_shots/1", "renders": {}})})
        case("two_inventories_are_refused", both.returncode != 0 and "two inventories of the same renders" in both.stdout + both.stderr
             and wrong.returncode != 0 and "this is not bench/render_ledger.py's --json" in wrong.stdout + wrong.stderr, (both.stdout + both.stderr + wrong.stdout + wrong.stderr)[-400:])

        # the gap check's JSON named by the table (assembly.stack.coverage_json, verified by table_sha256 and clip_check):
        # worst-first remake and capture lists, the 4 other run kinds, failures, per-row frame-subjects and states where the
        # original shows, refused when missing, malformed, or stale
        folder9 = Path(tempfile.mkdtemp(dir=d))
        paths9 = fixture(folder9)
        by_shot9 = {
            "remake": [
                {"shot": [30, 59], "frames": 15, "subjects": ["a"],
                 "by_kind": {"original_while_tracked": 10, "emptied_inside": 2, "laid_but_barely_changed": 2, "region_far_below_its_own_size": 1}},
                {"shot": [0, 29], "frames": 3, "subjects": ["a"],
                 "by_kind": {"original_while_tracked": 0, "emptied_inside": 1, "laid_but_barely_changed": 1, "region_far_below_its_own_size": 1}},
            ],
            "capture": [
                {"shot": [60, 99], "frames": 40, "subjects": ["b"], "by_kind": {"original_tracking_not_known": 40}},
            ],
        }
        runs9 = [
            {"first": 45, "last": 54, "shot": [30, 59], "subject": "a", "kind": "original_while_tracked", "pieces": [], "cause": "no piece laid"},
            {"first": 35, "last": 36, "shot": [30, 59], "subject": "a", "kind": "emptied_inside", "pieces": ["a_three_00001.mp4"], "cause": "floor"},
            # duplicate [35, 36] span on a for another kind: must be listed once, never repeated
            {"first": 35, "last": 36, "shot": [30, 59], "subject": "a", "kind": "laid_but_barely_changed", "pieces": ["a_three_00001.mp4"], "cause": "0.120 to 0.210"},
            {"first": 39, "last": 39, "shot": [30, 59], "subject": "a", "kind": "region_far_below_its_own_size", "pieces": ["a_three_00001.mp4"], "cause": "carried 400 px"},
            # first=50 overlaps [30, 59], but shot=[60, 99] restricts it to [60, 99] (tests SC6 shot-equality filter)
            {"first": 50, "last": 99, "shot": [60, 99], "subject": "b", "kind": "original_tracking_not_known", "pieces": [], "cause": "no tied capture"},
            # 4 other run kinds written by frame_coverage.py
            {"first": 30, "last": 31, "shot": [30, 59], "subject": "a", "kind": "carried_not_known", "pieces": ["a_three_00001.mp4"], "cause": "no region"},
            {"first": 10, "last": 12, "shot": [0, 29], "subject": "a", "kind": "original_her_track_empty", "pieces": [], "cause": "0 px"},
            {"first": 13, "last": 14, "shot": [0, 29], "subject": "a", "kind": "left_out_by_a_look", "pieces": [], "cause": "look"},
            {"first": 15, "last": 15, "shot": [0, 29], "subject": "a", "kind": "seam", "pieces": ["a_one_00001.mp4"], "cause": "seam"},
        ]
        t_json, cov_file, clip_check = sync_gap_check(folder9, by_shot=by_shot9, runs=runs9)
        got9 = rows_of(paths9, folder9)
        txt9 = run(paths9).stdout
        a0_9, a30_9, b30_9, b60_9 = row(got9, 0, "a"), row(got9, 30, "a"), row(got9, 30, "b"), row(got9, 60, "b")
        # failures (strings as frame_coverage writes), full-shot gap state ("original shows", N18),
        # multi-subject spans per subject, and multi-subject fallback guard (:363)
        sync_gap_check(
            folder9,
            by_shot={
                "remake": [
                    {"shot": [0, 29], "frames": 30, "subjects": ["a"], "by_kind": {"emptied_inside": 30}},
                    {"shot": [30, 59], "frames": 4, "subjects": ["a", "b"], "by_kind": {"original_while_tracked": 4}},
                ],
                "capture": [
                    {"shot": [60, 99], "frames": 20, "subjects": ["a", "b"], "by_kind": {"original_tracking_not_known": 20}},
                ],
            },
            runs=[
                {"first": 0, "last": 29, "shot": [0, 29], "subject": "a", "kind": "emptied_inside", "pieces": ["a_one_00001.mp4"], "cause": "all"},
                {"first": 60, "last": 69, "shot": [60, 99], "subject": "a", "kind": "original_tracking_not_known", "pieces": [], "cause": "no cap"},
                {"first": 70, "last": 79, "shot": [60, 99], "subject": "b", "kind": "original_tracking_not_known", "pieces": [], "cause": "no cap"},
            ],
            failures=["0-29 a: original_while_tracked on 2 frame(s)"],
        )
        got_fail9 = rows_of(paths9, folder9)
        txt_fail9 = run(paths9).stdout
        # span-limited table with its own matching gap check (reading=[], rows=[0..44], span=(0, 44) while shots.json is [0, 99])
        (folder9 / "t.txt").write_text("0-29 /share/Video/x/a_one/a_one_00001.mp4 0\n30-44 /share/Video/x/a_three/a_three_00001.mp4 30\n")
        sync_gap_check(folder9, span=(0, 44), reading_override=[])
        span_limited_ok9 = run(paths9)
        # restore full t.txt
        fixture(folder9)
        # shape refusal (SC10)
        sync_gap_check(folder9, by_shot={"remake": "bad"}, runs=[])
        bad_shape9 = run(paths9)
        # span mismatch refusal
        sync_gap_check(folder9, span=(0, 59), reading_override=[{"first": 0, "last": 99}], by_shot=by_shot9, runs=runs9)
        bad_span9 = run(paths9)
        # missing clip_check dict in gap check (N14)
        sync_gap_check(folder9, by_shot=by_shot9, runs=runs9, clip_check_override={})
        no_cc_dict9 = run(paths9)
        # missing --out in assembly.command (Item 6)
        sync_gap_check(folder9, by_shot=by_shot9, runs=runs9, cmd_out=None)
        no_out_cmd9 = run(paths9)
        # --out in assembly.command points at another clip (N16)
        sync_gap_check(folder9, by_shot=by_shot9, runs=runs9, cmd_out=str(folder9 / "other_clip.mp4"))
        wrong_out_cmd9 = run(paths9)
        # t.txt rows do not match t.json rows (Item 6)
        sync_gap_check(folder9, by_shot=by_shot9, runs=runs9, rows_override=[{"first": 0, "last": 10, "file": "other.mp4"}])
        bad_trows9 = run(paths9)
        # clip_check missing or stale refusal
        sync_gap_check(folder9, by_shot=by_shot9, runs=runs9)
        clip_check.unlink()
        no_cc_file9 = run(paths9)
        sync_gap_check(folder9, by_shot=by_shot9, runs=runs9)
        clip_check.write_text(json.dumps({"ok": True, "reassembled": True}))
        stale_clip9 = run(paths9)
        sync_gap_check(folder9, by_shot=by_shot9, runs=runs9)
        # table_sha256 mismatch (SC1)
        t_json.write_text(json.dumps({"assembly": {"stack": {"coverage_json": str(cov_file)}}, "changed": True}))
        stale9 = run(paths9)
        # missing coverage_json key in t.json (SC7)
        t_json.write_text(json.dumps({"assembly": {}}))
        no_key9 = run(paths9)
        # missing gap check file on disk (SC2)
        sync_gap_check(folder9, by_shot=by_shot9, runs=runs9)
        cov_file.unlink()
        missing9 = run(paths9)
        # missing t.json on disk, both without and with --gap-check (SC8 / Item 2)
        sync_gap_check(folder9, by_shot=by_shot9, runs=runs9)
        t_json.unlink()
        no_tjson9 = run(paths9)
        no_tjson_with_arg9 = run(paths9, more=["--gap-check", str(cov_file)])
        case("the_gap_check_is_read_and_verified",
             bool(got9.get("gap_check")) and [x["shot"] for x in got9["gap_check"]["by_shot"]["remake"]] == [[30, 59], [0, 29]]
             and [x["shot"] for x in got9["gap_check"]["by_shot"]["capture"]] == [[60, 99]]
             and a0_9.get("gap_check", {}).get("frames") == 3
             and a0_9["state"] == "original shows on some frames"
             and a30_9.get("gap_check", {}).get("frames") == 15
             and a30_9["state"] == "partly, original shows"
             and a30_9["gap_check"]["by_kind"] == {"original_while_tracked": 10, "emptied_inside": 2, "laid_but_barely_changed": 2, "region_far_below_its_own_size": 1}
             and b30_9.get("gap_check", {}).get("frames") == 0
             and b60_9.get("gap_check", {}).get("frames") == 40 and b60_9["gap_check"]["by_kind"] == {"original_tracking_not_known": 40}
             and got9["frames_by_state"]["a"] == {"original shows": 18, "as planned": 27, "partly, as planned": 15, "presence unsure": 40}
             and row(got_fail9, 0, "a")["state"] == "original shows"
             and row(got_fail9, 30, "a")["gap_check"]["frames"] == 0
             and "Where the original still shows" in txt9
             and txt9.index("30-59 (a): 15 frame-subject(s)") < txt9.index("0-29 (a): 3 frame-subject(s)")
             and "30-59 (a): 15 frame-subject(s) (original_while_tracked: 10, emptied_inside: 2, laid_but_barely_changed: 2, region_far_below_its_own_size: 1) on a: 35-36, 39, 45-54" in txt9
             and "0-29 (a): 3 frame-subject(s) (emptied_inside: 1, laid_but_barely_changed: 1, region_far_below_its_own_size: 1)" in txt9
             and "60-99 (b): 40 frame-subject(s) (original_tracking_not_known: 40) on b: 50-99" in txt9
             and "Other runs from the gap check" in txt9
             and "carried_not_known (tracked, whether the laid piece carried a mask is not known): 2 frame-subject(s) on 30-31 (a)" in txt9
             and "original_her_track_empty (original, her track is empty (0 px)): 3 frame-subject(s) on 10-12 (a)" in txt9
             and "left_out_by_a_look (left out by a look): 2 frame-subject(s) on 13-14 (a)" in txt9
             and "seam (seam where two laid pieces of the same person meet): 1 seam(s) on 15 (a)" in txt9
             and "captured_not_rendered = captured, not rendered; rendered_not_kept = rendered, not kept; emptied_inside = emptied inside a piece" in txt9
             and "no_capture = no capture" in txt9
             and "other gap-check runs: 30-31 carried_not_known (no region)" in txt9
             and "original shows (15 f: original_while_tracked 10, emptied_inside 2, laid_but_barely_changed 2, region_far_below_its_own_size 1)" in txt9
             and "FAILED GAP CHECK" in txt_fail9 and "0-29 a: original_while_tracked on 2 frame(s)" in txt_fail9
             and "60-99 (a, b): 20 frame-subject(s) (original_tracking_not_known: 20) on a: 60-69; b: 70-79" in txt_fail9
             and "30-59 (a, b): 4 frame-subject(s) (original_while_tracked: 4)" in txt_fail9
             and span_limited_ok9.returncode == 0
             and bad_shape9.returncode != 0 and "by_shot" in bad_shape9.stdout + bad_shape9.stderr and "Traceback" not in bad_shape9.stderr
             and bad_span9.returncode != 0 and "span" in bad_span9.stdout + bad_span9.stderr and "Traceback" not in bad_span9.stderr
             and no_cc_dict9.returncode != 0 and "names no clip_check" in no_cc_dict9.stdout + no_cc_dict9.stderr and "Traceback" not in no_cc_dict9.stderr
             and no_cc_file9.returncode != 0 and "missing on disk" in no_cc_file9.stdout + no_cc_file9.stderr and "Traceback" not in no_cc_file9.stderr
             and no_out_cmd9.returncode != 0 and "names no --out clip" in no_out_cmd9.stdout + no_out_cmd9.stderr and "Traceback" not in no_out_cmd9.stderr
             and wrong_out_cmd9.returncode != 0 and "read clip_check" in wrong_out_cmd9.stdout + wrong_out_cmd9.stderr and "Traceback" not in wrong_out_cmd9.stderr
             and bad_trows9.returncode != 0 and "companion JSON" in bad_trows9.stdout + bad_trows9.stderr and "Traceback" not in bad_trows9.stderr
             and stale_clip9.returncode != 0 and "clip_check sha256" in stale_clip9.stdout + stale_clip9.stderr and "Traceback" not in stale_clip9.stderr
             and stale9.returncode != 0 and "table_sha256" in stale9.stdout + stale9.stderr and "Traceback" not in stale9.stderr
             and no_key9.returncode != 0 and "coverage_json" in no_key9.stdout + no_key9.stderr and "Traceback" not in no_key9.stderr
             and missing9.returncode != 0 and "missing" in missing9.stdout + missing9.stderr and "Traceback" not in missing9.stderr
             and no_tjson9.returncode != 0 and "companion table JSON" in no_tjson9.stdout + no_tjson9.stderr and "Traceback" not in no_tjson9.stderr
             and no_tjson_with_arg9.returncode != 0 and "companion table JSON" in no_tjson_with_arg9.stdout + no_tjson_with_arg9.stderr and "Traceback" not in no_tjson_with_arg9.stderr,
             (got9.get("gap_check"), a0_9.get("state"), a30_9.get("state"), got9.get("frames_by_state"), txt9[-600:]))

        folder10 = Path(tempfile.mkdtemp(dir=d))
        paths10 = fixture(folder10)
        plan10 = json.loads(paths10["plan"].read_text())
        plan10["subjects"]["a"]["recipe"] = {
            "zoom_up_to": 4.0,
            "margin": 64,
            "crop": "cranium",
            "open_mouth": "outlined",
            "closed_mouth": "outlined",
            "by": "owner",
            "why": "standing subject recipe for second",
        }
        plan10["subjects"]["b"]["recipe"] = {
            "zoom_up_to": 2.0,
            "margin": 16,
            "crop": "cranium",
            "open_mouth": "inner outline",
            "closed_mouth": "sculpted",
            "by": "owner",
            "why": "standing subject recipe for third",
        }
        plan10["subjects"]["a"]["shot_overrides"] = [
            {"first": 30, "last": 59, "margin": 120, "by": "lead", "why": "wide arm motion on 30-59"},
        ]
        paths10["plan"].write_text(json.dumps(plan10))
        ok_recipe10 = run(paths10)

        bad_zoom10 = json.loads(json.dumps(plan10))
        bad_zoom10["subjects"]["a"]["recipe"]["zoom_up_to"] = 5.0
        paths10["plan"].write_text(json.dumps(bad_zoom10))
        res_bad_zoom10 = run(paths10)

        bad_ov10 = json.loads(json.dumps(plan10))
        bad_ov10["subjects"]["a"]["shot_overrides"] = [{"first": 60, "last": 30, "why": "backwards"}]
        paths10["plan"].write_text(json.dumps(bad_ov10))
        res_bad_ov10 = run(paths10)

        case(
            "recipe_and_shot_overrides_in_plan_are_validated",
            ok_recipe10.returncode == 0
            and res_bad_zoom10.returncode != 0
            and "zoom_up_to" in res_bad_zoom10.stderr
            and "Traceback" not in res_bad_zoom10.stderr
            and res_bad_ov10.returncode != 0
            and "shot_overrides[0]" in res_bad_ov10.stderr
            and "Traceback" not in res_bad_ov10.stderr,
            (ok_recipe10.returncode, res_bad_zoom10.stderr, res_bad_ov10.stderr),
        )

    print(f"{'FAIL' if fails else 'ok'}: {fails} failing case(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
