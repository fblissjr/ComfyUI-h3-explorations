#!/usr/bin/env python3
"""Check `bench/flags_before_render.py` and `bench/flag_census.py` without the server or the GPU.

Usage:
    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_flags_before_render.py
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import flag_census as fc  # noqa: E402
import flags_before_render as fbr  # noqa: E402

CASES: list[tuple[bool, str, str]] = []


def case(name: str, fn) -> None:
    try:
        why = fn()
        CASES.append((True, name, why or ""))
        print(f"ok    {name}" + (f": {why}" if why else ""))
    except Exception as e:
        CASES.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"FAIL  {name}: {type(e).__name__}: {e}")


def refused(fn, *words: str) -> str:
    try:
        fn()
    except fbr.Refused as e:
        assert all(w in str(e) for w in words), f"refused without saying {words}: {e}"
        return str(e)
    raise AssertionError(f"not refused; expected {words}")


def make_synthetic_capture(root: Path) -> Path:
    cap = root / "data" / "2026-10-24_lead_fp_4310"
    (cap / "subjects" / "lead").mkdir(parents=True, exist_ok=True)
    (cap / "runs" / "lead_fp_4310").mkdir(parents=True, exist_ok=True)
    (cap / "looks").mkdir(parents=True, exist_ok=True)

    ref_dir = root / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_video = ref_dir / "lead_head_reference_4310_4389.mp4"
    ref_video.write_bytes(b"video")
    ref_record = ref_dir / "lead_head_reference_4310_4389.json"
    ref_record.write_text(
        json.dumps(
            {
                "passed": True,
                "marks_against_the_body_model": {
                    "known": True,
                    "read_from": str(ref_dir / "pose_table.json"),
                    "flagged_count": 2,
                    "flagged_frames": [
                        {
                            "frame": 4370,
                            "nose_2d_vs_3d_fh": 0.431,
                            "head_pitch_down_3d": 0.892,
                            "reasons": ["head_pitch_down_3d=0.892>=0.8"],
                        },
                        {
                            "frame": 4371,
                            "nose_2d_vs_3d_fh": 0.562,
                            "head_pitch_down_3d": 0.871,
                            "reasons": ["nose_2d_vs_3d_fh=0.562>=0.5", "head_pitch_down_3d=0.871>=0.8"],
                        },
                    ],
                },
                "face_on_the_track": {
                    "known": True,
                    "frames_whose_face_middle_is_off_the_track": [4370],
                    "frames_filled_from_their_neighbours": {"4369": 4368},
                    "frames_off_the_track_not_filled": [4370, 4371],
                    "frames_off_the_track_with_no_face_drawn": [4388, 4389],
                },
            }
        )
    )

    (cap / "manifest.json").write_text(
        json.dumps(
            {
                "first_frame": 4310,
                "frames": 80,
                "runs": [{"name": "lead_fp_4310", "subject": "lead", "others": ["strangers"], "carried_from_the_node": True}],
                "subjects": [{"label": "lead", "sightings": [{"by": "pv"}]}],
            }
        )
    )
    (cap / "status.json").write_text(json.dumps({"written": "2026-10-24T11:34:41", "made_by": "gemlead"}))
    (cap / "flags.json").write_text(
        json.dumps(
            {
                "verdict": "clear",
                "blocking": [],
                "overridden": ["f001"],
                "flags": [
                    {
                        "id": "f001",
                        "rule": "load_lays_nothing_on_frames",
                        "level": "likely to fail",
                        "subject": "lead",
                        "source_frames": [[4388, 4389]],
                        "figures": {"frames": 2},
                        "why": "lead has 0 px on 4388-4389",
                    },
                    {
                        "id": "f002",
                        "rule": "part_held_from_a_neighbour",
                        "level": "likely fine",
                        "subject": "lead",
                        "source_frames": [[4367, 4370]],
                        "figures": {"frames": 4, "longest_run": 4},
                        "why": "lead's part held on 4 frames",
                    },
                ],
                "held_frames": {
                    "lead": [
                        {"subject": "lead", "source_frame": 4367, "from_source_frames": [4366]},
                        {"subject": "lead", "source_frame": 4368, "from_source_frames": [4366]},
                    ]
                },
                "emptied_by_the_grade": {"lead": {"frames": 2, "source_frames": [[4388, 4389]]}},
                "emptied_by_the_floor": {},
                "doubted_frames": [],
                "told_by_the_job": {"named_empty": {"lead": [[4388, 4389, "the job's drop="]]}},
            }
        )
    )
    (cap / "outcomes.json").write_text(
        json.dumps(
            {
                "outcomes": [
                    {
                        "flag": "f001",
                        "rule": "load_lays_nothing_on_frames",
                        "subject": "lead",
                        "source_frames": [[4388, 4389]],
                        "key": fc.flag_key({"rule": "load_lays_nothing_on_frames", "subject": "lead", "source_frames": [[4388, 4389]]}),
                        "outcome": "overridden",
                        "overridden": True,
                        "by": "gemlead",
                        "at": "2026-10-24T11:35:00",
                        "note": "edge frames 4388-4389 intentionally bare",
                    }
                ]
            }
        )
    )
    (cap / "looks" / "lead_fp_4310.shown.json").write_text(
        json.dumps(
            {
                "load": "lead_fp_4310",
                "frames": [4310, 4389],
                "motion": [{"file": str(ref_video), "first": 0, "sha256": "abc"}],
            }
        )
    )
    (cap / "runs" / "lead_fp_4310" / "step_carves.json").write_text(
        json.dumps(
            {
                "run": "lead_fp_4310",
                "subject": "lead",
                "others": ["strangers"],
                "grid": {"steps": 21, "latent_h": 48, "latent_w": 84},
                "totals": {"strangers": 8},
                "totals_touching_the_head": {"strangers": 0},
                "steps": [
                    {
                        "window": 1,
                        "step": 18,
                        "frames": [60, 63],
                        "source_frames": [4370, 4373],
                        "grown_units": 245,
                        "final_units": 241,
                        "carved_units_total": 4,
                        "carved_by_other": {"strangers": 4},
                        "carved_touching_head_units": 0,
                    },
                    {
                        "window": 1,
                        "step": 19,
                        "frames": [64, 67],
                        "source_frames": [4374, 4377],
                        "grown_units": 259,
                        "final_units": 255,
                        "carved_units_total": 4,
                        "carved_by_other": {"strangers": 4},
                        "carved_touching_head_units": 0,
                    },
                ],
            }
        )
    )
    (cap / "subjects" / "lead" / "part_grades.json").write_text(
        json.dumps(
            {
                "summary": {"held_from_a_neighbour": 2, "emptied": 2},
                "frames": [
                    {"frame": 57, "source_frame": 4367, "did": "held from a neighbour", "from_source_frames": [4366]},
                    {"frame": 58, "source_frame": 4368, "did": "held from a neighbour", "from_source_frames": [4366]},
                    {"frame": 78, "source_frame": 4388, "did": "emptied"},
                    {"frame": 79, "source_frame": 4389, "did": "emptied"},
                ],
            }
        )
    )
    return cap


def test_print_and_initial() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cap = make_synthetic_capture(root)
        buf = io.StringIO()
        info = fbr.print_capture_report(cap, out=buf)
        printed = buf.getvalue()
        assert info["loads"] == ["lead_fp_4310"] and info["flags_count"] == 2, info
        for expected in (
            "[1] gate flags (2026-10-24_lead_fp_4310/flags.json: 2 flag(s), blocking=[], overridden=['f001']):",
            "flag f001 [likely to fail] (OVERRIDDEN by gemlead) load_lays_nothing_on_frames (lead): frames 4388-4389",
            "[2] reference marks_against_the_body_model & face_on_the_track (1 reference record(s)):",
            "2 flagged frame(s) [4370-4371]",
            "flagged_frame 4370: nose_2d_vs_3d_fh=0.431, head_pitch_down_3d=0.892",
            "face_on_the_track (lead_head_reference_4310_4389.json): middle_off_track=1 [4370]",
            "off_track_no_face_drawn=2 [4388-4389]",
            "[3] step_carves (1 run file(s)):",
            "8 strangers cells across 2 step(s)",
            "window 1 step 18: frames 4370-4373 | strangers=4, carved_total=4 (strangers=4)",
            "[4] held or emptied frames:",
            "flags.held_frames (lead): frames 4367-4368",
            "flags.emptied_by_the_grade (lead): frames 4388-4389",
            "named_empty (lead): frames 4388-4389 | the job's drop=",
            "part_grades (lead) emptied (2 frame(s)): frames 4388-4389",
            "part_grades (lead) held from a neighbour (2 frame(s)): frames 4367-4368",
        ):
            assert expected in printed, f"missing {expected!r} in:\n{printed}"

        # Before `--initial`, check_flags_read refuses and names flags_before_render.py --initial
        refused(lambda: fbr.check_flags_read(cap, "lead_fp_4310"), "nobody has initialed", "flags_before_render.py", "--initial")
        refused(lambda: fbr.record_initial(cap, "   ", ["lead_fp_4310"]), "non-empty reader name")

        # Record initial via CLI main()
        rc = fbr.main([str(cap), "--initial", "gemreader"])
        assert rc == 0
        shown = json.loads((cap / "looks" / "lead_fp_4310.shown.json").read_text())
        beside = json.loads((cap / "looks" / "lead_fp_4310.flags_read.json").read_text())
        assert shown["flags_read"]["by"] == "gemreader" and beside["by"] == "gemreader"
        checked = fbr.check_flags_read(cap, "lead_fp_4310")
        assert checked["by"] == "gemreader" and checked["at_utc"].endswith("Z"), checked

        # Refusal when capture is rewritten after the initial
        (cap / "status.json").write_text(json.dumps({"written": "2099-01-01T00:00:00", "made_by": "gemlead"}))
        refused(lambda: fbr.check_flags_read(cap, "lead_fp_4310"), "not later than the capture", "earlier capture")
        (cap / "status.json").write_text(json.dumps({"written": "2026-10-24T11:34:41", "made_by": "gemlead"}))

        # Refusal when flags.json changes after the initial
        flags_obj = json.loads((cap / "flags.json").read_text())
        flags_obj["flags"].append({"id": "f003", "rule": "part_in_pieces", "level": "iffy"})
        (cap / "flags.json").write_text(json.dumps(flags_obj))
        refused(lambda: fbr.check_flags_read(cap, "lead_fp_4310"), "flags.json changed since gemreader initialed it")

        # Subprocess refusal exits non-zero with REFUSED and no traceback
        proc = subprocess.run(
            [sys.executable, str(HERE / "flags_before_render.py"), str(root / "nonexistent")],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 2 and "REFUSED:" in proc.stderr and "Traceback" not in proc.stderr, proc.stderr
    return "prints all 4 diagnostic blocks, records initial in shown.json and flags_read.json, and refuses missing or stale initials without traceback"


def test_flag_census() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        # Capture 1: f001 overridden in first, then in on-plan f001 is an iffy flag and f002 is the overridden fail flag
        c1 = data_dir / "2026-10-20_load_a"
        c1.mkdir(parents=True)
        (c1 / "flags.json").write_text(
            json.dumps(
                {
                    "verdict": "clear",
                    "blocking": [],
                    "overridden": ["f002"],
                    "flags": [
                        {
                            "id": "f001",
                            "rule": "segment_inside_region",
                            "level": "iffy",
                            "subject": "lead",
                            "segment": "second.Hair",
                            "source_frames": [[10, 12]],
                        },
                        {
                            "id": "f002",
                            "rule": "load_lays_nothing_on_frames",
                            "level": "likely to fail",
                            "subject": "lead",
                            "source_frames": [[10, 11]],
                        },
                    ],
                }
            )
        )
        (c1 / "outcomes.json").write_text(
            json.dumps(
                {
                    "outcomes": [
                        {
                            "flag": "f001",
                            "rule": "load_lays_nothing_on_frames",
                            "subject": "lead",
                            "source_frames": [[10, 11]],
                            "key": fc.flag_key({"rule": "load_lays_nothing_on_frames", "subject": "lead", "source_frames": [[10, 11]]}),
                            "outcome": "overridden",
                            "overridden": True,
                            "at": "2026-10-20T12:00:00",
                        }
                    ]
                }
            )
        )
        # Capture 2: a blocking flag that was NOT overridden (blocked a render that was then changed)
        c2 = data_dir / "2026-10-23_load_b"
        c2.mkdir(parents=True)
        (c2 / "flags.json").write_text(
            json.dumps(
                {
                    "verdict": "blocked",
                    "blocking": ["f001"],
                    "overridden": [],
                    "flags": [
                        {
                            "id": "f001",
                            "rule": "load_lays_nothing_on_frames",
                            "level": "likely to fail",
                            "subject": "lead",
                            "source_frames": [[50, 55]],
                        },
                        {
                            "id": "f002",
                            "rule": "part_held_from_a_neighbour",
                            "level": "likely fine",
                            "subject": "lead",
                            "source_frames": [[56, 57]],
                        },
                    ],
                }
            )
        )
        res = fc.run_census(data_dir)
        assert res["flags_files_count"] == 2 and res["outcomes_files_count"] == 1
        assert res["total_flags_in_files"] == 4 and res["sum_of_rule_flags"] == 4
        by_rule = {r["rule"]: r for r in res["rows"]}
        r_nothing = by_rule["load_lays_nothing_on_frames"]
        assert (r_nothing["flags"], r_nothing["loads"], r_nothing["likely_to_fail"], r_nothing["overridden"], r_nothing["blocked_changed"]) == (2, 2, 2, 1, 1), r_nothing
        assert r_nothing["last_blocked"] == "2026-10-23" and r_nothing["last_fired"] == "2026-10-23", r_nothing
        r_seg = by_rule["segment_inside_region"]
        assert (r_seg["flags"], r_seg["iffy"], r_seg["overridden"], r_seg["blocked_changed"]) == (1, 1, 0, 0), r_seg
        buf = io.StringIO()
        fc.print_census(res, out=buf)
        assert "load_lays_nothing_on_frames" in buf.getvalue() and "sum of rule flags (4) == total flags in files (4)" in buf.getvalue()
    return "counts every rule across flags.json and outcomes.json, matches overrides by flag key across ID shifts, and reconciles totals"


case("flags_before_render: 4 diagnostic blocks, --initial record, and refusals", test_print_and_initial)
case("flag_census: rule counts, key-based override matching, and blocked/changed dates", test_flag_census)

if __name__ == "__main__":
    bad = [c for c in CASES if not c[0]]
    print(f"\n{'all' if not bad else len(bad)} {len(CASES)} case(s) {'ok' if not bad else 'FAILED'}")
    sys.exit(1 if bad else 0)
