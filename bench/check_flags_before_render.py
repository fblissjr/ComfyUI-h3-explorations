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


def test_unlaid_while_tracked_gate() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cap = make_synthetic_capture(root)
        # Write per_frame.json files showing frames 4310..4315 tracked (track_share > 0) while carried_share == 0
        (cap / "subjects" / "lead" / "per_frame.json").write_text(
            json.dumps(
                {
                    "rows": [
                        {"source_frame": f, "track_share": 0.14 if f <= 4320 else 0.0}
                        for f in range(4310, 4330)
                    ]
                }
            )
        )
        (cap / "runs" / "lead_fp_4310" / "per_frame.json").write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "source_frame": f,
                            "subject": "lead",
                            "read": 1,
                            "carried_share": 0.0 if f <= 4315 else 0.12,
                        }
                        for f in range(4310, 4330)
                    ]
                }
            )
        )
        buf = io.StringIO()
        info = fbr.print_capture_report(cap, out=buf)
        printed = buf.getvalue()
        assert info["unlaid_while_tracked"] == {"lead": [[4310, 4315]]}, info
        assert "unlaid_while_tracked (lead): 6 frame(s) [4310-4315] (tracked > 0 px, carried == 0 px)" in printed, printed

        # Without --ack-unlaid WHY (and no peer look in looks.json), --initial WHO refuses
        refused(
            lambda: fbr.record_initial(cap, "gemreader", ["lead_fp_4310"]),
            "unlaid_while_tracked",
            "lead: frames 4310-4315",
            "--ack-unlaid WHY",
        )
        proc = subprocess.run(
            [sys.executable, str(HERE / "flags_before_render.py"), str(cap), "--initial", "gemreader"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 2 and "REFUSED:" in proc.stderr and "unlaid_while_tracked" in proc.stderr and "Traceback" not in proc.stderr, proc.stderr

        # Passing --ack-unlaid WHY records unlaid_while_tracked_ack and passes check_flags_read
        rc = fbr.main([str(cap), "--initial", "gemreader", "--ack-unlaid", "4310-4315 covered by sibling load_b"])
        assert rc == 0
        beside = json.loads((cap / "looks" / "lead_fp_4310.flags_read.json").read_text())
        assert beside["unlaid_while_tracked"] == {"lead": [[4310, 4315]]}
        assert beside["unlaid_while_tracked_ack"]["why"] == "4310-4315 covered by sibling load_b"
        checked = fbr.check_flags_read(cap, "lead_fp_4310")
        assert checked["unlaid_while_tracked_ack"]["why"] == "4310-4315 covered by sibling load_b", checked

        # If unlaid_while_tracked_ack is stripped and there is no peer look, check_flags_read refuses
        del beside["unlaid_while_tracked_ack"]
        (cap / "looks" / "lead_fp_4310.flags_read.json").write_text(json.dumps(beside))
        shown = json.loads((cap / "looks" / "lead_fp_4310.shown.json").read_text())
        del shown["flags_read"]["unlaid_while_tracked_ack"]
        (cap / "looks" / "lead_fp_4310.shown.json").write_text(json.dumps(shown))
        refused(lambda: fbr.check_flags_read(cap, "lead_fp_4310"), "unlaid_while_tracked", "--ack-unlaid WHY")

        # When looks.json already holds a valid peer look later than status.json, --initial WHO without --ack-unlaid succeeds
        (cap / "looks.json").write_text(
            json.dumps(
                {
                    "looks": [
                        {
                            "by": "claude_peer",
                            "at": "2026-10-24T12:00:00",
                            "loads": ["lead_fp_4310"],
                        }
                    ]
                }
            )
        )
        rc2 = fbr.main([str(cap), "--initial", "gemreader"])
        assert rc2 == 0
        checked2 = fbr.check_flags_read(cap, "lead_fp_4310")
        assert checked2["unlaid_while_tracked_ack"]["covered_by_look"] == "claude_peer", checked2
    return "prints unlaid_while_tracked in [4], refuses --initial without --ack-unlaid or peer look, and verifies unlaid_while_tracked_ack in check_flags_read"


def test_recipe_check_and_mouthless_reference() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cap = make_synthetic_capture(root)
        job_dir = cap.with_name(cap.name + "_job")
        job_dir.mkdir(parents=True)
        # 1. Lead/third recipe (closed_mouth='sculpted', zoom_up_to=2.0, margin=16, open_mouth='outlined') matching this load
        job_plan = job_dir / "job.plan.json"
        job_plan.write_text(
            json.dumps(
                {
                    "job": {
                        "name": "lead_fp_4310",
                        "recipe": {
                            "margin": 16,
                            "zoom_up_to": 2.0,
                            "crop": "cranium",
                            "open_mouth": "outlined",
                            "closed_mouth": "sculpted",
                            "by": "owner",
                            "why": "lead standing recipe",
                        },
                        "reference": {
                            "form": "drawn",
                            "crop": {"form": "cranium", "why": "lead standing recipe"},
                            "mouth": {
                                "form": "auto",
                                "closed_lip_line": {"on": True},
                                "open_mouth": {"form": "outlined", "why": "lead standing recipe"},
                                "closed_mouth": {"form": "sculpted", "why": "lead standing recipe"},
                            },
                        },
                    },
                    "defaults": {"margin": 16},
                    "loads": [{"name": "lead_fp_4310"}],
                }
            )
        )
        buf = io.StringIO()
        info = fbr.print_capture_report(cap, out=buf)
        printed = buf.getvalue()
        assert "[0] recipe check (knob | recipe | this load | status | why; unexplained=0):" in printed, printed
        assert "closed_mouth | sculpted | sculpted" in printed, printed
        assert fbr.main([str(cap), "--initial", "gemreader"]) == 0

        # 2. Second recipe (zoom_up_to=4.0, margin=64, closed_mouth='outlined') where load has unexplained zoom_up_to=2.0
        job_plan.write_text(
            json.dumps(
                {
                    "job": {
                        "name": "lead_fp_4310",
                        "recipe": {
                            "margin": 64,
                            "zoom_up_to": 4.0,
                            "crop": "cranium",
                            "open_mouth": "outlined",
                            "closed_mouth": "outlined",
                            "by": "owner",
                            "why": "second standing recipe",
                        },
                        "reference": {
                            "form": "drawn",
                            "crop": {"form": "cranium", "why": "second standing recipe"},
                            "mouth": {
                                "form": "auto",
                                "closed_lip_line": {"on": True},
                                "open_mouth": {"form": "outlined", "why": "second standing recipe"},
                                "closed_mouth": {"form": "outlined", "why": "second standing recipe"},
                            },
                        },
                    },
                    "defaults": {"margin": 64},  # zoom_up_to missing (defaults to 2.0 -> unexplained against recipe 4.0)
                    "loads": [{"name": "lead_fp_4310"}],
                }
            )
        )
        buf2 = io.StringIO()
        fbr.print_capture_report(cap, out=buf2)
        assert "unexplained=1" in buf2.getvalue(), buf2.getvalue()
        refused(
            lambda: fbr.record_initial(cap, "gemreader", ["lead_fp_4310"]),
            "unexplained recipe deviation(s): zoom_up_to (recipe=4.0, this_load=2.0)",
            "--ack-recipe WHY",
        )
        proc = subprocess.run(
            [sys.executable, str(HERE / "flags_before_render.py"), str(cap), "--initial", "gemreader"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 2 and "REFUSED:" in proc.stderr and "unexplained recipe deviation" in proc.stderr and "Traceback" not in proc.stderr, proc.stderr

        # 3. Mouthless reference when closed_mouth='outlined' has 0 open_frames and 0 frames_drawn_closed
        ref_rec_path = root / "references" / "lead_head_reference_4310_4389.json"
        ref_rec = json.loads(ref_rec_path.read_text())
        ref_rec["drawn"] = {
            "mouth": {
                "closed_lip_line": {
                    "open_frames": [],
                    "closed_mouth": {"form": "outlined", "frames_drawn_closed": []},
                }
            }
        }
        ref_rec_path.write_text(json.dumps(ref_rec))
        # Fix zoom_up_to in job_plan so only mouthless_reference fires
        jd_fixed = json.loads(job_plan.read_text())
        jd_fixed["defaults"]["zoom_up_to"] = 4.0
        jd_fixed["defaults"]["zoom_up_to_by"] = {"why": "second standing recipe", "by": "the plan"}
        job_plan.write_text(json.dumps(jd_fixed))
        buf3 = io.StringIO()
        fbr.print_capture_report(cap, out=buf3)
        assert "mouthless_reference (lead_head_reference_4310_4389.json)" in buf3.getvalue(), buf3.getvalue()
        refused(
            lambda: fbr.record_initial(cap, "gemreader", ["lead_fp_4310"]),
            "mouthless_reference",
            "lead_head_reference_4310_4389.json",
            "--ack-recipe WHY",
        )
        assert fbr.main([str(cap), "--initial", "gemreader", "--ack-recipe", "verified silent back-turned span"]) == 0
        assert fbr.check_flags_read(cap, "lead_fp_4310")["by"] == "gemreader"
    return "prints [0] recipe check table, flags mouthless_reference in [2], and refuses unexplained recipe deviations or mouthless_reference without --ack-recipe WHY"


def test_unmasked_diagnostics_and_unresolved_hair_in_hole_gate() -> str:
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cap = make_synthetic_capture(root)
        # 1. Authoritative region.npz over stale frame_table.jsonl:
        # Suppose frame_table.jsonl claims frames 4312 and 4315 have loaded_part_px == 0,
        # while runs/lead_fp_4310/region.npz has carried > 0 on 4315 and carried == 0 only on 4312.
        ft_lines = [
            json.dumps({"what": "frame_table", "subject": "lead", "run": "lead_fp_4310"}),
            json.dumps({"source_frame": 4312, "load_frame": 2, "tracked": {"px": 5000}, "loaded_part_px": 0}),
            json.dumps({"source_frame": 4315, "load_frame": 5, "tracked": {"px": 5000}, "loaded_part_px": 0}),
        ]
        (cap / "frame_table.jsonl").write_text("\n".join(ft_lines) + "\n")
        (cap / "subjects" / "lead" / "per_frame.json").write_text(json.dumps({
            "rows": [
                {"source_frame": 4310 + i, "seen_by": "pv", "track_share": 0.02, "track_px": 5000}
                for i in range(10)
            ]
        }))
        manifest = json.loads((cap / "manifest.json").read_text())
        manifest["size"] = [16, 16]
        manifest["frames"] = 10
        manifest["runs"] = [{"name": "lead_fp_4310", "subject": "lead"}]
        (cap / "manifest.json").write_text(json.dumps(manifest))

        carried_bool = np.ones((10, 16, 16), dtype=bool)
        carried_bool[2] = False  # frame 4312 has carried_px == 0; frame 4315 (idx 5) has carried_px == 256
        carried_bool[3, 4:, :] = False  # frame 4313 (idx 3) has carried_px == 64 (< 0.5 * 256 = 128 -> shrunk_mask_frame)
        read_bool = np.ones(10, dtype=bool)
        np.savez(
            cap / "runs" / "lead_fp_4310" / "region.npz",
            carried=np.packbits(carried_bool, axis=-1),
            region=np.packbits(np.ones((10, 16, 16), dtype=bool), axis=-1),
            read=read_bool,
        )

        uwt = fbr.unlaid_while_tracked_of(cap, ["lead_fp_4310"])
        assert uwt == {"lead": [[4312, 4312]]}, f"expected region.npz to override stale frame_table on 4315, got {uwt}"

        # 2. Add head_tube__pv.json, part_grades.json, and unresolved hair_in_hole__pv.json on frame 4312
        (cap / "subjects" / "lead" / "head_tube__pv.json").write_text(json.dumps({
            "off_tube_suppressed_px": [0, 0, 1850, 420, 0, 0, 0, 0, 0, 0],
            "face_points_in_part": [38, 38, 0, 12, 38, 38, 38, 38, 38, 38],
            "regain_refused": [{"frame": 2, "face_points_on_own_part": 0, "least": 3, "why": "refused"}],
        }))
        pg = json.loads((cap / "subjects" / "lead" / "part_grades.json").read_text())
        pg["side_with_eye_frames"] = [4312, 4313]
        (cap / "subjects" / "lead" / "part_grades.json").write_text(json.dumps(pg))
        (cap / "subjects" / "lead" / "hair_in_hole__pv.json").write_text(json.dumps({
            "lost_asked_on_source_frames": [[4312, 4312]],
            "hair_undecided_on_source_frames": [[4313, 4313]],
            "frames": [
                {"source_frame": 4312, "lost": True, "hair_unclaimed_px": 900, "hair_undecided_px": 0},
                {"source_frame": 4313, "lost": False, "hair_unclaimed_px": 450, "hair_undecided_px": 450},
            ],
        }))

        buf = io.StringIO()
        info = fbr.print_capture_report(cap, out=buf)
        text = buf.getvalue()
        assert "unresolved_hair_in_hole (lead) lost_asked_on_source_frames: frames 4312" in text, text
        assert "unresolved_hair_in_hole (lead) hair_undecided_on_source_frames: frames 4313" in text, text
        assert "tracker_and_mask_gap_table (lead, median_carried=256.0 px, 2 frame(s) empty or <50% median):" in text, text
        assert "unmasked_frame (lead) 4312: carried=0 px (tube=0 px, raw_part=0 px), track=5000 px, off_tube_suppressed=1850 px, pose=side_with_eye" in text, text
        assert "shrunk_mask_frame (lead) 4313: carried=64 px (tube=0 px, raw_part=0 px), track=5000 px, off_tube_suppressed=420 px, pose=side_with_eye" in text, text
        assert "PAUSE — VISIBLE HEAD/HAIR SIGNALS ON UNMASKED FRAME(S) (lead):" in text, text
        assert info["unresolved_hair_in_hole"] == {"lead": {"lost_asked": [[4312, 4312]], "hair_undecided": [[4313, 4313]]}}, info

        # 3. Both record_initial and check_flags_read refuse while unresolved_hair_in_hole is non-empty,
        # even if --ack-unlaid is passed!
        refused(
            lambda: fbr.record_initial(cap, "gemreader", ["lead_fp_4310"], ack_unlaid="tried to ack"),
            "unresolved_hair_in_hole on 2026-10-24_lead_fp_4310",
            "does NOT recover the mask (0 px in parts_held.mkv)",
            "head_marks",
        )
        # Clear hair_in_hole (simulating rebuild after head_marks in plan.json), record initial, then re-introduce unresolved hair_in_hole to test check_flags_read
        (cap / "subjects" / "lead" / "hair_in_hole__pv.json").write_text(json.dumps({
            "lost_asked_on_source_frames": [],
            "hair_undecided_on_source_frames": [],
            "frames": [],
        }))
        fbr.record_initial(cap, "gemreader", ["lead_fp_4310"], ack_unlaid="verified occluded frame", out=io.StringIO())
        assert fbr.check_flags_read(cap, "lead_fp_4310")["by"] == "gemreader"

        (cap / "subjects" / "lead" / "hair_in_hole__pv.json").write_text(json.dumps({
            "lost_asked_on_source_frames": [[4312, 4312]],
            "hair_undecided_on_source_frames": [],
            "frames": [{"source_frame": 4312, "lost": True, "hair_unclaimed_px": 900, "hair_undecided_px": 0}],
        }))
        refused(
            lambda: fbr.check_flags_read(cap, "lead_fp_4310"),
            "unresolved_hair_in_hole on 2026-10-24_lead_fp_4310",
            "does NOT recover the mask (0 px in parts_held.mkv)",
        )
    return "region.npz is authoritative over frame_table.jsonl, per-frame unmasked diagnostics print PAUSE banner, and unresolved_hair_in_hole refuses --initial and check_flags_read"


case("flags_before_render: 4 diagnostic blocks, --initial record, and refusals", test_print_and_initial)
case("flags_before_render: unlaid_while_tracked gate (--ack-unlaid WHY or peer look)", test_unlaid_while_tracked_gate)
case("flags_before_render: [0] recipe check and mouthless_reference gate (--ack-recipe WHY)", test_recipe_check_and_mouthless_reference)
case("flags_before_render: region.npz authority, unmasked frame diagnostics, and unresolved_hair_in_hole gate", test_unmasked_diagnostics_and_unresolved_hair_in_hole_gate)
case("flag_census: rule counts, key-based override matching, and blocked/changed dates", test_flag_census)

if __name__ == "__main__":
    bad = [c for c in CASES if not c[0]]
    print(f"\n{'all' if not bad else len(bad)} {len(CASES)} case(s) {'ok' if not bad else 'FAILED'}")
    sys.exit(1 if bad else 0)
