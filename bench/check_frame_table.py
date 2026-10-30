"""Verify bench/frame_table.py on a synthetic capture folder.

Covers:
1. Outcomes and causes using the capture's real closed `did` vocabulary:
   replaced, held (with from_frame), dropped (with reason),
   emptied for each cause ('floor', 'grade', 'nothing saved'), and not_in_load.
2. Top-of-table fields: sam_inputs (including MiniMaxH3SAM31Corrections and others wiring),
   in_view_spans, out_of_view_spans, in_the_shot_by_the_owner, owner_says_in_shot_track_empty,
   flag_spans, and outcome_counts/outcome_spans.
3. CLI entrypoint writes frame_table.jsonl and frame_table.md cleanly.
4. Missing mask file (track__<by>.mkv or parts_held__<by>.mkv) is red (Refused).
5. Dropped frame without `drop_by` reason in job file is red (Refused), with no fallback to `note`.
6. Missing load frame row in per_frame.csv is red (Refused).
7. D10-1: Closed `did` vocabulary: grade-emptied frame where `word_not_held` has no `least_px`
   is `emptied` (`cause='grade'`), never `replaced`; unknown `did` is refused naming frame and value.
8. D10-2: Flags are span-bridged in header (`flag_spans`) and per-frame field is
   `inside_flag_spans` (`["inside the span of fNNN"]`), never exact per-frame `flags`.
9. D10-3: `in_view` reads `shots__<by>.json` (`frames_without_subject`, `frames_on_others`)
   and `track_px` with `in_view_from`, never comparing `track_score` with `detection_threshold`.
10. D10-4: When `part_grades.json` is absent on a part load, non-dropped load frames get
    outcome `'no per-frame record'` rather than reading bridged manifest spans as exact.
11. D10-6: Disagreement between job `drop=` and capture `parts_emptied_on_source_frames` is refused naming both.
12. D10-10: Size columns separate `loaded_part_px`, `saved_part_px`, `saved_part_height_px` and
    state what `part_px` and `part_height_px` measure on each outcome.
13. D10-11: No silent fallbacks: follows graph wiring to the tracker's own CheckpointLoaderSimple
    and MiniMaxH3SAM31Corrections, reports `'not answered'` when owner presence is absent in shots.json,
    refuses missing `size` / `shots_first_source_frame` / `job['preview']`, and supports a whole-person
    track-only load (`sighting.parts is None`) and pair second subject (`--subject`).
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import case, finish
import frame_table
from workflows import h3_config


def _read_table(jsonl_path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[0], lines[1:]


def _make_synthetic_tree(root: Path) -> tuple[Path, Path, Path]:
    """Build a synthetic capture directory covering all outcomes and causes using real capture strings.

    Shot span: 100..108 (9 frames)
    Capture & Load span: 101..106 (6 frames, load_frame 0..5)
    left_out in job: 100..100 and 107..108 -> not_in_load (with reason)

    Load frames:
    - 101 (lf 0): replaced (`did="saved part kept, fill refused"`, saved_part_px=2400, saved_part_height_px=95, tracked=12000 px)
    - 102 (lf 1): held from frame 0 -> source_frame 101 (`did="filled on a person's word"`, loaded_part_px=2350, px_on_the_track=2100, saved_part_px=0, saved_part_height_px=0)
    - 103 (lf 2): emptied cause='floor' (`did="left empty: the word not held"`, least_px=480, px=180, why="too small to be drawn as the part")
    - 104 (lf 3): emptied cause='grade' (`did="emptied"`, saved_on_its_classes=0.12, fill_on_its_classes=0.08, saved_part_px=1500)
    - 105 (lf 4): emptied cause='nothing saved' (not in part_grades.json frames, saved_part_px=0)
    - 106 (lf 5): dropped with reason (in parts_emptied_on_source_frames [[106, 106]], drop="106-106", drop_by='hand crosses face on 106')

    SAM 3.1 visibility in shot 100..108 (shots__shot_s_0100a.json, shots_first_source_frame=100):
    - 100 (rel 0): in `frames_without_subject` [[0, 0]], track_score=-1.20 -> in_view=False
    - 101..107 (rel 1..7): with subject (102 seeded so track_score=None; 103 has track_score=0.30 < detection_threshold=0.50 to prove track_score is not compared to detection_threshold) -> in_view=True
    - 108 (rel 8): in `frames_on_others` [[8, 8]], track_score=0.12 -> in_view=False
    Owner shot presence: yes on shot [100, 108] -> owner_says_in_shot_track_empty = [[100, 100], [108, 108]].
    """
    data_dir = root / "data"
    cap = data_dir / "2026-10-16_synth_load_0101a"
    subj_dir = cap / "subjects" / "lead"
    subj_dir.mkdir(parents=True, exist_ok=True)

    # Mask files
    track_mkv = subj_dir / "track__shot_s_0100a.mkv"
    part_mkv = subj_dir / "parts_held__shot_s_0100a.mkv"
    track_mkv.write_bytes(b"SYNTH_TRACK_MKV_BYTES")
    part_mkv.write_bytes(b"SYNTH_PART_MKV_BYTES")

    # Preview directory and preview.json (ComfyUI prompt graph with unwired loader + wired loader + SAM31Corrections + others)
    prev_dir = data_dir / "2026-10-16_shot_s_0100a_job"
    prev_dir.mkdir(parents=True, exist_ok=True)
    preview_graph = {
        "prompt": {
            "0": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "unwired_ decoy_model.safetensors"},
            },
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": h3_config.SEGMENTER},
            },
            "101": {
                "class_type": "MiniMaxH3SAM31Corrections",
                "inputs": {
                    "segmenter": ["1", 0],
                    "segmenter_clip": ["1", 1],
                    "correct_image_range": True,
                    "correct_text_activation": True,
                    "memory_selection": True,
                },
            },
            "1005": {
                "class_type": "MiniMaxH3SubjectTrack",
                "inputs": {
                    "segmenter": ["101", 0],
                    "segmenter_clip": ["101", 1],
                    "subject_phrase": "person",
                    "head_phrase": "head",
                    "pick": "largest",
                    "pick_on": "automatic",
                    "match": "automatic",
                    "detection_threshold": 0.5,
                    "max_people": 16,
                    "cuts": "automatic",
                    "seeds": "frame 2: 10, 10, 50, 50",
                    "corrections": "",
                },
            },
            "1009": {
                "class_type": "MiniMaxH3SaveShotTable",
                "inputs": {
                    "filename_prefix": "shot_s_0100a_second_shots",
                    "shot_table": ["1005", 3],
                },
            },
            "2": {
                "class_type": "MiniMaxH3SubjectTrack",
                "inputs": {
                    "segmenter": ["101", 0],
                    "segmenter_clip": ["101", 1],
                    "subject_phrase": "person",
                    "head_phrase": "head",
                    "pick": "largest",
                    "pick_on": "automatic",
                    "match": "automatic",
                    "detection_threshold": 0.5,
                    "max_people": 16,
                    "cuts": "automatic",
                    "seeds": "frame 2: 200, 180, 220, 240",
                    "corrections": "frame 4: add 250, 210",
                    "others": ["1005", 0],
                },
            },
            "3": {
                "class_type": "MiniMaxH3SaveShotTable",
                "inputs": {
                    "filename_prefix": "shot_s_0100a_lead_shots",
                    "shot_table": ["2", 3],
                },
            },
        }
    }
    (prev_dir / "preview.json").write_text(json.dumps(preview_graph, indent=2) + "\n", encoding="utf-8")

    # shots__shot_s_0100a.json
    shots_table = {
        "table": "h3 shot table",
        "version": 2,
        "track_score_is": "the tracker's own score that its object is on the frame, per frame of the shot, as the tracker's number: over 0 it takes the object as present.",
        "shots": [
            {
                "shot": 1,
                "first_frame": 0,
                "last_frame": 8,
                "frames": 9,
                "shown_frame": 2,
                "seeded": {"frame": 2, "box": [200.0, 180.0, 220.0, 240.0], "person": 2},
                "subject": {
                    "person": 2,
                    "state": "taken (seeded)",
                    "why": "seeded: the detection the box names on frame 2 (overlap 1.00)",
                },
                "frames_with_subject": 7,
                "frames_without_subject": [[0, 0]],
                "frames_on_others": [[8, 8]],
                "track_score": [-1.20, 5.10, None, 0.30, 5.40, 5.50, 5.60, 4.80, 0.12],
            }
        ],
    }
    (subj_dir / "shots__shot_s_0100a.json").write_text(json.dumps(shots_table, indent=2) + "\n", encoding="utf-8")

    # Owner shots.json and clip_folder manifest.json
    fun_dir = data_dir / "fun"
    fun_dir.mkdir(parents=True, exist_ok=True)
    (fun_dir / "manifest.json").write_text(
        json.dumps({"schema": "h3_clip_analysis", "version": 1, "masks": [], "source": {"name": "fun.mp4"}}, indent=2) + "\n",
        encoding="utf-8",
    )
    shots_json = fun_dir / "shots.json"
    shots_obj = {
        "shots": [
            {
                "id": 11,
                "first": 100,
                "last": 108,
                "who": {
                    "lead": {
                        "present": "yes",
                        "how": "frame 102 boxed by owner",
                        "by": "owner, 2026-10-16, on the who page",
                        "at": "2026-10-16T12:00:00Z",
                        "frames_boxed": [102],
                    }
                },
            }
        ]
    }
    shots_json.write_text(json.dumps(shots_obj, indent=2) + "\n", encoding="utf-8")

    # Job file and <capture>_job directory
    job_dir = data_dir / "2026-10-16_synth_load_0101a_job"
    job_dir.mkdir(parents=True, exist_ok=True)
    job_file = job_dir / "job.plan.json"
    job_doc = {
        "job": {
            "source": "fun.mp4",
            "preview": str(prev_dir),
            "first": 101,
            "frames": 6,
            "note": "general job note that must never be used as a drop reason",
            "subjects": [
                {
                    "label": "lead",
                    "drop": "106-106",
                    "drop_by": "hand crosses face on 106",
                    "note": "subject note that must never be used as a drop reason",
                }
            ],
            "left_out": [
                {"subject": "lead", "first": 100, "frames": 1, "why": "100 outside load window: head not yet in frame"},
                {"subject": "lead", "first": 107, "frames": 2, "why": "107-108 outside load window: after cut"},
            ],
        },
        "loads": [
            {
                "name": "synth_load_0101a",
                "first": 101,
                "frames": 6,
            }
        ],
    }
    job_file.write_text(json.dumps(job_doc, indent=2) + "\n", encoding="utf-8")

    who_json = job_dir / "synth_load_0101a.who_is_who.json"
    who_obj = {
        "rows": [
            {
                "note": "a person's word: lead (seeded)",
                "trackers": {
                    "lead": {
                        "called_person": 2,
                        "called": "taken (seeded)",
                        "answered_by": "seeded",
                        "tile_frame": 102,
                    }
                },
            }
        ]
    }
    who_json.write_text(json.dumps(who_obj, indent=2) + "\n", encoding="utf-8")

    # part_grades.json using capture_masked_run.py's exact closed `did` strings
    # Note: frame 105 (untouched with 0 saved px) is omitted from `frames`, matching capture_masked_run.py:766-767
    part_grades_obj = {
        "subject": "lead",
        "seen_by": "shot_s_0100a",
        "least_px": 480,
        "part_sizes": {
            "emptied_by_the_floor": {
                "frames": 2,
                "source_frames": [[103, 105]],
            }
        },
        "frames": [
            {
                "source_frame": 101,
                "did": "saved part kept, fill refused",
                "saved_on_its_classes": 1.0,
                "fill_on_its_classes": 0.0,
            },
            {
                "source_frame": 102,
                "did": "filled on a person's word",
                "saved_on_its_classes": None,
                "fill_on_its_classes": 0.45,
                "px": 2350,
                "from_frames": [0],
                "px_on_the_track": 2100,
                "why": "the job's word= names this frame (a person looked); the grade had emptied it",
            },
            {
                "source_frame": 103,
                "did": "left empty: the word not held",
                "saved_on_its_classes": 1.0,
                "fill_on_its_classes": 0.31,
                "why": "too small to be drawn as the part",
                "px": 180,
                "least_px": 480,
                "word_not_held": "a person's word asks the part held here, and there is nothing to hold it from: no frame of the load has a part, the subject has no track on this frame, or the nearest part, moved, lies wholly off the subject's track. It stays empty",
            },
            {
                "source_frame": 104,
                "did": "emptied",
                "saved_on_its_classes": 0.12,
                "fill_on_its_classes": 0.08,
            },
        ],
    }
    (subj_dir / "part_grades.json").write_text(json.dumps(part_grades_obj, indent=2) + "\n", encoding="utf-8")

    # subjects/lead/per_frame.csv (canvas_area = 1000 * 768 = 768000)
    per_frame_lines = [
        "frame,source_frame,seen_by,track_share,parts_share,parts_height_px",
        f"0,101,shot_s_0100a,{12000/768000:.8f},{2400/768000:.8f},95",
        f"1,102,shot_s_0100a,{12000/768000:.8f},0.0,0",
        f"2,103,shot_s_0100a,{12000/768000:.8f},{180/768000:.8f},22",
        f"3,104,shot_s_0100a,{12000/768000:.8f},{1500/768000:.8f},64",
        f"4,105,shot_s_0100a,{12000/768000:.8f},0.0,0",
        f"5,106,shot_s_0100a,{12000/768000:.8f},{2200/768000:.8f},90",
    ]
    (subj_dir / "per_frame.csv").write_text("\n".join(per_frame_lines) + "\n", encoding="utf-8")

    # top-level frames.csv
    frames_csv_lines = [
        "frame,source_frame,region_on__synth_load_0101a__second,region_cells_holding__synth_load_0101a__second,margin_reached__synth_load_0101a__second,cells_given_back__synth_load_0101a__second",
        "0,101,0.0,0,0.0,0",
        "1,102,0.0125,4,0.005,2",
        "2,103,0.0,0,0.0,0",
        "3,104,0.0,0,0.0,0",
        "4,105,0.0,0,0.0,0",
        "5,106,0.0,0,0.0,0",
    ]
    (cap / "frames.csv").write_text("\n".join(frames_csv_lines) + "\n", encoding="utf-8")

    # flags.json
    flags_obj = {
        "flags": [
            {"id": "f001", "rule": "part_empty", "subject": "lead", "source_frames": [[101, 102]]},
            {"id": "f002", "rule": "word_not_held", "subject": "lead", "source_frames": [[103, 104]]},
        ]
    }
    (cap / "flags.json").write_text(json.dumps(flags_obj, indent=2) + "\n", encoding="utf-8")

    # manifest.json
    manifest_obj = {
        "clip": "fun.mp4",
        "first_frame": 101,
        "frames": 6,
        "size": [1000, 768],
        "runs": [
            {
                "name": "synth_load_0101a",
                "subject": "lead",
                "others": ["second"],
                "carried_is": "the mask each window was given",
                "left_as_the_source_with_no_mask": [105],
            }
        ],
        "subjects": [
            {
                "label": "lead",
                "sightings": [
                    {
                        "by": "shot_s_0100a",
                        "parts": "shot_s_0100a_lead_parts_00001.mkv",
                        "shots_first_source_frame": 100,
                        "parts_held_on_source_frames": [[101, 103]],
                        "parts_emptied_on_source_frames": [[106, 106]],
                        "parts_emptied_by_grade_on_source_frames": [[104, 104]],
                        "part_sizes": {
                            "emptied_by_the_floor": {
                                "least_px": 480,
                                "source_frames": [[103, 103]],
                            }
                        },
                    }
                ],
            }
        ],
    }
    (cap / "manifest.json").write_text(json.dumps(manifest_obj, indent=2) + "\n", encoding="utf-8")
    return cap, subj_dir, job_file


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="h3_check_frame_table_") as tmp:
        root = Path(tmp)
        cap, subj_dir, job_file = _make_synthetic_tree(root)

        wrote = frame_table.build_frame_table(cap)
        header, rows = _read_table(wrote["jsonl"])
        by_sf = {r["source_frame"]: r for r in rows}

        def c_outcomes() -> str:
            assert len(rows) == 9, f"expected 9 rows, got {len(rows)}"
            assert by_sf[100]["outcome"] == "not_in_load" and by_sf[100]["load_frame"] is None and by_sf[100]["tracked"] is False
            assert "outside load window" in by_sf[100].get("reason", "")
            assert by_sf[101]["outcome"] == "replaced" and by_sf[101]["load_frame"] == 0
            assert by_sf[101]["part_px"] == 2400 and by_sf[101]["part_height_px"] == 95
            assert by_sf[101]["tracked"]["px"] == 12000
            assert by_sf[102]["outcome"] == "held" and by_sf[102]["from_frame"] == 101
            assert by_sf[102]["part_px"] == 2350 and by_sf[102]["px_on_the_track"] == 2100 and by_sf[102]["saved_part_px"] == 0
            assert by_sf[102]["others_given_back"]["second"]["cells_given_back"] == 2
            assert by_sf[103]["outcome"] == "emptied" and by_sf[103]["cause"] == "floor"
            assert by_sf[103].get("did") == "left empty: the word not held" and "a person's word asks" in by_sf[103].get("word_not_held", "")
            assert by_sf[104]["outcome"] == "emptied" and by_sf[104]["cause"] == "grade"
            assert by_sf[105]["outcome"] == "emptied" and by_sf[105]["cause"] == "nothing saved" and by_sf[105]["part_px"] == 0
            assert by_sf[106]["outcome"] == "dropped" and by_sf[106]["reason"] == "hand crosses face on 106"
            assert by_sf[107]["outcome"] == "not_in_load" and by_sf[108]["outcome"] == "not_in_load"
            return f"outcomes={[(r['source_frame'], r['outcome'], r.get('cause'), r.get('from_frame')) for r in rows]}"

        case(
            "outcomes and causes (replaced, held+from_frame, emptied[floor/grade/nothing saved], dropped+reason, not_in_load)",
            c_outcomes,
        )

        def c_d10_1_closed_did() -> str:
            pg_path = subj_dir / "part_grades.json"
            orig_pg = json.loads(pg_path.read_text(encoding="utf-8"))
            try:
                # 1a. A grade-emptied frame that a person's word could not hold has no `least_px` and `did="left empty: the word not held"`:
                # must be `emptied` with `cause="grade"`, NEVER `replaced`
                pg_grade_word = copy.deepcopy(orig_pg)
                pg_grade_word["frames"][3] = {
                    "source_frame": 104,
                    "did": "left empty: the word not held",
                    "saved_on_its_classes": 0.12,
                    "fill_on_its_classes": 0.08,
                    "word_not_held": "a person's word asks the part held here, and there is nothing to hold it from",
                }
                pg_path.write_text(json.dumps(pg_grade_word, indent=2) + "\n", encoding="utf-8")
                w = frame_table.build_frame_table(cap)
                _, r_list = _read_table(w["jsonl"])
                r104 = next(r for r in r_list if r["source_frame"] == 104)
                assert r104["outcome"] == "emptied" and r104.get("cause") == "grade", (
                    f"grade-emptied word_not_held frame 104 became outcome={r104['outcome']!r} cause={r104.get('cause')!r}"
                )

                # 1b. Also check `filled` and `filled where nothing was labelled` map to `held`
                pg_fills = copy.deepcopy(orig_pg)
                pg_fills["frames"][0] = {"source_frame": 101, "did": "filled", "saved_on_its_classes": 0.1, "fill_on_its_classes": 0.9}
                pg_fills["frames"][1] = {
                    "source_frame": 102,
                    "did": "filled where nothing was labelled",
                    "saved_on_its_classes": None,
                    "fill_on_its_classes": None,
                    "px": 2100,
                    "why": "the part model labelled nothing under the fill",
                }
                pg_path.write_text(json.dumps(pg_fills, indent=2) + "\n", encoding="utf-8")
                w2 = frame_table.build_frame_table(cap)
                _, r_list2 = _read_table(w2["jsonl"])
                by2 = {r["source_frame"]: r for r in r_list2}
                assert by2[101]["outcome"] == "held" and by2[102]["outcome"] == "held", (by2[101]["outcome"], by2[102]["outcome"])

                # 1b'. A word hold placed in place from a frame a person named (capture_masked_run.HELD_IN_PLACE, 2026-10-23): the row says
                # the frame it was held from; a word hold without it says nothing of it
                pg_in_place = copy.deepcopy(orig_pg)
                pg_in_place["frames"][1]["held_in_place_from"] = 101
                pg_in_place["frames"][2]["held_in_place_from"] = 101      # 103: a hold in place that placed nothing (HELD_IN_PLACE_MISSED)
                pg_path.write_text(json.dumps(pg_in_place, indent=2) + "\n", encoding="utf-8")
                _, r_list_ip = _read_table(frame_table.build_frame_table(cap)["jsonl"])
                r102 = next(r for r in r_list_ip if r["source_frame"] == 102)
                assert r102["outcome"] == "held" and r102["from_frame"] == 101 and r102.get("held_in_place_from") == 101, r102
                r103 = next(r for r in r_list_ip if r["source_frame"] == 103)
                assert r103["outcome"] == "emptied" and r103.get("held_in_place_from") == 101 and "from_frame" not in r103, r103
                assert "held_in_place_from" not in by_sf[102], by_sf[102]

                # 1c. Any `did` outside the capture's closed vocabulary is refused naming the frame and the value
                pg_bad = copy.deepcopy(orig_pg)
                pg_bad["frames"][0] = {"source_frame": 101, "did": "invented_did_value"}
                pg_path.write_text(json.dumps(pg_bad, indent=2) + "\n", encoding="utf-8")
                try:
                    frame_table.build_frame_table(cap)
                except frame_table.Refused as exc:
                    assert "101" in str(exc) and "invented_did_value" in str(exc), f"unexpected Refused: {exc}"
                    return f"104->emptied/grade; unknown did refused: {exc}"
                raise AssertionError("expected Refused on unknown did='invented_did_value'")
            finally:
                pg_path.write_text(json.dumps(orig_pg, indent=2) + "\n", encoding="utf-8")

        case(
            "D10-1: closed did vocabulary: grade-emptied word_not_held is emptied/grade (never replaced), and unknown did is refused",
            c_d10_1_closed_did,
        )

        # Re-read clean baseline after D10-1 restored part_grades.json
        wrote = frame_table.build_frame_table(cap)
        header, rows = _read_table(wrote["jsonl"])
        by_sf = {r["source_frame"]: r for r in rows}

        def c_d10_2_flag_spans() -> str:
            assert "flag_spans" in header, "header lacks 'flag_spans'"
            fspans = {f["id"]: f["source_frames"] for f in header["flag_spans"]}
            assert fspans == {"f001": [[101, 102]], "f002": [[103, 104]]}, f"flag_spans={fspans}"
            assert "flags" not in by_sf[101], "per-frame row still has exact 'flags' field instead of 'inside_flag_spans'"
            assert by_sf[101].get("inside_flag_spans") == ["inside the span of f001"], f"101 inside_flag_spans={by_sf[101].get('inside_flag_spans')}"
            assert by_sf[103].get("inside_flag_spans") == ["inside the span of f002"], f"103 inside_flag_spans={by_sf[103].get('inside_flag_spans')}"
            assert by_sf[105].get("inside_flag_spans") == [], f"105 inside_flag_spans={by_sf[105].get('inside_flag_spans')}"
            return f"header.flag_spans={fspans}, 101={by_sf[101]['inside_flag_spans']}"

        case(
            "D10-2: flags are span-bridged in header (flag_spans) and per-frame field is inside_flag_spans ('inside the span of fNNN')",
            c_d10_2_flag_spans,
        )

        def c_d10_3_in_view() -> str:
            # Frame 103 has track_score=0.30 (< detection_threshold=0.50), but is present in shots__ and per_frame.csv (12000 px):
            # must be in_view=True, never compared with detection_threshold!
            assert by_sf[103]["sam"]["in_view"] is True, f"frame 103 (score 0.30) wrongly marked out of view: {by_sf[103]['sam']}"
            assert "in_view_from" in by_sf[103]["sam"], f"sam dict lacks 'in_view_from': {by_sf[103]['sam']}"
            # Frame 100 is in frames_without_subject [[0, 0]] -> source 100
            assert by_sf[100]["sam"]["in_view"] is False and "frames_without_subject" in by_sf[100]["sam"].get("in_view_from", ""), by_sf[100]["sam"]
            # Frame 108 is in frames_on_others [[8, 8]] -> source 108 (even though track_score=0.12 > 0)
            assert by_sf[108]["sam"]["in_view"] is False and "frames_on_others" in by_sf[108]["sam"].get("in_view_from", ""), by_sf[108]["sam"]
            assert header["in_view_spans"] == [[101, 107]], f"in_view_spans={header['in_view_spans']}"
            assert header["out_of_view_spans"] == [[100, 100], [108, 108]], f"out_of_view_spans={header['out_of_view_spans']}"
            return f"103.in_view={by_sf[103]['sam']['in_view']} ({by_sf[103]['sam']['in_view_from']}); 108.in_view={by_sf[108]['sam']['in_view']} ({by_sf[108]['sam']['in_view_from']})"

        case(
            "D10-3: in_view reads shots__<by>.json (frames_without_subject, frames_on_others) and track_px with in_view_from, never comparing with detection_threshold",
            c_d10_3_in_view,
        )

        def c_d10_4_no_part_grades() -> str:
            pg_path = subj_dir / "part_grades.json"
            saved_pg = pg_path.read_bytes()
            pg_path.unlink()
            try:
                w = frame_table.build_frame_table(cap)
                _, r_list = _read_table(w["jsonl"])
                by_no_pg = {r["source_frame"]: r for r in r_list}
                # Even though manifest.json has bridged `parts_held_on_source_frames: [[101, 103]]`,
                # without part_grades.json non-dropped load frames 101..105 must be 'no per-frame record'
                outcomes_101_105 = [by_no_pg[sf]["outcome"] for sf in range(101, 106)]
                assert all(o == "no per-frame record" for o in outcomes_101_105), f"expected all 'no per-frame record' on 101..105, got {outcomes_101_105}"
                assert by_no_pg[106]["outcome"] == "dropped"
                return f"101..105 outcomes without part_grades.json: {outcomes_101_105}"
            finally:
                pg_path.write_bytes(saved_pg)

        case(
            "D10-4: missing part_grades.json marks non-dropped load frames 'no per-frame record' instead of reading bridged manifest spans",
            c_d10_4_no_part_grades,
        )

        def c_d10_5_drop_by_only() -> str:
            # Job has both `job["note"]` and `subj["note"]` set; removing `drop_by` must still refuse!
            orig_job = json.loads(job_file.read_text(encoding="utf-8"))
            bad_job = copy.deepcopy(orig_job)
            bad_job["job"]["subjects"][0].pop("drop_by", None)
            assert bad_job["job"].get("note") and bad_job["job"]["subjects"][0].get("note")
            job_file.write_text(json.dumps(bad_job, indent=2) + "\n", encoding="utf-8")
            try:
                try:
                    frame_table.build_frame_table(cap)
                except frame_table.Refused as exc:
                    assert "drop_by" in str(exc), f"unexpected Refused message: {exc}"
                    return f"refused as expected even with note present: {exc}"
                raise AssertionError("expected Refused when drop_by is missing even when subject/job note is present")
            finally:
                job_file.write_text(json.dumps(orig_job, indent=2) + "\n", encoding="utf-8")

        case(
            "D10-5: dropped frame without drop_by is red (Refused), with no fallback to subject note or job note",
            c_d10_5_drop_by_only,
        )

        def c_d10_6_drop_disagreement() -> str:
            orig_job = json.loads(job_file.read_text(encoding="utf-8"))
            bad_job = copy.deepcopy(orig_job)
            bad_job["job"]["subjects"][0]["drop"] = "105-105"
            job_file.write_text(json.dumps(bad_job, indent=2) + "\n", encoding="utf-8")
            try:
                try:
                    frame_table.build_frame_table(cap)
                except frame_table.Refused as exc:
                    assert "105" in str(exc) and "106" in str(exc), f"expected both drop lists in Refused message, got: {exc}"
                    return f"refused naming both: {exc}"
                raise AssertionError("expected Refused when job drop= (105) and capture parts_emptied_on_source_frames (106) disagree")
            finally:
                job_file.write_text(json.dumps(orig_job, indent=2) + "\n", encoding="utf-8")

        case(
            "D10-6: disagreement between job drop= and capture parts_emptied_on_source_frames is refused naming both",
            c_d10_6_drop_disagreement,
        )

        # Re-read clean baseline after D10-4..6
        wrote = frame_table.build_frame_table(cap)
        header, rows = _read_table(wrote["jsonl"])
        by_sf = {r["source_frame"]: r for r in rows}

        def c_d10_10_size_columns() -> str:
            # 101 replaced: loaded_part_px=2400, saved_part_px=2400, saved_part_height_px=95
            assert by_sf[101].get("loaded_part_px") == 2400 and by_sf[101].get("saved_part_px") == 2400 and by_sf[101].get("saved_part_height_px") == 95
            assert by_sf[101].get("part_px_measures") == "saved_part (carried)"
            # 102 held: loaded_part_px=2350 (fill), saved_part_px=0, saved_part_height_px=0
            assert by_sf[102].get("loaded_part_px") == 2350 and by_sf[102].get("saved_part_px") == 0 and by_sf[102].get("saved_part_height_px") == 0
            assert by_sf[102].get("part_px_measures") == "held_fill (carried)"
            assert by_sf[102].get("part_height_px_measures") == "saved_part_before_fill"
            # 103 emptied (floor): loaded_part_px=0, saved_part_px=180, part_px=180 (removed_part)
            assert by_sf[103].get("loaded_part_px") == 0 and by_sf[103].get("saved_part_px") == 180 and by_sf[103].get("part_px") == 180
            assert by_sf[103].get("part_px_measures") == "removed_part (carried=0)"
            # 106 dropped: loaded_part_px=0, saved_part_px=2200, part_px=2200 (removed_part)
            assert by_sf[106].get("loaded_part_px") == 0 and by_sf[106].get("saved_part_px") == 2200
            assert by_sf[106].get("part_px_measures") == "removed_part (dropped, carried=0)"
            return f"102(held): loaded={by_sf[102]['loaded_part_px']} saved={by_sf[102]['saved_part_px']}; 103(emptied): loaded={by_sf[103]['loaded_part_px']} removed={by_sf[103]['part_px']}"

        case(
            "D10-10: size columns separate loaded_part_px, saved_part_px, saved_part_height_px and state what part_px measures per outcome",
            c_d10_10_size_columns,
        )

        def c_d10_11_no_silent_fallbacks() -> str:
            sam_in = header["sam_inputs"]
            # Graph has decoy CheckpointLoaderSimple at node "0" ("unwired_ decoy_model.safetensors") and wired node "1" (h3_config.SEGMENTER)
            assert sam_in["model_file"] == h3_config.SEGMENTER, f"picked unwired CheckpointLoaderSimple: {sam_in['model_file']!r}"
            assert sam_in.get("sam31_corrections") == {"node_id": "101", "correct_image_range": True, "correct_text_activation": True,
                                                       "memory_selection": True}, sam_in.get("sam31_corrections")
            assert sam_in.get("others_wired") == {"node_id": "1005", "output_slot": 0}, sam_in.get("others_wired")

            # Unanswered owner presence in shots.json -> "not answered" (never "no")
            shots_json = root / "data" / "fun" / "shots.json"
            orig_shots = json.loads(shots_json.read_text(encoding="utf-8"))
            try:
                no_ans_shots = copy.deepcopy(orig_shots)
                no_ans_shots["shots"][0]["who"] = {}
                shots_json.write_text(json.dumps(no_ans_shots, indent=2) + "\n", encoding="utf-8")
                w = frame_table.build_frame_table(cap)
                h_no_ans, _ = _read_table(w["jsonl"])
                assert h_no_ans["in_the_shot_by_the_owner"]["present"] == "not answered", h_no_ans["in_the_shot_by_the_owner"]
            finally:
                shots_json.write_text(json.dumps(orig_shots, indent=2) + "\n", encoding="utf-8")

            # Missing `size` in manifest.json -> Refused
            manifest_path = cap / "manifest.json"
            orig_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            try:
                no_size = copy.deepcopy(orig_manifest)
                no_size.pop("size", None)
                manifest_path.write_text(json.dumps(no_size, indent=2) + "\n", encoding="utf-8")
                try:
                    frame_table.build_frame_table(cap)
                    raise AssertionError("expected Refused when manifest['size'] is missing")
                except frame_table.Refused as exc:
                    assert "size" in str(exc)
            finally:
                manifest_path.write_text(json.dumps(orig_manifest, indent=2) + "\n", encoding="utf-8")

            return f"wired_model={sam_in['model_file']} corrections={sam_in['sam31_corrections']} others={sam_in['others_wired']}"

        case(
            "D10-11: no silent fallbacks (follows graph wiring to CheckpointLoaderSimple & SAM31Corrections & others, 'not answered' owner presence, refuses missing size)",
            c_d10_11_no_silent_fallbacks,
        )

        def c_d10_12_header_and_naming() -> str:
            sam_in = header["sam_inputs"]
            assert header["schema"] == frame_table.SCHEMA
            assert "owner_says_in_shot_track_empty" in header, "header lacks 'owner_says_in_shot_track_empty'"
            assert "owner_says_in_shot_but_sam_lost_her" not in header, "header still has old 'owner_says_in_shot_but_sam_lost_her' key"
            assert header["owner_says_in_shot_track_empty"] == [[100, 100], [108, 108]]
            assert header["in_the_shot_by_the_owner"]["present"] == "yes"
            assert sam_in["subject_phrase"] == "person" and sam_in["head_phrase"] == "head" and sam_in["pick"] == "largest"
            assert len(sam_in["seeds"]) == 1 and sam_in["seeds"][0]["source_frame"] == 102
            assert len(sam_in["corrections"]) == 1 and sam_in["corrections"][0]["source_frame"] == 104
            assert sam_in["who_is_who"]["tracked_person_id"] == 2
            assert by_sf[102]["sam"]["track_score"] is None and by_sf[102]["sam"]["in_view"] is True
            return f"owner_says_in_shot_track_empty={header['owner_says_in_shot_track_empty']}"

        case(
            "D10-12: top-of-table fields use owner_says_in_shot_track_empty (a fact, not a cause)",
            c_d10_12_header_and_naming,
        )

        def c_cli() -> str:
            rc = frame_table.main([str(cap)])
            jsonl_path = cap / "frame_table.jsonl"
            md_path = cap / "frame_table.md"
            assert rc == 0 and jsonl_path.is_file() and md_path.is_file()
            jsonl_lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            md_text = md_path.read_text(encoding="utf-8")
            assert len(jsonl_lines) == 1 + 9
            assert jsonl_lines[0]["schema"] == frame_table.SCHEMA
            assert [r["source_frame"] for r in jsonl_lines[1:]] == list(range(100, 109))
            assert "`owner_says_in_shot_track_empty`" in md_text
            assert "hand crosses face on 106" in md_text
            return f"rc={rc} jsonl_lines={len(jsonl_lines)} md_bytes={len(md_text)}"

        case(
            "CLI writes frame_table.jsonl (1 header + 9 frame rows) and frame_table.md",
            c_cli,
        )

        def c_missing_mask() -> str:
            track_mkv = subj_dir / "track__shot_s_0100a.mkv"
            saved_bytes = track_mkv.read_bytes()
            track_mkv.unlink()
            try:
                try:
                    frame_table.build_frame_table(cap)
                except frame_table.Refused as exc:
                    assert "track mask file is missing" in str(exc), f"unexpected Refused message: {exc}"
                    return f"refused as expected: {exc}"
                raise AssertionError("expected Refused when track__<by>.mkv is missing")
            finally:
                track_mkv.write_bytes(saved_bytes)

        case("missing mask file is red (Refused)", c_missing_mask)

        def c_missing_load_row() -> str:
            pf_path = subj_dir / "per_frame.csv"
            orig_pf = pf_path.read_text(encoding="utf-8")
            pf_lines = orig_pf.strip().splitlines()
            pf_path.write_text("\n".join(pf_lines[:-1]) + "\n", encoding="utf-8")
            try:
                try:
                    frame_table.build_frame_table(cap)
                except frame_table.Refused as exc:
                    assert "missing row(s) in per_frame.csv" in str(exc), f"unexpected Refused message: {exc}"
                    return f"refused as expected: {exc}"
                raise AssertionError("expected Refused when a load frame row is missing in per_frame.csv")
            finally:
                pf_path.write_text(orig_pf, encoding="utf-8")

        case("missing load frame row in per_frame.csv is red (Refused)", c_missing_load_row)

        def c_d12_2_floor_only_from_row_least_px() -> str:
            # In _make_synthetic_tree, part_sizes.emptied_by_the_floor.source_frames is [[103, 105]] (bridged with join=2),
            # while frame 104 has did="emptied" and NO "least_px". It must be cause="grade", NEVER cause="floor".
            assert by_sf[103]["outcome"] == "emptied" and by_sf[103]["cause"] == "floor", by_sf[103]
            assert by_sf[104]["outcome"] == "emptied" and by_sf[104]["cause"] == "grade", (
                f"grade-emptied frame 104 inside bridged emptied_by_the_floor [[103, 105]] became cause={by_sf[104].get('cause')!r}"
            )
            return f"103.cause={by_sf[103]['cause']} (has least_px); 104.cause={by_sf[104]['cause']} (no least_px, inside bridged [[103, 105]])"

        case(
            "D12-2 & D12-11: cause 'floor' only when 'least_px' in row, never from bridged part_sizes.emptied_by_the_floor",
            c_d12_2_floor_only_from_row_least_px,
        )

        def c_d12_5_header_input_sha256() -> str:
            assert "input_sha256" in header, "header lacks 'input_sha256'"
            in_sha = header["input_sha256"]
            for key in ("manifest", "part_grades", "flags", "job", "shots"):
                assert key in in_sha and isinstance(in_sha[key], dict) and len(in_sha[key].get("sha256", "")) == 64, (
                    f"missing or invalid input_sha256[{key!r}]: {in_sha.get(key)!r}"
                )
            return f"recorded sha256 for {sorted(in_sha.keys())}"

        case(
            "D12-5: header records input_sha256 for manifest, part_grades, flags, job, and shots.json",
            c_d12_5_header_input_sha256,
        )

        def c_d12_7_filled_no_px_is_null() -> str:
            pg_path = subj_dir / "part_grades.json"
            orig_pg = json.loads(pg_path.read_text(encoding="utf-8"))
            try:
                pg_filled = copy.deepcopy(orig_pg)
                # grade_holds writes did="filled" without "px"
                pg_filled["frames"][1] = {
                    "source_frame": 102,
                    "did": "filled",
                    "saved_on_its_classes": None,
                    "fill_on_its_classes": 0.62,
                    "from_frames": [0],
                    "px_on_the_track": 2100,
                }
                pg_path.write_text(json.dumps(pg_filled, indent=2) + "\n", encoding="utf-8")
                w = frame_table.build_frame_table(cap)
                _, r_filled = _read_table(w["jsonl"])
                r102 = {r["source_frame"]: r for r in r_filled}[102]
                assert r102["outcome"] == "held" and r102["from_frame"] == 101, r102
                assert r102["loaded_part_px"] is None and r102["part_px"] is None, (
                    f"did='filled' without px fell back to {r102['loaded_part_px']!r} instead of None"
                )
                assert "not in the record" in r102.get("part_px_measures", ""), r102
                return f"102(did='filled', no px): loaded_part_px={r102['loaded_part_px']}, part_px_measures={r102['part_px_measures']!r}"
            finally:
                pg_path.write_text(json.dumps(orig_pg, indent=2) + "\n", encoding="utf-8")

        case(
            "D12-7: 'filled' frame with no px in part_grades.json writes null loaded_part_px/part_px with 'not in the record'",
            c_d12_7_filled_no_px_is_null,
        )

        def c_d12_8_remaining_silent_defaults_refused() -> str:
            # 1. Unwired segmenter on MiniMaxH3SubjectTrack -> Refused even when only 1 CheckpointLoaderSimple exists
            prev_path = root / "data" / "2026-10-16_shot_s_0100a_job" / "preview.json"
            orig_prev = json.loads(prev_path.read_text(encoding="utf-8"))
            try:
                unwired = copy.deepcopy(orig_prev)
                unwired["prompt"].pop("0", None)
                unwired["prompt"]["2"]["inputs"].pop("segmenter", None)
                prev_path.write_text(json.dumps(unwired, indent=2) + "\n", encoding="utf-8")
                try:
                    frame_table.build_frame_table(cap)
                    raise AssertionError("expected Refused when segmenter is unwired even with 1 CheckpointLoaderSimple")
                except frame_table.Refused as exc:
                    assert "CheckpointLoaderSimple" in str(exc) or "segmenter" in str(exc)
            finally:
                prev_path.write_text(json.dumps(orig_prev, indent=2) + "\n", encoding="utf-8")

            # 2. Missing shots_first_source_frame -> Refused even when first_source_frame is present
            manifest_path = cap / "manifest.json"
            orig_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            try:
                no_sff = copy.deepcopy(orig_manifest)
                s0 = no_sff["subjects"][0]["sightings"][0]
                s0.pop("shots_first_source_frame", None)
                s0["first_source_frame"] = 100
                manifest_path.write_text(json.dumps(no_sff, indent=2) + "\n", encoding="utf-8")
                try:
                    frame_table.build_frame_table(cap)
                    raise AssertionError("expected Refused when shots_first_source_frame is missing")
                except frame_table.Refused as exc:
                    assert "shots_first_source_frame" in str(exc)
            finally:
                manifest_path.write_text(json.dumps(orig_manifest, indent=2) + "\n", encoding="utf-8")

            # 3. Empty track_share or parts_share in per_frame.csv -> Refused
            pf_path = subj_dir / "per_frame.csv"
            orig_pf = pf_path.read_text(encoding="utf-8")
            try:
                lines = orig_pf.splitlines()
                cols = lines[1].split(",")
                cols[3] = ""  # empty track_share
                pf_path.write_text("\n".join([lines[0], ",".join(cols)] + lines[2:]) + "\n", encoding="utf-8")
                try:
                    frame_table.build_frame_table(cap)
                    raise AssertionError("expected Refused on empty track_share")
                except frame_table.Refused as exc:
                    assert "track_share" in str(exc)
            finally:
                pf_path.write_text(orig_pf, encoding="utf-8")

            # 4. Missing who_is_who.json -> Refused (never making up answered_by from shots table)
            who_path = root / "data" / "2026-10-16_synth_load_0101a_job" / "synth_load_0101a.who_is_who.json"
            orig_who = who_path.read_text(encoding="utf-8")
            try:
                who_path.unlink()
                try:
                    frame_table.build_frame_table(cap)
                    raise AssertionError("expected Refused when who_is_who.json is missing")
                except frame_table.Refused as exc:
                    assert "who_is_who" in str(exc)
            finally:
                who_path.write_text(orig_who, encoding="utf-8")

            # 5. Load overlapping two shots in shots.json -> Refused
            shots_json = root / "data" / "fun" / "shots.json"
            orig_shots = json.loads(shots_json.read_text(encoding="utf-8"))
            try:
                two_shots = {
                    "shots": [
                        {"id": 11, "first": 100, "last": 103, "who": {"lead": {"present": "yes"}}},
                        {"id": 12, "first": 104, "last": 108, "who": {"lead": {"present": "yes"}}},
                    ]
                }
                shots_json.write_text(json.dumps(two_shots, indent=2) + "\n", encoding="utf-8")
                try:
                    frame_table.build_frame_table(cap)
                    raise AssertionError("expected Refused when load spans two shots in shots.json")
                except frame_table.Refused as exc:
                    assert "spans 2 shots" in str(exc)
            finally:
                shots_json.write_text(json.dumps(orig_shots, indent=2) + "\n", encoding="utf-8")

            return "all 5 silent fallbacks (unwired loader, first_source_frame, empty share, missing who_is_who, 2-shot overlap) refused"

        case(
            "D12-8: remaining silent defaults (unwired loader, first_source_frame, empty CSV share, missing who_is_who, 2-shot overlap) refused",
            c_d12_8_remaining_silent_defaults_refused,
        )

        def c_d12_9_pair_table_stem() -> str:
            # Make a second subject 'second' in the capture and job as a pair load, build second's table,
            # and verify pair.table_file for the primary subject 'lead' is 'frame_table.jsonl' (not 'frame_table__lead.jsonl')
            sec_dir = cap / "subjects" / "second"
            shutil.copytree(subj_dir, sec_dir)
            for p in list(sec_dir.iterdir()):
                if "lead" in p.name:
                    p.rename(sec_dir / p.name.replace("lead", "second"))
            pg_sec = json.loads((sec_dir / "part_grades.json").read_text(encoding="utf-8"))
            pg_sec["subject"] = "second"
            (sec_dir / "part_grades.json").write_text(json.dumps(pg_sec, indent=2) + "\n", encoding="utf-8")

            manifest_path = cap / "manifest.json"
            orig_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            orig_job = json.loads(job_file.read_text(encoding="utf-8"))
            who_path = root / "data" / "2026-10-16_synth_load_0101a_job" / "synth_load_0101a.who_is_who.json"
            orig_who = json.loads(who_path.read_text(encoding="utf-8"))
            prev_path = root / "data" / "2026-10-16_shot_s_0100a_job" / "preview.json"
            orig_prev = json.loads(prev_path.read_text(encoding="utf-8"))
            try:
                m_pair = copy.deepcopy(orig_manifest)
                s_sec = copy.deepcopy(m_pair["subjects"][0])
                s_sec["label"] = "second"
                m_pair["subjects"].append(s_sec)
                manifest_path.write_text(json.dumps(m_pair, indent=2) + "\n", encoding="utf-8")

                j_pair = copy.deepcopy(orig_job)
                j_pair["job"]["subjects"].append({"label": "second", "drop": "106-106", "drop_by": "second drop on 106"})
                j_pair["loads"][0]["subjects"] = [{"subject": "lead"}, {"subject": "second"}]
                job_file.write_text(json.dumps(j_pair, indent=2) + "\n", encoding="utf-8")

                w_pair = copy.deepcopy(orig_who)
                w_pair["rows"][0]["trackers"]["second"] = copy.deepcopy(w_pair["rows"][0]["trackers"]["lead"])
                who_path.write_text(json.dumps(w_pair, indent=2) + "\n", encoding="utf-8")

                p_pair = copy.deepcopy(orig_prev)
                p_pair["prompt"]["1009"]["inputs"]["filename_prefix"] = "shot_s_0100a_second_shots"
                prev_path.write_text(json.dumps(p_pair, indent=2) + "\n", encoding="utf-8")

                w_sec = frame_table.build_frame_table(cap, subject_arg="second")
                assert w_sec["jsonl"].name == "frame_table__second.jsonl"
                h_sec, _ = _read_table(w_sec["jsonl"])
                pair_block = h_sec["load"]["pair"]
                assert pair_block["also_subject"] == "lead"
                assert pair_block["table_file"] == "frame_table.jsonl", (
                    f"second subject's pair block named {pair_block['table_file']!r} instead of 'frame_table.jsonl'"
                )
                assert "frame_table.jsonl" in pair_block["note"] and "frame_table__lead.jsonl" not in pair_block["note"], pair_block["note"]
                return f"second table pair={pair_block}"
            finally:
                shutil.rmtree(sec_dir, ignore_errors=True)
                (cap / "frame_table__second.jsonl").unlink(missing_ok=True)
                (cap / "frame_table__second.md").unlink(missing_ok=True)
                manifest_path.write_text(json.dumps(orig_manifest, indent=2) + "\n", encoding="utf-8")
                job_file.write_text(json.dumps(orig_job, indent=2) + "\n", encoding="utf-8")
                who_path.write_text(json.dumps(orig_who, indent=2) + "\n", encoding="utf-8")
                prev_path.write_text(json.dumps(orig_prev, indent=2) + "\n", encoding="utf-8")

        case(
            "D12-9: pair note in second subject's table names 'frame_table.jsonl' for the primary subject",
            c_d12_9_pair_table_stem,
        )

        def c_d12_10_offscreen_shot_not_in_view() -> str:
            shots_table_path = subj_dir / "shots__shot_s_0100a.json"
            orig_st = json.loads(shots_table_path.read_text(encoding="utf-8"))
            try:
                st_off = copy.deepcopy(orig_st)
                # Split into shot 1 (rel 0..0, on_screen=False, frames_without_subject=[]) and shot 2 (rel 1..8, on_screen=True)
                st_off["shots"] = [
                    {
                        "shot": 1,
                        "first_frame": 0,
                        "last_frame": 0,
                        "frames": 1,
                        "on_screen": False,
                        "frames_with_subject": 0,
                        "frames_without_subject": [],
                        "frames_on_others": [],
                        "track_score": [-1.20],
                        "subject": st_off["shots"][0]["subject"],
                    },
                    {
                        "shot": 2,
                        "first_frame": 1,
                        "last_frame": 8,
                        "frames": 8,
                        "on_screen": True,
                        "frames_with_subject": 7,
                        "frames_without_subject": [],
                        "frames_on_others": [[8, 8]],
                        "track_score": [5.10, None, 0.30, 5.40, 5.50, 5.60, 4.80, 0.12],
                        "subject": st_off["shots"][0]["subject"],
                    },
                ]
                shots_table_path.write_text(json.dumps(st_off, indent=2) + "\n", encoding="utf-8")
                w = frame_table.build_frame_table(cap)
                _, r_off = _read_table(w["jsonl"])
                r100 = {r["source_frame"]: r for r in r_off}[100]
                assert r100["sam"]["in_view"] is False, (
                    f"frame 100 in shot with on_screen=False and empty frames_without_subject was marked in_view=True: {r100['sam']}"
                )
                assert "on_screen" in r100["sam"].get("in_view_from", ""), r100["sam"]
                return f"100.sam={r100['sam']}"
            finally:
                shots_table_path.write_text(json.dumps(orig_st, indent=2) + "\n", encoding="utf-8")

        case(
            "D12-10: shot with on_screen=false (and empty frames_without_subject) is out of view",
            c_d12_10_offscreen_shot_not_in_view,
        )

        def c_d22_1_floor_hold_and_emptied_dids() -> str:
            pg_path = subj_dir / "part_grades.json"
            orig_pg = json.loads(pg_path.read_text(encoding="utf-8"))
            try:
                pg_floor = copy.deepcopy(orig_pg)
                # Test "held by the floor rule" on 102 (with from_frames, source_px, face_points_inside)
                # and each of the 3 new EMPTIED_DIDS on 103..105 (including collapsed_against_px without least_px)
                pg_floor["frames"] = [
                    orig_pg["frames"][0],
                    {
                        "source_frame": 102,
                        "did": "held by the floor rule",
                        "why": "too small to be drawn as the part",
                        "least_px": 480,
                        "from_frames": [0],
                        "from_source_frames": [101],
                        "source_px": 2400,
                        "px": 2350,
                        "px_on_the_track": 2100,
                        "face_points_inside": {"inside": 38, "of": 38},
                    },
                    {
                        "source_frame": 103,
                        "did": "emptied: hold missed her track: the nearest standing part, moved, lies mostly off her track, so nothing is placed (data, which the gap check shows)",
                        "why": "too small to be drawn as the part",
                        "least_px": 480,
                        "px": 180,
                        "px_on_the_track": 0,
                    },
                    {
                        "source_frame": 104,
                        "did": "left empty: hold missed her track: the nearest standing part, moved, lies mostly off her track, so nothing is placed (data, which the gap check shows)",
                        "px": 0,
                        "px_on_the_track": 0,
                    },
                    {
                        "source_frame": 105,
                        "did": "emptied: no standing part in the load to hold it from",
                        "why": "collapsed against its neighbours",
                        "collapsed_against_px": 2400,
                        "px": 120,
                    },
                ]
                pg_path.write_text(json.dumps(pg_floor, indent=2) + "\n", encoding="utf-8")
                w = frame_table.build_frame_table(cap)
                _, r_fl = _read_table(w["jsonl"])
                by_f = {r["source_frame"]: r for r in r_fl}
                assert by_f[102]["outcome"] == "held" and by_f[102]["from_frame"] == 101 and by_f[102]["loaded_part_px"] == 2350, by_f[102]
                assert by_f[102].get("source_px") == 2400 and by_f[102].get("face_points_inside") == {"inside": 38, "of": 38}, by_f[102]
                assert by_f[103]["outcome"] == "emptied" and by_f[103]["cause"] == "floor", by_f[103]
                assert by_f[104]["outcome"] == "emptied" and by_f[104]["cause"] == "grade", by_f[104]
                assert by_f[105]["outcome"] == "emptied" and by_f[105]["cause"] == "floor", by_f[105]
                return f"102(held by the floor rule, from=101, face=38/38); 103..105(emptied causes={[by_f[f]['cause'] for f in (103, 104, 105)]})"
            finally:
                pg_path.write_text(json.dumps(orig_pg, indent=2) + "\n", encoding="utf-8")

        case(
            "D22-1: floor hold ('held by the floor rule') and floor/word missed-track emptied dids accepted in CLOSED_DID_VOCAB",
            c_d22_1_floor_hold_and_emptied_dids,
        )

        def c_no_face_hair_kept() -> str:
            pg_path = subj_dir / "part_grades.json"
            orig_pg = json.loads(pg_path.read_text(encoding="utf-8"))
            try:
                pg_nf = copy.deepcopy(orig_pg)
                # 104: the job's --no-face-on on a part made of Hair, its face classes dropped and the rest kept (the capture's own string)
                pg_nf["frames"] = [g for g in orig_pg["frames"] if g["source_frame"] != 104] + [{
                    "source_frame": 104,
                    "did": "face classes dropped on a person's word (--no-face-on), the rest of the part kept",
                    "saved_on_its_classes": 0.95,
                    "fill_on_its_classes": None,
                    "px": 900,
                    "face_px_dropped": 600,
                    "not_held": "no face on a person's word (--no-face-on)",
                    "why": "no face on a person's word (--no-face-on): the part's face classes dropped, the rest of it (her hair) kept",
                }]
                pg_path.write_text(json.dumps(pg_nf, indent=2) + "\n", encoding="utf-8")
                w = frame_table.build_frame_table(cap)
                _, r_nf = _read_table(w["jsonl"])
                r104 = {r["source_frame"]: r for r in r_nf}[104]
                # THE CASE THAT GOES RED when the record's own `did` is refused, read as emptied, or read as the whole saved part laid
                assert r104["outcome"] == "replaced" and r104["loaded_part_px"] == 900 and r104["part_px"] == 900, r104
                assert r104["face_px_dropped"] == 600 and "face classes" in r104["part_px_measures"], r104
                assert "face classes dropped on a person's word (600 px)" in w["md"].read_text(encoding="utf-8")
                return f"104: outcome={r104['outcome']}, loaded_part_px={r104['loaded_part_px']}, face_px_dropped={r104['face_px_dropped']}"
            finally:
                pg_path.write_text(json.dumps(orig_pg, indent=2) + "\n", encoding="utf-8")

        case(
            "no face on a person's word (2026-10-21): her hair kept is replaced with the px the record kept, the face px dropped said",
            c_no_face_hair_kept,
        )

        def c_restored_by_her_hair_in_hole() -> str:
            pg_path = subj_dir / "part_grades.json"
            orig_pg = json.loads(pg_path.read_text(encoding="utf-8"))
            try:
                pg_hih = copy.deepcopy(orig_pg)
                pg_hih["frames"] = [g for g in orig_pg["frames"] if g["source_frame"] != 104] + [{
                    "source_frame": 104,
                    "did": "saved part kept, restored by her_hair_in_hole",
                    "saved_on_its_classes": 1.0,
                    "fill_on_its_classes": None,
                    "why": "too small to be drawn as the part",
                    "px": 818,
                    "least_px": 400,
                }]
                pg_path.write_text(json.dumps(pg_hih, indent=2) + "\n", encoding="utf-8")
                w = frame_table.build_frame_table(cap)
                _, r_hih = _read_table(w["jsonl"])
                r104 = {r["source_frame"]: r for r in r_hih}[104]
                assert r104["outcome"] == "replaced" and r104["loaded_part_px"] == 818 and r104["part_px"] == 818, r104
                assert r104["did"] == "saved part kept, restored by her_hair_in_hole", r104
                return f"104: outcome={r104['outcome']}, loaded_part_px={r104['loaded_part_px']}, did={r104['did']!r}"
            finally:
                pg_path.write_text(json.dumps(orig_pg, indent=2) + "\n", encoding="utf-8")

        case(
            "restored by her_hair_in_hole (2026-10-27): accepted in CLOSED_DID_VOCAB and recorded as replaced with its restored px",
            c_restored_by_her_hair_in_hole,
        )

        def c_parts_held_fallback_4912p() -> str:
            """4912p check case (2026-10-29): when her_hair_in_hole restores frames absent from part_grades.json['frames']
            (4921: 320 px, 4943-4946: 972..4549 px -> synthetic frame 105: saved_part_px=0, parts_held=972 px) or augments
            a kept frame (4920: 5279 -> 7606 px; 4939: saved=4675, g['px']=8950 -> synthetic frame 101: saved_part_px=2400,
            g['px']=8950), frame_table reads parts_held__<by>.mkv and g['px'] rather than misreporting frame 105 as emptied (0 px)."""
            from capture_masked_run import write_mask_video
            import numpy as np
            pg_path = subj_dir / "part_grades.json"
            mf_path = cap / "manifest.json"
            part_mkv = subj_dir / "parts_held__shot_s_0100a.mkv"
            orig_pg = json.loads(pg_path.read_text(encoding="utf-8"))
            orig_mf = json.loads(mf_path.read_text(encoding="utf-8"))
            orig_mkv_bytes = part_mkv.read_bytes()
            try:
                # 6 frames (101..106) at 1024x768: put 8950 px on frame 0 (101) and 972 px on frame 4 (105, which is absent from part_grades.json and has saved_part_px=0)
                masks = np.zeros((6, 768, 1024), dtype=bool)
                masks[0].flat[:8950] = True
                masks[1].flat[:2350] = True
                masks[4].flat[:972] = True
                write_mask_video(part_mkv, masks)

                # On 4912p, restored frames 4921 and 4943-4946 are not in left_as_the_source_with_no_mask
                mf_4912 = copy.deepcopy(orig_mf)
                mf_4912["runs"][0]["left_as_the_source_with_no_mask"] = []
                mf_path.write_text(json.dumps(mf_4912, indent=2) + "\n", encoding="utf-8")

                pg_4912 = copy.deepcopy(orig_pg)
                pg_4912["frames"][0]["px"] = 8950  # frame 101: did="saved part kept, fill refused" with px=8950 updated by her_hair_in_hole (like 4939)
                pg_path.write_text(json.dumps(pg_4912, indent=2) + "\n", encoding="utf-8")

                w = frame_table.build_frame_table(cap)
                _, rows_4912 = _read_table(w["jsonl"])
                by_sf = {r["source_frame"]: r for r in rows_4912}
                assert by_sf[101]["outcome"] == "replaced" and by_sf[101]["loaded_part_px"] == 8950 and by_sf[101]["part_px"] == 8950, by_sf[101]
                assert by_sf[105]["outcome"] == "replaced" and by_sf[105]["loaded_part_px"] == 972 and by_sf[105]["part_px"] == 972, by_sf[105]
                assert by_sf[105]["saved_part_px"] == 0 and by_sf[105].get("did") == "saved part kept, restored by her_hair_in_hole", by_sf[105]

                # Verify that a part mask video whose frame count does not match the capture is refused (never cleared silently)
                write_mask_video(part_mkv, masks[:5])
                try:
                    frame_table.build_frame_table(cap)
                    raise AssertionError("expected Refused on mismatched part mask video frame count")
                except frame_table.Refused as e:
                    assert "has 5 frame(s), expected 6" in str(e), e

                return f"101: loaded_part_px={by_sf[101]['loaded_part_px']} (saved=2400); 105: outcome={by_sf[105]['outcome']}, loaded_part_px={by_sf[105]['loaded_part_px']} (saved=0); 5-frame mkv refused"
            finally:
                part_mkv.write_bytes(orig_mkv_bytes)
                mf_path.write_text(json.dumps(orig_mf, indent=2) + "\n", encoding="utf-8")
                pg_path.write_text(json.dumps(orig_pg, indent=2) + "\n", encoding="utf-8")

        case(
            "4912p parts_held fallback (2026-10-29): frame absent from part_grades.json restored in parts_held__<by>.mkv is replaced with its carried px, never emptied",
            c_parts_held_fallback_4912p,
        )

        def c_hair_outside_hole_share() -> str:
            """Step 6 check (2026-10-29): when per_frame.csv records hair_px, hair_outside_hole_px, and hair_outside_hole_share
            (written by capture_masked_run.subject_rows on hair loads), frame_table surfaces them on every load row and in frame_table.md."""
            pf_path = subj_dir / "per_frame.csv"
            orig_pf = pf_path.read_text(encoding="utf-8")
            try:
                lines = orig_pf.strip().splitlines()
                new_lines = [lines[0] + ",hair_px,hair_outside_hole_px,hair_outside_hole_share"]
                hair_vals = [
                    (9094, 136, 0.0150),
                    (8573, 119, 0.0139),
                    (0, 0, 0.0000),
                    (7122, 432, 0.0607),
                    (0, 0, 0.0000),
                    (2780, 143, 0.0514),
                ]
                for idx, row_line in enumerate(lines[1:]):
                    hp, hop, hos = hair_vals[idx]
                    new_lines.append(f"{row_line},{hp},{hop},{hos:.4f}")
                pf_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                w = frame_table.build_frame_table(cap)
                hdr_h, rows_h = _read_table(w["jsonl"])
                by_h = {r["source_frame"]: r for r in rows_h}
                assert "hair_outside_hole_share" in hdr_h["size_columns_are"], hdr_h["size_columns_are"]
                assert by_h[101]["hair_px"] == 9094 and by_h[101]["hair_outside_hole_px"] == 136 and by_h[101]["hair_outside_hole_share"] == 0.0150, by_h[101]
                assert by_h[103]["hair_px"] == 0 and by_h[103]["hair_outside_hole_px"] == 0 and by_h[103]["hair_outside_hole_share"] == 0.0, by_h[103]
                assert by_h[100]["hair_outside_hole_share"] is None, by_h[100]
                md_text = w["md"].read_text(encoding="utf-8")
                assert "0.0150 (136 / 9094 px)" in md_text, md_text
                return f"101: share={by_h[101]['hair_outside_hole_share']} ({by_h[101]['hair_outside_hole_px']}/{by_h[101]['hair_px']} px)"
            finally:
                pf_path.write_text(orig_pf, encoding="utf-8")

        case(
            "hair_outside_hole_share (2026-10-29): surfaces per-frame hair_px, hair_outside_hole_px, and hair_outside_hole_share in JSONL and Markdown",
            c_hair_outside_hole_share,
        )

    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
