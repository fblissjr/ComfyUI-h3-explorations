#!/usr/bin/env python3
"""Check ``bench/facts_pass.py`` on synthetic masked-run captures.

Claims, i.e. what breaks if a case is deleted:
* geometry_and_overlap_counts: per-frame pixel and cell counts (source_track_px,
  sapiens_face_px, sapiens_hair_px, head_track_px, parts_held_px, region_cells,
  region_px, face_outside_region_px, matte_px, other_overlap_px, zoom_box) or
  notable-span summaries (face_outside_region, parts_held_drop_below_half_rolling_12f_median,
  other_overlap) on the real capture_masked_run.py schema (manifest["subjects"][i]["sightings"],
  subjects/<label>/classes__<by>.npz, subjects/<label>/head_tube__<by>.npz, bit-packed
  masks__<by>.npz and carried, and cell-level region/given_back) could drift without detection.
* mouth_mode_and_ref_metadata: per-frame mouth_mode resolution (no_face, no_mouth,
  closed_lip_line, worded, class_map, lip_lines), mouth_mode_transitions, ref_crop_box,
  or ref_scale extraction from a head_reference.py JSON record (direct, via --job, or
  via sibling <capture>_job/job.render.json) could break silently.
* missing_optional_inputs: captures without a head_tube/head_track mask, window matte
  file, or head_reference.py JSON record could crash instead of recording null/unknown
  fallbacks, or malformed capture directories could fail with a raw traceback instead
  of SystemExit.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from _lib import case, finish, refused
import facts_pass


def _pack(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.astype(np.uint8), axis=-1)


def _build_base_capture(root: Path, *, include_head_tube: bool = True) -> Path:
    """Create a 4-frame, 32x32 (cell_px=16 -> 2x2 cells) capture directory matching capture_masked_run.py."""
    cap = root / "sample_capture"
    (cap / "subjects" / "alice").mkdir(parents=True)
    (cap / "subjects" / "bob").mkdir(parents=True)
    (cap / "runs" / "alice_pass1").mkdir(parents=True)

    frames, h, w, cell_px = 4, 32, 32, 16
    first_frame = 100
    by = "shot_a_0100"

    # Classes: 3 = Face_Neck, 4 = Hair
    classes = np.zeros((frames, h, w), dtype=np.uint8)
    # In all frames, rows 0..8, cols 0..8 (64 px) are Face_Neck (3)
    classes[:, 0:8, 0:8] = 3
    # In all frames, rows 0..4, cols 8..16 (32 px) are Hair (4)
    classes[:, 0:4, 8:16] = 4

    # Alice track covers top-left 16x16 quadrant (256 px) on frames 0,1,2,3
    # plus on frame 1 extends 4x4 (16 px) into top-right quadrant rows 4..8, cols 16..20
    alice_track = np.zeros((frames, h, w), dtype=bool)
    alice_track[:, 0:16, 0:16] = True
    alice_track[1, 4:8, 16:20] = True
    # Mark rows 4..8, cols 16..20 on frame 1 as Face_Neck (3) so 16 px of face are in cell (0, 1)
    classes[1, 4:8, 16:20] = 3

    # Alice head_tube covers rows 0..10, cols 0..16 (160 px); on frame 2 it also extends
    # into cell (0, 1) rows 0..4, cols 16..22 (24 px), where rows 0..2, cols 16..20 are Hair (8 px)
    # and the remaining 16 px are non-hair outside region cell (0, 0).
    alice_head = np.zeros((frames, h, w), dtype=np.uint8)
    alice_head[:, 0:10, 0:16] = 1
    alice_head[2, 0:4, 16:22] = 1
    classes[2, 0:2, 16:20] = 4  # 8 px of hair inside alice_track/head_tube
    alice_track[2, 0:2, 16:20] = True

    np.savez_compressed(
        cap / "subjects" / "alice" / f"classes__{by}.npz",
        classes=classes,
    )
    if include_head_tube:
        np.savez_compressed(
            cap / "subjects" / "alice" / f"head_tube__{by}.npz",
            tube=alice_head,
            class_map=classes,
        )

    # Bit-packed parts / carried: 200 px on frames 0, 1, 2, then drops to 50 px on frame 3
    carried = np.zeros((frames, h, w), dtype=bool)
    carried[0, 0:10, 0:20] = True  # 200 px
    carried[1, 0:10, 0:20] = True  # 200 px
    carried[2, 0:10, 0:20] = True  # 200 px
    carried[3, 0:5, 0:10] = True   # 50 px

    np.savez_compressed(
        cap / "subjects" / "alice" / f"masks__{by}.npz",
        track=_pack(alice_track),
        covered=np.ones((frames,), dtype=bool),
        parts=_pack(carried),
    )

    # Bob track overlaps Alice on frame 2 in a 4x5 block (20 px) at rows 10..14, cols 10..15
    bob_track = np.zeros((frames, h, w), dtype=bool)
    bob_track[2, 10:14, 10:15] = True
    np.savez_compressed(
        cap / "subjects" / "bob" / f"masks__{by}.npz",
        track=_pack(bob_track),
        covered=np.array([False, False, True, False], dtype=bool),
    )

    # Run region: 2x2 cells. Only top-left cell (0, 0) is active on all frames (1 cell = 256 px).
    region_cells = np.zeros((frames, h // cell_px, w // cell_px), dtype=bool)
    region_cells[:, 0, 0] = True

    # given_back__bob: cell (1, 1) on frame 3 (1 cell = 256 px)
    given_back_bob = np.zeros((frames, h // cell_px, w // cell_px), dtype=bool)
    given_back_bob[3, 1, 1] = True

    np.savez_compressed(
        cap / "runs" / "alice_pass1" / "region.npz",
        region=region_cells,
        carried=_pack(carried),
        read=np.ones((frames,), dtype=bool),
        given_back__bob=given_back_bob,
    )

    manifest = {
        "name": "sample_capture",
        "clip": "synthetic.mp4",
        "frames": frames,
        "first_frame": first_frame,
        "size": [w, h],
        "cell_px": cell_px,
        "subjects": [
            {
                "label": "alice",
                "sightings": [
                    {
                        "by": by,
                        "first_source_frame": first_frame,
                        "frames_covered": frames,
                    }
                ],
            },
            {
                "label": "bob",
                "sightings": [
                    {
                        "by": by,
                        "first_source_frame": first_frame,
                        "frames_covered": frames,
                    }
                ],
            },
        ],
        "runs": [
            {
                "name": "alice_pass1",
                "subject": "alice",
                "first_source_frame": first_frame,
                "frames_written": frames,
                "zoom": {"box": [2, 4, 30, 28]},
                "files": ["alice_window_1_planned_region.npz"],
            }
        ],
    }
    (cap / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return cap


def check_geometry_and_overlap_counts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cap = _build_base_capture(Path(tmp), include_head_tube=True)
        # Add a window matte file with 180 active pixels per frame
        matte = np.zeros((4, 32, 32), dtype=np.uint8)
        matte[:, 0:9, 0:20] = 128  # 180 px with value 128/255 > 0.01
        pred_dir = cap / "runs" / "alice_pass1" / "predicted"
        pred_dir.mkdir(parents=True, exist_ok=True)
        win_npz = pred_dir / "alice_window_1_planned_region.npz"
        np.savez_compressed(win_npz, matte=matte)

        out_json = Path(tmp) / "out.json"
        out_md = Path(tmp) / "out.md"
        rc = facts_pass.main(
            [
                str(cap),
                "--subject",
                "alice",
                "--run",
                "alice_pass1",
                "--output-json",
                str(out_json),
                "--output-md",
                str(out_md),
            ]
        )
        assert rc == 0, f"expected exit code 0, got {rc}"
        report = json.loads(out_json.read_text(encoding="utf-8"))
        md = out_md.read_text(encoding="utf-8")

        assert report["schema"] == "h3.facts_pass.v1"
        assert "pixel appearance" in report["cannot_see"]
        wm = report["what_measured"]
        assert wm["subject"] == "alice"
        assert wm["run"] == "alice_pass1"
        assert wm["window_source_frames"] == [100, 103]
        assert wm["frame_count"] == 4
        assert wm["head_track_source"] == "head_tube__shot_a_0100.npz:tube"

        rows = report["frames"]
        assert len(rows) == 4

        # Frame 0 (source_frame 100):
        # track = 256, face = 64, hair = 32, head_track = 160, region_cells = 1, region_px = 256
        # face_outside_region = 0 (all in top-left 16x16), matte_px = 180, other_overlap = 0
        f0 = rows[0]
        assert f0["window_frame"] == 0 and f0["source_frame"] == 100
        assert f0["source_track_px"] == 256
        assert f0["sapiens_face_px"] == 64
        assert f0["sapiens_hair_px"] == 32
        assert f0["head_track_px"] == 160
        assert f0["parts_held_px"] == 200
        assert f0["region_cells"] == 1
        assert f0["region_px"] == 256
        assert f0["face_outside_region_px"] == 0
        assert f0["matte_px"] == 180
        assert f0["other_overlap_px"] == 0
        assert f0["zoom_box"] == [2, 4, 30, 28]

        # Frame 1 (source_frame 101): 16 px of sapiens_face outside region cell (0,0)
        f1 = rows[1]
        assert f1["source_frame"] == 101
        assert f1["sapiens_face_px"] == 64 + 16
        assert f1["face_outside_region_px"] == 16

        # Frame 2 (source_frame 102): head_tube extends 24 px outside region cell (0,0),
        # of which 8 px are sapiens_hair -> 16 px of (head_tube & ~hair) outside region.
        # Also bob_track overlaps alice_track & region in 20 px.
        f2 = rows[2]
        assert f2["source_frame"] == 102
        assert f2["sapiens_hair_px"] == 32 + 8
        assert f2["head_track_px"] == 160 + 24
        assert f2["face_outside_region_px"] == 16
        assert f2["other_overlap_px"] == 20

        # Frame 3 (source_frame 103): parts_held drops to 50 px; given_back__bob has 1 cell (256 px)
        f3 = rows[3]
        assert f3["source_frame"] == 103
        assert f3["parts_held_px"] == 50
        assert f3["other_overlap_px"] == 256

        ns = report["notable_spans"]
        assert ns["face_outside_region"] == [
            {
                "source_frames": [101, 102],
                "frame_count": 2,
                "max_face_outside_region_px": 16,
                "sum_face_outside_region_px": 32,
            }
        ]
        assert ns["parts_held_drop_below_half_rolling_12f_median"] == [
            {
                "source_frames": [103, 103],
                "frame_count": 1,
                "max_parts_held_px": 50,
                "sum_parts_held_px": 50,
            }
        ]
        assert ns["other_overlap"] == [
            {
                "source_frames": [102, 103],
                "frame_count": 2,
                "max_other_overlap_px": 256,
                "sum_other_overlap_px": 276,
            }
        ]
        assert "## Notable spans" in md and "## Per-frame table" in md


def check_mouth_mode_and_ref_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cap = _build_base_capture(Path(tmp), include_head_tube=True)
        ref_path = Path(tmp) / "alice_ref.json"
        ref_record = {
            "its_first_frame_is_source_frame": 100,
            "box": [10, 20, 138, 148],
            "scale": [1.25, 1.25],
            "drawn": {
                "frames_left_bare": [100],
                "mouth": {
                    "drawn": True,
                    "no_mouth_on": {"head_yaw": [[101, 101]]},
                    "closed_lip_line": {"closed_frames": [[102, 102]]},
                    "mouth_on": {"drawn_on_the_word": [[103, 103]]},
                    "frames_by_the_class_map": [],
                    "frames_by_the_lip_lines": [],
                },
            },
        }
        ref_path.write_text(json.dumps(ref_record, indent=2), encoding="utf-8")

        report = facts_pass.compute_facts(cap, reference_path=ref_path)
        rows = report["frames"]
        modes = [r["mouth_mode"] for r in rows]
        assert modes == [
            "no_face",
            "no_mouth",
            "closed_lip_line",
            "worded",
        ], f"unexpected mouth modes: {modes}"
        for r in rows:
            assert r["ref_crop_box"] == [10, 20, 138, 148]
            assert r["ref_scale"] == [1.25, 1.25]

        transitions = report["notable_spans"]["mouth_mode_transitions"]
        assert len(transitions) == 3
        assert transitions[0] == {
            "from_source_frame": 100,
            "to_source_frame": 101,
            "from_mode": "no_face",
            "to_mode": "no_mouth",
        }
        assert transitions[2] == {
            "from_source_frame": 102,
            "to_source_frame": 103,
            "from_mode": "closed_lip_line",
            "to_mode": "worded",
        }

        # Verify class_map and lip_lines resolution via sibling <capture>_job/job.render.json discovery
        ref_record["drawn"]["frames_left_bare"] = []
        ref_record["drawn"]["mouth"]["no_mouth_on"] = {}
        ref_record["drawn"]["mouth"]["frames_by_the_class_map"] = [[100, 100]]
        ref_record["drawn"]["mouth"]["frames_by_the_lip_lines"] = [[101, 101]]
        ref_path.write_text(json.dumps(ref_record, indent=2), encoding="utf-8")

        sibling_job_dir = Path(tmp) / "sample_capture_job"
        sibling_job_dir.mkdir(parents=True, exist_ok=True)
        (sibling_job_dir / "job.render.json").write_text(
            json.dumps({"loads": [{"motion_file": str(ref_path.with_suffix(".mp4"))}]}),
            encoding="utf-8",
        )
        report2 = facts_pass.compute_facts(cap)
        modes2 = [r["mouth_mode"] for r in report2["frames"]]
        assert modes2 == ["class_map", "lip_lines", "closed_lip_line", "worded"]


def check_missing_optional_inputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cap = _build_base_capture(Path(tmp), include_head_tube=False)
        # Remove zoom from manifest to verify whole-canvas zoom_box fallback
        manifest = json.loads((cap / "manifest.json").read_text(encoding="utf-8"))
        manifest["runs"][0].pop("zoom", None)
        (cap / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        report = facts_pass.compute_facts(cap)
        rows = report["frames"]
        assert len(rows) == 4
        for r in rows:
            assert r["head_track_px"] is None
            assert r["mouth_mode"] == "unknown"
            assert r["ref_crop_box"] is None
            assert r["ref_scale"] is None
            assert r["zoom_box"] == [0, 0, 32, 32]

        # Without head_tube, frame 2 (which only had head_tube outside region) has 0 face_outside_region_px,
        # while frame 1 (which had sapiens_face outside region) still has 16 px.
        assert rows[1]["face_outside_region_px"] == 16
        assert rows[2]["face_outside_region_px"] == 0
        # Without window matte npz, matte_px falls back to carried (200 on f0..f2, 50 on f3)
        assert [r["matte_px"] for r in rows] == [200, 200, 200, 50]
        assert report["notable_spans"]["mouth_mode_transitions"] == []

        # Verify clean SystemExit refusal on missing capture directory or out-of-range window
        assert refused(
            lambda: facts_pass.compute_facts(Path(tmp) / "nonexistent_dir"),
            "not found",
        )
        assert refused(
            lambda: facts_pass.compute_facts(cap, first_frame=50, last_frame=60),
            "outside capture range",
        )


def main() -> int:
    case("geometry_and_overlap_counts", check_geometry_and_overlap_counts)
    case("mouth_mode_and_ref_metadata", check_mouth_mode_and_ref_metadata)
    case("missing_optional_inputs", check_missing_optional_inputs)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
