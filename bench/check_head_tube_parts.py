#!/usr/bin/env python3
"""Check head_tube_parts: SAM 3.1 head tube gating, temporal seg smoothing, and 39-point face cross-check."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import head_tube_parts as htp  # noqa: E402
from _lib import case, finish, refused  # noqa: E402


def _make_base_tube_and_logits(
    num_frames: int = 5,
    height: int = 64,
    width: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    num_classes = len(htp.CLASS_NAMES)
    tube = np.zeros((num_frames, height, width), dtype=bool)
    tube[:, 8:24, 20:44] = True

    logits = np.full((num_frames, num_classes, height, width), -4.0, dtype=np.float32)
    logits[:, 0, :, :] = 3.0  # Background default
    face_id = htp.NAME_TO_ID["Face_Neck"]
    hair_id = htp.NAME_TO_ID["Hair"]
    # Upper strip of tube is Hair, lower strip is Face_Neck
    logits[:, hair_id, 8:14, 20:44] = 6.0
    logits[:, 0, 8:14, 20:44] = -2.0
    logits[:, face_id, 14:24, 20:44] = 6.0
    logits[:, 0, 14:24, 20:44] = -2.0
    return tube, logits


def _case_off_tube_false_positive() -> str:
    tube, logits = _make_base_tube_and_logits()
    face_id = htp.NAME_TO_ID["Face_Neck"]

    # Inject frame-628 style hip false-positive Face_Neck blob outside shot_box (y=44..56, x=24..36)
    logits[2, face_id, 44:56, 24:36] = 8.0
    logits[2, 0, 44:56, 24:36] = -3.0
    # Also inject a collar false-positive speck inside shot_box but outside tube (y=25..27, x=28..34)
    logits[2, face_id, 25:27, 28:34] = 7.0
    logits[2, 0, 25:27, 28:34] = -3.0

    # Check SAM 3.1 [-1, 1] normalization, exact GELU patching, combined score, and other-person masking
    norm = htp.sam31_normalize_input(torch.tensor([[[0.0, 0.5, 1.0]]]))
    assert torch.allclose(norm, torch.tensor([[[-1.0, 0.0, 1.0]]]), atol=1e-6), (
        f"sam31_normalize_input failed on [0, 1]: {norm}"
    )
    comb = htp.sam31_combined_score(torch.tensor([0.0]), torch.tensor([0.0]))
    assert abs(float(comb.item()) - 0.25) < 1e-6, f"expected 0.5 * 0.5 = 0.25, got {comb.item()}"
    dummy_tower = nn.Sequential(nn.GELU(approximate="tanh"))
    assert htp.sam31_enforce_exact_gelu(dummy_tower) == 1 and dummy_tower[0].approximate == "none"

    dummy_frames = np.full((2, 16, 16, 3), 200, dtype=np.uint8)
    other_mask = np.zeros((2, 16, 16), dtype=bool)
    other_mask[:, 0:4, 0:4] = True
    masked_frames = htp.mask_other_people_only(dummy_frames, other_mask)
    assert int(masked_frames[0, 0, 0, 0]) == htp.OTHER_PERSON_FILL_RGB
    assert int(masked_frames[0, 10, 10, 0]) == 200
    shrunk = htp.resize_crop(dummy_frames[0], (8, 8))
    enlarged = htp.resize_crop(shrunk, (16, 16))
    assert shrunk.shape == (8, 8, 3) and enlarged.shape == (16, 16, 3)

    tube_info = htp.sam31_head_tube(tube, seed_frame=0)
    box = tube_info["shot_box"]
    assert all(c % 2 == 0 for c in box), f"shot_box not even-snapped: {box}"

    seg = htp.smooth_seg_in_tube(logits, tube, shot_box=box)
    part_mask = seg["part_mask"]
    hip_px = int(part_mask[:, 44:56, 24:36].sum())
    outside_tube_px = int((part_mask & ~tube).sum())
    assert hip_px == 0, f"expected hip false-positive blob to be zeroed, got {hip_px} px"
    assert outside_tube_px == 0, f"expected 0 part_mask px outside tube, got {outside_tube_px} px"

    suppressed_f2 = seg["off_tube_suppressed_px"][2]
    expected_min_suppressed = (56 - 44) * (36 - 24) + (27 - 25) * (34 - 28)
    assert suppressed_f2 >= expected_min_suppressed, (
        f"expected off_tube_suppressed_px[2] >= {expected_min_suppressed}, got {suppressed_f2}"
    )
    head_px = seg["part_px"][2]
    assert head_px == int(tube[2].sum()), f"expected full head tube coverage {int(tube[2].sum())}, got {head_px}"
    return f"shot_box={box} hip_px={hip_px} suppressed_f2={suppressed_f2} head_px={head_px}"


def _case_temporal_smoothing_dropout() -> str:
    tube, logits = _make_base_tube_and_logits()
    face_id = htp.NAME_TO_ID["Face_Neck"]
    hair_id = htp.NAME_TO_ID["Hair"]

    # Frame 2 suffers a 1-frame Sapiens2-Seg dropout inside the head tube
    logits[2, face_id, 14:24, 20:44] = -3.0
    logits[2, hair_id, 8:14, 20:44] = -3.0
    logits[2, 0, 8:24, 20:44] = 5.0

    seg = htp.smooth_seg_in_tube(logits, tube)
    raw_f2 = seg["raw_part_px"][2]
    smoothed_f2 = seg["part_px"][2]
    neighbor_px = seg["part_px"][1]
    bridged = seg["dropout_bridged_frames"]

    assert raw_f2 == 0, f"expected unsmoothed frame-2 part_px == 0 on dropout, got {raw_f2}"
    assert smoothed_f2 == neighbor_px and smoothed_f2 > 0, (
        f"expected temporal smoothing to bridge frame-2 dropout to {neighbor_px} px, got {smoothed_f2}"
    )
    assert 2 in bridged, f"expected frame 2 in dropout_bridged_frames, got {bridged}"
    return f"raw_f2={raw_f2} smoothed_f2={smoothed_f2} bridged={bridged}"


def _case_face_points_cross_check() -> str:
    tube, logits = _make_base_tube_and_logits(num_frames=4)
    face_id = htp.NAME_TO_ID["Face_Neck"]
    hair_id = htp.NAME_TO_ID["Hair"]

    # Frame 1: turned-away / bent head (like shot 2 frames 684-707): >90% Hair in tube, 0% Face_Neck
    for f_idx in (1,):
        logits[f_idx, hair_id, 8:24, 20:44] = 8.0
        logits[f_idx, face_id, 8:24, 20:44] = -5.0
        logits[f_idx, 0, 8:24, 20:44] = -4.0

    # Frame 2: 1-frame Face_Neck dropout inside tube while face points remain visible inside tube
    logits[2, face_id, 14:24, 20:44] = -3.0
    logits[2, 0, 14:24, 20:44] = 5.0

    seg = htp.smooth_seg_in_tube(logits, tube)

    # Construct 308-keypoint pose output across the 4 frames
    num_kpts = len(htp.POSE_KEYPOINT_INDEX)
    pose = np.zeros((4, num_kpts, 3), dtype=np.float32)
    face_ids = htp.scored_face_ids()
    assert len(face_ids) == 39, f"expected 39 scored facial keypoints, got {len(face_ids)}"

    # Frames 0 and 2: all 39 scored face points land inside the Face_Neck region of tube (x=32, y=18)
    for f_idx in (0, 2):
        for kid in face_ids:
            pose[f_idx, kid] = (32.0, 18.0, 0.92)

    # Frame 1 (turned away): scored face points have low score (< 0.30), while unscored pupil points
    # have high score inside the tube to verify pupils are excluded from scored_face_ids
    for kid in face_ids:
        pose[1, kid] = (32.0, 18.0, 0.05)
    for name, kid in htp.POSE_KEYPOINT_INDEX.items():
        if "pupil" in name and kid < num_kpts:
            pose[1, kid] = (32.0, 18.0, 0.99)

    # Frame 3 (off-track pose jump): scored face points jump outside tube to (x=58, y=58)
    for kid in face_ids:
        pose[3, kid] = (58.0, 58.0, 0.95)

    cc = htp.cross_check_face_points(
        pose,
        tube,
        seg["face_neck_mask"],
        seg["hair_mask"],
        raw_face_neck_px=seg["raw_face_neck_px"],
        dropout_bridged_seed=seg["dropout_bridged_frames"],
    )

    assert cc["turned_away_frames"] == [1], f"expected turned_away_frames=[1], got {cc['turned_away_frames']}"
    assert cc["dropout_bridged_frames"] == [2], (
        f"expected dropout_bridged_frames=[2], got {cc['dropout_bridged_frames']}"
    )
    assert cc["off_tube_pose_frames"] == [3], (
        f"expected off_tube_pose_frames=[3], got {cc['off_tube_pose_frames']}"
    )
    assert cc["face_points_in_tube"][0] == 39 and cc["face_points_in_tube"][2] == 39, (
        f"expected 39 face points in tube on frames 0 and 2, got {cc['face_points_in_tube']}"
    )
    assert cc["face_points_in_tube"][1] == 0, (
        f"expected 0 scored face points on turned-away frame 1, got {cc['face_points_in_tube'][1]}"
    )
    assert cc["frame_states"] == ["normal", "turned_away", "dropout_bridged", "off_tube_pose"], (
        f"unexpected frame_states: {cc['frame_states']}"
    )
    return (
        f"states={cc['frame_states']} in_tube={cc['face_points_in_tube']} "
        f"hair_share_f1={cc['hair_tube_share'][1]:.2f}"
    )


def _case_provenance_and_cli() -> str:
    tube, logits = _make_base_tube_and_logits(num_frames=3)
    num_kpts = len(htp.POSE_KEYPOINT_INDEX)
    pose = np.zeros((3, num_kpts, 3), dtype=np.float32)
    for kid in htp.scored_face_ids():
        pose[:, kid] = (32.0, 18.0, 0.90)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        tube_path = tmp_dir / "tube.npy"
        logits_path = tmp_dir / "logits.npy"
        pose_path = tmp_dir / "pose.npy"
        masks_path = tmp_dir / "masks.npz"
        json_path = tmp_dir / "provenance.json"

        np.save(tube_path, tube)
        np.save(logits_path, logits)
        np.save(pose_path, pose)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = htp.main(
                [
                    "--tube",
                    str(tube_path),
                    "--logits",
                    str(logits_path),
                    "--pose",
                    str(pose_path),
                    "--out-masks",
                    str(masks_path),
                    "--out-json",
                    str(json_path),
                ]
            )
        assert rc == 0, f"expected main() rc == 0, got {rc}"
        assert "shot_box=" in buf.getvalue(), f"expected summary output from main(), got {buf.getvalue()!r}"
        assert masks_path.is_file() and json_path.is_file(), "expected output masks.npz and provenance.json"

        prov = json.loads(json_path.read_text(encoding="utf-8"))
        required_keys = (
            "shot_box",
            "temporal_weights",
            "tube_px",
            "raw_part_px",
            "part_px",
            "off_tube_suppressed_px",
            "outside_person_track_clipped_px",
            "face_points_in_tube",
            "face_points_in_part",
            "turned_away_frames",
            "dropout_bridged_frames",
            "off_tube_pose_frames",
        )
        missing = [k for k in required_keys if k not in prov]
        assert not missing, f"provenance JSON missing keys: {missing}"
        assert prov["face_points_in_tube"] == [39, 39, 39], (
            f"expected face_points_in_tube=[39, 39, 39], got {prov['face_points_in_tube']}"
        )
        loaded_masks = np.load(masks_path)
        assert loaded_masks["class_map"].shape == tube.shape, (
            f"unexpected class_map shape {loaded_masks['class_map'].shape}"
        )
    return f"keys={len(required_keys)} shot_box={prov['shot_box']} face_pts={prov['face_points_in_tube']}"


def _case_person_track_bounds_tube() -> str:
    tube, logits = _make_base_tube_and_logits(num_frames=4, height=64, width=80)
    person_track = np.zeros_like(tube, dtype=bool)
    # Subject's person track covers left half where subject's head+body live (y=8..58, x=14..44).
    person_track[:, 8:58, 14:44] = True

    # Simulate SAM 3.1 head tube drifting/jumping onto a second person on the right (y=40..54, x=52..68)
    # on frames 1 and 2, plus Sapiens2-Seg predicting Face_Neck on that second person.
    drifting_tube = tube.copy()
    drifting_tube[1:3, 40:54, 52:68] = True
    face_id = htp.NAME_TO_ID["Face_Neck"]
    logits[1:3, face_id, 40:54, 52:68] = 9.0
    logits[1:3, 0, 40:54, 52:68] = -4.0

    res = htp.run_head_tube_parts(
        logits,
        drifting_tube,
        person_track=person_track,
        seed_frame=0,
    )
    bounded_tube = res["tube_info"]["tube"]
    part_mask = res["seg_info"]["part_mask"]
    prov = res["provenance"]

    outside_tube_px = int((bounded_tube & ~person_track).sum())
    outside_part_px = int((part_mask & ~person_track).sum())
    assert outside_tube_px == 0, f"expected 0 tube px outside person_track, got {outside_tube_px}"
    assert outside_part_px == 0, f"expected 0 part_mask px outside person_track, got {outside_part_px}"
    expected_clipped = [0, 14 * 16, 14 * 16, 0]
    assert prov["outside_person_track_clipped_px"] == expected_clipped, (
        f"expected outside_person_track_clipped_px={expected_clipped}, got {prov['outside_person_track_clipped_px']}"
    )

    # Also verify seed_mask intersection on 4D video frames and head_tube_for_capture helper.
    frames = np.full((4, 64, 80, 3), 120, dtype=np.uint8)
    seed_mask_2d = drifting_tube[1].copy()
    seeded = htp.sam31_head_tube(
        frames,
        seed_frame=1,
        seed_mask=seed_mask_2d,
        person_track=person_track,
    )
    assert int((seeded["tube"] & ~person_track).sum()) == 0, "seed_mask must be bounded by person_track"

    cmap = np.zeros_like(tube, dtype=np.uint8)
    cmap[tube] = face_id
    cmap[1:3, 40:54, 52:68] = face_id
    # Simulate 1-frame dropout on frame 2 inside subject's head (y=8..24, x=20..44) while person_track stays active.
    raw_parts = (cmap == face_id)
    raw_parts[2, 8:24, 20:44] = False
    cmap[2, 8:24, 20:44] = 0
    cap_res = htp.head_tube_for_capture(
        person_track,
        raw_parts,
        cmap,
    )
    assert int((cap_res["part_mask"] & ~person_track).sum()) == 0, (
        "head_tube_for_capture part_mask must never leave person_track"
    )
    assert int(cap_res["part_mask"][2, 8:24, 20:44].sum()) == 16 * 24, (
        "head_tube_for_capture must bridge 1-frame dropout inside person_track"
    )
    return f"clipped_px={prov['outside_person_track_clipped_px']} outside_px={outside_part_px}"


def _case_person_track_regains_tube_after_gap() -> str:
    tube, logits = _make_base_tube_and_logits(num_frames=6, height=64, width=80)
    person_track = np.zeros_like(tube, dtype=bool)
    # Frames 0, 3, 4, 5: subject is present in person_track (head at y=8..24, body down to y=56).
    # Frames 1 and 2: subject is occluded / off-screen (gap in person_track).
    for t in (0, 3, 4, 5):
        person_track[t, 8:56, 20:44] = True

    # Simulate the head tube being lost after the gap (empty on frames 1..5), while Sapiens2-Seg
    # still sees Hair + Face_Neck on frames 3..5 once the subject reappears.
    lost_tube = np.zeros_like(tube, dtype=bool)
    lost_tube[0] = tube[0]
    logits[1:3, :, :, :] = -4.0
    logits[1:3, 0, :, :] = 5.0

    res = htp.run_head_tube_parts(
        logits,
        lost_tube,
        person_track=person_track,
        seed_frame=0,
    )
    regained_tube = res["tube_info"]["tube"]
    part_mask = res["seg_info"]["part_mask"]
    prov = res["provenance"]

    assert prov["regained_from_person_track_frames"] == [3, 4, 5], (
        f"expected regained_from_person_track_frames=[3, 4, 5], got {prov['regained_from_person_track_frames']}"
    )
    assert int(regained_tube[1:3].sum()) == 0, "expected empty tube during gap frames 1..2"
    for t in (3, 4, 5):
        assert int(regained_tube[t].sum()) > 0, f"expected regained head tube on frame {t}"
        assert int(part_mask[t].sum()) == 16 * 24, (
            f"expected regained part_mask on frame {t} to recover 384 head pixels, got {int(part_mask[t].sum())}"
        )

    # Also test 4D video frames with a tracker that loses the head across the gap (frames 1..5)
    # unless sam31_head_tube re-seeds the tracker from person_track at frame 3.
    frames = np.full((6, 64, 80, 3), 120, dtype=np.uint8)
    seed_calls: list[int] = []

    def _gap_dropping_tracker(norm_t, *, seed_frame=0, seed_box=None, seed_mask=None, prompt="head"):
        seed_calls.append(int(seed_frame))
        out = np.zeros((6, 64, 80), dtype=bool)
        if seed_frame == 0:
            out[0, 8:24, 20:44] = True
            # Tracker loses the head across the gap on frames 1..5!
        elif seed_frame >= 3:
            out[seed_frame:, 8:24, 20:44] = True
        return out

    tracked_info = htp.sam31_head_tube(
        frames,
        seed_frame=0,
        person_track=person_track,
        tracker=_gap_dropping_tracker,
    )
    assert seed_calls == [0, 3], f"expected tracker to be seeded at 0 and re-seeded at 3 after gap, got {seed_calls}"
    assert tracked_info["regained_from_person_track_frames"] == [3, 4, 5], (
        f"expected regained frames [3, 4, 5], got {tracked_info['regained_from_person_track_frames']}"
    )
    for t in (3, 4, 5):
        assert int(tracked_info["tube"][t, 8:24, 20:44].sum()) == 16 * 24
    return (
        f"regained_frames={prov['regained_from_person_track_frames']} "
        f"tracker_seeds={seed_calls} post_gap_part_px={prov['part_px'][3:]}"
    )


def _expect_system_exit(fn, *args, **kwargs) -> str:
    try:
        fn(*args, **kwargs)
    except SystemExit as exc:
        msg = str(exc)
        assert msg, "expected non-empty SystemExit message"
        return msg
    raise AssertionError(f"expected {fn.__name__} to raise SystemExit")


def _case_refuses_invalid_inputs() -> str:
    tube, logits = _make_base_tube_and_logits(num_frames=4)

    # 1. Mismatched frame count between tube and logits
    refused(lambda: htp.smooth_seg_in_tube(logits[:3], tube), "frame count")

    # 2. Empty head tube across all frames
    empty_tube = np.zeros_like(tube)
    refused(lambda: htp.sam31_head_tube(empty_tube, seed_frame=0), "empty")

    # 3. Mismatched person_track shape
    refused(lambda: htp.sam31_head_tube(tube, person_track=tube[:2]), "person_track shape")

    # 4. Unknown segmentation class name
    refused(lambda: htp.smooth_seg_in_tube(logits, tube, head_parts=("NotAPart",)), "NotAPart")

    # 5. Invalid (even-length or negative) temporal smoothing kernel
    refused(lambda: htp.smooth_seg_in_tube(logits, tube, temporal_weights=(0.5, 0.5)), "odd length")
    refused(
        lambda: htp.smooth_seg_in_tube(logits, tube, temporal_weights=(-0.25, 1.5, -0.25)),
        "non-negative",
    )

    # 6. CLI subprocess refusal on missing --tube path with no traceback
    script = BENCH_DIR / "head_tube_parts.py"
    cli_err = refused(
        [
            sys.executable,
            str(script),
            "--tube",
            "/nonexistent/tube.npy",
            "--logits",
            "/nonexistent/logits.npy",
            "--out-json",
            "/tmp/unused.json",
        ],
        "missing --tube file",
    )
    return f"refused 6 invalid inputs cleanly ({cli_err})"


def main() -> int:
    case("off_tube_false_positive_suppressed", _case_off_tube_false_positive)
    case("temporal_smoothing_bridges_dropout", _case_temporal_smoothing_dropout)
    case("face_points_cross_check_turned_away_vs_dropout", _case_face_points_cross_check)
    case("person_track_bounds_tube", _case_person_track_bounds_tube)
    case("person_track_regains_tube_after_gap", _case_person_track_regains_tube_after_gap)
    case("provenance_json_and_cli", _case_provenance_and_cli)
    case("refuses_invalid_inputs", _case_refuses_invalid_inputs)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
