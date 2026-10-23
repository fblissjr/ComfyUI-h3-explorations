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

    # Frame 1: back of the head (like shot 2 frames 687-708 and 747-750): >85% Hair in tube, 0% Face_Neck
    for f_idx in (1,):
        logits[f_idx, hair_id, 8:24, 20:44] = 8.0
        logits[f_idx, face_id, 8:24, 20:44] = -5.0
        logits[f_idx, 0, 8:24, 20:44] = -4.0

    # Frame 2: side_with_eye / 1-frame Face_Neck dropout inside tube while face points remain visible inside tube
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

    # Frame 1 (back_of_head): scored face points have low score (< 0.30), while unscored pupil points
    # have high score inside the tube to verify pupils are excluded from scored_face_ids
    for kid in face_ids:
        pose[1, kid] = (32.0, 18.0, 0.05)
    for name, kid in htp.POSE_KEYPOINT_INDEX.items():
        if "pupil" in name and kid < num_kpts:
            pose[1, kid] = (32.0, 18.0, 0.99)

    # Frame 3 (off-track pose jump): scored face points jump outside tube to (x=58, y=58)
    for kid in face_ids:
        pose[3, kid] = (58.0, 58.0, 0.95)

    part_mask_before = seg["part_mask"].copy()
    cc = htp.cross_check_face_points(
        pose,
        tube,
        seg["face_neck_mask"],
        seg["hair_mask"],
        raw_face_neck_px=seg["raw_face_neck_px"],
        dropout_bridged_seed=seg["dropout_bridged_frames"],
    )

    assert cc["facing_frames"] == [0], f"expected facing_frames=[0], got {cc['facing_frames']}"
    assert cc["back_of_head_frames"] == [1], f"expected back_of_head_frames=[1], got {cc['back_of_head_frames']}"
    assert cc["proposed_drop_frames"] == [1], f"expected proposed_drop_frames=[1], got {cc['proposed_drop_frames']}"
    assert cc["turned_away_frames"] == [1], f"expected turned_away_frames=[1], got {cc['turned_away_frames']}"
    assert cc["side_with_eye_frames"] == [2], f"expected side_with_eye_frames=[2], got {cc['side_with_eye_frames']}"
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
        f"expected 0 scored face points on back_of_head frame 1, got {cc['face_points_in_tube'][1]}"
    )
    assert cc["frame_states"] == ["facing", "back_of_head", "side_with_eye", "off_tube_pose"], (
        f"unexpected frame_states: {cc['frame_states']}"
    )
    assert cc["face_replace_frames"] == [0, 2] and cc["face_lines_on_frames"] == [0, 2], (
        f"expected facing and side_with_eye to replace and keep face lines on: {cc['face_replace_frames']}"
    )
    # back_of_head is a proposal and empties nothing by itself
    assert np.array_equal(seg["part_mask"], part_mask_before) and int(seg["part_mask"][1].sum()) == 384, (
        "back_of_head must propose only and never empty part_mask by itself"
    )
    return (
        f"states={cc['frame_states']} replace={cc['face_replace_frames']} "
        f"proposed_drop={cc['proposed_drop_frames']} hair_share_f1={cc['hair_tube_share'][1]:.2f}"
    )


def _case_person_word_overrides_three_states() -> str:
    tube, logits = _make_base_tube_and_logits(num_frames=3)
    face_id = htp.NAME_TO_ID["Face_Neck"]
    hair_id = htp.NAME_TO_ID["Hair"]

    # Frame 0: facing (both eyes + full Face_Neck)
    # Frame 1: back_of_head (Hair > 85%, no Face_Neck)
    logits[1, hair_id, 8:24, 20:44] = 8.0
    logits[1, face_id, 8:24, 20:44] = -5.0
    logits[1, 0, 8:24, 20:44] = -4.0
    # Frame 2: side_with_eye (profile slice on right side, only right eye visible)
    logits[2, face_id, 14:24, 20:32] = -5.0
    logits[2, hair_id, 14:24, 20:32] = 6.0

    seg = htp.smooth_seg_in_tube(logits, tube)
    num_kpts = len(htp.POSE_KEYPOINT_INDEX)
    pose = np.zeros((3, num_kpts, 3), dtype=np.float32)
    for kid in htp.scored_face_ids():
        pose[0, kid] = (32.0, 18.0, 0.92)
    for name, kid in htp.POSE_KEYPOINT_INDEX.items():
        if name.startswith("r_") and name.endswith("_lash_line") and "pupil" not in name:
            pose[2, kid] = (38.0, 18.0, 0.90)

    # Without person words: [facing, back_of_head, side_with_eye]
    raw_cc = htp.cross_check_face_points(pose, tube, seg["face_neck_mask"], seg["hair_mask"])
    assert raw_cc["frame_states"] == ["facing", "back_of_head", "side_with_eye"], raw_cc["frame_states"]
    assert raw_cc["proposed_drop_frames"] == [1], raw_cc["proposed_drop_frames"]
    assert raw_cc["face_replace_frames"] == [0, 2], raw_cc["face_replace_frames"]

    # Person words override the label both ways:
    # - drop on frame 0 (facing -> dropped_by_word)
    # - hold on frame 1 (back_of_head -> held_by_word)
    # - no_face on frame 2 (side_with_eye -> no_face_by_word)
    over_cc = htp.cross_check_face_points(
        pose,
        tube,
        seg["face_neck_mask"],
        seg["hair_mask"],
        drop_frames=[0],
        hold_frames=[1],
        no_face_frames=[2],
    )
    assert over_cc["frame_states"] == ["facing", "back_of_head", "side_with_eye"], (
        "raw frame_states must remain the model's 3-state label"
    )
    assert over_cc["effective_frame_states"] == ["dropped_by_word", "held_by_word", "no_face_by_word"], (
        f"expected person words to override all 3 frames, got {over_cc['effective_frame_states']}"
    )
    assert over_cc["overridden_by_word_frames"] == [0, 1, 2], over_cc["overridden_by_word_frames"]
    assert over_cc["face_replace_frames"] == [1] and over_cc["face_lines_on_frames"] == [1], (
        f"expected only held frame 1 in face_replace_frames/face_lines_on_frames, got {over_cc['face_replace_frames']}"
    )
    assert over_cc["proposed_drop_frames"] == [], (
        f"held back_of_head frame must be removed from proposed_drop_frames, got {over_cc['proposed_drop_frames']}"
    )
    return (
        f"raw={over_cc['frame_states']} effective={over_cc['effective_frame_states']} "
        f"overridden={over_cc['overridden_by_word_frames']}"
    )


def _case_real_0613g_three_states() -> str:
    candidates = (
        htp.REPO_ROOT / "data" / "2026-10-17_lead_fp_0613g",
        htp.REPO_ROOT.parent.parent / "custom_nodes" / "ComfyUI-h3-explorations" / "data" / "2026-10-17_lead_fp_0613g",
    )
    cap_dir = next((p for p in candidates if p.is_dir()), None)
    if cap_dir is None:
        return "skipped (data/2026-10-17_lead_fp_0613g not on disk)"
    manifest_path = cap_dir / "manifest.json"
    masks_files = sorted((cap_dir / "subjects" / "lead").glob("masks__*.npz"))
    classes_files = sorted((cap_dir / "subjects" / "lead").glob("classes__*.npz"))
    if not manifest_path.is_file() or not masks_files or not classes_files:
        return "skipped (0613g masks/classes/manifest not on disk)"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = int(manifest["first_frame"])
    assert first == 613, f"expected 0613g first_frame=613, got {first}"

    lead_subj = next((s for s in manifest.get("subjects", []) if s.get("label") == "lead"), {})
    lead_sight = (lead_subj.get("sightings") or [{}])[0]
    pts_str = (lead_sight.get("parts_dropped_off_her_head") or {}).get("points")
    if not pts_str or not Path(pts_str).is_file():
        return "skipped (lead_0613_0974_points.npz not on disk)"
    pts_path = Path(pts_str)

    m_npz = np.load(masks_files[0])
    c_npz = np.load(classes_files[0])
    s_first, s_last = 680, 760
    s0, s1 = s_first - first, s_last - first + 1
    cmap = c_npz["classes"][s0:s1]
    w_canvas = cmap.shape[2]
    track = np.unpackbits(m_npz["track"][s0:s1], axis=-1)[..., :w_canvas].astype(bool)
    parts = np.unpackbits(m_npz["parts"][s0:s1], axis=-1)[..., :w_canvas].astype(bool)
    face_part_names = ("Face_Neck", "Upper_Lip", "Lower_Lip", "Upper_Teeth", "Lower_Teeth", "Tongue")

    res = htp.head_tube_for_capture(
        track,
        parts,
        cmap,
        points_path=pts_path,
        first_frame=s_first,
        head_parts=face_part_names,
    )
    prov = res["provenance"]
    part_px = prov["part_px"]
    states = {s_first + i: s for i, s in enumerate(prov["frame_states"])}
    boh_src = {s_first + i for i in prov["back_of_head_frames"]}
    prop_src = {s_first + i for i in prov["proposed_drop_frames"]}
    side_src = {s_first + i for i in prov["side_with_eye_frames"]}

    # Owner and lead looks on 0613g:
    # - 687-708 and 747-750: back_of_head (hair only, proposed drop)
    # - 709-712 and 733-746: side_with_eye (profile face with an eye / cheek / nose / chin)
    # - 751-760: facing or side_with_eye (face showing as she turns back)
    want_boh = set(range(687, 709)) | set(range(747, 751))
    want_side = set(range(709, 713)) | set(range(733, 747))
    assert want_boh <= boh_src and want_boh <= prop_src, (
        f"expected 687-708 and 747-750 in back_of_head_frames, missing {sorted(want_boh - boh_src)}"
    )
    assert want_side <= side_src, (
        f"expected 709-712 and 733-746 in side_with_eye_frames, got {[states[f] for f in sorted(want_side)]}"
    )
    px_709_712 = [part_px[f - s_first] for f in range(709, 713)]
    px_733_746 = [part_px[f - s_first] for f in range(733, 747)]
    assert min(px_709_712) >= 300, f"expected >=300 px on 709-712, got {px_709_712}"
    assert min(px_733_746) >= 300, f"expected >=300 px on 733-746, got {px_733_746}"
    for f in range(751, 761):
        assert states[f] in ("facing", "side_with_eye"), f"expected face showing on {f}, got {states[f]}"

    # Also verify shot_to_load.capture_face_frames on 0613g catches 709-710 and 733-738 inside the old drop/no_face_on
    import shot_to_load as stl

    old_drop = set(range(687, 711)) | set(range(733, 739)) | {974}
    face_seen = set(stl.capture_face_frames(cap_dir, "lead", old_drop | want_boh))
    flagged = sorted(old_drop & face_seen)
    assert flagged == [709, 710, 733, 734, 735, 736, 737, 738], (
        f"expected capture_face_frames to flag [709, 710, 733..738] inside 0613g's old drop, got {flagged}"
    )
    assert not (want_boh & face_seen), f"expected 0 back_of_head frames in capture_face_frames, got {sorted(want_boh & face_seen)}"
    return (
        f"0613g: boh=687-708,747-750 side_709_712={px_709_712} "
        f"side_733_738={px_733_746[:6]} flagged_in_old_drop={flagged}"
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
            "facing_frames",
            "side_with_eye_frames",
            "back_of_head_frames",
            "proposed_drop_frames",
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

    # Her face points over the line on frames 3..5, where she reappears: the evidence a re-seed needs. Without them nothing is re-seeded.
    shown = [39, 0, 0, 39, 39, 39]
    unseen = htp.run_head_tube_parts(logits, lost_tube, person_track=person_track, seed_frame=0)
    assert unseen["provenance"]["regained_from_person_track_frames"] == [], unseen["provenance"]["regained_from_person_track_frames"]
    assert [r["frame"] for r in unseen["provenance"]["regain_refused"]] == [3, 4, 5], unseen["provenance"]["regain_refused"]
    assert all(r["why"] == "no face points were given" for r in unseen["provenance"]["regain_refused"]), unseen["provenance"]["regain_refused"]
    res = htp.run_head_tube_parts(
        logits,
        lost_tube,
        person_track=person_track,
        seed_frame=0,
        regain_face_points=shown,
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
        regain_face_points=shown,
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


def _case_off_head_islands_not_resurrected() -> str:
    # 1. Deterministic synthetic check (runs even in break_checks copies without data/):
    # Models (a) dad's 1306 bicep island (off_head dropped y=46..60 from parts while cmap still has Face_Neck there),
    # (b) 0606 hands island (off_head dropped y=48..58 from parts while cmap still has Face_Neck there),
    # and (c) 1306t/0438e collar-edge/neck-edge piece kept by off_head in parts.
    face_id = htp.NAME_TO_ID["Face_Neck"]
    person_track = np.zeros((3, 64, 64), dtype=bool)
    person_track[:, 6:62, 10:54] = True

    # Head at y=10..26, x=20..44 (384 px) + kept collar-edge piece at y=28..32, x=26..38 (48 px, like 1306t/0438e)
    pruned_parts = np.zeros((3, 64, 64), dtype=bool)
    pruned_parts[:, 10:26, 20:44] = True
    pruned_parts[:, 28:32, 26:38] = True

    # Uncleaned cmap still holds Face_Neck on the dad's bicep island (frame 0, y=46..60, x=12..28)
    # and on the 0606 hand islands (frames 1..2, y=48..58, x=34..50), both inside person_track.
    cmap_uncleaned = np.zeros((3, 64, 64), dtype=np.uint8)
    cmap_uncleaned[pruned_parts] = face_id
    cmap_uncleaned[0, 46:60, 12:28] = face_id
    cmap_uncleaned[1:3, 48:58, 34:50] = face_id

    syn_res = htp.head_tube_for_capture(
        person_track,
        pruned_parts,
        cmap_uncleaned,
        head_parts=("Face_Neck",),
    )
    syn_pm = syn_res["part_mask"]
    syn_cm = syn_res["class_map"]
    assert int(syn_pm[0, 46:60, 12:28].sum()) == 0, (
        f"expected dad's bicep island to stay 0 px, got {int(syn_pm[0, 46:60, 12:28].sum())}"
    )
    assert int(syn_pm[1:3, 48:58, 34:50].sum()) == 0, (
        f"expected 0606 hand islands to stay 0 px, got {int(syn_pm[1:3, 48:58, 34:50].sum())}"
    )
    assert np.array_equal(syn_pm, pruned_parts), (
        "expected head + kept collar-edge piece to remain unchanged when off_head islands are dropped"
    )
    assert int((syn_cm[:, 46:60, :] == face_id).sum()) == 0, (
        "expected updated class_map to clear off_head dropped islands"
    )

    # 2. Real capture verification when data/ captures are on disk:
    # - dad's 1306 (2026-10-18_third_fp_1306d): 18,474 px bicep island at y>=500 gone (0 px), 23,129 px head at y<500 unchanged
    # - 0606 hands (2026-10-18_lead_fp_0606 frame 772): 5,887 px hand island at y>=440 gone (0 px), 15,695 px head at y<440 unchanged
    # - 1306t (2026-10-18_lead_fp_1306t frame 1391) and 0438e (2026-10-18_lead_fp_0438e frame 444): unchanged (0 px diff)
    data_roots = (
        htp.REPO_ROOT / "data",
        htp.REPO_ROOT.parent.parent / "custom_nodes" / "ComfyUI-h3-explorations" / "data",
    )
    data_root = next((p for p in data_roots if (p / "2026-10-18_third_fp_1306d").is_dir()), None)
    if data_root is None:
        return "synthetic bicep/hands/collar ok (real 10-18 captures not on disk)"

    import capture_masked_run as cmr

    def _run_slice(cap_name: str, subj: str, s0: int, s1: int, s_first: int):
        c_dir = data_root / cap_name
        man = json.loads((c_dir / "manifest.json").read_text(encoding="utf-8"))
        sight = next(s for s in man["subjects"] if s["label"] == subj)["sightings"][0]
        dropped_meta = sight.get("parts_dropped_off_her_head") or {}
        pts_str = dropped_meta.get("points")
        if not pts_str or not Path(pts_str).is_file():
            return None
        margin = int(dropped_meta.get("margin", 16))
        chosen = sight["part_is_made_of"]
        m_npz = np.load(next((c_dir / "subjects" / subj).glob("masks__*.npz")))
        c_npz = np.load(next((c_dir / "subjects" / subj).glob("classes__*.npz")))
        cm_slice = c_npz["classes"][s0:s1]
        w_c = cm_slice.shape[2]
        tr_slice = np.unpackbits(m_npz["track"][s0:s1], axis=-1)[..., :w_c].astype(bool)
        pt_in = np.unpackbits(m_npz["parts"][s0:s1], axis=-1)[..., :w_c].astype(bool)
        pt_pruned, _, _, _ = cmr.off_head(pt_in, pts_str, s_first, margin)
        res = htp.head_tube_for_capture(
            tr_slice,
            pt_pruned,
            cm_slice,
            points_path=pts_str,
            first_frame=s_first,
            head_parts=chosen,
            include_eyeglass="Eyeglass" in chosen,
        )
        return pt_in, pt_pruned, res["part_mask"]

    # Dad's 1306 (frames 1306..1308, k=0..3)
    dad = _run_slice("2026-10-18_third_fp_1306d", "third", 0, 3, 1306)
    if dad is None:
        return "synthetic bicep/hands/collar ok (points file not on disk)"
    _, dad_pruned, dad_pm = dad
    assert int(dad_pm[0, 500:].sum()) == 0, f"expected 0 bicep island px on dad 1306, got {int(dad_pm[0, 500:].sum())}"
    assert int(dad_pm[0, :500].sum()) == 23129 and np.array_equal(dad_pm[0, :500], dad_pruned[0, :500]), (
        f"expected dad 1306 head unchanged at 23129 px, got {int(dad_pm[0, :500].sum())}"
    )

    # 0606 hands (frames 771..773, k=165..168, frame 772 at index 1)
    k0606 = _run_slice("2026-10-18_lead_fp_0606", "lead", 165, 168, 771)
    if k0606 is not None:
        _, k_pruned, k_pm = k0606
        assert int(k_pm[1, 440:].sum()) == 0, f"expected 0 hand island px on 0606 f772, got {int(k_pm[1, 440:].sum())}"
        assert int(k_pm[1, :440].sum()) == 15695 and np.array_equal(k_pm[1, :440], k_pruned[1, :440]), (
            f"expected 0606 f772 head unchanged at 15695 px, got {int(k_pm[1, :440].sum())}"
        )

    # 1306t (frames 1390..1392, k=84..87, frame 1391 at index 1 with 1,361 px collar-edge neck piece)
    t1306 = _run_slice("2026-10-18_lead_fp_1306t", "lead", 84, 87, 1390)
    if t1306 is not None:
        t_in, _, t_pm = t1306
        assert int(t_pm[1].sum()) == 33120 and np.array_equal(t_pm[1], t_in[1]), (
            f"expected 1306t f1391 unchanged at 33120 px, got {int(t_pm[1].sum())}"
        )

    # 0438e (frames 443..445, k=5..8, frame 444 at index 1 with 169 px neck-edge piece)
    e0438 = _run_slice("2026-10-18_lead_fp_0438e", "lead", 5, 8, 443)
    if e0438 is not None:
        e_in, _, e_pm = e0438
        assert int(e_pm[1].sum()) == 8220 and np.array_equal(e_pm[1], e_in[1]), (
            f"expected 0438e f444 unchanged at 8220 px, got {int(e_pm[1].sum())}"
        )

    return "dad 1306 bicep=0/head=23129, 0606 f772 hands=0/head=15695, 1306t f1391=33120, 0438e f444=8220"


def _case_head_track_seed_mask() -> str:
    """When `head_track` is None (default), `head_tube_for_capture` uses its bbox tube; when `head_track` is given,
    it is passed as `seed_mask` to `sam31_head_tube` (clipped to `person_track`), fills unclassified head pixels from
    adjacent frames' head classes, and refuses a wrong-shaped `head_track`."""
    face_id = htp.NAME_TO_ID["Face_Neck"]
    person_track = np.zeros((4, 64, 64), dtype=bool)
    person_track[:, 8:56, 12:52] = True
    # Tight propagated SAM head track at y=12..28, x=20..44 (plus 4 cols outside person_track on frame 0 to test clipping)
    head_track = np.zeros((4, 64, 64), dtype=bool)
    head_track[:, 12:28, 20:44] = True
    head_track[0, 12:28, 52:56] = True
    # On frames 1 and 2, Sapiens left the lower half of the face (y=20..28, x=20..44) unclassified (0), while frames 0 and 3 have Face_Neck there.
    cmap = np.zeros((4, 64, 64), dtype=np.uint8)
    cmap[:, 12:28, 20:44] = face_id
    cmap[1:3, 20:28, 20:44] = 0
    raw_parts = cmap == face_id
    kps = np.zeros((308, 3), dtype=np.float64)
    kps[0] = [28.0, 22.0, 0.9]
    kps[1] = [36.0, 22.0, 0.9]
    kps[2] = [32.0, 26.0, 0.9]
    pose_kps = [kps.copy() for _ in range(4)]

    off_res = htp.head_tube_for_capture(person_track, raw_parts, cmap, head_parts=("Face_Neck",))
    on_res = htp.head_tube_for_capture(person_track, raw_parts, cmap, head_parts=("Face_Neck",), head_track=head_track)
    cmap_kp = cmap.copy()
    htp._fill_unclassified_in_head_tube(
        cmap_kp,
        head_track & person_track,
        part_ids=(face_id,),
        hair_id=htp.NAME_TO_ID["Hair"],
        face_id=face_id,
        pose_kps=pose_kps,
    )
    assert np.all(cmap_kp[1:3, 20:28, 20:44] == face_id), "pose_kps hull did not fill unclassified face hole"
    # A corner of tube_box where reference frames 0 and 3 had neither face nor hair (so face_vote == 0 and hair_vote == 0)
    # and outside kp_crop must not resolve to Face_Neck via 0.0 >= 0.0, or ponytail tips on face-only loads inflate zoom_box.
    # It keeps the part model's own class (Background here): a face-only subject's tube pixels are not relabelled Hair.
    ht_corner = (head_track & person_track).copy()
    ht_corner[:, 8:11, 45:50] = True
    cmap_corner = cmap.copy()
    htp._fill_unclassified_in_head_tube(
        cmap_corner,
        ht_corner,
        part_ids=(face_id,),
        hair_id=htp.NAME_TO_ID["Hair"],
        face_id=face_id,
        pose_kps=pose_kps,
    )
    assert np.all(cmap_corner[1, 8:11, 45:50] == 0), (
        "0 == 0 unclassified corner outside reference face/hair and kp_crop did not keep the part model's own class (Background): "
        f"{np.unique(cmap_corner[1, 8:11, 45:50]).tolist()}"
    )
    assert on_res["provenance"]["outside_person_track_clipped_px"] == [16 * 4, 0, 0, 0], on_res["provenance"]
    assert np.array_equal(on_res["tube_info"]["tube"], head_track & person_track), "tube is not head_track clipped to person_track"
    assert int(on_res["part_mask"][1].sum()) == 16 * 24 and int(on_res["part_mask"][2].sum()) == 16 * 24, (
        f"unclassified face hole inside head_track on frames 1..2 was not filled: {[int(on_res['part_mask'][t].sum()) for t in range(4)]}"
    )
    assert int(off_res["part_mask"][1].sum()) == 8 * 24, "control: without head_track fill, 2-frame hole on frame 1 stays 192 px"
    # C201 item 1 (head_tube_parts.py:1580): when head_track is given and parts has a pruned piece inside head_track alongside an
    # off-head island pruned outside head_track, cmap_arr keeps Sapiens' head classes inside head_track while zeroing the off-head island.
    hair_id = htp.NAME_TO_ID["Hair"]
    app_id = htp.NAME_TO_ID["Apparel"]
    lip_id = htp.NAME_TO_ID["Upper_Lip"]
    cmap_c201 = cmap.copy()
    cmap_c201[:, 14:18, 24:30] = lip_id
    cmap_c201[0, 40:48, 20:36] = face_id  # off-head island on person_track outside head_track
    parts_c201 = ((cmap == face_id) | (cmap_c201 == lip_id)).copy()
    parts_c201[0, 12:20, 20:32] = False   # piece inside head_track that was missing from parts
    res_c201 = htp.head_tube_for_capture(
        person_track, parts_c201, cmap_c201, head_parts=("Face_Neck", "Upper_Lip"), head_track=head_track
    )
    assert np.all(res_c201["part_mask"][0, 12:28, 20:44]), "Sapiens face pixels inside head_track were zeroed when missing from parts"
    assert np.all(res_c201["class_map"][0, 14:18, 24:30] == lip_id), (
        "Sapiens Upper_Lip class inside head_track was zeroed when missing from parts"
    )
    assert not np.any(res_c201["part_mask"][0, 40:48, 20:36]) and not np.any(res_c201["class_map"][0, 40:48, 20:36] == face_id), (
        "off-head island outside head_track was not zeroed in part_mask / class_map"
    )
    # C202/C203/C204: on a head-and-hair load with head_track, part_mask IS head_track & person_track, Sapiens marks only face inside it,
    # and every non-face pixel inside head_track & person_track is assigned Hair (even when Sapiens mislabeled hair as Apparel/Background).
    cmap_hfp = np.zeros((4, 64, 64), dtype=np.uint8)
    cmap_hfp[:, 18:28, 24:40] = face_id   # face inside lower center of head_track (10x16 = 160 px)
    cmap_hfp[:, 24:26, 28:36] = lip_id    # upper lip inside face (2x8 = 16 px)
    cmap_hfp[0, 12:18, 20:44] = hair_id   # frame 0: Sapiens labels top row as Hair
    cmap_hfp[1, 12:18, 20:44] = app_id    # frame 1: Sapiens mislabels all hair as Apparel (1097c/1156c hair-flash bug)
    cmap_hfp[2, 12:18, 20:44] = 0         # frame 2: Sapiens drops hair to Background (0751d/0604c turn bug)
    cmap_hfp[3, 12:18, 20:44] = hair_id   # frame 3: Sapiens labels top row as Hair
    parts_hfp = (cmap_hfp == face_id) | (cmap_hfp == lip_id) | (cmap_hfp == hair_id)
    res_hfp = htp.head_tube_for_capture(
        person_track,
        parts_hfp,
        cmap_hfp,
        head_parts=htp.DEFAULT_HEAD_PARTS,
        head_track=head_track,
    )
    expected_tube = head_track & person_track
    assert np.array_equal(res_hfp["part_mask"], expected_tube), "hfp part_mask with head_track is not equal to head_track & person_track"
    assert res_hfp["provenance"]["part_px"] == [16 * 24] * 4, res_hfp["provenance"]["part_px"]
    assert 1 in res_hfp["provenance"]["dropout_bridged_frames"] and 2 in res_hfp["provenance"]["dropout_bridged_frames"], (
        res_hfp["provenance"]["dropout_bridged_frames"]
    )
    for t in range(4):
        cls_t = res_hfp["class_map"][t]
        assert np.all(cls_t[24:26, 28:36] == lip_id), f"frame {t}: lip class lost inside head_track"
        assert int(res_hfp["seg_info"]["face_neck_mask"][t].sum()) == 160 - 16, f"frame {t}: face_neck_mask px mismatch"
        assert int(res_hfp["seg_info"]["hair_mask"][t].sum()) == (16 * 24) - 160, (
            f"frame {t}: non-face pixels inside head_track were not assigned Hair: {int(res_hfp['seg_info']['hair_mask'][t].sum())}"
        )
    refused(
        lambda: htp.head_tube_for_capture(person_track, raw_parts, cmap, head_parts=("Face_Neck",), head_track=head_track[:2]),
        "head_track shape",
    )
    return "head_track clipped to person_track, 2-frame unclassified hole filled, hfp head_track region + Hair assignment verified, wrong shape refused"


def _case_short_tube_dips_bridged() -> str:
    """Short (<= SHORT_DIP_MAX_FRAMES) sharp dips (< SHORT_DIP_RATIO of smaller flanking anchor) in the head tube are bridged
    from flanking frames clipped to person_track (1306c/f 1402-1404 and 1408-1411); a 7-frame dip or untracked dip is not."""
    num_frames, height, width = 25, 64, 64
    track_bool = np.zeros((num_frames, height, width), dtype=bool)
    track_bool[:, 6:58, 10:54] = True
    # Full head region is 25x40 = 1000 px (y=10..35, x=12..52), inside track_bool
    tube = np.zeros((num_frames, height, width), dtype=bool)
    tube[:, 10:35, 12:52] = True

    # 1. 3-frame dip on frames 2..4 (150 px = 15x10, < 0.6 * 1000 = 600 px)
    tube[2:5] = False
    tube[2:5, 10:25, 12:22] = True
    # 2. 4-frame dip on frames 7..10 (graduated dip: 500, 100, 150, 400 px, all < 600 px)
    tube[7:11] = False
    tube[7, 10:35, 12:32] = True    # 500 px
    tube[8, 10:20, 12:22] = True    # 100 px
    tube[9, 10:25, 12:22] = True    # 150 px
    tube[10, 10:30, 12:32] = True   # 400 px
    # 3. 7-frame dip on frames 13..19 (> SHORT_DIP_MAX_FRAMES = 6, stays unbridged at 150 px)
    tube[13:20] = False
    tube[13:20, 10:25, 12:22] = True
    # 4. 2-frame dip on frames 22..23 where track_bool is empty on frame 22 (stays unbridged)
    tube[22:24] = False
    tube[22:24, 10:25, 12:22] = True
    track_bool[22] = False

    info = htp.sam31_head_tube(tube, seed_frame=0, person_track=track_bool)
    bridged = info["bridged_short_dips"]
    assert bridged == [
        {"span": [2, 4], "before_px": [150, 150, 150], "after_px": [1000, 1000, 1000]},
        {"span": [7, 10], "before_px": [500, 100, 150, 400], "after_px": [1000, 1000, 1000, 1000]},
    ], f"unexpected bridged_short_dips: {bridged}"
    for t in (2, 3, 4, 7, 8, 9, 10):
        assert int(info["tube"][t].sum()) == 1000, f"frame {t} was not bridged to 1000 px"
    for t in range(13, 20):
        assert int(info["tube"][t].sum()) == 150, f"7-frame dip frame {t} should stay 150 px, got {int(info['tube'][t].sum())}"
    assert int(info["tube"][22].sum()) == 0 and int(info["tube"][23].sum()) == 150, (
        "dip with missing person_track on frame 22 was bridged"
    )

    # Also verify through head_tube_for_capture (without head_track, as on 1306c/f):
    face_id = htp.NAME_TO_ID["Face_Neck"]
    cmap = np.where(tube, face_id, 0).astype(np.uint8)
    cap_res = htp.head_tube_for_capture(track_bool, tube, cmap, head_parts=("Face_Neck",))
    assert any(d["span"] == [7, 10] for d in cap_res["provenance"]["bridged_short_dips"]), (
        cap_res["provenance"]["bridged_short_dips"]
    )
    for t in (2, 3, 4, 7, 8, 9, 10):
        assert int(cap_res["part_mask"][t].sum()) == 1000, (
            f"head_tube_for_capture part_mask on bridged dip frame {t} is {int(cap_res['part_mask'][t].sum())}, expected 1000"
        )
    return "3-frame and 4-frame dips bridged to 1000 px in tube and part_mask; 7-frame dip and untracked dip left unbridged"


def _points_file(folder: Path, per_frame: list[tuple[tuple[float, float], tuple[float, float], float]]) -> Path:
    """A 308-point file in the reference step's form, the 39 scored face points of each frame spread over one box at one score:
    `per_frame` is ((x0, y0), (x1, y1), score) a frame."""
    ids = htp.scored_face_ids()
    kps = np.zeros((len(per_frame), 308, 2), dtype=np.float32)
    scores = np.zeros((len(per_frame), 308), dtype=np.float32)
    for t, ((x0, y0), (x1, y1), score) in enumerate(per_frame):
        for j, i in enumerate(ids):
            kps[t, i] = (x0 + (x1 - x0) * ((j % 7) / 6.0), y0 + (y1 - y0) * ((j // 7) / 5.0))
            scores[t, i] = score
    path = folder / "points.npz"
    np.savez(path, keypoints=kps, scores=scores, first_frame=np.int64(0))
    return path


def _case_regain_needs_her_own_face_points() -> str:
    """4912e: where the head track loses her head and her person track still holds her, the top of the person track was her
    raised hand, which her class map calls face and hair. A head is re-seeded there only when at least
    `REGAIN_FACE_POINTS_MIN` of her face points over the line lie on her own replaced classes; otherwise the frame stays empty
    and `regain_refused` says why. Points over the line on her clothing do not count; the same hand with face points over the line
    on it IS re-seeded: the points on her own replaced classes decide."""
    face_id, hair_id, cloth_id = htp.NAME_TO_ID["Face_Neck"], htp.NAME_TO_ID["Hair"], htp.NAME_TO_ID["Upper_Clothing"]
    num_frames, size = 6, 64
    track = np.zeros((num_frames, size, size), dtype=bool)
    head = np.zeros_like(track)
    cmap = np.zeros((num_frames, size, size), dtype=np.uint8)
    for t in (0, 1, 2, 5):                           # her head at the top of her body
        track[t, 20:60, 16:48] = True
        cmap[t, 20:60, 16:48] = cloth_id
        cmap[t, 20:24, 24:40] = hair_id
        cmap[t, 24:32, 24:40] = face_id
    head[0:3, 20:32, 24:40] = True                    # the head track holds her head on 0..2 and loses it on 3..5
    for t in (3, 4):                                  # her head hidden; her raised hand is the top of her track
        track[t, 40:60, 16:48] = True
        cmap[t, 40:60, 16:48] = cloth_id
        track[t, 4:20, 8:14] = True
        cmap[t, 4:10, 8:14] = hair_id                 # the class map reads her hand as hair and face
        cmap[t, 10:20, 8:14] = face_id
    parts = np.isin(cmap, [face_id, hair_id])
    on_face, on_hand = ((25.0, 25.0), (38.0, 31.0)), ((8.0, 5.0), (13.0, 19.0))
    with tempfile.TemporaryDirectory() as tmp:
        under = _points_file(Path(tmp), [(*on_face, 0.9)] * 3 + [(*on_hand, 0.1)] * 2 + [(*on_face, 0.9)])
        res = htp.head_tube_for_capture(track, parts, cmap, head_track=head, points_path=under, head_parts=htp.DEFAULT_HEAD_PARTS)
        prov = res["provenance"]
        assert prov["regained_from_person_track_frames"] == [5], prov["regained_from_person_track_frames"]
        refused_frames = [(r["frame"], r["face_points_on_own_part"]) for r in prov["regain_refused"]]
        assert refused_frames == [(3, 0), (4, 0)], refused_frames
        for t in (3, 4):
            assert int(res["tube_info"]["tube"][t].sum()) == 0, f"frame {t}: her hand was re-seeded as her head ({int(res['tube_info']['tube'][t].sum())} px)"
            assert int(res["part_mask"][t].sum()) == 0, f"frame {t}: her hand is in her part"
            assert int(((res["class_map"][t] == hair_id) & track[t]).sum()) == 0, f"frame {t}: her hand was relabelled Hair"
        assert int(res["tube_info"]["tube"][5].sum()) > 0, "frame 5: her face shows with its points and was not re-seeded"
        on_body = ((20.0, 44.0), (44.0, 56.0))           # over the line, inside her track, on her clothing: not her own replaced classes
        body = _points_file(Path(tmp), [(*on_face, 0.9)] * 3 + [(*on_body, 0.9)] * 2 + [(*on_face, 0.9)])
        clothed = htp.head_tube_for_capture(track, parts, cmap, head_track=head, points_path=body, head_parts=htp.DEFAULT_HEAD_PARTS)
        assert clothed["provenance"]["regained_from_person_track_frames"] == [5], clothed["provenance"]["regained_from_person_track_frames"]
        over = _points_file(Path(tmp), [(*on_face, 0.9)] * 3 + [(*on_hand, 0.9)] * 2 + [(*on_face, 0.9)])
        shown = htp.head_tube_for_capture(track, parts, cmap, head_track=head, points_path=over, head_parts=htp.DEFAULT_HEAD_PARTS)
        assert shown["provenance"]["regained_from_person_track_frames"] == [3, 4, 5], shown["provenance"]["regained_from_person_track_frames"]
    return (f"regained {prov['regained_from_person_track_frames']}, refused {refused_frames} (least {htp.REGAIN_FACE_POINTS_MIN}); "
            f"with points over the line on the hand: regained {shown['provenance']['regained_from_person_track_frames']}")


def _case_source_facial_hair_kept_under_the_jaw() -> str:
    """THE SOURCE'S FACIAL HAIR (2026-10-23, third_fp_5011b and 5675c): the head track ends at the jaw, and the tube cut the beard and
    neck that are one piece with his face in the preview's part. Without `keep_facial_hair` nothing changes (the control). With it, the
    piece of the part joined to the face outside the tube, inside the person track and the face's columns widened by
    FACE_COLUMN_PAD_PX, less another person's mask, is given back with its own class, on every frame, no pose point read; what the
    columns cut is recorded; the provenance's part_px counts it and off_tube_suppressed_px loses it. The part model's Hair beside the
    face, a beard the map calls clothing, a piece not joined to the face and a dropped frame keep nothing; the tube's own part and the
    face states are the same; a part of hair is refused."""
    face_id, hair_id, cloth_id = htp.NAME_TO_ID["Face_Neck"], htp.NAME_TO_ID["Hair"], htp.NAME_TO_ID["Upper_Clothing"]
    n = 5
    track = np.zeros((n, 64, 64), dtype=bool)
    track[:, 8:62, 8:60] = True
    track[:, 38:40, 26:38] = False                     # the beard's last two rows lie outside his person track
    head = np.zeros_like(track)
    head[:, 12:30, 20:44] = True                       # the head track ends on row 29, at the jaw
    cmap = np.zeros((n, 64, 64), dtype=np.uint8)
    cmap[:, 46:62, 8:60] = cloth_id
    cmap[:, 18:30, 20:44] = face_id                    # his face, inside the tube
    cmap[:, 30:40, 26:38] = face_id                    # his beard and neck under the jaw, one piece with the face
    cmap[:, 14:40, 14:20] = hair_id                    # his head hair beside the face, falling below the jaw
    cmap[:, 52:54, 14:18] = face_id                    # a Face_Neck speck on his shirt that does not join the face
    cmap[0, 36:40, 38:58] = face_id                    # frame 0: a hand the map calls Face_Neck, touching the beard, past the face's columns
    cmap[2, 30:40, 26:38] = cloth_id                   # frame 2: his arm over the beard, which the map calls clothing
    parts = cmap == face_id
    others = np.zeros_like(track)
    others[3, 34:36, 26:38] = True                     # frame 3: another person's mask across it, cutting its lower rows off the face
    beard = np.zeros((64, 64), dtype=bool)
    beard[30:38, 26:38] = True
    given = dict(head_parts=("Face_Neck",), head_track=head)
    off = htp.head_tube_for_capture(track, parts, cmap, **given)
    on = htp.head_tube_for_capture(track, parts, cmap, **given, keep_facial_hair=True, others=others)
    dropped = htp.head_tube_for_capture(track, parts, cmap, **given, keep_facial_hair=True, others=others, drop_frames={4})
    refused(lambda: htp.head_tube_for_capture(track, parts, cmap, head_parts=("Face_Neck", "Hair"), head_track=head, keep_facial_hair=True),
            "face-only part", "before the head tube writes anything")
    # the control: the tube cuts the beard the part held, and says nothing of facial hair
    assert all(int((off["part_mask"][t] & beard).sum()) == 0 for t in range(n)), "control: the tube did not cut the beard under the jaw"
    assert "source_facial_hair" not in off["provenance"], off["provenance"].keys()
    fh = on["provenance"]["source_facial_hair"]
    # THE CASE THAT GOES RED when the face's own piece below the jaw is not given back, with no pose point given, or more is
    assert fh["kept_px"] == [152, 96, 0, 48, 96] and fh["kept_on_frames"] == [0, 1, 3, 4], (fh["kept_px"], fh["kept_on_frames"])
    assert fh["kept_px_by_class"] == {"Face_Neck": [152, 96, 0, 48, 96]}, fh["kept_px_by_class"]
    assert not on["part_mask"][:, 38:40, 26:38].any(), "the part outside his person track was kept"
    for t in (0, 1, 4):
        assert np.all(on["part_mask"][t][beard]) and np.all(on["class_map"][t][beard] == face_id), f"frame {t}: the beard is not in the part"
    # THE CASE THAT GOES RED when a hand the map calls Face_Neck is kept past the face's columns, unrecorded
    rows = {r["frame"]: r for r in fh["frames"]}
    assert rows[0]["face_columns"] == [12, 51] and rows[0]["cut_by_the_columns_px"] == 24, rows[0]
    assert on["part_mask"][0, 36:40, 38:52].all() and not on["part_mask"][0, 36:40, 52:58].any(), "the columns did not bound the hand"
    assert fh["cut_by_the_columns_px"] == [24, 0, 0, 0, 0], fh["cut_by_the_columns_px"]
    assert not on["part_mask"][:, 14:40, 14:20].any(), "the part model's Hair beside the face was kept"
    # another person's pixels are not kept, nor what they cut off from the face
    assert not on["part_mask"][3, 34:40, 26:38].any() and rows[3]["in_another_persons_mask_px"] == 24, rows[3]
    assert rows[3]["cut_off_from_the_face_px"] == 24, rows[3]
    assert not on["part_mask"][:, 52:54, 14:18].any(), "a Face_Neck piece that does not join the face was kept"
    assert "no piece of the part joins the face" in rows[2]["nothing_kept_because"], rows[2]
    assert fh["cannot_see"] == list(htp.FACIAL_HAIR_CANNOT_SEE), fh["cannot_see"]
    # what was kept, in the provenance's own figures, with the tube's own beside them
    assert fh["part_px_before_keeping"] == off["provenance"]["part_px"], (fh["part_px_before_keeping"], off["provenance"]["part_px"])
    assert on["provenance"]["part_px"] == [a + b for a, b in zip(off["provenance"]["part_px"], fh["kept_px"])], on["provenance"]["part_px"]
    assert fh["kept_px_it_counted_as_suppressed"] == fh["kept_px"], fh["kept_px_it_counted_as_suppressed"]
    assert on["provenance"]["off_tube_suppressed_px"] == [a - b for a, b in zip(off["provenance"]["off_tube_suppressed_px"], fh["kept_px"])], \
        (on["provenance"]["off_tube_suppressed_px"], off["provenance"]["off_tube_suppressed_px"])
    tube = on["tube_info"]["tube"]
    assert np.array_equal(on["part_mask"] & tube, off["part_mask"] & off["tube_info"]["tube"]), "the tube's own part changed"
    for key in ("facing_frames", "side_with_eye_frames", "back_of_head_frames", "frame_states"):
        assert on["provenance"][key] == off["provenance"][key], (key, on["provenance"][key], off["provenance"][key])
    drop_fh = dropped["provenance"]["source_facial_hair"]
    assert drop_fh["kept_px"][4] == 0 and "dropped" in drop_fh["frames"][4]["nothing_kept_because"], drop_fh["frames"][4]
    return (f"kept {fh['kept_px']} px a frame with no pose point (control: 0); the hand past the face's columns cut ({fh['cut_by_the_columns_px'][0]} px, "
            "recorded); Hair, clothing, an unjoined speck and a dropped frame keep nothing; the counts say what was kept")


def _case_face_only_tube_keeps_its_classes() -> str:
    """A face-only subject's head-tube pixels the part model did not call face keep its own class: the neighbour fill writes
    only the face, never Hair (a hair-and-face subject's non-face tube pixels are Hair, `head_track_seed_mask_and_neighbour_fill`).
    Its part is the face alone either way."""
    face_id, hair_id, app_id = htp.NAME_TO_ID["Face_Neck"], htp.NAME_TO_ID["Hair"], htp.NAME_TO_ID["Apparel"]
    track = np.zeros((4, 64, 64), dtype=bool)
    track[:, 8:56, 12:52] = True
    head = np.zeros_like(track)
    head[:, 12:28, 20:44] = True
    cmap = np.zeros((4, 64, 64), dtype=np.uint8)
    cmap[:, 18:28, 20:44] = face_id
    cmap[[0, 3], 12:18, 20:44] = hair_id              # frames 0 and 3: her hair above her face
    cmap[1:3, 12:18, 20:44] = app_id                  # frames 1 and 2: the part model reads that hair as Apparel
    parts = cmap == face_id
    res = htp.head_tube_for_capture(track, parts, cmap, head_parts=("Face_Neck",), head_track=head)
    hair_px = [int(res["seg_info"]["hair_mask"][t, 12:18, 20:44].sum()) for t in range(4)]
    assert hair_px[1] == 0 and hair_px[2] == 0, f"a face-only subject's Apparel tube pixels were relabelled Hair: {hair_px}"
    assert hair_px[0] == 6 * 24 and hair_px[3] == 6 * 24, f"the part model's own Hair was lost: {hair_px}"
    for t in range(4):
        assert int(res["part_mask"][t].sum()) == 10 * 24, f"frame {t}: the face part changed: {int(res['part_mask'][t].sum())}"
    return f"Hair px above the face by frame {hair_px}; the part is the face alone, {int(res['part_mask'][1].sum())} px a frame"


def main() -> int:
    case("off_tube_false_positive_suppressed", _case_off_tube_false_positive)
    case("temporal_smoothing_bridges_dropout", _case_temporal_smoothing_dropout)
    case("face_points_cross_check_turned_away_vs_dropout", _case_face_points_cross_check)
    case("person_word_overrides_three_states", _case_person_word_overrides_three_states)
    case("person_track_bounds_tube", _case_person_track_bounds_tube)
    case("person_track_regains_tube_after_gap", _case_person_track_regains_tube_after_gap)
    case("regain_needs_her_own_face_points", _case_regain_needs_her_own_face_points)
    case("face_only_tube_keeps_its_classes", _case_face_only_tube_keeps_its_classes)
    case("source_facial_hair_kept_under_the_jaw", _case_source_facial_hair_kept_under_the_jaw)
    case("off_head_islands_not_resurrected", _case_off_head_islands_not_resurrected)
    case("head_track_seed_mask_and_neighbour_fill", _case_head_track_seed_mask)
    case("short_tube_dips_bridged", _case_short_tube_dips_bridged)
    case("real_0613g_three_states", _case_real_0613g_three_states)
    case("provenance_json_and_cli", _case_provenance_and_cli)
    case("refuses_invalid_inputs", _case_refuses_invalid_inputs)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
