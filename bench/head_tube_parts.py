#!/usr/bin/env python3
"""Head-tube conditioned Sapiens2-Seg part masks and Sapiens2-Pose 39-point face cross-check.

Couples three signals on a shot so head/face part masks stay temporally smooth
and bounded without per-frame tight-crop jitter or off-head false positives:

1. ``sam31_head_tube``: seeds SAM 3.1 on the subject's head/face on the shot's
   seed frame (with ``[-1, 1]`` input normalization, exact ``F.gelu`` on the text
   tower, and ``class_score * presence_score`` filtering) to produce a continuous
   per-frame boolean head tube ``(T, H, W)`` and one fixed even-snapped bounding
   box ``shot_box = [x0, y0, x1, y1]`` across the entire shot.
2. ``smooth_seg_in_tube``: runs or consumes ``Sapiens2ForSemanticSegmentation``
   logits on ``shot_box`` (with only other people's tracks masked to neutral
   grey and ``cv2.INTER_AREA`` / ``cv2.INTER_LINEAR`` resizing), bilinearly
   upsamples logits, applies a 3-frame temporal filter ``[0.25, 0.50, 0.25]`` on
   the class probabilities, and restricts head/facial classes to lie inside the
   SAM 3.1 head tube.
3. ``cross_check_face_points``: cross-checks the 39 scored facial keypoints from
   the 308-keypoint Sapiens2-Pose layout (excluding unscored pupils) against the
   head tube and smoothed ``Face_Neck`` / ``Hair`` / part masks to label each
   frame with one of three face states: ``facing``, ``side_with_eye`` (face skin
   and at least one eye showing: replace, face lines on), or ``back_of_head``
   (hair only: a proposal that empties nothing by itself, listed for a look).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from _lib import face_points as fp  # noqa: E402
from _lib import npz_writer_fields, writer_stamp  # noqa: E402

SEG_CLASSES_PATH = (
    REPO_ROOT / "vendor" / "meta_sapiens2" / "seg_classes29.json"
)  #: inherited: upstream Sapiens2 29-class table path (sapiens2_parts.CLASSES_PATH)
POSE_KEYPOINTS_PATH = (
    REPO_ROOT / "vendor" / "meta_sapiens2" / "pose_keypoints308.json"
)  #: inherited: upstream Sapiens2 308-keypoint table path (pose_along_shot.Pose308)

DEFAULT_TEMPORAL_WEIGHTS: tuple[float, ...] = (
    0.25,
    0.50,
    0.25,
)  #: reasoned: 3-frame symmetric temporal kernel bridging 1-2 frame Sapiens2-Seg dropouts inside the SAM 3.1 head tube
DEFAULT_BOX_PAD_FRAC: float = (
    0.20  #: inherited: 20% padding around the shot-union head tube box (pose_along_shot.PAD_FRAC)
)
MIN_BOX_PX: int = (
    16  #: reasoned: minimum even crop side in pixels for Sapiens2-Seg on the shot-fixed box
)
OTHER_PERSON_FILL_RGB: int = (
    128  #: inherited: neutral mid-grey fill for masking other people's tracks only (pose_along_shot.ALONE_FILL)
)
DEFAULT_HEAD_PARTS: tuple[str, ...] = (
    "Face_Neck",
    "Hair",
    "Upper_Lip",
    "Lower_Lip",
    "Upper_Teeth",
    "Lower_Teeth",
    "Tongue",
)  #: reasoned: Sapiens2-Seg head and facial part classes restricted to lie inside the SAM 3.1 head tube
OPTIONAL_HEAD_PARTS: tuple[str, ...] = (
    "Eyeglass",
)  #: reasoned: optional head-mounted class included inside the SAM 3.1 head tube when requested
DEFAULT_PART_PROB_FLOOR: float = (
    0.35  #: reasoned: temporally smoothed head-part probability threshold bridging 1-2 frame dropouts inside the head tube
)
TURNED_AWAY_HAIR_SHARE_MIN: float = (
    0.85  #: measured: shot 2 frames 687-708 and 747-750 back-of-head frames hold >85% Hair inside the SAM 3.1 head tube
)
TURNED_AWAY_FACE_SHARE_MAX: float = (
    0.05  #: measured: shot 2 frames 687-708 and 747-750 back-of-head frames hold <5% Face_Neck inside the SAM 3.1 head tube
)
SIDE_PROFILE_MIN_AREA_PX: int = (
    300  #: measured: shot 2 profile frames 709-712 (368-588 px) and 733-746 (378-4391 px) vs <=238 px hair-edge specks on 687-708 and 747-750
)
SIDE_PROFILE_MIN_WIDTH_PX: int = (
    24  #: measured: shot 2 profile face main piece bounding-box width >=24 px on 709-711 and 733-746
)
SIDE_PROFILE_NEIGHBOR_MIN_WIDTH_PX: int = (
    19  #: measured: shot 2 frame 712 narrow profile slice width (19 px) adjacent to frame 711 (26 px width, 487 px area)
)
SIDE_PROFILE_NEIGHBOR_MIN_AREA_PX: int = (
    480  #: measured: shot 2 frame 711 profile face area (487 px) supporting adjacent narrow profile slice on frame 712
)
FACING_PART_TUBE_SHARE_MIN: float = (
    0.11  #: measured: shot 2 front/three-quarter facing frames hold >=11% part share of the head tube with >=20 scored face points in part
)
FACING_MIN_POINTS_IN_PART: int = (
    20  #: measured: shot 2 front/three-quarter facing frames hold >=20 of 39 scored facial keypoints inside the part mask
)
SUBJECT_MARGIN_PX: int = (
    16  #: inherited: Sapiens2 subject track dilation margin in pixels where attached profile face pieces live (sapiens2_parts.SUBJECT_MARGIN)
)
TRACK_TOUCH_DILATE_PX: int = (
    4  #: reasoned: 4 px bridge tolerance for connected profile face pieces touching the subject's person track inside SUBJECT_MARGIN_PX
)
DEFAULT_FACE_POINT_MIN_SCORE: float = (
    0.30  #: inherited: minimum confidence score for scored Sapiens2-Pose facial landmarks (face_points.FACE_MIN_SCORE)
)
SAM31_SCORE_FLOOR: float = (
    0.30  #: inherited: minimum combined class_score * presence_score for SAM 3.1 head proposals (sam31_corrections.SCORE_FLOOR)
)
SYNTHETIC_CLASS_LOGIT: float = (
    5.0  #: reasoned: log-odds assigned to the discrete Sapiens2-Seg class map winner when smoothing inside the head tube
)
SHORT_DIP_MAX_FRAMES: int = (
    6  #: measured: Hall 1306-1418 (third_fp_1306f vs 1306c, 2026-10-19), up to 6 frames (0.25 s at 24 fps) bridges the 3-frame (1402-1404) and 4-frame (1408-1411) head-tube dips
)
SHORT_DIP_RATIO: float = (
    0.6  #: measured: Hall 1306-1418 (2026-10-19), 1403 dipped to 20,129 px vs 33,748/36,052 px and 1409-1410 dipped to 2,413-5,736 px vs 36,585/47,200 px (<60% of smaller flanking anchor)
)
SHORT_DIP_BOUND_LEAST_SHARE: float = (
    0.25  #: reasoned (2026-10-23): a frame bounds a short dip only when its tube holds a whole head, at least this share of the shot's own median tube px over the frames that have a tube; with no floor a 1-5 px speck bounded a dip and let a 7-frame gap be bridged (a blind reader's cases, check_head_tube_parts::_case_short_dip_bounds_are_whole_heads). Chosen under every bound of every dip bridged in data/ before it (the lowest, 0.32 of its shot's median, second_hfp_1093b's 1098, 2026-10-20; second_hfp_4647b's 4690 is 0.40) and far over a speck; a starting value
)
REGAIN_FACE_POINTS_MIN: int = (
    9  #: measured: 2026-10-21, on every frame the head tube re-seeded from the person track on second_hfp_4912e, 1093c, 4647d, 4726c, 5011e and 5889e and lead_fp_0606d (own_part_face_points on each capture's own inputs): at most 1 such point where the seed held no head of hers (4912e's 4943; her raised hand on 4921-4923 and 4941-4945, which her class map calls face and hair), at least 17 where it held her face (4726c's 4751); the gap's midpoint, a starting value. Not seen by it: 5889e's 5923 and 5926, 38 points piled at the frame's edge as she leaves it
)
#: Minimum SAM 3.1 `head_track` seed pixels inside `raw_track_bool` required before a frame can expand into connected
#: upward crown components (`head_tube_for_capture`, 2026-10-28). **Reasoned starting value, not a universal measurement**,
#: set against `second_hfp_4912n` frame `4926` (`1 px` `.mp4` compression speck at `(691, 505)` inside `raw_track_bool`,
#: or `12 px` on `4922`, which must NOT seed component expansion into `5,209 px` of `parts_bool` and inflate `4926` from
#: `818 px` to `6,956 px`) vs frames `4927-4932` (`>= 2,715 px` lower-face SAM `head_track` masks: `2,715, 4,643, 4,583,
#: 4,454, 4,165, 4,108 px`).
_SPECK_MIN_PX: int = 64
#: Minimum connected `parts_bool & raw_track_bool` pixels strictly above `hy0` (`y < hy0`) required to join an upward
#: crown extension on a `head_and_hair_on_head_track` load while keeping the lower bound `y <= hy1` (`head_tube_for_capture`,
#: 2026-10-28). **Reasoned starting value, not a universal measurement**, set against `second_hfp_4912n` frames `4927-4932`
#: (`2,607-2,813 px` missing crowns above `hy0` inside `raw_track_bool`: `2,813, 2,724, 2,729, 2,729, 2,706, 2,607 px`) vs
#: `second_hfp_4647i` (`2-17 px` boundary jitter above `hy0`, including `12 px` on `4671` and `2 px` on `4696`),
#: `second_hfp_5114k` (`<= 5 px`), and `second_hfp_1039h` (`<= 7 px`).
_MIN_CROWN_ABOVE_HY0_PX: int = 256
#: THE SOURCE'S FACIAL HAIR (`face_piece_kept_whole`, 2026-10-23): how far past the columns of the tube's own part, on each side, the
#: piece of the part kept below the tube may reach. **Reasoned**, not measured: the edge of a beard and the part model's wobble at the
#: jaw line, set to the 8 px the job holds a part to its own track by (h3_config.MASKED_STRANGERS['grow_px'], the job's
#: `parts_held_to_track_px`). It stops only what lies to the side of the face: a hand or thing inside the face's columns and joined to
#: the face is kept (FACIAL_HAIR_CANNOT_SEE). What it cut on the replays of 2026-10-23, recorded a frame as `cut_by_the_columns_px`:
#: 16,313 px on 20 of third_fp_5675c's 41 frames, 22 px on 3 of 5011b's 33, none on 4726b, 4602c or 5461b; on second_hfp_4912e (a
#: reader's replay) none on its 35 frames, where her raised hand was kept out by the tube's part being empty (her head track lost on
#: 4921-4926 and 4941-4946), not by the columns.
FACE_COLUMN_PAD_PX: int = 8
#: Downward jaw/beard extension fraction of the face height (`max(fh_h, med_fh)`) in each face column when `keep_facial_hair=True`
#: (`face_piece_kept_whole`, 2026-10-24, replacing 0764f585's `Face_Neck`-only assumption). **Reasoned starting value**, verified on
#: CPU against the four raw preview class videos before any render:
#: - `third_fp_5675f` (med_fh=204 px): `0764f585` kept 0 px on 10/37 headed frames where Sapiens labelled the beard `Hair` / `Upper_Clothing`
#:   or where `tube[t]` already enclosed the beard (5683-5687); `0.35` keeps the beard on 37/37 headed frames (min 112 px, med 11,567 px).
#: - `third_fp_5011e` (med_fh=100 px): `0764f585` missed 7/33 frames (5020-5022, 5038-5041); `0.35` keeps 33/33 (min 566 px, med 3,080 px).
#: - `third_fp_5461d` (med_fh=103 px): `0764f585` missed 16/37 frames; `0.35` keeps 37/37 (min 691 px, med 3,888 px).
#: - `third_fp_4479d` (med_fh=64 px): `0764f585` missed 29/31 frames (`Upper_Clothing` / `Torso`); `0.35` keeps 31/31 (min 1,122 px, med 1,584 px).
JAW_BEARD_PAD_FRAC: float = 0.35
#: Sapiens body-part classes excluded from the downward jaw/beard extension so hands or arms crossing in front of the chin (`5011e`:
#: 11,685 px; `5461d`: 9,512 px; `4479d`: 94 px) are not pulled into the beard mask.
JAW_BEARD_EXCLUDE_CLASSES: tuple[str, ...] = (
    "Left_Hand",
    "Right_Hand",
    "Left_Lower_Arm",
    "Right_Lower_Arm",
    "Left_Upper_Arm",
    "Right_Upper_Arm",
)
FACIAL_HAIR_CANNOT_SEE: tuple[str, ...] = (
    "a hand or arm the part model calls Face_Neck where it touches the face (4912e's raised hand): kept where it lies in the face's "
    "columns; what the columns cut is recorded a frame (`cut_by_the_columns_px`); hands and arms labelled as hand/arm classes in the "
    "class map (`JAW_BEARD_EXCLUDE_CLASSES`) are excluded from the downward jaw extension (`excluded_hand_arm_px`)",
    "a beard wider than the face's columns and FACE_COLUMN_PAD_PX (a profile where the face inside the tube is a sliver): cut there, "
    "and recorded",
    "a piece of the part not joined to the face and below JAW_BEARD_PAD_FRAC of the face height: not kept",
    "where the beard ends: the connected Face_Neck piece is the neck as well as the beard, and the downward jaw extension reaches "
    "JAW_BEARD_PAD_FRAC of the face height below each column of the face",
)


def _load_seg_classes() -> tuple[tuple[str, ...], dict[str, int]]:
    with open(SEG_CLASSES_PATH, encoding="utf-8") as f:
        names = tuple(json.load(f)["names"])
    return names, {name: idx for idx, name in enumerate(names)}


def _load_pose_keypoint_index() -> dict[str, int]:
    with open(POSE_KEYPOINTS_PATH, encoding="utf-8") as f:
        names = json.load(f)["names"]
    index = {name: idx for idx, name in enumerate(names)}
    if "tip_of_nose" not in index and "nose" in index:
        index["tip_of_nose"] = index["nose"]
    return index


CLASS_NAMES, NAME_TO_ID = _load_seg_classes()
POSE_KEYPOINT_INDEX = _load_pose_keypoint_index()


def sam31_normalize_input(x: torch.Tensor) -> torch.Tensor:
    """Normalize ``[0, 1]`` or ``[0, 255]`` image tensor to SAM 3.1's ``[-1, 1]`` range."""
    if not isinstance(x, torch.Tensor):
        raise SystemExit("sam31_normalize_input: input must be a torch.Tensor")
    if x.numel() == 0 or x.ndim < 3:
        raise SystemExit(
            f"sam31_normalize_input: expected non-empty image tensor with ndim >= 3, got shape {tuple(x.shape)}"
        )
    xf = x.float()
    if float(xf.max().item()) > 1.0 + 1e-3:
        xf = xf / 255.0
    return (xf.clamp(0.0, 1.0) * 2.0) - 1.0


def sam31_combined_score(
    pred_logits: torch.Tensor,
    presence_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute SAM 3.1 proposal scores via ``class_score * presence_score``."""
    if pred_logits.numel() == 0:
        raise SystemExit("sam31_combined_score: pred_logits is empty")
    scores = pred_logits.sigmoid()
    if presence_logits is not None:
        scores = scores * presence_logits.sigmoid()
    return scores


def sam31_enforce_exact_gelu(module: nn.Module) -> int:
    """Replace any approximate GELU inside a SAM 3.1 text tower with exact ``F.gelu``."""
    patched = 0
    for sub in module.modules():
        if isinstance(sub, nn.GELU) and getattr(sub, "approximate", "none") != "none":
            sub.approximate = "none"
            patched += 1
    return patched


def _snap_even_box(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
    min_box_px: int = MIN_BOX_PX,
) -> list[int]:
    if width < 2 or height < 2:
        raise SystemExit(f"shot_box_from_tube: invalid frame dimensions ({height}, {width})")
    ix0 = max(0, int(np.floor(x0)) & ~1)
    iy0 = max(0, int(np.floor(y0)) & ~1)
    ix1 = min(width & ~1, ((int(np.ceil(x1)) + 1) & ~1))
    iy1 = min(height & ~1, ((int(np.ceil(y1)) + 1) & ~1))
    min_w = min(width & ~1, max(2, int(min_box_px) & ~1))
    min_h = min(height & ~1, max(2, int(min_box_px) & ~1))
    if ix1 - ix0 < min_w:
        ix1 = min(width & ~1, ix0 + min_w)
        ix0 = max(0, ix1 - min_w)
    if iy1 - iy0 < min_h:
        iy1 = min(height & ~1, iy0 + min_h)
        iy0 = max(0, iy1 - min_h)
    return [int(ix0), int(iy0), int(ix1), int(iy1)]


def shot_box_from_tube(
    tube: np.ndarray | torch.Tensor,
    frame_shape: tuple[int, int] | None = None,
    pad_frac: float = DEFAULT_BOX_PAD_FRAC,
) -> list[int]:
    """Compute one fixed even-snapped bounding box ``[x0, y0, x1, y1]`` across the entire shot."""
    if isinstance(tube, torch.Tensor):
        tube_np = tube.detach().cpu().numpy()
    else:
        tube_np = np.asarray(tube)
    if tube_np.ndim != 3 or tube_np.shape[0] < 1:
        raise SystemExit(
            f"shot_box_from_tube: expected 3D head tube (T, H, W) with T >= 1, got shape {tube_np.shape}"
        )
    if pad_frac < 0.0:
        raise SystemExit(f"shot_box_from_tube: pad_frac must be >= 0, got {pad_frac}")
    tube_bool = tube_np > 0
    if not np.any(tube_bool):
        raise SystemExit("shot_box_from_tube: head tube is empty across all frames in the shot")
    _, h_tube, w_tube = tube_bool.shape
    height, width = (int(frame_shape[0]), int(frame_shape[1])) if frame_shape is not None else (h_tube, w_tube)
    if (height, width) != (h_tube, w_tube):
        raise SystemExit(
            f"shot_box_from_tube: frame_shape {(height, width)} does not match tube spatial shape {(h_tube, w_tube)}"
        )
    union_2d = np.any(tube_bool, axis=0)
    ys, xs = np.nonzero(union_2d)
    x0_raw, x1_raw = float(xs.min()), float(xs.max() + 1)
    y0_raw, y1_raw = float(ys.min()), float(ys.max() + 1)
    pad_x = (x1_raw - x0_raw) * float(pad_frac)
    pad_y = (y1_raw - y0_raw) * float(pad_frac)
    return _snap_even_box(
        x0_raw - pad_x,
        y0_raw - pad_y,
        x1_raw + pad_x,
        y1_raw + pad_y,
        width=width,
        height=height,
    )


def mask_other_people_only(
    frames: np.ndarray,
    other_people_tube: np.ndarray | None = None,
    fill_rgb: int = OTHER_PERSON_FILL_RGB,
) -> np.ndarray:
    """Mask only other people's tracks to neutral grey, keeping the subject's context intact."""
    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise SystemExit(
            f"mask_other_people_only: expected frames (T, H, W, 3), got shape {arr.shape}"
        )
    out = arr.copy()
    if other_people_tube is None:
        return out
    other = np.asarray(other_people_tube) > 0
    if other.shape != arr.shape[:3]:
        raise SystemExit(
            f"mask_other_people_only: other_people_tube shape {other.shape} does not match frames {arr.shape[:3]}"
        )
    out[other] = int(fill_rgb)
    return out


def resize_crop(crop: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize an image crop using ``cv2.INTER_AREA`` when shrinking and ``cv2.INTER_LINEAR`` when enlarging."""
    if crop.ndim not in (2, 3) or crop.size == 0:
        raise SystemExit(f"resize_crop: expected non-empty 2D or 3D crop, got shape {crop.shape}")
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if target_h <= 0 or target_w <= 0:
        raise SystemExit(f"resize_crop: invalid target_hw {target_hw}")
    h, w = crop.shape[:2]
    if (h, w) == (target_h, target_w):
        return crop.copy()
    interp = cv2.INTER_AREA if (target_h < h or target_w < w) else cv2.INTER_LINEAR
    return cv2.resize(crop, (target_w, target_h), interpolation=interp)


def _head_seed_from_person_mask(
    person_mask_2d: np.ndarray,
    head_rel_y1: float = 0.40,
    own_head_2d: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Extract the head region and bounding box from a single-frame 2D person-track mask.
    When ``own_head_2d`` (Sapiens head/hair pixels on the frame) is provided and non-empty inside
    ``person_mask_2d``, take ``own_head_2d & person_mask_2d`` directly rather than slicing a
    horizontal rectangle of ``person_mask_2d`` that would union ``Upper_Clothing``/``Torso``/``Hands``."""
    ys, xs = np.nonzero(person_mask_2d)
    if len(ys) == 0:
        return np.zeros_like(person_mask_2d, dtype=bool), [0, 0, 0, 0]
    if own_head_2d is not None:
        own_in_person = np.asarray(own_head_2d > 0, dtype=bool) & person_mask_2d
        if np.any(own_in_person):
            sys_y, sxs_x = np.nonzero(own_in_person)
            box = [int(sxs_x.min()), int(sys_y.min()), int(sxs_x.max()) + 1, int(sys_y.max()) + 1]
            return own_in_person, box
    py0, py1 = int(ys.min()), int(ys.max()) + 1
    px0, px1 = int(xs.min()), int(xs.max()) + 1
    rel = min(max(float(head_rel_y1), 0.25), 1.0)
    hy1 = max(py0 + 1, int(np.ceil(py0 + rel * (py1 - py0))))
    seed = np.zeros_like(person_mask_2d, dtype=bool)
    seed[py0:hy1, px0:px1] = person_mask_2d[py0:hy1, px0:px1]
    if not np.any(seed):
        seed = person_mask_2d.copy()
    sys_y, sxs_x = np.nonzero(seed)
    box = [int(sxs_x.min()), int(sys_y.min()), int(sxs_x.max()) + 1, int(sys_y.max()) + 1]
    return seed, box


def own_part_face_points(
    pose_kps: Sequence[np.ndarray | None],
    own_part: np.ndarray,
    *,
    keypoint_index: dict[str, int] | None = None,
    min_score: float = DEFAULT_FACE_POINT_MIN_SCORE,
) -> list[int]:
    """Per frame, how many of the 39 scored face points at or over ``min_score`` lie on ``own_part`` (T, H, W): the pixels of the
    subject's person track whose class is one of the parts the subject replaces. The evidence ``sam31_head_tube`` needs before it
    re-seeds a lost head from the person track: on 4912e her raised hand was class face and hair, and held no face point over the line."""
    own = np.asarray(own_part, dtype=bool)
    num_frames, height, width = own.shape
    if len(pose_kps) != num_frames:
        raise SystemExit(f"own_part_face_points: {len(pose_kps)} frames of points for {num_frames} frames of the part")
    ids = scored_face_ids(keypoint_index)
    out: list[int] = []
    for t in range(num_frames):
        pts = pose_kps[t]
        if pts is None:
            out.append(0)
            continue
        p = np.asarray(pts, dtype=np.float64)
        p = p[[i for i in ids if i < len(p)]]
        p = p[np.isfinite(p).all(axis=1) & (p[:, 2] >= float(min_score))]
        xs, ys = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        out.append(int(own[t, ys[inside], xs[inside]].sum()))
    return out


def _measure_head_rel_y1(head_mask_2d: np.ndarray, person_mask_2d: np.ndarray, default: float = 0.40) -> float:
    """Measure the vertical fraction of ``person_mask_2d`` spanned by ``head_mask_2d`` from the top of the person."""
    pys, _ = np.nonzero(person_mask_2d)
    hys, _ = np.nonzero(head_mask_2d & person_mask_2d)
    if len(pys) == 0 or len(hys) == 0:
        return default
    py0, py1 = int(pys.min()), int(pys.max()) + 1
    hy1 = int(hys.max()) + 1
    span = max(1, py1 - py0)
    return min(max(float(hy1 - py0) / float(span), 0.25), 1.0)


def sam31_head_tube(
    frames_or_tube: np.ndarray | torch.Tensor,
    *,
    seed_frame: int = 0,
    seed_box: Sequence[int] | None = None,
    seed_mask: np.ndarray | None = None,
    person_track: np.ndarray | torch.Tensor | None = None,
    prompt: str = "head",
    pad_frac: float = DEFAULT_BOX_PAD_FRAC,
    tracker: Callable[..., np.ndarray | torch.Tensor] | None = None,
    regain_face_points: Sequence[int] | None = None,
    regain_least: int = REGAIN_FACE_POINTS_MIN,
    hidden_frames: Sequence[int] | set[int] | None = None,
) -> dict[str, Any]:
    """Produce a continuous per-frame boolean head tube ``(T, H, W)`` bounded by ``person_track`` and one fixed ``shot_box``.

    ``hidden_frames`` (frames of the tube, 0 = its first) are the frames a person marked her head ``hidden`` on (the plan's
    ``head_marks``): ``bridge_short_tube_dips`` never bridges onto one, and none bounds a dip.

    Where the tube is lost on a frame and ``person_track`` still holds the subject, a head is re-seeded from the top of the
    person track only when ``regain_face_points[t]`` (``own_part_face_points``: her face points over the line on her own
    replaced classes) is at least ``regain_least``. Otherwise the frame stays empty, since her head is hidden there, and is
    listed in ``regain_refused`` with its count. With no ``regain_face_points``, no frame is re-seeded. On 4912e the top of
    her person track was her raised hand on 4921-4923 and 4941-4945, and a re-seed there made her hand her head."""
    arr = (
        frames_or_tube.detach().cpu().numpy()
        if isinstance(frames_or_tube, torch.Tensor)
        else np.asarray(frames_or_tube)
    )
    if arr.ndim == 3:
        num_frames, height, width = arr.shape
    elif arr.ndim == 4 and arr.shape[-1] == 3:
        num_frames, height, width, _ = arr.shape
    else:
        raise SystemExit(
            f"sam31_head_tube: expected 3D tube (T, H, W) or 4D frames (T, H, W, 3), got shape {arr.shape}"
        )
    if seed_frame < 0 or seed_frame >= num_frames:
        raise SystemExit(
            f"sam31_head_tube: seed_frame {seed_frame} out of range for T={num_frames}"
        )

    pt_bool: np.ndarray | None = None
    if person_track is not None:
        pt_arr = (
            person_track.detach().cpu().numpy()
            if isinstance(person_track, torch.Tensor)
            else np.asarray(person_track)
        )
        if pt_arr.shape != (num_frames, height, width):
            raise SystemExit(
                f"sam31_head_tube: person_track shape {pt_arr.shape} does not match {(num_frames, height, width)}"
            )
        pt_bool = np.asarray(pt_arr > 0, dtype=bool)
        if seed_mask is None and arr.ndim == 4 and seed_box is None and np.any(pt_bool[seed_frame]):
            seed_mask, seed_box = _head_seed_from_person_mask(pt_bool[seed_frame])

    sm_unclipped: np.ndarray | None = None
    if seed_mask is not None:
        sm_raw = (
            seed_mask.detach().cpu().numpy()
            if isinstance(seed_mask, torch.Tensor)
            else np.asarray(seed_mask)
        )
        if sm_raw.shape == (height, width):
            sm_2d = np.asarray(sm_raw > 0, dtype=bool)
            sm_unclipped = np.broadcast_to(sm_2d[None, :, :], (num_frames, height, width)).copy()
            seed_mask = (sm_2d & pt_bool[seed_frame]) if pt_bool is not None else sm_2d
        elif sm_raw.shape == (num_frames, height, width):
            sm_unclipped = np.asarray(sm_raw > 0, dtype=bool).copy()
            seed_mask = (sm_unclipped & pt_bool) if pt_bool is not None else sm_unclipped
        else:
            raise SystemExit(
                f"sam31_head_tube: seed_mask shape {sm_raw.shape} incompatible with {(num_frames, height, width)}"
            )

    norm_t: torch.Tensor | None = None
    if arr.ndim == 3:
        tube = sm_unclipped.copy() if sm_unclipped is not None else np.asarray(arr > 0, dtype=bool)
        normalized_range: tuple[float, float] | None = None
    else:
        norm_t = sam31_normalize_input(torch.from_numpy(arr).permute(0, 3, 1, 2))
        normalized_range = (float(norm_t.min().item()), float(norm_t.max().item()))
        if tracker is not None:
            tracked = tracker(
                norm_t,
                seed_frame=seed_frame,
                seed_box=seed_box,
                seed_mask=seed_mask,
                prompt=prompt,
            )
            tube_arr = (
                tracked.detach().cpu().numpy()
                if isinstance(tracked, torch.Tensor)
                else np.asarray(tracked)
            )
            if tube_arr.shape != (num_frames, height, width):
                raise SystemExit(
                    f"sam31_head_tube: tracker returned shape {tube_arr.shape}, expected {(num_frames, height, width)}"
                )
            tube = np.asarray(tube_arr > 0, dtype=bool)
        elif sm_unclipped is not None:
            tube = sm_unclipped.copy()
        elif seed_box is not None and len(seed_box) == 4:
            bx0, by0, bx1, by1 = [int(v) for v in seed_box]
            if bx0 < 0 or by0 < 0 or bx1 <= bx0 or by1 <= by0 or bx1 > width or by1 > height:
                raise SystemExit(
                    f"sam31_head_tube: invalid seed_box {list(seed_box)} for frame {(height, width)}"
                )
            tube = np.zeros((num_frames, height, width), dtype=bool)
            tube[:, by0:by1, bx0:bx1] = True
        else:
            raise SystemExit(
                "sam31_head_tube: 4D video frames require a tracker, seed_mask, or seed_box"
            )

    regained_from_person_track_frames: list[int] = []
    regain_refused: list[dict[str, Any]] = []
    if regain_face_points is not None and len(regain_face_points) != num_frames:
        raise SystemExit(
            f"sam31_head_tube: regain_face_points has {len(regain_face_points)} frames, expected {num_frames}"
        )
    if pt_bool is not None:
        clipped = tube & ~pt_bool
        outside_person_track_clipped_px = [int(clipped[t].sum()) for t in range(num_frames)]
        tube = tube & pt_bool
        head_rel_y1 = _measure_head_rel_y1(tube[seed_frame], pt_bool[seed_frame], default=0.40)
        had_loss_or_gap = False
        for t in range(seed_frame + 1, num_frames):
            if np.any(tube[t]):
                head_rel_y1 = _measure_head_rel_y1(tube[t], pt_bool[t], default=head_rel_y1)
                had_loss_or_gap = False
                continue
            if not np.any(pt_bool[t]):
                had_loss_or_gap = True
                continue
            # Head tube is lost on frame t while person_track holds/regains the subject:
            # re-seed from person_track[t] so the head tube inherits subject_track's regain search,
            # only where her own face points show (hidden is hidden).
            had_loss_or_gap = True
            seen = None if regain_face_points is None else int(regain_face_points[t])
            if seen is None or seen < int(regain_least):
                regain_refused.append({
                    "frame": int(t),
                    "face_points_on_own_part": seen,
                    "least": int(regain_least),
                    "why": ("no face points were given" if seen is None
                            else f"{seen} of her face points over the line on her own part, under {int(regain_least)}: her head is hidden here"),
                })
                continue
            own_head_t = (np.asarray(arr[t] > 0, dtype=bool) & pt_bool[t]) if arr.ndim == 3 else None
            regain_seed, regain_box = _head_seed_from_person_mask(
                pt_bool[t], head_rel_y1=head_rel_y1, own_head_2d=own_head_t
            )
            if norm_t is not None and tracker is not None:
                re_tracked = tracker(
                    norm_t,
                    seed_frame=t,
                    seed_box=regain_box,
                    seed_mask=regain_seed,
                    prompt=prompt,
                )
                re_arr = (
                    re_tracked.detach().cpu().numpy()
                    if isinstance(re_tracked, torch.Tensor)
                    else np.asarray(re_tracked)
                )
                if re_arr.shape == (num_frames, height, width):
                    re_bounded = np.asarray(re_arr[t:] > 0, dtype=bool) & pt_bool[t:]
                    for dt in range(num_frames - t):
                        if not np.any(tube[t + dt]) and np.any(re_bounded[dt]):
                            tube[t + dt] = re_bounded[dt]
                            regained_from_person_track_frames.append(t + dt)
            if not np.any(tube[t]) and np.any(regain_seed):
                tube[t] = regain_seed
                if t not in regained_from_person_track_frames:
                    regained_from_person_track_frames.append(t)
        tube, bridged_short_dips, short_dips = bridge_short_tube_dips(tube, pt_bool, hidden_frames=hidden_frames)
    else:
        outside_person_track_clipped_px = [0] * num_frames
        bridged_short_dips, short_dips = [], None

    if not np.any(tube[seed_frame]):
        raise SystemExit(
            f"sam31_head_tube: head tube is empty on seed_frame={seed_frame}"
        )
    shot_box = shot_box_from_tube(tube, frame_shape=(height, width), pad_frac=pad_frac)
    tube_px = [int(tube[t].sum()) for t in range(num_frames)]
    return {
        "tube": tube,
        "shot_box": shot_box,
        "seed_frame": int(seed_frame),
        "prompt": str(prompt),
        "tube_px": tube_px,
        "outside_person_track_clipped_px": outside_person_track_clipped_px,
        "regained_from_person_track_frames": sorted(regained_from_person_track_frames),
        "regain_refused": regain_refused,
        "bridged_short_dips": bridged_short_dips,
        "short_dips": short_dips,
        "normalized_range": normalized_range,
    }


def _short_dip_spans(
    counts: np.ndarray,
    track_present: np.ndarray,
    may_bound: np.ndarray,
    max_dip_frames: int,
    dip_ratio: float,
) -> list[tuple[int, int]]:
    """The dips ``(s, e)`` of ``bridge_short_tube_dips``'s scan, left to right, on per-frame tube px. ``may_bound[t]``: frame
    ``t`` may bound a dip. Read on the counts before any bridging: no bound or run frame of a later dip is a frame bridged
    earlier, since the scan goes on after each dip's right bound."""
    num_frames = len(counts)
    spans: list[tuple[int, int]] = []
    s = 1
    while s < num_frames - 1:
        left_px = int(counts[s - 1])
        if not may_bound[s - 1] or not track_present[s - 1] or counts[s] >= dip_ratio * left_px:
            s += 1
            continue
        dip_end: int | None = None
        for e in range(min(s + max_dip_frames - 1, num_frames - 2), s - 1, -1):
            right_px = int(counts[e + 1])
            if (
                may_bound[e + 1]
                and bool(track_present[s - 1 : e + 2].all())
                and int(counts[s : e + 1].max()) < dip_ratio * min(left_px, right_px)
            ):
                dip_end = e
                break
        if dip_end is None:
            s += 1
            continue
        spans.append((s, dip_end))
        s = dip_end + 2
    return spans


def bridge_short_tube_dips(
    tube: np.ndarray,
    track_bool: np.ndarray,
    max_dip_frames: int = SHORT_DIP_MAX_FRAMES,
    dip_ratio: float = SHORT_DIP_RATIO,
    *,
    hidden_frames: Sequence[int] | set[int] | None = None,
    bound_least_share: float = SHORT_DIP_BOUND_LEAST_SHARE,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Bridge every short dip in ``tube`` from the two frames that bound it, where ``track_bool`` holds the subject.

    A dip is a run of frames ``s..e``, 1 to ``max_dip_frames`` long, bounded by frames ``s - 1`` and ``e + 1`` that each
    hold a whole head, with ``track_bool`` present on ``s - 1..e + 1`` and every frame of the run under ``dip_ratio`` times
    the smaller of the two bounding frames' pixels. A whole head is at least ``bound_least_share`` of the shot's own
    median tube px over the frames that have a tube, on a frame not in ``hidden_frames``. Each dip frame takes the union of
    the two bounding frames' tubes, clipped to its own ``track_bool``, except a frame in ``hidden_frames`` (a person marked
    her head hidden there), which is left as it is. The scan goes left to right; from a start ``s`` it takes the longest
    run that is a dip and goes on after that run's right bound, and where no run from ``s`` is a dip it moves on one frame.

    Until 2026-10-23 the right bound was the first frame back at ``dip_ratio`` of the LEFT bound, and a run that did
    not get back there within ``max_dip_frames`` was jumped over whole (or ended the scan at the clip's end), so a
    short dip after a partial recovery was never looked at: on second_hfp_4647b, 4691 inside the long run from 4673,
    and 4697-4698 inside the run from 4695 that never came back (``tube_px`` in that capture's
    ``subjects/second/head_tube__shot_c_4647e.json``). Where the old rule took a run from ``s``, it is the longest
    run from ``s`` that is a dip, so the new rule takes the same one from that start. Until then too, a bound of any size
    counted, so a speck of a few px bounded a dip, and a frame marked hidden was bridged like any other.

    Returns the tube, one record a dip with a frame bridged (``{"span": [s, e], "flank_px": [px of s - 1, px of e + 1],
    "before_px": [...], "after_px": [...], "hidden_left": [frames of the span left as they are]}``), and the rule as it
    was applied: ``{"median_tube_px", "bound_least_share", "bound_least_px", "hidden_frames", "left_on_hidden"}``, where
    ``left_on_hidden`` holds each dip the scan finds when the marks are not read whose frames or bounds hold a marked
    frame: ``{"span", "flank_px", "hidden": [marked frames of s - 1..e + 1], "bridged": [its frames bridged]}``.
    """
    out = np.asarray(tube, dtype=bool).copy()
    tr = np.asarray(track_bool, dtype=bool)
    num_frames = out.shape[0]
    hidden = sorted({int(f) for f in (hidden_frames or ()) if 0 <= int(f) < num_frames})
    rule: dict[str, Any] = {
        "median_tube_px": 0.0,
        "bound_least_share": float(bound_least_share),
        "bound_least_px": 0.0,
        "hidden_frames": hidden,
        "left_on_hidden": [],
    }
    if num_frames < 3 or max_dip_frames < 1:
        return out, [], rule
    counts = out.reshape(num_frames, -1).sum(axis=1).astype(np.int64)
    track_present = tr.reshape(num_frames, -1).any(axis=1)
    held = np.zeros(num_frames, dtype=bool)
    held[hidden] = True
    filled = counts[counts > 0]
    median_px = float(np.median(filled)) if filled.size else 0.0
    least_px = float(bound_least_share) * median_px
    rule["median_tube_px"], rule["bound_least_px"] = round(median_px, 1), round(least_px, 1)
    whole = (counts > 0) & (counts >= least_px)

    bridged_list: list[dict[str, Any]] = []
    bridged_frames: set[int] = set()
    for s, e in _short_dip_spans(counts, track_present, whole & ~held, max_dip_frames, dip_ratio):
        before_px = [int(counts[t]) for t in range(s, e + 1)]
        flank_union = out[s - 1] | out[e + 1]
        left = [t for t in range(s, e + 1) if held[t]]
        for t in range(s, e + 1):
            if not held[t]:
                out[t] = out[t] | (flank_union & tr[t])
                bridged_frames.add(t)
        if len(left) == e - s + 1:
            continue
        bridged_list.append(
            {
                "span": [int(s), int(e)],
                "flank_px": [int(counts[s - 1]), int(counts[e + 1])],
                "before_px": before_px,
                "after_px": [int(out[t].sum()) for t in range(s, e + 1)],
                "hidden_left": left,
            }
        )
    if held.any():
        for s, e in _short_dip_spans(counts, track_present, whole, max_dip_frames, dip_ratio):
            marked = [t for t in range(s - 1, e + 2) if held[t]]
            if marked:
                rule["left_on_hidden"].append(
                    {
                        "span": [int(s), int(e)],
                        "flank_px": [int(counts[s - 1]), int(counts[e + 1])],
                        "hidden": marked,
                        "bridged": [t for t in range(s, e + 1) if t in bridged_frames],
                    }
                )
    return out, bridged_list, rule


def temporal_smooth_probs(
    probs: torch.Tensor,
    weights: Sequence[float] = DEFAULT_TEMPORAL_WEIGHTS,
) -> torch.Tensor:
    """Apply an edge-clamped temporal convolution across frames ``dim=0`` on class probabilities ``(T, C, H, W)``."""
    if probs.ndim != 4 or probs.shape[0] < 1:
        raise SystemExit(
            f"temporal_smooth_probs: expected 4D probability tensor (T, C, H, W), got shape {tuple(probs.shape)}"
        )
    if len(weights) == 0 or len(weights) % 2 == 0:
        raise SystemExit(
            f"temporal_smooth_probs: weights must have a positive odd length, got {list(weights)}"
        )
    w_floats = [float(w) for w in weights]
    if any(w < 0.0 for w in w_floats):
        raise SystemExit(
            f"temporal_smooth_probs: weights must be non-negative, got {w_floats}"
        )
    w_sum = sum(w_floats)
    if w_sum <= 0.0:
        raise SystemExit(
            f"temporal_smooth_probs: weights sum must be > 0, got {w_floats}"
        )
    w_norm = [w / w_sum for w in w_floats]
    num_frames = probs.shape[0]
    radius = len(w_norm) // 2
    out = torch.zeros_like(probs, dtype=torch.float32)
    for t in range(num_frames):
        acc = torch.zeros_like(probs[0], dtype=torch.float32)
        for offset, weight in enumerate(w_norm, start=-radius):
            clamped_t = min(max(t + offset, 0), num_frames - 1)
            acc = acc + weight * probs[clamped_t].float()
        out[t] = acc
    return out


def smooth_seg_in_tube(
    logits: torch.Tensor | np.ndarray,
    tube: np.ndarray | torch.Tensor,
    shot_box: Sequence[int] | None = None,
    *,
    temporal_weights: Sequence[float] = DEFAULT_TEMPORAL_WEIGHTS,
    head_parts: Sequence[str] = DEFAULT_HEAD_PARTS,
    include_eyeglass: bool = False,
    part_prob_floor: float = DEFAULT_PART_PROB_FLOOR,
    class_names: Sequence[str] = CLASS_NAMES,
    logits_is_crop: bool | None = None,
    synthetic_class_logit: float | None = None,
    return_part_probs: bool = True,
) -> dict[str, Any]:
    """Smooth Sapiens2-Seg probabilities temporally and restrict head-part classes to the SAM 3.1 head tube."""
    if len(temporal_weights) == 0 or len(temporal_weights) % 2 == 0:
        raise SystemExit(
            f"temporal_smooth_probs: weights must have a positive odd length, got {list(temporal_weights)}"
        )
    w_floats = [float(w) for w in temporal_weights]
    if any(w < 0.0 for w in w_floats) or sum(w_floats) <= 0.0:
        raise SystemExit(
            f"temporal_smooth_probs: weights must be non-negative with positive sum, got {w_floats}"
        )
    w_sum = sum(w_floats)
    w_norm = [w / w_sum for w in w_floats]
    radius = len(w_norm) // 2

    tube_np = (
        tube.detach().cpu().numpy()
        if isinstance(tube, torch.Tensor)
        else np.asarray(tube)
    )
    if tube_np.ndim != 3:
        raise SystemExit(
            f"smooth_seg_in_tube: expected 3D tube (T, H, W), got shape {tube_np.shape}"
        )
    num_frames, height, width = tube_np.shape
    tube_bool = np.asarray(tube_np > 0, dtype=bool)
    if not np.any(tube_bool):
        raise SystemExit("smooth_seg_in_tube: head tube is empty across all frames")

    is_discrete_cmap = synthetic_class_logit is not None and (
        (isinstance(logits, torch.Tensor) and logits.ndim == 3)
        or (not isinstance(logits, torch.Tensor) and np.asarray(logits).ndim == 3)
    )
    if is_discrete_cmap:
        cmap_in = (
            logits.detach().cpu().numpy().astype(np.uint8)
            if isinstance(logits, torch.Tensor)
            else np.asarray(logits, dtype=np.uint8)
        )
        if cmap_in.shape[0] != num_frames:
            raise SystemExit(
                f"smooth_seg_in_tube: logits frame count {cmap_in.shape[0]} != tube frame count {num_frames}"
            )
        num_classes = len(class_names)
        in_h, in_w = int(cmap_in.shape[1]), int(cmap_in.shape[2])
        logits_t: torch.Tensor | None = None
    else:
        logits_t = (
            logits.detach().float()
            if isinstance(logits, torch.Tensor)
            else torch.from_numpy(np.asarray(logits, dtype=np.float32))
        )
        if logits_t.ndim != 4 or logits_t.shape[0] < 1 or logits_t.shape[1] < 2:
            raise SystemExit(
                f"smooth_seg_in_tube: expected 4D logits (T, C, h, w) with T >= 1 and C >= 2, got shape {tuple(logits_t.shape)}"
            )
        if logits_t.shape[0] != num_frames:
            raise SystemExit(
                f"smooth_seg_in_tube: logits frame count {logits_t.shape[0]} != tube frame count {num_frames}"
            )
        num_classes = int(logits_t.shape[1])
        in_h, in_w = int(logits_t.shape[2]), int(logits_t.shape[3])
        cmap_in = None

    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    chosen_names = list(head_parts)
    if include_eyeglass:
        for opt in OPTIONAL_HEAD_PARTS:
            if opt not in chosen_names:
                chosen_names.append(opt)
    if not chosen_names:
        raise SystemExit("smooth_seg_in_tube: head_parts must name at least one class")
    part_ids: list[int] = []
    for name in chosen_names:
        if name not in name_to_id:
            raise SystemExit(f"smooth_seg_in_tube: unknown segmentation class {name!r}")
        cid = name_to_id[name]
        if cid >= num_classes:
            raise SystemExit(
                f"smooth_seg_in_tube: class {name!r} (id={cid}) exceeds logits channel count {num_classes}"
            )
        part_ids.append(cid)

    if shot_box is None:
        box = shot_box_from_tube(tube_bool, frame_shape=(height, width))
    else:
        if len(shot_box) != 4:
            raise SystemExit(f"smooth_seg_in_tube: shot_box must have 4 ints, got {shot_box}")
        box = [int(v) for v in shot_box]
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0 or x1 > width or y1 > height:
        raise SystemExit(
            f"smooth_seg_in_tube: invalid shot_box {box} for frame shape {(height, width)}"
        )
    if (x0 % 2 != 0) or (y0 % 2 != 0) or ((x1 - x0) % 2 != 0) or ((y1 - y0) % 2 != 0):
        raise SystemExit(
            f"smooth_seg_in_tube: shot_box {box} must be snapped to even integer coordinates"
        )
    box_h, box_w = y1 - y0, x1 - x0

    full_frame_logits = (
        (in_h == height and in_w == width)
        if logits_is_crop is None
        else (not logits_is_crop)
    )
    extra_off_tube_px = [0] * num_frames
    if not is_discrete_cmap and logits_t is not None and full_frame_logits and (box_h != height or box_w != width):
        outside_box_mask = np.ones((height, width), dtype=bool)
        outside_box_mask[y0:y1, x0:x1] = False
        for t in range(num_frames):
            frame_raw_cls = logits_t[t].argmax(dim=0).cpu().numpy()
            frame_raw_is_part = np.isin(frame_raw_cls, part_ids)
            extra_off_tube_px[t] = int(
                np.logical_and(frame_raw_is_part, outside_box_mask & ~tube_bool[t]).sum()
            )

    class_map = np.zeros((num_frames, height, width), dtype=np.uint8)
    part_probs_full = (
        np.zeros((num_frames, height, width), dtype=np.float32)
        if return_part_probs
        else np.zeros((0, 0, 0), dtype=np.float32)
    )
    face_id = name_to_id.get("Face_Neck", -1)
    hair_id = name_to_id.get("Hair", -1)
    part_id_arr = np.asarray(part_ids, dtype=np.uint8)

    tube_px: list[int] = []
    raw_part_px: list[int] = []
    part_px: list[int] = []
    raw_face_neck_px: list[int] = []
    face_neck_px: list[int] = []
    hair_px: list[int] = []
    off_tube_suppressed_px: list[int] = []
    dropout_bridged_frames: list[int] = []

    chunk_size = 16
    first_non_part = next((c for c in range(num_classes) if c not in part_ids), 0)
    for t0 in range(0, num_frames, chunk_size):
        t1 = min(num_frames, t0 + chunk_size)
        h0 = max(0, t0 - radius)
        h1 = min(num_frames, t1 + radius)
        chunk_len = t1 - t0
        tube_box = tube_bool[t0:t1, y0:y1, x0:x1]

        if is_discrete_cmap and cmap_in is not None:
            cmap_halo = (
                cmap_in[h0:h1, y0:y1, x0:x1]
                if (full_frame_logits and (box_h != height or box_w != width))
                else cmap_in[h0:h1]
            )
        else:
            cmap_halo = None

        if is_discrete_cmap and cmap_halo is not None and cmap_halo.shape[1:] == (box_h, box_w):
            exp_l = float(np.exp(float(synthetic_class_logit)))
            denom = exp_l + float(num_classes - 1)
            p_win = exp_l / denom
            p_other = 1.0 / denom
            p_diff = p_win - p_other
            k_parts = len(part_ids)

            raw_cls_box = cmap_halo[t0 - h0 : t1 - h0]
            raw_is_part_box = np.isin(raw_cls_box, part_ids)
            part_prob_box = np.zeros((chunk_len, box_h, box_w), dtype=np.float32)
            best_part_cls_box = np.zeros((chunk_len, box_h, box_w), dtype=np.uint8)
            non_part_cls_box = np.zeros((chunk_len, box_h, box_w), dtype=np.uint8)
            smoothed_argmax_box = np.zeros((chunk_len, box_h, box_w), dtype=np.uint8)

            for idx_in_chunk, t in enumerate(range(t0, t1)):
                c_center = raw_cls_box[idx_in_chunk]
                is_part_c = raw_is_part_box[idx_in_chunk]
                offsets_c = [
                    cmap_halo[min(max(t + offset, 0), num_frames - 1) - h0]
                    for offset in range(-radius, radius + 1)
                ]
                varying = np.zeros((box_h, box_w), dtype=bool)
                for c_off in offsets_c:
                    varying |= c_off != c_center

                p_prob_t = np.where(
                    is_part_c,
                    p_win + float(k_parts - 1) * p_other,
                    float(k_parts) * p_other,
                ).astype(np.float32)
                b_part_t = np.where(is_part_c, c_center, part_id_arr[0]).astype(np.uint8)
                n_part_t = np.where(is_part_c, first_non_part, c_center).astype(np.uint8)
                s_arg_t = c_center.copy()

                if np.any(varying):
                    n_var = int(varying.sum())
                    votes = np.zeros((num_classes, n_var), dtype=np.float32)
                    for c_off, weight in zip(offsets_c, w_norm):
                        c_var = c_off[varying]
                        for cid in np.unique(c_var):
                            votes[int(cid), c_var == cid] += float(weight)
                    probs_var = p_other + p_diff * votes
                    part_probs_var = probs_var[part_ids, :]
                    p_prob_t[varying] = part_probs_var.sum(axis=0)
                    b_part_t[varying] = part_id_arr[part_probs_var.argmax(axis=0)]
                    non_part_var = probs_var.copy()
                    non_part_var[part_ids, :] = -1.0
                    n_part_t[varying] = non_part_var.argmax(axis=0).astype(np.uint8)
                    s_arg_t[varying] = probs_var.argmax(axis=0).astype(np.uint8)

                part_prob_box[idx_in_chunk] = p_prob_t
                best_part_cls_box[idx_in_chunk] = b_part_t
                non_part_cls_box[idx_in_chunk] = n_part_t
                smoothed_argmax_box[idx_in_chunk] = s_arg_t

            smoothed_is_part_box = np.isin(smoothed_argmax_box, part_ids)
        else:
            if is_discrete_cmap and cmap_halo is not None:
                halo_np = np.zeros(
                    (h1 - h0, num_classes, cmap_halo.shape[1], cmap_halo.shape[2]),
                    dtype=np.float32,
                )
                for cid in np.unique(cmap_halo):
                    halo_np[:, int(cid), :, :][cmap_halo == cid] = float(synthetic_class_logit)
                crop_halo = torch.from_numpy(halo_np)
            else:
                assert logits_t is not None
                crop_halo = (
                    logits_t[h0:h1, :, y0:y1, x0:x1]
                    if (full_frame_logits and (box_h != height or box_w != width))
                    else logits_t[h0:h1]
                )

            if crop_halo.shape[2] != box_h or crop_halo.shape[3] != box_w:
                up_halo = F.interpolate(
                    crop_halo,
                    size=(box_h, box_w),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                up_halo = crop_halo

            raw_probs_halo = torch.softmax(up_halo, dim=1)
            smoothed_chunk = torch.zeros(
                (chunk_len, num_classes, box_h, box_w),
                dtype=torch.float32,
                device=raw_probs_halo.device,
            )
            for idx_in_chunk, t in enumerate(range(t0, t1)):
                acc = torch.zeros(
                    (num_classes, box_h, box_w),
                    dtype=torch.float32,
                    device=raw_probs_halo.device,
                )
                for offset, weight in enumerate(w_norm, start=-radius):
                    clamped_t = min(max(t + offset, 0), num_frames - 1)
                    acc = acc + weight * raw_probs_halo[clamped_t - h0]
                smoothed_chunk[idx_in_chunk] = acc

            raw_probs_chunk = raw_probs_halo[t0 - h0 : t1 - h0]
            raw_cls_box = raw_probs_chunk.argmax(dim=1).cpu().numpy()
            raw_is_part_box = np.isin(raw_cls_box, part_ids)

            part_id_tensor = torch.tensor(part_ids, dtype=torch.long, device=smoothed_chunk.device)
            part_probs_sub = smoothed_chunk.index_select(1, part_id_tensor)
            part_prob_box = part_probs_sub.sum(dim=1).cpu().numpy()
            best_part_sub_idx = part_probs_sub.argmax(dim=1).cpu().numpy()
            best_part_cls_box = part_id_arr[best_part_sub_idx]

            non_part_probs = smoothed_chunk.clone()
            non_part_probs[:, part_id_tensor, :, :] = -1.0
            non_part_cls_box = non_part_probs.argmax(dim=1).cpu().numpy().astype(np.uint8)

            smoothed_argmax_box = smoothed_chunk.argmax(dim=1).cpu().numpy().astype(np.uint8)
            smoothed_is_part_box = np.isin(smoothed_argmax_box, part_ids)

        active_in_tube = tube_box & (smoothed_is_part_box | (part_prob_box >= float(part_prob_floor)))
        cls_box = np.where(active_in_tube, best_part_cls_box, non_part_cls_box)
        class_map[t0:t1, y0:y1, x0:x1] = cls_box
        if return_part_probs:
            gated_part_prob_box = np.where(tube_box, part_prob_box, 0.0).astype(np.float32)
            part_probs_full[t0:t1, y0:y1, x0:x1] = gated_part_prob_box

        raw_face_neck_box = (raw_cls_box == face_id) & tube_box if face_id >= 0 else np.zeros_like(tube_box)
        for idx_in_chunk, t in enumerate(range(t0, t1)):
            t_px = int(tube_bool[t].sum())
            r_px = int(np.logical_and(raw_is_part_box[idx_in_chunk], tube_box[idx_in_chunk]).sum())
            s_px = int(np.isin(cls_box[idx_in_chunk], part_ids).sum())
            rf_px = int(raw_face_neck_box[idx_in_chunk].sum())
            f_px = int(((cls_box[idx_in_chunk] == face_id) & tube_box[idx_in_chunk]).sum()) if face_id >= 0 else 0
            h_px = int(((cls_box[idx_in_chunk] == hair_id) & tube_box[idx_in_chunk]).sum()) if hair_id >= 0 else 0
            off_box_px = int(
                np.logical_and(
                    raw_is_part_box[idx_in_chunk] | smoothed_is_part_box[idx_in_chunk],
                    ~tube_box[idx_in_chunk],
                ).sum()
            )
            tube_px.append(t_px)
            raw_part_px.append(r_px)
            part_px.append(s_px)
            raw_face_neck_px.append(rf_px)
            face_neck_px.append(f_px)
            hair_px.append(h_px)
            off_tube_suppressed_px.append(off_box_px + extra_off_tube_px[t])
            if (s_px > r_px and r_px < max(1, int(0.75 * s_px))) or (
                f_px > rf_px and rf_px < max(1, int(0.75 * f_px))
            ):
                dropout_bridged_frames.append(t)

    part_mask = np.isin(class_map, part_ids)
    face_neck_mask = ((class_map == face_id) & tube_bool) if face_id >= 0 else np.zeros_like(part_mask)
    hair_mask = ((class_map == hair_id) & tube_bool) if hair_id >= 0 else np.zeros_like(part_mask)
    norm_weights = [round(float(w) / w_sum, 6) for w in temporal_weights]

    return {
        "class_map": class_map,
        "part_mask": part_mask,
        "part_probs": part_probs_full,
        "face_neck_mask": face_neck_mask,
        "hair_mask": hair_mask,
        "shot_box": box,
        "temporal_weights": norm_weights,
        "part_ids": part_ids,
        "part_names": chosen_names,
        "tube_px": tube_px,
        "raw_part_px": raw_part_px,
        "part_px": part_px,
        "smoothed_part_px": list(part_px),
        "raw_face_neck_px": raw_face_neck_px,
        "face_neck_px": face_neck_px,
        "hair_px": hair_px,
        "off_tube_suppressed_px": off_tube_suppressed_px,
        "dropout_bridged_frames": dropout_bridged_frames,
    }


def scored_face_ids(keypoint_index: dict[str, int] | None = None) -> list[int]:
    """Return the scored Sapiens2-Pose facial keypoint indices (39 points in 308-layout), excluding unscored pupils."""
    idx_map = POSE_KEYPOINT_INDEX if keypoint_index is None else dict(keypoint_index)
    if "tip_of_nose" not in idx_map and "nose" in idx_map:
        idx_map["tip_of_nose"] = idx_map["nose"]
    pupil_ids = {i for name, i in idx_map.items() if "pupil" in name}
    base_ids = fp.face_ids(idx_map)
    return [i for i in base_ids if i not in pupil_ids]


def _main_piece_area_and_width(mask_2d: np.ndarray) -> tuple[int, int]:
    """Return the pixel area and bounding-box width of the largest connected component in ``mask_2d``."""
    u8 = np.asarray(mask_2d > 0, dtype=np.uint8)
    if not np.any(u8):
        return 0, 0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if num_labels <= 1:
        return 0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return int(stats[best, cv2.CC_STAT_AREA]), int(stats[best, cv2.CC_STAT_WIDTH])


def cross_check_face_points(
    pose_keypoints: np.ndarray | Sequence[np.ndarray | None],
    tube: np.ndarray | torch.Tensor,
    face_neck_mask: np.ndarray | torch.Tensor,
    hair_mask: np.ndarray | torch.Tensor,
    *,
    part_mask: np.ndarray | torch.Tensor | None = None,
    raw_face_neck_px: Sequence[int] | None = None,
    dropout_bridged_seed: Sequence[int] | None = None,
    keypoint_index: dict[str, int] | None = None,
    min_score: float = DEFAULT_FACE_POINT_MIN_SCORE,
    turned_away_hair_min: float = TURNED_AWAY_HAIR_SHARE_MIN,
    turned_away_face_max: float = TURNED_AWAY_FACE_SHARE_MAX,
    drop_frames: Sequence[int] | set[int] | None = None,
    no_face_frames: Sequence[int] | set[int] | None = None,
    hold_frames: Sequence[int] | set[int] | None = None,
) -> dict[str, Any]:
    """Cross-check Sapiens2-Pose 39 scored facial points against the head tube and Face_Neck / Hair masks.

    Labels each frame with one of three face states (plus ``off_tube_pose`` when the pose jumps off the head tube):
    - ``facing``: front or three-quarter face (replace, face lines on).
    - ``side_with_eye``: profile / side face with skin and at least one eye or visible profile slice (replace, face lines on).
    - ``back_of_head``: hair only (a proposal that empties nothing by itself, listed for a look in ``proposed_drop_frames``).

    When a person's word is supplied via ``drop_frames`` / ``no_face_frames`` or ``hold_frames``,
    it overrides the label both ways in ``effective_frame_states``, ``face_replace_frames``,
    and ``face_lines_on_frames``.
    """
    tube_bool = (
        tube.detach().cpu().numpy() > 0
        if isinstance(tube, torch.Tensor)
        else np.asarray(tube) > 0
    )
    face_bool = (
        face_neck_mask.detach().cpu().numpy() > 0
        if isinstance(face_neck_mask, torch.Tensor)
        else np.asarray(face_neck_mask) > 0
    )
    hair_bool = (
        hair_mask.detach().cpu().numpy() > 0
        if isinstance(hair_mask, torch.Tensor)
        else np.asarray(hair_mask) > 0
    )
    if tube_bool.ndim != 3 or face_bool.shape != tube_bool.shape or hair_bool.shape != tube_bool.shape:
        raise SystemExit(
            f"cross_check_face_points: tube {tube_bool.shape}, face_neck_mask {face_bool.shape}, and hair_mask {hair_bool.shape} must match (T, H, W)"
        )
    num_frames, height, width = tube_bool.shape
    if len(pose_keypoints) != num_frames:
        raise SystemExit(
            f"cross_check_face_points: pose_keypoints length {len(pose_keypoints)} != tube frame count {num_frames}"
        )

    idx_map = POSE_KEYPOINT_INDEX if keypoint_index is None else dict(keypoint_index)
    if "tip_of_nose" not in idx_map and "nose" in idx_map:
        idx_map["tip_of_nose"] = idx_map["nose"]
    face_id_list = scored_face_ids(idx_map)
    if not face_id_list:
        raise SystemExit("cross_check_face_points: keypoint_index contains no scored facial keypoints")

    left_eye_ids = [
        i
        for n, i in idx_map.items()
        if n.startswith("l_") and n.endswith("_lash_line") and "pupil" not in n
    ]
    right_eye_ids = [
        i
        for n, i in idx_map.items()
        if n.startswith("r_") and n.endswith("_lash_line") and "pupil" not in n
    ]

    if part_mask is not None:
        part_bool = (
            part_mask.detach().cpu().numpy() > 0
            if isinstance(part_mask, torch.Tensor)
            else np.asarray(part_mask) > 0
        )
        face_part_bool = (part_bool & ~hair_bool) | face_bool
    else:
        part_bool = face_bool | hair_bool
        face_part_bool = face_bool
    bridged_set = set(int(t) for t in (dropout_bridged_seed or ()))
    drop_set = set(int(t) for t in (drop_frames or ()))
    no_face_set = set(int(t) for t in (no_face_frames or ()))
    hold_set = set(int(t) for t in (hold_frames or ()))

    main_piece_area: list[int] = []
    main_piece_width: list[int] = []
    for t in range(num_frames):
        area_t, width_t = _main_piece_area_and_width(face_part_bool[t])
        main_piece_area.append(area_t)
        main_piece_width.append(width_t)

    face_points_scored: list[int] = []
    face_points_in_tube: list[int] = []
    face_points_in_part: list[int] = []
    face_points_on_face_neck: list[int] = []
    left_eye_points_in_part: list[int] = []
    right_eye_points_in_part: list[int] = []
    face_point_share: list[float] = []
    hair_tube_share: list[float] = []
    face_neck_tube_share: list[float] = []
    facing_frames: list[int] = []
    side_with_eye_frames: list[int] = []
    back_of_head_frames: list[int] = []
    turned_away_frames: list[int] = []
    dropout_bridged_frames: list[int] = []
    off_tube_pose_frames: list[int] = []
    frame_states: list[str] = []
    effective_frame_states: list[str] = []
    face_replace_frames: list[int] = []
    face_lines_on_frames: list[int] = []
    overridden_by_word_frames: list[int] = []

    for t in range(num_frames):
        t_px = int(tube_bool[t].sum())
        f_px = int(face_bool[t].sum())
        h_px = int(hair_bool[t].sum())
        p_px = int(face_part_bool[t].sum())
        h_share = (float(h_px) / float(t_px)) if t_px > 0 else 0.0
        f_share = (float(f_px) / float(t_px)) if t_px > 0 else 0.0
        p_share = (float(p_px) / float(t_px)) if t_px > 0 else 0.0
        hair_tube_share.append(round(h_share, 6))
        face_neck_tube_share.append(round(f_share, 6))

        pts_raw = pose_keypoints[t]
        if pts_raw is None:
            pts = np.zeros((0, 3), dtype=np.float32)
        else:
            pts = np.asarray(pts_raw, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise SystemExit(
                    f"cross_check_face_points: frame {t} keypoints must have shape (K, 3), got {pts.shape}"
                )

        valid_ids = [i for i in face_id_list if i < len(pts)]
        scored_pts = pts[valid_ids] if valid_ids else np.zeros((0, 3), dtype=np.float32)
        scored_pts = scored_pts[scored_pts[:, 2] >= float(min_score)]

        n_scored = int(len(scored_pts))
        n_in_tube = 0
        n_in_part = 0
        n_on_face = 0
        for x, y, _ in scored_pts:
            ix, iy = int(round(float(x))), int(round(float(y)))
            if 0 <= iy < height and 0 <= ix < width:
                if tube_bool[t, iy, ix]:
                    n_in_tube += 1
                if face_part_bool[t, iy, ix]:
                    n_in_part += 1
                if face_bool[t, iy, ix]:
                    n_on_face += 1

        l_eye_in_part = 0
        for eid in left_eye_ids:
            if eid < len(pts) and float(pts[eid, 2]) >= float(min_score):
                ix, iy = int(round(float(pts[eid, 0]))), int(round(float(pts[eid, 1])))
                if 0 <= iy < height and 0 <= ix < width and face_part_bool[t, iy, ix]:
                    l_eye_in_part += 1

        r_eye_in_part = 0
        for eid in right_eye_ids:
            if eid < len(pts) and float(pts[eid, 2]) >= float(min_score):
                ix, iy = int(round(float(pts[eid, 0]))), int(round(float(pts[eid, 1])))
                if 0 <= iy < height and 0 <= ix < width and face_part_bool[t, iy, ix]:
                    r_eye_in_part += 1

        face_points_scored.append(n_scored)
        face_points_in_tube.append(n_in_tube)
        face_points_in_part.append(n_in_part)
        face_points_on_face_neck.append(n_on_face)
        left_eye_points_in_part.append(l_eye_in_part)
        right_eye_points_in_part.append(r_eye_in_part)
        share = (float(n_in_part) / float(n_scored)) if n_scored > 0 else 0.0
        face_point_share.append(round(share, 6))

        middle_in_tube = (
            fp.face_middle_on_mask(pts[:, :2], idx_map, tube_bool[t])
            if len(pts) > max(valid_ids, default=0)
            else False
        )

        w3 = max(main_piece_width[max(0, t - 1) : min(num_frames, t + 2)])
        a3 = max(main_piece_area[max(0, t - 1) : min(num_frames, t + 2)])
        has_side_piece = main_piece_area[t] >= SIDE_PROFILE_MIN_AREA_PX and (
            main_piece_width[t] >= SIDE_PROFILE_MIN_WIDTH_PX
            or (
                main_piece_width[t] >= SIDE_PROFILE_NEIGHBOR_MIN_WIDTH_PX
                and w3 >= SIDE_PROFILE_MIN_WIDTH_PX
                and a3 >= SIDE_PROFILE_NEIGHBOR_MIN_AREA_PX
            )
        )
        small_tube = t_px < 1000
        is_back_of_head = (
            t_px > 0
            and h_share > float(turned_away_hair_min)
            and n_in_part <= 1
            and n_on_face <= 1
            and not has_side_piece
            and (not small_tube or f_share < float(turned_away_face_max))
            and not (t in bridged_set and n_in_tube >= 2)
        )
        is_off_tube_pose = n_scored > 0 and not middle_in_tube and n_in_tube == 0
        raw_f_px = int(raw_face_neck_px[t]) if raw_face_neck_px is not None else f_px
        is_dropout = (
            not is_back_of_head
            and not is_off_tube_pose
            and n_in_tube > 0
            and middle_in_tube
            and (t in bridged_set or raw_f_px < max(1, int(0.75 * f_px)) or f_share < float(turned_away_face_max))
        )
        if is_dropout:
            dropout_bridged_frames.append(t)

        if is_off_tube_pose:
            off_tube_pose_frames.append(t)

        if is_off_tube_pose and not has_side_piece:
            raw_state = "off_tube_pose"
        elif is_back_of_head:
            back_of_head_frames.append(t)
            turned_away_frames.append(t)
            raw_state = "back_of_head"
        else:
            is_facing = (p_share >= FACING_PART_TUBE_SHARE_MIN and n_in_part >= FACING_MIN_POINTS_IN_PART) or (
                small_tube
                and h_share <= float(turned_away_hair_min)
                and f_share >= float(turned_away_face_max)
                and (
                    (l_eye_in_part > 0 and r_eye_in_part > 0)
                    or (l_eye_in_part == 0 and r_eye_in_part == 0 and n_on_face > 0)
                )
            )
            if is_facing:
                facing_frames.append(t)
                raw_state = "facing"
            else:
                side_with_eye_frames.append(t)
                raw_state = "side_with_eye"

        frame_states.append(raw_state)

        if t in drop_set or t in no_face_set:
            eff_state = "dropped_by_word" if t in drop_set else "no_face_by_word"
            if raw_state in ("facing", "side_with_eye"):
                overridden_by_word_frames.append(t)
        elif t in hold_set:
            eff_state = "held_by_word"
            face_replace_frames.append(t)
            face_lines_on_frames.append(t)
            if raw_state in ("back_of_head", "off_tube_pose"):
                overridden_by_word_frames.append(t)
        else:
            eff_state = raw_state
            if raw_state in ("facing", "side_with_eye"):
                face_replace_frames.append(t)
                face_lines_on_frames.append(t)
        effective_frame_states.append(eff_state)

    proposed_drop_frames = [
        t for t in back_of_head_frames if t not in hold_set and t not in drop_set and t not in no_face_set
    ]

    return {
        "scored_face_keypoint_ids": face_id_list,
        "face_points_scored": face_points_scored,
        "face_points_in_tube": face_points_in_tube,
        "face_points_in_part": face_points_in_part,
        "face_points_on_face_neck": face_points_on_face_neck,
        "left_eye_points_in_part": left_eye_points_in_part,
        "right_eye_points_in_part": right_eye_points_in_part,
        "main_piece_area": main_piece_area,
        "main_piece_width": main_piece_width,
        "face_point_share": face_point_share,
        "hair_tube_share": hair_tube_share,
        "face_neck_tube_share": face_neck_tube_share,
        "facing_frames": facing_frames,
        "side_with_eye_frames": side_with_eye_frames,
        "back_of_head_frames": back_of_head_frames,
        "proposed_drop_frames": proposed_drop_frames,
        "turned_away_frames": turned_away_frames,
        "dropout_bridged_frames": dropout_bridged_frames,
        "off_tube_pose_frames": off_tube_pose_frames,
        "frame_states": frame_states,
        "effective_frame_states": effective_frame_states,
        "face_replace_frames": face_replace_frames,
        "face_lines_on_frames": face_lines_on_frames,
        "overridden_by_word_frames": overridden_by_word_frames,
    }


def build_provenance_record(
    seg_result: dict[str, Any],
    pose_check: dict[str, Any] | None = None,
    *,
    seed_frame: int = 0,
    outside_person_track_clipped_px: Sequence[int] | None = None,
    regained_from_person_track_frames: Sequence[int] | None = None,
    regain_refused: Sequence[dict[str, Any]] | None = None,
    bridged_short_dips: Sequence[dict[str, Any]] | None = None,
    source_facial_hair: dict[str, Any] | None = None,
    short_dips: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON-serializable provenance record for a shot's head-tube part masks. `source_facial_hair`, where the
    capture keeps it (`face_piece_kept_whole`), is written whole; absent, the key is left out."""
    bridged = sorted(
        set(seg_result["dropout_bridged_frames"])
        | set(pose_check["dropout_bridged_frames"] if pose_check is not None else ())
    )
    clipped_px = (
        [int(v) for v in outside_person_track_clipped_px]
        if outside_person_track_clipped_px is not None
        else [0] * len(seg_result["tube_px"])
    )
    regained_frames = (
        [int(v) for v in regained_from_person_track_frames]
        if regained_from_person_track_frames is not None
        else []
    )
    record: dict[str, Any] = {
        "writer_stamp": writer_stamp(__file__),
        "seed_frame": int(seed_frame),
        "shot_box": list(seg_result["shot_box"]),
        "temporal_weights": list(seg_result["temporal_weights"]),
        "part_names": list(seg_result["part_names"]),
        "tube_px": list(seg_result["tube_px"]),
        "raw_part_px": list(seg_result["raw_part_px"]),
        "part_px": list(seg_result["part_px"]),
        "off_tube_suppressed_px": list(seg_result["off_tube_suppressed_px"]),
        "outside_person_track_clipped_px": clipped_px,
        "regained_from_person_track_frames": regained_frames,
        "regain_refused": list(regain_refused) if regain_refused is not None else [],
        "bridged_short_dips": list(bridged_short_dips) if bridged_short_dips is not None else [],
        # the short-dip rule as applied (`bridge_short_tube_dips`): the whole-head floor a bound is held to, the frames marked
        # hidden it was given, and each dip it would bridge but for a mark; None where no person track bounded the tube
        "short_dip_rule": (
            {k: v for k, v in short_dips.items() if k != "left_on_hidden"} if short_dips is not None else None
        ),
        "short_dips_left_on_hidden": list(short_dips["left_on_hidden"]) if short_dips is not None else [],
        "face_points_in_tube": (
            list(pose_check["face_points_in_tube"]) if pose_check is not None else []
        ),
        "face_points_in_part": (
            list(pose_check["face_points_in_part"]) if pose_check is not None else []
        ),
        "face_point_share": (
            list(pose_check["face_point_share"]) if pose_check is not None else []
        ),
        "facing_frames": (
            list(pose_check["facing_frames"]) if pose_check is not None else []
        ),
        "side_with_eye_frames": (
            list(pose_check["side_with_eye_frames"]) if pose_check is not None else []
        ),
        "back_of_head_frames": (
            list(pose_check["back_of_head_frames"]) if pose_check is not None else []
        ),
        "proposed_drop_frames": (
            list(pose_check["proposed_drop_frames"]) if pose_check is not None else []
        ),
        "turned_away_frames": (
            list(pose_check["turned_away_frames"]) if pose_check is not None else []
        ),
        "dropout_bridged_frames": bridged,
        "off_tube_pose_frames": (
            list(pose_check["off_tube_pose_frames"]) if pose_check is not None else []
        ),
        "frame_states": (
            list(pose_check["frame_states"]) if pose_check is not None else []
        ),
        "effective_frame_states": (
            list(pose_check["effective_frame_states"]) if pose_check is not None else []
        ),
        "overridden_by_word_frames": (
            list(pose_check["overridden_by_word_frames"]) if pose_check is not None else []
        ),
    }
    if source_facial_hair is not None:
        record["source_facial_hair"] = source_facial_hair
    return record


def run_head_tube_parts(
    logits: torch.Tensor | np.ndarray,
    frames_or_tube: np.ndarray | torch.Tensor,
    pose_keypoints: np.ndarray | Sequence[np.ndarray | None] | None = None,
    *,
    person_track: np.ndarray | torch.Tensor | None = None,
    seed_frame: int = 0,
    pad_frac: float = DEFAULT_BOX_PAD_FRAC,
    temporal_weights: Sequence[float] = DEFAULT_TEMPORAL_WEIGHTS,
    head_parts: Sequence[str] = DEFAULT_HEAD_PARTS,
    include_eyeglass: bool = False,
    part_prob_floor: float = DEFAULT_PART_PROB_FLOOR,
    keypoint_index: dict[str, int] | None = None,
    min_score: float = DEFAULT_FACE_POINT_MIN_SCORE,
    drop_frames: Sequence[int] | set[int] | None = None,
    no_face_frames: Sequence[int] | set[int] | None = None,
    hold_frames: Sequence[int] | set[int] | None = None,
    regain_face_points: Sequence[int] | None = None,
    provenance_out: Path | None = None,
    masks_out: Path | None = None,
) -> dict[str, Any]:
    """Run SAM 3.1 head tube extraction, Sapiens2-Seg temporal smoothing in tube, and 39-point pose cross-check.

    ``regain_face_points`` is handed to ``sam31_head_tube`` as it is: without it, a head lost from the tube is not re-seeded."""
    tube_info = sam31_head_tube(
        frames_or_tube,
        seed_frame=seed_frame,
        person_track=person_track,
        pad_frac=pad_frac,
        regain_face_points=regain_face_points,
    )
    seg_info = smooth_seg_in_tube(
        logits,
        tube_info["tube"],
        shot_box=tube_info["shot_box"],
        temporal_weights=temporal_weights,
        head_parts=head_parts,
        include_eyeglass=include_eyeglass,
        part_prob_floor=part_prob_floor,
    )
    pose_info: dict[str, Any] | None = None
    if pose_keypoints is not None:
        pose_info = cross_check_face_points(
            pose_keypoints,
            tube_info["tube"],
            seg_info["face_neck_mask"],
            seg_info["hair_mask"],
            part_mask=seg_info["part_mask"],
            raw_face_neck_px=seg_info["raw_face_neck_px"],
            dropout_bridged_seed=seg_info["dropout_bridged_frames"],
            keypoint_index=keypoint_index,
            min_score=min_score,
            drop_frames=drop_frames,
            no_face_frames=no_face_frames,
            hold_frames=hold_frames,
        )
    provenance = build_provenance_record(
        seg_info,
        pose_info,
        seed_frame=seed_frame,
        outside_person_track_clipped_px=tube_info["outside_person_track_clipped_px"],
        regained_from_person_track_frames=tube_info["regained_from_person_track_frames"],
        regain_refused=tube_info["regain_refused"],
        bridged_short_dips=tube_info["bridged_short_dips"],
        short_dips=tube_info["short_dips"],
    )
    if provenance_out is not None:
        provenance_out.parent.mkdir(parents=True, exist_ok=True)
        provenance_out.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if masks_out is not None:
        masks_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            masks_out,
            tube=tube_info["tube"].astype(np.uint8),
            class_map=seg_info["class_map"],
            part_mask=seg_info["part_mask"].astype(np.uint8),
            part_probs=seg_info["part_probs"],
            shot_box=np.asarray(seg_info["shot_box"], dtype=np.int32),
            **npz_writer_fields(__file__),
        )
    return {
        "tube_info": tube_info,
        "seg_info": seg_info,
        "pose_info": pose_info,
        "provenance": provenance,
    }


def _load_pose_slice_for_capture(
    points_path: str | Path,
    first_frame: int,
    num_frames: int,
) -> tuple[list[np.ndarray | None], dict[str, int] | None]:
    path = Path(points_path)
    if not path.is_file():
        raise SystemExit(f"head_tube_for_capture: missing points file: {path}")
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "keypoints" not in loaded.files:
            raise SystemExit(f"head_tube_for_capture: points file {path} lacks 'keypoints'")
        kps = np.asarray(loaded["keypoints"], dtype=np.float32)
        scores = np.asarray(loaded["scores"], dtype=np.float32) if "scores" in loaded.files else None
        at = int(loaded["first_frame"]) if "first_frame" in loaded.files else int(first_frame)
        idx_map: dict[str, int] | None = None
        if "names" in loaded.files:
            names = [str(x) for x in loaded["names"]]
            idx_map = {n: i for i, n in enumerate(names)}
            if "tip_of_nose" not in idx_map and "nose" in idx_map:
                idx_map["tip_of_nose"] = idx_map["nose"]
    else:
        kps = np.asarray(loaded, dtype=np.float32)
        scores = None
        at = int(first_frame)
        idx_map = None

    if kps.ndim != 3:
        raise SystemExit(f"head_tube_for_capture: expected 3D keypoints in {path}, got {kps.shape}")
    if kps.shape[2] == 2:
        if scores is None or scores.shape != kps.shape[:2]:
            sc = np.ones((kps.shape[0], kps.shape[1], 1), dtype=np.float32)
        else:
            sc = scores[:, :, None]
        kps3 = np.concatenate([kps, sc], axis=2)
    elif kps.shape[2] >= 3:
        kps3 = kps[:, :, :3]
    else:
        raise SystemExit(f"head_tube_for_capture: invalid keypoint last dim in {path}: {kps.shape}")

    out: list[np.ndarray | None] = []
    for k in range(num_frames):
        j = int(first_frame) + k - at
        if 0 <= j < len(kps3):
            out.append(kps3[j])
        else:
            out.append(None)
    return out, idx_map


def _expand_track_with_attached_parts(
    track_bool: np.ndarray,
    parts_bool: np.ndarray,
    margin_px: int = SUBJECT_MARGIN_PX,
    touch_px: int = TRACK_TOUCH_DILATE_PX,
) -> np.ndarray:
    """Include connected components of ``parts_bool`` within ``margin_px`` of ``track_bool`` that touch ``track_bool``."""
    out = track_bool.copy()
    k_touch = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * touch_px + 1, 2 * touch_px + 1))
    k_margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
    for t in range(track_bool.shape[0]):
        outside = parts_bool[t] & ~track_bool[t]
        if not np.any(outside) or not np.any(track_bool[t]):
            continue
        tr_u8 = track_bool[t].astype(np.uint8)
        near_margin = cv2.dilate(tr_u8, k_margin) > 0
        cand = parts_bool[t] & near_margin
        if not np.any(cand & ~track_bool[t]):
            continue
        near_touch = cv2.dilate(tr_u8, k_touch) > 0
        num_labels, labels = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
        for lbl in range(1, num_labels):
            comp = labels == lbl
            if np.any(comp & near_touch):
                out[t] |= comp
    return out


def _fill_unclassified_in_head_tube(
    cmap_box: np.ndarray,
    tube_box: np.ndarray,
    *,
    part_ids: Sequence[int],
    hair_id: int,
    face_id: int,
    pose_kps: list[np.ndarray | None] | None = None,
    x0: int = 0,
    y0: int = 0,
) -> None:
    """Resolve pixels inside the SAM 3.1 head tube that Sapiens labeled as neither face (`part_ids`) nor `Hair`.

    Where Sapiens labeled a head-tube pixel as `part_ids` or `Hair`, that face-against-hair decision stands.
    Where Sapiens mislabeled head-tube pixels as body/clothing/background (e.g. `Upper_Clothing` or `Background`
    during a fast turn), transport the face-vs-hair layout from the nearest frames in the shot where Sapiens
    classified the head tube, reinforced by the convex hull of confident face keypoints on the frame.

    Only the face is written. A pixel the layout does not take as face keeps the part model's own class: this runs only
    for a part that does not replace hair (a face-only subject; `head_tube_for_capture` takes every non-face tube pixel
    of a hair-and-face part as `Hair` instead), and such a subject's tube pixels are not relabelled `Hair`.
    """
    import cv2

    num_frames, box_h, box_w = cmap_box.shape
    face_only_ids = [pid for pid in part_ids if pid != hair_id]
    is_face = tube_box & np.isin(cmap_box, face_only_ids)
    is_hair = (tube_box & (cmap_box == hair_id)) if hair_id >= 0 else np.zeros_like(tube_box)

    good_refs: list[int] = []
    for t in range(num_frames):
        t_px = int(tube_box[t].sum())
        if t_px <= 0:
            continue
        f_px = int(is_face[t].sum())
        h_px = int(is_hair[t].sum())
        if f_px > 0 and (f_px + h_px) / float(t_px) >= 0.65:
            good_refs.append(t)

    face_kp_indices = list(range(0, 5)) + list(range(238, 308))

    for t in range(num_frames):
        tb = tube_box[t]
        if not np.any(tb):
            continue
        uncls = tb & ~is_face[t] & ~is_hair[t]
        if not np.any(uncls):
            continue

        ys, xs = np.where(tb)
        ty0, ty1 = int(ys.min()), int(ys.max()) + 1
        tx0, tx1 = int(xs.min()), int(xs.max()) + 1
        th, tw = ty1 - ty0, tx1 - tx0

        face_vote = np.zeros((th, tw), dtype=np.float32)
        hair_vote = np.zeros((th, tw), dtype=np.float32)
        if good_refs and th > 0 and tw > 0:
            prev_refs = [r for r in good_refs if r < t]
            next_refs = [r for r in good_refs if r > t]
            chosen_refs: list[int] = []
            if prev_refs:
                chosen_refs.append(prev_refs[-1])
            if next_refs:
                chosen_refs.append(next_refs[0])
            if not chosen_refs and t in good_refs:
                chosen_refs.append(t)
            for r in chosen_refs:
                rys, rxs = np.where(tube_box[r])
                ry0, ry1 = int(rys.min()), int(rys.max()) + 1
                rx0, rx1 = int(rxs.min()), int(rxs.max()) + 1
                if ry1 <= ry0 or rx1 <= rx0:
                    continue
                w = 1.0 / float(max(1, abs(t - r)))
                f_crop = is_face[r, ry0:ry1, rx0:rx1].astype(np.float32)
                h_crop = is_hair[r, ry0:ry1, rx0:rx1].astype(np.float32)
                face_vote += w * cv2.resize(f_crop, (tw, th), interpolation=cv2.INTER_LINEAR)
                hair_vote += w * cv2.resize(h_crop, (tw, th), interpolation=cv2.INTER_LINEAR)

        kp_hull = np.zeros((box_h, box_w), dtype=bool)
        if pose_kps is not None and t < len(pose_kps) and pose_kps[t] is not None:
            kps = np.asarray(pose_kps[t], dtype=np.float64)
            valid_idx = [i for i in face_kp_indices if i < kps.shape[0]]
            if valid_idx:
                fk = kps[valid_idx]
                vis = fk[fk[:, 2] >= DEFAULT_FACE_POINT_MIN_SCORE]
                if len(vis) >= 3:
                    pts = np.round(vis[:, :2] - np.array([[x0, y0]], dtype=np.float64)).astype(np.int32)
                    hull = cv2.convexHull(pts)
                    mask_u8 = np.zeros((box_h, box_w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask_u8, hull, 1)
                    rad = max(3, int(round(0.06 * np.sqrt(float(tb.sum())))))
                    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
                    kp_hull = cv2.dilate(mask_u8, kern) > 0

        uncls_crop = uncls[ty0:ty1, tx0:tx1]
        kp_crop = kp_hull[ty0:ty1, tx0:tx1]
        if np.any(face_vote > 0) or np.any(hair_vote > 0):
            choose_face = uncls_crop & ((face_vote > hair_vote) | (kp_crop & (hair_vote < 0.65)))
        else:
            choose_face = uncls_crop & (kp_crop if np.any(kp_crop) else True)

        if face_id >= 0 and np.any(choose_face):
            cmap_box[t, ty0:ty1, tx0:tx1][choose_face] = np.uint8(face_id)


def face_piece_kept_whole(
    part_mask: np.ndarray,
    tube: np.ndarray,
    part: np.ndarray,
    person: np.ndarray,
    *,
    cmap: np.ndarray | None = None,
    others: np.ndarray | None = None,
    skip_frames: dict[int, str] | None = None,
    pad: int = FACE_COLUMN_PAD_PX,
    jaw_pad_frac: float = JAW_BEARD_PAD_FRAC,
) -> tuple[np.ndarray, dict[str, Any]]:
    """THE SOURCE'S FACIAL HAIR (2026-10-23; extended 2026-10-24 to replace 0764f585's Face_Neck-only assumption): what the head tube
    gives back to a face-only part whose plan says the source shows facial hair to replace (`subjects.<s>.source_facial_hair`). The head
    track is widened only by a piece of the part that fits within its rows (`head_tube_for_capture`, the attach rule), and the tube is the
    head track, so a face-only part is cut where SAM's head mask stops, at the jaw, or where Sapiens labels the beard `Hair` / `Upper_Clothing`
    / `Torso` (`5675f`, `5011e`, `5461d`, `4479d`).

    On each frame:
    1. The connected `part` piece (`part`, after off_head) inside the person track (`person`) and outside the tube (`tube`) inside the
       columns of the tube's part (`part_mask`) widened by `pad` on each side.
    2. When `jaw_pad_frac > 0`, the downward jaw/beard extension extending the bottom row of `part_mask[t]` in each column `x` of
       `part_mask[t]` downward by `max(int(round(jaw_pad_frac * max(fh_h, med_fh))), 1)` rows inside `person[t] & ~part_mask[t]`,
       excluding Sapiens hand/arm classes (`JAW_BEARD_EXCLUDE_CLASSES`) when `cmap` is given.
    Both are intersected with `~others[t]` and 8-connected to `part_mask[t]`. Nothing is kept on `skip_frames` (a person's drop or
    no-face word, with why). Returns the kept pixels (T, H, W) and the provenance record."""
    num_frames = part_mask.shape[0]
    kept = np.zeros(part_mask.shape, dtype=bool)
    rows: list[dict[str, Any]] = []
    excl_ids = [NAME_TO_ID[c] for c in JAW_BEARD_EXCLUDE_CLASSES if c in NAME_TO_ID]
    fhs: list[int] = []
    for t in range(num_frames):
        if skip_frames and t in skip_frames:
            continue
        if part_mask[t].any():
            ys_t = np.nonzero(part_mask[t])[0]
            fhs.append(int(ys_t.max() - ys_t.min() + 1))
    med_fh = float(np.median(fhs)) if fhs else 0.0
    for t in range(num_frames):
        row: dict[str, Any] = {"frame": t, "kept_px": 0}
        rows.append(row)
        if skip_frames and t in skip_frames:
            row["nothing_kept_because"] = skip_frames[t]
            continue
        face = part_mask[t]
        if not face.any():
            row["nothing_kept_because"] = "the head tube's part is empty on the frame"
            continue
        mine = part[t] & person[t]
        _, labels = cv2.connectedComponents((mine | face).astype(np.uint8), connectivity=8)
        own = np.unique(labels[face])
        piece = np.isin(labels, own[own > 0]) & mine & ~tube[t]
        ys_f, xs_f = np.nonzero(face)
        xs = np.unique(xs_f)
        c0, c1 = max(int(xs[0]) - int(pad), 0), min(int(xs[-1]) + int(pad), face.shape[1] - 1)
        columns = np.zeros(face.shape, dtype=bool)
        columns[:, c0:c1 + 1] = True
        jaw_ext = np.zeros(face.shape, dtype=bool)
        excl_px = 0
        if jaw_pad_frac > 0.0:
            fh_h = int(ys_f.max() - ys_f.min() + 1)
            pad_y = max(int(round(float(jaw_pad_frac) * max(float(fh_h), med_fh))), 1)
            for x in xs:
                by_y = int(np.nonzero(face[:, x])[0].max())
                jaw_ext[by_y + 1 : min(face.shape[0], by_y + 1 + pad_y), x] = True
            jaw_ext &= person[t] & ~face
            if cmap is not None and excl_ids:
                hand_arm = np.isin(cmap[t], excl_ids)
                excl_px = int((jaw_ext & hand_arm).sum())
                jaw_ext &= ~hand_arm
        cand = (piece & columns) | jaw_ext
        if not cand.any() and not piece.any():
            row["nothing_kept_because"] = "no piece of the part or jaw extension joins the face"
            if excl_px:
                row["excluded_hand_arm_px"] = excl_px
            continue
        row.update(
            face_columns=[c0, c1],
            cut_by_the_columns_px=int((piece & ~columns & ~jaw_ext).sum()),
            jaw_extended_px=int(jaw_ext.sum()),
            excluded_hand_arm_px=excl_px,
        )
        if others is not None:
            row["in_another_persons_mask_px"] = int((cand & others[t]).sum())
            cand &= ~others[t]
        _, joined = cv2.connectedComponents((cand | face).astype(np.uint8), connectivity=8)
        own = np.unique(joined[face])
        keep = np.isin(joined, own[own > 0]) & cand
        kept[t] = keep
        row.update(kept_px=int(keep.sum()), cut_off_from_the_face_px=int(cand.sum()) - int(keep.sum()))
    record = {
        "rule": "face_piece_kept_whole",
        "is": ("the part as the job hands it in the pieces that join the tube's part inside the face columns widened by "
               "FACE_COLUMN_PAD_PX, unioned with the downward jaw extension of JAW_BEARD_PAD_FRAC of the face height in each face "
               "column inside the person track (less hand/arm classes JAW_BEARD_EXCLUDE_CLASSES and every other person's mask)"),
        "said_by": "the plan's subjects.<s>.source_facial_hair (`bench/shot_to_load.py::source_facial_hair_of`), facial_hair=keep on --mask",
        "column_pad_px": int(pad),
        "jaw_pad_frac": float(jaw_pad_frac),
        "median_face_height_px": round(med_fh, 1),
        "others_given": others is not None,
        "kept_px": [int(r["kept_px"]) for r in rows],
        "kept_on_frames": [int(r["frame"]) for r in rows if r["kept_px"]],
        "cut_by_the_columns_px": [int(r.get("cut_by_the_columns_px", 0)) for r in rows],
        "excluded_hand_arm_px": [int(r.get("excluded_hand_arm_px", 0)) for r in rows],
        "frames": rows,
        "cannot_see": list(FACIAL_HAIR_CANNOT_SEE),
    }
    return kept, record


def head_tube_for_capture(
    track: np.ndarray,
    parts: np.ndarray,
    cmap: np.ndarray | None = None,
    *,
    head_track: np.ndarray | None = None,
    points_path: str | Path | None = None,
    first_frame: int = 0,
    head_parts: Sequence[str] = DEFAULT_HEAD_PARTS,
    include_eyeglass: bool = False,
    temporal_weights: Sequence[float] = DEFAULT_TEMPORAL_WEIGHTS,
    part_prob_floor: float = DEFAULT_PART_PROB_FLOOR,
    drop_frames: Sequence[int] | set[int] | None = None,
    no_face_frames: Sequence[int] | set[int] | None = None,
    hold_frames: Sequence[int] | set[int] | None = None,
    provenance_out: Path | None = None,
    masks_out: Path | None = None,
    keep_facial_hair: bool = False,
    others: np.ndarray | None = None,
    hidden_frames: Sequence[int] | set[int] | None = None,
) -> dict[str, Any]:
    """Run person-track-bounded head-tube smoothing on a capture's track, parts, class map, and 308-point file.

    `hidden_frames` (the capture's frames, 0 = `first_frame`, a person marked her head `hidden` on: the plan's `head_marks`
    as `bench/shot_to_load.py::head_marks_of` cuts them to the load): no short dip is bridged onto one or bounded by one
    (`bridge_short_tube_dips`), and the provenance's `short_dips_left_on_hidden` lists each dip a mark held back; the 1-2 frame
    fill of the candidate tube neither fills one nor fills a neighbour from one. The re-seed of a lost head
    (`sam31_head_tube`, on her own face points) does not read them.

    `keep_facial_hair` (a face-only part whose plan says the source shows facial hair to replace): the piece of the part joined to
    the face that the tube cut below the jaw is kept, within the face's columns and less `others` (every other person's mask,
    (T, H, W)) (`face_piece_kept_whole`), each px with its own class, and recorded in the provenance's `source_facial_hair`; the
    provenance's `part_px` and `off_tube_suppressed_px` then count the part with it and the suppressed px less it, and the tube's own
    figures are kept beside them. Off, nothing here changes."""
    raw_track_bool = np.asarray(track > 0, dtype=bool)
    parts_bool = np.asarray(parts > 0, dtype=bool)
    if raw_track_bool.ndim != 3 or parts_bool.shape != raw_track_bool.shape:
        raise SystemExit(
            f"head_tube_for_capture: track shape {raw_track_bool.shape} and parts shape {parts_bool.shape} must match (T, H, W)"
        )
    track_bool = _expand_track_with_attached_parts(raw_track_bool, parts_bool)
    num_frames, height, width = track_bool.shape
    head_track_bool: np.ndarray | None = None
    if head_track is not None:
        ht_arr = (
            head_track.detach().cpu().numpy()
            if isinstance(head_track, torch.Tensor)
            else np.asarray(head_track)
        )
        if ht_arr.shape != (num_frames, height, width):
            raise SystemExit(
                f"head_tube_for_capture: head_track shape {ht_arr.shape} does not match {(num_frames, height, width)}"
            )
        head_track_bool = np.asarray(ht_arr > 0, dtype=bool).copy()
        if not np.any(head_track_bool & track_bool):
            raise SystemExit(
                "head_tube_for_capture: head_track has no pixels inside the subject's person track across the span"
            )
        # H3SubjectTrack clips head_track to raw_track_bool; when _expand_track_with_attached_parts expands
        # track_bool by up to 8 px to include attached part components (e.g. side-profile nose/cheek on 0606c
        # frames 621-622 and 709-737), expand head_track_bool into those attached components that share pixels
        # with (head_track_bool & raw_track_bool) and lie within its vertical span [hy0, hy1].
        k_touch_ht = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for t in range(num_frames):
            ht_in_raw = head_track_bool[t] & raw_track_bool[t]
            if not np.any(ht_in_raw) or not np.any(parts_bool[t]):
                continue
            ys_ht = np.where(ht_in_raw)[0]
            hy0, hy1 = int(ys_ht.min()), int(ys_ht.max())
            near_ht = cv2.dilate(ht_in_raw.astype(np.uint8), k_touch_ht) > 0
            num_l, labels, stats, _ = cv2.connectedComponentsWithStats(
                (parts_bool[t] & track_bool[t]).astype(np.uint8), connectivity=8
            )
            for lbl in range(1, num_l):
                comp = labels == lbl
                cy0 = int(stats[lbl, cv2.CC_STAT_TOP])
                cy1 = cy0 + int(stats[lbl, cv2.CC_STAT_HEIGHT]) - 1
                if cy0 >= hy0 and cy1 <= hy1 and np.any(comp & near_ht) and np.any(comp & ht_in_raw):
                    head_track_bool[t] |= comp
    chosen_names = [n for n in head_parts if n in NAME_TO_ID]
    if include_eyeglass:
        for opt in OPTIONAL_HEAD_PARTS:
            if opt in NAME_TO_ID and opt not in chosen_names:
                chosen_names.append(opt)
    if not chosen_names:
        chosen_names = list(DEFAULT_HEAD_PARTS)
    part_ids = [NAME_TO_ID[n] for n in chosen_names]
    if keep_facial_hair:
        # THE SOURCE'S FACIAL HAIR is built and measured on a face-only part (2026-10-23): a part made of Hair takes the whole tube
        if "Hair" in chosen_names or "Face_Neck" not in chosen_names:
            raise SystemExit(f"head_tube_for_capture: keep_facial_hair is for a face-only part, and this part is made of {', '.join(chosen_names)}: "
                             "refused before the head tube writes anything")
        if others is not None and np.asarray(others).shape != raw_track_bool.shape:
            raise SystemExit(f"head_tube_for_capture: others shape {np.asarray(others).shape} does not match {raw_track_bool.shape}")

    # The whole-head tube always includes all whole-head classes (Face_Neck + Hair + lips/teeth/tongue),
    # even on a face-only (fp) load where chosen_names excludes Hair, so the tube stays ~35,000 px on
    # back-of-head frames (687-708, 747-750) while smooth_seg_in_tube extracts only chosen_names inside it.
    tube_names = list(DEFAULT_HEAD_PARTS)
    for n in chosen_names:
        if n not in tube_names:
            tube_names.append(n)
    tube_ids = [NAME_TO_ID[n] for n in tube_names if n in NAME_TO_ID]

    if cmap is not None:
        cmap_arr = np.asarray(cmap, dtype=np.uint8)
        if cmap_arr.shape != (num_frames, height, width):
            raise SystemExit(
                f"head_tube_for_capture: cmap shape {cmap_arr.shape} does not match {(num_frames, height, width)}"
            )
        # Drop any head-part pixels that were already pruned from parts (e.g. by off_head)
        # outside the SAM 3.1 head track so the class map never resurrects detached off-head islands (biceps, hands, skirts)
        # into the head tube or part mask, while keeping Sapiens' face/hair labels inside head_track_bool intact.
        outside_ht = ~head_track_bool if head_track_bool is not None else True
        cmap_arr = np.where(np.isin(cmap_arr, part_ids) & ~parts_bool & outside_ht, 0, cmap_arr)
        # Her own replaced classes on her person track (or held part), as the part model labelled them: what a re-seed's face points must land on.
        own_part = (track_bool | parts_bool) & np.isin(cmap_arr, part_ids)
        raw_head = parts_bool | (np.isin(cmap_arr, tube_ids) & track_bool)
    else:
        cmap_arr = np.where(parts_bool, NAME_TO_ID["Face_Neck"], 0).astype(np.uint8)
        raw_head = parts_bool.copy()
        own_part = track_bool & parts_bool

    # Bridge 1-2 frame segmentation dropouts inside the person track when forming the continuous head tube,
    # while leaving off-person pixels unclipped in candidate_tube so sam31_head_tube records outside_person_track_clipped_px.
    # A frame marked hidden (`hidden_frames`) is neither filled here nor lends its pixels to a neighbour's fill.
    candidate_tube = raw_head.copy()
    in_person = raw_head & track_bool
    hidden_set = {int(f) for f in (hidden_frames or ()) if 0 <= int(f) < num_frames}
    if hidden_set:
        in_person = in_person.copy()
        in_person[sorted(hidden_set)] = False
    for t in range(num_frames):
        if t in hidden_set:
            continue
        p1 = in_person[max(0, t - 1)]
        n1 = in_person[min(num_frames - 1, t + 1)]
        p2 = in_person[max(0, t - 2)]
        n2 = in_person[min(num_frames - 1, t + 2)]
        bridged_t = (p1 | p2) & (n1 | n2) & track_bool[t]
        candidate_tube[t] = candidate_tube[t] | bridged_t

    if head_track_bool is not None:
        ht_per_frame = (head_track_bool & track_bool).reshape(num_frames, -1).sum(axis=1)
        seed_frame = 0 if int(ht_per_frame[0]) > 0 else int(np.argmax(ht_per_frame > 0))
    else:
        bounded_per_frame = (candidate_tube & track_bool).reshape(num_frames, -1).sum(axis=1)
        if not np.any(bounded_per_frame > 0):
            raise SystemExit(
                "head_tube_for_capture: no head-part pixels lie inside the subject's person track across the span"
            )
        seed_frame = 0 if int(bounded_per_frame[0]) > 0 else int(np.argmax(bounded_per_frame > 0))

    pose_kps: list[np.ndarray | None] | None = None
    kp_idx: dict[str, int] | None = None
    if points_path is not None:
        pose_kps, kp_idx = _load_pose_slice_for_capture(points_path, first_frame, num_frames)

    tube_info = sam31_head_tube(
        candidate_tube,
        seed_frame=seed_frame,
        seed_mask=head_track_bool,
        person_track=track_bool,
        regain_face_points=(
            own_part_face_points(pose_kps, own_part, keypoint_index=kp_idx) if pose_kps is not None else None
        ),
        hidden_frames=hidden_frames,
    )
    hair_id = NAME_TO_ID.get("Hair", -1)
    face_id = NAME_TO_ID.get("Face_Neck", part_ids[0])
    head_and_hair_on_head_track = head_track_bool is not None and hair_id >= 0 and hair_id in part_ids
    if head_and_hair_on_head_track and head_track_bool is not None:
        # When SAM 3.1's head_track covers the lower face (`>= _SPECK_MIN_PX = 64 px`, excluding 1-px compression specks
        # like 4912n frame 4926 at (691, 505)) but clips the upper hair crown at `hy0` (4912n frames 4927-4932, where
        # `parts_bool & raw_track_bool` has 2,607-2,813 px connected above `hy0`), expand `head_track_bool` upward into
        # that connected component inside `raw_track_bool` while keeping the lower bound `y <= hy1` (so neck/collarbone
        # pixels below `hy1` are not added) and leaving `tube_info["tube"]` untouched for `her_hair_in_hole`.
        k_crown_ht = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for t in range(num_frames):
            if t in hidden_set:
                continue
            ht_in_raw = head_track_bool[t] & raw_track_bool[t]
            if int(np.count_nonzero(ht_in_raw)) < _SPECK_MIN_PX or not np.any(parts_bool[t]):
                continue
            ys_ht = np.where(ht_in_raw)[0]
            hy0, hy1 = int(ys_ht.min()), int(ys_ht.max())
            if hy0 <= 0:
                continue
            near_ht = cv2.dilate(ht_in_raw.astype(np.uint8), k_crown_ht) > 0
            num_l, labels, _, _ = cv2.connectedComponentsWithStats(
                (parts_bool[t] & track_bool[t]).astype(np.uint8), connectivity=8
            )
            for lbl in range(1, num_l):
                comp = labels == lbl
                if np.any(comp & near_ht) and np.any(comp & ht_in_raw):
                    comp_raw = comp & raw_track_bool[t]
                    if int(np.count_nonzero(comp_raw[:hy0, :])) >= _MIN_CROWN_ABOVE_HY0_PX:
                        head_track_bool[t, : hy1 + 1, :] |= comp_raw[: hy1 + 1, :]
    ht_in_track = (head_track_bool & track_bool) if head_track_bool is not None else None
    if head_and_hair_on_head_track and ht_in_track is not None:
        tube_updated = False
        for t in range(num_frames):
            if t in hidden_set or np.any(ht_in_track[t]) or not np.any(tube_info["tube"][t]):
                continue
            # On frames where the SAM 3.1 head track was empty and sam31_head_tube regained or bridged
            # the head from person_track, take her actual Sapiens head/hair part mask (parts_bool[t])
            # rather than a rectangular slice of person_track that includes Upper_Clothing/Torso/Hands.
            if np.any(parts_bool[t]):
                tube_info["tube"][t] = parts_bool[t].copy()
                tube_updated = True
        if tube_updated:
            tube_info["shot_box"] = shot_box_from_tube(tube_info["tube"], frame_shape=(height, width))
            tube_info["tube_px"] = [int(tube_info["tube"][t].sum()) for t in range(num_frames)]
    x0, y0, x1, y1 = tube_info["shot_box"]
    cmap_box = np.clip(cmap_arr[:, y0:y1, x0:x1], 0, len(CLASS_NAMES) - 1).copy()
    face_keep_ids = [pid for pid in part_ids if pid != hair_id] or [
        NAME_TO_ID[n] for n in DEFAULT_HEAD_PARTS if n != "Hair" and n in NAME_TO_ID
    ]
    tube_box_crop = tube_info["tube"][:, y0:y1, x0:x1]
    parts_box_crop = parts_bool[:, y0:y1, x0:x1]
    ht_box_crop = ht_in_track[:, y0:y1, x0:x1] if ht_in_track is not None else tube_box_crop
    relabel_hair_crop = ht_box_crop | (tube_box_crop & parts_box_crop)
    raw_part_px_in_tube: list[int] | None = None
    raw_face_neck_px_in_tube: list[int] | None = None
    if head_and_hair_on_head_track:
        raw_part_px_in_tube = [
            int(np.logical_and(np.isin(cmap_box[t], part_ids), tube_box_crop[t]).sum())
            for t in range(num_frames)
        ]
        # On a head-and-hair load with a SAM 3.1 head track, the head-and-hair region IS the head track
        # (`head_track & track_bool`) plus any regained Sapiens head/hair part (`parts_bool`); inside that region
        # Sapiens marks what is face (`face_keep_ids`) and non-face pixels are `Hair` (`hair_id`), without ever
        # relabelling `Upper_Clothing`/`Torso`/`Hands` from a `person_track` rectangle outside `head_track` and `parts_bool`.
        non_face_in_tube = relabel_hair_crop & ~np.isin(cmap_box, face_keep_ids)
        cmap_box[non_face_in_tube] = np.uint8(hair_id)
    else:
        unlabeled_part_box = parts_bool[:, y0:y1, x0:x1] & ~np.isin(cmap_box, part_ids)
        cmap_box[unlabeled_part_box] = face_id
        if head_track_bool is not None or tube_info["bridged_short_dips"]:
            raw_part_px_in_tube = [
                int(np.logical_and(np.isin(cmap_box[t], part_ids), tube_box_crop[t]).sum())
                for t in range(num_frames)
            ]
            raw_face_neck_px_in_tube = [
                int(np.logical_and(cmap_box[t] == face_id, tube_box_crop[t]).sum()) if face_id >= 0 else 0
                for t in range(num_frames)
            ]
        if head_track_bool is None and tube_info["bridged_short_dips"]:
            for dip in tube_info["bridged_short_dips"]:
                s, e = dip["span"]
                l_cls = cmap_box[s - 1]
                r_cls = cmap_box[e + 1]
                l_in = np.isin(l_cls, part_ids)
                r_in = np.isin(r_cls, part_ids)
                donor = np.where(l_in, l_cls, np.where(r_in, r_cls, np.uint8(face_id)))
                for t in range(s, e + 1):
                    if t in dip["hidden_left"]:      # a frame marked hidden is not bridged, so its classes are not lent either
                        continue
                    uncls_t = tube_box_crop[t] & ~np.isin(cmap_box[t], part_ids)
                    cmap_box[t] = np.where(uncls_t, donor, cmap_box[t])

    if head_track_bool is not None and not head_and_hair_on_head_track:
        _fill_unclassified_in_head_tube(
            cmap_box,
            tube_box_crop,
            part_ids=part_ids,
            hair_id=hair_id,
            face_id=face_id,
            pose_kps=pose_kps,
            x0=x0,
            y0=y0,
        )

    seg_info = smooth_seg_in_tube(
        cmap_box,
        tube_info["tube"],
        shot_box=tube_info["shot_box"],
        temporal_weights=temporal_weights,
        head_parts=chosen_names,
        include_eyeglass=include_eyeglass,
        part_prob_floor=part_prob_floor,
        logits_is_crop=True,
        synthetic_class_logit=SYNTHETIC_CLASS_LOGIT,
        return_part_probs=(masks_out is not None),
    )
    if head_and_hair_on_head_track and raw_part_px_in_tube is not None:
        tube_full = tube_info["tube"]
        relabel_hair_full = (ht_in_track | (tube_full & parts_bool)) if ht_in_track is not None else tube_full
        cls_full = seg_info["class_map"]
        non_face_full = relabel_hair_full & ~np.isin(cls_full, face_keep_ids)
        cls_full[non_face_full] = np.uint8(hair_id)
        seg_info["part_mask"] = relabel_hair_full.copy()
        seg_info["face_neck_mask"] = ((cls_full == face_id) & relabel_hair_full) if face_id >= 0 else np.zeros_like(tube_full)
        seg_info["hair_mask"] = (cls_full == hair_id) & relabel_hair_full
        seg_info["raw_part_px"] = raw_part_px_in_tube
        seg_info["part_px"] = [int(seg_info["part_mask"][t].sum()) for t in range(num_frames)]
        seg_info["smoothed_part_px"] = list(seg_info["part_px"])
        seg_info["face_neck_px"] = [int(seg_info["face_neck_mask"][t].sum()) for t in range(num_frames)]
        seg_info["hair_px"] = [int(seg_info["hair_mask"][t].sum()) for t in range(num_frames)]
        if masks_out is not None and seg_info["part_probs"].shape == tube_full.shape:
            seg_info["part_probs"] = np.where(relabel_hair_full, np.maximum(seg_info["part_probs"], 1.0), 0.0).astype(np.float32)
        bridged_hfp = set(seg_info["dropout_bridged_frames"])
        for t in range(num_frames):
            r_px = raw_part_px_in_tube[t]
            s_px = seg_info["part_px"][t]
            rf_px = seg_info["raw_face_neck_px"][t]
            f_px = seg_info["face_neck_px"][t]
            if (s_px > r_px and r_px < max(1, int(0.75 * s_px))) or (
                f_px > rf_px and rf_px < max(1, int(0.75 * f_px))
            ):
                bridged_hfp.add(t)
        seg_info["dropout_bridged_frames"] = sorted(bridged_hfp)
    elif raw_part_px_in_tube is not None and raw_face_neck_px_in_tube is not None:
        seg_info["raw_part_px"] = raw_part_px_in_tube
        seg_info["raw_face_neck_px"] = raw_face_neck_px_in_tube
        bridged_fp = set(seg_info["dropout_bridged_frames"])
        for t in range(num_frames):
            r_px = raw_part_px_in_tube[t]
            s_px = seg_info["part_px"][t]
            rf_px = raw_face_neck_px_in_tube[t]
            f_px = seg_info["face_neck_px"][t]
            if (s_px > r_px and r_px < max(1, int(0.75 * s_px))) or (
                f_px > rf_px and rf_px < max(1, int(0.75 * f_px))
            ):
                bridged_fp.add(t)
        seg_info["dropout_bridged_frames"] = sorted(bridged_fp)

    # Account for any raw requested-part pixels outside shot_box in off_tube_suppressed_px.
    outside_box = np.ones((height, width), dtype=bool)
    outside_box[y0:y1, x0:x1] = False
    raw_part_all = parts_bool | np.isin(cmap_arr, part_ids)
    for t in range(num_frames):
        extra = int(np.logical_and(raw_part_all[t], outside_box).sum())
        seg_info["off_tube_suppressed_px"][t] = int(seg_info["off_tube_suppressed_px"][t]) + extra

    pose_info = cross_check_face_points(
        pose_kps if pose_kps is not None else [None] * num_frames,
        tube_info["tube"],
        seg_info["face_neck_mask"],
        seg_info["hair_mask"],
        part_mask=seg_info["part_mask"],
        raw_face_neck_px=seg_info["raw_face_neck_px"],
        dropout_bridged_seed=seg_info["dropout_bridged_frames"],
        keypoint_index=kp_idx,
        drop_frames=drop_frames,
        no_face_frames=no_face_frames,
        hold_frames=hold_frames,
    )

    # THE SOURCE'S FACIAL HAIR (`face_piece_kept_whole`): after the face states are read, so they are the tube's own; the tube itself
    # is not widened (the attach rule stands), only what it cut of the face's own piece below the jaw is given back to the part
    facial_hair = None
    kept_facial_hair = None
    if keep_facial_hair:
        skip = {int(t): "no face on a person's word (--no-face-on)" for t in (no_face_frames or ())}
        skip.update({int(t): "dropped on a person's word (drop=)" for t in (drop_frames or ())})
        kept_facial_hair, facial_hair = face_piece_kept_whole(
            seg_info["part_mask"],
            tube_info["tube"],
            parts_bool,
            raw_track_bool,
            cmap=cmap_arr,
            others=np.asarray(others, dtype=bool) if others is not None else None,
            skip_frames=skip,
        )
        # each kept px keeps the part model's own class where it is one of the part's, else Face_Neck (so the part stays face-only
        # in the capture's class map, and the raw class breakdown is recorded in kept_px_by_raw_class)
        kept_cls = np.where(np.isin(cmap_arr, part_ids), cmap_arr, np.uint8(face_id)).astype(np.uint8)
        facial_hair["kept_px_by_class"] = {CLASS_NAMES[c]: [int(((kept_cls[t] == c) & kept_facial_hair[t]).sum()) for t in range(num_frames)]
                                           for c in sorted(int(v) for v in np.unique(kept_cls[kept_facial_hair]))}
        facial_hair["kept_px_by_raw_class"] = {CLASS_NAMES[c]: [int(((cmap_arr[t] == c) & kept_facial_hair[t]).sum()) for t in range(num_frames)]
                                               for c in sorted(int(v) for v in np.unique(cmap_arr[kept_facial_hair]))}
        # WHAT WAS KEPT, in the provenance's own figures: the part counts what it now holds, and the suppressed count loses the kept
        # px it had counted (the part's own px outside the tube). The tube's own figures stay in the record
        counted = kept_facial_hair & ~tube_info["tube"] & (np.isin(cmap_arr, part_ids) | parts_bool)
        facial_hair["part_px_before_keeping"] = [int(v) for v in seg_info["part_px"]]
        facial_hair["off_tube_suppressed_px_before_keeping"] = [int(v) for v in seg_info["off_tube_suppressed_px"]]
        facial_hair["kept_px_it_counted_as_suppressed"] = [int(counted[t].sum()) for t in range(num_frames)]
        seg_info["part_px"] = [int(v) + int(kept_facial_hair[t].sum()) for t, v in enumerate(seg_info["part_px"])]
        seg_info["off_tube_suppressed_px"] = [int(v) - int(counted[t].sum()) for t, v in enumerate(seg_info["off_tube_suppressed_px"])]
        seg_info["part_mask"] = seg_info["part_mask"] | kept_facial_hair
        seg_info["class_map"][kept_facial_hair] = kept_cls[kept_facial_hair]
        if masks_out is not None and seg_info["part_probs"].shape == kept_facial_hair.shape:
            seg_info["part_probs"] = np.where(kept_facial_hair, np.maximum(seg_info["part_probs"], 1.0), seg_info["part_probs"]).astype(np.float32)

    provenance = build_provenance_record(
        seg_info,
        pose_info,
        seed_frame=seed_frame,
        outside_person_track_clipped_px=tube_info["outside_person_track_clipped_px"],
        regained_from_person_track_frames=tube_info["regained_from_person_track_frames"],
        regain_refused=tube_info["regain_refused"],
        bridged_short_dips=tube_info["bridged_short_dips"],
        source_facial_hair=facial_hair,
        short_dips=tube_info["short_dips"],
    )
    if provenance_out is not None:
        provenance_out.parent.mkdir(parents=True, exist_ok=True)
        provenance_out.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if masks_out is not None:
        masks_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            masks_out,
            tube=tube_info["tube"].astype(np.uint8),
            class_map=seg_info["class_map"],
            part_mask=seg_info["part_mask"].astype(np.uint8),
            part_probs=seg_info["part_probs"],
            shot_box=np.asarray(seg_info["shot_box"], dtype=np.int32),
            **npz_writer_fields(__file__),
        )

    # Preserve non-head body classes in cmap while replacing head-part classes with the tube-bounded smoothed result.
    updated_cmap = cmap_arr.copy()
    updated_cmap[np.isin(updated_cmap, part_ids)] = 0
    box_cls = seg_info["class_map"][:, y0:y1, x0:x1]
    box_part = seg_info["part_mask"][:, y0:y1, x0:x1]
    updated_cmap[:, y0:y1, x0:x1] = np.where(box_part, box_cls, updated_cmap[:, y0:y1, x0:x1])
    if kept_facial_hair is not None:
        updated_cmap[kept_facial_hair] = kept_cls[kept_facial_hair]     # kept under the jaw, where it may lie outside the shot box

    return {
        "tube_info": tube_info,
        "seg_info": seg_info,
        "pose_info": pose_info,
        "provenance": provenance,
        "part_mask": seg_info["part_mask"],
        "class_map": updated_cmap,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tube", type=Path, required=True, help="Input .npy/.npz boolean head tube (T, H, W)")
    parser.add_argument("--logits", type=Path, required=True, help="Input .npy/.npz Sapiens2-Seg logits (T, C, h, w)")
    parser.add_argument(
        "--person-track",
        type=Path,
        default=None,
        help="Optional .npy/.npz boolean person-track mask (T, H, W) bounding the head tube",
    )
    parser.add_argument("--pose", type=Path, default=None, help="Optional .npy/.npz Sapiens2-Pose keypoints (T, K, 3)")
    parser.add_argument("--seed-frame", type=int, default=0, help="Shot seed frame index")
    parser.add_argument(
        "--temporal-weights",
        type=float,
        nargs="+",
        default=list(DEFAULT_TEMPORAL_WEIGHTS),
        help="Odd-length temporal probability smoothing kernel (default: 0.25 0.50 0.25)",
    )
    parser.add_argument("--include-eyeglass", action="store_true", help="Include Eyeglass in head-tube parts")
    parser.add_argument("--out-masks", type=Path, default=None, help="Output .npz path for smoothed masks")
    parser.add_argument("--out-json", type=Path, required=True, help="Output .json path for provenance record")
    args = parser.parse_args(argv)

    if not args.tube.is_file():
        raise SystemExit(f"head_tube_parts: missing --tube file: {args.tube}")
    if not args.logits.is_file():
        raise SystemExit(f"head_tube_parts: missing --logits file: {args.logits}")

    tube_arr = np.load(args.tube)
    if isinstance(tube_arr, np.lib.npyio.NpzFile):
        tube_arr = tube_arr["tube"] if "tube" in tube_arr.files else tube_arr[tube_arr.files[0]]
    logits_arr = np.load(args.logits)
    if isinstance(logits_arr, np.lib.npyio.NpzFile):
        logits_arr = (
            logits_arr["logits"] if "logits" in logits_arr.files else logits_arr[logits_arr.files[0]]
        )
    person_track_arr = None
    if args.person_track is not None:
        if not args.person_track.is_file():
            raise SystemExit(f"head_tube_parts: missing --person-track file: {args.person_track}")
        pt_loaded = np.load(args.person_track)
        if isinstance(pt_loaded, np.lib.npyio.NpzFile):
            pt_key = "track" if "track" in pt_loaded.files else ("tube" if "tube" in pt_loaded.files else pt_loaded.files[0])
            person_track_arr = pt_loaded[pt_key]
        else:
            person_track_arr = pt_loaded
    pose_arr = None
    if args.pose is not None:
        if not args.pose.is_file():
            raise SystemExit(f"head_tube_parts: missing --pose file: {args.pose}")
        pose_loaded = np.load(args.pose)
        pose_arr = (
            pose_loaded["keypoints"]
            if isinstance(pose_loaded, np.lib.npyio.NpzFile) and "keypoints" in pose_loaded.files
            else pose_loaded
        )

    res = run_head_tube_parts(
        logits_arr,
        tube_arr,
        pose_keypoints=pose_arr,
        person_track=person_track_arr,
        seed_frame=args.seed_frame,
        temporal_weights=args.temporal_weights,
        include_eyeglass=args.include_eyeglass,
        provenance_out=args.out_json,
        masks_out=args.out_masks,
    )
    prov = res["provenance"]
    print(
        f"shot_box={prov['shot_box']} frames={len(prov['tube_px'])} "
        f"suppressed_px={sum(prov['off_tube_suppressed_px'])} "
        f"clipped_person_px={sum(prov['outside_person_track_clipped_px'])} "
        f"bridged={prov['dropout_bridged_frames']} "
        f"facing={len(prov['facing_frames'])} "
        f"side_with_eye={len(prov['side_with_eye_frames'])} "
        f"back_of_head={prov['back_of_head_frames']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
