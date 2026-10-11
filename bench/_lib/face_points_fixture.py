"""A face made of points whose every figure is known by construction, for the checks of the tools that read them.

    from _lib.face_points_fixture import NAMES, face, save, eye_area

Not the pose model's points and not its names in full: only the names `bench/_lib/face_points.py` looks for.
Each eye's two lash lines are arcs that meet at both ends: the upper one stands `opening` pixels above the
line between the ends in the middle, the lower one hangs `SAG` pixels under it. So a frame's opening, its
area and where the head sits are numbers the check chose and can hold a tool's output to. The lower line is
bowed on purpose: with a level one an outline whose lower line runs the wrong way round encloses the same
area, and a tool that drew it that way went unseen.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

T = (0.0, 0.25, 0.5, 0.75, 1.0)                       # along an eye, inner end to outer
PARTS = ("inner_end", "midpoint_1", "centerpoint", "midpoint_2", "outer_end")
EYE_WIDE, EYE_APART, LIP_WIDE = 24.0, 16.0, 28.0      # pixels
SAG = 2.0                                             # how far the lower lash line hangs under the line between an eye's ends, pixels

NAMES = ["center_of_glabella", "tip_of_chin", "tip_of_nose"]
NAMES += [f"{s}_{part}_of_{which}_lash_line" for s in ("l", "r") for which in ("upper", "lower") for part in PARTS]
NAMES += [f"{n}_of_{which}_inner_lip" for which in ("upper", "lower") for n in ["center"] + [f"midpoint_{i}" for i in range(1, 7)]]


def face(opening, centre, face_px: float = 60.0, lips: float = 2.0) -> np.ndarray:
    """Points [frames, len(NAMES), 2]. `opening` is pixels a frame; `centre` is one (x, y) or one a frame."""
    opening = np.atleast_1d(np.asarray(opening, float))
    centre = np.broadcast_to(np.asarray(centre, float), (len(opening), 2))
    out = np.zeros((len(opening), len(NAMES), 2))
    k = {n: i for i, n in enumerate(NAMES)}
    for f, (o, (cx, cy)) in enumerate(zip(opening, centre)):
        out[f, k["center_of_glabella"]] = (cx, cy - 10)
        out[f, k["tip_of_chin"]] = (cx, cy - 10 + face_px)
        out[f, k["tip_of_nose"]] = (cx, cy + 18)
        for s, sign in (("l", 1.0), ("r", -1.0)):
            for t, part in zip(T, PARTS):
                x = cx + sign * (EYE_APART / 2 + t * EYE_WIDE)
                out[f, k[f"{s}_{part}_of_upper_lash_line"]] = (x, cy - o * np.sin(np.pi * t))
                out[f, k[f"{s}_{part}_of_lower_lash_line"]] = (x, cy + SAG * np.sin(np.pi * t))
        for j, n in enumerate(["center"] + [f"midpoint_{i}" for i in range(1, 7)]):
            x = cx + (j - 3) * LIP_WIDE / 6
            out[f, k[f"{n}_of_upper_inner_lip"]] = (x, cy + 34)
            out[f, k[f"{n}_of_lower_inner_lip"]] = (x, cy + 34 + lips)
    return out


def eye_area(opening: float) -> float:
    """The area of ONE eye's outline at an opening, by the trapezoids under the arc's five points: not by the tools' code."""
    y = [(opening + SAG) * np.sin(np.pi * t) for t in T]
    return float(sum((y[i] + y[i + 1]) / 2 * EYE_WIDE * (T[i + 1] - T[i]) for i in range(len(T) - 1)))


def save(path: Path, keypoints: np.ndarray, size: tuple[int, int], first: int | None = None, scores: np.ndarray | None = None,
         names: bool = True) -> Path:
    """Write a points file as `bench/pose_along_shot.py` does. `names=False` writes one from before the names were kept."""
    extra = {}
    if names:
        extra["names"] = np.array(NAMES)
    if first is not None:
        extra["first_frame"] = np.array(first)
    np.savez_compressed(path, keypoints=keypoints.astype(np.float32), size=np.array(size),
                        scores=(np.full(keypoints.shape[:2], 0.9, np.float32) if scores is None else scores.astype(np.float32)), **extra)
    return path
