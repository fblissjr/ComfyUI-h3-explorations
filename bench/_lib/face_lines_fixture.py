"""A whole face made of points whose every line is known by construction, for the checks of the tools that draw it.

    from _lib import face_lines_fixture as fl

`bench/_lib/face_points_fixture.py` has the few names `bench/_lib/face_points.py` reads. A tool that draws the
face needs the rest: the eyelid and crease lines, the iris and pupil rings, the brows, the nose, the lips, the
dots. `lines_of` places them, `face` returns the whole set of points a frame, and `truth` gives every line and
ring in the order it runs, built from the construction and from no tool's code. The midpoints are NAMED out of
order along each line, and the ring's border points are listed out of order round it, as the pose model does:
a tool that joins points in the file's order crosses itself here.

Moved out of `bench/check_draw_face_points.py` on 2026-10-12, when a second check needed the same face.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import face_points_fixture as fx

#: The face's height in pixels (glabella to chin), how far the lips part, and how far the eyes are open.
FACE_PX, LIPS, OPEN = 80.0, 6.0, 12.0

#: The clock positions of an iris's or a pupil's eight border points, as the model names them, in degrees with y downward.
#: Listed out of order round the ring on purpose: a ring joined in the file's order crosses itself.
CLOCK = {"12": -90, "6": 90, "3": 0, "9": 180, "midpoint_1": -45, "midpoint_3": 135, "midpoint_4": -135, "midpoint_2": 45}
IRIS_R, PUPIL_R = 4.0, 2.0


def lines_of(centre, opening: float = OPEN) -> dict[str, list[tuple[str, tuple[float, float]]]]:
    """One frame's extra points as {line: [(name, (x, y)) in order ALONG the line]}. The names are out of order on purpose."""
    cx, cy = centre
    out: dict[str, list] = {}
    for s, sign in (("l", 1.0), ("r", -1.0)):
        ex = lambda t: cx + sign * (fx.EYE_APART / 2 + t * fx.EYE_WIDE)  # noqa: E731
        arc = lambda t: float(np.sin(np.pi * t))                         # noqa: E731
        out[f"{s}_upper_eyelid_line"] = [(f"{s}_{n}_of_upper_eyelid_line", (ex(t), cy - (opening + 4) * arc(t)))
                                         for t, n in ((0.2, "midpoint_2"), (0.5, "centerpoint"), (0.8, "midpoint_1"), (1.0, "outer_end"))]
        out[f"{s}_lower_eyelid_line"] = [(f"{s}_{n}_of_lower_eyelid_line", (ex(t), cy + (fx.SAG + 4) * arc(t)))
                                         for t, n in ((0.2, "midpoint_1"), (0.5, "centerpoint"), (0.8, "midpoint_2"), (1.0, "outer_end"))]
        out[f"{s}_upper_crease_line"] = [(f"{s}_{n}_of_upper_crease_line", (ex(t), cy - 1 - (opening + 8) * arc(t)))
                                         for t, n in ((0.15, "midpoint_2"), (0.5, "centerpoint"), (0.85, "midpoint_1"))]
        mid = (ex(0.5), cy - opening / 2 + 1)
        for kind, r in (("iris", IRIS_R), ("pupil", PUPIL_R)):
            out[f"{s}_{kind}_centre"] = [(f"{s}_center_of_{kind}", mid)]
            out[f"{s}_{kind}_ring"] = [(f"{s}_border_of_{kind}_{n}", (mid[0] + r * np.cos(np.radians(a)), mid[1] + r * np.sin(np.radians(a))))
                                       for n, a in CLOCK.items()]
        bx = lambda t: cx + sign * (6 + t * 30)                          # noqa: E731
        up = [(f"{n}_of_{s}_eyebrow", (bx(t), cy - 34 - 4 * arc(t * 0.8))) for t, n in
              ((0.0, "upper_startpoint"), (0.3, "upper_midpoint_2"), (0.6, "upper_midpoint_1"), (0.85, "upper_midpoint_3"))]
        lo = [(f"{n}_of_{s}_eyebrow", (bx(t), cy - 30 - 4 * arc(t * 0.8))) for t, n in
              ((0.0, "lower_startpoint"), (0.3, "lower_midpoint_2"), (0.6, "lower_midpoint_1"), (0.85, "lower_midpoint_3"))]
        out[f"{s}_eyebrow"] = up + [(f"end_of_{s}_eyebrow", (bx(1.0), cy - 33))] + lo[::-1]
    out["nose_bridge"] = [(n, (cx, cy + dy)) for n, dy in (("center_of_nose_root", -4), ("midpoint_2_of_nose_bridge", 2),
                                                          ("midpoint_1_of_nose_bridge", 7), ("midpoint_3_of_nose_bridge", 12), ("tip_of_nose_bridge", 16))]
    out["nose_base"] = [(n, (cx + dx, cy + dy)) for n, dx, dy in (("r_outer_corner_of_nose", -8, 22), ("inner_corner_of_r_nostril", -3, 24),
                                                                  ("bottom_center_of_nose", 0, 25), ("inner_corner_of_l_nostril", 3, 24),
                                                                  ("l_outer_corner_of_nose", 8, 22))]
    out["dots"] = [(n, (cx + dx, cy + dy)) for n, dx, dy in (("outer_corner_of_r_nostril", -6, 20), ("upper_corner_of_r_nostril", -4, 19),
                                                             ("outer_corner_of_l_nostril", 6, 20), ("upper_corner_of_l_nostril", 4, 19),
                                                             ("center_of_labiomental_groove", 0, 54))]
    lip = lambda dx: float(np.sin(np.pi * (dx + 20) / 40))               # noqa: E731
    xs = (-15, -10, -5, 0, 5, 10, 15)
    up = [(n, (cx + dx, cy + 35 - 5 * lip(dx))) for dx, n in zip(xs, ("midpoint_3_of_upper_outer_lip", "midpoint_1_of_upper_outer_lip",
          "midpoint_4_of_upper_outer_lip", "center_of_cupid_bow", "midpoint_5_of_upper_outer_lip", "midpoint_2_of_upper_outer_lip", "midpoint_6_of_upper_outer_lip"))]
    lo = [(n, (cx + dx, cy + 35 + 12 * lip(dx))) for dx, n in zip(xs, ("midpoint_3_of_lower_outer_lip", "midpoint_1_of_lower_outer_lip",
          "midpoint_4_of_lower_outer_lip", "center_of_lower_outer_lip", "midpoint_5_of_lower_outer_lip", "midpoint_2_of_lower_outer_lip", "midpoint_6_of_lower_outer_lip"))]
    out["outer_lip_ring"] = [("r_outer_corner_of_mouth", (cx - 20, cy + 35))] + up + [("l_outer_corner_of_mouth", (cx + 20, cy + 35))] + lo[::-1]
    out["inner_corners"] = [("r_inner_corner_of_mouth", (cx - 16, cy + 37)), ("l_inner_corner_of_mouth", (cx + 16, cy + 37))]
    return out


EXTRA = [n for line in lines_of((0.0, 0.0)).values() for n, _ in line]
NAMES = fx.NAMES + EXTRA
INDEX = {n: i for i, n in enumerate(NAMES)}


def face(centres, opening: float = OPEN) -> np.ndarray:
    """Points [frames, len(NAMES), 2]: the shared fixture's points, then this check's lines."""
    centres = np.atleast_2d(np.asarray(centres, float))
    base = fx.face(np.full(len(centres), opening), centres, face_px=FACE_PX, lips=LIPS)
    extra = np.array([[xy for line in lines_of(c, opening).values() for _, xy in line] for c in centres])
    return np.concatenate([base, extra], 1)


def save(path: Path, keypoints: np.ndarray, size: tuple[int, int], first: int | None = None, scores: np.ndarray | None = None, names: bool = True) -> Path:
    """A points file as `bench/pose_along_shot.py` writes one, with this check's names."""
    extra = {}
    if names:
        extra["names"] = np.array(NAMES)
    if first is not None:
        extra["first_frame"] = np.array(first)
    np.savez_compressed(path, keypoints=keypoints.astype(np.float32), size=np.array(size),
                        scores=(np.full(keypoints.shape[:2], 0.9, np.float32) if scores is None else scores.astype(np.float32)), **extra)
    return path


def truth(centre, opening: float = OPEN) -> dict[str, tuple[np.ndarray, bool]]:
    """Every line and ring of one frame as (points in order, whether it closes), built here from the construction and no tool's code."""
    built = lines_of(centre, opening)
    cx, cy = centre
    out = {}
    for s, sign in (("l", 1.0), ("r", -1.0)):
        x = [cx + sign * (fx.EYE_APART / 2 + t * fx.EYE_WIDE) for t in fx.T]
        up = [(xi, cy - opening * np.sin(np.pi * t)) for xi, t in zip(x, fx.T)]
        lo = [(xi, cy + fx.SAG * np.sin(np.pi * t)) for xi, t in zip(x, fx.T)]
        out[f"{s}_lash_ring"] = (np.array(up + lo[::-1]), True)
        for line in ("upper_eyelid_line", "lower_eyelid_line", "upper_crease_line"):
            out[f"{s}_{line}"] = (np.array([xy for _, xy in built[f"{s}_{line}"]]), False)
        for kind in ("iris", "pupil"):                                # round the ring by the clock, not in the order the names are listed
            at = dict(built[f"{s}_{kind}_ring"])
            out[f"{s}_{kind}_ring"] = (np.array([at[f"{s}_border_of_{kind}_{n}"] for n in sorted(CLOCK, key=CLOCK.get)]), True)
        out[f"{s}_eyebrow"] = (np.array([xy for _, xy in built[f"{s}_eyebrow"]]), True)
    out["nose_bridge"] = (np.array([xy for _, xy in built["nose_bridge"]]), False)
    out["nose_base"] = (np.array([xy for _, xy in built["nose_base"]]), False)
    out["outer_lip_ring"] = (np.array([xy for _, xy in built["outer_lip_ring"]]), True)
    lipx = [cx + (j - 3) * fx.LIP_WIDE / 6 for j in range(7)]
    out["inner_lip_ring"] = (np.array([(cx - 16, cy + 37)] + [(x, cy + 34) for x in lipx] + [(cx + 16, cy + 37)] + [(x, cy + 34 + LIPS) for x in lipx[::-1]]), True)
    return out
