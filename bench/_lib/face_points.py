"""What is read off the pose model's face points: one definition of each figure, for every tool that uses them.

    from _lib.face_points import load, eye_opening, lid_gap_px, lip_gap, face_height, lid_rings, head_place

The Sapiens2 pose model gives 308 named points a frame. `bench/pose_along_shot.py` writes them along a shot,
`bench/draw_lid_marks.py` draws the eyes from them and `bench/read_render_against_original.py` sets a render's
beside its original's. On 2026-10-11 three sessions each read the same points with a formula of their own;
this is the one copy.

Points are found by NAME, never by index: the names travel with a points file (`names` in the `.npz`), read
from the model's own config when the file is made. A points file from before the names were kept is read with
a names list given beside it.

What each figure is good for is not decided here. It was judged on consecutive frames of one close-up
(`bench/results/2026-10-11_sapiens2_pose_first_frames.md`): the eye opening rises and falls with the picture
and is right at its two ends; the lip gap is NOT a measure of an open or a shut mouth. The lines those
judgements left are `workflows/h3_config.py::POSE_READING`.

Nothing here imports torch or ComfyUI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SIDES = ("l", "r")


def load(path: str | Path, names: list[str] | None = None) -> dict:
    """A points file as `keypoints` [frames, points, 2], `scores` [frames, points], `names`, `index` and `first`.

    `first` is the source frame of row 0 when the file says it, else None. A file with no names in it needs
    `names`; one that has them ignores the argument, so the file's own always win."""
    z = np.load(str(path))
    own = [str(n) for n in z["names"]] if "names" in z.files else None
    if own is None and names is None:
        raise ValueError(f"{path} holds no point names (it was made before they were kept): give the names list it was made with")
    names = own if own is not None else list(names or [])
    kp = z["keypoints"].astype(np.float64)
    if kp.shape[1] != len(names):
        raise ValueError(f"{path} has {kp.shape[1]} points a frame and the names list has {len(names)}")
    return {"keypoints": kp, "scores": z["scores"].astype(np.float64), "names": names, "index": {n: i for i, n in enumerate(names)},
            "first": int(z["first_frame"]) if "first_frame" in z.files else None,
            "size": tuple(int(v) for v in z["size"]) if "size" in z.files else None}


def _line(p: np.ndarray, index: dict, side: str, which: str) -> np.ndarray:
    """One lash line's points, in order from the inner end to the outer."""
    tail = f"_of_{which}_lash_line"
    ids = [i for n, i in index.items() if n.startswith(f"{side}_") and n.endswith(tail)]
    a, b = p[index[f"{side}_inner_end{tail}"]], p[index[f"{side}_outer_end{tail}"]]
    pts = p[ids]
    return pts[np.argsort((pts - a) @ (b - a))]


def _gap(up: np.ndarray, lo: np.ndarray) -> float:
    """The mean, over the upper line's points, of the distance to the nearest point of the lower line."""
    return float(np.linalg.norm(up[:, None, :] - lo[None, :, :], axis=-1).min(1).mean())


def face_height(p: np.ndarray, index: dict) -> float:
    """Glabella to chin, in the picture's pixels: the length every opening is taken over."""
    return float(np.linalg.norm(p[index["center_of_glabella"]] - p[index["tip_of_chin"]]))


def lid_gap_px(p: np.ndarray, index: dict) -> tuple[float, float]:
    """The gap between the upper and the lower lash line of each eye (left, right), in pixels."""
    l, r = (_gap(_line(p, index, s, "upper"), _line(p, index, s, "lower")) for s in SIDES)
    return l, r


def eye_opening(p: np.ndarray, index: dict) -> float:
    """The two eyes' lash-line gap over the face's height, averaged. The one eye figure.

    Inherited: the formula of the pose model's first run here (2026-10-11), kept so that day's records stay
    comparable. A second formula (the area between the lines over the eye's width squared) sorted the same
    clips the same way and is not carried."""
    return sum(lid_gap_px(p, index)) / 2 / max(face_height(p, index), 1e-6)


def lip_gap(p: np.ndarray, index: dict) -> float:
    """The gap between the upper and the lower inner lip line over the face's height.

    NOT an opening. Judged on a sung close-up, it failed: with the lips together the lower line lands on the
    lip's outer edge and the gap reads as the lip's thickness, and in profile an open mouth reads nearly
    shut. It is kept as a column so two tables can be set side by side; no tool calls a mouth open or shut
    from it."""
    up = [index["center_of_upper_inner_lip"]] + [index[f"midpoint_{i}_of_upper_inner_lip"] for i in range(1, 7)]
    lo = [index["center_of_lower_inner_lip"]] + [index[f"midpoint_{i}_of_lower_inner_lip"] for i in range(1, 7)]
    return _gap(p[up], p[lo]) / max(face_height(p, index), 1e-6)


def lid_rings(p: np.ndarray, index: dict) -> list[np.ndarray]:
    """Each eye's outline (left, right): the upper lash line, then the lower one back. What a mark is drawn from."""
    return [np.concatenate([_line(p, index, s, "upper"), _line(p, index, s, "lower")[::-1]]) for s in SIDES]


def ring_area(ring: np.ndarray) -> float:
    """The area an outline encloses, in square pixels."""
    x, y = ring[:, 0], ring[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def face_ids(index: dict) -> list[int]:
    """The points that ARE the face for placing it: the lash lines, the glabella, the nose's tip and the chin's. The
    head reference's one box is drawn round these on every frame (`bench/crop_reference_to_head.py::face_box`)."""
    return [i for n, i in index.items() if n.endswith("_lash_line")] + [index[n] for n in ("center_of_glabella", "tip_of_nose", "tip_of_chin")]


def face_on_mask(p: np.ndarray, index: dict, mask: np.ndarray) -> float:
    """The share of one frame's face points (`face_ids`) that lie inside a mask [h, w] of bool; a point off the
    picture is off the mask. 1.0 is a face wholly on the mask, 0.0 a face wholly off it."""
    h, w = mask.shape
    q = np.round(p[face_ids(index)]).astype(int)
    on = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    inside = np.zeros(len(q), bool)
    inside[on] = mask[q[on, 1], q[on, 0]]
    return float(inside.mean())


def face_middle_on_mask(p: np.ndarray, index: dict, mask: np.ndarray) -> bool:
    """Whether one frame's face lies on a mask: its middle, the median of its face points (`face_ids`), is a pixel
    of the mask. The median and not the mean, so one stray point does not move the face; a face drawn from another
    person's points is off its subject's track by this, though one of its points may touch it (2026-10-14: the
    chin of a woman above the lead touched the lead's hair on four of the eleven frames read off her track)."""
    h, w = mask.shape
    x, y = np.round(np.median(p[face_ids(index)], 0)).astype(int)
    return bool(0 <= x < w and 0 <= y < h and mask[y, x])


def head_place(p: np.ndarray, index: dict) -> dict:
    """Where the head sits, in pixels: three points that are not on the eyes, and the middle of the two eyes."""
    eyes = np.mean([r.mean(0) for r in lid_rings(p, index)], axis=0)
    return {"nose": p[index["tip_of_nose"]], "glabella": p[index["center_of_glabella"]], "chin": p[index["tip_of_chin"]], "eyes": eyes}


def _score(s: np.ndarray, index: dict, tails: tuple[str, ...]) -> float:
    return float(np.median([s[i] for n, i in index.items() if n.endswith(tails)]))


def lid_score(s: np.ndarray, index: dict) -> float:
    """The model's own score at the median over the lash, eyelid and crease lines.

    It does NOT say lids were seen: on a head turned away the model still places the points and scores them
    well over a half, and on a face a few pixels tall it scores them near one. Read the face's height and the
    gap in pixels beside it."""
    return _score(s, index, ("_lash_line", "_eyelid_line", "_crease_line"))


def lip_score(s: np.ndarray, index: dict) -> float:
    """The model's own score at the median over the lips' points."""
    return _score(s, index, ("_inner_lip", "_outer_lip", "_of_mouth", "_cupid_bow"))


#: The columns of a table of a shot, in order. One row a frame.
COLUMNS = ("frame", "source_frame", "face_height_px", "eyes", "lid_gap_px", "lip_gap", "score_lids", "score_lips",
           "nose_x", "nose_y", "glabella_x", "glabella_y", "chin_x", "chin_y", "eyes_x", "eyes_y")


def table(points: dict, first: int | None = None) -> list[dict]:
    """One row a frame of every figure above, positions in the picture's pixels. `first` is row 0's source frame."""
    idx, rows = points["index"], []
    first = points["first"] if first is None else first
    for f, (p, s) in enumerate(zip(points["keypoints"], points["scores"])):
        at = head_place(p, idx)
        rows.append({"frame": f, "source_frame": None if first is None else first + f,
                     "face_height_px": face_height(p, idx), "eyes": eye_opening(p, idx), "lid_gap_px": sum(lid_gap_px(p, idx)) / 2,
                     "lip_gap": lip_gap(p, idx), "score_lids": lid_score(s, idx), "score_lips": lip_score(s, idx),
                     "nose_x": at["nose"][0], "nose_y": at["nose"][1], "glabella_x": at["glabella"][0], "glabella_y": at["glabella"][1],
                     "chin_x": at["chin"][0], "chin_y": at["chin"][1], "eyes_x": at["eyes"][0], "eyes_y": at["eyes"][1]})
    return rows
