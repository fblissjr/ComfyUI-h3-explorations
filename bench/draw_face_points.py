#!/usr/bin/env python3
"""Draw a shot's face on its body-model video, as lines and dots through the pose model's points.

    python bench/draw_face_points.py --mesh MESH.mp4 --points SHOT_points.npz --first 1039 --count 22 \\
        --out <share>/Video/<video>/<who>/second_mesh_with_face_points_1039.mp4 --by <session> [--source SOURCE.mov] \\
        [--bare 1045-1047 --bare-by <who looked>] [--width 2] [--no-dots] [--leave-out mouth] \
        [--mesh-first N] [--points-first N] [--source-first N]

The body-model video carries the head; its face is a neutral modelled one that never changes. The lid marks
(`bench/draw_lid_marks.py`) put two filled shapes on it, and a model shown them is not told what they are. This
draws the ORIGINAL's face on the body model's own frame from the pose model's points of that frame
(`bench/pose_along_shot.py`), so that an eye is drawn as an eye: white lines with a thin black edge, and a dot at
every point. Everything is at the face's TRUE size and place: nothing is grown or moved.

It is the marks tool's sibling and shares what is not drawing with it (the writer, the look at the body model
against the source, the slip rule), imported from that file so there is one copy.

WHICH POINT BELONGS TO WHICH LINE. Points are found by name (`vendor/meta_sapiens2/pose_keypoints308.json`).
A ring closes on itself; a line does not.

- each eye (`l`, `r`): the lash ring, the upper lash line and the lower one back
  (`bench/_lib/face_points.py::lid_rings`); the upper eyelid line, the lower eyelid line and the upper crease
  line, each of every point named `<side>_..._of_<that line>`; the iris ring, the eight
  `<side>_border_of_iris_...` points, and the pupil ring, the eight `<side>_border_of_pupil_...` points.
- each eyebrow: one ring, the points named `upper_..._of_<side>_eyebrow`, then `end_of_<side>_eyebrow`, then
  the points named `lower_..._of_<side>_eyebrow` back.
- the nose: the bridge, a line through `center_of_nose_root`, the `midpoint_N_of_nose_bridge` points and
  `tip_of_nose_bridge`; the base, a line through `r_outer_corner_of_nose`, `inner_corner_of_r_nostril`,
  `bottom_center_of_nose`, `inner_corner_of_l_nostril` and `l_outer_corner_of_nose`; and as dots alone,
  `tip_of_nose` and each nostril's outer and upper corner.
- the mouth: the outer lip ring (`r_outer_corner_of_mouth`, the `..._of_upper_outer_lip` points with
  `center_of_cupid_bow`, `l_outer_corner_of_mouth`, the `..._of_lower_outer_lip` points back) and the inner
  lip ring (the two inner corners with the `..._of_upper_inner_lip` and `..._of_lower_inner_lip` points).
- dots alone: `center_of_glabella`, `center_of_labiomental_groove`, `tip_of_chin`.

THE POINTS DRAWN ALONE are those three and, with the nose, `tip_of_nose` and each nostril's outer and upper
corner: eight points that belong to no line. They are dots, so `--no-dots` leaves them out with the dots on the
lines: lines only means lines only. Until the evening of 2026-10-12 `--no-dots` took the dot off each point of a
line and still drew these eight, each a speck as wide as a line, between the brows, under the lip, at the chin
(over the jaw or the neck on a tilted head) and round the nose. The references drawn that day before the change
carry them: the second subject's short kitchen shot (`second_mesh_head_crop_then_face_points_no_mouth_1039`, the
reference of the load `1039_f`). The record lists the points drawn alone on a run, or none.

Not the ears and not the body: the body model carries those.

A PART CAN BE LEFT OUT by name (`--leave-out`, any of `eyes`, `brows`, `nose`, `mouth`, `dots`), for a shot
where a person saw that part's points are not on her face. The tool does not find that out: on the first shot
drawn the lips' points sat above an open mouth on most frames, with good scores. The record says what was
left out.

THE ORDER ALONG A LINE is by where a point lies between the line's two ends, never by the number in its name:
the model's `midpoint_1` to `midpoint_6` do not run in order along a line, and run differently on the two
sides of the face. A lash line has named ends (`_line` in `face_points.py`); an eyelid or a crease line has
none, so its points are laid along the same eye's upper lash line, inner end to outer; a brow's run from its
start to its end, the bridge's from the root to its tip, a lip's from one corner of the mouth to the other.
An iris or a pupil ring runs by angle round its own centre point.

AN IRIS OR A PUPIL RING IS DRAWN ONLY BETWEEN THE LIDS (`IRIS_FORMS`, since 2026-10-14): the ring is sampled densely
(`RING_SAMPLES`) and only its stretches inside that eye's lash ring are drawn, each as an open line; a ring wholly
inside is drawn closed, as before. `--iris whole` draws the whole ring, the form before; `--iris filled` draws each iris as a dark disc inside the lash ring
(the edge colour) and no ring, a form a render is to decide on; `--iris contrast` fills, before the lines, each lash ring's
sclera solid white and its iris solid black, the lines' black edge round them (the owner's word: the grey disc was lossy), the white left out of the read-back. With the lids lowered the
whole ring was larger than the opening and read as an open eye looking down. The record says which was drawn and,
for each ring, the share of it that lay outside the lids, least, median and most over the frames.

WHAT IS LEFT OUT, and said in the record: an iris or a pupil ring on a frame where the model's score, at the
median over that ring's points, is under `h3_config.POSE_READING["eye_ring_score_at_least"]`. On a small face
the pupil points score low and lie far off the face; that line is a starting value from one shot.

THE WIDTH. One width for the whole shot: the median face height over `h3_config.FACE_POINTS["face_over_width"]`,
never under its `least_px`, or `--width` to set it by hand. A dot's radius is the line's width. The record
says the width and what set it.

MADE THE WAY META DRAWS THEM, IN PART. Meta's own drawing of these points is dots only: its config links no
face point to another (`coderef/sapiens2`, `sapiens/pose/configs/_base_/keypoints308.py`, `skeleton_info`).
The lines are this tool's.

WHAT IT CANNOT KNOW, and says:
- Whether a head is turned away. The model still places face points on the back of a head and scores them
  well. Frames to leave bare come from a person, by name (`--bare`, `--bare-by`).
- Whether the points and the body-model video are on the source's clock. It is told which source frame each
  begins on and believes it. With `--source` it sets how much the body-model video changes from frame to frame
  against how much the source does, a frame early and a frame late, and prints the three figures with the box
  they were read in (data, no part of PASSED: whatever else moves in that box counts too); the
  sheet lays the drawn frames over the original for a person; and `verify`, below, is for a second reader.
- Whether the lines can be seen by a model at this size. On a small face a whole eye is a few pixels across
  and its lines and dots run together: open the sheet.

WHAT THE READ-BACK IS. After writing, the tool reads the video back. It counts the white it finds on each frame
beside the white the frame's lines should leave, and it measures how far that white lies from the lines of the
frame's own points, of the frame before's and of the frame after's. A face's lines are nearly as long on one
frame as on the next, so the COUNT cannot tell a frame from its neighbour, and it is the DISTANCE that is
fitted by a frame of slip. That catches a file whose frames do not hold what was drawn for them: one dropped,
repeated or out of order, lines on a frame left bare, a frame with none. When the face does not move between
frames no slip can be told, and the tool says so. It cannot catch points given on the wrong clock, since the
drawing and the lines it is read against come from the same rows.

    python bench/draw_face_points.py verify --video DRAWN.mp4 --mesh MESH.mp4 --points SHOT_points.npz --first 1039 --count 22

`verify` reads a drawn video somebody else made against a points file, with the reader's OWN account of which
frame each begins on. There the same measure does catch a wrong clock, because the rows are chosen by the
reader and not by the maker.

A passing run says the video holds the lines its points describe, on the frames they describe. It does not
say the lines are on her face: open the sheet.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO  # noqa: E402
from _lib import face_points as fp  # noqa: E402
from _lib.frames import probe, stream  # noqa: E402

import draw_lid_marks as marks  # noqa: E402  the writer, the look against the source and the slip rule: one copy

sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402

SUPER, WHITE = marks.SUPER, marks.WHITE
#: How much of a line's width is not white: the black edge and the encoder take about this much across it, half
#: on each side. Measured: `bench/check_draw_face_points.py` draws a face at several widths and holds the white it
#: reads back to the lines' length times the width less this, to about a tenth (lines that touch share white, and a
#: round end adds some).
EDGE_PX = 1.0
#: A frame whose lines should leave less white than this (pixels) may leave none: it is listed as too thin and is
#: not a failure. Inherited: the marks tool's `THIN_PX2`.
THIN_PX = marks.THIN_PX2
#: How much nearer the white must lie to its own frame's lines than to a neighbour's for the read-back to say the
#: frames are in place, and how much nearer to a neighbour's for it to fail, in pixels of mean distance. Under it
#: either way, the face did not move enough to tell. Reasoned: a face that moves a pixel between frames moves the
#: mean distance by several times this (`bench/check_draw_face_points.py` holds a slipped file to it).
SLIP_PX = 0.05

#: A pixel is a line's white only where the body model under it was darker than white by at least this many grey
#: levels. Inherited: the marks tool's, which has the measurement beside it; found here first.
NEAR_WHITE = marks.NEAR_WHITE

SIDES = fp.SIDES
RINGS = ("iris", "pupil")
#: How an iris or a pupil ring is drawn: only the stretches of it that lie between the lids (the default since
#: 2026-10-14), or `whole` (as before, behind `--iris whole`). The fault it answers, found twice that day
#: independently (the lead's investigation of the 10/12 references, internal/claude/2026-10-14_mrtwo_scripts/eyes/
#: findings.md item 4; the card holder on second_hfp_0484's held reference): with her lids drawn lowered, the
#: iris ring drawn whole is larger than the opening, about 42% of it outside the lids, and the drawing reads as
#: an open eye looking down.
#: `filled` (2026-10-14, the lead's order, behind its flag until a render decides): the iris a dark disc clipped to the lid
#: outline, no ring. On lead_fp_1097b the face points carried her gaze to screen right (the iris centre +0.25 to +0.31 eye
#: widths on 1121-1133) and the reference drew the ring off-centre, and the render looked at the camera: a thin outline
#: did not carry it (internal/claude/2026-10-14_mrtwo_scripts/gaze/).
#: `contrast` (2026-10-14, the owner's word after seeing `filled` as "a slightly darker grey" at 1131): the sclera inside each
#: lash ring filled solid white and the iris solid black, both clipped to the lids, drawn before the lid lines so their
#: black edge stays round both (drawn after them until an outside reviewer's sheet showed the edge lost).
IRIS_FORMS = ("between the lids", "whole", "filled", "contrast")
#: Points a ring is sampled at to find where it lies between the lids. Reasoned: a ring is eight points round a
#: centre; 64 along it puts a sample about every 6 degrees, finer than a line is wide on any face drawn so far.
RING_SAMPLES = 64
#: What `--leave-out` can name: every shape drawn belongs to one.
PARTS = ("eyes", "brows", "nose", "mouth", "dots")


def _ids(index: dict, head: str = "", tail: str = "") -> list[int]:
    return [i for n, i in index.items() if n.startswith(head) and n.endswith(tail)]


def _along(p: np.ndarray, ids: list[int], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The points `ids` in order of where each lies between `a` and `b`."""
    pts = p[ids]
    return pts[np.argsort((pts - a) @ (b - a), kind="stable")]


def _round(p: np.ndarray, ids: list[int], centre: np.ndarray) -> np.ndarray:
    """The points `ids` in order of angle round `centre`."""
    pts = p[ids]
    return pts[np.argsort(np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0]), kind="stable")]


def missing(index: dict) -> list[str]:
    """The names this drawing needs that a points file does not have."""
    need = ["center_of_glabella", "center_of_labiomental_groove", "tip_of_chin", "tip_of_nose", "center_of_nose_root", "tip_of_nose_bridge",
            "bottom_center_of_nose", "r_outer_corner_of_nose", "l_outer_corner_of_nose", "center_of_cupid_bow"]
    need += [f"{s}_{end}_corner_of_mouth" for s in SIDES for end in ("outer", "inner")]
    need += [f"{kind}_corner_of_{s}_nostril" for s in SIDES for kind in ("inner", "outer", "upper")]
    need += [f"{kind}_of_{s}_eyebrow" for s in SIDES for kind in ("upper_startpoint", "lower_startpoint", "end")]
    need += [f"{s}_{end}_end_of_upper_lash_line" for s in SIDES for end in ("inner", "outer")]
    need += [f"{s}_center_of_{ring}" for s in SIDES for ring in RINGS]
    return [n for n in need if n not in index]


def ring_scores(s: np.ndarray, index: dict) -> dict[str, float]:
    """The model's score for each eye's iris and pupil ring, at the median over the ring's points and its centre."""
    return {f"{side}_{ring}": float(np.median(s[_ids(index, f"{side}_border_of_{ring}") + [index[f"{side}_center_of_{ring}"]]]))
            for side in SIDES for ring in RINGS}


def inside(xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Which of the points `xy` [k, 2] lie inside the closed `polygon` [m, 2] (even-odd rule)."""
    x, y = xy[:, 0][:, None], xy[:, 1][:, None]
    a, b = polygon, np.roll(polygon, -1, axis=0)
    crosses = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < (b[:, 0] - a[:, 0]) * (y - a[:, 1]) / np.where(b[:, 1] == a[:, 1], 1e-12, b[:, 1] - a[:, 1]) + a[:, 0])
    return (crosses.sum(1) % 2) == 1


def between_the_lids(ring: np.ndarray, lids: np.ndarray) -> tuple[list[np.ndarray], bool, float]:
    """A ring's stretches that lie inside the lid ring: (the stretches, whether the whole ring is inside, the share
    of the ring outside). The ring is sampled at `RING_SAMPLES` points along it."""
    k = len(ring)
    t = np.linspace(0.0, k, RING_SAMPLES, endpoint=False)
    i = np.floor(t).astype(int)
    dense = ring[i % k] + (t - i)[:, None] * (ring[(i + 1) % k] - ring[i % k])
    keep = inside(dense, lids)
    if keep.all():
        return [ring], True, 0.0
    if not keep.any():
        return [], False, 1.0
    start = int(np.argmin(keep))                                   # begin on a sample outside, so no stretch is cut at the seam
    order = np.r_[start:RING_SAMPLES, 0:start]
    runs, run = [], []
    for j in order:
        if keep[j]:
            run.append(dense[j])
        elif run:
            runs.append(np.array(run))
            run = []
    if run:
        runs.append(np.array(run))
    return [r for r in runs if len(r) > 1], False, float(1.0 - keep.mean())


def shapes(p: np.ndarray, s: np.ndarray, index: dict, score_line: float, leave_out: frozenset = frozenset(),
           iris: str = IRIS_FORMS[0], outside: dict | None = None) -> list[tuple[str, np.ndarray, bool]]:
    """One frame's drawing as (name, points in order, whether it closes on itself). A dot alone is one point.
    `iris`: a ring drawn between the lids or whole (`IRIS_FORMS`); `outside`, when given, gets each drawn ring's
    share outside the lids, by its name."""
    out: list[tuple[str, np.ndarray, bool]] = []
    scores = ring_scores(s, index)
    for side, ring in zip(SIDES, fp.lid_rings(p, index)):
        if "eyes" not in leave_out:
            a, b = p[index[f"{side}_inner_end_of_upper_lash_line"]], p[index[f"{side}_outer_end_of_upper_lash_line"]]
            out.append((f"{side}_lash_ring", ring, True))
            for line in ("upper_eyelid_line", "lower_eyelid_line", "upper_crease_line"):
                out.append((f"{side}_{line}", _along(p, _ids(index, f"{side}_", f"_of_{line}"), a, b), False))
            for kind in RINGS:
                if scores[f"{side}_{kind}"] >= score_line:
                    whole = _round(p, _ids(index, f"{side}_border_of_{kind}"), p[index[f"{side}_center_of_{kind}"]])
                    runs, closed, off = between_the_lids(whole, ring)
                    if outside is not None:
                        outside[f"{side}_{kind}"] = off
                    if iris in (IRIS_FORMS[2], IRIS_FORMS[3]):
                        continue                                       # filled or contrast: the iris is a disc (`discs`), and no ring is drawn
                    if iris == IRIS_FORMS[1] or closed:
                        out.append((f"{side}_{kind}_ring", whole, True))
                    else:
                        out += [(f"{side}_{kind}_ring", run, False) for run in runs]
        if "brows" not in leave_out:
            end = p[index[f"end_of_{side}_eyebrow"]]
            upper = _along(p, _ids(index, "upper_", f"_of_{side}_eyebrow"), p[index[f"upper_startpoint_of_{side}_eyebrow"]], end)
            lower = _along(p, _ids(index, "lower_", f"_of_{side}_eyebrow"), p[index[f"lower_startpoint_of_{side}_eyebrow"]], end)
            out.append((f"{side}_eyebrow", np.concatenate([upper, end[None], lower[::-1]]), True))
    alone = []
    if "nose" not in leave_out:
        root, tip = p[index["center_of_nose_root"]], p[index["tip_of_nose_bridge"]]
        bridge = [index["center_of_nose_root"], index["tip_of_nose_bridge"]] + _ids(index, "midpoint_", "_of_nose_bridge")
        out.append(("nose_bridge", _along(p, bridge, root, tip), False))
        base = [index[n] for n in ("r_outer_corner_of_nose", "inner_corner_of_r_nostril", "bottom_center_of_nose", "inner_corner_of_l_nostril",
                                   "l_outer_corner_of_nose")]
        out.append(("nose_base", _along(p, base, p[index["r_outer_corner_of_nose"]], p[index["l_outer_corner_of_nose"]]), False))
        alone += ["tip_of_nose"] + [f"{kind}_corner_of_{side}_nostril" for side in SIDES for kind in ("outer", "upper")]
    if "mouth" not in leave_out:
        for where, extra in (("outer", [index["center_of_cupid_bow"]]), ("inner", [])):
            a, b = p[index[f"r_{where}_corner_of_mouth"]], p[index[f"l_{where}_corner_of_mouth"]]
            upper = _along(p, _ids(index, tail=f"_of_upper_{where}_lip") + extra, a, b)
            lower = _along(p, _ids(index, tail=f"_of_lower_{where}_lip"), a, b)
            out.append((f"{where}_lip_ring", np.concatenate([a[None], upper, b[None], lower[::-1]]), True))
    if "dots" not in leave_out:
        alone += ["center_of_glabella", "center_of_labiomental_groove", "tip_of_chin"]
    out += [(n, p[index[n]][None], False) for n in alone]
    return out


def discs(p: np.ndarray, s: np.ndarray, index: dict, score_line: float, leave_out: frozenset = frozenset()) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """One frame's irises for `--iris filled`, as (name, the iris ring sampled at `RING_SAMPLES` points, that eye's lash ring):
    `draw` fills each in the edge colour where it lies inside its lash ring. An iris under the score line is left out, as its ring is."""
    if "eyes" in leave_out:
        return []
    scores, out = ring_scores(s, index), []
    for side, ring in zip(SIDES, fp.lid_rings(p, index)):
        if scores[f"{side}_iris"] >= score_line:
            whole = _round(p, _ids(index, f"{side}_border_of_iris"), p[index[f"{side}_center_of_iris"]])
            k = len(whole)
            t = np.linspace(0.0, k, RING_SAMPLES, endpoint=False)
            i = np.floor(t).astype(int)
            out.append((f"{side}_iris_disc", whole[i % k] + (t - i)[:, None] * (whole[(i + 1) % k] - whole[i % k]), ring))
    return out


def parts_arg(text: str) -> frozenset:
    """`mouth,brows` as the parts to leave out; a name that is not a part is refused."""
    asked = frozenset(t.strip() for t in text.split(",") if t.strip())
    if asked - set(PARTS):
        raise SystemExit(f"--leave-out takes any of {', '.join(PARTS)}; got {', '.join(sorted(asked - set(PARTS)))}")
    return asked


def lines_only(drawing: list[tuple[str, np.ndarray, bool]]) -> list[tuple[str, np.ndarray, bool]]:
    """A frame's drawing without the points that belong to no line: what `--no-dots` draws."""
    return [shape for shape in drawing if len(shape[1]) > 1]


def segments(drawing: list[tuple[str, np.ndarray, bool]]) -> np.ndarray:
    """Every stretch of a frame's drawing as [m, 2, 2]: a line's steps, a ring's with the step that closes it, a dot as a stretch of no length."""
    out = []
    for _, pts, closed in drawing:
        run = np.vstack([pts, pts[:1]]) if closed else pts
        out.append(np.stack([run[:-1], run[1:]], 1) if len(run) > 1 else np.stack([run, run], 1))
    return np.concatenate(out) if out else np.zeros((0, 2, 2))


def length(drawing: list[tuple[str, np.ndarray, bool]]) -> float:
    """How long a frame's lines and rings are, in pixels."""
    seg = segments(drawing)
    return float(np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1).sum())


def distance(xy: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """From each of the points `xy` [k, 2] to the nearest stretch of `seg` [m, 2, 2], in pixels."""
    if not len(xy) or not len(seg):
        return np.full(len(xy), np.inf)
    a, d = seg[:, 0], seg[:, 1] - seg[:, 0]
    dd = np.maximum((d * d).sum(1), 1e-12)
    out = np.empty(len(xy))
    for i in range(0, len(xy), 4096):                                # in pieces, so a large face does not need the whole table at once
        q = xy[i:i + 4096, None, :] - a[None]
        t = np.clip((q * d[None]).sum(-1) / dd[None], 0.0, 1.0)
        r = q - t[..., None] * d[None]
        out[i:i + 4096] = np.sqrt((r * r).sum(-1).min(1))
    return out


def white_expected(drawing: list[tuple[str, np.ndarray, bool]], width: float, dots: bool) -> float:
    """The white a frame's drawing should leave, in pixels: the pixels nearer a line than half its white core, or
    nearer a point than a dot's. Worked out from the points by distance, not by the code that draws."""
    seg = segments(drawing)
    if not len(seg):
        return 0.0
    reach = int(np.ceil(max(width, 1.0))) + 1
    lo = np.floor(seg.reshape(-1, 2).min(0)).astype(int) - reach
    size = np.ceil(seg.reshape(-1, 2).max(0)).astype(int) + reach - lo + 1
    inside = np.zeros((size[1], size[0]), bool)

    def mark(piece: np.ndarray, within: float) -> None:              # one stretch, looked at only in the pixels round it
        x0, y0 = np.floor(piece.min(0)).astype(int) - reach - lo
        x1, y1 = np.ceil(piece.max(0)).astype(int) + reach - lo + 1
        xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        xy = np.stack([xs.ravel() + lo[0], ys.ravel() + lo[1]], 1).astype(float)
        inside[y0:y1, x0:x1] |= (distance(xy, piece[None]) < within).reshape(ys.shape)

    for piece in seg:
        mark(piece, max(width - EDGE_PX, 0.0) / 2)
    if dots:
        for _, pts, _ in drawing:
            for point in pts:
                mark(np.stack([point, point]), max(width - EDGE_PX / 2, 0.0))
    return float(inside.sum())


def mouth_shapes(whole: np.ndarray, hole: np.ndarray) -> list[tuple[str, np.ndarray, bool]]:
    """A class-map mouth's outlines as the drawing's shapes: the lips' outer outline and, where the mouth is open, its inner
    edge, each a closed line (`mouth_outer`, `mouth_inner`)."""
    import cv2
    out = []
    for name, m in (("mouth_outer", whole), ("mouth_inner", hole)):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            c = cv2.approxPolyDP(c, 1.0, True)[:, 0, :].astype(float)
            if len(c) >= 3:
                out.append((name, c, True))
    return out


def draw(mesh: np.ndarray, drawings: list[list], width: float, bare: set[int], dots: bool = True, fills: list[list] | None = None,
         contrast: bool = False, mouths: list | None = None) -> np.ndarray:
    """The mesh frames [n, h, w, 3] with every frame's drawing on it. With `fills`: first, inside each lash ring only, the
    iris in the edge colour, and with `contrast` the sclera solid white (the line colour) under it; then the lines over
    them, black a little wider first, then white, so the black edge stays round the white sclera and the black iris (an
    outside reviewer, 2026-10-14: filled after the lines, the white sclera covered the lash ring's black edge and the
    eye ran into the lid line)."""
    from PIL import ImageChops
    look = h3_config.FACE_POINTS
    h, w = mesh.shape[1:3]
    out = np.empty_like(mesh)

    def fill(im, f: int, white: bool) -> None:
        for _, disc, lids in ([] if f in bare or not fills else fills[f]):
            a, b = Image.new("L", im.size, 0), Image.new("L", im.size, 0)
            ImageDraw.Draw(a).polygon([tuple(v * SUPER) for v in disc], fill=255)
            ImageDraw.Draw(b).polygon([tuple(v * SUPER) for v in lids], fill=255)
            if white:
                im.paste(tuple(look["line"]), mask=b)
            im.paste(tuple(look["edge"]), mask=ImageChops.multiply(a, b))
    def mouth(im, f: int) -> None:
        """A class-map mouth (`mouths[f]`: the mouth's whole outline filled, and its open inside, as masks of the frame): the body
        model's own lips painted out to the grey round them, and the open inside in the edge colour; its outlines are lines."""
        import cv2
        got = None if f in bare or not mouths else mouths[f]
        if got is None or not got[0].any():
            return
        whole, hole = got
        ring = cv2.dilate(whole.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~cv2.dilate(whole.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        grey = tuple(int(v) for v in np.median(mesh[f][ring], axis=0)) if ring.any() else (128, 128, 128)
        big = lambda m: Image.fromarray((m.astype(np.uint8) * 255)).resize(im.size, Image.NEAREST)        # noqa: E731
        im.paste(grey, mask=big(cv2.dilate(whole.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)))
        im.paste(tuple(look["edge"]), mask=big(hole))
    for f in range(len(mesh)):
        im = Image.fromarray(mesh[f]).resize((w * SUPER, h * SUPER), Image.BICUBIC)
        mouth(im, f)
        fill(im, f, contrast)
        d = ImageDraw.Draw(im)
        for colour, pad in ((tuple(look["edge"]), look["edge_px"]), (tuple(look["line"]), 0.0)):
            for _, pts, closed in ([] if f in bare else drawings[f]):
                run = [tuple(v * SUPER) for v in (np.vstack([pts, pts[:1]]) if closed else pts)]
                if len(run) > 1:
                    d.line(run, fill=colour, width=max(1, int(round((width + 2 * pad) * SUPER))), joint="curve")
                r = ((width if dots else width / 2) + pad) * SUPER   # a dot at every point; with none, a round end and a round joint
                for x, y in run:
                    d.ellipse([x - r, y - r, x + r, y + r], fill=colour)
        out[f] = np.asarray(im.resize((w, h), Image.LANCZOS))
    return out


def filled_px(fills: list[list] | None, size: tuple[int, int], grow: int = 2) -> np.ndarray | None:
    """Where each frame's lash rings are filled (`--iris contrast`), [n, h, w] of bool, grown a little: white a line did not draw."""
    if not fills:
        return None
    import cv2
    w, h = size
    out = np.zeros((len(fills), h, w), np.uint8)
    for f, frame in enumerate(fills):
        for _, _, lids in frame:
            cv2.fillPoly(out[f], [np.round(lids).astype(np.int32)], 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    return np.stack([cv2.dilate(m, k) for m in out]).astype(bool)


def white_at(path: Path | str, mesh: np.ndarray, skip: int = 0, but: np.ndarray | None = None) -> list[np.ndarray]:
    """For each frame, where (x, y) the written video is a line's white and the mesh it was drawn on was not; less `but`
    (the filled sclera of `--iris contrast`, white that no line drew)."""
    n, h, w = mesh.shape[:3]
    got = np.stack(list(stream(str(path), (w, h), skip, n, pix="rgb24")))
    if len(got) != n:
        raise RuntimeError(f"{path} reads back {len(got)} frames, {n} were asked for")
    found = (got.min(-1) > WHITE) & (mesh.min(-1) <= WHITE - NEAR_WHITE)
    if but is not None:
        found &= ~but
    return [np.stack(np.nonzero(f)[::-1], 1).astype(float) for f in found]


def fit_by_lag(white: list[np.ndarray], segs: list[np.ndarray], drawn: np.ndarray) -> dict[int, float]:
    """The mean distance, in pixels, of the white found on each frame from the lines of the frame that many frames
    away, over the frames that carry a drawing at both ends. Smaller is nearer."""
    out = {}
    for lag in (-1, 0, 1):
        total, count = 0.0, 0
        for f in range(len(white)):
            g = f - lag                                              # a file one frame late (lag 1) shows frame f the lines of f - 1
            if 0 <= g < len(segs) and drawn[f] and drawn[g] and len(white[f]):
                dist = distance(white[f], segs[g])
                total, count = total + float(dist.sum()), count + len(dist)
        out[lag] = total / count if count else float("nan")
    return out


def reading(lags: dict[int, float]) -> tuple[bool, bool]:
    """(a neighbour's lines fit the white better, no neighbour can be told from the frame itself)."""
    other = [v for k, v in lags.items() if k != 0 and not np.isnan(v)]
    if np.isnan(lags[0]) or not other:
        return False, True
    ahead = min(other) - lags[0]                                     # how much nearer the white is to its own frame's lines
    return bool(ahead <= -SLIP_PX), bool(abs(ahead) < SLIP_PX)


def read_back(video: Path, mesh: np.ndarray, drawings: list[list], expected: np.ndarray, drawn: np.ndarray, width: float,
              but: np.ndarray | None = None) -> dict:
    """The written video against the lines each frame should hold: what was found, how it fits by slip, a verdict."""
    white = white_at(video, mesh, but=but)
    count = np.array([len(v) for v in white], float)
    lags = fit_by_lag(white, [segments(d) for d in drawings], drawn)
    on_bare = int(count[~drawn].max()) if (~drawn).any() else 0
    empty = [int(f) for f in np.nonzero(drawn & (count == 0) & (expected > THIN_PX))[0]]
    thin = [int(f) for f in np.nonzero(drawn & (expected <= THIN_PX))[0]]
    slipped, cannot_tell = reading(lags)
    off_the_lines = bool(not np.isnan(lags[0]) and lags[0] > width)    # white that is not this drawing's: a dot reaches no further than the width
    nothing = bool(count[drawn].sum() == 0)
    passed = bool(on_bare == 0 and not empty and not slipped and not off_the_lines and (not nothing or len(thin) == int(drawn.sum())))
    return {"white": count, "lags": lags, "on_bare": on_bare, "drawn_frames_with_no_white": empty, "too_thin": thin, "cannot_tell": cannot_tell,
            "slipped": slipped, "off_the_lines": off_the_lines, "passed": passed}


def sheet(path: Path, source_frames: np.ndarray, first: int, drawn: np.ndarray, points: dict, count: int) -> list[int]:
    """A few frames in three panels (the original, the drawn body model, the two blended), cut round the nose."""
    h, w = drawn.shape[1:3]
    idx = points["index"]
    picks = sorted(set(np.linspace(0, count - 1, 6).astype(int).tolist()))
    side, tile = max(120, int(2.2 * np.median([fp.face_height(p, idx) for p in points["keypoints"]]))), 440
    rows = []
    for f in picks:
        x, y = points["keypoints"][f][idx["tip_of_nose"]]
        x0, y0 = int(min(max(x - side / 2, 0), w - side)), int(min(max(y - side * 0.6, 0), h - side))
        a, b = (Image.fromarray(v[f][y0:y0 + side, x0:x0 + side]) for v in (source_frames, drawn))
        row = Image.new("RGB", (tile * 3, tile))
        for i, t in enumerate((a, b, Image.blend(a, b, 0.5))):
            row.paste(t.resize((tile, tile), Image.LANCZOS), (i * tile, 0))
        ImageDraw.Draw(row).text((6, 6), f"source frame {first + f}: the original | the body model with the face drawn | the two blended", fill=(255, 255, 0))
        rows.append(row)
    out = Image.new("RGB", (tile * 3, tile * len(rows)))
    for i, r in enumerate(rows):
        out.paste(r, (0, i * tile))
    out.save(path, quality=90)
    return [first + f for f in picks]


def rows_of(points_path: str, names: list[str] | None, first: int, count: int, points_first: int | None,
            whole_face: bool = True) -> tuple[dict, int]:
    """A points file cut to the frames asked for, and the source frame of its row 0. With `whole_face` it must hold
    every name a face is drawn from; a tool that reads only what `face_points.py` reads asks for less."""
    points = fp.load(points_path, names)
    pf = points["first"] if points_first is None else points_first
    if pf is None:
        raise SystemExit(f"{points_path} does not say which source frame its first row is: give --points-first")
    off = first - pf
    if off < 0 or off + count > len(points["keypoints"]):
        raise SystemExit(f"{points_path} covers source frames {pf} to {pf + len(points['keypoints']) - 1}; frames {first} to {first + count - 1} were asked for")
    gone = missing(points["index"]) if whole_face else []
    if gone:
        raise SystemExit(f"{points_path} lacks {len(gone)} of the names a face is drawn from (the first: {', '.join(gone[:4])}): it is not the pose model's 308 points")
    return dict(points, keypoints=points["keypoints"][off:off + count], scores=points["scores"][off:off + count]), pf


def width_for(points: dict, width: float | None) -> tuple[float, float, str]:
    """(the width drawn, the shot's median face height, what set the width). The width is in whole steps of the drawing's grid."""
    look = h3_config.FACE_POINTS
    face = float(np.median([fp.face_height(p, points["index"]) for p in points["keypoints"]]))
    asked = float(width) if width else max(face / look["face_over_width"], look["least_px"])
    by = "--width" if width else (f"the shot's median face height ({face:.0f} px) over h3_config.FACE_POINTS['face_over_width']"
                                  + (", lifted to its 'least_px'" if face / look["face_over_width"] < look["least_px"] else ""))
    return max(1, int(round(asked * SUPER))) / SUPER, face, by


def run(mesh_path: str, points_path: str, first: int, count: int, out: Path, by: str, *, names: list[str] | None = None,
        mesh_first: int | None = None, points_first: int | None = None, width: float | None = None, dots: bool = True,
        leave_out: frozenset = frozenset(), bare: list[int] | None = None, bare_by: str = "", source: str = "", source_first: int = 0,
        iris: str = IRIS_FORMS[0], mouths: list | None = None, mouth_lines: bool = True, no_mouth: list[int] | None = None,
        class_mouth_frames: list[int] | None = None, say=print) -> dict:
    """Draw, write, read back, record. `mouths`: a class-map mouth a frame ((whole, hole) masks of the frame, or None), drawn in
    place of the points' mouth (`head_reference`, `--mouth auto` where the lip check fails and the class map holds the mouth);
    with `mouth_lines` False its lips are painted out and its inside filled under the points' own mouth lines, and no
    outline of its own is drawn. `no_mouth`: source frames on which no mouth is drawn at all, neither the points' lip rings nor a
    class-map mouth (`head_reference`: the lips do not lie on the mouth there, or the load's part holds no lips): the body model's
    own lips stay. `class_mouth_frames`: source frames on which, with `mouth_lines` False, the class map's mouth is drawn with its own
    outline in place of the points' lip rings (`head_reference`, a frame at a time: the lip lines miss there and the class map's mouth
    passes its gate). Returns the record; `passed` is False when the read-back or the look at the source fails."""
    if bare and not bare_by:
        raise SystemExit("--bare names frames a person looked at: say who with --bare-by")
    points, pf = rows_of(points_path, names, first, count, points_first)
    idx = points["index"]
    w, h, have = probe(mesh_path)
    skip = first - (first if mesh_first is None else mesh_first)
    if skip < 0 or skip + count > have:
        raise SystemExit(f"{mesh_path} has {have} frames from source frame {first - skip}; frames {first} to {first + count - 1} were asked for")
    if points["size"] and tuple(points["size"]) != (w, h):
        raise SystemExit(f"the points were made on a picture of {points['size']} and the body-model video is {(w, h)}: they are not on one canvas")
    mesh = np.stack(list(stream(mesh_path, (w, h), skip, count, pix="rgb24")))
    line = h3_config.POSE_READING["eye_ring_score_at_least"]
    if iris not in IRIS_FORMS:
        raise SystemExit(f"--iris is one of {IRIS_FORMS}; got {iris!r}")
    offs = [{} for _ in range(count)]
    drawings = [shapes(p, s, idx, line, leave_out, iris, offs[f]) for f, (p, s) in enumerate(zip(points["keypoints"], points["scores"]))]
    if not dots:
        drawings = [lines_only(d) for d in drawings]                # no dots is no dots: the points that belong to no line go too
    wide, face, wide_by = width_for(points, width)
    left = {f - first for f in (bare or []) if first <= f < first + count}
    fills = [discs(p, s, idx, line, leave_out) for p, s in zip(points["keypoints"], points["scores"])] if iris in IRIS_FORMS[2:] else None
    bare_mouth = sorted({f - first for f in (no_mouth or []) if first <= f < first + count})
    for f in bare_mouth:
        drawings[f] = [d for d in drawings[f] if not d[0].endswith("_lip_ring")]
    if mouths:
        if len(mouths) != count:
            raise SystemExit(f"{len(mouths)} class-map mouths for {count} frames")
        mouths = [None if f in bare_mouth else m for f, m in enumerate(mouths)]
        own = {f - first for f in (class_mouth_frames or []) if first <= f < first + count} - set(bare_mouth)
        for f in own:
            drawings[f] = [d for d in drawings[f] if not d[0].endswith("_lip_ring")]
        if mouth_lines or own:
            drawings = [d + (mouth_shapes(*mouths[f]) if mouths[f] is not None and (mouth_lines or f in own) else []) for f, d in enumerate(drawings)]
    pictures = draw(mesh, drawings, wide, left, dots, fills, contrast=iris == IRIS_FORMS[3], mouths=mouths)
    but = filled_px(fills, (w, h)) if iris == IRIS_FORMS[3] else None
    out.parent.mkdir(parents=True, exist_ok=True)
    marks.write_video(out, pictures)
    drawn = np.array([f not in left for f in range(count)])
    expected = np.array([white_expected(drawings[f], wide, dots) if drawn[f] else 0.0 for f in range(count)])
    back = read_back(out, mesh, drawings, expected, drawn, wide, but)
    white, lags = back["white"], back["lags"]
    src = marks.source_frames(source, (w, h), first, count, source_first) if source else None
    clock = marks.against_the_source(src, mesh, first) if src is not None else None
    passed = bool(back["passed"])
    named = [{n for n, _, _ in drawings[f]} | {n for n, _, _ in (fills[f] if fills else [])} for f in range(count)]
    kinds = ("iris",) if fills else RINGS
    shape = "disc" if fills else "ring"
    rings = {f"{side}_{kind}": [first + f for f in range(count) if f"{side}_{kind}_{shape}" not in named[f]]
             for side in SIDES for kind in kinds} if "eyes" not in leave_out else {}
    scores = {k: [round(float(v), 2) for v in np.percentile([ring_scores(s, idx)[k] for s in points["scores"]], [0, 50, 100])] for k in rings}
    fit = {str(k): (None if np.isnan(v) else round(v, 3)) for k, v in lags.items()}
    record = {
        "video": out.name, "made_by": by, "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frames": [first, first + count - 1], "size": [w, h],
        "made_from": {"body_model_video": str(mesh_path), "its_first_frame_is_source_frame": first - skip, "points": str(points_path),
                      "their_first_row_is_source_frame": pf},
        "line_width_px": wide, "width_set_by": wide_by, "a_dot_at_every_point": dots, "face_height_px_median": round(face, 1),
        "points_drawn_alone": sorted({n for d in drawings for n, pts, _ in d if len(pts) == 1}),
        "parts_left_out": sorted(leave_out),
        "lines_px_a_frame_median": round(float(np.median([length(d) for d in drawings])), 1),
        "eye_rings": {"score_at_least": line, "that_line_is": "a starting value from one shot: under it a ring's points lay far off the face",
                      "frames_a_ring_is_left_out_on": rings, "score_a_frame_least_median_most": scores,
                      "drawn": iris, "drawn_is": {IRIS_FORMS[0]: "only the stretches of each ring between the lids", IRIS_FORMS[1]: "each ring whole",
                                                  IRIS_FORMS[2]: "each iris a dark disc inside the lash ring, and no ring",
                                                  IRIS_FORMS[3]: "inside each lash ring the sclera solid white and the iris solid black, drawn before "
                                                                 "the lines, whose black edge stays round them; the white of the sclera is left out of the read-back"}[iris],
                      "share_of_the_ring_outside_the_lids_least_median_most": {
                          k: ([round(float(v), 2) for v in np.percentile(vals, [0, 50, 100])] if vals else None)
                          for k in rings for vals in [[o[k] for o in offs if k in o]]}},
        "frames_left_bare": sorted(first + f for f in left), "left_bare_by": bare_by,
        "frames_with_no_mouth": [first + f for f in bare_mouth],
        "frames_with_the_class_maps_own_mouth": sorted(first + f for f in ({f - first for f in (class_mouth_frames or []) if first <= f < first + count}
                                                                          - set(bare_mouth))),
        "not_known": ["whether a head is turned away on any frame: the model places and scores face points on the back of a head",
                      "whether the lines sit on her face, and whether they can be told apart at this size: open the sheet",
                      "whether the lids are drawn shut where hers are: on a laughing squint the pose model's lid points do not close, and the lid "
                      "outline stays an open almond (the lead on 3247, 3261 and 3276, 2026-10-14, on the lead's look at the sheet)",
                      "what the body model's own eyes do under the drawing: its sculpted eyes are open on every frame, whatever is drawn on them"],
        "read_back": {"white_px_a_frame": {"least": int(white.min()), "median": int(np.median(white)), "most": int(white.max())},
                      "white_px_expected_a_frame": {"least": int(expected[drawn].min()) if drawn.any() else 0,
                                                    "median": int(np.median(expected[drawn])) if drawn.any() else 0,
                                                    "most": int(expected[drawn].max()) if drawn.any() else 0},
                      "white_from_the_lines_px_mean_by_frames_of_slip": fit,
                      "which_is": "how far the white found on a frame lies from the lines of that frame's points (0), of the frame before's (1: the file is a frame late) and of the frame after's (-1); smaller is nearer",
                      "a_neighbour_fits_better": back["slipped"], "no_neighbour_can_be_told_from_the_frame": back["cannot_tell"],
                      "white_off_the_lines": back["off_the_lines"],
                      "white_on_a_bare_frame": back["on_bare"], "drawn_frames_with_no_white": [first + f for f in back["drawn_frames_with_no_white"]],
                      "lines_too_thin_to_leave_much_white": {"frames": [first + f for f in back["too_thin"]],
                                                             "which_are": f"frames whose lines should leave under {THIN_PX:g} pixels of white"},
                      "passed": back["passed"],
                      "what_it_cannot_catch": "points or a body-model video given on the wrong clock: the drawing and the lines it is read against come from the same rows"},
        "body_model_against_the_source": "not looked at: no --source" if clock is None else clock,
    }
    if source:
        sheet_path = out.with_name(out.stem + "_sheet.jpg")
        record["sheet"] = {"file": sheet_path.name, "frames": sheet(sheet_path, src, first, pictures, points, count), "looked_at_by": ""}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    say(f"face {face:.0f} px tall at the median; lines {wide:g} px wide ({wide_by}), {'a dot at every point' if dots else 'no dots'}; "
        f"{record['lines_px_a_frame_median']:.0f} px of line a frame at the median" + (f"; left out: {', '.join(sorted(leave_out))}" if leave_out else ""))
    for key, gone in rings.items():
        lo, mid, hi = scores[key]
        say(f"{key} ring: left out on {len(gone)} of {count} frames (score a frame {lo:.2f} to {hi:.2f}, median {mid:.2f}, against {line:g})")
    if back["too_thin"]:
        say(f"on {len(back['too_thin'])} of {count} frames the lines should leave under {THIN_PX:g} pixels of white")
    say(f"frames left bare: {len(left)}" + (f", by {bare_by}" if left else " (none named; a head turned away is not found by this tool)"))
    say("read back: white a frame %d to %d (expected %d to %d); mean distance of the white from its own frame's lines %s px, from the frame before's %s, from the frame after's %s; white on a bare frame %d" % (
        white.min(), white.max(), record["read_back"]["white_px_expected_a_frame"]["least"], record["read_back"]["white_px_expected_a_frame"]["most"],
        *(("n/a" if np.isnan(lags[k]) else f"{lags[k]:.3f}") for k in (0, 1, -1)), back["on_bare"]))
    if clock is not None:
        say(marks.against_the_source_line(clock))
    say(("PASSED" if passed else "FAILED") + ": the video " + ("holds" if back["passed"] else "does NOT hold") + " the lines its points describe on the frames they describe"
        + ("; the face moves too little between frames here for a slip of a frame to be told" if back["cannot_tell"] and passed else ""))
    say(f"wrote {out} and {out.with_suffix('.json').name}" + (f" and {record['sheet']['file']}: look at it before the video is used" if source else "; no sheet (give --source)"))
    record["passed"] = passed
    return record


def verify(video: str, mesh_path: str, points_path: str, first: int, count: int, *, names: list[str] | None = None,
           mesh_first: int | None = None, points_first: int | None = None, video_first: int | None = None, width: float | None = None,
           leave_out: frozenset = frozenset(), iris: str = IRIS_FORMS[0], say=print) -> bool:
    """A drawn video somebody made, read against a points file on the READER's account of each one's first frame."""
    points, _ = rows_of(points_path, names, first, count, points_first)
    w, h, _ = probe(mesh_path)
    mesh = np.stack(list(stream(mesh_path, (w, h), first - (first if mesh_first is None else mesh_first), count, pix="rgb24")))
    if len(mesh) != count:
        raise SystemExit(f"asked for {count} frames; the body-model video gave {len(mesh)}")
    fills = ([discs(p, s, points["index"], h3_config.POSE_READING["eye_ring_score_at_least"], leave_out) for p, s in zip(points["keypoints"], points["scores"])]
             if iris == IRIS_FORMS[3] else None)
    try:
        white = white_at(video, mesh, first - (first if video_first is None else video_first), but=filled_px(fills, (w, h)))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    wide = width_for(points, width)[0]
    line = h3_config.POSE_READING["eye_ring_score_at_least"]
    segs = [segments(shapes(p, s, points["index"], line, leave_out, iris)) for p, s in zip(points["keypoints"], points["scores"])]
    count_white = np.array([len(v) for v in white])
    drawn = count_white > 0
    lags = fit_by_lag(white, segs, drawn)
    slipped, cannot_tell = reading(lags)
    off = bool(not np.isnan(lags[0]) and lags[0] > wide)
    bare = [first + int(f) for f in np.nonzero(~drawn)[0]]
    ok = bool(drawn.any() and not slipped and not off)
    say(f"frames {first} to {first + count - 1}: white a frame {int(count_white.min())} to {int(count_white.max())}; frames with no lines: {len(bare)} {bare[:40]}")
    say("mean distance of the white from the lines of the points, px: the frame's own %s, the frame before's %s, the frame after's %s" % tuple(
        ("n/a" if np.isnan(lags[k]) else f"{lags[k]:.3f}") for k in (0, 1, -1)))
    say("PASSED: no neighbouring frame's points fit the lines better" + ("; the face moves too little here for a slip to be told" if cannot_tell else "")
        if ok else "FAILED: " + ("the video has no lines" if not drawn.any() else
                                 "a neighbouring frame's points fit the lines better than the frame's own" if slipped else
                                 f"the white lies further from these points' lines than a line and its dots reach ({wide:g} px)"))
    return ok


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        a = argparse.ArgumentParser(prog="draw_face_points.py verify", description="Read a drawn video against a points file.")
        a.add_argument("--video", required=True); a.add_argument("--mesh", required=True); a.add_argument("--points", required=True)
        a.add_argument("--first", type=int, required=True); a.add_argument("--count", type=int, required=True)
        a.add_argument("--names"); a.add_argument("--mesh-first", type=int); a.add_argument("--points-first", type=int); a.add_argument("--video-first", type=int)
        a.add_argument("--width", type=float, help="the width the lines were drawn at, from the video's record; the size rule's when not given")
        a.add_argument("--leave-out", default="", help="the parts the video's record says were left out")
        a.add_argument("--iris", default=IRIS_FORMS[0], choices=IRIS_FORMS, help="how the video's record says its eye rings were drawn")
        v = a.parse_args(sys.argv[2:])
        names = json.loads(Path(v.names).read_text()) if v.names else None
        return 0 if verify(v.video, v.mesh, v.points, v.first, v.count, names=names, mesh_first=v.mesh_first, points_first=v.points_first,
                           video_first=v.video_first, width=v.width, leave_out=parts_arg(v.leave_out), iris=v.iris) else 1
    a = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    a.add_argument("--mesh", required=True, help="the body-model video, at the canvas's size")
    a.add_argument("--points", required=True, help="the original's points along the shot (bench/pose_along_shot.py)")
    a.add_argument("--first", type=int, required=True, help="the shot's first source frame")
    a.add_argument("--count", type=int, required=True)
    a.add_argument("--out", required=True, help="the motion video to write; its record and sheet go beside it")
    a.add_argument("--by", required=True, help="who is making it")
    a.add_argument("--names", help="a json list of the points' names, for a points file made before they were kept")
    a.add_argument("--mesh-first", type=int, help="the source frame of the body-model video's frame 0; the shot's first when left out")
    a.add_argument("--points-first", type=int, help="the source frame of the points file's row 0; the file's own when it says")
    a.add_argument("--width", type=float, help="set the lines' width in pixels by hand in place of the size rule")
    a.add_argument("--no-dots", action="store_true", help="lines only: no dot at each point of a line, and none of the points that belong to no line")
    a.add_argument("--leave-out", default="", help=f"parts not to draw, as a person saw their points are off her face: any of {', '.join(PARTS)}")
    a.add_argument("--bare", default="", help="source frames to leave with nothing drawn, as a person saw them: 1045-1047,1060")
    a.add_argument("--bare-by", default="", help="who looked")
    a.add_argument("--iris", default=IRIS_FORMS[0], choices=IRIS_FORMS,
                   help="an iris or pupil ring drawn only where it lies between the lids (the default), whole (the form before 2026-10-14), the iris filled: a dark disc inside the lids and no ring, or contrast: the sclera solid white and the iris solid black inside the lids (until a render decides)")
    a.add_argument("--source", default="", help="the original video: writes a sheet to look at")
    a.add_argument("--source-first", type=int, default=0, help="the source frame of that video's frame 0, for one cut from the original; 0 when left out")
    v = a.parse_args()
    names = json.loads(Path(v.names).read_text()) if v.names else None
    rec = run(v.mesh, v.points, v.first, v.count, Path(v.out), v.by, names=names, mesh_first=v.mesh_first, points_first=v.points_first,
              width=v.width, dots=not v.no_dots, leave_out=parts_arg(v.leave_out), bare=marks.frames_arg(v.bare), bare_by=v.bare_by, source=v.source,
              source_first=v.source_first, iris=v.iris)
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
