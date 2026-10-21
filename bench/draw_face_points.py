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

A FRAME'S EYES ARE DRAWN SHUT BY THE RULE under `--iris contrast` with the closed lip line (the chain's default form; the owner's
direction of 2026-10-21): where the two eyes' mean iris ring outside the lids (or that eye's share when only one scores) is
`DOWNCAST_RING_OUTSIDE_MIN` or more, both eyes are drawn as one line along the upper lash line, the "lids" of `EYES_SHUT_FORMS`, with no
sclera, iris or lid lines (`eye_reading`). A person's word (`eyes_shut`) wins on its frames. The record (`eye_rings.closed_by_the_rule`)
gives every frame in spans of one form, with the eyes that met the line and each eye's share.

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
from _lib import REPO, check_upstream_mtime, writer_stamp  # noqa: E402
from _lib import face_points as fp  # noqa: E402
from _lib import pick_frames as picker  # noqa: E402
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
#: HER EYES SHUT ON A SPAN, on a person's word (`--eyes-shut-on`, the lead's ruling of 2026-10-15 on 2823b, whose eyes never closed in
#: the render): the pose model's lid points stop at a slit on a closed eye (2823's reference, G3's six closed runs: the drawn opening's
#: median 0.0105-0.0170 of a face height against 0.0200 open), and nothing else measured tells closed on her (the part model has no eye
#: class; the dark inside the lids is her makeup; the model's own scores are high either way). On those frames the eye's lash ring,
#: lid and crease lines, iris and pupil and their fill are not drawn; the brows stay. "none": nothing more, the body model's own eyes
#: show (whose sculpted eyes are open on every frame); "lids": each eye drawn shut as one line along its upper lash line. Both for one
#: test (2945-2981). "lids" is also the closed eye drawn BY THE RULE (`eye_reading`, 2026-10-21) under the chain's default form; a
#: person's word wins over the rule on its frames, in its own form.
EYES_SHUT_FORMS = ("none", "lids")
EYE_PARTS = ("_lash_ring", "_upper_eyelid_line", "_lower_eyelid_line", "_upper_crease_line", "_iris_ring", "_pupil_ring", "_shut_lid")
#: Points a ring is sampled at to find where it lies between the lids. Reasoned: a ring is eight points round a
#: centre; 64 along it puts a sample about every 6 degrees, finer than a line is wide on any face drawn so far.
RING_SAMPLES = 64
#: What `--leave-out` can name: every shape drawn belongs to one.
PARTS = ("eyes", "brows", "nose", "mouth", "dots")
#: Under `--closed-lip-line`, a lip-line mouth whose upper and lower inner-lip arcs lie at or under this median
#: distance (pixels on the reference canvas), or whose class-map mouth has no hole (`not hole.any()`), is drawn as
#: one closed parting line (`inner_lip_line`) rather than an open ring. Measured: on `second_hfp_0278d` (`278-293`),
#: the pose model places the inner lip arcs 0.64 to 2.79 px apart on a closed mouth, leaving a 10-19 px dark slit
#: (`luma < 40`) between the two 1.0 px black edge strokes on `278-281` when drawn as a ring (2026-10-18).
INNER_GAP_CLOSED_PX = 3.0
#: Silhouette erosion in pixels (`M20`, 2026-10-18) for the inner head mask that clips the white sclera of `--iris contrast`,
#: keeping a dark outline rim between the white of the eye and the black background at the head's edge.
HEAD_SILHOUETTE_ERODE_PX = 2
#: Under `--darken-head` (`M12`, `M14` item 2 & `M15`, 2026-10-18), the grey body model is darkened by this multiplicative scale
#: (about a third darker, a starting value) before drawing the white lines and fills, keeping its 3D shading gradients intact
#: while raising line contrast.
HEAD_DARKEN_SCALE = 0.67
#: Minimum chin clearance below a drawn mouth (`fit_mouth_to_head`, `M8`, `M17` & `M21`, 2026-10-18), as a fraction of the vertical
#: distance from the eye line to the grey head's chin (`MOUTH_CHIN_CLEAR_FRAC`) and in pixels (`MOUTH_CHIN_CLEAR_MIN_PX`). Measured
#: on `third_1306_1418` (`1340-1360`): at `1356` the pose lip ring (`y=403.9`) and class-map mouth (`y=414.0`) reached or hung past
#: the grey head's chin (`y=411`), where the render grew a grey goatee; `max(8.0, 0.14 * (chin_y - eye_y))` keeps `20.9-70.4 px` of
#: clean grey chin below the lower lip across all 21 frames (`22.5 px` at `1356`) with `0/21` mouths cut by the head silhouette.
MOUTH_CHIN_CLEAR_FRAC = 0.14
MOUTH_CHIN_CLEAR_MIN_PX = 8.0
#: Share of the iris ring outside the eyelids (`between_the_lids`) at or above which an eye is DOWNCAST (`eye_reading`): under
#: `--iris contrast` with `--closed-lip-line`, the frame's eyes are then drawn shut where the MEAN of the scored eyes' shares is at or
#: above this line (that eye's share when only one iris ring scores). Measured on `second_0484_0515_c` (`0.69–0.78` on all 32 downcast
#: frames; two-eye mean `0.703–0.766` on `second_0484_0515_d`), `second_0415_0437_c` (`0.72–0.73` on downcast frame `437`, mean `0.727`,
#: vs `0.31–0.61` on open frames `415–436`), and `second_1039_1060_c` (`0.64–0.77` on downcast frames `1053–1060`, mean `0.664–0.719` on
#: `second_1039_1060_d`, vs `0.20–0.50` on open frames `1039–1051`): at `>= 0.65`, the black edge of the upper and lower lash lines covers
#: `100%` of the clipped black iris disc (`0 px` visible black iris) while the `255,255,255` white sclera under the lash ring turns the
#: narrowed eye into a bright white slit that H3 reads as wide-open camera-staring eyes (`2026-10-20`). From 2026-10-20 such an eye was
#: drawn with no sclera and its iris as an arc between the lids, inside the lash ring and the lid lines; the owner judged the renders
#: (second_hfp_0415g at 437, second_hfp_1039d at 1053-1060) to read that as an open eye looking ahead, and on a small face the white lines
#: touched into a white almond. Since 2026-10-21 both eyes are drawn shut where the two eyes' mean share is at or above this line (the
#: owner: closed and gazing down look the same at this size; the text stays as it is). Decided on the two-eye mean rather than the two-eye
#: max (`0b8ab179`, dropped the same day): across eight references (`second_0415_0437_d` 1/1 on 437, `second_1039_1060_d` 8/8 on 1053-1060,
#: `second_0484_0515_d` 32/32 on 484-515, `second_4602_4646_b` 0/0, `second_4726_4773_b` 7 max vs 2 mean on 4760 and 4766,
#: `second_4479_4509` 6 max vs 0 mean, `second_5768_5791_c` 2 max vs 0 mean, `second_5533_5564` 15 max vs 7 mean on 5551-5552, 5559-5560
#: and 5562-5564), the mean keeps every downcast frame on the three eye tests while dropping the one- or two-frame flicker where a single
#: foreshortened eye on a three-quarter face crosses 0.65 alone. NOT a closed eye by the lash gap: an eye whose lash-line gap was under
#: `h3_config.POSE_READING["eyes_shut_under"]` of the face's height was tried as a second rule and dropped the same day: on the references
#: of second_hfp_4602c (whose eyes the owner called correct, data/fun/owner_answers.jsonl:1373) and second_hfp_4726c (its face called right,
#: :1371) it met 40 of 45 and 30 of 48 frames on which no eye was downcast (measured from their saved points through `eye_reading` and
#: `_lib/face_points.py::lid_gap_px`).
DOWNCAST_RING_OUTSIDE_MIN = 0.65


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


def closes_downcast_eyes(iris: str, closed_lip_line: bool, leave_out: frozenset = frozenset()) -> bool:
    """Whether the closed-eye rule (`eye_reading`) acts: only under `--iris contrast` with `--closed-lip-line` and the eyes drawn."""
    return "eyes" not in leave_out and iris == IRIS_FORMS[3] and bool(closed_lip_line)


def eye_reading(p: np.ndarray, s: np.ndarray, index: dict, score_line: float) -> dict:
    """One frame's two eyes against the closed eye (2026-10-21): for each eye (`eyes[side]`) the share of its iris ring outside its lids
    (`ring_outside`, None where the ring scores under `score_line` and is not drawn) and whether it is `downcast` (the share at or over
    `DOWNCAST_RING_OUTSIDE_MIN`); `mean_outside`, the mean of the scored eyes' shares (that eye's share when only one scores, None when
    neither does); and `shut`, whether the frame's eyes are drawn shut under `--iris contrast` with the closed lip line (`mean_outside` at
    or over `DOWNCAST_RING_OUTSIDE_MIN`). BOTH EYES OR NEITHER on a frame: where the two eyes' mean is downcast both are shut, since one eye
    shut and the other open (a wink) is not what the source showed on the loads this was made for (the lead, 2026-10-21, on the outside
    reviewer's G13 and G21)."""
    scores = ring_scores(s, index)
    eyes = {}
    for side, ring in zip(SIDES, fp.lid_rings(p, index)):
        share = None
        if scores[f"{side}_iris"] >= score_line:
            share = between_the_lids(_round(p, _ids(index, f"{side}_border_of_iris"), p[index[f"{side}_center_of_iris"]]), ring)[2]
        eyes[side] = {"ring_outside": share, "downcast": share is not None and share >= DOWNCAST_RING_OUTSIDE_MIN}
    scored = [e["ring_outside"] for e in eyes.values() if e["ring_outside"] is not None]
    mean_outside = float(sum(scored) / len(scored)) if scored else None
    return {"eyes": eyes, "mean_outside": mean_outside, "shut": mean_outside is not None and mean_outside >= DOWNCAST_RING_OUTSIDE_MIN}


def shapes(p: np.ndarray, s: np.ndarray, index: dict, score_line: float, leave_out: frozenset = frozenset(),
           iris: str = IRIS_FORMS[0], outside: dict | None = None,
           closed_lip_line: bool = False) -> list[tuple[str, np.ndarray, bool]]:
    """One frame's drawing as (name, points in order, whether it closes on itself). A dot alone is one point.
    `iris`: a ring drawn between the lids or whole (`IRIS_FORMS`); `outside`, when given, gets each scored ring's
    share outside the lids, by its name. Under `--iris contrast` with `closed_lip_line` (the chain's default form), on a frame
    `eye_reading` calls shut each eye is one line along its upper lash line (`<side>_shut_lid`, the "lids" of `EYES_SHUT_FORMS`) and
    nothing else of it is drawn; the brows stay."""
    out: list[tuple[str, np.ndarray, bool]] = []
    scores = ring_scores(s, index)
    shut = closes_downcast_eyes(iris, closed_lip_line, leave_out) and eye_reading(p, s, index, score_line)["shut"]
    for side, ring in zip(SIDES, fp.lid_rings(p, index)):
        if "eyes" not in leave_out:
            if shut:
                out.append((f"{side}_shut_lid", fp._line(p, index, side, "upper"), False))
            else:
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
                    if shut or iris == IRIS_FORMS[2] or (iris == IRIS_FORMS[3] and not (closed_lip_line and kind == "iris" and off >= DOWNCAST_RING_OUTSIDE_MIN)):
                        continue                                       # shut, or filled or contrast: the iris is a disc (`discs`), unless downcast on an open frame
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


def discs(p: np.ndarray, s: np.ndarray, index: dict, score_line: float, leave_out: frozenset = frozenset(),
          closed_lip_line: bool = False, iris: str = IRIS_FORMS[3]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """One frame's irises for `--iris filled`, as (name, the iris ring sampled at `RING_SAMPLES` points, that eye's lash ring):
    `draw` fills each in the edge colour where it lies inside its lash ring. An iris under the score line is left out, as its ring is.
    Under `--iris contrast` with `closed_lip_line`, none on a frame `eye_reading` calls shut: no iris and no sclera (`shapes` draws its
    eyes shut); on any other frame with `closed_lip_line`, a downcast eye's own disc is dropped per eye as before."""
    if "eyes" in leave_out or (closes_downcast_eyes(iris, closed_lip_line, leave_out) and eye_reading(p, s, index, score_line)["shut"]):
        return []
    scores, out = ring_scores(s, index), []
    for side, ring in zip(SIDES, fp.lid_rings(p, index)):
        if scores[f"{side}_iris"] >= score_line:
            whole = _round(p, _ids(index, f"{side}_border_of_iris"), p[index[f"{side}_center_of_iris"]])
            if closed_lip_line and between_the_lids(whole, ring)[2] >= DOWNCAST_RING_OUTSIDE_MIN:
                continue
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


def white_expected(drawing: list[tuple[str, np.ndarray, bool]], width: float, dots: bool, edge_px: float = EDGE_PX) -> float:
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

    for name, pts, closed in drawing:
        ep = 0.0 if name in ("inner_lip_line", "mouth_closed") else edge_px
        for piece in segments([(name, pts, closed)]):
            mark(piece, max(width - ep, 0.0) / 2)
        if dots:
            for point in pts:
                mark(np.stack([point, point]), max(width - ep / 2, 0.0))
    return float(inside.sum())


def head_silhouette(frame: np.ndarray, erode_px: int = HEAD_SILHOUETTE_ERODE_PX,
                    p: np.ndarray | None = None, index: dict | None = None) -> np.ndarray:
    """The lit head model silhouette above the chin crease (`frame.max(-1) > 16` cut at `find_mesh_chin_y` when `p` and `index` are
    given), eroded by `erode_px` (`M7`, `M20`, `M24`) so no drawn feature sits on the neck/shoulders/arms, on the outer silhouette
    rim, or in black background."""
    import cv2
    lit = (frame.max(-1) > 16).astype(np.uint8)
    if p is not None and index is not None and "tip_of_chin" in index and "bottom_center_of_nose" in index:
        chin_y = int(np.ceil(find_mesh_chin_y(frame, p, index)))
        if 0 <= chin_y + 1 < lit.shape[0]:
            lit[chin_y + 1:, :] = 0
    if erode_px > 0:
        k = 2 * erode_px + 1
        lit = cv2.erode(lit, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return lit > 0


def clip_shape_to_mask(name: str, pts: np.ndarray, closed: bool, mask: np.ndarray) -> list[tuple[str, np.ndarray, bool]]:
    """Clip a shape `(name, pts, closed)` to the boolean `mask` (`M7`): if wholly inside, returns `[(name, pts, closed)]`;
    if wholly outside, returns `[]`; if partly outside, returns the contiguous stretches inside `mask` as open polylines."""
    h, w = mask.shape
    if len(pts) == 0:
        return []
    if len(pts) == 1:
        x, y = int(np.clip(round(pts[0, 0]), 0, w - 1)), int(np.clip(round(pts[0, 1]), 0, h - 1))
        return [(name, pts, False)] if mask[y, x] else []
    seg_pts = np.vstack([pts, pts[:1]]) if closed else pts
    diffs = np.diff(seg_pts, axis=0)
    lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    total = float(lengths.sum())
    if total < 1e-6:
        x, y = int(np.clip(round(pts[0, 0]), 0, w - 1)), int(np.clip(round(pts[0, 1]), 0, h - 1))
        return [(name, pts, closed)] if mask[y, x] else []
    n_samp = max(int(np.ceil(total * 2.0)), 16)
    cum = np.r_[0.0, np.cumsum(lengths)]
    s = np.linspace(0.0, total, n_samp, endpoint=not closed)
    idx_s = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(diffs) - 1)
    frac = (s - cum[idx_s]) / np.maximum(lengths[idx_s], 1e-9)
    dense = seg_pts[idx_s] + frac[:, None] * diffs[idx_s]
    xs = np.clip(np.round(dense[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(dense[:, 1]).astype(int), 0, h - 1)
    keep = mask[ys, xs]
    if keep.all():
        return [(name, pts, closed)]
    if not keep.any():
        return []
    order = np.r_[int(np.argmin(keep)):n_samp, 0:int(np.argmin(keep))] if closed else np.arange(n_samp)
    runs, run = [], []
    for j in order:
        if keep[j]:
            run.append(dense[j])
        elif run:
            if len(run) > 1:
                runs.append((name, np.array(run), False))
            run = []
    if len(run) > 1:
        runs.append((name, np.array(run), False))
    return runs


def find_mesh_chin_y(frame: np.ndarray, p: np.ndarray, index: dict) -> float:
    """Bottom y of the grey head model's chin (`M8`), found in the mouth-to-chin horizontal column (`[mx0..mx1]`) below the outer
    lips where the head silhouette ends in black background (`~lit[y, x]`); when the neck continues below (`first_black` empty),
    returns `tip_of_chin` without scanning full-row widths across ears, shoulders, or hands."""
    h, w = frame.shape[:2]
    lit = frame.max(-1) > 16
    eye_y = 0.5 * (
        p[[index["l_inner_end_of_upper_lash_line"], index["l_outer_end_of_upper_lash_line"]], 1].mean()
        + p[[index["r_inner_end_of_upper_lash_line"], index["r_outer_end_of_upper_lash_line"]], 1].mean()
    )
    nose_y = float(p[index["bottom_center_of_nose"], 1])
    chin_kp_y = float(p[index["tip_of_chin"], 1])
    y_start = max(0, int(round(nose_y)))
    y_end = min(h - 1, int(round(chin_kp_y + 0.5 * max(chin_kp_y - eye_y, 20.0))))
    mx0 = int(round(min(p[index["r_outer_corner_of_mouth"], 0], p[index["l_outer_corner_of_mouth"], 0], p[index["tip_of_chin"], 0])))
    mx1 = int(round(max(p[index["r_outer_corner_of_mouth"], 0], p[index["l_outer_corner_of_mouth"], 0], p[index["tip_of_chin"], 0])))
    first_black = []
    for x in range(max(0, mx0), min(w, mx1 + 1)):
        if not lit[y_start, x]:
            continue
        b = np.nonzero(~lit[y_start:y_end + 1, x])[0]
        if len(b):
            first_black.append(y_start + int(b[0]))
    if len(first_black) >= max(3, int(0.25 * max(mx1 - mx0 + 1, 1))):
        return float(np.median(first_black))
    return float(min(y_end, int(round(chin_kp_y))))


def fit_mouth_to_head(frame: np.ndarray, p: np.ndarray, index: dict, m_shape: tuple[str, np.ndarray, bool],
                      m_masks: tuple[np.ndarray, np.ndarray]) -> tuple[tuple[str, np.ndarray, bool], tuple[np.ndarray, np.ndarray]]:
    """Keep a drawn mouth inside the grey head model's mouth-to-chin space (`M8`, `M21`):
    1. When the grey head's chin (`find_mesh_chin_y`) is higher than the pose chin (`tip_of_chin`, e.g. when the person's jaw drops
       open on a neutral-jaw 3D mesh over black background), map vertical coordinates below `bottom_center_of_nose` proportionally
       onto the grey head's nose-to-chin height and keep at least `max(MOUTH_CHIN_CLEAR_MIN_PX, 0.22 * (chin_y - nose_y))` of chin room.
    2. Shift horizontal coordinates inside the eroded head silhouette across the mouth's scanlines so `head_silhouette` never nips a corner."""
    import cv2
    h, w = frame.shape[:2]
    name, pts, closed = m_shape
    whole, hole = m_masks
    if len(pts) == 0:
        return m_shape, m_masks
    nose_y = float(p[index["bottom_center_of_nose"], 1])
    chin_kp_y = float(p[index["tip_of_chin"], 1])
    chin_y = find_mesh_chin_y(frame, p, index)
    scale_n2c = min(1.0, max(chin_y - nose_y, 12.0) / max(chin_kp_y - nose_y, 12.0))
    new_pts = pts.copy()
    if scale_n2c < 0.99:
        new_pts[:, 1] = nose_y + (new_pts[:, 1] - nose_y) * scale_n2c
    chin_clear = max(MOUTH_CHIN_CLEAR_MIN_PX, 0.22 * max(chin_y - nose_y, 20.0))
    max_bottom_y = chin_y - chin_clear
    min_top_y = nose_y + 3.0
    if max_bottom_y <= min_top_y + 2.0:
        max_bottom_y = min_top_y + 2.0
    y0, y1 = float(new_pts[:, 1].min()), float(new_pts[:, 1].max())
    mh = max(y1 - y0, 0.0)
    avail_h = max(max_bottom_y - min_top_y, 2.0)
    if y1 > max_bottom_y:
        scale_y = min(1.0, (avail_h * 0.85) / max(mh, 1e-6)) if mh > avail_h * 0.85 else 1.0
        if scale_y < 1.0:
            new_pts[:, 1] = y0 + (new_pts[:, 1] - y0) * scale_y
        dy = max_bottom_y - float(new_pts[:, 1].max())
        new_pts[:, 1] += dy
        if float(new_pts[:, 1].min()) < min_top_y:
            new_pts[:, 1] += min_top_y - float(new_pts[:, 1].min())
    sil_inner = head_silhouette(frame, erode_px=HEAD_SILHOUETTE_ERODE_PX + 1, p=p, index=index)
    over_l, over_r = 0.0, 0.0
    for x_pt, y_pt in new_pts:
        yi = int(np.clip(round(y_pt), 0, h - 1))
        xs = np.nonzero(sil_inner[yi])[0]
        if len(xs) >= 4:
            over_l = max(over_l, float(xs[0]) - float(x_pt))
            over_r = max(over_r, float(x_pt) - float(xs[-1]))
    if over_r > 0.0 and over_l == 0.0:
        new_pts[:, 0] -= over_r
    elif over_l > 0.0 and over_r == 0.0:
        new_pts[:, 0] += over_l
    for k in range(len(new_pts)):
        yi = int(np.clip(round(new_pts[k, 1]), 0, h - 1))
        xs = np.nonzero(sil_inner[yi])[0]
        if len(xs) >= 4:
            new_pts[k, 0] = np.clip(new_pts[k, 0], float(xs[0]), float(xs[-1]))
    new_hole = np.zeros((h, w), bool)
    if closed and hole.any():
        hm = np.zeros((h, w), np.uint8)
        cv2.fillPoly(hm, [np.round(new_pts).astype(np.int32)], 1)
        new_hole = (hm > 0) & head_silhouette(frame, erode_px=HEAD_SILHOUETTE_ERODE_PX, p=p, index=index)
    wm = np.zeros((h, w), np.uint8)
    if closed:
        cv2.fillPoly(wm, [np.round(new_pts).astype(np.int32)], 1)
    else:
        cv2.polylines(wm, [np.round(new_pts).astype(np.int32)], False, 1, thickness=6)
    new_whole = cv2.dilate(wm, np.ones((7, 7), np.uint8)) > 0
    return (name, new_pts, closed), (new_whole, new_hole)


def smooth_mouth_area(frame: np.ndarray, whole: np.ndarray) -> np.ndarray:
    """Flatten the body model's sculpted lips, mouth corners, lower-lip highlight, and labiomental shadow smoothly into the surrounding
    grey face (`--closed-lip-line`, Test B): an asymmetric elliptical dilation of `whole` (covering the 3D mesh's lower lip and chin groove
    8-12 px below the source mouth), Navier-Stokes inpainted and Gaussian-smoothed from the upper/cheek skin grey, blended with an 8 px
    distance-transform feather inside the body model (`frame.max(-1) > 16`)."""
    import cv2
    ys, xs = np.nonzero(whole)
    if not len(ys):
        return frame
    mh = int(ys.max() - ys.min() + 1)
    mw = int(xs.max() - xs.min() + 1)
    pad_top = max(6, int(round(0.25 * mh)))
    pad_bot = max(26, int(round(1.25 * mh)))
    pad_x = max(20, int(round(0.28 * mw)))
    dy, dx = np.ogrid[-pad_bot:pad_bot + 1, -pad_x:pad_x + 1]
    off_y = -dy
    ry = np.where(off_y < 0, float(max(pad_top, 1)), float(max(pad_bot, 1)))
    kern = (((dx / float(max(pad_x, 1))) ** 2 + (off_y / ry) ** 2) <= 1.0).astype(np.uint8)
    erase_raw = cv2.dilate(whole.astype(np.uint8), kern) > 0
    on_mesh_raw = frame.max(-1) > 16
    on_mesh = cv2.morphologyEx(on_mesh_raw.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0
    erase = erase_raw & on_mesh
    if not erase.any():
        return frame
    ring = (cv2.dilate(erase_raw.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~erase_raw & on_mesh_raw
    upper_ring = ring.copy()
    upper_ring[int(ys.max()):, :] = False
    sample = upper_ring if upper_ring.any() else ring
    grey = np.median(frame[sample], axis=0).astype(np.uint8) if sample.any() else np.array([128, 128, 128], np.uint8)
    work = frame.copy()
    work[~on_mesh_raw] = grey
    low_shadow = (cv2.dilate(erase_raw.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & (work.mean(-1) < float(grey.mean()) - 12.0)
    work[low_shadow] = grey
    inpainted = cv2.inpaint(work, erase.astype(np.uint8), 7, cv2.INPAINT_NS)
    smoothed = cv2.GaussianBlur(inpainted, (31, 31), 9.0)
    dist = cv2.distanceTransform(erase_raw.astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(dist / 8.0, 0.0, 1.0)[..., None]
    out = frame.copy()
    blended = np.round(alpha * smoothed + (1.0 - alpha) * frame).astype(np.uint8)
    out[erase] = blended[erase]
    return out


def inner_lip_centerline(p: np.ndarray, index: dict) -> tuple[np.ndarray, float]:
    """The centerline between the upper and lower inner-lip arcs from corner to corner, and their median separation in pixels."""
    a, b = p[index["r_inner_corner_of_mouth"]], p[index["l_inner_corner_of_mouth"]]
    upper = _along(p, _ids(index, tail="_of_upper_inner_lip"), a, b)
    lower = _along(p, _ids(index, tail="_of_lower_inner_lip"), a, b)
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return np.vstack([a, b]), 0.0
    u_pts = np.vstack([a[None], upper, b[None]])
    l_pts = np.vstack([a[None], lower, b[None]])
    tu = np.clip(((u_pts - a) @ ab) / denom, 0.0, 1.0)
    tl = np.clip(((l_pts - a) @ ab) / denom, 0.0, 1.0)
    iu, il = np.argsort(tu, kind="stable"), np.argsort(tl, kind="stable")
    tu, u_pts = tu[iu], u_pts[iu]
    tl, l_pts = tl[il], l_pts[il]
    t = np.linspace(0.0, 1.0, max(len(u_pts), len(l_pts)))
    up = np.column_stack([np.interp(t, tu, u_pts[:, 0]), np.interp(t, tu, u_pts[:, 1])])
    lo = np.column_stack([np.interp(t, tl, l_pts[:, 0]), np.interp(t, tl, l_pts[:, 1])])
    gaps = np.linalg.norm(lo - up, axis=1)
    gap = float(np.median(gaps[1:-1])) if len(gaps) > 2 else float(np.median(gaps))
    return 0.5 * (up + lo), gap


def lip_shape(p: np.ndarray, index: dict, size: tuple[int, int],
              got: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[tuple[str, np.ndarray, bool], tuple[np.ndarray, np.ndarray]]:
    """A lip-line frame's single mouth shape and its `(whole, hole)` masks under `--closed-lip-line`:
    when there is no class-map hole (`not hole.any()`) or the inner lip gap is at or under `INNER_GAP_CLOSED_PX`,
    one white centerline (`inner_lip_line`, `closed=False`) over flat skin grey with an empty hole; when open,
    one white outline (`inner_lip_ring`, `closed=True`) around a solid black interior (`inner_lip_ring` filled)."""
    import cv2
    w, h = size
    rings = {}
    for where, extra in (("outer", [index["center_of_cupid_bow"]]), ("inner", [])):
        a, b = p[index[f"r_{where}_corner_of_mouth"]], p[index[f"l_{where}_corner_of_mouth"]]
        upper = _along(p, _ids(index, tail=f"_of_upper_{where}_lip") + extra, a, b)
        lower = _along(p, _ids(index, tail=f"_of_lower_{where}_lip"), a, b)
        rings[where] = np.concatenate([a[None], upper, b[None], lower[::-1]])
    outer_m, inner_m = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    cv2.fillPoly(outer_m, [np.round(rings["outer"]).astype(np.int32)], 1)
    cv2.fillPoly(inner_m, [np.round(rings["inner"]).astype(np.int32)], 1)
    whole = (got[0] | outer_m.astype(bool)) if got is not None else outer_m.astype(bool)
    has_hole = True if got is None else bool(got[1].any())
    mid, gap = inner_lip_centerline(p, index)
    if (not has_hole) or (gap <= INNER_GAP_CLOSED_PX):
        return ("inner_lip_line", mid, False), (whole, np.zeros((h, w), bool))
    return ("inner_lip_ring", rings["inner"], True), (whole, inner_m.astype(bool))


def mouth_shapes(whole: np.ndarray, hole: np.ndarray, *, closed_as_line: bool = False) -> list[tuple[str, np.ndarray, bool]]:
    """A class-map mouth's outlines as the drawing's shapes: by default, the lips' outer outline and, where the mouth is open, its
    inner edge, each a closed line (`mouth_outer`, `mouth_inner`). With `closed_as_line` True, the mouth is one unambiguous shape:
    where `not hole.any()` (or a 1-2 px speck has no 3-point contour), one open parting line from corner to corner (`mouth_closed`,
    `closed=False`), never a hollow 'O' ring; where `hole.any()`, one closed white outline around the dark interior (`mouth_inner`,
    `closed=True`), dropping `mouth_outer`."""
    import cv2
    if closed_as_line:
        if hole.any():
            bridged = cv2.morphologyEx(hole.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            cs, _ = cv2.findContours(bridged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            valid = []
            for c in cs:
                poly = cv2.approxPolyDP(c, 1.0, True)[:, 0, :].astype(float)
                if len(poly) >= 3:
                    valid.append((float(cv2.contourArea(c)), poly))
            if valid:
                valid.sort(key=lambda x: x[0], reverse=True)
                return [("mouth_inner", valid[0][1], True)]
        xs = np.nonzero(whole.any(0))[0]
        if len(xs) < 2:
            return []
        ys = np.array([0.5 * (u[0] + u[-1]) for x in xs for u in (np.nonzero(whole[:, x])[0],)], float)
        pts = cv2.approxPolyDP(np.column_stack([xs.astype(float), ys])[:, None, :].astype(np.float32), 1.0, False)[:, 0, :].astype(float)
        return [("mouth_closed", pts, False)] if len(pts) >= 2 else []
    out = []
    for name, m in (("mouth_outer", whole), ("mouth_inner", hole)):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            c = cv2.approxPolyDP(c, 1.0, True)[:, 0, :].astype(float)
            if len(c) >= 3:
                out.append((name, c, True))
    return out


def draw(mesh: np.ndarray, drawings: list[list], width: float, bare: set[int], dots: bool = True, fills: list[list] | None = None,
         contrast: bool = False, mouths: list | None = None, *, smooth_lips: bool = False, darken_head: bool = False,
         keypoints: np.ndarray | None = None, index: dict | None = None) -> np.ndarray:
    """The mesh frames [n, h, w, 3] with every frame's drawing on it. With `fills`: first, inside each lash ring only, the
    iris in the edge colour, and with `contrast` the sclera solid white (the line colour) under it; then the lines over
    them, black a little wider first, then white, so the black edge stays round the white sclera and the black iris (an
    outside reviewer, 2026-10-14: filled after the lines, the white sclera covered the lash ring's black edge and the
    eye ran into the lid line). With `smooth_lips` (`--closed-lip-line`): the whole lip and under-lip area is smoothly flattened
    into the face's grey (`smooth_mouth_area`), closed mouth lines (`inner_lip_line`, `mouth_closed`) have no dark border or
    end-cap dot, and eye fills are clipped to the grey head's silhouette with a dark rim inside the silhouette edge (`M20`).
    With `darken_head`: the grey body model is scaled by `HEAD_DARKEN_SCALE` (`0.67`)."""
    from PIL import ImageChops
    if darken_head:
        mesh = np.round(mesh.astype(np.float32) * HEAD_DARKEN_SCALE).astype(np.uint8)
    look = h3_config.FACE_POINTS
    h, w = mesh.shape[1:3]
    out = np.empty_like(mesh)
    line_col = tuple(look["line"])
    edge_col = tuple(look["edge"])
    passes = ((edge_col, look["edge_px"]), (line_col, 0.0))
    erode_inner = max(HEAD_SILHOUETTE_ERODE_PX, int(np.ceil(width)))

    def fill(im, f: int, white: bool, full_head: np.ndarray | None = None, inner_head: np.ndarray | None = None) -> None:
        big = lambda m: Image.fromarray((m.astype(np.uint8) * 255)).resize(im.size, Image.NEAREST)        # noqa: E731
        fh_mask = big(full_head) if full_head is not None else None
        ih_mask = big(inner_head) if inner_head is not None else None
        for _, disc, lids in ([] if f in bare or not fills else fills[f]):
            a, b = Image.new("L", im.size, 0), Image.new("L", im.size, 0)
            ImageDraw.Draw(a).polygon([tuple(v * SUPER) for v in disc], fill=255)
            ImageDraw.Draw(b).polygon([tuple(v * SUPER) for v in lids], fill=255)
            if white:
                if fh_mask is not None and ih_mask is not None:
                    im.paste(edge_col, mask=ImageChops.multiply(b, fh_mask))
                    im.paste(line_col, mask=ImageChops.multiply(b, ih_mask))
                else:
                    im.paste(line_col, mask=b)
            iris_mask = ImageChops.multiply(a, b)
            if fh_mask is not None:
                iris_mask = ImageChops.multiply(iris_mask, fh_mask)
            im.paste(edge_col, mask=iris_mask)

    def mouth(im, f: int, inner_head: np.ndarray | None = None) -> None:
        """A mouth's paint-out and dark interior (`mouths[f]`: the mouth's whole outline filled, and its open inside, as masks of the frame):
        the body model's own lips painted out to the grey round them (sampling only the body model, never the black ground beside a profile
        face), and the open inside in the edge colour; its outlines are lines."""
        import cv2
        got = None if f in bare or not mouths else mouths[f]
        if got is None or not got[0].any():
            return
        whole, hole = got
        big = lambda m: Image.fromarray((m.astype(np.uint8) * 255)).resize(im.size, Image.NEAREST)        # noqa: E731
        if not smooth_lips:
            ring = cv2.dilate(whole.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~cv2.dilate(whole.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            on_body = ring & (mesh[f].max(-1) > 16)
            sample = on_body if on_body.any() else ring
            grey = tuple(int(v) for v in np.median(mesh[f][sample], axis=0)) if sample.any() else (128, 128, 128)
            im.paste(grey, mask=big(cv2.dilate(whole.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)))
        if hole.any():
            h_mask = (hole & inner_head) if inner_head is not None else hole
            if h_mask.any():
                im.paste(edge_col, mask=big(h_mask))

    for f in range(len(mesh)):
        base = mesh[f]
        kp_f = keypoints[f] if keypoints is not None else None
        full_h = head_silhouette(mesh[f], erode_px=0, p=kp_f, index=index) if smooth_lips else None
        inner_h = head_silhouette(mesh[f], erode_px=erode_inner, p=kp_f, index=index) if smooth_lips else None
        im_base = Image.fromarray(base).resize((w * SUPER, h * SUPER), Image.BICUBIC)
        im = im_base.copy()
        mouth(im, f, inner_h)
        fill(im, f, contrast, full_h, inner_h)
        d = ImageDraw.Draw(im)
        for colour, pad in passes:
            for name, pts, closed in ([] if f in bare else drawings[f]):
                if pad > 0 and name in ("inner_lip_line", "mouth_closed"):
                    continue
                run = [tuple(v * SUPER) for v in (np.vstack([pts, pts[:1]]) if closed else pts)]
                if len(run) > 1:
                    d.line(run, fill=colour, width=max(1, int(round((width + 2 * pad) * SUPER))), joint="curve")
                r = ((width if dots else width / 2) + pad) * SUPER   # a dot at every point; with none, a round end and a round joint
                for x, y in run:
                    d.ellipse([x - r, y - r, x + r, y + r], fill=colour)
        frame_out = np.asarray(im.resize((w, h), Image.LANCZOS)).copy()
        if full_h is not None and inner_h is not None:
            ih_big = np.asarray(Image.fromarray((inner_h.astype(np.uint8) * 255)).resize(im.size, Image.NEAREST)) > 0
            if np.any((np.asarray(im) != np.asarray(im_base))[~ih_big]):
                frame_out[~full_h] = base[~full_h]
        out[f] = frame_out
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
    """For each frame, where (x, y) the written video is a line's white and the mesh it was drawn on was not;
    less `but` (the filled sclera of `--iris contrast`, white that no line drew)."""
    n, h, w = mesh.shape[:3]
    got = np.stack(list(stream(str(path), (w, h), skip, n, pix="rgb24")))
    if len(got) != n:
        raise RuntimeError(f"{path} reads back {len(got)} frames, {n} were asked for")
    # A CODEC MOVES AN EDGE BY A PIXEL: the body model's own bright edge, shifted one pixel in the encode, read back as one pixel of white
    # on a bare frame and failed lead_fp_2823c's remake (3109, 2026-10-15; mesh 227 beside 255, the written frame over WHITE). So white is a
    # line's only where the mesh is not near white within a pixel either way
    import cv2
    near = (mesh.min(-1) > WHITE - NEAR_WHITE).astype(np.uint8)
    near = np.stack([cv2.dilate(x, np.ones((3, 3), np.uint8)) for x in near]).astype(bool)
    found = (got.min(-1) > WHITE) & ~near
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


#: The shapes that draw a mouth shut, and those that draw it open, by their names in a frame's drawing. Inherited: the
#: names `lip_shape`, `mouth_shapes` and `shapes` above give their shapes. Without `--closed-lip-line` the points' lip
#: rings are drawn on every frame and a closed class-map mouth draws `mouth_outer` alone, so both read as open: the
#: sheet sees a mouth drawn shut only under that option.
CLOSED_MOUTHS, OPEN_MOUTHS = ("inner_lip_line", "mouth_closed"), ("mouth_outer", "mouth_inner", "inner_lip_ring")


def sheet_picks(source_frames: np.ndarray, first: int, points: dict, count: int, drawings: list, *, left: set = frozenset(),
                shut: list | None = None, mouths: list | None = None, closed_lip_line: bool = False, closed: list | None = None) -> dict:
    """The frames the sheet shows (`_lib/pick_frames.py`): where the face's points, the inner-lip gap (its crossings of
    `INNER_GAP_CLOSED_PX` and its peaks), the mouth drawn, a mouth drawn shut while the points hold it open (or, with
    `closed_lip_line`, drawn open while they hold it shut), or the picture round the nose change, either side of each;
    the edges of the frames left bare or drawn with the eyes shut (a person's word, so always), and of those whose eyes
    the rule drew shut (`closed`, `eye_reading`: the drawing changes form there, so always); and one a second."""
    h, w = source_frames.shape[1:3]
    idx = points["index"]
    gap = [inner_lip_centerline(p, idx)[1] for p in points["keypoints"]]
    drawn = []
    for f, d in enumerate(drawings):
        names = {n for n, _, _ in d}
        drawn.append("shut" if names & set(CLOSED_MOUTHS) else "open" if any(n.endswith("_lip_ring") or n in OPEN_MOUTHS for n in names)
                     else "class map" if mouths and mouths[f] is not None else "none")
    against = ["drawn shut, the points open" if g > INNER_GAP_CLOSED_PX and m == "shut"
               else "drawn open, the points shut" if closed_lip_line and g <= INNER_GAP_CLOSED_PX and m == "open" else "agree"
               for g, m in zip(gap, drawn)]
    series = {**picker.points_series(points), "inner_lip_gap_px": {"values": gap, "rules": ("line", "peak"), "line": INNER_GAP_CLOSED_PX},
              "mouth_drawn": drawn, "mouth_drawn_against_the_gap": against,
              "left_bare": {"values": picker.labels_of(count, 0, left, "bare", "drawn"), "record": True},
              "eyes": {"values": picker.labels_of(count, 0, shut, "drawn shut", "as the points"), "record": True},
              **({"eyes_by_the_rule": {"values": picker.labels_of(count, 0, closed, "drawn shut by the rule", "open"), "record": True}}
                 if closed else {}),
              "picture_round_the_nose": {"values": picker.difference_in_boxes(source_frames, picker.nose_boxes(points, (w, h))), "rules": ("peak",)}}
    return picker.sheet_picks(first, count, series)


def sheet(path: Path, source_frames: np.ndarray, first: int, drawn: np.ndarray, points: dict, count: int, picked: dict | None = None) -> list[int]:
    """The picked frames (`sheet_picks`; the points' changes and one a second when not given) in three panels (the
    original, the drawn body model, the two blended), cut round the nose, each with why it was picked."""
    h, w = drawn.shape[1:3]
    idx = points["index"]
    picked = picked or picker.sheet_picks(first, count, picker.points_series(points))
    why = picker.first_reasons(picked)
    picks = [f - first for f in picker.frames_of(picked)]
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
        ImageDraw.Draw(row).text((6, 20), f"picked: {why.get(first + f, '')}", fill=(255, 255, 255))
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


def _spans(frames: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def closed_by_the_rule(reads: list[dict], first: int, left: set, shut: list) -> dict:
    """The record of the closed eye (`eye_reading`) over a span: the frames whose eyes it drew shut, and every frame in spans of one
    form, each with the eyes that were downcast and each eye's share of its iris ring outside the lids, least and most over the span, so
    that a reader sees what was drawn and why without the picture. A frame left bare or shut on a person's word is said as such: the
    rule does not act there."""
    def form(f: int, r: dict) -> tuple[str, tuple]:
        if f in left:
            return "no face drawn", ()
        if f in shut:
            return "shut on a person's word", ()
        if r["shut"]:
            return "shut", tuple(side for side in SIDES if r["eyes"][side]["downcast"])
        return "open", ()

    def least_most(vals: list, digits: int) -> list | None:
        vals = [v for v in vals if v is not None]
        return [round(min(vals), digits), round(max(vals), digits)] if vals else None
    spans: list[dict] = []
    for f, r in enumerate(reads):
        drawn, met = form(f, r)
        if spans and (spans[-1]["drawn"], tuple(spans[-1]["downcast_eyes"])) == (drawn, met):
            spans[-1]["last"] = first + f
            spans[-1]["rows"].append(r)
        else:
            spans.append({"first": first + f, "last": first + f, "drawn": drawn, "downcast_eyes": list(met), "rows": [r]})
    for sp in spans:
        rows = sp.pop("rows")
        sp["mean_outside"] = least_most([r["mean_outside"] for r in rows], 3)
        for side in SIDES:
            sp[side] = {"ring_outside": least_most([r["eyes"][side]["ring_outside"] for r in rows], 3)}
    shut_here = [f for f, r in enumerate(reads) if form(f, r)[0] == "shut"]
    one_under = [first + f for f in shut_here
                 if any(reads[f]["eyes"][side]["ring_outside"] is not None and not reads[f]["eyes"][side]["downcast"] for side in SIDES)]
    return {"on": True, "drawn_as": "both eyes shut, each one line along its upper lash line (EYES_SHUT_FORMS 'lids'): no lash ring, lid or crease "
                                    "line, iris, pupil or sclera; the brows stay",
            "rule": "on a frame where the mean of the scored eyes' iris rings outside their lids (mean_outside; that eye's share when only one "
                    "scores) is downcast_at_least or more; both eyes or neither; ring_outside is each eye's share, least and most over a span, "
                    "and none where its iris ring scores under score_at_least",
            "downcast_at_least": DOWNCAST_RING_OUTSIDE_MIN,
            "frames": _spans([first + f for f in shut_here]),
            "one_eye_under_the_line_on": _spans(one_under),
            "spans": spans}


def closed_said(block: dict, count: int) -> str:
    """The closed eye's record (`closed_by_the_rule`) in a line: what the reference step prints."""
    frames = [f for a, b in block["frames"] for f in range(a, b + 1)]
    under = [f for a, b in block["one_eye_under_the_line_on"] for f in range(a, b + 1)]
    spans = lambda sp: ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in sp) or "none"      # noqa: E731
    return (f"BOTH EYES DRAWN SHUT BY THE RULE on {len(frames)} of {count} frames ({spans(block['frames'])}): one line along each upper "
            f"lash line where the two eyes' mean iris ring outside the lids is {block['downcast_at_least']:g} or more; one eye under the line "
            f"on {len(under)} ({spans(block['one_eye_under_the_line_on'])}); the spans with each eye's share are in the record "
            "(eye_rings.closed_by_the_rule)")


def run(mesh_path: str, points_path: str, first: int, count: int, out: Path, by: str, *, names: list[str] | None = None,
        mesh_first: int | None = None, points_first: int | None = None, width: float | None = None, dots: bool = True,
        leave_out: frozenset = frozenset(), bare: list[int] | None = None, bare_by: str = "", source: str = "", source_first: int = 0,
        iris: str = IRIS_FORMS[0], mouths: list | None = None, mouth_lines: bool = True, no_mouth: list[int] | None = None,
        class_mouth_frames: list[int] | None = None, closed_lip_line: bool = False, darken_head: bool = False,
        eyes_shut: list[int] | None = None, eyes_shut_form: str | None = None, eyes_shut_by: str = "", say=print) -> dict:
    """Draw, write, read back, record. `mouths`: a class-map mouth a frame ((whole, hole) masks of the frame, or None), drawn in
    place of the points' mouth (`head_reference`, `--mouth auto` where the lip check fails and the class map holds the mouth);
    with `mouth_lines` False its lips are painted out and its inside filled under the points' own mouth lines, and no
    outline of its own is drawn. `no_mouth`: source frames on which no mouth is drawn at all, neither the points' lip rings nor a
    class-map mouth (`head_reference`: the window is silent or the lips are left out): the body model's own lips stay.
    `class_mouth_frames`: source frames on which, with `mouth_lines` False, the class map's mouth is drawn with its own
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
    drawings = [shapes(p, s, idx, line, leave_out, iris, offs[f], closed_lip_line=closed_lip_line) for f, (p, s) in enumerate(zip(points["keypoints"], points["scores"]))]
    if not dots:
        drawings = [lines_only(d) for d in drawings]                # no dots is no dots: the points that belong to no line go too
    wide, face, wide_by = width_for(points, width)
    left = {f - first for f in (bare or []) if first <= f < first + count}
    fills = [discs(p, s, idx, line, leave_out, closed_lip_line=closed_lip_line, iris=iris) for p, s in zip(points["keypoints"], points["scores"])] if iris in IRIS_FORMS[2:] else None
    shut = sorted({f - first for f in (eyes_shut or []) if first <= f < first + count})
    if shut and (eyes_shut_form not in EYES_SHUT_FORMS or not str(eyes_shut_by).strip()):
        raise SystemExit(f"--eyes-shut-on takes a form, one of {EYES_SHUT_FORMS} (--eyes-shut-form), and whose word it is; got {eyes_shut_form!r}")
    for f in shut:
        drawings[f] = [d for d in drawings[f] if not any(d[0].startswith(f"{side}{part}") for side in SIDES for part in EYE_PARTS)]
        if eyes_shut_form == "lids" and "eyes" not in leave_out:
            drawings[f] += [(f"{side}_shut_lid", fp._line(points["keypoints"][f], idx, side, "upper"), False) for side in SIDES]
        if fills:
            fills[f] = []
    # THE CLOSED EYE BY THE RULE (2026-10-21): under --iris contrast with the closed lip line, `shapes` and `discs` drew a frame's eyes shut
    # where `eye_reading` says so; read again here for the record. A person's word (`shut`) wins on its frames, in its own form
    ruled = closes_downcast_eyes(iris, closed_lip_line, leave_out)
    reads = [eye_reading(p, s, idx, line) for p, s in zip(points["keypoints"], points["scores"])] if ruled else []
    by_rule = {f for f, r in enumerate(reads) if r["shut"]}
    cmap_set = {f - first for f in (class_mouth_frames or []) if first <= f < first + count}
    bare_mouth = sorted({f - first for f in (no_mouth or []) if first <= f < first + count})
    for f in bare_mouth:
        drawings[f] = [d for d in drawings[f] if not d[0].endswith("_lip_ring") and d[0] not in ("inner_lip_line", "mouth_outer", "mouth_inner", "mouth_closed")]
    if mouths:
        if len(mouths) != count:
            raise SystemExit(f"{len(mouths)} class-map mouths for {count} frames")
        mouths = [None if f in bare_mouth else m for f, m in enumerate(mouths)]
        own = cmap_set - set(bare_mouth)
        for f in own:
            drawings[f] = [d for d in drawings[f] if not d[0].endswith("_lip_ring")]
        if mouth_lines or own:
            import cv2
            for f in range(count):
                if mouths[f] is not None and (mouth_lines or f in own):
                    sh = mouth_shapes(*mouths[f], closed_as_line=closed_lip_line)
                    if closed_lip_line:
                        gap_f = inner_lip_centerline(points["keypoints"][f], idx)[1]
                        if gap_f <= INNER_GAP_CLOSED_PX or any(n == "mouth_closed" for n, _, _ in sh):
                            sh = []
                            mouths[f] = None
                        elif any(n == "mouth_inner" for n, _, _ in sh):
                            bridged = cv2.morphologyEx(mouths[f][1].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
                            sh_fit, m_fit = fit_mouth_to_head(mesh[f], points["keypoints"][f], idx, sh[0], (mouths[f][0] | bridged, bridged))
                            sh = [sh_fit]
                            mouths[f] = m_fit
                    drawings[f] = drawings[f] + sh
    if closed_lip_line:
        if mouths is None:
            mouths = [None] * count
        for f in range(count):
            if any(d[0].endswith("_lip_ring") for d in drawings[f]):
                sh_f, m_f = lip_shape(points["keypoints"][f], idx, (w, h), mouths[f])
                if sh_f[0] == "inner_lip_line":
                    drawings[f] = [d for d in drawings[f] if not d[0].endswith("_lip_ring")]
                    mouths[f] = None
                else:
                    sh_f, m_f = fit_mouth_to_head(mesh[f], points["keypoints"][f], idx, sh_f, m_f)
                    drawings[f] = [d for d in drawings[f] if not d[0].endswith("_lip_ring")] + [sh_f]
                    mouths[f] = m_f
            if f not in left:
                hm_f = head_silhouette(mesh[f], p=points["keypoints"][f], index=idx)
                drawings[f] = [c for n, pts_d, cl_d in drawings[f] for c in clip_shape_to_mask(n, pts_d, cl_d, hm_f)]
    mesh_draw = np.round(mesh.astype(np.float32) * HEAD_DARKEN_SCALE).astype(np.uint8) if darken_head else mesh
    pictures = draw(mesh_draw, drawings, wide, left, dots, fills, contrast=iris == IRIS_FORMS[3], mouths=mouths,
                    smooth_lips=closed_lip_line, keypoints=points["keypoints"], index=idx)
    but = filled_px(fills, (w, h)) if iris == IRIS_FORMS[3] else None
    out.parent.mkdir(parents=True, exist_ok=True)
    marks.write_video(out, pictures)
    drawn = np.array([f not in left for f in range(count)])
    expected = np.array([white_expected(drawings[f], wide, dots, edge_px=EDGE_PX) if drawn[f] else 0.0 for f in range(count)])
    back = read_back(out, mesh_draw, drawings, expected, drawn, wide, but)
    white, lags = back["white"], back["lags"]
    src = marks.source_frames(source, (w, h), first, count, source_first) if source else None
    clock = marks.against_the_source(src, mesh, first) if src is not None else None
    passed = bool(back["passed"])
    named = [{n for n, _, _ in drawings[f]} | {n for n, _, _ in (fills[f] if fills else [])} for f in range(count)]
    kinds = ("iris",) if fills else RINGS
    shape = "disc" if fills else "ring"
    rings = {f"{side}_{kind}": [first + f for f in range(count)
                                if f"{side}_{kind}_{shape}" not in named[f] and f"{side}_{kind}_ring" not in named[f] and f not in shut
                                and f not in by_rule]
             for side in SIDES for kind in kinds} if "eyes" not in leave_out else {}
    scores = {k: [round(float(v), 2) for v in np.percentile([ring_scores(s, idx)[k] for s in points["scores"]], [0, 50, 100])] for k in rings}
    fit = {str(k): (None if np.isnan(v) else round(v, 3)) for k, v in lags.items()}
    record = {
        "video": out.name, "made_by": by, "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "writer_stamp": writer_stamp(__file__),
        "frames": [first, first + count - 1], "size": [w, h],
        "made_from": {"body_model_video": str(mesh_path), "its_first_frame_is_source_frame": first - skip, "points": str(points_path),
                      "their_first_row_is_source_frame": pf},
        "line_width_px": wide, "width_set_by": wide_by, "a_dot_at_every_point": dots, "face_height_px_median": round(face, 1),
        "points_drawn_alone": sorted({n for d in drawings for n, pts, _ in d if len(pts) == 1}),
        "parts_left_out": sorted(leave_out),
        "lines_px_a_frame_median": round(float(np.median([length(d) for d in drawings])), 1),
        "eye_rings": {"score_at_least": line, "that_line_is": "a starting value from one shot: under it a ring's points lay far off the face",
                      "frames_a_ring_is_left_out_on": rings, "score_a_frame_least_median_most": scores,
                      "not_counted": "the frames her eyes are drawn shut on a person's word (frames_with_the_eyes_shut) or by the rule "
                                     "(closed_by_the_rule): no ring there by that word or that rule",
                      "drawn": iris, "drawn_is": {IRIS_FORMS[0]: "only the stretches of each ring between the lids", IRIS_FORMS[1]: "each ring whole",
                                                  IRIS_FORMS[2]: "each iris a dark disc inside the lash ring, and no ring",
                                                  IRIS_FORMS[3]: "inside each lash ring the sclera solid white and the iris solid black, drawn before "
                                                                 "the lines, whose black edge stays round them; the white of the sclera is left out of the read-back"}[iris],
                      "share_of_the_ring_outside_the_lids_least_median_most": {
                          k: ([round(float(v), 2) for v in np.percentile(vals, [0, 50, 100])] if vals else None)
                          for k in rings for vals in [[o[k] for o in offs if k in o]]},
                      "closed_by_the_rule": (closed_by_the_rule(reads, first, left, shut) if ruled else
                                             {"on": False, "why": "the rule acts on the eyes drawn under --iris contrast with the closed lip line"
                                                                  + ("; the eyes are left out" if "eyes" in leave_out else "")})},
        **({"darken_head": {"scale": HEAD_DARKEN_SCALE}} if darken_head else {}),
        "frames_left_bare": sorted(first + f for f in left), "left_bare_by": bare_by,
        "frames_with_no_mouth": [first + f for f in bare_mouth],
        "frames_with_the_eyes_shut": ({"frames": [first + f for f in shut], "form": eyes_shut_form, "by": eyes_shut_by} if shut else None),
        "frames_with_the_class_maps_own_mouth": sorted(first + f for f in (cmap_set - set(bare_mouth))),
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
        assert src is not None                      # read above whenever there is a --source
        picked = sheet_picks(src, first, points, count, drawings, left=left, shut=shut, mouths=mouths, closed_lip_line=closed_lip_line,
                             closed=sorted(by_rule - set(shut) - left))
        record["sheet"] = {"file": sheet_path.name, "frames": sheet(sheet_path, src, first, pictures, points, count, picked),
                           "why": {str(k): v for k, v in picker.first_reasons(picked).items()}, "dropped": picked["dropped"],
                       "cap": picked["cap"], "over_the_cap": picked["over_the_cap"], "looked_at_by": ""}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    say(f"face {face:.0f} px tall at the median; lines {wide:g} px wide ({wide_by}), {'a dot at every point' if dots else 'no dots'}; "
        f"{record['lines_px_a_frame_median']:.0f} px of line a frame at the median" + (f"; left out: {', '.join(sorted(leave_out))}" if leave_out else ""))
    for key, gone in rings.items():
        lo, mid, hi = scores[key]
        say(f"{key} ring: left out on {len(gone)} of {count} frames (score a frame {lo:.2f} to {hi:.2f}, median {mid:.2f}, against {line:g})")
    if back["too_thin"]:
        say(f"on {len(back['too_thin'])} of {count} frames the lines should leave under {THIN_PX:g} pixels of white")
    if shut:
        say(f"her eyes drawn shut ({eyes_shut_form!r}) on {len(shut)} of {count} frames, on a person's word: {eyes_shut_by}")
    if ruled:
        say(closed_said(record["eye_rings"]["closed_by_the_rule"], count))
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
           leave_out: frozenset = frozenset(), iris: str = IRIS_FORMS[0], darken_head: bool = False, say=print) -> bool:
    """A drawn video somebody made, read against a points file on the READER's account of each one's first frame."""
    stale = check_upstream_mtime(video, {"--mesh": mesh_path, "--points": points_path})
    if stale:
        say(f"WARNING: {'; '.join(stale)}")
    points, _ = rows_of(points_path, names, first, count, points_first)
    w, h, _ = probe(mesh_path)
    mesh = np.stack(list(stream(mesh_path, (w, h), first - (first if mesh_first is None else mesh_first), count, pix="rgb24")))
    if len(mesh) != count:
        raise SystemExit(f"asked for {count} frames; the body-model video gave {len(mesh)}")
    line = h3_config.POSE_READING["eye_ring_score_at_least"]
    fills = ([discs(p, s, points["index"], line, leave_out) for p, s in zip(points["keypoints"], points["scores"])]
             if iris == IRIS_FORMS[3] else None)
    mesh_draw = np.round(mesh.astype(np.float32) * HEAD_DARKEN_SCALE).astype(np.uint8) if darken_head else mesh
    try:
        white = white_at(video, mesh_draw, first - (first if video_first is None else video_first), but=filled_px(fills, (w, h)))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    wide = width_for(points, width)[0]
    segs = [segments(shapes(p, s, points["index"], line, leave_out, iris))
            for p, s in zip(points["keypoints"], points["scores"])]
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
        a.add_argument("--darken-head", action="store_true", help="the video was drawn on a darkened grey head (--darken-head)")
        v = a.parse_args(sys.argv[2:])
        names = json.loads(Path(v.names).read_text()) if v.names else None
        return 0 if verify(v.video, v.mesh, v.points, v.first, v.count, names=names, mesh_first=v.mesh_first, points_first=v.points_first,
                           video_first=v.video_first, width=v.width, leave_out=parts_arg(v.leave_out), iris=v.iris,
                           darken_head=v.darken_head) else 1
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
    a.add_argument("--darken-head", action="store_true", help="darken the grey body model by scale (HEAD_DARKEN_SCALE) to raise line contrast")
    a.add_argument("--source", default="", help="the original video: writes a sheet to look at")
    a.add_argument("--source-first", type=int, default=0, help="the source frame of that video's frame 0, for one cut from the original; 0 when left out")
    v = a.parse_args()
    names = json.loads(Path(v.names).read_text()) if v.names else None
    rec = run(v.mesh, v.points, v.first, v.count, Path(v.out), v.by, names=names, mesh_first=v.mesh_first, points_first=v.points_first,
              width=v.width, dots=not v.no_dots, leave_out=parts_arg(v.leave_out), bare=marks.frames_arg(v.bare), bare_by=v.bare_by, source=v.source,
              source_first=v.source_first, iris=v.iris, darken_head=v.darken_head)
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
