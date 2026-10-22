#!/usr/bin/env python3
"""Cut a reference video to ONE box round a shot's head and scale it back up to the canvas.

    python bench/crop_reference_to_head.py --video REFERENCE.mp4 --points SHOT_points.npz --first 1039 --count 22 \\
        --out <share>/Video/<video>/<who>/second_mesh_with_lids_head_crop_1039.mp4 --by <session> \\
        [--room 1.2] [--head-track HEAD_TRACK.mkv] [--head-track-first N] [--no-head-track] [--cranium] [--box X0,Y0,X1,Y1] \\
        [--video-first N] [--points-first N] [--names NAMES.json] [--fit]

A reference video drawn at the canvas's size shows a small face small: on a face a few dozen pixels tall an
eye drawn at its true size is a patch a dozen pixels wide, and no line in it can be told from the next
(`bench/draw_face_points.py`, its first sheet). The reference does not have to be the whole frame. This takes a
reference video at the canvas's size (a body model with lid marks, or with the face drawn) and the pose model's
points along the shot (`bench/pose_along_shot.py`), and writes the same frames cut to one box round her head
and enlarged to the canvas, so the face fills the picture.

ONE BOX FOR THE WHOLE SPAN, so nothing shakes: the head moves inside the picture as it moves in the shot, and
the picture's edges stand still. The box is:

- round the HEAD TRACK and the FACES of every frame asked for: when the subject's head track is given
  (`--head-track`) or sits beside the points' box-mask track (`<prefix>_<subject>_head_track_*.mkv` beside
  `<prefix>_<subject>_track_*.mkv` in `<points>.json`, or beside `<prefix>_<subject>_mesh_*.mp4`), the box holds
  every frame's head mask with `CROWN_ROOM` face heights above the crown and `HEAD_TRACK_ROOM` on the sides and
  bottom, together with every frame's face (the lash lines, the glabella, the tip of the nose and the tip of the
  chin that `bench/_lib/face_points.py` reads) with `HEAD_ROOM` (`--room`) face heights on every side;
- when no head track is there (or `--no-head-track` is given), round the faces alone with `HEAD_ROOM` (`--room`)
  times the shot's median face height on every side. The pose model's points stop at the brows and the chin, so on
  a tilted or bowed head where the crown reaches more than a face height above the glabella, the head track keeps
  the crown inside the box;
- with `--cranium` (`head_reference.py --crop cranium`, the plan's `reference.crop`; 2026-10-22), round the GREY HEAD
  of the body model instead (`cranium_box`): a body-model reference holds only the bald grey figure with the face drawn,
  so the head framed is the figure's, not the real hair. Each frame's face, with the figure's pixels above mid-face
  (halfway from the glabella to the chin; below it the neck and shoulders join the arms) joined to it, `CRANIUM_ROOM` above and beside them and `CRANIUM_NECK` below the chin; no head track read. It says on
  which frames the grey head still reaches the box's edge, and what the subject's travel over the span costs the
  enlargement (each frame's own box against the one box): figures, not a verdict. The box does not follow the head;
- with `--box`, the box given, as it is (a source cut by the box a cranium cut of its body model found);
- in the CANVAS'S SHAPE, grown the short way to reach it, then moved to lie inside the canvas if it hung over
  an edge. A box that would be larger than the canvas is the whole frame, and the tool says so.

It writes the video, a record beside it (`.json`: the box, the scale, the frames, what it was made from, where a
point of the canvas lands in the new picture), the span's points carried into the new picture
(`<out>_points.npz`, as `bench/pose_along_shot.py` writes one) and a sheet, the reference with the box drawn on it
beside the cut picture, to look at before the video is used. Its frames are where the face's points, the head track
or the picture change, either side of each, and one a second (`bench/_lib/pick_frames.py`), each saying why.

THE ORIGINAL, CUT THE SAME WAY. With `--fit` the video is the source itself, of whatever size: it is fitted to
the points' canvas as the lane's loader fits it (`bench/_lib/frames.py::FIT`) and then cut by the same box, since
the box comes from the points and not from the video. That is the picture a sheet or a strip lays under a cut
reference, and what `--source` of the two drawing tools takes with `--source-first`. A source's frame 0 is source
frame 0 unless `--video-first` says otherwise.

CUT FIRST, THEN DRAW. Enlarging a drawn video enlarges its lines with it: what ran together stays together.
To draw a face on the enlarged picture, cut the PLAIN body-model video with this tool and give
`bench/draw_face_points.py` the cut video and the points file written beside it; its lines are then drawn at the
new size.

WHAT IT CANNOT KNOW, and says:
- Whether the points are on her. It takes the points file it is given, and a box round the wrong person is a
  box round the wrong person: open the sheet. The record carries what the points file's own record says it was
  made from (the video, and the track whose box the pose model was given), when that record is beside it.
- Whether the reference video and the points are on one clock. It is told which source frame each begins on.
- What a model makes of a reference that is not the frame's own size any more: the picture no longer lies over
  the source pixel for pixel. That is a render's question.

WHAT THE READ-BACK IS. The written video is read back and each frame set against the cut picture made for it,
for the frame before and for the frame after: a file with a frame dropped, repeated or out of order fits a
neighbour better and fails. Where the picture does not change between frames no slip can be told, and the tool
says so.

The enlarging is the picture's own pixels resampled: it adds no detail. A line a pixel wide comes out wider
and softer by the scale, which the record gives.
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
from _lib import face_points as fp  # noqa: E402
from _lib import npz_writer_fields, writer_stamp  # noqa: E402
from _lib import pick_frames as picker  # noqa: E402
from _lib.frames import FIT, probe, read_mask, stream  # noqa: E402

import draw_face_points as faces  # noqa: E402  the points loader: one copy
import draw_lid_marks as marks  # noqa: E402  the writer: one copy

#: Room left round the faces on every side, in face heights (glabella to chin). Reasoned, and a starting value: a
#: head is a little under twice that length from crown to chin, so with this much room a head that holds still
#: sits in a box about twice its own height. With no head track it alone sizes the box: every reference before ba061ef3 (2026-10-19)
#: was cut so (0278f among them), and `head_reference.py --crop face` (the plan's `reference.crop`) cuts so still.
HEAD_ROOM = 1.2
#: Room left above the head track's crown, in face heights (glabella to chin), when a head track is read. A
#: STARTING VALUE, measured 2026-10-18 on the kitchen shot (`shot_c_0604c`, SAM 3.1 head track) and frontal mom
#: (`shot_c_0278c`): on tilted frames (`767, 771, 772`) the head track's crown reaches 0.93 to 1.16 face heights
#: above the glabella and `HEAD_ROOM` alone leaves 3 to 20 px above the crown (`-43 px` on `751-778`); `0.4` face
#: heights above the head track's top (`hy0`) leaves 27 to 31 px above the crown on `767, 771, 772`, holds the
#: whole head on `751-778` and `751-785` (`y0 = 0`, 5 px above `hy0`, 33 px above the 3D head), and keeps frontal
#: `805-833` within 4% of the face-only box (`410x307` vs `394x295`, 30 px above the crown) while holding all of
#: frontal `278-293`'s crown (`+47 px` vs `-17 px`). Scales with `--room / HEAD_ROOM`.
CROWN_ROOM = 0.4
#: Room left on the sides and bottom of the head track, in face heights, when a head track is read. A STARTING
#: VALUE, measured 2026-10-18 alongside `CROWN_ROOM`: on tilted (`767, 771, 772`), profile (`630-657`) and
#: wide-hair frontal (`278-293`) frames the back of the head or hair reaches 1.00 to 1.75 face heights sideways
#: from the facial features; `0.25` face heights round the head track's sides and bottom holds the whole head
#: (`0` frames touching the crop edge) without widening frontal `805-833` beyond `410x307`. Scales with
#: `--room / HEAD_ROOM`.
HEAD_TRACK_ROOM = 0.25
#: The body model's figure, under `--cranium`: a pixel whose brightest channel is over this grey level. INHERITED: the level the drawing
#: reads the body model by (`draw_face_points.head_silhouette`'s `frame.max(-1) > 16`, `head_reference.CLASS_MOUTH_GATE["body_grey"]`).
BODY_GREY = 16
#: Under `--cranium`, room above and beside the grey head and the face, in face heights (glabella to chin). REASONED, a starting value
#: (the Gemini reviewer's design, 2026-10-22, accepted by the lead): the grey head is the figure's own edge, so the room is clearance,
#: not a guess at hair: enough that the enlarged picture's resampled edge and a pixel's jitter between frames stay off the box's edge
#: (about 15 px on a 100 px face), small so the head fills the picture. Not yet judged on a render. Scales with `--room / HEAD_ROOM`.
CRANIUM_ROOM = 0.15
#: Under `--cranium`, room below the chin, in face heights. REASONED, a starting value (the same design): the neck and the line of the
#: shoulders stay in the picture, so the head's turn against the body can be read; the shoulders' full width is not held. Not yet
#: judged on a render. Scales with `--room / HEAD_ROOM`.
CRANIUM_NECK = 0.35
#: Under `--cranium`, the most the one box may be enlarged: the box is never narrower than the canvas's width over this. The body model
#: is rendered at the canvas's size and the cut only resamples it, so a small head enlarged far is a soft picture (the owner, 2026-10-15:
#: "maybe the zoom went too far in and it got blurry"); without it the cranium box reached 6.51 to 9.85 times on second 4479 and 5.30
#: to 8.35 on 4602 (the Gemini reviewer's scan, 2026-10-22). MEASURED, a starting value: the largest enlargement of a reference on a
#: render the owner answered "keep this one" (`data/2026-10-22_fun_delivery_tables/ledger_fun_2026-10-22T1915Z.json`, 61 such
#: renders with a reference record: median 2.07, 90th percentile 3.74, most 5.753 on third_fp_4653b, then 5.278 on second_hfp_4602c
#: and d). `head` and `face` have no such bound. The lever past it is the body model rendered at the cut's size, not this number.
CRANIUM_MOST = 5.75
#: The written video must sit within this many grey levels of the picture made for it, at the mean. Reasoned: the
#: encode moves a picture by about a level; a wrong frame of a moving head moves it by many.
ENCODE_LEVELS = 6.0
#: How much nearer its own frame than a neighbour a written frame must be for the frames to be called in place, in
#: grey levels at the mean; under it either way nothing moved enough to tell.
SLIP_LEVELS = 0.25


def face_box(p: np.ndarray, index: dict) -> np.ndarray:
    """One frame's face as [x0, y0, x1, y1]: round the lash lines, the glabella, the nose's tip and the chin's."""
    pts = p[fp.face_ids(index)]
    return np.concatenate([pts.min(0), pts.max(0)])


def head_boxes_of(head_track: np.ndarray | list | None) -> np.ndarray:
    """Non-empty `[x0, y0, x1, y1]` boxes (the far side not included, as `box` is) from a `[n, h, w]` mask array
    or a sequence of `[x0, y0, x1, y1]` boxes."""
    if head_track is None:
        return np.zeros((0, 4), float)
    arr = np.asarray(head_track)
    if arr.ndim == 3:
        boxes = []
        for m in arr:
            ys, xs = np.nonzero(m > 0 if m.dtype != bool else m)
            if len(xs):
                boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
        return np.array(boxes, float) if boxes else np.zeros((0, 4), float)
    if arr.ndim == 2 and arr.shape[1] == 4:
        keep = (arr[:, 2] > arr[:, 0]) & (arr[:, 3] > arr[:, 1])
        return arr[keep].astype(float)
    return np.zeros((0, 4), float)


def one_box(points: dict, size: tuple[int, int], room: float, head_track: np.ndarray | list | None = None,
            crown_room: float | None = None, head_track_room: float | None = None) -> dict:
    """The one box of a span: round every frame's head track (with crown room) and face (with room), in the
    canvas's shape, inside the canvas."""
    w, h = size
    idx = points["index"]
    each = np.array([face_box(p, idx) for p in points["keypoints"]])
    union = np.concatenate([each[:, :2].min(0), each[:, 2:].max(0)])
    face = float(np.median([fp.face_height(p, idx) for p in points["keypoints"]]))
    said = {"faces": [round(float(v), 1) for v in union], "face_height_px_median": round(face, 1), "room_in_face_heights": room}
    hboxes = head_boxes_of(head_track)
    if len(hboxes):
        heads = np.concatenate([hboxes[:, :2].min(0), hboxes[:, 2:].max(0)])
        head_h = float(np.median(hboxes[:, 3] - hboxes[:, 1]))
        scale_r = room / HEAD_ROOM if HEAD_ROOM else 1.0
        c_room = (CROWN_ROOM * scale_r) if crown_room is None else float(crown_room)
        ht_room = (HEAD_TRACK_ROOM * scale_r) if head_track_room is None else float(head_track_room)
        face_want = np.array([union[0] - room * face, union[1] - room * face, union[2] + room * face, union[3] + room * face], float)
        head_want = np.array([heads[0] - ht_room * face, heads[1] - c_room * face, heads[2] + ht_room * face, heads[3] + ht_room * face], float)
        want = np.concatenate([np.minimum(face_want[:2], head_want[:2]), np.maximum(face_want[2:], head_want[2:])])
        need_w, need_h = float(want[2] - want[0]), float(want[3] - want[1])
        cx, cy = float((want[0] + want[2]) / 2), float((want[1] + want[3]) / 2)
        said.update(heads=[round(float(v), 1) for v in heads], head_height_px_median=round(head_h, 1),
                    crown_room_in_face_heights=round(c_room, 3), head_track_room_in_face_heights=round(ht_room, 3),
                    head_track_frames=int(len(hboxes)))
        whole_by = "the whole frame: the head track and faces with their room are larger than the canvas"
        shaped_by = "the head track with crown room and the faces with their room, grown to the canvas's shape"
    else:
        need_w, need_h = (union[2] - union[0]) + 2 * room * face, (union[3] - union[1]) + 2 * room * face
        cx, cy = float((union[0] + union[2]) / 2), float((union[1] + union[3]) / 2)
        whole_by = "the whole frame: the faces with their room are larger than the canvas"
        shaped_by = "the faces with their room, grown to the canvas's shape"
    box, moved = in_the_canvas(need_w, need_h, cx, cy, size)
    return dict(said, box=box, set_by=whole_by if moved is None else shaped_by, moved_to_lie_inside=bool(moved))


def in_the_canvas(need_w: float, need_h: float, cx: float, cy: float, size: tuple[int, int]) -> tuple[list[int], bool | None]:
    """A box at least `need_w` by `need_h` about (cx, cy), grown the short way to the canvas's shape and moved to lie inside it:
    `(box, moved)`; `moved` is None where the box would be as large as the canvas, and the box is then the whole frame."""
    w, h = size
    box_h = int(np.ceil(max(need_h, need_w * h / w)))
    box_w = int(np.ceil(box_h * w / h))
    if box_w >= w or box_h >= h:
        return [0, 0, w, h], None
    x0, y0 = int(round(cx - box_w / 2)), int(round(cy - box_h / 2))
    inside = (min(max(x0, 0), w - box_w), min(max(y0, 0), h - box_h))
    return [inside[0], inside[1], inside[0] + box_w, inside[1] + box_h], inside != (x0, y0)


def grey_head(frame: np.ndarray, p: np.ndarray, index: dict) -> tuple[np.ndarray, int]:
    """One frame's grey head under `--cranium`: the body model's pixels over `BODY_GREY` above MID-FACE (the row halfway between the
    pose model's glabella and tip of the chin) that are joined to the face's box, as [h, w] of bool; and how many such pixels above
    mid-face are NOT joined to it (arms raised clear of the head, anything else of the figure), which are left out.

    Why mid-face and not the chin (the Gemini reviewer, measured 2026-10-22 on the plain body-model videos): above the chin line the
    neck and shoulders join the head to the rest of the figure. On lead 1156 the arms are raised above the head, and the grey joined
    to the face above the chin reached them through the neck and shoulders (642 px across on frame 1156); on second 0278 the chin is
    tucked and her shoulders lie above the chin line (the grey 435 px across, the head 160). The neck and shoulders attach below
    mid-face, so above it the skull (forehead, crown, temples, the back of the head in profile) is all that joins the face."""
    import cv2
    h, w = frame.shape[:2]
    mid = int(np.clip((int(np.floor(p[index["center_of_glabella"], 1])) + int(np.floor(p[index["tip_of_chin"], 1]))) // 2, -1, h - 1))
    lit = (frame[..., :3].max(-1) > BODY_GREY).astype(np.uint8)
    lit[mid + 1:, :] = 0
    n, labels = cv2.connectedComponents(lit, connectivity=8)
    fb = face_box(p, index)
    x0, y0 = (int(np.clip(np.floor(v), 0, m - 1)) for v, m in ((fb[0], w), (fb[1], h)))
    x1, y1 = (int(np.clip(np.ceil(v), 0, m - 1)) + 1 for v, m in ((fb[2], w), (fb[3], h)))
    on_face = set(np.unique(labels[y0:y1, x0:x1]).tolist()) - {0}
    head = np.isin(labels, list(on_face)) if on_face else np.zeros((h, w), bool)
    return head, int(lit.sum() - head.sum())


def cranium_box(points: dict, size: tuple[int, int], room: float, frames: np.ndarray, first: int = 0) -> dict:
    """The one box of a span under `--cranium`: round every frame's face and its grey head (`grey_head`) with `CRANIUM_ROOM` above
    and beside them and `CRANIUM_NECK` below the chin, in the canvas's shape, inside the canvas. With it, as figures: the frames on
    which (source frames, the span's first being `first`) the grey head above mid-face reaches the box's edge (or the picture's own edge), and each frame's own box's enlargement against the one
    box's, which is what the subject's travel over the span costs."""
    w, h = size
    idx = points["index"]
    face = float(np.median([fp.face_height(p, idx) for p in points["keypoints"]]))
    scale_r = room / HEAD_ROOM if HEAD_ROOM else 1.0
    c_room, neck = CRANIUM_ROOM * scale_r, CRANIUM_NECK * scale_r
    each, heads, left_out = [], [], []
    for p, frame in zip(points["keypoints"], frames):
        fb = face_box(p, idx)
        head, out = grey_head(frame, p, idx)
        ys, xs = np.nonzero(head)
        top = [fb[0], fb[1], fb[2]] if not len(xs) else [min(fb[0], xs.min()), min(fb[1], ys.min()), max(fb[2], xs.max() + 1)]
        chin = max(float(fb[3]), float(p[idx["tip_of_chin"], 1]))
        each.append([top[0] - c_room * face, top[1] - c_room * face, top[2] + c_room * face, chin + neck * face])
        heads.append(head)
        left_out.append(out)
    each = np.array(each, float)
    want = np.concatenate([each[:, :2].min(0), each[:, 2:].max(0)])
    need_w, need_h = float(want[2] - want[0]), float(want[3] - want[1])
    least_w = w / CRANIUM_MOST                                          # the box no narrower than this: enlarged at most CRANIUM_MOST
    capped = bool(max(need_w, need_h * w / h) < least_w)
    free = in_the_canvas(need_w, need_h, float((want[0] + want[2]) / 2), float((want[1] + want[3]) / 2), size)[0]
    uncapped = w / (free[2] - free[0])
    box, moved = in_the_canvas(max(need_w, least_w), need_h, float((want[0] + want[2]) / 2), float((want[1] + want[3]) / 2), size)
    alone = [w / (b[2] - b[0]) for b in (in_the_canvas(max(float(e[2] - e[0]), least_w), float(e[3] - e[1]), float((e[0] + e[2]) / 2),
                                                       float((e[1] + e[3]) / 2), size)[0] for e in each)]
    x0, y0, x1, y1 = box
    edge, picture_edge, none = [], [], []
    for k, head in enumerate(heads):
        ys, xs = np.nonzero(head)
        if not len(xs):
            none.append(first + k)
            continue
        if xs.min() <= x0 or ys.min() <= y0 or xs.max() >= x1 - 1 or ys.max() >= y1 - 1:
            edge.append(first + k)
        if xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1:
            picture_edge.append(first + k)
    one = w / (x1 - x0)
    return {"faces": [round(float(v), 1) for v in (each[:, 0].min(), each[:, 1].min(), each[:, 2].max(), each[:, 3].max())],
            "face_height_px_median": round(face, 1), "room_in_face_heights": room,
            "cranium": {"room_above_and_beside_in_face_heights": round(c_room, 3), "below_the_chin_in_face_heights": round(neck, 3),
                        "body_grey": BODY_GREY,
                        "frames_the_grey_head_reaches_the_box_edge": edge,
                        "frames_the_grey_head_reaches_the_picture_edge": picture_edge,
                        "frames_with_no_grey_head_joined_to_the_face": none,
                        "grey_above_mid_face_not_joined_to_the_face_px_least_median_most": [int(np.min(left_out)), int(np.median(left_out)), int(np.max(left_out))],
                        "each_frame_alone_would_enlarge_least_median_most": [round(float(np.min(alone)), 3), round(float(np.median(alone)), 3), round(float(np.max(alone)), 3)],
                        "the_one_box_enlarges": round(one, 3),
                        "held_to_the_most": {"capped": capped, "most": CRANIUM_MOST, "would_enlarge_without_it": round(float(uncapped), 3)},
                        "the_travel_keeps_of_the_median_frame_enlargement": round(one / float(np.median(alone)), 3),
                        "is": "figures, not a verdict: the box does not follow the head; a head that travels lowers the one box's enlargement"},
            "box": box, "moved_to_lie_inside": bool(moved),
            "set_by": ("the whole frame: the grey head and faces with their room are larger than the canvas" if moved is None else
                       "the grey head above mid-face and the faces with CRANIUM_ROOM, CRANIUM_NECK below the chin, grown to the canvas's shape"
                       + (f", then to CRANIUM_MOST ({CRANIUM_MOST:g} times at most)" if capped else ""))}


def cranium_lines(c: dict) -> list[str]:
    """What a cranium box says, as lines: where the grey head reaches the box's or the picture's edge, and the travel's cost. Figures."""
    def frames(fs: list) -> str:
        return f"{len(fs)} frame(s): {fs[0]}..{fs[-1]}" if len(fs) > 3 else (", ".join(str(f) for f in fs) or "none")
    least, median, most = c["each_frame_alone_would_enlarge_least_median_most"]
    return [f"the grey head reaches the box's edge on {frames(c['frames_the_grey_head_reaches_the_box_edge'])}; the picture's own edge on "
            f"{frames(c['frames_the_grey_head_reaches_the_picture_edge'])}; no grey head joined to the face on "
            f"{frames(c['frames_with_no_grey_head_joined_to_the_face'])}",
            f"travel: each frame's own box would enlarge {least:.3f} / {median:.3f} / {most:.3f} times (least / median / most); the one box "
            f"enlarges {c['the_one_box_enlarges']:.3f}, {c['the_travel_keeps_of_the_median_frame_enlargement']:.3f} of the median frame's "
            "(figures: the box does not follow the head)"
            + (f"; held to CRANIUM_MOST, {c['held_to_the_most']['most']:g}: it would enlarge {c['held_to_the_most']['would_enlarge_without_it']:.3f} "
               "times without it" if (c.get("held_to_the_most") or {}).get("capped") else "")]


def discover_head_track(points_path: str, video_path: str | None = None, video_first: int | None = None) -> tuple[str | None, int | None, str | None]:
    """Find a sibling `*_head_track_*.mkv` beside the points' box-mask video (`<points>.json`) or beside a
    body-model video (`*_mesh_*`). Returns `(path, first_frame, found_by)` or `(None, None, None)`."""
    record = Path(points_path).with_suffix(".json")
    if record.is_file():
        try:
            said = json.loads(record.read_text())
        except ValueError:
            said = {}
        box = (said.get("made_from") or {}).get("box")
        if isinstance(box, dict) and box.get("mask_video") and box.get("its_first_frame_is_source_frame") is not None:
            mv = Path(str(box["mask_video"]))
            mf = int(box["its_first_frame_is_source_frame"])
            if "_head_track" in mv.name and mv.is_file():
                return str(mv), mf, "the points' box mask (already a head track)"
            if "_track_" in mv.name:
                cand = mv.with_name(mv.name.replace("_track_", "_head_track_", 1))
                if cand.is_file():
                    return str(cand), mf, "beside the points' box mask"
    if video_path and video_first is not None:
        vp = Path(str(video_path))
        if "_mesh_" in vp.name:
            cand = vp.with_name(vp.name.replace("_mesh_", "_head_track_", 1)).with_suffix(".mkv")
            if cand.is_file():
                return str(cand), int(video_first), "beside the body-model video"
    return None, None, None


def points_say(points_path: str) -> dict | str:
    """What a points file's own record says it was made from: who made it, when, from which video and which track's
    box (`bench/pose_along_shot.py` writes it beside the file)."""
    record = Path(points_path).with_suffix(".json")
    if not record.is_file():
        return f"no record beside the points file ({record.name}): which track they follow is not known here"
    try:
        said = json.loads(record.read_text())
    except ValueError:
        return f"the record beside the points file ({record.name}) could not be read"
    return {key: said[key] for key in ("made_by", "made_at", "model", "made_from") if key in said}


def cut(frames: np.ndarray, box: list[int]) -> np.ndarray:
    """The frames [n, h, w, 3] cut to the box and enlarged to their own size."""
    h, w = frames.shape[1:3]
    x0, y0, x1, y1 = box
    if (x0, y0, x1, y1) == (0, 0, w, h):
        return frames.copy()
    return np.stack([np.asarray(Image.fromarray(f[y0:y1, x0:x1]).resize((w, h), Image.LANCZOS)) for f in frames])


def fit_by_lag(written: np.ndarray, made: np.ndarray) -> dict[int, float]:
    """The mean difference, in grey levels, of each written frame from the picture made for the frame that many
    frames away. Smaller is nearer."""
    out = {}
    a, b = written.astype(np.int16), made.astype(np.int16)
    for lag in (-1, 0, 1):
        x, y = a[max(0, lag):len(a) - max(0, -lag)], b[max(0, -lag):len(b) - max(0, lag)]   # a file one frame late (lag 1) shows frame f the picture of f - 1
        out[lag] = float(np.abs(x - y).mean()) if len(x) else float("nan")
    return out


def read_back(video: Path, made: np.ndarray) -> dict:
    """The written video against the pictures made for it: how each frame fits, by slip, and a verdict."""
    n, h, w = made.shape[:3]
    got = np.stack(list(stream(str(video), (w, h), 0, n, pix="rgb24")))
    if len(got) != n or probe(str(video))[2] != n:
        return {"lags": {0: float("nan")}, "frames_read": len(got), "slipped": False, "cannot_tell": False, "passed": False}
    lags = fit_by_lag(got, made)
    other = [v for k, v in lags.items() if k != 0 and not np.isnan(v)]
    ahead = (min(other) - lags[0]) if other else float("nan")          # how much nearer a written frame is to its own picture than to a neighbour's
    slipped = bool(other and ahead <= -SLIP_LEVELS)
    return {"lags": lags, "frames_read": len(got), "slipped": slipped, "cannot_tell": bool(not other or abs(ahead) < SLIP_LEVELS),
            "passed": bool(not slipped and lags[0] <= ENCODE_LEVELS)}


def sheet_picks(reference: np.ndarray, box: list[int], first: int, points: dict, head_track=None) -> dict:
    """The frames the sheet shows (`_lib/pick_frames.py`): where the face's points, the head track (its pixels and its
    place, when it is a mask a frame), or the picture inside the box or over the whole frame change; either side of
    each, and one a second. A head track given as boxes, or none, adds nothing."""
    n, h, w = reference.shape[:3]
    series = {**picker.points_series(points),
              "picture_in_the_box": {"values": picker.difference_in_boxes(reference, [tuple(box)] * n), "rules": ("peak",)},
              "picture_whole": {"values": picker.difference_in_boxes(reference, [(0, 0, w, h)] * n), "rules": ("peak",)}}
    track = np.asarray(head_track) if head_track is not None else None
    if track is not None and track.ndim == 3 and len(track) == n:
        series["head_track_px"] = [int((m > 0).sum()) for m in track]
        series["head_track_place"] = picker.centroids(track > 0)
    return picker.sheet_picks(first, n, series)


def sheet(path: Path, reference: np.ndarray, cut_frames: np.ndarray, box: list[int], first: int, scale: float, picked: dict | None = None) -> list[int]:
    """The picked frames (`sheet_picks`; first, last and one a second when not given) in two panels: the reference with
    the box drawn on it, and the cut picture, each with why it was picked."""
    n, h, w = reference.shape[:3]
    picked = picked or picker.sheet_picks(first, n)
    why = picker.first_reasons(picked)
    picks = [f - first for f in picker.frames_of(picked)]
    tw = 560
    th = int(round(tw * h / w))
    rows = []
    for f in picks:
        whole = Image.fromarray(reference[f]).resize((tw, th), Image.LANCZOS)
        k = tw / w
        ImageDraw.Draw(whole).rectangle([box[0] * k, box[1] * k, box[2] * k - 1, box[3] * k - 1], outline=(255, 255, 0), width=2)
        row = Image.new("RGB", (tw * 2, th))
        row.paste(whole, (0, 0))
        row.paste(Image.fromarray(cut_frames[f]).resize((tw, th), Image.LANCZOS), (tw, 0))
        ImageDraw.Draw(row).text((6, 6), f"source frame {first + f}: the reference with the box | the box enlarged {scale:.2f} times", fill=(255, 255, 0))
        ImageDraw.Draw(row).text((6, 20), f"picked: {why.get(first + f, '')}", fill=(255, 255, 255))
        rows.append(row)
    out = Image.new("RGB", (tw * 2, th * len(rows)))
    for i, r in enumerate(rows):
        out.paste(r, (0, i * th))
    out.save(path, quality=90)
    return [first + f for f in picks]


def run(video: str, points_path: str, first: int, count: int, out: Path, by: str, *, names: list[str] | None = None,
        video_first: int | None = None, points_first: int | None = None, room: float | None = None,
        head_track: str | Path | np.ndarray | list | None = None, head_track_first: int | None = None,
        no_head_track: bool = False, head_track_how: str | None = None, fit: bool = False, cranium: bool = False,
        box: list[int] | None = None, say=print) -> dict:
    """Find the box, cut, write, read back, record. Returns the record; `passed` is False when the read-back fails.

    `cranium`: the box round the body model's grey head (`cranium_box`), read off this video, which must be the body model; no head
    track is read. `box`: cut by this box as given (`[x0, y0, x1, y1]` on the canvas), found by nothing here."""
    if no_head_track and head_track is not None:
        raise SystemExit("cannot give both --head-track and --no-head-track")
    if cranium and (fit or head_track is not None or box is not None):
        raise SystemExit("--cranium reads the box off the body model's own grey head: not with --fit, --head-track or --box "
                         "(a source is cut by the box its body model's cut found, with --box)")
    points, pf = faces.rows_of(points_path, names, first, count, points_first, whole_face=False)
    w, h, have = probe(video)
    own = (w, h)
    skip = first - ((0 if fit else first) if video_first is None else video_first)
    if skip < 0 or skip + count > have:
        raise SystemExit(f"{video} has {have} frames from source frame {first - skip}; frames {first} to {first + count - 1} were asked for")
    if fit:
        if not points["size"]:
            raise SystemExit(f"{points_path} does not say the size of the picture its points were made on: a source cannot be fitted to it")
        w, h = points["size"]
    elif points["size"] and tuple(points["size"]) != (w, h):
        raise SystemExit(f"the points were made on a picture of {points['size']} and the reference video is {(w, h)}: they are not on one canvas "
                         "(give --fit for the original itself)")
    if box is not None:
        box = [int(v) for v in box]
        if len(box) != 4 or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
            raise SystemExit(f"--box {box} is not a box inside the {w} by {h} canvas")
    ht_masks, ht_info = None, None
    if not no_head_track and not cranium and box is None:
        if isinstance(head_track, (str, Path)):
            ht_path = str(head_track)
            if not Path(ht_path).is_file():
                raise SystemExit(f"--head-track {ht_path} is not a file")
            ht_first = (first - skip) if head_track_first is None else int(head_track_first)
            _, _, have_ht = probe(ht_path)
            if first - ht_first < 0 or first - ht_first + count > have_ht:
                raise SystemExit(f"{ht_path} has {have_ht} frames from source frame {ht_first}; frames {first} to {first + count - 1} were asked for")
            ht_masks, _ = read_mask(ht_path, (w, h), ht_first, first, count)
            ht_info = {"mask_video": ht_path, "its_first_frame_is_source_frame": ht_first,
                       "found_by": head_track_how or "--head-track", "frames_with_a_mask": int(ht_masks.any(axis=(1, 2)).sum())}
        elif head_track is not None:
            ht_masks = head_track
            ht_info = {"found_by": head_track_how or "given array", "frames_with_a_mask": int(len(head_boxes_of(head_track)))}
        else:
            ht_path, ht_first, ht_how = discover_head_track(points_path, None if fit else video, None if fit else (first - skip))
            if ht_path is not None and ht_first is not None:
                _, _, have_ht = probe(ht_path)
                if 0 <= first - ht_first and first - ht_first + count <= have_ht:
                    ht_masks, _ = read_mask(ht_path, (w, h), ht_first, first, count)
                    ht_info = {"mask_video": ht_path, "its_first_frame_is_source_frame": ht_first,
                               "found_by": ht_how, "frames_with_a_mask": int(ht_masks.any(axis=(1, 2)).sum())}
    room = HEAD_ROOM if room is None else float(room)
    reference = np.stack(list(stream(video, (w, h), skip, count, vf=FIT.format(w=w, h=h) if fit else "", pix="rgb24")))
    if box is not None:
        found = {**{k: v for k, v in one_box(points, (w, h), room).items() if k in ("faces", "face_height_px_median", "room_in_face_heights")},
                 "box": box, "set_by": "given (--box), as it is", "moved_to_lie_inside": False}
    elif cranium:
        found = cranium_box(points, (w, h), room, reference, first)
    else:
        found = one_box(points, (w, h), room, head_track=ht_masks)
    box = found["box"]
    scale = [w / (box[2] - box[0]), h / (box[3] - box[1])]
    made = cut(reference, box)
    out.parent.mkdir(parents=True, exist_ok=True)
    marks.write_video(out, made)
    back = read_back(out, made)
    lags = back["lags"]
    sheet_path = out.with_name(out.stem + "_sheet.jpg")
    record = {
        "video": out.name, "made_by": by, "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "writer_stamp": writer_stamp(__file__),
        "frames": [first, first + count - 1], "size": [w, h],
        "made_from": {"reference_video": str(video), "its_first_frame_is_source_frame": first - skip, "points": str(points_path),
                      "their_first_row_is_source_frame": pf, "the_points_own_record_says": points_say(points_path),
                      **({"head_track": ht_info} if ht_info is not None else {}),
                      "fitted": (f"a source of {own[0]} by {own[1]}, fitted to the canvas as the loader fits it before the cut" if fit else "no: already the canvas's size")},
        "box": box, "box_is": "[x0, y0, x1, y1] in the canvas's pixels, the far side not included; one box for every frame",
        "box_set_by": found["set_by"], "moved_to_lie_inside_the_canvas": found["moved_to_lie_inside"],
        "scale": [round(v, 4) for v in scale], "scale_is": "how many times the box is enlarged, across and down",
        "a_point_of_the_canvas_lands_at": "((x - x0) * scale across, (y - y0) * scale down)",
        "faces_over_the_span": found["faces"], "face_height_px_median": found["face_height_px_median"],
        "face_height_px_in_this_video": round(found["face_height_px_median"] * scale[1], 1),
        **({"heads_over_the_span": found["heads"], "head_height_px_median": found["head_height_px_median"],
            "head_height_px_in_this_video": round(found["head_height_px_median"] * scale[1], 1),
            "crown_room_in_face_heights": found["crown_room_in_face_heights"],
            "head_track_room_in_face_heights": found["head_track_room_in_face_heights"]} if "heads" in found else {}),
        **({"cranium": found["cranium"]} if "cranium" in found else {}),
        "room_in_face_heights": found["room_in_face_heights"], "room_set_by": "--room" if room != HEAD_ROOM else "HEAD_ROOM, a starting value",
        "not_known": ["whether the points are on her: open the sheet",
                      "what a model makes of a reference that no longer lies over the source pixel for pixel: a render's question"],
        "read_back": {"frames_read": back["frames_read"],
                      "grey_levels_from_the_picture_made_by_frames_of_slip": {str(k): (None if np.isnan(v) else round(v, 3)) for k, v in lags.items()},
                      "a_neighbour_fits_better": back["slipped"], "no_neighbour_can_be_told_from_the_frame": back["cannot_tell"], "passed": back["passed"]},
    }
    picked = sheet_picks(reference, box, first, points, ht_masks)
    record["sheet"] = {"file": sheet_path.name, "frames": sheet(sheet_path, reference, made, box, first, scale[0], picked),
                       "why": {str(k): v for k, v in picker.first_reasons(picked).items()}, "dropped": picked["dropped"],
                       "cap": picked["cap"], "over_the_cap": picked["over_the_cap"], "looked_at_by": ""}
    carried = out.with_name(out.stem + "_points.npz")
    np.savez_compressed(carried, keypoints=((points["keypoints"] - box[:2]) * scale).astype(np.float32), scores=points["scores"].astype(np.float32),
                        names=np.array(points["names"]), first_frame=np.array(first), size=np.array([w, h]),
                        **npz_writer_fields(__file__))
    record["points_in_this_picture"] = {"file": carried.name, "which_are": "the span's points, each moved as a point of the canvas is; scores and names as given"}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    say(f"faces over frames {first} to {first + count - 1} lie in {found['faces']}; face {found['face_height_px_median']:.0f} px tall at the median; "
        + (f"the grey head with {found['cranium']['room_above_and_beside_in_face_heights']:g} face heights above and beside, "
           f"{found['cranium']['below_the_chin_in_face_heights']:g} below the chin" if "cranium" in found else f"room {room:g} face heights a side"))
    if "heads" in found:
        say(f"heads over frames {first} to {first + count - 1} lie in {found['heads']} ({found['head_track_frames']} frame(s) on the head track; "
            f"head {found['head_height_px_median']:.0f} px tall at the median; crown room {found['crown_room_in_face_heights']:g}, "
            f"side/bottom room {found['head_track_room_in_face_heights']:g} face heights)")
    say(f"box {box} ({box[2] - box[0]} by {box[3] - box[1]}), set by {found['set_by']}" + ("; moved to lie inside the canvas" if found["moved_to_lie_inside"] else ""))
    if "cranium" in found:
        for line in cranium_lines(found["cranium"]):
            say(line)
    say(f"enlarged {scale[0]:.3f} times across and {scale[1]:.3f} down: the face is {record['face_height_px_in_this_video']:.0f} px tall in the new picture")
    say("read back: %s frames; grey levels from the picture made for the frame %s, for the frame before %s, for the frame after %s" % (
        back["frames_read"], *(("n/a" if np.isnan(lags.get(k, float("nan"))) else f"{lags[k]:.2f}") for k in (0, 1, -1))))
    say(("PASSED" if back["passed"] else "FAILED") + ": the video " + ("holds" if back["passed"] else "does NOT hold") + " the cut frames in their places"
        + ("; the picture changes too little between frames here for a slip to be told" if back["cannot_tell"] and back["passed"] else ""))
    say(f"wrote {out}, {out.with_suffix('.json').name}, {carried.name} and {sheet_path.name}: look at the sheet before the video is used")
    record["passed"] = back["passed"]
    return record


def main() -> int:
    a = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    a.add_argument("--video", required=True, help="the reference video, at the canvas's size")
    a.add_argument("--points", required=True, help="the original's points along the shot (bench/pose_along_shot.py)")
    a.add_argument("--first", type=int, required=True, help="the span's first source frame")
    a.add_argument("--count", type=int, required=True)
    a.add_argument("--out", required=True, help="the cut video to write; its record and sheet go beside it")
    a.add_argument("--by", required=True, help="who is making it")
    a.add_argument("--room", type=float, help=f"room round the faces on every side, in face heights; {HEAD_ROOM:g} when left out")
    a.add_argument("--head-track", help="the subject's head-track mask video; found beside the points' box mask or body-model video when left out")
    a.add_argument("--head-track-first", type=int, help="the source frame of the head-track video's frame 0; the reference video's first when left out")
    a.add_argument("--no-head-track", action="store_true", help="size the box from the faces alone, even when a head track is beside the points or video")
    a.add_argument("--cranium", action="store_true", help="the box round the body model's grey head above mid-face and the face, "
                   "CRANIUM_ROOM above and beside, CRANIUM_NECK below the chin; read off --video, which is the body model; no head track read")
    a.add_argument("--box", help="X0,Y0,X1,Y1: cut by this box as given, on the canvas (a source by the box its body model's --cranium cut found)")
    a.add_argument("--names", help="a json list of the points' names, for a points file made before they were kept")
    a.add_argument("--video-first", type=int, help="the source frame of the reference video's frame 0; the span's first when left out")
    a.add_argument("--points-first", type=int, help="the source frame of the points file's row 0; the file's own when it says")
    a.add_argument("--fit", action="store_true", help="the video is the original itself: fit it to the points' canvas as the loader does, then cut")
    v = a.parse_args()
    names = json.loads(Path(v.names).read_text()) if v.names else None
    try:
        box = [int(x) for x in v.box.split(",")] if v.box else None
    except ValueError:
        raise SystemExit(f"--box takes X0,Y0,X1,Y1 in whole pixels; got {v.box!r}") from None
    rec = run(v.video, v.points, v.first, v.count, Path(v.out), v.by, names=names, video_first=v.video_first, points_first=v.points_first, room=v.room,
              head_track=v.head_track, head_track_first=v.head_track_first, no_head_track=v.no_head_track, fit=v.fit,
              cranium=v.cranium, box=box)
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
