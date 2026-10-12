#!/usr/bin/env python3
"""Cut a reference video to ONE box round a shot's head and scale it back up to the canvas.

    python bench/crop_reference_to_head.py --video REFERENCE.mp4 --points SHOT_points.npz --first 1039 --count 22 \\
        --out <share>/Video/<video>/<who>/second_mesh_with_lids_head_crop_1039.mp4 --by <session> \\
        [--room 1.2] [--video-first N] [--points-first N] [--names NAMES.json] [--fit]

A reference video drawn at the canvas's size shows a small face small: on a face a few dozen pixels tall an
eye drawn at its true size is a patch a dozen pixels wide, and no line in it can be told from the next
(`bench/draw_face_points.py`, its first sheet). The reference does not have to be the whole frame. This takes a
reference video at the canvas's size (a body model with lid marks, or with the face drawn) and the pose model's
points along the shot (`bench/pose_along_shot.py`), and writes the same frames cut to one box round her head
and enlarged to the canvas, so the face fills the picture.

ONE BOX FOR THE WHOLE SPAN, so nothing shakes: the head moves inside the picture as it moves in the shot, and
the picture's edges stand still. The box is:

- round the FACES of every frame asked for: each frame's face is the box round its lash lines, the glabella,
  the tip of the nose and the tip of the chin (the points `bench/_lib/face_points.py` reads, none of the ones the
  model scores low), and the box holds all of them;
- with ROOM on every side: `HEAD_ROOM` times the shot's median face height, or `--room`. The model's points stop
  at the brows and the chin, and a head is taller than a face;
- in the CANVAS'S SHAPE, grown the short way to reach it, then moved to lie inside the canvas if it hung over
  an edge. A box that would be larger than the canvas is the whole frame, and the tool says so.

It writes the video, a record beside it (`.json`: the box, the scale, the frames, what it was made from, where a
point of the canvas lands in the new picture), the span's points carried into the new picture
(`<out>_points.npz`, as `bench/pose_along_shot.py` writes one) and a sheet of six frames, the reference with the
box drawn on it beside the cut picture, to look at before the video is used.

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
  box round the wrong person: open the sheet.
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
from _lib.frames import FIT, probe, stream  # noqa: E402

import draw_face_points as faces  # noqa: E402  the points loader: one copy
import draw_lid_marks as marks  # noqa: E402  the writer: one copy

#: Room left round the faces on every side, in face heights (glabella to chin). Reasoned, and a starting value: a
#: head is a little under twice that length from crown to chin, so with this much room a head that holds still
#: sits in a box about twice its own height. No render has been made from a reference cut this way.
HEAD_ROOM = 1.2
#: The written video must sit within this many grey levels of the picture made for it, at the mean. Reasoned: the
#: encode moves a picture by about a level; a wrong frame of a moving head moves it by many.
ENCODE_LEVELS = 6.0
#: How much nearer its own frame than a neighbour a written frame must be for the frames to be called in place, in
#: grey levels at the mean; under it either way nothing moved enough to tell.
SLIP_LEVELS = 0.25


def face_box(p: np.ndarray, index: dict) -> np.ndarray:
    """One frame's face as [x0, y0, x1, y1]: round the lash lines, the glabella, the nose's tip and the chin's."""
    ids = [i for n, i in index.items() if n.endswith("_lash_line")] + [index[n] for n in ("center_of_glabella", "tip_of_nose", "tip_of_chin")]
    pts = p[ids]
    return np.concatenate([pts.min(0), pts.max(0)])


def one_box(points: dict, size: tuple[int, int], room: float) -> dict:
    """The one box of a span: round every frame's face, with room, in the canvas's shape, inside the canvas."""
    w, h = size
    idx = points["index"]
    each = np.array([face_box(p, idx) for p in points["keypoints"]])
    union = np.concatenate([each[:, :2].min(0), each[:, 2:].max(0)])
    face = float(np.median([fp.face_height(p, idx) for p in points["keypoints"]]))
    need_w, need_h = (union[2] - union[0]) + 2 * room * face, (union[3] - union[1]) + 2 * room * face
    box_h = int(np.ceil(max(need_h, need_w * h / w)))
    box_w = int(np.ceil(box_h * w / h))
    said = {"faces": [round(float(v), 1) for v in union], "face_height_px_median": round(face, 1), "room_in_face_heights": room}
    if box_w >= w or box_h >= h:
        return dict(said, box=[0, 0, w, h], set_by="the whole frame: the faces with their room are larger than the canvas", moved_to_lie_inside=False)
    cx, cy = (union[0] + union[2]) / 2, (union[1] + union[3]) / 2
    x0, y0 = int(round(cx - box_w / 2)), int(round(cy - box_h / 2))
    inside = (min(max(x0, 0), w - box_w), min(max(y0, 0), h - box_h))
    return dict(said, box=[inside[0], inside[1], inside[0] + box_w, inside[1] + box_h], set_by="the faces with their room, grown to the canvas's shape",
                moved_to_lie_inside=bool(inside != (x0, y0)))


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


def sheet(path: Path, reference: np.ndarray, cut_frames: np.ndarray, box: list[int], first: int, scale: float) -> list[int]:
    """Six frames in two panels: the reference with the box drawn on it, and the cut picture."""
    n, h, w = reference.shape[:3]
    picks = sorted(set(np.linspace(0, n - 1, 6).astype(int).tolist()))
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
        rows.append(row)
    out = Image.new("RGB", (tw * 2, th * len(rows)))
    for i, r in enumerate(rows):
        out.paste(r, (0, i * th))
    out.save(path, quality=90)
    return [first + f for f in picks]


def run(video: str, points_path: str, first: int, count: int, out: Path, by: str, *, names: list[str] | None = None,
        video_first: int | None = None, points_first: int | None = None, room: float | None = None, fit: bool = False, say=print) -> dict:
    """Find the box, cut, write, read back, record. Returns the record; `passed` is False when the read-back fails."""
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
    room = HEAD_ROOM if room is None else float(room)
    found = one_box(points, (w, h), room)
    box = found["box"]
    scale = [w / (box[2] - box[0]), h / (box[3] - box[1])]
    reference = np.stack(list(stream(video, (w, h), skip, count, vf=FIT.format(w=w, h=h) if fit else "", pix="rgb24")))
    made = cut(reference, box)
    out.parent.mkdir(parents=True, exist_ok=True)
    marks.write_video(out, made)
    back = read_back(out, made)
    lags = back["lags"]
    sheet_path = out.with_name(out.stem + "_sheet.jpg")
    record = {
        "video": out.name, "made_by": by, "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frames": [first, first + count - 1], "size": [w, h],
        "made_from": {"reference_video": str(video), "its_first_frame_is_source_frame": first - skip, "points": str(points_path),
                      "their_first_row_is_source_frame": pf,
                      "fitted": (f"a source of {own[0]} by {own[1]}, fitted to the canvas as the loader fits it before the cut" if fit else "no: already the canvas's size")},
        "box": box, "box_is": "[x0, y0, x1, y1] in the canvas's pixels, the far side not included; one box for every frame",
        "box_set_by": found["set_by"], "moved_to_lie_inside_the_canvas": found["moved_to_lie_inside"],
        "scale": [round(v, 4) for v in scale], "scale_is": "how many times the box is enlarged, across and down",
        "a_point_of_the_canvas_lands_at": "((x - x0) * scale across, (y - y0) * scale down)",
        "faces_over_the_span": found["faces"], "face_height_px_median": found["face_height_px_median"],
        "face_height_px_in_this_video": round(found["face_height_px_median"] * scale[1], 1),
        "room_in_face_heights": found["room_in_face_heights"], "room_set_by": "--room" if room != HEAD_ROOM else "HEAD_ROOM, a starting value",
        "not_known": ["whether the points are on her: open the sheet",
                      "what a model makes of a reference that no longer lies over the source pixel for pixel: a render's question"],
        "read_back": {"frames_read": back["frames_read"],
                      "grey_levels_from_the_picture_made_by_frames_of_slip": {str(k): (None if np.isnan(v) else round(v, 3)) for k, v in lags.items()},
                      "a_neighbour_fits_better": back["slipped"], "no_neighbour_can_be_told_from_the_frame": back["cannot_tell"], "passed": back["passed"]},
        "sheet": {"file": sheet_path.name, "frames": sheet(sheet_path, reference, made, box, first, scale[0]), "looked_at_by": ""},
    }
    carried = out.with_name(out.stem + "_points.npz")
    np.savez_compressed(carried, keypoints=((points["keypoints"] - box[:2]) * scale).astype(np.float32), scores=points["scores"].astype(np.float32),
                        names=np.array(points["names"]), first_frame=np.array(first), size=np.array([w, h]))
    record["points_in_this_picture"] = {"file": carried.name, "which_are": "the span's points, each moved as a point of the canvas is; scores and names as given"}
    out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    say(f"faces over frames {first} to {first + count - 1} lie in {found['faces']}; face {found['face_height_px_median']:.0f} px tall at the median; "
        f"room {room:g} face heights a side")
    say(f"box {box} ({box[2] - box[0]} by {box[3] - box[1]}), set by {found['set_by']}" + ("; moved to lie inside the canvas" if found["moved_to_lie_inside"] else ""))
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
    a.add_argument("--names", help="a json list of the points' names, for a points file made before they were kept")
    a.add_argument("--video-first", type=int, help="the source frame of the reference video's frame 0; the span's first when left out")
    a.add_argument("--points-first", type=int, help="the source frame of the points file's row 0; the file's own when it says")
    a.add_argument("--fit", action="store_true", help="the video is the original itself: fit it to the points' canvas as the loader does, then cut")
    v = a.parse_args()
    names = json.loads(Path(v.names).read_text()) if v.names else None
    rec = run(v.video, v.points, v.first, v.count, Path(v.out), v.by, names=names, video_first=v.video_first, points_first=v.points_first, room=v.room,
              fit=v.fit)
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
