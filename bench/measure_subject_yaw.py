#!/usr/bin/env python3
"""Which way a person faces, frame by frame, in a render and in the source it was made from: a number for "did he turn".

The masked lane's question on a moving shot is whether the replaced subject
moves as the original did. On 2026-10-05 it was answered by reading five
frames of a contact sheet per render
(`bench/results/2026-10-05_masked_v2v_motion_arms.md`). This measures it.

**What it buys** (card `build-turn-metric`, the owner's ask): a number per
clip in place of a reading by eye, so that the per-step half of the same
card can say at which sigma the facing is decided, and a knot list for a
distill can be placed there. It applies to a shot where the subject turns
and the source's mask is kept. It is accepted when it puts the clips already
judged by eye in the same order, and the record says where it does not.

**The measurement.** Core's SAM 3D Body predictor runs on the subject's box
in each sampled frame and returns 3D keypoints in the camera's frame. The
yaw is the direction of the line from the right shoulder to the left
shoulder in the horizontal plane: 0 degrees facing the camera, 90 side-on,
180 with the back to it. The hips give a second reading of the same thing,
recorded beside it. Nothing about the model's own rotation parameters is
used: the root rotation it reports is the rig's, not the body's
(`pred_global_rots[0]` is the identity on every frame probed), and its
`global_rot` Euler angles wrap where the shoulder line does not.

**The box** is the source's kept mask (a file the removed kept-mask store,
`mask_store.py`, wrote under the output folder's `masks/` until 2026-10-10;
no run writes one any more, so this tool reads only masks kept before that
day), fitted to the render's canvas by the same centre
crop the song node gives the frames (`comfy.utils.common_upscale`,
"center"). A masked render regenerates the subject inside that mask, so the
source's box finds the new subject too. A window with no kept mask can give
the box two other ways: `--mask-video`, a mask saved as a video from the
window's first frame (what a Subject Track graph writes when its mask is
sent to a video), or `--find-box`, which asks the body model itself and is
for a clip with one person in it. `--find-box` invents a person in an empty
frame, because the body model always returns a pose; a mask does not.

**More than one shot.** The source's cuts are found with the Subject
Track's own cut finder and written as `shots`. A turn is a property of one
shot, so the comparisons are made per shot (`by_shot`), and `--reanalyse`
with `--shot` reads one.

**The head beside the shoulders.** Each reading also carries the head's
facing, from the line between the ears, and each comparison the chin's
lift, from the nose against that line. A singer who turns the head to a
microphone faces one way with the body and another with the head, and the
eye reads the head. Both are reported under `head`; the verdict stays the
shoulders', the reading that was calibrated.

**Per clip, against the source over the same frames:** the yaw at the end
of the shot (the circular mean of its last `END_SAMPLES` samples), its
difference from the source's, the mean difference over the shot, the first
sampled frame from which the two stay apart by more than the tolerance, the
largest turn (how far the yaw gets from where the shot began, which tells a
part-turn that comes back from no turn at all), and a verdict: `holds` when
the end difference is within the tolerance, `fails` otherwise. With `--eye` the clips' by-eye verdicts are read from a JSON
(`bench/turn_metric_eye_verdicts.json`) and the output says whether the end
differences put them in the same order (every "yes" below every "partial"
below every "no").

Not measured: anything but the facing. A clip can hold the turn and lose
the look, the lip sync or the hands; those stay with the eye.

In its own process, not through a server. Masked it runs on the CPU, which
is how the 2026-10-05 calibration ran. On the card ask whoever holds it
first; `bench/_lib::server_memory_mode` is called for that case.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/measure_subject_yaw.py \\
        --source <video> --start 112 --rate 24 --mask <kept mask .npz> \\
        --shot 237 279 --canvas 1344x768 --clips-dir <dir> \\
        --eye bench/turn_metric_eye_verdicts.json --eye-set band_turn --out <record.json>
    <comfy venv python> bench/measure_subject_yaw.py --reanalyse <record.json> [--shot F L] --out <record.json>
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _lib import REPO, bootstrap, needs, server_memory_mode  # noqa: E402

# inherited: Meta's MHR70 keypoint order (sam_3d_body/metadata/mhr70.py),
# which core's predictor returns unchanged.
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 5, 6, 9, 10
#: The body joints kept per frame for the motion metric
#: (`bench/measure_subject_motion.py`), by name, from the same table. Fingers
#: and toes are left out: at a clip's size they are the model's guess.
BODY_JOINTS = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_hip": 9, "right_hip": 10, "left_knee": 11, "right_knee": 12,
    "left_ankle": 13, "right_ankle": 14, "right_wrist": 41, "left_wrist": 62,
}

# reasoned: half way from facing the source's way to side-on. A render that
# ends within it faces where the source faces to the eye; set before the
# calibration was run, not fitted to it.
TOLERANCE_DEGREES = 45.0
# reasoned: the last few samples of the shot, so one bad frame does not
# decide the end.
END_SAMPLES = 4
# reasoned: every second frame draws a half-second turn with six samples.
EVERY = 2
# reasoned: the joints span the skeleton, not the body; a fifth of the span
# on each side takes in the head's top, the hands and the feet.
AUTO_BOX_MARGIN = 0.2

RELEASES = {"vith": "sam_3d_body_vith.safetensors", "dinov3": "sam_3d_body_dinov3.safetensors"}
ORDER = {"yes": 0, "partial": 1, "no": 2}


# ----------------------------------------------------------------- the angles

def wrap(degrees: float) -> float:
    """An angle in (-180, 180]."""
    d = (float(degrees) + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def apart(a: float, b: float) -> float:
    """How far apart two angles are, 0 to 180."""
    return abs(wrap(a - b))


def circular_mean(angles) -> float | None:
    values = [a for a in angles if a is not None]
    if not values:
        return None
    s = sum(math.sin(math.radians(a)) for a in values)
    c = sum(math.cos(math.radians(a)) for a in values)
    return wrap(math.degrees(math.atan2(s, c)))


def yaw_of(keypoints_3d, left: int, right: int) -> float:
    """The facing angle from a left and right joint: 0 toward the camera, 180 away.

    `keypoints_3d` is [K, 3] in the camera's frame, x to the image's right
    and z away from the camera. Facing the camera, a person's left side is
    on the image's right, so the left-minus-right vector points along +x.
    """
    k = np.asarray(keypoints_3d, dtype=np.float64)
    dx, dz = k[left, 0] - k[right, 0], k[left, 2] - k[right, 2]
    return wrap(math.degrees(math.atan2(dz, dx)))


def head_yaw(row: dict) -> float | None:
    """The head's facing for one reading, from the line between the ears, on the shoulder yaw's scale.

    The shoulders say which way the body faces. A singer who turns the head
    to a microphone faces one way with the body and another with the head,
    and an eye judging "does he face where the source faces" reads the head
    (seen on the solo clip, 2026-10-05). Read from the joints the pose pass
    already stores, so an older record has it too.
    """
    if row.get("head_yaw") is not None:
        return row["head_yaw"]
    body = row.get("in_body")
    if not body:
        return None
    left, right = body["left_ear"], body["right_ear"]
    return round(wrap(math.degrees(math.atan2(left[2] - right[2], left[0] - right[0]))), 1)


def head_lift(row: dict) -> float | None:
    """How far the chin is lifted, in degrees: the nose above the line between the ears is positive.

    Facing has a second angle. On the solo clip the source sings with the
    chin up, and the render the eye called closest was the one that lifted
    its chin, not the one nearest in yaw (2026-10-05).
    """
    body = row.get("in_body")
    if not body:
        return None
    middle = [(a + b) / 2 for a, b in zip(body["left_ear"], body["right_ear"])]
    out = [n - m for n, m in zip(body["nose"], middle)]
    # the image's y grows downward, so a lifted nose has the smaller y
    return round(math.degrees(math.atan2(-out[1], math.hypot(out[0], out[2]))), 1)


def _mean_of(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 1) if values else None


def largest_turn(curve: list[float | None]) -> float | None:
    """How far the yaw gets from its first reading, 0 to 180: 180 for a full turn away, a few degrees for none."""
    values = [v for v in curve if v is not None]
    if not values:
        return None
    return round(max(apart(v, values[0]) for v in values), 1)


def eye_set(path: Path, name: str | None, field: str) -> tuple[dict[str, str], dict]:
    """({render label: its by-eye verdict for `field`}, the set's own entry) from the verdicts file.

    The file holds one set per source window (`sets`); `name` picks one and
    may be left out when there is only one. A render nobody judged on
    `field` is left out, so it is measured and not ranked.
    """
    spec = json.loads(path.read_text())
    sets = spec.get("sets") or {"": spec}
    if name is None:
        if len(sets) != 1:
            raise SystemExit(f"{path.name} holds {len(sets)} sets ({', '.join(sets)}); name one with --eye-set")
        name = next(iter(sets))
    if name not in sets:
        raise SystemExit(f"{path.name} has no set {name!r}; it has {', '.join(sets)}")
    chosen = sets[name]
    return {label: row[field] for label, row in chosen["clips"].items() if row.get(field)}, chosen


def compare(source: list[float | None], clip: list[float | None], frames: list[int],
            tolerance: float = TOLERANCE_DEGREES) -> dict:
    """How a clip's yaw curve stands against the source's over the same frames."""
    pairs = [(f, s, c) for f, s, c in zip(frames, source, clip) if s is not None and c is not None]
    if not pairs:
        return {"comparable": False, "verdict": "not measured"}
    end_source = circular_mean(source[-END_SAMPLES:])
    end_clip = circular_mean(clip[-END_SAMPLES:])
    diffs = [apart(s, c) for _f, s, c in pairs]
    diverges = None
    for i in range(len(pairs)):
        if all(d > tolerance for d in diffs[i:]):
            diverges = pairs[i][0]
            break
    end_difference = None if end_source is None or end_clip is None else apart(end_source, end_clip)
    return {
        "comparable": True,
        "samples": len(pairs),
        "largest_turn": largest_turn(clip),
        "largest_turn_source": largest_turn(source),
        "end_yaw_source": None if end_source is None else round(end_source, 1),
        "end_yaw": None if end_clip is None else round(end_clip, 1),
        "end_difference": None if end_difference is None else round(end_difference, 1),
        "mean_difference": round(sum(diffs) / len(diffs), 1),
        "diverges_at_frame": diverges,
        "tolerance": tolerance,
        "verdict": "not measured" if end_difference is None else ("holds" if end_difference <= tolerance else "fails"),
    }


def shots_to_read(record: dict, asked: list[int] | None, spec: dict | None = None) -> list[list[int]]:
    """Which frame ranges to read a pose record over: the one asked for, else the eye set's, else every shot it holds.

    A pose record covers a window and may hold several shots (`shots`, cut
    by the source). A turn or a motion is a property of one shot: across a
    cut the body is somewhere else for no reason of its own.
    """
    if asked:
        return [list(asked)]
    if spec and spec.get("shot"):
        return [list(spec["shot"])]
    return [list(s) for s in (record.get("shots") or [record["shot"]])]


def rows_in(record: dict, shot: list[int]) -> list[int]:
    """The positions in a record's curves of the sampled frames inside `shot`."""
    return [i for i, f in enumerate(record["frames"]) if shot[0] <= f <= shot[1]]


def judge(record: dict, shot: list[int], eye: dict[str, str], tolerance: float) -> dict:
    """Every clip of a pose record against its source over one shot."""
    at = rows_in(record, shot)
    frames = [record["frames"][i] for i in at]
    source_yaw = [record["source"]["curve"][i]["yaw"] for i in at]
    out = {"shot": list(shot), "samples": len(at), "source": {"largest_turn": largest_turn(source_yaw)}, "clips": {}}
    source_head = [head_yaw(record["source"]["curve"][i]) for i in at]
    source_lift = [head_lift(record["source"]["curve"][i]) for i in at]
    for label, clip in record["clips"].items():
        result = compare(source_yaw, [clip["curve"][i]["yaw"] for i in at], frames, tolerance)
        head = compare(source_head, [head_yaw(clip["curve"][i]) for i in at], frames, tolerance)
        # the head beside the shoulders: reported, and not the verdict
        result["head"] = {key: head.get(key) for key in ("end_yaw_source", "end_yaw", "end_difference", "mean_difference")}
        lift = [head_lift(clip["curve"][i]) for i in at]
        result["head"].update({
            "end_lift_source": _mean_of(source_lift[-END_SAMPLES:]), "end_lift": _mean_of(lift[-END_SAMPLES:]),
            "mean_lift_difference": _mean_of(abs(a - b) for a, b in zip(source_lift, lift) if a is not None and b is not None),
        })
        if label in eye:
            result["eye"] = eye[label]
        out["clips"][label] = result
    if eye:
        out["against_the_eye"] = ranks_as_the_eye({label: r.get("end_difference") for label, r in out["clips"].items()}, eye)
    return out


def ranks_as_the_eye(measured: dict[str, float], eye: dict[str, str]) -> dict:
    """Whether end differences order the clips as the eye did: every yes below every partial below every no."""
    rows = [(label, eye[label], measured[label]) for label in eye if measured.get(label) is not None]
    wrong = []
    for a_label, a_eye, a_value in rows:
        for b_label, b_eye, b_value in rows:
            if ORDER[a_eye] < ORDER[b_eye] and not a_value < b_value:
                wrong.append({"turned_more_by_eye": a_label, "end_difference": a_value,
                              "turned_less_by_eye": b_label, "its_end_difference": b_value})
    groups = {}
    for _label, verdict, value in rows:
        groups.setdefault(verdict, []).append(value)
    return {
        "clips_compared": len(rows),
        "same_order": not wrong,
        "disagreements": wrong,
        "end_difference_by_eye_verdict": {v: {"lowest": min(vals), "highest": max(vals), "clips": len(vals)}
                                          for v, vals in sorted(groups.items(), key=lambda kv: ORDER[kv[0]])},
        "not_judged_by_eye": sorted(set(measured) - set(eye)),
    }


# ------------------------------------------------------------------ the media

def decode(path: Path, first_second: float, rate: float, count: int, size: tuple[int, int],
           crop_to: tuple[int, int] | None = None) -> torch.Tensor:
    """`count` frames from `first_second` at `rate`, [count, H, W, 3] in 0..1.

    `size` is (width, height) to scale to. With `crop_to` the frames are then
    centre-cropped to that aspect and scaled to it, as the song node fits a
    source to its canvas.
    """
    w, h = size
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{first_second:.5f}", "-i", str(path), "-vf", f"fps={rate},scale={w}:{h}",
         "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    if done.returncode != 0:
        raise SystemExit(f"ffmpeg could not read {path.name}: {done.stderr.decode(errors='replace')[-300:]}")
    got = len(done.stdout) // (w * h * 3)
    frames = torch.from_numpy(np.frombuffer(done.stdout[:got * w * h * 3], np.uint8).reshape(got, h, w, 3).copy())
    frames = frames.float().div(255)
    if crop_to is not None and crop_to != (w, h):
        import comfy.utils
        frames = comfy.utils.common_upscale(frames.movedim(-1, 1), crop_to[0], crop_to[1], "bilinear", "center").movedim(1, -1)
    return frames


def video_size(path: Path) -> tuple[int, int]:
    """(width, height) of a video's first stream."""
    done = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=width,height", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    width, height = (int(v) for v in done.stdout.strip().split(",")[:2])
    return width, height


def mask_from_video(path: Path, count: int) -> torch.Tensor:
    """The first `count` frames of a mask saved as a video (white is the subject), [count, H, W] of 0 and 1."""
    w, h = video_size(path)
    done = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", str(count), "-f", "rawvideo",
                           "-pix_fmt", "gray", "-"], capture_output=True)
    if done.returncode != 0:
        raise SystemExit(f"ffmpeg could not read {path.name}: {done.stderr.decode(errors='replace')[-300:]}")
    got = len(done.stdout) // (w * h)
    frames = np.frombuffer(done.stdout[:got * w * h], np.uint8).reshape(got, h, w)
    return torch.from_numpy((frames >= 128).astype(np.float32))


def source_cuts(frames: torch.Tensor) -> list[int]:
    """The frames at which the source cuts, by the Subject Track's own cut finder and its automatic threshold."""
    import importlib
    import types
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    track = importlib.import_module("_h3pack.subject_track")
    scores = track.cut_scores(frames)
    return [int(c) for c in track.find_cuts(scores, track.auto_cuts(scores))]


def boxes(mask: torch.Tensor) -> list[dict | None]:
    """One {x, y, width, height} per frame of a [N, H, W] mask, None where it is empty."""
    out = []
    for m in mask:
        ys, xs = torch.nonzero(m > 0.5, as_tuple=True)
        if xs.numel() == 0:
            out.append(None)
            continue
        out.append({"x": float(xs.min()), "y": float(ys.min()),
                    "width": float(xs.max() - xs.min() + 1), "height": float(ys.max() - ys.min() + 1)})
    return out


def joints_in_box(keypoints_2d, box: dict) -> dict[str, list[float]]:
    """Each body joint's image position from the subject's box's top left, in units of the box's HEIGHT.

    One unit for both axes, so a distance between two readings is a fraction
    of the subject's height on screen whichever way it points.
    """
    k = np.asarray(keypoints_2d, dtype=np.float64)
    return {name: [round(float((k[i, 0] - box["x"]) / box["height"]), 4),
                   round(float((k[i, 1] - box["y"]) / box["height"]), 4)] for name, i in BODY_JOINTS.items()}


def joints_in_body(keypoints_3d) -> dict[str, list[float]]:
    """Each body joint in 3D, from the middle of the hips, in units of the hips-to-shoulders distance.

    Centred and scaled on the body itself, so a subject who is larger, nearer
    or of another build reads the same when the pose is the same. The axes
    are the camera's, so the facing is part of it.
    """
    k = np.asarray(keypoints_3d, dtype=np.float64)
    hips = (k[LEFT_HIP] + k[RIGHT_HIP]) / 2
    torso = float(np.linalg.norm((k[LEFT_SHOULDER] + k[RIGHT_SHOULDER]) / 2 - hips)) or 1.0
    return {name: [round(float(v), 4) for v in (k[i] - hips) / torso] for name, i in BODY_JOINTS.items()}


def in_frame(row: dict, size: list[int] | tuple[int, int]) -> dict:
    """A reading with the joints that fall outside the picture taken out of `in_body` and `in_box`.

    The body model places every joint whether or not it can see it: in a
    close-up it still gives knees and feet, and they move. A joint outside
    the frame is the model's guess, on the source and on the render alike,
    and two guesses agreeing or not says nothing about the render. `size` is
    the (width, height) the frame was read at. A reading with no `box` (a
    record from before the box was kept) is returned as it is.
    """
    box, on_screen = row.get("box"), row.get("in_box")
    if not box or not on_screen:
        return row
    x0, y0, _width, height = box
    seen = {name for name, (x, y) in on_screen.items()
            if 0 <= x0 + x * height < size[0] and 0 <= y0 + y * height < size[1]}
    out = dict(row)
    for space in ("in_body", "in_box"):
        if row.get(space):
            out[space] = {name: value for name, value in row[space].items() if name in seen}
    return out


def frame_sizes(record: dict) -> tuple[list[int], list[int]]:
    """((width, height) the source was read at, the same for the renders) of a pose record."""
    window = record.get("window") or {}
    canvas = window.get("canvas") or [10 ** 9, 10 ** 9]
    return window.get("source_frame") or window.get("source_size") or canvas, canvas


def found_box(predict, image: torch.Tensor) -> tuple[dict | None, dict | None]:
    """(person, box) for a frame with no mask to say where the subject is.

    The body model is first given the whole frame, then the box its own
    joints span, widened by `AUTO_BOX_MARGIN` on each side, which is what it
    is read at. For a clip with ONE person in it: with several, the whole
    frame is not a person's box and whoever the model favours is taken.
    """
    height, width = int(image.shape[1]), int(image.shape[2])
    whole = {"x": 0.0, "y": 0.0, "width": float(width), "height": float(height)}
    first = predict(image, whole)
    if first is None:
        return None, None
    k = np.asarray(first["pred_keypoints_2d"], dtype=np.float64)[list(BODY_JOINTS.values())]
    x0, y0, x1, y1 = k[:, 0].min(), k[:, 1].min(), k[:, 0].max(), k[:, 1].max()
    pad_x, pad_y = (x1 - x0) * AUTO_BOX_MARGIN, (y1 - y0) * AUTO_BOX_MARGIN
    x0, y0 = max(0.0, x0 - pad_x), max(0.0, y0 - pad_y)
    x1, y1 = min(float(width), x1 + pad_x), min(float(height), y1 + pad_y)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return first, whole
    box = {"x": float(x0), "y": float(y0), "width": float(x1 - x0), "height": float(y1 - y0)}
    return (predict(image, box) or first), box


def yaw_curve(predict, frames: torch.Tensor, frame_boxes: list[dict | None] | None) -> list[dict]:
    """One reading per frame: shoulder yaw, hip yaw and the body joints, or None where nobody was found.

    `frame_boxes` None means no mask says where the subject is, and each
    frame's box is found (`found_box`).
    """
    empty = {"yaw": None, "hip_yaw": None, "in_box": None, "in_body": None}
    rows = []
    for n, image in enumerate(frames):
        if frame_boxes is None:
            person, box = found_box(predict, image[None])
        else:
            box = frame_boxes[n]
            person = predict(image[None], box) if box is not None else None
        if person is None:
            rows.append(dict(empty))
            continue
        k = person["pred_keypoints_3d"]
        rows.append({"yaw": round(yaw_of(k, LEFT_SHOULDER, RIGHT_SHOULDER), 1),
                     "hip_yaw": round(yaw_of(k, LEFT_HIP, RIGHT_HIP), 1),
                     "in_box": joints_in_box(person["pred_keypoints_2d"], box),
                     "head_yaw": round(yaw_of(k, BODY_JOINTS["left_ear"], BODY_JOINTS["right_ear"]), 1),
                     "in_body": joints_in_body(k),
                     "box": [round(box[key]) for key in ("x", "y", "width", "height")]})
    return rows


def reanalyse(args) -> int:
    """The comparisons again from an earlier output's curves, at this run's tolerance and eye verdicts."""
    record = json.loads(args.reanalyse.read_text())
    eye, spec = {}, None
    if args.eye:
        eye, spec = eye_set(args.eye, args.eye_set, "turn")
    record["tolerance_degrees"] = args.tolerance
    read = [judge(record, shot, eye, args.tolerance) for shot in shots_to_read(record, args.shot, spec)]
    for one in read:
        for label, result in one["clips"].items():
            print(f"shot {one['shot'][0]}-{one['shot'][1]} {label}: end difference {result.get('end_difference')}, "
                  f"largest turn {result.get('largest_turn')} (source {one['source']['largest_turn']}), "
                  f"{result['verdict']}" + (f" (eye: {eye[label]})" if label in eye else ""))
    record.pop("against_the_eye", None)
    record.pop("by_shot", None)
    if len(read) == 1:
        # one shot: the reading sits on each clip beside its curve, as the pose pass writes it
        record["shot_read"] = read[0]["shot"]
        record["source"]["largest_turn"] = read[0]["source"]["largest_turn"]
        for label, result in read[0]["clips"].items():
            record["clips"][label] = {**result, "curve": record["clips"][label]["curve"]}
        if eye:
            record["against_the_eye"] = read[0]["against_the_eye"]
    else:
        record["by_shot"] = read
    record["reanalysed_from"] = args.reanalyse.name
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {args.out.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--reanalyse", type=Path, metavar="JSON",
                    help="recompute the comparisons from an earlier output's curves; no model, no media")
    ap.add_argument("--source", type=Path)
    ap.add_argument("--start", type=float, default=0.0, help="second of the source the window starts at")
    ap.add_argument("--rate", type=float, default=24.0, help="frames per second the window was loaded at")
    ap.add_argument("--mask", type=Path, help="the window's kept mask (.npz from the output folder's masks/)")
    ap.add_argument("--mask-video", type=Path,
                    help="the window's mask saved as a video from the window's first frame (white is the subject), "
                         "when no kept mask exists; it is only read for the subject's box")
    ap.add_argument("--find-box", action="store_true",
                    help="no kept mask: find the subject's box in each frame from the body model itself. "
                         "For a clip with one person in it")
    ap.add_argument("--shot", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    help="frames of the window to measure, inclusive. With --reanalyse: the one shot to read; "
                         "left out, the eye set's shot, else every shot the record holds")
    ap.add_argument("--canvas", default="1344x768", help="the renders' width x height")
    ap.add_argument("--clips-dir", type=Path, help="where the renders are")
    ap.add_argument("--clip", action="append", default=[], metavar="LABEL=PATH", help="one render; repeatable")
    ap.add_argument("--eye", type=Path, help="by-eye verdicts JSON; its labels are also looked up in --clips-dir")
    ap.add_argument("--eye-set", help="which set of the verdicts file; not needed when it holds one")
    ap.add_argument("--every", type=int, default=EVERY)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_DEGREES)
    ap.add_argument("--release", choices=sorted(RELEASES), default="vith", help="which SAM 3D Body release reads the pose")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.reanalyse:
        return reanalyse(args)
    if not (args.source and args.shot and (args.mask or args.mask_video or args.find_box)):
        ap.error("--source, --shot and one of --mask, --mask-video or --find-box are required unless --reanalyse is given")
    needs(f"the source {args.source.name}", args.source.is_file())
    for given, what in ((args.mask, "kept mask"), (args.mask_video, "mask video")):
        if given:
            needs(f"the {what} {given.name}", given.is_file())

    bootstrap()
    sys.path.append(str(REPO))
    dynamic_vram = server_memory_mode()
    import comfy.model_management as mm
    import comfy.utils
    import comfy_extras.nodes_sam3d_body as nodes
    import folder_paths
    import sam3d_body_vith

    needs(f"models/detection/{RELEASES[args.release]}", RELEASES[args.release] in folder_paths.get_filename_list("detection"))
    canvas = tuple(int(v) for v in args.canvas.lower().split("x"))
    first, last = args.shot
    sampled = list(range(first, last + 1, max(1, args.every)))

    eye, pattern = {}, "{label}*.mp4"
    judged = []
    if args.eye:
        eye, spec = eye_set(args.eye, args.eye_set, "turn")
        pattern = spec.get("clip_pattern", pattern)
        judged = sorted(spec["clips"])
    clips: dict[str, Path] = {}
    if args.clips_dir:
        # every clip the set names, found by its label in the folder, whether or not its turn was judged
        for label in judged:
            hits = sorted(args.clips_dir.glob(pattern.format(label=label)))
            if hits:
                clips[label] = hits[-1]
    for item in args.clip:
        label, _, path = item.partition("=")
        clips[label] = Path(path)
    missing = sorted(set(judged) - set(clips))

    if args.mask or args.mask_video:
        if args.mask:
            with np.load(args.mask) as z:
                mask = torch.from_numpy(z["mask"])
        else:
            mask = mask_from_video(args.mask_video, last + 1)
        src_h, src_w = int(mask.shape[1]), int(mask.shape[2])
        source_boxes_all = boxes(mask)
        fitted = comfy.utils.common_upscale(mask[:, None], canvas[0], canvas[1], "bilinear", "center")[:, 0]
        render_boxes_all = boxes(fitted)
        mask_frames, source_crop = int(mask.shape[0]), None
    else:
        # no mask: the source is read on the render's canvas, fitted as the song node fits it,
        # so its frames and a render's are the same picture apart from the subject
        src_w, src_h = video_size(args.source)
        source_boxes_all = render_boxes_all = None
        mask_frames, source_crop = None, canvas

    path = folder_paths.get_full_path_or_raise("detection", RELEASES[args.release])
    patcher = sam3d_body_vith.load_model(path) if args.release == "vith" else \
        nodes.SAM3DBody_Loader.execute(RELEASES[args.release]).args[0]

    def predict(image, box):
        pose = nodes.SAM3DBody_Predict.execute(patcher, image, bboxes=[box], run_hand_refinement=False).args[0]
        people = pose["frames"][0]
        return people[0] if people else None

    began = time.monotonic()
    record = {
        "script": "bench/measure_subject_yaw.py", "release": args.release, "device": str(mm.get_torch_device()),
        "dynamic_vram": dynamic_vram, "torch": torch.__version__,
        "window": {"start_second": args.start, "rate": args.rate, "mask_frames": mask_frames,
                   "source_size": [src_w, src_h], "canvas": list(canvas),
                   # what the source's frames measure once decoded, which is what a joint is in or out of
                   "source_frame": list(source_crop) if source_crop else [src_w, src_h]},
        "subject_box": ("the kept mask's" if args.mask else f"from the mask video {args.mask_video.name}" if args.mask_video
                        else "found per frame from the body model (one-person clips only)"),
        "shot": [first, last], "frames": sampled, "tolerance_degrees": args.tolerance,
        "yaw": "0 facing the camera, 90 side-on, 180 back to it; from the shoulder line, hips beside it",
        "source": {}, "clips": {}, "clips_not_found": missing,
    }
    with torch.inference_mode():
        span = decode(args.source, args.start + first / args.rate, args.rate, last - first + 1, (src_w, src_h),
                      crop_to=source_crop)
        cuts = [first + c for c in source_cuts(span)]
        record["cuts"] = cuts
        record["shots"] = [[a, b - 1] for a, b in zip([first, *cuts], [*cuts, last + 1]) if b - 1 >= a]
        rows = yaw_curve(predict, span[::max(1, args.every)],
                         None if source_boxes_all is None else [source_boxes_all[f] for f in sampled])
        record["source"] = {"curve": rows}
        source_yaw = [r["yaw"] for r in rows]
        print(f"source: {[r['yaw'] for r in rows]}", flush=True)
        for label, clip in clips.items():
            span = decode(clip, first / args.rate, args.rate, last - first + 1, canvas)
            rows = yaw_curve(predict, span[::max(1, args.every)],
                             None if render_boxes_all is None else [render_boxes_all[f] for f in sampled])
            result = compare(source_yaw, [r["yaw"] for r in rows], sampled, args.tolerance)
            result["curve"] = rows
            if label in eye:
                result["eye"] = eye[label]
            record["clips"][label] = result
            print(f"{label}: end {result.get('end_yaw')} vs source {result.get('end_yaw_source')}, "
                  f"difference {result.get('end_difference')}, {result['verdict']}"
                  + (f" (eye: {eye[label]})" if label in eye else ""), flush=True)
    record["seconds"] = round(time.monotonic() - began, 1)
    if len(record["shots"]) > 1:
        # the source cuts inside the window: the reading above is over the whole span, these are per shot
        record["by_shot"] = [judge(record, shot, eye, args.tolerance) for shot in record["shots"]]
    if eye:
        measured = {label: r.get("end_difference") for label, r in record["clips"].items()}
        record["against_the_eye"] = ranks_as_the_eye(measured, eye)
        print(json.dumps(record["against_the_eye"], indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {args.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
