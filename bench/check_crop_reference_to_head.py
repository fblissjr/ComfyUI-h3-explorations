#!/usr/bin/env python3
"""`bench/crop_reference_to_head.py` against a reference video whose every frame says where its face is, read from the VIDEO IT WROTE.

A reference cut to the wrong box, or enlarged by another factor than its record says, puts the face somewhere a
load's builder does not expect, and nothing downstream can see it. So this makes a small reference video with
a white square painted on the tip of the nose of each frame, a points file in which the face stands somewhere
else on every frame (`bench/_lib/face_points_fixture.py`), runs the tool, and reads the pictures back.

1. **One box, holding every frame's face with its room, in the canvas's shape, and the scale the record says.**
   The box worked out here from the points holds every face point of every frame with the room asked for; it is
   the canvas's shape and inside the canvas; and each frame's square is found in the written video where the
   record's box and scale put it, enlarged by that scale; and the points file written beside the video puts
   each frame's nose on that square.
2. **Asked for frames from the middle of both files, it cuts those frames.** The reference starts on one source
   frame and the points file on another; each reference frame carries its own number as a grey level.
3. **A head near an edge moves the box inside the canvas; a head that crosses the frame gets the whole frame.**
   Both are said in the record, and the box still holds every face.
4. **The original itself is fitted, then cut by the same box.** A source of another size and shape, read from
   its own frame numbers, comes out with each frame's square where the reference's is; without `--fit` it is
   refused.
5. **The read-back can fail.** The same frames written one out of place fail it; a still picture passes and is
   said to be untellable.
6. **What it refuses.** A span the points do not cover; a span the reference does not hold; points made on
   another canvas; a points file that does not say its first frame.

Needs ffmpeg and ffprobe on the path. No model, no card, no server.

    <comfy venv python> bench/check_crop_reference_to_head.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish, needs  # noqa: E402

needs("ffmpeg and ffprobe on the path", bool(shutil.which("ffmpeg") and shutil.which("ffprobe")))

from _lib import face_points_fixture as fx  # noqa: E402
from _lib.frames import stream  # noqa: E402

import crop_reference_to_head as cr  # noqa: E402

W, H, N = 384, 288, 16
FACE_PX, SQUARE = 60.0, 4                       # the fixture's face height; half the side of the square on the nose
WORK = Path(tempfile.mkdtemp(prefix="check_crop_reference_to_head_"))
quiet = lambda *_: None  # noqa: E731

#: Where the face stands on each frame, by steps that never repeat: a frame out of place shows.
STEP = np.array([[0, 0], [5, 2], [-3, 6], [8, -4], [-6, -5], [4, 7], [9, 1], [-7, 3], [3, -8], [6, 5], [-9, -2], [5, -6],
                 [-4, 8], [7, -3], [-8, 4], [2, 9]], float)
CENTRES = np.array([190.0, 120.0]) + np.cumsum(STEP, 0)


def video(path: Path, frames: np.ndarray) -> Path:
    """[n, h, w, 3] as a lossless video."""
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{frames.shape[2]}x{frames.shape[1]}",
                            "-r", "24", "-i", "-", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", str(path)], stdin=subprocess.PIPE)
    enc.communicate(frames.tobytes())
    assert enc.returncode == 0, path
    return path


def nose(centre) -> np.ndarray:
    """Where the fixture puts the tip of the nose."""
    return np.asarray(centre, float) + (0, 18)


def reference(centres, levels=None) -> np.ndarray:
    """A reference video: frame k a flat grey of `levels[k]` with a white square on the tip of that frame's nose."""
    out = []
    for k, c in enumerate(centres):
        f = np.full((H, W, 3), 90 if levels is None else levels[k], np.uint8)
        x, y = nose(c).round().astype(int)
        f[y - SQUARE:y + SQUARE + 1, x - SQUARE:x + SQUARE + 1] = 250
        out.append(f)
    return np.stack(out)


def faces(centres) -> np.ndarray:
    return fx.face(np.full(len(centres), 6.0), np.asarray(centres, float), face_px=FACE_PX)


def box_by_hand(centres, room: float) -> tuple[float, float, float, float]:
    """What the box must hold: every face of the fixture, by its construction, with the room asked for."""
    c = np.asarray(centres, float)
    reach = fx.EYE_APART / 2 + fx.EYE_WIDE                                  # an eye's outer end, from the face's middle
    pad = room * FACE_PX
    return (c[:, 0].min() - reach - pad, c[:, 1].min() - 10 - pad, c[:, 0].max() + reach + pad, c[:, 1].max() - 10 + FACE_PX + pad)


def square_at(frame: np.ndarray) -> tuple[float, float, float]:
    """The middle of the white square in a written frame, and its side."""
    ys, xs = np.nonzero(frame.min(-1) > 200)
    assert len(xs), "no square in the frame"
    return float(xs.mean()), float(ys.mean()), float(np.sqrt(len(xs)))


def read(path: Path, n: int) -> np.ndarray:
    return np.stack(list(stream(str(path), (W, H), 0, n, pix="rgb24")))


REF = video(WORK / "ref.mkv", reference(CENTRES))
POINTS = fx.save(WORK / "shot_points.npz", faces(CENTRES), (W, H), first=0)


def one_box_and_its_scale() -> str:
    rec = cr.run(str(REF), str(POINTS), 0, N, WORK / "a.mp4", "the check", say=quiet)
    assert rec["passed"] and not rec["read_back"]["no_neighbour_can_be_told_from_the_frame"], rec["read_back"]
    x0, y0, x1, y1 = rec["box"]
    need = box_by_hand(CENTRES, cr.HEAD_ROOM)
    assert x0 <= need[0] + 1 and y0 <= need[1] + 1 and x1 >= need[2] - 1 and y1 >= need[3] - 1, (rec["box"], need)   # to the rounding of a pixel
    assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H and rec["box_set_by"].startswith("the faces")
    assert abs((x1 - x0) / (y1 - y0) - W / H) < 0.01, ((x1 - x0), (y1 - y0))
    sx, sy = rec["scale"]
    assert abs(sx - W / (x1 - x0)) < 1e-3 and abs(sy - H / (y1 - y0)) < 1e-3 and sx > 1.1, rec["scale"]
    got, worst = read(WORK / "a.mp4", N), 0.0
    for f in range(N):
        gx, gy, side = square_at(got[f])
        wx, wy = (nose(CENTRES[f]).round() + 0.5 - (x0, y0)) * (sx, sy) - 0.5                  # a pixel's middle, carried through the box and the scale
        worst = max(worst, abs(gx - wx), abs(gy - wy))
        assert abs(gx - wx) < 1.5 and abs(gy - wy) < 1.5, (f, (gx, gy), (wx, wy))
        assert abs(side - (2 * SQUARE + 1) * sx) < 1.5, (f, side, (2 * SQUARE + 1) * sx)
    carried = cr.fp.load(WORK / "a_points.npz")
    assert carried["first"] == 0 and carried["size"] == (W, H) and carried["names"] == fx.NAMES and len(carried["keypoints"]) == N
    for f in range(N):
        gx, gy, _ = square_at(got[f])
        px, py = carried["keypoints"][f][carried["index"]["tip_of_nose"]]
        assert abs(px - 0.5 - gx) < 1.5 and abs(py - 0.5 - gy) < 1.5, (f, (px, py), (gx, gy))   # a point is a place between pixels; a square's middle is a pixel's
    assert abs(cr.fp.face_height(carried["keypoints"][0], carried["index"]) - FACE_PX * sy) < 0.5
    saved = json.loads((WORK / "a.json").read_text())
    assert saved["points_in_this_picture"]["file"] == "a_points.npz"
    assert saved["box"] == rec["box"] and saved["frames"] == [0, N - 1] and saved["made_from"]["reference_video"] == str(REF)
    assert (WORK / "a_sheet.jpg").exists() and len(saved["sheet"]["frames"]) == 6 and saved["sheet"]["looked_at_by"] == ""
    assert abs(saved["face_height_px_in_this_video"] - FACE_PX * sy) < 0.5
    tight = cr.run(str(REF), str(POINTS), 0, N, WORK / "tight.mp4", "the check", room=0.5, say=quiet)
    assert tight["scale"][0] > sx and tight["room_set_by"] == "--room" and rec["room_set_by"].startswith("HEAD_ROOM"), (tight["scale"], sx)
    need = box_by_hand(CENTRES, 0.5)
    assert tight["box"][0] <= need[0] + 1 and tight["box"][1] <= need[1] + 1 and tight["box"][2] >= need[2] - 1 and tight["box"][3] >= need[3] - 1
    return (f"one box {rec['box']} holds every face with its room and is the canvas's shape; enlarged {sx:.3f} times, each frame's square is within "
            f"{worst:.1f} px of where the record puts it and that many times larger; the points written beside it put each nose on its square; less room gives a tighter box ({tight['scale'][0]:.3f} times)")


def frames_from_the_middle() -> str:
    video_first, points_first, first, count = 95, 100, 105, 8
    centres = np.array([190.0, 120.0]) + np.array([[(k * 7) % 23 - 11, (k * 5) % 17 - 8] for k in range(30)], float)   # row k is source frame 100 + k
    ref_centres = np.concatenate([centres[:1].repeat(5, 0), centres])                                               # file frame k is source frame 95 + k
    ref = video(WORK / "ref_mid.mkv", reference(ref_centres, levels=[40 + 3 * k for k in range(len(ref_centres))]))
    pts = fx.save(WORK / "mid_points.npz", faces(centres), (W, H), first=points_first)
    rec = cr.run(str(ref), str(pts), first, count, WORK / "mid.mp4", "the check", video_first=video_first, say=quiet)
    assert rec["passed"], rec["read_back"]
    x0, y0, _, _ = rec["box"]
    got = read(WORK / "mid.mp4", count)
    for j in range(count):
        level = int(np.median(got[j, :12, :12]))
        assert abs(level - (40 + 3 * (first - video_first + j))) <= 2, (j, level)
        gx, gy, _ = square_at(got[j])
        wx, wy = (nose(centres[first - points_first + j]).round() + 0.5 - (x0, y0)) * rec["scale"] - 0.5
        assert abs(gx - wx) < 1.5 and abs(gy - wy) < 1.5, (j, (gx, gy), (wx, wy))
    used = box_by_hand(centres[first - points_first:first - points_first + count], cr.HEAD_ROOM)
    assert rec["box"][0] <= used[0] + 1 and rec["box"][2] >= used[2] - 1                           # the box is of the frames asked for
    assert rec["frames"] == [105, 112] and rec["made_from"]["its_first_frame_is_source_frame"] == 95 and rec["made_from"]["their_first_row_is_source_frame"] == 100
    return "frames 105 to 112 of a reference from 95 and points from 100: each written frame is its own reference frame, cut by a box of those frames' faces"


def edges_and_the_whole_frame() -> str:
    corner = np.array([70.0, 50.0]) + np.cumsum(STEP, 0) * 0.3                                       # a head in the top left corner
    pts = fx.save(WORK / "corner_points.npz", faces(corner), (W, H), first=0)
    rec = cr.run(str(video(WORK / "ref_corner.mkv", reference(corner))), str(pts), 0, N, WORK / "corner.mp4", "the check", say=quiet)
    x0, y0, x1, y1 = rec["box"]
    assert rec["passed"] and rec["moved_to_lie_inside_the_canvas"] is True and (x0, y0) == (0, 0) and x1 <= W and y1 <= H, rec["box"]
    reach = fx.EYE_APART / 2 + fx.EYE_WIDE
    assert x1 >= corner[:, 0].max() + reach and y1 >= corner[:, 1].max() - 10 + FACE_PX                # every face is still in it
    across = np.stack([np.linspace(60, 320, N), np.full(N, 130.0)], 1)                                # a head that crosses the frame
    pts = fx.save(WORK / "across_points.npz", faces(across), (W, H), first=0)
    whole = cr.run(str(video(WORK / "ref_across.mkv", reference(across))), str(pts), 0, N, WORK / "across.mp4", "the check", say=quiet)
    assert whole["passed"] and whole["box"] == [0, 0, W, H] and whole["scale"] == [1.0, 1.0] and "the whole frame" in whole["box_set_by"], whole["box"]
    got = read(WORK / "across.mp4", N)
    gx, gy, _ = square_at(got[5])
    assert abs(gx - nose(across[5]).round()[0]) < 1 and abs(gy - nose(across[5]).round()[1]) < 1
    return f"a head in the corner moves the box to {rec['box']} and is said; a head that crosses the frame gets the whole frame, said, at a scale of one"


def the_original_fitted_then_cut() -> str:
    big = (W * 2 + 64, H * 2)                                                    # twice the canvas and wider: the loader cuts 32 a side, then halves
    lead = 7                                                                     # the source's frame 0 is source frame 0; the span begins on its frame 7
    frames = []
    for c in np.concatenate([CENTRES[:1].repeat(lead, 0), CENTRES]):
        f = np.full((big[1], big[0], 3), 90, np.uint8)
        x, y = nose(c).round().astype(int)
        f[2 * (y - SQUARE):2 * (y + SQUARE + 1), 32 + 2 * (x - SQUARE):32 + 2 * (x + SQUARE + 1)] = 250
        frames.append(f)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{big[0]}x{big[1]}", "-r", "24", "-i", "-",
                            "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", str(WORK / "source.mkv")], stdin=subprocess.PIPE)
    enc.communicate(np.stack(frames).tobytes())
    pts = fx.save(WORK / "src_points.npz", faces(CENTRES), (W, H), first=lead)
    ref = cr.run(str(REF), str(pts), lead, N, WORK / "ref_cut.mp4", "the check", say=quiet)
    rec = cr.run(str(WORK / "source.mkv"), str(pts), lead, N, WORK / "src_cut.mp4", "the check", fit=True, say=quiet)
    assert rec["passed"] and rec["box"] == ref["box"] and rec["scale"] == ref["scale"], (rec["box"], ref["box"])
    assert "fitted to the canvas" in rec["made_from"]["fitted"] and rec["made_from"]["its_first_frame_is_source_frame"] == 0, rec["made_from"]
    a, b = read(WORK / "ref_cut.mp4", N), read(WORK / "src_cut.mp4", N)
    worst = 0.0
    for f in range(N):
        (ax, ay, _), (bx, by, _) = square_at(a[f]), square_at(b[f])
        worst = max(worst, abs(ax - bx), abs(ay - by))
        assert abs(ax - bx) < 1.5 and abs(ay - by) < 1.5, (f, (ax, ay), (bx, by))
    try:
        cr.run(str(WORK / "source.mkv"), str(pts), lead, N, WORK / "x.mp4", "the check", video_first=0, say=quiet)
    except SystemExit as exc:
        assert "--fit" in str(exc), exc
    else:
        raise AssertionError("a source of another size was cut without being fitted")
    return f"a source twice the canvas's size and wider, read from its own frame 7, is cut by the same box and its squares sit within {worst:.1f} px of the reference's; without --fit it is refused"


def the_read_back_can_fail() -> str:
    ref = reference(CENTRES)
    box = cr.one_box(dict(cr.faces.rows_of(str(POINTS), None, 0, N, None, whole_face=False)[0]), (W, H), cr.HEAD_ROOM)["box"]
    made = cr.cut(ref, box)
    cr.marks.write_video(WORK / "good.mp4", made)
    good = cr.read_back(WORK / "good.mp4", made)
    assert good["passed"] and not good["cannot_tell"], good["lags"]
    cr.marks.write_video(WORK / "late.mp4", np.concatenate([made[:1], made[:-1]]))                   # every frame one late, the first repeated
    late = cr.read_back(WORK / "late.mp4", made)
    assert not late["passed"] and late["slipped"] and late["lags"][1] < late["lags"][0], late["lags"]
    cr.marks.write_video(WORK / "short.mp4", made[:-1])                                              # a frame dropped from the end
    assert not cr.read_back(WORK / "short.mp4", made)["passed"]
    still = np.repeat(made[:1], N, 0)
    cr.marks.write_video(WORK / "still.mp4", still)
    same = cr.read_back(WORK / "still.mp4", still)
    assert same["passed"] and same["cannot_tell"], same["lags"]
    return (f"as cut it passes ({good['lags'][0]:.2f} grey levels against {min(good['lags'][-1], good['lags'][1]):.2f} a frame off); every frame one late fails; "
            "a file a frame short fails; a still picture passes and is said to be untellable")


def refusals() -> str:
    kp = faces(CENTRES)
    for what, points, first, video_first, word in (
            ("a span the points do not cover", POINTS, 6, None, "were asked for"),
            ("a span the reference does not hold", POINTS, 0, 5, "were asked for"),
            ("another canvas", fx.save(WORK / "big_points.npz", kp, (W * 2, H * 2), first=0), 0, None, "not on one canvas"),
            ("no first frame said", fx.save(WORK / "nofirst_points.npz", kp, (W, H)), 0, None, "--points-first")):
        try:
            cr.run(str(REF), str(points), first, N, WORK / "r.mp4", "the check", video_first=video_first, say=quiet)
        except (SystemExit, ValueError) as exc:
            assert word in str(exc), (what, exc)
        else:
            raise AssertionError(f"{what}: accepted")
    return "a span the points do not cover, a span the reference does not hold, points on another canvas and a file that does not say its first frame are each refused"


case("one box, holding every face with its room, and the scale the record says", one_box_and_its_scale)
case("frames from the middle of both files", frames_from_the_middle)
case("a head near an edge, and a head that crosses the frame", edges_and_the_whole_frame)
case("the original itself is fitted, then cut by the same box", the_original_fitted_then_cut)
case("the read-back can fail", the_read_back_can_fail)
case("what it refuses", refusals)
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(finish())
