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
7. **The sheet shows a change.** A reference whose whole picture changes at frame 9 has 9 on its sheet with the
   picture's change as the reason (`bench/_lib/pick_frames.py`).
8. **`--cranium` frames the body model's grey head** (2026-10-22). On a black picture with a grey figure (a head, a neck,
   shoulders, and a raised hand clear of the head) the box holds every frame's head above the chin with `CRANIUM_ROOM` above
   and beside it and `CRANIUM_NECK` below the chin, and not the hand; a frame whose grey reaches the picture's top is named
   as reaching the box's edge and the picture's; a head that travels gives the one box's enlargement and each frame's own,
   the one smaller; a small head is enlarged at most `CRANIUM_MOST`; `--box` cuts by the box given; `--cranium` with `--fit`, `--head-track` or `--box` is refused; a frame with no grey head joined to its face (she is hidden there) does not size the box.

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
    assert (WORK / "a_sheet.jpg").exists() and saved["sheet"]["looked_at_by"] == "" \
        and saved["sheet"]["frames"][::len(saved["sheet"]["frames"]) - 1] == [0, N - 1] and set(saved["sheet"]["why"]) == {str(f) for f in saved["sheet"]["frames"]}
    assert "no record beside the points file" in saved["made_from"]["the_points_own_record_says"], saved["made_from"]
    POINTS.with_suffix(".json").write_text(json.dumps({"made_by": "a session", "made_at": "then", "made_from": {"box_mask": "track.mkv"}, "frames": N}))
    told = cr.run(str(REF), str(POINTS), 0, N, WORK / "told.mp4", "the check", say=quiet)["made_from"]["the_points_own_record_says"]
    POINTS.with_suffix(".json").unlink()
    assert told == {"made_by": "a session", "made_at": "then", "made_from": {"box_mask": "track.mkv"}}, told   # which track the points follow, as their record has it
    assert abs(saved["face_height_px_in_this_video"] - FACE_PX * sy) < 0.5
    tight = cr.run(str(REF), str(POINTS), 0, N, WORK / "tight.mp4", "the check", room=0.5, say=quiet)
    assert tight["scale"][0] > sx and tight["room_set_by"] == "--room" and rec["room_set_by"].startswith("HEAD_ROOM"), (tight["scale"], sx)
    need = box_by_hand(CENTRES, 0.5)
    assert tight["box"][0] <= need[0] + 1 and tight["box"][1] <= need[1] + 1 and tight["box"][2] >= need[2] - 1 and tight["box"][3] >= need[3] - 1
    return (f"one box {rec['box']} holds every face with its room and is the canvas's shape; enlarged {sx:.3f} times, each frame's square is within "
            f"{worst:.1f} px of where the record puts it and that many times larger; the points written beside it put each nose on its square; less room gives a tighter box ({tight['scale'][0]:.3f} times)")


def the_sheet_shows_a_change() -> str:
    """The sheet's frames come from what the run holds: a reference whose whole picture changes at frame 9 has 9 on its
    sheet, with the picture's change as the reason (red when the sheet takes first, last and one a second alone)."""
    ref = video(WORK / "ref_change.mkv", reference(CENTRES, [90] * 9 + [200] * (N - 9)))
    rec = cr.run(str(ref), str(POINTS), 0, N, WORK / "change.mp4", "the check", say=quiet)
    assert 9 in rec["sheet"]["frames"] and "picture" in rec["sheet"]["why"]["9"], rec["sheet"]
    assert rec["sheet"]["over_the_cap"] is False and rec["sheet"]["cap"].endswith("within it"), rec["sheet"]
    return f"frame 9, where the picture changes, is on the sheet: {rec['sheet']['why']['9']}"


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
    for what, kw, word in (
            ("both --head-track and --no-head-track", {"head_track": str(REF), "no_head_track": True}, "cannot give both"),
            ("missing --head-track file", {"head_track": str(WORK / "no_such_head_track.mkv")}, "is not a file"),
            ("a span the head track does not hold", {"head_track": str(REF), "head_track_first": 5}, "were asked for")):
        try:
            cr.run(str(REF), str(POINTS), 0, N, WORK / "r_ht.mp4", "the check", say=quiet, **kw)
        except SystemExit as exc:
            assert word in str(exc), (what, exc)
        else:
            raise AssertionError(f"{what}: accepted")
    return "a span the points, reference or head track do not cover, another canvas, no first frame, missing --head-track and conflicting flags are each refused"


def head_track_and_crown_room() -> str:
    """The head box from the head track, with `CROWN_ROOM` above the crown (`C201` item 2, `M22`): on tilted frames where
    the crown reaches further above the glabella than `room` alone leaves, the head track holds the whole head with crown
    room; on a frontal head whose crown sits inside the face room, the box is no worse; and when the untracked kitchen
    (`767, 771, 772`, `751-778`, `805-833`) and frontal mom (`278-293`) captures are on disk, checks those exact frames."""
    from _lib.frames import write_mask_video
    # 1. Synthetic tilted head track vs frontal head track
    room = 0.6                                                                  # tight enough that a tilted crown 1.0 * FACE_PX above glabella clips without head_track
    pts_dir = WORK / "ht_case"
    pts_dir.mkdir(exist_ok=True)
    pts_file = fx.save(pts_dir / "shot_points.npz", faces(CENTRES), (W, H), first=0)
    P, _ = cr.faces.rows_of(str(pts_file), None, 0, N, 0, whole_face=False)
    face_only = cr.one_box(P, (W, H), room)
    # Tilted head mask: on frames 4..7 the crown reaches 1.0 * FACE_PX above the glabella (y = centre_y - 10 - FACE_PX)
    tilted_mask = np.zeros((N, H, W), bool)
    frontal_mask = np.zeros((N, H, W), bool)
    for f, (cx, cy) in enumerate(CENTRES):
        glab_y = int(round(cy - 10))
        chin_y = int(round(cy - 10 + FACE_PX))
        crown_frontal = max(2, int(round(glab_y - 0.35 * FACE_PX)))
        crown_tilted = max(2, int(round(glab_y - (1.0 * FACE_PX if 4 <= f <= 7 else 0.35 * FACE_PX))))
        x_l, x_r = max(2, int(round(cx - 0.55 * FACE_PX))), min(W - 2, int(round(cx + (0.85 if 4 <= f <= 7 else 0.55) * FACE_PX)))
        frontal_mask[f, crown_frontal:chin_y, int(round(cx - 0.55 * FACE_PX)):int(round(cx + 0.55 * FACE_PX))] = True
        tilted_mask[f, crown_tilted:chin_y, x_l:x_r] = True
    tilted_boxes = cr.head_boxes_of(tilted_mask)
    assert face_only["box"][1] > tilted_boxes[:, 1].min(), (face_only["box"], tilted_boxes[:, 1].min())   # face-only clips the tilted crown
    with_tilted = cr.one_box(P, (W, H), room, head_track=tilted_mask)
    c_room_px = (cr.CROWN_ROOM * (room / cr.HEAD_ROOM)) * FACE_PX
    assert with_tilted["box"][1] <= tilted_boxes[:, 1].min() - c_room_px + 1, (with_tilted["box"], tilted_boxes[:, 1].min(), c_room_px)
    assert with_tilted["box"][0] <= tilted_boxes[:, 0].min() and with_tilted["box"][2] >= tilted_boxes[:, 2].max()
    assert with_tilted["box"][3] >= tilted_boxes[:, 3].max() and "the head track with crown room" in with_tilted["set_by"]
    with_frontal = cr.one_box(P, (W, H), room, head_track=frontal_mask)
    assert with_frontal["box"] == face_only["box"], (with_frontal["box"], face_only["box"])               # frontal box is no worse (identical when inside face room)
    # 2. Auto-discovery beside the points' box mask (`_track_` -> `_head_track_`) and `--no-head-track` override
    body_tr = pts_dir / "shot_lead_track_00001.mkv"
    head_tr = pts_dir / "shot_lead_head_track_00001.mkv"
    write_mask_video(body_tr, frontal_mask)
    write_mask_video(head_tr, tilted_mask)
    pts_file.with_suffix(".json").write_text(json.dumps({
        "made_by": "the check",
        "made_from": {"box": {"mask_video": str(body_tr), "its_first_frame_is_source_frame": 0}},
    }))
    rec_auto = cr.run(str(REF), str(pts_file), 0, N, pts_dir / "auto.mp4", "the check", room=room, say=quiet)
    rec_no_ht = cr.run(str(REF), str(pts_file), 0, N, pts_dir / "no_ht.mp4", "the check", room=room, no_head_track=True, say=quiet)
    assert rec_auto["box"] == with_tilted["box"] and rec_auto["made_from"]["head_track"]["found_by"] == "beside the points' box mask", rec_auto
    assert rec_no_ht["box"] == face_only["box"] and "head_track" not in rec_no_ht["made_from"], rec_no_ht
    # 3. Real kitchen tilted frames (767, 771, 772, 751-778, 751-785) and frontal shots (805-833, 278-293) when present on disk
    vdir = Path("/nonexistent")
    try:
        vdir = cr.faces.h3_config.output_dir() / "Video" / "fun"
    except BaseException:
        root = Path(__file__).resolve().parents[1]
        for d in (root / "data", *sorted(root.parent.parent.glob("custom_nodes/*/data")), *sorted(root.parent.parent.glob("ComfyUI/custom_nodes/*/data"))):
            for jp in sorted(d.glob("*_lead_fp_0805*_job/job.plan.json")):
                rec = json.loads(jp.read_text()).get("job", {}).get("reference", {}).get("record")
                if rec:
                    vdir = Path(rec).parents[2]
                    break
    ht_604 = vdir / "mrgemson" / "shot_c_0604c_sam_head_test" / "shot_c_0604c_lead_head_track_00001.mkv"
    pts_606 = vdir / "references" / "lead_0606_0998" / "lead_0606_0998_points.npz"
    if ht_604.is_file() and pts_606.is_file():
        for f_tilt, min_clear in ((767, 30), (771, 27), (772, 27)):
            pts_f, _ = cr.faces.rows_of(str(pts_606), None, f_tilt, 1, 606, whole_face=False)
            hm_f, _ = cr.read_mask(str(ht_604), (1024, 768), 604, f_tilt, 1)
            hb_f = cr.head_boxes_of(hm_f)[0]
            b_old = cr.one_box(pts_f, (1024, 768), cr.HEAD_ROOM)["box"]
            b_new = cr.one_box(pts_f, (1024, 768), cr.HEAD_ROOM, head_track=hm_f)["box"]
            assert b_new[0] <= hb_f[0] and b_new[1] <= hb_f[1] - min_clear and b_new[2] >= hb_f[2] and b_new[3] >= hb_f[3], (f_tilt, b_new, hb_f)
            assert b_new[1] < b_old[1], (f_tilt, b_old, b_new)
        for s_first, s_count in ((751, 28), (751, 35)):
            pts_s, _ = cr.faces.rows_of(str(pts_606), None, s_first, s_count, 606, whole_face=False)
            hm_s, _ = cr.read_mask(str(ht_604), (1024, 768), 604, s_first, s_count)
            hb_s = cr.head_boxes_of(hm_s)
            b_old = cr.one_box(pts_s, (1024, 768), cr.HEAD_ROOM)["box"]
            b_new = cr.one_box(pts_s, (1024, 768), cr.HEAD_ROOM, head_track=hm_s)["box"]
            assert b_old[1] > hb_s[:, 1].min() + 30, (b_old, hb_s[:, 1].min())                            # face-only cuts 39..43 px off the crown
            assert b_new[0] <= hb_s[:, 0].min() and b_new[1] <= hb_s[:, 1].min() and b_new[2] >= hb_s[:, 2].max() and b_new[3] >= hb_s[:, 3].max(), (b_new, hb_s)
        # Frontal 805-833: whole head held with >= 28 px crown room, and box width within 5% of face-only
        pts_805, _ = cr.faces.rows_of(str(pts_606), None, 805, 29, 606, whole_face=False)
        hm_805, _ = cr.read_mask(str(ht_604), (1024, 768), 604, 805, 29)
        hb_805 = cr.head_boxes_of(hm_805)
        b805_old = cr.one_box(pts_805, (1024, 768), cr.HEAD_ROOM)["box"]
        b805_new = cr.one_box(pts_805, (1024, 768), cr.HEAD_ROOM, head_track=hm_805)["box"]
        assert b805_new[0] <= hb_805[:, 0].min() and b805_new[1] <= hb_805[:, 1].min() - 28 and b805_new[2] >= hb_805[:, 2].max() and b805_new[3] >= hb_805[:, 3].max()
        assert (b805_new[2] - b805_new[0]) <= (b805_old[2] - b805_old[0]) * 1.05, (b805_old, b805_new)
        kitchen_note = f"; kitchen 767/771/772 and 751-785 hold the whole head ({b_new} vs {b_old}), frontal 805-833 {b805_new} vs {b805_old}"
    else:
        kitchen_note = " (kitchen captures not on disk)"
    return f"tilted head track widens box from {face_only['box']} to {with_tilted['box']} with crown room; frontal head track stays {with_frontal['box']}{kitchen_note}"


def figure(centres, crown_up=None, hand: bool = True, tucked: bool = False, arms: bool = False) -> np.ndarray:
    """A body model's picture: black, with a grey (90) head over each frame's face (an ellipse from `crown_up[k]` face heights above
    the glabella, 0.9 when not given, to the chin, 45 px either side of the face's middle), a neck, shoulders, and a raised hand
    clear of the head above the chin. `tucked`: the chin tucked, the shoulders 130 px either side rising to 15 px above the chin and
    joined to the head (second 0278); `arms`: both arms raised from the shoulders to above the crown (lead 1156)."""
    out = []
    yy, xx = np.mgrid[0:H, 0:W]
    for k, (cx, cy) in enumerate(np.asarray(centres, float)):
        f = np.zeros((H, W, 3), np.uint8)
        top, chin = cy - 10 - (0.9 if crown_up is None else crown_up[k]) * FACE_PX, cy - 10 + FACE_PX
        my, ry = (top + chin) / 2, (chin - top) / 2
        f[((xx - cx) / 45.0) ** 2 + ((yy - my) / ry) ** 2 <= 1.0] = 90
        f[int(chin) - 4:int(chin) + 30, int(cx) - 15:int(cx) + 16] = 90
        f[int(chin) + 30:, max(int(cx) - 80, 0):int(cx) + 81] = 90
        if tucked:
            f[int(chin) - 15:, max(int(cx) - 130, 0):int(cx) + 131] = 90
        if arms:
            for x in (int(cx) - 95, int(cx) + 80):
                f[max(int(top) - 20, 0):int(chin) + 40, max(x, 0):x + 15] = 90
        if hand:
            f[(xx - (cx + 110)) ** 2 + (yy - (cy - 20)) ** 2 <= 100] = 90
        out.append(f)
    return np.stack(out)


def heads_by_hand(centres, crown_up=None) -> np.ndarray:
    """Each frame's grey head above the chin as [x0, y0, x1, y1], by the figure's construction."""
    c = np.asarray(centres, float)
    up = np.full(len(c), 0.9) if crown_up is None else np.asarray(crown_up, float)
    return np.stack([c[:, 0] - 45, np.maximum(c[:, 1] - 10 - up * FACE_PX, 0), c[:, 0] + 45, c[:, 1] - 10 + FACE_PX], 1)


def cranium_box_round_the_grey_head() -> str:
    folder = WORK / "cranium"
    folder.mkdir()
    mesh = video(folder / "mesh.mkv", figure(CENTRES))
    rec = cr.run(str(mesh), str(POINTS), 0, N, folder / "a.mp4", "the check", cranium=True, say=quiet)
    x0, y0, x1, y1 = rec["box"]
    heads, room, neck = heads_by_hand(CENTRES), cr.CRANIUM_ROOM * FACE_PX, cr.CRANIUM_NECK * FACE_PX
    need = (heads[:, 0].min() - room, heads[:, 1].min() - room, heads[:, 2].max() + room, heads[:, 3].max() + neck)
    # THE CASE THAT GOES RED on a tool with no `--cranium`, or a box sized by the face's room or the hand
    assert rec["passed"] and x0 <= need[0] + 1 and y0 <= need[1] + 1 and x1 >= need[2] - 1 and y1 >= need[3] - 1, (rec["box"], need)
    assert abs((x1 - x0) / (y1 - y0) - W / H) < 0.01 and (y1 - y0) <= int(np.ceil(max(need[3] - need[1], (need[2] - need[0]) * H / W))) + 2, (rec["box"], need)
    handless = cr.run(str(video(folder / "handless.mkv", figure(CENTRES, hand=False))), str(POINTS), 0, N, folder / "handless.mp4", "the check",
                      cranium=True, say=quiet)
    assert handless["box"] == rec["box"], ("the raised hand, clear of the head, moved the box", handless["box"], rec["box"])
    # THE CASES THAT GO RED under the grey above the CHIN (the Gemini reviewer's measure of 2026-10-22): shoulders risen above the chin
    # line and joined to the head (second 0278), and arms raised beside the head joined to them (lead 1156). Above mid-face neither joins
    for what, kw in (("shoulders above the chin", {"tucked": True}), ("arms raised above the head", {"tucked": True, "arms": True}),
                     ("arms raised, shoulders low", {"arms": True})):
        got = cr.run(str(video(folder / "body.mkv", figure(CENTRES, hand=False, **kw))), str(POINTS), 0, N, folder / f"{'_'.join(kw)}.mp4", "the check",
                     cranium=True, say=quiet)
        (folder / "body.mkv").unlink()
        assert got["box"] == rec["box"] and got["cranium"]["frames_the_grey_head_reaches_the_box_edge"] == [], (what, got["box"], rec["box"])
    c = rec["cranium"]
    assert rec["box_set_by"].startswith("the grey head") and c["frames_the_grey_head_reaches_the_box_edge"] == [] \
        and c["frames_the_grey_head_reaches_the_picture_edge"] == [] and c["frames_with_no_grey_head_joined_to_the_face"] == [], c
    assert c["grey_above_mid_face_not_joined_to_the_face_px_least_median_most"][0] > 250, c       # the hand, about 300 px, left out on every frame
    assert json.loads((folder / "a.json").read_text())["cranium"] == c
    # a frame whose grey reaches the picture's top: named, as reaching the box's edge and the picture's
    up = [0.9] * N
    up[5] = 3.0
    hat_v = video(folder / "hat.mkv", figure(CENTRES, up))
    hat = cr.run(str(hat_v), str(POINTS), 0, N, folder / "hat.mp4", "the check", cranium=True, say=quiet)
    assert hat["box"][1] == 0 and hat["cranium"]["frames_the_grey_head_reaches_the_box_edge"] == [5] \
        and hat["cranium"]["frames_the_grey_head_reaches_the_picture_edge"] == [5], (hat["box"], hat["cranium"])
    # bare and dropped frames (`leave_out_frames`, 2026-10-25: 2050g's 2249-2270 and 4310c's 4388-4389) do not size the cranium box:
    # leaving out frame 5 restores `rec["box"]` and clears the edge hits; leaving out every frame falls back cleanly to `hat["box"]`
    hat_skip = cr.run(str(hat_v), str(POINTS), 0, N, folder / "hat_skip.mp4", "the check", cranium=True, leave_out_frames=[5], say=quiet)
    assert hat_skip["box"] == rec["box"] != hat["box"] and hat_skip["cranium"]["frames_left_out_of_the_box"] == [5] \
        and hat_skip["cranium"]["frames_the_grey_head_reaches_the_box_edge"] == [] \
        and hat_skip["cranium"]["frames_the_grey_head_reaches_the_picture_edge"] == [], (hat_skip["box"], hat_skip["cranium"])
    Pn_hat, _ = cr.faces.rows_of(str(POINTS), None, 0, N, 0, whole_face=False)
    all_out = cr.cranium_box(Pn_hat, (W, H), cr.HEAD_ROOM, figure(CENTRES, up), first=0, leave_out_frames=list(range(N)))
    assert all_out["box"] == hat["box"] and all_out["cranium"]["frames_left_out_of_the_box"] == list(range(N)), all_out
    # a head that travels: each frame's own box against the one box, as figures
    far = np.stack([np.linspace(110, 270, N), np.full(N, 130.0)], 1)
    far_pts = fx.save(folder / "far_points.npz", faces(far), (W, H), first=0)
    trav = cr.run(str(video(folder / "far.mkv", figure(far))), str(far_pts), 0, N, folder / "far.mp4", "the check", cranium=True, say=quiet)["cranium"]
    least, median, most = trav["each_frame_alone_would_enlarge_least_median_most"]
    assert trav["the_one_box_enlarges"] < least <= median <= most and abs(trav["the_travel_keeps_of_the_median_frame_enlargement"]
                                                                          - trav["the_one_box_enlarges"] / median) < 0.002, trav
    # a small head is enlarged at most CRANIUM_MOST, and the record says what it would have been: a face a quarter the fixture's size
    small = np.asarray(CENTRES)[:, None, :] + (faces(CENTRES) - np.asarray(CENTRES)[:, None, :]) * 0.25
    P, _ = cr.faces.rows_of(str(fx.save(folder / "small_points.npz", small, (W, H), first=0)), None, 0, N, 0, whole_face=False)
    tiny = cr.cranium_box(P, (W, H), cr.HEAD_ROOM, np.zeros((N, H, W, 3), np.uint8))
    held = tiny["cranium"]["held_to_the_most"]
    assert held["capped"] and held["would_enlarge_without_it"] > cr.CRANIUM_MOST >= W / (tiny["box"][2] - tiny["box"][0]) > cr.CRANIUM_MOST - 0.2 \
        and "CRANIUM_MOST" in tiny["set_by"] and not c["held_to_the_most"]["capped"], (tiny["box"], held)
    # a frame where she is hidden (no grey head; the pose model's face points are a guess far from her head) does not size the box
    # (second_1093_1133_f, 2026-10-22: a stray face box 200 px below her head took the box from about 5.7 to 2.27 times)
    pics = figure(CENTRES, hand=False)
    hid = pics.copy()
    hid[5] = 0
    stray = faces(CENTRES).copy()
    stray[5] = stray[5] + np.array([0.0, 150.0])
    Ph, _ = cr.faces.rows_of(str(fx.save(folder / "stray_points.npz", stray, (W, H), first=0)), None, 0, N, 0, whole_face=False)
    Pn, _ = cr.faces.rows_of(str(POINTS), None, 0, N, 0, whole_face=False)
    plain = cr.cranium_box(Pn, (W, H), cr.HEAD_ROOM, hid)
    strayed = cr.cranium_box(Ph, (W, H), cr.HEAD_ROOM, hid)
    # THE CASE THAT GOES RED when a frame with no grey head sizes the box
    assert strayed["box"] == plain["box"] and strayed["cranium"]["frames_with_no_grey_head_joined_to_the_face"] == [5] \
        and strayed["cranium"]["those_frames_size_the_box"] is False, (strayed["box"], plain["box"])
    # an occlusion frame whose 2D face box touches a grey limb but has low confidence (< CRANIUM_SCORE_FLOOR) or a huge outlier
    # face height (> CRANIUM_FACE_OUTLIER * median, as on lead_0294_0414 frame 317: 389 px at score 0.04) does not size the box
    occ_kp = faces(CENTRES).copy()
    occ_kp[5] = CENTRES[5][None, :] + (occ_kp[5] - CENTRES[5][None, :]) * 3.5
    occ_sc = np.full(occ_kp.shape[:2], 0.9, np.float32)
    occ_sc[5] = 0.04
    P_occ, _ = cr.faces.rows_of(str(fx.save(folder / "occ_points.npz", occ_kp, (W, H), first=0, scores=occ_sc)), None, 0, N, 0, whole_face=False)
    occ_box = cr.cranium_box(P_occ, (W, H), cr.HEAD_ROOM, pics)
    assert occ_box["box"] == rec["box"], (occ_box["box"], rec["box"])
    # when more than half the span is in leave_out_frames with tiny bogus face heights (as on lead_2254_2598_c where 186 of 345
    # frames are left out), CRANIUM_FACE_OUTLIER is measured against the active kept & seen & scored frames' median, not diluted
    # by the excluded frames
    many_excl_kp = faces(CENTRES).copy()
    for k in range(7):
        many_excl_kp[k] = CENTRES[k][None, :] + (many_excl_kp[k] - CENTRES[k][None, :]) * 0.25
    P_many, _ = cr.faces.rows_of(str(fx.save(folder / "many_excl_points.npz", many_excl_kp, (W, H), first=0, scores=np.full(many_excl_kp.shape[:2], 0.9, np.float32))), None, 0, N, 0, whole_face=False)
    kept_plain = cr.cranium_box(Pn, (W, H), cr.HEAD_ROOM, pics, first=0, leave_out_frames=list(range(7)))
    kept_many = cr.cranium_box(P_many, (W, H), cr.HEAD_ROOM, pics, first=0, leave_out_frames=list(range(7)))
    assert kept_many["box"] == kept_plain["box"], (kept_many["box"], kept_plain["box"])
    # --box cuts by the box given, as it is
    given = cr.run(str(REF), str(POINTS), 0, N, folder / "given.mp4", "the check", box=rec["box"], say=quiet)
    assert given["box"] == rec["box"] and given["box_set_by"].startswith("given") and "cranium" not in given, given["box_set_by"]
    for kw, word in (({"fit": True}, "--cranium reads the box"), ({"head_track": str(REF)}, "--cranium reads the box"),
                     ({"box": [0, 0, 10, 10]}, "--cranium reads the box")):
        try:
            cr.run(str(mesh), str(POINTS), 0, N, folder / "refused.mp4", "the check", cranium=True, say=quiet, **kw)
        except SystemExit as exc:
            assert word in str(exc) and not (folder / "refused.mp4").exists(), (kw, exc)
        else:
            raise AssertionError(f"--cranium with {kw}: accepted")
    try:
        cr.run(str(REF), str(POINTS), 0, N, folder / "refused.mp4", "the check", box=[0, 0, W + 1, H], say=quiet)
    except SystemExit as exc:
        assert "not a box inside" in str(exc), exc
    else:
        raise AssertionError("a --box past the canvas: accepted")
    return (f"the grey head's box {rec['box']} holds every frame's head with its room and the neck, not the raised hand; a head reaching the "
            f"picture's top named on frame 5; a travelling head enlarges {trav['the_one_box_enlarges']:.3f} in the one box against "
            f"{median:.3f} a frame alone; a small head held to {cr.CRANIUM_MOST:g} (it would be {held['would_enlarge_without_it']:.2f}); "
            "--box cuts as given; three clashes refused")


case("one box, holding every face with its room, and the scale the record says", one_box_and_its_scale)
case("the sheet shows a change", the_sheet_shows_a_change)
case("frames from the middle of both files", frames_from_the_middle)
case("a head near an edge, and a head that crosses the frame", edges_and_the_whole_frame)
case("the original itself is fitted, then cut by the same box", the_original_fitted_then_cut)
case("the read-back can fail", the_read_back_can_fail)
case("the head box from the head track, with room above the crown", head_track_and_crown_room)
case("--cranium: the box round the body model's grey head", cranium_box_round_the_grey_head)
case("what it refuses", refusals)
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(finish())
