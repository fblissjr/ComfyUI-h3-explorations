#!/usr/bin/env python3
"""`bench/draw_face_points.py` against a face whose every line is known by construction, read from the VIDEO IT WROTE.

A motion video whose face is drawn from the wrong frame, or whose lines run through the points in the wrong
order, tells the model something about her face that is not so, and nothing downstream can see it: the load
builds, the gate passes and the render follows the reference. So this makes a small body-model video and a
points file in which every line's points, their order and where the face stands on each frame are chosen
here (`bench/_lib/face_lines_fixture.py`, on top of `bench/_lib/face_points_fixture.py`), runs the tool, and reads the pictures back.
The fixture names its midpoints out of order along each line, as the pose model does.

1. **The lines run through their points in order, and are as long as they say.** Every line and ring the tool
   makes of a frame is the one built here, point for point, whatever the numbers in the names; an iris ring
   encloses what a ring of its points encloses. The white in the written video is the lines' length, worked
   out here, times their width less the edge, and with dots on there is white at every point. With no dots
   the points that belong to no line are not drawn either, and the record lists the points drawn alone.
2. **The lines are where the points are.** The face stands somewhere else on every frame; the white found on
   a frame lies on that frame's lines and nowhere else, and its middle moves as the face does.
3. **Asked for frames from the middle of both files, it draws those frames.** The body-model video starts on
   one source frame and the points file on another; each mesh frame carries its own number as a grey level.
4. **Frames left bare come from a person and are bare.** No white on them, the record names them and who
   looked, and a list with no name is refused.
5. **The read-back can fail.** The same frames written one out of place fail it; a drawing on a bare frame
   fails it; a drawn frame with none fails it. A face that never moves passes, and is said to be untellable.
6. **`verify` catches points read on the wrong clock.** A second reader who takes the points file to begin a
   frame early or late is told a neighbour fits better; on the right clock it passes.
7. **A body-model video off the source's clock: figures, not a failure.** The three figures, their difference,
   the box and frames they were read on and what the measure cannot see are recorded and printed, and the run
   still passes (until 2026-10-12 it failed the run).
8. **An iris or a pupil ring is left out under the score line and drawn over it**, per eye and per frame, in
   the picture and in the record.
9. **The width rule, and lines too thin to read back.** With no width given it is the face's height over
   `h3_config.FACE_POINTS["face_over_width"]`, never under `least_px`; at the least width the lines leave no
   white to count, which is said and is not a failure.
10. **A part left out is not drawn, and the body model's own highlights are not taken for lines.** With the
    mouth left out there is no white where the lips' points are and the record says so; a name that is not a
    part is refused. A highlight on the body model that the encode lifts over white is not counted.
11. **What it refuses.** A points file with no names and none given; one that is not the pose model's; points
    made on another canvas; frames the files do not hold; a file that does not say its first frame.

Needs ffmpeg and ffprobe on the path. No model, no card, no server.

    <comfy venv python> bench/check_draw_face_points.py
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
from _lib import REPO, case, finish, needs  # noqa: E402

needs("ffmpeg and ffprobe on the path", bool(shutil.which("ffmpeg") and shutil.which("ffprobe")))

from _lib import face_points_fixture as fx  # noqa: E402
from _lib.frames import stream  # noqa: E402

import draw_face_points as df  # noqa: E402

sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402

W, H, N, WIDE = 384, 288, 24, 3.0
BASE = np.array([190.0, 110.0])
WORK = Path(tempfile.mkdtemp(prefix="check_draw_face_points_"))
quiet = lambda *_: None  # noqa: E731

#: Where the face stands on each frame, as steps from the one before that never repeat: a frame out of place shows.
STEP = np.array([[0, 0], [5, 2], [-3, 6], [8, -4], [-6, -5], [4, 7], [9, 1], [-7, 3], [3, -8], [6, 5], [-9, -2], [5, -6],
                 [-4, 8], [7, -3], [-8, 4], [2, 9], [10, -5], [-5, -7], [6, 3], [-3, -9], [8, 6], [-10, 2], [4, -4], [-6, 7]], float)
CENTRES = BASE + np.cumsum(STEP, 0) * 0.5 + np.array([0.0, 0.0])

from _lib.face_lines_fixture import CLOCK, FACE_PX, INDEX, IRIS_R, LIPS, NAMES, OPEN, PUPIL_R, face, lines_of, truth  # noqa: E402,F401
from _lib import face_lines_fixture as fl  # noqa: E402


def save(path: Path, keypoints: np.ndarray, size=(W, H), first: int | None = None, scores: np.ndarray | None = None, names: bool = True) -> Path:
    """A points file as `bench/pose_along_shot.py` writes one, with the fixture's names, on this check's canvas."""
    return fl.save(path, keypoints, size, first=first, scores=scores, names=names)


def long(pts: np.ndarray, closed: bool) -> float:
    run = np.vstack([pts, pts[:1]]) if closed else pts
    return float(np.linalg.norm(np.diff(run, axis=0), axis=1).sum())


def video(path: Path, frames: np.ndarray) -> Path:
    """[n, h, w, 3] as a lossless video."""
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{frames.shape[2]}x{frames.shape[1]}",
                            "-r", "24", "-i", "-", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", str(path)], stdin=subprocess.PIPE)
    enc.communicate(frames.tobytes())
    assert enc.returncode == 0, path
    return path


def grey(levels) -> np.ndarray:
    """A body model with nothing on it: frame k is a flat grey of `levels[k]`, so a frame says which one it is."""
    return np.stack([np.full((H, W, 3), v, np.uint8) for v in levels])


def white(path: Path, n: int) -> np.ndarray:
    """[n, h, w] of bool: a line's white in a written video, read here and not by the tool."""
    return np.stack(list(stream(str(path), (W, H), 0, n, pix="rgb24"))).min(-1) > df.WHITE


def near(xy: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """From each pixel to the nearest of the points `pts` sampled densely along this check's own lines."""
    return np.linalg.norm(xy[:, None, :] - pts[None], axis=-1).min(1)


def dense(centre, opening: float = OPEN, step: float = 0.25) -> np.ndarray:
    """Points every quarter pixel along every line of a frame, and the dots: where white may be, by this check's construction."""
    out = [np.array([xy for _, xy in lines_of(centre, opening)["dots"]]), np.array([centre + (0, -10), centre + (0, -10 + FACE_PX), centre + (0, 18)])]
    for pts, closed in truth(centre, opening).values():
        run = np.vstack([pts, pts[:1]]) if closed else pts
        for a, b in zip(run[:-1], run[1:]):
            k = max(2, int(np.linalg.norm(b - a) / step) + 1)
            out.append(a + (b - a) * np.linspace(0, 1, k)[:, None])
    return np.concatenate(out)


MESH = video(WORK / "mesh.mkv", grey([90] * N))
POINTS = save(WORK / "shot_points.npz", face(CENTRES), first=0)


def in_order_and_as_long() -> str:
    pts = face(CENTRES[:1])[0]
    made = {n: (p, c) for n, p, c in df.shapes(pts, np.full(len(NAMES), 0.9), INDEX, 0.5)}
    want = truth(CENTRES[0])
    assert set(want) <= set(made), sorted(set(want) - set(made))
    for name, (p, closed) in want.items():
        got, got_closed = made[name]
        assert got_closed == closed and len(got) == len(p), (name, len(got), len(p))
        if "iris" in name or "pupil" in name:                         # a ring by angle may start anywhere: hold what it encloses
            r = IRIS_R if "iris" in name else PUPIL_R
            assert abs(df.fp.ring_area(got) - 2 * np.sqrt(2) * r * r) < 1e-3, (name, df.fp.ring_area(got))
        else:
            assert np.allclose(got, p, atol=1e-3) or np.allclose(got, p[::-1], atol=1e-3), (name, got, p)
    by_hand = sum(long(p, c) for p, c in want.values())
    assert abs(df.length(list((n, *made[n]) for n in want)) - by_hand) < 1e-2
    alone = sorted(n for n, (p, _) in made.items() if len(p) == 1)
    assert alone == sorted(["center_of_glabella", "center_of_labiomental_groove", "tip_of_chin", "tip_of_nose", "outer_corner_of_l_nostril",
                            "outer_corner_of_r_nostril", "upper_corner_of_l_nostril", "upper_corner_of_r_nostril"]), alone
    # in the picture: the white is the length worked out here times the width less the edge, at several widths, with no dots
    still = save(WORK / "still_points.npz", face(np.repeat(CENTRES[:1], 4, 0)), first=0)
    said = []
    for wide in (2.0, 3.0, 4.0):
        rec = df.run(str(MESH), str(still), 0, 4, WORK / f"long_{wide:g}.mp4", "the check", width=wide, dots=False, say=quiet)
        assert rec["passed"], rec["read_back"]
        found = white(WORK / f"long_{wide:g}.mp4", 4).sum((1, 2)).mean()
        want_white = by_hand * (wide - df.EDGE_PX)
        assert abs(found - want_white) < 0.12 * want_white, (wide, found, want_white)
        assert abs(rec["read_back"]["white_px_expected_a_frame"]["median"] - found) < 0.12 * found, (wide, rec["read_back"]["white_px_expected_a_frame"], found)
        said.append(f"{wide:g} px: {found:.0f} against {want_white:.0f}")
    dotted = df.run(str(MESH), str(still), 0, 4, WORK / "dotted.mp4", "the check", width=WIDE, say=quiet)
    got = white(WORK / "dotted.mp4", 4)
    assert dotted["passed"] and dotted["a_dot_at_every_point"] is True
    for name, (x, y) in zip(NAMES, pts):
        if "pupil" not in name and "iris" not in name and not name.endswith("inner_lip"):
            assert got[0, int(round(y)), int(round(x))], f"no white at {name} with dots on"
    assert got.sum((1, 2)).mean() > white(WORK / "long_3.mp4", 4).sum((1, 2)).mean()
    # no dots is no dots: a point that belongs to no line is drawn with the dots and not without them
    bare, cx, cy = white(WORK / "long_3.mp4", 4), *CENTRES[0]
    for name, (x, y) in (("center_of_glabella", (cx, cy - 10)), ("center_of_labiomental_groove", (cx, cy + 54)), ("tip_of_chin", (cx, cy - 10 + FACE_PX))):
        spot = (slice(int(round(y)) - 2, int(round(y)) + 3), slice(int(round(x)) - 2, int(round(x)) + 3))
        assert got[0][spot].any() and not bare[:, spot[0], spot[1]].any(), f"{name}: drawn alone with dots {got[0][spot].sum()}, without {bare[:, spot[0], spot[1]].sum()}"
    plain = json.loads((WORK / "long_3.json").read_text())
    assert plain["points_drawn_alone"] == [] and plain["a_dot_at_every_point"] is False, plain["points_drawn_alone"]
    assert dotted["points_drawn_alone"] == alone, dotted["points_drawn_alone"]
    assert abs(dotted["read_back"]["white_px_expected_a_frame"]["median"] - got.sum((1, 2)).mean()) < 0.12 * got.sum((1, 2)).mean(), dotted["read_back"]
    return (f"every line runs through its points in the order built here, {by_hand:.0f} px of line; white by width with no dots: " + "; ".join(said)
            + "; with dots there is white at every point, and without them none at the points that belong to no line")


def where_the_points_are() -> str:
    rec = df.run(str(MESH), str(POINTS), 0, N, WORK / "a.mp4", "the check", width=WIDE, say=quiet)
    # a whole run in the contrast form reads back as passed: the sclera's white is left out of what the lines are held to
    hi = df.run(str(MESH), str(POINTS), 0, N, WORK / "a_contrast.mp4", "the check", width=WIDE, iris="contrast", say=quiet)
    assert hi["passed"] and hi["eye_rings"]["drawn"] == "contrast", hi["read_back"]
    pts = df.rows_of(str(POINTS), None, 0, N, None)[0]
    fills = [df.discs(p, sc, pts["index"], h3_config.POSE_READING["eye_ring_score_at_least"]) for p, sc in zip(pts["keypoints"], pts["scores"])]
    but = df.filled_px(fills, (W, H))
    mesh_px = np.stack(list(stream(str(MESH), (W, H), 0, N, pix="rgb24")))
    with_sclera, without = df.white_at(WORK / "a_contrast.mp4", mesh_px), df.white_at(WORK / "a_contrast.mp4", mesh_px, but=but)
    inside = lambda found, f: int(but[f][found[f][:, 1].astype(int), found[f][:, 0].astype(int)].sum())      # noqa: E731
    # THE CASE THAT GOES RED when the sclera's white is read back as a line's
    assert sum(inside(with_sclera, f) for f in range(N)) > 0 and sum(inside(without, f) for f in range(N)) == 0, "the sclera's white is not left out"
    assert hi["read_back"]["white_px_a_frame"]["most"] == max(len(v) for v in without), (hi["read_back"]["white_px_a_frame"], max(len(v) for v in without))
    assert rec["passed"] and not rec["read_back"]["no_neighbour_can_be_told_from_the_frame"], rec["read_back"]
    got = white(WORK / "a.mp4", N)
    mids, off = [], []
    for f in range(N):
        ys, xs = np.nonzero(got[f])
        assert len(xs), f"frame {f} has no white"
        d = near(np.stack([xs, ys], 1).astype(float), dense(CENTRES[f]))
        off.append(d.max())
        assert d.max() <= WIDE + 0.5, (f, d.max())                    # no white further from a line than a dot reaches
        mids.append((xs.mean(), ys.mean()))
    moved = np.array(mids) - np.array(mids)[0]
    assert np.abs(moved - (CENTRES - CENTRES[0])).max() < 0.75, np.abs(moved - (CENTRES - CENTRES[0])).max()
    fit = rec["read_back"]["white_from_the_lines_px_mean_by_frames_of_slip"]
    assert fit["0"] < WIDE / 2 and fit["1"] > fit["0"] + 4 * df.SLIP_PX and fit["-1"] > fit["0"] + 4 * df.SLIP_PX, fit
    return (f"on all {N} frames no white is further than {max(off):.1f} px from the lines built here, and its middle moves with the face; "
            f"mean distance {fit['0']} px on the frame's own lines against {fit['1']} and {fit['-1']} a frame off")


def frames_from_the_middle() -> str:
    mesh_first, points_first, first, count = 95, 100, 105, 12
    mesh = video(WORK / "mesh_mid.mkv", grey([40 + 3 * k for k in range(40)]))      # frame k of the file is source frame 95 + k
    centres = BASE + np.array([[(k * 7) % 23 - 11, (k * 5) % 17 - 8] for k in range(30)], float)   # row k of the points is source frame 100 + k
    pts = save(WORK / "mid_points.npz", face(centres), first=points_first)
    rec = df.run(str(mesh), str(pts), first, count, WORK / "mid.mp4", "the check", width=WIDE, mesh_first=mesh_first, say=quiet)
    assert rec["passed"], rec["read_back"]
    out = np.stack(list(stream(str(WORK / "mid.mp4"), (W, H), 0, count, pix="rgb24")))
    for j in range(count):
        level = int(np.median(out[j, :20, :20]))                                  # a corner, away from the face
        assert abs(level - (40 + 3 * (first - mesh_first + j))) <= 2, (j, level)
        ys, xs = np.nonzero(out[j].min(-1) > df.WHITE)
        d = near(np.stack([xs, ys], 1).astype(float), dense(centres[first - points_first + j]))
        assert len(xs) and d.max() <= WIDE + 0.5, (j, d.max())
    assert rec["frames"] == [105, 116] and rec["made_from"]["its_first_frame_is_source_frame"] == 95 and rec["made_from"]["their_first_row_is_source_frame"] == 100
    return "frames 105 to 116 of a body-model video from 95 and points from 100: each written frame is its own mesh frame with its own face"


def bare_frames() -> str:
    rec = df.run(str(MESH), str(POINTS), 0, N, WORK / "bare.mp4", "the check", width=WIDE, bare=[6, 7, 30], bare_by="a person", say=quiet)
    got = white(WORK / "bare.mp4", N).sum((1, 2))
    assert rec["passed"] and got[6] == 0 and got[7] == 0 and got[5] > 0 and got[8] > 0, got[4:10]
    assert rec["frames_left_bare"] == [6, 7] and rec["left_bare_by"] == "a person", rec           # 30 is not in the shot and is not listed
    saved = json.loads((WORK / "bare.json").read_text())
    assert saved["frames_left_bare"] == [6, 7] and any("turned away" in line for line in saved["not_known"])
    try:
        df.run(str(MESH), str(POINTS), 0, N, WORK / "x.mp4", "the check", width=WIDE, bare=[6], say=quiet)
    except SystemExit as exc:
        assert "--bare-by" in str(exc), exc
    else:
        raise AssertionError("frames left bare with nobody named were accepted")
    return "two frames left bare by a named person carry no white and are in the record; a list with no name is refused"


def the_read_back_can_fail() -> str:
    mesh = np.stack(list(stream(str(MESH), (W, H), 0, N, pix="rgb24")))
    kp = face(CENTRES)
    drawings = [df.shapes(p, np.full(len(NAMES), 0.9), INDEX, 0.5) for p in kp]
    pictures = df.draw(mesh, drawings, WIDE, set())
    drawn = np.ones(N, bool)
    expected = np.array([df.white_expected(d, WIDE, True) for d in drawings])
    df.marks.write_video(WORK / "good.mp4", pictures)
    good = df.read_back(WORK / "good.mp4", mesh, drawings, expected, drawn, WIDE)
    assert good["passed"] and not good["cannot_tell"], good["lags"]
    df.marks.write_video(WORK / "late.mp4", np.concatenate([pictures[:1], pictures[:-1]]))   # every frame one late, the first repeated
    late = df.read_back(WORK / "late.mp4", mesh, drawings, expected, drawn, WIDE)
    assert not late["passed"] and late["slipped"] and late["lags"][1] < late["lags"][0], late["lags"]
    df.marks.write_video(WORK / "early.mp4", np.concatenate([pictures[1:], pictures[-1:]]))
    early = df.read_back(WORK / "early.mp4", mesh, drawings, expected, drawn, WIDE)
    assert not early["passed"] and early["lags"][-1] < early["lags"][0], early["lags"]
    drawn2 = drawn.copy(); drawn2[9] = False                                        # frame 9 was to be bare and is not
    painted = df.read_back(WORK / "good.mp4", mesh, drawings, np.where(drawn2, expected, 0.0), drawn2, WIDE)
    assert not painted["passed"] and painted["on_bare"] > 0, painted["on_bare"]
    blank = pictures.copy(); blank[12] = mesh[12]                                   # a drawn frame written with nothing on it
    df.marks.write_video(WORK / "blank.mp4", blank)
    lost = df.read_back(WORK / "blank.mp4", mesh, drawings, expected, drawn, WIDE)
    assert not lost["passed"] and lost["drawn_frames_with_no_white"] == [12], lost["drawn_frames_with_no_white"]
    still = [drawings[0]] * N                                                       # a face that never moves: nothing to tell a frame by
    df.marks.write_video(WORK / "still.mp4", df.draw(mesh, still, WIDE, set()))
    same = df.read_back(WORK / "still.mp4", mesh, still, np.full(N, expected[0]), drawn, WIDE)
    assert same["passed"] and same["cannot_tell"], same["lags"]
    return (f"as drawn it passes ({good['lags'][0]:.2f} px against {min(good['lags'][-1], good['lags'][1]):.2f} a frame off); every frame one late or one early fails; "
            "a drawing on a bare frame fails; a drawn frame with none fails; a face that never moves passes and is said to be untellable")


def verify_catches_the_wrong_clock() -> str:
    df.run(str(MESH), str(POINTS), 0, N, WORK / "v.mp4", "the check", width=WIDE, say=quiet)
    assert df.verify(str(WORK / "v.mp4"), str(MESH), str(POINTS), 0, N, width=WIDE, say=quiet)
    # a reader who takes the points' first row to be source frame 1, or -1, reads every frame against a neighbour's points
    early = df.verify(str(WORK / "v.mp4"), str(MESH), str(POINTS), 1, N - 2, points_first=1, video_first=0, mesh_first=0, width=WIDE, say=quiet)
    late = df.verify(str(WORK / "v.mp4"), str(MESH), str(POINTS), 1, N - 2, points_first=-1, video_first=0, mesh_first=0, width=WIDE, say=quiet)
    assert not early and not late, (early, late)
    assert not df.verify(str(MESH), str(MESH), str(POINTS), 0, N, width=WIDE, say=quiet)   # a video with no lines at all is not a pass
    return "on the right clock it passes; the points read a frame early or a frame late fail; a video with no lines fails"


def moving(n: int = N):
    """A source and a body model that move together by steps that never repeat: (source frames, mesh frames)."""
    step = np.array([0, 3, 9, 2, 14, 6, 1, 11, 4, 16, 7, 2, 13, 5, 18, 3, 10, 6, 1, 15, 8, 4, 12, 2, 9, 5][:n])
    x = 40 + np.cumsum(step) % 250
    rng = np.random.default_rng(7)
    ground = rng.integers(60, 120, (H, W, 3), dtype=np.uint8)
    src, mesh = [], []
    for k in range(n):
        s = ground.copy(); s[200:270, x[k]:x[k] + 50] = 200
        m = np.zeros((H, W, 3), np.uint8); m[200:270, x[k]:x[k] + 50] = 120
        src.append(s); mesh.append(m)
    return np.stack(src), np.stack(mesh)


def off_the_sources_clock() -> str:
    src, mesh = moving()
    source = video(WORK / "source.mkv", src)
    on = df.run(str(video(WORK / "mesh_on.mkv", mesh)), str(POINTS), 0, N, WORK / "on.mp4", "the check", width=WIDE, source=str(source), say=quiet)
    clock = on["body_model_against_the_source"]
    assert on["passed"] and clock["best_neighbour_less_no_slip"] < 0, clock
    # the figures say what they were read on and what they cannot see, and decide nothing
    assert clock["measured_on"]["frames"] == [0, N - 1] and len(clock["measured_on"]["box"]) == 4 and "a second person" in clock["cannot_see"], clock
    assert "no part of `passed`" in clock["decides"], clock
    assert (WORK / "on_sheet.jpg").exists() and on["sheet"]["looked_at_by"] == "" and len(on["sheet"]["frames"]) == 6
    said = []
    for name, frames in (("late", np.concatenate([mesh[:1], mesh[:-1]])), ("early", np.concatenate([mesh[1:], mesh[-1:]]))):
        rec = df.run(str(video(WORK / f"mesh_{name}.mkv", frames)), str(POINTS), 0, N, WORK / f"{name}.mp4", "the check", width=WIDE, source=str(source), say=said.append)
        # a body model a frame off the source: the figures show it, and the run is NOT failed on them (a figure of degree is data)
        assert rec["passed"] and rec["read_back"]["passed"], (name, rec["body_model_against_the_source"], rec["read_back"])
        assert rec["body_model_against_the_source"]["best_neighbour_less_no_slip"] > 0, (name, rec["body_model_against_the_source"])
        assert "a_neighbour_fits_better" not in rec["body_model_against_the_source"] and "says" not in rec["body_model_against_the_source"]
    lines = [line for line in said if line.startswith("the body model against the source")]
    assert len(lines) == 2 and all("no part of PASSED" in line and "the best neighbour less no slip 1." in line and "anything else that moves counts too" in line
                                   for line in lines), said
    assert not any(line.startswith("FAILED") for line in said), said
    none = df.run(str(WORK / "mesh_on.mkv"), str(POINTS), 0, N, WORK / "nosrc.mp4", "the check", width=WIDE, say=quiet)
    assert none["passed"] and none["body_model_against_the_source"] == "not looked at: no --source"
    part = video(WORK / "source_from_4.mkv", src[4:])                               # a source cut from the original: its frame 0 is source frame 4
    cut = df.run(str(WORK / "mesh_on.mkv"), str(POINTS), 4, N - 4, WORK / "cut.mp4", "the check", width=WIDE, mesh_first=0, source=str(part), source_first=4, say=quiet)
    assert cut["passed"] and cut["body_model_against_the_source"]["best_neighbour_less_no_slip"] < 0 and cut["sheet"]["frames"][0] == 4, cut["body_model_against_the_source"]
    assert cut["body_model_against_the_source"]["measured_on"]["frames"] == [4, N - 1]
    try:
        df.run(str(WORK / "mesh_on.mkv"), str(POINTS), 4, N - 4, WORK / "cut2.mp4", "the check", width=WIDE, mesh_first=0, source=str(part), say=quiet)
    except SystemExit as exc:
        assert "--source-first" in str(exc), exc
    else:
        raise AssertionError("a cut source read as if it began on source frame 0 was accepted")
    return ("a body model that moves with the source passes; the same one a frame late or a frame early fails the run while its lines read back; "
            "with no source the look is said not to be made; a source cut from the original is read from the frame it is said to begin on")


def eye_rings_by_score() -> str:
    line = h3_config.POSE_READING["eye_ring_score_at_least"]
    scores = np.full((N, len(NAMES)), 0.9)
    def ring(side, kind):                                                           # noqa: E306
        return [INDEX[n] for n in NAMES if n.startswith(f"{side}_border_of_{kind}") or n == f"{side}_center_of_{kind}"]
    scores[:, ring("l", "pupil")] = line - 0.2                                      # her left pupil is never scored well
    scores[:, ring("r", "pupil")] = line - 0.2
    scores[12:, ring("r", "iris")] = line - 0.05                                    # her right iris is lost half way through
    still = np.repeat(CENTRES[:1], N, 0)
    pts = save(WORK / "ring_points.npz", face(still), first=200, scores=scores)
    rec = df.run(str(MESH), str(pts), 200, N, WORK / "rings.mp4", "the check", width=2.0, dots=False, mesh_first=200, say=quiet)
    gone = rec["eye_rings"]["frames_a_ring_is_left_out_on"]
    assert rec["passed"] and rec["eye_rings"]["score_at_least"] == line
    assert gone["l_iris"] == [] and gone["r_iris"] == list(range(212, 224)) and gone["l_pupil"] == list(range(200, 224)) == gone["r_pupil"], gone
    got = white(WORK / "rings.mp4", N)
    def at(side, r):                                                                # noqa: E306  white on the top of a ring of radius r round that eye's centre
        cx = CENTRES[0][0] + (1 if side == "l" else -1) * (fx.EYE_APART / 2 + 0.5 * fx.EYE_WIDE)
        cy = CENTRES[0][1] - OPEN / 2 + 1
        return got[:, int(round(cy - r)), int(round(cx))]
    assert at("l", IRIS_R).all() and at("r", IRIS_R)[:12].all() and not at("r", IRIS_R)[12:].any(), (at("l", IRIS_R), at("r", IRIS_R))
    assert not at("l", 0.0).any() and not at("r", 0.0).any()                        # no pupil ring: nothing at the eye's middle
    both = save(WORK / "ring_both.npz", face(still), first=0)
    df.run(str(MESH), str(both), 0, N, WORK / "rings_all.mp4", "the check", width=2.0, dots=False, say=quiet)
    assert white(WORK / "rings_all.mp4", N)[:, int(round(CENTRES[0][1] - OPEN / 2 + 1 - PUPIL_R)), int(round(CENTRES[0][0] + fx.EYE_APART / 2 + 0.5 * fx.EYE_WIDE))].all()
    return f"under {line:g} a ring is left out, eye by eye and frame by frame, in the picture and in the record; over it the iris and the pupil are both drawn"


def the_width_rule() -> str:
    look = h3_config.FACE_POINTS
    rec = df.run(str(MESH), str(POINTS), 0, N, WORK / "rule.mp4", "the check", say=quiet)
    want = max(FACE_PX / look["face_over_width"], look["least_px"])
    assert abs(rec["line_width_px"] - want) < 1.0 / df.SUPER and "face_over_width" in rec["width_set_by"], (rec["line_width_px"], want, rec["width_set_by"])
    by_hand = df.run(str(MESH), str(POINTS), 0, N, WORK / "hand.mp4", "the check", width=2.5, say=quiet)
    assert by_hand["line_width_px"] == 2.5 and by_hand["width_set_by"] == "--width"
    thin = df.run(str(MESH), str(POINTS), 0, N, WORK / "thin.mp4", "the check", width=look["least_px"], dots=False, say=quiet)
    listed = thin["read_back"]["lines_too_thin_to_leave_much_white"]["frames"]
    assert thin["passed"] and listed == list(range(N)), (thin["passed"], len(listed), thin["read_back"]["white_px_a_frame"])
    return (f"with no width given it is the face's height over face_over_width ({rec['line_width_px']:g} px here); a width by hand is recorded as one; "
            f"at {look['least_px']:g} px with no dots every frame is listed as too thin and the run still passes")


def left_out_and_highlights() -> str:
    rec = df.run(str(MESH), str(POINTS), 0, N, WORK / "nomouth.mp4", "the check", width=WIDE, leave_out=frozenset({"mouth"}), say=quiet)
    assert rec["passed"] and rec["parts_left_out"] == ["mouth"], rec["parts_left_out"]
    got, full = white(WORK / "nomouth.mp4", N), white(WORK / "a.mp4", N)
    for f in range(N):
        cx, cy = CENTRES[f].round().astype(int)
        lips = (slice(cy + 29, cy + 49), slice(cx - 22, cx + 23))      # the lip rings, clear of the nose's base above and the groove's dot below
        assert not got[f][lips].any() and full[f][lips].sum() > 100, (f, got[f][lips].sum(), full[f][lips].sum())
        eyes, before = got[f][cy - 16:cy + 6].sum(), full[f][cy - 16:cy + 6].sum()
        assert before > 0 and abs(int(eyes) - int(before)) <= 0.02 * before, (f, eyes, before)   # her eyes are drawn as before, to what a second encode moves
    assert df.verify(str(WORK / "nomouth.mp4"), str(MESH), str(POINTS), 0, N, width=WIDE, leave_out=frozenset({"mouth"}), say=quiet)
    try:
        df.parts_arg("mouth,ears")
    except SystemExit as exc:
        assert "ears" in str(exc), exc
    else:
        raise AssertionError("a part that is not one was accepted")
    # a body model with a highlight just under white, which the written video holds just over it
    mesh = grey([90] * 4); mesh[:, 20:40, 20:60] = df.WHITE - 7
    drawings = [df.shapes(p, np.full(len(NAMES), 0.9), INDEX, 0.5) for p in face(CENTRES[:4])]
    lifted = df.draw(mesh, drawings, WIDE, set()); lifted[:, 20:40, 20:60] = df.WHITE + 6
    written = video(WORK / "lifted.mkv", lifted)
    found = df.white_at(written, mesh)
    assert all(len(v) and not ((v[:, 0] < 70) & (v[:, 1] < 50)).any() for v in found), [len(v) for v in found]
    back = df.read_back(written, mesh, drawings, np.array([df.white_expected(d, WIDE, True) for d in drawings]), np.ones(4, bool), WIDE)
    assert back["passed"] and not back["off_the_lines"], back["lags"]
    return ("with the mouth left out there is no white on the lips' points, her eyes are as before, the record says so and a second reader told the same passes; "
            "a name that is not a part is refused; a highlight the encode lifts over white is not counted and the read-back passes")


def refusals() -> str:
    kp = face(CENTRES)
    old = save(WORK / "old_points.npz", kp, first=0, names=False)
    lids_only = fx.save(WORK / "lids_points.npz", fx.face(np.full(N, OPEN), CENTRES), (W, H), first=0)
    for what, points, first, mesh_first, word in (
            ("no names", old, 0, None, "holds no point names"),
            ("not the pose model's points", lids_only, 0, None, "names a face is drawn from"),
            ("another canvas", save(WORK / "big_points.npz", kp, size=(W * 2, H * 2), first=0), 0, None, "not on one canvas"),
            ("frames the points do not hold", POINTS, 10, None, "were asked for"),
            ("frames the body model does not hold", POINTS, 0, 5, "were asked for"),
            ("no first frame said", save(WORK / "nofirst_points.npz", kp), 0, None, "--points-first")):
        try:
            df.run(str(MESH), str(points), first, N, WORK / "r.mp4", "the check", width=WIDE, mesh_first=mesh_first, say=quiet)
        except (SystemExit, ValueError) as exc:
            assert word in str(exc), (what, exc)
        else:
            raise AssertionError(f"{what}: accepted")
    ok = df.run(str(MESH), str(old), 0, N, WORK / "named.mp4", "the check", width=WIDE, names=NAMES, say=quiet)
    assert ok["passed"]
    return ("no names, points that are not the pose model's, another canvas, frames either file lacks and a file that does not say its first frame "
            "are each refused; an old file reads with its names given")



def the_iris_between_the_lids() -> str:
    """2026-10-14: with the lids lowered, a whole iris ring stood above them and read as an open eye looking down."""
    t = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    lids = np.stack([100 + 30 * np.cos(t), 100 + 4 * np.sin(t)], 1)            # a narrow opening, 8 px tall
    iris = np.stack([100 + 10 * np.cos(t), 100 + 10 * np.sin(t)], 1)            # a ring 20 px across, larger than the opening
    runs, closed, off = df.between_the_lids(iris, lids)
    assert not closed and runs and 0.4 < off < 0.9, (closed, len(runs), off)
    for run in runs:
        assert df.inside(run, lids).all(), "a stretch drawn outside the lids"
    small = np.stack([100 + 2 * np.cos(t), 100 + 2 * np.sin(t)], 1)
    assert df.between_the_lids(small, lids)[1:] == (True, 0.0), "a ring wholly inside is not drawn whole"
    far = iris + np.array([0, 60])
    assert df.between_the_lids(far, lids) == ([], False, 1.0), "a ring wholly outside is drawn"
    # through a frame's drawing: the default draws only open stretches inside; `whole` draws the closed ring, as before
    p = face([(160.0, 100.0)], opening=1.0)[0]                                 # lids nearly shut: the iris (radius IRIS_R) is larger than the opening
    s = np.full(len(NAMES), 0.9)
    outside: dict = {}
    between = [x for x in df.shapes(p, s, INDEX, 0.0, frozenset(), df.IRIS_FORMS[0], outside) if x[0].endswith("_iris_ring")]
    whole = [x for x in df.shapes(p, s, INDEX, 0.0, frozenset(), df.IRIS_FORMS[1]) if x[0].endswith("_iris_ring")]
    assert all(c for _, _, c in whole) and len(whole) == 2, whole
    assert outside and all(0 < v <= 1 for v in outside.values()), outside
    assert between and all(not c for _, _, c in between) and all(v > 0 for k, v in outside.items() if "iris" in k), (between, outside)
    # FILLED (2026-10-14, behind its flag): no iris or pupil ring is drawn; each iris is a disc, filled dark only inside its lash ring
    filled = df.shapes(p, s, INDEX, 0.0, frozenset(), df.IRIS_FORMS[2])
    assert not any(n.endswith("_ring") and ("iris" in n or "pupil" in n) for n, _, _ in filled), filled
    disc = df.discs(p, s, INDEX, 0.0)
    assert [n for n, _, _ in disc] == ["l_iris_disc", "r_iris_disc"] and all(len(d) == df.RING_SAMPLES for _, d, _ in disc), disc
    assert df.discs(p, s, INDEX, 2.0) == [] and df.discs(p, s, INDEX, 0.0, frozenset({"eyes"})) == [], "an iris under the score line, or with the eyes left out, was filled"
    W_, H_ = 320, 200
    mesh = np.full((1, H_, W_, 3), 128, np.uint8)
    lids = np.array([[100, 100], [140, 90], [180, 100], [140, 110]], float)
    iris = np.stack([140 + 15 * np.cos(np.linspace(0, 2 * np.pi, df.RING_SAMPLES, endpoint=False)),
                     100 + 15 * np.sin(np.linspace(0, 2 * np.pi, df.RING_SAMPLES, endpoint=False))], 1)
    pic = df.draw(mesh, [[]], 1.0, set(), False, [[("l_iris_disc", iris, lids)]])[0]
    dark = pic.max(-1) < 40
    assert dark[100, 140] and dark[100, 130] and not dark[88, 140] and not dark[112, 140] and not dark[100, 160], "the disc is not dark inside the lids only"
    assert (df.draw(mesh, [[]], 1.0, set(), False, None)[0] == 128).all(), "something was drawn with no fills"
    # CONTRAST (the owner's word, 2026-10-14): inside the lids the sclera solid white and the iris solid black, after the lines; outside, nothing
    hi = df.draw(mesh, [[]], 1.0, set(), False, [[("l_iris_disc", iris, lids)]], contrast=True)[0]
    assert hi[100, 140].max() < 40 and hi[100, 110].min() > 215 and hi[100, 170].min() > 215, "the iris is not black, or the sclera not white, inside the lids"
    assert (hi[80, 140] == 128).all() and (hi[120, 140] == 128).all() and (hi[100, 190] == 128).all(), "something was filled outside the lids"
    # THE EDGE STAYS (an outside reviewer, 2026-10-14): with the lash ring drawn over the fill, its black edge lies between the white
    # sclera and the white line; filled after the lines, the white ran from the sclera into the line with no dark between
    ring = [("l_lash_ring", lids, True)]
    edged = df.draw(mesh, [ring], 3.0, set(), False, [[("l_iris_disc", iris, lids)]], contrast=True)[0]
    cut = edged[100, 100:115].min(-1).astype(int)                       # from the lid's left corner into the sclera
    assert cut[-1] > 215 and cut.min() < 150, ("no dark edge between the white sclera and the white lid line", cut.tolist())
    but = df.filled_px([[("l_iris_disc", iris, lids)]], (W_, H_))
    assert but[0, 100, 110] and but[0, 100, 140] and not but[0, 80, 140], "the read-back's leave-out is not the lids' inside"
    return (f"a ring larger than a narrow opening is drawn only where it lies between the lids ({off:.2f} of it outside), a ring wholly inside "
            "is drawn closed and one wholly outside not at all; on nearly shut lids the default draws open stretches and `whole` both rings closed")

case("the iris between the lids", the_iris_between_the_lids)
case("the lines run through their points in order, and are as long as they say", in_order_and_as_long)
case("the lines are where the points are", where_the_points_are)
case("frames from the middle of both files", frames_from_the_middle)
case("frames left bare come from a person and are bare", bare_frames)
case("the read-back can fail", the_read_back_can_fail)
case("verify catches points read on the wrong clock", verify_catches_the_wrong_clock)
case("a body-model video off the source's clock: the figures are printed and recorded, and decide nothing", off_the_sources_clock)
case("an iris or a pupil ring is left out under the score line", eye_rings_by_score)
case("the width rule, and lines too thin to read back", the_width_rule)
case("a part left out is not drawn, and a highlight is not a line", left_out_and_highlights)
case("what it refuses", refusals)
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(finish())
