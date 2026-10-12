#!/usr/bin/env python3
"""`bench/draw_lid_marks.py` against a face whose eyes are known by construction, read from the VIDEO IT WROTE.

A motion video with the marks of the wrong frame tells the model her eyes are open when they are shut, and
nothing downstream can see it: the load builds, the gate passes and the render follows the reference. So this
makes a small body-model video and a points file in which every frame's eye opening is a number chosen here
(`bench/_lib/face_points_fixture.py`), runs the tool, and reads the pictures back.

1. **The marks are where the points are and as large as they say.** On every frame the white in the written
   video sits on the eyes and its size is the outline's own area times the factor squared, the area worked out
   here by hand and not by the tool's code.
2. **Asked for frames from the middle of both files, it draws those frames.** The body-model video starts on
   one source frame and the points file on another; each mesh frame carries its own number as a grey level.
   The written frames show the mesh frames and the openings of the source frames asked for.
3. **Frames left bare come from a person and are bare.** No white on them, the record names them and who
   looked, and a list with no name is refused.
4. **The read-back can fail.** The same frames written one out of place fail it; a mark painted on a bare
   frame fails it. A highlight of the body model that the encode lifts over white is not a mark: a bare frame
   with one still reads as bare, and the read-back passes.
5. **`verify` catches points read on the wrong clock.** A second reader who takes the points file to begin a
   frame early or late is told a neighbour fits better; on the right clock it passes.
6. **A body-model video off the source's clock: figures, not a failure.** With a source that moves as the body
   model does, no slip fits best; with the body model given a frame late or early a neighbour fits best, the
   record carries the three figures, their difference, the box and frames they were read on and what the
   measure cannot see, the line is printed, and the run still passes: a figure of degree is data
   (until 2026-10-12 it failed the run).
7. **The size rule and what the tool says of a small lid gap.** With no factor given, the factor is
   `h3_config.LID_MARKS["face_px"]` over the median face height; lash lines closer than
   `h3_config.POSE_READING["lid_gap_px_to_draw"]` are said to be under the line, and the marks are still drawn.
8. **What it refuses.** A points file with no names and none given; points made on another canvas; frames the
   files do not hold.

Needs ffmpeg and ffprobe on the path. No model, no card, no server.

    <comfy venv python> bench/check_draw_lid_marks.py
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

import draw_lid_marks as dm  # noqa: E402

sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402

W, H, N, GROW = 256, 192, 24, 2.0
CENTRE = (128.0, 80.0)
WORK = Path(tempfile.mkdtemp(prefix="check_draw_lid_marks_"))
quiet = lambda *_: None  # noqa: E731

#: An opening a frame, in pixels, that rises, falls and never repeats a step: a frame out of place shows.
OPEN = np.array([2, 3, 5, 8, 12, 9, 6, 4, 3, 2, 2, 4, 7, 11, 14, 10, 7, 5, 3, 2, 3, 6, 9, 5], float)


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


def expected(opening: float, grow: float = GROW) -> float:
    """The white two eyes' marks should leave at an opening: the outlines' area less a pixel of their length, worked out
    here from the fixture's own geometry (five points on an arc over a level line), not by the tool's code."""
    x = np.array(fx.T) * fx.EYE_WIDE
    length = sum(np.linalg.norm(np.diff(np.stack([x, d * np.sin(np.pi * np.array(fx.T))], 1), axis=0), axis=1).sum() for d in (opening, fx.SAG))
    return float(max(2 * (fx.eye_area(opening) * grow ** 2 - length * grow * 1.0), 0.0))


def white(path: Path, n: int) -> np.ndarray:
    """[n, h, w] of bool: the mark's white in a written video, read here and not by the tool."""
    return np.stack(list(stream(str(path), (W, H), 0, n, pix="rgb24"))).min(-1) > dm.WHITE


MESH = video(WORK / "mesh.mkv", grey([90] * N))
POINTS = fx.save(WORK / "shot_points.npz", fx.face(OPEN, CENTRE), (W, H), first=0)


def marks_where_the_points_are() -> str:
    rec = dm.run(str(MESH), str(POINTS), 0, N, WORK / "a.mp4", "the check", grow=GROW, say=quiet)
    assert rec["passed"], rec["read_back"]
    got = white(WORK / "a.mp4", N)
    off = []
    for f in range(N):
        ys, xs = np.nonzero(got[f])
        assert len(xs), f"frame {f} has no white"
        off.append(len(xs) - expected(OPEN[f]))
        assert abs(off[-1]) < dm.THIN_PX2, (f, OPEN[f], len(xs), expected(OPEN[f]))  # the outline's area less a pixel of its length
        if OPEN[f] >= 3:                                                            # a thinner mark is a broken sliver with no centre to speak of
            assert abs(xs.mean() - CENTRE[0]) < 3, (f, xs.mean())                   # the two eyes sit either side of the centre
        assert CENTRE[1] - OPEN[f] * GROW - 3 < ys.mean() < CENTRE[1] + fx.SAG * GROW + 3, (f, ys.mean(), OPEN[f])
        assert xs.min() > CENTRE[0] - 60 and xs.max() < CENTRE[0] + 60
    order = np.argsort(OPEN)
    assert np.corrcoef(got.sum((1, 2))[order], OPEN[order])[0, 1] > 0.99
    return (f"white sits on the eyes on all {N} frames and is the outline's area less a pixel of its length to within "
            f"{max(abs(v) for v in off):.0f} square pixels (the tool's EDGE_PX is {dm.EDGE_PX:g})")


def frames_from_the_middle() -> str:
    mesh_first, points_first, first, count = 95, 100, 105, 12
    mesh = video(WORK / "mesh_mid.mkv", grey([40 + 3 * k for k in range(40)]))      # frame k of the file is source frame 95 + k
    openings = np.array([2 + (k * 7) % 13 for k in range(30)], float)              # row k of the points is source frame 100 + k
    pts = fx.save(WORK / "mid_points.npz", fx.face(openings, CENTRE), (W, H), first=points_first)
    rec = dm.run(str(mesh), str(pts), first, count, WORK / "mid.mp4", "the check", grow=GROW, mesh_first=mesh_first, say=quiet)
    assert rec["passed"], rec["read_back"]
    out = np.stack(list(stream(str(WORK / "mid.mp4"), (W, H), 0, count, pix="rgb24")))
    for j in range(count):
        level = int(np.median(out[j, :20, :20]))                                  # a corner, away from the marks
        assert abs(level - (40 + 3 * (first - mesh_first + j))) <= 2, (j, level)
        want = expected(openings[first - points_first + j])
        got = int((out[j].min(-1) > dm.WHITE).sum())
        assert abs(got - want) < dm.THIN_PX2, (j, got, want)
    areas = np.array([(out[j].min(-1) > dm.WHITE).sum() for j in range(count)])
    assert np.corrcoef(areas, openings[5:5 + count])[0, 1] > 0.99 and np.corrcoef(areas, openings[4:4 + count])[0, 1] < 0.9
    assert rec["frames"] == [105, 116] and rec["made_from"]["its_first_frame_is_source_frame"] == 95 and rec["made_from"]["their_first_row_is_source_frame"] == 100
    return "frames 105 to 116 of a body-model video from 95 and points from 100: each written frame is its own mesh frame with its own opening"


def bare_frames() -> str:
    rec = dm.run(str(MESH), str(POINTS), 0, N, WORK / "bare.mp4", "the check", grow=GROW, bare=[6, 7, 30], bare_by="a person", say=quiet)
    got = white(WORK / "bare.mp4", N).sum((1, 2))
    assert rec["passed"] and got[6] == 0 and got[7] == 0 and got[5] > 0 and got[8] > 0, got[4:10]
    assert rec["frames_left_bare"] == [6, 7] and rec["left_bare_by"] == "a person", rec           # 30 is not in the shot and is not listed
    saved = json.loads((WORK / "bare.json").read_text())
    assert saved["frames_left_bare"] == [6, 7] and any("turned away" in line for line in saved["not_known"])
    try:
        dm.run(str(MESH), str(POINTS), 0, N, WORK / "x.mp4", "the check", grow=GROW, bare=[6], say=quiet)
    except SystemExit as exc:
        assert "--bare-by" in str(exc), exc
    else:
        raise AssertionError("frames left bare with nobody named were accepted")
    return "two frames left bare by a named person carry no white and are in the record; a list with no name is refused"


def the_read_back_can_fail() -> str:
    mesh = np.stack(list(stream(str(MESH), (W, H), 0, N, pix="rgb24")))
    idx = {n: i for i, n in enumerate(fx.NAMES)}
    rings = [dm.fp.lid_rings(p, idx) for p in fx.face(OPEN, CENTRE)]
    marked = dm.draw(mesh, rings, GROW, set())
    drawn = np.ones(N, bool)
    area = np.array([sum(dm.fp.ring_area(r) for r in rings[f]) * GROW ** 2 for f in range(N)])
    dm.write_video(WORK / "good.mp4", marked)
    good = dm.read_back(WORK / "good.mp4", mesh, area, drawn)
    assert good["passed"], good["lags"]
    dm.write_video(WORK / "late.mp4", np.concatenate([marked[:1], marked[:-1]]))    # every frame one late, the first repeated
    late = dm.read_back(WORK / "late.mp4", mesh, area, drawn)
    assert not late["passed"] and late["lags"][1] > late["lags"][0], late["lags"]
    drawn2 = drawn.copy(); drawn2[9] = False                                        # frame 9 was to be bare and is not
    area2 = np.where(drawn2, area, 0.0)
    painted = dm.read_back(WORK / "good.mp4", mesh, area2, drawn2)
    assert not painted["passed"] and painted["on_bare"] > 0, painted["on_bare"]
    blank = marked.copy(); blank[12] = mesh[12]                                     # a marked frame written with no mark
    dm.write_video(WORK / "blank.mp4", blank)
    lost = dm.read_back(WORK / "blank.mp4", mesh, area, drawn)
    assert not lost["passed"] and lost["marked_frames_with_no_white"] == [12], lost["marked_frames_with_no_white"]
    # a body model with a highlight just under white, which the written video holds just over it (2026-10-12)
    shiny = mesh.copy(); shiny[:, 150:170, 10:60] = dm.WHITE - 7
    drawn3 = drawn.copy(); drawn3[5] = False                                        # frame 5 is left bare
    lifted = dm.draw(shiny, rings, GROW, {5}); lifted[:, 150:170, 10:60] = dm.WHITE + 6
    written = video(WORK / "lifted.mkv", lifted)
    counted = dm.white_in(written, shiny)
    assert counted[5] == 0 and (counted[drawn3] > 0).all(), f"white counted on the bare frame: {counted[5]}"
    assert abs(counted[4] - dm.white_in(WORK / "good.mp4", mesh)[4]) < dm.THIN_PX2, "the highlight was counted with the marks"
    shone = dm.read_back(written, shiny, np.where(drawn3, area, 0.0), drawn3)
    assert shone["passed"] and shone["on_bare"] == 0, (shone["on_bare"], shone["lags"])
    flat = lifted.copy(); flat[:, 150:170, 10:60] = 255                             # the control: on a mesh that was dark there, it IS white
    video(WORK / "flat.mkv", flat)
    assert dm.white_in(WORK / "flat.mkv", mesh)[5] > 0, "control failed: white on a bare frame of a dark mesh was not counted"
    return (f"as drawn it passes ({good['lags'][0]:.3f} against {max(good['lags'][-1], good['lags'][1]):.3f} a frame off); every frame one late fails; "
            "a mark on a bare frame fails; a marked frame with none fails; a highlight the encode lifts over white is not a mark, and a bare frame with one passes")


def verify_catches_the_wrong_clock() -> str:
    dm.run(str(MESH), str(POINTS), 0, N, WORK / "v.mp4", "the check", grow=GROW, say=quiet)
    assert dm.verify(str(WORK / "v.mp4"), str(MESH), str(POINTS), 0, N, say=quiet)
    # a reader who takes the points' first row to be source frame 1, or -1, reads every mark against a neighbour's points
    early = dm.verify(str(WORK / "v.mp4"), str(MESH), str(POINTS), 1, N - 2, points_first=1, video_first=0, mesh_first=0, say=quiet)
    late = dm.verify(str(WORK / "v.mp4"), str(MESH), str(POINTS), 1, N - 2, points_first=-1, video_first=0, mesh_first=0, say=quiet)
    assert not early and not late, (early, late)
    assert not dm.verify(str(MESH), str(MESH), str(POINTS), 0, N, say=quiet)       # a video with no marks at all is not a pass
    return "on the right clock it passes; the points read a frame early or a frame late fail; a video with no marks fails"


def moving(n: int = N):
    """A source and a body model that move together by steps that never repeat: (source frames, mesh frames)."""
    step = np.array([0, 3, 9, 2, 14, 6, 1, 11, 4, 16, 7, 2, 13, 5, 18, 3, 10, 6, 1, 15, 8, 4, 12, 2, 9, 5][:n])
    x = 40 + np.cumsum(step) % 150
    rng = np.random.default_rng(7)
    ground = rng.integers(60, 120, (H, W, 3), dtype=np.uint8)
    src, mesh = [], []
    for k in range(n):
        s = ground.copy(); s[60:130, x[k]:x[k] + 50] = 200
        m = np.zeros((H, W, 3), np.uint8); m[60:130, x[k]:x[k] + 50] = 120
        src.append(s); mesh.append(m)
    return np.stack(src), np.stack(mesh)


def off_the_sources_clock() -> str:
    src, mesh = moving()
    source = video(WORK / "source.mkv", src)
    pts = fx.save(WORK / "mv_points.npz", fx.face(OPEN, CENTRE), (W, H), first=0)
    on = dm.run(str(video(WORK / "mesh_on.mkv", mesh)), str(pts), 0, N, WORK / "on.mp4", "the check", grow=GROW, source=str(source), say=quiet)
    clock = on["body_model_against_the_source"]
    assert on["passed"] and clock["best_neighbour_less_no_slip"] < 0, clock
    # the figures say what they were read on and what they cannot see, and decide nothing
    assert clock["measured_on"]["frames"] == [0, N - 1] and len(clock["measured_on"]["box"]) == 4 and "a second person" in clock["cannot_see"], clock
    assert "no part of `passed`" in clock["decides"], clock
    assert (WORK / "on_sheet.jpg").exists() and on["sheet"]["looked_at_by"] == ""
    said = []
    for name, frames in (("late", np.concatenate([mesh[:1], mesh[:-1]])), ("early", np.concatenate([mesh[1:], mesh[-1:]]))):
        rec = dm.run(str(video(WORK / f"mesh_{name}.mkv", frames)), str(pts), 0, N, WORK / f"{name}.mp4", "the check", grow=GROW, source=str(source), say=said.append)
        # a body model a frame off the source: the figures show it, and the run is NOT failed on them (a figure of degree is data)
        assert rec["passed"] and rec["read_back"]["passed"], (name, rec["body_model_against_the_source"], rec["read_back"])
        assert rec["body_model_against_the_source"]["best_neighbour_less_no_slip"] > 0, (name, rec["body_model_against_the_source"])
        assert "a_neighbour_fits_better" not in rec["body_model_against_the_source"] and "says" not in rec["body_model_against_the_source"]
    lines = [line for line in said if line.startswith("the body model against the source")]
    assert len(lines) == 2 and all("no part of PASSED" in line and "the best neighbour less no slip 1." in line and "anything else that moves counts too" in line
                                   for line in lines), said
    assert not any(line.startswith("FAILED") for line in said), said
    none = dm.run(str(WORK / "mesh_on.mkv"), str(pts), 0, N, WORK / "nosrc.mp4", "the check", grow=GROW, say=quiet)
    assert none["passed"] and none["body_model_against_the_source"] == "not looked at: no --source"
    part = video(WORK / "source_from_4.mkv", src[4:])                               # a source cut from the original: its frame 0 is source frame 4
    cut = dm.run(str(WORK / "mesh_on.mkv"), str(pts), 4, N - 4, WORK / "cut.mp4", "the check", grow=GROW, mesh_first=0, source=str(part), source_first=4, say=quiet)
    assert cut["passed"] and cut["body_model_against_the_source"]["best_neighbour_less_no_slip"] < 0, cut["body_model_against_the_source"]
    assert cut["body_model_against_the_source"]["measured_on"]["frames"] == [4, N - 1]
    try:
        dm.run(str(WORK / "mesh_on.mkv"), str(pts), 4, N - 4, WORK / "cut2.mp4", "the check", grow=GROW, mesh_first=0, source=str(part), say=quiet)
    except SystemExit as exc:
        assert "--source-first" in str(exc), exc
    else:
        raise AssertionError("a cut source read as if it began on source frame 0 was accepted")
    return ("a body model that moves with the source passes; the same one a frame late or a frame early fails the run while its marks read back; "
            "with no source the look is said not to be made; a source cut from the original is read from the frame it is said to begin on")


def the_size_rule() -> str:
    rec = dm.run(str(MESH), str(POINTS), 0, N, WORK / "rule.mp4", "the check", say=quiet)
    want = h3_config.LID_MARKS["face_px"] / 60.0
    assert abs(rec["grown"] - want) < 0.01 and "face_px" in rec["grown_by"], (rec["grown"], want)
    line = h3_config.POSE_READING["lid_gap_px_to_draw"]
    assert rec["lid_gap_px_to_draw"] == line and rec["lid_gap_median_is"] == "over that line" and "starting value" in rec["that_line_is"], rec["lid_gap_px_median"]
    small = fx.save(WORK / "small_points.npz", fx.face(np.full(N, 0.3) + OPEN / 40, CENTRE), (W, H), first=0)
    low = dm.run(str(MESH), str(small), 0, N, WORK / "small.mp4", "the check", grow=GROW, say=quiet)
    assert low["lid_gap_px_median"] < line and low["lid_gap_median_is"] == "under that line", low["lid_gap_px_median"]
    assert white(WORK / "small.mp4", N).sum() > 0                                  # said, and still drawn
    return f"with no factor given it grows by face_px over the face's height ({want:.2f}); a gap under the line is said and the marks are still drawn"


def refusals() -> str:
    old = fx.save(WORK / "old_points.npz", fx.face(OPEN, CENTRE), (W, H), first=0, names=False)
    for what, kwargs, word in (
            ("no names", dict(points=old), "holds no point names"),
            ("another canvas", dict(points=fx.save(WORK / "big_points.npz", fx.face(OPEN, CENTRE), (W * 2, H * 2), first=0)), "not on one canvas"),
            ("frames the points do not hold", dict(first=10, count=N), "were asked for"),
            ("frames the body model does not hold", dict(mesh_first=5), "were asked for"),
            ("no first frame said", dict(points=fx.save(WORK / "nofirst_points.npz", fx.face(OPEN, CENTRE), (W, H))), "--points-first")):
        a = dict(points=POINTS, first=0, count=N, mesh_first=None); a.update(kwargs)
        try:
            dm.run(str(MESH), str(a["points"]), a["first"], a["count"], WORK / "r.mp4", "the check", grow=GROW, mesh_first=a["mesh_first"], say=quiet)
        except (SystemExit, ValueError) as exc:
            assert word in str(exc), (what, exc)
        else:
            raise AssertionError(f"{what}: accepted")
    ok = dm.run(str(MESH), str(old), 0, N, WORK / "named.mp4", "the check", grow=GROW, names=fx.NAMES, say=quiet)
    assert ok["passed"]
    return "no names, another canvas, frames either file lacks and a file that does not say its first frame are each refused; an old file reads with its names given"


case("the marks are where the points are and as large as they say", marks_where_the_points_are)
case("frames from the middle of both files", frames_from_the_middle)
case("frames left bare come from a person and are bare", bare_frames)
case("the read-back can fail", the_read_back_can_fail)
case("verify catches points read on the wrong clock", verify_catches_the_wrong_clock)
case("a body-model video off the source's clock: the figures are printed and recorded, and decide nothing", off_the_sources_clock)
case("the size rule, and a small lid gap is said", the_size_rule)
case("what it refuses", refusals)
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(finish())
