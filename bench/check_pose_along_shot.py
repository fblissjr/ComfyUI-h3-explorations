#!/usr/bin/env python3
"""`bench/pose_along_shot.py` with the model replaced by a function whose answer is known: everything round the model.

The model needs the card and its answer cannot be known beforehand, so no check can hold it. What a check can
hold is what goes wrong round it without a sound: the wrong frames read, the wrong box given, a points file
whose first row is not the frame it says, names that are not the model's. So the model here is a function
that writes what it was GIVEN into the points it returns (the picture's level, its left edge, the box), and
the files the tool wrote are read back.

1. **The frames asked for are the frames read, by number.** In a clip whose frame k is a flat level of 10 k,
   eight frames from frame 5 come back as levels 50 to 120 in order, and the file says its first row is source
   frame 5; read as a render whose frame 5 is source frame 300, it says 300.
2. **The box is the tracked person's on that frame.** With a mask video that starts on another source frame
   and holds a rectangle that moves, each frame's box is that frame's rectangle; a frame with no mask keeps
   the box before it and is listed; with no mask video the box is the whole frame.
3. **A clip the canvas's size is read as it is; one of another shape is fitted.** A wider clip with bright
   bands at its sides comes through `bench/_lib/frames.FIT` with the bands cut away, and the record says it was
   fitted. This does not judge FIT, which has one place and its own check.
4. **Where the names come from.** Given by hand they are used as given; a folder whose config names its
   points is read; one whose labels are placeholders gets Meta's tracked list (`vendor/meta_sapiens2/`) when it has
   that many points and its own left-right pairs are the list's, and is refused when one pair differs, when it
   has no pairs, or when it is another model; a count that does not match the model's points is refused.
5. **`read` says what the points can carry.** On a face whose openings are chosen here it prints the face's
   height and the lid gap in pixels, and with `--table` writes one row a frame whose eye figure is the one
   worked out by hand, with how many frames sit under, over and between the eye opening's two starting lines;
   a face a few pixels tall is said to sit under the line.
6. **The record.** Who made it and when, the frames and which source frames they are, the box's source.

Needs ffmpeg and ffprobe on the path. No model, no card, no server.

    <comfy venv python> bench/check_pose_along_shot.py
"""
from __future__ import annotations

import csv
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

from _lib import face_points as fp  # noqa: E402
from _lib import face_points_fixture as fx  # noqa: E402
from _lib.frames import FIT as fr_fit  # noqa: E402
from _lib.frames import write_mask_video  # noqa: E402

import pose_along_shot as pa  # noqa: E402

sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402

W, H = 64, 48
WORK = Path(tempfile.mkdtemp(prefix="check_pose_along_shot_"))
NAMES = ["what_it_saw", "its_box_corner", "its_box_size"]
quiet = lambda *_: None  # noqa: E731


def model(picture: np.ndarray, box: list[float]):
    """A stand-in for the pose model: three 'points' that are what it was handed."""
    assert picture.shape == (H, W, 3), picture.shape
    return (np.array([[picture.mean(), picture[:, :2].mean()], [box[0], box[1]], [box[2], box[3]]], np.float32), np.full(3, 0.9, np.float32))


def clip(path: Path, frames: np.ndarray) -> str:
    """[n, h, w] of grey levels as a lossless clip."""
    h, w = frames.shape[1:]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                            "-c:v", "ffv1", "-level", "3", str(path)], stdin=subprocess.PIPE)
    enc.communicate(frames.astype(np.uint8).tobytes())
    assert enc.returncode == 0, path
    return str(path)


LEVELS = clip(WORK / "levels.mkv", np.stack([np.full((H, W), 10 * k, np.uint8) for k in range(20)]))


def frames_by_number() -> str:
    rec = pa.run(LEVELS, 5, 8, (W, H), WORK / "a_points.npz", "the check", model, NAMES, say=quiet)
    z = np.load(WORK / "a_points.npz")
    got = z["keypoints"][:, 0, 0]
    assert np.abs(got - 10 * np.arange(5, 13)).max() < 1.5, got
    assert int(z["first_frame"]) == 5 and list(z["names"]) == NAMES and z["size"].tolist() == [W, H] and z["scores"].shape == (8, 3)
    assert rec["made_from"]["its_frames"] == [5, 12] and rec["made_from"]["which_are_source_frames"] == [5, 12]
    pa.run(LEVELS, 5, 8, (W, H), WORK / "b_points.npz", "the check", model, NAMES, source_first=300, say=quiet)
    assert int(np.load(WORK / "b_points.npz")["first_frame"]) == 300
    for first, count in ((15, 8), (-1, 4)):
        try:
            pa.run(LEVELS, first, count, (W, H), WORK / "x_points.npz", "the check", model, NAMES, say=quiet)
        except SystemExit as exc:
            assert "were asked for" in str(exc), exc
        else:
            raise AssertionError(f"frames {first} on of a 20-frame clip were accepted")
    return "frames 5 to 12 of a clip come back as levels 50 to 120 in order; row 0 is source frame 5, or 300 when the clip is a render of it; frames the clip lacks are refused"


def the_box_is_that_frames() -> str:
    mask = np.zeros((12, H, W), bool)                      # the mask video's frame 0 is source frame 100
    for k in range(12):
        if k not in (6, 7):                                # the tracker lost the person on source frames 106 and 107
            mask[k, 10:20, 4 + 3 * k:4 + 3 * k + 12] = True
    write_mask_video(WORK / "track.mkv", mask)
    rec = pa.run(LEVELS, 3, 8, (W, H), WORK / "box_points.npz", "the check", model, NAMES, source_first=103, box_mask=str(WORK / "track.mkv"),
                 box_mask_first=100, say=quiet)
    z = np.load(WORK / "box_points.npz")["keypoints"]
    for j in range(8):
        k = 3 + j                                          # the mask frame of source frame 103 + j
        want = [4 + 3 * (k if k not in (6, 7) else 5), 10, 12, 10]
        assert [int(round(v)) for v in (*z[j, 1], *z[j, 2])] == want, (j, z[j, 1:], want)
    assert rec["frames_with_no_mask_that_kept_the_box_before"] == [106, 107], rec["frames_with_no_mask_that_kept_the_box_before"]
    assert rec["made_from"]["box"]["its_first_frame_is_source_frame"] == 100
    whole = pa.run(LEVELS, 3, 4, (W, H), WORK / "whole_points.npz", "the check", model, NAMES, say=quiet)
    zw = np.load(WORK / "whole_points.npz")["keypoints"]
    assert zw[:, 1].tolist() == [[0, 0]] * 4 and zw[:, 2].tolist() == [[W, H]] * 4 and whole["made_from"]["box"] == "the whole frame"
    return "each frame's box is its own rectangle from a mask video that starts three frames earlier; two frames with no mask keep the box before and are listed; with no mask the box is the whole frame"


def fitted_or_not() -> str:
    wide = np.full((6, H, W + 16), 100, np.uint8)
    wide[:, :, :6] = 250; wide[:, :, -6:] = 250           # bright bands at the two sides, outside the canvas's shape
    src = clip(WORK / "wide.mkv", wide)
    rec = pa.run(src, 0, 6, (W, H), WORK / "wide_points.npz", "the check", model, NAMES, say=quiet)
    z = np.load(WORK / "wide_points.npz")["keypoints"]
    assert np.abs(z[:, 0, 0] - 100).max() < 2 and np.abs(z[:, 0, 1] - 100).max() < 2, z[:, 0]     # the bands are cut away
    assert rec["made_from"]["fitted_to_the_canvas"].startswith("as the loader fits a source: cut")
    assert rec["made_from"]["the_filter_it_was_read_through"] == fr_fit.format(w=W, h=H), rec["made_from"]          # the filter itself, not only words for it
    same = pa.run(LEVELS, 0, 4, (W, H), WORK / "same_points.npz", "the check", model, NAMES, say=quiet)
    assert same["made_from"]["fitted_to_the_canvas"].startswith("no") and same["made_from"]["the_filter_it_was_read_through"] == ""
    return "a clip wider than the canvas comes through the reader's fit with its side bands cut away and is recorded as fitted; a clip the canvas's size is not"


def names() -> str:
    def folder(labels: dict | None, extra: list[str] | None = None, pairs: list | None = None) -> str:
        d = Path(tempfile.mkdtemp(dir=WORK))
        (d / "config.json").write_text(json.dumps(({"id2label": labels} if labels is not None else {}) | ({"flip_pairs": pairs} if pairs is not None else {})))
        if extra is not None:
            (d / "names.json").write_text(json.dumps(extra))
        return str(d)
    assert pa.point_names(folder({"0": "nose", "1": "left_eye", "2": "chin"})) == ["nose", "left_eye", "chin"]
    assert pa.point_names(folder({"10": "k", "2": "c", "0": "a"})) == ["a", "c", "k"]              # in the labels' own order, not as text
    assert pa.point_names(folder({"0": "LABEL_0", "1": "LABEL_1"}, extra=["x", "y"])) == ["x", "y"]
    for labels in ({"0": "LABEL_0", "1": "LABEL_1"}, None):
        try:
            pa.point_names(folder(labels))
        except SystemExit as exc:
            assert "--names" in str(exc), exc
        else:
            raise AssertionError("a folder that does not name its points was accepted")
    # Meta's tracked list: for a folder with placeholders, that many points and the list's own left-right pairs
    table = json.loads(pa.TRACKED_NAMES.read_text())
    listed = table["names"]
    at = {n: i for i, n in enumerate(listed)}
    pairs = sorted({tuple(sorted((at[a], at[b]))) for a, b in table["swap"].items()})
    blank = {str(i): f"LABEL_{i}" for i in range(len(listed))}
    got, source = pa.names_and_source(folder(blank, pairs=[list(p) for p in pairs]))
    assert got == listed and "vendor/meta_sapiens2/pose_keypoints308.json" in source, source
    assert pa.names_and_source(folder({"0": "nose", "1": "left_eye"}))[1] == "the model's own config"
    (a, b), (c, d) = pairs[0], pairs[1]
    for why, its_labels, its_pairs in (("one pair that is not the list's", blank, [[a, c], [b, d]] + [list(p) for p in pairs[2:]]),
                                       ("no pairs at all", blank, None),
                                       ("one point more than the list", blank | {str(len(listed)): "LABEL_x"}, [list(p) for p in pairs])):
        try:
            pa.point_names(folder(its_labels, pairs=its_pairs))
        except SystemExit as exc:
            assert "--names" in str(exc) and "vendor/meta_sapiens2/pose_keypoints308.json" in str(exc), exc
        else:
            raise AssertionError(f"a folder with {why} was given Meta's list")
    try:
        pa.run(LEVELS, 0, 4, (W, H), WORK / "n_points.npz", "the check", model, NAMES + ["one_too_many"], say=quiet)
    except SystemExit as exc:
        assert "names" in str(exc), exc
    else:
        raise AssertionError("four names for three points were accepted")
    return (f"a config that names its points is read in the labels' order; placeholders get Meta's {len(listed)} tracked names when the folder's own "
            f"{len(pairs)} left-right pairs are the list's, and are refused with one pair moved, with no pairs, with a point more, or with nothing at all; "
            "a count that does not match the model's points is refused")


def read_says_what_they_carry() -> str:
    opening = np.array([2, 4, 8, 12, 6, 3], float)
    f = fx.save(WORK / "face_points.npz", fx.face(opening, (160, 100)), (320, 240), first=40)
    lines: list[str] = []
    said = pa.read(f, table=True, say=lines.append)
    want = np.sin(np.pi * np.array(fx.T)).sum() / len(fx.T) * (opening + fx.SAG)                     # the lid gap in pixels, by hand
    assert abs(said["lid_gap_px"]["median"] - float(np.median(want))) < 0.01 and abs(said["face_height_px"]["median"] - 60.0) < 0.1, said
    assert said["first_source_frame"] == 40 and said["median_under_it"] is False and said["frames_under_it"] == int((want < h3_config.POSE_READING["lid_gap_px_to_draw"]).sum())
    eyes_by_hand = want / 60.0
    band = h3_config.POSE_READING
    against = said["eyes_against_the_starting_lines"]
    assert against["under_the_shut_line"] == int((eyes_by_hand < band["eyes_shut_under"]).sum()) and against["over_the_open_line"] == int((eyes_by_hand > band["eyes_open_over"]).sum()) == 4
    assert against["between_the_two_lines"] == 2 and "starting value" in against["which_are"] and "one close-up" in against["which_are"], against
    text = "\n".join(lines)
    assert "4 over the open line" in text and "2 between them." in text and "a starting value and not this clip's" in text, text
    assert not any(w in text.lower() for w in ("not readable", "too close", "good", "bad", "passed", "failed")), text      # it measures; the words are the reader's
    assert "source frames 40 to 45" in text and "does not say lids were seen" in text and "not a measure of a mouth" in text
    rows = list(csv.DictReader(open(WORK / "face.csv")))
    assert list(rows[0]) == list(fp.COLUMNS) and [int(r["source_frame"]) for r in rows] == list(range(40, 46))
    assert max(abs(float(r["eyes"]) - w / 60.0) for r, w in zip(rows, want)) < 1e-4
    small = fx.save(WORK / "small_points.npz", fx.face(opening / 20, (160, 100), face_px=14.0), (320, 240), first=40)
    lines2: list[str] = []
    s2 = pa.read(small, say=lines2.append)
    assert s2["median_under_it"] is True and any("the shot's median is under it" in line for line in lines2), s2
    assert any("the shot's median is over it" in line for line in lines) and "starting value" in said["that_line_is"]
    assert "table" not in s2 and not (WORK / "small.csv").exists()
    return f"a face 60 px tall with lash lines {said['lid_gap_px']['median']:.2f} px apart is read as that and its table's eye figure is the hand-worked one; a face 14 px tall sits under the line"


def the_record() -> str:
    said: list[str] = []
    rec = pa.run(LEVELS, 2, 5, (W, H), WORK / "rec_points.npz", "a session", model, NAMES, source_first=50, names_from="a list of the check's", say=said.append)
    assert any(line.startswith("wrote ") for line in said) and any("no summary of the face" in line for line in said), said   # written, then said plainly
    saved = json.loads((WORK / "rec_points.json").read_text())
    assert saved == rec and saved["made_by"] == "a session" and saved["made_at"].endswith("Z") and saved["model"] == h3_config.POSE_MODEL
    assert saved["made_from"]["its_frames"] == [2, 6] and saved["made_from"]["which_are_source_frames"] == [50, 54] and saved["frames"] == 5
    assert len(saved["not_known"]) == 3 and saved["names_from"] == "a list of the check's"
    assert pa.run(LEVELS, 2, 5, (W, H), WORK / "rec2_points.npz", "a session", model, NAMES, say=quiet)["names_from"] == "given by the caller"
    return ("the record says who made it and when, where the points' names came from, the clip's frames and the source frames they are, and what the tool cannot know; "
            "points that name no face are written all the same and the tool says it has no summary of them")


def held_to_the_track() -> str:
    """A close group, as on the lead's 3167 (2026-10-14): another, brighter face inside the subject's box and off
    their track. The stand-in model finds the brightest blob in its box and puts a face there."""
    cw, ch, n = 320, 240, 6
    frames = np.full((n, ch, cw), 60, np.uint8)
    frames[:, 80:100, 55:75] = 200                          # the subject's face, on their track
    frames[:, 80:100, 210:230] = 255                        # another's face: inside the box, off the track
    src = clip(WORK / "group.mkv", frames)
    mask = np.zeros((n, ch, cw), bool)
    mask[:, 20:230, 20:110] = True                          # the subject: a column, and the floor they stand on, so the box holds the corner
    mask[:, 180:230, 20:300] = True
    mask[3] = False                                         # the tracker lost them on one frame
    write_mask_video(WORK / "group_track.mkv", mask)
    seen = []

    def brightest(picture: np.ndarray, box: list[float]):
        x0, y0, w, h = (int(v) for v in box)
        part = picture[y0:y0 + h, x0:x0 + w].mean(2)
        ys, xs = np.nonzero(part >= part.max() - 5)
        centre = (x0 + xs.mean(), y0 + ys.mean() + 10)
        seen.append(centre)
        return fx.face([6.0], centre)[0].astype(np.float32), np.full(len(fx.NAMES), 0.9, np.float32)
    said: list[str] = []
    rec = pa.run(src, 0, n, (cw, ch), WORK / "group_points.npz", "the check", brightest, fx.NAMES, box_mask=str(WORK / "group_track.mkv"),
                 box_mask_first=0, say=said.append)
    assert all(abs(x - 65) < 2 for x, _ in seen), seen
    on = rec["face_on_the_mask"]
    assert on["frames_whose_face_middle_is_off_the_mask"] == [] and on["share_a_frame"] == [1.0] * n and all(on["middle_on_a_frame"]), on
    assert rec["held_to_the_mask"]["frames_held_to_an_earlier_frames_mask"] == [3] and str(pa.HOLD_GROW_PX) in rec["held_to_the_mask"]["how"], rec["held_to_the_mask"]
    assert any(f"on all {n} frames" in line for line in said), said
    # the same with the hold taken out: the brighter face is read on every frame, and every frame is listed as off the track
    real = pa.held_masks
    pa.held_masks = lambda m, c: (np.ones_like(m), real(m, c)[1], real(m, c)[2])
    try:
        seen.clear()
        said.clear()
        rec = pa.run(src, 0, n, (cw, ch), WORK / "group_unheld_points.npz", "the check", brightest, fx.NAMES, box_mask=str(WORK / "group_track.mkv"),
                     box_mask_first=0, say=said.append)
    finally:
        pa.held_masks = real
    assert all(abs(x - 220) < 2 for x, _ in seen), seen
    assert rec["face_on_the_mask"]["frames_whose_face_middle_is_off_the_mask"] == list(range(n)) and rec["face_on_the_mask"]["share_a_frame"] == [0.0] * n
    assert any(line.startswith(f"THE FACE'S MIDDLE IS OFF THE TRACKED MASK on {n} of {n} frames: 0-{n - 1}") for line in said), said
    assert any("0:0.00*" in line for line in said), said
    return ("another, brighter face inside the box and off the track is greyed and the subject's face is read on every frame, the lost frame held to the mask "
            "before it; with the hold taken out that face is read and every frame is listed and printed as off the track")


case("the frames asked for are the frames read", frames_by_number)
case("held to the track: another face in the box is not read", held_to_the_track)
case("the box is the tracked person's on that frame", the_box_is_that_frames)
case("a clip of another shape is fitted, one the canvas's size is not", fitted_or_not)
case("where the names come from", names)
case("read says what the points can carry", read_says_what_they_carry)
case("the record", the_record)
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(finish())
