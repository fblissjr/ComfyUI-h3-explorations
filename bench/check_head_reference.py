#!/usr/bin/env python3
"""`bench/head_reference.py` against a face whose every line is known by construction, read from the files it wrote.

The one command runs three steps, each with the others' outputs for arguments, and writes one record. What can
go wrong in the joining, and nowhere else: the drawing made from points that were not carried into the cut
picture; the source cut by another box than the body model; a step's failure lost; an old file written over.
The steps themselves are held by their own checks.

1. **One run makes the form.** Every file it names exists; the reference is the cut body model, frame for frame,
   with white only on the face's lines as they land through the record's box and scale, none on the lips and
   none at the points that belong to no line; the source is cut by the same box; the record names every input
   by path, the commit and each tool's last commit, what was drawn and the three read-backs, and it passed.
2. **Asked for frames from the middle of all three files, it makes those frames.** The body-model video, the
   points and the source each start on another source frame.
3. **It writes over nothing, and nothing when it cannot finish.** An output that exists, an `--out` that is not
   an .mp4, an input that is not a file and a span the points do not cover are each refused with no file made.
4. **The form's defaults can be set aside, and the record says so.** With dots and nothing left out there is
   white on the lips and at the points drawn alone, and a width given by hand is recorded as one.
5. **A step that fails fails the run.** With the drawing step's read-back made to fail, the run does not pass
   and the record says which step.

Needs ffmpeg and ffprobe on the path. No model, no card, no server.

    <comfy venv python> bench/check_head_reference.py
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

from _lib import face_lines_fixture as fl  # noqa: E402
from _lib.frames import stream  # noqa: E402

import head_reference as hr  # noqa: E402

W, H, N = 384, 288, 12
ROOM = 0.4                                      # the fixture's face is large on this canvas: with the tool's own room the box is the whole frame
WORK = Path(tempfile.mkdtemp(prefix="check_head_reference_"))
quiet = lambda *_: None  # noqa: E731
STEP = np.array([[0, 0], [5, 2], [-3, 6], [8, -4], [-6, -5], [4, 7], [9, 1], [-7, 3], [3, -8], [6, 5], [-9, -2], [5, -6]], float)
CENTRES = np.array([190.0, 120.0]) + np.cumsum(STEP, 0) * 0.5


def video(path: Path, frames: np.ndarray) -> Path:
    """[n, h, w, 3] as a lossless video."""
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{frames.shape[2]}x{frames.shape[1]}",
                            "-r", "24", "-i", "-", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", str(path)], stdin=subprocess.PIPE)
    enc.communicate(frames.tobytes())
    assert enc.returncode == 0, path
    return path


def grey(levels, size=(W, H)) -> np.ndarray:
    """Frame k is a flat grey of `levels[k]`, so a frame says which one it is."""
    return np.stack([np.full((size[1], size[0], 3), v, np.uint8) for v in levels])


def read(path: Path, n: int) -> np.ndarray:
    return np.stack(list(stream(str(path), (W, H), 0, n, pix="rgb24")))


def landed(centre, box, scale, without=("lip",)) -> np.ndarray:
    """Points every quarter pixel along the face's lines as built here, carried through a box and a scale."""
    out = []
    for name, (pts, closed) in fl.truth(centre).items():
        if any(w in name for w in without) or "pupil" in name:
            continue
        run = np.vstack([pts, pts[:1]]) if closed else pts
        for a, b in zip(run[:-1], run[1:]):
            out.append(a + (b - a) * np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 0.25) + 1))[:, None])
    return (np.concatenate(out) - box[:2]) * scale


def rooms(name: str, mesh_levels, source_levels, centres, points_first: int) -> tuple[Path, Path, Path, Path]:
    """(the folder, the plain body model, the points file, the source) for one case."""
    folder = WORK / name
    folder.mkdir()
    scores = np.full((len(centres), len(fl.NAMES)), 0.9)
    scores[:, [i for n, i in fl.INDEX.items() if "pupil" in n]] = 0.2             # as on a real small face: the pupils are not scored
    pts = fl.save(folder / "shot_points.npz", fl.face(centres), (W, H), first=points_first, scores=scores)
    return (folder, video(folder / "mesh.mkv", grey(mesh_levels)), pts,
            video(folder / "source.mkv", grey(source_levels, size=(W * 2 + 64, H * 2))))


def one_run_makes_the_form() -> str:
    folder, mesh, pts, source = rooms("one", [90] * N, [40 + 4 * k for k in range(N)], CENTRES, 0)
    out = folder / "made" / "head.mp4"
    said = []
    rec = hr.run(str(mesh), str(pts), str(source), 0, N, out, "the check", room=ROOM, say=said.append)
    made = hr.outputs(out)
    assert all(p.is_file() for k, p in made.items() if k != "filled_points") and not made["filled_points"].exists(), \
        [k for k, p in made.items() if p.is_file() != (k != "filled_points")]          # nothing was filled: no filled copy
    assert rec["passed"] and json.loads(made["record"].read_text()) == rec and any(line.startswith("PASSED") for line in said)
    # the body model against the source is printed on a line of its own, as figures: it is no step's PASSED or FAILED
    assert any(line.strip().startswith("the body model against the source") and "no part of PASSED" in line for line in said), said
    assert "decides" in rec["body_model_against_the_source"] and "a_neighbour_fits_better" not in rec["body_model_against_the_source"]
    box, scale = np.array(rec["box"], float), np.array(rec["scale"])
    assert scale[0] > 1.05 and json.loads(made["source_cut_record"].read_text())["box"] == rec["box"] == json.loads(made["plain_cut_record"].read_text())["box"]
    got, plain = read(out, N), read(made["plain_cut"], N)
    worst = 0.0
    for f in range(N):
        ys, xs = np.nonzero(got[f].min(-1) > hr.faces.WHITE)
        assert len(xs), f"frame {f} has no white"
        xy = np.stack([xs, ys], 1).astype(float)
        d = np.linalg.norm(xy[:, None, :] - landed(CENTRES[f], box, scale)[None], axis=-1).min(1)
        worst = max(worst, d.max())
        assert d.max() <= hr.FORM_WIDTH + 1.0, (f, d.max())            # all of it on the lines as carried; none on the lips, none alone
        clear = np.ones((H, W), bool)                                  # well away from every line: there the reference IS the cut body model
        x0, y0, x1, y1 = (int(v) for v in (xs.min() - 8, ys.min() - 8, xs.max() + 9, ys.max() + 9))
        clear[max(y0, 0):y1, max(x0, 0):x1] = False
        assert clear.sum() > 1000 and np.abs(got[f][clear].astype(int) - plain[f][clear].astype(int)).mean() < 2.0, \
            f"frame {f} is not the cut body model away from the lines"
    under = read(made["source_cut"], N)
    assert all(abs(int(np.median(under[k])) - (40 + 4 * k)) <= 2 for k in range(N)), "the cut source's frames are not the source's, in order"
    assert rec["inputs"]["body_model_video"] == str(mesh) and rec["inputs"]["points"] == str(pts) and rec["inputs"]["source"] == str(source)
    assert "fitted to the canvas" in rec["inputs"]["the_source_was"] and rec["its_first_frame_is_source_frame"] == 0 and rec["frames"] == [0, N - 1]
    assert rec["drawn"] == {**rec["drawn"], "a_dot_at_every_point": False, "points_drawn_alone": [], "parts_left_out": ["mouth"], "line_width_px": hr.FORM_WIDTH}
    assert "FORM_WIDTH" in rec["drawn"]["width_set_by"] and len(rec["drawn"]["eye_rings"]["frames_a_ring_is_left_out_on"]["l_pupil"]) == N
    assert set(rec["read_backs"]) == {"the_cut_body_model", "the_cut_source", "the_drawing"} and all(r["passed"] for r in rec["read_backs"].values())
    assert rec["ran_on"]["commit"] and set(rec["ran_on"]["tools"]) == set(hr.TOOLS)
    assert all(isinstance(t["changed_since"], bool) and t["last_changed_in"] for t in rec["ran_on"]["tools"].values())
    assert rec["sheet"]["file"] == "head_sheet.jpg" and len(rec["sheet"]["frames"]) == 6 and rec["files"]["reference"] == "head.mp4"
    assert any('"motion_file"' in line and str(out) in line and '"motion_first": 0' in line for line in said), "the load's keys are not printed"
    return (f"every file is there; the white is within {worst:.1f} px of the face's lines as the record's box and scale carry them, with no lips and "
            f"no points alone; the source is cut by the same box; the record names the inputs, the commits and the three read-backs")


def frames_from_the_middle() -> str:
    mesh_first, points_first, first, count = 95, 100, 104, 8
    centres = np.array([190.0, 120.0]) + np.array([[(k * 7) % 19 - 9, (k * 5) % 13 - 6] for k in range(20)], float)   # row k is source frame 100 + k
    folder, mesh, pts, source = rooms("middle", [40 + 3 * k for k in range(30)], [20 + k for k in range(120)], centres, points_first)
    out = folder / "head.mp4"
    rec = hr.run(str(mesh), str(pts), str(source), first, count, out, "the check", mesh_first=mesh_first, room=ROOM, say=quiet)
    assert rec["passed"] and rec["frames"] == [104, 111] and rec["inputs"]["its_first_frame_is_source_frame"] == 95 and rec["inputs"]["their_first_row_is_source_frame"] == 100
    got, under = read(out, count), read(hr.outputs(out)["source_cut"], count)
    box, scale = np.array(rec["box"], float), np.array(rec["scale"])
    for j in range(count):
        assert abs(int(np.median(got[j, :10, :10])) - (40 + 3 * (first - mesh_first + j))) <= 2, f"frame {j} is not the body model's frame {first - mesh_first + j}"
        assert abs(int(np.median(under[j])) - (20 + first + j)) <= 2, f"frame {j} of the cut source is not source frame {first + j}"
        ys, xs = np.nonzero(got[j].min(-1) > hr.faces.WHITE)
        d = np.linalg.norm(np.stack([xs, ys], 1).astype(float)[:, None, :] - landed(centres[first - points_first + j], box, scale)[None], axis=-1).min(1)
        assert len(xs) and d.max() <= hr.FORM_WIDTH + 1.0, (j, d.max())
    return "frames 104 to 111 of a body model from 95, points from 100 and a source from 0: each frame is its own, in all three"


def it_writes_over_nothing() -> str:
    folder, mesh, pts, source = rooms("over", [90] * N, [60] * N, CENTRES, 0)
    out = folder / "head.mp4"

    def refused(word: str, **change) -> None:
        a = dict(mesh=str(mesh), points=str(pts), source=str(source), first=0, count=N, out=out)
        a.update(change)
        before = sorted(p.name for p in folder.iterdir())
        try:
            hr.run(a["mesh"], a["points"], a["source"], a["first"], a["count"], a["out"], "the check", say=quiet)
        except (SystemExit, ValueError) as exc:
            assert word in str(exc), (word, exc)
        else:
            raise AssertionError(f"not refused: {word}")
        assert sorted(p.name for p in folder.iterdir()) == before, f"refused for {word!r}, and a file was written"

    refused(".mp4", out=folder / "head.mkv")
    refused("is not a file", mesh=str(folder / "nothing.mp4"))
    refused("is not a file", source=str(folder / "nothing.mov"))
    refused("were asked for", first=4)
    hr.outputs(out)["source_cut"].write_bytes(b"somebody's")
    refused("takes a new name")
    assert hr.outputs(out)["source_cut"].read_bytes() == b"somebody's", "a file that was there was written over"
    return "an existing output, an --out that is no .mp4, a missing input and a span the points do not cover are each refused, and no file is made"


def the_defaults_can_be_set_aside() -> str:
    folder, mesh, pts, source = rooms("aside", [90] * N, [60] * N, CENTRES, 0)
    rec = hr.run(str(mesh), str(pts), str(source), 0, N, folder / "all.mp4", "the check", width=2.5, room=ROOM, leave_out=frozenset(), dots=True, say=quiet)
    assert rec["passed"] and rec["drawn"]["parts_left_out"] == [] and rec["drawn"]["a_dot_at_every_point"] is True
    assert rec["drawn"]["line_width_px"] == 2.5 and rec["drawn"]["width_set_by"] == "--width" and "tip_of_chin" in rec["drawn"]["points_drawn_alone"]
    box, scale = np.array(rec["box"], float), np.array(rec["scale"])
    got = read(folder / "all.mp4", N)
    lips = landed(CENTRES[0], box, scale, without=("eye", "brow", "nose", "iris", "lash", "crease"))
    x, y = lips.mean(0).round().astype(int)
    assert (got[0, y - 12:y + 13, x - 20:x + 21].min(-1) > hr.faces.WHITE).sum() > 30, "with nothing left out there is no white on the lips"
    return "with dots, nothing left out and a width by hand, the lips and the points alone are drawn and the record says each was set"


def a_step_that_fails_fails_the_run() -> str:
    folder, mesh, pts, source = rooms("fails", [90] * N, [60] * N, CENTRES, 0)
    real = hr.faces.read_back
    hr.faces.read_back = lambda *a, **k: dict(real(*a, **k), passed=False)        # the drawing's read-back says no
    try:
        said = []
        rec = hr.run(str(mesh), str(pts), str(source), 0, N, folder / "head.mp4", "the check", say=said.append)
    finally:
        hr.faces.read_back = real
    assert rec["passed"] is False and rec["read_backs"]["the_drawing"]["passed"] is False and rec["read_backs"]["the_cut_body_model"]["passed"] is True
    assert json.loads((folder / "head.json").read_text())["passed"] is False and any(line.startswith("FAILED") for line in said)
    return "the drawing's read-back failing fails the run, in the record and in what is printed, with the other two steps still said to have passed"


def a_face_off_the_track_fails() -> str:
    """The points' own record says on which frames the face's middle is off the subject's track (2026-10-14: on the
    lead's 3167 eleven frames drew another woman's face and the run said PASSED). A short run of them inside the span
    takes its points from its neighbours (the lead's ruling that day, for the second subject's 895 and 897)."""
    seen = {}
    gap = hr.FILL_GAP
    longer = list(range(6, 6 + gap + 1))                                             # one frame longer than the gap: not filled
    for name, said_of_track in (("off", {"what": "the check's", "frames_whose_face_middle_is_off_the_mask": [0, 3, 4, *longer, 40]}),
                                ("short", {"what": "the check's", "frames_whose_face_middle_is_off_the_mask": [3, 4]}),
                                ("on", {"what": "the check's", "frames_whose_face_middle_is_off_the_mask": []}), ("unsaid", None)):
        folder, mesh, pts, source = rooms(f"track_{name}", [90] * N, [60] * N, CENTRES, 0)
        pts.with_suffix(".json").write_text(json.dumps({"made_by": "the check", "face_on_the_mask": said_of_track}))
        said: list[str] = []
        rec = hr.run(str(mesh), str(pts), str(source), 0, N, folder / "head.mp4", "the check", room=ROOM, say=said.append)
        seen[name] = (rec, said, json.loads((folder / "head.json").read_text()), folder, pts)
    rec, said, kept, folder, _ = seen["off"]
    t = rec["face_on_the_track"]
    # THE CASE THAT GOES RED when a frame at the span's edge, or a run longer than the gap, is filled; or a short one is not
    assert rec["passed"] is False and kept["passed"] is False and t["frames_off_the_track_not_filled"] == [0, *longer] \
        and t["frames_filled_from_their_neighbours"] == {"3": [2, 5], "4": [2, 5]}, t
    assert all(r["passed"] for r in rec["read_backs"].values()) and (folder / "head_sheet.jpg").is_file(), "the sheet is not there to be read"
    assert any(f"THE FACE IS OFF THE SUBJECT'S TRACK on {1 + len(longer)} of" in line for line in said) and any(line.startswith("FAILED") for line in said), said
    rec, said, kept, folder, pts = seen["short"]
    t = rec["face_on_the_track"]
    assert rec["passed"] is True and t["frames_filled_from_their_neighbours"] == {"3": [2, 5], "4": [2, 5]} and "filled, not read" in t["filled_are"], t
    assert any("3 from 2 and 5, 4 from 2 and 5" in line for line in said) and rec["inputs"]["points"] == str(pts), said
    z, f = np.load(pts), np.load(folder / "head_points_filled.npz")
    assert np.allclose(f["keypoints"][3], z["keypoints"][2] + (z["keypoints"][5] - z["keypoints"][2]) / 3, atol=1e-3) and np.allclose(f["keypoints"][6], z["keypoints"][6]), \
        "a filled frame is not the straight line between its neighbours, or a frame not filled changed"
    assert json.loads((folder / "head_points_filled.json").read_text())["filled_from_neighbours"] == {"3": [2, 5], "4": [2, 5]}
    rec, said, _, _, _ = seen["on"]
    assert rec["passed"] is True and rec["face_on_the_track"]["known"] and any(f"on all {N} frames" in line for line in said), said
    rec, said, _, _, _ = seen["unsaid"]
    assert rec["passed"] is True and rec["face_on_the_track"]["known"] is False and any("NOT KNOWN" in line and "says nothing of a track" in line for line in said), said
    assert hr.fill_from_neighbours([5, 6], 0, 12, gap=1) == ({}, [5, 6]) and hr.fill_from_neighbours([11], 0, 12) == ({}, [11])
    return (f"a run of frames off the track at most {gap} long, inside the span with frames on the track either side, takes its points from them, "
            "interpolated, and the record says so and calls them filled, not read; a frame at the span's edge and a longer run fail the run, with every "
            "read-back passing and the sheet written; a frame outside the span is not counted; a record with none passes; one that says nothing passes as not known")

def her_glasses_over_her_eyes() -> str:
    """2823 (2026-10-15): her white sunglasses over her eyes lie off her track, so the face's middle (the lash lines outvote the rest)
    lands off it on 13 frames though only she is there. Where the plan gives back a class over her eyes (`--over-the-eyes`), the face
    counts as on her track by the lower face's middle; nowhere else (895 and 897: another face behind her shoulder)."""
    from _lib import face_points as fp
    from _lib.frames import write_mask_video
    out = {}
    for name, spans in (("glasses", [(1, 8, "HER GLASSES")]), ("no_span", [])):
        folder, mesh, pts, source = rooms(f"track_{name}", [90] * N, [60] * N, CENTRES, 0)
        P = fp.load(str(pts), None)
        lash = [i for n, i in P["index"].items() if n.endswith("_lash_line")]
        track = np.ones((N, H, W), bool)
        for f in range(2, 8):                                                       # the glasses: a band over her eyes left out of the track
            ys = P["keypoints"][f][lash][:, 1]
            track[f, int(ys.min()) - 6:int(ys.max()) + 7, :] = False
        for f in (8, 9):                                                            # the whole face off the track
            track[f] = False
            track[f, :20, :20] = True
        pad = np.zeros((3, H, W), bool)                                             # the track starts three frames before the points
        pad[:, :20, :20] = True
        write_mask_video(folder / "track.mkv", np.concatenate([pad, track]))
        pts.with_suffix(".json").write_text(json.dumps({"made_by": "the check", "size": [W, H], "made_from": {"which_are_source_frames": [0, N - 1],
                                                       "box": {"mask_video": str(folder / "track.mkv"), "its_first_frame_is_source_frame": -3}},
                                                       "face_on_the_mask": {"what": "the check's", "frames_whose_face_middle_is_off_the_mask": [2, 3, 4, 5, 6, 7, 8, 9]}}))
        said: list[str] = []
        out[name] = (hr.run(str(mesh), str(pts), str(source), 0, N, folder / "head.mp4", "the check", room=ROOM, over_the_eyes=spans, say=said.append), said)
    rec, said = out["glasses"]
    t = rec["face_on_the_track"]
    # THE CASE THAT GOES RED when her own glasses off her track fail a span the plan gives them back on, or a face wholly off her track
    # (8) is taken on inside it, or one outside it (9) is
    assert t["over_the_eyes"]["on_the_track_by_the_lower_face"] == [2, 3, 4, 5, 6, 7] and t["frames_whose_face_middle_is_off_the_track"] == [8, 9], t
    assert rec["passed"] is True and t["frames_filled_from_their_neighbours"] == {"8": [7, 10], "9": [7, 10]}, t
    assert any("6 frame(s) whose face's middle lies on her own glasses" in x and "2-7" in x for x in said), said
    assert t["over_the_eyes"]["spans"] == [[1, 8, "HER GLASSES"]] and t["over_the_eyes"]["read"] is True, t
    # a record that does not say where its track starts: not read, and said
    folder, mesh, pts, source = rooms("track_glasses_unsaid", [90] * N, [60] * N, CENTRES, 0)
    pts.with_suffix(".json").write_text(json.dumps({"made_by": "the check", "face_on_the_mask": {"what": "the check's", "frames_whose_face_middle_is_off_the_mask": [3]}}))
    said = []
    t = hr.run(str(mesh), str(pts), str(source), 0, N, folder / "head.mp4", "the check", room=ROOM, over_the_eyes=[(1, 8, "HER GLASSES")], say=said.append)["face_on_the_track"]
    assert t["over_the_eyes"]["read"] is False and t["over_the_eyes"]["frames"] == [3] and any("--over-the-eyes NOT READ on 1 frame" in x for x in said), (t, said)
    rec, said = out["no_span"]
    assert rec["passed"] is False and rec["face_on_the_track"]["frames_off_the_track_not_filled"] == [2, 3, 4, 5, 6, 7, 8, 9], rec["face_on_the_track"]
    assert hr.no_mouth_arg("3063-3097=her glasses", "--over-the-eyes") == (3063, 3097, "her glasses")
    return ("inside a span the plan gives back her glasses on, a face whose middle lies on them, off her track, is on it by the lower face; a face "
            "wholly off her track stays off, inside the span or out; with no span, the same frames fail the run")


def lt_says_plain(drawn: dict) -> bool:
    import load_text as lt
    return lt.subject_2(drawn).endswith("posed as the person moves.")


def no_face_where_hair_alone() -> str:
    """second_hp_5461b (2026-10-15): a hair-alone load's reference drew eyes, brows and a nose on her crown. `--no-face-on` draws no
    line of the face on its frames, the plain body model alone; where it covers the whole span the record lists every part left out."""
    import draw_face_points as dfp
    seen = {}
    for name, spans in (("some", [(3, 5, "HAIR ALONE")]), ("all", [(0, N - 1, "HAIR ALONE")])):
        folder, mesh, pts, source = rooms(f"no_face_{name}", [90] * N, [60] * N, CENTRES, 0)
        said: list[str] = []
        rec = hr.run(str(mesh), str(pts), str(source), 0, N, folder / "head.mp4", "the check", room=ROOM, no_face_on=spans, say=said.append)
        seen[name] = (rec, said, folder)
    rec, said, folder = seen["some"]
    rw, rh = rec["size"]
    white = [int((next(dfp.stream(str(folder / "head.mp4"), (rw, rh), f, 1, pix="rgb24")).max(-1) > 200).sum()) for f in range(N)]
    # THE CASE THAT GOES RED when a line of the face is drawn on a hair-alone frame, or the face is dropped where it is not
    assert all(white[f] == 0 for f in (3, 4, 5)) and all(white[f] > 0 for f in (0, 1, 2, 6, 7)), white
    assert rec["drawn"]["frames_left_bare"] == [3, 4, 5] and rec["passed"] and any("NO FACE LINES on 3 of 12 frames (3-5): HAIR ALONE" in x for x in said), said
    rec, said, _ = seen["all"]
    assert {"eyes", "brows", "nose", "mouth"} <= set(rec["drawn"]["parts_left_out"]) and rec["drawn"]["frames_left_bare"] == list(range(N)), rec["drawn"]
    # a person's word that the mouth is drawn does not draw one where no face is: the record says no mouth, and the text none
    folder, mesh, pts, source = rooms("no_face_word", [90] * N, [60] * N, CENTRES, 0)
    worded = hr.run(str(mesh), str(pts), str(source), 0, N, folder / "head.mp4", "the check", room=ROOM, mouth="drawn", mouth_why="A WORD",
                    no_face_on=[(0, N - 1, "HAIR ALONE")], say=lambda *_: None)
    # THE CASE THAT GOES RED when a reference with no face on any frame records (and the text then says) a drawn mouth
    assert worded["drawn"]["mouth"]["drawn"] is False and lt_says_plain(worded["drawn"]), worded["drawn"]["mouth"]
    assert lt_says_plain(rec["drawn"]), "the text's <Subject 2> still speaks of drawn lines"
    # HER EYES SHUT on a person's word (--eyes-shut-on, the lead's ruling of 2026-10-15 on 2823b): passed to the drawing, said, recorded,
    # and not on a frame that has no face at all
    folder, mesh, pts, source = rooms("eyes_shut", [90] * N, [60] * N, CENTRES, 0)
    said = []
    shut = hr.run(str(mesh), str(pts), str(source), 0, N, folder / "head.mp4", "the check", room=ROOM, eyes_shut_on=[(2, 4, "G3 LOOKED")],
                  eyes_shut_form="lids", no_face_on=[(4, 4, "HAIR ALONE")], say=said.append)
    # THE CASE THAT GOES RED when the shut frames do not reach the drawing, or the record does not say them with their form and word
    assert shut["drawn"]["eyes_shut"] == {"form": "lids", "frames": [[2, 3]], "on_whose_word": [{"first": 2, "last": 4, "why": "G3 LOOKED"}],
                                          "drawn_shut_on": [[2, 3]]}, shut["drawn"]
    assert any("HER EYES SHUT on 2 of 12 frames (2-3), drawn 'lids': G3 LOOKED" in x for x in said), said
    try:
        hr.run(str(mesh), str(pts), str(source), 0, N, folder / "x.mp4", "the check", room=ROOM, eyes_shut_on=[(2, 4, "X")], say=lambda *_: None)
        raise AssertionError("--eyes-shut-on with no form was taken")
    except SystemExit as e:
        assert "takes --eyes-shut-form" in str(e), e
    return ("no line of the face on the frames --no-face-on names, the others drawn; over the whole span every part is recorded left out, and "
            "the text's <Subject 2> sentence then speaks of no drawing")


def the_mouth_by_the_check() -> str:
    """The chain's default of 2026-10-14: the mouth drawn when the lip lines lie on the class map's mouth, left out when they do not
    (the lead's 0294d: the points bunched on her upper lip while the class map held her open mouth)."""
    import cv2
    import capture_masked_run as cap
    from _lib import face_points as fp
    names = cap._pack("sapiens2_parts").CLASS_NAMES
    seen = {}
    for name, on in (("on", True), ("off", False), ("none", None), ("mixed", "mixed"), ("mostly", "mostly")):
        folder, mesh, pts, source = rooms(f"mouth_{name}", [90] * N, [60] * N, CENTRES, 0)
        P = fp.load(str(pts), None)
        lips = [i for k, i in P["index"].items() if "outer_lip" in k or k in ("r_outer_corner_of_mouth", "l_outer_corner_of_mouth")]
        cm = np.zeros((N, H, W), np.uint8)
        for f in range(N):
            if on is None:
                continue                                                  # a class map with no lips on it
            xy = P["keypoints"][f][lips]
            if on is True or (on == "mixed" and f >= 8) or (on == "mostly" and f < 9):    # the lips on the points, an open mouth inside them
                ring = np.round(xy).astype(np.int32)
                cv2.fillPoly(cm[f], [cv2.convexHull(ring)], names.index("Tongue"))
                cv2.polylines(cm[f], [cv2.convexHull(ring)], True, names.index("Upper_Lip"), 3)
            else:                                                         # the lips 40 px off the points, an open mouth inside them
                cx, cy = (int(round(v)) + 40 for v in xy.mean(0))
                cv2.ellipse(cm[f], (cx, cy), (14, 9), 0, 0, 360, names.index("Upper_Lip"), -1)
                cv2.ellipse(cm[f], (cx, cy), (8, 4), 0, 0, 360, names.index("Tongue"), -1)
        classes = folder / "classes.mkv"
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", "24", "-i", "-", "-c:v", "ffv1",
                                "-level", "3", str(classes)], stdin=subprocess.PIPE)
        enc.stdin.write(cm.tobytes()); enc.stdin.close(); assert enc.wait() == 0
        said: list[str] = []
        rec = hr.run(str(mesh), str(pts), str(source), 0, N, folder / "head.mp4", "the check", room=ROOM, mouth="auto", classes=str(classes), classes_first=0,
                     no_mouth_on=[(10, 11, "HAIR ALONE")] if on == "mixed" else None, say=said.append)
        seen[name] = (rec, said)
    on, said_on = seen["on"]
    off, said_off = seen["off"]
    none, said_none = seen["none"]
    # THE CASE THAT GOES RED when the mouth is drawn whatever the check says, or left out whatever it says
    assert on["drawn"]["mouth"]["drawn"] and on["drawn"]["mouth"]["drawn_as"] == "the lip lines" and "mouth" not in on["drawn"]["parts_left_out"] \
        and on["drawn"]["mouth"]["check"]["frames_at_or_over_the_share"] == N, on["drawn"]
    # the lines drawn over the class map's inside, not over the body model's sculpted lips: no hollow mouth (the outside reviewer)
    assert on["drawn"]["mouth"]["inside"] == "the class map" and on["drawn"]["mouth"]["inside_on_frames"] == N, on["drawn"]["mouth"]
    import draw_face_points as dfp
    P = fp.load(str(WORK / "mouth_on" / "shot_points.npz"), None)
    lips = [i for k, i in P["index"].items() if "outer_lip" in k or k in ("r_outer_corner_of_mouth", "l_outer_corner_of_mouth")]
    rw, rh = on["size"]
    x0, y0, x1, y1 = on["box"]
    cx, cy = P["keypoints"][N // 2][lips].mean(0)
    pic = next(dfp.stream(str(WORK / "mouth_on" / "head.mp4"), (rw, rh), N // 2, 1, pix="rgb24"))
    inner = pic[int(round((cy - y0) * rh / (y1 - y0))), int(round((cx - x0) * rw / (x1 - x0)))]
    assert inner.max() < 60, ("the lip lines' inside is not dark where the class map's mouth is open", inner.tolist())
    assert any("the mouth: DRAWN by the check" in x for x in said_on), said_on
    # where the class map's lips sit 40 px off the pose model's mouth on every frame (far, as on 1010d's chest), its gate keeps no
    # frame: no mouth on any frame, by the class map's own gate and not the lines' share
    m = off["drawn"]["mouth"]
    assert m["drawn"] and m["drawn_as"] == "the class map" and "mouth" in off["drawn"]["parts_left_out"] and m["check"]["frames_at_or_over_the_share"] == 0 \
        and m["no_mouth_on"] == {hr.NO_MOUTH_CLASS_GATE: [[0, N - 1]]} and m["frames_with_no_mouth"] == N and m["gate"]["frames_drawn"] == 0, m

    def at(name: str, rec: dict, f: int, shift: float) -> np.ndarray:
        """The reference's pixel at the points' mouth centre (+ `shift` px), frame `f`."""
        rw, rh = rec["size"]
        x0, y0, x1, y1 = rec["box"]
        sx, sy = rw / (x1 - x0), rh / (y1 - y0)
        P = fp.load(str(WORK / f"mouth_{name}" / "shot_points.npz"), None)
        lips = [i for k, i in P["index"].items() if "outer_lip" in k or k in ("r_outer_corner_of_mouth", "l_outer_corner_of_mouth")]
        cx, cy = P["keypoints"][f][lips].mean(0) + shift
        pic = next(dfp.stream(str(WORK / f"mouth_{name}" / "head.mp4"), (rw, rh), f, 1, pix="rgb24"))
        return pic, int(round((round(cx) - x0) * sx)), int(round((round(cy) - y0) * sy)), sy

    pic, ux, uy, _ = at("off", off, N // 2, 40)
    assert pic[uy, ux].max() > 60, ("a class-map mouth was drawn where the lip lines miss it", pic[uy, ux].tolist())
    # 630b's 672 and 1010d's 1026, on one load: the lines miss on 0-7 (672: share 0, the class map's lips a quarter face off), lie on it on
    # 8-11, and 10-11 are frames whose part holds no lips (1026: hair alone). A mouth on 8 and 9 alone, by the lip lines, with the
    # class map's inside under them
    x = seen["mixed"][0]["drawn"]["mouth"]
    said_mixed = seen["mixed"][1]
    # THE CASE THAT GOES RED when a mouth is drawn on a frame whose lips miss (672), or on a frame whose part holds no lips (1026)
    assert x["drawn_as"] == "the class map" and x["no_mouth_on"] == {hr.NO_MOUTH_CLASS_GATE: [[0, 7]],
                                                                     hr.NO_MOUTH_PART.format(why="HAIR ALONE"): [[10, 11]]} and x["frames_with_no_mouth"] == 10, x
    assert x["frames_by_the_lip_lines"] == [[8, 9]] and x["frames_by_the_class_map"] == [], x     # 10-11: no mouth, whatever places it
    assert any("NO MOUTH on 8 of 12 frames (0-7)" in y for y in said_mixed) and any("NO MOUTH on 2 of 12 frames (10-11)" in y for y in said_mixed), said_mixed
    pic, ux, uy, _ = at("mixed", seen["mixed"][0], 3, 40)
    assert pic[uy, ux].max() > 60, ("672: a mouth drawn where the lips miss", pic[uy, ux].tolist())
    pic, ux, uy, _ = at("mixed", seen["mixed"][0], 10, 0)
    assert pic[uy, ux].max() > 60, ("1026: a mouth drawn on a frame whose part holds no lips", pic[uy, ux].tolist())
    # ... and where it is drawn, the class map's own mouth: the open inside dark, ringed by a white outline
    pic, ux, uy, sy = at("mixed", seen["mixed"][0], 9, 0)
    assert pic[uy, ux].max() < 60, ("the class map's open mouth is not dark in the reference", pic[uy, ux].tolist())
    col = pic[:, ux].max(-1)
    assert (col[max(uy - int(30 * sy), 0):uy] > 200).any(), "no white outline above the class map's open mouth"
    # the lip lines drawn (they lie on the mouth on 9 of 12 frames) and none on the three they miss: no white ring at her mouth there
    y = seen["mostly"][0]["drawn"]["mouth"]
    assert y["drawn_as"] == "the lip lines" and y["no_mouth_on"] == {hr.NO_MOUTH_CLASS_GATE: [[9, 11]]}, y

    def white_at_mouth(f: int) -> int:
        pic, ux, uy, sy = at("mostly", seen["mostly"][0], f, 0)
        r = int(25 * sy)
        return int((pic[max(uy - r, 0):uy + r, max(ux - r, 0):ux + r].max(-1) > 200).sum())
    # THE CASE THAT GOES RED when the points' lip lines are still drawn on a frame whose lines miss the mouth (805b's 813-818)
    # (the window also holds the nose's and the chin's lines, so the frame with no mouth is held to a quarter of one with it)
    assert white_at_mouth(10) * 4 < white_at_mouth(4), (white_at_mouth(4), white_at_mouth(10))
    assert hr.no_mouth_arg("1025-1031=hair alone") == (1025, 1031, "hair alone")
    for bad in ("1025-1031", "1031-1025=x", "x=y", "1011-1017-1030=x"):
        try:
            hr.no_mouth_arg(bad)
            raise AssertionError(f"taken: {bad}")
        except SystemExit:
            pass
    # with no lips in the class map either, nothing is drawn of the mouth, and the record says why
    n = none["drawn"]["mouth"]
    assert not n["drawn"] and "drawn_as" not in n and n["class_map"]["frames_holding_the_lips"] == 0 and "mouth" in none["drawn"]["parts_left_out"], none["drawn"]
    assert any("the mouth: LEFT OUT: the class map holds her lips on none of the 12 frames" in x for x in said_none), said_none
    folder, mesh, pts, source = rooms("mouth_word", [90] * N, [60] * N, CENTRES, 0)
    try:
        hr.run(str(mesh), str(pts), str(source), 0, N, folder / "head.mp4", "the check", room=ROOM, mouth="drawn", say=lambda *_: None)
    except SystemExit as e:
        assert "--mouth-why" in str(e), e
    else:
        raise AssertionError("the mouth drawn on a word with no reason")
    return ("the mouth drawn where the lip lines lie on the class map's mouth, its inside under them; where they do not on most frames, the span "
            "in the class map's form, the lines still drawn on the frames they lie on it; no mouth on a frame whose lines miss and whose class-map "
            "lips lie far off (630b's 672) or whose part holds no lips (1010d's 1026); the figures in the record; drawn on a word only with its reason")

def the_class_map_mouth_by_its_gate() -> str:
    """The lead's fault of 2026-10-15: on a small or turned face the lip lines miss her mouth on most frames, so the class map's own mouth
    is drawn, and the lines' share then dropped it on those same frames (630's 40 of 51). Here the class map's lips sit at the pose model's
    mouth, clear of the lip ring, on every frame (the lines' share under `MOUTH_CHECK["on_share"]` on all 12), and on 6-11 the body
    model is black round the mouth (630b's 658-680, drawn in the air)."""
    import cv2
    import capture_masked_run as cap
    import draw_face_points as dfp
    from _lib import face_points as fp
    names = cap._pack("sapiens2_parts").CLASS_NAMES
    folder, mesh, pts, source = rooms("mouth_gate", [90] * N, [60] * N, CENTRES, 0)
    P = fp.load(str(pts), None)
    lips = [i for k, i in P["index"].items() if "outer_lip" in k or k in ("r_outer_corner_of_mouth", "l_outer_corner_of_mouth")]
    cm = np.zeros((N, H, W), np.uint8)
    body = grey([90] * N)
    for f in range(N):
        xy = P["keypoints"][f][lips]
        cx, cy = (int(round(v)) for v in xy.mean(0))
        ax, ay = (max(int(0.35 * (xy[:, i].max() - xy[:, i].min()) / 2), 3) for i in (0, 1))
        cv2.ellipse(cm[f], (cx, cy), (ax, ay), 0, 0, 360, names.index("Upper_Lip"), -1)
        cv2.ellipse(cm[f], (cx, cy), (max(ax // 2, 1), max(ay // 2, 1)), 0, 0, 360, names.index("Tongue"), -1)
        if f >= 6:
            cv2.circle(body[f], (cx, cy), 40, (0, 0, 0), -1)
    video(mesh, body)
    classes = folder / "classes.mkv"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", "24", "-i", "-", "-c:v", "ffv1",
                            "-level", "3", str(classes)], stdin=subprocess.PIPE)
    enc.stdin.write(cm.tobytes()); enc.stdin.close(); assert enc.wait() == 0
    said: list[str] = []
    rec = hr.run(str(mesh), str(pts), str(source), 0, N, folder / "head.mp4", "the check", room=ROOM, mouth="auto", classes=str(classes), classes_first=0,
                 say=said.append)
    m = rec["drawn"]["mouth"]
    assert m["check"]["frames_at_or_over_the_share"] == 0 and m["drawn_as"] == "the class map", m
    rows = {r["frame"]: r for r in m["gate"]["a_frame_at_a_time"]}
    # THE CASE THAT GOES RED when the lines' share drops the class map's mouth on frames where it sits on her face (the 2026-10-15 fault),
    # or when a mouth over no body model (in the air) is kept
    assert m["no_mouth_on"] == {hr.NO_MOUTH_CLASS_GATE: [[6, 11]]} and m["gate"]["frames_drawn"] == 6, m
    assert all(rows[f]["drawn"] and rows[f]["on_body"] == 1.0 and rows[f]["from_the_pose_mouth_in_face_heights"] < 0.05 for f in range(6)), rows
    assert all(not rows[f]["drawn"] and rows[f]["on_body"] == 0.0 for f in range(6, N)), rows
    assert any("the class map's own mouth on 6 (" in x for x in said) and any("NO MOUTH on 6 of 12 frames (6-11)" in x for x in said), said
    rw, rh = rec["size"]
    x0, y0, x1, y1 = rec["box"]

    def inside(f: int) -> int:
        cx, cy = P["keypoints"][f][lips].mean(0)
        pic = next(dfp.stream(str(folder / "head.mp4"), (rw, rh), f, 1, pix="rgb24"))
        return int(pic[int(round((cy - y0) * rh / (y1 - y0))), int(round((cx - x0) * rw / (x1 - x0)))].max())
    assert inside(2) < 60 and inside(8) < 60, (inside(2), inside(8))   # dark: the open mouth on 2, the black body model on 8
    pic = next(dfp.stream(str(folder / "head.mp4"), (rw, rh), 8, 1, pix="rgb24"))
    assert (pic.max(-1) > 200).sum() < (next(dfp.stream(str(folder / "head.mp4"), (rw, rh), 2, 1, pix="rgb24")).max(-1) > 200).sum(), \
        "the class map's white outline drawn on a frame its gate dropped"
    return ("where the lines miss on every frame and the class map's lips sit at the pose model's mouth, its mouth is drawn on each frame its "
            "own gate keeps (on the body model, near the pose mouth) and dropped where the body model is not under it; the figures in the record")


def no_clip_gate_and_the_box_first() -> str:
    """The outside reviewer's G4 of ca86c125: (A) a span whose class map holds her lips on few frames still draws them where each frame
    passes; (B) a larger lip blob outside the head's box does not win over her mouth inside it."""
    import cv2
    import capture_masked_run as cap
    from _lib import face_points as fp
    names = cap._pack("sapiens2_parts").CLASS_NAMES
    folder, mesh, pts, source = rooms("clip_and_box", [90] * N, [60] * N, CENTRES, 0)
    P = fp.load(str(pts), None)
    lips = [i for k, i in P["index"].items() if "outer_lip" in k or k in ("r_outer_corner_of_mouth", "l_outer_corner_of_mouth")]
    cm = np.zeros((N, H, W), np.uint8)
    for f in range(4):                                                  # her lips on 4 of 12 frames, clear of the lip ring
        xy = P["keypoints"][f][lips]
        cx, cy = (int(round(v)) for v in xy.mean(0))
        ax, ay = (max(int(0.35 * (xy[:, i].max() - xy[:, i].min()) / 2), 3) for i in (0, 1))
        cv2.ellipse(cm[f], (cx, cy), (ax, ay), 0, 0, 360, names.index("Upper_Lip"), -1)
    cm[0, 2:60, 2:60] = names.index("Upper_Lip")                         # a larger lip blob, far outside the head's box
    classes = folder / "classes.mkv"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", "24", "-i", "-", "-c:v", "ffv1",
                            "-level", "3", str(classes)], stdin=subprocess.PIPE)
    enc.stdin.write(cm.tobytes()); enc.stdin.close(); assert enc.wait() == 0
    rec = hr.run(str(mesh), str(pts), str(source), 0, N, folder / "head.mp4", "the check", room=ROOM, mouth="auto", classes=str(classes), classes_first=0,
                 say=lambda *_: None)
    m = rec["drawn"]["mouth"]
    rows = {r["frame"]: r for r in m["gate"]["a_frame_at_a_time"]}
    x0, y0, x1, y1 = rec["box"]
    assert x0 > 60 or y0 > 60, ("the fixture's far blob is inside the box", rec["box"])
    # THE CASE THAT GOES RED when a span with her lips on under half its frames loses the mouths each frame passes (A), or the far
    # blob is taken for her mouth on frame 0 (B)
    assert m["drawn"] and m["frames_by_the_class_map"] == [[0, 3]] and m["class_map"]["frames_holding_the_lips"] == 4, m
    assert rows[0]["drawn"] and rows[0]["from_the_pose_mouth_in_face_heights"] < 0.05, rows[0]
    return ("a span whose class map holds her lips on 4 of 12 frames draws them on those 4 (no clip-level gate); a larger lip blob outside the "
            "head's box is not taken for her mouth")


def needs_hole_option() -> str:
    """The lead's word of 2026-10-15 for one test (5114e): with --class-mouth-needs-hole, a frame given the class map's own mouth with no
    teeth or tongue labelled draws no mouth; without it, every such frame is drawn as before."""
    import cv2
    import capture_masked_run as cap
    from _lib import face_points as fp
    names = cap._pack("sapiens2_parts").CLASS_NAMES
    out = {}
    for name, option in (("off", None), ("on", "A TEST")):
        folder, mesh, pts, source = rooms(f"hole_{name}", [90] * N, [60] * N, CENTRES, 0)
        P = fp.load(str(pts), None)
        lips = [i for k, i in P["index"].items() if "outer_lip" in k or k in ("r_outer_corner_of_mouth", "l_outer_corner_of_mouth")]
        cm = np.zeros((N, H, W), np.uint8)
        for f in range(N):
            xy = P["keypoints"][f][lips]
            cx, cy = (int(round(v)) for v in xy.mean(0))
            ax, ay = (max(int(0.35 * (xy[:, i].max() - xy[:, i].min()) / 2), 3) for i in (0, 1))
            cv2.ellipse(cm[f], (cx, cy), (ax, ay), 0, 0, 360, names.index("Upper_Lip"), -1)
            if f >= 6:                                                  # 6-11 open: a tongue inside; 0-5 closed: lips alone
                cv2.ellipse(cm[f], (cx, cy), (max(ax // 2, 1), max(ay // 2, 1)), 0, 0, 360, names.index("Tongue"), -1)
        classes = folder / "classes.mkv"
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", "24", "-i", "-", "-c:v", "ffv1",
                                "-level", "3", str(classes)], stdin=subprocess.PIPE)
        enc.stdin.write(cm.tobytes()); enc.stdin.close(); assert enc.wait() == 0
        out[name] = hr.run(str(mesh), str(pts), str(source), 0, N, folder / "head.mp4", "the check", room=ROOM, mouth="auto", classes=str(classes),
                           classes_first=0, class_mouth_needs_hole=option, say=lambda *_: None)["drawn"]["mouth"]
    # THE CASE THAT GOES RED when the option is on by default, or when on it still draws a mouth with no teeth or tongue
    assert out["off"]["frames_by_the_class_map"] == [[0, N - 1]] and "class_mouth_needs_hole" not in out["off"], out["off"]
    assert out["on"]["frames_by_the_class_map"] == [[6, N - 1]] and out["on"]["class_mouth_needs_hole"]["no_mouth_on"] == [[0, 5]], out["on"]
    assert any("--class-mouth-needs-hole asks for none drawn (A TEST)" in k for k in out["on"]["no_mouth_on"]), out["on"]["no_mouth_on"]
    rows = out["on"]["gate"]["a_frame_at_a_time"]
    # the record's rows say what was drawn after every rule, and why not
    assert rows[0]["no_teeth_or_tongue"] and not rows[0]["drawn"] and rows[7]["drawn"] and out["on"]["gate"]["frames_drawn"] == 6, rows[:1] + rows[7:8]
    return "off unless asked; on, a class-map frame with no teeth or tongue draws no mouth and the record says why, the open frames drawn"


def the_mouth_over_time() -> str:
    """Mouth v2 (the lead's ruling of 2026-10-15): the form held against its neighbours, and a short drop-out filled (`MOUTH_RUNS`)."""
    lines, leave = hr.MOUTH_CHECK["on_share"], hr.MOUTH_RUNS["leave_lines"]
    shares = [1.0, leave, leave - 0.01, 1.0, lines - 0.01, 1.0]
    on, passes, held_on, filled, _ = hr.mouth_over_time(shares, [False] * 6, [None] * 6, [None] * 6, set(), 100)
    # THE CASE THAT GOES RED when the lines are left at the share they entered by (2823's 46 switches), or entered under it
    assert on == [True, True, False, True, True, True] and held_on == [101, 104], (on, held_on)
    assert hr.mouth_over_time([0.7, 0.7, 1.0], [False] * 3, [None] * 3, [None] * 3, set(), 0)[0] == [False, False, True], "the lines entered under their share"
    assert hr.mouth_over_time([1.0, 0.7, 1.0], [False] * 3, [None] * 3, [None] * 3, {0}, 0)[0] == [True, False, True], "held on the lines after a no-mouth frame"
    body = hr.MOUTH_RUNS["fill_on_body"]
    near_ok = hr.CLASS_MOUTH_GATE["near"]
    def fill(passes, on_body, partless=frozenset(), near=None, inside=None):
        return hr.mouth_over_time([0.0] * len(passes), passes, on_body, near or [near_ok] * len(passes), set(partless), 0, inside)[3]
    # THE CASE THAT GOES RED when a drop-out inside one sung vowel is not filled (805's 822-823), or a longer one, one under the floor,
    # one at the edge or one on a frame with no mouth whatever is
    assert fill([True, False, False, True], [0.9, body, body, 0.9]) == [1, 2]
    assert fill([True, False, False, False, True], [0.9, body, body, body, 0.9]) == [], "a run of three was filled"
    assert fill([True, False, True], [0.9, body - 0.01, 0.9]) == [], "a frame under the fill's floor was filled"
    assert fill([False, False, True], [body, body, 0.9]) == [], "a run at the edge was filled"
    assert fill([True, False, True], [0.9, 0.9, 0.9], {1}) == [], "a frame with no mouth whatever places it was filled"
    assert fill([True, False, True], [0.9, None, 0.9]) == [], "a frame with no lips was filled"
    # THE CASE THAT GOES RED when the fill draws a mouth far from the pose mouth (1010d's chest), which the gate's near keeps out
    assert fill([True, False, True], [0.9, 0.9, 0.9], near=[0.1, near_ok + 0.01, 0.1]) == [], "a far mouth was filled"
    # the hole option first: a frame it drops is not filled back, and a frame on the lines keeps them whatever its teeth
    on, passes, _, filled, holeless = hr.mouth_over_time([0.0, 0.0, 0.0, 1.0], [True, True, True, False], [0.9] * 4, [0.1] * 4, set(), 0, [5, 0, 5, 0])
    # THE CASE THAT GOES RED when the option is undone by the fill, or drops a frame on the lines
    assert holeless == [1] and filled == [] and passes == [True, False, True, False] and on[3], (holeless, filled, passes, on)
    return ("the lip lines entered at their share and left only under leave_lines; a drop-out of one or two frames between mouths filled where "
            "the class map's mouth is on the body model at fill_on_body, never a longer one, one at the edge, one under it or one with no mouth")


def the_gate_at_its_bounds() -> str:
    """`class_mouth_gate` on either side of each value: a mouth of 100 px with 60 of them on the body model passes and 59 does not
    (`on_body` 0.6); a centre 0.399 face heights from the pose mouth passes and 0.401 does not (`near` 0.4)."""
    from _lib import face_points as fp
    folder, _, pts, _ = rooms("gate_bounds", [90] * 4, [60] * 4, CENTRES[:4], 0)
    P = fp.load(str(pts), None)
    idx = P["index"]
    lips = [i for k, i in idx.items() if k.endswith("_of_upper_outer_lip") or k.endswith("_of_lower_outer_lip") or k in ("r_outer_corner_of_mouth", "l_outer_corner_of_mouth")]
    pw, ph = 64, 48
    whole = np.zeros((ph, pw), bool)
    whole[10:20, 10:20] = True                                          # 100 px
    body = np.zeros((4, ph, pw, 3), np.uint8)
    body[:, 10:20, 10:16] = 90                                          # 60 of them
    body[1, 19, 15] = 0                                                 # 59 on frame 1
    body[2:, 10:20, 10:20] = 90                                         # all of them on 2 and 3
    plain = video(folder / "plain.mkv", body)
    centres = []
    for f, at in enumerate((0.0, 0.0, 0.399, 0.401)):
        k = P["keypoints"][f]
        c = k[lips].mean(axis=0)
        centres.append((float(c[0]) + at * fp.face_height(k, idx), float(c[1])))
    passes, gate = hr.class_mouth_gate([(whole, whole)] * 4, centres, str(plain), str(pts), 0, 4)
    rows = gate["a_frame_at_a_time"]
    # THE CASE THAT GOES RED when either value is read the wrong side of its bound
    assert passes == [True, False, True, False], (passes, rows)
    # where the pose and the class map agree (share and near), on_body's floor is agreed_on_body: 47 of 100 px pass, 46 do not
    g = hr.CLASS_MOUTH_GATE
    agreed_body = np.zeros((4, ph, pw, 3), np.uint8)
    agreed_body[:, 10:20, 10:15] = 90                                    # 50 px
    agreed_body[0, 10:13, 14] = 0                                        # 47 on frame 0
    agreed_body[1, 10:14, 14] = 0                                        # 46 on frame 1
    agreed_plain = video(folder / "agreed.mkv", agreed_body)
    near = []                                                           # 0.1 of a face height from the pose mouth: inside agreed_near
    for f in range(4):
        k = P["keypoints"][f]
        c = k[lips].mean(axis=0)
        near.append((float(c[0]) + 0.1 * fp.face_height(k, idx), float(c[1])))
    agreed, _ = hr.class_mouth_gate([(whole, whole)] * 4, near, str(agreed_plain), str(pts), 0, 4,
                                    [g["agreed_share"], g["agreed_share"], g["agreed_share"] - 0.01, g["agreed_share"]])
    # THE CASE THAT GOES RED when 805's open three-quarter mouths are not kept where the pose agrees, or the floor drops for any frame
    assert agreed == [True, False, False, True], agreed
    far = [(x + (g["agreed_near"] + 0.02 - 0.1) * fp.face_height(P["keypoints"][f], idx), y) for f, (x, y) in enumerate(near)]
    assert hr.class_mouth_gate([(whole, whole)] * 4, far, str(agreed_plain), str(pts), 0, 4, [g["agreed_share"]] * 4)[0] == [False] * 4, \
        "the agreed floor taken past agreed_near"
    assert rows[0]["on_body"] == 0.6 and rows[1]["on_body"] == 0.59 and abs(rows[2]["from_the_pose_mouth_in_face_heights"] - 0.399) < 1e-3, rows
    return "60 of 100 px on the body model passes and 59 does not; 0.399 of a face height from the pose mouth passes and 0.401 does not"


def a_frame_at_a_time_in_the_lines_form() -> str:
    """The lead's widening of 2026-10-15: 2823's close-up, drawn in the lip lines' form, had no mouth on 61 frames where her mouth was
    wide open, because the lines' share alone dropped them. A frame at a time: the lines where they lie on the class map's mouth, the
    class map's own mouth where they miss and its gate passes, no mouth otherwise. Here 0-5 lips on the lip ring (the lines); 6-7 lips
    at the pose mouth clear of the ring (the class map's own); 8 no lips; 9 lips 40 px off over the body model (far); 10-11 lips at
    the pose mouth over black (in the air). The class map starts two frames before the span (`classes_first`)."""
    import cv2
    import capture_masked_run as cap
    import draw_face_points as dfp
    from _lib import face_points as fp
    names = cap._pack("sapiens2_parts").CLASS_NAMES
    folder, mesh, pts, source = rooms("mouth_each_frame", [90] * N, [60] * N, CENTRES, 0)
    P = fp.load(str(pts), None)
    lips = [i for k, i in P["index"].items() if "outer_lip" in k or k in ("r_outer_corner_of_mouth", "l_outer_corner_of_mouth")]
    pad = 2
    cm = np.zeros((N + pad, H, W), np.uint8)
    cm[:pad, 10:40, 10:40] = names.index("Upper_Lip")                  # read by an offset bug, these would be far lips on 0-1
    body = grey([90] * N)
    for f in range(N):
        xy = P["keypoints"][f][lips]
        cx, cy = (int(round(v)) for v in xy.mean(0))
        ax, ay = (max(int(0.35 * (xy[:, i].max() - xy[:, i].min()) / 2), 3) for i in (0, 1))
        if f < 6:
            ring = np.round(xy).astype(np.int32)
            cv2.fillPoly(cm[pad + f], [cv2.convexHull(ring)], names.index("Tongue"))
            cv2.polylines(cm[pad + f], [cv2.convexHull(ring)], True, names.index("Upper_Lip"), 3)
        elif f == 8:
            continue
        else:
            dx = 40 if f == 9 else 0
            cv2.ellipse(cm[pad + f], (cx + dx, cy + dx), (ax, ay), 0, 0, 360, names.index("Upper_Lip"), -1)
            cv2.ellipse(cm[pad + f], (cx + dx, cy + dx), (max(ax // 2, 1), max(ay // 2, 1)), 0, 0, 360, names.index("Tongue"), -1)
        if f >= 10:
            cv2.circle(body[f], (cx, cy), 40, (0, 0, 0), -1)
    video(mesh, body)
    classes = folder / "classes.mkv"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", "24", "-i", "-", "-c:v", "ffv1",
                            "-level", "3", str(classes)], stdin=subprocess.PIPE)
    enc.stdin.write(cm.tobytes()); enc.stdin.close(); assert enc.wait() == 0
    said: list[str] = []
    rec = hr.run(str(mesh), str(pts), str(source), 0, N, folder / "head.mp4", "the check", room=ROOM, mouth="auto", classes=str(classes),
                 classes_first=-pad, say=said.append)
    m = rec["drawn"]["mouth"]
    rows = {r["frame"]: r for r in m["gate"]["a_frame_at_a_time"]}
    # THE CASE THAT GOES RED when, in the lines' form, a frame whose lines miss loses a mouth the class map places on her face (2823's
    # open mouth), or keeps one the gate drops: far with the body model under it (9), in the air (10-11), or with no lips (8)
    assert m["drawn_as"] == "the lip lines" and m["frames_by_the_lip_lines"] == [[0, 5]] and m["frames_by_the_class_map"] == [[6, 7]], m
    assert m["no_mouth_on"] == {hr.NO_MOUTH_CLASS_GATE: [[8, 11]]} and "mouth" not in rec["drawn"]["parts_left_out"], m
    assert rows[8] == {"frame": 8, "lips": False, "drawn": False} and rows[9]["on_body"] == 1.0 and rows[9]["from_the_pose_mouth_in_face_heights"] > 0.4 \
        and rows[10]["on_body"] == 0.0 and rows[10]["from_the_pose_mouth_in_face_heights"] < 0.05, rows
    rw, rh = rec["size"]
    x0, y0, x1, y1 = rec["box"]

    def white_near(f: int, shift: float = 0.0) -> int:
        cx, cy = P["keypoints"][f][lips].mean(0) + shift
        pic = next(dfp.stream(str(folder / "head.mp4"), (rw, rh), f, 1, pix="rgb24"))
        ux, uy, r = int(round((cx - x0) * rw / (x1 - x0))), int(round((cy - y0) * rh / (y1 - y0))), int(30 * rh / (y1 - y0))
        return int((pic[max(uy - r, 0):uy + r, max(ux - r, 0):ux + r].max(-1) > 200).sum())
    # white at the mouth where a mouth is drawn (the lines on 3, the class map's outline on 6), none at the far lips on 9
    assert white_near(3) > 0 and white_near(6) > 0 and white_near(9, 40) == 0, (white_near(3), white_near(6), white_near(9, 40))

    def on_the_ring(f: int) -> int:
        """How many of the outer-lip ring's points carry white within a pixel: the lip lines drawn there."""
        pic = next(dfp.stream(str(folder / "head.mp4"), (rw, rh), f, 1, pix="rgb24")).max(-1) > 200
        n = 0
        for x, y in P["keypoints"][f][lips]:
            ux, uy = int(round((x - x0) * rw / (x1 - x0))), int(round((y - y0) * rh / (y1 - y0)))
            n += bool(pic[max(uy - 1, 0):uy + 2, max(ux - 1, 0):ux + 2].any())
        return n
    # THE CASE THAT GOES RED when a frame given the class map's own mouth keeps the points' lip rings as well (two mouths)
    assert on_the_ring(3) > len(lips) // 2 and on_the_ring(6) == 0, (on_the_ring(3), on_the_ring(6))
    assert any("the lip lines on 6 of 12 frames" in x and "the class map's own mouth on 2 (" in x for x in said), said
    return ("in the lip lines' form, a frame at a time: the lines where they lie on the class map's mouth, the class map's own mouth where they "
            "miss and its gate passes (2823's open mouth), no mouth where it holds no lips, lies far from the pose mouth or over no body model; "
            "the class map read from its own first frame")


case("one run makes the form", one_run_makes_the_form)
case("a face off the subject's track fails the run", a_face_off_the_track_fails)
case("her own glasses over her eyes, off her track", her_glasses_over_her_eyes)
case("no face lines where the part is her hair alone", no_face_where_hair_alone)
case("frames from the middle of all three files", frames_from_the_middle)
case("the mouth by the check", the_mouth_by_the_check)
case("the class map's mouth by its own gate", the_class_map_mouth_by_its_gate)
case("a frame at a time in the lip lines' form", a_frame_at_a_time_in_the_lines_form)
case("the gate at its bounds", the_gate_at_its_bounds)
case("the mouth over time", the_mouth_over_time)
case("an option: no class-map mouth without teeth or tongue", needs_hole_option)
case("no clip-level gate, and the head's box before the pieces", no_clip_gate_and_the_box_first)
case("it writes over nothing, and nothing when it cannot finish", it_writes_over_nothing)
case("the form's defaults can be set aside, and the record says so", the_defaults_can_be_set_aside)
case("a step that fails fails the run", a_step_that_fails_fails_the_run)
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(finish())
