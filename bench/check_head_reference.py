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
    assert all(p.is_file() for p in made.values()), [k for k, p in made.items() if not p.is_file()]
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
    lead's 3167 eleven frames drew another woman's face and the run said PASSED)."""
    seen = {}
    for name, said_of_track in (("off", {"what": "the check's", "frames_whose_face_middle_is_off_the_mask": [3, 4, 7, 40]}),
                                ("on", {"what": "the check's", "frames_whose_face_middle_is_off_the_mask": []}), ("unsaid", None)):
        folder, mesh, pts, source = rooms(f"track_{name}", [90] * N, [60] * N, CENTRES, 0)
        pts.with_suffix(".json").write_text(json.dumps({"made_by": "the check", "face_on_the_mask": said_of_track}))
        said: list[str] = []
        rec = hr.run(str(mesh), str(pts), str(source), 0, N, folder / "head.mp4", "the check", room=ROOM, say=said.append)
        seen[name] = (rec, said, json.loads((folder / "head.json").read_text()), folder)
    rec, said, kept, folder = seen["off"]
    assert rec["passed"] is False and kept["passed"] is False and rec["face_on_the_track"]["frames_whose_face_middle_is_off_the_track"] == [3, 4, 7], rec["face_on_the_track"]
    assert all(r["passed"] for r in rec["read_backs"].values()) and (folder / "head_sheet.jpg").is_file(), "the sheet is not there to be read"
    assert any("THE FACE IS OFF THE SUBJECT'S TRACK on 3 of" in line and "(3, 4, 7)" in line for line in said) and any(line.startswith("FAILED") for line in said), said
    rec, said, _, _ = seen["on"]
    assert rec["passed"] is True and rec["face_on_the_track"]["known"] and any(f"on all {N} frames" in line for line in said), said
    rec, said, _, _ = seen["unsaid"]
    assert rec["passed"] is True and rec["face_on_the_track"]["known"] is False and any("NOT KNOWN" in line and "says nothing of a track" in line for line in said), said
    return ("three frames of the span whose face is off the track fail the run, with every read-back passing and the sheet written to be read, and a frame "
            "outside the span is not counted; a record with none passes; one that says nothing of a track passes and says not known")


case("one run makes the form", one_run_makes_the_form)
case("a face off the subject's track fails the run", a_face_off_the_track_fails)
case("frames from the middle of all three files", frames_from_the_middle)
case("it writes over nothing, and nothing when it cannot finish", it_writes_over_nothing)
case("the form's defaults can be set aside, and the record says so", the_defaults_can_be_set_aside)
case("a step that fails fails the run", a_step_that_fails_fails_the_run)
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(finish())
