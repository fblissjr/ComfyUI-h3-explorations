#!/usr/bin/env python3
"""The reference video of a head replacement, in one command: the body model cut to the head, with the face drawn on it.

    python bench/head_reference.py --mesh PLAIN_MESH.mp4 --points SHOT_points.npz --source SOURCE.mov \\
        --first 1222 --count 84 --out <share>/Video/<video>/<who>/<shot>_inputs/<name>.mp4 --by <session> \\
        [--width 1.5] [--room 1.2] [--leave-out mouth] [--dots] [--mesh-first N] [--points-first N] [--names NAMES.json]

**The form** (the owner, 2026-10-12, 20:12Z; `docs/wiki/decisions.md`): every head or head-and-hair replacement
takes as its motion video the body model cut to the head with the face points drawn on it, and its text says
what that video is. Made by hand it was three commands, each with the others' outputs for arguments. This runs
the three and writes one record:

1. `bench/crop_reference_to_head.py` on the PLAIN body-model video and the shot's points: one box round the head
   for the whole span, enlarged to the canvas, and the points carried into the new picture.
2. The same on the SOURCE itself (`--fit`), by the same box, for the sheet and for a strip.
3. `bench/draw_face_points.py` on the cut body model with the carried points: lines only (no dots) and the mouth
   left out, unless told otherwise, with the cut source under it on the sheet.

It makes no decision of its own: the box, the read-backs and the sheet are those tools', and their docstrings
say what each can and cannot know. It adds the order, the names and the record.

**The defaults are the first references of this form, not a rule.** The width (`FORM_WIDTH`), no dots and the
mouth left out are what read as eyes on the first shot drawn this way, by one look. On that shot the lips'
points sat above an open mouth, which is why the mouth is out; a shot whose lips' points are on her lips can
have it back with `--leave-out ""`. Look at the sheet.

**What it writes**, all beside `--out`, with `<stem>` the name of `--out` without `.mp4`:

    <stem>.mp4                 the reference: frame 0 is source frame --first, at the canvas's size
    <stem>.json                ONE record: every input by path, the box and scale, what was drawn, the three
                               read-backs, who made it and when, and the commit it ran on with the last commit
                               of each of the two tools (and whether a tool's file was changed since)
    <stem>_sheet.jpg           six frames: the cut source, the drawn body model, the two blended
    <stem>_plain_cut.mp4       the cut body model with nothing drawn, and its record, points and sheet
    <stem>_source_cut.mp4      the source cut by the same box, and its record and sheet

It refuses to write over any of them: a reference made again takes a new name.

**A reference drawn from another person's face is a failure.** The points file's own record
(`bench/pose_along_shot.py`) says, a frame at a time, whether the face's middle lies on the subject's tracked
mask. A reference whose span has a frame where it does not is written, so its sheet can be read, and marked
FAILED with those frames (`face_on_the_track`), but for one case: a run of such frames at most `FILL_GAP` long with
frames on the track just before and after it takes its face points from those two, interpolated, before anything is
cut (`fill_from_neighbours`); the record names each frame so filled and from which, and calls them filled, not read. On the lead's 3167-3213 (2026-10-14) eleven frames drew the face
of a woman above her and the run said PASSED. A points file whose record says nothing of a track (made before
then, or read on the whole frame) is said as not known and does not fail.

A passing run says each step's file holds what that step made, on the frames it was told, and that the face it
drew lies on the subject's track on every frame. It does not say the track is the right person or that the lines
are on her face: open the sheet.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO  # noqa: E402

import crop_reference_to_head as crop  # noqa: E402
import draw_face_points as faces  # noqa: E402

#: The lines' width in pixels of the cut picture. Inherited: the first references of this form (the second
#: subject's short kitchen shot and the lead's close-up, 2026-10-12), where it was one try by eye, with no dots.
#: The drawing tool's own rule (the face's height over a constant) gives a wider line on a face this large, and
#: on the first shot a wider line with dots filled the eyes in.
FORM_WIDTH = 1.5
#: What the form leaves out unless told otherwise. Inherited: the same references; the lips' points were off the
#: mouth on the first shot.
FORM_LEAVES_OUT = "mouth"
#: The longest run of frames whose face's middle is off the subject's track that takes its face points from the frames
#: on either side, interpolated (`fill_from_neighbours`). A STARTING VALUE, reasoned: one latent step of the video
#: model; asked for by the lead on 2026-10-14 for the second subject's 604-948, where the lead stands just behind her
#: shoulder and the pose pass took the lead's face on 895 and 897 (a frame on track between them). A longer run fails, as before.
FILL_GAP = 4
TOOLS = ("bench/head_reference.py", "bench/crop_reference_to_head.py", "bench/draw_face_points.py")


def _git(*args: str, repo: Path = REPO) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()


def ran_on(repo: Path = REPO) -> dict:
    """The commit the tree is at, and for each tool the last commit that changed it and whether its file differs from that."""
    return {"commit": _git("rev-parse", "--short", "HEAD", repo=repo) or "not known: no git here",
            "tools": {name: {"last_changed_in": _git("log", "-1", "--format=%h", "--", name, repo=repo) or "never committed",
                             "changed_since": bool(_git("status", "--porcelain", "--", name, repo=repo))} for name in TOOLS}}


def outputs(out: Path) -> dict[str, Path]:
    """Every file a run writes, by what it is."""
    stem = out.with_suffix("")
    plain, source = Path(f"{stem}_plain_cut.mp4"), Path(f"{stem}_source_cut.mp4")
    made = {"reference": out, "record": out.with_suffix(".json"), "sheet": Path(f"{stem}_sheet.jpg"), "plain_cut": plain, "source_cut": source,
            "filled_points": Path(f"{stem}_points_filled.npz")}
    for key, video in (("plain_cut", plain), ("source_cut", source)):
        made.update({f"{key}_record": video.with_suffix(".json"), f"{key}_points": video.with_name(video.stem + "_points.npz"),
                     f"{key}_sheet": video.with_name(video.stem + "_sheet.jpg")})
    return made


def face_on_the_track(points: str, first: int, count: int) -> dict:
    """What the points file's own record says of the face on the subject's track, on the span's frames."""
    rec = Path(points).with_suffix(".json")
    said = json.loads(rec.read_text()).get("face_on_the_mask") if rec.is_file() else None
    if not said:
        return {"known": False, "why": f"{rec.name} " + ("says nothing of a track" if rec.is_file() else "is not there"),
                "frames_whose_face_middle_is_off_the_track": []}
    return {"known": True, "read_from": rec.name, "what": said["what"],
            "frames_whose_face_middle_is_off_the_track": [f for f in said["frames_whose_face_middle_is_off_the_mask"] if first <= f < first + count]}


def fill_from_neighbours(off: list[int], first: int, count: int, gap: int = FILL_GAP) -> tuple[dict[int, tuple[int, int]], list[int]]:
    """Which off-track frames take their points from the frames on either side: ({frame: (before, after)}, the frames
    that do not). A run fills when it is at most `gap` frames long and the frames just before and just after it are in
    the span and on the track."""
    bad = set(off)
    filled, left = {}, []
    runs, run = [], []
    for f in sorted(bad):
        if run and f != run[-1] + 1:
            runs.append(run)
            run = []
        run.append(f)
    if run:
        runs.append(run)
    for r in runs:
        before, after = r[0] - 1, r[-1] + 1
        if len(r) <= gap and before >= first and after <= first + count - 1 and before not in bad and after not in bad:
            filled.update({f: (before, after) for f in r})
        else:
            left += r
    return filled, left


def write_filled(points: str, filled: dict[int, tuple[int, int]], out: Path) -> None:
    """A copy of a points file with each filled frame's points the straight line between its two neighbours' (the
    score of a point the lesser of theirs), and a record beside it saying which, from which."""
    import numpy as np
    z = np.load(points)
    kp, sc = z["keypoints"].copy(), z["scores"].copy()
    at = int(z["first_frame"])
    for f, (a, b) in filled.items():
        t = (f - a) / (b - a)
        kp[f - at] = (1 - t) * kp[a - at] + t * kp[b - at]
        sc[f - at] = np.minimum(sc[a - at], sc[b - at])
    np.savez_compressed(out, **{k: z[k] for k in z.files if k not in ("keypoints", "scores")}, keypoints=kp, scores=sc)
    rec = Path(points).with_suffix(".json")
    said = json.loads(rec.read_text()) if rec.is_file() else {}
    out.with_suffix(".json").write_text(json.dumps({**said, "filled_from_neighbours": {str(f): list(v) for f, v in sorted(filled.items())},
                                                    "filled_from": str(points), "filled_is": "each filled frame's points the straight line between its "
                                                    "neighbours' (bench/head_reference.py, FILL_GAP)"}, indent=1) + "\n")


def run(mesh: str, points: str, source: str, first: int, count: int, out: Path, by: str, *, names: list[str] | None = None,
        mesh_first: int | None = None, points_first: int | None = None, width: float = FORM_WIDTH, room: float | None = None,
        leave_out: frozenset = frozenset({FORM_LEAVES_OUT}), dots: bool = False, iris: str = faces.IRIS_FORMS[0], repo: Path = REPO, say=print) -> dict:
    """Cut, cut, draw, record. Returns the record; `passed` is True only when all three steps passed."""
    if out.suffix != ".mp4":
        raise SystemExit(f"--out {out} must end in .mp4: the other files are named from it")
    made = outputs(out)
    there = [str(p) for p in made.values() if p.exists()]
    if there:
        raise SystemExit(f"{len(there)} of the files this would write exist ({there[0]}): a reference made again takes a new name")
    for what, path in (("--mesh", mesh), ("--points", points), ("--source", source)):
        if not Path(path).is_file():
            raise SystemExit(f"{what} {path} is not a file")
    faces.rows_of(points, names, first, count, points_first)        # the span and the names, before anything is written
    quiet = lambda *_: None  # noqa: E731
    # A SHORT RUN OF FRAMES OFF THE TRACK takes its points from its neighbours, before anything is cut: another person's face
    # on a frame would move the box as well as draw them (the lead's ruling, 2026-10-14, for the second subject's 895 and 897)
    on_track = face_on_the_track(points, first, count)
    filled, unfilled = fill_from_neighbours(on_track["frames_whose_face_middle_is_off_the_track"], first, count)
    read_from = points
    if filled:
        write_filled(points, filled, made["filled_points"])
        points = str(made["filled_points"])
        say(f"    {len(filled)} frame(s) off the subject's track take their face points from the frames either side: "
            + ", ".join(f"{f} from {a} and {b}" for f, (a, b) in sorted(filled.items())))
    say(f"1 of 3: the plain body model cut to the head, frames {first} to {first + count - 1}")
    cut = crop.run(mesh, points, first, count, made["plain_cut"], by, names=names, video_first=mesh_first, points_first=points_first,
                   room=room, say=quiet)
    say(f"    box {cut['box']}, enlarged {cut['scale'][0]:.3f} times: the face is {cut['face_height_px_in_this_video']:.0f} px tall; "
        + ("PASSED" if cut["passed"] else "FAILED"))
    say("2 of 3: the source cut by the same box")
    under = crop.run(source, points, first, count, made["source_cut"], by, names=names, points_first=points_first, room=room, fit=True, say=quiet)
    if under["box"] != cut["box"]:
        raise SystemExit(f"the source was cut by {under['box']} and the body model by {cut['box']}: one box is the form")
    say("    " + ("PASSED" if under["passed"] else "FAILED"))
    say(f"3 of 3: the face drawn on the cut body model, {width:g} px, {'a dot at every point' if dots else 'no dots'}"
        + (f", left out: {', '.join(sorted(leave_out))}" if leave_out else ""))
    drawn = faces.run(str(made["plain_cut"]), str(made["plain_cut_points"]), first, count, out, by, width=width, dots=dots, leave_out=leave_out,
                      source=str(made["source_cut"]), source_first=first, iris=iris, say=quiet)
    fit = drawn["read_back"]["white_from_the_lines_px_mean_by_frames_of_slip"]
    say(f"    white {fit['0']} px from its own frame's lines, {fit['1']} and {fit['-1']} from the neighbours'; "
        + ("PASSED" if drawn["passed"] else "FAILED"))
    say("    " + faces.marks.against_the_source_line(drawn["body_model_against_the_source"]))
    on_track = {**on_track, "frames_filled_from_their_neighbours": {str(f): list(v) for f, v in sorted(filled.items())},
                "frames_off_the_track_not_filled": unfilled, "fill_gap_frames": FILL_GAP,
                "filled_are": ("filled, not read: their points are the straight line between their neighbours', and the read-back reads the drawing "
                               "from those points, so it says nothing of the person there") if filled else None}
    if unfilled:
        say(f"    THE FACE IS OFF THE SUBJECT'S TRACK on {len(unfilled)} of {count} frames ({', '.join(map(str, unfilled))}), in runs longer than "
            f"{FILL_GAP} frames or at the span's edge: another person's face is drawn there; FAILED")
    elif filled:
        say(f"    the face's middle is on the subject's track on every frame but the {len(filled)} filled, not read")
    elif on_track["known"]:
        say(f"    the face's middle is on the subject's track on all {count} frames")
    else:
        say(f"    whether the face is on the subject's track is NOT KNOWN: {on_track['why']}")
    passed = bool(cut["passed"] and under["passed"] and drawn["passed"] and not unfilled)
    record = {
        "video": out.name, "what": "the reference of a head replacement: the body model cut to the head, with the face drawn on it",
        "made_by": by, "made_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frames": [first, first + count - 1], "its_first_frame_is_source_frame": first, "size": drawn["size"],
        "inputs": {"body_model_video": str(mesh), "its_first_frame_is_source_frame": cut["made_from"]["its_first_frame_is_source_frame"],
                   "points": str(read_from), **({"points_filled": str(points)} if filled else {}),
                   "their_first_row_is_source_frame": cut["made_from"]["their_first_row_is_source_frame"],
                   "the_points_own_record_says": cut["made_from"]["the_points_own_record_says"],
                   "source": str(source), "the_source_was": under["made_from"]["fitted"]},
        "ran_on": ran_on(repo),
        "box": cut["box"], "box_is": cut["box_is"], "box_set_by": cut["box_set_by"], "scale": cut["scale"],
        "a_point_of_the_canvas_lands_at": cut["a_point_of_the_canvas_lands_at"],
        "face_height_px": {"on_the_canvas": cut["face_height_px_median"], "in_this_video": cut["face_height_px_in_this_video"]},
        "drawn": {"line_width_px": drawn["line_width_px"], "width_set_by": "--width" if width != FORM_WIDTH else "FORM_WIDTH, the first references' own",
                  "a_dot_at_every_point": drawn["a_dot_at_every_point"], "points_drawn_alone": drawn["points_drawn_alone"],
                  "parts_left_out": drawn["parts_left_out"], "eye_rings": drawn["eye_rings"], "iris": iris, "frames_left_bare": drawn["frames_left_bare"]},
        "read_backs": {"the_cut_body_model": cut["read_back"], "the_cut_source": under["read_back"], "the_drawing": drawn["read_back"]},
        "body_model_against_the_source": drawn["body_model_against_the_source"],
        "face_on_the_track": on_track,
        "sheet": drawn.get("sheet"),
        "files": {key: path.name for key, path in made.items() if path.exists() or key == "record"},
        "not_known": ["whether the track is the right person and the lines are on her face: open the sheet",
                      "what a model makes of a reference that no longer lies over the source pixel for pixel: the load's text has to say it is a closer view"],
        "passed": passed,
    }
    made["record"].write_text(json.dumps(record, indent=1) + "\n")   # over the drawing tool's own record of the same name: it is under read_backs and drawn
    say(("PASSED" if passed else "FAILED") + f": {out}")
    say(f"the record: {made['record'].name}; the sheet to look at before it is used: {made['sheet'].name}")
    say(f"in a load: \"motion\": \"mesh\", \"motion_file\": \"{out}\", \"motion_first\": {first}")
    return record


def main() -> int:
    a = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    a.add_argument("--mesh", required=True, help="the PLAIN body-model video, at the canvas's size, nothing drawn on it")
    a.add_argument("--points", required=True, help="the original's points along the shot (bench/pose_along_shot.py)")
    a.add_argument("--source", required=True, help="the original video itself; its frame 0 is source frame 0")
    a.add_argument("--first", type=int, required=True, help="the span's first source frame")
    a.add_argument("--count", type=int, required=True)
    a.add_argument("--out", required=True, help="the reference to write (.mp4); every other file is named from it and goes beside it")
    a.add_argument("--by", required=True, help="who is making it")
    a.add_argument("--width", type=float, default=FORM_WIDTH, help=f"the lines' width in pixels of the cut picture; {FORM_WIDTH:g} when left out")
    a.add_argument("--room", type=float, help="room round the faces in face heights; the crop tool's own when left out")
    a.add_argument("--leave-out", default=FORM_LEAVES_OUT, help=f"parts not drawn, any of {', '.join(faces.PARTS)}; `{FORM_LEAVES_OUT}` when left out, \"\" for none")
    a.add_argument("--dots", action="store_true", help="a dot at every point, and the points that belong to no line; lines only when left out")
    a.add_argument("--iris", default=faces.IRIS_FORMS[0], choices=faces.IRIS_FORMS,
                   help="an eye ring drawn only between the lids (the default since 2026-10-14), whole (the form before), or the iris filled: a dark disc inside the lids (until a render decides)")
    a.add_argument("--names", help="a json list of the points' names, for a points file made before they were kept")
    a.add_argument("--mesh-first", type=int, help="the source frame of the body-model video's frame 0; the span's first when left out")
    a.add_argument("--points-first", type=int, help="the source frame of the points file's row 0; the file's own when it says")
    v = a.parse_args()
    names = json.loads(Path(v.names).read_text()) if v.names else None
    rec = run(v.mesh, v.points, v.source, v.first, v.count, Path(v.out), v.by, names=names, mesh_first=v.mesh_first, points_first=v.points_first,
              width=v.width, room=v.room, leave_out=faces.parts_arg(v.leave_out), dots=v.dots, iris=v.iris)
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
