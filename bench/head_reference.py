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

**The mouth, by a check** (`--mouth auto`, the chain's default since 2026-10-14): drawn when the drawn outer-lip ring's points
lie on the subject's mouth classes in the preview's class map on enough frames (`MOUTH_CHECK`, starting values), left out
otherwise, the figures in the record (`drawn.mouth`). Where the lines fail and the class map holds the subject's lips on
enough frames (`CLASS_MOUTH_OF_FRAMES`), the mouth is drawn from the class map itself, cut by the same box: the body model's
sculpted lips painted out to the grey round them, the lips' outline and inner edge white, the open inside dark
(`drawn.mouth.drawn_as`: "the class map"; the lead's ruling of 2026-10-14 on 0294d). Where the lines pass, they are drawn
over the class map's painted-out lips and dark inside on each frame where they lie on its mouth (`drawn.mouth.inside`). `--mouth drawn|left out` is a person's
word with `--mouth-why`. NO MOUTH on a frame whose lip lines lie on the class map's mouth under `MOUTH_CHECK["on_share"]`, whichever form it is
drawn in, nor on the frames `--no-mouth-on` names (the job's frames whose part holds no lips, hair alone): the body model's own lips stay, and
the record says on which frames and by which rule (`drawn.mouth.no_mouth_on`; the lead's ruling of 2026-10-14 on 630b's profile, 805b and
1010d's chest).
Without `--mouth` the tool leaves the mouth out as before (`FORM_LEAVES_OUT`), unless `--leave-out` says otherwise.

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
#: The mouth drawn when the drawn lip lines lie on the subject's mouth (`mouth_check`, `--mouth auto`, the chain's default
#: since 2026-10-14): a frame passes when at least `on_share` of the outer-lip ring's points lie on the mouth classes of the
#: preview's class map (Lower_Lip, Upper_Lip, the teeth, Tongue) grown `reach_px`; the mouth is drawn when `of_frames` of the
#: span's frames pass. STARTING VALUES, ruled by the lead that day, printed with their figures: on 1730, 1156 and 484c the lines
#: lay on the mouth (18 of 25, 21 of 33, 32 of 32 frames at 0.8); on 0294d they did not (0 of 120), the pose points bunched on
#: her upper lip in near-profile under a hat while her mouth was open (frame 375).
MOUTH_CHECK = dict(on_share=0.8, of_frames=0.5, reach_px=3)
#: Where the lip lines fail the check, the mouth drawn from the class map itself (`class_mouths`; the lead's ruling of 2026-10-14
#: on mrdrill's read of 0294d, where the class map held her open mouth on 370, 375 and 380 and the points did not) when the
#: class map holds the subject's lips on at least this share of the span's frames. A STARTING VALUE, reasoned as the check's
#: own half: 0294d holds them on 113 of 120.
CLASS_MOUTH_OF_FRAMES = 0.5
#: The mouth's forms: drawn whatever the check says, left out, or by the check.
MOUTH_FORMS = ("auto", "drawn", "left out")
TOOLS = ("bench/head_reference.py", "bench/crop_reference_to_head.py", "bench/draw_face_points.py")


def cap_probe(path: str) -> tuple[int, int, int]:
    """(width, height, frames) of a video, by the capture tool's own probe."""
    import capture_masked_run as cap
    return cap.probe(path)


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


def mouth_check(points: str, classes: str, classes_first: int, first: int, count: int) -> dict:
    """On each frame of the span, the share of the outer-lip ring's points on the subject's mouth classes in a class map (its
    frame 0 is source frame `classes_first`), grown `MOUTH_CHECK["reach_px"]`; and whether the mouth is to be drawn."""
    import cv2
    import numpy as np
    import capture_masked_run as cap
    from _lib import face_points as fp
    names = cap._pack("sapiens2_parts").CLASS_NAMES
    mouth = [names.index(c) for c in ("Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue")]
    P = fp.load(points, None)
    idx = P["index"]
    outer = [i for k, i in idx.items() if k.endswith("_of_upper_outer_lip") or k.endswith("_of_lower_outer_lip") or k in ("r_outer_corner_of_mouth", "l_outer_corner_of_mouth")]
    w, h, _ = cap.probe(classes)
    cm = cap.read_classes(classes, (w, h), classes_first, first, count)
    r = int(MOUTH_CHECK["reach_px"])
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    shares = []
    for f in range(count):
        m = cv2.dilate(np.isin(cm[f], mouth).astype(np.uint8), disc).astype(bool)
        xy = P["keypoints"][first - P["first"] + f][outer]
        x, y = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1), np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
        shares.append(float(m[y, x].mean()))
    s = np.array(shares)
    on = int((s >= MOUTH_CHECK["on_share"]).sum())
    return {"what": "the share of the drawn outer-lip ring's points on the mouth classes of the class map, a frame at a time", "class_map": str(classes),
            "share_least_median_most": [round(float(s.min()), 2), round(float(np.median(s)), 2), round(float(s.max()), 2)],
            "frames_at_or_over_the_share": on, "frames": count, "rule": dict(MOUTH_CHECK), "rule_is": "starting values, the lead's ruling of 2026-10-14",
            "drawn": bool(on >= MOUTH_CHECK["of_frames"] * count), "_shares": shares}


#: Why a frame has no mouth (`no_mouth_frames`), as the record and the run say it.
NO_MOUTH_UNDER_THE_SHARE = (f"the lip lines lie on the class map's mouth under {MOUTH_CHECK['on_share']:g} (MOUTH_CHECK on_share): the lips are not on her "
                            "mouth there, and whatever is drawn of a mouth lands off her face (630b's profile, 805b)")
NO_MOUTH_PART = "the load's part holds no lips there (--no-mouth-on, the job's parts_on: {why})"


def no_mouth_frames(shares: list[float] | None, first: int, count: int, spans: list[tuple[int, int, str]]) -> dict[str, list[int]]:
    """The source frames that get no mouth at all, by rule: a frame whose lip-ring share is under `MOUTH_CHECK["on_share"]` (with
    `--mouth auto`, so `shares` is the check's), whichever form the mouth is drawn in; and the frames of `spans` (`--no-mouth-on`),
    the load's frames whose part holds no lips (hair alone). The body model's own lips stay on them."""
    out: dict[str, list[int]] = {}
    if shares is not None:
        under = [first + f for f, v in enumerate(shares) if v < MOUTH_CHECK["on_share"]]
        if under:
            out[NO_MOUTH_UNDER_THE_SHARE] = under
    for a, b, why in spans:
        fs = [f for f in range(max(a, first), min(b, first + count - 1) + 1)]
        if fs:
            out.setdefault(NO_MOUTH_PART.format(why=why), []).extend(fs)
    return out


def spans_of(frames: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def class_mouths(classes: str, classes_first: int, first: int, count: int, box: list[int], size: tuple[int, int]) -> tuple[list, dict]:
    """The subject's mouth from a class map, a frame at a time, cut by the reference's box and brought to its size: (whole, hole)
    masks, the whole being the lips and what they hold (the teeth, Tongue) filled, its largest piece, and the hole its inside
    with the lips taken out; None on a frame with no lips. With the figures: frames held, frames open, the open area."""
    import cv2
    import numpy as np
    from scipy import ndimage
    import capture_masked_run as cap
    names = cap._pack("sapiens2_parts").CLASS_NAMES
    lips_of = [names.index(c) for c in ("Upper_Lip", "Lower_Lip")]
    held_in = [names.index(c) for c in ("Upper_Teeth", "Lower_Teeth", "Tongue")]
    w, h, _ = cap.probe(classes)
    cm = cap.read_classes(classes, (w, h), classes_first, first, count)
    x0, y0, x1, y1 = box
    out, open_px = [], []
    for f in range(count):
        lips = np.isin(cm[f], lips_of)
        if not lips.any():
            out.append(None); open_px.append(0)
            continue
        whole = ndimage.binary_fill_holes(lips | np.isin(cm[f], held_in))
        parts, n = ndimage.label(whole)
        if n > 1:
            whole = parts == 1 + int(np.argmax(ndimage.sum(whole, parts, range(1, n + 1))))
        hole = whole & ~lips
        open_px.append(int(hole.sum()))
        cut = lambda m: cv2.resize(m.astype(np.uint8)[max(y0, 0):y1, max(x0, 0):x1], size, interpolation=cv2.INTER_NEAREST).astype(bool)  # noqa: E731
        out.append((cut(whole), cut(hole)))
    held = sum(m is not None for m in out)
    a = np.array(open_px)
    return out, {"what": "the subject's lips and what they hold in the class map, cut by the reference's box", "class_map": str(classes),
                 "frames_holding_the_lips": held, "frames": count, "frames_open": int((a > 0).sum()),
                 "open_px_on_the_canvas_most": [int(a.max()), first + int(a.argmax())] if a.any() else None,
                 "held_of_frames_at_least": CLASS_MOUTH_OF_FRAMES, "that_is": "a starting value, the check's own half",
                 "held": bool(held >= CLASS_MOUTH_OF_FRAMES * count)}


def run(mesh: str, points: str, source: str, first: int, count: int, out: Path, by: str, *, names: list[str] | None = None,
        mesh_first: int | None = None, points_first: int | None = None, width: float = FORM_WIDTH, room: float | None = None,
        leave_out: frozenset = frozenset({FORM_LEAVES_OUT}), dots: bool = False, iris: str = faces.IRIS_FORMS[0], repo: Path = REPO,
        mouth: str | None = None, mouth_why: str = "", classes: str | None = None, classes_first: int | None = None,
        no_mouth_on: list[tuple[int, int, str]] | None = None, say=print) -> dict:
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
    # THE MOUTH (`mouth`, the chain's default `auto` since 2026-10-14): by the check, or drawn or left out on a person's word
    mouth_said = None
    if mouth is not None:
        if mouth not in MOUTH_FORMS:
            raise SystemExit(f"--mouth is one of {MOUTH_FORMS}")
        if mouth == "auto":
            if not classes or classes_first is None:
                raise SystemExit("--mouth auto checks the lip lines against the subject's class map: give --classes and --classes-first")
            check = mouth_check(points, classes, classes_first, first, count)
            shares = check.pop("_shares")
            mouth_said = {"form": "auto", "check": check, "drawn": check["drawn"], **({"drawn_as": "the lip lines"} if check["drawn"] else {})}
            say(f"    the mouth: {'DRAWN' if check['drawn'] else 'NOT DRAWN FROM THE POINTS'} by the check: the lip lines on the mouth classes at "
                f"{MOUTH_CHECK['on_share']:g} or more on {check['frames_at_or_over_the_share']} of {count} frames (half is {MOUTH_CHECK['of_frames'] * count:g}); "
                f"share least/median/most {check['share_least_median_most']}")
        else:
            if not str(mouth_why or "").strip():
                raise SystemExit(f"--mouth {mouth} is a person's word: say why with --mouth-why")
            mouth_said = {"form": mouth, "why": " ".join(mouth_why.split()), "drawn": mouth == "drawn"}
        by_points = mouth_said["drawn"] and mouth_said.get("drawn_as") != "the class map"
        leave_out = (frozenset(leave_out) - {"mouth"}) if by_points else (frozenset(leave_out) | {"mouth"})
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
    # THE CLASS MAP'S MOUTH, cut by the box just made. Where the lip lines pass, it gives their inside on the frames they lie on
    # it (the lead's ruling of 2026-10-14, the outside reviewer's: a white outline over the body model's sculpted, closed lips is
    # a hollow mouth); where they fail, the mouth itself (the lead's ruling of 2026-10-14 on 0294d)
    mouths, lines = None, True
    if mouth_said and mouth_said["form"] == "auto":
        cw, ch, _ = cap_probe(str(made["plain_cut"]))
        mouths, held = class_mouths(classes, classes_first, first, count, cut["box"], (cw, ch))
        mouth_said["class_map"] = held
    if mouths is not None and mouth_said["drawn"]:
        lines = False
        mouths = [m if shares[f] >= MOUTH_CHECK["on_share"] else None for f, m in enumerate(mouths)]
        agree = sum(m is not None for m in mouths)
        mouth_said["inside"] = "the class map"
        mouth_said["inside_on_frames"] = agree
        say(f"    the mouth's inside from the class map on the {agree} of {count} frames the lip lines lie on it: the sculpted lips painted "
            "out, the inside dark where it is open")
    elif mouths is not None:
        if held["held"]:
            mouth_said.update(drawn=True, drawn_as="the class map", inside="the class map")
            say(f"    the mouth: DRAWN FROM THE CLASS MAP, which holds her lips on {held['frames_holding_the_lips']} of {count} frames "
                f"(at least {CLASS_MOUTH_OF_FRAMES:g} of them), open on {held['frames_open']}: the sculpted lips painted out, the outline white, "
                "the inside dark")
        else:
            mouths = None
            say(f"    the mouth: LEFT OUT: the class map holds her lips on {held['frames_holding_the_lips']} of {count} frames, under "
                f"{CLASS_MOUTH_OF_FRAMES:g} of them")
    # NO MOUTH ON A FRAME where the lips do not lie on her mouth, or where the load's part holds no lips (the lead's ruling of
    # 2026-10-14 on the blind readers of 630b, 805b and 1010d: the mouth drawn off her face in profile and on her chest)
    bare = no_mouth_frames(shares if mouth_said and mouth_said["form"] == "auto" else None, first, count, no_mouth_on or [])
    if mouth_said and mouth_said["drawn"] and bare:
        mouth_said["no_mouth_on"] = {rule: spans_of(fs) for rule, fs in bare.items()}
        mouth_said["frames_with_no_mouth"] = len({f for fs in bare.values() for f in fs})
        for rule, fs in bare.items():
            say(f"    NO MOUTH on {len(fs)} of {count} frames ({', '.join(f'{a}-{b}' if a != b else str(a) for a, b in spans_of(fs))}): {rule}")
    no_mouth = sorted({f for fs in bare.values() for f in fs}) if mouth_said and mouth_said["drawn"] else []
    say(f"3 of 3: the face drawn on the cut body model, {width:g} px, {'a dot at every point' if dots else 'no dots'}"
        + (f", left out: {', '.join(sorted(leave_out))}" if leave_out else "")
        + ((", the mouth from the class map" if lines else ", the mouth's inside from the class map") if mouths else ""))
    drawn = faces.run(str(made["plain_cut"]), str(made["plain_cut_points"]), first, count, out, by, width=width, dots=dots, leave_out=leave_out,
                      source=str(made["source_cut"]), source_first=first, iris=iris, mouths=mouths, mouth_lines=lines, no_mouth=no_mouth, say=quiet)
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
                  "parts_left_out": drawn["parts_left_out"], "eye_rings": drawn["eye_rings"], "iris": iris, "frames_left_bare": drawn["frames_left_bare"],
                  **({"mouth": mouth_said} if mouth_said else {})},
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


def no_mouth_arg(text: str) -> tuple[int, int, str]:
    rng, sep, why = text.partition("=")
    try:
        a, b = (int(x) for x in (rng.split("-") * 2)[:2])
    except ValueError:
        raise SystemExit(f"--no-mouth-on {text!r} is not FIRST-LAST=WHY") from None
    if not sep or not why.strip() or a > b:
        raise SystemExit(f"--no-mouth-on {text!r} is not FIRST-LAST=WHY, with the why")
    return a, b, " ".join(why.split())


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
                   help="an eye ring drawn only between the lids (the default since 2026-10-14), whole (the form before), the iris filled: a dark disc inside the lids, or contrast: the sclera solid white and the iris solid black inside the lids (until a render decides)")
    a.add_argument("--mouth", choices=MOUTH_FORMS, help="the mouth: `auto` draws it when the lip lines lie on the class map's mouth (MOUTH_CHECK; the "
                   "chain's default), `drawn` or `left out` on a person's word with --mouth-why; --leave-out decides when left out")
    a.add_argument("--mouth-why", default="", help="why the mouth is drawn or left out whatever the check says")
    a.add_argument("--classes", help="the subject's class map from the preview, for --mouth auto")
    a.add_argument("--classes-first", type=int, help="the source frame of the class map's frame 0")
    a.add_argument("--no-mouth-on", action="append", default=[], metavar="FIRST-LAST=WHY",
                   help="source frames on which no mouth is drawn: the load's part holds no lips there (the job's parts_on, hair alone); repeatable")
    a.add_argument("--names", help="a json list of the points' names, for a points file made before they were kept")
    a.add_argument("--mesh-first", type=int, help="the source frame of the body-model video's frame 0; the span's first when left out")
    a.add_argument("--points-first", type=int, help="the source frame of the points file's row 0; the file's own when it says")
    v = a.parse_args()
    names = json.loads(Path(v.names).read_text()) if v.names else None
    rec = run(v.mesh, v.points, v.source, v.first, v.count, Path(v.out), v.by, names=names, mesh_first=v.mesh_first, points_first=v.points_first,
              width=v.width, room=v.room, leave_out=faces.parts_arg(v.leave_out), dots=v.dots, iris=v.iris, mouth=v.mouth, mouth_why=v.mouth_why,
              classes=v.classes, classes_first=v.classes_first, no_mouth_on=[no_mouth_arg(x) for x in v.no_mouth_on])
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
