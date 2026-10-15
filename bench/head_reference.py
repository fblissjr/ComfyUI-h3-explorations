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
any frame, the span is in the class map's form (`drawn.mouth.drawn_as`: "the class map"; the lead's
ruling of 2026-10-14 on 0294d). The class map's own mouth, on a frame it is drawn, is cut by the same box: the body model's sculpted
lips painted out to the grey round them, the lips' outline and inner edge white, the open inside dark. The lip lines, on a frame they
are drawn, lie over the class map's painted-out lips and dark inside (`drawn.mouth.inside`). Which frame gets which, in either form: `--mouth drawn|left out` is a person's
word with `--mouth-why`. In either form, A FRAME AT A TIME (the lead, 2026-10-15: under the lines' share alone the remade
references had no mouth on 40 of 51, 19 of 29, 22 of 22, 16 of 16 and 61 of 275 frames, 2823's wide-open mouth among them): the lip
lines where they lie on the class map's mouth at `MOUTH_CHECK["on_share"]` or over, with its inside under them; else the class map's
own mouth where it lies on the body model's picture in the cut and near the pose model's mouth (`CLASS_MOUTH_GATE`,
`class_mouth_gate`); else NO MOUTH; and no mouth on the frames `--no-mouth-on` names (the job's frames whose part holds no lips, hair alone, or
a person's word on a look where her mouth is hidden). The body model's own lips stay, and the record says on
which frames and by which rule, with the gate's figures a frame at a time (`drawn.mouth.no_mouth_on`, `drawn.mouth.gate`; the lead's
rulings of 2026-10-14 on 630b's profile, 805b and 1010d's chest). 0294d's piece stands as made (a better form does not remake a kept
piece).
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
#: On a frame whose lip lines lie under `MOUTH_CHECK["on_share"]`, in either form, the class map's own mouth is drawn only when, cut by
#: the reference's box, it lies on the body model's picture in the plain cut (`on_body`: at least that share of it on pixels over
#: `body_grey`; a mouth over black is drawn in the air) and its centre lies within `near` of the pose model's outer-lip ring's centre,
#: in face heights (`_lib.face_points.face_height`; a mouth the part model labelled on her chest is far from it). STARTING VALUES,
#: fitted 2026-10-15 on the five references remade under eed4acb9 against 101 answered frames (the 10/14 blind readers of 630b, 805b
#: and 1010d, and the lead's word on 2823's open mouth; `bench/results/2026-10-15_class_mouth_gate.md`): 6 go against them, all on
#: 805. `on_body` 0.6 draws the fewest mouths a reader faulted (807, 813 and 814, at 0.72 to 0.98 on the body model: 0.76 drops
#: 813 and 814 and leaves out 812, 820, 821, 824 and 828) and leaves out 819, 822 and 823; 0.45 keeps those three and draws 630's 677
#: in the air and 805's 818. Any value over 0.553 and up to 0.62 scores the same; 824 sits at 0.62. `near` decides no answered
#: frame anywhere from 0.25 to 0.45 (the chest frames are 0.47 and more); 0.4 sits above the small faces nobody faulted (0.20 to
#: 0.38). The outside reviewer's parked shape (2026-10-14, 5A) was tried first and does not separate them (the record says how).
CLASS_MOUTH_GATE = dict(on_body=0.6, near=0.4, body_grey=16, agreed_share=0.60, agreed_near=0.20, agreed_on_body=0.47)
#: `agreed_*`: where the pose model and the class map agree on the mouth (the lines' share at `agreed_share` or more and the class
#: map's mouth within `agreed_near` of the pose mouth), `on_body`'s floor is `agreed_on_body`. A STARTING VALUE, the outside
#: reviewer's G6 (2026-10-15): it keeps 805's open three-quarter mouths 819, 822 and 823 (on_body 0.524, 0.499, 0.478), and admits
#: none of 630's 658-680 (share at most 0.40, near at least 0.219) nor 805's 818 (`bench/results/2026-10-15_mouth_runs.md`).
#: THE MOUTH OVER TIME, a frame's form held against its neighbours (mouth v2, the lead's ruling of 2026-10-15, the outside reviewer's
#: G6 section 4): `leave_lines`, the share under which a frame on the lip lines leaves them (it enters at MOUTH_CHECK on_share), so
#: the form does not switch frame by frame as the share jitters round 0.8 (2823: 46 switches and 16 one-frame runs become 8 and 0);
#: `fill_frames` and `fill_on_body`, a run of at most that many frames with no mouth between two with one, each with the class map's
#: mouth at least that much on the body model and within the gate's `near`, takes the class map's mouth (the outside reviewer's case,
#: 805's 822-823 inside one sung vowel under the landed floor; the agreed floor now keeps those, so the fill fills no frame on today's
#: five references and stays as a guard). STARTING VALUES, their figures per reference in `bench/results/2026-10-15_mouth_runs.md`. No rule drops a one-frame mouth: it would drop 805's
#: open 824 to clear 1010's closed 1021.
MOUTH_RUNS = dict(leave_lines=0.6, fill_frames=2, fill_on_body=0.45)
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


#: The classes that sit over the eyes. Where the plan gives one back (the subject's own glasses, a restore span: `--over-the-eyes`),
#: the track may leave the glasses out, and the eyes and brows behind them with it; the face's middle, which the lash lines outvote,
#: then lands off the track though the face is hers. On such a frame only, the face counts as on the track when the lower face's middle
#: (`_lib.face_points.LOWER_FACE`: nose, mouth, chin) is on it. Measured 2026-10-15 on every points file under the references read
#: held to a track (internal/claude/2026-10-15_mrlol_scripts/lower_face_on_track.txt): the lower face's middle is on the track on all
#: 13 of the lead's 2823 frames the check failed (3075-3087, her white sunglasses over her eyes, off her track). Read on every frame it
#: would also take on the second subject's 895 and 897 (the lead's face read from behind her shoulder, FILL_GAP's case), which lie in a
#: span the plan gives her hands back on (data/fun/plan.json, second, 604-948), and 623, 1099, 1127 and 4698, which nobody has looked
#: at; so only classes over the eyes, never the hands. The 3167-3213 frames of this docstring's top stay off, all eleven, either way.
OVER_THE_EYES = ("Eyeglass",)


def face_on_the_track(points: str, first: int, count: int, over_the_eyes: list[tuple[int, int, str]] | None = None) -> dict:
    """What the points file's own record says of the face on the subject's track, on the span's frames; and, on a frame of an
    `over_the_eyes` span that it calls off, whether the lower face's middle is on the mask that frame was held to (the pose pass's
    own: `pose_along_shot.held_masks`), which then counts it on."""
    rec = Path(points).with_suffix(".json")
    record = json.loads(rec.read_text()) if rec.is_file() else {}
    said = record.get("face_on_the_mask")
    if not said:
        return {"known": False, "why": f"{rec.name} " + ("says nothing of a track" if rec.is_file() else "is not there"),
                "frames_whose_face_middle_is_off_the_track": []}
    off = [f for f in said["frames_whose_face_middle_is_off_the_mask"] if first <= f < first + count]
    out = {"known": True, "read_from": rec.name, "what": said["what"], "frames_whose_face_middle_is_off_the_track": off}
    if over_the_eyes:
        out["over_the_eyes"] = {"spans": [[a, b, why] for a, b, why in over_the_eyes], "frames": [], "read": False,
                                "why": "no frame the record calls off lies in a span"}
    under = [f for f in off if any(a <= f <= b for a, b, _ in over_the_eyes or [])]
    if under:
        import numpy as np
        import pose_along_shot as pas
        from _lib import face_points as fp
        from _lib.frames import read_mask
        box = (record.get("made_from") or {}).get("box")
        span = (record.get("made_from") or {}).get("which_are_source_frames")
        if not isinstance(box, dict) or not Path(str(box.get("mask_video"))).is_file() or box.get("its_first_frame_is_source_frame") is None \
                or not span or not record.get("size"):
            out["over_the_eyes"].update(frames=under, why="the record does not name the track the points were held to, where it starts, the "
                                                         "points' frames and their size, or the track is not there")
            return out
        P = fp.load(points, None)
        w, h = record["size"]
        mask, covered = read_mask(box["mask_video"], (w, h), box["its_first_frame_is_source_frame"], span[0], span[1] - span[0] + 1)
        _, given, _ = pas.held_masks(mask, covered)
        # the points' rows and the masks both start on the record's first source frame (pose_along_shot writes them so)
        on = [f for f in under if fp.lower_face_middle_on_mask(P["keypoints"][f - span[0]], P["index"], given[f - span[0]])]
        out["frames_whose_face_middle_is_off_the_track"] = [f for f in off if f not in on]
        out["over_the_eyes"].update(frames=under, read=True, on_the_track_by_the_lower_face=on,
                                    why=("the plan gives back a class over her eyes there (OVER_THE_EYES): the face's middle lies on it, off the "
                                        "track; the lower face's middle (nose, mouth, chin) is read on the track in its place"))
    return out


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


#: Why a frame has no mouth in the class map's form (`class_mouth_gate`), as the record and the run say it.
NO_MOUTH_CLASS_GATE = (f"the lip lines lie on the class map's mouth under {MOUTH_CHECK['on_share']:g} and the class map's mouth is not on the body model's picture (under {CLASS_MOUTH_GATE['on_body']:g} of it) or not near the pose "
                       f"model's mouth (over {CLASS_MOUTH_GATE['near']:g} of a face height), or the class map holds no lips there (CLASS_MOUTH_GATE): "
                       "drawn there it lands in the air or off her face (630b's 658-680, 1010d's chest)")
#: Why a frame has no mouth (`no_mouth_frames`), as the record and the run say it.
NO_MOUTH_NO_HOLE = "the class map labels no teeth or tongue in her mouth there, and --class-mouth-needs-hole asks for none drawn ({why})"
NO_FACE_PART = "no line of the face there (--no-face-on: {why})"
NO_MOUTH_PART = "no mouth there on the job's word (--no-mouth-on: {why})"


def no_mouth_frames(first: int, count: int, spans: list[tuple[int, int, str]]) -> dict[str, list[int]]:
    """The source frames of `spans` (`--no-mouth-on`: the load's frames whose part holds no lips, hair alone, or a person's word on a look
    where her mouth is hidden; each span's why says which): no mouth at all there,
    the body model's own lips stay. The frames neither the lip lines nor the class map's gate place a mouth on are the gate's
    (`NO_MOUTH_CLASS_GATE`)."""
    out: dict[str, list[int]] = {}
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
    inbox = np.zeros(cm.shape[1:], bool)
    inbox[max(y0, 0):y1, max(x0, 0):x1] = True
    out, open_px, centres, inside = [], [], [], []
    for f in range(count):
        # cut to the box before the pieces are labelled (the outside reviewer's G4 of ca86c125): a larger piece outside the head's box
        # cannot win over her mouth inside it, and lips only outside it are no lips there
        lips = np.isin(cm[f], lips_of) & inbox
        if not lips.any():
            out.append(None); open_px.append(0); centres.append(None); inside.append(0)
            continue
        whole = ndimage.binary_fill_holes(lips | (np.isin(cm[f], held_in) & inbox))
        parts, n = ndimage.label(whole)
        if n > 1:
            whole = parts == 1 + int(np.argmax(ndimage.sum(whole, parts, range(1, n + 1))))
        hole = whole & ~lips
        open_px.append(int(hole.sum()))
        inside.append(int((np.isin(cm[f], held_in) & inbox).sum()))
        ys, xs = np.nonzero(whole)
        centres.append((float(xs.mean()), float(ys.mean())))
        cut = lambda m: cv2.resize(m.astype(np.uint8)[max(y0, 0):y1, max(x0, 0):x1], size, interpolation=cv2.INTER_NEAREST).astype(bool)  # noqa: E731
        out.append((cut(whole), cut(hole)))
    held = sum(m is not None for m in out)
    a = np.array(open_px)
    return out, {"what": "the subject's lips and what they hold in the class map, cut by the reference's box", "class_map": str(classes),
                 "frames_holding_the_lips": held, "frames": count, "frames_open": int((a > 0).sum()),
                 "open_px_on_the_canvas_most": [int(a.max()), first + int(a.argmax())] if a.any() else None,
                 "_centres": centres, "_inside": inside}


def mouth_over_time(shares: list[float], passes: list[bool], on_body: list, near: list, partless: set[int], first: int,
                    inside: list[int] | None = None) -> tuple[list[bool], list[bool], list[int], list[int], list[int]]:
    """A frame's form held against its neighbours (`MOUTH_RUNS`): (on the lip lines, the class map's mouth drawn, the source frames held on
    the lines under the share, the source frames filled, the source frames dropped for no teeth or tongue). `partless` holds frame indices
    with no mouth whatever places it. In order: a frame enters the lip lines at MOUTH_CHECK's share and leaves them only under
    `leave_lines`; with `inside` (each frame's teeth and tongue px; `--class-mouth-needs-hole`), a frame off the lines given the class map's
    own mouth with none draws no mouth; then a run of at most `fill_frames` with no mouth between frames with one takes the class map's
    mouth where it lies at least `fill_on_body` on the body model and within the gate's `near` of the pose mouth (never a frame the hole
    rule dropped)."""
    count = len(shares)
    on_lines, held_on, prev = [], [], False
    for f in range(count):
        now = shares[f] >= MOUTH_CHECK["on_share"] or (prev and shares[f] >= MOUTH_RUNS["leave_lines"])
        if now and shares[f] < MOUTH_CHECK["on_share"] and f not in partless:
            held_on.append(first + f)
        on_lines.append(now)
        prev = now and f not in partless
    passes = list(passes)
    holeless = []
    if inside is not None:
        for f in range(count):
            if passes[f] and not on_lines[f] and f not in partless and not inside[f]:
                passes[f] = False
                holeless.append(first + f)
    drawn = [f not in partless and (on_lines[f] or passes[f]) for f in range(count)]
    filled, f = [], 0
    while f < count:
        if drawn[f]:
            f += 1
            continue
        g = f
        while g + 1 < count and not drawn[g + 1]:
            g += 1
        if 0 < f and g < count - 1 and g - f + 1 <= MOUTH_RUNS["fill_frames"] and \
                all(k not in partless and first + k not in holeless and on_body[k] is not None and on_body[k] >= MOUTH_RUNS["fill_on_body"]
                    and near[k] is not None and near[k] <= CLASS_MOUTH_GATE["near"] for k in range(f, g + 1)):
            for k in range(f, g + 1):
                passes[k] = True
                filled.append(first + k)
        f = g + 1
    return on_lines, passes, held_on, filled, holeless


def class_mouth_gate(mouths: list, centres: list, plain_cut: str, points: str, first: int, count: int,
                     shares: list[float] | None = None) -> tuple[list[bool], dict]:
    """A frame at a time, whether the class map's mouth is drawn (`CLASS_MOUTH_GATE`): the share of it (`mouths`, cut by the box, at the
    plain cut's size) on the body model's picture in the plain cut, and how far its centre on the canvas (`centres`) lies from the pose
    model's outer-lip ring's centre, in face heights. A frame with no lips in the class map does not pass. With the figures."""
    import numpy as np
    import capture_masked_run as cap
    from _lib import face_points as fp
    from _lib.frames import stream
    P = fp.load(points, None)
    idx = P["index"]
    outer = [i for k, i in idx.items() if k.endswith("_of_upper_outer_lip") or k.endswith("_of_lower_outer_lip") or k in ("r_outer_corner_of_mouth", "l_outer_corner_of_mouth")]
    pw, ph, _ = cap.probe(plain_cut)
    body = [f[..., :3].max(-1) > CLASS_MOUTH_GATE["body_grey"] for f in stream(plain_cut, (pw, ph), 0, count, pix="rgb24")]
    passes, rows = [], []
    for f in range(count):
        if mouths[f] is None or centres[f] is None:
            passes.append(False)
            rows.append({"frame": first + f, "lips": False, "drawn": False})
            continue
        whole = mouths[f][0]
        on = float((whole & body[f]).sum() / max(int(whole.sum()), 1))
        k = P["keypoints"][first - P["first"] + f]
        c = k[outer].mean(axis=0)
        near = float(np.hypot(centres[f][0] - c[0], centres[f][1] - c[1]) / max(fp.face_height(k, idx), 1e-6))
        agreed = shares is not None and shares[f] >= CLASS_MOUTH_GATE["agreed_share"] and near <= CLASS_MOUTH_GATE["agreed_near"]
        floor = CLASS_MOUTH_GATE["agreed_on_body"] if agreed else CLASS_MOUTH_GATE["on_body"]
        ok = on >= floor and near <= CLASS_MOUTH_GATE["near"]
        passes.append(bool(ok))
        rows.append({"frame": first + f, "on_body": round(on, 3), "from_the_pose_mouth_in_face_heights": round(near, 3), "on_body_floor": floor,
                     "passed": bool(ok), "drawn": bool(ok)})
    return passes, {"what": ("a frame at a time: the share of the class map's mouth on the body model's picture in the plain cut, and its centre's "
                             "distance from the pose model's outer-lip ring's centre in face heights"), "rule": dict(CLASS_MOUTH_GATE),
                    "rule_is": "starting values (bench/results/2026-10-15_class_mouth_gate.md)", "frames_drawn": int(sum(passes)), "frames": count,
                    "a_frame_at_a_time": rows}


def run(mesh: str, points: str, source: str, first: int, count: int, out: Path, by: str, *, names: list[str] | None = None,
        mesh_first: int | None = None, points_first: int | None = None, width: float = FORM_WIDTH, room: float | None = None,
        leave_out: frozenset = frozenset({FORM_LEAVES_OUT}), dots: bool = False, iris: str = faces.IRIS_FORMS[0], repo: Path = REPO,
        mouth: str | None = None, mouth_why: str = "", classes: str | None = None, classes_first: int | None = None,
        no_mouth_on: list[tuple[int, int, str]] | None = None, over_the_eyes: list[tuple[int, int, str]] | None = None,
        no_face_on: list[tuple[int, int, str]] | None = None, class_mouth_needs_hole: str | None = None,
        eyes_shut_on: list[tuple[int, int, str]] | None = None, eyes_shut_form: str | None = None, say=print) -> dict:
    """Cut, cut, draw, record. Returns the record; `passed` is True only when all three steps passed.

    `no_face_on`: source frames on which no line of the face is drawn at all, the plain cut body model alone, each with its reason:
    the load's part is her hair alone there (the lead's ruling of 2026-10-15 on second_hp_5461b, whose reference drew eyes, brows and
    a nose as large as on a plain face while her face was hidden behind her hands), or a person's word that the face does not show
    there (a head turned away, hidden behind somebody: the hall shot's references of 2026-10-15). Those frames have no mouth either;
    where they cover the whole span, the record lists every part as left out."""
    faceless = sorted({f for a, b, _ in no_face_on or [] for f in range(max(a, first), min(b, first + count - 1) + 1)})
    # HER EYES SHUT ON A SPAN, on a person's word (`--eyes-shut-on` with `--eyes-shut-form`, the lead's ruling of 2026-10-15 on 2823b;
    # `draw_face_points.EYES_SHUT_FORMS` says why and what each form draws). The pose model's lids stop at a slit on a shut eye
    shut = sorted({f for a, b, _ in eyes_shut_on or [] for f in range(max(a, first), min(b, first + count - 1) + 1)} - set(faceless))
    if eyes_shut_on and eyes_shut_form not in faces.EYES_SHUT_FORMS:
        raise SystemExit(f"--eyes-shut-on takes --eyes-shut-form, one of {faces.EYES_SHUT_FORMS}; got {eyes_shut_form!r}")
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
    on_track = face_on_the_track(points, first, count, over_the_eyes)
    if (on_track.get("over_the_eyes") or {}).get("frames") and not on_track["over_the_eyes"]["read"]:
        say(f"    --over-the-eyes NOT READ on {len(on_track['over_the_eyes']['frames'])} frame(s) off the track: {on_track['over_the_eyes']['why']}")
    if (on_track.get("over_the_eyes") or {}).get("on_the_track_by_the_lower_face"):
        got = on_track["over_the_eyes"]["on_the_track_by_the_lower_face"]
        say(f"    {len(got)} frame(s) whose face's middle lies on her own glasses, off the track, are on it by the lower face (--over-the-eyes): "
            + ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in spans_of(got)))
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
    mouths, lines, centres, gated, by_class, holeless = None, True, None, [], [], []
    if mouth_said and mouth_said["form"] == "auto":
        cw, ch, _ = cap_probe(str(made["plain_cut"]))
        mouths, held = class_mouths(classes, classes_first, first, count, cut["box"], (cw, ch))
        centres = held.pop("_centres")
        mouth_said["class_map"] = held
    # no clip-level gate (the outside reviewer's G4 of ca86c125, the lead's ruling of 2026-10-15): every frame is gated, so a span
    # turned away on most frames keeps the mouths its other frames pass; only a class map with no lips on any frame leaves it out
    if mouths is not None and (mouth_said["drawn"] or held["frames_holding_the_lips"]):
        # A FRAME AT A TIME, in either form (the lead, 2026-10-15: under eed4acb9 the lines' share alone left the remade references with
        # no mouth on most frames, 2823's wide-open mouth among them): the lip lines where they lie on the class map's mouth, with its
        # inside under them; else the class map's own mouth where its gate passes (`CLASS_MOUTH_GATE`); else no mouth
        if not mouth_said["drawn"]:
            mouth_said.update(drawn=True, drawn_as="the class map")
            say(f"    the mouth: IN THE CLASS MAP'S FORM, which holds her lips on {held['frames_holding_the_lips']} of {count} frames, "
                f"open on {held['frames_open']}: its own mouth (the sculpted lips painted out, the "
                "outline white, the inside dark) on the frames the lip lines miss and its gate passes, the lip lines on the frames they lie on it")
        lines = False
        passes, gate = class_mouth_gate(mouths, centres, str(made["plain_cut"]), points, first, count, shares)
        partless = {f for fs in no_mouth_frames(first, count, no_mouth_on or []).values() for f in fs} | set(faceless)  # no mouth there whatever places it
        # AN OPTION, off unless asked (`--class-mouth-needs-hole WHY`, the lead's word of 2026-10-15 for one test, 5114e): a frame given
        # the class map's own mouth with no teeth or tongue labelled draws no mouth. Not a default: the class map labels none in open mouths
        # as well as closed ones (bench/results/2026-10-15_mouth_runs.md, item 1); the renders of 5114d and 630d suggest the model opens
        # her mouth with the source where none is drawn, which the test decides
        rows = gate["a_frame_at_a_time"]
        on_lines, passes, held_on, refilled, holeless = mouth_over_time(
            shares, passes, [r.get("on_body") for r in rows], [r.get("from_the_pose_mouth_in_face_heights") for r in rows],
            {f - first for f in partless}, first, held.pop("_inside") if class_mouth_needs_hole else None)
        held.pop("_inside", None)
        for f in refilled:
            rows[f - first].update(passed=True, filled=True)
        for f in holeless:
            rows[f - first].update(passed=False, no_teeth_or_tongue=True)
        by_lines = [first + f for f in range(count) if on_lines[f] and first + f not in partless]
        by_class = [first + f for f in range(count) if not on_lines[f] and passes[f] and first + f not in partless]
        for f in range(count):
            rows[f]["drawn"] = first + f in by_class                    # the class map's own mouth drawn there, after every rule
        gate["frames_drawn"] = len(by_class)
        agreed = [r["frame"] for r in rows if r.get("on_body_floor") == CLASS_MOUTH_GATE["agreed_on_body"] and r["drawn"] and not r.get("filled")]
        gated = [first + f for f in range(count) if not on_lines[f] and not passes[f] and first + f not in holeless]
        mouths = [m if on_lines[f] or passes[f] else None for f, m in enumerate(mouths)]
        mouth_said.update(inside="the class map", inside_on_frames=sum(m is not None for f, m in enumerate(mouths) if on_lines[f]), gate=gate,
                          frames_by_the_lip_lines=spans_of(by_lines), frames_by_the_class_map=spans_of(by_class),
                          over_time={"rule": dict(MOUTH_RUNS), "rule_is": "starting values (bench/results/2026-10-15_mouth_runs.md)",
                                     "held_on_the_lip_lines": spans_of(held_on), "filled_by_the_class_map": spans_of(refilled)},
                          **({"class_mouth_needs_hole": {"why": class_mouth_needs_hole, "no_mouth_on": spans_of(holeless)}} if class_mouth_needs_hole else {}))
        leave_out = (frozenset(leave_out) - {"mouth"}) if by_lines else (frozenset(leave_out) | {"mouth"})
        say(f"    a frame at a time: the lip lines on {len(by_lines)} of {count} frames ({len(held_on)} held on them under the share, over "
            f"{MOUTH_RUNS['leave_lines']:g}), the class map's own mouth on {len(by_class)} (on the body model at {CLASS_MOUTH_GATE['on_body']:g} or "
            f"more, {CLASS_MOUTH_GATE['agreed_on_body']:g} where the pose and the class map agree, on {len(agreed)}; {len(refilled)} drop-out frame(s) "
            f"filled at {MOUTH_RUNS['fill_on_body']:g}; within {CLASS_MOUTH_GATE['near']:g} of a face height of the pose mouth)"
            + (f", none for no teeth or tongue on {len(holeless)}" if class_mouth_needs_hole else "") + "; its inside under the lines where it is open")
    elif mouths is not None:
        mouths = None
        say(f"    the mouth: LEFT OUT: the class map holds her lips on none of the {count} frames")
    # NO MOUTH ON A FRAME where neither the lines nor the class map's gate place it, or where the load's part holds no lips (the lead's
    # rulings of 2026-10-14 on the blind readers of 630b, 805b and 1010d: the mouth drawn off her face in profile and on her chest)
    bare = no_mouth_frames(first, count, no_mouth_on or [])
    for a_, b_, why in no_face_on or []:
        fs = [f for f in range(max(a_, first), min(b_, first + count - 1) + 1)]
        if fs:
            bare.setdefault(NO_FACE_PART.format(why=why), []).extend(fs)
    if gated:
        bare = {NO_MOUTH_CLASS_GATE: gated, **bare}
    if holeless:
        bare = {NO_MOUTH_NO_HOLE.format(why=class_mouth_needs_hole): holeless, **bare}
    if mouth_said and mouth_said["drawn"] and bare:
        mouth_said["no_mouth_on"] = {rule: spans_of(fs) for rule, fs in bare.items()}
        mouth_said["frames_with_no_mouth"] = len({f for fs in bare.values() for f in fs})
        for rule, fs in bare.items():
            say(f"    NO MOUTH on {len(fs)} of {count} frames ({', '.join(f'{a}-{b}' if a != b else str(a) for a, b in spans_of(fs))}): {rule}")
    no_mouth = sorted({f for fs in bare.values() for f in fs}) if mouth_said and mouth_said["drawn"] else []
    if faceless and len(faceless) == count:
        leave_out = frozenset(leave_out) | {"eyes", "brows", "nose", "mouth"}
        if mouth_said:                                  # no face on any frame: no mouth, whatever the form or a person's word said
            mouth_said.update(drawn=False, why_not_drawn="no line of the face on any frame (--no-face-on over the whole span)")
    if faceless:
        say(f"    NO FACE LINES on {len(faceless)} of {count} frames ({', '.join(f'{a}-{b}' if a != b else str(a) for a, b in spans_of(faceless))}): "
            + "; ".join(why for _, _, why in no_face_on))
    if shut:
        say(f"    HER EYES SHUT on {len(shut)} of {count} frames ({', '.join(f'{a}-{b}' if a != b else str(a) for a, b in spans_of(shut))}), drawn "
            f"{eyes_shut_form!r}: " + "; ".join(why for _, _, why in eyes_shut_on or []))
    say(f"3 of 3: the face drawn on the cut body model, {width:g} px, {'a dot at every point' if dots else 'no dots'}"
        + (f", left out: {', '.join(sorted(leave_out))}" if leave_out else "")
        + ((", the mouth from the class map" if lines else ", the class map's mouth a frame at a time") if mouths else ""))
    drawn = faces.run(str(made["plain_cut"]), str(made["plain_cut_points"]), first, count, out, by, width=width, dots=dots, leave_out=leave_out,
                      source=str(made["source_cut"]), source_first=first, iris=iris, mouths=mouths, mouth_lines=lines, no_mouth=no_mouth,
                      class_mouth_frames=by_class, bare=faceless,
                      eyes_shut=shut, eyes_shut_form=eyes_shut_form if shut else None,
                      eyes_shut_by="; ".join(f"{a_}-{b_}: {why}" for a_, b_, why in eyes_shut_on or []),
                      bare_by="; ".join(f"no face lines: {why}" for _, _, why in no_face_on or []), say=quiet)
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
                  **({"mouth": mouth_said} if mouth_said else {}),
                  **({"eyes_shut": {"form": eyes_shut_form, "frames": spans_of(shut), "on_whose_word": [{"first": a_, "last": b_, "why": why}
                                                                                                        for a_, b_, why in eyes_shut_on or []],
                                    "drawn_shut_on": spans_of((drawn.get("frames_with_the_eyes_shut") or {}).get("frames") or [])}}
                     if shut else {})},
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


def no_mouth_arg(text: str, flag: str = "--no-mouth-on") -> tuple[int, int, str]:
    rng, sep, why = text.partition("=")
    try:
        if len(rng.split("-")) > 2:
            raise ValueError
        a, b = (int(x) for x in (rng.split("-") * 2)[:2])
    except ValueError:
        raise SystemExit(f"{flag} {text!r} is not FIRST-LAST=WHY") from None
    if not sep or not why.strip() or a > b:
        raise SystemExit(f"{flag} {text!r} is not FIRST-LAST=WHY, with the why")
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
                   help="source frames on which no mouth is drawn, with why: the load's part holds no lips there (the job's parts_on, hair alone), "
                        "or a person's word on a look (the job's mouth.no_mouth_on); repeatable")
    a.add_argument("--class-mouth-needs-hole", metavar="WHY", help="an option, off unless asked: on a frame given the class map's own mouth "
                   "with no teeth or tongue labelled, draw no mouth; the reason is kept in the record")
    a.add_argument("--no-face-on", action="append", default=[], metavar="FIRST-LAST=WHY",
                   help="source frames on which no line of the face is drawn, the plain body model alone, with the reason: the load's part "
                        "is her hair alone there, or a person's word that the face does not show there; repeatable")
    a.add_argument("--over-the-eyes", action="append", default=[], metavar="FIRST-LAST=WHY",
                   help="source frames on which the plan gives back a class over her eyes (OVER_THE_EYES, her own glasses): the face counts as on "
                        "her track there when the lower face's middle is; repeatable")
    a.add_argument("--eyes-shut-on", action="append", default=[], metavar="FIRST-LAST=WHY",
                   help="source frames on which her eyes are shut, on a person's word on a look (the pose model's lids stop at a slit): drawn "
                        "by --eyes-shut-form; repeatable")
    a.add_argument("--eyes-shut-form", choices=faces.EYES_SHUT_FORMS,
                   help="what is drawn for her eyes on the --eyes-shut-on frames: none (no eye, the body model's own) or lids (each shut as one line)")
    a.add_argument("--names", help="a json list of the points' names, for a points file made before they were kept")
    a.add_argument("--mesh-first", type=int, help="the source frame of the body-model video's frame 0; the span's first when left out")
    a.add_argument("--points-first", type=int, help="the source frame of the points file's row 0; the file's own when it says")
    v = a.parse_args()
    names = json.loads(Path(v.names).read_text()) if v.names else None
    rec = run(v.mesh, v.points, v.source, v.first, v.count, Path(v.out), v.by, names=names, mesh_first=v.mesh_first, points_first=v.points_first,
              width=v.width, room=v.room, leave_out=faces.parts_arg(v.leave_out), dots=v.dots, iris=v.iris, mouth=v.mouth, mouth_why=v.mouth_why,
              classes=v.classes, classes_first=v.classes_first, no_mouth_on=[no_mouth_arg(x) for x in v.no_mouth_on],
              over_the_eyes=[no_mouth_arg(x, "--over-the-eyes") for x in v.over_the_eyes],
              no_face_on=[no_mouth_arg(x, "--no-face-on") for x in v.no_face_on], class_mouth_needs_hole=v.class_mouth_needs_hole,
              eyes_shut_on=[no_mouth_arg(x, "--eyes-shut-on") for x in v.eyes_shut_on], eyes_shut_form=v.eyes_shut_form)
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
