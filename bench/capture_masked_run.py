#!/usr/bin/env python3
"""Capture what the tracker and the masks did on a masked run, for any number of subjects, and show it.

    <python> bench/capture_masked_run.py files --name NAME --source CLIP --first N --frames F \\
        --mask a:track=A_track.mp4,parts=A_parts.mp4,shots=A_shots.json,by=preview_a \\
        --mask b:track=B_track.mp4,by=preview_a \\
        --run pass_a:render=A.mp4,subject=a,others=b [--voice voice_per_frame.csv]
    <python> bench/capture_masked_run.py files --name NAME ... --mask a:... --mask b:... \
        --plan pass_a:subject=a,others=b,margin=64          # before anything samples: the region a run WOULD take
    <python> bench/capture_masked_run.py preflight data/<date>_<NAME> [--not-in a:294-414] [--text P.txt --voice-spans V.json]
    <python> bench/capture_masked_run.py video data/<date>_<NAME> --source CLIP [--render R.mp4] [--out NAME_diag.mp4]
    <python> bench/capture_masked_run.py outcome data/<date>_<NAME> f003 yes --render R.mp4 --note "what was seen"
    <python> bench/capture_masked_run.py diagnose data/<date>_<NAME> --frames 758-778     # after a render went wrong
    <python> bench/capture_masked_run.py look data/<date>_<NAME> --source CLIP --render R.mp4@14 --held REF.mp4@14
    <python> bench/capture_masked_run.py changed data/<date>_<NAME> --run pass_a --source CLIP --render A.mp4 --series a.Face_Neck
    <python> bench/capture_masked_run.py mouth data/<date>_<NAME> --subject a --arm pass_a=A_MOUTH.mkv
    <python> bench/capture_masked_run.py mask data/<date>_<NAME> --subject b --classes Face_Neck+Hair --out keep_b.mkv

**What it buys.** One folder per captured span, `data/<date>_<name>/` (untracked), that says for every frame
and every subject where the tracker's mask was, what part of it was taken, what the sampler regenerated and
what was given back to somebody else, with a `subject` on every row and file. And one stacked, frame-numbered
video drawn from those same files, so the picture and the table cannot disagree. Before it, each of these
was a log line, a tinted picture or a scratch script (`data/CAPTURE_GAPS.md`).

**The words.** A *subject* is a label you choose (`a`, `b`, any number of them). A *sighting* is one saved
mask video of a subject, made by one tracker run (`by=`): the same subject seen by two runs is two
sightings, and how far they agree is a column. A *run* is one pass that regenerated one subject (`subject=`)
and kept others out of its margin (`others=`, labels joined by `+`); its region is the one each of its
windows saved beside its latent (`<name>_window_N_region.npz`, the mask and the token region the window was
run with), and for a render made before those were written it is read from the render's own review video
(`<render stem>_with_mask.mp4`), which is drawn from the token mask the sampler was given. The manifest
says which (`read_from`).

**files** reads saved videos only: no server, no model, no card. A mask video is white on the mask, at the
canvas, and its frame 0 is source frame `at=` (the span's `--first` when not given); a preview graph that
saves every mask output through `MaskToImage` writes exactly these. What it writes:

    manifest.json            the clip's file name, the span, the canvas, every subject, sighting and run
    frames.csv / .json       one row a frame, across subjects: each pair's mask overlap, each run's region
                             on each other subject, the margin cells given back, two runs' regions on the
                             same cells, two sightings of one subject against each other, and `voiced`
    subjects/<label>/        masks__<by>.npz (track, parts), per_frame.csv / .json, shots__<by>.json;
                             with a pose table (`pose=`, what `MiniMaxH3BodyPose` writes): the table as
                             pose_table__<by>.json and pose__<by>.csv / .json, a row a frame: whose box
                             the body was fitted to, which hands the hand decoder refined, how many
                             keypoints fall outside the frame, how many bodies carry this subject's name;
                             and, from a table with 3D keypoints, what the body is doing (`pose_state`):
                             which way the body, the hips and the head face, the chin's lift, and each
                             wrist's distance from the nose in torso units (a hand at the face);
                             with a class map (`classes=`): classes__<by>.npz and segments.csv / .json, the
                             pixels of each segment `<label>.<class>` per frame; segments_whose.csv / .json,
                             how many of them lie inside the subject's own track, another's, or neither
    runs/<run>/              region.npz (the region in latent cells, and the mask the review shows as
                             carried), per_frame.csv / .json, graph.json (read from the render's picture);
                             segments_in_region.csv / .json: every segment inside the region that is not
                             the carried mask, in pixels and cells
    owners.npz               with more than one subject: `owner` (uint8 per pixel, an index into `labels`, 255
                             nobody, 254 contested): a pixel one track claims is that subject's, a pixel
                             several claim is the one's whose class map names it (`owner_map`)
    status.json, README.md   how it came to be, and what it does not hold

**A plan** (`--plan`) is a run that has not rendered: its region is worked out from the saved masks with the
node's own `grow` and the Masked Source's rule for the others, in whole tokens unless `edge=cells`. It is per
frame; the sampler's region is shared by the frames of a latent step, so the real one differs a little at
moving edges (`planned_region` has the figure from one render). It exists so that `preflight` and `video` can be run on a no-sampling preview, before a render is queued.

**A plan read from the node, not estimated.** A preview of the song node writes what each window WILL be
given (`<name>_plan.json` and a `_planned_region.npz` per window, the file a render saves). `--run
NAME:preview=<the windows folder or the plan file>,subject=LABEL[,others=..]` reads those: the region per
latent step, the window that writes each frame, the frames the composite will leave as the source across a
cut. That is the plan to gate a job on; `--plan` stays for a what-if without the card, and what it works
out is an estimate (per frame, where the node's region is per latent step).

**A plan can be a load, and a pass a list of loads.** A whole-subject pass is rendered one load per shot the
subject is in, each from its own first frame, so a plan takes `at=FIRST` and `frames=N` (the load: its
region exists on those frames only, and its latent steps are counted from its own first frame, which is what
the cut rule needs), and `text=`, `audio=` and `still=` to say what the load is given, kept in the manifest
under `load`. `--loads FILE.json` gives a whole pass at once, as a job builder writes it:
`{"defaults": {"margin": 64, ...}, "loads": [{"first": F, "frames": N, "subject": LABEL, "text": LABEL, ...}]}`;
each load is a plan named `name`, or `<subject>_<first>`, and any `--plan` key may sit in a load or in the
defaults. Preflight adds one rule for a load: `load_starts_on_a_small_subject`.

**preflight** reads a capture folder and writes `flags.json`: each flag has its rule, the subject's label, the
source frames and the figure that raised it. A flag is a prompt to look at those frames, never a refusal. The
rules, each with its constant below: a shot taken close to the match line or in frames the caller says the
subject is not in (`--not-in`); a shot called absent with somebody on screen; a part mask that spills off its
subject, changes size against its own recent frames, or moves within the subject's box; one run's region over
another subject's mask; a region that is mostly not the subject; a track empty inside a shot the subject was
taken in; and a text that names a voice over frames with none, or the reverse (`--text`, `--voice-spans`).
With a pose table (`pose=` on `--mask`), before its mesh is used as a motion video: a body fitted to the whole
frame and not to the subject's box, top level when another subject is in the frame; several bodies under one
name; a frame with a mask and no body; a hand the hand decoder did not refine; a body mostly outside the frame
(`flag_pose`).

**verify** is the first check of a job, before any plan is trusted: does the capture read what the nodes wrote.
It reads the capture folder only and writes `verify.json`; exit 0 when every check held, `VERIFY_FAILED` when
one did not or could not be made (a check that cannot run is not a pass). The checks: every mask video of a
sighting was written in one run (`WRITTEN_TOGETHER_S`: a folder that holds two previews under one name is
how a morning's mask gets loaded in the afternoon), is at the capture's canvas and reaches the span; a
subject's track is not empty on frames its shot table says it was taken in; a plan that carries a part has
that subject's class map; and for every rendered run, the region the windows SAVED against the region read
back from the review picture, cell for cell, with the carried masks' pixels over a pixel apart (`files` reads both when both
exist and keeps the comparison). The saved files are what the node gave the sampler; the review reader is
what every capture of a render without them rests on.

**The gate.** `flags.json` carries a `verdict`: `blocked` while any flag at the top level has not been
overridden, else `clear`, with the blocking flags' ids in `blocking`. `preflight --gate` exits `GATE_BLOCKED`
when it is blocked, so a job builder can refuse to queue; without `--gate` the exit is 0, for a reader. A flag
is overridden by `outcome <capture> <flag> overridden --by WHO --note WHY`, both required, written to
`outcomes.json` as an override and not as a result. An override belongs to the flag as it was when overridden
(its `key`: the rule, who it is about and its frames): run preflight again and a flag that names more or other
frames blocks again.

**outcome** records, beside `flags.json`, what a render did against one flag (`outcomes.json`): it happened or it
did not, in which render, who looked. That is how a threshold's provenance goes from reasoned to measured.
**diagnose** prints what every table says about a stretch of source frames somebody marked as wrong, for every
subject and run, with the flags that were raised on those frames: the first step after a bad render.

**look** answers one question about a whole-subject render, per frame: is this the new subject or a look-alike
of the original. The routine it is for (2026-10-10, after one window of a long pass came back as the original on
one seed and not on the next): on any whole-subject pass of more than one window, read the look on window 1's
saved file while window 2 samples, and stop the run if it reads as the original. It reads the mean grey level over the top of the subject's mask and places the render between
the source (0) and a render of the same subject that held (1). It tells two subjects apart only where they
differ in lightness there, and refuses when the held render does not. Its blind side, met the day it was
written: the area is the ORIGINAL's head, so a new subject who sits lower or smaller in the frame leaves wall
there and reads near a half while being plainly the new subject in stills. A figure near 0 is the original's
look; a figure in between is a frame to look at, not a verdict.

**changed** says what a run redrew: each segment and each subject inside its region, as the mean grey difference
from the source against the floor (pixels outside the region), a segment's difference frame by frame
(`--series`), and the change from the frame before inside the region for render and source. It is the
after-the-render half of `segment_inside_region`: that rule says what a region will take, this says what was
taken.

**mouth** scores a render's mouth against the source's: how open it is per frame (`mouth_openings`), the
agreement of the two series with the render shifted a few frames either way, and the same for the source
against itself as the control. With a voice table it also says whether either mouth rests where the voice does.

**mask** writes any of a subject's classes from its class map as a lossless mask video, for a graph's `keep` or
`others`: the same file form as every other mask here.

**status.json** says how the folder came to be: `running`, `done` or `failed`, with the message and the inputs
that were missing. A folder without it was never finished. A finished folder is not rebuilt in place unless
`--overwrite` is given, since a queued render may be loading its masks. Every mask is also saved as a lossless video
(`track__<by>.mkv`, `parts__<by>.mkv`, `runs/<run>/region.mkv` and `carried.mkv`: ffv1, grey, white on the
mask, frame 0 the span's first frame), so a render can load exactly what was looked at. `parts__<by>.mkv` is
the part as the preview saved it; `parts_held__<by>.mkv` is the same with the frames the gate doubts (empty on
the subject, a jump in size, moved within the subject's box) filled from their undoubted neighbours
(`held_parts`). A part can be rightly empty, so look first and pass `hold=FIRST-LAST+FIRST-LAST` in the
`--mask` spec to fill only the source frames you chose; without it every doubted frame in reach is offered a
fill. With a class map every offer is graded (`grade_holds`, `part_grades.json`): the fill is taken only
where it lies on the part's own classes better than the saved part does, and a frame where neither does is
emptied; `grade=off` in the spec leaves the fill ungraded.
`drop=FIRST-LAST+FIRST-LAST` empties the part on source frames where it should take nothing (it lies on
something that is not the part and there is nothing to fill it from). A plan with `carried=held` works its
region out from that mask. The manifest lists the frames filled, the doubted frames left and those emptied. Written only when the mask video covers
the whole span.

**video** stacks the source, the masks and a render, each with a zoom on the region beside it, and burns the
source frame number and the render frame number above every row, with the render's audio. Every subject has
one colour: an outline for the tracked mask, a solid fill for what was taken from the still, a light fill for
the margin regenerated round it, and white boxes on margin cells that were given back to somebody else. Under
the rows is a panel read from the same tables: the frame number large, each subject's mask and part, each
run's region against its subject, what it covers of anybody else, the change inside the region since the
last frame for source and render, the preflight's flags that are live on that frame, the motion video the
model was shown (`--motion`) or the word none, and the reference stills (`--still`). One frame pulled as a
still should explain itself. The form follows the two reviews the owner called the best they had had
(2026-10-09, a numbered tracked review and a vertical how-it-was-made cut, both made by session scripts that
are gone or tied to one clip); colour by part and contacts are not in it yet.

**What it does not hold.** Anything inside the tracker (its confidence, each call); the part model's own
doubt; the matte and the held frames (their outputs are not saved by the previews this reads); the token
mask itself, which no file holds: the region here is read back from a compressed, tinted picture, cell by
cell, and under the review's legend from a quarter of the signal. `taken_back` is worked out from the
masks with the node's own `grow`, under a fixed margin only. The server-driven capture (build the
no-sampling graph from a render's own graph, queue it, pull `/history`) is not built yet.

Run it from ComfyUI's environment. Nothing here describes what a clip shows: a clip is a file name and a
subject is a label.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import importlib
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, bootstrap  # noqa: E402
from _lib.frames import FIT, FPS, probe, read_mask, stream, write_mask_video  # noqa: E402

#: A latent cell's side in pixels. Inherited: the video VAE's spatial factor; a token is two cells a side, and
#: the Masked Source's `edge` says which of the two the region's edge follows. Read at the finer one.
CELL = 16
#: The share of a cell's readable pixels that must be tinted for the cell to count as regenerated. Reasoned:
#: the region is constant over a cell, so anything but a majority is the codec.
CELL_MAJORITY = 0.5
#: A pixel this bright in every channel under the legend is its lettering, and is not read. Measured on the
#: Song node's review, 2026-10-10: the patch is black at three quarters, so picture under it stays below this.
LEGEND_TEXT = 150
#: Columns in a row that must all be undarkened for the legend to have ended. Reasoned: wider than a letter's
#: gap in the legend's own text, which is lit, and far narrower than the legend.
LEGEND_GAP = 24
#: A token's side in latent cells. Inherited: the video model patches two cells a side (`video_mask.token_mask`).
TOKEN_CELLS = 2

# The gate's thresholds. Each raises a flag to look at, never a refusal.
#: How close to the match line a taken or absent shot is flagged. Measured on one clip, 2026-10-10: two wrong
#: takes sat 0.011 and 0.016 above the line and the nearest right one 0.029 above, so this flags that one too.
NEAR_LINE = 0.05
#: How far UNDER the line a shot called absent is worded as "close enough to be them". It no longer sets the
#: level: on the same clip a second wrong absence scored 0.133 under, past this, so the level is the caller's
#: word (`flag_shots`). Measured: wrong absences at 0.052 and 0.133 under, right ones 0.20 to 0.37 under.
NEAR_UNDER = 0.10
#: A part mask with under this share of itself inside its subject's mask has spilled. Inherited: the
#: 2026-10-07 captures' own line for a spill (`data/2026-10-07_capture_windows/README.md`).
PART_INSIDE = 0.8
#: A part's area under this fraction of its own recent median, or over its inverse, is a jump. Inherited:
#: the same captures called a part under half its usual size a collapse.
PART_SIZE = 0.5
#: How far a part's centre may sit from its own recent median, as a share of the subject's box, before it is
#: flagged as having moved onto something else. Reasoned, then run on one clip the same day (the docstring of
#: `flag_parts` says what it caught and what it did not).
PART_MOVE = 0.15
#: Frames either side that "recent" means. Reasoned: about a second, longer than a blink or a turn of the head.
RECENT = 12
#: The furthest a doubted part frame may be from an undoubted one and still be filled from it. Reasoned: half a
#: second; past that a straight line between two frames says nothing about where a head went.
HOLD_REACH = 12
#: A run's region over this share of ANOTHER subject's mask is flagged. Reasoned: under it is the margin's
#: ordinary brush against a neighbour.
REGION_ON_OTHER = 0.05
#: A segment with fewer pixels than this inside a region is its edge and is not flagged. Reasoned: under a
#: quarter of one latent cell.
SEGMENT_PX = 64
#: A segment a plan relies on with under this share of its pixels inside its subject's own tracked mask is
#: flagged as possibly somebody else's. Reasoned: more of it outside the subject than inside.
OWN_SHARE = 0.5
#: Two subjects' tracked masks with more than this share of the smaller inside the other are one person tracked
#: twice. Reasoned high, and measured on the stretch it was written for (2026-10-10): the frames where a second
#: tracker had taken the first one's person read 0.99 to 1.00, and two people touching read under 0.1.
SAME_PERSON = 0.8
#: The least a held render must differ from the source over the look's area, in grey levels, for the look
#: figure to be read. Reasoned: several times the codec's own difference on an untouched pixel.
LOOK_LIFT = 10.0
#: The share of the cells holding a subject's own part that a `keep` may leave as the original's before it is
#: flagged. Measured on two renders, 2026-10-10, which do not agree at the low end: a face pass with a `keep` on an
#: earring read as the new face while 1 to 2% of the face's cells were kept and went back to the original at 6
#: to 7%; a whole-subject pass with a `keep` on another person's face and hair went back to the original's hair
#: for eighteen frames, of which only the middle nine had over 3% kept and the rest 1 to 2%. So the line is at
#: 1%: it names the whole of the second and some frames of the first that were fine.
KEPT_IN_PART = 0.01
#: A region of which more than this share is not the subject's own mask is flagged: that share is background
#: and props, which the model draws again from the text. Reasoned: more outside the subject than inside.
NOT_SUBJECT = 0.5
#: Words that make a sentence about a voice, and words that turn such a sentence into its denial. Reasoned
#: from the lane's own prompt texts; a heuristic, and `flag_text` prints the sentences it counted.
VOICE_WORDS = ("sing", "voice", "vocal", "lip", "syllable", "speak", "talk", "mouth")
DENIALS = ("not", "none", "never", "without", "no voice", "silent")
#: One colour a subject, in the order subjects are given. Reasoned: told apart by hue at a glance, none of
#: them the review's own red and cyan, so the two pictures are never confused.
COLOURS = ((60, 220, 90), (255, 70, 220), (255, 170, 40), (70, 150, 255), (240, 240, 60), (120, 240, 240))
WHITE, DIM, AMBER, GREEN = (255, 255, 255), (150, 150, 150), (255, 170, 60), (90, 230, 110)
SOLID, LIGHT = 0.55, 0.28      # the two fills' strengths; reasoned: both readable over a dimmed picture
HEADER, ZOOM_MIN = 44, 192     # the strip above each row, and the smallest zoom box, in pixels; reasoned
PANEL = 392                    # the band of numbers under the rows, in pixels; reasoned: fits a 4:3 tile and ten lines
PANEL_TEXT = 1150              # where the panel's lines of text end and the motion tile begins; reasoned: the longest line
LEVEL_COLOURS = {"likely fine": (150, 150, 150), "iffy": (255, 190, 60), "likely to fail": (255, 90, 90)}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def _pack(name: str):
    """A root module of the pack, loaded as the checks load it (a stand-in package, so its relative imports work)."""
    bootstrap(cpu=True)        # no device is used, and no CUDA context is opened beside a render
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    return importlib.import_module(f"_h3pack.{name}")


# ------------------------------------------------------------------ reading videos

def read_classes(path: str, size: tuple[int, int], at: int, first: int, frames: int) -> np.ndarray:
    """A saved class map as [frames, h, w] of uint8 class indices (`sapiens2_parts.CLASS_NAMES`), 0 off the subject.

    The video's grey level is the index (`sapiens2_parts.class_mask`), read back by the node's own
    `class_indices` from one colour channel, never through a conversion to grey: a level one off is another
    class. Nearest-neighbour when the size differs, for the same reason."""
    import torch
    sp = _pack("sapiens2_parts")
    w, h = size
    out = np.zeros((frames, h, w), np.uint8)
    skip, lead = max(first - at, 0), max(at - first, 0)
    for n, frame in enumerate(stream(path, size, skip, max(frames - lead, 0), vf=f"scale={w}:{h}:flags=neighbor", pix="rgb24")):
        out[lead + n] = sp.class_indices(torch.from_numpy(frame[..., 0].copy()).to(torch.float32) / 255.0).numpy()
    return out


# ------------------------------------------------------------------ the tables (pure: the check drives these)

def segment_rows(label: str, by: str, first: int, classes: np.ndarray, names: tuple[str, ...]) -> list[dict]:
    """One row a frame: the pixels of every class the part model put on this subject. A segment's id is
    `<label>.<class name>`, the same in every table, picture and frame; a class that never appears has no column."""
    counts = np.stack([np.bincount(c.ravel(), minlength=len(names))[:len(names)] for c in classes])
    present = [k for k in range(1, len(names)) if counts[:, k].any()]
    return [{"frame": n, "source_frame": first + n, "subject": label, "seen_by": by,
             **{f"{label}.{names[k]}": int(counts[n, k]) for k in present}} for n in range(len(classes))]


def whose_rows(label: str, first: int, classes: np.ndarray, own: np.ndarray, others: dict[str, np.ndarray],
               names: tuple[str, ...]) -> list[dict]:
    """One row a frame and segment: of a class's pixels on this subject, how many lie inside the subject's own
    tracked mask, inside another subject's, and in neither.

    The part node cuts its class map to the subject's mask WIDENED by its margin, so a class says what a thing
    is and, within that margin of the outline, not whose: a neighbour's hand lying along the subject's outline
    is labelled a hand of the subject's (measured 2026-10-10: on one frame 81% of a "hand" was in the band and
    57% of it inside the other subject's mask). `in_other` is the largest count over the other subjects, with
    its label; a pixel inside both masks counts as the subject's own."""
    rows = []
    for n in range(len(classes)):
        counts = np.bincount(classes[n].ravel(), minlength=len(names))
        for k in np.nonzero(counts[1:len(names)])[0] + 1:
            px = classes[n] == k
            mine = int((px & own[n]).sum())
            theirs = {o: int((px & t[n] & ~own[n]).sum()) for o, t in others.items()}
            who = max(theirs, key=theirs.get) if theirs else None
            rows.append({"frame": n, "source_frame": first + n, "segment": f"{label}.{names[k]}", "px": int(counts[k]),
                         "in_own_track": mine, "in_other_track": theirs[who] if who else 0, "other": who if who and theirs[who] else None,
                         "in_neither": int(counts[k]) - mine - int(np.logical_or.reduce([px & t[n] & ~own[n] for t in others.values()]).sum()
                                                                    if others else 0)})
    return rows


NOBODY, CONTESTED = 255, 254


def owner_map(tracks: dict[str, np.ndarray], classes: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Whose each pixel is, [n, h, w] of uint8: an index into the labels returned, `NOBODY` where no track claims
    it, `CONTESTED` where the tracks and the class maps together do not decide. Also the pixels more than one
    track claims per frame, and the contested ones.

    A pixel one track claims is that subject's. A pixel several claim is the subject's whose class map names it
    something (not Background), when exactly one of the claimants' does; when none does, or more than one, it is
    contested and not guessed. The class is asked only of pixels several tracks claim: within its margin of the
    outline a class map labels a neighbour's things as the subject's (`whose_rows`). Measured where the rule
    came from (2026-10-10, two subjects, 447 frames): it settled 93% of the pixels both tracks claimed."""
    labels = list(tracks)
    stack = np.stack([tracks[label] for label in labels])                    # [s, n, h, w]
    claims = stack.sum(axis=0)
    owner = np.full(claims.shape, NOBODY, np.uint8)
    for i in range(len(labels)):
        owner[stack[i] & (claims == 1)] = i
    shared = claims > 1
    named = np.stack([stack[i] & shared & (classes[label] > 0 if label in classes else False) for i, label in enumerate(labels)])
    votes = named.sum(axis=0)
    for i in range(len(labels)):
        owner[named[i] & (votes == 1)] = i
    owner[shared & (votes != 1)] = CONTESTED
    return owner, labels, shared.sum(axis=(1, 2)), (owner == CONTESTED).sum(axis=(1, 2))


def segments_in_region(first: int, region: np.ndarray, carried: np.ndarray, classes: dict, names: tuple[str, ...]) -> list[dict]:
    """One row a frame for one run: every segment of every subject inside the run's region and outside the mask
    it carries, as pixels and as the cells that hold any of it. `classes[label]` is that subject's class map.
    This is what gets drawn again without being the thing replaced: a neighbour's hand, the subject's own
    hair or earring, something held."""
    rows = []
    for n in range(len(region)):
        open_px = cells_up(region[n]) & ~carried[n]
        row = {"frame": n, "source_frame": first + n}
        for label, cmap in classes.items():
            inside = np.where(open_px, cmap[n], 0)
            counts = np.bincount(inside.ravel(), minlength=len(names))
            for k in np.nonzero(counts[1:len(names)])[0] + 1:
                row[f"{label}.{names[k]}__px"] = int(counts[k])
                row[f"{label}.{names[k]}__cells"] = int((cells_any((cmap[n] == k)[None])[0] & region[n]).sum())
        rows.append(row)
    return rows


#: A part mask with under this share of its pixels on the part's own classes is not on the part. Measured on two
#: shots of a face on a subject who spins (2026-10-10): a part that was on the face read 0.92 to 1.0; a fill,
#: the node's or this tool's, that lay on a raised arm, a hat or the back of a head read 0.0 to 0.78, and every
#: one of those over a half had a saved part beside it that read higher. The bar is a half, with "the higher
#: of the two" doing the rest.
HOLD_INSIDE = 0.5


def grade_holds(parts: np.ndarray, filled: np.ndarray, classes: np.ndarray, made: list[int]) -> tuple[np.ndarray, dict[int, dict]]:
    """The part to load, frame by frame, from the saved part and a proposed fill, graded by the class map.

    A fill takes a neighbour's shape to a frame the gate doubted. On a fast turn the doubted part was right
    and the neighbour's shape lands on an arm or the back of a head (`data/CAPTURE_GAPS.md`, 39). So each is
    scored by the share of its pixels on the part's own classes (`made`) on that frame: the fill is taken
    only where it scores at least `HOLD_INSIDE` and above the saved part; else the saved part stays where
    it scores at least `HOLD_INSIDE`; else the frame is emptied, because nothing of the part is under
    either (turned away, hidden, a blur) and the original there is better than a part drawn on something
    else. Returns the mask and, for every frame where a fill was offered or the saved part was emptied,
    what was done and both scores.

    What it cannot see: a frame where the class map itself is wrong. A class map that calls a blurred
    figure the part's class scores the saved part high; `drop=` is still the caller's."""
    own = np.isin(classes, made)
    out, decisions = parts.copy(), {}

    def inside(mask: np.ndarray, f: int) -> float | None:
        return round(float((mask & own[f]).sum() / mask.sum()), 3) if mask.any() else None

    for f in range(len(parts)):
        offered = bool((filled[f] != parts[f]).any())
        saved, fill = inside(parts[f], f), inside(filled[f], f) if offered else None
        if not offered and (saved is None or saved >= HOLD_INSIDE):
            continue
        if offered and fill is not None and fill >= HOLD_INSIDE and fill > (saved or 0.0):
            out[f], did = filled[f], "filled"
        elif saved is not None and saved >= HOLD_INSIDE:
            did = "saved part kept, fill refused"
        else:
            out[f], did = False, "emptied"
        decisions[f] = {"did": did, "saved_on_its_classes": saved, "fill_on_its_classes": fill}
    return out, decisions


def part_classes(parts: np.ndarray, classes: np.ndarray, names: tuple[str, ...]) -> list[int]:
    """Which classes a saved part mask is made of: those with most of their pixels inside it over the clip.
    The preview saves the mask and not the ticks that made it, so this reads them back."""
    inside = np.bincount(classes[parts].ravel(), minlength=len(names))[:len(names)]
    total = np.bincount(classes.ravel(), minlength=len(names))[:len(names)]
    return [k for k in range(1, len(names)) if total[k] and inside[k] / total[k] > 0.5]


def cells_any(mask: np.ndarray, cell: int = CELL) -> np.ndarray:
    """[n, h, w] of bool to [n, h / cell, w / cell]: true where any pixel of the cell is."""
    n, h, w = mask.shape
    return mask.reshape(n, h // cell, cell, w // cell, cell).any(axis=(2, 4))


def cells_up(cells: np.ndarray, cell: int = CELL) -> np.ndarray:
    return np.repeat(np.repeat(cells, cell, axis=-2), cell, axis=-1)


def whole_tokens_of(cells: np.ndarray) -> np.ndarray:
    """[n, ch, cw] of cells widened to whole tokens: a token is on when any of its cells is."""
    n, ch, cw = cells.shape
    t = TOKEN_CELLS
    padded = np.pad(cells, ((0, 0), (0, ch % t), (0, cw % t)), mode="edge")
    tokens = padded.reshape(n, padded.shape[1] // t, t, padded.shape[2] // t, t).any(axis=(2, 4))
    return np.repeat(np.repeat(tokens, t, axis=1), t, axis=2)[:, :ch, :cw]


def planned_region(carried: np.ndarray, others: list[np.ndarray], margin: int, grow, whole_tokens: bool = True,
                   keep: list[np.ndarray] | None = None) -> np.ndarray:
    """The cells a run would regenerate, [n, h / CELL, w / CELL], from the masks alone.

    The Masked Source's rule (`video_mask.window`), a frame at a time: the carried mask grown by the margin,
    in whole tokens; less every token the others touch, unless the subject's own mask, before any margin, has
    a pixel in it; then less every token a `keep` mask touches, the subject's own included. With `whole_tokens`
    off the same in cells. The frames of one latent step share a region in
    the sampler, for the subject and for the others alike, which this does not model: the real region is a
    little larger where the subject moves and a little smaller where the others do. Set against one render's
    own region the day it was written (447 frames, a whole subject, another kept out): intersection over
    union 0.99 at the median and 0.92 at the fifth percentile."""
    import torch
    grown = grow(torch.from_numpy(carried).to(torch.float32), int(margin)).numpy() > 0.5
    widen = whole_tokens_of if whole_tokens else (lambda cells: cells)
    region, own = widen(cells_any(grown)), widen(cells_any(carried))
    for other in others:
        region &= ~(widen(cells_any(other)) & ~own)
    for kept in keep or []:
        region &= ~widen(cells_any(kept))          # after the others, and over the subject's own mask too: keep wins
    return region


def overlap(a: np.ndarray, b: np.ndarray) -> float | None:
    """Intersection over union of two masks; None when both are empty (nothing to agree about)."""
    union = int((a | b).sum())
    return round(float((a & b).sum()) / union, 4) if union else None


def box_of(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None


def pieces(mask: np.ndarray) -> int:
    import cv2
    return int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0]) - 1


def subject_rows(label: str, by: str, first: int, track: np.ndarray, covered: np.ndarray,
                 parts: np.ndarray | None = None) -> list[dict]:
    """One row a frame for one sighting of one subject. The columns the 2026-10-07 captures had, plus who."""
    rows, area = [], float(track[0].size)
    for n in range(len(track)):
        row = {"frame": n, "source_frame": first + n, "subject": label, "seen_by": by, "covered": int(covered[n])}
        if covered[n]:
            box = box_of(track[n])
            row.update(track_share=round(float(track[n].sum()) / area, 5), track_box=box, track_pieces=pieces(track[n]),
                       track_step=overlap(track[n], track[n - 1]) if n and covered[n - 1] else None)
            if parts is not None:
                p, t = parts[n], track[n]
                ys, xs = np.nonzero(p)
                centre = None
                if len(xs) and box:
                    centre = [round((float(xs.mean()) - box[0]) / max(box[2] - box[0], 1), 3),
                              round((float(ys.mean()) - box[1]) / max(box[3] - box[1], 1), 3)]
                row.update(parts_share=round(float(p.sum()) / area, 5),
                           parts_of_track=round(float(p.sum()) / max(float(t.sum()), 1.0), 4),
                           parts_inside_track=round(float((p & t).sum()) / float(p.sum()), 4) if p.any() else None,
                           parts_step=overlap(p, parts[n - 1]) if n and covered[n - 1] else None,
                           parts_centre_in_box=centre)
        rows.append(row)
    return rows


def taken_back(carried: np.ndarray, other: np.ndarray, region: np.ndarray, margin: int, grow) -> tuple[np.ndarray, np.ndarray]:
    """Where a run's margin reached another subject, and the cells of it that stayed the source's.

    `reached` is the Masked Source's own expression for its log line (the grown mask, on the other's pixels,
    off the subject's own), [n, h, w]. `back` is the cells that hold such a pixel and are not in the region
    the review shows, [n, h / CELL, w / CELL]. `grow` is `video_mask.grow`, handed in so the check can too."""
    import torch
    grown = grow(torch.from_numpy(carried).to(torch.float32), int(margin)).numpy() > 0.5
    reached = grown & other & ~carried
    return reached, cells_any(reached) & ~region


def run_rows(run: str, label: str, first: int, region: np.ndarray, carried: np.ndarray, read: np.ndarray) -> list[dict]:
    rows = []
    for n in range(len(region)):
        row = {"frame": n, "source_frame": first + n, "run": run, "subject": label, "read": int(read[n])}
        if read[n]:
            cells = cells_any(carried[n:n + 1])[0]
            row.update(carried_share=round(float(carried[n].mean()), 5), region_share=round(float(region[n].mean()), 5),
                       margin_share=round(float((region[n] & ~cells).mean()), 5),
                       region_cells=int(region[n].sum()),
                       region_step=overlap(region[n], region[n - 1]) if n and read[n - 1] else None)
        rows.append(row)
    return rows


def cross_rows(first: int, frames: int, sightings: dict, runs: dict, voice: dict | None = None) -> list[dict]:
    """One row a frame across subjects. `sightings[label][by] = (track, covered, parts or None)`;
    `runs[name] = {"subject", "others", "region", "carried", "read", "reached": {label: px}, "back": {label: cells}}`.

    A pair of subjects is read on each one's first sighting; every other sighting of a subject is set
    against its first in `same_subject__<label>__<by>__<by>`, which is where two runs' trackers disagree."""
    labels = list(sightings)
    lead = {label: next(iter(sightings[label].values())) for label in labels}
    rows = []
    for n in range(frames):
        row: dict = {"frame": n, "source_frame": first + n}
        if voice is not None:
            row.update(voice.get(first + n, {"voiced": None}))
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                (ta, ca, _), (tb, cb, _) = lead[a], lead[b]
                if ca[n] and cb[n]:
                    both = float((ta[n] & tb[n]).sum())
                    row[f"masks_overlap__{a}__{b}"] = round(both / ta[n].size, 5)
                    row[f"masks_overlap_of_smaller__{a}__{b}"] = round(both / max(min(float(ta[n].sum()), float(tb[n].sum())), 1.0), 4)
            seen = list(sightings[a].items())
            for by, (track, covered, _) in seen[1:]:
                if covered[n] and seen[0][1][1][n]:
                    row[f"same_subject__{a}__{seen[0][0]}__{by}"] = overlap(seen[0][1][0][n], track[n])
        names = list(runs)
        for i, name in enumerate(names):
            r = runs[name]
            if not r["read"][n]:
                continue
            up = cells_up(r["region"][n])
            for b in labels:
                if b == r["subject"] or not lead[b][1][n]:
                    continue
                tb = lead[b][0][n]
                row[f"region_on__{name}__{b}"] = round(float((up & tb).sum()) / max(float(tb.sum()), 1.0), 4)
                row[f"region_cells_holding__{name}__{b}"] = int((r["region"][n] & cells_any(tb[None])[0]).sum())
                if b in r.get("back", {}):
                    row[f"margin_reached__{name}__{b}"] = round(float(r["reached"][b][n].mean()), 5)
                    row[f"cells_given_back__{name}__{b}"] = int(r["back"][b][n].sum())
            for other in names[i + 1:]:
                o = runs[other]
                if o["read"][n]:
                    both = int((r["region"][n] & o["region"][n]).sum())
                    row[f"regions_overlap_cells__{name}__{other}"] = both
        rows.append(row)
    return rows


# ------------------------------------------------------------------ reading a run's region from its review

def legend_width(review: str, size: tuple[int, int], rows: int) -> int:
    """How far the review's legend reaches from the left, read off its top edge on the first frames."""
    w, h = size
    ratio = []
    for tile in stream(review, size, 0, 8, vf=f"crop={w}:{h}:0:{h}", pix="rgb24"):
        t = tile.astype(np.float32).sum(-1)
        ratio.append((t[h - rows + 1:h - rows + 4].mean(0) + 1.0) / (t[h - rows - 4:h - rows - 1].mean(0) + 1.0))
    dark = np.convolve(np.median(ratio, axis=0), np.ones(9) / 9, mode="same") < 0.6
    # the legend starts at the left edge and is one piece: stop at the first stretch that is not darkened,
    # since a column of the picture that happens to darken downwards is not the legend
    lit = np.convolve(~dark, np.ones(LEGEND_GAP), mode="valid") >= LEGEND_GAP
    return int(np.argmax(lit)) if lit.any() else (w if dark.all() else 0)


def read_region(render: str, source: str, first: int, frames: int, size: tuple[int, int], at: int | None = None):
    """A run's region and the mask its review shows as carried, un-tinted against the source's own frames.

    The Song node's review tints each pixel by one layer only (`video_mask.overlay_pieces`): the mask the
    source carries, then what else regenerates. Each pixel here is whichever of plain, the first tint and
    the second it is nearest; the region is both tints, by the majority of a cell's readable pixels.
    The render's frame 0 is source frame `at` (the span's `first` when not given); a span frame the render
    does not reach is marked not read."""
    vm = _pack("video_mask")
    w, h = size
    (c1, s1), (c2, s2) = vm.OVERLAY_SUBJECT, vm.OVERLAY_REGION
    t1, t2 = np.array(c1, np.float32) * 255.0, np.array(c2, np.float32) * 255.0
    legend, opacity = vm.overlay_legend([], h, w)
    lh = int(legend.shape[0])
    review = render[:-len(".mp4")] + "_with_mask.mp4"
    lw = legend_width(review, size, lh)
    region = np.zeros((frames, h // CELL, w // CELL), bool)
    carried, read = np.zeros((frames, h, w), bool), np.zeros(frames, bool)
    import cv2
    at = first if at is None else int(at)
    lead = max(at - first, 0)
    tiles = stream(review, size, max(first - at, 0), max(frames - lead, 0), vf=f"crop={w}:{h}:0:{h}", pix="rgb24")
    plates = stream(source, size, first + lead, max(frames - lead, 0), vf=FIT.format(w=w, h=h), pix="rgb24")
    for n, (tile, plate) in enumerate(zip(tiles, plates), start=lead):
        tile, plate = tile.astype(np.float32), plate.astype(np.float32)
        unknown = np.zeros((h, w), bool)
        if lw:
            under = tile[h - lh:, :lw]
            unknown[h - lh:, :lw] = under.min(-1) > LEGEND_TEXT
            tile[h - lh:, :lw] = np.clip(under / (1.0 - opacity), 0, 255)
        kind = np.argmin(np.stack([np.abs(tile - plate).sum(-1),
                                   np.abs(tile - (plate * (1 - s1) + t1 * s1)).sum(-1),
                                   np.abs(tile - (plate * (1 - s2) + t2 * s2)).sum(-1)]), 0)
        known = (~unknown).reshape(h // CELL, CELL, w // CELL, CELL).sum((1, 3)).clip(1)
        tinted = ((kind > 0) & ~unknown).reshape(h // CELL, CELL, w // CELL, CELL).sum((1, 3))
        region[n] = tinted / known > CELL_MAJORITY
        first_tint = cv2.medianBlur(((kind == 1) & ~unknown).astype(np.uint8), 5).astype(bool)
        carried[n], read[n] = first_tint & cells_up(region[n]), True
    return region, carried, read, {"read_from": Path(review).name, "legend_px": [lw, lh]}


# ------------------------------------------------------------------ a run's region from the files its windows saved

def window_region_files(render: str) -> list[Path]:
    """The region files a render's windows saved (`video_mask.save_window_region`), in window order.

    A render `<dir>/<name>_00001.mp4` keeps its windows in `<dir>/<name>_windows/`, and each window rendered
    over a source leaves `<name>_window_N_region.npz` beside its latent. None from a render made before
    the song node wrote them."""
    path = Path(render)
    name = path.stem.rsplit("_", 1)[0] if path.stem.rsplit("_", 1)[-1].isdigit() else path.stem
    found = {}
    for f in (path.parent / f"{name}_windows").glob(f"{name}_window_*_region.npz"):
        number = f.name[len(name) + len("_window_"):-len("_region.npz")]
        if number.isdigit():
            found[int(number)] = f
    return [found[n] for n in sorted(found)]


def read_saved_regions(files: list[Path], first: int, frames: int, size: tuple[int, int], at: int | None = None,
                       render_frames: int | None = None):
    """A run's region and carried mask from what each window was run with, in place of un-tinting the review.

    Each file is one window: its fitted mask per frame, its token region per latent step, the frame of the
    load it starts on and how many of its frames its video leaves off the front (the context it shares with
    the window before). A frame of the load belongs to the window whose video holds it. The region is the
    token region over each latent step's run of frames (`video_mask.pixel_alpha` with no feather, at the
    cell grid), taken off the frames `video_mask.cut_gate` leaves as the source across a cut, because
    those were not laid. Under `only what changed` the composite keeps less than this; the review reader
    cannot see that either, and `how` names the composite. The load's frame 0 is source frame `at`.

    Returns what `read_region` does. The last window of a load can run past the load's last frame (its tail
    is held frames, which the render does not write): frames past `render_frames` are dropped, and window
    files after the one that reaches the render's end are an earlier, longer run's and are left unread and
    named. Refuses (None) when the windows leave a gap or stop short of the render's end."""
    import torch
    vm = _pack("video_mask")
    w, h = size
    region = np.zeros((frames, h // CELL, w // CELL), bool)
    carried, read = np.zeros((frames, h, w), bool), np.zeros(frames, bool)
    at = first if at is None else int(at)
    end, across, names, composite = 0, [], [], None
    last = None if render_frames is None else int(render_frames)
    for f in files:
        if last is not None and end >= last:
            break
        got = vm.load_window_region(str(f))
        start, trim, mask, tokens = got["first_frame"], got["trim"], got["mask"], got["tokens"]
        count = int(mask.shape[0])
        if start + trim != end:
            return None, None, None, {"refused": f"{f.name} starts its video on frame {start + trim} of the load and the "
                                                 f"window before it ended on {end}"}
        cells = vm.pixel_alpha(tokens, h // CELL, w // CELL, 0) > 0.5
        gate = vm.cut_gate(mask, int(tokens.shape[0]), got["source"].get("cuts"), start) > 0.5
        if tuple(mask.shape[1:]) != (h, w):
            mask = torch.nn.functional.interpolate(mask[:, None], size=(h, w), mode="nearest")[:, 0]
        for k in range(trim, count if last is None else min(count, last - start)):
            n = at + start + k - first
            if 0 <= n < frames:
                region[n], carried[n], read[n] = (cells[k] & gate[k]).numpy(), (mask[k] > 0.5).numpy(), True
            if not bool(gate[k]):
                across.append(at + start + k)
        end, composite = start + count, got["source"].get("composite")
        names.append(f.name)
    if last is not None and end < last:
        return None, None, None, {"refused": f"the windows' files cover {end} frames and the render has {render_frames}"}
    return region, carried, read, {"read_from": "the region each window saved beside its latent", "files": names, "legend_px": None,
                                   "composite": composite, "left_as_the_source_across_a_cut": across,
                                   "held_frames_past_the_render": 0 if last is None else max(end - last, 0),
                                   "files_of_another_run_left_unread": [f.name for f in files if f.name not in names]}


def plan_file(preview: str) -> Path:
    """The plan a preview of the song node wrote (`<name>_plan.json`), from its path or its windows folder."""
    path = Path(preview)
    if path.is_file():
        return path
    found = sorted(path.glob("*_plan.json"))
    if len(found) != 1:
        raise SystemExit(f"{preview}: {len(found)} plan file(s) (*_plan.json); a preview's windows folder has one. "
                         "Give the plan file itself when a folder holds more")
    return found[0]


def held_tail_facts(mask, tokens, held: int, first_source_frame: int, run_lengths) -> dict:
    """What a window is given past the end of its load: the frames held, and whether the model sees them clean.

    A window longer than its load is filled with the load's last frame, held. `mask` is the window's fitted
    mask per frame and `tokens` its region per latent step, as the node will give them; the last `held`
    frames are the held ones. A held frame is shown CLEAN when its latent step has no region: the model is
    then given the source's own last frame as a frame to keep, and if the subject's mask is on that last
    frame, it is a picture of the subject the pass is replacing. With the tail's region open (the Masked
    Source's `held_tail`) the held frames carry the last frame's region and nothing is shown clean.

    Returns the window's length, the held count and share, whether the subject is on the last real frame, how
    many held frames are shown clean, and the source frames of the latent step that holds both real and held
    frames (the first to go: its region is made from fewer masked frames than it has)."""
    total = int(mask.shape[0])
    real = total - int(held)
    out = {"frames": total, "held_frames": int(held), "held_share": round(held / max(total, 1), 3),
           "subject_on_the_last_real_frame": bool(real > 0 and bool((mask[real - 1] > 0.5).any())),
           "held_frames_shown_clean": 0, "step_with_real_and_held_source_frames": None}
    if held <= 0 or real <= 0:
        return out
    at = 0
    for i, n in enumerate(run_lengths(int(tokens.shape[0]))):
        inside = [f for f in range(at, at + n) if f >= real]
        if inside and not bool((tokens[i] > 0.5).any()):
            out["held_frames_shown_clean"] += len(inside)
        if at < real < at + n:
            out["step_with_real_and_held_source_frames"] = [first_source_frame + at, first_source_frame + real - 1]
        at += n
    return out


def read_planned_regions(plan_path: Path, first: int, frames: int, size: tuple[int, int], at: int | None = None):
    """A run's region as the node's own preview planned it: what each window WILL be given, nothing estimated.

    The plan (`audio_freeze_song.write_plan`) lists every window with the frames it writes and its region
    file, the same file a render saves (`<name>_window_N_planned_region.npz`, read by
    `video_mask.load_window_region`). A frame of the load belongs to the window that writes it, by the plan's
    own `first_written_frame` and `last_written_frame`, so a held tail past a short load is the plan's to
    know. The frames the composite will leave as the source across a cut are the plan's
    `frames_left_as_source` (the gate on the mask the window is given), and have no region here. All of the
    plan's numbers are the load's; the load's frame 0 is source frame `at`.

    Returns what `read_region` does. Refuses (None) when a window has no region file (the plan says why: it
    starts past the source's end), when a file the plan names is not there, or when the windows' written
    frames leave a gap."""
    vm = _pack("video_mask")
    plan = json.loads(plan_path.read_text())
    if plan.get("plan") != "h3 song plan":
        raise SystemExit(f"{plan_path.name} is not a song node plan (its `plan` is {plan.get('plan')!r})")
    w, h = size
    region = np.zeros((frames, h // CELL, w // CELL), bool)
    carried, read = np.zeros((frames, h, w), bool), np.zeros(frames, bool)
    at = first if at is None else int(at)
    end, across, windows = 0, [], []
    for row in plan["windows"]:
        if not row.get("region_file"):
            return None, None, None, {"refused": f"window {row['number']} is not planned: {row.get('why', 'no region file')}"}
        path = plan_path.parent / row["region_file"]
        if not path.is_file():
            return None, None, None, {"refused": f"the plan names {row['region_file']} and it is not beside the plan"}
        if row["first_written_frame"] != end:
            return None, None, None, {"refused": f"window {row['number']} writes from frame {row['first_written_frame']} of the load and "
                                                 f"the window before it ended on {end}"}
        got = vm.load_window_region(str(path))
        cells = vm.pixel_alpha(got["tokens"], h // CELL, w // CELL, 0) > 0.5
        mask, left = got["mask"], set(row.get("frames_left_as_source", []))
        if tuple(mask.shape[1:]) != (h, w):
            import torch
            mask = torch.nn.functional.interpolate(mask[:, None], size=(h, w), mode="nearest")[:, 0]
        for f in range(row["first_written_frame"], row["last_written_frame"] + 1):
            k, n = f - row["first_frame"], at + f - first
            if 0 <= n < frames:
                region[n], carried[n], read[n] = (cells[k].numpy() if f not in left else False), (mask[k] > 0.5).numpy(), True
            if f in left:
                across.append(at + f)
        end = row["last_written_frame"] + 1
        windows.append({"window": row["number"], "writes_source_frames": [at + row["first_written_frame"], at + row["last_written_frame"]],
                        "text": row.get("text"), "regenerating_share": row.get("regenerating_share"), "kept_frames": int(row.get("trim", 0)),
                        "first_frame": at + row["first_frame"],
                        **held_tail_facts(got["mask"], got["tokens"], int(row.get("source_frames_held", 0)), at + row["first_frame"], vm.run_lengths)})
    return region, carried, read, {"read_from": "the region the node's preview planned for each window", "plan": plan_path.name,
                                   "files": [r["region_file"] for r in plan["windows"]], "legend_px": None,
                                   "composite": plan["source"].get("composite"), "left_as_the_source_across_a_cut": across,
                                   "planned_windows": windows, "frames_written": end, "source_settings": plan["source"]}


def graph_of(render: str) -> dict | None:
    """The graph a render's picture carries (`<stem>.png`, the `prompt` text chunk), or None."""
    picture = Path(render[:-len(".mp4")] + ".png")
    if not picture.is_file():
        return None
    from PIL import Image
    text = Image.open(picture).info.get("prompt")
    return json.loads(text) if text else None


def window_settings(graph: dict | None) -> dict | None:
    for node in (graph or {}).values():
        if node.get("class_type") == "MiniMaxH3AudioFreezeSong" and "window_frames" in node["inputs"]:
            return {"window_frames": int(node["inputs"]["window_frames"]), "context_frames": int(node["inputs"].get("context_frames", 0))}
    return None


def source_settings(graph: dict | None) -> dict:
    for node in (graph or {}).values():
        if node.get("class_type") == "MiniMaxH3MaskedSource":
            return {k: node["inputs"].get(k) for k in ("grow_pixels", "grow_by", "replace", "edge", "feather_pixels", "composite")}
    return {}


# ------------------------------------------------------------------ files

def spec(text: str) -> tuple[str, dict]:
    label, _, rest = text.partition(":")
    return label, dict(item.split("=", 1) for item in rest.split(",") if item)


#: What a load says it is given, beside its frames: carried into the manifest, not read.
LOAD_GIVEN = ("text", "audio", "still")


def loads_from_file(path: str) -> list[tuple[str, dict]]:
    """A pass written as a list of loads, as `--plan` specs: one plan a load, `first` as its `at`.

    The file is `{"defaults": {...}, "loads": [{...}, ...]}`. A load needs `first`, `frames` and `subject`;
    everything else a `--plan` takes may be given in a load or once in the defaults. Names must differ."""
    given = json.loads(Path(path).read_text())
    out, seen = [], set()
    for n, load in enumerate(given.get("loads", [])):
        one = {**given.get("defaults", {}), **load}
        lacking = [k for k in ("first", "frames", "subject") if k not in one]
        if lacking:
            raise SystemExit(f"{path}: load {n} lacks {', '.join(lacking)}")
        name = str(one.pop("name", f"{one['subject']}_{int(one['first']):04d}"))
        if name in seen:
            raise SystemExit(f"{path}: two loads are named {name!r}; give one a `name`")
        seen.add(name)
        one["at"] = one.pop("first")
        out.append((name, {k: "+".join(v) if isinstance(v, list) else str(v) for k, v in one.items() if v is not None}))
    return out


def load_frames(first: int, frames: int, at: int, count: int | None) -> np.ndarray:
    """Which of a span's frames (from source frame `first`) a load holds: `count` frames from source frame `at`."""
    n = first + np.arange(frames)
    return (n >= at) & (n < at + count if count is not None else True)


def from_the_load(present: np.ndarray, first: int, load_first: int, count: int | None = None) -> np.ndarray:
    """A per-frame row of a span, renumbered from a load's own first frame: padded with nothing before the span
    when the load starts earlier, cut when it starts later, and ended on the load's last frame when it has
    one (`count` frames). A load ends where it ends: the source's next frame is not in its last latent step."""
    lead = first - load_first
    row = np.r_[np.zeros(lead, present.dtype), present] if lead >= 0 else present[-lead:]
    return row if count is None else row[:count]


def write_table(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    path.with_suffix(".json").write_text(json.dumps({"columns": keys, "rows": rows}) + "\n")
    with open(path.with_suffix(".csv"), "w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=keys)
        out.writeheader()
        out.writerows(rows)


def read_voice(path: str) -> dict:
    with open(path, newline="") as fh:
        return {int(r["frame"]): {"voiced": int(r["voiced"]), "vocals_stem_dbfs": float(r["vocals_stem_dbfs"])}
                for r in csv.DictReader(fh)}


def write_status(out: Path, state: str, **more) -> None:
    """How the folder came to be. Written first as `running`, so a folder that stops there says so."""
    (out / "status.json").write_text(json.dumps({"state": state, "written": datetime.datetime.now().isoformat(timespec="seconds"),
                                                 "by": "bench/capture_masked_run.py files", **more}, indent=1) + "\n")


def files(a: argparse.Namespace) -> Path:
    out = Path(a.out) / f"{a.date}_{a.name}"
    done = out / "status.json"
    if done.is_file() and json.loads(done.read_text()).get("state") == "done" and not a.overwrite:
        # a render may be loading this folder's mask videos: on 2026-10-10 a rebuild rewrote them forty seconds
        # after a queued render had read them
        raise SystemExit(f"{out} is a finished capture; a render may be reading its masks. Give another --name, "
                         "or --overwrite when nothing queued reads it")
    out.mkdir(parents=True, exist_ok=True)
    masks, runs, plans = [spec(m) for m in a.mask], [spec(r) for r in a.run], [spec(r) for r in a.plan]
    plans += [load for path in a.loads for load in loads_from_file(path)]
    # a run with a preview's plan and no render has not rendered: its region is the node's own plan, not an estimate
    plans += [(n, r) for n, r in runs if r.get("preview") and not r.get("render")]
    runs = [(n, r) for n, r in runs if r.get("render")]
    named = [m[k] for _, m in masks for k in ("track", "parts", "shots", "classes", "held", "pose") if m.get(k)] + [a.source]
    named += [x for _, r in runs for x in (r["render"], r["render"][:-len(".mp4")] + "_with_mask.mp4")]
    missing = [x for x in named if not Path(x).is_file()]
    write_status(out, "running", inputs=[Path(x).name for x in named])
    if missing or not masks:
        why = "no --mask given: a capture needs at least one subject's saved mask video" if not masks else \
            "inputs named and not found: " + ", ".join(Path(x).name for x in missing)
        write_status(out, "failed", message=why, inputs_missing=[Path(x).name for x in missing])
        raise SystemExit(why)
    try:
        _files(a, out, masks, runs, plans)
    except BaseException as exc:
        write_status(out, "failed", message=f"{type(exc).__name__}: {exc}", inputs_missing=[])
        raise
    write_status(out, "done", inputs=[Path(x).name for x in named], outputs=sorted(
        str(q.relative_to(out)) for q in out.rglob("*") if q.is_file() and q.name != "status.json"))
    print("wrote", out)
    return out


def _files(a: argparse.Namespace, out: Path, masks: list, runs: list, plans: list) -> None:
    first, frames = int(a.first), int(a.frames)
    width, height, _ = probe(masks[0][1]["track"])
    size = (width, height)
    sightings: dict = {}
    class_maps: dict = {}
    made_of: dict = {}
    held_masks: dict = {}
    manifest = {"name": a.name, "date": a.date, "clip": Path(a.source).name, "first_frame": first, "frames": frames,
                "size": [width, height], "fps": FPS, "cell_px": CELL, "subjects": [], "runs": [],
                "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True).stdout.strip()}
    for label, m in masks:
        by, at = m.get("by", "preview"), int(m.get("at", first))
        track, covered = read_mask(m["track"], size, at, first, frames)
        parts = read_mask(m["parts"], size, at, first, frames)[0] if m.get("parts") else None
        if m.get("classes"):
            names = _pack("sapiens2_parts").CLASS_NAMES
            cmap = read_classes(m["classes"], size, at, first, frames)
            class_maps[label] = cmap
            (out / "subjects" / label).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "subjects" / label / f"classes__{by}.npz", classes=cmap)
            write_table(out / "subjects" / label / "segments", segment_rows(label, by, first, cmap, names))
            made_of[label] = [names[k] for k in part_classes(parts, cmap, names)] if parts is not None else []
        if m.get("held"):
            (out / "subjects" / label).mkdir(parents=True, exist_ok=True)
            write_mask_video(out / "subjects" / label / f"held__{by}.mkv", read_mask(m["held"], size, at, first, frames)[0])
        if by in sightings.setdefault(label, {}):
            raise SystemExit(f"subject {label!r} has two masks by {by!r}; name each sighting's run with by=")
        sightings[label][by] = (track, covered, parts)
        folder = out / "subjects" / label
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f"masks__{by}.npz", track=np.packbits(track, axis=-1), covered=covered,
                            **({"parts": np.packbits(parts, axis=-1)} if parts is not None else {}))
        write_mask_video(folder / f"track__{by}.mkv", track)
        held_frames: list[int] = []
        left_frames: list[int] = []
        dropped: list[list[int]] = []
        refused: list[int] = []
        emptied_by_grade: list[int] = []
        if parts is not None:
            write_mask_video(folder / f"parts__{by}.mkv", parts)
            mine = [r for r in subject_rows(label, by, first, track, covered, parts)]
            if all(r["covered"] for r in mine):
                only = [[int(x) for x in (span.split("-") * 2)[:2]] for span in m["hold"].split("+")] if m.get("hold") else None
                held, held_frames, left_frames = held_parts(track, parts, mine, only, first)
                if label in class_maps:
                    names = _pack("sapiens2_parts").CLASS_NAMES
                    every, doubted, _ = held_parts(track, parts, mine)
                    made = [names.index(c) for c in made_of.get(label, [])]
                    if m.get("grade") != "off":
                        held, graded = grade_holds(parts, held, class_maps[label], made)
                        held_frames = [f for f, d in graded.items() if d["did"] == "filled"]
                        refused = [first + f for f, d in graded.items() if d["did"].startswith("saved")]
                        emptied_by_grade = [first + f for f, d in graded.items() if d["did"] == "emptied"]
                        (folder / "part_grades.json").write_text(json.dumps(
                            {"subject": label, "seen_by": by, "bar": HOLD_INSIDE, "part_is_made_of": made_of.get(label),
                             "frames": [{"source_frame": first + f, **d} for f, d in sorted(graded.items())]}, indent=1) + "\n")
                    others_now = [t for other, seen in sightings.items() if other != label and other in class_maps
                                  for t in [next(iter(seen.values()))[0]]]
                    what = under_doubted(made, names.index("Hair"), every, doubted, class_maps[label], others_now)
                    (folder / "doubted_frames.json").write_text(json.dumps(
                        {"subject": label, "seen_by": by, "what_each_means": UNDER,
                         "frames": [{"source_frame": first + f, "filled_in_parts_held": f in held_frames, **v} for f, v in sorted(what.items())]},
                        indent=1) + "\n")
                for span in (m["drop"].split("+") if m.get("drop") else []):
                    lo, hi = (int(x) for x in (span.split("-") * 2)[:2])
                    held[max(lo - first, 0):max(hi - first + 1, 0)] = False
                    dropped.append([lo, hi])
                held_masks[label] = held
                write_mask_video(folder / f"parts_held__{by}.mkv", held)
        if m.get("shots"):
            shutil.copyfile(m["shots"], folder / f"shots__{by}.json")
            beside = Path(m["shots"]).with_suffix(".md")
            if beside.is_file():
                shutil.copyfile(beside, folder / f"shots__{by}.md")
        posed = None
        if m.get("pose"):
            given = json.loads(Path(m["pose"]).read_text())
            write_table(folder / f"pose__{by}", pose_rows(given, label, first, frames, int(m["pose_at"]) if m.get("pose_at") else None))
            shutil.copyfile(m["pose"], folder / f"pose_table__{by}.json")
            posed = {"file": Path(m["pose"]).name, "hand_refinement": bool(given.get("hand_refinement")),
                     "hand_box_threshold_px": given.get("hand_box_threshold_px"), "camera": given.get("camera")}
        entry = next((s for s in manifest["subjects"] if s["label"] == label), None)
        if entry is None:
            entry = {"label": label, "colour": list(COLOURS[len(manifest["subjects"]) % len(COLOURS)]), "sightings": []}
            manifest["subjects"].append(entry)
        entry["sightings"].append({"by": by, "track": Path(m["track"]).name, "parts": Path(m["parts"]).name if m.get("parts") else None,
                                   "shots": bool(m.get("shots")), "first_source_frame": at,
                                   # a shot table counts from its own load's first frame, which need not be the mask video's
                                   "shots_first_source_frame": int(m.get("shots_at", at)),
                                   "frames_covered": int(covered.sum()), "pose": posed,
                                   "inputs": {k: input_facts(m[k]) for k in ("track", "parts", "classes", "held", "shots", "pose") if m.get(k)},
                                   "classes": Path(m["classes"]).name if m.get("classes") else None,
                                   "part_is_made_of": made_of.get(label),
                                   "parts_held_on_source_frames": frame_spans([first + f for f in held_frames]),
                                   "parts_doubted_and_left_on_source_frames": frame_spans([first + f for f in left_frames]),
                                   "parts_emptied_on_source_frames": dropped,
                                   "parts_fill_refused_by_grade_on_source_frames": frame_spans(refused),
                                   "parts_emptied_by_grade_on_source_frames": frame_spans(emptied_by_grade)})
        print(f"subject {label} seen by {by}: {int(covered.sum())} of {frames} frames covered, "
              f"mask on {int(track.reshape(frames, -1).any(1).sum())}", flush=True)
    for label, seen in sightings.items():
        rows = [row for by, (track, covered, parts) in seen.items() for row in subject_rows(label, by, first, track, covered, parts)]
        write_table(out / "subjects" / label / "per_frame", rows)
        if label in class_maps:
            # only other subjects that are things with a class map of their own: a kept-out mask is not a "who"
            theirs = {o: next(iter(x.values()))[0] for o, x in sightings.items() if o != label and o in class_maps}
            write_table(out / "subjects" / label / "segments_whose",
                        whose_rows(label, first, class_maps[label], next(iter(seen.values()))[0], theirs,
                                   _pack("sapiens2_parts").CLASS_NAMES))
    captured: dict = {}
    for name, r in [(n, {**x, "planned": False}) for n, x in runs] + [(n, {**x, "planned": True}) for n, x in plans]:
        label, others = r["subject"], [o for o in r.get("others", "").split("+") if o]
        kept_labels = [o for o in r.get("keep", "").split("+") if o] if r["planned"] else []
        for who in [label] + others + kept_labels:
            if who not in sightings:
                raise SystemExit(f"run {name!r} names subject {who!r}, which no --mask gives")
        lead = {who: next(iter(sightings[who].values())) for who in [label] + others + kept_labels}
        if r["planned"] and r.get("preview"):
            where = plan_file(r["preview"])
            region, carried, read, how = read_planned_regions(where, first, frames, size, r.get("at"))
            if region is None:
                raise SystemExit(f"run {name!r}: {how['refused']} ({where})")
            graph, settings = None, {k: how["source_settings"].get(k) for k in ("grow_pixels", "grow_by", "edge", "replace", "composite")}
            r = {**r, "margin": settings.get("grow_pixels") if settings.get("grow_by") in (None, "a fixed margin") else r.get("margin"),
                 "at": r.get("at", first)}
            how["load"] = {"first_frame": int(r["at"]), "frames": int(how["frames_written"])}
        elif r["planned"]:
            if "margin" not in r:
                raise SystemExit(f"plan {name!r} needs margin=PX: the region is worked out from the masks and it")
            graph, settings = None, {"grow_pixels": int(r["margin"]), "grow_by": "a fixed margin",
                                     "edge": "latent cells" if r.get("edge") == "cells" else "whole tokens"}
            track, covered, parts = lead[label]
            carried = parts if parts is not None and r.get("carried", "parts") == "parts" else track
            if r.get("carried") == "held":
                if label not in held_masks:
                    raise SystemExit(f"plan {name!r} asks for carried=held and {label!r} has no parts_held mask")
                carried = held_masks[label]
            region = planned_region(carried, [lead[o][0] for o in others], int(r["margin"]), _pack("video_mask").grow,
                                    whole_tokens=r.get("edge") != "cells", keep=[lead[o][0] for o in kept_labels])
            read, how = covered.copy(), {"read_from": "worked out from the saved masks: a plan, nothing rendered", "legend_px": None}
            if "at" in r or "frames" in r:
                # a load: the plan exists on its own frames only
                held = load_frames(first, frames, int(r.get("at", first)), int(r["frames"]) if "frames" in r else None)
                region, carried, read = region & held[:, None, None], carried & held[:, None, None], read & held
                how["load"] = {"first_frame": int(r.get("at", first)), "frames": int(r["frames"]) if "frames" in r else None,
                               **{k: (Path(r[k]).name if k != "text" else r[k]) for k in LOAD_GIVEN if r.get(k)}}
        else:
            graph = graph_of(r["render"])
            settings = source_settings(graph)
            saved = [] if r.get("region") == "review" else window_region_files(r["render"])
            region = None
            if saved:
                region, carried, read, how = read_saved_regions(saved, first, frames, size, r.get("at"), probe(r["render"])[2])
                if region is None:
                    print(f"run {name}: the windows' saved regions are not this render's ({how['refused']}); reading the review")
                elif Path(r["render"][:-len(".mp4")] + "_with_mask.mp4").is_file():
                    # both readers on the same frames: what the node saved against what the review picture gives back
                    back = read_region(r["render"], a.source, first, frames, size, r.get("at"))
                    how["readers"] = readers_agreement((region, carried, read), back[:3])
            if region is None:
                refused = (how or {}).get("refused") if saved else None
                region, carried, read, how = read_region(r["render"], a.source, first, frames, size, r.get("at"))
                if refused:
                    how["saved_regions_refused"] = refused
        folder = out / "runs" / name
        folder.mkdir(parents=True, exist_ok=True)
        entry = {"subject": label, "others": others, "region": region, "carried": carried, "read": read, "reached": {}, "back": {}}
        margin = r.get("margin", settings.get("grow_pixels"))
        fixed = settings.get("grow_by") in (None, "a fixed margin")
        if margin is not None and fixed and others:
            grow = _pack("video_mask").grow
            for o in others:
                entry["reached"][o], entry["back"][o] = taken_back(carried, lead[o][0], region, int(margin), grow)
        np.savez_compressed(folder / "region.npz", region=region, carried=np.packbits(carried, axis=-1), read=read,
                            **{f"given_back__{o}": cells for o, cells in entry["back"].items()})
        write_mask_video(folder / "region.mkv", cells_up(region))
        write_mask_video(folder / "carried.mkv", carried)
        write_table(folder / "per_frame", run_rows(name, label, first, region, carried, read))
        if class_maps:
            write_table(folder / "segments_in_region", segments_in_region(first, region, carried, class_maps,
                                                                           _pack("sapiens2_parts").CLASS_NAMES))
        if graph is not None:
            (folder / "graph.json").write_text(json.dumps(graph, indent=1) + "\n")
        captured[name] = entry
        manifest["runs"].append({"name": name, "planned": r["planned"], "render": None if r["planned"] else Path(r["render"]).name,
                                 "subject": label, "others": others, "keep": kept_labels,
                                 "windows": ({"window_frames": int(r["window"]), "context_frames": int(r.get("context", 0))}
                                             if r["planned"] and r.get("window") else window_settings(graph)),
                                 "carried_is": ("the held part" if r.get("carried") == "held" else "the part" if lead[label][2] is not None
                                                and r.get("carried", "parts") == "parts" else "the track") if r["planned"] and not r.get("preview") else "the mask each window was given" if how.get("files") else "read from the review",
                                 "first_source_frame": int(r.get("at", first)), "margin_px": None if margin is None else int(margin),
                                 "masked_source": settings,
                                 "given_back_worked_out": bool(entry["back"]), "frames_read": int(read.sum()), **how})
        print(f"{'plan' if r['planned'] else 'run'} {name}: region {'worked out' if r['planned'] and not r.get('preview') else 'read'} on "
              f"{int(read.sum())} of {frames} frames, mean share {float(region[read].mean()) if read.any() else 0.0:.4f}", flush=True)
    voice = read_voice(a.voice) if a.voice else None
    manifest["voice_table"] = Path(a.voice).name if a.voice else None
    cross = cross_rows(first, frames, sightings, captured, voice)
    # Who owns each pixel. A subject is a "who" when it has a class map, or when no subject has one: a kept-out
    # or keep mask given as a --mask is a union of things and owns nothing.
    whos = [label for label in sightings if label in class_maps] or list(sightings)
    if len(whos) > 1:
        owner, labels, shared, contested = owner_map({label: next(iter(sightings[label].values()))[0] for label in whos}, class_maps)
        np.savez_compressed(out / "owners.npz", owner=owner, labels=np.array(labels), nobody=NOBODY, contested=CONTESTED)
        for n, row in enumerate(cross):
            row["claimed_by_more_than_one_px"], row["contested_px"] = int(shared[n]), int(contested[n])
        manifest["owners"] = {"file": "owners.npz", "labels": labels, "nobody": NOBODY, "contested": CONTESTED,
                              "claimed_by_more_than_one_px": int(shared.sum()), "contested_px": int(contested.sum())}
        print(f"owners: {int(shared.sum())} pixels claimed by more than one track over the span, {int(contested.sum())} left contested", flush=True)
    write_table(out / "frames", cross)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (out / "README.md").write_text(readme(manifest))


def readme(m: dict) -> str:
    subjects = "\n".join(f"- `{s['label']}`: " + "; ".join(
        f"seen by `{x['by']}` ({x['track']}" + (f", parts {x['parts']}" if x["parts"] else "") + f", {x['frames_covered']} frames)"
        for x in s["sightings"]) for s in m["subjects"])
    runs = "\n".join(f"- `{r['name']}`: " + ("a PLAN, nothing rendered, for" if r["planned"] else f"`{r['render']}` regenerated")
                     + f" `{r['subject']}`"
                     + (f", kept out of its margin: {', '.join('`' + o + '`' for o in r['others'])}" if r["others"] else "")
                     + f"; region: {r['read_from']}" for r in m["runs"]) or "- none: masks only, nothing sampled"
    return f"""# Capture `{m['name']}`: `{m['clip']}`, source frames {m['first_frame']} to {m['first_frame'] + m['frames'] - 1}

Untracked on purpose (`data/` is ignored). Written by `bench/capture_masked_run.py files` at commit {m['commit']}
from saved videos: no server, no model. `manifest.json` has every input by file name. Row `frame` counts from
the span's first frame; `source_frame` is the clip's own.

## Subjects

{subjects}

## Runs

{runs}

## Files

- `frames.csv` / `.json`: one row a frame across subjects. `masks_overlap__a__b` (share of the frame both
  masks cover), `same_subject__a__x__y` (one subject's two sightings, intersection over union),
  `region_on__run__b` (share of b's mask inside the run's region), `region_cells_holding__run__b` (regenerated
  cells that also hold some of b), `margin_reached__run__b` and
  `cells_given_back__run__b` (where the run's margin reached b, and the cells of it left the source's),
  `regions_overlap_cells__run__run`, and `voiced` when a voice table was given.
- `subjects/<label>/per_frame.csv`: one row a frame and sighting, with `subject` and `seen_by`.
  `masks__<by>.npz`: `track`, `parts` (bits packed along the width), `covered`.
- `runs/<run>/per_frame.csv`: the region and the margin per frame; `region.npz`: `region` in cells of
  {m['cell_px']} pixels, `carried` (the mask the review tints as the one the source carries), `given_back__<label>`.

## What this does not hold

Anything inside the tracker; the part model's doubt, its matte and the frames it held; the token mask
itself (the region is read back from a compressed, tinted picture and is a cell coarse); the song node's
report. Gaps go in `data/CAPTURE_GAPS.md`.
"""


# ------------------------------------------------------------------ preflight

LEVELS = ("likely fine", "iffy", "likely to fail")
#: `preflight --gate`'s exit status when a flag at the top level stands with no override. Reasoned: not 1 (an
#: error) and not 2 (argparse's, and the sweep's "not run").
GATE_BLOCKED = 3


#: `verify`'s exit status when a check failed or could not be made. Reasoned: beside GATE_BLOCKED, not 1 or 2.
VERIFY_FAILED = 4
#: Mask videos of one sighting written further apart than this are from different runs. Reasoned, not
#: measured: a no-sampling preview writes all its masks within a few minutes; half an hour is far outside one.
WRITTEN_TOGETHER_S = 1800
#: The two readers of a run's region agree when at most this share of cells differs, and at most
#: `READERS_CARRIED_OFF` of the two carried masks' pixels, on the median frame, lie more than a pixel from the
#: other mask. Measured 2026-10-10 on five renders: a whole person, no cell of 433,152 to 473,088 differing and
#: nothing over a pixel off; a face of a few thousand pixels, about one cell a frame differing (the review
#: reader's own one-cell misread, `data/CAPTURE_GAPS.md` 32) and a thousandth of the pixels over a pixel off.
#: The bars sit a few times above those. The plain overlap is kept in the record and not judged: it falls
#: with the mask's size for the same edge (0.994 on the person, 0.973 on the face).
READERS_CELLS = 0.001
READERS_CARRIED_OFF = 0.01


def input_facts(path: str) -> dict:
    """What `verify` needs to know of one input file: its name, when it was written and, for a video, its size."""
    p = Path(path)
    facts = {"file": p.name, "written": datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")}
    if p.suffix.lower() in (".mkv", ".mp4", ".mov", ".webm"):
        w, h, n = probe(path)
        facts.update(size=[w, h], frames=n)
    return facts


def readers_agreement(saved: tuple, review: tuple) -> dict:
    """The region and carried mask a run's windows saved against those read back from its review, on the frames both read."""
    (region, carried, read), (region2, carried2, read2) = saved, review
    both = read & read2
    differ = (region != region2) & both[:, None, None]
    overlap = [float((a & b).sum() / max((a | b).sum(), 1)) for a, b, ok in zip(carried, carried2, both) if ok and (a.any() or b.any())]
    # an overlap falls with the mask's size for the same one-pixel edge, so a face reads lower than a whole
    # person for no fault; the pixels of either mask more than a pixel from the other do not
    import cv2
    near = np.ones((3, 3), np.uint8)
    far = [float(((a & ~cv2.dilate(b.astype(np.uint8), near).astype(bool)) | (b & ~cv2.dilate(a.astype(np.uint8), near).astype(bool))).sum()
                 / max((a | b).sum(), 1)) for a, b, ok in zip(carried, carried2, both) if ok and (a.any() or b.any())]
    return {"frames_compared": int(both.sum()), "cells": int(region[both].size), "cells_differing": int(differ.sum()),
            "frames_differing": frame_spans(np.nonzero(differ.any(axis=(1, 2)))[0].tolist()),
            "carried_over_a_pixel_off_median": round(float(np.median(far)), 4) if far else None,
            "carried_over_a_pixel_off_max": round(float(max(far)), 4) if far else None,
            "carried_overlap_median": round(float(np.median(overlap)), 4) if overlap else None,
            "carried_overlap_min": round(float(min(overlap)), 4) if overlap else None}


def verify_checks(manifest: dict, track_flags: list[dict]) -> list[dict]:
    """Every check `verify` makes, each `{"check", "about", "ok", "why"}`; `ok` is None when it could not be made.

    `track_flags` are `flag_track`'s for every sighting: a track empty inside a shot it was taken in."""
    out = []

    def add(check: str, about: str, ok: bool | None, why: str, **more) -> None:
        out.append({"check": check, "about": about, "ok": ok, "why": why, **more})

    size, first, frames = manifest["size"], manifest["first_frame"], manifest["frames"]
    for s in manifest["subjects"]:
        for seen in s["sightings"]:
            about, given = f"{s['label']} by {seen['by']}", seen.get("inputs")
            if not given:
                add("inputs_written_together", about, None, "this capture was made before inputs were recorded: run `files` again")
                continue
            videos = {k: v for k, v in given.items() if "size" in v}
            # a pose table may come from a pass of its own; the masks and the shot table are one preview's
            given = {k: v for k, v in given.items() if k != "pose"}
            times = sorted(datetime.datetime.fromisoformat(v["written"]) for v in given.values())
            spread = (times[-1] - times[0]).total_seconds()
            add("inputs_written_together", about, spread <= WRITTEN_TOGETHER_S,
                f"written within {int(spread)} s of each other" if spread <= WRITTEN_TOGETHER_S else
                f"written {int(spread)} s apart: {', '.join(v['file'] + ' at ' + v['written'][11:] for v in given.values())}. "
                "They are not one run's", seconds=int(spread))
            wrong = [f"{v['file']} is {v['size'][0]}x{v['size'][1]}" for v in videos.values() if v["size"] != size]
            add("inputs_at_the_canvas", about, not wrong, "; ".join(wrong) + f"; the capture is {size[0]}x{size[1]} and a mask of "
                "another size is resized without a word" if wrong else f"every mask video is {size[0]}x{size[1]}")
            at = seen["first_source_frame"]
            short = [f"{v['file']} ends on source frame {at + v['frames'] - 1}" for v in videos.values() if at + v["frames"] < first + frames]
            add("inputs_reach_the_span", about, not short and at <= first, ("; ".join(short) or f"the masks start on source frame {at}")
                + f"; the span is {first}-{first + frames - 1}" if short or at > first else f"every mask video covers {first}-{first + frames - 1}")
    for f in track_flags:
        add("track_where_the_shot_table_says_taken", f"{f['subject']} by {f['seen_by']}", False, f["why"], source_frames=f["source_frames"])
    if not track_flags:
        tables = [s["label"] for s in manifest["subjects"] if any(x.get("shots") for x in s["sightings"])]
        add("track_where_the_shot_table_says_taken", ", ".join(tables) or "no subject", True if tables else None,
            "no track is empty inside a shot it was taken in" if tables else "no sighting has a shot table: nothing says where a track should be")
    classed = {s["label"] for s in manifest["subjects"] if any(x.get("classes") for x in s["sightings"])}
    for run in manifest["runs"]:
        if run.get("planned") and run.get("carried_is") in ("the part", "the held part"):
            add("class_map_for_a_planned_part", run["name"], run["subject"] in classed,
                f"{run['subject']} has a class map" if run["subject"] in classed else
                f"plan {run['name']} carries {run['subject']}'s part and {run['subject']} has no class map: what the part is made of, "
                "and what else lies inside the region, cannot be read")
        if run.get("planned"):
            continue
        got = run.get("readers")
        if run.get("saved_regions_refused"):
            add("region_saved_against_region_read_back", run["name"], False, "the windows' saved files are not this render's: " + run["saved_regions_refused"])
        elif not run.get("files"):
            add("region_saved_against_region_read_back", run["name"], None,
                "the render saved no region files (made before the song node wrote them): its region is a reading of the review only")
        elif not got:
            add("region_saved_against_region_read_back", run["name"], None, "the render has no review video to read back, or `files` was run before it compared the two")
        else:
            share, off = got["cells_differing"] / max(got["cells"], 1), got.get("carried_over_a_pixel_off_median")
            if off is None:
                add("region_saved_against_region_read_back", run["name"], None, "this capture compared the carried masks by overlap only "
                    "(made before the size-free measure): run `files` again", **got)
                continue
            ok = share <= READERS_CELLS and off <= READERS_CARRIED_OFF
            add("region_saved_against_region_read_back", run["name"], ok,
                f"{got['cells_differing']} of {got['cells']} cells differ on {got['frames_compared']} frames; of the carried masks' pixels "
                f"{off:.2%} lie over a pixel from the other on the median frame (worst frame {got['carried_over_a_pixel_off_max']:.2%}; "
                f"plain overlap {got['carried_overlap_median']})"
                + ("" if ok else f"; frames that differ: {got['frames_differing'][:8]}"), **got)
    return out


def verify(a: argparse.Namespace) -> None:
    folder = Path(a.capture)
    m = json.loads((folder / "manifest.json").read_text())
    track_flags = []
    for s in m["subjects"]:
        rows = json.loads((folder / "subjects" / s["label"] / "per_frame.json").read_text())["rows"]
        for seen in s["sightings"]:
            path = folder / "subjects" / s["label"] / f"shots__{seen['by']}.json"
            if path.is_file():
                track_flags += flag_track(s["label"], seen["by"], [r for r in rows if r["seen_by"] == seen["by"]],
                                          json.loads(path.read_text()), seen.get("shots_first_source_frame", seen["first_source_frame"]))
    checks = verify_checks(m, track_flags)
    failed, unmade = [c for c in checks if c["ok"] is False], [c for c in checks if c["ok"] is None]
    verdict = "does not read what the nodes wrote" if failed else "not verified: a check could not be made" if unmade else "reads what the nodes wrote"
    (folder / "verify.json").write_text(json.dumps({"capture": m["name"], "verdict": verdict, "failed": len(failed), "not_made": len(unmade),
                                                    "written": datetime.datetime.now().isoformat(timespec="seconds"), "checks": checks}, indent=1) + "\n")
    for c in checks:
        print(f"{'ok  ' if c['ok'] else 'FAIL' if c['ok'] is False else '??  '}  {c['check']}  [{c['about']}]\n      {c['why']}")
    print(f"verify: {verdict}; wrote {folder / 'verify.json'}")
    if failed or unmade:
        sys.exit(VERIFY_FAILED)


def flag_key(flag: dict) -> str:
    """What a flag is about, apart from its number: the rule, the subject or run and other it names, its frames.

    Flag ids are the order of one preflight run. An override is held against this, so that a flag which
    names other frames on the next run is a flag nobody has overridden."""
    import hashlib
    about = [flag.get(k) for k in ("rule", "subject", "run", "other", "seen_by", "source_frames")]
    return hashlib.sha256(json.dumps(about, sort_keys=True).encode()).hexdigest()[:16]


def gate_verdict(flags: list[dict], outcomes: list[dict]) -> dict:
    """`clear` or `blocked`: a flag at the top level blocks until an outcome overrides the flag as it now is."""
    passed = {o.get("key") for o in outcomes if o.get("overridden")}
    top = [f for f in flags if f["level"] == LEVELS[2]]
    blocking = [f["id"] for f in top if flag_key(f) not in passed]
    return {"verdict": "blocked" if blocking else "clear", "blocking": blocking,
            "overridden": [f["id"] for f in top if flag_key(f) in passed]}


def read_outcomes(folder: Path) -> list[dict]:
    path = folder / "outcomes.json"
    return json.loads(path.read_text())["outcomes"] if path.is_file() else []


def frame_spans(frames: list[int], join: int = 2) -> list[list[int]]:
    """Sorted frame numbers as [first, last] runs, runs under `join` frames apart made one."""
    out: list[list[int]] = []
    for f in sorted(frames):
        if out and f - out[-1][1] <= join:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def recent_median(values: np.ndarray, reach: int = RECENT) -> np.ndarray:
    """Each frame's median over the frames within `reach` of it, itself left out; NaN where none is known."""
    out = np.full(len(values), np.nan)
    for n in range(len(values)):
        near = np.r_[values[max(n - reach, 0):n], values[n + 1:n + reach + 1]]
        near = near[~np.isnan(near)]
        if len(near):
            out[n] = np.median(near)
    return out


def shot_state(shot: dict) -> tuple[str, bool]:
    """A shot's state as the rules read it (`picked`, `taken` or `absent`) and whether a person typed it.

    A corrected shot's state carries the fact in its words (`taken (corrected)`), and its `corrected` field
    says what was typed."""
    state = shot["subject"]["state"]
    return state.split(" (")[0], bool(shot.get("corrected")) or "(corrected)" in state


def flag_shots(label: str, by: str, table: dict, at: int, not_in: list[list[int]]) -> tuple[list[dict], list[dict]]:
    """A shot table's risks: who was taken close to the line, taken where the caller says the subject is not,
    or called absent with somebody on screen. Frames in the table count from the load's first frame, `at`.

    An absence with people detected is UNRESOLVED until somebody answers it in writing, and is at the top
    level until then: a typed correction on the tracker (`shot N: person K`, or `shot N: none`) or `--not-in`
    for the whole shot. The rule was "iffy" until 2026-10-10, when a shot both trackers had called absent,
    each with its subject on screen, went through a day of renders with nothing laid on it and reached a
    viewer as the original."""
    flags, shots = [], []
    line = float(table["match"])
    for shot in table["shots"]:
        a, b = at + shot["first_frame"], at + shot["last_frame"]
        (state, typed), sim, people = shot_state(shot), shot["subject"].get("similarity"), len(shot.get("people", []))
        level = LEVELS[0]
        barred = [x for x in not_in if x[0] <= b and a <= x[1]]
        if state == "absent" and people:
            barred = [x for x in not_in if x[0] <= a and b <= x[1]]      # the whole shot, for an absence to be "as said"
        if state in ("taken", "picked") and barred:
            level = LEVELS[2]
            flags.append({"rule": "taken_where_not_expected", "level": level, "subject": label, "seen_by": by, "source_frames": [[a, b]],
                          "why": f"{label} is {state} in a shot the caller says they are not in", "figures": {"similarity": sim, "line": line}})
        elif state == "taken" and not typed and sim is not None and sim - line < NEAR_LINE:      # a typed take is not a guess
            level = LEVELS[1]
            flags.append({"rule": "taken_near_the_line", "level": level, "subject": label, "seen_by": by, "source_frames": [[a, b]],
                          "why": f"{label} was taken across a cut at {sim:.3f}, only {sim - line:.3f} above the line: "
                                 "it may be somebody else", "figures": {"similarity": sim, "line": line},
                          "threshold": {"NEAR_LINE": NEAR_LINE}})
        elif state == "absent" and people:
            # The score cannot settle this one. Measured on one clip, 2026-10-10: the subject was on screen in
            # two shots called absent, at 0.052 and 0.133 under the line (a cut to another framing lowers the
            # same person's score), and truly absent in three at 0.20 to 0.37 under. So the caller's word does:
            # a shot inside --not-in is fine, any other is a shot to look at, with the distance said.
            near = sim is not None and line - sim < NEAR_UNDER
            answered = "a correction typed on the tracker" if typed else "--not-in" if barred else None
            level = LEVELS[0] if answered else LEVELS[2]
            flags.append({"rule": "absent_with_people_on_screen", "level": level, "subject": label, "seen_by": by, "source_frames": [[a, b]],
                          "why": f"{label} is called absent with {people} detected; the closest scored "
                                 + (f"{sim:.3f} against a line of {line:.3f}" if sim is not None else "nothing")
                                 + (f": answered by {answered}" if answered else
                                    (": close enough to be them" if near else "")
                                    + ". UNRESOLVED: nothing is laid on this shot for them. Look at the tile and answer in writing: "
                                      "a correction on the tracker (`shot N: person K`, read off this tracker's own tile, or "
                                      "`shot N: none`), or --not-in for the whole shot"),
                          "figures": {"similarity": sim, "line": line, "people": people, "answered_by": answered},
                          "threshold": {"NEAR_UNDER": NEAR_UNDER}})
        shots.append({"subject": label, "seen_by": by, "source_frames": [a, b], "state": state, "similarity": sim, "level": level})
    return flags, shots


#: The pose table this reads (`body_pose.py::TABLE_SCHEMA`). Another version is refused, not guessed at.
POSE_SCHEMA = "h3_body_pose_table/1"
#: A body with over this share of its keypoints outside the frame is mostly out of the picture: those joints
#: are the model's guess. Reasoned, not measured: a third.
POSE_OUTSIDE = 1 / 3


def pose_state(keypoints_3d) -> dict:
    """What a body is doing on one frame, from its 70 keypoints in the camera's frame: the columns a state is read from.

    The angles are `bench/measure_subject_yaw.py`'s own (`yaw_of`, `head_lift`), one definition: 0 toward the
    camera, 90 side-on, 180 away, from the shoulders (`body_yaw`), the hips and the ears (`head_yaw`);
    `head_lift` is the nose above the line between the ears, in degrees. `*_wrist_to_nose` is a wrist's
    distance from the nose in units of the torso (hips to shoulders), so a hand at the face reads small at
    any size or distance. Whether the hand's own fingers can be trusted is the row's `*_hand_refined`."""
    import measure_subject_yaw as facing
    k, j = np.asarray(keypoints_3d, dtype=np.float64), facing.BODY_JOINTS
    torso = float(np.linalg.norm((k[j["left_shoulder"]] + k[j["right_shoulder"]]) / 2 - (k[j["left_hip"]] + k[j["right_hip"]]) / 2))
    head = {name: k[j[name]].tolist() for name in ("left_ear", "right_ear", "nose")}
    out = {"body_yaw": round(facing.yaw_of(k, j["left_shoulder"], j["right_shoulder"]), 1),
           "hip_yaw": round(facing.yaw_of(k, j["left_hip"], j["right_hip"]), 1),
           "head_yaw": round(facing.yaw_of(k, j["left_ear"], j["right_ear"]), 1),
           "head_lift": facing.head_lift({"in_body": head})}
    for side in ("left", "right"):
        out[f"{side}_wrist_to_nose"] = round(float(np.linalg.norm(k[j[f"{side}_wrist"]] - k[j["nose"]])) / max(torso, 1e-6), 3)
    return out


def pose_rows(table: dict, label: str, first: int, frames: int, at: int | None = None) -> list[dict]:
    """A subject's rows of a pose table over a span: one a frame the table has, in the span's numbering.

    The table numbers its own frames (`source_frame`); `at` moves its frame 0 to another source frame. A body
    belongs to the subject when it carries the subject's label; a table made under one other name throughout
    is taken as this subject's, and the row says so. Several bodies under one name on a frame are counted,
    and the row is the first's."""
    if table.get("schema") != POSE_SCHEMA:
        raise SystemExit(f"a pose table of schema {table.get('schema')!r}; this reads {POSE_SCHEMA!r}")
    names = {p.get("subject") for f in table["frames"] for p in f["people"]}
    other = next(iter(names)) if label not in names and len(names) == 1 else None
    shift = 0 if at is None else int(at) - int(table.get("first_source_frame", 0))
    total = len(table["frames"][0]["people"][0]["keypoints_2d"]) if table["frames"] and table["frames"][0]["people"] else 0
    rows = []
    for f in table["frames"]:
        n = int(f["source_frame"]) + shift
        if not first <= n < first + frames:
            continue
        mine = [p for p in f["people"] if p.get("subject") == (other or label)]
        row = {"frame": n - first, "source_frame": n, "bodies": len(mine), "bodies_in_frame": len(f["people"])}
        if mine:
            p = mine[0]
            row.update(box_source=p["box_source"], keypoints_outside_frame=p["keypoints_outside_frame"],
                       keypoints_outside_share=round(p["keypoints_outside_frame"] / max(total, 1), 3))
            for side in ("left_hand", "right_hand"):
                row[f"{side}_refined"] = bool(p[side]["decoder_used"])
                row[f"{side}_crop_px"] = p[side]["crop_side_px"]
            if p.get("keypoints_3d"):
                row.update(pose_state(p["keypoints_3d"]))
        if other:
            row["named_in_table"] = other
        rows.append(row)
    return rows


def flag_pose(label: str, by: str, pose: list[dict], masks: list[dict], others_on: set[int], refining: bool) -> list[dict]:
    """What a pose table says before its mesh is used as a motion video.

    `pose` are `pose_rows`, `masks` the subject's own per-frame rows, `others_on` the source frames another
    subject of the capture has a mask on, `refining` whether the table was made with hand refinement on.

    - `pose_fitted_to_the_whole_frame`: the body was fitted to a box that is the whole frame, not this
      subject's. With somebody else in the frame the body it returns can be theirs: top level then.
    - `several_bodies_under_one_name`: more than one body carries the subject's name on a frame.
    - `no_pose_where_the_subject_is`: the subject has a mask and the table has no body for it.
    - `hand_not_refined`: a hand the hand decoder did not refine while refinement was on: its fingers and
      wrist in the mesh are the body decoder's. One flag a side.
    - `pose_mostly_outside_the_frame`: over `POSE_OUTSIDE` of the keypoints fall outside the frame."""
    out = []

    def add(rule: str, level: str, frames_: list[int], why: str, **figures) -> None:
        if frames_:
            out.append({"rule": rule, "level": level, "subject": label, "seen_by": by, "source_frames": frame_spans(sorted(frames_)),
                        "why": why.format(n=len(frames_)), "figures": {"frames": len(frames_), **figures}})

    seen = {r["source_frame"]: r for r in pose}
    whole = [n for n, r in seen.items() if r.get("box_source") == "whole frame"]
    crowded = [n for n in whole if n in others_on]
    add("pose_fitted_to_the_whole_frame", LEVELS[2] if crowded else LEVELS[1], whole,
        f"{label}'s body was fitted to the whole frame on {{n}} frame(s), not to {label}'s own box"
        + (f"; on {len(crowded)} of them another subject is in the frame, and the body returned can be theirs. Give the pose "
           "node this subject's boxes" if crowded else "; nobody else is tracked there, so it is likely this subject's"),
        with_another_subject_in_frame=len(crowded))
    add("several_bodies_under_one_name", LEVELS[1], [n for n, r in seen.items() if r["bodies"] > 1],
        f"{{n}} frame(s) have more than one body named {label} in the pose table; the capture reads the first")
    add("no_pose_where_the_subject_is", LEVELS[1],
        [r["source_frame"] for r in masks if r.get("track_share") and not seen.get(r["source_frame"], {}).get("bodies")],
        f"{label} has a mask and no body in the pose table on {{n}} frame(s): the motion video has nobody there")
    if refining:
        for side in ("left_hand", "right_hand"):
            missed = [n for n, r in seen.items() if r.get(f"{side}_refined") is False]
            add("hand_not_refined", LEVELS[1], missed,
                f"{label}'s {side.replace('_', ' ')} was not refined by the hand decoder on {{n}} frame(s): the fingers and wrist "
                "of the mesh there are the body decoder's guess", side=side,
                median_crop_px=round(float(np.median([seen[n][f"{side}_crop_px"] for n in missed])), 1) if missed else None)
    add("pose_mostly_outside_the_frame", LEVELS[1], [n for n, r in seen.items() if r.get("keypoints_outside_share", 0) > POSE_OUTSIDE],
        f"over a third of {label}'s keypoints fall outside the frame on {{n}} frame(s): those joints are a guess",
        threshold_share=round(POSE_OUTSIDE, 3))
    return out


def flag_track(label: str, by: str, rows: list[dict], table: dict | None, at: int) -> list[dict]:
    """Frames with no mask inside a shot the subject was taken in."""
    if table is None:
        return []
    taken = [(at + s["first_frame"], at + s["last_frame"]) for s in table["shots"] if shot_state(s)[0] in ("taken", "picked")]
    empty = [r["source_frame"] for r in rows if r["covered"] and not r["track_share"]
             and any(a <= r["source_frame"] <= b for a, b in taken)]
    if not empty:
        return []
    return [{"rule": "track_empty_in_a_taken_shot", "level": LEVELS[2], "subject": label, "seen_by": by,
             "source_frames": frame_spans(empty), "why": f"{label} has no mask on {len(empty)} frame(s) of shots they were taken in: "
             "nothing regenerates there", "figures": {"frames": len(empty)}}]


def part_faults(rows: list[dict]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per rule, the rows (all covered, all with a part column) on which a part mask left its subject; and the
    share of the part inside the subject. One place for the rules, read by the flags and by the held part."""
    seen = np.array([bool(r["track_share"]) for r in rows])
    size = np.array([r["parts_share"] if r["parts_share"] else np.nan for r in rows], float)
    inside = np.array([r["parts_inside_track"] if r["parts_inside_track"] is not None else np.nan for r in rows], float)
    cx = np.array([r["parts_centre_in_box"][0] if r.get("parts_centre_in_box") else np.nan for r in rows], float)
    cy = np.array([r["parts_centre_in_box"][1] if r.get("parts_centre_in_box") else np.nan for r in rows], float)
    ratio = size / recent_median(size)
    moved = np.hypot(cx - recent_median(cx), cy - recent_median(cy))
    return {"part_empty_on_the_subject": seen & np.isnan(size), "part_off_the_subject": inside < PART_INSIDE,
            "part_changes_size": (ratio < PART_SIZE) | (ratio > 1.0 / PART_SIZE),
            "part_moves_on_the_subject": moved > PART_MOVE}, inside


#: The faults a held part repairs. Not a spill: a part lying partly off its subject is the right part in the
#: wrong outline, and the Masked Source already cuts a wired part to the subject's mask.
HELD_FOR = ("part_empty_on_the_subject", "part_changes_size", "part_moves_on_the_subject")


def held_parts(track: np.ndarray, parts: np.ndarray, rows: list[dict], only: list[list[int]] | None = None,
               first: int = 0) -> tuple[np.ndarray, list[int], list[int]]:
    """The part mask with doubted frames filled from their neighbours; the frames filled; the frames left.

    A doubted frame (`part_faults` under `HELD_FOR`) takes the shape of the nearest undoubted frame's part,
    shifted so its centre lies where a straight line between the undoubted frames either side puts it, and
    cut to this frame's subject. No scaling. A frame more than `HOLD_REACH` from an undoubted one on either
    side is left as it is and listed: a line across that long a gap is a guess.

    Why not the part node's own `hold`, which moves a part with the subject's box: tried first on one preview
    (2026-10-10, a face on a dancing subject). The box is the whole body's, so a raised arm stretched the face
    to several times its size and a passer-by in front shrank the box and dropped the part altogether.

    `only` limits it to source-frame spans the caller chose after looking, since a part can be rightly empty
    (the face turned away, somebody in front): on that same preview two of seven doubted stretches were."""
    import torch
    vm, sp = _pack("video_mask"), _pack("sapiens2_parts")
    faults, _ = part_faults(rows)
    bad = np.zeros(len(rows), bool)
    for rule in HELD_FOR:
        bad |= faults[rule]
    has = np.array([bool(r.get("parts_share")) for r in rows])
    good = np.nonzero(~bad & has)[0]
    out, held, left = parts.copy(), [], []
    if not len(good) or not bad.any():
        return out, held, left

    def centre(g):
        ys, xs = np.nonzero(parts[g])
        return np.array([xs.mean(), ys.mean()])

    for f in np.nonzero(bad)[0].tolist():
        if only is not None and not any(a <= first + f <= b for a, b in only):
            continue
        before, after = good[good < f], good[good > f]
        g0, g1 = (int(before[-1]) if len(before) else None), (int(after[0]) if len(after) else None)
        near = [g for g in (g0, g1) if g is not None and abs(g - f) <= HOLD_REACH]
        if not track[f].any() or len(near) < (2 if g0 is not None and g1 is not None else 1) or not near:
            left.append(f)
            continue
        g = min(near, key=lambda x: abs(x - f))
        if g0 is not None and g1 is not None:
            at = centre(g0) + (centre(g1) - centre(g0)) * (f - g0) / (g1 - g0)
        else:
            at = centre(g)
        dx, dy = (int(round(v)) for v in (at - centre(g)))
        moved = np.zeros_like(parts[g])
        h, w = moved.shape
        ys, xs = np.nonzero(parts[g])
        keep = (ys + dy >= 0) & (ys + dy < h) & (xs + dx >= 0) & (xs + dx < w)
        moved[ys[keep] + dy, xs[keep] + dx] = True
        on_subject = vm.grow(torch.from_numpy(track[f:f + 1]).to(torch.float32), sp.SUBJECT_MARGIN)[0].numpy() > 0.5
        out[f] = moved & on_subject
        held.append(f)
    return out, held, left


#: What a doubted part frame shows where the part should be, and what that says. Provenance: sixteen doubted
#: frames of one preview read by eye, 2026-10-10 (a face on a moving subject); the split named the right one of
#: "the model lost it", "turned away" and "something of theirs is in front" on fourteen. The two it got wrong
#: were the subject entering and leaving at the frame's edge, read as lost.
UNDER = {"the part": "the part is there; its size or place changed. Leave it as saved",
         "nothing labelled": "the part model labelled nothing there: it lost the part. A fill is likely right",
         "their own hair": "their own hair is there: turned away. Do not fill",
         "their other classes": "something else of theirs is there (a hand, clothing): covered or moved. Do not fill",
         "another subject": "another subject is there: hidden behind them. Do not fill"}


def under_doubted(parts_made_of: list[int], hair: int, filled: np.ndarray, frames: list[int], classes: np.ndarray,
                  others: list[np.ndarray]) -> dict[int, dict]:
    """For each doubted frame, what the class map shows under the shape a fill would put there (`held_parts`
    run with no limit): shares of the part's own classes, hair, the subject's other classes, another subject,
    and nothing. The largest names the frame (`UNDER`)."""
    out = {}
    for f in frames:
        where = filled[f]
        if not where.any():
            continue
        c = classes[f][where]
        mine = np.isin(c, parts_made_of)
        share = {"the part": float(mine.mean()), "their own hair": float((c == hair).mean()),
                 "their other classes": float(((c > 0) & ~mine & (c != hair)).mean()),
                 "another subject": max([float(o[f][where].mean()) for o in others], default=0.0),
                 "nothing labelled": float((c == 0).mean())}
        out[f] = {"mostly": max(share, key=share.get), **{k: round(v, 3) for k, v in share.items()}}
    return out


def flag_parts(label: str, by: str, rows: list[dict]) -> list[dict]:
    """A part mask that left its subject: spilled off the subject's mask, changed size against its own recent
    frames, moved within the subject's box, or is empty while the subject is there.

    Run on one clip the day it was written (a face-and-neck part on a small subject under a hat): the size and
    the move rules together named the stretches a person had found by eye where the part sat on something
    else; neither says WHAT it sat on, which needs the part model's own class map."""
    rows = [r for r in rows if r["covered"] and "parts_share" in r]
    if not rows:
        return []
    src = np.array([r["source_frame"] for r in rows])
    faults, inside = part_faults(rows)
    out = []

    def add(rule, level, where, why, **figures):
        if where.any():
            out.append({"rule": rule, "level": level, "subject": label, "seen_by": by, "source_frames": frame_spans(src[where].tolist()),
                        "why": why.format(n=int(where.sum())), "figures": {"frames": int(where.sum()), **figures}})

    add("part_empty_on_the_subject", LEVELS[2], faults["part_empty_on_the_subject"],
        label + "'s part mask is empty on {n} frame(s) where the subject is there: nothing of the part regenerates")
    add("part_off_the_subject", LEVELS[1], faults["part_off_the_subject"],
        label + "'s part mask lies partly off the subject's own mask on {n} frame(s)", threshold_PART_INSIDE=PART_INSIDE,
        lowest=round(float(np.nanmin(inside)), 3) if np.isfinite(inside).any() else None)
    add("part_changes_size", LEVELS[1], faults["part_changes_size"],
        label + "'s part mask is under half or over twice its recent size on {n} frame(s)", threshold_PART_SIZE=PART_SIZE)
    add("part_moves_on_the_subject", LEVELS[1], faults["part_moves_on_the_subject"],
        label + "'s part mask sits somewhere else in the subject's box than on the frames around it, on {n} frame(s): "
        "it may be on something that is not the part", threshold_PART_MOVE=PART_MOVE)
    return out


def flag_runs(manifest: dict, folder: Path) -> list[dict]:
    """A run's region over another subject, and a region that is mostly not its subject."""
    cross = json.loads((folder / "frames.json").read_text())["rows"]
    labels = [s["label"] for s in manifest["subjects"]]
    out = []
    for run in manifest["runs"]:
        name, kind = run["name"], "plan" if run.get("planned") else "run"
        for other in labels:
            col = f"region_on__{name}__{other}"
            hit = [(r["source_frame"], r[col]) for r in cross if r.get(col) is not None and r[col] > REGION_ON_OTHER]
            if hit:
                worst = max(v for _, v in hit)
                held = sum(1 for r in cross if r.get(f"cells_given_back__{name}__{other}"))
                out.append({"rule": "region_over_another_subject", "level": LEVELS[2] if worst > 5 * REGION_ON_OTHER else LEVELS[1],
                            "subject": run["subject"], "run": name, "other": other, "source_frames": frame_spans([f for f, _ in hit]),
                            "why": f"{kind} {name} regenerates up to {100 * worst:.0f}% of {other}'s mask: {other} is drawn again "
                                   f"there by {run['subject']}'s pass" + (f" (margin cells were given back on {held} frames)" if held else ""),
                            "figures": {"frames": len(hit), "worst_share": round(worst, 3)}, "threshold": {"REGION_ON_OTHER": REGION_ON_OTHER}})
        rows = json.loads((folder / "runs" / name / "per_frame.json").read_text())["rows"]
        outside = [(r["source_frame"], r["margin_share"] / r["region_share"]) for r in rows if r.get("region_share")]
        hit = [(f, v) for f, v in outside if v > NOT_SUBJECT]
        if hit:
            typical = float(np.median([v for _, v in outside]))
            out.append({"rule": "region_mostly_not_the_subject", "level": LEVELS[1], "subject": run["subject"], "run": name,
                        "source_frames": frame_spans([f for f, _ in hit], join=RECENT),
                        "why": f"{100 * typical:.0f}% of what {kind} {name} regenerates on a typical frame is not {run['subject']} "
                               f"(over half on {len(hit)} of {len(outside)} frames): background and things near them are drawn "
                               "again from the text, so name them in it or keep them out",
                        "figures": {"frames": len(hit), "of": len(outside), "median_share_not_subject": round(typical, 3)},
                        "threshold": {"NOT_SUBJECT": NOT_SUBJECT}})
    return out


def flag_segments(manifest: dict, folder: Path) -> list[dict]:
    """What the class map adds: a part that is not there to be replaced, and what else is inside a region.

    `part_not_visible`: the classes a part is made of cover under `PART_SIZE` of their own recent area (or
    nothing) while the subject's mask is there: the face is turned away or something is in front of it. It is
    the answer to "is this empty part a fault": no, and a fill would paint one in.
    `segment_inside_region`: a segment that is not the carried part lies inside a run's region, another
    subject's or the subject's own (hair, an earring, a hand): it is drawn again from the text and the still.
    One flag a segment, with its frames and the most cells it took."""
    out = []
    for s in manifest["subjects"]:
        seen = s["sightings"][0]
        path = folder / "subjects" / s["label"] / "segments.json"
        if not path.is_file() or not seen.get("part_is_made_of"):
            continue
        rows = json.loads(path.read_text())["rows"]
        track = {r["frame"]: r for r in json.loads((folder / "subjects" / s["label"] / "per_frame.json").read_text())["rows"]
                 if r["seen_by"] == seen["by"]}
        px = np.array([sum(r.get(f"{s['label']}.{c}", 0) for c in seen["part_is_made_of"]) for r in rows], float)
        usual = recent_median(np.where(px > 0, px, np.nan))
        there = np.array([bool(track[r["frame"]].get("track_share")) for r in rows])
        gone = there & ((px == 0) | (px < PART_SIZE * usual))
        if gone.any():
            src = np.array([r["source_frame"] for r in rows])
            out.append({"rule": "part_not_visible", "level": LEVELS[1], "subject": s["label"], "seen_by": seen["by"],
                        "segment": [f"{s['label']}.{c}" for c in seen["part_is_made_of"]],
                        "source_frames": frame_spans(src[gone].tolist()),
                        "why": f"{s['label']}'s {', '.join(seen['part_is_made_of'])} is absent or under half its recent size on "
                               f"{int(gone.sum())} frame(s) while {s['label']} is there: turned away or hidden. There is "
                               "nothing to replace on them; do not fill the part there",
                        "figures": {"frames": int(gone.sum()), "absent": int((there & (px == 0)).sum())}, "threshold": {"PART_SIZE": PART_SIZE}})
    relied_on = {f"{s['label']}.{c}" for s in manifest["subjects"] for c in (s["sightings"][0].get("part_is_made_of") or [])}
    for run in manifest["runs"]:
        path = folder / "runs" / run["name"] / "segments_in_region.json"
        if path.is_file():
            relied_on |= {k[:-len("__px")] for r in json.loads(path.read_text())["rows"] for k in r if k.endswith("__px") and r[k] >= SEGMENT_PX}
    for s in manifest["subjects"]:
        path = folder / "subjects" / s["label"] / "segments_whose.json"
        if not path.is_file():
            continue
        by_segment: dict = {}
        for r in json.loads(path.read_text())["rows"]:
            if r["segment"] in relied_on and r["px"] >= SEGMENT_PX and r["in_own_track"] < OWN_SHARE * r["px"]:
                by_segment.setdefault(r["segment"], []).append(r)
        for segment, hit in sorted(by_segment.items()):
            worst = max(hit, key=lambda r: r["in_other_track"] / r["px"])
            out.append({"rule": "segment_mostly_outside_its_own_track", "level": LEVELS[1], "subject": s["label"], "segment": segment,
                        "source_frames": frame_spans([r["source_frame"] for r in hit], join=RECENT),
                        "why": f"on {len(hit)} frame(s) most of {segment} is not inside {s['label']}'s own tracked mask"
                               + (f"; at worst {100 * worst['in_other_track'] / worst['px']:.0f}% of it is inside {worst['other']}'s "
                                  f"(source frame {worst['source_frame']}): it may be theirs" if worst["in_other_track"] else
                                  ": it lies in the margin round the outline") + ". A class says what a thing is, not whose",
                        "figures": {"frames": len(hit), "worst_share_in_another": round(worst["in_other_track"] / worst["px"], 3),
                                    "other": worst["other"]}, "threshold": {"OWN_SHARE": OWN_SHARE}})
    for run in manifest["runs"]:
        path = folder / "runs" / run["name"] / "segments_in_region.json"
        if not path.is_file():
            continue
        rows = json.loads(path.read_text())["rows"]
        own = next((s["sightings"][0].get("part_is_made_of") or [] for s in manifest["subjects"] if s["label"] == run["subject"]), [])
        kind = "plan" if run.get("planned") else "run"
        for key in sorted({k[:-len("__cells")] for r in rows for k in r if k.endswith("__cells")}):
            label, cls = key.split(".", 1)
            if label == run["subject"] and cls in own:
                continue                               # the carried part's own classes just outside its mask: the edge
            cells = [(r["source_frame"], r.get(f"{key}__cells", 0), r.get(f"{key}__px", 0)) for r in rows]
            hit = [(f, c, p) for f, c, p in cells if p >= SEGMENT_PX]
            if not hit:
                continue
            other = label != run["subject"]
            out.append({"rule": "segment_inside_region", "level": LEVELS[1], "subject": run["subject"], "run": run["name"],
                        "segment": key, "source_frames": frame_spans([f for f, _, _ in hit], join=RECENT),
                        "why": f"{key} lies inside what {kind} {run['name']} regenerates on {len(hit)} frame(s), up to "
                               f"{max(c for _, c, _ in hit)} cells: it is drawn again without being what is replaced"
                               + ("; name it in the text or keep it out" if not other else f"; {label}'s own pass must win there, or keep it out"),
                        "figures": {"frames": len(hit), "most_cells": max(c for _, c, _ in hit), "median_px": int(np.median([p for _, _, p in hit]))},
                        "threshold": {"SEGMENT_PX": SEGMENT_PX}})
    return out


def lent_frames(carried: np.ndarray, region: np.ndarray, read: np.ndarray) -> list[int]:
    """The frames that have a region and no mask of their own: lent one by their latent step.

    The sampler's region is one per latent step, made from every frame of the step. A frame whose own mask
    is empty (the part emptied because the face is turned away, a subject gone for a frame) inside a step
    whose other frames have one is regenerated all the same."""
    return np.nonzero(read & region.any(axis=(1, 2)) & ~carried.any(axis=(1, 2)))[0].tolist()


def flag_lent(manifest: dict, folder: Path) -> list[dict]:
    """A run or a plan read from the node's files that regenerates on frames with no mask of their own.

    Measured 2026-10-10 on two face-only renders: six such frames, each beside a turn; the render was 7 to 25
    grey levels from the source inside the lent region against a floor of 2 to 3, and on one of them a face
    was drawn under a hat brim where the source shows none. A plan worked out from the masks (`--plan`)
    cannot show it: its region is per frame."""
    first, w, out = manifest["first_frame"], manifest["size"][0], []
    for run in manifest["runs"]:
        saved = np.load(folder / "runs" / run["name"] / "region.npz")
        lent = lent_frames(np.unpackbits(saved["carried"], axis=-1)[..., :w].astype(bool), saved["region"], saved["read"])
        if lent:
            kind = "plan" if run.get("planned") else "run"
            out.append({"rule": "region_on_a_frame_with_no_mask", "level": LEVELS[1], "subject": run["subject"], "run": run["name"],
                        "source_frames": frame_spans([first + f for f in lent]),
                        "why": f"{kind} {run['name']}: {len(lent)} frame(s) have no mask of {run['subject']}'s and are regenerated all the "
                               "same, because their latent step holds frames that do have one. If the mask was emptied there on "
                               "purpose (turned away, hidden), the render can draw the part where the source shows none: look at "
                               "these frames, or empty the step's other frames too",
                        "figures": {"frames": len(lent)}})
    return out


def flag_held_tail(manifest: dict, folder: Path) -> list[dict]:
    """The invariant, checked before anything samples: in no planned window is the model shown, as a frame to
    keep, a picture of the subject it is replacing.

    From the node's own plan (`--run NAME:preview=`), per window: the frames held past the load's end that the
    model sees clean, when the subject's mask is on the load's last frame. Top level, always: on 2026-10-10 two
    per-shot loads of 29 frames in a 141-frame window (79% held) both ended on the original: one faded to it
    over its last twenty frames with the subject at 1.5% of the frame, the other over its last four on a large
    face; both had the source at the last frame. The remedy is named: the Masked Source's `held_tail` with
    the region open, which the plan then shows as no held frame clean. A plan worked out from the masks
    (`--plan`) has no windows and cannot be checked: the rule says so once."""
    first, out, blind = manifest["first_frame"], [], []
    for run in manifest["runs"]:
        if not run.get("planned"):
            continue
        if not run.get("planned_windows"):
            blind.append(run["name"])
            continue
        rows = json.loads((folder / "subjects" / run["subject"] / "per_frame.json").read_text())["rows"]
        seen = next(s for s in manifest["subjects"] if s["label"] == run["subject"])["sightings"][0]["by"]
        shares = [r["track_share"] for r in rows if r["seen_by"] == seen and r.get("track_share")]
        size = float(np.median(shares)) if shares else None
        for win in run["planned_windows"]:
            clean = win.get("held_frames_shown_clean", 0)
            if not clean or not win.get("subject_on_the_last_real_frame"):
                continue
            last = win["writes_source_frames"][1]
            step = win.get("step_with_real_and_held_source_frames")
            out.append({"rule": "original_shown_in_the_held_tail", "level": LEVELS[2], "subject": run["subject"], "run": run["name"],
                        "source_frames": [step or [last, last]],
                        "why": f"plan {run['name']} window {win['window']}: {clean} of its {win['frames']} frames ({clean / win['frames']:.0%}) are "
                               f"the load's last frame held past its end, shown to the model clean, and {run['subject']} is on that frame"
                               + (f" ({size:.2%} of the frame)" if size is not None else "") + ": the model is given a picture of the subject "
                               "it is replacing as a frame to keep, and the render slides to it as the load ends"
                               + (f"; the latent step of source frames {step[0]}-{step[1]} is part held and goes first" if step else "")
                               + ". Remedy: `held_tail` with its region open on the Masked Source, or a load that fills its window",
                        "figures": {"window": win["window"], "window_frames": win["frames"], "held_frames": win["held_frames"],
                                    "held_share": win["held_share"], "held_frames_shown_clean": clean,
                                    "subject_share_of_frame_median": None if size is None else round(size, 5)}})
    if blind:
        out.append({"rule": "held_tail_not_checked", "level": LEVELS[1], "subject": None, "run": ", ".join(blind),
                    "source_frames": [[first, first]],
                    "why": f"plan(s) {', '.join(blind)} are worked out from the masks and have no windows: whether a window would hold "
                           "the load's last frame, with the subject on it, past the load's end cannot be checked. Gate on the node's "
                           "own plan (`--run NAME:preview=<windows folder>`)",
                    "figures": {"plans": len(blind)}})
    return out


#: A continuation whose source turns its head this far from where it faces on the last kept frame, within the
#: window's new frames, is asked to leave its kept frames' pose. Inherited: `bench/measure_subject_yaw.py`'s
#: tolerance for "faces where the source faces". Measured once (2026-10-10): a turn of about 75 degrees starting
#: on a window's first new frames was missed by every render that kept frames of the subject turned away, at
#: two sizes of motion video and with two patches, and made by the one load with no kept frames.
KEPT_TURN = 45.0


def turn_from_kept(pose: dict[int, dict], last_kept: int, new: tuple[int, int]) -> dict | None:
    """How far the source's pose over a window's new frames is from its pose on the last kept frame.

    `pose` is `pose_rows` by source frame (with `pose_state`'s columns); `new` the first and last new source
    frame. None when the table has no facing on the last kept frame or on any new frame. Returns the largest
    turn of the head (and the frame it is reached on), the largest change of the chin's lift, and how much
    nearer the nose either wrist comes."""
    at = pose.get(last_kept, {})
    ahead = [pose[f] for f in range(new[0], new[1] + 1) if f in pose and pose[f].get("head_yaw") is not None]
    if at.get("head_yaw") is None or not ahead:
        return None
    turn = [abs((r["head_yaw"] - at["head_yaw"] + 180.0) % 360.0 - 180.0) for r in ahead]
    hand = lambda r: min(r["left_wrist_to_nose"], r["right_wrist_to_nose"])      # noqa: E731
    return {"head_turn_degrees": round(max(turn), 1), "reached_on_source_frame": ahead[int(np.argmax(turn))]["source_frame"],
            "chin_change_degrees": round(max(abs(r["head_lift"] - at["head_lift"]) for r in ahead), 1),
            "a_wrist_nearer_the_nose_by": round(hand(at) - min(hand(r) for r in ahead), 2)}


def flag_kept(manifest: dict, folder: Path) -> list[dict]:
    """A planned continuation whose kept frames show the subject in a pose the motion ahead leaves.

    Kept frames show the NEW subject, so they are not the original in the picture; what they pin is the pose.
    From the node's plan (which window keeps how many frames) and the subject's pose table: the source's head
    on the window's new frames against the last kept frame. Iffy at `KEPT_TURN` or more, with the chin and
    the wrists as figures. Without a pose table that carries 3D keypoints it cannot be checked, and says so."""
    out = []
    for run in manifest["runs"]:
        wins = [x for x in run.get("planned_windows") or [] if x.get("kept_frames")]
        if not run.get("planned") or not wins:
            continue
        seen = next(s for s in manifest["subjects"] if s["label"] == run["subject"])["sightings"][0]["by"]
        path = folder / "subjects" / run["subject"] / f"pose__{seen}.json"
        pose = {r["source_frame"]: r for r in json.loads(path.read_text())["rows"]} if path.is_file() else {}
        for win in wins:
            last_kept = win["first_frame"] + win["kept_frames"] - 1
            found = turn_from_kept(pose, last_kept, tuple(win["writes_source_frames"]))
            if found is None:
                out.append({"rule": "kept_frames_not_checked", "level": LEVELS[1], "subject": run["subject"], "run": run["name"],
                            "source_frames": [[last_kept, last_kept]],
                            "why": f"plan {run['name']} window {win['window']} keeps {win['kept_frames']} frames and {run['subject']} has no "
                                   "pose table with 3D keypoints over them (`pose=` on --mask): whether the motion ahead leaves the "
                                   "kept frames' pose cannot be checked",
                            "figures": {"window": win["window"], "kept_frames": win["kept_frames"]}})
            elif found["head_turn_degrees"] >= KEPT_TURN:
                out.append({"rule": "kept_frames_far_from_the_pose_ahead", "level": LEVELS[1], "subject": run["subject"], "run": run["name"],
                            "source_frames": [[win["writes_source_frames"][0], found["reached_on_source_frame"]]],
                            "why": f"plan {run['name']} window {win['window']} keeps {win['kept_frames']} frames ending on source frame {last_kept}, "
                                   f"and over its new frames the source's head turns {found['head_turn_degrees']:.0f} degrees from where it faces "
                                   f"there (by source frame {found['reached_on_source_frame']}; the chin moves {found['chin_change_degrees']:.0f} "
                                   f"degrees, a wrist comes {found['a_wrist_nearer_the_nose_by']} torso lengths nearer the nose). A render "
                                   "follows its kept frames over its motion video: start a load of its own at the turn, keep fewer "
                                   "frames, or noise the kept ones (`context_noise`)",
                            "figures": {"window": win["window"], "kept_frames": win["kept_frames"], **found}, "threshold": {"KEPT_TURN": KEPT_TURN}})
    return out


def flag_mouth(manifest: dict, folder: Path) -> list[dict]:
    """A subject of a run whose mouth is open with no voice on a run of frames: a state a pass has no channel for.

    The source's own openings (`mouth_openings` on its class map) against the capture's voice table, by
    `open_runs`. Iffy: measured once (2026-10-10), a face-only render kept the mouth's timing and reached
    about six tenths of the source's opening on such a run. Needs the subject's class map and `--voice`."""
    out, w, first = [], manifest["size"][0], manifest["first_frame"]
    cross = json.loads((folder / "frames.json").read_text())["rows"]
    voiced = np.array([np.nan if r.get("voiced") in (None, "") else float(r["voiced"]) for r in cross])
    names = _pack("sapiens2_parts").CLASS_NAMES
    for label in sorted({run["subject"] for run in manifest["runs"]}):
        seen = next(s for s in manifest["subjects"] if s["label"] == label)["sightings"][0]["by"]
        classes = folder / "subjects" / label / f"classes__{seen}.npz"
        if not classes.is_file() or not (voiced == 0).any():
            continue
        saved = np.load(folder / "subjects" / label / f"masks__{seen}.npz")
        face = np.unpackbits(saved["parts" if "parts" in saved.files else "track"], axis=-1)[..., :w].astype(bool)
        opening, _ = mouth_openings(class_mask_of(np.load(classes)["classes"], list(MOUTH_CLASSES), names), face)
        for run in open_runs(opening, voiced, {}, first)["runs"]:
            out.append({"rule": "mouth_open_with_no_voice", "level": LEVELS[1], "subject": label, "source_frames": [run["source_frames"]],
                        "why": f"{label}'s mouth is open on {run['frames']} frames with no voice on the track (opening "
                               f"{run['source_opening_median']} at the median): nothing a pass is given says so unless a motion "
                               "video carries the mouth. Expect the timing kept and the opening smaller",
                        "figures": {"frames": run["frames"], "source_opening_median": run["source_opening_median"]},
                        "threshold": {"OPEN_SHARE": OPEN_SHARE, "OPEN_RUN": OPEN_RUN}})
    return out


def flag_keep(manifest: dict, folder: Path) -> list[dict]:
    """Kept pixels of the original inside the subject's own part: expect the original back.

    A run whose graph wires the Masked Source's `keep`, on frames where cells holding the carried part were not
    regenerated: those cells show the original's own pixels inside the thing being replaced, and the model
    draws the rest to match them. Provenance, one pair of renders, 2026-10-10: a face pass with a `keep` on an
    earring left 6 to 7% of the face's cells as the original's cheek, and on the pixels it was free to redraw
    the face fell from 13 grey levels off the source to about 6, with stills that read as the original; the
    same pass without `keep` stayed at 13 to 15. The lane's 2026-10-07 record has the same thing for a kept
    body beside a head. For a rendered run the region is the review's; for a plan, `keep=` on the plan."""
    out = []
    for run in manifest["runs"]:
        graph = folder / "runs" / run["name"] / "graph.json"
        if run.get("planned"):
            wired = bool(run.get("keep"))
        else:
            wired = graph.is_file() and any(node.get("class_type") == "MiniMaxH3MaskedSource" and node["inputs"].get("keep") is not None
                                            for node in json.loads(graph.read_text()).values())
        if not wired:
            continue
        z = np.load(folder / "runs" / run["name"] / "region.npz")
        w = manifest["size"][0]
        # the part as the capture's own mask has it: the review's carried mask is cut to the region when it is
        # read, so it cannot show a cell that was kept
        seen = next(s for s in manifest["subjects"] if s["label"] == run["subject"])["sightings"][0]
        saved = np.load(folder / "subjects" / run["subject"] / f"masks__{seen['by']}.npz")
        whole = run.get("carried_is") == "the track" or "parts" not in saved.files
        part = np.unpackbits(saved["track" if whole else "parts"], axis=-1)[..., :w].astype(bool)
        carried = cells_any(part)
        kept = (carried & ~z["region"]).sum(axis=(1, 2))
        share = kept / np.maximum(carried.sum(axis=(1, 2)), 1)
        hit = np.nonzero((share > KEPT_IN_PART) & z["read"])[0]
        if len(hit):
            first = manifest["first_frame"]
            out.append({"rule": "kept_pixels_inside_the_part", "level": LEVELS[2], "subject": run["subject"], "run": run["name"],
                        "source_frames": frame_spans((hit + first).tolist(), join=RECENT),
                        "why": f"{'plan' if run.get('planned') else 'run'} {run['name']} wires `keep`, and on {len(hit)} frame(s) up to {100 * share.max():.0f}% of the cells "
                               f"holding {run['subject']}'s own part were kept as the original's pixels: expect the original's "
                               "look to come back in the rest of the part",
                        "figures": {"frames": int(len(hit)), "worst_share": round(float(share.max()), 3),
                                    "median_share": round(float(np.median(share[hit])), 3)}, "threshold": {"KEPT_IN_PART": KEPT_IN_PART}})
    return out


def cut_frames(cuts: list[int], first: int, present: np.ndarray) -> tuple[list[int], list[int]]:
    """The frames of a load that a latent step lays a subject's region on across a cut, and the frames of a step
    the subject is on BOTH sides of a cut in. Both in the numbering `cuts` and `first` are given in.

    `present` is whether the subject's mask is non-empty on each frame of the load, from its first frame. The
    arithmetic is `loop_plan.split_steps`, which owns it: the steps' edges are fixed for a whole load, counted
    from its first frame, whatever the windows, so no window plan is needed. `across` are the frames the node's
    `video_mask.cut_gate` leaves as the source; a render made before that gate has them repainted. `shared`
    are frames where each side of the cut is given the other side's region too, which no gate covers."""
    on = [(first + lo, first + hi) for lo, hi in frame_spans(np.nonzero(present)[0].tolist(), join=1)]
    split = _pack("loop_plan").split_steps(cuts, first_frame=first, present=on, frames=len(present))
    return sorted(f for s_ in split for f in s_["across"]), sorted(f for s_ in split for f in s_["shared"])


def flag_cuts(manifest: dict, folder: Path) -> list[dict]:
    """A run's region carried across a cut by a latent step: the masks, the shot table's cuts and the load's
    first frame are all it needs."""
    first, out = manifest["first_frame"], []
    for run in manifest["runs"]:
        seen = next(s for s in manifest["subjects"] if s["label"] == run["subject"])["sightings"][0]
        table = folder / "subjects" / run["subject"] / f"shots__{seen['by']}.json"
        if not table.is_file():
            # cuts are the source's, the same in every tracker's table: any subject's table in the capture will do
            for other in manifest["subjects"]:
                seen = other["sightings"][0]
                table = folder / "subjects" / other["label"] / f"shots__{seen['by']}.json"
                if table.is_file():
                    break
        if not table.is_file():
            continue
        cuts = [seen.get("shots_first_source_frame", seen["first_source_frame"]) + c for c in json.loads(table.read_text())["cuts"]]
        w = manifest["size"][0]
        present = np.unpackbits(np.load(folder / "runs" / run["name"] / "region.npz")["carried"], axis=-1)[..., :w].any(axis=(1, 2))
        # the load's own first frame: a run's render may start before the span this capture covers
        load_first = int(run.get("first_source_frame", first))
        across, shared = cut_frames(cuts, load_first, from_the_load(present, first, load_first, (run.get("load") or {}).get("frames")))
        across, shared = [f for f in across if f >= first], [f for f in shared if f >= first]
        kind = "plan" if run.get("planned") else "run"
        if across:
            out.append({"rule": "region_carried_across_a_cut", "level": LEVELS[2], "subject": run["subject"], "run": run["name"],
                        "source_frames": frame_spans(across),
                        "why": f"{kind} {run['name']}: on {len(across)} frame(s) beside a cut the region of {run['subject']} is laid on "
                               "the other shot, because one latent step covers frames on both sides of the cut. A render made "
                               "without the node's cut gate repaints them",
                        "figures": {"frames": len(across), "load_first_frame": load_first}})
        if shared:
            out.append({"rule": "region_shared_across_a_cut", "level": LEVELS[1], "subject": run["subject"], "run": run["name"],
                        "source_frames": frame_spans(shared),
                        "why": f"{kind} {run['name']}: {run['subject']} is on both sides of a cut inside one latent step on "
                               f"{len(shared)} frame(s): each side is given the other side's region as well as its own, and no gate "
                               "covers that. Look at those frames after the render; starting the load a few frames earlier can "
                               "move the cut onto a step's edge (`loop_plan.first_frame_choices`)",
                        "figures": {"frames": len(shared), "load_first_frame": load_first}})
    return out


#: A load whose subject is, on its first frame, under this share of the largest it gets in the load starts on
#: a small subject. Reasoned, not measured: half. The reason for the rule is measured once, on another clip
#: (the 2026-10-04 postmortem's item 9: a window that began on a large subject held it when it became small).
SMALL_START = 0.5


#: A subject whose mask never reaches this share of the frame in a load is small for the whole of it. Inherited,
#: from one clip: the 2026-10-04 postmortem's item 9 (a subject at about this share was held only where its
#: window began on frames in which it was large). Not measured on this lane.
SMALL_SUBJECT = 0.02


def small_start(shares: np.ndarray) -> dict | None:
    """Whether a load starts on its subject small: `shares` is the subject's mask as a share of the frame, per
    frame of the load. None when the load never shows the subject or starts at half its largest or more."""
    if not len(shares) or not shares.max():
        return None
    largest = int(np.argmax(shares))
    if shares[0] >= SMALL_START * shares[largest]:
        return None
    return {"first_frame_share": round(float(shares[0]), 5), "largest_share": round(float(shares[largest]), 5),
            "largest_on_load_frame": largest}


def shots_without(present: np.ndarray, cuts: list[int], first: int) -> list[list[int]]:
    """The shots of a load its subject has no mask on at all: `present` per frame from source frame `first`,
    `cuts` the source frames a new shot starts on. A shot with a mask on any frame is not one."""
    edges = [first] + sorted(c for c in cuts if first < c < first + len(present)) + [first + len(present)]
    return [[a, b - 1] for a, b in zip(edges, edges[1:]) if not present[a - first:b - first].any()]


def flag_loads(manifest: dict, folder: Path, not_in: dict | None = None) -> list[dict]:
    """A planned load that starts where its subject is small, when the same load shows it larger later; and a
    shot inside a planned load on which its subject has no mask at all, which nobody has said is right."""
    first, out = manifest["first_frame"], []
    cuts = []
    for s in manifest["subjects"]:
        for seen in s["sightings"]:
            table = folder / "subjects" / s["label"] / f"shots__{seen['by']}.json"
            if table.is_file() and not cuts:
                cuts = [seen.get("shots_first_source_frame", seen["first_source_frame"]) + c for c in json.loads(table.read_text())["cuts"]]
    for run in manifest["runs"]:
        load = run.get("load")
        if not load:
            continue
        seen = next(s for s in manifest["subjects"] if s["label"] == run["subject"])["sightings"][0]
        rows = [r for r in json.loads((folder / "subjects" / run["subject"] / "per_frame.json").read_text())["rows"]
                if r["seen_by"] == seen["by"]]
        shares = np.array([r.get("track_share") or 0.0 for r in rows])
        held = load_frames(first, len(shares), load["first_frame"], load["frames"])
        said = (not_in or {}).get(run["subject"], [])
        bare = [x for x in shots_without(shares[held] > 0, cuts, max(load["first_frame"], first))
                if not any(a <= x[0] and x[1] <= b for a, b in said)]
        if bare:
            out.append({"rule": "load_has_a_shot_without_the_subject", "level": LEVELS[2], "subject": run["subject"], "run": run["name"],
                        "source_frames": bare,
                        "why": f"plan {run['name']}: {run['subject']} has no mask on any frame of {len(bare)} shot(s) inside the load, so "
                               "nothing is laid there and the original shows. If they are in the shot, correct the tracker; if they "
                               "are not, say so with --not-in or split the load so it does not hold the shot",
                        "figures": {"shots": len(bare), "frames": sum(b - a + 1 for a, b in bare)}})
        mine = shares[held]
        if len(mine) and 0 < mine.max() < SMALL_SUBJECT:
            out.append({"rule": "subject_small_for_the_whole_load", "level": LEVELS[1], "subject": run["subject"], "run": run["name"],
                        "source_frames": [[max(load["first_frame"], first), max(load["first_frame"], first) + len(mine) - 1]],
                        "why": f"plan {run['name']}: {run['subject']}'s mask is never over {mine.max():.2%} of the frame in this load "
                               f"(median {float(np.median(mine[mine > 0])):.2%}). There is no frame in it where they are large to "
                               "start from; the margin and the text carry more of the result than the subject does",
                        "figures": {"largest_share": round(float(mine.max()), 5), "median_share": round(float(np.median(mine[mine > 0])), 5)},
                        "threshold": {"SMALL_SUBJECT": SMALL_SUBJECT}})
        found = small_start(shares[held])
        if found:
            at = load["first_frame"] + found["largest_on_load_frame"]
            out.append({"rule": "load_starts_on_a_small_subject", "level": LEVELS[1], "subject": run["subject"], "run": run["name"],
                        "source_frames": [[max(load["first_frame"], first), max(load["first_frame"], first)]],
                        "why": f"plan {run['name']}: on the load's first frame {run['subject']}'s mask is {found['first_frame_share']:.2%} "
                               f"of the frame, and {found['largest_share']:.2%} on source frame {at}. A subject is held best from "
                               "where it is large: a load that reaches that frame from another direction, or a text that states "
                               "the small state, is the plan to weigh",
                        "figures": {**found, "largest_on_source_frame": at}, "threshold": {"SMALL_START": SMALL_START}})
    return out


def voice_sentences(text: str) -> tuple[list[str], list[str]]:
    """The sentences of a prompt that say a voice is performed, and those that deny one."""
    import re
    says, denies = [], []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        low = sentence.lower()
        if any(w in low for w in VOICE_WORDS):
            (denies if any(re.search(rf"\b{re.escape(d)}\b", low) for d in DENIALS) else says).append(sentence.strip())
    return says, denies


def flag_text(text: str, spans: list[list[int]], first: int, frames: int) -> list[dict]:
    """A text that names a voice over frames with none, or says nothing of one over frames that have it."""
    says, denies = voice_sentences(text)
    last = first + frames - 1
    voiced = sum(max(0, min(b, last) - max(a, first) + 1) for a, b in spans)
    out = []
    if says and not voiced:
        out.append({"rule": "text_names_a_voice_where_there_is_none", "level": LEVELS[1], "subject": None,
                    "source_frames": [[first, last]], "why": "the text has a voice sentence and no frame of this span is voiced",
                    "figures": {"voiced_frames": 0, "sentences": says}})
    if voiced and not says:
        out.append({"rule": "voiced_frames_and_no_voice_sentence", "level": LEVELS[0] if not denies else LEVELS[1], "subject": None,
                    "source_frames": [[max(a, first), min(b, last)] for a, b in spans if a <= last and b >= first],
                    "why": f"{voiced} of {frames} frames are voiced and the text " + ("denies a voice" if denies else "says nothing of one")
                           + ": fine if that is meant", "figures": {"voiced_frames": voiced, "denials": denies}})
    return out


def preflight(a: argparse.Namespace) -> None:
    folder = Path(a.capture)
    m = json.loads((folder / "manifest.json").read_text())
    not_in: dict = {}
    for item in a.not_in:
        label, _, rng = item.partition(":")
        not_in.setdefault(label, []).append([int(x) for x in rng.split("-")])
    flags, shots = [], []
    for s in m["subjects"]:
        rows = json.loads((folder / "subjects" / s["label"] / "per_frame.json").read_text())["rows"]
        for seen in s["sightings"]:
            by, at = seen["by"], seen.get("shots_first_source_frame", seen["first_source_frame"])
            mine = [r for r in rows if r["seen_by"] == by]
            path = folder / "subjects" / s["label"] / f"shots__{by}.json"
            table = json.loads(path.read_text()) if path.is_file() else None
            if table is not None:
                f, sh = flag_shots(s["label"], by, table, at, not_in.get(s["label"], []))
                span = (m["first_frame"], m["first_frame"] + m["frames"] - 1)
                flags += [x for x in f if x["source_frames"][0][0] <= span[1] and x["source_frames"][0][1] >= span[0]]
                shots += [x for x in sh if x["source_frames"][0] <= span[1] and x["source_frames"][1] >= span[0]]
            flags += flag_track(s["label"], by, mine, table, at) + flag_parts(s["label"], by, mine)
            path = folder / "subjects" / s["label"] / f"pose__{by}.json"
            if path.is_file():
                others_on = {r["source_frame"] for o in m["subjects"] if o["label"] != s["label"]
                             for r in json.loads((folder / "subjects" / o["label"] / "per_frame.json").read_text())["rows"]
                             if r.get("track_share")}
                flags += flag_pose(s["label"], by, json.loads(path.read_text())["rows"], mine, others_on,
                                   bool((seen.get("pose") or {}).get("hand_refinement")))
    flags += flag_runs(m, folder) + flag_segments(m, folder) + flag_keep(m, folder) + flag_cuts(m, folder) + flag_loads(m, folder, not_in) + flag_lent(m, folder) + flag_held_tail(m, folder) + flag_kept(m, folder) + flag_mouth(m, folder)
    cross = json.loads((folder / "frames.json").read_text())["rows"]
    # a kept-out or keep mask given as a subject is a union of things, not somebody: it has no class map
    whos = {x["label"] for x in m["subjects"] if x["sightings"][0].get("classes")} or {x["label"] for x in m["subjects"]}
    for key in sorted({k for r in cross for k in r if k.startswith("masks_overlap_of_smaller__")}):
        _, one, two = key.split("__")
        if one not in whos or two not in whos:
            continue
        same = [r["source_frame"] for r in cross if (r.get(key) or 0) > SAME_PERSON]
        if same:
            flags.append({"rule": "two_tracks_on_one_person", "level": LEVELS[2], "subject": one, "other": two,
                          "source_frames": frame_spans(same),
                          "why": f"{one}'s and {two}'s tracked masks are nearly the same mask on {len(same)} frame(s): one of the "
                                 "two trackers took the other's person there. Correct that tracker's shot before either pass renders",
                          "figures": {"frames": len(same)}, "threshold": {"SAME_PERSON": SAME_PERSON}})
    hit = [r for r in cross if (r.get("contested_px") or 0) >= SEGMENT_PX]
    if hit:
        worst = max(hit, key=lambda r: r["contested_px"])
        flags.append({"rule": "contested_by_class_too", "level": LEVELS[1], "subject": None,
                      "source_frames": frame_spans([r["source_frame"] for r in hit]),
                      "why": f"on {len(hit)} frame(s) more than one track claims pixels that the class maps do not settle "
                             f"(up to {worst['contested_px']} px on source frame {worst['source_frame']}): `owners.npz` marks them "
                             "contested; look and decide whose they are",
                      "figures": {"frames": len(hit), "worst_px": worst["contested_px"]}, "threshold": {"SEGMENT_PX": SEGMENT_PX}})
    for s in m["subjects"]:
        path = folder / "subjects" / s["label"] / "doubted_frames.json"
        if path.is_file():
            doubted = json.loads(path.read_text())
            for kind, meaning in UNDER.items():
                hit = [x["source_frame"] for x in doubted["frames"] if x["mostly"] == kind]
                if hit:
                    flags.append({"rule": "doubted_part_" + kind.replace(" ", "_"), "level": LEVELS[1] if kind == "nothing labelled" else LEVELS[0],
                                  "subject": s["label"], "seen_by": doubted["seen_by"], "source_frames": frame_spans(hit),
                                  "why": f"on {len(hit)} doubted frame(s) of {s['label']}'s part, {meaning.lower()}",
                                  "figures": {"frames": len(hit), "filled_in_parts_held": sum(x["filled_in_parts_held"] for x in doubted["frames"] if x["mostly"] == kind)}})
    if a.text and a.voice_spans:
        spans = json.loads(Path(a.voice_spans).read_text())["voiced_spans_inclusive"]
        flags += flag_text(Path(a.text).read_text(), spans, m["first_frame"], m["frames"])
    order = {level: i for i, level in enumerate(LEVELS)}
    flags.sort(key=lambda f: (-order[f["level"]], f["source_frames"][0][0]))
    for i, f in enumerate(flags, 1):
        f["id"], f["key"] = f"f{i:03d}", flag_key(f)
    gate = gate_verdict(flags, read_outcomes(folder))
    record = {**gate, "capture": m["name"], "clip": m["clip"], "span": [m["first_frame"], m["first_frame"] + m["frames"] - 1],
              "written": datetime.datetime.now().isoformat(timespec="seconds"), "levels": list(LEVELS),
              # a tracker's own object id can sit beside the label when a tracker hands one back; none does today
              "subjects": [{"label": s["label"], "tracker_object_id": None, "seen_by": [x["by"] for x in s["sightings"]]}
                           for s in m["subjects"]],
              "shots": shots, "flags": flags,
              "outcomes": "outcomes.json, beside this file: what happened in a render against each flag id, once one exists"}
    (folder / "flags.json").write_text(json.dumps(record, indent=1) + "\n")
    for f in flags:
        spans_text = ", ".join(f"{x}-{y}" if x != y else str(x) for x, y in f["source_frames"][:8]) + (" ..." if len(f["source_frames"]) > 8 else "")
        print(f"{f['id']}  {f['level'].upper():14}  {f['rule']}\n      {f['why']}\n      source frames {spans_text}")
    counts = {level: sum(f["level"] == level for f in flags) for level in LEVELS}
    print(f"{len(flags)} flag(s): " + ", ".join(f"{n} {level}" for level, n in counts.items()) + f"; wrote {folder / 'flags.json'}")
    print(f"gate: {gate['verdict']}" + (f", on {', '.join(gate['blocking'])}" if gate["blocking"] else "")
          + (f"; overridden: {', '.join(gate['overridden'])}" if gate["overridden"] else ""))
    if a.gate and gate["blocking"]:
        sys.exit(GATE_BLOCKED)


def outcome(a: argparse.Namespace) -> None:
    """Record what a render did against one flag, so a threshold's provenance can go from reasoned to measured."""
    folder = Path(a.capture)
    listed = json.loads((folder / "flags.json").read_text())
    known = {f["id"]: f for f in listed["flags"]}
    if a.flag not in known:
        raise SystemExit(f"no flag {a.flag} in {folder / 'flags.json'}; it has {sorted(known)}")
    override = a.happened == "overridden"
    if override and not (a.by.strip() and a.note.strip()):
        raise SystemExit("an override needs --by (who) and --note (why): it is the record of a decision to render past a flag")
    if not override and not a.render:
        raise SystemExit("--render is needed: the render the flag was seen in, or not seen in")
    path = folder / "outcomes.json"
    record = json.loads(path.read_text()) if path.is_file() else {"outcomes": []}
    entry = {"flag": a.flag, "key": flag_key(known[a.flag]), "rule": known[a.flag]["rule"], "level": known[a.flag]["level"],
             "render": a.render, "note": a.note, "by": a.by, "written": datetime.datetime.now().isoformat(timespec="seconds")}
    # an override is a decision made before a render, not what a render did: it carries no `happened`
    entry.update({"overridden": True, "said": f"overridden by {a.by}: {a.note}"} if override else {"happened": a.happened == "yes"})
    record["outcomes"].append(entry)
    path.write_text(json.dumps(record, indent=1) + "\n")
    print("recorded", a.flag, a.happened, "in", path)
    if override:
        gate = gate_verdict(listed["flags"], record["outcomes"])
        (folder / "flags.json").write_text(json.dumps({**listed, **gate}, indent=1) + "\n")
        print(f"gate: {gate['verdict']}" + (f", still on {', '.join(gate['blocking'])}" if gate["blocking"] else ""))


def frame_changes(diff: np.ndarray, region: np.ndarray, maps: dict[str, np.ndarray], tracks: dict[str, np.ndarray],
                  names: tuple[str, ...], least: int = SEGMENT_PX) -> dict:
    """What one frame of a render changed, read off `diff` (the absolute grey difference from the source).

    `region` is the run's region on this frame in pixels, `maps` each subject's class map and `tracks` each
    subject's mask, all for this frame. Returns the floor (labelled or tracked pixels outside the region: the
    codec and the VAE, nothing regenerated; every pixel outside the region when a subject lies wholly inside
    it), each segment's pixels and mean difference INSIDE the region, and
    each subject's inside and outside it. A figure well above the floor is a thing drawn again."""
    labelled = np.zeros(region.shape, bool)
    for cm in maps.values():
        labelled |= cm > 0
    for t in tracks.values():
        labelled |= t
    rest = labelled & ~region
    if rest.sum() < 500 and labelled.any():
        rest = ~region          # a subject wholly inside its region leaves no labelled pixel outside: take the picture's
    out = {"floor": float(diff[rest].mean()) if rest.sum() >= 500 else None, "segments": {}, "subjects": {}}
    for label, cm in maps.items():
        inside = np.where(region, cm, 0)
        for k in np.nonzero(np.bincount(inside.ravel(), minlength=len(names))[1:len(names)] >= least)[0] + 1:
            px = inside == k
            out["segments"][f"{label}.{names[k]}"] = (int(px.sum()), float(diff[px].mean()))
    for label, t in tracks.items():
        a, b = t & region, t & ~region
        out["subjects"][label] = {"inside_px": int(a.sum()), "inside": float(diff[a].mean()) if a.sum() >= least else None,
                                  "outside": float(diff[b].mean()) if b.sum() >= least else None}
    return out


#: A render ends (or starts) on the source when its last (first) frame is under `END_LAST` of the load's own
#: level of difference from the source under the mask it carried, and the run is the frames from that end under
#: `END_RUN` of the level. Reasoned bars, measured on two renders that did (2026-10-10: last frames at 0.11
#: and 0.16 of their level, runs of six and three frames) and three that did not (last frames at 0.6 to 1.1).
END_LAST = 1 / 3
END_RUN = 2 / 3
#: Fewer frames with a mask than this, and the ends are not read.
END_LEAST = 12


def source_at_the_ends(frames: list[int], values: list[float], floor: float) -> dict:
    """Does a render start or end on the source's own picture, under the mask it was given.

    `values` is the mean difference from the source under the run's carried mask inside its region, one a
    frame of `frames` (frames with no mask left out). The load's level is their median. An end is "on the
    source" when its outermost frame is under `END_LAST` of the level; the run is the frames from that end
    that stay under `END_RUN` of it. A render that replaced nothing (a level within three floors) is not
    read: every frame of it is the source.

    Why it exists: a load shorter than its window ended on the original twice on 2026-10-10, found by eye and
    by a viewer. `flag_held_tail` names the cause before a render; this is the outcome that grades it, read on
    every `changed`. The same reading at the start is for a continuation behind kept frames."""
    out = {"frames_read": len(values), "level": None, "starts_on_the_source": None, "ends_on_the_source": None}
    if len(values) < END_LEAST:
        return {**out, "why_not_read": f"under {END_LEAST} frames carry a mask"}
    level = float(np.median(values))
    out["level"] = round(level, 2)
    if level < 3 * floor:
        return {**out, "why_not_read": "the render is within three floors of the source under its mask on a typical frame: nothing was replaced"}
    for name, order in (("ends_on_the_source", range(len(values) - 1, -1, -1)), ("starts_on_the_source", range(len(values)))):
        order = list(order)
        if values[order[0]] > level * END_LAST:
            continue
        run = []
        for i in order:
            if values[i] > level * END_RUN:
                break
            run.append(i)
        out[name] = {"source_frames": [frames[min(run)], frames[max(run)]], "frames": len(run),
                     "outermost_frame_off_the_source": round(values[order[0]], 2),
                     "share_of_the_level": round(values[order[0]] / level, 2), "times_the_floor": round(values[order[0]] / max(floor, 1e-6), 1)}
    return out


def changed(a: argparse.Namespace) -> None:
    """What a run redrew, per segment and per subject, against the floor; and how it moves from frame to frame.

    The reading behind four of the first day's findings (2026-10-10): a pass that redrew the other subject where
    its region ran over them; a face that went back toward the original once kept pixels sat beside it (its
    difference from the source on freely regenerated face pixels fell by half over ninety frames); a face pass
    that stayed level across a window boundary; and no step at a seam. `--series LABEL.Class` prints that
    segment's difference frame by frame in bins, which is the "stays level or falls" reading. `step` is the
    change from the frame before inside the region, render beside source: a seam is a render step well above
    its neighbours where the source's is not. Writes `runs/<run>/changed.json`."""
    folder = Path(a.capture)
    m = json.loads((folder / "manifest.json").read_text())
    w, h = m["size"]
    first, frames = m["first_frame"], m["frames"]
    lo, hi = (int(x) for x in a.frames.split("-")) if a.frames else (first, first + frames - 1)
    names = _pack("sapiens2_parts").CLASS_NAMES
    region = cells_up(np.load(folder / "runs" / a.run / "region.npz")["region"])
    carried = np.unpackbits(np.load(folder / "runs" / a.run / "region.npz")["carried"], axis=-1)[..., :w].astype(bool)
    own: list[tuple[int, float]] = []
    maps, tracks = {}, {}
    for s in m["subjects"]:
        by = s["sightings"][0]["by"]
        z = np.load(folder / "subjects" / s["label"] / f"masks__{by}.npz")
        path = folder / "subjects" / s["label"] / f"classes__{by}.npz"
        if path.is_file():
            maps[s["label"]] = np.load(path)["classes"]
            tracks[s["label"]] = np.unpackbits(z["track"], axis=-1)[..., :w].astype(bool)
        elif not maps:
            tracks[s["label"]] = np.unpackbits(z["track"], axis=-1)[..., :w].astype(bool)
    start = next((r.get("first_source_frame", first) for r in m["runs"] if r["name"] == a.run), first)
    src = stream(a.source, (w, h), first, frames, vf=FIT.format(w=w, h=h))
    ren = stream(a.render, (w, h), max(first - start, 0), frames)
    floors, segs, subs, series, steps, before = [], {}, {}, {}, [], None
    for n, (s_, r_) in enumerate(zip(src, ren)):
        s16, r16 = s_.astype(np.int16), r_.astype(np.int16)
        if before is not None and lo <= first + n <= hi:
            both = region[n] & region[n - 1]
            if both.any():
                steps.append((first + n, float(np.abs(s16 - before[0])[both].mean()), float(np.abs(r16 - before[1])[both].mean())))
        before = (s16, r16)
        if not lo <= first + n <= hi:
            continue
        mine = carried[n] & region[n]
        if mine.sum() >= SEGMENT_PX:
            own.append((first + n, float(np.abs(s16 - r16)[mine].mean())))
        one = frame_changes(np.abs(s16 - r16), region[n], {k: v[n] for k, v in maps.items()}, {k: v[n] for k, v in tracks.items()}, names)
        if one["floor"] is not None:
            floors.append(one["floor"])
        for key, (px, d) in one["segments"].items():
            segs.setdefault(key, []).append((first + n, px, d))
        for key, v in one["subjects"].items():
            subs.setdefault(key, []).append((first + n, v))
        if a.series and a.series in one["segments"]:
            series[first + n] = one["segments"][a.series][1]
    if not floors:
        raise SystemExit("no frame had enough labelled pixels outside the region to set a floor")
    floor = float(np.median(floors))
    record = {"run": a.run, "render": Path(a.render).name, "source_frames": [lo, hi], "floor": round(floor, 2), "segments": {}, "subjects": {}}
    print(f"run {a.run}, source frames {lo}-{hi}; floor {floor:.2f} grey levels (labelled pixels outside the region)")
    for key, rows in sorted(segs.items(), key=lambda kv: -len(kv[1])):
        d = float(np.median([x[2] for x in rows]))
        record["segments"][key] = {"frames": len(rows), "first": rows[0][0], "last": rows[-1][0], "px_median": int(np.median([x[1] for x in rows])),
                                   "off_the_source": round(d, 2), "times_the_floor": round(d / floor, 1)}
        print(f"  {key.ljust(26)} {len(rows):4d} frames ({rows[0][0]}-{rows[-1][0]}), {int(np.median([x[1] for x in rows])):6d} px a frame, "
              f"off the source {d:5.1f} ({d / floor:.1f}x the floor)")
    for key, rows in subs.items():
        ins = [v["inside"] for _, v in rows if v["inside"] is not None]
        out_ = [v["outside"] for _, v in rows if v["outside"] is not None]
        record["subjects"][key] = {"frames_with_pixels_inside": len(ins), "inside": round(float(np.median(ins)), 2) if ins else None,
                                   "outside": round(float(np.median(out_)), 2) if out_ else None}
        print(f"  subject {key}: its mask inside the region on {len(ins)} frames, off the source there "
              + (f"{np.median(ins):.1f}" if ins else "-") + ", outside it " + (f"{np.median(out_):.1f}" if out_ else "-"))
    record["ends"] = ends = source_at_the_ends([f for f, _ in own], [v for _, v in own], floor)
    record["under_the_carried_mask_by_frame"] = [[f, round(v, 1)] for f, v in own]
    for name in ("starts_on_the_source", "ends_on_the_source"):
        if ends[name]:
            e = ends[name]
            print(f"  {name.replace('_', ' ').upper()}: source frames {e['source_frames'][0]}-{e['source_frames'][1]} ({e['frames']} frame(s)); the "
                  f"outermost frame is {e['outermost_frame_off_the_source']} off the source under the carried mask, {e['share_of_the_level']} of "
                  f"the load's level ({ends['level']}) and {e['times_the_floor']}x the floor")
    if not ends["starts_on_the_source"] and not ends["ends_on_the_source"]:
        print("  the ends: " + (ends.get("why_not_read") or f"neither end is on the source (level {ends['level']} under the carried mask)"))
    if steps:
        sv, rv = np.array([x[1] for x in steps]), np.array([x[2] for x in steps])
        ratio = rv / np.maximum(sv, 0.5)
        top = sorted(zip(ratio.tolist(), steps), reverse=True)[:8]
        record["step"] = {"source_median": round(float(np.median(sv)), 2), "render_median": round(float(np.median(rv)), 2),
                          "largest_render_over_source": [[f, round(x, 2), round(y, 2)] for _, (f, x, y) in top]}
        print(f"  change in the region from the frame before: source median {np.median(sv):.2f}, render {np.median(rv):.2f}; "
              f"largest render over source (frame, source, render): {record['step']['largest_render_over_source'][:5]}")
        for mark in a.around or []:
            near = [(f, round(x, 1), round(y, 1)) for f, x, y in steps if abs(f - mark) <= 3]
            print(f"  around source frame {mark} (frame, source, render): {near}")
    if a.series:
        got = sorted(series.items())
        record["series"] = {"segment": a.series, "by_frame": {str(f): round(v, 2) for f, v in got}}
        if got:
            vals = np.array([v for _, v in got])
            print(f"  {a.series} off the source, {len(got)} frames, median {np.median(vals):.1f}; by {a.bin} frames from {got[0][0]}: "
                  f"{[round(float(np.mean(vals[i:i + a.bin])), 1) for i in range(0, len(vals), a.bin)]}")
            for mark in a.around or []:
                print(f"  around source frame {mark}: {[(f, round(v, 1)) for f, v in got if abs(f - mark) <= 6]}")
    (folder / "runs" / a.run / "changed.json").write_text(json.dumps(record, indent=1) + "\n")
    print("wrote", folder / "runs" / a.run / "changed.json")


#: The part model's classes that make a mouth. Inherited: `sapiens2_parts._MOUTH`.
MOUTH_CLASSES = ("Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue")
#: A mouth mask's largest piece under this many pixels is not read. Reasoned: a few pixels have no shape.
MOUTH_PX = 12
#: Two mouths further apart than this, in pixels, are not the same mouth (the part model found one elsewhere).
#: Reasoned: under a face's width at the sizes this lane renders; measured apart at a median of 6 on one pass.
MOUTH_APART = 40
MOUTH_SHIFTS = 6               # frames either way an arm is tried against the reference; reasoned: a quarter second


def mouth_openings(mouth: np.ndarray, face: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, how open a mouth is, and where it is: [n] and [n, 2] (NaN where it is not seen).

    On the mask's LARGEST piece (the part model also puts lip labels on other things in the frame): the
    geometric mean of the two axes of the ellipse with its second moments, which is the size of lips, teeth
    and tongue together whichever way the head is tilted or turned; over the face's size, the square root of
    the face mask's area as the median over `RECENT` frames either side, because a face mask collapses on
    single frames. With no face mask the figure is in pixels.

    Tried first and thrown out, 2026-10-10: the mask's box, height over width. Wrong in profile (the width
    shrinks and a shut mouth reads open) and wherever a stray label makes the box hundreds of pixels tall.
    This form ranked an open frame above a shut one 91 times in 100 on 19 frames of one stretch read by eye
    (the short axis alone 89, the area 85, short over long 73); it was chosen on those frames, so that
    flatters it. Where it is wrong: a shut mouth in a wide smile reads large, an open one in profile small."""
    import cv2
    n = len(mouth)
    opening, centre = np.full(n, np.nan), np.full((n, 2), np.nan)
    area = None if face is None else np.sqrt(face.reshape(n, -1).sum(axis=1).astype(np.float64))
    for f in range(n):
        count, labels, stats, cents = cv2.connectedComponentsWithStats(mouth[f].astype(np.uint8), connectivity=8)
        if count < 2:
            continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        scale = 1.0 if area is None else float(np.median(area[max(f - RECENT, 0):f + RECENT + 1]))
        if stats[k, cv2.CC_STAT_AREA] < MOUTH_PX or scale < 8:
            continue
        ys, xs = np.nonzero(labels == k)
        lam = np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(np.float64)))
        opening[f] = 4.0 * float(np.sqrt(np.sqrt(max(lam[0], 0.0)) * np.sqrt(lam[1]))) / scale
        centre[f] = cents[k]
    return opening, centre


def shifted_agreement(reference: np.ndarray, arm: np.ndarray, shifts: int = MOUTH_SHIFTS) -> dict[int, float | None]:
    """Rank correlation of an arm's series with the reference's, the arm taken `k` frames LATE for positive k.
    A peak at 0 is a mouth in time with the reference; a peak elsewhere is one early or late."""
    from scipy.stats import spearmanr
    out = {}
    for k in range(-shifts, shifts + 1):
        a = reference[max(-k, 0):len(reference) - max(k, 0)]
        b = arm[max(k, 0):len(arm) - max(-k, 0)]
        ok = ~np.isnan(a) & ~np.isnan(b)
        out[k] = round(float(spearmanr(a[ok], b[ok]).statistic), 3) if ok.sum() > 20 else None
    return out


def score_mouth(reference: np.ndarray, ref_centre: np.ndarray, arm: np.ndarray, arm_centre: np.ndarray) -> dict:
    """An arm's mouth against the reference's, on the frames both are seen and are the same mouth."""
    apart = np.hypot(*(ref_centre - arm_centre).T)
    arm = np.where(apart < MOUTH_APART, arm, np.nan)
    ok = ~np.isnan(reference) & ~np.isnan(arm)
    by_shift = shifted_agreement(reference, arm)
    known = {k: v for k, v in by_shift.items() if v is not None}
    best = max(known, key=known.get) if known else None
    median = float(np.nanmedian(reference))
    diff = arm[ok] - reference[ok]
    return {"frames_same_mouth": int(ok.sum()), "of": int(len(reference)), "agreement_at_no_shift": by_shift.get(0),
            "best_shift_arm_late_positive": best, "agreement_at_best_shift": known.get(best) if best is not None else None,
            "by_shift": by_shift,
            "same_side_of_the_reference_median": round(float(((reference[ok] > median) == (arm[ok] > median)).mean()), 3) if ok.any() else None,
            "level_difference_median": round(float(np.median(diff)), 3) if ok.any() else None,
            "level_difference_quartiles": [round(float(x), 3) for x in np.percentile(diff, [25, 75])] if ok.any() else None}


def with_the_voice(level: np.ndarray, series: np.ndarray) -> dict:
    """Does a mouth open and shut with the voice's level: a singer's does, a listener's should not.

    `level` is the vocal stem's level per frame and `series` the mouth's opening; the agreement is
    `shifted_agreement`'s, at no shift and at the shift that fits best (positive when the mouth is late).
    """
    by = {k: v for k, v in shifted_agreement(level, series).items() if v is not None}
    best = max(by, key=by.get) if by else None
    return {"with_the_vocal_level_at_no_shift": by.get(0),
            "with_the_vocal_level_best": [best, by[best]] if best is not None else None}


#: A mouth is "open for the shot" on the frames in the top quarter of the source's own openings over the shot.
#: Reasoned, not measured: the ledger's entry (`docs/wiki/state_signals.md`, "A mouth open with no voice")
#: asks for the top share; a quarter is the first try.
OPEN_SHARE = 0.25
#: A run of open frames is one at least this long. Inherited: the same entry's "a dozen frames or longer".
OPEN_RUN = 12


def open_runs(reference: np.ndarray, voiced: np.ndarray, arms: dict[str, np.ndarray], first: int = 0) -> dict:
    """The runs where the source's mouth is open with no voice to open it, and what each render's mouth does there.

    `reference` is the source's opening per frame (`mouth_openings`; nan where no mouth is seen), `voiced` the
    voice table's column on the same frames (1 voiced, 0 not, nan unknown), `arms` each render's opening. A
    frame counts when the source's opening is in the top `OPEN_SHARE` of its seen frames over the shot and
    above the shot's median, and the frame is unvoiced; a frame the voice table does not cover does not count, so a capture with no voice
    table finds nothing and says so. Runs join across two frames, and are kept at `OPEN_RUN` frames or more.
    Per run and arm: the median opening, and its share of the source's. A face pass has no channel for this
    state (no voice to follow, no motion video), so a render whose share is near 0 has a closed mouth there."""
    seen = reference[~np.isnan(reference)]
    if not len(seen):
        return {"bar": None, "runs": [], "why_none": "the source's mouth is not seen on any frame"}
    if not (voiced == 0).any():
        return {"bar": None, "runs": [], "why_none": "no frame is known to be unvoiced: the capture has no voice table on these frames (--voice)"}
    bar = float(np.quantile(seen, 1.0 - OPEN_SHARE))
    # above the shot's own median too: on a mouth that hardly moves the top quarter is the same value as the rest
    hit = np.nonzero((reference >= bar) & (reference > float(np.median(seen))) & (voiced == 0))[0].tolist()
    runs = []
    for lo, hi in frame_spans(hit, join=2):
        if hi - lo + 1 < OPEN_RUN:
            continue
        ref = reference[lo:hi + 1]
        entry = {"source_frames": [first + lo, first + hi], "frames": hi - lo + 1, "source_opening_median": round(float(np.nanmedian(ref)), 3),
                 "renders": {}}
        for name, series in arms.items():
            mine = series[lo:hi + 1]
            got = float(np.nanmedian(mine)) if (~np.isnan(mine)).any() else None
            entry["renders"][name] = {"opening_median": None if got is None else round(got, 3),
                                      "share_of_the_source": None if got is None else round(got / max(float(np.nanmedian(ref)), 1e-9), 2),
                                      "frames_with_a_mouth": int((~np.isnan(mine)).sum())}
        runs.append(entry)
    return {"bar": round(bar, 3), "top_share": OPEN_SHARE, "least_frames": OPEN_RUN, "shot_opening_median": round(float(np.median(seen)), 3), "runs": runs}


def frames_inside(spec: str | None, first: int, frames: int) -> np.ndarray:
    """Which of a span's frames a FIRST-LAST of source frames names; all of them when none is given."""
    if not spec:
        return np.ones(frames, bool)
    lo, hi = (int(x) for x in spec.split("-"))
    return np.array([lo <= first + n <= hi for n in range(frames)])


def mouth(a: argparse.Namespace) -> None:
    """Does a render's mouth do what the source's mouth does, frame by frame.

    The reference is the source's mouth: a mask video (`--reference`), or the subject's own class map in the
    capture. Each `--arm` is a mouth mask of a render (a no-sampling preview with the part node on `mouth`,
    loader on the render), or a class map of it (`:classes`). Printed first, the control: the reference against
    ITSELF shifted by 1, 2, 3, 6 and 12 frames, which is what the agreement reads for a mouth that is right but
    that many frames out; an arm at the 12-frame figure ignores the reference's timing. Then each arm. With a
    voice table in the capture, the opening on voiced against unvoiced frames, and in each gap of the voice
    against the frames either side: a source that holds its mouth open through a breath is no rest for a render
    to follow (measured: the first gap of the pass this was written on). Writes `mouth__<subject>.json`."""
    folder = Path(a.capture)
    m = json.loads((folder / "manifest.json").read_text())
    w, h = m["size"]
    first, frames = m["first_frame"], m["frames"]
    names = _pack("sapiens2_parts").CLASS_NAMES
    seen = next(s for s in m["subjects"] if s["label"] == a.subject)["sightings"][0]
    saved = np.load(folder / "subjects" / a.subject / f"masks__{seen['by']}.npz")
    face = np.unpackbits(saved["parts" if "parts" in saved.files else "track"], axis=-1)[..., :w].astype(bool)

    def read(spec: str) -> np.ndarray:
        as_classes = spec.endswith(":classes")
        path, _, at = (spec[:-len(":classes")] if as_classes else spec).partition("@")
        at = int(at) if at else first
        if as_classes:
            return class_mask_of(read_classes(path, (w, h), at, first, frames), list(MOUTH_CLASSES), names)
        return read_mask(path, (w, h), at, first, frames)[0]

    if a.reference:
        ref_mask = read(a.reference)
    else:
        ref_mask = class_mask_of(np.load(folder / "subjects" / a.subject / f"classes__{seen['by']}.npz")["classes"], list(MOUTH_CLASSES), names)
    reference, ref_centre = mouth_openings(ref_mask, face)
    scored = frames_inside(a.frames, first, frames)
    reference = np.where(scored, reference, np.nan)
    control = {k: v for k, v in shifted_agreement(reference, reference, 24).items() if k in (1, 2, 3, 6, 12, 24)}
    cross = json.loads((folder / "frames.json").read_text())["rows"]
    voiced = np.array([np.nan if r.get("voiced") in (None, "") else float(r["voiced"]) for r in cross])
    level = np.array([np.nan if r.get("vocals_stem_dbfs") in (None, "") else float(r["vocals_stem_dbfs"]) for r in cross])
    if a.voice:
        # a capture made without --voice: the clip's own table, on the clip's frames
        table = read_voice(a.voice)
        voiced = np.array([float(table[first + n]["voiced"]) if first + n in table else np.nan for n in range(frames)])
        level = np.array([table[first + n]["vocals_stem_dbfs"] if first + n in table else np.nan for n in range(frames)])
    record = {"subject": a.subject, "reference": a.reference or "the subject's class map", "frames_seen": int((~np.isnan(reference)).sum()),
              "control_reference_against_itself_shifted": control, "arms": {}}
    print(f"reference mouth seen on {record['frames_seen']} of {frames} frames; against itself shifted by frames: {control}")

    def rests(series: np.ndarray) -> dict | None:
        if np.isnan(voiced).all():
            return None
        on, off = series[voiced == 1], series[voiced == 0]
        gaps = []
        quiet = frame_spans(np.nonzero(voiced == 0)[0].tolist(), join=1)
        for lo, hi in quiet:
            if lo == 0 or hi >= frames - 1:
                continue
            inside = series[lo:hi + 1]
            near = np.r_[series[max(lo - RECENT, 0):lo], series[hi + 1:hi + 1 + RECENT]]
            if (~np.isnan(inside)).any() and (~np.isnan(near)).any():
                gaps.append({"source_frames": [first + lo, first + hi], "in_the_gap": round(float(np.nanmedian(inside)), 3),
                             "either_side": round(float(np.nanmedian(near)), 3)})
        return {"voiced_median": round(float(np.nanmedian(on)), 3) if (~np.isnan(on)).any() else None,
                "unvoiced_median": round(float(np.nanmedian(off)), 3) if (~np.isnan(off)).any() else None,
                **with_the_voice(level, series), "gaps": gaps}

    record["reference_and_the_voice"] = rests(reference)
    if record["reference_and_the_voice"]:
        print("  the reference and the voice:", record["reference_and_the_voice"])
    openings = {}
    for item in a.arm:
        name, _, spec = item.partition("=")
        series, centre = mouth_openings(read(spec), face)
        series = np.where(scored, series, np.nan)
        openings[name] = series
        score = score_mouth(reference, ref_centre, series, centre)
        score["frames_seen"] = int((~np.isnan(series)).sum())
        score["and_the_voice"] = rests(series)
        record["arms"][name] = score
        print(f"  {name.ljust(22)} same mouth {score['frames_same_mouth']} of {score['of']} | at no shift {score['agreement_at_no_shift']} | "
              f"best {score['agreement_at_best_shift']} at {score['best_shift_arm_late_positive']} | same side "
              f"{score['same_side_of_the_reference_median']} | level {score['level_difference_median']} {score['level_difference_quartiles']}"
              + (f" | voice {score['and_the_voice']}" if score["and_the_voice"] else ""))
    record["open_with_no_voice"] = found = open_runs(reference, voiced, openings, first)
    if found["runs"]:
        print(f"  the source's mouth open with no voice (opening at least {found['bar']}, the shot's median {found['shot_opening_median']}):")
        for run in found["runs"]:
            print(f"    source frames {run['source_frames'][0]}-{run['source_frames'][1]} ({run['frames']}): source {run['source_opening_median']}"
                  + "".join(f" | {name} {r['opening_median']} ({r['share_of_the_source']} of the source's, a mouth on {r['frames_with_a_mouth']})"
                            for name, r in run["renders"].items()))
    else:
        print("  the source's mouth open with no voice: no run" + (f" ({found['why_none']})" if found.get("why_none") else ""))
    out = folder / f"mouth__{a.subject}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print("wrote", out)


def class_mask_of(classes: np.ndarray, wanted: list[str], names: tuple[str, ...]) -> np.ndarray:
    """[n, h, w] of bool: the pixels of a class map that are any of the classes named. An unknown name is refused."""
    unknown = [c for c in wanted if c not in names]
    if unknown:
        raise SystemExit(f"no class named {unknown}; the part model's are {', '.join(names[1:])}")
    return np.isin(classes, [names.index(c) for c in wanted])


def owned_class_mask(classes: np.ndarray, owner: np.ndarray, index: int, wanted: list[str], names: tuple[str, ...]) -> np.ndarray:
    """The pixels of the classes named that the owner map gives to subject `index`: nothing in the margin round
    the subject's outline, where a class map labels a neighbour's things, and nothing two tracks contest."""
    return class_mask_of(classes, wanted, names) & (owner == index)


def mask(a: argparse.Namespace) -> None:
    """Write some of a subject's classes as a lossless mask video a graph can load (for `keep`, `others`, a region).

    `--classes` takes them as the class map has them, the margin round the outline included: right for a class
    that is the subject's own wherever it is labelled (a face, hair). `--classes-owned` takes only the pixels
    the capture's owner map gives the subject: right for a class a neighbour can lend (a hand, apparel, a held
    thing). The two are joined. This is the `keep` mask for a pass on ANOTHER subject."""
    folder = Path(a.capture)
    m = json.loads((folder / "manifest.json").read_text())
    subject = next((s for s in m["subjects"] if s["label"] == a.subject), None)
    if subject is None:
        raise SystemExit(f"no subject {a.subject!r} in this capture; it has {[s['label'] for s in m['subjects']]}")
    path = folder / "subjects" / a.subject / f"classes__{subject['sightings'][0]['by']}.npz"
    if not path.is_file():
        raise SystemExit(f"{a.subject} has no class map in this capture (give classes= on its --mask when capturing)")
    names = _pack("sapiens2_parts").CLASS_NAMES
    classes = np.load(path)["classes"]
    out = class_mask_of(classes, a.classes.split("+"), names) if a.classes else np.zeros(classes.shape, bool)
    if a.classes_owned:
        if not (folder / "owners.npz").is_file():
            raise SystemExit("--classes-owned needs the capture's owners.npz (a capture with more than one subject and class maps)")
        owners = np.load(folder / "owners.npz")
        out |= owned_class_mask(classes, owners["owner"], list(owners["labels"]).index(a.subject), a.classes_owned.split("+"), names)
    if not out.any():
        raise SystemExit("the mask would be empty on every frame: nothing was written")
    write_mask_video(Path(a.out), out)
    said = " and ".join(x for x in ((a.classes or "").replace("+", ", "), (a.classes_owned or "").replace("+", ", ") + " where the owner map gives them the pixel"
                                    if a.classes_owned else "") if x)
    print(f"wrote {a.out}: {a.subject}'s {said}, {len(out)} frames from source frame {m['first_frame']}, "
          f"on {int(out.reshape(len(out), -1).any(1).sum())} of them")


def diagnose(a: argparse.Namespace) -> None:
    """Everything the capture says about a stretch somebody marked as wrong: the first step after a bad render.

    For the source frames given: each subject's shots, mask and part over those frames; each run's region and
    what else was inside it; what the rows across subjects say; whether there was a voice; and every flag
    that was raised on those frames before the render, with its outcome if one was recorded. It reads the
    tables only. A cause is then a row or a figure from here, or it is a reading and says so."""
    folder = Path(a.capture)
    m = json.loads((folder / "manifest.json").read_text())
    lo, hi = (int(x) for x in a.frames.split("-"))
    inside = lambda r: lo <= r["source_frame"] <= hi          # noqa: E731

    def stat(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return "no value" if not vals else f"median {np.median(vals):.4g}, from {min(vals):.4g} to {max(vals):.4g}"

    print(f"capture {m['name']}, {m['clip']}, source frames {lo}-{hi} (span frames {lo - m['first_frame']}-{hi - m['first_frame']})")
    for s in m["subjects"]:
        rows = json.loads((folder / "subjects" / s["label"] / "per_frame.json").read_text())["rows"]
        for seen in s["sightings"]:
            mine = [r for r in rows if r["seen_by"] == seen["by"] and inside(r)]
            print(f"\nSUBJECT {s['label']}, seen by {seen['by']}: {sum(r['covered'] for r in mine)} of {len(mine)} frames covered by its mask video")
            path = folder / "subjects" / s["label"] / f"shots__{seen['by']}.json"
            if path.is_file():
                table, at = json.loads(path.read_text()), seen.get("shots_first_source_frame", seen["first_source_frame"])
                for shot in table["shots"]:
                    x, y = at + shot["first_frame"], at + shot["last_frame"]
                    if x <= hi and y >= lo:
                        print(f"  shot {x}-{y}: {shot['subject']['state']}, {len(shot.get('people', []))} detected; {shot['subject'].get('why', '')}")
            else:
                print("  no shot table saved for this sighting")
            empty = [r["source_frame"] for r in mine if r["covered"] and not r.get("track_share")]
            print(f"  mask share of the frame: {stat(mine, 'track_share')}; empty on {len(empty)} frame(s) {frame_spans(empty)[:6]}")
            print(f"  pieces: {stat(mine, 'track_pieces')}; overlap with the frame before: {stat(mine, 'track_step')}")
            if any("parts_share" in r for r in mine):
                print(f"  part share of the frame: {stat(mine, 'parts_share')}; of the subject: {stat(mine, 'parts_of_track')}; "
                      f"inside the subject: {stat(mine, 'parts_inside_track')}")
    cross = [r for r in json.loads((folder / "frames.json").read_text())["rows"] if inside(r)]
    for run in m["runs"]:
        rows = [r for r in json.loads((folder / "runs" / run["name"] / "per_frame.json").read_text())["rows"] if inside(r)]
        print(f"\n{'PLAN' if run.get('planned') else 'RUN'} {run['name']}: subject {run['subject']}, kept out {run['others'] or 'nobody'}, "
              f"margin {run['margin_px']} px, {run['masked_source'].get('edge')}; region from {run['read_from']}")
        print(f"  region share of the frame: {stat(rows, 'region_share')}; carried mask: {stat(rows, 'carried_share')}; "
              f"margin: {stat(rows, 'margin_share')}; region against the frame before: {stat(rows, 'region_step')}")
        for key in sorted({k for r in cross for k in r if f"__{run['name']}__" in k}):
            print(f"  {key}: {stat(cross, key)}")
    for key in sorted({k for r in cross for k in r if k.startswith(("masks_overlap__", "same_subject__", "regions_overlap"))}):
        print(f"\nACROSS {key}: {stat(cross, key)}")
    voiced = [r["source_frame"] for r in cross if r.get("voiced")]
    if any("voiced" in r for r in cross):
        print(f"\nVOICE: voiced on {len(voiced)} of {len(cross)} frames {frame_spans(voiced)[:6]}")
    path = folder / "flags.json"
    if path.is_file():
        outcomes = json.loads((folder / "outcomes.json").read_text())["outcomes"] if (folder / "outcomes.json").is_file() else []
        hit = [f for f in json.loads(path.read_text())["flags"] if any(x <= hi and y >= lo for x, y in f["source_frames"])]
        print(f"\nFLAGS raised before the render that touch these frames: {len(hit)}")
        for f in hit:
            said = [o for o in outcomes if o["flag"] == f["id"]]
            print(f"  {f['id']} {f['level']}: {f['why']}" + "".join(f"\n      outcome in {o['render']}: {'happened' if o['happened'] else 'did not happen'} {o['note']}" for o in said))
        if not hit:
            print("  none: if something is wrong here, no rule predicted it; that is a new rule or an entry in data/CAPTURE_GAPS.md")
    else:
        print("\nno flags.json: preflight was not run on this capture")
    for path in sorted(folder.glob("delivery__*.json")):     # written by bench/assemble_delivery.py, in the same shape
        hit = [f for f in json.loads(path.read_text()).get("flags", []) if any(x <= hi and y >= lo for x, y in f["source_frames"])]
        print(f"\nFLAGS from the delivery {path.name} that touch these frames: {len(hit)}")
        for f in hit:
            print(f"  {f.get('id', '')} {f['level']}: {f['why']}")


def look_figure(source: np.ndarray, render: np.ndarray, held: np.ndarray, area: np.ndarray) -> tuple[np.ndarray, float]:
    """Per frame, where a render sits between the source (0) and a render that held the new subject (1).

    `source`, `render` and `held` are each frame's mean grey level over `area` (NaN where there is none);
    `held` may cover fewer frames. The lift is the median of what the held render added over the source on
    the frames it covers; the figure is the render's own difference from the source over that lift. It reads
    one thing only, how light or dark the area is, so it tells two subjects apart when they differ there (dark
    hair against fair) and says nothing when they do not: a lift near zero is refused by the caller."""
    both = ~np.isnan(source[:len(held)]) & ~np.isnan(held)
    lift = float(np.median(held[both] - source[:len(held)][both])) if both.any() else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):       # a lift of zero is the caller's to refuse
        return (render - source) / lift, lift


def look(a: argparse.Namespace) -> None:
    """Did a whole-subject render draw the new subject or go back to the original's look, frame by frame.

    Written for a fault no mask flag predicts (2026-10-10: one window of a long render drew a look-alike of
    the original, the next the new subject, with identical regions). Reads the subject's mask from a capture
    folder and compares the render with a render of the same subject that held."""
    folder = Path(a.capture)
    m = json.loads((folder / "manifest.json").read_text())
    w, h = m["size"]
    first, frames = m["first_frame"], m["frames"]
    subject = next(s for s in m["subjects"] if a.subject in (None, s["label"]))
    z = np.load(folder / "subjects" / subject["label"] / f"masks__{subject['sightings'][0]['by']}.npz")
    track = np.unpackbits(z["track"], axis=-1)[..., :w].astype(bool)

    def series(path: str, at: int, vf: str = "") -> np.ndarray:
        out = np.full(frames, np.nan)
        lead = max(at - first, 0)
        for n, img in enumerate(stream(path, (w, h), max(first - at, 0), max(frames - lead, 0), vf=vf), start=lead):
            box = box_of(track[n])
            if box is None:
                continue
            area = track[n].copy()
            area[box[1] + int(a.top * (box[3] - box[1])):] = False
            if area.sum() >= 500:
                out[n] = float(img[area].mean())
        return out

    def named(text: str) -> tuple[str, int]:
        path, _, at = text.partition("@")
        return path, int(at) if at else first

    src = series(a.source, 0, FIT.format(w=w, h=h))
    ren = series(*named(a.render))
    ref = series(*named(a.held))
    # Only where this render regenerated the subject: its own region when it is one of the capture's runs (the
    # capture's mask can come from another tracker run that took somebody else on other shots), or --frames.
    own = next((r for r in m["runs"] if r.get("render") == Path(named(a.render)[0]).name), None)
    if a.frames:
        keep = np.zeros(frames, bool)
        for span in a.frames.split("+"):
            lo, hi = (int(x) for x in (span.split("-") * 2)[:2])
            keep[max(lo - first, 0):max(hi - first + 1, 0)] = True
        ren[~keep] = np.nan
    elif own is not None:
        ren[~np.load(folder / "runs" / own["name"] / "region.npz")["region"].any(axis=(1, 2))] = np.nan
    seen = ~np.isnan(ref)
    figure, lift = look_figure(src, ren, ref[:int(np.nonzero(seen)[0].max()) + 1] if seen.any() else ref[:0], None)
    if not np.isfinite(lift) or abs(lift) < LOOK_LIFT:
        raise SystemExit(f"the held render differs from the source by {lift:.1f} grey levels over this area: too little to tell "
                         "the two subjects apart by it; give another --top or another reference")
    have = np.nonzero(~np.isnan(figure))[0]
    runs_of = frame_spans(have.tolist(), join=1)
    record = {"render": Path(named(a.render)[0]).name, "held_reference": Path(named(a.held)[0]).name, "subject": subject["label"],
              "area": f"the top {a.top:.2f} of the subject's mask", "lift_grey_levels": round(lift, 1), "spans": [],
              "figure_by_span_frame": [None if np.isnan(x) else round(float(x), 3) for x in figure]}
    print(f"{record['render']} against {record['held_reference']}: 0 is the source's look, 1 the held render's "
          f"(the held render is {lift:.1f} grey levels from the source over {record['area']})")
    for lo, hi in runs_of:
        seg = figure[lo:hi + 1]
        seg = seg[~np.isnan(seg)]
        row = {"span_frames": [lo, hi], "source_frames": [first + lo, first + hi], "median": round(float(np.median(seg)), 2),
               "min": round(float(seg.min()), 2), "max": round(float(seg.max()), 2),
               "reads_as": "the new subject" if np.median(seg) > 0.6 else ("the original's look" if np.median(seg) < 0.4 else "between")}
        record["spans"].append(row)
        print(f"  frames {lo}-{hi} (source {first + lo}-{first + hi}): median {row['median']:.2f}, from {row['min']:.2f} to {row['max']:.2f}: {row['reads_as']}")
    out = folder / f"look__{Path(named(a.render)[0]).stem}.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    print("wrote", out)


# ------------------------------------------------------------------ video

def load_capture(folder: Path):
    m = json.loads((folder / "manifest.json").read_text())
    w, h = m["size"]
    subjects = []
    for s in m["subjects"]:
        z = np.load(folder / "subjects" / s["label"] / f"masks__{s['sightings'][0]['by']}.npz")
        subjects.append({"label": s["label"], "colour": tuple(s["colour"]),
                         "track": np.unpackbits(z["track"], axis=-1)[..., :w].astype(bool)})
    runs = []
    for r in m["runs"]:
        z = np.load(folder / "runs" / r["name"] / "region.npz")
        back = {k[len("given_back__"):]: z[k] for k in z.files if k.startswith("given_back__")}
        runs.append({**r, "region": z["region"], "carried": np.unpackbits(z["carried"], axis=-1)[..., :w].astype(bool), "back": back})
    with open(folder / "frames.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return m, subjects, runs, rows


def zoom_boxes(focus: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A square box round `focus` ([n, h, w] of bool, any resolution) for every frame, smoothed; x, y, side."""
    import cv2
    w, h = size
    sx, sy = w / focus.shape[2], h / focus.shape[1]
    boxes = np.zeros((len(focus), 3), np.float32)
    last = (w / 2, h / 2, float(h))
    for n, f in enumerate(focus):
        box = box_of(f)
        if box:
            x0, y0, x1, y1 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
            last = ((x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 1.25)
        boxes[n] = last
    k = cv2.getGaussianKernel(25, 5)[:, 0]
    padded = np.pad(boxes, ((12, 12), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(padded[:, i], k, mode="valid") for i in range(3)], 1)
    side = np.clip(smooth[:, 2], ZOOM_MIN, h)
    return np.stack([np.clip(smooth[:, 0] - side / 2, 0, w - side), np.clip(smooth[:, 1] - side / 2, 0, h - side), side], 1).round().astype(int)


def draw_masks(plate: np.ndarray, n: int, subjects: list, runs: list) -> np.ndarray:
    """The middle row: the source dimmed, every subject in its colour. Fills first, outlines over them."""
    import cv2
    img = (plate.astype(np.float32) * 0.45)
    colour = {s["label"]: np.array(s["colour"], np.float32) for s in subjects}
    for r in runs:
        c = colour[r["subject"]]
        region, carried = cells_up(r["region"][n]), r["carried"][n]
        margin = region & ~carried
        img[margin] = img[margin] * (1 - LIGHT) + c * LIGHT
        img[carried] = img[carried] * (1 - SOLID) + c * SOLID
    img = img.astype(np.uint8)
    for s in subjects:
        contours, _ = cv2.findContours(s["track"][n].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, s["colour"], 2, cv2.LINE_AA)
    for r in runs:
        for cells in r["back"].values():
            for y, x in zip(*np.nonzero(cells[n])):
                cv2.rectangle(img, (int(x) * CELL, int(y) * CELL), (int(x + 1) * CELL - 1, int(y + 1) * CELL - 1), WHITE, 1)
    return img


def legend_lines(subjects: list, runs: list) -> list[tuple[tuple, str, str]]:
    lines = []
    for s in subjects:
        lines.append((s["colour"], "outline", f"{s['label']}: the tracked mask"))
        for r in runs:
            if r["subject"] == s["label"]:
                will = "would be " if r.get("planned") else ""
                lines.append((s["colour"], "solid", f"{s['label']}: {will}taken from the still ({'plan' if r.get('planned') else 'run'} {r['name']})"))
                lines.append((s["colour"], "light", f"{s['label']}: margin {will}regenerated round it"))
    if any(r["back"] for r in runs):
        lines.append((WHITE, "outline", "margin cells given back to another subject"))
    return lines


def video(a: argparse.Namespace) -> None:
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    folder = Path(a.capture)
    m, subjects, runs, rows = load_capture(folder)
    w, h = m["size"]
    first, frames = m["first_frame"], m["frames"]
    source, render = a.source, a.render
    if Path(source).name != m["clip"]:
        raise SystemExit(f"--source is {Path(source).name}; this capture was made from {m['clip']}")
    skip = 0
    if render is not None:
        # the render's frame 0 on the span's clock: the run that wrote it says, or --render-first
        known = next((r["first_source_frame"] for r in runs if r.get("render") == Path(render).name), first)
        skip = first - (known if a.render_first is None else int(a.render_first))
        if skip < 0:
            raise SystemExit("the render starts after the span's first frame; capture the span the render covers")
        frames = min(frames, probe(render)[2] - skip)
        # a render that is one of the capture's runs is shown with that run's region alone; any other with all
        own = [r for r in runs if r.get("render") == Path(render).name]
        runs = own if own and not a.all_runs else runs
    focus = np.zeros((m["frames"], h // CELL, w // CELL), bool)
    for r in runs:
        focus |= r["region"]
    if not focus.any():
        focus = cells_any(subjects[0]["track"])
    boxes = zoom_boxes(focus, (w, h))
    z = h                                             # the zoom tile is the row's height, square
    row_h, n_rows = h + HEADER, 3 if render else 2
    canvas_w, canvas_h = w + z, row_h * n_rows + PANEL
    big, text, small = ImageFont.truetype(MONO, 30), ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 17)
    huge, line = ImageFont.truetype(MONO, 92), ImageFont.truetype(FONT, 19)
    # the panel's numbers come from the same tables the folder holds, so the picture and the tables agree
    table = {s["label"]: [r for r in json.loads((folder / "subjects" / s["label"] / "per_frame.json").read_text())["rows"]
                          if r["seen_by"] == next(x for x in m["subjects"] if x["label"] == s["label"])["sightings"][0]["by"]]
             for s in subjects}
    run_table = {r["name"]: json.loads((folder / "runs" / r["name"] / "per_frame.json").read_text())["rows"] for r in runs}
    flags = json.loads((folder / "flags.json").read_text())["flags"] if (folder / "flags.json").is_file() else None
    starts = sorted(int(x) for x in a.windows.split(",")) if a.windows else []
    stills = []
    for item in a.still:
        label, _, path = item.partition("=")
        pic = Image.open(path)
        pic = pic.convert("RGB").resize((max(1, round(pic.width * 150 / pic.height)), 150))
        stills.append((label, pic))
    motions = stream(a.motion, (456, 342), skip, frames, vf="scale=456:342:force_original_aspect_ratio=decrease,pad=456:342:(ow-iw)/2:(oh-ih)/2",
                     pix="rgb24") if a.motion else None
    region_px = [cells_up(r["region"]) for r in runs]
    before = None
    lines = legend_lines(subjects, runs)
    out = a.out or str(folder / f"{m['name']}_diag.mp4")
    audio = ["-i", render] if render else ["-ss", f"{first / FPS:.6f}", "-i", str(source)]
    if render and skip:
        audio = ["-ss", f"{skip / FPS:.6f}"] + audio
    enc = subprocess.Popen(
        ["nice", "-n", "19", "ffmpeg", "-v", "error", "-y", "-threads", "4", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{canvas_w}x{canvas_h}", "-r", str(FPS), "-i", "-", *audio, "-map", "0:v:0", "-map", "1:a:0?",
         "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-t", f"{frames / FPS:.3f}", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    plates = stream(str(source), (w, h), first, frames, vf=FIT.format(w=w, h=h), pix="rgb24")
    renders = stream(render, (w, h), skip, frames, vf=f"scale={w}:{h}", pix="rgb24") if render else None
    names = ["SOURCE", "MASKS", "RENDER"]
    render_name = Path(render).name if render else ""
    done = 0
    for n, plate in enumerate(plates):
        tiles = [plate, draw_masks(plate, n, subjects, runs)] + ([next(renders)] if render else [])
        x, y, side = boxes[n]
        canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)
        for k, tile in enumerate(tiles):
            top = k * row_h + HEADER
            canvas[top:top + h, :w] = tile
            canvas[top:top + h, w:] = cv2.resize(tile[y:y + side, x:x + side], (z, z),
                                                 interpolation=cv2.INTER_AREA if side >= z else cv2.INTER_CUBIC)
        cv2.rectangle(canvas, (int(x), HEADER + row_h + int(y)), (int(x + side) - 1, HEADER + row_h + int(y + side) - 1), DIM, 1)
        img = Image.fromarray(canvas)
        d = ImageDraw.Draw(img)
        voiced = rows[n].get("voiced")
        said = ("", DIM) if voiced in (None, "") else (("VOICE", GREEN) if int(float(voiced)) else ("no voice", DIM))
        for k in range(n_rows):
            top = k * row_h
            d.text((10, top + 6), f"SOURCE FRAME {first + n:05d}  RENDER FRAME {skip + n:04d}", font=big, fill=WHITE)
            d.text((w - 250, top + 10), f"{(first + n) / FPS:6.2f} s", font=text, fill=DIM)
            d.text((w - 140, top + 10), said[0], font=text, fill=said[1])
            d.text((w + 12, top + 10), f"{names[k]}, zoomed on the region" + (f"   {render_name}" if k == 2 else ""),
                   font=text, fill=AMBER)
        ly = row_h + HEADER + h - 12 - 24 * len(lines)
        for colour, kind, label in lines:
            fill = None if kind == "outline" else tuple(int(c * (0.85 if kind == "solid" else 0.4)) for c in colour)
            d.rectangle([12, ly + 3, 30, ly + 19], fill=fill, outline=colour, width=2)
            for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                d.text((38 + dx, ly + dy), label, font=small, fill=(0, 0, 0))
            d.text((38, ly), label, font=small, fill=WHITE)
            ly += 24
        # ---- the panel: one frame of it should explain itself
        py = n_rows * row_h
        d.line([(0, py), (canvas_w, py)], fill=DIM)
        d.text((14, py + 6), f"{first + n:05d}", font=huge, fill=WHITE)
        d.text((16, py + 104), f"source frame     {(first + n) / FPS:.2f} s", font=line, fill=DIM)
        d.text((16, py + 130), f"render frame {skip + n:04d}" if render else "nothing rendered: a preview", font=text, fill=WHITE)
        if starts:
            window = sum(skip + n >= x for x in starts)
            d.text((16, py + 160), f"window {window} of {len(starts)}", font=line, fill=WHITE)
            d.text((16, py + 184), "new frames from " + ", ".join(str(x) for x in starts), font=small, fill=DIM)
        d.text((16, py + 214), said[0] or "no voice table", font=text, fill=said[1])
        inside = np.zeros((h, w), bool)
        for px in region_px:
            inside |= px[n]
        grey = [cv2.cvtColor(t, cv2.COLOR_RGB2GRAY).astype(np.int16) for t in (tiles[0], tiles[-1])]
        both = inside & before[2] if before is not None else inside
        moved = None if before is None or not both.any() else (float(np.abs(grey[0] - before[0])[both].mean()),
                                                               float(np.abs(grey[1] - before[1])[both].mean()))
        before = (grey[0], grey[1], inside)
        tx, ty = 330, py + 10
        for sub_ in subjects:
            r = table[sub_["label"]][n]
            note = "no mask video on this frame" if not r["covered"] else (
                f"mask {100 * r['track_share']:.2f}% of the frame" + (f", in {r['track_pieces']} pieces" if r.get("track_pieces", 1) > 1 else "")
                + (f"; part {100 * r['parts_share']:.2f}%, {100 * (r.get('parts_inside_track') or 0):.0f}% of it on the subject"
                   if r.get("parts_share") is not None else ""))
            d.rectangle([tx, ty + 4, tx + 14, ty + 18], outline=sub_["colour"], width=3)
            d.text((tx + 24, ty), f"{sub_['label']}: {note}", font=line, fill=WHITE)
            ty += 25
        for r_ in runs:
            r = run_table[r_["name"]][n]
            if r.get("region_share"):
                own = table[r_["subject"]][n].get("track_share") or 0
                d.text((tx, ty), f"{'plan' if r_.get('planned') else 'run'} {r_['name']}: region {100 * r['region_share']:.1f}% of the frame"
                       + (f", {r['region_share'] / own:.1f}x {r_['subject']}'s mask" if own else "")
                       + f"; {100 * r['margin_share'] / r['region_share']:.0f}% of it margin", font=line, fill=WHITE)
            else:
                d.text((tx, ty), f"{r_['name']}: nothing regenerates on this frame", font=line, fill=AMBER)
            ty += 25
            for key, value in rows[n].items():
                if key.startswith(f"region_on__{r_['name']}__") and value not in ("", None) and float(value) > 0:
                    other = key.rsplit("__", 1)[1]
                    back = rows[n].get(f"cells_given_back__{r_['name']}__{other}") or 0
                    d.text((tx + 24, ty), f"covers {100 * float(value):.0f}% of {other}'s mask; {int(float(back))} margin cells given back",
                           font=line, fill=AMBER)
                    ty += 25
        if moved is not None:
            d.text((tx, ty), f"change in the region since the last frame: source {moved[0]:.1f}"
                   + (f", render {moved[1]:.1f}" if render else "") + " grey levels", font=line, fill=WHITE)
            ty += 25
        if flags is None:
            d.text((tx, ty + 4), "no preflight was run on this capture", font=small, fill=DIM)
        else:
            shown = {r_["name"] for r_ in runs}
            live = [f for f in flags if any(x <= first + n <= y for x, y in f["source_frames"]) and f["level"] != LEVELS[0]
                    and f.get("run") in (None, *shown)]
            if not live:
                d.text((tx, ty + 4), "no flag on this frame", font=small, fill=DIM)
            for f in live[:max(1, (py + PANEL - ty - 8) // 23)]:
                words = f"{f['id']} {f['level'].upper()}: {f['why']}"
                while d.textlength(words, font=small) > PANEL_TEXT - tx and len(words) > 20:
                    words = words[:-8] + "..."
                d.text((tx, ty + 4), words, font=small, fill=LEVEL_COLOURS[f["level"]])
                ty += 23
        mx = PANEL_TEXT + 12
        d.text((mx, py + 8), "movement, as the model was shown it", font=small, fill=AMBER)
        if motions is not None:
            shown = next(motions, None)
            if shown is not None:
                img.paste(Image.fromarray(shown), (mx, py + 34))
        else:
            d.text((mx, py + 150), "none: this run has no motion video" if not a.motion_note else a.motion_note, font=line, fill=DIM)
        sx = mx + 456 + 14
        d.text((sx, py + 8), "taken from" if stills else "", font=small, fill=AMBER)
        sy = py + 34
        for label, pic in stills:
            img.paste(pic, (sx, sy))
            d.rectangle([sx, sy + 150, sx + pic.width - 1, sy + 172], fill=(0, 0, 0))
            d.text((sx + 2, sy + 151), label, font=small, fill=WHITE)
            sy += 180
        enc.stdin.write(np.asarray(img).tobytes())
        done = n + 1
        if n % 100 == 0:
            print("frame", n, flush=True)
    enc.stdin.close()
    code = enc.wait()
    print("wrote", out, "frames", done, "encoder exit", code)
    if done != frames or code:
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("files", help="build the capture folder from saved mask videos and renders")
    f.add_argument("--name", required=True)
    f.add_argument("--date", default=datetime.date.today().isoformat())
    f.add_argument("--source", required=True, help="the clip every run was loaded from")
    f.add_argument("--first", type=int, required=True, help="the source frame that is the span's frame 0")
    f.add_argument("--frames", type=int, required=True)
    f.add_argument("--mask", action="append", default=[], metavar="LABEL:track=V[,parts=V][,classes=V][,held=V][,shots=J][,shots_at=N][,pose=J][,pose_at=N][,by=RUN][,at=N][,hold=A-B+C-D][,drop=A-B+C-D][,grade=off]",
                   help="one sighting of a subject: its saved mask videos; repeatable")
    f.add_argument("--run", action="append", default=[], metavar="RUN:render=V|preview=DIR,subject=LABEL[,others=L+L][,margin=PX][,at=N][,region=review]",
                   help="one pass that regenerated a subject; its region is the one each window saved beside its latent, or is "
                        "read from the render's review when there is none (or with region=review). With preview= and no "
                        "render, a pass that has not rendered: its region is the one the song node's preview planned for "
                        "each window (the plan file or its windows folder); repeatable")
    f.add_argument("--plan", action="append", default=[], metavar="RUN:subject=LABEL,margin=PX[,others=L+L][,keep=L+L][,carried=track|held][,edge=cells][,window=F,context=F][,at=FIRST,frames=N][,text=LABEL][,audio=FILE][,still=FILE]",
                   help="a run that has not rendered: its region is worked out from the masks; with at= and frames= it is one "
                        "load, on those frames only; repeatable")
    f.add_argument("--loads", action="append", default=[], metavar="FILE.json",
                   help='a pass as a list of loads: {"defaults": {...}, "loads": [{"first", "frames", "subject", "text", ...}]}; '
                        "each load is a plan; repeatable")
    f.add_argument("--voice", help="a per-frame voice table (frame, voiced, vocals_stem_dbfs) on the clip's frames")
    f.add_argument("--out", default=str(REPO / "data"))
    f.add_argument("--overwrite", action="store_true", help="rebuild a finished capture folder in place; only when "
                   "no queued render loads its mask videos")
    v = sub.add_parser("video", help="the stacked, frame-numbered video from a capture folder")
    v.add_argument("capture")
    v.add_argument("--source", required=True, help="the clip the capture was made from")
    v.add_argument("--render", help="the render shown in the bottom row; two rows without one")
    v.add_argument("--render-first", type=int, help="the source frame that is the render's frame 0, when the render "
                   "is not one of the capture's runs (a merged file, say); a run's own is in the manifest")
    v.add_argument("--motion", help="the motion video the run was given, frame 0 its first frame, shown in the panel")
    v.add_argument("--motion-note", default="", help="words for the panel when there is no motion video file to show")
    v.add_argument("--still", action="append", default=[], metavar="LABEL=IMAGE", help="a reference still and the subject "
                   "it belongs to, shown small in the panel; repeatable")
    v.add_argument("--windows", help="the render frames where each window's new frames start, e.g. 0,345")
    v.add_argument("--all-runs", action="store_true", help="draw every run's region, also when --render is one run's")
    v.add_argument("--out")
    v = sub.add_parser("verify", help=f"does the capture read what the nodes wrote: writes verify.json, exits {VERIFY_FAILED} when not")
    v.add_argument("capture")
    g = sub.add_parser("preflight", help="the risks in a capture folder, before a render: writes flags.json")
    g.add_argument("capture")
    g.add_argument("--not-in", action="append", default=[], metavar="LABEL:FIRST-LAST",
                   help="source frames the subject is not in, as the caller knows the clip; repeatable")
    g.add_argument("--text", help="the prompt file of the run this capture is for")
    g.add_argument("--voice-spans", help="voice_spans.json for the clip")
    g.add_argument("--gate", action="store_true", help=f"exit {GATE_BLOCKED} when a flag at the top level stands with no override")
    o = sub.add_parser("outcome", help="record what a render did against one flag")
    o.add_argument("capture")
    o.add_argument("flag")
    o.add_argument("happened", choices=("yes", "no", "overridden"),
                   help="what a render did against the flag, or `overridden`: render past it, with --by and --note")
    o.add_argument("--render", help="the render it was seen in, by file name; not needed for an override")
    o.add_argument("--note", default="")
    o.add_argument("--by", default="", help="who looked")
    k = sub.add_parser("look", help="did a whole-subject render draw the new subject or the original's look, by frame")
    k.add_argument("capture")
    k.add_argument("--source", required=True)
    k.add_argument("--render", required=True, metavar="R.mp4[@FIRST]", help="the render to read; FIRST is the source frame of its frame 0")
    k.add_argument("--held", required=True, metavar="REF.mp4[@FIRST]", help="a render of the same subject that held the new one")
    k.add_argument("--subject", help="the subject's label; the first when not given")
    k.add_argument("--frames", metavar="A-B+C-D", help="source frames to read; when not given and the render is one of the "
                   "capture's runs, the frames its region is not empty on")
    k.add_argument("--top", type=float, default=0.45, help="the share of the subject's mask, from its top, the figure is read over")
    c = sub.add_parser("changed", help="what a run redrew, per segment and subject, against the floor")
    c.add_argument("capture")
    c.add_argument("--run", required=True)
    c.add_argument("--source", required=True)
    c.add_argument("--render", required=True)
    c.add_argument("--frames", metavar="FIRST-LAST", help="source frames; the whole span when not given")
    c.add_argument("--series", metavar="LABEL.Class", help="print this segment's difference from the source frame by frame")
    c.add_argument("--bin", type=int, default=24, help="frames a bin of the series")
    c.add_argument("--around", type=int, action="append", help="a source frame to print the step and the series round; repeatable")
    u = sub.add_parser("mouth", help="a render's mouth against the source's, frame by frame")
    u.add_argument("capture")
    u.add_argument("--subject", required=True)
    u.add_argument("--reference", metavar="MASK[@FIRST][:classes]", help="the source's mouth mask (or a class map of it); the "
                   "subject's own class map in the capture when not given")
    u.add_argument("--arm", action="append", default=[], metavar="NAME=MASK[@FIRST][:classes]", help="a render's mouth mask; repeatable")
    u.add_argument("--frames", metavar="FIRST-LAST", help="source frames to score; the whole span when not given")
    u.add_argument("--voice", help="the clip's per-frame voice table, for a capture made without one")
    x = sub.add_parser("mask", help="some of a subject's classes as a lossless mask video")
    x.add_argument("capture")
    x.add_argument("--subject", required=True)
    x.add_argument("--classes", metavar="Class+Class", help="classes as the class map has them, joined by +")
    x.add_argument("--classes-owned", metavar="Class+Class", help="classes taken only where the owner map gives the subject the pixel")
    x.add_argument("--out", required=True, help="the .mkv to write: ffv1, grey, white on the mask, frame 0 the span's first frame")
    d = sub.add_parser("diagnose", help="everything the capture says about a stretch marked as wrong")
    d.add_argument("capture")
    d.add_argument("--frames", required=True, metavar="FIRST-LAST", help="source frames")
    a = p.parse_args()
    {"files": files, "video": video, "verify": verify, "preflight": preflight, "outcome": outcome, "diagnose": diagnose, "look": look, "mask": mask, "changed": changed, "mouth": mouth}[a.mode](a)


if __name__ == "__main__":
    main()
