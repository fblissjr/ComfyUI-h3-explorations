#!/usr/bin/env python3
"""Capture what the tracker and the masks did on a masked run, for any number of subjects, and show it.

    <python> bench/capture_masked_run.py files --name NAME --source CLIP --first N --frames F \\
        --mask a:track=A_track.mp4,parts=A_parts.mp4,shots=A_shots.json,by=preview_a \\
        --mask b:track=B_track.mp4,by=preview_a \\
        --run pass_a:render=A.mp4,subject=a,others=b [--voice voice_per_frame.csv]
    <python> bench/capture_masked_run.py files --name NAME ... --mask a:... --mask b:... \
        --plan pass_a:subject=a,others=b,margin=64          # before anything samples: the region a run WOULD take
    <python> bench/capture_masked_run.py verify data/<date>_<NAME> [--no-mask b:1632-1729="walks out of the shot" --by WHO]
    <python> bench/capture_masked_run.py preflight data/<date>_<NAME> [--not-in a:294-414] [--text P.txt --voice-spans V.json]
    <python> bench/capture_masked_run.py video data/<date>_<NAME> --source CLIP [--render R.mp4] [--out NAME_diag.mp4]
    <python> bench/capture_masked_run.py shots data/<date>_<NAME> [--loads]   # the gate per shot and subject, or per load
    <python> bench/capture_masked_run.py video data/<date>_<NAME> --source CLIP --job JOB.json --load LOAD     # the look's strip, one a load
                                                    # a load of two subjects: --motion twice, in the list's order, and the heading says which label is whose
    <python> bench/capture_masked_run.py looked data/<date>_<NAME> --by WHO --note "every frame of each load and its text, as the strips show them"
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
taken in (the top level for a subject a load replaces; for one who is only an other of the loads it is data, with
their size beside the gap and how far the load's region lies from their last box); and a text that names a voice over frames with none, or the reverse (`--text`, `--voice-spans`).
With a pose table (`pose=` on `--mask`), before its mesh is used as a motion video: a body fitted to the whole
frame and not to the subject's box, top level when another subject is in the frame; several bodies under one
name; a frame with a mask and no body; a hand the hand decoder did not refine; a body mostly outside the frame
(`flag_pose`).

**verify** is the first check of a job, before any plan is trusted: does the capture read what the nodes wrote.
It reads the capture folder only and writes `verify.json`; exit 0 when every check held, `VERIFY_FAILED` when
one did not or could not be made (a check that cannot run is not a pass). The checks: every mask video of a
sighting was written in one run (`WRITTEN_TOGETHER_S`: a folder that holds two previews under one name is
how a morning's mask gets loaded in the afternoon), is at the capture's canvas and reaches the span; a
subject's track is not empty on frames its shot table says it was taken in (a subject a load replaces; an
other's empty track passes as data with its figures); a plan that carries a part has
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
emptied; `grade=off` in the spec leaves the fill ungraded. The grade also empties a part smaller than a floor,
which is a STARTING VALUE and not this clip's (`PART_LEAST_PX`): the capture prints the part's size on this
clip's frames against it for every subject (`part_sizes`), and `least=PX` in the spec sets it for that subject
(`least=0` empties nothing by size). `sizes CAPTURE...` prints the same figures for captures that exist.
`drop=FIRST-LAST+FIRST-LAST` empties the part on source frames where it should take nothing (it lies on
something that is not the part and there is nothing to fill it from). A plan with `carried=held` works its
region out from that mask. The manifest lists the frames filled, the doubted frames left and those emptied. Written only when the mask video covers
the whole span.

**video** stacks the source, the masks and a render, each with a zoom on the region beside it, and burns the
source frame number and the render frame number above every row, with the render's audio, or the source's at the
render's first frame when the render file carries none. A render need not begin or end where the capture's span
does (a load whose first frames are left as the source, a run that continues a stored window): the strip is the
frames both hold, `render_span` owns the arithmetic, and every frame of it names in the panel the span's frames
it does not show. The render's frame 0 is the source frame `--render-first` gives, or the one the run that
wrote the file recorded, or the span's first frame when neither says, and the command prints which. Every subject has
one colour: an outline for the tracked mask, a solid fill for what was taken from the still, a light fill for
the margin regenerated round it, and white boxes on margin cells that were given back to somebody else. Under
the rows is a panel read from the same tables: the frame number large, each subject's mask and part, each
run's region against its subject, what it covers of anybody else, the change inside the region since the
last frame for source and render, the preflight's flags that are live on that frame, the motion video the
model was shown (`--motion`) or the word none, and the reference stills (`--still`). One frame pulled as a
still should explain itself. The form follows the two reviews the owner called the best they had had
(2026-10-09, a numbered tracked review and a vertical how-it-was-made cut, both made by session scripts that
are gone or tied to one clip); colour by part and contacts are not in it yet.

**What it does not hold.** Anything inside the tracker beyond its shot table (each call); the part model's
own doubt; the matte. For a render made before the song node saved each window's region, the token mask
itself: the region is then read back from a compressed, tinted picture, cell by cell, and under the review's
legend from a quarter of the signal (`verify` says how far that reading is from the node's own file where
both exist). `taken_back` is worked out from the masks with the node's own `grow`, under a fixed margin
only. Whether a render looks like its reference still: every reading here asks whether it differs from the
source.

**The lines the gate judges by, and where each comes from.** A rule that depends on the picture should take
its value from the clip in hand, measured in the no-sampling preview (owner, 2026-10-10; `AGENTS.md`). Almost
none of these does yet. This list is where a change to any of them starts; the value and its own provenance
are beside each constant. Three kinds: THIS CLIP (measured on the capture's own files each time), CARRIED
(measured on some other clip or render and kept as a starting value), REASONED (never measured).

    THIS CLIP
      the floor in `changed`             the render's own difference from the source outside its region
      the level in `ends`                the load's own median difference under the mask it carried
      the scale in `look`                the source and a held render of the same frames
      the bar in `open_runs`             the shot's own mouth openings (`OPEN_SHARE` of them is REASONED)
      the tracker's match and cut lines  the tracker's, in its shot table; this tool only reads them
    CARRIED, with what a preview of the clip could measure in its place
      FACE_TOKENS                        unmeasured even on its own clip. The face sizes at which this clip's
                                         own accepted renders hold: the outcomes of its first face load
      KEPT_TURN                          another clip's facing tolerance; one case here. The turn at which
                                         this clip's own continuations held or failed, once one has rendered;
                                         before that, the spread of the source's own head angle within a shot
      SMALL_SUBJECT                      one other clip. The subject's share of the frame on the shots of this
                                         clip whose renders held
      HOLD_INSIDE                        two shots of one clip. The score of the saved part on this clip's own
                                         undoubted frames: a bar just under their lowest
      KEPT_IN_PART                       one render's outcome. Nothing in a preview; an outcome on this clip
      SAME_PERSON                        one stretch. The overlaps of this clip's own tracks, which fall in two
                                         groups when one tracker is on another's person
      READERS_CELLS, READERS_CARRIED_OFF five renders of one clip. The same comparison on this job's first render
      END_LAST, END_RUN                  seven renders of one clip. The spread of the load's own curve away from
                                         its ends, so that a noisy load does not read as a fall
      MOUTH_PX, MOUTH_APART              a handful of frames read by eye on one clip. The mouth's size against the
                                         face part's on this clip
      NEAR_UNDER                         one clip (how far under the line a present subject scored). The tracker's
                                         own gap between a shot's first-look scores
    REASONED
      NEAR_LINE, PART_INSIDE, PART_SIZE, PART_MOVE, RECENT, HOLD_REACH   the part and track rules. Each could be
                                         the clip's own frame-to-frame spread of the same figure on frames
                                         nobody doubts, in place of a fixed share
      REGION_ON_OTHER, NOT_SUBJECT, OWN_SHARE, SEGMENT_PX   shares and a pixel count for "inside a region".
                                         SEGMENT_PX should scale with the canvas and the subject's size
      SMALL_START, POSE_OUTSIDE, OPEN_SHARE, OPEN_RUN, END_LEAST, LOOK_LIFT
      WRITTEN_TOGETHER_S                 not about the picture: how long a preview takes to write its files

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
#: The least of the STRANGERS mask's pixels inside a run's region on a frame for that frame to count toward `region_over_another_subject`
#: at the top level: a share of a mask a few pixels across means nothing (the lead, 2026-10-14: on second_hfp_0014d the strangers
#: mask was specks of 1-657 px on her own apron edge and wrist, read as "up to 100%"). A STARTING VALUE, reasoned: four 16 px
#: latent cells, the least a person's face or hand redrawn could fill. Frames under it are said as data, with their size.
REGION_ON_OTHER_LEAST_PX = 1024
#: The strangers keep-out's label in a capture (`bench/strangers.py`, `bench/shot_to_load.py`): the one other the floor applies to.
STRANGERS_LABEL = "strangers"
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
#: A sentence that names a reference picture says what is taken from a still, never what is performed, and is not a
#: voice sentence whatever its words. Measured 2026-10-13 on the lane's texts under `data/`: every sentence there
#: that holds a voice word and names a picture is a retention line ("only the face is retained: her eyes,
#: eyebrows, nose, mouth and skin; the hair ... of <Picture 1> are not used"), read until then as a voice denied,
#: or, without its "not", as a voice performed; no sentence of a performance names a picture.
REFERENCE_PICTURE = "<picture"
#: A sentence that defines a subject as what a reference video holds ("<Subject 2> is the movements ... performed by the
#: figure in <Video 1>: a closer view of ...") is read for a voice by its head, before the colon: after it, the sentence
#: says what the reference looks like. Reasoned, 2026-10-14: the drawn mouth's words (`load_text.subject_2`, "the mouth
#: is outlined ... and it opens and closes where the person's mouth does") were read as a voice on lead_fp_1097c and
#: second_hfp_0014d, which have none; the head still counts ("<Subject 2> is the singing performed by ... <Video 1>").
REFERENCE_VIDEO_HEAD = r"<subject \d+> is [^:]*<video \d+>:"
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
# The least area a part has and is still loaded: two tokens a side (a token is 32 px of the canvas). Reasoned: under
# that the model has nothing to draw the part in. Held once against a reading by eye (2026-10-10, two face loads of
# one clip, every frame, by a session that did not make the masks): on one load it empties exactly the frames the
# reader called fragments; on the other it empties most of the stretch the reader cut and leaves a few frames of
# it standing. So it is a floor, not a judge of a part. The class map cannot see these frames: it calls the
# fragment the part's own class, which is how the part came to be there.
# A STARTING VALUE, NOT THE CLIP'S (2026-10-13; the owner's answer on the Open List's row p07, 2026-10-12). It was
# reasoned from the model's token and held against that one clip. On 2026-10-12 it emptied a small face on about
# half the frames of a load of another clip whose whole-frame render the owner had accepted, so the reasoning
# ("nothing to draw the part in") does not hold as a rule. A capture therefore says, for every subject with a
# part, what size the part is on this clip's own frames and where the floor falls among them (`part_sizes`:
# printed, and written to the manifest and to `part_grades.json`), and `least=PX` on a `--mask` sets the floor for
# that subject from those figures (`least=0` empties nothing by size). `sizes` prints the same for a capture
# that exists. Nothing here judges the figures: a floor that empties many frames is said, not refused.
PART_LEAST_PX = 4 * 32 * 32
#: Who set a sighting's floor, as the record says it.
LEAST_STARTING = ("the starting value (PART_LEAST_PX): reasoned from the model's token and held once against another "
                  "clip; not set from this clip")
LEAST_GIVEN = "this capture's --mask (least=)"
#: What the sizes cannot say, carried with them wherever they are written.
SIZES_CANNOT_SEE = ("whether a small part is a sliver of the part or a whole small face (the class map calls both the "
                    "part's class): look at a frame; and the size the part is drawn at on a load rendered zoomed in, "
                    "which is larger than these whole-frame pixels by the zoom, squared")


#: What the pixels `pieces=largest` took away cannot say, carried with them wherever they are written.
PIECES_CANNOT_SEE = ("whether a piece taken away was the part (a face behind a raised arm or behind hair is rightly in two pieces, and "
                     "the smaller one is then lost) or something else the class map calls the part: look at a frame it names")


def pieces_taken(stray: np.ndarray, first: int) -> dict:
    """What `pieces=largest` took away from a part, a frame at a time: `stray` is each frame's pixels away from its
    largest piece (`largest_piece`). Figures and what they cannot see; nothing is judged.

    With the option set, `flag_pieces` says nothing of the subject (the pieces are gone from the part it reads), so
    until 2026-10-13 the choice showed in one key of the manifest and nowhere a person reads."""
    hit = np.nonzero(stray > 0)[0]
    return {"frames": int(len(hit)), "frames_read": int(len(stray)), "px_most": int(stray.max()) if len(stray) else 0,
            "px_by_source_frame": [[int(first + f), int(stray[f])] for f in hit], "cannot_see": PIECES_CANNOT_SEE}


def pieces_lines(label: str, taken: dict) -> list[str]:
    """What the files step prints of `pieces_taken`, every frame named with its pixels."""
    if not taken["frames"]:
        return [f"    pieces=largest is set for {label} and took nothing away: the part is one piece on every frame read"]
    return [f"    PIECES=LARGEST TOOK PIXELS OF {label}'s PART AWAY on {taken['frames']} of {taken['frames_read']} frame(s), up to {taken['px_most']} px: "
            "each frame's largest piece is kept and the others are not loaded. Source frame: px taken: "
            + ", ".join(f"{f}: {px}" for f, px in taken["px_by_source_frame"]),
            f"    cannot see: {taken['cannot_see']}"]


def part_sizes(parts: np.ndarray, first: int, least_px: int, set_by: str, emptied: list[int] | None = None,
               not_applied: str = "") -> dict:
    """The size of a subject's part on each frame of the span, and where the floor falls among them.

    Measured on the saved part (`parts__<by>.mkv`, before any fill), in pixels of the capture's own canvas, over
    the frames that have a part: the least, the most and five percentiles; how many of those frames are under
    the floor and which source frames; and how many are under a quarter, a half and twice the STARTING value,
    so a reader can see whether the small frames sit apart from the rest or run into them. `emptied` is the
    frames (of the span) the floor emptied in the part to load, which is counted after the fills and can differ
    from the count on the saved part; `not_applied` says why nothing was emptied by size on this sighting.
    Figures and nothing else: no level here is good or bad."""
    px = parts.reshape(len(parts), -1).sum(axis=1)
    has = px > 0
    under = np.nonzero(has & (px < least_px))[0]
    levels = sorted({PART_LEAST_PX // 4, PART_LEAST_PX // 2, PART_LEAST_PX, PART_LEAST_PX * 2} | ({int(least_px)} if least_px else set()))
    return {"measured_on": "the saved part, before any fill, in pixels of the capture's canvas, on the frames that have a part",
            "frames": int(len(px)), "frames_with_a_part": int(has.sum()),
            "px": {name: int(np.percentile(px[has], q)) for name, q in (("least", 0), ("p10", 10), ("p25", 25), ("median", 50),
                                                                        ("p75", 75), ("p90", 90), ("most", 100))} if has.any() else None,
            "floor_px": int(least_px), "floor_set_by": set_by,
            "frames_under_the_floor": int(len(under)), "source_frames_under_the_floor": frame_spans([first + int(f) for f in under]),
            "frames_under_px": {str(level): int((has & (px < level)).sum()) for level in levels},
            "emptied_by_the_floor": None if emptied is None else {"frames": len(emptied), "source_frames": frame_spans([first + int(f) for f in sorted(emptied)])},
            "floor_not_applied": not_applied or None,
            "cannot_see": SIZES_CANNOT_SEE}


def size_lines(where: str, sizes: dict) -> list[str]:
    """What `part_sizes` found, as lines to print: the figures, the floor and who set it, and what it emptied."""
    if not sizes["frames_with_a_part"]:
        return [f"{where}: the part is on no frame of the span's {sizes['frames']}; nothing to size"]
    px, spans = sizes["px"], lambda rows: ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in rows)
    lines = [f"{where}: the part is on {sizes['frames_with_a_part']} of {sizes['frames']} frames; its size in px: least {px['least']}, "
             f"p10 {px['p10']}, p25 {px['p25']}, median {px['median']}, p75 {px['p75']}, p90 {px['p90']}, most {px['most']}; "
             + "frames under " + ", ".join(f"{level} px: {n}" for level, n in sizes["frames_under_px"].items())]
    floor = f"    the floor is {sizes['floor_px']} px, set by {sizes['floor_set_by']}; under it on the saved part: {sizes['frames_under_the_floor']} frame(s)"
    if sizes["frames_under_the_floor"]:
        floor += f" (source frames {spans(sizes['source_frames_under_the_floor'])})"
    lines.append(floor)
    emptied = sizes["emptied_by_the_floor"]
    if emptied is None:
        lines.append(f"    nothing was emptied by size on this sighting: {sizes['floor_not_applied']}")
    elif emptied["frames"]:
        lines.append(f"    EMPTIED BY THE FLOOR in the part to load: {emptied['frames']} frame(s), source frames {spans(emptied['source_frames'])}. "
                     "The original stays on those frames. To set the floor from the figures above: least=PX on this subject's "
                     "--mask (least=0 empties nothing by size)")
    else:
        lines.append("    the floor emptied no frame of the part to load")
    lines.append(f"    cannot see: {sizes['cannot_see']}")
    return lines


def grade_holds(parts: np.ndarray, filled: np.ndarray, classes: np.ndarray, made: list[int],
                least_px: int = 0) -> tuple[np.ndarray, dict[int, dict]]:
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
    figure the part's class scores the saved part high; `drop=` is still the caller's. One case of that it
    does catch, by size: a part left under `least_px` (`PART_LEAST_PX`) is emptied, whatever the class map
    says of it, and the decision says how many pixels it had. An emptied frame is then a frame with no part,
    which the gate names at its top level."""
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
    for f in range(len(parts)):
        px = int(out[f].sum())
        if 0 < px < least_px:
            out[f] = False
            decisions[f] = {**decisions.get(f, {"saved_on_its_classes": inside(parts[f], f), "fill_on_its_classes": None}),
                            "did": "emptied", "why": "too small to be drawn as the part", "px": px, "least_px": least_px}
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
                part_box = box_of(p)
                row.update(parts_share=round(float(p.sum()) / area, 5), parts_box=part_box,
                           parts_height_px=None if part_box is None else part_box[3] - part_box[1],
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


def picture_rows(source: str, first: int, frames: int, size: tuple[int, int]) -> list[tuple[float | None, float]]:
    """Two crude figures of the source's own picture, a frame: how much the whole frame changed from the frame
    before (mean grey levels; None on the first), and how sharp it is (the variance of its Laplacian).

    They rank a clip's shots against each other (a locked-off shot against a hand-held one, soft old film
    against sharp video) so that a line can be set from the clip in hand. Neither is a measure of anything
    absolute: the first rises with motion of any kind, the second with grain as well as with detail."""
    import cv2
    out, before = [], None
    for frame in stream(source, size, first, frames, vf=FIT.format(w=size[0], h=size[1])):
        grey = frame.astype(np.float32)
        out.append((None if before is None else round(float(np.abs(grey - before).mean()), 2),
                    round(float(cv2.Laplacian(grey, cv2.CV_32F).var()), 1)))
        before = grey
    return out + [(None, None)] * (frames - len(out))


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
                if b in subjects_of(r) or not lead[b][1][n]:
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


#: What a zoomed run's record says of its cells, carried with them.
ZOOMED_CELLS_ARE = ("the node's token region and fitted mask are of the zoomed picture; they were laid on the canvas as the node lays the render "
                    "(`video_mask.lay_back`) and cut at a half. A cell of this capture is in the region when any pixel of it is, so the "
                    "region here covers up to one cell more at its own edges than the node regenerated")


def zoomed_on_the_canvas(vm, mask, tokens, zoom: dict, size: tuple[int, int]):
    """A zoomed window's fitted mask and token region, on the whole frame: (mask [F, h, w] of 0 or 1, the share of
    each cell the region covers [F, h // CELL, w // CELL], no words), or (None, None, a refusal in words). A cell is
    in the region when its share is over 0 (`ZOOMED_CELLS_ARE`); the shares are kept so a run can say how many of
    its cells that rule added (`zoomed_cells_counted`).

    A window rendered zoomed in (`video_mask.zoomed_source`) holds both in the ZOOMED picture: one box of the load's
    frames scaled to the canvas. The node puts its render back with `video_mask.lay_back`: scaled down to the
    box's size on the canvas and placed at the box's own position. The same call is made here with a weight of 1
    where the mask (or the region) is on and an empty frame under it, and the result cut at a half: a carried
    mask and a region in canvas pixels, where the node laid them. The region is laid once a latent step, since a
    step's region is the same on every frame of its run (`video_mask.run_lengths`), and given to each of them.

    What it cannot say: where the node's `only what changed` kept less than the region (no reader can), and
    anything about a pixel the weight leaves under a half at the box's edge, which the node blends."""
    import torch
    w, h = size
    if [int(v) for v in zoom.get("canvas", [])] != [w, h]:
        return None, None, f"the zoom is for a canvas of {zoom.get('canvas')} and the capture is {w}x{h}"
    if tuple(mask.shape[1:]) != (h, w):
        return None, None, f"the zoomed mask is {int(mask.shape[2])}x{int(mask.shape[1])} and the canvas is {w}x{h}"

    def laid(weight):                                   # [N, h, w] of 0 or 1, in the zoomed picture, onto the canvas
        on = (weight > 0.5).to(torch.float32)
        back = vm.lay_back(torch.ones(*on.shape, 3), on, torch.zeros(int(on.shape[0]), h, w, 3), zoom)
        return back[..., 0] > 0.5

    steps = torch.nn.functional.interpolate(tokens[:, None].to(torch.float32), size=(h, w), mode="nearest")[:, 0]
    region = laid(steps)                                # one a latent step
    share = region.reshape(-1, h // CELL, CELL, w // CELL, CELL).to(torch.float32).mean((2, 4))
    runs = torch.tensor(vm.run_lengths(int(tokens.shape[0])))
    share = share.repeat_interleave(runs, dim=0)[:int(mask.shape[0])]
    on_canvas = torch.cat([laid(mask[i:i + 24]) for i in range(0, int(mask.shape[0]), 24)]).to(torch.float32)
    return on_canvas, share, None


def zoomed_cells_counted(shares: list) -> dict:
    """How many cells a zoomed run's region has by three rules, over the frames read: any pixel of the cell in the
    region (the rule used), more than half of it, every pixel. Figures: the box is on no grid of the canvas's, so a
    token of the zoomed picture can lie across two cells, and the first figure is the larger by those."""
    import torch
    if not shares:
        return {"any_pixel": 0, "more_than_half": 0, "every_pixel": 0, "counted_over": "the frames read"}
    every = torch.stack(shares)
    return {"any_pixel": int((every > 0).sum()), "more_than_half": int((every > 0.5).sum()), "every_pixel": int((every >= 1).sum()),
            "counted_over": "the frames read"}


def canvas_margin(margin, how: dict) -> tuple[int | None, int | None]:
    """A run's margin in canvas pixels, and, for a run rendered zoomed in, the node's own figure beside it:
    (margin_px, margin_px_in_the_zoomed_picture or None).

    The node grows its region by `grow_pixels` in the picture it renders, which for a zoomed run is the box enlarged
    by the zoom's `scale` (zoomed pixels a canvas pixel). Everything that reads a run's `margin_px` reads canvas
    pixels (this tool's reach of a region into another subject, the delivery assembler), so a zoomed run's is the
    node's figure divided by the scale, rounded, with the node's own kept beside it."""
    if margin is None:
        return None, None
    if not how.get("zoom"):
        return int(margin), None
    return int(round(int(margin) / float(how["zoom"]["scale"]))), int(margin)


def review_is_of_a_box(files: list[Path], graph: dict | None, vm) -> str | None:
    """Why a render's review picture must not be read back onto the canvas, or None: the review of a run rendered
    zoomed in is of the BOX (`video_mask.render_zoom_note`), and un-tinted against the whole frame's cells it puts
    a region where there is none. Said when a window's own region file says the run was zoomed; and, when no window
    saved its region, when the render's graph ASKS for a zoomed render: whether it was one cannot then be read (a
    zoomed choice comes to the whole frame for a subject too large), so it is refused and not guessed."""
    for f in files[:1]:
        zoom = vm.load_window_region(str(f)).get("zoom")
        if zoom:
            return (f"{f.name} says the run was rendered zoomed in (box {zoom.get('box')}): its review is of the box, not of the frame, "
                    "and is not read back onto the canvas. Read the regions its windows saved (leave `region=review` out)")
    if not files:
        asked = next((n.get("inputs", {}).get("render_region") for n in (graph or {}).values() if n.get("class_type") == "MiniMaxH3MaskedSource"), None)
        if asked and asked != vm.RENDER_WHOLE:
            return (f"the render's graph asks for `{asked}` and no window saved its region: whether it was rendered zoomed in cannot be "
                    "read, and if it was its review is of the box. Not read back onto the canvas")
    return None


def zoom_of_the_run(got: dict, zoom, name: str):
    """A window's own box held to the run's: a run is rendered in ONE box (`zoomed_source`), so a window that says
    another, or says none where the run is zoomed, is not this run's. Returns the refusal in words, or None."""
    if (got.get("zoom") or {}).get("box") != (zoom or {}).get("box"):
        return (f"{name} says it was rendered in the box {(got.get('zoom') or {}).get('box')} and the run's box is {(zoom or {}).get('box')}: "
                "a zoomed run has one box for the whole load")
    return None


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
    end, across, bare, names, composite, zoom, shares = 0, [], [], [], None, None, []
    last = None if render_frames is None else int(render_frames)
    for f in files:
        if last is not None and end >= last:
            break
        got = vm.load_window_region(str(f))
        if not names:
            zoom = got.get("zoom")                      # the run's box is its first window's; every window is held to it
        wrong = zoom_of_the_run(got, zoom, f.name)
        if wrong:
            return None, None, None, {"refused": wrong, "zoomed": True}
        start, trim, mask, tokens = got["first_frame"], got["trim"], got["mask"], got["tokens"]
        count = int(mask.shape[0])
        if start + trim != end:
            return None, None, None, {"refused": f"{f.name} starts its video on frame {start + trim} of the load and the "
                                                 f"window before it ended on {end}"}
        # the gate and "no mask on this frame" are the node's own, on the mask it was given (the zoomed one, for a zoomed run)
        gate = vm.cut_gate(mask, int(tokens.shape[0]), got["source"].get("cuts"), start) > 0.5
        given = mask
        if zoom:
            # a window rendered zoomed in holds its mask and tokens in the zoomed picture: laid on the canvas as the node lays them
            mask, share, why = zoomed_on_the_canvas(vm, mask, tokens, zoom, size)
            if why:
                return None, None, None, {"refused": f"{f.name}: {why}", "zoomed": True}
            cells = share > 0
        else:
            cells = vm.pixel_alpha(tokens, h // CELL, w // CELL, 0) > 0.5
            if tuple(mask.shape[1:]) != (h, w):
                mask = torch.nn.functional.interpolate(mask[:, None], size=(h, w), mode="nearest")[:, 0]
        # the composite leaves every frame with no mask of its own as the source. A render from before that rule
        # (2026-10-10) laid such frames with their step's region, and read back here it differs on them: an edge
        # accepted with the node, which no longer keeps the switch that said which kind a file was
        for k in range(trim, count if last is None else min(count, last - start)):
            n = at + start + k - first
            empty = not bool((given[k] > 0.5).any())
            unlaid = empty and bool(cells[k].any())
            if 0 <= n < frames:
                region[n], carried[n], read[n] = (cells[k] & gate[k]).numpy() & (not unlaid), (mask[k] > 0.5).numpy(), True
                if zoom and bool(gate[k]) and not unlaid:
                    shares.append(share[k])
            if unlaid:
                bare.append(at + start + k)
            elif not bool(gate[k]):
                across.append(at + start + k)
        end, composite = start + count, got["source"].get("composite")
        names.append(f.name)
    if last is not None and end < last:
        return None, None, None, {"refused": f"the windows' files cover {end} frames and the render has {render_frames}"}
    return region, carried, read, {"read_from": "the region each window saved beside its latent", "files": names, "legend_px": None,
                                   "composite": composite, "left_as_the_source_across_a_cut": across,
                                   "left_as_the_source_with_no_mask": bare,
                                   "held_frames_past_the_render": 0 if last is None else max(end - last, 0),
                                   "files_of_another_run_left_unread": [f.name for f in files if f.name not in names],
                                   **({"zoom": zoom, "zoomed_cells_are": ZOOMED_CELLS_ARE, "zoomed_cells_counted": zoomed_cells_counted(shares)} if zoom else {})}


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

    A run that continues a stored window (the plan's `continue_from`, `audio_freeze_song`'s input of that
    name) writes nothing on its first window's first `trim` frames: they are the stored window's, kept. Those
    frames are not a gap: they have no region here and are not read, and `kept_from_a_stored_window` names
    the file and the source frames. Added 2026-10-12, when the first continuation load met this reader and
    was refused as a gap.

    A load rendered zoomed in (the plan's `zoom`: one box of the load's frames, scaled to the canvas) holds its
    masks and tokens in the zoomed picture. They are laid on the canvas as the node lays the render
    (`zoomed_on_the_canvas`), and `how` carries the `zoom` and what the cells then are. Refused until 2026-10-13.

    Returns what `read_region` does. Refuses (None) when a window has no region file (the plan says why: it
    starts past the source's end), when a file the plan names is not there, when the windows' written
    frames leave a gap, when a window's box is not the plan's, or when the zoom is for another canvas."""
    vm = _pack("video_mask")
    plan = json.loads(plan_path.read_text())
    if plan.get("plan") != "h3 song plan":
        raise SystemExit(f"{plan_path.name} is not a song node plan (its `plan` is {plan.get('plan')!r})")
    zoom = plan.get("zoom")
    w, h = size
    region = np.zeros((frames, h // CELL, w // CELL), bool)
    carried, read = np.zeros((frames, h, w), bool), np.zeros(frames, bool)
    placed = at is not None
    at = first if at is None else int(at)
    end, across, bare, windows, stored, shares = 0, [], [], [], None, []
    for row in plan["windows"]:
        if not row.get("region_file"):
            return None, None, None, {"refused": f"window {row['number']} is not planned: {row.get('why', 'no region file')}"}
        path = plan_path.parent / row["region_file"]
        if not path.is_file():
            return None, None, None, {"refused": f"the plan names {row['region_file']} and it is not beside the plan"}
        head = int(row.get("trim", 0))
        if not windows and plan.get("continue_from") and head and row["first_written_frame"] == row["first_frame"] + head:
            # the first window of a run that continues a stored one: its kept frames are that window's, not written here
            stored = {"file": plan["continue_from"], "source_frames": [at + row["first_frame"], at + row["first_written_frame"] - 1]}
            end = row["first_written_frame"]
        if row["first_written_frame"] != end:
            return None, None, None, {"refused": f"window {row['number']} writes from frame {row['first_written_frame']} of the load and "
                                                 f"the window before it ended on {end}"}
        got = vm.load_window_region(str(path))
        wrong = zoom_of_the_run(got, zoom, row["region_file"])
        if wrong:
            return None, None, None, {"refused": wrong}
        mask, left = got["mask"], set(row.get("frames_left_as_source", []))
        if zoom:
            # a load rendered zoomed in: the plan's mask and tokens are of the zoomed picture, laid on the canvas as the node lays them
            mask, share, why = zoomed_on_the_canvas(vm, mask, got["tokens"], zoom, size)
            if why:
                return None, None, None, {"refused": f"{row['region_file']}: {why}"}
            cells = share > 0
        else:
            cells = vm.pixel_alpha(got["tokens"], h // CELL, w // CELL, 0) > 0.5
            if tuple(mask.shape[1:]) != (h, w):
                import torch
                mask = torch.nn.functional.interpolate(mask[:, None], size=(h, w), mode="nearest")[:, 0]
        for f in range(row["first_written_frame"], row["last_written_frame"] + 1):
            k, n = f - row["first_frame"], at + f - first
            if 0 <= n < frames:
                region[n], carried[n], read[n] = (cells[k].numpy() if f not in left else False), (mask[k] > 0.5).numpy(), True
                if zoom and f not in left:
                    shares.append(share[k])
            if f in left:
                # a plan of version 2 says which kind; an older one has only the cut gate's
                (bare if f in set(row.get("frames_with_no_mask", [])) else across).append(at + f)
        end = row["last_written_frame"] + 1
        windows.append({"window": row["number"], "writes_source_frames": [at + row["first_written_frame"], at + row["last_written_frame"]],
                        "text": row.get("text"), "regenerating_share": row.get("regenerating_share"), "kept_frames": int(row.get("trim", 0)),
                        "first_frame": at + row["first_frame"],
                        **held_tail_facts(got["mask"], got["tokens"], int(row.get("source_frames_held", 0)), at + row["first_frame"], vm.run_lengths)})
    if not placed and end != frames:
        # a plan does not know which source frame its load starts on. Placed at the span's first frame by default,
        # ten loads of one capture were each read as starting on the capture's first frame (2026-10-10), and every
        # flag about them named the wrong frames and the wrong people
        return None, None, None, {"refused": f"the plan writes {end} frame(s) and the capture holds {frames}: say which source frame the load "
                                             "starts on (`at=N` on its --run). It is not assumed to start on the capture's first frame"}
    return region, carried, read, {"read_from": "the region the node's preview planned for each window", "plan": plan_path.name,
                                   "files": [r["region_file"] for r in plan["windows"]], "legend_px": None,
                                   "composite": plan["source"].get("composite"), "left_as_the_source_across_a_cut": across,
                                   "left_as_the_source_with_no_mask": bare, "plan_version": plan.get("version"),
                                   "planned_windows": windows, "frames_written": end, "source_settings": plan["source"],
                                   **({"kept_from_a_stored_window": stored} if stored else {}),
                                   **({"zoom": zoom, "zoomed_cells_are": ZOOMED_CELLS_ARE, "zoomed_cells_counted": zoomed_cells_counted(shares)} if zoom else {})}


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
    sys.path.insert(0, str(REPO / "workflows"))
    import masked_job_graph                                         # the one reading of a two-subject load, the job builder's
    for n, load in enumerate(given.get("loads", [])):
        try:
            # a load of two subjects is this tool's FIRST subject's load: its region then lies over the second, and
            # the rule about a region over another subject says so, to be answered by a person (`outcome`)
            one = masked_job_graph.one_subject_view({**given.get("defaults", {}), **load})
        except ValueError as e:
            raise SystemExit(f"{path}: {e}") from e
        lacking = [k for k in ("first", "frames", "subject") if k not in one]
        if lacking:
            raise SystemExit(f"{path}: load {n} lacks {', '.join(lacking)}")
        name = str(one.pop("name", f"{one['subject']}_{int(one['first']):04d}"))
        if name in seen:
            raise SystemExit(f"{path}: two loads are named {name!r}; give one a `name`")
        seen.add(name)
        one["at"] = one.pop("first")
        out.append((name, {k: ("+".join(v) if isinstance(v, list) and all(isinstance(x, str) for x in v) else
                               json.dumps(v) if isinstance(v, (list, dict)) else str(v)) for k, v in one.items() if v is not None}))
    return out


def subjects_of(run: dict) -> list[str]:
    """The subjects a run replaces: one, or both of a load of two subjects (`subjects`; its first is `subject`).

    The rule every reader of a run holds to (2026-10-12): NO SUBJECT OF THE LOAD IS "ANOTHER SUBJECT" TO IT. A
    region over the second subject of a pair is the load doing what it was asked, not a pass drawing over
    somebody else; until then the rule about a region over another subject fired on every two-subject load and
    was answered by hand."""
    return list(run.get("subjects") or [run["subject"]])


def by_subject(runs: list[dict]) -> list[dict]:
    """Each run once for each subject it replaces, with `subject` that one: what a rule that is a FACT ABOUT A SUBJECT
    walks (a shot of the load without them, their size, their face, their mouth, their pose ahead of kept frames).
    A run of one subject comes back as it is. A rule that reads the mask the load carries does not walk this: that
    mask is both subjects' parts as one, and whose a missing part is cannot be read off it (`docs/wiki/next_steps.md`)."""
    return [run if len(subjects_of(run)) == 1 else {**run, "subject": who} for run in runs for who in subjects_of(run)]


def second_carried(also: dict, track: np.ndarray, parts: np.ndarray | None, held: np.ndarray | None) -> np.ndarray:
    """The second subject's share of the mask a two-subject load carries, from the capture's own masks.

    `also` is the load's second subject as the builder's `one_subject_view` gives it: its `carried` is `held` for
    a face or a face and hair (the held part where the capture has one, else the part) and `track` for the whole
    person. The node is given both subjects' parts as ONE mask, so a plan worked out here is worked out from the
    two together. A part file the load names itself (`parts_mask`) is the builder's to wire and is not read here:
    this is the capture's own part of that subject."""
    if also.get("carried") == "track":
        return track
    part = held if held is not None else parts
    if part is None:
        raise SystemExit(f"a load's second subject {also.get('subject')!r} is replaced by a part ({also.get('region')}) and the capture has "
                         f"no part mask of that subject: give its --mask a parts= video")
    return part


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
    """How the folder came to be. Written first as `running`, so a folder that stops there says so.
    `made_by` (from `files --made-by`) is who built it: a look at the capture is somebody else's."""
    (out / "status.json").write_text(json.dumps({"state": state, "written": datetime.datetime.now().isoformat(timespec="seconds"),
                                                 "by": "bench/capture_masked_run.py files", **more}, indent=1) + "\n")


def files(a: argparse.Namespace) -> Path:
    if not (a.folder or a.name):
        raise SystemExit("give --name (the capture goes to <out>/<date>_<name>) or --folder (exactly where it goes)")
    out = Path(a.folder) if a.folder else Path(a.out) / f"{a.date}_{a.name}"
    a.name = a.name or out.name
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
    who = {"made_by": a.made_by} if getattr(a, "made_by", "") else {}
    write_status(out, "running", inputs=[Path(x).name for x in named], **who)
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
    write_status(out, "done", **who, inputs=[Path(x).name for x in named], outputs=sorted(
        str(q.relative_to(out)) for q in out.rglob("*") if q.is_file() and q.name != "status.json"))
    print("wrote", out)
    return out


def _files(a: argparse.Namespace, out: Path, masks: list, runs: list, plans: list) -> None:
    first, frames = int(a.first), int(a.frames)
    width, height, _ = probe(masks[0][1]["track"])
    size = (width, height)
    sightings: dict = {}
    pieces_by_sighting: dict = {}
    class_maps: dict = {}
    made_of: dict = {}
    held_masks: dict = {}
    manifest = {"name": a.name, "date": a.date, "clip": Path(a.source).name, "first_frame": first, "frames": frames,
                "size": [width, height], "fps": FPS, "cell_px": CELL, "subjects": [], "runs": [],
                "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True).stdout.strip()}
    for label, m in masks:
        by, at = m.get("by", "preview"), int(m.get("at", first))
        least, least_by = (int(m["least"]), LEAST_GIVEN) if m.get("least") is not None else (PART_LEAST_PX, LEAST_STARTING)
        if least < 0:
            raise SystemExit(f"subject {label!r}: least={m['least']}; a part's least size is 0 (nothing emptied by size) or more, in px")
        track, covered = read_mask(m["track"], size, at, first, frames)
        parts = read_mask(m["parts"], size, at, first, frames)[0] if m.get("parts") else None
        took = None
        if parts is not None and m.get("pieces") == "largest":
            # the caller looked and the part's other pieces are not the part (`flag_pieces`); what that took away is
            # counted, printed and kept in the manifest, since the flag is then silent
            parts, stray = largest_piece(parts)
            took = pieces_taken(stray, first)
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
        pieces_by_sighting[(label, by)] = took
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
        sizes = None
        if parts is not None:
            sizes = part_sizes(parts, first, least, least_by, not_applied=(
                "grade=off on this --mask" if m.get("grade") == "off" else "no class map was given for this subject, so its part is not graded"
                if not m.get("classes") else "the mask video does not cover the whole span, so no part to load was written"))
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
                        held, graded = grade_holds(parts, held, class_maps[label], made, least)
                        sizes = part_sizes(parts, first, least, least_by, emptied=[f for f, d in graded.items() if "least_px" in d])
                        held_frames = [f for f, d in graded.items() if d["did"] == "filled"]
                        refused = [first + f for f, d in graded.items() if d["did"].startswith("saved")]
                        emptied_by_grade = [first + f for f, d in graded.items() if d["did"] == "emptied"]
                        (folder / "part_grades.json").write_text(json.dumps(
                            {"subject": label, "seen_by": by, "bar": HOLD_INSIDE, "part_is_made_of": made_of.get(label),
                             "least_px": least, "least_px_set_by": least_by, "part_sizes": sizes,
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
                                   "frames_covered": int(covered.sum()), "pose": posed, "parts_pieces": m.get("pieces"),
                                   # what `pieces=largest` took away, a frame at a time (`pieces_taken`); None when it is not set
                                   "parts_pieces_taken": pieces_by_sighting.get((label, by)),
                                   "inputs": {k: {**input_facts(m[k]), **({"made_from": input_facts(m["parts_from"])} if k == "parts" and m.get("parts_from") else {})}
                                              for k in ("track", "parts", "classes", "held", "shots", "pose") if m.get(k)},
                                   "classes": Path(m["classes"]).name if m.get("classes") else None,
                                   "part_is_made_of": made_of.get(label),
                                   "parts_held_on_source_frames": frame_spans([first + f for f in held_frames]),
                                   "parts_doubted_and_left_on_source_frames": frame_spans([first + f for f in left_frames]),
                                   "parts_emptied_on_source_frames": dropped,
                                   "parts_fill_refused_by_grade_on_source_frames": frame_spans(refused),
                                   "parts_emptied_by_grade_on_source_frames": frame_spans(emptied_by_grade),
                                   # the part's size on this clip's frames, and where the floor fell among them (`part_sizes`)
                                   "part_sizes": sizes})
        print(f"subject {label} seen by {by}: {int(covered.sum())} of {frames} frames covered, "
              f"mask on {int(track.reshape(frames, -1).any(1).sum())}", flush=True)
        if sizes is not None:
            print("\n".join(size_lines(f"  {label}'s part, source frames {first}-{first + frames - 1}, canvas {width}x{height}", sizes)), flush=True)
        if pieces_by_sighting.get((label, by)) is not None:
            print("\n".join(pieces_lines(label, pieces_by_sighting[(label, by)])), flush=True)
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
        # a load of two subjects (`loads_from_file`): the first is `subject`, the second comes beside it in `also`
        also = json.loads(r["also"]) if r.get("also") else None
        both = [label] + ([also["subject"]] if also else [])
        for who in both + others + kept_labels:
            if who not in sightings:
                raise SystemExit(f"run {name!r} names subject {who!r}, which no --mask gives")
        lead = {who: next(iter(sightings[who].values())) for who in both + others + kept_labels}
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
            if also:
                carried = carried | second_carried(also, lead[also["subject"]][0], lead[also["subject"]][2], held_masks.get(also["subject"]))
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
            every_saved = window_region_files(r["render"])
            saved = [] if r.get("region") == "review" else every_saved
            region = None
            if saved:
                region, carried, read, how = read_saved_regions(saved, first, frames, size, r.get("at"), probe(r["render"])[2])
                if region is None and how.get("zoomed"):
                    # the review of a zoomed run is not on the canvas's cell grid either: there is nothing right to fall back to
                    raise SystemExit(f"run {name!r}: {how['refused']}; its review cannot be read back either")
                if region is None:
                    print(f"run {name}: the windows' saved regions are not this render's ({how['refused']}); reading the review")
                elif Path(r["render"][:-len(".mp4")] + "_with_mask.mp4").is_file():
                    # both readers on the same frames: what the node saved against what the review picture gives back
                    back = read_region(r["render"], a.source, first, frames, size, r.get("at"))
                    how["readers"] = readers_agreement((region, carried, read), back[:3])
            if region is None:
                # the two ways a zoomed run reached the review reader with nothing in between (2026-10-13)
                boxed = review_is_of_a_box(every_saved, graph, _pack("video_mask"))
                if boxed:
                    raise SystemExit(f"run {name!r}: {boxed}")
                refused = (how or {}).get("refused") if saved else None
                region, carried, read, how = read_region(r["render"], a.source, first, frames, size, r.get("at"))
                if refused:
                    how["saved_regions_refused"] = refused
        folder = out / "runs" / name
        folder.mkdir(parents=True, exist_ok=True)
        entry = {"subject": label, **({"subjects": both} if also else {}), "others": others, "region": region, "carried": carried,
                 "read": read, "reached": {}, "back": {}}
        margin, zoomed_margin = canvas_margin(r.get("margin", settings.get("grow_pixels")), how)
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
                                 "subject": label, **({"subjects": both} if also else {}), "others": others, "keep": kept_labels,
                                 "windows": ({"window_frames": int(r["window"]), "context_frames": int(r.get("context", 0))}
                                             if r["planned"] and r.get("window") else window_settings(graph)),
                                 "carried_is": ("the held part" if r.get("carried") == "held" else "the part" if lead[label][2] is not None
                                                and r.get("carried", "parts") == "parts" else "the track") if r["planned"] and not r.get("preview") else "the mask each window was given" if how.get("files") else "read from the review",
                                 # a plain field for a reader that must know it (the delivery assembler lays nothing on a frame
                                 # the node carried no mask on): the carried mask is the node's own, not read off a review
                                 "carried_from_the_node": bool(how.get("files")),
                                 "first_source_frame": int(r.get("at", first)), "margin_px": None if margin is None else int(margin),
                                 # a run rendered zoomed in: the node's own grow, in pixels of the zoomed picture (`canvas_margin`)
                                 **({"margin_px_in_the_zoomed_picture": zoomed_margin} if zoomed_margin is not None else {}),
                                 "masked_source": settings,
                                 "given_back_worked_out": bool(entry["back"]), "frames_read": int(read.sum()), **how})
        print(f"{'plan' if r['planned'] else 'run'} {name}: region {'worked out' if r['planned'] and not r.get('preview') else 'read'} on "
              f"{int(read.sum())} of {frames} frames, mean share {float(region[read].mean()) if read.any() else 0.0:.4f}", flush=True)
        if how.get("zoom"):
            print(f"  RENDERED ZOOMED IN {float(how['zoom']['scale']):.2f}x, in the box {how['zoom']['box']} of its {how['zoom']['source_size'][0]}x"
                  f"{how['zoom']['source_size'][1]} frames: {ZOOMED_CELLS_ARE}. Cells in the region over the frames read: "
                  f"{how['zoomed_cells_counted']['any_pixel']} by that rule, {how['zoomed_cells_counted']['more_than_half']} more than half "
                  f"covered, {how['zoomed_cells_counted']['every_pixel']} wholly covered. Its margin is {zoomed_margin} px of the zoomed picture, "
                  f"{margin} px on the canvas. NOT changed for the zoom: the part's sizes, the floor and the face-size flag, which are "
                  "of the whole frame, where the part is smaller than it is drawn by the scale squared", flush=True)
    voice = read_voice(a.voice) if a.voice else None
    manifest["voice_table"] = Path(a.voice).name if a.voice else None
    cross = cross_rows(first, frames, sightings, captured, voice)
    for row, (step, sharp) in zip(cross, picture_rows(a.source, first, frames, size)):
        row.update(source_step=step, source_sharpness=sharp)
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


def verify_checks(manifest: dict, track_flags: list[dict], statements: list[dict] | tuple = ()) -> list[dict]:
    """Every check `verify` makes, each `{"check", "about", "ok", "why"}`; `ok` is None when it could not be made.

    `track_flags` are `flag_track`'s for every sighting: a track empty inside a shot it was taken in. One of a
    subject who is only an other of the loads is data, passed with its figures and what it cannot see.
    `statements` are a person's, kept in the capture (`no_mask_statements.json`): a subject has no mask from
    one source frame to another and that is known (they walk out of the shot; no load is made for them there).
    An empty frame a statement covers is accounted for, with who said it and why; one that no statement covers
    still fails. A subject who leaves a shot is not a capture that misread its files."""
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
            # a pose table may come from a pass of its own; the masks and the shot table are one preview's. A load's own part
            # (`parts_from=` on its mask: a cut of some of the subject's parts, or the part held to its track, made by the job step
            # from the preview's own files) is written when the job is, and is held to the preview file it was made from
            # (2026-10-14: the first load to carry a held part failed here on a preview of the day before)
            given = {k: v for k, v in given.items() if k != "pose"}
            stamp = lambda v: (v.get("made_from") or v)["written"]                                     # noqa: E731
            times = sorted(datetime.datetime.fromisoformat(stamp(v)) for v in given.values())
            spread = (times[-1] - times[0]).total_seconds()
            add("inputs_written_together", about, spread <= WRITTEN_TOGETHER_S,
                f"written within {int(spread)} s of each other" + "".join(f"; {v['file']} is the load's own, made from {v['made_from']['file']}"
                                                                          for v in given.values() if v.get("made_from")) if spread <= WRITTEN_TOGETHER_S else
                f"written {int(spread)} s apart: {', '.join(v['file'] + ' at ' + v['written'][11:] + (' (made from ' + v['made_from']['file'] + ' at ' + v['made_from']['written'][11:] + ')' if v.get('made_from') else '') for v in given.values())}. "
                "They are not one run's", seconds=int(spread))
            wrong = [f"{v['file']} is {v['size'][0]}x{v['size'][1]}" for v in videos.values() if v["size"] != size]
            add("inputs_at_the_canvas", about, not wrong, "; ".join(wrong) + f"; the capture is {size[0]}x{size[1]} and a mask of "
                "another size is resized without a word" if wrong else f"every mask video is {size[0]}x{size[1]}")
            at = seen["first_source_frame"]
            short = [f"{v['file']} ends on source frame {at + v['frames'] - 1}" for v in videos.values() if at + v["frames"] < first + frames]
            add("inputs_reach_the_span", about, not short and at <= first, ("; ".join(short) or f"the masks start on source frame {at}")
                + f"; the span is {first}-{first + frames - 1}" if short or at > first else f"every mask video covers {first}-{first + frames - 1}")
    for f in track_flags:
        if f["rule"] == "other_has_no_track_in_a_taken_shot":      # data: an other of the loads, replaced by none of them
            add("track_where_the_shot_table_says_taken", f"{f['subject']} by {f['seen_by']}", True, f["why"], source_frames=f["source_frames"],
                an_other_of=f["figures"]["an_other_of"], figures=f["figures"])
            continue
        each = f.get("figures", {}).get("each") or [n for a, b in f["source_frames"] for n in range(a, b + 1)]
        said = [s for s in statements if s["subject"] == f["subject"]]
        used = [s for s in said if any(s["source_frames"][0] <= n <= s["source_frames"][1] for n in each)]
        left = [n for n in each if not any(s["source_frames"][0] <= n <= s["source_frames"][1] for s in used)]
        told = "; ".join(f"{s['source_frames'][0]}-{s['source_frames'][1]} \"{s['why']}\" ({s['by']})" for s in used)
        if not used:
            add("track_where_the_shot_table_says_taken", f"{f['subject']} by {f['seen_by']}", False, f["why"], source_frames=f["source_frames"])
        elif left:
            add("track_where_the_shot_table_says_taken", f"{f['subject']} by {f['seen_by']}", False,
                f"{f['subject']} has no mask on {len(each)} frame(s) of shots they were taken in; {len(each) - len(left)} are accounted for "
                f"({told}) and {len(left)} are not: nothing regenerates there", source_frames=frame_spans(left, join=1), accounted_for=used)
        else:
            add("track_where_the_shot_table_says_taken", f"{f['subject']} by {f['seen_by']}", True,
                f"{f['subject']} has no mask on {len(each)} frame(s) of shots they were taken in, every one accounted for by a person: {told}",
                source_frames=f["source_frames"], accounted_for=used)
    if not track_flags:
        tables = [s["label"] for s in manifest["subjects"] if any(x.get("shots") for x in s["sightings"])]
        add("track_where_the_shot_table_says_taken", ", ".join(tables) or "no subject", True if tables else None,
            "no track is empty inside a shot it was taken in" if tables else "no sighting has a shot table: nothing says where a track should be")
    # a statement excuses frames no load covers. A load of that subject over frames it excused is the two said at once
    for run in by_subject(manifest["runs"]):
        at, count = (run["load"]["first_frame"], run["load"]["frames"]) if run.get("load") else (run.get("first_source_frame"), run.get("frames_read"))
        if at is None or not count:
            continue
        for f in track_flags:
            if f["subject"] != run["subject"]:
                continue
            each = f.get("figures", {}).get("each") or [n for a, b in f["source_frames"] for n in range(a, b + 1)]
            for st in statements:
                hit = [n for n in each if st["subject"] == run["subject"] and st["source_frames"][0] <= n <= st["source_frames"][1] and at <= n < at + count]
                if hit:
                    add("no_mask_statement_against_a_load", run["name"], False,
                        f"{st['by']} said {st['subject']} has no mask on {st['source_frames'][0]}-{st['source_frames'][1]} and that this is known "
                        f"(\"{st['why']}\"); load {run['name']} regenerates {st['subject']} on {at}-{at + count - 1} and {len(hit)} of those frames "
                        "have no mask: end the load before them or take the statement back", source_frames=frame_spans(hit, join=1))
    classed = {s["label"] for s in manifest["subjects"] if any(x.get("classes") for x in s["sightings"])}
    for run in by_subject(manifest["runs"]):
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


def preview_first_of(by: str, data: Path) -> tuple[int | None, str]:
    """The source frame a preview's files begin on, from the preview's own job: (first, where it was read), or
    (None, why not). A sighting's `by` is the preview's job name, and its job folder is `data/<day>_<by>_job`;
    one such folder, or it is not said which."""
    found = sorted(Path(data).glob(f"*_{by}_job/job.json"))
    if len(found) != 1:
        return None, (f"no job folder data/<day>_{by}_job holds a job.json" if not found else f"{len(found)} job folders are named for {by}")
    try:
        return int(json.loads(found[0].read_text())["job"]["first"]), str(found[0].parent.name)
    except (ValueError, KeyError, TypeError):
        return None, f"{found[0].parent.name}/job.json does not say the job's first frame"


def at_checks(manifest: dict, firsts: dict[str, tuple[int | None, str]]) -> tuple[list[dict], list[dict]]:
    """Is every sighting's `at` the source frame its preview's files begin on: (checks, not checked).

    `at` (`first_source_frame`, and `shots_first_source_frame` for the shot table) says which source frame a
    mask FILE's frame 0 is on. The preview's own job says where its files begin. Where the two differ the
    capture read the masks on the wrong frames, and the manifest and the preview's job contradict each other: a
    hard failure. On 2026-10-13 two captures were made so by a script that passed the capture's first frame
    as `at`; both verified, gated clear and were looked at before the masks were seen to sit four and eight
    frames off the picture. `firsts` maps a sighting's `by` to `preview_first_of`'s answer. A sighting whose
    preview's job cannot be found (masks made by hand, or by a run with no job folder) is NOT CHECKED and said
    so by name: it is no failure and no pass."""
    checks, unchecked = [], []
    for s in manifest["subjects"]:
        for seen in s["sightings"]:
            about = f"{s['label']} by {seen['by']}"
            first, where = firsts.get(seen["by"], (None, "its preview's job was not looked for"))
            if first is None:
                unchecked.append({"check": "at_is_the_previews_first_frame", "about": about, "why": where})
                continue
            said = {"at": seen.get("first_source_frame"), "shots_at": seen.get("shots_first_source_frame", seen.get("first_source_frame"))}
            wrong = {k: v for k, v in said.items() if v != first}
            checks.append({"check": "at_is_the_previews_first_frame", "about": about, "ok": not wrong,
                           "why": (f"the masks were read with {', '.join(f'{k}={v}' for k, v in wrong.items())}, and the preview's own job ({where}) says its files "
                                   f"begin on source frame {first}: every mask of this sighting is {abs(next(iter(wrong.values())) - first)} frame(s) off the picture"
                                   if wrong else f"at={first}, as the preview's own job ({where}) says its files begin"), "previews_first": first})
    return checks, unchecked


def verify(a: argparse.Namespace) -> None:
    folder = Path(a.capture)
    m = json.loads((folder / "manifest.json").read_text())
    track_flags, regions = [], run_regions(folder, m["runs"])
    for s in m["subjects"]:
        rows = json.loads((folder / "subjects" / s["label"] / "per_frame.json").read_text())["rows"]
        for seen in s["sightings"]:
            path = folder / "subjects" / s["label"] / f"shots__{seen['by']}.json"
            if path.is_file():
                track_flags += flag_track(s["label"], seen["by"], [r for r in rows if r["seen_by"] == seen["by"]],
                                          json.loads(path.read_text()), seen.get("shots_first_source_frame", seen["first_source_frame"]),
                                          m["runs"], regions, m["size"], m.get("cell_px", CELL))
    kept = folder / "no_mask_statements.json"
    statements = json.loads(kept.read_text())["statements"] if kept.is_file() else []
    if a.no_mask:
        if not a.by:
            raise SystemExit("--no-mask is a person's statement: say who makes it with --by")
        labels = {s["label"] for s in m["subjects"]}
        for item in a.no_mask:
            head, _, why = item.partition("=")
            label, _, rng = head.partition(":")
            if label not in labels or not why.strip() or "-" not in rng:
                raise SystemExit(f"--no-mask {item!r}: LABEL:FIRST-LAST=WHY, with a subject of this capture ({', '.join(sorted(labels))}) and a reason")
            span = [int(x) for x in rng.split("-")]
            statements = [s for s in statements if (s["subject"], s["source_frames"]) != (label, span)]
            statements.append({"subject": label, "source_frames": span, "why": why.strip(), "by": a.by,
                               "said": datetime.datetime.now().isoformat(timespec="seconds")})
        # kept in the capture: the builder runs `verify` again with no arguments before it queues a render
        kept.write_text(json.dumps({"what": "a person's statements that a subject has no mask on a stretch and that this is known; "
                                            "read by every later `verify` of this capture", "statements": statements}, indent=1) + "\n")
    checks = verify_checks(m, track_flags, statements)
    about_at, unchecked = at_checks(m, {seen["by"]: preview_first_of(seen["by"], REPO / "data") for s in m["subjects"] for seen in s["sightings"]})
    checks += about_at
    failed, unmade = [c for c in checks if c["ok"] is False], [c for c in checks if c["ok"] is None]
    verdict = "does not read what the nodes wrote" if failed else "not verified: a check could not be made" if unmade else "reads what the nodes wrote"
    (folder / "verify.json").write_text(json.dumps({"capture": m["name"], "verdict": verdict, "failed": len(failed), "not_made": len(unmade),
                                                    "written": datetime.datetime.now().isoformat(timespec="seconds"),
                                                    "no_mask_statements": statements, "checks": checks, "not_checked": unchecked}, indent=1) + "\n")
    for c in checks:
        print(f"{'ok  ' if c['ok'] else 'FAIL' if c['ok'] is False else '??  '}  {c['check']}  [{c['about']}]\n      {c['why']}")
    for c in unchecked:
        print(f"not checked  {c['check']}  [{c['about']}]\n      {c['why']}")
    print(f"verify: {verdict}; wrote {folder / 'verify.json'}")
    if failed or unmade:
        sys.exit(VERIFY_FAILED)


RULES_INPUT_VERSION = 1


def shot_edges(first: int, frames: int, cuts: list[int]) -> list[list[int]]:
    """A span's shots as [first, last] source frames, from the source frames a new shot starts on."""
    edges = [first] + sorted(c for c in set(cuts) if first < c < first + frames) + [first + frames]
    return [[a, b - 1] for a, b in zip(edges, edges[1:])]


def head_across(pose: list[dict]) -> dict | None:
    """The head over a shot, from `pose_state`'s columns on its frames in order: where it faces on the first and
    last frame that have a pose, the largest turn from the first and the frame it is reached on, the chin's
    range and the nearest a wrist comes to the nose. None when no frame has a facing."""
    rows = [r for r in pose if r.get("head_yaw") is not None]
    if not rows:
        return None
    turn = [abs((r["head_yaw"] - rows[0]["head_yaw"] + 180.0) % 360.0 - 180.0) for r in rows]
    lift = [r["head_lift"] for r in rows]
    return {"frames_with_a_pose": len(rows), "head_yaw_first": rows[0]["head_yaw"], "head_yaw_last": rows[-1]["head_yaw"],
            "largest_turn_from_the_first_degrees": round(max(turn), 1), "reached_on_source_frame": rows[int(np.argmax(turn))]["source_frame"],
            "chin_degrees": [round(min(lift), 1), round(max(lift), 1)],
            "a_wrist_nearest_the_nose": round(min(min(r["left_wrist_to_nose"], r["right_wrist_to_nose"]) for r in rows), 2)}


def part_owners(part: np.ndarray, owner: np.ndarray, labels: list[str], label: str, first: int = 0) -> dict | None:
    """Whose pixels a subject's part sits on, by the capture's owner map, over the frames given.

    `shares` is of the part's own pixels: each subject the owner map gives them to, `contested` (several tracks
    claim them and the class maps did not settle it) and `nobody` (outside every track). A part pass redraws
    the part, so a part on another subject's or on contested pixels is where it can draw over someone else:
    the frames it happens on are counted and named (from `first`), with the most pixels on any one of them."""
    total = int(part.sum())
    if not total:
        return None
    under = owner[part]
    shares = {name: round(float((under == k).sum()) / total, 4) for k, name in enumerate(labels)}
    shares["contested"], shares["nobody"] = round(float((under == CONTESTED).sum()) / total, 4), round(float((under == NOBODY).sum()) / total, 4)
    mine = labels.index(label)
    off = (part & (owner != mine) & (owner != NOBODY)).reshape(len(part), -1).sum(1)
    return {"pixels": total, "shares": shares, "frames_on_another_or_contested": int((off > 0).sum()),
            "source_frames_on_another_or_contested": frame_spans([first + int(i) for i in np.nonzero(off)[0]], join=1),
            "most_px_on_one_frame": int(off.max())}


def rules_input(folder: Path, not_in: dict | None = None) -> dict:
    """Every figure of a capture that a per-clip rule needs, in one file: per shot and subject, and per load.

    It holds figures and no line: which value a rule judges by is set from the clip by whoever reads this
    (the job builder's stage), and the lines this tool's own gate uses are listed in the module docstring. A
    figure that cannot be made is null with a `why`, so a reader can say "not in the capture" and not guess.
    `sources` names the table each figure came from. Nothing here needs a render."""
    m = json.loads((folder / "manifest.json").read_text())
    first, frames, (w, h) = m["first_frame"], m["frames"], m["size"]
    cross = json.loads((folder / "frames.json").read_text())["rows"]
    voiced = np.array([np.nan if r.get("voiced") in (None, "") else float(r["voiced"]) for r in cross])
    names = _pack("sapiens2_parts").CLASS_NAMES
    tables, cuts = {}, []
    for s in m["subjects"]:
        seen = s["sightings"][0]
        path = folder / "subjects" / s["label"] / f"shots__{seen['by']}.json"
        if path.is_file():
            tables[s["label"]] = (json.loads(path.read_text()), seen.get("shots_first_source_frame", seen["first_source_frame"]))
            cuts = cuts or [tables[s["label"]][1] + c for c in tables[s["label"]][0]["cuts"]]
    owners = np.load(folder / "owners.npz") if (folder / "owners.npz").is_file() else None
    per, poses, mouths, part_bits = {}, {}, {}, {}
    for s in m["subjects"]:
        label, seen = s["label"], s["sightings"][0]
        base = folder / "subjects" / label
        stored = np.load(base / f"masks__{seen['by']}.npz")
        part_bits[label] = stored["parts"] if "parts" in stored.files else None
        per[label] = [r for r in json.loads((base / "per_frame.json").read_text())["rows"] if r["seen_by"] == seen["by"]]
        pose = base / f"pose__{seen['by']}.json"
        poses[label] = json.loads(pose.read_text())["rows"] if pose.is_file() else None
        classes = base / f"classes__{seen['by']}.npz"
        if not classes.is_file():
            mouths[label] = "no class map for this subject (classes= on --mask)"
        elif not (voiced == 0).any():
            mouths[label] = "every frame is voiced" if (voiced == 1).any() and not np.isnan(voiced).any() else "no voice table over these frames (--voice)"
        else:
            saved = np.load(base / f"masks__{seen['by']}.npz")
            face = np.unpackbits(saved["parts" if "parts" in saved.files else "track"], axis=-1)[..., :w].astype(bool)
            opening, _ = mouth_openings(class_mask_of(np.load(classes)["classes"], list(MOUTH_CLASSES), names), face)
            mouths[label] = open_runs(opening, voiced, {}, first)["runs"]

    def q(values: list[float], scale: float = 1.0, digits: int = 1) -> dict | None:
        return {"median": round(float(np.median(values)) / scale, digits), "least": round(min(values) / scale, digits),
                "most": round(max(values) / scale, digits)} if values else None

    who = json.loads((folder / "who_is_who.json").read_text()) if (folder / "who_is_who.json").is_file() else None
    shots = []
    for number, (a, b) in enumerate(shot_edges(first, frames, cuts), 1):
        rows = [r for r in cross if a <= r["source_frame"] <= b]
        step = [r["source_step"] for r in rows[1:] if r.get("source_step") is not None]
        sharp = [r["source_sharpness"] for r in rows if r.get("source_sharpness") is not None]
        one = {"shot": number, "source_frames": [a, b],
               "picture": {"change_from_the_frame_before_median": round(float(np.median(step)), 2) if step else None,
                           "sharpness_median": round(float(np.median(sharp)), 1) if sharp else None,
                           "why": None if sharp else "this capture was made before the picture's own figures were read: run `files` again"},
               "subjects": {}}
        for s in m["subjects"]:
            label = s["label"]
            mine = [r for r in per[label] if a <= r["source_frame"] <= b]
            masked = [r for r in mine if r.get("track_share")]
            entry = {"frames_with_a_mask": len(masked), "frames": len(mine), "share_of_the_frame": q([r["track_share"] for r in masked], digits=5)}
            said, state, sim, line, typed = "nobody: this subject has no shot table", None, None, None, False
            if label in tables:
                table, at = tables[label]
                hit = next((x for x in table["shots"] if at + x["first_frame"] <= a <= at + x["last_frame"]), None)
                if hit is not None:
                    (state, typed), sim, line = shot_state(hit), hit["subject"].get("similarity"), float(table["match"])
                    said = "a correction typed on the tracker" if typed else "the tracker"
            if any(x[0] <= a and b <= x[1] for x in (not_in or {}).get(label, [])):
                said = "the caller's --not-in" if state in (None, "absent") else said + "; the caller's --not-in says they are not in it"
            entry["on_screen"] = {"state": state, "said_by": said, "typed": typed, "likeness": sim, "line": line}
            tall = [r["parts_height_px"] for r in mine if r.get("parts_height_px")]
            entry["part_height"] = None if not tall else {"px": q(tall), "cells": q(tall, CELL), "tokens": q(tall, TOKEN_PX, 2)}
            if not tall:
                entry["part_height_why"] = (
                    "no part for this subject" if not s["sightings"][0].get("parts") else
                    "the subject has no mask on this shot" if not masked else
                    "the part is empty on every frame of this shot" if any("parts_height_px" in r for r in mine) else
                    "this capture was made before the part's box was read: run `files` again")
            if owners is None or part_bits[label] is None:
                entry["part_pixels_owned_by"] = None
                entry["part_pixels_owned_by_why"] = "no part for this subject" if part_bits[label] is None else \
                    "no owner map: the capture holds one subject, or no class maps"
            else:
                part = np.unpackbits(part_bits[label][a - first:b - first + 1], axis=-1)[..., :w].astype(bool)
                entry["part_pixels_owned_by"] = part_owners(part, owners["owner"][a - first:b - first + 1], [str(x) for x in owners["labels"]], label, a)
                if entry["part_pixels_owned_by"] is None:
                    entry["part_pixels_owned_by_why"] = "the part is empty on every frame of this shot"
            entry["head"] = head_across([r for r in (poses[label] or []) if a <= r["source_frame"] <= b])
            if entry["head"] is None:
                entry["head_why"] = "no pose table for this subject (pose= on --mask)" if poses[label] is None else \
                    "the pose table has no 3D keypoints on this shot's frames"
            entry["mouth_open_on_unvoiced_frames"] = mouths[label] if isinstance(mouths[label], str) else \
                [r for r in mouths[label] if a <= r["source_frames"][0] and r["source_frames"][1] <= b]
            one["subjects"][label] = entry
        if who is None:
            one["who_is_who"], one["who_is_who_why"] = None, "no corrections file was given to preflight (--who-is-who)"
        else:
            at = who.get("first_source_frame") or 0
            row = next((r for r in who.get("rows", []) if r["frames"][0] + at <= a <= r["frames"][1] + at), None)
            one["who_is_who"] = None if row is None else {
                "status": row["status"], "why": row.get("why"), "note": row.get("note"),
                "trackers": {label: {k: t.get(k) for k in ("called", "called_person", "answered_by", "ask", "numbers_on_its_tile",
                                                           "typed_in_this_table", "tile_frame")}
                             for label, t in row.get("trackers", {}).items()},
                "frames_are_source_frames": who.get("first_source_frame") is not None}
            if row is None:
                one["who_is_who_why"] = "the corrections file has no row over this shot's first frame"
        shots.append(one)

    loads = []
    for run in m["runs"]:
        base = folder / "runs" / run["name"]
        inside, path = {}, base / "segments_in_region.json"
        if path.is_file():
            rows = json.loads(path.read_text())["rows"]
            for key in sorted({k[:-len("__cells")] for r in rows for k in r if k.endswith("__cells")}):
                hit = [(r["source_frame"], r[key + "__cells"]) for r in rows if r.get(key + "__cells")]
                if hit:
                    inside[key] = {"frames": len(hit), "source_frames": frame_spans([f for f, _ in hit]), "most_cells": int(max(c for _, c in hit)),
                                   "whose": "the subject's own, not the part" if key.split(".", 1)[0] in subjects_of(run) else "another subject's"}
        others = {}
        for other in [s["label"] for s in m["subjects"] if s["label"] not in subjects_of(run)]:
            key = f"region_on__{run['name']}__{other}"
            hit = [(r["source_frame"], r[key]) for r in cross if r.get(key)]
            if hit:
                others[other] = {"frames": len(hit), "source_frames": frame_spans([f for f, _ in hit]), "most_share_of_its_mask": round(max(v for _, v in hit), 4)}
        saved = np.load(base / "region.npz")
        bare = [first + f for f in lent_frames(np.unpackbits(saved["carried"], axis=-1)[..., :w].astype(bool), saved["region"], saved["read"])]
        windows = []
        pose = {r["source_frame"]: r for r in (poses.get(run["subject"]) or [])}
        for win in run.get("planned_windows") or []:
            kept = None
            if win.get("kept_frames"):
                kept = turn_from_kept(pose, win["first_frame"] + win["kept_frames"] - 1, tuple(win["writes_source_frames"])) or \
                    {"why": "no pose table with 3D keypoints over the kept and the new frames"}
            windows.append({**{k: win.get(k) for k in ("window", "writes_source_frames", "kept_frames", "frames", "held_frames", "held_share",
                                                        "held_frames_shown_clean", "subject_on_the_last_real_frame",
                                                        "step_with_real_and_held_source_frames")},
                            "the_pose_ahead_against_the_kept_frames": kept})
        read = np.nonzero(saved["read"])[0]
        # the voice over the load: a load given a vocal as its audio where there is none was given silence
        heard = voiced[read] if len(read) else np.array([])
        said = None if not len(heard) or np.isnan(heard).all() else {"voiced_frames": int(np.nansum(heard)), "frames": int(len(heard))}
        if said is not None and not isinstance(mouths.get(run["subject"]), str):
            lo, hi = int(first + read.min()), int(first + read.max())
            said["source_mouth_open_on_unvoiced_frames"] = sum(
                min(r["source_frames"][1], hi) - max(r["source_frames"][0], lo) + 1
                for r in mouths.get(run["subject"]) or [] if r["source_frames"][0] <= hi and r["source_frames"][1] >= lo)
        loads.append({"run": run["name"], "subject": run["subject"], "others": run.get("others"), "planned": bool(run.get("planned")),
                      "voice": said,
                      "region_read_from": run.get("read_from"), "source_frames": [int(first + read.min()), int(first + read.max())] if len(read) else None,
                      "margin_px": run.get("margin_px"), "region_share_of_the_frame": round(float(saved["region"][saved["read"]].mean()), 4) if len(read) else None,
                      "inside_the_region": {"by_class": inside, "other_subjects": others,
                                            "why": None if path.is_file() else "no class map in this capture: what lies inside the region is not read by class"},
                      "frames_with_no_mask_inside_a_step": frame_spans(sorted(set(bare) | set(run.get("left_as_the_source_with_no_mask") or []))),
                      "windows": windows or None,
                      "windows_why": None if windows else ("a what-if plan worked out from the masks has no windows: gate on the node's own plan "
                                                           "(--run NAME:preview=)" if run.get("planned") else "a rendered run: its windows are in its own files")})
    return {"rules_input": RULES_INPUT_VERSION, "capture": m["name"], "clip": m["clip"], "first_source_frame": first, "frames": frames,
            "canvas": [w, h], "cell_px": CELL, "token_px": TOKEN_PX, "written": datetime.datetime.now().isoformat(timespec="seconds"),
            "shots": shots, "loads": loads,
            "sources": {"shots": "the cuts of the first shot table in the capture (subjects/<label>/shots__<by>.json)",
                        "picture": "frames.json: source_step, source_sharpness", "on_screen": "subjects/<label>/shots__<by>.json and --not-in",
                        "share_of_the_frame, part_height": "subjects/<label>/per_frame.json: track_share, parts_height_px",
                        "head, the_pose_ahead_against_the_kept_frames": "subjects/<label>/pose__<by>.json (pose_state's columns)",
                        "mouth_open_on_unvoiced_frames": "subjects/<label>/classes__<by>.npz and frames.json: voiced",
                        "part_pixels_owned_by": "subjects/<label>/masks__<by>.npz (parts, as the node wrote it) under owners.npz",
                        "inside_the_region": "runs/<run>/segments_in_region.json and frames.json: region_on__<run>__<other>",
                        "windows": "manifest.json: runs[].planned_windows (the node's own plan)",
                        "who_is_who": "who_is_who.json in the capture (bench/who_is_who_across_shots.py corrections, given to preflight)"}}


def rules(a: argparse.Namespace) -> None:
    folder = Path(a.capture)
    not_in: dict = {}
    for item in a.not_in:
        label, _, rng = item.partition(":")
        not_in.setdefault(label, []).append([int(x) for x in rng.split("-")])
    record = rules_input(folder, not_in)
    (folder / "rules_input.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {folder / 'rules_input.json'}: {len(record['shots'])} shot(s), {len(record['loads'])} load(s)")


def by_shot(record: dict, flags: dict) -> list[dict]:
    """The gate's answer shot by shot and subject by subject: clear, or the rules that block there.

    `record` is rules_input.json and `flags` is flags.json of the same capture. A flag counts where its frames
    touch the shot, and only while it blocks (an overridden flag does not). A flag with no subject (one about
    the shot itself, as an unanswered who-is-who row) is under `every_subject`. Each subject's cell also says
    what the tracker called, at what likeness, how large the subject is and on how many frames it has a mask:
    "clear" says no rule blocks, never that the take is right, and those four are what a reader checks it by.
    A state a person typed on the tracker is marked `typed`, and its likeness is not printed: the table's
    figure there is the tracker's own best in the shot, which need not be the person who was typed."""
    blocking = [f for f in flags["flags"] if f["id"] in set(flags.get("blocking", []))]
    out = []
    for shot in record["shots"]:
        a, b = shot["source_frames"]
        here = [f for f in blocking if any(x[0] <= b and x[1] >= a for x in f["source_frames"])]
        one = {"shot": shot["shot"], "source_frames": [a, b], "subjects": {},
               "every_subject": sorted({f["rule"] for f in here if f.get("subject") is None}),
               "who_is_who": None if shot.get("who_is_who") is None else shot["who_is_who"]["status"]}
        for label, e in shot["subjects"].items():
            share = e.get("share_of_the_frame")
            one["subjects"][label] = {"blocked_by": sorted({f["rule"] for f in here if f.get("subject") == label}),
                                      "state": e["on_screen"]["state"], "likeness": e["on_screen"]["likeness"],
                                      "typed": bool(e["on_screen"].get("typed")),
                                      "share_of_the_frame": None if not share else share["median"],
                                      "frames_with_a_mask": e["frames_with_a_mask"], "frames": e["frames"]}
            whose = e.get("part_pixels_owned_by")
            if whose and whose["frames_on_another_or_contested"]:
                one["subjects"][label]["part_on_another_or_contested"] = {
                    k: whose[k] for k in ("frames_on_another_or_contested", "source_frames_on_another_or_contested", "most_px_on_one_frame")}
        out.append(one)
    return out


def by_load(record: dict, flags: dict) -> list[dict]:
    """The gate's answer load by load: the flags that name a load and block, the warnings by rule, and what the
    lead of a job asks of every load before it renders: frames kept from a window before, held frames the model
    is shown clean, and which other subjects lie inside its region. An overridden flag is listed apart."""
    blocking, passed = set(flags.get("blocking", [])), set(flags.get("overridden", []))
    out = []
    for load in record["loads"]:
        mine = [f for f in flags["flags"] if f.get("run") == load["run"]]
        short = lambda f: {"id": f["id"], "rule": f["rule"], "source_frames": f["source_frames"], **({"other": f["other"]} if f.get("other") else {})}
        warned: dict = {}
        for f in mine:
            if f["id"] not in blocking and f["id"] not in passed and f["level"] != LEVELS[0]:
                warned[f["rule"]] = warned.get(f["rule"], 0) + 1
        windows = load.get("windows") or []
        out.append({"run": load["run"], "subject": load["subject"], "source_frames": load["source_frames"],
                    "blocked_by": [short(f) for f in mine if f["id"] in blocking], "overridden": [short(f) for f in mine if f["id"] in passed],
                    "warnings": dict(sorted(warned.items())),
                    "voice": load.get("voice"),
                    "kept_frames": sum(int(w.get("kept_frames") or 0) for w in windows) if windows else None,
                    "held_frames_shown_clean": sum(int(w.get("held_frames_shown_clean") or 0) for w in windows) if windows else None,
                    "other_subjects_inside": {k: {"frames": v["frames"], "most_share_of_its_mask": v["most_share_of_its_mask"]}
                                              for k, v in ((load.get("inside_the_region") or {}).get("other_subjects") or {}).items()}})
    return out


def shots(a: argparse.Namespace) -> None:
    folder = Path(a.capture)
    for name in ("rules_input.json", "flags.json"):
        if not (folder / name).is_file():
            raise SystemExit(f"{folder / name} is missing: run `preflight` on the capture first")
    flags = json.loads((folder / "flags.json").read_text())
    record = json.loads((folder / "rules_input.json").read_text())
    if a.loads:
        loads = by_load(record, flags)
        if a.json:
            print(json.dumps(loads, indent=1))
            return
        print(f"{folder.name}: gate {flags['verdict']}, {len(flags.get('blocking', []))} blocking; {len(loads)} load(s)")
        for row in loads:
            print(f"{row['run']}: {row['subject']}, source frames {row['source_frames'][0]}-{row['source_frames'][1]}; "
                  + ("no windows read" if row["kept_frames"] is None else
                     f"kept frames {row['kept_frames']}, held frames shown clean {row['held_frames_shown_clean']}")
                  + "".join(f"; {k} inside on {v['frames']} frame(s), up to {100 * v['most_share_of_its_mask']:.0f}% of their mask"
                            for k, v in row["other_subjects_inside"].items())
                  + ("" if not row.get("voice") else f"; voiced on {row['voice']['voiced_frames']} of {row['voice']['frames']} frame(s)"
                     + ("" if not row["voice"].get("source_mouth_open_on_unvoiced_frames") else
                        f", the source's mouth open on {row['voice']['source_mouth_open_on_unvoiced_frames']} unvoiced")))
            for f in row["blocked_by"]:
                print(f"    BLOCKED {f['id']} {f['rule']}" + (f" ({f['other']})" if f.get("other") else "") + f" {f['source_frames'][:6]}")
            for f in row["overridden"]:
                print(f"    overridden {f['id']} {f['rule']}" + (f" ({f['other']})" if f.get("other") else "") + f" {f['source_frames'][:6]}")
            if row["warnings"]:
                print("    warnings: " + ", ".join(f"{k} x{n}" if n > 1 else k for k, n in row["warnings"].items()))
        return
    rows = by_shot(record, flags)
    if a.json:
        print(json.dumps(rows, indent=1))
        return
    print(f"{folder.name}: gate {flags['verdict']}, {len(flags.get('blocking', []))} blocking. "
          "`clear` says no rule blocks, not that a take is right: read the state, likeness and size beside it")
    for row in rows:
        print(f"shot {row['shot']}, source frames {row['source_frames'][0]}-{row['source_frames'][1]}"
              + (f"; who-is-who {row['who_is_who']}" if row["who_is_who"] else "")
              + (f"; BLOCKED for every subject: {', '.join(row['every_subject'])}" if row["every_subject"] else ""))
        for label, c in row["subjects"].items():
            said = (c["state"] or "no shot table") + (", typed" if c["typed"] else "" if c["likeness"] is None else f" {c['likeness']:.2f}") \
                + ("" if c["share_of_the_frame"] is None else f", {100 * c['share_of_the_frame']:.1f}% of the frame") \
                + f", mask on {c['frames_with_a_mask']} of {c['frames']}"
            print(f"    {label}: " + (f"BLOCKED {', '.join(c['blocked_by'])}" if c["blocked_by"] else "clear") + f" ({said})")
            if "part_on_another_or_contested" in c:
                w = c["part_on_another_or_contested"]
                print(f"        its part is on another subject's or contested pixels on {w['frames_on_another_or_contested']} frame(s) "
                      f"{w['source_frames_on_another_or_contested']}, up to {w['most_px_on_one_frame']} px on one")


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
        # the people nobody else holds: a tracker's list may MARK those an earlier tracker has (`held_by_another_tracker`),
        # and a subject truly absent beside somebody else's subject is not a question
        (state, typed), sim = shot_state(shot), shot["subject"].get("similarity")
        people = sum(1 for p in shot.get("people", []) if not p.get("held_by_another_tracker"))
        level = LEVELS[0]
        barred = [x for x in not_in if x[0] <= b and a <= x[1]]
        if state == "absent" and people:
            barred = [x for x in not_in if x[0] <= a and b <= x[1]]      # the whole shot, for an absence to be "as said"
        if state in ("taken", "picked") and barred:
            level = LEVELS[2]
            flags.append({"rule": "taken_where_not_expected", "level": level, "subject": label, "seen_by": by, "source_frames": [[a, b]],
                          "why": f"{label} is {state} in a shot the caller says they are not in", "figures": {"similarity": sim, "line": line}})
        elif state == "taken" and not typed and sim is not None and sim < line:
            # the tracker took somebody UNDER its own match line (its rule for the only person in a shot). Found
            # 2026-10-10 on a load with three subjects: a take at 0.59 against a line of 0.8, of a figure a hundredth
            # of the frame. Under the line the tracker itself does not vouch for it, so a person has to
            level = LEVELS[2]
            flags.append({"rule": "taken_under_its_own_line", "level": level, "subject": label, "seen_by": by, "source_frames": [[a, b]],
                          "why": f"{label} was taken at {sim:.3f}, UNDER the tracker's own line of {line:.3f} ({shot['subject']['state']}): the "
                                 "tracker took the only candidate it had, and does not vouch for it. Look at the tile and answer in "
                                 "writing: `shot N: person K` if it is them, `shot N: none` if not",
                          "figures": {"similarity": sim, "line": line, "people": people}})
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


def flag_never(label: str, by: str, rows: list[dict], table: dict | None, first: int) -> list[dict]:
    """A named subject with no mask on any frame of the capture: the load was planned for somebody it never found.

    Shot by shot an absence with nobody detected is not a question, so a subject whose pick found nobody reads
    "clear" on every such shot (2026-10-10: a tracker given another's mask on `others`, where that other tracker
    had already taken her). Over a whole load it is the first thing to know. Top level; answered by fixing the
    pick or the order of the trackers, or by an override that says the subject is not in this load."""
    if not rows or any(r.get("track_share") for r in rows):
        return []
    picked = table is not None and any(shot_state(s)[0] == "picked" for s in table["shots"])
    return [{"rule": "subject_never_found", "level": LEVELS[2], "subject": label, "seen_by": by,
             "source_frames": [[first, first + len(rows) - 1]],
             "why": f"{label} has no mask on any of the {len(rows)} frames of this load"
                    + ("" if table is None else ": no shot of its table is a pick" if not picked else ": its table has a pick and the track is empty all the same")
                    + ". Nothing can be planned for them. If another tracker runs before this one with its mask on `others`, it may "
                      "have taken this subject first: look at that tracker's takes, change the order or type the pick",
             "figures": {"frames": len(rows), "a_shot_is_picked": bool(picked)}}]


#: What nobody can read off an other's empty track, said wherever one is reported.
OTHER_CANNOT_SEE = ("whether they are hidden on those frames or the tracker lost them while they are in view; in view, with the load's region "
                    "reaching them, nothing would keep the region off them")


def box_gap(box: list[int], cells: np.ndarray, cell: int) -> float | None:
    """The least distance in pixels between a box [x0, y0, x1, y1] and a frame's region cells; 0 when they touch
    or overlap, None when the frame has no region."""
    ys, xs = np.nonzero(cells)
    if not len(ys):
        return None
    dx = np.maximum(0, np.maximum(box[0] - (xs + 1) * cell, xs * cell - box[2]))
    dy = np.maximum(0, np.maximum(box[1] - (ys + 1) * cell, ys * cell - box[3]))
    return round(float(np.hypot(dx, dy).min()), 1)


def run_regions(folder: Path, runs: list[dict]) -> dict:
    """Each run's saved region cells, as {name: (the source frame of its first, [n, h / CELL, w / CELL])}, where it has one."""
    out = {}
    for run in runs:
        path = folder / "runs" / run["name"] / "region.npz"
        if path.is_file() and run.get("first_source_frame") is not None:
            out[run["name"]] = (int(run["first_source_frame"]), np.load(path)["region"])
    return out


def flag_track(label: str, by: str, rows: list[dict], table: dict | None, at: int, runs: list[dict] | None = None,
               regions: dict | None = None, size: list[int] | None = None, cell: int = CELL) -> list[dict]:
    """Frames with no mask inside a shot the subject was taken in.

    FOR A SUBJECT A LOAD REPLACES this is the top level and fails `verify`: nothing regenerates where the load's
    own subject has no mask. FOR ONE WHO IS ONLY AN OTHER of the capture's loads (kept out of a region, replaced
    by none: `runs`, the manifest's) it is data and blocks nothing: a small figure at the back who drops out
    behind somebody is no fault of the load (2026-10-13: lead_fp_1424 stopped at `first` on the second subject,
    hidden behind the third on nine frames of a load of the lead). The flag then says the frames, how large they
    are on the frames beside each gap, and how far each load's region lies from their last box on the empty
    frames (`regions`, the runs' saved cells; `box_gap`), with what it cannot see. A subject of one load and an
    other of another is a subject: it fails as before. With no `runs` handed in everybody is held as a subject."""
    if table is None:
        return []
    taken = [(at + s["first_frame"], at + s["last_frame"]) for s in table["shots"] if shot_state(s)[0] in ("taken", "picked")]
    empty = [r["source_frame"] for r in rows if r["covered"] and not r["track_share"]
             and any(a <= r["source_frame"] <= b for a, b in taken)]
    if not empty:
        return []
    other_of = [run["name"] for run in runs or [] if label in (run.get("others") or [])]
    if not other_of or any(label in subjects_of(run) for run in runs or []):
        return [{"rule": "track_empty_in_a_taken_shot", "level": LEVELS[2], "subject": label, "seen_by": by,
                 "source_frames": frame_spans(empty), "why": f"{label} has no mask on {len(empty)} frame(s) of shots they were taken in: "
                 "nothing regenerates there", "figures": {"frames": len(empty), "each": empty}}]
    area = (size[0] * size[1]) if size else None
    there = sorted((r for r in rows if r.get("track_share") and r.get("track_box")), key=lambda r: r["source_frame"])
    said = lambda r: None if r is None else {"source_frame": r["source_frame"], "share_of_the_frame": r["track_share"],                 # noqa: E731
                                             "px": None if area is None else int(round(r["track_share"] * area)), "box": r["track_box"]}
    beside, gaps = [], {}
    for a, b in frame_spans(empty, join=1):
        before = next((r for r in reversed(there) if r["source_frame"] < a), None)
        after = next((r for r in there if r["source_frame"] > b), None)
        beside.append({"source_frames": [a, b], "before": said(before), "after": said(after)})
        last = before or after
        for name in other_of:
            first, cells = (regions or {}).get(name, (None, None))
            each = {} if cells is None or last is None else {
                f: box_gap(last["track_box"], cells[f - first], cell) for f in range(a, b + 1) if 0 <= f - first < len(cells)}
            had = [g for g in each.values() if g is not None]
            gaps.setdefault(name, []).append({"source_frames": [a, b], "their_last_box_is_of_frame": None if last is None else last["source_frame"],
                                              "least_px": min(had) if had else None, "frames_with_a_region": len(had), "each": each})
    sizes = "; ".join(f"{x['source_frames'][0]}-{x['source_frames'][1]}: " + ", ".join(
        f"{word} it (frame {v['source_frame']}) " + (f"{v['px']} px" if v["px"] is not None else f"{v['share_of_the_frame']} of the frame")
        for word, v in (("before", x["before"]), ("after", x["after"])) if v) or f"{x['source_frames'][0]}-{x['source_frames'][1]}: no frame beside it has a mask"
        for x in beside)
    far = "; ".join(f"{name}: " + ", ".join(
        f"{g['source_frames'][0]}-{g['source_frames'][1]} at least {g['least_px']} px" if g["least_px"] is not None else
        f"{g['source_frames'][0]}-{g['source_frames'][1]} not worked out (no saved region on those frames, or no box of theirs to measure from)" for g in rows_)
        for name, rows_ in gaps.items())
    return [{"rule": "other_has_no_track_in_a_taken_shot", "level": LEVELS[0], "subject": label, "seen_by": by, "source_frames": frame_spans(empty),
             "why": f"{label} is an OTHER of {', '.join(other_of)}, not a subject of any load here, and has no mask on {len(empty)} frame(s) of shots they "
                    f"were taken in. Nothing of theirs regenerates, so this is data and not a failure; on those frames nothing keeps a region off them. "
                    f"Their size on the frames beside: {sizes}. From their last box to the load's region on the empty frames: {far}. "
                    f"CANNOT SEE: {OTHER_CANNOT_SEE}",
             "figures": {"frames": len(empty), "each": empty, "an_other_of": other_of, "beside": beside, "region_gap_px": gaps, "cannot_see": OTHER_CANNOT_SEE}}]


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


def spans_said(frames) -> str:
    return ", ".join(f"{a}-{b}" for a, b in frame_spans(sorted(frames), join=1))


def flag_runs(manifest: dict, folder: Path, own: dict | None = None, given: dict | None = None) -> list[dict]:
    """A run's region over another subject, and a region that is mostly not its subject.

    `own`: the keep-outs that are the load's subject's own things (`--own-keep-out`: her hands, clothes, held things),
    with the classes a restore span would name; `given`: the frames on which something gives a keep-out's pixels back
    (`--given-back`: the plan's restore span of those classes). The lead's ruling of 2026-10-14: a region over her OWN
    hands or held things is the contact fault the owner named (an egg in thin air), so it does not go quiet; it is
    said at the level below the top, with its figure, only where it is given back on every frame it is over, and
    otherwise stays where it was with the restore that would answer it."""
    own, given = own or {}, given or {}
    w_h = int(manifest["size"][0]) * int(manifest["size"][1]) if manifest.get("size") else 0
    cross = json.loads((folder / "frames.json").read_text())["rows"]
    labels = [s["label"] for s in manifest["subjects"]]
    out = []
    for run in manifest["runs"]:
        name, kind = run["name"], "plan" if run.get("planned") else "run"
        whom = " and ".join(subjects_of(run))
        for other in labels:
            if other in subjects_of(run):
                continue                               # a subject of the load is not another subject to it
            col = f"region_on__{name}__{other}"
            hit = [(r["source_frame"], r[col]) for r in cross if r.get(col) is not None and r[col] > REGION_ON_OTHER]
            # the other's own pixels inside the region a frame: its mask's share of the frame (its per_frame `track_share`) times the
            # frame's pixels times the share of it the region holds
            path = folder / "subjects" / other / "per_frame.json"
            share_of_frame = {r["source_frame"]: r.get("track_share") for r in (json.loads(path.read_text())["rows"] if path.is_file() else [])}
            px_in = {f: (share_of_frame.get(f) or 0) * w_h * v for f, v in hit} if share_of_frame and w_h else {}
            # the floor is the STRANGERS mask's only: a subject of ours at a small size can be under it (the third's face at 27 px is about
            # 700 px), and a region over a subject's face is what this flag exists to stop (the lead, 2026-10-14)
            specks = [(f, v) for f, v in hit if other == STRANGERS_LABEL and px_in and px_in[f] < REGION_ON_OTHER_LEAST_PX]
            if specks:
                out.append({"rule": "region_over_another_subject_specks", "level": LEVELS[0], "subject": run["subject"], "run": name, "other": other,
                            "source_frames": frame_spans([f for f, _ in specks]),
                            "why": f"{kind} {name}'s region holds {other}'s mask on {len(specks)} frame(s) where under {REGION_ON_OTHER_LEAST_PX} px of it are "
                                   f"inside (at most {round(max(px_in[f] for f, _ in specks))} px): specks, said as data and not counted against the load",
                            "figures": {"frames": len(specks), "px_inside_most": round(max(px_in[f] for f, _ in specks))},
                            "threshold": {"REGION_ON_OTHER_LEAST_PX": REGION_ON_OTHER_LEAST_PX}})
                hit = [(f, v) for f, v in hit if px_in[f] >= REGION_ON_OTHER_LEAST_PX]
            if hit:
                worst = max(v for _, v in hit)
                held = sum(1 for r in cross if r.get(f"cells_given_back__{name}__{other}"))
                why = (f"{kind} {name} regenerates up to {100 * worst:.0f}% of {other}'s mask: {other} is drawn again "
                       f"there by {whom}'s pass" + (f" (margin cells were given back on {held} frames)" if held else ""))
                level = LEVELS[2] if worst > 5 * REGION_ON_OTHER else LEVELS[1]
                figures = {"frames": len(hit), "worst_share": round(worst, 3), **({"px_inside_most": round(max(px_in[f] for f, _ in hit))} if px_in else {})}
                if px_in:
                    why += f"; at most {round(max(px_in[f] for f, _ in hit))} px of it inside the region on a frame"
                if other in own:
                    back = {f for a, b, _ in given.get(other, []) for f in range(a, b + 1)}
                    rest = [(f, v) for f, v in hit if f not in back]
                    said = sorted({w for a, b, w in given.get(other, []) if any(a <= f <= b for f, _ in hit)})
                    why += f". {other} is {run['subject']}'s OWN"
                    if not rest:
                        level = LEVELS[1]
                        why += f", and on every one of those frames it is given back: {'; '.join(said)}"
                    else:
                        level = LEVELS[2] if max(v for _, v in rest) > 5 * REGION_ON_OTHER else LEVELS[1]
                        why += ((f"; it is given back on {len(hit) - len(rest)} of the frames ({'; '.join(said)})" if said else "")
                                + (f"; a restore span of the plan naming {'+'.join(own[other])} on {spans_said(f for f, _ in rest)} would give "
                                   "those pixels back" if own[other] else
                                   f"; it has no class a restore span can name, so nothing gives it back on {spans_said(f for f, _ in rest)}"))
                    figures.update(own=True, frames_given_back=len(hit) - len(rest))
                out.append({"rule": "region_over_another_subject", "level": level,
                            "subject": run["subject"], "run": name, "other": other, "source_frames": frame_spans([f for f, _ in hit]),
                            "why": why, "figures": figures, "threshold": {"REGION_ON_OTHER": REGION_ON_OTHER}})
        rows = json.loads((folder / "runs" / name / "per_frame.json").read_text())["rows"]
        outside = [(r["source_frame"], r["margin_share"] / r["region_share"]) for r in rows if r.get("region_share")]
        hit = [(f, v) for f, v in outside if v > NOT_SUBJECT]
        if hit:
            typical = float(np.median([v for _, v in outside]))
            out.append({"rule": "region_mostly_not_the_subject", "level": LEVELS[1], "subject": run["subject"], "run": name,
                        "source_frames": frame_spans([f for f, _ in hit], join=RECENT),
                        "why": f"{100 * typical:.0f}% of what {kind} {name} regenerates on a typical frame is not {whom} "
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
        # each subject of the load has its own part: a load of two subjects carries both
        own = {who: next((s["sightings"][0].get("part_is_made_of") or [] for s in manifest["subjects"] if s["label"] == who), [])
               for who in subjects_of(run)}
        kind = "plan" if run.get("planned") else "run"
        for key in sorted({k[:-len("__cells")] for r in rows for k in r if k.endswith("__cells")}):
            label, cls = key.split(".", 1)
            if cls in own.get(label, ()):
                continue                               # the carried part's own classes just outside its mask: the edge
            cells = [(r["source_frame"], r.get(f"{key}__cells", 0), r.get(f"{key}__px", 0)) for r in rows]
            hit = [(f, c, p) for f, c, p in cells if p >= SEGMENT_PX]
            if not hit:
                continue
            other = label not in own
            out.append({"rule": "segment_inside_region", "level": LEVELS[1], "subject": run["subject"] if other else label, "run": run["name"],
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


def bare_frames(carried: np.ndarray, region: np.ndarray, read: np.ndarray, tracked: np.ndarray) -> list[int]:
    """Frames a load reads on which the subject is tracked, the mask the load carries is empty and no region is
    on them: nothing is laid there and the source shows. A part the part model lost, a part the grade emptied
    (too small, or on something else) and a frame the caller dropped all end here."""
    has, on = carried.reshape(len(carried), -1).any(1), region.reshape(len(region), -1).any(1)
    return [int(f) for f in np.nonzero(read & tracked & ~has & ~on)[0]]


def lays_nothing(run: dict, bare: list[int], said: dict, total: int) -> list[dict]:
    """The flags of a load that lays nothing on `bare` frames where its subject is tracked: the frames its job names in
    `said` ({frame: the job's reason}) at the level below the top with the reason; the others at the top."""
    out = []
    meant = [f for f in bare if f in said]
    bare = [f for f in bare if f not in said]
    if meant:
        kind = "plan" if run.get("planned") else "run"
        out.append({"rule": "load_lays_nothing_where_named", "level": LEVELS[1], "subject": run["subject"], "run": run["name"],
                    "source_frames": frame_spans(meant, join=1),
                    "why": f"{kind} {run['name']}: on {len(meant)} frame(s) {run['subject']} is tracked and the load lays nothing, as its job "
                           f"says it means to: " + "; ".join(f"{spans_said(f for f in meant if said[f] == w)}: {w}" for w in sorted({said[f] for f in meant})),
                    "figures": {"frames": len(meant)}})
    if bare:
        kind = "plan" if run.get("planned") else "run"
        out.append({"rule": "load_lays_nothing_on_frames", "level": LEVELS[2], "subject": run["subject"], "run": run["name"],
                    "source_frames": frame_spans(bare, join=1),
                    "why": f"{kind} {run['name']}: on {len(bare)} of its {total} frame(s) {run['subject']} is tracked and the mask the "
                           "load carries is empty: nothing is laid there and the source shows. The part was lost, dropped, or emptied "
                           f"by the grade (subjects/{run['subject']}/part_grades.json says which and why). If the subject is rightly "
                           "left as the source there, override with the reason or cut the load to the frames it does lay; if the "
                           "part should be there, look at those frames before anything renders",
                    "figures": {"frames": len(bare), "frames_of_the_load": total}})
    return out


def flag_lent(manifest: dict, folder: Path, named: dict | None = None) -> list[dict]:
    """A run, or a plan read from the node's files, with frames that have no mask of their own inside a latent
    step that has a region.

    What happens on such a frame depends on the node that made the files. Until 2026-10-10 the composite laid
    the step's region on it: measured on two face-only renders, six such frames beside turns were 7 to 25 grey
    levels from the source inside the lent region against a floor of 2 to 3, and on one a face was drawn under
    a hat brim where the source shows none. Since then the composite leaves the frame as the source, so the
    original shows for that frame. That is right where the mask was emptied on purpose (turned away, hidden)
    and wrong where a tracker or the part model dropped a subject who is plainly there: one frame of the
    original in a run of the new subject. So it is at the top level either way, until the mask is fixed or an
    override says in writing why the frame is rightly empty. A plan worked out from the masks (`--plan`)
    cannot show it: its region is per frame.

    `named`: frames the job leaves empty on purpose, with the reason it writes (`--named-empty`: the job's `drop=` with
    its `drop_by`, its `left_out`). The lead's ruling of 2026-10-14 (21 of 26 overrides on 0-2049 said only that): those
    frames are said with that reason, a level below the top, and only frames nobody named stay at the top."""
    named_frames = {lab: {f: w for a, b, w in v for f in range(a, b + 1)} for lab, v in (named or {}).items()}
    first, w, out = manifest["first_frame"], manifest["size"][0], []
    for run in manifest["runs"]:
        saved = np.load(folder / "runs" / run["name"] / "region.npz")
        carried = np.unpackbits(saved["carried"], axis=-1)[..., :w].astype(bool)
        lent = [first + f for f in lent_frames(carried, saved["region"], saved["read"])]
        # the same question a load at a time: where the subject is tracked and this load lays nothing at all. The
        # grade's own emptyings were only a line in the summary ("by design"), and on 2026-10-10 a face load went
        # to the queue with nothing of a face on most of its frames
        seen = next(x for x in manifest["subjects"] if x["label"] == run["subject"])["sightings"][0]["by"]
        rows = json.loads((folder / "subjects" / run["subject"] / "per_frame.json").read_text())["rows"]
        tracked = np.zeros(manifest["frames"], bool)
        for r in rows:
            if r["seen_by"] == seen and r.get("track_share") and 0 <= r["source_frame"] - first < manifest["frames"]:
                tracked[r["source_frame"] - first] = True
        named = set(lent) | set(run.get("left_as_the_source_with_no_mask") or [])     # the rule below names these
        bare = [first + f for f in bare_frames(carried, saved["region"], saved["read"], tracked) if first + f not in named]
        out += lays_nothing(run, bare, named_frames.get(run["subject"], {}), int(saved["read"].sum()))
        unlaid = [f for f in run.get("left_as_the_source_with_no_mask") or [] if first <= f < first + manifest["frames"]]
        if lent or unlaid:
            kind = "plan" if run.get("planned") else "run"
            what = []
            if lent:
                what.append(f"{len(lent)} frame(s) are regenerated all the same, with their latent step's region: the render can draw the "
                            "part where the source shows none")
            if unlaid:
                what.append(f"{len(unlaid)} frame(s) are left as the source by the composite: the original shows on them")
            out.append({"rule": "region_on_a_frame_with_no_mask", "level": LEVELS[2], "subject": run["subject"], "run": run["name"],
                        "source_frames": frame_spans(sorted(set(lent) | set(unlaid))),
                        "why": f"{kind} {run['name']}: {len(set(lent) | set(unlaid))} frame(s) have no mask of {run['subject']}'s inside a latent "
                               "step that has one. " + "; ".join(what) + ". If the mask is rightly empty there (turned away, hidden), "
                               "override this flag with a note that says so; if the subject is plainly there, the mask was dropped: fix it",
                        "figures": {"regenerated": len(lent), "left_as_the_source": len(unlaid)}})
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
            # the last source frame the window really holds; what it writes after that is a copy of that frame
            real_last = None if win.get("first_frame") is None else win["first_frame"] + win["frames"] - win["held_frames"] - 1
            if real_last is not None and win.get("held_frames") and win["writes_source_frames"][1] > real_last:
                past = win["writes_source_frames"][1] - real_last
                out.append({"rule": "plan_writes_past_the_load", "level": LEVELS[2], "subject": run["subject"], "run": run["name"],
                            "source_frames": [[real_last + 1, real_last + past]],
                            "why": f"plan {run['name']} window {win['window']} writes {past} frame(s) past the load's last frame (source frame "
                                   f"{real_last}): each is that frame held, written into the render where the next source frames belong. "
                                   "The node counts frames by the audio it is given (found 2026-10-10: a track a fraction of a frame long "
                                   "wrote one frame into the next shot). Remedy: cut the audio to the load, or drop those frames when the "
                                   "piece is laid",
                            "figures": {"window": win["window"], "frames_past_the_load": past, "last_frame_of_the_load": real_last}})
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
    for run in by_subject(manifest["runs"]):
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
    for label in sorted({who for run in manifest["runs"] for who in subjects_of(run)}):
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


#: A face part under this many of the model's tokens tall is small for a face-only pass. A token is 32 px.
#: UNMEASURED, and weaker than two data points: on 2026-10-10 the owner's eye called a face about 2.9 tokens tall
#: "not replaced" in one joined file and "correctly swapped" in the next, the SAME render both times with other
#: things around it changed; a face about 5.1 tokens tall read as replaced both times. So this warns that
#: nothing here measures likeness on a small face; it does not say a small face fails.
FACE_TOKENS = 4.0
TOKEN_PX = 32


def largest_piece(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each frame's largest connected piece of a mask, and how many pixels of the frame lie in its other pieces."""
    import cv2
    out, stray = np.zeros_like(mask), np.zeros(len(mask), np.int64)
    for f in range(len(mask)):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask[f].astype(np.uint8), connectivity=8)
        if count > 1:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            out[f] = labels == big
            stray[f] = int(mask[f].sum()) - int(stats[big, cv2.CC_STAT_AREA])
    return out, stray


def flag_pieces(manifest: dict, folder: Path) -> list[dict]:
    """A part with pieces away from its largest piece: the part model labelled something else as the part.

    Found 2026-10-10 on a hair-and-face part: on most frames of a shot a sliver along a forearm or beside a
    dish on the table carried the part's class. They pass the graded fill (they ARE on the part's classes; it
    is the class map that is wrong there), and a margin grows round each of them, so a region appears on an arm
    and on the table. Iffy, with the frames and the most pixels: a part can rightly be in two pieces (a face
    behind a raised arm), so look, and give `pieces=largest` on `--mask` to keep each frame's largest piece. With
    that set this flag is silent for the subject, and what the option took away is the files step's to say
    (`pieces_taken`: printed a frame at a time, and kept in the manifest as `parts_pieces_taken`)."""
    out, w, first = [], manifest["size"][0], manifest["first_frame"]
    for s in manifest["subjects"]:
        seen = s["sightings"][0]
        path = folder / "subjects" / s["label"] / f"parts_held__{seen['by']}.mkv"
        if not path.is_file() or seen.get("parts_pieces") == "largest":
            continue
        held = read_mask(str(path), tuple(manifest["size"]), first, first, manifest["frames"])[0]
        _, stray = largest_piece(held)
        hit = np.nonzero(stray >= SEGMENT_PX)[0]
        if len(hit):
            out.append({"rule": "part_in_pieces", "level": LEVELS[1], "subject": s["label"], "seen_by": seen["by"],
                        "source_frames": frame_spans([int(first + f) for f in hit]),
                        "why": f"{s['label']}'s part has pixels away from its largest piece on {len(hit)} frame(s), up to {int(stray.max())} px: "
                               "something else carries the part's class there, and a margin grows round it. Look at one of the "
                               "frames; `pieces=largest` on --mask keeps each frame's largest piece",
                        "figures": {"frames": int(len(hit)), "stray_px_most": int(stray.max()), "stray_px_median": int(np.median(stray[hit]))},
                        "threshold": {"SEGMENT_PX": SEGMENT_PX}})
    return out


def sizes_of(folder: Path) -> list[tuple[str, dict, str]]:
    """Every subject's part sizes in a capture folder: (where, `part_sizes`, how it was got) a sighting with a part.

    A capture made since 2026-10-13 recorded them and they are read from its manifest. An earlier one is
    measured again here from its own saved masks (`masks__<by>.npz`), with the floor it ran under (the starting
    value: the only one there was) and the frames its `part_grades.json` says were emptied for size."""
    manifest = json.loads((folder / "manifest.json").read_text())
    first, frames, (width, height) = manifest["first_frame"], manifest["frames"], manifest["size"]
    out = []
    for s in manifest["subjects"]:
        for seen in s["sightings"]:
            if not seen.get("parts"):
                continue
            where = f"{folder.name} {s['label']} (by {seen['by']}), source frames {first}-{first + frames - 1}, canvas {width}x{height}"
            if seen.get("part_sizes"):
                out.append((where, seen["part_sizes"], "as the capture recorded it"))
                continue
            saved = np.load(folder / "subjects" / s["label"] / f"masks__{seen['by']}.npz")
            parts = np.unpackbits(saved["parts"], axis=-1)[..., :width].astype(bool)
            grades = folder / "subjects" / s["label"] / "part_grades.json"
            emptied = [d["source_frame"] - first for d in json.loads(grades.read_text())["frames"] if "least_px" in d] if grades.is_file() else None
            out.append((where, part_sizes(parts, first, PART_LEAST_PX, LEAST_STARTING, emptied=emptied,
                                          not_applied="" if emptied is not None else "this capture graded no part (no part_grades.json)"),
                        "measured now from the capture's saved masks; the capture is from before it recorded sizes"))
    return out


def sizes(a: argparse.Namespace) -> None:
    """Print the part sizes of one or more captures, each against the floor it ran under."""
    rows = []
    for name in a.capture:
        for where, found, how in sizes_of(Path(name)):
            rows.append({"capture": Path(name).name, "where": where, "how": how, **found})
            if not a.json:
                print("\n".join(size_lines(where, found)) + f"\n    ({how})")
    if a.json:
        print(json.dumps(rows, indent=1))


def flag_face_size(manifest: dict, folder: Path) -> list[dict]:
    """A run or plan that replaces a face which is only a few of the model's tokens tall.

    Every other reading asks whether the render differs from the source; none asks whether it looks like the
    reference. A small face can pass all of them and still be nobody in particular. From the saved part of a
    subject whose part is made of the face class: its height per frame in tokens, the median over the frames
    that have one. Iffy under `FACE_TOKENS`."""
    out, w = [], manifest["size"][0]
    for label in sorted({who for run in manifest["runs"] for who in subjects_of(run)}):
        seen = next(s for s in manifest["subjects"] if s["label"] == label)["sightings"][0]
        if "Face_Neck" not in (seen.get("part_is_made_of") or []):
            continue
        saved = np.load(folder / "subjects" / label / f"masks__{seen['by']}.npz")
        parts = np.unpackbits(saved["parts"], axis=-1)[..., :w].astype(bool)
        tall = [int(np.ptp(np.nonzero(m.any(axis=1))[0])) + 1 for m in parts if m.any()]
        if not tall:
            continue
        tokens = float(np.median(tall)) / TOKEN_PX
        if tokens < FACE_TOKENS:
            first = manifest["first_frame"]
            out.append({"rule": "face_small_in_tokens", "level": LEVELS[1], "subject": label,
                        "source_frames": [[first, first + manifest["frames"] - 1]],
                        "why": f"{label}'s face part is {np.median(tall):.0f} px tall at the median, {tokens:.1f} of the model's tokens. A face "
                               "that small can be replaced by a face that differs from the source and still does not read as the "
                               "reference's person; no reading here measures likeness to the reference. A tighter canvas on the "
                               "subject, or a look at the first render before the rest, is the check",
                        "figures": {"part_height_px_median": float(np.median(tall)), "tokens": round(tokens, 2)},
                        "threshold": {"FACE_TOKENS": FACE_TOKENS}})
    return out


#: A who-is-who row that needs nobody: `answered` as the tool writes it now, `agrees` as a file made before 2026-10-11 does.
WHO_SETTLED = ("answered", "agrees")


def flag_who(corrections: dict, first: int, frames: int) -> list[dict]:
    """A shot the who-is-who file asks a person about (a tracker's call that is nobody's word, or two trackers on one
    person), until it is answered.

    `corrections` is what `bench/who_is_who_across_shots.py corrections` writes: one row a shot of what each
    tracker called and whose word that is. A row whose `status` is `ask` is at the top level, one flag a shot, so
    a person answers by looking at one tile. The message is each tracker's own `ask` (the question, in that
    tracker's tile's numbers) where there is one, else the row's `why`, with the row's `note` beside it. A
    tracker that has its subject on no shot (`not_picked`) is said first. Once the answers are typed and the
    preview run again the row is `answered`, and the flag is gone by itself. A file written before the tool
    was cut to this (statuses `agrees`, `differs`, `unresolved`) is read the same way: only `agrees` is settled.
    Rows outside the capture's span are left out when the file says which source frame its load starts on;
    when it does not, every row is judged and the flag's frames are the load's own."""
    out, start = [], corrections.get("first_source_frame")
    unpicked = "; ".join(f"{label} picked nobody ({why})" for label, why in (corrections.get("not_picked") or {}).items())
    for row in corrections.get("rows", []):
        if row.get("status") in WHO_SETTLED:
            continue
        a, b = (row["frames"][0] + (start or 0), row["frames"][1] + (start or 0))
        if start is not None and (b < first or a > first + frames - 1):
            continue
        said = [t["ask"] for t in row.get("trackers", {}).values() if t.get("ask")]
        out.append({"rule": "who_is_who_unanswered", "level": LEVELS[2], "subject": None, "source_frames": [[a, b]],
                    "why": (f"FIRST: {unpicked}. " if unpicked else "")
                           + f"shot {row['shot']}: a person is asked ({row['status']}). "
                           + ("; ".join(said) if said else row.get("why") or "no reason given")
                           + (f" [{row['note']}]" if row.get("note") else "")
                           + ("" if start is not None else " (frames are the load's own: the file does not say where the load starts)"),
                    "figures": {"shot": row["shot"], "status": row["status"], "frames_are_source_frames": start is not None}})
    return out


def flag_who_not_run(manifest: dict) -> list[dict]:
    """Several tracked subjects and no who-is-who file in the capture: the joint decision was not run on this load.

    Each tracker decides for its own subject only; `bench/who_is_who_across_shots.py corrections` decides every
    shot for all of them together, and the gate reads its file when it is given. Not given, nothing was said
    and the gate passed: handing the file over was a thing a session had to remember (2026-10-10). Top level,
    over the whole span; answered by running the tool on the load's shot tables and giving the file to
    preflight (`--who-is-who`). A capture with one tracked subject has nothing to decide jointly."""
    tracked = [s["label"] for s in manifest["subjects"] if any(x.get("shots") for x in s["sightings"])]
    if len(tracked) < 2:
        return []
    first, frames = manifest["first_frame"], manifest["frames"]
    return [{"rule": "who_is_who_not_run", "level": LEVELS[2], "subject": None, "source_frames": [[first, first + frames - 1]],
             "why": f"{len(tracked)} subjects are tracked here ({', '.join(tracked)}) and the capture holds no who-is-who file: every shot was "
                    "decided by each tracker alone, and nothing compared them. Run `bench/who_is_who_across_shots.py corrections` on "
                    "this load's shot tables and give its file to preflight with --who-is-who",
             "figures": {"tracked_subjects": tracked}}]


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
                               "covers that. Look at those frames after the render; a load "
                               "that ends on the cut does not have this",
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
    for run in by_subject(manifest["runs"]):
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
    """The sentences of a prompt that say a voice is performed, and those that deny one. A sentence that names a
    reference picture is neither (`REFERENCE_PICTURE`); one that defines a subject by a reference video is read by its
    head only (`REFERENCE_VIDEO_HEAD`)."""
    import re
    says, denies = [], []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        low = sentence.lower()
        if REFERENCE_PICTURE in low:
            continue
        if re.match(REFERENCE_VIDEO_HEAD, low.strip()):
            low = low.split(":", 1)[0]
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


def told_by_the_job(a: argparse.Namespace) -> dict:
    """What the caller says of the load's job (`bench/load_chain.py` reads it from the job file): its subject's own keep-outs
    with their classes, the frames something gives a keep-out back on, and the frames the job leaves empty on purpose."""
    def span_why(text: str, flag: str) -> tuple[str, int, int, str]:
        who, _, rest = text.partition(":")
        rng, sep, why = rest.partition("=")
        try:
            lo, hi = (int(x) for x in rng.split("-"))
        except ValueError:
            raise SystemExit(f"{flag} {text!r}: not NAME:FIRST-LAST=WHY") from None
        if not who or not sep or not why.strip() or lo > hi:
            raise SystemExit(f"{flag} {text!r}: not NAME:FIRST-LAST=WHY, with the why")
        return who, lo, hi, " ".join(why.split())
    own = {}
    for x in getattr(a, "own_keep_out", None) or []:
        name, _, classes = x.partition("=")
        own[name] = [c for c in classes.split("+") if c]
    given, named = {}, {}
    for x in getattr(a, "given_back", None) or []:
        n, lo, hi, why = span_why(x, "--given-back")
        if n not in own:
            raise SystemExit(f"--given-back {x!r}: {n!r} is not one of the --own-keep-out names ({', '.join(own) or 'none'})")
        given.setdefault(n, []).append((lo, hi, why))
    for x in getattr(a, "named_empty", None) or []:
        n, lo, hi, why = span_why(x, "--named-empty")
        named.setdefault(n, []).append((lo, hi, why))
    return {"own": own, "given": given, "named": named}


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
            flags += flag_track(s["label"], by, mine, table, at, m["runs"], run_regions(folder, m["runs"]), m["size"], m.get("cell_px", CELL)) + flag_parts(s["label"], by, mine)
            flags += flag_never(s["label"], by, mine, table, m["first_frame"])
            path = folder / "subjects" / s["label"] / f"pose__{by}.json"
            if path.is_file():
                others_on = {r["source_frame"] for o in m["subjects"] if o["label"] != s["label"]
                             for r in json.loads((folder / "subjects" / o["label"] / "per_frame.json").read_text())["rows"]
                             if r.get("track_share")}
                flags += flag_pose(s["label"], by, json.loads(path.read_text())["rows"], mine, others_on,
                                   bool((seen.get("pose") or {}).get("hand_refinement")))
    told = told_by_the_job(a)
    flags += flag_runs(m, folder, told["own"], told["given"]) + flag_segments(m, folder) + flag_keep(m, folder) + flag_cuts(m, folder) + flag_loads(m, folder, not_in) + flag_lent(m, folder, told["named"]) + flag_held_tail(m, folder) + flag_kept(m, folder) + flag_mouth(m, folder) + flag_face_size(m, folder) + flag_pieces(m, folder)
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
    if a.who_is_who:
        # kept in the capture, so the rules' input and a later run of the gate read the same answer
        shutil.copyfile(a.who_is_who, folder / "who_is_who.json")
    if (folder / "who_is_who.json").is_file():
        flags += flag_who(json.loads((folder / "who_is_who.json").read_text()), m["first_frame"], m["frames"])
    else:
        flags += flag_who_not_run(m)
    if a.text and a.voice_spans:
        spans = json.loads(Path(a.voice_spans).read_text())["voiced_spans_inclusive"]
        flags += flag_text(Path(a.text).read_text(), spans, m["first_frame"], m["frames"])
    order = {level: i for i, level in enumerate(LEVELS)}
    flags.sort(key=lambda f: (-order[f["level"]], f["source_frames"][0][0]))
    for i, f in enumerate(flags, 1):
        f["id"], f["key"] = f"f{i:03d}", flag_key(f)
    gate = gate_verdict(flags, read_outcomes(folder))
    # decisions the capture made itself, each with its record: seen in the summary, not asked as a question
    by_design = {s["label"]: n for s in m["subjects"] for n in
                 [sum(b - a + 1 for x in s["sightings"] for a, b in x.get("parts_emptied_by_grade_on_source_frames") or [])] if n}
    record = {**gate, "emptied_by_the_grade": by_design, "capture": m["name"], "clip": m["clip"], "span": [m["first_frame"], m["first_frame"] + m["frames"] - 1],
              "written": datetime.datetime.now().isoformat(timespec="seconds"), "levels": list(LEVELS),
              # a tracker's own object id can sit beside the label when a tracker hands one back; none does today
              "subjects": [{"label": s["label"], "tracker_object_id": None, "seen_by": [x["by"] for x in s["sightings"]]}
                           for s in m["subjects"]],
              "shots": shots, "flags": flags,
              "told_by_the_job": {"own_keep_outs": told["own"], "given_back": {k: [list(x) for x in v] for k, v in told["given"].items()},
                                  "named_empty": {k: [list(x) for x in v] for k, v in told["named"].items()}},
              "outcomes": "outcomes.json, beside this file: what happened in a render against each flag id, once one exists"}
    if (folder / "flags.json").is_file() and (folder / "outcomes.json").is_file():
        # outcomes name flags by the id they had when they were written; a new list renumbers. Keep the list they
        # were written against (an outcome's own `rule` and `key` say which flag it is in any list)
        kept = folder / f"flags.before_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        shutil.copyfile(folder / "flags.json", kept)
        print(f"this capture has outcomes: the flag list they were written against is kept as {kept.name}")
    (folder / "flags.json").write_text(json.dumps(record, indent=1) + "\n")
    for f in flags:
        spans_text = ", ".join(f"{x}-{y}" if x != y else str(x) for x, y in f["source_frames"][:8]) + (" ..." if len(f["source_frames"]) > 8 else "")
        print(f"{f['id']}  {f['level'].upper():14}  {f['rule']}\n      {f['why']}\n      source frames {spans_text}")
    counts = {level: sum(f["level"] == level for f in flags) for level in LEVELS}
    print(f"{len(flags)} flag(s): " + ", ".join(f"{n} {level}" for level, n in counts.items()) + f"; wrote {folder / 'flags.json'}")
    for label, n in by_design.items():
        print(f"{n} frame(s) of {label}'s part emptied by the grade, by design: subjects/{label}/part_grades.json")
    # the figures a per-clip rule reads, in one file, never older than the flags
    (folder / "rules_input.json").write_text(json.dumps(rules_input(folder, not_in), indent=1) + "\n")
    print(f"gate: {gate['verdict']}" + (f", on {', '.join(gate['blocking'])}" if gate["blocking"] else "")
          + (f"; overridden: {', '.join(gate['overridden'])}" if gate["overridden"] else ""))
    if a.gate and gate["blocking"]:
        sys.exit(GATE_BLOCKED)


def look_entry(status: dict, shown: list[dict], by: str, note: str, now: str) -> dict:
    """One line of `looks.json`: who went through a capture's loads with their own eyes, when, what each strip
    showed, and what they say. A passing check says no rule failed, never that the masks are on the right person
    or that a text fits its frames (owner, 2026-10-10), so a capture's loads are looked at by somebody who did
    not make it before a plan or a render is built on them.

    `shown` is one row a load, as `video --load` wrote it beside the strip (`<load>.shown.json`): the load's
    frames, the frames its text was written from and the hash of the text that was under the picture. The job
    builder compares those three with what it is about to build, and the look's time with the capture's own
    (`status.json`, `written`): a capture rebuilt after the look, a load whose frames moved, or a text that is
    not the one that was read, is looked at again. A hash and not a time for the text, because a file's time
    passes a text copied in with an old one."""
    if not by.strip() or not note.strip():
        raise SystemExit("a look is a person's word: give --by and a --note that says what was looked at and what was seen")
    if status.get("made_by") and status["made_by"] == by:
        raise SystemExit(f"{by} made this capture: the look is somebody else's")
    if not shown:
        raise SystemExit("no strip was shown: a look of nothing. Draw each load's with `video <capture> --source CLIP --job JOB.json --load NAME`")
    old = [r["load"] for r in shown if r.get("made", "") <= str(status.get("written", ""))]
    if old:
        raise SystemExit(f"the strip of {', '.join(old)} was drawn before this capture was last written: draw it again and look at that one")
    return {"by": by, "at": now, "loads": [r["load"] for r in shown], "note": note.strip(), "shown": shown,
            "capture_written": status.get("written"), "maker": status.get("made_by") or "not recorded"}


def strip_home(folder: Path, name: str = "") -> Path:
    """Where a look's strip is written: the share's `Video/<video>/looks/<capture>/`, never under `data/` (the
    owner, 2026-10-12, in the lead's terminal: a video clip goes under the share's output folder, where they can
    see it; AGENTS.md has the rule). The video's name is its analysis folder's (`clip_folder`, from the capture's
    `clip`) and the output folder the server's (`h3_config.output_dir`, which on this box needs `H3_OUTPUT_DIR`).
    Until 2026-10-12 the strip went beside the capture under `looks/`; the record of what it showed stays there
    (`looks/<load>.shown.json`) and names the strip's full path, so `looked` and the job builder find it in
    either place. `name` (`video --strip-folder`) is the folder under `looks/` in place of the capture's own
    name: a capture that will be built again in place keeps its first strip by drawing it under another name
    from the start, so nothing is moved later and no record has its path rewritten."""
    if name and (Path(name).name != name or name in (".", "..")):
        raise SystemExit(f"--strip-folder {name!r}: one folder name under the video's looks/, not a path")
    sys.path.insert(0, str(REPO))
    import clip_folder  # noqa: E402
    sys.path.insert(0, str(REPO / "workflows"))
    import h3_config  # noqa: E402
    clip = json.loads((folder / "manifest.json").read_text())["clip"]
    home = h3_config.output_dir() / "Video" / clip_folder.folder_of_file(clip, REPO / "data").name / "looks" / (name or folder.name)
    home.mkdir(parents=True, exist_ok=True)
    return home


def strip_of(folder: Path, name: str) -> Path | None:
    """The strip drawn for a load, where its record says it is (`looks/<load>.shown.json`, `strip`: a full path
    since 2026-10-12, a bare file name beside the capture before), or beside the capture when no record was written;
    None when none is drawn."""
    record = folder / "looks" / f"{name}.shown.json"
    named = json.loads(record.read_text()).get("strip") if record.is_file() else None
    for path in ((Path(named) if named and Path(named).is_absolute() else folder / "looks" / (named or f"{name}.mp4")),
                 folder / "looks" / f"{name}.mp4"):
        if path.is_file():
            return path
    return None


def looked(a: argparse.Namespace) -> None:
    folder = Path(a.capture)
    status = json.loads((folder / "status.json").read_text())
    have = {p.name[:-len(".shown.json")]: p for p in sorted((folder / "looks").glob("*.shown.json"))} if (folder / "looks").is_dir() else {}
    rows = {n: json.loads(p.read_text()) for n, p in have.items()}
    # with no --loads: every strip drawn since the capture was last written; an older strip is of an earlier capture
    names = sorted(n for n, r in rows.items() if r.get("made", "") > str(status.get("written", ""))) if not a.loads or a.loads == "all" \
        else a.loads.split("+")
    unrecorded = [n for n in names if n not in have and strip_of(folder, n) is not None]
    if unrecorded:
        raise SystemExit(f"a strip of {', '.join(unrecorded)} was drawn and nothing was recorded for it: the job file does not say which "
                         "frames its text was written from, or by whom, so no look of it can be recorded")
    missing = [n for n in names if n not in have]
    if missing:
        raise SystemExit(f"no strip was made for {', '.join(missing)}: a look of nothing. This capture has strips for "
                         f"{', '.join(sorted(have)) or 'no load'}")
    kept = folder / "looks.json"
    looks = json.loads(kept.read_text())["looks"] if kept.is_file() else []
    looks.append(look_entry(status, [rows[n] for n in names], a.by, a.note,
                            datetime.datetime.now().isoformat(timespec="seconds")))
    kept.write_text(json.dumps({"what": "who looked through this capture's loads, appended and never rewritten; each line holds what "
                                        "the strips it looked at showed (`shown`)", "looks": looks}, indent=1) + "\n")
    print(f"recorded: {a.by} looked at {', '.join(names)} of {folder.name}; {len(looks)} look(s) in {kept}")


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
        why = "every frame is voiced" if (voiced == 1).any() and not np.isnan(voiced).any() else \
            "no frame is known to be unvoiced: the voice table does not cover these frames (--voice)"
        return {"bar": None, "runs": [], "why_none": why}
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
    track = {s["label"]: s["track"][n] for s in subjects}
    for r in runs:
        c = colour[r["subject"]]
        region, carried = cells_up(r["region"][n]), r["carried"][n]
        margin = region & ~carried
        img[margin] = img[margin] * (1 - LIGHT) + c * LIGHT
        img[carried] = img[carried] * (1 - SOLID) + c * SOLID
        # a load of two subjects carries both parts as one mask: what lies on the second's track and not on the first's is
        # drawn in the second's colour, over the first's. A pixel both tracks claim stays the first's; the margin is one, the first's
        for other in (r.get("subjects") or [])[1:]:
            theirs = carried & track[other] & ~track[r["subject"]]
            img[theirs] = plate.astype(np.float32)[theirs] * 0.45 * (1 - SOLID) + colour[other] * SOLID
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
            both = r.get("subjects") or [r["subject"]]
            will = "would be " if r.get("planned") else ""
            if s["label"] in both:
                lines.append((s["colour"], "solid", f"{s['label']}: {will}taken from the still ({'plan' if r.get('planned') else 'run'} {r['name']})"))
            if r["subject"] == s["label"]:
                lines.append((s["colour"], "light", f"{s['label']}: margin {will}regenerated round it" if len(both) == 1 else
                              f"one margin {will}regenerated round {' and '.join(both)}"))
    if any(r["back"] for r in runs):
        lines.append((WHITE, "outline", "margin cells given back to another subject"))
    return lines


def render_said(run: dict, job_path: Path, name: str) -> list[str]:
    """What the node will render of a load, said on its strip (the lead, 2026-10-14: every blind reader of that day read a strip
    that said nowhere whether its load renders zoomed, nor its window, nor the defaults it carries): zoomed in, by how much and on
    which box, or on the whole frame and the node's words why; each planned window's frames and held frames; the job's defaults."""
    import re
    zoom = run.get("zoom")
    if zoom:
        out = [f"RENDERED ZOOMED IN {float(zoom['scale']):.2f}x on the box {list(zoom['box'])} of the {zoom['source_size'][0]}x{zoom['source_size'][1]} "
               f"frame (the node's plan; {zoom.get('set_by', '')}): what this strip shows is enlarged that much in the render"]
    else:
        why = ""
        job = json.loads(job_path.read_text()) if job_path.is_file() else {}
        report = REPO / f"{(job.get('job') or {}).get('capture', '')}_job" / f"{name}.plan.queued.json"
        if report.is_file():
            found = re.search(r"rendered on the whole frame: ([^\\\n\"]*)", report.read_text())
            why = found.group(1) if found else ""
        out = ["RENDERED ON THE WHOLE FRAME" + (f": the node's words, {why}" if why else " (the node's plan names no zoom)")]
    for w in run.get("planned_windows") or []:
        out.append(f"window {w.get('window')}: {w.get('frames')} frames from source frame {w.get('first_frame')}, writing {w['writes_source_frames'][0]}-"
                   f"{w['writes_source_frames'][1]}; {w.get('held_frames', 0)} held frame(s), {w.get('kept_frames', 0)} kept")
    defaults = (json.loads(job_path.read_text()).get("job") or {}).get("defaults") if job_path.is_file() else None
    if isinstance(defaults, list) and defaults:
        out.append("THE DEFAULTS THIS LOAD CARRIES: " + "; ".join(f"{d['what']}: {d['carries']}" for d in defaults))
    return out


def load_and_text(job_path: Path, name: str) -> dict:
    """One load of a job file and the text it will be given, as the look shows them and records them.

    The keys are the job builder's (`bench/masked_job.py`): a load's `first`, `frames` and `text` (a label), and
    in the file's `job` block `texts[label]` = {"file": a name beside the job file, "frames": the source frames
    the text was written from, "by": who wrote it}. `text_sha256` is of the file's text with the whitespace at its ends stripped, the
    string the builder puts in the graph: the builder's rule, restated here so that a strip can be drawn for a
    job its build refuses. A job file with no such entry (every one made before 2026-10-11) is shown with the
    built text found beside it and `recorded` false: where the text was written from and by whom is not known,
    and no look of it can be recorded.

    A load of two subjects is read by the builder's own `one_subject_view`, as `files` reads it: `subject` is its
    first subject, and `pair` lists both in the list's order with which label is whose (`<Subject N>`, the still
    `<Picture N>`, the reference `<Video N>`), by the builder's rule (`bench/masked_job.py::pair_of`). Nothing but
    the text tells the model that binding, so the look's strip prints it. `motion` is each reference video the
    load names with the source frame its frame 0 is on (`motion_first`, else the job's `first`, the builder's
    rule; none when neither is given), in the same order."""
    import hashlib
    sys.path.insert(0, str(REPO / "workflows"))
    import masked_job_graph                                         # the one reading of a two-subject load, the job builder's
    file = json.loads(job_path.read_text())
    named = [x.get("name") for x in file.get("loads", [])]
    if name not in named:
        raise SystemExit(f"no load named {name} in {job_path.name}; it has {', '.join(str(x) for x in named) or 'none'}")
    try:
        load = masked_job_graph.one_subject_view({**(file.get("defaults") or {}), **next(x for x in file["loads"] if x.get("name") == name)})
    except ValueError as e:
        raise SystemExit(f"{job_path.name}: {e}") from e
    first, count, label = int(load["first"]), int(load["frames"]), load.get("text")
    out = {"load": name, "frames": [first, first + count - 1], "subject": load.get("subject"), "text_label": label, "recorded": False,
           "text": "", "text_from": None,
           "motion": []}
    job_first = (file.get("job") or {}).get("first")              # the builder's rule: a reference with no `motion_first` begins on the job's first frame
    for x in ("", "_2"):
        if load.get(f"motion_file{x}"):
            begins = load.get(f"motion_first{x}", job_first)
            out["motion"].append({"file": str(load[f"motion_file{x}"]), "first": None if begins is None else int(begins)})
    also = load.get("also")
    if also:
        # a subject's still, given once in the job block, is every load's of that subject unless the load names its own (the builder's rule)
        stills = {s_["label"]: s_["still"] for s_ in (file.get("job") or {}).get("subjects", []) if s_.get("still")}
        out["pair"] = [{"subject": who, "region": given["region"], "is": f"<Subject {k}>", "picture": f"<Picture {k}>",
                        "still": own.get("still", stills.get(who)), "video": f"<Video {k}>" if load.get(key) else None}
                       for k, (who, given, own, key) in enumerate(((load["subject"], load["subjects"][0], load, "motion_file"),
                                                                   (also["subject"], load["subjects"][1], also, "motion_file_2")), 1)]
    entry = ((file.get("job") or {}).get("texts") or {}).get(label)      # in the job block, where the builder reads it
    if isinstance(entry, dict) and entry.get("file") and entry.get("frames") and entry.get("by"):
        text = (job_path.parent / entry["file"]).read_text().strip()
        out.update(recorded=True, text=text, text_from=entry["file"], text_frames=[int(x) for x in entry["frames"]], text_by=str(entry["by"]),
                   text_sha256=hashlib.sha256(text.encode()).hexdigest())
    else:
        for stage in ("render", "plan"):
            built = job_path.parent / f"{name}.{stage}.txt"
            if built.is_file():
                out.update(text=built.read_text().strip(), text_from=built.name)
                break
    return out


def wrap_text(text: str, width: float, measure) -> list[str]:
    """A text as lines no wider than `width` by `measure` (a function of a string), every word kept and its own
    line breaks kept: the look shows the whole text, never a shortened one."""
    out = []
    for para in text.splitlines():
        line = ""
        for word in para.split():
            if line and measure(line + " " + word) > width:
                out.append(line)
                line = word
            else:
                line = (line + " " + word).strip()
        out.append(line)
    return out


def render_span(first: int, frames: int, render_first: int, render_frames: int) -> dict:
    """The frames a strip of a render shows, on the span's clock and on the render's, and what of the span it leaves out.

    `first` and `frames` are the capture's span in source frames; `render_first` is the source frame the render's
    frame 0 is on and `render_frames` its length. A render need not begin where its capture does: a load whose
    first frames are left as the source, and a run that continues a stored window, both write a file that begins
    inside the span. `lo` is how many of the span's frames lie before the render and `skip` how many of the
    render's lie before the span (one of the two is 0); `frames` is what both hold from there; `left_out` is the
    span's source frames the strip does not show, first and last, with why. No frame in both is refused."""
    skip, lo = max(0, first - render_first), max(0, render_first - first)
    shown = min(frames - lo, render_frames - skip)
    if shown <= 0:
        raise SystemExit(f"the render is source frames {render_first}-{render_first + render_frames - 1} and the capture's span is "
                         f"{first}-{first + frames - 1}: no frame is in both")
    left = []
    if lo:
        left.append({"frames": [first, first + lo - 1], "why": "before the render"})
    if lo + shown < frames:
        left.append({"frames": [first + lo + shown, first + frames - 1], "why": "after the render"})
    return {"skip": skip, "lo": lo, "frames": shown, "left_out": left}


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
    skip, lo, shown, left_out = 0, 0, None, []
    if a.load:
        # the look: one load's own frames, every one, with the text it will be given under the picture
        if not a.job or render is not None:
            raise SystemExit("--load draws one load's strip from its job file, before anything renders: give --job, and no --render")
        shown = load_and_text(Path(a.job), a.load)
        lo, frames = shown["frames"][0] - first, shown["frames"][1] - shown["frames"][0] + 1
        if lo < 0 or lo + frames > m["frames"]:
            raise SystemExit(f"load {a.load} is source frames {shown['frames'][0]}-{shown['frames'][1]}; this capture holds "
                             f"{first}-{first + m['frames'] - 1}")
        replaced = [p_["subject"] for p_ in shown["pair"]] if shown.get("pair") else [shown["subject"]]
        for who in replaced:
            if who not in [s_["label"] for s_ in subjects]:
                raise SystemExit(f"load {a.load} replaces {who!r}, and this capture has no subject of that name: it was not made "
                                 f"for this job (it has {', '.join(s_['label'] for s_ in subjects)})")
        # a run of a two-subject load is the first subject's in the capture: the strip is told both, from the job file
        runs = [{**r, "subjects": replaced} if len(replaced) > 1 else r for r in runs if r["name"] == a.load]
        for r in runs:
            if r.get("planned") and not r.get("files") and len(replaced) > 1 and "subjects" not in next(x for x in m["runs"] if x["name"] == r["name"]):
                print(f"this capture's plan of {a.load} was worked out for {r['subject']} alone (it was made before the capture read a load of two "
                      "subjects): the region drawn is not the second subject's. Build the capture again", flush=True)
        # before a plan the capture holds no run of the load: the look is then of its frames, masks and text, and says so
        shown["region"] = "the node's plan of this load" if runs and runs[0].get("planned") else \
            "this load's render" if runs else "none: the capture holds no plan of this load yet"
    if render is not None:
        # the render's frame 0 on the span's clock: --render-first, or the run that wrote it says, or the span's first frame
        known = next((r["first_source_frame"] for r in runs if r.get("render") == Path(render).name), None)
        render_first = int(a.render_first) if a.render_first is not None else first if known is None else known
        took = "--render-first" if a.render_first is not None else "the run that wrote it" if known is not None else \
            "NOT SAID: no run of this capture names this file, so the span's first frame"
        print(f"the render's frame 0 is taken as source frame {render_first} ({took})" + ("" if a.render_first is not None or known is not None else
              "; the capture's runs begin at " + ", ".join(f"{r['name']}: {r.get('first_source_frame', first)}" for r in runs)), flush=True)
        span = render_span(first, frames, render_first, probe(render)[2])
        skip, lo, frames, left_out = span["skip"], span["lo"], span["frames"], span["left_out"]
        for one in left_out:
            print(f"not in the strip: span frames {one['frames'][0]}-{one['frames'][1]}, {one['why']}", flush=True)
        # a render that is one of the capture's runs is shown with that run's region alone; any other with all
        own = [r for r in runs if r.get("render") == Path(render).name]
        runs = own if own and not a.all_runs else runs
    focus = np.zeros((m["frames"], h // CELL, w // CELL), bool)
    for r in runs:
        focus |= r["region"]
    if not focus.any():
        # nothing regenerates in what is drawn: zoom on the subject the load replaces, or the first when no load is named
        focus = cells_any(next((s_ for s_ in subjects if shown is not None and s_["label"] == shown["subject"]), subjects[0])["track"])
        for p_ in (shown or {}).get("pair", [])[1:]:       # both subjects of a two-subject load are zoomed on
            focus = focus | cells_any(next(s_ for s_ in subjects if s_["label"] == p_["subject"])["track"])
    boxes = zoom_boxes(focus, (w, h))
    z = h                                             # the zoom tile is the row's height, square
    row_h, n_rows = h + HEADER, 3 if render else 2
    canvas_w = w + z
    big, text, small = ImageFont.truetype(MONO, 30), ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 17)
    huge, line = ImageFont.truetype(MONO, 92), ImageFont.truetype(FONT, 19)
    text_lines = [] if shown is None else wrap_text(shown["text"] or "(no text was found for this load)", canvas_w - 32, line.getlength)
    notice = 26 if shown is not None and not runs else 0   # one more line when the capture holds no plan of the load
    # where the text was written from, wrapped to the strip's width like every line under it (a generated text's "by" is long)
    covers = shown is not None and shown["recorded"] and shown["text_frames"][0] <= shown["frames"][0] and shown["frames"][1] <= shown["text_frames"][1]
    written = [] if shown is None else wrap_text(
        (f"written from source frames {shown['text_frames'][0]}-{shown['text_frames'][1]} by {shown['text_by']}"
         + ("" if covers else "   THIS LOAD HAS FRAMES THE TEXT WAS NOT WRITTEN FROM")) if shown["recorded"]
        else "NOT RECORDED: which frames this text was written from, and by whom. No look of it can be recorded", canvas_w - 32, line.getlength)
    more = 24 * max(0, len(written) - 1)
    notice += more
    # a load of two subjects: which label is whose, in the list's order. Only the text tells the model, so the look says it
    whose = [] if shown is None or not shown.get("pair") else wrap_text("TWO SUBJECTS IN ONE PASS, IN THE LIST'S ORDER: " + "; ".join(
        f"{p_['subject']} ({p_['region']}) is {p_['is']}, its still {p_['picture']}" + (f" ({Path(p_['still']).name})" if p_.get("still") else " (none named)")
        + (f", its reference {p_['video']}" if p_["video"] else ", no reference video") for p_ in shown["pair"]), canvas_w - 32, line.getlength)
    notice += 24 * len(whose) + (6 if whose else 0)
    # what the node will render of it, on the header of the strip: zoom or whole frame, windows, the defaults
    rendered = [] if shown is None or not runs else [x for one in render_said(next((r for r in runs if r["name"] == a.load), runs[0]), Path(a.job), a.load)
                                                    for x in wrap_text(one, canvas_w - 32, line.getlength)]
    notice += 24 * len(rendered) + (6 if rendered else 0)
    band = 0 if shown is None else 2 * ((74 + notice + 24 * len(text_lines)) // 2 + 1)   # even: the encoder's pixel format needs it
    canvas_h = row_h * n_rows + PANEL + band
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
    # the motion video's frame 0 is the run's first frame: on the render's clock with a render, on the load's without.
    # A load that says which source frame its reference begins on (`motion_first`) is read from there. One video fills
    # the tile; two (a load of two subjects) stand side by side at half its size, each under its own label.
    given = a.motion or []
    if len(given) > 2:
        raise SystemExit("--motion: one reference video, or two for a load of two subjects")
    motion_tile = (456, 342) if len(given) < 2 else (228, 171)
    told, pair = (shown or {}).get("motion") or [], (shown or {}).get("pair") or []
    motions, motion_names = [], []
    for path in given:
        j = next((j for j, m_ in enumerate(told) if Path(m_["file"]).name == Path(path).name), None)
        begins = None if j is None else told[j]["first"]
        start = skip if render else lo if begins is None else max(0, shown["frames"][0] - begins)
        if begins is not None and begins > shown["frames"][0]:
            print(f"--motion {Path(path).name} begins on source frame {begins}, after the load's first ({shown['frames'][0]}): shown from its frame 0", flush=True)
        mw, mh = motion_tile
        motions.append(stream(path, motion_tile, start, frames, pix="rgb24",
                              vf=f"scale={mw}:{mh}:force_original_aspect_ratio=decrease,pad={mw}:{mh}:(ow-iw)/2:(oh-ih)/2"))
        # the load's k-th reference is its k-th subject's (a second reference without a first is refused by the builder)
        motion_names.append(f"{pair[j]['video']} {pair[j]['subject']}" if pair and j is not None else
                            Path(path).name[:26] if len(given) > 1 else "")
    region_px = [cells_up(r["region"]) for r in runs]
    before = None
    lines = legend_lines(subjects, runs)
    out = a.out or str(folder / f"{m['name']}_diag.mp4")
    if shown is not None and not a.out:
        (folder / "looks").mkdir(exist_ok=True)       # the record of what was shown stays beside the capture
        out = str(strip_home(folder, getattr(a, "strip_folder", "") or "") / f"{a.load}.mp4")   # the strip itself goes to the share, never under data/
    # the sound: the render's own when it carries one; else the source's at the render's first frame (a render saved
    # without an audio stream, as the first window of a two-window run and a joined viewing copy were on 2026-10-12,
    # gave the owner a silent clip; the source has the sound on every frame)
    has_audio = bool(render) and subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type",
                                                 "-of", "csv=p=0", render], capture_output=True, text=True).stdout.strip() != ""
    if has_audio:
        audio = (["-ss", f"{skip / FPS:.6f}"] if skip else []) + ["-i", render]
    else:
        audio = ["-ss", f"{(first + lo) / FPS:.6f}", "-i", str(source)]
    enc = subprocess.Popen(
        ["nice", "-n", "19", "ffmpeg", "-v", "error", "-y", "-threads", "4", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{canvas_w}x{canvas_h}", "-r", str(FPS), "-i", "-", *audio, "-map", "0:v:0", "-map", "1:a:0?",
         "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-t", f"{frames / FPS:.3f}", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    plates = stream(str(source), (w, h), first + lo, frames, vf=FIT.format(w=w, h=h), pix="rgb24")
    renders = stream(render, (w, h), skip, frames, vf=f"scale={w}:{h}", pix="rgb24") if render else None
    names = ["SOURCE", "MASKS", "RENDER"]
    render_name = Path(render).name if render else ""
    done = 0
    for n, plate in enumerate(plates, lo):
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
            d.text((10, top + 6), f"SOURCE FRAME {first + n:05d}  " + (f"RENDER FRAME {skip + n - lo:04d}" if shown is None else
                                                                   f"LOAD FRAME {n - lo:04d}"), font=big, fill=WHITE)
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
        if shown is None:
            d.text((16, py + 130), f"render frame {skip + n - lo:04d}" if render else "nothing rendered: a preview", font=text, fill=WHITE)
        else:                                             # two lines: a load's name is as long as the column is wide
            d.text((16, py + 126), f"load {a.load}", font=small, fill=WHITE)
            d.text((16, py + 144), f"frame {n - lo + 1} of {frames}", font=small, fill=WHITE)
        if starts:
            window = sum(skip + n - lo >= x for x in starts)
            d.text((16, py + 160), f"window {window} of {len(starts)}", font=line, fill=WHITE)
            d.text((16, py + 184), "new frames from " + ", ".join(str(x) for x in starts), font=small, fill=DIM)
        d.text((16, py + 214), said[0] or "no voice table", font=text, fill=said[1])
        if left_out:
            # a render that covers part of the span: every frame of the strip says which of the span's frames it does not show
            d.text((16, py + 250), "span frames not in this strip:", font=small, fill=AMBER)
            for k, one in enumerate(left_out):
                d.text((16, py + 274 + 23 * k), f"{one['frames'][0]}-{one['frames'][1]}, {one['why']}", font=small, fill=AMBER)
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
                own = sum(table[who][n].get("track_share") or 0 for who in subjects_of(r_))
                d.text((tx, ty), f"{'plan' if r_.get('planned') else 'run'} {r_['name']}: region {100 * r['region_share']:.1f}% of the frame"
                       + (f", {r['region_share'] / own:.1f}x {' and '.join(subjects_of(r_))}'s mask" if own else "")
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
            drawn = {r_["name"] for r_ in runs}
            live = [f for f in flags if any(x <= first + n <= y for x, y in f["source_frames"]) and f["level"] != LEVELS[0]
                    and f.get("run") in (None, *drawn)]
            if not live:
                d.text((tx, ty + 4), "no flag on this frame", font=small, fill=DIM)
            for f in live[:max(1, (py + PANEL - ty - 8) // 23)]:
                words = f"{f['id']} {f['level'].upper()}: {f['why']}"
                while d.textlength(words, font=small) > PANEL_TEXT - tx and len(words) > 20:
                    words = words[:-8] + "..."
                d.text((tx, ty + 4), words, font=small, fill=LEVEL_COLOURS[f["level"]])
                ty += 23
        mx = PANEL_TEXT + 12
        d.text((mx, py + 8), "movement: the WHOLE reference frame, as shown" if motions else "movement, as the model was shown it",
               font=small, fill=AMBER)
        if motions:
            for at, (one_motion, label) in enumerate(zip(motions, motion_names)):
                moving = next(one_motion, None)
                if moving is not None:
                    img.paste(Image.fromarray(moving), (mx + at * motion_tile[0], py + 34))
                # The tile's own edge, drawn: a reference on a black ground has no edge to see on a black panel, so a
                # figure its own frame cuts (a head crop does, at the frame's side) read as the PANEL cropping it. Twice
                # on 2026-10-13 a reader took it so; measured that day, the tile is the whole frame scaled, to the pixel.
                d.rectangle([mx + at * motion_tile[0], py + 34, mx + (at + 1) * motion_tile[0] - 1, py + 34 + motion_tile[1] - 1], outline=DIM)
                if label:
                    d.text((mx + at * motion_tile[0] + 4, py + 36 + motion_tile[1]), label, font=small, fill=WHITE)
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
        if shown is not None:
            # the text the load will be given, whole, on every frame: a look reads it against the frames beside it
            by_ = py + PANEL
            d.line([(0, by_), (canvas_w, by_)], fill=DIM)
            d.text((16, by_ + 8), f"THE TEXT OF LOAD {a.load} (source frames {shown['frames'][0]}-{shown['frames'][1]}), label "
                   f"{shown['text_label']!r}, from {shown['text_from'] or 'nowhere'}", font=text, fill=AMBER)
            if not runs:
                d.text((16, by_ + 64 + more), "NO PLAN OF THIS LOAD IN THE CAPTURE YET: ITS REGION IS NOT DRAWN", font=line, fill=AMBER)
            for k, one_line in enumerate(whose):
                d.text((16, by_ + 64 + more + (26 if not runs else 0) + 24 * k), one_line, font=line, fill=AMBER)
            for k, one_line in enumerate(rendered):
                d.text((16, by_ + 64 + more + 24 * len(whose) + (6 if whose else 0) + 24 * k), one_line, font=line, fill=AMBER)
            for k, one_line in enumerate(written):
                d.text((16, by_ + 38 + 24 * k), one_line, font=line, fill=WHITE if covers else AMBER)
            for k, one_line in enumerate(text_lines):
                d.text((16, by_ + 68 + notice + 24 * k), one_line, font=line, fill=WHITE)
        enc.stdin.write(np.asarray(img).tobytes())
        done = n - lo + 1
        if n % 100 == 0:
            print("frame", n, flush=True)
    enc.stdin.close()
    code = enc.wait()
    print("wrote", out, "frames", done, "encoder exit", code)
    if done != frames or code:
        raise SystemExit(1)
    if shown is not None and shown["recorded"] and not a.out:
        # what this strip showed, for `looked` to copy: a look is held by what was shown, not by a time alone
        row = {"load": a.load, "frames": shown["frames"], "text_label": shown["text_label"], "text_frames": shown["text_frames"],
               "text_by": shown["text_by"], "text_sha256": shown["text_sha256"], "region": shown["region"], "strip": out,
               "made": datetime.datetime.now().isoformat(timespec="seconds")}
        if shown.get("pair"):
            row["pair"] = shown["pair"]               # which label the strip said is whose: a look of a two-subject load is of this too
        (folder / "looks" / f"{a.load}.shown.json").write_text(json.dumps(row, indent=1) + "\n")
        print("wrote", folder / "looks" / f"{a.load}.shown.json")
    elif shown is not None:
        print("no record of what was shown was written: " + ("--out puts the strip outside the capture" if shown["recorded"] else
              "the job file does not say which frames this load's text was written from, or by whom"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("files", help="build the capture folder from saved mask videos and renders")
    f.add_argument("--name", help="the capture goes to <out>/<date>_<name>; not needed with --folder")
    f.add_argument("--folder", help="write the capture into exactly this folder: no date and no name are joined on")
    f.add_argument("--date", default=datetime.date.today().isoformat())
    f.add_argument("--source", required=True, help="the clip every run was loaded from")
    f.add_argument("--first", type=int, required=True, help="the source frame that is the span's frame 0")
    f.add_argument("--frames", type=int, required=True)
    f.add_argument("--mask", action="append", default=[], metavar="LABEL:track=V[,parts=V][,classes=V][,held=V][,shots=J][,shots_at=N][,pose=J][,pose_at=N][,by=RUN][,at=N][,hold=A-B+C-D][,drop=A-B+C-D][,grade=off][,least=PX][,pieces=largest]",
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
    f.add_argument("--made-by", default="", help="who builds this capture, kept in status.json: a look at it (`looked`) is somebody else's")
    f.add_argument("--overwrite", action="store_true", help="rebuild a finished capture folder in place; only when "
                   "no queued render loads its mask videos")
    v = sub.add_parser("video", help="the stacked, frame-numbered video from a capture folder")
    v.add_argument("capture")
    v.add_argument("--source", required=True, help="the clip the capture was made from")
    v.add_argument("--render", help="the render shown in the bottom row; two rows without one")
    v.add_argument("--render-first", type=int, help="the source frame that is the render's frame 0, when the render "
                   "is not one of the capture's runs (a merged file, say); a run's own is in the manifest")
    v.add_argument("--motion", action="append", help="the motion video the run was given, frame 0 its first frame, shown in the panel; "
                   "twice for a load of two subjects, in the list's order")
    v.add_argument("--motion-note", default="", help="words for the panel when there is no motion video file to show")
    v.add_argument("--still", action="append", default=[], metavar="LABEL=IMAGE", help="a reference still and the subject "
                   "it belongs to, shown small in the panel; repeatable")
    v.add_argument("--windows", help="the render frames where each window's new frames start, e.g. 0,345")
    v.add_argument("--all-runs", action="store_true", help="draw every run's region, also when --render is one run's")
    v.add_argument("--load", metavar="NAME", help="the look: one load's own frames, every one, with the text it will be given under "
                   "the picture; written to <capture>/looks/NAME.mp4 with NAME.shown.json beside it. Needs --job")
    v.add_argument("--job", metavar="JOB.json", help="the job file the load and its text are read from")
    v.add_argument("--strip-folder", default="", metavar="NAME", help="with --load: the folder under the video's looks/ the strip goes in, "
                   "in place of the capture's own name; the record of what was shown names the strip there")
    v.add_argument("--out")
    v = sub.add_parser("verify", help=f"does the capture read what the nodes wrote: writes verify.json, exits {VERIFY_FAILED} when not")
    v.add_argument("capture")
    v.add_argument("--no-mask", action="append", default=[], metavar="LABEL:FIRST-LAST=WHY",
                   help="a person's statement: this subject has no mask on these source frames and that is known (walks out of the shot; "
                        "no load is made there). Kept in the capture; an empty frame it does not cover still fails")
    v.add_argument("--by", default="", help="who makes the --no-mask statements")
    g = sub.add_parser("preflight", help="the risks in a capture folder, before a render: writes flags.json")
    g.add_argument("capture")
    g.add_argument("--not-in", action="append", default=[], metavar="LABEL:FIRST-LAST",
                   help="source frames the subject is not in, as the caller knows the clip; repeatable")
    g.add_argument("--text", help="the prompt file of the run this capture is for")
    g.add_argument("--voice-spans", help="voice_spans.json for the clip")
    g.add_argument("--who-is-who", help="the corrections file of bench/who_is_who_across_shots.py for this load; copied into the capture")
    g.add_argument("--gate", action="store_true", help=f"exit {GATE_BLOCKED} when a flag at the top level stands with no override")
    g.add_argument("--own-keep-out", action="append", default=[], metavar="NAME=CLASS+CLASS",
                   help="a keep-out that is the load's subject's own (her hands, clothes, held things), with the classes a restore span names; repeatable")
    g.add_argument("--given-back", action="append", default=[], metavar="NAME:FIRST-LAST=WHY",
                   help="source frames on which an own keep-out is given back to the original (the plan's restore span), and why; repeatable")
    g.add_argument("--named-empty", action="append", default=[], metavar="SUBJECT:FIRST-LAST=WHY",
                   help="source frames the job leaves the subject empty on purpose (its drop= or left_out), with its reason; repeatable")
    ri = sub.add_parser("rules", help="every figure a per-clip rule needs, in one file: writes rules_input.json")
    ri.add_argument("capture")
    ri.add_argument("--not-in", action="append", default=[], metavar="LABEL:FIRST-LAST")
    sh = sub.add_parser("shots", help="the gate's answer per shot and subject: clear, or the rules that block there")
    sh.add_argument("capture")
    sh.add_argument("--json", action="store_true", help="the rows as json, for a tool")
    sh.add_argument("--loads", action="store_true", help="per load in place of per shot: what blocks it, its warnings, kept and held frames, who else is inside")
    lk = sub.add_parser("looked", help="record that somebody who did not make the capture looked through its frames: one line in looks.json")
    lk.add_argument("capture")
    lk.add_argument("--by", required=True, help="who looked")
    lk.add_argument("--loads", metavar="A+B", help="the loads looked at, joined by +; every load a strip was drawn for when not given")
    lk.add_argument("--note", required=True, help="what was looked at (every frame? which picture?) and what was seen")
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
    sz = sub.add_parser("sizes", help="each subject's part size on a capture's frames, against the floor that empties a small part")
    sz.add_argument("capture", nargs="+", help="capture folders")
    sz.add_argument("--json", action="store_true", help="the figures as json, for a tool")
    d = sub.add_parser("diagnose", help="everything the capture says about a stretch marked as wrong")
    d.add_argument("capture")
    d.add_argument("--frames", required=True, metavar="FIRST-LAST", help="source frames")
    a = p.parse_args()
    {"files": files, "sizes": sizes, "video": video, "verify": verify, "preflight": preflight, "rules": rules, "shots": shots, "looked": looked, "outcome": outcome, "diagnose": diagnose, "look": look, "mask": mask, "changed": changed, "mouth": mouth}[a.mode](a)


if __name__ == "__main__":
    main()
