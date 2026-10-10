#!/usr/bin/env python3
"""One delivery file from masked renders and the untouched original: the source's own rate and audio packets,
passes over the same frames merged by what each changed, and a check by decode that every frame is there once.

    <python> bench/assemble_delivery.py --source SOURCE.mp4 --table TABLE.txt --out OUT.mp4 \\
        [--span FIRST-LAST] [--size canvas|source] [--capture data/<date>_<name>]... [--soften SIGMA] [--crf 12]
        [--note "why this build"] [--check-only]
    <python> bench/assemble_delivery.py --compare A.mp4.check.json B.mp4.check.json

**What it buys.** A masked render is a window of a clip, at the lane's canvas and the lane's rate, with a track
that was resampled to fit. What gets watched is the whole stretch: several renders by frame range, the frames
nobody regenerated, the clip's own audio. This writes that file with one encode, says how it was put together,
and proves the parts an eye cannot check: that no frame was dropped, doubled or moved, and that the audio is
the source's own packets.

**The table.** One row per piece (`first-last  piece  the piece's first source frame`, then any of `restore=` and
`subject=`, described below; and `source` lines, which lay nothing and say a shot is left as it is on purpose),
in SOURCE frame numbers (a render made from a frame-for-frame copy of the
source at the lane's rate has the source's numbering):

    # first-last   piece             the source frame that is the piece's frame 0
    14-167         pass_a.mp4        14
    278-437        pass_b.mp4        278

Ranges are inclusive. Every frame of `--span` (default: 0 to the last frame any row names) that no row covers
is the original, fitted to the pieces' canvas as the lane's loader fits it and coded as the lane's writer codes
it (`original_frames`), so a cut from an original range to a piece is not a step in colour or position.

**Rows that share frames** are merged by region, in the table's order; each must be a pass made from the
ORIGINAL, not from another pass's file. A piece's kept pixels equal the fitted original to within its codec
noise, so where `|piece - fitted original|` is over `CHANGE` the piece changed the picture (`changed`). A frame
one row covers is that piece, whole and untouched. A frame several rows cover is the first row's piece, whole,
with each later row's changed region laid over it, grown a little and feathered. Where two pieces changed the
SAME pixel, order alone cannot say whose it is: with a capture folder (`--capture`, what
`bench/capture_masked_run.py` writes) the pieces whose subject holds the pixel are the ones it can go to, and
the latest of those in the table takes it; a pixel no changing piece's subject holds falls to the latest row
that changed it. So a subject's piece and a patch of it are settled by order between themselves, and neither
loses to another person's pass on its own subject. A piece whose subject is not known (no run in any capture
given, and no `subject=` on its row) is never ruled out by a mask it was not asked about: it can take every
pixel it changed. Either way the frames are flagged.

**`subject=<label>`** on a row says whose a piece is when no capture holds it as a run: a patch of a render, made
from that render, has no run of its own. The first delivery built with a patch row and captures (2026-10-10)
gave the patched frames back to the piece they replaced, inside its subject's mask, because the piece's subject
was known and the patch's was not; every proof passed and the shared-pixel figure was what showed it.

**Whose a pixel is.** Two tracked masks can claim one pixel (an arm reaching across somebody), and then a mask
alone does not say whose it is. A capture folder made with more than one subject holds `owners.npz`
(`capture_masked_run.owner_map`: a pixel several tracks claim is the subject's whose class map names it
something, and is marked contested where that does not decide). Where a capture given holds one that covers
the frame and lists the subject, "the subject's pixels" means the pixels that subject OWNS, for the rule above
and for `restore=<subject>`; a contested pixel is nobody's, so in a merge it falls to the later row and a
restore does not give it back. With no owner map the tracked masks are used as they are, and the check record
says which it was (`whose_pixels`).

**`restore=`** on a row gives a class of a subject back to the source: `restore=<subject>.<Class>[+<Class>...]`
after the row's three fields, the class one of the part model's (`sapiens2_parts.CLASS_NAMES`) or its index,
the subject a capture's label whose class map the capture holds (`classes__<by>.npz`). Wherever that class is
on the source frame, grown by `RESTORE_GROW` and feathered, the piece is not laid: the source's own pixels
show there, or an earlier row's. Where an owner map covers the frame, the class is kept off whatever the map gives
to ANOTHER subject: a class map says what a thing is and not whose, and beside another person it names some of
their things too (`class_restores_kept_to_their_subject` in the record counts what was left). Pixels no track
claims, and contested ones, are still given back. It is for a thing that sits inside a region and should not have been redrawn
(an earring inside a face), given back here because asking the sampler to keep it changes what the sampler
draws beside it (measured 2026-10-10 on one render, by another session: the redrawn face went back toward the
original's). The table reports per frame how many changed pixels were given back, and a frame where the
restored pixels sit beside a large change is flagged: that edge is a join between two pictures.
`restore=<subject>` with no class gives back the whole of ANOTHER subject: wherever that subject's tracked mask
is and the piece's own subject's is not (the piece's subject is its run's, from the capture), or, where an owner
map covers the frame, wherever that subject owns the pixel. It is for a pass
on one person whose region took in part of another: whatever it changed of the other person goes back to the
source, and where the two masks both claim a pixel the piece keeps it. `restore=<subject>:whole` takes nothing
out: the piece is laid nowhere inside that subject's mask, for a pass that has no business there whoever is in
front (a face pass and anybody else).

**`--size source`** writes the file at the SOURCE's size and not the canvas's. Every frame is the source's own
picture, never scaled, and only what a piece changed is put back over it, scaled up from the canvas to the
place the loader's crop took it from; the rows or columns that crop dropped stay the source's. Every row is
laid by region there, the first as well. Measured 2026-10-10 on two fixed-camera renders of one clip: scaled
up by 1.39 the regenerated region read about as sharp and as busy as the source's own pixels beside it, where
at the canvas's size it read crisper; not judged on playback. The source must be BT.709, as the pieces are.

**Flags** (in the check json, and in each capture folder given), each with frames and a figure, to be looked at
and never a refusal (what a restore gave back is not counted against a piece): a piece that changes the picture
on frames where its subject has no tracked mask at all
(a pass that redrew somebody else; one was caught this way on 2026-10-10, past a cut the tracker had matched
across wrongly); a piece that changes pixels further from its subject's tracked mask than its run's margin and
a token (the new subject reaching further than the old one stood, or something else redrawn: the box says
where); a piece whose changed area steps up or down between one frame and the next against the frames either
side (`JUMP` over `STEP_FRAMES`), which needs no capture and is what catches a tracker that followed the wrong
person past a cut, since the mask it left then says the subject is there; it names the frame of the step, and a
change of framing steps too; a piece that changes a frame or two just across a cut of the source and no further
(`CUT`, `SPILL`), which is a pass whose region ran over the cut and redrew the next shot for a moment, found from
the source's own frame-to-frame change with no capture and no shot table (measured 2026-10-10: one whole-person
render changed a fifth of the frame on five such frames, and nothing else here said so). For each such cut the
flag says whether the cut falls inside a latent step of the piece's load (`loop_plan.step_span`, counted from the
piece's first frame) and whether that step holds the spilled frames: if it does, it is the known way a region is
carried over a cut; a spill at a cut on a step's edge is something else; two pieces changing the same
pixels, with the box and how many were settled by a mask and how many by
order; a piece that changes nothing on frames a row gives it; restored pixels beside a large change, and a restore with
no mask or class map on some frames; and, at the source's size, a piece whose change reaches
the edge the loader's crop cut at, beyond which only the source's picture exists. The first two need a capture; without one they
are not checked and the json says so.

**What is written.** Pieces are decoded to their own YUV and never pass through RGB; the frames go down one
pipe into one x264 encode, stamped by number at the source's rate, so no timestamp is ever rescaled. The file
is BT.709 and says so in all four fields. A piece written before the lane's writer converted and tagged
(`loop_output.TO_BT709`; such a file carries no matrix tag and holds BT.601 values) is converted on the way in.

**Audio.** The source's packets, copied. A copied stream can only start on a packet, and a decoder needs the
packet before the one it starts on. Cut straight from the source, ffmpeg starts the track at the picture's
keyframe before the span and hides the head behind an edit list (measured 2026-10-10 on one clip: 0.05 s for
one span, 0.31 s for another); a player that ignores edit lists then plays the track that much late. So for a
span that does not start at frame 0 the track is first copied alone (`track_only`), where a cut lands on an
audio packet, and cut one packet before the span: the head is one to two packets, skipped by the edit list.

**`--soften SIGMA`** (off unless given) blurs the luma by that many pixels where a piece changed the picture,
feathered at the edge, before the one encode. Provenance: measured 2026-10-10 on two fixed-camera renders of
one clip, where the regenerated region's edges were crisper than the plate's at the canvas's size and a sigma
of 0.5 to 0.7 matched them; not judged on playback.

**Every row shows where it was meant to** (`rows_shown`, a proof: the build fails on it). A row is meant to show
on every pixel its piece changed, less what a `restore=` on it gives back, what a later row takes, and what an
earlier row's subject holds when that is another, known subject and not the row's own. A row that shows on less
than `SHOWN` of that has an earlier row laid over it, and the build says which row. The frame-for-frame proofs
below cannot see this: a file can be exactly the frames that were fed and the frames fed can be the wrong
row's.

**Shots where a named subject is the source's own** (`shots` in the record; a proof). A file can pass every
proof above and still show a person untouched for a whole shot: no row laid a pixel there, so there was nothing
for a proof to be wrong about. The first locked file did, for both its subjects, on a shot both trackers had
called their subject absent on; a viewer found it. So the span is cut into shots (the source's own cuts, `CUT`,
read from the fitted source on every frame of the span) and, for every subject a row names (its run's subject,
or `subject=`), each shot says on how many frames anything of that subject's was laid, on how many it is
tracked, and what the captures say of people there (a shot table's detections on the shot's shown frame, and the
capture's own flag `absent_with_people_on_screen`). A shot with nothing laid for a named subject is listed under
its own heading with one reading:

- *tracked, nothing laid*: the subject's mask is there and no row covers it. **Fails.**
- *not tracked, people detected*: the tracker called the subject absent and people are on screen who are not
  accounted for by the other named subjects laid or tracked on that shot. **Fails.**
- *not tracked, the people detected are other subjects'*, *not tracked, nobody detected*, *not known* (no shot
  table and no flag covers the shot): listed and counted, and do not fail, or every table that covers part of a
  clip would.
- *by intent*: the table says so, one line a shot: `source  first-last  subject=<label>  words saying why`. This
  is where the question is answered; the words are kept in the record. A `source` line over a shot where
  something IS laid for that subject, or one that covers no whole shot, fails as a contradiction.

Inside a shot that has rows laid, the runs of frames where the subject is tracked and nothing is laid are listed
with the longest (said, not failed: a face pass lays nothing while a face is turned away, and a second of that is
not a skipped frame). The record's `verdict_line` carries the count, and the frames when any fail. What it cannot
know is in `shots.missing`: a people count is one frame a shot; a subject with no shot table borrows another
subject's count for the shot; a dissolve is not a cut; a row whose subject is not known is not asked about.

**A locked file is never written over.** A folder's `LOCKED.md` lists the files the owner has accepted, one to a
list line: the name in backticks, then `md5 <sum>`. A build whose `--out` is one of them is refused before anything
is read, with the names that are free; a later version of the same span is a new file beside it. `--check-only`
still reads a locked file (it writes the record and nothing else) and fails if the file is no longer the one that
was locked. The lock is in the tool because a rule that every session has to remember is not a lock.

**The check** (`<out>.check.json`; exit 1 when it fails; it records the table's rows, the captures given, when it ran
and `--note`). Count and timestamps: as many frames as the span,
frame n stamped n frames in at the source's rate. Order: the file is decoded and each frame compared with the
frame fed for its place and the ones fed either side; it must be nearest its own, and no `BLOCK`-pixel square of
it may sit more than `APART` levels from the same square of the frame fed, so a file made from another table or
by another rule is not passed for being in order: a whole-frame mean passes a file with the wrong row over a
region of it. Frames whose
neighbours are the same picture cannot be placed by content and are counted apart. Audio: the file's
packets are a run of the source's packets, byte for byte, the head is under `HEAD_PACKETS` packets, and the
decoded samples fit the source's at their own place better than a sample or more either side (`SHIFTS`).
Colour: the four fields, and the decoded file's planes have no bias against the frames fed (`BIAS`), which is
what a silent matrix conversion would leave. Regions: per piece and frame, the pixels it changed.

**What the file did to each subject** (`subjects` in the record, when captures are given). The delivered file is
read back at the canvas by the loader's fit and set against the fitted source, on the frames a piece covers: for
every subject a capture knows, the pixels of its tracked mask (and of what it owns, with an owner map) more than
`OFF` levels from the source, in all and by class of its class map, inside its track and outside it; and the
pixels more than one track claims, how many an owner map settled and how many it left contested. A pass on one
person should leave the others at the floor, which is reported beside them (`floor`: the same share on pixels no
track claims and no piece changed). `--compare` prints two records side by side: the same stretch with and
without a restore is the before and after.

**What it does not show.** Whether the result looks right. That a piece's changed region is the region the
render kept: the node draws what it kept and saves no mask of it, so this rebuilds it from a threshold, and a
change under `CHANGE` further than `GROW` from stronger change is lost to a later row of a merge
(`data/CAPTURE_GAPS.md`). `CHANGE` was measured on footage: on a hard test pattern that moves fast the writer's
own codec noise reached twice it, so on such a picture a kept pixel can read as changed and a flag can be noise. `bench/check_assemble_delivery.py` holds the cases.

**Which proof reads what, and what to read before a file is handed over.** Two kinds of proof are here and they
are not the same thing. `rows_shown`, the flags and the regions are read from the frames this tool MAKES from the
table: they say the rule did what the table asked. The count, the order, the squares (`squares_apart`), the colour
and the audio are read from the FILE: they say the file is those frames. A wrong rule passes the second kind and
a file built another way passes the first, so a file goes to the owner only when the record shows all of: the
verdict passes; every row of `rows_shown` at a share of 1; `squares_apart` 0; `pieces_with_no_capture` empty, or
`subject=` on each such row. Then the flags are read and each given an outcome, and the shared-pixel counts
(`settled_by_a_mask_px`, `settled_by_order_px`) are set against what was expected of the table BEFORE the build:
a count nobody predicted is how the first wrong file was caught, with every proof passing. Build to a name the
owner has not been given, read the record, then copy and run `--check-only` on the placed file.

**What the lock is not.** It binds this tool: a copy, a move or an encoder writes over a locked name unhindered.
It is per folder and by file name, and only a list line of `LOCKED.md` with the name in backticks counts. The
tool never writes `LOCKED.md`; whoever has the owner's word does.

The files built with this tool, their figures and what changed in it since, by date:
`bench/results/2026-10-10_delivery_files_and_their_records.md`. A change to a proof is a dated line added there.

Nothing here describes what a clip shows: a clip is a file name and a subject is a capture's label.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

FFMPEG = "ffmpeg"
BATCH = 24        # frames converted at a time; reasoned: bounded memory on the loader's 16-bit frames
STATIC = 0.15     # mean |difference| under which two fed frames are the same picture; reasoned: under the codec
                  # noise the check itself reports for a frame against its own
CHANGE = 6.0      # levels; measured 2026-10-10 on three renders of one clip (53 frames): over kept pixels the
                  # averaged difference never passed 5.8 and its 99.9th percentile stayed under 3.4. The node's own
                  # rule keeps a change over 12.75 (`video_mask.CHANGE_THRESHOLD` of full scale)
CHANGE_MEAN = 5   # pixels across the difference is averaged over; reasoned: tames single-pixel codec noise
GROW = 8          # pixels a later piece's region is grown by; reasoned: costs nothing, both sides are equal there
FEATHER = 3.0     # sigma of the edge's blur, pixels; reasoned: well inside GROW, so real change keeps full weight
SPECK = 64        # changed pixels away from the subject under which nothing is flagged; reasoned: a blob the
                  # opening in `changed` lets through is a few pixels, a redrawn feature is hundreds
JUMP = 2.5        # times the changed area must step by, between the frames before and the frames after, to be
                  # flagged; measured 2026-10-10 on four renders of one clip: within a subject's shots the largest
                  # frame was 1.1 to 1.7 times the median, and the pass that redrew the wrong person past a cut
                  # was 3.3 to 3.7 times. As one median for a whole piece it lit 110 frames of a render whose
                  # framing gets closer partway, so it is a step between neighbours now
STEP_FRAMES = 12  # frames either side the step is read over; reasoned: half a second, longer than a blink of the
                  # region and shorter than the shortest shot met so far
CUT = 40.0        # levels the fitted source must move from one frame to the next to be taken for a cut; measured
                  # 2026-10-10 on one clip: 61 to 69 at the cuts read, 2 to 18 inside shots
SPILL = 3         # frames: a run of changed frames this short on one side of a cut, joined to changed frames on the
                  # other, is a spill; reasoned: the spills seen were one and two frames, and a piece that goes on
                  # for longer past a cut is following its subject into the next shot
TOKEN = 32        # pixels; inherited: a token is two latent cells of 16 a side (`bench/capture_masked_run.py::CELL`),
                  # and a region widened to whole tokens reaches that far past its margin
MARGIN = 64       # pixels, when a capture's run names no margin; inherited: the Masked Source's default grow
RESTORE_GROW = 2  # pixels a restored class is grown by; reasoned: a class map's edge is a pixel or two inside the thing
RESTORE_FEATHER = 1.0   # sigma of that edge's blur; reasoned: the thing is small, a wide feather would thin it
JOIN = 25.5       # levels of averaged difference that count as a large change beside restored pixels; inherited:
                  # twice the node's own line for a kept change (`video_mask.CHANGE_THRESHOLD` of full scale)
JOIN_PX = 16      # pixels of such an edge under which nothing is flagged; reasoned: a few pixels is the codec
OFF = 12.0        # levels of luma a delivered pixel may sit from the source and still count as the source's, in the
                  # `subjects` table; reasoned: twice the change threshold, the line the 2026-10-10 figures were read at
BLOCK = 16        # pixels: the side of the squares a decoded frame is compared with the frame fed for it in
APART = 8.0       # levels a square's mean |difference| in luma may reach; measured 2026-10-10 on one full-size build at
                  # crf 12: 2.27 the largest of a file's own, 141.7 where a patch of it was laid under the piece it
                  # replaces (9,731 squares over, all on the patched frames). A whole-frame mean read 0.42 and 0.49
SHOWN = 0.98      # of the pixels a row was meant to show on, the share it must show on; reasoned: the rule is exact,
                  # so anything under all of them is a row laid over; the slack is for nothing but a later change
                  # to how a restore's edge is counted
HEAD_PACKETS = 2  # audio packets the track may begin before the span; reasoned: the one the decoder needs, and
                  # the one the span's first sample is in
# What the lane's writer does to piped rgb, and what its files say. Inherited: `loop_output.TO_BT709`,
# `SAY_BT709`, `BT709_TAGS`; the check builds a piece with the writer itself and fails if these drift from it.
TO_BT709 = "scale=out_color_matrix=bt709"
SAY_BT709 = "setparams=color_primaries=bt709:color_trc=bt709"
BT709_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
# What the frames piped to the one encode are. Without it ffmpeg takes unlabelled frames for another matrix and
# converts them on the way to a BT.709 file: measured 2026-10-10, a mean difference of 2.3 levels from the frames
# fed against 0.45 with the label, and a file that still passed every other proof.
PIPED_IS_BT709 = "setparams=colorspace=bt709:range=tv:color_primaries=bt709:color_trc=bt709"
BIAS = 0.5        # levels the decoded file's Y, U or V may sit from the frames fed, on average; reasoned: codec noise
                  # has no sign (measured under 0.1 with the label above), a wrong matrix moves a plane by levels
SHIFTS = 48       # samples either way the decoded audio is tried against the source's; reasoned: wider than a
                  # packet-boundary slip seen here (39 samples), narrower than half a period of any tone under 450 Hz
SAYS = ["tv", "bt709", "bt709", "bt709"]          # range, matrix, transfer, primaries, as the check reads them
LEVELS = ("likely fine", "iffy", "likely to fail")  # inherited: the capture tool's words for a flag's weight


def sh(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, **kw)
    if proc.returncode != 0:
        raise SystemExit(f"failed: {' '.join(str(c) for c in cmd[:12])} ...\n{proc.stderr.decode(errors='replace')[-600:]}")
    return proc.stdout


def probe(path) -> dict:
    out = sh(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
              "stream=width,height,pix_fmt,r_frame_rate,time_base,nb_read_packets,color_space,color_range,"
              "color_transfer,color_primaries", "-of", "json", str(path)])
    s = json.loads(out)["streams"][0]
    num, den = (int(v) for v in s["r_frame_rate"].split("/"))
    return {"width": s["width"], "height": s["height"], "pix_fmt": s["pix_fmt"], "rate": (num, den),
            "time_base": s["time_base"], "frames": int(s["nb_read_packets"]), "matrix": s.get("color_space", "unknown"),
            "says": [s.get(k, "unknown") for k in ("color_range", "color_space", "color_transfer", "color_primaries")]}


def audio_rate(path) -> int | None:
    out = sh(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(path)])
    return int(out.decode().strip().split(",")[0]) if out.strip() else None


def class_names() -> tuple[str, ...]:
    """The part model's class names, read from `sapiens2_parts.py` beside this folder without importing it (it
    imports the models' libraries); a class's index in a class map is its place in this table."""
    import ast
    tree = ast.parse((Path(__file__).resolve().parents[1] / "sapiens2_parts.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CLASS_NAMES" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise SystemExit("sapiens2_parts.py has no CLASS_NAMES")


def read_table(path) -> list[dict]:
    rows = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line or line.split()[0] == "source":      # a `source` line lays nothing: `read_intents`
            continue
        span, piece, first, *more = line.split()
        lo, hi = (int(v) for v in span.split("-"))
        row = {"first": lo, "last": hi, "piece": piece, "piece_first": int(first), "restore": []}
        for token in more:
            key, _, value = token.partition("=")
            if key == "subject" and value:
                row["subject"] = value                   # whose the piece is, when no capture holds it as a run
                continue
            if key != "restore" or not value:
                raise SystemExit(f"{path}: `{token}` is not `subject=<label>`, `restore=<subject>` or `restore=<subject>.<Class>[+<Class>...]`")
            label, dot, classes = value.partition(".")
            if not dot:
                label, colon, how = label.partition(":")
                if colon and how != "whole":
                    raise SystemExit(f"{path}: `{token}`: after the colon only `whole` is known")
                # the subject's tracked mask: less the piece's own subject (None), or with nothing taken out ("whole")
                row["restore"].append((label, "whole" if colon else None))
                continue
            names = class_names()
            wanted = []
            for c in classes.split("+"):
                if c not in names and not (c.isdigit() and int(c) < len(names)):
                    raise SystemExit(f"{path}: `{c}` is not a class of the part model ({', '.join(names)})")
                wanted.append(int(c) if c.isdigit() else names.index(c))
            row["restore"].append((label, wanted))
        rows.append(row)
    return rows                      # in the file's order: where rows share frames, order is the last resort


def read_intents(path) -> list[dict]:
    """The table's `source  first-last  subject=<label>  words` lines: shots left as the source's own on purpose."""
    out = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line or line.split()[0] != "source":
            continue
        _, *rest = line.split()
        try:
            lo, hi = (int(v) for v in rest[0].split("-"))
            key, _, label = rest[1].partition("=")
            if key != "subject" or not label or lo > hi:
                raise ValueError
        except (IndexError, ValueError):
            raise SystemExit(f"{path}: `{line}` is not `source first-last subject=<label> words saying why`") from None
        out.append({"first": lo, "last": hi, "subject": label, "why": " ".join(rest[2:])})
    return out


def run_for(row, captures):
    """(folder, manifest, run) for a row's piece: the capture's run whose render it is, or, for a row that says
    whose it is (`subject=`), a stand-in with that subject and no folder; None when neither."""
    found = captures.run_of(row["piece"]) if captures else None
    if found is None and captures and row.get("subject"):
        return None, None, {"name": None, "subject": row["subject"], "margin_px": None}
    return found


def restore_text(label, wanted) -> str:
    if wanted is None or wanted == "whole":
        return label + (":whole" if wanted else "")
    return f"{label}." + "+".join(class_names()[c] for c in wanted)


def segments(rows, span):
    """The span as (first, last, [rows covering it, in table order]); an empty list is the original."""
    marks = {span[0], span[1] + 1}
    for r in rows:
        if r["last"] >= span[0] and r["first"] <= span[1]:
            marks |= {max(r["first"], span[0]), min(r["last"], span[1]) + 1}
    marks = sorted(marks)
    return [(a, b - 1, [r for r in rows if r["first"] <= a and r["last"] >= b - 1]) for a, b in zip(marks, marks[1:])]


def frame_spans(frames) -> list[list[int]]:
    out: list[list[int]] = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def pipe_frames(cmd, size):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=size)
    assert proc.stdout is not None and proc.stderr is not None
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield buf
    finally:
        proc.stdout.close()
        err = proc.stderr.read()
        proc.stderr.close()
        if proc.wait() not in (0, -13) and err:
            sys.stderr.write(err.decode(errors="replace")[-400:])


def loader_fit(w, h):
    # VHS_LoadVideoFFmpeg's own filter for a custom width and height (its `ffmpeg_frame_generator`, read
    # 2026-10-10): a centre crop to the canvas's shape, then one scale. Inherited, not chosen.
    ar = float(w) / float(h)
    return f"crop=if(gt({ar}\\,a)\\,iw\\,ih*{ar}):if(gt({ar}\\,a)\\,iw/{ar}\\,ih),scale={w}:{h}"


def original_frames(source, first, last, w, h):
    """Source frames first..last as the yuv420p a piece holds where it kept the source.

    The path of a kept pixel, step for step: the loader's ffmpeg command to rgba64le, its division by 65535,
    the writer's x255 and round to 8 bits, the writer's rgb24 pipe into ffmpeg and its conversion. Measured
    2026-10-10 against a render's kept pixels (the writer of that day, which left the matrix to ffmpeg): no
    bias in Y, U or V, a mean difference at the render's own codec noise. One-command stand-ins were off by a
    third of a level in Y, because ffmpeg's 8-bit rgb path reads darker than its 16-bit one.
    """
    many = last - first + 1
    vf = f"select='between(n\\,{first}\\,{last})'," + loader_fit(w, h)
    dec = [FFMPEG, "-v", "error", "-an", "-i", str(source), "-pix_fmt", "rgba64le", "-vf", vf, "-fps_mode", "passthrough",
           "-frames:v", str(many), "-f", "rawvideo", "-"]
    batch, got = [], 0

    def flush():
        raw = np.frombuffer(b"".join(batch), np.dtype(np.uint16).newbyteorder("<")).reshape(len(batch), h, w, 4)
        rgb = np.round(np.clip((raw[..., :3] / (2 ** 16 - 1)).astype(np.float32), 0, 1) * 255.0).astype(np.uint8)
        yuv = sh([FFMPEG, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                  "-vf", TO_BT709, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"], input=rgb.tobytes())
        per = w * h * 3 // 2
        assert len(yuv) == per * len(batch), "the rgb to yuv step lost frames"
        return [yuv[i * per:(i + 1) * per] for i in range(len(batch))]

    for buf in pipe_frames(dec, w * h * 8):
        batch.append(buf)
        if len(batch) == BATCH:
            for f in flush():
                got += 1
                yield f
            batch = []
    if batch:
        for f in flush():
            got += 1
            yield f
    if got != many:
        raise SystemExit(f"the source gave {got} of {many} frames for {first}-{last}")


def piece_frames(row, first, last, w, h):
    """A piece's frames for source frames first..last, as BT.709 yuv420p bytes."""
    a, b = first - row["piece_first"], last - row["piece_first"]
    # a file with no matrix tag is the writer's older form: BT.601 values
    vf = f"select='between(n\\,{a}\\,{b})'" + ("" if row.get("matrix") == "bt709" else ",colormatrix=bt601:bt709")
    cmd = [FFMPEG, "-v", "error", "-an", "-i", row["piece"], "-vf", vf, "-fps_mode", "passthrough",
           "-frames:v", str(b - a + 1), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    got = 0
    for buf in pipe_frames(cmd, w * h * 3 // 2):
        got += 1
        yield buf
    if got != b - a + 1:
        raise SystemExit(f"{row['piece']} gave {got} of {b - a + 1} frames for {first}-{last}")


def planes(buf, w, h):
    a = np.frombuffer(buf, np.uint8).astype(np.float32)
    return [a[:w * h].reshape(h, w), a[w * h:w * h * 5 // 4].reshape(h // 2, w // 2), a[w * h * 5 // 4:].reshape(h // 2, w // 2)]


def difference(piece, orig):
    """How far a piece is from the original at each pixel of the canvas: the largest of Y, U and V, averaged."""
    h, w = piece[0].shape
    chroma = np.maximum(np.abs(piece[1] - orig[1]), np.abs(piece[2] - orig[2]))
    diff = np.maximum(np.abs(piece[0] - orig[0]), cv2.resize(chroma, (w, h), interpolation=cv2.INTER_NEAREST))
    return cv2.blur(diff, (CHANGE_MEAN, CHANGE_MEAN))


def changed(piece, orig, diff=None):
    """Where a piece is not the original: a bool mask at the canvas's size."""
    hard = ((difference(piece, orig) if diff is None else diff) > CHANGE).astype(np.uint8)
    return cv2.morphologyEx(hard, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)    # a lone speck is noise


def box_of(mask) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None


def disc(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


class Captures:
    """The capture folders given: which run made a piece, and each subject's tracked mask by source frame."""

    def __init__(self, folders, canvas):
        self.folders, self.masks, self.used_owner_map = [], {}, set()
        self.class_px_left = {}       # (row's piece, subject, frame) -> class pixels a restore left because another owns them
        for folder in folders:
            folder = Path(folder)
            m = json.loads((folder / "manifest.json").read_text())
            if tuple(m["size"]) != tuple(canvas):
                raise SystemExit(f"{folder} was captured at {m['size']}, the pieces are {list(canvas)}")
            self.folders.append((folder, m))

    def run_of(self, piece):
        """(folder, manifest, run) of the run whose render is this piece, or None."""
        for folder, m in self.folders:
            for run in m["runs"]:
                if run.get("render") and os.path.basename(run["render"]) == os.path.basename(piece):
                    return folder, m, run
        return None

    def said_absent_with_people(self) -> list[dict]:
        """Every capture's own flag that a subject was called absent while people were detected, in source frames."""
        out = []
        for folder, _m in self.folders:
            path = folder / "flags.json"
            if not path.is_file():
                continue
            held = json.loads(path.read_text())
            for flag in (held if isinstance(held, list) else held.get("flags", [])):
                if flag.get("rule") == PEOPLE_FLAG and flag.get("subject"):
                    out.append({"subject": flag["subject"], "spans": [list(x) for x in flag.get("source_frames", [])],
                                "people": (flag.get("figures") or {}).get("people"), "figures": flag.get("figures") or {},
                                "capture": folder.name, "id": flag.get("id")})
        return out

    def shot_tables(self) -> list[dict]:
        """Every shot table the captures hold: whose, the capture, and per shot its source frames, how many people
        the tracker's detector found on the frame its tile shows, and the tracker's word for its subject there."""
        out = []
        for folder, m in self.folders:
            for s in m["subjects"]:
                sight = s["sightings"][0]
                path = folder / "subjects" / s["label"] / f"shots__{sight['by']}.json"
                if not path.is_file():
                    continue
                at = int(sight.get("shots_first_source_frame", sight.get("first_source_frame", m["first_frame"])))
                table = json.loads(path.read_text())
                out.append({"subject": s["label"], "capture": folder.name, "cuts": [at + int(c) for c in table.get("cuts", [])],
                            "shots": [{"frames": [at + int(x["first_frame"]), at + int(x["last_frame"])], "people": len(x.get("people") or []),
                                       "state": (x.get("subject") or {}).get("state")} for x in table.get("shots", [])]})
        return out

    def owned(self, label, frame, prefer=None):
        """The pixels a subject OWNS on a source frame by a capture's `owners.npz`, or None where no capture given
        holds an owner map that lists the subject and covers the frame."""
        for folder, m in sorted(self.folders, key=lambda fm: fm[0] != prefer):
            k = frame - m["first_frame"]
            if not 0 <= k < m["frames"] or not (folder / "owners.npz").is_file():
                continue
            key = (folder, "owners")
            if key not in self.masks:
                z = np.load(folder / "owners.npz")
                self.masks[key] = (z["owner"], [str(x) for x in z["labels"]], int(z["nobody"]))
            owner, labels, _nobody = self.masks[key]
            if label in labels:
                self.used_owner_map.add(str(folder))
                return owner[k] == labels.index(label)
        return None

    def owned_by_another(self, label, frame, prefer=None):
        """The pixels an owner map gives to a subject OTHER than `label` on a source frame (not the contested ones,
        not nobody's), or None where no capture given holds an owner map that lists the subject and covers the frame."""
        for folder, m in sorted(self.folders, key=lambda fm: fm[0] != prefer):
            k = frame - m["first_frame"]
            if not 0 <= k < m["frames"] or not (folder / "owners.npz").is_file():
                continue
            if self.owned(label, frame, prefer=folder) is None:
                continue
            owner, labels, _nobody = self.masks[(folder, "owners")]
            return (owner[k] < len(labels)) & (owner[k] != labels.index(label))
        return None

    def classes(self, label, frame, prefer=None):
        """The part model's class map of a subject on a source frame (class indices, 0 off the subject), or None."""
        for folder, m in sorted(self.folders, key=lambda fm: fm[0] != prefer):
            k = frame - m["first_frame"]
            if not 0 <= k < m["frames"]:
                continue
            for s in m["subjects"]:
                path = folder / "subjects" / label / f"classes__{s['sightings'][0]['by']}.npz"
                if s["label"] != label or not path.is_file():
                    continue
                key = (folder, label, "classes")
                if key not in self.masks:
                    self.masks[key] = np.load(path)["classes"]
                return self.masks[key][k]
        return None

    def mask(self, label, frame, prefer=None):
        """The subject's tracked mask on a source frame, or None where no capture covers it."""
        order = sorted(self.folders, key=lambda fm: fm[0] != prefer)
        for folder, m in order:
            k = frame - m["first_frame"]
            if not 0 <= k < m["frames"]:
                continue
            for s in m["subjects"]:
                if s["label"] != label:
                    continue
                key = (folder, label)
                if key not in self.masks:
                    z = np.load(folder / "subjects" / label / f"masks__{s['sightings'][0]['by']}.npz")
                    self.masks[key] = (z["track"], z["covered"])
                track, covered = self.masks[key]
                if covered[k]:
                    return np.unpackbits(track[k], axis=-1)[:, :m["size"][0]].astype(bool)
        return None


def owners(hard, subject_masks):
    """Which piece each changed pixel is taken from (-1: none changed it), and how the shared pixels were settled.
    `hard[i]` is `changed` of piece i; `subject_masks[i]` is what its subject holds (its tracked mask, or what an
    owner map gives it), or None when the piece's subject is not known.

    Among the pieces that changed a pixel, those whose subject holds it can take it, and the latest of them does;
    a piece with no known subject can take anything it changed. A pixel none of them can take goes to the latest
    piece that changed it."""
    h, w = hard[0].shape
    owner = np.full((h, w), -1, np.int8)
    for i, m in enumerate(hard):
        owner[m] = i                                   # the last resort: the later row
    shared = np.sum(hard, axis=0) > 1
    by_mask = 0
    if shared.any():
        winner = np.full((h, w), -1, np.int8)
        ruled_out = np.zeros((h, w), bool)
        for i, (m, s) in enumerate(zip(hard, subject_masks)):
            can = m & shared if s is None else m & shared & s
            winner[can] = i                            # in table order, so the latest that can take it
            if s is not None:
                ruled_out |= m & shared & ~s
        decided = winner >= 0
        owner[decided] = winner[decided]
        by_mask = int((decided & ruled_out).sum())     # a mask kept some piece that changed the pixel from taking it
    return owner, {"px": int(shared.sum()), "box": box_of(shared), "by_mask": by_mask, "by_order": int(shared.sum()) - by_mask}


def weight_of(owner, i):
    """Piece i's weight over the canvas: one where it owns the pixel, feathered into pixels nobody changed."""
    mine = cv2.dilate((owner == i).astype(np.uint8), disc(GROW)).astype(bool) & ((owner == i) | (owner == -1))
    return cv2.GaussianBlur(mine.astype(np.float32), (0, 0), FEATHER)


def to_bytes(planes_):
    return b"".join(np.clip(np.round(x), 0, 255).astype(np.uint8).tobytes() for x in planes_)


def mask_of_anyone(captures, frame, prefer, shape):
    """Every pixel some owner map gives to a subject or marks contested on this frame: where a restore's grown
    margin must not reach, because the pixel is somebody's or in dispute."""
    for folder, m in sorted(captures.folders, key=lambda fm: fm[0] != prefer):
        k = frame - m["first_frame"]
        if 0 <= k < m["frames"] and (folder, "owners") in captures.masks:
            owner, _labels, nobody = captures.masks[(folder, "owners")]
            return owner[k] != nobody
    return np.zeros(shape, bool)


def restore_weight(row, captures, run, frame, shape):
    """(weight in 0..1 over the canvas where the row's piece is not to be laid, the hard mask of what is given
    back), or (None, None) when the row asks for nothing; the mask is None on a frame no capture covers."""
    if not row.get("restore"):
        return None, None
    hard, grown = np.zeros(shape, bool), np.zeros(shape, bool)
    prefer = run[0] if run else None
    for label, wanted in row["restore"]:
        if wanted is None or wanted == "whole":
            # another subject's tracked mask; less the piece's own unless the row says whole, so that where both
            # masks claim a pixel the piece keeps it
            mask = captures.mask(label, frame, prefer=prefer) if captures else None
            if mask is None:
                return np.zeros(shape, np.float32), None
            owns = captures.owned(label, frame, prefer=prefer) if wanted is None else None
            if owns is not None:
                # an owner map covers this frame: what the subject owns, which already leaves out what the piece's
                # own subject owns and what is contested
                hard |= owns
                grown |= cv2.dilate(owns.astype(np.uint8), disc(RESTORE_GROW)).astype(bool) & (owns | ~mask_of_anyone(captures, frame, prefer, shape))
                continue
            own = (captures.mask(run[2]["subject"], frame, prefer=prefer)
                   if wanted is None and run and run[2]["subject"] != label else None)
            mask = mask & ~own if own is not None else mask
            hard |= mask
            wide = cv2.dilate(mask.astype(np.uint8), disc(RESTORE_GROW)).astype(bool)
            grown |= wide & ~own if own is not None else wide
            continue
        cmap = captures.classes(label, frame, prefer=prefer) if captures else None
        if cmap is None:
            return np.zeros(shape, np.float32), None
        mask = np.isin(cmap, wanted)
        # a class map says what a thing is, not whose: it is cut to its subject's track and margin, and beside
        # another person it calls some of their things by the class too. Where an owner map covers the frame, what
        # it gives to ANOTHER subject is not this subject's to give back; pixels no track claims, and contested
        # ones, stay (measured 2026-10-10 on one two-person shot: a quarter of one subject's upper clothing class
        # lay on what the other owned)
        theirs = captures.owned_by_another(label, frame, prefer=prefer)
        if theirs is not None:
            captures.class_px_left[(row["piece"], label, frame)] = int((mask & theirs).sum())
            mask = mask & ~theirs
        hard |= mask
        wide = cv2.dilate(mask.astype(np.uint8), disc(RESTORE_GROW)).astype(bool)
        grown |= wide & ~theirs if theirs is not None else wide
    return cv2.GaussianBlur(grown.astype(np.float32), (0, 0), RESTORE_FEATHER), hard


def merged(pieces, owner, held, orig):
    """At the canvas's size: the first piece whole, each later piece's changed region over it. `held[i]` is the
    weight where piece i is not to be laid (a restore), or None."""
    h, w = owner.shape
    out = [p.copy() for p in pieces[0]]
    for i in range(len(pieces)):
        if i == 0 and held[0] is None:
            continue
        # the first piece is the base, whole; what it must not show is the original's. A later piece is its own
        # region, less what it must not show, and under that is whatever was laid before it
        weight = held[0] if i == 0 else weight_of(owner, i) * (1.0 - (held[i] if held[i] is not None else 0.0))
        half = cv2.resize(weight, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        over = orig if i == 0 else pieces[i]
        for k, wgt in enumerate((weight, half, half)):
            out[k] += (over[k] - out[k]) * wgt
    return to_bytes(out)


def crop_of(sw, sh, w, h):
    """The box of the source the loader's filter keeps for a w x h canvas: (width, height, x, y), as ffmpeg's crop
    rounds them for 4:2:0 (down to even, centred)."""
    ar = float(w) / float(h)
    cw, ch = (sw, sw / ar) if ar > sw / sh else (sh * ar, sh)
    cw, ch = int(cw) & ~1, int(ch) & ~1
    return cw, ch, ((sw - cw) // 2) & ~1, ((sh - ch) // 2) & ~1


def laid_over(base, pieces, owner, box, held=None):
    """At the source's size: the source's own planes with every piece's changed region scaled up and laid over,
    less what `held[i]` says piece i is not to be laid on."""
    cw, ch, x0, y0 = box
    out = [p.copy() for p in base]
    for i, piece in enumerate(pieces):
        if not (owner == i).any():
            continue
        small = weight_of(owner, i) * (1.0 - (held[i] if held is not None and held[i] is not None else 0.0))
        weight = cv2.resize(small, (cw, ch), interpolation=cv2.INTER_LINEAR)
        half = cv2.resize(weight, (cw // 2, ch // 2), interpolation=cv2.INTER_AREA)
        for k, (wgt, (pw, ph, px, py)) in enumerate(((weight, (cw, ch, x0, y0)), (half, (cw // 2, ch // 2, x0 // 2, y0 // 2)),
                                                     (half, (cw // 2, ch // 2, x0 // 2, y0 // 2)))):
            up = cv2.resize(piece[k], (pw, ph), interpolation=cv2.INTER_CUBIC)
            view = out[k][py:py + ph, px:px + pw]
            view += (up - view) * wgt
    return to_bytes(out)


def source_frames(source, first, last, sw, sh):
    """The source's own frames first..last, as they decode: nothing scaled, nothing converted."""
    many = last - first + 1
    cmd = [FFMPEG, "-v", "error", "-an", "-i", str(source), "-vf", f"select='between(n\\,{first}\\,{last})'", "-fps_mode",
           "passthrough", "-frames:v", str(many), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    got = 0
    for buf in pipe_frames(cmd, sw * sh * 3 // 2):
        got += 1
        yield buf
    if got != many:
        raise SystemExit(f"the source gave {got} of {many} frames for {first}-{last}")


def softened(frame, hard, sigma, w, h, held=()):
    """`frame` (yuv420p bytes) with its luma blurred where any of the `hard` masks is set and nothing was given back."""
    y = np.frombuffer(frame, np.uint8, w * h).reshape(h, w).astype(np.float32)
    weight = cv2.GaussianBlur(np.any(hard, axis=0).astype(np.float32), (0, 0), FEATHER)
    for back in held:
        if back is not None:
            weight = weight * (1.0 - back)
    y += (cv2.GaussianBlur(y, (0, 0), sigma) - y) * weight
    return np.clip(np.round(y), 0, 255).astype(np.uint8).tobytes() + frame[w * h:]


def fed_frames(source, segs, w, h, record=None, soften=0.0, captures=None, full=None):
    """Every frame of the span as yuv420p bytes. `record`, when given, is filled with what each piece changed.
    `full` is (source width, source height) for a file at the source's size, None for one at the canvas's."""
    box = crop_of(*full, w, h) if full else None
    before = None                                    # the fitted source's luma on the frame before, within a run of rows
    for first, last, rows in segs:
        if not rows:
            before = None
            yield from (source_frames(source, first, last, *full) if full else original_frames(source, first, last, w, h))
            continue
        orig = original_frames(source, first, last, w, h)
        whole = source_frames(source, first, last, *full) if full else iter(lambda: None, 0)
        runs = [run_for(r, captures) for r in rows]
        got = 0
        for k, (obuf, sbuf, *pbufs) in enumerate(zip(orig, whole, *(piece_frames(r, first, last, w, h) for r in rows))):
            n = first + k
            o = planes(obuf, w, h)
            pieces = [planes(b, w, h) for b in pbufs]
            diffs = [difference(p, o) for p in pieces]
            hard = [changed(p, o, d) for p, d in zip(pieces, diffs)]
            masks = [captures.mask(run[2]["subject"], n, prefer=run[0]) if captures and run else None for run in runs]
            whose = [captures.owned(run[2]["subject"], n, prefer=run[0]) if captures and run else None for run in runs]
            holds = [w_ if w_ is not None else m_ for w_, m_ in zip(whose, masks)]
            owner, shared = owners(hard, holds)
            backs = [restore_weight(r, captures, run, n, (h, w)) for r, run in zip(rows, runs)]
            held = [b[0] for b in backs]
            if record is not None and before is not None:
                record.setdefault("source_moved", {})[n] = float(np.abs(o[0] - before).mean())
            before = o[0]
            if record is not None and "fitted" in record:
                record["fitted"][n] = (o[0], np.any(hard, axis=0))       # taken out again by the check, a frame later
            if record is not None:
                for r, m, run, s, d, (back, classes) in zip(rows, hard, runs, masks, diffs, backs):
                    row = {"px": int(m.sum()), "box": box_of(m), "away": None, "away_box": None, "subject_px": None,
                           "at_the_crop": bool(box and ((box[3] and (m[0].any() or m[-1].any())) or (box[2] and (m[:, 0].any() or m[:, -1].any()))))}
                    if r.get("restore"):
                        row.update({"restored_px": None, "class_px": None, "join_px": None, "join_box": None})
                        if classes is not None:
                            given = back > 0.5
                            ring = cv2.dilate(given.astype(np.uint8), disc(3)).astype(bool) & ~given & (d > JOIN)
                            row.update({"restored_px": int((given & m).sum()), "class_px": int(classes.sum()),
                                        "join_px": int(ring.sum()), "join_box": box_of(ring)})
                    if s is not None and run is not None:
                        reach = int(run[2].get("margin_px") or MARGIN) + TOKEN
                        laid = m & ~(back > 0.5) if back is not None else m          # what a restore gave back is not laid
                        far = laid & ~cv2.dilate(s.astype(np.uint8), disc(reach)).astype(bool)
                        row.update({"away": int(far.sum()), "away_box": box_of(far), "subject_px": int(s.sum())})
                    record["rows"].setdefault(r["piece"], {})[n] = row
                if len(rows) > 1:
                    record["shared"][n] = shared
            if record is not None:
                subjects = [run[2]["subject"] if run else None for run in runs]
                for i, r in enumerate(rows):
                    back = held[i] > 0.5 if held[i] is not None else np.zeros((h, w), bool)
                    later = np.zeros((h, w), bool)
                    for j in range(i + 1, len(rows)):
                        later |= owner == j
                    other = np.zeros((h, w), bool)         # an earlier row's subject holds it: another, known subject
                    if holds[i] is not None:
                        for j in range(i):
                            if holds[j] is not None and subjects[j] != subjects[i]:
                                other |= hard[j] & holds[j] & ~holds[i]
                    meant = hard[i] & ~back & ~later & ~other
                    tally = record.setdefault("shown", {}).setdefault((r["piece"], r.get("first"), r.get("last")), [0, 0, 0])
                    shows = int((meant & (owner == i)).sum())
                    tally[0] += int(meant.sum())
                    tally[1] += shows
                    tally[2] += 1
                    record["rows"][r["piece"]][n]["laid"] = shows      # what of this row is in the file on this frame
            got += 1
            if full:
                yield laid_over(planes(sbuf, *full), pieces, owner, box, held)
                continue
            untouched = len(rows) == 1 and held[0] is None
            frame = pbufs[0] if untouched else merged(pieces, owner, held, o)
            yield softened(frame, hard, soften, w, h, held) if soften else frame
        if got != last - first + 1:
            raise SystemExit(f"{got} of {last - first + 1} frames for {first}-{last}")


def audio_packets(path):
    """(pts, bytes, md5, duration) of every audio packet, timestamps in samples."""
    out = sh([FFMPEG, "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "framemd5", "-"]).decode()
    rows = [line.split(",") for line in out.splitlines() if line and not line.startswith("#")]
    return [(int(r[2]), int(r[4]), r[5].strip(), int(r[3])) for r in rows]


def pcm(path, pre=()):
    return np.frombuffer(sh([FFMPEG, "-v", "error", *pre, "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-ac", "2", "-"]),
                         np.int16).reshape(-1, 2)


def track_only(source, path):
    """The source's audio alone, packets copied: a file whose seeks land on audio packets, not on the picture's keyframes."""
    sh([FFMPEG, "-y", "-v", "error", "-i", str(source), "-vn", "-map", "0:a:0", "-c:a", "copy", "-map_metadata", "-1", path])
    return path


def span_seconds(span, rate):
    return span[0] * rate[1] / rate[0], (span[1] - span[0] + 1) * rate[1] / rate[0]


def audio_input(source, span, rate, lead, track):
    """ffmpeg input arguments for the span's audio, stamped so the span's first frame is time zero."""
    start, length = span_seconds(span, rate)
    if not span[0]:
        return ["-t", f"{length:.6f}", "-i", str(source)]
    return ["-ss", f"{start - lead:.6f}", "-itsoffset", f"{-lead:.6f}", "-t", f"{length + lead:.6f}", "-i", track]


def packet_samples(source) -> int:
    """One audio packet's length in samples, read off the source's first packet."""
    out = sh([FFMPEG, "-v", "error", "-i", str(source), "-map", "0:a:0", "-c", "copy", "-frames:a", "1", "-f", "framemd5", "-"]).decode()
    return next(int(line.split(",")[3]) for line in out.splitlines() if line and not line.startswith("#"))


def build(args, segs, canvas, span, src, captures):
    w, h = canvas
    full = (src["width"], src["height"]) if args.size == "source" else None
    ow, oh = full or canvas
    many = span[1] - span[0] + 1
    rate = src["rate"]
    sound = audio_rate(args.source)
    track = track_only(args.source, args.out + ".track.mov") if sound and span[0] else None
    cmd = [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{ow}x{oh}",
           "-framerate", f"{rate[0]}/{rate[1]}", "-i", "-"]
    if sound:
        cmd += audio_input(args.source, span, rate, packet_samples(args.source) / sound, track) + ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    cmd += ["-vf", PIPED_IS_BT709, "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf), "-pix_fmt", "yuv420p", *BT709_TAGS,
            "-video_track_timescale", str(rate[0]), "-map_metadata", "-1", "-movflags", "+faststart", args.out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert enc.stdin is not None and enc.stderr is not None
    fed = 0
    try:
        for frame in fed_frames(args.source, segs, w, h, soften=args.soften, captures=captures, full=full):
            enc.stdin.write(frame)
            fed += 1
    finally:
        enc.stdin.close()
        err = enc.stderr.read()
        enc.wait()
        if track:
            os.remove(track)
    if enc.returncode != 0 or fed != many:
        raise SystemExit(f"encode failed (fed {fed} of {many}): {err.decode(errors='replace')[-600:]}")
    return fed


def latent_step(frame_in_load):
    """(first frame, length) of the latent step holding a frame counted from a load's first frame, from the pack's
    `loop_plan.step_span`; or a string saying why it could not be asked. Imported only when a flag needs it: the
    pack's module imports ComfyUI core."""
    try:
        import importlib
        import types
        lp = sys.modules.get("_h3pack.loop_plan")
        if lp is None:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from _lib import REPO, bootstrap
            if "comfy.model_management" not in sys.modules:
                bootstrap(cpu=True)
            if "_h3pack" not in sys.modules:
                pkg = types.ModuleType("_h3pack")
                pkg.__path__ = [str(REPO)]
                sys.modules["_h3pack"] = pkg
            lp = importlib.import_module("_h3pack.loop_plan")
        return lp.step_span(frame_in_load)
    except Exception as exc:  # noqa: BLE001 - the flag stands without the note
        return f"loop_plan.step_span could not be asked: {type(exc).__name__}: {exc}"


def flags_of(record, rows, captures) -> tuple[list[dict], list[str]]:
    """What the build's own record says should be looked at, in the capture tool's shape for a flag."""
    out, unchecked, done = [], [], set()
    for row in rows:
        if row["piece"] in done:                 # a piece cut into several rows is one piece: its flags once
            continue
        done.add(row["piece"])
        area = record["rows"].get(row["piece"], {})
        run = run_for(row, captures)
        names = {"piece": os.path.basename(row["piece"]), "run": run[2]["name"] if run else None,
                 "subject": run[2]["subject"] if run else None}
        if run is None:
            unchecked.append(names["piece"])
        seen = {n: v for n, v in area.items() if v["away"] is not None}
        gone = {n: v["px"] for n, v in seen.items() if not v["subject_px"] and v["px"] > SPECK}
        far = {n: v for n, v in seen.items() if v["subject_px"] and v["away"] > SPECK}
        if gone and run is not None:
            out.append({"rule": "piece_changes_where_its_subject_is_not", "level": LEVELS[2], **names,
                        "source_frames": frame_spans(gone),
                        "why": f"{names['piece']} changes up to {max(gone.values())} px on {len(gone)} frame(s) where "
                               f"{names['subject']} has no tracked mask: the pass redrew something that is not its subject",
                        "figures": {"frames": len(gone), "worst_px": max(gone.values()), "frames_with_a_mask": len(seen)},
                        "threshold": {"SPECK": SPECK}})
        if far and run is not None:
            reach = int(run[2].get("margin_px") or MARGIN) + TOKEN
            boxes = np.array([v["away_box"] for v in far.values()])
            out.append({"rule": "piece_changes_away_from_its_subject", "level": LEVELS[1], **names,
                        "source_frames": frame_spans(far),
                        "why": f"{names['piece']} changes up to {max(v['away'] for v in far.values())} px further than {reach} px "
                               f"from {names['subject']}'s tracked mask on {len(far)} frame(s): the new subject reaches further "
                               f"than the old one stood, or something else was redrawn",
                        "figures": {"frames": len(far), "worst_px": max(v["away"] for v in far.values()),
                                    "box": [int(boxes[:, 0].min()), int(boxes[:, 1].min()), int(boxes[:, 2].max()), int(boxes[:, 3].max())],
                                    "frames_with_a_mask": len(seen)},
                        "threshold": {"SPECK": SPECK, "reach_px": reach}})
        moved = record.get("source_moved", {})
        spilled, at_cuts = {}, []
        for cut in sorted(n for n in area if moved.get(n, 0.0) > CUT):

            def changed_at(n):
                return n in area and area[n]["px"] > SPECK
            if not (changed_at(cut) and changed_at(cut - 1)):
                continue
            for step in (1, -1):                         # the frames after the cut, then the frames before it
                run, n = [], cut if step == 1 else cut - 1
                while changed_at(n) and len(run) <= SPILL:
                    run.append(n)
                    n += step
                if len(run) <= SPILL:
                    spilled.update({k: area[k]["px"] for k in run})
                    step = latent_step(cut - row["piece_first"])
                    if isinstance(step, str):
                        at_cuts.append({"cut": cut, "spilled": sorted(run), "latent_step": step})
                    else:
                        a, b = row["piece_first"] + step[0], row["piece_first"] + step[0] + step[1] - 1
                        at_cuts.append({"cut": cut, "spilled": sorted(run), "latent_step": [a, b], "the_cut_splits_it": a != cut,
                                        "it_holds_the_spill": a != cut and all(a <= k <= b for k in run)})
        if spilled:
            known = [c for c in at_cuts if c.get("it_holds_the_spill")]
            other = [c for c in at_cuts if c.get("it_holds_the_spill") is False]
            out.append({"rule": "piece_changes_across_a_cut", "level": LEVELS[2], **names, "source_frames": frame_spans(spilled),
                        "why": f"{names['piece']} changes up to {max(spilled.values())} px on {len(spilled)} frame(s) just across a cut "
                               f"of the source and no further: its region ran over the cut and it redrew the next shot for a moment. "
                               f"End the row at the cut."
                               + (" At " + ", ".join(str(c["cut"]) for c in known) + " the cut falls inside a latent step of the piece's "
                                  "load that holds the spilled frames: the known way a region is carried over a cut." if known else "")
                               + (" At " + ", ".join(str(c["cut"]) for c in other) + " it does not: the cut is on a step's edge or the "
                                  "spill runs past the step, so this is not that mechanism." if other else ""),
                        "figures": {"frames": len(spilled), "worst_px": max(spilled.values()), "at_each_cut": at_cuts,
                                    "the_source_moves_at_those_cuts": sorted({round(v, 1) for n, v in moved.items() if v > CUT
                                                                             and any(abs(n - k) <= SPILL for k in spilled)})},
                        "threshold": {"CUT": CUT, "SPILL": SPILL, "SPECK": SPECK}})
        frames_, px = sorted(area), np.array([area[n]["px"] for n in sorted(area)], np.float64)
        steps = []
        for i in range(4, len(px) - 4):
            if frames_[i] != frames_[i - 1] + 1:
                continue
            before, after = float(np.median(px[max(0, i - STEP_FRAMES):i])), float(np.median(px[i:i + STEP_FRAMES]))
            if min(before, after) > SPECK and max(before, after) > JUMP * min(before, after):
                steps.append((abs(px[i] - px[i - 1]), max(before, after) / min(before, after), frames_[i], before, after))
        found = []
        for group in frame_spans([t[2] for t in steps]):             # one step lights the frames round it: keep the frame
            found.append(max(t for t in steps if group[0] <= t[2] <= group[1])[1:])      # where the area itself moves most
        if found:
            out.append({"rule": "changed_area_steps", "level": LEVELS[1], **names, "source_frames": [[n, n] for _, n, _, _ in found],
                        "why": f"the area {names['piece']} changes steps by {JUMP:g} times or more at " + ", ".join(
                            f"{n} ({b:.0f} to {a:.0f} px)" for _, n, b, a in found) + ": a cut, a change of framing, or another person",
                        "figures": {"steps": len(found), "largest_ratio": round(max(r for r, _, _, _ in found), 2)},
                        "threshold": {"JUMP": JUMP, "STEP_FRAMES": STEP_FRAMES}})
        join = {n: v for n, v in area.items() if (v.get("join_px") or 0) > JOIN_PX}
        if join:
            boxes = np.array([v["join_box"] for v in join.values()])
            out.append({"rule": "restored_pixels_beside_a_large_change", "level": LEVELS[1], **names, "source_frames": frame_spans(join),
                        "why": f"what {names['piece']} gave back to the source sits beside pixels it changed by over {JOIN:g} levels on "
                               f"{len(join)} frame(s), up to {max(v['join_px'] for v in join.values())} px of edge: a join between two pictures",
                        "figures": {"frames": len(join), "worst_px": max(v["join_px"] for v in join.values()),
                                    "box": [int(boxes[:, 0].min()), int(boxes[:, 1].min()), int(boxes[:, 2].max()), int(boxes[:, 3].max())],
                                    "restored_px_most": max(v["restored_px"] for v in area.values() if v.get("restored_px") is not None)},
                        "threshold": {"JOIN": JOIN, "JOIN_PX": JOIN_PX}})
        blind = [n for n, v in area.items() if "restored_px" in v and v["restored_px"] is None]
        if blind:
            out.append({"rule": "restore_has_no_mask", "level": LEVELS[2], **names, "source_frames": frame_spans(blind),
                        "why": f"{names['piece']} was to give pixels back to the source and no capture given holds the subject's "
                               f"mask or class map on {len(blind)} frame(s): nothing was restored there",
                        "figures": {"frames": len(blind)}, "threshold": {}})
        edge = [n for n, v in area.items() if v.get("at_the_crop")]
        if edge:
            out.append({"rule": "piece_changes_up_to_the_loader's_crop", "level": LEVELS[1], **names, "source_frames": frame_spans(edge),
                        "why": f"{names['piece']} changes pixels on the canvas's edge on {len(edge)} frame(s), where the loader's crop cut "
                               f"the source: beyond it the file at the source's size holds only the source's picture",
                        "figures": {"frames": len(edge)}, "threshold": {"CHANGE": CHANGE}})
        idle = [n for n, v in area.items() if not v["px"]]
        if idle:
            out.append({"rule": "piece_changes_nothing", "level": LEVELS[1], **names, "source_frames": frame_spans(idle),
                        "why": f"{names['piece']} is the original on {len(idle)} frame(s) a row gives it",
                        "figures": {"frames": len(idle)}, "threshold": {"CHANGE": CHANGE}})
    both = {n: v for n, v in record["shared"].items() if v["px"]}
    if both:
        boxes = np.array([v["box"] for v in both.values()])
        by_mask, by_order = sum(v["by_mask"] for v in both.values()), sum(v["by_order"] for v in both.values())
        out.append({"rule": "two_pieces_change_the_same_pixels", "level": LEVELS[1] if by_order else LEVELS[0],
                    "piece": None, "run": None, "subject": None, "source_frames": frame_spans(both),
                    "why": f"two pieces changed the same pixels on {len(both)} frame(s), up to {max(v['px'] for v in both.values())} px; "
                           f"{by_mask} px went to the piece whose subject's mask they lie in and {by_order} px to the later row",
                    "figures": {"frames": len(both), "largest_px": max(v["px"] for v in both.values()),
                                "box": [int(boxes[:, 0].min()), int(boxes[:, 1].min()), int(boxes[:, 2].max()), int(boxes[:, 3].max())],
                                "settled_by_a_mask_px": by_mask, "settled_by_order_px": by_order},
                    "threshold": {"CHANGE": CHANGE}})
    for i, f in enumerate(out, 1):
        f["id"] = f"d{i:03d}"
    return out, unchecked


def source_moved_over(source, span, w, h) -> dict[int, float]:
    """The fitted source's mean luma change from the frame before, for every frame of the span after its first."""
    moved, before = {}, None
    for n, buf in zip(range(span[0], span[1] + 1), original_frames(source, span[0], span[1], w, h)):
        y = planes(buf, w, h)[0]
        if before is not None:
            moved[n] = float(np.abs(y - before).mean())
        before = y
    return moved


def shots_of(record, rows, captures, span, moved, intents) -> tuple[dict, list[str]]:
    """The record's `shots` section and the failures it raises: per shot and named subject, was anything laid."""
    cuts = sorted(n for n, v in moved.items() if v > CUT)
    edges = [span[0], *cuts, span[1] + 1]
    shots = [(a, b - 1) for a, b in zip(edges, edges[1:]) if b > a]
    named, unknown = {}, []                           # label -> its rows' pieces
    for row in rows:
        run = run_for(row, captures)
        if run is None:
            unknown.append(os.path.basename(row["piece"]))
        else:
            named.setdefault(run[2]["subject"], []).append(row["piece"])

    def laid_on(label, n) -> int:
        return sum(record["rows"].get(piece, {}).get(n, {}).get("laid", 0) for piece in set(named[label]))

    def tracked_on(label, n):
        mask = captures.mask(label, n) if captures else None
        return None if mask is None else bool(mask.any())

    flagged = captures.said_absent_with_people() if captures else []
    tables = captures.shot_tables() if captures else []

    def people_on(a, b):
        """(how many people a shot table found on this shot, whose table) or (None, None): the table shot that
        holds most of these frames, the largest count where tables differ."""
        best = (None, None)
        for table in tables:
            for shot in table["shots"]:
                lo, hi = shot["frames"]
                if 2 * (min(b, hi) - max(a, lo) + 1) > b - a + 1 and (best[0] is None or shot["people"] > best[0]):
                    best = (shot["people"], f"{table['subject']}'s shot table in {table['capture']}")
        return best

    out, own, problems, used = [], [], [], set()
    for a, b in shots:
        people, whose = people_on(a, b)
        entry = {"frames": [a, b], "people_detected": people, "people_counted_by": whose, "subjects": {}}
        state = {}
        for label in named:
            frames = range(a, b + 1)
            laid = [n for n in frames if laid_on(label, n) > 0]
            seen = [tracked_on(label, n) for n in frames]
            tracked = [n for n, v in zip(frames, seen) if v]
            state[label] = {"frames_laid": len(laid), "frames_tracked": len(tracked) if any(v is not None for v in seen) else None}
            if laid:
                idle = [n for n in tracked if n not in set(laid)]
                runs = frame_spans(idle)
                if runs:
                    state[label]["tracked_with_nothing_laid"] = {"frames": len(idle), "runs": runs[:24],
                                                                 "longest": max(y - x + 1 for x, y in runs)}
        for label in named:
            st = state[label]
            if st["frames_laid"]:
                st["reading"] = "laid"
                clash = [i for i in intents if i["subject"] == label and i["first"] <= b and i["last"] >= a]
                for i in clash:
                    used.add(id(i))
                    problems.append(f"the table says {label} is left as the source's on {i['first']}-{i['last']}, and "
                                    f"{st['frames_laid']} frame(s) of shot {a}-{b} have something of theirs laid")
                entry["subjects"][label] = st
                continue
            others = [x for x in named if x != label and (state[x]["frames_laid"] or state[x]["frames_tracked"])]
            flag = next((f for f in flagged if f["subject"] == label and any(x <= b and y >= a for x, y in f["spans"])), None)
            count = flag["people"] if flag and flag.get("people") is not None else people
            said = next((i for i in intents if i["subject"] == label and i["first"] <= a and i["last"] >= b), None)
            fails = False
            if st["frames_tracked"]:
                reading, fails = "tracked, nothing laid", True
            elif count is not None and count > len(others):
                reading, fails = "not tracked, people detected", True
            elif count:
                reading = "not tracked, the people detected are other subjects'"
            elif count == 0:
                reading = "not tracked, nobody detected"
            else:
                reading = "not known"
            st.update({"reading": reading, "people_detected": count, "other_subjects_there": others})
            if flag:
                st["the_capture_said"] = {"flag": flag["id"], "capture": flag["capture"], **flag["figures"]}
            if said is not None:
                used.add(id(said))
                st.update({"reading": "by intent", "would_have_read": reading, "why": said["why"]})
            elif fails:
                problems.append(f"nothing laid for {label} on {a}-{b}: {reading}"
                                + (f" ({count} detected" + (f", {len(others)} of them other subjects laid or tracked there" if others else "") + ")"
                                   if count is not None else "")
                                + f". Lay a row there, or say `source {a}-{b} subject={label} <why>` in the table")
            entry["subjects"][label] = st
            own.append({"frames": [a, b], "subject": label, **{k: v for k, v in st.items() if k not in ("frames_laid",)}})
        out.append(entry)
    for i in intents:
        if id(i) in used:
            continue
        if i["subject"] not in named:
            problems.append(f"the table's `source {i['first']}-{i['last']}` names {i['subject']}, a subject no row of it has")
        else:
            problems.append(f"the table's `source {i['first']}-{i['last']} subject={i['subject']}` covers no whole shot "
                            f"(the shots: {', '.join(f'{a}-{b}' for a, b in shots[:24])})")
    apart = sorted({c for t in tables for c in t["cuts"] if span[0] < c <= span[1]} ^ set(cuts))
    missing = ["a shot's people count is from one frame of it, the frame its tracker's tile shows",
               "a subject with no shot table of its own is given the count another subject's table has for the shot",
               "a dissolve is not a cut here: two shots it joins are read as one"]
    if unknown:
        missing.append("rows whose subject is not known are not asked about: " + ", ".join(sorted(set(unknown))))
    if not tables and not flagged:
        missing.append("no capture given holds a shot table or the flag, so no shot can read as people detected")
    return ({"cut_over_levels": CUT, "cuts": cuts, "cuts_a_capture_puts_elsewhere": apart, "shots": out,
             "where_a_named_subject_is_the_source's_own": own, "missing": missing}, problems)


def write_to_captures(captures, record, rows, flags, out_path, canvas) -> list[str]:
    """Each capture's runs get the table of what their piece changed, and the folder gets the flags."""
    written = []
    stem = Path(out_path).stem
    pixels = canvas[0] * canvas[1]
    for folder, _manifest in captures.folders:
        mine = []
        for row in rows:
            run = run_for(row, captures)
            if run is None or run[0] != folder:
                continue
            mine.append(run[2]["name"])
            path = folder / "runs" / run[2]["name"] / f"changed__{stem}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["frame", "source_frame", "run", "subject", "changed_px", "changed_share", "changed_box",
                             "away_from_subject_px", "shared_px", "shared_settled_by_mask_px", "restored_px", "restored_class_px",
                             "restored_beside_a_large_change_px"])
                for n, v in sorted(record["rows"].get(row["piece"], {}).items()):
                    shared = record["shared"].get(n, {})
                    wr.writerow([n - row["piece_first"], n, run[2]["name"], run[2]["subject"], v["px"], round(v["px"] / pixels, 5),
                                 json.dumps(v["box"]) if v["box"] else "", "" if v["away"] is None else v["away"],
                                 shared.get("px", ""), shared.get("by_mask", ""),
                                 *("" if v.get(k) is None else v[k] for k in ("restored_px", "class_px", "join_px"))])
            written.append(str(path))
        if mine:
            path = folder / f"delivery__{stem}.json"
            path.write_text(json.dumps({"delivery": os.path.basename(out_path), "written": datetime.datetime.now().isoformat(timespec="seconds"),
                                        "runs": mine, "flags": [f for f in flags if f["run"] in mine or f["run"] is None]}, indent=1) + "\n")
            written.append(str(path))
    return written


def luma_at_canvas(path, w, h):
    """A file's luma as the loader's fit would lay it on the canvas, frame by frame, in the file's own levels.
    Read as yuv420p and not as `gray`: ffmpeg widens a tv-range picture to full range on its way to gray, which
    moved bright and dark pixels by up to sixteen levels and read a quarter of an untouched frame as off the
    source (measured 2026-10-10 on footage; a mid-toned test picture did not show it)."""
    for buf in pipe_frames([FFMPEG, "-v", "error", "-an", "-i", str(path), "-vf", loader_fit(w, h), "-fps_mode", "passthrough",
                            "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"], w * h * 3 // 2):
        yield np.frombuffer(buf, np.uint8, w * h).astype(np.float32).reshape(h, w)


class Subjects:
    """What the delivered file did to each subject a capture knows: its pixels more than `OFF` from the source."""

    def __init__(self, captures):
        self.captures = captures
        self.labels = []
        for _, m in captures.folders:
            self.labels += [s["label"] for s in m["subjects"] if s["label"] not in self.labels]
        self.rows = {label: {"frames": 0, "tracked_px": 0, "tracked_off": 0, "owned_px": 0, "owned_off": 0, "classes": {}}
                     for label in self.labels}
        self.shared = {"px": 0, "off": 0, "settled_px": 0, "contested_px": 0, "frames": 0, "most": [0, None]}
        self.floor = [0, 0]

    def add(self, frame, delivered, fitted, changed_):
        off = np.abs(delivered - fitted) > OFF
        tracks = {}
        for label in self.labels:
            track = self.captures.mask(label, frame)
            if track is None:
                continue
            tracks[label] = track
            row = self.rows[label]
            row["frames"] += 1
            row["tracked_px"] += int(track.sum())
            row["tracked_off"] += int((off & track).sum())
            owns = self.captures.owned(label, frame)
            if owns is not None:
                row["owned_px"] += int(owns.sum())
                row["owned_off"] += int((off & owns).sum())
            cmap = self.captures.classes(label, frame)
            if cmap is not None:
                for c in np.unique(cmap[cmap > 0]):
                    m = cmap == c
                    t = row["classes"].setdefault(int(c), [0, 0, 0, 0])
                    t[0] += int((m & track).sum()); t[1] += int((off & m & track).sum())
                    t[2] += int((m & ~track).sum()); t[3] += int((off & m & ~track).sum())
        if tracks:
            claims = np.sum(list(tracks.values()), axis=0)
            both = claims > 1
            if both.any():
                px = int(both.sum())
                self.shared["px"] += px
                self.shared["off"] += int((off & both).sum())
                self.shared["frames"] += 1
                if px > self.shared["most"][0]:
                    self.shared["most"] = [px, frame]
                owned = [self.captures.owned(label, frame) for label in tracks]
                if all(o is not None for o in owned):
                    settled = int((both & np.any(owned, axis=0)).sum())
                    self.shared["settled_px"] += settled
                    self.shared["contested_px"] += px - settled
            quiet = (claims == 0) & ~changed_
            self.floor[0] += int(quiet.sum())
            self.floor[1] += int((off & quiet).sum())

    def report(self):
        names = class_names()

        def share(off, px):
            return round(100.0 * off / px, 3) if px else None
        out = {}
        for label, r in self.rows.items():
            if not r["frames"]:
                continue
            out[label] = {"frames": r["frames"], "tracked_px": r["tracked_px"], "tracked_off_pct": share(r["tracked_off"], r["tracked_px"]),
                          **({"owned_px": r["owned_px"], "owned_off_pct": share(r["owned_off"], r["owned_px"])} if r["owned_px"] else {}),
                          "by_class": {names[c]: {"px_in_its_track": t[0], "off_pct_in_its_track": share(t[1], t[0]),
                                                  "px_outside_its_track": t[2], "off_pct_outside_its_track": share(t[3], t[2])}
                                       for c, t in sorted(r["classes"].items())}}
        s = self.shared
        return {"off_over_levels": OFF, "floor_off_pct": share(self.floor[1], self.floor[0]), "subjects": out,
                "claimed_by_more_than_one_track": {"px": s["px"], "off_pct": share(s["off"], s["px"]), "frames": s["frames"],
                                                   "most_px_on_a_frame": s["most"], "settled_by_an_owner_map_px": s["settled_px"],
                                                   "left_contested_px": s["contested_px"]}}


def compare(a_path, b_path):
    """Two check records side by side: the same stretch built two ways is a before and an after."""
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    print("A:", a["out"], "|", a["verdict"], "| rows:", "; ".join(f"{os.path.basename(r['piece'])} {r['frames']}"
          + (" restore=" + ",".join(r["restore"]) if r["restore"] else "") for r in a["table"]))
    print("B:", b["out"], "|", b["verdict"], "| rows:", "; ".join(f"{os.path.basename(r['piece'])} {r['frames']}"
          + (" restore=" + ",".join(r["restore"]) if r["restore"] else "") for r in b["table"]))
    print("whose pixels: A", a.get("whose_pixels"), "| B", b.get("whose_pixels"))
    sa, sb = a.get("subjects"), b.get("subjects")
    if not sa or not sb:
        print("one of the records has no `subjects` table (built with no capture, or before the table existed)")
        return

    def cell(v):
        return "     -" if v is None else f"{v:6.2f}"
    print(f"pixels more than {sa['off_over_levels']:g} levels from the source, percent, A -> B   (floor: {cell(sa['floor_off_pct'])} -> {cell(sb['floor_off_pct'])})")
    for label in sorted(set(sa["subjects"]) | set(sb["subjects"])):
        ra, rb = sa["subjects"].get(label, {}), sb["subjects"].get(label, {})
        print(f"  {label}: its tracked mask ({ra.get('tracked_px', rb.get('tracked_px'))} px) {cell(ra.get('tracked_off_pct'))} -> {cell(rb.get('tracked_off_pct'))}"
              + (f"; what it owns {cell(ra.get('owned_off_pct'))} -> {cell(rb.get('owned_off_pct'))}" if "owned_px" in ra or "owned_px" in rb else ""))
        for name in sorted(set(ra.get("by_class", {})) | set(rb.get("by_class", {}))):
            ca, cb = ra.get("by_class", {}).get(name, {}), rb.get("by_class", {}).get(name, {})
            print(f"      {name:16s} in its track ({ca.get('px_in_its_track', cb.get('px_in_its_track'))} px) {cell(ca.get('off_pct_in_its_track'))} -> {cell(cb.get('off_pct_in_its_track'))}"
                  f"   outside it ({ca.get('px_outside_its_track', cb.get('px_outside_its_track'))} px) {cell(ca.get('off_pct_outside_its_track'))} -> {cell(cb.get('off_pct_outside_its_track'))}")
    ca, cb = sa["claimed_by_more_than_one_track"], sb["claimed_by_more_than_one_track"]
    print(f"  claimed by more than one track: {ca['px']} px on {ca['frames']} frames (most {ca['most_px_on_a_frame']}), "
          f"{cell(ca['off_pct'])} -> {cell(cb['off_pct'])}; an owner map settled {cb['settled_by_an_owner_map_px']} and left {cb['left_contested_px']} contested")
    fa, fb = ([f"{f['rule']} {f['source_frames'][:6]}" for f in r["flags"]] for r in (a, b))
    print("flags only in A:", [f for f in fa if f not in fb] or "none")
    print("flags only in B:", [f for f in fb if f not in fa] or "none")


def check(args, segs, canvas, span, src, rows, captures):
    w, h = canvas
    full = (src["width"], src["height"]) if args.size == "source" else None
    ow, oh = full or canvas
    many = span[1] - span[0] + 1
    per = ow * oh * 3 // 2
    rate = src["rate"]
    result = {"out": args.out, "source": os.path.basename(args.source), "span": list(span), "frames_expected": many,
              "size": [ow, oh], "at": args.size, "soften_sigma": args.soften, "failures": [],
              "table": [{"frames": [r["first"], r["last"]], "piece": r["piece"], "piece_first": r["piece_first"],
                         "restore": [restore_text(label, wanted) for label, wanted in r.get("restore") or []]} for r in rows],
              "captures": [str(folder) for folder, _ in captures.folders] if captures else [],
              "checked": datetime.datetime.now().isoformat(timespec="seconds"), **({"note": args.note} if args.note else {})}
    if full:
        cw, ch, x0, y0 = crop_of(*full, w, h)
        result["the_loader's_crop_of_the_source"] = {"width": cw, "height": ch, "x": x0, "y": y0}
    fail = result["failures"].append

    # count, timestamps and what the file says about its colour
    info = probe(args.out)
    pts = sorted(int(v) for v in sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts",
                                     "-of", "csv=p=0", args.out]).decode().replace(",", "").split())
    tb = info["time_base"]
    step = rate[1] * int(tb.split("/")[1]) / rate[0]
    wrong = [i for i, p in enumerate(pts) if p != round(i * step)]
    result["video"] = {"frames": info["frames"], "rate": "%d/%d" % info["rate"], "time_base": tb,
                       "timestamps_off_their_place": len(wrong), "first_off": wrong[:5],
                       "says_range_matrix_transfer_primaries": info["says"]}
    if info["says"] != SAYS:
        fail(f"the file says {info['says']} about its colour, not {SAYS}")
    if (info["width"], info["height"]) != (ow, oh):
        fail(f"the file is {info['width']}x{info['height']}, not {ow}x{oh}")
    if info["frames"] != many:
        fail(f"{info['frames']} frames in the file, {many} expected")
    if info["rate"] != rate:
        fail(f"rate {info['rate']}, not the source's {rate}")
    if wrong or len(pts) != many or step != int(step):
        fail(f"{len(wrong)} timestamps are not n x {step:g} in {tb}")

    # order, by decode
    dec = pipe_frames([FFMPEG, "-v", "error", "-an", "-i", args.out, "-fps_mode", "passthrough",
                       "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"], per)

    def arr(buf):
        return np.frombuffer(buf, np.uint8).astype(np.int16)

    cuts = (0, ow * oh, ow * oh * 5 // 4, per)
    bias = np.zeros(3)

    def mad(a, b):
        return float(np.abs(a - b).mean())

    record = {"rows": {}, "shared": {}}
    tally = Subjects(captures) if captures else None
    if tally:
        record["fitted"] = {}
    # the delivered file as the loader would hand it over: at the canvas, for the table of what it did to each subject
    at_canvas = luma_at_canvas(args.out, w, h) if tally and full else None
    fed = fed_frames(args.source, segs, w, h, record, args.soften, captures, full)
    window = [None, arr(next(fed)), None]            # fed k-1, k, k+1
    placed = static = misplaced = decoded = 0
    own, worst, bad, local = [], (0.0, None), [], []
    for k in range(many):
        nxt = next(fed, None)
        window[2] = arr(nxt) if nxt is not None else None
        got = next(dec, None)
        if got is None:
            break
        decoded += 1
        got = arr(got)
        small = next(at_canvas, None) if at_canvas is not None else None
        if tally and span[0] + k in record["fitted"]:
            fitted, touched = record["fitted"].pop(span[0] + k)
            delivered = small if small is not None else got[:w * h].astype(np.float32).reshape(h, w)
            tally.add(span[0] + k, delivered, fitted, touched)
        d0 = mad(got, window[1])
        own.append(d0)
        squares = np.abs(got[:ow * oh] - window[1][:ow * oh]).reshape(oh, ow)[:oh - oh % BLOCK, :ow - ow % BLOCK]
        squares = squares.reshape(oh // BLOCK, BLOCK, ow // BLOCK, BLOCK).mean(axis=(1, 3))
        local.append((float(squares.max()), int((squares > APART).sum())))
        bias += [float((got[a:b] - window[1][a:b]).mean()) for a, b in zip(cuts, cuts[1:])]
        if d0 > worst[0]:
            worst = (d0, span[0] + k)
        others = [(mad(window[1], x), mad(got, x)) for x in (window[0], window[2]) if x is not None]
        if any(d < d0 for _, d in others):
            misplaced += 1
            bad.append(span[0] + k)
        elif all(same < STATIC for same, _ in others):
            static += 1
        else:
            placed += 1
        window = [window[1], window[2], None]
    extra = sum(1 for _ in dec)
    result["order"] = {"decoded": decoded + extra, "nearest_their_own_fed_frame": placed,
                       "neighbours_are_the_same_picture": static, "nearer_a_neighbour": misplaced,
                       "first_nearer_a_neighbour": bad[:10],
                       "mean_difference_from_own": round(float(np.mean(own)), 3) if own else None,
                       "largest_difference_from_own": [round(worst[0], 3), worst[1]],
                       "largest_difference_from_own_in_a_square": (lambda i: [round(local[i][0], 2), span[0] + i])(
                           int(np.argmax([m for m, _ in local]))) if local else None,
                       "squares_apart": sum(n for _, n in local),
                       "bias_y_u_v": [round(float(x) / max(decoded, 1), 3) for x in bias]}
    if max(abs(x) for x in result["order"]["bias_y_u_v"]) > BIAS:
        fail(f"the decoded file sits {result['order']['bias_y_u_v']} levels (Y, U, V) from the frames fed: a colour conversion")
    if decoded + extra != many:
        fail(f"{decoded + extra} frames decode, {many} expected")
    if misplaced:
        fail(f"{misplaced} frames are nearer a neighbour's picture than their own (first: {bad[:5]})")
    apart = [span[0] + i for i, (_, n) in enumerate(local) if n]
    if apart:
        most = max(range(len(local)), key=lambda i: local[i][1])
        fail(f"{len(apart)} frames hold a part that is not the frame this table makes (frames {frame_spans(apart)[:6]}; most on "
             f"{span[0] + most}: {local[most][1]} squares of {BLOCK} px over {APART:g} levels): the file was not built from "
             f"this table by this tool as it is now")

    # audio: the source's packets, and the source's samples
    sound = audio_rate(args.source)
    if not sound:
        result["audio"] = "the source has no audio; none was written"
    else:
        src_packets, out = audio_packets(args.source), (audio_packets(args.out) if audio_rate(args.out) else [])
        start, length = span_seconds(span, rate)
        runs = [i for i in range(len(src_packets) - len(out) + 1) if src_packets[i][2] == out[0][2]
                and all(src_packets[i + j][1:3] == out[j][1:3] for j in range(len(out)))] if out else []
        audio = {"packets": len(out), "a_run_of_the_source's_packets": bool(runs)}
        if not runs:
            fail("the audio packets are not a run of the source's packets")
        else:
            k0, size = runs[0], src_packets[0][3]
            head = max(0, -out[0][0])
            audio.update({
                "first_source_packet": k0,
                "packets_begin_before_the_span_by_s": round(start - src_packets[k0][0] / sound, 6),
                "head_the_edit_list_skips_s": round(head / sound, 6),
                "sync_error_s": round(out[0][0] / sound - (src_packets[k0][0] / sound - start), 6),
                "track_ends_after_the_picture_by_s": round((out[-1][0] + size) / sound - length, 6),
            })
            if head > HEAD_PACKETS * size:
                fail(f"the track begins {head / sound:.3f} s before the span, over {HEAD_PACKETS} packets")
            # Decoded samples, against the source decoded whole: the file's audio must be nearest the source's at
            # its own place and not a sample either side. Not sample for sample: a decoder's output can depend on the
            # packet it starts from (measured 2026-10-10: one track decoded from 12 s differed from itself decoded
            # from 0 in a few hundred packets), so the count of differing samples is reported and not judged.
            a, b = pcm(args.source).astype(np.int32), pcm(args.out).astype(np.int32)
            s0 = int(round(start * sound))
            lo = 2 * size + SHIFTS                      # past the packets a decoder started mid-stream gets wrong
            n = min(len(b), len(a) - s0 - SHIFTS) - lo
            windows = [(lo + k * max(n - 5 * sound, 0) // 2, min(n, 5 * sound)) for k in range(3)]   # head, middle, tail

            def off(shift):
                return float(np.mean([np.abs(a[s0 + shift + at:s0 + shift + at + many_] - b[at:at + many_]).mean() for at, many_ in windows]))
            here = off(0)
            beside = min((off(k), k) for k in range(-SHIFTS, SHIFTS + 1) if k)
            differ = int((a[s0 + lo:s0 + lo + n] != b[lo:lo + n]).any(axis=1).sum())
            audio.update({"samples_compared": n, "samples_that_differ": differ, "mean_difference_in_place": round(here, 3),
                          "nearest_other_place": [beside[1], round(beside[0], 3)]})
            if not here < beside[0]:
                fail(f"the decoded audio fits the source better {beside[1]} samples from its place ({beside[0]:.2f} against {here:.2f})")
            if abs(audio["sync_error_s"]) > 1.5 / sound:
                fail(f"audio is {audio['sync_error_s']} s off its place")
        result["audio"] = audio

    # what each piece changed, from the pass that fed the order check
    pixels = w * h
    result["regions"] = {"change_over_levels": CHANGE, "pieces": {}}
    for piece, area in record["rows"].items():
        share = [v["px"] / pixels for v in area.values()]
        result["regions"]["pieces"][piece] = {
            "frames": [min(area), max(area)], "share_of_frame_mean": round(float(np.mean(share)), 4),
            "share_of_frame_least": round(min(share), 4), "share_of_frame_most": round(max(share), 4),
            "changed_px_per_frame": [v["px"] for v in area.values()],
            **({"restored_px_per_frame": [v.get("restored_px") for v in area.values()],
                "restored_beside_a_large_change_px_per_frame": [v.get("join_px") for v in area.values()]}
               if any("restored_px" in v for v in area.values()) else {})}
    result["rows_shown"] = []
    for (piece, first_, last_), (meant, shown, n_frames) in record.get("shown", {}).items():
        share = shown / meant if meant else None
        result["rows_shown"].append({"row": f"{os.path.basename(piece)} {first_}-{last_}", "frames": n_frames, "meant_px": meant,
                                     "shown_px": shown, "share": None if share is None else round(share, 4)})
        if share is not None and share < SHOWN:
            fail(f"row {os.path.basename(piece)} {first_}-{last_} was meant to show on {meant} px and shows on {shown} "
                 f"({100 * share:.1f}%): an earlier row is laid over it")
    if tally:
        result["subjects"] = tally.report()
    result["shots"], unlaid = shots_of(record, rows, captures, span, source_moved_over(args.source, span, w, h),
                                       read_intents(args.table))
    for line in unlaid:
        fail(line)
    result["flags"], result["pieces_with_no_capture"] = flags_of(record, rows, captures)
    result["whose_pixels"] = ("no capture given: the table's order" if not captures else
                              "owners.npz of " + ", ".join(sorted(os.path.basename(f) for f in captures.used_owner_map))
                              if captures.used_owner_map else "tracked masks (no owner map covered a frame that asked)")
    if captures and captures.class_px_left:
        left = captures.class_px_left
        result["class_restores_kept_to_their_subject"] = {
            "px_left_because_another_subject_owns_them": int(sum(left.values())), "frames": sum(1 for v in left.values() if v),
            "by_subject": {label: int(sum(v for (_p, who, _f), v in left.items() if who == label)) for label in sorted({k[1] for k in left})}}
    if captures:
        result["written_to_captures"] = write_to_captures(captures, record, rows, result["flags"], args.out, canvas)
    result["verdict"] = "passes" if not result["failures"] else "FAILS"
    own = result["shots"]["where_a_named_subject_is_the_source's_own"]
    kinds = {}
    for entry in own:
        kinds[entry["reading"]] = kinds.get(entry["reading"], 0) + 1
    result["verdict_line"] = result["verdict"] + (
        f"; {len(own)} shot(s) where a named subject is the source's own ("
        + ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items())) + ")" if own
        else "; every named subject has something laid on every shot" if any(e["subjects"] for e in result["shots"]["shots"])
        else "; no row's subject is known, so no shot is asked about")
    if unlaid:
        result["verdict_line"] += ": " + "; ".join(line.split(". Lay a row")[0] for line in unlaid)
    return result


PEOPLE_FLAG = "absent_with_people_on_screen"   # the capture's rule name (`bench/capture_masked_run.py`), read from its flags.json
LOCKS = "LOCKED.md"   # beside the files it names
LOCK_LINE = re.compile(r"^\s*[-*]\s+`([^`]+)`(?:\s+md5\s+([0-9a-fA-F]{32}))?")


def locked(out: str):
    """(the lock file's path, the md5 it gives or None) when `out` is listed in its folder's `LOCKED.md`, else None."""
    path = os.path.join(os.path.dirname(os.path.abspath(out)), LOCKS)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = LOCK_LINE.match(line)
            if m and os.path.basename(m.group(1)) == os.path.basename(out):
                return path, (m.group(2) or "").lower() or None
    return None


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def free_name(out: str) -> str:
    """The first `<stem>_<letter><ext>` beside a locked file that does not exist and is not locked itself."""
    stem, ext = os.path.splitext(out)
    for letter in "bcdefghijklmnopqrstuvwxyz":
        name = f"{stem}_{letter}{ext}"
        if not os.path.exists(name) and not locked(name):
            return name
    return f"{stem}_<name>{ext}"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", help="the original file: untouched frames and the audio come from it")
    p.add_argument("--table")
    p.add_argument("--out")
    p.add_argument("--compare", nargs=2, metavar=("A.check.json", "B.check.json"), help="print two check records side by side, and stop")
    p.add_argument("--span", help="first-last in source frames; default 0 to the last frame any row names")
    p.add_argument("--size", choices=("canvas", "source"), default="canvas",
                   help="canvas: the pieces' size. source: the source's own size, with only what the pieces changed put back over it")
    p.add_argument("--capture", action="append", default=[], help="a capture folder of bench/capture_masked_run.py; repeatable")
    p.add_argument("--crf", type=int, default=12)
    p.add_argument("--soften", type=float, default=0.0, help="sigma in pixels of a luma blur inside what the pieces changed")
    p.add_argument("--note", help="a sentence written into the check record: why this file was built or rebuilt")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    if not (args.source and args.table and args.out):
        p.error("--source, --table and --out are needed to build or check a file")
    lock = locked(args.out)
    if lock and not args.check_only:
        raise SystemExit(f"{args.out} is locked ({lock[0]}): an accepted file is never built over. Write the new version "
                         f"beside it, for one {os.path.basename(free_name(args.out))}, and say in {LOCKS} which stands")

    rows = read_table(args.table)
    src = probe(args.source)
    canvas = None
    for r in rows:
        info = probe(r["piece"])
        size = (info["width"], info["height"])
        canvas = canvas or size
        if size != canvas or info["pix_fmt"] != "yuv420p":
            raise SystemExit(f"{r['piece']}: {size} {info['pix_fmt']}, expected {canvas} yuv420p")
        if info["matrix"] not in ("unknown", "bt709", "smpte170m", "bt470bg"):
            raise SystemExit(f"{r['piece']} is tagged {info['matrix']}; this knows bt709, and untagged or BT.601 values")
        r["matrix"] = info["matrix"]
        if r["first"] < r["piece_first"] or r["last"] - r["piece_first"] >= info["frames"]:
            raise SystemExit(f"{r['piece']} holds source frames {r['piece_first']}-{r['piece_first'] + info['frames'] - 1}, "
                             f"the row asks for {r['first']}-{r['last']}")
    if canvas is None:
        raise SystemExit("the table names no piece")
    span = tuple(int(v) for v in args.span.split("-")) if args.span else (0, max(r["last"] for r in rows))
    if span[1] >= src["frames"]:
        raise SystemExit(f"the source ends at frame {src['frames'] - 1}")
    if args.size == "source":
        if args.soften:
            raise SystemExit("--soften is for a file at the canvas's size; at the source's the scale-up softens the region")
        if src["matrix"] != "bt709":
            raise SystemExit(f"{args.source} is tagged {src['matrix']}; a file at the source's size needs a BT.709 source, as the pieces are")
    if any(r["restore"] for r in rows) and not args.capture:
        raise SystemExit("a row asks for a restore and no --capture was given: the mask or class map comes from a capture folder")
    captures = Captures(args.capture, canvas) if args.capture else None
    segs = segments(rows, span)
    for first, last, covering in segs:
        print(f"{first:5d}-{last:<5d} {last - first + 1:5d} frames  " + (" + ".join(
            f"{os.path.basename(r['piece'])}[{first - r['piece_first']}-{last - r['piece_first']}]"
            + "".join(" less " + restore_text(label, wanted) for label, wanted in r["restore"])
            for r in covering) or "original"),
            flush=True)
    if not args.check_only:
        print("fed", build(args, segs, canvas, span, src, captures), "frames to", args.out, flush=True)
    result = check(args, segs, canvas, span, src, rows, captures)
    result["segments"] = [[a, b, [r["piece"] for r in covering]] for a, b, covering in segs]
    if lock:
        result["locked"] = {"by": lock[0], "md5_locked": lock[1], "md5_now": md5_of(args.out)}
        if lock[1] and lock[1] != result["locked"]["md5_now"]:
            result["failures"].append(f"the file is locked in {lock[0]} at md5 {lock[1]} and is now {result['locked']['md5_now']}: "
                                      "it is not the file that was accepted")
            result["verdict"] = "FAILS"
            result["verdict_line"] = "FAILS (not the locked file)" + result["verdict_line"][len("passes"):] \
                if result["verdict_line"].startswith("passes") else result["verdict_line"] + "; not the locked file"
    with open(args.out + ".check.json", "w") as fh:
        fh.write(json.dumps(result, indent=1) + "\n")
    shown = {k: result[k] for k in ("video", "order", "audio", "rows_shown", "locked", "failures", "verdict", "verdict_line") if k in result}
    shown["shots"] = result["shots"]["where_a_named_subject_is_the_source's_own"]
    shown["regions"] = {os.path.basename(k): {a: b for a, b in v.items() if not a.endswith("_per_frame")}
                        for k, v in result["regions"]["pieces"].items()}
    shown["flags"] = [{k: f[k] for k in ("id", "rule", "level", "source_frames", "why")} for f in result["flags"]]
    if "subjects" in result:
        shown["subjects"] = {"floor_off_pct": result["subjects"]["floor_off_pct"],
                             **{k: {a: b for a, b in v.items() if a != "by_class"} for k, v in result["subjects"]["subjects"].items()},
                             "claimed_by_more_than_one_track": result["subjects"]["claimed_by_more_than_one_track"]}
    shown["pieces_with_no_capture"] = result["pieces_with_no_capture"]
    print(json.dumps(shown, indent=1))
    sys.exit(0 if not result["failures"] else 1)


if __name__ == "__main__":
    main()
