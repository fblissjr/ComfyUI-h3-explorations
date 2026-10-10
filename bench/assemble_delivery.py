#!/usr/bin/env python3
"""One delivery file from masked renders and the untouched original: the source's own rate and audio packets,
passes over the same frames merged by what each changed, and a check by decode that every frame is there once.

    <python> bench/assemble_delivery.py --source SOURCE.mp4 --table TABLE.txt --out OUT.mp4 \\
        [--span FIRST-LAST] [--capture data/<date>_<name>]... [--soften SIGMA] [--crf 12] [--check-only]

**What it buys.** A masked render is a window of a clip, at the lane's canvas and the lane's rate, with a track
that was resampled to fit. What gets watched is the whole stretch: several renders by frame range, the frames
nobody regenerated, the clip's own audio. This writes that file with one encode, says how it was put together,
and proves the parts an eye cannot check: that no frame was dropped, doubled or moved, and that the audio is
the source's own packets.

**The table.** One row per piece, in SOURCE frame numbers (a render made from a frame-for-frame copy of the
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
`bench/capture_masked_run.py` writes) the pixel goes to the piece whose subject's tracked mask it lies in, and
only a pixel in both masks or neither falls to the later row. Either way the frames are flagged.

**Flags** (in the check json, and in each capture folder given), each with frames and a figure, to be looked at
and never a refusal: a piece that changes the picture on frames where its subject has no tracked mask at all
(a pass that redrew somebody else; one was caught this way on 2026-10-10, past a cut the tracker had matched
across wrongly); a piece that changes pixels further from its subject's tracked mask than its run's margin and
a token (the new subject reaching further than the old one stood, or something else redrawn: the box says
where); a piece whose changed area jumps against its own median (`JUMP`), which needs no capture and is what
catches a tracker that followed the wrong person, since the mask it left then says the subject is there; two
pieces changing the same pixels, with the box and how many were settled by a mask and how many by
order; a piece that changes nothing on frames a row gives it. The first two need a capture; without one they
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

**The check** (`<out>.check.json`; exit 1 when it fails). Count and timestamps: as many frames as the span,
frame n stamped n frames in at the source's rate. Order: the file is decoded and each frame compared with the
frame fed for its place and the ones fed either side; it must be nearest its own. Frames whose neighbours are
the same picture cannot be placed by content and are counted apart. Audio: the file's
packets are a run of the source's packets, byte for byte, the head is under `HEAD_PACKETS` packets, and the
decoded samples fit the source's at their own place better than a sample or more either side (`SHIFTS`).
Colour: the four fields, and the decoded file's planes have no bias against the frames fed (`BIAS`), which is
what a silent matrix conversion would leave. Regions: per piece and frame, the pixels it changed.

**What it does not show.** Whether the result looks right. That a piece's changed region is the region the
render kept: the node draws what it kept and saves no mask of it, so this rebuilds it from a threshold, and a
change under `CHANGE` further than `GROW` from stronger change is lost to a later row of a merge
(`data/CAPTURE_GAPS.md`). `CHANGE` was measured on footage: on a hard test pattern that moves fast the writer's
own codec noise reached twice it, so on such a picture a kept pixel can read as changed and a flag can be noise. `bench/check_assemble_delivery.py` holds the cases.

Nothing here describes what a clip shows: a clip is a file name and a subject is a capture's label.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
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
JUMP = 2.5        # times a piece's own median changed area that is flagged as a jump; measured 2026-10-10 on four
                  # renders of one clip: within a subject's shots the largest frame was 1.1 to 1.7 times the
                  # median, and the pass that redrew the wrong person past a cut was 3.3 to 3.7 times
TOKEN = 32        # pixels; inherited: a token is two latent cells of 16 a side (`bench/capture_masked_run.py::CELL`),
                  # and a region widened to whole tokens reaches that far past its margin
MARGIN = 64       # pixels, when a capture's run names no margin; inherited: the Masked Source's default grow
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


def read_table(path) -> list[dict]:
    rows = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        span, piece, first = line.split()
        lo, hi = (int(v) for v in span.split("-"))
        rows.append({"first": lo, "last": hi, "piece": piece, "piece_first": int(first)})
    return rows                      # in the file's order: where rows share frames, order is the last resort


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


def changed(piece, orig):
    """Where a piece is not the original: a bool mask at the canvas's size."""
    h, w = piece[0].shape
    chroma = np.maximum(np.abs(piece[1] - orig[1]), np.abs(piece[2] - orig[2]))
    diff = np.maximum(np.abs(piece[0] - orig[0]), cv2.resize(chroma, (w, h), interpolation=cv2.INTER_NEAREST))
    hard = (cv2.blur(diff, (CHANGE_MEAN, CHANGE_MEAN)) > CHANGE).astype(np.uint8)
    return cv2.morphologyEx(hard, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)    # a lone speck is noise


def box_of(mask) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None


def disc(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


class Captures:
    """The capture folders given: which run made a piece, and each subject's tracked mask by source frame."""

    def __init__(self, folders, canvas):
        self.folders, self.masks = [], {}
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


def merged(pieces, hard, subject_masks):
    """The first piece whole, each later piece's changed region over it. Returns the frame and how the shared
    pixels were settled. `hard[i]` is `changed` of piece i; `subject_masks[i]` its subject's mask or None."""
    h, w = hard[0].shape
    owner = np.full((h, w), -1, np.int8)
    for i, m in enumerate(hard):
        owner[m] = i                                   # the last resort: the later row
    shared = np.sum(hard, axis=0) > 1
    by_mask = 0
    if shared.any():
        claims = np.zeros((h, w), np.int8)
        whose = np.full((h, w), -1, np.int8)
        for i, (m, s) in enumerate(zip(hard, subject_masks)):
            if s is None:
                continue
            mine = m & s & shared
            claims[mine] += 1
            whose[mine] = i
        settled = shared & (claims == 1)               # in exactly one of the changing pieces' subjects
        owner[settled] = whose[settled]
        by_mask = int(settled.sum())
    out = [p.copy() for p in pieces[0]]
    for i in range(1, len(pieces)):
        mine = cv2.dilate((owner == i).astype(np.uint8), disc(GROW)).astype(bool) & ((owner == i) | (owner == -1))
        weight = cv2.GaussianBlur(mine.astype(np.float32), (0, 0), FEATHER)
        half = cv2.resize(weight, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        for k, wgt in enumerate((weight, half, half)):
            out[k] += (pieces[i][k] - out[k]) * wgt
    frame = b"".join(np.clip(np.round(x), 0, 255).astype(np.uint8).tobytes() for x in out)
    return frame, {"px": int(shared.sum()), "box": box_of(shared), "by_mask": by_mask, "by_order": int(shared.sum()) - by_mask}


def softened(frame, hard, sigma, w, h):
    """`frame` (yuv420p bytes) with its luma blurred where any of the `hard` masks is set."""
    y = np.frombuffer(frame, np.uint8, w * h).reshape(h, w).astype(np.float32)
    weight = cv2.GaussianBlur(np.any(hard, axis=0).astype(np.float32), (0, 0), FEATHER)
    y += (cv2.GaussianBlur(y, (0, 0), sigma) - y) * weight
    return np.clip(np.round(y), 0, 255).astype(np.uint8).tobytes() + frame[w * h:]


def fed_frames(source, segs, w, h, record=None, soften=0.0, captures=None):
    """Every frame of the span as yuv420p bytes. `record`, when given, is filled with what each piece changed."""
    for first, last, rows in segs:
        orig = original_frames(source, first, last, w, h)
        if not rows:
            yield from orig
            continue
        runs = [captures.run_of(r["piece"]) if captures else None for r in rows]
        got = 0
        for k, bufs in enumerate(zip(orig, *(piece_frames(r, first, last, w, h) for r in rows))):
            n = first + k
            o = planes(bufs[0], w, h)
            pieces = [planes(b, w, h) for b in bufs[1:]]
            hard = [changed(p, o) for p in pieces]
            masks = [captures.mask(run[2]["subject"], n, prefer=run[0]) if captures and run else None for run in runs]
            if record is not None:
                for r, m, run, s in zip(rows, hard, runs, masks):
                    row = {"px": int(m.sum()), "box": box_of(m), "away": None, "away_box": None, "subject_px": None}
                    if s is not None and run is not None:
                        reach = int(run[2].get("margin_px") or MARGIN) + TOKEN
                        far = m & ~cv2.dilate(s.astype(np.uint8), disc(reach)).astype(bool)
                        row.update({"away": int(far.sum()), "away_box": box_of(far), "subject_px": int(s.sum())})
                    record["rows"].setdefault(r["piece"], {})[n] = row
            got += 1
            if len(rows) == 1:
                frame = bufs[1]
            else:
                frame, shared = merged(pieces, hard, masks)
                if record is not None:
                    record["shared"][n] = shared
            yield softened(frame, hard, soften, w, h) if soften else frame
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
    many = span[1] - span[0] + 1
    rate = src["rate"]
    sound = audio_rate(args.source)
    track = track_only(args.source, args.out + ".track.mov") if sound and span[0] else None
    cmd = [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}",
           "-framerate", f"{rate[0]}/{rate[1]}", "-i", "-"]
    if sound:
        cmd += audio_input(args.source, span, rate, packet_samples(args.source) / sound, track) + ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    cmd += ["-vf", PIPED_IS_BT709, "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf), "-pix_fmt", "yuv420p", *BT709_TAGS,
            "-video_track_timescale", str(rate[0]), "-map_metadata", "-1", "-movflags", "+faststart", args.out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert enc.stdin is not None and enc.stderr is not None
    fed = 0
    try:
        for frame in fed_frames(args.source, segs, w, h, soften=args.soften, captures=captures):
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


def flags_of(record, rows, captures) -> tuple[list[dict], list[str]]:
    """What the build's own record says should be looked at, in the capture tool's shape for a flag."""
    out, unchecked = [], []
    for row in rows:
        area = record["rows"].get(row["piece"], {})
        run = captures.run_of(row["piece"]) if captures else None
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
        usual = float(np.median([v["px"] for v in area.values()])) if area else 0.0
        jump = {n: v["px"] for n, v in area.items() if usual and v["px"] > JUMP * usual}
        if jump:
            out.append({"rule": "piece_changes_far_more_than_it_usually_does", "level": LEVELS[1], **names,
                        "source_frames": frame_spans(jump),
                        "why": f"{names['piece']} changes up to {max(jump.values()) / usual:.1f} times its usual area on "
                               f"{len(jump)} frame(s): a different shot, a different person, or a region that grew",
                        "figures": {"frames": len(jump), "worst_px": max(jump.values()), "usual_px": int(usual)},
                        "threshold": {"JUMP": JUMP}})
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


def write_to_captures(captures, record, rows, flags, out_path, canvas) -> list[str]:
    """Each capture's runs get the table of what their piece changed, and the folder gets the flags."""
    written = []
    stem = Path(out_path).stem
    pixels = canvas[0] * canvas[1]
    for folder, _manifest in captures.folders:
        mine = []
        for row in rows:
            run = captures.run_of(row["piece"])
            if run is None or run[0] != folder:
                continue
            mine.append(run[2]["name"])
            path = folder / "runs" / run[2]["name"] / f"changed__{stem}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["frame", "source_frame", "run", "subject", "changed_px", "changed_share", "changed_box",
                             "away_from_subject_px", "shared_px", "shared_settled_by_mask_px"])
                for n, v in sorted(record["rows"].get(row["piece"], {}).items()):
                    shared = record["shared"].get(n, {})
                    wr.writerow([n - row["piece_first"], n, run[2]["name"], run[2]["subject"], v["px"], round(v["px"] / pixels, 5),
                                 json.dumps(v["box"]) if v["box"] else "", "" if v["away"] is None else v["away"],
                                 shared.get("px", ""), shared.get("by_mask", "")])
            written.append(str(path))
        if mine:
            path = folder / f"delivery__{stem}.json"
            path.write_text(json.dumps({"delivery": os.path.basename(out_path), "written": datetime.datetime.now().isoformat(timespec="seconds"),
                                        "runs": mine, "flags": [f for f in flags if f["run"] in mine or f["run"] is None]}, indent=1) + "\n")
            written.append(str(path))
    return written


def check(args, segs, canvas, span, src, rows, captures):
    w, h = canvas
    many = span[1] - span[0] + 1
    per = w * h * 3 // 2
    rate = src["rate"]
    result = {"out": args.out, "source": os.path.basename(args.source), "span": list(span), "frames_expected": many,
              "soften_sigma": args.soften, "failures": []}
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

    cuts = (0, w * h, w * h * 5 // 4, per)
    bias = np.zeros(3)

    def mad(a, b):
        return float(np.abs(a - b).mean())

    record = {"rows": {}, "shared": {}}
    fed = fed_frames(args.source, segs, w, h, record, args.soften, captures)
    window = [None, arr(next(fed)), None]            # fed k-1, k, k+1
    placed = static = misplaced = decoded = 0
    own, worst, bad = [], (0.0, None), []
    for k in range(many):
        nxt = next(fed, None)
        window[2] = arr(nxt) if nxt is not None else None
        got = next(dec, None)
        if got is None:
            break
        decoded += 1
        got = arr(got)
        d0 = mad(got, window[1])
        own.append(d0)
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
                       "bias_y_u_v": [round(float(x) / max(decoded, 1), 3) for x in bias]}
    if max(abs(x) for x in result["order"]["bias_y_u_v"]) > BIAS:
        fail(f"the decoded file sits {result['order']['bias_y_u_v']} levels (Y, U, V) from the frames fed: a colour conversion")
    if decoded + extra != many:
        fail(f"{decoded + extra} frames decode, {many} expected")
    if misplaced:
        fail(f"{misplaced} frames are nearer a neighbour's picture than their own (first: {bad[:5]})")

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
            "changed_px_per_frame": [v["px"] for v in area.values()]}
    result["flags"], result["pieces_with_no_capture"] = flags_of(record, rows, captures)
    if captures:
        result["written_to_captures"] = write_to_captures(captures, record, rows, result["flags"], args.out, canvas)
    result["verdict"] = "passes" if not result["failures"] else "FAILS"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", required=True, help="the original file: untouched frames and the audio come from it")
    p.add_argument("--table", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--span", help="first-last in source frames; default 0 to the last frame any row names")
    p.add_argument("--capture", action="append", default=[], help="a capture folder of bench/capture_masked_run.py; repeatable")
    p.add_argument("--crf", type=int, default=12)
    p.add_argument("--soften", type=float, default=0.0, help="sigma in pixels of a luma blur inside what the pieces changed")
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()

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
    captures = Captures(args.capture, canvas) if args.capture else None
    segs = segments(rows, span)
    for first, last, covering in segs:
        print(f"{first:5d}-{last:<5d} {last - first + 1:5d} frames  " + (" + ".join(
            f"{os.path.basename(r['piece'])}[{first - r['piece_first']}-{last - r['piece_first']}]" for r in covering) or "original"),
            flush=True)
    if not args.check_only:
        print("fed", build(args, segs, canvas, span, src, captures), "frames to", args.out, flush=True)
    result = check(args, segs, canvas, span, src, rows, captures)
    result["segments"] = [[a, b, [r["piece"] for r in covering]] for a, b, covering in segs]
    with open(args.out + ".check.json", "w") as fh:
        fh.write(json.dumps(result, indent=1) + "\n")
    shown = {k: result[k] for k in ("video", "order", "audio", "failures", "verdict")}
    shown["regions"] = {os.path.basename(k): {a: b for a, b in v.items() if a != "changed_px_per_frame"}
                        for k, v in result["regions"]["pieces"].items()}
    shown["flags"] = [{k: f[k] for k in ("id", "rule", "level", "source_frames", "why")} for f in result["flags"]]
    shown["pieces_with_no_capture"] = result["pieces_with_no_capture"]
    print(json.dumps(shown, indent=1))
    sys.exit(0 if not result["failures"] else 1)


if __name__ == "__main__":
    main()
