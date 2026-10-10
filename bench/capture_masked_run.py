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

**What it buys.** One folder per captured span, `data/<date>_<name>/` (untracked), that says for every frame
and every subject where the tracker's mask was, what part of it was taken, what the sampler regenerated and
what was given back to somebody else, with a `subject` on every row and file. And one stacked, frame-numbered
video drawn from those same files, so the picture and the table cannot disagree. Before it, each of these
was a log line, a tinted picture or a scratch script (`data/CAPTURE_GAPS.md`).

**The words.** A *subject* is a label you choose (`a`, `b`, any number of them). A *sighting* is one saved
mask video of a subject, made by one tracker run (`by=`): the same subject seen by two runs is two
sightings, and how far they agree is a column. A *run* is one pass that regenerated one subject (`subject=`)
and kept others out of its margin (`others=`, labels joined by `+`); its region is read from the render's
own review video (`<render stem>_with_mask.mp4`), which is drawn from the token mask the sampler was given.

**files** reads saved videos only: no server, no model, no card. A mask video is white on the mask, at the
canvas, and its frame 0 is source frame `at=` (the span's `--first` when not given); a preview graph that
saves every mask output through `MaskToImage` writes exactly these. What it writes:

    manifest.json            the clip's file name, the span, the canvas, every subject, sighting and run
    frames.csv / .json       one row a frame, across subjects: each pair's mask overlap, each run's region
                             on each other subject, the margin cells given back, two runs' regions on the
                             same cells, two sightings of one subject against each other, and `voiced`
    subjects/<label>/        masks__<by>.npz (track, parts), per_frame.csv / .json, shots__<by>.json
    runs/<run>/              region.npz (the region in latent cells, and the mask the review shows as
                             carried), per_frame.csv / .json, graph.json (read from the render's picture)
    status.json, README.md   how it came to be, and what it does not hold

**A plan** (`--plan`) is a run that has not rendered: its region is worked out from the saved masks with the
node's own `grow`, the others' pixels taken out, and widened to whole tokens unless `edge=cells`. It is per
frame; the sampler's region also covers every frame of a latent step's run, so the real one is never smaller.
It exists so that `preflight` and `video` can be run on a no-sampling preview, before a render is queued.

**preflight** reads a capture folder and writes `flags.json`: each flag has its rule, the subject's label, the
source frames and the figure that raised it. A flag is a prompt to look at those frames, never a refusal. The
rules, each with its constant below: a shot taken close to the match line or in frames the caller says the
subject is not in (`--not-in`); a shot called absent with somebody on screen; a part mask that spills off its
subject, changes size against its own recent frames, or moves within the subject's box; one run's region over
another subject's mask; a region that is mostly not the subject; a track empty inside a shot the subject was
taken in; and a text that names a voice over frames with none, or the reverse (`--text`, `--voice-spans`).

**outcome** records, beside `flags.json`, what a render did against one flag (`outcomes.json`): it happened or it
did not, in which render, who looked. That is how a threshold's provenance goes from reasoned to measured.
**diagnose** prints what every table says about a stretch of source frames somebody marked as wrong, for every
subject and run, with the flags that were raised on those frames: the first step after a bad render.

**status.json** says how the folder came to be: `running`, `done` or `failed`, with the message and the inputs
that were missing. A folder without it was never finished. Every mask is also saved as a lossless video
(`track__<by>.mkv`, `parts__<by>.mkv`, `runs/<run>/region.mkv` and `carried.mkv`: ffv1, grey, white on the
mask, frame 0 the span's first frame), so a render can load exactly what was looked at.

**video** stacks the source, the masks and a render, each with a zoom on the region beside it, and burns the
source frame number and the render frame number above every row, with the render's audio. Every subject has
one colour: an outline for the tracked mask, a solid fill for what was taken from the still, a light fill for
the margin regenerated round it, and white boxes on margin cells that were given back to somebody else.

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
#: The loader's fit: scale to cover the canvas, crop the centre. Inherited: `bench/masked_render_against_source.py::FITS`.
FIT = "scale={w}:{h}:force_original_aspect_ratio=increase:flags=bicubic,crop={w}:{h}"
FPS = 24
#: A token's side in latent cells. Inherited: the video model patches two cells a side (`video_mask.token_mask`).
TOKEN_CELLS = 2

# The gate's thresholds. Each raises a flag to look at, never a refusal.
#: How close to the match line a taken or absent shot is flagged. Measured on one clip, 2026-10-10: two wrong
#: takes sat 0.011 and 0.016 above the line and the nearest right one 0.029 above, so this flags that one too.
NEAR_LINE = 0.05
#: How far UNDER the line a shot called absent is still flagged as possibly the subject. Measured on the same
#: clip: one wrong absence scored 0.052 under the line, the three right ones 0.20 to 0.37 under.
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
#: A run's region over this share of ANOTHER subject's mask is flagged. Reasoned: under it is the margin's
#: ordinary brush against a neighbour.
REGION_ON_OTHER = 0.05
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

def probe(path: str) -> tuple[int, int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=width,height,nb_read_packets", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip().split(",")
    return int(out[0]), int(out[1]), int(out[2])


def stream(path: str, size: tuple[int, int], first: int = 0, count: int | None = None, vf: str = "", pix: str = "gray"):
    """Frames of `path` from frame `first`, picked by number and never by time, after `vf`, at `size` (w, h)."""
    w, h = size
    c = 1 if pix == "gray" else 3
    chain = ([f"select=gte(n\\,{int(first)})"] if first else []) + ([vf] if vf else [])
    cmd = ["nice", "-n", "19", "ffmpeg", "-v", "error", "-threads", "2", "-i", path, "-map", "0:v:0"]
    cmd += (["-vf", ",".join(chain)] if chain else []) + ["-fps_mode", "passthrough"]
    cmd += (["-frames:v", str(int(count))] if count is not None else []) + ["-f", "rawvideo", "-pix_fmt", pix, "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=w * h * c)
    try:
        while True:
            buf = proc.stdout.read(w * h * c)
            if len(buf) < w * h * c:
                return
            yield np.frombuffer(buf, np.uint8).reshape(h, w, c) if c == 3 else np.frombuffer(buf, np.uint8).reshape(h, w)
    finally:
        proc.stdout.close()
        proc.wait()


def read_mask(path: str, size: tuple[int, int], at: int, first: int, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """A saved mask video as [frames, h, w] of bool on the span's clock, and which frames the video covers.

    The video's frame 0 is source frame `at`. A span frame the video does not reach stays empty and is
    marked not covered, so an empty mask and no mask are never the same row."""
    w, h = size
    mask, covered = np.zeros((frames, h, w), bool), np.zeros(frames, bool)
    skip, lead = max(first - at, 0), max(at - first, 0)
    for n, frame in enumerate(stream(path, size, skip, max(frames - lead, 0), vf=f"scale={w}:{h}:flags=neighbor")):
        mask[lead + n], covered[lead + n] = frame > 127, True
    return mask, covered


# ------------------------------------------------------------------ the tables (pure: the check drives these)

def cells_any(mask: np.ndarray, cell: int = CELL) -> np.ndarray:
    """[n, h, w] of bool to [n, h / cell, w / cell]: true where any pixel of the cell is."""
    n, h, w = mask.shape
    return mask.reshape(n, h // cell, cell, w // cell, cell).any(axis=(2, 4))


def cells_up(cells: np.ndarray, cell: int = CELL) -> np.ndarray:
    return np.repeat(np.repeat(cells, cell, axis=-2), cell, axis=-1)


def write_mask_video(path: Path, mask: np.ndarray) -> None:
    """[n, h, w] of bool as a lossless grey video, white on the mask: what a loader and ImageToMask read back."""
    n, h, w = mask.shape
    enc = subprocess.Popen(["nice", "-n", "19", "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                            "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "ffv1", "-level", "3", str(path)],
                           stdin=subprocess.PIPE)
    for i in range(0, n, 32):
        enc.stdin.write((mask[i:i + 32].astype(np.uint8) * 255).tobytes())
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError(f"ffmpeg could not write {path}")


def planned_region(carried: np.ndarray, others: list[np.ndarray], margin: int, grow, whole_tokens: bool = True) -> np.ndarray:
    """The cells a run would regenerate, [n, h / CELL, w / CELL], from the masks alone.

    The Masked Source's rule: the carried mask grown by the margin, less every pixel of the others that is
    not the subject's own; then any cell with such a pixel, and with `whole_tokens` any token with such a
    cell. Per frame: the frames of one latent step share a region in the sampler, which this does not model."""
    import torch
    grown = grow(torch.from_numpy(carried).to(torch.float32), int(margin)).numpy() > 0.5
    for other in others:
        grown &= ~(other & ~carried)
    cells = cells_any(grown)
    if not whole_tokens:
        return cells
    n, ch, cw = cells.shape
    t = TOKEN_CELLS
    padded = np.pad(cells, ((0, 0), (0, ch % t), (0, cw % t)), mode="edge")
    tokens = padded.reshape(n, padded.shape[1] // t, t, padded.shape[2] // t, t).any(axis=(2, 4))
    return np.repeat(np.repeat(tokens, t, axis=1), t, axis=2)[:, :ch, :cw]


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


def graph_of(render: str) -> dict | None:
    """The graph a render's picture carries (`<stem>.png`, the `prompt` text chunk), or None."""
    picture = Path(render[:-len(".mp4")] + ".png")
    if not picture.is_file():
        return None
    from PIL import Image
    text = Image.open(picture).info.get("prompt")
    return json.loads(text) if text else None


def source_settings(graph: dict | None) -> dict:
    for node in (graph or {}).values():
        if node.get("class_type") == "MiniMaxH3MaskedSource":
            return {k: node["inputs"].get(k) for k in ("grow_pixels", "grow_by", "replace", "edge", "feather_pixels", "composite")}
    return {}


# ------------------------------------------------------------------ files

def spec(text: str) -> tuple[str, dict]:
    label, _, rest = text.partition(":")
    return label, dict(item.split("=", 1) for item in rest.split(",") if item)


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
    out.mkdir(parents=True, exist_ok=True)
    masks, runs, plans = [spec(m) for m in a.mask], [spec(r) for r in a.run], [spec(r) for r in a.plan]
    named = [m[k] for _, m in masks for k in ("track", "parts", "shots") if m.get(k)] + [a.source]
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
    manifest = {"name": a.name, "date": a.date, "clip": Path(a.source).name, "first_frame": first, "frames": frames,
                "size": [width, height], "fps": FPS, "cell_px": CELL, "subjects": [], "runs": [],
                "commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True).stdout.strip()}
    for label, m in masks:
        by, at = m.get("by", "preview"), int(m.get("at", first))
        track, covered = read_mask(m["track"], size, at, first, frames)
        parts = read_mask(m["parts"], size, at, first, frames)[0] if m.get("parts") else None
        if by in sightings.setdefault(label, {}):
            raise SystemExit(f"subject {label!r} has two masks by {by!r}; name each sighting's run with by=")
        sightings[label][by] = (track, covered, parts)
        folder = out / "subjects" / label
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f"masks__{by}.npz", track=np.packbits(track, axis=-1), covered=covered,
                            **({"parts": np.packbits(parts, axis=-1)} if parts is not None else {}))
        write_mask_video(folder / f"track__{by}.mkv", track)
        if parts is not None:
            write_mask_video(folder / f"parts__{by}.mkv", parts)
        if m.get("shots"):
            shutil.copyfile(m["shots"], folder / f"shots__{by}.json")
            beside = Path(m["shots"]).with_suffix(".md")
            if beside.is_file():
                shutil.copyfile(beside, folder / f"shots__{by}.md")
        entry = next((s for s in manifest["subjects"] if s["label"] == label), None)
        if entry is None:
            entry = {"label": label, "colour": list(COLOURS[len(manifest["subjects"]) % len(COLOURS)]), "sightings": []}
            manifest["subjects"].append(entry)
        entry["sightings"].append({"by": by, "track": Path(m["track"]).name, "parts": Path(m["parts"]).name if m.get("parts") else None,
                                   "shots": bool(m.get("shots")), "first_source_frame": at,
                                   "frames_covered": int(covered.sum())})
        print(f"subject {label} seen by {by}: {int(covered.sum())} of {frames} frames covered, "
              f"mask on {int(track.reshape(frames, -1).any(1).sum())}", flush=True)
    for label, seen in sightings.items():
        rows = [row for by, (track, covered, parts) in seen.items() for row in subject_rows(label, by, first, track, covered, parts)]
        write_table(out / "subjects" / label / "per_frame", rows)
    captured: dict = {}
    for name, r in [(n, {**x, "planned": False}) for n, x in runs] + [(n, {**x, "planned": True}) for n, x in plans]:
        label, others = r["subject"], [o for o in r.get("others", "").split("+") if o]
        for who in [label] + others:
            if who not in sightings:
                raise SystemExit(f"run {name!r} names subject {who!r}, which no --mask gives")
        lead = {who: next(iter(sightings[who].values())) for who in [label] + others}
        if r["planned"]:
            if "margin" not in r:
                raise SystemExit(f"plan {name!r} needs margin=PX: the region is worked out from the masks and it")
            graph, settings = None, {"grow_pixels": int(r["margin"]), "grow_by": "a fixed margin",
                                     "edge": "latent cells" if r.get("edge") == "cells" else "whole tokens"}
            track, covered, parts = lead[label]
            carried = parts if parts is not None and r.get("carried", "parts") == "parts" else track
            region = planned_region(carried, [lead[o][0] for o in others], int(r["margin"]), _pack("video_mask").grow,
                                    whole_tokens=r.get("edge") != "cells")
            read, how = covered.copy(), {"read_from": "worked out from the saved masks: a plan, nothing rendered", "legend_px": None}
        else:
            graph = graph_of(r["render"])
            settings = source_settings(graph)
            region, carried, read, how = read_region(r["render"], a.source, first, frames, size, r.get("at"))
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
        if graph is not None:
            (folder / "graph.json").write_text(json.dumps(graph, indent=1) + "\n")
        captured[name] = entry
        manifest["runs"].append({"name": name, "planned": r["planned"], "render": None if r["planned"] else Path(r["render"]).name,
                                 "subject": label, "others": others,
                                 "first_source_frame": int(r.get("at", first)), "margin_px": None if margin is None else int(margin),
                                 "masked_source": settings,
                                 "given_back_worked_out": bool(entry["back"]), "frames_read": int(read.sum()), **how})
        print(f"{'plan' if r['planned'] else 'run'} {name}: region {'worked out' if r['planned'] else 'read'} on "
              f"{int(read.sum())} of {frames} frames, mean share {float(region[read].mean()) if read.any() else 0.0:.4f}", flush=True)
    voice = read_voice(a.voice) if a.voice else None
    manifest["voice_table"] = Path(a.voice).name if a.voice else None
    write_table(out / "frames", cross_rows(first, frames, sightings, captured, voice))
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


def flag_shots(label: str, by: str, table: dict, at: int, not_in: list[list[int]]) -> tuple[list[dict], list[dict]]:
    """A shot table's risks: who was taken close to the line, taken where the caller says the subject is not,
    or called absent with somebody on screen. Frames in the table count from the load's first frame, `at`."""
    flags, shots = [], []
    line = float(table["match"])
    for shot in table["shots"]:
        a, b = at + shot["first_frame"], at + shot["last_frame"]
        state, sim, people = shot["subject"]["state"], shot["subject"].get("similarity"), len(shot.get("people", []))
        level = LEVELS[0]
        barred = [x for x in not_in if x[0] <= b and a <= x[1]]
        if state in ("taken", "picked") and barred:
            level = LEVELS[2]
            flags.append({"rule": "taken_where_not_expected", "level": level, "subject": label, "seen_by": by, "source_frames": [[a, b]],
                          "why": f"{label} is {state} in a shot the caller says they are not in", "figures": {"similarity": sim, "line": line}})
        elif state == "taken" and sim is not None and sim - line < NEAR_LINE:
            level = LEVELS[1]
            flags.append({"rule": "taken_near_the_line", "level": level, "subject": label, "seen_by": by, "source_frames": [[a, b]],
                          "why": f"{label} was taken across a cut at {sim:.3f}, only {sim - line:.3f} above the line: "
                                 "it may be somebody else", "figures": {"similarity": sim, "line": line},
                          "threshold": {"NEAR_LINE": NEAR_LINE}})
        elif state == "absent" and people:
            near = sim is not None and line - sim < NEAR_UNDER
            level = LEVELS[1] if near else LEVELS[0]
            flags.append({"rule": "absent_with_people_on_screen", "level": level, "subject": label, "seen_by": by, "source_frames": [[a, b]],
                          "why": f"{label} is called absent with {people} detected; the closest scored "
                                 + (f"{sim:.3f} against a line of {line:.3f}" if sim is not None else "nothing")
                                 + (": close enough to be them" if near else ""),
                          "figures": {"similarity": sim, "line": line, "people": people}, "threshold": {"NEAR_UNDER": NEAR_UNDER}})
        shots.append({"subject": label, "seen_by": by, "source_frames": [a, b], "state": state, "similarity": sim, "level": level})
    return flags, shots


def flag_track(label: str, by: str, rows: list[dict], table: dict | None, at: int) -> list[dict]:
    """Frames with no mask inside a shot the subject was taken in."""
    if table is None:
        return []
    taken = [(at + s["first_frame"], at + s["last_frame"]) for s in table["shots"] if s["subject"]["state"] in ("taken", "picked")]
    empty = [r["source_frame"] for r in rows if r["covered"] and not r["track_share"]
             and any(a <= r["source_frame"] <= b for a, b in taken)]
    if not empty:
        return []
    return [{"rule": "track_empty_in_a_taken_shot", "level": LEVELS[2], "subject": label, "seen_by": by,
             "source_frames": frame_spans(empty), "why": f"{label} has no mask on {len(empty)} frame(s) of shots they were taken in: "
             "nothing regenerates there", "figures": {"frames": len(empty)}}]


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
    seen = np.array([bool(r["track_share"]) for r in rows])
    size = np.array([r["parts_share"] if r["parts_share"] else np.nan for r in rows], float)
    inside = np.array([r["parts_inside_track"] if r["parts_inside_track"] is not None else np.nan for r in rows], float)
    cx = np.array([r["parts_centre_in_box"][0] if r.get("parts_centre_in_box") else np.nan for r in rows], float)
    cy = np.array([r["parts_centre_in_box"][1] if r.get("parts_centre_in_box") else np.nan for r in rows], float)
    usual = recent_median(size)
    moved = np.hypot(cx - recent_median(cx), cy - recent_median(cy))
    out = []

    def add(rule, level, where, why, **figures):
        if where.any():
            out.append({"rule": rule, "level": level, "subject": label, "seen_by": by, "source_frames": frame_spans(src[where].tolist()),
                        "why": why.format(n=int(where.sum())), "figures": {"frames": int(where.sum()), **figures}})

    add("part_empty_on_the_subject", LEVELS[2], seen & np.isnan(size),
        label + "'s part mask is empty on {n} frame(s) where the subject is there: nothing of the part regenerates")
    add("part_off_the_subject", LEVELS[1], inside < PART_INSIDE,
        label + "'s part mask lies partly off the subject's own mask on {n} frame(s)", threshold_PART_INSIDE=PART_INSIDE,
        lowest=round(float(np.nanmin(inside)), 3) if np.isfinite(inside).any() else None)
    ratio = size / usual
    add("part_changes_size", LEVELS[1], (ratio < PART_SIZE) | (ratio > 1.0 / PART_SIZE),
        label + "'s part mask is under half or over twice its recent size on {n} frame(s)", threshold_PART_SIZE=PART_SIZE)
    add("part_moves_on_the_subject", LEVELS[1], moved > PART_MOVE,
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
            by, at = seen["by"], seen["first_source_frame"]
            mine = [r for r in rows if r["seen_by"] == by]
            path = folder / "subjects" / s["label"] / f"shots__{by}.json"
            table = json.loads(path.read_text()) if path.is_file() else None
            if table is not None:
                f, sh = flag_shots(s["label"], by, table, at, not_in.get(s["label"], []))
                span = (m["first_frame"], m["first_frame"] + m["frames"] - 1)
                flags += [x for x in f if x["source_frames"][0][0] <= span[1] and x["source_frames"][0][1] >= span[0]]
                shots += [x for x in sh if x["source_frames"][0] <= span[1] and x["source_frames"][1] >= span[0]]
            flags += flag_track(s["label"], by, mine, table, at) + flag_parts(s["label"], by, mine)
    flags += flag_runs(m, folder)
    if a.text and a.voice_spans:
        spans = json.loads(Path(a.voice_spans).read_text())["voiced_spans_inclusive"]
        flags += flag_text(Path(a.text).read_text(), spans, m["first_frame"], m["frames"])
    order = {level: i for i, level in enumerate(LEVELS)}
    flags.sort(key=lambda f: (-order[f["level"]], f["source_frames"][0][0]))
    for i, f in enumerate(flags, 1):
        f["id"] = f"f{i:03d}"
    record = {"capture": m["name"], "clip": m["clip"], "span": [m["first_frame"], m["first_frame"] + m["frames"] - 1],
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


def outcome(a: argparse.Namespace) -> None:
    """Record what a render did against one flag, so a threshold's provenance can go from reasoned to measured."""
    folder = Path(a.capture)
    known = {f["id"]: f for f in json.loads((folder / "flags.json").read_text())["flags"]}
    if a.flag not in known:
        raise SystemExit(f"no flag {a.flag} in {folder / 'flags.json'}; it has {sorted(known)}")
    path = folder / "outcomes.json"
    record = json.loads(path.read_text()) if path.is_file() else {"outcomes": []}
    record["outcomes"].append({"flag": a.flag, "rule": known[a.flag]["rule"], "level": known[a.flag]["level"],
                               "happened": a.happened == "yes", "render": a.render, "note": a.note, "by": a.by,
                               "written": datetime.datetime.now().isoformat(timespec="seconds")})
    path.write_text(json.dumps(record, indent=1) + "\n")
    print("recorded", a.flag, a.happened, "in", path)


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
                table, at = json.loads(path.read_text()), seen["first_source_frame"]
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
    canvas_w, canvas_h = w + z, row_h * n_rows
    big, text, small = ImageFont.truetype(MONO, 30), ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 17)
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
    f.add_argument("--mask", action="append", default=[], metavar="LABEL:track=V[,parts=V][,shots=J][,by=RUN][,at=N]",
                   help="one sighting of a subject: its saved mask videos; repeatable")
    f.add_argument("--run", action="append", default=[], metavar="RUN:render=V,subject=LABEL[,others=L+L][,margin=PX]",
                   help="one pass that regenerated a subject; its region is read from the render's review; repeatable")
    f.add_argument("--plan", action="append", default=[], metavar="RUN:subject=LABEL,margin=PX[,others=L+L][,carried=track][,edge=cells]",
                   help="a run that has not rendered: its region is worked out from the masks; repeatable")
    f.add_argument("--voice", help="a per-frame voice table (frame, voiced, vocals_stem_dbfs) on the clip's frames")
    f.add_argument("--out", default=str(REPO / "data"))
    v = sub.add_parser("video", help="the stacked, frame-numbered video from a capture folder")
    v.add_argument("capture")
    v.add_argument("--source", required=True, help="the clip the capture was made from")
    v.add_argument("--render", help="the render shown in the bottom row; two rows without one")
    v.add_argument("--render-first", type=int, help="the source frame that is the render's frame 0, when the render "
                   "is not one of the capture's runs (a merged file, say); a run's own is in the manifest")
    v.add_argument("--all-runs", action="store_true", help="draw every run's region, also when --render is one run's")
    v.add_argument("--out")
    g = sub.add_parser("preflight", help="the risks in a capture folder, before a render: writes flags.json")
    g.add_argument("capture")
    g.add_argument("--not-in", action="append", default=[], metavar="LABEL:FIRST-LAST",
                   help="source frames the subject is not in, as the caller knows the clip; repeatable")
    g.add_argument("--text", help="the prompt file of the run this capture is for")
    g.add_argument("--voice-spans", help="voice_spans.json for the clip")
    o = sub.add_parser("outcome", help="record what a render did against one flag")
    o.add_argument("capture")
    o.add_argument("flag")
    o.add_argument("happened", choices=("yes", "no"))
    o.add_argument("--render", required=True, help="the render it was seen in, by file name")
    o.add_argument("--note", default="")
    o.add_argument("--by", default="", help="who looked")
    d = sub.add_parser("diagnose", help="everything the capture says about a stretch marked as wrong")
    d.add_argument("capture")
    d.add_argument("--frames", required=True, metavar="FIRST-LAST", help="source frames")
    a = p.parse_args()
    {"files": files, "video": video, "preflight": preflight, "outcome": outcome, "diagnose": diagnose}[a.mode](a)


if __name__ == "__main__":
    main()
