#!/usr/bin/env python3
"""A sheet of numbered frames from several videos on one clock, for reading a render beside its source by eye.

    <python> bench/frame_sheet.py OUT.png (--frames 604,640,676 | --frames-from PICK.json) \\
        --row source=CLIP.mov:fit --row render=R.mp4@604 [--crop X,Y,W,H | --follow MASK.mkv@604 --side 300] \\
        [--outline red=PART.mkv@604] [--cell 380] [--size 1024x768]

**What it buys.** A still cannot judge a clip, but a row of the same frames from the source and from each render,
each tile carrying its frame number, is how every "as seen in stills" line gets written, and every session that
read a render had written its own tool for it. This is the one in the tree.

**The clock.** `--frames` are source frame numbers. A row's video starts at source frame `@FIRST` (0 when not
given), so a render that begins at source frame 604 is `R.mp4@604` and its tile for 640 is its own frame 36,
which the label says (`r36`). `:whole` leaves a row out of `--crop` and `--follow`. `:fit` fits a row to the canvas as the lane's loader does (cut to the canvas's shape at the
centre, then scale: `bench/_lib/frames.py::FIT`); a row without it is scaled to the canvas. A frame a video does not have is a black tile.

**Which frames.** `--frames` names them; `--frames-from` takes the list `bench/pick_frames.py` wrote (the frames
where something changes, either side of each, and one a second) and writes each frame's first reason along the
foot of its tiles in the first row, so a stills reader and a video reader look at the same frames for the same
reasons.

**Upstream or downstream.** Three rows tell whether a fault was already in the reference or came from the render:

    <python> bench/frame_sheet.py OUT.png --frames-from PICK.json --row source=SOURCE.mov:fit \\
        --row reference=HEAD_REFERENCE.mp4@5889:whole --row cut=CUT.mp4 --follow HEAD_TRACK.mkv@5889

The head reference's frame 0 is the shot's first frame, and it is already cut to the head, so `:whole` shows each
of its frames whole while the source and cut rows are cropped round the followed head.

**The crop.** `--crop` is one box on the canvas for every tile. `--follow` takes a mask video and centres a
square of `--side` pixels on the mask's centre frame by frame, for a face that moves across the frame; a frame
where the mask is empty falls back on the whole canvas. `--outline COLOUR=MASK` draws a mask's outline on every
tile of that frame (red, green, yellow, magenta, cyan or white).

It writes a picture and nothing else. Where a sheet ends up is the caller's business: a scratch folder for a
look, never a tracked path for a picture of somebody's clip.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.frames import FIT, read_mask, stream  # noqa: E402

COLOURS = {"red": (255, 40, 40), "green": (60, 255, 90), "yellow": (255, 235, 0), "magenta": (255, 70, 220),
           "cyan": (60, 220, 255), "white": (255, 255, 255)}
LABEL = 22                     # the strip above a tile that holds its label, in pixels; reasoned: one line of text
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def named(text: str) -> tuple[str, int, bool]:
    """`PATH[@FIRST][:fit]` as (path, first source frame, fit as the loader does)."""
    fit = text.endswith(":fit")
    path, _, at = (text[:-len(":fit")] if fit else text).partition("@")
    return path, int(at) if at else 0, fit


def picked_frames(path: str) -> tuple[list[int], dict[int, str]]:
    """The frames of a `bench/pick_frames.py` list, and each one's first reason, short."""
    import json
    if not Path(path).is_file():
        raise SystemExit(f"--frames-from {path}: no such file")
    try:
        got = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SystemExit(f"--frames-from {path} is not JSON ({e}): is it a bench/pick_frames.py list?") from None
    if not isinstance(got, dict) or not got.get("frames") or not all(isinstance(f, dict) and "source_frame" in f for f in got["frames"]):
        raise SystemExit(f"{path} lists no frames as bench/pick_frames.py writes them ({{\"frames\": [{{\"source_frame\": ...}}]}})")
    return [int(f["source_frame"]) for f in got["frames"]], {int(f["source_frame"]): f["reasons"][0]["short"] for f in got["frames"] if f.get("reasons")}


def grab(path: str, first: int, frames: list[int], size: tuple[int, int], fit: bool) -> dict[int, np.ndarray]:
    """The frames asked for, by source frame number, from a video whose frame 0 is source frame `first`."""
    w, h = size
    want = sorted({f - first for f in frames if f >= first})
    if not want:
        return {}
    pick = "select=" + "+".join(f"eq(n\\,{n})" for n in want)
    vf = pick + "," + (FIT.format(w=w, h=h) if fit else f"scale={w}:{h}")
    return {n + first: img.copy() for n, img in zip(want, stream(path, size, vf=vf, pix="rgb24"))}


def follow_box(mask: np.ndarray, side: int, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """A square of `side` centred on a mask's centre and kept inside the canvas; the whole canvas for an empty mask."""
    w, h = size
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, w, h
    side = min(side, w, h)
    x = int(np.clip(xs.mean() - side / 2, 0, w - side))
    y = int(np.clip(ys.mean() - side / 2, 0, h - side))
    return x, y, side, side


def tile_size(box: tuple[int, int, int, int], cell: int) -> tuple[int, int]:
    return cell, max(1, int(round(cell * box[3] / box[2])))


def sheet(rows: list[tuple], frames: list[int], boxes: dict[int, tuple[int, int, int, int]],
          outlines: list[tuple[tuple[int, int, int], dict[int, np.ndarray]]], cell: int, why: dict[int, str] | None = None) -> np.ndarray:
    """The picture: a row per video, a tile per frame, each with its label above it. A row is (label, frames, first)
    or (label, frames, first, whole): a whole row is shown uncropped. `why` is written along the foot of the first
    row's tiles."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    shown = [img for row in rows for img in row[1].values()]
    canvas = (0, 0, shown[0].shape[1], shown[0].shape[0]) if shown else None     # every row is brought to the canvas
    uncut = [canvas] if canvas is not None and any(len(row) > 3 and row[3] for row in rows) else []
    tall = max(tile_size(b, cell)[1] for b in [boxes[f] for f in frames] + uncut)
    out = np.zeros((len(rows) * (tall + LABEL), len(frames) * cell, 3), np.uint8)
    marks, notes = [], []
    for r, row in enumerate(rows):
        label, got, first = row[:3]
        whole = len(row) > 3 and row[3]
        for c, f in enumerate(frames):
            x, y, w, h = canvas if whole and canvas else boxes[f]
            if f in got:
                tile = got[f].copy()
                for colour, masks in outlines:
                    if f in masks:
                        cont, _ = cv2.findContours(masks[f].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                        cv2.drawContours(tile, cont, -1, colour, 2)
                tw, th = tile_size((x, y, w, h), cell)
                top = r * (tall + LABEL) + LABEL
                out[top:top + th, c * cell:c * cell + tw] = cv2.resize(
                    tile[y:y + h, x:x + w], (tw, th), interpolation=cv2.INTER_CUBIC if w < tw else cv2.INTER_AREA)
            marks.append((c * cell + 3, r * (tall + LABEL) + 3, f"{label} {f}" + (f" (r{f - first})" if first else "")))
            if r == 0 and why and f in why:
                notes.append((c * cell + 3, (tall + LABEL) - 16, why[f]))
    img = Image.fromarray(out)
    d = ImageDraw.Draw(img)
    font, small = ImageFont.truetype(MONO, 14), ImageFont.truetype(MONO, 11)
    for x, y, text in marks:
        d.text((x, y), text, font=font, fill=(255, 255, 0))
    for x, y, text in notes:                       # the reason a frame was picked, cut to the tile's width
        text = text[:max(4, (cell - 6) // 7)]
        d.rectangle([x - 2, y - 1, x + 7 * len(text) + 1, y + 13], fill=(0, 0, 0))
        d.text((x, y), text, font=small, fill=(255, 255, 255))
    return np.asarray(img)


def main() -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    p.add_argument("out")
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--frames", help="source frame numbers, comma separated")
    which.add_argument("--frames-from", metavar="PICK.json", help="the frames bench/pick_frames.py picked, each tile footed with its first reason")
    p.add_argument("--row", action="append", required=True, metavar="LABEL=VIDEO[@FIRST][:fit|:whole]")
    p.add_argument("--crop", metavar="X,Y,W,H", help="one box on the canvas for every tile")
    p.add_argument("--follow", metavar="MASK[@FIRST]", help="centre a square on this mask, frame by frame")
    p.add_argument("--side", type=int, default=300, help="the followed square's side on the canvas")
    p.add_argument("--outline", action="append", default=[], metavar="COLOUR=MASK[@FIRST]")
    p.add_argument("--cell", type=int, default=256, help="a tile's width in the sheet")
    p.add_argument("--size", default="1024x768", help="the canvas every row is brought to")
    a = p.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
    why: dict[int, str] = {}
    if a.frames_from:
        frames, why = picked_frames(a.frames_from)
    else:
        frames = [int(v) for v in a.frames.split(",")]

    def masks_at(spec: str) -> dict[int, np.ndarray]:
        path, at, _ = named(spec)
        lo, hi = min(frames), max(frames)
        mask, covered = read_mask(path, (w, h), at, lo, hi - lo + 1)
        return {f: mask[f - lo] for f in frames if covered[f - lo]}

    whole = tuple(int(v) for v in a.crop.split(",")) if a.crop else (0, 0, w, h)
    boxes = {f: whole for f in frames}
    if a.follow:
        followed = masks_at(a.follow)
        boxes = {f: follow_box(followed[f], a.side, (w, h)) if f in followed else whole for f in frames}
    outlines = []
    for item in a.outline:
        colour, _, spec = item.partition("=")
        if colour not in COLOURS:
            raise SystemExit(f"no colour {colour!r}; one of {', '.join(COLOURS)}")
        outlines.append((COLOURS[colour], masks_at(spec)))
    rows = []
    for item in a.row:
        label, _, spec = item.partition("=")
        whole = spec.endswith(":whole")
        path, at, fit = named(spec[:-len(":whole")] if whole else spec)
        rows.append((label, grab(path, at, frames, (w, h), fit), at, whole))
    from PIL import Image
    picture = sheet(rows, frames, boxes, outlines, a.cell, why)
    Image.fromarray(picture).save(a.out)
    print("wrote", a.out, f"{picture.shape[1]}x{picture.shape[0]}")


if __name__ == "__main__":
    main()
