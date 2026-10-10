#!/usr/bin/env python3
"""A sheet of numbered frames from several videos on one clock, for reading a render beside its source by eye.

    <python> bench/frame_sheet.py OUT.png --frames 604,640,676 \\
        --row source=CLIP.mov:fit --row render=R.mp4@604 [--crop X,Y,W,H | --follow MASK.mkv@604 --side 300] \\
        [--outline red=PART.mkv@604] [--cell 380] [--size 1024x768]

**What it buys.** A still cannot judge a clip, but a row of the same frames from the source and from each render,
each tile carrying its frame number, is how every "as seen in stills" line gets written, and every session that
read a render had written its own tool for it. This is the one in the tree.

**The clock.** `--frames` are source frame numbers. A row's video starts at source frame `@FIRST` (0 when not
given), so a render that begins at source frame 604 is `R.mp4@604` and its tile for 640 is its own frame 36,
which the label says (`r36`). `:fit` fits a row to the canvas as the lane's loader does (scale to cover, crop
the centre); a row without it is scaled to the canvas. A frame a video does not have is a black tile.

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


def sheet(rows: list[tuple[str, dict[int, np.ndarray], int]], frames: list[int], boxes: dict[int, tuple[int, int, int, int]],
          outlines: list[tuple[tuple[int, int, int], dict[int, np.ndarray]]], cell: int) -> np.ndarray:
    """The picture: a row per video, a tile per frame, each with its label above it."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    tall = max(tile_size(boxes[f], cell)[1] for f in frames)
    out = np.zeros((len(rows) * (tall + LABEL), len(frames) * cell, 3), np.uint8)
    marks = []
    for r, (label, got, first) in enumerate(rows):
        for c, f in enumerate(frames):
            x, y, w, h = boxes[f]
            if f in got:
                tile = got[f].copy()
                for colour, masks in outlines:
                    if f in masks:
                        cont, _ = cv2.findContours(masks[f].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                        cv2.drawContours(tile, cont, -1, colour, 2)
                tw, th = tile_size(boxes[f], cell)
                top = r * (tall + LABEL) + LABEL
                out[top:top + th, c * cell:c * cell + tw] = cv2.resize(
                    tile[y:y + h, x:x + w], (tw, th), interpolation=cv2.INTER_CUBIC if w < tw else cv2.INTER_AREA)
            marks.append((c * cell + 3, r * (tall + LABEL) + 3, f"{label} {f}" + (f" (r{f - first})" if first else "")))
    img = Image.fromarray(out)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(MONO, 14)
    for x, y, text in marks:
        d.text((x, y), text, font=font, fill=(255, 255, 0))
    return np.asarray(img)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out")
    p.add_argument("--frames", required=True, help="source frame numbers, comma separated")
    p.add_argument("--row", action="append", required=True, metavar="LABEL=VIDEO[@FIRST][:fit]")
    p.add_argument("--crop", metavar="X,Y,W,H", help="one box on the canvas for every tile")
    p.add_argument("--follow", metavar="MASK[@FIRST]", help="centre a square on this mask, frame by frame")
    p.add_argument("--side", type=int, default=300, help="the followed square's side on the canvas")
    p.add_argument("--outline", action="append", default=[], metavar="COLOUR=MASK[@FIRST]")
    p.add_argument("--cell", type=int, default=256, help="a tile's width in the sheet")
    p.add_argument("--size", default="1024x768", help="the canvas every row is brought to")
    a = p.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
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
        path, at, fit = named(spec)
        rows.append((label, grab(path, at, frames, (w, h), fit), at))
    from PIL import Image
    picture = sheet(rows, frames, boxes, outlines, a.cell)
    Image.fromarray(picture).save(a.out)
    print("wrote", a.out, f"{picture.shape[1]}x{picture.shape[0]}")


if __name__ == "__main__":
    main()
