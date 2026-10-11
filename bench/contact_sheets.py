#!/usr/bin/env python3
"""Contact sheets of an input video's own frames, made once and kept: one folder a video, one file a frame range.

    <python> bench/contact_sheets.py make VIDEO            # contact_sheets/<stem>/<stem>_<first>_<last>.jpg
    <python> bench/contact_sheets.py find STEM FRAME       # the sheet that holds a frame, as a path
    <python> bench/contact_sheets.py list                  # which videos have sheets, and what they cover

**What it buys.** Every session that had to know what is on a frame of the input sampled the video again, to
its own scratch folder, at its own spacing, and looked at six or eight stills. The sheets are that look made
once for the whole video and left where the next session finds it: every tile is a raw frame of the input
with its frame number on it, nothing drawn over it. A question that a frame number and the raw frame can
answer is answered by opening a sheet. A closer look, a mask, a render or anything else a raw frame does not
hold still takes its own tool (`bench/frame_sheet.py`, the capture's video).

**Where.** `contact_sheets/` at the top of the repo, ignored by git except for its `AGENTS.md`, which says all
of this to a session that lands in the folder. Under it ONE folder a video, named by the video's file name
without its extension and nothing else, so `fun.mp4` is `contact_sheets/fun/`. A second file of the same name
that is not the same file is refused by name; the same file asked for again makes nothing.

**A sheet's name is its first and last frame**, as frames of the input file counted from 0:
`<stem>_<first>_<last>.jpg`. A sheet holds a tile every `--every` frames from its first frame, so any frame
is within a few frames of a tile, and the frame numbers on the tiles are the input's own. `sheets.json` beside
them holds the file's identity (`clip_folder.identity`), the spacing and the list of sheets.

Frames are picked by number through `bench/_lib/frames.py`, never by time. Needs ffmpeg and ffprobe on the
path. No model, no card, no server.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import clip_folder  # noqa: E402
from _lib.frames import stream  # noqa: E402

#: The folder at the top of the repo that holds every video's sheets. Decided: the owner, 2026-10-10 (a new
#: ignored folder of its own, named by the sessions).
ROOT = HERE.parent / "contact_sheets"
INDEX = "sheets.json"
SCHEMA = "h3_contact_sheets"
#: Frames a sheet covers. Reasoned: ten seconds at the lane's 24 frames a second, forty tiles at EVERY.
SPAN = 240
#: A tile every this many frames. Reasoned: four a second, so any frame is within three frames of a tile.
EVERY = 6
#: Tiles across a sheet, and a tile's width in pixels. Reasoned: eight across at 320 is a 2560 pixel sheet,
#: which a reader opens whole and still tells one person from another on.
COLS = 8
CELL = 320
LABEL = 22                     # the strip above a tile that holds its frame number; inherited: frame_sheet.py
HEAD = 30                      # the strip at the top of a sheet that says what it is; reasoned: one line of text
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


class Refused(SystemExit):
    pass


def ranges(frames: int, span: int) -> list[tuple[int, int]]:
    """The frame ranges of a video's sheets, first and last frame of each, covering every frame once."""
    return [(a, min(a + span, frames) - 1) for a in range(0, frames, span)]


def sheet_name(stem: str, first: int, last: int, frames: int) -> str:
    wide = max(4, len(str(frames - 1)))
    return f"{stem}_{first:0{wide}d}_{last:0{wide}d}.jpg"


def tile_shape(size: tuple[int, int], cell: int) -> tuple[int, int]:
    w, h = size
    return cell, max(2, int(round(cell * h / w / 2)) * 2)


def tile_origin(index: int, cols: int, cell: int, tall: int) -> tuple[int, int]:
    """Where the picture of a sheet's `index`-th tile starts, as (x, y): under its own label strip."""
    return (index % cols) * cell, HEAD + (index // cols) * (tall + LABEL) + LABEL


def draw(tiles: list[tuple[int, np.ndarray]], head: str, cols: int, cell: int, tall: int) -> "np.ndarray":
    from PIL import Image, ImageDraw, ImageFont
    rows = -(-len(tiles) // cols)
    out = np.zeros((HEAD + rows * (tall + LABEL), cols * cell, 3), np.uint8)
    for i, (_, picture) in enumerate(tiles):
        x, y = tile_origin(i, cols, cell, tall)
        out[y:y + tall, x:x + cell] = picture
    img = Image.fromarray(out)
    d = ImageDraw.Draw(img)
    try:
        font, small = ImageFont.truetype(MONO, 16), ImageFont.truetype(MONO, 15)
    except OSError:
        font = small = ImageFont.load_default()
    d.text((4, 6), head, font=font, fill=(255, 255, 255))
    for i, (frame, _) in enumerate(tiles):
        x, y = tile_origin(i, cols, cell, tall)
        d.text((x + 3, y - LABEL + 3), str(frame), font=small, fill=(255, 235, 0))
    return np.asarray(img)


def read_index(folder: Path) -> dict | None:
    path = folder / INDEX
    if not path.exists():
        return None
    index = json.loads(path.read_text())
    if index.get("schema") != SCHEMA:
        raise Refused(f"{path} is not a contact sheet index this tool wrote; move it or the folder aside")
    return index


def make(video: Path, root: Path = ROOT, span: int = SPAN, every: int = EVERY, cols: int = COLS, cell: int = CELL,
         again: bool = False) -> tuple[Path, list[str]]:
    """Write a video's sheets into its one folder. Returns the folder and the names of the sheets written."""
    if span % every:
        raise Refused(f"--span {span} is not a whole number of --every {every}: a sheet must start on a tile")
    video = Path(video)
    ident = clip_folder.identity(video)
    stem = clip_folder.stem(video.name)
    folder = root / stem
    frames = ident["frames"]
    wanted = [sheet_name(stem, a, b, frames) for a, b in ranges(frames, span)]
    old = read_index(folder)
    if old is not None:
        differs = clip_folder.same_file(old["source"], ident)
        if differs:
            raise Refused(f"{folder} already holds the sheets of another file named {old['source']['name']} "
                          f"(differs in {', '.join(differs)}). One folder a video: rename this file, or move that folder aside.")
        same_grid = (old["span"], old["every"], old["cols"], old["cell"]) == (span, every, cols, cell)
        if same_grid and all((folder / name).exists() for name in wanted) and not again:
            return folder, []
        if not same_grid and not again:
            raise Refused(f"{folder} holds sheets at span {old['span']}, every {old['every']}; asked for span {span}, every "
                          f"{every}. One set a video: give --again to replace them.")
        for item in old["sheets"]:
            (folder / item["file"]).unlink(missing_ok=True)
    elif folder.exists() and any(folder.iterdir()):
        raise Refused(f"{folder} exists and holds something this tool did not write; move it aside")
    folder.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    tw, th = tile_shape(tuple(ident["size"]), cell)
    spans = ranges(frames, span)
    held: list[tuple[int, np.ndarray]] = []
    written, at = [], 0

    def flush() -> None:
        nonlocal held, at
        a, b = spans[at]
        head = f"{video.name}   frames {a} to {b} of 0 to {frames - 1}   a tile every {every} frames   raw input, nothing drawn on it"
        name = sheet_name(stem, a, b, frames)
        Image.fromarray(draw(held, head, cols, cell, th)).save(folder / name, quality=88)
        written.append(name)
        held, at = [], at + 1

    picked = stream(str(video), (tw, th), vf=f"select=not(mod(n\\,{every})),scale={tw}:{th}:flags=area", pix="rgb24")
    for k, picture in enumerate(picked):
        frame = k * every
        while at < len(spans) and frame > spans[at][1]:
            flush()
        if at >= len(spans):
            break
        held.append((frame, picture.copy()))
    while at < len(spans):
        flush()
    index = {"schema": SCHEMA, "version": 1, "source": ident, "span": span, "every": every, "cols": cols, "cell": cell,
             "tile": [tw, th], "label": LABEL, "head": HEAD,
             "sheets": [{"first": a, "last": b, "file": sheet_name(stem, a, b, frames)} for a, b in spans],
             "made": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    (folder / INDEX).write_text(json.dumps(index, indent=1) + "\n")
    return folder, written


def find(stem: str, frame: int, root: Path = ROOT) -> Path:
    """The sheet of a video that holds a frame."""
    folder = root / clip_folder.stem(stem)
    index = read_index(folder)
    if index is None:
        raise Refused(f"no contact sheets for {stem!r} under {root}: make them with `bench/contact_sheets.py make VIDEO`")
    for item in index["sheets"]:
        if item["first"] <= frame <= item["last"]:
            return folder / item["file"]
    raise Refused(f"{index['source']['name']} has frames 0 to {index['source']['frames'] - 1}; {frame} is not one of them")


def listing(root: Path = ROOT) -> list[str]:
    lines = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        index = read_index(folder)
        if index is None:
            continue
        s = index["source"]
        lines.append(f"{folder.name}/  {s['name']}  frames 0 to {s['frames'] - 1}  {len(index['sheets'])} sheets, "
                     f"{index['span']} frames each, a tile every {index['every']}  made {index['made']}")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="what", required=True)
    m = sub.add_parser("make", help="write a video's sheets into contact_sheets/<stem>/; nothing if they are there")
    m.add_argument("video")
    m.add_argument("--span", type=int, default=SPAN, help="frames a sheet covers")
    m.add_argument("--every", type=int, default=EVERY, help="a tile every this many frames")
    m.add_argument("--cols", type=int, default=COLS)
    m.add_argument("--cell", type=int, default=CELL, help="a tile's width in pixels")
    m.add_argument("--again", action="store_true", help="replace the sheets that are there")
    m.add_argument("--root", default=str(ROOT))
    f = sub.add_parser("find", help="the sheet that holds a frame")
    f.add_argument("stem", help="the video's file name, with or without its extension")
    f.add_argument("frame", type=int)
    f.add_argument("--root", default=str(ROOT))
    ls = sub.add_parser("list", help="which videos have sheets")
    ls.add_argument("--root", default=str(ROOT))
    a = p.parse_args()
    root = Path(a.root)
    if a.what == "make":
        try:
            folder, written = make(Path(a.video), root, a.span, a.every, a.cols, a.cell, a.again)
        except clip_folder.Refused as e:
            raise SystemExit(str(e))
        if written:
            print(f"wrote {len(written)} sheets to {folder}/: {written[0]} to {written[-1]}")
        else:
            print(f"already there: {folder}/ holds this file's sheets. Open them; make them again only with --again.")
    elif a.what == "find":
        print(find(a.stem, a.frame, root))
    else:
        lines = listing(root)
        print("\n".join(lines) if lines else f"no contact sheets under {root}")


if __name__ == "__main__":
    main()
