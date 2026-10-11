#!/usr/bin/env python3
"""Contact sheets of an input video's own frames, made once and kept: one folder a video, one file a frame range.

    <python> bench/contact_sheets.py make VIDEO            # contact_sheets/<stem>/<stem>_<first>_<last>.jpg
    <python> bench/contact_sheets.py verify VIDEO          # are the sheets that are there still this file's, and numbered right
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

**One place.** The tool writes under `contact_sheets/` and takes no other folder: a second set somewhere else
is two answers to "what is on frame N". (`make()` takes a root for its check and for nothing else.)

**A frame number is proved, not assumed.** A number on a tile is the frame's place in the file's own order,
counted from 0, every frame, with no rate conversion; no frame rate is assumed to get it. Before anything is
drawn the tool reads every frame's own timestamp from the file (`times`), and says in each sheet's header and
in `sheets.json` whether they are evenly spaced. After the sheets are written it reads them back: at the
tiles that differ most from the tile before them (the cuts), the frame is read a SECOND way, by its
timestamp and not by its number, and set against the tile in the written picture (`check_written`). The tile
must match its own frame and not the frame one tile earlier. If the count of frames read is not the count
the file reports, if the file gives no timestamps, or if the two readings disagree, everything written is
removed and the tool says why: a sheet with wrong numbers is worse than none. `verify` runs the same
reading on a set that is already there.

What the numbers are NOT: the numbers of a copy made at another rate, or of a loader that changes the rate
as it loads (a forced rate drops or repeats frames). The header says so.

Frames are picked by number through `bench/_lib/frames.py`. Needs ffmpeg and ffprobe on the path. No model,
no card, no server.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
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
#: 2: the index says how the numbers were got and holds the read-back check; a set without it is made again.
VERSION = 2
#: Frames a sheet covers. Reasoned: ten seconds at the lane's 24 frames a second, forty tiles at EVERY.
SPAN = 240
#: A tile every this many frames. Reasoned: four a second, so any frame is within three frames of a tile.
EVERY = 6
#: Tiles across a sheet, and a tile's width in pixels. Reasoned: eight across at 320 is a 2560 pixel sheet,
#: which a reader opens whole and still tells one person from another on.
COLS = 8
CELL = 320
#: How many tiles `check_written` reads a second way, and how far a tile may sit from its own frame (mean
#: absolute difference in grey levels of the written JPEG tile against the frame read by its timestamp).
#: Reasoned: six cuts is a look at both ends and the middle of most clips; the level is set well above what
#: JPEG at quality 88 does to a tile and far below what a different shot gives, and the check also requires
#: the tile to be nearer its own frame than the frame one tile earlier, which no fixed level decides.
CHECKS = 6
NEAR = 8.0
#: Why there is one set, said wherever the tool declines to make another.
WHY = ("There is one set of sheets a video so that every session reads the same frames by the same numbers, and "
       "none makes its own again at its own spacing. contact_sheets/AGENTS.md has the rule.")
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


def times(video: Path) -> list[float] | None:
    """Every video frame's own timestamp in seconds, in presentation order; None when the file gives none."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout.split()
    try:
        return sorted(float(v.strip(",")) for v in out)
    except ValueError:
        return None


def even(stamps: list[float]) -> tuple[bool, float]:
    """Are the frames evenly spaced, and the usual gap between two. Even: no gap more than a tenth off the usual."""
    gaps = np.diff(np.asarray(stamps)) if len(stamps) > 1 else np.asarray([0.0])
    usual = float(np.median(gaps))
    return bool(usual > 0 and np.all(np.abs(gaps - usual) <= 0.1 * usual)), usual


def frame_by_time(video: Path, stamps: list[float], frame: int, size: tuple[int, int]) -> np.ndarray | None:
    """One frame read by its TIMESTAMP and not by its number: the second clock `check_written` sets a tile against."""
    w, h = size
    before = stamps[frame] - stamps[frame - 1] if frame else 0.0
    at = max(0.0, stamps[frame] - 0.4 * before)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{at:.6f}", "-i", str(video), "-map", "0:v:0", "-frames:v", "1",
                          "-vf", f"scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w).astype(np.float32) if len(raw) == w * h else None


def tiles_of(index: dict, folder: Path) -> dict[int, np.ndarray]:
    """Every tile of a set, read from the WRITTEN pictures, in grey: frame number -> tile."""
    from PIL import Image
    tw, th = index["tile"]
    got = {}
    for item in index["sheets"]:
        picture = np.asarray(Image.open(folder / item["file"]).convert("L"), np.float32)
        frames = [f for f in range(item["first"], item["last"] + 1) if f % index["every"] == 0]
        for i, f in enumerate(frames):
            x, y = tile_origin(i, index["cols"], index["cell"], th)
            got[f] = picture[y:y + th, x:x + tw]
    return got


def check_written(video: Path, folder: Path, index: dict, stamps: list[float]) -> dict:
    """Read the written sheets back and set tiles against their frames read by timestamp. Returns what was found.

    The tiles chosen are the ones that differ most from the tile before them, so a tile that carries its
    neighbour's number cannot pass by looking alike. `ok` is False with `why` when a tile is nearer the frame
    one tile earlier than its own, or further than NEAR from its own."""
    got = tiles_of(index, folder)
    every, frames = index["every"], sorted(got)
    jumps = sorted(((float(np.abs(got[f] - got[f - every]).mean()), f) for f in frames if f - every in got), reverse=True)
    rows, why = [], ""
    for jump, f in jumps[:CHECKS]:
        own = frame_by_time(video, stamps, f, tuple(index["tile"]))
        other = frame_by_time(video, stamps, f - every, tuple(index["tile"]))
        if own is None or other is None:
            why = why or f"frame {f} could not be read by its timestamp"
            continue
        d_own, d_other = float(np.abs(got[f] - own).mean()), float(np.abs(got[f] - other).mean())
        rows.append({"frame": f, "to_its_own_frame": round(d_own, 2), "to_the_frame_one_tile_earlier": round(d_other, 2)})
        if d_own > NEAR or d_own >= d_other:
            why = why or (f"the tile numbered {f} is {d_own:.1f} grey levels from frame {f} read by its timestamp and "
                          f"{d_other:.1f} from frame {f - every}: the numbers on the tiles are not the frames'")
    if not rows and not why:
        why = "no two tiles of this video differ, so no tile could be told from its neighbour"
    return {"how": "tiles that differ most from the tile before them, each set against its frame read by timestamp",
            "tiles": rows, "ok": not why, "why": why}


def remove_set(folder: Path, names: list[str]) -> None:
    for name in names + [INDEX]:
        (folder / name).unlink(missing_ok=True)
    if folder.exists() and not any(folder.iterdir()):
        folder.rmdir()


def read_index(folder: Path) -> dict | None:
    path = folder / INDEX
    if not path.exists():
        return None
    index = json.loads(path.read_text())
    if index.get("schema") != SCHEMA:
        raise Refused(f"{path} is not a contact sheet index this tool wrote; move it or the folder aside")
    return index


def make(video: Path, root: Path = ROOT, span: int = SPAN, every: int = EVERY, cols: int = COLS, cell: int = CELL,
         again: bool = False) -> tuple[Path, list[str], dict]:
    """Write a video's sheets into its one folder, then read them back against the video.

    Returns the folder, the names of the sheets written (none when the set was already there) and the index.
    Refuses, leaving nothing it wrote, when the frame numbers cannot be shown to be the frames'."""
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
                          f"(differs in {', '.join(differs)}). One folder a video: rename this file, or move that folder aside. {WHY}")
        same_grid = (old["span"], old["every"], old["cols"], old["cell"]) == (span, every, cols, cell)
        current = old.get("version") == VERSION and (old.get("check") or {}).get("ok")
        if same_grid and current and all((folder / name).exists() for name in wanted) and not again:
            return folder, [], old
        if not same_grid and not again:
            raise Refused(f"{folder} holds sheets at span {old['span']}, every {old['every']}; asked for span {span}, every "
                          f"{every}. Give --again to replace them. {WHY}")
        for item in old["sheets"]:
            (folder / item["file"]).unlink(missing_ok=True)
    elif folder.exists() and any(folder.iterdir()):
        raise Refused(f"{folder} exists and holds something this tool did not write; move it aside")
    stamps = times(video)
    if not stamps or len(stamps) != frames:
        raise Refused(f"{video.name}: the file reports {frames} frames and gives {len(stamps or [])} timestamps, so a frame's "
                      f"number cannot be tied to the frame. Nothing was written. Do not make sheets of it another way and "
                      f"trust the numbers on them: say so to the owner.")
    steady, gap = even(stamps)
    folder.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    tw, th = tile_shape(tuple(ident["size"]), cell)
    spans = ranges(frames, span)
    held: list[tuple[int, np.ndarray]] = []
    written, at = [], 0

    def flush() -> None:
        nonlocal held, at
        a, b = spans[at]
        head = (f"{video.name}  frames {a} to {b} of 0 to {frames - 1} ({clock(stamps[a])} to {clock(stamps[b])})  a tile every "
                f"{every}  numbers: the file's frames in order from 0, every frame"
                + ("" if steady else "  FRAME TIMES NOT EVENLY SPACED") + "  raw input")
        name = sheet_name(stem, a, b, frames)
        Image.fromarray(draw(held, head, cols, cell, th)).save(folder / name, quality=88)
        written.append(name)
        held, at = [], at + 1

    picked = stream(str(video), (tw, th), vf=f"select=not(mod(n\\,{every})),scale={tw}:{th}:flags=area", pix="rgb24")
    k = -1
    for k, picture in enumerate(picked):
        frame = k * every
        while at < len(spans) and frame > spans[at][1]:
            flush()
        if at >= len(spans):
            break
        held.append((frame, picture.copy()))
    read = k + 1
    while at < len(spans):
        flush()
    expected = -(-frames // every)
    index = {"schema": SCHEMA, "version": VERSION, "source": ident, "span": span, "every": every, "cols": cols, "cell": cell,
             "tile": [tw, th], "label": LABEL, "head": HEAD,
             "numbers": "a tile's number is the frame's place in the file's own order, counted from 0, every frame; no rate "
                        "conversion. Not the numbers of a copy at another rate or of a loader that forces a rate.",
             "times": {"read_from": "each frame's own timestamp (ffprobe packet pts)", "evenly_spaced": steady,
                       "usual_gap_seconds": round(gap, 6), "first": round(stamps[0], 6), "last": round(stamps[-1], 6)},
             "sheets": [{"first": a, "last": b, "file": sheet_name(stem, a, b, frames), "from_seconds": round(stamps[a], 3),
                         "to_seconds": round(stamps[b], 3)} for a, b in spans],
             "made": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if read != expected:
        remove_set(folder, written)
        raise Refused(f"{video.name}: {read} frames came out of the read where {expected} were expected for {frames} frames at "
                      f"every {every}, so the numbers on the tiles would be wrong. Nothing is left on disk. Do not make "
                      f"sheets of it another way and trust their numbers: say so to the owner.")
    index["check"] = check_written(video, folder, index, stamps)
    if not index["check"]["ok"]:
        remove_set(folder, written)
        raise Refused(f"{video.name}: the sheets were written and read back, and {index['check']['why']}. They have been "
                      f"removed. Do not make sheets of it another way and trust their numbers: say so to the owner.")
    (folder / INDEX).write_text(json.dumps(index, indent=1) + "\n")
    return folder, written, index


def verify(video: Path, root: Path = ROOT) -> dict:
    """Is the set that is there still this file's, and are its tiles the frames their numbers say. Writes nothing."""
    video = Path(video)
    folder = root / clip_folder.stem(video.name)
    index = read_index(folder)
    if index is None:
        raise Refused(f"no contact sheets for {video.name} under {root}: make them with `bench/contact_sheets.py make VIDEO`")
    differs = clip_folder.same_file(index["source"], clip_folder.identity(video))
    if differs:
        return {"ok": False, "why": f"{folder} holds the sheets of a file that differs from this one in {', '.join(differs)}", "tiles": []}
    missing = [i["file"] for i in index["sheets"] if not (folder / i["file"]).exists()]
    if missing:
        return {"ok": False, "why": f"{len(missing)} sheets named in {INDEX} are not on disk, the first {missing[0]}", "tiles": []}
    stamps = times(video)
    if not stamps or len(stamps) != index["source"]["frames"]:
        return {"ok": False, "why": "the file's timestamps could not be read for every frame", "tiles": []}
    return check_written(video, folder, index, stamps)


def clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"


def find(stem: str, frame: int, root: Path = ROOT) -> Path:
    """The sheet of a video that holds a frame."""
    folder = root / clip_folder.stem(stem)
    index = read_index(folder)
    if index is None:
        raise Refused(f"no contact sheets for {stem!r} under {root}: make them with `bench/contact_sheets.py make VIDEO`")
    if index.get("version") != VERSION or not (index.get("check") or {}).get("ok"):
        raise Refused(f"{folder} holds sheets with no read-back check of their frame numbers: run `bench/contact_sheets.py make VIDEO` "
                      f"again, which replaces them, before reading a frame number off one")
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
        checked = "numbers checked against the frames' timestamps" if (index.get("check") or {}).get("ok") else "NUMBERS NOT CHECKED: make again"
        spaced = "" if (index.get("times") or {}).get("evenly_spaced", True) else "  frame times NOT evenly spaced"
        lines.append(f"{folder.name}/  {s['name']}  frames 0 to {s['frames'] - 1}  {len(index['sheets'])} sheets, "
                     f"{index['span']} frames each, a tile every {index['every']}  made {index['made']}  {checked}{spaced}")
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
    v = sub.add_parser("verify", help="are the sheets that are there this file's, and numbered right; writes nothing")
    v.add_argument("video")
    f = sub.add_parser("find", help="the sheet that holds a frame")
    f.add_argument("stem", help="the video's file name, with or without its extension")
    f.add_argument("frame", type=int)
    sub.add_parser("list", help="which videos have sheets")
    a = p.parse_args()
    root = ROOT                # one place: this tool takes no other folder
    if a.what == "make":
        try:
            folder, written, index = make(Path(a.video), root, a.span, a.every, a.cols, a.cell, a.again)
        except clip_folder.Refused as e:
            raise SystemExit(str(e))
        proof = (f"{len(index['check']['tiles'])} tiles at the largest changes were read back and set against their frames read "
                 f"by timestamp; each is its own frame")
        spaced = "evenly spaced" if index["times"]["evenly_spaced"] else "NOT evenly spaced (the numbers are still the frames' order)"
        if written:
            print(f"wrote {len(written)} sheets to {folder}/: {written[0]} to {written[-1]}\n"
                  f"numbers: the file's frames in order from 0, every frame; frame times read from the file, {spaced}\n"
                  f"checked: {proof}")
        else:
            print(f"nothing written: {folder}/ already holds this file's sheets, made {index['made']} and checked then "
                  f"({proof}).\nwhy: {WHY}\nwhat to do: `bench/contact_sheets.py find {folder.name} FRAME` and open that "
                  f"file; `verify VIDEO` if you doubt them; make something else only for a closer look, a mask, or what a "
                  f"frame number and the raw frame cannot answer.")
    elif a.what == "verify":
        try:
            found = verify(Path(a.video), root)
        except clip_folder.Refused as e:
            raise SystemExit(str(e))
        for row in found["tiles"]:
            print(f"  tile {row['frame']}: {row['to_its_own_frame']} grey levels from its own frame, "
                  f"{row['to_the_frame_one_tile_earlier']} from the frame one tile earlier")
        if not found["ok"]:
            raise SystemExit(f"NOT RIGHT: {found['why']}. Run `make VIDEO --again` to replace the set; do not read frame numbers off it until then.")
        print("ok: the sheets are this file's and each tile read is the frame its number says")
    elif a.what == "find":
        print(find(a.stem, a.frame, root))
    else:
        lines = listing(root)
        print("\n".join(lines) if lines else f"no contact sheets under {root}")


if __name__ == "__main__":
    main()
