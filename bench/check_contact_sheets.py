#!/usr/bin/env python3
"""`bench/contact_sheets.py` against a clip made for the purpose: the folder, the names, the tiles, and no second set.

A sheet that puts frame 36 under the number 30 sends a reader to the wrong frame, and a second folder for one
video means two answers to "what is on frame N". So this writes a small lossless clip in which frame n is a
flat grey of level 10 n, makes its sheets into a scratch root, and reads the PICTURES THAT WERE WRITTEN.

1. **One folder, named by the file.** `clip one.mkv` goes to `<root>/clip one/` and nowhere else, and the only
   things in it are its sheets and `sheets.json`.
2. **A sheet's name is its first and last frame**, and the sheets cover every frame of the clip once.
3. **A tile is the frame its number says.** Each sheet is opened and each tile's level read at its centre: it is
   the level of the frame the index puts there. The last sheet, which is shorter, too.
4. **Made once.** Asked for again, nothing is written and no file changes; another spacing is refused without
   `--again` and replaces the set with it, leaving no sheet of the old set behind.
5. **Another file of the same name is refused**, by name, and the folder is untouched.
6. **`find`** gives the sheet that holds a frame, and refuses a frame the clip does not have.

Needs ffmpeg and ffprobe on the path. No model, no card, no server.

    <comfy venv python> bench/check_contact_sheets.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish, needs  # noqa: E402

needs("ffmpeg and ffprobe on the path", bool(shutil.which("ffmpeg") and shutil.which("ffprobe")))

import contact_sheets as cs  # noqa: E402

W, H, N = 64, 32, 22                      # 22 frames at a span of 12: one full sheet and one short one
SPAN, EVERY, COLS, CELL = 12, 3, 2, 64
WORK = Path(tempfile.mkdtemp(prefix="check_contact_sheets_"))
ROOT = WORK / "sheets"


def clip(path: Path, n: int = N, step: int = 10) -> None:
    """A lossless grey clip: frame k is a flat level of `step` k."""
    frames = np.zeros((n, H, W), np.uint8)
    for k in range(n):
        frames[k] = step * k
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", "24",
                            "-i", "-", "-c:v", "ffv1", "-level", "3", str(path)], stdin=subprocess.PIPE)
    enc.stdin.write(frames.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0


SRC = WORK / "clip one.mkv"
clip(SRC)


def levels(folder: Path) -> dict[int, int]:
    """Every tile of every sheet in a folder, read from the written pictures: frame number -> level at the tile's centre."""
    from PIL import Image
    index = json.loads((folder / cs.INDEX).read_text())
    tw, th = index["tile"]
    got = {}
    for item in index["sheets"]:
        picture = np.asarray(Image.open(folder / item["file"]).convert("L"))
        tiles = [f for f in range(item["first"], item["last"] + 1) if f % index["every"] == 0]
        for i, f in enumerate(tiles):
            x, y = cs.tile_origin(i, index["cols"], index["cell"], th)
            got[f] = int(picture[y + th // 2, x + tw // 2])
    return got


def one_folder() -> str:
    folder, written = cs.make(SRC, ROOT, SPAN, EVERY, COLS, CELL)
    assert folder == ROOT / "clip one", folder
    assert [p.name for p in ROOT.iterdir()] == ["clip one"], list(ROOT.iterdir())
    assert sorted(p.name for p in folder.iterdir()) == sorted(written + [cs.INDEX]), sorted(p.name for p in folder.iterdir())
    return "the folder is the file's name without its extension, and holds its sheets and the index only"


def names_cover() -> str:
    folder = ROOT / "clip one"
    index = json.loads((folder / cs.INDEX).read_text())
    assert [i["file"] for i in index["sheets"]] == ["clip one_0000_0011.jpg", "clip one_0012_0021.jpg"], index["sheets"]
    covered = [f for i in index["sheets"] for f in range(i["first"], i["last"] + 1)]
    assert covered == list(range(N)), covered
    assert cs.ranges(25, 12) == [(0, 11), (12, 23), (24, 24)] and cs.sheet_name("a", 0, 99999, 100000) == "a_00000_99999.jpg"
    return "two sheets named by first and last frame; every frame of the clip is in exactly one"


def tiles_are_their_frames() -> str:
    got = levels(ROOT / "clip one")
    assert sorted(got) == list(range(0, N, EVERY)), sorted(got)
    wrong = {f: v for f, v in got.items() if abs(v - 10 * f) > 4}
    assert not wrong, f"a tile is not the frame its number says (frame: level read) {wrong}"
    return f"{len(got)} tiles read from the written pictures, each within 4 levels of its own frame's level"


def made_once() -> str:
    folder = ROOT / "clip one"
    before = {p.name: p.stat().st_mtime_ns for p in folder.iterdir()}
    again_folder, written = cs.make(SRC, ROOT, SPAN, EVERY, COLS, CELL)
    assert again_folder == folder and written == [], written
    assert before == {p.name: p.stat().st_mtime_ns for p in folder.iterdir()}, "a second make touched a file"
    try:
        cs.make(SRC, ROOT, 6, EVERY, COLS, CELL)
        raise AssertionError("another spacing was accepted without --again")
    except cs.Refused as e:
        assert "--again" in str(e), e
    _, written = cs.make(SRC, ROOT, 6, EVERY, COLS, CELL, again=True)
    names = sorted(p.name for p in folder.iterdir() if p.suffix == ".jpg")
    assert names == sorted(written) and len(names) == 4 and "clip one_0000_0011.jpg" not in names, names
    cs.make(SRC, ROOT, SPAN, EVERY, COLS, CELL, again=True)
    return "asked again, nothing is written; another spacing needs --again and leaves no sheet of the old set"


def another_file_same_name() -> str:
    other = WORK / "elsewhere" / "clip one.mkv"
    other.parent.mkdir()
    clip(other, n=N + 5, step=9)
    folder = ROOT / "clip one"
    before = {p.name: p.stat().st_mtime_ns for p in folder.iterdir()}
    try:
        cs.make(other, ROOT, SPAN, EVERY, COLS, CELL)
        raise AssertionError("a different file of the same name was given the same folder")
    except cs.Refused as e:
        assert "another file" in str(e) and "frames" in str(e), e
    assert before == {p.name: p.stat().st_mtime_ns for p in folder.iterdir()}, "the refusal touched the folder"
    return "a different file of the same name is refused, saying what differs, and the folder is untouched"


def finding() -> str:
    assert cs.find("clip one", 11, ROOT).name == "clip one_0000_0011.jpg"
    assert cs.find("clip one.mkv", 12, ROOT).name == "clip one_0012_0021.jpg"
    for bad, word in ((N, "not one of them"), (0, "no contact sheets")):
        try:
            cs.find("clip one" if bad else "no such clip", bad, ROOT)
            raise AssertionError("a frame or a clip that is not there was found")
        except cs.Refused as e:
            assert word in str(e), e
    assert len(cs.listing(ROOT)) == 1 and cs.listing(ROOT)[0].startswith("clip one/")
    return "find gives the sheet a frame is in, by the file's name with or without its extension; refuses what is not there"


case("one folder, named by the file", one_folder)
case("a sheet's name is its first and last frame", names_cover)
case("a tile is the frame its number says", tiles_are_their_frames)
case("made once", made_once)
case("another file of the same name", another_file_same_name)
case("find and list", finding)
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(finish())
