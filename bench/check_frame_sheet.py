#!/usr/bin/env python3
"""The frame reader in `bench/_lib/frames.py` and the sheet tool built on it, against a clip made for the purpose.

A sheet that shows frame 36 and labels it 35 sends a reader to the wrong frame, and a reader that fits a source
differently from the loader compares a render with a picture it was not made from. So this writes a small
lossless clip in which frame n is a flat grey of level 10 n with a bright block at a place that depends on n,
and reads it back.

1. **By number.** Frames picked by number are those frames: the level says which. From a start frame, a count
   of frames, and a list of single frames out of order.
2. **The fit.** A source twice as wide as the canvas is cropped at its centre, not stretched: the block that sits
   at the source's centre lands at the canvas's centre at its own size.
3. **A mask.** A mask video read on a span's clock: the frames it covers carry its mask, the frames before its
   first are empty and marked not covered; written lossless and read back, it is the same mask.
4. **The sheet.** A row whose video starts at source frame F shows source frame f as its own frame f - F: the
   tile's level says so. A followed crop is centred on the mask and stays inside the canvas; an empty mask
   falls back on the whole canvas. The picture has a row per video and a tile per frame.

Needs ffmpeg and ffprobe on the path. No model, no card, no server.

    <comfy venv python> bench/check_frame_sheet.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish, needs  # noqa: E402

needs("ffmpeg and ffprobe on the path", bool(shutil.which("ffmpeg") and shutil.which("ffprobe")))

import frame_sheet as fs  # noqa: E402
from _lib import frames as fr  # noqa: E402

W, H, N = 64, 32, 12
WORK = Path(tempfile.mkdtemp(prefix="check_frame_sheet_"))


def clip(path: Path, w: int = W, h: int = H, n: int = N) -> np.ndarray:
    """A lossless grey clip: frame k is level 10 k, with a 4 by 4 block of 250 at column 2 k, and at the centre."""
    frames = np.zeros((n, h, w), np.uint8)
    for k in range(n):
        frames[k] = 10 * k
        frames[k, 2:6, 2 * k:2 * k + 4] = 250
        frames[k, h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2] = 250
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", "24",
                            "-i", "-", "-c:v", "ffv1", "-level", "3", str(path)], stdin=subprocess.PIPE)
    enc.stdin.write(frames.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0
    return frames


SRC = WORK / "clip.mkv"
MADE = clip(SRC)


def by_number() -> str:
    assert fr.probe(str(SRC)) == (W, H, N), fr.probe(str(SRC))
    got = list(fr.stream(str(SRC), (W, H), first=5, count=3))
    assert [int(g[20, 60]) for g in got] == [50, 60, 70], [int(g[20, 60]) for g in got]
    assert all((g == MADE[5 + i]).all() for i, g in enumerate(got)), "a frame read back is not the frame written"
    some = fs.grab(str(SRC), 0, [9, 2, 7], (W, H), fit=False)
    assert sorted(some) == [2, 7, 9] and all(int(some[k][20, 60, 0]) == 10 * k for k in some), {k: int(v[20, 60, 0]) for k, v in some.items()}
    return "three frames from a start, and three single frames out of order, each the frame its number says"


def the_fit() -> str:
    got = next(fr.stream(str(SRC), (H, H), first=4, count=1, vf=fr.FIT.format(w=H, h=H)))
    assert got.shape == (H, H) and int(got[20, 28]) == 40, (got.shape, int(got[20, 28]))
    ys, xs = np.nonzero(got > 200)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (H // 2 - 2, H // 2 + 1, H // 2 - 2, H // 2 + 1), (xs.min(), xs.max(), ys.min(), ys.max())
    return "a source twice the canvas's width is cropped at its centre: the centre block lands at the centre, its own size"


def a_mask() -> str:
    mask = np.zeros((N, H, W), bool)
    for k in range(N):
        mask[k, 8:16, 3 * k:3 * k + 8] = True
    path = WORK / "mask.mkv"
    fr.write_mask_video(path, mask)
    back, covered = fr.read_mask(str(path), (W, H), at=100, first=100, frames=N)
    assert covered.all() and (back == mask).all(), "a mask written lossless and read back is not the same mask"
    late, covered = fr.read_mask(str(path), (W, H), at=104, first=100, frames=N)
    assert covered.tolist() == [False] * 4 + [True] * 8 and not late[:4].any() and (late[4:] == mask[:8]).all(), covered.tolist()
    return "the same mask back; a span that starts before the video has empty, uncovered frames first"


def the_sheet() -> str:
    render = WORK / "render.mkv"
    clip(render, n=6)                               # a "render" whose frame 0 is source frame 4
    rows = [("src", fs.grab(str(SRC), 0, [4, 6, 9], (W, H), False), 0), ("ren", fs.grab(str(render), 4, [4, 6, 9], (W, H), False), 4)]
    assert int(rows[1][1][6][20, 60, 0]) == 20 and int(rows[0][1][6][20, 60, 0]) == 60, "source frame 6 of a row starting at 4 is not its frame 2"
    assert 9 in rows[1][1] and int(rows[1][1][9][20, 60, 0]) == 50
    mask = np.zeros((H, W), bool)
    mask[0:4, 60:64] = True                         # in a corner: the box must stay inside the canvas
    assert fs.follow_box(mask, 16, (W, H)) == (48, 0, 16, 16), fs.follow_box(mask, 16, (W, H))
    assert fs.follow_box(np.zeros((H, W), bool), 16, (W, H)) == (0, 0, W, H)
    picture = fs.sheet(rows, [4, 6, 9], {f: (0, 0, W, H) for f in (4, 6, 9)}, [], cell=64)
    assert picture.shape == (2 * (H + fs.LABEL), 3 * 64, 3), picture.shape
    assert int(picture[fs.LABEL + 20, 64 + 60, 0]) == 60 and int(picture[H + 2 * fs.LABEL + 20, 64 + 60, 0]) == 20, "a tile is not the frame its place says"
    assert fs.named("a.mp4@604:fit") == ("a.mp4", 604, True) and fs.named("a.mp4") == ("a.mp4", 0, False)
    return "a row starting at frame 4 shows source frame 6 as its own frame 2; a followed box stays in the canvas; a tile per frame and row"


case("frames by number", by_number)
case("the loader's fit", the_fit)
case("a mask on a span's clock", a_mask)
case("the sheet", the_sheet)
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(finish())
