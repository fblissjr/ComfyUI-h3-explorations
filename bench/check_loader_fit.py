#!/usr/bin/env python3
"""`bench/_lib/frames.py::FIT` against what the graphs' video loader does to a source that is not the canvas's shape.

Every tracked tool that sets a render beside its source reads the source through `FIT`. Until 2026-10-11 `FIT`
scaled the source to cover the canvas and then cut the centre, while the loader cuts the source to the canvas's
shape and then scales it. For a source already the canvas's shape the two are the same picture, and no check
used any other, so it stood. On that day's video (1424 by 1080 at 1024 by 768) the two sit about half a row
apart: every capture read its source that far off, and it showed only when a pose model run through the
tracked reader did not reproduce the points of a one-off that had cut first.

1. **Cut first, then scale: on a source wider than the canvas's shape and on one taller.** `FIT`'s frames are,
   pixel for pixel, those of a filter written here by hand with the cut worked out in whole numbers, and are
   NOT those of the old order. Both clips are sized so that the two orders sample the picture differently, as
   the real source does; a source exactly twice the canvas would not tell them apart.
2. **The tracked expression is the loader's own line.** The loader's file is read as code and its `crop=`
   expression rebuilt; it is `frames.LOADER_CROP`, character for character, it comes before the loader's
   `scale=`, and `FIT` is that expression with the ratio written as the canvas's width over its height, then a
   scale. A change in the loader goes red here and not in a render. Not graded where the loader is not
   installed.
3. **A source already the canvas's shape is only scaled, and one the canvas's size is untouched.**
4. **One place.** No tracked file under `bench/` or `workflows/` holds the old order's string or writes the
   loader's crop expression out again: `bench/masked_render_against_source.py`, which kept a copy of the old
   order, reads `FIT`, and `bench/assemble_delivery.py` and `bench/patch_render_window.py`, which each kept a copy
   of the right one, build their filter from `LOADER_CROP`. This case found the second of those two the first
   time it ran.

Needs ffmpeg and ffprobe on the path. No model, no card, no server.

    <comfy venv python> bench/check_loader_fit.py
"""
from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import BENCH, COMFY, REPO, case, finish, needs, skip  # noqa: E402

needs("ffmpeg and ffprobe on the path", bool(shutil.which("ffmpeg") and shutil.which("ffprobe")))

from _lib import frames as fr  # noqa: E402

WORK = Path(tempfile.mkdtemp(prefix="check_loader_fit_"))
#: The order `FIT` had until 2026-10-11, kept here only to show the two orders are told apart.
OLD = "scale={w}:{h}:force_original_aspect_ratio=increase:flags=bicubic,crop={w}:{h}"
#: (name, the clip's size, the canvas, the cut as whole numbers: width, height, left, top). Worked out by hand
#: from the loader's rule: keep the side that is too short whole and cut the other to the canvas's shape, from the
#: middle. ffmpeg takes the cut's size to the nearest whole pixel, and its offset as half of what the size BEFORE
#: rounding leaves, to the nearest whole pixel. That last part was found on 2026-10-11 by trying the neighbouring
#: cuts and from ffmpeg's own log line for the filter: the first numbers written here, with every part of a pixel
#: dropped, were wrong for the second clip.
CLIPS = (("wider than the canvas's shape", (142, 100), (64, 48), (133, 100, 4, 0)),      # 100 * 64/48 = 133.33; (142 - 133.33) / 2 = 4.33
         ("taller than the canvas's shape", (178, 135), (128, 96), (178, 134, 0, 1)))    # 178 * 96/128 = 133.5; (135 - 133.5) / 2 = 0.75


def clip(path: Path, size: tuple[int, int], n: int = 3) -> str:
    """A lossless grey clip whose picture changes smoothly both ways and has hard blocks, so half a pixel shows."""
    w, h = size
    y, x = np.mgrid[0:h, 0:w]
    frames = []
    for k in range(n):
        p = 128 + 55 * np.sin((x + 3 * k) / 6.0) + 55 * np.cos(y / 4.5)
        p[(x // 9 + y // 7) % 5 == 0] = 20 + 10 * k
        frames.append(np.clip(p, 0, 255).astype(np.uint8))
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                            "-c:v", "ffv1", "-level", "3", str(path)], stdin=subprocess.PIPE)
    enc.communicate(np.stack(frames).tobytes())
    assert enc.returncode == 0, path
    return str(path)


def read(path: str, size: tuple[int, int], vf: str) -> np.ndarray:
    return np.stack(list(fr.stream(path, size, 0, 3, vf=vf)))


def cut_first_then_scale() -> str:
    said = []
    for name, src, canvas, (cw, ch, cx, cy) in CLIPS:
        path = clip(WORK / f"{src[0]}x{src[1]}.mkv", src)
        w, h = canvas
        fit = read(path, canvas, fr.FIT.format(w=w, h=h))
        by_hand = read(path, canvas, f"crop={cw}:{ch}:{cx}:{cy},scale={w}:{h}")
        old = read(path, canvas, OLD.format(w=w, h=h))
        assert fit.shape == by_hand.shape == (3, h, w), (name, fit.shape)
        assert np.array_equal(fit, by_hand), f"{name}: FIT is not the cut {cw}x{ch} at ({cx}, {cy}) then a scale; largest difference {int(np.abs(fit.astype(int) - by_hand).max())}"
        apart = float(np.abs(fit.astype(int) - old).mean())
        assert apart > 1.0, f"{name}: FIT gives the picture the old order gave ({apart:.2f} grey levels apart): this clip cannot tell the two"
        said.append(f"{name} ({src[0]}x{src[1]} to {w}x{h}): the cut by hand to the pixel, {apart:.1f} grey levels from the old order")
    return "; ".join(said)


def the_loaders_own_line() -> str:
    # ComfyUI's own files are links in a copy made by bench/break_checks.py: follow one to the real checkout
    loader = (COMFY / "main.py").resolve().parent / "custom_nodes" / "ComfyUI-VideoHelperSuite" / "videohelpersuite" / "load_video_nodes.py"
    if not loader.exists():
        skip("ComfyUI-VideoHelperSuite is not installed beside this pack: the loader's own line cannot be read")
    text = loader.read_text()
    found = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.JoinedStr) and node.values and isinstance(node.values[0], ast.Constant) and str(node.values[0].value).startswith(("crop=", "scale=")):
            made = "".join(str(v.value) if isinstance(v, ast.Constant) else "{" + ast.unparse(getattr(v, "value")) + "}" for v in node.values)
            found.append((node.lineno, made))
    crops = [(n, t) for n, t in found if t.startswith("crop=")]
    assert len(crops) == 1, f"the loader has {len(crops)} crop expressions where one was read: {crops}"
    line, crop = crops[0]
    assert crop == fr.LOADER_CROP, f"the loader's crop (line {line}) is no longer frames.LOADER_CROP:\n  loader  {crop}\n  tracked {fr.LOADER_CROP}"
    after = [(n, t) for n, t in found if t.startswith("scale=") and n > line]
    assert after and after[0][0] - line <= 3 and after[0][1] == "scale={size_arg}", f"the loader no longer scales straight after its crop: {found}"
    assert "ar = float(size[0])/float(size[1])" in text, "the loader's ratio is no longer the target's width over its height"
    assert fr.FIT == fr.LOADER_CROP.format(ar="({w}/{h})") + ",scale={w}:{h}", fr.FIT
    return f"the loader's crop expression (line {line} of its file) is frames.LOADER_CROP character for character, its scale follows it, and FIT is the two with the ratio as w over h"


def same_shape_is_only_scaled() -> str:
    path = clip(WORK / "same_shape.mkv", (128, 96))
    assert np.array_equal(read(path, (64, 48), fr.FIT.format(w=64, h=48)), read(path, (64, 48), "scale=64:48"))
    whole = read(path, (128, 96), fr.FIT.format(w=128, h=96))
    assert np.array_equal(whole, np.stack(list(fr.stream(path, (128, 96), 0, 3)))), "a source the canvas's size came back changed"
    return "a 4:3 source at a 4:3 canvas is a plain scale, and at its own size comes back untouched"


def one_place() -> str:
    held = []
    for f in sorted(list(BENCH.glob("*.py")) + list(BENCH.glob("_lib/*.py")) + list((REPO / "workflows").glob("*.py"))):
        if f.name != Path(__file__).name and "force_original_aspect_ratio=increase" in f.read_text():
            held.append(str(f.relative_to(REPO)))
    assert not held, f"the old order's string is still in: {held}"
    copies = [str(f.relative_to(REPO)) for f in sorted(list(BENCH.glob("*.py")) + list((REPO / "workflows").glob("*.py")))
              if f.name != Path(__file__).name and "crop=if(gt(" in f.read_text()]
    assert not copies, f"the loader's crop expression is written out again in: {copies} (it is frames.LOADER_CROP)"
    for name, how in (("assemble_delivery.py", "LOADER_CROP.format(ar=float(w) / float(h))"), ("patch_render_window.py", "LOADER_CROP.format(ar=ar)")):
        assert how in (BENCH / name).read_text(), f"bench/{name} does not build its fit from LOADER_CROP"
    older = (BENCH / "masked_render_against_source.py").read_text()
    assert "from _lib.frames import FIT" in older and '"crop": FIT,' in older, "bench/masked_render_against_source.py does not read FIT"
    users = sorted(f.name for f in BENCH.glob("*.py") if "FIT" in f.read_text() and "_lib.frames import" in f.read_text() and not f.name.startswith("check_"))
    assert {"capture_masked_run.py", "frame_sheet.py", "recomposite_window.py", "masked_render_against_source.py"} <= set(users), users
    return f"no tracked file holds the old order or a copy of the loader's crop; {len(users)} tools read the one FIT: " + ", ".join(users) + "; the delivery and the patch tool build theirs from LOADER_CROP"


case("cut first, then scale: wider and taller", cut_first_then_scale)
case("the tracked expression is the loader's own line", the_loaders_own_line)
case("a source already the canvas's shape is only scaled", same_shape_is_only_scaled)
case("one place", one_place)
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(finish())
