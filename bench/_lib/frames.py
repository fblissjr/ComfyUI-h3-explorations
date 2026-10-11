"""Frames and masks in and out of video files, by frame number, for the tools that read renders.

    from _lib.frames import FIT, probe, stream, read_mask, write_mask_video

One reader for every bench tool that sets a render beside its source. It picks frames by NUMBER and never by
time (a start time once landed a window two frames before the one it named), passes them through with no
frame repeated or dropped to fill a gap, and fits a source to a canvas the way the lane's loader does. Every
session that looked at a render had written its own; `bench/capture_masked_run.py` and `bench/frame_sheet.py`
share this one.

Nothing here imports torch or ComfyUI. It needs `ffmpeg` and `ffprobe` on the path.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

#: The crop the graphs' video loader makes before it scales, as its own code writes it, with the canvas's aspect
#: ratio as `{ar}`. Inherited, word for word: `ComfyUI-VideoHelperSuite/videohelpersuite/load_video_nodes.py`,
#: the line under "Aspect ratio is changed. Crop to new aspect ratio before scale", read 2026-10-11.
#: `bench/check_loader_fit.py` holds this text to that line, so a change in the loader goes red there.
LOADER_CROP = "crop=if(gt({ar}\\,a)\\,iw\\,ih*{ar}):if(gt({ar}\\,a)\\,iw/{ar}\\,ih)"
#: The loader's fit: CUT the source to the canvas's shape at its centre, THEN scale it. Formatted with `w` and
#: `h`, the canvas's size; the loader's ratio is written as their quotient, which ffmpeg works out to the same
#: number. For a source already the canvas's shape the cut removes nothing and this is a plain scale.
#:
#: Until 2026-10-11 this was the other order (scale to cover, then cut the centre), said to have been read off
#: the same loader. The two differ whenever the source's shape is not the canvas's: on a 1424 by 1080 source at
#: 1024 by 768 they sit about half a row apart. Found three ways that day: the loader's code; two renders'
#: own pixels outside their redrawn region, which sit nearer a source cut first; and the pose model, whose
#: points on a source fitted the old way were half a pixel from those of a run that cut first, and identical
#: once the fit was changed. The loader also rounds a custom size to a multiple of eight, which this does not:
#: give it the canvas's own size.
FIT = LOADER_CROP.format(ar="({w}/{h})") + ",scale={w}:{h}"
#: The rate a mask video is written at. Inherited: the lane renders at 24 frames a second.
FPS = 24


def probe(path: str) -> tuple[int, int, int]:
    """(width, height, frames) of a video's first video stream, the frames counted and not read off a header."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=width,height,nb_read_packets", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip().split(",")
    return int(out[0]), int(out[1]), int(out[2])


def stream(path: str, size: tuple[int, int], first: int = 0, count: int | None = None, vf: str = "", pix: str = "gray"):
    """Frames of `path` from frame `first`, picked by number, after `vf`, at `size` (w, h).

    `pix` is `gray` (frames [h, w]) or `rgb24` ([h, w, 3]). The filter must leave frames at `size`: give
    `FIT.format(w=w, h=h)` for a source, a `scale=` for anything else that is not already that size.

    `gray` is ffmpeg's own conversion, which widens a tv-range picture to full range. Two files read here
    are compared like with like, but a grey level from this reader is a full-range one: never set it beside
    a figure read straight from a file's planes (found 2026-10-10, when a quarter of untouched footage read
    as changed that way; `bench/assemble_delivery.py::luma_at_canvas` reads a file in its own levels)."""
    w, h = size
    c = 1 if pix == "gray" else 3
    chain = ([f"select=gte(n\\,{int(first)})"] if first else []) + ([vf] if vf else [])
    cmd = ["nice", "-n", "19", "ffmpeg", "-v", "error", "-threads", "2", "-i", path, "-map", "0:v:0"]
    cmd += (["-vf", ",".join(chain)] if chain else []) + ["-fps_mode", "passthrough"]
    cmd += (["-frames:v", str(int(count))] if count is not None else []) + ["-f", "rawvideo", "-pix_fmt", pix, "-"]
    # stderr is dropped: a reader that stops early (a shorter clip beside a longer one) makes ffmpeg complain
    # about the closed pipe, which is not an error of the read
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * c)
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
    """A saved mask video as [frames, h, w] of bool on a span's clock, and which of its frames the video covers.

    The video's frame 0 is source frame `at`; the span starts at source frame `first`. A span frame the video
    does not reach stays empty and is marked not covered, so an empty mask and no mask are never the same."""
    w, h = size
    mask, covered = np.zeros((frames, h, w), bool), np.zeros(frames, bool)
    skip, lead = max(first - at, 0), max(at - first, 0)
    for n, frame in enumerate(stream(path, size, skip, max(frames - lead, 0), vf=f"scale={w}:{h}:flags=neighbor")):
        mask[lead + n], covered[lead + n] = frame > 127, True
    return mask, covered


def write_mask_video(path: Path | str, mask: np.ndarray) -> None:
    """[n, h, w] of bool as a lossless grey video, white on the mask: what a video loader and ImageToMask read back."""
    n, h, w = mask.shape
    enc = subprocess.Popen(["nice", "-n", "19", "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                            "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "ffv1", "-level", "3", str(path)],
                           stdin=subprocess.PIPE)
    for i in range(0, n, 32):
        enc.stdin.write((mask[i:i + 32].astype(np.uint8) * 255).tobytes())
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError(f"ffmpeg could not write {path}")
