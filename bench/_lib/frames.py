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


#: The most a part of a one-a-second copy may hold, in bytes. **Inherited** from the outside reviewer's video reader,
#: which takes a file up to 100 MB (internal/gemini/2026-10-15_clips_review.md, section 5); a fifth under it as room.
ONE_A_SECOND_BYTES = 80_000_000
#: What a one-a-second frame costs, in bytes a pixel of the canvas, at the encoder settings below. **Measured** on the
#: three review clips of 2026-10-15 (0.079, 0.085 and 0.088 a pixel at crf 17-18): the largest, with room.
ONE_A_SECOND_COST = 0.1


class OneASecond:
    """The same frames as a review clip, at one a second and with no sound, for a video reader that takes one frame a
    second (the outside reviewer's, C1b, 2026-10-15), in parts no larger than its reader takes: each part named
    `<stem>.1fps.<first>-<last>.mp4` by the source frames it holds. `write(frame_bytes, source_frame)` a frame at a
    time; `close()` gives each part as {file, frames, bytes}. How many frames a part holds is reckoned from the
    canvas's size (`ONE_A_SECOND_COST`); a part that still comes out over `ONE_A_SECOND_BYTES` is said so."""

    def __init__(self, stem: Path | str, size: tuple[int, int], meta: list[str] | None = None):
        self.stem, self.size, self.meta = Path(stem), size, list(meta or [])
        self.per_part = max(1, int(ONE_A_SECOND_BYTES / (ONE_A_SECOND_COST * size[0] * size[1])))
        self.proc, self.first, self.last, self.count, self.parts, self.code = None, None, None, 0, [], 0

    def _open(self, source_frame: int) -> None:
        self.first, self.count = source_frame, 0
        self.tmp = self.stem.with_name(f"{self.stem.name}.1fps.part.mp4")
        self.proc = subprocess.Popen(["nice", "-n", "19", "ffmpeg", "-v", "error", "-y", "-threads", "2", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                      "-s", f"{self.size[0]}x{self.size[1]}", "-r", "1", "-i", "-", "-c:v", "libx264", "-crf", "18",
                                      "-preset", "veryfast", "-pix_fmt", "yuv420p", *self.meta, "-movflags", "+faststart", str(self.tmp)],
                                     stdin=subprocess.PIPE)

    def _shut(self) -> int:
        self.proc.stdin.close()
        code = self.proc.wait()
        name = self.stem.with_name(f"{self.stem.name}.1fps.{self.first:05d}-{self.last:05d}.mp4")
        self.tmp.replace(name)
        size = name.stat().st_size
        self.parts.append({"file": str(name), "frames": [self.first, self.last], "bytes": size})
        if size > ONE_A_SECOND_BYTES * 1.25:
            print(f"{name.name} is {size} bytes, over the {int(ONE_A_SECOND_BYTES * 1.25)} a reader takes", flush=True)
        self.proc = None
        return code

    def write(self, data: bytes, source_frame: int) -> None:
        if self.proc is not None and self.count >= self.per_part:
            self.code = self.code or self._shut()
        if self.proc is None:
            self._open(source_frame)
        self.proc.stdin.write(data)
        self.last, self.count = source_frame, self.count + 1

    def close(self) -> tuple[int, list[dict]]:
        """The first encoder's exit that was not 0, else 0; and the parts."""
        if self.proc is not None:
            self.code = self.code or self._shut()
        return self.code, self.parts
