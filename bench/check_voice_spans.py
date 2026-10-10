#!/usr/bin/env python3
"""`bench/voice_spans.py`'s frame arithmetic and span rule, on a signal made for the purpose.

A span one frame out puts a lip-sync sentence on a shot with no voice, and a cut one frame out lays a window's
audio a frame off its picture. The separation itself needs the weights and a real track and is not graded here.

1. **Levels by frame.** A tone on known video frames of a silent signal: those frames read at the tone's level,
   the others at the floor, at two frame rates.
2. **The spans.** The tone's frames come back as its span, first and last inclusive; a gap of `JOIN` frames is
   closed and one of `JOIN + 1` is not; a burst under `LEAST` frames is dropped; a signal that never passes the
   threshold has no span (the control).
3. **A cut.** Frames first to last of a stem start on their first frame and are (last - first + 1) / fps long,
   and two cuts that meet share no sample and lose none.

No model, no weights, no ffmpeg, no card.

    <comfy venv python> bench/check_voice_spans.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import voice_spans as vs  # noqa: E402

RATE = vs.RATE


def signal(frames: int, fps: float, on: list[tuple[int, int]], amp: float = 0.1) -> np.ndarray:
    """A mono signal of `frames` video frames, a 440 Hz tone at `amp` on the inclusive frame runs in `on`."""
    n = int(round(frames * RATE / fps))
    t = np.arange(n) / RATE
    out = np.zeros(n, np.float32)
    for lo, hi in on:
        a, b = int(round(lo * RATE / fps)), int(round((hi + 1) * RATE / fps))
        out[a:b] = amp * np.sin(2 * np.pi * 440 * t[a:b])
    return out


def levels() -> str:
    for fps in (24.0, 25.0):
        lv = vs.frame_levels(signal(100, fps, [(20, 39)]), RATE, fps)
        assert len(lv) == 100, len(lv)
        assert abs(float(np.median(lv[20:40])) - 20 * np.log10(0.1 / np.sqrt(2))) < 0.2, float(np.median(lv[20:40]))
        assert float(lv[:20].max()) < -100 and float(lv[40:].max()) < -100, (float(lv[:20].max()), float(lv[40:].max()))
    return "a tone on frames 20-39 reads at its level on those frames and at the floor elsewhere, at 24 and 25 a second"


def the_spans() -> str:
    fps = 24.0
    lv = vs.frame_levels(signal(200, fps, [(20, 59), (60 + vs.JOIN, 99), (100 + vs.JOIN + 1, 139), (160, 160 + vs.LEAST - 2)]), RATE, fps)
    got = vs.voiced_spans(lv)
    assert got == [[20, 99], [100 + vs.JOIN + 1, 139]], got
    assert vs.voiced_spans(vs.frame_levels(signal(100, fps, [(10, 50)], amp=0.001), RATE, fps)) == [], "a signal under the threshold was called a voice"
    return f"a gap of {vs.JOIN} frames is closed, one of {vs.JOIN + 1} is not, a burst under {vs.LEAST} frames is dropped, a quiet signal has no span"


def a_cut() -> str:
    fps = 24.0
    stereo = np.stack([signal(240, fps, [(0, 239)])] * 2)
    one, two = vs.frame_cut(stereo, 100, 149, RATE, fps), vs.frame_cut(stereo, 150, 199, RATE, fps)
    assert abs(one.shape[-1] / RATE - 50 / fps) < 1.0 / RATE, one.shape[-1] / RATE
    start = int(round(100 * RATE / fps))
    assert (one == stereo[:, start:start + one.shape[-1]]).all(), "a cut does not start on its first frame"
    assert one.shape[-1] + two.shape[-1] == vs.frame_cut(stereo, 100, 199, RATE, fps).shape[-1], "two cuts that meet lose or share a sample"
    return "a cut starts on its first frame, is its frames long, and two that meet neither overlap nor leave a gap"


case("levels by video frame", levels)
case("the span rule", the_spans)
case("a cut by frames", a_cut)
sys.exit(finish())
