#!/usr/bin/env python3
"""A copy's sound against its source's (`bench/copy_sound.py`) on made sound whose answer is known.

A minute of made sound (noise through a low filter, so it has a shape that fits only itself) stands for a
source at 24000/1001 frames a second. Copies of it are made as arrays, at 24 frames a second:

1. **A copy in step** (the sound resampled to run a thousandth fast, as the picture does) reads within a
   millisecond of its picture at every place and fits, against the source's window brought to the picture's
   speed. The made sound is broad on purpose: read against the source's window as it is, it fits at a tenth,
   which is how the first form of this tool would have called a good copy unreadable.
2. **A copy with no stretch**, the fault this tool was written on, reads late by a thousandth of the elapsed time
   at every place: the figure the tool prints beside each place for scale, and the furthest is the last. It
   fits the source's window as it is.
3. **A copy in step whose sound is a twentieth of a second late throughout** reads that at every place.
4. **A copy that begins on a later frame of the source** reads in step when told the frame, and is out or does
   not fit when not told.
5. **What cannot be read is said**: silence fits nowhere and has no furthest place; a copy too short to read and
   a copy that runs past its source's sound are refused.
6. **The caller's line** (`--refuse-over`): met by the copy in step, not met by the one with no stretch with the
   place named, and not met where no place fits.

No file, no ffmpeg, no card.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_copy_sound.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import copy_sound as cs  # noqa: E402

RATE = cs.RATE
SOURCE_FPS, COPY_FPS = 24000 / 1001, 24.0
SECONDS = 60


def made_source() -> np.ndarray:
    """A minute of noise through a moving average: broad, with no repeat, so a window fits one place only."""
    rng = np.random.default_rng(730451892)
    white = rng.standard_normal(SECONDS * RATE + 64).astype(np.float32)
    return np.convolve(white, np.ones(8, np.float32) / 8, mode="valid")[:SECONDS * RATE]


SOURCE = made_source()


def stretched(sound: np.ndarray) -> np.ndarray:
    """The sound run a thousandth fast, as a copy's picture runs: 1000 samples out for every 1001 in."""
    from scipy.signal import resample_poly
    return resample_poly(sound, 1000, 1001).astype(np.float32)


def read(copy: np.ndarray, first_frame: int = 0, source: np.ndarray = SOURCE) -> list[dict]:
    return cs.places_along(copy, source, COPY_FPS, SOURCE_FPS, first_frame, places=6, window=2.0)


def a_copy_in_step() -> str:
    rows = read(stretched(SOURCE))
    assert all(abs(r["sound_late_by_ms"]) <= 1.0 and r["fit"] >= 0.9 and r["fitted_to"] == "the source's window at the picture's speed" for r in rows), rows
    return f"a copy whose sound was resampled a thousandth fast, as its picture runs, reads within a millisecond at {len(rows)} places, fitted to the source's window at the picture's speed"


def a_copy_with_no_stretch() -> str:
    copy = SOURCE[:int(len(SOURCE) / 1.001)]                    # the source's sound as it is, cut to the picture's length
    rows = read(copy)
    for r in rows:
        want = r["copy_s"] * (1 - 1 / 1.001) * 1000             # a thousandth of the elapsed time, by the picture's clock
        assert abs(r["sound_late_by_ms"] - want) <= 1.5 and abs(r["a_copy_with_no_stretch_would_read_ms"] - want) <= 0.2 and r["fit"] >= 0.8 \
            and r["fitted_to"] == "the source's window as it is", (r, want)
    late = [r["sound_late_by_ms"] for r in rows]
    assert late == sorted(late) and late[-1] > 10 * max(late[0], 0.5) and cs.worst(rows) == rows[-1], late
    return f"a copy with the source's sound unstretched reads late by a thousandth of the elapsed time, {late[0]:+.1f} ms at the first place to {late[-1]:+.1f} at the last"


def a_copy_late_throughout() -> str:
    shift = int(0.05 * RATE)
    copy = np.concatenate([np.zeros(shift, np.float32), stretched(SOURCE)])
    rows = read(copy)
    assert all(abs(r["sound_late_by_ms"] - 50.0) <= 1.0 for r in rows), rows
    early = read(stretched(SOURCE)[shift:])
    assert all(abs(r["sound_late_by_ms"] + 50.0) <= 1.0 for r in early), early
    return "a copy in step with its sound a twentieth of a second late reads +50 ms at every place, and one that much early reads -50"


def a_copy_from_a_later_frame() -> str:
    first = 240                                                  # the copy's frame 0 is the source's frame 240
    copy = stretched(SOURCE[int(round(first / SOURCE_FPS * RATE)):])
    told = read(copy, first)
    assert all(abs(r["sound_late_by_ms"]) <= 1.0 and r["fit"] >= 0.9 for r in told), told
    untold = read(copy)
    assert all(r["fit"] < cs.FIT or abs(r["sound_late_by_ms"]) > 100 for r in untold), "a copy from a later frame read as in step without being told the frame"
    return "a copy that begins on the source's frame 240 reads in step when told so; not told, no place reads as in step"


def what_cannot_be_read() -> str:
    silent = read(np.zeros(len(SOURCE) - 1000, np.float32))
    assert all(r["fit"] == 0.0 for r in silent) and cs.worst(silent) is None, silent
    for copy, words in ((stretched(SOURCE)[:3 * RATE], "too short"), (np.concatenate([stretched(SOURCE), stretched(SOURCE)]), "is not a copy of that source")):
        try:
            read(copy)
        except cs.Refused as e:
            assert words in str(e), str(e)
        else:
            raise AssertionError(f"not refused: {words}")
    k, fit = cs.slide(SOURCE[:RATE], SOURCE[:RATE], RATE // 2)
    assert (k, round(fit, 3)) == (0, 1.0), (k, fit)
    return "silence fits nowhere and has no furthest place; a copy too short and one that runs past its source's sound are refused; a window against itself is 0 and 1"


def the_callers_line() -> str:
    good, bad = read(stretched(SOURCE)), read(SOURCE[:int(len(SOURCE) / 1.001)])
    assert cs.over_the_line(good, 0.25) is None
    why = cs.over_the_line(bad, 0.25)
    assert why and f"at copy frame {bad[-1]['copy_frame']}" in why and "over the caller's line of 0.25" in why, why
    assert cs.over_the_line(bad, 5.0) is None, "a line of five frames was not met by a copy a frame and a half out"
    none = cs.over_the_line(read(np.zeros(len(SOURCE) - 1000, np.float32)), 0.25)
    assert none and "no place fits" in none, none
    return "the caller's line is met by the copy in step, not met by the one with no stretch (the place named), and not met where nothing fits"


case("a copy in step", a_copy_in_step)
case("a copy with no stretch: the fault it was written on", a_copy_with_no_stretch)
case("a copy late, or early, throughout", a_copy_late_throughout)
case("a copy that begins on a later frame of the source", a_copy_from_a_later_frame)
case("what cannot be read is said", what_cannot_be_read)
case("the caller's line", the_callers_line)

if __name__ == "__main__":
    sys.exit(finish())
