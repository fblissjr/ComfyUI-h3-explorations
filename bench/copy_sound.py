#!/usr/bin/env python3
"""A working copy's sound against its source's, at the time the copy's picture says it is: is the sound in step.

    <python> bench/copy_sound.py --source SOURCE --copy COPY [--first-frame N] [--places 8] [--window 4] [--refuse-over FRAMES]

**What it buys.** A working copy of a video keeps every frame and re-stamps them to a whole frame rate, so that a
frame number is a time: the source's 24000/1001 frames a second become 24. That makes the picture run a thousandth
fast, and the sound has to be stretched by the same share or it falls behind, a little more every second. A copy
was made without the stretch (2026-10-13): its picture was held to the source's frame by frame and found equal,
nothing looked at its sound, and by its end the sound was more than six frames behind the mouths. A render from
it would have been sampled hearing the song late. This reads the sound.

**What it measures.** At `--places` places spread along the copy, a window of the copy's sound, and the source's
sound at the time the copy's PICTURE there comes from: copy time t shows source frame `first-frame + t * the
copy's rate`, which the source plays at that frame number over the source's rate. The two windows are slid
against each other and the slide that fits best (the peak of their cross-correlation, within half a second) is
how far the copy's sound is from its picture there: positive, the sound is late. The copy's window is set against
the source's twice, as it is and resampled to the speed the copy's picture runs at, and the better fit is the
place's reading; which of the two it was is printed. That word is NOT the verdict on the copy: a stretch made by
resampling fits the second, but a stretch that keeps the pitch (ffmpeg's `atempo`, which drops a sliver of
sound every so often and leaves the rest as it was) fits the first, as a copy with no stretch does. How far
the sound is from the picture, place after place, is what says whether a copy is in step. Beside it, how well they fit at
that slide (1 is the same sound; a low figure means the place is near silence or the two are not the same sound,
and the slide beside it means little; a stretch that splices the sound, as ffmpeg's `atempo` does, does not
fit at 1 even in step), and what a copy with NO stretch would read at that place, for scale. The
two frame rates are read from the files.

**What it prints, and what it refuses.** Figures, a line a place, and the worst. It judges nothing by itself: how
far is too far depends on what is rendered from the copy. `--refuse-over FRAMES` is the caller's own line: with
it the tool exits 5 when the sound is further than that many of the copy's frames from the picture at any place
that fits (a fit of `FIT` or over), and says where. A copy or a source with no sound, and a copy longer than its
source can cover, are refused outright.

**What it cannot see.** Whether the sound is the right sound in any other way (its level, its channels, a
dropout between two places). A place where the track is near silent. Which of the two files is wrong when they
disagree: it reads the copy against the source and takes the source's picture and sound to be in step.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: The sound's rate both files are read at.
RATE = 44100
#: How far the two windows are slid either way, in seconds. Reasoned: a copy of a ten-minute video with no stretch
#: is 0.6 s out at its end, so half a second covers every copy shorter than eight minutes, and a longer one is
#: said to be at the edge.
MOST_S = 0.5
#: The fit under which a place's slide is printed and not counted by `--refuse-over`. A starting value: unrelated
#: windows of music fit under 0.1, and what the copies of one video read on 2026-10-13 is in the record beside
#: this tool's index row (a stretch made by ffmpeg's `atempo` splices the sound, so a copy in step does not fit
#: its source at 1).
FIT = 0.5


class Refused(Exception):
    pass


def slide(source: np.ndarray, copy: np.ndarray, most: int) -> tuple[int, float]:
    """(samples by which `copy` is LATER than `source`, how well they fit there in 0..1): the peak of the two
    windows' cross-correlation within `most` samples either way. Two silent windows fit at 0."""
    from scipy.signal import fftconvolve
    n = min(len(source), len(copy))
    a, b = source[:n] - source[:n].mean(), copy[:n] - copy[:n].mean()
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    if n == 0 or scale < 1e-9:
        return 0, 0.0
    c = fftconvolve(b, a[::-1], mode="full")
    lo, hi = max(0, n - 1 - most), min(len(c), n + most)
    k = int(c[lo:hi].argmax()) + lo - (n - 1)
    return k, float(c[lo:hi].max() / scale)


def places_along(copy: np.ndarray, source: np.ndarray, copy_fps: float, source_fps: float, first_frame: int = 0, places: int = 8,
                 window: float = 4.0, rate: int = RATE) -> list[dict]:
    """The reading at each place: mono `copy` and `source` at `rate`, the copy's frame 0 being the source's frame
    `first_frame`. Refused when the copy's picture runs past what the source's sound covers."""
    length = len(copy) / rate
    if length < 2 * window:
        raise Refused(f"the copy's sound is {length:.1f} s: too short to read at a window of {window:g} s")
    needed = (first_frame + length * copy_fps) / source_fps
    if needed > len(source) / rate + window:
        raise Refused(f"the copy runs to the source's {needed:.1f} s by its picture and the source's sound ends at {len(source) / rate:.1f} s: "
                      "this is not a copy of that source from that frame")
    out, most = [], int(MOST_S * rate)
    # how much faster the copy's picture runs than the source's, as a ratio of whole numbers (1001 to 1000 for a
    # 24000/1001 source re-stamped to 24)
    speed = Fraction(copy_fps / source_fps).limit_denominator(100000)
    for t in np.linspace(window, length - 2 * window, places):
        at = (first_frame + t * copy_fps) / source_fps                    # when the source plays what the copy shows at t
        c = copy[int(round(t * rate)):int(round((t + window) * rate))]
        # two readings, and the one that fits better is the place's: the source's sound as it is (what a copy with
        # no stretch holds, at the source's speed), and the source's sound over the stretch of ITS time the copy's
        # picture crosses in the window, resampled to the window's length (what a copy in step holds, at the
        # picture's speed). Two windows a thousandth apart in speed drift by milliseconds across a window and stop
        # fitting wherever the sound is not all bass: made noise in step fitted its source at 0.1 read the first way
        # alone, and made noise with no stretch at 0.1 read the second way alone (`bench/check_copy_sound.py`)
        k, fit = slide(source[int(round(at * rate)):int(round((at + window) * rate))], c, most)
        runs_at = "the source's window as it is"
        if speed != 1:
            from scipy.signal import resample_poly
            span = source[int(round(at * rate)):int(round((at + window * float(speed)) * rate))]
            k2, fit2 = slide(resample_poly(span, speed.denominator, speed.numerator), c, most)
            if fit2 > fit:
                k, fit, runs_at = k2, fit2, "the source's window at the picture's speed"
        unstretched = t - (at - first_frame / source_fps)                 # a copy with the source's sound as it is: behind by this
        out.append({"copy_s": round(float(t), 2), "copy_frame": int(round(t * copy_fps)), "source_s": round(float(at), 3),
                    "sound_late_by_ms": round(k / rate * 1000, 1), "sound_late_by_frames": round(k / rate * copy_fps, 2), "fit": round(fit, 3), "fitted_to": runs_at,
                    "at_the_edge_of_the_search": abs(k) >= most - 1,
                    "a_copy_with_no_stretch_would_read_ms": round(-unstretched * 1000, 1)})
    return out


def worst(rows: list[dict]) -> dict | None:
    """The place that fits (`FIT` or over) where the sound is furthest from the picture, or None when none fits."""
    fits = [r for r in rows if r["fit"] >= FIT]
    return max(fits, key=lambda r: abs(r["sound_late_by_frames"])) if fits else None


def over_the_line(rows: list[dict], line: float) -> str | None:
    """Why the caller's line is not met, in words, or None when it is: the furthest place that fits is further from
    the picture than `line` frames, or no place fits and so nothing was held to the line."""
    w = worst(rows)
    if w is None:
        return f"--refuse-over {line:g} was asked and no place fits, so nothing was held to it"
    if abs(w["sound_late_by_frames"]) > line:
        return (f"the sound is {w['sound_late_by_frames']:+.2f} frames from the picture at copy frame {w['copy_frame']}, over the caller's line of {line:g}")
    return None


def rate_of(path: str) -> Fraction:
    """A file's video frame rate, as the file says it."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip().rstrip(",")
    if not out:
        raise Refused(f"{path} has no picture to read a frame rate from")
    return Fraction(out)


def sound_of(path: str) -> np.ndarray:
    import voice_spans
    audio = voice_spans.read_audio(path, RATE)
    if audio.shape[-1] == 0:
        raise Refused(f"{path} has no sound")
    return audio.mean(0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", required=True, help="the video the copy was made from")
    p.add_argument("--copy", required=True, help="the working copy")
    p.add_argument("--first-frame", type=int, default=0, help="the source frame the copy's frame 0 is; 0 when left out")
    p.add_argument("--places", type=int, default=8)
    p.add_argument("--window", type=float, default=4.0, help="seconds of sound read at a place")
    p.add_argument("--refuse-over", type=float, metavar="FRAMES", help="the caller's line: exit 5 when the sound is further than this from the picture at a place that fits")
    p.add_argument("--json", help="the figures as data")
    a = p.parse_args()
    try:
        copy_fps, source_fps = rate_of(a.copy), rate_of(a.source)
        rows = places_along(sound_of(a.copy), sound_of(a.source), float(copy_fps), float(source_fps), a.first_frame, a.places, a.window)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 5
    print(f"{Path(a.copy).name} against {Path(a.source).name}: the copy's picture at {copy_fps} frames a second, the source's at {source_fps}; "
          f"the copy's frame 0 is the source's frame {a.first_frame}. A window of {a.window:g} s at each of {a.places} places.")
    print("  copy time   copy frame   the sound is late by        fit    fitted to                                    (a copy with no stretch would read)")
    for r in rows:
        print(f"  {r['copy_s']:8.2f} s  {r['copy_frame']:9d}   {r['sound_late_by_ms']:+8.1f} ms {r['sound_late_by_frames']:+6.2f} frames   {r['fit']:.3f}   {r['fitted_to']:<42}   "
              f"({r['a_copy_with_no_stretch_would_read_ms']:+.1f} ms)" + ("  AT THE EDGE OF THE SEARCH: it may be further" if r["at_the_edge_of_the_search"] else "")
              + ("" if r["fit"] >= FIT else f"  fits under {FIT}: near silence, or not the same sound"))
    w = worst(rows)
    if w is None:
        print(f"no place fits at {FIT} or over: nothing here says where the sound is")
    else:
        print(f"furthest, among the places that fit: {w['sound_late_by_frames']:+.2f} frames ({w['sound_late_by_ms']:+.1f} ms) at copy frame {w['copy_frame']}. "
              "Positive is the sound after the picture. What it cannot see: a dropout between two places, the sound's level, which file is wrong.")
    if a.json:
        Path(a.json).write_text(json.dumps({"copy": Path(a.copy).name, "source": Path(a.source).name, "copy_fps": str(copy_fps), "source_fps": str(source_fps),
                                            "first_frame": a.first_frame, "window_s": a.window, "places": rows, "furthest_that_fits": w}, indent=1) + "\n")
    why = over_the_line(rows, a.refuse_over) if a.refuse_over is not None else None
    if why:
        print(f"REFUSED: {why}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
