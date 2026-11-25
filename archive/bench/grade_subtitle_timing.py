#!/usr/bin/env python3
"""Measure burned-in subtitle spans and vocal-burst structure in a rendered clip.

    python bench/grade_subtitle_timing.py CLIP.mp4 [CLIP2.mp4 ...]
    python bench/grade_subtitle_timing.py --expect-no-subtitles CLIP.mp4
    python bench/grade_subtitle_timing.py --json CLIP.mp4

**A report, not a gate.** It answers "what happened in this clip", and there is
no repo state it could be right or wrong about: the correct number of subtitle
spans depends on the prompt, and their exact timing depends on the sampler. So
it lives with `grade_*` / `analyze_*` and exits 0 on any readable file. The one
exception is `--expect-no-subtitles`, which IS an assertion and exits 1 when it
finds text -- that flag exists so this file can be shown red (see below).

## Why it exists, and the escaped instance

Written 2026-08-23. `<|caption_start|>` is declared by the release, appears in
neither official prompt guide, and its embedding row is untrained. Whether it
did anything was argued from prose for two days. What settled it was counting:
in a controlled pair differing ONLY by three caption markers, the marker arm
produced twice the vocal bursts in the window where one line belonged, and a
fourth subtitle for a three-line prompt. Neither fact is visible by watching --
"it sounded doubled" is exactly the claim this repo does not accept -- and
neither is visible to any static check, because both are properties of a
rendered clip.

## What the numbers are, and what they are NOT

**Subtitle presence** is the count of near-white pixels in the lower third,
thresholded. A count rather than a mean, deliberately: a bright desk raises the
mean and not the count.

**THE SCAR.** The first version of this scaled the band to 160x48 before
counting and reported a peak of 4 white pixels on a clip whose subtitles are
plainly legible -- downscaling averages thin glyphs into the background. It ran
clean, produced a number, and the number was garbage. It was caught only by
disbelieving the output, so the band is now read at native resolution and the
peak is printed on every run for exactly that reason: a peak in the low tens
means the detector found nothing, whatever the span list says.

**Vocal bursts** are per-frame audio RMS over a threshold. This is loud-event
detection, NOT speech segmentation: a palm-slap, a door, and a keyboard all
register. It cannot tell you WHICH WORDS or WHICH LANGUAGE is spoken, so it
cannot by itself establish "the line was delivered twice, once per language" --
that claim needs a person to listen. What it establishes is the weaker,
checkable thing: how many separated loud events sit in a window, and for how
long. Both are printed with this caveat attached.

Neither number supports a perceptual claim about which clip looks or sounds
better. CLAUDE.md's rule is unchanged: two arms differing in a knob are two
different samples, and a preference between them needs a distribution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

try:
    import numpy as np
except ImportError:
    print("needs numpy: uv pip install numpy", file=sys.stderr)
    raise SystemExit(2)

FPS = 24
WHITE = 235          # a burned-in subtitle here is white text with a dark edge
MIN_SPAN_FRAMES = 3  # below this, a "span" is a flicker, not a caption
MIN_BURST_FRAMES = 3


def _probe(path: str) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        raise SystemExit(f"cannot read {path}: {out.stderr.strip()[:200]}")
    w, h = out.stdout.strip().split(",")[:2]
    return int(w), int(h)


def caption_band(path: str) -> np.ndarray:
    """Per-frame count of near-white pixels in the lower third, native res.

    Native resolution is load-bearing; see THE SCAR in the module docstring.
    """
    w, h = _probe(path)
    band_h = h // 3
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf",
         f"crop=in_w:in_h/3:0:in_h*2/3,format=gray", "-f", "rawvideo", "-"],
        capture_output=True).stdout
    px = band_h * w
    if px == 0 or len(raw) < px:
        raise SystemExit(f"no video frames read from {path}")
    frames = np.frombuffer(raw[:(len(raw) // px) * px], np.uint8)
    return (frames.reshape(-1, band_h, w) > WHITE).sum(axis=(1, 2))


def audio_rms(path: str, sr: int = 16000) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr),
         "-f", "s16le", "-"], capture_output=True).stdout
    if not raw:
        return np.zeros(0, dtype=np.float32)
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    n = sr // FPS
    usable = (len(a) // n) * n
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt((a[:usable].reshape(-1, n) ** 2).mean(axis=1))


def spans(mask: np.ndarray, min_frames: int) -> list[tuple[float, float]]:
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            if j - i + 1 >= min_frames:
                out.append((i / FPS, (j + 1) / FPS))
            i = j + 1
        else:
            i += 1
    return out


def grade(path: str, window: tuple[float, float] | None = None) -> dict:
    white = caption_band(path)
    rms = audio_rms(path)
    n = min(len(white), len(rms)) if len(rms) else len(white)
    white = white[:n]
    peak = int(white.max()) if n else 0
    # Relative to this clip's own peak: absolute pixel counts scale with canvas.
    # The floor keeps a clip with NO subtitles from thresholding on its noise.
    thr = max(150, int(peak * 0.12))
    subs = spans(white > thr, MIN_SPAN_FRAMES)

    bursts, voiced = [], 0.0
    if len(rms):
        rms = rms[:n]
        athr = max(0.02, float(rms.max()) * 0.18)
        loud = rms > athr
        bursts = spans(loud, MIN_BURST_FRAMES)
        voiced = float(loud.sum()) / FPS

    res = {
        "clip": path.split("/")[-1],
        "frames": n,
        "white_peak": peak,
        "white_threshold": thr,
        "detector_found_text": peak > 150,
        "subtitle_spans": [[round(a, 2), round(b, 2)] for a, b in subs],
        "subtitle_on_seconds": round(sum(b - a for a, b in subs), 2),
        "vocal_bursts": [[round(a, 2), round(b, 2)] for a, b in bursts],
        "voiced_seconds": round(voiced, 2),
    }
    if window:
        lo, hi = window
        res["window"] = [lo, hi]
        res["bursts_in_window"] = sum(1 for a, b in bursts if a >= lo and b <= hi)
        res["subs_in_window"] = sum(1 for a, b in subs if a < hi and b > lo)
    return res


def report(r: dict) -> None:
    print(f"=== {r['clip']}  ({r['frames']} frames)")
    print(f"  white-pixel peak {r['white_peak']} (threshold {r['white_threshold']})"
          f"{'' if r['detector_found_text'] else '   <- NO TEXT DETECTED'}")
    if not r["detector_found_text"]:
        print("      a peak this low means the detector found nothing. If the "
              "clip visibly has\n      subtitles, the detector is wrong, not "
              "the clip -- see THE SCAR in this file.")
    print(f"  subtitle spans ({r['subtitle_on_seconds']}s total):")
    for a, b in r["subtitle_spans"]:
        print(f"      {a:6.2f} -> {b:6.2f}   ({b - a:.2f}s)")
    if not r["subtitle_spans"]:
        print("      none")
    print(f"  vocal bursts ({r['voiced_seconds']}s voiced) -- LOUD EVENTS, not "
          f"speech segmentation;")
    print(f"      impacts and props register, and language is not detectable "
          f"here:")
    for a, b in r["vocal_bursts"]:
        print(f"      {a:6.2f} -> {b:6.2f}   ({b - a:.2f}s)")
    if "window" in r:
        lo, hi = r["window"]
        print(f"  in window {lo}-{hi}s: {r['bursts_in_window']} burst(s), "
              f"{r['subs_in_window']} subtitle span(s)")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--window", nargs=2, type=float, metavar=("LO", "HI"),
                    help="count bursts and spans inside this second range")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--expect-no-subtitles", action="store_true",
                    help="ASSERT the clip carries no burned-in text; exit 1 if "
                         "it does. The deliberate-violation path: an instrument "
                         "that cannot fail is not an instrument.")
    a = ap.parse_args()

    results = [grade(c, tuple(a.window) if a.window else None) for c in a.clips]
    if a.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            report(r)

    if a.expect_no_subtitles:
        bad = [r for r in results if r["subtitle_spans"]]
        for r in bad:
            print(f"FAIL {r['clip']}: expected no burned-in text, found "
                  f"{len(r['subtitle_spans'])} span(s)", file=sys.stderr)
        if bad:
            return 1
        print("ok: no burned-in text found, as expected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
