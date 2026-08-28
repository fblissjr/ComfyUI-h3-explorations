#!/usr/bin/env python3
"""Spectral centroid over clip TIME, for any set of rendered H3 audio tracks.

## Why a within-clip trajectory rather than another scalar

Every audio number this lane has produced -- including
`bench/analyze_pdd_stream_energy.py`'s energy collapse -- is ONE scalar per arm.
A scalar cannot see a symptom that develops over the clip, and the owner's
description of one render is exactly that shape: a tin-can quality whose pitch
rises toward the end.

That distinction matters beyond one clip. A partition-coarseness statistic is
also one number per arm, so it can imitate any scalar ordering -- which is
precisely how integrated drift turned out to be unidentifiable from plain
coarseness (`bench/results/2026-08-28_pdd_stream_energy.json`). **A within-clip
trajectory is a shape coarseness cannot produce at all**, so it is the first
observable in this lane that could attribute anything to a mechanism rather
than to how coarse the schedule is.

## What it reports

Per file: spectral centroid per frame, then the centroid's first-quarter mean,
last-quarter mean, and a least-squares slope in Hz per second. Plus the
high-over-low band energy ratio on the same schedule, because centroid and
band ratio fail differently and agreeing is worth more than either.

## The confound this cannot resolve on its own, stated up front

**A rising centroid is what a DECAYING LOW-FREQUENCY component looks like.**
The energy collapse already measured is broadband loss; if it is not spectrally
flat -- and loss rarely is -- it produces a centroid trend by itself, with no
separate mechanism. So a tilt here is not evidence of a new phenomenon until it
survives normalising for the energy trend, which is why `rms_slope` is reported
beside `centroid_slope` and never omitted. Read them together or not at all.

Stereo is de-interleaved rather than treated as one stream: these files are
32 kHz two-channel, and reading interleaved samples as a waveform doubles the
apparent frequency of everything.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402

#: 2048 at 32 kHz is 64 ms, short enough to resolve a trend across a 15 s clip
#: into ~470 points and long enough that the lowest musical band is represented.
NFFT, HOP = 2048, 1024

#: The split for the band-energy ratio, in Hz. 2 kHz sits above speech
#: fundamentals and below the range a "tin can" description points at, so the
#: ratio moves for the thing being looked for rather than for pitch.
BAND_SPLIT = 2000.0


def decode(path):
    """(samples, rate) as mono float64, de-interleaved first."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels",
         "-of", "json", str(path)], capture_output=True, check=True, text=True)
    st = json.loads(probe.stdout)["streams"][0]
    rate, ch = int(st["sample_rate"]), int(st["channels"])
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le", "-acodec",
         "pcm_f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    if ch > 1:
        x = x[:len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    return x, rate


def frames(x, rate):
    """Per-frame centroid (Hz), band ratio, and rms, plus frame times (s)."""
    win = np.hanning(NFFT)
    freqs = np.fft.rfftfreq(NFFT, 1.0 / rate)
    lo = freqs < BAND_SPLIT
    cent, ratio, rms, t = [], [], [], []
    for i in range(0, len(x) - NFFT, HOP):
        seg = x[i:i + NFFT]
        mag = np.abs(np.fft.rfft(seg * win))
        p = mag ** 2
        tot = p.sum()
        if tot <= 1e-20:
            continue
        cent.append(float((freqs * p).sum() / p.sum()))
        low = p[lo].sum()
        ratio.append(float(p[~lo].sum() / low) if low > 1e-20 else np.nan)
        rms.append(float(np.sqrt((seg ** 2).mean())))
        t.append(i / rate)
    return (np.array(t), np.array(cent), np.array(ratio), np.array(rms))


def describe(path):
    x, rate = decode(path)
    t, cent, ratio, rms = frames(x, rate)
    if len(t) < 8:
        return None
    q = len(t) // 4
    slope = float(np.polyfit(t, cent, 1)[0])
    # In dB per second, so it is comparable to the energy numbers elsewhere.
    nz = rms > 1e-12
    rms_slope = (float(np.polyfit(t[nz], 20.0 * np.log10(rms[nz]), 1)[0])
                 if nz.sum() > 8 else float("nan"))
    return {
        "seconds": float(t[-1]),
        "centroid_first_quarter_hz": float(cent[:q].mean()),
        "centroid_last_quarter_hz": float(cent[-q:].mean()),
        "centroid_slope_hz_per_s": slope,
        "band_ratio_first_quarter": float(np.nanmean(ratio[:q])),
        "band_ratio_last_quarter": float(np.nanmean(ratio[-q:])),
        "rms_slope_db_per_s": rms_slope,
        "rms_overall": float(np.sqrt((x ** 2).mean())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*",
                    help="audio files. Default: every arm in carryprobe/.")
    args = ap.parse_args()
    paths = [Path(p) for p in args.paths]
    if not paths:
        paths = sorted((h3_config.output_dir() / "carryprobe").glob("*_a_*.flac"))
    if not paths:
        raise SystemExit(
            "no audio files given and none found in carryprobe/. Pass paths, "
            "or set H3_OUTPUT_DIR to the directory the server writes to.")

    rows = {}
    print(f"{'file':<26}{'cent Q1':>9}{'cent Q4':>9}{'Hz/s':>9}"
          f"{'hi/lo Q1':>10}{'hi/lo Q4':>10}{'dB/s':>8}")
    print("-" * 81)
    for p in paths:
        d = describe(p)
        if d is None:
            print(f"{p.name:<26} too short to trend")
            continue
        rows[p.name] = d
        print(f"{p.name:<26}{d['centroid_first_quarter_hz']:>9.0f}"
              f"{d['centroid_last_quarter_hz']:>9.0f}"
              f"{d['centroid_slope_hz_per_s']:>9.1f}"
              f"{d['band_ratio_first_quarter']:>10.4f}"
              f"{d['band_ratio_last_quarter']:>10.4f}"
              f"{d['rms_slope_db_per_s']:>8.2f}")

    print("\nRead centroid_slope BESIDE rms_slope. A clip losing low-frequency "
          "energy over time\nshows a rising centroid with no separate "
          "mechanism -- that is the energy collapse\nalready measured, seen "
          "through a different lens, not a new phenomenon.")

    out = REPO / "bench/results/2026-08-28_audio_spectral_tilt.json"
    out.write_text(json.dumps({
        "date": "2026-08-28",
        "script": "bench/analyze_audio_spectral_tilt.py",
        "nfft": NFFT, "hop": HOP, "band_split_hz": BAND_SPLIT,
        "files": rows,
        "do_not_rely_on": [
            "A rising centroid is also what a decaying low-frequency "
            "component produces. This does not separate the two on its own; "
            "rms_slope is reported beside it for that reason and the two must "
            "be read together.",
            "Centroid is computed on the channel MEAN of a 32 kHz stereo "
            "file. A correlated-noise difference between channels would not "
            "show here.",
            "No clip was judged by a person, and no perceptual claim is "
            "licensed by any number in this file.",
        ],
    }, indent=1) + "\n")
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
