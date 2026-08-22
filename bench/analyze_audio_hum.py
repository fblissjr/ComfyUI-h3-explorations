#!/usr/bin/env python3
"""Measure the narrowband hum in rendered clips, per arm.

The owner reported a machine hum through every clip rendered on 2026-08-22
after the Sol kernel swap, with one exception. A hum is a narrowband tone, so
it is measurable: it shows as a spectral peak standing far above its own
neighbourhood, and it either is or is not there.

**What is measured.** Welch PSD of the mono-mixed audio track, in dB, minus a
median-filtered copy of itself. That difference is PROMINENCE: how far a peak
rises above the local noise floor, which is what "a tone you can hear over the
scene" means. An absolute level would not do -- a loud scene and a quiet one
have different floors and the same hum.

**Why prominence at fixed frequencies rather than a peak search.** The peak
search was run first and found the fundamental near 400 Hz with harmonics at
800 and 1600. Fixing the bands after that makes the number comparable across
clips, including clips where the hum is absent and a search would return
whatever noise happened to be tallest.

**What this CANNOT say.** Which stage produces it. The audio VAE, the DiT and
the attention kernel are all upstream of this file and it sees only the mixed
output. It also cannot attribute a difference between two clips to any single
knob when the clips come from renders that differed in more than one -- read
`--group` labels as descriptions, not as arms of a controlled experiment.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

#: Fundamental and harmonics, Hz. Found by peak search, then fixed.
BANDS = (400, 800, 1600)
HALFWIDTH = 25.0


def prominences(path: pathlib.Path) -> dict[int, float] | None:
    """Peak prominence in dB at each band, or None if there is no audio."""
    with tempfile.TemporaryDirectory() as td:
        wav = pathlib.Path(td) / "a.wav"
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(path),
             "-ac", "1", "-ar", "32000", "-f", "wav", str(wav)],
            capture_output=True)
        if r.returncode != 0 or not wav.exists():
            return None
        sr, x = wavfile.read(wav)
    x = x.astype(np.float64) / 32768.0
    if x.size < 8192:
        return None
    f, pxx = signal.welch(x, sr, nperseg=8192)
    db = 10 * np.log10(pxx + 1e-20)
    prom = db - signal.medfilt(db, 51)
    out = {}
    for b in BANDS:
        band = (f > b - HALFWIDTH) & (f < b + HALFWIDTH)
        out[b] = float(prom[band].max())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True, metavar="LABEL=GLOB",
                    help="a named set of clips; repeatable")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    groups: dict[str, dict] = {}
    for spec in args.group:
        label, _, pattern = spec.partition("=")
        paths = sorted(pathlib.Path("/").glob(pattern.lstrip("/")))
        rows = []
        for p in paths:
            pr = prominences(p)
            if pr is None:
                print(f"  no audio track: {p.name}", file=sys.stderr)
                continue
            rows.append({"clip": p.name, **{str(k): round(v, 1) for k, v in pr.items()}})
        if not rows:
            print(f"group {label!r} matched no clips with audio", file=sys.stderr)
            return 1
        groups[label] = {
            "n": len(rows),
            "clips": rows,
            **{f"median_{b}Hz_db": round(statistics.median(r[str(b)] for r in rows), 1)
               for b in BANDS},
        }

    print(f"{'group':22s} {'n':>3s} " + " ".join(f"{b:>8d}Hz" for b in BANDS))
    for label, g in groups.items():
        cells = " ".join(f"{g[f'median_{b}Hz_db']:>10.1f}" for b in BANDS)
        print(f"{label:22s} {g['n']:>3d} {cells}")
    print("\ndB of peak prominence above the local noise floor; median per group.")
    print("Higher is a stronger tone. These are DESCRIPTIONS of groups of")
    print("renders, not arms of a controlled experiment -- check what else")
    print("differed between any two before attributing a gap to one knob.")

    if args.out:
        args.out.write_text(json.dumps(
            {"bands_hz": list(BANDS), "halfwidth_hz": HALFWIDTH,
             "metric": "welch PSD dB minus median-filtered self, peak in band",
             "groups": groups}, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
