#!/usr/bin/env python3
"""The tone of a rendered clip: how bright, how contrasty, how saturated, how hazy.

Written 2026-09-26 for the owner's words on the distills, which named a look
without a measure: FlashGen "more contrasty", FastH3 "more washed out /
contrasty than flashgen was, even" (`bench/results/2026-09-26_distill_compare_s1.md`).
Those are claims about a tone curve and a colour gamut, and a curve has
numbers. This prints them per clip so the words can be checked against them,
and against the undistilled model at the same seed and prompt.

**Report only. It grades nothing and has no threshold.** Which of these
numbers is what the owner calls "washed out" is exactly what running it
against the owner's words is for. It is not assumed here.

Per clip, over every 12th frame at 480x270, on the sRGB-coded values the file
holds (no linearisation), with luma Y from BT.709 weights:

- `black` / `white`: the 1st and 99th percentile of Y. Lifted blacks and
  pulled-down whites are the usual meaning of "washed out".
- `range`: `white - black`, the tonal range actually used.
- `mid`: the median Y.
- `rms_contrast`: the standard deviation of Y over the frame.
- `crushed` / `clipped`: the fraction of pixels with Y at or below 0.02, or at
  or above 0.98.
- `chroma`: the mean of max(R,G,B) - min(R,G,B), a saturation measure that
  does not blow up in the shadows as HSV S does.
- `haze`: the mean of the dark channel (the minimum over R, G and B in a 15x15
  patch). Haze, fog and lifted, desaturated shadows raise it; a clean image with
  real blacks keeps it low.
- `detail`: the mean absolute Laplacian of Y, a local-contrast and sharpness
  measure. It is scale-dependent, so compare only within one run.

`--quarters` also prints each quarter of the clip, since a multi-shot scene can
change look shot to shot.

The scale is this file's own. Quote differences between clips measured in one
run, never absolute values against another tool.

    <comfy venv python> bench/measure_clip_tone.py [--quarters] [--json OUT] <clip.mp4> ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

W, H = 480, 270
#: Frames sampled: every Nth. **Reasoned**: a 345-frame clip gives 29 frames,
#: enough to average over motion and cheap to decode.
EVERY = 12
#: Dark-channel patch side. **Inherited** from He et al.'s dark channel prior,
#: 15x15 at their image size.
PATCH = 15


def frames(path: Path) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf",
           f"select=not(mod(n\\,{EVERY})),scale={W}:{H}", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3).astype(np.float32) / 255.0


def _min_filter(x: np.ndarray, k: int) -> np.ndarray:
    """k x k minimum filter by separable sliding minima, edge-padded."""
    p = k // 2
    x = np.pad(x, ((p, p), (p, p)), mode="edge")
    v = np.lib.stride_tricks.sliding_window_view(x, k, axis=0).min(axis=-1)
    return np.lib.stride_tricks.sliding_window_view(v, k, axis=1).min(axis=-1)


def stats(f: np.ndarray) -> dict:
    y = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
    chroma = f.max(axis=-1) - f.min(axis=-1)
    dark = np.stack([_min_filter(fr.min(axis=-1), PATCH) for fr in f])
    lap = np.abs(4 * y[:, 1:-1, 1:-1] - y[:, :-2, 1:-1] - y[:, 2:, 1:-1]
                 - y[:, 1:-1, :-2] - y[:, 1:-1, 2:])
    black, mid, white = np.percentile(y, [1, 50, 99])
    return {
        "black": round(float(black), 3), "white": round(float(white), 3),
        "range": round(float(white - black), 3), "mid": round(float(mid), 3),
        "rms_contrast": round(float(y.reshape(len(y), -1).std(axis=1).mean()), 3),
        "crushed": round(float((y <= 0.02).mean()), 4),
        "clipped": round(float((y >= 0.98).mean()), 4),
        "chroma": round(float(chroma.mean()), 3),
        "haze": round(float(dark.mean()), 3),
        "detail": round(float(lap.mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--quarters", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    keys = ["black", "white", "range", "mid", "rms_contrast", "crushed", "clipped",
            "chroma", "haze", "detail"]
    rows = []
    print(f"{'clip':<58}" + "".join(f"{k:>13}" for k in keys))
    for clip in args.clips:
        f = frames(clip)
        parts = [("all", f)]
        if args.quarters:
            q = np.array_split(np.arange(len(f)), 4)
            parts += [(f"q{i + 1}", f[idx]) for i, idx in enumerate(q)]
        for tag, sub in parts:
            s = stats(sub)
            rows.append({"clip": clip.name, "part": tag, **s})
            name = clip.stem if tag == "all" else f"  {tag}"
            print(f"{name[:58]:<58}" + "".join(f"{s[k]:>13}" for k in keys))
    if args.json:
        args.json.write_text(json.dumps({"tool": Path(__file__).name, "every": EVERY,
                                         "size": [W, H], "rows": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
