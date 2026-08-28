#!/usr/bin/env python3
"""How much a rendered clip CHANGES frame to frame.

**Delta, not motion, and the distinction is load-bearing.** What this computes
is mean(|frame[n] - frame[n-1]|): frame-to-frame change, whatever causes it.
No optical flow, no tracking. A cut, a light switching on and a camera whip all
score high with nothing moving in the scene's own terms. This file was named
for motion until 2026-08-28 and renamed when a peer corrected the same gloss in
their own lane -- the owner's framing is "motion is one thing but its also
just... massive delta changes", and delta is what is measured.

Exists because of a 2026-08-28 finding that is about the CORPUS, not about any
prompt: artifact severity under PDD tracks inter-frame delta (+0.676), so a scene with no
high-delta frames cannot discriminate a delta-dependent defect however
carefully it is scored or however blind the scoring. `docs/eval_comparison.md`
carries the process rule; this is the instrument that says which regime a clip is in.

**Report only. It grades nothing and has no threshold**, because what counts as
"enough delta" depends on the defect being judged and nobody has established
that. It prints a number so a perceptual claim can state its regime.

**The scale is this file's own and is not comparable to another tool's.** Median
absolute difference between consecutive greyscale frames at a fixed small
resolution, in [0, 1]. A different resolution or colour handling moves every
number by roughly a constant, so compare within one run, never across tools --
measured against another session's numbers on the same clips, this file reads
about 0.75x, with the ratios preserved. Quote ratios, or re-run everything here.

Cuts are NOT masked, so a clip with hard cuts carries a spike at each one. That
inflates `pct_above` for multi-shot scenes relative to single-take ones; with
three cuts in 362 frames the effect is under 1% of frames and is ignored here,
but a scene with many cuts would need them removed first.

    uv run --active --no-sync python bench/measure_clip_delta.py <clip.mp4> ...
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

W, H = 160, 96
BUSY = 0.02


def motion(path: Path, w: int = W, h: int = H) -> dict | None:
    """Median inter-frame absolute difference, and the share of busy frames."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
        capture_output=True)
    raw = proc.stdout
    n = len(raw) // (w * h)
    if n < 2:
        return None
    a = np.frombuffer(raw, dtype=np.uint8)[:n * w * h]
    a = a.reshape(n, h, w).astype(np.float32) / 255.0
    d = np.abs(np.diff(a, axis=0)).mean(axis=(1, 2))
    return {
        "frames": n,
        "median": float(np.median(d)),
        "mean": float(d.mean()),
        "p90": float(np.percentile(d, 90)),
        "pct_above_busy": float((d > BUSY).mean() * 100.0),
    }


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: measure_clip_delta.py <clip.mp4> ...")
        return 2
    print(f"  {'clip':>44} {'median':>9} {'p90':>9} {'%busy':>8} {'frames':>7}")
    for p in argv:
        path = Path(p)
        if not path.exists():
            print(f"  {path.name:>44}  (missing)")
            continue
        r = motion(path)
        if r is None:
            print(f"  {path.name:>44}  (unreadable)")
            continue
        print(f"  {path.name[-44:]:>44} {r['median']:>9.4f} {r['p90']:>9.4f} "
              f"{r['pct_above_busy']:>7.1f}% {r['frames']:>7}")
    print(f"\n  busy threshold {BUSY}; scale is this file's own -- see the docstring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
