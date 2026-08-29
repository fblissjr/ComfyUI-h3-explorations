#!/usr/bin/env python3
"""Score the two-shot ablation arms on both candidate axes: delta and spatial detail.

The arms hold length, cut count, seed and every knob, and vary only which two of
the market scene's three shots fill the canvas. `shots12` discriminates, because
it is the only arm carrying both high-delta shots AND the only one without the
high-detail shot:

    delta hypothesis   -> shots12 worst
    detail hypothesis  -> shots12 best

**This file does NOT import `measure_clip_delta.motion`, and that is deliberate.**
That file measures at 160x96 through ffmpeg's `format=gray` (BT.601 luma). The
per-shot table this ablation is scored against was measured at 320x192 rgb24
with a FLAT mean over RGB. Delta survives a resolution change roughly intact;
**spatial detail does not** -- a spatial gradient shrinks as you downsample while
a temporal one does not. So importing the 160x96 delta and computing detail at
the same size would produce a self-consistent set that is not comparable to the
table it is meant to test. One pipeline, every number through it.

    NEVER mix a number from this file with one from measure_clip_delta.py.
    They are different scales on both axes. Re-run rather than convert.

**Absolute detail cannot adjudicate the detail hypothesis.** Ghosted fruit reads
as LOW detail, so "low-detail scene" and "destroyed detail" are the same number.
The fix is `--ref`: pass a reference render of the SAME arm (same prompt, same
seed, more evaluations) and the reported ratio detail(arm)/detail(ref) holds
scene content exactly, so it measures destruction alone. Without `--ref` the
detail column is descriptive only and the perceptual call decides.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

W, H = 320, 192


def load(path: Path, w: int = W, h: int = H) -> np.ndarray:
    """Frames as [n, h, w] in [0,1], flat mean over RGB (not BT.601 luma)."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"scale={w}:{h}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True)
    n = len(proc.stdout) // (w * h * 3)
    if n < 2:
        return np.empty((0, h, w))
    a = np.frombuffer(proc.stdout, dtype=np.uint8)[:n * w * h * 3]
    return a.reshape(n, h, w, 3).astype(np.float64).mean(axis=3) / 255.0


def delta(g: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))


def detail(g: np.ndarray) -> np.ndarray:
    gx = np.abs(np.diff(g, axis=2)).mean(axis=(1, 2))
    gy = np.abs(np.diff(g, axis=1)).mean(axis=(1, 2))
    return gx + gy


def score(path: Path) -> dict | None:
    g = load(path)
    if not len(g):
        return None
    d, s = delta(g), detail(g)
    return {"frames": len(g), "delta": float(np.median(d)),
            "detail": float(np.median(s)), "detail_p10": float(np.percentile(s, 10))}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--ref", help="reference render of the same arm; enables the "
                                  "detail ratio, which is the only controlled reading")
    a = ap.parse_args(argv)

    ref = score(Path(a.ref)) if a.ref else None
    if a.ref and ref is None:
        print(f"reference unreadable: {a.ref}")
        return 1

    hdr = f"  {'clip':>36} {'delta':>9} {'detail':>9} {'frames':>7}"
    print(hdr + ("  detail/ref" if ref else ""))
    for p in map(Path, a.clips):
        r = score(p)
        if r is None:
            print(f"  {p.name:>36}   unreadable")
            continue
        line = (f"  {p.name:>36} {r['delta']:>9.4f} {r['detail']:>9.4f} "
                f"{r['frames']:>7}")
        if ref:
            line += f"  {r['detail'] / ref['detail']:>10.3f}"
        print(line)

    if not ref:
        print("\n  No --ref, so the detail column is descriptive only: it cannot")
        print("  separate a low-detail scene from destroyed detail. Pass a")
        print("  higher-evaluation render of one arm to get the controlled ratio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
