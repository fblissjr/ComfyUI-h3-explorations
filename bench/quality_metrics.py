#!/usr/bin/env python3
"""Rank two renders of the same brief: pixel metrics, no model deps.

This repo's open experiment 14 records that `bench/` has no output-quality
instrument of any kind. This is the smallest one that earns its place, ported
in shape (not in code) from NVLabs' Sol-Engine collect_run metrics after the
2026-08-18 survey of that repo: MSE/PSNR, a sharpness proxy, temporal jitter,
and a multi-scale patch-boundary discontinuity score -- the artifact class
block-sparse attention and tiled VAE decode actually produce.

**These metrics RANK, they do not GATE.** Sol-Engine's own tiering policy
states LPIPS-class numbers are "telemetry and ranking signal, not an absolute
delivery threshold for lossy generative dimensions", and this repo's own
history says a person watching the clip is the only quality verdict. Use
this to order arms and to flag the worst frames for a person to look at,
never to declare an arm "fine".

Two standing cautions, both this repo's own findings:
  - An h264 round trip on identical pixels measures ~1.63/255 mean abs
    difference, so distances below that are codec noise, not signal.
    Compare like-encoded files and read small numbers as "same".
  - A pair rendered on `er_sde` differs by trajectory chaos under ANY
    numeric perturbation (CLAUDE.md, 2026-08-18), so cross-arm distance on
    er_sde ranks "how different", never "how degraded". Pair on a
    deterministic sampler before reading these numbers as quality.
  - Audio is a generated modality and none of this sees it. Sol-Engine's
    H3 notes say a visual metric alone rates the model too highly.

Decoding is ffmpeg on PATH piping rawvideo; sampling is `--fps` (default 4).

  bench/quality_metrics.py a.mp4 b.mp4 --fps 4 --json out.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np


def _probe(path: str) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, _, den = s["avg_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)
    return int(s["width"]), int(s["height"]), fps


def decode(path: str, fps: float) -> np.ndarray:
    """[N, H, W] float32 grayscale in 0..255, sampled at `fps`."""
    w, h, _ = _probe(path)
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-vf", f"fps={fps},format=gray",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True)
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = buf.size // (w * h)
    return buf[: n * w * h].reshape(n, h, w).astype(np.float32)


def mean_abs_gradient(f: np.ndarray) -> float:
    """Sharpness proxy: mean |dx| + |dy| over the frame."""
    return float(np.abs(np.diff(f, axis=-1)).mean()
                 + np.abs(np.diff(f, axis=-2)).mean())


def patch_boundary_score(f: np.ndarray, patch: int) -> float:
    """Discontinuity at a `patch`-pixel grid vs everywhere else.

    Ratio of the mean absolute adjacent-pixel difference ACROSS grid
    boundaries (columns/rows at multiples of `patch`) to the mean over all
    other adjacent pairs. 1.0 = the grid is invisible; above 1.0 the seams
    are statistically brighter than the image's own texture. Multi-scale
    because the DiT token grid (32 px at this model's packing), the VAE
    tiling, and any block-sparse artifact land at different strides.
    """
    dx = np.abs(np.diff(f, axis=-1))          # [H, W-1]; dx[:, j] = |f[:, j+1] - f[:, j]|
    dy = np.abs(np.diff(f, axis=-2))
    xb = np.arange(dx.shape[-1]) % patch == patch - 1
    yb = np.arange(dy.shape[-2]) % patch == patch - 1
    boundary = np.concatenate([dx[..., xb].ravel(), dy[..., yb, :].ravel()])
    interior = np.concatenate([dx[..., ~xb].ravel(), dy[..., ~yb, :].ravel()])
    if boundary.size == 0 or interior.size == 0:
        return float("nan")
    return float(boundary.mean() / max(interior.mean(), 1e-6))


def temporal_stats(frames: np.ndarray) -> dict:
    """Adjacent-sampled-frame deltas: motion energy and its roughness."""
    d = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))   # [N-1]
    if d.size < 2:
        return {"delta_mean": float(d.mean()) if d.size else None,
                "jitter": None}
    return {"delta_mean": float(d.mean()),
            # second difference of motion energy: flicker/popping reads as
            # high roughness at equal mean motion
            "jitter": float(np.abs(np.diff(d)).mean())}


def compare(a: np.ndarray, b: np.ndarray, patches=(16, 32, 64)) -> dict:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.shape[1:] != b.shape[1:]:
        raise SystemExit(f"frame sizes differ: {a.shape[1:]} vs {b.shape[1:]}")
    per = ((a - b) ** 2).mean(axis=(1, 2))                   # [N]
    psnr = [float(20 * np.log10(255.0 / np.sqrt(m))) if m > 0 else None
            for m in per]
    worst = np.argsort(per)[-3:][::-1].tolist()
    out = {
        "frames_compared": int(n),
        "mse_mean": float(per.mean()),
        "psnr_mean": float(np.mean([p for p in psnr if p is not None]))
        if any(p is not None for p in psnr) else None,
        "psnr_worst_frames": {int(i): psnr[i] for i in worst},
        "sharpness_ratio_b_over_a":
            mean_abs_gradient(b) / max(mean_abs_gradient(a), 1e-6),
        "temporal_a": temporal_stats(a),
        "temporal_b": temporal_stats(b),
        "patch_boundary_a": {p: float(np.mean([patch_boundary_score(f, p)
                                               for f in a[:: max(1, n // 8)]]))
                             for p in patches},
        "patch_boundary_b": {p: float(np.mean([patch_boundary_score(f, p)
                                               for f in b[:: max(1, n // 8)]]))
                             for p in patches},
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_a")
    ap.add_argument("video_b")
    ap.add_argument("--fps", type=float, default=4.0,
                    help="sampling rate for both videos (default 4)")
    ap.add_argument("--json", default=None, help="also write the result here")
    args = ap.parse_args()

    a = decode(args.video_a, args.fps)
    b = decode(args.video_b, args.fps)
    result = {"a": args.video_a, "b": args.video_b, "fps": args.fps,
              **compare(a, b)}
    text = json.dumps(result, indent=1)
    print(text)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
