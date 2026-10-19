#!/usr/bin/env python3
"""Is there an edge grid in a clip at VSA's cube period? Decoded pixels only.

Written 2026-09-26 for the FastH3 texture question in `docs/distills/which_distill.md`
("ps2 polygons"; vaedude's annotation under FlashGen and FastH3). VSA works on
4x4x4-token cubes (`comfy_extras/nodes_sparse_attention.py`, `VSA_CUBE`). A
token is a 2x2 patch of a 16x latent, so a cube spans 128x128 pixels. If VSA's
coarse branch invents texture per cube, frames should carry stronger edges on
128-pixel boundaries than elsewhere.

**The comparison that isolates the cube.** Every 128-pixel boundary is also
a 32-pixel token boundary and a 16-pixel latent boundary, and the VAE or
patching can leave a grid of its own at those. So the headline ratio is the
mean absolute gradient across 128-pixel boundaries over the mean across the
other 32-pixel boundaries (x = 31, 63, 95 mod 128): 1.0 means no excess at the
cube period beyond the token grid. Also printed are the token ratio
(32-pixel boundaries against non-boundary columns) and the latent ratio
(16-pixel boundaries), for context. Both axes, averaged over sampled frames.

A ratio is a property of a clip, and a scene with strong structure on a
128-pixel period would show it without any VSA. Read FastH3 against the base
and the other distills rendered on the same scene and canvas.

    python bench/measure_block_period.py CLIP.mp4 [CLIP.mp4 ...] [--every 8] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

CUBE = 128
TOKEN = 32
LATENT = 16


def frames(path: Path, every: int):
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(path)],
                           capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=not(mod(n\\,{every}))",
                           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                          capture_output=True, check=True).stdout
    arr = np.frombuffer(proc, dtype=np.uint8)
    return arr.reshape(-1, h, w).astype(np.float32), w, h


def profile(stack: np.ndarray, axis: int) -> np.ndarray:
    """Mean absolute gradient across each boundary between pixel i and i+1."""
    g = np.abs(np.diff(stack, axis=axis))
    return g.mean(axis=tuple(a for a in range(stack.ndim) if a != axis))


def ratios(prof: np.ndarray) -> dict:
    idx = np.arange(prof.size)            # boundary i sits between pixel i and i+1
    b = (idx + 1)                          # boundary position in pixels
    cube = prof[b % CUBE == 0]
    token_not_cube = prof[(b % TOKEN == 0) & (b % CUBE != 0)]
    token = prof[b % TOKEN == 0]
    latent = prof[b % LATENT == 0]
    off = prof[b % LATENT != 0]
    return {"cube_over_other_token": float(cube.mean() / token_not_cube.mean()),
            "token_over_off_grid": float(token.mean() / off.mean()),
            "latent_over_off_grid": float(latent.mean() / off.mean()),
            "n_cube_boundaries": int(cube.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--every", type=int, default=8, help="use every Nth frame (default 8)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    out = {"measured_by": "bench/measure_block_period.py", "every": args.every, "clips": {}}
    for clip in args.clips:
        stack, w, h = frames(clip, args.every)
        row = {"frames": int(stack.shape[0]), "size": f"{w}x{h}",
               "x": ratios(profile(stack, 2)), "y": ratios(profile(stack, 1))}
        out["clips"][clip.name] = row
        print(f"{clip.name:<64} cube/token x {row['x']['cube_over_other_token']:.3f} "
              f"y {row['y']['cube_over_other_token']:.3f} | token/off x {row['x']['token_over_off_grid']:.3f} "
              f"y {row['y']['token_over_off_grid']:.3f} | latent/off x {row['x']['latent_over_off_grid']:.3f}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
