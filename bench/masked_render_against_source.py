#!/usr/bin/env python3
"""Measure a masked render against the clip it was made from: flicker at the seam, and when a late region takes effect.

    <python> bench/masked_render_against_source.py flicker --source SRC.mkv --start 67.0 \\
        --arm always_on=A.mp4 --arm switched=B.mp4 --spans 172-236,238-279 --out flicker.json
    <python> bench/masked_render_against_source.py landing --control A.mp4 --arm switched=B.mp4 \\
        --box 430,270,560,420 --frames 140-180 --out landing.json

**What it buys.** Two things an eye reports and a still cannot show (the owner, 2026-10-07, on a
region switched on partway through a window): a flicker that follows the edge of the regenerated
region, and a delay between the region turning on and the new subject appearing. Both become a
number per frame, so two arms can be compared on the same frames.

**flicker.** The residual, render minus source, cancels the scene's own motion; what is left is
what the render added, and its change from one frame to the next is flicker. It is reported in
three bands read off the render's own overlay video (`<render stem>_with_mask.mp4`, the lower
half, as the Song node writes it): just outside the region (frozen pixels beside the seam), just
inside it, and deep inside. The band outside is the floor: the video codec and the VAE, nothing
regenerated. `brightness_step` is the change of the band's mean residual, a pumping the eye
reads as flicker even when single pixels move little. The source is decoded with ffmpeg at the
render's size and rate; `calibration` says how close that decode is to a stretch of a render
known to be all source, and a poor figure there voids the rest.

**landing.** Mean absolute difference between an arm and a control arm inside a box, per frame.
Where the arm shows the original and the control shows the new subject the difference is large;
the frame it falls on is where the switch took effect. The fall is reported as the first frame
under the midpoint between the levels before and after.

**What it does not show.** Whether a difference is visible, or which arm looks better: that is
the owner's eye. The bands are read from one arm's overlay and applied to all, so arms must share
a region on the frames compared. A residual against the source is large wherever the new subject
differs from the old one, so the band deep inside mixes flicker with the subject's own motion;
the band just inside the seam is mostly regenerated background and is the one to read.
"""
import argparse
import json
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

SEAM_BAND = 10   # pixels either side of the region's edge; reasoned: under the smallest margin the lane grows
DEEP = 40        # pixels inside the edge counted as "deep inside"; reasoned, inside a 64 pixel margin
OVERLAY_DIFF = 40.0  # summed channel difference that marks a tinted overlay pixel; measured on the Song node's overlay


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    w, h = out.strip().split(",")[:2]
    return int(w), int(h)


def frames(path, size, pre=(), vf=None, pix="gray"):
    w, h = size
    c = 1 if pix == "gray" else 3
    cmd = ["ffmpeg", "-v", "error", *pre, "-i", path, "-vf", vf or "null", "-f", "rawvideo", "-pix_fmt", pix, "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, c).astype(np.float32)


def region_of(render, size):
    """The regenerated region per frame, from the overlay video written beside a render."""
    w, h = size
    plain = frames(render, size, pix="rgb24")
    overlay = frames(render[:-len(".mp4")] + "_with_mask.mp4", size, vf=f"crop={w}:{h}:0:{h}", pix="rgb24")
    n = min(len(plain), len(overlay))
    return torch.from_numpy(np.abs(overlay[:n] - plain[:n]).sum(-1) > OVERLAY_DIFF).float()[:, None]


def bands(region):
    def grown(x, k):
        return F.max_pool2d(x, 2 * k + 1, 1, k)
    shrunk = 1 - grown(1 - region, SEAM_BAND)
    return {
        "outside the seam (frozen)": (grown(region, SEAM_BAND) - region)[:, 0].numpy() > 0.5,
        "inside the seam": (region - shrunk)[:, 0].numpy() > 0.5,
        "deep inside": (1 - grown(1 - region, DEEP))[:, 0].numpy() > 0.5,
    }


def flicker(args):
    arms = dict(a.split("=", 1) for a in args.arm)
    first = next(iter(arms.values()))
    size = probe(first)
    w, h = size
    source = frames(args.source, size, pre=("-ss", str(args.start)), vf=f"fps={args.rate},scale={w}:{h}")[..., 0]
    zones = bands(region_of(arms[args.region_from] if args.region_from else first, size))
    spans = [tuple(int(v) for v in s.split("-")) for s in args.spans.split(",")]
    out = {"source": args.source, "start": args.start, "rate": args.rate, "size": [w, h], "arms": arms,
           "seam_band_px": SEAM_BAND, "deep_px": DEEP, "spans": {}}
    residual = {}
    for name, path in arms.items():
        r = frames(path, size)[..., 0]
        residual[name] = r[:len(source)] - source[:len(r)]
    if args.all_source:
        name, lo, hi = args.all_source.split(",")
        out["calibration"] = {"arm": name, "frames": [int(lo), int(hi)],
                              "mean_abs_residual": round(float(np.abs(residual[name][int(lo):int(hi)]).mean()), 3)}
    for lo, hi in spans:
        rows = {}
        for zone, mask in zones.items():
            rows[zone] = {}
            for name, r in residual.items():
                change, pump = [], []
                for f in range(lo + 1, min(hi, len(r), len(mask))):
                    m = mask[f] & mask[f - 1]
                    if m.sum() < 300:
                        continue
                    change.append(float(np.abs(r[f][m] - r[f - 1][m]).mean()))
                    pump.append(abs(float(r[f][m].mean() - r[f - 1][m].mean())))
                rows[zone][name] = {"frames_used": len(change),
                                    "flicker": round(float(np.mean(change)), 3) if change else None,
                                    "brightness_step": round(float(np.mean(pump)), 3) if pump else None}
        out["spans"][f"{lo}-{hi}"] = rows
    return out


def landing(args):
    arms = dict(a.split("=", 1) for a in args.arm)
    size = probe(args.control)
    control = frames(args.control, size)[..., 0]
    x0, y0, x1, y1 = (int(v) for v in args.box.split(","))
    lo, hi = (int(v) for v in args.frames.split("-"))
    out = {"control": args.control, "box": [x0, y0, x1, y1], "frames": [lo, hi], "arms": {}}
    for name, path in arms.items():
        a = frames(path, size)[..., 0]
        diff = [round(float(np.abs(a[f][y0:y1, x0:x1] - control[f][y0:y1, x0:x1]).mean()), 2) for f in range(lo, hi + 1)]
        before, after = np.mean(diff[:5]), np.mean(diff[-5:])
        middle = (before + after) / 2
        landed = next((lo + i for i, d in enumerate(diff) if d < middle), None) if before > after else None
        out["arms"][name] = {"render": path, "difference_per_frame": diff, "level_before": round(float(before), 2),
                             "level_after": round(float(after), 2), "first_frame_under_the_midpoint": landed}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("flicker")
    f.add_argument("--source", required=True)
    f.add_argument("--start", type=float, required=True, help="seconds into the source where the render's window begins")
    f.add_argument("--rate", type=float, default=24.0, help="the rate the lane loaded the source at")
    f.add_argument("--arm", action="append", required=True, help="NAME=render.mp4, repeatable")
    f.add_argument("--region-from", help="the arm whose overlay gives the region; the first arm when not given")
    f.add_argument("--spans", required=True, help="frame ranges within one shot each, e.g. 172-236,238-279")
    f.add_argument("--all-source", help="NAME,first,last: a stretch of an arm that is all source, for the calibration")
    f.add_argument("--out")
    g = sub.add_parser("landing")
    g.add_argument("--control", required=True, help="the render the arms are compared with")
    g.add_argument("--arm", action="append", required=True, help="NAME=render.mp4, repeatable")
    g.add_argument("--box", required=True, help="x0,y0,x1,y1 in the render's pixels, on the subject")
    g.add_argument("--frames", required=True, help="first-last, spanning the switch")
    g.add_argument("--out")
    args = p.parse_args()
    result = flicker(args) if args.mode == "flicker" else landing(args)
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
