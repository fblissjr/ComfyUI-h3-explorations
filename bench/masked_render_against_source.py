#!/usr/bin/env python3
"""Measure a masked render against the clip it was made from: flicker at the seam, and when a late region takes effect.

    <python> bench/masked_render_against_source.py flicker --source SRC.mkv --start 67.0 \\
        --arm always_on=A.mp4 --arm switched=B.mp4 --spans 172-236,238-279 --out flicker.json
    <python> bench/masked_render_against_source.py landing --control A.mp4 --arm switched=B.mp4 \\
        --box 430,270,560,420 --frames 140-180 --out landing.json
    <python> bench/masked_render_against_source.py align --render A.mp4 --source SRC.mov --start-frame 220

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

**align** (2026-10-09). Which source frame a render's first, middle and last frames are, and how
the source was fitted to the canvas. For each of the three it tries the source frames around
where that frame should be, picked by number and never by time, under each of `FITS`, on the
columns at the two sides of the picture where a centred subject is not. Run it before either
measurement above: both decode the source from a start time and stretch it to the render's
size, and this says whether that is the picture the render was made from. Written after a
window was found two frames before the one its start time named, with a loader that crops
where `flicker` stretches (`bench/results/2026-10-09_masked_text_and_edge_one_window.md`).
An offset that differs between the three frames is a frame rate that does not match, or
frames dropped or repeated; `verdict` says so. Its control, run the day it was written: a clip
cut from known source frames reports offset 0 at all three.

**timing** (2026-10-09). Is the mask drawn on frame n the subject's outline on frame n of the
source, or on another frame's? The mask is read from the render's overlay (the subject's parts,
tinted `SUBJECT_TINT`) or from `--mask-video` (white is the subject: a tracker's own mask saved as
a video). For each frame the source's edges are summed along the mask's outline, with the source
taken from the same frame and from `--search` frames either side; a moving subject's outline
lies on real edges only on the frame it was cut from. One frame's outline is too noisy to place
by itself, so the scores are averaged over stretches of `--stretch` frames and over the whole
window, and each reports the offset that fits best and its lead over the next. A stretch where
nothing moves has no lead. It is the check for a mask held over from another frame (the part model
does that on frames it does not trust) and for a tracker and a loader that disagree about which
frame is which. Its control is `--shift`: the masks moved that many frames on purpose must be
reported at that offset.

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
# How a source can have been fitted to the canvas. `crop` is what VHS's ffmpeg loader does with both a custom
# width and a custom height (a centre crop to that shape, then one scale; read in its code 2026-10-09).
FITS = {
    "crop": "scale={w}:{h}:force_original_aspect_ratio=increase:flags=bicubic,crop={w}:{h}",
    "stretch": "scale={w}:{h}:flags=bicubic",
    "pad": "scale={w}:{h}:force_original_aspect_ratio=decrease:flags=bicubic,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
}
SIDE = 0.12      # share of the width compared at each side; measured 2026-10-09: at a fifth, a subject filling a
                 # quarter of the frame reached the band on its first frames
SEARCH = 3       # source frames tried each way; reasoned: a start time is not off by more
SUBJECT_TINT = 25.0  # rise in red that marks the subject's parts in the overlay; measured on the Song node's overlay
OUTLINE = 2      # pixels either side of a mask's edge that count as its outline; reasoned, about one edge's width
LEAD = 0.01      # how far a stretch's best offset must stand above the next, in the frame-normalised score, to be
                 # called; measured on the control runs of 2026-10-09 (masks shifted on purpose), one clip
STRETCH = 24     # frames read together: one frame's outline is too noisy to place by itself (measured the same day)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    w, h = out.strip().split(",")[:2]
    return int(w), int(h)


def frames(path, size, pre=(), vf=None, pix="gray", post=()):
    w, h = size
    c = 1 if pix == "gray" else 3
    cmd = ["ffmpeg", "-v", "error", *pre, "-i", path, "-vf", vf or "null", *post, "-f", "rawvideo", "-pix_fmt", pix, "-"]
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
    if args.start_frame is not None:
        # what `align` reported: the frame by number, fitted as the loader fitted it
        n = max(count(p) for p in arms.values())
        source = frames(args.source, size, vf=f"select='gte(n,{args.start_frame})*lt(n,{args.start_frame + n})',"
                        + FITS[args.fit].format(w=w, h=h), post=("-fps_mode", "passthrough"))[..., 0]
    else:
        source = frames(args.source, size, pre=("-ss", str(args.start)), vf=f"fps={args.rate},scale={w}:{h}")[..., 0]
    zones = bands(region_of(arms[args.region_from] if args.region_from else first, size))
    spans = [tuple(int(v) for v in s.split("-")) for s in args.spans.split(",")]
    out = {"source": args.source, "start": args.start, "rate": args.rate, "start_frame": args.start_frame,
           "fit": args.fit if args.start_frame is not None else "stretch", "size": [w, h], "arms": arms,
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


def count(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return int(out.strip().split(",")[0])


def align(args):
    size = probe(args.render)
    w, h = size
    n = count(args.render)
    side = max(1, int(w * args.side))
    cols = np.r_[0:side, w - side:w]

    def numbered(path, first, many, fit=None):
        vf = f"select='gte(n,{first})*lt(n,{first + many})'" + ("," + FITS[fit].format(w=w, h=h) if fit else "")
        # passthrough: frames picked by number are written as picked, none repeated to fill the gap before them
        return frames(path, size, vf=vf, post=("-fps_mode", "passthrough"))[..., 0][:, :, cols] / 255.0

    probes = sorted({0, n // 2, n - 1})
    render = {p: numbered(args.render, p, 1)[0] for p in probes}
    best = {}
    for fit in FITS:
        for p in probes:
            lo = max(0, args.start_frame + p - args.search)
            errs = [round(float(np.abs(render[p] - s).mean()), 4) for s in numbered(args.source, lo, 2 * args.search + 1, fit)]
            k = int(np.argmin(errs))
            best[fit, p] = {"source_frame": lo + k, "offset": lo + k - (args.start_frame + p), "difference": errs[k],
                            "over_the_frames_tried": errs}
    fit = min(FITS, key=lambda f: sum(best[f, p]["difference"] for p in probes))
    offsets = [best[fit, p]["offset"] for p in probes]
    if len(set(offsets)) > 1:
        verdict = "the offset drifts across the window: a frame rate that does not match, or frames dropped or repeated"
    else:
        verdict = f"one offset ({offsets[0]:+d}) across the window; the source was fitted by `{fit}`"
    return {"render": args.render, "source": args.source, "size": [w, h], "frames": n, "start_frame": args.start_frame,
            "fit": fit, "difference_by_fit": {f: [best[f, p]["difference"] for p in probes] for f in FITS},
            "probes": {str(p): best[fit, p] for p in probes}, "offsets": offsets, "verdict": verdict}


def timing(args):
    size = probe(args.render)
    w, h = size
    n = count(args.render)
    if args.mask_video:
        mask = frames(args.mask_video, size, vf=f"scale={w}:{h}")[..., 0] > 127
    else:
        plain = frames(args.render, size, pix="rgb24")
        overlay = frames(args.render[:-len(".mp4")] + "_with_mask.mp4", size, vf=f"crop={w}:{h}:0:{h}", pix="rgb24")
        mask = (overlay[:n, ..., 0] - plain[:n, ..., 0]) > SUBJECT_TINT
    n = min(n, len(mask))
    lo = max(0, args.start_frame - args.search)
    source = frames(args.source, size, vf=f"select='gte(n,{lo})*lt(n,{args.start_frame + n + args.search})',"
                    + FITS[args.fit].format(w=w, h=h), post=("-fps_mode", "passthrough"))[..., 0]
    gy, gx = np.gradient(source, axis=(1, 2))
    edges = np.hypot(gx, gy)
    m = torch.from_numpy(mask[:n]).float()[:, None]
    grown = F.max_pool2d(m, 2 * OUTLINE + 1, 1, OUTLINE)
    outline = ((grown - (1 - F.max_pool2d(1 - m, 2 * OUTLINE + 1, 1, OUTLINE)))[:, 0] > 0.5).numpy()
    offsets = list(range(-args.search, args.search + 1))
    table = np.full((n, len(offsets)), np.nan, np.float32)
    for f in range(n):
        g = f + args.shift                      # the control: the mask of another frame, on purpose
        if not 0 <= g < n or outline[g].sum() < 200:
            continue
        for c, k in enumerate(offsets):
            i = args.start_frame + f + k - lo
            if 0 <= i < len(edges):
                table[f, c] = edges[i][outline[g]].mean()
    table /= np.nanmean(table, axis=1, keepdims=True)       # each frame's scores against its own mean

    def read(first, last):
        rows = table[first:last + 1]
        rows = rows[~np.isnan(rows).any(axis=1)]
        if not len(rows):
            return {"frames": [first, last], "read": 0}
        mean = rows.mean(axis=0)
        order = np.argsort(mean)[::-1]
        return {"frames": [first, last], "read": int(len(rows)), "best_offset": offsets[int(order[0])],
                "lead": round(float(mean[order[0]] - mean[order[1]]), 4), "next_best": offsets[int(order[1])],
                "score_by_offset": [round(float(x), 3) for x in mean]}

    stretches = [read(a, min(a + args.stretch, n) - 1) for a in range(0, n, args.stretch)]
    return {"render": args.render, "mask_from": args.mask_video or "the overlay's subject tint", "source": args.source,
            "start_frame": args.start_frame, "fit": args.fit, "shift": args.shift, "offsets": offsets,
            "whole_window": read(0, n - 1), "stretches": stretches,
            "stretches_that_fit_another_frame": [x["frames"] for x in stretches
                                                 if x.get("read") and x["best_offset"] != 0 and x["lead"] >= LEAD]}


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
    f.add_argument("--start", type=float, default=0.0, help="seconds into the source where the render's window begins")
    f.add_argument("--start-frame", type=int, help="the source frame that is the render's frame 0, as `align` reports "
                   "it; used in place of --start and --rate, with the source fitted by --fit")
    f.add_argument("--fit", choices=sorted(FITS), default="crop")
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
    a = sub.add_parser("align")
    a.add_argument("--render", required=True)
    a.add_argument("--source", required=True)
    a.add_argument("--start-frame", type=int, required=True, help="the source frame the render's frame 0 should be")
    a.add_argument("--search", type=int, default=SEARCH)
    a.add_argument("--side", type=float, default=SIDE)
    a.add_argument("--out")
    t = sub.add_parser("timing")
    t.add_argument("--render", required=True, help="a render with its _with_mask.mp4 beside it")
    t.add_argument("--source", required=True)
    t.add_argument("--start-frame", type=int, required=True, help="the source frame that is the render's frame 0, as `align` reports it")
    t.add_argument("--fit", choices=sorted(FITS), default="crop")
    t.add_argument("--mask-video", help="a mask saved as a video from the window's first frame, white on the subject")
    t.add_argument("--search", type=int, default=4)
    t.add_argument("--stretch", type=int, default=STRETCH)
    t.add_argument("--shift", type=int, default=0, help="the control: read each frame against the mask of the frame this many later")
    t.add_argument("--out")
    args = p.parse_args()
    result = {"flicker": flicker, "landing": landing, "align": align, "timing": timing}[args.mode](args)
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
