#!/usr/bin/env python3
"""How a regenerated region sits in its plate: its detail, its grain and its tone against the plate's.

    <python> bench/region_against_plate.py measure --source SOURCE.mp4 --piece PIECE.mp4 --first N \\
        [--frames A-B] [--step K] [--out look.json] [--control-noise SD]

**What it buys.** A render is laid into footage that has its own softness, grain and cast, and "it does not sit
in the picture" is said by eye. This puts numbers on it in three separate ways, so the smallest post step can
be chosen and then checked with the same numbers before anyone's eye is asked, and so a guess about WHY it does
not sit (too clean, too sharp, the wrong colour) can be refuted. The first time it was run (2026-10-10, two
fixed-camera renders of one clip) the guess was "cleaner than the plate" and the answer was the opposite: crisper
at the edges and busier, with tone and cast already matching.

**What is compared.** No mask from the graph is needed: where a piece differs from the fitted original it changed
the picture (`assemble_delivery.changed`). Four sets of pixels:

- `new`: the piece, inside its changed region, `INSET` pixels in from the edge.
- `old`: the ORIGINAL, in the same place: what stood there before, in the same light.
- `plate`: the piece's own kept pixels in a band just outside the region. Same file, same codec generation as
  `new`, so this is the fair reference for grain.
- `plate, original`: the original in that band: what the piece's encode did to the plate.

Both sides are read as the same YUV, the piece's planes and the original through the path of a kept pixel
(`assemble_delivery.original_frames`), never one through 8-bit rgb or `gray` and one not: ffmpeg's 8-bit rgb path
reads a level darker than its 16-bit one, and its gray widens a tv-range picture.

**Detail** is energy in three bands of the luma (finer than 1 pixel sigma, 1 to 2, 2 to 4), with the grain taken
out: grain is what changes from one frame to the next where the picture itself does not (`STILL`), measured in
the same bands. What is left is the picture's own detail, and the ratio between neighbouring bands says how soft
it is whatever its contrast. `new` is also measured under a few trial blurs, so the blur whose ratios match the
plate's can be read off (`assemble_delivery.py --soften` applies one). A trial blurs the piece as decoded; a blur
applied before an encode comes back with the encoder's own fine detail on top, so read the delivered file again
after softening it (measured 2026-10-10: within 3% of the trial on soft footage, 29% above it on a noisy test
picture at crf 10).

**Texture where the picture is flat** is a different thing from either, and it is what "clean" looks like: the
size of the finest band at its median pixel (a wall, flat skin) beside its 99th percentile (an edge). A plate with
noise in it has a high median and soft edges; a clean render has a low median and crisp edges, and can have the
same energy on average.

**Grain** is that frame-to-frame figure, as a standard deviation in levels, per band, for luma and for the two
chroma planes, and for luma in three ranges of brightness.

**Tone**: percentiles of luma, the mean chroma offset in dark, middle and bright pixels (a cast), and percentiles
of saturation.

**What it does not show.** Whether a difference is visible. `old` is something else in the same place: its tone
is a guide to the light, not a target. Grain that the source's own encoder froze between frames does not move, so
it counts here as detail, not grain. A shot where the camera moves has few still pixels and its grain figures are
motion, not grain: keep `--frames` inside one shot with a fixed camera.

**Its control** is `--control-noise SD`: white noise of that size, new on every frame, is added to the piece's luma
before anything is measured. The finest band holds 0.872 of white noise's size (a sigma-1 blur taken from the
identity), so `new`'s grain there must come back as the root of its own squared plus (0.872 x SD) squared, and
its detail must not move. `bench/check_region_against_plate.py` holds that and two more.

Nothing here describes what a clip shows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assemble_delivery as A  # noqa: E402  (the piece's planes, the fitted original, and what changed)

INSET = 6        # pixels in from the region's edge; reasoned: clear of the composite's feather
NEAR, FAR = 12, 48   # the plate band, pixels out from the region; reasoned: past the seam, still the same surfaces
STILL = 0.75     # levels the picture (blurred to sigma 4) may move between frames and count as still; reasoned
BANDS = ((1.0, "finer than 1"), (2.0, "1 to 2"), (4.0, "2 to 4"))
TRIAL_BLURS = (0.5, 0.7, 1.0, 1.3, 1.6)
LUMA_RANGES = ((0, 70, "dark"), (70, 140, "middle"), (140, 256, "bright"))


def bands_of(y):
    """The luma split into BANDS, plus the picture at the coarsest blur (what `STILL` is judged on)."""
    out, last = [], y
    for sigma, _ in BANDS:
        blurred = cv2.GaussianBlur(y, (0, 0), sigma)
        out.append(last - blurred)
        last = blurred
    return out, last


class Tally:
    def __init__(self):
        self.total = np.zeros(len(BANDS)); self.total_n = 0
        self.moved = np.zeros(len(BANDS)); self.moved_n = 0
        self.by_luma = np.zeros(len(LUMA_RANGES)); self.by_luma_n = np.zeros(len(LUMA_RANGES))
        self.sizes = np.zeros((2, 400))               # |finest band| and |next band|, in tenths of a level

    def add(self, now, before, mask, still):
        (bands, low), (bands0, _) = now, before
        if mask.sum():
            self.total += [float((b[mask] ** 2).sum()) for b in bands]
            self.total_n += int(mask.sum())
            for i in range(2):
                self.sizes[i] += np.bincount(np.minimum(np.abs(bands[i][mask]) * 10, 399).astype(np.int64), minlength=400)
        both = mask & still
        if both.sum():
            self.moved += [float(((b[both] - b0[both]) ** 2).sum()) / 2 for b, b0 in zip(bands, bands0)]
            self.moved_n += int(both.sum())
            for i, (lo, hi, _) in enumerate(LUMA_RANGES):
                m = both & (low >= lo) & (low < hi)
                self.by_luma[i] += float(((bands[0][m] - bands0[0][m]) ** 2).sum()) / 2
                self.by_luma_n[i] += int(m.sum())

    def report(self):
        total = self.total / max(self.total_n, 1)
        grain = self.moved / max(self.moved_n, 1)
        detail = np.maximum(total - grain, 0)

        def pct(hist):
            c = np.cumsum(hist) / max(hist.sum(), 1)
            return [round(float(np.searchsorted(c, q)) / 10, 1) for q in (0.5, 0.9, 0.99)]
        return {"pixels": self.total_n, "still_pixels": self.moved_n,
                "finest_band_size_at_50_90_99": pct(self.sizes[0]), "next_band_size_at_50_90_99": pct(self.sizes[1]),
                "grain_sd_by_band": [round(float(np.sqrt(g)), 3) for g in grain],
                "detail_sd_by_band": [round(float(np.sqrt(d)), 3) for d in detail],
                "detail_ratio_fine_over_mid": round(float(detail[0] / max(detail[1], 1e-9)), 4),
                "detail_ratio_mid_over_coarse": round(float(detail[1] / max(detail[2], 1e-9)), 4),
                "grain_sd_finest_band_by_luma": {name: round(float(np.sqrt(self.by_luma[i] / max(self.by_luma_n[i], 1))), 3)
                                                 for i, (_, _, name) in enumerate(LUMA_RANGES)}}


class Tone:
    def __init__(self):
        self.y = np.zeros(256); self.sat = np.zeros(182)
        self.uv = np.zeros((len(LUMA_RANGES), 2)); self.uv_n = np.zeros(len(LUMA_RANGES))
        self.chroma_moved = np.zeros(2); self.chroma_n = 0

    def add(self, planes, planes0, mask, still):
        y, u, v = planes
        self.y += np.bincount(y[mask].astype(np.int64), minlength=256)[:256]
        half, half_still = mask[::2, ::2], still[::2, ::2]
        yh = y[::2, ::2]
        self.sat += np.bincount(np.hypot(u[half] - 128, v[half] - 128).astype(np.int64), minlength=182)[:182]
        for i, (lo, hi, _) in enumerate(LUMA_RANGES):
            m = half & (yh >= lo) & (yh < hi)
            self.uv[i] += [float((u[m] - 128).sum()), float((v[m] - 128).sum())]
            self.uv_n[i] += int(m.sum())
        both = half & half_still
        if both.sum() and planes0 is not None:
            self.chroma_moved += [float(((u[both] - planes0[1][both]) ** 2).sum()) / 2, float(((v[both] - planes0[2][both]) ** 2).sum()) / 2]
            self.chroma_n += int(both.sum())

    def report(self):
        def pct(hist, qs):
            c = np.cumsum(hist) / max(hist.sum(), 1)
            return [int(np.searchsorted(c, q)) for q in qs]
        return {"luma_percentiles_1_5_25_50_75_95_99": pct(self.y, (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)),
                "saturation_percentiles_50_95": pct(self.sat, (0.5, 0.95)),
                "chroma_offset_u_v_by_luma": {name: [round(float(x), 2) for x in self.uv[i] / max(self.uv_n[i], 1)]
                                              for i, (_, _, name) in enumerate(LUMA_RANGES)},
                "share_of_pixels_by_luma": {name: round(float(self.uv_n[i] / max(self.uv_n.sum(), 1)), 3)
                                            for i, (_, _, name) in enumerate(LUMA_RANGES)},
                "chroma_grain_sd_u_v": [round(float(np.sqrt(x / max(self.chroma_n, 1))), 3) for x in self.chroma_moved]}


def measure(args):
    info = A.probe(args.piece)
    w, h = info["width"], info["height"]
    lo, hi = (int(v) for v in args.frames.split("-")) if args.frames else (0, info["frames"] - 1)
    row = {"piece": args.piece, "piece_first": args.first, "matrix": info["matrix"]}
    first, last = args.first + lo, args.first + hi
    inset = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * INSET + 1, 2 * INSET + 1))
    near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * NEAR + 1, 2 * NEAR + 1))
    far = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * FAR + 1, 2 * FAR + 1))
    names = ["new", "old", "plate", "plate, original"] + [f"new, blurred {s}" for s in TRIAL_BLURS]
    detail = {n: Tally() for n in names}
    tone = {n: Tone() for n in names[:4]}
    before = None
    used = 0
    rng = np.random.default_rng(0)
    for k, (pb, ob) in enumerate(zip(A.piece_frames(row, first, last, w, h), A.original_frames(args.source, first, last, w, h))):
        if k % args.step not in (0, args.step - 1) and args.step > 1:
            before = None
            continue
        p, o = A.planes(pb, w, h), A.planes(ob, w, h)
        hard = A.changed(p, o).astype(np.uint8)
        if args.control_noise:
            p[0] = p[0] + rng.normal(0, args.control_noise, p[0].shape).astype(np.float32)
        inner = cv2.erode(hard, inset).astype(bool)
        band = cv2.dilate(hard, far).astype(bool) & ~cv2.dilate(hard, near).astype(bool)
        now = {"p": bands_of(p[0]), "o": bands_of(o[0]), "planes_p": p, "planes_o": o, "inner": inner, "band": band}
        for s in TRIAL_BLURS:
            now[f"blurred {s}"] = bands_of(cv2.GaussianBlur(p[0], (0, 0), s))
        if before is not None and (k % args.step == 0 or args.step == 1):
            still_p = np.abs(now["p"][1] - before["p"][1]) < STILL
            still_o = np.abs(now["o"][1] - before["o"][1]) < STILL
            m_in, m_band = inner & before["inner"], band & before["band"]
            detail["new"].add(now["p"], before["p"], m_in, still_p)
            detail["old"].add(now["o"], before["o"], m_in, still_o)
            detail["plate"].add(now["p"], before["p"], m_band, still_p)
            detail["plate, original"].add(now["o"], before["o"], m_band, still_o)
            for s in TRIAL_BLURS:
                detail[f"new, blurred {s}"].add(now[f"blurred {s}"], before[f"blurred {s}"], m_in, still_p)
            tone["new"].add(p, before["planes_p"], m_in, still_p)
            tone["old"].add(o, before["planes_o"], m_in, still_o)
            tone["plate"].add(p, before["planes_p"], m_band, still_p)
            tone["plate, original"].add(o, before["planes_o"], m_band, still_o)
            used += 1
        before = now
    out = {"piece": args.piece, "first": args.first, "frames": [lo, hi], "step": args.step, "frame_pairs_read": used,
           "bands_sigma": [name for _, name in BANDS], "inset_px": INSET, "plate_band_px": [NEAR, FAR], "still_levels": STILL,
           "sets": {n: {**detail[n].report(), **(tone[n].report() if n in tone else {})} for n in names}}
    return out


def show(r):
    print(os.path.basename(r["piece"]), "| frame pairs", r["frame_pairs_read"])
    print("  %-22s %-24s %-24s %-9s %-9s %s" % ("set", "detail sd (fine,mid,coarse)", "grain sd (fine,mid,coarse)", "fine/mid", "mid/crs", "grain finest by luma (dark,mid,bright)"))
    for n, s in r["sets"].items():
        print("  %-22s %-27s %-26s %-9.3f %-9.3f %s" % (n, s["detail_sd_by_band"], s["grain_sd_by_band"], s["detail_ratio_fine_over_mid"],
                                                       s["detail_ratio_mid_over_coarse"], list(s["grain_sd_finest_band_by_luma"].values())))
    print("  %-22s %-26s %s" % ("set", "finest band at 50,90,99 %", "next band at 50,90,99 %"))
    for n, s in r["sets"].items():
        print("  %-22s %-26s %s" % (n, s["finest_band_size_at_50_90_99"], s["next_band_size_at_50_90_99"]))
    print("  %-22s %-34s %-12s %-44s %s" % ("set", "luma pct 1,5,25,50,75,95,99", "sat 50,95", "chroma offset u,v (dark | middle | bright)", "chroma grain u,v"))
    for n, s in r["sets"].items():
        if "luma_percentiles_1_5_25_50_75_95_99" in s:
            print("  %-22s %-34s %-12s %-44s %s" % (n, s["luma_percentiles_1_5_25_50_75_95_99"], s["saturation_percentiles_50_95"],
                                                  " | ".join(str(v) for v in s["chroma_offset_u_v_by_luma"].values()), s["chroma_grain_sd_u_v"]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--source", required=True)
    m.add_argument("--piece", required=True)
    m.add_argument("--first", type=int, required=True, help="the copy frame that is the piece's frame 0")
    m.add_argument("--frames", help="first-last of the piece; default all. Keep it inside one shot")
    m.add_argument("--step", type=int, default=1, help="read one pair of neighbouring frames in every this many")
    m.add_argument("--control-noise", type=float, default=0.0, help="the control: white noise of this sd added to the piece's luma")
    m.add_argument("--out")
    args = p.parse_args()
    result = measure(args)
    show(result)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
