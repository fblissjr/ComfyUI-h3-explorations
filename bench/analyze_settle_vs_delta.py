#!/usr/bin/env python3
"""Does a latent frame settle later in sampling when the picture changes more there? CPU only.

Written 2026-09-27 for `docs/state/open_experiments.md` #43, the premise under the
owner's motion-adaptive schedule (`docs/state/next_steps.md`, "fit the
schedule to how much of the frame changes"): the idea only pays if frames that
change a lot frame to frame are still being decided late, where PDD's tail is
coarse.

Joins two existing measurements of one render:
- `bench/x0_step_frames.py --json`: per latent frame, each sampling step's
  `to_final` (how far that step's x0 prediction still is from the final
  latent) and the frame's video-frame span;
- `bench/measure_clip_delta.py --json`: the clip's per-frame delta series
  (frame n to n+1) and the cut times ffmpeg's scene score finds.

Per latent frame it takes:
- **delta:** the mean of the delta series over the frame's video span, with
  a latent frame of one video frame using the delta into its successor;
- **settle:** the first step at which `to_final` falls under a threshold,
  as a fraction of the render's steps (0 = the first evaluation, 1 = the
  last). Several thresholds are reported, never one chosen, because the scale
  of `to_final` differs between samplers;
- **cut:** whether a cut time falls inside the frame's span.

Reported per threshold: Spearman's rank correlation between delta and settle
over the non-cut frames, and the median settle of the cut frames against the
median of the rest. Report only; no threshold decides anything here.

**What it does not show.** Delta is change, not motion (`measure_clip_delta.py`
docstring). One render is one trajectory. A correlation is not a cause: a
frame that changes a lot may settle late because it is hard, or because the
sampler treats every frame alike and hard frames are simply farther away.

    python bench/analyze_settle_vs_delta.py --x0 X0.json --delta DELTA.json --clip NAME.mp4 [--json OUT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

THRESHOLDS = (0.5, 0.3, 0.2, 0.1)


def ranks(x) -> np.ndarray:
    """Average ranks, so tied values share one rank (a few-step sampler ties
    most frames on a step index, and plain argsort ranks would then order the
    ties by frame index, which is arbitrary)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def settle(to_final, tau: float) -> float:
    """The fractional step at which `to_final` first crosses under `tau`,
    linearly interpolated between steps, as a fraction of the render's steps.
    Fractional so a few-step sampler does not tie every frame on one index."""
    n = len(to_final)
    for i, v in enumerate(to_final):
        if v < tau:
            if i == 0:
                return 0.0
            prev = to_final[i - 1]
            frac = (prev - tau) / (prev - v) if prev != v else 0.0
            return (i - 1 + frac) / (n - 1)
    return 1.0


def late_mass(to_final) -> float:
    """Mean `to_final` over the steps, normalised by the first step's: how
    much of the frame is still undecided, averaged over the run. Tie-free and
    threshold-free; higher means the frame settles later relative to where it
    started."""
    v = np.asarray(to_final, float)
    return float(v.mean() / v[0]) if v[0] > 0 else 0.0


def analyse(x0: dict, clip: dict) -> dict:
    series = np.asarray(clip["series"], float)
    fps = float(x0["fps"])
    cuts = [round(t * fps) for t in clip.get("cut_times_s", [])]
    frames = []
    for lf in x0["latent_frames"]:
        a, b = lf["video_frames"]
        lo, hi = a, max(a, b)
        seg = series[lo:min(hi + 1, len(series))]
        if not len(seg):
            seg = series[max(len(series) - 1, 0):]
        frames.append({"k": lf["k"], "t_s": lf["t_s"], "delta": float(seg.mean()),
                       "cut": any(lo <= c <= hi + 1 for c in cuts),
                       "to_final": lf["to_final"]})
    out = {"steps": len(frames[0]["to_final"]), "latent_frames": len(frames),
           "cut_frames": [f["k"] for f in frames if f["cut"]], "per_threshold": {}}
    nc = [f for f in frames if not f["cut"]]
    lm = [late_mass(f["to_final"]) for f in nc]
    out["late_mass"] = {"spearman_delta_late_mass": spearman([f["delta"] for f in nc], lm),
                        "median_noncut": float(np.median(lm)),
                        "median_cut": (float(np.median([late_mass(f["to_final"]) for f in frames if f["cut"]]))
                                       if out["cut_frames"] else None)}
    for tau in THRESHOLDS:
        s = [settle(f["to_final"], tau) for f in frames]
        body = [(f["delta"], v) for f, v in zip(frames, s) if not f["cut"]]
        cut_s = [v for f, v in zip(frames, s) if f["cut"]]
        rest = [v for _, v in body]
        out["per_threshold"][str(tau)] = {
            "spearman_delta_settle": spearman([d for d, _ in body], rest),
            "settle_median_noncut": float(np.median(rest)) if rest else None,
            "settle_median_cut": float(np.median(cut_s)) if cut_s else None,
            "settle_spread_noncut": [float(np.percentile(rest, 10)), float(np.percentile(rest, 90))] if rest else None,
            "never_settled": sum(1 for v, f in zip(s, frames) if v == 1.0 and f["to_final"][-2] >= tau)}
    out["frames"] = [{"k": f["k"], "t_s": f["t_s"], "delta": round(f["delta"], 5), "cut": f["cut"],
                      "settle": {str(t): settle(f["to_final"], t) for t in THRESHOLDS}} for f in frames]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--x0", required=True, help="bench/x0_step_frames.py --json output")
    ap.add_argument("--delta", required=True, help="bench/measure_clip_delta.py --json output")
    ap.add_argument("--clip", required=True, help="the clip's file name inside --delta")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    x0 = json.loads(Path(args.x0).read_text())
    d = json.loads(Path(args.delta).read_text())
    clip = next((c for c in d["clips"] if Path(c["clip"]).name == Path(args.clip).name), None)
    if clip is None:
        raise SystemExit(f"{args.clip} is not in {args.delta}")
    res = {"measured_by": "bench/analyze_settle_vs_delta.py", "x0": Path(args.x0).name,
           "clip": clip["clip"], **analyse(x0, clip)}
    print(f"{res['clip']}: {res['steps']} steps, {res['latent_frames']} latent frames, "
          f"cut frames {res['cut_frames']}")
    print(f"  {'tau':>5} {'spearman':>9} {'settle med':>11} {'p10-p90':>13} {'cut med':>8} {'unsettled':>9}")
    for tau, r in res["per_threshold"].items():
        sp = "n/a" if r["spearman_delta_settle"] is None else f"{r['spearman_delta_settle']:+.3f}"
        spread = "n/a" if r["settle_spread_noncut"] is None else "{:.2f}-{:.2f}".format(*r["settle_spread_noncut"])
        cm = "n/a" if r["settle_median_cut"] is None else f"{r['settle_median_cut']:.2f}"
        print(f"  {tau:>5} {sp:>9} {r['settle_median_noncut']:>11.2f} {spread:>13} {cm:>8} {r['never_settled']:>9}")
    lm = res["late_mass"]
    sp = "n/a" if lm["spearman_delta_late_mass"] is None else f"{lm['spearman_delta_late_mass']:+.3f}"
    cm = "n/a" if lm["median_cut"] is None else f"{lm['median_cut']:.3f}"
    print(f"  late mass (threshold-free): spearman {sp}, median {lm['median_noncut']:.3f}, cut median {cm}")
    if args.json:
        args.json.write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
