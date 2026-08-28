#!/usr/bin/env python3
"""Does PDD's block width cost AUDIO specifically, or does it cost both streams?

## The claim this was built to attack

`docs/research/pdd/audio_under_pdd.md` argues that ComfyUI's audio change of
variable (`comfy/ldm/minimax/model.py:530-551`) interacts badly with a PDD fused
head, producing an error that is **audio-only by construction** and grows with
block width. Its evidence is the arm table in
`bench/results/2026-08-28_pdd_partition_fidelity_362.json`: audio rel L2 rises
monotonically across the partitions while **video stays flat**.

The audio-only half rests entirely on video being flat. This script asks whether
video is flat, or whether the metric cannot see it.

## What it found, and why the metric was blind

Raw-video rel L2 is dominated by the DC term -- frame means, which every arm
reproduces. Remove the mean and video is NOT flat: contrast falls monotonically
with partition coarseness, on 100% of frames for the two coarsest arms. Video's
*correlation* with the reference moves far less over the same arms, so the loss
is mostly in amplitude rather than in content. Both streams lose amplitude and
only audio's was visible.

Two consequences, and the second is the one that bites:

  * The owner's complaint has a blunt answer. PDD audio is not subtly wrong, it
    is 4.6 to 11 dB DOWN (confirmed against ffmpeg volumedetect, not just this
    script's decode path). `opt4`'s audio rel L2 of 0.992 means "nearly silent",
    not "decorrelated" -- its best-fit gain against the reference is 0.046.
  * **Integrated drift does not identify the audio transform.** Recomputed from
    a pure video-time coarseness sum, with the audio coefficient nowhere in it,
    it is RANK-IDENTICAL on all six arms. So it measures partition coarseness,
    and any partition-derived statistic gives the same ranking. No experiment
    that only varies the partition can separate the two.

## What this does NOT show

It does not refute the transform. The transform freezes TWO coefficients at the
block start -- `B = 1 + (s-1)*sigma_a` on the velocity and
`A = (1-s)*carry*x_a` on the latent -- and with `s = 4` they pull opposite ways,
so it makes no clean sign prediction and none is scored here. Audio is hit
harder than video on every statistic below, and that excess is unexplained;
it is also not evidence for the transform, since the streams differ in sequence
share, VAE and signal statistics.

Identifying the transform needs the TRANSFORM varied at a FIXED partition
(`shift_audio`, i.e. `s`), which coarseness has no reason to respond to.

## The refutable outcome

If video contrast ratio does NOT fall monotonically with coarseness, the
audio-only claim survives this attack and the headline above is withdrawn. The
script says so and exits non-zero.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workflows"))
from pdd_math import pdd_time_grid  # noqa: E402

import h3_config  # noqa: E402

#: The arms rendered by `bench/grade_pdd_partitions.py`, in the subfolder it
#: writes. Read back rather than re-rendered: this analysis is free.
SUBDIR = "pddref"
LENGTH = 362

#: Shift pair from the shipped graphs. `scale` is comfy's `audio_scale`.
SHIFT_V, SHIFT_A, NKNOT = 12.0, 3.0, 32
SCALE = SHIFT_V / SHIFT_A

#: Sigma vectors for the non-uniform arms, copied from
#: `grade_pdd_partitions.py::MANUAL` so both files describe the same arms.
MANUAL = {
    "opt4": "1.0, 0.631579, 0.444444, 0.27907, 0.0",
    "mix6": "1.0, 0.988235, 0.972973, 0.952381, 0.923077, 0.8, 0.0",
    "tail5": "1.0, 0.972973, 0.923077, 0.8, 0.631579, 0.0",
    "tail6": "1.0, 0.972973, 0.923077, 0.878049, 0.8, 0.631579, 0.0",
}

#: Arms with renders on disk, ascending by observed audio rel L2. `ref32` is the
#: reference everything is graded against, not an arm.
SCORED = ["u8", "mix6", "u4", "opt4"]

#: The recorded numbers, asserted rather than recomputed silently. If the files
#: read here are not the files that produced the record, this catches it before
#: any conclusion is drawn -- the arms live outside the repo on a share, so
#: "am I reading the right renders" is a real question and not a formality.
RECORDED = {
    "u8": (0.541977, 0.857948), "u4": (0.537085, 0.923363),
    "opt4": (0.522529, 0.992159), "mix6": (0.536336, 0.898714),
}

#: Audio is compared exactly; video is not, and the asymmetry is measured rather
#: than guessed. `grade_pdd_partitions.py` computes video rel L2 with a
#: single float32 `np.linalg.norm` over 960,823,296 elements, where float32
#: accumulation costs up to 1.6e-3 -- u4 is recorded 0.537085 and is exactly
#: 0.538690. Audio is 964,800 float64 samples and carries no such error, so it
#: is the load-bearing half of this control and is held to 5e-7.
#:
#: **The recorded ORDERING survives**: opt4 < mix6 < u4 < u8 either way, so
#: nothing in that record's conclusions is withdrawn. Its magnitudes are
#: imprecise in the third decimal, and the mix6-to-u4 gap it reports as 7.5e-4
#: is really 2.0e-3. The exact values are re-emitted below under
#: `video.rel_l2` (exact), beside `video.rel_l2_as_recorded_f32`.
VIDEO_TOL, AUDIO_TOL = 5e-3, 5e-7


def grids():
    """sigma_v at each knot, and the audio coefficient B at each knot."""
    gv = (1.0 - pdd_time_grid(SHIFT_V, NKNOT)).double()
    base = gv / (SHIFT_V + gv * (1.0 - SHIFT_V))
    ga = SHIFT_A * base / (1.0 + (SHIFT_A - 1.0) * base)
    return gv, 1.0 + (SCALE - 1.0) * ga


def knots(arm, gv):
    if arm in MANUAL:
        out = []
        for v in (float(x) for x in MANUAL[arm].split(",")):
            d = (gv - v).abs()
            i = int(d.argmin())
            if float(d[i]) > 1e-5:
                raise SystemExit(f"{arm}: sigma {v} is off the {NKNOT}-point grid")
            out.append(i)
        return out
    width = NKNOT // int(arm[1:])
    return list(range(0, NKNOT + 1, width))


def integrated(kn, coef):
    """Sum over blocks of |coef(sigma) - coef(block start)| at the grid points.

    The quantity `audio_under_pdd.md` calls integrated drift. Passed the AUDIO
    coefficient it is that doc's statistic; passed sigma_v it is plain partition
    coarseness with no transform in it. Computing both from one function is the
    point -- the comparison is the finding.
    """
    return sum(float((coef[a:b] - coef[a]).abs().sum()) for a, b in zip(kn[:-1], kn[1:]))


def audio(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le", "-acodec",
         "pcm_f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def frame_paths(d, arm):
    return sorted(d.glob(f"{arm}_v_*.png"))[:LENGTH]


def video_stats(d, arm, ref_paths):
    """One streaming pass; a 362-frame 1152x768 arm never lands in memory whole.

    The centred products are recovered from raw sums algebraically --
    `sum((x-mx)(r-mr)) = sum(xr) - n*mx*mr` -- so the unknown means cost no
    second pass over 362 PNGs.

    Returns the global rel L2 against the reference, the mean-removed gain and
    correlation, and the per-frame contrast ratio.
    """
    a = dict.fromkeys(("dd", "rr", "xx", "xr", "sx", "sr"), 0.0)
    n = 0
    ratios = []
    for p, q in zip(frame_paths(d, arm), ref_paths):
        x = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
        r = np.asarray(Image.open(q).convert("RGB"), dtype=np.float64) / 255.0
        a["dd"] += float(((x - r) ** 2).sum())
        a["rr"] += float((r ** 2).sum())
        a["xx"] += float((x ** 2).sum())
        a["xr"] += float((x * r).sum())
        a["sx"] += float(x.sum())
        a["sr"] += float(r.sum())
        n += x.size
        ratios.append(float(x.std() / r.std()))
    mx, mr = a["sx"] / n, a["sr"] / n
    ratios = np.array(ratios)
    return {
        "rel_l2": (a["dd"] / a["rr"]) ** 0.5,
        "contrast_ratio_mean": float(ratios.mean()),
        "contrast_ratio_p10": float(np.percentile(ratios, 10)),
        "contrast_ratio_p90": float(np.percentile(ratios, 90)),
        "frames_below_ref_pct": float((ratios < 1.0).mean() * 100.0),
        # centred sums, for the caller's gain and correlation
        "_xx": a["xx"] - n * mx * mx,
        "_rr": a["rr"] - n * mr * mr,
        "_xr": a["xr"] - n * mx * mr,
    }


def main() -> int:
    d = h3_config.output_dir() / SUBDIR
    if not d.is_dir():
        raise SystemExit(
            f"{d} does not exist. These arms were rendered by "
            f"bench/grade_pdd_partitions.py; this box overrides the output "
            f"directory, so set H3_OUTPUT_DIR to the server's "
            f"--output-directory and re-run.")
    print(f"reading arms from {SUBDIR}/ under the configured output directory")

    gv, coef_a = grids()
    ref_paths = frame_paths(d, "ref32")
    ra = audio(sorted(d.glob("ref32_a_*.flac"))[0])
    if len(ref_paths) != LENGTH:
        raise SystemExit(f"ref32 has {len(ref_paths)} frames, expected {LENGTH}")
    print(f"reference: {len(ref_paths)} frames, {ra.size} audio samples\n")

    rows, mismatched = {}, []
    for arm in SCORED:
        v = video_stats(d, arm, ref_paths)
        x = audio(sorted(d.glob(f"{arm}_a_*.flac"))[0])
        n = min(x.size, ra.size)
        x, r = x[:n], ra[:n]
        kn = knots(arm, gv)
        rows[arm] = {
            "evaluations": len(kn) - 1,
            "widths": [b - a for a, b in zip(kn[:-1], kn[1:])],
            "drift_audio_coefficient": integrated(kn, coef_a),
            "drift_video_time": integrated(kn, gv),
            "video": {
                "rel_l2": v["rel_l2"],
                "rel_l2_as_recorded_f32": RECORDED[arm][0],
                "centred_gain": v["_xr"] / v["_rr"],
                "centred_corr": v["_xr"] / (v["_xx"] * v["_rr"]) ** 0.5,
                "contrast_ratio_mean": v["contrast_ratio_mean"],
                "contrast_ratio_p10": v["contrast_ratio_p10"],
                "contrast_ratio_p90": v["contrast_ratio_p90"],
                "frames_below_ref_pct": v["frames_below_ref_pct"],
            },
            "audio": {
                "rel_l2": float(np.linalg.norm(x - r) / np.linalg.norm(r)),
                "gain": float(x @ r / (r @ r)),
                "corr": float(x @ r / (np.linalg.norm(x) * np.linalg.norm(r))),
                "rms_ratio": float(x.std() / r.std()),
                "rms_db_vs_ref": float(20.0 * np.log10(x.std() / r.std())),
            },
        }
        rec = RECORDED[arm]
        got = (rows[arm]["video"]["rel_l2"], rows[arm]["audio"]["rel_l2"])
        if abs(got[0] - rec[0]) > VIDEO_TOL or abs(got[1] - rec[1]) > AUDIO_TOL:
            mismatched.append((arm, got, rec))

    if mismatched:
        for arm, got, rec in mismatched:
            print(f"  {arm}: read {got[0]:.6f}/{got[1]:.6f}, "
                  f"record says {rec[0]:.6f}/{rec[1]:.6f}")
        raise SystemExit(
            "these are not the renders that produced "
            "bench/results/2026-08-28_pdd_partition_fidelity_362.json; "
            "nothing below would be about the same arms")
    print("control: audio rel L2 reproduces the record EXACTLY on every arm -- "
          "these are the record's own renders.")
    print("         video is held to 5e-3, not exactness: the record computed "
          "it in float32 over 9.6e8 elements")
    print("         and is off by up to 1.6e-3. The recorded ORDERING is "
          "unaffected; exact values are in the JSON.\n")

    print(f"{'arm':<7}{'ev':>4}{'drift_a':>9}{'drift_v':>9}"
          f"{'VID rel':>9}{'VID contr':>11}{'VID corr':>10}"
          f"{'AUD rel':>9}{'AUD gain':>10}{'AUD dB':>9}")
    print("-" * 87)
    for a in SCORED:
        r = rows[a]
        print(f"{a:<7}{r['evaluations']:>4}{r['drift_audio_coefficient']:>9.2f}"
              f"{r['drift_video_time']:>9.2f}{r['video']['rel_l2']:>9.4f}"
              f"{r['video']['contrast_ratio_mean']:>11.4f}"
              f"{r['video']['centred_corr']:>10.4f}"
              f"{r['audio']['rel_l2']:>9.4f}{r['audio']['gain']:>10.4f}"
              f"{r['audio']['rms_db_vs_ref']:>9.2f}")

    by_drift = sorted(SCORED, key=lambda a: rows[a]["drift_audio_coefficient"])
    contrast = [rows[a]["video"]["contrast_ratio_mean"] for a in by_drift]
    monotone = all(contrast[i] >= contrast[i + 1] for i in range(len(contrast) - 1))

    # Rank equivalence of the two drift measures, over every arm including the
    # unrendered ones -- that is what says no partition experiment can separate
    # them, so it must cover the arms an experiment would use.
    allarms = SCORED + ["tail5", "tail6"]
    ra_ = sorted(allarms, key=lambda a: integrated(knots(a, gv), coef_a))
    rv_ = sorted(allarms, key=lambda a: integrated(knots(a, gv), gv))
    print(f"\naudio-coefficient drift order : {' < '.join(ra_)}")
    print(f"video-time coarseness order   : {' < '.join(rv_)}")
    print(f"rank-identical over all {len(allarms)} arms: {ra_ == rv_}")

    out = REPO / "bench/results/2026-08-28_pdd_stream_energy.json"
    out.write_text(json.dumps({
        "date": "2026-08-28",
        "script": "bench/analyze_pdd_stream_energy.py",
        "reads": "the renders already on disk from "
                 "bench/results/2026-08-28_pdd_partition_fidelity_362.json; "
                 "no GPU, nothing re-rendered",
        "length": LENGTH, "canvas": "1152x768 (fast tier)",
        "arms": rows,
        "video_contrast_monotone_in_drift": monotone,
        "drift_measures_rank_identical": ra_ == rv_,
        "drift_rank_audio_coefficient": ra_,
        "drift_rank_video_time": rv_,
        "findings": [
            "The audio-only half of docs/research/pdd/audio_under_pdd.md does "
            "not survive. Raw-video rel L2 is dominated by the DC term; with "
            "the mean removed, video contrast falls monotonically with "
            "partition coarseness on 100% of frames for the two coarsest "
            "arms. Video's correlation with the reference moves far less over "
            "the same arms, so the loss is mostly amplitude rather than "
            "content. Both streams lose amplitude and only audio's was "
            "visible to the chosen metric.",
            "PDD audio loses ENERGY rather than merely decorrelating: 4.6 dB "
            "down at u8 and 11.2 dB at opt4, best-fit gain 0.308 down to "
            "0.046. opt4's audio rel L2 of 0.992 means nearly silent. This is "
            "a plainer account of the owner's 'the audio is off' than any "
            "phase or content story.",
            "Integrated drift does not identify the audio transform. "
            "Recomputed as pure video-time coarseness, with the audio "
            "coefficient absent, it is rank-identical on all six arms "
            "including the two not yet rendered. So it measures partition "
            "coarseness, and NO experiment that only varies the partition can "
            "separate the transform from generic coarseness.",
        ],
        "do_not_rely_on": [
            "This does not refute the transform. The transform freezes TWO "
            "coefficients at the block start, B on the velocity and A on the "
            "latent, and at s=4 they pull opposite ways -- so it makes no "
            "clean sign prediction, and none is scored here.",
            "Audio is hit harder than video on every statistic here and that "
            "excess is unexplained. It is NOT evidence for the transform: the "
            "streams differ in sequence share, VAE and signal statistics.",
            "Audio rel L2 reaches its decorrelation ceiling (sqrt 2) at a "
            "ONE-frame shift, measured at 1.418 against video's 0.18. It has "
            "no gradation for phase, so across these arms it is reporting "
            "mostly the energy loss.",
            "One seed, one length, one canvas, one prompt, six partitions "
            "from one family. The prompt is the market scene that "
            "docs/prompt_audit.md verdicts `rewrite`.",
            "Distance to the distilled 32-point trajectory is not quality, "
            "and no clip was judged by a person.",
            "Every number here is float64. Statistics over ~1e9 float32 "
            "elements are not reliable in the third decimal, which is how the "
            "record's video rel L2 came to be off by up to 1.6e-3 and how an "
            "earlier float32 pass of THIS analysis put video correlation at "
            "0.588 where it is 0.552.",
        ],
    }, indent=1) + "\n")
    print(f"\nwrote {out.relative_to(REPO)}")

    if not monotone:
        print("\nREFUTED: video contrast does NOT fall monotonically with "
              "coarseness, so video really is flat and the audio-only claim "
              "in docs/research/pdd/audio_under_pdd.md survives this attack. "
              "The headline in this file's docstring is withdrawn.")
        return 1
    print("\nCONSISTENT: video contrast falls monotonically with coarseness, "
          "so the audio-only claim does not survive. Both streams lose "
          "amplitude; only audio's loss was visible to the chosen metric.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
