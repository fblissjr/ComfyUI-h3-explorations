#!/usr/bin/env python3
"""What the unscored part of a pitch-lock score is made of.

`pitch_lock_metrics.py` scores only the frames where the output vocal and the
target vocal both have a pitch (`voiced_overlap_frac`). This module provides the
two ongoing diagnostic modes (`frames` and `periodic`); the five one-off
investigation modes (`agree`, `levels`, `ltas`, `rests`, `under`) from the
2026-11-27 coverage audit live in `data/pure_h3_cover_song/pitch_lock_coverage_oneoff.py`.

- `frames`: splits the target's voiced frames into scored (a pitch in both),
  silent (output under YIN's level gate) and unpitched (output loud, no pitch
  found); gives the time shift at which the two voicings overlap most, the
  scored share in each quarter of the target's frames ranked by the target's own
  steadiness, level and pitch, and the longest scored and unpitched spans, in
  seconds, for finding them by ear; then loosens YIN's periodicity threshold on
  the output only and scores the frames that gain a pitch against the target and
  against the target rotated in time (the null of `pitch_lock_metrics.lag_null`,
  same shifts). A looser threshold takes harmonics for the pitch more often, so
  the in-tune share of gained frames is a floor on how much melody is there,
  never a ceiling.
- `periodic`: the share of a file's active frames in which YIN finds a pitch,
  for comparing an output vocal with a real voice, and the quartiles of the
  pitch it finds, for comparing registers.

`frames --highpass HZ` filters the output first, to ask whether the band under
HZ (a bass left in the stem) is what hides the pitch.

    <comfy-venv-python> bench/pitch_lock_coverage.py frames [--highpass HZ] OUT_VOC TARGET_VOC [OUT_VOC TARGET_VOC ...]
    <comfy-venv-python> bench/pitch_lock_coverage.py periodic FILE [FILE ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pitch_lock_metrics as plm  # noqa: E402
import probe_yin_f0_drift as yin_probe  # noqa: E402

RATE = plm.RATE
HOP_S = plm.HOP_S
IN_TUNE_CENTS = 50.0
# The first entry is the threshold every score in the records was made at.
THRESHOLDS = (yin_probe.DEFAULT_THRESHOLD, 0.25, 0.35, 0.50)
LADDER_STEP = 0.05
OVERLAP_SEARCH_S = 1.0
SPANS_LISTED = 4
SPAN_BRIDGE_FRAMES = 5
# A frame counts as active when it is within this of the clip's loud frames.
ACTIVE_BELOW_LOUD_DB = 20.0
LOUD_PERCENTILE = 95.0
HIGHPASS_PASSES = 4  # second-order sections in a row


def voiced(f0: np.ndarray) -> np.ndarray:
    return np.isfinite(f0) & (f0 > 0)


def _in_tune(fo: np.ndarray, ft: np.ndarray, frames: np.ndarray) -> float | None:
    """Share of `frames` (where both have a pitch) within `IN_TUNE_CENTS`, octaves folded, no transposition."""
    use = frames & voiced(fo) & voiced(ft)
    if int(use.sum()) < plm.MIN_SHARED_FRAMES:
        return None
    cents = 1200.0 * np.log2(fo[use] / ft[use])
    return float(np.mean(np.abs(((cents + 600.0) % 1200.0) - 600.0) <= IN_TUNE_CENTS))


def split_frames(fo: np.ndarray, rms_dbfs: np.ndarray, ft: np.ndarray) -> dict[str, int]:
    """Count the target's voiced frames by what the output is doing in them."""
    target, out = voiced(ft), voiced(fo)
    missed = target & ~out
    silent = missed & (rms_dbfs < yin_probe.DEFAULT_MIN_DBFS)
    return {
        "target_voiced": int(target.sum()),
        "scored": int((target & out).sum()),
        "silent": int(silent.sum()),
        "unpitched": int((missed & ~silent).sum()),
        "output_pitch_outside_target": int((out & ~target).sum()),
    }


def best_overlap(fo: np.ndarray, ft: np.ndarray, *, hop_s: float = HOP_S) -> dict[str, Any]:
    """Frames voiced in both at no shift, and at the shift within `OVERLAP_SEARCH_S` that has the most."""
    target, out = voiced(ft), voiced(fo)
    reach = round(OVERLAP_SEARCH_S / hop_s)
    counts = {lag: int((np.roll(target, lag) & out).sum()) for lag in range(-reach, reach + 1)}
    lag = max(counts, key=lambda k: (counts[k], -abs(k)))
    return {"at_no_shift": counts[0], "best": counts[lag], "best_shift_ms": round(lag * hop_s * 1000)}


def scored_by_quartile(values: np.ndarray, target: np.ndarray, out: np.ndarray) -> dict[str, list[float]]:
    """Share of the target's voiced frames that are scored, in each quarter of them ranked by `values`.

    Quarters are cut by rank, lowest first, so each holds the same number of frames.
    """
    frames = np.flatnonzero(target)
    order = frames[np.argsort(values[frames], kind="stable")]
    quarters = np.array_split(order, 4)
    return {
        "upper_edges": [round(float(values[q[-1]]), 3) for q in quarters if len(q)],
        "scored_frac": [round(float(out[q].mean()), 3) for q in quarters if len(q)],
    }


def longest_spans(frames: np.ndarray, *, hop_s: float = HOP_S) -> list[list[float]]:
    """The `SPANS_LISTED` longest runs of `frames` in seconds, in time order, for finding them by ear.

    Runs separated by `SPAN_BRIDGE_FRAMES` or fewer are one span.
    """
    runs: list[list[int]] = []
    for i in np.flatnonzero(frames):
        if runs and i - runs[-1][1] <= SPAN_BRIDGE_FRAMES + 1:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    longest = sorted(runs, key=lambda r: r[0] - r[1])[:SPANS_LISTED]
    return [[round(a * hop_s, 2), round((b + 1) * hop_s, 2)] for a, b in sorted(longest)]


def gained_frames_null(fo: np.ndarray, ft: np.ndarray, gained: np.ndarray, *, hop_s: float = HOP_S) -> dict[str, Any] | None:
    """In-tune share of the gained frames, aligned and against the target rotated in time."""
    aligned = _in_tune(fo, ft, gained)
    n = len(fo)
    step, edge = max(1, round(plm.LAG_STEP_S / hop_s)), round(plm.LAG_EDGE_S / hop_s)
    shifted = [v for v in (_in_tune(fo, np.roll(ft, lag), gained) for lag in range(edge, n - edge, step)) if v is not None]
    if aligned is None or len(shifted) < plm.MIN_LAGS:
        return None
    values = np.array(shifted)
    return {
        "in_tune_frac_50c": round(aligned, 3),
        "null_median": round(float(np.median(values)), 3),
        "null_best": round(float(values.max()), 3),
        "beats_frac": round(float(np.mean(values < aligned)), 3),
        "n_lags": len(shifted),
    }


def threshold_sweep(out_voc_24k: torch.Tensor, fo: np.ndarray, ft: np.ndarray) -> list[dict[str, Any]]:
    """Loosen YIN's periodicity threshold on the output only; score the frames that gain a pitch.

    The threshold climbs in `LADDER_STEP` steps and a frame keeps the pitch it had
    at the first threshold that gave it one. YIN takes the first dip under its
    threshold, so asking a noisy frame at a loose threshold returns an early
    shallow dip; asking at the tightest threshold that passes returns its deepest.
    """
    target, strict = voiced(ft), voiced(fo)
    merged = fo.copy()
    rows: list[dict[str, Any]] = []
    for threshold in np.arange(THRESHOLDS[0], THRESHOLDS[-1] + LADDER_STEP / 2, LADDER_STEP):
        threshold = round(float(threshold), 2)
        if threshold > THRESHOLDS[0]:
            loose = yin_probe.yin_f0(out_voc_24k, sr=RATE, threshold=threshold)[0].numpy()[: len(fo)]
            take = ~voiced(merged) & voiced(loose)
            merged[take] = loose[take]
        if threshold not in THRESHOLDS:
            continue
        gained = voiced(merged) & ~strict & target
        rows.append(
            {
                "threshold": threshold,
                "voiced_overlap_frac": round(float((voiced(merged) & target).sum()) / max(1, int(target.sum())), 3),
                "gained_frames": int(gained.sum()),
                "gained": gained_frames_null(merged, ft, gained) if int(gained.sum()) else None,
            }
        )
    return rows


def frames_report(out_voc_24k: torch.Tensor, tgt_voc_24k: torch.Tensor) -> dict[str, Any]:
    fo, cmndf, rms = (x.numpy() for x in yin_probe.yin_f0(out_voc_24k, sr=RATE))
    ft, tgt_cmndf, tgt_rms = (x.numpy() for x in yin_probe.yin_f0(tgt_voc_24k, sr=RATE))
    n = min(len(fo), len(ft))
    fo, cmndf, rms, ft, tgt_cmndf, tgt_rms = fo[:n], cmndf[:n], rms[:n], ft[:n], tgt_cmndf[:n], tgt_rms[:n]
    target, out = voiced(ft), voiced(fo)
    scored, missed = target & out, target & ~out

    def middle(values: np.ndarray, frames: np.ndarray) -> list[float] | None:
        return [round(float(v), 2) for v in np.percentile(values[frames], (25, 50, 75))] if frames.any() else None

    return {
        **split_frames(fo, rms, ft),
        "overlap": best_overlap(fo, ft),
        "output_dbfs_q25_50_75": {"scored": middle(rms, scored), "missed": middle(rms, missed)},
        "output_min_cmndf_q25_50_75": {"scored": middle(cmndf, scored), "missed": middle(cmndf, missed)},
        # Which of the target's frames keep a pitch: by how unsteady, how loud and how high the target is there.
        "scored_by_target_quartile": {
            "min_cmndf": scored_by_quartile(tgt_cmndf, target, out),
            "dbfs": scored_by_quartile(tgt_rms, target, out),
            "pitch_hz": scored_by_quartile(np.nan_to_num(ft), target, out),
        },
        "longest_spans_s": {
            "scored": longest_spans(scored),
            "unpitched": longest_spans(missed & (rms >= yin_probe.DEFAULT_MIN_DBFS)),
        },
        "threshold_sweep": threshold_sweep(out_voc_24k, fo, ft),
    }


def periodic_share(f0: np.ndarray, cmndf: np.ndarray, rms_dbfs: np.ndarray) -> dict[str, Any]:
    """Share of the active frames (near the clip's loud level) in which YIN found a pitch."""
    active = rms_dbfs > np.percentile(rms_dbfs, LOUD_PERCENTILE) - ACTIVE_BELOW_LOUD_DB
    return {
        "active_frames": int(active.sum()),
        "pitched_frac_of_active": round(float((voiced(f0) & active).sum()) / max(1, int(active.sum())), 3),
        "min_cmndf_q25_50_75": [round(float(v), 2) for v in np.percentile(cmndf[active], (25, 50, 75))],
        # The register sung, over every pitched frame.
        "pitch_hz_q25_50_75": [round(float(v), 1) for v in np.percentile(f0[voiced(f0)], (25, 50, 75))] if voiced(f0).any() else None,
    }


def highpass(wav_24k: torch.Tensor, hz: float) -> torch.Tensor:
    for _ in range(HIGHPASS_PASSES):
        wav_24k = torchaudio.functional.highpass_biquad(wav_24k, RATE, hz)
    return wav_24k


def _yin(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f0, cmndf, rms = yin_probe.yin_f0(yin_probe.load_mono_audio(path, sr=RATE), sr=RATE)
    return f0.numpy(), cmndf.numpy(), rms.numpy()


def _label(paths: list[Path]) -> list[str]:
    # Stems from one separator share a file name, so fall back to folder and name.
    unique = len({p.name for p in paths}) == len(paths)
    return [p.name if unique else f"{p.parent.name}/{p.name}" for p in paths]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    p_frames = sub.add_parser("frames", help="split the target's voiced frames and sweep the YIN threshold")
    p_frames.add_argument("pairs", nargs="+", type=Path, metavar="VOC", help="OUT_VOC TARGET_VOC, repeated")
    p_frames.add_argument("--highpass", type=float, default=None, metavar="HZ", help="filter each OUT_VOC above HZ first")
    p_periodic = sub.add_parser("periodic", help="share of active frames with a pitch")
    p_periodic.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)

    results: dict[str, Any]
    if args.mode == "frames":
        if len(args.pairs) % 2:
            parser.error("give files in pairs: OUT_VOC TARGET_VOC")
        outputs = args.pairs[0::2]

        def load_out(path: Path) -> torch.Tensor:
            wav = yin_probe.load_mono_audio(path, sr=RATE)
            return wav if args.highpass is None else highpass(wav, args.highpass)

        results = {
            name: {
                "target": target.name,
                "highpass_hz": args.highpass,
                **frames_report(load_out(out), yin_probe.load_mono_audio(target, sr=RATE)),
            }
            for name, out, target in zip(_label(outputs), outputs, args.pairs[1::2])
        }
    else:
        results = {name: periodic_share(*_yin(path)) for name, path in zip(_label(args.files), args.files)}
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
