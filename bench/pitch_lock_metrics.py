#!/usr/bin/env python3
"""Vocal pitch-lock scoring, with the level a misaligned pair reaches.

Scores how closely an output vocal follows a target vocal: octave-folded cent
error, the best whole-semitone transposition, and a contour correlation. The
legacy columns are computed exactly as
`music_cover_pitch_lock_gate.py::_vocal_pitch_metrics` computes them, so earlier
records stay comparable and that function can be replaced by a call to
`vocal_pitch_metrics` here.

Two things are added, because a figure alone does not say whether the output
followed the target:

- **A null for every pair (`lag_null`).** The output is scored again against
  the target rotated in time, at each of a set of shifts. That keeps the
  target's key, range and amount of singing and removes only the alignment,
  and every shift gets the same transposition search the aligned score got.
  For each column it reports the null's median, the null's best value, and the
  share of shifts the aligned score beats. An aligned score that sits inside
  its null has not shown lock, whatever its size. A melody that repeats within
  the clip scores well at the shift of its own period, which makes the null
  harder to beat, not easier.
- **`vocal_f0_r_one_shift`.** `vocal_f0_r` moves every output frame to the
  octave nearest its target frame before correlating, which copies the target's
  contour into the output. The one-shift column moves the whole clip by one
  octave count. `vocal_f0_r` is kept so old tables can be re-derived; do not
  read it as contour agreement. The record is
  `bench/results/2026-11-27_music_cover_pitch_lock_metric_null.md`.
- **Coverage (`voiced_overlap_frac`).** Every column is computed only on
  frames where both tracks have a pitch. This is the share of the target's
  voiced frames that the output also voices, so it says how much of the melody
  a score rests on. An output that sings a tenth of the notes in tune scores
  as in tune.

    <comfy-venv-python> bench/pitch_lock_metrics.py OUT_VOC TARGET_VOC [OUT_VOC TARGET_VOC ...] [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_yin_f0_drift as yin_probe  # noqa: E402

RATE = 24000
HOP_S = yin_probe.DEFAULT_HOP_LEN / RATE
MIN_SHARED_FRAMES = 12
TRANSPOSITIONS = range(-6, 7)
# Time shifts for the null: every LAG_STEP_S, staying LAG_EDGE_S away from the
# aligned position at both ends so a slightly late output is not counted as a
# misaligned one. Reasoned, not tuned.
LAG_STEP_S = 0.25
LAG_EDGE_S = 1.0
MIN_LAGS = 8
# Column name, and whether a lower value is the better one.
NULL_COLUMNS = (
    ("med_abs_cent_err", True),
    ("in_tune_frac_50c", False),
    ("rel_med_abs_cent_err", True),
    ("rel_in_tune_frac_50c", False),
    ("vocal_f0_r_one_shift", False),
    # Legacy column, included so its null is on record wherever it is still printed.
    ("vocal_f0_r", False),
)
LEGACY_COLUMNS = (
    "med_abs_cent_err",
    "med_signed_cent_bias",
    "in_tune_frac_50c",
    "best_tr_st",
    "rel_med_abs_cent_err",
    "rel_in_tune_frac_50c",
    "vocal_f0_r",
)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    fa, fb = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(fa) * np.linalg.norm(fb))
    return float(np.dot(fa, fb) / denom) if denom > 1e-12 else 0.0


def score_f0(fo: np.ndarray, ft: np.ndarray) -> dict[str, float] | None:
    """Score two frame-aligned F0 tracks in Hz (NaN where unvoiced), unrounded.

    Returns None when fewer than `MIN_SHARED_FRAMES` frames are voiced in both.
    """
    n = min(len(fo), len(ft))
    fo, ft = fo[:n], ft[:n]
    valid = np.isfinite(fo) & np.isfinite(ft) & (fo > 0) & (ft > 0)
    if int(np.sum(valid)) < MIN_SHARED_FRAMES:
        return None
    # Octave-folded cent error in [-600, +600).
    raw_cents = 1200.0 * np.log2(fo[valid] / ft[valid])
    folded = ((raw_cents + 600.0) % 1200.0) - 600.0
    abs_err = np.abs(folded)

    best_st, best_rel_med, best_rel_in50 = 0, 999.0, 0.0
    for k in TRANSPOSITIONS:
        d_k = ((raw_cents - 100.0 * k + 600.0) % 1200.0) - 600.0
        m_k = float(np.median(np.abs(d_k)))
        if m_k < best_rel_med:
            best_st, best_rel_med = k, m_k
            best_rel_in50 = float(np.mean(np.abs(d_k) <= 50.0))

    log_o, log_t = np.log2(fo[valid]), np.log2(ft[valid])
    octaves = log_t - log_o
    return {
        "mut_voiced_frames": float(np.sum(valid)),
        "med_abs_cent_err": float(np.median(abs_err)),
        "med_signed_cent_bias": float(np.median(folded)),
        "in_tune_frac_50c": float(np.mean(abs_err <= 50.0)),
        "best_tr_st": float(best_st),
        "rel_med_abs_cent_err": best_rel_med,
        "rel_in_tune_frac_50c": best_rel_in50,
        "vocal_f0_r": _corr(log_o + np.round(octaves), log_t),
        "vocal_f0_r_one_shift": _corr(log_o + np.round(np.median(octaves)), log_t),
    }


def lag_null(fo: np.ndarray, ft: np.ndarray, *, hop_s: float = HOP_S) -> dict[str, Any] | None:
    """Score `fo` against `ft` rotated in time; say where the aligned score sits among the shifts.

    Returns None when the aligned pair cannot be scored or fewer than
    `MIN_LAGS` shifts can.
    """
    n = min(len(fo), len(ft))
    fo, ft = fo[:n], ft[:n]
    aligned = score_f0(fo, ft)
    step, edge = max(1, round(LAG_STEP_S / hop_s)), round(LAG_EDGE_S / hop_s)
    shifted = [s for s in (score_f0(fo, np.roll(ft, lag)) for lag in range(edge, n - edge, step)) if s]
    if aligned is None or len(shifted) < MIN_LAGS:
        return None
    out: dict[str, Any] = {"n_lags": len(shifted)}
    for column, lower_is_better in NULL_COLUMNS:
        values = np.array([s[column] for s in shifted])
        beaten = values > aligned[column] if lower_is_better else values < aligned[column]
        out[column] = {
            "null_median": round(float(np.median(values)), 3),
            "null_best": round(float(values.min() if lower_is_better else values.max()), 3),
            "beats_frac": round(float(np.mean(beaten)), 3),
        }
    return out


def vocal_pitch_metrics(out_voc_24k: torch.Tensor, tgt_voc_24k: torch.Tensor) -> dict[str, Any]:
    """Score an output vocal against a target vocal, both mono at 24 kHz.

    The first nine keys and their rounding match the gate's
    `_vocal_pitch_metrics`. `vocal_f0_r_one_shift`, `lag_null` and the two
    coverage keys follow.
    """
    fo = yin_probe.yin_f0(out_voc_24k, sr=RATE)[0].numpy()
    ft = yin_probe.yin_f0(tgt_voc_24k, sr=RATE)[0].numpy()
    voiced_out = fo[np.isfinite(fo) & (fo > 0)]
    med_voc = float(np.median(voiced_out)) if len(voiced_out) else float("nan")
    result: dict[str, Any] = {"vocal_med_f0_hz": round(med_voc, 2) if math.isfinite(med_voc) else None}
    n = min(len(fo), len(ft))
    target_voiced = np.isfinite(ft[:n]) & (ft[:n] > 0)
    shared = int(np.sum(target_voiced & np.isfinite(fo[:n]) & (fo[:n] > 0)))
    coverage = {
        "tgt_voiced_frames": int(np.sum(target_voiced)),
        "voiced_overlap_frac": round(shared / int(np.sum(target_voiced)), 3) if np.any(target_voiced) else None,
    }
    score = score_f0(fo, ft)
    if score is None:
        result["mut_voiced_frames"] = shared
        result.update({column: None for column in LEGACY_COLUMNS})
        result.update({"vocal_f0_r_one_shift": None, "lag_null": None, **coverage})
        return result
    result.update(
        {
            "mut_voiced_frames": int(score["mut_voiced_frames"]),
            "med_abs_cent_err": round(score["med_abs_cent_err"], 1),
            "med_signed_cent_bias": round(score["med_signed_cent_bias"], 1),
            "in_tune_frac_50c": round(score["in_tune_frac_50c"], 3),
            "best_tr_st": int(score["best_tr_st"]),
            "rel_med_abs_cent_err": round(score["rel_med_abs_cent_err"], 1),
            "rel_in_tune_frac_50c": round(score["rel_in_tune_frac_50c"], 3),
            "vocal_f0_r": round(score["vocal_f0_r"], 3),
            "vocal_f0_r_one_shift": round(score["vocal_f0_r_one_shift"], 3),
            "lag_null": lag_null(fo, ft),
            **coverage,
        }
    )
    return result


def _report(name: str, metrics: dict[str, Any]) -> str:
    lines = [name]
    if metrics["med_abs_cent_err"] is None:
        return f"{name}\n  too few frames voiced in both ({metrics['mut_voiced_frames']})"
    lines.append(
        f"  med_abs_cent_err {metrics['med_abs_cent_err']}  bias {metrics['med_signed_cent_bias']:+}  "
        f"best_tr_st {metrics['best_tr_st']:+d}  vocal_f0_r {metrics['vocal_f0_r']} (legacy)"
    )
    lines.append(
        f"  scored on {metrics['mut_voiced_frames']} frames voiced in both, of {metrics['tgt_voiced_frames']} "
        f"the target voices (voiced_overlap_frac {metrics['voiced_overlap_frac']})"
    )
    null = metrics["lag_null"]
    if null is None:
        lines.append("  no null: too few time shifts could be scored")
        return "\n".join(lines)
    lines.append(f"  {'column':24s} {'aligned':>9s} {'null median':>12s} {'null best':>10s} {'beats':>6s}")
    for column, _ in NULL_COLUMNS:
        row = null[column]
        lines.append(
            f"  {column:24s} {metrics[column]:9.3f} {row['null_median']:12.3f} {row['null_best']:10.3f} "
            f"{row['beats_frac']:6.2f}"
        )
    lines.append(f"  beats = share of {null['n_lags']} time shifts the aligned score is better than")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("pairs", nargs="+", type=Path, metavar="VOC", help="OUT_VOC TARGET_VOC, repeated")
    parser.add_argument("--json", action="store_true", help="print one JSON object keyed by output file")
    args = parser.parse_args(argv)
    if len(args.pairs) % 2:
        parser.error("give files in pairs: OUT_VOC TARGET_VOC")
    results: dict[str, Any] = {}
    outputs = args.pairs[0::2]
    # Stems from one separator share a file name, so fall back to folder and name.
    unique = len({p.name for p in outputs}) == len(outputs)
    for out_path, target_path in zip(outputs, args.pairs[1::2]):
        results[out_path.name if unique else f"{out_path.parent.name}/{out_path.name}"] = {
            "target": target_path.name,
            **vocal_pitch_metrics(
                yin_probe.load_mono_audio(out_path, sr=RATE), yin_probe.load_mono_audio(target_path, sr=RATE)
            ),
        }
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n\n".join(_report(f"{name}  vs  {m['target']}", m) for name, m in results.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
