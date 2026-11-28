#!/usr/bin/env python3
"""Offline checks for `pitch_lock_metrics.py`: the legacy columns, the time-shift null, the contour fix."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402
import pitch_lock_metrics as plm  # noqa: E402

SR = plm.RATE
NOTE_S = 0.25
NOTES = 32  # 8 s, the first gate's clip length


def _melody(seed: int, *, transpose_st: float = 0.0) -> torch.Tensor:
    """A seeded melody with no repeated phrase: one harmonic tone per note, pitches drawn over an octave and a half."""
    rng = np.random.default_rng(seed)
    midi = rng.integers(48, 67, size=NOTES) + transpose_st
    t = torch.arange(int(round(NOTE_S * SR)), dtype=torch.float32) / float(SR)
    tones = []
    for note in midi:
        hz = 440.0 * 2.0 ** ((float(note) - 69.0) / 12.0)
        tones.append(
            0.25
            * (
                0.65 * torch.sin(2.0 * math.pi * hz * t)
                + 0.25 * torch.sin(2.0 * math.pi * 2.0 * hz * t)
                + 0.10 * torch.sin(2.0 * math.pi * 3.0 * hz * t)
            )
        )
    return torch.cat(tones)


def check_legacy_columns_on_known_tracks() -> str:
    # Every frame 240 cents sharp: the fold keeps it, and +2 st is the nearest whole transposition.
    ft = np.full(40, 200.0)
    fo = ft * 2.0 ** (240.0 / 1200.0)
    s = plm.score_f0(fo, ft)
    assert s is not None
    assert abs(s["med_abs_cent_err"] - 240.0) < 1e-6 and abs(s["med_signed_cent_bias"] - 240.0) < 1e-6, s
    assert s["in_tune_frac_50c"] == 0.0 and s["best_tr_st"] == 2.0, s
    assert abs(s["rel_med_abs_cent_err"] - 40.0) < 1e-6 and s["rel_in_tune_frac_50c"] == 1.0, s

    # An octave and 30 cents up folds to 30 cents, and no transposition beats zero.
    s = plm.score_f0(ft * 2.0 ** (1230.0 / 1200.0), ft)
    assert s is not None and abs(s["med_abs_cent_err"] - 30.0) < 1e-6 and s["best_tr_st"] == 0.0, s

    # Unvoiced frames are NaN; too few shared frames means no score.
    sparse = ft.copy()
    sparse[plm.MIN_SHARED_FRAMES - 1 :] = np.nan
    assert plm.score_f0(sparse, ft) is None
    return "240 c sharp scores 240 / +2 st / 40 c, an octave folds away, and too few shared frames returns None"


def check_identical_and_transposed_pairs_beat_every_shift() -> str:
    target = _melody(1)
    same = plm.vocal_pitch_metrics(target, target)
    assert same["rel_med_abs_cent_err"] < 5.0 and same["rel_in_tune_frac_50c"] > 0.95, same
    assert same["vocal_f0_r_one_shift"] > 0.95, same
    for column, _ in plm.NULL_COLUMNS:
        assert same["lag_null"][column]["beats_frac"] == 1.0, (column, same["lag_null"])

    up3 = plm.vocal_pitch_metrics(_melody(1, transpose_st=3.0), target)
    assert up3["best_tr_st"] == 3 and up3["rel_med_abs_cent_err"] < 10.0, up3
    assert abs(up3["med_signed_cent_bias"] - 300.0) < 15.0, up3
    assert up3["lag_null"]["rel_med_abs_cent_err"]["beats_frac"] == 1.0, up3["lag_null"]
    assert up3["lag_null"]["rel_in_tune_frac_50c"]["beats_frac"] == 1.0, up3["lag_null"]
    return (
        f"a copy and a +3 st copy beat all {same['lag_null']['n_lags']} time shifts; "
        f"the copy's transposition is recovered as {up3['best_tr_st']:+d} st"
    )


def check_unrelated_pair_sits_inside_its_null() -> str:
    out, target = _melody(2), _melody(1)
    m = plm.vocal_pitch_metrics(out, target)
    null = m["lag_null"]
    # The aligned pair is no better placed than a shifted one, on either transposed column.
    assert null["rel_med_abs_cent_err"]["beats_frac"] < 0.9, null
    assert null["rel_in_tune_frac_50c"]["beats_frac"] < 0.9, null
    # The per-frame octave alignment reports agreement that is not there; one shift for the clip does not.
    assert m["vocal_f0_r"] > 0.5, m
    assert abs(m["vocal_f0_r_one_shift"]) < 0.4, m
    assert m["vocal_f0_r"] - m["vocal_f0_r_one_shift"] > 0.3, m
    return (
        f"two unrelated melodies: legacy vocal_f0_r {m['vocal_f0_r']:.3f} against one-shift "
        f"{m['vocal_f0_r_one_shift']:.3f}; aligned rel error beats {null['rel_med_abs_cent_err']['beats_frac']:.2f} "
        "of its shifts"
    )


def check_silence_and_short_clips_return_nones() -> str:
    target = _melody(1)
    silent = plm.vocal_pitch_metrics(torch.zeros_like(target), target)
    assert silent["vocal_med_f0_hz"] is None and silent["mut_voiced_frames"] == 0, silent
    assert all(silent[column] is None for column in plm.LEGACY_COLUMNS), silent
    assert silent["vocal_f0_r_one_shift"] is None and silent["lag_null"] is None, silent

    # Scored, but too short to hold MIN_LAGS shifts clear of both ends: columns present, null absent.
    short = _melody(1)[: int(2.5 * SR)]
    m = plm.vocal_pitch_metrics(short, short)
    assert m["rel_med_abs_cent_err"] is not None and m["lag_null"] is None, m
    return "a silent output returns every column as None; a clip too short for the null is scored without one"


def main() -> int:
    case("legacy_columns_on_known_tracks", check_legacy_columns_on_known_tracks)
    case("identical_and_transposed_pairs_beat_every_shift", check_identical_and_transposed_pairs_beat_every_shift)
    case("unrelated_pair_sits_inside_its_null", check_unrelated_pair_sits_inside_its_null)
    case("silence_and_short_clips_return_nones", check_silence_and_short_clips_return_nones)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
