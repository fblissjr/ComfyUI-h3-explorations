#!/usr/bin/env python3
"""Offline checks for `pitch_lock_coverage.py`: frame split, threshold sweep null, quartiles, lag overlap, periodic share, and high-pass."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402
from check_pitch_lock_metrics import NOTE_S, NOTES, SR, _melody  # noqa: E402
import pitch_lock_coverage as plc  # noqa: E402
import probe_yin_f0_drift as yin_probe  # noqa: E402

NOTE = int(round(NOTE_S * SR))
FRAMES_PER_NOTE = NOTE_S / plc.HOP_S
# Noise this far under the tone leaves a pitch YIN's default threshold refuses and a looser one finds.
ROUGH_NOISE_DB = -6.0


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x**2)))


def _noise(n: int, level: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g)
    return x * (level / _rms(x))


def _span(first_note: int, last_note: int) -> slice:
    return slice(first_note * NOTE, last_note * NOTE)


def _f0(wav: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f0, cmndf, rms = yin_probe.yin_f0(wav, sr=SR)
    return f0.numpy(), cmndf.numpy(), rms.numpy()


def check_split_names_silence_and_noise() -> str:
    target = _melody(1)
    out = target.clone()
    out[_span(8, 16)] = 0.0  # a quarter of the clip silent
    out[_span(16, 24)] = _noise(8 * NOTE, _rms(target), seed=3)  # a quarter as loud as the tone, with no pitch
    fo, _, rms = _f0(out)
    ft = _f0(target)[0]
    n = min(len(fo), len(ft))
    split = plc.split_frames(fo[:n], rms[:n], ft[:n])
    quarter = 8 * FRAMES_PER_NOTE
    assert split["scored"] + split["silent"] + split["unpitched"] == split["target_voiced"], split
    assert abs(split["silent"] - quarter) < 0.15 * quarter, split
    assert abs(split["unpitched"] - quarter) < 0.15 * quarter, split
    assert abs(split["scored"] - 2 * quarter) < 0.15 * 2 * quarter, split
    # The noise quarter is found as one span, at the time it was put in.
    spans = plc.frames_report(out, target)["longest_spans_s"]["unpitched"]
    assert len(spans) == 1 and abs(spans[0][0] - 16 * NOTE_S) < 0.1 and abs(spans[0][1] - 24 * NOTE_S) < 0.1, spans
    return (
        f"of {split['target_voiced']} target frames: {split['scored']} scored, {split['silent']} silent, "
        f"{split['unpitched']} unpitched, for a clip that is half sung, a quarter silent and a quarter noise"
    )


def check_rough_voice_is_credited_and_beats_its_null() -> str:
    target = _melody(1)
    out = target.clone()
    level = _rms(target) * 10.0 ** (ROUGH_NOISE_DB / 20.0)
    out[_span(16, NOTES)] += _noise((NOTES - 16) * NOTE, level, seed=4)  # the second half sung under noise
    rep = plc.frames_report(out, target)
    strict, loosest = rep["threshold_sweep"][0], rep["threshold_sweep"][-1]
    assert strict["gained_frames"] == 0 and strict["voiced_overlap_frac"] < 0.65, strict
    assert rep["unpitched"] > 0.7 * 16 * FRAMES_PER_NOTE and rep["silent"] == 0, rep
    gained = loosest["gained"]
    assert loosest["voiced_overlap_frac"] > 0.9 and gained is not None, loosest
    assert gained["in_tune_frac_50c"] > 0.9 and gained["beats_frac"] == 1.0, gained
    return (
        f"the right melody under noise: {rep['unpitched']} frames unpitched at the default threshold; at "
        f"{loosest['threshold']} they gain a pitch, {gained['in_tune_frac_50c']:.2f} in tune, beating all "
        f"{gained['n_lags']} shifts"
    )


def check_wrong_melody_under_noise_gets_no_credit() -> str:
    target, other = _melody(1), _melody(2)
    out = target.clone()
    level = _rms(target) * 10.0 ** (ROUGH_NOISE_DB / 20.0)
    out[_span(16, NOTES)] = other[_span(16, NOTES)] + _noise((NOTES - 16) * NOTE, level, seed=4)
    rep = plc.frames_report(out, target)
    gained = rep["threshold_sweep"][-1]["gained"]
    assert gained is not None, rep["threshold_sweep"]
    assert gained["in_tune_frac_50c"] < 0.3 and gained["in_tune_frac_50c"] <= gained["null_best"], gained
    # Pure noise in place of the second half gains almost nothing even at the loosest threshold.
    out[_span(16, NOTES)] = _noise((NOTES - 16) * NOTE, _rms(target), seed=5)
    noise_rep = plc.frames_report(out, target)
    noise_gained = noise_rep["threshold_sweep"][-1]["gained_frames"]
    assert noise_gained < 0.1 * 16 * FRAMES_PER_NOTE, noise_rep["threshold_sweep"]
    return (
        f"a different melody under the same noise: gained frames {gained['in_tune_frac_50c']:.2f} in tune, "
        f"inside a null whose best is {gained['null_best']:.2f}; plain noise gains {noise_gained} frames"
    )


def check_quartile_split_finds_which_notes_were_lost() -> str:
    target = _melody(1)
    ft = _f0(target)[0]
    note_hz = [float(np.nanmedian(ft[int(k * FRAMES_PER_NOTE) + 5 : int((k + 1) * FRAMES_PER_NOTE) - 5])) for k in range(NOTES)]
    out = target.clone()
    for k in np.argsort(note_hz)[: NOTES // 2]:  # the lower half of the notes sung as noise
        out[_span(int(k), int(k) + 1)] = _noise(NOTE, _rms(target), seed=10 + int(k))
    by_pitch = plc.frames_report(out, target)["scored_by_target_quartile"]["pitch_hz"]
    low, high = by_pitch["scored_frac"][0], by_pitch["scored_frac"][-1]
    assert by_pitch["upper_edges"] == sorted(by_pitch["upper_edges"]), by_pitch
    assert low < 0.15 and high > 0.85, by_pitch
    return f"with the lower half of the notes lost, the lowest quarter by target pitch is {low:.2f} scored and the highest {high:.2f}"


def check_overlap_finds_a_late_output() -> str:
    target = _melody(1)
    for note in range(3, NOTES, 4):  # rests, so the voicing has a pattern to line up
        target[_span(note, note + 1)] = 0.0
    late = int(0.3 * SR)
    out = torch.cat([torch.zeros(late), target[:-late]])
    on_time = plc.best_overlap(_f0(target)[0], _f0(target)[0])
    shifted = plc.best_overlap(_f0(out)[0], _f0(target)[0])
    assert on_time["best_shift_ms"] == 0 and on_time["best"] == on_time["at_no_shift"], on_time
    assert abs(shifted["best_shift_ms"] - 300) <= 20 and shifted["best"] > shifted["at_no_shift"], shifted
    return f"an output 300 ms late is found at {shifted['best_shift_ms']:+d} ms; an on-time one at 0"


def check_periodic_share_and_highpass() -> str:
    target = _melody(1)
    sung = plc.periodic_share(*_f0(target))
    hiss = plc.periodic_share(*_f0(_noise(target.numel(), _rms(target), seed=6)))
    assert sung["pitched_frac_of_active"] > 0.9 and hiss["pitched_frac_of_active"] < 0.05, (sung, hiss)
    q25, q50, q75 = sung["pitch_hz_q25_50_75"]
    assert 130.0 < q25 <= q50 <= q75 < 371.0, sung
    lower = plc.periodic_share(*_f0(_melody(1, transpose_st=-5.0)))["pitch_hz_q25_50_75"][1]
    assert abs(1200.0 * np.log2(q50 / lower) - 500.0) < 30.0, (q50, lower)

    t = torch.arange(target.numel(), dtype=torch.float32) / float(SR)
    bass = 0.35 * (torch.sin(2.0 * torch.pi * 55.0 * t) + 0.6 * torch.sin(2.0 * torch.pi * 110.0 * t))
    clip = target + bass
    hidden = plc.frames_report(clip, target)
    cleared = plc.frames_report(plc.highpass(clip, 150.0), target)
    assert hidden["scored"] < 0.6 * hidden["target_voiced"], (hidden["scored"], hidden["target_voiced"])
    assert cleared["scored"] > 0.9 * cleared["target_voiced"], (cleared["scored"], cleared["target_voiced"])
    return (
        f"pitched share {sung['pitched_frac_of_active']:.2f} for a tone, {hiss['pitched_frac_of_active']:.2f} for noise; "
        f"high-pass lifts scored frames under 55/110 Hz bass from {hidden['scored']} to {cleared['scored']} of {cleared['target_voiced']}"
    )


def main() -> int:
    case("split_names_silence_and_noise", check_split_names_silence_and_noise)
    case("rough_voice_is_credited_and_beats_its_null", check_rough_voice_is_credited_and_beats_its_null)
    case("wrong_melody_under_noise_gets_no_credit", check_wrong_melody_under_noise_gets_no_credit)
    case("quartile_split_finds_which_notes_were_lost", check_quartile_split_finds_which_notes_were_lost)
    case("overlap_finds_a_late_output", check_overlap_finds_a_late_output)
    case("periodic_share_and_highpass", check_periodic_share_and_highpass)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
