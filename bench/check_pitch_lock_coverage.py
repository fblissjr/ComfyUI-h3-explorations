#!/usr/bin/env python3
"""Offline checks for `pitch_lock_coverage.py`: the frame split, the threshold sweep's null, agreement, levels and the timbre figure."""

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


def check_agreement_levels_and_periodic_share() -> str:
    target = _melody(1)
    clean = _f0(target)[0]
    gapped_wav = target.clone()
    gapped_wav[_span(0, 8)] = 0.0
    agree = plc.agreement(clean, _f0(gapped_wav)[0], clean)
    assert agree["both"] + agree["a_only"] + agree["b_only"] + agree["neither"] == agree["target_voiced"], agree
    assert agree["b_only"] == 0 and abs(agree["a_only"] - 8 * FRAMES_PER_NOTE) < 0.15 * 8 * FRAMES_PER_NOTE, agree
    assert agree["same_pitch_frac_50c"] == 1.0, agree

    a = torch.stack([target, target]).numpy()
    b = 0.5 * torch.stack([_melody(2), _melody(2)]).numpy()
    levels = plc.stem_levels(a + b, {"a": a, "b": b})
    # Taking a stem out of a two-stem mix leaves the other stem.
    assert abs(levels["a"]["mix_minus_stem_re_mix_db"] - levels["b"]["stem_re_mix_db"]) < 0.01, levels
    assert abs(levels["a"]["stem_re_mix_db"] - levels["b"]["stem_re_mix_db"] - 6.02) < 0.3, levels

    sung = plc.periodic_share(*_f0(target))
    hiss = plc.periodic_share(*_f0(_noise(target.numel(), _rms(target), seed=6)))
    assert sung["pitched_frac_of_active"] > 0.9 and hiss["pitched_frac_of_active"] < 0.05, (sung, hiss)
    # The melody is drawn between MIDI 48 and 66; a copy five semitones down reads that much lower.
    q25, q50, q75 = sung["pitch_hz_q25_50_75"]
    assert 130.0 < q25 <= q50 <= q75 < 371.0, sung
    lower = plc.periodic_share(*_f0(_melody(1, transpose_st=-5.0)))["pitch_hz_q25_50_75"][1]
    assert abs(1200.0 * np.log2(q50 / lower) - 500.0) < 30.0, (q50, lower)
    return (
        f"a stem missing a quarter is a_only in {agree['a_only']} frames; a stem 6 dB down reads "
        f"{levels['b']['stem_re_mix_db']:+.2f} dB against {levels['a']['stem_re_mix_db']:+.2f}; pitched share "
        f"{sung['pitched_frac_of_active']:.2f} for a tone, {hiss['pitched_frac_of_active']:.2f} for noise"
    )


def check_rests_and_highpass_find_a_bass_left_in_the_stem() -> str:
    target = _melody(1)
    for note in range(3, NOTES, 4):  # rests
        target[_span(note, note + 1)] = 0.0
    t = torch.arange(target.numel(), dtype=torch.float32) / float(SR)
    bass = 0.35 * (torch.sin(2.0 * torch.pi * 55.0 * t) + 0.6 * torch.sin(2.0 * torch.pi * 110.0 * t))
    clip = target + bass
    rep = plc.rest_report(target, clip, {"bass": bass, "voice": target})
    # A frame is a rest only when its whole analysis window lies inside one.
    rests = (NOTES // 4) * ((NOTE - plc.BAND_N_FFT) / yin_probe.DEFAULT_HOP_LEN + 1)
    assert abs(rep["rest_frames"] - rests) < 0.15 * rests, (rep["rest_frames"], rests)
    # The rests hold the bass and nothing else: all of it under 150 Hz, all of it in the bass stem.
    assert rep["bands_in_rests_db"]["0-150"] > -0.5 and rep["bands_in_rests_db"]["150-1000"] < -20.0, rep
    assert abs(rep["stems_re_clip_in_rests_db"]["bass"]) < 0.5 and rep["stems_re_clip_in_rests_db"]["voice"] < -20.0, rep
    # While the voice sings its band is in the clip, as it was not in the rests; and a brighter voice reads brighter there.
    assert rep["bands_sung_frames_db"]["150-1000"] > rep["bands_in_rests_db"]["150-1000"] + 20.0, rep
    bright = plc.rest_report(target, target + 3.0 * plc.highpass(target, 1000.0), {})["bands_sung_frames_db"]
    plain = plc.rest_report(target, target, {})["bands_sung_frames_db"]
    assert bright["1000-4000"] - bright["150-1000"] > plain["1000-4000"] - plain["150-1000"] + 6.0, (bright, plain)
    # The bass hides the voice's pitch; a high-pass under the voice's range brings it back.
    hidden = plc.frames_report(clip, target)
    cleared = plc.frames_report(plc.highpass(clip, 150.0), target)
    assert hidden["scored"] < 0.6 * hidden["target_voiced"], (hidden["scored"], hidden["target_voiced"])
    assert cleared["scored"] > 0.9 * cleared["target_voiced"], (cleared["scored"], cleared["target_voiced"])
    return (
        f"a bass under a voice with rests: {rep['rest_frames']} rest frames, all under 150 Hz and all in the bass stem; "
        f"{hidden['scored']} of {hidden['target_voiced']} frames scored with the bass in, {cleared['scored']} high-passed"
    )


def check_a_bed_under_a_voice_costs_frames_by_level() -> str:
    kept = plc.kept_under_bed(_melody(1), _melody(2))["kept_frac_by_bed_db"]
    values = [kept[k] for k in ("-20", "-15", "-10", "-5", "+0")]
    assert values == sorted(values, reverse=True), kept
    # Far under the voice it costs almost nothing; at the voice's level it costs at least a fifth of the frames.
    assert values[0] > 0.95 and values[-1] < values[0] - 0.2, kept
    return f"a second melody under a voice: {values[0]:.2f} of its frames keep a pitch at -20 dB, {values[-1]:.2f} at equal level"


def check_ltas_moves_with_a_bass_under_the_same_voice() -> str:
    voice = _melody(1)
    t = torch.arange(voice.numel(), dtype=torch.float32) / float(SR)
    bass = 0.25 * torch.sin(2.0 * torch.pi * 55.0 * t)
    same = plc.ltas_cosine(voice, voice)
    with_bass = plc.ltas_cosine(voice + bass, voice)
    other_notes = plc.ltas_cosine(_melody(2), voice)
    assert abs(same - 1.0) < 1e-4, same
    # Nothing about the voice changed, and the figure falls further than it does for a different melody.
    assert with_bass < other_notes - 0.05, (with_bass, other_notes)
    return (
        f"a voice against itself reads {same:.2f}, a different melody in the same voice {other_notes:.2f}, "
        f"and the same voice with a bass under it {with_bass:.2f}"
    )


def main() -> int:
    case("split_names_silence_and_noise", check_split_names_silence_and_noise)
    case("rough_voice_is_credited_and_beats_its_null", check_rough_voice_is_credited_and_beats_its_null)
    case("wrong_melody_under_noise_gets_no_credit", check_wrong_melody_under_noise_gets_no_credit)
    case("quartile_split_finds_which_notes_were_lost", check_quartile_split_finds_which_notes_were_lost)
    case("overlap_finds_a_late_output", check_overlap_finds_a_late_output)
    case("agreement_levels_and_periodic_share", check_agreement_levels_and_periodic_share)
    case("rests_and_highpass_find_a_bass_left_in_the_stem", check_rests_and_highpass_find_a_bass_left_in_the_stem)
    case("a_bed_under_a_voice_costs_frames_by_level", check_a_bed_under_a_voice_costs_frames_by_level)
    case("ltas_moves_with_a_bass_under_the_same_voice", check_ltas_moves_with_a_bass_under_the_same_voice)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
