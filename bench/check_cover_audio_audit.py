#!/usr/bin/env python3
"""Offline self-tests for `bench/cover_audio_audit.py` on known synthetic signals."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "bench") not in sys.path:
    sys.path.insert(0, str(REPO / "bench"))

import cover_audio_audit as audit


def _synth_melody(duration_s: float = 12.25, notes_hz: list[float] | None = None, *, bright: bool = False) -> np.ndarray:
    n = int(round(duration_s * audit.SR))
    t = np.arange(n, dtype=np.float64) / audit.SR
    notes = notes_hz or [220.0, 261.63, 293.66, 329.63, 392.0, 329.63, 261.63, 196.0]
    sig = np.zeros(n, dtype=np.float64)
    step = n // len(notes)
    for idx, f0 in enumerate(notes):
        s, e = idx * step, (idx + 1) * step if idx + 1 < len(notes) else n
        seg_t = t[s:e] - t[s]
        env = np.sin(math.pi * np.clip(seg_t / max(seg_t[-1], 1e-6), 0.0, 1.0)) ** 0.3
        if bright:
            harm = 0.3 * np.sin(2.0 * math.pi * f0 * seg_t) + sum(
                (1.0 / (h ** 0.4)) * np.sin(2.0 * math.pi * h * f0 * seg_t) for h in range(4, 16)
            )
        else:
            harm = sum((1.0 / (h ** 1.8)) * np.sin(2.0 * math.pi * h * f0 * seg_t) for h in range(1, 8))
        sig[s:e] = 0.35 * env * harm
    return np.stack([sig, sig], axis=0).astype(np.float32)


def main() -> int:
    # 1. Identical harmonic melody passes CPP, stereo coherence, and circular lag null
    mel_a = _synth_melody()
    mid_a = mel_a[0].astype(np.float64)
    nat_clean = audit.measure_naturalness(mel_a)
    mr_ident = audit.compare_melody_and_rhythm(mid_a, mid_a)
    assert nat_clean["stereo"]["full"]["lr_r"] > 0.999
    assert nat_clean["stereo"]["vocal_300_3500"]["lr_r"] > 0.999
    assert nat_clean["cpp_db_med"] > 1.0, f"expected clean harmonic CPP > 1 dB, got {nat_clean['cpp_db_med']}"
    assert mr_ident["n_lags"] >= 35, f"expected >= 35 lags, got {mr_ident['n_lags']}"
    assert mr_ident["chroma_beats_frac"] == 1.0, f"expected 1.0 beats_frac, got {mr_ident}"
    assert mr_ident["onset_beats_frac"] == 1.0, f"expected 1.0 onset_beats_frac, got {mr_ident}"
    print(f"ok    identical: cpp={nat_clean['cpp_db_med']:.2f} dB, chroma_lift={mr_ident['chroma_lift']:.3f}, beats={mr_ident['chroma_beats_frac']}")

    # 2. Phase-scrambled R channel drops vocal-band L/R_r & raises Side/Mid while noisy smear kills CPP/flatness
    rng = np.random.default_rng(7)
    r_spec = np.fft.rfft(mel_a[1].astype(np.float64))
    r_scram = np.fft.irfft(np.abs(r_spec) * np.exp(1j * rng.uniform(-math.pi, math.pi, len(r_spec))), n=mel_a.shape[1])
    phasey = np.stack([mel_a[0], r_scram.astype(np.float32)], axis=0)
    nat_phasey = audit.measure_naturalness(phasey)
    assert nat_phasey["stereo"]["vocal_300_3500"]["lr_r"] < 0.25, f"expected vocal lr_r drop, got {nat_phasey['stereo']}"
    assert nat_phasey["stereo"]["vocal_300_3500"]["side_mid_db"] > -3.0

    noisy_mix = (0.4 * mel_a + 0.25 * rng.standard_normal(mel_a.shape)).astype(np.float32)
    nat_noisy = audit.measure_naturalness(noisy_mix)
    assert nat_noisy["cpp_db_med"] < nat_clean["cpp_db_med"]
    assert nat_noisy["flatness_1_4k_med"] > nat_clean["flatness_1_4k_med"]
    print(f"ok    phasey/noisy: vocal_lr_r={nat_phasey['stereo']['vocal_300_3500']['lr_r']:.3f}, noisy_cpp={nat_noisy['cpp_db_med']:.2f} < {nat_clean['cpp_db_med']:.2f}")

    # 3. Circularly time-shifted melody fails lag null
    shifted_mid = np.roll(mid_a, int(round(3.1 * audit.SR)))
    mr_shift = audit.compare_melody_and_rhythm(shifted_mid, mid_a)
    assert mr_shift["chroma_beats_frac"] < 0.85, f"expected shifted copy to fail lag null, got {mr_shift}"
    print(f"ok    shifted_null: chroma_beats_frac={mr_shift['chroma_beats_frac']}")

    # 4. Timbre swap margin separates dark warm B vs bright A cleanly regardless of melody
    ref_b = _synth_melody(notes_hz=[196.0, 220.0, 246.94, 196.0], bright=False)
    ref_a = _synth_melody(notes_hz=[220.0, 261.63, 293.66, 329.63], bright=True)
    # Candidate sings A's melody with B's warm spectral envelope -> must score closer to B than A
    cand = _synth_melody(notes_hz=[220.0, 261.63, 293.66, 329.63], bright=False)
    mb = audit.extract_mfcc_profile(ref_b[0].astype(np.float64))
    ma = audit.extract_mfcc_profile(ref_a[0].astype(np.float64))
    mc = audit.extract_mfcc_profile(cand[0].astype(np.float64))
    margin_b = float(np.dot(mc, mb) - np.dot(mc, ma))
    margin_a = float(np.dot(ma, mb) - np.dot(ma, ma))
    assert margin_b > 0.05 and margin_a < -0.05, f"expected timbre separation, got {margin_b}, {margin_a}"
    print(f"ok    timbre_margin: b_like={margin_b:+.3f}, a_like={margin_a:+.3f}")

    print("\nall 4 case(s) ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
