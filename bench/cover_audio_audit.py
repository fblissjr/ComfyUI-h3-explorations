#!/usr/bin/env python3
"""Independent first-principles audio audit for cover-song renders (zero RoFormer/YIN dependency).

Loads any audio or video file directly as 32 kHz stereo float32 via ffmpeg and scores
four independent axes directly on the full output waveform:
  1. Naturalness / Anti-Sheen: multi-band L/R coherence & Side/Mid dB, active-frame
     vocal-band (300-3500 Hz) Cepstral Peak Prominence (cpp_db), 1-4 kHz spectral
     flatness, mid presence ratio, HF air ratio, and crest factor.
  2. Melody / Harmony Lock: 12-bin log-frequency harmonic-salience chromagram (40 Hz)
     against a shifted target mix with circular lag-null (chroma_lift, beats_frac).
  3. Rhythm / Onset Lock: positive log-spectral flux envelope correlation (+/- 50 ms)
     against circular lag-null.
  4. Singer / Style Transfer: 12-dim MFCC (c1..c12) timbral centroid cosine against
     Target B ref vs Target A shifted mix (timbre_margin = sim_B - sim_A).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

SR = 32000
HOP = 800  # 40 Hz H3 audio latent rate
N_FFT_SPEC = 2048
N_FFT_CHROMA = 4096
ACTIVE_BELOW_P95_DB = 22.0
MIN_ACTIVE_DBFS = -55.0
LAG_EXCL_S = 1.0
LAG_STEP_FRAMES = 4


def load_audio_32k(path: Path) -> np.ndarray:
    """Decode any media path to float32 [2, N] at 32 kHz stereo via ffmpeg."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing media file: {p}")
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(p),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "2",
            "-ar",
            str(SR),
            "-",
        ]
    )
    arr = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()
    return arr


def _frame_view(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    strides = (x.strides[0] * hop, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=(n_frames, n_fft), strides=strides).copy()


def _stft_complex(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    frames = _frame_view(x.astype(np.float64), n_fft, hop)
    win = np.hanning(n_fft)
    return np.fft.rfft(frames * win[None, :], axis=-1)


def _active_mask_from_mid(mid: np.ndarray, hop: int = HOP, n_fft: int = N_FFT_SPEC) -> np.ndarray:
    frames = _frame_view(mid.astype(np.float64), n_fft, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=-1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    if len(db) == 0:
        return np.zeros(0, dtype=bool)
    thr = max(MIN_ACTIVE_DBFS, float(np.percentile(db, 95)) - ACTIVE_BELOW_P95_DB)
    return db >= thr


def measure_stereo_coherence(stereo_32k: np.ndarray) -> dict[str, dict[str, float]]:
    """Compute zero-lag L/R correlation and Side/Mid dB overall and in 3 frequency bands."""
    l = stereo_32k[0].astype(np.float64)
    r = stereo_32k[1].astype(np.float64)
    sl = _stft_complex(l, N_FFT_SPEC, HOP)
    sr = _stft_complex(r, N_FFT_SPEC, HOP)
    freqs = np.fft.rfftfreq(N_FFT_SPEC, d=1.0 / SR)

    bands = {
        "full": (20.0, 16000.0),
        "low_lt300": (20.0, 300.0),
        "vocal_300_3500": (300.0, 3500.0),
        "air_gt3500": (3500.0, 16000.0),
    }
    out: dict[str, dict[str, float]] = {}
    for name, (f_lo, f_hi) in bands.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        lb = sl[:, mask]
        rb = sr[:, mask]
        num = float(np.sum(np.real(lb * np.conj(rb))))
        den = float(math.sqrt((np.sum(np.abs(lb) ** 2) + 1e-12) * (np.sum(np.abs(rb) ** 2) + 1e-12)))
        lr_r = max(-1.0, min(1.0, num / max(den, 1e-12)))
        mid_e = float(np.sum(np.abs(0.5 * (lb + rb)) ** 2))
        side_e = float(np.sum(np.abs(0.5 * (lb - rb)) ** 2))
        sm_db = float(10.0 * np.log10((side_e + 1e-12) / (mid_e + 1e-12)))
        out[name] = {"lr_r": round(lr_r, 4), "side_mid_db": round(sm_db, 2)}
    return out


def measure_naturalness(stereo_32k: np.ndarray) -> dict[str, Any]:
    """Measure vocal-band CPP, 1-4 kHz spectral flatness, presence/air balance, crest factor."""
    mid = 0.5 * (stereo_32k[0].astype(np.float64) + stereo_32k[1].astype(np.float64))
    spec = _stft_complex(mid, N_FFT_SPEC, HOP)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(N_FFT_SPEC, d=1.0 / SR)
    active = _active_mask_from_mid(mid, HOP, N_FFT_SPEC)[: len(power)]
    if not np.any(active):
        active = np.ones(len(power), dtype=bool)

    # 1. Vocal-band Cepstral Peak Prominence (Hillenbrand log-power cepstrum on 0..4500 Hz)
    # Using bins <= 4500 Hz avoids high-frequency H3 roll-off biasing the regression baseline.
    v_bins = freqs <= 4500.0
    sr_ceps = 2.0 * float(freqs[v_bins][-1])
    log_p = 10.0 * np.log10(np.maximum(power[active][:, v_bins], 1e-12))
    ceps = np.fft.irfft(log_p, axis=-1)
    q_lo = max(2, int(round(sr_ceps / 500.0)))
    q_hi = min(ceps.shape[-1] // 2 - 1, int(round(sr_ceps / 70.0)))
    q_axis = np.arange(q_lo, q_hi + 1, dtype=np.float64)
    q_centered = q_axis - np.mean(q_axis)
    q_den = float(np.sum(q_centered * q_centered))
    cpp_vals = []
    for row in ceps[:, q_lo : q_hi + 1]:
        slope = float(np.sum(q_centered * (row - np.mean(row)))) / q_den
        baseline = np.mean(row) + slope * q_centered
        diff = row - baseline
        cpp_vals.append(float(np.max(diff)))
    cpp_arr = np.asarray(cpp_vals, dtype=np.float64) if cpp_vals else np.zeros(1)

    # 2. 1-4 kHz Spectral Flatness (harmonic = near 0, smeared/noisy = higher)
    m_1_4 = (freqs >= 1000.0) & (freqs <= 4000.0)
    p_1_4 = np.maximum(power[active][:, m_1_4], 1e-15)
    gmean = np.exp(np.mean(np.log(p_1_4), axis=-1))
    amean = np.mean(p_1_4, axis=-1)
    flat_1_4 = gmean / np.maximum(amean, 1e-15)

    # 3. Spectral balance ratios & crest factor
    m_low = (freqs >= 200.0) & (freqs < 800.0)
    m_air = (freqs >= 8000.0) & (freqs <= 15000.0)
    e_low = float(np.mean(power[active][:, m_low]))
    e_mid = float(np.mean(power[active][:, m_1_4]))
    e_air = float(np.mean(power[active][:, m_air]))
    mid_presence_db = 10.0 * math.log10((e_mid + 1e-15) / (e_low + 1e-15))
    hf_air_db = 10.0 * math.log10((e_air + 1e-15) / (e_mid + 1e-15))

    rms_all = math.sqrt(float(np.mean(mid * mid)) + 1e-12)
    peak_abs = float(np.max(np.abs(mid)))
    crest_db = 20.0 * math.log10(max(peak_abs, 1e-9) / max(rms_all, 1e-9))

    return {
        "stereo": measure_stereo_coherence(stereo_32k),
        "cpp_db_med": round(float(np.median(cpp_arr)), 3),
        "cpp_db_mean": round(float(np.mean(cpp_arr)), 3),
        "flatness_1_4k_med": round(float(np.median(flat_1_4)), 4),
        "mid_presence_db": round(mid_presence_db, 2),
        "hf_air_db": round(hf_air_db, 2),
        "crest_db": round(crest_db, 2),
        "active_frames": int(np.sum(active)),
    }


def extract_chromagram(mid_32k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract L2-normalized 12-bin chromagram [T, 12] and active mask [T] on 40 Hz grid."""
    spec = _stft_complex(mid_32k.astype(np.float64), N_FFT_CHROMA, HOP)
    mag = np.abs(spec) ** 1.5
    freqs = np.fft.rfftfreq(N_FFT_CHROMA, d=1.0 / SR)
    valid = (freqs >= 65.41) & (freqs <= 1046.50)  # C2..C6
    f_v = freqs[valid]
    midi = np.round(12.0 * np.log2(f_v / 440.0) + 69.0).astype(int)
    pc = midi % 12
    chroma = np.zeros((len(mag), 12), dtype=np.float64)
    mag_v = mag[:, valid]
    for b in range(12):
        sel = pc == b
        if np.any(sel):
            chroma[:, b] = np.sum(mag_v[:, sel], axis=-1)
    norms = np.linalg.norm(chroma, axis=-1, keepdims=True)
    chroma = np.divide(chroma, np.maximum(norms, 1e-12))
    active = _active_mask_from_mid(mid_32k, HOP, N_FFT_CHROMA)[: len(chroma)]
    return chroma, active


def extract_onset_env(mid_32k: np.ndarray) -> np.ndarray:
    """Extract z-scored half-wave rectified log-spectral flux onset curve on 40 Hz grid."""
    spec = _stft_complex(mid_32k.astype(np.float64), N_FFT_SPEC, HOP)
    freqs = np.fft.rfftfreq(N_FFT_SPEC, d=1.0 / SR)
    band = (freqs >= 60.0) & (freqs <= 8000.0)
    log_m = np.log1p(100.0 * np.abs(spec[:, band]))
    diff = np.maximum(0.0, np.diff(log_m, axis=0, prepend=log_m[:1]))
    flux = np.mean(diff, axis=-1)
    mu = float(np.mean(flux))
    sd = float(np.std(flux))
    return (flux - mu) / max(sd, 1e-9)


def _mel_filterbank(n_fft: int, n_mels: int = 40, f_min: float = 80.0, f_max: float = 14000.0) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mels = np.linspace(float(hz_to_mel(f_min)), float(hz_to_mel(f_max)), n_mels + 2)
    hz_pts = mel_to_hz(mels)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SR)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float64)
    for i in range(n_mels):
        left, center, right = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up = (freqs - left) / max(center - left, 1e-9)
        down = (right - freqs) / max(right - center, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


def extract_mfcc_profile(mid_32k: np.ndarray, n_mfcc: int = 12) -> np.ndarray:
    """Compute mean c1..c12 MFCC vector over active frames for global timbre comparison."""
    spec = _stft_complex(mid_32k.astype(np.float64), N_FFT_SPEC, HOP)
    power = np.abs(spec) ** 2
    active = _active_mask_from_mid(mid_32k, HOP, N_FFT_SPEC)[: len(power)]
    if not np.any(active):
        active = np.ones(len(power), dtype=bool)
    fb = _mel_filterbank(N_FFT_SPEC)
    mel_e = np.log(np.maximum(power[active] @ fb.T, 1e-10))
    n_mels = fb.shape[0]
    n_idx = np.arange(n_mels, dtype=np.float64)[None, :]
    k_idx = np.arange(1, n_mfcc + 1, dtype=np.float64)[:, None]
    dct = np.cos(math.pi * k_idx * (n_idx + 0.5) / n_mels) * math.sqrt(2.0 / n_mels)
    mfcc = mel_e @ dct.T
    prof = np.mean(mfcc, axis=0)
    norm = float(np.linalg.norm(prof))
    return prof / max(norm, 1e-12)


def _valid_circular_lags(n_frames: int) -> list[int]:
    excl = int(round(LAG_EXCL_S * (SR / HOP)))
    lags = [k for k in range(excl, n_frames - excl, LAG_STEP_FRAMES)]
    if len(lags) < 10:
        lags = [k for k in range(1, n_frames) if min(k, n_frames - k) >= max(4, excl // 4)]
    return lags


def compare_melody_and_rhythm(out_mid: np.ndarray, tgt_mid: np.ndarray) -> dict[str, Any]:
    """Score chromagram & spectral-flux onset correlation vs circular lag null."""
    c_out, a_out = extract_chromagram(out_mid)
    c_tgt, a_tgt = extract_chromagram(tgt_mid)
    n = min(len(c_out), len(c_tgt))
    c_out, a_out = c_out[:n], a_out[:n]
    c_tgt, a_tgt = c_tgt[:n], a_tgt[:n]

    lags = _valid_circular_lags(n)

    def _chroma_cos(shift: int) -> float:
        ct = np.roll(c_tgt, shift, axis=0)
        at = np.roll(a_tgt, shift, axis=0)
        use = a_out & at
        if int(np.sum(use)) < 12:
            return 0.0
        return float(np.mean(np.sum(c_out[use] * ct[use], axis=-1)))

    c0 = _chroma_cos(0)
    c_nulls = np.asarray([_chroma_cos(k) for k in lags], dtype=np.float64)
    c_null_med = float(np.median(c_nulls)) if len(c_nulls) else 0.0
    c_null_best = float(np.max(c_nulls)) if len(c_nulls) else 0.0
    c_beats = float(np.mean(c_nulls < c0)) if len(c_nulls) else 0.0

    o_out = extract_onset_env(out_mid)[:n]
    o_tgt = extract_onset_env(tgt_mid)[:n]

    def _onset_r_near(shift: int) -> float:
        best = -1.0
        for jit in (-2, -1, 0, 1, 2):
            ot = np.roll(o_tgt, shift + jit)
            num = float(np.dot(o_out, ot))
            den = float(math.sqrt((np.dot(o_out, o_out) + 1e-12) * (np.dot(ot, ot) + 1e-12)))
            best = max(best, num / max(den, 1e-12))
        return best

    o0 = _onset_r_near(0)
    o_nulls = np.asarray([_onset_r_near(k) for k in lags], dtype=np.float64)
    o_null_med = float(np.median(o_nulls)) if len(o_nulls) else 0.0
    o_beats = float(np.mean(o_nulls < o0)) if len(o_nulls) else 0.0

    return {
        "n_frames": int(n),
        "n_lags": int(len(lags)),
        "chroma_cos_0": round(c0, 4),
        "chroma_null_med": round(c_null_med, 4),
        "chroma_null_best": round(c_null_best, 4),
        "chroma_lift": round(c0 - c_null_med, 4),
        "chroma_beats_frac": round(c_beats, 3),
        "onset_r_0": round(o0, 4),
        "onset_null_med": round(o_null_med, 4),
        "onset_beats_frac": round(o_beats, 3),
    }


def audit_audio(
    output_path: Path,
    *,
    target_ref: Path | None = None,
    target_b_ref: Path | None = None,
) -> dict[str, Any]:
    """Run full 4-axis audit on an audio/video file."""
    out_32k = load_audio_32k(output_path)
    out_mid = 0.5 * (out_32k[0].astype(np.float64) + out_32k[1].astype(np.float64))
    res: dict[str, Any] = {
        "file": str(output_path),
        "duration_s": round(out_32k.shape[1] / SR, 3),
        "naturalness": measure_naturalness(out_32k),
    }
    if target_ref is not None:
        tgt_32k = load_audio_32k(target_ref)
        tgt_mid = 0.5 * (tgt_32k[0].astype(np.float64) + tgt_32k[1].astype(np.float64))
        res["melody_rhythm"] = compare_melody_and_rhythm(out_mid, tgt_mid)
    if target_ref is not None and target_b_ref is not None:
        ref_b_32k = load_audio_32k(target_b_ref)
        ref_b_mid = 0.5 * (ref_b_32k[0].astype(np.float64) + ref_b_32k[1].astype(np.float64))
        mfcc_out = extract_mfcc_profile(out_mid)
        mfcc_a = extract_mfcc_profile(tgt_mid)
        mfcc_b = extract_mfcc_profile(ref_b_mid)
        sim_b = float(np.dot(mfcc_out, mfcc_b))
        sim_a = float(np.dot(mfcc_out, mfcc_a))
        res["timbre"] = {
            "timbre_sim_b": round(sim_b, 4),
            "timbre_sim_a": round(sim_a, 4),
            "timbre_margin": round(sim_b - sim_a, 4),
        }
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Independent first-principles cover audio audit.")
    ap.add_argument("outputs", nargs="+", help="One or more rendered .mp4/.flac/.wav files to audit.")
    ap.add_argument("--target-ref", default=None, help="Key-shifted Target A audio/mix for melody/onset/source-timbre lock.")
    ap.add_argument("--target-b-ref", default=None, help="Target B singer/style reference audio for timbre margin.")
    ap.add_argument("--json-out", default=None, help="Optional JSON output path.")
    args = ap.parse_args(argv)

    t_ref = Path(args.target_ref) if args.target_ref else None
    b_ref = Path(args.target_b_ref) if args.target_b_ref else None
    rows = [audit_audio(Path(p), target_ref=t_ref, target_b_ref=b_ref) for p in args.outputs]
    payload = rows[0] if len(rows) == 1 else rows
    text = json.dumps(payload, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
