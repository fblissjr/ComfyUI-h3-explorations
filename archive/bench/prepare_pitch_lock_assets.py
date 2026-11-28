#!/usr/bin/env python3
"""Prepare solo-Kurt visual anchors and MelBand-RoFormer + RubberBand stems for the pitch-lock cover gate.

Outputs:
  - `$H3_INPUT_DIR/music_cover_pitch_lock/*.png` & `*.wav` (exact 192f = 256,000 samples @ 32 kHz stereo)
  - `data/music_cover_pitch_lock/stem_prep_report.json`
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as TAF
from scipy.io import wavfile

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
sys.path.insert(0, str(REPO / "bench"))
from _lib import bootstrap  # noqa: E402

bootstrap(cpu=False)
import probe_yin_f0_drift as yin_probe  # noqa: E402

TARGET_SR = 32000
TARGET_SAMPLES = 256000  # 192f / 24 fps = 8.000 s -> 320 * 800 samples at 32 kHz
TARGET_FRAMES = 192
FPS = 24.0


def _require_dir(env_name: str) -> Path:
    val = os.environ.get(env_name)
    if not val:
        raise SystemExit(f"{env_name} is unset; run `set -a; . ./h3.env; set +a` first")
    p = Path(val)
    if not p.is_dir():
        raise SystemExit(f"{env_name} does not exist: {p}")
    return p


def _load_mel_roformer(device: torch.device):
    mbr_dir = COMFY / "custom_nodes" / "ComfyUI-MelBandRoFormer"
    ckpt_path = COMFY / "models" / "diffusion_models" / "MelBandRoformer_fp16.safetensors"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing MelBandRoFormer checkpoint: {ckpt_path}")
    import importlib
    from safetensors.torch import load_file

    if str(mbr_dir) not in sys.path:
        sys.path.insert(0, str(mbr_dir))
    MelBandRoformer = importlib.import_module("model.mel_band_roformer").MelBandRoformer

    model_config = {
        "dim": 384,
        "depth": 6,
        "stereo": True,
        "num_stems": 1,
        "time_transformer_depth": 1,
        "freq_transformer_depth": 1,
        "num_bands": 60,
        "dim_head": 64,
        "heads": 8,
        "attn_dropout": 0,
        "ff_dropout": 0,
        "flash_attn": True,
        "dim_freqs_in": 1025,
        "sample_rate": 44100,
        "stft_n_fft": 2048,
        "stft_hop_length": 441,
        "stft_win_length": 2048,
        "stft_normalized": False,
        "mask_estimator_depth": 2,
        "multi_stft_resolution_loss_weight": 1.0,
        "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
        "multi_stft_hop_size": 147,
        "multi_stft_normalized": False,
    }
    model = MelBandRoformer(**model_config).eval()
    model.load_state_dict(load_file(str(ckpt_path)), strict=True)
    return model.to(device)


def _windowing_array(window_size: int, fade_size: int, device: torch.device) -> torch.Tensor:
    fadein = torch.linspace(0, 1, fade_size, device=device)
    fadeout = torch.linspace(1, 0, fade_size, device=device)
    w = torch.ones(window_size, device=device)
    w[-fade_size:] *= fadeout
    w[:fade_size] *= fadein
    return w


@torch.no_grad()
def separate_roformer(model, wav_44k: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run MelBandRoformer on [2, L] 44.1 kHz tensor; return (vocals_44k, instruments_44k) on CPU."""
    audio_input = wav_44k.to(torch.float32)
    orig = audio_input.clone()
    audio_length = int(audio_input.shape[1])
    C = 352800
    step = C // 2
    fade_size = C // 10
    border = C - step
    if audio_length > 2 * border and border > 0:
        audio_input = F.pad(audio_input, (border, border), mode="reflect")
    win_arr = _windowing_array(C, fade_size, device)
    audio_dev = audio_input.to(device)
    vocals = torch.zeros_like(audio_dev, dtype=torch.float32)
    counter = torch.zeros_like(audio_dev, dtype=torch.float32)
    total_len = int(audio_dev.shape[1])
    for i in range(0, total_len, step):
        part = audio_dev[:, i : i + C]
        length = int(part.shape[-1])
        if length < C:
            if length > C // 2 + 1:
                part = F.pad(part, (0, C - length), mode="reflect")
            else:
                part = F.pad(part, (0, C - length), mode="constant", value=0)
        x = model(part.unsqueeze(0))[0]
        w = win_arr.clone()
        if i == 0:
            w[:fade_size] = 1.0
        elif i + C >= total_len:
            w[-fade_size:] = 1.0
        vocals[:, i : i + length] += x[:, :length] * w[:length]
        counter[:, i : i + length] += w[:length]
    est = vocals / counter.clamp(min=1e-8)
    if audio_length > 2 * border and border > 0:
        est = est[:, border:-border]
    est_cpu = est.cpu()
    inst_cpu = orig - est_cpu
    return est_cpu, inst_cpu


def _load_ffmpeg_stereo(path: Path, start_s: float, dur_s: float, sr: int) -> torch.Tensor:
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{start_s:.4f}",
            "-t",
            f"{dur_s:.4f}",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(sr),
            "-f",
            "f32le",
            "-",
        ]
    )
    arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(-1, 2).T
    return torch.from_numpy(arr)


def _save_pcm16_wav(path: Path, wav_stereo: torch.Tensor, sr: int) -> None:
    x = wav_stereo.detach().cpu().clamp(-1.0, 1.0).transpose(0, 1).contiguous()
    pcm = (x * 32767.0).round().to(torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(int(x.shape[1]))
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm)


def _load_wav_tensor(path: Path) -> tuple[torch.Tensor, int]:
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        arr = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        arr = data.astype(np.float32) / 2147483648.0
    else:
        arr = data.astype(np.float32)
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=-1)
    return torch.from_numpy(arr.T.copy()), int(sr)


def _write_wav_32k_exact(path: Path, wav_stereo: torch.Tensor, sr_in: int) -> Path:
    if sr_in != TARGET_SR:
        w = TAF.resample(wav_stereo.to(torch.float32), orig_freq=sr_in, new_freq=TARGET_SR)
    else:
        w = wav_stereo.to(torch.float32).clone()
    if w.shape[1] < TARGET_SAMPLES:
        w = F.pad(w, (0, TARGET_SAMPLES - w.shape[1]))
    elif w.shape[1] > TARGET_SAMPLES:
        w = w[:, :TARGET_SAMPLES]
    peak = float(w.abs().max().item())
    if peak > 0.95:
        w = w * (0.95 / peak)
    _save_pcm16_wav(path, w, TARGET_SR)
    return path


def _rubberband_shift(wav_44k: torch.Tensor, semitones: float, formant: bool = False) -> torch.Tensor:
    if abs(semitones) < 1e-6:
        return wav_44k.clone()
    pitch_ratio = 2.0 ** (semitones / 12.0)
    flt = f"rubberband=pitch={pitch_ratio:.9f}:pitchq=quality"
    if formant:
        flt += ":formant=preserved"
    with tempfile.TemporaryDirectory() as td:
        pin = Path(td) / "in.wav"
        pout = Path(td) / "out.wav"
        _save_pcm16_wav(pin, wav_44k, 44100)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(pin), "-af", flt, "-ar", "44100", "-ac", "2", str(pout)],
            check=True,
        )
        w, sr = _load_wav_tensor(pout)
        assert sr == 44100
        return w.to(torch.float32)


def _hpss_harmonic_band(inst_44k: torch.Tensor, lo_hz: float = 80.0, hi_hz: float = 1200.0) -> torch.Tensor:
    """Keep warm harmonic chord roots from instruments stem with zero drums/transients."""
    n_fft = 2048
    hop = 512
    win = torch.hann_window(n_fft)
    out_ch = []
    for c in range(inst_44k.shape[0]):
        X = torch.stft(inst_44k[c], n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        mag = X.abs()
        # Median filtering via 2D unfold approximation (time=17 frames, freq=17 bins)
        t_pad = F.pad(mag.unsqueeze(0).unsqueeze(0), (8, 8, 0, 0), mode="replicate")
        H_med = t_pad.unfold(-1, 17, 1).median(dim=-1).values.squeeze(0).squeeze(0)
        f_pad = F.pad(mag.unsqueeze(0).unsqueeze(0), (0, 0, 8, 8), mode="replicate")
        P_med = f_pad.unfold(-2, 17, 1).median(dim=-1).values.squeeze(0).squeeze(0)
        mask_h = (H_med ** 2) / (H_med ** 2 + P_med ** 2 + 1e-12)
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / 44100.0)
        band = ((freqs >= lo_hz) & (freqs <= hi_hz)).to(torch.float32).unsqueeze(-1)
        # Smooth cosine rolloff at upper edge
        roll = ((freqs > hi_hz * 0.7) & (freqs <= hi_hz))
        if roll.any():
            band[roll, 0] = 0.5 * (1.0 + torch.cos(math.pi * (freqs[roll] - hi_hz * 0.7) / (hi_hz * 0.3)))
        X_h = X * mask_h * band
        y = torch.istft(X_h, n_fft=n_fft, hop_length=hop, window=win, length=inst_44k.shape[1])
        out_ch.append(y)
    return torch.stack(out_ch, dim=0)


def _mix_voc_and_chord(voc_44k: torch.Tensor, chord_44k: torch.Tensor, chord_db: float = -11.0) -> torch.Tensor:
    n = min(voc_44k.shape[1], chord_44k.shape[1])
    v = voc_44k[:, :n]
    c = chord_44k[:, :n]
    v_rms = float(v.pow(2).mean().sqrt().clamp(min=1e-7))
    c_rms = float(c.pow(2).mean().sqrt().clamp(min=1e-7))
    target_c_rms = v_rms * (10.0 ** (chord_db / 20.0))
    mix = v + c * (target_c_rms / c_rms)
    peak = float(mix.abs().max())
    if peak > 0.92:
        mix = mix * (0.92 / peak)
    return mix


def _chroma(y: np.ndarray, sr: int = 24000) -> np.ndarray:
    spec = torch.stft(torch.from_numpy(y), n_fft=4096, hop_length=600, window=torch.hann_window(4096), return_complex=True).abs().numpy() ** 2
    freqs = np.fft.rfftfreq(4096, 1.0 / sr)
    valid = (freqs >= 65.0) & (freqs <= 2000.0)
    pc = np.round(69.0 + 12.0 * np.log2(freqs[valid] / 440.0)).astype(int) % 12
    C = np.zeros((12, spec.shape[1]), dtype=np.float64)
    for k in range(12):
        C[k] = spec[valid][pc == k].sum(axis=0)
    C = np.log1p(C * 1e3)
    C = C - C.mean(axis=0, keepdims=True)
    return C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-8)


def _chroma_same_key(a: np.ndarray, b: np.ndarray, tr_st: int = 0) -> float:
    n = min(a.shape[1], b.shape[1])
    bt = np.roll(b, tr_st, axis=0) if tr_st else b
    best = -1.0
    for lag in range(-6, 7):
        aa = a[:, lag:n] if lag >= 0 else a[:, :n + lag]
        bb = bt[:, :n - lag] if lag >= 0 else bt[:, -lag:n]
        if aa.shape[1] < 20:
            continue
        s = float(np.mean(np.sum(aa * bb, axis=0)))
        if s > best:
            best = s
    return best


def _voice_activity_summary(wav_32k: Path) -> list[list[float]]:
    mono = yin_probe.load_mono_audio(wav_32k, sr=24000).numpy()
    hop = 1000  # 24 fps
    spans = []
    active = False
    st = 0
    for f in range(TARGET_FRAMES):
        seg = mono[f * hop : (f + 1) * hop]
        db = 20.0 * math.log10(max(float(np.sqrt(np.mean(seg ** 2))), 1e-7))
        on = db > -38.0
        if on and not active:
            active = True
            st = f
        elif not on and active:
            active = False
            if f - st >= 4:
                spans.append([round(st / FPS, 2), round((f - 1) / FPS, 2)])
    if active and TARGET_FRAMES - st >= 4:
        spans.append([round(st / FPS, 2), round((TARGET_FRAMES - 1) / FPS, 2)])
    return spans


def main() -> int:
    inp = _require_dir("H3_INPUT_DIR")
    out_dir = _require_dir("H3_OUTPUT_DIR")
    dest_inp = inp / "music_cover_pitch_lock"
    dest_out = out_dir / "Video" / "music_cover_pitch_lock_2026-11-27"
    dest_inp.mkdir(parents=True, exist_ok=True)
    dest_out.mkdir(parents=True, exist_ok=True)
    data_dir = REPO / "data" / "music_cover_pitch_lock"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Solo Kurt stills (clean 4:3 edge-to-edge without pillarbox bars or foreground competing vocalist)
    src_vid = inp / "unplugged_compressed.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", "72.0", "-i", str(src_vid), "-frames:v", "1", "-vf", "crop=1380:1035:270:22,scale=1440:1080:flags=lanczos", str(dest_inp / "kurt_closeup.png")],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", "195.0", "-i", str(src_vid), "-frames:v", "1", "-vf", "crop=1380:1035:270:22,scale=1440:1080:flags=lanczos", str(dest_inp / "kurt_solo_medium.png")],
        check=True,
    )

    # 2. Load raw slices (with 1s extra buffer if needed, here exact 8.0s slices padded safely)
    slices = {
        "kurt": (src_vid, 24.8, 8.0),
        "flicker": (inp / "justaflicker.mp3", 32.0, 8.0),
        "whatever": (inp / "whatevercomes.wav", 0.0, 8.0),
        "fridge": (inp / "In The Fridge.mp3", 60.6, 8.0),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_mel_roformer(device)
    stems: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    try:
        for name, (src_p, ss, dur) in slices.items():
            raw44 = _load_ffmpeg_stereo(src_p, ss, dur, 44100)
            voc44, inst44 = separate_roformer(model, raw44, device)
            stems[name] = (raw44, voc44, inst44)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. Export exact 32 kHz 192f stems + scaffolds
    raw_kurt, voc_kurt, _ = stems["kurt"]
    _write_wav_32k_exact(dest_inp / "kurt_vocal_only_192f.wav", voc_kurt, 44100)
    _write_wav_32k_exact(dest_inp / "kurt_vocal_guitar_192f.wav", raw_kurt, 44100)

    raw_flk, voc_flk, inst_flk = stems["flicker"]
    _write_wav_32k_exact(dest_inp / "flicker_raw_192f.wav", raw_flk, 44100)
    _write_wav_32k_exact(dest_inp / "flicker_voc_orig_192f.wav", voc_flk, 44100)
    voc_flk_oct = _rubberband_shift(voc_flk, -12.0, formant=False)
    _write_wav_32k_exact(dest_inp / "flicker_voc_oct_192f.wav", voc_flk_oct, 44100)
    chord_flk = _hpss_harmonic_band(inst_flk, 80.0, 1200.0)
    _write_wav_32k_exact(dest_inp / "flicker_chord_scaf_192f.wav", _mix_voc_and_chord(voc_flk_oct, chord_flk, -11.0), 44100)

    raw_wh, voc_wh, inst_wh = stems["whatever"]
    _write_wav_32k_exact(dest_inp / "whatever_raw_192f.wav", raw_wh, 44100)
    _write_wav_32k_exact(dest_inp / "whatever_voc_orig_192f.wav", voc_wh, 44100)
    voc_wh_oct = _rubberband_shift(voc_wh, -12.0, formant=False)
    chord_wh = _hpss_harmonic_band(inst_wh, 80.0, 1200.0)
    _write_wav_32k_exact(dest_inp / "whatever_origkey_scaf_192f.wav", _mix_voc_and_chord(voc_wh_oct, chord_wh, -11.0), 44100)

    voc_wh_capo = _rubberband_shift(voc_wh, -5.0, formant=True)
    _write_wav_32k_exact(dest_inp / "whatever_voc_capo_csharp_192f.wav", voc_wh_capo, 44100)
    chord_wh_capo = _rubberband_shift(chord_wh, -5.0, formant=False)
    _write_wav_32k_exact(dest_inp / "whatever_capo_csharp_scaf_192f.wav", _mix_voc_and_chord(voc_wh_capo, chord_wh_capo, -11.0), 44100)

    raw_fr, voc_fr, inst_fr = stems["fridge"]
    _write_wav_32k_exact(dest_inp / "fridge_raw_192f.wav", raw_fr, 44100)
    _write_wav_32k_exact(dest_inp / "fridge_voc_orig_192f.wav", voc_fr, 44100)
    chord_fr = _hpss_harmonic_band(inst_fr, 80.0, 1100.0)
    _write_wav_32k_exact(dest_inp / "fridge_scaf_192f.wav", _mix_voc_and_chord(voc_fr, chord_fr, -12.0), 44100)

    # 4. Verify sample counts, YIN F0, chroma_sameKey, and voice spans
    report: dict[str, Any] = {"items": {}}
    checks = [
        ("kurt_vocal_only_192f.wav", "kurt_vocal_guitar_192f.wav", "kurt_vocal_only_192f.wav", 0),
        ("kurt_vocal_guitar_192f.wav", "kurt_vocal_guitar_192f.wav", "kurt_vocal_only_192f.wav", 0),
        ("flicker_voc_oct_192f.wav", "flicker_raw_192f.wav", "flicker_voc_orig_192f.wav", 0),
        ("flicker_chord_scaf_192f.wav", "flicker_raw_192f.wav", "flicker_voc_orig_192f.wav", 0),
        ("whatever_origkey_scaf_192f.wav", "whatever_raw_192f.wav", "whatever_voc_orig_192f.wav", 0),
        ("whatever_capo_csharp_scaf_192f.wav", "whatever_raw_192f.wav", "whatever_voc_orig_192f.wav", -5),
        ("fridge_scaf_192f.wav", "fridge_raw_192f.wav", "fridge_voc_orig_192f.wav", 0),
    ]
    for fn, src_fn, voc_fn, tr_st in checks:
        p = dest_inp / fn
        w_chk, sr_chk = _load_wav_tensor(p)
        assert sr_chk == TARGET_SR and w_chk.shape == (2, TARGET_SAMPLES), f"{fn}: {w_chk.shape} @ {sr_chk}"
        y = yin_probe.load_mono_audio(p, sr=24000)
        y_src = yin_probe.load_mono_audio(dest_inp / src_fn, sr=24000)
        y_voc = yin_probe.load_mono_audio(dest_inp / voc_fn, sr=24000)
        f0, _, _ = yin_probe.yin_f0(y)
        f0_np = f0.numpy()
        valid = np.isfinite(f0_np) & (f0_np > 0)
        med = float(np.median(f0_np[valid])) if np.any(valid) else float("nan")
        csim_mix = _chroma_same_key(_chroma(y.numpy()), _chroma(y_src.numpy()), tr_st=tr_st)
        csim_voc = _chroma_same_key(_chroma(y.numpy()), _chroma(y_voc.numpy()), tr_st=tr_st)
        spans = _voice_activity_summary(p)
        report["items"][fn] = {
            "samples": int(w_chk.shape[1]),
            "sr": int(sr_chk),
            "median_f0_hz": round(med, 2) if math.isfinite(med) else None,
            "chroma_sameKey_vs_mix": round(csim_mix, 3),
            "chroma_sameKey_vs_voc": round(csim_voc, 3),
            "tr_st": tr_st,
            "voiced_spans_s": spans,
        }
        if "scaf" in fn or "oct" in fn:
            assert max(csim_mix, csim_voc) >= 0.55, f"Scaffold {fn} failed chroma_sameKey >= 0.55: mix={csim_mix:.3f}, voc={csim_voc:.3f}"

    rp = data_dir / "stem_prep_report.json"
    rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
