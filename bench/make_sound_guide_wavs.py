#!/usr/bin/env python3
"""Synthesize the two 32 kHz PCM `.wav` timeline sound-guide islands for `h3_text_to_video_pika_sound_guide_api.json`.

Writes into `--out-dir` (default `$H3_INPUT_DIR`):
  - `h3_guide_hop_200ms.wav`      (`6400` samples @ 32 kHz = exact `8` tokens on the 40 Hz Audio VAE grid):
                                  `190 -> 88 Hz` muffled shell/asphalt landing thud over a continuous `-28 dBFS`
                                  urban street-rumble floor (no digital-zero tail).
  - `h3_guide_yank_slam_300ms.wav` (`9600` samples @ 32 kHz = exact `12` tokens on the 40 Hz Audio VAE grid):
                                  compound single-island event with a `160–720 Hz` taut-leather snap at `t = 0 ms`
                                  (`cond_t + 0`) followed at `t = +125 ms` (`+5` audio tokens = `+3` frames at 24 fps)
                                  by a sharp `220 -> 88 Hz` slammed-shell impact over the same `-28 dBFS` floor,
                                  avoiding duplicate RoPE timestamps from overlapping `.wav` guides.

    <comfy-venv-python> bench/make_sound_guide_wavs.py
"""
from __future__ import annotations

import argparse
import os
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "workflows"))
import h3_config as cfg  # noqa: E402

SR = 32000


def _street_ambience_floor(n: int, seed: int) -> np.ndarray:
    """Continuous ~-28 dBFS urban street rumble (54/108 Hz diesel idle + 90-420 Hz road air)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / SR
    hum = 0.018 * np.sin(2.0 * np.pi * 54.0 * t + 0.3) + 0.011 * np.sin(2.0 * np.pi * 108.0 * t + 1.1)
    raw = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    spec = np.fft.rfft(raw)
    mask = ((f >= 75.0) & (f <= 420.0)).astype(np.float64) * np.exp(-np.maximum(f - 140.0, 0.0) / 180.0)
    air = np.fft.irfft(spec * mask, n=n)
    air = 0.032 * air / (np.sqrt(np.mean(air**2)) + 1e-9)
    amb = hum + air
    fade = min(160, n // 8)
    amb[:fade] *= np.linspace(0.35, 1.0, fade)
    amb[-fade:] *= np.linspace(1.0, 0.35, fade)
    return amb


def synthesize_sound_guide_wavs(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026111202)

    # 1. Free tortoise hop impact: 200 ms = 6400 samples @ 32 kHz = 8 audio latent tokens
    n_free = 6400
    t_free = np.arange(n_free, dtype=np.float64) / SR
    env_free = np.exp(-t_free / 0.036) * (1.0 - np.exp(-t_free / 0.0025))
    phase_free = 2.0 * np.pi * (88.0 * t_free + (190.0 - 88.0) * 0.018 * (1.0 - np.exp(-t_free / 0.018)))
    body_free = np.sin(phase_free) + 0.35 * np.sin(1.65 * phase_free)
    scrape_raw = rng.standard_normal(n_free)
    f_free = np.fft.rfftfreq(n_free, 1.0 / SR)
    scrape_free = np.fft.irfft(
        np.fft.rfft(scrape_raw) * ((f_free >= 70.0) & (f_free <= 380.0)),
        n=n_free,
    )
    scrape_free /= np.max(np.abs(scrape_free)) + 1e-9
    sig_free = 0.86 * env_free * (0.78 * body_free + 0.32 * scrape_free) + _street_ambience_floor(n_free, 101)
    sig_free = np.clip(sig_free, -0.96, 0.96)

    p_free = out_dir / cfg.SOUND_GUIDE_HOP_AUDIO
    with wave.open(str(p_free), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((sig_free * 32767.0).astype(np.int16).tobytes())

    # 2. Compound mid-air taut-leather snap (t=0 ms) + slammed shell impact (t=+125 ms = +3 frames): 300 ms = 9600 samples
    n_yank = 9600
    t_yank = np.arange(n_yank, dtype=np.float64) / SR
    env_snap = np.exp(-t_yank / 0.014) * (1.0 - np.exp(-t_yank / 0.0008))
    snap_raw = rng.standard_normal(n_yank)
    f_yank = np.fft.rfftfreq(n_yank, 1.0 / SR)
    snap_band = ((f_yank >= 160.0) & (f_yank <= 720.0)).astype(np.float64) * np.exp(
        -((f_yank - 380.0) ** 2) / (2.0 * 170.0**2)
    )
    snap_tex = np.fft.irfft(np.fft.rfft(snap_raw) * snap_band, n=n_yank)
    snap_tex /= np.max(np.abs(snap_tex)) + 1e-9
    phase_tug = 2.0 * np.pi * (210.0 * t_yank + (520.0 - 210.0) * 0.008 * (1.0 - np.exp(-t_yank / 0.008)))
    sig_snap = 0.82 * env_snap * (0.58 * snap_tex + 0.48 * np.sin(phase_tug))

    dt_slam = np.maximum(t_yank - 0.125, 0.0)
    gate_slam = (t_yank >= 0.125).astype(np.float64)
    env_slam = gate_slam * np.exp(-dt_slam / 0.032) * (1.0 - np.exp(-dt_slam / 0.0018))
    phase_slam = 2.0 * np.pi * (86.0 * dt_slam + (225.0 - 86.0) * 0.015 * (1.0 - np.exp(-dt_slam / 0.015)))
    slam_raw = rng.standard_normal(n_yank)
    slam_tex = np.fft.irfft(
        np.fft.rfft(slam_raw) * ((f_yank >= 65.0) & (f_yank <= 410.0)),
        n=n_yank,
    )
    slam_tex /= np.max(np.abs(slam_tex)) + 1e-9
    sig_slam = 0.92 * env_slam * (0.76 * (np.sin(phase_slam) + 0.38 * np.sin(1.7 * phase_slam)) + 0.34 * slam_tex)

    sig_yank = np.clip(sig_snap + sig_slam + _street_ambience_floor(n_yank, 202), -0.96, 0.96)
    p_yank = out_dir / cfg.SOUND_GUIDE_YANK_AUDIO
    with wave.open(str(p_yank), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((sig_yank * 32767.0).astype(np.int16).tobytes())

    return p_free, p_yank


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-dir",
        default=os.environ.get("H3_INPUT_DIR"),
        help="Destination folder for the .wav files (defaults to $H3_INPUT_DIR)",
    )
    args = ap.parse_args()
    if not args.out_dir:
        print("REFUSED: pass --out-dir or set $H3_INPUT_DIR", file=sys.stderr)
        return 1
    p_free, p_yank = synthesize_sound_guide_wavs(Path(args.out_dir))
    print(f"wrote {p_free.name} and {p_yank.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
