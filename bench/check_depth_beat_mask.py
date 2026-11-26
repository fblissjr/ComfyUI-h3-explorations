#!/usr/bin/env python3
"""Offline CPU check for `MiniMaxH3DepthBeatMask` (`depth_beat_mask.py`).

Tests all 5 modes (`depth_band`, `depth_wave`, `audio_beat_cut`, `audio_beat_pulse`,
`depth_band_on_beat`), `HyDenMoGe.tiny_for_tests()` end-to-end depth estimation without
heavy weights, pure-NumPy WAV onset beat snapping, and node schema consistency.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO_DIR = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _lib import bootstrap

bootstrap(cpu=True)

import torch

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from depth_beat_mask import (  # noqa: E402
    MiniMaxH3DepthBeatMask,
    analyze_audio_waveform_beats,
    compute_depth_beat_mask,
    compute_moge2_depth_frame,
)
from vendor.meta_moge2.hyden_moge2 import HyDenMoGe  # noqa: E402


def _synthetic_audio(duration_s: float = 5.0, sr: int = 24000, hit_frames: tuple[int, ...] = (51, 85)) -> dict:
    n = int(round(duration_s * sr))
    wav = torch.zeros((1, 2, n), dtype=torch.float32)
    t = torch.arange(n, dtype=torch.float32) / float(sr)
    # Soft noise floor + sharp transient kicks at hit_frames
    wav += 0.002 * torch.sin(2.0 * math.pi * 110.0 * t)
    for idx, cf in enumerate(hit_frames):
        amp = 0.85 if idx == 0 else 0.45
        s0 = int(round((cf / 24.0) * sr))
        span = int(0.08 * sr)
        env = torch.exp(-torch.arange(span, dtype=torch.float32) / (0.015 * sr))
        wav[:, :, s0 : s0 + span] += amp * env
    return {"waveform": wav, "sample_rate": sr}


def main() -> int:
    # 1. Synthetic depth gradient (top/far = 20 m, bottom/near = 2 m)
    h, w, length = 64, 96, 141
    y_grad = torch.linspace(20.0, 2.0, h, dtype=torch.float32).unsqueeze(1).expand(h, w)

    # 2. depth_band selects top/far wall (quantiles 0.65..1.0) and spares foreground bottom
    m_band, s_band = compute_depth_beat_mask(
        mode="depth_band",
        length=length,
        width=w,
        height=h,
        near_quantile=0.65,
        far_quantile=1.0,
        feather=0.08,
        depth_hw=y_grad,
    )
    assert m_band.shape == (length, h, w), m_band.shape
    assert float(m_band[:, 4:14, :].mean()) > 0.90, "far back wall must be selected"
    assert float(m_band[:, -14:, :].mean()) < 0.02, "near foreground must stay unmasked"
    band_info = json.loads(s_band)
    assert band_info["depth_stats"]["far_m"] == 20.0
    print("[0/5 OK] depth_band isolates far depth quantile and preserves foreground")

    # 3. depth_wave advances near -> far across frames
    m_wave, _ = compute_depth_beat_mask(
        mode="depth_wave",
        length=length,
        width=w,
        height=h,
        near_quantile=0.1,
        far_quantile=0.95,
        feather=0.06,
        depth_hw=y_grad,
    )
    assert m_wave.shape == (length, h, w)
    assert float(m_wave[5, -8:, :].mean()) > float(m_wave[5, :8, :].mean())
    assert float(m_wave[-1, :8, :].mean()) > 0.85
    print("[1/5 OK] depth_wave sweeps cleanly from near to far depth")

    # 4. audio_beat_cut + audio_beat_pulse detect causal 17*m downbeat cleanly
    aud = _synthetic_audio()
    beat_res = analyze_audio_waveform_beats(aud["waveform"], aud["sample_rate"], length)
    assert beat_res["recommended_cut_frame"] == 51, beat_res
    m_cut, _ = compute_depth_beat_mask(
        mode="audio_beat_cut",
        beat_timing="auto_from_audio",
        length=length,
        width=w,
        height=h,
        audio=aud,
    )
    assert float(m_cut[:51].max()) == 0.0 and float(m_cut[51:].min()) == 1.0

    m_pulse, _ = compute_depth_beat_mask(
        mode="audio_beat_pulse",
        beat_timing="auto_from_audio",
        length=length,
        width=w,
        height=h,
        pulse_decay_frames=10,
        audio=aud,
    )
    assert float(m_pulse[:40].max()) == 0.0
    assert float(m_pulse[51].mean()) > 0.70 and float(m_pulse[56].mean()) < float(m_pulse[51].mean())
    assert float(m_pulse[70].max()) == 0.0
    print("[2/5 OK] audio_beat_cut and audio_beat_pulse lock onto 17*m beats with clean zero floor")

    # 5. depth_band_on_beat holds entire room 0.0 before f12 then opens only back wall
    m_tear, s_tear = compute_depth_beat_mask(
        mode="depth_band_on_beat",
        beat_timing="manual_frame",
        cut_frame=12,
        length=length,
        width=w,
        height=h,
        near_quantile=0.68,
        far_quantile=1.0,
        feather=0.06,
        depth_hw=y_grad,
    )
    assert float(m_tear[:12].max()) == 0.0
    assert float(m_tear[30:, 4:14, :].mean()) > 0.90
    assert float(m_tear[30:, -14:, :].mean()) < 0.02
    assert json.loads(s_tear)["resolved_cut_frame"] == 12
    print("[3/5 OK] depth_band_on_beat locks pre-tear room at 0.0 and opens far depth zone post-cut")

    # 6. HyDenMoGe.tiny_for_tests() + node schema parity
    tiny = HyDenMoGe.tiny_for_tests().cpu()
    fake_img = torch.rand((1, 64, 96, 3), dtype=torch.float32)
    d_tiny, d_stats = compute_moge2_depth_frame(fake_img, width=96, height=64, device="cpu", _model=tiny)
    assert d_tiny.shape == (64, 96) and torch.isfinite(d_tiny).all() and d_stats["median_m"] > 0.0

    schema = MiniMaxH3DepthBeatMask.define_schema()
    assert schema.node_id == "MiniMaxH3DepthBeatMask"
    out = MiniMaxH3DepthBeatMask.execute(
        mode="audio_beat_cut",
        beat_timing="manual_frame",
        length=39,
        width=64,
        height=64,
        cut_frame=12,
        invert=True,
    )
    assert out[0].shape == (39, 64, 64)
    assert float(out[0][:12].min()) == 1.0 and float(out[0][12:].max()) == 0.0
    print("[4/5 OK] HyDenMoGe tiny inference + MiniMaxH3DepthBeatMask schema & inversion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
