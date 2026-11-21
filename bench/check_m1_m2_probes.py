#!/usr/bin/env python3
"""Offline unit tests for M1 (`probe_dinov3_copy_freeze.py`) and M2 (`probe_yin_f0_drift.py`)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402
import probe_dinov3_copy_freeze as m1  # noqa: E402
import probe_yin_f0_drift as m2  # noqa: E402


def _harmonic_wave(freq_hz: float, duration_s: float = 1.0, sr: int = 24000, amp: float = 0.25) -> torch.Tensor:
    t = torch.arange(int(round(duration_s * sr)), dtype=torch.float32) / float(sr)
    sig = (
        0.65 * torch.sin(2.0 * math.pi * freq_hz * t)
        + 0.25 * torch.sin(2.0 * math.pi * 2.0 * freq_hz * t)
        + 0.10 * torch.sin(2.0 * math.pi * 3.0 * freq_hz * t)
    )
    return sig * amp


def check_m2_yin_sine_and_harmonic_shift() -> str:
    sr = 24000
    for target_hz in (110.0, 220.0, 330.0):
        t = torch.arange(sr, dtype=torch.float32) / float(sr)
        sine = 0.25 * torch.sin(2.0 * math.pi * target_hz * t)
        f0, _, _ = m2.yin_f0(sine, sr=sr)
        finite = f0[torch.isfinite(f0)]
        assert finite.numel() > 20, f"too few voiced frames for {target_hz} Hz"
        med = float(torch.median(finite).item())
        err_st = abs(12.0 * math.log2(med / target_hz))
        assert err_st < 0.15, (target_hz, med, err_st)

    # +3.0 semitone shift on harmonic wave (150 Hz -> 178.38 Hz)
    base = _harmonic_wave(150.0, duration_s=1.0, sr=sr)
    shift3 = _harmonic_wave(150.0 * (2.0 ** (3.0 / 12.0)), duration_s=1.0, sr=sr)
    b_f0, _, _ = m2.yin_f0(base, sr=sr)
    s_f0, _, _ = m2.yin_f0(shift3, sr=sr)
    diff = m2.compare_f0_drift(b_f0, s_f0)
    assert diff["median_shift_semitones"] is not None
    assert abs(diff["median_shift_semitones"] - 3.0) < 0.15, diff
    assert abs(diff["paired_frame_shift_median_st"] - 3.0) < 0.15, diff
    assert any("shifted" in n for n in diff["triage_notes"]), diff
    return f"110/220/330 Hz sines + 150Hz +3.0st harmonic shift accurately recovered ({diff['median_shift_semitones']:+.3f} st)"


def check_m2_silence_gate_and_voice_spans() -> str:
    sr = 24000
    # 0.5s silence + 0.5s 140 Hz (frames 12..23) + 0.5s 280 Hz (frames 24..35) @ 24fps
    silence = torch.zeros(sr // 2, dtype=torch.float32)
    low = _harmonic_wave(140.0, duration_s=0.5, sr=sr)
    high = _harmonic_wave(280.0, duration_s=0.5, sr=sr)
    wav = torch.cat([silence, low, high])
    f0, _, _ = m2.yin_f0(wav, sr=sr)

    with tempfile.TemporaryDirectory() as td:
        vs_path = Path(td) / "voice_spans.json"
        vs_path.write_text(
            json.dumps({"fps": 24.0, "voiced_spans_inclusive": [[13, 22]]}),
            encoding="utf-8",
        )
        mask = m2.load_voice_span_mask(vs_path, f0.numel(), sr=sr)
        summary = m2.summarize_f0_series(f0, span_mask=mask)
    assert summary["voiced_hops"] >= 5 and summary["median_f0_hz"] is not None, summary
    assert abs(12.0 * math.log2(summary["median_f0_hz"] / 140.0)) < 0.2, summary

    # Pure quiet signal stays unvoiced under -45 dBFS gate
    quiet = _harmonic_wave(180.0, duration_s=0.5, sr=sr, amp=1e-4)
    qf0, _, _ = m2.yin_f0(quiet, sr=sr)
    assert int(torch.isfinite(qf0).sum().item()) == 0
    return "silence gate rejects -80 dBFS & voice_spans.json isolates [13..22] 140 Hz band"


def check_m1_score_clip_embeddings() -> str:
    torch.manual_seed(0)
    ref = torch.randn(1024)
    ref_pat = torch.randn(196, 1024)

    # Identical copy / frozen frames trigger both copy and freeze advisory notes
    frozen_cls = ref.unsqueeze(0).repeat(6, 1)
    frozen_pat = ref_pat.unsqueeze(0).repeat(6, 1, 1)
    frozen_res = m1.score_clip_embeddings(frozen_cls, frame_patches=frozen_pat, ref_vec=ref, ref_patches=ref_pat)
    assert abs(frozen_res["ref_cos_mean"] - 1.0) < 1e-4, frozen_res
    assert abs(frozen_res["ref_patch_cos_mean"] - 1.0) < 1e-4, frozen_res
    assert abs(frozen_res["temporal_self_cos_mean"] - 1.0) < 1e-4, frozen_res
    assert len(frozen_res["triage_notes"]) >= 2, frozen_res

    # Uncorrelated moving frames produce ~0.0 cosine and 0 triage notes
    moving_cls = torch.randn(6, 1024)
    moving_pat = torch.randn(6, 196, 1024)
    moving_res = m1.score_clip_embeddings(moving_cls, frame_patches=moving_pat, ref_vec=ref, ref_patches=ref_pat)
    assert moving_res["ref_cos_mean"] < 0.25 and moving_res["temporal_self_cos_mean"] < 0.25, moving_res
    assert moving_res["triage_notes"] == [], moving_res
    return "DINOv3 copy & freeze scoring flags identical traces and stays silent on normal motion"


def main() -> int:
    case("m2_yin_sine_and_harmonic_shift", check_m2_yin_sine_and_harmonic_shift)
    case("m2_silence_gate_and_voice_spans", check_m2_silence_gate_and_voice_spans)
    case("m1_score_clip_embeddings", check_m1_score_clip_embeddings)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
