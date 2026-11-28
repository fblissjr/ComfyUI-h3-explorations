#!/usr/bin/env python3
"""Deterministic verification suite for `bench/h3_stem_sep.py`.

Covers:
1. Nulltest controls: identical copy (<= -90 dB), half-level copy (-6.02 +- 0.1 dB),
   and 24 kHz <-> 44.1 kHz sample-rate copy (<= -45 dB).
2. Cascade separation on a 6.0 s test audio slice (or `$H3_TEST_AUDIO` if provided) with
   manifest & on-disk nulltest <= -80.0 dB.
3. build_cover_scaffold(..., semitones=-8.0) on those separated stems verifying
   32 kHz stereo output, exact sample count, non-silence, and peak <= 0.93.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torchaudio.functional as TAF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402
import h3_stem_sep as stem_sep  # noqa: E402


def _make_bandlimited_stereo(sr: int = 44100, dur_s: float = 2.0) -> torch.Tensor:
    t = torch.arange(int(round(sr * dur_s)), dtype=torch.float32) / float(sr)
    l = 0.28 * torch.sin(2.0 * math.pi * 220.0 * t) + 0.18 * torch.sin(2.0 * math.pi * 880.0 * t)
    r = 0.24 * torch.sin(2.0 * math.pi * 330.0 * t) + 0.16 * torch.sin(2.0 * math.pi * 1320.0 * t)
    return torch.stack([l, r], dim=0)


def check_nulltest_identical() -> str:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sig = _make_bandlimited_stereo(44100, 1.5)
        ref = stem_sep._write_wav_pcm16(tmp / "ref.wav", sig, 44100)
        copy = stem_sep._write_wav_pcm16(tmp / "copy.wav", sig, 44100)
        res = stem_sep.run_nulltest(ref, [copy])
        assert res["null_depth_db"] <= -90.0, f"expected <= -90 dB, got {res}"
        assert res["passed"] is True
        return f"null_depth_db={res['null_depth_db']:.3f} dB"


def check_nulltest_half_level() -> str:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sig = _make_bandlimited_stereo(44100, 1.5)
        ref = stem_sep._write_wav_pcm16(tmp / "ref.wav", sig, 44100)
        half = stem_sep._write_wav_pcm16(tmp / "half.wav", 0.5 * sig, 44100)
        res = stem_sep.run_nulltest(ref, [half])
        expected = 20.0 * math.log10(0.5)  # -6.0206 dB
        assert abs(res["null_depth_db"] - expected) <= 0.1, (
            f"expected {expected:.2f} +- 0.1 dB, got {res}"
        )
        assert res["passed"] is False
        return f"null_depth_db={res['null_depth_db']:.3f} dB (expected ~{expected:.2f} dB)"


def check_nulltest_sample_rate_roundtrip() -> str:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sig_44k = _make_bandlimited_stereo(44100, 2.0)
        sig_24k = TAF.resample(sig_44k, orig_freq=44100, new_freq=24000)
        ref = stem_sep._write_wav_pcm16(tmp / "ref_44k.wav", sig_44k, 44100)
        stem_24k = stem_sep._write_wav_pcm16(tmp / "stem_24k.wav", sig_24k, 24000)
        res = stem_sep.run_nulltest(ref, [stem_24k])
        assert res["sample_rate_mismatch"] is True
        assert res["null_depth_db"] <= -45.0, f"expected <= -45 dB, got {res}"
        return f"null_depth_db={res['null_depth_db']:.3f} dB (24k <-> 44.1k)"


def check_cascade_and_cover_scaffold() -> str:
    test_audio = os.environ.get("H3_TEST_AUDIO", "")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        if test_audio and Path(test_audio).is_file():
            src_audio = Path(test_audio)
            slice_start = 32.0
        else:
            src_audio = tmp / "synth_mix.wav"
            sr = stem_sep.STEM_SR
            t = torch.arange(int(round(6.0 * sr)), dtype=torch.float32) / float(sr)
            inst_hz = 280.0 + 18.0 * torch.sin(2.0 * math.pi * 5.2 * t)
            ph = 2.0 * math.pi * torch.cumsum(inst_hz, dim=0) / float(sr)
            syllable = 0.65 + 0.35 * torch.sin(2.0 * math.pi * 3.0 * t)
            voc = syllable * (0.24 * torch.sin(ph) + 0.18 * torch.sin(2.0 * ph) + 0.12 * torch.sin(3.0 * ph) + 0.07 * torch.sin(4.0 * ph))
            bass = 0.20 * torch.sin(2.0 * math.pi * 73.42 * t)
            harm = 0.12 * torch.sin(2.0 * math.pi * 220.0 * t) + 0.08 * torch.sin(2.0 * math.pi * 440.0 * t)
            synth = torch.stack([voc + bass + harm, voc * 0.95 + bass + harm * 1.05], dim=0)
            stem_sep._write_wav_pcm16(src_audio, synth, sr)
            slice_start = 0.0

        stems_dir = tmp / "stems"
        manifest = stem_sep.separate_to_dir(
            src_audio,
            stems_dir,
            recipe="cascade",
            start_s=slice_start,
            duration_s=6.0,
        )
        for name in ("mix.wav", "vocals.wav", "drums.wav", "bass.wav", "harmonic.wav", "manifest.json"):
            assert (stems_dir / name).is_file(), f"missing output file {name}"

        stem_list = [stems_dir / s for s in ("vocals.wav", "drums.wav", "bass.wav", "harmonic.wav")]
        null_res = stem_sep.run_nulltest(stems_dir / "mix.wav", stem_list)
        assert null_res["null_depth_db"] <= -80.0, f"cascade nulltest failed: {null_res}"
        assert null_res["passed"] is True
        assert abs(manifest["null_depth_db"] - null_res["null_depth_db"]) < 1e-3

        out_scaf = tmp / "cover_scaf_32k.wav"
        out_voc = tmp / "cover_voc_32k.wav"
        meta = stem_sep.build_cover_scaffold(
            stems_dir,
            start_s=0.0,
            duration_s=6.0,
            semitones=-8.0,
            out_wav_32k=out_scaf,
            out_voc_wav_32k=out_voc,
        )
        expected_samples = round(6.0 * 32000)
        assert meta["samples"] == expected_samples
        assert meta["inst_semitones"] == 4.0, meta
        assert stem_sep._fold_inst_semitones(-12.0) == 0.0
        assert stem_sep._fold_inst_semitones(-5.0) == -5.0
        for wav_path in (out_scaf, out_voc):
            assert wav_path.is_file(), f"missing scaffold output {wav_path}"
            w, sr = stem_sep._load_audio_native(wav_path)
            assert sr == 32000, f"{wav_path.name}: expected 32000 Hz, got {sr}"
            assert w.shape == (2, expected_samples), f"{wav_path.name}: got shape {tuple(w.shape)}"
            rms = float(w.pow(2).mean().sqrt().item())
            peak = float(w.abs().max().item())
            assert rms > 1e-4, f"{wav_path.name}: unexpectedly silent (rms={rms})"
            assert peak <= 0.93, f"{wav_path.name}: peak {peak:.4f} exceeds 0.93"

        # Mono-vocal collapse produces bit-identical L/R channels on vocal-only scaffold
        out_voc_only = tmp / "voc_only_32k.wav"
        meta_vo = stem_sep.build_cover_scaffold(
            stems_dir,
            start_s=0.0,
            duration_s=6.0,
            semitones=-5.0,
            harmonic_db=-120.0,
            bass_db=-120.0,
            mono_vocal=True,
            out_wav_32k=out_voc_only,
        )
        assert meta_vo["mono_vocal"] is True
        w_vo, _ = stem_sep._load_audio_native(out_voc_only)
        lr_corr = float(torch.corrcoef(w_vo)[0, 1].item())
        assert lr_corr >= 0.9999, f"expected mono_vocal L/R corr >= 0.9999, got {lr_corr:.6f}"

        # Shifter 'atempo' (WSOLA) and backing-only (vocal_db=-120.0) both produce exact shapes & non-silence
        out_atempo = tmp / "atempo_32k.wav"
        meta_at = stem_sep.build_cover_scaffold(
            stems_dir,
            start_s=0.0,
            duration_s=6.0,
            semitones=-12.0,
            shifter="atempo",
            mono_vocal=True,
            out_wav_32k=out_atempo,
        )
        assert meta_at["shifter"] == "atempo"
        w_at, sr_at = stem_sep._load_audio_native(out_atempo)
        assert sr_at == 32000 and w_at.shape == (2, expected_samples)
        assert float(w_at.pow(2).mean().sqrt().item()) > 1e-4

        out_back = tmp / "backing_only_32k.wav"
        stem_sep.build_cover_scaffold(
            stems_dir,
            start_s=0.0,
            duration_s=6.0,
            semitones=-5.0,
            vocal_db=-120.0,
            harmonic_db=0.0,
            bass_db=-3.0,
            out_wav_32k=out_back,
        )
        w_bk, _ = stem_sep._load_audio_native(out_back)
        assert w_bk.shape == (2, expected_samples)
        assert float(w_bk.pow(2).mean().sqrt().item()) > 1e-4

        # Quick CLI smoke test on recipes subcommand
        cli_out = subprocess.check_output(
            [sys.executable, str(Path(__file__).resolve().parent / "h3_stem_sep.py"), "recipes", "--json"],
            text=True,
        )
        recipes = json.loads(cli_out)
        assert set(recipes) == {"cascade", "vocals", "four", "karaoke", "no_drums"}

        return (
            f"cascade null_depth_db={null_res['null_depth_db']:.3f} dB; "
            f"scaffold={expected_samples} samples @ 32 kHz, peak={meta['peak']:.4f}"
        )


def main() -> int:
    case("nulltest_identical_copy", check_nulltest_identical)
    case("nulltest_half_level_copy", check_nulltest_half_level)
    case("nulltest_24k_44k_roundtrip", check_nulltest_sample_rate_roundtrip)
    case("cascade_and_cover_scaffold", check_cascade_and_cover_scaffold)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
