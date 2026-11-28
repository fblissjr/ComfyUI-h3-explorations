#!/usr/bin/env python3
"""Null-locked stem separation (MelBand-RoFormer + HDemucs) and cover-scaffold builder.

Library + CLI for exact-residual stem separation at 44,100 Hz stereo and
RubberBand-pitched 32,000 Hz cover scaffolds for MiniMax H3 audio-video experiments.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as TAF
from scipy.io import wavfile

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
if str(REPO / "bench") not in sys.path:
    sys.path.insert(0, str(REPO / "bench"))
from _lib import bootstrap  # noqa: E402

bootstrap(cpu=None)

STEM_SR = 44100
SCAFFOLD_SR = 32000
ROFORMER_CHUNK_SAMPLES = 352800  # 8.0 s at 44.1 kHz (800 * hop_length 441)
COMFY_SERVER_URL = "http://127.0.0.1:8188"

RECIPES: dict[str, dict[str, Any]] = {
    "cascade": {
        "description": (
            "MelBandRoFormer vocals + HDemucs drums/bass on exact instrumental, "
            "with harmonic = inst - drums - bass for bit-exact mix reconstruction."
        ),
        "stems": ["vocals.wav", "drums.wav", "bass.wav", "harmonic.wav"],
        "derived": [],
    },
    "vocals": {
        "description": "MelBandRoFormer 2-stem separation (vocals.wav + exact residual instrumental.wav).",
        "stems": ["vocals.wav", "instrumental.wav"],
        "derived": [],
    },
    "four": {
        "description": "HDemucs 4-stem separation (drums, bass, other, vocals) with null-locked other.",
        "stems": ["drums.wav", "bass.wav", "other.wav", "vocals.wav"],
        "derived": [],
    },
    "karaoke": {
        "description": "MelBandRoFormer karaoke split (instrumental.wav + vocals.wav).",
        "stems": ["instrumental.wav", "vocals.wav"],
        "derived": [],
    },
    "no_drums": {
        "description": "Cascade separation plus summed no_drums.wav (vocals + harmonic + bass).",
        "stems": ["vocals.wav", "drums.wav", "bass.wav", "harmonic.wav"],
        "derived": ["no_drums.wav"],
    },
}


# ---------------------------------------------------------------------------
# VRAM & ComfyUI server safety
# ---------------------------------------------------------------------------


def _comfy_queue_busy(timeout_s: float = 1.5) -> bool:
    """Return True if local ComfyUI :8188 has running or pending queue items."""
    try:
        with urllib.request.urlopen(f"{COMFY_SERVER_URL}/queue", timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        running = data.get("queue_running") or []
        pending = data.get("queue_pending") or []
        return bool(running or pending)
    except (urllib.error.URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError):
        return False


def _free_comfy_vram(timeout_s: float = 2.0) -> None:
    """Ask idle :8188 to unload models and free VRAM before CUDA allocation."""
    payload = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFY_SERVER_URL}/free",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read()
    except (urllib.error.URLError, OSError, TimeoutError):
        pass


def _resolve_compute_device(device: torch.device | str | None) -> torch.device:
    """Resolve requested device, checking :8188 queue and freeing VRAM when idle."""
    if device is None:
        want = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        want = torch.device(device)
    if want.type != "cuda" or not torch.cuda.is_available():
        return torch.device("cpu")
    if _comfy_queue_busy():
        print("note: :8188 queue is active; falling back to CPU for stem separation", file=sys.stderr)
        return torch.device("cpu")
    _free_comfy_vram()
    return want


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# ---------------------------------------------------------------------------
# Audio I/O (ffmpeg + stdlib wave; never torchaudio.load / torchaudio.save)
# ---------------------------------------------------------------------------


def _ensure_stereo_2d(wav: torch.Tensor) -> torch.Tensor:
    x = wav.detach().cpu().to(torch.float32)
    if x.ndim == 3:
        x = x[0]
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"Expected 1D/2D/3D audio tensor, got shape {tuple(wav.shape)}")
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    elif x.shape[0] > 2:
        x = x[:2, :]
    return x.contiguous()


def _write_wav_pcm16(path: Path, tensor_c_t: torch.Tensor, sample_rate: int) -> Path:
    """Write a [C, T] float tensor as PCM16 WAV using stdlib wave."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = tensor_c_t.detach().cpu().to(torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    x = x.clamp(-1.0, 1.0).transpose(0, 1).contiguous()
    pcm = (x * 32767.0).round().clamp(-32768, 32767).to(torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(int(x.shape[1]))
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm)
    return path


def _load_audio_ffmpeg(
    path: Path,
    sr: int = STEM_SR,
    start_s: float | None = None,
    duration_s: float | None = None,
    channels: int = 2,
) -> torch.Tensor:
    """Decode any audio/video container to [channels, T] float32 via ffmpeg f32le pipe."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing audio/media input: {path}")
    cmd = ["ffmpeg", "-v", "error"]
    if start_s is not None and start_s > 0:
        cmd.extend(["-ss", f"{start_s:.6f}"])
    if duration_s is not None:
        cmd.extend(["-t", f"{duration_s:.6f}"])
    cmd.extend([
        "-i", str(path),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sr),
        "-f", "f32le",
        "-",
    ])
    raw = subprocess.check_output(cmd)
    if not raw:
        raise RuntimeError(f"ffmpeg produced empty audio stream from {path}")
    arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(-1, channels).T
    return torch.from_numpy(arr)


def _probe_sample_rate(path: Path) -> int:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return int(out)


def _load_audio_native(path: Path) -> tuple[torch.Tensor, int]:
    """Load WAV (via wave/scipy) or arbitrary media (via ffprobe + ffmpeg) as [2, T] float32 + sr."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing audio file: {path}")
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                n_ch = wf.getnchannels()
                sw = wf.getsampwidth()
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)
            if sw == 2:
                arr = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            elif sw == 4:
                arr = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
            elif sw == 1:
                arr = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                arr = None
            if arr is not None:
                arr = arr.reshape(-1, n_ch).T.copy()
                return _ensure_stereo_2d(torch.from_numpy(arr)), int(sr)
        except wave.Error:
            pass
        try:
            sr, data = wavfile.read(str(path))
            if data.dtype == np.int16:
                arr = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                arr = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                arr = (data.astype(np.float32) - 128.0) / 128.0
            else:
                arr = data.astype(np.float32)
            if arr.ndim == 1:
                arr = np.stack([arr, arr], axis=-1)
            return _ensure_stereo_2d(torch.from_numpy(arr.T.copy())), int(sr)
        except Exception:
            pass
    sr = _probe_sample_rate(path)
    return _load_audio_ffmpeg(path, sr=sr), sr


# ---------------------------------------------------------------------------
# Model loaders & chunked overlap-add separation
# ---------------------------------------------------------------------------


def _roformer_model_config() -> dict[str, Any]:
    return {
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
        "sample_rate": STEM_SR,
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


def _load_roformer_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    mbr_dir = COMFY / "custom_nodes" / "ComfyUI-MelBandRoFormer"
    if not mbr_dir.is_dir():
        raise FileNotFoundError(f"Missing ComfyUI-MelBandRoFormer directory: {mbr_dir}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing MelBandRoFormer checkpoint: {ckpt_path}")

    if str(mbr_dir) not in sys.path:
        sys.path.insert(0, str(mbr_dir))
    MelBandRoformer = importlib.import_module("model.mel_band_roformer").MelBandRoformer

    if ckpt_path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(ckpt_path))
    else:
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    model = MelBandRoformer(**_roformer_model_config()).eval()
    model.load_state_dict(sd, strict=True)
    return model.to(device)


def _load_mel_roformer(device: torch.device) -> torch.nn.Module:
    ckpt_path = COMFY / "models" / "diffusion_models" / "MelBandRoformer_fp16.safetensors"
    return _load_roformer_model(ckpt_path, device)


def _dereverb_ckpt_path() -> Path:
    return (
        COMFY
        / "models"
        / "diffusion_models"
        / "oulianov_melband-roformer-denoise"
        / "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"
    )


def _trapezoidal_window(chunk: int, fade_size: int) -> torch.Tensor:
    win = torch.ones(chunk, dtype=torch.float32)
    win[:fade_size] *= torch.linspace(0.0, 1.0, fade_size, dtype=torch.float32)
    win[-fade_size:] *= torch.linspace(1.0, 0.0, fade_size, dtype=torch.float32)
    return win


def _pad_reflect_safe(x: torch.Tensor, pad_left: int, pad_right: int) -> torch.Tensor:
    """Reflect-pad [C, L] even when pad_left/pad_right >= L."""
    cur = x
    left_rem = int(pad_left)
    right_rem = int(pad_right)
    while left_rem > 0 or right_rem > 0:
        cur_len = int(cur.shape[-1])
        if cur_len <= 1:
            cur = F.pad(cur, (left_rem, right_rem), mode="constant", value=0.0)
            break
        step_l = min(left_rem, cur_len - 1)
        step_r = min(right_rem, cur_len - 1)
        cur = F.pad(cur, (step_l, step_r), mode="reflect")
        left_rem -= step_l
        right_rem -= step_r
    return cur


@torch.no_grad()
def _run_roformer_on_device(
    wav_44k: torch.Tensor,
    device: torch.device,
    *,
    ckpt_path: Path | None = None,
) -> torch.Tensor:
    model = _load_mel_roformer(device) if ckpt_path is None else _load_roformer_model(ckpt_path, device)
    try:
        orig_len = int(wav_44k.shape[1])
        if orig_len == 0:
            return torch.zeros_like(wav_44k)

        chunk = ROFORMER_CHUNK_SAMPLES
        step = chunk // 2  # 50% overlap
        fade_size = chunk // 10

        def _forward_chunk(part_cpu: torch.Tensor) -> torch.Tensor:
            part_dev = part_cpu.unsqueeze(0).to(device=device, dtype=torch.float32)
            out = model(part_dev)[0]
            return out.detach().to(device="cpu", dtype=torch.float32)

        if orig_len <= chunk:
            padded = _pad_reflect_safe(wav_44k, 0, chunk - orig_len)
            return _forward_chunk(padded)[:, :orig_len]

        border = chunk - step
        padded = _pad_reflect_safe(wav_44k, border, border)
        total_len = int(padded.shape[1])
        base_win = _trapezoidal_window(chunk, fade_size)

        vocals_acc = torch.zeros_like(padded, dtype=torch.float32)
        weight_acc = torch.zeros(total_len, dtype=torch.float32)

        for start in range(0, total_len, step):
            part = padded[:, start : start + chunk]
            length = int(part.shape[1])
            if length < chunk:
                part = _pad_reflect_safe(part, 0, chunk - length)
            est = _forward_chunk(part)[:, :length]
            win = base_win.clone()
            if start == 0:
                win[:fade_size] = 1.0
            elif start + chunk >= total_len:
                win[-fade_size:] = 1.0
            w = win[:length]
            vocals_acc[:, start : start + length] += est * w.unsqueeze(0)
            weight_acc[start : start + length] += w
            if start + chunk >= total_len:
                break

        vocals_full = vocals_acc / weight_acc.clamp(min=1e-8).unsqueeze(0)
        return vocals_full[:, border : border + orig_len].contiguous()
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def dereverb_vocals_roformer(wav_44k: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
    """Run oulianov MelBand-RoFormer denoise/dereverb (`target_instrument: dry`) on [2, L] 44.1 kHz vocals."""
    ckpt = _dereverb_ckpt_path()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing dereverb checkpoint: {ckpt}")
    orig = _ensure_stereo_2d(wav_44k)
    run_device = _resolve_compute_device(device)
    try:
        return _run_roformer_on_device(orig, run_device, ckpt_path=ckpt)
    except Exception as exc:
        if run_device.type == "cuda" and _is_cuda_oom(exc):
            print("warning: CUDA OOM in dereverb RoFormer; retrying on CPU", file=sys.stderr)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return _run_roformer_on_device(orig, torch.device("cpu"), ckpt_path=ckpt)
        raise


def separate_vocals_roformer(
    wav_44k: torch.Tensor,
    device: torch.device,
    *,
    null_lock: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate vocals from a [2, L] 44.1 kHz tensor via chunked overlap-add MelBandRoFormer.

    When `null_lock=True`, enforces exact residual `instrumental = wav_44k - vocals`.
    """
    orig = _ensure_stereo_2d(wav_44k)
    run_device = _resolve_compute_device(device)
    try:
        vocals = _run_roformer_on_device(orig, run_device)
    except Exception as exc:
        if run_device.type == "cuda" and _is_cuda_oom(exc):
            print("warning: CUDA OOM in MelBandRoFormer; retrying on CPU", file=sys.stderr)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            vocals = _run_roformer_on_device(orig, torch.device("cpu"))
        else:
            raise
    if null_lock:
        instrumental = orig - vocals
    else:
        instrumental = orig - vocals
    return vocals.contiguous(), instrumental.contiguous()


@torch.no_grad()
def _run_hdemucs_on_device(
    wav_44k: torch.Tensor,
    device: torch.device,
    *,
    segment_s: float,
    overlap: float,
) -> tuple[list[str], torch.Tensor]:
    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model().eval().to(device)
    try:
        sources = list(model.sources)
        n_sources = len(sources)
        orig_len = int(wav_44k.shape[1])
        if orig_len == 0:
            return sources, torch.zeros((n_sources, 2, 0), dtype=torch.float32)

        seg_len = max(4410, int(round(segment_s * STEM_SR)))
        ov = min(max(float(overlap), 0.0), 0.75)
        step = max(1, int(round(seg_len * (1.0 - ov))))

        if orig_len <= seg_len:
            in_len = max(orig_len, 4410)
            chunk = _pad_reflect_safe(wav_44k, 0, in_len - orig_len) if orig_len < in_len else wav_44k
            out = model(chunk.unsqueeze(0).to(device=device, dtype=torch.float32))[0]
            return sources, out[:, :, :orig_len].detach().to(device="cpu", dtype=torch.float32).contiguous()

        win = torch.hann_window(seg_len + 2, dtype=torch.float32)[1:-1]
        out_acc = torch.zeros((n_sources, 2, orig_len), dtype=torch.float32)
        weight_acc = torch.zeros(orig_len, dtype=torch.float32)

        for start in range(0, orig_len, step):
            part = wav_44k[:, start : start + seg_len]
            length = int(part.shape[1])
            if length < seg_len:
                part = _pad_reflect_safe(part, 0, seg_len - length)
            est = model(part.unsqueeze(0).to(device=device, dtype=torch.float32))[0]
            est_cpu = est[:, :, :length].detach().to(device="cpu", dtype=torch.float32)
            w = win[:length]
            out_acc[:, :, start : start + length] += est_cpu * w.view(1, 1, -1)
            weight_acc[start : start + length] += w
            if start + seg_len >= orig_len:
                break

        out_full = out_acc / weight_acc.clamp(min=1e-8).view(1, 1, -1)
        return sources, out_full.contiguous()
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def separate_hdemucs(
    wav_44k: torch.Tensor,
    device: torch.device,
    *,
    target_mix: torch.Tensor | None = None,
    null_lock: bool = True,
    segment_s: float = 10.0,
    overlap: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Separate [2, L] 44.1 kHz audio into {'drums', 'bass', 'other', 'vocals'} via HDemucs."""
    orig = _ensure_stereo_2d(wav_44k)
    run_device = _resolve_compute_device(device)
    try:
        names, out_tensor = _run_hdemucs_on_device(orig, run_device, segment_s=segment_s, overlap=overlap)
    except Exception as exc:
        if run_device.type == "cuda" and _is_cuda_oom(exc):
            print("warning: CUDA OOM in HDemucs; retrying on CPU", file=sys.stderr)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            names, out_tensor = _run_hdemucs_on_device(
                orig, torch.device("cpu"), segment_s=segment_s, overlap=overlap
            )
        else:
            raise

    stems: dict[str, torch.Tensor] = {
        name: out_tensor[idx].contiguous() for idx, name in enumerate(names)
    }
    if null_lock:
        ref = orig if target_mix is None else _ensure_stereo_2d(target_mix)[:, : orig.shape[1]]
        stems["other"] = (ref - (stems["drums"] + stems["bass"] + stems["vocals"])).contiguous()
    return stems


# ---------------------------------------------------------------------------
# Nulltest & recipes
# ---------------------------------------------------------------------------


def compute_nulltest(
    ref_wav: torch.Tensor,
    stem_wavs: list[torch.Tensor],
    *,
    ref_sr: int = STEM_SR,
    stem_srs: list[int] | None = None,
) -> dict[str, Any]:
    """Compare reference audio against sum(stems) and compute null depth in dB."""
    if not stem_wavs:
        raise ValueError("At least one stem tensor is required for nulltest")
    ref = _ensure_stereo_2d(ref_wav)
    if stem_srs is None:
        stem_srs = [ref_sr] * len(stem_wavs)
    if len(stem_srs) != len(stem_wavs):
        raise ValueError("stem_srs length must match stem_wavs length")

    mismatch = any(int(sr) != int(ref_sr) for sr in stem_srs)
    aligned_stems: list[torch.Tensor] = []
    for wav, sr in zip(stem_wavs, stem_srs):
        w = _ensure_stereo_2d(wav)
        if int(sr) != int(ref_sr):
            w = TAF.resample(w, orig_freq=int(sr), new_freq=int(ref_sr))
        aligned_stems.append(w)

    n = min(int(ref.shape[1]), *(int(s.shape[1]) for s in aligned_stems))
    if n == 0:
        raise ValueError("Cannot compute nulltest on empty audio")

    ref_a = ref[:, :n]
    stem_sum = torch.zeros_like(ref_a)
    for s in aligned_stems:
        stem_sum += s[:, :n]
    residual = ref_a - stem_sum

    ref_rms = float(ref_a.pow(2).mean().sqrt().item())
    res_rms = float(residual.pow(2).mean().sqrt().item())
    max_abs = float(residual.abs().max().item())

    ref_rms_dbfs = 20.0 * math.log10(max(ref_rms, 1e-12))
    res_rms_dbfs = 20.0 * math.log10(max(res_rms, 1e-12))
    null_depth_db = res_rms_dbfs - ref_rms_dbfs
    threshold_db = -55.0 if mismatch else -80.0

    return {
        "ref_rms_dbfs": round(ref_rms_dbfs, 3),
        "residual_rms_dbfs": round(res_rms_dbfs, 3),
        "null_depth_db": round(null_depth_db, 3),
        "max_abs_diff": round(max_abs, 6),
        "sample_rate_hz": int(ref_sr),
        "sample_rate_mismatch": bool(mismatch),
        "samples_compared": int(n),
        "threshold_db": threshold_db,
        "passed": bool(null_depth_db <= threshold_db),
    }


def _resolve_stem_paths(targets: list[Path]) -> list[Path]:
    if len(targets) == 1 and targets[0].is_dir():
        d = targets[0]
        manifest_path = d / "manifest.json"
        if manifest_path.is_file():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            stem_names = data.get("stems") or []
            if stem_names:
                return [d / name for name in stem_names]
        excluded = {"mix.wav", "no_drums.wav"}
        wavs = sorted(p for p in d.glob("*.wav") if p.name not in excluded)
        if not wavs:
            raise FileNotFoundError(f"No stem WAVs found in {d}")
        return wavs
    return list(targets)


def run_nulltest(reference: Path, targets: list[Path]) -> dict[str, Any]:
    """Load reference + stem files from disk and return nulltest JSON dict."""
    stem_paths = _resolve_stem_paths(targets)
    ref_wav, ref_sr = _load_audio_native(reference)
    stem_wavs: list[torch.Tensor] = []
    stem_srs: list[int] = []
    for sp in stem_paths:
        sw, ssr = _load_audio_native(sp)
        stem_wavs.append(sw)
        stem_srs.append(ssr)
    res = compute_nulltest(ref_wav, stem_wavs, ref_sr=ref_sr, stem_srs=stem_srs)
    res["reference"] = str(reference)
    res["stems"] = [str(p) for p in stem_paths]
    return res


def _quantize_pcm16_grid(tensor_c_t: torch.Tensor) -> torch.Tensor:
    """Snap [C, T] tensor onto the PCM16 quantization grid for exact on-disk residuals."""
    x = tensor_c_t.detach().cpu().to(torch.float32).clamp(-1.0, 1.0)
    return (x * 32767.0).round().clamp(-32768, 32767) / 32767.0


def separate_to_dir(
    input_path: Path,
    out_dir: Path,
    *,
    recipe: str = "cascade",
    device: torch.device | str | None = None,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> dict[str, Any]:
    """Separate input media at 44.1 kHz stereo into `out_dir` with manifest + mix.wav."""
    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe {recipe!r}; expected one of {sorted(RECIPES)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    mix_44k = _load_audio_ffmpeg(input_path, sr=STEM_SR, start_s=start_s, duration_s=duration_s)

    raw_exported: dict[str, torch.Tensor] = {}
    raw_derived: dict[str, torch.Tensor] = {}

    if recipe in ("cascade", "no_drums"):
        vocals, inst = separate_vocals_roformer(mix_44k, dev, null_lock=True)
        h_stems = separate_hdemucs(inst, dev, target_mix=inst, null_lock=True)
        harmonic = (inst - h_stems["drums"] - h_stems["bass"]).contiguous()
        raw_exported = {
            "vocals.wav": vocals,
            "drums.wav": h_stems["drums"],
            "bass.wav": h_stems["bass"],
            "harmonic.wav": harmonic,
        }
        if recipe == "no_drums":
            raw_derived["no_drums.wav"] = (vocals + harmonic + h_stems["bass"]).contiguous()
    elif recipe in ("vocals", "karaoke"):
        vocals, inst = separate_vocals_roformer(mix_44k, dev, null_lock=True)
        if recipe == "vocals":
            raw_exported = {"vocals.wav": vocals, "instrumental.wav": inst}
        else:
            raw_exported = {"instrumental.wav": inst, "vocals.wav": vocals}
    elif recipe == "four":
        h_stems = separate_hdemucs(mix_44k, dev, target_mix=mix_44k, null_lock=True)
        raw_exported = {
            "drums.wav": h_stems["drums"],
            "bass.wav": h_stems["bass"],
            "other.wav": h_stems["other"],
            "vocals.wav": h_stems["vocals"],
        }

    max_peak = max(
        [float(mix_44k.abs().max())]
        + [float(t.abs().max()) for t in raw_exported.values()]
        + [float(t.abs().max()) for t in raw_derived.values()]
    )
    shared_gain = 0.98 / max_peak if max_peak > 0.999 else 1.0
    mix_scaled = mix_44k * shared_gain if shared_gain != 1.0 else mix_44k
    mix_path = _write_wav_pcm16(out_dir / "mix.wav", mix_scaled, STEM_SR)
    mix_q = _quantize_pcm16_grid(mix_scaled)

    stem_keys = list(raw_exported.keys())
    residual_key = stem_keys[-1]
    exported: dict[str, torch.Tensor] = {}
    for k in stem_keys[:-1]:
        exported[k] = _quantize_pcm16_grid(raw_exported[k] * shared_gain)
    exported[residual_key] = (mix_q - sum(exported.values())).contiguous()
    derived = {k: _quantize_pcm16_grid(v * shared_gain) for k, v in raw_derived.items()}

    stem_paths: list[Path] = []
    for fname, tensor in exported.items():
        stem_paths.append(_write_wav_pcm16(out_dir / fname, tensor, STEM_SR))
    for fname, tensor in derived.items():
        _write_wav_pcm16(out_dir / fname, tensor, STEM_SR)

    null_report = run_nulltest(mix_path, stem_paths)
    manifest = {
        "recipe": recipe,
        "source": str(input_path),
        "sample_rate": STEM_SR,
        "samples": int(mix_44k.shape[1]),
        "duration_s": round(float(mix_44k.shape[1]) / STEM_SR, 6),
        "shared_headroom_gain": round(shared_gain, 6),
        "stems": list(exported.keys()),
        "derived": list(derived.keys()),
        "null_depth_db": null_report["null_depth_db"],
        "nulltest": null_report,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# Cover scaffold builder (32 kHz stereo, librubberband / WSOLA atempo pitch shift)
# ---------------------------------------------------------------------------

SHIFTERS: tuple[str, ...] = ("rubberband", "atempo")


def _atempo_filter_chain(tempo: float) -> str:
    """Build chained ffmpeg `atempo` filters keeping each stage in [0.5, 2.0]."""
    t = float(tempo)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError(f"tempo must be positive and finite, got {tempo!r}")
    stages: list[float] = []
    while t > 2.0 + 1e-6:
        stages.append(2.0)
        t /= 2.0
    while t < 0.5 - 1e-6:
        stages.append(0.5)
        t /= 0.5
    stages.append(t)
    return ",".join(f"atempo={s:.9f}" for s in stages)


def _shift_44k(
    wav_44k: torch.Tensor,
    semitones: float,
    *,
    formant: bool = False,
    shifter: str = "rubberband",
    collapse_mono: bool = False,
) -> torch.Tensor:
    """Pitch-shift [2, T] 44.1 kHz audio with `channels=together` / optional mono collapse."""
    if shifter not in SHIFTERS:
        raise ValueError(f"Unsupported shifter {shifter!r}; expected one of {SHIFTERS}")
    target_len = int(wav_44k.shape[1])
    if target_len == 0:
        return wav_44k.clone()

    if collapse_mono:
        mono = wav_44k.to(torch.float32).mean(dim=0, keepdim=True)
        work = mono
        n_ch = 1
    else:
        work = _ensure_stereo_2d(wav_44k)
        n_ch = 2

    if abs(semitones) <= 1e-6:
        out = work.repeat(2, 1) if collapse_mono else work.clone()
        return out.contiguous()

    ratio = 2.0 ** (float(semitones) / 12.0)
    if shifter == "rubberband":
        flt = f"rubberband=pitch={ratio:.9f}:pitchq=quality:channels=together"
        if formant:
            flt += ":formant=preserved"
    else:
        atempo_chain = _atempo_filter_chain(1.0 / ratio)
        flt = (
            f"asetrate={STEM_SR * ratio:.6f},"
            f"aresample={STEM_SR}:resampler=soxr,"
            f"{atempo_chain}"
        )

    with tempfile.TemporaryDirectory() as td:
        pin = Path(td) / "in.wav"
        pout = Path(td) / "out.wav"
        _write_wav_pcm16(pin, work, STEM_SR)
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(pin),
                "-af", flt,
                "-ar", str(STEM_SR),
                "-ac", str(n_ch),
                str(pout),
            ],
            check=True,
        )
        shifted, sr = _load_audio_native(pout)
        if sr != STEM_SR:
            shifted = TAF.resample(shifted, orig_freq=sr, new_freq=STEM_SR)
    if collapse_mono:
        shifted = shifted[:1].repeat(2, 1)
    if shifted.shape[1] < target_len:
        shifted = F.pad(shifted, (0, target_len - shifted.shape[1]))
    elif shifted.shape[1] > target_len:
        shifted = shifted[:, :target_len]
    return shifted.contiguous()


def _rubberband_shift_44k(
    wav_44k: torch.Tensor,
    semitones: float,
    *,
    formant: bool = False,
    collapse_mono: bool = False,
) -> torch.Tensor:
    return _shift_44k(
        wav_44k,
        semitones,
        formant=formant,
        shifter="rubberband",
        collapse_mono=collapse_mono,
    )


def _pad_or_trim(wav: torch.Tensor, target_samples: int) -> torch.Tensor:
    if wav.shape[1] < target_samples:
        return F.pad(wav, (0, target_samples - wav.shape[1]))
    if wav.shape[1] > target_samples:
        return wav[:, :target_samples]
    return wav


def _fold_inst_semitones(semitones: float) -> float:
    """Fold accompaniment transposition into [-6.0, +6.0] st (same musical key, natural bass/chord octave)."""
    inst = float(semitones)
    while inst < -6.0:
        inst += 12.0
    while inst > 6.0:
        inst -= 12.0
    return round(inst, 6)


def build_cover_scaffold(
    stems_dir: Path,
    *,
    start_s: float,
    duration_s: float,
    semitones: float,
    vocal_db: float = 0.0,
    harmonic_db: float = -9.5,
    bass_db: float = -13.0,
    vocal_formant: bool = False,
    inst_semitones: float | None = None,
    shifter: str = "rubberband",
    mono_vocal: bool = True,
    dereverb_vocal: bool = False,
    out_wav_32k: Path,
    out_voc_wav_32k: Path | None = None,
) -> dict[str, Any]:
    """Slice cascade stems, pitch-shift via `shifter` (`channels=together`/mono vocal), and export 32 kHz stereo."""
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    if start_s < 0:
        raise ValueError(f"start_s must be >= 0, got {start_s}")
    if shifter not in SHIFTERS:
        raise ValueError(f"Unsupported shifter {shifter!r}; expected one of {SHIFTERS}")

    stems_dir = Path(stems_dir)
    start_idx = int(round(start_s * STEM_SR))
    slice_len = int(round(duration_s * STEM_SR))

    def _load_slice(key: str) -> torch.Tensor:
        p = stems_dir / f"{key}.wav"
        wav, sr = _load_audio_native(p)
        if sr != STEM_SR:
            wav = TAF.resample(wav, orig_freq=sr, new_freq=STEM_SR)
        return _pad_or_trim(wav[:, start_idx : start_idx + slice_len], slice_len)

    inst_st = _fold_inst_semitones(semitones) if inst_semitones is None else float(inst_semitones)
    if vocal_db > -100.0:
        voc_slice = _load_slice("vocals")
        if dereverb_vocal:
            voc_slice = dereverb_vocals_roformer(voc_slice)
        voc_shifted = _shift_44k(
            voc_slice,
            semitones,
            formant=vocal_formant,
            shifter=shifter,
            collapse_mono=mono_vocal,
        )
    else:
        voc_shifted = torch.zeros((2, slice_len), dtype=torch.float32)

    if harmonic_db > -100.0:
        harm_shifted = _shift_44k(
            _load_slice("harmonic"),
            inst_st,
            formant=False,
            shifter=shifter,
            collapse_mono=False,
        )
    else:
        harm_shifted = torch.zeros((2, slice_len), dtype=torch.float32)

    if bass_db > -100.0:
        bass_shifted = _shift_44k(
            _load_slice("bass"),
            inst_st,
            formant=False,
            shifter=shifter,
            collapse_mono=False,
        )
    else:
        bass_shifted = torch.zeros((2, slice_len), dtype=torch.float32)

    mix_44k = torch.zeros((2, slice_len), dtype=torch.float32)
    if vocal_db > -100.0:
        mix_44k = mix_44k + (10.0 ** (vocal_db / 20.0)) * voc_shifted
    if harmonic_db > -100.0:
        mix_44k = mix_44k + (10.0 ** (harmonic_db / 20.0)) * harm_shifted
    if bass_db > -100.0:
        mix_44k = mix_44k + (10.0 ** (bass_db / 20.0)) * bass_shifted

    target_samples = int(round(duration_s * SCAFFOLD_SR))
    mix_32k = _pad_or_trim(
        TAF.resample(mix_44k.to(torch.float32), orig_freq=STEM_SR, new_freq=SCAFFOLD_SR),
        target_samples,
    )
    peak = float(mix_32k.abs().max().item())
    if peak > 0.92:
        mix_32k = mix_32k * (0.92 / peak)
        peak = float(mix_32k.abs().max().item())
    _write_wav_pcm16(Path(out_wav_32k), mix_32k, SCAFFOLD_SR)

    if out_voc_wav_32k is not None:
        voc_32k = _pad_or_trim(
            TAF.resample(voc_shifted.to(torch.float32), orig_freq=STEM_SR, new_freq=SCAFFOLD_SR),
            target_samples,
        )
        v_peak = float(voc_32k.abs().max().item())
        if v_peak > 0.92:
            voc_32k = voc_32k * (0.92 / v_peak)
        _write_wav_pcm16(Path(out_voc_wav_32k), voc_32k, SCAFFOLD_SR)

    return {
        "out_wav_32k": str(out_wav_32k),
        "out_voc_wav_32k": str(out_voc_wav_32k) if out_voc_wav_32k is not None else None,
        "samples": int(target_samples),
        "sample_rate": SCAFFOLD_SR,
        "duration_s": round(target_samples / SCAFFOLD_SR, 6),
        "peak": round(peak, 6),
        "semitones": float(semitones),
        "inst_semitones": float(inst_st),
        "vocal_formant": bool(vocal_formant),
        "shifter": str(shifter),
        "mono_vocal": bool(mono_vocal),
        "vocal_db": float(vocal_db),
        "harmonic_db": float(harmonic_db),
        "bass_db": float(bass_db),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="H3 null-locked stem separation and cover scaffold tool.")
    sub = ap.add_subparsers(dest="command", required=True)

    sp_stems = sub.add_parser("stems", help="Separate an audio/video file into 44.1 kHz stereo stems.")
    sp_stems.add_argument("input", type=Path, help="Input audio or video file.")
    sp_stems.add_argument("-o", "--out-dir", type=Path, required=True, help="Output directory for stems + manifest.")
    sp_stems.add_argument("-r", "--recipe", choices=sorted(RECIPES), default="cascade", help="Separation recipe.")
    sp_stems.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device.")
    sp_stems.add_argument("--start-s", type=float, default=None, help="Optional start offset in seconds.")
    sp_stems.add_argument("--duration-s", type=float, default=None, help="Optional slice duration in seconds.")

    sp_null = sub.add_parser("nulltest", help="Verify sum(stems) against reference audio.")
    sp_null.add_argument("reference", type=Path, help="Reference WAV or media file.")
    sp_null.add_argument("targets", type=Path, nargs="+", help="Stems directory or explicit stem WAV paths.")
    sp_null.add_argument("--assert-db", type=float, default=None, help="Optional failure threshold in dB.")

    sp_rec = sub.add_parser("recipes", help="List available stem separation recipes.")
    sp_rec.add_argument("--json", action="store_true", help="Emit recipes as JSON.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "recipes":
        if args.json:
            print(json.dumps(RECIPES, indent=2))
        else:
            for name, info in RECIPES.items():
                stems = ", ".join(info["stems"])
                print(f"{name:10s}  [{stems}] -- {info['description']}")
        return 0

    if args.command == "stems":
        dev = None if args.device == "auto" else torch.device(args.device)
        manifest = separate_to_dir(
            args.input,
            args.out_dir,
            recipe=args.recipe,
            device=dev,
            start_s=args.start_s,
            duration_s=args.duration_s,
        )
        print(json.dumps(manifest, indent=2))
        return 0

    if args.command == "nulltest":
        report = run_nulltest(args.reference, args.targets)
        print(json.dumps(report, indent=2))
        if args.assert_db is not None and report["null_depth_db"] > args.assert_db:
            return 1
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
