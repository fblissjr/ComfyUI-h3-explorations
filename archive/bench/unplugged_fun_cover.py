#!/usr/bin/env python3
"""Real-world Unplugged-covers-Fun (`unplugged_compressed.mp4` covering `fun.mp4`) harness.

Tests both zero-restart baselines (`u0`..`u3`) and matched-seed Phase 2
`MiniMaxH3AudioRefRouter` arms (`u4`..`u7`) at `1024x768` (exact 4:3 MTV Unplugged
aspect), `192f = 8.000 s` (exact dual-clock grid `39 + 51*3`), `seed = 840519203`:

  Zero-restart baselines (work on live :8188 before reload):
    - `u0_ceiling`             -- <Picture 1,2> + <Audio 1> (Kurt) only, no fun.mp4
    - `u1_b2_raw_rel50`        -- Raw fun.mp4 cond_audio released @ 0.50 (matched Phase 2 control)
    - `u2_b2_harm_rel50`       -- HPSS drum-stripped fun.mp4 cond_audio released @ 0.50
    - `u3_base16_raw_rel38`    -- Base ref2va (16-step euler/simple) raw cond_audio released @ 0.38

  Phase 2 `MiniMaxH3AudioRefRouter` matched-seed A/B (requires :8188 reload):
    - `u4_router_raw_s20`         -- Raw cond_audio + AudioRefRouter(split=20, late_mel=0, late_tim=2.0, rel=0.50)
    - `u5_router_raw_s24_soft70`  -- Raw cond_audio + AudioRefRouter(split=24, early_tim=0.75, late_tim=2.5, soft=0.70)
    - `u6_router_harm_s24_soft75` -- Harmonic cond_audio + AudioRefRouter(split=24, early_tim=0.75, late_tim=2.5, soft=0.75)
    - `u7_router_raw_s16_rel67`   -- Raw cond_audio + AudioRefRouter(split=16, late_tim=3.0, rel=0.67)

Usage:
  "$PY" bench/unplugged_fun_cover.py --dry-run
  "$PY" bench/unplugged_fun_cover.py --run --arms u0_ceiling u1_b2_raw_rel50 u2_b2_harm_rel50 u3_base16_raw_rel38
  "$PY" bench/unplugged_fun_cover.py --run --arms u4_router_raw_s20 u5_router_raw_s24_soft70 u6_router_harm_s24_soft75 u7_router_raw_s16_rel67
  "$PY" bench/unplugged_fun_cover.py --analyze
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))
from _lib import bootstrap, comfy_client  # noqa: E402

bootstrap(cpu=True)
sys.path = [p for p in sys.path if p != str(REPO)]
if str(REPO / "workflows") not in sys.path:
    sys.path.insert(1, str(REPO / "workflows"))

import h3_config  # noqa: E402
import preflight_graph as pf  # noqa: E402
import probe_yin_f0_drift as yin_probe  # noqa: E402

DEFAULT_URL = comfy_client.DEFAULT_COMFY_URL
OUT_PREFIX_DIR = "Video/unplugged_covers_fun_2026-11-27"
DATA_DIR = REPO / "data" / "music_cover_gate"
GRAPHS_DIR = DATA_DIR / "graphs_unplugged_fun"

PIC_FACE = "music_cover_gate/unplugged_b_face.png"
PIC_STAGE = "music_cover_gate/unplugged_b_stage.png"
AUDIO_TARGET_B = "music_cover_gate/unplugged_b_audio_192f.wav"
AUDIO_SONG_X_RAW = "music_cover_gate/fun_x_chorus_clean_192f.wav"
AUDIO_SONG_X_HARM = "music_cover_gate/fun_x_chorus_clean_harmonic_192f.wav"
PROMPT_PATH = DATA_DIR / "ref2va_unplugged_covers_fun.txt"

WIDTH = 1024
HEIGHT = 768
LENGTH = 192
SEED = 840519203

UNET_REF2VA = h3_config.MODELS["unet_ref2va"]
ENCODER_BF16 = h3_config.MODELS["clip"]
VIDEO_VAE = h3_config.MODELS["video_vae"]
AUDIO_VAE = h3_config.MODELS["audio_vae"]
LORA_PIKA_REF2VA = h3_config.PIKA_REF2VA_SVD995_LORA
PIKA_SIGMAS = h3_config.PIKA_MANUAL_SIGMAS

BASELINE_ARMS: tuple[str, ...] = (
    "u0_ceiling",
    "u1_b2_raw_rel50",
    "u2_b2_harm_rel50",
    "u3_base16_raw_rel38",
)
ROUTER_ARMS: tuple[str, ...] = (
    "u4_router_raw_s20",
    "u5_router_raw_s24_soft70",
    "u6_router_harm_s24_soft75",
    "u7_router_raw_s16_rel67",
    "u8_router_base16_harm_s24",
)
ALL_ARMS: tuple[str, ...] = BASELINE_ARMS + ROUTER_ARMS


def _require_dir(env_name: str) -> Path:
    val = os.environ.get(env_name)
    if not val:
        raise SystemExit(f"{env_name} is unset; run `set -a; . ./h3.env; set +a` first")
    p = Path(val)
    if not p.is_dir():
        raise SystemExit(f"{env_name} does not exist: {p}")
    return p


def _base_loaders() -> dict[str, Any]:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET_REF2VA, "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3EncoderLoader", "inputs": {"encoder_name": ENCODER_BF16, "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
    }


def _wire_model_chain(g: dict[str, Any], *, use_pika: bool) -> str:
    if use_pika:
        g["18"] = {
            "class_type": "MiniMaxH3LoRABranch",
            "inputs": {
                "model": ["1", 0],
                "lora_name": LORA_PIKA_REF2VA,
                "strength": 1.0,
                "modules": "all",
                "blocks": "all",
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        }
        shift_in = ["18", 0]
    else:
        shift_in = ["1", 0]
    g["19"] = {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {"model": shift_in, "shift_video": 12.0, "shift_audio": 3.0},
    }
    g["58"] = {
        "class_type": "ModelAttentionBackend",
        "inputs": {"model": ["19", 0], "attention": "comfy kitchen attention"},
    }
    g["21"] = {
        "class_type": "MiniMaxH3Sol",
        "inputs": {
            "model": ["58", 0],
            "tau": 1.0,
            "start_percent": 0.2,
            "end_percent": 1.0,
            "min_tokens": 12288,
            "sink_conditioning": "exact_kv_and_all_rows",
            "verbose": True,
            "dense_blocks": "49",
            "quantizer": "rotated",
            "token_routing": "off",
            "rows": "per segment",
            "rows.text_rows": "exact",
            "rows.reference_rows": "exact",
            "rows.audio_rows": "exact",
            "rows.video_reference_rows": "routed",
        },
    }
    return "21"


def _wire_unplugged_refs(g: dict[str, Any], *, prompt_text: str, use_pika: bool) -> tuple[str, str]:
    """Wire <Picture 1> (face) + <Picture 2> (stage) + <Audio 1> (Kurt voice/guitar)."""
    g["15"] = {"class_type": "LoadImage", "inputs": {"image": PIC_FACE}}
    g["16"] = {"class_type": "LoadImage", "inputs": {"image": PIC_STAGE}}
    pic_policy = (
        {"size_policy": "tiered", "qwen_view": "separate", "qwen_view.qwen_short_edge": 512}
        if use_pika
        else {
            "size_policy": "max",
            "size_policy.dit_short_edge": 2048,
            "size_policy.allow_upscale": False,
            "qwen_view": "separate",
            "qwen_view.qwen_short_edge": 512,
        }
    )
    g["50"] = {"class_type": "MiniMaxH3AppendRefImage", "inputs": {"image": ["15", 0], **pic_policy}}
    g["51"] = {
        "class_type": "MiniMaxH3AppendRefImage",
        "inputs": {"image": ["16", 0], "references": ["50", 0], **pic_policy},
    }
    g["33"] = {"class_type": "LoadAudio", "inputs": {"audio": AUDIO_TARGET_B}}
    g["52"] = {
        "class_type": "MiniMaxH3AppendRefAudio",
        "inputs": {"audio": ["33", 0], "references": ["51", 0]},
    }
    g["132"] = {
        "class_type": "MiniMaxH3EncodeReferences",
        "inputs": {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "audio_vae": ["4", 0],
            "references": ["52", 0],
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "video_policy": "comfy",
            "image_policy": "comfy",
        },
    }
    g["5"] = {
        "class_type": "MiniMaxH3PromptOnReferences",
        "inputs": {"encoded_references": ["132", 0], "prompt": prompt_text},
    }
    return "5", "132"


def _wire_sampler_and_decode(g: dict[str, Any], *, model_id: str, preflight_id: str, use_pika: bool, arm: str) -> None:
    if use_pika:
        g["7"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}}
        g["60"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": PIKA_SIGMAS}}
    else:
        g["7"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        g["60"] = {
            "class_type": "BasicScheduler",
            "inputs": {"model": [model_id, 0], "scheduler": "simple", "steps": 16, "denoise": 1.0},
        }
    g["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}}
    g["9"] = {"class_type": "BasicGuider", "inputs": {"model": [model_id, 0], "conditioning": [preflight_id, 0]}}
    g["10"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["6", 0],
            "guider": ["9", 0],
            "sampler": ["7", 0],
            "sigmas": ["60", 0],
            "latent_image": [preflight_id, 1],
        },
    }
    g["30"] = {
        "class_type": "MiniMaxH3ProvenanceStamp",
        "inputs": {"latent": ["10", 0], "model": [model_id, 0], "sigmas": ["60", 0], "clip": ["2", 0], "note": arm},
    }
    g["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}}
    g["12"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}}
    g["13"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["11", 0],
            "audio": ["12", 0],
            "frame_rate": 24.0,
            "loop_count": 0,
            "filename_prefix": f"{OUT_PREFIX_DIR}/{arm}",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
        },
    }


def build_arm_graph(arm: str) -> dict[str, Any]:
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8").strip()
    use_pika = arm not in ("u3_base16_raw_rel38", "u8_router_base16_harm_s24")

    g = _base_loaders()
    sol_id = _wire_model_chain(g, use_pika=use_pika)
    cond_id, enc_id = _wire_unplugged_refs(g, prompt_text=prompt_text, use_pika=use_pika)

    if arm == "u0_ceiling":
        g["26"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": [cond_id, 0], "samples": [enc_id, 1]}}
        _wire_sampler_and_decode(g, model_id=sol_id, preflight_id="26", use_pika=True, arm=arm)
        return g

    # All other arms inject Song X melody scaffold into cond_audio via MiniMaxH3GuideLatent
    use_harm = arm in ("u2_b2_harm_rel50", "u6_router_harm_s24_soft75", "u8_router_base16_harm_s24")
    g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": AUDIO_SONG_X_HARM if use_harm else AUDIO_SONG_X_RAW}}
    g["151"] = {
        "class_type": "MiniMaxH3GuideLatent",
        "inputs": {
            "positive": [cond_id, 0],
            "latent": [enc_id, 1],
            "frame_idx": 0,
            "single_frame": False,
            "audio": ["48", 0],
            "audio_vae": ["4", 0],
        },
    }
    g["26"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": ["151", 0], "samples": ["151", 1]}}

    if arm in ("u1_b2_raw_rel50", "u2_b2_harm_rel50", "u3_base16_raw_rel38"):
        aud_end = 0.38 if arm == "u3_base16_raw_rel38" else 0.50
        g["150"] = {
            "class_type": "MiniMaxH3ReleaseReferences",
            "inputs": {
                "model": [sol_id, 0],
                "end_percent": 1.0,
                "clean_level": 0.999,
                "audio_end_percent": float(aud_end),
                "audio_clean_level": 1.0,
                "video_end_percent": 1.0,
                "guide_end_percent": 1.0,
            },
        }
        _wire_sampler_and_decode(g, model_id="150", preflight_id="26", use_pika=use_pika, arm=arm)
        return g

    router_specs: dict[str, dict[str, Any]] = {
        "u4_router_raw_s20": {
            "melody_source": "cond_audio",
            "split_block": 20,
            "melody_gain_early": 1.0,
            "melody_gain_late": 0.0,
            "timbre_gain_early": 1.0,
            "timbre_gain_late": 2.0,
            "melody_end_percent": 0.50,
            "melody_clean_level": 1.0,
            "start_percent": 0.0,
            "end_percent": 1.0,
        },
        "u5_router_raw_s24_soft70": {
            "melody_source": "cond_audio",
            "split_block": 24,
            "melody_gain_early": 1.0,
            "melody_gain_late": 0.0,
            "timbre_gain_early": 0.75,
            "timbre_gain_late": 2.5,
            "melody_end_percent": 0.50,
            "melody_clean_level": 0.70,
            "start_percent": 0.0,
            "end_percent": 1.0,
        },
        "u6_router_harm_s24_soft75": {
            "melody_source": "cond_audio",
            "split_block": 24,
            "melody_gain_early": 1.0,
            "melody_gain_late": 0.0,
            "timbre_gain_early": 0.75,
            "timbre_gain_late": 2.5,
            "melody_end_percent": 0.50,
            "melody_clean_level": 0.75,
            "start_percent": 0.0,
            "end_percent": 1.0,
        },
        "u7_router_raw_s16_rel67": {
            "melody_source": "cond_audio",
            "split_block": 16,
            "melody_gain_early": 1.0,
            "melody_gain_late": 0.0,
            "timbre_gain_early": 0.75,
            "timbre_gain_late": 3.0,
            "melody_end_percent": 0.67,
            "melody_clean_level": 0.80,
            "start_percent": 0.0,
            "end_percent": 1.0,
        },
        "u8_router_base16_harm_s24": {
            "melody_source": "cond_audio",
            "split_block": 24,
            "melody_gain_early": 1.0,
            "melody_gain_late": 0.0,
            "timbre_gain_early": 0.75,
            "timbre_gain_late": 2.5,
            "melody_end_percent": 0.44,
            "melody_clean_level": 0.80,
            "start_percent": 0.0,
            "end_percent": 1.0,
        },
    }
    if arm in router_specs:
        g["150"] = {
            "class_type": "MiniMaxH3AudioRefRouter",
            "inputs": {"model": [sol_id, 0], **router_specs[arm]},
        }
        _wire_sampler_and_decode(g, model_id="150", preflight_id="26", use_pika=use_pika, arm=arm)
        return g

    raise ValueError(f"Unknown arm: {arm}")


def _preflight_graph(graph_path: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(REPO / "bench" / "preflight_graph.py"), str(graph_path)],
        capture_output=True,
        text=True,
    )
    return proc.returncode, (proc.stdout + "\n" + proc.stderr).strip()


def _comfy_server_pid() -> int | None:
    try:
        out = subprocess.check_output(["pgrep", "-f", r"main\.py.*--port[= ]8188"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if line.strip().isdigit():
                return int(line.strip())
    except Exception:
        pass
    return None


def _external_gpu_hogs(comfy_pid: int | None, *, min_mib: int = 1000) -> list[tuple[int, int]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    hogs: list[tuple[int, int]] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid, mib = int(parts[0]), int(float(parts[1]))
        except ValueError:
            continue
        if comfy_pid is not None and pid == comfy_pid:
            continue
        if mib >= min_mib:
            hogs.append((pid, mib))
    return hogs


def _wait_until_queue_idle(url: str, *, poll_s: float = 10.0, max_wait_s: float = 3600.0) -> None:
    t0 = time.monotonic()
    logged = False
    comfy_pid = _comfy_server_pid()
    while True:
        data = json.loads(urllib.request.urlopen(f"{url}/queue", timeout=5).read().decode("utf-8"))
        running, pending = len(data.get("queue_running", [])), len(data.get("queue_pending", []))
        hogs = _external_gpu_hogs(comfy_pid)
        if running == 0 and pending == 0 and not hogs:
            if logged:
                print(f"[gpu-wait] GPU & :8188 idle after {time.monotonic() - t0:.1f}s", flush=True)
            return
        if not logged:
            print(f"[gpu-wait] waiting (queue={running}/{pending}, external_gpu_pids={hogs})...", flush=True)
            logged = True
        if time.monotonic() - t0 > max_wait_s:
            raise TimeoutError(f"GPU/:8188 busy for > {max_wait_s}s")
        time.sleep(poll_s)


def _extract_saved_videos(history_entry: dict[str, Any], output_dir: Path) -> list[Path]:
    found: list[Path] = []
    for node_out in (history_entry.get("outputs") or {}).values():
        if not isinstance(node_out, dict):
            continue
        for key in ("gifs", "videos"):
            for item in node_out.get(key) or []:
                if isinstance(item, dict) and item.get("filename"):
                    p = output_dir / (item.get("subfolder") or "") / item["filename"]
                    if p.is_file():
                        found.append(p)
    return found


def _rms_envelope(wav: torch.Tensor, sr: int = 24000, hop_ms: float = 25.0) -> np.ndarray:
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    x = wav.detach().cpu().numpy().astype(np.float64)
    n_frames = len(x) // hop
    if n_frames < 2:
        return np.zeros(1, dtype=np.float64)
    return np.sqrt(np.mean(x[: n_frames * hop].reshape(n_frames, hop) ** 2, axis=1) + 1e-12)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    aa = a[:n] - float(np.mean(a[:n]))
    bb = b[:n] - float(np.mean(b[:n]))
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / den) if den >= 1e-12 else float("nan")


def _logmel(wav: torch.Tensor, sr: int = 24000) -> torch.Tensor:
    return torch.log(torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=256, n_mels=64)(wav) + 1e-6)


def _ltas_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    va = _logmel(a).mean(dim=-1)
    vb = _logmel(b).mean(dim=-1)
    va = va - va.mean()
    vb = vb - vb.mean()
    return float(torch.dot(va, vb).item() / (float((va.norm() * vb.norm()).item()) + 1e-12))


def _frame_logmel_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    A, B = _logmel(a), _logmel(b)
    n = min(A.shape[-1], B.shape[-1])
    A, B = A[..., :n], B[..., :n]
    A = A - A.mean(dim=0, keepdim=True)
    B = B - B.mean(dim=0, keepdim=True)
    return float(((A * B).sum(dim=0) / (A.norm(dim=0) * B.norm(dim=0) + 1e-12)).mean().item())


def _hf_share(wav: torch.Tensor, sr: int = 24000, cutoff_hz: float = 4000.0) -> float:
    x = wav.detach().cpu().numpy().astype(np.float64)
    if len(x) < 1024:
        return 0.0
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    return float(np.sum(spec[freqs >= cutoff_hz]) / (float(np.sum(spec)) + 1e-12))


def _f0_contour_corr(f0_a: torch.Tensor, f0_b: torch.Tensor) -> tuple[float, int]:
    n = min(int(f0_a.numel()), int(f0_b.numel()))
    a, b = f0_a[:n].numpy(), f0_b[:n].numpy()
    both = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    k = int(np.sum(both))
    if k < 8:
        return float("nan"), k
    return _pearson(np.log2(a[both]), np.log2(b[both])), k


def analyze_renders(input_dir: Path, output_dir: Path, arm_files: dict[str, Path]) -> dict[str, Any]:
    ref_b = yin_probe.load_mono_audio(input_dir / AUDIO_TARGET_B, sr=24000)
    ref_x = yin_probe.load_mono_audio(input_dir / AUDIO_SONG_X_RAW, sr=24000)
    ref_xh = yin_probe.load_mono_audio(input_dir / AUDIO_SONG_X_HARM, sr=24000)

    f0_b, _, _ = yin_probe.yin_f0(ref_b)
    f0_x, _, _ = yin_probe.yin_f0(ref_x)
    f0_xh, _, _ = yin_probe.yin_f0(ref_xh)
    env_b = _rms_envelope(ref_b)
    env_x = _rms_envelope(ref_x)
    env_xh = _rms_envelope(ref_xh)

    med_b = float(np.nanmedian(f0_b.numpy()))
    med_x = float(np.nanmedian(f0_x.numpy()))

    summary: dict[str, Any] = {
        "references": {
            "unplugged_b_median_f0_hz": round(med_b, 2),
            "unplugged_b_hf4k_share": round(_hf_share(ref_b), 4),
            "fun_x_raw_median_f0_hz": round(med_x, 2),
            "fun_x_raw_hf4k_share": round(_hf_share(ref_x), 4),
            "fun_x_harm_hf4k_share": round(_hf_share(ref_xh), 4),
        },
        "arms": {},
    }
    for name, mp4 in arm_files.items():
        if not mp4.is_file():
            continue
        wav = yin_probe.load_mono_audio(mp4, sr=24000)
        rms = float(torch.sqrt(torch.mean(wav ** 2) + 1e-12).item())
        f0, _, _ = yin_probe.yin_f0(wav)
        f0_np = f0.numpy()
        valid = np.isfinite(f0_np) & (f0_np > 0)
        med_f0 = float(np.median(f0_np[valid])) if np.any(valid) else float("nan")
        st_b = 12.0 * math.log2(med_f0 / med_b) if (math.isfinite(med_f0) and med_b > 0) else float("nan")
        st_x = 12.0 * math.log2(med_f0 / med_x) if (math.isfinite(med_f0) and med_x > 0) else float("nan")
        env = _rms_envelope(wav)
        cf_x, nf_x = _f0_contour_corr(f0, f0_x)
        cf_xh, nf_xh = _f0_contour_corr(f0, f0_xh)
        cf_b, _ = _f0_contour_corr(f0, f0_b)
        summary["arms"][name] = {
            "file": str(mp4.relative_to(output_dir)),
            "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-9)), 2),
            "hf4k_share": round(_hf_share(wav), 4),
            "voiced_frac": round(float(np.mean(valid)), 3),
            "median_f0_hz": round(med_f0, 2) if math.isfinite(med_f0) else None,
            "st_vs_unplugged_b": round(st_b, 2) if math.isfinite(st_b) else None,
            "st_vs_fun_x": round(st_x, 2) if math.isfinite(st_x) else None,
            "f0_contour_corr_fun_raw": round(cf_x, 3) if math.isfinite(cf_x) else None,
            "f0_contour_n_fun_raw": nf_x,
            "f0_contour_corr_fun_harm": round(cf_xh, 3) if math.isfinite(cf_xh) else None,
            "f0_contour_n_fun_harm": nf_xh,
            "f0_contour_corr_unplugged_b": round(cf_b, 3) if math.isfinite(cf_b) else None,
            "env_corr_fun_raw": round(_pearson(env, env_x), 3),
            "env_corr_fun_harm": round(_pearson(env, env_xh), 3),
            "env_corr_unplugged_b": round(_pearson(env, env_b), 3),
            "ltas_cos_unplugged_b": round(_ltas_cosine(wav, ref_b), 3),
            "ltas_cos_fun_raw": round(_ltas_cosine(wav, ref_x), 3),
            "frame_mel_corr_fun_raw": round(_frame_logmel_corr(wav, ref_x), 3),
            "frame_mel_corr_unplugged_b": round(_frame_logmel_corr(wav, ref_b), 3),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Unplugged-covers-Fun real-world silicon harness")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(ALL_ARMS))
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args(argv)
    if not (args.dry_run or args.run or args.analyze):
        ap.error("Specify one of --dry-run, --run, or --analyze")

    input_dir = _require_dir("H3_INPUT_DIR")
    output_dir = _require_dir("H3_OUTPUT_DIR")
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    for rel in (PIC_FACE, PIC_STAGE, AUDIO_TARGET_B, AUDIO_SONG_X_RAW, AUDIO_SONG_X_HARM):
        if not (input_dir / rel).is_file():
            raise SystemExit(f"Missing required input asset under $H3_INPUT_DIR: {rel}")

    graphs: dict[str, tuple[Path, dict[str, Any]]] = {}
    if args.dry_run or args.run:
        any_fail = False
        for arm in args.arms:
            g = build_arm_graph(arm)
            gp = GRAPHS_DIR / f"{arm}.json"
            gp.write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")
            rc, out = _preflight_graph(gp)
            print(f"[preflight] {arm}: {'PASS' if rc == 0 else f'FAIL(rc={rc})'}")
            for line in out.splitlines():
                if any(k in line for k in ("FAIL", "WARN", "tokens", "total", "grid")):
                    print("   ", line)
            if rc != 0:
                any_fail = True
            graphs[arm] = (gp, g)
        if any_fail:
            return 1
        if args.dry_run and not args.run:
            return 0

    arm_files: dict[str, Path] = {}
    timings: dict[str, float] = {}
    if args.run:
        client_id = str(uuid.uuid4())
        for arm in args.arms:
            _wait_until_queue_idle(args.url)
            _, g = graphs[arm]
            print(f"\n[run] queuing {arm}...", flush=True)
            t0 = time.monotonic()
            prompt_id = comfy_client.submit_prompt(g, base_url=args.url, client_id=client_id)
            hist = comfy_client.wait_for_prompt(prompt_id, base_url=args.url, timeout=1800.0)
            dt = comfy_client.execution_duration_s(hist) or (time.monotonic() - t0)
            timings[arm] = round(dt, 2)
            saved = _extract_saved_videos(hist, output_dir)
            print(f"[run] {arm} finished in {dt:.1f}s -> {[p.name for p in saved]}", flush=True)
            if saved:
                arm_files[arm] = saved[-1]

    if args.analyze or args.run:
        vid_dir = output_dir / OUT_PREFIX_DIR
        if vid_dir.is_dir():
            for arm in ALL_ARMS:
                if arm not in arm_files:
                    matches = sorted(vid_dir.glob(f"{arm}_*-audio.mp4"))
                    if matches:
                        arm_files[arm] = matches[-1]
        report = analyze_renders(input_dir, output_dir, arm_files)
        if timings:
            report["timings_s"] = timings
        report_path = DATA_DIR / "unplugged_fun_triage_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\n[triage] wrote {report_path.relative_to(REPO)}")
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
