#!/usr/bin/env python3
"""6-Arm Pitch-Locked H3 Music-Video Cover Gate (MTV Unplugged covering AI songs).

Tests the three isolated fixes for `ai_song_sdedit_covers`'s pitch & attribution bugs:
  1. De-droned `<Audio 1>` (`kurt_vocal_only_192f.wav`) vs capo key-matched full vocal+guitar (`kurt_vocal_guitar_192f.wav`).
  2. Clean `MelBand-RoFormer` vocal stems + minimal `RubberBand` pitch matching + optional `-11 dB` HPSS harmonic chord root bed.
  3. Solo-Kurt medium & close-up `<Picture>` framing + Pass-1 `MiniMaxH3QueryRouter(section_split="soundscape_to_audio")`
     to eliminate `[Shot 1]` background-singer lip-sync theft, plus Pass-2 `MiniMaxH3EditGuider(streams="audio_only")`
     contrastive velocity steering.

All 6 arms run at `1024x768`, `192f = 8.000 s` (`on_both_clocks`), `seed = 910427311`:
  - `p1_flicker_chord_d36`          : `justaflicker` (`flicker_chord_scaf`, natural C#/G# key) + `kurt_vocal_guitar` + Base-12 `denoise=0.36`
  - `p2_flicker_editguider_d40`     : Cached Pass 1 (`p1`) + Pass-2 `MiniMaxH3EditGuider(streams="audio_only", scale=1.75)` `denoise=0.40`
  - `p3_flicker_voconly_d40`        : `justaflicker` (`flicker_voc_oct` ONLY) + de-droned `kurt_vocal_only` + Base-12 `denoise=0.40`
  - `p4_whatever_dedroned_origkey_d38`: `whatevercomes` (`whatever_origkey_scaf`, F#) + de-droned `kurt_vocal_only` + Base-12 `denoise=0.38`
  - `p5_whatever_capo_csharp_d38`   : `whatevercomes` (`whatever_capo_csharp_scaf`, -5 st -> C#) + `kurt_vocal_guitar` + Base-12 `denoise=0.38`
  - `p6_fridge_gfunk_unplugged_d40` : `In The Fridge` (`fridge_scaf`, 0 st male) + de-droned `kurt_vocal_only` + Base-12 `denoise=0.40`
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
COMFY = REPO.parents[1]
sys.path.insert(0, str(REPO / "bench"))
from _lib import bootstrap, comfy_client  # noqa: E402

bootstrap(cpu=True)
sys.path = [p for p in sys.path if p != str(REPO)]
if str(REPO / "workflows") not in sys.path:
    sys.path.insert(1, str(REPO / "workflows"))

import h3_config  # noqa: E402
import prepare_pitch_lock_assets as prep_mod  # noqa: E402
import probe_yin_f0_drift as yin_probe  # noqa: E402

DEFAULT_URL = comfy_client.DEFAULT_COMFY_URL
OUT_PREFIX_DIR = "Video/music_cover_pitch_lock_2026-11-27"
DATA_DIR = REPO / "data" / "music_cover_pitch_lock"
GRAPHS_DIR = DATA_DIR / "graphs"

PIC_FACE = "music_cover_pitch_lock/kurt_closeup.png"
PIC_STAGE = "music_cover_pitch_lock/kurt_solo_medium.png"

WIDTH = 1024
HEIGHT = 768
LENGTH = 192
SEED = 910427311

UNET_REF2VA = h3_config.MODELS["unet_ref2va"]
ENCODER_BF16 = h3_config.MODELS["clip"]
VIDEO_VAE = h3_config.MODELS["video_vae"]
AUDIO_VAE = h3_config.MODELS["audio_vae"]
LORA_PIKA_REF2VA = h3_config.PIKA_REF2VA_SVD995_LORA
PIKA_SIGMAS = h3_config.PIKA_MANUAL_SIGMAS

ARM_SPECS: dict[str, dict[str, Any]] = {
    "p1_flicker_chord_d36": {
        "ref_audio": "music_cover_pitch_lock/kurt_vocal_guitar_192f.wav",
        "song_x": "music_cover_pitch_lock/flicker_chord_scaf_192f.wav",
        "src_mix": "music_cover_pitch_lock/flicker_raw_192f.wav",
        "src_voc": "music_cover_pitch_lock/flicker_voc_orig_192f.wav",
        "target_tr_st": 0,
        "prompt": DATA_DIR / "ref2va_unplugged_flicker.txt",
        "neutral_prompt": None,
        "denoise": 0.36,
    },
    "p2_flicker_editguider_d40": {
        "ref_audio": "music_cover_pitch_lock/kurt_vocal_guitar_192f.wav",
        "song_x": "music_cover_pitch_lock/flicker_chord_scaf_192f.wav",
        "src_mix": "music_cover_pitch_lock/flicker_raw_192f.wav",
        "src_voc": "music_cover_pitch_lock/flicker_voc_orig_192f.wav",
        "target_tr_st": 0,
        "prompt": DATA_DIR / "ref2va_unplugged_flicker.txt",
        "neutral_prompt": DATA_DIR / "ref2va_unplugged_flicker_neutral.txt",
        "denoise": 0.40,
        "edit_scale": 1.75,
        "edit_last_step": 8,
    },
    "p3_flicker_voconly_d40": {
        "ref_audio": "music_cover_pitch_lock/kurt_vocal_only_192f.wav",
        "song_x": "music_cover_pitch_lock/flicker_voc_oct_192f.wav",
        "src_mix": "music_cover_pitch_lock/flicker_raw_192f.wav",
        "src_voc": "music_cover_pitch_lock/flicker_voc_orig_192f.wav",
        "target_tr_st": 0,
        "prompt": DATA_DIR / "ref2va_unplugged_flicker.txt",
        "neutral_prompt": None,
        "denoise": 0.40,
    },
    "p4_whatever_dedroned_origkey_d38": {
        "ref_audio": "music_cover_pitch_lock/kurt_vocal_only_192f.wav",
        "song_x": "music_cover_pitch_lock/whatever_origkey_scaf_192f.wav",
        "src_mix": "music_cover_pitch_lock/whatever_raw_192f.wav",
        "src_voc": "music_cover_pitch_lock/whatever_voc_orig_192f.wav",
        "target_tr_st": 0,
        "prompt": DATA_DIR / "ref2va_unplugged_whatever_192f.txt",
        "neutral_prompt": None,
        "denoise": 0.38,
    },
    "p5_whatever_capo_csharp_d38": {
        "ref_audio": "music_cover_pitch_lock/kurt_vocal_guitar_192f.wav",
        "song_x": "music_cover_pitch_lock/whatever_capo_csharp_scaf_192f.wav",
        "src_mix": "music_cover_pitch_lock/whatever_raw_192f.wav",
        "src_voc": "music_cover_pitch_lock/whatever_voc_capo_csharp_192f.wav",
        "target_tr_st": -5,
        "prompt": DATA_DIR / "ref2va_unplugged_whatever_192f.txt",
        "neutral_prompt": None,
        "denoise": 0.38,
    },
    "p6_fridge_gfunk_unplugged_d40": {
        "ref_audio": "music_cover_pitch_lock/kurt_vocal_only_192f.wav",
        "song_x": "music_cover_pitch_lock/fridge_scaf_192f.wav",
        "src_mix": "music_cover_pitch_lock/fridge_raw_192f.wav",
        "src_voc": "music_cover_pitch_lock/fridge_voc_orig_192f.wav",
        "target_tr_st": 0,
        "prompt": DATA_DIR / "ref2va_unplugged_fridge.txt",
        "neutral_prompt": None,
        "denoise": 0.40,
    },
}
ALL_ARMS: tuple[str, ...] = tuple(ARM_SPECS.keys())


def _require_dir(env_name: str) -> Path:
    val = os.environ.get(env_name)
    if not val:
        raise SystemExit(f"{env_name} is unset; run `set -a; . ./h3.env; set +a` first")
    p = Path(val)
    if not p.is_dir():
        raise SystemExit(f"{env_name} does not exist: {p}")
    return p


def _wire_sol(g: dict[str, Any], *, in_id: str, shift_id: str, attn_id: str, sol_id: str) -> str:
    g[shift_id] = {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {"model": [in_id, 0], "shift_video": 12.0, "shift_audio": 3.0},
    }
    g[attn_id] = {
        "class_type": "ModelAttentionBackend",
        "inputs": {"model": [shift_id, 0], "attention": "comfy kitchen attention"},
    }
    g[sol_id] = {
        "class_type": "MiniMaxH3Sol",
        "inputs": {
            "model": [attn_id, 0],
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
    return sol_id


def build_arm_graph(arm: str) -> dict[str, Any]:
    spec = ARM_SPECS[arm]
    prompt_text = spec["prompt"].read_text(encoding="utf-8").strip()
    g: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET_REF2VA, "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3EncoderLoader", "inputs": {"encoder_name": ENCODER_BF16, "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "18": {
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
        },
    }
    pika_sol = _wire_sol(g, in_id="18", shift_id="19", attn_id="58", sol_id="21")
    base_sol = _wire_sol(g, in_id="1", shift_id="119", attn_id="158", sol_id="121")

    # References: <Picture 1> closeup + <Picture 2> solo medium stage + <Audio 1> Kurt ref
    g["15"] = {"class_type": "LoadImage", "inputs": {"image": PIC_FACE}}
    g["16"] = {"class_type": "LoadImage", "inputs": {"image": PIC_STAGE}}
    pic_policy = {"size_policy": "tiered", "qwen_view": "separate", "qwen_view.qwen_short_edge": 512}
    g["50"] = {"class_type": "MiniMaxH3AppendRefImage", "inputs": {"image": ["15", 0], **pic_policy}}
    g["51"] = {"class_type": "MiniMaxH3AppendRefImage", "inputs": {"image": ["16", 0], "references": ["50", 0], **pic_policy}}
    g["33"] = {"class_type": "LoadAudio", "inputs": {"audio": spec["ref_audio"]}}
    g["52"] = {"class_type": "MiniMaxH3AppendRefAudio", "inputs": {"audio": ["33", 0], "references": ["51", 0]}}
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
    g["5"] = {"class_type": "MiniMaxH3PromptOnReferences", "inputs": {"encoded_references": ["132", 0], "prompt": prompt_text}}
    g["26"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": ["5", 0], "samples": ["132", 1]}}

    # Pass 1: Pika 6-step + FreezeAudio(scaffold) + QueryRouter(soundscape_to_audio)
    # Identical node IDs/inputs between p1 and p2 so ComfyUI caches Pass 1!
    g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": spec["song_x"]}}
    g["49"] = {
        "class_type": "MiniMaxH3FreezeAudio",
        "inputs": {
            "latent": ["26", 1],
            "audio_vae": ["4", 0],
            "audio": ["48", 0],
            "start_seconds": 0.0,
            "audio_mask": 0.0,
            "level": "clip_guard",
        },
    }
    g["45"] = {
        "class_type": "MiniMaxH3QueryRouter",
        "inputs": {
            "model": [pika_sol, 0],
            "section_split": "soundscape_to_audio",
            "video_block_band": "all",
            "audio_block_band": "all",
            "start_percent": 0.0,
            "end_percent": 1.0,
        },
    }
    g["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}}
    g["7"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}}
    g["60"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": PIKA_SIGMAS}}
    g["9"] = {"class_type": "BasicGuider", "inputs": {"model": ["45", 0], "conditioning": ["26", 0]}}
    g["10"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["6", 0],
            "guider": ["9", 0],
            "sampler": ["7", 0],
            "sigmas": ["60", 0],
            "latent_image": ["49", 0],
        },
    }

    # Pass 2: Base ref2va 12-step euler/simple + AudioRefineMask(0, 1)
    g["152"] = {"class_type": "MiniMaxH3AudioRefineMask", "inputs": {"latent": ["10", 0], "video_mask": 0.0, "audio_mask": 1.0}}
    g["106"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED + 17}}
    g["107"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    g["160"] = {
        "class_type": "BasicScheduler",
        "inputs": {"model": [base_sol, 0], "scheduler": "simple", "steps": 12, "denoise": float(spec["denoise"])},
    }
    if spec.get("neutral_prompt") is not None:
        neutral_text = spec["neutral_prompt"].read_text(encoding="utf-8").strip()
        g["105"] = {"class_type": "MiniMaxH3PromptOnReferences", "inputs": {"encoded_references": ["132", 0], "prompt": neutral_text}}
        g["109"] = {
            "class_type": "MiniMaxH3EditGuider",
            "inputs": {
                "model": [base_sol, 0],
                "original": ["105", 0],
                "edited": ["26", 0],
                "scale": float(spec["edit_scale"]),
                "first_step": 0,
                "align_positions": False,
                "last_step": int(spec["edit_last_step"]),
                "streams": "audio_only",
                "audio_scale": float(spec["edit_scale"]),
            },
        }
    else:
        g["109"] = {"class_type": "BasicGuider", "inputs": {"model": [base_sol, 0], "conditioning": ["26", 0]}}

    g["110"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["106", 0],
            "guider": ["109", 0],
            "sampler": ["107", 0],
            "sigmas": ["160", 0],
            "latent_image": ["152", 0],
        },
    }
    g["30"] = {
        "class_type": "MiniMaxH3ProvenanceStamp",
        "inputs": {"latent": ["110", 0], "model": [base_sol, 0], "sigmas": ["160", 0], "clip": ["2", 0], "note": arm},
    }
    g["111"] = {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}}
    g["112"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}}
    g["113"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["111", 0],
            "audio": ["112", 0],
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
    return g


def _preflight(gp: Path) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(REPO / "bench" / "preflight_graph.py"), str(gp)], capture_output=True, text=True)
    return p.returncode, (p.stdout + "\n" + p.stderr).strip()


def _comfy_pid() -> int | None:
    try:
        for line in subprocess.check_output(["pgrep", "-f", r"main\.py.*--port[= ]8188"], text=True).splitlines():
            if line.strip().isdigit():
                return int(line.strip())
    except Exception:
        pass
    return None


def _wait_idle(url: str) -> None:
    pid = _comfy_pid()
    while True:
        q = json.loads(urllib.request.urlopen(f"{url}/queue", timeout=5).read().decode("utf-8"))
        hogs = []
        try:
            for line in subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True
            ).splitlines():
                p_s, m_s = [x.strip() for x in line.split(",")[:2]]
                if int(float(m_s)) >= 1000 and (pid is None or int(p_s) != pid):
                    hogs.append((int(p_s), int(float(m_s))))
        except Exception:
            pass
        if not q.get("queue_running") and not q.get("queue_pending") and not hogs:
            return
        time.sleep(10.0)


def _extract_videos(hist: dict[str, Any], out_dir: Path) -> list[Path]:
    res: list[Path] = []
    for no in (hist.get("outputs") or {}).values():
        if not isinstance(no, dict):
            continue
        for key in ("gifs", "videos"):
            for it in no.get(key) or []:
                if isinstance(it, dict) and it.get("filename"):
                    p = out_dir / (it.get("subfolder") or "") / it["filename"]
                    if p.is_file():
                        res.append(p)
    return res


def _load_mono_np(p: Path, sr: int = 24000) -> np.ndarray:
    return np.frombuffer(
        subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(p), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]),
        dtype=np.float32,
    ).copy()


def _load_stereo_44k(p: Path) -> torch.Tensor:
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(p), "-vn", "-ac", "2", "-ar", "44100", "-f", "f32le", "-"]
    )
    arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(-1, 2).T
    return torch.from_numpy(arr)


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


def _chroma_sim(a: np.ndarray, b: np.ndarray, any_key: bool = False, base_tr_st: int = 0) -> tuple[float, int, int]:
    n = min(a.shape[1], b.shape[1])
    best, blag, btr = -1.0, 0, 0
    tr_range = range(-6, 7) if any_key else (0,)
    for tr in tr_range:
        bt = np.roll(b, base_tr_st + tr, axis=0) if (base_tr_st + tr) else b
        for lag in range(-12, 13):
            aa = a[:, lag:n] if lag >= 0 else a[:, :n + lag]
            bb = bt[:, :n - lag] if lag >= 0 else bt[:, -lag:n]
            if aa.shape[1] < 20:
                continue
            s = float(np.mean(np.sum(aa * bb, axis=0)))
            if s > best:
                best, blag, btr = s, lag, tr
    return best, blag, btr


def _onset_corr(a_y: np.ndarray, b_y: np.ndarray) -> tuple[float, int]:
    def flux(y: np.ndarray) -> np.ndarray:
        S = np.log1p(torch.stft(torch.from_numpy(y), n_fft=1024, hop_length=600, window=torch.hann_window(1024), return_complex=True).abs().numpy())
        return np.maximum(0.0, np.diff(S, axis=1)).mean(axis=0)

    fa, fb = flux(a_y), flux(b_y)
    n = min(len(fa), len(fb))
    best, blag = -1.0, 0
    for lag in range(-12, 13):
        aa = fa[lag:n] if lag >= 0 else fa[:n + lag]
        bb = fb[:n - lag] if lag >= 0 else fb[-lag:n]
        aa, bb = aa - aa.mean(), bb - bb.mean()
        r = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))
        if r > best:
            best, blag = r, lag
    return best, blag


def _ltas_cos(a_y: np.ndarray, b_y: np.ndarray) -> float:
    m = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=64)
    va = torch.log(m(torch.from_numpy(a_y)) + 1e-6).mean(dim=-1)
    vb = torch.log(m(torch.from_numpy(b_y)) + 1e-6).mean(dim=-1)
    va, vb = va - va.mean(), vb - vb.mean()
    return float(torch.dot(va, vb).item() / (float((va.norm() * vb.norm()).item()) + 1e-12))


def _vocal_pitch_metrics(out_voc_24k: torch.Tensor, tgt_voc_24k: torch.Tensor) -> dict[str, Any]:
    f0_o, _, _ = yin_probe.yin_f0(out_voc_24k, sr=24000)
    f0_t, _, _ = yin_probe.yin_f0(tgt_voc_24k, sr=24000)
    n = min(int(f0_o.numel()), int(f0_t.numel()))
    fo = f0_o[:n].numpy()
    ft = f0_t[:n].numpy()
    valid = np.isfinite(fo) & np.isfinite(ft) & (fo > 0) & (ft > 0)
    n_mut = int(np.sum(valid))
    out_valid = fo[np.isfinite(fo) & (fo > 0)]
    med_voc = float(np.median(out_valid)) if len(out_valid) else float("nan")
    if n_mut < 12:
        return {
            "vocal_med_f0_hz": round(med_voc, 2) if math.isfinite(med_voc) else None,
            "mut_voiced_frames": n_mut,
            "med_abs_cent_err": None,
            "med_signed_cent_bias": None,
            "in_tune_frac_50c": None,
            "best_tr_st": None,
            "rel_med_abs_cent_err": None,
            "rel_in_tune_frac_50c": None,
            "vocal_f0_r": None,
        }
    # Octave-folded cent error in [-600, +600]
    raw_cents = 1200.0 * np.log2(fo[valid] / ft[valid])
    folded = ((raw_cents + 600.0) % 1200.0) - 600.0
    med_bias = float(np.median(folded))
    abs_err = np.abs(folded)
    med_abs = float(np.median(abs_err))
    in50 = float(np.mean(abs_err <= 50.0))

    # Best integer semitone transposition (separating global register/key shift from interval fidelity)
    best_st, best_rel_med, best_rel_in50 = 0, 999.0, 0.0
    for k in range(-6, 7):
        d_k = ((raw_cents - 100.0 * k + 600.0) % 1200.0) - 600.0
        m_k = float(np.median(np.abs(d_k)))
        if m_k < best_rel_med:
            best_st = k
            best_rel_med = m_k
            best_rel_in50 = float(np.mean(np.abs(d_k) <= 50.0))

    # Octave-aligned F0 contour correlation: fold fo into nearest octave of ft
    oct_shift = np.round(np.log2(ft[valid] / fo[valid]))
    fo_aligned = np.log2(fo[valid]) + oct_shift
    ft_log = np.log2(ft[valid])
    fa = fo_aligned - fo_aligned.mean()
    fb = ft_log - ft_log.mean()
    denom = float(np.linalg.norm(fa) * np.linalg.norm(fb))
    r = float(np.dot(fa, fb) / denom) if denom > 1e-12 else 0.0
    return {
        "vocal_med_f0_hz": round(med_voc, 2) if math.isfinite(med_voc) else None,
        "mut_voiced_frames": n_mut,
        "med_abs_cent_err": round(med_abs, 1),
        "med_signed_cent_bias": round(med_bias, 1),
        "in_tune_frac_50c": round(in50, 3),
        "best_tr_st": best_st,
        "rel_med_abs_cent_err": round(best_rel_med, 1),
        "rel_in_tune_frac_50c": round(best_rel_in50, 3),
        "vocal_f0_r": round(r, 3),
    }


def analyze(input_dir: Path, output_dir: Path, arm_files: dict[str, Path]) -> dict[str, Any]:
    kurt_ref_y = _load_mono_np(input_dir / "music_cover_pitch_lock" / "kurt_vocal_guitar_192f.wav")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roformer = prep_mod._load_mel_roformer(device)
    stems_dir = output_dir / OUT_PREFIX_DIR / "vocal_stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    rep: dict[str, Any] = {"arms": {}}
    try:
        for name, mp4 in arm_files.items():
            spec = ARM_SPECS[name]
            y_mix = _load_mono_np(mp4)
            sx_mix = _load_mono_np(input_dir / spec["src_mix"])
            c_mix = _chroma(y_mix)
            c_sx = _chroma(sx_mix)
            tr_tgt = int(spec["target_tr_st"])
            cx0, lx0, _ = _chroma_sim(c_mix, c_sx, any_key=False, base_tr_st=tr_tgt)
            cxT, lxT, tr = _chroma_sim(c_mix, c_sx, any_key=True, base_tr_st=tr_tgt)
            ox, olx = _onset_corr(y_mix, sx_mix)

            # Separate output vocal stem for true cent-error pitch grading
            out_44k = _load_stereo_44k(mp4)
            voc_44k, _ = prep_mod.separate_roformer(roformer, out_44k, device)
            prep_mod._write_wav_32k_exact(stems_dir / f"{name}_voc.wav", voc_44k, 44100)
            out_voc_24k = torchaudio.functional.resample(voc_44k.mean(dim=0, keepdim=True), 44100, 24000).squeeze(0)
            tgt_voc_24k = torch.from_numpy(_load_mono_np(input_dir / spec["src_voc"]))
            v_metrics = _vocal_pitch_metrics(out_voc_24k, tgt_voc_24k)

            rep["arms"][name] = {
                "file": str(mp4.relative_to(output_dir)),
                "target_tr_st": tr_tgt,
                "chroma_sameKey_x": round(cx0, 3),
                "chroma_lag_x": lx0,
                "chroma_anyKey_x": round(cxT, 3),
                "chroma_drift_st": tr,
                "onset_x": round(ox, 3),
                "onset_lag_x": olx,
                "ltas_cos_kurt_unplugged": round(_ltas_cos(y_mix, kurt_ref_y), 3),
                **v_metrics,
            }
    finally:
        del roformer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(ALL_ARMS))
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args(argv)

    input_dir = _require_dir("H3_INPUT_DIR")
    output_dir = _require_dir("H3_OUTPUT_DIR")
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    graphs: dict[str, dict[str, Any]] = {}
    if args.dry_run or args.run:
        for arm in args.arms:
            g = build_arm_graph(arm)
            gp = GRAPHS_DIR / f"{arm}.json"
            gp.write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")
            rc, out = _preflight(gp)
            print(f"[preflight] {arm}: {'PASS' if rc == 0 else f'FAIL({rc})'}")
            if rc != 0:
                print(out)
                return 1
            graphs[arm] = g
        if args.dry_run and not args.run:
            return 0

    arm_files: dict[str, Path] = {}
    timings: dict[str, float] = {}
    if args.run:
        cid = str(uuid.uuid4())
        for arm in args.arms:
            _wait_idle(args.url)
            print(f"\n[run] queuing {arm}...", flush=True)
            t0 = time.monotonic()
            pid = comfy_client.submit_prompt(graphs[arm], base_url=args.url, client_id=cid)
            hist = comfy_client.wait_for_prompt(pid, base_url=args.url, timeout=1800.0)
            dt = comfy_client.execution_duration_s(hist) or (time.monotonic() - t0)
            timings[arm] = round(dt, 2)
            vids = _extract_videos(hist, output_dir)
            print(f"[run] {arm} finished in {dt:.1f}s -> {[v.name for v in vids]}", flush=True)
            if vids:
                arm_files[arm] = vids[-1]

    if args.analyze or args.run:
        vid_dir = output_dir / OUT_PREFIX_DIR
        if vid_dir.is_dir():
            for arm in args.arms:
                if arm not in arm_files:
                    m = sorted(vid_dir.glob(f"{arm}_*-audio.mp4"))
                    if m:
                        arm_files[arm] = m[-1]
        rp = DATA_DIR / "pitch_lock_triage_report.json"
        prev_timings = {}
        if rp.is_file():
            try:
                prev_timings = json.loads(rp.read_text(encoding="utf-8")).get("timings_s") or {}
            except Exception:
                prev_timings = {}
        rep = analyze(input_dir, output_dir, arm_files)
        if timings or prev_timings:
            rep["timings_s"] = timings or prev_timings
        rp.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
