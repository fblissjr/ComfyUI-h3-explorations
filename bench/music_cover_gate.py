#!/usr/bin/env python3
"""Phase 1 Zero-Code Diagnostic Gate for H3 Music-Video Voice & Style Adaptation.

Tests how MiniMax H3 `ref2va` adapts Song X (sung by Person A in Style A) into
Target B's visual identity, singing voice, and acoustic instrument style using
100% existing nodes at matched seed (`seed=730451892`, `1152x768`, `141f = 5.875 s`):

  1. `c0_voice_ceiling`            -- <Picture 1> + <Audio 1> (Target B) + <d> lyrics (no Song X audio)
  2. `a1_native_dual_raw`          -- Method A: <Audio 1> (Target B) + <Audio 2> (raw Song X)
  3. `a2_native_dual_scaffold`     -- Method A: <Audio 1> (Target B) + <Audio 2> (baritone harmonic scaffold)
  4. `b1_cond_release_33_raw`      -- Method B: <Audio 1> (t<0) + raw Song X cond_audio (t>=0) released @ 0.33
  5. `b2_cond_release_50_raw`      -- Method B: <Audio 1> (t<0) + raw Song X cond_audio (t>=0) released @ 0.50
  6. `b3_cond_release_50_scaffold` -- Method B: <Audio 1> (t<0) + baritone scaffold cond_audio released @ 0.50
  7. `c1_twopass_rescore`          -- Method C: Pass-1 FreezeAudio timing lock -> Pass-2 AudioRefineMask + b3
  8. `b4_base16_scaffold`          -- Undistilled Base ref2va (16-step euler/simple) control on b3 recipe

Usage:
  "$PY" bench/music_cover_gate.py --dry-run
  "$PY" bench/music_cover_gate.py --run
  "$PY" bench/music_cover_gate.py --run --arms c0_voice_ceiling a1_native_dual_raw b1_cond_release_33_raw
  "$PY" bench/music_cover_gate.py --analyze
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

COMFY_ROOT = bootstrap(cpu=True)
sys.path = [p for p in sys.path if p != str(REPO)]
if str(REPO / "workflows") not in sys.path:
    sys.path.insert(1, str(REPO / "workflows"))

import h3_config  # noqa: E402
import preflight_graph as pf  # noqa: E402
import probe_yin_f0_drift as yin_probe  # noqa: E402

DEFAULT_URL = comfy_client.DEFAULT_COMFY_URL
OUT_PREFIX_DIR = "Video/music_cover_gate_2026-11-27"
DATA_DIR = REPO / "data" / "music_cover_gate"
GRAPHS_DIR = DATA_DIR / "graphs"

TARGET_B_IMAGE = "music_cover_gate/target_b_solo.png"
TARGET_B_AUDIO = "music_cover_gate/target_b_audio.wav"
SONG_X_RAW = "music_cover_gate/song_x_raw.wav"
SONG_X_HARMONIC = "music_cover_gate/song_x_harmonic_scaffold.wav"
SONG_X_SCAFFOLD = "music_cover_gate/song_x_baritone_scaffold.wav"

PROMPT_B_ONLY_PATH = DATA_DIR / "ref2va_music_cover_gate_b_only.txt"
PROMPT_DUAL_PATH = DATA_DIR / "ref2va_music_cover_gate_dual_audio.txt"

WIDTH = 1152
HEIGHT = 768
LENGTH = 141
SEED = 730451892

UNET_REF2VA = h3_config.MODELS["unet_ref2va"]
ENCODER_BF16 = h3_config.MODELS["clip"]
VIDEO_VAE = h3_config.MODELS["video_vae"]
AUDIO_VAE = h3_config.MODELS["audio_vae"]
LORA_PIKA_REF2VA = h3_config.PIKA_REF2VA_SVD995_LORA
PIKA_SIGMAS = h3_config.PIKA_MANUAL_SIGMAS

ALL_ARMS: tuple[str, ...] = (
    "c0_voice_ceiling",
    "a1_native_dual_raw",
    "a2_native_dual_scaffold",
    "b1_cond_release_33_raw",
    "b2_cond_release_50_raw",
    "b3_cond_release_50_scaffold",
    "c1_twopass_rescore",
    "b4_base16_scaffold",
    "p1b_b2_clean55_raw",
    "p1b_b2_clean65_harmonic",
    "p1b_base16_raw_rel38",
    "p1b_base16_raw_rel45",
    "p1b_twopass_raw_clean55",
    "p2_b2_router_raw_s24",
    "p2_b2_router_harm_s24",
    "p2_b2_router_raw_clean100_s20",
)


def _require_dir(env_name: str) -> Path:
    val = os.environ.get(env_name)
    if not val:
        raise SystemExit(f"{env_name} is unset; run `set -a; . ./h3.env; set +a` first")
    p = Path(val)
    if not p.is_dir():
        raise SystemExit(f"{env_name} does not exist: {p}")
    return p


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


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


def _wire_ref_chain(
    g: dict[str, Any],
    *,
    prompt_text: str,
    second_audio: str | None = None,
    use_pika: bool = True,
) -> tuple[str, str]:
    """Wire <Picture 1> + <Audio 1> (+ optional <Audio 2>) via split-cached EncodeReferences."""
    g["15"] = {"class_type": "LoadImage", "inputs": {"image": TARGET_B_IMAGE}}
    if use_pika:
        g["50"] = {
            "class_type": "MiniMaxH3AppendRefImage",
            "inputs": {
                "image": ["15", 0],
                "size_policy": "tiered",
                "qwen_view": "separate",
                "qwen_view.qwen_short_edge": 512,
            },
        }
    else:
        g["50"] = {
            "class_type": "MiniMaxH3AppendRefImage",
            "inputs": {
                "image": ["15", 0],
                "size_policy": "max",
                "size_policy.dit_short_edge": 2048,
                "size_policy.allow_upscale": False,
                "qwen_view": "separate",
                "qwen_view.qwen_short_edge": 512,
            },
        }
    g["33"] = {"class_type": "LoadAudio", "inputs": {"audio": TARGET_B_AUDIO}}
    g["51"] = {
        "class_type": "MiniMaxH3AppendRefAudio",
        "inputs": {"audio": ["33", 0], "references": ["50", 0]},
    }
    ref_tail = "51"
    if second_audio is not None:
        g["34"] = {"class_type": "LoadAudio", "inputs": {"audio": second_audio}}
        g["52"] = {
            "class_type": "MiniMaxH3AppendRefAudio",
            "inputs": {"audio": ["34", 0], "references": ["51", 0]},
        }
        ref_tail = "52"

    g["132"] = {
        "class_type": "MiniMaxH3EncodeReferences",
        "inputs": {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "audio_vae": ["4", 0],
            "references": [ref_tail, 0],
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


def _wire_vhs(
    g: dict[str, Any],
    *,
    vhs_id: str,
    images_ref: list[Any],
    audio_ref: list[Any],
    prefix: str,
) -> None:
    g[vhs_id] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": images_ref,
            "audio": audio_ref,
            "frame_rate": 24.0,
            "loop_count": 0,
            "filename_prefix": prefix,
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
        },
    }


def _wire_sampler_schedule(g: dict[str, Any], *, model_id: str, use_pika: bool) -> tuple[str, str]:
    if use_pika:
        g["7"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}}
        g["60"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": PIKA_SIGMAS}}
        return "7", "60"
    g["7"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    g["60"] = {
        "class_type": "BasicScheduler",
        "inputs": {"model": [model_id, 0], "scheduler": "simple", "steps": 16, "denoise": 1.0},
    }
    return "7", "60"


def build_arm_graph(arm: str) -> dict[str, Any]:
    prompt_b = _load_prompt(PROMPT_B_ONLY_PATH)
    prompt_dual = _load_prompt(PROMPT_DUAL_PATH)
    use_pika = arm not in ("b4_base16_scaffold", "p1b_base16_raw_rel38", "p1b_base16_raw_rel45")

    g = _base_loaders()
    sol_id = _wire_model_chain(g, use_pika=use_pika)

    if arm == "c0_voice_ceiling":
        cond_id, enc_id = _wire_ref_chain(g, prompt_text=prompt_b, second_audio=None, use_pika=True)
        g["26"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": [cond_id, 0], "samples": [enc_id, 1]}}
        model_id = sol_id
        sampler_id, sigmas_id = _wire_sampler_schedule(g, model_id=model_id, use_pika=True)
        g["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}}
        g["9"] = {"class_type": "BasicGuider", "inputs": {"model": [model_id, 0], "conditioning": ["26", 0]}}
        g["10"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["6", 0],
                "guider": ["9", 0],
                "sampler": [sampler_id, 0],
                "sigmas": [sigmas_id, 0],
                "latent_image": ["26", 1],
            },
        }
        g["30"] = {
            "class_type": "MiniMaxH3ProvenanceStamp",
            "inputs": {"latent": ["10", 0], "model": [model_id, 0], "sigmas": [sigmas_id, 0], "clip": ["2", 0], "note": arm},
        }
        g["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}}
        g["12"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}}
        _wire_vhs(g, vhs_id="13", images_ref=["11", 0], audio_ref=["12", 0], prefix=f"{OUT_PREFIX_DIR}/{arm}")
        return g

    if arm in ("a1_native_dual_raw", "a2_native_dual_scaffold"):
        song_ref = SONG_X_RAW if arm == "a1_native_dual_raw" else SONG_X_SCAFFOLD
        cond_id, enc_id = _wire_ref_chain(g, prompt_text=prompt_dual, second_audio=song_ref, use_pika=True)
        g["26"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": [cond_id, 0], "samples": [enc_id, 1]}}
        model_id = sol_id
        sampler_id, sigmas_id = _wire_sampler_schedule(g, model_id=model_id, use_pika=True)
        g["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}}
        g["9"] = {"class_type": "BasicGuider", "inputs": {"model": [model_id, 0], "conditioning": ["26", 0]}}
        g["10"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["6", 0],
                "guider": ["9", 0],
                "sampler": [sampler_id, 0],
                "sigmas": [sigmas_id, 0],
                "latent_image": ["26", 1],
            },
        }
        g["30"] = {
            "class_type": "MiniMaxH3ProvenanceStamp",
            "inputs": {"latent": ["10", 0], "model": [model_id, 0], "sigmas": [sigmas_id, 0], "clip": ["2", 0], "note": arm},
        }
        g["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}}
        g["12"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}}
        _wire_vhs(g, vhs_id="13", images_ref=["11", 0], audio_ref=["12", 0], prefix=f"{OUT_PREFIX_DIR}/{arm}")
        return g

    cond_release_specs: dict[str, tuple[str, float, float]] = {
        "b1_cond_release_33_raw": (SONG_X_RAW, 0.33, 1.0),
        "b2_cond_release_50_raw": (SONG_X_RAW, 0.50, 1.0),
        "b3_cond_release_50_scaffold": (SONG_X_SCAFFOLD, 0.50, 1.0),
        "b4_base16_scaffold": (SONG_X_SCAFFOLD, 0.50, 1.0),
        "p1b_b2_clean55_raw": (SONG_X_RAW, 0.50, 0.55),
        "p1b_b2_clean65_harmonic": (SONG_X_HARMONIC, 0.50, 0.65),
        "p1b_base16_raw_rel38": (SONG_X_RAW, 0.38, 1.0),
        "p1b_base16_raw_rel45": (SONG_X_RAW, 0.45, 1.0),
    }
    if arm in cond_release_specs:
        guide_audio, aud_end, aud_clean = cond_release_specs[arm]
        cond_id, enc_id = _wire_ref_chain(g, prompt_text=prompt_b, second_audio=None, use_pika=use_pika)
        g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": guide_audio}}
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
        g["150"] = {
            "class_type": "MiniMaxH3ReleaseReferences",
            "inputs": {
                "model": [sol_id, 0],
                "end_percent": 1.0,
                "clean_level": 0.999,
                "audio_end_percent": float(aud_end),
                "audio_clean_level": float(aud_clean),
                "video_end_percent": 1.0,
                "guide_end_percent": 1.0,
            },
        }
        model_id = "150"
        sampler_id, sigmas_id = _wire_sampler_schedule(g, model_id=model_id, use_pika=use_pika)
        g["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}}
        g["9"] = {"class_type": "BasicGuider", "inputs": {"model": [model_id, 0], "conditioning": ["26", 0]}}
        g["10"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["6", 0],
                "guider": ["9", 0],
                "sampler": [sampler_id, 0],
                "sigmas": [sigmas_id, 0],
                "latent_image": ["26", 1],
            },
        }
        g["30"] = {
            "class_type": "MiniMaxH3ProvenanceStamp",
            "inputs": {"latent": ["10", 0], "model": [model_id, 0], "sigmas": [sigmas_id, 0], "clip": ["2", 0], "note": arm},
        }
        g["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}}
        g["12"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}}
        _wire_vhs(g, vhs_id="13", images_ref=["11", 0], audio_ref=["12", 0], prefix=f"{OUT_PREFIX_DIR}/{arm}")
        return g

    if arm in ("c1_twopass_rescore", "p1b_twopass_raw_clean55"):
        pass1_audio = SONG_X_SCAFFOLD if arm == "c1_twopass_rescore" else SONG_X_RAW
        pass2_guide_audio = SONG_X_SCAFFOLD if arm == "c1_twopass_rescore" else SONG_X_HARMONIC
        pass2_clean = 1.0 if arm == "c1_twopass_rescore" else 0.60
        cond_id, enc_id = _wire_ref_chain(g, prompt_text=prompt_b, second_audio=None, use_pika=True)
        g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": pass1_audio}}
        g["148"] = {"class_type": "LoadAudio", "inputs": {"audio": pass2_guide_audio}}
        g["26"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": [cond_id, 0], "samples": [enc_id, 1]}}
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
        sampler_id, sigmas_id = _wire_sampler_schedule(g, model_id=sol_id, use_pika=True)
        g["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}}
        g["9"] = {"class_type": "BasicGuider", "inputs": {"model": [sol_id, 0], "conditioning": ["26", 0]}}
        g["10"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["6", 0],
                "guider": ["9", 0],
                "sampler": [sampler_id, 0],
                "sigmas": [sigmas_id, 0],
                "latent_image": ["49", 0],
            },
        }
        g["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}}
        _wire_vhs(
            g,
            vhs_id="13",
            images_ref=["11", 0],
            audio_ref=["49", 1],
            prefix=f"{OUT_PREFIX_DIR}/{arm}_pass1",
        )

        g["151"] = {
            "class_type": "MiniMaxH3GuideLatent",
            "inputs": {
                "positive": [cond_id, 0],
                "latent": ["10", 0],
                "frame_idx": 0,
                "single_frame": False,
                "audio": ["148", 0],
                "audio_vae": ["4", 0],
            },
        }
        g["152"] = {
            "class_type": "MiniMaxH3AudioRefineMask",
            "inputs": {"latent": ["151", 1], "video_mask": 0.0, "audio_mask": 1.0},
        }
        g["150"] = {
            "class_type": "MiniMaxH3ReleaseReferences",
            "inputs": {
                "model": [sol_id, 0],
                "end_percent": 1.0,
                "clean_level": 0.999,
                "audio_end_percent": 0.50,
                "audio_clean_level": float(pass2_clean),
                "video_end_percent": 1.0,
                "guide_end_percent": 1.0,
            },
        }
        g["106"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED + 17}}
        g["109"] = {"class_type": "BasicGuider", "inputs": {"model": ["150", 0], "conditioning": ["151", 0]}}
        g["110"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["106", 0],
                "guider": ["109", 0],
                "sampler": [sampler_id, 0],
                "sigmas": [sigmas_id, 0],
                "latent_image": ["152", 0],
            },
        }
        g["30"] = {
            "class_type": "MiniMaxH3ProvenanceStamp",
            "inputs": {"latent": ["110", 0], "model": ["150", 0], "sigmas": [sigmas_id, 0], "clip": ["2", 0], "note": arm},
        }
        g["111"] = {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}}
        g["112"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}}
        _wire_vhs(
            g,
            vhs_id="113",
            images_ref=["111", 0],
            audio_ref=["112", 0],
            prefix=f"{OUT_PREFIX_DIR}/{arm}_pass2",
        )
        return g

    if arm in ("p2_b2_router_raw_s24", "p2_b2_router_harm_s24", "p2_b2_router_raw_clean100_s20"):
        guide_audio = SONG_X_HARMONIC if arm == "p2_b2_router_harm_s24" else SONG_X_RAW
        is_clean100 = arm == "p2_b2_router_raw_clean100_s20"
        cond_id, enc_id = _wire_ref_chain(g, prompt_text=prompt_b, second_audio=None, use_pika=True)
        g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": guide_audio}}
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
        g["150"] = {
            "class_type": "MiniMaxH3AudioRefRouter",
            "inputs": {
                "model": [sol_id, 0],
                "melody_source": "cond_audio",
                "split_block": 20 if is_clean100 else 24,
                "melody_gain_early": 1.5 if is_clean100 else 1.0,
                "melody_gain_late": 0.0,
                "timbre_gain_early": 0.75,
                "timbre_gain_late": 2.5,
                "melody_end_percent": 0.50,
                "melody_clean_level": 1.0 if is_clean100 else (0.70 if arm == "p2_b2_router_raw_s24" else 0.75),
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        }
        model_id = "150"
        sampler_id, sigmas_id = _wire_sampler_schedule(g, model_id=model_id, use_pika=True)
        g["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}}
        g["9"] = {"class_type": "BasicGuider", "inputs": {"model": [model_id, 0], "conditioning": ["26", 0]}}
        g["10"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["6", 0],
                "guider": ["9", 0],
                "sampler": [sampler_id, 0],
                "sigmas": [sigmas_id, 0],
                "latent_image": ["26", 1],
            },
        }
        g["30"] = {
            "class_type": "MiniMaxH3ProvenanceStamp",
            "inputs": {"latent": ["10", 0], "model": [model_id, 0], "sigmas": [sigmas_id, 0], "clip": ["2", 0], "note": arm},
        }
        g["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}}
        g["12"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}}
        _wire_vhs(g, vhs_id="13", images_ref=["11", 0], audio_ref=["12", 0], prefix=f"{OUT_PREFIX_DIR}/{arm}")
        return g

    raise ValueError(f"Unknown arm: {arm}")


def _preflight_graph(graph_path: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(REPO / "bench" / "preflight_graph.py"), str(graph_path)],
        capture_output=True,
        text=True,
    )
    out = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode, out


def _queue_status(url: str) -> tuple[int, int]:
    req = urllib.request.urlopen(f"{url}/queue", timeout=5)
    data = json.loads(req.read().decode("utf-8"))
    return len(data.get("queue_running", [])), len(data.get("queue_pending", []))


def _comfy_server_pid() -> int | None:
    try:
        out = subprocess.check_output(["pgrep", "-f", r"main\.py.*--port[= ]8188"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            tok = line.strip()
            if tok.isdigit():
                return int(tok)
    except Exception:
        pass
    try:
        out = subprocess.check_output(["ss", "-ltnp", "sport = :8188"], text=True, stderr=subprocess.DEVNULL)
        for part in out.split("pid=")[1:]:
            num = "".join(ch for ch in part if ch.isdigit())
            if num:
                return int(num)
    except Exception:
        pass
    return None


def _external_gpu_hogs(comfy_pid: int | None, *, min_mib: int = 1000) -> list[tuple[int, int]]:
    """Return `(pid, used_mib)` for non-ComfyUI GPU compute processes above `min_mib`."""
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
            pid = int(parts[0])
            mib = int(float(parts[1]))
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
        running, pending = _queue_status(url)
        hogs = _external_gpu_hogs(comfy_pid)
        if running == 0 and pending == 0 and not hogs:
            if logged:
                print(f"[gpu-wait] GPU & :8188 idle after {time.monotonic() - t0:.1f}s", flush=True)
            return
        if not logged:
            print(
                f"[gpu-wait] waiting politely for GPU/:8188 (queue={running}/{pending}, external_gpu_pids={hogs})...",
                flush=True,
            )
            logged = True
        if time.monotonic() - t0 > max_wait_s:
            raise TimeoutError(f"GPU/:8188 stayed busy for > {max_wait_s}s")
        time.sleep(poll_s)


def _extract_saved_videos(history_entry: dict[str, Any], output_dir: Path) -> list[Path]:
    found: list[Path] = []
    outputs = history_entry.get("outputs") or {}
    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        for key in ("gifs", "videos"):
            items = node_out.get(key) or []
            for item in items:
                if not isinstance(item, dict) or not item.get("filename"):
                    continue
                sub = item.get("subfolder") or ""
                p = output_dir / sub / item["filename"]
                if p.is_file():
                    found.append(p)
    return found


def _load_mono(path: Path, sr: int = 24000) -> torch.Tensor:
    return yin_probe.load_mono_audio(path, sr=sr)


def _rms_envelope(wav: torch.Tensor, sr: int = 24000, hop_ms: float = 25.0) -> np.ndarray:
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    x = wav.detach().cpu().numpy().astype(np.float64)
    n_frames = len(x) // hop
    if n_frames < 2:
        return np.zeros(1, dtype=np.float64)
    blocks = x[: n_frames * hop].reshape(n_frames, hop)
    return np.sqrt(np.mean(blocks * blocks, axis=1) + 1e-12)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    aa = a[:n] - float(np.mean(a[:n]))
    bb = b[:n] - float(np.mean(b[:n]))
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den < 1e-12:
        return float("nan")
    return float(np.dot(aa, bb) / den)


def _logmel(wav: torch.Tensor, sr: int = 24000) -> torch.Tensor:
    spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=256, n_mels=64)(wav)
    return torch.log(spec + 1e-6)


def _frame_logmel_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    A, B = _logmel(a), _logmel(b)
    n = min(A.shape[-1], B.shape[-1])
    A, B = A[..., :n], B[..., :n]
    A = A - A.mean(dim=0, keepdim=True)
    B = B - B.mean(dim=0, keepdim=True)
    num = (A * B).sum(dim=0)
    den = A.norm(dim=0) * B.norm(dim=0) + 1e-12
    return float((num / den).mean().item())


def _ltas_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of mean-centered long-term average log-mel timbre spectra."""
    va = _logmel(a).mean(dim=-1)
    vb = _logmel(b).mean(dim=-1)
    va = va - va.mean()
    vb = vb - vb.mean()
    den = float((va.norm() * vb.norm()).item()) + 1e-12
    return float((torch.dot(va, vb).item()) / den)


def _hf_share(wav: torch.Tensor, sr: int = 24000, cutoff_hz: float = 4000.0) -> float:
    x = wav.detach().cpu().numpy().astype(np.float64)
    if len(x) < 1024:
        return 0.0
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    tot = float(np.sum(spec)) + 1e-12
    return float(np.sum(spec[freqs >= cutoff_hz]) / tot)


def _f0_contour_corr(f0_a: torch.Tensor, f0_b: torch.Tensor) -> tuple[float, int]:
    """Octave/key-invariant pitch contour correlation on mutually voiced hops."""
    n = min(int(f0_a.numel()), int(f0_b.numel()))
    a = f0_a[:n].numpy()
    b = f0_b[:n].numpy()
    both = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    k = int(np.sum(both))
    if k < 8:
        return float("nan"), k
    la = np.log2(a[both])
    lb = np.log2(b[both])
    return _pearson(la, lb), k


def _video_mae_between(mp4_a: Path, mp4_b: Path) -> float:
    def _decode_rgb(p: Path) -> np.ndarray:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(p), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True,
            check=True,
        )
        return np.frombuffer(proc.stdout, dtype=np.uint8)

    a = _decode_rgb(mp4_a)
    b = _decode_rgb(mp4_b)
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(a[:n].astype(np.float32) - b[:n].astype(np.float32))))


def analyze_renders(input_dir: Path, output_dir: Path, arm_files: dict[str, Path]) -> dict[str, Any]:
    ref_b_wav = _load_mono(input_dir / TARGET_B_AUDIO)
    ref_x_raw_wav = _load_mono(input_dir / SONG_X_RAW)
    ref_x_scaf_wav = _load_mono(input_dir / SONG_X_SCAFFOLD)

    f0_b, _, _ = yin_probe.yin_f0(ref_b_wav)
    f0_x_raw, _, _ = yin_probe.yin_f0(ref_x_raw_wav)
    f0_x_scaf, _, _ = yin_probe.yin_f0(ref_x_scaf_wav)

    env_b = _rms_envelope(ref_b_wav)
    env_x_raw = _rms_envelope(ref_x_raw_wav)
    env_x_scaf = _rms_envelope(ref_x_scaf_wav)

    med_b = float(np.nanmedian(f0_b.numpy()))
    med_x_raw = float(np.nanmedian(f0_x_raw.numpy()))
    med_x_scaf = float(np.nanmedian(f0_x_scaf.numpy()))

    summary: dict[str, Any] = {
        "references": {
            "target_b_median_f0_hz": round(med_b, 2),
            "target_b_hf4k_share": round(_hf_share(ref_b_wav), 4),
            "song_x_raw_median_f0_hz": round(med_x_raw, 2),
            "song_x_raw_hf4k_share": round(_hf_share(ref_x_raw_wav), 4),
            "song_x_scaffold_median_f0_hz": round(med_x_scaf, 2),
            "song_x_scaffold_hf4k_share": round(_hf_share(ref_x_scaf_wav), 4),
        },
        "arms": {},
    }

    for name, mp4 in arm_files.items():
        if not mp4.is_file():
            continue
        wav = _load_mono(mp4)
        rms = float(torch.sqrt(torch.mean(wav ** 2) + 1e-12).item())
        rms_dbfs = 20.0 * math.log10(max(rms, 1e-9))
        f0, _, _ = yin_probe.yin_f0(wav)
        f0_np = f0.numpy()
        valid = np.isfinite(f0_np) & (f0_np > 0)
        med_f0 = float(np.median(f0_np[valid])) if np.any(valid) else float("nan")
        st_vs_b = 12.0 * math.log2(med_f0 / med_b) if (math.isfinite(med_f0) and med_b > 0) else float("nan")
        st_vs_x = 12.0 * math.log2(med_f0 / med_x_raw) if (math.isfinite(med_f0) and med_x_raw > 0) else float("nan")

        env = _rms_envelope(wav)
        corr_f0_x_raw, n_f0_x_raw = _f0_contour_corr(f0, f0_x_raw)
        corr_f0_x_scaf, n_f0_x_scaf = _f0_contour_corr(f0, f0_x_scaf)
        corr_f0_b, n_f0_b = _f0_contour_corr(f0, f0_b)

        row = {
            "file": str(mp4.relative_to(output_dir)),
            "rms_dbfs": round(rms_dbfs, 2),
            "hf4k_share": round(_hf_share(wav), 4),
            "voiced_frac": round(float(np.mean(valid)), 3),
            "median_f0_hz": round(med_f0, 2) if math.isfinite(med_f0) else None,
            "st_vs_target_b": round(st_vs_b, 2) if math.isfinite(st_vs_b) else None,
            "st_vs_song_x_raw": round(st_vs_x, 2) if math.isfinite(st_vs_x) else None,
            "f0_contour_corr_song_x_raw": round(corr_f0_x_raw, 3) if math.isfinite(corr_f0_x_raw) else None,
            "f0_contour_n_song_x_raw": n_f0_x_raw,
            "f0_contour_corr_song_x_scaf": round(corr_f0_x_scaf, 3) if math.isfinite(corr_f0_x_scaf) else None,
            "f0_contour_n_song_x_scaf": n_f0_x_scaf,
            "f0_contour_corr_target_b": round(corr_f0_b, 3) if math.isfinite(corr_f0_b) else None,
            "env_corr_song_x_raw": round(_pearson(env, env_x_raw), 3),
            "env_corr_song_x_scaf": round(_pearson(env, env_x_scaf), 3),
            "env_corr_target_b": round(_pearson(env, env_b), 3),
            "ltas_cos_target_b": round(_ltas_cosine(wav, ref_b_wav), 3),
            "ltas_cos_song_x_raw": round(_ltas_cosine(wav, ref_x_raw_wav), 3),
            "frame_mel_corr_song_x_raw": round(_frame_logmel_corr(wav, ref_x_raw_wav), 3),
            "frame_mel_corr_target_b": round(_frame_logmel_corr(wav, ref_b_wav), 3),
        }
        summary["arms"][name] = row

    p1 = arm_files.get("c1_twopass_rescore_pass1")
    p2 = arm_files.get("c1_twopass_rescore_pass2")
    if p1 and p2 and p1.is_file() and p2.is_file():
        summary["c1_pass2_vs_pass1_video_mae"] = round(_video_mae_between(p1, p2), 4)

    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 1 Zero-Code H3 Music-Video Cover Adaptation Gate")
    ap.add_argument("--dry-run", action="store_true", help="Compile all arm JSONs and run preflight_graph.py offline")
    ap.add_argument("--run", action="store_true", help="Queue arms on :8188 (waiting politely if queue is busy) and analyze")
    ap.add_argument("--analyze", action="store_true", help="Run CPU audio triage on already-rendered outputs")
    ap.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(ALL_ARMS), help="Subset of arms to build/run")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"ComfyUI server URL (default: {DEFAULT_URL})")
    args = ap.parse_args(argv)

    if not (args.dry_run or args.run or args.analyze):
        ap.error("Specify one of --dry-run, --run, or --analyze")

    input_dir = _require_dir("H3_INPUT_DIR")
    output_dir = _require_dir("H3_OUTPUT_DIR")
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    for rel in (TARGET_B_IMAGE, TARGET_B_AUDIO, SONG_X_RAW, SONG_X_SCAFFOLD):
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
            status = "PASS" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[preflight] {arm}: {status}")
            for line in out.splitlines():
                if any(k in line for k in ("FAIL", "WARN", "tokens", "total", "grid")):
                    print("   ", line)
            if rc != 0:
                any_fail = True
            graphs[arm] = (gp, g)
        if any_fail:
            print("Aborting: one or more arm graphs failed preflight.", file=sys.stderr)
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
            for p in saved:
                if arm in ("c1_twopass_rescore", "p1b_twopass_raw_clean55"):
                    if "_pass1" in p.name:
                        arm_files[f"{arm}_pass1"] = p
                    elif "_pass2" in p.name:
                        arm_files[f"{arm}_pass2"] = p
                else:
                    arm_files[arm] = p

    if args.analyze or args.run:
        vid_dir = output_dir / OUT_PREFIX_DIR
        if vid_dir.is_dir():
            for arm in ALL_ARMS:
                if arm in ("c1_twopass_rescore", "p1b_twopass_raw_clean55"):
                    for suf in ("_pass1", "_pass2"):
                        key = f"{arm}{suf}"
                        if key not in arm_files:
                            matches = sorted(vid_dir.glob(f"{arm}{suf}_*-audio.mp4"))
                            if matches:
                                arm_files[key] = matches[-1]
                elif arm not in arm_files:
                    matches = sorted(vid_dir.glob(f"{arm}_*-audio.mp4"))
                    if matches:
                        arm_files[arm] = matches[-1]

        report = analyze_renders(input_dir, output_dir, arm_files)
        if timings:
            report["timings_s"] = timings
        report_path = DATA_DIR / "phase1_triage_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\n[triage] wrote {report_path.relative_to(REPO)}")
        print(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
