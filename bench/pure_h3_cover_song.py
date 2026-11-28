#!/usr/bin/env python3
"""Pure-H3 Calibrated & 4-Window Chained Music-Video Cover (`justaflicker`).

Stages phase-clean MelBand-RoFormer + HDemucs cascade stems via `bench/h3_stem_sep.py`
and executes two GPU tracks against `:8188`:
  - `--calibrate`: 9-arm 243f (`10.125 s`) multi-song register-locked gate (`justaflicker`
    `-5 st` Verse / Chorus / Peak Bridge with `d=0.26..0.32`, 2-seed `d=0.30` repeat, and
    `MiniMaxH3QueryRouter(aud_to_vid_gain=0.0)` rain-diner split; `thatsnotakite` `-6 st` and
    `inthefridge` `+2 st` `vocals + bass` choruses; `vocal_formant=False`).
  - `--song`: 4-window chained 294f (`44.125 s`, `1059` frames, `8.875..53.000 s`) Verse+Chorus
    pure-H3 cover using `TrimAudioDuration + AudioConcat -> MiniMaxH3FreezeAudioWindow(
    context_frames=39, context_noise=0.35, context_profile="seam_taper")` on stock RoPE across
    `w1 -> w2 -> w3 -> w4`, stitching decoded H3 video and equal-power cosine-crossfaded H3 audio.
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

bootstrap(cpu=None)
sys.path = [p for p in sys.path if p != str(REPO)]
if str(REPO / "workflows") not in sys.path:
    sys.path.insert(1, str(REPO / "workflows"))

import h3_config  # noqa: E402
import h3_stem_sep as stem_sep  # noqa: E402
import pitch_lock_metrics  # noqa: E402
import probe_yin_f0_drift as yin_probe  # noqa: E402

DEFAULT_URL = comfy_client.DEFAULT_COMFY_URL
OUT_PREFIX_DIR = "Video/pure_h3_cover_song_2026-11-27"
DATA_DIR = REPO / "data" / "pure_h3_cover_song"
GRAPHS_DIR = DATA_DIR / "graphs"

PIC_FACE = "music_cover_pitch_lock/kurt_closeup.png"
PIC_STAGE = "music_cover_pitch_lock/kurt_solo_medium.png"
REF_AUDIO = "music_cover_stems/kurt_vocal_clean_192f.wav"
STEMS_SUBDIR = "music_cover_stems/justaflicker"

WIDTH = 1024
HEIGHT = 768
CAL_LENGTH = 243
CAL_DUR_S = CAL_LENGTH / 24.0  # 10.125 s
VERSE_CAL_START_S = 10.500
CHORUS_CAL_START_S = 42.200
BRIDGE_CAL_START_S = 103.000
KITE_CAL_START_S = 63.625
FRIDGE_CAL_START_S = 60.500
KITE_SEMITONES = -6.0
FRIDGE_SEMITONES = 2.0
SONG_WIN_FRAMES = 294
SONG_CTX_FRAMES = 39
SONG_STRIDE_FRAMES = SONG_WIN_FRAMES - SONG_CTX_FRAMES  # 255
SONG_WIN_S = SONG_WIN_FRAMES / 24.0  # 12.250 s
SONG_STRIDE_S = SONG_STRIDE_FRAMES / 24.0  # 10.625 s
SONG_N_WINDOWS = 4
SONG_TOTAL_FRAMES = SONG_WIN_FRAMES + (SONG_N_WINDOWS - 1) * SONG_STRIDE_FRAMES  # 1059
SONG_START_S = 8.875
SONG_DUR_S = SONG_TOTAL_FRAMES / 24.0  # 44.125 s
COVER_SEMITONES = -5.0
SEED = 910427311

VERSE_SCAFFOLD_REL = "music_cover_stems/flicker_m5_verse_cal_243f.wav"
VERSE_VOC_TGT_REL = "music_cover_stems/flicker_voc_m5_verse_243f.wav"
VERSE_RAW_ORIG_REL = "music_cover_stems/flicker_raw_orig_verse_243f.wav"
CHORUS_SCAFFOLD_REL = "music_cover_stems/flicker_m5_chorus_cal_243f.wav"
CHORUS_VOC_TGT_REL = "music_cover_stems/flicker_voc_m5_chorus_243f.wav"
CHORUS_RAW_ORIG_REL = "music_cover_stems/flicker_raw_orig_chorus_243f.wav"
BRIDGE_SCAFFOLD_REL = "music_cover_stems/flicker_m5_bridge_cal_243f.wav"
BRIDGE_VOC_TGT_REL = "music_cover_stems/flicker_voc_m5_bridge_243f.wav"
BRIDGE_RAW_ORIG_REL = "music_cover_stems/flicker_raw_orig_bridge_243f.wav"
KITE_SCAFFOLD_REL = "music_cover_stems/kite_m6_voc_bass_cal_243f.wav"
KITE_VOC_TGT_REL = "music_cover_stems/kite_voc_m6_243f.wav"
KITE_RAW_ORIG_REL = "music_cover_stems/kite_raw_orig_243f.wav"
FRIDGE_SCAFFOLD_REL = "music_cover_stems/fridge_p2_voc_bass_cal_243f.wav"
FRIDGE_VOC_TGT_REL = "music_cover_stems/fridge_voc_p2_243f.wav"
FRIDGE_RAW_ORIG_REL = "music_cover_stems/fridge_raw_orig_243f.wav"
SONG_SCAFFOLD_REL = "music_cover_stems/flicker_m5_verse_chorus_44s.wav"
SONG_VOC_TGT_REL = "music_cover_stems/flicker_voc_m5_44s.wav"
SONG_RAW_ORIG_REL = "music_cover_stems/flicker_raw_orig_44s.wav"

UNET_REF2VA = h3_config.MODELS["unet_ref2va"]
ENCODER_BF16 = h3_config.MODELS["clip"]
VIDEO_VAE = h3_config.MODELS["video_vae"]
AUDIO_VAE = h3_config.MODELS["audio_vae"]
LORA_PIKA_REF2VA = h3_config.PIKA_REF2VA_SVD995_LORA
PIKA_SIGMAS = h3_config.PIKA_MANUAL_SIGMAS

CAL_ARMS: dict[str, dict[str, Any]] = {
    "c1_verse_m5_d26": {
        "denoise": 0.26,
        "custom_rain": False,
        "tr_st": -5,
        "scaffold_rel": VERSE_SCAFFOLD_REL,
        "voc_tgt_rel": VERSE_VOC_TGT_REL,
        "raw_orig_rel": VERSE_RAW_ORIG_REL,
        "cam_prompt": DATA_DIR / "ref2va_cal_unplugged_v1.txt",
        "mic_prompt": None,
    },
    "c2_verse_m5_d32": {
        "denoise": 0.32,
        "custom_rain": False,
        "tr_st": -5,
        "scaffold_rel": VERSE_SCAFFOLD_REL,
        "voc_tgt_rel": VERSE_VOC_TGT_REL,
        "raw_orig_rel": VERSE_RAW_ORIG_REL,
        "cam_prompt": DATA_DIR / "ref2va_cal_unplugged_v1.txt",
        "mic_prompt": None,
    },
    "c3_chorus_m5_d30": {
        "denoise": 0.30,
        "custom_rain": False,
        "tr_st": -5,
        "scaffold_rel": CHORUS_SCAFFOLD_REL,
        "voc_tgt_rel": CHORUS_VOC_TGT_REL,
        "raw_orig_rel": CHORUS_RAW_ORIG_REL,
        "cam_prompt": DATA_DIR / "ref2va_cal_unplugged_chorus.txt",
        "mic_prompt": None,
    },
    "c4_verse_m5_rain_a2v0_d30": {
        "denoise": 0.30,
        "custom_rain": True,
        "tr_st": -5,
        "scaffold_rel": VERSE_SCAFFOLD_REL,
        "voc_tgt_rel": VERSE_VOC_TGT_REL,
        "raw_orig_rel": VERSE_RAW_ORIG_REL,
        "cam_prompt": DATA_DIR / "ref2va_cal_rain_diner_camera.txt",
        "mic_prompt": DATA_DIR / "ref2va_cal_rain_diner_mic_dry.txt",
    },
    "c5_verse_m5_d30_s1": {
        "denoise": 0.30,
        "custom_rain": False,
        "tr_st": -5,
        "seed": SEED,
        "scaffold_rel": VERSE_SCAFFOLD_REL,
        "voc_tgt_rel": VERSE_VOC_TGT_REL,
        "raw_orig_rel": VERSE_RAW_ORIG_REL,
        "cam_prompt": DATA_DIR / "ref2va_cal_unplugged_v1.txt",
        "mic_prompt": None,
    },
    "c6_verse_m5_d30_s2": {
        "denoise": 0.30,
        "custom_rain": False,
        "tr_st": -5,
        "seed": SEED + 1000,
        "scaffold_rel": VERSE_SCAFFOLD_REL,
        "voc_tgt_rel": VERSE_VOC_TGT_REL,
        "raw_orig_rel": VERSE_RAW_ORIG_REL,
        "cam_prompt": DATA_DIR / "ref2va_cal_unplugged_v1.txt",
        "mic_prompt": None,
    },
    "c7_bridge_peak_m5_d30": {
        "denoise": 0.30,
        "custom_rain": False,
        "tr_st": -5,
        "scaffold_rel": BRIDGE_SCAFFOLD_REL,
        "voc_tgt_rel": BRIDGE_VOC_TGT_REL,
        "raw_orig_rel": BRIDGE_RAW_ORIG_REL,
        "cam_prompt": DATA_DIR / "ref2va_cal_unplugged_bridge.txt",
        "mic_prompt": None,
    },
    "c8_kite_chorus_d30": {
        "denoise": 0.30,
        "custom_rain": False,
        "tr_st": -6,
        "scaffold_rel": KITE_SCAFFOLD_REL,
        "voc_tgt_rel": KITE_VOC_TGT_REL,
        "raw_orig_rel": KITE_RAW_ORIG_REL,
        "cam_prompt": DATA_DIR / "ref2va_cal_kite_chorus.txt",
        "mic_prompt": None,
    },
    "c9_fridge_chorus_d30": {
        "denoise": 0.30,
        "custom_rain": False,
        "tr_st": 2,
        "scaffold_rel": FRIDGE_SCAFFOLD_REL,
        "voc_tgt_rel": FRIDGE_VOC_TGT_REL,
        "raw_orig_rel": FRIDGE_RAW_ORIG_REL,
        "cam_prompt": DATA_DIR / "ref2va_cal_fridge_chorus.txt",
        "mic_prompt": None,
    },
}
ALL_CAL_ARMS: tuple[str, ...] = tuple(CAL_ARMS.keys())


def _require_dir(env_name: str) -> Path:
    val = os.environ.get(env_name)
    if not val:
        raise SystemExit(f"{env_name} is unset; run `set -a; . ./h3.env; set +a` first")
    p = Path(val)
    if not p.is_dir():
        raise SystemExit(f"{env_name} does not exist: {p}")
    return p


def _ensure_assets(input_dir: Path) -> dict[str, Any]:
    stems_dir = input_dir / STEMS_SUBDIR
    if not (stems_dir / "vocals.wav").is_file():
        raise FileNotFoundError(f"Missing stems in {stems_dir}; run bench/h3_stem_sep.py stems first")
    kurt_ref = input_dir / REF_AUDIO
    if not kurt_ref.is_file():
        raise FileNotFoundError(f"Missing clean Kurt ref {kurt_ref}")

    verse_info = stem_sep.build_cover_scaffold(
        stems_dir,
        start_s=VERSE_CAL_START_S,
        duration_s=CAL_DUR_S,
        semitones=COVER_SEMITONES,
        vocal_formant=False,
        out_wav_32k=input_dir / VERSE_SCAFFOLD_REL,
        out_voc_wav_32k=input_dir / VERSE_VOC_TGT_REL,
    )
    chorus_info = stem_sep.build_cover_scaffold(
        stems_dir,
        start_s=CHORUS_CAL_START_S,
        duration_s=CAL_DUR_S,
        semitones=COVER_SEMITONES,
        vocal_formant=False,
        out_wav_32k=input_dir / CHORUS_SCAFFOLD_REL,
        out_voc_wav_32k=input_dir / CHORUS_VOC_TGT_REL,
    )
    bridge_info = stem_sep.build_cover_scaffold(
        stems_dir,
        start_s=BRIDGE_CAL_START_S,
        duration_s=CAL_DUR_S,
        semitones=COVER_SEMITONES,
        vocal_formant=False,
        out_wav_32k=input_dir / BRIDGE_SCAFFOLD_REL,
        out_voc_wav_32k=input_dir / BRIDGE_VOC_TGT_REL,
    )
    kite_info: dict[str, Any] = {}
    kite_stems = input_dir / "music_cover_stems" / "thatsnotakite"
    if (kite_stems / "vocals.wav").is_file():
        kite_info = stem_sep.build_cover_scaffold(
            kite_stems,
            start_s=KITE_CAL_START_S,
            duration_s=CAL_DUR_S,
            semitones=KITE_SEMITONES,
            harmonic_db=-120.0,
            bass_db=-11.0,
            vocal_formant=False,
            out_wav_32k=input_dir / KITE_SCAFFOLD_REL,
            out_voc_wav_32k=input_dir / KITE_VOC_TGT_REL,
        )
    fridge_info: dict[str, Any] = {}
    fridge_stems = input_dir / "music_cover_stems" / "inthefridge"
    if (fridge_stems / "vocals.wav").is_file():
        fridge_info = stem_sep.build_cover_scaffold(
            fridge_stems,
            start_s=FRIDGE_CAL_START_S,
            duration_s=CAL_DUR_S,
            semitones=FRIDGE_SEMITONES,
            harmonic_db=-120.0,
            bass_db=-11.0,
            vocal_formant=False,
            out_wav_32k=input_dir / FRIDGE_SCAFFOLD_REL,
            out_voc_wav_32k=input_dir / FRIDGE_VOC_TGT_REL,
        )
    song_info = stem_sep.build_cover_scaffold(
        stems_dir,
        start_s=SONG_START_S,
        duration_s=SONG_DUR_S,
        semitones=COVER_SEMITONES,
        harmonic_db=-14.5,
        bass_db=-12.0,
        vocal_formant=False,
        out_wav_32k=input_dir / SONG_SCAFFOLD_REL,
        out_voc_wav_32k=input_dir / SONG_VOC_TGT_REL,
    )
    raw_jobs: list[tuple[Path, str, float, float]] = [
        (stems_dir / "mix.wav", VERSE_RAW_ORIG_REL, VERSE_CAL_START_S, CAL_DUR_S),
        (stems_dir / "mix.wav", CHORUS_RAW_ORIG_REL, CHORUS_CAL_START_S, CAL_DUR_S),
        (stems_dir / "mix.wav", BRIDGE_RAW_ORIG_REL, BRIDGE_CAL_START_S, CAL_DUR_S),
        (stems_dir / "mix.wav", SONG_RAW_ORIG_REL, SONG_START_S, SONG_DUR_S),
    ]
    if (kite_stems / "mix.wav").is_file():
        raw_jobs.append((kite_stems / "mix.wav", KITE_RAW_ORIG_REL, KITE_CAL_START_S, CAL_DUR_S))
    if (fridge_stems / "mix.wav").is_file():
        raw_jobs.append((fridge_stems / "mix.wav", FRIDGE_RAW_ORIG_REL, FRIDGE_CAL_START_S, CAL_DUR_S))
    for mix_src, rel, ss, dur in raw_jobs:
        out_p = input_dir / rel
        if rel == SONG_RAW_ORIG_REL or not out_p.is_file():
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{dur:.3f}", "-i", str(mix_src), "-ar", "32000", "-ac", "2", str(out_p)],
                check=True,
            )
    return {
        "verse_cal": verse_info,
        "chorus_cal": chorus_info,
        "bridge_cal": bridge_info,
        "kite_cal": kite_info,
        "fridge_cal": fridge_info,
        "song": song_info,
    }


def _wire_sol(g: dict[str, Any], *, in_id: str, shift_id: str, attn_id: str, sol_id: str) -> str:
    g[shift_id] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": [in_id, 0], "shift_video": 12.0, "shift_audio": 3.0}}
    g[attn_id] = {"class_type": "ModelAttentionBackend", "inputs": {"model": [shift_id, 0], "attention": "comfy kitchen attention"}}
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


def _add_loaders_and_refs(g: dict[str, Any], *, length: int, two_pics: bool = True) -> tuple[str, str]:
    g["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": UNET_REF2VA, "weight_dtype": "default"}}
    g["2"] = {"class_type": "MiniMaxH3EncoderLoader", "inputs": {"encoder_name": ENCODER_BF16, "device": "default"}}
    g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}}
    g["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}}
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
    pika_sol = _wire_sol(g, in_id="18", shift_id="19", attn_id="58", sol_id="21")
    base_sol = _wire_sol(g, in_id="1", shift_id="119", attn_id="158", sol_id="121")

    g["15"] = {"class_type": "LoadImage", "inputs": {"image": PIC_FACE}}
    pic_policy = {"size_policy": "tiered", "qwen_view": "separate", "qwen_view.qwen_short_edge": 512}
    g["50"] = {"class_type": "MiniMaxH3AppendRefImage", "inputs": {"image": ["15", 0], **pic_policy}}
    last_ref = "50"
    if two_pics:
        g["16"] = {"class_type": "LoadImage", "inputs": {"image": PIC_STAGE}}
        g["51"] = {"class_type": "MiniMaxH3AppendRefImage", "inputs": {"image": ["16", 0], "references": ["50", 0], **pic_policy}}
        last_ref = "51"
    g["33"] = {"class_type": "LoadAudio", "inputs": {"audio": REF_AUDIO}}
    g["52"] = {"class_type": "MiniMaxH3AppendRefAudio", "inputs": {"audio": ["33", 0], "references": [last_ref, 0]}}
    g["132"] = {
        "class_type": "MiniMaxH3EncodeReferences",
        "inputs": {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "audio_vae": ["4", 0],
            "references": ["52", 0],
            "width": WIDTH,
            "height": HEIGHT,
            "length": length,
            "video_policy": "comfy",
            "image_policy": "comfy",
        },
    }
    return pika_sol, base_sol


def build_cal_graph(arm: str) -> dict[str, Any]:
    spec = CAL_ARMS[arm]
    g: dict[str, Any] = {}
    pika_sol, base_sol = _add_loaders_and_refs(g, length=CAL_LENGTH, two_pics=not spec["custom_rain"])
    cam_text = spec["cam_prompt"].read_text(encoding="utf-8").strip()

    g["5"] = {"class_type": "MiniMaxH3PromptOnReferences", "inputs": {"encoded_references": ["132", 0], "prompt": cam_text}}
    g["26"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": ["5", 0], "samples": ["132", 1]}}
    g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": spec["scaffold_rel"]}}
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

    qr1_inputs: dict[str, Any] = {
        "model": [pika_sol, 0],
        "section_split": "off" if spec["custom_rain"] else "soundscape_to_audio",
        "video_block_band": "all",
        "audio_block_band": "all",
        "start_percent": 0.0,
        "end_percent": 1.0,
    }
    if spec["custom_rain"]:
        mic_text = spec["mic_prompt"].read_text(encoding="utf-8").strip()
        g["105"] = {"class_type": "MiniMaxH3PromptOnReferences", "inputs": {"encoded_references": ["132", 0], "prompt": mic_text}}
        qr1_inputs["audio_conditioning"] = ["105", 0]
        qr1_inputs["aud_to_vid_gain"] = 0.0

    seed = int(spec.get("seed", SEED))
    g["45"] = {"class_type": "MiniMaxH3QueryRouter", "inputs": qr1_inputs}
    g["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["7"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}}
    g["60"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": PIKA_SIGMAS}}
    g["9"] = {"class_type": "BasicGuider", "inputs": {"model": ["45", 0], "conditioning": ["26", 0]}}
    g["10"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0], "sigmas": ["60", 0], "latent_image": ["49", 0]},
    }

    # Pass 2: Base ref2va 12-step euler/simple + AudioRefineMask(0, 1)
    g["152"] = {"class_type": "MiniMaxH3AudioRefineMask", "inputs": {"latent": ["10", 0], "video_mask": 0.0, "audio_mask": 1.0}}
    g["106"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + 17}}
    g["107"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    if spec["custom_rain"]:
        # Recipe 7: omit FrozenVideoCache and wire dry Mic Prompt B to BOTH Sampler & QueryRouter(a2v=0.0)
        g["126"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": ["105", 0], "samples": ["132", 1]}}
        g["145"] = {
            "class_type": "MiniMaxH3QueryRouter",
            "inputs": {
                "model": [base_sol, 0],
                "section_split": "off",
                "video_block_band": "all",
                "audio_block_band": "all",
                "aud_to_vid_gain": 0.0,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "audio_conditioning": ["126", 0],
            },
        }
        pass2_model, pass2_cond = "145", "126"
    else:
        pass2_model, pass2_cond = base_sol, "26"

    g["160"] = {
        "class_type": "BasicScheduler",
        "inputs": {"model": [pass2_model, 0], "scheduler": "simple", "steps": 12, "denoise": float(spec["denoise"])},
    }
    g["109"] = {"class_type": "BasicGuider", "inputs": {"model": [pass2_model, 0], "conditioning": [pass2_cond, 0]}}
    g["110"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {"noise": ["106", 0], "guider": ["109", 0], "sampler": ["107", 0], "sigmas": ["160", 0], "latent_image": ["152", 0]},
    }
    g["30"] = {
        "class_type": "MiniMaxH3ProvenanceStamp",
        "inputs": {"latent": ["110", 0], "model": [pass2_model, 0], "sigmas": ["160", 0], "clip": ["2", 0], "note": arm},
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


def build_song_graph(denoise: float) -> dict[str, Any]:
    g: dict[str, Any] = {}
    pika_sol, base_sol = _add_loaders_and_refs(g, length=SONG_WIN_FRAMES, two_pics=True)
    g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": SONG_SCAFFOLD_REL}}
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
    g["7"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}}
    g["60"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": PIKA_SIGMAS}}
    g["107"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    g["160"] = {
        "class_type": "BasicScheduler",
        "inputs": {"model": [base_sol, 0], "scheduler": "simple", "steps": 12, "denoise": float(denoise)},
    }

    for k in range(1, SONG_N_WINDOWS + 1):
        sfx = f"w{k}"
        prompt_text = (DATA_DIR / f"ref2va_song_{sfx}.txt").read_text(encoding="utf-8").strip()
        start_s = round((k - 1) * SONG_STRIDE_S, 3)

        g[f"5_{sfx}"] = {"class_type": "MiniMaxH3PromptOnReferences", "inputs": {"encoded_references": ["132", 0], "prompt": prompt_text}}
        g[f"26_{sfx}"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": [f"5_{sfx}", 0], "samples": ["132", 1]}}

        fw_inputs: dict[str, Any] = {
            "latent": [f"26_{sfx}", 1],
            "audio_vae": ["4", 0],
            "audio": ["48", 0],
            "start_seconds": start_s,
            "context_frames": SONG_CTX_FRAMES,
            "audio_mask": 0.0,
            "level": "clip_guard",
            "context_noise": 0.35,
            "context_profile": "seam_taper",
        }
        pass2_cond = f"26_{sfx}"
        if k > 1:
            prev_sfx = f"w{k - 1}"
            # Overlap continuity: prepend Window (k-1)'s 1.625s generated Kurt tail onto Window k's
            # stem scaffold via AudioConcat on 100% stock RoPE (no MiniMaxH3GuideLatent collision).
            g[f"trim_prev_{sfx}"] = {
                "class_type": "TrimAudioDuration",
                "inputs": {"audio": [f"112_{prev_sfx}", 0], "start_index": round(SONG_STRIDE_S, 3), "duration": 1.625},
            }
            g[f"trim_new_{sfx}"] = {
                "class_type": "TrimAudioDuration",
                "inputs": {"audio": ["48", 0], "start_index": round(start_s + 1.625, 3), "duration": round(SONG_STRIDE_S, 3)},
            }
            g[f"hyb_aud_{sfx}"] = {
                "class_type": "AudioConcat",
                "inputs": {"audio1": [f"trim_prev_{sfx}", 0], "audio2": [f"trim_new_{sfx}", 0], "direction": "after"},
            }
            fw_inputs["previous"] = [f"110_{prev_sfx}", 0]
            fw_inputs["audio"] = [f"hyb_aud_{sfx}", 0]
            fw_inputs["start_seconds"] = 0.0

        g[f"49_{sfx}"] = {"class_type": "MiniMaxH3FreezeAudioWindow", "inputs": fw_inputs}

        g[f"6_{sfx}"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED + (k - 1)}}
        g[f"9_{sfx}"] = {"class_type": "BasicGuider", "inputs": {"model": ["45", 0], "conditioning": [f"26_{sfx}", 0]}}
        g[f"10_{sfx}"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": [f"6_{sfx}", 0],
                "guider": [f"9_{sfx}", 0],
                "sampler": ["7", 0],
                "sigmas": ["60", 0],
                "latent_image": [f"49_{sfx}", 0],
            },
        }
        g[f"152_{sfx}"] = {"class_type": "MiniMaxH3AudioRefineMask", "inputs": {"latent": [f"10_{sfx}", 0], "video_mask": 0.0, "audio_mask": 1.0}}
        g[f"106_{sfx}"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED + 17 + (k - 1)}}
        g[f"109_{sfx}"] = {"class_type": "BasicGuider", "inputs": {"model": [base_sol, 0], "conditioning": [pass2_cond, 0]}}
        g[f"110_{sfx}"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": [f"106_{sfx}", 0],
                "guider": [f"109_{sfx}", 0],
                "sampler": ["107", 0],
                "sigmas": ["160", 0],
                "latent_image": [f"152_{sfx}", 0],
            },
        }
        g[f"111_{sfx}"] = {"class_type": "VAEDecode", "inputs": {"samples": [f"110_{sfx}", 0], "vae": ["3", 0]}}
        g[f"112_{sfx}"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": [f"110_{sfx}", 0], "vae": ["4", 0]}}
        g[f"113_{sfx}"] = {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": [f"111_{sfx}", 0],
                "audio": [f"112_{sfx}", 0],
                "frame_rate": 24.0,
                "loop_count": 0,
                "filename_prefix": f"{OUT_PREFIX_DIR}/song_{sfx}",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        }
        g[f"114_{sfx}"] = {
            "class_type": "SaveAudio",
            "inputs": {"audio": [f"112_{sfx}", 0], "filename_prefix": f"{OUT_PREFIX_DIR}/song_aud_{sfx}"},
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
            for line in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True).splitlines():
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


def _load_audio_np(p: Path, sr: int = 24000, channels: int = 1) -> np.ndarray:
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(p), "-vn", "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"])
    arr = np.frombuffer(raw, dtype=np.float32).copy()
    return arr if channels == 1 else arr.reshape(-1, channels).T


def _chroma(y: np.ndarray, sr: int = 24000) -> np.ndarray:
    win = torch.hann_window(4096)
    spec = torch.stft(torch.from_numpy(y), n_fft=4096, hop_length=600, window=win, return_complex=True).abs().numpy() ** 2
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
    win = torch.hann_window(1024)
    def flux(y: np.ndarray) -> np.ndarray:
        S = np.log1p(torch.stft(torch.from_numpy(y), n_fft=1024, hop_length=600, window=win, return_complex=True).abs().numpy())
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


def _voicing_gap_breakdown(
    out_voc_24k: torch.Tensor,
    tgt_voc_24k: torch.Tensor,
    *,
    relaxed_thr: float = 0.28,
) -> dict[str, Any]:
    """Partition unvoiced target frames and score relaxed YIN (raspy fry) + 0.25 st sweep."""
    fo_s, cmndf_o, rms_o = (t.numpy() for t in yin_probe.yin_f0(out_voc_24k, sr=24000))
    ft_s = yin_probe.yin_f0(tgt_voc_24k, sr=24000)[0].numpy()
    fo_rel = yin_probe.yin_f0(out_voc_24k, sr=24000, threshold=relaxed_thr)[0].numpy()

    n = min(len(fo_s), len(ft_s), len(fo_rel))
    fo_s, cmndf_o, rms_o, ft_s, fo_rel = fo_s[:n], cmndf_o[:n], rms_o[:n], ft_s[:n], fo_rel[:n]
    tgt_v = np.isfinite(ft_s) & (ft_s > 0)
    out_v = np.isfinite(fo_s) & (fo_s > 0)
    n_tgt = int(np.sum(tgt_v))
    gap = tgt_v & (~out_v)
    n_gap = int(np.sum(gap))

    silent_m = gap & (rms_o < yin_probe.DEFAULT_MIN_DBFS)
    raspy_m = gap & (rms_o >= yin_probe.DEFAULT_MIN_DBFS) & (cmndf_o < relaxed_thr)
    aper_m = gap & (rms_o >= yin_probe.DEFAULT_MIN_DBFS) & (cmndf_o >= relaxed_thr)
    n_sil, n_raspy, n_aper = int(np.sum(silent_m)), int(np.sum(raspy_m)), int(np.sum(aper_m))

    rel_score = pitch_lock_metrics.score_f0(fo_rel, ft_s)
    rel_null = pitch_lock_metrics.lag_null(fo_rel, ft_s)
    rel_cov = float(np.sum(tgt_v & np.isfinite(fo_rel) & (fo_rel > 0))) / max(1, n_tgt)

    out: dict[str, Any] = {
        "n_tgt_voiced": n_tgt,
        "n_gap": n_gap,
        "below_min_dbfs": n_sil,
        "cmndf_015_to_028": n_raspy,
        "cmndf_ge_028": n_aper,
        "below_min_dbfs_frac_of_tgt": round(n_sil / max(1, n_tgt), 3),
        "cmndf_015_to_028_frac_of_tgt": round(n_raspy / max(1, n_tgt), 3),
        "cmndf_ge_028_frac_of_tgt": round(n_aper / max(1, n_tgt), 3),
        "relaxed_thr": float(relaxed_thr),
        "relaxed_voiced_overlap_frac": round(rel_cov, 3),
        "relaxed_med_abs_cent_err": round(rel_score["med_abs_cent_err"], 1) if rel_score else None,
        "relaxed_in_tune_frac_50c": round(rel_score["in_tune_frac_50c"], 3) if rel_score else None,
        "relaxed_best_tr_st": int(rel_score["best_tr_st"]) if rel_score else None,
        "relaxed_vocal_f0_r_one_shift": round(rel_score["vocal_f0_r_one_shift"], 3) if rel_score else None,
        "relaxed_null_best_cent_err": rel_null["med_abs_cent_err"]["null_best"] if rel_null else None,
        "relaxed_beats_frac": rel_null["med_abs_cent_err"]["beats_frac"] if rel_null else None,
        "relaxed_beats_all_shifts": bool(rel_null and rel_null["med_abs_cent_err"]["beats_frac"] == 1.0),
    }

    shared = tgt_v & out_v
    if int(np.sum(shared)) >= pitch_lock_metrics.MIN_SHARED_FRAMES:
        raw_cents = 1200.0 * np.log2(fo_s[shared] / ft_s[shared])
        best_frac_st, best_frac_in50 = 0.0, -1.0
        for s_st in np.linspace(-6.0, 6.0, 49):
            in50 = float(np.mean(np.abs(((raw_cents - 100.0 * s_st + 600.0) % 1200.0) - 600.0) <= 50.0))
            if in50 > best_frac_in50 + 1e-12 or (abs(in50 - best_frac_in50) <= 1e-12 and abs(s_st) < abs(best_frac_st)):
                best_frac_st, best_frac_in50 = float(s_st), in50
        best_int_st, best_int_in50 = 0, -1.0
        for k_st in pitch_lock_metrics.TRANSPOSITIONS:
            in50 = float(np.mean(np.abs(((raw_cents - 100.0 * k_st + 600.0) % 1200.0) - 600.0) <= 50.0))
            if in50 > best_int_in50 + 1e-12 or (abs(in50 - best_int_in50) <= 1e-12 and abs(k_st) < abs(best_int_st)):
                best_int_st, best_int_in50 = int(k_st), in50
        out["best_in50_shift_st"] = round(best_frac_st, 2)
        out["best_in50_frac"] = round(best_frac_in50, 3)
        out["best_int_in50_st"] = best_int_st
        out["best_int_in50_frac"] = round(best_int_in50, 3)
    else:
        out["best_in50_shift_st"] = None
        out["best_in50_frac"] = None
        out["best_int_in50_st"] = None
        out["best_int_in50_frac"] = None
    return out


def _vocal_pitch_metrics(out_voc_24k: torch.Tensor, tgt_voc_24k: torch.Tensor) -> dict[str, Any]:
    base = pitch_lock_metrics.vocal_pitch_metrics(out_voc_24k, tgt_voc_24k)
    ln = base.get("lag_null")
    base["beats_all_shifts"] = bool(ln and ln["med_abs_cent_err"]["beats_frac"] == 1.0)
    base["voicing_gap"] = _voicing_gap_breakdown(out_voc_24k, tgt_voc_24k)
    return base


def _score_clip(
    clip_path: Path,
    *,
    src_mix_path: Path,
    tgt_voc_path: Path,
    kurt_ref_y: np.ndarray,
    stem_out_wav: Path,
    base_tr_st: int,
    output_dir: Path,
) -> dict[str, Any]:
    y_mix = _load_audio_np(clip_path, sr=24000, channels=1)
    sx_mix = _load_audio_np(src_mix_path, sr=24000, channels=1)
    c_mix, c_sx = _chroma(y_mix), _chroma(sx_mix)
    cx0, lx0, _ = _chroma_sim(c_mix, c_sx, any_key=False, base_tr_st=base_tr_st)
    cxT, _, tr = _chroma_sim(c_mix, c_sx, any_key=True, base_tr_st=base_tr_st)
    ox, olx = _onset_corr(y_mix, sx_mix)

    if stem_out_wav.is_file() and stem_out_wav.stat().st_size > 44100:
        out_voc_24k = torch.from_numpy(_load_audio_np(stem_out_wav, sr=24000, channels=1))
    else:
        out_44k = torch.from_numpy(_load_audio_np(clip_path, sr=44100, channels=2))
        dev = stem_sep._resolve_compute_device(None)
        voc_44k, _ = stem_sep.separate_vocals_roformer(out_44k, dev)
        stem_sep._write_wav_pcm16(stem_out_wav, voc_44k, 44100)
        out_voc_24k = torchaudio.functional.resample(voc_44k.mean(dim=0, keepdim=True), 44100, 24000).squeeze(0)
    tgt_voc_24k = torch.from_numpy(_load_audio_np(tgt_voc_path, sr=24000, channels=1))
    v_metrics = _vocal_pitch_metrics(out_voc_24k, tgt_voc_24k)
    return {
        "file": str(clip_path.relative_to(output_dir)),
        "target_tr_st": base_tr_st,
        "chroma_sameKey_x": round(cx0, 3),
        "chroma_lag_x": lx0,
        "chroma_anyKey_x": round(cxT, 3),
        "chroma_drift_st": tr,
        "onset_x": round(ox, 3),
        "onset_lag_x": olx,
        "ltas_cos_kurt_unplugged": round(_ltas_cos(y_mix, kurt_ref_y), 3),
        **v_metrics,
    }


def _decode_rgb_frames(mp4_path: Path, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(mp4_path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    n = len(raw) // (height * width * 3)
    return np.frombuffer(raw, dtype=np.uint8).copy().reshape(n, height, width, 3)


def stitch_pure_h3_windows(output_dir: Path, window_mp4s: list[Path], window_flacs: list[Path] | None = None) -> dict[str, Any]:
    """Concatenate decoded H3 video frames (dropping trim=39 overlap head on w2..w4)
    and crossfade decoded H3 stereo audio (`300 ms` equal-power cosine) with zero external audio mux.
    """
    out_dir = output_dir / OUT_PREFIX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "kurt_unplugged_justaflicker_verse_chorus_44s-audio.mp4"

    vid_chunks: list[np.ndarray] = []
    seam_metrics: list[dict[str, Any]] = []
    local_means: list[float] = []
    for idx, mp4 in enumerate(window_mp4s):
        frames = _decode_rgb_frames(mp4)
        chunk = frames if idx == 0 else frames[SONG_CTX_FRAMES:]
        adj = np.mean(np.abs(chunk[1:].astype(np.float32) - chunk[:-1].astype(np.float32)), axis=(1, 2, 3))
        local_means.append(float(np.mean(adj)))
        if vid_chunks:
            prev_last = vid_chunks[-1][-1].astype(np.float32)
            cur_first = chunk[0].astype(np.float32)
            seam_l1 = float(np.mean(np.abs(cur_first - prev_last)))
            base_adj = float(np.mean(adj[:10]))
            seam_metrics.append({
                "join": f"w{idx}->w{idx + 1}",
                "frame_transition": f"f{sum(len(c) for c in vid_chunks) - 1}->f{sum(len(c) for c in vid_chunks)}",
                "seam_l1": round(seam_l1, 3),
                "post_10f_adj_mean": round(base_adj, 3),
                "seam_ratio": round(seam_l1 / max(base_adj, 1e-6), 3),
            })
        vid_chunks.append(chunk)

    all_frames = np.concatenate(vid_chunks, axis=0)
    assert len(all_frames) == SONG_TOTAL_FRAMES, f"Expected {SONG_TOTAL_FRAMES} frames, got {len(all_frames)}"

    # Join H3's decoded 32 kHz audio via centered 300 ms equal-power cosine crossfade over the 1.625 s overlap
    sr = 32000
    overlap_samps = int(round(SONG_CTX_FRAMES / 24.0 * sr))  # 52,000
    stride_samps = int(round(SONG_STRIDE_FRAMES / 24.0 * sr))  # 340,000
    win_samps = int(round(SONG_WIN_FRAMES / 24.0 * sr))  # 392,000
    total_samps = int(round(SONG_TOTAL_FRAMES / 24.0 * sr))  # 1,412,000
    xfade_samps = int(round(0.300 * sr))  # 9,600
    xfade_offset = (overlap_samps - xfade_samps) // 2  # centered inside the 1.625 s context overlap

    audio_sources = window_flacs if (window_flacs and len(window_flacs) == len(window_mp4s)) else window_mp4s
    stitched = np.zeros((2, total_samps), dtype=np.float32)
    t = np.linspace(0.0, math.pi / 2.0, xfade_samps, dtype=np.float32)
    fade_in = np.sin(t)[None, :]
    fade_out = np.cos(t)[None, :]

    for idx, ap in enumerate(audio_sources):
        a = _load_audio_np(ap, sr=sr, channels=2)
        if a.shape[1] < win_samps:
            a = np.pad(a, ((0, 0), (0, win_samps - a.shape[1])))
        else:
            a = a[:, :win_samps]
        pos = idx * stride_samps
        if idx == 0:
            stitched[:, :win_samps] = a
        else:
            # Hold previous window through pos + xfade_offset, crossfade over xfade_samps, then pure current window
            x0 = pos + xfade_offset
            x1 = x0 + xfade_samps
            stitched[:, x0:x1] = stitched[:, x0:x1] * fade_out + a[:, xfade_offset : xfade_offset + xfade_samps] * fade_in
            stitched[:, x1 : pos + win_samps] = a[:, xfade_offset + xfade_samps :]

    tmp_wav = out_dir / "_stitched_h3_audio.wav"
    stem_sep._write_wav_pcm16(tmp_wav, torch.from_numpy(stitched), sr)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", "24", "-i", "pipe:0",
        "-i", str(tmp_wav),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    proc.stdin.write(all_frames.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg stitch failed with code {rc}")
    tmp_wav.unlink(missing_ok=True)
    return {
        "stitched_mp4": str(out_mp4.relative_to(output_dir)),
        "frames": int(len(all_frames)),
        "duration_s": round(len(all_frames) / 24.0, 3),
        "audio_samples_32k": int(total_samps),
        "seams": seam_metrics,
        "mean_adj_l1": round(float(np.mean(local_means)), 3),
    }


def _find_latest(vid_dir: Path, prefix: str, ext: str) -> Path | None:
    m = sorted(vid_dir.glob(f"{prefix}*{ext}"))
    return m[-1] if m else None


def _metric_controls(input_dir: Path, stems_out: Path | None = None) -> dict[str, Any]:
    """Positive AudioVAE round-trip + Negative unrelated-song meter calibration controls."""
    rt_voc_wav = (stems_out / "audiovae_roundtrip_voc.wav") if stems_out is not None else None
    if rt_voc_wav is not None and rt_voc_wav.is_file() and rt_voc_wav.stat().st_size > 44100:
        voc_rt_24k = torch.from_numpy(_load_audio_np(rt_voc_wav, sr=24000, channels=1))
    else:
        import comfy.sd
        from safetensors.torch import load_file

        if not stem_sep._comfy_queue_busy():
            stem_sep._free_comfy_vram()
        vae_path = Path(os.environ["H3_MODEL_STORE"]) / "Comfy-Org_MiniMax-H3" / "vae" / "minimax_h3_audio_vae_fp32.safetensors"
        if not vae_path.is_file():
            vae_path = REPO.parents[1] / "models" / "vae" / "minimax_h3_audio_vae_fp32.safetensors"
        vae = comfy.sd.VAE(sd=load_file(str(vae_path)))
        scaf_32k = torch.from_numpy(_load_audio_np(input_dir / VERSE_SCAFFOLD_REL, sr=32000, channels=2)).unsqueeze(0)
        with torch.no_grad():
            z = vae.encode(scaf_32k.movedim(1, -1))
            rec = vae.decode(z)
        if rec.shape[1] != 2 and rec.shape[-1] == 2:
            rec = rec.movedim(-1, 1)
        rec_44k = torchaudio.functional.resample(rec[0].cpu().float(), 32000, 44100)
        del vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dev = stem_sep._resolve_compute_device(None)
        voc_rt_44k, _ = stem_sep.separate_vocals_roformer(rec_44k, dev)
        if rt_voc_wav is not None:
            stem_sep._write_wav_pcm16(rt_voc_wav, voc_rt_44k, 44100)
        voc_rt_24k = torchaudio.functional.resample(voc_rt_44k.mean(dim=0, keepdim=True), 44100, 24000).squeeze(0)
    tgt_voc_24k = torch.from_numpy(_load_audio_np(input_dir / VERSE_VOC_TGT_REL, sr=24000, channels=1))
    pos = _vocal_pitch_metrics(voc_rt_24k, tgt_voc_24k)

    neg_pool = [
        ("flicker_verse", input_dir / VERSE_VOC_TGT_REL),
        ("flicker_chorus", input_dir / CHORUS_VOC_TGT_REL),
        ("kite_chorus", input_dir / KITE_VOC_TGT_REL),
        ("fridge_chorus", input_dir / FRIDGE_VOC_TGT_REL),
    ]
    avail = [(label, p) for label, p in neg_pool if p.is_file()]
    pairs: dict[str, Any] = {}
    raw_errs, rel_errs, raw_in50, rel_in50 = [], [], [], []
    for idx_a in range(len(avail)):
        for idx_b in range(idx_a + 1, len(avail)):
            la, pa = avail[idx_a]
            lb, pb = avail[idx_b]
            m = _vocal_pitch_metrics(
                torch.from_numpy(_load_audio_np(pa, sr=24000, channels=1)),
                torch.from_numpy(_load_audio_np(pb, sr=24000, channels=1)),
            )
            pairs[f"{la}_vs_{lb}"] = m
            if m.get("med_abs_cent_err") is not None:
                raw_errs.append(m["med_abs_cent_err"])
                rel_errs.append(m["rel_med_abs_cent_err"])
                raw_in50.append(m["in_tune_frac_50c"])
                rel_in50.append(m["rel_in_tune_frac_50c"])

    summary_6x = {
        "n_pairs": len(pairs),
        "med_raw_cent_err": round(float(np.median(raw_errs)), 1) if raw_errs else None,
        "med_best_tr_cent_err": round(float(np.median(rel_errs)), 1) if rel_errs else None,
        "mean_raw_in50c": round(float(np.mean(raw_in50)), 3) if raw_in50 else None,
        "mean_best_tr_in50c": round(float(np.mean(rel_in50)), 3) if rel_in50 else None,
        "pairs": pairs,
    }
    null_rep_p = DATA_DIR / "stem_nulltest_report.json"
    bed_null = json.loads(null_rep_p.read_text(encoding="utf-8")) if null_rep_p.is_file() else {}
    return {
        "audiovae_roundtrip": pos,
        "unrelated_pairs_6x": summary_6x,
        "bed_nulltest": {k: {"mix_db": v["mix_null"]["null_depth_db"], "bed_db": v["bed_null"]["null_depth_db"]} for k, v in bed_null.items()},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--song", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--song-denoise", type=float, default=0.29)
    ap.add_argument("--arms", nargs="+", choices=ALL_CAL_ARMS, default=list(ALL_CAL_ARMS))
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args(argv)

    input_dir = _require_dir("H3_INPUT_DIR")
    output_dir = _require_dir("H3_OUTPUT_DIR")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    _ensure_assets(input_dir)
    do_cal = args.calibrate or (args.dry_run and not args.song)
    do_song = args.song

    if do_cal:
        cal_graphs: dict[str, dict[str, Any]] = {}
        for arm in args.arms:
            g = build_cal_graph(arm)
            gp = GRAPHS_DIR / f"{arm}.json"
            gp.write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")
            rc, out = _preflight(gp)
            print(f"[preflight] {arm}: {'PASS' if rc == 0 else f'FAIL({rc})'}")
            if rc != 0:
                print(out)
                return 1
            cal_graphs[arm] = g

        if args.calibrate:
            cid = str(uuid.uuid4())
            timings: dict[str, float] = {}
            tp = DATA_DIR / "cal_timings.json"
            if tp.is_file():
                timings = json.loads(tp.read_text(encoding="utf-8"))
            for arm in args.arms:
                _wait_idle(args.url)
                print(f"\n[run] queuing {arm}...", flush=True)
                t0 = time.monotonic()
                pid = comfy_client.submit_prompt(cal_graphs[arm], base_url=args.url, client_id=cid)
                hist = comfy_client.wait_for_prompt(pid, base_url=args.url, timeout=2400.0)
                dt = comfy_client.execution_duration_s(hist) or (time.monotonic() - t0)
                timings[arm] = round(dt, 2)
                vids = _extract_videos(hist, output_dir)
                print(f"[run] {arm} finished in {dt:.1f}s -> {[v.name for v in vids]}", flush=True)
            tp.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")

    if do_song:
        sg = build_song_graph(args.song_denoise)
        sgp = GRAPHS_DIR / f"song_4win_d{int(round(args.song_denoise * 100))}.json"
        sgp.write_text(json.dumps(sg, indent=2) + "\n", encoding="utf-8")
        rc, out = _preflight(sgp)
        print(f"[preflight] song_4win: {'PASS' if rc == 0 else f'FAIL({rc})'}")
        if rc != 0:
            print(out)
            return 1
        if not args.dry_run:
            _wait_idle(args.url)
            print(f"\n[run] queuing 4-window pure-H3 song (denoise={args.song_denoise})...", flush=True)
            t0 = time.monotonic()
            pid = comfy_client.submit_prompt(sg, base_url=args.url, client_id=str(uuid.uuid4()))
            hist = comfy_client.wait_for_prompt(pid, base_url=args.url, timeout=5400.0)
            dt = comfy_client.execution_duration_s(hist) or (time.monotonic() - t0)
            print(f"[run] song_4win finished in {dt:.1f}s", flush=True)
            vid_dir = output_dir / OUT_PREFIX_DIR
            w_mp4s = [_find_latest(vid_dir, f"song_w{k}_", "-audio.mp4") for k in range(1, SONG_N_WINDOWS + 1)]
            w_flacs = [_find_latest(vid_dir, f"song_aud_w{k}_", ".flac") for k in range(1, SONG_N_WINDOWS + 1)]
            if any(p is None for p in w_mp4s):
                raise RuntimeError(f"Missing window MP4s: {w_mp4s}")
            stitch_info = stitch_pure_h3_windows(
                output_dir,
                [p for p in w_mp4s if p is not None],
                [p for p in w_flacs if p is not None] if all(w_flacs) else None,
            )
            stitch_info["wall_s"] = round(dt, 2)
            stitch_info["denoise"] = float(args.song_denoise)
            (DATA_DIR / "song_stitch_meta.json").write_text(json.dumps(stitch_info, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(stitch_info, indent=2))

    if args.analyze or args.calibrate or (args.song and not args.dry_run):
        vid_dir = output_dir / OUT_PREFIX_DIR
        stems_out = vid_dir / "vocal_stems"
        stems_out.mkdir(parents=True, exist_ok=True)
        kurt_ref_y = _load_audio_np(input_dir / REF_AUDIO, sr=24000, channels=1)

        if vid_dir.is_dir():
            cal_rep: dict[str, Any] = {"arms": {}}
            for arm in args.arms:
                mp4 = _find_latest(vid_dir, f"{arm}_", "-audio.mp4")
                if mp4 is not None:
                    spec = CAL_ARMS[arm]
                    cal_rep["arms"][arm] = _score_clip(
                        mp4,
                        src_mix_path=input_dir / spec["raw_orig_rel"],
                        tgt_voc_path=input_dir / spec["voc_tgt_rel"],
                        kurt_ref_y=kurt_ref_y,
                        stem_out_wav=stems_out / f"{arm}_voc.wav",
                        base_tr_st=int(spec.get("tr_st", int(COVER_SEMITONES))),
                        output_dir=output_dir,
                    )
            if (DATA_DIR / "cal_timings.json").is_file():
                cal_rep["timings_s"] = json.loads((DATA_DIR / "cal_timings.json").read_text(encoding="utf-8"))
            if cal_rep["arms"]:
                cal_rep["controls"] = _metric_controls(input_dir, stems_out)
                (DATA_DIR / "cal_triage_report.json").write_text(json.dumps(cal_rep, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(cal_rep, indent=2))

            song_mp4 = vid_dir / "kurt_unplugged_justaflicker_verse_chorus_44s-audio.mp4"
            if song_mp4.is_file():
                song_metrics = _score_clip(
                    song_mp4,
                    src_mix_path=input_dir / SONG_RAW_ORIG_REL,
                    tgt_voc_path=input_dir / SONG_VOC_TGT_REL,
                    kurt_ref_y=kurt_ref_y,
                    stem_out_wav=stems_out / "song_44s_voc.wav",
                    base_tr_st=int(COVER_SEMITONES),
                    output_dir=output_dir,
                )
                # Score individual 294f windows (w1..w4) from MelBand-RoFormer stems vs 12.25s target slices
                win_reports: dict[str, Any] = {}
                tgt_full_24k = _load_audio_np(input_dir / SONG_VOC_TGT_REL, sr=24000, channels=1)
                raw_full_24k = _load_audio_np(input_dir / SONG_RAW_ORIG_REL, sr=24000, channels=1)
                stride_24k = int(round(SONG_STRIDE_S * 24000))
                win_24k = int(round(SONG_WIN_S * 24000))
                for k in range(1, SONG_N_WINDOWS + 1):
                    w_flac = _find_latest(vid_dir, f"song_aud_w{k}_", ".flac")
                    w_mp4 = _find_latest(vid_dir, f"song_w{k}_", "-audio.mp4")
                    src_audio = w_flac or w_mp4
                    if src_audio is None:
                        continue
                    w_stem = stems_out / f"song_w{k}_voc.wav"
                    if not (w_stem.is_file() and w_stem.stat().st_size > 44100):
                        w_44k = torch.from_numpy(_load_audio_np(src_audio, sr=44100, channels=2))
                        dev = stem_sep._resolve_compute_device(None)
                        voc_w_44k, _ = stem_sep.separate_vocals_roformer(w_44k, dev)
                        stem_sep._write_wav_pcm16(w_stem, voc_w_44k, 44100)
                    w_voc_24k = torch.from_numpy(_load_audio_np(w_stem, sr=24000, channels=1))
                    s0 = (k - 1) * stride_24k
                    tgt_slice = torch.from_numpy(tgt_full_24k[s0 : s0 + win_24k].copy())
                    raw_slice = raw_full_24k[s0 : s0 + win_24k]
                    w_mix_24k = _load_audio_np(src_audio, sr=24000, channels=1)
                    cx0, lx0, _ = _chroma_sim(_chroma(w_mix_24k), _chroma(raw_slice), any_key=False, base_tr_st=int(COVER_SEMITONES))
                    cxT, _, tr = _chroma_sim(_chroma(w_mix_24k), _chroma(raw_slice), any_key=True, base_tr_st=int(COVER_SEMITONES))
                    win_reports[f"w{k}"] = {
                        "file": str(src_audio.relative_to(output_dir)),
                        "stem_voc": str(w_stem.relative_to(output_dir)),
                        "chroma_sameKey_x": round(cx0, 3),
                        "chroma_anyKey_x": round(cxT, 3),
                        "chroma_drift_st": tr,
                        **_vocal_pitch_metrics(w_voc_24k, tgt_slice),
                    }
                if win_reports:
                    song_metrics["windows"] = win_reports
                if (DATA_DIR / "song_stitch_meta.json").is_file():
                    song_metrics.update(json.loads((DATA_DIR / "song_stitch_meta.json").read_text(encoding="utf-8")))
                (DATA_DIR / "song_report.json").write_text(json.dumps(song_metrics, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(song_metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
