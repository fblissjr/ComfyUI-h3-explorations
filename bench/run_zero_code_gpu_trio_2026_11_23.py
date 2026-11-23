#!/usr/bin/env python3
"""Run and analyze the Zero-Code GPU Trio on live ComfyUI (:8188).

Three zero-code-change architectural experiments (8 submissions total, single
shared `client_id` for zero-swap/zero-reload execution caching within families):

  Trio 1 (`fl2va` 6-step Pika, 1152x768 @ 141f, seed=4409, fresh street bucket drummer):
    - `trio1a_freeze_mask000`: `MiniMaxH3FreezeAudio(audio_mask=0.0)`
    - `trio1b_freeze_mask025`: `MiniMaxH3FreezeAudio(audio_mask=0.25)` (+ companion decoded latent audio)
    - `trio1c_joint_audiorefine_cache`: Joint Pass-1 Pika (`trio1c_pass1_pika6`) feeding
      Pass-2 undistilled `MiniMaxH3FrozenVideoCache(int4)` + `MiniMaxH3AudioRefineMask`
      (`trio1c_pass2_audiorefine`)

  Trio 2 (Zero-swap `P-H3-Image LoRA` Tier 2 + `MiniMaxH3GuideLatent` @ f068, 1152x768 @ 141f, seed=5817):
    - Fresh Victorian glasshouse botanist -> stormy brass grow-lamp twilight transition
    - `trio2a_guide_end_100`: `MiniMaxH3ReleaseReferences(guide_end_percent=1.00)` (+ saved anchors)
    - `trio2b_guide_end_033`: `MiniMaxH3ReleaseReferences(guide_end_percent=0.33)`

  Trio 3 (M3 `ref2va` on `woman_blonde.jpg`, `size_policy="tiered"`, 1152x768 @ 141f, seed=6923):
    - Fresh crouched-blocks-to-upright-sprint 6-section prompt with spoken dialogue
    - `trio3a_ref_end100_clean999`: `end_percent=1.00, clean_level=0.999`
    - `trio3b_ref_end033_clean999`: `end_percent=0.33, clean_level=0.999`
    - `trio3c_ref_end050_clean900`: `end_percent=0.50, clean_level=0.900`

Also synthesizes `$H3_INPUT_DIR/h3_trio_groove_20261123.wav` (32 kHz stereo 118 BPM
percussive groove), generates `.1fps` review strips for native inspection, computes
motion/MAE/audio metrics, and runs `probe_dinov3_copy_freeze.py` + `probe_yin_f0_drift.py`
on Trio 3 renders.

Usage:
  "$PY" bench/run_zero_code_gpu_trio_2026_11_23.py --dry-run
  "$PY" bench/run_zero_code_gpu_trio_2026_11_23.py --run
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))
from _lib import bootstrap, comfy_client, frames  # noqa: E402

COMFY_ROOT = bootstrap(cpu=True)
sys.path = [p for p in sys.path if p != str(REPO)]
if str(REPO / "workflows") not in sys.path:
    sys.path.insert(1, str(REPO / "workflows"))

import h3_config  # noqa: E402
import preflight_graph as pf  # noqa: E402
import prompts  # noqa: E402

DEFAULT_URL = comfy_client.DEFAULT_COMFY_URL
OUT_PREFIX_DIR = "Video/zero_code_trio_2026_11_23"
DATA_DIR = REPO / "data" / "zero_code_trio_2026_11_23"

AUDIO_GROOVE_NAME = "h3_trio_groove_20261123.wav"
REF_WOMAN_NAME = "woman_blonde.jpg"
REF_KITCHEN_NAME = "t2i_30_photoreal_kitchen_three_cooks_seed11.png"

WIDTH = 1152
HEIGHT = 768
LENGTH = 141
GUIDE_FRAME_IDX = 68

SEED_TRIO1 = 4409
SEED_TRIO2_T2I = 5801
SEED_TRIO2_EDIT = 5802
SEED_TRIO2_VID = 5817
SEED_TRIO3 = 6923
SEED_TRIO3_KITCHEN = 7109

UNET_FL2VA = h3_config.MODELS["unet_fl2va"]
UNET_REF2VA = h3_config.MODELS["unet_ref2va"]
ENCODER_BF16 = h3_config.MODELS["clip"]
VIDEO_VAE = h3_config.MODELS["video_vae"]
AUDIO_VAE = h3_config.MODELS["audio_vae"]
IMAGE_VAE = h3_config.IMAGE_GRAPH_VAE
LORA_PH3 = h3_config.PH3_IMAGE_LORA
LORA_PIKA_FL2VA = h3_config.PIKA_FL2VA_SVD995_LORA
LORA_PIKA_REF2VA = h3_config.PIKA_REF2VA_SVD995_LORA
PH3_SIGMAS = h3_config.PH3_IMAGE_MANUAL_SIGMAS
PIKA_SIGMAS = h3_config.PIKA_MANUAL_SIGMAS

TRIO1_PROMPT = (
    "integrated_multimodal_description:\n"
    "[Shot 1] Live-action, cinematic, handheld camera on a 35mm lens at f/4.0 with natural twilight streetlamp exposure and crisp wet cobblestone textures. "
    "A medium-wide shot frames a young street percussionist in a faded olive canvas jacket seated on a folding stool in an arched stone passageway at dusk, flanked by inverted plastic five-gallon buckets, an upturned aluminum pot, and a cracked brass splash cymbal mounted on a weighted stand. "
    "Under steady practical sodium streetlamp light, he drives a brisk syncopated rhythm with wooden drumsticks, alternating tight two-handed rimshot rolls against the bucket edges, heavy open-palm heel kicks into the upright bass tub, and clipped upbeat strikes across the brass cymbal. "
    "His head nods cleanly on each downbeat while his shoulders and wrists snap rapidly from bucket to pot rim in locked time, lips closed throughout. "
    "At 2.40 seconds, the camera glides clockwise around him in a smooth low arc past a rain-damp curb, holding both busy forearms, flying stick tips, and ringing cymbal edge sharply framed until the end of the continuous take.\n"
    "overall_soundscape:\n"
    "Crisp wooden drumstick rimshots on hard plastic bucket rims, punchy hollow bass thuds from the upright tub, bright metallic splash-cymbal accents, sneaker soles tapping wet cobblestones, and distant quiet evening street air under a stone arch.\n\n"
    "non_diegetic_music:\n"
    "N/A"
)

TRIO2_T2I_PROMPT = (
    "A medium-wide eye-level photograph inside a sunlit Victorian iron-and-glass conservatory in morning daylight, "
    "tall clear window panes backed by leafy palms and trailing ferns behind a weathered cedar potting bench. "
    "At center, a botanist in her forties with dark hair loosely pinned back wears an olive waxed-canvas apron over a rolled-sleeve cream linen shirt, "
    "holding fine copper pruning shears gently above a potted white moth orchid in terracotta clay surrounded by terracotta pots, glass misting bottle, and brass plant markers. "
    "Directly above the bench hangs an unlit brass-caged grow lamp on a dark cloth cord. "
    "Natural morning sunlight streams evenly through the clear conservatory glass at f/4.0 on a 50mm lens, crisp leaf veins and cedar grain."
)

TRIO2_EDIT_PROMPT = (
    "Edit <Picture 1>. Transform the sunlit morning conservatory into a stormy deep-indigo twilight evening with dark storm clouds and heavy raindrops streaked down the tall glass panes outside, "
    "and switch on the brass-caged grow lamp hanging above the bench so it emits a warm amber-tungsten downward pool across the white moth orchid petals, terracotta pot, and the botanist's hands while cool blue-hour twilight rims her shoulders and surrounding ferns. "
    "Preserve the exact botanist, her face, pinned hair, olive apron, rolled cream sleeves, cedar potting bench, copper shears pose, pots, and identical medium-wide framing from <Picture 1>."
)

TRIO2_VIDEO_PROMPT = (
    "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
    "integrated_multimodal_description:\n"
    "[Shot 1] Live-action, cinematic, one steady medium-wide shot on a 50mm lens at f/4.0, beginning in the exact morning conservatory framing and natural daylight established in <Picture 1>. "
    "Keeping smooth continuous movement throughout, the botanist trims a dry stem cleanly with the copper shears, sets the shears onto the cedar potting bench, picks up the glass atomizer beside her elbow, and sprays a fine mist over the white orchid blooms, her lips closed. "
    "Midway through the continuous shot, the bright daylight outside the tall conservatory panes deepens into stormy indigo twilight streaked with steady rain while the hanging brass-caged grow lamp switches on overhead, pouring focused amber-tungsten warmth across the wet petals, her apron, and terracotta pots as the camera pushes in smoothly with small amplitude.\n\n"
    "overall_soundscape:\n"
    "Quiet glasshouse room tone, the crisp metallic snip of copper pruning shears, tools set gently on bare cedar wood, two quick mist-spritzer hisses, the soft click of a brass lamp toggle, and heavy evening raindrops pattering steadily against glass roof panes.\n\n"
    "non_diegetic_music:\n"
    "N/A"
)

TRIO3_PROMPT = (
    "subject_definitions:\n"
    "<Subject 1> is the woman from <Picture 1>, preserving her facial features, bright blue eyes, light freckles, and wavy blonde hair tied back into a flying ponytail, dressed in a fitted dark charcoal athletic zip top and black running shorts.\n\n"
    "summary:\n"
    "[reference generation] The target video tracks <Subject 1> exploding out of starting blocks into a full outdoor track sprint.\n\n"
    "retention_analysis:\n"
    "<Subject 1> (appears in [Shot 1], [Shot 2]): partially_preserved - facial structure, bright blue eyes, light freckles, and wavy blonde hair are retained while wardrobe adapts to athletic track gear and active sprint posture.\n\n"
    "detailed_description:\n"
    "The target video is in a cinematic live-action style.\n"
    "[Shot 1] Wide low-angle tracking shot at 35mm and f/4.0 under crisp overcast daylight along a rain-damp crimson rubber eight-lane running track, wet chalk lane lines stretching straight ahead toward empty concrete bleachers, low steel railings, and tall dark evergreen trees in the background. <Subject 1>, retaining the distinct facial bone structure, bright blue eyes, light freckles, and wavy golden-blonde hair established in <Picture 1> now gathered into a loose flying ponytail, begins crouched low in brushed steel starting blocks, weight loaded onto her fingertips directly behind the crisp white start stripe, jaw set and lips closed while faint cool breath condenses above the glistening track surface. On an unseen starter beat she drives explosively upward and forward out of the blocks, right knee punching high past her torso, left elbow snapping cleanly backward as silver beads of rainwater spray off the soaked crimson rubber beneath her metal spikes. The camera tracks diagonally backward at fast speed, leading her surging acceleration down lane four as her shoulders square upright, ponytail streaming horizontally behind her neck, and her torso leans forward through powerful mid-stride drive cycles.\n"
    "[Shot 2] The shot cuts cleanly to a tight front-three-quarter moving medium tracking shot matching pace just ahead of <Subject 1> at f/4.0 while she charges forward at top sprint speed, glistening wet lane stripes and rain puddles blurring steadily beneath her pumping knees and driving elbows. Her face stays clearly visible in soft overcast sky light with stray golden strands sweeping back past her temples, sweat beads shining lightly across her cheeks, and her chest rising rapidly between forceful strides. Passing the painted twenty-meter track marker without breaking rhythm, <Subject 1>, speaking in a clear, breathy, urgent alto (S1), calls straight ahead to a coach standing off-frame beyond the finish lane: <d>[English] Clear the lane, coming through!</d> Immediately her lips close firmly again as she drops her chin, pumps both forearms crisply past her ribs, leans deeper into full velocity, and drives closer toward the moving lens across slick wet track rubber while wet reflections shimmer along the curved white lane lines widening past both edges of frame.\n\n"
    "overall_soundscape:\n"
    "Metal starting-block springs snapping cleanly, sharp track-spike cleats ripping wet crimson rubber in rapid accelerating cadence, crisp cloth wind-whip across a fitted zip top, rhythmic runner exhalations, and distant open-stadium air.\n\n"
    "non_diegetic_music:\n"
    "N/A"
)

TRIO3_KITCHEN_PROMPT = (
    "subject_definitions:\n"
    "<Subject 1> is the bearded mature line chef on the left in <Picture 1>, wearing a dark charcoal T-shirt and navy apron.\n"
    "<Subject 2> is the ponytail saute chef at center in <Picture 1>, wearing a white double-breasted chef jacket.\n"
    "<Subject 3> is the clean-shaven bald prep cook on the right in <Picture 1> with a red neck bandana and white chef shirt.\n\n"
    "summary:\n"
    "[reference generation] Inside a busy restaurant pass during dinner service, <Subject 1>, <Subject 2>, and <Subject 3> plate seared fish, toss a flaring wok, call table pickups, and ring the counter bell.\n\n"
    "retention_analysis:\n"
    "<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - face, cropped salt-and-pepper beard, dark T-shirt, and navy apron match <Picture 1>.\n"
    "<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - face, sleek high ponytail, and white double-breasted chef jacket match <Picture 1>.\n"
    "<Subject 3> (appears in [Shot 1], [Shot 2]): fully_preserved - facial likeness, shaved head, red bandana, and white shirt match <Picture 1>.\n\n"
    "detailed_description:\n"
    "The target video is in a cinematic live-action style.\n"
    "[Shot 1] Medium-wide eye-level handheld shot on a 35mm lens at f/4.0 inside the cramped stainless-steel restaurant pass seen in <Picture 1>, softly lit by warm under-shelf heat lamps, fluorescent kitchen overheads, and golden range fire with crisp metal textures. At foreground left, <Subject 1> carefully crowns a whole roasted sea bass on a white oval plate with fresh herb sprigs using slender plating tongs, slides the finished platter down the gleaming steel shelf, and calls briskly across the pass without raising his eyes in a gravelly baritone (S1): <d>[English] Pickup table six, fire the halibut!</d> Before turning back to another tray, he presses his lips firmly shut while nudging stacked porcelain dishes straight. Beside him at center, <Subject 2> snaps a seasoned black steel wok smoothly back and forth above flaring orange gas flames, tossing blistering green stalks through rising steam before sweeping her wide steel spatula underneath to lift the vegetables cleanly into a warmed bowl. Farther along the counter at right, <Subject 3> flips a white bar towel neatly onto his shoulder, pulls a fluttering paper order ticket free from the hanging silver clip rail, and secures both edges of the finished fish plate using folded clean cloths as kitchen steam drifts upward past rows of hung wire tickets.\n"
    "[Shot 2] The camera cuts to a tighter medium two-shot across the pass at f/4.0, gently pushing past stacked ceramic plates to keep <Subject 2> and <Subject 1> in clean balanced profile while <Subject 3> clears dishes along the far rail. Catching her breath after killing the high burner flame, <Subject 2> blots her forehead once against her dry forearm sleeve and replies directly to <Subject 1> in a sharp, focused alto (S2): <d>[English] Two behind on nine, pan is ready!</d> She closes her mouth cleanly as she ladles glossy pan glaze over crisp charred greens, while <Subject 1> spoons shimmering herb oil slowly across the halibut fillet without spilling a drop and <Subject 3> strikes the small brass service bell cleanly once with a crisp tap of his open palm as evening rush service settles smoothly into steady plating rhythm.\n\n"
    "overall_soundscape:\n"
    "Steady commercial exhaust-hood hum, brisk sizzling fish oil and fierce wok flame whoosh, crisp ceramic plate clinks on stainless steel, rustling paper order tickets overhead, and one bright single chime from a brass counter bell.\n\n"
    "non_diegetic_music:\n"
    "N/A"
)


def _input_dir() -> Path:
    val = os.environ.get("H3_INPUT_DIR")
    if not val:
        raise SystemExit("H3_INPUT_DIR is unset; source h3.env first")
    p = Path(val)
    if not p.is_dir():
        raise SystemExit(f"H3_INPUT_DIR does not exist: {p}")
    return p


def ensure_trio_audio(input_dir: Path) -> Path:
    """Synthesize a crisp 32 kHz stereo 118 BPM drum+bass groove WAV in `$H3_INPUT_DIR`."""
    dest = input_dir / AUDIO_GROOVE_NAME
    sr = 32000
    dur_s = 7.0
    n = int(round(sr * dur_s))
    t = np.arange(n, dtype=np.float64) / float(sr)
    rng = np.random.default_rng(20261123)

    beat_s = 60.0 / 118.0
    left = np.zeros(n, dtype=np.float64)
    right = np.zeros(n, dtype=np.float64)

    # Kick on beats 0, 2 and syncopated 2.5 in each 4-beat bar; snare on 1, 3; 8th hats.
    n_beats = int(dur_s / beat_s) + 1
    for b in range(n_beats):
        pos_in_bar = b % 4
        t0 = b * beat_s
        i0 = int(round(t0 * sr))
        if i0 >= n:
            break
        if pos_in_bar in (0, 2):
            dur = min(int(0.24 * sr), n - i0)
            lt = np.arange(dur, dtype=np.float64) / sr
            freq = 48.0 + 95.0 * np.exp(-lt * 26.0)
            phase = 2.0 * math.pi * np.cumsum(freq) / sr
            kick = 0.85 * np.sin(phase) * np.exp(-lt * 11.0)
            left[i0 : i0 + dur] += kick
            right[i0 : i0 + dur] += kick
        else:
            dur = min(int(0.18 * sr), n - i0)
            lt = np.arange(dur, dtype=np.float64) / sr
            tone = 0.35 * np.sin(2.0 * math.pi * 185.0 * lt) * np.exp(-lt * 22.0)
            noise_l = 0.50 * rng.standard_normal(dur) * np.exp(-lt * 24.0)
            noise_r = 0.50 * rng.standard_normal(dur) * np.exp(-lt * 24.0)
            # Emphasize crisp rimshot transient
            snap = 0.25 * np.sin(2.0 * math.pi * 620.0 * lt) * np.exp(-lt * 65.0)
            left[i0 : i0 + dur] += tone + noise_l + snap * 0.9
            right[i0 : i0 + dur] += tone + noise_r + snap * 1.1

        # 8th-note hi-hats
        for sub, vel in ((0.0, 0.22), (0.5, 0.16)):
            ih = int(round((t0 + sub * beat_s) * sr))
            if ih >= n:
                continue
            hdur = min(int(0.065 * sr), n - ih)
            ht = np.arange(hdur, dtype=np.float64) / sr
            raw = rng.standard_normal(hdur)
            hp = np.diff(np.concatenate(([0.0], raw)))
            hat = vel * hp * np.exp(-ht * 55.0)
            left[ih : ih + hdur] += hat * 0.85
            right[ih : ih + hdur] += hat * 1.15

        # Subtle bass pluck per beat
        bass_notes = (82.41, 82.41, 98.00, 110.00)
        f0 = bass_notes[pos_in_bar]
        bdur = min(int(0.32 * sr), n - i0)
        bt = np.arange(bdur, dtype=np.float64) / sr
        bass = (
            0.22 * np.sin(2.0 * math.pi * f0 * bt) + 0.08 * np.sin(4.0 * math.pi * f0 * bt)
        ) * np.exp(-bt * 6.5)
        left[i0 : i0 + bdur] += bass
        right[i0 : i0 + bdur] += bass

    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo)))
    target_peak = 10.0 ** (-1.5 / 20.0)
    if peak > 1e-8:
        stereo = stereo * (target_peak / peak)
    pcm = np.clip(np.round(stereo * 32767.0), -32768, 32767).astype("<i2")

    tmp = dest.with_suffix(".tmp.wav")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    tmp.replace(dest)
    return dest


def _base_loaders(unet_name: str, *, include_image_vae: bool = False) -> dict[str, Any]:
    g: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3EncoderLoader", "inputs": {"encoder_name": ENCODER_BF16, "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
    }
    if include_image_vae:
        g["101"] = {"class_type": "VAELoader", "inputs": {"vae_name": IMAGE_VAE}}
    return g


def _wire_pika_chain(g: dict[str, Any], *, unet_id: str = "1", lora_name: str) -> str:
    g.update({
        "18": {
            "class_type": "MiniMaxH3LoRABranch",
            "inputs": {
                "model": [unet_id, 0],
                "lora_name": lora_name,
                "strength": 1.0,
                "modules": "all",
                "blocks": "all",
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "19": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["18", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "58": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["19", 0], "attention": "comfy kitchen attention"}},
        "21": {
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
        },
    })
    return "21"


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


def _build_trio1_arm(arm: str, *, mode: str) -> dict[str, Any]:
    g = _base_loaders(UNET_FL2VA)
    sol_id = _wire_pika_chain(g, lora_name=LORA_PIKA_FL2VA)
    g.update({
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED_TRIO1}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}},
        "60": {"class_type": "ManualSigmas", "inputs": {"sigmas": PIKA_SIGMAS}},
        "5": {
            "class_type": "MiniMaxH3Conditioning",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "prompt": TRIO1_PROMPT,
                "width": WIDTH,
                "height": HEIGHT,
                "length": LENGTH,
                "canvas": "explicit",
            },
        },
        "26": {
            "class_type": "MiniMaxH3Preflight",
            "inputs": {"conditioning": ["5", 0], "samples": ["5", 1]},
        },
        "9": {"class_type": "BasicGuider", "inputs": {"model": [sol_id, 0], "conditioning": ["26", 0]}},
    })

    if mode in ("mask000", "mask025"):
        mask_val = 0.0 if mode == "mask000" else 0.25
        g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": AUDIO_GROOVE_NAME}}
        g["49"] = {
            "class_type": "MiniMaxH3FreezeAudio",
            "inputs": {
                "latent": ["26", 1],
                "audio_vae": ["4", 0],
                "audio": ["48", 0],
                "start_seconds": 0.0,
                "audio_mask": mask_val,
                "level": "clip_guard",
            },
        }
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
        g["30"] = {
            "class_type": "MiniMaxH3ProvenanceStamp",
            "inputs": {"latent": ["10", 0], "model": [sol_id, 0], "sigmas": ["60", 0], "clip": ["2", 0], "note": arm},
        }
        g["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}}
        _wire_vhs(g, vhs_id="13", images_ref=["11", 0], audio_ref=["49", 1], prefix=f"{OUT_PREFIX_DIR}/{arm}")
        if mode == "mask025":
            g["12"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}}
            _wire_vhs(
                g,
                vhs_id="113",
                images_ref=["11", 0],
                audio_ref=["12", 0],
                prefix=f"{OUT_PREFIX_DIR}/{arm}_decoded",
            )
    elif mode == "audiorefine":
        g["10"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["6", 0],
                "guider": ["9", 0],
                "sampler": ["7", 0],
                "sigmas": ["60", 0],
                "latent_image": ["26", 1],
            },
        }
        # Pass-1 comparison video+audio decode
        g["111"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}}
        g["112"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}}
        _wire_vhs(
            g,
            vhs_id="113",
            images_ref=["111", 0],
            audio_ref=["112", 0],
            prefix=f"{OUT_PREFIX_DIR}/trio1c_pass1_pika6",
        )
        # Pass-2 undistilled audio-refine + FrozenVideoCache (matches director/graph_builder.py)
        g["80"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": 12.0, "shift_audio": 3.0}}
        g["81"] = {"class_type": "ModelAttentionBackend", "inputs": {"model": ["80", 0], "attention": "comfy kitchen attention"}}
        g["82"] = copy.deepcopy(g["21"])
        g["82"]["inputs"]["model"] = ["81", 0]
        g["88"] = {
            "class_type": "MiniMaxH3FrozenVideoCache",
            "inputs": {"model": ["82", 0], "precision": "int4", "refresh": False, "refresh_every": 2, "verify": False, "halo": 0},
        }
        g["83"] = {"class_type": "MiniMaxH3AudioRefineMask", "inputs": {"latent": ["10", 0], "video_mask": 0.0, "audio_mask": 1.0}}
        g["84"] = {"class_type": "BasicScheduler", "inputs": {"model": ["88", 0], "scheduler": "simple", "steps": 6, "denoise": 0.5}}
        g["85"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
        g["86"] = {"class_type": "BasicGuider", "inputs": {"model": ["88", 0], "conditioning": ["26", 0]}}
        g["87"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["6", 0], "guider": ["86", 0], "sampler": ["85", 0], "sigmas": ["84", 0], "latent_image": ["83", 0]},
        }
        g["30"] = {
            "class_type": "MiniMaxH3ProvenanceStamp",
            "inputs": {"latent": ["87", 0], "model": ["88", 0], "sigmas": ["84", 0], "clip": ["2", 0], "note": arm},
        }
        g["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}}
        g["12"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}}
        _wire_vhs(
            g,
            vhs_id="13",
            images_ref=["11", 0],
            audio_ref=["12", 0],
            prefix=f"{OUT_PREFIX_DIR}/trio1c_pass2_audiorefine",
        )
    else:
        raise ValueError(f"unknown trio1 mode: {mode}")
    return g


def _build_trio2_arm(arm: str, *, guide_end_percent: float, save_anchors: bool) -> dict[str, Any]:
    g = _base_loaders(UNET_FL2VA, include_image_vae=True)
    # P-H3-Image LoRA T2I + 2D Edit branches (zero swap; shares UNET `1`, clip `2`, image VAE `101`)
    g.update({
        "102": {
            "class_type": "MiniMaxH3LoRABranch",
            "inputs": {
                "model": ["1", 0],
                "lora_name": LORA_PH3,
                "strength": 1.0,
                "modules": "all",
                "blocks": "all",
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "103": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["102", 0], "shift_video": 1.0, "shift_audio": 1.0}},
        "104": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["103", 0], "attention": "comfy kitchen attention"}},
        "105": {"class_type": "ManualSigmas", "inputs": {"sigmas": PH3_SIGMAS}},
        "106": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "107": {"class_type": "MiniMaxH3ImageSize", "inputs": {"megapixels": 4.0, "shape": "3:2", "generate_megapixels": 8.0, "enforce_4mp_floor": True}},
        "108": {
            "class_type": "MiniMaxH3PH3ImageConditioning",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["101", 0],
                "prompt": TRIO2_T2I_PROMPT,
                "width": ["107", 0],
                "height": ["107", 1],
                "canvas": "explicit",
            },
        },
        "109": {"class_type": "BasicGuider", "inputs": {"model": ["104", 0], "conditioning": ["108", 0]}},
        "113": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED_TRIO2_T2I}},
        "110": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["113", 0], "guider": ["109", 0], "sampler": ["106", 0], "sigmas": ["105", 0], "latent_image": ["108", 1]},
        },
        "114": {"class_type": "VAEDecode", "inputs": {"samples": ["110", 0], "vae": ["101", 0]}},
        "111": {"class_type": "MiniMaxH3ImageDownscale", "inputs": {"images": ["114", 0], "target_width": ["107", 2], "target_height": ["107", 3]}},
        # Edit branch -> mid-clip guide frame 68
        "120": {
            "class_type": "MiniMaxH3ImageSize",
            "inputs": {"megapixels": 4.0, "shape": "same as the picture", "picture": ["111", 0], "generate_megapixels": 8.0, "enforce_4mp_floor": True},
        },
        "121": {
            "class_type": "MiniMaxH3PH3ImageConditioning",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["101", 0],
                "prompt": TRIO2_EDIT_PROMPT,
                "width": ["120", 0],
                "height": ["120", 1],
                "canvas": "explicit",
                "images.image_1": ["111", 0],
            },
        },
        "122": {"class_type": "BasicGuider", "inputs": {"model": ["104", 0], "conditioning": ["121", 0]}},
        "123": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED_TRIO2_EDIT}},
        "124": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["123", 0], "guider": ["122", 0], "sampler": ["106", 0], "sigmas": ["105", 0], "latent_image": ["121", 1]},
        },
        "125": {"class_type": "VAEDecode", "inputs": {"samples": ["124", 0], "vae": ["101", 0]}},
        "126": {"class_type": "MiniMaxH3ImageDownscale", "inputs": {"images": ["125", 0], "target_width": ["120", 2], "target_height": ["120", 3]}},
    })
    if save_anchors:
        g["112"] = {"class_type": "SaveImage", "inputs": {"images": ["111", 0], "filename_prefix": f"{OUT_PREFIX_DIR}/trio2_t2i_first_frame"}}
        g["127"] = {"class_type": "SaveImage", "inputs": {"images": ["126", 0], "filename_prefix": f"{OUT_PREFIX_DIR}/trio2_edit_guide_f068"}}

    sol_id = _wire_pika_chain(g, lora_name=LORA_PIKA_FL2VA)
    g.update({
        "60": {"class_type": "ManualSigmas", "inputs": {"sigmas": PIKA_SIGMAS}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED_TRIO2_VID}},
        "5": {
            "class_type": "MiniMaxH3Conditioning",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "prompt": TRIO2_VIDEO_PROMPT,
                "width": WIDTH,
                "height": HEIGHT,
                "length": LENGTH,
                "canvas": "explicit",
                "first_frame": ["111", 0],
            },
        },
        "151": {
            "class_type": "MiniMaxH3GuideLatent",
            "inputs": {
                "positive": ["5", 0],
                "latent": ["5", 1],
                "frame_idx": GUIDE_FRAME_IDX,
                "single_frame": False,
                "vae": ["3", 0],
                "guide": ["126", 0],
            },
        },
        "26": {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": ["151", 0], "samples": ["151", 1]}},
        "150": {
            "class_type": "MiniMaxH3ReleaseReferences",
            "inputs": {
                "model": [sol_id, 0],
                "end_percent": 1.0,
                "clean_level": 0.999,
                "audio_end_percent": 1.0,
                "audio_clean_level": 1.0,
                "video_end_percent": 1.0,
                "guide_end_percent": float(guide_end_percent),
            },
        },
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["150", 0], "conditioning": ["26", 0]}},
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0], "sigmas": ["60", 0], "latent_image": ["26", 1]},
        },
        "30": {
            "class_type": "MiniMaxH3ProvenanceStamp",
            "inputs": {"latent": ["10", 0], "model": ["150", 0], "sigmas": ["60", 0], "clip": ["2", 0], "note": arm},
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}},
    })
    _wire_vhs(g, vhs_id="13", images_ref=["11", 0], audio_ref=["12", 0], prefix=f"{OUT_PREFIX_DIR}/{arm}")
    return g


def _build_trio3_arm(
    arm: str,
    *,
    end_percent: float,
    clean_level: float,
    ref_image: str = REF_WOMAN_NAME,
    prompt_text: str = TRIO3_PROMPT,
    seed: int = SEED_TRIO3,
) -> dict[str, Any]:
    g = _base_loaders(UNET_REF2VA)
    sol_id = _wire_pika_chain(g, lora_name=LORA_PIKA_REF2VA)
    g.update({
        "15": {"class_type": "LoadImage", "inputs": {"image": ref_image}},
        "50": {
            "class_type": "MiniMaxH3AppendRefImage",
            "inputs": {
                "image": ["15", 0],
                "size_policy": "tiered",
                "qwen_view": "separate",
                "qwen_view.qwen_short_edge": 512,
            },
        },
        "132": {
            "class_type": "MiniMaxH3EncodeReferences",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "audio_vae": ["4", 0],
                "references": ["50", 0],
                "width": WIDTH,
                "height": HEIGHT,
                "length": LENGTH,
                "video_policy": "comfy",
                "image_policy": "comfy",
            },
        },
        "5": {"class_type": "MiniMaxH3PromptOnReferences", "inputs": {"encoded_references": ["132", 0], "prompt": prompt_text}},
        "26": {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": ["5", 0], "samples": ["132", 1]}},
        "150": {
            "class_type": "MiniMaxH3ReleaseReferences",
            "inputs": {
                "model": [sol_id, 0],
                "end_percent": float(end_percent),
                "clean_level": float(clean_level),
                "audio_end_percent": 1.0,
                "audio_clean_level": 1.0,
                "video_end_percent": 1.0,
                "guide_end_percent": 1.0,
            },
        },
        "60": {"class_type": "ManualSigmas", "inputs": {"sigmas": PIKA_SIGMAS}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["150", 0], "conditioning": ["26", 0]}},
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0], "sigmas": ["60", 0], "latent_image": ["26", 1]},
        },
        "30": {
            "class_type": "MiniMaxH3ProvenanceStamp",
            "inputs": {"latent": ["10", 0], "model": ["150", 0], "sigmas": ["60", 0], "clip": ["2", 0], "note": arm},
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}},
        "12": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}},
    })
    _wire_vhs(g, vhs_id="13", images_ref=["11", 0], audio_ref=["12", 0], prefix=f"{OUT_PREFIX_DIR}/{arm}")
    return g


def build_all_arms(*, only_ensemble_m3: bool = False) -> list[tuple[str, str, dict[str, Any]]]:
    """Return ordered `(trio_tag, arm_name, api_graph)` list maximizing resident cache reuse."""
    ensemble_arms = [
        ("trio3_ensemble", "trio3d_kitchen_end100", _build_trio3_arm("trio3d_kitchen_end100", end_percent=1.00, clean_level=0.999, ref_image=REF_KITCHEN_NAME, prompt_text=TRIO3_KITCHEN_PROMPT, seed=SEED_TRIO3_KITCHEN)),
        ("trio3_ensemble", "trio3e_kitchen_end033", _build_trio3_arm("trio3e_kitchen_end033", end_percent=0.33, clean_level=0.999, ref_image=REF_KITCHEN_NAME, prompt_text=TRIO3_KITCHEN_PROMPT, seed=SEED_TRIO3_KITCHEN)),
    ]
    if only_ensemble_m3:
        return ensemble_arms
    return [
        ("trio1", "trio1a_freeze_mask000", _build_trio1_arm("trio1a_freeze_mask000", mode="mask000")),
        ("trio1", "trio1b_freeze_mask025", _build_trio1_arm("trio1b_freeze_mask025", mode="mask025")),
        ("trio1", "trio1c_joint_audiorefine_cache", _build_trio1_arm("trio1c_joint_audiorefine_cache", mode="audiorefine")),
        ("trio2", "trio2a_guide_end_100", _build_trio2_arm("trio2a_guide_end_100", guide_end_percent=1.00, save_anchors=True)),
        ("trio2", "trio2b_guide_end_033", _build_trio2_arm("trio2b_guide_end_033", guide_end_percent=0.33, save_anchors=False)),
        ("trio3", "trio3a_ref_end100_clean999", _build_trio3_arm("trio3a_ref_end100_clean999", end_percent=1.00, clean_level=0.999)),
        ("trio3", "trio3b_ref_end033_clean999", _build_trio3_arm("trio3b_ref_end033_clean999", end_percent=0.33, clean_level=0.999)),
        ("trio3", "trio3c_ref_end050_clean900", _build_trio3_arm("trio3c_ref_end050_clean900", end_percent=0.50, clean_level=0.900)),
        *ensemble_arms,
    ]


def verify_prompts_and_server(arms: list[tuple[str, str, dict[str, Any]]], *, base_url: str) -> None:
    """Grade every graph's video prompt carrier offline and validate node schemas on `:8188`."""
    obj_info = comfy_client.get_object_info(base_url=base_url, timeout=20.0)
    trio3_dd = TRIO3_PROMPT.split("detailed_description:\n", 1)[1].split("\n\noverall_soundscape:", 1)[0]
    trio3_words = len(trio3_dd.split())
    kitchen_dd = TRIO3_KITCHEN_PROMPT.split("detailed_description:\n", 1)[1].split("\n\noverall_soundscape:", 1)[0]
    kitchen_words = len(kitchen_dd.split())
    print(f"Trio 3 detailed_description words: sprint={trio3_words}, kitchen_ensemble={kitchen_words} (target 350..500)")

    seen_classes: set[str] = set()
    fail_count = 0
    for _trio, arm_name, graph in arms:
        carried = prompts.carriers(graph)
        if not carried:
            raise RuntimeError(f"{arm_name}: no prompt carriers found")
        for car in carried:
            node = graph[car.node_id]
            findings = pf.grade(node, graph, arm_name)
            for lvl, msg in findings:
                print(f"  [{arm_name} node={car.node_id}] {lvl}: {msg}")
                if lvl == "FAIL":
                    fail_count += 1
            if car.node_id == "5":
                for line in pf.price(node, graph):
                    if arm_name in ("trio1a_freeze_mask000", "trio2a_guide_end_100", "trio3a_ref_end100_clean999"):
                        print(f"  [price {arm_name}] {line}")
        for nid, node in graph.items():
            ct = node["class_type"]
            seen_classes.add(ct)
            spec = obj_info.get(ct)
            if not isinstance(spec, dict):
                raise RuntimeError(f"{arm_name} node {nid}: class {ct!r} not found in {base_url}/object_info")
            inp = spec.get("input") or {}
            req = set((inp.get("required") or {}).keys())
            opt = set((inp.get("optional") or {}).keys())
            allowed = req | opt
            if ct == "VHS_VideoCombine":
                fmt = (node.get("inputs") or {}).get("format")
                fmts = (((inp.get("required") or {}).get("format") or [None, {}])[1] or {}).get("formats") or {}
                for item in fmts.get(fmt) or ():
                    if isinstance(item, list) and item:
                        allowed.add(str(item[0]))
            # Note: autogrow/nested widget keys like `rows.*`, `qwen_view.*`, `images.*` are dotted subfields
            for k in node.get("inputs") or {}:
                base_k = k.split(".", 1)[0]
                if base_k not in allowed:
                    raise RuntimeError(f"{arm_name} node {nid} ({ct}): unexpected input {k!r}")
    if fail_count:
        raise RuntimeError(f"Prompt preflight found {fail_count} FAIL(s)")
    print(f"Preflight clean across {len(arms)} arm(s), {len(seen_classes)} distinct node class(es) verified against {base_url}.")


def _pick_output_files(entry: dict[str, Any], out_root: Path) -> dict[str, list[str]]:
    items = comfy_client.collect_saved_files(entry)
    out: dict[str, list[str]] = {"mp4": [], "png": []}
    for it in items:
        sub = str(it.get("subfolder") or "").strip("/")
        fn = str(it.get("filename") or "")
        if not fn:
            continue
        rel = f"{sub}/{fn}" if sub else fn
        if not (out_root / rel).exists():
            continue
        if fn.endswith("-audio.mp4") or (fn.endswith(".mp4") and rel not in out["mp4"]):
            if rel not in out["mp4"]:
                out["mp4"].append(rel)
        elif fn.endswith(".png") and not any(rel.endswith("_00001.png") and rel.replace(".png", "-audio.mp4") in out["mp4"] for _ in [0]):
            out["png"].append(rel)
    return out


def _make_1fps_strip(video_path: Path) -> dict[str, Any]:
    w, h, n_frames = frames.probe(str(video_path))
    stem = video_path.with_suffix("")
    writer = frames.OneASecond(stem, (w, h))
    # Sample at 1 fps (every 24 frames) plus final frame to inspect bookend/tail
    keep = set(range(0, n_frames, 24))
    if n_frames > 0:
        keep.add(n_frames - 1)
    for idx, rgb in enumerate(frames.stream(str(video_path), (w, h), first=0, pix="rgb24")):
        if idx in keep:
            writer.write(rgb.tobytes(), idx)
    code, parts = writer.close()
    if code != 0:
        raise RuntimeError(f"OneASecond failed on {video_path.name} with exit code {code}")
    return {"w": w, "h": h, "frames": n_frames, "parts": [{"file": Path(p["file"]).name, "frames": p["frames"], "bytes": p["bytes"]} for p in parts]}


def _decode_rgb_frames(video_path: Path, size: tuple[int, int] = (WIDTH, HEIGHT)) -> np.ndarray:
    arr = list(frames.stream(str(video_path), size, first=0, pix="rgb24"))
    return np.stack(arr, axis=0) if arr else np.zeros((0, size[1], size[0], 3), dtype=np.uint8)


def _decode_mono_audio(path: Path, sr: int = 24000) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _audio_stats(path: Path, ref_audio: np.ndarray | None = None) -> dict[str, Any]:
    wav = _decode_mono_audio(path, sr=24000)
    if wav.size == 0:
        return {"samples": 0}
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(wav)))
    rms_db = 20.0 * math.log10(max(rms, 1e-12))
    peak_db = 20.0 * math.log10(max(peak, 1e-12))
    res: dict[str, Any] = {
        "duration_s": round(wav.size / 24000.0, 3),
        "rms_dbfs": round(rms_db, 2),
        "peak_dbfs": round(peak_db, 2),
    }
    if ref_audio is not None and ref_audio.size > 0:
        # Compare 25 ms RMS onset-envelope Pearson correlation
        hop = 600  # 25 ms @ 24 kHz
        n_tok = min(wav.size // hop, ref_audio.size // hop)
        if n_tok >= 8:
            e_clip = np.sqrt(np.mean(wav[: n_tok * hop].reshape(n_tok, hop) ** 2, axis=1))
            e_ref = np.sqrt(np.mean(ref_audio[: n_tok * hop].reshape(n_tok, hop) ** 2, axis=1))
            if float(np.std(e_clip)) > 1e-9 and float(np.std(e_ref)) > 1e-9:
                corr = float(np.corrcoef(e_clip, e_ref)[0, 1])
                res["env_corr_vs_groove"] = round(corr, 4)
    return res


def _motion_stats(rgb: np.ndarray) -> dict[str, Any]:
    if rgb.shape[0] < 2:
        return {"frames": int(rgb.shape[0])}
    diff = np.abs(rgb[1:].astype(np.float32) - rgb[:-1].astype(np.float32)).mean(axis=(1, 2, 3))
    return {
        "frames": int(rgb.shape[0]),
        "interframe_mae_mean": round(float(diff.mean()), 3),
        "interframe_mae_first_half": round(float(diff[: len(diff) // 2].mean()), 3),
        "interframe_mae_second_half": round(float(diff[len(diff) // 2 :].mean()), 3),
        "interframe_mae_max": round(float(diff.max()), 3),
    }


def _load_png_rgb(path: Path, size: tuple[int, int] = (WIDTH, HEIGHT)) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def _scrub(obj: Any, out_root: Path, in_root: Path) -> Any:
    o_str = str(out_root).rstrip("/")
    i_str = str(in_root).rstrip("/")
    if isinstance(obj, str):
        return obj.replace(o_str, "$H3_OUTPUT_DIR").replace(i_str, "$H3_INPUT_DIR")
    if isinstance(obj, list):
        return [_scrub(v, out_root, in_root) for v in obj]
    if isinstance(obj, dict):
        return {k: _scrub(v, out_root, in_root) for k, v in obj.items()}
    return obj


def run_all(args: argparse.Namespace) -> int:
    in_root = _input_dir()
    out_root = h3_config.output_dir()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    wav_path = ensure_trio_audio(in_root)
    print(f"Groove WAV ready: {wav_path.name} ({wav_path.stat().st_size} bytes)")
    ref_woman = in_root / REF_WOMAN_NAME
    ref_kitchen = in_root / REF_KITCHEN_NAME
    if not args.only_ensemble_m3 and not ref_woman.is_file():
        raise SystemExit(f"Missing required Trio 3 reference image: {REF_WOMAN_NAME}")
    if not ref_kitchen.is_file():
        raise SystemExit(f"Missing required ensemble reference image: {REF_KITCHEN_NAME}")

    arms = build_all_arms(only_ensemble_m3=bool(args.only_ensemble_m3))
    verify_prompts_and_server(arms, base_url=args.url)
    if not args.run:
        print("Dry-run complete; pass --run to execute on live ComfyUI.")
        return 0

    q = comfy_client.get_queue(base_url=args.url)
    if q.get("queue_running") or q.get("queue_pending"):
        raise SystemExit(f"Refusing to queue: ComfyUI queue is non-empty: {q}")

    client_id = f"zero_code_trio_{uuid.uuid4().hex[:8]}"
    records: list[dict[str, Any]] = []
    video_dirs: dict[str, Path] = {}
    png_dirs: dict[str, Path] = {}

    for trio_tag, arm_name, graph in arms:
        print(f"\n>>> Submitting [{trio_tag}] {arm_name} ...", flush=True)
        pid = comfy_client.submit_prompt(graph, client_id=client_id, base_url=args.url)
        hist = comfy_client.wait_for_prompt(pid, base_url=args.url, poll_interval=2.0, timeout=1800.0)
        dur_s = comfy_client.execution_duration_s(hist)
        files = _pick_output_files(hist, out_root)
        # Prefer -audio.mp4 when both silent and audio muxes exist
        audio_mp4s = [rel for rel in files["mp4"] if rel.endswith("-audio.mp4")]
        strips: dict[str, Any] = {}
        for rel in audio_mp4s:
            vp = out_root / rel
            strips[Path(rel).name] = _make_1fps_strip(vp)
            video_dirs[Path(rel).name] = vp
        for rel in files["png"]:
            png_dirs[Path(rel).name] = out_root / rel

        print(f"<<< Done {arm_name}: {dur_s}s, outputs={files['mp4'] + files['png']}", flush=True)
        records.append({
            "trio": trio_tag,
            "arm": arm_name,
            "prompt_id": pid,
            "execution_seconds": dur_s,
            "mp4_files": [Path(r).name for r in audio_mp4s or files["mp4"]],
            "png_files": [Path(r).name for r in files["png"]],
            "strips": strips,
        })

    # Quantitative cross-arm analysis
    ref_groove = _decode_mono_audio(wav_path, sr=24000)
    analysis: dict[str, Any] = {"audio": {}, "video_motion": {}, "comparisons": {}}

    for fname, vp in sorted(video_dirs.items()):
        analysis["audio"][fname] = _audio_stats(vp, ref_audio=ref_groove)
        rgb = _decode_rgb_frames(vp)
        analysis["video_motion"][fname] = _motion_stats(rgb)

    # Trio 1 comparisons
    def _find_clip(prefix: str) -> np.ndarray | None:
        for fname, vp in video_dirs.items():
            if fname.startswith(prefix) and not fname.endswith("_decoded_00001-audio.mp4"):
                return _decode_rgb_frames(vp)
        return None

    r1a = _find_clip("trio1a_freeze_mask000")
    r1b = _find_clip("trio1b_freeze_mask025")
    r1c_p1 = _find_clip("trio1c_pass1_pika6")
    r1c_p2 = _find_clip("trio1c_pass2_audiorefine")
    if r1a is not None and r1b is not None:
        analysis["comparisons"]["trio1a_vs_1b_video_mae"] = round(
            float(np.abs(r1a.astype(np.float32) - r1b.astype(np.float32)).mean()), 4
        )
    if r1c_p1 is not None and r1c_p2 is not None:
        analysis["comparisons"]["trio1c_pass1_vs_pass2_video_mae"] = round(
            float(np.abs(r1c_p1.astype(np.float32) - r1c_p2.astype(np.float32)).mean()), 4
        )

    # Trio 2 guide alignment & late-half continuity vs anchors
    g68_path = next((p for n, p in png_dirs.items() if "trio2_edit_guide_f068" in n), None)
    ff_path = next((p for n, p in png_dirs.items() if "trio2_t2i_first_frame" in n), None)
    r2a = _find_clip("trio2a_guide_end_100")
    r2b = _find_clip("trio2b_guide_end_033")
    if g68_path is not None and ff_path is not None and r2a is not None and r2b is not None:
        ff_rgb = _load_png_rgb(ff_path).astype(np.float32)
        g68_rgb = _load_png_rgb(g68_path).astype(np.float32)
        analysis["comparisons"]["trio2_anchors_mae"] = round(float(np.abs(ff_rgb - g68_rgb).mean()), 3)
        for label, vid in (("trio2a_guide_end_100", r2a), ("trio2b_guide_end_033", r2b)):
            v = vid.astype(np.float32)
            analysis["comparisons"][f"{label}_f0_vs_t2i_mae"] = round(float(np.abs(v[0] - ff_rgb).mean()), 3)
            analysis["comparisons"][f"{label}_f68_vs_guide_mae"] = round(float(np.abs(v[68] - g68_rgb).mean()), 3)
            analysis["comparisons"][f"{label}_f64_to_68_mae"] = round(float(np.abs(v[68] - v[64]).mean()), 3)
            analysis["comparisons"][f"{label}_f68_to_72_mae"] = round(float(np.abs(v[72] - v[68]).mean()), 3)
            analysis["comparisons"][f"{label}_f140_vs_guide_mae"] = round(float(np.abs(v[140] - g68_rgb).mean()), 3)
        analysis["comparisons"]["trio2a_vs_2b_first_half_mae"] = round(
            float(np.abs(r2a[:68].astype(np.float32) - r2b[:68].astype(np.float32)).mean()), 3
        )
        analysis["comparisons"]["trio2a_vs_2b_second_half_mae"] = round(
            float(np.abs(r2a[68:].astype(np.float32) - r2b[68:].astype(np.float32)).mean()), 3
        )

    summary_path = DATA_DIR / "summary.json"
    if args.only_ensemble_m3 and summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        new_arm_names = {r["arm"] for r in records}
        merged_records = [r for r in (existing.get("records") or []) if r.get("arm") not in new_arm_names]
        merged_records.extend(records)
        ex_an = existing.get("analysis") or {}
        for subk in ("audio", "video_motion", "comparisons"):
            merged_sub = dict(ex_an.get(subk) or {})
            merged_sub.update(analysis.get(subk) or {})
            ex_an[subk] = merged_sub
        existing["records"] = merged_records
        existing["trio3_ensemble_ref"] = REF_KITCHEN_NAME
        existing["analysis"] = ex_an
        summary_path.write_text(json.dumps(_scrub(existing, out_root, in_root), indent=2) + "\n", encoding="utf-8")
    else:
        summary = {
            "run_tag": "zero_code_trio_2026_11_23",
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "audio_groove": AUDIO_GROOVE_NAME,
            "trio3_ref": REF_WOMAN_NAME,
            "trio3_ensemble_ref": REF_KITCHEN_NAME,
            "records": records,
            "analysis": analysis,
        }
        summary_path.write_text(json.dumps(_scrub(summary, out_root, in_root), indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {summary_path.relative_to(REPO)}")

    # Run M1 DINOv3 & M2 YIN F0 probes across Trio 3 & ensemble Trio 3 outputs
    py = sys.executable
    trio3_paths = [
        video_dirs[f]
        for arm_prefix in ("trio3a_ref_end100_clean999", "trio3b_ref_end033_clean999", "trio3c_ref_end050_clean900")
        for f in sorted(video_dirs)
        if f.startswith(arm_prefix)
    ]
    if len(trio3_paths) == 3:
        dino_json = DATA_DIR / "trio3_dinov3.json"
        yin_json = DATA_DIR / "trio3_yin_f0.json"
        print("\n=== Running probe_dinov3_copy_freeze.py on Trio 3 ===", flush=True)
        subprocess.run(
            [py, str(REPO / "bench" / "probe_dinov3_copy_freeze.py"), "--video", *map(str, trio3_paths), "--ref", str(ref_woman), "--stride", "12", "--json", str(dino_json)],
            check=True,
        )
        print("\n=== Running probe_yin_f0_drift.py on Trio 3 ===", flush=True)
        subprocess.run(
            [py, str(REPO / "bench" / "probe_yin_f0_drift.py"), "--baseline", str(trio3_paths[0]), "--steered", str(trio3_paths[1]), str(trio3_paths[2]), "--json", str(yin_json)],
            check=True,
        )
        for jp in (dino_json, yin_json):
            data = json.loads(jp.read_text(encoding="utf-8"))
            jp.write_text(json.dumps(_scrub(data, out_root, in_root), indent=2) + "\n", encoding="utf-8")

    ens_paths = [
        video_dirs[f]
        for arm_prefix in ("trio3d_kitchen_end100", "trio3e_kitchen_end033")
        for f in sorted(video_dirs)
        if f.startswith(arm_prefix)
    ]
    if len(ens_paths) == 2:
        edino_json = DATA_DIR / "trio3_ensemble_dinov3.json"
        eyin_json = DATA_DIR / "trio3_ensemble_yin_f0.json"
        print("\n=== Running probe_dinov3_copy_freeze.py on Trio 3 Ensemble (Kitchen) ===", flush=True)
        subprocess.run(
            [py, str(REPO / "bench" / "probe_dinov3_copy_freeze.py"), "--video", *map(str, ens_paths), "--ref", str(ref_kitchen), "--stride", "12", "--json", str(edino_json)],
            check=True,
        )
        print("\n=== Running probe_yin_f0_drift.py on Trio 3 Ensemble (Kitchen) ===", flush=True)
        subprocess.run(
            [py, str(REPO / "bench" / "probe_yin_f0_drift.py"), "--baseline", str(ens_paths[0]), "--steered", str(ens_paths[1]), "--json", str(eyin_json)],
            check=True,
        )
        for jp in (edino_json, eyin_json):
            data = json.loads(jp.read_text(encoding="utf-8"))
            jp.write_text(json.dumps(_scrub(data, out_root, in_root), indent=2) + "\n", encoding="utf-8")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=DEFAULT_URL, help=f"ComfyUI server URL (default {DEFAULT_URL})")
    ap.add_argument("--only-ensemble-m3", action="store_true", help="Run only the two-speaker three-cook kitchen ref2va companion arms (Trio 3D/3E)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Synthesize audio, validate graphs & prompts, and exit")
    mode.add_argument("--run", action="store_true", help="Run graphs on live ComfyUI and probe outputs")
    args = ap.parse_args()
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
