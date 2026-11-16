#!/usr/bin/env python3
"""Live 1-clip Pika probes for block-scoped AudioAttentionGain and perturbed Frame-0 anchor transfer.

Validates via `--dry-run` (default) or runs on live `:8188` via `--run`:
  Probe 1A (`MiniMaxH3AudioAttentionGain` block scoping on `t2va_studio_dancer_close.txt`, 141f):
    1. `p1a_gain_off`   -- `key_gain=1.0, value_gain=1.0, blocks="all"` (baseline)
    2. `p1a_gain_all`   -- `key_gain=2.0, value_gain=1.5, blocks="all"`
    3. `p1a_gain_0_29`  -- `key_gain=2.0, value_gain=1.5, blocks="0-29"`

  Probe 1B (Perturbed Frame-0 luma transfer vs. camera pull-back on `t2va_pika_balcony_orbit.txt`, 141f):
    0. `p1b_0_scout_t2v`           -- unanchored `t2va` (`seed=730451892`) -> yields `f0_clean` & `1.30x` `f0_bright130`
    1. `p1b_1_guide0_clean_hold100`  -- DiT `MiniMaxH3GuideLatent(frame_idx=0)` on `f0_clean`, `end_percent=1.00`
    2. `p1b_2_guide0_bright_hold100` -- DiT `MiniMaxH3GuideLatent(frame_idx=0)` on `f0_bright130`, `end_percent=1.00`
    3. `p1b_3_guide0_bright_rel050`  -- DiT `MiniMaxH3GuideLatent(frame_idx=0)` on `f0_bright130`, `end_percent=0.50`
    4. `p1b_4_guide0_bright_rel033`  -- DiT `MiniMaxH3GuideLatent(frame_idx=0)` on `f0_bright130`, `end_percent=0.33`
    5. `p1b_5_i2va_bright_hold100`   -- Full `i2va` `MiniMaxH3Conditioning(first_frame=f0_bright130)`, `end_percent=1.00`
    6. `p1b_6_i2va_bright_rel033`    -- Full `i2va` `MiniMaxH3Conditioning(first_frame=f0_bright130)`, `end_percent=0.33`

Replaces `ModelAttentionBackend` (`node "58"`) with `MiniMaxH3SageAttention(mode="auto")` so
`MiniMaxH3AudioAttentionGain` executes instead of passing through as a no-op, while keeping
`MiniMaxH3Sol(dense_blocks="49")` composed downstream. Writes all media under
`$H3_OUTPUT_DIR/Video/<out-subdir>/` (zero media in `data/`).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
if str(_REPO_ROOT / "workflows") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "workflows"))

from _lib import bootstrap  # noqa: E402

bootstrap()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import _paths  # noqa: E402
import h3_config  # noqa: E402
import h3_rules  # noqa: E402
import preflight_graph  # noqa: E402
from _lib import comfy_client, frames  # noqa: E402

BASE_WORKFLOW = _REPO_ROOT / "workflows" / "h3_text_to_video_pika_api.json"
PROMPT_DANCER = _REPO_ROOT / "prompt_bank" / "t2va_studio_dancer_close.txt"
PROMPT_BALCONY = _REPO_ROOT / "prompt_bank" / "t2va_pika_balcony_orbit.txt"

DEFAULT_OUT_SUBDIR = "anchor_probe_2026-11-16"
DEFAULT_WIDTH = 1344
DEFAULT_HEIGHT = 768
DEFAULT_LENGTH = 141  # 5.875s at 24 fps, on_both_clocks (39 + 51*2)
DEFAULT_SEED = 730451892
CONT_SEED = DEFAULT_SEED + 1
BRIGHT_GAIN = 1.30

DRUM_WAV_NAME = "h3_probe_20261116_drum_120bpm.wav"
F0_CLEAN_NAME = "h3_probe_20261116_f0_clean.png"
F0_BRIGHT_NAME = "h3_probe_20261116_f0_bright130.png"

I2VA_HEADER = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
)


def _scrub_paths(obj: Any, out_root: Path, in_root: Path | None = None) -> Any:
    """Replace concrete `$H3_OUTPUT_DIR` / `$H3_INPUT_DIR` paths with literal env names."""
    out_str = str(out_root).rstrip("/")
    in_str = str(in_root).rstrip("/") if in_root is not None else None
    if isinstance(obj, str):
        val = obj.replace(out_str, "$H3_OUTPUT_DIR")
        if in_str:
            val = val.replace(in_str, "$H3_INPUT_DIR")
        return val
    if isinstance(obj, list):
        return [_scrub_paths(v, out_root, in_root) for v in obj]
    if isinstance(obj, dict):
        return {k: _scrub_paths(v, out_root, in_root) for k, v in obj.items()}
    return obj


def _make_base_graph(
    *,
    prompt: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    length: int = DEFAULT_LENGTH,
    seed: int = DEFAULT_SEED,
    out_subdir: str = DEFAULT_OUT_SUBDIR,
    arm: str,
) -> dict[str, Any]:
    if h3_rules.snap_length(length) != length:
        raise ValueError(f"length={length} is not a valid 17n+5 H3 length")
    if length not in h3_rules.chain_lengths() and not h3_rules.on_both_clocks(length):
        raise ValueError(f"length={length} must land on both clocks")
    g = json.loads(BASE_WORKFLOW.read_text(encoding="utf-8"))
    g["5"]["inputs"]["prompt"] = prompt.strip()
    g["6"]["inputs"]["noise_seed"] = int(seed)
    g["27"]["inputs"]["length"] = int(length)
    if width != DEFAULT_WIDTH or height != DEFAULT_HEIGHT:
        g["5"]["inputs"]["width"] = int(width)
        g["5"]["inputs"]["height"] = int(height)
        g["27"]["inputs"]["width"] = int(width)
        g["27"]["inputs"]["height"] = int(height)
    # Replace ModelAttentionBackend with MiniMaxH3SageAttention so AudioAttentionGain executes
    # while keeping MiniMaxH3Sol(dense_blocks="49") composed downstream.
    g["58"] = {
        "class_type": "MiniMaxH3SageAttention",
        "inputs": {
            "model": ["19", 0],
            "mode": "auto",
            "patch_token_refiner": False,
            "head_chunks": 1,
        },
    }
    g["13"]["inputs"]["filename_prefix"] = f"Video/{out_subdir}/{arm}"
    return g


def _wire_provenance_stamp(g: dict[str, Any], *, model_nid: str, arm: str) -> None:
    g["30"] = {
        "class_type": "MiniMaxH3ProvenanceStamp",
        "inputs": {
            "latent": ["10", 0],
            "model": [model_nid, 0],
            "sigmas": ["60", 0],
            "clip": ["2", 0],
            "note": f"anchor_probe:{arm}",
        },
    }
    g["11"]["inputs"]["samples"] = ["30", 0]
    if "12" in g:
        g["12"]["inputs"]["samples"] = ["30", 0]


def synthesize_drum_wav(dest: Path, *, sample_rate: int = 32000, duration_s: float = 7.0) -> Path:
    """Create a deterministic 120 BPM drum-pulse stereo PCM WAV (hits every 0.50s = 12 frames)."""
    n_samples = int(round(duration_s * sample_rate))
    t_all = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
    rng = np.random.default_rng(20261116)
    sig = rng.normal(0.0, 0.0015, size=n_samples)  # -56 dBFS room floor

    beat_interval = 0.50  # 120 BPM = every 12 frames at 24 fps
    n_beats = int(duration_s / beat_interval) + 1
    for b in range(n_beats):
        t0 = b * beat_interval
        s0 = int(round(t0 * sample_rate))
        if s0 >= n_samples:
            break
        hit_len = min(int(0.24 * sample_rate), n_samples - s0)
        tau = np.arange(hit_len, dtype=np.float64) / float(sample_rate)
        # Kick thump on every beat (exponential pitch drop 135 Hz -> 52 Hz)
        freq_inst = 52.0 + 83.0 * np.exp(-tau / 0.032)
        phase = 2.0 * np.pi * np.cumsum(freq_inst) / float(sample_rate)
        kick = 0.62 * np.sin(phase) * np.exp(-tau / 0.075)
        sig[s0 : s0 + hit_len] += kick
        # Crisp snare burst on alternating beats (beats 1, 3, 5...)
        if b % 2 == 1:
            snare_len = min(int(0.14 * sample_rate), n_samples - s0)
            ts = tau[:snare_len]
            noise_burst = rng.normal(0.0, 1.0, size=snare_len)
            tone_snap = np.sin(2.0 * np.pi * 195.0 * ts) * np.exp(-ts / 0.025)
            snare = (0.36 * noise_burst * np.exp(-ts / 0.040)) + (0.28 * tone_snap)
            sig[s0 : s0 + snare_len] += snare
        # Subtle off-beat closed hi-hat at +0.25s
        s_hat = int(round((t0 + 0.25) * sample_rate))
        if s_hat < n_samples:
            h_len = min(int(0.045 * sample_rate), n_samples - s_hat)
            th = np.arange(h_len, dtype=np.float64) / float(sample_rate)
            raw_hat = rng.normal(0.0, 1.0, size=h_len)
            hp_hat = np.diff(raw_hat, prepend=0.0)
            sig[s_hat : s_hat + h_len] += 0.10 * hp_hat * np.exp(-th / 0.010)

    peak = float(np.max(np.abs(sig)))
    if peak > 1e-8:
        sig = sig * (0.707 / peak)  # -3.0 dBFS peak

    stereo = np.stack([sig, sig], axis=1)
    pcm16 = np.clip(np.rint(stereo * 32767.0), -32768, 32767).astype("<i2")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    assert len(t_all) == n_samples
    return dest


def build_probe1a_graph(
    *,
    arm: str,
    key_gain: float,
    value_gain: float,
    blocks: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    length: int = DEFAULT_LENGTH,
    seed: int = DEFAULT_SEED,
    out_subdir: str = DEFAULT_OUT_SUBDIR,
    audio_name: str = DRUM_WAV_NAME,
) -> dict[str, Any]:
    """Build a 6-step Pika AudioFreeze + AudioAttentionGain graph."""
    prompt = PROMPT_DANCER.read_text(encoding="utf-8").strip()
    g = _make_base_graph(
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        seed=seed,
        out_subdir=out_subdir,
        arm=arm,
    )
    g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": audio_name}}
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
    g["10"]["inputs"]["latent_image"] = ["49", 0]
    g["13"]["inputs"]["audio"] = ["49", 1]
    g.pop("12", None)
    g["73"] = {
        "class_type": "MiniMaxH3AudioAttentionGain",
        "inputs": {
            "model": ["21", 0],
            "key_gain": float(key_gain),
            "value_gain": float(value_gain),
            "start_percent": 0.0,
            "end_percent": 1.0,
            "blocks": str(blocks),
            "include_reference_rows": False,
        },
    }
    g["9"]["inputs"]["model"] = ["73", 0]
    _wire_provenance_stamp(g, model_nid="73", arm=arm)
    return g


def build_probe1b_scout_graph(
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    length: int = DEFAULT_LENGTH,
    seed: int = DEFAULT_SEED,
    out_subdir: str = DEFAULT_OUT_SUBDIR,
) -> dict[str, Any]:
    prompt = PROMPT_BALCONY.read_text(encoding="utf-8").strip()
    g = _make_base_graph(
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        seed=seed,
        out_subdir=out_subdir,
        arm="p1b_0_scout_t2v",
    )
    _wire_provenance_stamp(g, model_nid="21", arm="p1b_0_scout_t2v")
    return g


def _balcony_i2va_prompt(base_prompt: str) -> str:
    body = base_prompt.strip()
    if "[Shot 1] " not in body:
        raise ValueError("Expected '[Shot 1] ' in balcony prompt")
    body = body.replace(
        "[Shot 1] ",
        "[Shot 1] Starting from the exact opening frame in <Picture 1>, ",
        1,
    )
    return I2VA_HEADER + body


def build_probe1b_anchor_graph(
    *,
    arm: str,
    anchor_mode: str,
    image_name: str,
    end_percent: float,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    length: int = DEFAULT_LENGTH,
    seed: int = CONT_SEED,
    out_subdir: str = DEFAULT_OUT_SUBDIR,
) -> dict[str, Any]:
    """Build either a DiT-only `MiniMaxH3GuideLatent(frame_idx=0)` or `i2va` `first_frame` arm."""
    if anchor_mode not in ("guide0", "i2va"):
        raise ValueError(f"Unsupported anchor_mode={anchor_mode!r}")
    base_prompt = PROMPT_BALCONY.read_text(encoding="utf-8").strip()
    prompt = base_prompt if anchor_mode == "guide0" else _balcony_i2va_prompt(base_prompt)
    g = _make_base_graph(
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        seed=seed,
        out_subdir=out_subdir,
        arm=arm,
    )
    g["15"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    if anchor_mode == "guide0":
        g["70"] = {
            "class_type": "MiniMaxH3GuideLatent",
            "inputs": {
                "positive": ["5", 0],
                "vae": ["3", 0],
                "latent": ["5", 1],
                "guide": ["15", 0],
                "frame_idx": 0,
                "single_frame": False,
            },
        }
        g["26"]["inputs"]["conditioning"] = ["70", 0]
        g["26"]["inputs"]["samples"] = ["70", 1]
    else:
        g["5"]["inputs"]["first_frame"] = ["15", 0]
        g["5"]["inputs"]["canvas"] = "explicit"

    g["75"] = {
        "class_type": "MiniMaxH3ReleaseReferences",
        "inputs": {
            "model": ["21", 0],
            "end_percent": float(end_percent),
            "clean_level": 0.999,
            "audio_end_percent": 1.0,
            "audio_clean_level": 1.0,
            "video_end_percent": 1.0,
            "guide_end_percent": 1.0,
        },
    }
    g["9"]["inputs"]["model"] = ["75", 0]
    _wire_provenance_stamp(g, model_nid="75", arm=arm)
    return g


def probe1a_arm_specs() -> list[tuple[str, float, float, str]]:
    return [
        ("p1a_gain_off", 1.0, 1.0, "all"),
        ("p1a_gain_all", 2.0, 1.5, "all"),
        ("p1a_gain_0_29", 2.0, 1.5, "0-29"),
    ]


def probe1b_arm_specs() -> list[tuple[str, str, str, float]]:
    return [
        ("p1b_1_guide0_clean_hold100", "guide0", F0_CLEAN_NAME, 1.00),
        ("p1b_2_guide0_bright_hold100", "guide0", F0_BRIGHT_NAME, 1.00),
        ("p1b_3_guide0_bright_rel050", "guide0", F0_BRIGHT_NAME, 0.50),
        ("p1b_4_guide0_bright_rel033", "guide0", F0_BRIGHT_NAME, 0.33),
        ("p1b_5_i2va_bright_hold100", "i2va", F0_BRIGHT_NAME, 1.00),
        ("p1b_6_i2va_bright_rel033", "i2va", F0_BRIGHT_NAME, 0.33),
    ]


def build_all_graphs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    graphs: dict[str, dict[str, Any]] = {}
    if args.only in ("1a", "all"):
        for arm, kg, vg, blks in probe1a_arm_specs():
            graphs[arm] = build_probe1a_graph(
                arm=arm,
                key_gain=kg,
                value_gain=vg,
                blocks=blks,
                width=args.width,
                height=args.height,
                length=args.length,
                seed=args.seed,
                out_subdir=args.out_subdir,
            )
    if args.only in ("1b", "all"):
        graphs["p1b_0_scout_t2v"] = build_probe1b_scout_graph(
            width=args.width,
            height=args.height,
            length=args.length,
            seed=args.seed,
            out_subdir=args.out_subdir,
        )
        for arm, mode, img, ep in probe1b_arm_specs():
            graphs[arm] = build_probe1b_anchor_graph(
                arm=arm,
                anchor_mode=mode,
                image_name=img,
                end_percent=ep,
                width=args.width,
                height=args.height,
                length=args.length,
                seed=args.seed + 1,
                out_subdir=args.out_subdir,
            )
    return graphs


def verify_dry_run(args: argparse.Namespace) -> int:
    graphs = build_all_graphs(args)
    for arm, g in graphs.items():
        assert g["58"]["class_type"] == "MiniMaxH3SageAttention", f"{arm}: Sage node missing"
        assert g["30"]["class_type"] == "MiniMaxH3ProvenanceStamp", f"{arm}: ProvenanceStamp missing"
        assert g["11"]["inputs"]["samples"] == ["30", 0], f"{arm}: VAEDecode must read ProvenanceStamp"
        findings = preflight_graph.grade(g["5"], g, stem=arm)
        fails = [msg for lvl, msg in findings if lvl == "FAIL"]
        if fails:
            raise AssertionError(f"preflight_graph FAIL on {arm}: {fails}")
        # Check no raw file paths leaked
        serialized = json.dumps(g)
        assert "/home/" not in serialized and "/mnt/" not in serialized, f"{arm}: non-relative path in graph"
    print(
        f"[dry-run OK] {len(graphs)} graphs validated ({args.width}x{args.height}, length={args.length}, "
        f"seed={args.seed}, only={args.only}); 0 preflight FAIL; all use MiniMaxH3SageAttention + MiniMaxH3Sol(49)."
    )
    return 0


def _pick_saved_video(
    history_entry: dict[str, Any],
    out_root: Path,
    *,
    file_prefix: str,
) -> Path:
    items = comfy_client.collect_saved_files(history_entry)
    hits: list[Path] = []
    for item in items:
        fn = str(item.get("filename") or "")
        if not fn.endswith(".mp4"):
            continue
        if not (fn.startswith(f"{file_prefix}_") or fn == f"{file_prefix}.mp4"):
            continue
        sub = str(item.get("subfolder") or "")
        path = (out_root / sub / fn) if sub else (out_root / fn)
        if path.is_file():
            hits.append(path)
    if not hits:
        raise RuntimeError(f"No .mp4 matching prefix={file_prefix!r} in {items!r}")
    audio_hits = [p for p in hits if p.stem.endswith("-audio")]
    return audio_hits[0] if audio_hits else hits[0]


def _collect_provenance_stamp(prov_dir: Path, existing: set[Path], arm: str) -> dict[str, Any] | None:
    if not prov_dir.is_dir():
        return None
    target_note = f"anchor_probe:{arm}"
    for sp in sorted(set(prov_dir.glob("stamp_*.json")) - existing):
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("note") == target_note:
            return {
                "file": f"$H3_OUTPUT_DIR/provenance/{sp.name}",
                "note": data.get("note"),
                "sol": data.get("sol"),
                "resolved": data.get("resolved"),
            }
    return None


def _load_rgb_frames(path: Path, size: tuple[int, int], count: int) -> np.ndarray:
    w, h = size
    raw = list(frames.stream(str(path), size, count=count, pix="rgb24"))
    if len(raw) != count:
        raise RuntimeError(f"{path.name}: expected {count} frames at {w}x{h}, got {len(raw)}")
    return np.stack(raw, axis=0)


def _luma_series(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _measure_audio_dbfs(mp4_path: Path, *, sr: int = 32000) -> dict[str, float]:
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(mp4_path),
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0 or not res.stdout:
        return {"rms_dbfs": float("nan"), "peak_dbfs": float("nan")}
    pcm = np.frombuffer(res.stdout, dtype=np.float32).astype(np.float64)
    rms = math.sqrt(float(np.mean(pcm * pcm)))
    peak = float(np.max(np.abs(pcm)))
    return {
        "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-12)), 2),
        "peak_dbfs": round(20.0 * math.log10(max(peak, 1e-12)), 2),
    }


def _measure_probe1a_arm(rgb: np.ndarray, mp4_path: Path) -> dict[str, Any]:
    luma = _luma_series(rgb)  # [T, H, W]
    frame_luma = luma.mean(axis=(1, 2))
    diffs = np.abs(np.diff(luma, axis=0)).mean(axis=(1, 2))  # [T-1], index i corresponds to t=i+1 vs t=i
    t_idx = np.arange(1, rgb.shape[0])
    on_beat_mask = (t_idx % 12) <= 2
    off_beat_mask = ((t_idx % 12) >= 6) & ((t_idx % 12) <= 8)
    on_motion = float(diffs[on_beat_mask].mean())
    off_motion = float(diffs[off_beat_mask].mean())

    # Rolling 7-frame centered median baseline to detect beat-locked global exposure pumping
    n = len(frame_luma)
    med = np.array(
        [np.median(frame_luma[max(0, i - 3) : min(n, i + 4)]) for i in range(n)],
        dtype=np.float64,
    )
    pump_dev = np.abs(frame_luma - med)
    on_pump = float(pump_dev[(np.arange(n) % 12) <= 2].mean())
    off_pump = float(pump_dev[((np.arange(n) % 12) >= 6) & ((np.arange(n) % 12) <= 8)].mean())
    audio_stats = _measure_audio_dbfs(mp4_path)

    return {
        "mean_frame_diff": round(float(diffs.mean()), 4),
        "on_beat_motion": round(on_motion, 4),
        "off_beat_motion": round(off_motion, 4),
        "on_off_beat_ratio": round(on_motion / max(off_motion, 1e-8), 4),
        "global_luma_mean": round(float(frame_luma.mean()), 3),
        "global_luma_std": round(float(frame_luma.std()), 3),
        "beat_luma_pump_mean": round(float(pump_dev.mean()), 4),
        "on_beat_luma_pump": round(on_pump, 4),
        "off_beat_luma_pump": round(off_pump, 4),
        **audio_stats,
    }


def _measure_probe1b_arm(
    rgb: np.ndarray,
    *,
    f0_clean_rgb: np.ndarray,
    clean_hold_luma_frames: np.ndarray,
    scout_luma_frames: np.ndarray,
) -> dict[str, Any]:
    luma_frames = _luma_series(rgb)  # [T, H, W]
    frame_mean_luma = luma_frames.mean(axis=(1, 2))  # [T]
    f0_luma = float(frame_mean_luma[0])

    # Tail window past Frame 0's 17-frame causal VAE block: frames 24..140
    body_slice = slice(24, rgb.shape[0])
    arm_body_luma = float(frame_mean_luma[body_slice].mean())
    clean_body_luma = float(clean_hold_luma_frames[body_slice].mean())
    scout_body_luma = float(scout_luma_frames[body_slice].mean())

    q_slices = {
        "q1_f24_52": slice(24, 53),
        "q2_f53_81": slice(53, 82),
        "q3_f82_110": slice(82, 111),
        "q4_f111_140": slice(111, rgb.shape[0]),
    }
    q_pct_vs_clean = {}
    for qk, qs in q_slices.items():
        c_q = float(clean_hold_luma_frames[qs].mean())
        a_q = float(frame_mean_luma[qs].mean())
        q_pct_vs_clean[qk] = round((a_q - c_q) / max(c_q, 1e-6) * 100.0, 2)

    # Camera travel / Frame-0 pull-back: pixel L1 distance against f0_clean_rgb
    f0_f = f0_clean_rgb.astype(np.float32)
    l1_to_f0 = {
        f"f{idx:03d}": round(float(np.abs(rgb[idx].astype(np.float32) - f0_f).mean()), 3)
        for idx in (0, 35, 70, 105, rgb.shape[0] - 1)
    }
    mean_l1_body = float(
        np.mean([np.abs(rgb[idx].astype(np.float32) - f0_f).mean() for idx in range(24, rgb.shape[0])])
    )
    motion_diffs = np.abs(np.diff(luma_frames[body_slice], axis=0)).mean(axis=(1, 2))
    sharp_vals = [float(cv2.Laplacian(luma_frames[idx], cv2.CV_32F).var()) for idx in range(24, rgb.shape[0], 4)]

    return {
        "f0_luma": round(f0_luma, 3),
        "body_luma_mean_24_140": round(arm_body_luma, 3),
        "luma_pct_vs_clean_hold": round((arm_body_luma - clean_body_luma) / max(clean_body_luma, 1e-6) * 100.0, 2),
        "luma_pct_vs_scout": round((arm_body_luma - scout_body_luma) / max(scout_body_luma, 1e-6) * 100.0, 2),
        "quarters_pct_vs_clean": q_pct_vs_clean,
        "l1_dist_to_f0_clean": l1_to_f0,
        "mean_l1_dist_to_f0_body": round(mean_l1_body, 3),
        "mean_motion_body": round(float(motion_diffs.mean()), 4),
        "sharpness_lap_var_body": round(float(np.mean(sharp_vals)), 2),
    }


def _write_grid_video(
    out_dir: Path,
    stem: str,
    ordered_arms: list[tuple[str, np.ndarray]],
    *,
    cols: int,
    rows: int,
    tile_w: int = 640,
    tile_h: int = 366,
    audio_source_mp4: Path | None = None,
) -> dict[str, Any]:
    """Write a labeled MP4 + .1fps. companion with burned-in frame counters (`working_with_the_owner.md`)."""
    width = cols * tile_w
    height = rows * tile_h
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError(f"Grid dimensions must be even, got {width}x{height}")

    mp4_path = out_dir / f"{stem}.mp4"
    tmp_vid = out_dir / f"{stem}.video_only.mp4"
    target_vid = tmp_vid if audio_source_mp4 is not None else mp4_path

    cmd = [
        "nice", "-n", "19", "ffmpeg", "-v", "error", "-y", "-threads", "2",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "24", "-i", "-",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target_vid),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    strip = frames.OneASecond(out_dir / stem, (width, height))
    n_frames = ordered_arms[0][1].shape[0]
    try:
        for idx in range(n_frames):
            canvas = Image.new("RGB", (width, height), (12, 12, 12))
            draw = ImageDraw.Draw(canvas)
            for slot_i, (label, arr) in enumerate(ordered_arms):
                r, c = divmod(slot_i, cols)
                ox, oy = c * tile_w, r * tile_h
                tile = Image.fromarray(arr[idx]).resize((tile_w, tile_h), Image.Resampling.BILINEAR)
                canvas.paste(tile, (ox, oy))
                tag = f"f{idx:03d} | {label}"
                draw.rectangle([ox + 8, oy + 8, ox + 14 + 6 * len(tag), oy + 26], fill=(0, 0, 0))
                draw.text((ox + 12, oy + 11), tag, fill=(255, 255, 255))
            raw = canvas.tobytes()
            assert proc.stdin is not None
            proc.stdin.write(raw)
            strip.write(raw, idx)
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        code_vid = proc.wait()
        code_strip, parts = strip.close()
    if code_vid != 0 or code_strip != 0:
        raise RuntimeError(f"ffmpeg grid encode failed: vid={code_vid}, strip={code_strip}")

    if audio_source_mp4 is not None:
        mux_cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-i", str(tmp_vid), "-i", str(audio_source_mp4),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-shortest", "-movflags", "+faststart",
            str(mp4_path),
        ]
        subprocess.check_call(mux_cmd)
        tmp_vid.unlink(missing_ok=True)

    return {"mp4": mp4_path.name, "strip_parts": [Path(p["file"]).name for p in parts]}


def run_live(args: argparse.Namespace) -> int:
    q = comfy_client.get_queue(base_url=args.url)
    if (q.get("queue_running") or q.get("queue_pending")) and not args.resume:
        raise RuntimeError(f"ComfyUI queue is not empty: {q}")

    out_root = h3_config.output_dir()
    in_root = _paths.comfy_input()
    out_dir = out_root / "Video" / args.out_subdir
    prov_dir = out_root / "provenance"
    data_dir = _REPO_ROOT / "data" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    saved_prompts_file = data_dir / "prompts.json"
    saved_summary_file = data_dir / "probe_summary.json"
    prev_prompts = json.loads(saved_prompts_file.read_text(encoding="utf-8")) if (args.resume and saved_prompts_file.is_file()) else {}
    prev_summary = json.loads(saved_summary_file.read_text(encoding="utf-8")) if (args.resume and saved_summary_file.is_file()) else {}
    prev_graphs = prev_prompts.get("graphs") if isinstance(prev_prompts.get("graphs"), dict) else {}
    prev_arms = prev_summary.get("arms") if isinstance(prev_summary.get("arms"), dict) else {}

    queued_graphs: dict[str, Any] = dict(prev_graphs) if args.resume else {}
    arm_files: dict[str, str] = (
        dict(prev_prompts.get("saved_files"))
        if (args.resume and isinstance(prev_prompts.get("saved_files"), dict))
        else {}
    )

    def _flush_prompts() -> None:
        rec = {"out_subdir": args.out_subdir, "saved_files": arm_files, "graphs": queued_graphs}
        saved_prompts_file.write_text(json.dumps(_scrub_paths(rec, out_root, in_root), indent=2), encoding="utf-8")

    def _find_existing(prefix: str) -> Path | None:
        audio_hits = sorted(out_dir.glob(f"{prefix}_*-audio.mp4"))
        if audio_hits:
            return audio_hits[-1]
        hits = sorted(out_dir.glob(f"{prefix}_*.mp4"))
        return hits[-1] if hits else None

    def _run_arm(arm: str, graph: dict[str, Any]) -> tuple[Path, float | None, float, dict[str, Any] | None]:
        queued_graphs[arm] = graph
        if args.resume and arm in prev_graphs:
            cand = _find_existing(arm)
            if cand is not None and json.dumps(prev_graphs[arm], sort_keys=True) == json.dumps(
                _scrub_paths(graph, out_root, in_root), sort_keys=True
            ):
                prev_rec = prev_arms.get(arm) if isinstance(prev_arms.get(arm), dict) else {}
                arm_files[arm] = str(cand)
                _flush_prompts()
                print(f"[resume] {arm} -> reusing {cand.name}", flush=True)
                return cand, prev_rec.get("execution_s"), 0.0, prev_rec.get("provenance")

        existing_stamps = set(prov_dir.glob("stamp_*.json")) if prov_dir.is_dir() else set()
        t0 = time.monotonic()
        pid = comfy_client.submit_prompt(graph, base_url=args.url)
        print(f"[queue] {arm} -> {pid}", flush=True)
        entry = comfy_client.wait_for_prompt(pid, base_url=args.url)
        elapsed_s = round(time.monotonic() - t0, 2)
        exec_s = comfy_client.execution_duration_s(entry)
        saved = _pick_saved_video(entry, out_root, file_prefix=arm)
        stamp = _collect_provenance_stamp(prov_dir, existing_stamps, arm)
        arm_files[arm] = str(saved)
        _flush_prompts()
        print(f"[done] {arm} in {elapsed_s:.2f}s (gpu {exec_s or 0:.2f}s) -> {saved.name}", flush=True)
        return saved, exec_s, elapsed_s, stamp

    summary: dict[str, Any] = {
        "out_subdir": args.out_subdir,
        "width": args.width,
        "height": args.height,
        "length": args.length,
        "seed": args.seed,
        "arms": dict(prev_arms) if args.resume else {},
    }
    if args.resume:
        for stack_key in ("probe1a_stack", "probe1b_stack"):
            if stack_key in prev_summary:
                summary[stack_key] = prev_summary[stack_key]

    size = (args.width, args.height)

    # ---- Probe 1A ----
    if args.only in ("1a", "all"):
        drum_wav = synthesize_drum_wav(in_root / DRUM_WAV_NAME)
        print(f"[fixture] synthesized {drum_wav.name} in $H3_INPUT_DIR", flush=True)
        p1a_videos: list[tuple[str, np.ndarray]] = []
        audio_ref_mp4: Path | None = None
        for arm, kg, vg, blks in probe1a_arm_specs():
            g = build_probe1a_graph(
                arm=arm,
                key_gain=kg,
                value_gain=vg,
                blocks=blks,
                width=args.width,
                height=args.height,
                length=args.length,
                seed=args.seed,
                out_subdir=args.out_subdir,
            )
            mp4, exec_s, elapsed_s, stamp = _run_arm(arm, g)
            if audio_ref_mp4 is None:
                audio_ref_mp4 = mp4
            rgb = _load_rgb_frames(mp4, size, args.length)
            p1a_videos.append((arm.replace("p1a_", ""), rgb))
            metrics = _measure_probe1a_arm(rgb, mp4)
            summary["arms"][arm] = {
                "file": str(mp4),
                "execution_s": exec_s,
                "elapsed_s": elapsed_s,
                "key_gain": kg,
                "value_gain": vg,
                "blocks": blks,
                "provenance": stamp,
                **metrics,
            }
        summary["probe1a_stack"] = _write_grid_video(
            out_dir, "probe1a_stacked", p1a_videos, cols=3, rows=1, audio_source_mp4=audio_ref_mp4
        )

    # ---- Probe 1B ----
    if args.only in ("1b", "all"):
        g_scout = build_probe1b_scout_graph(
            width=args.width,
            height=args.height,
            length=args.length,
            seed=args.seed,
            out_subdir=args.out_subdir,
        )
        scout_mp4, scout_exec, scout_elapsed, scout_stamp = _run_arm("p1b_0_scout_t2v", g_scout)
        scout_rgb = _load_rgb_frames(scout_mp4, size, args.length)
        scout_luma = _luma_series(scout_rgb)

        f0_clean_rgb = scout_rgb[0]
        f0_bright_rgb = np.clip(np.rint(f0_clean_rgb.astype(np.float32) * BRIGHT_GAIN), 0, 255).astype(np.uint8)
        Image.fromarray(f0_clean_rgb).save(in_root / F0_CLEAN_NAME)
        Image.fromarray(f0_bright_rgb).save(in_root / F0_BRIGHT_NAME)
        clean_f0_luma = float(_luma_series(f0_clean_rgb[None])[0].mean())
        bright_f0_luma = float(_luma_series(f0_bright_rgb[None])[0].mean())
        print(
            f"[fixture] saved {F0_CLEAN_NAME} (luma={clean_f0_luma:.2f}) and "
            f"{F0_BRIGHT_NAME} (luma={bright_f0_luma:.2f}, {(bright_f0_luma - clean_f0_luma)/clean_f0_luma*100:+.1f}%)",
            flush=True,
        )

        # Anchor arms: must run b1 (clean hold) first so later arms compare against it cleanly
        p1b_rgbs: dict[str, np.ndarray] = {}
        p1b_meta: dict[str, tuple[Path, float | None, float, dict[str, Any] | None, str, float]] = {}
        for arm, mode, img, ep in probe1b_arm_specs():
            g = build_probe1b_anchor_graph(
                arm=arm,
                anchor_mode=mode,
                image_name=img,
                end_percent=ep,
                width=args.width,
                height=args.height,
                length=args.length,
                seed=args.seed + 1,
                out_subdir=args.out_subdir,
            )
            mp4, exec_s, elapsed_s, stamp = _run_arm(arm, g)
            p1b_rgbs[arm] = _load_rgb_frames(mp4, size, args.length)
            p1b_meta[arm] = (mp4, exec_s, elapsed_s, stamp, mode, ep)

        clean_hold_luma = _luma_series(p1b_rgbs["p1b_1_guide0_clean_hold100"])
        summary["arms"]["p1b_0_scout_t2v"] = {
            "file": str(scout_mp4),
            "execution_s": scout_exec,
            "elapsed_s": scout_elapsed,
            "seed": args.seed,
            "provenance": scout_stamp,
            **_measure_probe1b_arm(
                scout_rgb,
                f0_clean_rgb=f0_clean_rgb,
                clean_hold_luma_frames=clean_hold_luma,
                scout_luma_frames=scout_luma,
            ),
        }
        for arm, _, _, _ in probe1b_arm_specs():
            mp4, exec_s, elapsed_s, stamp, mode, ep = p1b_meta[arm]
            summary["arms"][arm] = {
                "file": str(mp4),
                "execution_s": exec_s,
                "elapsed_s": elapsed_s,
                "seed": args.seed + 1,
                "anchor_mode": mode,
                "end_percent": ep,
                "provenance": stamp,
                **_measure_probe1b_arm(
                    p1b_rgbs[arm],
                    f0_clean_rgb=f0_clean_rgb,
                    clean_hold_luma_frames=clean_hold_luma,
                    scout_luma_frames=scout_luma,
                ),
            }

        ordered_1b = [(arm.replace("p1b_", ""), p1b_rgbs[arm]) for arm, *_ in probe1b_arm_specs()]
        summary["probe1b_stack"] = _write_grid_video(
            out_dir, "probe1b_stacked", ordered_1b, cols=3, rows=2
        )

    _flush_prompts()
    saved_summary_file.write_text(json.dumps(_scrub_paths(summary, out_root, in_root), indent=2), encoding="utf-8")
    print(f"[complete] Wrote $H3_OUTPUT_DIR/Video/{args.out_subdir} and data/{args.out_subdir}/probe_summary.json", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate all graphs statically (default)")
    parser.add_argument("--run", action="store_true", help="Submit graphs sequentially to live :8188 ComfyUI")
    parser.add_argument("--resume", action="store_true", help="Reuse verified matching renders in --out-subdir")
    parser.add_argument("--only", default="all", choices=("1a", "1b", "all"), help="Which probe subset to run")
    parser.add_argument("--out-subdir", default=DEFAULT_OUT_SUBDIR, help="Subdirectory under $H3_OUTPUT_DIR/Video and data/")
    parser.add_argument("--url", default=comfy_client.DEFAULT_COMFY_URL, help="ComfyUI server base URL")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.run:
        return run_live(args)
    return verify_dry_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
