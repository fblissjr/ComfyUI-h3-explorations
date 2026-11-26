"""Beat 3B Closed-Loop Self-Steering (`H3LivePolicy`) & Zero-Coordinate Self-Masked Routing (`MiniMaxH3SelfMaskRouter`).

Built strictly on the three sensors empirically validated in
`bench/results/2026-11-26_h3_4sensor_calibration.md` (and intentionally omitting
uncalibrated deep-block `B20..B48` `J-Lens` readouts):

1. **Sensor 1 (`Q_vid @ K_txt[phrase]^T` at `B35 + B40`, Step `0`):**
   Calibrated at `7/7` (`100%`, `3.54x–3.67x`) on `MiniMax-H3-Image` `4 MP` and
   `4/4` (`100%`, `2.11x–2.14x`) on `192f` joint AV `MiniMax-H3`, explicitly
   avoiding `B45` horizontal RoPE stripes and diffuse `B10..B25` layers.
2. **Sensor 2 (`Q_vid @ K_aud[t]^T` at `B35 + B40` & `B25`):**
   Zero-training stereo speaker-face spotlight (`aud:stereo` / `aud:L` / `aud:R` at
   `B35 + B40`) and stereo `L/R` spatial separation (`B25`, `lr_x_sep = +0.140`).
3. **Sensor 3 (Pure-latent `audio_x0` via `sampler_post_cfg_function`, `< 1 ms`):**
   Step-`0/1/2` stereo RMS, `L/R` latent correlation, and `24 fps`-aligned onset flux
   peaks with zero Audio VAE decode.
4. **Visual Thermostat (`evaluate_preview_frame`, `0.0 ms` extra `taeh3` decode):**
   Deterministic luma, saturation, highlight-burn, and motion-freeze evaluator for
   `ModelPreviewOverrideKJ` (`kj_wrapper.frames[-1]`).

`attach_self_mask_router` / `MiniMaxH3SelfMaskRouter` harvests a clean continuous
`[latent_t, patch_h, patch_w]` probability silhouette during Step `0` (`B35 + B40`)
and feeds it directly into `SpatialVisualQRRuntime` (`steering/spatial_query_router.py`)
for Steps `1..5` inside ONE single `1x` pass — zero coordinates, zero `SAM 3.1`,
and zero edits to `spatial_query_router.py` or ComfyUI core.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import base64
import functools
import io as _pyio
import json
import logging
import math
import os
import re
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

from . import hooks as _steer_hooks
from .constants import POST_REFINER_DIM
from .query_router import _extract_cond_tags, _extract_cond_tensor
from .spatial_query_router import (
    AUDIO_ZONES,
    ISOLATION_MODES,
    SpatialVisualQRRuntime,
    _has_spatial_block_patch,
    _make_spatial_attn_override,
)

logger = logging.getLogger("ComfyUI-h3-explorations.live_policy")

LIVE_POLICY_KEY = "_h3_live_policy"
SELF_MASK_WRAPPER_KEY = "minimax_h3_self_mask_router"
GOLDEN_SEMANTIC_BLOCKS: frozenset[int] = frozenset({35, 40})
STEREO_SPLIT_BLOCK: int = 25
MASK_SOURCES: tuple[str, ...] = (
    "text_phrase",
    "active_speaker",
    "left_speaker",
    "right_speaker",
    "audio_transients",
    "thermostat_only",
)
FRAME_PER_TOKEN: tuple[int, ...] = (1, 4, 4, 4, 4)
FRAME_RESCALE: float = 5.0 / 3.0
AUDIO_HZ: float = 40.0
VIDEO_FPS: float = 24.0
_REF_CACHE_MAX = 4


# ------------------------------------------------------------------ token & clock helpers


def _norm_piece(piece: str) -> tuple[str, bool]:
    """Return `(cleaned_lower_text, starts_new_word)` for a BPE token piece."""
    starts_word = piece.startswith(("Ġ", "▁", " ", "\n", "\t"))
    cleaned = piece.lstrip("Ġ▁ \n\t").strip()
    cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", cleaned)
    return cleaned.lower(), starts_word


def resolve_phrase_token_indices(
    token_pieces: Sequence[str],
    target_phrase: str,
) -> list[int]:
    """Match comma-separated target noun(s)/phrase(s) onto relative indices into `token_pieces`.

    Handles Qwen3-VL BPE `Ġ`/`▁` boundaries, multi-subword nouns, and multi-word phrases
    like `"olive oil"` or `"bread, cheeses"`.
    """
    if not token_pieces or not str(target_phrase or "").strip():
        return []

    words: list[tuple[str, list[int]]] = []
    cur_text = ""
    cur_idxs: list[int] = []
    for idx, raw in enumerate(token_pieces):
        s = str(raw or "")
        if not s or (s.startswith("<|") and s.endswith("|>")):
            if cur_text:
                words.append((cur_text, cur_idxs))
                cur_text, cur_idxs = "", []
            continue
        cleaned, starts_word = _norm_piece(s)
        if not cleaned:
            if cur_text:
                words.append((cur_text, cur_idxs))
                cur_text, cur_idxs = "", []
            continue
        if starts_word or not cur_text:
            if cur_text:
                words.append((cur_text, cur_idxs))
            cur_text = cleaned
            cur_idxs = [idx]
        else:
            cur_text += cleaned
            cur_idxs.append(idx)
    if cur_text:
        words.append((cur_text, cur_idxs))

    matched: list[int] = []
    seen: set[int] = set()
    for part in str(target_phrase).split(","):
        toks = [re.sub(r"^[^\w]+|[^\w]+$", "", w).lower() for w in part.strip().split()]
        toks = [w for w in toks if w]
        if not toks:
            continue
        k = len(toks)
        for i in range(0, len(words) - k + 1):
            window = [words[i + j][0] for j in range(k)]
            if window == toks or (
                k == 1 and (window[0] == toks[0] or window[0].rstrip("s") == toks[0].rstrip("s"))
            ):
                for j in range(k):
                    for ti in words[i + j][1]:
                        if ti not in seen:
                            seen.add(ti)
                            matched.append(ti)
    return sorted(matched)


def extract_aligned_token_pieces(clip: Any, prompt: str, expected_len: int) -> list[str]:
    """Tokenize `prompt` through ComfyUI's MiniMax Qwen3-VL tokenizer aligned to `expected_len`."""
    from .constants import import_repo_module

    h3_spans = import_repo_module("h3_spans")
    tok = getattr(clip, "tokenizer", None)
    hf_tok = h3_spans.unwrap_hf_tokenizer(tok)
    if tok is None or hf_tok is None:
        raise ValueError("CLIP object does not expose the MiniMax Qwen3-VL tokenizer")
    clean = str(prompt or "").strip()
    if not clean:
        return ["<|empty|>"] * max(0, int(expected_len))
    raw = tok.tokenize_with_weights(clean) if hasattr(tok, "tokenize_with_weights") else tok.tokenize(clean)
    entries = h3_spans._extract_batch_entries(raw)
    prompt_ids = h3_spans.expand_token_ids(entries, [1] * len(entries))
    if hasattr(hf_tok, "convert_ids_to_tokens"):
        pieces = [str(p or "") for p in hf_tok.convert_ids_to_tokens(prompt_ids)]
    else:
        pieces = [hf_tok.decode([int(i)], skip_special_tokens=False) for i in prompt_ids]
    if len(pieces) == expected_len:
        return pieces
    if len(pieces) < expected_len:
        return (["<|prefix|>"] * (expected_len - len(pieces))) + pieces
    return pieces[-expected_len:]


def video_latent_audio_windows(latent_t: int, audio_t: int) -> list[tuple[int, int]]:
    """Map each causal `[1, 4, 4, 4, 4]` video latent step onto `[a0, a1)` audio ticks (`FRAME_RESCALE=5/3`)."""
    if latent_t <= 0 or audio_t <= 0:
        return []
    windows: list[tuple[int, int]] = []
    cur = 0.0
    for k in range(latent_t):
        span = FRAME_RESCALE * float(FRAME_PER_TOKEN[k % 5])
        a0 = max(0, min(audio_t - 1, int(math.floor(cur))))
        a1 = max(a0 + 1, min(audio_t, int(math.ceil(cur + span))))
        windows.append((a0, a1))
        cur += span
    return windows


def expand_latent_mask_to_frames(mask_thw: torch.Tensor) -> torch.Tensor:
    """Expand a `[latent_t, patch_h, patch_w]` mask into `[F_expected, patch_h, patch_w]`.

    Matches `SpatialVisualQRRuntime._downsample_mask_to_patches` bit-for-bit so
    `_downsample_mask_to_patches(expanded, latent_t, patch_h, patch_w)` recovers `mask_thw`
    exactly with zero edits to `steering/spatial_query_router.py`.
    """
    m = mask_thw.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.ndim != 3:
        raise ValueError(f"mask_thw must have shape [T, H, W], got {tuple(m.shape)}")
    latent_t = int(m.shape[0])
    if latent_t == 1:
        return m.contiguous()
    parts: list[torch.Tensor] = []
    for k in range(latent_t):
        rep = int(FRAME_PER_TOKEN[k % 5])
        parts.append(m[k : k + 1].expand(rep, -1, -1))
    return torch.cat(parts, dim=0).contiguous()


# ------------------------------------------------------------------ 4D attention mask synthesis


def _spatial_blur_3x3(x_thw: torch.Tensor) -> torch.Tensor:
    """Light separable `[1, 2, 1] / 4` 2D spatial blur over `[T, H, W]` with reflect padding."""
    t, h, w = x_thw.shape
    if h < 2 or w < 2:
        return x_thw
    k = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=x_thw.dtype,
        device=x_thw.device,
    ).view(1, 1, 3, 3) / 16.0
    inp = F.pad(x_thw.unsqueeze(1), (1, 1, 1, 1), mode="reflect")
    return F.conv2d(inp, k).squeeze(1)


def _temporal_smooth_thw(x_thw: torch.Tensor) -> torch.Tensor:
    """Gentle `[0.25, 0.50, 0.25]` temporal smoothing across causal video latent steps."""
    if int(x_thw.shape[0]) < 3:
        return x_thw
    out = x_thw.clone()
    out[1:-1] = 0.25 * x_thw[:-2] + 0.50 * x_thw[1:-1] + 0.25 * x_thw[2:]
    out[0] = 0.75 * x_thw[0] + 0.25 * x_thw[1]
    out[-1] = 0.25 * x_thw[-2] + 0.75 * x_thw[-1]
    return out


def synthesize_4d_attention_mask(
    raw_scores_thw: torch.Tensor,
    *,
    threshold_z: float = 1.35,
    softness: float = 0.25,
    dilation_passes: int = 0,
    spatial_blur: bool = True,
    temporal_smooth: bool = True,
    invert: bool = False,
    sparse_floor: float = 0.10,
    cross_time_scale_floor: float = 0.55,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convert raw fused `B35 + B40` attention logits `[T, H, W]` into a clean sparse `[0..1]` mask.

    Hard-zeroes background below `sparse_floor` (and re-normalizes `[sparse_floor, 1 - sparse_floor]`)
    so non-target background patches equal exact `0.0` and never leak into `SpatialVisualQRRuntime`'s
    `idx_region_soft` (`eff_weights > 1e-4`), keeping 1-way `*_shield` crisp and fast.

    For temporal masks (`T > 1`, e.g. `24 fps` AV `active_speaker`/`left_speaker`/`right_speaker`),
    `cross_time_scale_floor` floors each frame's dispersion scale at a fraction of the clip's peak
    frame dispersion so silent/off-screen frames do not amplify low-variance background noise into
    false hotspots (leaves `T == 1` `MiniMax-H3-Image` bit-identical).
    """
    s = raw_scores_thw.detach().to(device="cpu", dtype=torch.float32)
    if s.ndim == 2:
        s = s.unsqueeze(0)
    if s.ndim != 3 or s.numel() == 0:
        raise ValueError(f"raw_scores_thw must be non-empty [T, H, W], got {tuple(s.shape)}")

    if spatial_blur:
        s = _spatial_blur_3x3(s)
    if temporal_smooth and int(s.shape[0]) > 1:
        s = _temporal_smooth_thw(s)

    t_len, ph, pw = (int(d) for d in s.shape)
    flat = s.reshape(t_len, -1)
    med = flat.median(dim=1, keepdim=True).values
    mad = (flat - med).abs().median(dim=1, keepdim=True).values * 1.4826
    std = flat.std(dim=1, unbiased=False, keepdim=True)
    scale = torch.maximum(mad, std * 0.5).clamp(min=1e-5)
    if t_len > 1:
        ct_floor = max(0.0, float(cross_time_scale_floor))
        scale = torch.maximum(scale, ct_floor * scale.max(dim=0, keepdim=True).values).clamp(min=1e-5)
    z = ((flat - med) / scale).reshape(t_len, ph, pw)

    n_dil = max(0, int(dilation_passes))
    if ph >= 3 and pw >= 3:
        z_4d = z.unsqueeze(1)
        for _ in range(n_dil):
            z_4d = F.max_pool2d(z_4d, kernel_size=3, stride=1, padding=1)
        z = z_4d.squeeze(1)

    soft = max(float(softness), 1e-3)
    prob = torch.sigmoid((z - float(threshold_z)) / soft)

    fl = max(0.0, min(0.45, float(sparse_floor)))
    if fl > 0.0:
        prob = ((prob - fl) / max(1.0 - 2.0 * fl, 1e-4)).clamp(0.0, 1.0)
    else:
        prob = prob.clamp(0.0, 1.0)

    if invert:
        prob = (1.0 - prob).clamp(0.0, 1.0)

    # Compute normalized overlay geometry for Studio `h3_self_steer` pills & bounding boxes.
    # Use pre-saturation `z` for `peak_xy` so clamped 1.0 plateaus never bias argmax top-left.
    mean_hw = prob.mean(dim=0)
    peak_src = (-z if invert else z).mean(dim=0)
    ys = (torch.arange(ph, dtype=torch.float32) + 0.5) / float(ph)
    xs = (torch.arange(pw, dtype=torch.float32) + 0.5) / float(pw)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    mass = float(mean_hw.sum().item())
    if mass > 1e-6:
        cx = float((mean_hw * gx).sum().item() / mass)
        cy = float((mean_hw * gy).sum().item() / mass)
    else:
        cx, cy = 0.5, 0.5

    peak_flat = int(peak_src.reshape(-1).argmax().item())
    py, px = divmod(peak_flat, pw)
    peak_xy = [round((px + 0.5) / float(pw), 4), round((py + 0.5) / float(ph), 4)]

    active = mean_hw >= 0.5
    if bool(active.any().item()):
        ay, ax = active.nonzero(as_tuple=True)
        bbox_xyxy = [
            round(float(ax.min().item()) / float(pw), 4),
            round(float(ay.min().item()) / float(ph), 4),
            round(float(ax.max().item() + 1) / float(pw), 4),
            round(float(ay.max().item() + 1) / float(ph), 4),
        ]
    else:
        bbox_xyxy = [0.0, 0.0, 0.0, 0.0]

    frame_hard = (prob >= 0.5).float().mean(dim=(1, 2))
    frame_hard_vals = [round(float(v), 4) for v in frame_hard.tolist()]
    silent_frames = int((frame_hard < 0.005).sum().item())

    stats = {
        "shape_thw": [t_len, ph, pw],
        "coverage_mean": round(float(prob.mean().item()), 4),
        "hard_coverage_mean": round(float((prob >= 0.5).float().mean().item()), 4),
        "frame_hard_coverage_min": min(frame_hard_vals),
        "frame_hard_coverage_max": max(frame_hard_vals),
        "silent_frames_count": silent_frames,
        "sparse_zero_frac": round(float((prob == 0.0).float().mean().item()), 4),
        "peak_xy": peak_xy,
        "centroid_xy": [round(cx, 4), round(cy, 4)],
        "bbox_xyxy": bbox_xyxy,
    }
    if t_len > 1:
        stats["frame_hard_coverage"] = frame_hard_vals
    return prob.contiguous(), stats


# ------------------------------------------------------------------ Visual & Acoustic Thermostats


def evaluate_preview_frame(
    rgb_hwc: np.ndarray,
    *,
    prev_rgb_hwc: np.ndarray | None = None,
    sat_ceiling: float = 0.72,
    clip_ceiling: float = 0.14,
    freeze_mae_floor: float = 1.5,
) -> dict[str, Any]:
    """Pure numpy Visual Thermostat for `ModelPreviewOverrideKJ` (`kj_wrapper.frames[-1]`, `0.0 ms` VAE).

    Returns luma, HSV saturation, highlight/shadow clip fractions, optional frame-to-frame
    motion MAE, risk flags, and a smooth multiplicative `recommended_strength_scale` in `[0.55, 1.15]`.
    """
    arr = np.asarray(rgb_hwc, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"rgb_hwc must have shape [H, W, 3], got {arr.shape}")
    rgb = arr[..., :3]
    if float(rgb.max()) > 1.5:
        rgb01 = np.clip(rgb / 255.0, 0.0, 1.0)
        rgb255 = np.clip(rgb, 0.0, 255.0)
    else:
        rgb01 = np.clip(rgb, 0.0, 1.0)
        rgb255 = rgb01 * 255.0

    luma = 0.2126 * rgb01[..., 0] + 0.7152 * rgb01[..., 1] + 0.0722 * rgb01[..., 2]
    c_max = rgb01.max(axis=-1)
    c_min = rgb01.min(axis=-1)
    sat = np.where(c_max > 1e-6, (c_max - c_min) / np.maximum(c_max, 1e-6), 0.0)

    mean_luma = float(luma.mean())
    mean_sat = float(sat.mean())
    hi_clip = float((c_max >= 0.985).mean())
    lo_clip = float((c_max <= 0.015).mean())

    motion_mae: float | None = None
    if prev_rgb_hwc is not None:
        prev = np.asarray(prev_rgb_hwc, dtype=np.float32)[..., :3]
        if float(prev.max()) <= 1.5:
            prev = prev * 255.0
        if prev.shape == rgb255.shape:
            motion_mae = float(np.abs(rgb255 - prev).mean())

    flags: list[str] = []
    scale = 1.0
    if mean_sat > sat_ceiling:
        flags.append("high_saturation")
        excess = min(1.0, (mean_sat - sat_ceiling) / max(1.0 - sat_ceiling, 0.05))
        scale *= max(0.45, 1.0 - 0.55 * excess)
    if hi_clip > clip_ceiling:
        flags.append("highlight_burn")
        excess = min(1.0, (hi_clip - clip_ceiling) / max(0.5 - clip_ceiling, 0.05))
        scale *= max(0.45, 1.0 - 0.55 * excess)
    if lo_clip > 0.65:
        flags.append("shadow_crush")
        scale *= 0.85
    elif mean_luma < 0.24 and "high_saturation" not in flags and "highlight_burn" not in flags:
        flags.append("dim_understeer_risk")
        scale *= 1.12
    if motion_mae is not None and motion_mae < freeze_mae_floor:
        flags.append("low_motion_freeze_risk")

    scale = max(0.35, min(1.15, float(scale)))
    return {
        "mean_luma": round(mean_luma, 4),
        "mean_sat": round(mean_sat, 4),
        "hi_clip_frac": round(hi_clip, 4),
        "lo_clip_frac": round(lo_clip, 4),
        "motion_mae": None if motion_mae is None else round(motion_mae, 4),
        "flags": flags,
        "recommended_strength_scale": round(scale, 4),
    }


def evaluate_region_separation(
    rgb_hwc: np.ndarray,
    *,
    region_box: tuple[float, float, float, float] | None = None,
    region_mask_hw: np.ndarray | None = None,
    invert: bool = False,
    max_coverage: float = 0.32,
    sep_threshold: float = 12.0,
) -> dict[str, Any] | None:
    """Measure inside-vs-outside RGB/luma contrast for Dynamic Step-1 Auto-Shield (`early_0_24 -> *_shield`).

    Small portals / props (`<= 32%` canvas) routed without `*_shield` can wash out into the
    surrounding background by Step 1 (`sep_score < 12.0`). Returns `None` when no region is registered.
    """
    if region_mask_hw is None and region_box is None:
        return None
    arr = np.asarray(rgb_hwc, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    rgb = arr[..., :3]
    rgb255 = np.clip(rgb, 0.0, 255.0) if float(rgb.max()) > 1.5 else np.clip(rgb, 0.0, 1.0) * 255.0
    h, w = int(rgb255.shape[0]), int(rgb255.shape[1])
    if h < 2 or w < 2:
        return None

    if region_mask_hw is not None:
        m_arr = np.asarray(region_mask_hw, dtype=np.float32)
        if m_arr.ndim == 3:
            m_arr = m_arr.mean(axis=0)
        if m_arr.ndim != 2 or m_arr.size == 0:
            return None
        m_t = torch.from_numpy(m_arr).unsqueeze(0).unsqueeze(0)
        m_rs = F.interpolate(m_t, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()
        reg_mask = m_rs >= 0.5
        if invert:
            reg_mask = ~reg_mask
    else:
        assert region_box is not None
        x0, x1, y0, y1 = (float(v) for v in region_box)
        x_lo, x_hi = min(x0, x1), max(x0, x1)
        y_lo, y_hi = min(y0, y1), max(y0, y1)
        xs = (np.arange(w, dtype=np.float32) + 0.5) / float(w)
        ys = (np.arange(h, dtype=np.float32) + 0.5) / float(h)
        gx, gy = np.meshgrid(xs, ys)
        reg_mask = (gx >= x_lo) & (gx <= x_hi) & (gy >= y_lo) & (gy <= y_hi)
        if invert:
            reg_mask = ~reg_mask

    cov = float(reg_mask.mean())
    if cov <= 0.005 or cov >= 0.995:
        return {
            "region_coverage": round(cov, 4),
            "small_region": False,
            "rgb_mae": 0.0,
            "luma_diff": 0.0,
            "std_diff": 0.0,
            "sep_score": 0.0,
            "washed_out": False,
        }

    bg_mask = ~reg_mask
    luma255 = 0.2126 * rgb255[..., 0] + 0.7152 * rgb255[..., 1] + 0.0722 * rgb255[..., 2]
    rgb_in = rgb255[reg_mask].mean(axis=0)
    rgb_out = rgb255[bg_mask].mean(axis=0)
    rgb_mae = float(np.abs(rgb_in - rgb_out).mean())
    luma_diff = abs(float(luma255[reg_mask].mean()) - float(luma255[bg_mask].mean()))
    std_diff = abs(float(luma255[reg_mask].std()) - float(luma255[bg_mask].std()))
    sep_score = max(rgb_mae, luma_diff, std_diff)
    small_region = cov <= float(max_coverage)
    washed_out = bool(small_region and sep_score < float(sep_threshold))
    return {
        "region_coverage": round(cov, 4),
        "small_region": small_region,
        "rgb_mae": round(rgb_mae, 3),
        "luma_diff": round(luma_diff, 3),
        "std_diff": round(std_diff, 3),
        "sep_score": round(sep_score, 3),
        "washed_out": washed_out,
    }


_STEP0_CRITIC_SYSTEM_PROMPT = (
    "You inspect a Step-0 diffusion preview frame for glaring framing or rendering defects that should be "
    "projected out of the text conditioning on remaining steps. Return ONLY a single compact JSON object "
    "with keys: has_border_seam (bool), has_color_bleed (bool), has_watermark_text (bool), "
    "scrub_words (array of 0-4 single lowercase English artifact nouns such as border, frame, letterbox, "
    "seam, watermark, text, logo, subtitle, neon, oversaturated), reason (short string <= 12 words). "
    "Be conservative: only flag clear visible defects (black letterbox bars, hard split-screen seam, "
    "picture-frame borders, giant text/watermark overlays, or extreme neon color wash). "
    "If the preview looks normal, set all booleans false and scrub_words to []."
)

_STEP0_CRITIC_USER_PROMPT = (
    "Inspect this Step-0 video/image preview frame. Flag any obvious border/letterbox/split-seam, "
    "extreme unnatural color bleed, or unwanted text/watermark overlay, and return the JSON object."
)


def _parse_step0_critic_json(raw_text: str) -> dict[str, Any]:
    """Extract and sanitize a Step-0 critic JSON payload from raw VLM output."""
    text = str(raw_text or "").strip()
    if "```" in text:
        fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if fenced:
            text = fenced[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"critic output did not contain a JSON object: {raw_text!r}")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("critic JSON root must be an object")

    has_border = bool(parsed.get("has_border_seam", False))
    has_bleed = bool(parsed.get("has_color_bleed", False))
    has_text = bool(parsed.get("has_watermark_text", False))
    raw_words = parsed.get("scrub_words")
    words: list[str] = []
    seen: set[str] = set()
    if isinstance(raw_words, (list, tuple)):
        for item in raw_words:
            for tok in re.findall(r"[a-zA-Z]{2,24}", str(item or "").lower()):
                if tok not in seen:
                    seen.add(tok)
                    words.append(tok)
                if len(words) >= 6:
                    break
            if len(words) >= 6:
                break
    if not words:
        if has_border:
            words.extend(["border", "frame", "letterbox"])
        if has_text:
            words.extend(["watermark", "text", "logo"])
        if has_bleed:
            words.extend(["neon", "oversaturated"])
        words = list(dict.fromkeys(words))[:6]

    return {
        "ok": True,
        "has_border_seam": has_border,
        "has_color_bleed": has_bleed,
        "has_watermark_text": has_text,
        "scrub_words": words,
        "reason": str(parsed.get("reason") or ("artifact_detected" if words else "clean"))[:160],
    }


def run_heylook_step0_critic(
    rgb_hwc: np.ndarray,
    *,
    base_url: str = "",
    model: str = "",
    timeout_s: float = 18.0,
    api_key: str | None = None,
    max_edge: int = 512,
) -> dict[str, Any]:
    """Run a zero-local-VRAM remote `heylook` VLM Step-0 artifact check (`thinking="off"`, graceful fallback)."""
    t0 = time.perf_counter()
    try:
        from PIL import Image
        import httpx
        from .constants import import_repo_module

        hc = import_repo_module("heylook_client")
        arr = np.asarray(rgb_hwc, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"rgb_hwc must have shape [H, W, 3], got {arr.shape}")
        rgb = arr[..., :3]
        u8 = (
            np.clip(np.rint(rgb), 0.0, 255.0).astype(np.uint8)
            if float(rgb.max()) > 1.5
            else np.clip(np.rint(rgb * 255.0), 0.0, 255.0).astype(np.uint8)
        )
        img = Image.fromarray(u8, mode="RGB")
        img.thumbnail((max(128, int(max_edge)), max(128, int(max_edge))), Image.Resampling.LANCZOS)
        buf = _pyio.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64_img = base64.b64encode(buf.getvalue()).decode("ascii")

        base = hc.resolve_heylook_base_url(base_url)
        timeout = httpx.Timeout(connect=5.0, read=max(3.0, float(timeout_s)), write=10.0, pool=5.0)
        with hc.make_heylook_client(base, timeout=timeout, api_key=api_key) as client:
            model_row = hc.discover_heylook_model(
                client,
                base,
                model,
                need=frozenset({"chat", "vision"}),
                api_key=api_key,
            )
            body = hc.build_heylook_request_body(
                model_row=model_row,
                system_prompt=_STEP0_CRITIC_SYSTEM_PROMPT,
                user_prompt=_STEP0_CRITIC_USER_PROMPT,
                prepared_images=[(b64_img, "image/jpeg")],
                thinking="off",
                sampler_mode="custom",
                sampler_overrides={"max_tokens": 140, "temperature": 0.0},
            )
            res = hc.stream_heylook_message(base, body, client=client, api_key=api_key)
        parsed = _parse_step0_critic_json(res.text)
        parsed["model"] = str(res.model or model_row.get("id") or "")
        parsed["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return parsed
    except Exception as exc:
        logger.warning("[h3] heylook Step-0 visual critic skipped (%s): %s", type(exc).__name__, exc)
        return {
            "ok": False,
            "error": str(exc)[:240],
            "has_border_seam": False,
            "has_color_bleed": False,
            "has_watermark_text": False,
            "scrub_words": [],
            "reason": "critic_unavailable",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }


def synthesize_policy_scrub_unit_vec(
    words: Sequence[str],
    *,
    family: str = "fl2va",
) -> torch.Tensor | None:
    """Synthesize a unit-norm `5376`-d post-refiner direction in `< 3 ms` CPU via `steering.xray`."""
    clean = [str(w).strip().lower() for w in words if str(w).strip()]
    if not clean:
        return None
    try:
        from . import xray as _xray

        if not _xray.bundle_available():
            return None
        synth = _xray.synthesize_from_words(clean, family=family)
        vec = synth.get("vector.post_refiner")
        if not torch.is_tensor(vec) or vec.ndim != 1:
            return None
        v32 = vec.detach().to(device="cpu", dtype=torch.float32)
        nrm = float(torch.linalg.vector_norm(v32).item())
        if not math.isfinite(nrm) or nrm <= 1e-8:
            return None
        return (v32 / nrm).contiguous()
    except Exception as exc:
        logger.debug("[h3] scrub word synthesis skipped for %r: %s", clean, exc)
        return None


def apply_live_policy_text_scrub(
    context: torch.Tensor,
    tags_raw: Any,
    policy: "H3LivePolicy | None",
    *,
    dm: Any = None,
    transformer_options: dict[str, Any] | None = None,
    family: str = "fl2va",
) -> tuple[torch.Tensor, bool]:
    """Project `policy.project_out_words` out of text rows (`tags != 0`, excluding sink row `0`).

    Preserves vision prefix rows (`tags == 0`) and sink row `0` bitwise untouched while scrubbing
    artifact directions (`alpha = policy.critic_scrub_alpha`, default `0.85`) from `5376`-d text rows.
    """
    if policy is None or not policy.project_out_words:
        return context, False
    alpha = max(0.0, min(1.0, float(policy.critic_scrub_alpha)))
    if alpha <= 0.0:
        return context, False

    unit_vec = policy.resolve_scrub_unit_vec(family=family)
    if unit_vec is None:
        return context, False

    ctx_work = context
    if ctx_work.ndim == 3 and int(ctx_work.shape[-1]) != POST_REFINER_DIM:
        if dm is None or not (hasattr(dm, "token_refiner") and hasattr(dm, "condition_proj")):
            return context, False
        ctx_work = _steer_hooks.refine_text_context(
            dm,
            ctx_work,
            transformer_options,
            caller_name="H3LivePolicy.text_scrub",
        )
    if ctx_work.ndim != 3 or int(ctx_work.shape[-1]) != int(unit_vec.shape[0]):
        return context, False

    bsz, seq_len, _dim = (int(d) for d in ctx_work.shape)
    if tags_raw is not None:
        tags = torch.as_tensor(tags_raw, device=ctx_work.device, dtype=torch.long)
        if tags.ndim == 1:
            tags = tags.unsqueeze(0).expand(bsz, -1)
        if tuple(tags.shape[:2]) != (bsz, seq_len):
            row_mask = torch.ones((bsz, seq_len), device=ctx_work.device, dtype=torch.bool)
        else:
            row_mask = tags != 0
    else:
        row_mask = torch.ones((bsz, seq_len), device=ctx_work.device, dtype=torch.bool)
    if seq_len > 1:
        row_mask = row_mask.clone()
        row_mask[:, 0] = False
    if not bool(row_mask.any().item()):
        return ctx_work, False

    u = unit_vec.to(device=ctx_work.device, dtype=torch.float32)
    ctx_f = ctx_work.detach().to(dtype=torch.float32)
    dots = torch.matmul(ctx_f, u)  # [B, L]
    delta = (alpha * dots).unsqueeze(-1) * u.view(1, 1, -1)
    out = ctx_work.clone()
    scrubbed = (ctx_f - delta).to(dtype=ctx_work.dtype)
    out[row_mask] = scrubbed[row_mask]
    return out, True


def evaluate_audio_x0(
    audio_x0: torch.Tensor,
    *,
    z_thresh: float = 2.0,
    min_gap_steps: int = 6,
    binaural_expected: bool = False,
) -> dict[str, Any]:
    """`< 1 ms` pure-latent Acoustic Thermostat on predicted `audio_x0` (`[B, 32, 2, T_a]`)."""
    a = audio_x0.detach().float()
    if a.ndim == 4:
        a = a[0]
    if a.ndim != 3:
        raise ValueError(f"Expected 3D/4D audio_x0 [B?, C, 2, T], got {tuple(audio_x0.shape)}")
    if int(a.shape[1]) == 2:
        c_dim, t_a = int(a.shape[0]), int(a.shape[2])
        left, right = a[:, 0, :], a[:, 1, :]
    elif int(a.shape[0]) == 2:
        c_dim, t_a = int(a.shape[1]), int(a.shape[2])
        left, right = a[0, :, :], a[1, :, :]
    else:
        raise ValueError(f"Cannot locate stereo channel dim=2 in audio_x0 shape {tuple(a.shape)}")

    e_l = torch.linalg.vector_norm(left, dim=0) / math.sqrt(float(max(1, c_dim)))
    e_r = torch.linalg.vector_norm(right, dim=0) / math.sqrt(float(max(1, c_dim)))
    e_st = 0.5 * (e_l + e_r)

    rms_l = float(torch.sqrt((e_l * e_l).mean()).item())
    rms_r = float(torch.sqrt((e_r * e_r).mean()).item())
    rms_st = float(torch.sqrt((e_st * e_st).mean()).item())
    lr_db_diff = 20.0 * math.log10((rms_l + 1e-8) / (rms_r + 1e-8))

    lf = left.reshape(-1) - left.mean()
    rf = right.reshape(-1) - right.mean()
    denom = float((torch.linalg.vector_norm(lf) * torch.linalg.vector_norm(rf)).item())
    lr_corr = float((lf @ rf).item() / denom) if denom > 1e-12 else 0.0

    flux = torch.zeros_like(e_st)
    if t_a >= 2:
        flux[1:] = torch.clamp(e_st[1:] - e_st[:-1], min=0.0)
    f_std = float(flux.std(unbiased=False).item())
    f_mean = float(flux.mean().item())
    z_curve = (flux - f_mean) / max(f_std, 1e-6)

    transients: list[dict[str, Any]] = []
    last_i = -min_gap_steps
    for idx in range(1, t_a - 1):
        zv = float(z_curve[idx].item())
        if zv >= z_thresh and flux[idx] >= flux[idx - 1] and flux[idx] >= flux[idx + 1]:
            if (idx - last_i) >= min_gap_steps:
                t_sec = idx / AUDIO_HZ
                transients.append(
                    {
                        "a_idx": int(idx),
                        "t_sec": round(t_sec, 3),
                        "video_frame_24fps": int(round(t_sec * VIDEO_FPS)),
                        "flux_z": round(zv, 3),
                    }
                )
                last_i = idx

    flags: list[str] = []
    if rms_st < 0.25:
        flags.append("silent_dead_air")
    if binaural_expected and lr_corr > 0.85:
        flags.append("stereo_collapse_risk")
    if rms_st > 3.2 and len(transients) >= 18:
        flags.append("visual_hiss_bleed_risk")

    return {
        "audio_t": t_a,
        "rms_stereo": round(rms_st, 4),
        "rms_left": round(rms_l, 4),
        "rms_right": round(rms_r, 4),
        "lr_db_diff": round(lr_db_diff, 3),
        "lr_latent_corr": round(lr_corr, 4),
        "n_transients": len(transients),
        "transients": transients,
        "flags": flags,
    }


def evaluate_video_x0_motion(video_x0: torch.Tensor) -> dict[str, Any]:
    """Measure phase-matched `[1, 4, 4, 4, 4]` temporal latent motion from denoised `video_x0` (`< 0.2 ms`).

    Compares adjacent 4-frame latent steps (`k % 5 in (1, 2, 3)`) and standalone 1-frame clip
    starts (`k % 5 == 0` spaced by 5) normalized by global latent std so causal chunk phase
    transitions (`1f -> 4f`) and step-to-step denoising sharpening are never confused with
    temporal subject motion.
    """
    v = video_x0.detach().to(dtype=torch.float32)
    if v.ndim == 5:
        v = v[0]
    if v.ndim != 4:
        raise ValueError(f"Expected 4D/5D video_x0 [B?, C, T_lat, H, W], got {tuple(video_x0.shape)}")
    t_lat = int(v.shape[1])
    if t_lat < 2:
        return {
            "latent_t": t_lat,
            "norm_temporal_delta": None,
            "freeze_risk": False,
            "high_motion": False,
            "flags": [],
        }
    std = float(v.std(unbiased=False).item())
    denom = max(std, 1e-5)

    deltas: list[float] = []
    for k in range(t_lat - 1):
        if (k % 5) in (1, 2, 3):
            deltas.append(float((v[:, k + 1] - v[:, k]).abs().mean().item()))
    for k in range(0, t_lat - 5, 5):
        deltas.append(float((v[:, k + 5] - v[:, k]).abs().mean().item()))
    if not deltas:
        deltas.append(float((v[:, 1] - v[:, 0]).abs().mean().item()))

    norm_delta = float(sum(deltas) / len(deltas)) / denom
    flags: list[str] = []
    freeze_risk = norm_delta < 0.12
    high_motion = norm_delta > 0.75
    if freeze_risk:
        flags.append("copy_freeze_risk")
    elif high_motion:
        flags.append("high_temporal_motion")
    return {
        "latent_t": t_lat,
        "norm_temporal_delta": round(norm_delta, 4),
        "freeze_risk": freeze_risk,
        "high_motion": high_motion,
        "flags": flags,
    }


def synthesize_audio_transient_mask(
    audio_x0: torch.Tensor,
    *,
    latent_t: int,
    patch_h: int,
    patch_w: int,
    z_thresh: float = 2.2,
    softness: float = 0.25,
    invert: bool = False,
    sparse_floor: float = 0.10,
    decay_steps: int = 2,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convert Step-0 `audio_x0` `40 Hz` onset flux (`flux_z >= z_thresh`) into a `[T, H, W]` burst mask.

    Applies a fast-attack, capped `decay_steps` (`[1.0, 0.58, 0.28]`) release tail after onset
    thresholding so thunderclaps/hits roll across ~2 latent steps instead of 1-frame strobing
    while keeping quiet spans strictly `0.0`.
    """
    a = audio_x0.detach().float().cpu()
    if a.ndim == 4:
        a = a[0]
    if a.ndim != 3:
        raise ValueError(f"Expected 3D/4D audio_x0 [B?, C, 2, T], got {tuple(audio_x0.shape)}")
    if int(a.shape[1]) == 2:
        c_dim, t_a = int(a.shape[0]), int(a.shape[2])
        left, right = a[:, 0, :], a[:, 1, :]
    elif int(a.shape[0]) == 2:
        c_dim, t_a = int(a.shape[1]), int(a.shape[2])
        left, right = a[0, :, :], a[1, :, :]
    else:
        raise ValueError(f"Cannot locate stereo channel dim=2 in audio_x0 shape {tuple(a.shape)}")

    e_l = torch.linalg.vector_norm(left, dim=0) / math.sqrt(float(max(1, c_dim)))
    e_r = torch.linalg.vector_norm(right, dim=0) / math.sqrt(float(max(1, c_dim)))
    e_st = 0.5 * (e_l + e_r)
    flux = torch.zeros_like(e_st)
    if t_a >= 2:
        flux[1:] = torch.clamp(e_st[1:] - e_st[:-1], min=0.0)
    f_std = float(flux.std(unbiased=False).item())
    f_mean = float(flux.mean().item())
    z_curve = (flux - f_mean) / max(f_std, 1e-6)

    lt = max(1, int(latent_t))
    ph = max(1, int(patch_h))
    pw = max(1, int(patch_w))
    windows = video_latent_audio_windows(lt, t_a)
    peak_z = torch.zeros(lt, dtype=torch.float32)
    for k_t, (a0, a1) in enumerate(windows):
        if a1 > a0:
            peak_z[k_t] = z_curve[a0:a1].max()

    soft = max(float(softness), 1e-3)
    zt = float(z_thresh)
    w_t = torch.sigmoid((peak_z - zt) / soft)
    fl = max(0.0, min(0.45, float(sparse_floor)))
    if fl > 0.0:
        w_t = ((w_t - fl) / max(1.0 - 2.0 * fl, 1e-4)).clamp(0.0, 1.0)
    else:
        w_t = w_t.clamp(0.0, 1.0)

    n_decay = max(0, int(decay_steps))
    if n_decay > 0 and lt > 1:
        tail_weights = (0.58, 0.28) if n_decay >= 2 else (0.58,)
        src_w = w_t.clone()
        env = w_t.clone()
        for offset, mult in enumerate(tail_weights, start=1):
            if offset < lt:
                env[offset:] = torch.maximum(env[offset:], src_w[:-offset] * float(mult))
        w_t = env.clamp(0.0, 1.0)

    if invert:
        w_t = (1.0 - w_t).clamp(0.0, 1.0)

    prob = w_t.view(lt, 1, 1).expand(lt, ph, pw).contiguous()
    frame_hard = (prob >= 0.5).float().mean(dim=(1, 2))
    frame_hard_vals = [round(float(v), 4) for v in frame_hard.tolist()]
    active_steps = [int(i) for i, v in enumerate(frame_hard_vals) if v >= 0.5]
    has_any = len(active_steps) > 0
    stats = {
        "shape_thw": [lt, ph, pw],
        "coverage_mean": round(float(prob.mean().item()), 4),
        "hard_coverage_mean": round(float((prob >= 0.5).float().mean().item()), 4),
        "frame_hard_coverage_min": min(frame_hard_vals),
        "frame_hard_coverage_max": max(frame_hard_vals),
        "silent_frames_count": int((frame_hard < 0.005).sum().item()),
        "sparse_zero_frac": round(float((prob == 0.0).float().mean().item()), 4),
        "peak_xy": [0.5, 0.5],
        "centroid_xy": [0.5, 0.5],
        "bbox_xyxy": [0.0, 0.0, 1.0, 1.0] if has_any else [0.0, 0.0, 0.0, 0.0],
        "active_latent_steps": active_steps,
        "z_thresh": round(zt, 3),
    }
    if lt > 1:
        stats["frame_hard_coverage"] = frame_hard_vals
    return prob, stats


# ------------------------------------------------------------------ Shared H3LivePolicy bus


@dataclass
class H3LivePolicy:
    """Mutable per-render sensor/policy ledger shared through `transformer_options["_h3_live_policy"]`."""

    mask_source: str = "text_phrase"
    target_phrase: str = ""
    isolation: str = "early_0_24_shield"
    audio_zone: str = "base"
    replay_step0: bool = False
    strength_scale: float = 1.0
    recommended_video_end_percent: float | None = None
    recommended_aud_to_vid_gain: float | None = None
    emit_ws_events: bool = True
    prompt_id: str | None = None

    # Phase 2 Dynamic Step-1 Auto-Shield (`early_0_24 -> early_0_24_shield`)
    force_shield: bool = False
    auto_shield_sep_threshold: float = 12.0
    auto_shield_max_coverage: float = 0.32
    active_region_box: tuple[float, float, float, float] | None = None
    active_region_invert: bool = False
    active_region_already_shielded: bool = False
    active_region_mask_hw: np.ndarray | None = field(default=None, repr=False)

    # Phase 2 Remote `heylook` Step-0 Visual Critic -> `< 3 ms` CPU WordVector(mode="project_out") auto-scrub
    critic_scrub_alpha: float = 0.85
    project_out_words: list[str] = field(default_factory=list)
    critic_history: list[dict[str, Any]] = field(default_factory=list)
    _scrub_vec_cache: dict[tuple[tuple[str, ...], str], torch.Tensor | None] = field(
        default_factory=dict, repr=False
    )
    _scrub_unit_vec_override: torch.Tensor | None = field(default=None, repr=False)

    harvested_mask_thw: torch.Tensor | None = field(default=None, repr=False)
    harvested_mask_frames: torch.Tensor | None = field(default=None, repr=False)
    mask_stats: dict[str, Any] = field(default_factory=dict)
    step_history: list[dict[str, Any]] = field(default_factory=list)
    audio_x0_history: list[dict[str, Any]] = field(default_factory=list)
    video_x0_history: list[dict[str, Any]] = field(default_factory=list)
    preview_history: list[dict[str, Any]] = field(default_factory=list)
    _prev_preview_rgb: np.ndarray | None = field(default=None, repr=False)

    def reset_for_new_run(self, *, clear_region: bool = False) -> None:
        self.harvested_mask_frames = None
        self.strength_scale = 1.0
        self.recommended_video_end_percent = None
        self.recommended_aud_to_vid_gain = None
        self.prompt_id = None
        self.force_shield = False
        if clear_region:
            self.active_region_box = None
            self.active_region_invert = False
            self.active_region_already_shielded = False
            self.active_region_mask_hw = None
        self.project_out_words = []
        self.critic_history.clear()
        self._scrub_vec_cache.clear()
        self._prev_preview_rgb = None

    def register_spatial_region(
        self,
        *,
        x_start: float = 0.0,
        x_end: float = 1.0,
        y_start: float = 0.0,
        y_end: float = 1.0,
        invert: bool = False,
        mask: torch.Tensor | None = None,
        mask_hw: np.ndarray | None = None,
        already_shielded: bool = False,
    ) -> None:
        """Register active `SpatialQueryRouter` geometry so Step-1 preview can detect surround washout."""
        self.active_region_box = (float(x_start), float(x_end), float(y_start), float(y_end))
        self.active_region_invert = bool(invert)
        self.active_region_already_shielded = bool(already_shielded)
        if mask_hw is not None:
            self.active_region_mask_hw = np.asarray(mask_hw, dtype=np.float32)
            return
        if mask is not None and torch.is_tensor(mask):
            m = mask.detach().to(device="cpu", dtype=torch.float32)
            if m.ndim == 4:
                m = m[:, 0] if int(m.shape[1]) == 1 else m[..., 0]
            if m.ndim == 3:
                m = m.mean(dim=0)
            if m.ndim == 2 and m.numel() > 0:
                if max(int(m.shape[0]), int(m.shape[1])) > 64:
                    scale = 64.0 / float(max(int(m.shape[0]), int(m.shape[1])))
                    nh = max(2, int(round(int(m.shape[0]) * scale)))
                    nw = max(2, int(round(int(m.shape[1]) * scale)))
                    m = F.interpolate(m.unsqueeze(0).unsqueeze(0), size=(nh, nw), mode="area")[0, 0]
                self.active_region_mask_hw = m.clamp(0.0, 1.0).numpy()
            else:
                self.active_region_mask_hw = None
        else:
            self.active_region_mask_hw = None

    def resolve_scrub_unit_vec(self, *, family: str = "fl2va") -> torch.Tensor | None:
        if self._scrub_unit_vec_override is not None:
            v = self._scrub_unit_vec_override.detach().to(device="cpu", dtype=torch.float32)
            nrm = float(torch.linalg.vector_norm(v).item())
            return (v / nrm) if nrm > 1e-8 else None
        if not self.project_out_words:
            return None
        key = (tuple(self.project_out_words), str(family or "fl2va"))
        if key not in self._scrub_vec_cache:
            self._scrub_vec_cache[key] = synthesize_policy_scrub_unit_vec(self.project_out_words, family=key[1])
        return self._scrub_vec_cache[key]

    def update_from_critic(self, result: Mapping[str, Any], *, step: int = 0) -> dict[str, Any]:
        """Record remote `heylook` Step-0 Visual Critic readout and arm CPU `project_out` text scrubbing."""
        rec = dict(result)
        rec["step"] = int(step)
        words_clean = [
            str(w).strip().lower()
            for w in (rec.get("scrub_words") or ())
            if str(w).strip()
        ]
        rec["scrub_words"] = words_clean
        self.critic_history.append(rec)
        if words_clean:
            self.project_out_words = list(dict.fromkeys(words_clean))[:6]
            self._scrub_vec_cache.clear()
            logger.info(
                "[h3] H3LivePolicy Step-%d Visual Critic armed CPU project_out scrub: words=%s alpha=%.2f (%s)",
                int(step),
                self.project_out_words,
                float(self.critic_scrub_alpha),
                rec.get("reason", ""),
            )
        self.emit_event({"kind": "heylook_critic", **rec})
        return rec

    def update_from_preview(
        self,
        rgb_hwc: np.ndarray,
        *,
        step: int,
        frames_batch: Sequence[np.ndarray] | None = None,
    ) -> dict[str, Any]:
        metrics = evaluate_preview_frame(rgb_hwc, prev_rgb_hwc=self._prev_preview_rgb)
        metrics["step"] = int(step)
        self._prev_preview_rgb = np.asarray(rgb_hwc, dtype=np.float32).copy()
        rec_scale = float(metrics["recommended_strength_scale"])
        flags_cur = metrics.get("flags") or ()
        if "highlight_burn" in flags_cur or "high_saturation" in flags_cur:
            # Fast downward attack so burn/oversaturation tapers immediately inside H3's early 0..50% palette window.
            self.strength_scale = round(min(self.strength_scale, rec_scale), 4)
        else:
            self.strength_scale = round(0.5 * self.strength_scale + 0.5 * rec_scale, 4)
        metrics["applied_strength_scale"] = self.strength_scale

        intra_motion_mae: float | None = None
        if frames_batch is not None and len(frames_batch) >= 2:
            diffs: list[float] = []
            prev_f = None
            for frm in frames_batch:
                arr_f = np.asarray(frm, dtype=np.float32)[..., :3]
                if float(arr_f.max()) <= 1.5:
                    arr_f = arr_f * 255.0
                if prev_f is not None and prev_f.shape == arr_f.shape:
                    diffs.append(float(np.abs(arr_f - prev_f).mean()))
                prev_f = arr_f
            if diffs:
                intra_motion_mae = round(float(sum(diffs) / len(diffs)), 4)
        metrics["intra_step_motion_mae"] = intra_motion_mae

        # Drive anti-copy-freeze strictly from true intra-step multi-frame motion (`frames_batch`),
        # never from step-to-step denoising delta (`|step1_f0 - step0_f0|`).
        if int(step) >= 1 and intra_motion_mae is not None:
            if float(intra_motion_mae) < 1.8:
                self.recommended_video_end_percent = 0.33
            elif float(intra_motion_mae) > 18.0:
                self.recommended_video_end_percent = 0.50
        metrics["recommended_video_end_percent"] = self.recommended_video_end_percent

        # Dynamic Step-1 Auto-Shield (`early_0_24 -> early_0_24_shield`): if a small routed region
        # (`<= 32%` canvas) shows `< 12.0` inside-vs-outside RGB/luma separation by Step 1, flip
        # `force_shield = True` for remaining Steps 2..5.
        reg_mask_hw = self.active_region_mask_hw
        reg_invert = self.active_region_invert
        reg_box = self.active_region_box
        if reg_mask_hw is None and self.harvested_mask_thw is not None:
            reg_mask_hw = self.harvested_mask_thw.detach().to(device="cpu", dtype=torch.float32).mean(dim=0).numpy()
            reg_invert = False  # inversion already baked into `harvested_mask_thw`
        reg_sep = evaluate_region_separation(
            rgb_hwc,
            region_box=reg_box,
            region_mask_hw=reg_mask_hw,
            invert=reg_invert,
            max_coverage=self.auto_shield_max_coverage,
            sep_threshold=self.auto_shield_sep_threshold,
        )
        if reg_sep is not None:
            metrics["region_separation"] = reg_sep
            already_sh = self.active_region_already_shielded or (
                self.active_region_box is None
                and self.active_region_mask_hw is None
                and str(self.isolation).endswith("_shield")
            )
            if int(step) >= 1 and reg_sep["washed_out"] and not self.force_shield and not already_sh:
                self.force_shield = True
                metrics["flags"] = [*list(metrics.get("flags") or ()), "auto_shield_engaged"]
                logger.info(
                    "[h3] H3LivePolicy Step-%d Auto-Shield engaged (cov=%.1f%%, sep=%.1f < %.1f): upgrading SpatialQR to *_shield",
                    int(step),
                    100.0 * float(reg_sep["region_coverage"]),
                    float(reg_sep["sep_score"]),
                    float(self.auto_shield_sep_threshold),
                )
                self.emit_event({"kind": "auto_shield_engaged", "step": int(step), **reg_sep})
        metrics["force_shield"] = self.force_shield

        self.preview_history.append(metrics)
        self.emit_event({"kind": "preview_thermostat", **metrics})
        return metrics

    def update_from_video_x0(self, video_x0: torch.Tensor, *, step: int, sigma: float | None = None) -> dict[str, Any]:
        """Update anti-copy-freeze governor from phase-matched `[1,4,4,4,4]` `video_x0` temporal motion."""
        metrics = evaluate_video_x0_motion(video_x0)
        metrics["step"] = int(step)
        if sigma is not None:
            metrics["sigma"] = round(float(sigma), 6)
        if int(step) >= 1 and metrics["norm_temporal_delta"] is not None:
            if metrics["freeze_risk"]:
                self.recommended_video_end_percent = 0.33
            elif metrics["high_motion"] and self.recommended_video_end_percent is None:
                self.recommended_video_end_percent = 0.50
        metrics["recommended_video_end_percent"] = self.recommended_video_end_percent
        self.video_x0_history.append(metrics)
        if metrics["freeze_risk"] or metrics["high_motion"]:
            self.emit_event({"kind": "preview_thermostat", **metrics})
        return metrics

    def update_from_audio_x0(self, audio_x0: torch.Tensor, *, step: int, sigma: float | None = None) -> dict[str, Any]:
        metrics = evaluate_audio_x0(
            audio_x0,
            binaural_expected=(self.audio_zone == "binaural_split"),
        )
        metrics["step"] = int(step)
        if sigma is not None:
            metrics["sigma"] = round(float(sigma), 6)
        flags = metrics.get("flags") or ()
        if int(step) >= 1 and ("stereo_collapse_risk" in flags or "visual_hiss_bleed_risk" in flags):
            self.recommended_aud_to_vid_gain = 0.0
        metrics["recommended_aud_to_vid_gain"] = self.recommended_aud_to_vid_gain
        self.audio_x0_history.append(metrics)
        self.emit_event({"kind": "audio_x0_thermostat", **metrics})
        return metrics

    def emit_event(self, payload: Mapping[str, Any]) -> bool:
        """Stream live `h3_self_steer` JSON telemetry to Studio alongside `kj_preview_override`."""
        if not self.emit_ws_events:
            return False
        try:
            import server  # ComfyUI's PromptServer (only present inside live server)

            ps = getattr(server, "PromptServer", None)
            inst = getattr(ps, "instance", None) if ps is not None else None
            send_sync = getattr(inst, "send_sync", None) if inst is not None else None
            if callable(send_sync):
                msg = dict(payload)
                pid = self.prompt_id
                if pid is None:
                    raw_pid = getattr(inst, "last_prompt_id", None)
                    if raw_pid is not None:
                        pid = str(raw_pid)
                        self.prompt_id = pid
                if pid is not None:
                    msg.setdefault("prompt_id", str(pid))
                send_sync("h3_self_steer", msg)
                return True
        except Exception:
            pass
        return False

    def to_report(self) -> dict[str, Any]:
        return {
            "mask_source": self.mask_source,
            "target_phrase": self.target_phrase,
            "isolation": self.isolation,
            "audio_zone": self.audio_zone,
            "replay_step0": self.replay_step0,
            "strength_scale": self.strength_scale,
            "recommended_video_end_percent": self.recommended_video_end_percent,
            "recommended_aud_to_vid_gain": self.recommended_aud_to_vid_gain,
            "force_shield": self.force_shield,
            "project_out_words": list(self.project_out_words),
            "critic_scrub_alpha": self.critic_scrub_alpha,
            "mask_stats": dict(self.mask_stats),
            "step_history": list(self.step_history),
            "audio_x0_history": list(self.audio_x0_history),
            "video_x0_history": list(self.video_x0_history),
            "preview_history": list(self.preview_history),
            "critic_history": list(self.critic_history),
        }


# ------------------------------------------------------------------ Step-0 Harvest + Steps 1..5 SpatialQR runtime


class _Step0HarvestState:
    """Per-step read-only `B35 + B40` attention harvester (`malloc_graph`-safe)."""

    def __init__(
        self,
        *,
        mask_source: str,
        noun_indices: Sequence[int],
        harvest_blocks: frozenset[int],
    ) -> None:
        self.mask_source = mask_source
        self.noun_indices = [int(i) for i in noun_indices]
        self.harvest_blocks = harvest_blocks
        self.tx_a = 0
        self.tx_b = 0
        self.va = 0
        self.vb = 0
        self.aa: int | None = None
        self.ab: int | None = None
        self.latent_t = 1
        self.patch_h = 1
        self.patch_w = 1
        self.audio_t = 0
        self.maps_thw: list[torch.Tensor] = []

    def register_layout(self, layout: Any) -> None:
        segs = layout.segments
        tx = [(a, b) for (a, b, k) in segs if k == "text" and b > a]
        vd = [(a, b) for (a, b, k) in segs if k == "video" and b > a]
        au = [(a, b) for (a, b, k) in segs if k == "audio" and b > a]
        if not tx or not vd:
            return
        self.tx_a, self.tx_b = int(tx[0][0]), int(tx[0][1])
        self.va, self.vb = int(vd[0][0]), int(vd[0][1])
        if au:
            self.aa, self.ab = int(au[0][0]), int(au[0][1])
        else:
            self.aa = self.ab = None
        _t_len, lt, lh, lw, at = layout.signature
        self.latent_t = max(1, int(lt))
        self.patch_h = max(1, int(lh) // 2)
        self.patch_w = max(1, int(lw) // 2)
        self.audio_t = max(0, int(at))

    def probe_block(self, block_index: int, q_p: torch.Tensor, k_p: torch.Tensor, scale: float | None) -> None:
        if self.mask_source == "audio_transients":
            return
        if block_index not in self.harvest_blocks:
            return
        va, vb = self.va, self.vb
        if vb <= va or int(q_p.shape[2]) < vb:
            return
        if self.latent_t * self.patch_h * self.patch_w != (vb - va):
            return
        head_dim = int(q_p.shape[-1])
        scale_eff = float(scale) if scale is not None else (1.0 / math.sqrt(float(max(1, head_dim))))
        q_vid = q_p[0, :, va:vb, :].detach().float()  # [H, S_vid, D]

        if self.mask_source == "text_phrase":
            l_txt = self.tx_b - self.tx_a
            valid = [self.tx_a + i for i in self.noun_indices if 0 <= i < l_txt]
            if not valid:
                return
            idx_t = torch.tensor(valid, dtype=torch.long, device=k_p.device)
            k_span = k_p[0].index_select(1, idx_t).detach().float()  # [H, K, D]
            logits = torch.einsum("hsd,hkd->hsk", q_vid, k_span) * scale_eff
            score = logits.mean(dim=(0, 2)).reshape(self.latent_t, self.patch_h, self.patch_w).cpu().clone()
            self.maps_thw.append(score)
            del idx_t, k_span, logits
            return

        # Audio-driven speaker spotlight (`active_speaker`, `left_speaker`, `right_speaker`)
        if self.aa is None or self.ab is None or self.audio_t <= 0 or (self.ab - self.aa) < self.audio_t * 2:
            return
        n_hw = self.patch_h * self.patch_w
        windows = video_latent_audio_windows(self.latent_t, self.audio_t)
        k_full = k_p[0].detach().float()
        out_thw = torch.empty((self.latent_t, self.patch_h, self.patch_w), dtype=torch.float32)
        for k_t, (a0, a1) in enumerate(windows):
            qf = q_vid[:, k_t * n_hw : (k_t + 1) * n_hw, :]
            if self.mask_source == "left_speaker":
                ka = k_full[:, self.aa + a0 : self.aa + a1, :]
                sc = (torch.einsum("hsd,had->hsa", qf, ka) * scale_eff).mean(dim=(0, 2))
            elif self.mask_source == "right_speaker":
                ka = k_full[:, self.aa + self.audio_t + a0 : self.aa + self.audio_t + a1, :]
                sc = (torch.einsum("hsd,had->hsa", qf, ka) * scale_eff).mean(dim=(0, 2))
            else:
                kal = k_full[:, self.aa + a0 : self.aa + a1, :]
                kar = k_full[:, self.aa + self.audio_t + a0 : self.aa + self.audio_t + a1, :]
                sl = (torch.einsum("hsd,had->hsa", qf, kal) * scale_eff).mean(dim=(0, 2))
                sr = (torch.einsum("hsd,had->hsa", qf, kar) * scale_eff).mean(dim=(0, 2))
                sc = 0.5 * (sl + sr)
            out_thw[k_t] = sc.reshape(self.patch_h, self.patch_w).cpu()
        self.maps_thw.append(out_thw)
        del k_full

    def fused_map(self) -> torch.Tensor | None:
        if not self.maps_thw:
            return None
        return torch.stack(self.maps_thw, dim=0).mean(dim=0)


class _SelfMaskBlockPatch:
    """Stateless per-clone `patches_replace['dit'][('double_block', i)]` dual-phase adapter."""

    def __init__(self, block_index: int, previous: Any = None) -> None:
        self.block_index = int(block_index)
        self.previous = previous

    def __call__(self, args: dict[str, Any], extra_args: dict[str, Any]) -> dict[str, Any]:
        to = args.get("transformer_options")
        harvest = to.get("_h3_self_mask_harvest") if isinstance(to, dict) else None
        if harvest is not None and self.block_index == 0:
            with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
                harvest.register_layout(args["layout"])

        rt = to.get("_h3_spatial_qr_rt") if isinstance(to, dict) else None
        if rt is not None and getattr(rt, "active", False):
            with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
                rt.prepare_block(self.block_index, args)

        orig_blk = extra_args["original_block"]
        out = orig_blk(args) if self.previous is None else self.previous(args, extra_args)

        if rt is not None and getattr(rt, "active", False):
            with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
                rt.finish_block(self.block_index)
        return out

    def to(self, device_or_dtype: Any):
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def cleanup(self) -> None:
        if hasattr(self.previous, "cleanup"):
            self.previous.cleanup()

    def models(self) -> list[Any]:
        return list(self.previous.models()) if hasattr(self.previous, "models") else []


def _make_harvest_attn_override(previous: Any = None):
    """Peek-only attention override for Step-0 `B35 + B40` harvest (zero mutation)."""

    def override(
        func,
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        options = kwargs.get("transformer_options")
        harvest = options.get("_h3_self_mask_harvest") if isinstance(options, dict) else None
        b_idx = options.get("block_index") if isinstance(options, dict) else None
        if harvest is not None and skip_reshape and b_idx is not None:
            q_p = q if torch.is_tensor(q) else q.peek()
            k_p = k if torch.is_tensor(k) else k.peek()
            if q_p.ndim == 4 and k_p.ndim == 4:
                with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
                    harvest.probe_block(int(b_idx), q_p, k_p, scale=kwargs.get("scale", None))
            del q_p, k_p

        q_u, k_u, v_u = q, k, v
        if not torch.is_tensor(q_u):
            if previous is not None and getattr(previous, "container_function", None) is not None:
                return previous.container_function(
                    q_u,
                    k_u,
                    v_u,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    **kwargs,
                )
            q_u, k_u, v_u = q_u.take(), k_u.take(), v_u.take()
        target = func if previous is None else functools.partial(previous, func)
        return target(
            q_u,
            k_u,
            v_u,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    if getattr(previous, "container_function", None) is not None:
        override.container_function = functools.partial(override, None)
    return override


def _has_self_mask_block_patch(patch: Any) -> bool:
    cur = patch
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, _SelfMaskBlockPatch):
            return True
        seen.add(id(cur))
        cur = getattr(cur, "previous", None)
    return False


def _resolve_harvest_blocks(n_blocks: int, harvest_blocks: Sequence[int] | None) -> frozenset[int]:
    if harvest_blocks is not None:
        valid = frozenset(int(b) for b in harvest_blocks if 0 <= int(b) < n_blocks)
        if valid:
            return valid
    default_valid = frozenset(b for b in GOLDEN_SEMANTIC_BLOCKS if b < n_blocks)
    if default_valid:
        return default_valid
    # Small synthetic test DiTs (e.g. 4 blocks): pick the ~70%–85% semantic band automatically.
    hi = max(0, n_blocks - 1)
    lo = max(0, int(round(0.7 * (n_blocks - 1))))
    return frozenset({lo, hi})


def attach_self_mask_router(
    model: Any,
    *,
    region_conditioning: Any = None,
    mask_source: str = "text_phrase",
    target_phrase: str = "",
    clip: Any = None,
    base_prompt: str = "",
    token_pieces: Sequence[str] | None = None,
    noun_indices: Sequence[int] | None = None,
    threshold_z: float = 1.35,
    softness: float = 0.25,
    dilation_passes: int = 0,
    isolation: str = "early_0_24_shield",
    audio_zone: str = "base",
    invert: bool = False,
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    replay_step0: bool = False,
    cross_time_scale_floor: float = 0.55,
    harvest_blocks: Sequence[int] | None = None,
    enable_audio_x0_thermostat: bool = True,
    emit_ws_events: bool = True,
    enable_heylook_critic: bool = False,
    heylook_base_url: str = "",
    heylook_model: str = "",
    heylook_timeout_s: float = 18.0,
    critic_scrub_alpha: float = 0.85,
    critic_fn: Callable[[np.ndarray], Mapping[str, Any]] | None = None,
    auto_shield_sep_threshold: float = 12.0,
    auto_shield_max_coverage: float = 0.32,
) -> tuple[Any, H3LivePolicy]:
    """Clone `model` and attach zero-coordinate Self-Masked Routing + live `H3LivePolicy` bus.

    - When `region_conditioning` is wired and not inside an `EditGuider` pass (`h3_edit_pass`
      not in `("original", "edited")`), Phase 1 harvests a 4D mask on Step 0 (`B35 + B40`, or
      Step-0 `audio_x0` onset bursts for `mask_source="audio_transients"`) and Phase 2 routes
      `region_conditioning` via `SpatialVisualQRRuntime` (`1.0x` default, or `1.17x` when
      `replay_step0=True`).
    - When `region_conditioning is None` or inside `MiniMaxH3EditGuider`, Phase-2 `SpatialQR` is
      skipped while `m._h3_live_policy` (`transformer_options["_h3_live_policy"]`) supplies
      `harvested_mask_frames` to `EditGuider._effective_raw_mask()` and drives the closed-loop
      Visual/Acoustic/Anti-Copy-Freeze thermostats across `steering/nodes.py` and
      `steering/query_router.py`.

    Returns `(cloned_model, policy)` so callers & tests can inspect the live harvested mask,
    bounding box, `audio_x0` telemetry, and thermostat state cleanly.
    """
    src = str(mask_source or "text_phrase").strip()
    if src not in MASK_SOURCES:
        raise ValueError(f"mask_source must be one of {MASK_SOURCES}, got {mask_source!r}")
    iso_mode = str(isolation or "early_0_24_shield").strip()
    if iso_mode not in ISOLATION_MODES:
        raise ValueError(f"isolation must be one of {ISOLATION_MODES}, got {isolation!r}")
    aud_z = str(audio_zone or "base").strip()
    if aud_z not in AUDIO_ZONES:
        raise ValueError(f"audio_zone must be one of {AUDIO_ZONES}, got {audio_zone!r}")
    if src == "text_phrase" and not (str(target_phrase or "").strip() or (noun_indices and len(noun_indices) > 0)):
        raise ValueError("MiniMaxH3SelfMaskRouter(mask_source='text_phrase') requires non-empty target_phrase or noun_indices")

    s_pct = float(start_percent)
    e_pct = float(end_percent)
    if not (math.isfinite(s_pct) and math.isfinite(e_pct)) or s_pct < 0.0 or e_pct > 1.0:
        raise ValueError(f"start_percent/end_percent must lie in [0.0, 1.0], got [{s_pct}, {e_pct}]")

    thermostat_only = src == "thermostat_only"
    raw_b = (
        None
        if (thermostat_only or region_conditioning is None)
        else _extract_cond_tensor(region_conditioning, name="region_conditioning")
    )
    tags_b = None if raw_b is None else _extract_cond_tags(region_conditioning)

    m = model.clone()
    dm = getattr(getattr(m, "model", None), "diffusion_model", None)
    if not _steer_hooks.is_hookable_dit(dm):
        raise ValueError("MiniMaxH3SelfMaskRouter requires a MiniMax H3 model with hookable DiT blocks")

    from comfy.patcher_extension import CallbacksMP, WrappersMP

    t_opts = m.model_options.setdefault("transformer_options", {})
    dm_wrappers = m.wrappers.get(WrappersMP.DIFFUSION_MODEL, {})
    dit_replaces = t_opts.get("patches_replace", {}).get("dit", {})

    if SELF_MASK_WRAPPER_KEY in dm_wrappers:
        raise ValueError("MiniMaxH3SelfMaskRouter cannot stack with another MiniMaxH3SelfMaskRouter")

    if not thermostat_only:
        if (
            hasattr(model, "_h3_frozen_video_cache")
            or hasattr(m, "_h3_frozen_video_cache")
            or any("frozen_video_cache" in str(wk) for wk in dm_wrappers)
        ):
            raise ValueError("MiniMaxH3SelfMaskRouter cannot stack with MiniMaxH3FrozenVideoCache")

        if any(_has_self_mask_block_patch(bp) or _has_spatial_block_patch(bp) for bp in dit_replaces.values()):
            raise ValueError(
                "MiniMaxH3SelfMaskRouter cannot stack with another MiniMaxH3SelfMaskRouter or MiniMaxH3SpatialQueryRouter"
            )

        for b_patch in dit_replaces.values():
            b_mod = getattr(b_patch, "__module__", "") or ""
            b_qname = getattr(b_patch, "__qualname__", "") or ""
            if "nodes_sparse_attention" in b_mod or "make_h3_block_patch" in b_qname:
                raise ValueError(
                    "MiniMaxH3SelfMaskRouter requires dense optimized_attention or MiniMaxH3Sol, "
                    "not core chunked BlockSparseAttention"
                )

        for key, patched in getattr(m, "object_patches", {}).items():
            if not str(key).endswith(".forward"):
                continue
            parts = str(key).rsplit(".", 2)
            owner = parts[-2].lower() if len(parts) >= 2 else ""
            if "attn" not in owner or "cross" in owner or owner == "attn2":
                continue
            if getattr(patched, "_uses_optimized_attention", False):
                continue
            raise ValueError(
                f"MiniMaxH3SelfMaskRouter cannot run with object-patched attention forward ({key}, e.g. "
                "MiniMaxH3SageAttention). Use ModelAttentionBackend('comfy kitchen attention') with optional MiniMaxH3Sol."
            )

    n_blocks = len(dm.blocks)
    from .spatial_query_router import parse_isolation_blocks

    iso_set = parse_isolation_blocks(iso_mode, n_blocks)
    shield_region = iso_mode.endswith("_shield")
    h_blocks = _resolve_harvest_blocks(n_blocks, harvest_blocks)

    if not thermostat_only:
        for b_idx in range(n_blocks):
            existing = dit_replaces.get(("double_block", b_idx))
            m.set_model_patch_replace(_SelfMaskBlockPatch(b_idx, existing), "dit", "double_block", b_idx)

    policy = H3LivePolicy(
        mask_source=src,
        target_phrase=str(target_phrase or "").strip(),
        isolation=iso_mode,
        audio_zone=aud_z,
        replay_step0=bool(replay_step0),
        emit_ws_events=bool(emit_ws_events),
        auto_shield_sep_threshold=float(auto_shield_sep_threshold),
        auto_shield_max_coverage=float(auto_shield_max_coverage),
        critic_scrub_alpha=float(critic_scrub_alpha),
    )
    eff_critic_enabled = (
        bool(enable_heylook_critic)
        or (critic_fn is not None)
        or (str(os.environ.get("H3_LIVE_CRITIC", "")).strip().lower() in ("1", "true", "yes", "on"))
    )
    # Expose on BOTH the ModelPatcher attribute and `transformer_options` (preserved across `.clone()`).
    m._h3_live_policy = policy
    t_opts[LIVE_POLICY_KEY] = policy

    ref_cache: OrderedDict[tuple[Any, ...], torch.Tensor] = OrderedDict()
    step_state: dict[str, Any] = {"step_count": 0, "fresh_run": True, "last_layout": None}

    def _on_cleanup(*_args, **_kwargs) -> None:
        ref_cache.clear()
        step_state["step_count"] = 0
        step_state["fresh_run"] = True
        step_state["last_layout"] = None

    m.add_callback_with_key(CallbacksMP.ON_CLEANUP, f"{SELF_MASK_WRAPPER_KEY}_cleanup", _on_cleanup)

    def _resolve_noun_idxs(context: torch.Tensor, kwargs: Mapping[str, Any]) -> list[int]:
        if src != "text_phrase":
            return []
        if noun_indices is not None and len(noun_indices) > 0:
            return [int(i) for i in noun_indices]
        txt_len = int(context.shape[1])
        pieces = list(token_pieces) if token_pieces is not None else None
        if pieces is None and clip is not None:
            prompt_eff = str(base_prompt or "").strip()
            if not prompt_eff:
                payload = kwargs.get("minimax_payload") or {}
                sm = kwargs.get("h3_span_masks") or payload.get("h3_span_masks")
                if sm is None and payload.get("text_token_tags") is not None:
                    sm = getattr(payload.get("text_token_tags"), "_h3_span_masks", None)
                if isinstance(sm, dict) and "scene" in sm:
                    prompt_eff = str(getattr(sm["scene"], "_h3_tail_text", "") or "").strip()
            if prompt_eff:
                pieces = extract_aligned_token_pieces(clip, prompt_eff, txt_len)
        if pieces is None:
            raise ValueError(
                "MiniMaxH3SelfMaskRouter(mask_source='text_phrase') needs either `noun_indices`, "
                "`token_pieces`, or `clip` + `base_prompt` (or conditioning with `_h3_tail_text`) "
                "to locate target_phrase token spans."
            )
        idxs = resolve_phrase_token_indices(pieces, str(target_phrase))
        if not idxs:
            raise ValueError(
                f"MiniMaxH3SelfMaskRouter: target_phrase={target_phrase!r} matched 0 tokens in prompt"
            )
        return idxs

    def _wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        cond_or_uncond = (
            transformer_options.get("cond_or_uncond") if isinstance(transformer_options, dict) else None
        )
        if isinstance(cond_or_uncond, (list, tuple)) and cond_or_uncond and 0 not in cond_or_uncond:
            return executor(x, timestep, context, transformer_options, **kwargs)

        cur_step = int(step_state["step_count"])
        step_state["step_count"] = cur_step + 1

        if s_pct >= e_pct or not _steer_hooks.step_in_window(transformer_options, timestep, s_pct, e_pct):
            return executor(x, timestep, context, transformer_options, **kwargs)

        if step_state["fresh_run"]:
            step_state["fresh_run"] = False
            policy.reset_for_new_run()

        payload = kwargs.get("minimax_payload") or {}
        tags_cur = payload.get("text_token_tags") if isinstance(payload, Mapping) else None
        if policy.project_out_words:
            with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
                context, did_scrub = apply_live_policy_text_scrub(
                    context,
                    tags_cur,
                    policy,
                    dm=dm,
                    transformer_options=transformer_options,
                )
            if did_scrub:
                policy.step_history.append(
                    {
                        "step": cur_step,
                        "phase": "critic_scrub",
                        "words": list(policy.project_out_words),
                        "alpha": float(policy.critic_scrub_alpha),
                    }
                )

        if thermostat_only:
            to_pass = dict(transformer_options or {})
            to_pass[LIVE_POLICY_KEY] = policy
            return executor(x, timestep, context, to_pass, **kwargs)

        edit_pass = payload.get("h3_edit_pass") if isinstance(payload, Mapping) else None
        can_route_phase2 = (raw_b is not None) and (edit_pass not in ("original", "edited"))

        if can_route_phase2 and isinstance(transformer_options, dict) and transformer_options.get("_h3_steer_runtime") is not None:
            raise ValueError(
                "MiniMaxH3SelfMaskRouter cannot share the same sampler step with an active 1x H3SteerRuntime"
            )

        # Phase 1 (Step 0, or before a mask has been harvested): unmutated pass harvesting B35 + B40
        # (or capturing PackedLayout dims for post-CFG `audio_transients` burst synthesis).
        replayed_step0 = False
        if policy.harvested_mask_frames is None:
            with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
                idxs = _resolve_noun_idxs(context, kwargs)
                harvest = _Step0HarvestState(
                    mask_source=src,
                    noun_indices=idxs,
                    harvest_blocks=h_blocks,
                )
                to_next = dict(transformer_options or {})
                to_next[LIVE_POLICY_KEY] = policy
                to_next["_h3_self_mask_harvest"] = harvest
                if src != "audio_transients":
                    to_next["optimized_attention_override"] = _make_harvest_attn_override(
                        to_next.get("optimized_attention_override")
                    )
            try:
                out = executor(x, timestep, context, to_next, **kwargs)
            finally:
                with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
                    step_state["last_layout"] = (
                        int(harvest.latent_t),
                        int(harvest.patch_h),
                        int(harvest.patch_w),
                        int(harvest.audio_t),
                    )
                    if src != "audio_transients":
                        fused = harvest.fused_map()
                        if fused is not None:
                            m_thw, m_stats = synthesize_4d_attention_mask(
                                fused,
                                threshold_z=float(threshold_z),
                                softness=float(softness),
                                dilation_passes=int(dilation_passes),
                                invert=bool(invert),
                                cross_time_scale_floor=float(cross_time_scale_floor),
                            )
                            policy.harvested_mask_thw = m_thw
                            policy.harvested_mask_frames = expand_latent_mask_to_frames(m_thw)
                            policy.mask_stats = m_stats
                            policy.step_history.append(
                                {
                                    "step": cur_step,
                                    "phase": "harvest",
                                    "blocks": sorted(h_blocks),
                                    "noun_indices": idxs,
                                    **m_stats,
                                }
                            )
                            policy.emit_event(
                                {
                                    "kind": "self_mask_locked",
                                    "step": cur_step,
                                    "mask_source": src,
                                    "target_phrase": policy.target_phrase,
                                    **m_stats,
                                }
                            )
                            logger.info(
                                "[h3] SelfMaskRouter locked Step-%d mask (source=%s, phrase=%r, cov=%.1f%%, bbox=%s)",
                                cur_step,
                                src,
                                policy.target_phrase,
                                100.0 * float(m_stats["hard_coverage_mean"]),
                                m_stats["bbox_xyxy"],
                            )
                        else:
                            logger.warning(
                                "[h3] SelfMaskRouter harvested 0 attention maps on Step %d (blocks=%s); "
                                "subsequent steps will pass through unchanged.",
                                cur_step,
                                sorted(h_blocks),
                            )
            if not (replay_step0 and can_route_phase2 and policy.harvested_mask_frames is not None):
                return out
            # `1.17x` mode: discard throwaway Step-0 harvest output and immediately replay Step 0
            # routed through Phase 2 (`SpatialVisualQRRuntime`) on the newly locked mask.
            del out
            replayed_step0 = True

        # Skip Phase-2 SpatialQR when operating in Sensor/Policy-only or `MiniMaxH3EditGuider` mode.
        if not can_route_phase2:
            to_pass = dict(transformer_options or {})
            to_pass[LIVE_POLICY_KEY] = policy
            return executor(x, timestep, context, to_pass, **kwargs)

        # Phase 2 (Steps 1..5, or replayed Step 0): route `region_conditioning` strictly onto the harvested mask!
        with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
            cache_key = (id(raw_b), str(context.device), str(context.dtype))
            ctx_b = ref_cache.get(cache_key)
            if ctx_b is None:
                ctx_b = _steer_hooks.refine_text_context(
                    dm,
                    raw_b.to(device=context.device, dtype=context.dtype),
                    transformer_options,
                    caller_name="MiniMaxH3SelfMaskRouter",
                )
                ref_cache[cache_key] = ctx_b
                while len(ref_cache) > _REF_CACHE_MAX:
                    ref_cache.popitem(last=False)
            ctx_b_use = ctx_b
            if policy.project_out_words:
                ctx_b_use, _ = apply_live_policy_text_scrub(
                    ctx_b,
                    tags_b,
                    policy,
                    dm=dm,
                    transformer_options=transformer_options,
                )
            eff_shield = bool(shield_region) or bool(policy.force_shield)
            rt = SpatialVisualQRRuntime(
                dm=dm,
                shadow_b_init=ctx_b_use,
                shadow_b_tags=tags_b,
                x_start=0.0,
                x_end=1.0,
                y_start=0.0,
                y_end=1.0,
                invert=False,  # inversion already baked into `harvested_mask_frames`
                mask=policy.harvested_mask_frames,
                isolate_blocks=iso_set,
                audio_zone=aud_z,
                shield_region=eff_shield,
            )
            to_next = dict(transformer_options or {})
            to_next[LIVE_POLICY_KEY] = policy
            to_next["_h3_spatial_qr_rt"] = rt
            to_next["optimized_attention_override"] = _make_spatial_attn_override(
                to_next.get("optimized_attention_override")
            )
        try:
            out = executor(x, timestep, context, to_next, **kwargs)
            if rt.active and rt.h_shadow_b is None:
                raise RuntimeError(
                    "MiniMaxH3SelfMaskRouter block hooks never ran (e.g. downstream core "
                    "BlockSparseAttention overwrote patches_replace['dit'][('double_block', i)])."
                )
            rec_step: dict[str, Any] = {"step": cur_step, "phase": "routed", "isolation": iso_mode}
            if replayed_step0:
                rec_step["replayed_step0"] = True
            policy.step_history.append(rec_step)
            return out
        finally:
            with _steer_hooks.pause_malloc_graph_ctx():
                rt.clear()

    wrappers = m.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
    wrappers[SELF_MASK_WRAPPER_KEY] = [_wrapper]

    # Bridge `ModelPreviewOverrideKJ` (`kj_wrapper.frames[-1]`, 0.0 ms extra decode) -> `policy.update_from_preview`
    # regardless of whether KJ was wired upstream or downstream of `MiniMaxH3SelfMaskRouter`.
    if hasattr(m, "add_wrapper_with_key"):
        def _preview_bridge_outer(
            executor,
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask=None,
            callback=None,
            disable_pbar=False,
            seed=None,
            **outer_kwargs,
        ):
            guider_obj = getattr(executor, "class_obj", None)
            mp = getattr(guider_obj, "model_patcher", None) or m
            kj_list: list[Any] = []
            if mp is not None and hasattr(mp, "get_wrappers"):
                try:
                    kj_list.extend(mp.get_wrappers(WrappersMP.OUTER_SAMPLE, "kj_preview_override") or ())
                except Exception:
                    pass
            seen = [len(getattr(kj, "frames", ()) or ()) for kj in kj_list]

            def _drain_kj_frames(step_i: int) -> None:
                for idx_k, kj in enumerate(kj_list):
                    frms = getattr(kj, "frames", None)
                    if not isinstance(frms, list):
                        continue
                    cur_n = len(frms)
                    if cur_n > seen[idx_k]:
                        new_slice = frms[seen[idx_k] : cur_n]
                        seen[idx_k] = cur_n
                        last_f = new_slice[-1]
                        try:
                            rgb_arr = np.asarray(last_f, dtype=np.float32)
                            if rgb_arr.ndim == 3 and rgb_arr.shape[2] >= 3:
                                policy.update_from_preview(rgb_arr, step=int(step_i))
                                if eff_critic_enabled and int(step_i) == 0 and not policy.critic_history:
                                    c_res = (
                                        critic_fn(rgb_arr)
                                        if critic_fn is not None
                                        else run_heylook_step0_critic(
                                            rgb_arr,
                                            base_url=heylook_base_url,
                                            model=heylook_model,
                                            timeout_s=heylook_timeout_s,
                                        )
                                    )
                                    if isinstance(c_res, Mapping):
                                        policy.update_from_critic(c_res, step=int(step_i))
                        except Exception as exc:
                            logger.debug("[h3] preview thermostat bridge skipped step %s: %s", step_i, exc)

            def _bridge_cb(step_i, x0, x_cur, total_steps_):
                _drain_kj_frames(step_i)
                if callback is not None:
                    callback(step_i, x0, x_cur, total_steps_)
                _drain_kj_frames(step_i)

            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                _bridge_cb,
                disable_pbar,
                seed,
                **outer_kwargs,
            )

        m.add_wrapper_with_key(
            WrappersMP.OUTER_SAMPLE,
            f"{SELF_MASK_WRAPPER_KEY}_preview_bridge",
            _preview_bridge_outer,
        )

    need_post_cfg = bool(enable_audio_x0_thermostat) or (src in ("audio_transients", "thermostat_only"))
    if need_post_cfg and hasattr(m, "set_model_sampler_post_cfg_function"):
        def _post_cfg(args: Mapping[str, Any]):
            denoised = args.get("denoised")
            mdl = args.get("model")
            shapes = getattr(mdl, "latent_shapes", None)
            if denoised is not None and shapes is not None and len(shapes) >= 2:
                cur_s = max(0, int(step_state["step_count"]) - 1)
                if cur_s in (0, 1, 2):
                    with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
                        import comfy.utils

                        streams = comfy.utils.unpack_latents(denoised, shapes)
                        if len(streams) >= 2:
                            sig = args.get("sigma")
                            sig_f = (
                                float(torch.as_tensor(sig).detach().flatten()[0].item())
                                if sig is not None
                                else None
                            )
                            if torch.is_tensor(streams[0]) and streams[0].ndim >= 4 and int(streams[0].shape[-3]) >= 2:
                                policy.update_from_video_x0(streams[0], step=cur_s, sigma=sig_f)
                            if enable_audio_x0_thermostat:
                                policy.update_from_audio_x0(streams[1], step=cur_s, sigma=sig_f)
                            if src == "audio_transients" and policy.harvested_mask_frames is None:
                                ll = step_state.get("last_layout")
                                if ll is not None and ll[0] >= 1 and ll[1] >= 1 and ll[2] >= 1:
                                    lt, ph, pw = int(ll[0]), int(ll[1]), int(ll[2])
                                else:
                                    v_sh = shapes[0]
                                    lt = max(1, int(v_sh[2]))
                                    ph = max(1, int(v_sh[3]) // 2)
                                    pw = max(1, int(v_sh[4]) // 2)
                                m_thw, m_stats = synthesize_audio_transient_mask(
                                    streams[1],
                                    latent_t=lt,
                                    patch_h=ph,
                                    patch_w=pw,
                                    z_thresh=max(2.2, float(threshold_z)),
                                    softness=float(softness),
                                    invert=bool(invert),
                                )
                                policy.harvested_mask_thw = m_thw
                                policy.harvested_mask_frames = expand_latent_mask_to_frames(m_thw)
                                policy.mask_stats = m_stats
                                policy.step_history.append(
                                    {
                                        "step": cur_s,
                                        "phase": "harvest",
                                        "blocks": [],
                                        "noun_indices": [],
                                        **m_stats,
                                    }
                                )
                                policy.emit_event(
                                    {
                                        "kind": "self_mask_locked",
                                        "step": cur_s,
                                        "mask_source": src,
                                        "target_phrase": policy.target_phrase,
                                        **m_stats,
                                    }
                                )
            return denoised

        m.set_model_sampler_post_cfg_function(_post_cfg)

    return m, policy


class MiniMaxH3SelfMaskRouter(io.ComfyNode):
    """Zero-coordinate, zero-SAM3 `1x` Self-Masked Router (Step-0 `B35+B40` harvest -> Steps `1..5` SpatialQR / EditGuider)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3SelfMaskRouter",
            display_name="MiniMax H3 Self-Mask Router",
            category="MiniMax H3/Steering",
            description=(
                "Harvest a zero-coordinate 4D object, active-speaker, or audio-transient mask from H3's "
                "Step-0 B35+B40 attention (or audio_x0 onsets) and route region_conditioning onto that target "
                "(or feed the harvested mask and H3LivePolicy thermostats into MiniMaxH3EditGuider / Steer / QueryRouter)."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input(
                    "region_conditioning",
                    optional=True,
                    tooltip=(
                        "Optional second prompt CONDITIONING routed onto the self-masked target in 1x SpatialQR mode. "
                        "Leave unwired to harvest masks & run H3LivePolicy thermostats for MiniMaxH3EditGuider / Steer / QueryRouter."
                    ),
                ),
                io.Combo.Input(
                    "mask_source",
                    options=list(MASK_SOURCES),
                    default="text_phrase",
                    tooltip=(
                        "'text_phrase': localize target_phrase via Q_vid @ K_txt[phrase]^T (B35+B40); "
                        "'active_speaker'/'left_speaker'/'right_speaker': zero-training speaker spotlight "
                        "via Q_vid @ K_aud[t]^T (B35+B40); 'audio_transients': Step-0 audio_x0 onset burst mask "
                        "(flux_z >= 2.2 with 2-step smooth decay); 'thermostat_only': pure H3LivePolicy Visual/"
                        "Video-Freeze/Audio-Firewall thermostats with zero attention hooks (stacks with any router/EditGuider)."
                    ),
                ),
                io.String.Input(
                    "target_phrase",
                    default="",
                    tooltip="Comma-separated noun(s) or phrase(s) from the base prompt to mask (e.g. 'bread', 'olive oil').",
                ),
                io.Float.Input(
                    "threshold_z",
                    default=1.35,
                    min=0.20,
                    max=3.00,
                    step=0.05,
                    tooltip="Spatial z-score threshold above frame median (higher = tighter mask around target peak; 1.35 calibrated on 4 MP H3-Image).",
                ),
                io.Float.Input(
                    "softness",
                    default=0.25,
                    min=0.05,
                    max=2.00,
                    step=0.05,
                    tooltip="Sigmoid transition width for feathered mask edges.",
                ),
                io.Int.Input(
                    "dilation_passes",
                    default=0,
                    min=0,
                    max=4,
                    step=1,
                    tooltip="3x3 patch-space max-pool dilation passes before feathering (0 hugs tight silhouettes; 1 widens thin props).",
                ),
                io.Combo.Input(
                    "isolation",
                    options=list(ISOLATION_MODES),
                    default="early_0_24_shield",
                    tooltip="Spatial QR isolation mode used on Steps 1..5 (default 'early_0_24_shield' resists surround washout).",
                ),
                io.Combo.Input(
                    "audio_zone",
                    options=list(AUDIO_ZONES),
                    default="base",
                    tooltip="Stereo audio routing mode for Steps 1..5 (ignored cleanly on MiniMax-H3-Image).",
                ),
                io.Boolean.Input(
                    "invert",
                    default=False,
                    tooltip="False: route region_conditioning onto the target. True: route onto everything EXCEPT the target.",
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Normalized sampler start fraction [0.0, 1.0).",
                ),
                io.Float.Input(
                    "end_percent",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Normalized sampler end fraction (0.0, 1.0].",
                ),
                io.Clip.Input(
                    "clip",
                    optional=True,
                    tooltip="Optional H3 CLIP/Qwen3-VL encoder used to tokenize base_prompt for text_phrase alignment.",
                ),
                io.String.Input(
                    "base_prompt",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Base prompt text (optional if conditioning carries _h3_span_masks._h3_tail_text).",
                ),
                io.Boolean.Input(
                    "replay_step0",
                    default=False,
                    tooltip=(
                        "False (default 1.0x): keep Step-0 harvest output and route Steps 1..5. "
                        "True (1.17x): drop throwaway Step-0 harvest and immediately replay Step 0 routed "
                        "so dominant centerpieces adopt region_conditioning from the first step with soft shield seams."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        region_conditioning: Any = None,
        mask_source: str = "text_phrase",
        target_phrase: str = "",
        threshold_z: float = 1.35,
        softness: float = 0.25,
        dilation_passes: int = 0,
        isolation: str = "early_0_24_shield",
        audio_zone: str = "base",
        invert: bool = False,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        clip: Any = None,
        base_prompt: str = "",
        replay_step0: bool = False,
    ) -> io.NodeOutput:
        routed, _policy = attach_self_mask_router(
            model,
            region_conditioning=region_conditioning,
            mask_source=mask_source,
            target_phrase=target_phrase,
            clip=clip,
            base_prompt=base_prompt,
            threshold_z=threshold_z,
            softness=softness,
            dilation_passes=dilation_passes,
            isolation=isolation,
            audio_zone=audio_zone,
            invert=invert,
            start_percent=start_percent,
            end_percent=end_percent,
            replay_step0=replay_step0,
        )
        return io.NodeOutput(routed)
