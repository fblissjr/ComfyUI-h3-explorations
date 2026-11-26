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
import functools
import logging
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

from . import hooks as _steer_hooks
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
        scale *= max(0.60, 1.0 - 0.35 * excess)
    if hi_clip > clip_ceiling:
        flags.append("highlight_burn")
        excess = min(1.0, (hi_clip - clip_ceiling) / max(0.5 - clip_ceiling, 0.05))
        scale *= max(0.65, 1.0 - 0.30 * excess)
    if lo_clip > 0.65:
        flags.append("shadow_crush")
        scale *= 0.85
    elif mean_luma < 0.24 and "high_saturation" not in flags and "highlight_burn" not in flags:
        flags.append("dim_understeer_risk")
        scale *= 1.12
    if motion_mae is not None and motion_mae < freeze_mae_floor:
        flags.append("low_motion_freeze_risk")

    scale = max(0.55, min(1.15, float(scale)))
    return {
        "mean_luma": round(mean_luma, 4),
        "mean_sat": round(mean_sat, 4),
        "hi_clip_frac": round(hi_clip, 4),
        "lo_clip_frac": round(lo_clip, 4),
        "motion_mae": None if motion_mae is None else round(motion_mae, 4),
        "flags": flags,
        "recommended_strength_scale": round(scale, 4),
    }


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
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convert Step-0 `audio_x0` `40 Hz` onset flux (`flux_z >= z_thresh`) into a `[T, H, W]` burst mask."""
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

    harvested_mask_thw: torch.Tensor | None = field(default=None, repr=False)
    harvested_mask_frames: torch.Tensor | None = field(default=None, repr=False)
    mask_stats: dict[str, Any] = field(default_factory=dict)
    step_history: list[dict[str, Any]] = field(default_factory=list)
    audio_x0_history: list[dict[str, Any]] = field(default_factory=list)
    preview_history: list[dict[str, Any]] = field(default_factory=list)
    _prev_preview_rgb: np.ndarray | None = field(default=None, repr=False)

    def reset_for_new_run(self) -> None:
        self.harvested_mask_frames = None
        self.strength_scale = 1.0
        self.recommended_video_end_percent = None
        self.recommended_aud_to_vid_gain = None
        self._prev_preview_rgb = None

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
        self.strength_scale = round(
            0.5 * self.strength_scale + 0.5 * float(metrics["recommended_strength_scale"]), 4
        )
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

        eff_motion = intra_motion_mae if intra_motion_mae is not None else metrics.get("motion_mae")
        if int(step) >= 1 and eff_motion is not None:
            if float(eff_motion) < 1.8:
                self.recommended_video_end_percent = 0.33
            elif float(eff_motion) > 18.0:
                self.recommended_video_end_percent = 0.50
        metrics["recommended_video_end_percent"] = self.recommended_video_end_percent

        self.preview_history.append(metrics)
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
                if self.prompt_id is not None:
                    msg.setdefault("prompt_id", self.prompt_id)
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
            "mask_stats": dict(self.mask_stats),
            "step_history": list(self.step_history),
            "audio_x0_history": list(self.audio_x0_history),
            "preview_history": list(self.preview_history),
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

    raw_b = (
        None
        if region_conditioning is None
        else _extract_cond_tensor(region_conditioning, name="region_conditioning")
    )
    tags_b = None if region_conditioning is None else _extract_cond_tags(region_conditioning)

    m = model.clone()
    dm = getattr(getattr(m, "model", None), "diffusion_model", None)
    if not _steer_hooks.is_hookable_dit(dm):
        raise ValueError("MiniMaxH3SelfMaskRouter requires a MiniMax H3 model with hookable DiT blocks")

    from comfy.patcher_extension import CallbacksMP, WrappersMP

    t_opts = m.model_options.setdefault("transformer_options", {})
    dm_wrappers = m.wrappers.get(WrappersMP.DIFFUSION_MODEL, {})
    dit_replaces = t_opts.get("patches_replace", {}).get("dit", {})

    if (
        hasattr(model, "_h3_frozen_video_cache")
        or hasattr(m, "_h3_frozen_video_cache")
        or any("frozen_video_cache" in str(wk) for wk in dm_wrappers)
    ):
        raise ValueError("MiniMaxH3SelfMaskRouter cannot stack with MiniMaxH3FrozenVideoCache")

    if SELF_MASK_WRAPPER_KEY in dm_wrappers or any(
        _has_self_mask_block_patch(bp) or _has_spatial_block_patch(bp) for bp in dit_replaces.values()
    ):
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

        if isinstance(transformer_options, dict) and transformer_options.get("_h3_steer_runtime") is not None:
            raise ValueError(
                "MiniMaxH3SelfMaskRouter cannot share the same sampler step with an active 1x H3SteerRuntime"
            )

        if step_state["fresh_run"]:
            step_state["fresh_run"] = False
            policy.reset_for_new_run()

        payload = kwargs.get("minimax_payload") or {}
        edit_pass = payload.get("h3_edit_pass") if isinstance(payload, Mapping) else None
        can_route_phase2 = (raw_b is not None) and (edit_pass not in ("original", "edited"))

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
            rt = SpatialVisualQRRuntime(
                dm=dm,
                shadow_b_init=ctx_b,
                shadow_b_tags=tags_b,
                x_start=0.0,
                x_end=1.0,
                y_start=0.0,
                y_end=1.0,
                invert=False,  # inversion already baked into `harvested_mask_frames`
                mask=policy.harvested_mask_frames,
                isolate_blocks=iso_set,
                audio_zone=aud_z,
                shield_region=shield_region,
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

    need_post_cfg = bool(enable_audio_x0_thermostat) or (src == "audio_transients")
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
                        "via Q_vid @ K_aud[t]^T (B35+B40); 'audio_transients': Step-0 audio_x0 onset burst mask (flux_z >= 2.2)."
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
