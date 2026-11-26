"""Composable MoGe-2 Metric Depth + WAV Beat `[F, H, W]` MASK producer (`MiniMaxH3DepthBeatMask`).

Turns any still/video (`image`) and optional audio track (`audio`) into a temporal/spatial
`[F, H, W]` `MASK` ready for `MiniMaxH3SpatialQueryRouter` or `MiniMaxH3EditGuider`:
  - `depth_band`: Isolates a relative depth quantile band (`near_quantile..far_quantile`), e.g.
    `0.65..1.0` for the far back wall/doorway of a room while protecting foreground furniture.
  - `depth_wave`: Advances a feathered transformation wave through scene depth across frames.
  - `audio_beat_cut`: Steps cleanly `0 -> 1` at the strongest `17*m` downbeat or `cut_frame`.
  - `audio_beat_pulse`: Emits fast-attack, exponential-decay pulses on WAV onset peaks.
  - `depth_band_on_beat`: Gates the `depth_band` silhouette behind the resolved `cut_frame` so
    `0..cut_frame-1` stays `0.0` (Base World) before tearing open the far depth zone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import io

DEPTH_BEAT_MODES: tuple[str, ...] = (
    "depth_band",
    "depth_wave",
    "audio_beat_cut",
    "audio_beat_pulse",
    "depth_band_on_beat",
)

BEAT_TIMING_MODES: tuple[str, ...] = (
    "auto_from_audio",
    "manual_frame",
)

_REPO_ROOT = Path(__file__).resolve().parent


def _cosine_smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    span = max(float(edge1 - edge0), 1e-4)
    u = ((x - float(edge0)) / span).clamp(0.0, 1.0)
    return 0.5 - 0.5 * torch.cos(math.pi * u)


def analyze_audio_waveform_beats(
    waveform: torch.Tensor | np.ndarray,
    sample_rate: int,
    length: int,
) -> dict[str, Any]:
    """Extract 24-fps onset peak frames and top `17*m` causal cut frame from an audio waveform."""
    n_frames = max(1, int(length))
    fallback_cut = min(max(1, min(85, n_frames - 1)), max(1, n_frames - 1))
    empty = {
        "beats_s": [],
        "beat_frames": [],
        "causal_cut_frames": [],
        "recommended_cut_frame": fallback_cut,
    }
    if sample_rate <= 0 or n_frames < 4:
        return empty

    arr = waveform.detach().cpu().numpy() if torch.is_tensor(waveform) else np.asarray(waveform)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        mono = arr[0].mean(axis=0)
    elif arr.ndim == 2:
        mono = arr.mean(axis=0) if arr.shape[0] <= 8 else arr.mean(axis=1)
    else:
        mono = arr.reshape(-1)
    n_samples = int(mono.shape[0])
    if n_samples < 8:
        return empty

    hp = np.empty_like(mono)
    hp[0] = 0.0
    hp[1:] = mono[1:] - 0.95 * mono[:-1]

    hop = float(sample_rate) / 24.0
    avail_frames = min(n_frames, int(n_samples // hop))
    if avail_frames < 4:
        return empty

    edges = np.floor(np.arange(avail_frames + 1, dtype=np.float64) * hop).astype(np.int64)
    edges[-1] = min(n_samples, int(edges[-1]))
    starts = edges[:-1]
    counts = np.maximum(1, edges[1:] - starts).astype(np.float32)
    trim_end = int(edges[-1])
    if trim_end <= 0 or int(starts[-1]) >= trim_end:
        return empty

    rms_full = np.sqrt(np.add.reduceat(mono[:trim_end] * mono[:trim_end], starts) / counts)
    rms_hp = np.sqrt(np.add.reduceat(hp[:trim_end] * hp[:trim_end], starts) / counts)
    energy = 0.5 * rms_full + 0.5 * rms_hp

    flux = np.zeros(avail_frames, dtype=np.float32)
    flux[1:] = np.maximum(0.0, energy[1:] - energy[:-1])
    max_flux = float(flux.max())
    if max_flux <= 1e-6:
        return empty

    min_sep = max(1, int(round(0.35 * 24.0)))
    noise_floor = max(1e-5, float(np.median(flux)) + 0.15 * max_flux)
    peak_mask = np.zeros(avail_frames, dtype=bool)
    peak_mask[1:-1] = (flux[1:-1] >= flux[:-2]) & (flux[1:-1] > flux[2:]) & (flux[1:-1] >= noise_floor)
    if avail_frames >= 2 and flux[-1] > flux[-2] and flux[-1] >= noise_floor:
        peak_mask[-1] = True
    peak_indices = np.flatnonzero(peak_mask)

    beat_frames: list[int] = []
    if peak_indices.size > 0:
        order = peak_indices[np.argsort(flux[peak_indices])[::-1]]
        for idx in order:
            idx_i = int(idx)
            if all(abs(idx_i - prev) >= min_sep for prev in beat_frames):
                beat_frames.append(idx_i)
                if len(beat_frames) >= 12:
                    break
        beat_frames.sort()

    candidates = [cf for cf in range(17, max(18, avail_frames - 4), 17) if cf < avail_frames]
    scored = []
    for cf in candidates:
        lo = max(0, cf - 2)
        hi = min(avail_frames, cf + 3)
        scored.append((cf, float(flux[lo:hi].max())))
    scored.sort(key=lambda item: (-item[1], abs(item[0] - 85), item[0]))
    causal_cut_frames = [cf for cf, _s in scored[:6]]

    if causal_cut_frames and scored[0][1] > 1e-6:
        rec_cut = causal_cut_frames[0]
    elif beat_frames:
        rec_cut = max(1, min(n_frames - 1, beat_frames[0]))
    else:
        rec_cut = fallback_cut

    return {
        "beats_s": [round(f / 24.0, 3) for f in beat_frames],
        "beat_frames": beat_frames,
        "causal_cut_frames": causal_cut_frames,
        "recommended_cut_frame": int(rec_cut),
    }


def compute_moge2_depth_frame(
    image: torch.Tensor,
    *,
    width: int,
    height: int,
    checkpoint: Path | str | None = None,
    device: str = "auto",
    _model: Any = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run HyDen MoGe-2 on Frame 0 (or `[B, H, W, C]`), returning `(depth_hw, depth_stats)` on CPU."""
    if not torch.is_tensor(image) or image.ndim != 4 or int(image.shape[0]) < 1:
        raise ValueError(f"MiniMaxH3DepthBeatMask expects [B, H, W, C] image tensor, got {type(image).__name__}")

    img0 = image[0:1, ..., :3].detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    nchw = img0.permute(0, 3, 1, 2)
    if int(nchw.shape[2]) != int(height) or int(nchw.shape[3]) != int(width):
        nchw = F.interpolate(nchw, size=(int(height), int(width)), mode="bilinear", align_corners=False)

    if device == "auto":
        run_dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        run_dev = device

    loaded_here = _model is None
    if loaded_here:
        import sys

        repo_str = str(_REPO_ROOT)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        from vendor.meta_moge2.hyden_moge2 import DEFAULT_CHECKPOINT, load_hyden_moge2

        ckpt = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
        model = load_hyden_moge2(ckpt, device=run_dev)
    else:
        model = _model.to(run_dev)

    try:
        with torch.inference_mode():
            out = model.infer(nchw.to(device=run_dev), apply_mask=False)
        depth_hw = out["depth"][0].detach().to(device="cpu", dtype=torch.float32)
        valid_hw = (
            out["mask"][0].detach().to(device="cpu", dtype=torch.bool)
            if "mask" in out and torch.is_tensor(out["mask"])
            else (torch.isfinite(depth_hw) & (depth_hw > 0.0))
        )
    finally:
        if loaded_here:
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
            if run_dev.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    valid_vals = depth_hw[valid_hw & torch.isfinite(depth_hw) & (depth_hw > 0.0)]
    if valid_vals.numel() == 0:
        depth_hw = torch.ones((int(height), int(width)), dtype=torch.float32)
        stats = {"near_m": 1.0, "far_m": 1.0, "median_m": 1.0}
    else:
        median_m = float(valid_vals.median().item())
        fill_depth = torch.where(torch.isfinite(depth_hw) & (depth_hw > 0.0), depth_hw, median_m)
        depth_hw = fill_depth
        stats = {
            "near_m": round(float(torch.quantile(valid_vals, 0.02).item()), 3),
            "far_m": round(float(torch.quantile(valid_vals, 0.98).item()), 3),
            "median_m": round(median_m, 3),
        }
    return depth_hw, stats


def _depth_to_quantile_map(depth_hw: torch.Tensor) -> torch.Tensor:
    """Convert metric depth `[H, W]` into monotonically ordered `[0, 1]` depth quantiles."""
    flat = depth_hw.reshape(-1).to(dtype=torch.float32)
    n = int(flat.numel())
    if n <= 1:
        return torch.zeros_like(depth_hw, dtype=torch.float32)
    order = torch.argsort(flat, stable=True)
    ranks = torch.empty(n, dtype=torch.float32, device=flat.device)
    ranks[order] = torch.linspace(0.0, 1.0, n, dtype=torch.float32, device=flat.device)
    return ranks.view_as(depth_hw)


def compute_depth_beat_mask(
    *,
    mode: str = "depth_band",
    beat_timing: str = "auto_from_audio",
    length: int = 141,
    width: int = 1344,
    height: int = 768,
    near_quantile: float = 0.65,
    far_quantile: float = 1.0,
    feather: float = 0.08,
    cut_frame: int = 85,
    pulse_decay_frames: int = 12,
    invert: bool = False,
    image: torch.Tensor | None = None,
    audio: dict[str, Any] | None = None,
    depth_hw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str]:
    """Build a `[F, H, W]` `MASK` tensor in `[0, 1]` plus JSON summary string."""
    if mode not in DEPTH_BEAT_MODES:
        raise ValueError(f"MiniMaxH3DepthBeatMask: mode must be one of {DEPTH_BEAT_MODES}, got {mode!r}")
    if beat_timing not in BEAT_TIMING_MODES:
        raise ValueError(
            f"MiniMaxH3DepthBeatMask: beat_timing must be one of {BEAT_TIMING_MODES}, got {beat_timing!r}"
        )

    f_len = max(1, int(length))
    w_px = max(16, int(width))
    h_px = max(16, int(height))
    q_lo = float(min(near_quantile, far_quantile))
    q_hi = float(max(near_quantile, far_quantile))
    q_lo = max(0.0, min(1.0, q_lo))
    q_hi = max(0.0, min(1.0, q_hi))
    feath = max(0.01, min(0.5, float(feather)))

    needs_depth = mode in ("depth_band", "depth_wave", "depth_band_on_beat")
    depth_stats: dict[str, Any] = {}
    q_map: torch.Tensor | None = None
    if needs_depth:
        if depth_hw is not None:
            d_tensor = depth_hw.detach().to(device="cpu", dtype=torch.float32)
            if d_tensor.shape != (h_px, w_px):
                d_tensor = F.interpolate(
                    d_tensor.unsqueeze(0).unsqueeze(0), size=(h_px, w_px), mode="bilinear", align_corners=False
                )[0, 0]
            depth_stats = {
                "near_m": round(float(d_tensor.min().item()), 3),
                "far_m": round(float(d_tensor.max().item()), 3),
                "median_m": round(float(d_tensor.median().item()), 3),
            }
        else:
            if image is None:
                raise ValueError(f"MiniMaxH3DepthBeatMask: mode={mode!r} requires `image` to estimate MoGe-2 depth")
            d_tensor, depth_stats = compute_moge2_depth_frame(image, width=w_px, height=h_px)
        q_map = _depth_to_quantile_map(d_tensor)

    beat_info: dict[str, Any] = {
        "beats_s": [],
        "beat_frames": [],
        "causal_cut_frames": [],
        "recommended_cut_frame": min(max(1, int(cut_frame)), max(1, f_len - 1)),
    }
    if audio is not None and isinstance(audio, dict) and "waveform" in audio:
        sr = int(audio.get("sample_rate", 32000))
        beat_info = analyze_audio_waveform_beats(audio["waveform"], sr, f_len)

    if beat_timing == "auto_from_audio" and audio is not None:
        resolved_cut = int(beat_info["recommended_cut_frame"])
        beat_frames = list(beat_info["beat_frames"]) or [resolved_cut]
    else:
        resolved_cut = min(max(1, int(cut_frame)), max(1, f_len - 1)) if f_len > 1 else 0
        beat_frames = [resolved_cut]

    if mode == "depth_band":
        assert q_map is not None
        low_gate = _cosine_smoothstep(q_lo - feath, q_lo, q_map) if q_lo > 0.0 else torch.ones_like(q_map)
        high_gate = (1.0 - _cosine_smoothstep(q_hi, q_hi + feath, q_map)) if q_hi < 1.0 else torch.ones_like(q_map)
        band_hw = (low_gate * high_gate).clamp(0.0, 1.0)
        mask = band_hw.unsqueeze(0).expand(f_len, h_px, w_px).clone()
    elif mode == "depth_wave":
        assert q_map is not None
        t_u = torch.linspace(0.0, 1.0, f_len, dtype=torch.float32)
        thresholds = q_lo + t_u * max(1e-4, (q_hi + feath) - q_lo)
        mask = torch.empty((f_len, h_px, w_px), dtype=torch.float32)
        for fi in range(f_len):
            thr = float(thresholds[fi].item())
            mask[fi] = (1.0 - _cosine_smoothstep(thr - feath, thr + feath, q_map)).clamp(0.0, 1.0)
    elif mode == "audio_beat_cut":
        t_env = torch.zeros(f_len, dtype=torch.float32)
        if f_len == 1 or resolved_cut < f_len:
            t_env[resolved_cut:] = 1.0
        mask = t_env.view(f_len, 1, 1).expand(f_len, h_px, w_px).clone()
    elif mode == "audio_beat_pulse":
        tau = max(float(pulse_decay_frames) / 3.0, 1.0)
        t_env = torch.zeros(f_len, dtype=torch.float32)
        frame_idx = torch.arange(f_len, dtype=torch.float32)
        for bf in beat_frames:
            dt = frame_idx - float(bf)
            pulse = torch.where((dt >= 0.0) & (dt <= float(pulse_decay_frames)), torch.exp(-dt / tau), 0.0)
            pulse = torch.where(pulse < 0.05, torch.zeros_like(pulse), pulse)
            t_env = torch.maximum(t_env, pulse)
        mask = t_env.clamp(0.0, 1.0).view(f_len, 1, 1).expand(f_len, h_px, w_px).clone()
    else:  # depth_band_on_beat
        assert q_map is not None
        low_gate = _cosine_smoothstep(q_lo - feath, q_lo, q_map) if q_lo > 0.0 else torch.ones_like(q_map)
        high_gate = (1.0 - _cosine_smoothstep(q_hi, q_hi + feath, q_map)) if q_hi < 1.0 else torch.ones_like(q_map)
        band_hw = (low_gate * high_gate).clamp(0.0, 1.0)
        t_env = torch.zeros(f_len, dtype=torch.float32)
        if resolved_cut < f_len:
            ramp_frames = max(1, min(4, f_len - resolved_cut))
            t_env[resolved_cut : resolved_cut + ramp_frames] = torch.linspace(
                1.0 / ramp_frames, 1.0, ramp_frames, dtype=torch.float32
            )
            if resolved_cut + ramp_frames < f_len:
                t_env[resolved_cut + ramp_frames :] = 1.0
        mask = (t_env.view(f_len, 1, 1) * band_hw.unsqueeze(0)).clamp(0.0, 1.0)

    if invert:
        mask = (1.0 - mask).clamp(0.0, 1.0)

    summary = {
        "mode": mode,
        "beat_timing": beat_timing,
        "shape": list(mask.shape),
        "coverage_mean": round(float(mask.mean().item()), 4),
        "resolved_cut_frame": int(resolved_cut),
        "beats_s": beat_info["beats_s"],
        "causal_cut_frames": beat_info["causal_cut_frames"],
        "depth_stats": depth_stats,
    }
    return mask.contiguous(), json.dumps(summary)


class MiniMaxH3DepthBeatMask(io.ComfyNode):
    """Composable MoGe-2 depth + WAV onset beat `[F, H, W]` `MASK` producer."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3DepthBeatMask",
            display_name="MiniMax H3 Depth / Beat Mask",
            category="conditioning/minimax_h3/masks",
            description=(
                "Builds a [F, H, W] MASK from MoGe-2 relative 3D depth quantiles (`depth_band`, "
                "`depth_wave`, `depth_band_on_beat`) and/or WAV onset beat detection "
                "(`audio_beat_cut`, `audio_beat_pulse`) for SpatialQueryRouter or EditGuider."
            ),
            inputs=[
                io.Combo.Input("mode", options=list(DEPTH_BEAT_MODES), default="depth_band"),
                io.Combo.Input("beat_timing", options=list(BEAT_TIMING_MODES), default="auto_from_audio"),
                io.Int.Input("length", default=141, min=1, max=362, step=1),
                io.Int.Input("width", default=1344, min=64, max=4096, step=16),
                io.Int.Input("height", default=768, min=64, max=4096, step=16),
                io.Float.Input("near_quantile", default=0.65, min=0.0, max=1.0, step=0.01),
                io.Float.Input("far_quantile", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("feather", default=0.08, min=0.01, max=0.5, step=0.01),
                io.Int.Input("cut_frame", default=85, min=1, max=360, step=1),
                io.Int.Input("pulse_decay_frames", default=12, min=1, max=96, step=1),
                io.Boolean.Input("invert", default=False),
                io.Image.Input("image", optional=True),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[
                io.Mask.Output(display_name="mask"),
                io.String.Output(display_name="summary_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        mode: str = "depth_band",
        beat_timing: str = "auto_from_audio",
        length: int = 141,
        width: int = 1344,
        height: int = 768,
        near_quantile: float = 0.65,
        far_quantile: float = 1.0,
        feather: float = 0.08,
        cut_frame: int = 85,
        pulse_decay_frames: int = 12,
        invert: bool = False,
        image: torch.Tensor | None = None,
        audio: dict[str, Any] | None = None,
    ) -> io.NodeOutput:
        mask, summary_json = compute_depth_beat_mask(
            mode=mode,
            beat_timing=beat_timing,
            length=length,
            width=width,
            height=height,
            near_quantile=near_quantile,
            far_quantile=far_quantile,
            feather=feather,
            cut_frame=cut_frame,
            pulse_decay_frames=pulse_decay_frames,
            invert=invert,
            image=image,
            audio=audio,
        )
        return io.NodeOutput(mask, summary_json)
