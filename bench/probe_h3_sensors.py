#!/usr/bin/env python3
"""Read-only 4-Sensor Calibration Probe for MiniMax H3 (`MiniMaxH3Model` & `MiniMaxH3ImageModel`).

Measures four internal H3 sensors inside a single forward pass with **zero output mutation**
(`max_abs == 0.0`) and strict `comfy_aimdo` `malloc_graph` safety (`pause_malloc_graph_ctx()`,
peek-only attention containers, detached CPU `.clone()` snapshots):

1. **Sensor 1 — `Q_vid @ K_txt[noun]^T` spatial/4D localization heatmaps** (`blocks 10..45`,
   Steps `0..2`): tests whether H3's own attention localizes named nouns (`"mirror"`, `"clock"`,
   `"jacket"`) cleanly enough to drive zero-coordinate Self-Masked Routing (`B1`).
2. **Sensor 2 — `Q_vid @ K_aud[t]^T` audio-source / speaker spotlight maps** (`blocks 25..40`,
   Steps `0..2`): maps each video latent step (`[1, 4, 4, 4, 4]`) to its aligned `40 Hz` stereo
   (`L` / `R`) audio window via `FRAME_RESCALE = 5/3` to measure spatial speaker grounding and
   stereo `L -> R` visual separation.
3. **Sensor 3 — Step-`0/1` pure-latent `audio_x0` transient detector** (`sampler_post_cfg_function`,
   `< 1 ms`, zero `taeh3`/VAE decode): measures per-`25 ms` stereo RMS energy, onset flux peaks
   (timestamps in seconds + `24 fps` frame indices), and `L/R` correlation/balance directly from
   predicted `audio_x0`.
4. **Sensor 4 — Block-by-block text-state & `J-Lens` trace** (`blocks 0..48`): tracks evolving
   `h_txt_a` (and `h_shadow_b` when stacked with `MiniMaxH3SpatialQueryRouter`) across DiT blocks,
   measuring row norms, drift from Block `0`, cross-world cosine convergence, and optional CPU
   `steering/xray.py::decode_vector` word readouts (`< 15 ms` on CPU).

Media/PNG exports strictly refuse repo `data/` paths (`AGENTS.md` §2 — write visual outputs under
`$H3_OUTPUT_DIR`). Composes cleanly with `MiniMaxH3SpatialQueryRouter`, `MiniMaxH3QueryRouter`,
`MiniMaxH3Steer`, and `MiniMaxH3Sol` without editing any node files. **Attach order:** always attach
`attach_h3_sensor_probe` **last** (after `MiniMaxH3SpatialQueryRouter` / `MiniMaxH3QueryRouter`) so
the probe's attention override peeks `Q/K` ahead of `SpatialQR`'s `iso`-block fast path and
`_SensorBlockPatch` reads post-`finish_block` `h_shadow_b` across all 50 blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

try:
    from _lib import REPO, bootstrap
except ImportError:
    from bench._lib import REPO, bootstrap

FRAME_PER_TOKEN: tuple[int, ...] = (1, 4, 4, 4, 4)
FRAME_RESCALE: float = 5.0 / 3.0
AUDIO_HZ: float = 40.0
VIDEO_FPS: float = 24.0
SENSOR_WRAPPER_KEY = "minimax_h3_sensor_probe"


# ------------------------------------------------------------------ config & helpers


@dataclass(frozen=True)
class SensorProbeConfig:
    """Configuration for read-only 4-sensor H3 calibration."""

    noun_spans: Mapping[str, Sequence[int]] = field(default_factory=dict)
    expected_boxes: Mapping[str, tuple[float, float, float, float]] = field(default_factory=dict)
    probe_steps: frozenset[int] = frozenset({0, 1, 2})
    text_probe_blocks: frozenset[int] = frozenset({10, 14, 18, 20, 24, 28, 32, 36, 40, 45})
    audio_probe_blocks: frozenset[int] = frozenset({25, 28, 30, 34, 38, 40})
    jlens_blocks: frozenset[int] = frozenset({0, 10, 18, 24, 30, 35, 42, 48})
    capture_audio_x0: bool = True
    audio_transient_z: float = 1.75
    audio_min_gap_steps: int = 4  # 100 ms at 40 Hz
    run_jlens: bool = False
    jlens_family: str = "fl2va"
    jlens_top_k: int = 6
    jlens_bundle: Mapping[str, Any] | None = None
    store_maps: bool = True


def _import_pause_malloc_graph_ctx():
    try:
        from steering.hooks import pause_malloc_graph_ctx

        return pause_malloc_graph_ctx
    except Exception:
        import contextlib

        return contextlib.nullcontext


def _guard_not_in_repo_data(path: Path) -> None:
    """Refuse writing media/PNG outputs inside tracked repo `data/` (`AGENTS.md` §2)."""
    resolved = path.resolve()
    data_dir = (REPO / "data").resolve()
    try:
        resolved.relative_to(data_dir)
    except ValueError:
        return
    raise ValueError(
        f"Refusing to write sensor media inside repo data/ ({path}); "
        "write PNG/video artifacts under $H3_OUTPUT_DIR instead."
    )


def find_noun_token_spans(
    token_pieces: Sequence[str],
    nouns: Sequence[str],
    *,
    case_sensitive: bool = False,
) -> dict[str, list[int]]:
    """Map target noun/phrase strings onto relative indices into `token_pieces`.

    Works on decoded Qwen3-VL BPE token strings (handling `Ġ`/`▁` word boundaries and
    multi-subword nouns) as well as plain word lists.
    """
    if not token_pieces:
        return {str(n): [] for n in nouns}

    # Build normalized character stream with per-character token-index attribution.
    norm_chars: list[str] = []
    char_to_tok: list[int] = []
    for idx, raw in enumerate(token_pieces):
        piece = str(raw).replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")
        if not case_sensitive:
            piece = piece.lower()
        for ch in piece:
            norm_chars.append(ch)
            char_to_tok.append(idx)
    joined = "".join(norm_chars)

    out: dict[str, list[int]] = {}
    for noun in nouns:
        q = str(noun).strip()
        if not q:
            continue
        q_norm = q if case_sensitive else q.lower()
        spans_idx: set[int] = set()
        pattern = re.compile(r"(?<![a-z0-9_])" + re.escape(q_norm) + r"(?![a-z0-9_])")
        matches = list(pattern.finditer(joined))
        if not matches:
            # Fallback to substring match when punctuation/compound words wrap the stem.
            matches = list(re.finditer(re.escape(q_norm), joined))
        for m in matches:
            for c_i in range(m.start(), m.end()):
                spans_idx.add(char_to_tok[c_i])
        out[str(noun)] = sorted(spans_idx)
    return out


def video_latent_audio_windows(latent_t: int, audio_t: int) -> list[tuple[int, int]]:
    """Map each `[1, 4, 4, 4, 4]` video latent step `k` to its `[a0, a1)` slice on `40 Hz` audio."""
    if latent_t <= 0 or audio_t <= 0:
        return []
    windows: list[tuple[int, int]] = []
    f_cur = 0
    for k in range(latent_t):
        f_span = int(FRAME_PER_TOKEN[k % 5])
        f_end = f_cur + f_span
        a0 = max(0, min(audio_t - 1, int(math.floor(f_cur * FRAME_RESCALE))))
        a1 = max(a0 + 1, min(audio_t, int(math.ceil(f_end * FRAME_RESCALE))))
        windows.append((a0, a1))
        f_cur = f_end
    return windows


def _spatial_metrics(
    score_thw: torch.Tensor,
    *,
    expected_box: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Compute localization sharpness, entropy, centroid, peak, and optional box concentration."""
    t_dim, h_dim, w_dim = (int(d) for d in score_thw.shape)
    n_hw = max(1, h_dim * w_dim)
    flat = score_thw.float().reshape(t_dim, n_hw)
    probs = F.softmax(flat, dim=-1)

    log_n = math.log(float(n_hw)) if n_hw > 1 else 1.0
    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    ent_norm = float((ent / log_n).mean().item()) if n_hw > 1 else 0.0

    k10 = max(1, int(math.ceil(0.10 * n_hw)))
    top10_vals = torch.topk(probs, k=k10, dim=-1).values
    top10_mass = float(top10_vals.sum(dim=-1).mean().item())

    mean_v = flat.mean(dim=-1, keepdim=True)
    std_v = flat.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
    z_max = float(((flat.max(dim=-1, keepdim=True).values - mean_v) / std_v).mean().item())

    # Mean temporal map for aggregate spatial peak & centroid.
    mean_hw = probs.mean(dim=0).reshape(h_dim, w_dim)
    ys = (torch.arange(h_dim, dtype=torch.float32) + 0.5) / float(h_dim)
    xs = (torch.arange(w_dim, dtype=torch.float32) + 0.5) / float(w_dim)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    p_sum = float(mean_hw.sum().item())
    cx = float((mean_hw * xx).sum().item() / max(p_sum, 1e-12))
    cy = float((mean_hw * yy).sum().item() / max(p_sum, 1e-12))

    peak_idx = int(mean_hw.reshape(-1).argmax().item())
    py_i, px_i = divmod(peak_idx, w_dim)
    peak_x = (px_i + 0.5) / float(w_dim)
    peak_y = (py_i + 0.5) / float(h_dim)

    res: dict[str, Any] = {
        "shape_thw": [t_dim, h_dim, w_dim],
        "spatial_entropy_norm": round(ent_norm, 6),
        "top10_mass": round(top10_mass, 6),
        "peak_zscore": round(z_max, 4),
        "peak_xy": [round(peak_x, 4), round(peak_y, 4)],
        "centroid_xy": [round(cx, 4), round(cy, 4)],
    }

    if expected_box is not None:
        x0, y0, x1, y1 = (float(v) for v in expected_box)
        in_box = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        box_mass = float(mean_hw[in_box].sum().item() / max(p_sum, 1e-12)) if bool(in_box.any()) else 0.0
        box_area = max(1e-6, float(in_box.float().mean().item()))
        res["box_mass"] = round(box_mass, 6)
        res["box_enrichment"] = round(box_mass / box_area, 4)
        res["peak_in_box"] = bool(x0 <= peak_x <= x1 and y0 <= peak_y <= y1)
    return res


def analyze_audio_x0_latent(
    audio_x0: torch.Tensor,
    *,
    z_thresh: float = 1.75,
    min_gap_steps: int = 4,
) -> dict[str, Any]:
    """Extract `< 1 ms` pure-latent stereo energy, onset flux transients, and `L/R` metrics."""
    a = audio_x0.detach().to(device="cpu", dtype=torch.float32)
    if a.ndim == 4:
        a = a[0]  # [C, 2, T_aud]
    if a.ndim != 3 or int(a.shape[1]) < 1 or int(a.shape[2]) < 1:
        raise ValueError(f"Expected audio_x0 shape [B, C, 2, T] or [C, 2, T], got {tuple(audio_x0.shape)}")
    c_dim, n_ch, t_aud = int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
    denom = math.sqrt(float(max(1, c_dim)))

    e_l = torch.linalg.vector_norm(a[:, 0, :], dim=0) / denom
    e_r = (torch.linalg.vector_norm(a[:, 1, :], dim=0) / denom) if n_ch >= 2 else e_l.clone()
    e_st = torch.sqrt(0.5 * (e_l * e_l + e_r * e_r))

    flux = torch.zeros_like(e_st)
    if t_aud > 1:
        flux[1:] = (e_st[1:] - e_st[:-1]).clamp_min(0.0)
    f_mean = float(flux.mean().item())
    f_std = float(flux.std(unbiased=False).item())
    thresh = f_mean + float(z_thresh) * max(f_std, 1e-6)

    transients: list[dict[str, Any]] = []
    last_idx = -int(min_gap_steps) - 1
    for i in range(1, t_aud):
        val = float(flux[i].item())
        if val < thresh:
            continue
        is_local_max = (val >= float(flux[i - 1].item())) and (
            i == t_aud - 1 or val >= float(flux[i + 1].item())
        )
        if not is_local_max:
            continue
        if (i - last_idx) < int(min_gap_steps):
            if transients and val > float(transients[-1]["flux"]):
                transients[-1] = {
                    "a_idx": i,
                    "t_sec": round(i / AUDIO_HZ, 4),
                    "video_frame_24fps": int(round(i * VIDEO_FPS / AUDIO_HZ)),
                    "energy": round(float(e_st[i].item()), 5),
                    "flux": round(val, 5),
                    "flux_z": round((val - f_mean) / max(f_std, 1e-6), 3),
                }
                last_idx = i
            continue
        transients.append(
            {
                "a_idx": i,
                "t_sec": round(i / AUDIO_HZ, 4),
                "video_frame_24fps": int(round(i * VIDEO_FPS / AUDIO_HZ)),
                "energy": round(float(e_st[i].item()), 5),
                "flux": round(val, 5),
                "flux_z": round((val - f_mean) / max(f_std, 1e-6), 3),
            }
        )
        last_idx = i

    lr_corr = 1.0
    if n_ch >= 2 and t_aud >= 2:
        al_c = a[:, 0, :].reshape(-1) - a[:, 0, :].mean()
        ar_c = a[:, 1, :].reshape(-1) - a[:, 1, :].mean()
        den = float(torch.linalg.vector_norm(al_c).item() * torch.linalg.vector_norm(ar_c).item())
        lr_corr = float((al_c * ar_c).sum().item() / den) if den > 1e-12 else 0.0

    rms_l = float(torch.sqrt((e_l * e_l).mean()).item())
    rms_r = float(torch.sqrt((e_r * e_r).mean()).item())
    lr_db = 20.0 * math.log10(max(rms_l, 1e-8) / max(rms_r, 1e-8))

    return {
        "audio_t": t_aud,
        "rms_stereo": round(float(torch.sqrt((e_st * e_st).mean()).item()), 6),
        "rms_left": round(rms_l, 6),
        "rms_right": round(rms_r, 6),
        "lr_db_diff": round(lr_db, 4),
        "lr_latent_corr": round(lr_corr, 6),
        "n_transients": len(transients),
        "transients": transients,
        "energy_curve": [round(float(v), 5) for v in e_st.tolist()],
    }


# ------------------------------------------------------------------ runtime recorder


class H3SensorRecorder:
    """Collects strictly detached CPU read-only sensor telemetry across sampling steps."""

    def __init__(self, config: SensorProbeConfig) -> None:
        self.config = config
        self.step_count = 0
        self._last_sigma: float | None = None
        self._cur_step_idx = 0
        self._cur_layout: Any = None
        self._cur_tx_a = 0
        self._cur_tx_b = 0
        self._cur_va = 0
        self._cur_vb = 0
        self._cur_aa: int | None = None
        self._cur_ab: int | None = None
        self._cur_latent_t = 1
        self._cur_patch_h = 1
        self._cur_patch_w = 1
        self._cur_audio_t = 0
        self._step_blk0_txt_a: torch.Tensor | None = None
        self._step_blk0_txt_b: torch.Tensor | None = None
        self._jlens_bundle_cache: Mapping[str, Any] | None = config.jlens_bundle

        # Recorded outputs (all plain JSON-serializable dicts + optional detached CPU tensors).
        self.text_attention_records: list[dict[str, Any]] = []
        self.audio_attention_records: list[dict[str, Any]] = []
        self.audio_x0_records: list[dict[str, Any]] = []
        self.jlens_records: list[dict[str, Any]] = []
        self.heatmaps_text: dict[tuple[int, int, str], torch.Tensor] = {}
        self.heatmaps_audio: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

    def begin_step(self, transformer_options: dict[str, Any] | None, timestep: Any) -> int:
        sigma = self._resolve_sigma(transformer_options, timestep)
        if self._last_sigma is not None and sigma is not None and sigma >= self._last_sigma - 1e-9:
            # New sampling run started (sigma did not decrease).
            self.step_count = 0
        self._cur_step_idx = self.step_count
        self.step_count += 1
        if sigma is not None:
            self._last_sigma = sigma
        self._step_blk0_txt_a = None
        self._step_blk0_txt_b = None
        return self._cur_step_idx

    @staticmethod
    def _resolve_sigma(transformer_options: dict[str, Any] | None, timestep: Any) -> float | None:
        t_opts = transformer_options or {}
        cur = t_opts.get("sigmas")
        if cur is None:
            cur = timestep
        if cur is None:
            return None
        if torch.is_tensor(cur):
            val = float(cur.detach().flatten()[0].item())
        elif isinstance(cur, (list, tuple)) and cur:
            val = float(cur[0])
        else:
            val = float(cur)
        if val > 1.0001:
            val /= 1000.0
        return val

    def register_layout(self, layout: Any) -> None:
        self._cur_layout = layout
        segs = getattr(layout, "segments", ())
        tx_spans = [(int(a), int(b)) for (a, b, k) in segs if k == "text" and b > a]
        vid_spans = [(int(a), int(b)) for (a, b, k) in segs if k == "video" and b > a]
        aud_spans = [(int(a), int(b)) for (a, b, k) in segs if k == "audio" and b > a]
        if tx_spans:
            self._cur_tx_a, self._cur_tx_b = tx_spans[0]
        if vid_spans:
            self._cur_va, self._cur_vb = vid_spans[0]
        if aud_spans:
            self._cur_aa, self._cur_ab = aud_spans[0]
        else:
            self._cur_aa, self._cur_ab = None, None

        sig = getattr(layout, "signature", None)
        if sig is not None and len(sig) >= 5:
            _t_len, lt, lh, lw, at = (int(v) for v in sig[:5])
            self._cur_latent_t = max(1, lt)
            self._cur_patch_h = max(1, lh // 2)
            self._cur_patch_w = max(1, lw // 2)
            self._cur_audio_t = max(0, at)

    def probe_attention(
        self,
        block_index: int,
        q_p: torch.Tensor,
        k_p: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> None:
        if self._cur_step_idx not in self.config.probe_steps:
            return
        want_txt = bool(self.config.noun_spans) and (block_index in self.config.text_probe_blocks)
        want_aud = (
            (self._cur_aa is not None)
            and (self._cur_audio_t > 0)
            and (block_index in self.config.audio_probe_blocks)
        )
        if not want_txt and not want_aud:
            return

        va, vb = self._cur_va, self._cur_vb
        expected_v = self._cur_latent_t * self._cur_patch_h * self._cur_patch_w
        if vb <= va or (vb - va) != expected_v or int(q_p.shape[2]) < vb or int(k_p.shape[2]) < vb:
            return

        head_dim = int(q_p.shape[-1])
        scale_eff = float(scale) if scale is not None else (1.0 / math.sqrt(float(max(1, head_dim))))
        q_vid = q_p[0, :, va:vb, :].detach().float()  # [H, S_vid, D]

        if want_txt:
            self._probe_text_nouns(block_index, q_vid, k_p, scale_eff)
        if want_aud:
            self._probe_audio_sources(block_index, q_vid, k_p, scale_eff)

    def _probe_text_nouns(
        self,
        block_index: int,
        q_vid: torch.Tensor,
        k_p: torch.Tensor,
        scale_eff: float,
    ) -> None:
        tx_a, tx_b = self._cur_tx_a, self._cur_tx_b
        l_txt = tx_b - tx_a
        if l_txt <= 0:
            return

        k_sink = k_p[0, :, tx_a : tx_a + 1, :].detach().float()  # [H, 1, D]
        sink_logits = torch.einsum("hsd,hkd->hs", q_vid, k_sink) * scale_eff
        sink_mean_logit = float(sink_logits.mean().item())

        lt, ph, pw = self._cur_latent_t, self._cur_patch_h, self._cur_patch_w
        for noun, rel_idxs in self.config.noun_spans.items():
            valid = [tx_a + int(i) for i in rel_idxs if 0 <= int(i) < l_txt]
            if not valid:
                continue
            idx_t = torch.tensor(valid, dtype=torch.long, device=k_p.device)
            k_span = k_p[0, :, :, :].index_select(1, idx_t).detach().float()  # [H, K, D]
            logits = torch.einsum("hsd,hkd->hsk", q_vid, k_span) * scale_eff  # [H, S_vid, K]
            score_mean = logits.mean(dim=(0, 2)).reshape(lt, ph, pw).cpu().clone()
            score_headmax = logits.mean(dim=2).amax(dim=0).reshape(lt, ph, pw).cpu().clone()

            flat_s = score_mean.reshape(-1)
            k10 = max(1, int(math.ceil(0.10 * flat_s.numel())))
            top10_logit = float(torch.topk(flat_s, k=k10).values.mean().item())

            metrics = _spatial_metrics(
                score_mean,
                expected_box=self.config.expected_boxes.get(noun),
            )
            metrics_hmax = _spatial_metrics(
                score_headmax,
                expected_box=self.config.expected_boxes.get(noun),
            )
            self.text_attention_records.append(
                {
                    "step": self._cur_step_idx,
                    "block": block_index,
                    "noun": noun,
                    "n_tokens": len(valid),
                    "sink_mean_logit": round(sink_mean_logit, 4),
                    "top10_sink_margin": round(top10_logit - sink_mean_logit, 4),
                    "mean_heads": metrics,
                    "max_head_top10_mass": metrics_hmax["top10_mass"],
                    "max_head_peak_xy": metrics_hmax["peak_xy"],
                }
            )
            if self.config.store_maps:
                self.heatmaps_text[(self._cur_step_idx, block_index, noun)] = score_mean

    def _probe_audio_sources(
        self,
        block_index: int,
        q_vid: torch.Tensor,
        k_p: torch.Tensor,
        scale_eff: float,
    ) -> None:
        assert self._cur_aa is not None and self._cur_ab is not None
        aa, ab = self._cur_aa, self._cur_ab
        audio_t = self._cur_audio_t
        if (ab - aa) < audio_t * 2:
            return
        lt, ph, pw = self._cur_latent_t, self._cur_patch_h, self._cur_patch_w
        n_hw = ph * pw
        windows = video_latent_audio_windows(lt, audio_t)

        map_l = torch.empty((lt, ph, pw), dtype=torch.float32)
        map_r = torch.empty((lt, ph, pw), dtype=torch.float32)
        k_full = k_p[0].detach().float()  # [H, S, D]

        for k_t, (a0, a1) in enumerate(windows):
            qf = q_vid[:, k_t * n_hw : (k_t + 1) * n_hw, :]  # [H, n_hw, D]
            kal = k_full[:, aa + a0 : aa + a1, :]
            kar = k_full[:, aa + audio_t + a0 : aa + audio_t + a1, :]
            sl = (torch.einsum("hsd,had->hsa", qf, kal) * scale_eff).mean(dim=(0, 2))
            sr = (torch.einsum("hsd,had->hsa", qf, kar) * scale_eff).mean(dim=(0, 2))
            map_l[k_t] = sl.reshape(ph, pw).cpu()
            map_r[k_t] = sr.reshape(ph, pw).cpu()

        map_st = 0.5 * (map_l + map_r)
        m_st = _spatial_metrics(map_st)
        m_l = _spatial_metrics(map_l)
        m_r = _spatial_metrics(map_r)
        self.audio_attention_records.append(
            {
                "step": self._cur_step_idx,
                "block": block_index,
                "stereo": m_st,
                "left_ear_centroid_xy": m_l["centroid_xy"],
                "right_ear_centroid_xy": m_r["centroid_xy"],
                "stereo_lr_x_sep": round(float(m_r["centroid_xy"][0] - m_l["centroid_xy"][0]), 4),
            }
        )
        if self.config.store_maps:
            self.heatmaps_audio[(self._cur_step_idx, block_index)] = {
                "stereo": map_st.clone(),
                "left": map_l.clone(),
                "right": map_r.clone(),
            }

    def record_text_block(
        self,
        block_index: int,
        h_full: torch.Tensor,
        shadow_b: torch.Tensor | None = None,
    ) -> None:
        if self._cur_step_idx not in self.config.probe_steps:
            return
        tx_a, tx_b = self._cur_tx_a, self._cur_tx_b
        if tx_b <= tx_a or int(h_full.shape[0]) < tx_b:
            return
        need_blk0 = block_index == 0
        need_log = block_index in self.config.jlens_blocks
        if not need_blk0 and not need_log:
            return

        h_a = h_full[tx_a:tx_b].detach().to(device="cpu", dtype=torch.float32).clone()
        h_b = (
            shadow_b.detach().to(device="cpu", dtype=torch.float32).clone()
            if torch.is_tensor(shadow_b)
            else None
        )
        if need_blk0:
            self._step_blk0_txt_a = h_a
            self._step_blk0_txt_b = h_b
        if not need_log:
            return

        mean_a = h_a.mean(dim=0)
        row_norm_a = float(torch.linalg.vector_norm(h_a, dim=-1).mean().item())
        rec: dict[str, Any] = {
            "step": self._cur_step_idx,
            "block": block_index,
            "l_txt_a": int(h_a.shape[0]),
            "mean_row_norm_a": round(row_norm_a, 4),
        }

        delta_a: torch.Tensor | None = None
        if self._step_blk0_txt_a is not None and self._step_blk0_txt_a.shape == h_a.shape:
            diff_a = h_a - self._step_blk0_txt_a
            delta_a = diff_a.mean(dim=0)
            rec["delta_vs_blk0_norm_a"] = round(
                float(torch.linalg.vector_norm(diff_a, dim=-1).mean().item()), 4
            )
            rec["cos_vs_blk0_a"] = round(
                float(F.cosine_similarity(mean_a.unsqueeze(0), self._step_blk0_txt_a.mean(dim=0).unsqueeze(0)).item()),
                6,
            )

        diff_ab: torch.Tensor | None = None
        if h_b is not None:
            mean_b = h_b.mean(dim=0)
            rec["l_txt_b"] = int(h_b.shape[0])
            rec["mean_row_norm_b"] = round(float(torch.linalg.vector_norm(h_b, dim=-1).mean().item()), 4)
            rec["cross_world_mean_cos"] = round(
                float(F.cosine_similarity(mean_a.unsqueeze(0), mean_b.unsqueeze(0)).item()), 6
            )
            diff_ab = mean_b - mean_a
            rec["cross_world_mean_diff_norm"] = round(float(torch.linalg.vector_norm(diff_ab).item()), 4)

        if self.config.run_jlens:
            bundle = self._get_jlens_bundle()
            if bundle is not None:
                from steering.xray import decode_vector

                if delta_a is not None and float(torch.linalg.vector_norm(delta_a).item()) > 1e-6:
                    dec_a = decode_vector(
                        delta_a,
                        space="post_refiner",
                        family=self.config.jlens_family,
                        top_k=self.config.jlens_top_k,
                        bundle=bundle,
                    )
                    rec["jlens_delta_a_pos"] = [w["word"] for w in dec_a.get("positive_words", [])]
                    rec["jlens_delta_a_neg"] = [w["word"] for w in dec_a.get("negative_words", [])]
                if diff_ab is not None and float(torch.linalg.vector_norm(diff_ab).item()) > 1e-6:
                    dec_ab = decode_vector(
                        diff_ab,
                        space="post_refiner",
                        family=self.config.jlens_family,
                        top_k=self.config.jlens_top_k,
                        bundle=bundle,
                    )
                    rec["jlens_b_minus_a_pos"] = [w["word"] for w in dec_ab.get("positive_words", [])]
                    rec["jlens_b_minus_a_neg"] = [w["word"] for w in dec_ab.get("negative_words", [])]

        self.jlens_records.append(rec)

    def _get_jlens_bundle(self) -> Mapping[str, Any] | None:
        if self._jlens_bundle_cache is not None:
            return self._jlens_bundle_cache
        try:
            from steering.xray import bundle_available, load_j_lens_bundle

            if bundle_available():
                self._jlens_bundle_cache = load_j_lens_bundle()
                return self._jlens_bundle_cache
        except Exception:
            return None
        return None

    def record_post_cfg(self, args: Mapping[str, Any]) -> None:
        if not self.config.capture_audio_x0:
            return
        step_idx = max(0, self.step_count - 1)
        if step_idx not in self.config.probe_steps:
            return
        denoised = args.get("denoised")
        model = args.get("model")
        if denoised is None:
            return
        shapes = getattr(model, "latent_shapes", None)
        if shapes is None or len(shapes) < 2:
            # Single-stream model (e.g. MiniMax-H3-Image): clean no-op for audio_x0.
            return
        import comfy.utils

        streams = comfy.utils.unpack_latents(denoised, shapes)
        if len(streams) < 2:
            return
        sigma_raw = args.get("sigma")
        sigma_val = (
            float(torch.as_tensor(sigma_raw).detach().flatten()[0].item())
            if sigma_raw is not None
            else None
        )
        stats = analyze_audio_x0_latent(
            streams[1],
            z_thresh=self.config.audio_transient_z,
            min_gap_steps=self.config.audio_min_gap_steps,
        )
        stats["step"] = step_idx
        if sigma_val is not None:
            stats["sigma"] = round(sigma_val, 6)
        self.audio_x0_records.append(stats)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of all four sensors."""
        return {
            "steps_observed": self.step_count,
            "text_attention": list(self.text_attention_records),
            "audio_attention": list(self.audio_attention_records),
            "audio_x0_transients": list(self.audio_x0_records),
            "block_text_jlens": list(self.jlens_records),
        }

    def save_report(self, path: str | Path) -> Path:
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(self.summary(), indent=2) + "\n", encoding="utf-8")
        return out_p


# ------------------------------------------------------------------ ComfyUI model attachment


class _SensorBlockPatch:
    """Read-only `patches_replace['dit'][('double_block', i)]` wrapper composing cleanly with peers."""

    def __init__(self, block_index: int, recorder: H3SensorRecorder, previous: Any = None) -> None:
        self.block_index = int(block_index)
        self.recorder = recorder
        self.previous = previous

    def __call__(self, args: dict[str, Any], extra_args: dict[str, Any]) -> dict[str, Any]:
        pause_ctx = _import_pause_malloc_graph_ctx()
        if self.block_index == 0:
            with pause_ctx(), torch.no_grad():
                self.recorder.register_layout(args["layout"])
        orig_blk = extra_args["original_block"]
        out = orig_blk(args) if self.previous is None else self.previous(args, extra_args)
        to = args.get("transformer_options")
        rt_sqr = to.get("_h3_spatial_qr_rt") if isinstance(to, dict) else None
        shadow_b = getattr(rt_sqr, "h_shadow_b", None) if rt_sqr is not None else None
        with pause_ctx(), torch.no_grad():
            self.recorder.record_text_block(self.block_index, out["img"], shadow_b=shadow_b)
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


def _make_sensor_attn_override(recorder: H3SensorRecorder, previous: Any = None):
    pause_ctx = _import_pause_malloc_graph_ctx()

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
        b_idx = options.get("block_index") if isinstance(options, dict) else None

        # Peek outside `pause_malloc_graph_ctx()` so container refs stay intact, never `.take()`.
        if skip_reshape and b_idx is not None:
            q_p = q if torch.is_tensor(q) else q.peek()
            k_p = k if torch.is_tensor(k) else k.peek()
            if q_p.ndim == 4 and k_p.ndim == 4:
                with pause_ctx(), torch.no_grad():
                    recorder.probe_attention(int(b_idx), q_p, k_p, scale=kwargs.get("scale", None))
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


def attach_h3_sensor_probe(
    model: Any,
    config: SensorProbeConfig | None = None,
) -> tuple[Any, H3SensorRecorder]:
    """Clone `model`, attach strictly read-only 4-sensor hooks, and return `(cloned_model, recorder)`."""
    cfg = config or SensorProbeConfig()
    recorder = H3SensorRecorder(cfg)
    m = model.clone()
    dm = getattr(getattr(m, "model", None), "diffusion_model", None)
    blocks = getattr(dm, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise ValueError("attach_h3_sensor_probe requires a hookable MiniMax H3 model with `.blocks`")

    from comfy.patcher_extension import WrappersMP

    t_opts = m.model_options.setdefault("transformer_options", {})
    dit_replaces = t_opts.get("patches_replace", {}).get("dit", {})
    for b_idx in range(len(blocks)):
        existing = dit_replaces.get(("double_block", b_idx))
        m.set_model_patch_replace(_SensorBlockPatch(b_idx, recorder, existing), "dit", "double_block", b_idx)

    pause_ctx = _import_pause_malloc_graph_ctx()

    def _wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        with pause_ctx(), torch.no_grad():
            recorder.begin_step(transformer_options, timestep)
            to_next = dict(transformer_options or {})
            to_next["optimized_attention_override"] = _make_sensor_attn_override(
                recorder, to_next.get("optimized_attention_override")
            )
        return executor(x, timestep, context, to_next, **kwargs)

    wrappers = m.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
    wrappers[SENSOR_WRAPPER_KEY] = [_wrapper]

    if cfg.capture_audio_x0 and hasattr(m, "set_model_sampler_post_cfg_function"):
        def _post_cfg(args):
            with pause_ctx(), torch.no_grad():
                recorder.record_post_cfg(args)
            return args["denoised"]

        m.set_model_sampler_post_cfg_function(_post_cfg)

    return m, recorder


# ------------------------------------------------------------------ heatmap visualization


def _heatmap_rgb(score_hw: torch.Tensor) -> np.ndarray:
    """Convert `[H, W]` float score map into an inferno-like `[H, W, 3]` uint8 RGB array."""
    s = score_hw.detach().float().cpu().numpy()
    lo = float(np.percentile(s, 5.0))
    hi = float(np.percentile(s, 99.0))
    u = np.clip((s - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    r = np.clip(1.6 * u, 0.0, 1.0)
    g = np.clip(2.0 * (u - 0.35), 0.0, 1.0)
    b = np.clip(1.2 * (0.55 - np.abs(u - 0.35)) + 0.6 * np.maximum(0.0, u - 0.75), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def render_sensor_contact_sheet(
    recorder: H3SensorRecorder,
    out_path: str | Path,
    *,
    base_frame_rgb: np.ndarray | None = None,
    step: int = 0,
    tile_w: int = 320,
    tile_h: int = 180,
) -> Path:
    """Render a compact dark-card PNG contact sheet of recorded text & audio heatmaps.

    Refuses writing into repo `data/` (`AGENTS.md` §2).
    """
    out_p = Path(out_path)
    _guard_not_in_repo_data(out_p)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    tiles: list[tuple[str, torch.Tensor]] = []
    for (s_idx, b_idx, noun), t_map in sorted(recorder.heatmaps_text.items()):
        if s_idx == step:
            tiles.append((f"S{s_idx} B{b_idx:02d} txt:{noun}", t_map.mean(dim=0)))
    for (s_idx, b_idx), a_maps in sorted(recorder.heatmaps_audio.items()):
        if s_idx == step:
            tiles.append((f"S{s_idx} B{b_idx:02d} aud:stereo", a_maps["stereo"].mean(dim=0)))
            tiles.append((f"S{s_idx} B{b_idx:02d} aud:L", a_maps["left"].mean(dim=0)))
            tiles.append((f"S{s_idx} B{b_idx:02d} aud:R", a_maps["right"].mean(dim=0)))

    if not tiles:
        img = Image.new("RGB", (tile_w, tile_h), (16, 20, 28))
        ImageDraw.Draw(img).text((12, 12), f"No heatmaps recorded for step {step}", fill=(220, 226, 236))
        img.save(out_p)
        return out_p

    cols = min(4, len(tiles))
    rows = int(math.ceil(len(tiles) / float(cols)))
    pad = 8
    header_h = 20
    canvas = Image.new(
        "RGB",
        (cols * (tile_w + pad) + pad, rows * (tile_h + header_h + pad) + pad),
        (14, 18, 26),
    )
    draw = ImageDraw.Draw(canvas)
    bg_resized = (
        Image.fromarray(base_frame_rgb.astype(np.uint8), mode="RGB").resize(
            (tile_w, tile_h), Image.Resampling.BILINEAR
        )
        if base_frame_rgb is not None
        else None
    )

    for idx, (label, hw_map) in enumerate(tiles):
        r_i, c_i = divmod(idx, cols)
        x0 = pad + c_i * (tile_w + pad)
        y0 = pad + r_i * (tile_h + header_h + pad)
        draw.text((x0 + 2, y0 + 2), label[:40], fill=(226, 232, 240))
        hm = Image.fromarray(_heatmap_rgb(hw_map), mode="RGB").resize(
            (tile_w, tile_h), Image.Resampling.NEAREST
        )
        tile_img = Image.blend(bg_resized, hm, alpha=0.58) if bg_resized is not None else hm
        canvas.paste(tile_img, (x0, y0 + header_h))

    canvas.save(out_p)
    return out_p


if __name__ == "__main__":
    bootstrap(cpu=True)
    print(
        "bench/probe_h3_sensors.py: read-only 4-sensor H3 probe module loaded cleanly "
        "(run `bench/check_probe_h3_sensors.py` for offline CPU verification)."
    )
