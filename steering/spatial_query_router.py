"""Standalone `1x` Spatial Visual Query Router (`MiniMaxH3SpatialQueryRouter`).

Routes two distinct text prompts (base prompt = outside region, `region_conditioning` = inside
region, arbitrary token lengths) to disjoint spatial regions of H3's video tokens (`x_start..x_end`,
`y_start..y_end`, optional `invert`, or an optional 2D/video `MASK`) in a single `1.00x` pass with
**100% stock 3D RoPE** (zero causal Video VAE shimmer).

- During isolated blocks (`isolation`, default `"early_0_24"`): runs symmetric half-length K/V
  isolation (`attn_mask=None`, using `comfy_kitchen.int8_attention` on CUDA for large video zones)
  so base tokens attend only to `[txt_base, shared_non_vid, vid_base]` and region tokens attend
  only to `[txt_region, shared_non_vid, vid_region]`, blocking 1-hop cross-world text bleed.
- On remaining blocks (`25..49` by default): blends seamlessly across `K_vid_all` via the main
  pass (`Sol` / `comfy kitchen`) plus one coalesced `index_copy_` splice for the region.
"""

from __future__ import annotations

from collections import OrderedDict
import functools
import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from . import hooks as _steer_hooks
from .query_router import (
    _extract_cond_tags,
    _extract_cond_tensor,
)

logger = logging.getLogger("ComfyUI-h3-explorations.spatial_query_router")

SPATIAL_ROUTER_WRAPPER_KEY = "minimax_h3_spatial_query_router"
ISOLATION_MODES: tuple[str, ...] = (
    "early_0_24",
    "seamless_off",
    "early_0_16",
    "early_0_32",
    "all_0_49",
    "mid_20_34",
    "mid_20_49",
    "early_0_24_shield",
    "mid_20_34_shield",
)
AUDIO_ZONES: tuple[str, ...] = ("base", "region", "binaural_split", "follow_mask")
_REF_CACHE_MAX = 4
_CK_MIN_Q = 512
_SOFT_EPS = 1e-4


def parse_isolation_blocks(isolation: str, n_blocks: int = 50) -> frozenset[int]:
    """Resolve `isolation` combo preset into the set of DiT block indices using symmetric K/V isolation."""
    mode = str(isolation or "early_0_24").strip()
    n_b = max(0, int(n_blocks))
    if mode == "seamless_off":
        return frozenset()
    if mode == "early_0_16":
        return frozenset(range(0, min(17, n_b)))
    if mode in ("early_0_24", "early_0_24_shield"):
        return frozenset(range(0, min(25, n_b)))
    if mode == "early_0_32":
        return frozenset(range(0, min(33, n_b)))
    if mode == "all_0_49":
        return frozenset(range(0, n_b))
    if mode in ("mid_20_34", "mid_20_34_shield"):
        return frozenset(range(20, min(35, n_b))) if n_b > 20 else frozenset()
    if mode == "mid_20_49":
        return frozenset(range(20, n_b)) if n_b > 20 else frozenset()
    raise ValueError(f"isolation must be one of {ISOLATION_MODES}, got {isolation!r}")


def _fast_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Use `comfy_kitchen.int8_attention` on large CUDA bf16/fp16 queries; PyTorch SDPA fallback otherwise."""
    if q.is_cuda and int(q.shape[2]) >= _CK_MIN_Q and q.dtype in (torch.bfloat16, torch.float16):
        try:
            import comfy_kitchen as _ck

            if hasattr(_ck, "int8_attention"):
                return _ck.int8_attention(
                    q.contiguous(), k.contiguous(), v.contiguous(), scale=scale
                )
        except Exception:
            pass
    return F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=False, scale=scale
    )


def _downsample_mask_to_patches(
    mask: torch.Tensor,
    *,
    latent_t: int,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    """Downsample ComfyUI `MASK` (`[H, W]`, `[1, H, W]`, or `[F, H, W]`) to continuous `[0, 1]` patch weights."""
    if not torch.is_tensor(mask):
        raise ValueError(f"MiniMaxH3SpatialQueryRouter: mask must be a Tensor, got {type(mask).__name__}")
    m = mask.detach().to(device="cpu", dtype=torch.float32)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    elif m.ndim == 4:
        m = m[:, 0] if int(m.shape[1]) == 1 else m[..., 0]
    if m.ndim != 3:
        raise ValueError(f"MiniMaxH3SpatialQueryRouter mask must have 2 or 3 dims, got {tuple(mask.shape)}")
    if m.numel() == 0 or any(int(d) < 1 for d in m.shape):
        raise ValueError(f"MiniMaxH3SpatialQueryRouter: mask is empty ({tuple(mask.shape)})")
    if not bool(torch.isfinite(m).all()):
        raise ValueError("MiniMaxH3SpatialQueryRouter: mask contains NaN or Inf values")
    m = m.clamp(0.0, 1.0)

    f_in = int(m.shape[0])
    if f_in == 1:
        m_spatial = F.interpolate(
            m.unsqueeze(1), size=(patch_h, patch_w), mode="area"
        )[0, 0]
        return m_spatial.clamp(0.0, 1.0).unsqueeze(0).expand(latent_t, patch_h, patch_w).reshape(-1)

    from comfy.ldm.minimax.model import FRAME_PER_TOKEN

    f_expected = sum(int(FRAME_PER_TOKEN[k % 5]) for k in range(latent_t))
    if f_in != f_expected:
        raise ValueError(
            f"MiniMaxH3SpatialQueryRouter: mask has {f_in} frame(s), expected 1 or {f_expected} "
            f"for {latent_t} video latent step(s)"
        )

    m_hw = F.interpolate(m.unsqueeze(1), size=(patch_h, patch_w), mode="area")[:, 0]
    steps: list[torch.Tensor] = []
    f_cur = 0
    for k in range(latent_t):
        span = int(FRAME_PER_TOKEN[k % 5])
        steps.append(m_hw[f_cur : f_cur + span].mean(dim=0))
        f_cur += span
    return torch.stack(steps, dim=0).clamp(0.0, 1.0).reshape(-1)


def _build_spatial_partition(
    layout: Any,
    *,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    invert: bool,
    mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build 1D `torch.long` index partitions, continuous soft weights, and L/R binaural audio indices."""
    segs = layout.segments
    pos = layout.position_ids
    tx_spans = [(a, b) for (a, b, k) in segs if k == "text" and b > a]
    vid_spans = [(a, b) for (a, b, k) in segs if k == "video" and b > a]
    if not tx_spans or not vid_spans:
        raise ValueError("MiniMaxH3SpatialQueryRouter requires text and video segments in PackedLayout")
    tx_a, tx_b = tx_spans[0]
    va, vb = vid_spans[0]

    _t_len, latent_t, lat_h, lat_w, _a_t = layout.signature
    latent_t_i = max(int(latent_t), 1)
    n_vid = vb - va
    frame_patches = max(n_vid // latent_t_i, 1)

    vh = pos[va:vb, 1]
    vw = pos[va:vb, 2]
    h_min = float(vh.min().item())
    w_min = float(vw.min().item())
    h_span = max(float((vh.max() - vh.min()).item()), 1e-6)
    w_span = max(float((vw.max() - vw.min()).item()), 1e-6)
    x_lo, x_hi = (x_start, x_end) if x_start <= x_end else (x_end, x_start)
    y_lo, y_hi = (y_start, y_end) if y_start <= y_end else (y_end, y_start)

    vid_all = torch.arange(va, vb, dtype=torch.long)
    eff_weights_2d: torch.Tensor | None = None
    if mask is not None:
        patch_h = int(lat_h) // 2
        patch_w = int(lat_w) // 2
        expected = int(latent_t) * patch_h * patch_w
        if expected != n_vid:
            raise ValueError(
                f"MiniMaxH3SpatialQueryRouter layout patch count mismatch ({expected} != {n_vid})"
            )
        weights = _downsample_mask_to_patches(
            mask, latent_t=int(latent_t), patch_h=patch_h, patch_w=patch_w
        )
        eff_weights = (1.0 - weights).clamp(0.0, 1.0) if invert else weights
        eff_weights_2d = eff_weights.view(latent_t_i, frame_patches)
        mask_region = ~(weights >= 0.5) if invert else (weights >= 0.5)
        has_soft = bool(((eff_weights > _SOFT_EPS) & (eff_weights < 1.0 - _SOFT_EPS)).any().item())
        if has_soft:
            soft_sel = eff_weights > _SOFT_EPS
            idx_region_soft = vid_all[soft_sel]
            w_region_soft: torch.Tensor | None = eff_weights[soft_sel].to(dtype=torch.float32)
        else:
            idx_region_soft = vid_all[mask_region]
            w_region_soft = None
        frame_weights_mean = eff_weights_2d.mean(dim=1)
    else:
        h_u = (vh - h_min) / h_span
        w_u = (vw - w_min) / w_span
        mask_region = (w_u >= x_lo) & (w_u <= x_hi) & (h_u >= y_lo) & (h_u <= y_hi)
        if invert:
            mask_region = ~mask_region
        idx_region_soft = vid_all[mask_region]
        w_region_soft = None
        frame_weights_mean = (
            mask_region.float().view(latent_t_i, frame_patches).mean(dim=1)
            if (latent_t_i * frame_patches == n_vid)
            else mask_region.float().mean().unsqueeze(0)
        )

    idx_region = vid_all[mask_region]
    idx_base = vid_all[~mask_region]

    vid_t_steps = (
        pos[va : vb : frame_patches, 0]
        if (latent_t_i * frame_patches == n_vid and frame_patches > 0)
        else pos[va:va+1, 0]
    )
    frame_hw_grid = pos[va : va + min(frame_patches, n_vid), 1:3]

    aud_segs = [(int(a), int(b)) for (a, b, k) in segs if k == "audio" and b > a]
    aud_parts = [torch.arange(a, b, dtype=torch.long) for (a, b) in aud_segs]
    idx_aud = torch.cat(aud_parts, dim=0) if aud_parts else torch.empty(0, dtype=torch.long)

    aud_l_parts: list[torch.Tensor] = []
    aud_r_parts: list[torch.Tensor] = []
    w_aud_parts: list[torch.Tensor] = []
    for sa, sb in aud_segs:
        span_len = sb - sa
        if span_len >= 2 and (span_len % 2) == 0:
            mid = sa + span_len // 2
            aud_l_parts.append(torch.arange(sa, mid, dtype=torch.long))
            aud_r_parts.append(torch.arange(mid, sb, dtype=torch.long))
            # Channel-major packing: [L_0..L_{T-1}, R_0..R_{T-1}] share identical temporal ticks
            aud_t_half = pos[sa:mid, 0]
            k_aud = (aud_t_half.unsqueeze(1) - vid_t_steps.unsqueeze(0)).abs().argmin(dim=1)
            w_half = frame_weights_mean[k_aud]
            w_aud_parts.extend([w_half, w_half])
        else:
            aud_t_all = pos[sa:sb, 0]
            k_aud = (aud_t_all.unsqueeze(1) - vid_t_steps.unsqueeze(0)).abs().argmin(dim=1)
            w_aud_parts.append(frame_weights_mean[k_aud])
    idx_aud_l = torch.cat(aud_l_parts, dim=0) if aud_l_parts else torch.empty(0, dtype=torch.long)
    idx_aud_r = torch.cat(aud_r_parts, dim=0) if aud_r_parts else torch.empty(0, dtype=torch.long)
    w_aud = (
        torch.cat(w_aud_parts, dim=0).clamp(0.0, 1.0).to(dtype=torch.float32)
        if w_aud_parts
        else torch.empty(0, dtype=torch.float32)
    )

    # Partition keyframe/first-frame/last-frame `cond` segments by their resolved timestamp & region:
    # `cond` patches outside the region remain available to `idx_base_kv` and `idx_nt_region`;
    # portal-interior `cond` patches are excluded from both `idx_region_kv` and `idx_nt_region`.
    mask_region_2d = (
        mask_region.view(latent_t_i, frame_patches)
        if (mask is not None and (latent_t_i * frame_patches == n_vid))
        else None
    )
    cond_base_parts: list[torch.Tensor] = []
    for ca, cb, kind in segs:
        if kind != "cond" or cb <= ca:
            continue
        c_idx = torch.arange(ca, cb, dtype=torch.long)
        c_len = cb - ca
        if mask_region_2d is not None:
            if (c_len % frame_patches) == 0:
                c_reg_parts: list[torch.Tensor] = []
                for c_off in range(ca, cb, frame_patches):
                    t_c = float(pos[c_off, 0].item())
                    k_lat = int((vid_t_steps - t_c).abs().argmin().item())
                    c_reg_parts.append(mask_region_2d[k_lat])
                c_in_reg = torch.cat(c_reg_parts, dim=0)
            else:
                k_lat = (pos[ca:cb, 0:1] - vid_t_steps.unsqueeze(0)).abs().argmin(dim=1)
                p_idx = ((pos[ca:cb, 1:3].unsqueeze(1) - frame_hw_grid.unsqueeze(0)) ** 2).sum(dim=-1).argmin(dim=1)
                c_in_reg = mask_region_2d[k_lat, p_idx]
        else:
            ch_u = (pos[ca:cb, 1] - h_min) / h_span
            cw_u = (pos[ca:cb, 2] - w_min) / w_span
            c_in_reg = (cw_u >= x_lo) & (cw_u <= x_hi) & (ch_u >= y_lo) & (ch_u <= y_hi)
            if invert:
                c_in_reg = ~c_in_reg
        outside = c_idx[~c_in_reg]
        if outside.numel() > 0:
            cond_base_parts.append(outside)
    idx_cond_base = torch.cat(cond_base_parts, dim=0) if cond_base_parts else torch.empty(0, dtype=torch.long)

    # Non-text, non-video, non-cond tokens (cond_audio, ref_img, ref_audio, audio) stay shared in isolated KV.
    other_parts = [
        torch.arange(a, b, dtype=torch.long)
        for (a, b, k) in segs
        if k not in ("text", "video", "cond") and b > a
    ]
    idx_other = torch.cat(other_parts, dim=0) if other_parts else torch.empty(0, dtype=torch.long)
    idx_nt_region = torch.cat([idx_other, idx_cond_base, vid_all], dim=0)

    non_txt_parts = [torch.arange(a, b, dtype=torch.long) for (a, b, k) in segs if k != "text" and b > a]
    idx_non_txt = torch.cat(non_txt_parts, dim=0) if non_txt_parts else torch.empty(0, dtype=torch.long)

    reg_for_w = idx_region if idx_region.numel() > 0 else idx_region_soft
    binaural_region_on_left = False
    if reg_for_w.numel() > 0 and idx_base.numel() > 0:
        mean_w_reg = float(pos[reg_for_w, 2].float().mean().item())
        mean_w_base = float(pos[idx_base, 2].float().mean().item())
        binaural_region_on_left = mean_w_reg < (mean_w_base - 1e-4)

    return {
        "idx_base": idx_base,
        "idx_region": idx_region,
        "idx_region_soft": idx_region_soft,
        "w_region_soft": w_region_soft,
        "idx_aud": idx_aud,
        "idx_aud_l": idx_aud_l,
        "idx_aud_r": idx_aud_r,
        "w_aud": w_aud,
        "binaural_region_on_left": binaural_region_on_left,
        "idx_cond_base": idx_cond_base,
        "idx_other": idx_other,
        "idx_nt_region": idx_nt_region,
        "idx_non_txt": idx_non_txt,
        "tx_a": tx_a,
        "tx_b": tx_b,
    }


def _positions_in_index_vec(s_tot: int, target_idx: torch.Tensor, container_idx: torch.Tensor) -> torch.Tensor:
    """Return positions within `container_idx` whose values appear in `target_idx`."""
    if target_idx.numel() == 0 or container_idx.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    marker = torch.zeros(s_tot, dtype=torch.bool)
    marker[target_idx] = True
    return marker[container_idx].nonzero(as_tuple=False).flatten()


def build_spatial_indices(
    layout: Any,
    *,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    invert: bool,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Build 1D `torch.long` sequence index partitions for base, region, audio, and shared K/V tokens.

    Returns `(idx_base, idx_region, idx_aud, idx_other_non_vid, idx_non_txt_all, tx_a, tx_b)`.
    """
    part = _build_spatial_partition(
        layout,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        invert=invert,
        mask=mask,
    )
    return (
        part["idx_base"],
        part["idx_region"],
        part["idx_aud"],
        part["idx_other"],
        part["idx_non_txt"],
        part["tx_a"],
        part["tx_b"],
    )


class SpatialVisualQRRuntime:
    """Per-step `1x` Spatial Visual Query Router runtime."""

    def __init__(
        self,
        *,
        dm: Any,
        shadow_b_init: torch.Tensor,
        shadow_b_tags: torch.Tensor | None,
        x_start: float,
        x_end: float,
        y_start: float,
        y_end: float,
        invert: bool,
        mask: torch.Tensor | None,
        isolate_blocks: frozenset[int],
        audio_zone: str,
        shield_region: bool = False,
        binaural_start_block: int | None = None,
    ) -> None:
        self.dm = dm
        self.shadow_b_init = shadow_b_init
        self.shadow_b_tags = shadow_b_tags
        self.x_start = float(x_start)
        self.x_end = float(x_end)
        self.y_start = float(y_start)
        self.y_end = float(y_end)
        self.invert = bool(invert)
        self.mask = mask
        self.isolate_blocks = isolate_blocks
        self.warmup_blocks = (
            frozenset(range(0, min(isolate_blocks)))
            if (isolate_blocks and min(isolate_blocks) > 0)
            else frozenset()
        )
        self.audio_zone = audio_zone
        self.shield_region = bool(shield_region)
        self.n_blocks = len(dm.blocks)
        self.binaural_start_block = (
            (30 if self.n_blocks >= 40 else (self.n_blocks // 2))
            if binaural_start_block is None
            else int(binaural_start_block)
        )
        self.active = True

        self.h_shadow_b: torch.Tensor | None = None
        self.rope_b: torch.Tensor | None = None
        self.idx_base: torch.Tensor | None = None
        self.idx_region: torch.Tensor | None = None
        self.idx_region_soft: torch.Tensor | None = None
        self.w_region_soft: torch.Tensor | None = None
        self.idx_aud: torch.Tensor | None = None
        self.idx_aud_l: torch.Tensor | None = None
        self.idx_aud_r: torch.Tensor | None = None
        self.w_aud: torch.Tensor | None = None
        self.idx_aud_fm_base: torch.Tensor | None = None
        self.idx_aud_fm_reg: torch.Tensor | None = None
        self.idx_aud_fm_pure_base: torch.Tensor | None = None
        self.idx_aud_fm_pure_reg: torch.Tensor | None = None
        self.nt_aud_l_pos: torch.Tensor | None = None
        self.nt_aud_r_pos: torch.Tensor | None = None
        self.nt_base_vid_pos: torch.Tensor | None = None
        self.nt_fm_base_aud_pos: torch.Tensor | None = None
        self.nt_fm_pure_base_aud_pos: torch.Tensor | None = None
        self.reg_kv_aud_l_pos: torch.Tensor | None = None
        self.reg_kv_aud_r_pos: torch.Tensor | None = None
        self.reg_kv_base_vid_pos: torch.Tensor | None = None
        self.reg_kv_fm_base_aud_pos: torch.Tensor | None = None
        self.reg_kv_fm_pure_base_aud_pos: torch.Tensor | None = None
        self.binaural_region_on_left = False
        self.idx_base_kv: torch.Tensor | None = None
        self.idx_region_kv: torch.Tensor | None = None
        self.idx_region_soft_kv: torch.Tensor | None = None
        self.idx_nt_all: torch.Tensor | None = None
        self.tx_a = 0
        self.tx_b = 0
        self._cur: dict[str, Any] | None = None

    def prepare_block(self, block_index: int, args: dict[str, Any]) -> None:
        if not self.active:
            return
        from comfy.ldm.minimax.model import _mod_scale_shift

        x_full = args["img"]
        layout = args["layout"]
        device = x_full.device
        dtype = x_full.dtype
        block = self.dm.blocks[block_index]
        heads = int(block.attn.heads)
        head_dim = int(block.attn.head_dim)
        l_txt_b = int(self.shadow_b_init.shape[1] if self.shadow_b_init.ndim == 3 else self.shadow_b_init.shape[0])

        if block_index == 0 or self.h_shadow_b is None:
            part = _build_spatial_partition(
                layout,
                x_start=self.x_start,
                x_end=self.x_end,
                y_start=self.y_start,
                y_end=self.y_end,
                invert=self.invert,
                mask=self.mask,
            )
            tx_a, tx_b = part["tx_a"], part["tx_b"]
            self.tx_a, self.tx_b = tx_a, tx_b
            self.idx_base = part["idx_base"].to(device=device)
            self.idx_region = part["idx_region"].to(device=device)
            self.idx_region_soft = part["idx_region_soft"].to(device=device)
            w_soft = part["w_region_soft"]
            self.w_region_soft = None if w_soft is None else w_soft.to(device=device)
            self.idx_aud = part["idx_aud"].to(device=device)
            self.idx_aud_l = part["idx_aud_l"].to(device=device)
            self.idx_aud_r = part["idx_aud_r"].to(device=device)
            w_aud_cpu = part["w_aud"]
            self.w_aud = w_aud_cpu.to(device=device)
            fm_reg_sel = w_aud_cpu >= 0.5
            aud_fm_base_cpu = part["idx_aud"][~fm_reg_sel]
            aud_fm_reg_cpu = part["idx_aud"][fm_reg_sel]
            aud_fm_pure_base_cpu = part["idx_aud"][w_aud_cpu <= _SOFT_EPS]
            aud_fm_pure_reg_cpu = part["idx_aud"][w_aud_cpu >= (1.0 - _SOFT_EPS)]
            self.idx_aud_fm_base = aud_fm_base_cpu.to(device=device)
            self.idx_aud_fm_reg = aud_fm_reg_cpu.to(device=device)
            self.idx_aud_fm_pure_base = aud_fm_pure_base_cpu.to(device=device)
            self.idx_aud_fm_pure_reg = aud_fm_pure_reg_cpu.to(device=device)
            self.binaural_region_on_left = bool(part["binaural_region_on_left"])
            nt_reg_cpu = part["idx_nt_region"]
            self.idx_nt_all = nt_reg_cpu.to(device=device)
            idx_other_dev = part["idx_other"].to(device=device)
            idx_cond_base_dev = part["idx_cond_base"].to(device=device)
            self.idx_base_kv = torch.cat([idx_other_dev, idx_cond_base_dev, self.idx_base], dim=0)
            self.idx_region_kv = torch.cat([idx_other_dev, self.idx_region], dim=0)
            reg_soft_kv_cpu = torch.cat([part["idx_other"], part["idx_region_soft"]], dim=0)
            self.idx_region_soft_kv = reg_soft_kv_cpu.to(device=device)

            s_tot = int(layout.position_ids.shape[0])
            if part["idx_aud_l"].numel() > 0 and part["idx_aud_r"].numel() > 0:
                self.nt_aud_l_pos = _positions_in_index_vec(s_tot, part["idx_aud_l"], nt_reg_cpu).to(device=device)
                self.nt_aud_r_pos = _positions_in_index_vec(s_tot, part["idx_aud_r"], nt_reg_cpu).to(device=device)
                self.reg_kv_aud_l_pos = _positions_in_index_vec(s_tot, part["idx_aud_l"], reg_soft_kv_cpu).to(device=device)
                self.reg_kv_aud_r_pos = _positions_in_index_vec(s_tot, part["idx_aud_r"], reg_soft_kv_cpu).to(device=device)
            else:
                self.nt_aud_l_pos = torch.empty(0, dtype=torch.long, device=device)
                self.nt_aud_r_pos = torch.empty(0, dtype=torch.long, device=device)
                self.reg_kv_aud_l_pos = torch.empty(0, dtype=torch.long, device=device)
                self.reg_kv_aud_r_pos = torch.empty(0, dtype=torch.long, device=device)

            self.nt_base_vid_pos = _positions_in_index_vec(s_tot, part["idx_base"], nt_reg_cpu).to(device=device)
            self.reg_kv_base_vid_pos = _positions_in_index_vec(s_tot, part["idx_base"], reg_soft_kv_cpu).to(device=device)
            self.nt_fm_base_aud_pos = _positions_in_index_vec(s_tot, aud_fm_base_cpu, nt_reg_cpu).to(device=device)
            self.reg_kv_fm_base_aud_pos = _positions_in_index_vec(s_tot, aud_fm_base_cpu, reg_soft_kv_cpu).to(device=device)
            self.nt_fm_pure_base_aud_pos = _positions_in_index_vec(s_tot, aud_fm_pure_base_cpu, nt_reg_cpu).to(device=device)
            self.reg_kv_fm_pure_base_aud_pos = _positions_in_index_vec(s_tot, aud_fm_pure_base_cpu, reg_soft_kv_cpu).to(device=device)

            if self.idx_region_soft.numel() == 0 and self.audio_zone in ("base", "follow_mask"):
                self.active = False
                self._cur = None
                return

            src_b = self.shadow_b_init[0] if self.shadow_b_init.ndim == 3 else self.shadow_b_init
            self.h_shadow_b = src_b.to(device=device, dtype=dtype).clone()
            l_txt = tx_b - tx_a
            rope_txt_init = _steer_hooks._slice_rope(args.get("rope_freqs"), tx_a, tx_b)
            self.rope_b = _steer_hooks._build_shadow_text_rope(
                self.dm, l_txt, l_txt_b, device, dtype, rope_txt_init
            )

        l_txt = self.tx_b - self.tx_a
        mod_txt = _steer_hooks._slice_mod_segments(args["mod_segments"], self.tx_a, self.tx_b)
        mod_txt_b = _steer_hooks._adapt_text_mod_segments(
            mod_txt, l_txt, l_txt_b, shadow_tags=self.shadow_b_tags
        )
        rope_txt = _steer_hooks._slice_rope(args.get("rope_freqs"), self.tx_a, self.tx_b)
        rope_b = rope_txt if l_txt_b == l_txt else self.rope_b

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(args["t_emb"])
        h_in_b = _mod_scale_shift(block.norm1(self.h_shadow_b), shift_msa, scale_msa, mod_txt_b)
        q_b, k_b, v_b = block.attn.qkv_proj(h_in_b).split(heads * head_dim, dim=-1)
        v_b = v_b.view(l_txt_b, heads, head_dim)
        q_b, k_b = _steer_hooks._apply_rms_rope_txt(
            block, q_b, k_b, rope_b, heads=heads, head_dim=head_dim, device=device
        )

        self._cur = {
            "block_index": block_index,
            "block": block,
            "heads": heads,
            "head_dim": head_dim,
            "l_txt_b": l_txt_b,
            "q_txt_b": q_b.transpose(0, 1).unsqueeze(0),
            "k_txt_b": k_b.transpose(0, 1).unsqueeze(0),
            "v_txt_b": v_b.transpose(0, 1).unsqueeze(0),
            "mod_txt_b": mod_txt_b,
            "gate_msa": gate_msa,
            "shift_mlp": shift_mlp,
            "scale_mlp": scale_mlp,
            "gate_mlp": gate_mlp,
            "shadow_b_attn_2d": None,
            "warmup": block_index in self.warmup_blocks,
            "iso": (
                (block_index in self.isolate_blocks)
                and (self.idx_base.numel() > 0)
                and (self.idx_region.numel() > 0)
            ),
            "done": False,
        }

    def clear(self) -> None:
        """Release all per-step CUDA tensors (call inside `pause_malloc_graph_ctx()`)."""
        self.active = False
        self._cur = None
        self.h_shadow_b = None
        self.rope_b = None
        self.idx_base = None
        self.idx_region = None
        self.idx_region_soft = None
        self.w_region_soft = None
        self.idx_aud = None
        self.idx_aud_l = None
        self.idx_aud_r = None
        self.w_aud = None
        self.idx_aud_fm_base = None
        self.idx_aud_fm_reg = None
        self.idx_aud_fm_pure_base = None
        self.idx_aud_fm_pure_reg = None
        self.nt_aud_l_pos = None
        self.nt_aud_r_pos = None
        self.nt_base_vid_pos = None
        self.nt_fm_base_aud_pos = None
        self.nt_fm_pure_base_aud_pos = None
        self.reg_kv_aud_l_pos = None
        self.reg_kv_aud_r_pos = None
        self.reg_kv_base_vid_pos = None
        self.reg_kv_fm_base_aud_pos = None
        self.reg_kv_fm_pure_base_aud_pos = None
        self.binaural_region_on_left = False
        self.idx_base_kv = None
        self.idx_region_kv = None
        self.idx_region_soft_kv = None
        self.idx_nt_all = None

    def _binaural_audio_outputs(
        self,
        q_p: torch.Tensor,
        k_p: torch.Tensor,
        v_p: torch.Tensor,
        k_b_world: torch.Tensor,
        v_b_world: torch.Tensor,
        l_txt_b: int,
        scale: float | None,
        *,
        shielded: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.idx_aud_l is not None and self.idx_aud_r is not None
        assert self.idx_base is not None and self.idx_region is not None
        b_aud_l = self.reg_kv_aud_l_pos if shielded else self.nt_aud_l_pos
        b_aud_r = self.reg_kv_aud_r_pos if shielded else self.nt_aud_r_pos
        b_base_vid = self.reg_kv_base_vid_pos if shielded else self.nt_base_vid_pos
        assert b_aud_l is not None and b_aud_r is not None
        if self.binaural_region_on_left:
            idx_aud_base, idx_aud_reg = self.idx_aud_r, self.idx_aud_l
            b_masked_base = b_aud_r
        else:
            idx_aud_base, idx_aud_reg = self.idx_aud_l, self.idx_aud_r
            b_masked_base = b_aud_l

        has_both_vid = (self.idx_base.numel() > 0) and (self.idx_region.numel() > 0)
        s_total = int(k_p.shape[2])
        bias_base = torch.zeros((1, 1, 1, s_total), device=k_p.device, dtype=k_p.dtype)
        bias_base[:, :, :, idx_aud_reg] = float("-inf")
        if has_both_vid:
            bias_base[:, :, :, self.idx_region] = float("-inf")
        q_aud_base = q_p.index_select(2, idx_aud_base).to(dtype=k_p.dtype)
        out_aud_base = F.scaled_dot_product_attention(
            q_aud_base, k_p, v_p, attn_mask=bias_base, dropout_p=0.0, is_causal=False, scale=scale
        )
        del bias_base, q_aud_base

        s_b_len = int(k_b_world.shape[2])
        bias_reg = torch.zeros((1, 1, 1, s_b_len), device=k_b_world.device, dtype=k_b_world.dtype)
        if b_masked_base.numel() > 0:
            bias_reg[:, :, :, b_masked_base + l_txt_b] = float("-inf")
        if has_both_vid and b_base_vid is not None and b_base_vid.numel() > 0:
            bias_reg[:, :, :, b_base_vid + l_txt_b] = float("-inf")
        q_aud_reg = q_p.index_select(2, idx_aud_reg).to(dtype=k_b_world.dtype)
        out_aud_reg = F.scaled_dot_product_attention(
            q_aud_reg, k_b_world, v_b_world, attn_mask=bias_reg, dropout_p=0.0, is_causal=False, scale=scale
        )
        del bias_reg, q_aud_reg
        return idx_aud_base, out_aud_base, idx_aud_reg, out_aud_reg

    def _follow_mask_audio_outputs(
        self,
        q_p: torch.Tensor,
        k_p: torch.Tensor,
        v_p: torch.Tensor,
        k_b_world: torch.Tensor,
        v_b_world: torch.Tensor,
        l_txt_b: int,
        block_index: int,
        heads: int,
        head_dim: int,
        scale: float | None,
        *,
        shielded: bool = False,
    ) -> torch.Tensor:
        assert self.idx_aud is not None and self.w_aud is not None
        assert self.idx_aud_fm_base is not None and self.idx_aud_fm_reg is not None
        assert self.idx_aud_fm_pure_base is not None and self.idx_aud_fm_pure_reg is not None
        assert self.idx_base is not None and self.idx_region is not None

        idx_aud = self.idx_aud
        w_aud = self.w_aud
        n_aud = int(idx_aud.numel())
        s_total = int(k_p.shape[2])
        s_b_len = int(k_b_world.shape[2])
        has_both_vid = (self.idx_base.numel() > 0) and (self.idx_region.numel() > 0)
        b_base_vid = self.reg_kv_base_vid_pos if shielded else self.nt_base_vid_pos
        b_fm_base_aud = self.reg_kv_fm_base_aud_pos if shielded else self.nt_fm_base_aud_pos
        b_fm_pure_base_aud = (
            self.reg_kv_fm_pure_base_aud_pos if shielded else self.nt_fm_pure_base_aud_pos
        )

        def _attn_base(q_sub: torch.Tensor, opposite_mask_aud: torch.Tensor) -> torch.Tensor:
            q_b = q_sub.to(dtype=k_p.dtype)
            has_reg_aud = opposite_mask_aud.numel() > 0
            if not has_reg_aud and not has_both_vid:
                return _fast_attn(q_b, k_p, v_p, scale=scale)
            bias_b = torch.zeros((1, 1, 1, s_total), device=k_p.device, dtype=k_p.dtype)
            if has_reg_aud:
                bias_b[:, :, :, opposite_mask_aud] = float("-inf")
            if has_both_vid:
                bias_b[:, :, :, self.idx_region] = float("-inf")
            return F.scaled_dot_product_attention(
                q_b, k_p, v_p, attn_mask=bias_b, dropout_p=0.0, is_causal=False, scale=scale
            )

        def _attn_reg(q_sub: torch.Tensor, opposite_mask_aud_pos: torch.Tensor | None) -> torch.Tensor:
            q_r = q_sub.to(dtype=k_b_world.dtype)
            mask_base_aud = opposite_mask_aud_pos is not None and opposite_mask_aud_pos.numel() > 0
            mask_base_vid = has_both_vid and b_base_vid is not None and b_base_vid.numel() > 0
            if not mask_base_aud and not mask_base_vid:
                return _fast_attn(q_r, k_b_world, v_b_world, scale=scale)
            bias_r = torch.zeros((1, 1, 1, s_b_len), device=k_b_world.device, dtype=k_b_world.dtype)
            if mask_base_aud:
                bias_r[:, :, :, opposite_mask_aud_pos + l_txt_b] = float("-inf")
            if mask_base_vid:
                bias_r[:, :, :, b_base_vid + l_txt_b] = float("-inf")
            return F.scaled_dot_product_attention(
                q_r, k_b_world, v_b_world, attn_mask=bias_r, dropout_p=0.0, is_causal=False, scale=scale
            )

        out_aud = torch.empty((1, heads, n_aud, head_dim), device=q_p.device, dtype=k_p.dtype)
        is_reg_hard = w_aud >= 0.5
        if block_index < self.binaural_start_block:
            pos_b = (~is_reg_hard).nonzero(as_tuple=False).flatten()
            pos_r = is_reg_hard.nonzero(as_tuple=False).flatten()
            if pos_b.numel() > 0:
                out_aud.index_copy_(
                    2, pos_b, _attn_base(q_p.index_select(2, idx_aud[pos_b]), self.idx_aud_fm_reg)
                )
            if pos_r.numel() > 0:
                out_aud.index_copy_(
                    2,
                    pos_r,
                    _attn_reg(q_p.index_select(2, idx_aud[pos_r]), b_fm_base_aud).to(dtype=out_aud.dtype),
                )
            return out_aud

        is_pure_base = w_aud <= _SOFT_EPS
        is_pure_reg = w_aud >= (1.0 - _SOFT_EPS)
        is_trans = (~is_pure_base) & (~is_pure_reg)
        if not bool(is_trans.any().item()):
            pos_b = (~is_reg_hard).nonzero(as_tuple=False).flatten()
            pos_r = is_reg_hard.nonzero(as_tuple=False).flatten()
            if pos_b.numel() > 0:
                out_aud.index_copy_(
                    2, pos_b, _attn_base(q_p.index_select(2, idx_aud[pos_b]), self.idx_aud_fm_reg)
                )
            if pos_r.numel() > 0:
                out_aud.index_copy_(
                    2,
                    pos_r,
                    _attn_reg(q_p.index_select(2, idx_aud[pos_r]), b_fm_base_aud).to(dtype=out_aud.dtype),
                )
            return out_aud

        # Late crossfade: mask ONLY pure opposite-world audio so transitional 0 < w < 1 tokens retain self-attention.
        need_base = ~is_pure_reg
        need_reg = ~is_pure_base
        out_b_buf = _attn_base(q_p.index_select(2, idx_aud[need_base]), self.idx_aud_fm_pure_reg)
        out_r_buf = _attn_reg(q_p.index_select(2, idx_aud[need_reg]), b_fm_pure_base_aud)
        b_full = torch.zeros_like(out_aud, dtype=torch.float32)
        r_full = torch.zeros_like(out_aud, dtype=torch.float32)
        b_full.index_copy_(2, need_base.nonzero(as_tuple=False).flatten(), out_b_buf.float())
        r_full.index_copy_(2, need_reg.nonzero(as_tuple=False).flatten(), out_r_buf.float())
        w_view = w_aud.view(1, 1, n_aud, 1).float()
        blended = (1.0 - w_view) * b_full + w_view * r_full
        out_aud.copy_(blended.to(dtype=out_aud.dtype))
        return out_aud

    def run_attention_override(
        self,
        q_in: Any,
        k_in: Any,
        v_in: Any,
        pass_through_fn: Any,
        *,
        skip_output_reshape: bool,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Execute spatial attention routing while strictly respecting ComfyUI `malloc_graph` ownership.

        CRITICAL (`comfy_aimdo` + `CUDAMallocAsyncAllocator`):
        - `q_in`/`k_in`/`v_in` are allocated inside the active `malloc_graph`, so they MUST only be
          `.take()`-ed/freed OUTSIDE `pause_malloc_graph_ctx()`.
        - The returned tensor MUST be allocated OUTSIDE `pause_malloc_graph_ctx()` (either pre-allocated
          as `out_ret` in `iso` mode or produced by `pass_through_fn()` in seamless mode), while all
          internal split/concat/`_fast_attn` temporaries are allocated and freed INSIDE `pause_malloc_graph_ctx()`.
        """
        cur = self._cur
        assert cur is not None and not cur["done"]
        assert self.idx_base is not None and self.idx_region is not None and self.idx_aud is not None
        assert self.idx_region_soft is not None and self.idx_aud_l is not None and self.idx_aud_r is not None
        assert self.idx_base_kv is not None and self.idx_region_kv is not None and self.idx_nt_all is not None
        assert self.h_shadow_b is not None

        cur["done"] = True
        heads, head_dim = cur["heads"], cur["head_dim"]
        inner = heads * head_dim
        l_txt_b = cur["l_txt_b"]
        need_finish_b = cur["block_index"] < (self.n_blocks - 1)
        route_aud_region = (self.audio_zone == "region") and (self.idx_aud.numel() > 0)
        binaural_active = (
            (self.audio_zone == "binaural_split")
            and (cur["block_index"] >= self.binaural_start_block)
            and (self.idx_aud_l.numel() > 0)
            and (self.idx_aud_r.numel() > 0)
        )

        # Peek while `malloc_graph` is ACTIVE so containers keep strong refs throughout the paused region.
        q_p = q_in if torch.is_tensor(q_in) else q_in.peek()
        k_p = k_in if torch.is_tensor(k_in) else k_in.peek()
        v_p = v_in if torch.is_tensor(v_in) else v_in.peek()

        if cur["iso"]:
            s_total = int(q_p.shape[2])
            if skip_output_reshape:
                out_ret = torch.empty((1, heads, s_total, head_dim), device=q_p.device, dtype=q_p.dtype)
            else:
                out_ret = torch.empty((1, s_total, inner), device=q_p.device, dtype=q_p.dtype)

            with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
                k_dtype, v_dtype = k_p.dtype, v_p.dtype
                q_txt_b = cur["q_txt_b"].to(dtype=k_dtype)
                k_txt_b = cur["k_txt_b"].to(dtype=k_dtype)
                v_txt_b = cur["v_txt_b"].to(dtype=v_dtype)
                cur["q_txt_b"] = cur["k_txt_b"] = cur["v_txt_b"] = None

                q_txt_a = q_p[:, :, self.tx_a : self.tx_b, :]
                k_txt_a = k_p[:, :, self.tx_a : self.tx_b, :]
                v_txt_a = v_p[:, :, self.tx_a : self.tx_b, :]

                q_vid_a = q_p.index_select(2, self.idx_base)
                k_a_kv = torch.cat([k_txt_a, k_p.index_select(2, self.idx_base_kv)], dim=2)
                v_a_kv = torch.cat([v_txt_a, v_p.index_select(2, self.idx_base_kv)], dim=2)
                q_comb_a = torch.cat([q_txt_a, q_vid_a], dim=2)
                out_a_all = _fast_attn(q_comb_a, k_a_kv, v_a_kv, scale=scale)
                l_txt_a = self.tx_b - self.tx_a
                out_txt_a = out_a_all[:, :, :l_txt_a, :]
                out_vid_a = out_a_all[:, :, l_txt_a:, :]
                del q_vid_a, k_a_kv, v_a_kv, q_comb_a, out_a_all

                q_vid_b = q_p.index_select(2, self.idx_region)
                k_b_kv = torch.cat([k_txt_b, k_p.index_select(2, self.idx_region_kv)], dim=2)
                v_b_kv = torch.cat([v_txt_b, v_p.index_select(2, self.idx_region_kv)], dim=2)
                q_comb_b = torch.cat([q_txt_b, q_vid_b], dim=2) if need_finish_b else q_vid_b
                out_vid_b: torch.Tensor | None = None
                if q_comb_b.shape[2] > 0:
                    out_b_all = _fast_attn(q_comb_b, k_b_kv, v_b_kv, scale=scale)
                    if need_finish_b:
                        cur["shadow_b_attn_2d"] = (
                            out_b_all[:, :, :l_txt_b, :][0]
                            .transpose(0, 1)
                            .reshape(l_txt_b, inner)
                            .to(dtype=self.h_shadow_b.dtype)
                        )
                        out_vid_b = out_b_all[:, :, l_txt_b:, :]
                    else:
                        out_vid_b = out_b_all
                    del out_b_all
                del q_vid_b, k_b_kv, v_b_kv, q_comb_b

                def _write_iso(idx: torch.Tensor, src_bh: torch.Tensor) -> None:
                    if skip_output_reshape:
                        out_ret.index_copy_(2, idx, src_bh.to(dtype=out_ret.dtype))
                    else:
                        src_2d = (
                            src_bh[0].transpose(0, 1).reshape(1, int(idx.numel()), inner).to(dtype=out_ret.dtype)
                        )
                        out_ret.index_copy_(1, idx, src_2d)

                if skip_output_reshape:
                    out_ret[:, :, self.tx_a : self.tx_b, :].copy_(out_txt_a.to(dtype=out_ret.dtype))
                else:
                    out_ret[:, self.tx_a : self.tx_b, :].copy_(
                        out_txt_a[0].transpose(0, 1).reshape(1, l_txt_a, inner).to(dtype=out_ret.dtype)
                    )
                if self.idx_base.numel() > 0:
                    _write_iso(self.idx_base, out_vid_a)
                if out_vid_b is not None and self.idx_region.numel() > 0:
                    _write_iso(self.idx_region, out_vid_b)

                other_mask = torch.ones(s_total, dtype=torch.bool, device=q_p.device)
                other_mask[self.tx_a : self.tx_b] = False
                other_mask[self.idx_base] = False
                other_mask[self.idx_region] = False
                if self.idx_aud.numel() > 0:
                    other_mask[self.idx_aud] = False
                    if self.audio_zone == "follow_mask":
                        k_b_world = torch.cat([k_txt_b, k_p.index_select(2, self.idx_nt_all)], dim=2)
                        v_b_world = torch.cat([v_txt_b, v_p.index_select(2, self.idx_nt_all)], dim=2)
                        out_aud = self._follow_mask_audio_outputs(
                            q_p, k_p, v_p, k_b_world, v_b_world, l_txt_b,
                            cur["block_index"], heads, head_dim, scale, shielded=False,
                        )
                        _write_iso(self.idx_aud, out_aud)
                        del k_b_world, v_b_world, out_aud
                    elif binaural_active:
                        k_b_world = torch.cat([k_txt_b, k_p.index_select(2, self.idx_nt_all)], dim=2)
                        v_b_world = torch.cat([v_txt_b, v_p.index_select(2, self.idx_nt_all)], dim=2)
                        idx_ab, out_ab, idx_ar, out_ar = self._binaural_audio_outputs(
                            q_p, k_p, v_p, k_b_world, v_b_world, l_txt_b, scale, shielded=False
                        )
                        _write_iso(idx_ab, out_ab)
                        _write_iso(idx_ar, out_ar)
                        del k_b_world, v_b_world, out_ab, out_ar
                    else:
                        q_aud = q_p.index_select(2, self.idx_aud)
                        if route_aud_region:
                            k_aud_kv = torch.cat([k_txt_b, k_p.index_select(2, self.idx_nt_all)], dim=2)
                            v_aud_kv = torch.cat([v_txt_b, v_p.index_select(2, self.idx_nt_all)], dim=2)
                            out_aud = _fast_attn(q_aud, k_aud_kv, v_aud_kv, scale=scale)
                            del k_aud_kv, v_aud_kv
                        else:
                            out_aud = _fast_attn(q_aud, k_p, v_p, scale=scale)
                        _write_iso(self.idx_aud, out_aud)
                        del q_aud, out_aud

                idx_misc = other_mask.nonzero(as_tuple=False).flatten()
                if idx_misc.numel() > 0:
                    q_misc = q_p.index_select(2, idx_misc)
                    out_misc = _fast_attn(q_misc, k_p, v_p, scale=scale)
                    _write_iso(idx_misc, out_misc)
                    del q_misc, out_misc
                del (
                    q_txt_a,
                    k_txt_a,
                    v_txt_a,
                    q_txt_b,
                    k_txt_b,
                    v_txt_b,
                    out_txt_a,
                    out_vid_a,
                    out_vid_b,
                    other_mask,
                    idx_misc,
                )

            # Back in ACTIVE `malloc_graph`: release `q_in`/`k_in`/`v_in` buffers in their owning allocator!
            del q_p, k_p, v_p
            if not torch.is_tensor(q_in):
                q_in.take()
            if not torch.is_tensor(k_in):
                k_in.take()
            if not torch.is_tensor(v_in):
                v_in.take()
            return out_ret

        if cur.get("warmup", False):
            with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
                k_dtype, v_dtype = k_p.dtype, v_p.dtype
                q_txt_b = cur["q_txt_b"].to(dtype=k_dtype)
                k_txt_b = cur["k_txt_b"].to(dtype=k_dtype)
                v_txt_b = cur["v_txt_b"].to(dtype=v_dtype)
                cur["q_txt_b"] = cur["k_txt_b"] = cur["v_txt_b"] = None
                if need_finish_b:
                    k_b_world = torch.cat([k_txt_b, k_p.index_select(2, self.idx_nt_all)], dim=2)
                    v_b_world = torch.cat([v_txt_b, v_p.index_select(2, self.idx_nt_all)], dim=2)
                    out_txt_b = _fast_attn(q_txt_b, k_b_world, v_b_world, scale=scale)
                    cur["shadow_b_attn_2d"] = (
                        out_txt_b[0].transpose(0, 1).reshape(l_txt_b, inner).to(dtype=self.h_shadow_b.dtype)
                    )
                    del k_b_world, v_b_world, out_txt_b
                del q_txt_b, k_txt_b, v_txt_b
            del q_p, k_p, v_p
            return pass_through_fn()

        # Seamless blend phase:
        # 1) Pause `malloc_graph` to compute the region/audio splice + shadow-B text update from peeked Q/K/V.
        reg_idx = self.idx_region_soft
        use_shield = self.shield_region and (self.idx_region_soft_kv is not None) and (self.idx_region_soft_kv.numel() > 0)
        b_world_nt_idx = self.idx_region_soft_kv if use_shield else self.idx_nt_all
        assert b_world_nt_idx is not None
        follow_mask_active = (self.audio_zone == "follow_mask") and (self.idx_aud.numel() > 0)
        with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
            k_dtype, v_dtype = k_p.dtype, v_p.dtype
            q_txt_b = cur["q_txt_b"].to(dtype=k_dtype)
            k_txt_b = cur["k_txt_b"].to(dtype=k_dtype)
            v_txt_b = cur["v_txt_b"].to(dtype=v_dtype)
            cur["q_txt_b"] = cur["k_txt_b"] = cur["v_txt_b"] = None

            q_parts: list[torch.Tensor] = []
            if need_finish_b:
                q_parts.append(q_txt_b)
            n_reg = int(reg_idx.numel())
            if n_reg > 0:
                q_parts.append(q_p.index_select(2, reg_idx))
            n_aud = int(self.idx_aud.numel()) if route_aud_region else 0
            if route_aud_region:
                q_parts.append(q_p.index_select(2, self.idx_aud))

            k_b_world = torch.cat([k_txt_b, k_p.index_select(2, b_world_nt_idx)], dim=2)
            v_b_world = torch.cat([v_txt_b, v_p.index_select(2, b_world_nt_idx)], dim=2)

            out_vid_b = None
            out_aud_b = None
            if q_parts:
                q_comb_b = q_parts[0] if len(q_parts) == 1 else torch.cat(q_parts, dim=2)
                out_b_all = _fast_attn(q_comb_b, k_b_world, v_b_world, scale=scale)
                del q_comb_b
                off = 0
                if need_finish_b:
                    cur["shadow_b_attn_2d"] = (
                        out_b_all[:, :, :l_txt_b, :][0]
                        .transpose(0, 1)
                        .reshape(l_txt_b, inner)
                        .to(dtype=self.h_shadow_b.dtype)
                    )
                    off += l_txt_b
                out_vid_b = out_b_all[:, :, off : off + n_reg, :].clone() if n_reg > 0 else None
                off += n_reg
                out_aud_b = out_b_all[:, :, off : off + n_aud, :].clone() if route_aud_region else None
                del out_b_all

            if follow_mask_active:
                fm_res = self._follow_mask_audio_outputs(
                    q_p, k_p, v_p, k_b_world, v_b_world, l_txt_b,
                    cur["block_index"], heads, head_dim, scale, shielded=use_shield,
                )
                bin_res = None
            elif binaural_active:
                fm_res = None
                bin_res = self._binaural_audio_outputs(
                    q_p, k_p, v_p, k_b_world, v_b_world, l_txt_b, scale, shielded=use_shield
                )
            else:
                fm_res = None
                bin_res = None
            del k_b_world, v_b_world, q_parts, q_txt_b, k_txt_b, v_txt_b

        # 2) Back in ACTIVE `malloc_graph`: drop peek refs and let Sol/Kitchen consume Q/K/V and allocate `out_main`.
        del q_p, k_p, v_p
        out_main = pass_through_fn()

        # 3) Pause `malloc_graph` to splice pause-allocated buffers in-place into `out_main` and free them cleanly.
        with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
            def _splice(idx: torch.Tensor, src_bh: torch.Tensor | None) -> None:
                if src_bh is None or idx.numel() == 0:
                    return
                if skip_output_reshape:
                    out_main.index_copy_(2, idx, src_bh.to(dtype=out_main.dtype))
                else:
                    src_2d = (
                        src_bh[0].transpose(0, 1).reshape(1, int(idx.numel()), inner).to(dtype=out_main.dtype)
                    )
                    out_main.index_copy_(1, idx, src_2d)

            if out_vid_b is not None and reg_idx.numel() > 0:
                if self.w_region_soft is None:
                    _splice(reg_idx, out_vid_b)
                elif skip_output_reshape:
                    cur_main = out_main.index_select(2, reg_idx).float()
                    w = self.w_region_soft.view(1, 1, -1, 1)
                    blended = (1.0 - w) * cur_main + w * out_vid_b.float()
                    out_main.index_copy_(2, reg_idx, blended.to(dtype=out_main.dtype))
                    del cur_main, blended
                else:
                    cur_main = out_main.index_select(1, reg_idx).float()
                    src_2d = out_vid_b[0].transpose(0, 1).reshape(1, int(reg_idx.numel()), inner).float()
                    w = self.w_region_soft.view(1, -1, 1)
                    blended = (1.0 - w) * cur_main + w * src_2d
                    out_main.index_copy_(1, reg_idx, blended.to(dtype=out_main.dtype))
                    del cur_main, src_2d, blended

            if fm_res is not None:
                _splice(self.idx_aud, fm_res)
                del fm_res
            elif bin_res is not None:
                _splice(bin_res[0], bin_res[1])
                _splice(bin_res[2], bin_res[3])
                del bin_res
            elif route_aud_region:
                _splice(self.idx_aud, out_aud_b)
            del out_vid_b, out_aud_b

        return out_main

    def finish_block(self, block_index: int) -> None:
        cur = self._cur
        self._cur = None
        if not self.active or cur is None or self.h_shadow_b is None:
            return
        if cur["block_index"] != block_index:
            return
        if not cur.get("done", False):
            raise RuntimeError(
                f"MiniMaxH3SpatialQueryRouter attention override was bypassed on block {block_index} "
                "(e.g. downstream MiniMaxH3SageAttention or a custom attn.forward patch skipped "
                "optimized_attention_override). Use ModelAttentionBackend('comfy kitchen attention') "
                "with optional MiniMaxH3Sol instead."
            )
        if block_index >= (self.n_blocks - 1):
            return
        if cur["shadow_b_attn_2d"] is None:
            return
        from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift

        block = cur["block"]
        mod_txt_b = cur["mod_txt_b"]
        attn_out_b = block.attn.out_proj(cur["shadow_b_attn_2d"])
        cur["shadow_b_attn_2d"] = None
        _mod_gate(self.h_shadow_b, cur["gate_msa"], attn_out_b, mod_txt_b)
        del attn_out_b
        h_mlp_b = _mod_scale_shift(
            block.norm2(self.h_shadow_b), cur["shift_mlp"], cur["scale_mlp"], mod_txt_b
        )
        _mod_gate(self.h_shadow_b, cur["gate_mlp"], block.mlp(h_mlp_b), mod_txt_b)
        del h_mlp_b


class _SpatialBlockPatch:
    """Stateless per-clone `patches_replace['dit'][('double_block', i)]` adapter."""

    def __init__(self, block_index: int, previous: Any = None) -> None:
        self.block_index = int(block_index)
        self.previous = previous

    def __call__(self, args: dict[str, Any], extra_args: dict[str, Any]) -> dict[str, Any]:
        to = args.get("transformer_options")
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


def _make_spatial_attn_override(previous: Any = None):
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

        def _pass():
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

        rt = options.get("_h3_spatial_qr_rt") if isinstance(options, dict) else None
        if rt is None or not getattr(rt, "active", False) or not skip_reshape or mask is not None:
            return _pass()
        cur = getattr(rt, "_cur", None)
        if cur is None or cur.get("done"):
            return _pass()
        b_idx = options.get("block_index") if isinstance(options, dict) else None
        if b_idx is not None and int(b_idx) != int(cur["block_index"]):
            return _pass()

        q_p = q if torch.is_tensor(q) else q.peek()
        k_p = k if torch.is_tensor(k) else k.peek()
        v_p = v if torch.is_tensor(v) else v.peek()
        if q_p.ndim != 4 or k_p.ndim != 4 or v_p.ndim != 4:
            return _pass()
        del q_p, k_p, v_p

        return rt.run_attention_override(
            q,
            k,
            v,
            _pass,
            skip_output_reshape=bool(skip_output_reshape),
            scale=kwargs.get("scale", None),
        )

    if getattr(previous, "container_function", None) is not None:
        override.container_function = functools.partial(override, None)
    return override


def _has_spatial_block_patch(patch: Any) -> bool:
    """Walk `.previous` chain to detect an already-installed `_SpatialBlockPatch`."""
    cur = patch
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, _SpatialBlockPatch):
            return True
        seen.add(id(cur))
        cur = getattr(cur, "previous", None)
    return False


def attach_spatial_query_router(
    model: Any,
    *,
    region_conditioning: Any,
    x_start: float = 0.5,
    x_end: float = 1.0,
    y_start: float = 0.0,
    y_end: float = 1.0,
    invert: bool = False,
    isolation: str = "early_0_24",
    audio_zone: str = "base",
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    mask: torch.Tensor | None = None,
) -> Any:
    """Clone `model` and attach `1x` Spatial Visual Query Routing hooks."""
    iso_mode = str(isolation or "early_0_24").strip()
    if iso_mode not in ISOLATION_MODES:
        raise ValueError(f"isolation must be one of {ISOLATION_MODES}, got {isolation!r}")
    aud_z = str(audio_zone or "base").strip()
    if aud_z not in AUDIO_ZONES:
        raise ValueError(f"audio_zone must be one of {AUDIO_ZONES}, got {audio_zone!r}")

    s_pct = float(start_percent)
    e_pct = float(end_percent)
    if not (math.isfinite(s_pct) and math.isfinite(e_pct)) or s_pct < 0.0 or e_pct > 1.0:
        raise ValueError(f"start_percent/end_percent must lie in [0.0, 1.0], got [{s_pct}, {e_pct}]")
    if s_pct >= e_pct:
        return model

    for nm, val in (("x_start", x_start), ("x_end", x_end), ("y_start", y_start), ("y_end", y_end)):
        fv = float(val)
        if not math.isfinite(fv) or fv < 0.0 or fv > 1.0:
            raise ValueError(f"{nm} must lie in [0.0, 1.0], got {val!r}")

    raw_b = _extract_cond_tensor(region_conditioning, name="region_conditioning")
    tags_b = _extract_cond_tags(region_conditioning)

    m = model.clone()
    dm = getattr(getattr(m, "model", None), "diffusion_model", None)
    if not _steer_hooks.is_hookable_dit(dm):
        raise ValueError("MiniMaxH3SpatialQueryRouter requires a MiniMax H3 model with hookable DiT blocks")

    from comfy.patcher_extension import CallbacksMP, WrappersMP

    t_opts = m.model_options.setdefault("transformer_options", {})
    dm_wrappers = m.wrappers.get(WrappersMP.DIFFUSION_MODEL, {})
    dit_replaces = t_opts.get("patches_replace", {}).get("dit", {})
    if (
        hasattr(model, "_h3_frozen_video_cache")
        or hasattr(m, "_h3_frozen_video_cache")
        or any("frozen_video_cache" in str(wk) for wk in dm_wrappers)
    ):
        raise ValueError("MiniMaxH3SpatialQueryRouter cannot stack with MiniMaxH3FrozenVideoCache")

    if SPATIAL_ROUTER_WRAPPER_KEY in dm_wrappers or any(
        _has_spatial_block_patch(b_patch) for b_patch in dit_replaces.values()
    ):
        raise ValueError("MiniMaxH3SpatialQueryRouter cannot be stacked twice on the same model")

    for b_patch in dit_replaces.values():
        b_mod = getattr(b_patch, "__module__", "") or ""
        b_qname = getattr(b_patch, "__qualname__", "") or ""
        if "nodes_sparse_attention" in b_mod or "make_h3_block_patch" in b_qname:
            raise ValueError(
                "MiniMaxH3SpatialQueryRouter requires dense optimized_attention or MiniMaxH3Sol, "
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
            f"MiniMaxH3SpatialQueryRouter cannot run with object-patched attention forward ({key}, e.g. "
            "MiniMaxH3SageAttention — even under MiniMaxH3Sol, Step 0 bypasses optimized_attention_override). "
            "Use ModelAttentionBackend('comfy kitchen attention') with optional MiniMaxH3Sol instead."
        )

    iso_set = parse_isolation_blocks(iso_mode, len(dm.blocks))
    shield_region = iso_mode.endswith("_shield")
    for b_idx in range(len(dm.blocks)):
        existing = dit_replaces.get(("double_block", b_idx))
        m.set_model_patch_replace(_SpatialBlockPatch(b_idx, existing), "dit", "double_block", b_idx)

    ref_cache: OrderedDict[tuple[Any, ...], torch.Tensor] = OrderedDict()
    m.add_callback_with_key(
        CallbacksMP.ON_CLEANUP,
        f"{SPATIAL_ROUTER_WRAPPER_KEY}_refcache",
        lambda *args, _rc=ref_cache, **kwargs: _rc.clear(),
    )

    def _wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        if not _steer_hooks.step_in_window(transformer_options, timestep, s_pct, e_pct):
            return executor(x, timestep, context, transformer_options, **kwargs)
        if isinstance(transformer_options, dict) and transformer_options.get("_h3_steer_runtime") is not None:
            raise ValueError(
                "MiniMaxH3SpatialQueryRouter cannot share the same sampler step with an active 1x H3SteerRuntime "
                "(e.g. MiniMaxH3QueryRouter or hook-mode MiniMaxH3Steer); separate their start/end_percent windows "
                "or use MiniMaxH3SpatialQueryRouter's audio_zone input."
            )
        with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
            cache_key = (id(raw_b), str(context.device), str(context.dtype))
            ctx_b = ref_cache.get(cache_key)
            if ctx_b is None:
                ctx_b = _steer_hooks.refine_text_context(
                    dm,
                    raw_b.to(device=context.device, dtype=context.dtype),
                    transformer_options,
                    caller_name="MiniMaxH3SpatialQueryRouter",
                )
                ref_cache[cache_key] = ctx_b
                while len(ref_cache) > _REF_CACHE_MAX:
                    ref_cache.popitem(last=False)
            rt = SpatialVisualQRRuntime(
                dm=dm,
                shadow_b_init=ctx_b,
                shadow_b_tags=tags_b,
                x_start=float(x_start),
                x_end=float(x_end),
                y_start=float(y_start),
                y_end=float(y_end),
                invert=bool(invert),
                mask=mask,
                isolate_blocks=iso_set,
                audio_zone=aud_z,
                shield_region=shield_region,
            )
            to_next = dict(transformer_options or {})
            to_next["_h3_spatial_qr_rt"] = rt
            to_next["optimized_attention_override"] = _make_spatial_attn_override(
                to_next.get("optimized_attention_override")
            )
        try:
            out = executor(x, timestep, context, to_next, **kwargs)
            if rt.active and rt.h_shadow_b is None:
                raise RuntimeError(
                    "MiniMaxH3SpatialQueryRouter block hooks never ran (e.g. downstream core "
                    "BlockSparseAttention overwrote patches_replace['dit'][('double_block', i)]). "
                    "Use ModelAttentionBackend('comfy kitchen attention') with optional MiniMaxH3Sol instead."
                )
            return out
        finally:
            with _steer_hooks.pause_malloc_graph_ctx():
                rt.clear()

    wrappers = m.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
    wrappers[SPATIAL_ROUTER_WRAPPER_KEY] = [_wrapper]
    return m


class MiniMaxH3SpatialQueryRouter(io.ComfyNode):
    """Single-pass (`1x`) Spatial Visual Query Router (split-screen, portal box, or mask)."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3SpatialQueryRouter",
            display_name="MiniMax H3 Spatial Query Router",
            category="MiniMax H3/Steering",
            description=(
                "Route a second visual prompt to any normalized box (left/right, top/bottom, portal) or "
                "optional 2D/video MASK in one 1x pass with 100% stock RoPE and zero VAE shimmer "
                "(supports joint AV H3 video models and single-frame MiniMax-H3-Image)."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input(
                    "region_conditioning",
                    tooltip="Second prompt CONDITIONING (any token length) read by video tokens inside the region.",
                ),
                io.Float.Input(
                    "x_start",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Normalized left edge [0..1] of the routed region (0.5..1.0 = right half; 0.28..0.72 = center portal).",
                ),
                io.Float.Input(
                    "x_end",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Normalized right edge [0..1] of the routed region.",
                ),
                io.Float.Input(
                    "y_start",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Normalized top edge [0..1] of the routed region.",
                ),
                io.Float.Input(
                    "y_end",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Normalized bottom edge [0..1] of the routed region.",
                ),
                io.Boolean.Input(
                    "invert",
                    default=False,
                    tooltip="False: inside box/mask reads region_conditioning. True: outside reads region_conditioning.",
                ),
                io.Combo.Input(
                    "isolation",
                    options=list(ISOLATION_MODES),
                    default="early_0_24",
                    tooltip=(
                        "Symmetric K/V isolation band before seamless blending ('early_0_24' isolates 0..24 "
                        "then blends 25..49; 'mid_20_34' runs shared warmup 0..19, iso 20..34, blend 35..49; "
                        "'mid_20_49' keeps warmup 0..19 then isolates 20..49 for strict scanner beams; "
                        "'*_shield' keeps k_b_world shielded to idx_region_soft on seamless blocks so small "
                        "masks resist bright-surround washout)."
                    ),
                ),
                io.Combo.Input(
                    "audio_zone",
                    options=list(AUDIO_ZONES),
                    default="base",
                    tooltip=(
                        "Which world prompt stereo audio queries read: 'base' (main prompt), 'region' "
                        "(region_conditioning), 'binaural_split' (shared 0..29, late_30_49 visually firewalled "
                        "L/R split matching L/R visual worlds), or 'follow_mask' (temporal audio switching & "
                        "late_30_49 crossfade following per-frame mask coverage)."
                    ),
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
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Optional 2D [H, W] or video [F, H, W] MASK (frame count must be 1 or match render F); "
                        "isolates at >=0.5 on early iso blocks and feathers smoothly with continuous [0..1] weights "
                        "across seamless blocks 25..49. Overrides x/y box when connected."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        region_conditioning: Any,
        x_start: float = 0.5,
        x_end: float = 1.0,
        y_start: float = 0.0,
        y_end: float = 1.0,
        invert: bool = False,
        isolation: str = "early_0_24",
        audio_zone: str = "base",
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        mask: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            attach_spatial_query_router(
                model,
                region_conditioning=region_conditioning,
                x_start=x_start,
                x_end=x_end,
                y_start=y_start,
                y_end=y_end,
                invert=invert,
                isolation=isolation,
                audio_zone=audio_zone,
                start_percent=start_percent,
                end_percent=end_percent,
                mask=mask,
            )
        )
