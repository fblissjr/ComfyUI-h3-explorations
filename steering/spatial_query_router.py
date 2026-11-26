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
    _is_h3_image_model,
)

logger = logging.getLogger("ComfyUI-h3-explorations.spatial_query_router")

SPATIAL_ROUTER_WRAPPER_KEY = "minimax_h3_spatial_query_router"
ISOLATION_MODES: tuple[str, ...] = (
    "early_0_24",
    "seamless_off",
    "early_0_16",
    "early_0_32",
    "all_0_49",
)
AUDIO_ZONES: tuple[str, ...] = ("base", "region")
_REF_CACHE_MAX = 4
_CK_MIN_Q = 512


def parse_isolation_blocks(isolation: str, n_blocks: int = 50) -> frozenset[int]:
    """Resolve `isolation` combo preset into the set of DiT block indices using symmetric K/V isolation."""
    mode = str(isolation or "early_0_24").strip()
    n_b = max(0, int(n_blocks))
    if mode == "seamless_off":
        return frozenset()
    if mode == "early_0_16":
        return frozenset(range(0, min(17, n_b)))
    if mode == "early_0_24":
        return frozenset(range(0, min(25, n_b)))
    if mode == "early_0_32":
        return frozenset(range(0, min(33, n_b)))
    if mode == "all_0_49":
        return frozenset(range(0, n_b))
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
    """Downsample ComfyUI `MASK` (`[H, W]`, `[1, H, W]`, or `[F, H, W]`) to `[latent_t * patch_h * patch_w]` bool."""
    m = mask.detach().to(device="cpu", dtype=torch.float32)
    if m.ndim == 2:
        m = m.unsqueeze(0)
    elif m.ndim == 4:
        m = m[:, 0] if int(m.shape[1]) == 1 else m[..., 0]
    if m.ndim != 3:
        raise ValueError(f"MiniMaxH3SpatialQueryRouter mask must have 2 or 3 dims, got {tuple(mask.shape)}")

    f_in = int(m.shape[0])
    if f_in == 1:
        m_spatial = F.interpolate(
            m.unsqueeze(1), size=(patch_h, patch_w), mode="area"
        )[0, 0]
        return (m_spatial >= 0.5).unsqueeze(0).expand(latent_t, patch_h, patch_w).reshape(-1)

    from comfy.ldm.minimax.model import FRAME_PER_TOKEN

    m_hw = F.interpolate(m.unsqueeze(1), size=(patch_h, patch_w), mode="area")[:, 0]
    steps: list[torch.Tensor] = []
    f_cur = 0
    for k in range(latent_t):
        span = int(FRAME_PER_TOKEN[k % 5])
        lo = min(f_cur, f_in - 1)
        hi = min(f_cur + span, f_in)
        steps.append(m_hw[lo] if hi <= lo else m_hw[lo:hi].mean(dim=0))
        f_cur += span
    return (torch.stack(steps, dim=0) >= 0.5).reshape(-1)


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
    segs = layout.segments
    pos = layout.position_ids
    tx_spans = [(a, b) for (a, b, k) in segs if k == "text" and b > a]
    vid_spans = [(a, b) for (a, b, k) in segs if k == "video" and b > a]
    if not tx_spans or not vid_spans:
        raise ValueError("MiniMaxH3SpatialQueryRouter requires text and video segments in PackedLayout")
    tx_a, tx_b = tx_spans[0]
    va, vb = vid_spans[0]

    if mask is not None:
        _t_len, latent_t, lat_h, lat_w, _a_t = layout.signature
        patch_h = int(lat_h) // 2
        patch_w = int(lat_w) // 2
        expected = int(latent_t) * patch_h * patch_w
        if expected != (vb - va):
            raise ValueError(
                f"MiniMaxH3SpatialQueryRouter layout patch count mismatch ({expected} != {vb - va})"
            )
        mask_region = _downsample_mask_to_patches(
            mask, latent_t=int(latent_t), patch_h=patch_h, patch_w=patch_w
        )
    else:
        vh = pos[va:vb, 1]
        vw = pos[va:vb, 2]
        h_span = max(float((vh.max() - vh.min()).item()), 1e-6)
        w_span = max(float((vw.max() - vw.min()).item()), 1e-6)
        h_u = (vh - vh.min()) / h_span
        w_u = (vw - vw.min()) / w_span
        x_lo, x_hi = (x_start, x_end) if x_start <= x_end else (x_end, x_start)
        y_lo, y_hi = (y_start, y_end) if y_start <= y_end else (y_end, y_start)
        mask_region = (w_u >= x_lo) & (w_u <= x_hi) & (h_u >= y_lo) & (h_u <= y_hi)

    if invert:
        mask_region = ~mask_region

    vid_all = torch.arange(va, vb, dtype=torch.long)
    idx_region = vid_all[mask_region]
    idx_base = vid_all[~mask_region]

    aud_parts = [torch.arange(a, b, dtype=torch.long) for (a, b, k) in segs if k == "audio" and b > a]
    idx_aud = torch.cat(aud_parts, dim=0) if aud_parts else torch.empty(0, dtype=torch.long)

    # All non-text, non-target-video tokens (cond, cond_audio, ref_img, ref_audio, audio) stay shared in isolated KV.
    other_parts = [
        torch.arange(a, b, dtype=torch.long) for (a, b, k) in segs if k not in ("text", "video") and b > a
    ]
    idx_other = torch.cat(other_parts, dim=0) if other_parts else torch.empty(0, dtype=torch.long)

    non_txt_parts = [torch.arange(a, b, dtype=torch.long) for (a, b, k) in segs if k != "text" and b > a]
    idx_non_txt = torch.cat(non_txt_parts, dim=0) if non_txt_parts else torch.empty(0, dtype=torch.long)
    return idx_base, idx_region, idx_aud, idx_other, idx_non_txt, tx_a, tx_b


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
        self.audio_zone = audio_zone
        self.n_blocks = len(dm.blocks)
        self.active = True

        self.h_shadow_b: torch.Tensor | None = None
        self.rope_b: torch.Tensor | None = None
        self.idx_base: torch.Tensor | None = None
        self.idx_region: torch.Tensor | None = None
        self.idx_aud: torch.Tensor | None = None
        self.idx_base_kv: torch.Tensor | None = None
        self.idx_region_kv: torch.Tensor | None = None
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
            idx_base, idx_region, idx_aud, idx_other, idx_nt, tx_a, tx_b = build_spatial_indices(
                layout,
                x_start=self.x_start,
                x_end=self.x_end,
                y_start=self.y_start,
                y_end=self.y_end,
                invert=self.invert,
                mask=self.mask,
            )
            self.tx_a, self.tx_b = tx_a, tx_b
            self.idx_base = idx_base.to(device=device)
            self.idx_region = idx_region.to(device=device)
            self.idx_aud = idx_aud.to(device=device)
            self.idx_nt_all = idx_nt.to(device=device)
            idx_other_dev = idx_other.to(device=device)
            self.idx_base_kv = torch.cat([idx_other_dev, self.idx_base], dim=0)
            self.idx_region_kv = torch.cat([idx_other_dev, self.idx_region], dim=0)

            if self.idx_region.numel() == 0 and self.audio_zone != "region":
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
            "iso": (
                (block_index in self.isolate_blocks)
                and (self.idx_base.numel() > 0)
                and (self.idx_region.numel() > 0)
            ),
            "done": False,
        }

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
        cur = self._cur
        assert cur is not None and not cur["done"]
        assert self.idx_base is not None and self.idx_region is not None and self.idx_aud is not None
        assert self.idx_base_kv is not None and self.idx_region_kv is not None and self.idx_nt_all is not None
        assert self.h_shadow_b is not None

        cur["done"] = True
        heads, head_dim = cur["heads"], cur["head_dim"]
        inner = heads * head_dim
        l_txt_b = cur["l_txt_b"]
        need_finish_b = cur["block_index"] < (self.n_blocks - 1)
        route_aud_region = (self.audio_zone == "region") and (self.idx_aud.numel() > 0)

        if cur["iso"]:
            # Bypass pass_through_fn -> take containers cleanly so buffers release immediately.
            q_bh = q_in if torch.is_tensor(q_in) else q_in.take()
            k_bh = k_in if torch.is_tensor(k_in) else k_in.take()
            v_bh = v_in if torch.is_tensor(v_in) else v_in.take()
            k_dtype, v_dtype = k_bh.dtype, v_bh.dtype
            q_txt_b = cur["q_txt_b"].to(dtype=k_dtype)
            k_txt_b = cur["k_txt_b"].to(dtype=k_dtype)
            v_txt_b = cur["v_txt_b"].to(dtype=v_dtype)

            q_txt_a = q_bh[:, :, self.tx_a : self.tx_b, :]
            k_txt_a = k_bh[:, :, self.tx_a : self.tx_b, :]
            v_txt_a = v_bh[:, :, self.tx_a : self.tx_b, :]

            q_vid_a = q_bh.index_select(2, self.idx_base)
            k_a_kv = torch.cat([k_txt_a, k_bh.index_select(2, self.idx_base_kv)], dim=2)
            v_a_kv = torch.cat([v_txt_a, v_bh.index_select(2, self.idx_base_kv)], dim=2)
            q_comb_a = torch.cat([q_txt_a, q_vid_a], dim=2)
            out_a_all = _fast_attn(q_comb_a, k_a_kv, v_a_kv, scale=scale)
            l_txt_a = self.tx_b - self.tx_a
            out_txt_a = out_a_all[:, :, :l_txt_a, :]
            out_vid_a = out_a_all[:, :, l_txt_a:, :]
            del q_vid_a, k_a_kv, v_a_kv, q_comb_a, out_a_all

            q_vid_b = q_bh.index_select(2, self.idx_region)
            k_b_kv = torch.cat([k_txt_b, k_bh.index_select(2, self.idx_region_kv)], dim=2)
            v_b_kv = torch.cat([v_txt_b, v_bh.index_select(2, self.idx_region_kv)], dim=2)
            q_comb_b = torch.cat([q_txt_b, q_vid_b], dim=2) if need_finish_b else q_vid_b
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
            del q_vid_b, k_b_kv, v_b_kv, q_comb_b, out_b_all

            s_total = int(q_bh.shape[2])
            out_bh = torch.empty((1, heads, s_total, head_dim), device=q_bh.device, dtype=out_vid_a.dtype)
            out_bh[:, :, self.tx_a : self.tx_b, :].copy_(out_txt_a)
            out_bh.index_copy_(2, self.idx_base, out_vid_a)
            out_bh.index_copy_(2, self.idx_region, out_vid_b)

            other_mask = torch.ones(s_total, dtype=torch.bool, device=q_bh.device)
            other_mask[self.tx_a : self.tx_b] = False
            other_mask[self.idx_base] = False
            other_mask[self.idx_region] = False
            if self.idx_aud.numel() > 0:
                other_mask[self.idx_aud] = False
                q_aud = q_bh.index_select(2, self.idx_aud)
                if route_aud_region:
                    k_aud_kv = torch.cat([k_txt_b, k_bh.index_select(2, self.idx_nt_all)], dim=2)
                    v_aud_kv = torch.cat([v_txt_b, v_bh.index_select(2, self.idx_nt_all)], dim=2)
                    out_aud = _fast_attn(q_aud, k_aud_kv, v_aud_kv, scale=scale)
                    del k_aud_kv, v_aud_kv
                else:
                    out_aud = _fast_attn(q_aud, k_bh, v_bh, scale=scale)
                out_bh.index_copy_(2, self.idx_aud, out_aud)
                del q_aud, out_aud

            idx_misc = other_mask.nonzero(as_tuple=False).flatten()
            if idx_misc.numel() > 0:
                q_misc = q_bh.index_select(2, idx_misc)
                out_misc = _fast_attn(q_misc, k_bh, v_bh, scale=scale)
                out_bh.index_copy_(2, idx_misc, out_misc)
                del q_misc, out_misc
            del out_txt_a, out_vid_a, out_vid_b, q_bh, k_bh, v_bh

            if skip_output_reshape:
                return out_bh
            return out_bh.transpose(1, 2).reshape(1, s_total, inner)

        # Seamless blend phase: peek before calling pass_through_fn so pass_through_fn still owns container take().
        q_bh = q_in if torch.is_tensor(q_in) else q_in.peek()
        k_bh = k_in if torch.is_tensor(k_in) else k_in.peek()
        v_bh = v_in if torch.is_tensor(v_in) else v_in.peek()
        k_dtype, v_dtype = k_bh.dtype, v_bh.dtype
        q_txt_b = cur["q_txt_b"].to(dtype=k_dtype)
        k_txt_b = cur["k_txt_b"].to(dtype=k_dtype)
        v_txt_b = cur["v_txt_b"].to(dtype=v_dtype)

        q_parts: list[torch.Tensor] = []
        if need_finish_b:
            q_parts.append(q_txt_b)
        n_reg = int(self.idx_region.numel())
        if n_reg > 0:
            q_parts.append(q_bh.index_select(2, self.idx_region))
        n_aud = int(self.idx_aud.numel()) if route_aud_region else 0
        if route_aud_region:
            q_parts.append(q_bh.index_select(2, self.idx_aud))

        k_b_world = torch.cat([k_txt_b, k_bh.index_select(2, self.idx_nt_all)], dim=2)
        v_b_world = torch.cat([v_txt_b, v_bh.index_select(2, self.idx_nt_all)], dim=2)
        q_comb_b = q_parts[0] if len(q_parts) == 1 else torch.cat(q_parts, dim=2)
        out_b_all = _fast_attn(q_comb_b, k_b_world, v_b_world, scale=scale)
        del k_b_world, v_b_world, q_comb_b, q_parts, q_bh, k_bh, v_bh

        off = 0
        if need_finish_b:
            cur["shadow_b_attn_2d"] = (
                out_b_all[:, :, :l_txt_b, :][0]
                .transpose(0, 1)
                .reshape(l_txt_b, inner)
                .to(dtype=self.h_shadow_b.dtype)
            )
            off += l_txt_b
        out_vid_b = out_b_all[:, :, off : off + n_reg, :] if n_reg > 0 else None
        off += n_reg
        out_aud_b = out_b_all[:, :, off : off + n_aud, :] if route_aud_region else None
        del out_b_all

        out_main = pass_through_fn()

        def _splice(idx: torch.Tensor, src_bh: torch.Tensor | None) -> None:
            if src_bh is None or idx.numel() == 0:
                return
            if skip_output_reshape:
                out_main.index_copy_(2, idx, src_bh.to(dtype=out_main.dtype))
            else:
                src_2d = src_bh[0].transpose(0, 1).reshape(1, int(idx.numel()), inner).to(dtype=out_main.dtype)
                out_main.index_copy_(1, idx, src_2d)

        _splice(self.idx_region, out_vid_b)
        if route_aud_region:
            _splice(self.idx_aud, out_aud_b)
        del out_vid_b, out_aud_b
        return out_main

    def finish_block(self, block_index: int) -> None:
        cur = self._cur
        self._cur = None
        if not self.active or cur is None or self.h_shadow_b is None:
            return
        if cur["block_index"] != block_index or block_index >= (self.n_blocks - 1):
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

        with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
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
    if _is_h3_image_model(dm):
        raise ValueError("MiniMaxH3SpatialQueryRouter targets joint audio-video MiniMax H3 models")

    t_opts = m.model_options.setdefault("transformer_options", {})
    dit_replaces = t_opts.get("patches_replace", {}).get("dit", {})
    if any(
        getattr(patch, "_h3_frozen_video_cache", False)
        for key, patch in dit_replaces.items()
        if isinstance(key, tuple) and len(key) == 2 and key[0] == "double_block"
    ):
        raise ValueError("MiniMaxH3SpatialQueryRouter cannot stack with MiniMaxH3FrozenVideoCache")

    iso_set = parse_isolation_blocks(iso_mode, len(dm.blocks))
    for b_idx in range(len(dm.blocks)):
        existing = dit_replaces.get(("double_block", b_idx))
        m.set_model_patch_replace(_SpatialBlockPatch(b_idx, existing), "dit", "double_block", b_idx)

    from comfy.patcher_extension import CallbacksMP, WrappersMP

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
            )
            to_next = dict(transformer_options or {})
            to_next["_h3_spatial_qr_rt"] = rt
            to_next["optimized_attention_override"] = _make_spatial_attn_override(
                to_next.get("optimized_attention_override")
            )
        return executor(x, timestep, context, to_next, **kwargs)

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
                "optional 2D/video MASK in one 1x pass with 100% stock RoPE and zero VAE shimmer."
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
                        "Symmetric K/V isolation band before seamless blending ('early_0_24' isolates blocks 0..24 "
                        "and blends seamlessly on 25..49; 'seamless_off' blends across all 50 blocks)."
                    ),
                ),
                io.Combo.Input(
                    "audio_zone",
                    options=list(AUDIO_ZONES),
                    default="base",
                    tooltip="Which world prompt stereo audio queries read ('base' = main prompt, 'region' = region_conditioning).",
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
                    tooltip="Optional 2D [H, W] or video [F, H, W] MASK (>=0.5 = region); overrides x/y box when connected.",
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
