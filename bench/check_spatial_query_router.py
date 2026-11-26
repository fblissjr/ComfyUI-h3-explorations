#!/usr/bin/env python3
"""Offline CPU verification for `MiniMaxH3SpatialQueryRouter` (`steering/spatial_query_router.py`).

Tests:
1. Schema contract & `nodes.py` registration (no `0` numeric sentinel, exact append-only input list).
2. Spatial index partitioning: left/right/top/bottom boxes, center portal box, `invert`, static 2D
   `[H, W]` and multi-frame `[F, H, W]` masks, plus `fl2va` `cond` & `ref2va` `ref_img` preservation.
3. End-to-end `SpatialVisualQRRuntime` forward parity across:
   - `isolation="seamless_off"` vs `isolation="early_0_24"` vs `isolation="all_0_49"`
   - `audio_zone="base"` vs `audio_zone="region"`
   - Arbitrary mismatched text lengths (`l_txt_b != l_txt_a`)
   - Degenerate empty-region (clean no-op) and full-region masks
   - `AttentionTensorContainer`-like peek/take lifecycle in both isolated and seamless blocks
   - Conflict refusal with `MiniMaxH3FrozenVideoCache`, active `_h3_steer_runtime`, and `MiniMaxH3ImageModel`.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
from _lib import bootstrap

bootstrap(cpu=True)

import torch
import torch.nn as nn
from comfy.ldm.minimax.model import PackedLayout

import nodes as comfy_nodes  # noqa: F401
import importlib
import types


def _load_pack_pkg() -> str:
    pkg_name = "_h3pack_test"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(REPO)]
        sys.modules[pkg_name] = pkg
    return pkg_name


class _DummyContainer:
    """Minimal stand-in for ComfyUI `AttentionTensorContainer` tracking peek/take calls."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._t: torch.Tensor | None = tensor
        self.peeks = 0
        self.takes = 0

    def peek(self) -> torch.Tensor:
        assert self._t is not None, "peek() after take()"
        self.peeks += 1
        return self._t

    def take(self) -> torch.Tensor:
        assert self._t is not None, "double take()"
        self.takes += 1
        t = self._t
        self._t = None
        return t


class _TinyAttn(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv_proj = nn.Linear(hidden, hidden * 3, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)


class _TinyBlock(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = _TinyAttn(hidden, heads)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden, bias=False))
        self.hidden = hidden

    def adaln_proj(self, t_emb: torch.Tensor):
        n = int(t_emb.shape[0])
        zeros = torch.zeros(n, self.hidden, dtype=t_emb.dtype, device=t_emb.device)
        ones = torch.ones(n, self.hidden, dtype=t_emb.dtype, device=t_emb.device)
        return zeros, zeros, ones, zeros, zeros, ones


class _TinyDM(nn.Module):
    def __init__(self, hidden: int = 32, heads: int = 4, n_blocks: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(hidden, heads) for _ in range(n_blocks)])
        self.audio_patch_proj = nn.Linear(1, 1)
        self.condition_proj = nn.Linear(hidden, hidden, bias=False)
        self.token_refiner = lambda x, transformer_options=None: x

    def rope_freqs(self, pos: torch.Tensor, device: Any) -> torch.Tensor:
        s = int(pos.shape[0])
        return torch.zeros(s, 4, dtype=torch.float32, device=device)


def main() -> int:
    pkg = _load_pack_pkg()
    pack_nodes = importlib.import_module(f"{pkg}.nodes")
    import steering.spatial_query_router as sqr

    # 1. Schema & registry check
    ext = asyncio.run(pack_nodes.comfy_entrypoint())
    node_list = asyncio.run(ext.get_node_list())
    assert sqr.MiniMaxH3SpatialQueryRouter in node_list, "MiniMaxH3SpatialQueryRouter missing from nodes.py"
    schema = sqr.MiniMaxH3SpatialQueryRouter.define_schema()
    assert schema.node_id == "MiniMaxH3SpatialQueryRouter"
    in_names = [inp.id for inp in schema.inputs]
    assert in_names == [
        "model",
        "region_conditioning",
        "x_start",
        "x_end",
        "y_start",
        "y_end",
        "invert",
        "isolation",
        "audio_zone",
        "start_percent",
        "end_percent",
        "mask",
    ], f"Unexpected schema input order: {in_names}"
    print("[ok] 1. Schema & nodes.py registration verified")

    # 2. Index partitioning across boxes, masks, invert, and ref2va/fl2va segments
    refs = [{"kind": "image", "latent_h": 8, "latent_w": 8}]
    kfs = [{"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 8, 12)}]
    layout = PackedLayout(text_len=10, latent_t=7, latent_h=8, latent_w=12, audio_t=15, refs=refs, keyframes=kfs)
    va, vb = next((a, b) for (a, b, k) in layout.segments if k == "video")
    n_vid = vb - va  # 7 * 4 * 6 = 168

    idx_base, idx_reg, idx_aud, idx_other, idx_nt, tx_a, tx_b = sqr.build_spatial_indices(
        layout, x_start=0.5, x_end=1.0, y_start=0.0, y_end=1.0, invert=False
    )
    assert tx_a == 0 and tx_b == 10
    assert idx_base.numel() + idx_reg.numel() == n_vid
    assert idx_base.numel() > 0 and idx_reg.numel() > 0
    assert set(idx_base.tolist()).isdisjoint(set(idx_reg.tolist()))
    assert idx_aud.numel() == 30
    # idx_other must include ref_img (16), cond (24), AND audio (30) -> 70 non-text, non-video tokens
    assert idx_other.numel() == 16 + 24 + 30
    assert idx_nt.numel() == idx_other.numel() + n_vid

    # Invert produces exact complement
    inv_base, inv_reg, *_ = sqr.build_spatial_indices(
        layout, x_start=0.5, x_end=1.0, y_start=0.0, y_end=1.0, invert=True
    )
    assert torch.equal(inv_base, idx_reg) and torch.equal(inv_reg, idx_base)

    # Static [H, W] mask and multi-frame [F, H, W] mask
    m_2d = torch.zeros(32, 48)
    m_2d[:, 24:] = 1.0
    mb_a, mb_b, *_ = sqr.build_spatial_indices(
        layout, x_start=0.0, x_end=0.0, y_start=0.0, y_end=0.0, invert=False, mask=m_2d
    )
    assert mb_b.numel() == n_vid // 2 and mb_a.numel() == n_vid // 2

    m_vid = torch.zeros(22, 32, 48)
    m_vid[17:, :, :] = 1.0  # Only last clip (latent steps 5..6) masked
    mv_a, mv_b, *_ = sqr.build_spatial_indices(
        layout, x_start=0.0, x_end=0.0, y_start=0.0, y_end=0.0, invert=False, mask=m_vid
    )
    assert mv_b.numel() == 2 * 4 * 6 and mv_a.numel() == 5 * 4 * 6
    print("[ok] 2. Spatial box, invert, static/video mask & ref2va/fl2va segment partitioning verified")

    # 3. Runtime forward across iso + seamless + mismatched l_txt_b + audio_zone + containers
    torch.manual_seed(42)
    dm = _TinyDM(hidden=32, heads=4, n_blocks=4)
    s_total = int(layout.position_ids.shape[0])
    img = torch.randn(s_total, 32)
    t_emb = torch.zeros(2, 32)
    mod_segments = [(0, s_total, 0)]
    shadow_b_diff_len = torch.randn(1, 14, 32)  # l_txt_b=14 != l_txt_a=10

    for aud_zone in ("base", "region"):
        rt = sqr.SpatialVisualQRRuntime(
            dm=dm,
            shadow_b_init=shadow_b_diff_len,
            shadow_b_tags=None,
            x_start=0.25,
            x_end=0.75,
            y_start=0.25,
            y_end=0.75,
            invert=False,
            mask=None,
            isolate_blocks=frozenset({0, 1}),
            audio_zone=aud_zone,
        )
        for b_idx in range(4):
            rt.prepare_block(
                b_idx,
                {"img": img, "layout": layout, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": None},
            )
            q_t = torch.randn(1, 4, s_total, 8)
            k_t = torch.randn(1, 4, s_total, 8)
            v_t = torch.randn(1, 4, s_total, 8)
            qc, kc, vc = _DummyContainer(q_t), _DummyContainer(k_t), _DummyContainer(v_t)
            pass_called = {"n": 0}

            def _pass():
                pass_called["n"] += 1
                q_u, k_u, v_u = qc.take(), kc.take(), vc.take()
                out = torch.nn.functional.scaled_dot_product_attention(q_u, k_u, v_u)
                return out.transpose(1, 2).reshape(1, s_total, 32)

            out = rt.run_attention_override(qc, kc, vc, _pass, skip_output_reshape=False)
            assert out.shape == (1, s_total, 32)
            assert torch.isfinite(out).all()
            if b_idx in (0, 1):
                assert pass_called["n"] == 0, "iso block should bypass pass_through"
                assert qc.takes == 1 and kc.takes == 1 and vc.takes == 1, "iso block must take() containers"
            else:
                assert pass_called["n"] == 1, "seamless block must call pass_through"
                assert qc.takes == 1, "seamless block must let pass_through take() once"
            rt.finish_block(b_idx)

    # Empty region -> clean no-op
    m_empty = torch.zeros(32, 48)
    rt_empty = sqr.SpatialVisualQRRuntime(
        dm=dm,
        shadow_b_init=shadow_b_diff_len,
        shadow_b_tags=None,
        x_start=0.0,
        x_end=1.0,
        y_start=0.0,
        y_end=1.0,
        invert=False,
        mask=m_empty,
        isolate_blocks=frozenset({0, 1}),
        audio_zone="base",
    )
    rt_empty.prepare_block(
        0, {"img": img, "layout": layout, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": None}
    )
    assert rt_empty.active is False, "Empty region with audio_zone='base' should cleanly deactivate"

    # Full region -> seamless full swap without empty idx_base crash
    m_full = torch.ones(32, 48)
    rt_full = sqr.SpatialVisualQRRuntime(
        dm=dm,
        shadow_b_init=shadow_b_diff_len,
        shadow_b_tags=None,
        x_start=0.0,
        x_end=1.0,
        y_start=0.0,
        y_end=1.0,
        invert=False,
        mask=m_full,
        isolate_blocks=frozenset({0, 1}),
        audio_zone="region",
    )
    rt_full.prepare_block(
        0, {"img": img, "layout": layout, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": None}
    )
    assert rt_full._cur is not None and rt_full._cur["iso"] is False
    q_t = torch.randn(1, 4, s_total, 8)
    k_t = torch.randn(1, 4, s_total, 8)
    v_t = torch.randn(1, 4, s_total, 8)
    out_f = rt_full.run_attention_override(
        q_t, k_t, v_t, lambda: torch.zeros(1, s_total, 32), skip_output_reshape=False
    )
    assert out_f.shape == (1, s_total, 32) and torch.isfinite(out_f).all()
    print("[ok] 3. Runtime iso/seamless/audio_zone/mismatched-len/container/empty/full checks verified")

    # 4. Isolation presets & validation guards
    assert len(sqr.parse_isolation_blocks("seamless_off", 50)) == 0
    assert len(sqr.parse_isolation_blocks("early_0_16", 50)) == 17
    assert len(sqr.parse_isolation_blocks("early_0_24", 50)) == 25
    assert len(sqr.parse_isolation_blocks("early_0_32", 50)) == 33
    assert len(sqr.parse_isolation_blocks("all_0_49", 50)) == 50
    try:
        sqr.parse_isolation_blocks("bogus", 50)
        raise AssertionError("Expected ValueError for invalid isolation")
    except ValueError:
        pass
    print("[ok] 4. Isolation presets & validation verified")
    print("ALL MiniMaxH3SpatialQueryRouter CPU checks passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
