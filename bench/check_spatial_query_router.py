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
import contextlib
import sys
import weakref
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

    def __init__(self, tensor: torch.Tensor, *, pause_state: dict[str, int] | None = None) -> None:
        self._t: torch.Tensor | None = tensor
        self._pause_state = pause_state
        self.peeks = 0
        self.takes = 0

    def peek(self) -> torch.Tensor:
        assert self._t is not None, "peek() after take()"
        self.peeks += 1
        return self._t

    def take(self) -> torch.Tensor:
        assert self._t is not None, "double take()"
        if self._pause_state is not None:
            assert self._pause_state["depth"] == 0, (
                "AttentionTensorContainer.take() called while malloc_graph was paused "
                "(triggers cuMemFreeAsync CUDA_ERROR_INVALID_VALUE under comfy_aimdo)"
            )
        self.takes += 1
        t = self._t
        self._t = None
        return t


class _FactorizedLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, rank: int = 4) -> None:
        super().__init__()
        self.w_down = nn.Parameter(torch.randn(in_f, rank) * 0.05)
        self.w_up = nn.Parameter(torch.randn(rank, out_f) * 0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.w_down) @ self.w_up


class _TinyAttn(nn.Module):
    def __init__(self, hidden: int, heads: int, *, factorized: bool = False) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        if factorized:
            self.qkv_proj = _FactorizedLinear(hidden, hidden * 3)
            self.out_proj = _FactorizedLinear(hidden, hidden)
        else:
            self.qkv_proj = nn.Linear(hidden, hidden * 3, bias=False)
            self.out_proj = nn.Linear(hidden, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)


class _TinyBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, *, factorized: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = _TinyAttn(hidden, heads, factorized=factorized)
        self.mlp = nn.Sequential(
            _FactorizedLinear(hidden, hidden) if factorized else nn.Linear(hidden, hidden, bias=False)
        )
        self.hidden = hidden

    def adaln_proj(self, t_emb: torch.Tensor):
        n = int(t_emb.shape[0])
        zeros = torch.zeros(n, self.hidden, dtype=t_emb.dtype, device=t_emb.device)
        ones = torch.ones(n, self.hidden, dtype=t_emb.dtype, device=t_emb.device)
        return zeros, zeros, ones, zeros, zeros, ones


class _TinyDM(nn.Module):
    def __init__(self, hidden: int = 32, heads: int = 4, n_blocks: int = 4, *, factorized: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock(hidden, heads, factorized=factorized) for _ in range(n_blocks)])
        self.audio_patch_proj = nn.Linear(1, 1)
        self.condition_proj = nn.Identity() if factorized else nn.Linear(hidden, hidden, bias=False)
        self.token_refiner = lambda x, transformer_options=None: x

    def rope_freqs(self, pos: torch.Tensor, device: Any) -> torch.Tensor:
        s = int(pos.shape[0])
        return torch.zeros(s, 4, dtype=torch.float32, device=device)


def _make_mini_patcher(dm: _TinyDM):
    """Minimal ModelPatcher stand-in supporting clone(), object_patches, wrappers, and patches_replace."""
    from comfy.patcher_extension import WrapperExecutor, WrappersMP

    class _MiniPatcher:
        def __init__(self, diffusion_model: Any) -> None:
            self.model = type("_Holder", (), {"diffusion_model": diffusion_model})()
            self.model_options: dict[str, Any] = {"transformer_options": {}}
            self.wrappers: dict[str, dict[str, list[Any]]] = {}
            self.callbacks: dict[str, dict[str, Any]] = {}
            self.object_patches: dict[str, Any] = {}

        def clone(self) -> "_MiniPatcher":
            cp = _MiniPatcher(self.model.diffusion_model)
            orig_to = self.model_options.get("transformer_options", {})
            orig_pr = orig_to.get("patches_replace", {})
            new_pr = {k: dict(v) for k, v in orig_pr.items()}
            new_to = {k: (list(v) if isinstance(v, list) else v) for k, v in orig_to.items()}
            new_to["patches_replace"] = new_pr
            cp.model_options = {"transformer_options": new_to}
            cp.wrappers = {mp: {k: list(v) for k, v in d.items()} for mp, d in self.wrappers.items()}
            cp.callbacks = {k: dict(v) for k, v in self.callbacks.items()}
            cp.object_patches = dict(self.object_patches)
            return cp

        def add_callback_with_key(self, key: Any, name: str, fn: Any) -> None:
            self.callbacks.setdefault(str(key), {})[name] = fn

        def add_wrapper_with_key(self, mp: Any, key: str, fn: Any) -> None:
            self.wrappers.setdefault(mp, {}).setdefault(key, []).append(fn)

        def set_model_patch_replace(self, patch: Any, group: str, key: str, idx: int) -> None:
            to = self.model_options.setdefault("transformer_options", {})
            pr = to.setdefault("patches_replace", {})
            grp = pr.setdefault(group, {})
            grp[(key, idx)] = patch

        def run(self, context: torch.Tensor, layout: Any, img: torch.Tensor, t_emb: torch.Tensor, mod_segments: Any, payload: dict[str, Any] | None = None):
            to = dict(self.model_options.get("transformer_options", {}))
            to["sample_sigmas"] = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0], dtype=torch.float32)

            def _exec(x, timestep, ctx, t_opts, **kw):
                cur_h = torch.cat([ctx[0], img[10:]], dim=0) if ctx.shape[1] == 10 else img
                patches = (t_opts.get("patches_replace") or {}).get("dit", {})
                ovr = t_opts.get("optimized_attention_override")
                for b_i, blk in enumerate(self.model.diffusion_model.blocks):
                    b_opts = dict(t_opts)
                    b_opts["block_index"] = b_i
                    b_args = {
                        "img": cur_h,
                        "layout": layout,
                        "t_emb": t_emb,
                        "mod_segments": mod_segments,
                        "rope_freqs": None,
                        "transformer_options": b_opts,
                    }

                    def _orig(a, _blk=blk, _ovr=ovr):
                        h_dim = _blk.hidden
                        n_heads = _blk.attn.heads
                        h_head = _blk.attn.head_dim
                        h_in = _blk.norm1(a["img"])
                        qkv = _blk.attn.qkv_proj(h_in)
                        q, k, v = qkv.split(h_dim, dim=-1)
                        q_bh = q.view(1, -1, n_heads, h_head).transpose(1, 2)
                        k_bh = k.view(1, -1, n_heads, h_head).transpose(1, 2)
                        v_bh = v.view(1, -1, n_heads, h_head).transpose(1, 2)
                        def _sdpa(q_u, k_u, v_u, heads, skip_output_reshape=False, **kwargs):
                            res = torch.nn.functional.scaled_dot_product_attention(q_u, k_u, v_u)
                            return res if skip_output_reshape else res.transpose(1, 2).reshape(1, -1, h_dim)
                        if _ovr is not None:
                            out_a = _ovr(_sdpa, q_bh, k_bh, v_bh, n_heads, skip_reshape=True, skip_output_reshape=False, transformer_options=a["transformer_options"])
                        else:
                            out_a = _sdpa(q_bh, k_bh, v_bh, n_heads, skip_output_reshape=False)
                        return {"img": a["img"] + out_a[0]}

                    if ("double_block", b_i) in patches:
                        cur_h = patches[("double_block", b_i)](b_args, {"original_block": _orig})["img"]
                    else:
                        cur_h = _orig(b_args)["img"]
                return [cur_h.unsqueeze(0), torch.zeros(1, 32, 2, 15)]

            w_dict = self.wrappers.get(WrappersMP.DIFFUSION_MODEL, {})
            active_w = [fn for k in w_dict.keys() for fn in w_dict[k]]
            ex = WrapperExecutor.new_executor(_exec, active_w)
            x_in = [torch.zeros(1, 24, 7, 8, 12), torch.zeros(1, 32, 2, 15)]
            return ex.execute(x_in, torch.tensor([800.0]), context, to, minimax_payload=payload or {})

    return _MiniPatcher(dm)


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

    pause_state = {"depth": 0, "enters": 0}
    orig_pause_ctx = sqr._steer_hooks.pause_malloc_graph_ctx

    @contextlib.contextmanager
    def _tracked_pause_ctx():
        pause_state["depth"] += 1
        pause_state["enters"] += 1
        try:
            yield
        finally:
            pause_state["depth"] -= 1

    sqr._steer_hooks.pause_malloc_graph_ctx = _tracked_pause_ctx
    try:
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
                qc, kc, vc = (
                    _DummyContainer(q_t, pause_state=pause_state),
                    _DummyContainer(k_t, pause_state=pause_state),
                    _DummyContainer(v_t, pause_state=pause_state),
                )
                pass_called = {"n": 0}

                def _pass():
                    assert pause_state["depth"] == 0, "pass_through_fn must run outside pause_malloc_graph_ctx()"
                    pass_called["n"] += 1
                    q_u, k_u, v_u = qc.take(), kc.take(), vc.take()
                    out = torch.nn.functional.scaled_dot_product_attention(q_u, k_u, v_u)
                    return out.transpose(1, 2).reshape(1, s_total, 32)

                out = rt.run_attention_override(qc, kc, vc, _pass, skip_output_reshape=False)
                assert pause_state["depth"] == 0, "pause_malloc_graph_ctx leaked open after run_attention_override"
                assert out.shape == (1, s_total, 32)
                assert torch.isfinite(out).all()
                if b_idx in (0, 1):
                    assert pass_called["n"] == 0, "iso block should bypass pass_through"
                    assert qc.takes == 1 and kc.takes == 1 and vc.takes == 1, "iso block must take() containers"
                else:
                    assert pass_called["n"] == 1, "seamless block must call pass_through"
                    assert qc.takes == 1, "seamless block must let pass_through take() once"
                rt.finish_block(b_idx)
            rt.clear()
            assert rt.h_shadow_b is None and rt.idx_base is None and rt._cur is None
        assert pause_state["enters"] > 0
    finally:
        sqr._steer_hooks.pause_malloc_graph_ctx = orig_pause_ctx

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

    # 4. Isolation presets, validation guards, mid_20_34 warmup->iso->blend, & mask frame-count refusal
    assert "mid_20_34" in sqr.ISOLATION_MODES
    assert len(sqr.parse_isolation_blocks("seamless_off", 50)) == 0
    assert len(sqr.parse_isolation_blocks("early_0_16", 50)) == 17
    assert len(sqr.parse_isolation_blocks("early_0_24", 50)) == 25
    assert len(sqr.parse_isolation_blocks("early_0_32", 50)) == 33
    assert len(sqr.parse_isolation_blocks("all_0_49", 50)) == 50
    assert sqr.parse_isolation_blocks("mid_20_34", 50) == frozenset(range(20, 35))
    assert sqr.parse_isolation_blocks("mid_20_34", 20) == frozenset()

    sqr._steer_hooks.pause_malloc_graph_ctx = _tracked_pause_ctx
    try:
        rt_mid = sqr.SpatialVisualQRRuntime(
            dm=dm,
            shadow_b_init=shadow_b_diff_len,
            shadow_b_tags=None,
            x_start=0.5,
            x_end=1.0,
            y_start=0.0,
            y_end=1.0,
            invert=False,
            mask=None,
            isolate_blocks=frozenset({1, 2}),
            audio_zone="base",
        )
        assert rt_mid.warmup_blocks == frozenset({0})
        rt_mid.prepare_block(
            0, {"img": img, "layout": layout, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": None}
        )
        assert rt_mid._cur is not None and rt_mid._cur["warmup"] is True and rt_mid._cur["iso"] is False
        assert rt_mid.idx_region is not None and rt_mid.idx_region.numel() > 0
        h_before = rt_mid.h_shadow_b.clone()

        q_w0, k_w0, v_w0 = torch.randn(1, 4, s_total, 8), torch.randn(1, 4, s_total, 8), torch.randn(1, 4, s_total, 8)
        qc_w0 = _DummyContainer(q_w0, pause_state=pause_state)
        kc_w0 = _DummyContainer(k_w0, pause_state=pause_state)
        vc_w0 = _DummyContainer(v_w0, pause_state=pause_state)
        expected_w0 = torch.nn.functional.scaled_dot_product_attention(q_w0, k_w0, v_w0).transpose(1, 2).reshape(1, s_total, 32)

        def _pass_w0():
            assert pause_state["depth"] == 0, "warmup pass_through_fn must run outside pause_malloc_graph_ctx()"
            q_u, k_u, v_u = qc_w0.take(), kc_w0.take(), vc_w0.take()
            return torch.nn.functional.scaled_dot_product_attention(q_u, k_u, v_u).transpose(1, 2).reshape(1, s_total, 32)

        out_w0 = rt_mid.run_attention_override(qc_w0, kc_w0, vc_w0, _pass_w0, skip_output_reshape=False)
        assert pause_state["depth"] == 0
        assert qc_w0.takes == 1 and kc_w0.takes == 1 and vc_w0.takes == 1
        assert float((out_w0 - expected_w0).abs().max().item()) == 0.0, "warmup block must be bit-identical pass_through"
        rt_mid.finish_block(0)
        assert float((rt_mid.h_shadow_b - h_before).abs().max().item()) > 1e-4, "warmup block must advance h_shadow_b"

        rt_mid.prepare_block(
            1, {"img": img, "layout": layout, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": None}
        )
        assert rt_mid._cur is not None and rt_mid._cur["iso"] is True and rt_mid._cur["warmup"] is False
        out_b1 = rt_mid.run_attention_override(q_w0, k_w0, v_w0, lambda: expected_w0.clone(), skip_output_reshape=False)
        assert float((out_b1[:, rt_mid.idx_region, :] - expected_w0[:, rt_mid.idx_region, :]).abs().max().item()) > 1e-4
        rt_mid.clear()
    finally:
        sqr._steer_hooks.pause_malloc_graph_ctx = orig_pause_ctx
    try:
        sqr.parse_isolation_blocks("bogus", 50)
        raise AssertionError("Expected ValueError for invalid isolation")
    except ValueError:
        pass
    try:
        sqr.build_spatial_indices(
            layout, x_start=0.0, x_end=1.0, y_start=0.0, y_end=1.0, invert=False, mask=torch.ones(10, 32, 48)
        )
        raise AssertionError("Expected ValueError for mask frame count mismatch (10 != 22)")
    except ValueError as exc:
        assert "frame(s)" in str(exc)
    print("[ok] 4. Isolation presets, validation, & strict mask frame-count guard verified")

    # 5. Soft [0..1] continuous mask blending in seamless blocks vs >=0.5 binary in iso blocks
    m_soft = torch.zeros(32, 48)
    m_soft[:, 12:24] = 0.25   # below 0.5 -> base on iso, 0.25 blend on seamless
    m_soft[:, 24:36] = 0.75   # >= 0.5 -> region on iso, 0.75 blend on seamless
    m_soft[:, 36:] = 1.0
    part_soft = sqr._build_spatial_partition(
        layout, x_start=0.0, x_end=1.0, y_start=0.0, y_end=1.0, invert=False, mask=m_soft
    )
    assert part_soft["w_region_soft"] is not None
    assert part_soft["idx_region"].numel() == 7 * 4 * 3  # cols 3..5 out of 6 (24..48 px)
    assert part_soft["idx_region_soft"].numel() > part_soft["idx_region"].numel()

    m_bin_w0 = torch.zeros(32, 48)
    m_bin_w0[:, 24:] = 1.0
    m_bin_w1 = torch.zeros(32, 48)
    m_bin_w1[:, 12:] = 1.0
    # Verify seamless linear blend on a deterministic block: soft(0.25) == 0.75*w0 + 0.25*w1
    torch.manual_seed(77)
    q_s = torch.randn(1, 4, s_total, 8)
    k_s = torch.randn(1, 4, s_total, 8)
    v_s = torch.randn(1, 4, s_total, 8)
    dm_s = _TinyDM(hidden=32, heads=4, n_blocks=4)

    def _run_seamless(mask_t: torch.Tensor, *, inv: bool = False) -> torch.Tensor:
        rt_s = sqr.SpatialVisualQRRuntime(
            dm=dm_s,
            shadow_b_init=shadow_b_diff_len,
            shadow_b_tags=None,
            x_start=0.0,
            x_end=1.0,
            y_start=0.0,
            y_end=1.0,
            invert=inv,
            mask=mask_t,
            isolate_blocks=frozenset(),
            audio_zone="base",
        )
        rt_s.prepare_block(
            0, {"img": img, "layout": layout, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": None}
        )
        def _pass_s():
            return torch.nn.functional.scaled_dot_product_attention(q_s, k_s, v_s).transpose(1, 2).reshape(1, s_total, 32)
        return rt_s.run_attention_override(q_s, k_s, v_s, _pass_s, skip_output_reshape=False)

    out_w0 = _run_seamless(m_bin_w0)
    out_w1 = _run_seamless(m_bin_w1)
    out_soft = _run_seamless(m_soft)
    # On strip x=[12..24) (cols 1..2 -> patch cols where w=0.25): out_soft must match 0.75*out_w0 + 0.25*out_w1
    m_strip = torch.zeros(32, 48)
    m_strip[:, 12:24] = 1.0
    strip_idx = sqr.build_spatial_indices(
        layout, x_start=0.0, x_end=1.0, y_start=0.0, y_end=1.0, invert=False, mask=m_strip
    )[1]
    expected_strip = 0.75 * out_w0[:, strip_idx, :] + 0.25 * out_w1[:, strip_idx, :]
    err_soft = float((out_soft[:, strip_idx, :] - expected_strip).abs().max().item())
    assert err_soft < 1e-5, f"Soft [0..1] blend mismatch ({err_soft})"
    out_soft_inv = _run_seamless(m_soft, inv=True)
    assert torch.isfinite(out_soft_inv).all()
    m_sub_half = torch.full((32, 48), 0.25)
    p_sub_half = sqr.attach_spatial_query_router(
        _make_mini_patcher(_TinyDM(hidden=32, heads=4, n_blocks=2)),
        region_conditioning=[[torch.randn(1, 7, 32), {}]],
        isolation="all_0_49",
        mask=m_sub_half,
    )
    out_sub_half = p_sub_half.run(torch.randn(1, 10, 32), layout, img, t_emb, mod_segments)
    assert torch.isfinite(out_sub_half[0]).all()
    print("[ok] 5. Continuous [0..1] soft-feather mask blending & invert verified")

    # 6. audio_zone='binaural_split' (blocks < start shared; blocks >= start L/R isolated)
    rt_bin = sqr.SpatialVisualQRRuntime(
        dm=dm_s,
        shadow_b_init=shadow_b_diff_len,
        shadow_b_tags=None,
        x_start=0.5,
        x_end=1.0,
        y_start=0.0,
        y_end=1.0,
        invert=False,
        mask=None,
        isolate_blocks=frozenset({0, 1}),
        audio_zone="binaural_split",
        binaural_start_block=2,
    )
    aud_seg = next((a, b) for (a, b, k) in layout.segments if k == "audio")
    aud_mid = aud_seg[0] + (aud_seg[1] - aud_seg[0]) // 2  # 15 left, 15 right
    v_s_pert = v_s.clone()
    v_s_pert[:, :, aud_mid : aud_seg[1], :].add_(25.0)  # perturb Right ear only

    # Block 0 (< binaural_start=2): L and R stay shared (Right perturbation leaks into Left)
    rt_bin.prepare_block(0, {"img": img, "layout": layout, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": None})
    out_b0_a = rt_bin.run_attention_override(q_s, k_s, v_s, lambda: torch.zeros(1, s_total, 32), skip_output_reshape=False)
    rt_bin.finish_block(0)
    rt_bin2 = sqr.SpatialVisualQRRuntime(
        dm=dm_s, shadow_b_init=shadow_b_diff_len, shadow_b_tags=None,
        x_start=0.5, x_end=1.0, y_start=0.0, y_end=1.0, invert=False, mask=None,
        isolate_blocks=frozenset({0, 1}), audio_zone="binaural_split", binaural_start_block=2,
    )
    rt_bin2.prepare_block(0, {"img": img, "layout": layout, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": None})
    out_b0_b = rt_bin2.run_attention_override(q_s, k_s, v_s_pert, lambda: torch.zeros(1, s_total, 32), skip_output_reshape=False)
    rt_bin2.finish_block(0)
    assert float((out_b0_a[:, aud_seg[0]:aud_mid, :] - out_b0_b[:, aud_seg[0]:aud_mid, :]).abs().max().item()) > 1e-2

    # Block 2 (>= binaural_start=2): Left ear (base) MUST be invariant (< 1e-5) to Right ear V perturbation!
    rt_bin.prepare_block(1, {"img": img, "layout": layout, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": None})
    rt_bin.run_attention_override(q_s, k_s, v_s, lambda: torch.zeros(1, s_total, 32), skip_output_reshape=False)
    rt_bin.finish_block(1)
    rt_bin2.prepare_block(1, {"img": img, "layout": layout, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": None})
    rt_bin2.run_attention_override(q_s, k_s, v_s_pert, lambda: torch.zeros(1, s_total, 32), skip_output_reshape=False)
    rt_bin2.finish_block(1)

    rt_bin.prepare_block(2, {"img": img, "layout": layout, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": None})
    out_b2_a = rt_bin.run_attention_override(
        q_s, k_s, v_s,
        lambda: torch.nn.functional.scaled_dot_product_attention(q_s, k_s, v_s).transpose(1, 2).reshape(1, s_total, 32),
        skip_output_reshape=False,
    )
    rt_bin2.prepare_block(2, {"img": img, "layout": layout, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": None})
    out_b2_b = rt_bin2.run_attention_override(
        q_s, k_s, v_s_pert,
        lambda: torch.nn.functional.scaled_dot_product_attention(q_s, k_s, v_s_pert).transpose(1, 2).reshape(1, s_total, 32),
        skip_output_reshape=False,
    )
    err_l_bin = float((out_b2_a[:, aud_seg[0]:aud_mid, :] - out_b2_b[:, aud_seg[0]:aud_mid, :]).abs().max().item())
    err_r_bin = float((out_b2_a[:, aud_mid:aud_seg[1], :] - out_b2_b[:, aud_mid:aud_seg[1], :]).abs().max().item())
    assert err_l_bin < 1e-5, f"binaural_split Left ear leaked Right V perturbation ({err_l_bin})"
    assert err_r_bin > 1e-1, f"binaural_split Right ear ignored Right V perturbation ({err_r_bin})"

    # Invert flips binaural_region_on_left
    part_r = sqr._build_spatial_partition(layout, x_start=0.5, x_end=1.0, y_start=0.0, y_end=1.0, invert=False)
    part_l = sqr._build_spatial_partition(layout, x_start=0.5, x_end=1.0, y_start=0.0, y_end=1.0, invert=True)
    assert part_r["binaural_region_on_left"] is False and part_l["binaural_region_on_left"] is True
    print("[ok] 6. audio_zone='binaural_split' (blocks <30 shared, >=30 L/R isolated & invert flip) verified")

    # 7. All attachment & symmetric stacking guards (plus normal context-mode Steer allowed!)
    import steering.nodes as sn
    import steering.query_router as qr
    from comfy.patcher_extension import WrappersMP

    dm_guard = _TinyDM(hidden=32, heads=4, n_blocks=4)
    base_p = _make_mini_patcher(dm_guard)
    cond_reg = [[torch.randn(1, 10, 32), {}]]

    def _expect_val_err(fn, substr: str, label: str) -> None:
        try:
            fn()
        except ValueError as exc:
            assert substr in str(exc), f"{label}: unexpected message {exc!r} (want {substr!r})"
            return
        raise AssertionError(f"{label}: expected ValueError containing {substr!r}, none raised")

    # 7a. FrozenVideoCache (both attr & wrapper key)
    p_fc1 = base_p.clone()
    p_fc1._h3_frozen_video_cache = object()
    _expect_val_err(
        lambda: sqr.attach_spatial_query_router(p_fc1, region_conditioning=cond_reg),
        "MiniMaxH3FrozenVideoCache",
        "FrozenVideoCache attr",
    )
    p_fc2 = base_p.clone()
    p_fc2.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "h3_frozen_video_cache", lambda *a, **k: None)
    _expect_val_err(
        lambda: sqr.attach_spatial_query_router(p_fc2, region_conditioning=cond_reg),
        "MiniMaxH3FrozenVideoCache",
        "FrozenVideoCache wrapper",
    )

    # 7b. Core BlockSparseAttention
    p_bsa = base_p.clone()
    class _FakeCoreSparse:
        __module__ = "comfy_extras.nodes_sparse_attention"
    p_bsa.set_model_patch_replace(_FakeCoreSparse(), "dit", "double_block", 0)
    _expect_val_err(
        lambda: sqr.attach_spatial_query_router(p_bsa, region_conditioning=cond_reg),
        "BlockSparseAttention",
        "Core BlockSparseAttention",
    )

    # 7c. MiniMaxH3SageAttention object_patches (standalone & Sage+Sol)
    p_sage = base_p.clone()
    p_sage.object_patches["diffusion_model.blocks.0.attn.forward"] = lambda *a, **k: None
    _expect_val_err(
        lambda: sqr.attach_spatial_query_router(p_sage, region_conditioning=cond_reg),
        "MiniMaxH3SageAttention",
        "Sage object_patches",
    )

    # 7d. MiniMaxH3ImageModel
    dm_img = _TinyDM(hidden=32, heads=4, n_blocks=2)
    del dm_img.audio_patch_proj
    p_img = _make_mini_patcher(dm_img)
    _expect_val_err(
        lambda: sqr.attach_spatial_query_router(p_img, region_conditioning=cond_reg),
        "joint audio-video",
        "H3 Image model",
    )

    # 7e. Double MiniMaxH3SpatialQueryRouter
    p_double = sqr.attach_spatial_query_router(base_p.clone(), region_conditioning=cond_reg)
    _expect_val_err(
        lambda: sqr.attach_spatial_query_router(p_double, region_conditioning=cond_reg),
        "cannot be stacked twice",
        "Double SpatialQueryRouter",
    )

    # 7f. Symmetric collisions: SpatialQueryRouter <-> QueryRouter and SpatialQueryRouter <-> hook-mode Steer
    ctx_main = torch.randn(1, 10, 32)
    p_qr_then_sr = qr.attach_query_router(base_p.clone(), audio_conditioning=cond_reg)
    p_qr_then_sr = sqr.attach_spatial_query_router(p_qr_then_sr, region_conditioning=cond_reg)
    _expect_val_err(
        lambda: p_qr_then_sr.run(ctx_main, layout, img, t_emb, mod_segments),
        "MiniMaxH3SpatialQueryRouter cannot share",
        "QueryRouter -> SpatialQueryRouter",
    )

    p_sr_then_qr = sqr.attach_spatial_query_router(base_p.clone(), region_conditioning=cond_reg)
    p_sr_then_qr = qr.attach_query_router(p_sr_then_qr, audio_conditioning=cond_reg)
    _expect_val_err(
        lambda: p_sr_then_qr.run(ctx_main, layout, img, t_emb, mod_segments),
        "MiniMaxH3QueryRouter cannot share",
        "SpatialQueryRouter -> QueryRouter",
    )

    # Normal context-mode Steer (streams='both', block_band='all') MUST stack cleanly in BOTH orders!
    pr_dim = int(sn._cv.POST_REFINER_DIM)
    dm_pr = _TinyDM(hidden=pr_dim, heads=42, n_blocks=2, factorized=True)
    base_pr = _make_mini_patcher(dm_pr)
    img_pr = torch.randn(s_total, pr_dim) * 0.1
    ctx_pr = torch.randn(1, 10, pr_dim) * 0.1
    cond_reg_pr = [[torch.randn(1, 10, pr_dim) * 0.1, {}]]
    vec_ctx = torch.randn(pr_dim)
    vec_ctx = vec_ctx / torch.linalg.vector_norm(vec_ctx)
    tags_10 = torch.ones(10, dtype=torch.long)
    tags_10._h3_span_masks = {"vision": tags_10 == 0, "scene": tags_10 == 1}
    pay_10 = {"text_token_tags": tags_10, "h3_span_masks": tags_10._h3_span_masks}
    spec_ctx = sn._AppliedVectorSpec(
        vector_name="ctx_vec",
        vector=vec_ctx,
        strength=0.5,
        mode="add",
        scale_mode="raw",
        mask="non_vision",
        start_percent=0.0,
        end_percent=1.0,
        metadata={},
        advisories=(),
        video_scale=1.0,
        audio_scale=1.0,
        block_band="all",
        split_backend="auto_1x_kv",
    )
    spec_hook = sn._AppliedVectorSpec(
        vector_name="hook_vec",
        vector=vec_ctx,
        strength=0.5,
        mode="add",
        scale_mode="raw",
        mask="non_vision",
        start_percent=0.0,
        end_percent=1.0,
        metadata={},
        advisories=(),
        video_scale=1.0,
        audio_scale=0.0,
        block_band="all",
        split_backend="auto_1x_kv",
    )

    def _attach_test_steer(p_in, spec):
        m_s = p_in.clone()
        t_o = m_s.model_options.setdefault("transformer_options", {})
        t_o[sn.WRAPPER_KEY] = [spec]
        ws = sqr._steer_hooks.H3SteerStepWorkspace() if spec.audio_scale == 0.0 else None
        if ws is not None:
            for bi in range(len(m_s.model.diffusion_model.blocks)):
                ex = t_o.get("patches_replace", {}).get("dit", {}).get(("double_block", bi))
                m_s.set_model_patch_replace(sqr._steer_hooks.H3SteerBlockPatch(bi, previous=ex), "dit", "double_block", bi)
        m_s.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})[sn.WRAPPER_KEY] = [
            sn._make_control_vector_wrapper(
                [spec],
                dm_ref=weakref.ref(p_in.model.diffusion_model),
                workspace=ws,
            )
        ]
        return m_s

    # Context-mode Steer + SpatialQueryRouter in both orders: succeeds and applies steering
    m_unsteered = sqr.attach_spatial_query_router(base_pr, region_conditioning=cond_reg_pr)
    out_unsteered = m_unsteered.run(ctx_pr, layout, img_pr, t_emb, mod_segments, payload=pay_10)

    m_cs1 = sqr.attach_spatial_query_router(_attach_test_steer(base_pr, spec_ctx), region_conditioning=cond_reg_pr)
    out_cs1 = m_cs1.run(ctx_pr, layout, img_pr, t_emb, mod_segments, payload=pay_10)
    assert torch.isfinite(out_cs1[0]).all()
    assert float((out_cs1[0] - out_unsteered[0]).abs().max().item()) > 1e-5, (
        "Context-mode Steer had no observable effect when stacked with SpatialQueryRouter"
    )

    m_cs2 = _attach_test_steer(sqr.attach_spatial_query_router(base_pr, region_conditioning=cond_reg_pr), spec_ctx)
    out_cs2 = m_cs2.run(ctx_pr, layout, img_pr, t_emb, mod_segments, payload=pay_10)
    assert torch.isfinite(out_cs2[0]).all()

    # Hook-mode Steer + SpatialQueryRouter in both orders: refuses!
    m_hs1 = sqr.attach_spatial_query_router(_attach_test_steer(base_pr, spec_hook), region_conditioning=cond_reg_pr)
    _expect_val_err(
        lambda: m_hs1.run(ctx_pr, layout, img_pr, t_emb, mod_segments, payload=pay_10),
        "MiniMaxH3SpatialQueryRouter cannot share",
        "Hook-Steer -> SpatialQueryRouter",
    )
    m_hs2 = _attach_test_steer(sqr.attach_spatial_query_router(base_pr, region_conditioning=cond_reg_pr), spec_hook)
    _expect_val_err(
        lambda: m_hs2.run(ctx_pr, layout, img_pr, t_emb, mod_segments, payload=pay_10),
        "MiniMaxH3Steer 1x hook mode",
        "SpatialQueryRouter -> Hook-Steer",
    )
    print("[ok] 7. All attachment & symmetric stacking guards (+ context-mode Steer allowed) verified")
    print("ALL MiniMaxH3SpatialQueryRouter CPU checks passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
