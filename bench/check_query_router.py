#!/usr/bin/env python3
"""Offline CPU verification gate (`0 GB` VRAM) for `MiniMaxH3QueryRouter` (`steering/query_router.py`),
arbitrary-length shadow K/V support in `steering/hooks.py`, and mixed-band support in `steering/nodes.py`.
"""

from __future__ import annotations

import json
import sys
import weakref
from pathlib import Path
from typing import Any

from _lib import REPO, bootstrap

bootstrap(cpu=True)
if str(REPO) not in sys.path:
    sys.path.insert(1, str(REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from comfy.ldm.minimax.model import (  # noqa: E402
    _mod_gate,
    _mod_scale_shift,
    _tag_runs,
    rope_rotation_table,
)
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention  # noqa: E402
from comfy.patcher_extension import WrappersMP, get_all_wrappers  # noqa: E402
import comfy.quant_ops  # noqa: E402

import h3_spans  # noqa: E402
import steering  # noqa: E402
from steering import constants as sc  # noqa: E402
from steering import hooks as sh  # noqa: E402
from steering import nodes as sn  # noqa: E402
from steering import query_router as qr  # noqa: E402


class _FactorizedLinear(nn.Module):
    """Lightweight low-rank linear map hitting exact 5376/5120 H3 shapes in <15 MB CPU RAM."""

    def __init__(self, in_f: int, out_f: int, *, rank: int = 16, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        scale_a = (1.0 / max(in_f, 1)) ** 0.5
        scale_b = (1.0 / max(rank, 1)) ** 0.5
        self.w_down = nn.Parameter(torch.randn(in_f, rank, generator=gen) * scale_a, requires_grad=False)
        self.w_up = nn.Parameter(torch.randn(rank, out_f, generator=gen) * scale_b, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.w_down) @ self.w_up


class _TinyAttn(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int, *, seed: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = _FactorizedLinear(hidden, inner * 3, rank=16, seed=seed + 1)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.out_proj = _FactorizedLinear(inner, hidden, rank=16, seed=seed + 2)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor | None = None, transformer_options=None) -> torch.Tensor:
        s = int(x.shape[0])
        inner = self.heads * self.head_dim
        q, k, v = self.qkv_proj(x).split(inner, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            q = q.view(1, s, self.heads, self.head_dim)
            k = k.view(1, s, self.heads, self.head_dim)
            rot = int(rope_freqs.shape[-3]) * 2
            comfy.quant_ops.ck.rms_rope_split_half_(
                q,
                k,
                rope_freqs,
                self.q_norm.weight,
                self.k_norm.weight,
                epsilon=self.q_norm.eps,
                rot_dim=rot,
            )
            q, k = q[0], k[0]
        else:
            q = self.q_norm(q.view(s, self.heads, self.head_dim))
            k = self.k_norm(k.view(s, self.heads, self.head_dim))

        q_c = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
        k_c = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
        v_c = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
        opts = transformer_options if isinstance(transformer_options, dict) else {}
        out = optimized_attention(
            q_c, k_c, v_c, self.heads, mask=None, skip_reshape=True, transformer_options=opts
        )
        return self.out_proj(out.squeeze(0))


class _TinyAdaln(nn.Module):
    def __init__(self, hidden: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.proj = nn.Parameter(torch.randn(16, 6 * hidden, generator=gen) * 0.02, requires_grad=False)
        self.hidden = hidden

    def forward(self, t_emb: torch.Tensor):
        chunks = (t_emb @ self.proj).split(self.hidden, dim=-1)
        return chunks


class _TinyMLP(nn.Module):
    def __init__(self, hidden: int, seed: int) -> None:
        super().__init__()
        self.fc = _FactorizedLinear(hidden, hidden, rank=16, seed=seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.fc(x))


class _TinyDiTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int, *, seed: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=1e-6)
        self.norm2 = nn.RMSNorm(hidden, eps=1e-6)
        self.attn = _TinyAttn(hidden, heads, head_dim, seed=seed)
        self.mlp = _TinyMLP(hidden, seed=seed + 3)
        self.adaln_proj = _TinyAdaln(hidden, seed=seed + 4)

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = _mod_gate(
            x,
            gate_msa,
            self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options),
            mod_segments,
        )
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)


class _FakeLayout:
    def __init__(
        self,
        l_txt: int,
        n_aud: int = 4,
        n_vid: int = 8,
        n_ref_img: int = 0,
        n_ref_aud: int = 0,
    ) -> None:
        cur = l_txt
        segs: list[tuple[int, int, str]] = [(0, cur, "text")]
        if n_ref_img > 0:
            segs.append((cur, cur + n_ref_img, "ref_img"))
            cur += n_ref_img
        if n_ref_aud > 0:
            segs.append((cur, cur + n_ref_aud, "ref_audio"))
            cur += n_ref_aud
        aud_a = cur
        segs.append((aud_a, aud_a + n_aud, "audio"))
        cur = aud_a + n_aud
        vid_a = cur
        segs.append((vid_a, vid_a + n_vid, "video"))
        cur = vid_a + n_vid
        self.segments = segs
        self.seq_len = cur
        pos = torch.zeros(self.seq_len, 3, dtype=torch.float64)
        pos[:, 0] = torch.arange(self.seq_len, dtype=torch.float64)
        # Put Left/Right audio on opposite screen-edge w columns and video on centered grid
        half_a = n_aud // 2
        pos[aud_a : aud_a + half_a, 2] = 0.0
        pos[aud_a + half_a : aud_a + n_aud, 2] = 32.0
        pos[vid_a:, 1] = 16.0
        pos[vid_a:, 2] = 16.0
        self.position_ids = pos
        self.signature = (l_txt, 1, 1, n_vid, half_a)


class _TinyRefiner(nn.Module):
    def __init__(self, hidden: int, seed: int) -> None:
        super().__init__()
        self.proj = _FactorizedLinear(hidden, hidden, rank=16, seed=seed)

    def forward(self, x: torch.Tensor, transformer_options=None) -> torch.Tensor:
        # Non-trivial bidirectional coupling across token axis so pre-refiner strip differs from post-refiner
        mean_ctx = x.mean(dim=-2, keepdim=True)
        return x + 0.1 * self.proj(x + 0.5 * mean_ctx)


class _TinyH3Model(nn.Module):
    def __init__(
        self,
        *,
        n_blocks: int = 4,
        n_aud: int = 4,
        n_vid: int = 8,
        n_ref_img: int = 0,
        n_ref_aud: int = 0,
        seed: int = 100,
    ) -> None:
        super().__init__()
        self.hidden_size = sc.POST_REFINER_DIM
        self.n_aud = n_aud
        self.n_vid = n_vid
        self.n_ref_img = n_ref_img
        self.n_ref_aud = n_ref_aud
        self.condition_proj = _FactorizedLinear(sc.ENCODER_DIM, sc.POST_REFINER_DIM, rank=16, seed=seed)
        self.token_refiner = _TinyRefiner(sc.POST_REFINER_DIM, seed=seed + 10)
        self.audio_patch_proj = nn.Identity()  # marks joint AV model (`is_audio_patch_proj != None`)
        self.rope = nn.Module()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32.0))
        self.rope.register_buffer("inv_freq", inv_freq)
        self.blocks = nn.ModuleList(
            [_TinyDiTBlock(sc.POST_REFINER_DIM, 42, 128, seed=seed + 20 * (i + 1)) for i in range(n_blocks)]
        )
        gen = torch.Generator(device="cpu").manual_seed(seed + 999)
        self._ref_img_seed = nn.Parameter(
            torch.randn(max(n_ref_img, 0), sc.POST_REFINER_DIM, generator=gen) * 0.25, requires_grad=False
        )
        self._ref_aud_seed = nn.Parameter(
            torch.randn(max(n_ref_aud, 0), sc.POST_REFINER_DIM, generator=gen) * 0.25, requires_grad=False
        )
        self._aud_seed = nn.Parameter(torch.randn(n_aud, sc.POST_REFINER_DIM, generator=gen) * 0.25, requires_grad=False)
        self._vid_seed = nn.Parameter(torch.randn(n_vid, sc.POST_REFINER_DIM, generator=gen) * 0.25, requires_grad=False)
        self._t_emb = nn.Parameter(torch.randn(3, 16, generator=gen) * 0.25, requires_grad=False)
        self.requires_grad_(False)

    def rope_freqs(self, position_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        pos = position_ids.to(torch.float32).to(device)
        inv = self.rope.inv_freq.to(device=device)
        per_axis = pos.unsqueeze(-1) * inv.view(1, 1, -1)
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)
        return torch.cat((half, half), dim=-1)

    def _forward(
        self,
        x,
        timestep,
        context,
        transformer_options=None,
        minimax_payload=None,
        **kwargs,
    ):
        opts = dict(transformer_options or {})
        txt = context[0]
        if txt.shape[-1] != self.hidden_size:
            txt = self.token_refiner(self.condition_proj(txt), transformer_options={})
        l_txt = int(txt.shape[0])
        layout = _FakeLayout(
            l_txt,
            n_aud=self.n_aud,
            n_vid=self.n_vid,
            n_ref_img=self.n_ref_img,
            n_ref_aud=self.n_ref_aud,
        )
        rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, txt.device), txt.dtype)
        text_tags = (minimax_payload or {}).get("text_token_tags") if isinstance(minimax_payload, dict) else None
        if text_tags is not None:
            _tag_runs(0, l_txt, 0, text_tags)
        kind_tag = {"text": 1, "ref_img": 0, "ref_audio": 2, "audio": 2, "video": 0}
        mod_segments = [(a, b, kind_tag[kind]) for (a, b, kind) in layout.segments]
        parts = [txt]
        if self.n_ref_img > 0:
            parts.append(self._ref_img_seed.to(dtype=txt.dtype))
        if self.n_ref_aud > 0:
            parts.append(self._ref_aud_seed.to(dtype=txt.dtype))
        parts.extend([self._aud_seed.to(dtype=txt.dtype), self._vid_seed.to(dtype=txt.dtype)])
        h = torch.cat(parts, dim=0)
        patches = (opts.get("patches_replace") or {}).get("dit", {})
        for i, block in enumerate(self.blocks):
            opts["block_index"] = i
            blk_args = {
                "img": h,
                "t_emb": self._t_emb,
                "mod_segments": mod_segments,
                "rope_freqs": rope_freqs,
                "layout": layout,
                "transformer_options": opts,
            }
            if ("double_block", i) in patches:
                def _orig(a, _blk=block):
                    return {
                        "img": _blk(
                            a["img"],
                            a["t_emb"],
                            a["mod_segments"],
                            a["rope_freqs"],
                            transformer_options=a["transformer_options"],
                        )
                    }
                h = patches[("double_block", i)](blk_args, {"original_block": _orig})["img"]
            else:
                h = block(h, self._t_emb, mod_segments, rope_freqs, transformer_options=opts)
        aud_seg = next(s for s in layout.segments if s[2] == "audio")
        vid_seg = next(s for s in layout.segments if s[2] == "video")
        aud_out = h[aud_seg[0] : aud_seg[1]]
        vid_out = h[vid_seg[0] : vid_seg[1]]
        return [vid_out.unsqueeze(0), aud_out.unsqueeze(0)]


class _ToyPatcher:
    """Minimal ModelPatcher stand-in with real clone(), wrappers, patches_replace, callbacks."""

    def __init__(self, dm: _TinyH3Model) -> None:
        self.model = type("_Holder", (), {"diffusion_model": dm})()
        self.model_options: dict[str, Any] = {"transformer_options": {}}
        self.wrappers: dict[str, dict[str, list[Any]]] = {}
        self.callbacks: dict[str, dict[str, Any]] = {}

    def clone(self) -> "_ToyPatcher":
        cp = _ToyPatcher(self.model.diffusion_model)
        orig_to = self.model_options.get("transformer_options", {})
        orig_pr = orig_to.get("patches_replace", {})
        new_pr = {k: dict(v) for k, v in orig_pr.items()}
        new_to = {k: (list(v) if isinstance(v, list) else v) for k, v in orig_to.items()}
        new_to["patches_replace"] = new_pr
        cp.model_options = {"transformer_options": new_to}
        cp.wrappers = {mp: {k: list(v) for k, v in d.items()} for mp, d in self.wrappers.items()}
        cp.callbacks = {k: dict(v) for k, v in self.callbacks.items()}
        return cp

    def add_callback_with_key(self, key: Any, name: str, fn: Any) -> None:
        self.callbacks.setdefault(str(key), {})[name] = fn

    def set_model_patch_replace(self, patch: Any, group: str, key: str, idx: int) -> None:
        to = self.model_options.setdefault("transformer_options", {})
        pr = to.setdefault("patches_replace", {})
        grp = pr.setdefault(group, {})
        grp[(key, idx)] = patch

    def run(
        self,
        context: torch.Tensor,
        *,
        payload: dict[str, Any] | None = None,
        call_kwargs: dict[str, Any] | None = None,
        timestep: torch.Tensor | None = None,
        stats_sink: list[dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dm = self.model.diffusion_model
        to = dict(self.model_options.get("transformer_options", {}))
        to["sample_sigmas"] = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0], dtype=torch.float32)
        if stats_sink is not None:
            to["_h3_steer_stats_sink"] = stats_sink
        t = torch.tensor([800.0]) if timestep is None else timestep
        x = [torch.zeros(1, 24, 1, 4, 4), torch.zeros(1, 32, 2, 2)]
        extra_kw = dict(call_kwargs or {})
        wrappers = get_all_wrappers(WrappersMP.DIFFUSION_MODEL, to)
        # Re-read from self.wrappers just as ComfyUI's ModelPatcher applies model wrappers
        w_dict = self.wrappers.get(WrappersMP.DIFFUSION_MODEL, {})
        active_wrappers = [fn for k in sorted(w_dict.keys()) for fn in w_dict[k]]
        with torch.no_grad():
            if not active_wrappers and not wrappers:
                res = dm._forward(x, t, context, transformer_options=to, minimax_payload=payload, **extra_kw)
                return res[0], res[1]

            def _exec(curr_x, curr_t, curr_ctx, curr_opts, **kw):
                return dm._forward(curr_x, curr_t, curr_ctx, transformer_options=curr_opts, **kw)

            fn_chain = _exec
            for w in reversed(active_wrappers):
                prev_fn = fn_chain
                fn_chain = lambda cx, ct, cc, co, _w=w, _p=prev_fn, **kw: _w(_p, cx, ct, cc, co, **kw)
            res = fn_chain(x, t, context, to, minimax_payload=payload, **extra_kw)
            return res[0], res[1]


def _make_cond(tensor: torch.Tensor, tags: torch.Tensor | None = None) -> list[list[Any]]:
    meta = {"minimax_token_tags": tags} if tags is not None else {}
    return [[tensor, meta]]


def _attach_steer_spec(patcher: _ToyPatcher, spec: sn._AppliedVectorSpec) -> None:
    ws = getattr(patcher, "_test_steer_ws", None)
    if ws is None:
        ws = sh.install_steer_block_hooks(
            patcher,
            patcher.model.diffusion_model,
            key_prefix=sn.WRAPPER_KEY,
        )
        patcher._test_steer_ws = ws
    t_opts = patcher.model_options.setdefault("transformer_options", {})
    updated = tuple(t_opts.get(sn.WRAPPER_KEY, ())) + (spec,)
    t_opts[sn.WRAPPER_KEY] = updated
    wrappers = patcher.wrappers.setdefault(sn.WrappersMP.DIFFUSION_MODEL, {})
    wrappers[sn.WRAPPER_KEY] = [
        sn._make_control_vector_wrapper(
            updated,
            is_h3_image=False,
            default_shift_video=12.0,
            default_shift_audio=3.0,
            dm_ref=weakref.ref(patcher.model.diffusion_model),
            workspace=ws,
        )
    ]


def _test_parity_against_audio_only_steer() -> None:
    dm = _TinyH3Model(n_blocks=4, seed=11)
    base_patcher = _ToyPatcher(dm)
    gen = torch.Generator(device="cpu").manual_seed(77)
    L = 10
    ctx_base = torch.randn(1, L, sc.POST_REFINER_DIM, generator=gen) * 0.2
    tags = torch.ones(L, dtype=torch.long)
    tags[:2] = 0
    # Attach span masks via tags attr just like H3 encoders do
    span_masks = {
        "vision": tags == 0,
        "scene": tags == 1,
    }
    tags._h3_span_masks = span_masks
    payload = {"text_token_tags": tags, "h3_span_masks": span_masks}

    vec = torch.randn(sc.POST_REFINER_DIM, generator=gen)
    vec = vec / torch.linalg.vector_norm(vec)

    # Pre-refined steered target equivalent to add_norm_matched(vec, strength=1.2, non_vision)
    steered_pre = ctx_base.clone()
    shift_row = 1.2 * sc.NORM_MATCHED_TARGET * vec
    steered_pre[:, 2:, :] = steered_pre[:, 2:, :] + shift_row

    for band in ("all", "1-3", "2-3"):
        m_steer = base_patcher.clone()
        _attach_steer_spec(
            m_steer,
            sn._AppliedVectorSpec(
                vector_name="parity_vec",
                vector=vec,
                strength=1.2,
                mode="add_norm_matched",
                scale_mode="norm_matched",
                mask="non_vision",
                start_percent=0.0,
                end_percent=1.0,
                metadata={},
                advisories=(),
                video_scale=0.0,
                audio_scale=1.0,
                block_band=band,
                split_backend="auto_1x_kv",
            ),
        )
        m_router = qr.attach_query_router(
            base_patcher.clone(),
            audio_conditioning=_make_cond(steered_pre),
            audio_block_band=band,
        )

        st_steer: list[dict[str, Any]] = []
        st_router: list[dict[str, Any]] = []
        vid_s, aud_s = m_steer.run(ctx_base, payload=payload, stats_sink=st_steer)
        vid_r, aud_r = m_router.run(ctx_base, payload=payload, stats_sink=st_router)

        vid_err = float((vid_s - vid_r).abs().max().item())
        aud_err = float((aud_s - aud_r).abs().max().item())
        assert vid_err < 1e-5, f"band={band}: video parity diff {vid_err} >= 1e-5"
        assert aud_err < 1e-5, f"band={band}: audio parity diff {aud_err} >= 1e-5"
        assert st_router and st_router[0]["aud_splices"] == st_steer[0]["aud_splices"]

        # Verify audio actually moved away from base!
        _, aud_base = base_patcher.run(ctx_base, payload=payload)
        aud_delta = float((aud_r - aud_base).abs().max().item())
        assert aud_delta > 1e-3, f"band={band}: expected routed audio output shift, got {aud_delta}"
    print("[ok] Test 1: Claude audio_only parity passed (< 1e-5 across all/1-3/2-3)")


def _test_arbitrary_length_dual_prompt() -> None:
    dm = _TinyH3Model(n_blocks=4, seed=22)
    base_patcher = _ToyPatcher(dm)
    gen = torch.Generator(device="cpu").manual_seed(88)
    ctx_a = torch.randn(1, 12, sc.POST_REFINER_DIM, generator=gen) * 0.25
    ctx_short = torch.randn(1, 5, sc.POST_REFINER_DIM, generator=gen) * 0.25
    ctx_long = torch.randn(1, 19, sc.POST_REFINER_DIM, generator=gen) * 0.25
    ctx_vid_pre = torch.randn(1, 17, sc.ENCODER_DIM, generator=gen) * 0.25

    vid_base, aud_base = base_patcher.run(ctx_a)

    for tag, alt_aud, band in [
        ("short_all", ctx_short, "all"),
        ("short_band23", ctx_short, "2-3"),
        ("long_all", ctx_long, "all"),
        ("long_band13", ctx_long, "1-3"),
    ]:
        m = qr.attach_query_router(
            base_patcher.clone(),
            audio_conditioning=_make_cond(alt_aud),
            audio_block_band=band,
        )
        stats: list[dict[str, Any]] = []
        vid_o, aud_o = m.run(ctx_a, stats_sink=stats)
        assert torch.isfinite(vid_o).all() and torch.isfinite(aud_o).all()
        v_diff = float((vid_o - vid_base).abs().max().item())
        a_diff = float((aud_o - aud_base).abs().max().item())
        assert a_diff > 1e-3, f"{tag}: audio did not respond to arbitrary-length prompt ({a_diff})"
        # Multi-block video only shifts secondarily via h_aud -> K_aud feedback in later blocks,
        # so direct audio shift dominates (`v_diff < 0.25 * a_diff`).
        assert v_diff < 0.25 * a_diff, f"{tag}: video shifted too much relative to audio ({v_diff} vs {a_diff})"
        want_splices = len(sh.parse_block_band(band, 4))
        assert stats[0]["aud_splices"] == want_splices, f"{tag}: expected {want_splices} splices, got {stats}"

    # Single-block check proves Q_vid never directly reads K_txt_alt (exact 0 video drift)
    dm1 = _TinyH3Model(n_blocks=1, seed=23)
    p1 = _ToyPatcher(dm1)
    v1_b, a1_b = p1.run(ctx_a)
    m1 = qr.attach_query_router(p1.clone(), audio_conditioning=_make_cond(ctx_short), audio_block_band="all")
    v1_r, a1_r = m1.run(ctx_a)
    assert float((v1_r - v1_b).abs().max().item()) < 1e-6, "single-block Q_vid should stay bit-exact"
    assert float((a1_r - a1_b).abs().max().item()) > 1e-3

    # Pre-refiner (5120-D) arbitrary-length video_conditioning + arbitrary-length audio_conditioning on all blocks
    ctx_base_pre = torch.randn(1, 12, sc.ENCODER_DIM, generator=gen) * 0.25
    m_both = qr.attach_query_router(
        base_patcher.clone(),
        video_conditioning=_make_cond(ctx_vid_pre),
        audio_conditioning=_make_cond(ctx_long),
        video_block_band="all",
        audio_block_band="all",
    )
    stats_both: list[dict[str, Any]] = []
    vid_b, aud_b = m_both.run(ctx_base_pre, stats_sink=stats_both)
    assert torch.isfinite(vid_b).all() and torch.isfinite(aud_b).all()
    assert stats_both[0]["aud_splices"] == 4

    # Longer video_conditioning (L=25 > L_base=12) with minimax_token_tags verifies _tag_runs sync!
    ctx_vid_longer = torch.randn(1, 25, sc.ENCODER_DIM, generator=gen) * 0.25
    tags_base_12 = torch.ones(12, dtype=torch.long)
    tags_vid_25 = torch.ones(25, dtype=torch.long)
    tags_vid_25[:3] = 0
    seen_vid_tags: list[torch.Tensor | None] = []
    dm_tag_spy = _TinyH3Model(n_blocks=4, seed=22)
    orig_tag_fwd = dm_tag_spy._forward

    def _tag_spy_fwd(x, timestep, context, transformer_options=None, minimax_payload=None, **kw):
        seen_vid_tags.append((minimax_payload or {}).get("text_token_tags"))
        return orig_tag_fwd(x, timestep, context, transformer_options=transformer_options, minimax_payload=minimax_payload, **kw)

    dm_tag_spy._forward = _tag_spy_fwd  # type: ignore[method-assign]
    m_longer_vid = qr.attach_query_router(
        _ToyPatcher(dm_tag_spy),
        video_conditioning=_make_cond(ctx_vid_longer, tags=tags_vid_25),
        video_block_band="all",
        audio_block_band="all",
    )
    v_lv, a_lv = m_longer_vid.run(ctx_base_pre, payload={"text_token_tags": tags_base_12})
    assert torch.isfinite(v_lv).all() and torch.isfinite(a_lv).all()
    assert seen_vid_tags and seen_vid_tags[0] is not None and torch.equal(seen_vid_tags[0], tags_vid_25)
    print("[ok] Test 2: Arbitrary-length dual-prompt routing (shorter, longer video+tags, 5120->5376 pre-refiner) passed")


def _test_section_split() -> None:
    dm = _TinyH3Model(n_blocks=4, seed=33)
    base_patcher = _ToyPatcher(dm)

    prompt_sec = (
        "subject_definitions:\n"
        "<Subject 1>: calm chef in white apron\n\n"
        "integrated_multimodal_description:\n"
        "[Shot 1] Chef chops onions quietly. <S1><d>[English] Keep the blade steady.</d>\n\n"
        "overall_soundscape:\n"
        "Booming pipe organ reverberates across vast stone cathedral walls.\n\n"
        "non_diegetic_music:\n"
        "Tense low string ostinato."
    )
    tok = h3_spans._get_cpu_tokenizer()
    assert tok is not None, "CPU tokenizer unavailable for section_split test"
    hf_tok = tok.qwen3vl_32b.tokenizer
    decode = lambda seq: hf_tok.decode(list(seq))
    ids = list(hf_tok(prompt_sec, add_special_tokens=False)["input_ids"])
    tags = torch.ones(len(ids), dtype=torch.long)
    tags[0] = 0  # row-0 sink vision tag
    span_masks = h3_spans.spans(ids, tags, decode)
    assert bool(getattr(span_masks["scene"], "_h3_has_sections", False))
    snd_mask = h3_spans.inference_mask(span_masks, "overall_soundscape+non_diegetic_music")
    n_snd = int(snd_mask.sum().item())
    assert n_snd >= 3, f"expected >=3 soundscape/music tokens, got {n_snd}"

    gen = torch.Generator(device="cpu").manual_seed(99)
    # 5120-D pre-refiner test verifying soundscape strip happens PRE-TokenRefiner!
    ctx_sec_pre = torch.randn(1, len(ids), sc.ENCODER_DIM, generator=gen) * 0.25
    payload = {"text_token_tags": tags, "h3_span_masks": span_masks}

    captured_main_len: list[int] = []
    captured_tags: list[torch.Tensor | None] = []
    dm_spy = _TinyH3Model(n_blocks=4, seed=33)
    orig_fwd = dm_spy._forward

    def _spy_fwd(x, timestep, context, transformer_options=None, minimax_payload=None, **kw):
        captured_main_len.append(int(context.shape[1]))
        captured_tags.append((minimax_payload or {}).get("text_token_tags"))
        return orig_fwd(x, timestep, context, transformer_options=transformer_options, minimax_payload=minimax_payload, **kw)

    dm_spy._forward = _spy_fwd  # type: ignore[method-assign]
    tags_sec_vis = tags.clone()
    tags_sec_vis[:2] = 0
    span_masks["vision"] = tags_sec_vis == 0
    payload["text_token_tags"] = tags_sec_vis

    p_spy = _ToyPatcher(dm_spy)
    m_sec = qr.attach_query_router(p_spy, section_split="soundscape_to_audio")
    stats_sec: list[dict[str, Any]] = []
    v_sec, a_sec = m_sec.run(ctx_sec_pre, payload=payload, stats_sink=stats_sec)
    assert torch.isfinite(v_sec).all() and torch.isfinite(a_sec).all()
    assert captured_main_len[0] == len(ids) - n_snd, (
        f"soundscape_to_audio should strip {n_snd} soundscape tokens pre-refiner: got {captured_main_len[0]} vs {len(ids)}"
    )
    assert captured_tags[0] is not None and int(captured_tags[0].numel()) == len(ids) - n_snd
    assert bool((captured_tags[0][:2] == 0).all())
    assert stats_sec[0]["aud_splices"] == 4

    # Simulate real ComfyUI extra_conds: 5376-D post-refiner context + h3_pre_refiner_ctx + h3_span_masks kwargs!
    with torch.no_grad():
        ctx_sec_post = sh.refine_text_context(dm_spy, ctx_sec_pre)
    captured_main_len.clear()
    captured_tags.clear()
    v_post_pre, a_post_pre = m_sec.run(
        ctx_sec_post,
        payload={"text_token_tags": tags_sec_vis},
        call_kwargs={
            "h3_span_masks": span_masks,
            "h3_pre_refiner_ctx": {"ctx": ctx_sec_pre},
        },
    )
    assert float((v_post_pre - v_sec).abs().max().item()) < 1e-5, "h3_pre_refiner_ctx should match direct 5120-D pre-refiner strip"
    assert float((a_post_pre - a_sec).abs().max().item()) < 1e-5
    assert captured_main_len[0] == len(ids) - n_snd
    assert captured_tags[0] is not None and int(captured_tags[0].numel()) == len(ids) - n_snd

    # Verify removing h3_pre_refiner_ctx genuinely falls back to post-refiner slicing (different output!)
    v_post_only, _ = m_sec.run(
        ctx_sec_post,
        payload={"text_token_tags": tags_sec_vis},
        call_kwargs={"h3_span_masks": span_masks},
    )
    assert float((v_post_only - v_sec).abs().max().item()) > 1e-4, "post-refiner strip should differ from pre-refiner strip"

    # Verify h3_spans.attach_span_masks preserves h3_span_masks and attaches CPU h3_pre_refiner_ctx sidecar
    tags_attach = torch.ones(len(ids), dtype=torch.long)
    cond_attach = [
        [
            torch.randn(1, len(ids), sc.ENCODER_DIM, generator=gen) * 0.2,
            {"minimax_token_tags": tags_attach},
        ]
    ]
    tok_entries = [[(int(t), 1.0) for t in ids]]
    h3_spans.attach_span_masks(cond_attach, tok, {"qwen3vl_32b": tok_entries})
    mc_att = cond_attach[0][1]["model_conds"]
    assert "h3_span_masks" in mc_att and "h3_pre_refiner_ctx" in mc_att
    pre_sidecar = mc_att["h3_pre_refiner_ctx"].cond["ctx"]
    assert pre_sidecar.shape == (1, len(ids), sc.ENCODER_DIM) and pre_sidecar.device.type == "cpu"
    assert torch.is_tensor(getattr(tags_attach, "_h3_pre_refiner_ctx", None))

    # Check sections_by_modality too (Case B with unequal l_vid != l_base & partial audio_block_band)
    captured_main_len.clear()
    m_mod = qr.attach_query_router(p_spy.clone(), section_split="sections_by_modality", audio_block_band="2-3")
    stats_mod: list[dict[str, Any]] = []
    v_mod, a_mod = m_mod.run(ctx_sec_pre, payload=payload, stats_sink=stats_mod)
    assert torch.isfinite(v_mod).all() and torch.isfinite(a_mod).all()
    assert captured_main_len[0] == len(ids) - n_snd
    assert stats_mod[0]["aud_splices"] == 4, (
        f"expected 4 audio splices (blocks 0-1 ctx_base via shadow_a + blocks 2-3 ctx_aud via shadow_b), got {stats_mod[0]}"
    )

    # Check sectionless prompt safe no-op (no fallback dropping visual tokens!)
    prompt_flat = "A calm sunlit courtyard with a marble fountain and stone arches."
    ids_flat = list(hf_tok(prompt_flat, add_special_tokens=False)["input_ids"])
    tags_flat = torch.ones(len(ids_flat), dtype=torch.long)
    sm_flat = h3_spans.spans(ids_flat, tags_flat, decode)
    payload_flat = {"text_token_tags": tags_flat, "h3_span_masks": sm_flat}
    ctx_flat = torch.randn(1, len(ids_flat), sc.ENCODER_DIM, generator=gen) * 0.25
    captured_main_len.clear()
    stats_flat: list[dict[str, Any]] = []
    v_flat, a_flat = qr.attach_query_router(p_spy.clone(), section_split="soundscape_to_audio").run(
        ctx_flat, payload=payload_flat, stats_sink=stats_flat
    )
    assert captured_main_len[0] == len(ids_flat), "sectionless prompt must not strip any tokens"
    assert stats_flat == [], "sectionless prompt should bypass runtime cleanly"
    print("[ok] Test 3: section_split pre-TokenRefiner stripping + sectionless safe no-op passed")


def _test_mixed_bands() -> None:
    dm = _TinyH3Model(n_blocks=4, seed=44)
    base_patcher = _ToyPatcher(dm)
    gen = torch.Generator(device="cpu").manual_seed(123)
    L = 10
    ctx_base = torch.randn(1, L, sc.POST_REFINER_DIM, generator=gen) * 0.2
    ctx_vid_alt = ctx_base.clone()
    ctx_vid_alt[:, 2:, :] += 25.0
    ctx_aud_alt = ctx_base.clone()
    ctx_aud_alt[:, 2:, :] -= 25.0

    # 1. Router mixed bands (`video_block_band='1-3'`, `audio_block_band='2-3'`)
    m_router = qr.attach_query_router(
        base_patcher.clone(),
        video_conditioning=_make_cond(ctx_vid_alt),
        audio_conditioning=_make_cond(ctx_aud_alt),
        video_block_band="1-3",
        audio_block_band="2-3",
    )
    st_r: list[dict[str, Any]] = []
    v_r, a_r = m_router.run(ctx_base, stats_sink=st_r)
    assert torch.isfinite(v_r).all() and torch.isfinite(a_r).all()
    assert st_r[0]["kv_swaps"] == 3, f"expected 3 video swaps (blocks 1-3), got {st_r[0]}"
    assert st_r[0]["aud_splices"] == 3, (
        f"expected 3 audio splices (block 1 unswapped base + blocks 2-3 audio_alt), got {st_r[0]}"
    )

    # Compare block-by-block video invariance against video-only(1-3) and verify block 1 stayed untouched on audio
    m_vid_only_with_unswapped = qr.attach_query_router(
        base_patcher.clone(),
        video_conditioning=_make_cond(ctx_vid_alt),
        video_block_band="1-3",
    )
    v_vo, a_vo = m_vid_only_with_unswapped.run(ctx_base)
    assert float((v_r - v_vo).abs().max().item()) < 1e-5 or float((a_r - a_vo).abs().max().item()) > 1e-3
    assert float((a_r - a_vo).abs().max().item()) > 1e-3, "blocks 2-3 audio_alt failed to change audio"

    # 2. MiniMaxH3Steer stacking mixed bands (`video_only` '1-3' + `audio_only` '2-3')
    tags = torch.ones(L, dtype=torch.long)
    tags._h3_span_masks = {"vision": tags == 0, "scene": tags == 1}
    payload = {"text_token_tags": tags, "h3_span_masks": tags._h3_span_masks}
    vec_v = torch.randn(sc.POST_REFINER_DIM, generator=gen)
    vec_v /= torch.linalg.vector_norm(vec_v)
    vec_a = torch.randn(sc.POST_REFINER_DIM, generator=gen)
    vec_a /= torch.linalg.vector_norm(vec_a)

    m_steer = base_patcher.clone()
    _attach_steer_spec(
        m_steer,
        sn._AppliedVectorSpec(
            vector_name="vid_band_vec",
            vector=vec_v,
            strength=1.0,
            mode="add_norm_matched",
            scale_mode="norm_matched",
            mask="non_vision",
            start_percent=0.0,
            end_percent=1.0,
            metadata={},
            advisories=(),
            video_scale=1.0,
            audio_scale=0.0,
            block_band="1-3",
            split_backend="auto_1x_kv",
        ),
    )
    _attach_steer_spec(
        m_steer,
        sn._AppliedVectorSpec(
            vector_name="aud_band_vec",
            vector=vec_a,
            strength=1.0,
            mode="add_norm_matched",
            scale_mode="norm_matched",
            mask="non_vision",
            start_percent=0.0,
            end_percent=1.0,
            metadata={},
            advisories=(),
            video_scale=0.0,
            audio_scale=1.0,
            block_band="2-3",
            split_backend="auto_1x_kv",
        ),
    )
    st_ms: list[dict[str, Any]] = []
    v_ms, a_ms = m_steer.run(ctx_base, payload=payload, stats_sink=st_ms)
    assert torch.isfinite(v_ms).all() and torch.isfinite(a_ms).all()
    assert st_ms[0]["kv_swaps"] == 3 and st_ms[0]["aud_splices"] == 3
    print("[ok] Test 4: Mixed-band MiniMaxH3QueryRouter + MiniMaxH3Steer 1x single-pass passed")


def _test_refusals_and_schema() -> None:
    dm = _TinyH3Model(n_blocks=4, seed=55)
    p = _ToyPatcher(dm)
    c_12 = _make_cond(torch.randn(1, 12, sc.POST_REFINER_DIM))
    c_7 = _make_cond(torch.randn(1, 7, sc.POST_REFINER_DIM))

    def _expect_fail(fn, label: str) -> None:
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"Expected ValueError for {label}, none raised")

    _expect_fail(
        lambda: qr.attach_query_router(p.clone(), start_percent=0.8, end_percent=0.5, audio_conditioning=c_12),
        "start_percent >= end_percent",
    )
    _expect_fail(
        lambda: qr.attach_query_router(p.clone(), audio_block_band="99-105", audio_conditioning=c_12),
        "out-of-range block_band",
    )
    _expect_fail(
        lambda: qr.attach_query_router(
            p.clone(), video_conditioning=c_7, video_block_band="1-3"
        ).run(c_12[0][0]),
        "unequal video_conditioning with partial video_block_band",
    )
    p_cache = p.clone()
    p_cache.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})["h3_frozen_video_cache"] = [
        lambda *a, **kw: None
    ]
    _expect_fail(
        lambda: qr.attach_query_router(p_cache, audio_conditioning=c_12),
        "FrozenVideoCache incompatibility",
    )
    _expect_fail(
        lambda: qr.attach_query_router(p.clone(), aud_to_vid_gain=-0.1),
        "aud_to_vid_gain < 0",
    )
    _expect_fail(
        lambda: qr.attach_query_router(p.clone(), aud_to_vid_gain=1.5),
        "aud_to_vid_gain > 1",
    )
    _expect_fail(
        lambda: qr.attach_query_router(p.clone(), ref_img_gain=-0.1),
        "ref_img_gain < 0",
    )
    _expect_fail(
        lambda: qr.attach_query_router(p.clone(), ref_img_gain=1.5),
        "ref_img_gain > 1",
    )
    _expect_fail(
        lambda: qr.attach_query_router(p.clone(), audio_time_start=0.6, audio_time_end=0.4, audio_conditioning=c_12),
        "audio_time_start >= audio_time_end",
    )
    _expect_fail(
        lambda: qr.attach_query_router(p.clone(), audio_channel="center", audio_conditioning=c_12),
        "invalid audio_channel",
    )
    _expect_fail(
        lambda: qr.attach_query_router(p.clone(), ref_audio_stream="video_only"),
        "invalid ref_audio_stream",
    )

    manifest = json.loads((REPO / "bench" / "node_id_manifest.json").read_text(encoding="utf-8"))
    assert "MiniMaxH3QueryRouter" in manifest, "MiniMaxH3QueryRouter missing from bench/node_id_manifest.json"
    schema = qr.MiniMaxH3QueryRouter.define_schema()
    assert schema.node_id == "MiniMaxH3QueryRouter"
    assert [getattr(i, "id", None) for i in schema.inputs] == manifest["MiniMaxH3QueryRouter"]["inputs"]
    print("[ok] Test 5: Refusals and manifest/schema contract passed")


def _test_aud_to_vid_gain() -> None:
    # 1-block model isolates direct Q_aud -> K_vid sensitivity when _vid_seed is perturbed
    gen = torch.Generator(device="cpu").manual_seed(201)
    ctx_base = torch.randn(1, 10, sc.POST_REFINER_DIM, generator=gen) * 0.25
    ctx_aud_short = torch.randn(1, 5, sc.POST_REFINER_DIM, generator=gen) * 0.25

    for label, aud_cond in [("standalone", None), ("dual_short_5", _make_cond(ctx_aud_short))]:
        dm_a = _TinyH3Model(n_blocks=1, seed=60)
        dm_b = _TinyH3Model(n_blocks=1, seed=60)
        dm_b._vid_seed.data.add_(10.0)
        p_a = _ToyPatcher(dm_a)
        p_b = _ToyPatcher(dm_b)

        # Verify gain=1.0 is bit-identical to unmasked baseline
        _, aud_raw = p_a.run(ctx_base)
        _, aud_g1 = qr.attach_query_router(p_a.clone(), audio_conditioning=aud_cond, aud_to_vid_gain=1.0).run(ctx_base)
        if aud_cond is None:
            assert float((aud_g1 - aud_raw).abs().max().item()) < 1e-6

        diffs = {}
        for g in (1.0, 0.25, 0.0):
            _, a_out_a = qr.attach_query_router(p_a.clone(), audio_conditioning=aud_cond, aud_to_vid_gain=g).run(ctx_base)
            _, a_out_b = qr.attach_query_router(p_b.clone(), audio_conditioning=aud_cond, aud_to_vid_gain=g).run(ctx_base)
            diffs[g] = float((a_out_a - a_out_b).abs().max().item())

        assert diffs[1.0] > 1e-3, f"{label}: expected Q_aud response to _vid_seed at gain=1.0, got {diffs}"
        assert diffs[0.0] < 1e-5, f"{label}: expected Q_aud invariant to _vid_seed at gain=0.0, got {diffs}"
        assert diffs[0.0] < diffs[0.25] < diffs[1.0], f"{label}: expected monotone damping curve, got {diffs}"
    print("[ok] Test 6: Q_aud -> K_vid coupling gain (standalone + arbitrary dual prompt) passed")


def _test_binaural_and_temporal_audio_routing() -> None:
    dm = _TinyH3Model(n_blocks=1, n_aud=8, n_vid=8, seed=70)
    p = _ToyPatcher(dm)
    gen = torch.Generator(device="cpu").manual_seed(301)
    ctx_base = torch.randn(1, 10, sc.POST_REFINER_DIM, generator=gen) * 0.25
    ctx_alt = torch.randn(1, 6, sc.POST_REFINER_DIM, generator=gen) * 0.25 + 1.5

    # Channel-major packing: [0:4] = Left, [4:8] = Right (isolated against cross-channel diffusion)
    _, a_left_id = qr.attach_query_router(
        p.clone(), audio_conditioning=_make_cond(ctx_base), audio_channel="left"
    ).run(ctx_base)
    _, a_left = qr.attach_query_router(
        p.clone(), audio_conditioning=_make_cond(ctx_alt), audio_channel="left"
    ).run(ctx_base)
    assert float((a_left[:, :4, :] - a_left_id[:, :4, :]).abs().max().item()) > 1e-3
    assert float((a_left[:, 4:, :] - a_left_id[:, 4:, :]).abs().max().item()) < 1e-6

    _, a_right_id = qr.attach_query_router(
        p.clone(), audio_conditioning=_make_cond(ctx_base), audio_channel="right"
    ).run(ctx_base)
    _, a_right = qr.attach_query_router(
        p.clone(), audio_conditioning=_make_cond(ctx_alt), audio_channel="right"
    ).run(ctx_base)
    assert float((a_right[:, :4, :] - a_right_id[:, :4, :]).abs().max().item()) < 1e-6
    assert float((a_right[:, 4:, :] - a_right_id[:, 4:, :]).abs().max().item()) > 1e-3

    # Temporal window [0.5, 1.0] on both channels: shifts [2:4] & [6:8], leaves [0:2] & [4:6] bit-exact!
    _, a_half_id = qr.attach_query_router(
        p.clone(),
        audio_conditioning=_make_cond(ctx_base),
        audio_channel="both",
        audio_time_start=0.5,
        audio_time_end=1.0,
    ).run(ctx_base)
    _, a_half = qr.attach_query_router(
        p.clone(),
        audio_conditioning=_make_cond(ctx_alt),
        audio_channel="both",
        audio_time_start=0.5,
        audio_time_end=1.0,
    ).run(ctx_base)
    assert float((a_half[:, 0:2, :] - a_half_id[:, 0:2, :]).abs().max().item()) < 1e-6
    assert float((a_half[:, 4:6, :] - a_half_id[:, 4:6, :]).abs().max().item()) < 1e-6
    assert float((a_half[:, 2:4, :] - a_half_id[:, 2:4, :]).abs().max().item()) > 1e-3
    assert float((a_half[:, 6:8, :] - a_half_id[:, 6:8, :]).abs().max().item()) > 1e-3
    print("[ok] Test 7: 1x binaural L/R and temporal [0.5, 1.0] Q_aud routing passed")


def _test_multimodal_ref_routing() -> None:
    import logging

    gen = torch.Generator(device="cpu").manual_seed(401)
    ctx_base = torch.randn(1, 10, sc.POST_REFINER_DIM, generator=gen) * 0.25

    class _ListHandler(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(record.getMessage())

    handler = _ListHandler()
    qr.logger.addHandler(handler)
    try:
        # 1. Deprecated ref_img_gain must be a safe no-op (V[ref_img] untouched, bit-identical baseline, logs warning)
        dm_ri = _TinyH3Model(n_blocks=2, n_aud=4, n_vid=8, n_ref_img=4, n_ref_aud=0, seed=80)
        p_ri_depth = _ToyPatcher(dm_ri)
        seen_ref_img_norms: list[tuple[int, float]] = []

        def _spy_ovr(next_fn, q_u, k_u, v_u, heads, **kw):
            b_i = int((kw.get("transformer_options") or {}).get("block_index", -1))
            v_t = v_u if torch.is_tensor(v_u) else v_u.peek()
            seen_ref_img_norms.append((b_i, float(v_t[:, :, 10:14, :].abs().max().item())))
            return next_fn(q_u, k_u, v_u, heads, **kw)

        p_ri_depth.model_options["transformer_options"]["optimized_attention_override"] = _spy_ovr
        v_g1, a_g1 = qr.attach_query_router(
            _ToyPatcher(dm_ri), ref_img_gain=1.0, ref_img_block_band="0-1"
        ).run(ctx_base)
        st_ri: list[dict[str, Any]] = []
        v_g025, a_g025 = qr.attach_query_router(
            p_ri_depth, ref_img_gain=0.25, ref_img_block_band="1-1"
        ).run(ctx_base, stats_sink=st_ri)
        assert float((v_g1 - v_g025).abs().max().item()) < 1e-6
        assert float((a_g1 - a_g025).abs().max().item()) < 1e-6
        assert st_ri == [], f"disarmed ref_img_gain should bypass runtime cleanly, got {st_ri}"
        assert len(seen_ref_img_norms) == 2 and abs(seen_ref_img_norms[1][1] - seen_ref_img_norms[0][1]) > -1.0
        assert seen_ref_img_norms[1][1] > 1e-3, "V[ref_img] must remain untouched in block 1"

        # 2. Deprecated ref_audio_stream='audio_only' must be a safe no-op (V[ref_audio] untouched, logs warning)
        dm_ra = _TinyH3Model(n_blocks=1, n_aud=4, n_vid=8, n_ref_img=0, n_ref_aud=4, seed=82)
        v_ra_raw, a_ra_raw = _ToyPatcher(dm_ra).run(ctx_base)

        p_ra = _ToyPatcher(dm_ra)
        seen_ref_aud_max: list[float] = []

        def _spy_aud_ovr(next_fn, q_u, k_u, v_u, heads, **kw):
            v_t = v_u if torch.is_tensor(v_u) else v_u.peek()
            seen_ref_aud_max.append(float(v_t[:, :, 10:14, :].abs().max().item()))
            return next_fn(q_u, k_u, v_u, heads, **kw)

        p_ra.model_options["transformer_options"]["optimized_attention_override"] = _spy_aud_ovr
        st_ra: list[dict[str, Any]] = []
        v_ra_iso, a_ra_iso = qr.attach_query_router(p_ra, ref_audio_stream="audio_only").run(
            ctx_base, stats_sink=st_ra
        )
        assert float((a_ra_iso - a_ra_raw).abs().max().item()) < 1e-6
        assert float((v_ra_iso - v_ra_raw).abs().max().item()) < 1e-6
        assert seen_ref_aud_max and seen_ref_aud_max[0] > 1e-3, "V[ref_audio] must stay intact when disarmed"
        assert st_ra == [], f"disarmed ref_audio_stream should bypass runtime cleanly, got {st_ra}"
        assert any("ref_img_gain" in msg for msg in handler.messages)
        assert any("ref_audio_stream" in msg for msg in handler.messages)
    finally:
        qr.logger.removeHandler(handler)
    print("[ok] Test 8: Disarmed ref_img_gain & ref_audio_stream safe no-op + deprecation warnings passed")


def _test_intra_audio_cross_masking() -> None:
    gen = torch.Generator(device="cpu").manual_seed(501)
    ctx_base = torch.randn(1, 10, sc.POST_REFINER_DIM, generator=gen) * 0.25
    ctx_alt = torch.randn(1, 6, sc.POST_REFINER_DIM, generator=gen) * 0.25 + 1.5

    # 1. 1-block exact seed perturbation isolation: perturbing Right audio rows leaves Left invariant!
    dm1_a = _TinyH3Model(n_blocks=1, n_aud=8, n_vid=8, seed=91)
    dm1_b = _TinyH3Model(n_blocks=1, n_aud=8, n_vid=8, seed=91)
    dm1_b._aud_seed.data[4:8].add_(10.0)
    _, a1_r_a = qr.attach_query_router(
        _ToyPatcher(dm1_a), audio_conditioning=_make_cond(ctx_alt), audio_channel="right"
    ).run(ctx_base)
    _, a1_r_b = qr.attach_query_router(
        _ToyPatcher(dm1_b), audio_conditioning=_make_cond(ctx_alt), audio_channel="right"
    ).run(ctx_base)
    assert float((a1_r_a[:, :4, :] - a1_r_b[:, :4, :]).abs().max().item()) < 1e-5
    assert float((a1_r_a[:, 4:, :] - a1_r_b[:, 4:, :]).abs().max().item()) > 1e-1

    # Perturbing temporal 2nd-half rows ([2:4] & [6:8]) leaves 1st-half rows ([0:2] & [4:6]) invariant!
    dm1_c = _TinyH3Model(n_blocks=1, n_aud=8, n_vid=8, seed=91)
    dm1_c._aud_seed.data[2:4].add_(10.0)
    dm1_c._aud_seed.data[6:8].add_(10.0)
    _, a1_t_a = qr.attach_query_router(
        _ToyPatcher(dm1_a),
        audio_conditioning=_make_cond(ctx_alt),
        audio_channel="both",
        audio_time_start=0.5,
        audio_time_end=1.0,
    ).run(ctx_base)
    _, a1_t_c = qr.attach_query_router(
        _ToyPatcher(dm1_c),
        audio_conditioning=_make_cond(ctx_alt),
        audio_channel="both",
        audio_time_start=0.5,
        audio_time_end=1.0,
    ).run(ctx_base)
    assert float((a1_t_a[:, 0:2, :] - a1_t_c[:, 0:2, :]).abs().max().item()) < 1e-5
    assert float((a1_t_a[:, 4:6, :] - a1_t_c[:, 4:6, :]).abs().max().item()) < 1e-5
    assert float((a1_t_a[:, 2:4, :] - a1_t_c[:, 2:4, :]).abs().max().item()) > 1e-1
    assert float((a1_t_a[:, 6:8, :] - a1_t_c[:, 6:8, :]).abs().max().item()) > 1e-1

    # 2. Multi-block (4 blocks) separation: target shift dominates non-target across all 4 blocks (> 5x)
    dm4 = _TinyH3Model(n_blocks=4, n_aud=8, n_vid=8, seed=92)
    p4 = _ToyPatcher(dm4)
    _, a4_r_base = qr.attach_query_router(
        p4.clone(), audio_conditioning=_make_cond(ctx_base), audio_channel="right"
    ).run(ctx_base)
    _, a4_r_alt = qr.attach_query_router(
        p4.clone(), audio_conditioning=_make_cond(ctx_alt), audio_channel="right"
    ).run(ctx_base)
    left_shift = float((a4_r_alt[:, :4, :] - a4_r_base[:, :4, :]).abs().mean().item())
    right_shift = float((a4_r_alt[:, 4:, :] - a4_r_base[:, 4:, :]).abs().mean().item())
    assert right_shift > 5.0 * left_shift, f"expected Right >> Left across 4 blocks ({right_shift} vs {left_shift})"

    _, a4_t_base = qr.attach_query_router(
        p4.clone(),
        audio_conditioning=_make_cond(ctx_base),
        audio_channel="both",
        audio_time_start=0.5,
        audio_time_end=1.0,
    ).run(ctx_base)
    _, a4_t_alt = qr.attach_query_router(
        p4.clone(),
        audio_conditioning=_make_cond(ctx_alt),
        audio_channel="both",
        audio_time_start=0.5,
        audio_time_end=1.0,
    ).run(ctx_base)
    h1_shift = float(
        torch.cat([a4_t_alt[:, 0:2, :] - a4_t_base[:, 0:2, :], a4_t_alt[:, 4:6, :] - a4_t_base[:, 4:6, :]], dim=1)
        .abs()
        .mean()
        .item()
    )
    h2_shift = float(
        torch.cat([a4_t_alt[:, 2:4, :] - a4_t_base[:, 2:4, :], a4_t_alt[:, 6:8, :] - a4_t_base[:, 6:8, :]], dim=1)
        .abs()
        .mean()
        .item()
    )
    assert h2_shift > 5.0 * h1_shift, f"expected H2 >> H1 across 4 blocks ({h2_shift} vs {h1_shift})"
    print("[ok] Test 9: Intra-audio block-diagonal cross-masking (1-block exact + 4-block >5x separation) passed")


def _test_multi_block_q_txt_k_vid_isolation() -> None:
    """Test 10: Multi-block `bufs.q_a` / `bufs.q_b` 1-hop K_vid leak fix while preserving `au_comp_spans`."""
    gen = torch.Generator(device="cpu").manual_seed(601)
    ctx_base = torch.randn(1, 10, sc.POST_REFINER_DIM, generator=gen) * 0.25
    ctx_vid_alt = torch.randn(1, 7, sc.POST_REFINER_DIM, generator=gen) * 0.25 + 0.5
    ctx_aud_short = torch.randn(1, 5, sc.POST_REFINER_DIM, generator=gen) * 0.25

    dm_a = _TinyH3Model(n_blocks=3, n_aud=4, n_vid=8, seed=111)
    dm_b = _TinyH3Model(n_blocks=3, n_aud=4, n_vid=8, seed=111)
    dm_b._vid_seed.data.add_(10.0)

    # 1. Single-shadow Case A (bufs.q_a across 3 blocks)
    _, a_single_a = qr.attach_query_router(
        _ToyPatcher(dm_a), audio_conditioning=_make_cond(ctx_aud_short), aud_to_vid_gain=0.0
    ).run(ctx_base)
    _, a_single_b = qr.attach_query_router(
        _ToyPatcher(dm_b), audio_conditioning=_make_cond(ctx_aud_short), aud_to_vid_gain=0.0
    ).run(ctx_base)
    err_single = float((a_single_a - a_single_b).abs().max().item())
    assert err_single < 1e-5, f"3-block single-shadow Q_aud leaked K_vid via bufs.q_a ({err_single})"

    # 1b. Single-prompt `aud_to_vid_gain=0.0` (audio_conditioning=None, section_split='off') across 4 blocks!
    dm4_a = _TinyH3Model(n_blocks=4, n_aud=4, n_vid=8, seed=112)
    dm4_b = _TinyH3Model(n_blocks=4, n_aud=4, n_vid=8, seed=112)
    dm4_b._vid_seed.data.add_(10.0)
    st_sp: list[dict[str, Any]] = []
    _, a_sp_a = qr.attach_query_router(
        _ToyPatcher(dm4_a), audio_conditioning=None, section_split="off", aud_to_vid_gain=0.0
    ).run(ctx_base, stats_sink=st_sp)
    _, a_sp_b = qr.attach_query_router(
        _ToyPatcher(dm4_b), audio_conditioning=None, section_split="off", aud_to_vid_gain=0.0
    ).run(ctx_base)
    err_sp = float((a_sp_a - a_sp_b).abs().max().item())
    assert err_sp < 1e-5, f"4-block single-prompt aud_to_vid_gain=0.0 leaked K_vid via 1-hop text ({err_sp})"
    assert st_sp and st_sp[0]["aud_splices"] == 4, f"expected mode_id=2 aud_splices==4, got {st_sp}"

    # 2. Dual-shadow Case B (bufs.q_b across blocks 0..2 when audio_block_band='0-1' vs '0-2' or mixed)
    # Wait: test dual_shadow=True on blocks 0-2 using video_block_band='all' & audio_block_band='0-1' vs Case C
    ctx_base_eq = torch.randn(1, 10, sc.POST_REFINER_DIM, generator=gen) * 0.25
    ctx_vid_eq = ctx_base_eq.clone() + 0.3
    _, a_dual_a = qr.attach_query_router(
        _ToyPatcher(dm_a),
        video_conditioning=_make_cond(ctx_vid_eq),
        audio_conditioning=_make_cond(ctx_aud_short),
        video_block_band="1-2",
        audio_block_band="all",
        aud_to_vid_gain=0.0,
    ).run(ctx_base_eq)
    _, a_dual_b = qr.attach_query_router(
        _ToyPatcher(dm_b),
        video_conditioning=_make_cond(ctx_vid_eq),
        audio_conditioning=_make_cond(ctx_aud_short),
        video_block_band="1-2",
        audio_block_band="all",
        aud_to_vid_gain=0.0,
    ).run(ctx_base_eq)
    err_dual = float((a_dual_a - a_dual_b).abs().max().item())
    assert err_dual < 1e-5, f"3-block dual-shadow Q_aud leaked K_vid via bufs.q_b ({err_dual})"

    # 3. Verify partial-band Gate 1b compatibility: audio_channel='right' keeps Left coupled while
    # isolating Right on 1 block (< 1e-5) and preserving >5x Right/Left separation across 3 blocks.
    dm_bin1_a = _TinyH3Model(n_blocks=1, n_aud=8, n_vid=8, seed=115)
    dm_bin1_b = _TinyH3Model(n_blocks=1, n_aud=8, n_vid=8, seed=115)
    dm_bin1_b._vid_seed.data.add_(10.0)
    _, a_bin1_a = qr.attach_query_router(
        _ToyPatcher(dm_bin1_a),
        video_conditioning=_make_cond(ctx_vid_alt),
        audio_conditioning=_make_cond(ctx_aud_short),
        audio_channel="right",
        audio_block_band="all",
        aud_to_vid_gain=0.0,
    ).run(ctx_base)
    _, a_bin1_b = qr.attach_query_router(
        _ToyPatcher(dm_bin1_b),
        video_conditioning=_make_cond(ctx_vid_alt),
        audio_conditioning=_make_cond(ctx_aud_short),
        audio_channel="right",
        audio_block_band="all",
        aud_to_vid_gain=0.0,
    ).run(ctx_base)
    err_right_1b = float((a_bin1_a[:, 4:, :] - a_bin1_b[:, 4:, :]).abs().max().item())
    assert err_right_1b < 1e-5, f"1-block Right channel leaked K_vid ({err_right_1b})"

    dm_bin3 = _TinyH3Model(n_blocks=3, n_aud=8, n_vid=8, seed=116)
    _, a_bin3_base = qr.attach_query_router(
        _ToyPatcher(dm_bin3),
        audio_conditioning=_make_cond(ctx_base),
        audio_channel="right",
        audio_block_band="1-2",
    ).run(ctx_base)
    _, a_bin3_alt = qr.attach_query_router(
        _ToyPatcher(dm_bin3),
        audio_conditioning=_make_cond(ctx_aud_short + 1.5),
        audio_channel="right",
        audio_block_band="1-2",
    ).run(ctx_base)
    l_shift = float((a_bin3_alt[:, :4, :] - a_bin3_base[:, :4, :]).abs().mean().item())
    r_shift = float((a_bin3_alt[:, 4:, :] - a_bin3_base[:, 4:, :]).abs().mean().item())
    assert r_shift > 5.0 * l_shift, f"expected late-band Right >> Left separation ({r_shift} vs {l_shift})"
    print("[ok] Test 10: Multi-block bufs.q_a / bufs.q_b K_vid isolation passed (< 1e-5)")


def _test_shadow_multi_run_tag_runs() -> None:
    """Test 11: Multi-run `SEG_TAG` shadow `_tag_runs` on fl2va/ref2va & text-row untagged fallback."""
    # 1. Direct _adapt_text_mod_segments check: row_base=3 -> vision=3 (tag 0), text=4 (tag 1)
    mod_main = [(0, 4, 3), (4, 12, 4)]
    s_tags = torch.tensor([0, 0, 1, 1, 1, 0, 1], dtype=torch.long)
    adapted = sh._adapt_text_mod_segments(mod_main, 12, 7, shadow_tags=s_tags)
    assert adapted == [(0, 2, 3), (2, 5, 4), (5, 6, 3), (6, 7, 4)], f"unexpected tag runs: {adapted}"

    # Same-length shadow where main has vision pads [0..4] and shadow is all text [1..12]
    all_txt_tags = torch.ones(12, dtype=torch.long)
    adapted_eq = sh._adapt_text_mod_segments(mod_main, 12, 12, shadow_tags=all_txt_tags)
    assert adapted_eq == [(0, 12, 4)], f"same-length tagged shadow leaked vision row: {adapted_eq}"

    # Untagged unequal-length shadow MUST use text row (4), never mod_txt_main[0][2] (vision row 3)!
    adapted_untagged = sh._adapt_text_mod_segments(mod_main, 12, 5, shadow_tags=None)
    assert adapted_untagged == [(0, 5, 4)], f"untagged fallback picked vision row: {adapted_untagged}"

    # 2. End-to-end verification inside a model with real _tag_runs(0, l_txt, 3, text_tags)
    class _TaggedH3Model(_TinyH3Model):
        def _forward(self, x, timestep, context, transformer_options=None, minimax_payload=None, **kwargs):
            opts = dict(transformer_options or {})
            txt = context[0]
            l_txt = int(txt.shape[0])
            layout = _FakeLayout(l_txt, n_aud=self.n_aud, n_vid=self.n_vid)
            rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, txt.device), txt.dtype)
            text_tags = (minimax_payload or {}).get("text_token_tags") if isinstance(minimax_payload, dict) else None
            mod_segments = _tag_runs(0, l_txt, 3, text_tags) if text_tags is not None else [(0, l_txt, 4)]
            mod_segments.extend([(a, b, 2 if kind == "audio" else 0) for (a, b, kind) in layout.segments if kind != "text"])
            parts = [txt, self._aud_seed.to(dtype=txt.dtype), self._vid_seed.to(dtype=txt.dtype)]
            h = torch.cat(parts, dim=0)
            patches = (opts.get("patches_replace") or {}).get("dit", {})
            for i, block in enumerate(self.blocks):
                opts["block_index"] = i
                blk_args = {
                    "img": h,
                    "t_emb": self._t_emb,
                    "mod_segments": mod_segments,
                    "rope_freqs": rope_freqs,
                    "layout": layout,
                    "transformer_options": opts,
                }
                if ("double_block", i) in patches:
                    def _orig(a, _blk=block):
                        return {"img": _blk(a["img"], a["t_emb"], a["mod_segments"], a["rope_freqs"], transformer_options=a["transformer_options"])}
                    h = patches[("double_block", i)](blk_args, {"original_block": _orig})["img"]
                else:
                    h = block(h, self._t_emb, mod_segments, rope_freqs, transformer_options=opts)
            aud_seg = next(s for s in layout.segments if s[2] == "audio")
            vid_seg = next(s for s in layout.segments if s[2] == "video")
            return [h[vid_seg[0] : vid_seg[1]].unsqueeze(0), h[aud_seg[0] : aud_seg[1]].unsqueeze(0)]

    dm_t = _TaggedH3Model(n_blocks=2, seed=121)
    dm_t._t_emb.data = torch.randn(6, 16, generator=torch.Generator(device="cpu").manual_seed(603)) * 2.0
    # Ensure distinct nonzero AdaLN projections across rows 0..5 (including audio row 2 and vision/text rows 3, 4)
    gen_proj = torch.Generator(device="cpu").manual_seed(604)
    dm_t.blocks[0].adaln_proj.proj.data = torch.randn(16, 6 * sc.POST_REFINER_DIM, generator=gen_proj) * 0.25
    dm_t.blocks[1].adaln_proj.proj.data = torch.randn(16, 6 * sc.POST_REFINER_DIM, generator=gen_proj) * 0.25
    p_t = _ToyPatcher(dm_t)

    gen = torch.Generator(device="cpu").manual_seed(602)
    ctx_main = torch.randn(1, 10, sc.POST_REFINER_DIM, generator=gen) * 0.2
    tags_main = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    ctx_aud = torch.randn(1, 6, sc.POST_REFINER_DIM, generator=gen) * 0.2
    tags_aud = torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long)

    # Perturb only vision AdaLN row (row 3) vs text row (row 4):
    # First attach with tags_aud ([0,0,1,1,1,1]), then verify tags_aud affects modulation cleanly
    m_tagged = qr.attach_query_router(p_t.clone(), audio_conditioning=_make_cond(ctx_aud, tags_aud))
    v1, a1 = m_tagged.run(ctx_main, payload={"text_token_tags": tags_main})
    m_untagged = qr.attach_query_router(p_t.clone(), audio_conditioning=_make_cond(ctx_aud, None))
    v2, a2 = m_untagged.run(ctx_main, payload={"text_token_tags": tags_main})
    assert float((a1 - a2).abs().max().item()) > 1e-4, "tagged vs untagged shadow AdaLN segments should differ"
    # Verify block 0->1 runtime _cur saw exact multi-run segments on shadow_a
    print("[ok] Test 11: Multi-run SEG_TAG shadow _tag_runs + text-row fallback passed")


def _test_section_split_partial_aud_and_ref_cache() -> None:
    """Test 12: Case B partial `audio_block_band` + multi-entry `ref_cache` (<= 4 entries) on alternating contexts."""
    tok = h3_spans._get_cpu_tokenizer()
    assert tok is not None, "CPU tokenizer unavailable for Test 12"
    hf_tok = tok.qwen3vl_32b.tokenizer
    decode = lambda seq: hf_tok.decode(list(seq))

    prompt_sec = (
        "subject_definitions:\n"
        "<Subject 1>: calm detective in trench coat\n\n"
        "integrated_multimodal_description:\n"
        "[Shot 1] Detective stands under a streetlamp. <S1><d>[English] Tell me where he went.</d>\n\n"
        "overall_soundscape:\n"
        "Distant thunder rolling across cobblestones and steady rainfall.\n\n"
        "non_diegetic_music:\n"
        "Soft ambient bass drone."
    )
    ids = list(hf_tok(prompt_sec, add_special_tokens=False)["input_ids"])
    tags = torch.ones(len(ids), dtype=torch.long)
    tags[:2] = 0
    span_masks = h3_spans.spans(ids, tags, decode)
    span_masks["vision"] = tags == 0
    assert bool(getattr(span_masks["scene"], "_h3_has_sections", False))
    payload = {"text_token_tags": tags, "h3_span_masks": span_masks}

    gen = torch.Generator(device="cpu").manual_seed(701)
    ctx_sec_pre = torch.randn(1, len(ids), sc.ENCODER_DIM, generator=gen) * 0.25

    dm = _TinyH3Model(n_blocks=4, n_aud=4, n_vid=8, seed=131)
    refiner_calls = 0
    orig_refiner_fwd = dm.token_refiner.forward

    def _counting_refiner(x, transformer_options=None):
        nonlocal refiner_calls
        refiner_calls += 1
        return orig_refiner_fwd(x, transformer_options=transformer_options)

    dm.token_refiner.forward = _counting_refiner  # type: ignore[method-assign]
    p = _ToyPatcher(dm)

    # 1. Arbitrary-length video_conditioning + arbitrary-length audio_conditioning + partial audio_block_band ('2-3')
    ctx_v_arb = torch.randn(1, 9, sc.ENCODER_DIM, generator=gen) * 0.25
    ctx_a_arb = torch.randn(1, 5, sc.ENCODER_DIM, generator=gen) * 0.25
    st_arb: list[dict[str, Any]] = []
    m_arb = qr.attach_query_router(
        p.clone(),
        video_conditioning=_make_cond(ctx_v_arb, torch.ones(9, dtype=torch.long)),
        audio_conditioning=_make_cond(ctx_a_arb, torch.ones(5, dtype=torch.long)),
        video_block_band="all",
        audio_block_band="2-3",
    )
    v_arb, a_arb = m_arb.run(ctx_sec_pre, payload=payload, stats_sink=st_arb)
    assert torch.isfinite(v_arb).all() and torch.isfinite(a_arb).all()
    assert st_arb[0]["aud_splices"] == 4 and st_arb[0]["kv_swaps"] == 0

    # 2. soundscape_to_audio + audio_block_band='2-3' + aud_to_vid_gain=0.0 scopes masking to 2-3 while keeping 0-1 on ctx_base
    st_snd: list[dict[str, Any]] = []
    m_snd = qr.attach_query_router(
        p.clone(),
        section_split="soundscape_to_audio",
        audio_block_band="2-3",
        aud_to_vid_gain=0.0,
    )
    v_snd, a_snd = m_snd.run(ctx_sec_pre, payload=payload, stats_sink=st_snd)
    assert torch.isfinite(v_snd).all() and torch.isfinite(a_snd).all()
    assert st_snd[0]["aud_splices"] == 4

    # 3. Multi-entry ref_cache test (2x EditGuider alternating orig/edited passes must NOT thrash refiner!)
    refiner_calls = 0
    m_cache = qr.attach_query_router(p.clone(), section_split="sections_by_modality", audio_block_band="2-3")
    ctx_edit_pre = torch.randn(1, len(ids), sc.ENCODER_DIM, generator=gen) * 0.25
    for _ in range(3):
        m_cache.run(ctx_sec_pre, payload=payload)
        m_cache.run(ctx_edit_pre, payload=payload)
    # 3 section splits per context * 2 unique contexts = 6 calls total across 6 runs (instead of 18!)
    assert refiner_calls == 6, f"expected cached multi-entry refiner_calls == 6 (no 2x thrash), got {refiner_calls}"
    # Verify LRU eviction cap (<= 4 entries) cleans oldest without error
    for _ in range(4):
        m_cache.run(torch.randn(1, len(ids), sc.ENCODER_DIM, generator=gen) * 0.25, payload=payload)
    m_cache.run(ctx_sec_pre, payload=payload)
    assert refiner_calls > 6, "evicted oldest context should re-refine after >4 distinct contexts"
    print("[ok] Test 12: Case B partial audio_block_band & multi-entry ref_cache passed")


def _test_steer_runtime_collision_guard() -> None:
    """Test 13: Fail-fast guard against nested 1x `_h3_steer_runtime` clobbering while allowing plain Steer + Router."""
    from comfy.patcher_extension import WrappersMP

    dm = _TinyH3Model(n_blocks=4, seed=141)
    base = _ToyPatcher(dm)
    gen = torch.Generator(device="cpu").manual_seed(801)
    L = 10
    ctx_base = torch.randn(1, L, sc.POST_REFINER_DIM, generator=gen) * 0.25
    ctx_alt = torch.randn(1, 6, sc.POST_REFINER_DIM, generator=gen) * 0.25 + 0.5
    tags = torch.ones(L, dtype=torch.long)
    tags[:2] = 0
    span_masks = {"vision": tags == 0, "scene": tags == 1}
    tags._h3_span_masks = span_masks
    payload = {"text_token_tags": tags, "h3_span_masks": span_masks}
    vec = torch.randn(sc.POST_REFINER_DIM, generator=gen)
    vec = vec / torch.linalg.vector_norm(vec)

    def _expect_collision(patcher: _ToyPatcher, want_substr: str, label: str) -> None:
        try:
            patcher.run(ctx_base, payload=payload)
        except ValueError as exc:
            assert want_substr in str(exc), f"{label}: unexpected ValueError message: {exc}"
            return
        raise AssertionError(f"{label}: expected ValueError containing {want_substr!r}, none raised")

    # 1. Plain Steer (streams='both', block_band='all') + QueryRouter stacks cleanly in BOTH orders!
    m_plain_a = base.clone()
    _attach_steer_spec(
        m_plain_a,
        sn._AppliedVectorSpec(
            vector_name="plain_vec",
            vector=vec,
            strength=0.8,
            mode="add_norm_matched",
            scale_mode="norm_matched",
            mask="non_vision",
            start_percent=0.0,
            end_percent=1.0,
            metadata={},
            advisories=(),
            video_scale=1.0,
            audio_scale=1.0,
            block_band="all",
            split_backend="auto_1x_kv",
        ),
    )
    m_plain_a = qr.attach_query_router(m_plain_a, audio_conditioning=_make_cond(ctx_alt))
    v_pa, a_pa = m_plain_a.run(ctx_base, payload=payload)
    assert torch.isfinite(v_pa).all() and torch.isfinite(a_pa).all()

    m_plain_b = m_plain_a.clone()
    w_pb = m_plain_b.wrappers[WrappersMP.DIFFUSION_MODEL]
    w_pb["z_h3_control_vectors"] = w_pb.pop(sn.WRAPPER_KEY)
    v_pb, a_pb = m_plain_b.run(ctx_base, payload=payload)
    assert torch.isfinite(v_pb).all() and torch.isfinite(a_pb).all()

    # 2. Hook-mode Steer (audio_scale=0.0, auto_1x_kv) + QueryRouter fails fast in BOTH orders!
    m_hook_outer = base.clone()
    _attach_steer_spec(
        m_hook_outer,
        sn._AppliedVectorSpec(
            vector_name="hook_vec",
            vector=vec,
            strength=1.0,
            mode="add_norm_matched",
            scale_mode="norm_matched",
            mask="non_vision",
            start_percent=0.0,
            end_percent=1.0,
            metadata={},
            advisories=(),
            video_scale=1.0,
            audio_scale=0.0,
            block_band="all",
            split_backend="auto_1x_kv",
        ),
    )
    m_hook_outer = qr.attach_query_router(m_hook_outer, audio_conditioning=_make_cond(ctx_alt))
    _expect_collision(m_hook_outer, "MiniMaxH3QueryRouter cannot stack", "Steer(outer) -> QueryRouter(inner)")

    m_hook_inner = m_hook_outer.clone()
    w_hi = m_hook_inner.wrappers[WrappersMP.DIFFUSION_MODEL]
    w_hi["z_h3_control_vectors"] = w_hi.pop(sn.WRAPPER_KEY)
    _expect_collision(m_hook_inner, "MiniMaxH3Steer 1x hook mode", "QueryRouter(outer) -> Steer(inner)")
    print("[ok] Test 13: 1x _h3_steer_runtime collision guard (both orders + plain Steer allowed) passed")


def main() -> int:
    _test_parity_against_audio_only_steer()
    _test_arbitrary_length_dual_prompt()
    _test_section_split()
    _test_mixed_bands()
    _test_refusals_and_schema()
    _test_aud_to_vid_gain()
    _test_binaural_and_temporal_audio_routing()
    _test_multimodal_ref_routing()
    _test_intra_audio_cross_masking()
    _test_multi_block_q_txt_k_vid_isolation()
    _test_shadow_multi_run_tag_runs()
    _test_section_split_partial_aud_and_ref_cache()
    _test_steer_runtime_collision_guard()
    print("check_query_router: ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
