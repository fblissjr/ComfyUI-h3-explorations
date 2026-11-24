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
        ovr = opts.get("optimized_attention_override")
        if ovr is not None:
            out = ovr(
                optimized_attention,
                q_c,
                k_c,
                v_c,
                self.heads,
                mask=None,
                skip_reshape=True,
                transformer_options=opts,
            )
        else:
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
    def __init__(self, l_txt: int, n_aud: int = 4, n_vid: int = 8) -> None:
        self.segments = [
            (0, l_txt, "text"),
            (l_txt, l_txt + n_aud, "audio"),
            (l_txt + n_aud, l_txt + n_aud + n_vid, "video"),
        ]
        self.seq_len = l_txt + n_aud + n_vid
        pos = torch.zeros(self.seq_len, 3, dtype=torch.float64)
        pos[:, 0] = torch.arange(self.seq_len, dtype=torch.float64)
        # Put Left/Right audio on opposite screen-edge w columns and video on centered grid
        half_a = n_aud // 2
        pos[l_txt : l_txt + half_a, 2] = 0.0
        pos[l_txt + half_a : l_txt + n_aud, 2] = 32.0
        pos[l_txt + n_aud :, 1] = 16.0
        pos[l_txt + n_aud :, 2] = 16.0
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
    def __init__(self, *, n_blocks: int = 4, n_aud: int = 4, n_vid: int = 8, seed: int = 100) -> None:
        super().__init__()
        self.hidden_size = sc.POST_REFINER_DIM
        self.n_aud = n_aud
        self.n_vid = n_vid
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
            txt = self.token_refiner(self.condition_proj(txt), transformer_options=opts)
        l_txt = int(txt.shape[0])
        layout = _FakeLayout(l_txt, n_aud=self.n_aud, n_vid=self.n_vid)
        rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, txt.device), txt.dtype)
        text_tags = (minimax_payload or {}).get("text_token_tags") if isinstance(minimax_payload, dict) else None
        if text_tags is not None:
            _tag_runs(0, l_txt, 0, text_tags)
        mod_segments = [
            (0, l_txt, 1),
            (l_txt, l_txt + self.n_aud, 2),
            (l_txt + self.n_aud, l_txt + self.n_aud + self.n_vid, 0),
        ]
        h = torch.cat([txt, self._aud_seed.to(dtype=txt.dtype), self._vid_seed.to(dtype=txt.dtype)], dim=0)
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
        aud_out = h[l_txt : l_txt + self.n_aud]
        vid_out = h[l_txt + self.n_aud :]
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

    # Check sections_by_modality too
    captured_main_len.clear()
    m_mod = qr.attach_query_router(p_spy.clone(), section_split="sections_by_modality", audio_block_band="2-3")
    stats_mod: list[dict[str, Any]] = []
    v_mod, a_mod = m_mod.run(ctx_sec_pre, payload=payload, stats_sink=stats_mod)
    assert torch.isfinite(v_mod).all() and torch.isfinite(a_mod).all()
    assert captured_main_len[0] == len(ids) - n_snd
    assert stats_mod[0]["aud_splices"] == 2

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

    manifest = json.loads((REPO / "bench" / "node_id_manifest.json").read_text(encoding="utf-8"))
    assert "MiniMaxH3QueryRouter" in manifest, "MiniMaxH3QueryRouter missing from bench/node_id_manifest.json"
    schema = qr.MiniMaxH3QueryRouter.define_schema()
    assert schema.node_id == "MiniMaxH3QueryRouter"
    assert [getattr(i, "id", None) for i in schema.inputs] == manifest["MiniMaxH3QueryRouter"]["inputs"]
    print("[ok] Test 5: Refusals and manifest/schema contract passed")


def main() -> int:
    _test_parity_against_audio_only_steer()
    _test_arbitrary_length_dual_prompt()
    _test_section_split()
    _test_mixed_bands()
    _test_refusals_and_schema()
    print("check_query_router: ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
