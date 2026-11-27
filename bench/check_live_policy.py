#!/usr/bin/env python3
"""Offline CPU verification gate for Beat 3B `steering/live_policy.py` (`H3LivePolicy` & `MiniMaxH3SelfMaskRouter`).

Verifies with zero CUDA and zero model weights:
  0. `resolve_phrase_token_indices` handles Qwen3-VL BPE `Ġ`/`▁` boundaries, multi-word phrases
     (`"olive oil"`), plural stems, and comma-separated multi-target lists.
  1. `synthesize_4d_attention_mask` + `expand_latent_mask_to_frames`: sparse background is exact `0.0`,
     peak/bbox align cleanly with ground-truth target quadrant, inversion is exact complement, and
     `SpatialVisualQRRuntime._downsample_mask_to_patches` round-trips expanded 4D masks bit-for-bit.
  2. Visual Thermostat (`evaluate_preview_frame`) & Acoustic Thermostat (`evaluate_audio_x0`):
     saturation/highlight-burn damping, low-motion freeze detection, silent dead-air & binaural
     stereo-collapse flags, and exact `40 Hz -> 24 fps` transient timestamps.
  3. End-to-end 2-step `text_phrase` `MiniMaxH3SelfMaskRouter` (`Step 0 harvest -> Step 1 SpatialQR`
     with `early_0_24_shield`):
       - Step 0 output is bitwise identical (`max_abs == 0.0`) to unpatched baseline while harvesting
         the synthetic target quadrant cleanly;
       - Step 1 routes `region_conditioning` exclusively into the harvested target quadrant while
         keeping zero-valued background shielded from region `V` perturbations;
       - `H3LivePolicy` records `harvest` + `routed` step history and `audio_x0` telemetry untouched.
  4. End-to-end `left_speaker` / `right_speaker` / `active_speaker` zero-coordinate speaker-mask
     routing via `Q_vid @ K_aud[t]^T` on joint AV layout.
  5. `MiniMax-H3-Image` (`latent_t=1, audio_t=0`) parity across `text_phrase` (full Step-0 harvest ->
     Step-1 route) and `active_speaker` (safe no-op passthrough with zero crash).
  6. Multi-run `ON_CLEANUP` reset re-harvests fresh Step-0 masks on Run 2 while keeping Run 1
     report inspectable after completion; pure `uncond` CFG passes skip cleanly.
  7. Attachment refusal guards (double-stacking, stacking with `SpatialQueryRouter`,
     `FrozenVideoCache`, `BlockSparseAttention`, `SageAttention`, missing `target_phrase`,
     non-hookable model, and `MiniMaxH3SelfMaskRouter.define_schema()` validation).
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
REPO_DIR = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _lib import bootstrap

bootstrap(cpu=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.minimax.model import PackedLayout
from comfy.ldm.modules.attention import AttentionTensorContainer
from comfy.patcher_extension import CallbacksMP, WrappersMP

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from steering import live_policy as lp
from steering import spatial_query_router as sqr


# ------------------------------------------------------------------ synthetic H3 test harness


class _MockAttn(nn.Module):
    def __init__(self, hidden: int = 32, heads: int = 4, target_q_bias: torch.Tensor | None = None):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv_proj = nn.Linear(hidden, hidden * 3, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.custom_qkv = target_q_bias

    def forward(self, x: torch.Tensor, rope_freqs=None, transformer_options=None):
        opts = transformer_options or {}
        s = x.shape[0]
        if self.custom_qkv is not None:
            q, k, v = self.custom_qkv(x, opts)
        else:
            q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
            q = self.q_norm(q.view(s, self.heads, self.head_dim)).transpose(0, 1).unsqueeze(0)
            k = self.k_norm(k.view(s, self.heads, self.head_dim)).transpose(0, 1).unsqueeze(0)
            v = v.view(s, self.heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        q_c = AttentionTensorContainer(q)
        k_c = AttentionTensorContainer(k)
        v_c = AttentionTensorContainer(v)

        def _base_sdpa(qq, kk, vv, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kw):
            qq_t = qq if torch.is_tensor(qq) else qq.take()
            kk_t = kk if torch.is_tensor(kk) else kk.take()
            vv_t = vv if torch.is_tensor(vv) else vv.take()
            out = F.scaled_dot_product_attention(qq_t, kk_t, vv_t, is_causal=False)
            if skip_output_reshape:
                return out
            return out[0].transpose(0, 1).reshape(1, out.shape[2], self.heads * self.head_dim)

        override = opts.get("optimized_attention_override")
        if override is not None:
            out = override(
                _base_sdpa, q_c, k_c, v_c, self.heads,
                mask=None, skip_reshape=True, skip_output_reshape=False, transformer_options=opts,
            )
        else:
            out = _base_sdpa(q_c, k_c, v_c, self.heads, skip_reshape=True, skip_output_reshape=False)
        return self.out_proj(out.squeeze(0))


class _MockBlock(nn.Module):
    def __init__(self, hidden: int = 32, heads: int = 4, custom_qkv=None):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, eps=1e-5)
        self.norm2 = nn.RMSNorm(hidden, eps=1e-5)
        self.attn = _MockAttn(hidden, heads, target_q_bias=custom_qkv)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden, bias=False), nn.SiLU(), nn.Linear(hidden, hidden, bias=False))
        self._hidden = hidden

    def adaln_proj(self, t_emb: torch.Tensor):
        m = int(t_emb.shape[0]) * 3
        zeros = torch.zeros((m, self._hidden), device=t_emb.device, dtype=t_emb.dtype)
        return zeros, zeros, zeros + 0.25, zeros, zeros, zeros + 0.25

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        opts = transformer_options or {}
        h = self.norm1(x)
        x = x + 0.25 * self.attn(h, rope_freqs=rope_freqs, transformer_options=opts)
        x = x + 0.25 * self.mlp(self.norm2(x))
        return x


class _MockDM(nn.Module):
    def __init__(self, n_blocks: int = 4, hidden: int = 32, heads: int = 4, is_image: bool = False, custom_qkv=None):
        super().__init__()
        self.hidden_size = hidden
        self.blocks = nn.ModuleList([_MockBlock(hidden, heads, custom_qkv=custom_qkv) for _ in range(n_blocks)])
        self.condition_proj = nn.Linear(hidden, hidden, bias=False)
        self.token_refiner = lambda x, transformer_options=None: x
        if not is_image:
            self.audio_patch_proj = nn.Linear(32, hidden)

    def _forward(self, x, timestep, context, transformer_options=None, **kwargs):
        opts = dict(transformer_options or {})
        h = kwargs["img"].clone()
        layout = kwargs["layout"]
        t_emb = kwargs["t_emb"]
        mod_segments = kwargs["mod_segments"]
        dit_rep = opts.get("patches_replace", {}).get("dit", {})
        for b_idx, blk in enumerate(self.blocks):
            opts["block_index"] = b_idx
            args_d = {
                "img": h,
                "layout": layout,
                "t_emb": t_emb,
                "mod_segments": mod_segments,
                "rope_freqs": None,
                "transformer_options": opts,
            }
            rep = dit_rep.get(("double_block", b_idx))
            if rep is not None:
                res = rep(
                    args_d,
                    {
                        "original_block": lambda a, _b=blk: {
                            "img": _b(a["img"], a["t_emb"], a["mod_segments"], a["rope_freqs"], a["transformer_options"])
                        }
                    },
                )
                h = res["img"]
            else:
                h = blk(h, t_emb, mod_segments, None, opts)
        return h.unsqueeze(0)


class _MiniPatcher:
    def __init__(self, dm: _MockDM, latent_shapes=None):
        self.model = types.SimpleNamespace(diffusion_model=dm, latent_shapes=latent_shapes)
        self.model_options: dict = {"transformer_options": {}}
        self.wrappers: dict = {}
        self.callbacks: dict = {}
        self.object_patches: dict = {}
        self._post_cfg_fns: list = []

    def clone(self) -> "_MiniPatcher":
        c = _MiniPatcher(self.model.diffusion_model, latent_shapes=self.model.latent_shapes)
        t_opts = dict(self.model_options.get("transformer_options", {}))
        pr = t_opts.get("patches_replace", {})
        if pr:
            t_opts["patches_replace"] = {k: dict(v) for k, v in pr.items()}
        c.model_options = {**self.model_options, "transformer_options": t_opts}
        c.wrappers = {k: {wk: list(wv) for wk, wv in v.items()} for k, v in self.wrappers.items()}
        c.callbacks = {k: {ck: list(cv) for ck, cv in v.items()} for k, v in self.callbacks.items()}
        c.object_patches = dict(self.object_patches)
        c._post_cfg_fns = list(self._post_cfg_fns)
        return c

    def set_model_patch_replace(self, patch, name: str, key: str, index: int) -> None:
        t_opts = self.model_options.setdefault("transformer_options", {})
        t_opts.setdefault("patches_replace", {}).setdefault(name, {})[(key, index)] = patch

    def add_callback_with_key(self, call_type, key: str, callback) -> None:
        self.callbacks.setdefault(call_type, {}).setdefault(key, []).append(callback)

    def set_model_sampler_post_cfg_function(self, fn) -> None:
        self._post_cfg_fns.append(fn)

    def add_wrapper_with_key(self, wrapper_type, key: str, fn) -> None:
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(fn)

    def get_wrappers(self, wrapper_type, key: str | None = None):
        bucket = self.wrappers.get(wrapper_type, {})
        if key is not None:
            return list(bucket.get(key, ()))
        out = []
        for vals in bucket.values():
            out.extend(vals)
        return out

    def trigger_cleanup(self) -> None:
        for fns in self.callbacks.get(CallbacksMP.ON_CLEANUP, {}).values():
            for fn in fns:
                fn(self)

    def run_step(
        self,
        ctx: torch.Tensor,
        layout: PackedLayout,
        img: torch.Tensor,
        t_emb: torch.Tensor,
        mod_segments,
        *,
        sigma: float = 1.0,
        denoised=None,
        cond_or_uncond=None,
    ):
        dm = self.model.diffusion_model
        to = dict(self.model_options.get("transformer_options", {}))
        to["sigmas"] = torch.tensor([sigma])
        to["sample_sigmas"] = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
        if cond_or_uncond is not None:
            to["cond_or_uncond"] = cond_or_uncond

        def _base_exec(x, timestep, context, transformer_options, **kw):
            return dm._forward(x, timestep, context, transformer_options, **kw)

        exec_fn = _base_exec
        dm_w = self.wrappers.get(WrappersMP.DIFFUSION_MODEL, {})
        for wlist in reversed(list(dm_w.values())):
            for w in reversed(wlist):
                prev = exec_fn
                exec_fn = functools.partial(w, prev)

        out = exec_fn(
            [torch.zeros(1), torch.zeros(1)],
            torch.tensor([sigma * 1000.0]),
            ctx,
            to,
            img=img,
            layout=layout,
            t_emb=t_emb,
            mod_segments=mod_segments,
        )
        den_out = denoised
        if denoised is not None:
            for fn in self._post_cfg_fns:
                den_out = fn({"denoised": den_out, "sigma": torch.tensor([sigma]), "model": self.model})
        return out, den_out


def _make_planted_qkv(layout: PackedLayout, *, target_tok_idx: int, quad: str = "bottom_right"):
    """Construct deterministic Q/K/V where video patches in `quad` align sharply with `target_tok_idx` & stereo audio."""
    segs = layout.segments
    va, vb = next((a, b) for (a, b, k) in segs if k == "video")
    au = [(a, b) for (a, b, k) in segs if k == "audio"]
    _tl, lt, lh, lw, at = layout.signature
    ph, pw = lh // 2, lw // 2

    def _fn(x: torch.Tensor, opts: dict):
        s = x.shape[0]
        h, d = 4, 8
        q = torch.zeros((1, h, s, d), dtype=x.dtype, device=x.device)
        k = torch.zeros((1, h, s, d), dtype=x.dtype, device=x.device)
        v = x.view(1, s, h, d).transpose(1, 2).contiguous()

        # Plant orthogonal basis vectors:
        #   dim 0 -> text noun target in `quad`
        #   dim 1 -> left audio speaker (left half)
        #   dim 2 -> right audio speaker (right half)
        k[:, :, target_tok_idx, 0] = 6.0
        if au:
            aa, _ab = au[0]
            k[:, :, aa : aa + at, 1] = 6.0
            k[:, :, aa + at : aa + 2 * at, 2] = 6.0

        for kt in range(lt):
            for y in range(ph):
                for xx in range(pw):
                    p = va + kt * (ph * pw) + y * pw + xx
                    in_q = (
                        (xx >= pw // 2 and y >= ph // 2)
                        if quad == "bottom_right"
                        else (xx < pw // 2 and y < ph // 2)
                    )
                    if in_q:
                        q[:, :, p, 0] = 6.0
                    if xx < pw // 2:
                        q[:, :, p, 1] = 6.0
                    else:
                        q[:, :, p, 2] = 6.0
        return q, k, v

    return _fn


def main() -> int:
    # 0. BPE phrase token matching
    pieces = ["The", "Ġwoven", "Ġbasket", "Ġholds", "Ġfresh", "Ġbread", ",", "Ġgolden", "Ġolive", "Ġoil", ",", "Ġand", "Ġchees", "es", "."]
    assert lp.resolve_phrase_token_indices(pieces, "bread") == [5]
    assert lp.resolve_phrase_token_indices(pieces, "olive oil") == [8, 9]
    assert lp.resolve_phrase_token_indices(pieces, "cheese") == [12, 13]
    assert lp.resolve_phrase_token_indices(pieces, "bread, olive oil") == [5, 8, 9]
    assert lp.resolve_phrase_token_indices(pieces, "nonexistent") == []
    print("[ok] 0. resolve_phrase_token_indices handles BPE boundaries, multi-word phrases & plural stems")

    # 1. 4D mask synthesis + sparse zero background + lossless SpatialQR downsample round-trip
    raw = torch.zeros((6, 8, 12), dtype=torch.float32)
    raw[:, 4:8, 6:12] = 8.0  # bottom-right quadrant
    m_thw, stats = lp.synthesize_4d_attention_mask(raw, threshold_z=0.9, softness=0.25, dilation_passes=0, spatial_blur=False)
    assert m_thw.shape == (6, 8, 12)
    assert float(m_thw[:, 0:3, 0:5].max().item()) == 0.0, "Background must be exact 0.0 for sparse *_shield"
    assert float(m_thw[:, 4:8, 6:12].min().item()) > 0.95
    assert stats["peak_xy"][0] >= 0.5 and stats["peak_xy"][1] >= 0.5
    assert stats["bbox_xyxy"][0] >= 0.45 and stats["bbox_xyxy"][1] >= 0.45
    assert stats["sparse_zero_frac"] > 0.60

    m_inv, _ = lp.synthesize_4d_attention_mask(raw, threshold_z=0.9, softness=0.25, dilation_passes=0, spatial_blur=False, invert=True)
    assert float(((m_thw + m_inv) - 1.0).abs().max().item()) < 1e-5

    expanded = lp.expand_latent_mask_to_frames(m_thw)
    # 6 latent steps -> frames 1+4+4+4+4+1 = 18
    assert expanded.shape == (18, 8, 12)
    recovered = sqr._downsample_mask_to_patches(expanded, latent_t=6, patch_h=8, patch_w=12)
    assert float((recovered - m_thw.reshape(-1)).abs().max().item()) < 1e-6, "Expanded mask must round-trip losslessly"

    # 1b. Cross-time silent-pause scale floor (`T > 1` silences background noise; `T == 1` bit-identical)
    torch.manual_seed(99)
    raw_pause = torch.randn((8, 8, 12), dtype=torch.float32) * 0.12
    raw_pause[0:3, 4:8, 6:12] += 8.0  # speaker active on frames 0..2, silent on frames 3..7
    m_pause_on, st_pause = lp.synthesize_4d_attention_mask(
        raw_pause, threshold_z=1.35, softness=0.25, dilation_passes=0,
        spatial_blur=False, temporal_smooth=False, cross_time_scale_floor=0.55,
    )
    m_pause_off, _ = lp.synthesize_4d_attention_mask(
        raw_pause, threshold_z=1.35, softness=0.25, dilation_passes=0,
        spatial_blur=False, temporal_smooth=False, cross_time_scale_floor=0.0,
    )
    assert float(m_pause_on[0:3, 4:8, 6:12].min().item()) > 0.90, "Active frames must stay lit"
    assert float(m_pause_on[3:8].max().item()) == 0.0, f"Silent frames must be exact 0.0, got {m_pause_on[3:8].max()}"
    assert float(m_pause_off[3:8].max().item()) > 0.50, "Unfloored control should amplify silent-frame noise"
    assert st_pause["silent_frames_count"] == 5 and st_pause["frame_hard_coverage_min"] == 0.0
    assert st_pause["frame_hard_coverage_max"] > 0.20

    raw_t1 = raw_pause[0:1]
    m_t1_a, _ = lp.synthesize_4d_attention_mask(raw_t1, cross_time_scale_floor=0.55)
    m_t1_b, _ = lp.synthesize_4d_attention_mask(raw_t1, cross_time_scale_floor=0.0)
    assert float((m_t1_a - m_t1_b).abs().max().item()) == 0.0, "T==1 H3-Image must be bit-identical"
    print("[ok] 1. synthesize_4d_attention_mask sparse-zero, silent-pause floor & SpatialQR round-trip verified")

    # 2. Visual & Acoustic Thermostats
    calm = np.full((64, 64, 3), 120, dtype=np.uint8)
    burn = np.zeros((64, 64, 3), dtype=np.uint8)
    burn[..., 0] = 255  # pure saturated red + highlight clip
    ev_calm = lp.evaluate_preview_frame(calm)
    ev_freeze = lp.evaluate_preview_frame(calm, prev_rgb_hwc=calm)
    ev_burn = lp.evaluate_preview_frame(burn, prev_rgb_hwc=calm)
    assert ev_calm["flags"] == [] and ev_calm["recommended_strength_scale"] == 1.0
    assert "low_motion_freeze_risk" in ev_freeze["flags"]
    assert "high_saturation" in ev_burn["flags"] and "highlight_burn" in ev_burn["flags"]
    assert ev_burn["recommended_strength_scale"] < 0.75

    aud_quiet = torch.zeros((1, 32, 2, 80))
    ev_q = lp.evaluate_audio_x0(aud_quiet, binaural_expected=True)
    assert "silent_dead_air" in ev_q["flags"]

    torch.manual_seed(7)
    aud_mono = torch.randn((1, 32, 1, 80)).expand(1, 32, 2, 80).clone() * 1.5
    aud_mono[:, :, :, 20] *= 5.0  # t = 0.50 s -> f12
    ev_m = lp.evaluate_audio_x0(aud_mono, z_thresh=2.0, min_gap_steps=6, binaural_expected=True)
    assert "stereo_collapse_risk" in ev_m["flags"]
    assert any(t["a_idx"] == 20 and t["video_frame_24fps"] == 12 for t in ev_m["transients"]), ev_m
    print("[ok] 2. Visual & Acoustic Thermostats (burn, freeze, dead-air, stereo collapse, peaks) verified")

    # 3. End-to-end 2-step `text_phrase` Self-Mask Router (Step 0 harvest -> Step 1 early_0_24_shield)
    torch.manual_seed(11)
    layout_av = PackedLayout(text_len=10, latent_t=2, latent_h=8, latent_w=12, audio_t=10)
    s_tot = layout_av.seq_len
    img = torch.randn(s_tot, 32)
    ctx = torch.randn(1, 10, 32)
    t_emb = torch.randn(2, 32)
    mod_segs = [(0, s_tot, 0)]
    qkv_fn = _make_planted_qkv(layout_av, target_tok_idx=5, quad="bottom_right")

    v_shape, a_shape = (1, 24, 2, 8, 12), (1, 32, 2, 10)
    import comfy.utils
    packed_x0 = comfy.utils.pack_latents([torch.randn(*v_shape), torch.randn(*a_shape)])[0]

    base_dm = _MockDM(n_blocks=4, custom_qkv=qkv_fn)
    base_p = _MiniPatcher(base_dm, latent_shapes=[v_shape, a_shape])
    reg_cond = [[torch.randn(1, 7, 32) * 3.0, {}]]
    tok_pieces = ["The", "Ġtable", "Ġhas", "Ġa", "Ġwarm", "Ġbread", "Ġloaf", "Ġand", "Ġwood", "."]

    routed_p, pol = lp.attach_self_mask_router(
        base_p,
        region_conditioning=reg_cond,
        mask_source="text_phrase",
        target_phrase="bread",
        token_pieces=tok_pieces,
        threshold_z=0.8,
        softness=0.25,
        dilation_passes=0,
        isolation="early_0_24_shield",
        audio_zone="base",
        harvest_blocks=(2, 3),
    )

    # Step 0: bitwise parity with unpatched baseline, AND mask harvested in bottom-right!
    out0_ref, _ = base_p.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=1.0, denoised=packed_x0)
    out0_sm, den0 = routed_p.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=1.0, denoised=packed_x0)
    assert float((out0_ref - out0_sm).abs().max().item()) == 0.0, "Step-0 harvest must not mutate output!"
    assert den0 is packed_x0, "post_cfg thermostat must return denoised tensor untouched"
    assert pol.harvested_mask_thw is not None and pol.harvested_mask_frames is not None
    assert pol.mask_stats["peak_xy"][0] >= 0.5 and pol.mask_stats["peak_xy"][1] >= 0.5, pol.mask_stats
    assert len(pol.audio_x0_history) == 1 and pol.audio_x0_history[0]["step"] == 0

    # Step 1: routes region_conditioning strictly into the harvested bottom-right mask!
    out1_ref, _ = base_p.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=0.8, denoised=packed_x0)
    out1_sm, _ = routed_p.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=0.8, denoised=packed_x0)
    assert torch.isfinite(out1_sm).all()
    va, vb = next((a, b) for (a, b, k) in layout_av.segments if k == "video")
    diff_vid = (out1_sm[0, va:vb] - out1_ref[0, va:vb]).abs().mean(dim=-1).view(2, 4, 6)
    diff_br = float(diff_vid[:, 2:4, 3:6].mean().item())
    diff_tl = float(diff_vid[:, 0:2, 0:3].mean().item())
    assert diff_br > diff_tl * 3.0 + 1e-4, f"Step-1 route did not focus on bottom-right! br={diff_br}, tl={diff_tl}"
    assert [h["phase"] for h in pol.step_history] == ["harvest", "routed"]
    print(f"[ok] 3. End-to-end text_phrase Step-0 harvest (0.0 diff) -> Step-1 routed (br={diff_br:.4f} >> tl={diff_tl:.4f}) verified")

    # 3b. `replay_step0=True` (`1.17x` mode): Step 0 harvests & immediately replays routed on Step 0!
    rep_p, pol_rep = lp.attach_self_mask_router(
        base_p,
        region_conditioning=reg_cond,
        mask_source="text_phrase",
        target_phrase="bread",
        token_pieces=tok_pieces,
        threshold_z=0.8,
        softness=0.25,
        dilation_passes=0,
        isolation="mid_20_34_shield",
        audio_zone="base",
        replay_step0=True,
        harvest_blocks=(2, 3),
    )
    out0_rep, _ = rep_p.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=1.0, denoised=packed_x0)
    assert torch.isfinite(out0_rep).all()
    diff_rep0 = (out0_rep[0, va:vb] - out0_ref[0, va:vb]).abs().mean(dim=-1).view(2, 4, 6)
    diff_rep0_br = float(diff_rep0[:, 2:4, 3:6].mean().item())
    diff_rep0_tl = float(diff_rep0[:, 0:2, 0:3].mean().item())
    assert diff_rep0_br > diff_rep0_tl * 3.0 + 1e-4, (
        f"replay_step0 did not route Step 0 onto target! br={diff_rep0_br}, tl={diff_rep0_tl}"
    )
    assert [h["phase"] for h in pol_rep.step_history] == ["harvest", "routed"]
    assert pol_rep.step_history[1].get("replayed_step0") is True
    assert len(pol_rep.audio_x0_history) == 1 and pol_rep.audio_x0_history[0]["step"] == 0
    # Step 1 executes normal single-pass routed (no extra harvest)
    out1_rep, _ = rep_p.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=0.8, denoised=packed_x0)
    assert torch.isfinite(out1_rep).all()
    assert [h["phase"] for h in pol_rep.step_history] == ["harvest", "routed", "routed"]
    assert not pol_rep.step_history[2].get("replayed_step0")
    # Cleanup resets cleanly so Run 2 harvests & replays Step 0 again
    rep_p.trigger_cleanup()
    rep_p.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=1.0)
    assert [h["phase"] for h in pol_rep.step_history[-2:]] == ["harvest", "routed"]
    assert pol_rep.step_history[-1].get("replayed_step0") is True
    print(f"[ok] 3b. replay_step0=True (1.17x) Step-0 harvest+replay (br={diff_rep0_br:.4f} >> tl={diff_rep0_tl:.4f}) & cleanup verified")

    # 4. Speaker-spotlight mask sources (`left_speaker` vs `right_speaker` vs `active_speaker`)
    for spk_src, expect_x_lo, expect_x_hi in (
        ("left_speaker", 0.0, 0.5),
        ("right_speaker", 0.5, 1.0),
        ("active_speaker", 0.0, 1.0),
    ):
        p_spk, pol_spk = lp.attach_self_mask_router(
            base_p,
            region_conditioning=reg_cond,
            mask_source=spk_src,
            threshold_z=0.5,
            softness=0.25,
            dilation_passes=0,
            isolation="early_0_24_shield",
            harvest_blocks=(2, 3),
        )
        p_spk.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=1.0)
        assert pol_spk.harvested_mask_thw is not None
        cx = pol_spk.mask_stats["centroid_xy"][0]
        assert expect_x_lo <= cx <= expect_x_hi, (spk_src, pol_spk.mask_stats)
        out_spk1, _ = p_spk.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=0.8)
        assert torch.isfinite(out_spk1).all()
    print("[ok] 4. Zero-coordinate speaker spotlight (`left_speaker`, `right_speaker`, `active_speaker`) verified")

    # 5. MiniMax-H3-Image (`latent_t=1, audio_t=0`) parity + safe audio no-op
    layout_img = PackedLayout(text_len=10, latent_t=1, latent_h=8, latent_w=12, audio_t=0)
    img_t1 = torch.randn(layout_img.seq_len, 32)
    mod_t1 = [(0, layout_img.seq_len, 0)]
    qkv_img = _make_planted_qkv(layout_img, target_tok_idx=5, quad="top_left")
    p_img = _MiniPatcher(_MockDM(n_blocks=4, is_image=True, custom_qkv=qkv_img), latent_shapes=[(1, 24, 1, 8, 12)])
    p_img_r, pol_img = lp.attach_self_mask_router(
        p_img,
        region_conditioning=reg_cond,
        mask_source="text_phrase",
        target_phrase="bread",
        token_pieces=tok_pieces,
        threshold_z=0.8,
        softness=0.25,
        dilation_passes=0,
        isolation="early_0_24_shield",
        harvest_blocks=(2, 3),
    )
    out_i0, _ = p_img_r.run_step(ctx, layout_img, img_t1, t_emb, mod_t1, sigma=1.0, denoised=torch.randn(1, 24, 1, 8, 12))
    out_i1, _ = p_img_r.run_step(ctx, layout_img, img_t1, t_emb, mod_t1, sigma=0.8)
    assert torch.isfinite(out_i0).all() and torch.isfinite(out_i1).all()
    assert pol_img.mask_stats["peak_xy"][0] < 0.5 and pol_img.mask_stats["peak_xy"][1] < 0.5
    assert len(pol_img.audio_x0_history) == 0

    # Speaker mode on image model cleanly passes through without crashing
    p_img_spk, pol_img_spk = lp.attach_self_mask_router(
        p_img, region_conditioning=reg_cond, mask_source="active_speaker", harvest_blocks=(2, 3)
    )
    out_is0, _ = p_img_spk.run_step(ctx, layout_img, img_t1, t_emb, mod_t1, sigma=1.0)
    assert pol_img_spk.harvested_mask_frames is None and torch.isfinite(out_is0).all()
    print("[ok] 5. MiniMax-H3-Image (latent_t=1, audio_t=0) text_phrase route & speaker no-op verified")

    # 6. Multi-run ON_CLEANUP re-harvest + pure uncond CFG skip
    routed_p.trigger_cleanup()
    # Report from Run 1 must remain intact immediately after cleanup!
    assert pol.harvested_mask_thw is not None and len(pol.to_report()["step_history"]) == 2
    # Pure uncond CFG call must skip cleanly without triggering harvest
    out_unc_ref, _ = base_p.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=1.0, cond_or_uncond=[1])
    out_unc_sm, _ = routed_p.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=1.0, cond_or_uncond=[1])
    assert float((out_unc_ref - out_unc_sm).abs().max().item()) == 0.0
    # Run 2 Step 0 triggers a fresh harvest (`phase == "harvest"` again)
    routed_p.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=1.0, cond_or_uncond=[0])
    assert pol.step_history[-1]["phase"] == "harvest"
    print("[ok] 6. Multi-run ON_CLEANUP re-harvest & pure-uncond CFG skip verified")

    # 7. Refusal guards & V3 schema check
    def _expect_err(fn, sub: str):
        try:
            fn()
        except ValueError as exc:
            assert sub.lower() in str(exc).lower(), f"Expected {sub!r} in {exc!r}"
            return
        raise AssertionError(f"Expected ValueError containing {sub!r}")

    _expect_err(
        lambda: lp.attach_self_mask_router(base_p, region_conditioning=reg_cond, mask_source="text_phrase", target_phrase=""),
        "target_phrase",
    )
    _expect_err(
        lambda: lp.attach_self_mask_router(routed_p, region_conditioning=reg_cond, mask_source="active_speaker"),
        "cannot stack",
    )
    p_sqr = sqr.attach_spatial_query_router(base_p, region_conditioning=reg_cond)
    _expect_err(
        lambda: lp.attach_self_mask_router(p_sqr, region_conditioning=reg_cond, mask_source="active_speaker"),
        "cannot stack",
    )
    p_sage = base_p.clone()
    p_sage.object_patches["diffusion_model.blocks.0.attn.forward"] = lambda *a, **k: None
    _expect_err(
        lambda: lp.attach_self_mask_router(p_sage, region_conditioning=reg_cond, mask_source="active_speaker"),
        "object-patched attention",
    )

    schema = lp.MiniMaxH3SelfMaskRouter.define_schema()
    assert schema.node_id == "MiniMaxH3SelfMaskRouter"
    input_ids = [inp.id for inp in schema.inputs]
    assert input_ids[:4] == ["model", "region_conditioning", "mask_source", "target_phrase"]
    assert input_ids[-2:] == ["replay_step0", "step0_critic"] and len(input_ids) == 16
    assert getattr(schema.inputs[1], "optional", False) is True, "region_conditioning must be optional=True"
    assert getattr(schema.inputs[-1], "optional", False) is True, "step0_critic must be optional=True"
    print(f"[ok] 7. All refusal guards & MiniMaxH3SelfMaskRouter schema ({len(input_ids)} inputs) verified")

    # 8. `audio_transients` Step-0 burst mask (with 2-step smooth decay) + optional `region_conditioning=None` + `SelfMask -> 2x EditGuider` bridge
    torch.manual_seed(21)
    aud_burst = torch.randn((1, 32, 2, 36)) * 0.6
    aud_burst[:, :, :, 14] *= 7.5  # strong transient at a_idx=14 -> latent step 2 (windows: [0,2), [1,9), [8,16), [15,22)...)
    m_tr, st_tr = lp.synthesize_audio_transient_mask(aud_burst, latent_t=6, patch_h=4, patch_w=6, z_thresh=2.2)
    assert m_tr.shape == (6, 4, 6)
    assert float(m_tr[2].mean().item()) > 0.85, st_tr
    assert 0.50 <= float(m_tr[3].mean().item()) <= 0.65, f"Expected step+1 hold tail ~0.58, got {m_tr[3].mean().item()}"
    assert 0.20 <= float(m_tr[4].mean().item()) <= 0.35, f"Expected step+2 soft decay ~0.28, got {m_tr[4].mean().item()}"
    assert float(m_tr[0].mean().item()) == 0.0 and float(m_tr[5].mean().item()) == 0.0, st_tr
    assert st_tr["active_latent_steps"] == [2, 3]
    # Impulse mode (`decay_steps=0`) isolates strictly step 2
    m_tr0, st_tr0 = lp.synthesize_audio_transient_mask(aud_burst, latent_t=6, patch_h=4, patch_w=6, z_thresh=2.2, decay_steps=0)
    assert st_tr0["active_latent_steps"] == [2] and float(m_tr0[3].mean().item()) == 0.0

    # Sensor-only attach (`region_conditioning=None`) + `audio_transients` + `m._h3_live_policy` + clone propagation
    p_tr, pol_tr = lp.attach_self_mask_router(
        base_p,
        region_conditioning=None,
        mask_source="audio_transients",
        harvest_blocks=(2, 3),
        emit_ws_events=False,
    )
    assert getattr(p_tr, "_h3_live_policy", None) is pol_tr
    assert p_tr.clone().model_options["transformer_options"]["_h3_live_policy"] is pol_tr
    aud_burst10 = torch.randn((1, 32, 2, 10)) * 0.5
    aud_burst10[:, :, :, 6] *= 8.0  # transient inside latent step 1 window [1, 9)
    packed_burst10 = comfy.utils.pack_latents([torch.randn(1, 24, 2, 8, 12), aud_burst10])[0]
    out_tr0, _ = p_tr.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=1.0, denoised=packed_burst10)
    assert float((out_tr0 - out0_ref).abs().max().item()) == 0.0
    assert pol_tr.harvested_mask_frames is not None and pol_tr.harvested_mask_frames.shape == (5, 4, 6)
    assert float(pol_tr.harvested_mask_frames[1:].mean().item()) > 0.80
    assert float(pol_tr.harvested_mask_frames[0].mean().item()) == 0.0
    # Step 1 skips Phase-2 SpatialQR cleanly when region_conditioning is None
    out_tr1, _ = p_tr.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=0.8)
    assert float((out_tr1 - out1_ref).abs().max().item()) == 0.0

    # `EditGuider._effective_raw_mask()` falls back to `model_patcher._h3_live_policy.harvested_mask_frames`
    import edit_guidance as eg

    eg_guider = eg.EditGuider.__new__(eg.EditGuider)
    eg_guider.model_patcher = p_tr
    eg_guider.set_edit(scale=1.6, first_step=1, align_positions=True, last_step=3, streams="video_only", mask=None)
    eff_m = eg_guider._effective_raw_mask()
    assert eff_m is not None and eff_m.shape == (5, 4, 6)
    v_m_eg, a_m_eg = eg_guider._prepare_masks_for_step((1, 24, 2, 8, 12), (1, 32, 2, 10), torch.device("cpu"), torch.float32)
    assert v_m_eg is not None and v_m_eg.shape == (1, 1, 2, 8, 12)
    assert float(v_m_eg[0, 0, 1].mean().item()) > 0.80 and float(v_m_eg[0, 0, 0].mean().item()) == 0.0
    assert a_m_eg is not None
    # Explicit `h3_edit_pass` skips Phase-2 SpatialQR even when `region_conditioning` is wired!
    p_ed, pol_ed = lp.attach_self_mask_router(
        base_p,
        region_conditioning=reg_cond,
        mask_source="text_phrase",
        target_phrase="bread",
        token_pieces=tok_pieces,
        threshold_z=0.8,
        replay_step0=True,
        harvest_blocks=(2, 3),
        emit_ws_events=False,
    )
    dm_w = p_ed.wrappers[WrappersMP.DIFFUSION_MODEL][lp.SELF_MASK_WRAPPER_KEY][0]
    to_ed = dict(p_ed.model_options["transformer_options"])
    to_ed["sigmas"] = torch.tensor([1.0])
    to_ed["sample_sigmas"] = torch.tensor([1.0, 0.8, 0.0])
    out_ed0 = dm_w(
        lambda x_, t_, c_, to_, **kw_: base_dm._forward(x_, t_, c_, to_, **kw_),
        [torch.zeros(1), torch.zeros(1)], torch.tensor([1000.0]), ctx, to_ed,
        img=img, layout=layout_av, t_emb=t_emb, mod_segments=mod_segs,
        minimax_payload={"h3_edit_pass": "original"},
    )
    assert float((out_ed0 - out0_ref).abs().max().item()) == 0.0, "EditGuider Step-0 original must harvest without replay!"
    assert pol_ed.harvested_mask_frames is not None
    out_ed1 = dm_w(
        lambda x_, t_, c_, to_, **kw_: base_dm._forward(x_, t_, c_, to_, **kw_),
        [torch.zeros(1), torch.zeros(1)], torch.tensor([800.0]), ctx, to_ed,
        img=img, layout=layout_av, t_emb=t_emb, mod_segments=mod_segs,
        minimax_payload={"h3_edit_pass": "edited"},
    )
    assert float((out_ed1 - out1_ref).abs().max().item()) == 0.0, "EditGuider Step-1 edited must skip Phase-2 SpatialQR!"
    print("[ok] 8. audio_transients, optional region_conditioning=None & SelfMask -> 2x EditGuider bridge verified")

    # 9. Closed-loop H3LivePolicy actuators: KJ preview bridge + Steer/ReleaseRefs + QueryRouter a2v=0.0 clamp
    dim_frame = np.full((32, 32, 3), 40, dtype=np.uint8)  # mean_luma ~ 0.157 -> dim_understeer_risk (+1.12x)
    ev_dim = lp.evaluate_preview_frame(dim_frame)
    assert "dim_understeer_risk" in ev_dim["flags"] and ev_dim["recommended_strength_scale"] > 1.05

    class _MockKJWrapper:
        def __init__(self):
            self.frames = []

    kj_mock = _MockKJWrapper()
    p_tr.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "kj_preview_override", kj_mock)
    bridge_fn = p_tr.get_wrappers(WrappersMP.OUTER_SAMPLE, f"{lp.SELF_MASK_WRAPPER_KEY}_preview_bridge")[0]

    class _MockOuterExec:
        def __init__(self, mp):
            self.class_obj = types.SimpleNamespace(model_patcher=mp)

        def __call__(self, noise, lat_i, sampler, sigmas, denoise_mask, cb, disable_pbar, seed, **kw):
            # Step 0: dim frame; Step 1: identical static dim frame -> freeze risk -> recommended_video_end_percent = 0.33!
            kj_mock.frames.append(dim_frame)
            cb(0, None, None, 6)
            kj_mock.frames.append(dim_frame)
            cb(1, None, None, 6)
            return lat_i

    pol_tr.reset_for_new_run()
    bridge_fn(_MockOuterExec(p_tr), None, None, None, None, None, None, True, 42)
    assert len(pol_tr.preview_history) == 2
    assert pol_tr.strength_scale > 1.05, f"Dim preview should boost strength_scale, got {pol_tr.strength_scale}"
    # Single-frame KJ preview MUST NOT falsely set recommended_video_end_percent from step-to-step delta!
    assert pol_tr.recommended_video_end_percent is None, (
        f"Single-frame preview must not trigger copy-freeze governor, got {pol_tr.recommended_video_end_percent}"
    )
    # Multi-frame `frames_batch` DOES trigger 0.33 anti-copy-freeze governor when temporal motion < 1.8!
    pol_tr.update_from_preview(dim_frame, step=1, frames_batch=[dim_frame, dim_frame, dim_frame])
    assert pol_tr.recommended_video_end_percent == 0.33, (
        f"Static multi-frame batch should trigger 0.33 anti-copy-freeze governor, got {pol_tr.recommended_video_end_percent}"
    )

    # Verify `steering/query_router.py` respects `policy.recommended_aud_to_vid_gain = 0.0`!
    from steering import query_router as qr_mod

    pol_hiss = lp.H3LivePolicy(audio_zone="binaural_split", emit_ws_events=False)
    pol_hiss.update_from_audio_x0(aud_mono, step=1, sigma=0.97)
    assert pol_hiss.recommended_aud_to_vid_gain == 0.0, pol_hiss.to_report()
    p_for_qr = base_p.clone()
    p_for_qr._h3_live_policy = pol_hiss
    p_for_qr.model_options["transformer_options"][lp.LIVE_POLICY_KEY] = pol_hiss
    p_qr = qr_mod.attach_query_router(p_for_qr, section_split="off", aud_to_vid_gain=1.0)
    captured_stats: list[dict] = []
    qr_w = p_qr.wrappers[WrappersMP.DIFFUSION_MODEL][qr_mod.ROUTER_WRAPPER_KEY][0]
    to_qr = dict(p_qr.model_options["transformer_options"])
    to_qr["sigmas"] = torch.tensor([0.8])
    to_qr["sample_sigmas"] = torch.tensor([1.0, 0.8, 0.0])
    to_qr["_h3_steer_stats_sink"] = captured_stats
    qr_w(
        lambda x_, t_, c_, to_, **kw_: base_dm._forward(x_, t_, c_, to_, **kw_),
        [torch.zeros(1), torch.zeros(1)], torch.tensor([800.0]), ctx, to_qr,
        img=img, layout=layout_av, t_emb=t_emb, mod_segments=mod_segs,
        minimax_payload={},
    )
    assert len(captured_stats) == 1, "QueryRouter must activate H3SteerRuntime when policy clamps a2v=0.0!"

    # Verify `steering/nodes.py` respects `policy.strength_scale` & `policy.recommended_video_end_percent`!
    from steering import nodes as sn_mod

    dummy_vec = torch.zeros(5376, dtype=torch.float32)
    dummy_vec[0] = 1.0
    spec_s = sn_mod._AppliedVectorSpec(
        vector_name="test_vec",
        vector=dummy_vec,
        strength=1.0,
        mode="add",
        mask="all_rows",
        start_percent=0.0,
        end_percent=1.0,
        metadata={},
        advisories=(),
        scale_mode="raw",
    )
    cv_w = sn_mod._make_control_vector_wrapper((spec_s,), is_h3_image=False)
    seen_ctxs: list[torch.Tensor] = []
    seen_payloads: list[dict] = []

    def _cap_exec(x_, t_, c_, to_, **kw_):
        seen_ctxs.append(c_.detach().clone())
        seen_payloads.append(dict(kw_.get("minimax_payload") or {}))
        return torch.zeros(1)

    ctx_5376 = torch.ones((1, 4, 5376), dtype=torch.float32)
    to_s1 = {"sigmas": torch.tensor([0.8]), "sample_sigmas": torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])}
    cv_w(_cap_exec, [torch.zeros(1), torch.zeros(1)], torch.tensor([800.0]), ctx_5376, to_s1, minimax_payload={})
    pol_boost = lp.H3LivePolicy(strength_scale=1.12, emit_ws_events=False)
    to_s2 = {**to_s1, lp.LIVE_POLICY_KEY: pol_boost}
    cv_w(_cap_exec, [torch.zeros(1), torch.zeros(1)], torch.tensor([800.0]), ctx_5376, to_s2, minimax_payload={})
    delta_base = float((seen_ctxs[0][0, 0, 0] - 1.0).item())
    delta_pol = float((seen_ctxs[1][0, 0, 0] - 1.0).item())
    assert abs(delta_pol - delta_base * 1.12) < 1e-4, f"Expected 1.12x scaled steer: base={delta_base}, pol={delta_pol}"

    # Anti-Copy-Freeze Governor (`recommended_video_end_percent = 0.33`) drops `<Video N>` refs at Step 2 (`2/6 >= 0.33`)!
    pol_rel = lp.H3LivePolicy(recommended_video_end_percent=0.33, emit_ws_events=False)
    to_s3 = {"sigmas": torch.tensor([0.6]), "sample_sigmas": torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]), lp.LIVE_POLICY_KEY: pol_rel}
    vid_ref_payload = {
        "refs": [{"kind": "video", "latent": torch.zeros((1, 24, 2, 8, 12)), "latent_t": 2, "latent_h": 8, "latent_w": 12, "ref_audio_t": 0}],
        "cond_video_latents": [torch.zeros((1, 24, 2, 8, 12))],
    }
    cv_w(_cap_exec, [torch.zeros((1, 24, 2, 8, 12)), torch.zeros((1, 32, 2, 10))], torch.tensor([600.0]), ctx_5376, to_s3, minimax_payload=vid_ref_payload)
    assert seen_payloads[-1].get("cond_video_latents") == [], (
        f"Expected recommended_video_end_percent=0.33 to drop <Video N> refs at Step 2, got {seen_payloads[-1]}"
    )

    # Explicit user `video_ref_end_percent = 0.33` MUST NEVER be raised to `0.50` by `pol_vid_end = 0.50`!
    spec_rel33 = sn_mod._AppliedVectorSpec(
        vector_name="none",
        vector=dummy_vec,
        strength=0.0,
        mode="add",
        mask="all_rows",
        start_percent=0.0,
        end_percent=1.0,
        metadata={},
        advisories=(),
        scale_mode="raw",
        video_ref_end_percent=0.33,
    )
    cv_w33 = sn_mod._make_control_vector_wrapper((spec_rel33,), is_h3_image=False)
    pol_rel50 = lp.H3LivePolicy(recommended_video_end_percent=0.50, emit_ws_events=False)
    to_s4 = {"sigmas": torch.tensor([0.6]), "sample_sigmas": torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]), lp.LIVE_POLICY_KEY: pol_rel50}
    cv_w33(_cap_exec, [torch.zeros((1, 24, 2, 8, 12)), torch.zeros((1, 32, 2, 10))], torch.tensor([600.0]), ctx_5376, to_s4, minimax_payload=vid_ref_payload)
    assert seen_payloads[-1].get("cond_video_latents") == [], (
        f"pol_vid_end=0.50 must NOT clobber explicit video_ref_end_percent=0.33! got {seen_payloads[-1]}"
    )

    # Anti-Copy-Freeze (`pol_vid_end=0.33`) ALSO releases `<Picture 1>` still refs (`eff_ref_end=0.33`) on AV video!
    pic_ref_payload = {
        "refs": [{"kind": "image", "latent": torch.zeros((1, 24, 1, 8, 12)), "latent_t": 1, "latent_h": 8, "latent_w": 12}],
        "cond_video_latents": [torch.zeros((1, 24, 1, 8, 12))],
    }
    cv_w(_cap_exec, [torch.zeros((1, 24, 2, 8, 12)), torch.zeros((1, 32, 2, 10))], torch.tensor([600.0]), ctx_5376, to_s3, minimax_payload=pic_ref_payload)
    assert seen_payloads[-1].get("cond_video_latents") == [], (
        f"Expected pol_vid_end=0.33 to drop <Picture 1> ref on AV copy-freeze, got {seen_payloads[-1]}"
    )
    # Fast downward burn attack: highlight_burn + high_saturation drops immediately to <= 0.55 on Step 0 (floor 0.35)!
    burn_img = np.zeros((32, 32, 3), dtype=np.uint8)
    burn_img[:, :, 0] = 255  # pure clipped saturated red -> high_saturation + highlight_burn
    pol_burn = lp.H3LivePolicy(emit_ws_events=False)
    pol_burn.update_from_preview(burn_img, step=0)
    assert 0.35 <= pol_burn.strength_scale <= 0.50, f"Expected fast burn attack [0.35, 0.50], got {pol_burn.strength_scale}"
    to_burn = {**to_s1, lp.LIVE_POLICY_KEY: pol_burn}
    cv_w(_cap_exec, [torch.zeros(1), torch.zeros(1)], torch.tensor([800.0]), ctx_5376, to_burn, minimax_payload={})
    delta_burn = float((seen_ctxs[-1][0, 0, 0] - 1.0).item())
    assert abs(delta_burn - delta_base * pol_burn.strength_scale) < 1e-4, (delta_burn, delta_base, pol_burn.strength_scale)
    print("[ok] 9. Closed-loop H3LivePolicy actuators (KJ preview bridge, fast burn taper, <Picture 1>/<Video 1> 0.33 release, QueryRouter a2v=0.0) verified")

    # 10. Phase-matched `video_x0` temporal freeze motion, `prompt_id` WS stamping, & `thermostat_only` coexistence
    torch.manual_seed(33)
    # Frozen video: identical 4f frames (1..4, 6..9) despite distinct 1f chunk anchors (0, 5)!
    v_frozen = torch.zeros((1, 24, 10, 4, 6))
    anchor_lat = torch.randn((1, 24, 1, 4, 6)) * 2.5
    body_lat = torch.randn((1, 24, 1, 4, 6)) * 1.0
    v_frozen[:, :, 0:1] = anchor_lat
    v_frozen[:, :, 5:6] = anchor_lat + 0.01 * torch.randn_like(anchor_lat)
    for k in (1, 2, 3, 4, 6, 7, 8, 9):
        v_frozen[:, :, k : k + 1] = body_lat + 0.01 * torch.randn_like(body_lat)
    ev_frz = lp.evaluate_video_x0_motion(v_frozen)
    assert ev_frz["freeze_risk"] is True and ev_frz["norm_temporal_delta"] < 0.12, ev_frz
    # Fast-moving video: distinct per-step latents -> high_motion
    v_moving = torch.randn((1, 24, 10, 4, 6))
    ev_mov = lp.evaluate_video_x0_motion(v_moving)
    assert ev_mov["freeze_risk"] is False and ev_mov["high_motion"] is True, ev_mov
    # Single-frame image (`T_lat == 1`) safely skips
    ev_t1 = lp.evaluate_video_x0_motion(torch.randn((1, 24, 1, 4, 6)))
    assert ev_t1["norm_temporal_delta"] is None and ev_t1["freeze_risk"] is False

    pol_vx = lp.H3LivePolicy(mask_source="thermostat_only", emit_ws_events=False)
    pol_vx.update_from_video_x0(v_frozen, step=1, sigma=0.97)
    assert pol_vx.recommended_video_end_percent == 0.33, pol_vx.to_report()

    # `emit_event` automatically populates `prompt_id` from `PromptServer.instance.last_prompt_id`
    sent_events: list[tuple[str, dict]] = []
    fake_server = types.ModuleType("server")
    fake_server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(
            last_prompt_id="pid-scout-42",
            send_sync=lambda ev_name, payload: sent_events.append((ev_name, payload)),
        )
    )
    prev_server = sys.modules.get("server")
    sys.modules["server"] = fake_server
    try:
        pol_ws = lp.H3LivePolicy(mask_source="thermostat_only", emit_ws_events=True)
        assert pol_ws.emit_event({"kind": "preview_thermostat", "step": 0}) is True
        assert sent_events and sent_events[-1][0] == "h3_self_steer"
        assert sent_events[-1][1].get("prompt_id") == "pid-scout-42", sent_events[-1]
        pol_ws.reset_for_new_run()
        assert pol_ws.prompt_id is None
    finally:
        if prev_server is None:
            sys.modules.pop("server", None)
        else:
            sys.modules["server"] = prev_server

    # `mask_source="thermostat_only"` stacks cleanly with `SpatialQueryRouter`, `QueryRouter`, `SageAttention`, and unmasked `EditGuider`!
    p_th_sqr, pol_th_sqr = lp.attach_self_mask_router(
        p_sqr, region_conditioning=None, mask_source="thermostat_only", emit_ws_events=False
    )
    p_th_sage, _ = lp.attach_self_mask_router(
        p_sage, region_conditioning=None, mask_source="thermostat_only", emit_ws_events=False
    )
    out_th0, _ = p_th_sqr.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=1.0, denoised=packed_burst10)
    assert torch.isfinite(out_th0).all()
    assert pol_th_sqr.harvested_mask_frames is None, "thermostat_only must NEVER write harvested_mask_frames!"
    assert len(pol_th_sqr.video_x0_history) == 1 and len(pol_th_sqr.audio_x0_history) == 1
    eg_th = eg.EditGuider.__new__(eg.EditGuider)
    eg_th.model_patcher = p_th_sqr
    eg_th.set_edit(scale=1.5, first_step=1, align_positions=True, last_step=3, streams="both", mask=None)
    assert eg_th._effective_raw_mask() is None, "Unmasked EditGuider must stay unmasked under thermostat_only!"
    print("[ok] 10. Phase-matched video_x0 motion, prompt_id WS stamp, & thermostat_only universal coexistence verified")

    # 11. Dynamic Step-1 Auto-Shield (`early_0_24 -> early_0_24_shield`) on washed-out small portals (both wire orders)
    portal_box = (0.35, 0.65, 0.30, 0.70)  # 0.30 * 0.40 = 12% canvas (< 32%)
    washed_img = np.full((64, 64, 3), 120, dtype=np.uint8)
    washed_img[20:45, 22:42, :] = 124  # sep_score ~ 4.0 < 12.0 -> washed out!
    crisp_img = np.full((64, 64, 3), 40, dtype=np.uint8)
    crisp_img[20:45, 22:42, :] = [230, 120, 40]  # strong color/luma separation -> no-op!

    sep_washed = lp.evaluate_region_separation(washed_img, region_box=portal_box)
    assert sep_washed is not None and sep_washed["washed_out"] is True and sep_washed["small_region"] is True, sep_washed
    sep_crisp = lp.evaluate_region_separation(crisp_img, region_box=portal_box)
    assert sep_crisp is not None and sep_crisp["washed_out"] is False, sep_crisp
    # Large 50% split-screen (`x_start=0.5..1.0`) must NOT trigger Auto-Shield even if colors look close!
    sep_half = lp.evaluate_region_separation(washed_img, region_box=(0.5, 1.0, 0.0, 1.0))
    assert sep_half is not None and sep_half["small_region"] is False and sep_half["washed_out"] is False, sep_half

    # Wire order A: SpatialQR -> SelfMask(thermostat_only); Wire order B: SelfMask(thermostat_only) -> SpatialQR
    # Use a realistic non-planted Q/K/V mock DiT (`custom_qkv=None`) with `seamless_off` on the 4-block DiT so
    # blocks 0..3 exercise the exact seamless branch that blocks 25..49 run under `early_0_24` on 50-block H3.
    base_p_rand = _MiniPatcher(_MockDM(n_blocks=4, custom_qkv=None))
    p_sqr_portal = sqr.attach_spatial_query_router(
        base_p_rand,
        region_conditioning=reg_cond,
        x_start=0.35,
        x_end=0.65,
        y_start=0.30,
        y_end=0.70,
        isolation="seamless_off",
    )
    p_order_a, pol_a = lp.attach_self_mask_router(
        p_sqr_portal, region_conditioning=None, mask_source="thermostat_only", emit_ws_events=False
    )
    p_th_first, pol_b = lp.attach_self_mask_router(
        base_p_rand, region_conditioning=None, mask_source="thermostat_only", emit_ws_events=False
    )
    p_order_b = sqr.attach_spatial_query_router(
        p_th_first,
        region_conditioning=reg_cond,
        x_start=0.35,
        x_end=0.65,
        y_start=0.30,
        y_end=0.70,
        isolation="seamless_off",
    )
    for order_name, p_ord, pol_ord in (("SQR->Thermostat", p_order_a, pol_a), ("Thermostat->SQR", p_order_b, pol_b)):
        # Step 0 unshielded
        out_unsh, _ = p_ord.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=1.0)
        assert pol_ord.active_region_box == portal_box, (order_name, pol_ord.active_region_box)
        assert pol_ord.force_shield is False
        # Step 0 preview never flips force_shield prematurely; Step 1 washed-out preview engages Auto-Shield!
        pol_ord.update_from_preview(washed_img, step=0)
        assert pol_ord.force_shield is False
        m_s1 = pol_ord.update_from_preview(washed_img, step=1)
        assert pol_ord.force_shield is True and "auto_shield_engaged" in m_s1["flags"], (order_name, m_s1)
        # Step 2 now executes SpatialVisualQRRuntime with `shield_region=True`, producing distinct shielded output!
        out_sh, _ = p_ord.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=0.6)
        assert torch.isfinite(out_sh).all()
        assert float((out_sh - out_unsh).abs().max().item()) > 1e-2, f"{order_name}: force_shield had negligible effect!"

    # Crisp Step-1 portal must NOT engage Auto-Shield; already-shielded portal must NOT emit auto_shield_engaged.
    pol_crisp = lp.H3LivePolicy(mask_source="thermostat_only", emit_ws_events=False)
    pol_crisp.register_spatial_region(x_start=0.35, x_end=0.65, y_start=0.30, y_end=0.70, already_shielded=False)
    pol_crisp.update_from_preview(crisp_img, step=1)
    assert pol_crisp.force_shield is False
    pol_already = lp.H3LivePolicy(mask_source="thermostat_only", emit_ws_events=False)
    pol_already.register_spatial_region(x_start=0.35, x_end=0.65, y_start=0.30, y_end=0.70, already_shielded=True)
    pol_already.update_from_preview(washed_img, step=1)
    assert pol_already.force_shield is False
    print("[ok] 11. Dynamic Step-1 Auto-Shield (`early_0_24 -> *_shield`, both wire orders, crisp/half/already-shielded no-ops) verified")

    # 12. Remote `heylook` Step-0 Visual Critic JSON parser + `< 3 ms` CPU `WordVector(mode="project_out")` text scrub
    raw_fenced = '```json\n{"has_border_seam": true, "has_color_bleed": false, "has_watermark_text": false, "scrub_words": ["Border", "frame", "letterbox"], "reason": "black border bars"}\n```'
    parsed_c = lp._parse_step0_critic_json(raw_fenced)
    assert parsed_c["ok"] is True and parsed_c["has_border_seam"] is True
    assert parsed_c["scrub_words"] == ["border", "frame", "letterbox"], parsed_c

    # Verify vision prefix rows (`tags == 0`) & sink row `0` stay bitwise identical while text rows drop by `1 - alpha`!
    u_test = torch.zeros(5376, dtype=torch.float32)
    u_test[7] = 1.0
    pol_scrub = lp.H3LivePolicy(mask_source="thermostat_only", critic_scrub_alpha=0.85, emit_ws_events=False)
    pol_scrub._scrub_unit_vec_override = u_test
    pol_scrub.update_from_critic(parsed_c, step=0)
    assert pol_scrub.project_out_words == ["border", "frame", "letterbox"]

    ctx_mix = torch.zeros((1, 6, 5376), dtype=torch.float32)
    ctx_mix[:, :, 7] = 10.0  # component along scrub direction
    ctx_mix[:, :, 11] = 3.0  # orthogonal semantic component
    tags_mix = torch.tensor([[1, 0, 0, 1, 1, 1]], dtype=torch.long)  # row 0 = sink, rows 1..2 = vision, rows 3..5 = text
    ctx_out, did_sc = lp.apply_live_policy_text_scrub(ctx_mix, tags_mix, pol_scrub)
    assert did_sc is True
    assert torch.equal(ctx_out[:, 0, :], ctx_mix[:, 0, :]), "Sink row 0 must stay bitwise identical!"
    assert torch.equal(ctx_out[:, 1:3, :], ctx_mix[:, 1:3, :]), "Vision rows (tags == 0) must stay bitwise identical!"
    assert abs(float(ctx_out[0, 3, 7].item()) - 1.5) < 1e-4, f"Expected 10*(1-0.85)=1.5 on text row, got {ctx_out[0, 3, 7].item()}"
    assert abs(float(ctx_out[0, 3, 11].item()) - 3.0) < 1e-5, "Orthogonal text channel must stay untouched!"

    # End-to-end `_preview_bridge_outer` + `critic_fn` + Step-1 `SelfMaskRouter` scrub!
    critic_calls: list[int] = []
    p_cr, pol_cr = lp.attach_self_mask_router(
        base_p,
        region_conditioning=None,
        mask_source="thermostat_only",
        critic_scrub_alpha=0.80,
        critic_fn=lambda rgb: (critic_calls.append(1) or parsed_c),
        emit_ws_events=False,
    )
    pol_cr._scrub_unit_vec_override = u_test
    kj_cr = _MockKJWrapper()
    p_cr.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "kj_preview_override", kj_cr)
    bridge_cr = p_cr.get_wrappers(WrappersMP.OUTER_SAMPLE, f"{lp.SELF_MASK_WRAPPER_KEY}_preview_bridge")[0]

    class _MockOuterCriticExec:
        def __init__(self, mp, kj_obj):
            self.class_obj = types.SimpleNamespace(model_patcher=mp)
            self.kj_obj = kj_obj

        def __call__(self, noise, lat_i, sampler, sigmas, denoise_mask, cb, disable_pbar, seed, **kw):
            self.kj_obj.frames.append(dim_frame)
            cb(0, None, None, 6)
            self.kj_obj.frames.append(dim_frame)
            cb(1, None, None, 6)
            return lat_i

    bridge_cr(_MockOuterCriticExec(p_cr, kj_cr), None, None, None, None, None, None, True, 7)
    assert len(critic_calls) == 1, f"Critic must run strictly once on Step 0, ran {len(critic_calls)} times"
    assert pol_cr.project_out_words == ["border", "frame", "letterbox"]

    # Verify Step-1 `SelfMaskRouter._wrapper` scrubs `context` before passing to `executor`!
    seen_cr_ctx: list[torch.Tensor] = []
    dm_w_cr = p_cr.wrappers[WrappersMP.DIFFUSION_MODEL][lp.SELF_MASK_WRAPPER_KEY][0]
    to_cr = dict(p_cr.model_options["transformer_options"])
    to_cr["sigmas"] = torch.tensor([0.8])
    to_cr["sample_sigmas"] = torch.tensor([1.0, 0.8, 0.0])
    # Mark fresh_run already consumed (simulating Step 0 already completed before Step-0 preview fired critic)
    dm_w_cr(
        lambda x_, t_, c_, to_, **kw_: (seen_cr_ctx.append(c_.detach().clone()) or torch.zeros(1)),
        [torch.zeros(1), torch.zeros(1)], torch.tensor([1000.0]), ctx_mix, to_cr,
        minimax_payload={"text_token_tags": tags_mix},
    )
    pol_cr.update_from_critic(parsed_c, step=0)
    dm_w_cr(
        lambda x_, t_, c_, to_, **kw_: (seen_cr_ctx.append(c_.detach().clone()) or torch.zeros(1)),
        [torch.zeros(1), torch.zeros(1)], torch.tensor([800.0]), ctx_mix, to_cr,
        minimax_payload={"text_token_tags": tags_mix},
    )
    assert abs(float(seen_cr_ctx[-1][0, 3, 7].item()) - 2.0) < 1e-4, f"Expected 10*(1-0.80)=2.0, got {seen_cr_ctx[-1][0, 3, 7].item()}"
    assert torch.equal(seen_cr_ctx[-1][:, 1:3, :], ctx_mix[:, 1:3, :]), "Vision rows must remain bitwise untouched!"

    # Also verify real J-Lens CPU word synthesis latency if `j_lens_h3.safetensors` is installed on disk!
    from steering import xray as _xray
    if _xray.bundle_available():
        import time as _time
        _t0 = _time.perf_counter()
        u_real = lp.synthesize_policy_scrub_unit_vec(["border", "frame", "letterbox"], family="fl2va")
        _ms = (_time.perf_counter() - _t0) * 1000.0
        assert u_real is not None and tuple(u_real.shape) == (5376,), None if u_real is None else u_real.shape
        assert abs(float(torch.linalg.vector_norm(u_real).item()) - 1.0) < 1e-4
        print(f"[ok] 12. Step-0 Visual Critic JSON parser + CPU WordVector project_out scrub verified (real J-Lens vec in {_ms:.1f} ms)")
    else:
        print("[ok] 12. Step-0 Visual Critic JSON parser + CPU WordVector project_out scrub verified (J-Lens bundle absent)")

    # 13. Holistic regressions: reset_for_new_run full clear, Run-#2 cached KJ frames=[] reset,
    #     reverse wire order QueryRouter -> SelfMask(thermostat_only), and transparent audio_transients threshold_z
    pol_rst = lp.H3LivePolicy(mask_source="text_phrase", emit_ws_events=False)
    pol_rst.harvested_mask_thw = torch.ones((1, 4, 4))
    pol_rst.harvested_mask_frames = torch.ones((1, 4, 4))
    pol_rst.mask_stats["coverage_mean"] = 0.5
    pol_rst.step_history.append({"step": 0})
    pol_rst.audio_x0_history.append({"step": 0})
    pol_rst.video_x0_history.append({"step": 0})
    pol_rst.preview_history.append({"step": 0})
    pol_rst.reset_for_new_run()
    assert pol_rst.harvested_mask_thw is None and pol_rst.harvested_mask_frames is None
    assert not pol_rst.mask_stats and not pol_rst.step_history
    assert not pol_rst.audio_x0_history and not pol_rst.video_x0_history and not pol_rst.preview_history

    # Run-#2 cached ModelPreviewOverrideKJ (`kj.frames = []` reset inside inner OUTER_SAMPLE call)
    p_r2, pol_r2 = lp.attach_self_mask_router(
        base_p, region_conditioning=None, mask_source="thermostat_only", emit_ws_events=False
    )
    kj_r2 = _MockKJWrapper()
    p_r2.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "kj_preview_override", kj_r2)
    bridge_r2 = p_r2.get_wrappers(WrappersMP.OUTER_SAMPLE, f"{lp.SELF_MASK_WRAPPER_KEY}_preview_bridge")[0]

    class _MockCachedKJOuter:
        def __init__(self, mp, kj_obj):
            self.class_obj = types.SimpleNamespace(model_patcher=mp)
            self.kj_obj = kj_obj

        def __call__(self, noise, lat_i, sampler, sigmas, denoise_mask, cb, disable_pbar, seed, **kw):
            # Mimic ModelPreviewOverrideKJ resetting `self.frames = []` at start of every run!
            self.kj_obj.frames = []
            for st in range(3):
                self.kj_obj.frames.append(dim_frame)
                cb(st, None, None, 3)
            return lat_i

    bridge_r2(_MockCachedKJOuter(p_r2, kj_r2), None, None, None, None, None, None, True, 1)
    assert len(pol_r2.preview_history) == 3, f"Run 1 expected 3 preview frames, got {len(pol_r2.preview_history)}"
    pol_r2.reset_for_new_run()
    bridge_r2(_MockCachedKJOuter(p_r2, kj_r2), None, None, None, None, None, None, True, 2)
    assert len(pol_r2.preview_history) == 3, f"Run 2 after KJ frames=[] reset stalled: got {len(pol_r2.preview_history)}"

    # Reverse wire order: QueryRouter(sec_mode='off', aud_to_vid_gain=1.0) -> SelfMask(thermostat_only)
    from steering import query_router as _qr_mod

    p_qr_first = _qr_mod.attach_query_router(base_p.clone(), section_split="off", aud_to_vid_gain=1.0)
    assert _qr_mod.ROUTER_WRAPPER_KEY in p_qr_first.wrappers.get(WrappersMP.DIFFUSION_MODEL, {}), (
        "QueryRouter(sec_mode='off', a2v_gain=1.0) on hookable AV H3 must install its wrapper for downstream thermostat_only"
    )
    p_rev_qr, pol_rev_qr = lp.attach_self_mask_router(
        p_qr_first, region_conditioning=None, mask_source="thermostat_only", emit_ws_events=False
    )
    out_rev_open, _ = p_rev_qr.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=0.8)
    pol_rev_qr.recommended_aud_to_vid_gain = 0.0
    out_rev_clamped, _ = p_rev_qr.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=0.6)
    assert float((out_rev_clamped - out_rev_open).abs().max().item()) > 1e-4, (
        "Reverse wire order QueryRouter -> SelfMask(thermostat_only) failed to apply live a2v=0.0 clamp!"
    )

    # Transparent `audio_transients` threshold_z (no silent max(2.2, threshold_z) override)
    p_at_low, pol_at_low = lp.attach_self_mask_router(
        base_p.clone(),
        region_conditioning=reg_cond,
        mask_source="audio_transients",
        threshold_z=1.5,
        emit_ws_events=False,
    )
    p_at_low.run_step(ctx, layout_av, img, t_emb, mod_segs, sigma=1.0, denoised=packed_burst10)
    assert pol_at_low.mask_stats.get("z_thresh") == 1.5, f"threshold_z was silently overridden: {pol_at_low.mask_stats}"

    print("[ok] 13. Regressions (full reset_for_new_run, Run-2 KJ frames=[] reset, reverse QR->Thermostat, transparent audio_transients z_thresh) verified")

    # 14. True in-process Step-0 Visual Critic Rollback / Replay (1.0x when clean, 1.17x only on defect;
    #     merged with replay_step0=True into ONE single Step-0 replay; plus real CPU taeh3 decode if installed)
    # 14a. Clean Step-0 preview -> ZERO replay (1.0x)
    clean_critic = {"ok": True, "has_border_seam": False, "has_color_bleed": False, "has_watermark_text": False, "scrub_words": [], "reason": "clean"}
    p_c_clean, pol_c_clean = lp.attach_self_mask_router(
        base_p.clone(),
        region_conditioning=None,
        mask_source="thermostat_only",
        enable_heylook_critic=True,
        critic_fn=lambda rgb: clean_critic,
        preview_decoder_fn=lambda _x, _o: dim_frame,
        emit_ws_events=False,
    )
    pol_c_clean._scrub_unit_vec_override = u_test
    dm_w_clean = p_c_clean.wrappers[WrappersMP.DIFFUSION_MODEL][lp.SELF_MASK_WRAPPER_KEY][0]
    calls_clean: list[torch.Tensor] = []
    to_c0 = dict(p_c_clean.model_options["transformer_options"])
    to_c0["sigmas"] = torch.tensor([1.0])
    to_c0["sample_sigmas"] = torch.tensor([1.0, 0.8, 0.0])
    dm_w_clean(
        lambda x_, t_, c_, to_, **kw_: (calls_clean.append(c_.detach().clone()) or torch.zeros(1)),
        [torch.zeros(1), torch.zeros(1)], torch.tensor([1000.0]), ctx_mix, to_c0,
        minimax_payload={"text_token_tags": tags_mix},
    )
    assert len(calls_clean) == 1, f"Clean Step-0 critic must NOT replay (expected 1 call, got {len(calls_clean)})"
    assert len(pol_c_clean.critic_history) == 1 and not pol_c_clean.project_out_words

    # 14b. Defective Step-0 preview in `thermostat_only` -> immediate 1.17x Step-0 rollback replay with scrubbed context!
    p_c_bad, pol_c_bad = lp.attach_self_mask_router(
        base_p.clone(),
        region_conditioning=None,
        mask_source="thermostat_only",
        enable_heylook_critic=True,
        critic_scrub_alpha=0.80,
        critic_fn=lambda rgb: parsed_c,
        preview_decoder_fn=lambda _x, _o: dim_frame,
        emit_ws_events=False,
    )
    pol_c_bad._scrub_unit_vec_override = u_test
    dm_w_bad = p_c_bad.wrappers[WrappersMP.DIFFUSION_MODEL][lp.SELF_MASK_WRAPPER_KEY][0]
    calls_bad: list[torch.Tensor] = []
    to_cb = dict(p_c_bad.model_options["transformer_options"])
    to_cb["sigmas"] = torch.tensor([1.0])
    to_cb["sample_sigmas"] = torch.tensor([1.0, 0.8, 0.0])
    dm_w_bad(
        lambda x_, t_, c_, to_, **kw_: (calls_bad.append(c_.detach().clone()) or torch.zeros(1)),
        [torch.zeros(1), torch.zeros(1)], torch.tensor([1000.0]), ctx_mix, to_cb,
        minimax_payload={"text_token_tags": tags_mix},
    )
    assert len(calls_bad) == 2, f"Defective Step-0 critic must replay Step 0 once (expected 2 calls, got {len(calls_bad)})"
    assert abs(float(calls_bad[0][0, 3, 7].item()) - 10.0) < 1e-4, "Initial Step-0 pass should be un-scrubbed"
    assert abs(float(calls_bad[1][0, 3, 7].item()) - 2.0) < 1e-4, "Replayed Step-0 pass must scrub text row (10 -> 2.0)"
    assert torch.equal(calls_bad[1][:, 0, :], ctx_mix[:, 0, :]), "Sink row 0 must remain bitwise identical"
    assert torch.equal(calls_bad[1][:, 1:3, :], ctx_mix[:, 1:3, :]), "Vision rows must remain bitwise identical"
    assert any(h.get("phase") == "critic_replay_step0" for h in pol_c_bad.step_history)

    # 14c. Combined `replay_step0=True` + defective Step-0 critic -> ONE merged Step-0 replay (2 calls total, not 3!)
    p_c_merge, pol_c_merge = lp.attach_self_mask_router(
        base_p.clone(),
        region_conditioning=reg_cond,
        mask_source="text_phrase",
        target_phrase="bread",
        token_pieces=tok_pieces,
        threshold_z=0.8,
        replay_step0=True,
        harvest_blocks=(2, 3),
        enable_heylook_critic=True,
        critic_scrub_alpha=0.80,
        critic_fn=lambda rgb: parsed_c,
        preview_decoder_fn=lambda _x, _o: dim_frame,
        emit_ws_events=False,
    )
    u_test_mock = torch.zeros(int(ctx.shape[-1]), dtype=torch.float32)
    u_test_mock[0] = 1.0
    pol_c_merge._scrub_unit_vec_override = u_test_mock
    dm_w_merge = p_c_merge.wrappers[WrappersMP.DIFFUSION_MODEL][lp.SELF_MASK_WRAPPER_KEY][0]
    to_cm = dict(p_c_merge.model_options["transformer_options"])
    to_cm["sigmas"] = torch.tensor([1.0])
    to_cm["sample_sigmas"] = torch.tensor([1.0, 0.8, 0.0])
    merge_calls: list[int] = []
    dm_w_merge(
        lambda x_, t_, c_, to_, **kw_: (merge_calls.append(1) or base_dm._forward(x_, t_, c_, to_, **kw_)),
        [torch.zeros(1), torch.zeros(1)], torch.tensor([1000.0]), ctx, to_cm,
        img=img, layout=layout_av, t_emb=t_emb, mod_segments=mod_segs,
        minimax_payload={},
    )
    assert len(merge_calls) == 2, f"replay_step0 + critic rollback must merge into 1 replay (2 calls, got {len(merge_calls)})"
    assert any(h.get("phase") == "critic_replay_step0" for h in pol_c_merge.step_history)
    assert any(h.get("phase") == "routed" and h.get("replayed_step0") for h in pol_c_merge.step_history)

    # 14d. Real CPU `taeh3.safetensors` & `taeh3_image.safetensors` single-frame decode (0 MB GPU)
    tae_v = lp._resolve_taeh3_path(is_image=False)
    tae_i = lp._resolve_taeh3_path(is_image=True)
    tae_note = "taeh3 absent"
    if tae_v is not None:
        rgb_v = lp.decode_step0_latent_preview_rgb(torch.randn(1, 24, 3, 16, 28), is_image=False)
        assert rgb_v is not None and rgb_v.ndim == 3 and rgb_v.shape[2] == 3 and rgb_v.dtype == np.uint8, rgb_v
        tae_note = f"taeh3 video {rgb_v.shape}"
    if tae_i is not None:
        rgb_i = lp.decode_step0_latent_preview_rgb(torch.randn(1, 24, 1, 32, 32), is_image=True, max_edge=512)
        assert rgb_i is not None and rgb_i.ndim == 3 and rgb_i.shape[2] == 3 and rgb_i.dtype == np.uint8, rgb_i
        tae_note += f" + image {rgb_i.shape}"
    print(f"[ok] 14. True Step-0 Visual Critic rollback replay (1.0x clean / 1.17x defect / merged replay_step0 + {tae_note}) verified")

    print("ALL H3 LivePolicy & SelfMaskRouter CPU checks passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
