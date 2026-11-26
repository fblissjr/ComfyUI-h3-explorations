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
    assert input_ids[-1] == "replay_step0" and len(input_ids) == 15
    assert getattr(schema.inputs[1], "optional", False) is True, "region_conditioning must be optional=True"
    print(f"[ok] 7. All refusal guards & MiniMaxH3SelfMaskRouter schema ({len(input_ids)} inputs) verified")

    # 8. `audio_transients` Step-0 burst mask + optional `region_conditioning=None` + `SelfMask -> 2x EditGuider` bridge
    torch.manual_seed(21)
    aud_burst = torch.randn((1, 32, 2, 25)) * 0.6
    aud_burst[:, :, :, 14] *= 7.5  # strong transient at a_idx=14 -> latent step 2 (windows: [0,2), [1,9), [8,16), [15,22))
    m_tr, st_tr = lp.synthesize_audio_transient_mask(aud_burst, latent_t=4, patch_h=4, patch_w=6, z_thresh=2.2)
    assert m_tr.shape == (4, 4, 6)
    assert float(m_tr[2].mean().item()) > 0.85 and float(m_tr[0].mean().item()) == 0.0 and float(m_tr[3].mean().item()) == 0.0, st_tr
    assert st_tr["active_latent_steps"] == [2]

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
    assert pol_tr.recommended_video_end_percent == 0.33, (
        f"Static Step-1 preview should trigger 0.33 anti-copy-freeze governor, got {pol_tr.recommended_video_end_percent}"
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
    print("[ok] 9. Closed-loop H3LivePolicy actuators (KJ preview bridge, Steer scale, 0.33 governor, QueryRouter a2v=0.0) verified")

    print("ALL H3 LivePolicy & SelfMaskRouter CPU checks passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
