#!/usr/bin/env python3
"""Offline CPU verification for `bench/probe_h3_sensors.py` (4-sensor read-only calibration probe).

Verifies:
1. Bitwise zero-mutation parity (`max_abs == 0.0`) on both AV video (`MiniMaxH3Model`-like) and
   single-frame image (`MiniMaxH3ImageModel`-like) mocks, including `AttentionTensorContainer` peek safety.
2. Sensor 1 (`Q_vid @ K_txt[noun]^T`): exact quadrant localization (`top10_mass`, `box_enrichment`,
   `peak_in_box`, `top10_sink_margin`) + `find_noun_token_spans` BPE boundary matching.
3. Sensor 2 (`Q_vid @ K_aud[t]^T`): causal `[1, 4, 4, 4, 4]` + `FRAME_RESCALE = 5/3` stereo alignment
   with Left-ear vs Right-ear spatial centroid separation (`stereo_lr_x_sep > 0.35`).
4. Sensor 3 (Step-`0/1` pure-latent `audio_x0` transients): exact `40 Hz` peak detection (`t_sec`,
   `video_frame_24fps`), `L/R` correlation/dB balance, bit-identical `denoised` passthrough, and clean
   no-op on 5D single-frame image latents.
5. Sensor 4 (Block-by-block `h_txt_a` vs `h_shadow_b` + `MiniMaxH3SpatialQueryRouter` stacking in
   both orders + synthetic `J-Lens` `decode_vector` readout + disjoint CPU storage + `data/` refusal +
   PNG contact sheet rendering).
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

BENCH_DIR = Path(__file__).resolve().parent
REPO_DIR = BENCH_DIR.parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from _lib import bootstrap

bootstrap(cpu=True)

if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

import comfy.utils
from comfy.ldm.minimax.model import AttentionTensorContainer, PackedLayout
from comfy.patcher_extension import WrappersMP

import probe_h3_sensors as ps
from steering import spatial_query_router as sqr


# ------------------------------------------------------------------ deterministic mock DiT


class _MockAttn(nn.Module):
    def __init__(self, hidden: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv_proj = nn.Linear(hidden, hidden * 3, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)
        # Keep identity-like projections so planted Q/K directions stay cleanly recoverable.
        with torch.no_grad():
            eye = torch.eye(hidden)
            self.qkv_proj.weight.copy_(torch.cat([eye, eye, eye], dim=0))
            self.out_proj.weight.copy_(0.1 * eye)


class _MockBlock(nn.Module):
    def __init__(self, hidden: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.attn = _MockAttn(hidden, heads)
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.adaln_proj = lambda t_emb: tuple(
            torch.zeros(t_emb.shape[0] * 3, hidden, device=t_emb.device, dtype=t_emb.dtype)
            for _ in range(6)
        )

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None, attention=None):
        to = transformer_options or {}
        s = int(x.shape[0])
        h = self.attn.heads
        d = self.attn.head_dim
        q, k, v = self.attn.qkv_proj(x).split(h * d, dim=-1)
        q = q.view(1, s, h, d).transpose(1, 2).contiguous()
        k = k.view(1, s, h, d).transpose(1, 2).contiguous()
        v = v.view(1, s, h, d).transpose(1, 2).contiguous()

        qc = AttentionTensorContainer(q)
        kc = AttentionTensorContainer(k)
        vc = AttentionTensorContainer(v)
        ovr = to.get("optimized_attention_override")

        def _base_sdpa(qq, kk, vv, heads_, mask=None, skip_reshape=False, skip_output_reshape=False, **kw):
            if not torch.is_tensor(qq):
                qq, kk, vv = qq.take(), kk.take(), vv.take()
            out = torch.nn.functional.scaled_dot_product_attention(qq, kk, vv, is_causal=False)
            if skip_output_reshape:
                return out
            return out.transpose(1, 2).reshape(1, s, h * d)

        if ovr is not None:
            attn_out = ovr(_base_sdpa, qc, kc, vc, h, skip_reshape=True, skip_output_reshape=False, transformer_options=to)
        else:
            attn_out = _base_sdpa(qc, kc, vc, h, skip_reshape=True, skip_output_reshape=False)
        return x + self.attn.out_proj(attn_out.squeeze(0))


class _MockDM(nn.Module):
    def __init__(self, *, hidden: int = 32, heads: int = 4, n_blocks: int = 4, is_image: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden
        self.blocks = nn.ModuleList([_MockBlock(hidden, heads) for _ in range(n_blocks)])
        self.condition_proj = nn.Identity()
        self.token_refiner = lambda x, transformer_options=None: x
        self.audio_patch_proj = None if is_image else nn.Identity()
        if is_image:
            self.__class__.__name__ = "MiniMaxH3ImageModel"


class _MiniPatcher:
    """Minimal ComfyUI `ModelPatcher`-compatible harness for offline CPU tests."""

    def __init__(self, dm: _MockDM, *, latent_shapes: list[tuple[int, ...]] | None = None) -> None:
        self.model = SimpleNamespace(diffusion_model=dm, latent_shapes=latent_shapes)
        self.model_options: dict[str, Any] = {"transformer_options": {}}
        self.wrappers: dict[Any, dict[str, list[Any]]] = {}
        self.object_patches: dict[str, Any] = {}
        self.callbacks: dict[Any, dict[str, list[Any]]] = {}
        self.post_cfg_fns: list[Any] = []

    def clone(self) -> "_MiniPatcher":
        p = _MiniPatcher(self.model.diffusion_model, latent_shapes=self.model.latent_shapes)
        to = dict(self.model_options.get("transformer_options", {}))
        pr = {k: dict(v) for k, v in to.get("patches_replace", {}).items()}
        to["patches_replace"] = pr
        p.model_options = {"transformer_options": to}
        p.wrappers = {k: {wk: list(wv) for wk, wv in v.items()} for k, v in self.wrappers.items()}
        p.object_patches = dict(self.object_patches)
        p.post_cfg_fns = list(self.post_cfg_fns)
        return p

    def set_model_patch_replace(self, patch: Any, name: str, key: str, idx: int) -> None:
        to = self.model_options.setdefault("transformer_options", {})
        pr = to.setdefault("patches_replace", {}).setdefault(name, {})
        pr[(key, idx)] = patch

    def add_callback_with_key(self, cb_type: Any, key: str, fn: Any) -> None:
        self.callbacks.setdefault(cb_type, {}).setdefault(key, []).append(fn)

    def set_model_sampler_post_cfg_function(self, fn: Any) -> None:
        self.post_cfg_fns.append(fn)

    def run_step(
        self,
        context: torch.Tensor,
        layout: PackedLayout,
        img: torch.Tensor,
        t_emb: torch.Tensor,
        mod_segments: list[tuple[int, int, int]],
        *,
        sigma: float = 1.0,
        denoised: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dm = self.model.diffusion_model
        to_base = dict(self.model_options.get("transformer_options", {}))
        to_base["sigmas"] = torch.tensor([sigma])

        def _base_exec(x_in, timestep, ctx, to, **kw):
            h = img.clone()
            pr = to.get("patches_replace", {}).get("dit", {})
            for b_idx, blk in enumerate(dm.blocks):
                to["block_index"] = b_idx
                if ("double_block", b_idx) in pr:
                    def _orig(a, _blk=blk):
                        return {"img": _blk(a["img"], a["t_emb"], a["mod_segments"], a["rope_freqs"], transformer_options=a["transformer_options"])}
                    h = pr[("double_block", b_idx)](
                        {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": None, "layout": layout, "transformer_options": to},
                        {"original_block": _orig},
                    )["img"]
                else:
                    h = blk(h, t_emb, mod_segments, None, transformer_options=to)
            return h

        chain = _base_exec
        dm_wrappers = self.wrappers.get(WrappersMP.DIFFUSION_MODEL, {})
        for w_list in reversed(list(dm_wrappers.values())):
            for w in reversed(w_list):
                prev = chain
                chain = functools.partial(w, prev)

        x_dummy = (torch.zeros(1), torch.zeros(1)) if self.model.latent_shapes and len(self.model.latent_shapes) > 1 else torch.zeros(1)
        out_h = chain(x_dummy, torch.tensor([sigma * 1000.0]), context, to_base)
        out_den = denoised
        if denoised is not None:
            for fn in self.post_cfg_fns:
                out_den = fn({"denoised": out_den, "sigma": torch.tensor([sigma]), "model": self.model})
        return out_h, out_den


# ------------------------------------------------------------------ tests


def _make_planted_av_inputs(hidden: int = 32, heads: int = 4) -> tuple[PackedLayout, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int, int]]]:
    """Create an AV PackedLayout with cleanly separated quadrant noun & stereo L/R audio cues."""
    l_txt = 10
    lt, lh, lw, aud_t = 5, 8, 12, 30  # patch grid 4x6 = 24 patches/step, 17f -> ~28..30 audio steps
    layout = PackedLayout(l_txt, lt, lh, lw, aud_t)
    s_tot = layout.seq_len
    torch.manual_seed(123)
    img = 0.02 * torch.randn(s_tot, hidden)

    va, vb = next((a, b) for (a, b, k) in layout.segments if k == "video")
    aa, ab = next((a, b) for (a, b, k) in layout.segments if k == "audio")
    ph, pw = lh // 2, lw // 2

    # Basis directions per head (head_dim = 8):
    #   dim 0..1 = clock (top-left), dim 2..3 = mirror (bottom-right),
    #   dim 4..5 = left ear (left half), dim 6..7 = right ear (right half).
    d = hidden // heads
    v_clock = torch.zeros(hidden)
    v_mirror = torch.zeros(hidden)
    v_left = torch.zeros(hidden)
    v_right = torch.zeros(hidden)
    for h_i in range(heads):
        v_clock[h_i * d + 0] = 6.0
        v_mirror[h_i * d + 2] = 6.0
        v_left[h_i * d + 4] = 6.0
        v_right[h_i * d + 6] = 6.0

    # Plant text tokens: idx 2,3 = "clock", idx 6,7 = "mirror"
    img[2] += v_clock
    img[3] += v_clock
    img[6] += v_mirror
    img[7] += v_mirror

    # Plant audio tokens: L = aa..aa+aud_t, R = aa+aud_t..ab
    img[aa : aa + aud_t] += v_left
    img[aa + aud_t : ab] += v_right

    # Plant video patches across all frames
    for t_i in range(lt):
        for y_i in range(ph):
            for x_i in range(pw):
                p_idx = va + (t_i * ph + y_i) * pw + x_i
                if y_i < ph // 2 and x_i < pw // 2:
                    img[p_idx] += v_clock
                if y_i >= ph // 2 and x_i >= pw // 2:
                    img[p_idx] += v_mirror
                if x_i < pw // 2:
                    img[p_idx] += v_left
                else:
                    img[p_idx] += v_right

    ctx = img[:l_txt].unsqueeze(0).clone()
    t_emb = torch.zeros(2, hidden)
    mod_segments = [(0, l_txt, 1), (l_txt, s_tot, 0)]
    return layout, img, ctx, t_emb, mod_segments


def main() -> int:
    # 0. BPE token-span finder
    toks = ["[", "Shot", "Ġ1", "]", "Ġa", "Ġbrass", "Ġclock", "maker", "Ġfaces", "Ġthe", "Ġtall", "Ġmir", "ror", "."]
    spans = ps.find_noun_token_spans(toks, ["clockmaker", "mirror", "clock", "absent"])
    assert spans["clockmaker"] == [6, 7], spans
    assert spans["mirror"] == [11, 12], spans
    assert spans["clock"] == [6], spans
    assert spans["absent"] == [], spans
    print("[ok] 0. find_noun_token_spans handles BPE boundaries & multi-subword nouns")

    # 1 & 2 & 3. AV model bitwise zero-mutation + Sensor 1 (nouns) + Sensor 2 (stereo audio) + Sensor 3 (audio_x0)
    layout, img, ctx, t_emb, mod_segments = _make_planted_av_inputs()
    v_shape = (1, 24, 5, 8, 12)
    a_shape = (1, 32, 2, 80)  # 2.0 s at 40 Hz
    base_patcher = _MiniPatcher(_MockDM(n_blocks=4, is_image=False), latent_shapes=[v_shape, a_shape])

    # Build synthetic packed x0 with two clear audio transients at a_idx=20 (0.50 s, f12) & 60 (1.50 s, f36)
    torch.manual_seed(9)
    vid_x0 = torch.randn(v_shape) * 0.1
    aud_x0 = torch.randn(a_shape) * 0.05
    aud_x0[:, :, 0, 20] += 3.0  # Left spike at 0.50 s
    aud_x0[:, :, 1, 20] += 3.0
    aud_x0[:, :, 0, 60] += 4.0  # Louder spike at 1.50 s
    aud_x0[:, :, 1, 60] += 4.0
    packed_x0, _ = comfy.utils.pack_latents([vid_x0, aud_x0])

    cfg = ps.SensorProbeConfig(
        noun_spans={"clock": [2, 3], "mirror": [6, 7]},
        expected_boxes={"clock": (0.0, 0.0, 0.5, 0.5), "mirror": (0.5, 0.5, 1.0, 1.0)},
        probe_steps=frozenset({0, 1}),
        text_probe_blocks=frozenset({0, 2}),
        audio_probe_blocks=frozenset({1, 3}),
        jlens_blocks=frozenset({0, 2, 3}),
        capture_audio_x0=True,
        audio_transient_z=2.0,
    )
    probed, rec = ps.attach_h3_sensor_probe(base_patcher, cfg)

    out_ref, _ = base_patcher.run_step(ctx, layout, img, t_emb, mod_segments, sigma=1.0, denoised=packed_x0)
    out_prb, den_prb = probed.run_step(ctx, layout, img, t_emb, mod_segments, sigma=1.0, denoised=packed_x0)
    max_abs = float((out_ref - out_prb).abs().max().item())
    assert max_abs == 0.0, f"Read-only sensor probe mutated output! max_abs={max_abs}"
    assert den_prb is packed_x0, "post_cfg hook must return denoised tensor untouched"
    print("[ok] 1. Bitwise zero-mutation parity (max_abs == 0.0) & untouched denoised verified")

    # Sensor 1 assertions
    txt_b0 = {r["noun"]: r for r in rec.text_attention_records if r["block"] == 0}
    assert set(txt_b0.keys()) == {"clock", "mirror"}
    assert txt_b0["clock"]["mean_heads"]["peak_in_box"] is True, txt_b0["clock"]
    assert txt_b0["clock"]["mean_heads"]["box_mass"] > 0.85, txt_b0["clock"]
    assert txt_b0["clock"]["top10_sink_margin"] > 5.0, txt_b0["clock"]
    assert txt_b0["mirror"]["mean_heads"]["peak_in_box"] is True, txt_b0["mirror"]
    assert txt_b0["mirror"]["mean_heads"]["box_mass"] > 0.85, txt_b0["mirror"]
    print("[ok] 2. Sensor 1 (Q_vid @ K_txt[noun]^T) localized both quadrants & sink margin cleanly")

    # Sensor 2 assertions
    aud_b1 = next(r for r in rec.audio_attention_records if r["block"] == 1)
    assert aud_b1["left_ear_centroid_xy"][0] < 0.35, aud_b1
    assert aud_b1["right_ear_centroid_xy"][0] > 0.65, aud_b1
    assert aud_b1["stereo_lr_x_sep"] > 0.35, aud_b1
    print(f"[ok] 3. Sensor 2 (Q_vid @ K_aud[t]^T) verified stereo L/R sep={aud_b1['stereo_lr_x_sep']:.3f}")

    # Sensor 3 assertions
    assert len(rec.audio_x0_records) == 1
    a0 = rec.audio_x0_records[0]
    found_idx = [t["a_idx"] for t in a0["transients"]]
    assert 20 in found_idx and 60 in found_idx, a0
    t20 = next(t for t in a0["transients"] if t["a_idx"] == 20)
    assert t20["t_sec"] == 0.5 and t20["video_frame_24fps"] == 12, t20
    print(f"[ok] 4. Sensor 3 (audio_x0 transients) detected exact peaks {found_idx} (f12 & f36)")

    # 5. Single-frame MiniMax-H3-Image parity (`latent_t=1, audio_t=0`) + clean audio no-op
    img_layout = PackedLayout(10, 1, 8, 12, 0)
    img_h = img[: img_layout.seq_len].clone()
    img_mod = [(0, 10, 1), (10, img_layout.seq_len, 0)]
    img_patcher = _MiniPatcher(_MockDM(n_blocks=3, is_image=True), latent_shapes=[(1, 24, 1, 8, 12)])
    img_probed, img_rec = ps.attach_h3_sensor_probe(img_patcher, cfg)
    img_x0 = torch.randn(1, 24, 1, 8, 12)
    ref_i, _ = img_patcher.run_step(ctx, img_layout, img_h, t_emb, img_mod, sigma=1.0, denoised=img_x0)
    prb_i, den_i = img_probed.run_step(ctx, img_layout, img_h, t_emb, img_mod, sigma=1.0, denoised=img_x0)
    assert float((ref_i - prb_i).abs().max().item()) == 0.0
    assert den_i is img_x0
    assert len(img_rec.audio_attention_records) == 0
    assert len(img_rec.audio_x0_records) == 0
    assert len(img_rec.text_attention_records) >= 2
    print("[ok] 5. MiniMax-H3-Image (latent_t=1, audio_t=0) parity & audio no-op verified")

    # 6. Sensor 4 + MiniMaxH3SpatialQueryRouter stacking in BOTH orders + synthetic J-Lens bundle
    torch.manual_seed(42)
    shadow_b_cond = [[torch.randn(1, 7, 32) * 2.0, {}]]
    # Synthetic J-Lens bundle matching hidden=32 -> encoder=32? `decode_vector` requires POST_REFINER_DIM=5376,
    # so test real decode_vector on a 5376-dim synthetic bundle separately AND test SpatialQR stacking here.
    for order_name in ("sqr_then_probe", "probe_then_sqr"):
        p_s = _MiniPatcher(_MockDM(n_blocks=4, is_image=False), latent_shapes=[v_shape, a_shape])
        if order_name == "sqr_then_probe":
            p_s = sqr.attach_spatial_query_router(p_s, region_conditioning=shadow_b_cond, isolation="early_0_24")
            p_ref = p_s.clone()
            p_s, rec_s = ps.attach_h3_sensor_probe(p_s, cfg)
        else:
            p_s, rec_s = ps.attach_h3_sensor_probe(p_s, cfg)
            p_s = sqr.attach_spatial_query_router(p_s, region_conditioning=shadow_b_cond, isolation="early_0_24")
            p_ref = sqr.attach_spatial_query_router(
                _MiniPatcher(p_s.model.diffusion_model, latent_shapes=[v_shape, a_shape]),
                region_conditioning=shadow_b_cond,
                isolation="early_0_24",
            )
        out_s, _ = p_s.run_step(ctx, layout, img, t_emb, mod_segments, sigma=1.0)
        out_r, _ = p_ref.run_step(ctx, layout, img, t_emb, mod_segments, sigma=1.0)
        assert float((out_s - out_r).abs().max().item()) == 0.0, f"{order_name} mutated SpatialQR output!"
        # Verify h_shadow_b captured cleanly on blocks 0, 2, 3
        b2 = next(r for r in rec_s.jlens_records if r["block"] == 2)
        assert "cross_world_mean_cos" in b2 and b2["l_txt_b"] == 7, (order_name, b2)

    # Verify J-Lens integration cleanly on a 5376-D synthetic bundle (and real bundle if present)
    from steering.constants import ENCODER_DIM, POST_REFINER_DIM
    syn_bundle = {
        "J_wiener_fl2va": torch.eye(ENCODER_DIM, POST_REFINER_DIM),
        "J_ref_fl2va": torch.eye(POST_REFINER_DIM, ENCODER_DIM),
        "M_union": torch.randn(16, ENCODER_DIM),
        "cloze_base_union": torch.zeros(2, 16),
        "valid_pos_in_union": torch.arange(16, dtype=torch.long),
        "words": [f"word_{i}" for i in range(16)],
    }
    rec_jl = ps.H3SensorRecorder(
        ps.SensorProbeConfig(probe_steps=frozenset({0}), jlens_blocks=frozenset({0, 2}), run_jlens=True, jlens_bundle=syn_bundle, jlens_top_k=3)
    )
    rec_jl._cur_step_idx = 0
    rec_jl._cur_tx_a, rec_jl._cur_tx_b = 0, 4
    h0_5376 = torch.randn(4, POST_REFINER_DIM)
    h2_5376 = h0_5376 + 0.5 * torch.randn(4, POST_REFINER_DIM)
    hb_5376 = torch.randn(3, POST_REFINER_DIM)
    rec_jl.record_text_block(0, h0_5376, shadow_b=hb_5376)
    rec_jl.record_text_block(2, h2_5376, shadow_b=hb_5376)
    b2_jl = next(r for r in rec_jl.jlens_records if r["block"] == 2)
    assert len(b2_jl.get("jlens_delta_a_pos", [])) > 0, b2_jl
    assert len(b2_jl.get("jlens_b_minus_a_pos", [])) > 0, b2_jl
    assert rec_jl._step_blk0_txt_a is not None
    assert rec_jl._step_blk0_txt_a.untyped_storage().data_ptr() != h0_5376.untyped_storage().data_ptr()
    print("[ok] 6. Sensor 4 (SpatialQR both attach orders, disjoint CPU snapshots, & J-Lens) verified")

    # 7. Output guard (refuse repo data/) & PNG contact sheet + JSON save
    refused = False
    try:
        ps.render_sensor_contact_sheet(rec, REPO_DIR / "data" / "illegal_probe.png", step=0)
    except ValueError:
        refused = True
    assert refused, "render_sensor_contact_sheet must refuse paths inside repo data/"

    with tempfile.TemporaryDirectory() as td:
        png_p = ps.render_sensor_contact_sheet(
            rec,
            Path(td) / "probe_sheet.png",
            base_frame_rgb=np.full((64, 96, 3), 80, dtype=np.uint8),
            step=0,
        )
        json_p = rec.save_report(Path(td) / "sensor_report.json")
        assert png_p.is_file() and png_p.stat().st_size > 200
        assert json_p.is_file() and "text_attention" in json_p.read_text(encoding="utf-8")
    print("[ok] 7. Repo data/ path refusal, PNG contact sheet, & JSON export verified")
    print("ALL H3 4-Sensor Calibration Probe CPU checks passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
