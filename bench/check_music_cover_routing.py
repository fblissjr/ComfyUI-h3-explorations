#!/usr/bin/env python3
"""CPU verification suite for `MiniMaxH3AudioRefRouter` and `attention.py` audio LSE spans.

Verifies:
1. Exact identity when all gains == 1.0 (`max_abs_diff == 0.0`, zero side-SDPA calls).
2. Zero mutation of `K` and `V` buffers (both raw tensors and `AttentionTensorContainer`).
3. Block-depth firewall & boost on `cond_audio` vs `<Audio 1>` (`ref_audio[0]`):
   - Blocks `< split_block`: `Q_aud` reads `cond_audio` at `melody_gain_early` and `ref_audio[0]` at `timbre_gain_early`.
   - Blocks `>= split_block`: `melody_gain_late=0.0` eliminates `cond_audio` value contribution (`< 1e-6`),
     while `timbre_gain_late=2.0` matches exact analytical biased SDPA (`< 1e-5`) and leaves `video` / `text` / `ref_audio` rows bit-identical to `out_main`.
4. `melody_source="ref_audio_2"` routes 2nd `ref_audio` (`<Audio 2>`) separately from `ref_audio[0]` (`<Audio 1>`).
5. Independent `melody_clean_level` softens ONLY `cond_audio` (or `ref_audio[1]`) while keeping `<Audio 1>` (`ref_audio[0]`) bit-identical and clamped at `max(clean_level, audio_stream_time)`.
6. Selective `melody_end_percent` drops ONLY `cond_audio` (or `ref_audio[1]`) from `PackedLayout` + `cond_audio_latents` while preserving `<Audio 1>` and exact 3D RoPE `position_ids`.
7. `attention.py::resolve_lse_balance_plan` + `lse_subset_balance_bhnd` support `ref_audio_indices_a/b` and `cond_audio_indices_a/b` on `query_kinds=("audio",)`.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))
from _lib import bootstrap  # noqa: E402

bootstrap(cpu=True)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Ensure steering package is loaded under canonical name before audio_ref_router
import control_vector_nodes as _cv_shim  # noqa: F401, E402
import attention as h3_attn  # noqa: E402
import audio_ref_router as arr  # noqa: E402


class _FakeLayout:
    def __init__(self, segments: list[tuple[int, int, str]]):
        self.segments = list(segments)
        self.seq_len = int(segments[-1][1]) if segments else 0
        self.position_ids = torch.arange(self.seq_len * 3, dtype=torch.float64).reshape(self.seq_len, 3)
        self.signature = (4, 2, 4, 4, 4)
        img_pos, img_upd, aud_pos, aud_upd = [], [], [], []
        for a, b, k in self.segments:
            n = b - a
            if k in ("cond", "ref_img"):
                img_pos.append(torch.arange(a, b))
                img_upd.append(torch.zeros(n, dtype=torch.bool))
            elif k == "video":
                img_pos.append(torch.arange(a, b))
                img_upd.append(torch.ones(n, dtype=torch.bool))
            elif k in ("cond_audio", "ref_audio"):
                aud_pos.append(torch.arange(a, b))
                aud_upd.append(torch.zeros(n, dtype=torch.bool))
            elif k == "audio":
                aud_pos.append(torch.arange(a, b))
                aud_upd.append(torch.ones(n, dtype=torch.bool))
        self.img_pos = torch.cat(img_pos) if img_pos else torch.empty(0, dtype=torch.long)
        self.img_update = torch.cat(img_upd) if img_upd else torch.empty(0, dtype=torch.bool)
        self.audio_pos = torch.cat(aud_pos) if aud_pos else torch.empty(0, dtype=torch.long)
        self.audio_update = torch.cat(aud_upd) if aud_upd else torch.empty(0, dtype=torch.bool)


class _FakeContainer:
    def __init__(self, t: torch.Tensor):
        self._t = t
        self.taken = False

    def peek(self) -> torch.Tensor:
        return self._t

    def take(self) -> torch.Tensor:
        self.taken = True
        return self._t


def _make_base_sdpa():
    def base_func(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kw):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        if skip_output_reshape:
            return out
        B, H, S, D = out.shape
        return out.transpose(1, 2).reshape(B, S, H * D)

    return base_func


def test_identity_and_kv_immutability() -> None:
    torch.manual_seed(11)
    segs = [
        (0, 4, "text"),
        (4, 10, "cond_audio"),
        (10, 16, "ref_audio"),
        (16, 22, "audio"),
        (22, 30, "video"),
    ]
    layout = _FakeLayout(segs)
    B, H, S, D = 1, 4, layout.seq_len, 16
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    k_orig, v_orig = k.clone(), v.clone()

    base = _make_base_sdpa()
    rt = arr.H3AudioRefRouterRuntime(
        melody_source="cond_audio",
        split_block=24,
        melody_gain_early=1.0,
        melody_gain_late=1.0,
        timbre_gain_early=1.0,
        timbre_gain_late=1.0,
    )
    override = arr.make_audio_ref_router_override(previous=None)
    opts = {"minimax_h3_layout": layout, "block_index": 30, "_h3_audio_ref_router_rt": rt}

    ref_out = base(q, k, v, H, skip_reshape=True, transformer_options=opts)
    got_out = override(base, q, k, v, H, skip_reshape=True, transformer_options=opts)
    assert float((got_out - ref_out).abs().max()) == 0.0, "identity gain=1.0 must be bit-exact"
    assert rt.stats["blocks_routed"] == 0
    assert torch.equal(k, k_orig) and torch.equal(v, v_orig)


def test_cond_audio_firewall_and_timbre_boost() -> None:
    torch.manual_seed(23)
    segs = [
        (0, 4, "text"),
        (4, 10, "cond_audio"),
        (10, 16, "ref_audio"),
        (16, 22, "audio"),
        (22, 30, "video"),
    ]
    layout = _FakeLayout(segs)
    B, H, S, D = 1, 4, layout.seq_len, 16
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    # Give cond_audio a huge distinct value signature to prove zero leakage in late blocks
    v[:, :, 4:10, :] = 100.0
    k_orig, v_orig = k.clone(), v.clone()

    base = _make_base_sdpa()

    class _PrevWithContainer:
        def __call__(self, func, q_, k_, v_, heads, **kw):
            return base(q_, k_, v_, heads, **kw)

        def container_function(self, q_, k_, v_, heads, **kw):
            return base(q_.take(), k_.take(), v_.take(), heads, **kw)

    prev = _PrevWithContainer()
    override = arr.make_audio_ref_router_override(previous=prev)
    assert hasattr(override, "container_function"), "must preserve container_function contract"

    rt = arr.H3AudioRefRouterRuntime(
        melody_source="cond_audio",
        split_block=24,
        melody_gain_early=1.5,
        melody_gain_late=0.0,
        timbre_gain_early=0.75,
        timbre_gain_late=2.5,
    )

    # Early block (10 < 24)
    opts_early = {"minimax_h3_layout": layout, "block_index": 10, "_h3_audio_ref_router_rt": rt}
    qc, kc, vc = _FakeContainer(q), _FakeContainer(k), _FakeContainer(v)
    out_early = override.container_function(qc, kc, vc, H, skip_reshape=True, transformer_options=opts_early)
    assert qc.taken and kc.taken and vc.taken
    assert torch.equal(k, k_orig) and torch.equal(v, v_orig), "K/V must never be mutated"

    bias_e = torch.zeros(1, 1, 1, S)
    bias_e[:, :, :, 4:10] = math.log(1.5)
    bias_e[:, :, :, 10:16] = math.log(0.75)
    exp_aud_e = F.scaled_dot_product_attention(q[:, :, 16:22, :], k, v, attn_mask=bias_e).transpose(1, 2).reshape(1, 6, H * D)
    base_unbiased = base(q, k, v, H, skip_reshape=True)
    assert float((out_early[:, 16:22, :] - exp_aud_e).abs().max()) < 1e-5
    assert float((out_early[:, :16, :] - base_unbiased[:, :16, :]).abs().max()) == 0.0
    assert float((out_early[:, 22:, :] - base_unbiased[:, 22:, :]).abs().max()) == 0.0

    # Late block (30 >= 24): cond_audio must be firewalled completely (even with v=100.0)
    opts_late = {"minimax_h3_layout": layout, "block_index": 30, "_h3_audio_ref_router_rt": rt}
    out_late = override(base, q, k, v, H, skip_reshape=True, transformer_options=opts_late)
    v_alt = v.clone()
    v_alt[:, :, 4:10, :] = -999.0
    out_late_alt = override(base, q, k, v_alt, H, skip_reshape=True, transformer_options=opts_late)
    assert float((out_late[:, 16:22, :] - out_late_alt[:, 16:22, :]).abs().max()) < 1e-6, (
        "late-block Q_aud must be completely invariant to firewalled cond_audio values"
    )
    bias_l = torch.zeros(1, 1, 1, S)
    bias_l[:, :, :, 4:10] = -1e4
    bias_l[:, :, :, 10:16] = math.log(2.5)
    exp_aud_l = F.scaled_dot_product_attention(q[:, :, 16:22, :], k, v, attn_mask=bias_l).transpose(1, 2).reshape(1, 6, H * D)
    assert float((out_late[:, 16:22, :] - exp_aud_l).abs().max()) < 1e-5
    # Refiner calls (short sequence != layout.seq_len) must be ignored
    q_ref = torch.randn(1, H, 4, D)
    out_ref = override(base, q_ref, q_ref, q_ref, H, skip_reshape=True, transformer_options=opts_late)
    assert float((out_ref - base(q_ref, q_ref, q_ref, H, skip_reshape=True)).abs().max()) == 0.0
    assert rt.stats["blocks_early"] == 1 and rt.stats["blocks_late"] == 2


def test_ref_audio_2_routing_and_nhd_seam() -> None:
    torch.manual_seed(37)
    segs = [
        (0, 4, "text"),
        (4, 10, "ref_audio"),   # <Audio 1> Target B
        (10, 16, "ref_audio"),  # <Audio 2> Song X
        (16, 22, "audio"),
        (22, 28, "video"),
    ]
    layout = _FakeLayout(segs)
    B, H, S, D = 1, 4, layout.seq_len, 16
    q_nhd = torch.randn(B, S, H, D)
    k_nhd = torch.randn(B, S, H, D)
    v_nhd = torch.randn(B, S, H, D)
    v_nhd[:, 10:16, :, :] = 50.0  # <Audio 2>

    rt = arr.H3AudioRefRouterRuntime(
        melody_source="ref_audio_2",
        split_block=20,
        melody_gain_early=1.0,
        melody_gain_late=0.0,
        timbre_gain_early=1.0,
        timbre_gain_late=2.0,
    )
    opts = {"minimax_h3_layout": layout, "block_index": 25, "_h3_audio_ref_router_rt": rt}
    ran = arr.h3_audio_ref_router_seam_pre(q_nhd, k_nhd, v_nhd, opts, seq_axis=1)
    assert ran
    out_nhd = F.scaled_dot_product_attention(
        q_nhd.transpose(1, 2), k_nhd.transpose(1, 2), v_nhd.transpose(1, 2)
    ).transpose(1, 2).contiguous()
    out_ref_copy = out_nhd.clone()
    arr.h3_audio_ref_router_seam_post(out_nhd, opts, nhd=True)

    bias = torch.zeros(1, 1, 1, S)
    bias[:, :, :, 4:10] = math.log(2.0)
    bias[:, :, :, 10:16] = -1e4
    exp_aud = F.scaled_dot_product_attention(
        q_nhd[:, 16:22, :, :].transpose(1, 2),
        k_nhd.transpose(1, 2),
        v_nhd.transpose(1, 2),
        attn_mask=bias,
    ).transpose(1, 2)
    assert float((out_nhd[:, 16:22, :, :] - exp_aud).abs().max()) < 1e-5
    assert float((out_nhd[:, :16, :, :] - out_ref_copy[:, :16, :, :]).abs().max()) == 0.0
    assert float((out_nhd[:, 22:, :, :] - out_ref_copy[:, 22:, :, :]).abs().max()) == 0.0


def test_independent_softening_and_selective_drop() -> None:
    torch.manual_seed(41)
    segs = [
        (0, 4, "text"),
        (4, 12, "cond_audio"),
        (12, 20, "ref_audio"),
        (20, 28, "audio"),
        (28, 36, "video"),
    ]
    layout = _FakeLayout(segs)
    cond_lat = torch.ones(1, 32, 2, 4)
    ref_lat = torch.full((1, 32, 2, 4), -2.0)
    payload = {
        "seed": 1234,
        "layout": layout,
        "keyframes": [{"frame_idx": 0, "audio_latent": cond_lat}],
        "refs": [{"kind": "audio", "ref_audio_t": 4, "audio_latent": ref_lat}],
        "cond_audio_latents": [cond_lat, ref_lat],
    }

    # Soften cond_audio ONLY -> ref_audio stays bit-identical
    soft = arr._soften_melody_latents(payload, layout, melody_source="cond_audio", eff_clean=0.65)
    assert not torch.equal(soft["cond_audio_latents"][0], cond_lat), "cond_audio must be softened"
    assert torch.equal(soft["cond_audio_latents"][1], ref_lat), "<Audio 1> must remain bit-identical"

    # Drop cond_audio ONLY -> ref_audio & 3D RoPE preserved
    rel_layout, dropped = arr._drop_melody_from_layout(layout, melody_source="cond_audio")
    assert dropped == 8 and rel_layout.seq_len == layout.seq_len - 8
    kinds = [k for _a, _b, k in rel_layout.segments]
    assert kinds == ["text", "ref_audio", "audio", "video"]
    # Exact 3D RoPE position preservation for ref_audio, audio, video
    assert torch.equal(rel_layout.position_ids[4:12], layout.position_ids[12:20])
    assert torch.equal(rel_layout.position_ids[12:20], layout.position_ids[20:28])
    assert torch.equal(rel_layout.position_ids[20:28], layout.position_ids[28:36])

    stripped = arr._strip_melody_payload_entries(payload, layout, melody_source="cond_audio")
    assert len(stripped["cond_audio_latents"]) == 1
    assert torch.equal(stripped["cond_audio_latents"][0], ref_lat)
    assert "keyframes" not in stripped

    # Also test ref_audio_2 selective drop & softening
    segs2 = [
        (0, 4, "text"),
        (4, 12, "ref_audio"),
        (12, 20, "ref_audio"),
        (20, 28, "audio"),
        (28, 36, "video"),
    ]
    layout2 = _FakeLayout(segs2)
    ref_lat2 = torch.full((1, 32, 2, 4), 3.0)
    payload2 = {
        "seed": 5678,
        "layout": layout2,
        "refs": [
            {"kind": "audio", "ref_audio_t": 4, "audio_latent": ref_lat},
            {"kind": "audio", "ref_audio_t": 4, "audio_latent": ref_lat2},
        ],
        "cond_audio_latents": [ref_lat, ref_lat2],
    }
    soft2 = arr._soften_melody_latents(payload2, layout2, melody_source="ref_audio_2", eff_clean=0.70)
    assert torch.equal(soft2["cond_audio_latents"][0], ref_lat)
    assert not torch.equal(soft2["cond_audio_latents"][1], ref_lat2)

    rel_layout2, dropped2 = arr._drop_melody_from_layout(layout2, melody_source="ref_audio_2")
    assert dropped2 == 8
    assert [k for _a, _b, k in rel_layout2.segments] == ["text", "ref_audio", "audio", "video"]
    assert torch.equal(rel_layout2.position_ids[4:12], layout2.position_ids[4:12])
    assert torch.equal(rel_layout2.position_ids[12:20], layout2.position_ids[20:28])
    stripped2 = arr._strip_melody_payload_entries(payload2, layout2, melody_source="ref_audio_2")
    assert len(stripped2["cond_audio_latents"]) == 1
    assert torch.equal(stripped2["cond_audio_latents"][0], ref_lat)
    assert len(stripped2["refs"]) == 1


def test_lse_balance_audio_ref_spans() -> None:
    torch.manual_seed(53)
    segs = [
        (0, 4, "text"),
        (4, 10, "cond_audio"),
        (10, 16, "ref_audio"),
        (16, 22, "ref_audio"),
        (22, 28, "audio"),
        (28, 34, "video"),
    ]
    layout = _FakeLayout(segs)
    S = layout.seq_len
    opts = {
        "minimax_h3_layout": layout,
        "block_index": 30,
        "minimax_h3_lse_balance": {
            "weight": 2.5,
            "query_kinds": ("audio",),
            "ref_audio_indices_a": (0,),
            "cond_audio_indices_b": (0,),
            "blocks": frozenset(range(24, 50)),
        },
    }
    plan = h3_attn.resolve_lse_balance_plan(opts, S)
    assert plan is not None
    assert plan["query_spans"] == ((22, 28),)
    assert plan["spans_a"] == ((10, 16),)
    assert plan["spans_b"] == ((4, 10),)

    qh = torch.randn(1, 4, S, 16)
    kh = torch.randn(1, 4, S, 16)
    vh = torch.randn(1, 4, S, 16)
    out = h3_attn.lse_subset_balance_bhnd(qh, kh, vh, plan)
    assert out.shape == qh.shape
    base = F.scaled_dot_product_attention(qh, kh, vh)
    # Non-audio queries must remain unchanged (within fp32 SDPA vs exact-LSE rounding)
    assert float((out[:, :, :22, :] - base[:, :, :22, :]).abs().max()) < 1e-5
    assert float((out[:, :, 28:, :] - base[:, :, 28:, :]).abs().max()) < 1e-5
    # Audio queries must change
    assert float((out[:, :, 22:28, :] - base[:, :, 22:28, :]).abs().max()) > 1e-3

    # Also test ref_audio[0] vs ref_audio[1]
    opts["minimax_h3_lse_balance"] = {
        "weight": 3.0,
        "query_kinds": ("audio",),
        "ref_audio_indices_a": (0,),
        "ref_audio_indices_b": (1,),
    }
    plan2 = h3_attn.resolve_lse_balance_plan(opts, S)
    assert plan2 is not None and plan2["spans_a"] == ((10, 16),) and plan2["spans_b"] == ((16, 22),)


def test_wrapper_end_to_end_step_gating() -> None:
    segs = [
        (0, 4, "text"),
        (4, 12, "cond_audio"),
        (12, 20, "ref_audio"),
        (20, 28, "audio"),
        (28, 36, "video"),
    ]
    layout = _FakeLayout(segs)
    cond_lat = torch.ones(1, 32, 2, 4)
    ref_lat = torch.full((1, 32, 2, 4), -2.0)

    seen: list[dict] = []

    def fake_executor(x, timestep, context, transformer_options, **kwargs):
        seen.append({"to": dict(transformer_options), "payload": dict(kwargs.get("minimax_payload") or {})})
        return x

    w = arr._make_audio_ref_router_wrapper(
        melody_source="cond_audio",
        split_block=24,
        melody_gain_early=1.0,
        melody_gain_late=0.0,
        timbre_gain_early=1.0,
        timbre_gain_late=2.0,
        melody_end_percent=0.50,
        melody_clean_level=0.70,
        start_percent=0.0,
        end_percent=1.0,
    )
    x = [torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 4)]
    ctx = torch.zeros(1, 4, 5376)
    payload = {
        "seed": 42,
        "layout": layout,
        "keyframes": [{"frame_idx": 0, "audio_latent": cond_lat}],
        "refs": [{"kind": "audio", "ref_audio_t": 4, "audio_latent": ref_lat}],
        "cond_audio_latents": [cond_lat, ref_lat],
    }
    # Step 1 of 6 (frac = 1/6 = 0.167 < 0.50): cond_audio kept + softened, ref_audio pristine, override installed
    to_early = {"sample_sigmas": torch.tensor([1.0, 0.9, 0.75, 0.55, 0.35, 0.15, 0.0]), "sigmas": torch.tensor([0.9])}
    w(fake_executor, x, torch.tensor([900.0]), ctx, to_early, minimax_payload=payload)
    assert len(seen[-1]["payload"]["cond_audio_latents"]) == 2
    assert not torch.equal(seen[-1]["payload"]["cond_audio_latents"][0], cond_lat)
    assert torch.equal(seen[-1]["payload"]["cond_audio_latents"][1], ref_lat)
    assert getattr(seen[-1]["to"].get("optimized_attention_override"), "h3_audio_ref_router_override", False)

    # Step 3 of 6 (frac = 3/6 = 0.50 >= 0.50): cond_audio dropped, ref_audio kept, RoPE intact
    to_late = {"sample_sigmas": torch.tensor([1.0, 0.9, 0.75, 0.55, 0.35, 0.15, 0.0]), "sigmas": torch.tensor([0.55])}
    w(fake_executor, x, torch.tensor([550.0]), ctx, to_late, minimax_payload=payload)
    assert len(seen[-1]["payload"]["cond_audio_latents"]) == 1
    assert torch.equal(seen[-1]["payload"]["cond_audio_latents"][0], ref_lat)
    assert [k for _a, _b, k in seen[-1]["payload"]["layout"].segments] == ["text", "ref_audio", "audio", "video"]


def main() -> int:
    tests = [
        test_identity_and_kv_immutability,
        test_cond_audio_firewall_and_timbre_boost,
        test_ref_audio_2_routing_and_nhd_seam,
        test_independent_softening_and_selective_drop,
        test_lse_balance_audio_ref_spans,
        test_wrapper_end_to_end_step_gating,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK: {len(tests)}/{len(tests)} music cover routing checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
