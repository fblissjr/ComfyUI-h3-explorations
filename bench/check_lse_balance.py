#!/usr/bin/env python3
"""Verify clean-room LSE subset attention balance, Sol/Provenance composition, and CUDA kernels.

Runs full CPU float64 oracle + override composition + container-protocol + `sol_observe`
verification under `CUDA_VISIBLE_DEVICES=`, and additionally exercises live CUDA FlashAttention-2
(`D = 128`) + `sage-fork` (`sageattn_consume(..., return_lse=True)`) when unmasked on a GPU.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import tempfile
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO.parent) not in sys.path:
    sys.path.insert(0, str(REPO.parent))

from _lib import bootstrap, card_visible, case, finish  # noqa: E402

bootstrap(cpu=None, front=True)
if not card_visible():
    import comfy.cli_args as _cli_args

    _cli_args.args.cpu = True

_PKG_NAME = REPO.name
att = importlib.import_module(f"{_PKG_NAME}.attention")
prov = importlib.import_module(f"{_PKG_NAME}.provenance")
sol = importlib.import_module(f"{_PKG_NAME}.sol_attn_h3")
sol_obs = importlib.import_module(f"{_PKG_NAME}.sol_observe")


def _explicit_reweighted_attn(
    qh: torch.Tensor,
    kh: torch.Tensor,
    vh: torch.Tensor,
    spans_a: tuple[tuple[int, int], ...],
    spans_b: tuple[tuple[int, int], ...] | None,
    query_spans: tuple[tuple[int, int], ...],
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Float64 softmax ground truth that directly reweights probs within `A cup B`."""
    qf = qh.to(torch.float64)
    kf = kh.to(torch.float64)
    vf = vh.to(torch.float64)
    d = qf.shape[-1]
    s = kf.shape[-2]
    scale = d ** -0.5
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    probs_orig = torch.softmax(scores, dim=-1)
    out_orig = torch.matmul(probs_orig, vf)

    mask_a = torch.zeros(s, dtype=torch.bool, device=qh.device)
    for a, b in spans_a:
        mask_a[a:b] = True

    if spans_b is not None:
        mask_b = torch.zeros(s, dtype=torch.bool, device=qh.device)
        for a, b in spans_b:
            mask_b[a:b] = True
        assert not bool((mask_a & mask_b).any())
        mask_comp = None
    else:
        mask_b = ~mask_a
        mask_comp = mask_b

    probs_mod = probs_orig.clone()
    pa = probs_orig[..., mask_a].sum(dim=-1, keepdim=True)
    pb = probs_orig[..., mask_b].sum(dim=-1, keepdim=True)
    sum_ab = pa + pb
    pa_new = (sum_ab * float(weight) * pa) / (float(weight) * pa + pb).clamp_min(1e-30)
    pb_new = sum_ab - pa_new
    scale_a = torch.where(pa > 0, pa_new / pa, torch.ones_like(pa))
    scale_b = torch.where(pb > 0, pb_new / pb, torch.ones_like(pb))
    probs_mod[..., mask_a] = probs_orig[..., mask_a] * scale_a
    probs_mod[..., mask_b] = probs_orig[..., mask_b] * scale_b

    q_mask = torch.zeros(s, dtype=torch.bool, device=qh.device)
    for qa, qb in query_spans:
        q_mask[qa:qb] = True
    probs_target = torch.where(q_mask.view(1, 1, s, 1), probs_mod, probs_orig)
    out_target = torch.matmul(probs_target, vf)
    return out_orig, out_target, probs_orig, probs_target


def check_identity_at_weight_one() -> None:
    torch.manual_seed(11)
    b, h, s, d = 1, 4, 48, 32
    qh = torch.randn(b, h, s, d, dtype=torch.float64)
    kh = torch.randn(b, h, s, d, dtype=torch.float64)
    vh = torch.randn(b, h, s, d, dtype=torch.float64)
    layout = types.SimpleNamespace(seq_len=s, segments=[(0, 8, "text"), (8, 14, "ref_img"), (14, 24, "audio"), (24, s, "video")])
    opts = {
        "minimax_h3_layout": layout,
        "block_index": 35,
        "minimax_h3_lse_balance": {"weight": 1.0, "spans_a": [(8, 14)], "blocks": frozenset({35})},
    }
    assert att.resolve_lse_balance_plan(opts, s) is None

    plan = {"weight": 1.0, "spans_a": ((8, 14),), "spans_b": ((24, 32),), "query_spans": ((32, s),), "block": 35}
    out = att.lse_subset_balance_bhnd(qh, kh, vh, plan)
    o_ref, _ = att._flash_or_exact_lse_bhnd(qh, kh, vh)
    diff = float((out - o_ref).abs().max().item())
    assert diff == 0.0, f"expected exact identity at w=1.0, got diff={diff}"


def check_two_subset_mass_conservation_float64() -> None:
    torch.manual_seed(23)
    b, h, s, d = 1, 4, 60, 32
    # [text: 0..8 | ref_anchor: 8..14 | audio: 14..22 | vid_ctx: 22..34 | vid_target: 34..60]
    qh = torch.randn(b, h, s, d, dtype=torch.float64)
    kh = torch.randn(b, h, s, d, dtype=torch.float64)
    vh = torch.randn(b, h, s, d, dtype=torch.float64)
    spans_a = ((8, 14),)
    spans_b = ((22, 34),)
    q_spans = ((34, s),)

    for weight in (4.0, 0.25):
        out_orig, out_ref, probs_orig, probs_target = _explicit_reweighted_attn(
            qh, kh, vh, spans_a, spans_b, q_spans, weight
        )
        # Verify exact conservation outside A cup B (`text` + `audio` + `vid_target`)
        outside_mask = torch.ones(s, dtype=torch.bool)
        outside_mask[8:14] = False
        outside_mask[22:34] = False
        out_mass_diff = float((probs_target[:, :, 34:, :][:, :, :, outside_mask] - probs_orig[:, :, 34:, :][:, :, :, outside_mask]).abs().max().item())
        assert out_mass_diff < 1e-13, f"tokens outside A cup B perturbed: {out_mass_diff}"

        pa_0 = probs_orig[:, :, 34:, 8:14].sum(dim=-1)
        pb_0 = probs_orig[:, :, 34:, 22:34].sum(dim=-1)
        pa_1 = probs_target[:, :, 34:, 8:14].sum(dim=-1)
        pb_1 = probs_target[:, :, 34:, 22:34].sum(dim=-1)
        assert float(((pa_1 + pb_1) - (pa_0 + pb_0)).abs().max().item()) < 1e-13
        ratio_ratio = ((pa_1 / pb_1) / (pa_0 / pb_0))
        assert float((ratio_ratio - weight).abs().max().item()) < 1e-12

        plan = {"weight": weight, "spans_a": spans_a, "spans_b": spans_b, "query_spans": q_spans, "block": 38}
        got = att.lse_subset_balance_bhnd(qh, kh, vh, plan)
        o_unbalanced, _ = att._flash_or_exact_lse_bhnd(qh, kh, vh)
        max_diff = float((got - out_ref).abs().max().item())
        assert max_diff < 1e-12, f"float64 LSE formula diff={max_diff} at weight={weight}"
        # Non-target query rows (0..34) must stay bitwise identical to unbalanced o_all
        nontarget_diff = float((got[:, :, :34] - o_unbalanced[:, :, :34]).abs().max().item())
        assert nontarget_diff == 0.0, f"non-target query rows mutated: {nontarget_diff}"
        assert float((got[:, :, :34] - out_orig[:, :, :34]).abs().max().item()) < 1e-14


def check_complement_mode_float64() -> None:
    torch.manual_seed(31)
    b, h, s, d = 1, 4, 52, 32
    qh = torch.randn(b, h, s, d, dtype=torch.float64)
    kh = torch.randn(b, h, s, d, dtype=torch.float64)
    vh = torch.randn(b, h, s, d, dtype=torch.float64)
    spans_a = ((10, 18),)
    q_spans = ((28, s),)
    weight = 3.5

    # Compare against adding +ln(weight) to logit scores for A
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * (d ** -0.5)
    scores_biased = scores.clone()
    scores_biased[:, :, 28:, 10:18] += math.log(weight)
    ref_out = torch.matmul(torch.softmax(scores_biased, dim=-1), vh)

    plan = {"weight": weight, "spans_a": spans_a, "spans_b": None, "query_spans": q_spans, "block": 12}
    got = att.lse_subset_balance_bhnd(qh, kh, vh, plan)
    max_diff = float((got - ref_out).abs().max().item())
    assert max_diff < 1e-12, f"complement mode diff vs logit+ln(w)={max_diff}"


def check_guards_and_refiner_exclusion() -> None:
    s = 40
    layout = types.SimpleNamespace(seq_len=s, segments=[(0, 8, "text"), (8, 14, "ref_img"), (14, s, "video")])
    opts = {
        "minimax_h3_layout": layout,
        "block_index": 35,
        "minimax_h3_lse_balance": {
            "weight": 4.0,
            "spans_a": [(8, 14)],
            "spans_b": [(12, 20)],
            "blocks": frozenset({35}),
        },
    }
    # Token refiner calls (shorter seq len than packed layout) are safely ignored
    assert att.resolve_lse_balance_plan(opts, 8) is None
    # Overlapping A and B must raise cleanly on the full sequence
    raised = False
    try:
        att.resolve_lse_balance_plan(opts, s)
    except ValueError:
        raised = True
    assert raised, "overlapping spans_a and spans_b did not raise ValueError"

    # Outside block set or outside sigma bounds returns None
    opts_ok = {
        "minimax_h3_layout": layout,
        "block_index": 20,
        "sigmas": torch.tensor([0.5]),
        "minimax_h3_lse_balance": {
            "weight": 4.0,
            "spans_a": [(8, 14)],
            "blocks": frozenset(range(30, 50)),
            "sigma_start": 1.0,
            "sigma_end": 0.8,
        },
    }
    assert att.resolve_lse_balance_plan(opts_ok, s) is None
    opts_ok["block_index"] = 35
    assert att.resolve_lse_balance_plan(opts_ok, s) is None
    opts_ok["sigmas"] = torch.tensor([0.9])
    assert att.resolve_lse_balance_plan(opts_ok, s) is not None


def check_override_composition_containers_and_sol_observe() -> None:
    from comfy.ldm.modules.attention import AttentionTensorContainer as Box, wrap_attn
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import CallbacksMP

    class TinyH3(torch.nn.Module):
        def __init__(self, n_blocks=50):
            super().__init__()
            self.diffusion_model = types.SimpleNamespace(blocks=[types.SimpleNamespace() for _ in range(n_blocks)])

    seen_under = {"entries": []}

    @wrap_attn
    def base_stock(q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False, **kw):
        seen_under["entries"].append("stock")
        return torch.zeros_like(q)

    def make_kitchen_fallback():
        @wrap_attn
        def kitchen_stub(q, k, v, heads, **kw):
            return q * 0.5

        def container_fn(q_box, k_box, v_box, heads, **kw):
            seen_under["entries"].append("kitchen_containers")
            qt = q_box.take()
            _kt = k_box.take()
            _vt = v_box.take()
            return qt + 2.0

        def override(func, q, k, v, heads, **kw):
            seen_under["entries"].append("kitchen_tensors")
            return q + 2.0

        override.container_function = container_fn
        return override

    def _build_mp_with_kitchen_and_sol(lse_first: bool) -> ModelPatcher:
        mp = ModelPatcher(TinyH3(), load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
        kitchen_ov = make_kitchen_fallback()
        mp.model_options["transformer_options"] = {"optimized_attention_override": kitchen_ov}
        if lse_first:
            mp = att.install_lse_balance_override(mp)
        settings = {"node": "MiniMaxH3Sol", "tau": 1.0, "min_tokens": 64, "dense_blocks": [49], "n_blocks": 50}
        sol_installed = set()

        def install_sol(to):
            cur = to.get("optimized_attention_override")
            if cur in sol_installed or sol._chain_contains(cur, sol_installed):
                return
            fb = sol._describe_override(cur)
            merged_settings = dict(settings, dense_fallback=fb)
            ov = sol.make_override(
                tau=1.0,
                min_tokens=64,
                sigma_start=1.0,
                sigma_end=0.1,
                sink_conditioning="off",
                dense_blocks=frozenset({49}),
                previous=cur,
                settings=merged_settings,
            )
            sol_installed.add(ov)
            to["optimized_attention_override"] = ov
            to["sol_compose"] = {"sigma_start": 1.0, "sigma_end": 0.1, "min_tokens": 64, "settings": merged_settings}

        install_sol(mp.model_options["transformer_options"])
        mp.add_callback_with_key(
            CallbacksMP.ON_PREPARE_STATE,
            "h3_sol",
            lambda _mp, _ts, mo: install_sol(mo["transformer_options"]),
        )
        if not lse_first:
            mp = att.install_lse_balance_override(mp)
        return mp

    # 1. Verify Provenance never reports "broken" without Sol, and reports "present" with Sol in both orders
    mp_no_sol = ModelPatcher(TinyH3(), load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    mp_no_sol = att.install_lse_balance_override(mp_no_sol)
    st_absent = prov._sol_state(mp_no_sol.model_options["transformer_options"], None)
    assert st_absent["state"] == "absent", f"lse without Sol reported broken: {st_absent}"
    assert st_absent.get("override_installed_by") == ["lse_balance"]

    for lse_first in (False, True):
        mp = _build_mp_with_kitchen_and_sol(lse_first=lse_first)
        top_ov = mp.model_options["transformer_options"]["optimized_attention_override"]
        assert hasattr(top_ov, "container_function"), f"container_function dropped for lse_first={lse_first}"
        st_present = prov._sol_state(mp.model_options["transformer_options"], torch.tensor([1.0, 0.5, 0.0]))
        assert st_present["state"] == "present", f"Sol state broke with lse_first={lse_first}: {st_present}"
        assert st_present["closure"]["tau"] == 1.0
        fb_str = mp.model_options["transformer_options"]["sol_compose"]["settings"]["dense_fallback"]
        assert "lse_balance" in fb_str, f"expected lse_balance in dense_fallback for lse_first={lse_first}, got {fb_str!r}"
        assert getattr(top_ov, "h3_kernel", None) is None
        under_sol = att._next_override(top_ov)
        assert getattr(under_sol, "h3_kernel", None) == "lse_balance"

        # Verify idempotence
        mp2 = att.install_lse_balance_override(mp)
        assert mp2.model_options["transformer_options"]["optimized_attention_override"] is top_ov

    # 2. Container preservation on inactive blocks vs LSE balance + `sol_observe` `"lse_balance"` route on active blocks
    mp = _build_mp_with_kitchen_and_sol(lse_first=False)
    ov = mp.model_options["transformer_options"]["optimized_attention_override"]
    s, h, d = 128, 2, 16
    layout = types.SimpleNamespace(
        seq_len=s,
        segments=[(0, 16, "text"), (16, 24, "ref_img"), (24, 48, "audio"), (48, s, "video")],
    )
    lse_cfg = {
        "weight": 4.0,
        "spans_a": ((16, 24),),
        "spans_b": ((48, 72),),
        "query_spans": ((72, s),),
        "blocks": frozenset(range(30, 50)),
        "require_layout": True,
    }

    with tempfile.TemporaryDirectory(prefix="lse_obs_") as tmp:
        sol_obs.arm(f"dir={tmp},raw=0")
        try:
            torch.manual_seed(41)
            q0 = torch.randn(1, h, s, d, dtype=torch.float32)
            k0 = torch.randn(1, h, s, d, dtype=torch.float32)
            v0 = torch.randn(1, h, s, d, dtype=torch.float32)

            # Call A: dense_block 49 WITHOUT lse_cfg active -> containers pass straight to kitchen_containers untaken
            seen_under["entries"].clear()
            boxes_a = [Box(q0.clone()), Box(k0.clone()), Box(v0.clone())]
            opts_a = {
                "minimax_h3_layout": layout,
                "block_index": 49,
                "sigmas": torch.tensor([0.9]),
                "optimized_attention_override": ov,
            }
            out_a = base_stock(*boxes_a, h, mask=None, skip_reshape=True, skip_output_reshape=True, transformer_options=opts_a)
            assert seen_under["entries"] == ["kitchen_containers"], f"containers not passed untaken: {seen_under['entries']}"
            assert all(b.tensor is None for b in boxes_a)
            assert torch.equal(out_a, q0 + 2.0)
            assert opts_a.get("h3_attn_route") == "dense_block"

            # Call B: block 35 WITH lse_cfg active -> Sol routes `"lse_balance"` and calls dense() -> LSE executes cleanly
            seen_under["entries"].clear()
            boxes_b = [Box(q0.clone()), Box(k0.clone()), Box(v0.clone())]
            opts_b = {
                "minimax_h3_layout": layout,
                "block_index": 35,
                "sigmas": torch.tensor([0.9]),
                "optimized_attention_override": ov,
                "minimax_h3_lse_balance": lse_cfg,
            }
            out_b = base_stock(*boxes_b, h, mask=None, skip_reshape=True, skip_output_reshape=True, transformer_options=opts_b)
            assert opts_b.get("h3_attn_route") == "lse_balance", f"expected route 'lse_balance', got {opts_b.get('h3_attn_route')!r}"
            assert not seen_under["entries"], f"lse_balance fell through unexpectedly: {seen_under['entries']}"
            assert all(b.tensor is None for b in boxes_b)
            want_b = att.lse_subset_balance_bhnd(q0, k0, v0, {**lse_cfg, "block": 35})
            assert float((out_b - want_b).abs().max().item()) < 1e-6

            # Call C: block 10 WITH lse_cfg active -> block 10 is NOT in 30-49, so Sol runs normally (not lse_balance)
            boxes_c = [Box(q0.clone()), Box(k0.clone()), Box(v0.clone())]
            opts_c = {
                "minimax_h3_layout": layout,
                "block_index": 10,
                "sigmas": torch.tensor([0.05]),  # outside [0.1, 1.0] on CPU so dense fallback handles without CUDA kernel
                "optimized_attention_override": ov,
                "minimax_h3_lse_balance": lse_cfg,
            }
            out_c = base_stock(*boxes_c, h, mask=None, skip_reshape=True, skip_output_reshape=True, transformer_options=opts_c)
            assert opts_c.get("h3_attn_route") == "outside_range", f"block 10 should not route lse_balance: {opts_c.get('h3_attn_route')!r}"
            assert torch.equal(out_c, q0 + 2.0)

            rows = sol_obs.read_rows(sorted(Path(tmp).glob("sol_observe_*.jsonl"))[0])
            call_routes = [(r["block"], r["route"]) for r in rows if r.get("kind") == "call"]
            assert call_routes == [(49, "dense_block"), (35, "lse_balance"), (10, "outside_range")], (
                f"unexpected sol_observe call routes: {call_routes}"
            )
        finally:
            sol_obs.arm(None)


def check_cuda_flash_and_sage_fork_parity() -> None:
    if os.environ.get("H3_CHECK_SWEEP") or not torch.cuda.is_available():
        return
    import comfy.ops
    from comfy.ldm.minimax.model import Attention, rope_rotation_table

    torch.manual_seed(53)
    b, h, s, d = 1, 8, 320, 128
    qh = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16)
    kh = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16)
    vh = torch.randn(b, h, s, d, device="cuda", dtype=torch.bfloat16)
    spans_a = ((16, 48),)
    spans_b = ((80, 176),)
    q_spans = ((176, s),)
    weight = 5.0

    _, out_ref, _, _ = _explicit_reweighted_attn(qh, kh, vh, spans_a, spans_b, q_spans, weight)
    plan = {"weight": weight, "spans_a": spans_a, "spans_b": spans_b, "query_spans": q_spans, "block": 35}

    # 1. Pure PyTorch FlashAttention-2 HND override (no Sage)
    flash_ov = att.make_lse_balance_override(previous=None)
    layout = types.SimpleNamespace(seq_len=s, segments=[(0, 16, "text"), (16, 48, "ref_img"), (48, 80, "audio"), (80, s, "video")])
    opts = {
        "minimax_h3_layout": layout,
        "block_index": 35,
        "minimax_h3_lse_balance": {
            "weight": weight,
            "spans_a": spans_a,
            "spans_b": spans_b,
            "query_spans": q_spans,
            "blocks": frozenset({35}),
        },
    }
    got_flash = flash_ov(
        lambda *a, **kw: qh,
        qh,
        kh,
        vh,
        h,
        skip_reshape=True,
        skip_output_reshape=True,
        transformer_options=opts,
    )
    flash_mae = float((got_flash.to(torch.float64) - out_ref).abs().mean().item())
    assert flash_mae < 5e-3, f"CUDA FlashAttention-2 LSE MAE too high: {flash_mae}"

    # 2. Sage-fork return_lse=True backed HND override
    kfn, kkw = att.build_kernel("auto")
    sage_ov = att.make_sage_override(kfn, kkw, previous=None)
    lse_over_sage = att.make_lse_balance_override(previous=sage_ov)
    got_sage_lse = lse_over_sage(
        lambda *a, **kw: qh,
        qh,
        kh,
        vh,
        h,
        skip_reshape=True,
        skip_output_reshape=True,
        transformer_options=opts,
    )
    sage_mae = float((got_sage_lse.to(torch.float64) - out_ref).abs().mean().item())
    assert sage_mae < 1.5e-2, f"CUDA sage-fork return_lse MAE too high: {sage_mae}"

    # 3. Direct make_minimax_attn_forward with fused QKV on GPU (when LSE active vs inactive)
    hidden = h * d
    mod = Attention(hidden, h, d, 1e-5, dtype=torch.bfloat16, device="cuda", operations=comfy.ops.manual_cast).to("cuda").requires_grad_(False)
    mod.qkv_proj.weight.normal_(0.0, 0.02)
    mod.q_norm.weight.fill_(1.0)
    mod.k_norm.weight.fill_(1.0)
    mod.out_proj = torch.nn.Identity()
    fwd = att.make_minimax_attn_forward(kfn, kkw, head_chunks=1, clone_v=att.mode_releases_qkv("auto"))
    mod.forward = fwd.__get__(mod, mod.__class__)
    rope = rope_rotation_table(torch.zeros(s, 96, device="cuda", dtype=torch.float32), torch.bfloat16)
    x_in = torch.randn(s, hidden, device="cuda", dtype=torch.bfloat16)
    out_active = mod(x_in, rope_freqs=rope, transformer_options=opts)
    assert out_active.shape == (s, hidden), f"unexpected shape: {out_active.shape}"
    assert bool(torch.isfinite(out_active).all()), "out_active had non-finite elements"
    opts_off = {
        "minimax_h3_layout": layout,
        "block_index": 12,
        "minimax_h3_lse_balance": opts["minimax_h3_lse_balance"],
    }
    out_inactive = mod(x_in, rope_freqs=rope, transformer_options=opts_off)
    assert out_inactive.shape == (s, hidden), f"unexpected inactive shape: {out_inactive.shape}"
    assert bool(torch.isfinite(out_inactive).all()), "out_inactive had non-finite elements"
    assert not torch.equal(out_active, out_inactive), "active and inactive LSE outputs were equal"


def main() -> int:
    case("identity_at_weight_one", check_identity_at_weight_one)
    case("two_subset_mass_conservation_float64", check_two_subset_mass_conservation_float64)
    case("complement_mode_float64", check_complement_mode_float64)
    case("guards_and_refiner_exclusion", check_guards_and_refiner_exclusion)
    case("override_composition_containers_and_sol_observe", check_override_composition_containers_and_sol_observe)
    if card_visible() and not os.environ.get("H3_CHECK_SWEEP"):
        case("cuda_flash_and_sage_fork_parity", check_cuda_flash_and_sage_fork_parity)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
