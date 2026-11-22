#!/usr/bin/env python3
"""Reproducible CPU mathematical verification for Clamped Pole-A Subspace Erasure.

Consolidates the core proofs from the 2026-11-20 unpooled multi-vector study into a
fast (`< 2 s`), zero-GPU, zero-owner-path CLI verifier:

1. Algebraic ridge split (`float64`): regressing `D = B_matched - A` onto centered `A`
   equals `W_B - P_{V_r, lambda}` (`max_abs_err <= 1e-13`), and reports the ratio of
   Pole-A erasure `||ctx_{-A}||` vs. Pole-B transport `||ctx_B||` plus `cos(ctx_D, -A)`.
2. Balanced Sinkhorn OT collapse vs. asymmetric row-softmax (`tau = 0.15`): proves on
   variable-length `(N_A, N_B)` pairs that uniform-marginal Sinkhorn OT algebraically
   collapses `mean(N_A * T @ B - A)` back to `mean(B) - mean(A)` (`cos == 1.000000`),
   whereas row-softmax (`tau = 0.15`) preserves distinct per-token cross-attention
   alignment (`erasure.mu_d`).
3. Runtime operator & union composition invariants: verifies `s == 0` bitwise identity,
   `s < 0` exact equivalence to 1D `add_norm_matched` (`clip(s, 0, 1) == 0`), strict
   `|s| * 67.5` RMS shift matching for `s > 0`, per-row `hold_words` orthogonality, and
   rank-16 union preservation in `compose_erasure_payload`.

Runs against synthetic planted banks by default and automatically checks any saved
`data/exp_unpooled_sliders/banks/*.pt` and `$H3_MODEL_STORE/control_vectors/bank/*.safetensors`
when present on disk.

Usage:
    <comfy-venv-python> bench/verify_pole_a_erasure.py
    <comfy-venv-python> bench/verify_pole_a_erasure.py --cpu-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != (REPO / "bench").resolve()]
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO))

from _lib import bootstrap  # noqa: E402

bootstrap(cpu=True)

import control_vectors as cv  # noqa: E402


def _sinkhorn_uniform(cost: torch.Tensor, *, eps: float = 0.15, n_iters: int = 80) -> torch.Tensor:
    """Balanced Sinkhorn OT with uniform marginals `a = 1/N_A`, `b = 1/N_B`."""
    n_a, n_b = cost.shape
    a = torch.full((n_a,), 1.0 / n_a, dtype=torch.float64)
    b = torch.full((n_b,), 1.0 / n_b, dtype=torch.float64)
    k = torch.exp((-cost / eps).clamp(min=-60.0, max=60.0))
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(n_iters):
        u = a / (k @ v).clamp(min=1e-30)
        v = b / (k.T @ u).clamp(min=1e-30)
    return u.unsqueeze(1) * k * v.unsqueeze(0)


def check_algebraic_ridge_split(max_rank: int = cv.ERASURE_MAX_RANK, ridge_rel: float = cv.ERASURE_RIDGE_REL) -> list[str]:
    errs: list[str] = []
    gen = torch.Generator().manual_seed(20261120)
    d = 128

    # Build a synthetic multi-facet bank where Pole A has a strong 6-D color/daylight subspace
    # orthogonal to the target Pole-B shift, mirroring `noir` (`indoor_neutral`) on disk.
    q_all, _ = torch.linalg.qr(torch.randn((d, 16), generator=gen, dtype=torch.float64), mode="reduced")
    sub_a = q_all[:, :6]
    dir_b = q_all[:, 8]

    n_a, n_b, m_test = 96, 132, 32
    a_mat = torch.randn((n_a, d), generator=gen, dtype=torch.float64) * 0.4 + (
        torch.randn((n_a, 6), generator=gen, dtype=torch.float64) @ sub_a.T
    ) * 4.5
    b_raw = torch.randn((n_b, d), generator=gen, dtype=torch.float64) * 0.4 + 2.5 * dir_b.unsqueeze(0)
    a_u = a_mat / torch.linalg.vector_norm(a_mat, dim=-1, keepdim=True).clamp(min=1e-12)
    b_u = b_raw / torch.linalg.vector_norm(b_raw, dim=-1, keepdim=True).clamp(min=1e-12)
    b_matched = torch.softmax((a_u @ b_u.T) / cv.ERASURE_MATCH_TAU, dim=-1) @ b_raw

    d_mat = b_matched - a_mat
    mu_a = a_mat.mean(dim=0, keepdim=True)
    mu_b = b_matched.mean(dim=0, keepdim=True)
    a_c = a_mat - mu_a
    b_c = b_matched - mu_b
    d_c = d_mat - (mu_b - mu_a)

    u_svd, s_vals, vh_svd = torch.linalg.svd(a_c, full_matrices=False)
    r = min(max_rank, int((s_vals > 1e-9 * s_vals[0]).sum().item()))
    u_r, s_r, v_r = u_svd[:, :r], s_vals[:r], vh_svd[:r]
    lam = ridge_rel * float(s_vals[0].item() ** 2)

    w_d = v_r.T @ torch.diag(s_r / (s_r ** 2 + lam)) @ (u_r.T @ d_c)
    w_b = v_r.T @ torch.diag(s_r / (s_r ** 2 + lam)) @ (u_r.T @ b_c)
    p_vr = v_r.T @ torch.diag((s_r ** 2) / (s_r ** 2 + lam)) @ v_r

    h_test = torch.randn((m_test, d), generator=gen, dtype=torch.float64) * 0.4 + (
        torch.randn((m_test, 6), generator=gen, dtype=torch.float64) @ sub_a.T
    ) * 4.0
    h_c = h_test - mu_a
    lhs = h_c @ w_d
    ctx_b = h_c @ w_b
    ctx_neg_a = -(h_c @ p_vr)
    rhs = ctx_b + ctx_neg_a

    max_err = float((lhs - rhs).abs().max().item())
    cos_neg_a = float(
        (torch.sum(lhs * ctx_neg_a) / (torch.linalg.vector_norm(lhs) * torch.linalg.vector_norm(ctx_neg_a))).item()
    )
    print(
        f"  [1/3] Synthetic ridge split: max_err={max_err:.3e}, "
        f"||ctx_-A||={torch.linalg.vector_norm(ctx_neg_a).item():.2f} vs "
        f"||ctx_B||={torch.linalg.vector_norm(ctx_b).item():.2f}, cos(ctx_D,-A)={cos_neg_a:+.4f}"
    )
    if max_err > 1e-12:
        errs.append(f"algebraic ridge split identity exceeded 1e-12: {max_err:.3e}")
    if cos_neg_a < 0.90:
        errs.append(f"expected dominant -A alignment on synthetic multi-facet bank, got cos={cos_neg_a:.4f}")

    # Optional on-disk saved `.pt` bank check (from data/exp_unpooled_sliders/banks/*.pt)
    bank_dir = REPO / "data" / "exp_unpooled_sliders" / "banks"
    bank_pts = sorted(bank_dir.glob("*.pt")) if bank_dir.is_dir() else []
    if bank_pts:
        worst_disk_err = 0.0
        min_disk_cos = 1.0
        for bp in bank_pts:
            payload = torch.load(bp, map_location="cpu", weights_only=False)
            if "A_all" in payload and "D_matched" in payload:
                a_all = payload["A_all"].double()
                bm_all = a_all + payload["D_matched"].double()
            else:
                pairs = payload.get("pairs") or []
                if not pairs:
                    continue
                a_all = torch.cat([p["a"].double() for p in pairs], dim=0)
                bm_list = []
                for p in pairs:
                    ap = p["a"].double()
                    bp_t = p["b"].double()
                    au = ap / torch.linalg.vector_norm(ap, dim=-1, keepdim=True).clamp(min=1e-12)
                    bu = bp_t / torch.linalg.vector_norm(bp_t, dim=-1, keepdim=True).clamp(min=1e-12)
                    bm_list.append(torch.softmax((au @ bu.T) / cv.ERASURE_MATCH_TAU, dim=-1) @ bp_t)
                bm_all = torch.cat(bm_list, dim=0)
            mu_a_d = a_all.mean(dim=0, keepdim=True)
            mu_b_d = bm_all.mean(dim=0, keepdim=True)
            ac_d = a_all - mu_a_d
            bc_d = bm_all - mu_b_d
            dc_d = bc_d - ac_d
            ud, sd, vhd = torch.linalg.svd(ac_d, full_matrices=False)
            rd = min(max_rank, int((sd > 1e-9 * sd[0]).sum().item()))
            ur, sr, vr = ud[:, :rd], sd[:rd], vhd[:rd]
            lam_d = ridge_rel * float(sd[0].item() ** 2)
            wd = vr.T @ torch.diag(sr / (sr ** 2 + lam_d)) @ (ur.T @ dc_d)
            wb = vr.T @ torch.diag(sr / (sr ** 2 + lam_d)) @ (ur.T @ bc_d)
            pv = vr.T @ torch.diag((sr ** 2) / (sr ** 2 + lam_d)) @ vr
            # Test on held-out / independent probe rows rather than training rows (ac_d),
            # since training rows satisfy (ac_d @ wd) == ac_d @ (wb - pv) identically while
            # also checking cos(ctx_D, -A) on the actual test prompt `indoor_neutral.pt` if present.
            probe_pt = REPO / "data" / "exp_unpooled_sliders" / "prompts" / "indoor_neutral.pt"
            if probe_pt.is_file():
                p_doc = torch.load(probe_pt, map_location="cpu", weights_only=False)
                p_ref = p_doc["ref"]
                p_sm = p_doc["scene_mask"]
                h_probe = (p_ref[0, p_sm] if p_ref.ndim == 3 else p_ref[p_sm]).double()
                hc = h_probe - mu_a_d
            else:
                hc = ac_d[: min(32, ac_d.shape[0])]
            l_d = hc @ wd
            na_d = -(hc @ pv)
            r_d = (hc @ wb) + na_d
            err_d = float((l_d - r_d).abs().max().item())
            cos_d = float((torch.sum(l_d * na_d) / (torch.linalg.vector_norm(l_d) * torch.linalg.vector_norm(na_d))).item())
            worst_disk_err = max(worst_disk_err, err_d)
            min_disk_cos = min(min_disk_cos, cos_d)
        print(
            f"        On-disk banks ({len(bank_pts)} files): max_err={worst_disk_err:.3e}, min_cos(ctx_D,-A)={min_disk_cos:+.4f}"
        )
        if worst_disk_err > 1e-11:
            errs.append(f"on-disk algebraic ridge split exceeded 1e-11: {worst_disk_err:.3e}")
    return errs


def check_sinkhorn_collapse_vs_row_softmax() -> list[str]:
    errs: list[str] = []
    gen = torch.Generator().manual_seed(42)
    d = 64
    a_spans = []
    b_spans = []
    for n_a, n_b in ((9, 17), (12, 23), (7, 15)):
        a_p = torch.randn((n_a, d), generator=gen, dtype=torch.float64)
        b_p = torch.randn((n_b, d), generator=gen, dtype=torch.float64)
        # Make half of B_p strongly aligned with A_p's first token and half orthogonal
        b_p[: n_b // 2] = a_p[:1] * 2.0 + torch.randn((n_b // 2, d), generator=gen, dtype=torch.float64) * 0.2
        a_spans.append(a_p)
        b_spans.append(b_p)

    # 1. Exact 1D mean difference across pairs (per-pair average of mean(B_p) - mean(A_p))
    # Weighted by N_A/N_B to match token-sum mean(B) - mean(A):
    v_1d_diffs = []
    sink_diffs = []
    rs_diffs = []
    for a_p, b_p in zip(a_spans, b_spans):
        n_a, n_b = a_p.shape[0], b_p.shape[0]
        au = a_p / torch.linalg.vector_norm(a_p, dim=-1, keepdim=True).clamp(min=1e-12)
        bu = b_p / torch.linalg.vector_norm(b_p, dim=-1, keepdim=True).clamp(min=1e-12)
        sim = au @ bu.T
        t_ot = _sinkhorn_uniform(1.0 - sim, eps=cv.ERASURE_MATCH_TAU)
        d_ot = (n_a * t_ot) @ b_p - a_p
        d_rs = torch.softmax(sim / cv.ERASURE_MATCH_TAU, dim=-1) @ b_p - a_p
        v_1d_diffs.append(b_p.mean(dim=0) - a_p.mean(dim=0))
        sink_diffs.append(d_ot.mean(dim=0))
        rs_diffs.append(d_rs.mean(dim=0))

    max_ot_collapse_err = max(float((s - v).abs().max().item()) for s, v in zip(sink_diffs, v_1d_diffs))
    cos_ot_1d = min(
        float((torch.dot(s, v) / (torch.linalg.vector_norm(s) * torch.linalg.vector_norm(v))).item())
        for s, v in zip(sink_diffs, v_1d_diffs)
    )
    cos_rs_1d = min(
        float((torch.dot(r, v) / (torch.linalg.vector_norm(r) * torch.linalg.vector_norm(v))).item())
        for r, v in zip(rs_diffs, v_1d_diffs)
    )
    print(
        f"  [2/3] OT vs row-softmax: Sinkhorn uniform collapse max_err={max_ot_collapse_err:.3e} "
        f"(cos_1d={cos_ot_1d:.6f}) vs row-softmax min cos_1d={cos_rs_1d:.4f}"
    )
    if max_ot_collapse_err > 1e-7 or cos_ot_1d < 0.99999:
        errs.append(f"expected Sinkhorn uniform marginals to collapse to v_1d, got err={max_ot_collapse_err:.3e}")
    if cos_rs_1d > 0.999:
        errs.append(f"expected row-softmax to diverge from naive mean difference, got cos={cos_rs_1d:.6f}")
    return errs


def check_runtime_invariants_and_union() -> list[str]:
    errs: list[str] = []
    gen = torch.Generator().manual_seed(99)
    d = cv.POST_REFINER_DIM

    q, _ = torch.linalg.qr(torch.randn((d, 16), generator=gen, dtype=torch.float64), mode="reduced")
    vr1 = q[:, :8].T.to(dtype=torch.float32).contiguous()
    vr2 = q[:, 8:16].T.to(dtype=torch.float32).contiguous()
    diag1 = torch.linspace(0.95, 0.45, 8, dtype=torch.float32)
    diag2 = torch.linspace(0.92, 0.40, 8, dtype=torch.float32)
    mu1 = torch.randn(d, generator=gen, dtype=torch.float32)
    mu2 = torch.randn(d, generator=gen, dtype=torch.float32)
    hold = q[:, 0].to(dtype=torch.float32)

    ctx = torch.randn((1, 14, d), generator=gen, dtype=torch.bfloat16)
    tags = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)

    # 1. s = 0 identity
    out0 = cv.apply_post_refiner(ctx, tags, [(mu1, 0.0, "add_norm_matched_erasure")], erasure_specs=[(vr1, diag1, mu1)])
    if not torch.equal(out0, ctx):
        errs.append("s=0.0 did not preserve context bitwise")

    # 2. s < 0 disables erasure cleanly and matches 1D add_norm_matched
    out_neg_er = cv.apply_post_refiner(ctx, tags, [(mu1, -0.75, "add_norm_matched_erasure")], erasure_specs=[(vr1, diag1, mu1)])
    out_neg_1d = cv.apply_post_refiner(ctx, tags, [(mu1, -0.75, "add_norm_matched")])
    if not torch.equal(out_neg_er, out_neg_1d):
        errs.append("s<0 add_norm_matched_erasure did not match 1D add_norm_matched")

    # 3. Positive strength strict RMS norm = |s| * 67.5 + vision rows untouched
    s_pos = 1.5
    out_pos = cv.apply_post_refiner(ctx, tags, [(mu1, s_pos, "add_norm_matched_erasure")], erasure_specs=[(vr1, diag1, mu1)])
    if not torch.equal(out_pos[:, :2], ctx[:, :2]):
        errs.append("tag=0 vision rows mutated by add_norm_matched_erasure")
    delta = (out_pos[0, 2:].to(torch.float64) - ctx[0, 2:].to(torch.float64))
    got_rms = cv._rms_row_norm64(delta)
    exp_rms = abs(s_pos) * cv.NORM_MATCHED_TARGET
    if abs(got_rms - exp_rms) > 0.6:  # bfloat16 rounding tolerance over D=5376
        errs.append(f"positive shift RMS {got_rms:.3f} deviated from expected {exp_rms:.3f}")

    # 4. Per-row hold scrubbing leaves zero residual along held direction
    out_hold = cv.apply_post_refiner(
        ctx,
        tags,
        [(mu1, s_pos, "add_norm_matched_erasure")],
        erasure_specs=[(vr1, diag1, mu1, [(hold, 1.0)])],
    )
    delta_h = (out_hold[0, 2:].to(torch.float64) - ctx[0, 2:].to(torch.float64))
    max_hold_proj = float((delta_h @ hold.double()).abs().max().item())
    if max_hold_proj > 0.5:  # bf16 rounding on row ~268 norm
        errs.append(f"per-row hold projection residual too large: {max_hold_proj:.4f}")

    # 5. Clamped union multi-slider stacking invariants (K=1 exact, 0.5+0.5 split continuity,
    #    overlapping zero sign-inversion, orthogonal un-squashed norm, opposing-boost -e preservation)
    mu1_unit = mu1 / torch.linalg.vector_norm(mu1.double())
    ctx_f32 = torch.randn((1, 14, d), generator=gen, dtype=torch.float32)
    out_single_f32 = cv.apply_post_refiner(
        ctx_f32, tags, [(mu1, 1.0, "add_norm_matched_erasure")], erasure_specs=[(vr1, diag1, mu1)]
    )
    direct_row_sh = cv._erasure_row_shift64(
        ctx_f32[0, 2:].double(), mu1, 1.0, (vr1, diag1, mu1)
    ).float()
    if float((out_single_f32[0, 2:] - (ctx_f32[0, 2:] + direct_row_sh)).abs().max().item()) > 1e-6:
        errs.append("K=1 apply_post_refiner diverged from _erasure_row_shift64")

    # 0.5 + 0.5 split of same vector preserves e_budget = 67.5 and matches s=1.0
    out_split = cv.apply_post_refiner(
        ctx_f32,
        tags,
        [(mu1, 0.5, "add_norm_matched_erasure"), (mu1, 0.5, "add_norm_matched_erasure")],
        erasure_specs=[(vr1, diag1, mu1), (vr1, diag1, mu1)],
        warn_if_stacked_norm=False,
    )
    split_err = float((out_split - out_single_f32).abs().max().item())
    if split_err > 1e-4:
        errs.append(f"0.5 + 0.5 split continuity failed: max_diff={split_err:.6g}")

    # Identical rank-8 basis stacked twice at s=1.0 must clamp d_union <= 1.0 (zero -1 sign-inversion!)
    diag_ones = torch.ones_like(diag1)
    ctx_in_sub = torch.zeros((1, 14, d), dtype=torch.float64)
    # Row variance strictly inside vr1[0]
    ctx_in_sub[0, 2:8] = vr1[0].double().unsqueeze(0) * 10.0
    ctx_in_sub[0, 8:14] = -vr1[0].double().unsqueeze(0) * 10.0
    sh_dup = cv._stacked_erasure_block_shift64(
        ctx_in_sub[0, 2:],
        [(mu1, 1.0, (vr1, diag_ones, mu1)), (mu1, 1.0, (vr1, diag_ones, mu1))],
    )
    # Remove add_total mean along rows, inspect residual centered component along vr1[0]
    res_dup_c = (ctx_in_sub[0, 2:] + sh_dup)
    res_dup_c = res_dup_c - res_dup_c.mean(dim=0, keepdim=True)
    base_c = ctx_in_sub[0, 2:] - ctx_in_sub[0, 2:].mean(dim=0, keepdim=True)
    # Project onto vr1[0]: old parallel P_sum = 2.0 gave sign-flipped -10.0; clamped must stay >= 0 along base_c!
    dot_sign = float(torch.sum((res_dup_c @ vr1[0].double()) * (base_c @ vr1[0].double())).item())
    if dot_sign < -1e-6:
        errs.append(f"duplicate erasure stacking sign-inverted (I - P): dot={dot_sign:.6g}")

    # Orthogonal K=2 keeps ||add_total||_2 = sqrt(2)*67.5 un-squashed
    sh_ortho = cv._stacked_erasure_block_shift64(
        ctx_f32[0, 2:].double(),
        [(mu1, 1.0, (vr1, diag1, mu1)), (mu2, 1.0, (vr2, diag2, mu2))],
    )
    ortho_rms = cv._rms_row_norm64(sh_ortho)
    add_only_rms = float(torch.linalg.vector_norm(
        (cv.NORM_MATCHED_TARGET / torch.linalg.vector_norm(mu1.double())) * mu1.double()
        + (cv.NORM_MATCHED_TARGET / torch.linalg.vector_norm(mu2.double())) * mu2.double()
    ).item())
    if abs(ortho_rms - add_only_rms) > 1e-4 or ortho_rms <= cv.NORM_MATCHED_TARGET + 10.0:
        errs.append(f"orthogonal K=2 stacked RMS={ortho_rms:.2f} != ||add_total||={add_only_rms:.2f}")

    # Opposing boosts (+mu1 and -mu1 at s=1.0) keep bounded -e erasure instead of zeroing out
    sh_opp = cv._stacked_erasure_block_shift64(
        ctx_in_sub[0, 2:],
        [(mu1, 1.0, (vr1, diag_ones, mu1)), (-mu1, 1.0, (vr1, diag_ones, -mu1))],
    )
    opp_rms = cv._rms_row_norm64(sh_opp)
    if opp_rms <= 1e-3 or opp_rms > cv.NORM_MATCHED_TARGET + 1e-4:
        errs.append(f"opposing boosts expected 0 < rms <= 67.5, got {opp_rms:.4f}")

    # 6. Union composition at COMPOSED_ERASURE_MAX_RANK == 16 preserves both rank-8 operators cleanly
    comp = cv.compose_erasure_payload(
        [
            (mu1, 1.0, "add_norm_matched_erasure", mu1, vr1, diag1),
            (mu2, 1.0, "add_norm_matched_erasure", mu2, vr2, diag2),
        ]
    )
    if comp is None or int(comp[1].shape[0]) != 16:
        errs.append(f"expected rank-16 union composition, got {None if comp is None else comp[1].shape}")

    print(
        f"  [3/3] Runtime invariants: s=0 exact, s<0 matches 1D, 0.5+0.5 split err={split_err:.2e}, "
        f"RMS(s=1.5)={got_rms:.2f} (target {exp_rms:.2f}), stack_ortho_rms={ortho_rms:.2f}, union_rank={int(comp[1].shape[0]) if comp else 0}"
    )

    # Optional real bank union check under $H3_MODEL_STORE/control_vectors/bank or ComfyUI/models
    bank_candidates = [Path(__file__).resolve().parent.parent.parent / "models" / "control_vectors" / "bank"]
    mstore = os.environ.get("H3_MODEL_STORE", "").strip()
    if mstore:
        bank_candidates.insert(0, Path(mstore) / "control_vectors" / "bank")
    bank_dir = next((c for c in bank_candidates if c.is_dir()), None)
    if bank_dir is not None:
        p_noir = next(bank_dir.glob("noir__fl2va*.safetensors"), None)
        p_chaos = next(bank_dir.glob("chaos_panic__fl2va*.safetensors"), None)
        p_deslop = next(bank_dir.glob("de_llm_slop__fl2va*.safetensors"), None)
        if p_noir is not None and p_chaos is not None:
            tn, _ = cv.load_vector(p_noir)
            tc, _ = cv.load_vector(p_chaos)
            if "erasure.vr" in tn and "erasure.vr" in tc:
                _, vr_8, d_8 = cv.compose_erasure_payload(
                    [
                        (tn["vector.post_refiner"], 1.3, "add_norm_matched_erasure", tn["erasure.mu_d"], tn["erasure.vr"], tn["erasure.diag"]),
                        (tc["vector.post_refiner"], 1.0, "add_norm_matched_erasure", tc["erasure.mu_d"], tc["erasure.vr"], tc["erasure.diag"]),
                    ],
                    max_rank=8,
                )
                _, vr_16, d_16 = cv.compose_erasure_payload(
                    [
                        (tn["vector.post_refiner"], 1.3, "add_norm_matched_erasure", tn["erasure.mu_d"], tn["erasure.vr"], tn["erasure.diag"]),
                        (tc["vector.post_refiner"], 1.0, "add_norm_matched_erasure", tc["erasure.mu_d"], tc["erasure.vr"], tc["erasure.diag"]),
                    ],
                )
                pn = (tn["erasure.vr"].double().T * tn["erasure.diag"].double().unsqueeze(0)) @ tn["erasure.vr"].double()
                p8 = (vr_8.double().T * d_8.double().unsqueeze(0)) @ vr_8.double()
                p16 = (vr_16.double().T * d_16.double().unsqueeze(0)) @ vr_16.double()
                tr_n = float(torch.trace(tn["erasure.vr"].double() @ pn @ tn["erasure.vr"].double().T).item())
                ret8 = float(torch.trace(tn["erasure.vr"].double() @ p8 @ tn["erasure.vr"].double().T).item()) / tr_n
                ret16 = float(torch.trace(tn["erasure.vr"].double() @ p16 @ tn["erasure.vr"].double().T).item()) / tr_n
                print(
                    f"        Bank noir+chaos_panic union trace retention: rank-8={ret8:.3f} vs default rank-{vr_16.shape[0]}={ret16:.3f}"
                )
                if ret16 + 1e-4 < ret8:
                    errs.append(f"rank-16 union retention {ret16:.4f} regressed below rank-8 {ret8:.4f}")
                if p_deslop is not None:
                    td, _ = cv.load_vector(p_deslop)
                    if "erasure.vr" in td:
                        comp3 = cv.compose_erasure_payload(
                            [
                                (tn["vector.post_refiner"], 1.25, "add_norm_matched_erasure", tn["erasure.mu_d"], tn["erasure.vr"], tn["erasure.diag"]),
                                (tc["vector.post_refiner"], 1.0, "add_norm_matched_erasure", tc["erasure.mu_d"], tc["erasure.vr"], tc["erasure.diag"]),
                                (td["vector.post_refiner"], 0.9, "add_norm_matched_erasure", td["erasure.mu_d"], td["erasure.vr"], td["erasure.diag"]),
                            ],
                            max_rank=24,
                        )
                        if comp3 is None or float(comp3[2].max().item()) > 1.0 + 1e-6 or float(comp3[2].min().item()) < 0.0:
                            errs.append(f"3-slider clamped union eigenvalues out of [0, 1]: {None if comp3 is None else comp3[2].tolist()}")
    return errs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu-only", action="store_true", help="Explicit CPU-only mode (default; never uses CUDA)")
    args = parser.parse_args()
    _ = args.cpu_only

    print("Verifying Clamped Pole-A Subspace Erasure (CPU float64 proofs):")
    errs: list[str] = []
    errs.extend(check_algebraic_ridge_split())
    errs.extend(check_sinkhorn_collapse_vs_row_softmax())
    errs.extend(check_runtime_invariants_and_union())
    if errs:
        for e in errs:
            print(f"  FAIL: {e}")
        return 1
    print("\nok: all Pole-A subspace erasure checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
