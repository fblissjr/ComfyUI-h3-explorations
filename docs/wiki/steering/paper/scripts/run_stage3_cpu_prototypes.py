#!/usr/bin/env python3
"""Stage 3 CPU-only prototypes for Multi-Vector + Erasure steering (`CUDA_VISIBLE_DEVICES=""`).

Evaluates strictly on CPU against existing bank `.safetensors`, cached `.pt`
per-pair banks (`data/exp_unpooled_sliders/banks/*.pt`), cached prompts
(`data/exp_unpooled_sliders/prompts/*.pt`), and `j_lens.py`:

Part A — Bus-Scoped & Contrast-Conditioned `V_r` (vs. Naive Global `PC[0..15]` Removal)
Part B — Calibrated `ROS` + Soft Conditioning-Norm Cliff & Oblique Multi-Boost Sweep
Part C — Multi-Vector Push: Path 1A (Atomic B-Within Facets) vs. Path 1B (Zero-GPU Role Routing)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "control_vectors.py").is_file():
            return p
    return Path.cwd()


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

from safetensors import safe_open  # noqa: E402
import j_lens  # noqa: E402
from app.h3_image_tools import classify_bus  # noqa: E402

NORM_MATCHED_TARGET: float = 67.5


def _find_bank_dir() -> Path:
    try:
        from workflows import h3_env  # noqa: E402

        h3_env.load_env()
    except Exception:
        pass
    model_store = os.environ.get("H3_MODEL_STORE", "").strip()
    if model_store:
        cand = Path(model_store) / "control_vectors" / "bank"
        if cand.is_dir() and any(cand.glob("*.safetensors")):
            return cand
    return REPO_ROOT.parent.parent / "models" / "control_vectors" / "bank"


def _load_safetensors_cpu(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        meta = dict(f.metadata() or {})
    return tensors, meta


def _rms_row_norm64(rows: torch.Tensor) -> float:
    return float(rows.to(torch.float64).pow(2).sum(dim=-1).mean().sqrt().item())


def _cluster_ward(x_unit: torch.Tensor, k: int) -> torch.Tensor:
    n = int(x_unit.shape[0])
    if k >= n:
        return torch.arange(n, dtype=torch.long)
    x_np = x_unit.detach().cpu().numpy().astype(np.float64)
    if HAS_SCIPY:
        lbls = fcluster(linkage(x_np, method="ward"), t=k, criterion="maxclust") - 1
        return torch.from_numpy(lbls.astype(np.int64))
    centers = [x_np[0]]
    d2 = np.sum((x_np - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        centers.append(x_np[int(np.argmax(d2))])
        d2 = np.minimum(d2, np.sum((x_np - centers[-1]) ** 2, axis=1))
    c_mat = np.stack(centers, axis=0)
    lbls = np.zeros(n, dtype=np.int64)
    for _ in range(25):
        lbls = np.argmax(x_np @ c_mat.T, axis=1)
        for ci in range(k):
            mask = lbls == ci
            if np.any(mask):
                v = x_np[mask].mean(axis=0)
                c_mat[ci] = v / max(float(np.linalg.norm(v)), 1e-12)
    return torch.from_numpy(lbls)


def _soft_align_pole_b(a_p: torch.Tensor, b_p: torch.Tensor, tau: float = 0.15) -> torch.Tensor:
    an = a_p / a_p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    bn = b_p / b_p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.softmax((an @ bn.T) / float(tau), dim=-1) @ b_p


def _fit_pole_a(
    a_tr: torch.Tensor, max_rank: int = 8, ridge_rel: float = 0.15
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_mean = a_tr.mean(dim=0, keepdim=True)
    ac = a_tr - a_mean
    _, s_vals, vh = torch.linalg.svd(ac, full_matrices=False)
    r = min(max_rank, int((s_vals > 1e-9 * s_vals[0]).sum().item()))
    vr = vh[:r]
    s_sq = s_vals[:r] ** 2
    lam = ridge_rel * float(s_vals[0].item() ** 2)
    diag = (s_sq / (s_sq + lam)).clamp(0.0, 1.0)
    return a_mean[0], vr, diag


def _reorth_erasure_against_basis(
    vr: torch.Tensor,
    diag: torch.Tensor,
    orth_basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project row(V_r) orthogonal to columns of `orth_basis [D, K]` while preserving weighted PSD metric."""
    if orth_basis.numel() == 0:
        return vr, diag
    q_b, _ = torch.linalg.qr(orth_basis, mode="reduced")
    w = torch.sqrt(diag.clamp(0.0, 1.0)).unsqueeze(-1) * vr
    w_proj = w - (w @ q_b) @ q_b.T
    _, sv, vh = torch.linalg.svd(w_proj, full_matrices=False)
    d_new = (sv ** 2).clamp(0.0, 1.0)
    keep = d_new > 1e-6
    if not bool(keep.any()):
        return vr[:0], diag[:0]
    return vh[keep], d_new[keep]


def run_part_a_bus_scoped_vr(
    by_stem: dict[str, dict[str, Any]],
    prompts: list[tuple[str, torch.Tensor]],
) -> dict[str, Any]:
    stems = sorted(by_stem.keys())
    buses = {s: by_stem[s]["bus"] for s in stems}
    bus_counts = {b: sum(1 for s in stems if buses[s] == b) for b in sorted(set(buses.values()))}

    p_all = torch.zeros((5376, 5376), dtype=torch.float64)
    p_look = torch.zeros((5376, 5376), dtype=torch.float64)
    for s in stems:
        vr, d = by_stem[s]["vr"], by_stem[s]["diag"]
        mat = vr.T @ (d.unsqueeze(-1) * vr)
        p_all += mat
        if buses[s] == "look":
            p_look += mat
    _, ev_all = torch.linalg.eigh(p_all)
    u16_all = ev_all.flip(1)[:, :16]
    _, ev_look = torch.linalg.eigh(p_look)
    u8_look = ev_look.flip(1)[:, :8]
    u16_look = ev_look.flip(1)[:, :16]

    mu_units = torch.stack(
        [by_stem[s]["mu_d"] / by_stem[s]["mu_d"].norm().clamp_min(1e-12) for s in stems],
        dim=0,
    )

    test_stems = [
        "noir",
        "paper_origami",
        "claymation_stopmotion",
        "chaos_panic",
        "handheld_documentary_zoom",
        "reverb_cathedral",
        "whisper_asmr",
    ]
    variants = {
        "shipped": lambda s, vr, d: (vr, d),
        "naive_drop_global_pc16": lambda s, vr, d: _reorth_erasure_against_basis(vr, d, u16_all),
        "bus_scoped_audio_motion_drop_look_pc8": lambda s, vr, d: (
            (vr, d) if buses[s] == "look" else _reorth_erasure_against_basis(vr, d, u8_look)
        ),
        "bus_scoped_audio_motion_drop_look_pc16": lambda s, vr, d: (
            (vr, d) if buses[s] == "look" else _reorth_erasure_against_basis(vr, d, u16_look)
        ),
        "bus_scoped_am_pc16_plus_look_ros_crossbus": None,
    }

    var_ops: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    bus_mu_units: dict[str, list[torch.Tensor]] = {}
    for b_name in ("look", "motion", "audio"):
        bus_mu_units[b_name] = [mu_units[i] for i, s in enumerate(stems) if buses[s] == b_name]

    for vname, fn in variants.items():
        cur_map = {}
        for s in stems:
            vr, d = by_stem[s]["vr"], by_stem[s]["diag"]
            if vname == "bus_scoped_am_pc16_plus_look_ros_crossbus":
                vr_b, d_b = (vr, d) if buses[s] == "look" else _reorth_erasure_against_basis(vr, d, u16_look)
                if buses[s] in ("audio", "motion") and bus_mu_units["look"]:
                    other_mus = torch.stack(bus_mu_units["look"], dim=1)
                    u_om, _s_om, _ = torch.linalg.svd(other_mus, full_matrices=False)
                    vr_b, d_b = _reorth_erasure_against_basis(vr_b, d_b, u_om[:, :8])
                elif buses[s] == "look" and bus_mu_units["audio"]:
                    aud_mus = torch.stack(bus_mu_units["audio"], dim=1)
                    vr_b, d_b = _reorth_erasure_against_basis(vr_b, d_b, aud_mus)
                cur_map[s] = (vr_b, d_b)
            else:
                assert fn is not None
                cur_map[s] = fn(s, vr, d)
        var_ops[vname] = cur_map

    results = {"bus_counts": bus_counts, "variants": {}}
    for vname, cur_map in var_ops.items():
        retention: dict[str, float] = {}
        for ts in test_stems:
            if ts not in by_stem:
                continue
            vr_orig, d_orig = by_stem[ts]["vr"], by_stem[ts]["diag"]
            vr_new, d_new = cur_map[ts]
            ratios = []
            for _pn, h in prompts:
                hc = h - h.mean(dim=0, keepdim=True)
                e_o = ((hc @ vr_orig.T) * d_orig.unsqueeze(0)) @ vr_orig
                o_norm = float(e_o.pow(2).sum().sqrt().item())
                if vr_new.shape[0] == 0:
                    ratios.append(0.0)
                else:
                    e_n = ((hc @ vr_new.T) * d_new.unsqueeze(0)) @ vr_new
                    ratios.append(float(e_n.pow(2).sum().sqrt().item()) / max(o_norm, 1e-12))
            retention[ts] = round(float(np.mean(ratios)), 4)

        cross_bus_g, look_aud_g, look_mot_g = [], [], []
        all_cross_er_d, look_vs_aud_er_d, aud_vs_look_er_d = [], [], []
        for i in range(len(stems)):
            si, bi = stems[i], buses[stems[i]]
            vri, _di = cur_map[si]
            mui = mu_units[i]
            for j in range(len(stems)):
                if i == j:
                    continue
                sj, bj = stems[j], buses[stems[j]]
                vrj, dj = cur_map[sj]
                if vrj.shape[0] > 0:
                    er_ij = float((((mui @ vrj.T) ** 2) * dj).sum().item())
                else:
                    er_ij = 0.0
                all_cross_er_d.append(er_ij)
                if bi == "audio" and bj == "look":
                    aud_vs_look_er_d.append(er_ij)
                if bi == "look" and bj == "audio":
                    look_vs_aud_er_d.append(er_ij)

                if j > i:
                    if vri.shape[0] == 0 or vrj.shape[0] == 0:
                        g_ij = 0.0
                    else:
                        sv = torch.linalg.svdvals(vri @ vrj.T)
                        g_ij = float((sv ** 2).sum().item() / max(1, min(int(vri.shape[0]), int(vrj.shape[0]))))
                    if bi != bj:
                        cross_bus_g.append(g_ij)
                    if {bi, bj} == {"look", "audio"}:
                        look_aud_g.append(g_ij)
                    if {bi, bj} == {"look", "motion"}:
                        look_mot_g.append(g_ij)

        results["variants"][vname] = {
            "prompt_pole_a_retention": retention,
            "grassmann_cross_bus_mean": round(float(np.mean(cross_bus_g)), 4),
            "grassmann_look_audio_mean": round(float(np.mean(look_aud_g)), 4),
            "grassmann_look_motion_mean": round(float(np.mean(look_mot_g)), 4),
            "cross_er_d_mean": round(float(np.mean(all_cross_er_d)), 4),
            "cross_er_d_max": round(float(np.max(all_cross_er_d)), 4),
            "look_erases_audio_d_mean": round(float(np.mean(aud_vs_look_er_d)), 4),
            "look_erases_audio_d_max": round(float(np.max(aud_vs_look_er_d)), 4),
            "audio_erases_look_d_mean": round(float(np.mean(look_vs_aud_er_d)), 4),
            "audio_erases_look_d_max": round(float(np.max(look_vs_aud_er_d)), 4),
        }
    return results


def _apply_norm_cap(shift: torch.Tensor, norm_cap: float | None, mode: str) -> torch.Tensor:
    if norm_cap is None:
        return shift
    rms = float(_rms_row_norm64(shift))
    if rms <= 1e-12:
        return shift
    if mode == "hard":
        if rms <= norm_cap:
            return shift
        return shift * (norm_cap / rms)
    if mode == "soft_tanh":
        scale = (norm_cap * math.tanh(rms / norm_cap)) / rms
        return shift * scale
    raise ValueError(mode)


def run_part_b_norm_cap_and_oblique(
    bank_dir: Path,
    by_stem_fl2va: dict[str, dict[str, Any]],
    prompts: list[tuple[str, torch.Tensor]],
) -> dict[str, Any]:
    h3i_ops: dict[str, dict[str, torch.Tensor]] = {}
    for p in sorted(bank_dir.glob("*__h3_image__*.safetensors")):
        stem = p.name.split("__h3_image__", 1)[0]
        t, _ = _load_safetensors_cpu(p)
        if all(k in t for k in ("erasure.vr", "erasure.diag", "erasure.mu_d")):
            h3i_ops[stem] = {
                "vr": t["erasure.vr"].to(torch.float64),
                "diag": t["erasure.diag"].to(torch.float64).clamp(0.0, 1.0),
                "mu_d": t["erasure.mu_d"].to(torch.float64),
            }

    def _sim_stack(
        rows64: torch.Tensor,
        entries: list[tuple[str, dict[str, torch.Tensor], float]],
        use_ros: bool,
        oblique_gamma: float | None,
        cap_rms: float | None,
        cap_mode: str,
    ) -> dict[str, Any]:
        centered = rows64 - rows64.mean(dim=0, keepdim=True)
        target = NORM_MATCHED_TARGET
        mu_units = [op["mu_d"] / op["mu_d"].norm().clamp_min(1e-12) for _s, op, _k in entries]
        a_terms = [(float(s_k) * target) * mu_u for (_s, _op, s_k), mu_u in zip(entries, mu_units)]

        if oblique_gamma is not None:
            m_mat = torch.stack(mu_units, dim=0) @ torch.stack(mu_units, dim=0).T
            s_vec = torch.tensor([float(s_k) for _s, _op, s_k in entries], dtype=torch.float64)
            c_vec = torch.linalg.solve(
                m_mat + float(oblique_gamma) * torch.eye(len(entries), dtype=torch.float64),
                (1.0 + float(oblique_gamma)) * s_vec,
            )
            a_terms = [(float(c_vec[i].item()) * target) * mu_units[i] for i in range(len(entries))]

        add_total = sum(a_terms)
        target_rms = float(add_total.norm().item())

        w_blocks = []
        e_budget_sum = 0.0
        for idx, (_st, op, s_k) in enumerate(entries):
            vr_i, d_i = op["vr"], op["diag"]
            if use_ros:
                others = torch.stack([mu_units[m] for m in range(len(entries)) if m != idx], dim=1)
                vr_i, d_i = _reorth_erasure_against_basis(vr_i, d_i, others)
                if vr_i.shape[0] == 0:
                    continue
            w_eff = max(0.0, min(float(s_k), 1.0))
            e_budget_sum += w_eff * target
            w_blocks.append(torch.sqrt((w_eff * d_i).clamp(0.0, 1.0)).unsqueeze(-1) * vr_i)

        if not w_blocks:
            shift = add_total.unsqueeze(0).expand_as(rows64)
        else:
            _, sv, vh = torch.linalg.svd(torch.cat(w_blocks, dim=0), full_matrices=False)
            du = (sv ** 2).clamp(0.0, 1.0)
            keep = du > 1e-9
            vr_u, du_k = vh[keep], du[keep]
            e_own = ((centered @ vr_u.T) * du_k.unsqueeze(0)) @ vr_u
            e_rms = float(_rms_row_norm64(e_own))
            e_b = min(e_budget_sum, max(target, target_rms))
            e_cap = min(1.0, e_b / e_rms) if e_rms > 1e-9 else 1.0
            raw = add_total.unsqueeze(0) - e_own * e_cap
            raw_rms = float(_rms_row_norm64(raw))
            shift = raw * (target_rms / max(raw_rms, 1e-12))

        pre_cap_rms = float(_rms_row_norm64(shift))
        shift = _apply_norm_cap(shift, cap_rms, cap_mode)
        post_cap_rms = float(_rms_row_norm64(shift))
        post_cap_peak = float(shift.norm(dim=-1).max().item())
        steered = rows64 + shift
        st_c = steered - steered.mean(dim=0, keepdim=True)
        vr0, d0 = entries[0][1]["vr"], entries[0][1]["diag"]
        base_pa = float(torch.linalg.matrix_norm((centered @ vr0.T) * torch.sqrt(d0).unsqueeze(0)).item())
        st_pa = float(torch.linalg.matrix_norm((st_c @ vr0.T) * torch.sqrt(d0).unsqueeze(0)).item())
        dm = shift.mean(dim=0)
        return {
            "pre_cap_rms": round(pre_cap_rms, 2),
            "post_cap_rms": round(post_cap_rms, 2),
            "post_cap_peak": round(post_cap_peak, 2),
            "slider0_pa_resid": round(st_pa / max(base_pa, 1e-12), 4),
            "mean_shift_proj": {entries[i][0]: round(float(torch.dot(dm, mu_units[i]).item()), 2) for i in range(len(entries))},
        }

    cases = [
        ("h3i_claymation_noir_s120_115", h3i_ops, [("claymation_stopmotion", 1.20), ("noir", 1.15)]),
        ("h3i_claymation_amber_s120_115", h3i_ops, [("claymation_stopmotion", 1.20), ("look_amber_candlelight", 1.15)]),
        ("h3i_origami_slop_s120_115", h3i_ops, [("paper_origami", 1.20), ("de_llm_slop", 1.15)]),
        ("fl2va_showcase_a_stack3", by_stem_fl2va, [("noir", 1.25), ("chaos_panic", 1.00), ("de_llm_slop", 0.90)]),
        ("fl2va_anticorr_biolum_slop", by_stem_fl2va, [("bioluminescent_glow", 1.15), ("de_llm_slop", 1.00)]),
        ("fl2va_corr_16mm_amber", by_stem_fl2va, [("film_16mm_1974", 1.10), ("look_amber_candlelight", 1.10)]),
    ]
    configs = [
        ("shipped_uncapped", False, None, None, "hard"),
        ("ros_uncapped", True, None, None, "hard"),
        ("ros_soft115", True, None, 115.0, "soft_tanh"),
        ("ros_soft105", True, None, 105.0, "soft_tanh"),
        ("ros_hard110", True, None, 110.0, "hard"),
        ("ros_oblique025_soft110", True, 0.25, 110.0, "soft_tanh"),
    ]
    out: dict[str, Any] = {}
    for cname, ops_src, spec in cases:
        entries = [(s, ops_src[s], v) for s, v in spec if s in ops_src]
        if len(entries) != len(spec):
            continue
        c_res = {}
        for cfg_name, use_ros, obl, cap, cmode in configs:
            per_p = [_sim_stack(rows64, entries, use_ros, obl, cap, cmode) for _pn, rows64 in prompts]
            c_res[cfg_name] = {
                "mean_pre_cap_rms": round(float(np.mean([x["pre_cap_rms"] for x in per_p])), 2),
                "mean_post_cap_rms": round(float(np.mean([x["post_cap_rms"] for x in per_p])), 2),
                "max_peak_row_norm": round(float(np.max([x["post_cap_peak"] for x in per_p])), 2),
                "mean_slider0_pa_resid": round(float(np.mean([x["slider0_pa_resid"] for x in per_p])), 4),
                "mean_proj": {
                    entries[i][0]: round(float(np.mean([x["mean_shift_proj"][entries[i][0]] for x in per_p])), 2)
                    for i in range(len(entries))
                },
            }
        out[cname] = c_res
    return out


def _build_zero_gpu_role_bank(jl_bundle: dict[str, Any]) -> tuple[list[str], torch.Tensor, dict[str, Any]]:
    """Synthesize 5 orthogonal universal H3 semantic-role probe directions in 5376-D via J-Lens."""
    role_lex = {
        "camera": ["camera", "lens", "shot", "angle", "framing", "handheld", "focus", "close-up", "wide"],
        "lighting": ["lighting", "light", "shadow", "sunlight", "tungsten", "glow", "warm", "color", "exposure"],
        "subject": ["woman", "man", "person", "face", "eyes", "hands", "shirt", "jacket", "expression"],
        "environment": ["room", "kitchen", "wall", "floor", "table", "street", "building", "background", "furniture"],
        "motion": ["moves", "walks", "turns", "rushes", "leans", "gestures", "sudden", "fast", "slowly"],
    }
    names = list(role_lex.keys())
    raw_vecs = []
    meta = {}
    for rn, words in role_lex.items():
        syn = j_lens.synthesize_from_words(
            words,
            neg_words="neutral",
            family="fl2va",
            target_norm=1.0,
            bundle=jl_bundle,
        )
        v5376 = syn["vector.post_refiner"].to(torch.float64).flatten()
        v5376 = v5376 / v5376.norm().clamp_min(1e-12)
        raw_vecs.append(v5376)
        meta[rn] = {"matched_words": syn.get("pos_words_matched", words)}
    r_mat = torch.stack(raw_vecs, dim=0)
    gram = r_mat @ r_mat.T
    evals, evecs = torch.linalg.eigh(gram)
    inv_sqrt = evecs @ torch.diag(evals.clamp_min(1e-8).rsqrt()) @ evecs.T
    r_orth = inv_sqrt @ r_mat
    return names, r_orth, {"lexicon": meta, "raw_cos_matrix": torch.round(gram, decimals=3).tolist()}


def _fit_dual_ridge(x_tr: torch.Tensor, y_tr: torch.Tensor, ridge_rel: float = 0.5) -> torch.Tensor:
    gram = x_tr @ x_tr.T
    tr_scale = max(float(torch.trace(gram).item()) / max(1, int(x_tr.shape[0])), 1e-6)
    lam = ridge_rel * tr_scale
    yc = y_tr - y_tr.mean(dim=0, keepdim=True)
    alpha = torch.linalg.solve(gram + lam * torch.eye(x_tr.shape[0], dtype=torch.float64), yc)
    return x_tr.T @ alpha


def run_part_c_multivector_push(
    bank_dir: Path,
    prompts_full: list[dict[str, Any]],
) -> dict[str, Any]:
    jl_bundle = j_lens.load_j_lens_bundle()
    role_names, role_keys, role_meta = _build_zero_gpu_role_bank(jl_bundle)

    cm = prompts_full[0]
    cm_mask = cm["scene_mask"]
    cm_rows = cm["ref"][cm_mask].to(torch.float64)
    cm_c = cm_rows - cm_rows.mean(dim=0, keepdim=True)
    cm_u = cm_c / cm_c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cm_pieces = [pc for pc, keep in zip(cm["pieces"], cm_mask.tolist()) if keep]
    scores = cm_u @ role_keys.T
    top_role_tokens = {}
    for r_i, r_name in enumerate(role_names):
        top_vals, top_idxs = torch.topk(scores[:, r_i], k=min(10, len(cm_pieces)))
        top_role_tokens[r_name] = [
            {"piece": cm_pieces[int(ix)], "cos": round(float(tv), 4)}
            for tv, ix in zip(top_vals.tolist(), top_idxs.tolist())
        ]

    pt_files = sorted(bank_dir.glob("*.pt"))
    path1a_summary = {}
    path1b_lopo = {}

    for pt_path in pt_files:
        stem = pt_path.stem
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
        a_list = [x.to(torch.float64) for x in obj["a_spans_per_pair"]]
        b_list = [x.to(torch.float64) for x in obj["b_spans_per_pair"]]
        n_pairs = len(a_list)
        bt_list = [_soft_align_pole_b(a_p, b_p, tau=0.15) for a_p, b_p in zip(a_list, b_list)]
        ac_pair_list = [a_p - a_p.mean(dim=0, keepdim=True) for a_p in a_list]

        a_full = torch.cat(a_list, dim=0)
        bt_full = torch.cat(bt_list, dim=0)
        a_mean_f, vr_f, diag_f = _fit_pole_a(a_full, max_rank=8, ridge_rel=0.15)
        a_er_f = a_full - ((((a_full - a_mean_f.unsqueeze(0)) @ vr_f.T) * diag_f.unsqueeze(0)) @ vr_f)
        trans_f = bt_full - a_er_f
        mu_d_f = trans_f.mean(dim=0)
        trans_res_f = trans_f - mu_d_f.unsqueeze(0)
        trans_u_f = trans_f / trans_f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        lbl4 = _cluster_ward(trans_u_f, 4)

        facet_cards = []
        tot_sq = float(trans_f.pow(2).sum().item())
        oracle_err_sq = 0.0
        for ci in range(4):
            m = lbl4 == ci
            if not bool(m.any()):
                continue
            dk = trans_f[m].mean(dim=0)
            res_k = dk - mu_d_f
            res_k_orth = res_k - (res_k @ vr_f.T) @ vr_f
            facet_vec = mu_d_f + res_k_orth
            oracle_err_sq += float((trans_f[m] - facet_vec.unsqueeze(0)).pow(2).sum().item())
            ro_res = j_lens.decode_vector(
                res_k_orth.to(torch.float32), space="post_refiner", family="fl2va", top_k=6, bundle=jl_bundle
            )
            ro_full = j_lens.decode_vector(
                facet_vec.to(torch.float32), space="post_refiner", family="fl2va", top_k=6, bundle=jl_bundle
            )
            facet_cards.append({
                "cluster": ci,
                "n_tokens": int(m.sum().item()),
                "res_norm_over_mud": round(float(res_k_orth.norm().item() / max(mu_d_f.norm().item(), 1e-12)), 3),
                "full_top_words": [x["word"] for x in ro_full.get("positive_words", [])[:6]],
                "residual_top_words": [x["word"] for x in ro_res.get("positive_words", [])[:6]],
            })
        t1_err_sq = float(trans_res_f.pow(2).sum().item())
        res_vecs = []
        for ci in range(4):
            m = lbl4 == ci
            if bool(m.any()):
                rk = trans_f[m].mean(dim=0) - mu_d_f
                rk = rk - (rk @ vr_f.T) @ vr_f
                res_vecs.append(rk / rk.norm().clamp_min(1e-12))
        res_gram = (torch.stack(res_vecs, dim=0) @ torch.stack(res_vecs, dim=0).T) if res_vecs else torch.empty(0)
        tri = torch.triu_indices(len(res_vecs), len(res_vecs), offset=1) if len(res_vecs) >= 2 else None
        path1a_summary[stem] = {
            "t1_explained_r2_on_trans": round(1.0 - t1_err_sq / max(tot_sq, 1e-12), 4),
            "k4_oracle_r2_on_trans": round(1.0 - oracle_err_sq / max(tot_sq, 1e-12), 4),
            "oracle_mse_reduction_pct_vs_t1": round(100.0 * (oracle_err_sq - t1_err_sq) / max(t1_err_sq, 1e-12), 2),
            "facet_residual_min_cos": round(float(res_gram[tri[0], tri[1]].min().item()), 3) if tri is not None else 0.0,
            "facet_residual_max_cos": round(float(res_gram[tri[0], tri[1]].max().item()), 3) if tri is not None else 0.0,
            "facets": facet_cards,
        }

        t1_mses, f_ridge_mses, jrole_ridge_mses, hybrid_ridge_mses, jrole_zeroshot_mses = [], [], [], [], []
        for p_out in range(n_pairs):
            a_tr = torch.cat([a_list[i] for i in range(n_pairs) if i != p_out], dim=0)
            ac_tr = torch.cat([ac_pair_list[i] for i in range(n_pairs) if i != p_out], dim=0)
            bt_tr = torch.cat([bt_list[i] for i in range(n_pairs) if i != p_out], dim=0)
            a_te, ac_te, bt_te = a_list[p_out], ac_pair_list[p_out], bt_list[p_out]

            a_mean_tr, vr, diag = _fit_pole_a(a_tr, max_rank=8, ridge_rel=0.15)
            a_er_tr = a_tr - ((((a_tr - a_mean_tr.unsqueeze(0)) @ vr.T) * diag.unsqueeze(0)) @ vr)
            trans_tr = bt_tr - a_er_tr
            mu_d = trans_tr.mean(dim=0)
            a_er_te = a_te - (((ac_te @ vr.T) * diag.unsqueeze(0)) @ vr)
            pred_t1 = a_er_te + mu_d.unsqueeze(0)
            t1_mses.append(float(((bt_te - pred_t1) ** 2).mean().item()))

            trans_u_tr = trans_tr / trans_tr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            lbl_tr = _cluster_ward(trans_u_tr, 4)
            uniq = sorted(set(lbl_tr.tolist()))
            deltas = []
            onehot = torch.zeros((a_tr.shape[0], len(uniq)), dtype=torch.float64)
            for new_i, old_i in enumerate(uniq):
                m = lbl_tr == old_i
                onehot[m, new_i] = 1.0
                rk = trans_tr[m].mean(dim=0) - mu_d
                rk = rk - (rk @ vr.T) @ vr
                deltas.append(mu_d + rk)
            pd_fac = torch.stack(deltas, dim=0)

            ac_tr_u = ac_tr / ac_tr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            ac_te_u = ac_te / ac_te.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            rfeat_tr = ac_tr_u @ role_keys.T
            rfeat_te = ac_te_u @ role_keys.T

            w_fr = _fit_dual_ridge(ac_tr_u, onehot, ridge_rel=1.0)
            pr_fr = a_er_te + torch.softmax((ac_te_u @ w_fr) / 0.5, dim=-1) @ pd_fac
            f_ridge_mses.append(float(((bt_te - pr_fr) ** 2).mean().item()))

            r_aug_tr = torch.cat([rfeat_tr, torch.ones((rfeat_tr.shape[0], 1), dtype=torch.float64)], dim=1)
            r_aug_te = torch.cat([rfeat_te, torch.ones((rfeat_te.shape[0], 1), dtype=torch.float64)], dim=1)
            res_tr = trans_tr - mu_d.unsqueeze(0)
            lam_r = 25.0
            w_role_direct = torch.linalg.solve(
                r_aug_tr.T @ r_aug_tr + lam_r * torch.eye(r_aug_tr.shape[1], dtype=torch.float64),
                r_aug_tr.T @ res_tr,
            )
            delta_role = r_aug_te @ w_role_direct
            delta_role = delta_role - (delta_role @ vr.T) @ vr
            pr_role_dir = pred_t1 + delta_role
            jrole_ridge_mses.append(float(((bt_te - pr_role_dir) ** 2).mean().item()))

            w_z_tr = torch.softmax(rfeat_tr / 0.10, dim=-1)
            role_deltas = torch.linalg.solve(
                w_z_tr.T @ w_z_tr + 10.0 * torch.eye(len(role_names), dtype=torch.float64),
                w_z_tr.T @ res_tr,
            )
            role_deltas = role_deltas - (role_deltas @ vr.T) @ vr
            delta_zs = torch.softmax(rfeat_te / 0.10, dim=-1) @ role_deltas
            jrole_zeroshot_mses.append(float(((bt_te - (pred_t1 + delta_zs)) ** 2).mean().item()))

            hyb_tr = torch.cat([rfeat_tr * 3.0, ac_tr_u], dim=1)
            hyb_te = torch.cat([rfeat_te * 3.0, ac_te_u], dim=1)
            w_hyb = _fit_dual_ridge(hyb_tr, onehot, ridge_rel=1.0)
            pr_hyb = a_er_te + torch.softmax((hyb_te @ w_hyb) / 0.5, dim=-1) @ pd_fac
            hybrid_ridge_mses.append(float(((bt_te - pr_hyb) ** 2).mean().item()))

        t1_m = float(np.mean(t1_mses))
        path1b_lopo[stem] = {
            "t1_mse": round(t1_m, 4),
            "diff_span_facet_ridge_pct": round(100.0 * (float(np.mean(f_ridge_mses)) - t1_m) / t1_m, 2),
            "path1b_role5_direct_pct": round(100.0 * (float(np.mean(jrole_ridge_mses)) - t1_m) / t1_m, 2),
            "path1b_role5_softassign_pct": round(100.0 * (float(np.mean(jrole_zeroshot_mses)) - t1_m) / t1_m, 2),
            "path1b_hybrid_facet_pct": round(100.0 * (float(np.mean(hybrid_ridge_mses)) - t1_m) / t1_m, 2),
        }

    return {
        "zero_gpu_role_bank": {"roles": role_names, **role_meta},
        "covered_market_top_role_tokens": top_role_tokens,
        "path1a_atomic_facets": path1a_summary,
        "path1b_lopo_cv": path1b_lopo,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3 CPU-only prototypes for Multi-Vector + Erasure steering.")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output JSON path (defaults to phase5_cpu_audits/stage3_cpu_prototypes.json when writing).",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Overwrite default stage3_cpu_prototypes.json if it already exists.",
    )
    args = ap.parse_args(argv)

    bank_dir = _find_bank_dir()
    files = sorted(bank_dir.glob("*__fl2va__*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No *__fl2va__*.safetensors found under {bank_dir}")

    by_stem: dict[str, dict[str, Any]] = {}
    for p in files:
        stem = p.name.split("__fl2va__", 1)[0]
        if stem in by_stem:
            continue
        t, m = _load_safetensors_cpu(p)
        if all(k in t for k in ("erasure.vr", "erasure.diag", "erasure.mu_d")):
            by_stem[stem] = {
                "vr": t["erasure.vr"].to(torch.float64),
                "diag": t["erasure.diag"].to(torch.float64).clamp(0.0, 1.0),
                "mu_d": t["erasure.mu_d"].to(torch.float64),
                "bus": classify_bus(text=stem, explicit_bus=str(m.get("bus", ""))),
            }

    prompt_files = [
        REPO_ROOT / "data/exp_unpooled_sliders/prompts/covered_market.pt",
        REPO_ROOT / "data/exp_unpooled_sliders/prompts/indoor_evening.pt",
        REPO_ROOT / "data/exp_unpooled_sliders/prompts/indoor_neutral.pt",
    ]
    missing = [str(pf) for pf in prompt_files if not pf.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cached post-refiner prompt tensors are required for Stage 3 CPU prototypes: "
            + ", ".join(missing)
        )

    prompts_full = [torch.load(pf, map_location="cpu", weights_only=False) for pf in prompt_files]
    prompts = [(d["name"], d["ref"][d["scene_mask"]].to(torch.float64)) for d in prompts_full]

    print(f"Running Stage 3 CPU prototypes ({len(by_stem)} __fl2va__ stems, {len(prompts)} cached prompts)...")
    part_a = run_part_a_bus_scoped_vr(by_stem, prompts)
    part_b = run_part_b_norm_cap_and_oblique(bank_dir, by_stem, prompts)
    part_c = run_part_c_multivector_push(REPO_ROOT / "data/exp_unpooled_sliders/banks", prompts_full)

    out = {"part_a_bus_scoped_vr": part_a, "part_b_norm_cap_and_oblique": part_b, "part_c_multivector_push": part_c}
    out_dir = REPO_ROOT / "data/exp_unpooled_sliders/phase5_cpu_audits"
    default_out = out_dir / "stage3_cpu_prototypes.json"
    out_path = args.out.resolve() if args.out is not None else default_out
    if args.out is not None or args.write or not default_out.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(f"[dry-run] Computed Stage 3 CPU prototypes (n={len(by_stem)}); pass --write or --out to overwrite {default_out}")
    print("\n=== Part A Summary ===")
    for vn, vd in part_a["variants"].items():
        print(
            f"  {vn:42s} | look_er_aud={vd['look_erases_audio_d_mean']:.4f} | "
            f"aud_er_look={vd['audio_erases_look_d_mean']:.4f} | "
            f"G_la={vd['grassmann_look_audio_mean']:.4f}"
        )
    print("\n=== Part B Norm Cap Summary ===")
    for cn, cd in part_b.items():
        print(f"  {cn}:", {k: (v["mean_post_cap_rms"], v["max_peak_row_norm"], v["mean_proj"]) for k, v in cd.items()})
    print("\n=== Part C Path 1A Facets + Path 1B LOPO ===")
    for st in sorted(part_c["path1b_lopo_cv"].keys()):
        p1a = part_c["path1a_atomic_facets"][st]
        p1b = part_c["path1b_lopo_cv"][st]
        print(
            f"  {st:22s} | R2: T1={p1a['t1_explained_r2_on_trans']:.3f} -> "
            f"Oracle4={p1a['k4_oracle_r2_on_trans']:.3f} ({p1a['oracle_mse_reduction_pct_vs_t1']:+.1f}%) | "
            f"LOPO: fac={p1b['diff_span_facet_ridge_pct']:+.2f}%, "
            f"role5_dir={p1b['path1b_role5_direct_pct']:+.2f}%, "
            f"role5_zs={p1b['path1b_role5_softassign_pct']:+.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
