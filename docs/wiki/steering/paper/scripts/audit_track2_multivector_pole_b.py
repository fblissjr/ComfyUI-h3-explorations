#!/usr/bin/env python3
"""CPU-only Track 2 Leave-One-Pair-Out Empirical Audit: True Per-Token Multi-Vector Pole-B.

Tests Track 1 (`V_r` r=8 + single `mu_d`) against three zero-external-corpus Track 2 families:
  - Track 2A-raw:      Ward K clusters on raw unit Pole-A keys a_n -> delta_k
  - Track 2A-centered: Ward K clusters on prompt-centered Pole-A keys (a_n - a_bar_p) -> delta_k
  - Track 2A-facet:    Ward K clusters on Pole-B *transitions* (trans_n) to discover distinct
                       Pole-B facets first, paired with either mean centered-PoleA centroids
                       (`facet_cos`) or a L2-regularized linear logit router (`facet_ridge`)
  - Track 2B:          Lemma-1 rank-r ridge cross-covariance transport W_B (r in {4, 8})

Also measures residual Pole-B variance ratio `R_B_rms / trans_rms`, facet angular diversity
(`min_cos(delta_k)`), and OOD routing entropy + shift diversity on cached prompts (`prompts/*.pt`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

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


def _soft_align_pole_b(a_p: torch.Tensor, b_p: torch.Tensor, tau: float = 0.15) -> torch.Tensor:
    an = a_p / a_p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    bn = b_p / b_p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w = torch.softmax((an @ bn.T) / float(tau), dim=-1)
    return w @ b_p


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


def _cluster_ward(x_unit: torch.Tensor, k: int) -> torch.Tensor:
    n = int(x_unit.shape[0])
    if k >= n:
        return torch.arange(n, dtype=torch.long)
    x_np = x_unit.detach().cpu().numpy().astype(np.float64)
    if HAS_SCIPY:
        z = linkage(x_np, method="ward")
        lbls = fcluster(z, t=k, criterion="maxclust") - 1
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


def _route_cos_weights(
    q_unit: torch.Tensor,
    proto_a_unit: torch.Tensor,
    tau_mv: float,
    top_k: int | None = None,
) -> torch.Tensor:
    logits = (q_unit @ proto_a_unit.T) / float(tau_mv)
    if top_k is not None and top_k < proto_a_unit.shape[0]:
        vals, idxs = torch.topk(logits, k=top_k, dim=-1)
        w = torch.zeros_like(logits)
        w.scatter_(-1, idxs, torch.softmax(vals, dim=-1))
        return w
    return torch.softmax(logits, dim=-1)


def _build_proto_from_labels(
    key_unit: torch.Tensor,
    trans_tr: torch.Tensor,
    mu_d: torch.Tensor,
    vr: torch.Tensor,
    lbls: torch.Tensor,
    k_val: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    proto_a_list, delta_list = [], []
    for ci in range(k_val):
        m = lbls == ci
        if bool(m.any()):
            pk = key_unit[m].mean(dim=0)
            pk = pk / pk.norm().clamp_min(1e-12)
            dk = trans_tr[m].mean(dim=0)
            d_res = dk - mu_d
            d_res_orth = d_res - (d_res @ vr.T) @ vr
            proto_a_list.append(pk)
            delta_list.append(mu_d + d_res_orth)
    return torch.stack(proto_a_list, dim=0), torch.stack(delta_list, dim=0)


def _fit_ridge_router(x_tr: torch.Tensor, y_onehot: torch.Tensor, ridge_rel: float = 0.5) -> torch.Tensor:
    """Closed-form dual ridge solution W in R^{D x K} mapping x -> cluster logits."""
    gram = x_tr @ x_tr.T
    tr_scale = float(torch.trace(gram).item()) / max(1, int(x_tr.shape[0]))
    lam = ridge_rel * max(tr_scale, 1e-6)
    alpha = torch.linalg.solve(
        gram + lam * torch.eye(x_tr.shape[0], dtype=torch.float64),
        y_onehot - y_onehot.mean(dim=0, keepdim=True),
    )
    return x_tr.T @ alpha


def _pairwise_cos_std(rows: torch.Tensor) -> float:
    if rows.shape[0] < 2:
        return 0.0
    nrm = rows.norm(dim=-1, keepdim=True)
    ok = nrm.squeeze(-1) > 1e-9
    if int(ok.sum().item()) < 2:
        return 0.0
    u = rows[ok] / nrm[ok]
    c = u @ u.T
    tri = torch.triu_indices(u.shape[0], u.shape[0], offset=1)
    return float(c[tri[0], tri[1]].std().item())


def _summarize_cv(
    res_dict: dict[str, dict[str, list[float]]], t1_mse: float, t1_cos: float
) -> tuple[dict, tuple[str, dict]]:
    out = {}
    for k, v in res_dict.items():
        m = float(np.mean(v["mse"]))
        c = float(np.mean(v["cos"]))
        out[k] = {
            "mse": round(m, 4),
            "mse_delta_pct_vs_t1": round(100.0 * (m - t1_mse) / max(t1_mse, 1e-12), 2),
            "cos": round(c, 4),
            "cos_delta_vs_t1": round(c - t1_cos, 4),
            "shift_cos_std": round(float(np.mean(v["shift_std"])), 4),
        }
    best = min(out.items(), key=lambda kv: kv[1]["mse"])
    return out, best


def main() -> int:
    bank_dir = REPO_ROOT / "data" / "exp_unpooled_sliders" / "banks"
    census_path = REPO_ROOT / "data" / "exp_unpooled_sliders" / "paper_2026_11_21" / "census_38_bank.json"
    census_map: dict[str, dict] = {}
    if census_path.is_file():
        census = json.loads(census_path.read_text(encoding="utf-8"))
        census_map = {r["concept"]: r for r in census.get("records", [])}

    prompt_files = [
        REPO_ROOT / "data/exp_unpooled_sliders/prompts/covered_market.pt",
        REPO_ROOT / "data/exp_unpooled_sliders/prompts/indoor_evening.pt",
        REPO_ROOT / "data/exp_unpooled_sliders/prompts/indoor_neutral.pt",
    ]
    missing = [str(pf) for pf in prompt_files if not pf.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cached post-refiner prompt tensors are required for Track 2 audit: " + ", ".join(missing)
        )

    ood_prompts = []
    for pf in prompt_files:
        d = torch.load(pf, map_location="cpu", weights_only=False)
        raw_p = d["ref"][d["scene_mask"]].to(torch.float64)
        cent_p = raw_p - raw_p.mean(dim=0, keepdim=True)
        ood_prompts.append((d["name"], raw_p, cent_p))

    pt_files = sorted(bank_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No cached per-pair .pt banks found under {bank_dir}")

    all_bank_results = {}

    for pt_path in pt_files:
        stem = pt_path.stem
        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
        a_list = [x.to(torch.float64) for x in obj["a_spans_per_pair"]]
        b_list = [x.to(torch.float64) for x in obj["b_spans_per_pair"]]
        n_pairs = len(a_list)
        bt_list = [_soft_align_pole_b(a_p, b_p, tau=0.15) for a_p, b_p in zip(a_list, b_list)]
        ac_pair_list = [a_p - a_p.mean(dim=0, keepdim=True) for a_p in a_list]

        t1_mse_l, t1_cos_l = [], []
        cv_raw: dict[str, dict[str, list[float]]] = {}
        cv_cent: dict[str, dict[str, list[float]]] = {}
        cv_facet_cos: dict[str, dict[str, list[float]]] = {}
        cv_facet_ridge: dict[str, dict[str, list[float]]] = {}
        cv_2b: dict[str, dict[str, list[float]]] = {}

        for p_out in range(n_pairs):
            a_tr = torch.cat([a_list[i] for i in range(n_pairs) if i != p_out], dim=0)
            ac_in_tr = torch.cat([ac_pair_list[i] for i in range(n_pairs) if i != p_out], dim=0)
            bt_tr = torch.cat([bt_list[i] for i in range(n_pairs) if i != p_out], dim=0)
            a_te = a_list[p_out]
            ac_te = ac_pair_list[p_out]
            bt_te = bt_list[p_out]

            a_mean_tr, vr, diag = _fit_pole_a(a_tr, max_rank=8, ridge_rel=0.15)
            ac_glob_tr = a_tr - a_mean_tr.unsqueeze(0)
            a_er_tr = a_tr - (((ac_glob_tr @ vr.T) * diag.unsqueeze(0)) @ vr)
            trans_tr = bt_tr - a_er_tr
            mu_d = trans_tr.mean(dim=0)

            a_er_te = a_te - (((ac_te @ vr.T) * diag.unsqueeze(0)) @ vr)
            pred_t1 = a_er_te + mu_d.unsqueeze(0)
            t1_mse_l.append(float(((bt_te - pred_t1) ** 2).mean().item()))
            t1_cos_l.append(float(torch.nn.functional.cosine_similarity(pred_t1, bt_te, dim=-1).mean().item()))

            a_tr_u = a_tr / a_tr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            a_te_u = a_te / a_te.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            ac_tr_u = ac_in_tr / ac_in_tr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            ac_te_u = ac_te / ac_te.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            trans_u = trans_tr / trans_tr.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            for k_val in (2, 4, 8):
                lbl_raw = _cluster_ward(a_tr_u, k_val)
                pa_raw, pd_raw = _build_proto_from_labels(a_tr_u, trans_tr, mu_d, vr, lbl_raw, k_val)

                lbl_cent = _cluster_ward(ac_tr_u, k_val)
                pa_cent, pd_cent = _build_proto_from_labels(ac_tr_u, trans_tr, mu_d, vr, lbl_cent, k_val)

                lbl_fac = _cluster_ward(trans_u, k_val)
                pa_fac, pd_fac = _build_proto_from_labels(ac_tr_u, trans_tr, mu_d, vr, lbl_fac, k_val)
                n_actual_k = int(pd_fac.shape[0])
                onehot = torch.zeros((a_tr.shape[0], n_actual_k), dtype=torch.float64)
                uniq = sorted(set(lbl_fac.tolist()))
                idx_map = {old: new for new, old in enumerate(uniq)}
                mapped = torch.tensor([idx_map[int(v)] for v in lbl_fac.tolist()], dtype=torch.long)
                onehot.scatter_(1, mapped.unsqueeze(1), 1.0)
                w_ridge = _fit_ridge_router(ac_tr_u, onehot, ridge_rel=1.0)

                for tau_mv in (0.07, 0.15, 0.30):
                    for top_k, rtag in ((None, "dense"), (2, "warp2")):
                        if top_k is not None and k_val <= 2:
                            continue
                        cfg = f"K{k_val}_tau{tau_mv}_{rtag}"
                        for store, q_u, pa_m, pd_m in (
                            (cv_raw, a_te_u, pa_raw, pd_raw),
                            (cv_cent, ac_te_u, pa_cent, pd_cent),
                            (cv_facet_cos, ac_te_u, pa_fac, pd_fac),
                        ):
                            w = _route_cos_weights(q_u, pa_m, tau_mv, top_k=top_k)
                            sh = w @ pd_m
                            pr = a_er_te + sh
                            rec = store.setdefault(cfg, {"mse": [], "cos": [], "shift_std": []})
                            rec["mse"].append(float(((bt_te - pr) ** 2).mean().item()))
                            rec["cos"].append(
                                float(torch.nn.functional.cosine_similarity(pr, bt_te, dim=-1).mean().item())
                            )
                            rec["shift_std"].append(_pairwise_cos_std(sh))

                for r_tau in (0.25, 0.5, 1.0):
                    cfg_r = f"K{k_val}_rtau{r_tau}"
                    logits = (ac_te_u @ w_ridge) / float(r_tau)
                    w_r = torch.softmax(logits, dim=-1)
                    sh_r = w_r @ pd_fac
                    pr_r = a_er_te + sh_r
                    rec_r = cv_facet_ridge.setdefault(cfg_r, {"mse": [], "cos": [], "shift_std": []})
                    rec_r["mse"].append(float(((bt_te - pr_r) ** 2).mean().item()))
                    rec_r["cos"].append(
                        float(torch.nn.functional.cosine_similarity(pr_r, bt_te, dim=-1).mean().item())
                    )
                    rec_r["shift_std"].append(_pairwise_cos_std(sh_r))

            # Track 2B: Lemma-1 Ridge Cross-Covariance Transport W_B (prompt-centered)
            r_b_tr = trans_tr - mu_d.unsqueeze(0)
            u_ac, sv_ac, vh_ac = torch.linalg.svd(ac_in_tr, full_matrices=False)
            for rank_wb in (4, 8):
                r_use = min(rank_wb, int(sv_ac.numel()))
                v_sub, u_sub, sv_sub = vh_ac[:r_use], u_ac[:, :r_use], sv_ac[:r_use]
                ur_orth = u_sub.T @ r_b_tr
                ur_orth = ur_orth - (ur_orth @ vr.T) @ vr
                for ridge_wb in (0.5, 1.5, 4.0):
                    lam_wb = ridge_wb * float(sv_ac[0].item() ** 2)
                    gains = sv_sub / (sv_sub ** 2 + lam_wb)
                    delta_wb = ((ac_te @ v_sub.T) * gains.unsqueeze(0)) @ ur_orth
                    pr_2b = pred_t1 + delta_wb
                    rec_b = cv_2b.setdefault(f"r{rank_wb}_lam{ridge_wb}", {"mse": [], "cos": [], "shift_std": []})
                    rec_b["mse"].append(float(((bt_te - pr_2b) ** 2).mean().item()))
                    rec_b["cos"].append(
                        float(torch.nn.functional.cosine_similarity(pr_2b, bt_te, dim=-1).mean().item())
                    )
                    rec_b["shift_std"].append(_pairwise_cos_std(mu_d.unsqueeze(0) + delta_wb))

        t1_mse = float(np.mean(t1_mse_l))
        t1_cos = float(np.mean(t1_cos_l))
        _s_raw, b_raw = _summarize_cv(cv_raw, t1_mse, t1_cos)
        _s_cent, b_cent = _summarize_cv(cv_cent, t1_mse, t1_cos)
        _s_fcos, b_fcos = _summarize_cv(cv_facet_cos, t1_mse, t1_cos)
        _s_frid, b_frid = _summarize_cv(cv_facet_ridge, t1_mse, t1_cos)
        _s_2b, b_2b = _summarize_cv(cv_2b, t1_mse, t1_cos)

        a_full = torch.cat(a_list, dim=0)
        ac_full_pair = torch.cat(ac_pair_list, dim=0)
        bt_full = torch.cat(bt_list, dim=0)
        b_full = torch.cat(b_list, dim=0)
        b_cent_full = b_full - b_full.mean(dim=0, keepdim=True)
        _, sv_bf, _ = torch.linalg.svd(b_cent_full, full_matrices=False)
        b_top8_share = float((sv_bf[:8] ** 2).sum() / (sv_bf ** 2).sum())

        a_mean_f, vr_f, diag_f = _fit_pole_a(a_full, max_rank=8, ridge_rel=0.15)
        a_er_f = a_full - ((((a_full - a_mean_f.unsqueeze(0)) @ vr_f.T) * diag_f.unsqueeze(0)) @ vr_f)
        trans_f = bt_full - a_er_f
        mu_d_f = trans_f.mean(dim=0)
        rb_f = trans_f - mu_d_f.unsqueeze(0)
        _, sv_rb, _ = torch.linalg.svd(rb_f, full_matrices=False)
        rb_rms_ratio = float(rb_f.pow(2).mean().sqrt() / trans_f.pow(2).mean().sqrt())
        rb_top8_share = float((sv_rb[:8] ** 2).sum() / (sv_rb ** 2).sum())

        ac_full_u = ac_full_pair / ac_full_pair.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        trans_f_u = trans_f / trans_f.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        def _full_k4_profile(lbls_f: torch.Tensor, use_centered_ood: bool) -> dict:
            key_src = ac_full_u if use_centered_ood else (a_full / a_full.norm(dim=-1, keepdim=True))
            pa_4, pd_4 = _build_proto_from_labels(key_src, trans_f, mu_d_f, vr_f, lbls_f, 4)
            pu_4 = pd_4 / pd_4.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cm = pu_4 @ pu_4.T
            tri = torch.triu_indices(pd_4.shape[0], pd_4.shape[0], offset=1)
            sizes = [int((lbls_f == ci).sum().item()) for ci in range(4)]
            ood = {}
            for pname, pr_raw, pr_cent in ood_prompts:
                src = pr_cent if use_centered_ood else pr_raw
                qu = src / src.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                w_o = _route_cos_weights(qu, pa_4, tau_mv=0.07, top_k=2)
                ent = -(w_o * w_o.clamp_min(1e-12).log()).sum(dim=-1).mean().item() / math.log(4.0)
                ood[pname] = {
                    "norm_entropy_tau007_warp2": round(float(ent), 4),
                    "shift_cos_std_tau007_warp2": round(_pairwise_cos_std(w_o @ pd_4), 4),
                }
            return {
                "sizes": sizes,
                "delta_min_cos": round(float(cm[tri[0], tri[1]].min().item()), 4),
                "delta_mean_cos": round(float(cm[tri[0], tri[1]].mean().item()), 4),
                "ood": ood,
            }

        k4_cent_prof = _full_k4_profile(_cluster_ward(ac_full_u, 4), True)
        k4_fac_prof = _full_k4_profile(_cluster_ward(trans_f_u, 4), True)

        c_rec = census_map.get(stem, {})
        all_bank_results[stem] = {
            "n_pairs": n_pairs,
            "n_tokens_a": int(a_full.shape[0]),
            "n_tokens_b": int(bt_full.shape[0]),
            "census": {
                "regime": c_rec.get("regime"),
                "mpc": c_rec.get("mpc"),
                "exact_pole_a_eff_rank": c_rec.get("exact_pole_a_eff_rank"),
            },
            "pole_b_geometry": {
                "b_top8_energy_share": round(b_top8_share, 4),
                "rb_rms_over_trans_rms": round(rb_rms_ratio, 4),
                "rb_top8_energy_share": round(rb_top8_share, 4),
            },
            "track1_baseline": {"mse": round(t1_mse, 4), "cos": round(t1_cos, 4)},
            "track2a_raw_best": {"config": b_raw[0], **b_raw[1]},
            "track2a_centered_best": {"config": b_cent[0], **b_cent[1]},
            "track2a_facet_cos_best": {"config": b_fcos[0], **b_fcos[1]},
            "track2a_facet_ridge_best": {"config": b_frid[0], **b_frid[1]},
            "track2b_best": {"config": b_2b[0], **b_2b[1]},
            "k4_centered_pole_a": k4_cent_prof,
            "k4_facet_trans": k4_fac_prof,
        }
        print(
            f"[{stem:22s}] R_B/T={rb_rms_ratio:.3f} | T1 cos={t1_cos:.4f} | "
            f"raw={b_raw[1]['mse_delta_pct_vs_t1']:+.2f}% | "
            f"cent={b_cent[1]['mse_delta_pct_vs_t1']:+.2f}% | "
            f"fac_cos={b_fcos[1]['mse_delta_pct_vs_t1']:+.2f}% | "
            f"fac_rid={b_frid[1]['mse_delta_pct_vs_t1']:+.2f}% | "
            f"2B={b_2b[1]['mse_delta_pct_vs_t1']:+.2f}% | "
            f"fac_min_cos={k4_fac_prof['delta_min_cos']:.3f} cent_min_cos={k4_cent_prof['delta_min_cos']:.3f}"
        )

    out_dir = REPO_ROOT / "data" / "exp_unpooled_sliders" / "phase5_cpu_audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multivector_pole_b_audit.json"
    out_path.write_text(json.dumps(all_bank_results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
