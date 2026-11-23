#!/usr/bin/env python3
"""CPU-only Track 1 bank-wide cross-interference & multi-slider composition audit.

Reads all `*__fl2va__*.safetensors` in `$H3_MODEL_STORE/control_vectors/bank`
(or ComfyUI's `models/control_vectors/bank`), computes pairwise cross-interference metrics:
  1) Direction 1 (Peer Erasure vs Target): ||V_r^(j) mu_hat_d^(i)||^2 and D_j-weighted
  2) Direction 2 (Peer Injection vs Erased Pole-A): ||V_r^(i) mu_hat_d^(j)||^2
  3) Self-Overlap Trap (i == j): ||V_r^(i) mu_hat_d^(i)||^2
  4) Grassmann subspace overlap cos(theta_1), mean_sq(SVs), cos(mu_hat_d^(i), mu_hat_d^(j))

Simulates multi-slider composition algorithms on cached held-out post-refiner
prompt rows (`data/exp_unpooled_sliders/prompts/{covered_market,indoor_evening,indoor_neutral}.pt`):
  (a) Shipped Clamped Union (`control_vectors._stacked_erasure_block_shift64`)
  (b) Naive Global `V_r_union perp span(mu_d^(1..K))`
  (c) One-Sided Per-Slider RASteer ROS (`V_r^(j) perp {mu_d^(k): k!=j}`, then clamped union)
  (d) Two-Sided Equilibrium (Per-Slider ROS + clamped union + cross-erasure-filtered a_j)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "control_vectors.py").is_file():
            return p
    return Path.cwd()


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

import control_vectors as cv  # noqa: E402


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


def _summary_stats(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p99": round(float(np.percentile(arr, 99)), 4),
        "max": round(float(arr.max()), 4),
    }


def _subspace_project_matrix(vr: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
    return vr.T @ (diag.unsqueeze(-1) * vr)


def _eval_stack_on_rows(
    base_rows64: torch.Tensor,
    entries: list[tuple[str, dict[str, torch.Tensor], float]],
    mode_name: str,
    preserve_filtered_norm: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one multi-slider composition algorithm on `[M, D]` float64 rows."""
    device = base_rows64.device
    centered = base_rows64 - base_rows64.mean(dim=0, keepdim=True)
    target = cv.NORM_MATCHED_TARGET

    st0_name, op0, s0 = entries[0]
    vr0 = op0["vr"].to(dtype=torch.float64, device=device)
    d0 = op0["diag"].to(dtype=torch.float64, device=device).clamp(0.0, 1.0)

    a_terms_raw: list[torch.Tensor] = []
    mu_units: list[torch.Tensor] = []
    for _stem, op, s_k in entries:
        mu = op["mu_d"].to(dtype=torch.float64, device=device)
        nrm = float(mu.norm().item())
        mu_u = mu / max(nrm, 1e-12)
        mu_units.append(mu_u)
        a_terms_raw.append((float(s_k) * target / max(nrm, 1e-12)) * mu)

    if mode_name == "shipped_clamped_union":
        cv_entries = [
            (op["mu_d"], s_k, (op["vr"], op["diag"], op["mu_d"], None, None))
            for _stem, op, s_k in entries
        ]
        shift = cv._stacked_erasure_block_shift64(base_rows64, cv_entries)
        steered = base_rows64 + shift

        w_blocks = []
        e_budget_sum = 0.0
        for _stem, op, s_k in entries:
            w_eff = max(0.0, min(float(s_k), 1.0))
            e_budget_sum += w_eff * target
            dk = op["diag"].to(torch.float64).clamp(0.0, 1.0)
            vk = op["vr"].to(torch.float64)
            w_blocks.append(torch.sqrt((w_eff * dk).clamp(0.0, 1.0)).unsqueeze(-1) * vk)
        _, sv, vh = torch.linalg.svd(torch.cat(w_blocks, dim=0), full_matrices=False)
        du = (sv ** 2).clamp(0.0, 1.0)
        keep = du > 1e-9
        vr_u, du_k = vh[keep], du[keep]
        add_tot = sum(a_terms_raw)
        t_rms = float(add_tot.norm().item())
        e_own = ((centered @ vr_u.T) * du_k.unsqueeze(0)) @ vr_u
        e_rms = float(cv._rms_row_norm64(e_own))
        e_b = min(e_budget_sum, max(target, t_rms))
        e_cap = min(1.0, e_b / e_rms) if e_rms > 1e-9 else 1.0
        raw_rms = float(cv._rms_row_norm64(add_tot.unsqueeze(0) - e_own * e_cap))
        resc = t_rms / max(raw_rms, 1e-12)
        lam_max = float(du_k.max().item()) if du_k.numel() else 0.0
        eff_erasure_gain = e_cap * resc

    elif mode_name == "naive_global_orth":
        w_blocks = []
        e_budget_sum = 0.0
        for _stem, op, s_k in entries:
            w_eff = max(0.0, min(float(s_k), 1.0))
            e_budget_sum += w_eff * target
            dk = op["diag"].to(torch.float64).clamp(0.0, 1.0)
            vk = op["vr"].to(torch.float64)
            w_blocks.append(torch.sqrt((w_eff * dk).clamp(0.0, 1.0)).unsqueeze(-1) * vk)
        _, sv, vh = torch.linalg.svd(torch.cat(w_blocks, dim=0), full_matrices=False)
        du = (sv ** 2).clamp(0.0, 1.0)
        keep = du > 1e-9
        orth = cv.orthogonalize_erasure_against_holds(vh[keep], du[keep], mu_units)
        assert orth is not None
        vr_u = orth[0].to(torch.float64)
        du_k = orth[1].to(torch.float64)
        add_tot = sum(a_terms_raw)
        t_rms = float(add_tot.norm().item())
        e_own = ((centered @ vr_u.T) * du_k.unsqueeze(0)) @ vr_u
        e_rms = float(cv._rms_row_norm64(e_own))
        e_b = min(e_budget_sum, max(target, t_rms))
        e_cap = min(1.0, e_b / e_rms) if e_rms > 1e-9 else 1.0
        raw = add_tot.unsqueeze(0) - e_own * e_cap
        raw_rms = float(cv._rms_row_norm64(raw))
        resc = t_rms / max(raw_rms, 1e-12)
        shift = raw * resc
        steered = base_rows64 + shift
        lam_max = float(du_k.max().item()) if du_k.numel() else 0.0
        eff_erasure_gain = e_cap * resc

    elif mode_name in ("one_sided_rasteer_ros", "two_sided_equilibrium"):
        w_blocks = []
        e_budget_sum = 0.0
        for idx, (_stem, op, s_k) in enumerate(entries):
            others = [mu_units[m] for m in range(len(entries)) if m != idx]
            orth = cv.orthogonalize_erasure_against_holds(op["vr"], op["diag"], others)
            if orth is None:
                continue
            vk = orth[0].to(torch.float64)
            dk = orth[1].to(torch.float64).clamp(0.0, 1.0)
            w_eff = max(0.0, min(float(s_k), 1.0))
            e_budget_sum += w_eff * target
            w_blocks.append(torch.sqrt((w_eff * dk).clamp(0.0, 1.0)).unsqueeze(-1) * vk)
        _, sv, vh = torch.linalg.svd(torch.cat(w_blocks, dim=0), full_matrices=False)
        du = (sv ** 2).clamp(0.0, 1.0)
        keep = du > 1e-9
        vr_u, du_k = vh[keep], du[keep]

        if mode_name == "one_sided_rasteer_ros":
            a_terms = a_terms_raw
        else:
            a_terms = []
            for idx, a_k in enumerate(a_terms_raw):
                peer_w = []
                for m_idx, (_st_m, op_m, s_m) in enumerate(entries):
                    if m_idx == idx:
                        continue
                    others_m = [mu_units[q] for q in range(len(entries)) if q != m_idx]
                    orth_m = cv.orthogonalize_erasure_against_holds(op_m["vr"], op_m["diag"], others_m)
                    if orth_m is None:
                        continue
                    w_m = max(0.0, min(float(s_m), 1.0))
                    peer_w.append(
                        torch.sqrt((w_m * orth_m[1].to(torch.float64)).clamp(0.0, 1.0)).unsqueeze(-1)
                        * orth_m[0].to(torch.float64)
                    )
                if peer_w:
                    _, sv_p, vh_p = torch.linalg.svd(torch.cat(peer_w, dim=0), full_matrices=False)
                    du_p = (sv_p ** 2).clamp(0.0, 1.0)
                    kp = du_p > 1e-9
                    a_filt = a_k - (((a_k @ vh_p[kp].T) * du_p[kp]) @ vh_p[kp])
                    if preserve_filtered_norm:
                        fn = float(a_filt.norm().item())
                        if fn > 1e-9:
                            a_filt = a_filt * (float(a_k.norm().item()) / fn)
                    a_terms.append(a_filt)
                else:
                    a_terms.append(a_k)

        add_tot = sum(a_terms)
        t_rms = float(add_tot.norm().item())
        e_own = ((centered @ vr_u.T) * du_k.unsqueeze(0)) @ vr_u
        e_rms = float(cv._rms_row_norm64(e_own))
        e_b = min(e_budget_sum, max(target, t_rms))
        e_cap = min(1.0, e_b / e_rms) if e_rms > 1e-9 else 1.0
        raw = add_tot.unsqueeze(0) - e_own * e_cap
        raw_rms = float(cv._rms_row_norm64(raw))
        resc = t_rms / max(raw_rms, 1e-12)
        shift = raw * resc
        steered = base_rows64 + shift
        lam_max = float(du_k.max().item()) if du_k.numel() else 0.0
        eff_erasure_gain = e_cap * resc
    else:
        raise ValueError(mode_name)

    st_centered = steered - steered.mean(dim=0, keepdim=True)
    base_pa = float(torch.linalg.matrix_norm(((centered @ vr0.T) * torch.sqrt(d0).unsqueeze(0))).item())
    st_pa = float(torch.linalg.matrix_norm(((st_centered @ vr0.T) * torch.sqrt(d0).unsqueeze(0))).item())
    pa_resid_ratio = st_pa / max(base_pa, 1e-12)

    delta_mean = (steered - base_rows64).mean(dim=0)
    proj_per_slider = {}
    for (stem, _op, _s_k), mu_u in zip(entries, mu_units):
        proj_per_slider[stem] = round(float(torch.dot(delta_mean, mu_u).item()), 2)

    return steered, {
        "rms_shift_norm": round(float(cv._rms_row_norm64(steered - base_rows64)), 2),
        "target_rms": round(t_rms, 2),
        "e_cap": round(e_cap, 4),
        "rms_rescale_factor": round(resc, 4),
        "eff_erasure_gain": round(eff_erasure_gain, 4),
        "union_lam_max": round(lam_max, 4),
        "slider0_pole_a_resid_ratio": round(pa_resid_ratio, 4),
        "mean_shift_norm": round(float(delta_mean.norm().item()), 2),
        "proj_on_targets": proj_per_slider,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CPU Track 1 bank-wide cross-interference and stack audit.")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output JSON path (defaults to phase5_cpu_audits/bank_cross_interference_audit.json when writing).",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Overwrite default bank_cross_interference_audit.json if it already exists.",
    )
    args = ap.parse_args(argv)

    bank_dir = _find_bank_dir()
    files = sorted(bank_dir.glob("*__fl2va__*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No *__fl2va__*.safetensors found under {bank_dir}")

    by_stem: dict[str, tuple[Path, dict[str, torch.Tensor]]] = {}
    for p in files:
        stem = p.name.split("__fl2va__", 1)[0]
        t, _m = cv.load_vector(p)
        if all(k in t for k in ("erasure.vr", "erasure.diag", "erasure.mu_d")):
            if stem not in by_stem:
                by_stem[stem] = (p, {
                    "vr": t["erasure.vr"].to(torch.float64),
                    "diag": t["erasure.diag"].to(torch.float64).clamp(0.0, 1.0),
                    "mu_d": t["erasure.mu_d"].to(torch.float64),
                    "v1d": t["vector.post_refiner"].to(torch.float64),
                })

    stems = sorted(by_stem.keys())
    n = len(stems)
    print(f"Loaded {n} __fl2va__ vectors with complete erasure tensors from {len(files)} files.")

    mu_units = torch.stack(
        [by_stem[s][1]["mu_d"] / by_stem[s][1]["mu_d"].norm().clamp_min(1e-12) for s in stems],
        dim=0,
    )
    cos_mu_mat = (mu_units @ mu_units.T).cpu().numpy()

    e_raw_mat = np.zeros((n, n), dtype=np.float64)
    e_d_mat = np.zeros((n, n), dtype=np.float64)
    g_top1_mat = np.zeros((n, n), dtype=np.float64)
    g_msq_mat = np.zeros((n, n), dtype=np.float64)

    for j, sj in enumerate(stems):
        vr_j = by_stem[sj][1]["vr"]
        d_j = by_stem[sj][1]["diag"]
        proj = mu_units @ vr_j.T
        e_raw_mat[:, j] = (proj ** 2).sum(dim=1).cpu().numpy()
        e_d_mat[:, j] = ((proj ** 2) * d_j.unsqueeze(0)).sum(dim=1).cpu().numpy()

    for i in range(n):
        vri = by_stem[stems[i]][1]["vr"]
        for j in range(i + 1, n):
            vrj = by_stem[stems[j]][1]["vr"]
            sv = torch.linalg.svdvals(vri @ vrj.T)
            top1 = float(sv[0].item())
            msq = float((sv ** 2).mean().item())
            g_top1_mat[i, j] = g_top1_mat[j, i] = top1
            g_msq_mat[i, j] = g_msq_mat[j, i] = msq

    off_diag_mask = ~np.eye(n, dtype=bool)
    e_raw_off = e_raw_mat[off_diag_mask].tolist()
    e_d_off = e_d_mat[off_diag_mask].tolist()
    self_raw = np.diag(e_raw_mat).tolist()
    self_d = np.diag(e_d_mat).tolist()
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    g_top1_off = g_top1_mat[triu_mask].tolist()
    g_msq_off = g_msq_mat[triu_mask].tolist()
    cos_mu_off = np.abs(cos_mu_mat[triu_mask]).tolist()

    ordered_pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ordered_pairs.append({
                "target_concept": stems[i],
                "erasure_concept": stems[j],
                "raw_energy_frac": round(float(e_raw_mat[i, j]), 4),
                "d_weighted_energy_frac": round(float(e_d_mat[i, j]), 4),
                "target_mu_cos": round(float(cos_mu_mat[i, j]), 4),
                "subspace_top1_cos": round(float(g_top1_mat[i, j]), 4),
                "subspace_mean_sq": round(float(g_msq_mat[i, j]), 4),
            })

    top15_dir1_raw = sorted(ordered_pairs, key=lambda x: x["raw_energy_frac"], reverse=True)[:15]
    top15_dir1_d = sorted(ordered_pairs, key=lambda x: x["d_weighted_energy_frac"], reverse=True)[:15]

    self_rows = sorted(
        [
            {
                "concept": stems[i],
                "self_raw_energy_frac": round(float(e_raw_mat[i, i]), 4),
                "self_d_weighted_energy_frac": round(float(e_d_mat[i, i]), 4),
                "mu_d_norm": round(float(by_stem[stems[i]][1]["mu_d"].norm().item()), 2),
                "rank": int(by_stem[stems[i]][1]["vr"].shape[0]),
            }
            for i in range(n)
        ],
        key=lambda x: x["self_raw_energy_frac"],
        reverse=True,
    )

    prompt_files = [
        REPO_ROOT / "data/exp_unpooled_sliders/prompts/covered_market.pt",
        REPO_ROOT / "data/exp_unpooled_sliders/prompts/indoor_evening.pt",
        REPO_ROOT / "data/exp_unpooled_sliders/prompts/indoor_neutral.pt",
    ]
    missing_prompts = [str(pf) for pf in prompt_files if not pf.is_file()]
    if missing_prompts:
        raise FileNotFoundError(
            "Cached post-refiner prompt tensors are required for stack simulations: "
            + ", ".join(missing_prompts)
        )

    loaded_prompts = []
    for pf in prompt_files:
        d = torch.load(pf, map_location="cpu", weights_only=False)
        sm = d["scene_mask"]
        rows64 = d["ref"][sm].to(torch.float64)
        loaded_prompts.append((d["name"], rows64))

    stacks = [
        ("noir_chaos_de_slop", [("noir", 1.25), ("chaos_panic", 1.0), ("de_llm_slop", 0.9)]),
        ("origami_noir_de_slop", [("paper_origami", 1.2), ("noir", 1.15), ("de_llm_slop", 0.9)]),
        ("felt_noir_de_slop", [("felt_muppet", 1.2), ("noir", 1.15), ("de_llm_slop", 0.9)]),
        ("claymation_noir_chaos", [("claymation_stopmotion", 1.2), ("noir", 1.15), ("chaos_panic", 1.0)]),
    ]

    algos = [
        ("single_slider0_only", False),
        ("shipped_clamped_union", False),
        ("naive_global_orth", False),
        ("one_sided_rasteer_ros", False),
        ("two_sided_equilibrium", False),
        ("two_sided_equilibrium_renorm", True),
    ]

    stack_simulations = {}
    for stack_name, spec in stacks:
        entries = [(s, by_stem[s][1], float(v)) for s, v in spec if s in by_stem]
        if len(entries) != len(spec):
            missing = [s for s, _ in spec if s not in by_stem]
            print(f"Skipping stack [{stack_name}] (missing stems: {', '.join(missing)})")
            continue
        per_prompt = {}
        for pname, rows64 in loaded_prompts:
            p_res = {}
            for algo_lbl, renorm_flag in algos:
                if algo_lbl == "single_slider0_only":
                    _, m_res = _eval_stack_on_rows(rows64, [entries[0]], "shipped_clamped_union")
                elif algo_lbl == "two_sided_equilibrium_renorm":
                    _, m_res = _eval_stack_on_rows(
                        rows64, entries, "two_sided_equilibrium", preserve_filtered_norm=True
                    )
                else:
                    _, m_res = _eval_stack_on_rows(
                        rows64, entries, algo_lbl, preserve_filtered_norm=renorm_flag
                    )
                p_res[algo_lbl] = m_res
            per_prompt[pname] = p_res

        agg = {}
        for algo_lbl, _ in algos:
            keys = (
                "eff_erasure_gain",
                "slider0_pole_a_resid_ratio",
                "rms_rescale_factor",
                "e_cap",
                "union_lam_max",
                "rms_shift_norm",
                "mean_shift_norm",
            )
            row_agg = {
                k: round(float(np.mean([per_prompt[pn][algo_lbl][k] for pn, _ in loaded_prompts])), 4)
                for k in keys
            }
            slider0_stem = entries[0][0]
            slider_names = [e[0] for e in entries] if algo_lbl != "single_slider0_only" else [slider0_stem]
            row_agg["mean_proj_on_targets"] = {
                sn: round(
                    float(
                        np.mean([per_prompt[pn][algo_lbl]["proj_on_targets"][sn] for pn, _ in loaded_prompts])
                    ),
                    2,
                )
                for sn in slider_names
            }
            agg[algo_lbl] = row_agg
        stack_simulations[stack_name] = {"per_prompt": per_prompt, "mean_across_prompts": agg}

    out = {
        "n_fl2va_erasure_vectors": n,
        "stems": stems,
        "pairwise_stats": {
            "dir1_raw_energy_frac": _summary_stats(e_raw_off),
            "dir1_d_weighted_energy_frac": _summary_stats(e_d_off),
            "self_raw_energy_frac": _summary_stats(self_raw),
            "self_d_weighted_energy_frac": _summary_stats(self_d),
            "subspace_top1_cos": _summary_stats(g_top1_off),
            "subspace_mean_sq_sv": _summary_stats(g_msq_off),
            "abs_target_mu_cos": _summary_stats(cos_mu_off),
        },
        "top15_cross_interference_raw": top15_dir1_raw,
        "top15_cross_interference_d_weighted": top15_dir1_d,
        "top15_self_overlap_trap": self_rows[:15],
        "bottom5_self_overlap_trap": self_rows[-5:],
        "stack_simulations": stack_simulations,
    }

    out_dir = REPO_ROOT / "data" / "exp_unpooled_sliders" / "phase5_cpu_audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    default_out = out_dir / "bank_cross_interference_audit.json"
    out_path = args.out.resolve() if args.out is not None else default_out
    if args.out is not None or args.write or not default_out.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(f"[dry-run] Computed n={n} audit; pass --write or --out to overwrite {default_out}")
    print("Pairwise stats:", json.dumps(out["pairwise_stats"], indent=2))
    print("Top-5 self-overlap trap:", json.dumps(self_rows[:5], indent=2))
    print("Top-5 cross-interference (raw):", json.dumps(top15_dir1_raw[:5], indent=2))
    for sn, sd in stack_simulations.items():
        print(f"\nStack [{sn}] mean across prompts:")
        print(json.dumps(sd["mean_across_prompts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
