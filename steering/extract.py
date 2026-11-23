"""CPU `< 50 ms` contrastive extraction, Pole-A erasure SVD, statistical quality metrics,
`SpanCache` (`__spans.safetensors`) round-trip, and `.safetensors` I/O (`unit_mean=False` default)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save as save_safetensors_bytes

from .constants import (
    CANONICAL_PROVENANCE_KEYS,
    COMPOSED_ERASURE_MAX_RANK,
    CUSTOM_FAMILY_RE,
    ENCODER_DIM,
    ERASURE_MATCH_TAU,
    ERASURE_MAX_RANK,
    ERASURE_RIDGE_REL,
    FAMILY_TENSOR_KEYS,
    FORMAT_VERSION,
    KNOWN_FAMILIES,
    KNOWN_H3_CHECKPOINTS,
    KNOWN_REF2VA_GUIDE_LORAS,
    LLM_VECTOR_KEY_RE,
    NAME_RE,
    NORM_MATCHED_TARGET,
    NORM_SHIFT_WINDOW,
    PIKA_FLEX_VARIANT_RE,
    POST_REFINER_DIM,
    REQUIRED_METADATA_KEYS,
    SPAN_CACHE_FORMAT_VERSION,
    VALID_ALL_POOLING,
    VALID_FRAMES,
    VALID_MODES,
    VALID_POOLING,
    VIDEO_H3_FAMILIES,
    import_repo_module,
)
from .data import (
    PairSet,
    RoutingSpec,
    pair_set_differing_sections,
    parse_provenance_table,
    parse_routing_table,
)
from .runtime import (
    _apply_holds_to_rows64,
    _apply_holds_to_shift64,
    is_additive_mode,
    is_norm_matched_mode,
    orthogonalize_erasure_against_holds,
)

_h3_spans = import_repo_module("h3_spans")


def _check_pair_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError(
            f"a and b must be 2D tensors of identical shape [n, d], got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    n, d = a.shape
    if n < 1 or d < 1:
        raise ValueError(f"a and b must be non-empty [n, d], got {tuple(a.shape)}")
    a64 = a.detach().to(dtype=torch.float64)
    b64 = b.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(a64).all()) or not bool(torch.isfinite(b64).all()):
        raise ValueError("a and b must contain only finite values")

    if weights is None:
        w64 = torch.ones(n, dtype=torch.float64, device=a64.device)
    else:
        w_t = (
            weights
            if isinstance(weights, torch.Tensor)
            else torch.as_tensor(weights, dtype=torch.float64, device=a64.device)
        )
        if w_t.ndim != 1 or w_t.shape[0] != n:
            raise ValueError(f"weights must have shape [{n}], got {tuple(w_t.shape)}")
        w64 = w_t.detach().to(device=a64.device, dtype=torch.float64)
        if not bool(torch.isfinite(w64).all()) or bool((w64 == 0).any()):
            raise ValueError("weights must be finite and non-zero")
    return a64, b64, w64


@dataclass(frozen=True)
class PooledPairSet:
    a_mat: torch.Tensor
    b_mat: torch.Tensor
    w_vec: torch.Tensor
    raw_vec: torch.Tensor
    vector: torch.Tensor
    hold_basis: torch.Tensor | None
    pair_spans: list[dict[str, Any]]
    enc_raw_vec: torch.Tensor | None
    n_holds: int
    erasure_mu_d: torch.Tensor | None = None
    erasure_vr: torch.Tensor | None = None
    erasure_diag: torch.Tensor | None = None


@dataclass(frozen=True)
class SpanCache:
    """Exact unpooled `fp32` `__spans.safetensors` payload for `< 50 ms` CPU re-extraction."""

    span_a: torch.Tensor  # [sum(T_a_p), 5376] fp32
    span_b: torch.Tensor  # [sum(T_b_p), 5376] fp32
    lens_a: torch.Tensor  # [N_pairs] int32
    lens_b: torch.Tensor  # [N_pairs] int32
    pair_weights: torch.Tensor  # [N_pairs] fp32
    hold_a: torch.Tensor  # [N_holds, 5376] fp32
    hold_b: torch.Tensor  # [N_holds, 5376] fp32
    hold_weights: torch.Tensor  # [N_holds] fp32
    enc_a_rows: torch.Tensor | None  # [N_pairs, 5120] fp32
    enc_b_rows: torch.Tensor | None  # [N_pairs, 5120] fp32
    enc_hold_a: torch.Tensor | None  # [N_holds, 5120] fp32
    enc_hold_b: torch.Tensor | None  # [N_holds, 5120] fp32
    metadata: dict[str, Any]

    def split_spans(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        la = [int(x) for x in self.lens_a.detach().cpu().tolist()]
        lb = [int(x) for x in self.lens_b.detach().cpu().tolist()]
        return list(torch.split(self.span_a, la, dim=0)), list(torch.split(self.span_b, lb, dim=0))


def build_pole_a_erasure(
    a_spans: Sequence[torch.Tensor],
    b_spans: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    *,
    hold_basis: torch.Tensor | None = None,
    max_rank: int = ERASURE_MAX_RANK,
    ridge_rel: float = ERASURE_RIDGE_REL,
    tau: float = ERASURE_MATCH_TAU,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Build float32 `(erasure.mu_d [D], erasure.vr [r, D], erasure.diag [r])` from unpooled
    per-pair post-refiner span rows `(A_p, B_p)`."""
    if not a_spans or len(a_spans) != len(b_spans):
        return None
    w_list = [1.0] * len(a_spans) if weights is None else [float(w) for w in weights]
    if len(w_list) != len(a_spans):
        raise ValueError("build_pole_a_erasure: weights length mismatch")

    a_all_list: list[torch.Tensor] = []
    d_sum: torch.Tensor | None = None
    w_tok_total = 0.0
    for a_p, b_p, wp in zip(a_spans, b_spans, w_list):
        if a_p.ndim != 2 or b_p.ndim != 2 or a_p.shape[0] < 1 or b_p.shape[0] < 1:
            continue
        a64 = a_p.detach().to(device="cpu", dtype=torch.float64)
        b64 = b_p.detach().to(device="cpu", dtype=torch.float64)
        a_u = a64 / torch.linalg.vector_norm(a64, dim=-1, keepdim=True).clamp(min=1e-12)
        b_u = b64 / torch.linalg.vector_norm(b64, dim=-1, keepdim=True).clamp(min=1e-12)
        p_mat = torch.softmax((a_u @ b_u.T) / float(tau), dim=-1)
        d_p = p_mat @ b64 - a64
        a_all_list.append(a64)
        w_eff = abs(float(wp)) if float(wp) != 0.0 else 1.0
        contrib = w_eff * d_p.sum(dim=0)
        d_sum = contrib if d_sum is None else (d_sum + contrib)
        w_tok_total += w_eff * float(a64.shape[0])

    if not a_all_list or d_sum is None or w_tok_total <= 0.0:
        return None
    a_all = torch.cat(a_all_list, dim=0)
    if a_all.shape[0] < 2:
        return None

    mu_d = d_sum / w_tok_total
    a_c = a_all - a_all.mean(dim=0, keepdim=True)
    if hold_basis is not None and hold_basis.numel() > 0:
        bh = hold_basis.detach().to(device="cpu", dtype=torch.float64)
        mu_d = mu_d - (mu_d @ bh.T) @ bh
        a_c = a_c - (a_c @ bh.T) @ bh

    mu_norm = float(torch.linalg.vector_norm(mu_d).item())
    if not math.isfinite(mu_norm) or mu_norm <= 1e-12:
        return None

    _u, s_vals, vh = torch.linalg.svd(a_c, full_matrices=False)
    if s_vals.numel() == 0 or float(s_vals[0].item()) <= 1e-12:
        return None
    n_pos = int((s_vals > 1e-9 * s_vals[0]).sum().item())
    r = min(int(max_rank), n_pos)
    if r < 1:
        return None
    vr = vh[:r].contiguous()
    lam = float(ridge_rel) * float(s_vals[0].item() ** 2)
    s_sq = s_vals[:r] ** 2
    diag = (s_sq / (s_sq + lam)).clamp(0.0, 1.0).contiguous()
    return (
        mu_d.to(dtype=torch.float32).contiguous(),
        vr.to(dtype=torch.float32).contiguous(),
        diag.to(dtype=torch.float32).contiguous(),
    )


def attach_erasure_tensors(
    tensors_to_save: dict[str, torch.Tensor],
    pooled: PooledPairSet,
) -> None:
    if pooled.erasure_vr is not None and pooled.erasure_diag is not None:
        tensors_to_save["erasure.vr"] = pooled.erasure_vr
        tensors_to_save["erasure.diag"] = pooled.erasure_diag
        if pooled.erasure_mu_d is not None:
            tensors_to_save["erasure.mu_d"] = pooled.erasure_mu_d


def build_spatial_vision_erasure(
    a_spans: Sequence[torch.Tensor],
    b_spans: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
    *,
    hold_basis: torch.Tensor | None = None,
    max_rank: int = ERASURE_MAX_RANK,
    ridge_rel: float = ERASURE_RIDGE_REL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not a_spans or len(a_spans) != len(b_spans):
        return None
    w_list = [1.0] * len(a_spans) if weights is None else [float(w) for w in weights]
    if len(w_list) != len(a_spans):
        raise ValueError("build_spatial_vision_erasure: weights length mismatch")

    a_all_list: list[torch.Tensor] = []
    d_sum: torch.Tensor | None = None
    w_tok_total = 0.0
    mismatched_shapes = False
    for a_p, b_p, wp in zip(a_spans, b_spans, w_list):
        if a_p.ndim != 2 or b_p.ndim != 2 or a_p.shape[0] < 1 or b_p.shape[0] < 1:
            continue
        if a_p.shape != b_p.shape:
            mismatched_shapes = True
            break
        a64 = a_p.detach().to(device="cpu", dtype=torch.float64)
        b64 = b_p.detach().to(device="cpu", dtype=torch.float64)
        d_p = b64 - a64
        a_all_list.append(a64)
        w_eff = abs(float(wp)) if float(wp) != 0.0 else 1.0
        contrib = w_eff * d_p.sum(dim=0)
        d_sum = contrib if d_sum is None else (d_sum + contrib)
        w_tok_total += w_eff * float(a64.shape[0])

    if mismatched_shapes:
        fallback = build_pole_a_erasure(
            a_spans,
            b_spans,
            weights=weights,
            hold_basis=hold_basis,
            max_rank=max_rank,
            ridge_rel=ridge_rel,
        )
        if fallback is None:
            return None
        a_cat_fb = torch.cat(
            [ap.detach().to(device="cpu", dtype=torch.float64) for ap in a_spans if ap.ndim == 2 and ap.shape[0] >= 1],
            dim=0,
        )
        mu_a_fb = a_cat_fb.mean(dim=0).to(dtype=torch.float32).contiguous()
        return (*fallback, mu_a_fb)

    if not a_all_list or d_sum is None or w_tok_total <= 0.0:
        return None
    a_all = torch.cat(a_all_list, dim=0)
    if a_all.shape[0] < 2:
        return None

    mu_d = d_sum / w_tok_total
    mu_a = a_all.mean(dim=0)
    a_c = a_all - mu_a.unsqueeze(0)
    if hold_basis is not None and hold_basis.numel() > 0:
        bh = hold_basis.detach().to(device="cpu", dtype=torch.float64)
        mu_d = mu_d - (mu_d @ bh.T) @ bh
        a_c = a_c - (a_c @ bh.T) @ bh

    mu_norm = float(torch.linalg.vector_norm(mu_d).item())
    if not math.isfinite(mu_norm) or mu_norm <= 1e-12:
        return None

    _u, s_vals, vh = torch.linalg.svd(a_c, full_matrices=False)
    if s_vals.numel() == 0 or float(s_vals[0].item()) <= 1e-12:
        return None
    n_pos = int((s_vals > 1e-9 * s_vals[0]).sum().item())
    r = min(int(max_rank), n_pos)
    if r < 1:
        return None
    vr = vh[:r].contiguous()
    lam = float(ridge_rel) * float(s_vals[0].item() ** 2)
    s_sq = s_vals[:r] ** 2
    diag = (s_sq / (s_sq + lam)).clamp(0.0, 1.0).contiguous()
    return (
        mu_d.to(dtype=torch.float32).contiguous(),
        vr.to(dtype=torch.float32).contiguous(),
        diag.to(dtype=torch.float32).contiguous(),
        mu_a.to(dtype=torch.float32).contiguous(),
    )


def compute_bus_erasure_basis(
    erasure_entries: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    rank: int = 16,
) -> torch.Tensor:
    k_max = int(rank)
    dim = 0
    p_acc: torch.Tensor | None = None
    for vr, diag in erasure_entries:
        if vr is None or diag is None or vr.ndim != 2 or vr.shape[0] < 1:
            continue
        vr64 = vr.detach().to(device="cpu", dtype=torch.float64)
        d64 = diag.detach().to(device="cpu", dtype=torch.float64).flatten().clamp(0.0, 1.0)
        r_use = min(int(vr64.shape[0]), int(d64.numel()))
        if r_use < 1:
            continue
        cur_dim = int(vr64.shape[1])
        if p_acc is None:
            dim = cur_dim
            p_acc = torch.zeros((dim, dim), dtype=torch.float64)
        elif cur_dim != dim:
            raise ValueError(f"Mismatched erasure dimension in compute_bus_erasure_basis: {cur_dim} vs {dim}")
        vr_u = vr64[:r_use]
        d_u = d64[:r_use]
        p_acc += vr_u.T @ (d_u.unsqueeze(-1) * vr_u)
    if p_acc is None or k_max <= 0:
        return torch.zeros((0, dim), dtype=torch.float32)
    p_sym = 0.5 * (p_acc + p_acc.T)
    evals, evecs = torch.linalg.eigh(p_sym)
    order = torch.argsort(evals, descending=True)[:k_max]
    top_vals = evals[order]
    keep = top_vals > 1e-12
    if not bool(keep.any()):
        return torch.zeros((0, dim), dtype=torch.float32)
    return evecs[:, order[keep]].T.to(dtype=torch.float32).contiguous()


def project_erasure_orthogonal_to_basis(
    vr: torch.Tensor,
    diag: torch.Tensor,
    orth_basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if vr.ndim != 2 or vr.shape[0] < 1:
        return None
    if orth_basis is None or orth_basis.numel() == 0:
        return (
            vr.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            diag.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).contiguous(),
        )
    if orth_basis.ndim != 2 or int(orth_basis.shape[1]) != int(vr.shape[1]):
        raise ValueError(
            f"orth_basis shape {tuple(orth_basis.shape)} incompatible with vr shape {tuple(vr.shape)}"
        )
    b64 = orth_basis.detach().to(device="cpu", dtype=torch.float64)
    _ub, sb, vhb = torch.linalg.svd(b64, full_matrices=False)
    b_keep = sb > 1e-12
    if not bool(b_keep.any()):
        return (
            vr.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            diag.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0).contiguous(),
        )
    bh = vhb[b_keep]
    vr64 = vr.detach().to(device="cpu", dtype=torch.float64)
    d64 = diag.detach().to(device="cpu", dtype=torch.float64).flatten().clamp(0.0, 1.0)
    r_use = min(int(vr64.shape[0]), int(d64.numel()))
    vr_proj = vr64[:r_use] - (vr64[:r_use] @ bh.T) @ bh
    m_mat = torch.sqrt(d64[:r_use]).unsqueeze(-1) * vr_proj
    _u, s_vals, vh = torch.linalg.svd(m_mat, full_matrices=False)
    d_new = (s_vals ** 2).clamp(0.0, 1.0)
    keep = d_new > 1e-6
    if not bool(keep.any()):
        return None
    return (
        vh[keep].to(dtype=torch.float32).contiguous(),
        d_new[keep].to(dtype=torch.float32).contiguous(),
    )


def compose_erasure_payload(
    entries: Sequence[tuple[Any, ...]],
    *,
    holds: Sequence[tuple[torch.Tensor, float]] = (),
    max_rank: int = COMPOSED_ERASURE_MAX_RANK,
    return_mu_a: bool = False,
    peer_ros: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    mu_total64: torch.Tensor | None = None
    mu_a_sum64: torch.Tensor | None = None
    mu_a_weight = 0.0
    parsed_erasure: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []

    for entry in entries:
        v_pr, strength, mode, mu_d_t, vr_t, diag_t = entry[:6]
        mu_a_t = entry[6] if len(entry) >= 7 else None
        s_f = float(strength)
        if not is_additive_mode(mode) or s_f == 0.0:
            continue
        has_er = vr_t is not None and diag_t is not None and vr_t.ndim == 2 and vr_t.shape[0] >= 1
        use_er = s_f > 0.0 and has_er
        dir_src = mu_d_t if (use_er and mu_d_t is not None) else v_pr
        d64 = dir_src.detach().to(device="cpu", dtype=torch.float64).flatten()
        d_norm = float(torch.linalg.vector_norm(d64).item())
        if d_norm <= 1e-12:
            continue
        u64 = d64 / d_norm
        if is_norm_matched_mode(mode):
            term64 = (s_f * NORM_MATCHED_TARGET / d_norm) * d64
        else:
            term64 = s_f * d64
        mu_total64 = term64 if mu_total64 is None else (mu_total64 + term64)

        if use_er:
            vr64 = vr_t.detach().to(device="cpu", dtype=torch.float64)
            diag64 = diag_t.detach().to(device="cpu", dtype=torch.float64).flatten().clamp(0.0, 1.0)
            r_use = min(int(vr64.shape[0]), int(diag64.numel()))
            if r_use >= 1:
                parsed_erasure.append((u64, vr64[:r_use], diag64[:r_use], min(s_f, 1.0)))
            if mu_a_t is not None:
                ma64 = mu_a_t.detach().to(device="cpu", dtype=torch.float64).flatten()
                mu_a_sum64 = (s_f * ma64) if mu_a_sum64 is None else (mu_a_sum64 + s_f * ma64)
                mu_a_weight += s_f

    w_blocks: list[torch.Tensor] = []
    for idx_j, (u_j, vr_j, diag_j, w_eff_j) in enumerate(parsed_erasure):
        if peer_ros and len(parsed_erasure) >= 2:
            peer_targets = [
                u_k
                for idx_k, (u_k, _vr_k, _d_k, _w_k) in enumerate(parsed_erasure)
                if idx_k != idx_j and abs(float(torch.dot(u_j, u_k).item())) < 1.0 - 1e-4
            ]
            if peer_targets:
                ros_res = orthogonalize_erasure_against_holds(vr_j, diag_j, peer_targets)
                if ros_res is None:
                    continue
                vr_j = ros_res[0].to(dtype=torch.float64)
                diag_j = ros_res[1].to(dtype=torch.float64)
        w_blocks.append(torch.sqrt((w_eff_j * diag_j).clamp(0.0, 1.0)).unsqueeze(-1) * vr_j)

    if mu_total64 is None or not w_blocks:
        return None

    hold_pairs: list[tuple[torch.Tensor, float]] = [
        (hv.detach().to(device="cpu", dtype=torch.float64).flatten(), float(ha))
        for hv, ha in holds
        if float(ha) > 0.0
    ]
    if hold_pairs:
        mu_total64 = _apply_holds_to_shift64(mu_total64, hold_pairs, torch.device("cpu"))

    mu_norm = float(torch.linalg.vector_norm(mu_total64).item())
    if not math.isfinite(mu_norm) or mu_norm <= 1e-12:
        return None

    w_mat = torch.cat(w_blocks, dim=0)
    if hold_pairs:
        w_holds = [(hv, min(1.0, ha)) for hv, ha in hold_pairs]
        w_mat = _apply_holds_to_rows64(w_mat, w_holds, torch.device("cpu"))

    _u, s_vals, vh = torch.linalg.svd(w_mat, full_matrices=False)
    if s_vals.numel() == 0 or float(s_vals[0].item()) <= 1e-12:
        return None
    d_vals = (s_vals ** 2).clamp(0.0, 1.0)
    keep = d_vals > 1e-9
    n_pos = int(keep.sum().item())
    r = min(int(max_rank), n_pos)
    if r < 1:
        return None
    mu_d_out = mu_total64.to(dtype=torch.float32).contiguous()
    vr_out = vh[:r].to(dtype=torch.float32).contiguous()
    diag_out = d_vals[:r].to(dtype=torch.float32).contiguous()
    if return_mu_a:
        mu_a_out = (
            (mu_a_sum64 / mu_a_weight).to(dtype=torch.float32).contiguous()
            if (mu_a_sum64 is not None and mu_a_weight > 0.0)
            else None
        )
        return (mu_d_out, vr_out, diag_out, mu_a_out)
    return (mu_d_out, vr_out, diag_out)


def per_token_erasure_profile(
    rows: torch.Tensor,
    mask: torch.Tensor,
    vr: torch.Tensor,
    diag: torch.Tensor,
    *,
    mu_a: torch.Tensor | None = None,
) -> dict[str, list[float] | float]:
    m_bool = mask.flatten().to(device="cpu", dtype=torch.bool)
    h = rows.detach().to(device="cpu", dtype=torch.float64)[m_bool]
    min_rows = 1 if mu_a is not None else 2
    if h.shape[0] < min_rows or vr.ndim != 2 or vr.shape[0] < 1 or diag.numel() < 1:
        n_h = int(h.shape[0])
        return {"erasure_norms": [0.0] * n_h, "erasure_ratios": [0.0] * n_h, "erasure_rms": 0.0}
    if mu_a is not None:
        hc = h - mu_a.detach().to(device="cpu", dtype=torch.float64).flatten().unsqueeze(0)
    else:
        hc = h - h.mean(dim=0, keepdim=True)
    vr64 = vr.detach().to(device="cpu", dtype=torch.float64)
    d64 = diag.detach().to(device="cpu", dtype=torch.float64).flatten().clamp(0.0, 1.0)
    r_use = min(int(vr64.shape[0]), int(d64.numel()))
    e_rows = ((hc @ vr64[:r_use].T) * d64[:r_use].unsqueeze(0)) @ vr64[:r_use]
    norms = torch.linalg.vector_norm(e_rows, dim=-1)
    hc_norms = torch.linalg.vector_norm(hc, dim=-1).clamp(min=1e-12)
    ratios = (norms / hc_norms).clamp(0.0, 1.0)
    rms = float(torch.sqrt(torch.mean(norms ** 2)).item())
    return {
        "erasure_norms": [float(x) for x in norms.tolist()],
        "erasure_ratios": [float(x) for x in ratios.tolist()],
        "erasure_rms": rms,
    }


def erasure_spectral_summary(
    a_spans: Sequence[torch.Tensor],
    b_spans: Sequence[torch.Tensor],
    *,
    mu_d: torch.Tensor | None = None,
    v_1d: torch.Tensor | None = None,
    vr: torch.Tensor | None = None,
    diag: torch.Tensor | None = None,
) -> dict[str, Any]:
    if not a_spans or not b_spans:
        return {}
    a_cat = torch.cat([s.detach().to(device="cpu", dtype=torch.float64) for s in a_spans if s.numel() > 0], dim=0)
    b_cat = torch.cat([s.detach().to(device="cpu", dtype=torch.float64) for s in b_spans if s.numel() > 0], dim=0)
    if a_cat.shape[0] < 2 or b_cat.shape[0] < 1:
        return {}
    mu_a = a_cat.mean(dim=0, keepdim=True)
    shifts = b_cat - mu_a
    mean_sh = shifts.mean(dim=0, keepdim=True)
    denom_sh = float((shifts ** 2).sum(dim=-1).mean().item())
    rank1_share = float((mean_sh ** 2).sum().item() / max(denom_sh, 1e-12))

    ac = a_cat - mu_a
    _u, sv, _vh = torch.linalg.svd(ac, full_matrices=False)
    p_sv = (sv ** 2) / (sv ** 2).sum().clamp(min=1e-12)
    p_nz = p_sv[p_sv > 1e-12]
    eff_rank = float(torch.exp(-(p_nz * torch.log(p_nz)).sum()).item()) if p_nz.numel() > 0 else 0.0

    out: dict[str, Any] = {
        "rank1_energy_share": round(rank1_share, 4),
        "pole_a_effective_rank": round(eff_rank, 2),
        "n_a_tokens": int(a_cat.shape[0]),
        "n_b_tokens": int(b_cat.shape[0]),
    }
    if mu_d is not None and v_1d is not None:
        m64 = mu_d.detach().to(device="cpu", dtype=torch.float64).flatten()
        v64 = v_1d.detach().to(device="cpu", dtype=torch.float64).flatten()
        mn = float(torch.linalg.vector_norm(m64).item())
        vn = float(torch.linalg.vector_norm(v64).item())
        if mn > 1e-12 and vn > 1e-12:
            out["cos_mu_d_v1d"] = round(float(torch.dot(m64, v64).item() / (mn * vn)), 4)
    if vr is not None and diag is not None:
        out["erasure_rank"] = int(vr.shape[0])
        out["diag"] = [round(float(x), 4) for x in diag.detach().cpu().flatten().tolist()]
    return out


def _pool_rows(rows: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return rows.to(dtype=torch.float64).mean(dim=0).to(dtype=torch.float32)
    return rows.to(dtype=torch.float64)[mask].mean(dim=0).to(dtype=torch.float32)


def _encoder_rows_mask(mask: torch.Tensor) -> torch.Tensor:
    out = mask.clone()
    if out.numel() > 0:
        out[0] = False
    return out if bool(out.any()) else mask


def _pair_pooling_masks(enc: Mapping[str, Any], pooling: str) -> tuple[torch.Tensor, torch.Tensor]:
    first_diff, ids_a, ids_b = enc["first_diff"], enc["ids_a"], enc["ids_b"]
    mask_a = _h3_spans.pooling_mask(
        enc["spans_a"], pooling, first_diff=min(first_diff, len(ids_a) - 1), end_diff=enc["end_a"], other_ids=ids_b
    )
    mask_b = _h3_spans.pooling_mask(
        enc["spans_b"], pooling, first_diff=min(first_diff, len(ids_b) - 1), end_diff=enc["end_b"], other_ids=ids_a
    )
    return mask_a, mask_b


def build_span_cache(
    encoded_pairs: Sequence[Mapping[str, Any]],
    refined_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    encoded_holds: Sequence[Mapping[str, Any]] = (),
    refined_holds: Sequence[tuple[torch.Tensor, torch.Tensor]] = (),
    *,
    pooling: str = "diff_span",
    include_encoder: bool = True,
    median_ref_row_norm: float = 164.0,
    median_enc_row_norm_ex0: float = 292.3,
    extra_metadata: Mapping[str, Any] | None = None,
) -> SpanCache:
    """Pack unpooled `fp32` post-refiner span rows and pooled hold/encoder rows into a `SpanCache`."""
    if pooling not in VALID_POOLING:
        raise ValueError(f"pooling must be one of {VALID_POOLING}, got {pooling!r}")
    if len(encoded_pairs) != len(refined_pairs) or len(encoded_holds) != len(refined_holds):
        raise ValueError("build_span_cache: encoded and refined lists differ in length")
    if not encoded_pairs:
        raise ValueError("build_span_cache: no pairs")

    a_spans: list[torch.Tensor] = []
    b_spans: list[torch.Tensor] = []
    lens_a: list[int] = []
    lens_b: list[int] = []
    weights: list[float] = []
    enc_a_rows: list[torch.Tensor] = []
    enc_b_rows: list[torch.Tensor] = []
    pair_spans: list[dict[str, Any]] = []

    for enc, (ref_a, ref_b) in zip(encoded_pairs, refined_pairs):
        item = enc["item"]
        mask_a, mask_b = _pair_pooling_masks(enc, pooling)
        sa = ref_a[mask_a].detach().to(device="cpu", dtype=torch.float32).contiguous()
        sb = ref_b[mask_b].detach().to(device="cpu", dtype=torch.float32).contiguous()
        a_spans.append(sa)
        b_spans.append(sb)
        lens_a.append(int(sa.shape[0]))
        lens_b.append(int(sb.shape[0]))
        weights.append(float(item.weight))
        if include_encoder:
            enc_a_rows.append(_pool_rows(enc["enc_a"], _encoder_rows_mask(mask_a)).cpu().contiguous())
            enc_b_rows.append(_pool_rows(enc["enc_b"], _encoder_rows_mask(mask_b)).cpu().contiguous())
        pair_spans.append(
            {
                "index": int(enc["idx"]),
                "weight": float(item.weight),
                "first_diff": int(enc["first_diff"]),
                "end_a": int(enc["end_a"]),
                "end_b": int(enc["end_b"]),
                "a_span": item.a_span,
                "b_span": item.b_span,
                "len_a": int(len(enc["ids_a"])),
                "len_b": int(len(enc["ids_b"])),
                "pooled_rows_a": int(sa.shape[0]),
                "pooled_rows_b": int(sb.shape[0]),
                "pooled_tokens": int(sb.shape[0]),
            }
        )

    hold_a_list: list[torch.Tensor] = []
    hold_b_list: list[torch.Tensor] = []
    enc_hold_a_list: list[torch.Tensor] = []
    enc_hold_b_list: list[torch.Tensor] = []
    hold_groups: list[str] = []
    hold_weights_list: list[float] = []
    for enc_h, (ref_ha, ref_hb) in zip(encoded_holds, refined_holds):
        mask_ha, mask_hb = _pair_pooling_masks(enc_h, pooling)
        hold_a_list.append(_pool_rows(ref_ha, mask_ha).cpu().contiguous())
        hold_b_list.append(_pool_rows(ref_hb, mask_hb).cpu().contiguous())
        if include_encoder:
            enc_hold_a_list.append(_pool_rows(enc_h["enc_a"], _encoder_rows_mask(mask_ha)).cpu().contiguous())
            enc_hold_b_list.append(_pool_rows(enc_h["enc_b"], _encoder_rows_mask(mask_hb)).cpu().contiguous())
        hold_groups.append(str(enc_h["h_item"].group))
        hold_weights_list.append(float(enc_h["h_item"].weight))

    hold_a_t = (
        torch.stack(hold_a_list, dim=0)
        if hold_a_list
        else torch.zeros((0, POST_REFINER_DIM), dtype=torch.float32)
    )
    hold_b_t = (
        torch.stack(hold_b_list, dim=0)
        if hold_b_list
        else torch.zeros((0, POST_REFINER_DIM), dtype=torch.float32)
    )
    hold_w_t = torch.tensor(hold_weights_list, dtype=torch.float32)

    enc_a_t = torch.stack(enc_a_rows, dim=0) if include_encoder else None
    enc_b_t = torch.stack(enc_b_rows, dim=0) if include_encoder else None
    enc_ha_t = (
        (
            torch.stack(enc_hold_a_list, dim=0)
            if enc_hold_a_list
            else torch.zeros((0, ENCODER_DIM), dtype=torch.float32)
        )
        if include_encoder
        else None
    )
    enc_hb_t = (
        (
            torch.stack(enc_hold_b_list, dim=0)
            if enc_hold_b_list
            else torch.zeros((0, ENCODER_DIM), dtype=torch.float32)
        )
        if include_encoder
        else None
    )

    meta: dict[str, Any] = {
        "format": SPAN_CACHE_FORMAT_VERSION,
        "pooling": pooling,
        "hold_groups": hold_groups,
        "pair_spans": pair_spans,
        "median_ref_row_norm": float(median_ref_row_norm),
        "median_enc_row_norm_ex0": float(median_enc_row_norm_ex0),
    }
    if extra_metadata:
        for k, v in extra_metadata.items():
            if k not in meta:
                meta[str(k)] = v

    return SpanCache(
        span_a=torch.cat(a_spans, dim=0).contiguous(),
        span_b=torch.cat(b_spans, dim=0).contiguous(),
        lens_a=torch.tensor(lens_a, dtype=torch.int32),
        lens_b=torch.tensor(lens_b, dtype=torch.int32),
        pair_weights=torch.tensor(weights, dtype=torch.float32),
        hold_a=hold_a_t.contiguous(),
        hold_b=hold_b_t.contiguous(),
        hold_weights=hold_w_t.contiguous(),
        enc_a_rows=enc_a_t.contiguous() if enc_a_t is not None else None,
        enc_b_rows=enc_b_t.contiguous() if enc_b_t is not None else None,
        enc_hold_a=enc_ha_t.contiguous() if enc_ha_t is not None else None,
        enc_hold_b=enc_hb_t.contiguous() if enc_hb_t is not None else None,
        metadata=meta,
    )


def save_span_cache(path: str | Path, cache: SpanCache, *, overwrite: bool = True) -> Path:
    """Write a `SpanCache` to `<stem>__<family>__spans.safetensors` strictly in `fp32` / `int32`."""
    out_path = Path(path)
    if not overwrite and out_path.exists():
        raise ValueError(f"span cache file {out_path.name!r} already exists at {out_path}")
    if cache.span_a.dtype != torch.float32 or cache.span_b.dtype != torch.float32:
        raise ValueError("SpanCache span_a and span_b must be float32 (do not store bf16)")
    if cache.lens_a.dtype != torch.int32 or cache.lens_b.dtype != torch.int32:
        raise ValueError("SpanCache lens_a and lens_b must be int32")

    tensors: dict[str, torch.Tensor] = {
        "span_a": cache.span_a.detach().cpu().to(dtype=torch.float32).contiguous(),
        "span_b": cache.span_b.detach().cpu().to(dtype=torch.float32).contiguous(),
        "lens_a": cache.lens_a.detach().cpu().to(dtype=torch.int32).contiguous(),
        "lens_b": cache.lens_b.detach().cpu().to(dtype=torch.int32).contiguous(),
        "pair_weights": cache.pair_weights.detach().cpu().to(dtype=torch.float32).contiguous(),
        "hold_a": cache.hold_a.detach().cpu().to(dtype=torch.float32).contiguous(),
        "hold_b": cache.hold_b.detach().cpu().to(dtype=torch.float32).contiguous(),
        "hold_weights": cache.hold_weights.detach().cpu().to(dtype=torch.float32).contiguous(),
    }
    if cache.enc_a_rows is not None and cache.enc_b_rows is not None:
        tensors["enc_a_rows"] = cache.enc_a_rows.detach().cpu().to(dtype=torch.float32).contiguous()
        tensors["enc_b_rows"] = cache.enc_b_rows.detach().cpu().to(dtype=torch.float32).contiguous()
    if cache.enc_hold_a is not None and cache.enc_hold_b is not None:
        tensors["enc_hold_a"] = cache.enc_hold_a.detach().cpu().to(dtype=torch.float32).contiguous()
        tensors["enc_hold_b"] = cache.enc_hold_b.detach().cpu().to(dtype=torch.float32).contiguous()

    meta_dict = dict(cache.metadata)
    meta_dict.setdefault("format", SPAN_CACHE_FORMAT_VERSION)
    encoded_meta = {str(k): json.dumps(v, sort_keys=True) for k, v in meta_dict.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(save_safetensors_bytes(tensors, metadata=encoded_meta))
    _chmod_group_readable(out_path)
    return out_path


def load_span_cache(path: str | Path) -> SpanCache:
    """Load and validate an `fp32` `__spans.safetensors` `SpanCache` from disk."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"span cache file not found: {p}")
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(p), framework="pt", device="cpu") as sf:
        raw_meta = dict(sf.metadata() or {})
        for k in sf.keys():
            tensors[k] = sf.get_tensor(k).clone()
    meta: dict[str, Any] = {}
    for k, v in raw_meta.items():
        try:
            meta[k] = json.loads(v)
        except Exception:
            meta[k] = v
    if meta.get("format") != SPAN_CACHE_FORMAT_VERSION:
        raise ValueError(
            f"unsupported span cache format {meta.get('format')!r}; expected {SPAN_CACHE_FORMAT_VERSION!r}"
        )
    for req_t in ("span_a", "span_b", "lens_a", "lens_b", "pair_weights", "hold_a", "hold_b", "hold_weights"):
        if req_t not in tensors:
            raise ValueError(f"span cache {p.name} missing tensor {req_t!r}")
    if tensors["span_a"].dtype != torch.float32 or tensors["span_b"].dtype != torch.float32:
        raise ValueError(f"span cache {p.name} span_a/span_b must be float32")
    if tensors["lens_a"].dtype != torch.int32 or tensors["lens_b"].dtype != torch.int32:
        raise ValueError(f"span cache {p.name} lens_a/lens_b must be int32")
    n_pairs = int(tensors["pair_weights"].numel())
    if int(tensors["lens_a"].numel()) != n_pairs or int(tensors["lens_b"].numel()) != n_pairs:
        raise ValueError(f"span cache {p.name} lens_a/lens_b length mismatch vs pair_weights ({n_pairs})")
    if int(tensors["lens_a"].sum().item()) != int(tensors["span_a"].shape[0]):
        raise ValueError(f"span cache {p.name} sum(lens_a) != span_a.shape[0]")
    if int(tensors["lens_b"].sum().item()) != int(tensors["span_b"].shape[0]):
        raise ValueError(f"span cache {p.name} sum(lens_b) != span_b.shape[0]")
    for req_m in ("hold_groups", "pair_spans", "median_ref_row_norm", "median_enc_row_norm_ex0"):
        if req_m not in meta:
            raise ValueError(f"span cache {p.name} missing metadata key {req_m!r}")
    return SpanCache(
        span_a=tensors["span_a"],
        span_b=tensors["span_b"],
        lens_a=tensors["lens_a"],
        lens_b=tensors["lens_b"],
        pair_weights=tensors["pair_weights"],
        hold_a=tensors["hold_a"],
        hold_b=tensors["hold_b"],
        hold_weights=tensors["hold_weights"],
        enc_a_rows=tensors.get("enc_a_rows"),
        enc_b_rows=tensors.get("enc_b_rows"),
        enc_hold_a=tensors.get("enc_hold_a"),
        enc_hold_b=tensors.get("enc_hold_b"),
        metadata=meta,
    )


def pool_from_span_cache(
    cache: SpanCache,
    *,
    unit_mean: bool = False,
    apply_holds: bool = True,
    include_encoder: bool = True,
    min_erasure_pairs: int = 1,
) -> PooledPairSet:
    """Re-extract a `PooledPairSet` from an unpooled `fp32` `SpanCache` in `< 50 ms` on CPU,
    matching `pool_pair_set` to `< 1e-6` (`0.0` IEEE-754 identity on `fp32` inputs)."""
    a_spans, b_spans = cache.split_spans()
    a_rows = [_pool_rows(sa) for sa in a_spans]
    b_rows = [_pool_rows(sb) for sb in b_spans]
    a_mat = torch.stack(a_rows, dim=0)
    b_mat = torch.stack(b_rows, dim=0)
    w_vec = cache.pair_weights.to(dtype=torch.float32)
    weights = [float(x) for x in w_vec.tolist()]
    raw_vec = mean_difference(a_mat, b_mat, w_vec, unit_mean=unit_mean)

    n_holds = int(cache.hold_a.shape[0]) if apply_holds else 0
    basis: torch.Tensor | None = None
    enc_basis: torch.Tensor | None = None
    if apply_holds and n_holds > 0:
        groups = [str(g) for g in cache.metadata.get("hold_groups", ())]
        hw = cache.hold_weights.to(dtype=torch.float32)
        basis = hold_basis(cache.hold_a, cache.hold_b, groups, hw)
        if include_encoder and cache.enc_hold_a is not None and cache.enc_hold_b is not None and cache.enc_hold_a.shape[0] > 0:
            enc_basis = hold_basis(cache.enc_hold_a, cache.enc_hold_b, groups, hw)

    vector = project_out(raw_vec, basis) if basis is not None else raw_vec

    enc_raw_vec: torch.Tensor | None = None
    if include_encoder and cache.enc_a_rows is not None and cache.enc_b_rows is not None:
        enc_raw_vec = mean_difference(cache.enc_a_rows, cache.enc_b_rows, w_vec, unit_mean=unit_mean)
        if enc_basis is not None:
            enc_raw_vec = project_out(enc_raw_vec, enc_basis)

    if len(a_spans) >= max(1, int(min_erasure_pairs)):
        erasure_built = build_pole_a_erasure(a_spans, b_spans, weights, hold_basis=basis)
        erasure_mu_d, erasure_vr, erasure_diag = (
            erasure_built if erasure_built is not None else (None, None, None)
        )
    else:
        erasure_mu_d, erasure_vr, erasure_diag = (None, None, None)

    return PooledPairSet(
        a_mat=a_mat,
        b_mat=b_mat,
        w_vec=w_vec,
        raw_vec=raw_vec,
        vector=vector,
        hold_basis=basis,
        pair_spans=list(cache.metadata.get("pair_spans", ())),
        enc_raw_vec=enc_raw_vec,
        n_holds=n_holds,
        erasure_mu_d=erasure_mu_d,
        erasure_vr=erasure_vr,
        erasure_diag=erasure_diag,
    )


def pool_pair_set(
    encoded_pairs: Sequence[Mapping[str, Any]],
    refined_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    encoded_holds: Sequence[Mapping[str, Any]] = (),
    refined_holds: Sequence[tuple[torch.Tensor, torch.Tensor]] = (),
    *,
    pooling: str,
    unit_mean: bool = False,
    include_encoder: bool = True,
    min_erasure_pairs: int = 1,
) -> PooledPairSet:
    """Pool encoded + refined pairs and holds into a `PooledPairSet` (`unit_mean=False` default)."""
    cache = build_span_cache(
        encoded_pairs,
        refined_pairs,
        encoded_holds,
        refined_holds,
        pooling=pooling,
        include_encoder=include_encoder,
    )
    return pool_from_span_cache(
        cache,
        unit_mean=unit_mean,
        apply_holds=bool(encoded_holds),
        include_encoder=include_encoder,
        min_erasure_pairs=min_erasure_pairs,
    )


def encoder_point(enc_raw_vec: torch.Tensor) -> tuple[torch.Tensor | None, float]:
    norm = float(torch.linalg.vector_norm(enc_raw_vec.to(dtype=torch.float64)).item())
    if norm <= 1e-12:
        return None, norm
    return (enc_raw_vec.to(dtype=torch.float64) / norm).to(dtype=torch.float32), norm


def pairs_record(
    pair_set: PairSet,
    *,
    path: str,
    pair_spans: list[dict[str, Any]],
    n_holds: int,
    inline: bool,
    words_spec: str | None = None,
    inline_holds: bool = True,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": path,
        "sha256": pair_set.sha256,
        "n_pairs": len(pair_set.pairs),
        "n_holds": int(n_holds),
        "a_side": pair_set.a_side,
        "b_side": pair_set.b_side,
        "differing_sections": pair_set_differing_sections(pair_set.pairs, frame=pair_set.frame),
        "pair_spans": pair_spans,
    }
    if inline:
        if isinstance(words_spec, str) and words_spec.strip():
            out["words_spec"] = words_spec.strip()
        out["pairs_ab"] = [{"a": p.a, "b": p.b, "weight": float(p.weight)} for p in pair_set.pairs]
        if inline_holds and pair_set.holds:
            out["holds_ab"] = [
                {"group": h.group, "a": h.a, "b": h.b, "weight": float(h.weight)} for h in pair_set.holds
            ]
    return out


def merge_toml_metadata_into_vector(
    metadata: dict[str, Any],
    pair_set: PairSet | None,
) -> dict[str, Any]:
    """Propagate `pair_set.routing` and `pair_set.provenance` into `.safetensors` `metadata`."""
    if pair_set is None:
        return metadata
    if pair_set.routing is not None and "routing" not in metadata:
        metadata["routing"] = pair_set.routing.to_dict()
    if pair_set.provenance:
        cur_prov = dict(metadata.get("provenance") or {})
        for k, v in pair_set.provenance.items():
            if k == "created_utc" and "created_utc" in cur_prov:
                continue
            cur_prov.setdefault(k, v)
        metadata["provenance"] = cur_prov
    return metadata


def mean_difference(
    a: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    unit_mean: bool = False,
) -> torch.Tensor:
    a64, b64, w64 = _check_pair_tensors(a, b, weights)
    diffs = b64 - a64
    denom = w64.abs().sum()
    if not unit_mean:
        vec = (w64[:, None] * diffs).sum(dim=0) / denom
        return vec.to(dtype=torch.float32)

    norms = torch.linalg.vector_norm(diffs, dim=-1)
    if bool((norms <= 1e-12).any()):
        raise ValueError("cannot compute unit_mean: at least one pair difference has zero norm")
    unit_diffs = diffs / norms[:, None]
    mean_dir = (w64[:, None] * unit_diffs).sum(dim=0) / denom
    mean_norm = (w64.abs() * norms).sum() / denom
    vec = mean_dir * mean_norm
    return vec.to(dtype=torch.float32)


def flip_null(
    a: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    a64, b64, w64 = _check_pair_tensors(a, b, weights)
    n, _d = a64.shape
    if n < 2:
        raise ValueError(f"flip_null requires at least 2 pairs, got n={n}")

    diffs = b64 - a64
    denom = w64.abs().sum()
    true_vec = (w64[:, None] * diffs).sum(dim=0) / denom
    target_norm = torch.linalg.vector_norm(true_vec)
    if float(target_norm) == 0.0:
        raise ValueError("cannot construct flip_null: mean_difference has zero norm")

    gen_dev = generator.device if generator is not None else torch.device("cpu")
    perm = torch.randperm(n, generator=generator, device=gen_dev).to(a64.device)
    signs = torch.ones(n, dtype=torch.float64, device=a64.device)
    signs[perm[: n // 2]] = -1.0
    abs_w = w64.abs()
    signed_diffs = w64[:, None] * diffs
    pos_mask = signs > 0
    neg_mask = signs < 0
    w_pos = abs_w[pos_mask].sum()
    w_neg = abs_w[neg_mask].sum()
    if float(w_pos.item()) <= 1e-12 or float(w_neg.item()) <= 1e-12:
        raise ValueError("flip_null encountered a half with zero weight sum")
    half_pos = signed_diffs[pos_mask].sum(dim=0) / w_pos
    half_neg = signed_diffs[neg_mask].sum(dim=0) / w_neg
    flipped_vec = 0.5 * (half_pos - half_neg)

    flipped_norm = torch.linalg.vector_norm(flipped_vec)
    if float(flipped_norm) == 0.0:
        raise ValueError("flip_null produced a zero vector before rescaling")
    rescaled = flipped_vec * (target_norm / flipped_norm)
    return rescaled.to(dtype=torch.float32)


def random_null(
    norm: float,
    d: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"norm must be finite and positive, got {norm!r}")
    if d < 1:
        raise ValueError(f"dimension d must be >= 1, got {d}")
    gen_dev = generator.device if generator is not None else torch.device("cpu")
    raw = torch.randn(d, dtype=torch.float64, generator=generator, device=gen_dev)
    raw_norm = torch.linalg.vector_norm(raw)
    if float(raw_norm) == 0.0:
        raise ValueError("random_null drew a zero vector")
    out = raw * (float(norm) / raw_norm)
    return out.to(dtype=torch.float32)


def hold_basis(
    hold_a: torch.Tensor,
    hold_b: torch.Tensor,
    groups: Sequence[str],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    a64, b64, w64 = _check_pair_tensors(hold_a, hold_b, weights)
    m, _d = a64.shape
    if len(groups) != m:
        raise ValueError(f"groups length ({len(groups)}) must match hold_a rows ({m})")

    ordered_groups: list[str] = []
    group_indices: dict[str, list[int]] = {}
    for idx, g in enumerate(groups):
        if not isinstance(g, str) or not g.strip():
            raise ValueError(f"groups[{idx}] must be a non-empty string")
        if g not in group_indices:
            ordered_groups.append(g)
            group_indices[g] = []
        group_indices[g].append(idx)

    basis_rows: list[torch.Tensor] = []
    for g in ordered_groups:
        idxs = group_indices[g]
        ga = a64[idxs]
        gb = b64[idxs]
        gw = w64[idxs]
        vec = (gw[:, None] * (gb - ga)).sum(dim=0) / gw.abs().sum()
        raw_norm = float(torch.linalg.vector_norm(vec).item())
        if raw_norm <= 1e-12:
            raise ValueError(
                f"hold group {g!r} direction is zero or colinear with earlier hold groups"
            )
        for _pass in range(2):
            for u in basis_rows:
                vec = vec - torch.dot(vec, u) * u
        norm = torch.linalg.vector_norm(vec)
        if float(norm) <= max(1e-12, 1e-5 * raw_norm):
            raise ValueError(
                f"hold group {g!r} direction is zero or colinear with earlier hold groups"
            )
        basis_rows.append(vec / norm)

    return torch.stack(basis_rows, dim=0).to(dtype=torch.float32)


def project_out(v: torch.Tensor, basis: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    if v.ndim != 1:
        raise ValueError(f"v must be a 1D tensor [d], got shape {tuple(v.shape)}")
    d = v.shape[0]
    if basis.ndim != 2 or basis.shape[1] != d:
        raise ValueError(f"basis must have shape [k, {d}], got {tuple(basis.shape)}")
    alpha_f = float(alpha)
    if not math.isfinite(alpha_f) or not (0.0 <= alpha_f <= 2.0):
        raise ValueError(f"alpha must be finite and in [0.0, 2.0], got {alpha!r}")

    v64 = v.detach().to(dtype=torch.float64)
    b64 = basis.detach().to(device=v64.device, dtype=torch.float64)
    if not bool(torch.isfinite(v64).all()) or not bool(torch.isfinite(b64).all()):
        raise ValueError("v and basis must contain only finite values")

    if basis.shape[0] == 0 or alpha_f == 0.0:
        return v64.to(dtype=torch.float32)

    ortho_rows: list[torch.Tensor] = []
    for row in b64:
        u = row.clone()
        for _pass in range(2):
            for prev in ortho_rows:
                u = u - torch.dot(u, prev) * prev
        nrm = torch.linalg.vector_norm(u)
        if float(nrm) > 1e-12:
            ortho_rows.append(u / nrm)

    for u in ortho_rows:
        v64 = v64 - alpha_f * torch.dot(v64, u) * u
    return v64.to(dtype=torch.float32)


def _unit_rows64(diffs: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if diffs.ndim != 2 or diffs.shape[0] < 2:
        raise ValueError(f"diffs must be 2D [n, d] with n >= 2, got shape {tuple(diffs.shape)}")
    d64 = diffs.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(d64).all()):
        raise ValueError("diffs contains non-finite values")
    if weights is not None:
        w_t = (
            weights
            if isinstance(weights, torch.Tensor)
            else torch.as_tensor(weights, dtype=torch.float64, device=d64.device)
        )
        if w_t.ndim != 1 or w_t.shape[0] != d64.shape[0]:
            raise ValueError(f"weights must have shape [{d64.shape[0]}], got {tuple(w_t.shape)}")
        w64 = w_t.detach().to(device=d64.device, dtype=torch.float64)
        if not bool(torch.isfinite(w64).all()) or bool((w64 == 0).any()):
            raise ValueError("weights must be finite and non-zero")
        d64 = d64 * torch.sign(w64)[:, None]
    norms = torch.linalg.vector_norm(d64, dim=1, keepdim=True)
    if bool((norms == 0).any()):
        raise ValueError("diffs contains a zero row; pairwise cosine is undefined")
    return d64 / norms


def mean_pairwise_cosine(
    diffs: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> float:
    unit = _unit_rows64(diffs, weights=weights)
    n = unit.shape[0]
    gram = unit @ unit.T
    tri = torch.triu_indices(n, n, offset=1, device=gram.device)
    return float(gram[tri[0], tri[1]].mean().item())


def per_pair_cosine(
    diffs: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> list[float]:
    unit = _unit_rows64(diffs, weights=weights)
    n = unit.shape[0]
    gram = unit @ unit.T
    out: list[float] = []
    for i in range(n):
        row_sum = float(gram[i].sum().item()) - float(gram[i, i].item())
        out.append(row_sum / float(n - 1))
    return out


def split_half_cosine(
    diffs: torch.Tensor,
    generator: torch.Generator | None = None,
    n_splits: int = 64,
    weights: torch.Tensor | None = None,
) -> float:
    if diffs.ndim != 2 or diffs.shape[0] < 2:
        raise ValueError(f"split_half_cosine requires diffs [n, d] with n >= 2, got {tuple(diffs.shape)}")
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    d64 = diffs.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(d64).all()):
        raise ValueError("diffs contains non-finite values")
    n = d64.shape[0]
    if weights is None:
        w64 = torch.ones(n, dtype=torch.float64, device=d64.device)
    else:
        w_t = (
            weights
            if isinstance(weights, torch.Tensor)
            else torch.as_tensor(weights, dtype=torch.float64, device=d64.device)
        )
        if w_t.ndim != 1 or w_t.shape[0] != n:
            raise ValueError(f"weights must have shape [{n}], got {tuple(w_t.shape)}")
        w64 = w_t.detach().to(device=d64.device, dtype=torch.float64)
        if not bool(torch.isfinite(w64).all()) or bool((w64 == 0).any()):
            raise ValueError("weights must be finite and non-zero")

    weighted_diffs = w64[:, None] * d64
    half = n // 2
    gen_dev = generator.device if generator is not None else torch.device("cpu")
    cosines: list[float] = []
    for _ in range(n_splits):
        perm = torch.randperm(n, generator=generator, device=gen_dev).to(d64.device)
        idx1, idx2 = perm[:half], perm[half:]
        h1 = weighted_diffs[idx1].sum(dim=0) / w64[idx1].abs().sum()
        h2 = weighted_diffs[idx2].sum(dim=0) / w64[idx2].abs().sum()
        n1 = torch.linalg.vector_norm(h1)
        n2 = torch.linalg.vector_norm(h2)
        if float(n1) == 0.0 or float(n2) == 0.0:
            raise ValueError("split_half_cosine encountered a half with zero mean norm")
        cosines.append(float(torch.dot(h1, h2).item() / (float(n1.item()) * float(n2.item()))))
    return float(sum(cosines) / len(cosines))


def quality_report(
    a: torch.Tensor,
    b: torch.Tensor,
    weights: Sequence[float] | torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    n_splits: int = 64,
) -> dict[str, Any]:
    a64, b64, w64 = _check_pair_tensors(a, b, weights)
    diffs = b64 - a64
    if diffs.shape[0] == 1:
        norm0 = float(torch.linalg.vector_norm(diffs[0]).item())
        if norm0 == 0.0:
            raise ValueError("diffs contains a zero row; pairwise cosine is undefined")
        return {
            "mean_pairwise_cosine": None,
            "split_half_cosine": None,
            "per_pair_cosine": [],
        }
    return {
        "mean_pairwise_cosine": mean_pairwise_cosine(diffs, weights=w64),
        "split_half_cosine": split_half_cosine(
            diffs, generator=generator, n_splits=n_splits, weights=w64
        ),
        "per_pair_cosine": per_pair_cosine(diffs, weights=w64),
    }


def _tensor_bf16_bytes(t: torch.Tensor) -> memoryview:
    bf16 = t.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return bf16.view(torch.uint8).numpy().data


def _tensor_bf16_hash12(t: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_bf16_bytes(t)).hexdigest()[:12]


def family_from_tensor_hashes(hashes: tuple[str, str, str]) -> str:
    if hashes in KNOWN_FAMILIES:
        return KNOWN_FAMILIES[hashes]
    combined = hashlib.sha256(":".join(hashes).encode("utf-8")).hexdigest()[:12]
    return f"custom:{combined}"


_FAMILY_PATH_CACHE: dict[tuple[str, int, int], str] = {}
_DIGEST_PATH_CACHE: dict[tuple[str, int, int, tuple[str, ...]], str] = {}


def _file_stat_key(path: Path) -> tuple[str, int, int]:
    resolved = path.resolve()
    st = resolved.stat()
    return (str(resolved), int(st.st_mtime_ns), int(st.st_size))


def detect_family(source: str | Path | Mapping[str, torch.Tensor] | Any) -> str:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"checkpoint file not found: {path}")
        stat_key = _file_stat_key(path)
        cached = _FAMILY_PATH_CACHE.get(stat_key)
        if cached is not None:
            return cached
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            hashes: list[str] = []
            for k in FAMILY_TENSOR_KEYS:
                matched = next(
                    (
                        cand
                        for cand in (k, f"diffusion_model.{k}", f"model.diffusion_model.{k}")
                        if cand in keys
                    ),
                    None,
                )
                if matched is None:
                    raise ValueError(
                        f"checkpoint {path.name} is missing H3 text-path tensor {k!r}; "
                        "not a supported MiniMax H3 diffusion model"
                    )
                hashes.append(_tensor_bf16_hash12(f.get_tensor(matched)))
        fam = family_from_tensor_hashes((hashes[0], hashes[1], hashes[2]))
        _FAMILY_PATH_CACHE[stat_key] = fam
        return fam

    if isinstance(source, Mapping):
        hashes = []
        for k in FAMILY_TENSOR_KEYS:
            matched_t = None
            for cand in (k, f"diffusion_model.{k}", f"model.diffusion_model.{k}"):
                if cand in source and isinstance(source[cand], torch.Tensor):
                    matched_t = source[cand]
                    break
            if matched_t is None:
                raise ValueError(
                    f"state_dict is missing H3 text-path tensor {k!r}; "
                    "not a supported MiniMax H3 diffusion model"
                )
            hashes.append(_tensor_bf16_hash12(matched_t))
        return family_from_tensor_hashes((hashes[0], hashes[1], hashes[2]))

    init = getattr(source, "cached_patcher_init", None)
    if isinstance(init, tuple) and len(init) >= 2 and isinstance(init[1], tuple) and len(init[1]) >= 1:
        cand_path = init[1][0]
        if isinstance(cand_path, (str, Path)) and Path(cand_path).is_file():
            return detect_family(Path(cand_path))

    backup = getattr(source, "backup", None) or {}
    inner_model = getattr(source, "model", source)
    dm = getattr(inner_model, "diffusion_model", inner_model)
    hashes = []
    for k in FAMILY_TENSOR_KEYS:
        bk_key = f"diffusion_model.{k}"
        t: torch.Tensor | None = None
        if isinstance(backup, Mapping) and bk_key in backup:
            entry = backup[bk_key]
            t = getattr(entry, "weight", entry) if not isinstance(entry, torch.Tensor) else entry
        if t is None:
            obj = dm
            for part in k.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if isinstance(obj, torch.Tensor):
                t = obj
        if t is None or not isinstance(t, torch.Tensor):
            raise ValueError(
                f"model ({type(source).__name__}) has no checkpoint path and is missing H3 "
                f"text-path tensor {k!r}; cannot determine H3 text-path family"
            )
        hashes.append(_tensor_bf16_hash12(t))
    return family_from_tensor_hashes((hashes[0], hashes[1], hashes[2]))


def safetensors_recipe_digest(
    path: str | Path,
    *,
    extra_tensor_keys: Sequence[str] = (),
) -> str:
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"model file not found: {p}")
    stat_key = (*_file_stat_key(p), tuple(extra_tensor_keys))
    cached = _DIGEST_PATH_CACHE.get(stat_key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with p.open("rb") as fp:
        raw_len = fp.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{p}: invalid safetensors file (shorter than 8 bytes)")
        (header_len,) = struct.unpack("<Q", raw_len)
        header_bytes = fp.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"{p}: truncated safetensors header")
        h.update(raw_len)
        h.update(header_bytes)

    if extra_tensor_keys:
        with safe_open(str(p), framework="pt", device="cpu") as sf:
            keys = set(sf.keys())
            for k in extra_tensor_keys:
                matched = next(
                    (
                        cand
                        for cand in (k, f"diffusion_model.{k}", f"model.diffusion_model.{k}")
                        if cand in keys
                    ),
                    None,
                )
                if matched is None:
                    raise ValueError(
                        f"{p.name}: missing required tensor key {k!r} for recipe digest"
                    )
                h.update(k.encode("utf-8"))
                h.update(_tensor_bf16_bytes(sf.get_tensor(matched)))
    digest = h.hexdigest()
    _DIGEST_PATH_CACHE[stat_key] = digest
    return digest


def _validate_metadata_dict(metadata: Mapping[str, Any]) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("control vector metadata must be a mapping")
    for req in REQUIRED_METADATA_KEYS:
        if req not in metadata:
            raise ValueError(f"control vector metadata is missing required key {req!r}")

    fmt = metadata["format"]
    if fmt != FORMAT_VERSION:
        raise ValueError(
            f"unsupported control vector format {fmt!r}; expected {FORMAT_VERSION!r}"
        )

    name = metadata["name"]
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError(f"control vector metadata 'name' must match [a-z0-9_]+, got {name!r}")

    quality = metadata["quality"]
    if not isinstance(quality, str) or not quality.strip():
        raise ValueError("control vector metadata 'quality' must be a non-empty string")

    family = metadata["family"]
    if not isinstance(family, str) or (
        family not in ("fl2va", "ref2va", "fasth3", "h3_image") and not CUSTOM_FAMILY_RE.match(family)
    ):
        raise ValueError(
            "control vector metadata 'family' must be 'fl2va', 'ref2va', 'fasth3', 'h3_image', "
            f"or 'custom:<12 hex chars>', got {family!r}"
        )

    pairs = metadata["pairs"]
    if not isinstance(pairs, dict) or not {"path", "sha256", "n_pairs", "n_holds"}.issubset(pairs.keys()):
        raise ValueError(
            "control vector metadata 'pairs' must be a dict with keys {'path', 'sha256', 'n_pairs', 'n_holds'}"
        )
    if "differing_sections" in pairs:
        dsecs = pairs["differing_sections"]
        if not isinstance(dsecs, list) or not all(isinstance(s, str) and s.strip() for s in dsecs):
            raise ValueError(
                f"control vector metadata 'pairs.differing_sections' must be a list of non-empty strings, got {dsecs!r}"
            )

    recipe = metadata["recipe"]
    if (
        not isinstance(recipe, dict)
        or not {"checkpoint", "loras", "encoder"}.issubset(recipe.keys())
        or not isinstance(recipe["checkpoint"], dict)
        or not isinstance(recipe["loras"], list)
        or not isinstance(recipe["encoder"], dict)
    ):
        raise ValueError(
            "control vector metadata 'recipe' must be a dict with 'checkpoint' (dict), 'loras' (list), and 'encoder' (dict)"
        )

    measured = metadata["measured_under"]
    if (
        not isinstance(measured, dict)
        or not {"mode", "frame", "prefix", "pooling"}.issubset(measured.keys())
    ):
        raise ValueError(
            "control vector metadata 'measured_under' must be a dict with keys {'mode', 'frame', 'prefix', 'pooling'}"
        )
    if measured["mode"] not in VALID_MODES:
        raise ValueError(
            f"measured_under.mode must be one of {VALID_MODES}, got {measured['mode']!r}"
        )
    if measured["frame"] not in VALID_FRAMES:
        raise ValueError(
            f"measured_under.frame must be one of {VALID_FRAMES}, got {measured['frame']!r}"
        )
    if measured["pooling"] not in VALID_ALL_POOLING:
        raise ValueError(
            f"measured_under.pooling must be one of {VALID_ALL_POOLING}, got {measured['pooling']!r}"
        )

    points = metadata["points"]
    if not isinstance(points, dict) or not points:
        raise ValueError("control vector metadata 'points' must be a non-empty dict")
    for pt_key, pt_val in points.items():
        if not isinstance(pt_val, dict) or "unit" not in pt_val or "median_row_norm" not in pt_val:
            raise ValueError(
                f"points[{pt_key!r}] must be a dict with at least 'unit' and 'median_row_norm'"
            )
        if pt_key == "post_refiner":
            if "raw_norm" not in pt_val or pt_val["unit"] != "mean_pair_difference":
                raise ValueError(
                    "points['post_refiner'] must be a dict with unit='mean_pair_difference', 'raw_norm', and 'median_row_norm'"
                )
            num_keys: tuple[str, ...] = ("raw_norm", "median_row_norm")
        else:
            if pt_val["unit"] not in ("median_row_norm", "median_tag1_row_norm"):
                raise ValueError(
                    f"points[{pt_key!r}]['unit'] must be 'median_row_norm', got {pt_val['unit']!r}"
                )
            if "row0_excluded" in pt_val and not isinstance(pt_val["row0_excluded"], bool):
                raise ValueError(f"points[{pt_key!r}]['row0_excluded'] must be a bool")
            num_keys = ("raw_norm", "median_row_norm") if "raw_norm" in pt_val else ("median_row_norm",)
        for nk in num_keys:
            nv = pt_val[nk]
            if (
                isinstance(nv, bool)
                or not isinstance(nv, (int, float))
                or not math.isfinite(float(nv))
                or float(nv) <= 0.0
            ):
                raise ValueError(
                    f"points[{pt_key!r}][{nk!r}] must be a positive finite float, got {nv!r}"
                )

    qrep = metadata["quality_report"]
    if (
        not isinstance(qrep, dict)
        or not {"mean_pairwise_cosine", "split_half_cosine", "per_pair_cosine"}.issubset(qrep.keys())
    ):
        raise ValueError(
            "control vector metadata 'quality_report' must be a dict with keys "
            "{'mean_pairwise_cosine', 'split_half_cosine', 'per_pair_cosine'}"
        )
    n_pairs_q = int((metadata.get("pairs") or {}).get("n_pairs", 2))
    for k_float in ("mean_pairwise_cosine", "split_half_cosine"):
        v_f = qrep[k_float]
        if v_f is None and n_pairs_q < 2:
            continue
        if isinstance(v_f, bool) or not isinstance(v_f, (int, float)) or not math.isfinite(float(v_f)):
            raise ValueError(
                f"quality_report[{k_float!r}] must be a finite float when n_pairs >= 2, got {v_f!r}"
            )
    ppc = qrep["per_pair_cosine"]
    if (
        not isinstance(ppc, list)
        or (not ppc and n_pairs_q >= 2)
        or not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in ppc)
    ):
        raise ValueError(
            "quality_report['per_pair_cosine'] must be a list of finite floats (non-empty when n_pairs >= 2)"
        )

    prov = metadata["provenance"]
    has_build_prov = isinstance(prov, dict) and {"core_revision", "pack_commit", "created_utc", "made_by"}.issubset(
        prov.keys()
    )
    has_canonical_prov = isinstance(prov, dict) and {"author_kind", "created_utc"}.issubset(prov.keys()) and set(
        prov.keys()
    ).issubset({*CANONICAL_PROVENANCE_KEYS, "core_revision", "pack_commit", "made_by"})
    if not (has_build_prov or has_canonical_prov):
        raise ValueError(
            "control vector metadata 'provenance' must be a dict with keys "
            "{'core_revision', 'pack_commit', 'created_utc', 'made_by'} (or canonical HeylookProvenance keys)"
        )

    if "routing" in metadata and metadata["routing"] is not None:
        parse_routing_table(metadata["routing"], source="metadata['routing']")


def _validate_vector_payload(
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> None:
    _validate_metadata_dict(metadata)
    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("control vector tensors dict is empty")

    has_known_vector = False
    for key, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"tensor {key!r} is not a torch.Tensor")
        if t.dtype != torch.float32:
            raise ValueError(f"tensor {key!r} must have dtype float32, got {t.dtype}")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"tensor {key!r} contains non-finite values")

        if key == "vector.post_refiner":
            has_known_vector = True
            if tuple(t.shape) != (POST_REFINER_DIM,):
                raise ValueError(
                    f"tensor 'vector.post_refiner' must have shape ({POST_REFINER_DIM},), got {tuple(t.shape)}"
                )
            if float(torch.linalg.vector_norm(t).item()) == 0.0:
                raise ValueError("tensor 'vector.post_refiner' has zero norm")
        elif key == "holds.post_refiner":
            if t.ndim != 2 or t.shape[1] != POST_REFINER_DIM or t.shape[0] < 1:
                raise ValueError(
                    f"tensor 'holds.post_refiner' must have shape [k, {POST_REFINER_DIM}], got {tuple(t.shape)}"
                )
            k_rows = t.shape[0]
            b64 = t.detach().to(dtype=torch.float64)
            gram = b64 @ b64.T
            eye = torch.eye(k_rows, dtype=torch.float64, device=b64.device)
            if not torch.allclose(gram, eye, atol=1e-4):
                raise ValueError("tensor 'holds.post_refiner' rows are not orthonormal")
        elif key == "vector.encoder" or LLM_VECTOR_KEY_RE.match(key):
            has_known_vector = True
            if tuple(t.shape) != (ENCODER_DIM,):
                raise ValueError(
                    f"tensor {key!r} must have shape ({ENCODER_DIM},), got {tuple(t.shape)}"
                )
            if float(torch.linalg.vector_norm(t).item()) == 0.0:
                raise ValueError(f"tensor {key!r} has zero norm")
        elif key in ("erasure.mu_d", "erasure.mu_a"):
            if tuple(t.shape) != (POST_REFINER_DIM,):
                raise ValueError(
                    f"tensor {key!r} must have shape ({POST_REFINER_DIM},), got {tuple(t.shape)}"
                )
            if key == "erasure.mu_d" and float(torch.linalg.vector_norm(t).item()) == 0.0:
                raise ValueError("tensor 'erasure.mu_d' has zero norm")
        elif key == "erasure.vr":
            if t.ndim != 2 or t.shape[1] != POST_REFINER_DIM or t.shape[0] < 1:
                raise ValueError(
                    f"tensor 'erasure.vr' must have shape [r, {POST_REFINER_DIM}], got {tuple(t.shape)}"
                )
            r_rows = t.shape[0]
            vr64 = t.detach().to(dtype=torch.float64)
            gram_vr = vr64 @ vr64.T
            eye_vr = torch.eye(r_rows, dtype=torch.float64, device=vr64.device)
            if not torch.allclose(gram_vr, eye_vr, atol=1e-4):
                raise ValueError("tensor 'erasure.vr' rows are not orthonormal")
        elif key == "erasure.diag":
            if t.ndim != 1 or t.shape[0] < 1:
                raise ValueError(f"tensor 'erasure.diag' must have shape [r] with r >= 1, got {tuple(t.shape)}")
            if bool(((t < 0.0) | (t > 1.0)).any()):
                raise ValueError("tensor 'erasure.diag' values must lie in [0.0, 1.0]")
        else:
            raise ValueError(f"unknown control vector tensor key {key!r}")

    if ("erasure.vr" in tensors) != ("erasure.diag" in tensors):
        raise ValueError("tensors 'erasure.vr' and 'erasure.diag' must be provided together")
    if "erasure.vr" in tensors and tensors["erasure.vr"].shape[0] != tensors["erasure.diag"].shape[0]:
        raise ValueError(
            f"tensor 'erasure.vr' rank {tensors['erasure.vr'].shape[0]} != 'erasure.diag' len {tensors['erasure.diag'].shape[0]}"
        )
    if ("erasure.mu_d" in tensors or "erasure.mu_a" in tensors) and "erasure.vr" not in tensors:
        raise ValueError("tensor 'erasure.mu_d'/'erasure.mu_a' requires paired 'erasure.vr' and 'erasure.diag'")

    if not has_known_vector:
        raise ValueError("control vector must contain at least one 'vector.*' tensor")

    if "vector.post_refiner" in tensors and "holds.post_refiner" in tensors:
        v64 = tensors["vector.post_refiner"].detach().to(dtype=torch.float64)
        b64 = tensors["holds.post_refiner"].detach().to(dtype=torch.float64)
        v_norm = float(torch.linalg.vector_norm(v64).item())
        dots = b64 @ v64
        if float(dots.abs().max().item()) > 1e-4 * max(1.0, v_norm):
            raise ValueError(
                "tensor 'vector.post_refiner' is not orthogonal to 'holds.post_refiner'"
            )


def default_vector_filename(
    stem: str,
    family: str,
    *,
    pooling: str = "scene_text",
    prefix: Any = "none",
    apply_holds: bool = True,
    n_holds: int = 0,
    loras: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    safe_family = str(family).replace(":", "_")
    parts = [str(stem), safe_family]
    if pooling != "scene_text":
        parts.append(str(pooling))
    if prefix != "none" and prefix is not None:
        if isinstance(prefix, Mapping) and "references_sha256" in prefix:
            parts.append(f"ref_{str(prefix['references_sha256'])[:8]}")
        else:
            digest8 = hashlib.sha256(str(prefix).encode("utf-8")).hexdigest()[:8]
            parts.append(f"prefix_{digest8}")
    if n_holds > 0 and not apply_holds:
        parts.append("noholds")
    if loras:
        lora_digest8 = hashlib.sha256(
            json.dumps([dict(item) for item in loras], sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        parts.append(f"lora_{lora_digest8}")
    return "__".join(parts) + ".safetensors"


def strength_ranges(raw_norm: float) -> tuple[tuple[float, float], tuple[float, float] | None]:
    lo, hi = NORM_SHIFT_WINDOW
    matched = (lo / NORM_MATCHED_TARGET, hi / NORM_MATCHED_TARGET)
    add = (lo / raw_norm, hi / raw_norm) if raw_norm > 1e-9 else None
    return matched, add


def format_vector_sidecar_md(
    safetensors_filename: str,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    *,
    pair_spans: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    name = str(metadata.get("name", Path(safetensors_filename).stem))
    quality = str(metadata.get("quality", ""))
    family = str(metadata.get("family", ""))
    measured = dict(metadata.get("measured_under") or {})
    pooling = str(measured.get("pooling", "scene_text"))
    mode = str(measured.get("mode", ""))
    frame = str(measured.get("frame", ""))
    prefix = measured.get("prefix", "none")
    prefix_str = json.dumps(prefix, sort_keys=True) if isinstance(prefix, Mapping) else str(prefix)

    prov = dict(metadata.get("provenance") or {})
    pairs_meta = dict(metadata.get("pairs") or {})
    a_side = str(pairs_meta.get("a_side") or metadata.get("a_side") or "")
    b_side = str(pairs_meta.get("b_side") or metadata.get("b_side") or "")
    resolved_spans = (
        list(pair_spans)
        if pair_spans is not None
        else list(pairs_meta.get("pair_spans") or metadata.get("pair_spans") or ())
    )

    recipe = dict(metadata.get("recipe") or {})
    ckpt = dict(recipe.get("checkpoint") or {})
    enc = dict(recipe.get("encoder") or {})
    loras = list(recipe.get("loras") or ())
    points = dict(metadata.get("points") or {})
    qrep = dict(metadata.get("quality_report") or {})

    lines: list[str] = [
        f"# Control Vector: `{name}` (`{safetensors_filename}`)",
        "",
        "## How It Was Built (`measured_under` & `provenance`)",
        f"- **Safetensors file**: `{safetensors_filename}`",
        f"- **Pooling**: `{pooling}`",
        f"- **Family**: `{family}`",
        f"- **Mode**: `{mode}` | **Frame**: `{frame}` | **Prefix**: `{prefix_str}`",
        f"- **Created (UTC)**: `{prov.get('created_utc', 'unknown')}`",
        f"- **Made by**: `{prov.get('made_by', prov.get('author_kind', 'unknown'))}`",
        f"- **Pack commit**: `{prov.get('pack_commit', 'unknown')}` | **Core revision**: `{prov.get('core_revision', 'unknown')}`",
        "",
        "## Target Quality & Sides",
        f"- **Quality**: {quality}",
    ]
    if a_side:
        lines.append(f"- **Pole A (`a_side`, baseline)**: {a_side}")
    if b_side:
        lines.append(f"- **Pole B (`b_side`, steered)**: {b_side}")
    if isinstance(metadata.get("routing"), Mapping):
        lines.append(f"- **Routing**: `{json.dumps(dict(metadata['routing']), sort_keys=True)}`")

    ckpt_sha12 = str(ckpt.get("sha256", ""))[:12]
    enc_sha12 = str(enc.get("sha256", ""))[:12]
    lines.extend(
        [
            "",
            "## Model Recipe (`recipe`)",
            f"- **Checkpoint**: `{ckpt.get('file', 'unspecified')}` (`{ckpt_sha12}`)",
            f"- **Encoder**: `{enc.get('file', 'unspecified')}` (`{enc_sha12}`)",
        ]
    )
    if loras:
        for l_item in loras:
            l_sha12 = str(l_item.get("sha256", ""))[:12]
            l_str = l_item.get("strength", 1.0)
            l_app = l_item.get("applied", "branch")
            lines.append(
                f"- **LoRA**: `{l_item.get('file', 'unknown')}` (strength=`{l_str:g}`, sha256=`{l_sha12}`, applied=`{l_app}`)"
            )
    else:
        lines.append("- **LoRAs**: none")

    pairs_sha = str(pairs_meta.get("sha256", ""))
    lines.extend(
        [
            "",
            "## Source Pair Set (`pairs`)",
            f"- **Pair TOML path**: `{pairs_meta.get('path', 'unknown')}`",
            f"- **Pair TOML SHA-256**: `{pairs_sha[:12]}` (`{pairs_sha}`)",
            f"- **Pairs**: `{pairs_meta.get('n_pairs', 0)}` | **Holds**: `{pairs_meta.get('n_holds', 0)}`",
            "",
            "## Norms, Recommended Strength & Quality Report (`points` & `quality_report`)",
        ]
    )

    if "post_refiner" in points:
        pr = dict(points["post_refiner"])
        pr_norm = float(pr.get("raw_norm", 0.0))
        pr_med = float(pr.get("median_row_norm", 0.0))
        pr_shape = (
            list(tensors["vector.post_refiner"].shape)
            if "vector.post_refiner" in tensors
            else [POST_REFINER_DIM]
        )
        (matched_lo, matched_hi), add_range = strength_ranges(pr_norm)
        lines.append(
            f"- **`vector.post_refiner`**: shape=`{pr_shape}`, unit=`{pr.get('unit', 'mean_pair_difference')}`, "
            f"`raw_norm={pr_norm:.6f}`, `median_row_norm={pr_med:.4f}`"
        )
        if add_range is not None:
            window = f"`{NORM_SHIFT_WINDOW[0]:g}..{NORM_SHIFT_WINDOW[1]:g}`"
            lines.append(
                f"- **Recommended ComfyUI `strength` range** (targeting post-refiner shift norm {window}): "
                f"**`~{matched_lo:.2f} .. {matched_hi:.2f}`** in mode `add_norm_matched_erasure`, "
                f"`~{add_range[0]:.2f} .. {add_range[1]:.2f}` in mode `add`"
            )

    if "encoder" in points:
        en = dict(points["encoder"])
        en_norm = float(en.get("raw_norm", 0.0))
        en_med = float(en.get("median_row_norm", 0.0))
        en_shape = (
            list(tensors["vector.encoder"].shape)
            if "vector.encoder" in tensors
            else [ENCODER_DIM]
        )
        lines.append(
            f"- **`vector.encoder`**: shape=`{en_shape}`, unit=`{en.get('unit', 'median_row_norm')}`, "
            f"`raw_norm={en_norm:.6f}`, `median_row_norm={en_med:.4f}`, "
            f"`row0_excluded={en.get('row0_excluded', True)}`"
        )

    for t_key, t_val in tensors.items():
        if t_key not in ("vector.post_refiner", "vector.encoder"):
            lines.append(f"- **`{t_key}`**: shape=`{list(t_val.shape)}`, dtype=`{t_val.dtype}`")

    mpc_raw = qrep.get("mean_pairwise_cosine")
    shc_raw = qrep.get("split_half_cosine")
    mpc_str = f"{float(mpc_raw):.4f}" if mpc_raw is not None else "n/a (1 pair)"
    shc_str = f"{float(shc_raw):.4f}" if shc_raw is not None else "n/a (1 pair)"
    ppc = [float(x) for x in (qrep.get("per_pair_cosine") or ())]
    ppc_min = min(ppc) if ppc else 0.0
    ppc_max = max(ppc) if ppc else 0.0
    ppc_formatted = ", ".join(f"{x:.4f}" for x in ppc)
    lines.extend(
        [
            f"- **`mean_pairwise_cosine`**: `{mpc_str}`",
            f"- **`split_half_cosine`**: `{shc_str}`",
            f"- **`per_pair_cosine`**: min=`{ppc_min:.4f}`, max=`{ppc_max:.4f}` (`[{ppc_formatted}]`)",
        ]
    )
    for extra_qk in sorted(set(qrep.keys()) - {"mean_pairwise_cosine", "split_half_cosine", "per_pair_cosine"}):
        lines.append(f"- **`{extra_qk}`**: `{json.dumps(qrep[extra_qk], sort_keys=True)}`")

    if resolved_spans:
        lines.extend(
            [
                "",
                "## Per-Pair Token Spans",
                "",
                "| Pair | Weight | Bounds (`A` / `B`) | Pooled Toks (`A` / `B`) | Seq Len (`A` / `B`) | Differing Span (`A` -> `B`) |",
                "|---:|---:|---|---|---|---|",
            ]
        )
        for sp in resolved_spans:
            idx = int(sp.get("index", 0))
            w = float(sp.get("weight", 1.0))
            la = int(sp.get("len_a", 0))
            lb = int(sp.get("len_b", 0))
            pa = int(sp.get("pooled_rows_a", 0))
            pb = int(sp.get("pooled_rows_b", 0))
            if "first_diff" in sp and ("end_a" in sp or "end_b" in sp):
                fd = int(sp["first_diff"])
                ea = int(sp.get("end_a", la))
                eb = int(sp.get("end_b", lb))
                bounds_str = f"`[{fd}:{ea}]` / `[{fd}:{eb}]`"
            elif "first_diff" in sp:
                fd = int(sp["first_diff"])
                bounds_str = f"`[{fd}:{la}]` / `[{fd}:{lb}]`"
            else:
                bounds_str = "`all` / `all`"
            a_sp = str(sp.get("a_span", "")).replace("|", "\\|").replace("\n", " ")
            b_sp = str(sp.get("b_span", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{idx:02d}` | `{w:.2f}` | {bounds_str} | `{pa}` / `{pb}` | `{la}` / `{lb}` | `-{a_sp}` -> `+{b_sp}` |"
            )

    lines.extend(
        [
            "",
            "## How to Read These Scores",
            "- **`mean_pairwise_cosine` / `split_half_cosine`**: coherence of `(b - a)` across scenes (`>= 0.45` clean single-axis contrast).",
            "- **`add_norm_matched_erasure` (`MiniMaxH3Steer` default)**: rescales the push to `67.5` RMS at `strength=1.0` while erasing the Pole-A baseline subspace (`erasure.vr`/`erasure.diag`).",
        ]
    )

    return "\n".join(lines) + "\n"


def _chmod_group_readable(path: Path) -> None:
    try:
        os.chmod(path, 0o664)
    except PermissionError:
        pass


def save_vector(
    path: str | Path,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    *,
    overwrite: bool = True,
    write_sidecar_md: bool = True,
) -> Path:
    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("control vector tensors dict is empty")
    clean_tensors: dict[str, torch.Tensor] = {}
    for k, v in tensors.items():
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"tensor {k!r} is not a torch.Tensor")
        if v.dtype != torch.float32:
            raise ValueError(f"tensor {k!r} must have dtype float32, got {v.dtype}")
        clean_tensors[k] = v.detach().to(device="cpu", dtype=torch.float32).contiguous()

    _validate_vector_payload(clean_tensors, metadata)

    encoded_meta: dict[str, str] = {}
    for k, v in metadata.items():
        if v is None:
            continue
        encoded_meta[str(k)] = json.dumps(v, sort_keys=True)

    out_path = Path(path)
    if not overwrite and out_path.exists():
        raise ValueError(
            f"control vector file {out_path.name!r} already exists at {out_path} "
            "(pass overwrite=True / --overwrite to replace it, or set a distinct output_name)"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(save_safetensors_bytes(clean_tensors, metadata=encoded_meta))
    _chmod_group_readable(out_path)
    if write_sidecar_md:
        sidecar_path = out_path.with_suffix(".md")
        sidecar_md = format_vector_sidecar_md(out_path.name, clean_tensors, metadata)
        sidecar_path.write_text(sidecar_md, encoding="utf-8")
        _chmod_group_readable(sidecar_path)
    return out_path


def load_vector(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"control vector file not found: {p}")

    tensors: dict[str, torch.Tensor] = {}
    raw_meta: dict[str, str] = {}
    with safe_open(str(p), framework="pt", device="cpu") as sf:
        raw_meta = dict(sf.metadata() or {})
        for k in sf.keys():
            tensors[k] = sf.get_tensor(k).clone()

    metadata: dict[str, Any] = {}
    for k, v in raw_meta.items():
        try:
            metadata[k] = json.loads(v)
        except Exception:
            metadata[k] = v

    _validate_vector_payload(tensors, metadata)
    return tensors, metadata


def _applied_refiner_coverage(applied: str) -> str:
    applied_norm = str(applied or "branch").strip()
    if applied_norm in ("branch", "merged"):
        return "full_refiner"
    if applied_norm.startswith("branch:"):
        parts: dict[str, str] = {}
        last_key: str | None = None
        for chunk in applied_norm[len("branch:") :].split(","):
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                last_key = k.strip().lower()
                parts[last_key] = v.strip()
            elif last_key is not None:
                parts[last_key] = f"{parts[last_key]},{chunk.strip()}"
        modules_val = parts.get("modules", "all").strip().lower()
        blocks_val = parts.get("blocks", "all").strip().lower()
        block_tokens = {t.strip() for t in blocks_val.split(",") if t.strip()}
        covers_blocks = blocks_val == "all" or bool(block_tokens & {"refiner", "token_refiner"})
        if modules_val in ("all", "no adaln") and covers_blocks:
            return "full_refiner"
        return f"partial_refiner:modules={modules_val},blocks={blocks_val}"
    return f"other:{applied_norm}"


def _lora_family_signature(loras: Sequence[Any]) -> list[tuple[str, str]] | None:
    sig: list[tuple[str, str]] = []
    for item in loras:
        if not isinstance(item, Mapping):
            return None
        raw_file = str(item.get("file", item.get("name", "")))
        m = PIKA_FLEX_VARIANT_RE.match(raw_file)
        if m is None:
            return None
        strength_f = float(item.get("strength", 1.0))
        if not math.isfinite(strength_f):
            return None
        refiner_cov = _applied_refiner_coverage(str(item.get("applied", "branch")))
        sig.append((m.group(1), refiner_cov))
    return sig


def _is_known_h3_family_lora(item: Any, family: str) -> bool:
    if not isinstance(item, Mapping):
        return False
    if _applied_refiner_coverage(str(item.get("applied", "branch"))) != "full_refiner":
        return False
    if not math.isfinite(float(item.get("strength", 1.0))):
        return False
    raw_file = Path(str(item.get("file", item.get("name", "")))).name
    m = PIKA_FLEX_VARIANT_RE.match(raw_file)
    if m is not None and m.group(1) == f"pika_h3_flex_{family}":
        return True
    if raw_file == "p-h3-image_lora_bf16.safetensors":
        return True
    if family == "ref2va" and raw_file in KNOWN_REF2VA_GUIDE_LORAS:
        return True
    return False


def _is_same_family_h3_recipe_compatible(
    vec_recipe: Mapping[str, Any],
    cur_recipe: Mapping[str, Any],
    *,
    vec_family: str,
    loaded_family: str,
) -> bool:
    if not (
        (vec_family == loaded_family and vec_family in ("fl2va", "ref2va", "h3_image"))
        or {vec_family, loaded_family} <= VIDEO_H3_FAMILIES
    ):
        return False
    cur_ckpt = cur_recipe.get("checkpoint")
    vec_ckpt = vec_recipe.get("checkpoint")
    cur_ckpt_file = Path(str(cur_ckpt.get("file", ""))).name if isinstance(cur_ckpt, Mapping) else ""
    vec_ckpt_file = Path(str(vec_ckpt.get("file", ""))).name if isinstance(vec_ckpt, Mapping) else ""
    vec_known_ckpts = KNOWN_H3_CHECKPOINTS[vec_family]
    cur_known_ckpts = KNOWN_H3_CHECKPOINTS[loaded_family]
    if vec_ckpt != cur_ckpt and not (
        vec_ckpt_file in vec_known_ckpts and cur_ckpt_file in cur_known_ckpts
    ):
        return False
    cur_loras = cur_recipe.get("loras", [])
    vec_loras = vec_recipe.get("loras", [])
    if not isinstance(cur_loras, Sequence) or not isinstance(vec_loras, Sequence):
        return False
    if not all(_is_known_h3_family_lora(item, vec_family) for item in vec_loras):
        return False
    if not all(_is_known_h3_family_lora(item, loaded_family) for item in cur_loras):
        return False
    return True


def check_vector_compatibility(
    metadata: Mapping[str, Any],
    *,
    loaded_family: str,
    loaded_recipe: Mapping[str, Any] | None = None,
    loaded_prefix: Any | None = None,
    allow_recipe_mismatch: bool = True,
) -> list[str]:
    _validate_metadata_dict(metadata)

    warnings: list[str] = []
    vec_name = metadata["name"]
    vec_family = metadata["family"]
    compatible_video_families = {vec_family, loaded_family} <= VIDEO_H3_FAMILIES
    if vec_family != loaded_family:
        if compatible_video_families:
            warnings.append(
                f"control vector {vec_name!r} was made for H3 video family {vec_family!r}, "
                f"applying to loaded video family {loaded_family!r} (post-refiner cos ~0.99)"
            )
        else:
            raise ValueError(
                f"control vector {vec_name!r} was made for H3 text-path family {vec_family!r}, "
                f"cannot apply to loaded model of family {loaded_family!r}"
            )

    if loaded_recipe is None:
        if not allow_recipe_mismatch:
            raise ValueError("loaded_recipe is required unless allow_recipe_mismatch=True")
    else:
        if not isinstance(loaded_recipe, Mapping):
            raise ValueError("loaded_recipe must be a mapping")
        vec_recipe = metadata["recipe"]
        vec_ckpt = vec_recipe.get("checkpoint")
        cur_ckpt = loaded_recipe.get("checkpoint")
        prov_made_by = str(metadata.get("provenance", {}).get("made_by", ""))
        jlens_same_family = prov_made_by.startswith("j_lens.") and (
            vec_family == loaded_family or compatible_video_families
        )
        h3_family_compatible = _is_same_family_h3_recipe_compatible(
            vec_recipe, loaded_recipe, vec_family=vec_family, loaded_family=loaded_family
        )
        if vec_ckpt != cur_ckpt:
            msg = (
                f"control vector {vec_name!r} recipe checkpoint {vec_ckpt!r} differs "
                f"from loaded model checkpoint {cur_ckpt!r}"
            )
            if not allow_recipe_mismatch and not jlens_same_family and not h3_family_compatible:
                raise ValueError(
                    f"{msg} (pass allow_recipe_mismatch=True to apply across checkpoints)"
                )
            warnings.append(msg)

        vec_loras = vec_recipe.get("loras", [])
        cur_loras = loaded_recipe.get("loras", [])
        if vec_loras != cur_loras:
            msg = (
                f"control vector {vec_name!r} recipe LoRA set {vec_loras!r} differs "
                f"from loaded model LoRA set {cur_loras!r}"
            )
            vec_sig = _lora_family_signature(vec_loras) if vec_loras else None
            cur_sig = _lora_family_signature(cur_loras) if cur_loras else None
            same_pika_family = vec_sig is not None and cur_sig is not None and vec_sig == cur_sig
            if (
                not allow_recipe_mismatch
                and not same_pika_family
                and not jlens_same_family
                and not h3_family_compatible
            ):
                raise ValueError(
                    f"{msg} (pass allow_recipe_mismatch=True to apply across LoRA sets)"
                )
            warnings.append(msg)

    if loaded_prefix is not None:
        vec_prefix = metadata["measured_under"].get("prefix")
        if vec_prefix != loaded_prefix:
            warnings.append(
                f"control vector {vec_name!r} was measured under prefix {vec_prefix!r}, "
                f"render has prefix {loaded_prefix!r}"
            )

    return warnings


def vector_interactions(
    entries: Sequence[tuple[str, torch.Tensor, float]],
) -> dict[str, Any]:
    vecs64: list[tuple[str, torch.Tensor, float]] = []
    for name, vec, strength in entries:
        s_f = float(strength)
        if s_f == 0.0:
            continue
        v64 = vec.detach().to(device="cpu", dtype=torch.float64).flatten()
        vecs64.append((str(name), v64, s_f))

    if not vecs64:
        return {
            "combined_norm": 0.0,
            "sum_individual_norms": 0.0,
            "combined_norm_sq": 0.0,
            "sum_individual_norm_sq": 0.0,
            "total_interaction_sq": 0.0,
            "pairs": [],
        }

    d = vecs64[0][1].shape[0]
    total_vec = torch.zeros(d, dtype=torch.float64)
    sum_ind_norms = 0.0
    sum_ind_norm_sq = 0.0
    for _name, v64, s_f in vecs64:
        scaled = s_f * v64
        total_vec = total_vec + scaled
        nrm = float(torch.linalg.vector_norm(scaled).item())
        sum_ind_norms += nrm
        sum_ind_norm_sq += nrm * nrm

    pairs: list[dict[str, Any]] = []
    total_inter_sq = 0.0
    for i in range(len(vecs64)):
        name_i, v_i, s_i = vecs64[i]
        norm_i = float(torch.linalg.vector_norm(v_i).item())
        for j in range(i + 1, len(vecs64)):
            name_j, v_j, s_j = vecs64[j]
            norm_j = float(torch.linalg.vector_norm(v_j).item())
            dot_ij = float(torch.dot(v_i, v_j).item())
            cos_ij = dot_ij / (norm_i * norm_j) if (norm_i > 1e-12 and norm_j > 1e-12) else 0.0
            sign_ij = 1.0 if (s_i * s_j) >= 0.0 else -1.0
            signed_cos_ij = sign_ij * cos_ij
            inter_sq = 2.0 * s_i * s_j * dot_ij
            total_inter_sq += inter_sq
            pairs.append(
                {
                    "a": name_i,
                    "b": name_j,
                    "strength_a": s_i,
                    "strength_b": s_j,
                    "cosine": cos_ij,
                    "signed_cosine": signed_cos_ij,
                    "dot": dot_ij,
                    "interaction_sq": inter_sq,
                }
            )

    comb_norm_sq = float(torch.dot(total_vec, total_vec).item())
    return {
        "combined_norm": math.sqrt(max(0.0, comb_norm_sq)),
        "sum_individual_norms": sum_ind_norms,
        "combined_norm_sq": comb_norm_sq,
        "sum_individual_norm_sq": sum_ind_norm_sq,
        "total_interaction_sq": total_inter_sq,
        "pairs": pairs,
    }


def row_norm_census(rows: torch.Tensor, tags: torch.Tensor | None = None) -> dict[str, float]:
    r = rows[0] if rows.ndim == 3 else rows
    if r.ndim != 2 or r.shape[0] < 1:
        raise ValueError(f"rows must have shape [L, D] or [1, L, D] with L >= 1, got {tuple(rows.shape)}")
    r64 = r.detach().to(dtype=torch.float64)
    norms = torch.linalg.vector_norm(r64, dim=-1)
    row0 = float(norms[0].item())
    if tags is not None:
        t_flat = tags.flatten().to(device=r64.device)
        tag1 = norms[t_flat != 0]
        tag1_ex0 = norms[1:][t_flat[1:] != 0] if norms.shape[0] > 1 else tag1
    else:
        tag1 = norms
        tag1_ex0 = norms[1:] if norms.shape[0] > 1 else norms
    med_all = float(tag1.median().item()) if tag1.numel() > 0 else row0
    med_ex0 = float(tag1_ex0.median().item()) if tag1_ex0.numel() > 0 else med_all
    return {
        "row0_norm": row0,
        "median_row_norm": med_all,
        "median_row_norm_ex_row0": med_ex0,
        "row0_to_median_ratio": row0 / med_ex0 if med_ex0 > 0.0 else 1.0,
    }


def channel_census(vec: torch.Tensor) -> dict[str, float]:
    if vec.ndim != 1 or vec.shape[0] < 1:
        raise ValueError(f"vec must be 1D [D], got {tuple(vec.shape)}")
    v64 = vec.detach().to(dtype=torch.float64)
    sq = v64 * v64
    total_energy = float(sq.sum().item())
    if total_energy <= 0.0:
        raise ValueError("vec has zero energy")
    sorted_sq, _ = torch.sort(sq, descending=True)
    d = v64.shape[0]
    rms = math.sqrt(total_energy / d)
    max_abs = float(v64.abs().max().item())
    return {
        "top1_energy_share": float(sorted_sq[:1].sum().item() / total_energy),
        "top5_energy_share": float(sorted_sq[: min(5, d)].sum().item() / total_energy),
        "top10_energy_share": float(sorted_sq[: min(10, d)].sum().item() / total_energy),
        "max_abs_over_rms": max_abs / rms if rms > 0.0 else 0.0,
    }


def meter_score(
    rows: torch.Tensor,
    vec: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    tokens: Sequence[str] | None = None,
) -> dict[str, Any]:
    r = rows[0] if rows.ndim == 3 else rows
    if r.ndim != 2 or r.shape[1] != vec.shape[0]:
        raise ValueError(
            f"rows shape {tuple(rows.shape)} incompatible with vec shape {tuple(vec.shape)}"
        )
    r64 = r.detach().to(dtype=torch.float64)
    if mask is not None:
        m = mask.flatten().to(device=r64.device, dtype=torch.bool)
        if m.shape[0] != r64.shape[0] or not bool(m.any()):
            raise ValueError("meter_score mask must match row count L and select at least one row")
        pooled = r64[m].mean(dim=0)
        active_indices = [i for i, flag in enumerate(m.tolist()) if flag]
    else:
        pooled = r64.mean(dim=0)
        active_indices = list(range(int(r64.shape[0])))
    v64 = vec.detach().to(device=r64.device, dtype=torch.float64)
    v_norm_sq = float(torch.dot(v64, v64).item())
    if v_norm_sq <= 0.0:
        raise ValueError("meter_score vector has zero norm")
    v_norm = math.sqrt(v_norm_sq)
    p_norm = float(torch.linalg.vector_norm(pooled).item())
    dot = float(torch.dot(pooled, v64).item())

    v_unit = v64 / v_norm
    row_norms = torch.linalg.vector_norm(r64, dim=-1).clamp(min=1e-12)
    per_tok_cos_t = (r64 @ v_unit) / row_norms
    per_tok_cos_list = [float(x) for x in per_tok_cos_t.tolist()]
    active_cos_t = per_tok_cos_t[active_indices] if active_indices else per_tok_cos_t
    max_tok_cos = float(active_cos_t.max().item()) if active_cos_t.numel() > 0 else 0.0
    p90_tok_cos = (
        float(torch.quantile(active_cos_t, 0.90).item()) if active_cos_t.numel() > 0 else 0.0
    )
    top_coupled: list[dict[str, Any]] = []
    if tokens is not None:
        for idx_i in active_indices:
            tok_s = str(tokens[idx_i]).strip() if idx_i < len(tokens) else ""
            if tok_s:
                top_coupled.append(
                    {"index": idx_i, "token": tok_s, "cosine": float(per_tok_cos_list[idx_i])}
                )
        top_coupled.sort(key=lambda d: d["cosine"], reverse=True)
        top_coupled = top_coupled[:8]

    cos_val = dot / (p_norm * v_norm) if p_norm > 0.0 else 0.0
    proj_units = dot / v_norm_sq
    return {
        "projection_units": proj_units,
        "z_score": proj_units,
        "signed_projection": dot / v_norm,
        "cosine": cos_val,
        "mean_cosine": cos_val,
        "max_token_cosine": max_tok_cos,
        "p90_token_cosine": p90_tok_cos,
        "p90_cosine": p90_tok_cos,
        "per_token_cosines": per_tok_cos_list,
        "top_coupled_tokens": top_coupled,
    }
