"""Point C post-refiner prelude math: `h_bar` centering, `peer_ros=True` orthogonalization,
clamped `[0, 1]` union SVD, smooth `95.5 -> 98.5` RMS knee, and `s = 0` exact identity."""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

import torch

from .constants import (
    APPLY_MODES,
    NORM_MATCHED_TARGET,
    POST_REFINER_DIM,
    SCALE_MODES,
    STACK_SHIFT_WARN_NORM,
    STACK_SMOOTH_CEIL_NORM,
    STACK_SMOOTH_KNEE_NORM,
)

_logger = logging.getLogger("ComfyUI-h3-explorations.control_vectors")


def is_additive_mode(mode: str) -> bool:
    """Whether `mode` is an additive steering mode (`add`, `add_norm_matched`, or `add_norm_matched_erasure`)."""
    return mode in ("add", "add_norm_matched", "add_norm_matched_erasure")


def is_norm_matched_mode(mode: str) -> bool:
    """Whether `mode` normalizes the push to `NORM_MATCHED_TARGET` (`67.5`)."""
    return mode in ("add_norm_matched", "add_norm_matched_erasure")


def soft_cap_stacked_rms(
    rms: float,
    *,
    knee: float = STACK_SMOOTH_KNEE_NORM,
    ceil: float = STACK_SMOOTH_CEIL_NORM,
) -> float:
    """Smoothly compress distinct stacked shift RMS past `knee` (`95.5`) toward `ceil` (`98.5`) via `tanh`."""
    rms_f = float(rms)
    if not math.isfinite(rms_f) or rms_f <= knee or ceil <= knee:
        return rms_f
    span = float(ceil - knee)
    return float(knee + span * math.tanh((rms_f - knee) / span))


def _has_distinct_unit_peers(unit_vecs: Sequence[torch.Tensor]) -> bool:
    if len(unit_vecs) < 2:
        return False
    for idx_a in range(len(unit_vecs)):
        ua = unit_vecs[idx_a]
        for idx_b in range(idx_a + 1, len(unit_vecs)):
            ub = unit_vecs[idx_b]
            if abs(float(torch.dot(ua, ub).item())) < 1.0 - 1e-4:
                return True
    return False


def _resolve_stacked_target_rms(
    raw_rms: float,
    *,
    max_stacked_rms: float | str | None,
    has_distinct_peers: bool,
) -> float:
    if max_stacked_rms is None:
        return float(raw_rms)
    if isinstance(max_stacked_rms, str):
        mode_s = max_stacked_rms.strip().lower()
        if mode_s in ("", "none", "off", "uncapped"):
            return float(raw_rms)
        if mode_s == "smooth":
            if has_distinct_peers and raw_rms > STACK_SMOOTH_KNEE_NORM:
                return soft_cap_stacked_rms(raw_rms)
            return float(raw_rms)
        cap_f = float(mode_s)
    else:
        cap_f = float(max_stacked_rms)
    if not math.isfinite(cap_f):
        raise ValueError(f"max_stacked_rms must be finite, 'smooth', or None, got {max_stacked_rms!r}")
    if cap_f > 0.0 and raw_rms > cap_f:
        return cap_f
    return float(raw_rms)


def orthogonalize_erasure_against_holds(
    vr: torch.Tensor,
    diag: torch.Tensor,
    hold_vecs: Sequence[torch.Tensor],
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Attenuate or project `hold_vecs` out of symmetric erasure operator `Vr^T diag Vr`
    while preserving orthonormality of `Vr` and `[0, 1]` bounds on `diag` via thin SVD."""
    a = min(1.0, max(0.0, float(alpha)))
    if a <= 0.0 or not hold_vecs or vr.ndim != 2 or vr.shape[0] < 1:
        return vr, diag
    basis_rows: list[torch.Tensor] = []
    for hv in hold_vecs:
        u = hv.detach().to(device="cpu", dtype=torch.float64).flatten().clone()
        for _ in range(2):
            for prev in basis_rows:
                u = u - torch.dot(u, prev) * prev
        nrm = float(torch.linalg.vector_norm(u).item())
        if nrm > 1e-12:
            basis_rows.append(u / nrm)
    if not basis_rows:
        return vr, diag
    bh = torch.stack(basis_rows, dim=0)
    vr64 = vr.detach().to(device="cpu", dtype=torch.float64)
    d64 = diag.detach().to(device="cpu", dtype=torch.float64).clamp(0.0, 1.0)
    vr_proj = vr64 - a * ((vr64 @ bh.T) @ bh)
    m_mat = torch.sqrt(d64).unsqueeze(-1) * vr_proj
    _u, s_vals, vh = torch.linalg.svd(m_mat, full_matrices=False)
    d_new = (s_vals ** 2).clamp(0.0, 1.0)
    keep = d_new > 1e-9
    if not bool(keep.any()):
        return None
    return (
        vh[keep].to(dtype=torch.float32).contiguous(),
        d_new[keep].to(dtype=torch.float32).contiguous(),
    )


def _apply_holds_to_shift64(
    total_shift64: torch.Tensor,
    hold_entries: Sequence[tuple[torch.Tensor, float]],
    device: torch.device,
) -> torch.Tensor:
    if not hold_entries or float(torch.linalg.vector_norm(total_shift64).item()) <= 1e-24:
        return total_shift64
    ortho_holds: list[tuple[torch.Tensor, float]] = []
    for h_vec, h_alpha in hold_entries:
        h_alpha_f = float(h_alpha)
        if not math.isfinite(h_alpha_f) or not (0.0 <= h_alpha_f <= 2.0):
            raise ValueError(f"hold alpha must be finite and in [0.0, 2.0], got {h_alpha!r}")
        u = h_vec.detach().to(device=device, dtype=torch.float64).clone()
        for _pass in range(2):
            for prev_u, _ in ortho_holds:
                u = u - torch.dot(u, prev_u) * prev_u
        nrm = torch.linalg.vector_norm(u)
        if float(nrm) > 1e-12:
            ortho_holds.append((u / nrm, h_alpha_f))
    out_shift = total_shift64
    for u_hat, h_alpha in ortho_holds:
        out_shift = out_shift - h_alpha * torch.dot(out_shift, u_hat) * u_hat
    return out_shift


def _apply_holds_to_rows64(
    shift_rows64: torch.Tensor,
    hold_entries: Sequence[tuple[torch.Tensor, float]],
    device: torch.device,
) -> torch.Tensor:
    if not hold_entries or shift_rows64.numel() == 0:
        return shift_rows64
    ortho_holds: list[tuple[torch.Tensor, float]] = []
    for h_vec, h_alpha in hold_entries:
        h_alpha_f = float(h_alpha)
        if not math.isfinite(h_alpha_f) or not (0.0 <= h_alpha_f <= 2.0):
            raise ValueError(f"hold alpha must be finite and in [0.0, 2.0], got {h_alpha!r}")
        u = h_vec.detach().to(device=device, dtype=torch.float64).clone()
        for _pass in range(2):
            for prev_u, _ in ortho_holds:
                u = u - torch.dot(u, prev_u) * prev_u
        nrm = torch.linalg.vector_norm(u)
        if float(nrm) > 1e-12:
            ortho_holds.append((u / nrm, h_alpha_f))
    out_rows = shift_rows64
    for u_hat, h_alpha in ortho_holds:
        proj = out_rows @ u_hat
        out_rows = out_rows - (h_alpha * proj).unsqueeze(-1) * u_hat.unsqueeze(0)
    return out_rows


def resolve_post_refiner_row_mask(
    tags: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    mask: torch.Tensor | None = None,
    mask_vision: bool = True,
) -> torch.Tensor:
    """Return the `[B, L]` boolean row mask of context tokens steered by `apply_post_refiner`."""
    if tags.ndim == 1:
        tag_2d = tags.unsqueeze(0).expand(batch_size, -1)
    elif tags.ndim == 2:
        tag_2d = tags
    else:
        raise ValueError(f"tags must be 1D [L] or 2D [B, L], got shape {tuple(tags.shape)}")
    if tag_2d.shape != (batch_size, seq_len):
        raise ValueError(
            f"tags shape {tuple(tag_2d.shape)} does not match context batch/length {(batch_size, seq_len)}"
        )

    base_mask = (
        (tag_2d != 0).to(device=device)
        if mask_vision
        else torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    )
    if mask is None:
        return base_mask
    if mask.ndim == 1:
        row_mask = mask.to(device=device, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1)
    elif mask.ndim == 2:
        row_mask = mask.to(device=device, dtype=torch.bool)
    else:
        raise ValueError(f"mask must be 1D [L] or 2D [B, L], got shape {tuple(mask.shape)}")
    if row_mask.shape != (batch_size, seq_len):
        raise ValueError(
            f"mask shape {tuple(row_mask.shape)} does not match context batch/length {(batch_size, seq_len)}"
        )
    return row_mask & base_mask


def focus_row_mask(
    context: torch.Tensor,
    row_mask: torch.Tensor,
    vec: torch.Tensor,
    *,
    token_focus: float = 1.0,
) -> tuple[torch.Tensor, float]:
    tf = float(token_focus)
    if not math.isfinite(tf) or tf <= 0.0 or tf > 1.0:
        raise ValueError(f"token_focus must be finite in (0.0, 1.0], got {token_focus!r}")
    if tf >= 1.0 - 1e-9 or not bool(row_mask.any()):
        return row_mask, 0.0
    v64 = vec.detach().to(device=context.device, dtype=torch.float64).flatten()
    v_norm = float(torch.linalg.vector_norm(v64).item())
    if v_norm <= 1e-12:
        return row_mask, 0.0
    v_unit = v64 / v_norm
    B, _L = row_mask.shape
    focused = torch.zeros_like(row_mask)
    min_abs_cos = 1.0
    for b in range(B):
        idx = row_mask[b].nonzero(as_tuple=False).flatten()
        n_active = int(idx.numel())
        if n_active == 0:
            continue
        k = max(1, min(n_active, int(math.ceil(n_active * tf))))
        if k >= n_active:
            focused[b, idx] = True
            min_abs_cos = 0.0
            continue
        rows64 = context[b, idx].detach().to(dtype=torch.float64)
        row_norms = torch.linalg.vector_norm(rows64, dim=-1).clamp(min=1e-12)
        abs_cos = ((rows64 @ v_unit) / row_norms).abs()
        top_vals, top_pos = torch.topk(abs_cos, k=k, largest=True, sorted=True)
        focused[b, idx[top_pos]] = True
        min_abs_cos = min(min_abs_cos, float(top_vals[-1].item()))
    return focused, (0.0 if min_abs_cos > 1.0 else min_abs_cos)


def _rms_row_norm64(rows64: torch.Tensor) -> float:
    if rows64.numel() == 0:
        return 0.0
    return float(torch.sqrt((rows64 ** 2).sum(dim=-1).mean()).item())


def _erasure_row_shift64(
    base_rows64: torch.Tensor,
    vec: torch.Tensor,
    strength: float,
    erasure_spec: tuple[Any, ...] | None,
) -> torch.Tensor:
    """Compute per-row `[M, D]` RMS-matched clamped prompt-centered (`h_bar`) Pole-A erasure shift."""
    device = base_rows64.device
    s_f = float(strength)
    vr_t = erasure_spec[0] if (erasure_spec is not None and len(erasure_spec) >= 1) else None
    diag_t = erasure_spec[1] if (erasure_spec is not None and len(erasure_spec) >= 2) else None
    mu_d_t = erasure_spec[2] if (erasure_spec is not None and len(erasure_spec) >= 3) else None
    per_spec_holds = erasure_spec[3] if (erasure_spec is not None and len(erasure_spec) >= 4) else None
    mu_a_t = erasure_spec[4] if (erasure_spec is not None and len(erasure_spec) >= 5) else None
    base_vec = mu_d_t if (mu_d_t is not None and s_f > 0.0 and vr_t is not None) else vec
    v64 = base_vec.detach().to(device=device, dtype=torch.float64)
    v_nrm = float(torch.linalg.vector_norm(v64).item())
    if v_nrm <= 1e-12:
        raise ValueError("control vector has zero norm; cannot apply mode='add_norm_matched_erasure'")
    mu_n = v64 * (NORM_MATCHED_TARGET / v_nrm)
    add_rows = (s_f * mu_n).unsqueeze(0).expand_as(base_rows64)

    def _finish(shift_rows64: torch.Tensor) -> torch.Tensor:
        if per_spec_holds:
            return _apply_holds_to_rows64(shift_rows64, per_spec_holds, device)
        return shift_rows64

    if s_f <= 0.0 or vr_t is None or diag_t is None or (mu_a_t is None and base_rows64.shape[0] < 2):
        return _finish(add_rows)

    vr64 = vr_t.detach().to(device=device, dtype=torch.float64)
    diag64 = diag_t.detach().to(device=device, dtype=torch.float64)
    if mu_a_t is not None:
        h_bar = mu_a_t.detach().to(device=device, dtype=torch.float64).unsqueeze(0)
    else:
        h_bar = base_rows64.mean(dim=0, keepdim=True)
    centered = base_rows64 - h_bar
    e_own = ((centered @ vr64.T) * diag64.unsqueeze(0)) @ vr64
    e_own_rms = _rms_row_norm64(e_own)
    if not math.isfinite(e_own_rms) or e_own_rms <= 1e-9:
        return _finish(add_rows)
    e_cap = min(1.0, NORM_MATCHED_TARGET / e_own_rms)
    clamp_s = max(0.0, min(s_f, 1.0))
    ref_e_rms = clamp_s * e_cap * e_own_rms
    e = e_own * (ref_e_rms / e_own_rms)
    raw = add_rows - e
    raw_rms = _rms_row_norm64(raw)
    if not math.isfinite(raw_rms) or raw_rms <= 1e-12:
        return _finish(add_rows)
    target_rms = abs(s_f) * NORM_MATCHED_TARGET
    return _finish(raw * (target_rms / raw_rms))


def _stacked_erasure_block_shift64(
    base_rows64: torch.Tensor,
    erasure_entries: Sequence[tuple[torch.Tensor, float, tuple[Any, ...]]],
    *,
    use_mu_a: bool = False,
    peer_ros: bool = True,
    max_stacked_rms: float | str | None = "smooth",
) -> torch.Tensor:
    """Compute joint clamped-union `[M, D]` Pole-A erasure shift across K >= 2 stacked sliders."""
    device = base_rows64.device
    add_total: torch.Tensor | None = None
    parsed: list[tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, float]] = []
    all_target_units: list[torch.Tensor] = []
    mu_a_sum64: torch.Tensor | None = None
    mu_a_weight = 0.0
    e_budget_sum = 0.0

    for v_i, s_i, e_i in erasure_entries:
        s_f = float(s_i)
        vr_t = e_i[0]
        diag_t = e_i[1]
        mu_d_t = e_i[2] if len(e_i) >= 3 else None
        per_spec_holds = e_i[3] if len(e_i) >= 4 else None
        mu_a_t = e_i[4] if len(e_i) >= 5 else None
        base_vec = mu_d_t if (mu_d_t is not None and vr_t is not None) else v_i
        v64 = base_vec.detach().to(device=device, dtype=torch.float64).flatten()
        v_nrm = float(torch.linalg.vector_norm(v64).item())
        if v_nrm <= 1e-12:
            raise ValueError("control vector has zero norm; cannot apply mode='add_norm_matched_erasure'")
        a_i = (s_f * NORM_MATCHED_TARGET / v_nrm) * v64
        vr64 = vr_t.detach().to(device=device, dtype=torch.float64)
        diag64 = diag_t.detach().to(device=device, dtype=torch.float64).flatten().clamp(0.0, 1.0)
        r_use = min(int(vr64.shape[0]), int(diag64.numel()))
        vr64 = vr64[:r_use]
        diag64 = diag64[:r_use]
        if per_spec_holds:
            a_i = _apply_holds_to_shift64(a_i, per_spec_holds, device)
            vr_holds = [(hv, min(1.0, float(ha))) for hv, ha in per_spec_holds if float(ha) > 0.0]
            if vr_holds and vr64.numel() > 0:
                vr64 = _apply_holds_to_rows64(vr64, vr_holds, device)
        a_nrm = float(torch.linalg.vector_norm(a_i).item())
        u_i = (a_i.detach().cpu() / a_nrm) if a_nrm > 1e-12 else None
        if u_i is not None:
            all_target_units.append(u_i)
        add_total = a_i if add_total is None else (add_total + a_i)
        w_eff = max(0.0, min(s_f, 1.0))
        if w_eff > 0.0:
            e_budget_sum += w_eff * NORM_MATCHED_TARGET
            if r_use >= 1:
                parsed.append((u_i, vr64[:r_use].cpu(), diag64[:r_use].cpu(), w_eff))
        if use_mu_a and mu_a_t is not None and s_f > 0.0:
            ma64 = mu_a_t.detach().to(device=device, dtype=torch.float64).flatten()
            mu_a_sum64 = (s_f * ma64) if mu_a_sum64 is None else (mu_a_sum64 + s_f * ma64)
            mu_a_weight += s_f

    if add_total is None:
        return torch.zeros_like(base_rows64)
    add_rows = add_total.unsqueeze(0).expand_as(base_rows64)
    target_rms = float(torch.linalg.vector_norm(add_total).item())
    has_distinct_peers = _has_distinct_unit_peers(all_target_units)
    eff_target_rms = _resolve_stacked_target_rms(
        target_rms,
        max_stacked_rms=max_stacked_rms,
        has_distinct_peers=has_distinct_peers,
    )
    capped_add_rows = (
        add_rows * (eff_target_rms / target_rms)
        if (target_rms > 1e-12 and eff_target_rms < target_rms)
        else add_rows
    )

    w_blocks: list[torch.Tensor] = []
    for idx_j, (u_j, vr_j, diag_j, w_eff_j) in enumerate(parsed):
        if peer_ros and len(parsed) >= 2:
            peer_targets = [
                u_k
                for idx_k, (u_k, _vr_k, _d_k, _w_k) in enumerate(parsed)
                if idx_k != idx_j
                and u_k is not None
                and (u_j is None or abs(float(torch.dot(u_j, u_k).item())) < 1.0 - 1e-4)
            ]
            if peer_targets:
                ros_res = orthogonalize_erasure_against_holds(vr_j, diag_j, peer_targets)
                if ros_res is None:
                    continue
                vr_j = ros_res[0].to(dtype=torch.float64)
                diag_j = ros_res[1].to(dtype=torch.float64)
        w_blocks.append(torch.sqrt((w_eff_j * diag_j).clamp(0.0, 1.0)).unsqueeze(-1) * vr_j)

    has_mu_a = bool(use_mu_a and mu_a_sum64 is not None and mu_a_weight > 0.0)
    min_rows = 1 if has_mu_a else 2
    if not w_blocks or base_rows64.shape[0] < min_rows:
        return capped_add_rows

    w_mat_cpu = torch.cat(w_blocks, dim=0)
    if float(torch.linalg.matrix_norm(w_mat_cpu).item()) <= 1e-12:
        return capped_add_rows
    _u, s_vals_cpu, vh_cpu = torch.linalg.svd(w_mat_cpu, full_matrices=False)
    d_union_cpu = (s_vals_cpu ** 2).clamp(0.0, 1.0)
    keep = d_union_cpu > 1e-9
    if not bool(keep.any()):
        return capped_add_rows

    vr_union = vh_cpu[keep].to(device=device, dtype=torch.float64)
    d_union = d_union_cpu[keep].to(device=device, dtype=torch.float64)
    if has_mu_a:
        h_bar = (mu_a_sum64 / mu_a_weight).to(device=device, dtype=torch.float64).unsqueeze(0)
    else:
        h_bar = base_rows64.mean(dim=0, keepdim=True)
    centered = base_rows64 - h_bar
    e_own = ((centered @ vr_union.T) * d_union.unsqueeze(0)) @ vr_union
    e_own_rms = _rms_row_norm64(e_own)
    if not math.isfinite(e_own_rms) or e_own_rms <= 1e-9:
        return capped_add_rows
    e_budget = min(e_budget_sum, max(NORM_MATCHED_TARGET, eff_target_rms))
    e_cap = min(1.0, e_budget / e_own_rms) if e_budget > 0.0 else 0.0
    e = e_own * e_cap
    raw = add_rows - e
    if eff_target_rms <= 1e-6:
        return raw
    raw_rms = _rms_row_norm64(raw)
    if not math.isfinite(raw_rms) or raw_rms <= 1e-12:
        return capped_add_rows
    return raw * (eff_target_rms / raw_rms)


def apply_post_refiner(
    context: torch.Tensor,
    tags: torch.Tensor,
    vectors: Sequence[tuple[torch.Tensor, float] | tuple[torch.Tensor, float, str]],
    *,
    mode: str = "add",
    scale_mode: str = "raw",
    mask: torch.Tensor | None = None,
    mask_vision: bool = True,
    warn_if_stacked_norm: bool = True,
    token_focus: float = 1.0,
    erasure_specs: Sequence[tuple[Any, ...] | None] | None = None,
    max_stacked_rms: float | str | None = "smooth",
) -> torch.Tensor:
    """Apply `(vector_post_refiner, strength[, mode])` specs to `context` (`[B, L, 5376]`).
    Returns `context` bitwise unchanged when `s == 0` or `vectors` is empty."""
    if mode not in APPLY_MODES:
        raise ValueError(f"mode must be one of {APPLY_MODES}, got {mode!r}")
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {SCALE_MODES}, got {scale_mode!r}")
    tf_val = float(token_focus)
    if not math.isfinite(tf_val) or tf_val <= 0.0 or tf_val > 1.0:
        raise ValueError(f"token_focus must be finite in (0.0, 1.0], got {token_focus!r}")
    if context.ndim != 3 or context.shape[-1] != POST_REFINER_DIM:
        raise ValueError(
            f"context must have shape [B, L, {POST_REFINER_DIM}], got {tuple(context.shape)}"
        )
    B, L, D = context.shape
    row_mask = resolve_post_refiner_row_mask(
        tags,
        batch_size=B,
        seq_len=L,
        device=context.device,
        mask=mask,
        mask_vision=mask_vision,
    )

    if erasure_specs is not None and len(erasure_specs) != len(vectors):
        raise ValueError(
            f"erasure_specs length {len(erasure_specs)} != vectors length {len(vectors)}"
        )

    active: list[tuple[torch.Tensor, float, str, tuple[Any, ...] | None]] = []
    for idx_v, entry in enumerate(vectors):
        if len(entry) == 2:
            vec, strength = entry
            entry_mode = mode
        elif len(entry) == 3:
            vec, strength, entry_mode = entry
        else:
            raise ValueError(f"vector entry must be (vec, strength[, mode]), got {entry!r}")
        if entry_mode not in APPLY_MODES:
            raise ValueError(f"mode must be one of {APPLY_MODES}, got {entry_mode!r}")
        if not math.isfinite(strength):
            raise ValueError(f"control vector strength must be finite, got {strength!r}")
        if tuple(vec.shape) != (D,):
            raise ValueError(f"vector must have shape ({D},), got {tuple(vec.shape)}")
        if not bool(torch.isfinite(vec).all()):
            raise ValueError("control vector contains non-finite values")
        v64_chk = vec.detach().to(dtype=torch.float64)
        v_norm_sq = float(torch.dot(v64_chk, v64_chk).item())
        e_spec = erasure_specs[idx_v] if erasure_specs is not None else None
        has_per_spec_holds = bool(e_spec is not None and len(e_spec) >= 4 and e_spec[3])
        has_active_erasure = (
            entry_mode == "add_norm_matched_erasure"
            and (
                (
                    float(strength) > 0.0
                    and e_spec is not None
                    and e_spec[0] is not None
                    and e_spec[1] is not None
                )
                or has_per_spec_holds
            )
        )
        use_norm_match = (
            entry_mode == "add_norm_matched"
            or (entry_mode == "add" and scale_mode == "norm_matched")
            or (entry_mode == "add_norm_matched_erasure" and not has_active_erasure)
        )
        if (
            entry_mode in ("hold", "project_out", "amplify", "add_norm_matched_erasure")
            or use_norm_match
        ) and v_norm_sq <= 1e-24:
            raise ValueError(f"control vector has zero norm; cannot apply mode={entry_mode!r}")
        if strength != 0.0:
            if has_active_erasure:
                active.append((vec, float(strength), "add_norm_matched_erasure", e_spec))
            elif use_norm_match:
                eff_strength = float(strength) * (NORM_MATCHED_TARGET / math.sqrt(v_norm_sq))
                active.append((vec, eff_strength, "add", None))
            else:
                active.append((vec, float(strength), entry_mode, None))

    if not active or not bool(row_mask.any()):
        return context

    if not bool(torch.isfinite(context[row_mask]).all()):
        raise ValueError("context contains non-finite values in steered rows")

    out = context.clone()
    if all(em in ("add", "hold") for _, _, em, _ in active):
        total_shift64 = torch.zeros(D, dtype=torch.float64, device=context.device)
        hold_entries: list[tuple[torch.Tensor, float]] = []
        n_add = 0
        for vec, strength, em, _ in active:
            if em == "add":
                n_add += 1
                total_shift64 = total_shift64 + strength * vec.detach().to(
                    device=context.device, dtype=torch.float64
                )
            else:
                hold_entries.append((vec, strength))
        if hold_entries:
            total_shift64 = _apply_holds_to_shift64(total_shift64, hold_entries, context.device)
        shift_norm = float(torch.linalg.vector_norm(total_shift64).item())
        if warn_if_stacked_norm and n_add >= 2 and shift_norm >= STACK_SHIFT_WARN_NORM:
            _logger.warning(
                "[h3] apply_post_refiner stacked %d additive vectors to combined shift norm=%.2f "
                "(threshold=%.1f); lower per-vector strength to avoid prompt saturation",
                n_add,
                shift_norm,
                STACK_SHIFT_WARN_NORM,
            )
        if shift_norm > 0.0:
            eff_mask = row_mask
            if tf_val < 1.0 - 1e-9:
                eff_mask, _ = focus_row_mask(context, row_mask, total_shift64, token_focus=tf_val)
            shift = total_shift64.to(dtype=context.dtype)
            out[eff_mask] = out[eff_mask] + shift
    else:
        eff_mask = row_mask
        if tf_val < 1.0 - 1e-9:
            add_shift64 = torch.zeros(D, dtype=torch.float64, device=context.device)
            hold_aim: list[tuple[torch.Tensor, float]] = []
            has_add_aim = False
            for v_a, s_a, em_a, e_a in active:
                if em_a == "add":
                    has_add_aim = True
                    add_shift64 = add_shift64 + s_a * v_a.detach().to(device=context.device, dtype=torch.float64)
                elif em_a == "add_norm_matched_erasure":
                    has_add_aim = True
                    base_v = (
                        e_a[2]
                        if (e_a is not None and s_a > 0.0 and len(e_a) >= 3 and e_a[2] is not None)
                        else v_a
                    )
                    bv64 = base_v.detach().to(device=context.device, dtype=torch.float64)
                    bn = max(float(torch.linalg.vector_norm(bv64).item()), 1e-12)
                    term64 = (s_a * NORM_MATCHED_TARGET / bn) * bv64
                    if e_a is not None and len(e_a) >= 4 and e_a[3]:
                        term64 = _apply_holds_to_shift64(term64, e_a[3], context.device)
                    add_shift64 = add_shift64 + term64
                elif em_a == "hold":
                    hold_aim.append((v_a, s_a))
            if has_add_aim:
                if hold_aim:
                    add_shift64 = _apply_holds_to_shift64(add_shift64, hold_aim, context.device)
                aim_vec = add_shift64
            else:
                aim_vec = active[0][0]
            eff_mask, _ = focus_row_mask(context, row_mask, aim_vec, token_focus=tf_val)

        warned_stacked = False
        for b_idx in range(B):
            b_mask = eff_mask[b_idx]
            if not bool(b_mask.any()):
                continue
            base_rows64 = context[b_idx, b_mask].to(dtype=torch.float64)
            rows64 = out[b_idx, b_mask].to(dtype=torch.float64)
            idx = 0
            while idx < len(active):
                vec, strength, em, _ = active[idx]
                if em in ("add", "add_norm_matched_erasure", "hold"):
                    block_shift64 = torch.zeros_like(rows64)
                    hold_entries_blk: list[tuple[torch.Tensor, float]] = []
                    prompt_er_entries: list[tuple[torch.Tensor, float, tuple[Any, ...]]] = []
                    vision_er_entries: list[tuple[torch.Tensor, float, tuple[Any, ...]]] = []
                    other_add_items: list[tuple[torch.Tensor, float, str, tuple[Any, ...] | None]] = []
                    n_add_blk = 0
                    while idx < len(active) and active[idx][2] in ("add", "add_norm_matched_erasure", "hold"):
                        v_i, s_i, em_i, e_i = active[idx]
                        if em_i == "add":
                            n_add_blk += 1
                            other_add_items.append((v_i, s_i, em_i, e_i))
                        elif em_i == "add_norm_matched_erasure":
                            n_add_blk += 1
                            vr_chk = e_i[0] if (e_i is not None and len(e_i) >= 1) else None
                            diag_chk = e_i[1] if (e_i is not None and len(e_i) >= 2) else None
                            mu_a_chk = e_i[4] if (e_i is not None and len(e_i) >= 5) else None
                            if (
                                s_i > 0.0
                                and vr_chk is not None
                                and diag_chk is not None
                                and mu_a_chk is None
                                and base_rows64.shape[0] >= 2
                            ):
                                prompt_er_entries.append((v_i, s_i, e_i))
                            elif (
                                s_i > 0.0
                                and vr_chk is not None
                                and diag_chk is not None
                                and mu_a_chk is not None
                                and base_rows64.shape[0] >= 1
                            ):
                                vision_er_entries.append((v_i, s_i, e_i))
                            else:
                                other_add_items.append((v_i, s_i, em_i, e_i))
                        else:
                            hold_entries_blk.append((v_i, s_i))
                        idx += 1
                    iter_add_items = list(other_add_items)
                    if len(prompt_er_entries) >= 2:
                        block_shift64 = block_shift64 + _stacked_erasure_block_shift64(
                            base_rows64,
                            prompt_er_entries,
                            use_mu_a=False,
                            peer_ros=True,
                            max_stacked_rms=max_stacked_rms,
                        )
                    else:
                        for v_p, s_p, e_p in prompt_er_entries:
                            iter_add_items.append((v_p, s_p, "add_norm_matched_erasure", e_p))
                    if len(vision_er_entries) >= 2:
                        block_shift64 = block_shift64 + _stacked_erasure_block_shift64(
                            base_rows64,
                            vision_er_entries,
                            use_mu_a=True,
                            peer_ros=True,
                            max_stacked_rms=max_stacked_rms,
                        )
                    else:
                        for v_v, s_v, e_v in vision_er_entries:
                            iter_add_items.append((v_v, s_v, "add_norm_matched_erasure", e_v))
                    for v_i, s_i, em_i, e_i in iter_add_items:
                        if em_i == "add":
                            block_shift64 = block_shift64 + (
                                s_i * v_i.detach().to(device=context.device, dtype=torch.float64)
                            ).unsqueeze(0)
                        else:
                            block_shift64 = block_shift64 + _erasure_row_shift64(
                                base_rows64, v_i, s_i, e_i
                            )
                    if hold_entries_blk:
                        block_shift64 = _apply_holds_to_rows64(
                            block_shift64, hold_entries_blk, context.device
                        )
                    block_rms = _rms_row_norm64(block_shift64)
                    if (
                        isinstance(max_stacked_rms, (int, float))
                        and not isinstance(max_stacked_rms, bool)
                        and float(max_stacked_rms) > 0.0
                        and n_add_blk >= 2
                        and block_rms > float(max_stacked_rms)
                    ):
                        block_shift64 = block_shift64 * (float(max_stacked_rms) / block_rms)
                        block_rms = float(max_stacked_rms)
                    if (
                        warn_if_stacked_norm
                        and not warned_stacked
                        and n_add_blk >= 2
                        and block_rms >= STACK_SHIFT_WARN_NORM
                    ):
                        warned_stacked = True
                        _logger.warning(
                            "[h3] apply_post_refiner stacked %d additive vectors to combined RMS shift norm=%.2f "
                            "(threshold=%.1f); lower per-vector strength to avoid prompt saturation",
                            n_add_blk,
                            block_rms,
                            STACK_SHIFT_WARN_NORM,
                        )
                    rows64 = rows64 + block_shift64
                else:
                    v64 = vec.detach().to(device=context.device, dtype=torch.float64)
                    v_norm_sq = float(torch.dot(v64, v64).item())
                    coeff = (rows64 @ v64) / v_norm_sq
                    delta = (strength * coeff).unsqueeze(-1) * v64.unsqueeze(0)
                    if em == "project_out":
                        rows64 = rows64 - delta
                    else:
                        rows64 = rows64 + delta
                    idx += 1
            out[b_idx, b_mask] = rows64.to(dtype=context.dtype)

    if out.shape != (B, L, D):
        raise RuntimeError(f"apply_post_refiner changed context shape from {(B, L, D)} to {tuple(out.shape)}")
    return out
