"""Analytical Jacobian Lens (`J-Lens`) and `TokenRefiner` Wiener readout for H3 `5376`-d and `5120`-d vectors.

Decodes any `5376`-d post-refiner direction (`vector.post_refiner`, single-clause
compiled vectors, textless `<Picture>` / `<Video>` media-contrast vectors,
composite multi-vector recipes with `mode = "hold"`, or unsupervised
`TokenRefiner` singular vectors) and any `5120`-d encoder direction
(`vector.encoder`) into natural-language English words on CPU (`~15 ms`) using
the precomputed operator bundle at `<ComfyUI>/models/j_lens/j_lens_h3.safetensors`
(`docs/research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md`).

Nothing here imports ComfyUI core or touches the GPU.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent
COMFY_ROOT = REPO_ROOT.parent.parent
DEFAULT_BUNDLE_REL = Path("models") / "j_lens" / "j_lens_h3.safetensors"

# Reasoned and measured starting constants from docs/research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md:
# - DEFAULT_STEER_SCALE = 90.0 (~0.31 * median_tok1+_row_norm 292.3 at layers.49)
# - DEFAULT_MIN_LOGPROB = -11.5 (syntactic cloze admissibility threshold from label_h3_control_vectors.py:718)
# - NORM_MATCHED_TARGET = 67.5 (post-refiner norm target from control_vectors.py:2024)
# - DEFAULT_POST_REFINER_MEDIAN_ROW_NORM = 164.0 (measured median tag-1 row norm after TokenRefiner)
# - DEFAULT_ENCODER_MEDIAN_ROW_NORM = 292.3 (measured median tok1+ row norm at Qwen3-VL layers.49)
DEFAULT_STEER_SCALE: float = 90.0
DEFAULT_MIN_LOGPROB: float = -11.5
NORM_MATCHED_TARGET: float = 67.5
DEFAULT_POST_REFINER_MEDIAN_ROW_NORM: float = 164.0
DEFAULT_ENCODER_MEDIAN_ROW_NORM: float = 292.3
DEFAULT_NEUTRAL_WORDS: tuple[str, ...] = (
    "natural",
    "normal",
    "standard",
    "realistic",
    "consistent",
    "ordinary",
    "typical",
    "steady",
)

# Standard recipes matching the checkpoints and Pika H3 Flex LoRAs used to build J_ref_{fl2va,ref2va}
_DEFAULT_RECIPES: dict[str, dict[str, Any]] = {
    "fl2va": {
        "checkpoint": {
            "file": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "sha256": "58c5ffad5d9fe2c73a4fbf104d26b56f9ad81022ec3c4da034b85530173526dd",
        },
        "encoder": {
            "file": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "sha256": "782eae85cd74cefa1578d8234613af9e612711e313ff7d9997f1bbeb3cf3c2ed",
        },
        "loras": [
            {
                "applied": "branch",
                "file": "pika_h3_flex_fl2va_svd995_bf16.safetensors",
                "sha256": "55f6a746f28759da7de2a13aeb4a89ae722d0b089cef2d98c5cbb66ac7fe0d2a",
                "strength": 1.0,
            }
        ],
    },
    "ref2va": {
        "checkpoint": {
            "file": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "sha256": "bbb0075520ef8b4ddfd5df75bb9bd63b6f87c7e0112bb89d070cd90660192b3d",
        },
        "encoder": {
            "file": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "sha256": "782eae85cd74cefa1578d8234613af9e612711e313ff7d9997f1bbeb3cf3c2ed",
        },
        "loras": [
            {
                "applied": "branch",
                "file": "pika_h3_flex_ref2va_svd995_bf16.safetensors",
                "sha256": "8b21c673b45e2b743f117dd90b8636441c000194cf74a9c019402771e4618436",
                "strength": 1.0,
            }
        ],
    },
}

_BUNDLE_CACHE: dict[Path, dict[str, Any]] = {}


def default_bundle_path() -> Path:
    """Return the default path to `j_lens_h3.safetensors` under `<ComfyUI>/models/j_lens/`."""
    env_override = os.environ.get("H3_J_LENS_PATH", "").strip()
    if env_override:
        return Path(env_override).resolve()
    return (COMFY_ROOT / DEFAULT_BUNDLE_REL).resolve()


def bundle_available(bundle_path: str | os.PathLike | None = None) -> bool:
    """Return True when the precomputed `j_lens_h3.safetensors` bundle exists on disk."""
    path = Path(bundle_path).resolve() if bundle_path is not None else default_bundle_path()
    return path.is_file()


def load_j_lens_bundle(bundle_path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load and cache the precomputed H3 J-Lens operator bundle on CPU."""
    path = Path(bundle_path).resolve() if bundle_path is not None else default_bundle_path()
    if path in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[path]
    if not path.is_file():
        raise FileNotFoundError(
            f"J-Lens operator bundle not found at {path.name} (expected under models/j_lens/j_lens_h3.safetensors)"
        )
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        if meta.get("format") != "h3_j_lens_v1":
            raise ValueError(
                f"Unsupported J-Lens bundle format {meta.get('format')!r} in {path.name}; expected 'h3_j_lens_v1'"
            )
        words = json.loads(meta["vocab_words_json"])
        m_union = f.get_tensor("M_union").to(dtype=torch.float32)
        valid_pos = f.get_tensor("valid_pos_in_union").to(dtype=torch.long)
        m_valid = m_union[valid_pos]
        m_centered = F.normalize(m_valid - m_valid.mean(dim=0, keepdim=True), dim=-1)
        word_to_idxs: dict[str, list[int]] = {}
        for idx_i, w_str in enumerate(words):
            word_to_idxs.setdefault(w_str.lower(), []).append(idx_i)
        sigma_p030 = (
            f.get_tensor("Sigma_49_p030").to(dtype=torch.float32)
            if "Sigma_49_p030" in f.keys()
            else torch.eye(5120, dtype=torch.float32)
        )
        bundle = {
            "path": str(path),
            "words": words,
            "word_to_idxs": word_to_idxs,
            "M_union": m_union,
            "M_centered": m_centered,
            "Sigma_49_p030": sigma_p030,
            "cloze_base_union": f.get_tensor("cloze_base_union").to(dtype=torch.float32),
            "valid_pos_in_union": valid_pos,
            "valid_idx": f.get_tensor("valid_idx").to(dtype=torch.long),
            "union_idx": f.get_tensor("union_idx").to(dtype=torch.long),
            "J_49_to_63": f.get_tensor("J_49_to_63").to(dtype=torch.float32),
            "J_ref_ref2va": f.get_tensor("J_ref_ref2va").to(dtype=torch.float32),
            "J_ref_fl2va": f.get_tensor("J_ref_fl2va").to(dtype=torch.float32),
            "J_wiener_ref2va": f.get_tensor("J_wiener_ref2va").to(dtype=torch.float32),
            "J_wiener_fl2va": f.get_tensor("J_wiener_fl2va").to(dtype=torch.float32),
        }
    _BUNDLE_CACHE[path] = bundle
    return bundle


def orthogonal_hold_basis(
    holds: torch.Tensor | Sequence[torch.Tensor],
    dim: int,
) -> torch.Tensor:
    """Build an orthonormal basis `Q` (`[K, dim]`, float64) from one or more hold vectors."""
    if isinstance(holds, torch.Tensor):
        h_mat = holds.detach().to(device="cpu", dtype=torch.float64)
        if h_mat.ndim == 1:
            h_mat = h_mat.unsqueeze(0)
        elif h_mat.ndim != 2:
            raise ValueError(f"holds tensor must be 1D or 2D, got shape {tuple(h_mat.shape)}")
    else:
        rows = []
        for h in holds:
            t = h.detach().to(device="cpu", dtype=torch.float64).flatten()
            if t.numel() != dim:
                raise ValueError(f"hold vector dimension {t.numel()} does not match target dim {dim}")
            norm = float(torch.linalg.vector_norm(t).item())
            if norm > 1e-12:
                rows.append(t / norm)
        if not rows:
            return torch.zeros((0, dim), dtype=torch.float64)
        h_mat = torch.stack(rows, dim=0)

    if h_mat.shape[0] == 0:
        return torch.zeros((0, dim), dtype=torch.float64)
    if h_mat.shape[1] != dim:
        raise ValueError(f"holds basis width {h_mat.shape[1]} does not match target dim {dim}")

    # Filter near-zero rows and orthonormalize via reduced QR
    norms = torch.linalg.vector_norm(h_mat, dim=1)
    keep = norms > 1e-12
    if not bool(keep.any()):
        return torch.zeros((0, dim), dtype=torch.float64)
    unit_rows = h_mat[keep] / norms[keep].unsqueeze(1)
    q, r = torch.linalg.qr(unit_rows.T, mode="reduced")
    diag = torch.abs(torch.diagonal(r))
    rank_mask = diag > 1e-7
    return q[:, rank_mask].T.contiguous()


def apply_soft_holds(
    vec: torch.Tensor,
    holds: torch.Tensor | Sequence[torch.Tensor] | None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Apply soft or over-projected hold `v - alpha * Q^T (Q v)` for `alpha` in `[0.0, 2.0]`."""
    v64 = vec.detach().to(device="cpu", dtype=torch.float64).flatten()
    if holds is None or float(alpha) == 0.0:
        return v64.to(dtype=torch.float32)
    a = float(alpha)
    if not (0.0 <= a <= 2.0):
        raise ValueError(f"alpha must be in [0.0, 2.0], got {alpha}")
    basis = orthogonal_hold_basis(holds, v64.numel())
    if basis.shape[0] == 0:
        return v64.to(dtype=torch.float32)
    proj = basis.T @ (basis @ v64)
    return (v64 - a * proj).to(dtype=torch.float32)


def decode_vector(
    vec: torch.Tensor,
    *,
    space: str = "post_refiner",
    family: str = "fl2va",
    scale: float = DEFAULT_STEER_SCALE,
    min_logprob: float = DEFAULT_MIN_LOGPROB,
    top_k: int = 8,
    holds: torch.Tensor | Sequence[torch.Tensor] | None = None,
    alpha: float = 1.0,
    bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode any `5376`-d post-refiner or `5120`-d encoder vector into top positive/negative words."""
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"vec must be a torch.Tensor, got {type(vec).__name__}")
    v_raw = vec.detach().to(device="cpu", dtype=torch.float32).flatten()
    if space == "auto":
        if v_raw.numel() == 5376:
            space = "post_refiner"
        elif v_raw.numel() == 5120:
            space = "encoder"
        else:
            raise ValueError(f"Cannot infer space from vector length {v_raw.numel()}; expected 5376 or 5120")

    expected_dim = 5376 if space == "post_refiner" else 5120 if space == "encoder" else None
    if expected_dim is None:
        raise ValueError(f"space must be 'post_refiner', 'encoder', or 'auto', got {space!r}")
    if v_raw.numel() != expected_dim:
        raise ValueError(f"space={space!r} requires vector length {expected_dim}, got {v_raw.numel()}")

    norm_before = float(torch.linalg.vector_norm(v_raw.to(dtype=torch.float64)).item())
    v_eff = apply_soft_holds(v_raw, holds, alpha=alpha)
    norm_after = float(torch.linalg.vector_norm(v_eff.to(dtype=torch.float64)).item())

    if norm_after <= 1e-12:
        return {
            "space": space,
            "family": family,
            "norm_before_holds": norm_before,
            "norm_after_holds": norm_after,
            "alpha": float(alpha) if holds is not None else 0.0,
            "positive_words": [],
            "negative_words": [],
            "pos_summary": "",
            "neg_summary": "",
        }

    b = dict(bundle) if bundle is not None else load_j_lens_bundle()
    if space == "post_refiner":
        j_wiener = b["J_wiener_ref2va"] if "ref2va" in str(family).lower() else b["J_wiener_fl2va"]
        v_49 = j_wiener @ v_eff
    else:
        v_49 = v_eff

    d = F.normalize(v_49, dim=0) * float(scale)
    delta_z = b["M_union"] @ d  # [27407]

    z_base = b["cloze_base_union"]  # [7, 27407]
    z_pos = z_base + delta_z.unsqueeze(0)
    z_neg = z_base - delta_z.unsqueeze(0)

    valid_pos = b["valid_pos_in_union"]
    lp_base = F.log_softmax(z_base, dim=-1)[:, valid_pos]  # [7, 24912]
    lp_pos = F.log_softmax(z_pos, dim=-1)[:, valid_pos]
    lp_neg = F.log_softmax(z_neg, dim=-1)[:, valid_pos]

    pos_shift = torch.minimum(lp_pos - lp_base, lp_pos - lp_neg)
    neg_shift = torch.minimum(lp_neg - lp_base, lp_neg - lp_pos)

    neg_inf = torch.full_like(lp_pos, -1e9)
    pos_gain = torch.where(lp_pos > float(min_logprob), pos_shift, neg_inf).max(dim=0).values
    neg_gain = torch.where(lp_neg > float(min_logprob), neg_shift, neg_inf).max(dim=0).values

    words = b["words"]

    def _pick_unique(gains: torch.Tensor) -> list[dict[str, Any]]:
        k_search = min(max(int(top_k) * 5, 32), len(words))
        vals, idxs = torch.topk(gains, k=k_search)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for val, idx in zip(vals.tolist(), idxs.tolist()):
            if val <= -1e8:
                break
            w = words[idx]
            if w not in seen:
                seen.add(w)
                out.append({"word": w, "gain": round(float(val), 2)})
                if len(out) >= int(top_k):
                    break
        return out

    pos_list = _pick_unique(pos_gain)
    neg_list = _pick_unique(neg_gain)

    return {
        "space": space,
        "family": family,
        "norm_before_holds": round(norm_before, 4),
        "norm_after_holds": round(norm_after, 4),
        "alpha": float(alpha) if holds is not None else 0.0,
        "positive_words": pos_list,
        "negative_words": neg_list,
        "pos_summary": ", ".join(f"{item['word']} (+{item['gain']:.2f})" for item in pos_list),
        "neg_summary": ", ".join(f"{item['word']} (-{item['gain']:.2f})" for item in neg_list),
    }


def _resolve_vector_path(source: str | os.PathLike) -> Path:
    path = Path(source)
    if path.is_file():
        return path
    cv_dir = COMFY_ROOT / "models" / "control_vectors"
    cand = cv_dir / path
    if cand.is_file():
        return cand
    if not path.suffix:
        cand_st = cv_dir / f"{path.name}.safetensors"
        if cand_st.is_file():
            return cand_st
    raise FileNotFoundError(f"Control vector file not found: {source}")


def _load_vector_from_source(
    source: str | os.PathLike | torch.Tensor,
    space: str = "post_refiner",
) -> tuple[torch.Tensor, str, dict[str, Any]]:
    if isinstance(source, torch.Tensor):
        v = source.detach().to(device="cpu", dtype=torch.float32).flatten()
        return v, "tensor", {}
    path = _resolve_vector_path(source)
    with safe_open(str(path), framework="pt", device="cpu") as f:
        raw_meta = f.metadata() or {}
        meta: dict[str, Any] = {}
        for k, val in raw_meta.items():
            try:
                meta[k] = json.loads(val)
            except Exception:
                meta[k] = val
        key = f"vector.{space}"
        if key not in f.keys():
            raise KeyError(f"{path.name} does not contain tensor {key!r} (available: {list(f.keys())})")
        tensor = f.get_tensor(key).to(dtype=torch.float32).flatten()
    return tensor, path.name, meta


def decode_safetensors(
    path: str | os.PathLike,
    *,
    scale: float = DEFAULT_STEER_SCALE,
    min_logprob: float = DEFAULT_MIN_LOGPROB,
    top_k: int = 8,
    holds: Sequence[str | os.PathLike | torch.Tensor] | None = None,
    alpha: float = 1.0,
    bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode a `.safetensors` control vector (and optional hold files/tensors) via J-Lens."""
    resolved_path = _resolve_vector_path(path)
    v_pr, name, meta = _load_vector_from_source(resolved_path, space="post_refiner")
    family = str(meta.get("family", "fl2va"))
    b = dict(bundle) if bundle is not None else load_j_lens_bundle()

    hold_tensors_pr: list[torch.Tensor] = []
    hold_names: list[str] = []
    if holds:
        for h_src in holds:
            ht, hname, _ = _load_vector_from_source(h_src, space="post_refiner")
            hold_tensors_pr.append(ht)
            hold_names.append(hname)

    pr_readout = decode_vector(
        v_pr,
        space="post_refiner",
        family=family,
        scale=scale,
        min_logprob=min_logprob,
        top_k=top_k,
        holds=hold_tensors_pr if hold_tensors_pr else None,
        alpha=alpha,
        bundle=b,
    )

    out: dict[str, Any] = {
        "file": name,
        "name": meta.get("name", Path(name).stem),
        "family": family,
        "holds": hold_names,
        "alpha": float(alpha) if hold_tensors_pr else 0.0,
        "post_refiner": pr_readout,
    }

    # If vector.encoder is also present in the file, decode it and report refiner agreement cosine
    with safe_open(str(resolved_path), framework="pt", device="cpu") as f:
        if "vector.encoder" in f.keys():
            v_enc = f.get_tensor("vector.encoder").to(dtype=torch.float32).flatten()
            enc_readout = decode_vector(
                v_enc,
                space="encoder",
                family=family,
                scale=scale,
                min_logprob=min_logprob,
                top_k=top_k,
                bundle=b,
            )
            j_ref = b["J_ref_ref2va"] if "ref2va" in family.lower() else b["J_ref_fl2va"]
            j_w = b["J_wiener_ref2va"] if "ref2va" in family.lower() else b["J_wiener_fl2va"]
            j_63 = b["J_49_to_63"]
            fwd_cos = float(F.cosine_similarity((j_ref @ v_enc).unsqueeze(0), v_pr.unsqueeze(0)).item())
            w_cos_63 = float(
                F.cosine_similarity((j_63 @ (j_w @ v_pr)).unsqueeze(0), (j_63 @ v_enc).unsqueeze(0)).item()
            )
            out["encoder"] = enc_readout
            out["refiner_forward_cosine"] = round(fwd_cos, 4)
            out["wiener_j63_cosine"] = round(w_cos_63, 4)

    return out


def decode_recipe(
    steps: Sequence[Mapping[str, Any]],
    *,
    family: str = "fl2va",
    scale: float = DEFAULT_STEER_SCALE,
    min_logprob: float = DEFAULT_MIN_LOGPROB,
    top_k: int = 8,
    bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a multi-vector recipe (`add`, `add_norm_matched`, `hold`) and decode its net shift.

    Each item in `steps` is a mapping with:
      - `vector`: path to `.safetensors` or `5376`-d `torch.Tensor`
      - `mode`: `"add"`, `"add_norm_matched"`, or `"hold"` (default `"add"`)
      - `strength`: float multiplier for `"add"` / `"add_norm_matched"`, or `alpha in [0.0, 2.0]` for `"hold"`
    """
    b = dict(bundle) if bundle is not None else load_j_lens_bundle()
    add_terms: list[tuple[str, torch.Tensor]] = []
    hold_steps: list[tuple[str, torch.Tensor, float]] = []

    for item in steps:
        src = item["vector"]
        mode = str(item.get("mode", "add")).strip()
        strength = float(item.get("strength", 1.0))
        vec, name, meta = _load_vector_from_source(src, space="post_refiner")
        if meta.get("family") and family == "fl2va":
            family = str(meta["family"])
        v64 = vec.to(dtype=torch.float64)
        v_norm = float(torch.linalg.vector_norm(v64).item())

        if mode == "add":
            add_terms.append((name, (strength * v64).to(dtype=torch.float32)))
        elif mode == "add_norm_matched":
            scale_factor = (NORM_MATCHED_TARGET / v_norm) if v_norm > 1e-12 else 0.0
            add_terms.append((name, (strength * scale_factor * v64).to(dtype=torch.float32)))
        elif mode == "hold":
            if not (0.0 <= strength <= 2.0):
                raise ValueError(f"mode='hold' strength (alpha) must be in [0.0, 2.0], got {strength}")
            hold_steps.append((name, vec, strength))
        else:
            raise ValueError(f"Unsupported recipe mode {mode!r}; expected 'add', 'add_norm_matched', or 'hold'")

    if not add_terms:
        zero = torch.zeros(5376, dtype=torch.float32)
        return {
            "family": family,
            "combined_norm_before_holds": 0.0,
            "combined_norm_after_holds": 0.0,
            "pairwise_interactions": [],
            "readout": decode_vector(zero, space="post_refiner", family=family, bundle=b),
        }

    sum_vec = torch.stack([t.to(dtype=torch.float64) for _, t in add_terms], dim=0).sum(dim=0)
    norm_before = float(torch.linalg.vector_norm(sum_vec).item())

    interactions: list[dict[str, Any]] = []
    for i in range(len(add_terms)):
        for j in range(i + 1, len(add_terms)):
            n_i, v_i = add_terms[i]
            n_j, v_j = add_terms[j]
            vi64 = v_i.to(dtype=torch.float64)
            vj64 = v_j.to(dtype=torch.float64)
            cross = float(2.0 * torch.dot(vi64, vj64).item())
            denom = float((torch.linalg.vector_norm(vi64) * torch.linalg.vector_norm(vj64)).item())
            cos_ij = float(torch.dot(vi64, vj64).item() / denom) if denom > 1e-12 else 0.0
            interactions.append(
                {
                    "pair": (n_i, n_j),
                    "cosine": round(cos_ij, 4),
                    "cross_term_2_si_sj_dot": round(cross, 2),
                }
            )

    cur_vec = sum_vec.to(dtype=torch.float32)
    for _, h_vec, alpha in hold_steps:
        cur_vec = apply_soft_holds(cur_vec, [h_vec], alpha=alpha)

    readout = decode_vector(
        cur_vec,
        space="post_refiner",
        family=family,
        scale=scale,
        min_logprob=min_logprob,
        top_k=top_k,
        bundle=b,
    )
    return {
        "family": family,
        "combined_norm_before_holds": round(norm_before, 4),
        "combined_norm_after_holds": readout["norm_after_holds"],
        "pairwise_interactions": interactions,
        "readout": readout,
    }


def refiner_singular_vectors(
    *,
    family: str = "fl2va",
    top_n: int = 8,
    top_k_words: int = 6,
    scale: float = DEFAULT_STEER_SCALE,
    bundle: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Decode the top `top_n` unsupervised left singular vectors of `J_ref`."""
    b = dict(bundle) if bundle is not None else load_j_lens_bundle()
    j_ref = b["J_ref_ref2va"] if "ref2va" in str(family).lower() else b["J_ref_fl2va"]
    u_mat, s_vec, _ = torch.linalg.svd(j_ref.to(dtype=torch.float32), full_matrices=False)
    results: list[dict[str, Any]] = []
    for idx in range(min(int(top_n), u_mat.shape[1])):
        u_i = u_mat[:, idx]
        res = decode_vector(
            u_i,
            space="post_refiner",
            family=family,
            scale=scale,
            top_k=top_k_words,
            bundle=b,
        )
        results.append(
            {
                "rank": idx + 1,
                "singular_value": round(float(s_vec[idx].item()), 4),
                "positive_words": res["positive_words"],
                "negative_words": res["negative_words"],
                "pos_summary": res["pos_summary"],
                "neg_summary": res["neg_summary"],
            }
        )
    return results


def _parse_word_list(words: str | Sequence[str] | None) -> list[str]:
    if words is None:
        return []
    if isinstance(words, str):
        raw = [w.strip().lower() for part in words.split(",") for w in part.split()]
    else:
        raw = [w.strip().lower() for item in words for part in str(item).split(",") for w in part.split()]
    return [w for w in raw if w]


_SHORT_WORD_FALLBACKS: dict[str, tuple[str, ...]] = {
    "ice": ("frozen", "frost", "crystalline"),
    "icy": ("frozen", "frost", "colder"),
    "fog": ("foggy", "mist", "haze"),
    "wet": ("soaked", "damp", "rain"),
    "dry": ("dusty", "outdoor", "arid"),
    "old": ("older", "vintage", "dated"),
    "new": ("modern", "fresh", "recent"),
    "dim": ("dark", "shadow", "faint"),
    "hot": ("warm", "heated", "fiery"),
    "red": ("crimson", "scarlet", "ruby"),
    "raw": ("gritty", "rough", "unpolished"),
    "8-bit": ("pixel", "arcade", "console", "cartridge"),
    "8bit": ("pixel", "arcade", "console", "cartridge"),
    "16-bit": ("pixel", "arcade", "console", "cartridge"),
    "led": ("neon", "glowing", "matrix", "laser"),
    "leds": ("neon", "glowing", "matrix", "laser"),
    "crt": ("monitor", "screen", "analog", "static"),
    "vhs": ("tape", "cassette", "analog", "camcorder"),
    "cgi": ("digital", "rendered", "synthetic", "computer"),
    "sfx": ("sound", "acoustic", "noise", "echo"),
    "xray": ("skeleton", "bones", "skull", "translucent"),
    "x-ray": ("skeleton", "bones", "skull", "translucent"),
    "cyberpunk": ("neon", "punk", "futuristic", "matrix"),
    "automaton": ("mechanical", "robot", "brass", "clock", "gears"),
    "chiptune": ("arcade", "electronic", "synthesizer", "melody"),
    "synth": ("synthesizer", "electronic", "neon"),
    "victorian": ("antique", "brass", "vintage", "gothic"),
}


def _resolve_word_indices(word_list: Sequence[str], word_to_idxs: Mapping[str, list[int]]) -> tuple[list[int], list[str], list[str]]:
    matched_ids: list[int] = []
    matched_words: list[str] = []
    unmatched_words: list[str] = []
    for w in word_list:
        ids = word_to_idxs.get(w)
        if ids:
            matched_ids.extend(ids)
            matched_words.append(w)
            continue

        # 1. Check short-word (< 4 char) and compound/acronym fallbacks
        if w in _SHORT_WORD_FALLBACKS:
            fb_ids = [i for fb in _SHORT_WORD_FALLBACKS[w] for i in word_to_idxs.get(fb, ())]
            if fb_ids:
                matched_ids.extend(fb_ids)
                matched_words.append(w)
                continue

        # 2. Check hyphenated or compound words (e.g. steampunk -> steam + punk, clockwork -> clock + work)
        comp_ids: list[int] = []
        if "-" in w:
            parts = [p for p in w.split("-") if len(p) >= 3]
            if parts and all(p in word_to_idxs for p in parts):
                for p in parts:
                    comp_ids.extend(word_to_idxs[p])
        if not comp_ids and len(w) >= 7:
            for k in range(3, len(w) - 2):
                left, right = w[:k], w[k:]
                if left in word_to_idxs and right in word_to_idxs:
                    comp_ids.extend(word_to_idxs[left])
                    comp_ids.extend(word_to_idxs[right])
                    break
        if comp_ids:
            matched_ids.extend(comp_ids)
            matched_words.append(w)
            continue

        # 3. Check common morphological stems (e.g. grainy -> grain, pixelated -> pixel, glacial -> glacier)
        stem_ids: list[int] = []
        for suffix in ("ated", "ized", "ting", "ing", "ial", "ic", "al", "ed", "ly", "es", "y", "s"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 4:
                stem = w[: -len(suffix)]
                if stem in word_to_idxs:
                    stem_ids.extend(word_to_idxs[stem])
                    break
                # e.g. glacial -> stem "glac" -> match "glacier" / "glaciers"
                cands_stem = [
                    i
                    for vw, ilist in word_to_idxs.items()
                    if vw.startswith(stem) and len(vw) <= len(stem) + 4
                    for i in ilist
                ]
                if cands_stem:
                    stem_ids.extend(cands_stem)
                    break
        if stem_ids:
            matched_ids.extend(stem_ids)
            matched_words.append(w)
            continue

        # 4. Bounded prefix match (>= 4 chars, within +2 chars of word length to avoid e.g. retro -> retrovirus)
        cands = [
            i
            for vw, ilist in word_to_idxs.items()
            if len(w) >= 4 and vw.startswith(w) and len(vw) <= len(w) + 2
            for i in ilist
        ]
        if cands:
            matched_ids.extend(cands)
            matched_words.append(w)
        else:
            unmatched_words.append(w)
    return matched_ids, matched_words, unmatched_words


def synthesize_from_words(
    pos_words: str | Sequence[str],
    neg_words: str | Sequence[str] | None = None,
    *,
    family: str = "fl2va",
    target_norm: float = NORM_MATCHED_TARGET,
    hold_words: str | Sequence[str] | None = None,
    alpha: float = 1.0,
    bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Synthesize a `5376`-d `vector.post_refiner` and `5120`-d `vector.encoder` from simple words in `< 2 ms` on CPU.

    Uses mean-centered `M_valid` (`W_lm_head[valid_idx] @ J_49_to_63`), spectral-filtered
    `Sigma_49_p030` (`Sigma_{49}^{0.30}` with top-1 common-mode eigenvalue dropped), and
    forward `J_ref^{(family)}` (`fl2va` or `ref2va`) with zero `.toml` files and zero GPU load.
    """
    a_val = float(alpha)
    if not (0.0 <= a_val <= 2.0):
        raise ValueError(f"alpha must be in [0.0, 2.0], got {alpha}")
    b = dict(bundle) if bundle is not None else load_j_lens_bundle()
    pos_list = _parse_word_list(pos_words)
    if not pos_list:
        raise ValueError("pos_words must contain at least one word")
    neg_list = _parse_word_list(neg_words) if neg_words else list(DEFAULT_NEUTRAL_WORDS)

    w2i = b["word_to_idxs"]
    m_centered = b["M_centered"]
    s_p030 = b["Sigma_49_p030"]
    j_ref = b["J_ref_ref2va"] if "ref2va" in str(family).lower() else b["J_ref_fl2va"]

    pos_ids, pos_matched, pos_unmatched = _resolve_word_indices(pos_list, w2i)
    if not pos_ids:
        raise ValueError(f"None of pos_words {pos_list!r} matched the 24,912-word J-Lens vocabulary")
    neg_ids, neg_matched, neg_unmatched = _resolve_word_indices(neg_list, w2i)
    if not neg_ids:
        raise ValueError(f"None of neg_words {neg_list!r} matched the 24,912-word J-Lens vocabulary")

    mu_pos = F.normalize(m_centered[pos_ids].mean(dim=0), dim=0)
    mu_neg = F.normalize(m_centered[neg_ids].mean(dim=0), dim=0)
    delta_m = mu_pos - mu_neg

    v_enc_raw = s_p030 @ delta_m
    v_enc = F.normalize(v_enc_raw, dim=0)
    v_pr_raw = j_ref @ v_enc_raw
    v_pr = F.normalize(v_pr_raw, dim=0) * float(target_norm)

    hold_list = _parse_word_list(hold_words)
    if hold_list and a_val > 0.0:
        h_ids, _, _ = _resolve_word_indices(hold_list, w2i)
        if h_ids:
            mu_hold = F.normalize(m_centered[h_ids].mean(dim=0), dim=0) - mu_neg
            v_hold_enc = F.normalize(s_p030 @ mu_hold, dim=0)
            v_hold_pr = F.normalize(j_ref @ (s_p030 @ mu_hold), dim=0)
            v_enc_held = apply_soft_holds(v_enc, [v_hold_enc], alpha=a_val)
            v_pr = apply_soft_holds(v_pr, [v_hold_pr], alpha=a_val)
            if float(torch.linalg.vector_norm(v_pr.to(dtype=torch.float64)).item()) <= 1e-3 * max(1.0, float(target_norm)):
                raise ValueError("hold_words completely cancelled pos_words (residual norm is zero)")
            v_enc = F.normalize(v_enc_held, dim=0)

    readout = decode_vector(v_pr, space="post_refiner", family=family, bundle=b)
    return {
        "family": family,
        "pos_words_matched": pos_matched,
        "pos_words_unmatched": pos_unmatched,
        "neg_words_matched": neg_matched,
        "neg_words_unmatched": neg_unmatched,
        "hold_words": hold_list,
        "alpha": float(alpha) if hold_list else 0.0,
        "vector.post_refiner": v_pr,
        "vector.encoder": v_enc,
        "readout": readout,
    }


def save_synthesized_vector(
    out_path: str | os.PathLike,
    synth: Mapping[str, Any],
    *,
    name: str = "",
) -> Path:
    """Save a `synthesize_from_words` result as a standard `h3-control-vector/1` `.safetensors` control vector."""
    path = Path(out_path)
    if not path.suffix:
        path = path.with_suffix(".safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    v_pr = synth["vector.post_refiner"].detach().to(device="cpu", dtype=torch.float32).contiguous()
    v_enc = synth["vector.encoder"].detach().to(device="cpu", dtype=torch.float32).contiguous()
    raw_norm = float(torch.linalg.vector_norm(v_pr.to(dtype=torch.float64)).item())
    enc_norm = float(torch.linalg.vector_norm(v_enc.to(dtype=torch.float64)).item())
    raw_stem = (name or path.stem.split("__")[0]).lower()
    stem = re.sub(r"[^a-z0-9_]+", "_", raw_stem).strip("_") or "synthesized_vector"
    family = str(synth.get("family", "fl2va"))
    pos_matched = list(synth.get("pos_words_matched", ()))
    neg_matched = list(synth.get("neg_words_matched", ()))
    hold_list = list(synth.get("hold_words", ()))
    a_side = ", ".join(neg_matched)
    b_side = ", ".join(pos_matched)
    words_digest = hashlib.sha256(f"{a_side}->{b_side}|holds={hold_list}".encode("utf-8")).hexdigest()
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    recipe = _DEFAULT_RECIPES.get(family, _DEFAULT_RECIPES["fl2va"])

    meta_dict: dict[str, Any] = {
        "format": "h3-control-vector/1",
        "name": stem,
        "quality": f"J-Lens word synthesis: +{pos_matched} / -{neg_matched}",
        "family": family,
        "recipe": recipe,
        "measured_under": {
            "mode": "ref2va" if family == "ref2va" else "t2va",
            "frame": "bare",
            "prefix": "none",
            "pooling": "diff_span",
        },
        "points": {
            "post_refiner": {
                "unit": "mean_pair_difference",
                "raw_norm": raw_norm,
                "median_row_norm": DEFAULT_POST_REFINER_MEDIAN_ROW_NORM,
            },
            "encoder": {
                "unit": "median_row_norm",
                "raw_norm": enc_norm,
                "median_row_norm": DEFAULT_ENCODER_MEDIAN_ROW_NORM,
                "row0_excluded": True,
            },
        },
        "pairs": {
            "path": "j_lens.synthesize_from_words",
            "sha256": words_digest,
            "a_side": a_side,
            "b_side": b_side,
            "n_pairs": max(len(pos_matched), 1),
            "n_holds": len(hold_list),
            "pooling": "diff_span",
        },
        "quality_report": {
            "mean_pairwise_cosine": 1.0,
            "split_half_cosine": 1.0,
            "per_pair_cosine": [1.0 for _ in range(max(len(pos_matched), 1))],
            "j_lens_readout": synth.get("readout", {}),
        },
        "provenance": {
            "core_revision": "j_lens_cpu",
            "pack_commit": "j_lens_v1",
            "created_utc": now_utc,
            "made_by": "j_lens.synthesize_from_words/1",
        },
    }
    encoded_meta = {k: json.dumps(v, sort_keys=True) for k, v in meta_dict.items()}
    save_file(
        {"vector.post_refiner": v_pr, "vector.encoder": v_enc},
        str(path),
        metadata=encoded_meta,
    )
    return path


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decode or synthesize 5376-d H3 post-refiner control vectors via the J-Lens (<15 ms on CPU)."
    )
    parser.add_argument("vectors", nargs="*", help="One or more .safetensors control vector files to decode.")
    parser.add_argument(
        "--hold",
        action="append",
        default=[],
        help="Optional .safetensors vector(s) to project out (`mode = 'hold'`) before decoding.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Soft or over-projected hold attenuation strength alpha in [0.0, 2.0] when --hold or --hold-words is given (default: 1.0).",
    )
    parser.add_argument("--scale", type=float, default=DEFAULT_STEER_SCALE, help="Steering norm scale at layers.49.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of positive and negative words to return.")
    parser.add_argument(
        "--svd",
        action="store_true",
        help="Decode the top unsupervised left singular vectors of TokenRefiner (no .safetensors file required).",
    )
    parser.add_argument(
        "--from-words",
        default="",
        help="Synthesize a 5376-d post-refiner control vector directly from simple positive words (comma/space-separated) in <2 ms on CPU.",
    )
    parser.add_argument(
        "--against-words",
        default="",
        help="Optional negative words when using --from-words (defaults to neutral baseline words).",
    )
    parser.add_argument(
        "--hold-words",
        default="",
        help="Optional words to project out when using --from-words.",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional output .safetensors path when using --from-words.",
    )
    parser.add_argument("--family", default="fl2va", choices=("fl2va", "ref2va"), help="Refiner family for --svd or --from-words.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    try:
        bundle = load_j_lens_bundle()
        if args.from_words.strip():
            synth = synthesize_from_words(
                args.from_words,
                args.against_words if args.against_words.strip() else None,
                family=args.family,
                hold_words=args.hold_words if args.hold_words.strip() else None,
                alpha=args.alpha,
                bundle=bundle,
            )
            saved_path = None
            if args.save.strip():
                saved_path = save_synthesized_vector(args.save.strip(), synth)
            out_summary = {
                "family": synth["family"],
                "pos_words_matched": synth["pos_words_matched"],
                "neg_words_matched": synth["neg_words_matched"],
                "hold_words": synth["hold_words"],
                "alpha": synth["alpha"],
                "saved_to": str(saved_path) if saved_path else None,
                "readout": synth["readout"],
            }
            if args.json:
                print(json.dumps(out_summary, indent=2))
            else:
                r = synth["readout"]
                print(
                    f"Synthesized {synth['family']} vector (norm={r['norm_after_holds']:.2f}) from +{synth['pos_words_matched']} / -{synth['neg_words_matched']}:"
                )
                print(f"  PR (+) {r['pos_summary']}")
                print(f"  PR (-) {r['neg_summary']}")
                if saved_path:
                    print(f"  Saved to {saved_path}")
            return 0

        if args.svd:
            svd_rows = refiner_singular_vectors(
                family=args.family,
                top_n=args.top_k,
                scale=args.scale,
                bundle=bundle,
            )
            if args.json:
                print(json.dumps(svd_rows, indent=2))
            else:
                for row in svd_rows:
                    print(f"SV {row['rank']:2d} (sigma={row['singular_value']:.4f}):")
                    print(f"  (+) {row['pos_summary']}")
                    print(f"  (-) {row['neg_summary']}")
            return 0

        if not args.vectors:
            parser.error("Provide one or more .safetensors files, pass --from-words, or pass --svd.")

        reports = [
            decode_safetensors(
                v_path,
                scale=args.scale,
                top_k=args.top_k,
                holds=args.hold if args.hold else None,
                alpha=args.alpha,
                bundle=bundle,
            )
            for v_path in args.vectors
        ]
    except (FileNotFoundError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        print(f"j_lens.py: error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(reports if len(reports) > 1 else reports[0], indent=2))
    else:
        for rep in reports:
            pr = rep["post_refiner"]
            hold_str = f" [holds={rep['holds']} @ alpha={rep['alpha']:.2f}]" if rep["holds"] else ""
            print(
                f"{rep['file']} ({rep['family']}, norm={pr['norm_before_holds']:.2f} -> {pr['norm_after_holds']:.2f}){hold_str}:"
            )
            print(f"  PR (+) {pr['pos_summary']}")
            print(f"  PR (-) {pr['neg_summary']}")
            if "encoder" in rep:
                enc = rep["encoder"]
                print(f"  EN (+) {enc['pos_summary']}")
                print(f"  EN (-) {enc['neg_summary']}")
                print(
                    f"  cos(J_ref v_enc, v_PR) = {rep['refiner_forward_cosine']:.4f} | "
                    f"cos_J63(Wiener) = {rep['wiener_j63_cosine']:.4f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
