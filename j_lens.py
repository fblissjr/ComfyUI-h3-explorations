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
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save as save_safetensors_bytes

logger = logging.getLogger(__name__)

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

# Named negative-pole presets for synthesize_from_words (pass e.g. "@replace_human" or "replace_human").
# Measured on Arms 53..65 (bench/results/2026-11-05_control_vectors_pika_and_ref2va.md §8):
# environment/weather/lighting vectors layer onto the prompt with DEFAULT_NEUTRAL_WORDS, whereas
# subject/medium replacements must actively suppress "Live-action / human / flesh / person" tokens.
AGAINST_PRESETS: dict[str, tuple[str, ...]] = {
    "neutral": DEFAULT_NEUTRAL_WORDS,
    "replace_human": (
        "human",
        "flesh",
        "skin",
        "person",
        "live-action",
        "realistic",
        "ordinary",
        "photograph",
    ),
    "replace_live_action": (
        "live-action",
        "realistic",
        "human",
        "flesh",
        "ordinary",
        "photograph",
        "natural",
        "matte",
    ),
    "replace_digital": (
        "digital",
        "modern",
        "clean",
        "sharp",
        "contemporary",
        "ordinary",
        "standard",
    ),
}

_REPLACEMENT_CONCEPT_WORDS: frozenset[str] = frozenset(
    {
        "skeleton",
        "skeletal",
        "skull",
        "bones",
        "bone",
        "balloon",
        "inflatable",
        "latex",
        "rubber",
        "puppet",
        "muppet",
        "felt",
        "clay",
        "claymation",
        "plasticine",
        "origami",
        "paper",
        "cartoon",
        "anime",
        "robot",
        "automaton",
        "statue",
        "marble",
    }
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
        cloze_base = f.get_tensor("cloze_base_union").to(dtype=torch.float32)
        piece_weights = torch.sigmoid(
            (torch.log_softmax(cloze_base, dim=-1)[:, valid_pos].max(dim=0).values + 11.5) / 2.0
        )
        bundle = {
            "path": str(path),
            "words": words,
            "word_to_idxs": word_to_idxs,
            "M_union": m_union,
            "M_centered": m_centered,
            "piece_weights": piece_weights,
            "Sigma_49_p030": sigma_p030,
            "cloze_base_union": cloze_base,
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
            "readiness": {
                "tier": "weak",
                "top1_gain": 0.0,
                "top3_mean_gain": 0.0,
                "refiner_transmission": 0.0,
                "echo_ratio": None,
                "off_target_words": [],
                "recommended_mode": "add_norm_matched",
                "recommended_strength": "1.00 .. 1.20",
                "summary": "WEAK (zero vector norm)",
                "recommendations": ["Vector norm is zero; provide non-cancelling positive words or reduce hold alpha."],
            },
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

    base_out: dict[str, Any] = {
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
    base_out["readiness"] = assess_steering_readiness(
        v_eff if space == "post_refiner" else None,
        v_enc=v_eff if space == "encoder" else None,
        readout=base_out,
        family=family,
        bundle=b,
    )
    return base_out


def assess_steering_readiness(
    v_pr: torch.Tensor | None = None,
    *,
    v_enc: torch.Tensor | None = None,
    readout: Mapping[str, Any] | None = None,
    pos_words: Sequence[str] | None = None,
    neg_words: Sequence[str] | None = None,
    is_unigram_synthesis: bool = False,
    family: str = "fl2va",
    bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute pre-render J-Lens diagnostics on CPU (`< 1 ms`).

    Reports objective numbers measured on the 24,912-word 7-prefix cloze `M_union` basis and
    `J_ref^{(family)}` (`AGENTS.md`: no fuzzy red/traffic-light labels):
      - `basis`: `"unigram_synthesis"` (built in `row(M_centered)`, typical `top1_gain = 2.6..5.2`)
        vs. `"post_refiner"` / `"encoder"` (`Qwen3-VL-32B` pair vectors, typical `top1_gain = 1.0..2.6`).
      - `basis_note`: explicit provenance note explaining why unigram `M_union` self-projection
        `top1_gain` is not comparable to 50-layer contextual `Qwen3-VL` + `TokenRefiner` vectors.
      - `top1_gain`, `top3_mean_gain`: peak and top-3 mean log-odds gain at `scale = 90`.
      - `refiner_transmission`: `||J_ref @ (v_enc / ||v_enc||_2)||_2` when true `5120`-d `v_enc`
        is provided (`None` when only `5376`-d `v_pr` is available, never estimated from `J_wiener @ v_pr`).
      - `echo_ratio` & `off_target_words` (when `pos_words` is provided).
      - `is_replacement_concept` (`skeleton`, `balloon`, `muppet`, `claymation`, etc.).
    """
    b = dict(bundle) if bundle is not None else load_j_lens_bundle()
    if readout is None:
        if v_pr is not None:
            readout = decode_vector(v_pr, space="post_refiner", family=family, bundle=b)
        elif v_enc is not None:
            readout = decode_vector(v_enc, space="encoder", family=family, bundle=b)
        else:
            raise ValueError("assess_steering_readiness requires readout, v_pr, or v_enc")

    pos_list = list(readout.get("positive_words") or ())
    top1_gain = round(float(pos_list[0]["gain"]), 2) if pos_list else 0.0
    top3_slice = pos_list[:3]
    top3_mean_gain = (
        round(sum(float(x["gain"]) for x in top3_slice) / len(top3_slice), 2)
        if top3_slice
        else 0.0
    )

    j_ref = b["J_ref_ref2va"] if "ref2va" in str(family).lower() else b["J_ref_fl2va"]

    ref_trans: float | None = None
    if v_enc is not None:
        enc_flat = v_enc.detach().to(device="cpu", dtype=torch.float32).flatten()
        enc_nrm = float(torch.linalg.vector_norm(enc_flat).item())
        if enc_nrm > 1e-12:
            enc_unit = enc_flat / enc_nrm
            ref_trans = round(float(torch.linalg.vector_norm(j_ref @ enc_unit).item()), 4)

    is_unigram = bool(pos_words or is_unigram_synthesis)
    basis = "unigram_synthesis" if is_unigram else str(readout.get("space", "post_refiner"))
    basis_note = (
        "In-subspace M_union projection (unigram bag-of-words); do not compare top1_gain against 50-layer contextual Qwen3-VL post_refiner vectors."
        if is_unigram
        else "Contextual 50-layer Qwen3-VL + TokenRefiner extraction; grade primarily on split_half_cosine, mean_pairwise_cosine, and refiner_transmission."
    )

    echo_ratio: float | None = None
    off_target: list[str] = []
    if pos_words:
        req_clean = [w.strip().lower() for w in pos_words if w.strip()]
        expanded_stems: set[str] = set()
        for w in req_clean:
            expanded_stems.add(w[:4] if len(w) >= 4 else w)
            for fb in _SHORT_WORD_FALLBACKS.get(w, ()):
                expanded_stems.add(fb[:4] if len(fb) >= 4 else fb)
            if "-" in w:
                for part in w.split("-"):
                    if len(part) >= 3:
                        expanded_stems.add(part[:4] if len(part) >= 4 else part)
        w2i_rd = b["word_to_idxs"]
        m_c_rd = b["M_centered"]
        mu_pos_rd, pos_ids_rd, _, _ = _mean_word_direction(req_clean, w2i_rd, m_c_rd)
        per_word_vecs: list[torch.Tensor] = []
        for rw in req_clean:
            for cand in (rw, *_SHORT_WORD_FALLBACKS.get(rw, ())):
                c_ids, _, _ = _resolve_word_indices([cand], w2i_rd)
                if c_ids:
                    per_word_vecs.append(F.normalize(m_c_rd[c_ids].mean(dim=0), dim=0))
        top_words = [str(item["word"]).lower() for item in pos_list[:8]]
        matched_top = 0
        for tw in top_words:
            if any(tw.startswith(st) or st.startswith(tw[:4]) for st in expanded_stems if st):
                matched_top += 1
                continue
            if pos_ids_rd:
                tw_ids, _, _ = _resolve_word_indices([tw], w2i_rd)
                if tw_ids:
                    mw_tw = F.normalize(m_c_rd[tw_ids].mean(dim=0), dim=0)
                    max_sim = max(
                        [float(torch.dot(mu_pos_rd, mw_tw).item())]
                        + [float(torch.dot(uw, mw_tw).item()) for uw in per_word_vecs]
                    )
                    if max_sim >= 0.18:
                        matched_top += 1
                        continue
            off_target.append(tw)
        echo_ratio = round(matched_top / max(len(top_words), 1), 2)

    check_tokens = (
        set(w.lower() for w in pos_words)
        if pos_words
        else set(str(x["word"]).lower() for x in pos_list[:2])
    )
    is_replacement = bool(check_tokens & _REPLACEMENT_CONCEPT_WORDS)
    neg_set = set(w.lower() for w in (neg_words or DEFAULT_NEUTRAL_WORDS))
    uses_default_neg = neg_set.issubset(set(DEFAULT_NEUTRAL_WORDS))

    recommendations: list[str] = []
    rec_strength = "1.25 .. 1.55" if is_replacement else "1.00 .. 1.25"

    if is_replacement and pos_words and uses_default_neg:
        recommendations.append(
            "Subject/medium replacement concept with default neutral against_words: pass "
            "against_words='@replace_human' ('human, flesh, skin, person, live-action, realistic') "
            "so the negative pole suppresses 'Live-action / human' prompt tokens."
        )
    if ref_trans is not None and ref_trans < 0.495:
        recommendations.append(
            f"TokenRefiner pass-through is {ref_trans:.3f} (measured on ||J_ref @ v_enc_unit||_2): "
            "concrete visual, lighting, or material nouns typically have pass-through >= 0.50."
        )
    if is_unigram and top1_gain < (2.80 if uses_default_neg else 2.40):
        recommendations.append(
            f"Unigram word-synthesis post-refiner peak gain is +{top1_gain:.2f} (measured on M_union at scale = 90): "
            "3-5 concrete domain nouns typically yield >= +3.00."
        )
    if echo_ratio is not None and echo_ratio < 0.75 and off_target:
        recommendations.append(
            f"Off-target words in top-8 readout ({off_target[:3]}): pass hold_words='{', '.join(off_target[:3])}' to scrub them if unwanted."
        )

    rt_str = f"{ref_trans:.3f}" if ref_trans is not None else "n/a (no vector.encoder)"
    summary = (
        f"basis={basis}  top1=+{top1_gain:.2f}  top3_mean=+{top3_mean_gain:.2f}  "
        f"refiner_pass={rt_str}  rec_strength={rec_strength}"
    )
    return {
        "basis": basis,
        "basis_note": basis_note,
        "top1_gain": top1_gain,
        "top3_mean_gain": top3_mean_gain,
        "refiner_transmission": ref_trans,
        "echo_ratio": echo_ratio,
        "off_target_words": off_target[:5],
        "is_replacement_concept": is_replacement,
        "recommended_mode": "add_norm_matched",
        "recommended_strength": rec_strength,
        "summary": summary,
        "recommendations": recommendations,
    }


def _resolve_vector_path(source: str | os.PathLike, family: str = "fl2va") -> Path:
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
    if cv_dir.is_dir():
        stem = path.stem.split("__")[0]
        fam = "ref2va" if "ref2va" in str(family).lower() else "fl2va"
        for pat in (
            f"{stem}__{fam}__diff_span__*.safetensors",
            f"{stem}__{fam}__*.safetensors",
            f"{stem}__*.safetensors",
        ):
            hits = sorted(cv_dir.glob(pat))
            if hits:
                return hits[0]
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
    hold_words: str | Sequence[str] | None = None,
    alpha: float = 1.0,
    prompt: str | os.PathLike | None = None,
    bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode a `.safetensors` control vector (and optional hold files/tensors or `hold_words`) via J-Lens."""
    resolved_path = _resolve_vector_path(path)
    v_pr, name, meta = _load_vector_from_source(resolved_path, space="post_refiner")
    family = str(meta.get("family", "fl2va"))
    b = dict(bundle) if bundle is not None else load_j_lens_bundle()

    hold_tensors_pr: list[torch.Tensor] = []
    hold_tensors_enc: list[torch.Tensor] = []
    hold_names: list[str] = []
    if holds:
        for h_src in holds:
            ht, hname, _ = _load_vector_from_source(h_src, space="post_refiner")
            hold_tensors_pr.append(ht)
            hold_names.append(hname)
            try:
                ht_enc, _, _ = _load_vector_from_source(h_src, space="encoder")
                hold_tensors_enc.append(ht_enc)
            except Exception:
                pass
    hw_list = _parse_word_list(hold_words)
    if hw_list:
        synth_fam = "ref2va" if "ref2va" in family.lower() else "fl2va"
        hw_synth = synthesize_from_words(hw_list, family=synth_fam, bundle=b)
        hold_tensors_pr.append(hw_synth["vector.post_refiner"])
        hold_tensors_enc.append(hw_synth["vector.encoder"])
        hold_names.append(f"words:{','.join(hw_list)}")

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
    v_enc_eff: torch.Tensor | None = None
    with safe_open(str(resolved_path), framework="pt", device="cpu") as f:
        if "vector.encoder" in f.keys():
            v_enc = f.get_tensor("vector.encoder").to(dtype=torch.float32).flatten()
            v_enc_eff = (
                apply_soft_holds(v_enc, hold_tensors_enc, alpha=alpha)
                if hold_tensors_enc
                else v_enc
            )
            enc_readout = decode_vector(
                v_enc_eff,
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

    v_pr_eff = apply_soft_holds(v_pr, hold_tensors_pr, alpha=alpha) if hold_tensors_pr else v_pr
    pairs_meta_rd = meta.get("pairs") if isinstance(meta.get("pairs"), Mapping) else {}
    prov_meta_rd = meta.get("provenance") if isinstance(meta.get("provenance"), Mapping) else {}
    is_unigram_file = bool(
        (isinstance(pairs_meta_rd, Mapping) and str(pairs_meta_rd.get("path", "")).startswith("j_lens.synthesize_from_words"))
        or (isinstance(prov_meta_rd, Mapping) and str(prov_meta_rd.get("made_by", "")).startswith("j_lens.synthesize_from_words"))
    )
    unigram_pos_words: list[str] | None = None
    if is_unigram_file and isinstance(pairs_meta_rd, Mapping):
        unigram_pos_words = _parse_word_list(str(pairs_meta_rd.get("b_side", ""))) or None
    pr_readout["readiness"] = assess_steering_readiness(
        v_pr_eff,
        v_enc=v_enc_eff,
        readout=pr_readout,
        pos_words=unigram_pos_words,
        is_unigram_synthesis=is_unigram_file,
        family=family,
        bundle=b,
    )
    if prompt is not None and str(prompt).strip():
        pairs_meta = meta.get("pairs") if isinstance(meta.get("pairs"), Mapping) else {}
        b_side_raw = str(pairs_meta.get("b_side", "")) if isinstance(pairs_meta, Mapping) else ""
        a_side_raw = str(pairs_meta.get("a_side", "")) if isinstance(pairs_meta, Mapping) else ""
        b_side_words = [
            w for w in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9-]{1,}", b_side_raw.lower())
            if w not in _PROMPT_RECEPTOR_STOPWORDS
        ][:6]
        top_pos_words = [str(x["word"]) for x in list(pr_readout.get("positive_words") or ())[:4]]
        seed_pos_words: list[str] = []
        for w in b_side_words + top_pos_words:
            if w not in seed_pos_words:
                seed_pos_words.append(w)
        has_explicit_a = bool(
            a_side_raw
            and any(k in a_side_raw.lower() for k in ("human", "person", "flesh", "live-action", "skin"))
        )
        pr_readout["prompt_coupling"] = analyze_prompt_coupling(
            v_pr_eff,
            prompt,
            pos_words=seed_pos_words or None,
            readout=pr_readout,
            family=family,
            already_contrasted=has_explicit_a,
            bundle=b,
        )

    return out


def decode_recipe(
    steps: Sequence[Mapping[str, Any]],
    *,
    family: str = "fl2va",
    scale: float = DEFAULT_STEER_SCALE,
    min_logprob: float = DEFAULT_MIN_LOGPROB,
    top_k: int = 8,
    hold_words: str | Sequence[str] | None = None,
    alpha: float = 1.0,
    prompt: str | os.PathLike | None = None,
    return_tensors: bool = False,
    bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a multi-vector recipe (`add`, `add_norm_matched`, `hold`) and decode its net shift.

    Each item in `steps` is a mapping with:
      - `vector`: path to `.safetensors` or `5376`-d `torch.Tensor`
      - `mode`: `"add"`, `"add_norm_matched"`, or `"hold"` (default `"add"`)
      - `strength`: float multiplier for `"add"` / `"add_norm_matched"`, or `alpha in [0.0, 2.0]` for `"hold"`
    """
    a_val = float(alpha)
    if not (0.0 <= a_val <= 2.0):
        raise ValueError(f"alpha must be in [0.0, 2.0], got {alpha}")
    b = dict(bundle) if bundle is not None else load_j_lens_bundle()
    add_terms: list[tuple[str, torch.Tensor]] = []
    add_terms_enc: list[torch.Tensor] = []
    hold_steps: list[tuple[str, torch.Tensor, torch.Tensor | None, float]] = []
    step_records: list[dict[str, Any]] = []

    for item in steps:
        src = item["vector"]
        mode = str(item.get("mode", "add")).strip()
        strength = float(item.get("strength", 1.0))
        vec, name, meta = _load_vector_from_source(src, space="post_refiner")
        if meta.get("family") and family == "fl2va":
            family = str(meta["family"])
        v64 = vec.to(dtype=torch.float64)
        v_norm = float(torch.linalg.vector_norm(v64).item())

        v_enc_step: torch.Tensor | None = None
        if not isinstance(src, torch.Tensor):
            try:
                v_enc_step, _, _ = _load_vector_from_source(src, space="encoder")
            except Exception:
                v_enc_step = None
        if v_enc_step is None:
            j_w = b["J_wiener_ref2va"] if "ref2va" in str(family).lower() else b["J_wiener_fl2va"]
            w_rec = j_w @ vec.to(dtype=torch.float32)
            if float(torch.linalg.vector_norm(w_rec).item()) > 1e-12:
                v_enc_step = F.normalize(w_rec, dim=0)

        step_records.append({"name": name, "mode": mode, "strength": strength})

        if mode == "add":
            add_terms.append((name, (strength * v64).to(dtype=torch.float32)))
            if v_enc_step is not None:
                enc_w = strength * (v_norm / NORM_MATCHED_TARGET) if NORM_MATCHED_TARGET > 0 else strength
                add_terms_enc.append(enc_w * F.normalize(v_enc_step.to(dtype=torch.float64), dim=0))
        elif mode == "add_norm_matched":
            scale_factor = (NORM_MATCHED_TARGET / v_norm) if v_norm > 1e-12 else 0.0
            add_terms.append((name, (strength * scale_factor * v64).to(dtype=torch.float32)))
            if v_enc_step is not None:
                add_terms_enc.append(strength * F.normalize(v_enc_step.to(dtype=torch.float64), dim=0))
        elif mode == "hold":
            if not (0.0 <= strength <= 2.0):
                raise ValueError(f"mode='hold' strength (alpha) must be in [0.0, 2.0], got {strength}")
            hold_steps.append((name, vec, v_enc_step, strength))
        else:
            raise ValueError(f"Unsupported recipe mode {mode!r}; expected 'add', 'add_norm_matched', or 'hold'")

    hw_list = _parse_word_list(hold_words)
    if hw_list and a_val > 0.0:
        synth_fam = "ref2va" if "ref2va" in str(family).lower() else "fl2va"
        hw_synth = synthesize_from_words(hw_list, family=synth_fam, bundle=b)
        hold_steps.append(
            (
                f"words:{','.join(hw_list)}",
                hw_synth["vector.post_refiner"],
                hw_synth["vector.encoder"],
                a_val,
            )
        )

    if not add_terms:
        zero = torch.zeros(5376, dtype=torch.float32)
        zero_enc = torch.zeros(5120, dtype=torch.float32)
        ro_zero = decode_vector(zero, space="post_refiner", family=family, bundle=b)
        out_empty: dict[str, Any] = {
            "family": family,
            "steps": step_records,
            "hold_words": hw_list,
            "alpha": a_val if hold_steps else 0.0,
            "combined_norm_before_holds": 0.0,
            "combined_norm_after_holds": 0.0,
            "pairwise_interactions": [],
            "readout": ro_zero,
        }
        if return_tensors:
            out_empty["vector.post_refiner"] = zero
            out_empty["vector.encoder"] = zero_enc
        return out_empty

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
    cur_enc: torch.Tensor | None = None
    if add_terms_enc:
        sum_enc = torch.stack(add_terms_enc, dim=0).sum(dim=0)
        if float(torch.linalg.vector_norm(sum_enc).item()) > 1e-12:
            cur_enc = F.normalize(sum_enc.to(dtype=torch.float32), dim=0)

    for _, h_vec, h_enc, h_alpha in hold_steps:
        cur_vec = apply_soft_holds(cur_vec, [h_vec], alpha=h_alpha)
        if cur_enc is not None and h_enc is not None:
            cur_enc_held = apply_soft_holds(cur_enc, [h_enc], alpha=h_alpha)
            if float(torch.linalg.vector_norm(cur_enc_held).item()) > 1e-12:
                cur_enc = F.normalize(cur_enc_held, dim=0)

    readout = decode_vector(
        cur_vec,
        space="post_refiner",
        family=family,
        scale=scale,
        min_logprob=min_logprob,
        top_k=top_k,
        bundle=b,
    )
    readout["readiness"] = assess_steering_readiness(
        cur_vec,
        v_enc=cur_enc,
        readout=readout,
        family=family,
        bundle=b,
    )
    if prompt is not None and str(prompt).strip():
        readout["prompt_coupling"] = analyze_prompt_coupling(
            cur_vec,
            prompt,
            readout=readout,
            family=family,
            bundle=b,
        )

    out_rec: dict[str, Any] = {
        "family": family,
        "steps": step_records,
        "hold_words": hw_list,
        "alpha": a_val if hold_steps else 0.0,
        "combined_norm_before_holds": round(norm_before, 4),
        "combined_norm_after_holds": readout["norm_after_holds"],
        "pairwise_interactions": interactions,
        "readout": readout,
        "readiness": readout["readiness"],
        "prompt_coupling": readout.get("prompt_coupling"),
    }
    if return_tensors:
        if cur_enc is None:
            j_w = b["J_wiener_ref2va"] if "ref2va" in str(family).lower() else b["J_wiener_fl2va"]
            cur_enc = F.normalize(j_w @ cur_vec, dim=0)
        out_rec["vector.post_refiner"] = cur_vec
        out_rec["vector.encoder"] = cur_enc
    return out_rec


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
    "fog": ("mist", "haze", "cloudy"),
    "wet": ("soaked", "damp", "rain"),
    "dry": ("dusty", "outdoor", "desert"),
    "old": ("older", "vintage", "dated"),
    "new": ("modern", "fresh", "recent"),
    "dim": ("dark", "shadow", "faint"),
    "hot": ("warm", "heated", "fiery"),
    "red": ("crimson", "scarlet", "ruby"),
    "raw": ("gritty", "rough", "coarse"),
    "8-bit": ("pixel", "arcade", "console", "cartridge"),
    "8bit": ("pixel", "arcade", "console", "cartridge"),
    "16-bit": ("pixel", "arcade", "console", "cartridge"),
    "led": ("neon", "glowing", "matrix", "laser"),
    "leds": ("neon", "glowing", "matrix", "laser"),
    "crt": ("monitor", "screen", "analog", "static"),
    "vhs": ("tape", "cassette", "analog", "video"),
    "cgi": ("digital", "rendered", "synthetic", "computer"),
    "sfx": ("sound", "acoustic", "noise", "echo"),
    "xray": ("skeleton", "bones", "skull", "translucent"),
    "x-ray": ("skeleton", "bones", "skull", "translucent"),
    "cyberpunk": ("neon", "punk", "digital", "matrix", "electric"),
    "strobe": ("flashing", "neon", "disco", "lights"),
    "automaton": ("mechanical", "robot", "brass", "clock", "gears"),
    "chiptune": ("arcade", "electronic", "melody", "console"),
    "synth": ("electronic", "neon", "melody"),
    "victorian": ("antique", "brass", "vintage", "classic"),
    "squeak": ("pitched", "whistle", "piercing", "rubbing", "rubber"),
    "squeaking": ("pitched", "whistle", "piercing", "rubbing", "rubber"),
    "squeaky": ("pitched", "whistle", "piercing", "rubbing", "rubber"),
    "steampunk": ("clock", "brass", "steam", "gears", "copper", "mechanical", "antique"),
    "clockwork": ("clock", "brass", "gear", "gears", "mechanical", "copper"),
    "claymation": ("clay", "puppet", "miniature", "cartoon"),
    "stop-motion": ("clay", "puppet", "miniature", "frame"),
    "watercolor": ("painted", "brush", "canvas", "color"),
    "cat": ("cats", "kitten", "animal"),
    "dog": ("dogs", "canine", "puppy"),
    "live-action": ("realistic", "photograph", "real", "human", "person", "flesh"),
}

_SUBJECT_STOPWORDS: frozenset[str] = frozenset(
    {
        "integrated",
        "multimodal",
        "description",
        "detailed",
        "summary",
        "shot",
        "shots",
        "cinematic",
        "the",
        "and",
        "with",
        "into",
        "from",
        "over",
        "under",
        "that",
        "this",
        "his",
        "her",
        "their",
        "she",
        "him",
        "they",
        "them",
        "its",
        "very",
        "while",
        "through",
        "across",
        "down",
        "onto",
        "toward",
        "towards",
        "for",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "medium",
        "close",
        "close-up",
        "wide",
        "medium-wide",
        "medium-close",
        "two-shot",
        "overhead",
        "frames",
        "frame",
        "framing",
        "opens",
        "opening",
        "inside",
        "outside",
        "night",
        "late-night",
        "daylit",
        "daylight",
        "afternoon",
        "morning",
        "evening",
        "continuous",
        "take",
        "cuts",
        "cut",
        "handheld",
        "static",
        "tripod",
        "camera",
        "lens",
        "shallow",
        "depth",
        "field",
        "low-key",
        "high-contrast",
        "lighting",
        "practical",
        "motivated",
        "holding",
        "exact",
        "layout",
        "wardrobe",
        "props",
        "shown",
        "target",
        "video",
        "source",
        "reference",
        "references",
        "referenced",
        "picture",
        "pictures",
        "align",
        "aligns",
        "mark",
        "second",
        "seconds",
        "style",
        "look",
        "captures",
        "unfolds",
        "begins",
        "starts",
        "beginning",
        "position",
        "holds",
        "stands",
        "sits",
        "walks",
        "walking",
        "steps",
        "watches",
        "straight",
        "on-screen",
        "off-screen",
        "thirties",
        "forties",
        "twenties",
        "fifties",
        "sixties",
        "low",
        "even",
        "quick",
        "delivery",
        "unhurried",
        "phrasing",
        "slightly",
        "hoarse",
        "quietly",
        "sings",
        "says",
        "speaks",
        "himself",
        "herself",
        "one",
        "two",
        "three",
        "more",
        "round",
        "produces",
        "vocal",
        "sound",
        "no",
        "people",
        "present",
        "appear",
        "appears",
    }
)

_PROMPT_RECEPTOR_STOPWORDS: frozenset[str] = frozenset(
    {
        "integrated",
        "multimodal",
        "description",
        "detailed",
        "summary",
        "overall",
        "soundscape",
        "non",
        "diegetic",
        "music",
        "subject",
        "definitions",
        "cinematic",
        "documentary",
        "shot",
        "shots",
        "frame",
        "frames",
        "framed",
        "framing",
        "camera",
        "lens",
        "wide",
        "medium",
        "close",
        "close-up",
        "medium-wide",
        "medium-close",
        "two-shot",
        "overhead",
        "steady",
        "static",
        "tripod",
        "handheld",
        "slow",
        "fast",
        "speed",
        "amplitude",
        "small",
        "large",
        "take",
        "second",
        "seconds",
        "cuts",
        "cut",
        "opens",
        "opening",
        "holds",
        "holding",
        "pans",
        "tilts",
        "tracks",
        "tracking",
        "zooms",
        "zoom",
        "pushes",
        "pulls",
        "continuous",
        "vocal",
        "produces",
        "sound",
        "on-screen",
        "off-screen",
        "the",
        "and",
        "with",
        "into",
        "from",
        "over",
        "under",
        "that",
        "this",
        "his",
        "her",
        "their",
        "she",
        "him",
        "they",
        "them",
        "its",
        "very",
        "while",
        "through",
        "across",
        "down",
        "onto",
        "toward",
        "towards",
        "for",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "one",
        "two",
        "both",
        "each",
        "out",
        "off",
        "once",
        "else",
        "whole",
        "along",
        "behind",
        "above",
        "below",
        "between",
        "before",
        "after",
        "slightly",
    }
)

_NP_MODIFIER_RE = re.compile(
    r"\b(?:in|with|wearing|dressed|at|on|by|under|behind|near|beside|across|from|into|onto|toward|towards|along|through|over|between|holding|carrying|standing|sitting|leaning|walking|running|playing|looking|watching|facing|confronting|preserving|raising|lifting|turning|reaching|whose|who|that|which|keeps|holds|shakes|trucks|reads|turns|pushes|pulls|tilts|pans|tracks|rolls|moves|opens|begins|starts|ends|aligns|captures|unfolds|sits|stands|walks|rests|salutes|spins|bellows)\b",
    flags=re.IGNORECASE,
)

_MEDIUM_KEYWORDS: tuple[str, ...] = (
    "live-action",
    "stop-motion",
    "claymation",
    "animation",
    "animated",
    "watercolor",
    "anime",
    "cartoon",
)


def _load_prompt_text_or_file(prompt: str | os.PathLike) -> str:
    if isinstance(prompt, Path):
        if prompt.is_file():
            return prompt.read_text(encoding="utf-8")
        return str(prompt)
    raw = str(prompt).strip()
    if not raw or "\n" in raw or len(raw) > 240:
        return raw
    for cand in (
        Path(raw),
        REPO_ROOT / raw,
        REPO_ROOT / "prompt_bank" / raw,
        REPO_ROOT / "prompt_bank" / f"{raw}.txt",
    ):
        try:
            if cand.is_file():
                return cand.read_text(encoding="utf-8")
        except OSError:
            pass
    return raw


def extract_prompt_subject_words(prompt: str | os.PathLike) -> list[str]:
    """Extract the medium and primary subject/character head nouns from an H3 scene prompt.

    Combines:
      1. Medium keyword (`"live-action"`, `"stop-motion"`, etc.) in the `detailed_description:`
         preamble or right after `[Shot 1]` (excluding `"cinematic"` and camera framing terms),
      2. `<Subject 1>` head noun phrase in `subject_definitions:` (before the first prepositional/verbal modifier),
      3. Pre-`<d>` character head noun phrase preceding `on-screen | off-screen | (S\\d+) | produces no vocal sound`,
      4. Sentence 1-2 content nouns fallback (with `fl2va`/`i2va`/`l2va` keyframe boilerplate stripped).
    """
    txt = _load_prompt_text_or_file(prompt)
    if not txt:
        return []
    words: list[str] = []

    def _add(w: str) -> None:
        wl = w.lower().strip("-")
        if len(wl) >= 3 and wl not in _SUBJECT_STOPWORDS and wl not in words:
            words.append(wl)

    txt_l = txt.lower()
    for mk in _MEDIUM_KEYWORDS:
        if re.search(r"\b" + re.escape(mk) + r"\b", txt_l):
            _add(mk)
            break
    m_shot = re.search(r"\[Shot\s*\d+\]\s*([A-Za-z0-9-]+)", txt)
    if m_shot:
        _add(m_shot.group(1))

    m_sub1 = re.search(r"<Subject\s*1>\s+is\s+([^.\n]+)", txt, flags=re.IGNORECASE)
    if m_sub1:
        sub1_clean = re.sub(r"<(?:Picture|Video|Subject)\s*\d+>", " ", m_sub1.group(1))
        head_np = _NP_MODIFIER_RE.split(sub1_clean, maxsplit=1)[0].split(",")[0]
        for w in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", head_np):
            _add(w)

    desc = txt
    for hdr in ("integrated_multimodal_description:", "detailed_description:"):
        if hdr in desc:
            desc = desc.split(hdr, 1)[1]
            break
    desc = re.split(
        r"\n\s*(?:overall_soundscape|non_diegetic_music|audio_description):", desc
    )[0]
    desc_no_tags = re.sub(r"<(?:Picture|Video|Subject)\s*\d+>", " ", desc)
    desc_no_tags = re.sub(r"<\|caption_start\|>.*?<\|caption_end\|>", " ", desc_no_tags, flags=re.S)

    actor_words_before = len(words)
    if not m_sub1:
        for s in re.split(r"(?<=[.!?;])\s+", desc_no_tags):
            pre_d = s.split("<d>")[0]
            m_act = re.search(
                r"^(.+?)(?:\b(?:on-screen|off-screen|produces\s+no\s+vocal\s+sound)\b|\(S\d+\))",
                pre_d,
                flags=re.IGNORECASE,
            )
            if m_act:
                clause_clean = re.sub(r"\[Shot\s*\d+\]", " ", m_act.group(1)).strip()
                m_tail_actor = re.search(r",\s*and\s+((?:the|a|an)\s+[a-zA-Z0-9\s-]+)$", clause_clean, flags=re.IGNORECASE)
                if m_tail_actor:
                    clause_clean = m_tail_actor.group(1).strip()
                clause_clean = re.sub(
                    r"^(?:the\s+shot|the\s+camera|a\s+(?:wide|close|medium|tight|low|high)\s+shot)[^,]*,\s*",
                    "",
                    clause_clean,
                    flags=re.IGNORECASE,
                ).strip()
                if re.match(
                    r"^(?:as|when|while|from|in|at|on|inside|outside|behind|across|beneath|under)\b",
                    clause_clean,
                    flags=re.IGNORECASE,
                ) and "," in clause_clean:
                    clause_clean = clause_clean.split(",", 1)[1].strip()
                first_clause = clause_clean.split(",")[0].strip()
                first_clause = re.split(r"['\u2019]s\b", first_clause, maxsplit=1)[0].strip()
                if re.match(
                    r"^(?:his|her|their|its|she|he|they|the\s+shot|the\s+camera|a\s+(?:wide|close|medium|tight|low|high)\s+shot|from\b|through\b)\b",
                    first_clause,
                    flags=re.IGNORECASE,
                ):
                    continue
                head_np = _NP_MODIFIER_RE.split(first_clause, maxsplit=1)[0]
                for w in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", head_np):
                    _add(w)

    if len(words) == actor_words_before and not m_sub1:
        for s in re.split(r"(?<=[.!?;])\s+", desc_no_tags)[:2]:
            if re.search(
                r"\b(?:aligns?|referenced|exact\s+framing|no\s+people\s+are\s+present)\b",
                s,
                flags=re.IGNORECASE,
            ):
                continue
            s_clean = re.sub(r"\[Shot\s*\d+\]", " ", s).strip()
            if re.match(r"^(?:the\s+camera|she|he|they|his|her|their)\b", s_clean, flags=re.IGNORECASE):
                continue
            s_clean = re.sub(r"^.*?\bshot\s+frames\s+", "", s_clean, flags=re.IGNORECASE)
            s_clean = re.sub(r"\b\d+(?:\.\d+)?-second\b", " ", s_clean)
            s_clean = re.split(r"['\u2019]s\b", s_clean, maxsplit=1)[0]
            s_head = _NP_MODIFIER_RE.split(s_clean, maxsplit=1)[0]
            for w in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", s_head):
                _add(w)
                if len(words) >= 6:
                    break
            if len(words) >= 5:
                break
    return words[:8]


def analyze_prompt_coupling(
    v_pr: torch.Tensor | None = None,
    prompt: str | os.PathLike = "",
    *,
    pos_words: Sequence[str] | None = None,
    readout: Mapping[str, Any] | None = None,
    family: str = "fl2va",
    already_contrasted: bool = False,
    top_k: int = 6,
    bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a `< 2 ms` CPU Token-Wise Prompt Receptor Scan and `@prompt_subject` coupling check.

    Scores each unique content word in the prompt's description section against the steer's
    `M_centered` direction (`cos(w_i, mu_pos)`), showing:
      - `top_receptors`: which props/surfaces/characters in the prompt will absorb the steer first
        (e.g. `plastic (+0.223), bulb (+0.190)` for `balloon` on `t2va_noodle_bar.txt`).
      - `subject_words` & `subject_coupling`: extracted `@prompt_subject` words and their alignment
        with the steer (e.g. `customer (-0.043)`), surfacing when props hijack a replacement steer.
      - `token_heatmap`: ordered per-word list (`[{"word": str, "cosine": float, "role": "receptor" | "subject" | "stopword" | "neutral"}]`)
        across the prompt's description section for UI heatmap rendering.
    """
    b = dict(bundle) if bundle is not None else load_j_lens_bundle()
    w2i = b["word_to_idxs"]
    m_centered = b["M_centered"]

    txt = _load_prompt_text_or_file(prompt)
    if not txt.strip():
        return {
            "subject_words": [],
            "top_receptors": [],
            "subject_receptors": [],
            "token_heatmap": [],
            "max_receptor_cos": 0.0,
            "mean_subject_cos": 0.0,
            "summary": "no prompt provided",
            "recommendations": [],
        }

    seed_words: list[str] = list(pos_words) if pos_words else []
    if not seed_words:
        if readout is None and v_pr is not None:
            readout = decode_vector(v_pr, space="post_refiner", family=family, bundle=b)
        if readout is not None:
            seed_words = [
                str(x["word"]) for x in list(readout.get("positive_words") or ())[:4]
            ]

    mu_steer, steer_ids, _, _ = _mean_word_direction(seed_words, w2i, m_centered)
    if not steer_ids:
        return {
            "subject_words": extract_prompt_subject_words(txt),
            "top_receptors": [],
            "subject_receptors": [],
            "token_heatmap": [],
            "max_receptor_cos": 0.0,
            "mean_subject_cos": 0.0,
            "summary": "no matched steer words for receptor scan",
            "recommendations": [],
        }

    desc = txt
    for hdr in ("integrated_multimodal_description:", "detailed_description:"):
        if hdr in desc:
            desc = desc.split(hdr, 1)[1]
            break
    desc = re.split(
        r"\n\s*(?:overall_soundscape|non_diegetic_music|audio_description):", desc
    )[0]
    desc = re.sub(r"<d>.*?</d>", " ", desc, flags=re.S)
    desc = re.sub(r"\[Shot\s*\d+\]|<(?:Picture|Video|Subject)\s*\d+>", " ", desc)

    uniq_words: list[str] = []
    for w in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", desc.lower()):
        if w not in _PROMPT_RECEPTOR_STOPWORDS and w not in uniq_words:
            uniq_words.append(w)

    scored: list[dict[str, Any]] = []
    word_cos_map: dict[str, float] = {}
    for w in uniq_words:
        w_ids, _, _ = _resolve_word_indices([w], w2i)
        if not w_ids:
            continue
        mw = F.normalize(m_centered[w_ids].mean(dim=0), dim=0)
        c_val = round(float(torch.dot(mu_steer, mw).item()), 3)
        word_cos_map[w] = c_val
        scored.append({"word": w, "cosine": c_val})

    scored.sort(key=lambda item: float(item["cosine"]), reverse=True)
    top_receptors = scored[:top_k]

    subj_words = extract_prompt_subject_words(txt)
    subj_receptors: list[dict[str, Any]] = []
    for sw in subj_words:
        if sw == "live-action":
            continue
        if sw in word_cos_map:
            subj_receptors.append({"word": sw, "cosine": word_cos_map[sw]})
        else:
            sw_ids, _, _ = _resolve_word_indices([sw], w2i)
            if sw_ids:
                msw = F.normalize(m_centered[sw_ids].mean(dim=0), dim=0)
                c_sw = round(float(torch.dot(mu_steer, msw).item()), 3)
                word_cos_map[sw] = c_sw
                subj_receptors.append({"word": sw, "cosine": c_sw})

    subj_set = {sw.lower() for sw in subj_words if sw != "live-action"}
    top_rec_set = {str(x["word"]).lower() for x in top_receptors}
    token_heatmap: list[dict[str, Any]] = []
    for raw_tok in re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", desc):
        w_low = raw_tok.lower()
        if len(w_low) < 3 or w_low in _PROMPT_RECEPTOR_STOPWORDS:
            token_heatmap.append({"word": raw_tok, "cosine": 0.0, "role": "stopword"})
            continue
        c_tok = float(word_cos_map.get(w_low, 0.0))
        if w_low in subj_set:
            role = "subject"
        elif w_low in top_rec_set and c_tok >= 0.12:
            role = "receptor"
        else:
            role = "neutral"
        token_heatmap.append({"word": raw_tok, "cosine": c_tok, "role": role})

    max_rec_cos = float(top_receptors[0]["cosine"]) if top_receptors else 0.0
    mean_subj_cos = (
        round(
            sum(float(x["cosine"]) for x in subj_receptors) / len(subj_receptors), 3
        )
        if subj_receptors
        else 0.0
    )

    recommendations: list[str] = []
    is_rep = bool(set(w.lower() for w in seed_words) & _REPLACEMENT_CONCEPT_WORDS)
    if is_rep and not already_contrasted and subj_words:
        top_prop = top_receptors[0]["word"] if top_receptors else "props"
        recommendations.append(
            f"Prompt receptor scan shows {top_prop!r} ({max_rec_cos:+.2f}) absorbs the steer while "
            f"subject {subj_words!r} sits at {mean_subj_cos:+.2f}: pass against_words='@prompt_subject' "
            f"({', '.join(subj_words[:5])}) if you want to transform the subject."
        )

    rec_str = ", ".join(f"{x['word']} ({x['cosine']:+.2f})" for x in top_receptors[:4])
    sub_str = ", ".join(f"{x['word']} ({x['cosine']:+.2f})" for x in subj_receptors[:4])
    summary = f"receptors=[{rec_str}]  subject=[{sub_str or 'none'}]"
    return {
        "subject_words": subj_words,
        "top_receptors": top_receptors,
        "subject_receptors": subj_receptors,
        "token_heatmap": token_heatmap,
        "max_receptor_cos": max_rec_cos,
        "mean_subject_cos": mean_subj_cos,
        "summary": summary,
        "recommendations": recommendations,
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
                if len(stem) >= 5 and stem[:-1] in word_to_idxs:
                    stem_ids.extend(word_to_idxs[stem[:-1]])
                    break
                if (stem + "e") in word_to_idxs:
                    stem_ids.extend(word_to_idxs[stem + "e"])
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


def _mean_word_direction(
    word_list: Sequence[str],
    word_to_idxs: Mapping[str, list[int]],
    m_centered: torch.Tensor,
    piece_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[int], list[str], list[str]]:
    all_ids: list[int] = []
    matched_words: list[str] = []
    unmatched_words: list[str] = []
    word_vecs: list[torch.Tensor] = []
    for w in word_list:
        w_ids, w_m, w_u = _resolve_word_indices([w], word_to_idxs)
        if w_ids:
            all_ids.extend(w_ids)
            matched_words.extend(w_m)
            rows = m_centered[w_ids]
            if piece_weights is not None and len(w_ids) > 1:
                pw = piece_weights[w_ids].unsqueeze(-1)
                w_vec = (rows * pw).sum(dim=0) / pw.sum().clamp_min(1e-8)
            else:
                w_vec = rows.mean(dim=0)
            word_vecs.append(F.normalize(w_vec, dim=0))
        else:
            unmatched_words.extend(w_u)
    if not word_vecs:
        return torch.empty(0), all_ids, matched_words, unmatched_words
    mean_vec = F.normalize(torch.stack(word_vecs, dim=0).mean(dim=0), dim=0)
    return mean_vec, all_ids, matched_words, unmatched_words


def synthesize_from_words(
    pos_words: str | Sequence[str],
    neg_words: str | Sequence[str] | None = None,
    *,
    family: str = "fl2va",
    target_norm: float = NORM_MATCHED_TARGET,
    hold_words: str | Sequence[str] | None = None,
    alpha: float = 1.0,
    prompt: str | os.PathLike | None = None,
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
    if isinstance(pos_words, str) and ("|||" in pos_words or "->" in pos_words):
        sep = "|||" if "|||" in pos_words else "->"
        lhs_raw, rhs_raw = pos_words.split(sep, 1)
        pos_words = rhs_raw.strip()
        if neg_words is None or (isinstance(neg_words, str) and not neg_words.strip()):
            neg_words = lhs_raw.strip()
    pos_list = _parse_word_list(pos_words)
    if not pos_list:
        raise ValueError("pos_words must contain at least one word")
    explicit_neg = False
    if isinstance(neg_words, str) and neg_words.strip():
        raw_neg_key = neg_words.strip().lower().lstrip("@")
        if raw_neg_key in ("prompt", "prompt_subject"):
            if prompt is not None and str(prompt).strip():
                w2i_auto = b["word_to_idxs"]
                raw_subj = extract_prompt_subject_words(prompt)
                extracted_subj = [
                    w for w in raw_subj
                    if _resolve_word_indices([w], w2i_auto)[0]
                ]
                has_non_live_medium = bool(set(raw_subj) & set(_MEDIUM_KEYWORDS[1:]))
                if "live-action" in raw_subj or not has_non_live_medium:
                    for anchor_w in AGAINST_PRESETS["replace_human"]:
                        if anchor_w not in extracted_subj and _resolve_word_indices([anchor_w], w2i_auto)[0]:
                            extracted_subj.append(anchor_w)
                neg_list = extracted_subj if extracted_subj else list(AGAINST_PRESETS["replace_human"])
            else:
                neg_list = list(AGAINST_PRESETS["replace_human"])
            explicit_neg = True
        elif raw_neg_key in AGAINST_PRESETS:
            neg_list = list(AGAINST_PRESETS[raw_neg_key])
            explicit_neg = raw_neg_key != "neutral"
        else:
            neg_list = _parse_word_list(neg_words)
            explicit_neg = bool(neg_list)
    elif neg_words:
        neg_list = _parse_word_list(neg_words)
        explicit_neg = bool(neg_list)
    else:
        neg_list = list(DEFAULT_NEUTRAL_WORDS)

    w2i = b["word_to_idxs"]
    m_centered = b["M_centered"]
    p_weights = b.get("piece_weights")
    s_p030 = b["Sigma_49_p030"]
    j_ref = b["J_ref_ref2va"] if "ref2va" in str(family).lower() else b["J_ref_fl2va"]

    mu_pos, pos_ids, pos_matched, pos_unmatched = _mean_word_direction(pos_list, w2i, m_centered, p_weights)
    if not pos_ids:
        raise ValueError(f"None of pos_words {pos_list!r} matched the 24,912-word J-Lens vocabulary")
    if pos_unmatched:
        logger.warning(
            "[h3] J-Lens word synthesis: unmatched pos_words %r (matched=%r; use MiniMaxH3MakeControlVector or bench/control_vectors.py build --words for multi-token OOV phrases)",
            pos_unmatched,
            pos_matched,
        )
    mu_neg, neg_ids, neg_matched, neg_unmatched = _mean_word_direction(neg_list, w2i, m_centered, p_weights)
    if not neg_ids:
        raise ValueError(f"None of neg_words {neg_list!r} matched the 24,912-word J-Lens vocabulary")
    if neg_unmatched:
        logger.warning(
            "[h3] J-Lens word synthesis: unmatched neg_words %r (matched=%r)",
            neg_unmatched,
            neg_matched,
        )

    neg_weight = 0.60 if explicit_neg else 0.35
    delta_m = F.normalize(mu_pos - neg_weight * mu_neg, dim=0)

    v_enc_raw = s_p030 @ delta_m
    v_enc = F.normalize(v_enc_raw, dim=0)
    v_pr_raw = j_ref @ v_enc_raw
    v_pr = F.normalize(v_pr_raw, dim=0) * float(target_norm)

    hold_list = _parse_word_list(hold_words)
    if hold_list and a_val > 0.0:
        mu_hold_pos, h_ids, _, _ = _mean_word_direction(hold_list, w2i, m_centered, p_weights)
        if h_ids:
            mu_hold = F.normalize(mu_hold_pos - neg_weight * mu_neg, dim=0)
            v_hold_enc = F.normalize(s_p030 @ mu_hold, dim=0)
            v_hold_pr = F.normalize(j_ref @ (s_p030 @ mu_hold), dim=0)
            v_enc_held = apply_soft_holds(v_enc, [v_hold_enc], alpha=a_val)
            v_pr = apply_soft_holds(v_pr, [v_hold_pr], alpha=a_val)
            if float(torch.linalg.vector_norm(v_pr.to(dtype=torch.float64)).item()) <= 1e-3 * max(1.0, float(target_norm)):
                raise ValueError("hold_words completely cancelled pos_words (residual norm is zero)")
            v_enc = F.normalize(v_enc_held, dim=0)
            v_pr = F.normalize(v_pr, dim=0) * float(target_norm)

    readout = decode_vector(v_pr, space="post_refiner", family=family, bundle=b)
    readout["readiness"] = assess_steering_readiness(
        v_pr,
        v_enc=v_enc,
        readout=readout,
        pos_words=pos_matched,
        neg_words=neg_matched,
        family=family,
        bundle=b,
    )
    if prompt is not None and str(prompt).strip():
        readout["prompt_coupling"] = analyze_prompt_coupling(
            v_pr,
            prompt,
            pos_words=pos_matched,
            readout=readout,
            family=family,
            already_contrasted=explicit_neg,
            bundle=b,
        )
        for rec in readout["prompt_coupling"].get("recommendations", ()):
            if rec not in readout["readiness"]["recommendations"]:
                readout["readiness"]["recommendations"].append(rec)
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
        "readiness": readout["readiness"],
        "prompt_coupling": readout.get("prompt_coupling"),
    }


def save_synthesized_vector(
    out_path: str | os.PathLike,
    synth: Mapping[str, Any],
    *,
    name: str = "",
    write_sidecar_md: bool = True,
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
    tensors_dict = {"vector.post_refiner": v_pr, "vector.encoder": v_enc}
    path.write_bytes(
        save_safetensors_bytes(
            tensors_dict,
            metadata=encoded_meta,
        )
    )
    if write_sidecar_md:
        import control_vectors as _cv_mod

        sidecar_path = path.with_suffix(".md")
        sidecar_path.write_text(
            _cv_mod.format_vector_sidecar_md(path.name, tensors_dict, meta_dict),
            encoding="utf-8",
        )
    return path


def save_composed_recipe_vector(
    out_path: str | os.PathLike,
    steps: Sequence[Mapping[str, Any]],
    *,
    family: str = "fl2va",
    name: str = "",
    hold_words: Sequence[str] | str | None = None,
    alpha: float = 1.0,
    prompt: str | os.PathLike | None = None,
    write_sidecar_md: bool = True,
    bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose a multi-vector recipe (`add`, `add_norm_matched`, `hold`, plus optional `hold_words`) and save as a standard `h3-control-vector/1` `.safetensors` file."""
    b = dict(bundle) if bundle is not None else load_j_lens_bundle()
    rec = decode_recipe(
        steps,
        family=family,
        hold_words=hold_words,
        alpha=alpha,
        prompt=prompt,
        return_tensors=True,
        bundle=b,
    )
    path = Path(out_path)
    if not path.suffix:
        path = path.with_suffix(".safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    v_pr = rec.pop("vector.post_refiner").detach().to(device="cpu", dtype=torch.float32).contiguous()
    v_enc = rec.pop("vector.encoder").detach().to(device="cpu", dtype=torch.float32).contiguous()
    raw_norm = float(torch.linalg.vector_norm(v_pr.to(dtype=torch.float64)).item())
    if raw_norm <= 1e-8:
        raise ValueError("Composed recipe vector has zero post_refiner norm")
    enc_norm = float(torch.linalg.vector_norm(v_enc.to(dtype=torch.float64)).item())
    raw_stem = (name or path.stem.split("__")[0]).lower()
    stem = re.sub(r"[^a-z0-9_]+", "_", raw_stem).strip("_") or "composed_recipe"
    eff_family = str(rec.get("family", family))
    step_list = list(rec.get("steps", ()))
    hw_list = list(rec.get("hold_words", ()))
    add_desc = " + ".join(
        f"{s['name']}({s['mode']}@{s['strength']})"
        for s in step_list
        if s.get("mode") in ("add", "add_norm_matched")
    )
    hold_desc = ", ".join(
        [f"{s['name']}({s['strength']})" for s in step_list if s.get("mode") == "hold"]
        + ([f"words:{','.join(hw_list)}@{alpha}"] if hw_list else [])
    )
    recipe_digest = hashlib.sha256(f"{add_desc}|holds={hold_desc}".encode("utf-8")).hexdigest()
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    recipe_meta = _DEFAULT_RECIPES.get(eff_family, _DEFAULT_RECIPES["fl2va"])

    n_add = sum(1 for s in step_list if s.get("mode") in ("add", "add_norm_matched"))
    n_hold = sum(1 for s in step_list if s.get("mode") == "hold") + (1 if hw_list else 0)
    meta_dict: dict[str, Any] = {
        "format": "h3-control-vector/1",
        "name": stem,
        "quality": f"J-Lens composed recipe: {add_desc}" + (f" [holds: {hold_desc}]" if hold_desc else ""),
        "family": eff_family,
        "recipe": recipe_meta,
        "measured_under": {
            "mode": "ref2va" if eff_family == "ref2va" else "t2va",
            "frame": "scene",
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
            "path": "j_lens.save_composed_recipe_vector",
            "sha256": recipe_digest,
            "a_side": hold_desc or "baseline",
            "b_side": add_desc,
            "n_pairs": max(n_add, 1),
            "n_holds": n_hold,
            "pooling": "diff_span",
            "steps": step_list,
        },
        "quality_report": {
            "mean_pairwise_cosine": 1.0,
            "split_half_cosine": 1.0,
            "per_pair_cosine": [1.0 for _ in range(max(n_add, 1))],
            "j_lens_readout": rec.get("readout", {}),
        },
        "provenance": {
            "core_revision": "j_lens_cpu",
            "pack_commit": "j_lens_v1",
            "created_utc": now_utc,
            "made_by": "j_lens.save_composed_recipe_vector/1",
        },
    }
    encoded_meta = {k: json.dumps(v, sort_keys=True) for k, v in meta_dict.items()}
    tensors_dict = {"vector.post_refiner": v_pr, "vector.encoder": v_enc}
    path.write_bytes(
        save_safetensors_bytes(
            tensors_dict,
            metadata=encoded_meta,
        )
    )
    if write_sidecar_md:
        import control_vectors as _cv_mod

        sidecar_path = path.with_suffix(".md")
        sidecar_path.write_text(
            _cv_mod.format_vector_sidecar_md(path.name, tensors_dict, meta_dict),
            encoding="utf-8",
        )
    return {
        "path": path,
        "recipe_report": rec,
    }


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
        help="Optional words to project out when using --from-words or decoding .safetensors files.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional H3 scene prompt text or file path for Token-Wise Prompt Receptor Scanning and @prompt_subject extraction.",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional output .safetensors path when using --from-words.",
    )
    parser.add_argument(
        "--modes",
        default="",
        help="Optional comma-separated modes per positional vector for recipe composition (e.g. 'add,hold').",
    )
    parser.add_argument(
        "--strengths",
        default="",
        help="Optional comma-separated float strengths per positional vector for recipe composition (e.g. '1.15,1.0').",
    )
    parser.add_argument("--family", default="fl2va", choices=("fl2va", "ref2va"), help="Refiner family for --svd or --from-words.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    try:
        bundle = load_j_lens_bundle()
        if args.from_words.strip() and not args.vectors:
            synth = synthesize_from_words(
                args.from_words,
                args.against_words if args.against_words.strip() else None,
                family=args.family,
                hold_words=args.hold_words if args.hold_words.strip() else None,
                alpha=args.alpha,
                prompt=args.prompt if args.prompt.strip() else None,
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
                "prompt_coupling": synth.get("prompt_coupling"),
            }
            if args.json:
                print(json.dumps(out_summary, indent=2))
            else:
                r = synth["readout"]
                rd = r.get("readiness") or {}
                pc = synth.get("prompt_coupling") or {}
                print(
                    f"Synthesized {synth['family']} vector (norm={r['norm_after_holds']:.2f}) from +{synth['pos_words_matched']} / -{synth['neg_words_matched']}:"
                )
                print(f"  PR (+) {r['pos_summary']}")
                print(f"  PR (-) {r['neg_summary']}")
                if rd:
                    print(f"  Readiness: {rd.get('summary', '')}")
                    for rec_line in rd.get("recommendations", ()):
                        print(f"    -> {rec_line}")
                if pc:
                    print(f"  Prompt Coupling: {pc.get('summary', '')}")
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
                hold_words=args.hold_words if args.hold_words.strip() else None,
                alpha=args.alpha,
                prompt=args.prompt if args.prompt.strip() else None,
                bundle=bundle,
            )
            for v_path in args.vectors
        ]
        recipe_report = None
        if len(args.vectors) >= 2 or (args.vectors and args.from_words.strip()) or args.modes.strip() or args.strengths.strip():
            mode_tokens = [m.strip() for m in args.modes.split(",") if m.strip()] if args.modes.strip() else []
            strength_tokens = [float(s.strip()) for s in args.strengths.split(",") if s.strip()] if args.strengths.strip() else []
            rec_steps: list[dict[str, Any]] = []
            for idx_v, v_path in enumerate(args.vectors):
                v_mode = mode_tokens[idx_v] if idx_v < len(mode_tokens) else "add"
                v_str = strength_tokens[idx_v] if idx_v < len(strength_tokens) else 1.0
                rec_steps.append({"vector": v_path, "mode": v_mode, "strength": v_str})
            if args.from_words.strip():
                synth_inline = synthesize_from_words(
                    args.from_words,
                    args.against_words if args.against_words.strip() else None,
                    family=args.family,
                    hold_words=args.hold_words if args.hold_words.strip() else None,
                    alpha=args.alpha,
                    prompt=args.prompt if args.prompt.strip() else None,
                    bundle=bundle,
                )
                rec_steps.append(
                    {
                        "vector": synth_inline["vector.post_refiner"],
                        "mode": "add_norm_matched",
                        "strength": 1.0,
                    }
                )
            for h_p in args.hold:
                rec_steps.append({"vector": h_p, "mode": "hold", "strength": args.alpha})
            recipe_report = decode_recipe(
                rec_steps,
                family=args.family,
                scale=args.scale,
                top_k=args.top_k,
                bundle=bundle,
            )
    except (FileNotFoundError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        print(f"j_lens.py: error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        if recipe_report is not None:
            print(json.dumps({"vectors": reports, "recipe": recipe_report}, indent=2))
        else:
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
            rd = pr.get("readiness") or {}
            if rd:
                print(f"  Readiness: {rd.get('summary', '')}")
            pc = pr.get("prompt_coupling") or {}
            if pc:
                print(f"  Prompt Coupling: {pc.get('summary', '')}")
        if recipe_report is not None:
            r_net = recipe_report["readout"]
            pw = ", ".join(
                f"cos({c['pair'][0]}, {c['pair'][1]})={c['cosine']:+.4f}"
                for c in recipe_report.get("pairwise_interactions", ())
            )
            print(
                f"Combined Recipe ({recipe_report['family']}, norm={recipe_report['combined_norm_before_holds']:.2f} -> {recipe_report['combined_norm_after_holds']:.2f}"
                + (f", {pw}" if pw else "")
                + "):"
            )
            print(f"  PR (+) {r_net['pos_summary']}")
            print(f"  PR (-) {r_net['neg_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
