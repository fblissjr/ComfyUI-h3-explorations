"""UI state management, TOML/Markdown preview, and `< 1 ms` CPU cached-build repooling."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

import control_vector_nodes as cv_nodes
import control_vectors as cv
import h3_pair_compiler as pair_compiler
import h3_spans
import j_lens
import substrate
from workflows import h3_config

from .llm_pairs import _repair_multichange_markdown_pairs
from . import helpers as _h
from .helpers import (
    REPO_ROOT,
    _apply_post_refiner_holds,
    _cosine,
    _inject_target_prompt_into_pairset,
    _is_markdown_pairs_text,
    _pair_set_to_markdown,
    _recommended_strength_text,
)

class ControlVectorUIState:
    def __init__(
        self,
        *,
        toml_dir: Path | None = None,
        vec_dir: Path | None = None,
        default_encoder: str | None = None,
        default_dit: str | None = None,
        default_device: str | None = None,
        default_llm_provider: str | None = None,
        default_llm_model: str | None = None,
    ) -> None:
        self.toml_dir: Path = toml_dir or Path(
            os.environ.get("H3_CV_TOML_DIR", "").strip() or cv_nodes.PAIR_BANK_DIR
        )
        self.vec_dir: Path = vec_dir or Path(
            os.environ.get("H3_CV_VEC_DIR", "").strip()
            or _h.folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0]
        )
        self.default_encoder: str = (
            default_encoder
            or os.environ.get("H3_ENCODER", "").strip()
            or h3_config.MODELS["clip"]
        )
        self.default_dit: str | None = (
            default_dit or os.environ.get("H3_DIT", "").strip() or None
        )
        self.default_device: str = (
            default_device or os.environ.get("H3_DEVICE", "").strip() or "default"
        )
        self.default_llm_provider: str = (
            default_llm_provider
            or os.environ.get("H3_LLM_PROVIDER", "").strip()
            or ("exllamav3" if os.environ.get("H3_EXL3_MODEL_DIR", "").strip() else "transformers")
        )
        self.default_llm_model: str = (
            default_llm_model
            or os.environ.get("H3_LLM_MODEL", "").strip()
            or os.environ.get("H3_EXL3_MODEL_DIR", "").strip()
        )
        self._tokenizer: Any = None
        self._clips: dict[tuple[str, str], Any] = {}
        self._refiner_cache: dict[Any, Any] = {}
        self._llm_cache: dict[tuple[str, str, bool], Any] = {}
        self.last_build_cache: dict[str, Any] = {}
        self.shutdown_requested: bool = False

    def get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            import comfy.text_encoders.minimax as mm_te

            self._tokenizer = mm_te.MiniMaxH3Tokenizer()
        return self._tokenizer

    def release_llm_vram(self) -> None:
        if not self._llm_cache:
            return
        import gc

        for entry in list(self._llm_cache.values()):
            try:
                if isinstance(entry, tuple) and len(entry) >= 1 and hasattr(entry[0], "unload"):
                    entry[0].unload()
            except Exception:
                pass
        self._llm_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_clip_vram(self) -> None:
        if not self._clips:
            return
        import gc

        try:
            import comfy.model_management as mm

            mm.unload_all_models()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_clip(self, encoder_name: str | None = None, device: str | None = None) -> Any:
        self.release_llm_vram()
        resolved_enc = encoder_name or self.default_encoder or h3_config.MODELS["clip"]
        resolved_dev = device or self.default_device or "default"
        key = (str(resolved_enc), str(resolved_dev))
        if key not in self._clips:
            self._clips[key] = _h._load_clip(resolved_enc, device=str(resolved_dev))
        clip_obj = self._clips[key]
        csm = getattr(clip_obj, "cond_stage_model", None)
        if csm is not None and not hasattr(csm, "memory_estimation_function"):
            csm.memory_estimation_function = lambda _tokens, device=None: 2560 * 1024 * 1024
        return clip_obj



def _list_ui_toml_files(toml_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not toml_dir.is_dir():
        return out
    for p in sorted(toml_dir.glob("*.toml")):
        try:
            ps = cv.load_pairs(p, bank_dir=cv_nodes.PROMPT_BANK_DIR)
            out.append(
                {
                    "name": ps.name,
                    "stem": p.stem,
                    "file": p.name,
                    "pairs": len(ps.pairs),
                    "n_pairs": len(ps.pairs),
                    "holds": len(ps.holds),
                    "n_holds": len(ps.holds),
                    "mode": ps.mode,
                    "frame": ps.frame,
                    "quality": ps.quality,
                    "a_side": ps.a_side,
                    "b_side": ps.b_side,
                }
            )
        except Exception:
            continue
    return out


def _list_ui_vectors(vec_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not vec_dir.is_dir():
        return out
    for p in sorted(vec_dir.glob("*.safetensors")):
        try:
            _, meta = cv.load_vector(p)
            pr_meta = meta.get("points", {}).get("post_refiner", {})
            raw_norm = float(pr_meta.get("raw_norm", 0.0))
            qr = meta.get("quality_report", {})
            mpc = float(qr.get("mean_pairwise_cosine", 0.0))
            shc = float(qr.get("split_half_cosine", 0.0))
            pairs_meta = meta.get("pairs", {})
            loras_list = [
                str(l.get("file", ""))
                for l in meta.get("recipe", {}).get("loras", ())
                if isinstance(l, dict) and l.get("file")
            ]
            out.append(
                {
                    "file": p.name,
                    "name": str(meta.get("name", p.stem)),
                    "family": str(meta.get("family", "")),
                    "pooling": str(meta.get("measured_under", {}).get("pooling", "scene_text")),
                    "n_pairs": int(pairs_meta.get("n_pairs", 0)),
                    "n_holds": int(pairs_meta.get("n_holds", 0)),
                    "norm": raw_norm,
                    "raw_norm": raw_norm,
                    "mpc": mpc,
                    "mean_cosine": mpc,
                    "mean_pairwise_cosine": mpc,
                    "split_half": shc,
                    "split_half_cosine": shc,
                    "rec_strength": _recommended_strength_text(raw_norm),
                    "loras": loras_list,
                    "j_lens": qr.get("j_lens_readout"),
                }
            )
        except Exception:
            continue
    return out



def _preview_markdown_pairs(
    state: ControlVectorUIState,
    text: str,
    *,
    frame: str = "scene",
    name: str = "",
    slot: str = "auto",
    mode: str = "t2va",
) -> dict[str, Any]:
    if frame not in cv.VALID_FRAMES:
        return {"ok": False, "error": f"frame must be one of {cv.VALID_FRAMES}, got {frame!r}"}
    inferred_slot: str | None = None
    try:
        if _is_markdown_pairs_text(text):
            text = _repair_multichange_markdown_pairs(text)
            parsed = cv.parse_markdown_pairs(text)
        else:
            spec_hdr = f"slot: {slot}\nmode: {mode}\n" if slot != "auto" or mode != "t2va" else ""
            spec = pair_compiler.parse_short_contrast_spec(spec_hdr + text)
            ps = spec["pair_set"]
            inferred_slot = spec["slot"]
            if not name:
                name = ps.name
            parsed = {
                "mode": ps.mode,
                "test_scene_prompt": "",
                "pairs": [{"a": p.a, "b": p.b} for p in ps.pairs],
                "holds": [],
            }
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    tok = state.get_tokenizer()
    hf_tok = tok.qwen3vl_32b.tokenizer
    decode_fn = lambda ids: hf_tok.decode(list(ids))

    pair_rows: list[dict[str, Any]] = []
    ds_counts: list[float] = []
    ffd_counts: list[float] = []
    scene_counts: list[float] = []
    first_valid_spans: tuple[str, str] | None = None

    for idx, item in enumerate(parsed["pairs"], start=1):
        a_txt = item["a"]
        b_txt = item["b"]
        try:
            a_span, b_span = cv.differing_span(a_txt, b_txt, frame=frame)
            sec_idx, sec_name = cv.differing_section_info(a_txt, b_txt, frame=frame)
            if first_valid_spans is None:
                first_valid_spans = (a_span, b_span)
            c_prefix = len(os.path.commonprefix([a_txt, b_txt]))
            pos = a_txt.find(a_span, max(0, c_prefix - len(a_span)))
            if pos < 0:
                pos = a_txt.find(a_span)
            prefix = a_txt[:pos] if pos >= 0 else ""
            suffix = a_txt[pos + len(a_span) :] if pos >= 0 else ""

            entries_a = tok.tokenize_with_weights(a_txt)["qwen3vl_32b"][0]
            entries_b = tok.tokenize_with_weights(b_txt)["qwen3vl_32b"][0]
            ids_a = [int(e[0]) if not isinstance(e[0], dict) else 0 for e in entries_a]
            ids_b = [int(e[0]) if not isinstance(e[0], dict) else 0 for e in entries_b]
            tags_a = [1] * len(ids_a)
            tags_b = [1] * len(ids_b)
            first_diff, end_a, end_b = h3_spans.differing_token_bounds(ids_a, ids_b)
            spans_a = h3_spans.spans(ids_a, tags_a, decode_fn)
            spans_b = h3_spans.spans(ids_b, tags_b, decode_fn)

            m_ds_a = int(
                h3_spans.pooling_mask(
                    spans_a, "diff_span", first_diff=min(first_diff, len(ids_a) - 1), end_diff=end_a
                ).sum().item()
            )
            m_ds_b = int(
                h3_spans.pooling_mask(
                    spans_b, "diff_span", first_diff=min(first_diff, len(ids_b) - 1), end_diff=end_b
                ).sum().item()
            )
            m_ffd_a = int(
                h3_spans.pooling_mask(
                    spans_a, "from_first_diff", first_diff=min(first_diff, len(ids_a) - 1)
                ).sum().item()
            )
            m_ffd_b = int(
                h3_spans.pooling_mask(
                    spans_b, "from_first_diff", first_diff=min(first_diff, len(ids_b) - 1)
                ).sum().item()
            )
            m_sc_a = int(h3_spans.pooling_mask(spans_a, "scene_text").sum().item())
            m_sc_b = int(h3_spans.pooling_mask(spans_b, "scene_text").sum().item())

            ds_counts.append(0.5 * (m_ds_a + m_ds_b))
            ffd_counts.append(0.5 * (m_ffd_a + m_ffd_b))
            scene_counts.append(0.5 * (m_sc_a + m_sc_b))

            pair_rows.append(
                {
                    "index": idx,
                    "valid": True,
                    "section": f"sec {sec_idx}:{sec_name}",
                    "prefix": prefix,
                    "a_span": a_span,
                    "b_span": b_span,
                    "diff_a": a_span,
                    "diff_b": b_span,
                    "suffix": suffix,
                    "tokens": {
                        "first_diff": int(first_diff),
                        "diff_span": m_ds_b,
                        "scene_text": m_sc_b,
                        "diff_span_a": m_ds_a,
                        "diff_span_b": m_ds_b,
                        "ffd_a": m_ffd_a,
                        "ffd_b": m_ffd_b,
                        "scene_a": m_sc_a,
                        "scene_b": m_sc_b,
                    },
                }
            )
        except ValueError as exc:
            pair_rows.append(
                {
                    "index": idx,
                    "valid": False,
                    "error": str(exc),
                }
            )

    quality = ""
    if first_valid_spans is not None:
        quality = f"{first_valid_spans[1][:72]} vs {first_valid_spans[0][:72]}"

    avg_tokens = None
    if ds_counts:
        avg_tokens = {
            "diff_span": round(sum(ds_counts) / len(ds_counts), 1),
            "from_first_diff": round(sum(ffd_counts) / len(ffd_counts), 1),
            "scene_text": round(sum(scene_counts) / len(scene_counts), 1),
        }

    lint_warnings: list[str] = []
    if all(r.get("valid", False) for r in pair_rows):
        try:
            lint_warnings = cv.lint_pair_set(parsed["pairs"], frame=frame)
        except Exception:
            lint_warnings = []

    return {
        "ok": True,
        "name": name,
        "suggested_name": name,
        "mode": parsed["mode"],
        "frame": frame,
        "n_pairs": len(pair_rows),
        "n_holds": len(parsed.get("holds", [])),
        "inferred_slot": inferred_slot,
        "test_scene_prompt": parsed["test_scene_prompt"],
        "test_prompt": parsed["test_scene_prompt"],
        "quality": quality,
        "avg_tokens": avg_tokens,
        "lint_warnings": lint_warnings,
        "pairs": pair_rows,
    }


def _build_pairset_from_ui_rows(
    rows: list[dict[str, Any]],
    *,
    holds: list[dict[str, Any]] | None = None,
    name: str = "",
    slot: str = "auto",
    mode: str = "t2va",
    frame: str = "scene",
    target_prompt: str = "",
    target_weight: float = 2.5,
) -> cv.PairSet:
    """Build a `cv.PairSet` from structured 10-Row Scene-Diff Table rows (`rows` + optional `holds`)."""
    active_rows = [r for r in rows if bool(r.get("enabled", True)) and str(r.get("b", "")).strip()]
    if not active_rows:
        raise ValueError("at least one enabled row with a non-empty Pole B is required")

    first_b = str(active_rows[0].get("b", "")).strip()
    is_full_scene = (
        frame == "bare"
        or "integrated_multimodal_description:" in first_b
        or "detailed_description:" in first_b
    )
    if is_full_scene:
        pair_items: list[cv.PairItem] = []
        for idx, r in enumerate(active_rows):
            a_txt = str(r.get("a", "")).strip()
            b_txt = str(r.get("b", "")).strip()
            w_val = float(r.get("weight", 1.0))
            a_sp, b_sp = cv.differing_span(a_txt, b_txt, frame=frame, label=f"row[{idx}]")
            pair_items.append(
                cv.PairItem(
                    index=idx,
                    weight=w_val,
                    a=a_txt,
                    b=b_txt,
                    a_span=a_sp,
                    b_span=b_sp,
                )
            )
        hold_items: list[cv.HoldItem] = []
        for h_idx, hr in enumerate(holds or ()):
            ha = str(hr.get("a", "")).strip()
            hb = str(hr.get("b", "")).strip()
            if not ha or not hb:
                continue
            ha_sp, hb_sp = cv.differing_span(ha, hb, frame=frame, label=f"hold[{h_idx}]")
            hold_items.append(
                cv.HoldItem(
                    index=h_idx,
                    group=str(hr.get("group", "nuisance") or "nuisance"),
                    weight=float(hr.get("weight", 1.0)),
                    a=ha,
                    b=hb,
                    a_span=ha_sp,
                    b_span=hb_sp,
                )
            )
        a_side_val = pair_items[0].a_span[:72]
        b_side_val = pair_items[0].b_span[:72]
        resolved_name = (name or pair_compiler._slugify_name(b_side_val)).strip()
        if not cv.NAME_RE.match(resolved_name):
            raise ValueError(f"vector name must match [a-z0-9_]+, got {resolved_name!r}")
        digest = hashlib.sha256(
            json.dumps(
                {
                    "name": resolved_name,
                    "mode": mode,
                    "pairs": [(p.weight, p.a, p.b) for p in pair_items],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        ps = cv.PairSet(
            path=Path(f"<workbench:{resolved_name}>"),
            sha256=digest,
            name=resolved_name,
            quality=f"{b_side_val} vs. {a_side_val}",
            a_side=a_side_val,
            b_side=b_side_val,
            mode=mode,
            frame=frame,
            exclude_shots=(),
            pairs=tuple(pair_items),
            holds=tuple(hold_items),
        )
        if str(target_prompt or "").strip():
            ps = _inject_target_prompt_into_pairset(
                ps, target_prompt, target_weight=float(target_weight), slot_hint=slot
            )
        return ps

    clauses: list[pair_compiler.ContrastClause] = []
    sound_clauses: list[pair_compiler.ContrastClause] = []
    for r in active_rows:
        r_slot = str(r.get("slot") or slot or "auto")
        clauses.append(
            pair_compiler.ContrastClause(
                a=str(r.get("a", "")).strip(),
                b=str(r.get("b", "")).strip(),
                slot=r_slot if r_slot != "auto" else None,
                weight=float(r.get("weight", 1.0)),
            )
        )
        sb = str(r.get("sound_b", "") or "").strip()
        sa = str(r.get("sound_a", "") or "").strip()
        if sb:
            sound_clauses.append(
                pair_compiler.ContrastClause(
                    a=sa,
                    b=sb,
                    slot="soundscape",
                    weight=float(r.get("weight", 1.0)),
                )
            )
    hold_clauses: list[pair_compiler.ContrastClause] = []
    for hr in (holds or ()):
        ha = str(hr.get("a", "") or "").strip()
        hb = str(hr.get("b", "") or "").strip()
        if hb:
            hold_clauses.append(
                pair_compiler.ContrastClause(
                    a=ha,
                    b=hb,
                    slot="auto",
                    group=str(hr.get("group", "nuisance") or "nuisance"),
                    weight=float(hr.get("weight", 1.0)),
                )
            )
    tp_clean = str(target_prompt or "").strip() or None
    return pair_compiler.compile_words_to_pairset(
        clauses,
        name=name or None,
        slot=slot,
        mode=mode,
        n_pairs=min(10, max(1, len(clauses))),
        soundscape_contrasts=sound_clauses or None,
        hold_contrasts=hold_clauses or None,
        target_prompt=tp_clean,
        target_weight=float(target_weight),
    )


def _repool_cached_build(
    state: ControlVectorUIState,
    req: dict[str, Any],
) -> dict[str, Any]:
    """Re-pool the most recent build in `< 1 ms` on pure CPU (or re-encode only dirty edited pairs)."""
    if not state.last_build_cache:
        raise ValueError("no active build cache in memory; build a 10-pair vector first")

    t_start = time.perf_counter()
    weights_req = req.get("weights")
    enabled_req = req.get("enabled")
    hold_indices_req = req.get("hold_indices") or []
    rows_req = req.get("rows")
    unit_mean = bool(req.get("unit_mean", False))
    hold_words = str(req.get("hold_words", "") or "").strip() or None
    hold_alpha = float(req.get("hold_alpha", 1.0))
    hold_vectors = req.get("hold_vectors") or []
    prompt_req = str(req.get("prompt", "") or "").strip() or None

    dirty_reencoded = 0
    built_rows: list[dict[str, Any]] = []

    for cache_key, entry in state.last_build_cache.items():
        family = entry["family"]
        pool_mode = entry["pooling"]
        a_mat: torch.Tensor = entry["a_mat"]
        b_mat: torch.Tensor = entry["b_mat"]
        enc_a_mat: torch.Tensor | None = entry["enc_a_mat"]
        enc_b_mat: torch.Tensor | None = entry["enc_b_mat"]
        n_total = int(a_mat.shape[0])

        # Check if any row's text was edited; if so, re-encode only that dirty row
        if isinstance(rows_req, list) and len(rows_req) == n_total:
            for idx, r_item in enumerate(rows_req):
                if not isinstance(r_item, dict):
                    continue
                new_a = str(r_item.get("a", "")).strip()
                new_b = str(r_item.get("b", "")).strip()
                old_a, old_b = entry["pairs_ab"][idx]
                if new_a and new_b and (new_a != old_a or new_b != old_b):
                    clip = state.get_clip()
                    dit_p = entry.get("dit_path") or entry.get("unet_path")
                    ref_cache_key = (
                        str(dit_p),
                        str(entry["encoder_path"]),
                        str(entry["refiner_device"]),
                        json.dumps(entry["loras"], sort_keys=True),
                    )
                    if ref_cache_key in state._refiner_cache:
                        _, _, refiner_run = state._refiner_cache[ref_cache_key]
                    else:
                        refiner_run = cv_nodes.build_refiner_runner(
                            dit_p, device=entry["refiner_device"], loras=entry["loras"]
                        )
                    enc_a, tags_a, ids_a, dec_a = cv_nodes.encode_prompt_for_vector(clip, new_a)
                    enc_b, tags_b, ids_b, dec_b = cv_nodes.encode_prompt_for_vector(clip, new_b)
                    ref_a = refiner_run(enc_a)
                    ref_b = refiner_run(enc_b)
                    first_diff, end_a, end_b = h3_spans.differing_token_bounds(ids_a, ids_b)
                    spans_a = h3_spans.spans(ids_a, tags_a.tolist(), dec_a)
                    spans_b = h3_spans.spans(ids_b, tags_b.tolist(), dec_b)
                    mask_a = h3_spans.pooling_mask(
                        spans_a,
                        pool_mode,
                        first_diff=min(first_diff, len(ids_a) - 1),
                        end_diff=end_a,
                        other_ids=ids_b,
                    )
                    mask_b = h3_spans.pooling_mask(
                        spans_b,
                        pool_mode,
                        first_diff=min(first_diff, len(ids_b) - 1),
                        end_diff=end_b,
                        other_ids=ids_a,
                    )
                    a_mat[idx] = (
                        ref_a.to(dtype=torch.float64)[mask_a].mean(dim=0).to(dtype=torch.float32).cpu()
                    )
                    b_mat[idx] = (
                        ref_b.to(dtype=torch.float64)[mask_b].mean(dim=0).to(dtype=torch.float32).cpu()
                    )
                    if enc_a_mat is not None and enc_b_mat is not None:
                        ma_e = mask_a.clone()
                        mb_e = mask_b.clone()
                        if ma_e.numel() > 0:
                            ma_e[0] = False
                        if mb_e.numel() > 0:
                            mb_e[0] = False
                        if not bool(ma_e.any()):
                            ma_e = mask_a
                        if not bool(mb_e.any()):
                            mb_e = mask_b
                        enc_a_mat[idx] = (
                            enc_a.to(dtype=torch.float64)[ma_e].mean(dim=0).to(dtype=torch.float32).cpu()
                        )
                        enc_b_mat[idx] = (
                            enc_b.to(dtype=torch.float64)[mb_e].mean(dim=0).to(dtype=torch.float32).cpu()
                        )
                    a_sp, b_sp = cv.differing_span(new_a, new_b, frame=entry["pair_set"].frame)
                    entry["pair_spans"][idx] = {
                        "index": idx,
                        "weight": float(r_item.get("weight", entry["weights"][idx])),
                        "first_diff": int(first_diff),
                        "end_a": int(end_a),
                        "end_b": int(end_b),
                        "a_span": a_sp,
                        "b_span": b_sp,
                        "len_a": int(len(ids_a)),
                        "len_b": int(len(ids_b)),
                        "pooled_rows_a": int(mask_a.sum().item()),
                        "pooled_rows_b": int(mask_b.sum().item()),
                        "pooled_tokens": int(mask_b.sum().item()),
                    }
                    entry["pairs_ab"][idx] = (new_a, new_b)
                    dirty_reencoded += 1

        hold_idx_set = {
            int(i) for i in (hold_indices_req or ()) if 0 <= int(i) < n_total
        }
        active_indices_req = req.get("active_indices")
        if isinstance(active_indices_req, list):
            active_set_req = {int(i) for i in active_indices_req if 0 <= int(i) < n_total}
            active_indices = [
                i for i in range(n_total) if i in active_set_req and i not in hold_idx_set
            ]
        else:
            active_indices = [
                i
                for i in range(n_total)
                if (not isinstance(enabled_req, list) or i >= len(enabled_req) or bool(enabled_req[i]))
                and i not in hold_idx_set
            ]
        if not active_indices:
            raise ValueError("at least one pair must remain enabled and not moved to hold")

        eff_weights = [
            float(weights_req[i])
            if (isinstance(weights_req, list) and i < len(weights_req))
            else float(entry["weights"][i])
            for i in range(n_total)
        ]
        entry["weights"] = eff_weights
        for i in range(n_total):
            entry["pair_spans"][i]["weight"] = eff_weights[i]

        a_sub = a_mat[active_indices]
        b_sub = b_mat[active_indices]
        w_sub = torch.tensor([eff_weights[i] for i in active_indices], dtype=torch.float32)

        raw_vec = cv.mean_difference(a_sub, b_sub, w_sub, unit_mean=unit_mean)
        tensors_to_save: dict[str, torch.Tensor] = {}

        hold_a = list(entry["hold_a_list"]) + [a_mat[hi] for hi in sorted(hold_idx_set)]
        hold_b = list(entry["hold_b_list"]) + [b_mat[hi] for hi in sorted(hold_idx_set)]
        h_groups = list(entry["hold_groups"]) + [f"pair_{hi}" for hi in sorted(hold_idx_set)]
        h_weights = list(entry["hold_weights"]) + [1.0 for _ in sorted(hold_idx_set)]

        if hold_a:
            basis = cv.hold_basis(
                torch.stack(hold_a, dim=0),
                torch.stack(hold_b, dim=0),
                h_groups,
                torch.tensor(h_weights, dtype=torch.float32),
            )
            final_vec = cv.project_out(raw_vec, basis)
            tensors_to_save["holds.post_refiner"] = basis
        else:
            final_vec = raw_vec

        applied_holds: list[str] = []
        norm_bef = 0.0
        norm_aft = 0.0
        if hold_vectors or hold_words:
            final_vec, applied_holds, norm_bef, norm_aft = _apply_post_refiner_holds(
                final_vec,
                family=family,
                hold_specs=hold_vectors,
                hold_words=hold_words,
                hold_alpha=hold_alpha,
                prompt=prompt_req,
            )

        tensors_to_save["vector.post_refiner"] = final_vec
        gen = torch.Generator(device="cpu").manual_seed(int(entry["seed"]))
        if len(active_indices) >= 2:
            q_report = cv.quality_report(a_sub, b_sub, w_sub, generator=gen)
        else:
            q_report = {
                "mean_pairwise_cosine": 1.0,
                "split_half_cosine": 1.0,
                "per_pair_cosine": [1.0],
            }

        # Compute per-pair cosine for all n_total rows against final_vec so the UI table shows every row
        all_per_pair_cos: list[float] = []
        for i in range(n_total):
            d_i = b_mat[i] - a_mat[i]
            all_per_pair_cos.append(round(_cosine(d_i, final_vec), 4))

        raw_norm = float(torch.linalg.vector_norm(final_vec.to(dtype=torch.float64)).item())
        points_meta: dict[str, Any] = {
            "post_refiner": {
                "unit": "mean_pair_difference",
                "raw_norm": raw_norm,
                "median_row_norm": entry["median_ref_row_norm"],
            }
        }
        if enc_a_mat is not None and enc_b_mat is not None:
            enc_raw_vec = cv.mean_difference(
                enc_a_mat[active_indices],
                enc_b_mat[active_indices],
                w_sub,
                unit_mean=unit_mean,
            )
            enc_hold_a = list(entry.get("enc_hold_a_list") or []) + [
                enc_a_mat[hi] for hi in sorted(hold_idx_set)
            ]
            enc_hold_b = list(entry.get("enc_hold_b_list") or []) + [
                enc_b_mat[hi] for hi in sorted(hold_idx_set)
            ]
            if enc_hold_a:
                enc_basis = cv.hold_basis(
                    torch.stack(enc_hold_a, dim=0),
                    torch.stack(enc_hold_b, dim=0),
                    h_groups,
                    torch.tensor(h_weights, dtype=torch.float32),
                )
                enc_raw_vec = cv.project_out(enc_raw_vec, enc_basis)
            enc_raw_norm = float(torch.linalg.vector_norm(enc_raw_vec.to(dtype=torch.float64)).item())
            if enc_raw_norm > 1e-12:
                enc_vec = (enc_raw_vec.to(dtype=torch.float64) / enc_raw_norm).to(dtype=torch.float32)
                tensors_to_save["vector.encoder"] = enc_vec
                points_meta["encoder"] = {
                    "unit": "median_row_norm",
                    "raw_norm": enc_raw_norm,
                    "median_row_norm": entry["median_enc_row_norm_ex0"],
                    "row0_excluded": True,
                }

        jl_readout: dict[str, Any] | None = None
        if j_lens.bundle_available():
            try:
                jl_readout = j_lens.decode_vector(final_vec, space="post_refiner", family=family)
                if prompt_req:
                    jl_readout["prompt_coupling"] = j_lens.analyze_prompt_coupling(
                        final_vec,
                        prompt_req,
                        readout=jl_readout,
                        family=family,
                    )
                q_report["j_lens_readout"] = jl_readout
            except Exception:
                jl_readout = None

        pair_set = entry["pair_set"]
        pairs_dict_repool: dict[str, Any] = {
            "path": entry["rel_pair_path"],
            "sha256": pair_set.sha256,
            "n_pairs": len(active_indices),
            "n_holds": len(hold_a) + len(applied_holds),
            "a_side": pair_set.a_side,
            "b_side": pair_set.b_side,
            "pair_spans": [entry["pair_spans"][i] for i in active_indices],
        }
        if str(entry["rel_pair_path"]).startswith("<"):
            pairs_dict_repool["pairs_ab"] = [
                {
                    "a": entry["pairs_ab"][i][0],
                    "b": entry["pairs_ab"][i][1],
                    "weight": float(eff_weights[i]),
                }
                for i in active_indices
            ]
        metadata: dict[str, Any] = {
            "format": cv.FORMAT_VERSION,
            "name": entry["stem"],
            "quality": pair_set.quality,
            "family": family,
            "pairs": pairs_dict_repool,
            "recipe": entry["recipe"],
            "measured_under": {
                "mode": entry["measured_mode"],
                "frame": pair_set.frame,
                "prefix": entry["prefix_meta"],
                "pooling": pool_mode,
            },
            "points": points_meta,
            "quality_report": q_report,
            "provenance": {
                "core_revision": substrate.git_head(cv_nodes.COMFY_ROOT) or "unknown",
                "pack_commit": substrate.git_head(REPO_ROOT) or "unknown",
                "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "made_by": "bench/control_vectors.py::repool/1",
                **(
                    {
                        "applied_holds": {
                            "holds": applied_holds,
                            "alpha": hold_alpha,
                            "norm_before": round(norm_bef, 4),
                            "norm_after_raw": round(norm_aft, 4),
                        }
                    }
                    if applied_holds
                    else {}
                ),
            },
        }
        saved = cv.save_vector(entry["out_path"], tensors_to_save, metadata, overwrite=True)
        rec_band = [
            round(50.0 / max(raw_norm, 1e-6), 2),
            round(85.0 / max(raw_norm, 1e-6), 2),
        ]
        built_rows.append(
            {
                "file": saved.name,
                "family": family,
                "pooling": pool_mode,
                "raw_norm": raw_norm,
                "vector_norm": raw_norm,
                "mpc": float(q_report["mean_pairwise_cosine"]),
                "split_half": float(q_report["split_half_cosine"]),
                "split_half_cosine": float(q_report["split_half_cosine"]),
                "per_pair_cosine": all_per_pair_cos,
                "active_indices": active_indices,
                "hold_indices": sorted(hold_idx_set),
                "pair_spans": entry["pair_spans"],
                "rec_strength": _recommended_strength_text(raw_norm),
                "recommended_strength_band": rec_band,
                "dirty_reencoded": dirty_reencoded,
                "j_lens": jl_readout,
            }
        )

    elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    return {
        "ok": True,
        "elapsed_ms": elapsed_ms,
        "dirty_reencoded": dirty_reencoded,
        "built": built_rows,
        "vectors": _list_ui_vectors(state.vec_dir),
    }



def _discover_ui_clips() -> list[dict[str, Any]]:
    """Return available comparison clips under $H3_OUTPUT_DIR/Video/control_vectors/ (or empty list)."""
    out: list[dict[str, Any]] = []
    out_root = os.environ.get("H3_OUTPUT_DIR", "")
    if not out_root:
        return out
    cv_vid_dir = Path(out_root) / "Video" / "control_vectors"
    if not cv_vid_dir.is_dir():
        return out
    for mp4 in sorted(cv_vid_dir.rglob("*.mp4"))[:60]:
        try:
            rel = mp4.relative_to(Path(out_root)).as_posix()
            out.append({"name": mp4.name, "path": rel, "url": f"/media?path={rel}"})
        except Exception:
            continue
    return out

