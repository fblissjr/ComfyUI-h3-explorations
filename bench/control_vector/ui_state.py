"""UI state management, TOML/Markdown preview, and `< 1 ms` CPU cached-build repooling."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
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
            os.environ.get("H3_CV_TOML_DIR", "").strip() or cv_nodes.USER_PAIR_BANK_DIR
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
        self.shutdown_requested: bool = False
        #: The launch link's secret, kept by the browser as a cookie and required on every request (`ui_server.gate`).
        self.session_token: str = secrets.token_urlsafe(24)

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
    seen_stems: set[str] = set()
    search_dirs = [toml_dir]
    if toml_dir.resolve() != cv_nodes.PAIR_BANK_DIR.resolve():
        search_dirs.append(cv_nodes.PAIR_BANK_DIR)
    for d in search_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.toml")):
            if p.stem in seen_stems:
                continue
            try:
                ps = cv.load_pairs(p, bank_dir=cv_nodes.PROMPT_BANK_DIR)
                seen_stems.add(p.stem)
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
    out.sort(key=lambda r: str(r.get("stem", "")))
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
            mpc_raw = qr.get("mean_pairwise_cosine")
            shc_raw = qr.get("split_half_cosine")
            mpc = float(mpc_raw) if mpc_raw is not None else None
            shc = float(shc_raw) if shc_raw is not None else None
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
                    "differing_sections": list(pairs_meta.get("differing_sections") or ()),
                    "mode": str(meta.get("measured_under", {}).get("mode", "")),
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
    mode: str = "t2va",
    **_unused_kwargs: Any,
) -> dict[str, Any]:
    if frame not in cv.VALID_FRAMES:
        return {"ok": False, "error": f"frame must be one of {cv.VALID_FRAMES}, got {frame!r}"}
    try:
        if _is_markdown_pairs_text(text):
            text = _repair_multichange_markdown_pairs(text)
            parsed = cv.parse_markdown_pairs(text)
        else:
            spec_hdr = f"mode: {mode}\n" if mode != "t2va" else ""
            spec = pair_compiler.parse_short_contrast_spec(spec_hdr + text)
            ps = spec["pair_set"]
            if not name:
                name = ps.name
            parsed = {
                "mode": ps.mode,
                "test_scene_prompt": "",
                "pairs": [{"a": p.a, "b": p.b, "weight": float(p.weight)} for p in ps.pairs],
                "holds": [],
            }
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    tok = state.get_tokenizer()
    hf_tok = h3_spans.unwrap_hf_tokenizer(tok)
    if hf_tok is None:
        return {"ok": False, "error": "tokenizer does not expose the MiniMax Qwen3-VL tokenizer"}
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
                    spans_a,
                    "diff_span",
                    first_diff=min(first_diff, len(ids_a) - 1),
                    end_diff=end_a,
                    other_ids=ids_b,
                ).sum().item()
            )
            m_ds_b = int(
                h3_spans.pooling_mask(
                    spans_b,
                    "diff_span",
                    first_diff=min(first_diff, len(ids_b) - 1),
                    end_diff=end_b,
                    other_ids=ids_a,
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
        "test_scene_prompt": parsed["test_scene_prompt"],
        "test_prompt": parsed["test_scene_prompt"],
        "quality": quality,
        "avg_tokens": avg_tokens,
        "lint_warnings": lint_warnings,
        "pairs": pair_rows,
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

