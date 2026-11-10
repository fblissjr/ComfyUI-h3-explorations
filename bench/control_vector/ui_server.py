"""HTTP server and `/api/*` request router for the H3 Control Vector UI."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import time
from typing import Any

import control_vector_nodes as cv_nodes
import control_vectors as cv
import h3_pair_compiler as pair_compiler
import j_lens

from .build_cmd import cmd_build, cmd_rebuild
from .cli import build_parser
from .helpers import (
    WEB_DIR,
    _compile_words_or_spec_to_pairset,
    _display_path,
    _is_markdown_pairs_text,
    _pair_set_to_markdown,
    _recommended_strength_text,
    _resolve_requested_families,
)
from .llm_pairs import (
    DEFAULT_LLM_MAX_TOKENS,
    SUPPORTED_LLM_PROVIDERS,
    _extract_system_prompt_text,
    _generate_pairs_with_llm,
    _repair_multichange_markdown_pairs,
)
from .make_cmds import cmd_make_media
from .sculpt import _compose_and_save_sculpted_vector, _meter_prompt_tokens
from .ui_state import (
    ControlVectorUIState,
    _build_pairset_from_ui_rows,
    _discover_ui_clips,
    _list_ui_toml_files,
    _list_ui_vectors,
    _preview_markdown_pairs,
    _repool_cached_build,
)

def handle_ui_request(
    state: ControlVectorUIState,
    method: str,
    path: str,
    body: bytes = b"",
) -> tuple[int, str, str]:
    clean_path = path.split("?", 1)[0]
    if method == "GET" and clean_path in ("/", "/index.html"):
        html_path = WEB_DIR / "index.html"
        return 200, "text/html; charset=utf-8", html_path.read_text(encoding="utf-8")
    if method == "GET" and clean_path == "/ui/style.css":
        css_path = WEB_DIR / "style.css"
        return 200, "text/css; charset=utf-8", css_path.read_text(encoding="utf-8")
    if method == "GET" and clean_path == "/ui/app.js":
        js_path = WEB_DIR / "app.js"
        return 200, "application/javascript; charset=utf-8", js_path.read_text(encoding="utf-8")

    if method == "GET" and clean_path == "/api/init":
        payload = {
            "ok": True,
            "toml_dir": _display_path(state.toml_dir),
            "vec_dir": _display_path(state.vec_dir),
            "system_prompt": _extract_system_prompt_text(),
            "toml_files": _list_ui_toml_files(state.toml_dir),
            "vectors": _list_ui_vectors(state.vec_dir),
            "clips": _discover_ui_clips(),
            "j_lens_available": j_lens.bundle_available(),
            "llm_providers": list(SUPPORTED_LLM_PROVIDERS),
            "default_llm_provider": state.default_llm_provider,
            "default_llm_model": state.default_llm_model,
        }
        return 200, "application/json", json.dumps(payload)

    req: dict[str, Any] = {}
    if body:
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": f"invalid JSON: {exc}"})

    if method == "POST" and clean_path == "/api/generate-pairs-llm":
        try:
            prov_req = str(req.get("provider") or state.default_llm_provider or "transformers").strip()
            model_req = str(req.get("model") or state.default_llm_model or "").strip()
            concept_req = str(req.get("concept") or req.get("axis") or "").strip()
            load_4bit = bool(req.get("load_in_4bit", True))
            max_tok = int(req.get("max_new_tokens", DEFAULT_LLM_MAX_TOKENS))
            temp_req = float(req.get("temperature", 0.7))
            state.release_clip_vram()
            res_llm = _generate_pairs_with_llm(
                provider=prov_req,
                model_spec=model_req,
                concept_prompt=concept_req,
                load_in_4bit=load_4bit,
                max_new_tokens=max_tok,
                temperature=temp_req,
                llm_cache=state._llm_cache,
            )
            return 200, "application/json", json.dumps(res_llm)
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/expand-scenes":
        try:
            pole_a = str(req.get("pole_a", "") or "").strip()
            pole_b = str(req.get("pole_b", "") or "").strip()
            sound_a = str(req.get("sound_a", "") or "").strip()
            sound_b = str(req.get("sound_b", "") or "").strip()
            hold_a = str(req.get("hold_a", "") or "").strip()
            hold_b = str(req.get("hold_b", "") or "").strip()
            raw_mode = req.get("mode")
            name = str(req.get("name", "") or "").strip()
            target_prompt = str(req.get("target_prompt", "") or "").strip()
            target_weight = float(req.get("target_weight", 2.5))
            spec_text = str(req.get("text") or req.get("words") or "").strip()

            if not spec_text:
                if not pole_b:
                    return 400, "application/json", json.dumps({"ok": False, "error": "Pole B (target concept) is required"})
                lines = [f"{pole_a} ||| {pole_b}" if pole_a else pole_b]
                if sound_b:
                    lines.append(f"soundscape: {sound_a} ||| {sound_b}" if sound_a else f"soundscape: {sound_b}")
                if hold_b:
                    lines.append(f"hold: {hold_a} ||| {hold_b}" if hold_a else f"hold: {hold_b}")
                spec_text = "\n".join(lines)

            families_req = str(req.get("families", "") or "").strip().lower()
            raw_mode_str = str(raw_mode or "").strip().lower()
            eff_mode_req = raw_mode_str if raw_mode_str not in ("", "auto") else ("ref2va" if families_req == "ref2va" and not target_prompt else None)
            mode = pair_compiler.infer_compile_mode(
                eff_mode_req,
                target_prompt=target_prompt or None,
                words=spec_text,
            )

            ps = _compile_words_or_spec_to_pairset(
                spec_text,
                name=name or None,
                mode=mode,
                target_prompt=target_prompt,
                target_weight=target_weight,
            )
            rows_payload: list[dict[str, Any]] = []
            for idx, p in enumerate(ps.pairs):
                f_id = f"target_scene_{idx + 1:02d}" if len(ps.pairs) > 1 else "target_scene_prompt"
                f_domain = f"target_prompt (Pair {idx})"
                rows_payload.append(
                    {
                        "index": idx,
                        "frame_id": f_id,
                        "scene_id": f_id,
                        "scene_title": f"{f_id} ({f_domain})",
                        "domain": f_domain,
                        "weight": float(p.weight),
                        "enabled": True,
                        "a": p.a,
                        "b": p.b,
                        "a_span": p.a_span,
                        "b_span": p.b_span,
                    }
                )
            holds_payload = [
                {
                    "index": h.index,
                    "group": h.group,
                    "weight": float(h.weight),
                    "a": h.a,
                    "b": h.b,
                    "a_span": h.a_span,
                    "b_span": h.b_span,
                }
                for h in ps.holds
            ]
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "name": ps.name,
                        "mode": ps.mode,
                        "quality": ps.quality,
                        "a_side": ps.a_side,
                        "b_side": ps.b_side,
                        "pole_a": ps.a_side,
                        "pole_b": ps.b_side,
                        "test_scene_prompt": target_prompt or (ps.pairs[0].a if ps.pairs else ""),
                        "test_prompt": target_prompt or (ps.pairs[0].a if ps.pairs else ""),
                        "rows": rows_payload,
                        "pairs": rows_payload,
                        "holds": holds_payload,
                        "markdown": _pair_set_to_markdown(None, ps),
                    }
                ),
            )
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/preview":
        try:
            text_req = str(req.get("text", "") or "")
            rows_req = req.get("rows")
            frame_req = str(req.get("frame", "scene") or "scene")
            name_req = str(req.get("name", "") or "").strip()
            mode_req = str(req.get("mode", "t2va") or "t2va")
            if isinstance(rows_req, list) and rows_req and not text_req.strip():
                ps_rows = _build_pairset_from_ui_rows(
                    rows_req,
                    holds=req.get("holds"),
                    name=name_req,
                    mode=mode_req,
                    frame=frame_req,
                    target_prompt=str(req.get("target_prompt", "") or ""),
                    target_weight=float(req.get("target_weight", 2.5)),
                )
                text_req = _pair_set_to_markdown(None, ps_rows)
                if not name_req:
                    name_req = ps_rows.name
            res = _preview_markdown_pairs(
                state,
                text_req,
                frame=frame_req,
                name=name_req,
                mode=mode_req,
            )
            return 200, "application/json", json.dumps(res)
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/load-toml":
        fname = str(req.get("file", "")).strip()
        if not fname:
            return 400, "application/json", json.dumps({"ok": False, "error": "missing 'file'"})
        try:
            cand = state.toml_dir / (fname if fname.endswith(".toml") else f"{fname}.toml")
            tpath = cand if cand.is_file() else cv_nodes.resolve_pair_file_path(fname)
            ps = cv.load_pairs(tpath, bank_dir=cv_nodes.PROMPT_BANK_DIR)
            md_str = _pair_set_to_markdown(tpath, ps)
            rows_payload = [
                {
                    "index": idx,
                    "frame_id": f"pair_{idx + 1:02d}",
                    "scene_id": f"pair_{idx + 1:02d}",
                    "scene_title": f"pair_{idx + 1:02d} ({ps.mode})",
                    "domain": ps.mode,
                    "weight": float(p.weight),
                    "enabled": True,
                    "a": p.a,
                    "b": p.b,
                    "a_span": p.a_span,
                    "b_span": p.b_span,
                }
                for idx, p in enumerate(ps.pairs)
            ]
            holds_payload = [
                {
                    "index": h.index,
                    "group": h.group,
                    "weight": float(h.weight),
                    "a": h.a,
                    "b": h.b,
                    "a_span": h.a_span,
                    "b_span": h.b_span,
                }
                for h in ps.holds
            ]
            tp_cand = tpath.parent / f"{tpath.stem}_test_prompt.txt"
            tp_txt = tp_cand.read_text(encoding="utf-8").strip() if tp_cand.is_file() else (ps.pairs[0].a if ps.pairs else "")
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "file": tpath.name,
                        "name": ps.name,
                        "frame": ps.frame,
                        "mode": ps.mode,
                        "quality": ps.quality,
                        "a_side": ps.a_side,
                        "b_side": ps.b_side,
                        "n_pairs": len(ps.pairs),
                        "n_holds": len(ps.holds),
                        "test_scene_prompt": tp_txt,
                        "test_prompt": tp_txt,
                        "rows": rows_payload,
                        "pairs": rows_payload,
                        "holds": holds_payload,
                        "markdown": md_str,
                    }
                ),
            )
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/save-toml":
        text = str(req.get("text", ""))
        name = str(req.get("name", "")).strip()
        quality_req = str(req.get("quality") or req.get("axis") or "").strip() or None
        frame = str(req.get("frame", "scene") or "scene")
        mode = str(req.get("mode", "t2va") or "t2va")
        overwrite = bool(req.get("overwrite", True))
        rows_in = req.get("rows")
        holds_in = list(req.get("holds") or [])
        if not holds_in and str(req.get("hold_b", "") or "").strip():
            holds_in = [
                {
                    "a": str(req.get("hold_a", "") or "").strip(),
                    "b": str(req.get("hold_b", "") or "").strip(),
                }
            ]
        try:
            if isinstance(rows_in, list) and rows_in:
                ps = _build_pairset_from_ui_rows(
                    rows_in,
                    holds=holds_in or None,
                    name=name,
                    quality=quality_req,
                    mode=mode,
                    frame=frame,
                    target_prompt=str(req.get("target_prompt", "") or ""),
                    target_weight=float(req.get("target_weight", 2.5)),
                )
                if not name:
                    name = ps.name
                md_doc = _pair_set_to_markdown(None, ps)
                tp_out = None
                saved_toml = cv.write_markdown_to_toml(
                    md_doc,
                    state.toml_dir,
                    name=name,
                    mode=ps.mode,
                    frame=frame,
                    quality=quality_req or ps.quality,
                    a_side=ps.a_side,
                    b_side=ps.b_side,
                    overwrite=overwrite,
                    bank_dir=cv_nodes.PROMPT_BANK_DIR,
                )
            elif _is_markdown_pairs_text(text):
                if not name:
                    return 400, "application/json", json.dumps({"ok": False, "error": "vector name is required"})
                text = _repair_multichange_markdown_pairs(text)
                parsed = cv.parse_markdown_pairs(text)
                tp_out = state.toml_dir / f"{name}_test_prompt.txt" if parsed.get("test_scene_prompt") else None
                saved_toml = cv.write_markdown_to_toml(
                    text,
                    state.toml_dir,
                    name=name,
                    frame=frame,
                    quality=quality_req,
                    test_prompt_out=tp_out,
                    overwrite=overwrite,
                    bank_dir=cv_nodes.PROMPT_BANK_DIR,
                )
            else:
                spec_with_holds = text
                if holds_in and "hold:" not in spec_with_holds.lower():
                    hold_lines = [
                        f"hold: {hr.get('a', '').strip()} ||| {hr['b'].strip()}"
                        if str(hr.get("a", "")).strip()
                        else f"hold: {hr['b'].strip()}"
                        for hr in holds_in
                        if str(hr.get("b", "")).strip()
                    ]
                    if hold_lines:
                        spec_with_holds = spec_with_holds.rstrip() + "\n" + "\n".join(hold_lines)
                ps = _compile_words_or_spec_to_pairset(
                    spec_with_holds,
                    name=name or None,
                    mode=mode,
                    target_prompt=str(req.get("target_prompt", "") or ""),
                    target_weight=float(req.get("target_weight", 2.5)),
                )
                if not name:
                    name = ps.name
                md_doc = _pair_set_to_markdown(None, ps)
                tp_out = None
                saved_toml = cv.write_markdown_to_toml(
                    md_doc,
                    state.toml_dir,
                    name=name,
                    mode=ps.mode,
                    frame=frame,
                    quality=quality_req or ps.quality,
                    a_side=ps.a_side,
                    b_side=ps.b_side,
                    overwrite=overwrite,
                    bank_dir=cv_nodes.PROMPT_BANK_DIR,
                )
            n_saved_pairs = len(ps.pairs) if "ps" in locals() else len(parsed.get("pairs", []))
            disp_toml = _display_path(saved_toml)
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "name": name,
                        "toml": disp_toml,
                        "toml_path": disp_toml,
                        "saved_toml": disp_toml,
                        "pairs": n_saved_pairs,
                        "toml_files": _list_ui_toml_files(state.toml_dir),
                        "test_prompt_file": _display_path(tp_out) if tp_out and tp_out.is_file() else None,
                    }
                ),
            )
        except (ValueError, FileExistsError, SystemExit) as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/build":
        text = str(req.get("text", "") or req.get("words", "") or "")
        name = str(req.get("name", "") or req.get("output_name", "") or "").strip()
        quality_req = str(req.get("quality") or req.get("axis") or "").strip() or None
        frame = str(req.get("frame", "scene") or "scene")
        raw_mode = req.get("mode")
        families = str(req.get("families", "both") or "both")
        pooling = str(req.get("pooling", "diff_span") or "diff_span")
        no_lora = bool(req.get("no_lora", False))
        overwrite = bool(req.get("overwrite", True))
        include_encoder = bool(req.get("include_encoder", True))
        unit_mean = bool(req.get("unit_mean", False))
        save_toml_req = bool(req.get("save_toml", False))
        hold_words = str(req.get("hold_words", "") or "").strip()
        hold_alpha = float(req.get("hold_alpha", 1.0))
        hold_vectors = req.get("hold_vectors") or []
        target_prompt = str(req.get("target_prompt", "") or "").strip()
        target_weight = float(req.get("target_weight", 2.5))
        raw_mode_str = str(raw_mode or "").strip().lower()
        eff_mode_req = raw_mode_str if raw_mode_str not in ("", "auto") else ("ref2va" if families.strip().lower() == "ref2va" and not target_prompt else None)
        mode = pair_compiler.infer_compile_mode(
            eff_mode_req,
            target_prompt=target_prompt or None,
            words=text or None,
        )
        rows_in = req.get("rows")
        holds_in = list(req.get("holds") or [])
        if not holds_in and str(req.get("hold_b", "") or "").strip():
            holds_in = [
                {
                    "a": str(req.get("hold_a", "") or "").strip(),
                    "b": str(req.get("hold_b", "") or "").strip(),
                }
            ]
        try:
            import contextlib
            import dataclasses
            import io

            saved_toml: Path | None = None
            compiled_ps: cv.PairSet | None = None
            if isinstance(rows_in, list) and rows_in:
                compiled_ps = _build_pairset_from_ui_rows(
                    rows_in,
                    holds=holds_in or None,
                    name=name,
                    quality=quality_req,
                    mode=mode,
                    frame=frame,
                    target_prompt=target_prompt,
                    target_weight=target_weight,
                )
                if not name:
                    name = compiled_ps.name
                if save_toml_req:
                    md_doc = _pair_set_to_markdown(None, compiled_ps)
                    saved_toml = cv.write_markdown_to_toml(
                        md_doc,
                        state.toml_dir,
                        name=name,
                        mode=compiled_ps.mode,
                        frame=frame,
                        quality=quality_req or compiled_ps.quality,
                        a_side=compiled_ps.a_side,
                        b_side=compiled_ps.b_side,
                        overwrite=overwrite,
                        bank_dir=cv_nodes.PROMPT_BANK_DIR,
                    )
            elif _is_markdown_pairs_text(text):
                if not name:
                    return 400, "application/json", json.dumps({"ok": False, "error": "vector name is required"})
                text = _repair_multichange_markdown_pairs(text)
                parsed = cv.parse_markdown_pairs(text)
                tp_out = state.toml_dir / f"{name}_test_prompt.txt" if parsed.get("test_scene_prompt") else None
                cand_toml = state.toml_dir / f"{name}.toml"
                reused_existing = False
                if cand_toml.is_file():
                    try:
                        existing_ps = cv.load_pairs(cand_toml, bank_dir=cv_nodes.PROMPT_BANK_DIR)
                        existing_ab = [(round(float(p.weight), 6), p.a, p.b) for p in existing_ps.pairs]
                        new_ab = [(round(float(p.get("weight", 1.0)), 6), p["a"], p["b"]) for p in parsed["pairs"]]
                        existing_holds = [(h.group, h.a, h.b) for h in existing_ps.holds]
                        new_holds = [(h["group"], h["a"], h["b"]) for h in parsed.get("holds", [])]
                        if (
                            existing_ab == new_ab
                            and existing_holds == new_holds
                            and existing_ps.frame == frame
                            and (not quality_req or existing_ps.quality == quality_req)
                        ):
                            saved_toml = cand_toml
                            reused_existing = True
                            if tp_out is not None and parsed.get("test_scene_prompt"):
                                tp_out.parent.mkdir(parents=True, exist_ok=True)
                                tp_out.write_text(parsed["test_scene_prompt"] + "\n", encoding="utf-8")
                    except Exception:
                        reused_existing = False
                if not reused_existing:
                    saved_toml = cv.write_markdown_to_toml(
                        text,
                        state.toml_dir,
                        name=name,
                        frame=frame,
                        quality=quality_req,
                        test_prompt_out=tp_out,
                        overwrite=True if overwrite else False,
                        bank_dir=cv_nodes.PROMPT_BANK_DIR,
                    )
            else:
                spec_with_holds = text
                if holds_in and "hold:" not in spec_with_holds.lower():
                    hold_lines = [
                        f"hold: {hr.get('a', '').strip()} ||| {hr['b'].strip()}"
                        if str(hr.get("a", "")).strip()
                        else f"hold: {hr['b'].strip()}"
                        for hr in holds_in
                        if str(hr.get("b", "")).strip()
                    ]
                    if hold_lines:
                        spec_with_holds = spec_with_holds.rstrip() + "\n" + "\n".join(hold_lines)
                compiled_ps = _compile_words_or_spec_to_pairset(
                    spec_with_holds,
                    name=name or None,
                    mode=mode,
                    target_prompt=target_prompt,
                    target_weight=target_weight,
                )
                if quality_req:
                    compiled_ps = dataclasses.replace(compiled_ps, quality=quality_req)
                if not name:
                    name = compiled_ps.name
                if save_toml_req:
                    md_doc = _pair_set_to_markdown(None, compiled_ps)
                    saved_toml = cv.write_markdown_to_toml(
                        md_doc,
                        state.toml_dir,
                        name=name,
                        mode=compiled_ps.mode,
                        frame=frame,
                        quality=quality_req or compiled_ps.quality,
                        a_side=compiled_ps.a_side,
                        b_side=compiled_ps.b_side,
                        overwrite=overwrite,
                        bank_dir=cv_nodes.PROMPT_BANK_DIR,
                    )

            argv = [
                "build",
                *( [str(saved_toml)] if saved_toml is not None else [] ),
                "--name",
                name,
                "--families",
                families,
                "--pooling",
                pooling,
                "--out-dir",
                str(state.vec_dir),
            ]
            if no_lora:
                argv.append("--no-lora")
            if unit_mean:
                argv.append("--unit-mean")
            if hold_words:
                argv.extend(["--hold-words", hold_words, "--hold-alpha", str(hold_alpha)])
            for hv in hold_vectors:
                if str(hv).strip():
                    hv_str = str(hv).strip()
                    hv_cand = state.vec_dir / hv_str
                    argv.extend(["--hold", str(hv_cand if hv_cand.is_file() else hv_str)])
            if target_prompt and saved_toml is not None and compiled_ps is None:
                argv.extend(["--target-prompt", target_prompt, "--target-weight", str(target_weight)])
            if include_encoder:
                argv.append("--include-encoder")
            else:
                argv.append("--no-encoder")
            if overwrite:
                argv.append("--overwrite")
            b_args = build_parser().parse_args(argv)
            if compiled_ps is not None and saved_toml is None:
                b_args._compiled_pair_set = compiled_ps
            rows_out: list[dict[str, Any]] = []
            build_cache: dict[str, Any] = {}
            b_args._summary_rows_out = rows_out
            b_args._build_cache_out = build_cache
            b_args._clip_override = state.get_clip(b_args.encoder, b_args.device)
            b_args._refiner_cache = state._refiner_cache
            buf_out = io.StringIO()
            with contextlib.redirect_stdout(buf_out):
                cmd_build(b_args)
            state.last_build_cache = build_cache
            disp_saved = (
                _display_path(saved_toml)
                if saved_toml is not None
                else str(compiled_ps.path if compiled_ps is not None else "")
            )
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "name": name,
                        "toml_path": disp_saved,
                        "saved_toml": disp_saved,
                        "built": rows_out,
                        "toml_files": _list_ui_toml_files(state.toml_dir),
                        "vectors": _list_ui_vectors(state.vec_dir),
                        "log": buf_out.getvalue(),
                    }
                ),
            )
        except (ValueError, FileExistsError, SystemExit) as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/repool":
        try:
            req_name = str(req.get("name", "") or "").strip()
            if not state.last_build_cache and req_name:
                cand_toml = state.toml_dir / (req_name if req_name.endswith(".toml") else f"{req_name}.toml")
                if cand_toml.is_file():
                    import contextlib
                    import io

                    b_args = build_parser().parse_args(
                        [
                            "build",
                            str(cand_toml),
                            "--out-dir",
                            str(state.vec_dir),
                            "--overwrite",
                        ]
                    )
                    rows_out: list[dict[str, Any]] = []
                    build_cache: dict[str, Any] = {}
                    b_args._summary_rows_out = rows_out
                    b_args._build_cache_out = build_cache
                    b_args._clip_override = state.get_clip(b_args.encoder, b_args.device)
                    b_args._refiner_cache = state._refiner_cache
                    with contextlib.redirect_stdout(io.StringIO()):
                        cmd_build(b_args)
                    state.last_build_cache = build_cache
            res = _repool_cached_build(state, req)
            return 200, "application/json", json.dumps(res)
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/sculpt":
        try:
            steps = req.get("steps") or req.get("specs")
            if not isinstance(steps, list) or not steps:
                return 400, "application/json", json.dumps({"ok": False, "error": "missing 'steps' recipe list"})
            name = str(req.get("output_name") or req.get("name") or "sculpted_vector").strip()
            target_fams = _resolve_requested_families(req.get("families", "both"))
            hold_words = str(req.get("hold_words", "") or "").strip() or None
            hold_alpha = float(req.get("hold_alpha", 1.0))
            prompt_req = str(req.get("prompt", "") or "").strip() or None
            overwrite_req = bool(req.get("overwrite", True))
            save_req = bool(req.get("save", True))

            if not save_req and j_lens.bundle_available():
                resolved_steps = []
                for st in steps:
                    v_src = state.vec_dir / str(st["vector"])
                    if not v_src.is_file():
                        v_src = cv_nodes.resolve_vector_file_path(str(st["vector"]))
                    resolved_steps.append(
                        {
                            "vector": v_src,
                            "mode": str(st.get("mode", "add")),
                            "strength": float(st.get("strength", 1.0)),
                        }
                    )
                rep = j_lens.decode_recipe(
                    resolved_steps,
                    family=target_fams[0],
                    hold_words=hold_words,
                    alpha=hold_alpha,
                    prompt=prompt_req,
                )
                return (
                    200,
                    "application/json",
                    json.dumps(
                        {
                            "ok": True,
                            "saved": [],
                            "built": [],
                            "readout": rep.get("readout") or rep,
                            "readiness": rep.get("readiness") or (rep.get("readout") or {}).get("readiness"),
                            "prompt_coupling": rep.get("prompt_coupling") or (rep.get("readout") or {}).get("prompt_coupling"),
                        }
                    ),
                )

            saved_rows: list[dict[str, Any]] = []
            for tfam in target_fams:
                r = _compose_and_save_sculpted_vector(
                    steps,
                    name=name,
                    family=tfam,
                    hold_words=hold_words,
                    hold_alpha=hold_alpha,
                    prompt=prompt_req,
                    out_dir=state.vec_dir,
                    overwrite=overwrite_req,
                )
                saved_rows.append(
                    {
                        "file": r["file"],
                        "family": r["family"],
                        "raw_norm": r["raw_norm"],
                        "net_norm": r["raw_norm"],
                        "vector_norm": r["raw_norm"],
                        "rec_strength": r["rec_strength"],
                        "j_lens": r["j_lens"],
                    }
                )
            first_jl = (saved_rows[0].get("j_lens") if saved_rows else None) or {}
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "built": saved_rows,
                        "saved": saved_rows,
                        "readout": first_jl,
                        "readiness": first_jl.get("readiness"),
                        "prompt_coupling": first_jl.get("prompt_coupling"),
                        "vectors": _list_ui_vectors(state.vec_dir),
                    }
                ),
            )
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/prompt-meter":
        try:
            vec_arg = str(req.get("vector", "") or "").strip()
            prompt_arg = str(req.get("prompt", "") or "").strip()
            pooling_arg = str(req.get("pooling", "scene_text") or "scene_text")
            family_arg = str(req.get("family", "") or "").strip() or None
            if not vec_arg or not prompt_arg:
                return 400, "application/json", json.dumps({"ok": False, "error": "both 'vector' and 'prompt' are required"})
            res = _meter_prompt_tokens(
                state,
                vector_arg=vec_arg,
                prompt_arg=prompt_arg,
                pooling=pooling_arg,
                family=family_arg,
            )
            return 200, "application/json", json.dumps(res)
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path in ("/api/build-media", "/api/make-media"):
        try:
            import contextlib
            import io

            img_a = str(req.get("image_a_data") or req.get("image_a") or "").strip()
            img_b = str(req.get("image_b_data") or req.get("image_b") or "").strip()
            out_name = str(req.get("output_name") or req.get("name") or "media_contrast").strip()
            families = str(req.get("families", "ref2va") or "ref2va")
            pooling = str(req.get("pooling", "scene_text") or "scene_text")
            short_edge = int(req.get("short_edge", 512))
            scene_prompt = str(req.get("scene_prompt") or req.get("prompt") or "").strip()
            hold_words = str(req.get("hold_words", "") or "").strip()
            hold_alpha = float(req.get("hold_alpha", 1.0))
            no_lora = bool(req.get("no_lora", False))
            overwrite = bool(req.get("overwrite", True))
            if not img_a or not img_b:
                return (
                    400,
                    "application/json",
                    json.dumps({"ok": False, "error": "both Image A (baseline) and Image B (steered) are required"}),
                )
            argv = [
                "make-media",
                "--image-a",
                img_a,
                "--image-b",
                img_b,
                "--output-name",
                out_name,
                "--families",
                families,
                "--pooling",
                pooling,
                "--short-edge",
                str(short_edge),
                "--out-dir",
                str(state.vec_dir),
            ]
            if overwrite:
                argv.append("--overwrite")
            if no_lora:
                argv.append("--no-lora")
            if scene_prompt:
                argv.extend(["--scene-prompt", scene_prompt])
            if hold_words:
                argv.extend(["--hold-words", hold_words, "--hold-alpha", str(hold_alpha)])
            m_args = build_parser().parse_args(argv)
            m_args._clip_override = state.get_clip(m_args.encoder, m_args.device)
            buf_out = io.StringIO()
            with contextlib.redirect_stdout(buf_out):
                cmd_make_media(m_args)
            vec_list_after = _list_ui_vectors(state.vec_dir)
            built_media = [
                {
                    "file": v["file"],
                    "family": v.get("family", "ref2va"),
                    "grid": f"{short_edge}px (32-px M-RoPE)",
                    "raw_norm": v.get("raw_norm", 0.0),
                    "vector_norm": v.get("raw_norm", 0.0),
                    "rec_strength": v.get("rec_strength", ""),
                    "j_lens": v.get("j_lens"),
                }
                for v in vec_list_after
                if v["file"].startswith(f"{out_name}__")
            ]
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "name": out_name,
                        "built": built_media,
                        "vectors": vec_list_after,
                        "log": buf_out.getvalue(),
                    }
                ),
            )
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path in ("/api/j-lens", "/api/build-words", "/api/preview-words"):
        try:
            if not j_lens.bundle_available():
                return (
                    400,
                    "application/json",
                    json.dumps({"ok": False, "error": "J-Lens bundle models/j_lens/j_lens_h3.safetensors not found"}),
                )
            from_words = str(req.get("from_words") or req.get("words") or req.get("pos_words") or "").strip()
            prompt_req = str(req.get("prompt", "") or "").strip() or None
            if from_words:
                families_req = str(req.get("families", "") or "").strip().lower()
                raw_fam = str(req.get("family", "") or "").strip().lower()
                if raw_fam in ("fl2va", "ref2va"):
                    fam = raw_fam
                elif families_req:
                    fam = _resolve_requested_families(families_req)[0]
                else:
                    fam = "fl2va"
                against_w = str(req.get("against_words", "") or "").strip() or None
                hold_w = str(req.get("hold_words", "") or "").strip() or None
                alpha_w = float(req.get("alpha", req.get("hold_alpha", 1.0)))
                synth = j_lens.synthesize_from_words(
                    from_words,
                    against_w,
                    family=fam if fam in ("fl2va", "ref2va") else "fl2va",
                    hold_words=hold_w,
                    alpha=alpha_w,
                    prompt=prompt_req,
                )
                save_name = str(req.get("save_name") or req.get("name") or "").strip()
                should_save = (
                    clean_path == "/api/build-words"
                    or (clean_path != "/api/preview-words" and (bool(req.get("save", False)) or bool(save_name)))
                )
                if not save_name and should_save:
                    save_name = "_".join(re.findall(r"[a-z0-9]+", from_words.lower())[:4]) or "word_vector"
                saved_file = None
                saved_files: list[str] = []
                built_words: list[dict[str, Any]] = []
                if should_save and save_name:
                    target_fams = (
                        _resolve_requested_families(families_req)
                        if families_req
                        else [synth["family"]]
                    )
                    for tfam in target_fams:
                        s_fam = (
                            synth
                            if tfam == synth["family"]
                            else j_lens.synthesize_from_words(
                                from_words,
                                against_w,
                                family=tfam,
                                hold_words=hold_w,
                                alpha=alpha_w,
                                prompt=prompt_req,
                            )
                        )
                        loras_meta = j_lens._DEFAULT_RECIPES.get(tfam, {}).get("loras")
                        fname = cv.default_vector_filename(save_name, tfam, pooling="diff_span", loras=loras_meta)
                        saved_path = j_lens.save_synthesized_vector(state.vec_dir / fname, s_fam, name=save_name)
                        saved_files.append(saved_path.name)
                        r_fam = s_fam["readout"]
                        r_norm = float(r_fam.get("norm_after_holds", r_fam.get("norm_before_holds", 65.0)))
                        built_words.append(
                            {
                                "file": saved_path.name,
                                "family": tfam,
                                "raw_norm": r_norm,
                                "rec_strength": _recommended_strength_text(r_norm),
                                "readout": r_fam,
                                "readiness": s_fam.get("readiness") or r_fam.get("readiness"),
                                "prompt_coupling": s_fam.get("prompt_coupling") or r_fam.get("prompt_coupling"),
                            }
                        )
                    if saved_files:
                        saved_file = saved_files[0]
                rep_words = {
                    "family": synth["family"],
                    "pos_words_matched": synth["pos_words_matched"],
                    "pos_words_unmatched": synth.get("pos_words_unmatched", []),
                    "pos_matched": synth["pos_words_matched"],
                    "neg_words_matched": synth["neg_words_matched"],
                    "hold_words": synth["hold_words"],
                    "hold_words_matched": synth["hold_words"],
                    "hold_matched": synth["hold_words"],
                    "alpha": synth["alpha"],
                    "name": save_name or None,
                    "save_name": save_name or None,
                    "saved_file": saved_file,
                    "saved_files": saved_files,
                    "built": built_words,
                    "readout": synth["readout"],
                    "readiness": synth.get("readiness") or synth["readout"].get("readiness"),
                    "prompt_coupling": synth.get("prompt_coupling") or synth["readout"].get("prompt_coupling"),
                }
                return (
                    200,
                    "application/json",
                    json.dumps(
                        {
                            "ok": True,
                            **rep_words,
                            "report": rep_words,
                            "vectors": _list_ui_vectors(state.vec_dir) if saved_files else None,
                        }
                    ),
                )
            steps = req.get("steps")
            if isinstance(steps, list) and steps:
                resolved_steps = []
                for st in steps:
                    v_src = state.vec_dir / str(st["vector"])
                    if not v_src.is_file():
                        v_src = cv_nodes.resolve_vector_file_path(str(st["vector"]))
                    resolved_steps.append(
                        {
                            "vector": v_src,
                            "mode": str(st.get("mode", "add")),
                            "strength": float(st.get("strength", 1.0)),
                        }
                    )
                rep = j_lens.decode_recipe(
                    resolved_steps,
                    family=str(req.get("family", "fl2va") or "fl2va"),
                    hold_words=str(req.get("hold_words", "") or "").strip() or None,
                    alpha=float(req.get("alpha", req.get("hold_alpha", 1.0))),
                    prompt=prompt_req,
                )
                rep_out = {
                    **rep,
                    "net_norm": rep.get("norm_after_holds", rep.get("norm_before_holds", 0.0)),
                }
                return 200, "application/json", json.dumps({"ok": True, **rep_out, "report": rep_out})

            vec_name = str(req.get("vector", "") or "").strip()
            if not vec_name:
                return 400, "application/json", json.dumps({"ok": False, "error": "missing 'vector', 'steps', or 'from_words'"})
            v_path = state.vec_dir / vec_name
            if not v_path.is_file():
                v_path = cv_nodes.resolve_vector_file_path(vec_name)
            holds_raw = req.get("holds") or []
            hold_paths = []
            for h_name in holds_raw:
                hp = state.vec_dir / str(h_name)
                if not hp.is_file():
                    hp = cv_nodes.resolve_vector_file_path(str(h_name))
                hold_paths.append(hp)
            rep = j_lens.decode_safetensors(
                v_path,
                holds=hold_paths if hold_paths else None,
                hold_words=str(req.get("hold_words", "") or "").strip() or None,
                alpha=float(req.get("alpha", req.get("hold_alpha", 1.0))),
                prompt=prompt_req,
            )
            pr_ro = rep.get("post_refiner") or rep.get("readout") or {}
            rep_out = {
                **rep,
                "readout": pr_ro,
                "positive_words": pr_ro.get("positive_words", []),
                "negative_words": pr_ro.get("negative_words", []),
                "norm": float(pr_ro.get("norm_after_holds", pr_ro.get("norm_before_holds", 0.0))),
            }
            return 200, "application/json", json.dumps({"ok": True, **rep_out, "report": rep_out})
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/rebuild":
        try:
            import contextlib
            import io

            target_req = str(req.get("target") or req.get("stem") or req.get("name") or "").strip()
            families_req = str(req.get("families", "both") or "both").strip()
            pooling_req = str(req.get("pooling", "diff_span") or "diff_span").strip()
            no_lora = bool(req.get("no_lora", False))
            unit_mean = bool(req.get("unit_mean", False))
            include_encoder = bool(req.get("include_encoder", True))
            argv = [
                "rebuild",
                *( [target_req] if target_req else [] ),
                "--dir",
                str(state.vec_dir),
                "--toml-dir",
                str(state.toml_dir),
                "--families",
                families_req,
                "--pooling",
                pooling_req,
            ]
            if no_lora:
                argv.append("--no-lora")
            if unit_mean:
                argv.append("--unit-mean")
            if include_encoder:
                argv.append("--include-encoder")
            else:
                argv.append("--no-encoder")
            r_args = build_parser().parse_args(argv)
            rows_out_rebuild: list[dict[str, Any]] = []
            r_args._summary_rows_out = rows_out_rebuild
            r_args._clip_override = state.get_clip(r_args.encoder, r_args.device)
            r_args._refiner_cache = state._refiner_cache
            buf_out = io.StringIO()
            with contextlib.redirect_stdout(buf_out):
                cmd_rebuild(r_args)
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "rebuilt": rows_out_rebuild,
                        "built": rows_out_rebuild,
                        "vectors": _list_ui_vectors(state.vec_dir),
                        "toml_files": _list_ui_toml_files(state.toml_dir),
                        "log": buf_out.getvalue(),
                    }
                ),
            )
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/quit":
        state.shutdown_requested = True
        return 200, "application/json", json.dumps({"ok": True})

    return 404, "application/json", json.dumps({"ok": False, "error": f"not found: {clean_path}"})


def cmd_ui(args: argparse.Namespace) -> int:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    import webbrowser

    toml_dir_override = Path(args.toml_dir) if getattr(args, "toml_dir", None) else None
    vec_dir_override = Path(args.vec_dir) if getattr(args, "vec_dir", None) else None
    state = ControlVectorUIState(
        toml_dir=toml_dir_override,
        vec_dir=vec_dir_override,
        default_encoder=getattr(args, "encoder", None),
        default_dit=getattr(args, "dit", None),
        default_device=getattr(args, "device", None),
        default_llm_provider=getattr(args, "llm_provider", None),
        default_llm_model=getattr(args, "llm_model", None),
    )
    srv_ref: list[ThreadingHTTPServer] = []

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *log_args: Any) -> None:
            return

        def _dispatch(self, method: str) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length > 0 else b""
            code, ctype, payload = handle_ui_request(state, method, self.path, body)
            raw_bytes = payload.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw_bytes)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw_bytes)
            if state.shutdown_requested and srv_ref:
                threading.Thread(target=srv_ref[0].shutdown, daemon=True).start()

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

    server = ThreadingHTTPServer((str(args.host), int(args.port)), _Handler)
    srv_ref.append(server)
    host, actual_port = server.server_address[:2]
    url = f"http://{host}:{actual_port}/"
    print(f"=== MiniMax H3 Control Vector Studio ===")
    print(f"  url   : {url}")
    print(f"  note  : Click 'Quit Server' in the browser tab or press Ctrl+C here to exit.")
    if not getattr(args, "no_browser", False):
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0

