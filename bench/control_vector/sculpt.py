"""CPU vector sculpting (`sculpt` / `--invert`) and contextual prompt receptor metering."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

import control_vector_nodes as cv_nodes
import control_vectors as cv
import h3_spans
import j_lens
import substrate
from workflows import h3_config

from . import helpers as _h
from .helpers import (
    REPO_ROOT,
    _apply_post_refiner_holds,
    _recommended_strength_text,
    _resolve_prompt_text,
    _resolve_requested_families,
    _resolve_vector_for_family,
)

def _compose_and_save_sculpted_vector(
    steps: list[dict[str, Any]],
    *,
    name: str,
    family: str = "fl2va",
    hold_words: str | None = None,
    hold_alpha: float = 1.0,
    prompt: str | None = None,
    out_dir: Path | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Compose a multi-vector recipe (`add`, `add_norm_matched`, `hold`) + optional `hold_words` (`alpha`)
    and write `.safetensors` + `.md` sidecars for `family`.
    """
    raw_stem = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(name or "").strip().lower()).strip("_")
    if not raw_stem:
        raw_stem = "sculpted_vector"
    if not cv.NAME_RE.match(raw_stem):
        raise ValueError(f"sculpted vector stem must match [a-z0-9_]+, got {raw_stem!r}")

    target_dir = out_dir or Path(_h.folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
    target_dir.mkdir(parents=True, exist_ok=True)

    resolved_steps: list[dict[str, Any]] = []
    enc_add_terms: list[torch.Tensor] = []
    enc_hold_terms: list[tuple[torch.Tensor, float]] = []
    base_meta: dict[str, Any] | None = None
    step_labels: list[str] = []

    for item in steps:
        raw_src = item["vector"]
        mode = str(item.get("mode", "add")).strip()
        strength = float(item.get("strength", 1.0))
        vpath = _resolve_vector_for_family(raw_src, family=family, extra_dir=target_dir)
        tensors, meta = cv.load_vector(vpath)
        if base_meta is None and mode in ("add", "add_norm_matched"):
            base_meta = meta
        resolved_steps.append({"vector": vpath, "mode": mode, "strength": strength})
        step_labels.append(f"{mode}:{meta.get('name', vpath.stem)}@{strength:g}")
        if "vector.encoder" in tensors:
            ve = tensors["vector.encoder"].detach().to(device="cpu", dtype=torch.float32).flatten()
            if mode in ("add", "add_norm_matched"):
                enc_add_terms.append(strength * ve)
            elif mode == "hold":
                enc_hold_terms.append((ve, strength))

    recipe_for_fam = (
        base_meta["recipe"]
        if base_meta is not None and base_meta.get("family") == family and isinstance(base_meta.get("recipe"), dict)
        else j_lens._DEFAULT_RECIPES.get(family, j_lens._DEFAULT_RECIPES["fl2va"])
    )
    loras_meta = recipe_for_fam.get("loras")
    if hasattr(j_lens, "save_composed_recipe_vector") and j_lens.bundle_available():
        fname = cv.default_vector_filename(raw_stem, family, pooling="diff_span", loras=loras_meta)
        out_path = target_dir / fname
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing control vector: {out_path}")
        saved_res = j_lens.save_composed_recipe_vector(
            out_path,
            resolved_steps,
            name=raw_stem,
            family=family,
            hold_words=hold_words,
            alpha=hold_alpha,
            prompt=prompt,
            recipe=recipe_for_fam,
        )
        saved_path = saved_res["path"] if isinstance(saved_res, dict) and "path" in saved_res else Path(saved_res)
        _, saved_meta = cv.load_vector(saved_path)
        r_norm = float(saved_meta["points"]["post_refiner"]["raw_norm"])
        jl_rep = saved_meta.get("quality_report", {}).get("j_lens_readout")
        return {
            "file": saved_path.name,
            "path": saved_path,
            "family": family,
            "raw_norm": r_norm,
            "net_norm": r_norm,
            "rec_strength": _recommended_strength_text(r_norm),
            "j_lens": jl_rep,
        }

    # Compose directly via j_lens.apply_soft_holds + decode_vector
    add_vecs: list[torch.Tensor] = []
    hold_vecs: list[tuple[torch.Tensor, float]] = []
    for st in resolved_steps:
        v_pr, _, _ = j_lens._load_vector_from_source(st["vector"], space="post_refiner")
        m_st = st["mode"]
        s_st = float(st["strength"])
        v64 = v_pr.to(dtype=torch.float64)
        vn = float(torch.linalg.vector_norm(v64).item())
        if m_st == "add":
            add_vecs.append((s_st * v64).to(dtype=torch.float32))
        elif m_st == "add_norm_matched":
            scale_f = (j_lens.NORM_MATCHED_TARGET / vn) if vn > 1e-12 else 0.0
            add_vecs.append((s_st * scale_f * v64).to(dtype=torch.float32))
        elif m_st == "hold":
            hold_vecs.append((v_pr, s_st))
        else:
            raise ValueError(f"unsupported sculpt mode {m_st!r}")
    if not add_vecs:
        raise ValueError("sculpt requires at least one 'add' or 'add_norm_matched' vector")

    cur_vec = torch.stack([v.to(dtype=torch.float64) for v in add_vecs], dim=0).sum(dim=0).to(dtype=torch.float32)
    pre_hold_norm = float(torch.linalg.vector_norm(cur_vec.to(dtype=torch.float64)).item())
    for hv, h_alpha in hold_vecs:
        cur_vec = j_lens.apply_soft_holds(cur_vec, [hv], alpha=float(h_alpha))

    hw_labels: list[str] = []
    if hold_words and str(hold_words).strip() and float(hold_alpha) > 0.0:
        cur_vec, hw_labels, _, _ = _apply_post_refiner_holds(
            cur_vec,
            family=family,
            hold_words=hold_words,
            hold_alpha=float(hold_alpha),
            prompt=prompt,
        )
    else:
        post_norm = float(torch.linalg.vector_norm(cur_vec.to(dtype=torch.float64)).item())
        if post_norm <= 1e-6 * max(1.0, pre_hold_norm):
            raise ValueError("holds completely cancelled the sculpted vector")

    raw_norm = float(torch.linalg.vector_norm(cur_vec.to(dtype=torch.float64)).item())
    tensors_to_save: dict[str, torch.Tensor] = {"vector.post_refiner": cur_vec}
    points_meta: dict[str, Any] = {
        "post_refiner": {
            "unit": "mean_pair_difference",
            "raw_norm": raw_norm,
            "median_row_norm": (
                float(base_meta["points"]["post_refiner"]["median_row_norm"])
                if base_meta and "post_refiner" in base_meta.get("points", {})
                else j_lens.DEFAULT_POST_REFINER_MEDIAN_ROW_NORM
            ),
        }
    }
    if enc_add_terms:
        enc_cur = torch.stack([v.to(dtype=torch.float64) for v in enc_add_terms], dim=0).sum(dim=0).to(dtype=torch.float32)
        for h_ve, h_a in enc_hold_terms:
            enc_cur = j_lens.apply_soft_holds(enc_cur, [h_ve], alpha=float(h_a))
        en_norm = float(torch.linalg.vector_norm(enc_cur.to(dtype=torch.float64)).item())
        if en_norm > 1e-12:
            tensors_to_save["vector.encoder"] = (enc_cur.to(dtype=torch.float64) / en_norm).to(dtype=torch.float32)
            points_meta["encoder"] = {
                "unit": "median_row_norm",
                "raw_norm": 1.0,
                "median_row_norm": j_lens.DEFAULT_ENCODER_MEDIAN_ROW_NORM,
                "row0_excluded": True,
            }

    jl_readout: dict[str, Any] | None = None
    if j_lens.bundle_available():
        jl_readout = j_lens.decode_vector(cur_vec, space="post_refiner", family=family)
        if prompt:
            jl_readout["prompt_coupling"] = j_lens.analyze_prompt_coupling(
                cur_vec, prompt, readout=jl_readout, family=family
            )

    recipe = recipe_for_fam
    loras_meta = recipe.get("loras")
    spec_str = "|".join(step_labels + hw_labels)
    metadata: dict[str, Any] = {
        "format": cv.FORMAT_VERSION,
        "name": raw_stem,
        "quality": f"Sculpted recipe ({', '.join(step_labels + hw_labels)})",
        "family": family,
        "pairs": {
            "path": "bench/control_vectors.py::sculpt",
            "sha256": hashlib.sha256(spec_str.encode("utf-8")).hexdigest(),
            "n_pairs": len(add_vecs),
            "n_holds": len(hold_vecs) + len(hw_labels),
            "a_side": "baseline",
            "b_side": ", ".join(step_labels),
        },
        "recipe": recipe,
        "measured_under": {
            "mode": "ref2va" if family == "ref2va" else "t2va",
            "frame": "scene",
            "prefix": "none",
            "pooling": "diff_span",
        },
        "points": points_meta,
        "quality_report": {
            "mean_pairwise_cosine": 1.0 if len(add_vecs) >= 2 else None,
            "split_half_cosine": 1.0 if len(add_vecs) >= 2 else None,
            "per_pair_cosine": [1.0 for _ in add_vecs],
            **({"j_lens_readout": jl_readout} if jl_readout else {}),
        },
        "provenance": {
            "core_revision": substrate.git_head(cv_nodes.COMFY_ROOT) or "unknown",
            "pack_commit": substrate.git_head(REPO_ROOT) or "unknown",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "made_by": "bench/control_vectors.py::sculpt/1",
            "sculpt_steps": step_labels + hw_labels,
        },
    }
    fname = cv.default_vector_filename(
        raw_stem,
        family,
        pooling="diff_span",
        apply_holds=bool(hold_vecs or hw_labels),
        n_holds=len(hold_vecs) + len(hw_labels),
        loras=loras_meta,
    )
    saved = cv.save_vector(target_dir / fname, tensors_to_save, metadata, overwrite=overwrite)
    return {
        "file": saved.name,
        "path": saved,
        "family": family,
        "raw_norm": raw_norm,
        "rec_strength": _recommended_strength_text(raw_norm),
        "j_lens": jl_readout,
    }


def _meter_prompt_tokens(
    state: Any,
    *,
    vector_arg: str,
    prompt_arg: str,
    pooling: str = "scene_only",
    family: str | None = None,
) -> dict[str, Any]:
    """Run exact `5376`-d contextual `Qwen3-VL` + `TokenRefiner` per-token receptor metering (`cv.meter_score`)."""
    v_cand = state.vec_dir / vector_arg
    vpath = (
        v_cand
        if v_cand.is_file()
        else cv_nodes.resolve_vector_file_path(vector_arg, preferred_family=family)
    )
    tensors, metadata = cv.load_vector(vpath)
    vec = (
        tensors["vector.post_refiner"]
        if "vector.post_refiner" in tensors
        else next(iter(tensors.values()))
    )
    try:
        label, text = _resolve_prompt_text(prompt_arg)
    except Exception:
        label, text = "prompt", str(prompt_arg).strip()
    if not text:
        raise ValueError("prompt text must be non-empty")

    enc_name = metadata.get("recipe", {}).get("encoder", {}).get("file") or h3_config.MODELS["clip"]
    if enc_name == "unspecified":
        enc_name = h3_config.MODELS["clip"]
    clip = state.get_clip(enc_name, "default")

    eff_fam = family or metadata.get("family", "fl2va")
    dit_name = metadata.get("recipe", {}).get("checkpoint", {}).get("file") or (
        h3_config.MODELS["unet_ref2va"]
        if eff_fam == "ref2va"
        else h3_config.MODELS["unet_fl2va"]
    )
    if dit_name.startswith("in_memory:"):
        dit_name = (
            h3_config.MODELS["unet_ref2va"]
            if eff_fam == "ref2va"
            else h3_config.MODELS["unet_fl2va"]
        )
    dit_path = Path(_h.folder_paths.get_full_path_or_raise("diffusion_models", dit_name))
    enc_path = Path(_h.folder_paths.get_full_path_or_raise("text_encoders", enc_name))
    meter_loras = list(metadata.get("recipe", {}).get("loras", ()))
    ref_cache_key = (
        str(dit_path),
        str(enc_path),
        "cpu",
        json.dumps(meter_loras, sort_keys=True),
    )
    if ref_cache_key in state._refiner_cache:
        _, _, refiner_run = state._refiner_cache[ref_cache_key]
    else:
        recipe = cv_nodes.extract_model_recipe(
            None,
            None,
            dit_path_override=dit_path,
            encoder_path_override=enc_path,
            loras_override=meter_loras,
        )
        refiner_run = cv_nodes.build_refiner_runner(dit_path, device="cpu", loras=meter_loras)
        state._refiner_cache[ref_cache_key] = (eff_fam, recipe, refiner_run)

    t0 = time.perf_counter()
    enc_rows, tags, tids, dec_fn = cv_nodes.encode_prompt_for_vector(clip, text)
    ref_rows = refiner_run(enc_rows)
    sp = h3_spans.spans(tids, tags.tolist(), dec_fn)
    if pooling in cv.VALID_POOLING:
        eff_pooling = pooling if pooling in ("scene_text", "all_tag1") else "scene_text"
        mask = h3_spans.pooling_mask(sp, eff_pooling)
    elif pooling in cv_nodes.MASK_MODES:
        eff_pooling = pooling
        mask = h3_spans.inference_mask(sp, eff_pooling)
    else:
        eff_pooling = "scene_only"
        mask = h3_spans.inference_mask(sp, eff_pooling)
    m = cv.meter_score(ref_rows, vec, mask=mask)

    v64 = vec.detach().to(dtype=torch.float64).reshape(-1)
    v_norm = float(torch.linalg.vector_norm(v64).item())
    v_unit = v64 / max(v_norm, 1e-12)
    rows64 = ref_rows.detach().to(dtype=torch.float64)

    tokens_out: list[dict[str, Any]] = []
    for t_idx in range(len(tids)):
        piece = dec_fn([int(tids[t_idx])])
        r_t = rows64[t_idx]
        r_norm = float(torch.linalg.vector_norm(r_t).item())
        proj_t = float(torch.dot(r_t, v_unit).item())
        cos_t = proj_t / r_norm if r_norm > 1e-12 else 0.0
        in_mask = bool(mask[t_idx].item())
        tokens_out.append(
            {
                "index": t_idx,
                "text": piece,
                "word": piece,
                "token": piece,
                "in_scene_mask": in_mask,
                "cosine": round(cos_t, 4),
                "projection": round(proj_t, 4),
            }
        )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return {
        "ok": True,
        "vector": vpath.name,
        "prompt_label": label,
        "pooling": eff_pooling,
        "elapsed_ms": elapsed_ms,
        "meter": m,
        "tokens": tokens_out,
    }



def _sculpt_steps_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The recipe steps `sculpt`'s arguments describe: one per input vector with its mode and
    strength, `--invert` on the first, then a `hold` step per `--hold`. `cmd_sculpt` saves the
    recipe; the studio's preview route reads it out without saving."""
    vecs = list(args.vectors or ())
    if not vecs:
        raise ValueError("at least one input vector is required for 'sculpt'")
    raw_modes = [m.strip() for m in str(args.modes or "").split(",") if m.strip()]
    raw_strengths = [float(s.strip()) for s in str(args.strengths or "").split(",") if s.strip()]
    invert_flag = bool(getattr(args, "invert", False))

    steps: list[dict[str, Any]] = []
    for idx, v_spec in enumerate(vecs):
        m_val = raw_modes[idx] if idx < len(raw_modes) else (raw_modes[-1] if raw_modes else "add")
        s_val = raw_strengths[idx] if idx < len(raw_strengths) else (raw_strengths[-1] if raw_strengths else 1.0)
        if invert_flag and idx == 0 and s_val > 0:
            s_val = -s_val
        steps.append({"vector": v_spec, "mode": m_val, "strength": s_val})
    for h_spec in (getattr(args, "hold", None) or ()):
        if str(h_spec).strip():
            steps.append(
                {
                    "vector": str(h_spec).strip(),
                    "mode": "hold",
                    "strength": float(getattr(args, "hold_alpha", 1.0)),
                }
            )
    return steps


def cmd_sculpt(args: argparse.Namespace) -> int:
    """Compose one or more control vectors (`add`, `add_norm_matched`, `hold`) + optional `--hold-words` (`--hold-alpha`)
    into a Ready-to-Render `.safetensors` + `.md` sidecar on CPU in `< 5 ms`.
    """
    t0 = time.perf_counter()
    vecs = list(args.vectors or ())
    invert_flag = bool(getattr(args, "invert", False))
    steps = _sculpt_steps_from_args(args)

    stem = (getattr(args, "output_name", None) or getattr(args, "name", None) or "").strip()
    if not stem:
        first_stem = Path(str(vecs[0])).stem.split("__")[0]
        stem = f"{first_stem}_inverted" if invert_flag else f"{first_stem}_sculpted"

    target_fams = _resolve_requested_families(getattr(args, "families", "both"))
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else None
    hold_words = str(getattr(args, "hold_words", "") or "").strip() or None
    hold_alpha = float(getattr(args, "hold_alpha", 1.0))
    prompt_arg = str(getattr(args, "prompt", "") or "").strip() or None

    saved_rows: list[dict[str, Any]] = []
    for fam in target_fams:
        r = _compose_and_save_sculpted_vector(
            steps,
            name=stem,
            family=fam,
            hold_words=hold_words,
            hold_alpha=hold_alpha,
            prompt=prompt_arg,
            out_dir=out_dir,
            overwrite=bool(getattr(args, "overwrite", True)),
        )
        saved_rows.append(r)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"=== Sculpted {len(saved_rows)} Control Vector File(s) in {elapsed_ms:.1f} ms on CPU ===")
    for row in saved_rows:
        print(
            f"  * {row['file']}\n"
            f"      family={row['family']}  raw_norm={row['raw_norm']:.2f}  "
            f"recommended strength: {row['rec_strength']}"
        )
        jl = row.get("j_lens") or {}
        if jl.get("pos_summary"):
            print(f"      j_lens (+): {jl['pos_summary']}")
            print(f"      j_lens (-): {jl['neg_summary']}")
    return 0

