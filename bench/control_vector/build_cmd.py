"""One-shot and batch `build` and `words` CLI commands for H3 control vectors."""

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
    _compile_words_or_spec_to_pairset,
    _display_path,
    _inject_target_prompt_into_pairset,
    _pair_set_to_markdown,
    _parse_lora_specs,
    _recommended_strength_text,
    _resolve_requested_families,
)

def cmd_build(args: argparse.Namespace) -> int:
    if getattr(args, "build_all", False):
        toml_dir = Path(getattr(args, "toml_dir", None) or cv_nodes.PAIR_BANK_DIR)
        all_tomls = sorted(toml_dir.glob("*.toml"))
        if not all_tomls:
            raise ValueError(f"no .toml pair files found in {toml_dir}")
        encoder_name = getattr(args, "encoder", None) or h3_config.MODELS["clip"]
        t_batch_0 = time.perf_counter()
        clip_shared = getattr(args, "_clip_override", None) or _h._load_clip(
            encoder_name, device=args.device
        )
        refiner_cache: dict[Any, Any] = (
            args._refiner_cache
            if isinstance(getattr(args, "_refiner_cache", None), dict)
            else {}
        )
        all_rows: list[dict[str, Any]] = []
        print(
            f"=== Batch Building {len(all_tomls)} Control Vector Banks from {_display_path(toml_dir)} ==="
        )
        for toml_path in all_tomls:
            sub_args = argparse.Namespace(**vars(args))
            sub_args.build_all = False
            sub_args.source = str(toml_path)
            sub_args.name = None
            sub_args.output_name = ""
            sub_args._clip_override = clip_shared
            sub_args._refiner_cache = refiner_cache
            sub_args._summary_rows_out = all_rows
            cmd_build(sub_args)
        if getattr(args, "_summary_rows_out", None) is not None:
            args._summary_rows_out.extend(all_rows)
        t_batch_elapsed = time.perf_counter() - t_batch_0
        print(
            f"\n=== Batch Build Complete: {len(all_rows)} vector file(s) from {len(all_tomls)} TOML bank(s) in {t_batch_elapsed:.2f}s ==="
        )
        return 0

    from_words_str = str(getattr(args, "from_words", "") or "").strip()
    if from_words_str:
        t_w0 = time.perf_counter()
        raw_stem = (getattr(args, "output_name", None) or getattr(args, "name", None) or "").strip()
        if not raw_stem:
            first_w = [w.strip().lower() for part in from_words_str.split(",") for w in part.split() if w.strip()]
            raw_stem = "_".join(first_w[:3]) if first_w else "word_vector"
        stem = "".join(c if (c.isalnum() or c == "_") else "_" for c in raw_stem.lower()).strip("_") or "word_vector"
        if not cv.NAME_RE.match(stem):
            raise ValueError(f"output vector stem must match [a-z0-9_]+, got {stem!r}")

        target_fams = _resolve_requested_families(getattr(args, "families", "both"))
        out_dir = (
            Path(args.out_dir)
            if getattr(args, "out_dir", None)
            else Path(_h.folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        against_w = str(getattr(args, "against_words", "") or "").strip() or None
        hold_w = str(getattr(args, "hold_words", "") or "").strip() or None
        hold_alpha = float(getattr(args, "hold_alpha", 1.0))
        prompt_w = str(getattr(args, "prompt", "") or "").strip() or None

        summary_rows_w: list[dict[str, Any]] = []
        for fam in target_fams:
            synth = j_lens.synthesize_from_words(
                from_words_str,
                against_w,
                family=fam,
                hold_words=hold_w,
                alpha=hold_alpha,
                prompt=prompt_w,
            )
            loras_meta = j_lens._DEFAULT_RECIPES.get(fam, {}).get("loras")
            fname = cv.default_vector_filename(
                stem,
                fam,
                pooling="diff_span",
                loras=loras_meta,
            )
            saved = j_lens.save_synthesized_vector(out_dir / fname, synth, name=stem)
            r_norm = float(synth["readout"]["norm_after_holds"])
            summary_rows_w.append(
                {
                    "file": saved.name,
                    "family": fam,
                    "pooling": "diff_span (j_lens words)",
                    "raw_norm": r_norm,
                    "mpc": 1.0,
                    "split_half": 1.0,
                    "rec_strength": _recommended_strength_text(r_norm),
                    "j_lens": synth["readout"],
                }
            )
        if getattr(args, "_summary_rows_out", None) is not None:
            args._summary_rows_out.extend(summary_rows_w)
        elapsed_w = time.perf_counter() - t_w0
        print(
            f"\n=== Ready to Render in ComfyUI ({len(summary_rows_w)} word-synthesized vector file(s) built in {elapsed_w * 1000.0:.1f} ms on CPU) ==="
        )
        for row in summary_rows_w:
            jl = row.get("j_lens") or {}
            rd = jl.get("readiness") or {}
            print(
                f"  * {row['file']}\n"
                f"      family={row['family']}  pooling={row['pooling']}  "
                f"raw_norm={row['raw_norm']:.2f}  recommended strength: {row['rec_strength']}\n"
                f"      j_lens (+): {jl.get('pos_summary', '')}\n"
                f"      j_lens (-): {jl.get('neg_summary', '')}"
            )
            if rd:
                print(f"      readiness : {rd.get('summary', '')}")
                for rec_line in rd.get("recommendations", ()):
                    print(f"                  -> {rec_line}")
            pc = jl.get("prompt_coupling") or {}
            if pc:
                print(f"      coupling  : {pc.get('summary', '')}")
        return 0

    words_arg = str(getattr(args, "words", "") or "").strip()
    is_placebo = bool(getattr(args, "placebo", False))
    compiled_override: cv.PairSet | None = getattr(args, "_compiled_pair_set", None)

    if (
        not is_placebo
        and not words_arg
        and compiled_override is None
        and (not getattr(args, "source", None) or not str(args.source).strip())
    ):
        raise ValueError("either a source (.md/.toml/stem), --from-words, --words, --placebo, or --all is required for 'build'")

    raw_poolings = [p.strip() for p in str(args.pooling).split(",") if p.strip()]
    if not raw_poolings:
        raise ValueError("--pooling must specify at least one pooling mode")
    poolings: list[str] = []
    for pm in raw_poolings:
        if pm == "all":
            for cand in ("diff_span", "from_first_diff", "scene_text"):
                if cand not in poolings:
                    poolings.append(cand)
        elif pm in cv.VALID_POOLING:
            if pm not in poolings:
                poolings.append(pm)
        else:
            raise ValueError(
                f"pooling must be one of {cv.VALID_POOLING} or 'all', got {pm!r}"
            )

    resolved_pair_path: Path | None = None
    test_prompt_saved: Path | None = None
    stem_override = (getattr(args, "name", None) or getattr(args, "output_name", None) or "").strip() or None
    compile_mode = getattr(args, "mode", None) or ("ref2va" if getattr(args, "ref_images", None) else "t2va")
    slot_arg = str(getattr(args, "slot", "auto") or "auto")
    tp_arg = str(getattr(args, "target_prompt", "") or "").strip()
    tp_weight = float(getattr(args, "target_weight", 2.5))

    if compiled_override is not None:
        pair_set = compiled_override
        if tp_arg:
            pair_set = _inject_target_prompt_into_pairset(
                pair_set, tp_arg, target_weight=tp_weight, slot_hint=slot_arg
            )
    elif is_placebo:
        pair_set = pair_compiler.compile_placebo_pairset(
            slot=slot_arg if slot_arg != "auto" else "opening_framing",
            mode=compile_mode,
        )
    elif words_arg:
        pair_set = _compile_words_or_spec_to_pairset(
            words_arg,
            name=stem_override,
            quality=getattr(args, "quality", None),
            a_side=getattr(args, "a_side", None),
            b_side=getattr(args, "b_side", None),
            slot=slot_arg,
            mode=compile_mode,
            target_prompt=tp_arg,
            target_weight=tp_weight,
        )
        toml_out_arg = getattr(args, "toml_out", None) or (
            cv_nodes.PAIR_BANK_DIR if getattr(args, "save_toml", False) else None
        )
        if toml_out_arg:
            t_out = Path(toml_out_arg)
            if t_out.suffix.lower() != ".toml":
                t_out = t_out / f"{pair_set.name}.toml"
            md_doc_for_toml = _pair_set_to_markdown(None, pair_set)
            resolved_pair_path = cv.write_markdown_to_toml(
                md_doc_for_toml,
                t_out,
                name=t_out.stem,
                mode=pair_set.mode,
                frame=pair_set.frame,
                quality=getattr(args, "quality", None) or pair_set.quality,
                a_side=getattr(args, "a_side", None) or pair_set.a_side,
                b_side=getattr(args, "b_side", None) or pair_set.b_side,
                overwrite=bool(getattr(args, "overwrite", False)),
                bank_dir=cv_nodes.PROMPT_BANK_DIR,
            )
            pair_set = cv.load_pairs(resolved_pair_path, bank_dir=cv_nodes.PROMPT_BANK_DIR)
    else:
        src_arg = str(args.source).strip()
        src_path = Path(src_arg)
        if src_path.suffix.lower() == ".md" or (src_path.is_file() and src_arg.lower().endswith(".md")):
            if not src_path.is_file():
                raise ValueError(f"markdown file not found: {src_path}")
            md_text = _repair_multichange_markdown_pairs(src_path.read_text(encoding="utf-8"))
            parsed = cv.parse_markdown_pairs(md_text)
            toml_target_path = (
                Path(args.toml_out) if getattr(args, "toml_out", None) else cv_nodes.PAIR_BANK_DIR
            )
            eff_stem = (
                stem_override
                if stem_override
                else (toml_target_path.stem if toml_target_path.suffix.lower() == ".toml" else src_path.stem)
            )
            tp_out = getattr(args, "test_prompt_out", None)
            if tp_out is None and parsed.get("test_scene_prompt"):
                base_dir = toml_target_path if toml_target_path.suffix.lower() != ".toml" else toml_target_path.parent
                tp_out = base_dir / f"{eff_stem}_test_prompt.txt"
            try:
                resolved_pair_path = cv.write_markdown_to_toml(
                    md_text,
                    toml_target_path,
                    name=eff_stem,
                    mode=getattr(args, "mode", None),
                    frame=getattr(args, "frame", "scene") or "scene",
                    quality=getattr(args, "quality", None),
                    a_side=getattr(args, "a_side", None),
                    b_side=getattr(args, "b_side", None),
                    test_prompt_out=tp_out,
                    overwrite=bool(getattr(args, "overwrite", False)),
                    bank_dir=cv_nodes.PROMPT_BANK_DIR,
                )
            except (ValueError, FileExistsError) as exc:
                if "already exists" not in str(exc):
                    raise
                cand_toml = (
                    toml_target_path
                    if toml_target_path.suffix.lower() == ".toml"
                    else toml_target_path / f"{eff_stem}.toml"
                )
                if not cand_toml.is_file():
                    raise
                existing_ps = cv.load_pairs(cand_toml, bank_dir=cv_nodes.PROMPT_BANK_DIR)
                existing_ab = [(p.a, p.b) for p in existing_ps.pairs]
                new_ab = [(p["a"], p["b"]) for p in parsed["pairs"]]
                existing_holds = [(h.group, h.a, h.b) for h in existing_ps.holds]
                new_holds = [(h["group"], h["a"], h["b"]) for h in parsed.get("holds", [])]
                if existing_ab != new_ab or existing_holds != new_holds:
                    raise
                resolved_pair_path = cand_toml
                if tp_out is not None and parsed.get("test_scene_prompt"):
                    Path(tp_out).parent.mkdir(parents=True, exist_ok=True)
                    Path(tp_out).write_text(parsed["test_scene_prompt"] + "\n", encoding="utf-8")
            if tp_out is not None and Path(tp_out).is_file():
                test_prompt_saved = Path(tp_out)
            pair_set = cv.load_pairs(resolved_pair_path, bank_dir=cv_nodes.PROMPT_BANK_DIR)
        else:
            resolved_pair_path = cv_nodes.resolve_pair_file_path(src_arg)
            pair_set = cv.load_pairs(resolved_pair_path, bank_dir=cv_nodes.PROMPT_BANK_DIR)
        if tp_arg:
            pair_set = _inject_target_prompt_into_pairset(
                pair_set, tp_arg, target_weight=tp_weight, slot_hint=slot_arg
            )

    stem = (getattr(args, "output_name", None) or getattr(args, "name", None) or pair_set.name).strip()
    if not cv.NAME_RE.match(stem):
        raise ValueError(f"output vector stem must match [a-z0-9_]+, got {stem!r}")

    dit_arg = getattr(args, "dit", None) or getattr(args, "unet", None)
    if dit_arg:
        family_targets = [("custom", str(dit_arg))]
    else:
        auto_fam = "ref2va" if (getattr(args, "ref_images", None) or pair_set.mode == "ref2va") else "fl2va"
        req_fams = _resolve_requested_families(getattr(args, "families", "both"), auto_family=auto_fam)
        family_targets = [
            (f, h3_config.MODELS["unet_ref2va"] if f == "ref2va" else h3_config.MODELS["unet_fl2va"])
            for f in req_fams
        ]

    encoder_name = getattr(args, "encoder", None) or h3_config.MODELS["clip"]
    encoder_path = Path(_h.folder_paths.get_full_path_or_raise("text_encoders", encoder_name))

    t0 = time.perf_counter()
    clip = getattr(args, "_clip_override", None) or _h._load_clip(encoder_name, device=args.device)
    t_clip = time.perf_counter()

    encoded_refs = None
    if getattr(args, "ref_images", None):
        imgs = [s.strip() for s in str(args.ref_images).split(",") if s.strip()]
        encoded_refs = _h._load_encoded_references(
            clip, h3_config.MODELS["video_vae"], imgs
        )

    if encoded_refs is not None:
        ref_hash = hashlib.sha256()
        for sp in getattr(encoded_refs, "spans", ()):
            ref_hash.update(cv._tensor_bf16_bytes(sp.rows))
        prefix_meta: Any = {"references_sha256": ref_hash.hexdigest()}
    else:
        prefix_meta = "none"

    print(f"=== Control Vector One-Shot Build: {stem} ===")
    pairs_label = _display_path(resolved_pair_path) if resolved_pair_path is not None else str(pair_set.path)
    print(f"  pairs_toml        : {pairs_label} ({len(pair_set.pairs)} pairs)")
    if test_prompt_saved is not None:
        print(f"  test_prompt_file  : {_display_path(test_prompt_saved)}")
    print(f"  poolings          : {', '.join(poolings)}")
    print(f"  encoder           : {encoder_path.name} (loaded in {t_clip - t0:.2f}s)")

    # Encode all pairs and holds once through Qwen3-VL layers.49
    t_enc_start = time.perf_counter()
    cached_pairs: list[dict[str, Any]] = []
    enc_censuses: list[dict[str, float]] = []
    weights_list: list[float] = []
    print(f"\n--- Encoding {len(pair_set.pairs)} Contrastive Pairs (Single Pass) ---")
    for idx, item in enumerate(pair_set.pairs):
        enc_a, tags_a, ids_a, dec_a = cv_nodes.encode_prompt_for_vector(
            clip, item.a, encoded_references=encoded_refs
        )
        enc_b, tags_b, ids_b, dec_b = cv_nodes.encode_prompt_for_vector(
            clip, item.b, encoded_references=encoded_refs
        )
        enc_censuses.append(cv.row_norm_census(enc_a, tags_a))
        enc_censuses.append(cv.row_norm_census(enc_b, tags_b))
        first_diff, end_a, end_b = h3_spans.differing_token_bounds(ids_a, ids_b)
        spans_a = h3_spans.spans(ids_a, tags_a.tolist(), dec_a)
        spans_b = h3_spans.spans(ids_b, tags_b.tolist(), dec_b)
        cached_pairs.append(
            {
                "idx": idx,
                "item": item,
                "enc_a": enc_a,
                "tags_a": tags_a,
                "ids_a": ids_a,
                "spans_a": spans_a,
                "enc_b": enc_b,
                "tags_b": tags_b,
                "ids_b": ids_b,
                "spans_b": spans_b,
                "first_diff": first_diff,
                "end_a": end_a,
                "end_b": end_b,
            }
        )
        weights_list.append(float(item.weight))
        print(
            f"  [{idx:02d}] w={item.weight:.2f} diff_toks=[{first_diff}:{end_a}/{end_b}] "
            f"L_a={len(ids_a):3d} L_b={len(ids_b):3d} | "
            f"-{item.a_span[:42]!r} -> +{item.b_span[:42]!r}"
        )

    cached_holds: list[dict[str, Any]] = []
    if not getattr(args, "no_holds", False) and pair_set.holds:
        for h_item in pair_set.holds:
            h_enc_a, h_tags_a, h_ids_a, h_dec_a = cv_nodes.encode_prompt_for_vector(
                clip, h_item.a, encoded_references=encoded_refs
            )
            h_enc_b, h_tags_b, h_ids_b, h_dec_b = cv_nodes.encode_prompt_for_vector(
                clip, h_item.b, encoded_references=encoded_refs
            )
            h_diff, h_end_a, h_end_b = h3_spans.differing_token_bounds(h_ids_a, h_ids_b)
            cached_holds.append(
                {
                    "h_item": h_item,
                    "enc_a": h_enc_a,
                    "ids_a": h_ids_a,
                    "spans_a": h3_spans.spans(h_ids_a, h_tags_a.tolist(), h_dec_a),
                    "enc_b": h_enc_b,
                    "ids_b": h_ids_b,
                    "spans_b": h3_spans.spans(h_ids_b, h_tags_b.tolist(), h_dec_b),
                    "first_diff": h_diff,
                    "end_a": h_end_a,
                    "end_b": h_end_b,
                }
            )
    t_enc_done = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  encoded {len(pair_set.pairs)} pairs in {t_enc_done - t_enc_start:.2f}s")

    w_vec = torch.tensor(weights_list, dtype=torch.float32)
    median_enc_row_norm_ex0 = float(
        sum(c["median_row_norm_ex_row0"] for c in enc_censuses) / len(enc_censuses)
    )
    if resolved_pair_path is not None:
        try:
            rel_pair_path = str(resolved_pair_path.relative_to(REPO_ROOT))
        except ValueError:
            rel_pair_path = resolved_pair_path.name
    else:
        rel_pair_path = str(pair_set.path)

    out_dir = (
        Path(args.out_dir)
        if getattr(args, "out_dir", None)
        else Path(_h.folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    build_cache_out: dict[str, Any] | None = getattr(args, "_build_cache_out", None)
    hold_specs_build = getattr(args, "hold", None)
    hold_words_build = str(getattr(args, "hold_words", "") or "").strip() or None
    hold_alpha_build = float(getattr(args, "hold_alpha", 1.0))
    prompt_build = str(getattr(args, "prompt", "") or getattr(args, "target_prompt", "") or "").strip() or None

    for fam_hint, dit_name in family_targets:
        dit_path = Path(_h.folder_paths.get_full_path_or_raise("diffusion_models", dit_name))
        family = cv.detect_family(dit_path)
        if getattr(args, "no_lora", False):
            loras: list[dict[str, Any]] = []
        elif getattr(args, "lora", None):
            loras = _parse_lora_specs(args.lora)
        else:
            default_lora_name = (
                h3_config.PIKA_REF2VA_SVD995_LORA
                if (family == "ref2va" or fam_hint == "ref2va")
                else h3_config.PIKA_FL2VA_SVD995_LORA
            )
            try:
                loras = _parse_lora_specs([f"{default_lora_name}:1.0"])
            except Exception:
                loras = []

        ref_cache = getattr(args, "_refiner_cache", None)
        cache_key = (
            str(dit_path),
            str(encoder_path),
            str(args.refiner_device),
            json.dumps(loras, sort_keys=True),
        )
        if isinstance(ref_cache, dict) and cache_key in ref_cache:
            family, recipe, refiner_run = ref_cache[cache_key]
        else:
            recipe = cv_nodes.extract_model_recipe(
                None,
                None,
                dit_path_override=dit_path,
                encoder_path_override=encoder_path,
                loras_override=loras,
            )
            refiner_run = cv_nodes.build_refiner_runner(
                dit_path, device=args.refiner_device, loras=loras
            )
            if isinstance(ref_cache, dict):
                ref_cache[cache_key] = (family, recipe, refiner_run)

        # Refine all pairs & holds once for this (family, lora) target
        refined_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        ref_censuses: list[dict[str, float]] = []
        for cp in cached_pairs:
            ra = refiner_run(cp["enc_a"])
            rb = refiner_run(cp["enc_b"])
            refined_pairs.append((ra, rb))
            ref_censuses.append(cv.row_norm_census(ra, cp["tags_a"]))
            ref_censuses.append(cv.row_norm_census(rb, cp["tags_b"]))
        median_ref_row_norm = float(
            sum(c["median_row_norm"] for c in ref_censuses) / len(ref_censuses)
        )

        refined_holds: list[tuple[torch.Tensor, torch.Tensor]] = []
        for ch in cached_holds:
            refined_holds.append((refiner_run(ch["enc_a"]), refiner_run(ch["enc_b"])))

        for pool_mode in poolings:
            a_pooled_list: list[torch.Tensor] = []
            b_pooled_list: list[torch.Tensor] = []
            enc_a_pooled_list: list[torch.Tensor] = []
            enc_b_pooled_list: list[torch.Tensor] = []
            pair_spans: list[dict[str, Any]] = []

            for cp, (ref_a, ref_b) in zip(cached_pairs, refined_pairs):
                first_diff = cp["first_diff"]
                end_a = cp["end_a"]
                end_b = cp["end_b"]
                ids_a = cp["ids_a"]
                ids_b = cp["ids_b"]
                item = cp["item"]
                mask_a = h3_spans.pooling_mask(
                    cp["spans_a"],
                    pool_mode,
                    first_diff=min(first_diff, len(ids_a) - 1),
                    end_diff=end_a,
                    other_ids=ids_b,
                )
                mask_b = h3_spans.pooling_mask(
                    cp["spans_b"],
                    pool_mode,
                    first_diff=min(first_diff, len(ids_b) - 1),
                    end_diff=end_b,
                    other_ids=ids_a,
                )
                a_pooled_list.append(
                    ref_a.to(dtype=torch.float64)[mask_a].mean(dim=0).to(dtype=torch.float32)
                )
                b_pooled_list.append(
                    ref_b.to(dtype=torch.float64)[mask_b].mean(dim=0).to(dtype=torch.float32)
                )
                if getattr(args, "include_encoder", True):
                    mask_a_enc = mask_a.clone()
                    mask_b_enc = mask_b.clone()
                    if mask_a_enc.numel() > 0:
                        mask_a_enc[0] = False
                    if mask_b_enc.numel() > 0:
                        mask_b_enc[0] = False
                    if not bool(mask_a_enc.any()):
                        mask_a_enc = mask_a
                    if not bool(mask_b_enc.any()):
                        mask_b_enc = mask_b
                    enc_a_pooled_list.append(
                        cp["enc_a"].to(dtype=torch.float64)[mask_a_enc].mean(dim=0).to(dtype=torch.float32)
                    )
                    enc_b_pooled_list.append(
                        cp["enc_b"].to(dtype=torch.float64)[mask_b_enc].mean(dim=0).to(dtype=torch.float32)
                    )
                pair_spans.append(
                    {
                        "index": int(cp["idx"]),
                        "weight": float(item.weight),
                        "first_diff": int(first_diff),
                        "end_a": int(end_a),
                        "end_b": int(end_b),
                        "a_span": item.a_span,
                        "b_span": item.b_span,
                        "len_a": int(len(ids_a)),
                        "len_b": int(len(ids_b)),
                        "pooled_rows_a": int(mask_a.sum().item()),
                        "pooled_rows_b": int(mask_b.sum().item()),
                        "pooled_tokens": int(mask_b.sum().item()),
                    }
                )

            a_mat = torch.stack(a_pooled_list, dim=0)
            b_mat = torch.stack(b_pooled_list, dim=0)
            use_unit_mean = bool(getattr(args, "unit_mean", False))
            raw_vec = cv.mean_difference(a_mat, b_mat, w_vec, unit_mean=use_unit_mean)
            tensors_to_save: dict[str, torch.Tensor] = {}

            hold_a_list: list[torch.Tensor] = []
            hold_b_list: list[torch.Tensor] = []
            enc_hold_a_list: list[torch.Tensor] = []
            enc_hold_b_list: list[torch.Tensor] = []
            hold_groups: list[str] = []
            hold_weights: list[float] = []
            if cached_holds:
                for ch, (h_ref_a, h_ref_b) in zip(cached_holds, refined_holds):
                    h_diff = ch["first_diff"]
                    h_ids_a = ch["ids_a"]
                    h_ids_b = ch["ids_b"]
                    h_mask_a = h3_spans.pooling_mask(
                        ch["spans_a"],
                        pool_mode,
                        first_diff=min(h_diff, len(h_ids_a) - 1),
                        end_diff=ch["end_a"],
                        other_ids=h_ids_b,
                    )
                    h_mask_b = h3_spans.pooling_mask(
                        ch["spans_b"],
                        pool_mode,
                        first_diff=min(h_diff, len(h_ids_b) - 1),
                        end_diff=ch["end_b"],
                        other_ids=h_ids_a,
                    )
                    hold_a_list.append(
                        h_ref_a.to(dtype=torch.float64)[h_mask_a].mean(dim=0).to(dtype=torch.float32)
                    )
                    hold_b_list.append(
                        h_ref_b.to(dtype=torch.float64)[h_mask_b].mean(dim=0).to(dtype=torch.float32)
                    )
                    if getattr(args, "include_encoder", True):
                        hma_e = h_mask_a.clone()
                        hmb_e = h_mask_b.clone()
                        if hma_e.numel() > 0:
                            hma_e[0] = False
                        if hmb_e.numel() > 0:
                            hmb_e[0] = False
                        if not bool(hma_e.any()):
                            hma_e = h_mask_a
                        if not bool(hmb_e.any()):
                            hmb_e = h_mask_b
                        enc_hold_a_list.append(
                            ch["enc_a"].to(dtype=torch.float64)[hma_e].mean(dim=0).to(dtype=torch.float32)
                        )
                        enc_hold_b_list.append(
                            ch["enc_b"].to(dtype=torch.float64)[hmb_e].mean(dim=0).to(dtype=torch.float32)
                        )
                    hold_groups.append(ch["h_item"].group)
                    hold_weights.append(float(ch["h_item"].weight))
                basis = cv.hold_basis(
                    torch.stack(hold_a_list, dim=0),
                    torch.stack(hold_b_list, dim=0),
                    hold_groups,
                    torch.tensor(hold_weights, dtype=torch.float32),
                )
                final_vec = cv.project_out(raw_vec, basis)
                tensors_to_save["holds.post_refiner"] = basis
            else:
                final_vec = raw_vec

            applied_holds_build: list[str] = []
            norm_bef_build = 0.0
            norm_aft_build = 0.0
            if not getattr(args, "no_holds", False) and (hold_specs_build or hold_words_build):
                final_vec, applied_holds_build, norm_bef_build, norm_aft_build = _apply_post_refiner_holds(
                    final_vec,
                    family=family,
                    hold_specs=hold_specs_build,
                    hold_words=hold_words_build,
                    hold_alpha=hold_alpha_build,
                    prompt=prompt_build,
                )

            tensors_to_save["vector.post_refiner"] = final_vec
            gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
            q_report = cv.quality_report(a_mat, b_mat, w_vec, generator=gen)
            raw_norm = float(torch.linalg.vector_norm(final_vec.to(dtype=torch.float64)).item())

            points_meta: dict[str, Any] = {
                "post_refiner": {
                    "unit": "mean_pair_difference",
                    "raw_norm": raw_norm,
                    "median_row_norm": median_ref_row_norm,
                }
            }
            enc_a_mat: torch.Tensor | None = None
            enc_b_mat: torch.Tensor | None = None
            if getattr(args, "include_encoder", True):
                enc_a_mat = torch.stack(enc_a_pooled_list, dim=0)
                enc_b_mat = torch.stack(enc_b_pooled_list, dim=0)
                enc_raw_vec = cv.mean_difference(enc_a_mat, enc_b_mat, w_vec, unit_mean=use_unit_mean)
                if enc_hold_a_list:
                    enc_basis = cv.hold_basis(
                        torch.stack(enc_hold_a_list, dim=0),
                        torch.stack(enc_hold_b_list, dim=0),
                        hold_groups,
                        torch.tensor(hold_weights, dtype=torch.float32),
                    )
                    enc_raw_vec = cv.project_out(enc_raw_vec, enc_basis)
                enc_raw_norm = float(
                    torch.linalg.vector_norm(enc_raw_vec.to(dtype=torch.float64)).item()
                )
                if enc_raw_norm <= 1e-12:
                    raise ValueError("vector.encoder has zero norm")
                enc_vec = (enc_raw_vec.to(dtype=torch.float64) / enc_raw_norm).to(
                    dtype=torch.float32
                )
                tensors_to_save["vector.encoder"] = enc_vec
                points_meta["encoder"] = {
                    "unit": "median_row_norm",
                    "raw_norm": enc_raw_norm,
                    "median_row_norm": median_enc_row_norm_ex0,
                    "row0_excluded": True,
                }

            jl_build_readout: dict[str, Any] | None = None
            if j_lens.bundle_available():
                try:
                    jl_build_readout = j_lens.decode_vector(
                        final_vec, space="post_refiner", family=family
                    )
                    if prompt_build:
                        jl_build_readout["prompt_coupling"] = j_lens.analyze_prompt_coupling(
                            final_vec,
                            prompt_build,
                            readout=jl_build_readout,
                            family=family,
                        )
                    q_report["j_lens_readout"] = jl_build_readout
                except Exception:
                    jl_build_readout = None

            measured_mode = "ref2va" if encoded_refs is not None else pair_set.mode
            pairs_dict_build: dict[str, Any] = {
                "path": rel_pair_path,
                "sha256": pair_set.sha256,
                "n_pairs": len(pair_set.pairs),
                "n_holds": len(cached_holds) + len(applied_holds_build),
                "a_side": pair_set.a_side,
                "b_side": pair_set.b_side,
                "pair_spans": pair_spans,
            }
            if resolved_pair_path is None:
                if isinstance( words_arg, str) and words_arg.strip():
                    pairs_dict_build["words_spec"] = str(words_arg).strip()
                elif isinstance(getattr(args, "_words_spec_saved", None), str) and args._words_spec_saved.strip():
                    pairs_dict_build["words_spec"] = str(args._words_spec_saved).strip()
                if slot_arg and slot_arg != "auto":
                    pairs_dict_build["slot"] = slot_arg
                pairs_dict_build["pairs_ab"] = [
                    {"a": p.a, "b": p.b, "weight": float(p.weight)}
                    for p in pair_set.pairs
                ]
                if pair_set.holds:
                    pairs_dict_build["holds_ab"] = [
                        {"group": h.group, "a": h.a, "b": h.b, "weight": float(h.weight)}
                        for h in pair_set.holds
                    ]
            metadata: dict[str, Any] = {
                "format": cv.FORMAT_VERSION,
                "name": stem,
                "quality": pair_set.quality,
                "family": family,
                "pairs": pairs_dict_build,
                "recipe": recipe,
                "measured_under": {
                    "mode": measured_mode,
                    "frame": pair_set.frame,
                    "prefix": prefix_meta,
                    "pooling": pool_mode,
                },
                "points": points_meta,
                "quality_report": q_report,
                "provenance": {
                    "core_revision": substrate.git_head(cv_nodes.COMFY_ROOT) or "unknown",
                    "pack_commit": substrate.git_head(REPO_ROOT) or "unknown",
                    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "made_by": "bench/control_vectors.py",
                    **(
                        {
                            "applied_holds": {
                                "holds": applied_holds_build,
                                "alpha": hold_alpha_build,
                                "norm_before": round(norm_bef_build, 4),
                                "norm_after_raw": round(norm_aft_build, 4),
                            }
                        }
                        if applied_holds_build
                        else {}
                    ),
                },
            }

            fname = cv.default_vector_filename(
                stem,
                family,
                pooling=pool_mode,
                prefix=prefix_meta,
                apply_holds=not bool(getattr(args, "no_holds", False)),
                n_holds=len(pair_set.holds) + len(applied_holds_build),
                loras=loras,
            )
            out_path = out_dir / fname
            saved = cv.save_vector(
                out_path,
                tensors_to_save,
                metadata,
                overwrite=bool(getattr(args, "overwrite", False)),
            )
            if isinstance(build_cache_out, dict):
                build_cache_out[f"{family}:{pool_mode}"] = {
                    "family": family,
                    "pooling": pool_mode,
                    "stem": stem,
                    "out_path": saved,
                    "a_mat": a_mat.detach().cpu(),
                    "b_mat": b_mat.detach().cpu(),
                    "enc_a_mat": enc_a_mat.detach().cpu() if enc_a_mat is not None else None,
                    "enc_b_mat": enc_b_mat.detach().cpu() if enc_b_mat is not None else None,
                    "weights": [float(w) for w in weights_list],
                    "hold_a_list": [t.detach().cpu() for t in hold_a_list],
                    "hold_b_list": [t.detach().cpu() for t in hold_b_list],
                    "enc_hold_a_list": [t.detach().cpu() for t in enc_hold_a_list],
                    "enc_hold_b_list": [t.detach().cpu() for t in enc_hold_b_list],
                    "hold_groups": list(hold_groups),
                    "hold_weights": list(hold_weights),
                    "median_ref_row_norm": median_ref_row_norm,
                    "median_enc_row_norm_ex0": median_enc_row_norm_ex0,
                    "pair_spans": list(pair_spans),
                    "pairs_ab": [(p.a, p.b) for p in pair_set.pairs],
                    "pair_set": pair_set,
                    "rel_pair_path": rel_pair_path,
                    "recipe": recipe,
                    "loras": loras,
                    "prefix_meta": prefix_meta,
                    "measured_mode": measured_mode,
                    "dit_path": dit_path,
                    "unet_path": dit_path,
                    "encoder_path": encoder_path,
                    "refiner_device": str(args.refiner_device),
                    "seed": int(args.seed),
                }
            rec_band = [
                round(50.0 / max(raw_norm, 1e-6), 2),
                round(85.0 / max(raw_norm, 1e-6), 2),
            ]
            summary_rows.append(
                {
                    "file": saved.name,
                    "family": family,
                    "pooling": pool_mode,
                    "raw_norm": raw_norm,
                    "vector_norm": raw_norm,
                    "mpc": float(q_report["mean_pairwise_cosine"]),
                    "split_half": float(q_report["split_half_cosine"]),
                    "split_half_cosine": float(q_report["split_half_cosine"]),
                    "per_pair_cosine": [round(float(x), 4) for x in q_report["per_pair_cosine"]],
                    "pair_spans": pair_spans,
                    "rec_strength": _recommended_strength_text(raw_norm),
                    "recommended_strength_band": rec_band,
                    "j_lens": jl_build_readout,
                }
            )

    if getattr(args, "_summary_rows_out", None) is not None:
        args._summary_rows_out.extend(summary_rows)
    elapsed = time.perf_counter() - t0
    print(f"\n=== Ready to Render in ComfyUI ({len(summary_rows)} vector file(s) built in {elapsed:.2f}s) ===")
    for row in summary_rows:
        print(
            f"  * {row['file']}\n"
            f"      family={row['family']}  pooling={row['pooling']}  "
            f"raw_norm={row['raw_norm']:.2f}  cos={row['mpc']:.4f} (split_half={row['split_half']:.4f})  "
            f"recommended strength: {row['rec_strength']}"
        )
        jl_row = row.get("j_lens")
        if isinstance(jl_row, dict) and jl_row.get("pos_summary"):
            print(f"      j_lens (+): {jl_row['pos_summary']}")
            print(f"      j_lens (-): {jl_row['neg_summary']}")
    if test_prompt_saved is not None:
        print(f"\n  Test Scene Prompt saved to: {_display_path(test_prompt_saved)}")
    return 0



def cmd_words(args: argparse.Namespace) -> int:
    """Synthesize fl2va + ref2va control vectors directly from plain English words in < 2 ms via J-Lens."""
    args.from_words = args.pos_words
    args.build_all = False
    return cmd_build(args)


def _extract_hold_words_from_meta(meta: dict[str, Any]) -> tuple[str | None, float]:
    prov = meta.get("provenance") or {}
    ah = prov.get("applied_holds")
    if not isinstance(ah, dict):
        return None, 1.0
    alpha = float(ah.get("alpha", 1.0))
    word_parts: list[str] = []
    for item in ah.get("holds") or ():
        s = str(item).strip()
        if s.startswith("words:"):
            w = s.split("words:", 1)[1].strip()
            if w:
                word_parts.append(w)
    return (", ".join(word_parts) if word_parts else None), alpha


def _resolve_existing_toml_from_meta(meta: dict[str, Any], toml_dir: Path) -> Path | None:
    stem = str(meta.get("name", "")).strip()
    pairs_path = str((meta.get("pairs") or {}).get("path", "")).strip()
    if pairs_path and pairs_path.endswith(".toml") and not pairs_path.startswith("<"):
        for cand in (Path(pairs_path), REPO_ROOT / pairs_path, toml_dir / Path(pairs_path).name):
            if cand.is_file():
                return cand
    if stem:
        bank_cand = toml_dir / f"{stem}.toml"
        if bank_cand.is_file():
            return bank_cand
    return None


def cmd_rebuild(args: argparse.Namespace) -> int:
    """Rebuild all control vectors in a folder (defaults to `models/control_vectors/`
    + `prompt_bank/control_vectors/`) or a single specified control vector with zero
    hardcoded vector names."""
    t_start = time.perf_counter()
    raw_target = (
        getattr(args, "vector_target", None)
        or getattr(args, "target", None)
        or ""
    )
    raw_target = str(raw_target).strip()

    default_vec_dir = Path(
        getattr(args, "vec_dir", None)
        or getattr(args, "out_dir", None)
        or _h.folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0]
    )
    default_toml_dir = Path(getattr(args, "toml_dir", None) or cv_nodes.PAIR_BANK_DIR)
    encoder_name = getattr(args, "encoder", None) or h3_config.MODELS["clip"]
    clip_box: list[Any] = [getattr(args, "_clip_override", None)]
    refiner_cache: dict[Any, Any] = (
        args._refiner_cache
        if isinstance(getattr(args, "_refiner_cache", None), dict)
        else {}
    )

    def _get_shared_clip() -> Any:
        if clip_box[0] is None:
            clip_box[0] = _h._load_clip(encoder_name, device=getattr(args, "device", "default"))
        return clip_box[0]

    def _rebuild_from_toml(toml_path: Path, vec_dir: Path, families_arg: str, rows_out: list[dict[str, Any]]) -> None:
        sub_args = argparse.Namespace(**vars(args))
        sub_args.build_all = False
        sub_args.from_words = ""
        sub_args.words = ""
        sub_args.placebo = False
        sub_args.source = str(toml_path)
        sub_args.name = None
        sub_args.output_name = ""
        sub_args.families = families_arg
        sub_args.out_dir = str(vec_dir)
        sub_args.overwrite = True
        sub_args._clip_override = _get_shared_clip()
        sub_args._refiner_cache = refiner_cache
        sub_args._summary_rows_out = rows_out
        cmd_build(sub_args)

    def _rebuild_jlens_words(stem: str, items: list[tuple[Path, dict[str, torch.Tensor], dict[str, Any]]], vec_dir: Path, families_override: str | None, rows_out: list[dict[str, Any]]) -> None:
        meta0 = items[0][2]
        pairs_meta = meta0.get("pairs") or {}
        pos_words = str(pairs_meta.get("b_side", "")).strip()
        raw_against = str(pairs_meta.get("a_side", "")).strip()
        default_neutral_str = ", ".join(getattr(j_lens, "DEFAULT_NEUTRAL_WORDS", ()))
        against_words = (
            None
            if (
                not raw_against
                or raw_against == "neutral vocabulary mean"
                or raw_against == default_neutral_str
            )
            else raw_against
        )
        hold_w, hold_alpha = _extract_hold_words_from_meta(meta0)
        if families_override:
            fams = _resolve_requested_families(families_override)
        else:
            fams = sorted({str(m.get("family", "fl2va")) for _, _, m in items if m.get("family") in ("fl2va", "ref2va")}) or ["fl2va", "ref2va"]
        sub_args = argparse.Namespace(**vars(args))
        sub_args.build_all = False
        sub_args.from_words = pos_words
        sub_args.against_words = against_words or ""
        sub_args.hold_words = hold_w or ""
        sub_args.hold_alpha = hold_alpha
        sub_args.name = stem
        sub_args.output_name = stem
        sub_args.families = ",".join(fams)
        sub_args.out_dir = str(vec_dir)
        sub_args.overwrite = True
        sub_args._summary_rows_out = rows_out
        cmd_build(sub_args)
        for p_old, _, _ in items:
            if p_old.is_file() and "__" not in p_old.name and not p_old.with_suffix(".md").is_file():
                try:
                    t_old, m_old = cv.load_vector(p_old)
                    cv.save_vector(p_old, t_old, m_old, overwrite=True)
                except Exception:
                    pass

    def _rebuild_compiled_words(stem: str, items: list[tuple[Path, dict[str, torch.Tensor], dict[str, Any]]], vec_dir: Path, families_override: str | None, rows_out: list[dict[str, Any]]) -> None:
        meta0 = items[0][2]
        pairs_meta = meta0.get("pairs") or {}
        a_side = str(pairs_meta.get("a_side", "")).strip().rstrip(",")
        b_side = str(pairs_meta.get("b_side", "")).strip().rstrip(",")
        measured = meta0.get("measured_under") or {}
        mode_val = str(measured.get("mode", "t2va"))
        pooling_val = str(measured.get("pooling", getattr(args, "pooling", "diff_span")))
        hold_w, hold_alpha = _extract_hold_words_from_meta(meta0)
        # Guard against accidental self-hold where hold_words was set to b_side
        if hold_w and b_side and hold_w.lower().strip() == b_side.lower().strip():
            hold_w = None
        if families_override:
            fams = _resolve_requested_families(families_override)
        else:
            fams = sorted({str(m.get("family", "fl2va")) for _, _, m in items if m.get("family") in ("fl2va", "ref2va")}) or ["fl2va", "ref2va"]
        words_spec_saved = str(pairs_meta.get("words_spec", "")).strip()
        slot_saved = str(pairs_meta.get("slot", "auto")).strip() or "auto"
        raw_pairs_ab = pairs_meta.get("pairs_ab")
        raw_holds_ab = pairs_meta.get("holds_ab")
        reconstructed_ps: cv.PairSet | None = None
        if isinstance(raw_pairs_ab, list) and len(raw_pairs_ab) >= 2:
            try:
                p_items: list[cv.PairItem] = []
                for idx_p, item_p in enumerate(raw_pairs_ab):
                    a_txt = str(item_p["a"])
                    b_txt = str(item_p["b"])
                    w_val = float(item_p.get("weight", 1.0))
                    a_sp, b_sp = cv.differing_span(a_txt, b_txt, frame=str(measured.get("frame", "scene")))
                    p_items.append(cv.PairItem(index=idx_p, a=a_txt, b=b_txt, weight=w_val, a_span=a_sp, b_span=b_sp))
                h_items: list[cv.HoldItem] = []
                if isinstance(raw_holds_ab, list):
                    for idx_h, item_h in enumerate(raw_holds_ab):
                        ha_txt = str(item_h["a"])
                        hb_txt = str(item_h["b"])
                        hw_val = float(item_h.get("weight", 1.0))
                        h_grp = str(item_h.get("group", f"hold_{idx_h}"))
                        ha_sp, hb_sp = cv.differing_span(ha_txt, hb_txt, frame=str(measured.get("frame", "scene")))
                        h_items.append(cv.HoldItem(index=idx_h, group=h_grp, a=ha_txt, b=hb_txt, weight=hw_val, a_span=ha_sp, b_span=hb_sp))
                reconstructed_ps = cv.PairSet(
                    path=Path(str(pairs_meta.get("path", f"<compiled:{stem}>"))),
                    sha256=str(pairs_meta.get("sha256", "")),
                    name=stem,
                    quality=str(meta0.get("quality", stem)),
                    mode=mode_val if mode_val in cv.VALID_MODES else "t2va",
                    frame=str(measured.get("frame", "scene")),
                    a_side=a_side or "baseline",
                    b_side=b_side or stem,
                    pairs=tuple(p_items),
                    holds=tuple(h_items),
                    exclude_shots=(),
                )
            except Exception:
                reconstructed_ps = None

        words_spec = (
            words_spec_saved
            if words_spec_saved
            else (f"{a_side} -> {b_side}" if a_side and not a_side.startswith("balanced neutral contrast") else b_side)
        )
        sub_args = argparse.Namespace(**vars(args))
        sub_args.build_all = False
        sub_args.from_words = ""
        sub_args.placebo = False
        sub_args.source = None
        sub_args.words = "" if reconstructed_ps is not None else words_spec
        sub_args._words_spec_saved = words_spec
        sub_args._compiled_pair_set = reconstructed_ps
        sub_args.slot = slot_saved
        sub_args.mode = mode_val if mode_val in cv.VALID_MODES else "t2va"
        sub_args.pooling = pooling_val if pooling_val in cv.VALID_POOLING else "diff_span"
        sub_args.quality = str(meta0.get("quality", "")) or None
        sub_args.a_side = a_side or None
        sub_args.b_side = b_side or None
        sub_args.hold_words = hold_w or ""
        sub_args.hold_alpha = hold_alpha
        sub_args.name = stem
        sub_args.output_name = stem
        sub_args.families = ",".join(fams)
        sub_args.out_dir = str(vec_dir)
        sub_args.overwrite = True
        sub_args._clip_override = _get_shared_clip()
        sub_args._refiner_cache = refiner_cache
        sub_args._summary_rows_out = rows_out
        cmd_build(sub_args)

    def _refresh_sidecar_and_jlens(items: list[tuple[Path, dict[str, torch.Tensor], dict[str, Any]]], rows_out: list[dict[str, Any]]) -> None:
        for p, tensors, meta in items:
            fam = str(meta.get("family", "fl2va"))
            qrep = meta.setdefault("quality_report", {})
            if "j_lens_readout" not in qrep and j_lens.bundle_available() and "vector.post_refiner" in tensors:
                try:
                    qrep["j_lens_readout"] = j_lens.decode_vector(
                        tensors["vector.post_refiner"], space="post_refiner", family=fam
                    )
                except Exception:
                    pass
            saved = cv.save_vector(p, tensors, meta, overwrite=True)
            pr_norm = float(meta.get("points", {}).get("post_refiner", {}).get("raw_norm", 0.0))
            rows_out.append(
                {
                    "file": saved.name,
                    "family": fam,
                    "pooling": str(meta.get("measured_under", {}).get("pooling", "diff_span")),
                    "raw_norm": pr_norm,
                    "mpc": float(qrep.get("mean_pairwise_cosine", 1.0)),
                    "split_half": float(qrep.get("split_half_cosine", 1.0)),
                    "rec_strength": _recommended_strength_text(pr_norm),
                    "j_lens": qrep.get("j_lens_readout"),
                }
            )

    all_rows: list[dict[str, Any]] = []
    target_path = Path(raw_target) if raw_target else None

    # Mode 1: Single vector target (file or stem, not a directory)
    if target_path is not None and not target_path.is_dir():
        # Check if target resolves directly to a .toml file
        toml_cand: Path | None = None
        if target_path.suffix.lower() == ".toml" and target_path.is_file():
            toml_cand = target_path
        elif not target_path.suffix and (default_toml_dir / f"{raw_target}.toml").is_file():
            toml_cand = default_toml_dir / f"{raw_target}.toml"

        # Also check if a specific .safetensors file was named so we preserve its family if --families wasn't explicitly restricted
        matched_st: list[Path] = []
        if target_path.is_file() and target_path.suffix.lower() == ".safetensors":
            matched_st = [target_path]
        elif (default_vec_dir / raw_target).is_file():
            matched_st = [default_vec_dir / raw_target]
        elif (default_vec_dir / f"{raw_target}.safetensors").is_file():
            matched_st = [default_vec_dir / f"{raw_target}.safetensors"]
        else:
            stem_q = target_path.stem.split("__")[0]
            matched_st = sorted(default_vec_dir.glob(f"{stem_q}__*.safetensors"))
            if not matched_st:
                matched_st = sorted(default_vec_dir.glob(f"{stem_q}_*.safetensors"))
            if toml_cand is None and (default_toml_dir / f"{stem_q}.toml").is_file():
                toml_cand = default_toml_dir / f"{stem_q}.toml"

        if toml_cand is None and not matched_st:
            raise ValueError(
                f"control vector or pair TOML {raw_target!r} not found in {default_vec_dir} or {default_toml_dir}"
            )

        loaded_items: list[tuple[Path, dict[str, torch.Tensor], dict[str, Any]]] = []
        for p in matched_st:
            t, m = cv.load_vector(p)
            loaded_items.append((p, t, m))
            if toml_cand is None:
                toml_cand = _resolve_existing_toml_from_meta(m, default_toml_dir)

        # If the user pointed at a single .safetensors file with a specific family, rebuild that family unless --families was explicitly passed
        single_fam_override = getattr(args, "families", "both")
        if len(loaded_items) == 1 and target_path.suffix.lower() == ".safetensors" and single_fam_override == "both":
            fam_single = str(loaded_items[0][2].get("family", ""))
            if fam_single in ("fl2va", "ref2va"):
                single_fam_override = fam_single

        if toml_cand is not None:
            print(f"=== Rebuilding Single Control Vector `{toml_cand.stem}` from {_display_path(toml_cand)} ===")
            _rebuild_from_toml(toml_cand, default_vec_dir, single_fam_override, all_rows)
        else:
            stem = str(loaded_items[0][2].get("name", loaded_items[0][0].stem.split("__")[0]))
            made_by = str(loaded_items[0][2].get("provenance", {}).get("made_by", ""))
            pairs_path = str(loaded_items[0][2].get("pairs", {}).get("path", ""))
            a_side = str(loaded_items[0][2].get("pairs", {}).get("a_side", "")).strip()
            b_side = str(loaded_items[0][2].get("pairs", {}).get("b_side", "")).strip()
            if made_by.startswith("j_lens.synthesize_from_words") or pairs_path == "j_lens.synthesize_from_words":
                print(f"=== Rebuilding J-Lens Word Control Vector `{stem}` ===")
                _rebuild_jlens_words(stem, loaded_items, default_vec_dir, single_fam_override, all_rows)
            elif b_side and not pairs_path.startswith("<media_pairs:") and not made_by.startswith("j_lens.save_composed_recipe_vector"):
                print(f"=== Rebuilding Compiled Control Vector `{stem}` ({a_side!r} -> {b_side!r}) ===")
                _rebuild_compiled_words(stem, loaded_items, default_vec_dir, single_fam_override, all_rows)
            else:
                print(f"=== Refreshing Control Vector Sidecar & J-Lens Readout for `{stem}` ===")
                _refresh_sidecar_and_jlens(loaded_items, all_rows)

        if getattr(args, "_summary_rows_out", None) is not None:
            args._summary_rows_out.extend(all_rows)
        elapsed = time.perf_counter() - t_start
        print(f"\n=== Rebuild Complete: {len(all_rows)} vector file(s) updated in {elapsed:.2f}s ===")
        return 0

    # Mode 2: Folder mode (defaulting to models/control_vectors/ + prompt_bank/control_vectors/)
    if target_path is not None and target_path.is_dir():
        has_tomls = any(target_path.glob("*.toml"))
        has_sts = any(target_path.glob("*.safetensors"))
        if has_tomls and not has_sts:
            toml_dir = target_path
            vec_dir = default_vec_dir
            scan_tomls = True
            scan_vecs = False
        else:
            vec_dir = target_path
            toml_dir = default_toml_dir
            scan_tomls = (vec_dir.resolve() == default_vec_dir.resolve()) or has_tomls
            scan_vecs = True
    else:
        vec_dir = default_vec_dir
        toml_dir = default_toml_dir
        scan_tomls = True
        scan_vecs = True

    toml_by_stem: dict[str, Path] = {}
    if scan_tomls and toml_dir.is_dir():
        for tp in sorted(toml_dir.glob("*.toml")):
            toml_by_stem[tp.stem] = tp

    vecs_by_stem: dict[str, list[tuple[Path, dict[str, torch.Tensor], dict[str, Any]]]] = {}
    if scan_vecs and vec_dir.is_dir():
        for sp in sorted(vec_dir.glob("*.safetensors")):
            try:
                t, m = cv.load_vector(sp)
            except Exception:
                continue
            stem = str(m.get("name", sp.stem.split("__")[0])).strip()
            vecs_by_stem.setdefault(stem, []).append((sp, t, m))
            if stem not in toml_by_stem:
                resolved_t = _resolve_existing_toml_from_meta(m, toml_dir)
                if resolved_t is not None:
                    toml_by_stem[stem] = resolved_t

    all_stems = sorted(set(toml_by_stem.keys()) | set(vecs_by_stem.keys()))
    if not all_stems:
        raise ValueError(f"no control vectors or pair TOMLs found in {vec_dir} or {toml_dir}")

    print(
        f"=== Rebuilding {len(all_stems)} Control Vector Stem(s) (vec_dir={vec_dir}, toml_dir={_display_path(toml_dir)}) ==="
    )
    req_fams_str = getattr(args, "families", "both")
    req_fams_set = set(_resolve_requested_families(req_fams_str))
    force_rebuild = bool(getattr(args, "force", False))
    for idx, stem in enumerate(all_stems, start=1):
        if stem in toml_by_stem:
            tp = toml_by_stem[stem]
            existing_items = vecs_by_stem.get(stem, [])
            if not force_rebuild and existing_items:
                try:
                    ps_check = cv.load_pairs(tp, bank_dir=cv_nodes.PROMPT_BANK_DIR)
                    existing_fams = {str(m.get("family", "")) for _, _, m in existing_items}
                    all_complete = req_fams_set.issubset(existing_fams) and all(
                        ("vector.encoder" in t or not getattr(args, "include_encoder", True))
                        and "j_lens_readout" in (m.get("quality_report") or {})
                        and (m.get("pairs") or {}).get("sha256") == ps_check.sha256
                        for _, t, m in existing_items
                    )
                except Exception:
                    all_complete = False
                if all_complete:
                    print(
                        f"\n[{idx}/{len(all_stems)}] `{stem}` is up-to-date with {_display_path(tp)}; refreshing sidecars ..."
                    )
                    _refresh_sidecar_and_jlens(existing_items, all_rows)
                    continue
            print(f"\n[{idx}/{len(all_stems)}] Rebuilding `{stem}` from {_display_path(tp)} ...")
            _rebuild_from_toml(tp, vec_dir, req_fams_str, all_rows)
            continue

        items = vecs_by_stem.get(stem, [])
        if not items:
            continue
        meta0 = items[0][2]
        made_by = str(meta0.get("provenance", {}).get("made_by", ""))
        pairs_path = str(meta0.get("pairs", {}).get("path", ""))
        b_side = str(meta0.get("pairs", {}).get("b_side", "")).strip()
        needs_encoder = bool(getattr(args, "include_encoder", True)) and any(
            "vector.encoder" not in t for _, t, _ in items
        )
        if made_by.startswith("j_lens.synthesize_from_words") or pairs_path == "j_lens.synthesize_from_words":
            print(f"\n[{idx}/{len(all_stems)}] Rebuilding J-Lens word vector `{stem}` ...")
            _rebuild_jlens_words(stem, items, vec_dir, None, all_rows)
        elif needs_encoder and b_side and not pairs_path.startswith("<media_pairs:") and not made_by.startswith("j_lens.save_composed_recipe_vector"):
            print(f"\n[{idx}/{len(all_stems)}] Rebuilding compiled vector `{stem}` ...")
            _rebuild_compiled_words(stem, items, vec_dir, None, all_rows)
        else:
            print(f"\n[{idx}/{len(all_stems)}] Refreshing sidecar & J-Lens readout for `{stem}` ({len(items)} file(s)) ...")
            _refresh_sidecar_and_jlens(items, all_rows)

    if getattr(args, "_summary_rows_out", None) is not None:
        args._summary_rows_out.extend(all_rows)
    elapsed = time.perf_counter() - t_start
    print(
        f"\n=== Folder Rebuild Complete: {len(all_rows)} vector file(s) across {len(all_stems)} stem(s) updated in {elapsed:.2f}s ==="
    )
    return 0


