"""CLI commands for `make`, `make-media`, and `from-md` (`import-md`)."""

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
    _load_image_tensor_list,
    _parse_lora_specs,
    _resolve_cli_target_prompt,
    _resolve_requested_families,
)

def cmd_make_media(args: argparse.Namespace) -> int:
    if args.pooling not in cv.VALID_MEDIA_POOLING:
        raise ValueError(
            f"pooling must be one of {cv.VALID_MEDIA_POOLING}, got {args.pooling!r}"
        )
    list_a = _load_image_tensor_list(args.image_a)
    list_b = _load_image_tensor_list(args.image_b)
    if len(list_a) != len(list_b):
        raise ValueError(
            f"--image-a and --image-b must have the same number of images, got {len(list_a)} vs {len(list_b)}"
        )
    # Match geometry per pair to the first image's aspect ratio at short_edge
    matched_a_list: list[torch.Tensor] = []
    matched_b_list: list[torch.Tensor] = []
    target_size = (0, 0)
    for ia, ib in zip(list_a, list_b):
        ma, mb, t_sz = cv_nodes.match_media_pair_geometry(
            ia if not matched_a_list else cv_nodes._repo("reference_conditioning")._view_or_source(ia, ia.shape[2], ia.shape[1], target_size[0], target_size[1]),
            ib,
            short_edge=int(args.short_edge),
        )
        target_size = t_sz
        matched_a_list.append(ma)
        matched_b_list.append(mb)
    batch_a = torch.cat(matched_a_list, dim=0)
    batch_b = torch.cat(matched_b_list, dim=0)

    dit_override = getattr(args, "dit", None) or getattr(args, "unet", None)
    families_raw = getattr(args, "families", None)
    if dit_override:
        dit_names = [str(dit_override)]
    elif families_raw:
        fams = _resolve_requested_families(families_raw, auto_family="ref2va")
        dit_names = [
            h3_config.MODELS["unet_ref2va"] if f == "ref2va" else h3_config.MODELS["unet_fl2va"]
            for f in fams
        ]
    else:
        dit_names = [h3_config.MODELS["unet_ref2va"]]

    encoder_name = args.encoder or h3_config.MODELS["clip"]
    encoder_path = Path(_h.folder_paths.get_full_path_or_raise("text_encoders", encoder_name))
    loras = _parse_lora_specs(args.lora)

    clip = getattr(args, "_clip_override", None) or _h._load_clip(encoder_name, device=args.device)
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else None
    hold_specs = getattr(args, "hold", None)
    hold_words = str(getattr(args, "hold_words", "") or "").strip() or None
    hold_alpha = float(getattr(args, "hold_alpha", 1.0))

    for dit_name in dit_names:
        dit_path = Path(_h.folder_paths.get_full_path_or_raise("diffusion_models", dit_name))
        model_or_path = getattr(args, "_model_override", None) or dit_path
        if getattr(args, "no_lora", False):
            loras_fam: list[dict[str, Any]] = []
        elif loras:
            loras_fam = loras
        elif getattr(args, "_model_override", None) is not None or dit_override:
            loras_fam = []
        else:
            fam_detected = cv.detect_family(dit_path)
            default_lora_name = (
                h3_config.PIKA_REF2VA_SVD995_LORA
                if fam_detected == "ref2va"
                else h3_config.PIKA_FL2VA_SVD995_LORA
            )
            loras_fam = _parse_lora_specs([f"{default_lora_name}:1.0"])
        out_path, report = cv_nodes.make_control_vector_from_media_pairs(
            image_a=batch_a,
            image_b=batch_b,
            model=model_or_path,
            clip=clip,
            output_name=str(args.output_name or "media_contrast"),
            pooling=str(args.pooling),
            short_edge=int(args.short_edge),
            scene_prompt=str(args.scene_prompt or ""),
            output_dir=out_dir,
            overwrite=bool(args.overwrite),
            made_by="bench/control_vectors.py::make-media/1",
            loras=loras_fam,
            encoder_path_override=encoder_path,
            refiner_device=getattr(args, "refiner_device", "cpu"),
        )
        if hold_specs or hold_words:
            m_tensors, m_meta = cv.load_vector(out_path)
            m_fam = str(m_meta.get("family", "ref2va"))
            scrubbed_vec, hold_labels, n_bef, n_aft = _apply_post_refiner_holds(
                m_tensors["vector.post_refiner"],
                family=m_fam,
                hold_specs=hold_specs,
                hold_words=hold_words,
                hold_alpha=hold_alpha,
                prompt=str(args.scene_prompt or "") or None,
            )
            if hold_labels:
                m_tensors["vector.post_refiner"] = scrubbed_vec
                m_meta.setdefault("provenance", {})["applied_holds"] = {
                    "holds": hold_labels,
                    "alpha": hold_alpha,
                    "norm_before": round(n_bef, 4),
                    "norm_after_raw": round(n_aft, 4),
                }
                if j_lens.bundle_available():
                    try:
                        m_meta.setdefault("quality_report", {})["j_lens_readout"] = j_lens.decode_vector(
                            scrubbed_vec, space="post_refiner", family=m_fam
                        )
                    except Exception:
                        pass
                cv.save_vector(out_path, m_tensors, m_meta, overwrite=True)
                report["metadata"] = m_meta

        qrep = report["metadata"]["quality_report"]
        pts = report["metadata"]["points"]["post_refiner"]
        mpc_str = (
            f"{qrep['mean_pairwise_cosine']:.4f}"
            if qrep.get("mean_pairwise_cosine") is not None
            else "n/a (1 pair)"
        )
        sh_str = (
            f"{qrep['split_half_cosine']:.4f}"
            if qrep.get("split_half_cosine") is not None
            else "n/a (1 pair)"
        )
        print(f"=== Media-Contrast Control Vector Saved: {out_path.name} ===")
        print(f"  path            : {out_path}")
        print(f"  qwen_view_size  : {report['qwen_view_size'][0]}x{report['qwen_view_size'][1]} (32-px M-RoPE grid)")
        print(f"  pooling         : {args.pooling}")
        print(f"  raw_norm        : {pts['raw_norm']:.4f}   median_row_norm: {pts['median_row_norm']:.4f}")
        print(f"  pairwise_cosine : {mpc_str}   split_half: {sh_str}")
    return 0


def cmd_make(args: argparse.Namespace) -> int:
    if args.pooling not in cv.VALID_POOLING:
        raise ValueError(f"pooling must be one of {cv.VALID_POOLING}, got {args.pooling!r}")

    resolved_pair_path: Path | None = None
    words_arg = str(getattr(args, "words", "") or "").strip()
    tp_arg = _resolve_cli_target_prompt(args)
    tp_weight = float(getattr(args, "target_weight", 2.5))
    if words_arg:
        compile_mode = pair_compiler.infer_compile_mode(
            getattr(args, "mode", None),
            ref_images=getattr(args, "ref_images", None),
            target_prompt=tp_arg or None,
            words=words_arg,
        )
        pair_set = _compile_words_or_spec_to_pairset(
            words_arg,
            name=(getattr(args, "output_name", "") or "").strip() or None,
            mode=compile_mode,
            target_prompt=tp_arg,
            target_weight=tp_weight,
            bank_scenes=int(getattr(args, "bank_scenes", 0) or 0),
        )
    else:
        if not getattr(args, "pairs", None):
            raise ValueError("either a pair TOML file or --words is required for 'make'")
        if tp_arg:
            raise ValueError(
                "--target-prompt requires --words (or use 'from-prompt'); .toml pair banks are not spliced into --target-prompt"
            )
        resolved_pair_path = cv_nodes.resolve_pair_file_path(args.pairs)
        pair_set = cv.load_pairs(resolved_pair_path, bank_dir=cv_nodes.PROMPT_BANK_DIR)

    auto_fam = "ref2va" if (args.ref_images or pair_set.mode == "ref2va") else "fl2va"
    req_fams = _resolve_requested_families(
        getattr(args, "families", "both"), auto_family=auto_fam
    )
    primary_fam = auto_fam if auto_fam in req_fams else req_fams[0]
    dit_name = getattr(args, "dit", None) or getattr(args, "unet", None) or (
        h3_config.MODELS["unet_ref2va"]
        if primary_fam == "ref2va"
        else h3_config.MODELS["unet_fl2va"]
    )
    dit_path = Path(_h.folder_paths.get_full_path_or_raise("diffusion_models", dit_name))
    encoder_name = args.encoder or h3_config.MODELS["clip"]
    encoder_path = Path(_h.folder_paths.get_full_path_or_raise("text_encoders", encoder_name))

    family = cv.detect_family(dit_path)
    explicit_loras = _parse_lora_specs(args.lora)

    def _resolve_make_loras(fam: str) -> list[dict[str, Any]]:
        if getattr(args, "no_lora", False):
            return []
        if explicit_loras:
            return explicit_loras
        if getattr(args, "dit", None) or getattr(args, "unet", None):
            return []
        default_lora_name = (
            h3_config.PIKA_REF2VA_SVD995_LORA
            if fam == "ref2va"
            else h3_config.PIKA_FL2VA_SVD995_LORA
        )
        return _parse_lora_specs([f"{default_lora_name}:1.0"])

    loras = _resolve_make_loras(family)
    recipe = cv_nodes.extract_model_recipe(
        None,
        None,
        dit_path_override=dit_path,
        encoder_path_override=encoder_path,
        loras_override=loras,
    )

    t0 = time.perf_counter()
    clip = getattr(args, "_clip_override", None) or _h._load_clip(encoder_name, device=args.device)
    t_clip = time.perf_counter()
    ref_cache = getattr(args, "_refiner_cache", None)
    ref_key = (
        str(dit_path),
        str(args.refiner_device),
        tuple((l["file"], float(l["strength"])) for l in loras),
    )
    if isinstance(ref_cache, dict) and ref_key in ref_cache:
        refiner_run = ref_cache[ref_key]
    else:
        refiner_run = cv_nodes.build_refiner_runner(
            dit_path, device=args.refiner_device, loras=loras
        )
        if isinstance(ref_cache, dict):
            ref_cache[ref_key] = refiner_run
    t_ref = time.perf_counter()

    encoded_refs = None
    if args.ref_images:
        imgs = [s.strip() for s in args.ref_images.split(",") if s.strip()]
        encoded_refs = _h._load_encoded_references(
            clip, h3_config.MODELS["video_vae"], imgs
        )

    print(f"=== Control Vector Make: {pair_set.name} ===")
    pairs_label = (
        str(resolved_pair_path.relative_to(REPO_ROOT) if resolved_pair_path.is_relative_to(REPO_ROOT) else resolved_pair_path)
        if resolved_pair_path is not None
        else str(pair_set.path)
    )
    print(f"  pairs_file  : {pairs_label}")
    print(f"  pooling     : {args.pooling}   frame: {pair_set.frame}   mode: {pair_set.mode}")
    print(f"  family      : {family}")
    print(f"  checkpoint  : {recipe['checkpoint']['file']} ({recipe['checkpoint']['sha256'][:12]})")
    if loras:
        lora_summary = ", ".join(
            f"{l['file']}:{l['strength']:g} ({l['sha256'][:12]})" for l in loras
        )
        print(f"  loras       : {lora_summary}")
    print(f"  encoder     : {recipe['encoder']['file']} ({recipe['encoder']['sha256'][:12]})")
    print(f"  load_times  : encoder={t_clip - t0:.2f}s   refiner={t_ref - t_clip:.2f}s")

    a_pooled_list: list[torch.Tensor] = []
    b_pooled_list: list[torch.Tensor] = []
    enc_a_pooled_list: list[torch.Tensor] = []
    enc_b_pooled_list: list[torch.Tensor] = []
    cached_pair_encs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    ref_censuses: list[dict[str, float]] = []
    pair_spans: list[dict[str, Any]] = []

    print(f"\n--- Pair Spans ({len(pair_set.pairs)} pairs) ---")
    raw_cached_pairs, raw_cached_holds, enc_censuses, weights_list = _h._encode_pair_set_through_clip(
        clip,
        pair_set,
        encoded_refs=encoded_refs,
        no_holds=bool(args.no_holds),
        log_rows=False,
    )
    for cp in raw_cached_pairs:
        idx = cp["idx"]
        item = cp["item"]
        enc_a = cp["enc_a"]
        tags_a = cp["tags_a"]
        ids_a = cp["ids_a"]
        spans_a = cp["spans_a"]
        enc_b = cp["enc_b"]
        tags_b = cp["tags_b"]
        ids_b = cp["ids_b"]
        spans_b = cp["spans_b"]
        first_diff = cp["first_diff"]
        end_a = cp["end_a"]
        end_b = cp["end_b"]
        ref_a = refiner_run(enc_a)
        ref_b = refiner_run(enc_b)

        ref_censuses.append(cv.row_norm_census(ref_a, tags_a))
        ref_censuses.append(cv.row_norm_census(ref_b, tags_b))

        mask_a = h3_spans.pooling_mask(
            spans_a,
            args.pooling,
            first_diff=min(first_diff, len(ids_a) - 1),
            end_diff=end_a,
            other_ids=ids_b,
        )
        mask_b = h3_spans.pooling_mask(
            spans_b,
            args.pooling,
            first_diff=min(first_diff, len(ids_b) - 1),
            end_diff=end_b,
            other_ids=ids_a,
        )
        cached_pair_encs.append((enc_a, enc_b, tags_a, tags_b, mask_a, mask_b))

        a_pooled_list.append(ref_a.to(dtype=torch.float64)[mask_a].mean(dim=0).to(dtype=torch.float32))
        b_pooled_list.append(ref_b.to(dtype=torch.float64)[mask_b].mean(dim=0).to(dtype=torch.float32))
        if args.include_encoder:
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
            enc_a_pooled_list.append(enc_a.to(dtype=torch.float64)[mask_a_enc].mean(dim=0).to(dtype=torch.float32))
            enc_b_pooled_list.append(enc_b.to(dtype=torch.float64)[mask_b_enc].mean(dim=0).to(dtype=torch.float32))
        pair_spans.append(
            {
                "index": idx,
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
        pool_info = (
            f"pool=({int(mask_a.sum().item()):2d}t/{int(mask_b.sum().item()):2d}t [{first_diff}:{end_a}/{end_b}]) "
            if args.pooling == "diff_span"
            else ""
        )
        print(
            f"  [{idx:02d}] w={item.weight:.2f} first_diff_tok={first_diff:3d} "
            f"L_a={len(ids_a):3d} L_b={len(ids_b):3d} {pool_info}| "
            f"-{item.a_span[:44]!r} -> +{item.b_span[:44]!r}"
        )

    a_mat = torch.stack(a_pooled_list, dim=0)
    b_mat = torch.stack(b_pooled_list, dim=0)
    w_vec = torch.tensor(weights_list, dtype=torch.float32)

    use_unit_mean_make = bool(getattr(args, "unit_mean", False))
    raw_vec = cv.mean_difference(
        a_mat, b_mat, w_vec, unit_mean=use_unit_mean_make
    )
    tensors_to_save: dict[str, torch.Tensor] = {}

    enc_hold_a_make: list[torch.Tensor] = []
    enc_hold_b_make: list[torch.Tensor] = []
    cached_hold_encs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    hold_groups: list[str] = []
    hold_weights: list[float] = []
    if raw_cached_holds:
        hold_a_list: list[torch.Tensor] = []
        hold_b_list: list[torch.Tensor] = []
        for ch in raw_cached_holds:
            h_item = ch["h_item"]
            h_enc_a = ch["enc_a"]
            h_ids_a = ch["ids_a"]
            h_spans_a = ch["spans_a"]
            h_enc_b = ch["enc_b"]
            h_ids_b = ch["ids_b"]
            h_spans_b = ch["spans_b"]
            h_diff = ch["first_diff"]
            h_end_a = ch["end_a"]
            h_end_b = ch["end_b"]
            h_ref_a = refiner_run(h_enc_a)
            h_ref_b = refiner_run(h_enc_b)
            h_mask_a = h3_spans.pooling_mask(
                h_spans_a,
                args.pooling,
                first_diff=min(h_diff, len(h_ids_a) - 1),
                end_diff=h_end_a,
                other_ids=h_ids_b,
            )
            h_mask_b = h3_spans.pooling_mask(
                h_spans_b,
                args.pooling,
                first_diff=min(h_diff, len(h_ids_b) - 1),
                end_diff=h_end_b,
                other_ids=h_ids_a,
            )
            cached_hold_encs.append((h_enc_a, h_enc_b, h_mask_a, h_mask_b))
            hold_a_list.append(
                h_ref_a.to(dtype=torch.float64)[h_mask_a].mean(dim=0).to(dtype=torch.float32)
            )
            hold_b_list.append(
                h_ref_b.to(dtype=torch.float64)[h_mask_b].mean(dim=0).to(dtype=torch.float32)
            )
            if args.include_encoder:
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
                enc_hold_a_make.append(
                    h_enc_a.to(dtype=torch.float64)[hma_e].mean(dim=0).to(dtype=torch.float32)
                )
                enc_hold_b_make.append(
                    h_enc_b.to(dtype=torch.float64)[hmb_e].mean(dim=0).to(dtype=torch.float32)
                )
            hold_groups.append(h_item.group)
            hold_weights.append(float(h_item.weight))
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

    hold_specs_make = getattr(args, "hold", None)
    hold_words_make = str(getattr(args, "hold_words", "") or "").strip() or None
    hold_alpha_make = float(getattr(args, "hold_alpha", 1.0))
    extra_hold_dir = Path(args.out).parent if getattr(args, "out", None) else None
    applied_holds_make: list[str] = []
    if not args.no_holds and (hold_specs_make or hold_words_make):
        final_vec, applied_holds_make, norm_bef_make, norm_aft_make = _apply_post_refiner_holds(
            final_vec,
            family=family,
            hold_specs=hold_specs_make,
            hold_words=hold_words_make,
            hold_alpha=hold_alpha_make,
            prompt=tp_arg or None,
            extra_basis=tensors_to_save.get("holds.post_refiner"),
            extra_dir=extra_hold_dir,
        )

    tensors_to_save["vector.post_refiner"] = final_vec

    gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
    q_report = cv.quality_report(a_mat, b_mat, w_vec, generator=gen)
    raw_norm = float(torch.linalg.vector_norm(final_vec.to(dtype=torch.float64)).item())
    median_ref_row_norm = float(sum(c["median_row_norm"] for c in ref_censuses) / len(ref_censuses))
    median_enc_row_norm_ex0 = float(sum(c["median_row_norm_ex_row0"] for c in enc_censuses) / len(enc_censuses))

    points_meta: dict[str, Any] = {
        "post_refiner": {
            "unit": "mean_pair_difference",
            "raw_norm": raw_norm,
            "median_row_norm": median_ref_row_norm,
        }
    }
    if args.include_encoder:
        enc_a_mat = torch.stack(enc_a_pooled_list, dim=0)
        enc_b_mat = torch.stack(enc_b_pooled_list, dim=0)
        enc_raw_vec = cv.mean_difference(enc_a_mat, enc_b_mat, w_vec, unit_mean=use_unit_mean_make)
        if enc_hold_a_make:
            enc_basis = cv.hold_basis(
                torch.stack(enc_hold_a_make, dim=0),
                torch.stack(enc_hold_b_make, dim=0),
                hold_groups,
                torch.tensor(hold_weights, dtype=torch.float32),
            )
            enc_raw_vec = cv.project_out(enc_raw_vec, enc_basis)
        enc_raw_norm = float(torch.linalg.vector_norm(enc_raw_vec.to(dtype=torch.float64)).item())
        if enc_raw_norm <= 1e-12:
            raise ValueError("vector.encoder has zero norm")
        enc_vec = (enc_raw_vec.to(dtype=torch.float64) / enc_raw_norm).to(dtype=torch.float32)
        tensors_to_save["vector.encoder"] = enc_vec
        points_meta["encoder"] = {
            "unit": "median_row_norm",
            "raw_norm": enc_raw_norm,
            "median_row_norm": median_enc_row_norm_ex0,
            "row0_excluded": True,
        }

    if encoded_refs is not None:
        ref_hash = hashlib.sha256()
        for sp in getattr(encoded_refs, "spans", ()):
            ref_hash.update(cv._tensor_bf16_bytes(sp.rows))
        prefix_meta: Any = {"references_sha256": ref_hash.hexdigest()}
    else:
        prefix_meta = "none"

    stem = (args.output_name or pair_set.name).strip()
    if resolved_pair_path is not None:
        try:
            rel_pair_path = str(resolved_pair_path.relative_to(REPO_ROOT))
        except ValueError:
            rel_pair_path = resolved_pair_path.name
    else:
        rel_pair_path = str(pair_set.path)

    pairs_dict_make: dict[str, Any] = {
        "path": rel_pair_path,
        "sha256": pair_set.sha256,
        "n_pairs": len(pair_set.pairs),
        "n_holds": 0 if args.no_holds else (len(pair_set.holds) + len(applied_holds_make)),
        "a_side": pair_set.a_side,
        "b_side": pair_set.b_side,
        "pair_spans": pair_spans,
    }
    is_bank_toml_make = (
        resolved_pair_path is not None
        and resolved_pair_path.resolve().is_relative_to(cv_nodes.PAIR_BANK_DIR.resolve())
        and not tp_arg
    )
    if not is_bank_toml_make:
        if words_arg:
            pairs_dict_make["words_spec"] = str(words_arg)
        pairs_dict_make["pairs_ab"] = [
            {"a": p.a, "b": p.b, "weight": float(p.weight)}
            for p in pair_set.pairs
        ]
        if not args.no_holds and pair_set.holds:
            pairs_dict_make["holds_ab"] = [
                {"group": h.group, "a": h.a, "b": h.b, "weight": float(h.weight)}
                for h in pair_set.holds
            ]

    metadata: dict[str, Any] = {
        "format": cv.FORMAT_VERSION,
        "name": stem,
        "quality": pair_set.quality,
        "family": family,
        "pairs": pairs_dict_make,
        "recipe": recipe,
        "measured_under": {
            "mode": "ref2va" if encoded_refs is not None else pair_set.mode,
            "frame": pair_set.frame,
            "prefix": prefix_meta,
            "pooling": args.pooling,
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
                        "holds": applied_holds_make,
                        "alpha": hold_alpha_make,
                        "norm_before": round(norm_bef_make, 4),
                        "norm_after_raw": round(norm_aft_make, 4),
                    }
                }
                if applied_holds_make
                else {}
            ),
        },
    }

    ch_stats = cv.channel_census(final_vec)
    enc_r0_ratio = sum(c["row0_to_median_ratio"] for c in enc_censuses) / len(enc_censuses)
    ref_r0_ratio = sum(c["row0_to_median_ratio"] for c in ref_censuses) / len(ref_censuses)
    print("\n--- E2.1 Row-Norm & Channel Census ---")
    print(
        f"  encoder_rows : median_ex_row0={median_enc_row_norm_ex0:.4f}  "
        f"row0/median={enc_r0_ratio:.4f}"
    )
    print(
        f"  refiner_rows : median_tag1={median_ref_row_norm:.4f}  "
        f"row0/median={ref_r0_ratio:.4f}"
    )
    print(
        f"  channels     : top1_energy={ch_stats['top1_energy_share']:.4f}  "
        f"top5_energy={ch_stats['top5_energy_share']:.4f}  "
        f"top10_energy={ch_stats['top10_energy_share']:.4f}  "
        f"max_abs/rms={ch_stats['max_abs_over_rms']:.2f}"
    )

    jl_make_readout: dict[str, Any] | None = None
    if j_lens.bundle_available():
        try:
            jl_make_readout = j_lens.decode_vector(final_vec, space="post_refiner", family=family)
            if tp_arg:
                jl_make_readout["prompt_coupling"] = j_lens.analyze_prompt_coupling(
                    final_vec,
                    tp_arg,
                    readout=jl_make_readout,
                    family=family,
                )
            q_report["j_lens_readout"] = jl_make_readout
        except Exception:
            jl_make_readout = None

    mpc_val = q_report["mean_pairwise_cosine"]
    sh_val = q_report["split_half_cosine"]
    mpc_str = f"{mpc_val:.4f}" if mpc_val is not None else "n/a (1 pair)"
    sh_str = f"{sh_val:.4f}" if sh_val is not None else "n/a (1 pair)"
    print("\n--- Agreement & Norm Statistics ---")
    print(f"  raw_norm             : {raw_norm:.6f}")
    print(f"  mean_pairwise_cosine : {mpc_str}")
    print(f"  split_half_cosine    : {sh_str}")
    print(
        f"  per_pair_cosine      : min={min(q_report['per_pair_cosine']):.4f}  "
        f"max={max(q_report['per_pair_cosine']):.4f}"
    )
    if jl_make_readout is not None:
        print(f"  j_lens (+)           : {jl_make_readout['pos_summary']}")
        print(f"  j_lens (-)           : {jl_make_readout['neg_summary']}")

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute() and len(out_path.parts) == 1:
            out_dir = Path(_h.folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
            out_path = out_dir / out_path
    else:
        out_dir = Path(_h.folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
        out_path = out_dir / cv.default_vector_filename(
            stem,
            family,
            pooling=args.pooling,
            prefix=prefix_meta,
            apply_holds=not bool(args.no_holds),
            n_holds=len(pair_set.holds) + len(applied_holds_make),
            loras=loras,
        )

    t_enc_done = time.perf_counter()
    saved = cv.save_vector(
        out_path,
        tensors_to_save,
        metadata,
        overwrite=bool(getattr(args, "overwrite", False)),
    )
    elapsed = time.perf_counter() - t0
    print(
        f"\n  timings     : encoder_load={t_clip - t0:.2f}s  refiner_load={t_ref - t_clip:.2f}s  "
        f"encode_and_refine={t_enc_done - t_ref:.2f}s  total={elapsed:.2f}s"
    )
    print(f"  saved       : {saved.name} ({saved.stat().st_size} bytes, {elapsed:.2f}s)")

    if (
        not args.out
        and not getattr(args, "dit", None)
        and not getattr(args, "unet", None)
        and not explicit_loras
    ):
        for sib_fam in req_fams:
            if sib_fam == family or sib_fam not in ("fl2va", "ref2va"):
                continue
            sib_dit_name = (
                h3_config.MODELS["unet_ref2va"]
                if sib_fam == "ref2va"
                else h3_config.MODELS["unet_fl2va"]
            )
            try:
                sib_dit_file = _h.folder_paths.get_full_path("diffusion_models", sib_dit_name)
            except Exception:
                sib_dit_file = None
            if not sib_dit_file or not Path(sib_dit_file).is_file():
                continue
            sib_dit_path = Path(sib_dit_file)
            sib_loras = _resolve_make_loras(sib_fam)
            sib_key = (
                str(sib_dit_path),
                str(args.refiner_device),
                tuple((l["file"], float(l["strength"])) for l in sib_loras),
            )
            if isinstance(ref_cache, dict) and sib_key in ref_cache:
                sib_ref_run = ref_cache[sib_key]
            else:
                sib_ref_run = cv_nodes.build_refiner_runner(
                    sib_dit_path, device=args.refiner_device, loras=sib_loras
                )
                if isinstance(ref_cache, dict):
                    ref_cache[sib_key] = sib_ref_run
            sib_recipe = cv_nodes.extract_model_recipe(
                None,
                None,
                dit_path_override=sib_dit_path,
                encoder_path_override=encoder_path,
                loras_override=sib_loras,
            )
            sib_a_list: list[torch.Tensor] = []
            sib_b_list: list[torch.Tensor] = []
            sib_ref_cens: list[dict[str, float]] = []
            for enc_a, enc_b, tags_a, tags_b, mask_a, mask_b in cached_pair_encs:
                s_ref_a = sib_ref_run(enc_a)
                s_ref_b = sib_ref_run(enc_b)
                sib_ref_cens.append(cv.row_norm_census(s_ref_a, tags_a))
                sib_ref_cens.append(cv.row_norm_census(s_ref_b, tags_b))
                sib_a_list.append(
                    s_ref_a.to(dtype=torch.float64)[mask_a].mean(dim=0).to(dtype=torch.float32)
                )
                sib_b_list.append(
                    s_ref_b.to(dtype=torch.float64)[mask_b].mean(dim=0).to(dtype=torch.float32)
                )
            sib_a_mat = torch.stack(sib_a_list, dim=0)
            sib_b_mat = torch.stack(sib_b_list, dim=0)
            sib_raw_vec = cv.mean_difference(
                sib_a_mat, sib_b_mat, w_vec, unit_mean=use_unit_mean_make
            )
            sib_tensors: dict[str, torch.Tensor] = {}
            if cached_hold_encs:
                sib_ha: list[torch.Tensor] = []
                sib_hb: list[torch.Tensor] = []
                for h_enc_a, h_enc_b, h_mask_a, h_mask_b in cached_hold_encs:
                    sh_a = sib_ref_run(h_enc_a)
                    sh_b = sib_ref_run(h_enc_b)
                    sib_ha.append(
                        sh_a.to(dtype=torch.float64)[h_mask_a].mean(dim=0).to(dtype=torch.float32)
                    )
                    sib_hb.append(
                        sh_b.to(dtype=torch.float64)[h_mask_b].mean(dim=0).to(dtype=torch.float32)
                    )
                sib_basis = cv.hold_basis(
                    torch.stack(sib_ha, dim=0),
                    torch.stack(sib_hb, dim=0),
                    hold_groups,
                    torch.tensor(hold_weights, dtype=torch.float32),
                )
                sib_final_vec = cv.project_out(sib_raw_vec, sib_basis)
                sib_tensors["holds.post_refiner"] = sib_basis
            else:
                sib_final_vec = sib_raw_vec
            sib_applied_holds: list[str] = []
            sib_nb = 0.0
            sib_na = 0.0
            if not args.no_holds and (hold_specs_make or hold_words_make):
                sib_final_vec, sib_applied_holds, sib_nb, sib_na = _apply_post_refiner_holds(
                    sib_final_vec,
                    family=sib_fam,
                    hold_specs=hold_specs_make,
                    hold_words=hold_words_make,
                    hold_alpha=hold_alpha_make,
                    prompt=tp_arg or None,
                    extra_basis=sib_tensors.get("holds.post_refiner"),
                    extra_dir=out_dir,
                )
            sib_tensors["vector.post_refiner"] = sib_final_vec
            if "vector.encoder" in tensors_to_save:
                sib_tensors["vector.encoder"] = tensors_to_save["vector.encoder"]
            sib_gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
            sib_qrep = cv.quality_report(sib_a_mat, sib_b_mat, w_vec, generator=sib_gen)
            if j_lens.bundle_available():
                try:
                    sib_qrep["j_lens_readout"] = j_lens.decode_vector(
                        sib_final_vec, space="post_refiner", family=sib_fam
                    )
                except Exception:
                    pass
            sib_raw_norm = float(torch.linalg.vector_norm(sib_final_vec.to(dtype=torch.float64)).item())
            sib_med_ref = float(sum(c["median_row_norm"] for c in sib_ref_cens) / len(sib_ref_cens))
            sib_points: dict[str, Any] = {
                "post_refiner": {
                    "unit": "mean_pair_difference",
                    "raw_norm": sib_raw_norm,
                    "median_row_norm": sib_med_ref,
                }
            }
            if "encoder" in points_meta:
                sib_points["encoder"] = dict(points_meta["encoder"])
            sib_prov = {
                k: v for k, v in metadata["provenance"].items() if k != "applied_holds"
            }
            if sib_applied_holds:
                sib_prov["applied_holds"] = {
                    "holds": sib_applied_holds,
                    "alpha": hold_alpha_make,
                    "norm_before": round(sib_nb, 4),
                    "norm_after_raw": round(sib_na, 4),
                }
            sib_meta = {
                **metadata,
                "family": sib_fam,
                "pairs": {
                    **pairs_dict_make,
                    "n_holds": 0
                    if args.no_holds
                    else (len(pair_set.holds) + len(sib_applied_holds)),
                },
                "recipe": sib_recipe,
                "points": sib_points,
                "quality_report": sib_qrep,
                "provenance": sib_prov,
            }
            sib_out = out_dir / cv.default_vector_filename(
                stem,
                sib_fam,
                pooling=args.pooling,
                prefix=prefix_meta,
                apply_holds=not bool(args.no_holds),
                n_holds=len(pair_set.holds) + len(sib_applied_holds),
                loras=sib_loras,
            )
            sib_saved = cv.save_vector(
                sib_out,
                sib_tensors,
                sib_meta,
                overwrite=bool(getattr(args, "overwrite", False)),
            )
            print(f"  saved       : {sib_saved.name} ({sib_saved.stat().st_size} bytes, sibling={sib_fam})")
    return 0



def cmd_from_md(args: argparse.Namespace) -> int:
    md_path = Path(args.md_path)
    if not md_path.is_file():
        raise ValueError(f"markdown file not found: {md_path}")

    out_target: str | Path = (
        Path(args.out) if getattr(args, "out", None) else cv_nodes.USER_PAIR_BANK_DIR
    )
    repaired_md = _repair_multichange_markdown_pairs(md_path.read_text(encoding="utf-8"))
    parsed = cv.parse_markdown_pairs(repaired_md)
    saved_toml = cv.write_markdown_to_toml(
        repaired_md,
        out_target,
        name=getattr(args, "name", None) or md_path.stem,
        mode=getattr(args, "mode", None),
        frame=getattr(args, "frame", "scene") or "scene",
        quality=getattr(args, "quality", None),
        a_side=getattr(args, "a_side", None),
        b_side=getattr(args, "b_side", None),
        test_prompt_out=getattr(args, "test_prompt_out", None),
        overwrite=bool(getattr(args, "overwrite", False)),
        bank_dir=cv_nodes.PROMPT_BANK_DIR,
    )
    pair_set = cv.load_pairs(saved_toml, bank_dir=cv_nodes.PROMPT_BANK_DIR)
    test_prompt = parsed["test_scene_prompt"]
    has_test_prompt = bool(test_prompt)

    print(f"=== Control Vector From-MD: {pair_set.name} ===")
    print(f"  md_file           : {_display_path(md_path)}")
    print(f"  saved_toml        : {_display_path(saved_toml)}")
    print(
        f"  name              : {pair_set.name}   mode: {pair_set.mode}   "
        f"frame: {pair_set.frame}   n_pairs: {len(pair_set.pairs)}"
    )
    if has_test_prompt:
        tp_msg = f"yes ({len(test_prompt.split())} words, {len(test_prompt)} chars)"
        if getattr(args, "test_prompt_out", None):
            tp_msg += f" -> {_display_path(Path(args.test_prompt_out))}"
    else:
        tp_msg = "no"
    print(f"  test_scene_prompt : {tp_msg}")
    print(f"  quality           : {pair_set.quality}")
    print(f"  a_side            : {pair_set.a_side}")
    print(f"  b_side            : {pair_set.b_side}")
    print(f"\n--- Pair Spans ({len(pair_set.pairs)} pairs) ---")
    for item in pair_set.pairs:
        sec_idx, sec_name = cv.differing_section_info(item.a, item.b, frame=pair_set.frame)
        wa = len(item.a_span.split())
        wb = len(item.b_span.split())
        print(
            f"  [{item.index:02d}] [sec {sec_idx}:{sec_name}] ({wa:2d}w -> {wb:2d}w) | "
            f"-{item.a_span[:48]!r} -> +{item.b_span[:48]!r}"
        )
    return 0

