#!/usr/bin/env python3
"""Build, inspect, meter, and grade MiniMax H3 control vectors.

What it checks: builds `.safetensors` control vectors from contrastive prompt
pair files (`prompt_bank/control_vectors/*.toml`), inspects saved vector files
and compatibility recipes, meters bank prompts along a control vector without
rendering (E2.8), and grades activation/prediction movement against `flip_null`,
`random_null`, the `" ."` period edit, the prompt-edit baseline, and a
length-matched padding control (E2.1-E2.7).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
if __name__ == "__main__":
    import importlib.util as _ilu
    import os as _os

    if _ilu.find_spec("torch") is None:
        _venv_py = REPO_ROOT.parent.parent / ".venv" / "bin" / "python"
        if _venv_py.is_file() and Path(sys.executable).resolve() != _venv_py.resolve():
            _os.execv(str(_venv_py), [str(_venv_py), *sys.argv])

# Ensure REPO_ROOT precedes BENCH_DIR on sys.path so `import control_vectors`
# resolves to the root library module (`REPO_ROOT/control_vectors.py`), while
# keeping BENCH_DIR on sys.path so `bench/_lib/__init__.py` can import `_lib.*`.
sys.path = [p for p in sys.path if Path(p or ".").resolve() not in (REPO_ROOT, BENCH_DIR)]
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(REPO_ROOT))

from bench._lib import bootstrap  # noqa: E402

bootstrap()

import folder_paths  # noqa: E402
import torch  # noqa: E402

import control_vector_nodes as cv_nodes  # noqa: E402
import control_vectors as cv  # noqa: E402
import h3_spans  # noqa: E402
import j_lens  # noqa: E402
import substrate  # noqa: E402
from workflows import h3_config  # noqa: E402

LOOK_FAMILY_IDS: tuple[str, ...] = (
    "t2va_look_anchor",
    "t2va_look_faded_16mm",
    "t2va_look_noir",
    "t2va_look_golden_hour",
    "t2va_look_bleach_bypass",
    "t2va_look_neon",
    "t2va_look_pastel_ove",
    "t2va_look_sodium_rain",
    "t2va_look_sepia_plate",
)


def _resolve_prompt_text(arg: str) -> tuple[str, str]:
    """Return `(label, prompt_text)` from a prompt-bank id or file path."""
    p = Path(arg)
    if p.is_file():
        return p.stem, p.read_text(encoding="utf-8").strip()
    bank_txt = REPO_ROOT / "prompt_bank" / (
        arg if arg.endswith(".txt") else f"{arg}.txt"
    )
    if bank_txt.is_file():
        return bank_txt.stem, bank_txt.read_text(encoding="utf-8").strip()
    raise ValueError(f"prompt {arg!r} not found as a file or under prompt_bank/")


def _load_clip(encoder_name: str, device: str = "default") -> Any:
    enc_loader_mod = cv_nodes._repo("h3_encoder_loader")

    (clip,) = enc_loader_mod.MiniMaxH3EncoderLoader.execute(
        encoder_name=encoder_name, device=device
    )
    return clip


def _parse_lora_specs(specs: list[str] | None) -> list[dict[str, Any]]:
    if not specs:
        return []
    out: list[dict[str, Any]] = []
    for item in specs:
        if ":" in item:
            fname, s_str = item.rsplit(":", 1)
            strength = float(s_str)
        else:
            fname, strength = item, 1.0
        lpath = cv_nodes.resolve_lora_file_path(fname)
        out.append(
            {
                "file": Path(lpath).name,
                "sha256": cv.safetensors_recipe_digest(lpath),
                "strength": strength,
                "applied": "branch",
            }
        )
    return out


def _load_encoded_references(
    clip: Any,
    vae_name: str,
    ref_images: list[str],
    *,
    width: int = 1152,
    height: int = 768,
    length: int = h3_config.LENGTH,
) -> Any:
    """Build `H3EncodedReferences` from `--ref-images` so E2.4 can compare vectors
    made with and without a ref2va reference prefix."""
    from PIL import Image
    import numpy as np
    import comfy.sd
    import comfy.utils

    ref_cond_mod = cv_nodes._repo("reference_conditioning")
    ref_enc_mod = cv_nodes._repo("reference_encode")

    vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
    sd = comfy.utils.load_torch_file(vae_path)
    vae = comfy.sd.VAE(sd=sd)
    refs = None
    for img_spec in ref_images:
        img_path = Path(img_spec)
        if not img_path.is_file():
            img_path = Path(folder_paths.get_input_directory()) / img_spec
        if not img_path.is_file():
            raise ValueError(f"reference image {img_spec!r} not found")
        pil = Image.open(img_path).convert("RGB")
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        (refs,) = ref_cond_mod.MiniMaxH3AppendRefImage.execute(
            image=tensor,
            size_policy={"size_policy": "max", **h3_config.REF_IMAGE_SIZE},
            qwen_view=dict(h3_config.REF_QWEN_VIEW),
            references=refs,
            use_vae=True,
        )
    (encoded_refs,) = ref_enc_mod.MiniMaxH3EncodeReferences.execute(
        clip=clip,
        vae=vae,
        references=refs,
        width=width,
        height=height,
        length=length,
    )
    return encoded_refs


def cmd_make(args: argparse.Namespace) -> int:
    if args.pooling not in cv.VALID_POOLING:
        raise ValueError(f"pooling must be one of {cv.VALID_POOLING}, got {args.pooling!r}")

    resolved_pair_path = cv_nodes.resolve_pair_file_path(args.pairs)
    pair_set = cv.load_pairs(resolved_pair_path, bank_dir=cv_nodes.PROMPT_BANK_DIR)

    unet_name = args.unet or (
        h3_config.MODELS["unet_ref2va"]
        if (args.ref_images or pair_set.mode == "ref2va")
        else h3_config.MODELS["unet_fl2va"]
    )
    unet_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", unet_name))
    encoder_name = args.encoder or h3_config.MODELS["clip"]
    encoder_path = Path(folder_paths.get_full_path_or_raise("text_encoders", encoder_name))
    loras = _parse_lora_specs(args.lora)

    family = cv.detect_family(unet_path)
    recipe = cv_nodes.extract_model_recipe(
        None,
        None,
        unet_path_override=unet_path,
        encoder_path_override=encoder_path,
        loras_override=loras,
    )

    t0 = time.perf_counter()
    clip = getattr(args, "_clip_override", None) or _load_clip(encoder_name, device=args.device)
    t_clip = time.perf_counter()
    ref_cache = getattr(args, "_refiner_cache", None)
    ref_key = (
        str(unet_path),
        str(args.refiner_device),
        tuple((l["file"], float(l["strength"])) for l in loras),
    )
    if isinstance(ref_cache, dict) and ref_key in ref_cache:
        refiner_run = ref_cache[ref_key]
    else:
        refiner_run = cv_nodes.build_refiner_runner(
            unet_path, device=args.refiner_device, loras=loras
        )
        if isinstance(ref_cache, dict):
            ref_cache[ref_key] = refiner_run
    t_ref = time.perf_counter()

    encoded_refs = None
    if args.ref_images:
        imgs = [s.strip() for s in args.ref_images.split(",") if s.strip()]
        encoded_refs = _load_encoded_references(
            clip, h3_config.MODELS["video_vae"], imgs
        )

    print(f"=== Control Vector Make: {pair_set.name} ===")
    print(f"  pairs_file  : {resolved_pair_path.relative_to(REPO_ROOT) if resolved_pair_path.is_relative_to(REPO_ROOT) else resolved_pair_path}")
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
    weights_list: list[float] = []
    enc_censuses: list[dict[str, float]] = []
    ref_censuses: list[dict[str, float]] = []
    pair_spans: list[dict[str, Any]] = []

    print(f"\n--- Pair Spans ({len(pair_set.pairs)} pairs) ---")
    for idx, item in enumerate(pair_set.pairs):
        enc_a, tags_a, ids_a, dec_a = cv_nodes.encode_prompt_for_vector(
            clip, item.a, encoded_references=encoded_refs
        )
        enc_b, tags_b, ids_b, dec_b = cv_nodes.encode_prompt_for_vector(
            clip, item.b, encoded_references=encoded_refs
        )
        ref_a = refiner_run(enc_a)
        ref_b = refiner_run(enc_b)

        enc_censuses.append(cv.row_norm_census(enc_a, tags_a))
        enc_censuses.append(cv.row_norm_census(enc_b, tags_b))
        ref_censuses.append(cv.row_norm_census(ref_a, tags_a))
        ref_censuses.append(cv.row_norm_census(ref_b, tags_b))

        first_diff, end_a, end_b = h3_spans.differing_token_bounds(ids_a, ids_b)
        spans_a = h3_spans.spans(ids_a, tags_a.tolist(), dec_a)
        spans_b = h3_spans.spans(ids_b, tags_b.tolist(), dec_b)
        mask_a = h3_spans.pooling_mask(
            spans_a,
            args.pooling,
            first_diff=min(first_diff, len(ids_a) - 1),
            end_diff=end_a,
        )
        mask_b = h3_spans.pooling_mask(
            spans_b,
            args.pooling,
            first_diff=min(first_diff, len(ids_b) - 1),
            end_diff=end_b,
        )

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
        weights_list.append(float(item.weight))
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

    raw_vec = cv.mean_difference(
        a_mat, b_mat, w_vec, unit_mean=bool(getattr(args, "unit_mean", False))
    )
    tensors_to_save: dict[str, torch.Tensor] = {}

    if not args.no_holds and pair_set.holds:
        hold_a_list: list[torch.Tensor] = []
        hold_b_list: list[torch.Tensor] = []
        hold_groups: list[str] = []
        hold_weights: list[float] = []
        for h_item in pair_set.holds:
            h_enc_a, h_tags_a, h_ids_a, h_dec_a = cv_nodes.encode_prompt_for_vector(
                clip, h_item.a, encoded_references=encoded_refs
            )
            h_enc_b, h_tags_b, h_ids_b, h_dec_b = cv_nodes.encode_prompt_for_vector(
                clip, h_item.b, encoded_references=encoded_refs
            )
            h_ref_a = refiner_run(h_enc_a)
            h_ref_b = refiner_run(h_enc_b)
            h_diff, h_end_a, h_end_b = h3_spans.differing_token_bounds(h_ids_a, h_ids_b)
            h_spans_a = h3_spans.spans(h_ids_a, h_tags_a.tolist(), h_dec_a)
            h_spans_b = h3_spans.spans(h_ids_b, h_tags_b.tolist(), h_dec_b)
            h_mask_a = h3_spans.pooling_mask(
                h_spans_a,
                args.pooling,
                first_diff=min(h_diff, len(h_ids_a) - 1),
                end_diff=h_end_a,
            )
            h_mask_b = h3_spans.pooling_mask(
                h_spans_b,
                args.pooling,
                first_diff=min(h_diff, len(h_ids_b) - 1),
                end_diff=h_end_b,
            )
            hold_a_list.append(
                h_ref_a.to(dtype=torch.float64)[h_mask_a].mean(dim=0).to(dtype=torch.float32)
            )
            hold_b_list.append(
                h_ref_b.to(dtype=torch.float64)[h_mask_b].mean(dim=0).to(dtype=torch.float32)
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
        enc_raw_vec = cv.mean_difference(enc_a_mat, enc_b_mat, w_vec)
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
    try:
        rel_pair_path = str(resolved_pair_path.relative_to(REPO_ROOT))
    except ValueError:
        rel_pair_path = resolved_pair_path.name

    metadata: dict[str, Any] = {
        "format": cv.FORMAT_VERSION,
        "name": stem,
        "quality": pair_set.quality,
        "family": family,
        "pairs": {
            "path": rel_pair_path,
            "sha256": pair_set.sha256,
            "n_pairs": len(pair_set.pairs),
            "n_holds": 0 if args.no_holds else len(pair_set.holds),
            "a_side": pair_set.a_side,
            "b_side": pair_set.b_side,
            "pair_spans": pair_spans,
        },
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
            q_report["j_lens_readout"] = jl_make_readout
        except Exception:
            jl_make_readout = None

    print("\n--- Agreement & Norm Statistics ---")
    print(f"  raw_norm             : {raw_norm:.6f}")
    print(f"  mean_pairwise_cosine : {q_report['mean_pairwise_cosine']:.4f}")
    print(f"  split_half_cosine    : {q_report['split_half_cosine']:.4f}")
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
            out_dir = Path(folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
            out_path = out_dir / out_path
    else:
        out_dir = Path(folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
        out_path = out_dir / cv.default_vector_filename(
            stem,
            family,
            pooling=args.pooling,
            prefix=prefix_meta,
            apply_holds=not bool(args.no_holds),
            n_holds=len(pair_set.holds),
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
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    vpath = cv_nodes.resolve_vector_file_path(args.vector)
    tensors, metadata = cv.load_vector(vpath)
    vec = (
        tensors["vector.post_refiner"]
        if "vector.post_refiner" in tensors
        else next(iter(tensors.values()))
    )
    ch_stats = cv.channel_census(vec)

    print(f"=== Control Vector Inspect: {metadata['name']} ===")
    print(f"  file            : {vpath.name}")
    print(f"  format          : {metadata['format']}")
    print(f"  quality         : {metadata['quality']}")
    print(f"  family          : {metadata['family']}")
    print(f"  tensors         : { {k: list(t.shape) for k, t in tensors.items()} }")
    print(f"  pairs           : {json.dumps(metadata['pairs'], sort_keys=True)}")
    print(f"  recipe          : {json.dumps(metadata['recipe'], sort_keys=True)}")
    print(f"  measured_under  : {json.dumps(metadata['measured_under'], sort_keys=True)}")
    print(f"  points          : {json.dumps(metadata['points'], sort_keys=True)}")
    print(f"  quality_report  : {json.dumps(metadata['quality_report'], sort_keys=True)}")
    print(
        f"  channel_census  : top1={ch_stats['top1_energy_share']:.4f} "
        f"top5={ch_stats['top5_energy_share']:.4f} "
        f"top10={ch_stats['top10_energy_share']:.4f} "
        f"max_abs/rms={ch_stats['max_abs_over_rms']:.2f}"
    )

    if j_lens.bundle_available() and "vector.post_refiner" in tensors:
        hold_paths = [cv_nodes.resolve_vector_file_path(h) for h in (getattr(args, "hold", None) or ())]
        hold_alpha = float(getattr(args, "hold_alpha", 1.0))
        jl_rep = j_lens.decode_safetensors(
            vpath,
            holds=hold_paths if hold_paths else None,
            alpha=hold_alpha,
        )
        pr_jl = jl_rep["post_refiner"]
        if hold_paths:
            print(
                f"  j_lens_holds    : {[p.name for p in hold_paths]} (alpha={hold_alpha:.2f}, "
                f"norm {pr_jl['norm_before_holds']:.2f} -> {pr_jl['norm_after_holds']:.2f})"
            )
        print(f"  j_lens (PR +)   : {pr_jl['pos_summary']}")
        print(f"  j_lens (PR -)   : {pr_jl['neg_summary']}")
        if "encoder" in jl_rep:
            enc_jl = jl_rep["encoder"]
            print(f"  j_lens (EN +)   : {enc_jl['pos_summary']}")
            print(f"  j_lens (EN -)   : {enc_jl['neg_summary']}")
            print(
                f"  j_lens cosines  : cos(J_ref v_enc, v_PR)={jl_rep['refiner_forward_cosine']:.4f}  "
                f"cos_J63(Wiener)={jl_rep['wiener_j63_cosine']:.4f}"
            )

    if args.check_unet:
        unet_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", args.check_unet))
        fam = cv.detect_family(unet_path)
        enc_name = args.check_encoder or metadata["recipe"]["encoder"]["file"]
        enc_path = (
            Path(folder_paths.get_full_path_or_raise("text_encoders", enc_name))
            if enc_name != "unspecified"
            else None
        )
        runtime_recipe = cv_nodes.extract_model_recipe(
            None,
            None,
            unet_path_override=unet_path,
            encoder_path_override=enc_path,
            loras_override=_parse_lora_specs(args.check_lora),
        )
        warnings = cv.check_vector_compatibility(
            metadata,
            loaded_family=fam,
            loaded_recipe=runtime_recipe,
            allow_recipe_mismatch=args.allow_recipe_mismatch,
        )
        print(f"  compatibility   : ok (warnings={warnings})")

    meter_targets = list(args.meter or [])
    if args.meter_look_family:
        for pid in LOOK_FAMILY_IDS:
            if pid not in meter_targets:
                meter_targets.append(pid)

    if meter_targets:
        enc_name = args.check_encoder or metadata["recipe"]["encoder"]["file"]
        if enc_name == "unspecified":
            enc_name = h3_config.MODELS["clip"]
        clip = _load_clip(enc_name, device=args.device)
        unet_name = args.check_unet or metadata["recipe"]["checkpoint"]["file"]
        unet_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", unet_name))
        meter_loras = (
            _parse_lora_specs(args.check_lora)
            if args.check_lora
            else list(metadata["recipe"].get("loras", ()))
        )
        refiner_run = cv_nodes.build_refiner_runner(
            unet_path, device=args.refiner_device, loras=meter_loras
        )
        pooling = metadata["measured_under"].get("pooling", "scene_text")
        if pooling in ("from_first_diff", "diff_span"):
            pooling = "scene_text"

        print(f"\n--- Meter Projection along {metadata['name']} (pooling={pooling}) ---")
        rows: list[tuple[str, float, float, float, int]] = []
        for target in meter_targets:
            label, text = _resolve_prompt_text(target)
            enc_rows, tags, tids, dec_fn = cv_nodes.encode_prompt_for_vector(clip, text)
            ref_rows = refiner_run(enc_rows)
            sp = h3_spans.spans(tids, tags.tolist(), dec_fn)
            mask = h3_spans.pooling_mask(sp, pooling)
            m = cv.meter_score(ref_rows, vec, mask=mask)
            rows.append(
                (label, m["projection_units"], m["signed_projection"], m["cosine"], int(mask.sum().item()))
            )
        anchor_units = rows[0][1] if rows else 0.0
        for label, p_units, s_proj, cos_val, n_tok in rows:
            delta_u = p_units - anchor_units
            print(
                f"  {label:<36s} units={p_units:+8.4f} (delta={delta_u:+8.4f})  "
                f"proj={s_proj:+8.4f}  cos={cos_val:+7.4f}  tokens={n_tok:3d}"
            )
    return 0


def _prompt_token_count(clip: Any, text: str) -> int:
    clean = cv_nodes._rules.normalize_prompt(text)
    entries = cv_nodes._extract_token_entries(clip.tokenize(clean))
    return len(entries)


def _build_padding_prompt(base_text: str, target_extra_tokens: int, clip: Any) -> str:
    """Append neutral filler clauses to `base_text` so its token length matches
    the edit prompt's token length L (sizing the length-induced RoPE shift)."""
    if target_extra_tokens <= 0:
        return base_text
    base_len = _prompt_token_count(clip, base_text)
    target_len = base_len + target_extra_tokens
    marker = "\n\noverall_soundscape:"
    filler_unit = " The scene continues in steady, unbroken real time across the frame."
    pad = ""
    cur_text = base_text

    def _inject(p_str: str) -> str:
        return (
            base_text.replace(marker, p_str + marker, 1)
            if marker in base_text
            else base_text + p_str
        )

    for _ in range(64):
        if _prompt_token_count(clip, cur_text) >= target_len:
            break
        pad += filler_unit
        cur_text = _inject(pad)

    words = pad.split(" ")
    while len(words) > 1:
        cand_pad = " ".join(words[:-1])
        cand_text = _inject(cand_pad)
        if _prompt_token_count(clip, cand_text) >= target_len:
            words.pop()
            cur_text = cand_text
        else:
            # If trimming one word drops below target_len (e.g. a 2-token word),
            # try adding a single-token word so the length matches target_len.
            for one_tok in (" now", " here", " still", "."):
                exact_cand = _inject(cand_pad + one_tok)
                if _prompt_token_count(clip, exact_cand) == target_len:
                    return exact_cand
            break
    return cur_text


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.to(dtype=torch.float64) - b.to(dtype=torch.float64)).reshape(-1)
    base = b.to(dtype=torch.float64).reshape(-1)
    denom = float(torch.linalg.vector_norm(base).item())
    if denom <= 1e-12:
        return 0.0
    return float(torch.linalg.vector_norm(diff).item() / denom)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.detach().to(dtype=torch.float64).reshape(-1)
    b64 = b.detach().to(dtype=torch.float64).reshape(-1)
    na = float(torch.linalg.vector_norm(a64).item())
    nb = float(torch.linalg.vector_norm(b64).item())
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(torch.dot(a64, b64).item() / (na * nb))


def _sigma_shifted(model: Any) -> Any:
    """Apply `MiniMaxH3SigmaShift` parameters from `h3_config.SIGMA_SHIFT`."""
    import comfy.model_sampling

    m = model.clone()

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    original = m.get_model_object("model_sampling")
    sampling = ModelSamplingAdvanced(model.model.model_config)
    sampling.set_parameters(
        shift=h3_config.SIGMA_SHIFT["shift_video"],
        audio_shift=h3_config.SIGMA_SHIFT["shift_audio"],
    )
    if hasattr(original, "noise_scale"):
        sampling.set_noise_scale(original.noise_scale)
    m.add_object_patch("model_sampling", sampling)
    to = m.model_options["transformer_options"] = m.model_options.get(
        "transformer_options", {}
    ).copy()
    to["minimax_h3_sigma_shift_video"] = h3_config.SIGMA_SHIFT["shift_video"]
    to["minimax_h3_sigma_shift_audio"] = h3_config.SIGMA_SHIFT["shift_audio"]
    return m


def _one_forward(
    model: Any,
    cond: list[Any],
    noise: Any,
    latent: Any,
    sigmas: torch.Tensor,
    step: int,
    seed: int,
) -> Any:
    """Evaluate the DiT's denoised prediction at schedule index `step` through
    `Guider_Basic` + `KSAMPLER` (`bench/measure_marker_epsilon.py::one_forward`)."""
    import comfy.samplers
    from comfy_extras.nodes_custom_sampler import Guider_Basic

    captured: dict[str, Any] = {}

    def _single(model_fn, x, sigs, extra_args, callback=None, disable=None):
        sigma = sigs[0] * torch.ones((x.shape[0],), device=x.device, dtype=x.dtype)
        out = model_fn(x, sigma, **extra_args)
        captured["out"] = out
        return out

    guider = Guider_Basic(model)
    guider.set_conds(cond)
    result = guider.sample(
        noise,
        latent,
        comfy.samplers.KSAMPLER(_single),
        sigmas[step : step + 2],
        denoise_mask=None,
        callback=None,
        disable_pbar=True,
        seed=seed,
    )
    if "out" not in captured:
        raise RuntimeError("KSAMPLER never evaluated the model during grade forward")
    return result


def _attach_raw_vector(
    model: Any,
    name: str,
    vec: torch.Tensor,
    strength: float,
    *,
    mask: str = "non_vision",
    mode: str = "add",
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Clone `model` with an in-memory `(vec, strength, mask, mode, start_percent, end_percent)` attached via
    `control_vector_nodes._make_control_vector_wrapper` on `WrappersMP.DIFFUSION_MODEL`."""
    import comfy.patcher_extension as pe

    m = model.clone()
    spec = cv_nodes._AppliedVectorSpec(
        vector_name=name,
        vector=vec.detach().to(device="cpu", dtype=torch.float32),
        strength=float(strength),
        mask=mask,
        mode=mode,
        start_percent=float(start_percent),
        end_percent=float(end_percent),
        metadata=metadata or {"measured_under": {"prefix": "none"}},
        advisories=(),
    )
    t_opts = m.model_options.setdefault("transformer_options", {})
    t_opts[cv_nodes.WRAPPER_KEY] = (spec,)
    wrappers = m.wrappers.setdefault(pe.WrappersMP.DIFFUSION_MODEL, {})
    wrappers[cv_nodes.WRAPPER_KEY] = [cv_nodes._make_control_vector_wrapper((spec,))]
    return m


def _pred_blocks(pred: Any) -> dict[str, torch.Tensor]:
    """Unpack a DiT prediction into `{'video': Tensor, 'audio': Tensor}` flat float64 CPU vectors."""
    if isinstance(pred, dict) and "video" in pred and "audio" in pred:
        return {
            "video": pred["video"].detach().to("cpu", torch.float64).flatten(),
            "audio": pred["audio"].detach().to("cpu", torch.float64).flatten(),
        }
    parts = pred.unbind() if getattr(pred, "is_nested", False) else (pred,)
    if len(parts) >= 2:
        return {
            "video": parts[0].detach().to("cpu", torch.float64).flatten(),
            "audio": parts[1].detach().to("cpu", torch.float64).flatten(),
        }
    x = parts[0].detach().to("cpu", torch.float64).flatten()
    return {"video": x, "audio": x}


def _block_metrics(
    arm_blocks: dict[str, torch.Tensor],
    base_blocks: dict[str, torch.Tensor],
    edit_blocks: dict[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-block (`video`, `audio`) `relative_l2`, `cosine_with_base`,
    `max_abs_diff`, and (when `edit_blocks` is given) `delta_cosine_with_edit`."""
    out: dict[str, dict[str, float]] = {}
    for bname in ("video", "audio"):
        arm_x = arm_blocks[bname]
        base_x = base_blocks[bname]
        diff = arm_x - base_x
        base_norm = float(torch.linalg.vector_norm(base_x).item())
        rel_l2 = float(torch.linalg.vector_norm(diff).item() / base_norm) if base_norm > 1e-12 else 0.0
        entry: dict[str, float] = {
            "relative_l2": rel_l2,
            "cosine_with_base": _cosine(arm_x, base_x),
            "max_abs_diff": float(diff.abs().max().item()) if diff.numel() > 0 else 0.0,
        }
        if edit_blocks is not None:
            edit_diff = edit_blocks[bname] - base_x
            entry["delta_cosine_with_edit"] = _cosine(diff, edit_diff)
        out[bname] = entry
    return out


def cmd_grade(args: argparse.Namespace) -> int:
    vpath = cv_nodes.resolve_vector_file_path(args.vector)
    tensors, metadata = cv.load_vector(vpath)
    vec = tensors["vector.post_refiner"]
    prompt_label, prompt_text = _resolve_prompt_text(args.prompt)

    enc_name = args.encoder or metadata["recipe"]["encoder"]["file"]
    if enc_name == "unspecified":
        enc_name = h3_config.MODELS["clip"]
    unet_name = args.unet or metadata["recipe"]["checkpoint"]["file"]
    unet_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", unet_name))
    if getattr(args, "no_lora", False):
        grade_loras: list[dict[str, Any]] = []
    elif getattr(args, "lora", None):
        grade_loras = _parse_lora_specs(args.lora)
    else:
        grade_loras = list(metadata["recipe"].get("loras", ()))

    t_grade_start = time.perf_counter()
    clip = _load_clip(enc_name, device=args.device)
    t_clip_loaded = time.perf_counter()
    refiner_run = cv_nodes.build_refiner_runner(
        unet_path, device=args.refiner_device, loras=grade_loras
    )
    t_ref_loaded = time.perf_counter()

    encoded_refs = None
    if args.ref_images:
        imgs = [s.strip() for s in args.ref_images.split(",") if s.strip()]
        encoded_refs = _load_encoded_references(
            clip, h3_config.MODELS["video_vae"], imgs
        )

    base_cond, base_enc, base_tags, base_ids, _ = cv_nodes.encode_conditioning_for_vector(
        clip, prompt_text, encoded_references=encoded_refs
    )
    base_ctx = refiner_run(base_enc)
    base_ctx_b = base_ctx.unsqueeze(0)
    base_tags_b = base_tags.unsqueeze(0)

    period_cond, period_enc, period_tags, period_ids, _ = cv_nodes.encode_conditioning_for_vector(
        clip, prompt_text + " .", encoded_references=encoded_refs
    )
    period_ctx = refiner_run(period_enc)

    raw_norm = float(torch.linalg.vector_norm(vec.to(dtype=torch.float64)).item())
    gen_rand = torch.Generator(device="cpu").manual_seed(int(args.seed))
    rand_dir = cv.random_null(raw_norm, int(vec.shape[0]), generator=gen_rand)
    flip_dir: torch.Tensor | None = None
    if args.pairs:
        pair_set = cv.load_pairs(
            cv_nodes.resolve_pair_file_path(args.pairs), bank_dir=cv_nodes.PROMPT_BANK_DIR
        )
        pooling = metadata["measured_under"].get("pooling", "scene_text")
        p_a: list[torch.Tensor] = []
        p_b: list[torch.Tensor] = []
        p_w: list[float] = []
        for item in pair_set.pairs:
            ea, ta, ia, da = cv_nodes.encode_prompt_for_vector(
                clip, item.a, encoded_references=encoded_refs
            )
            eb, tb, ib, db = cv_nodes.encode_prompt_for_vector(
                clip, item.b, encoded_references=encoded_refs
            )
            ra = refiner_run(ea)
            rb = refiner_run(eb)
            fdiff, end_a, end_b = h3_spans.differing_token_bounds(ia, ib)
            ma = h3_spans.pooling_mask(
                h3_spans.spans(ia, ta.tolist(), da),
                pooling,
                first_diff=min(fdiff, len(ia) - 1),
                end_diff=end_a,
            )
            mb = h3_spans.pooling_mask(
                h3_spans.spans(ib, tb.tolist(), db),
                pooling,
                first_diff=min(fdiff, len(ib) - 1),
                end_diff=end_b,
            )
            p_a.append(ra.to(dtype=torch.float64)[ma].mean(dim=0).to(dtype=torch.float32))
            p_b.append(rb.to(dtype=torch.float64)[mb].mean(dim=0).to(dtype=torch.float32))
            p_w.append(float(item.weight))
        gen_flip = torch.Generator(device="cpu").manual_seed(int(args.seed))
        flip_dir = cv.flip_null(
            torch.stack(p_a, dim=0),
            torch.stack(p_b, dim=0),
            weights=torch.tensor(p_w, dtype=torch.float32),
            generator=gen_flip,
        )

    edit_cond: list[Any] | None = None
    edit_ctx: torch.Tensor | None = None
    edit_ids: list[int] | None = None
    edit_tags: torch.Tensor | None = None
    pad_cond: list[Any] | None = None
    pad_ctx: torch.Tensor | None = None
    pad_ids: list[int] | None = None
    pad_tags: torch.Tensor | None = None
    if args.edit_prompt:
        _edit_label, edit_text = _resolve_prompt_text(args.edit_prompt)
        edit_cond, edit_enc, edit_tags, edit_ids, _ = cv_nodes.encode_conditioning_for_vector(
            clip, edit_text, encoded_references=encoded_refs
        )
        edit_ctx = refiner_run(edit_enc)
        extra_tok = len(edit_ids) - len(base_ids)
        pad_text = _build_padding_prompt(prompt_text, extra_tok, clip)
        pad_cond, pad_enc, pad_tags, pad_ids, _ = cv_nodes.encode_conditioning_for_vector(
            clip, pad_text, encoded_references=encoded_refs
        )
        pad_ctx = refiner_run(pad_enc)

    # Free the 25 GB text encoder before any DiT load so they never contend for VRAM.
    t_enc_phase_done = time.perf_counter()
    del clip
    del encoded_refs
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass

    strengths = [float(s.strip()) for s in args.strengths.split(",") if s.strip()]
    n_vis = int((base_tags == 0).sum().item())
    n_txt = int((base_tags != 0).sum().item())

    print(f"=== Control Vector Grade: {metadata['name']} on {prompt_label} ===")
    print(f"  L_base={len(base_ids)} (tag1={n_txt}, tag0_vision={n_vis})  vector_norm={raw_norm:.4f}")
    print(
        f"  encode_times : encoder_load={t_clip_loaded - t_grade_start:.2f}s  "
        f"refiner_load={t_ref_loaded - t_clip_loaded:.2f}s  "
        f"encode_and_refine={t_enc_phase_done - t_ref_loaded:.2f}s"
    )
    if edit_ids is not None and pad_ids is not None:
        print(f"  L_edit={len(edit_ids)} (delta={len(edit_ids) - len(base_ids):+d})  L_pad={len(pad_ids)}")

    print("\n--- Post-Refiner Context Movement (Tag-1 Mean & Sequence Rel-L2) ---")
    base_tag1_mean = base_ctx.to(dtype=torch.float64)[base_tags != 0].mean(dim=0)
    base_tag1_norm = float(torch.linalg.vector_norm(base_tag1_mean).item())

    context_report: dict[str, Any] = {
        "L_base": len(base_ids),
        "n_tag1": n_txt,
        "n_vision": n_vis,
        "vector_raw_norm": raw_norm,
        "arms": {},
    }

    if edit_ctx is not None and edit_tags is not None and edit_ids is not None:
        edit_tag1_mean = edit_ctx.to(dtype=torch.float64)[edit_tags != 0].mean(dim=0)
        edit_delta = edit_tag1_mean - base_tag1_mean
        edit_move = float(torch.linalg.vector_norm(edit_delta).item()) / base_tag1_norm
        ctx_matched_s = float(torch.linalg.vector_norm(edit_delta).item()) / raw_norm
        context_report["L_edit"] = len(edit_ids)
        context_report["arms"]["edit_prompt"] = {
            "tag1_mean_rel_l2": edit_move,
            "cos_with_v": _cosine(edit_delta, vec),
            "context_matched_strength": ctx_matched_s,
        }
        print(
            f"  edit_prompt      : tag1_mean_rel_l2={edit_move:.6f}  "
            f"cos_with_v={_cosine(edit_delta, vec):+.4f}  "
            f"ctx_matched_strength={ctx_matched_s:.4f}"
        )
    if pad_ctx is not None and pad_tags is not None and pad_ids is not None:
        pad_tag1_mean = pad_ctx.to(dtype=torch.float64)[pad_tags != 0].mean(dim=0)
        pad_delta = pad_tag1_mean - base_tag1_mean
        pad_move = float(torch.linalg.vector_norm(pad_delta).item()) / base_tag1_norm
        context_report["L_pad"] = len(pad_ids)
        context_report["arms"]["padding_control"] = {
            "tag1_mean_rel_l2": pad_move,
            "cos_with_v": _cosine(pad_delta, vec),
        }
        print(
            f"  padding_control  : tag1_mean_rel_l2={pad_move:.6f}  "
            f"cos_with_v={_cosine(pad_delta, vec):+.4f}"
        )

    period_tag1_mean = period_ctx.to(dtype=torch.float64)[period_tags != 0].mean(dim=0)
    period_delta = period_tag1_mean - base_tag1_mean
    period_move = float(torch.linalg.vector_norm(period_delta).item()) / base_tag1_norm
    context_report["arms"]["period_null"] = {
        "tag1_mean_rel_l2": period_move,
        "cos_with_v": _cosine(period_delta, vec),
    }
    print(f"  period_null (' .'): tag1_mean_rel_l2={period_move:.6f}  cos_with_v={_cosine(period_delta, vec):+.4f}")

    for s in strengths:
        pushed_masked = cv.apply_post_refiner(
            base_ctx_b, base_tags_b, [(vec, s)], mask_vision=True
        )[0]
        rel_seq = _rel_l2(pushed_masked, base_ctx)
        vis_diff = (
            float((pushed_masked[base_tags == 0] - base_ctx[base_tags == 0]).abs().max().item())
            if n_vis > 0
            else 0.0
        )
        arm_entry: dict[str, Any] = {"seq_rel_l2": rel_seq, "vis_max_diff": vis_diff}
        msg = f"  vector s={s:+6.2f}  : seq_rel_l2={rel_seq:.6f}  vis_max_diff={vis_diff:.2e}"
        if n_vis > 0:
            pushed_unmasked = cv.apply_post_refiner(
                base_ctx_b, base_tags_b, [(vec, s)], mask_vision=False
            )[0]
            unmasked_vis_rel = _rel_l2(
                pushed_unmasked[base_tags == 0], base_ctx[base_tags == 0]
            )
            arm_entry["unmasked_vis_rel_l2"] = unmasked_vis_rel
            msg += f"  unmasked_vis_rel_l2={unmasked_vis_rel:.6f}"
        context_report["arms"][f"vector_s{s:g}"] = arm_entry
        print(msg)

    if flip_dir is not None:
        flip_pushed = cv.apply_post_refiner(
            base_ctx_b, base_tags_b, [(flip_dir, 1.0)], mask_vision=True
        )[0]
        context_report["arms"]["flip_null"] = {
            "seq_rel_l2": _rel_l2(flip_pushed, base_ctx),
            "cos_with_v": _cosine(flip_dir, vec),
        }
        print(
            f"  flip_null s=+1.0 : seq_rel_l2={_rel_l2(flip_pushed, base_ctx):.6f}  "
            f"cos_with_v={_cosine(flip_dir, vec):+.4f}"
        )
    rand_pushed = cv.apply_post_refiner(
        base_ctx_b, base_tags_b, [(rand_dir, 1.0)], mask_vision=True
    )[0]
    context_report["arms"]["random_null"] = {
        "seq_rel_l2": _rel_l2(rand_pushed, base_ctx),
        "cos_with_v": _cosine(rand_dir, vec),
    }
    print(
        f"  rand_null s=+1.0 : seq_rel_l2={_rel_l2(rand_pushed, base_ctx):.6f}  "
        f"cos_with_v={_cosine(rand_dir, vec):+.4f}"
    )

    dit_results: list[dict[str, Any]] = []
    dit_provenance: dict[str, Any] = {
        "unet_file": unet_path.name,
        "loras": grade_loras,
        "core_revision": substrate.git_head(cv_nodes.COMFY_ROOT) or "unknown",
        "pack_commit": substrate.git_head(REPO_ROOT) or "unknown",
        "attention": "sol" if getattr(args, "sol", False) else "stock_dense",
        "timings_s": {
            "encoder_load_s": round(t_clip_loaded - t_grade_start, 3),
            "refiner_load_s": round(t_ref_loaded - t_clip_loaded, 3),
            "encode_and_refine_s": round(t_enc_phase_done - t_ref_loaded, 3),
        },
    }
    if not getattr(args, "no_dit", False):
        use_pika = bool(getattr(args, "pika", False))
        manual_sigmas_str = getattr(args, "manual_sigmas", None) or (
            h3_config.PIKA_MANUAL_SIGMAS if use_pika else None
        )
        probe_steps_str = args.probe_steps
        if use_pika and probe_steps_str == "0,8,15":
            probe_steps_str = "0,2,5"
        probe_steps = [int(x.strip()) for x in probe_steps_str.split(",") if x.strip()]
        dit_runner = getattr(args, "_dit_runner", None)
        if dit_runner is None:
            import comfy.model_management as mm
            import comfy.sample
            import comfy.samplers
            import comfy.sd
            import comfy.utils
            from comfy_extras.nodes_minimax_h3 import _empty_av_latent

            t_dit_load_start = time.perf_counter()
            lora_note = (
                f", loras={[l['file'] for l in grade_loras]}" if grade_loras else ""
            )
            print(
                f"\n[grade] loading DiT {unet_path.name} "
                f"(core={dit_provenance['core_revision']}, attn={dit_provenance['attention']}{lora_note}) "
                f"for denoised-prediction grading..."
            )
            raw_model = comfy.sd.load_diffusion_model(str(unet_path))
            if grade_loras:
                lb = cv_nodes._repo("lora_branch")
                for l_spec in grade_loras:
                    l_path = cv_nodes.resolve_lora_file_path(
                        l_spec["file"], expected_sha256=l_spec.get("sha256")
                    )
                    l_sd = lb.native_keys(
                        comfy.utils.load_torch_file(str(l_path), safe_load=True),
                        raw_model,
                    )
                    l_branches = lb.parse_lora(l_sd, float(l_spec.get("strength", 1.0)))
                    raw_model = lb.attach(raw_model, l_branches)
            base_model = _sigma_shifted(raw_model)
            if getattr(args, "sol", False):
                sol_mod = cv_nodes._repo("sol_attn_h3")
                sol_steps = (
                    h3_config.PIKA_STEPS if use_pika else h3_config.SAMPLING["steps"]
                )
                sol_cfg = h3_config.sol_for_graph(False, sol_steps)
                (base_model,) = sol_mod._apply_sol(
                    base_model,
                    tau=sol_cfg["tau"],
                    quantizer=sol_cfg["quantizer"],
                    dense_blocks=sol_cfg["dense_blocks"],
                    sink_conditioning=sol_cfg["sink_conditioning"],
                    token_routing=sol_cfg["token_routing"],
                    routing_blocks=sol_cfg.get("routing_blocks", ""),
                    start_percent=sol_cfg["start_percent"],
                    end_percent=sol_cfg["end_percent"],
                    min_tokens=sol_cfg["min_tokens"],
                    verbose=False,
                )
            latent_dict, _frames = _empty_av_latent(args.width, args.height, args.length)
            latent = latent_dict["samples"]
            noise = comfy.sample.prepare_noise(latent, args.noise_seed)
            if manual_sigmas_str:
                sigmas = torch.tensor(
                    [float(s.strip()) for s in manual_sigmas_str.split(",") if s.strip()],
                    dtype=torch.float32,
                )
            else:
                sigmas = comfy.samplers.calculate_sigmas(
                    base_model.get_model_object("model_sampling"),
                    h3_config.SAMPLING["scheduler"],
                    h3_config.SAMPLING["steps"],
                )
            dit_provenance["sigmas"] = [round(float(x), 6) for x in sigmas.tolist()]
            t_dit_loaded = time.perf_counter()
            dit_provenance["timings_s"]["dit_load_s"] = round(t_dit_loaded - t_dit_load_start, 3)
            print(f"[grade] DiT + latent ready in {t_dit_loaded - t_dit_load_start:.2f}s (canvas={args.width}x{args.height}x{args.length})")

            def _default_dit_runner(m_obj: Any, cond_obj: list[Any], step_idx: int) -> tuple[float, Any]:
                return (
                    float(sigmas[step_idx]),
                    _one_forward(m_obj, cond_obj, noise, latent, sigmas, step_idx, args.noise_seed),
                )

            dit_runner = _default_dit_runner
        else:
            base_model = getattr(args, "_base_model")

        forward_times_s: list[float] = []
        raw_dit_runner = dit_runner

        def _timed_dit_runner(m_obj: Any, cond_obj: list[Any], step_idx: int) -> tuple[float, Any]:
            tf0 = time.perf_counter()
            out = raw_dit_runner(m_obj, cond_obj, step_idx)
            forward_times_s.append(time.perf_counter() - tf0)
            return out

        dit_runner = _timed_dit_runner

        print(f"\n--- DiT Denoised-Prediction Movement (steps={probe_steps}) ---")
        for step_idx in probe_steps:
            sigma_val, pred_base_raw = dit_runner(base_model, base_cond, step_idx)
            base_b = _pred_blocks(pred_base_raw)

            # Hard check: strength 0.0 hooked reproduces plain base prediction exactly
            m_zero = _attach_raw_vector(
                base_model, metadata["name"], vec, 0.0, mask="non_vision", metadata=metadata
            )
            _, pred_zero_raw = dit_runner(m_zero, base_cond, step_idx)
            zero_b = _pred_blocks(pred_zero_raw)
            zero_metrics = _block_metrics(zero_b, base_b)
            if zero_metrics["video"]["relative_l2"] != 0.0 or zero_metrics["audio"]["relative_l2"] != 0.0:
                raise ValueError(
                    f"DiT strength=0.0 null check failed at step {step_idx}: "
                    f"video_rel_l2={zero_metrics['video']['relative_l2']} "
                    f"audio_rel_l2={zero_metrics['audio']['relative_l2']}"
                )

            step_arms: dict[str, Any] = {"strength_0_null": zero_metrics}
            edit_b: dict[str, torch.Tensor] | None = None
            if edit_cond is not None:
                _, pred_edit_raw = dit_runner(base_model, edit_cond, step_idx)
                edit_b = _pred_blocks(pred_edit_raw)
                step_arms["edit_prompt"] = _block_metrics(edit_b, base_b)
                em = step_arms["edit_prompt"]
                print(
                    f"  [step {step_idx:2d} sig={sigma_val:.4f}] {'edit_prompt':<22s} "
                    f"video_rel_l2={em['video']['relative_l2']:.6f}  "
                    f"audio_rel_l2={em['audio']['relative_l2']:.6f}"
                )

            if pad_cond is not None:
                _, pred_pad_raw = dit_runner(base_model, pad_cond, step_idx)
                pad_b = _pred_blocks(pred_pad_raw)
                step_arms["padding_control"] = _block_metrics(pad_b, base_b, edit_b)
                pm = step_arms["padding_control"]
                print(
                    f"  [step {step_idx:2d} sig={sigma_val:.4f}] {'padding_control':<22s} "
                    f"video_rel_l2={pm['video']['relative_l2']:.6f}  "
                    f"audio_rel_l2={pm['audio']['relative_l2']:.6f}"
                )

            _, pred_period_raw = dit_runner(base_model, period_cond, step_idx)
            period_b = _pred_blocks(pred_period_raw)
            step_arms["period_null"] = _block_metrics(period_b, base_b, edit_b)
            prm = step_arms["period_null"]
            print(
                f"  [step {step_idx:2d} sig={sigma_val:.4f}] {'period_null':<22s} "
                f"video_rel_l2={prm['video']['relative_l2']:.6f}  "
                f"audio_rel_l2={prm['audio']['relative_l2']:.6f}"
            )

            if flip_dir is not None:
                m_flip = _attach_raw_vector(
                    base_model, "flip_null", flip_dir, 1.0, mask="non_vision", metadata=metadata
                )
                _, pred_flip_raw = dit_runner(m_flip, base_cond, step_idx)
                flip_b = _pred_blocks(pred_flip_raw)
                step_arms["flip_null"] = _block_metrics(flip_b, base_b, edit_b)
                fm = step_arms["flip_null"]
                print(
                    f"  [step {step_idx:2d} sig={sigma_val:.4f}] {'flip_null (s=+1.0)':<22s} "
                    f"video_rel_l2={fm['video']['relative_l2']:.6f}  "
                    f"audio_rel_l2={fm['audio']['relative_l2']:.6f}"
                )

            m_rand = _attach_raw_vector(
                base_model, "random_null", rand_dir, 1.0, mask="non_vision", metadata=metadata
            )
            _, pred_rand_raw = dit_runner(m_rand, base_cond, step_idx)
            rand_b = _pred_blocks(pred_rand_raw)
            step_arms["random_null"] = _block_metrics(rand_b, base_b, edit_b)
            rm = step_arms["random_null"]
            print(
                f"  [step {step_idx:2d} sig={sigma_val:.4f}] {'random_null (s=+1.0)':<22s} "
                f"video_rel_l2={rm['video']['relative_l2']:.6f}  "
                f"audio_rel_l2={rm['audio']['relative_l2']:.6f}"
            )

            for s in strengths:
                if s == 0.0:
                    step_arms["vector_s0"] = zero_metrics
                    continue
                m_vec = _attach_raw_vector(
                    base_model, metadata["name"], vec, s, mask="non_vision", metadata=metadata
                )
                _, pred_vec_raw = dit_runner(m_vec, base_cond, step_idx)
                vec_b = _pred_blocks(pred_vec_raw)
                vm = _block_metrics(vec_b, base_b, edit_b)
                step_arms[f"vector_s{s:g}"] = vm
                extra_str = ""
                if edit_b is not None:
                    extra_str = (
                        f"  cos_edit(v={vm['video']['delta_cosine_with_edit']:+.4f},"
                        f"a={vm['audio']['delta_cosine_with_edit']:+.4f})"
                    )
                if n_vis > 0 or getattr(args, "compare_unmasked", False):
                    m_unmasked = _attach_raw_vector(
                        base_model, metadata["name"], vec, s, mask="all_rows", metadata=metadata
                    )
                    _, pred_unm_raw = dit_runner(m_unmasked, base_cond, step_idx)
                    unm_b = _pred_blocks(pred_unm_raw)
                    step_arms[f"vector_unmasked_s{s:g}"] = _block_metrics(unm_b, base_b, edit_b)
                    step_arms[f"masked_vs_unmasked_s{s:g}"] = _block_metrics(unm_b, vec_b)
                    extra_str += (
                        f"  unmasked_diff_v={step_arms[f'masked_vs_unmasked_s{s:g}']['video']['relative_l2']:.6f}"
                    )
                print(
                    f"  [step {step_idx:2d} sig={sigma_val:.4f}] {f'vector (s={s:+.2f})':<22s} "
                    f"video_rel_l2={vm['video']['relative_l2']:.6f}  "
                    f"audio_rel_l2={vm['audio']['relative_l2']:.6f}{extra_str}"
                )

            matched_strength_video: float | None = None
            if edit_b is not None and "vector_s1" in step_arms:
                s1_v = step_arms["vector_s1"]["video"]["relative_l2"]
                ed_v = step_arms["edit_prompt"]["video"]["relative_l2"]
                if s1_v > 1e-12:
                    matched_strength_video = ed_v / s1_v
                    print(
                        f"  [step {step_idx:2d}] matched_strength_video = {matched_strength_video:.4f} "
                        f"(edit_video_rel_l2={ed_v:.6f} / vec_s1_video_rel_l2={s1_v:.6f})"
                    )

            dit_results.append(
                {
                    "step": step_idx,
                    "sigma": sigma_val,
                    "matched_strength_video": matched_strength_video,
                    "arms": step_arms,
                }
            )
        if forward_times_s:
            dit_provenance["timings_s"]["n_forwards"] = len(forward_times_s)
            dit_provenance["timings_s"]["mean_forward_s"] = round(
                sum(forward_times_s) / len(forward_times_s), 3
            )
            dit_provenance["timings_s"]["total_forward_s"] = round(sum(forward_times_s), 3)
            print(
                f"\n  [grade] {len(forward_times_s)} DiT forwards in {sum(forward_times_s):.2f}s "
                f"(mean={sum(forward_times_s) / len(forward_times_s):.2f}s/forward)"
            )

    if getattr(args, "out", None):
        out_record = {
            "vector": metadata["name"],
            "vector_file": vpath.name,
            "prompt": prompt_label,
            "edit_prompt": args.edit_prompt,
            "seed": int(args.seed),
            "provenance": dit_provenance,
            "context_movement": context_report,
            "dit_prediction_movement": dit_results,
        }
        Path(args.out).write_text(json.dumps(out_record, indent=2) + "\n", encoding="utf-8")

    return 0


def _display_path(p: Path) -> str:
    resolved = p.resolve()
    if resolved.is_relative_to(REPO_ROOT.resolve()):
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    parts = resolved.parts
    if "ComfyUI-h3-explorations" in parts:
        idx = parts.index("ComfyUI-h3-explorations")
        return str(Path(*parts[idx + 1 :]))
    return p.name


def cmd_from_md(args: argparse.Namespace) -> int:
    md_path = Path(args.md_path)
    if not md_path.is_file():
        raise ValueError(f"markdown file not found: {md_path}")

    out_target: str | Path = (
        Path(args.out) if getattr(args, "out", None) else cv_nodes.PAIR_BANK_DIR
    )
    parsed = cv.parse_markdown_pairs(md_path.read_text(encoding="utf-8"))
    saved_toml = cv.write_markdown_to_toml(
        md_path,
        out_target,
        name=getattr(args, "name", None),
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


def _recommended_strength_text(raw_norm: float) -> str:
    if raw_norm <= 1e-6:
        return "1.0"
    s_lo = max(0.25, round(50.0 / raw_norm, 2))
    s_hi = max(s_lo + 0.25, round(85.0 / raw_norm, 2))
    return f"{s_lo:.2f} .. {s_hi:.2f}"


def cmd_build(args: argparse.Namespace) -> int:
    if getattr(args, "build_all", False):
        toml_dir = Path(getattr(args, "toml_dir", None) or cv_nodes.PAIR_BANK_DIR)
        all_tomls = sorted(toml_dir.glob("*.toml"))
        if not all_tomls:
            raise ValueError(f"no .toml pair files found in {toml_dir}")
        encoder_name = getattr(args, "encoder", None) or h3_config.MODELS["clip"]
        t_batch_0 = time.perf_counter()
        clip_shared = getattr(args, "_clip_override", None) or _load_clip(
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

        fam_arg = str(getattr(args, "families", "both") or "both").lower()
        target_fams = ["fl2va", "ref2va"] if fam_arg in ("both", "auto") else [fam_arg]
        out_dir = (
            Path(args.out_dir)
            if getattr(args, "out_dir", None)
            else Path(folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        against_w = str(getattr(args, "against_words", "") or "").strip() or None
        hold_w = str(getattr(args, "hold_words", "") or "").strip() or None
        hold_alpha = float(getattr(args, "hold_alpha", 1.0))

        summary_rows_w: list[dict[str, Any]] = []
        for fam in target_fams:
            synth = j_lens.synthesize_from_words(
                from_words_str,
                against_w,
                family=fam,
                hold_words=hold_w,
                alpha=hold_alpha,
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
            print(
                f"  * {row['file']}\n"
                f"      family={row['family']}  pooling={row['pooling']}  "
                f"raw_norm={row['raw_norm']:.2f}  recommended strength: {row['rec_strength']}\n"
                f"      j_lens (+): {jl.get('pos_summary', '')}\n"
                f"      j_lens (-): {jl.get('neg_summary', '')}"
            )
        return 0

    if not getattr(args, "source", None) or not str(args.source).strip():
        raise ValueError("either a source (.md/.toml/stem), --from-words, or --all is required for 'build'")

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

    src_arg = str(args.source).strip()
    src_path = Path(src_arg)
    test_prompt_saved: Path | None = None
    if src_path.suffix.lower() == ".md" or (src_path.is_file() and src_arg.lower().endswith(".md")):
        if not src_path.is_file():
            raise ValueError(f"markdown file not found: {src_path}")
        md_text = src_path.read_text(encoding="utf-8")
        parsed = cv.parse_markdown_pairs(md_text)
        stem_override = (getattr(args, "name", None) or getattr(args, "output_name", None) or "").strip() or None
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
                src_path,
                toml_target_path,
                name=stem_override,
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
    else:
        resolved_pair_path = cv_nodes.resolve_pair_file_path(src_arg)

    pair_set = cv.load_pairs(resolved_pair_path, bank_dir=cv_nodes.PROMPT_BANK_DIR)
    stem = (getattr(args, "output_name", None) or getattr(args, "name", None) or pair_set.name).strip()
    if not cv.NAME_RE.match(stem):
        raise ValueError(f"output vector stem must match [a-z0-9_]+, got {stem!r}")

    if getattr(args, "unet", None):
        family_targets = [("custom", str(args.unet))]
    else:
        fam_choice = str(getattr(args, "families", "both")).lower()
        if fam_choice == "both":
            family_targets = [
                ("fl2va", h3_config.MODELS["unet_fl2va"]),
                ("ref2va", h3_config.MODELS["unet_ref2va"]),
            ]
        elif fam_choice == "fl2va":
            family_targets = [("fl2va", h3_config.MODELS["unet_fl2va"])]
        elif fam_choice == "ref2va":
            family_targets = [("ref2va", h3_config.MODELS["unet_ref2va"])]
        elif fam_choice == "auto":
            if getattr(args, "ref_images", None) or pair_set.mode == "ref2va":
                family_targets = [("ref2va", h3_config.MODELS["unet_ref2va"])]
            else:
                family_targets = [("fl2va", h3_config.MODELS["unet_fl2va"])]
        else:
            raise ValueError(f"unknown --families choice: {fam_choice!r}")

    encoder_name = getattr(args, "encoder", None) or h3_config.MODELS["clip"]
    encoder_path = Path(folder_paths.get_full_path_or_raise("text_encoders", encoder_name))

    t0 = time.perf_counter()
    clip = getattr(args, "_clip_override", None) or _load_clip(encoder_name, device=args.device)
    t_clip = time.perf_counter()

    encoded_refs = None
    if getattr(args, "ref_images", None):
        imgs = [s.strip() for s in str(args.ref_images).split(",") if s.strip()]
        encoded_refs = _load_encoded_references(
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
    print(f"  pairs_toml        : {_display_path(resolved_pair_path)} ({len(pair_set.pairs)} pairs)")
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
    print(f"  encoded {len(pair_set.pairs)} pairs in {t_enc_done - t_enc_start:.2f}s")

    w_vec = torch.tensor(weights_list, dtype=torch.float32)
    median_enc_row_norm_ex0 = float(
        sum(c["median_row_norm_ex_row0"] for c in enc_censuses) / len(enc_censuses)
    )
    try:
        rel_pair_path = str(resolved_pair_path.relative_to(REPO_ROOT))
    except ValueError:
        rel_pair_path = resolved_pair_path.name

    out_dir = (
        Path(args.out_dir)
        if getattr(args, "out_dir", None)
        else Path(folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for fam_hint, unet_name in family_targets:
        unet_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", unet_name))
        family = cv.detect_family(unet_path)
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
            str(unet_path),
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
                unet_path_override=unet_path,
                encoder_path_override=encoder_path,
                loras_override=loras,
            )
            refiner_run = cv_nodes.build_refiner_runner(
                unet_path, device=args.refiner_device, loras=loras
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
                )
                mask_b = h3_spans.pooling_mask(
                    cp["spans_b"],
                    pool_mode,
                    first_diff=min(first_diff, len(ids_b) - 1),
                    end_diff=end_b,
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
                    }
                )

            a_mat = torch.stack(a_pooled_list, dim=0)
            b_mat = torch.stack(b_pooled_list, dim=0)
            use_unit_mean = bool(getattr(args, "unit_mean", False))
            raw_vec = cv.mean_difference(a_mat, b_mat, w_vec, unit_mean=use_unit_mean)
            tensors_to_save: dict[str, torch.Tensor] = {}

            if cached_holds:
                hold_a_list: list[torch.Tensor] = []
                hold_b_list: list[torch.Tensor] = []
                hold_groups: list[str] = []
                hold_weights: list[float] = []
                for ch, (h_ref_a, h_ref_b) in zip(cached_holds, refined_holds):
                    h_diff = ch["first_diff"]
                    h_ids_a = ch["ids_a"]
                    h_ids_b = ch["ids_b"]
                    h_mask_a = h3_spans.pooling_mask(
                        ch["spans_a"],
                        pool_mode,
                        first_diff=min(h_diff, len(h_ids_a) - 1),
                        end_diff=ch["end_a"],
                    )
                    h_mask_b = h3_spans.pooling_mask(
                        ch["spans_b"],
                        pool_mode,
                        first_diff=min(h_diff, len(h_ids_b) - 1),
                        end_diff=ch["end_b"],
                    )
                    hold_a_list.append(
                        h_ref_a.to(dtype=torch.float64)[h_mask_a].mean(dim=0).to(dtype=torch.float32)
                    )
                    hold_b_list.append(
                        h_ref_b.to(dtype=torch.float64)[h_mask_b].mean(dim=0).to(dtype=torch.float32)
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
            if getattr(args, "include_encoder", True):
                enc_a_mat = torch.stack(enc_a_pooled_list, dim=0)
                enc_b_mat = torch.stack(enc_b_pooled_list, dim=0)
                enc_raw_vec = cv.mean_difference(enc_a_mat, enc_b_mat, w_vec)
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
                    q_report["j_lens_readout"] = jl_build_readout
                except Exception:
                    jl_build_readout = None

            measured_mode = "ref2va" if encoded_refs is not None else pair_set.mode
            metadata: dict[str, Any] = {
                "format": cv.FORMAT_VERSION,
                "name": stem,
                "quality": pair_set.quality,
                "family": family,
                "pairs": {
                    "path": rel_pair_path,
                    "sha256": pair_set.sha256,
                    "n_pairs": len(pair_set.pairs),
                    "n_holds": len(cached_holds),
                    "a_side": pair_set.a_side,
                    "b_side": pair_set.b_side,
                    "pair_spans": pair_spans,
                },
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
                },
            }

            fname = cv.default_vector_filename(
                stem,
                family,
                pooling=pool_mode,
                prefix=prefix_meta,
                apply_holds=not bool(getattr(args, "no_holds", False)),
                n_holds=len(pair_set.holds),
                loras=loras,
            )
            out_path = out_dir / fname
            saved = cv.save_vector(
                out_path,
                tensors_to_save,
                metadata,
                overwrite=bool(getattr(args, "overwrite", False)),
            )
            summary_rows.append(
                {
                    "file": saved.name,
                    "family": family,
                    "pooling": pool_mode,
                    "raw_norm": raw_norm,
                    "mpc": float(q_report["mean_pairwise_cosine"]),
                    "split_half": float(q_report["split_half_cosine"]),
                    "rec_strength": _recommended_strength_text(raw_norm),
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


class ControlVectorUIState:
    def __init__(
        self,
        *,
        toml_dir: Path | None = None,
        vec_dir: Path | None = None,
    ) -> None:
        self.toml_dir: Path = toml_dir or cv_nodes.PAIR_BANK_DIR
        self.vec_dir: Path = vec_dir or Path(
            folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0]
        )
        self._tokenizer: Any = None
        self._clips: dict[tuple[str, str], Any] = {}
        self._refiner_cache: dict[Any, Any] = {}
        self.shutdown_requested: bool = False

    def get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            import comfy.text_encoders.minimax as mm_te

            self._tokenizer = mm_te.MiniMaxH3Tokenizer()
        return self._tokenizer

    def get_clip(self, encoder_name: str | None = None, device: str = "default") -> Any:
        resolved_enc = encoder_name or h3_config.MODELS["clip"]
        key = (str(resolved_enc), str(device))
        if key not in self._clips:
            self._clips[key] = _load_clip(resolved_enc, device=str(device))
        return self._clips[key]


def _extract_system_prompt_text() -> str:
    candidates = [
        REPO_ROOT / "docs" / "wiki" / "system_prompts" / "h3_control_vector_system_prompt.md",
        REPO_ROOT / "docs" / "research" / "control_vectors" / "2026-11-05_control_vector_system_prompt.md",
    ]
    open_fence = "````markdown\n"
    for sp_path in candidates:
        if not sp_path.is_file():
            continue
        raw = sp_path.read_text(encoding="utf-8")
        fence_start = raw.find(open_fence)
        if fence_start >= 0:
            body_start = fence_start + len(open_fence)
            fence_end = raw.find("\n````", body_start)
            if fence_end >= 0:
                return raw[body_start:fence_end].strip()
    return ""


def _list_ui_toml_files(toml_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not toml_dir.is_dir():
        return out
    for p in sorted(toml_dir.glob("*.toml")):
        try:
            ps = cv.load_pairs(p, bank_dir=cv_nodes.PROMPT_BANK_DIR)
            out.append(
                {
                    "stem": p.stem,
                    "file": p.name,
                    "n_pairs": len(ps.pairs),
                    "mode": ps.mode,
                    "frame": ps.frame,
                    "quality": ps.quality,
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
            out.append(
                {
                    "file": p.name,
                    "name": str(meta.get("name", p.stem)),
                    "family": str(meta.get("family", "")),
                    "pooling": str(meta.get("measured_under", {}).get("pooling", "scene_text")),
                    "raw_norm": raw_norm,
                    "mpc": float(qr.get("mean_pairwise_cosine", 0.0)),
                    "split_half": float(qr.get("split_half_cosine", 0.0)),
                    "rec_strength": _recommended_strength_text(raw_norm),
                    "j_lens": qr.get("j_lens_readout"),
                }
            )
        except Exception:
            continue
    return out


def _pair_set_to_markdown(toml_path: Path, ps: cv.PairSet) -> str:
    tp_txt = ""
    tp_cand = toml_path.parent / f"{toml_path.stem}_test_prompt.txt"
    if tp_cand.is_file():
        tp_txt = tp_cand.read_text(encoding="utf-8").strip()
    else:
        raw_lines = toml_path.read_text(encoding="utf-8").splitlines()
        in_tp = False
        tp_lines: list[str] = []
        for line in raw_lines:
            if line.strip() == "# Test Scene Prompt:":
                in_tp = True
                continue
            if in_tp:
                if line.startswith("#"):
                    tp_lines.append(line[1:].lstrip(" "))
                else:
                    break
        tp_txt = "\n".join(tp_lines).strip()

    chunks: list[str] = []
    if tp_txt:
        chunks.append(f"### Test Scene Prompt\n\n{tp_txt}\n")
    chunks.append(f"### Contrastive Pairs (1–{len(ps.pairs)})\n")
    for idx, item in enumerate(ps.pairs, start=1):
        chunks.append(
            f"#### Pair {idx}\n"
            f"##### Prompt A (Baseline)\n{item.a}\n\n"
            f"##### Prompt B (Steered)\n{item.b}\n"
        )
    if ps.holds:
        chunks.append("### Hold Pairs\n")
        for idx, h_item in enumerate(ps.holds, start=1):
            chunks.append(
                f"#### Hold {idx} ({h_item.group})\n"
                f"##### Prompt A (Baseline)\n{h_item.a}\n\n"
                f"##### Prompt B (Nuisance)\n{h_item.b}\n"
            )
    return "\n".join(chunks)


def _preview_markdown_pairs(
    state: ControlVectorUIState,
    text: str,
    *,
    frame: str = "scene",
    name: str = "",
) -> dict[str, Any]:
    if frame not in cv.VALID_FRAMES:
        return {"ok": False, "error": f"frame must be one of {cv.VALID_FRAMES}, got {frame!r}"}
    try:
        parsed = cv.parse_markdown_pairs(text)
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
                    "suffix": suffix,
                    "tokens": {
                        "first_diff": int(first_diff),
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
        "mode": parsed["mode"],
        "test_scene_prompt": parsed["test_scene_prompt"],
        "quality": quality,
        "suggested_name": name,
        "avg_tokens": avg_tokens,
        "lint_warnings": lint_warnings,
        "pairs": pair_rows,
    }


def handle_ui_request(
    state: ControlVectorUIState,
    method: str,
    path: str,
    body: bytes = b"",
) -> tuple[int, str, str]:
    clean_path = path.split("?", 1)[0]
    if method == "GET" and clean_path in ("/", "/index.html"):
        html_path = BENCH_DIR / "control_vector_ui.html"
        return 200, "text/html; charset=utf-8", html_path.read_text(encoding="utf-8")

    if method == "GET" and clean_path == "/api/init":
        payload = {
            "ok": True,
            "system_prompt": _extract_system_prompt_text(),
            "toml_files": _list_ui_toml_files(state.toml_dir),
            "vectors": _list_ui_vectors(state.vec_dir),
        }
        return 200, "application/json", json.dumps(payload)

    req: dict[str, Any] = {}
    if body:
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": f"invalid JSON: {exc}"})

    if method == "POST" and clean_path == "/api/preview":
        res = _preview_markdown_pairs(
            state,
            str(req.get("text", "")),
            frame=str(req.get("frame", "scene") or "scene"),
            name=str(req.get("name", "") or "").strip(),
        )
        return 200, "application/json", json.dumps(res)

    if method == "POST" and clean_path == "/api/load-toml":
        fname = str(req.get("file", "")).strip()
        if not fname:
            return 400, "application/json", json.dumps({"ok": False, "error": "missing 'file'"})
        try:
            cand = state.toml_dir / (fname if fname.endswith(".toml") else f"{fname}.toml")
            tpath = cand if cand.is_file() else cv_nodes.resolve_pair_file_path(fname)
            ps = cv.load_pairs(tpath, bank_dir=cv_nodes.PROMPT_BANK_DIR)
            md_str = _pair_set_to_markdown(tpath, ps)
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "name": ps.name,
                        "frame": ps.frame,
                        "mode": ps.mode,
                        "markdown": md_str,
                    }
                ),
            )
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/save-toml":
        text = str(req.get("text", ""))
        name = str(req.get("name", "")).strip()
        frame = str(req.get("frame", "scene") or "scene")
        overwrite = bool(req.get("overwrite", True))
        if not name:
            return 400, "application/json", json.dumps({"ok": False, "error": "vector name is required"})
        try:
            parsed = cv.parse_markdown_pairs(text)
            tp_out = state.toml_dir / f"{name}_test_prompt.txt" if parsed.get("test_scene_prompt") else None
            saved_toml = cv.write_markdown_to_toml(
                text,
                state.toml_dir,
                name=name,
                frame=frame,
                test_prompt_out=tp_out,
                overwrite=overwrite,
                bank_dir=cv_nodes.PROMPT_BANK_DIR,
            )
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "saved_toml": _display_path(saved_toml),
                        "test_prompt_file": _display_path(tp_out) if tp_out and tp_out.is_file() else None,
                    }
                ),
            )
        except (ValueError, FileExistsError) as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/build":
        text = str(req.get("text", ""))
        name = str(req.get("name", "")).strip()
        frame = str(req.get("frame", "scene") or "scene")
        families = str(req.get("families", "both") or "both")
        pooling = str(req.get("pooling", "diff_span") or "diff_span")
        no_lora = bool(req.get("no_lora", False))
        overwrite = bool(req.get("overwrite", True))
        include_encoder = bool(req.get("include_encoder", True))
        if not name:
            return 400, "application/json", json.dumps({"ok": False, "error": "vector name is required"})
        try:
            import contextlib
            import io

            parsed = cv.parse_markdown_pairs(text)
            tp_out = state.toml_dir / f"{name}_test_prompt.txt" if parsed.get("test_scene_prompt") else None
            cand_toml = state.toml_dir / f"{name}.toml"
            reused_existing = False
            if cand_toml.is_file():
                try:
                    existing_ps = cv.load_pairs(cand_toml, bank_dir=cv_nodes.PROMPT_BANK_DIR)
                    existing_ab = [(p.a, p.b) for p in existing_ps.pairs]
                    new_ab = [(p["a"], p["b"]) for p in parsed["pairs"]]
                    existing_holds = [(h.group, h.a, h.b) for h in existing_ps.holds]
                    new_holds = [(h["group"], h["a"], h["b"]) for h in parsed.get("holds", [])]
                    if (
                        existing_ab == new_ab
                        and existing_holds == new_holds
                        and existing_ps.frame == frame
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
                    test_prompt_out=tp_out,
                    overwrite=True if overwrite else False,
                    bank_dir=cv_nodes.PROMPT_BANK_DIR,
                )
            argv = [
                "build",
                str(saved_toml),
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
            if include_encoder:
                argv.append("--include-encoder")
            else:
                argv.append("--no-encoder")
            if overwrite:
                argv.append("--overwrite")
            b_args = build_parser().parse_args(argv)
            rows_out: list[dict[str, Any]] = []
            b_args._summary_rows_out = rows_out
            b_args._clip_override = state.get_clip(b_args.encoder, b_args.device)
            b_args._refiner_cache = state._refiner_cache
            buf_out = io.StringIO()
            with contextlib.redirect_stdout(buf_out):
                cmd_build(b_args)
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "ok": True,
                        "saved_toml": _display_path(saved_toml),
                        "built": rows_out,
                        "vectors": _list_ui_vectors(state.vec_dir),
                        "log": buf_out.getvalue(),
                    }
                ),
            )
        except (ValueError, FileExistsError) as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": str(exc)})

    if method == "POST" and clean_path == "/api/j-lens":
        try:
            if not j_lens.bundle_available():
                return (
                    400,
                    "application/json",
                    json.dumps({"ok": False, "error": "J-Lens bundle models/j_lens/j_lens_h3.safetensors not found"}),
                )
            from_words = str(req.get("from_words", "") or "").strip()
            if from_words:
                fam = str(req.get("family", "fl2va") or "fl2va")
                against_w = str(req.get("against_words", "") or "").strip() or None
                hold_w = str(req.get("hold_words", "") or "").strip() or None
                alpha_w = float(req.get("alpha", 1.0))
                synth = j_lens.synthesize_from_words(
                    from_words,
                    against_w,
                    family=fam,
                    hold_words=hold_w,
                    alpha=alpha_w,
                )
                save_name = str(req.get("save_name", "") or "").strip()
                saved_file = None
                if save_name:
                    loras_meta = j_lens._DEFAULT_RECIPES.get(fam, {}).get("loras")
                    fname = cv.default_vector_filename(save_name, fam, pooling="diff_span", loras=loras_meta)
                    saved_path = j_lens.save_synthesized_vector(state.vec_dir / fname, synth, name=save_name)
                    saved_file = saved_path.name
                return (
                    200,
                    "application/json",
                    json.dumps(
                        {
                            "ok": True,
                            "family": synth["family"],
                            "pos_words_matched": synth["pos_words_matched"],
                            "neg_words_matched": synth["neg_words_matched"],
                            "hold_words": synth["hold_words"],
                            "alpha": synth["alpha"],
                            "saved_file": saved_file,
                            "readout": synth["readout"],
                            "vectors": _list_ui_vectors(state.vec_dir) if saved_file else None,
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
                rep = j_lens.decode_recipe(resolved_steps, family=str(req.get("family", "fl2va") or "fl2va"))
                return 200, "application/json", json.dumps({"ok": True, **rep})

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
                alpha=float(req.get("alpha", 1.0)),
            )
            return 200, "application/json", json.dumps({"ok": True, **rep})
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

    state = ControlVectorUIState()
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pui = sub.add_parser(
        "ui",
        help="Launch the self-contained local browser UI for authoring, previewing, and building control vectors",
    )
    pui.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    pui.add_argument("--port", type=int, default=0, help="Port (default: 0 picks an unused ephemeral port)")
    pui.add_argument("--no-browser", dest="no_browser", action="store_true", help="Do not auto-open a browser tab")

    pb = sub.add_parser(
        "build",
        help="One-shot end-to-end build: Markdown (.md) or TOML (.toml) -> pair file + test prompt + fl2va/ref2va .safetensors vectors",
    )
    pb.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Input LLM Markdown (.md) file, pair TOML (.toml) file, or stem under prompt_bank/control_vectors/ (omit when using --all)",
    )
    pb.add_argument(
        "--all",
        dest="build_all",
        action="store_true",
        help="Build vectors for all .toml banks in prompt_bank/control_vectors/ in a single encoder load",
    )
    pb.add_argument(
        "--from-words",
        dest="from_words",
        default="",
        help="Synthesize a 5376-d control vector directly from simple positive words in <5 ms on CPU via J-Lens (no .toml or Qwen load required)",
    )
    pb.add_argument(
        "--against-words",
        dest="against_words",
        default="",
        help="Optional negative words when using --from-words (defaults to neutral baseline words)",
    )
    pb.add_argument(
        "--hold-words",
        dest="hold_words",
        default="",
        help="Optional nuisance words to project out when using --from-words",
    )
    pb.add_argument(
        "--hold-alpha",
        dest="hold_alpha",
        type=float,
        default=1.0,
        help="Soft or over-projected hold attenuation alpha in [0.0, 2.0] when using --hold-words (default: 1.0)",
    )
    pb.add_argument(
        "--unit-mean",
        dest="unit_mean",
        action="store_true",
        help="Average unit-normalized pair directions (equal directional weight per pair) scaled to mean pair norm",
    )
    pb.add_argument(
        "--families",
        choices=("both", "fl2va", "ref2va", "auto"),
        default="both",
        help="Which H3 model families to build vectors for in one pass (default: 'both' builds fl2va and ref2va)",
    )
    pb.add_argument(
        "--pooling",
        default="diff_span",
        help="Pooling mode or comma-separated modes (e.g. 'diff_span', 'from_first_diff', 'scene_text', or 'all'; default: 'diff_span')",
    )
    pb.add_argument("--name", default=None, help="Optional vector name / file stem override")
    pb.add_argument("--output-name", dest="output_name", default="", help="Optional vector stem override")
    pb.add_argument("--mode", choices=cv.VALID_MODES, default=None, help="Explicit mode override when importing .md")
    pb.add_argument("--frame", choices=cv.VALID_FRAMES, default="scene", help="Pair frame ('scene' or 'bare')")
    pb.add_argument("--quality", default=None, help="Optional vector.quality description override")
    pb.add_argument("--a-side", dest="a_side", default=None, help="Optional vector.a_side description override")
    pb.add_argument("--b-side", dest="b_side", default=None, help="Optional vector.b_side description override")
    pb.add_argument("--toml-out", dest="toml_out", default=None, help="Optional output path/dir for generated .toml")
    pb.add_argument("--test-prompt-out", dest="test_prompt_out", default=None, help="Optional output path for extracted Test Scene Prompt")
    pb.add_argument("--unet", default=None, help="Explicit diffusion checkpoint filename override")
    pb.add_argument("--encoder", default=None, help="Text encoder checkpoint filename")
    pb.add_argument("--lora", action="append", default=None, help="Explicit LoRA file[:strength] override (default: standard Pika H3 Flex LoRA per family)")
    pb.add_argument("--no-lora", dest="no_lora", action="store_true", help="Build for stock 30-step base UNet without Pika LoRAs")
    pb.add_argument("--ref-images", default=None, help="Comma-separated reference images for ref2va prefix")
    pb.add_argument("--no-holds", action="store_true", help="Do not project out [[hold]] groups")
    pb.add_argument("--include-encoder", dest="include_encoder", action="store_true", default=True, help="Also save vector.encoder (5120-d; default: True)")
    pb.add_argument("--no-encoder", dest="include_encoder", action="store_false", help="Omit vector.encoder (5120-d) from saved .safetensors")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--device", default="default")
    pb.add_argument("--refiner-device", default="cpu")
    pb.add_argument("--out-dir", dest="out_dir", default=None, help="Optional output directory for .safetensors files")
    pb.add_argument("--overwrite", action="store_true", help="Overwrite existing .toml and .safetensors files")

    pfm = sub.add_parser(
        "from-md",
        aliases=["import-md"],
        help="Convert an LLM-authored control-vector Markdown file into a validated .toml pair file",
    )
    pfm.add_argument("md_path", help="Input Markdown (.md) file path")
    pfm.add_argument(
        "--out",
        default=None,
        help="Output .toml file path or output directory (defaults to prompt_bank/control_vectors/<stem>.toml)",
    )
    pfm.add_argument("--name", default=None, help="Optional vector name / file stem override")
    pfm.add_argument("--mode", choices=cv.VALID_MODES, default=None, help="Explicit mode override (auto-detected if omitted)")
    pfm.add_argument("--frame", choices=cv.VALID_FRAMES, default="scene", help="Pair frame ('scene' or 'bare')")
    pfm.add_argument("--quality", default=None, help="Optional vector.quality description override")
    pfm.add_argument("--a-side", dest="a_side", default=None, help="Optional vector.a_side description override")
    pfm.add_argument("--b-side", dest="b_side", default=None, help="Optional vector.b_side description override")
    pfm.add_argument("--test-prompt-out", dest="test_prompt_out", default=None, help="Optional path to write the extracted Test Scene Prompt text")
    pfm.add_argument("--overwrite", action="store_true", help="Overwrite an existing .toml file at the target path")

    pm = sub.add_parser("make", help="Build a control vector from a pair TOML file")
    pm.add_argument("pairs", help="Pair TOML file path or stem under prompt_bank/control_vectors/")
    pm.add_argument("--pooling", choices=cv.VALID_POOLING, default="diff_span")
    pm.add_argument("--unit-mean", dest="unit_mean", action="store_true", help="Average unit-normalized pair directions")
    pm.add_argument("--unet", default=None, help="Diffusion checkpoint filename")
    pm.add_argument("--encoder", default=None, help="Text encoder checkpoint filename")
    pm.add_argument("--lora", action="append", default=None, help="LoRA file[:strength]")
    pm.add_argument("--ref-images", default=None, help="Comma-separated reference images for ref2va prefix")
    pm.add_argument("--ref-role", default="subject", help="Role for first reference image")
    pm.add_argument("--no-holds", action="store_true", help="Do not project out [[hold]] groups")
    pm.add_argument("--include-encoder", dest="include_encoder", action="store_true", default=True, help="Also save vector.encoder (5120-d; default: True)")
    pm.add_argument("--no-encoder", dest="include_encoder", action="store_false", help="Omit vector.encoder (5120-d) from saved .safetensors")
    pm.add_argument("--output-name", default="", help="Optional vector stem override")
    pm.add_argument("--seed", type=int, default=0)
    pm.add_argument("--device", default="default")
    pm.add_argument("--refiner-device", default="cpu")
    pm.add_argument("--out", default=None, help="Output .safetensors path or filename")
    pm.add_argument("--overwrite", action="store_true", help="Overwrite an existing vector file at the target path")

    pi = sub.add_parser("inspect", help="Inspect a control vector file and optionally meter prompts")
    pi.add_argument("vector", help="Path or filename in models/control_vectors/")
    pi.add_argument(
        "--hold",
        action="append",
        default=None,
        help="Optional control vector .safetensors file(s) to project out before J-Lens readout",
    )
    pi.add_argument(
        "--hold-alpha",
        dest="hold_alpha",
        type=float,
        default=1.0,
        help="Soft or over-projected hold attenuation alpha in [0.0, 2.0] when --hold is passed (default: 1.0)",
    )
    pi.add_argument("--check-unet", default=None, help="Verify compatibility against a UNET checkpoint")
    pi.add_argument("--check-encoder", default=None, help="Verify compatibility against an encoder checkpoint")
    pi.add_argument("--check-lora", action="append", default=None)
    pi.add_argument("--allow-recipe-mismatch", action="store_true")
    pi.add_argument("--meter", action="append", default=None, help="Prompt-bank id or file path to meter")
    pi.add_argument("--meter-look-family", action="store_true", help="Meter t2va_look_anchor and all 8 look siblings (E2.8)")
    pi.add_argument("--device", default="default")
    pi.add_argument("--refiner-device", default="cpu")

    pg = sub.add_parser("grade", help="Grade activation/prediction movement against nulls and edit baselines")
    pg.add_argument("--vector", required=True, help="Vector .safetensors path or name")
    pg.add_argument("--prompt", required=True, help="Base prompt id or path")
    pg.add_argument("--edit-prompt", default=None, help="Edit baseline prompt id or path")
    pg.add_argument("--pairs", default=None, help="Pair TOML file to build flip_null")
    pg.add_argument("--strengths", default="0,0.5,1.0,2.0,-1.0")
    pg.add_argument("--probe-steps", default="0,8,15", help="Schedule step indices for DiT forward grading")
    pg.add_argument("--no-dit", action="store_true", help="Run post-refiner context movement only without loading the DiT")
    pg.add_argument("--sol", action="store_true", help="Opt in to MiniMaxH3Sol sparse attention during DiT grading (default: stock dense attention)")
    pg.add_argument("--compare-unmasked", action="store_true", help="Also evaluate mask='all_rows' for E2.3")
    pg.add_argument("--width", type=int, default=1152)
    pg.add_argument("--height", type=int, default=768)
    pg.add_argument("--length", type=int, default=h3_config.LENGTH)
    pg.add_argument("--noise-seed", type=int, default=730451892)
    pg.add_argument("--unet", default=None)
    pg.add_argument("--encoder", default=None)
    pg.add_argument("--lora", action="append", default=None, help="LoRA file[:strength] (defaults to vector recipe.loras)")
    pg.add_argument("--no-lora", action="store_true", help="Ignore vector recipe.loras during grading")
    pg.add_argument("--manual-sigmas", default=None, help="Comma-separated manual sigmas override")
    pg.add_argument("--pika", action="store_true", help="Use Pika H3 Flex 6-step manual sigmas and default probe steps 0,2,5")
    pg.add_argument("--ref-images", default=None)
    pg.add_argument("--ref-role", default="subject")
    pg.add_argument("--seed", type=int, default=0)
    pg.add_argument("--device", default="default")
    pg.add_argument("--refiner-device", default="cpu")
    pg.add_argument("--out", default=None, help="Optional JSON output path for grade report")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "ui":
            return cmd_ui(args)
        if args.cmd == "build":
            return cmd_build(args)
        if args.cmd in ("from-md", "import-md"):
            return cmd_from_md(args)
        if args.cmd == "make":
            return cmd_make(args)
        if args.cmd == "inspect":
            return cmd_inspect(args)
        if args.cmd == "grade":
            return cmd_grade(args)
    except (ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
