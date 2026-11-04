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
from pathlib import Path
import sys
import time
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
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
        lpath = folder_paths.get_full_path_or_raise("loras", fname)
        out.append(
            {
                "file": Path(fname).name,
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
    if loras:
        raise ValueError(
            "bench/control_vectors.py make: --lora requires an in-server ModelPatcher; "
            "pass no --lora for base checkpoint extraction"
        )

    family = cv.detect_family(unet_path)
    recipe = cv_nodes.extract_model_recipe(
        None,
        None,
        unet_path_override=unet_path,
        encoder_path_override=encoder_path,
        loras_override=loras,
    )

    t0 = time.perf_counter()
    clip = _load_clip(encoder_name, device=args.device)
    refiner_run = cv_nodes.build_refiner_runner(unet_path, device=args.refiner_device)

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
    print(f"  encoder     : {recipe['encoder']['file']} ({recipe['encoder']['sha256'][:12]})")

    a_pooled_list: list[torch.Tensor] = []
    b_pooled_list: list[torch.Tensor] = []
    enc_a_pooled_list: list[torch.Tensor] = []
    enc_b_pooled_list: list[torch.Tensor] = []
    weights_list: list[float] = []
    enc_censuses: list[dict[str, float]] = []
    ref_censuses: list[dict[str, float]] = []

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

        first_diff = h3_spans.first_differing_index(ids_a, ids_b)
        spans_a = h3_spans.spans(ids_a, tags_a.tolist(), dec_a)
        spans_b = h3_spans.spans(ids_b, tags_b.tolist(), dec_b)
        mask_a = h3_spans.pooling_mask(
            spans_a, args.pooling, first_diff=min(first_diff, len(ids_a) - 1)
        )
        mask_b = h3_spans.pooling_mask(
            spans_b, args.pooling, first_diff=min(first_diff, len(ids_b) - 1)
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
        print(
            f"  [{idx:02d}] w={item.weight:.2f} first_diff_tok={first_diff:3d} "
            f"L_a={len(ids_a):3d} L_b={len(ids_b):3d} | "
            f"-{item.a_span[:44]!r} -> +{item.b_span[:44]!r}"
        )

    a_mat = torch.stack(a_pooled_list, dim=0)
    b_mat = torch.stack(b_pooled_list, dim=0)
    w_vec = torch.tensor(weights_list, dtype=torch.float32)

    raw_vec = cv.mean_difference(a_mat, b_mat, w_vec)
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
            h_diff = h3_spans.first_differing_index(h_ids_a, h_ids_b)
            h_spans_a = h3_spans.spans(h_ids_a, h_tags_a.tolist(), h_dec_a)
            h_spans_b = h3_spans.spans(h_ids_b, h_tags_b.tolist(), h_dec_b)
            h_mask_a = h3_spans.pooling_mask(
                h_spans_a, args.pooling, first_diff=min(h_diff, len(h_ids_a) - 1)
            )
            h_mask_b = h3_spans.pooling_mask(
                h_spans_b, args.pooling, first_diff=min(h_diff, len(h_ids_b) - 1)
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

    print("\n--- Agreement & Norm Statistics ---")
    print(f"  raw_norm             : {raw_norm:.6f}")
    print(f"  mean_pairwise_cosine : {q_report['mean_pairwise_cosine']:.4f}")
    print(f"  split_half_cosine    : {q_report['split_half_cosine']:.4f}")
    print(
        f"  per_pair_cosine      : min={min(q_report['per_pair_cosine']):.4f}  "
        f"max={max(q_report['per_pair_cosine']):.4f}"
    )

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute() and len(out_path.parts) == 1:
            out_dir = Path(folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
            out_path = out_dir / out_path
    else:
        out_dir = Path(folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
        safe_family = family.replace(":", "_")
        out_path = out_dir / f"{stem}__{safe_family}.safetensors"

    saved = cv.save_vector(out_path, tensors_to_save, metadata)
    elapsed = time.perf_counter() - t0
    print(f"\n  saved       : {saved.name} ({saved.stat().st_size} bytes, {elapsed:.2f}s)")
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
        refiner_run = cv_nodes.build_refiner_runner(unet_path, device=args.refiner_device)
        pooling = metadata["measured_under"].get("pooling", "scene_text")
        if pooling == "from_first_diff":
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
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Clone `model` with an in-memory `(vec, strength, mask)` attached via
    `control_vector_nodes._make_control_vector_wrapper` on `WrappersMP.DIFFUSION_MODEL`."""
    import comfy.patcher_extension as pe

    m = model.clone()
    spec = cv_nodes._AppliedVectorSpec(
        vector_name=name,
        vector=vec.detach().to(device="cpu", dtype=torch.float32),
        strength=float(strength),
        mask=mask,
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

    clip = _load_clip(enc_name, device=args.device)
    refiner_run = cv_nodes.build_refiner_runner(unet_path, device=args.refiner_device)

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
            fdiff = h3_spans.first_differing_index(ia, ib)
            ma = h3_spans.pooling_mask(
                h3_spans.spans(ia, ta.tolist(), da), pooling, first_diff=min(fdiff, len(ia) - 1)
            )
            mb = h3_spans.pooling_mask(
                h3_spans.spans(ib, tb.tolist(), db), pooling, first_diff=min(fdiff, len(ib) - 1)
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
        "core_revision": substrate.git_head(cv_nodes.COMFY_ROOT) or "unknown",
        "pack_commit": substrate.git_head(REPO_ROOT) or "unknown",
        "attention": "sol" if getattr(args, "sol", False) else "stock_dense",
    }
    if not getattr(args, "no_dit", False):
        probe_steps = [int(x.strip()) for x in args.probe_steps.split(",") if x.strip()]
        dit_runner = getattr(args, "_dit_runner", None)
        if dit_runner is None:
            import comfy.model_management as mm
            import comfy.sample
            import comfy.samplers
            import comfy.sd
            from comfy_extras.nodes_minimax_h3 import _empty_av_latent

            print(
                f"\n[grade] loading DiT {unet_path.name} "
                f"(core={dit_provenance['core_revision']}, attn={dit_provenance['attention']}) "
                f"for denoised-prediction grading..."
            )
            raw_model = comfy.sd.load_diffusion_model(str(unet_path))
            base_model = _sigma_shifted(raw_model)
            if getattr(args, "sol", False):
                sol_mod = cv_nodes._repo("sol_attn_h3")
                sol_cfg = h3_config.sol_for_graph(False, h3_config.SAMPLING["steps"])
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
            sigmas = comfy.samplers.calculate_sigmas(
                base_model.get_model_object("model_sampling"),
                h3_config.SAMPLING["scheduler"],
                h3_config.SAMPLING["steps"],
            )

            def _default_dit_runner(m_obj: Any, cond_obj: list[Any], step_idx: int) -> tuple[float, Any]:
                return (
                    float(sigmas[step_idx]),
                    _one_forward(m_obj, cond_obj, noise, latent, sigmas, step_idx, args.noise_seed),
                )

            dit_runner = _default_dit_runner
        else:
            base_model = getattr(args, "_base_model")

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("make", help="Build a control vector from a pair TOML file")
    pm.add_argument("pairs", help="Pair TOML file path or stem under prompt_bank/control_vectors/")
    pm.add_argument("--pooling", choices=cv.VALID_POOLING, default="scene_text")
    pm.add_argument("--unet", default=None, help="Diffusion checkpoint filename")
    pm.add_argument("--encoder", default=None, help="Text encoder checkpoint filename")
    pm.add_argument("--lora", action="append", default=None, help="LoRA file[:strength]")
    pm.add_argument("--ref-images", default=None, help="Comma-separated reference images for ref2va prefix")
    pm.add_argument("--ref-role", default="subject", help="Role for first reference image")
    pm.add_argument("--no-holds", action="store_true", help="Do not project out [[hold]] groups")
    pm.add_argument("--include-encoder", action="store_true", help="Also save vector.encoder (5120-d)")
    pm.add_argument("--output-name", default="", help="Optional vector stem override")
    pm.add_argument("--seed", type=int, default=0)
    pm.add_argument("--device", default="default")
    pm.add_argument("--refiner-device", default="cpu")
    pm.add_argument("--out", default=None, help="Output .safetensors path or filename")

    pi = sub.add_parser("inspect", help="Inspect a control vector file and optionally meter prompts")
    pi.add_argument("vector", help="Path or filename in models/control_vectors/")
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
        if args.cmd == "make":
            return cmd_make(args)
        if args.cmd == "inspect":
            return cmd_inspect(args)
        if args.cmd == "grade":
            return cmd_grade(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
