"""Thin Option-B compatibility wrappers for legacy & companion ComfyUI nodes
(`MiniMaxH3ApplyControlVectors`, `MiniMaxH3MakeControlVector`,
`MiniMaxH3LoadControlVector`, `MiniMaxH3LayerNormMatch`) preserving zero
`node_id_manifest.json` diff (`unit_mean=False` default, `[routing]`/`[provenance]` propagated)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import folder_paths
import torch
from comfy_api.latest import io

from . import nodes as _nodes


def make_control_vector_from_pairs(
    *,
    pair_path: str | Path | None = None,
    model: Any,
    clip: Any,
    pooling: str = "diff_span",
    apply_holds: bool = True,
    unit_mean: bool = False,
    include_encoder: bool = True,
    output_name: str = "",
    output_dir: str | Path | None = None,
    encoded_references: Any | None = None,
    overwrite: bool = False,
    made_by: str = "MiniMaxH3MakeControlVector/1",
    pair_set: Any | None = None,
    words: str = "",
    mode: str = "auto",
    target_prompt: str = "",
) -> tuple[Path, dict[str, Any]]:
    """Build and save a `.safetensors` control vector from `pair_path`, `pair_set`, or `words`."""
    cv_mod = _nodes._cv
    jl_mod = _nodes._j_lens
    pair_compiler = _nodes._pair_compiler
    if pooling not in cv_mod.VALID_POOLING:
        raise ValueError(f"pooling must be one of {cv_mod.VALID_POOLING}, got {pooling!r}")

    family = cv_mod.detect_family(model)
    resolved_pair_path: Path | None = None
    if pair_set is not None:
        pass
    elif isinstance(words, str) and words.strip():
        explicit_mode = str(mode or "auto").strip().lower()
        req_mode = (
            explicit_mode
            if explicit_mode and explicit_mode != "auto"
            else (
                "ref2va"
                if (encoded_references is not None or "ref2va" in str(family).lower())
                else ("p_h3_image" if "image" in str(family).lower() else None)
            )
        )
        tp_str = str(target_prompt or "").strip() or None
        raw_words = words.strip()
        compile_mode = pair_compiler.infer_compile_mode(
            req_mode,
            ref_images=encoded_references,
            target_prompt=tp_str,
            words=raw_words,
        )
        has_spec_header = any(
            raw_words.lower().startswith(p) or f"\n{p}" in raw_words.lower()
            for p in (
                "name:",
                "quality:",
                "a_side:",
                "b_side:",
                "mode:",
                "target_prompt:",
                "scene_prompt:",
                "target_weight:",
                "bank_scenes:",
            )
        )
        default_bs = 5 if (tp_str is None and compile_mode != "p_h3_image") else 0
        if has_spec_header:
            spec_body = (
                f"name: {output_name.strip()}\n{raw_words}"
                if (output_name.strip() and "name:" not in raw_words.lower())
                else raw_words
            )
            if req_mode and "mode:" not in raw_words.lower():
                spec_body = f"mode: {compile_mode}\n{spec_body}"
            spec_res = pair_compiler.parse_short_contrast_spec(
                spec_body,
                target_prompt=tp_str,
                bank_scenes=default_bs,
                bank_dir=_nodes.PROMPT_BANK_DIR,
            )
            pair_set = spec_res["pair_set"]
        else:
            pair_set = pair_compiler.compile_words_to_pairset(
                raw_words,
                name=output_name.strip() or None,
                mode=compile_mode,
                target_prompt=tp_str,
                bank_scenes=default_bs,
                bank_dir=_nodes.PROMPT_BANK_DIR,
            )
    else:
        if pair_path is None:
            raise ValueError("either pair_path, pair_set, or words must be provided")
        resolved_pair_path = _nodes.resolve_pair_file_path(pair_path)
        pair_set = cv_mod.load_pairs(resolved_pair_path, bank_dir=_nodes.PROMPT_BANK_DIR)

    recipe = _nodes.extract_model_recipe(model, clip)
    refiner_run = _nodes.build_refiner_runner(model)

    encoded_pairs, encoded_holds, enc_censuses = _nodes.encode_pair_set(
        clip, pair_set, encoded_references=encoded_references, with_holds=apply_holds
    )
    refined_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    ref_censuses: list[dict[str, float]] = []
    for enc in encoded_pairs:
        ref_a = refiner_run(enc["enc_a"])
        ref_b = refiner_run(enc["enc_b"])
        refined_pairs.append((ref_a, ref_b))
        ref_censuses.append(cv_mod.row_norm_census(ref_a, enc["tags_a"]))
        ref_censuses.append(cv_mod.row_norm_census(ref_b, enc["tags_b"]))
    refined_holds = [(refiner_run(h["enc_a"]), refiner_run(h["enc_b"])) for h in encoded_holds]

    pooled = cv_mod.pool_pair_set(
        encoded_pairs,
        refined_pairs,
        encoded_holds,
        refined_holds,
        pooling=pooling,
        unit_mean=unit_mean,
        include_encoder=include_encoder,
        min_erasure_pairs=(
            cv_mod.MIN_ERASURE_PAIRS
            if str(pair_set.path).startswith("<compiled:")
            else 1
        ),
    )
    a_mat, b_mat, w_vec = pooled.a_mat, pooled.b_mat, pooled.w_vec
    pair_spans = pooled.pair_spans
    final_vec = pooled.vector
    tensors_to_save: dict[str, torch.Tensor] = {}
    if pooled.hold_basis is not None:
        tensors_to_save["holds.post_refiner"] = pooled.hold_basis
    tensors_to_save["vector.post_refiner"] = final_vec
    cv_mod.attach_erasure_tensors(tensors_to_save, pooled)

    gen = torch.Generator(device="cpu").manual_seed(0)
    q_report = cv_mod.quality_report(a_mat, b_mat, w_vec, generator=gen)
    if jl_mod is not None and jl_mod.bundle_available():
        try:
            jl_readout = jl_mod.decode_vector(
                final_vec, space="post_refiner", family=family, top_k=12
            )
            if "erasure.vr" in tensors_to_save and "erasure.diag" in tensors_to_save:
                jl_readout["erasure_readout"] = jl_mod.decode_erasure_payload(
                    final_vec,
                    tensors_to_save["erasure.vr"],
                    tensors_to_save["erasure.diag"],
                    erasure_mu_d=tensors_to_save.get("erasure.mu_d"),
                    v1d_readout=jl_readout,
                    family=family,
                )
            jl_align = jl_mod.check_b_side_alignment(
                jl_readout,
                pair_set.b_side,
                a_side=pair_set.a_side,
                erasure_readout=jl_readout.get("erasure_readout"),
            )
            jl_readout["b_side_alignment"] = jl_align
            q_report["j_lens_readout"] = jl_readout
            _nodes.logger.info(
                "[h3] built control vector %r (%s): j_lens (+) %s | (-) %s",
                output_name.strip() or pair_set.name,
                family,
                jl_readout.get("pos_summary", ""),
                jl_readout.get("neg_summary", ""),
            )
            if jl_align.get("warning"):
                _nodes.logger.warning(
                    "[h3] control vector %r (%s) J-Lens b_side check: %s",
                    output_name.strip() or pair_set.name,
                    family,
                    jl_align["warning"],
                )
        except Exception:
            pass
    raw_norm = float(torch.linalg.vector_norm(final_vec.to(dtype=torch.float64)).item())
    median_ref_row_norm = float(
        sum(c["median_row_norm"] for c in ref_censuses) / len(ref_censuses)
    )
    median_enc_row_norm_ex0 = float(
        sum(c["median_row_norm_ex_row0"] for c in enc_censuses) / len(enc_censuses)
    )

    if encoded_references is not None:
        ref_hash = hashlib.sha256()
        for sp in getattr(encoded_references, "spans", ()):
            ref_hash.update(cv_mod._tensor_bf16_bytes(sp.rows))
        prefix_meta: Any = {"references_sha256": ref_hash.hexdigest()}
    else:
        prefix_meta = "none"

    if resolved_pair_path is not None:
        try:
            rel_pair_path = str(resolved_pair_path.relative_to(_nodes.REPO_ROOT))
        except ValueError:
            rel_pair_path = resolved_pair_path.name
    else:
        rel_pair_path = str(pair_set.path)

    stem = (output_name or pair_set.name).strip()
    if not cv_mod.NAME_RE.match(stem):
        raise ValueError(f"output_name must match [a-z0-9_]+, got {stem!r}")

    points_meta: dict[str, Any] = {
        "post_refiner": {
            "unit": "mean_pair_difference",
            "raw_norm": raw_norm,
            "median_row_norm": median_ref_row_norm,
        }
    }
    if include_encoder and pooled.enc_raw_vec is not None:
        enc_vec, enc_raw_norm = cv_mod.encoder_point(pooled.enc_raw_vec)
        if enc_vec is not None:
            tensors_to_save["vector.encoder"] = enc_vec
            points_meta["encoder"] = {
                "unit": "median_row_norm",
                "raw_norm": enc_raw_norm,
                "median_row_norm": median_enc_row_norm_ex0,
                "row0_excluded": True,
            }

    pairs_dict_node = cv_mod.pairs_record(
        pair_set,
        path=rel_pair_path,
        pair_spans=pair_spans,
        n_holds=len(pair_set.holds) if apply_holds else 0,
        inline=resolved_pair_path is None,
        words_spec=words if isinstance(words, str) else None,
        inline_holds=apply_holds,
    )

    metadata: dict[str, Any] = {
        "format": cv_mod.FORMAT_VERSION,
        "name": stem,
        "quality": pair_set.quality,
        "family": family,
        "pairs": pairs_dict_node,
        "recipe": recipe,
        "measured_under": {
            "mode": "ref2va" if encoded_references is not None else pair_set.mode,
            "frame": pair_set.frame,
            "prefix": prefix_meta,
            "pooling": pooling,
        },
        "points": points_meta,
        "quality_report": q_report,
        "provenance": {
            "core_revision": _nodes._substrate.git_head(_nodes.COMFY_ROOT) or "unknown",
            "pack_commit": _nodes._substrate.git_head(_nodes.REPO_ROOT) or "unknown",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "made_by": made_by,
        },
    }
    cv_mod.merge_toml_metadata_into_vector(metadata, pair_set)

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(folder_paths.get_folder_paths(_nodes.CONTROL_VECTORS_FOLDER)[0])
    )
    out_filename = cv_mod.default_vector_filename(
        stem,
        family,
        pooling=pooling,
        prefix=prefix_meta,
        apply_holds=apply_holds,
        n_holds=len(pair_set.holds),
        loras=recipe.get("loras"),
    )
    out_path = cv_mod.save_vector(
        target_dir / out_filename, tensors_to_save, metadata, overwrite=overwrite
    )

    e2_1_census = {
        "encoder_row0_norm_mean": float(
            sum(c["row0_norm"] for c in enc_censuses) / len(enc_censuses)
        ),
        "encoder_median_row_norm_ex_row0": median_enc_row_norm_ex0,
        "encoder_row0_to_median_ratio": float(
            sum(c["row0_to_median_ratio"] for c in enc_censuses) / len(enc_censuses)
        ),
        "refiner_row0_norm_mean": float(
            sum(c["row0_norm"] for c in ref_censuses) / len(ref_censuses)
        ),
        "refiner_median_row_norm": median_ref_row_norm,
        "refiner_row0_to_median_ratio": float(
            sum(c["row0_to_median_ratio"] for c in ref_censuses) / len(ref_censuses)
        ),
        "vector_channel_census": cv_mod.channel_census(final_vec),
    }

    full_report = {
        "vector_file": out_filename,
        "path": str(out_path),
        "metadata": metadata,
        "e2_1_census": e2_1_census,
        "pair_spans": pair_spans,
    }
    return out_path, full_report


class MiniMaxH3MakeControlVector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        ref_enc_mod = _nodes._repo("reference_encode")
        return io.Schema(
            node_id="MiniMaxH3MakeControlVector",
            display_name="MiniMax H3 Make Control Vector",
            category="model/conditioning/minimax",
            description=(
                "Measures a contrastive control vector from a pair TOML file under "
                "data/control_vectors/ or prompt_bank/control_vectors/ (or compiles pairs "
                "directly from 'words'), encoding each pair through the loaded Qwen3-VL "
                "encoder and the DiT's condition_proj + token_refiner, pooling the "
                "differing span rows ('diff_span' by default) in float64, projecting out "
                "any [[hold]] directions, and saving <name>__<family>.safetensors under "
                "models/control_vectors/."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.String.Input(
                    "pair_file",
                    default="film_16mm_1974.toml",
                    tooltip=(
                        "Pair TOML filename under data/control_vectors/ or "
                        "prompt_bank/control_vectors/ (for example 'film_16mm_1974.toml' "
                        "or 'lettering.toml'). Ignored when 'words' below is non-empty."
                    ),
                ),
                io.Combo.Input(
                    "pooling",
                    options=list(_nodes._cv.VALID_POOLING),
                    default="diff_span",
                    tooltip=(
                        "Which tag-1 rows after the token refiner are mean-pooled per pair side: "
                        "'diff_span' (default, pools only the differing clause [first_diff:end_diff]), "
                        "'from_first_diff', 'scene_text', or 'all_tag1'."
                    ),
                ),
                io.Boolean.Input(
                    "apply_holds",
                    default=True,
                    optional=True,
                    tooltip="Project out any [[hold]] group directions declared in the pair file.",
                ),
                io.String.Input(
                    "output_name",
                    default="",
                    optional=True,
                    tooltip=(
                        "Optional override for the vector stem; writes "
                        "<stem>__<family>.safetensors under models/control_vectors/."
                    ),
                ),
                ref_enc_mod.H3EncodedReferences.Input(
                    "encoded_references",
                    optional=True,
                    tooltip=(
                        "Optional encoded reference prefix from MiniMaxH3EncodeReferences; "
                        "when connected, pairs are encoded on top of the reference pass (E2.4)."
                    ),
                ),
                io.Boolean.Input(
                    "overwrite",
                    default=False,
                    optional=True,
                    tooltip=(
                        "When False (default), refuses to overwrite an existing vector file "
                        "of the same name; set True to replace it."
                    ),
                ),
                io.String.Input(
                    "words",
                    default="",
                    optional=True,
                    tooltip=(
                        "Optional in-place 'pole_a -> pole_b' or rotated 'a1 | a2 | a3 -> pole_b' "
                        "contrast(s) (or a bare target concept when target_prompt is blank, which "
                        "compiles across 5 rotated prompt_bank scenes) with zero .toml file required."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["auto", *_nodes._pair_compiler.SUPPORTED_FRAME_MODES],
                    default="auto",
                    optional=True,
                    tooltip=(
                        "Prompt mode for compiled words ('auto' infers from encoded_references, "
                        "target_prompt, words, and model family across t2va, fl2va, ref2va, and p_h3_image)."
                    ),
                ),
                io.String.Input(
                    "target_prompt",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip=(
                        "Optional target H3 scene prompt when compiling from 'words'. When provided, "
                        "pole_a or pole_b must match a substring in target_prompt; when left blank on "
                        "a video mode, compiles across 5 prompt_bank scenes with rotated Pole-A baselines."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="vector_name"),
                io.String.Output(display_name="report"),
                _nodes.H3ControlVector.Output(display_name="control_vector"),
            ],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        pair_file="film_16mm_1974.toml",
        words="",
        mode="auto",
        target_prompt="",
        **kwargs,
    ):
        if isinstance(words, str) and words.strip():
            return (
                "words",
                words.strip(),
                str(mode),
                str(target_prompt or "").strip(),
            )
        try:
            full = _nodes.resolve_pair_file_path(pair_file)
            st = os.stat(full)
            return (str(full), st.st_mtime_ns, st.st_size)
        except Exception:
            return None

    @classmethod
    def execute(
        cls,
        model,
        clip,
        pair_file="film_16mm_1974.toml",
        pooling="diff_span",
        apply_holds=True,
        output_name="",
        encoded_references=None,
        overwrite=False,
        words="",
        mode="auto",
        target_prompt="",
    ) -> io.NodeOutput:
        dm = model.get_model_object("diffusion_model")
        if not _nodes._is_minimax_h3(dm):
            raise ValueError("MiniMaxH3MakeControlVector requires a MiniMax H3 diffusion model")
        out_path, report = _nodes.make_control_vector_from_pairs(
            pair_path=pair_file,
            model=model,
            clip=clip,
            pooling=pooling,
            apply_holds=bool(apply_holds),
            output_name=output_name,
            encoded_references=encoded_references,
            overwrite=bool(overwrite),
            made_by="MiniMaxH3MakeControlVector/1",
            words=str(words or ""),
            mode=str(mode or "auto"),
            target_prompt=str(target_prompt or ""),
        )
        return io.NodeOutput(
            out_path.name,
            json.dumps(report, indent=2, sort_keys=True),
            _nodes.H3VectorSpec(kind="file", vector_name=out_path.name),
        )


class MiniMaxH3ApplyControlVectors(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ApplyControlVectors",
            display_name="MiniMax H3 Apply Control Vectors",
            category="model/conditioning/minimax/legacy",
            is_deprecated=True,
            description=(
                "Legacy all-in-one control vector node (kept for saved graphs and API workflows; "
                "for new canvas builds use MiniMaxH3Steer, MiniMaxH3WordVector, "
                "MiniMaxH3LoadControlVector, and MiniMaxH3ReleaseReferences). "
                "Applies a post-refiner control vector (<name>__<family>.safetensors under "
                "models/control_vectors/) to the refined text conditioning via a "
                "DIFFUSION_MODEL wrapper. Leaves vision rows (tag 0) bitwise untouched by "
                "default and keeps sequence length L unchanged. Multiple chained nodes sum "
                "their shifts in float64 before casting to model dtype."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "vector_name",
                    options=_nodes._vector_combo_options(),
                    default=_nodes._APPLY_DEFAULTS["vector_name"],
                    tooltip=(
                        "Control vector file (<name>__<family>.safetensors) under "
                        "models/control_vectors/, or 'none'."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=_nodes._APPLY_DEFAULTS["strength"],
                    min=-20.0,
                    max=20.0,
                    step=0.05,
                    tooltip=(
                        "Multiplier in the vector point's unit (after the refiner, 1.0 is one "
                        "mean pair difference for 'add', 67.5 post-refiner shift norm for "
                        "'add_norm_matched', full projection removal for 'project_out', "
                        "or +100% component scaling for 'amplify'; 0.0 is an exact no-op)."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=list(_nodes._cv.APPLY_MODES),
                    default=_nodes._APPLY_DEFAULTS["mode"],
                    optional=True,
                    tooltip=(
                        "'add_norm_matched' (default) normalizes v to 67.5 post-refiner shift "
                        "norm at strength=1.0 across all pooling modes; 'add' adds strength * v "
                        "in raw mean-pair-difference units; 'hold' projects out strength * "
                        "(delta_r . u_hat) * u_hat from the accumulated additive steering shift "
                        "without modifying the base prompt; 'project_out' removes strength * "
                        "((r . v) / ||v||^2) * v from the prompt tokens (E4.4); 'amplify' adds "
                        "strength * ((r . v) / ||v||^2) * v."
                    ),
                ),
                io.Combo.Input(
                    "mask",
                    options=list(_nodes.MASK_MODES),
                    default=_nodes._APPLY_DEFAULTS["mask"],
                    tooltip=(
                        "'scene_only' (default) steers all visual prose (look_only | description_only), "
                        "'look_only', 'description_only', 'soundscape_only', 'non_vision', 'after_dialogue', "
                        "or 'all_rows'."
                    ),
                ),
                io.Float.Input(
                    "start_percent",
                    default=_nodes._APPLY_DEFAULTS["start_percent"],
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip="Start of the active step window as step_idx / N_steps.",
                ),
                io.Float.Input(
                    "end_percent",
                    default=_nodes._APPLY_DEFAULTS["end_percent"],
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip="End of the active step window as step_idx / N_steps.",
                ),
                io.Boolean.Input(
                    "allow_recipe_mismatch",
                    default=_nodes._APPLY_DEFAULTS["allow_recipe_mismatch"],
                    optional=True,
                    tooltip="When True (default), logs checkpoint or LoRA recipe differences and applies.",
                ),
                io.Float.Input(
                    "video_scale",
                    default=_nodes._APPLY_DEFAULTS["video_scale"],
                    min=-5.0,
                    max=5.0,
                    step=0.05,
                    optional=True,
                    tooltip="Multiplies this vector's strength for the video stream.",
                ),
                io.Float.Input(
                    "audio_scale",
                    default=_nodes._APPLY_DEFAULTS["audio_scale"],
                    min=-5.0,
                    max=5.0,
                    step=0.05,
                    optional=True,
                    tooltip="Multiplies this vector's strength for the audio stream.",
                ),
                io.Float.Input(
                    "split_end_percent",
                    default=_nodes._APPLY_DEFAULTS["split_end_percent"],
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip="End of the step window [0.0, split_end_percent) in which video_scale and audio_scale apply.",
                ),
                io.Float.Input(
                    "ref_vae_end_percent",
                    default=_nodes._APPLY_DEFAULTS["ref_vae_end_percent"],
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip="Fraction of steps [0.0, ref_vae_end_percent) where clean visual cond rows remain packed.",
                ),
                io.Float.Input(
                    "ref_vae_aug",
                    default=_nodes._APPLY_DEFAULTS["ref_vae_aug"],
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    optional=True,
                    tooltip="Cleanliness share of visual condition rows (default 0.999).",
                ),
                io.String.Input(
                    "words",
                    default=_nodes._APPLY_DEFAULTS["words"],
                    optional=True,
                    tooltip="Optional inline positive words synthesized on CPU via J-Lens.",
                ),
                io.String.Input(
                    "against_words",
                    default=_nodes._APPLY_DEFAULTS["against_words"],
                    optional=True,
                    tooltip="Optional negative words when 'words' is used.",
                ),
                io.String.Input(
                    "hold_words",
                    default=_nodes._APPLY_DEFAULTS["hold_words"],
                    optional=True,
                    tooltip="Optional nuisance words to project out on CPU.",
                ),
                io.Float.Input(
                    "hold_alpha",
                    default=_nodes._APPLY_DEFAULTS["hold_alpha"],
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    optional=True,
                    tooltip="Attenuation strength alpha in [0.0, 2.0] for 'hold_words' (default 1.0).",
                ),
                io.MultiCombo.Input(
                    "fields",
                    options=list(_nodes._spans.PROMPT_FIELDS),
                    default=[],
                    optional=True,
                    tooltip="Tick one or more fields of the prompt to steer exactly those and override 'mask'.",
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        vector_name=_nodes._APPLY_DEFAULTS["vector_name"],
        words=_nodes._APPLY_DEFAULTS["words"],
        against_words=_nodes._APPLY_DEFAULTS["against_words"],
        hold_words=_nodes._APPLY_DEFAULTS["hold_words"],
        hold_alpha=_nodes._APPLY_DEFAULTS["hold_alpha"],
        **kwargs,
    ):
        raw = str(vector_name).strip()
        w_str = str(words or "").strip()
        h_str = str(hold_words or "").strip()
        vec_fp: Any = "none"
        if raw and raw != "none":
            resolved_stats: list[tuple[str, int, int]] = []
            for pref_fam in (None, "fl2va", "ref2va", "h3_image"):
                try:
                    full = _nodes.resolve_vector_file_path(
                        raw, preferred_family=pref_fam, log_swap=False
                    )
                    st = os.stat(full)
                    entry = (str(full), st.st_mtime_ns, st.st_size)
                    if entry not in resolved_stats:
                        resolved_stats.append(entry)
                except Exception:
                    continue
            vec_fp = (
                resolved_stats[0]
                if len(resolved_stats) == 1
                else (tuple(resolved_stats) if resolved_stats else raw)
            )
        if not w_str and not h_str:
            return vec_fp
        return (
            vec_fp,
            "words",
            w_str,
            str(against_words or "").strip(),
            h_str,
            float(hold_alpha),
        )

    @classmethod
    def validate_inputs(
        cls,
        vector_name=_nodes._APPLY_DEFAULTS["vector_name"],
        strength=_nodes._APPLY_DEFAULTS["strength"],
        mode=_nodes._APPLY_DEFAULTS["mode"],
        mask=_nodes._APPLY_DEFAULTS["mask"],
        start_percent=_nodes._APPLY_DEFAULTS["start_percent"],
        end_percent=_nodes._APPLY_DEFAULTS["end_percent"],
        allow_recipe_mismatch=_nodes._APPLY_DEFAULTS["allow_recipe_mismatch"],
        video_scale=_nodes._APPLY_DEFAULTS["video_scale"],
        audio_scale=_nodes._APPLY_DEFAULTS["audio_scale"],
        split_end_percent=_nodes._APPLY_DEFAULTS["split_end_percent"],
        ref_vae_end_percent=_nodes._APPLY_DEFAULTS["ref_vae_end_percent"],
        ref_vae_aug=_nodes._APPLY_DEFAULTS["ref_vae_aug"],
        words=_nodes._APPLY_DEFAULTS["words"],
        against_words=_nodes._APPLY_DEFAULTS["against_words"],
        hold_words=_nodes._APPLY_DEFAULTS["hold_words"],
        hold_alpha=_nodes._APPLY_DEFAULTS["hold_alpha"],
        **kwargs,
    ) -> bool | str:
        raw = str(vector_name).strip()
        if raw and raw != "none":
            try:
                _nodes.resolve_vector_file_path(raw)
            except Exception as exc:
                return str(exc)
        picked = kwargs.get("fields")
        if picked:
            try:
                _nodes._spans.fields_mask(picked if isinstance(picked, (list, tuple)) else [picked])
            except ValueError as exc:
                return str(exc)
        w_str = str(words or "").strip()
        h_str = str(hold_words or "").strip()
        if h_str and (not raw or raw == "none") and not w_str:
            return (
                "hold_words requires vector_name != 'none' or non-empty words; "
                "cannot apply hold_words when no control vector or inline words are selected"
            )
        jl_mod = _nodes._j_lens
        if w_str or h_str:
            try:
                ha_val = float(hold_alpha)
                if not math.isfinite(ha_val) or not (0.0 <= ha_val <= 2.0):
                    return f"hold_alpha must be in [0.0, 2.0], got {hold_alpha!r}"
                if jl_mod.bundle_available():
                    if w_str:
                        jl_mod.synthesize_from_words(
                            w_str,
                            str(against_words or "").strip() or None,
                            family="fl2va",
                            hold_words=h_str or None,
                            alpha=ha_val,
                        )
                    elif h_str:
                        jl_mod.synthesize_from_words(
                            h_str,
                            family="fl2va",
                        )
            except Exception as exc:
                return str(exc)
        return True

    @classmethod
    def execute(
        cls,
        model,
        vector_name=_nodes._APPLY_DEFAULTS["vector_name"],
        strength=_nodes._APPLY_DEFAULTS["strength"],
        mode=_nodes._APPLY_DEFAULTS["mode"],
        mask=_nodes._APPLY_DEFAULTS["mask"],
        start_percent=_nodes._APPLY_DEFAULTS["start_percent"],
        end_percent=_nodes._APPLY_DEFAULTS["end_percent"],
        allow_recipe_mismatch=_nodes._APPLY_DEFAULTS["allow_recipe_mismatch"],
        video_scale=_nodes._APPLY_DEFAULTS["video_scale"],
        audio_scale=_nodes._APPLY_DEFAULTS["audio_scale"],
        split_end_percent=_nodes._APPLY_DEFAULTS["split_end_percent"],
        ref_vae_end_percent=_nodes._APPLY_DEFAULTS["ref_vae_end_percent"],
        ref_vae_aug=_nodes._APPLY_DEFAULTS["ref_vae_aug"],
        words=_nodes._APPLY_DEFAULTS["words"],
        against_words=_nodes._APPLY_DEFAULTS["against_words"],
        hold_words=_nodes._APPLY_DEFAULTS["hold_words"],
        hold_alpha=_nodes._APPLY_DEFAULTS["hold_alpha"],
        fields=[],  # noqa: B006 — read-only default matching io.MultiCombo schema default []
    ) -> io.NodeOutput:
        if fields:
            mask = _nodes._spans.fields_mask(fields if isinstance(fields, (list, tuple)) else [fields])
        m = _nodes.attach_control_vector(
            model,
            vector_name,
            strength=float(strength),
            mode=mode,
            mask=mask,
            start_percent=float(start_percent),
            end_percent=float(end_percent),
            allow_recipe_mismatch=bool(allow_recipe_mismatch),
            video_scale=float(video_scale),
            audio_scale=float(audio_scale),
            split_end_percent=float(split_end_percent),
            ref_vae_end_percent=float(ref_vae_end_percent),
            ref_vae_aug=float(ref_vae_aug),
            words=str(words or ""),
            against_words=str(against_words or ""),
            hold_words=str(hold_words or ""),
            hold_alpha=float(hold_alpha),
        )
        return io.NodeOutput(m)


class MiniMaxH3LoadControlVector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LoadControlVector",
            display_name="MiniMax H3 Load Control Vector",
            category="model/conditioning/minimax",
            description=(
                "Loads a saved control vector stem from models/control_vectors/ as a family-agnostic "
                "handle for MiniMaxH3Steer (custom_vector or hold_vector). Family (fl2va / ref2va) "
                "and LoRA sibling resolution happen automatically when wired to the model in MiniMaxH3Steer."
            ),
            inputs=[
                io.Combo.Input(
                    "vector",
                    options=_nodes._vector_stem_options(include_none=True),
                    default="none",
                    tooltip=(
                        "Saved control vector stem under models/control_vectors/ (for example "
                        "'film_16mm_1974' or 'noir'). Select a stem other than 'none'."
                    ),
                ),
                io.String.Input(
                    "hold_words",
                    default="",
                    optional=True,
                    tooltip="Optional comma-separated single-token nuisance words to project out on CPU (<2 ms).",
                ),
                io.Float.Input(
                    "hold_alpha",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    optional=True,
                    tooltip="Projection attenuation weight in [0.0, 2.0] for hold_words and any wired hold_vector.",
                ),
                _nodes.H3ControlVector.Input(
                    "hold_vector",
                    optional=True,
                    tooltip="Optional control vector or word vector to project out alongside this vector.",
                ),
            ],
            outputs=[_nodes.H3ControlVector.Output(display_name="control_vector")],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        vector="none",
        hold_words="",
        hold_alpha=1.0,
        **kwargs,
    ):
        vec_fp = _nodes._vector_file_fingerprint(str(vector or ""))
        h_str = str(hold_words or "").strip()
        if not h_str:
            return vec_fp
        return (vec_fp, h_str, float(hold_alpha))

    @classmethod
    def validate_inputs(
        cls,
        vector="none",
        hold_words="",
        hold_alpha=1.0,
        **kwargs,
    ) -> bool | str:
        raw = str(vector or "").strip()
        if not raw or raw == "none":
            return "MiniMaxH3LoadControlVector requires selecting a control vector (got 'none')"
        jl_mod = _nodes._j_lens
        try:
            _nodes.resolve_vector_file_path(raw, log_swap=False)
            ha_val = _nodes._validate_float_bounds("hold_alpha", hold_alpha, lo=0.0, hi=2.0)
            h_str = str(hold_words or "").strip()
            if h_str and jl_mod.bundle_available():
                jl_mod.synthesize_from_words(h_str, family="fl2va")
            _ = ha_val
        except Exception as exc:
            return str(exc)
        return True

    @classmethod
    def execute(
        cls,
        vector="none",
        hold_words="",
        hold_alpha=1.0,
        hold_vector=None,
    ) -> io.NodeOutput:
        raw = str(vector or "").strip()
        if not raw or raw == "none":
            raise ValueError(
                "MiniMaxH3LoadControlVector requires selecting a control vector (got 'none')"
            )
        _nodes.resolve_vector_file_path(raw, log_swap=False)
        ha_f = _nodes._validate_float_bounds("hold_alpha", hold_alpha, lo=0.0, hi=2.0)
        h_str = str(hold_words or "").strip()
        jl_mod = _nodes._j_lens
        if h_str and jl_mod.bundle_available():
            jl_mod.synthesize_from_words(h_str, family="fl2va")
        if hold_vector is not None and not isinstance(hold_vector, _nodes.H3VectorSpec):
            raise ValueError(
                f"hold_vector must be an H3VectorSpec, got {type(hold_vector).__name__}"
            )
        holds = ((hold_vector, ha_f),) if hold_vector is not None else ()
        return io.NodeOutput(
            _nodes.H3VectorSpec(
                kind="file",
                vector_name=raw,
                hold_words=h_str,
                hold_alpha=ha_f,
                holds=holds,
            )
        )


def match_conditioning_row_norm(
    conditioning: list[Any],
    *,
    reference_conditioning: list[Any] | None = None,
    target_median_row_norm: float = 0.0,
) -> list[Any]:
    """Scale tag-1 text rows of `conditioning` so their median L2 norm (excluding row 0) matches target."""
    if not isinstance(conditioning, list) or not conditioning:
        raise ValueError("conditioning must be a non-empty list")

    cv_mod = _nodes._cv
    target_norm = float(target_median_row_norm)
    if reference_conditioning is not None:
        if not isinstance(reference_conditioning, list) or not reference_conditioning:
            raise ValueError("reference_conditioning must be a non-empty list when provided")
        ref_t = reference_conditioning[0][0]
        ref_d = reference_conditioning[0][1] if len(reference_conditioning[0]) > 1 else {}
        ref_tags = ref_d.get("minimax_token_tags") if isinstance(ref_d, dict) else None
        target_norm = cv_mod.row_norm_census(ref_t, ref_tags)["median_row_norm_ex_row0"]

    if not math.isfinite(target_norm) or target_norm <= 0.0:
        raise ValueError(
            "MiniMaxH3LayerNormMatch requires either reference_conditioning or target_median_row_norm > 0"
        )

    out_cond: list[Any] = []
    for entry in conditioning:
        t = entry[0]
        d = dict(entry[1]) if len(entry) > 1 and isinstance(entry[1], dict) else {}
        tags = d.get("minimax_token_tags")
        cur_norm = cv_mod.row_norm_census(t, tags)["median_row_norm_ex_row0"]
        if not math.isfinite(cur_norm) or cur_norm <= 0.0:
            raise ValueError("conditioning has zero or non-finite median tag-1 row norm")
        scale = target_norm / cur_norm
        t_out = t.clone()
        if tags is not None:
            t_tags = torch.as_tensor(tags, device=t.device, dtype=torch.long).flatten()
            if t_tags.numel() != t.shape[1]:
                raise ValueError(
                    f"minimax_token_tags length {t_tags.numel()} != conditioning length {t.shape[1]}"
                )
            mask_2d = (t_tags != 0).unsqueeze(0).expand(t.shape[0], -1)
        else:
            mask_2d = torch.ones((t.shape[0], t.shape[1]), dtype=torch.bool, device=t.device)
        t_out[mask_2d] = (t_out[mask_2d].to(dtype=torch.float64) * scale).to(dtype=t.dtype)
        out_cond.append([t_out, d])
    return out_cond


class MiniMaxH3LayerNormMatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LayerNormMatch",
            display_name="MiniMax H3 Layer Norm Match",
            category="model/conditioning/minimax",
            description=(
                "Scales the tag-1 text rows of an earlier encoder layer's conditioning "
                "(from CLIPSetLastLayer, E4.3) so their median L2 norm (excluding row 0) "
                "matches the layer-50 reference conditioning or an explicit target norm, "
                "leaving vision rows (tag 0), tags, and sequence length L untouched."
            ),
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Conditioning.Input(
                    "reference_conditioning",
                    optional=True,
                    tooltip=(
                        "Optional layer-50 conditioning whose tag-1 median row norm "
                        "(excluding row 0) sets the target norm."
                    ),
                ),
                io.Float.Input(
                    "target_median_row_norm",
                    default=0.0,
                    min=0.0,
                    max=100000.0,
                    step=0.1,
                    optional=True,
                    tooltip=(
                        "Explicit target median tag-1 row norm (excluding row 0); used "
                        "when reference_conditioning is not connected."
                    ),
                ),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    def execute(
        cls,
        conditioning,
        reference_conditioning=None,
        target_median_row_norm=0.0,
    ) -> io.NodeOutput:
        matched = _nodes.match_conditioning_row_norm(
            conditioning,
            reference_conditioning=reference_conditioning,
            target_median_row_norm=float(target_median_row_norm),
        )
        return io.NodeOutput(matched)


for _sym in (
    "MiniMaxH3ApplyControlVectors",
    "MiniMaxH3LayerNormMatch",
    "MiniMaxH3LoadControlVector",
    "MiniMaxH3MakeControlVector",
    "make_control_vector_from_pairs",
    "match_conditioning_row_norm",
):
    setattr(_nodes, _sym, globals()[_sym])
