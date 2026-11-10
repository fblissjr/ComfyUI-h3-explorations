"""Inspection, prompt metering (`inspect`), and DiT velocity grading (`grade`) for H3 control vectors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    LOOK_FAMILY_IDS,
    REPO_ROOT,
    _cosine,
    _parse_lora_specs,
    _rel_l2,
    _resolve_prompt_text,
)

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
        hold_words_arg = str(getattr(args, "hold_words", "") or "").strip() or None
        prompt_arg = str(getattr(args, "prompt", "") or "").strip() or None
        hold_alpha = float(getattr(args, "hold_alpha", 1.0))
        jl_rep = j_lens.decode_safetensors(
            vpath,
            holds=hold_paths if hold_paths else None,
            hold_words=hold_words_arg,
            alpha=hold_alpha,
            prompt=prompt_arg,
        )
        pr_jl = jl_rep["post_refiner"]
        if jl_rep.get("holds"):
            print(
                f"  j_lens_holds    : {jl_rep['holds']} (alpha={hold_alpha:.2f}, "
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
        rd_jl = pr_jl.get("readiness") or {}
        if rd_jl:
            print(f"  j_lens readiness: {rd_jl.get('summary', '')}")
        pc_jl = pr_jl.get("prompt_coupling") or {}
        if pc_jl:
            print(f"  prompt coupling : {pc_jl.get('summary', '')}")
            for rec_line in pc_jl.get("recommendations", ()):
                print(f"                    -> {rec_line}")

    check_dit_arg = getattr(args, "check_dit", None) or getattr(args, "check_unet", None)
    if check_dit_arg:
        dit_path = Path(_h.folder_paths.get_full_path_or_raise("diffusion_models", check_dit_arg))
        fam = cv.detect_family(dit_path)
        enc_name = args.check_encoder or metadata["recipe"]["encoder"]["file"]
        enc_path = (
            Path(_h.folder_paths.get_full_path_or_raise("text_encoders", enc_name))
            if enc_name != "unspecified"
            else None
        )
        runtime_recipe = cv_nodes.extract_model_recipe(
            None,
            None,
            dit_path_override=dit_path,
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
        clip = _h._load_clip(enc_name, device=args.device)
        dit_name = check_dit_arg or metadata["recipe"]["checkpoint"]["file"]
        dit_path = Path(_h.folder_paths.get_full_path_or_raise("diffusion_models", dit_name))
        meter_loras = (
            _parse_lora_specs(args.check_lora)
            if args.check_lora
            else list(metadata["recipe"].get("loras", ()))
        )
        refiner_run = cv_nodes.build_refiner_runner(
            dit_path, device=args.refiner_device, loras=meter_loras
        )
        pooling = metadata["measured_under"].get("pooling", "scene_text")
        if pooling in ("from_first_diff", "diff_span"):
            pooling = "scene_text"

        print(f"\n--- Meter Projection along {metadata['name']} (pooling={pooling}) ---")
        rows: list[tuple[str, float, float, float, int, float, float, list[tuple[str, float]]]] = []
        v64_unit = torch.nn.functional.normalize(vec.detach().to(device="cpu", dtype=torch.float64), dim=-1)
        for target in meter_targets:
            label, text = _resolve_prompt_text(target)
            enc_rows, tags, tids, dec_fn = cv_nodes.encode_prompt_for_vector(clip, text)
            ref_rows = refiner_run(enc_rows)
            sp = h3_spans.spans(tids, tags.tolist(), dec_fn)
            mask = h3_spans.pooling_mask(sp, pooling)
            m = cv.meter_score(ref_rows, vec, mask=mask)
            r64_cpu = ref_rows.detach().to(device="cpu", dtype=torch.float64)
            r64_unit = torch.nn.functional.normalize(r64_cpu, dim=-1)
            tok_cos_all = (r64_unit @ v64_unit).tolist()
            m_bool = mask.flatten().to(device="cpu", dtype=torch.bool)
            active_idxs = [i for i, flag in enumerate(m_bool.tolist()) if flag]
            active_cos = [tok_cos_all[i] for i in active_idxs] if active_idxs else tok_cos_all
            max_tcos = max(active_cos) if active_cos else 0.0
            p90_tcos = (
                float(torch.quantile(torch.tensor(active_cos, dtype=torch.float64), 0.90).item())
                if active_cos
                else 0.0
            )
            scored_toks: list[tuple[str, float]] = []
            for idx_t in active_idxs:
                tok_str = dec_fn([int(tids[idx_t])]).strip() if idx_t < len(tids) else ""
                if len(tok_str) >= 2 and any(ch.isalnum() for ch in tok_str):
                    scored_toks.append((tok_str, float(tok_cos_all[idx_t])))
            scored_toks.sort(key=lambda x: x[1], reverse=True)
            rows.append(
                (
                    label,
                    m["projection_units"],
                    m["signed_projection"],
                    m["cosine"],
                    int(mask.sum().item()),
                    max_tcos,
                    p90_tcos,
                    scored_toks[:5],
                )
            )
        anchor_units = rows[0][1] if rows else 0.0
        for label, p_units, s_proj, cos_val, n_tok, max_tc, p90_tc, top_toks in rows:
            delta_u = p_units - anchor_units
            top_str = ", ".join(f"{w} ({c:+.3f})" for w, c in top_toks)
            print(
                f"  {label:<36s} units={p_units:+8.4f} (delta={delta_u:+8.4f})  "
                f"proj={s_proj:+8.4f}  cos={cos_val:+7.4f}  p90_tok={p90_tc:+6.3f}  max_tok={max_tc:+6.3f}  tokens={n_tok:3d}"
            )
            if top_str:
                print(f"      top_receptors: {top_str}")
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
    mask: str = "scene_only",
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
    dit_name = getattr(args, "dit", None) or getattr(args, "unet", None) or metadata["recipe"]["checkpoint"]["file"]
    dit_path = Path(_h.folder_paths.get_full_path_or_raise("diffusion_models", dit_name))
    if getattr(args, "no_lora", False):
        grade_loras: list[dict[str, Any]] = []
    elif getattr(args, "lora", None):
        grade_loras = _parse_lora_specs(args.lora)
    else:
        grade_loras = list(metadata["recipe"].get("loras", ()))

    t_grade_start = time.perf_counter()
    clip = _h._load_clip(enc_name, device=args.device)
    t_clip_loaded = time.perf_counter()
    refiner_run = cv_nodes.build_refiner_runner(
        dit_path, device=args.refiner_device, loras=grade_loras
    )
    t_ref_loaded = time.perf_counter()

    encoded_refs = None
    if args.ref_images:
        imgs = [s.strip() for s in args.ref_images.split(",") if s.strip()]
        # The DiT probe below samples with the references' latents, which
        # `encode_conditioning_for_vector` copies into the conditioning as `minimax_refs` when they
        # exist. So the VAE is loaded whenever the probe runs, and skipped with `--no-dit`.
        encoded_refs = _h._load_encoded_references(
            clip, h3_config.MODELS["video_vae"], imgs,
            use_vae=not getattr(args, "no_dit", False),
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
                other_ids=ib,
            )
            mb = h3_spans.pooling_mask(
                h3_spans.spans(ib, tb.tolist(), db),
                pooling,
                first_diff=min(fdiff, len(ib) - 1),
                end_diff=end_b,
                other_ids=ia,
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
        "dit_file": dit_path.name,
        "unet_file": dit_path.name,
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
                f"\n[grade] loading DiT {dit_path.name} "
                f"(core={dit_provenance['core_revision']}, attn={dit_provenance['attention']}{lora_note}) "
                f"for denoised-prediction grading..."
            )
            raw_model = comfy.sd.load_diffusion_model(str(dit_path))
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
                sol_cfg = h3_config.sol_for_graph(sol_steps)
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

