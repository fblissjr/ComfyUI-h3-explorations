"""LoRA-aware Qwen3-VL L49 + `TokenRefiner` encoding, media-pair geometry alignment,
and single-pass `encode_pair_set` / `encode_and_cache_spans` helpers."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import sys
import types
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import folder_paths
import torch
from safetensors import safe_open

from .constants import FAMILY_TENSOR_KEYS, POST_REFINER_DIM, VALID_MEDIA_POOLING, import_repo_module as _repo
from .data import PairSet
from .extract import (
    SpanCache,
    _tensor_bf16_bytes,
    build_span_cache,
    detect_family as _extract_detect_family,
    row_norm_census,
    safetensors_recipe_digest,
    save_span_cache,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

_spans = _repo("h3_spans")
_rules = _repo("h3_rules")


def _cv_detect_family(source: Any) -> str:
    cv_mod = sys.modules.get("control_vectors") or sys.modules.get("_h3pack.control_vectors")
    fn = getattr(cv_mod, "detect_family", None) if cv_mod is not None else None
    if callable(fn):
        return fn(source)
    return _extract_detect_family(source)


def _patcher_checkpoint_path(patcher: Any) -> Path | None:
    init = getattr(patcher, "cached_patcher_init", None)
    if isinstance(init, tuple) and len(init) >= 2 and isinstance(init[1], tuple) and len(init[1]) >= 1:
        cand = init[1][0]
        if isinstance(cand, (str, Path)):
            p = Path(cand)
            if p.is_file():
                return p
        elif isinstance(cand, (list, tuple)) and len(cand) >= 1 and isinstance(cand[0], (str, Path)):
            p = Path(cand[0])
            if p.is_file():
                return p
    return None


def resolve_lora_file_path(
    lora_name: str | Path,
    expected_sha256: str | None = None,
) -> Path:
    """Resolve a LoRA filename, relative `loras` key (e.g. `h3/pika_...`), or path."""
    raw = str(lora_name).strip()
    if not raw:
        raise ValueError("lora_name must be a non-empty string")
    direct = Path(raw)
    if direct.is_file():
        return direct
    try:
        resolved = folder_paths.get_full_path("loras", raw)
        if resolved and Path(resolved).is_file():
            return Path(resolved)
    except Exception:
        pass
    target_base = direct.name
    matches: list[Path] = []
    for cand in folder_paths.get_filename_list("loras"):
        if Path(cand).name == target_base:
            full = folder_paths.get_full_path("loras", cand)
            if full and Path(full).is_file():
                matches.append(Path(full))
    if expected_sha256:
        for mp in matches:
            if safetensors_recipe_digest(mp) == expected_sha256:
                return mp
    if matches:
        return matches[0]
    raise ValueError(f"LoRA file {raw!r} not found on disk or in models/loras/")


def extract_model_recipe(
    model: Any,
    clip: Any | None = None,
    *,
    dit_path_override: str | Path | None = None,
    unet_path_override: str | Path | None = None,
    encoder_path_override: str | Path | None = None,
    loras_override: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract the `{checkpoint, loras, encoder}` recipe dict from a `ModelPatcher` + `CLIP`."""
    resolved_dit_override = dit_path_override if dit_path_override is not None else unet_path_override
    ckpt_path: Path | None = (
        Path(resolved_dit_override) if resolved_dit_override is not None else _patcher_checkpoint_path(model)
    )
    if ckpt_path is not None and ckpt_path.is_file():
        ckpt_entry = {
            "file": ckpt_path.name,
            "sha256": safetensors_recipe_digest(
                ckpt_path, extra_tensor_keys=FAMILY_TENSOR_KEYS
            ),
        }
    else:
        inner_model = getattr(model, "model", model)
        dm = getattr(inner_model, "diffusion_model", inner_model)
        backup = getattr(model, "backup", None) or {}
        h = hashlib.sha256()
        tensors_map: dict[str, torch.Tensor] = {}
        for k in FAMILY_TENSOR_KEYS:
            bk_key = f"diffusion_model.{k}"
            t = None
            if isinstance(backup, dict) and bk_key in backup:
                entry = backup[bk_key]
                t = getattr(entry, "weight", entry) if not isinstance(entry, torch.Tensor) else entry
            if t is None:
                obj = dm
                for part in k.split("."):
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                if isinstance(obj, torch.Tensor):
                    t = obj
            if t is None or not isinstance(t, torch.Tensor):
                raise ValueError(
                    f"model ({type(model).__name__}) has no checkpoint path and is missing "
                    f"H3 text-path tensor {k!r}; cannot build control-vector recipe"
                )
            tensors_map[k] = t
            h.update(k.encode("utf-8"))
            h.update(_tensor_bf16_bytes(t))
        fam = _cv_detect_family(tensors_map)
        ckpt_entry = {
            "file": f"in_memory:{fam}",
            "sha256": h.hexdigest(),
        }

    if loras_override is not None:
        lora_entries = list(loras_override)
    else:
        t_opts = getattr(model, "model_options", {}).get("transformer_options", {})
        lora_entries = list(t_opts.get("h3_recipe_loras", ()))
        if not lora_entries:
            recorded_branches = list(t_opts.get("h3_lora_branches", ()))
            for b in recorded_branches:
                b_path = (
                    Path(b["path"])
                    if b.get("path") and Path(b["path"]).is_file()
                    else resolve_lora_file_path(b["file"])
                )
                applied = "branch"
                if (
                    b.get("modules", "all") != "all"
                    or str(b.get("blocks", "all")).strip().lower() != "all"
                    or (
                        float(b.get("start_percent", 0.0)),
                        float(b.get("end_percent", 1.0)),
                    )
                    != (0.0, 1.0)
                ):
                    applied = (
                        f"branch:modules={b.get('modules', 'all')},"
                        f"blocks={b.get('blocks', 'all')},"
                        f"window={float(b.get('start_percent', 0.0)):g}-{float(b.get('end_percent', 1.0)):g}"
                    )
                lora_entries.append(
                    {
                        "file": Path(str(b["file"])).name,
                        "sha256": safetensors_recipe_digest(b_path),
                        "strength": float(b.get("strength", 1.0)),
                        "applied": applied,
                    }
                )

            def _hash_patch_payload(hasher: Any, obj: Any) -> None:
                if isinstance(obj, torch.Tensor):
                    hasher.update(_tensor_bf16_bytes(obj))
                elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                    hasher.update(f":{float(obj):.6f}".encode("utf-8"))
                elif isinstance(obj, str):
                    hasher.update(obj.encode("utf-8"))
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        _hash_patch_payload(hasher, item)
                elif isinstance(obj, dict):
                    for k_sub in sorted(str(k) for k in obj.keys()):
                        hasher.update(k_sub.encode("utf-8"))
                        _hash_patch_payload(hasher, obj[k_sub])
                else:
                    for attr in ("weights", "a", "b", "diff_b", "scale"):
                        if hasattr(obj, attr):
                            hasher.update(attr.encode("utf-8"))
                            _hash_patch_payload(hasher, getattr(obj, attr))
                    b_splits_val = getattr(obj, "_b_splits", None)
                    if b_splits_val is not None:
                        hasher.update(b"_b_splits")
                        _hash_patch_payload(hasher, b_splits_val)
                    closure = getattr(obj, "__closure__", None)
                    if closure:
                        for cell in closure:
                            try:
                                _hash_patch_payload(hasher, cell.cell_contents)
                            except ValueError:
                                pass

            raw_patches = getattr(model, "patches", None)
            if isinstance(raw_patches, dict) and raw_patches:
                ph = hashlib.sha256()
                first_strength = 1.0
                for p_idx, p_key in enumerate(sorted(str(k) for k in raw_patches.keys())):
                    ph.update(p_key.encode("utf-8"))
                    entries = raw_patches.get(p_key, ())
                    if isinstance(entries, (list, tuple)):
                        for ent in entries:
                            if isinstance(ent, (list, tuple)) and len(ent) >= 1 and isinstance(ent[0], (int, float)):
                                if p_idx == 0:
                                    first_strength = float(ent[0])
                                ph.update(f":{float(ent[0]):.6f}".encode("utf-8"))
                                for sub_ent in ent[1:]:
                                    _hash_patch_payload(ph, sub_ent)
                            else:
                                _hash_patch_payload(ph, ent)
                lora_entries.append(
                    {
                        "file": f"in_memory_patches:{len(raw_patches)}_keys",
                        "sha256": ph.hexdigest(),
                        "strength": first_strength,
                        "applied": "merged",
                    }
                )

            obj_patches = getattr(model, "object_patches", None)
            if not recorded_branches and isinstance(obj_patches, dict):
                dm_obj_keys = sorted(
                    str(k)
                    for k in obj_patches.keys()
                    if str(k).startswith("diffusion_model.")
                )
                if dm_obj_keys:
                    bh = hashlib.sha256()
                    for o_key in dm_obj_keys:
                        bh.update(o_key.encode("utf-8"))
                        _hash_patch_payload(bh, obj_patches[o_key])
                    lora_entries.append(
                        {
                            "file": f"in_memory_branch:{len(dm_obj_keys)}_keys",
                            "sha256": bh.hexdigest(),
                            "strength": 1.0,
                            "applied": "branch",
                        }
                    )

    enc_path: Path | None = None
    if encoder_path_override is not None:
        enc_path = Path(encoder_path_override)
    elif clip is not None:
        clip_patcher = getattr(clip, "patcher", None)
        enc_path = _patcher_checkpoint_path(clip_patcher)
    if enc_path is not None and enc_path.is_file():
        enc_entry = {
            "file": enc_path.name,
            "sha256": safetensors_recipe_digest(enc_path),
        }
    else:
        enc_entry = {
            "file": "unspecified",
            "sha256": "0" * 64,
        }

    return {
        "checkpoint": ckpt_entry,
        "loras": lora_entries,
        "encoder": enc_entry,
    }


def _attach_refiner_lora_branches(
    ref: Any,
    loras: Sequence[Mapping[str, Any]],
) -> None:
    """Attach `token_refiner.*` branches from `loras` directly to `TokenRefiner` `ref`."""
    if not loras:
        return
    import comfy.utils

    lb = _repo("lora_branch")
    taken: list[str] = []
    existing_keys: set[str] = set(getattr(ref, "_h3_refiner_patched_keys", None) or ())
    new_keys: set[str] = set()
    linear_by_sub: dict[str, list[Any]] = {}
    fc2_by_parent: dict[str, list[Any]] = {}
    for spec in loras:
        raw_path = spec.get("path") or spec.get("file")
        if not raw_path:
            raise ValueError(f"LoRA recipe entry missing 'file'/'path': {spec!r}")
        lpath = (
            Path(str(raw_path))
            if Path(str(raw_path)).is_file()
            else resolve_lora_file_path(str(raw_path), expected_sha256=spec.get("sha256"))
        )
        raw_sd = comfy.utils.load_torch_file(str(lpath), safe_load=True)
        sd: dict[str, torch.Tensor] = {}
        for k, v in raw_sd.items():
            if k.startswith("token_refiner."):
                norm_k = f"diffusion_model.{k}"
            elif k.startswith("model.diffusion_model."):
                norm_k = k[len("model.") :]
            else:
                norm_k = k
            if norm_k.startswith("diffusion_model.token_refiner."):
                sd[norm_k] = v
        strength = float(spec.get("strength", 1.0))
        branches = lb.parse_lora(sd, strength)
        modules_choice = str(spec.get("modules", "all"))
        blocks_choice = str(spec.get("blocks", "all"))
        branches = lb.select(branches, modules_choice, lb.parse_blocks(blocks_choice))
        ref_branches = {
            k[len("token_refiner.") :]: br
            for k, br in branches.items()
            if k.startswith("token_refiner.")
        }
        fc2_subs = {sub for sub in ref_branches if sub.endswith("mlp.fc2")}
        for sub, branch in ref_branches.items():
            if sub in fc2_subs:
                continue
            patch_key = f"diffusion_model.token_refiner.{sub}.forward"
            if patch_key in existing_keys:
                taken.append(patch_key)
                continue
            new_keys.add(patch_key)
            linear_by_sub.setdefault(sub, []).append(branch)
        for sub in fc2_subs:
            parent_sub = sub[: -len(".fc2")]
            patch_key = f"diffusion_model.token_refiner.{parent_sub}.forward"
            if patch_key in existing_keys:
                taken.append(patch_key)
                continue
            new_keys.add(patch_key)
            fc2_by_parent.setdefault(parent_sub, []).append(ref_branches[sub])
    if taken:
        raise ValueError(
            f"token_refiner object patches already taken ({len(taken)}), e.g. {taken[:3]}: "
            "another LoRA patches these forwards"
        )
    for sub, branches_list in linear_by_sub.items():
        mod = ref.get_submodule(sub)
        mod.forward = lb._linear_forward(mod.forward, branches_list)
    for parent_sub, fc2_branches_list in fc2_by_parent.items():
        mlp = ref.get_submodule(parent_sub)
        mlp.forward = lb._mlp_forward(mlp, mlp.fc1.forward, fc2_branches_list)
    setattr(ref, "_h3_refiner_patched_keys", existing_keys | new_keys)


def build_refiner_runner(
    source: str | Path | Any,
    *,
    device: torch.device | str | None = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    loras: Sequence[Mapping[str, Any]] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return `run_refiner(encoder_rows_2d: Tensor[L, 5120]) -> Tensor[L, 5376]` (`float32` CPU)."""
    eff_device = device if device is not None else "cpu"
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"checkpoint file not found: {path}")
        import comfy.ldm.minimax.model as MM
        import comfy.ops

        sig_defaults = {
            k: v.default
            for k, v in inspect.signature(MM.MiniMaxH3Model.__init__).parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        ops = comfy.ops.disable_weight_init
        proj = ops.Linear(
            sig_defaults["text_dim"],
            sig_defaults["hidden_size"],
            bias=True,
            dtype=dtype,
            device=eff_device,
        )
        ref = MM.TokenRefiner(
            sig_defaults["token_refiner_num_layers"],
            sig_defaults["hidden_size"],
            sig_defaults["num_attention_heads"],
            sig_defaults["attention_head_dim"],
            sig_defaults["ffn_hidden_size"],
            sig_defaults["norm_eps"],
            sig_defaults["qk_norm_eps"],
            sig_defaults["final_norm_eps"],
            dtype=dtype,
            device=eff_device,
            operations=ops,
        )
        sd_p: dict[str, torch.Tensor] = {}
        sd_r: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as fh:
            for k in fh.keys():
                clean_k = k
                for prefix in ("model.diffusion_model.", "diffusion_model."):
                    if clean_k.startswith(prefix):
                        clean_k = clean_k[len(prefix) :]
                        break
                if clean_k.startswith("condition_proj."):
                    sd_p[clean_k[len("condition_proj.") :]] = fh.get_tensor(k).to(
                        device=eff_device, dtype=dtype
                    )
                elif clean_k.startswith("token_refiner."):
                    sd_r[clean_k[len("token_refiner.") :]] = fh.get_tensor(k).to(
                        device=eff_device, dtype=dtype
                    )
        for mod, sd, name in ((proj, sd_p, "condition_proj"), (ref, sd_r, "token_refiner")):
            missing, unexpected = mod.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise ValueError(
                    f"{path.name} {name}: missing={missing[:3]} unexpected={unexpected[:3]}"
                )
        proj.eval()
        ref.eval()
        if loras:
            _attach_refiner_lora_branches(ref, loras)

        def _run_from_file(x_2d: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                x_in = x_2d.detach().to(device=eff_device, dtype=dtype)
                return ref(proj(x_in)).detach().to(device="cpu", dtype=torch.float32)

        return _run_from_file

    ckpt_path = _patcher_checkpoint_path(source)
    t_opts = getattr(source, "model_options", {}).get("transformer_options", {})
    obj_patches = getattr(source, "object_patches", None) or {}
    has_dm_obj_patches = any(
        str(k).startswith("diffusion_model.") for k in obj_patches.keys()
    )
    has_loras = (
        bool(getattr(source, "patches", None))
        or has_dm_obj_patches
        or bool(t_opts.get("h3_recipe_loras"))
        or bool(t_opts.get("h3_lora_branches"))
    )
    if ckpt_path is not None and ckpt_path.is_file() and not has_loras:
        return build_refiner_runner(ckpt_path, device=eff_device, dtype=dtype, loras=loras)
    if (
        ckpt_path is not None
        and ckpt_path.is_file()
        and not getattr(source, "patches", None)
        and bool(t_opts.get("h3_lora_branches"))
    ):
        return build_refiner_runner(
            ckpt_path,
            device=eff_device,
            dtype=dtype,
            loras=loras if loras is not None else t_opts["h3_lora_branches"],
        )

    inner = getattr(source, "model", source)
    dm = getattr(inner, "diffusion_model", inner)
    has_preprocess = hasattr(dm, "preprocess_text_embeds")
    has_direct_refiner = hasattr(dm, "token_refiner") and hasattr(dm, "condition_proj")
    if not has_preprocess and not has_direct_refiner:
        raise ValueError(
            f"model ({type(source).__name__}) does not expose diffusion_model.preprocess_text_embeds or token_refiner"
        )
    infer_dtype = getattr(inner, "get_dtype_inference", lambda: dtype)()
    has_unapplied_patches = bool(getattr(source, "patches", None)) or has_dm_obj_patches

    def _run_from_model(x_2d: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if has_unapplied_patches and hasattr(source, "patch_model"):
                try:
                    import comfy.model_management as _mm

                    _mm.load_models_gpu([source])
                except Exception:
                    try:
                        source.patch_model(
                            device_to=torch.device(eff_device),
                            lowvram_model_memory=0,
                            load_weights=True,
                            force_patch_weights=True,
                        )
                    except Exception:
                        pass
            param = next(dm.condition_proj.parameters(), None)
            dev = param.device if param is not None else torch.device(eff_device)
            if has_preprocess:
                x_3d = x_2d.detach().unsqueeze(0).to(device=dev, dtype=infer_dtype)
                out_3d = dm.preprocess_text_embeds(x_3d)
                return out_3d[0].detach().to(device="cpu", dtype=torch.float32)
            hidden_size = int(getattr(dm, "hidden_size", POST_REFINER_DIM))
            if int(x_2d.shape[-1]) == hidden_size:
                return x_2d.detach().to(device="cpu", dtype=torch.float32)
            x_in = x_2d.detach().to(device=dev, dtype=infer_dtype)
            out_2d = dm.token_refiner(dm.condition_proj(x_in))
            return out_2d.detach().to(device="cpu", dtype=torch.float32)

    return _run_from_model


def _extract_token_entries(tokens: Any) -> list[Any]:
    if isinstance(tokens, dict):
        if not tokens:
            raise ValueError("clip.tokenize returned an empty dict")
        batches = next(iter(tokens.values()))
    else:
        batches = tokens
    if not isinstance(batches, (list, tuple)):
        raise TypeError(
            f"expected token batches list from clip.tokenize, got {type(batches).__name__}"
        )
    if not batches:
        return []
    first = batches[0]
    if isinstance(first, (list, tuple)) and (
        not first or isinstance(first[0], (list, tuple, dict))
    ):
        return list(first)
    return list(batches)


def encode_conditioning_for_vector(
    clip: Any,
    text: str,
    *,
    encoded_references: Any | None = None,
) -> tuple[list[Any], torch.Tensor, torch.Tensor, list[int], Any]:
    clean_text = _rules.normalize_prompt(text)
    hf_tok = _spans.unwrap_hf_tokenizer(clip)
    if hf_tok is None:
        raise ValueError("clip does not expose the MiniMax Qwen3-VL tokenizer")

    def decode_fn(ids: Sequence[int]) -> str:
        return hf_tok.decode(list(ids), skip_special_tokens=False)

    csm = getattr(clip, "cond_stage_model", None)
    if csm is not None and not hasattr(csm, "memory_estimation_function"):
        csm.memory_estimation_function = lambda _tokens, device=None: 2560 * 1024 * 1024

    if encoded_references is not None:
        import node_helpers

        ref_enc_mod = _repo("reference_encode")
        cond_list = ref_enc_mod.condition_prompt(encoded_references, clean_text)
        if getattr(encoded_references, "ref_blocks", None):
            cond_list = node_helpers.conditioning_set_values(
                cond_list, {"minimax_refs": encoded_references.ref_blocks}
            )
        cond_tensor = cond_list[0][0][0].detach().to(device="cpu", dtype=torch.float32)
        tags = torch.as_tensor(
            cond_list[0][1]["minimax_token_tags"], dtype=torch.long, device="cpu"
        ).flatten()
        ref_items = getattr(encoded_references, "ref_items", None)
        full_tokens = clip.tokenize(clean_text, minimax_ref_items=ref_items)
        entries = _extract_token_entries(full_tokens)
        token_ids = _spans.expand_token_ids(entries, tags)
        span_masks = _spans.spans(token_ids, tags.tolist(), decode_fn)
        setattr(tags, "_h3_span_masks", span_masks)
        cond_list[0][1]["h3_span_masks"] = span_masks
        raw_tags_obj = cond_list[0][1].get("minimax_token_tags")
        if isinstance(raw_tags_obj, torch.Tensor):
            setattr(raw_tags_obj, "_h3_span_masks", span_masks)
        return cond_list, cond_tensor, tags, token_ids, decode_fn

    tokens = clip.tokenize(clean_text)
    cond_list = clip.encode_from_tokens_scheduled(tokens)
    cond_tensor = cond_list[0][0][0].detach().to(device="cpu", dtype=torch.float32)
    tags = torch.as_tensor(
        cond_list[0][1]["minimax_token_tags"], dtype=torch.long, device="cpu"
    ).flatten()
    entries = _extract_token_entries(tokens)
    token_ids = _spans.expand_token_ids(entries, tags)
    span_masks = _spans.spans(token_ids, tags.tolist(), decode_fn)
    setattr(tags, "_h3_span_masks", span_masks)
    cond_list[0][1]["h3_span_masks"] = span_masks
    raw_tags_obj = cond_list[0][1].get("minimax_token_tags")
    if isinstance(raw_tags_obj, torch.Tensor):
        setattr(raw_tags_obj, "_h3_span_masks", span_masks)
    return cond_list, cond_tensor, tags, token_ids, decode_fn


def encode_prompt_for_vector(
    clip: Any,
    text: str,
    *,
    encoded_references: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int], Any]:
    _cond_list, cond_tensor, tags, token_ids, decode_fn = encode_conditioning_for_vector(
        clip, text, encoded_references=encoded_references
    )
    return cond_tensor, tags, token_ids, decode_fn


def encode_pair_set(
    clip: Any,
    pair_set: PairSet,
    *,
    encoded_references: Any | None = None,
    with_holds: bool = True,
    on_pair: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, float]]]:
    """Encode every pair and hold of `pair_set` through `clip` (Qwen3-VL `layers.49`) in one pass."""
    encoded_pairs: list[dict[str, Any]] = []
    enc_censuses: list[dict[str, float]] = []
    for idx, item in enumerate(pair_set.pairs):
        enc_a, tags_a, ids_a, dec_a = encode_prompt_for_vector(
            clip, item.a, encoded_references=encoded_references
        )
        enc_b, tags_b, ids_b, dec_b = encode_prompt_for_vector(
            clip, item.b, encoded_references=encoded_references
        )
        enc_censuses.append(row_norm_census(enc_a, tags_a))
        enc_censuses.append(row_norm_census(enc_b, tags_b))
        first_diff, end_a, end_b = _spans.differing_token_bounds(ids_a, ids_b)
        entry = {
            "idx": idx,
            "item": item,
            "enc_a": enc_a,
            "tags_a": tags_a,
            "ids_a": ids_a,
            "spans_a": _spans.spans(ids_a, tags_a.tolist(), dec_a),
            "enc_b": enc_b,
            "tags_b": tags_b,
            "ids_b": ids_b,
            "spans_b": _spans.spans(ids_b, tags_b.tolist(), dec_b),
            "first_diff": first_diff,
            "end_a": end_a,
            "end_b": end_b,
        }
        encoded_pairs.append(entry)
        if on_pair is not None:
            on_pair(entry)

    encoded_holds: list[dict[str, Any]] = []
    if with_holds and pair_set.holds:
        for h_item in pair_set.holds:
            h_enc_a, h_tags_a, h_ids_a, h_dec_a = encode_prompt_for_vector(
                clip, h_item.a, encoded_references=encoded_references
            )
            h_enc_b, h_tags_b, h_ids_b, h_dec_b = encode_prompt_for_vector(
                clip, h_item.b, encoded_references=encoded_references
            )
            h_diff, h_end_a, h_end_b = _spans.differing_token_bounds(h_ids_a, h_ids_b)
            encoded_holds.append(
                {
                    "h_item": h_item,
                    "enc_a": h_enc_a,
                    "tags_a": h_tags_a,
                    "ids_a": h_ids_a,
                    "spans_a": _spans.spans(h_ids_a, h_tags_a.tolist(), h_dec_a),
                    "enc_b": h_enc_b,
                    "tags_b": h_tags_b,
                    "ids_b": h_ids_b,
                    "spans_b": _spans.spans(h_ids_b, h_tags_b.tolist(), h_dec_b),
                    "first_diff": h_diff,
                    "end_a": h_end_a,
                    "end_b": h_end_b,
                }
            )
    return encoded_pairs, encoded_holds, enc_censuses


def refine_and_cache_spans(
    encoded_pairs: Sequence[Mapping[str, Any]],
    encoded_holds: Sequence[Mapping[str, Any]],
    enc_censuses: Sequence[Mapping[str, float]],
    refiner_run: Callable[[torch.Tensor], torch.Tensor],
    *,
    pooling: str = "diff_span",
    include_encoder: bool = True,
    cache_path: str | Path | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> tuple[SpanCache, list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]], list[dict[str, float]]]:
    """Run `refiner_run` across `encoded_pairs` / `encoded_holds` and build (and optionally save) an `fp32` `SpanCache`."""
    refined_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    ref_censuses: list[dict[str, float]] = []
    for enc in encoded_pairs:
        ra = refiner_run(enc["enc_a"])
        rb = refiner_run(enc["enc_b"])
        refined_pairs.append((ra, rb))
        ref_censuses.append(row_norm_census(ra, enc["tags_a"]))
        ref_censuses.append(row_norm_census(rb, enc["tags_b"]))
    refined_holds = [(refiner_run(h["enc_a"]), refiner_run(h["enc_b"])) for h in encoded_holds]

    med_ref = float(sum(c["median_row_norm"] for c in ref_censuses) / len(ref_censuses))
    med_enc = float(sum(c["median_row_norm_ex_row0"] for c in enc_censuses) / len(enc_censuses))
    cache = build_span_cache(
        encoded_pairs,
        refined_pairs,
        encoded_holds,
        refined_holds,
        pooling=pooling,
        include_encoder=include_encoder,
        median_ref_row_norm=med_ref,
        median_enc_row_norm_ex0=med_enc,
        extra_metadata=extra_metadata,
    )
    if cache_path is not None:
        save_span_cache(cache_path, cache, overwrite=True)
    return cache, refined_pairs, refined_holds, ref_censuses


def _validate_short_edge(short_edge: Any) -> int:
    se = int(short_edge)
    if se < _rules.CANVAS_MULTIPLE:
        raise ValueError(
            f"short_edge must be >= {_rules.CANVAS_MULTIPLE}, got {short_edge!r}"
        )
    return se


def match_media_pair_geometry(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    *,
    short_edge: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Resize `image_a` and `image_b` to an identical 32-px-snapped `(target_w, target_h)` RGB tensor."""
    ref_cond_mod = _repo("reference_conditioning")
    n_a, h_a, w_a = ref_cond_mod._image_shape(image_a, "image_a")
    n_b, h_b, w_b = ref_cond_mod._image_shape(image_b, "image_b")
    if n_a != n_b:
        raise ValueError(
            f"image_a and image_b must have the same frame/batch count, got {n_a} vs {n_b}"
        )
    if image_a.shape[-1] < 3 or image_b.shape[-1] < 3:
        raise ValueError(
            f"image_a and image_b must have at least 3 RGB channels, got {image_a.shape[-1]} and {image_b.shape[-1]}"
        )
    se = _validate_short_edge(short_edge)
    raw_w, raw_h = ref_cond_mod.qwen_view_size(w_a, h_a, se)
    target_w = max(_rules.CANVAS_MULTIPLE, int(raw_w))
    target_h = max(_rules.CANVAS_MULTIPLE, int(raw_h))

    frames_a: list[torch.Tensor] = []
    frames_b: list[torch.Tensor] = []
    for i in range(n_a):
        va = ref_cond_mod._view_or_source(image_a[i : i + 1], w_a, h_a, target_w, target_h)
        vb = ref_cond_mod._view_or_source(image_b[i : i + 1], w_b, h_b, target_w, target_h)
        frames_a.append(va)
        frames_b.append(vb)
    return torch.cat(frames_a, dim=0), torch.cat(frames_b, dim=0), (target_w, target_h)


def _encode_media_frame_and_prompt(
    clip: Any,
    frame_1hwc: torch.Tensor,
    text: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int], Any]:
    clean_text = _rules.normalize_prompt(text)
    hf_tok = _spans.unwrap_hf_tokenizer(clip)
    if hf_tok is None:
        raise ValueError("clip does not expose the MiniMax Qwen3-VL tokenizer")

    def decode_fn(ids: Sequence[int]) -> str:
        return hf_tok.decode(list(ids), skip_special_tokens=False)

    csm = getattr(clip, "cond_stage_model", None)
    if csm is not None and not hasattr(csm, "memory_estimation_function"):
        csm.memory_estimation_function = lambda _tokens, device=None: 2560 * 1024 * 1024

    ref_items = [{"type": "image", "data": frame_1hwc}]
    tokens = clip.tokenize(clean_text, minimax_ref_items=ref_items)
    cond_list = clip.encode_from_tokens_scheduled(tokens)
    cond_tensor = cond_list[0][0][0].detach().to(device="cpu", dtype=torch.float32)
    tags = torch.as_tensor(
        cond_list[0][1]["minimax_token_tags"], dtype=torch.long, device="cpu"
    ).flatten()
    entries = _extract_token_entries(tokens)
    token_ids = _spans.expand_token_ids(entries, tags)
    return cond_tensor, tags, token_ids, decode_fn


def _media_pooling_mask(
    spans_obj: Any,
    tags_1d: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if pooling == "scene_text":
        if bool(getattr(spans_obj["scene"], "_h3_has_sections", False)):
            m_scene = _spans.inference_mask(spans_obj, "scene_only")
            if bool(m_scene.any()):
                return m_scene
        return _spans.pooling_mask(spans_obj, pooling)
    if pooling == "all_tag1":
        return _spans.pooling_mask(spans_obj, pooling)
    if pooling in ("look_only", "description_only", "soundscape_only"):
        m_sub = _spans.inference_mask(spans_obj, pooling)
        if not bool(m_sub.any()):
            raise ValueError(f"media pooling={pooling!r} found zero matching section rows")
        return m_sub
    if pooling == "vision_tokens":
        m = (tags_1d == 0).to(dtype=torch.bool, device="cpu")
        if not bool(m.any()):
            raise ValueError("media pooling='vision_tokens' found zero tag-0 vision rows")
        return m
    if pooling == "all_rows":
        m = torch.ones((int(tags_1d.numel()),), dtype=torch.bool, device="cpu")
        if not bool(m.any()):
            raise ValueError("media pooling='all_rows' found empty sequence")
        return m
    raise ValueError(
        f"media pooling must be one of {VALID_MEDIA_POOLING}, got {pooling!r}"
    )
