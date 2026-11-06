"""ComfyUI nodes for making and applying MiniMax H3 contrastive control vectors.

`MiniMaxH3MakeControlVector` reads a pair file under `prompt_bank/control_vectors/`,
encodes each pair's `a` and `b` prompts (and any `[[hold]]` groups) through the
loaded H3 Qwen3-VL encoder (or on top of a cached `MiniMaxH3EncodeReferences`
pass), runs the DiT's `condition_proj` and 2-block `token_refiner`, pools the
selected span rows (`scene_text` by default), computes the float64 contrastive
direction, projects out any holds, and writes
`<ComfyUI models dir>/control_vectors/<name>__<family>.safetensors` with its
recipe, family, and quality report (`docs/research/control_vectors/2026-11-04_h3_control_vectors.md`).

`MiniMaxH3ApplyControlVectors` attaches a `WrappersMP.DIFFUSION_MODEL` wrapper on
a cloned `ModelPatcher` that accumulates `strength * vector` across stacked
vectors in float64, casts once to the context dtype, and adds the shift to the
selected rows of the refined `context` (`[B, L, 5376]`), masked off `tag == 0`
vision rows by default so reference pictures and video blocks stay bitwise
untouched and sequence length `L` is unchanged.

Nothing here patches ComfyUI core.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import folder_paths
import torch
from comfy.patcher_extension import WrappersMP
from comfy_api.latest import io
from safetensors import safe_open

logger = logging.getLogger(__name__)


def _repo(name: str):
    if __package__:
        try:
            return importlib.import_module(f".{name}", __package__)
        except ImportError:
            pass
    try:
        return importlib.import_module(name)
    except ImportError:
        import sys
        import types

        pkg_name = "_h3pack"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str( Path(__file__).resolve().parent )]
            sys.modules[pkg_name] = pkg
        return importlib.import_module(f"{pkg_name}.{name}")


_cv = _repo("control_vectors")
_spans = _repo("h3_spans")
_pair_compiler = _repo("h3_pair_compiler")
_rules = _repo("h3_rules")
_substrate = _repo("substrate")

REPO_ROOT = Path(__file__).resolve().parent
COMFY_ROOT = REPO_ROOT.parent.parent
PAIR_BANK_DIR = REPO_ROOT / "prompt_bank" / "control_vectors"
PROMPT_BANK_DIR = REPO_ROOT / "prompt_bank"

CONTROL_VECTORS_FOLDER = "control_vectors"
WRAPPER_KEY = "h3_control_vectors"
MASK_MODES = _spans.INFERENCE_MASK_MODES
_APPLY_DEFAULTS = _rules.CONTROL_VECTOR_DEFAULTS


def register_control_vectors_folder() -> None:
    """Register `<ComfyUI models dir>/control_vectors` in `folder_paths`."""
    default_cv_dir = os.path.join(folder_paths.models_dir, CONTROL_VECTORS_FOLDER)
    os.makedirs(default_cv_dir, exist_ok=True)
    folder_paths.add_model_folder_path(CONTROL_VECTORS_FOLDER, default_cv_dir, is_default=True)
    folder_paths.folder_names_and_paths[CONTROL_VECTORS_FOLDER] = (
        folder_paths.folder_names_and_paths[CONTROL_VECTORS_FOLDER][0],
        {".safetensors"},
    )


register_control_vectors_folder()


def _vector_combo_options() -> list[str]:
    files = sorted(
        f
        for f in folder_paths.get_filename_list(CONTROL_VECTORS_FOLDER)
        if f.endswith(".safetensors") and not f.replace("\\", "/").startswith("j_lens/")
    )
    return ["none", *files]


def _is_minimax_h3(diffusion_model: Any) -> bool:
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
    except ImportError:
        return False
    return isinstance(diffusion_model, MiniMaxH3Model)


def _patcher_checkpoint_path(patcher: Any) -> Path | None:
    init = getattr(patcher, "cached_patcher_init", None)
    if isinstance(init, tuple) and len(init) >= 2 and isinstance(init[1], tuple) and len(init[1]) >= 1:
        cand = init[1][0]
        if isinstance(cand, (str, Path)):
            p = Path(cand)
            if p.is_file():
                return p
            # Handle list/tuple of paths (e.g. CLIPLoader passes `[clip_path]`)
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
            if _cv.safetensors_recipe_digest(mp) == expected_sha256:
                return mp
    if matches:
        return matches[0]
    raise ValueError(f"LoRA file {raw!r} not found on disk or in models/loras/")


def extract_model_recipe(
    model: Any,
    clip: Any | None = None,
    *,
    unet_path_override: str | Path | None = None,
    encoder_path_override: str | Path | None = None,
    loras_override: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract the `CONTRACT.md` §3 recipe dict `{checkpoint, loras, encoder}` from a
    loaded `ModelPatcher` and optional `CLIP` (or explicit file paths)."""
    # 1. Diffusion checkpoint
    ckpt_path: Path | None = (
        Path(unet_path_override) if unet_path_override is not None else _patcher_checkpoint_path(model)
    )
    if ckpt_path is not None and ckpt_path.is_file():
        ckpt_entry = {
            "file": ckpt_path.name,
            "sha256": _cv.safetensors_recipe_digest(
                ckpt_path, extra_tensor_keys=_cv.FAMILY_TENSOR_KEYS
            ),
        }
    else:
        # In-memory model without cached_patcher_init: hash the three defining
        # text-path tensors from backup/diffusion_model in BF16.
        inner_model = getattr(model, "model", model)
        dm = getattr(inner_model, "diffusion_model", inner_model)
        backup = getattr(model, "backup", None) or {}
        h = hashlib.sha256()
        tensors_map: dict[str, torch.Tensor] = {}
        for k in _cv.FAMILY_TENSOR_KEYS:
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
            h.update(_cv._tensor_bf16_bytes(t))
        fam = _cv.detect_family(tensors_map)
        ckpt_entry = {
            "file": f"in_memory:{fam}",
            "sha256": h.hexdigest(),
        }

    # 2. LoRAs (explicit override, recorded on model_options, attached on model.patches,
    #    or attached as MiniMaxH3LoRABranch object_patches)
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
                        "sha256": _cv.safetensors_recipe_digest(b_path),
                        "strength": float(b.get("strength", 1.0)),
                        "applied": applied,
                    }
                )

            def _hash_patch_payload(hasher: Any, obj: Any) -> None:
                if isinstance(obj, torch.Tensor):
                    hasher.update(_cv._tensor_bf16_bytes(obj))
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

    # 3. Encoder checkpoint
    enc_path: Path | None = None
    if encoder_path_override is not None:
        enc_path = Path(encoder_path_override)
    elif clip is not None:
        clip_patcher = getattr(clip, "patcher", None)
        enc_path = _patcher_checkpoint_path(clip_patcher)
    if enc_path is not None and enc_path.is_file():
        enc_entry = {
            "file": enc_path.name,
            "sha256": _cv.safetensors_recipe_digest(enc_path),
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
    pending_keys: set[str] = set(existing_keys)
    staged_linear: list[tuple[str, Any, Any]] = []
    staged_mlp: list[tuple[str, Any, Any]] = []
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
            if patch_key in pending_keys:
                taken.append(patch_key)
                continue
            pending_keys.add(patch_key)
            staged_linear.append((patch_key, ref.get_submodule(sub), branch))
        for sub in fc2_subs:
            parent_sub = sub[: -len(".fc2")]
            patch_key = f"diffusion_model.token_refiner.{parent_sub}.forward"
            if patch_key in pending_keys:
                taken.append(patch_key)
                continue
            pending_keys.add(patch_key)
            staged_mlp.append((patch_key, ref.get_submodule(parent_sub), ref_branches[sub]))
    if taken:
        raise ValueError(
            f"token_refiner object patches already taken ({len(taken)}), e.g. {taken[:3]}: "
            "another LoRA patches these forwards"
        )
    for _, mod, branch in staged_linear:
        mod.forward = lb._linear_forward(mod.forward, branch)
    for _, mlp, branch in staged_mlp:
        mlp.forward = lb._mlp_forward(mlp, mlp.fc1.forward, branch)
    setattr(ref, "_h3_refiner_patched_keys", pending_keys)


def build_refiner_runner(
    source: str | Path | Any,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    loras: Sequence[Mapping[str, Any]] | None = None,
):
    """Return a callable `run_refiner(encoder_rows_2d: Tensor[L, 5120]) -> Tensor[L, 5376]`
    that runs `condition_proj` + `token_refiner` at `dtype` (default `bfloat16`,
    matching `MiniMaxH3.extra_conds`).

    Accepts either a `.safetensors` checkpoint path (loading only `condition_proj.*`
    and `token_refiner.*` in < 0.2s without loading the 50 DiT blocks, plus optional
    `loras` branches on `token_refiner.blocks.*`) or a loaded ComfyUI `ModelPatcher`
    / `MiniMaxH3Model`."""
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
            device=device,
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
            device=device,
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
                        device=device, dtype=dtype
                    )
                elif clean_k.startswith("token_refiner."):
                    sd_r[clean_k[len("token_refiner.") :]] = fh.get_tensor(k).to(
                        device=device, dtype=dtype
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
                x_in = x_2d.detach().to(device=device, dtype=dtype)
                return ref(proj(x_in)).detach().to(device="cpu", dtype=torch.float32)

        return _run_from_file

    # Loaded ModelPatcher or MiniMaxH3Model
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
        return build_refiner_runner(ckpt_path, device=device, dtype=dtype, loras=loras)
    if (
        ckpt_path is not None
        and ckpt_path.is_file()
        and not getattr(source, "patches", None)
        and bool(t_opts.get("h3_lora_branches"))
    ):
        return build_refiner_runner(
            ckpt_path,
            device=device,
            dtype=dtype,
            loras=loras if loras is not None else t_opts["h3_lora_branches"],
        )

    inner = getattr(source, "model", source)
    dm = getattr(inner, "diffusion_model", inner)
    if not hasattr(dm, "preprocess_text_embeds"):
        raise ValueError(
            f"model ({type(source).__name__}) does not expose diffusion_model.preprocess_text_embeds"
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
                        source.patch_model(device_to=torch.device(device), lowvram_model_memory=0, load_weights=True, force_patch_weights=True)
                    except Exception:
                        pass
            param = next(dm.condition_proj.parameters(), None)
            dev = param.device if param is not None else torch.device(device)
            x_3d = x_2d.detach().unsqueeze(0).to(device=dev, dtype=infer_dtype)
            out_3d = dm.preprocess_text_embeds(x_3d)
            return out_3d[0].detach().to(device="cpu", dtype=torch.float32)

    return _run_from_model


def _extract_token_entries(tokens: Any) -> list[Any]:
    """Extract the inner `[(token_or_dict, weight, ...), ...]` entry list from
    `clip.tokenize(...)`'s return value."""
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
    """Encode `text` through `clip` (or on top of `encoded_references` when provided)
    and return `(conditioning_list, encoder_rows_2d [L, 5120], tags_1d [L], token_ids, decode_fn)`."""
    clean_text = _rules.normalize_prompt(text)
    tok_root = getattr(clip, "tokenizer", None)
    mm_tok = getattr(tok_root, "minimax", tok_root)
    qwen_tok = getattr(mm_tok, "qwen3vl_32b", mm_tok)
    hf_tok = getattr(qwen_tok, "tokenizer", None)
    if hf_tok is None:
        raise ValueError("clip does not expose the MiniMax Qwen3-VL tokenizer")

    def decode_fn(ids: Sequence[int]) -> str:
        return hf_tok.decode(list(ids), skip_special_tokens=False)

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
    """Encode `text` through `clip` (or on top of `encoded_references` when provided)
    and return `(encoder_rows_2d [L, 5120], tags_1d [L], token_ids, decode_fn)`."""
    _cond_list, cond_tensor, tags, token_ids, decode_fn = encode_conditioning_for_vector(
        clip, text, encoded_references=encoded_references
    )
    return cond_tensor, tags, token_ids, decode_fn


def resolve_pair_file_path(pair_file: str | Path) -> Path:
    raw = str(pair_file).strip()
    if not raw:
        raise ValueError("pair_file must be a non-empty string")
    cand = Path(raw)
    if cand.is_file():
        return cand.resolve()
    if not raw.endswith(".toml"):
        raw_toml = f"{raw}.toml"
    else:
        raw_toml = raw
    for base in (PAIR_BANK_DIR, REPO_ROOT):
        p = base / raw_toml
        if p.is_file():
            return p.resolve()
    raise ValueError(
        f"pair file {pair_file!r} not found under {PAIR_BANK_DIR}"
    )


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
    pair_set: _cv.PairSet | None = None,
    words: str = "",
    slot: str = "auto",
) -> tuple[Path, dict[str, Any]]:
    """Build and save a `.safetensors` control vector from `pair_path`, `pair_set`,
    or simple `words` / short contrast clauses (zero `.toml` file required),
    returning `(saved_path, full_report_dict)`."""
    if pooling not in _cv.VALID_POOLING:
        raise ValueError(f"pooling must be one of {_cv.VALID_POOLING}, got {pooling!r}")

    resolved_pair_path: Path | None = None
    if pair_set is not None:
        pass
    elif isinstance(words, str) and words.strip():
        compile_mode = "ref2va" if encoded_references is not None else "t2va"
        pair_set = _pair_compiler.compile_words_to_pairset(
            words.strip(),
            name=output_name.strip() or None,
            slot=slot,
            mode=compile_mode,
        )
    else:
        if pair_path is None:
            raise ValueError("either pair_path, pair_set, or words must be provided")
        resolved_pair_path = resolve_pair_file_path(pair_path)
        pair_set = _cv.load_pairs(resolved_pair_path, bank_dir=PROMPT_BANK_DIR)
    family = _cv.detect_family(model)
    recipe = extract_model_recipe(model, clip)
    refiner_run = build_refiner_runner(model)

    a_pooled_list: list[torch.Tensor] = []
    b_pooled_list: list[torch.Tensor] = []
    enc_a_pooled_list: list[torch.Tensor] = []
    enc_b_pooled_list: list[torch.Tensor] = []
    weights_list: list[float] = []
    enc_censuses: list[dict[str, float]] = []
    ref_censuses: list[dict[str, float]] = []
    pair_spans: list[dict[str, Any]] = []

    for idx, item in enumerate(pair_set.pairs):
        enc_a, tags_a, ids_a, dec_a = encode_prompt_for_vector(
            clip, item.a, encoded_references=encoded_references
        )
        enc_b, tags_b, ids_b, dec_b = encode_prompt_for_vector(
            clip, item.b, encoded_references=encoded_references
        )
        ref_a = refiner_run(enc_a)
        ref_b = refiner_run(enc_b)

        enc_censuses.append(_cv.row_norm_census(enc_a, tags_a))
        enc_censuses.append(_cv.row_norm_census(enc_b, tags_b))
        ref_censuses.append(_cv.row_norm_census(ref_a, tags_a))
        ref_censuses.append(_cv.row_norm_census(ref_b, tags_b))

        first_diff, end_a, end_b = _spans.differing_token_bounds(ids_a, ids_b)
        spans_a = _spans.spans(ids_a, tags_a.tolist(), dec_a)
        spans_b = _spans.spans(ids_b, tags_b.tolist(), dec_b)
        mask_a = _spans.pooling_mask(
            spans_a,
            pooling,
            first_diff=min(first_diff, len(ids_a) - 1),
            end_diff=end_a,
        )
        mask_b = _spans.pooling_mask(
            spans_b,
            pooling,
            first_diff=min(first_diff, len(ids_b) - 1),
            end_diff=end_b,
        )

        pool_a = ref_a.to(dtype=torch.float64)[mask_a].mean(dim=0).to(dtype=torch.float32)
        pool_b = ref_b.to(dtype=torch.float64)[mask_b].mean(dim=0).to(dtype=torch.float32)
        a_pooled_list.append(pool_a)
        b_pooled_list.append(pool_b)
        if include_encoder:
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
                enc_a.to(dtype=torch.float64)[mask_a_enc].mean(dim=0).to(dtype=torch.float32)
            )
            enc_b_pooled_list.append(
                enc_b.to(dtype=torch.float64)[mask_b_enc].mean(dim=0).to(dtype=torch.float32)
            )
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
                "len_a": int(enc_a.shape[0]),
                "len_b": int(enc_b.shape[0]),
                "pooled_rows_a": int(mask_a.sum().item()),
                "pooled_rows_b": int(mask_b.sum().item()),
            }
        )

    a_mat = torch.stack(a_pooled_list, dim=0)
    b_mat = torch.stack(b_pooled_list, dim=0)
    w_vec = torch.tensor(weights_list, dtype=torch.float32)

    raw_vec = _cv.mean_difference(a_mat, b_mat, w_vec, unit_mean=unit_mean)
    tensors_to_save: dict[str, torch.Tensor] = {}

    if apply_holds and pair_set.holds:
        hold_a_list: list[torch.Tensor] = []
        hold_b_list: list[torch.Tensor] = []
        hold_groups: list[str] = []
        hold_weights: list[float] = []
        for h_item in pair_set.holds:
            h_enc_a, h_tags_a, h_ids_a, h_dec_a = encode_prompt_for_vector(
                clip, h_item.a, encoded_references=encoded_references
            )
            h_enc_b, h_tags_b, h_ids_b, h_dec_b = encode_prompt_for_vector(
                clip, h_item.b, encoded_references=encoded_references
            )
            h_ref_a = refiner_run(h_enc_a)
            h_ref_b = refiner_run(h_enc_b)
            h_diff, h_end_a, h_end_b = _spans.differing_token_bounds(h_ids_a, h_ids_b)
            h_spans_a = _spans.spans(h_ids_a, h_tags_a.tolist(), h_dec_a)
            h_spans_b = _spans.spans(h_ids_b, h_tags_b.tolist(), h_dec_b)
            h_mask_a = _spans.pooling_mask(
                h_spans_a,
                pooling,
                first_diff=min(h_diff, len(h_ids_a) - 1),
                end_diff=h_end_a,
            )
            h_mask_b = _spans.pooling_mask(
                h_spans_b,
                pooling,
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
        basis = _cv.hold_basis(
            torch.stack(hold_a_list, dim=0),
            torch.stack(hold_b_list, dim=0),
            hold_groups,
            torch.tensor(hold_weights, dtype=torch.float32),
        )
        final_vec = _cv.project_out(raw_vec, basis)
        tensors_to_save["holds.post_refiner"] = basis
    else:
        final_vec = raw_vec

    tensors_to_save["vector.post_refiner"] = final_vec

    gen = torch.Generator(device="cpu").manual_seed(0)
    q_report = _cv.quality_report(a_mat, b_mat, w_vec, generator=gen)
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
            ref_hash.update(_cv._tensor_bf16_bytes(sp.rows))
        prefix_meta: Any = {"references_sha256": ref_hash.hexdigest()}
    else:
        prefix_meta = "none"

    if resolved_pair_path is not None:
        try:
            rel_pair_path = str(resolved_pair_path.relative_to(REPO_ROOT))
        except ValueError:
            rel_pair_path = resolved_pair_path.name
    else:
        rel_pair_path = str(pair_set.path)

    stem = (output_name or pair_set.name).strip()
    if not _cv.NAME_RE.match(stem):
        raise ValueError(f"output_name must match [a-z0-9_]+, got {stem!r}")

    points_meta: dict[str, Any] = {
        "post_refiner": {
            "unit": "mean_pair_difference",
            "raw_norm": raw_norm,
            "median_row_norm": median_ref_row_norm,
        }
    }
    if include_encoder:
        enc_a_mat = torch.stack(enc_a_pooled_list, dim=0)
        enc_b_mat = torch.stack(enc_b_pooled_list, dim=0)
        enc_raw_vec = _cv.mean_difference(enc_a_mat, enc_b_mat, w_vec)
        enc_raw_norm = float(
            torch.linalg.vector_norm(enc_raw_vec.to(dtype=torch.float64)).item()
        )
        if enc_raw_norm > 1e-12:
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

    metadata: dict[str, Any] = {
        "format": _cv.FORMAT_VERSION,
        "name": stem,
        "quality": pair_set.quality,
        "family": family,
        "pairs": {
            "path": rel_pair_path,
            "sha256": pair_set.sha256,
            "n_pairs": len(pair_set.pairs),
            "n_holds": len(pair_set.holds) if apply_holds else 0,
            "a_side": pair_set.a_side,
            "b_side": pair_set.b_side,
            "pair_spans": pair_spans,
        },
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
            "core_revision": _substrate.git_head(COMFY_ROOT) or "unknown",
            "pack_commit": _substrate.git_head(REPO_ROOT) or "unknown",
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "made_by": made_by,
        },
    }

    target_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(folder_paths.get_folder_paths(CONTROL_VECTORS_FOLDER)[0])
    )
    out_filename = _cv.default_vector_filename(
        stem,
        family,
        pooling=pooling,
        prefix=prefix_meta,
        apply_holds=apply_holds,
        n_holds=len(pair_set.holds),
        loras=recipe.get("loras"),
    )
    out_path = _cv.save_vector(
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
        "vector_channel_census": _cv.channel_census(final_vec),
    }

    full_report = {
        "vector_file": out_filename,
        "path": str(out_path),
        "metadata": metadata,
        "e2_1_census": e2_1_census,
        "pair_spans": pair_spans,
    }
    return out_path, full_report


@dataclasses.dataclass(frozen=True)
class _AppliedVectorSpec:
    vector_name: str
    vector: torch.Tensor
    strength: float
    mode: str
    mask: str
    start_percent: float
    end_percent: float
    metadata: dict[str, Any]
    advisories: tuple[str, ...]
    scale_mode: str = "raw"
    prompt: str = ""


def _step_in_window(
    transformer_options: dict[str, Any] | None,
    timestep: Any,
    start_percent: float,
    end_percent: float,
) -> bool:
    """Whether a control vector's `[start_percent, end_percent)` step window is
    active at the current call (`lora_branch.py::_Window` step-index convention)."""
    if start_percent >= end_percent:
        return False
    if start_percent <= 0.0 and end_percent >= 1.0:
        return True
    t_opts = transformer_options or {}
    sched = t_opts.get("sample_sigmas")
    cur = t_opts.get("sigmas")
    if cur is None and timestep is not None:
        cur = timestep
    if sched is not None and cur is not None:
        sched_t = (
            sched.detach().flatten().to(dtype=torch.float64, device="cpu")
            if isinstance(sched, torch.Tensor)
            else torch.tensor(list(sched), dtype=torch.float64)
        )
        if int(sched_t.numel()) >= 2:
            cur_val = float(
                cur.detach().flatten()[0].item()
                if isinstance(cur, torch.Tensor)
                else cur[0]
                if isinstance(cur, (list, tuple))
                else cur
            )
            n_steps = int(sched_t.numel()) - 1
            step_idx = int((sched_t[:n_steps] - cur_val).abs().argmin().item())
            frac = step_idx / n_steps
            return bool(start_percent <= frac < end_percent)
    if cur is not None:
        cur_val = float(
            cur.detach().flatten()[0].item()
            if isinstance(cur, torch.Tensor)
            else cur[0]
            if isinstance(cur, (list, tuple))
            else cur
        )
        if 0.0 <= cur_val <= 1.0:
            frac = 1.0 - cur_val
            return bool(start_percent <= frac < end_percent)
    return True


def _resolve_custom_inference_mask(
    sp: _AppliedVectorSpec,
    *,
    tags_raw: Any,
    tags_2d: torch.Tensor,
    payload: Mapping[str, Any],
    L: int,
    device: torch.device,
) -> torch.Tensor | None:
    if sp.mask in ("non_vision", "all_rows"):
        return None
    span_masks = getattr(tags_raw, "_h3_span_masks", None) if tags_raw is not None else None
    if span_masks is None:
        span_masks = payload.get("h3_span_masks")
    if span_masks is None and sp.prompt and sp.prompt.strip():
        span_masks = _spans.span_masks_from_prompt(sp.prompt, L, tags=tags_2d[0])
    if span_masks is None:
        raise ValueError(
            f"control vector {sp.vector_name!r} uses mask={sp.mask!r}, which requires "
            "conditioning from an H3 conditioning node (_h3_span_masks) or prompt=..."
        )
    return _spans.inference_mask(span_masks, sp.mask).to(device=device)


def _make_control_vector_wrapper(specs: tuple[_AppliedVectorSpec, ...]):
    logged_prefix_notes: set[tuple[str, str]] = set()
    logged_interactions: set[tuple[tuple[str, float, str], ...]] = set()

    def _effective_entry_mode(sp: _AppliedVectorSpec) -> str:
        if sp.mode == "add" and sp.scale_mode == "norm_matched":
            return "add_norm_matched"
        return sp.mode

    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        if context is None or not specs:
            return executor(x, timestep, context, transformer_options, **kwargs)
        active_specs = [
            sp
            for sp in specs
            if sp.strength != 0.0
            and _step_in_window(
                transformer_options, timestep, sp.start_percent, sp.end_percent
            )
        ]
        if not active_specs:
            return executor(x, timestep, context, transformer_options, **kwargs)

        payload = kwargs.get("minimax_payload") or {}
        tags_raw = payload.get("text_token_tags")
        B, L, _D = context.shape
        if tags_raw is not None:
            tags = torch.as_tensor(tags_raw, device=context.device, dtype=torch.long)
            if tags.ndim == 1:
                tags = tags.unsqueeze(0).expand(B, -1)
        else:
            if any(sp.mask != "all_rows" for sp in active_specs):
                raise ValueError(
                    "minimax_payload is missing 'text_token_tags'; cannot apply "
                    "non_vision mask without token tags"
                )
            tags = torch.ones((B, L), device=context.device, dtype=torch.long)

        has_vision = bool((tags == 0).any().item())
        render_prefix_kind = "references_or_keyframes" if has_vision else "none"
        for sp in active_specs:
            vec_prefix = sp.metadata.get("measured_under", {}).get("prefix", "none")
            vec_prefix_kind = "none" if vec_prefix == "none" else "references_or_keyframes"
            note_key = (sp.vector_name, render_prefix_kind)
            if vec_prefix != render_prefix_kind and note_key not in logged_prefix_notes:
                logged_prefix_notes.add(note_key)
                logger.warning(
                    "[h3] control vector %s: measured_under.prefix=%r differs from render prefix (%s)",
                    sp.vector_name,
                    vec_prefix,
                    render_prefix_kind,
                )

        # Group consecutive specs with the same (mask, prompt) so stacked additive
        # vectors and inference-time 'hold' vectors accumulate in float64 before
        # casting back to context.dtype (`CONTRACT.md` §3 / paper §2.1).
        cur = context
        i = 0
        n = len(active_specs)
        while i < n:
            sp = active_specs[i]
            group_key = (sp.mask, sp.prompt)
            custom_mask = _resolve_custom_inference_mask(
                sp,
                tags_raw=tags_raw,
                tags_2d=tags,
                payload=payload,
                L=L,
                device=context.device,
            )
            mask_vision = sp.mask != "all_rows"
            group_specs: list[_AppliedVectorSpec] = [sp]
            batch_entries: list[tuple[torch.Tensor, float, str]] = [
                (sp.vector, sp.strength, _effective_entry_mode(sp))
            ]
            j = i + 1
            while (
                j < n
                and (active_specs[j].mask, active_specs[j].prompt)
                == group_key
            ):
                group_specs.append(active_specs[j])
                batch_entries.append(
                    (
                        active_specs[j].vector,
                        active_specs[j].strength,
                        _effective_entry_mode(active_specs[j]),
                    )
                )
                j += 1

            add_inter_entries: list[tuple[str, torch.Tensor, float]] = []
            for gsp in group_specs:
                em = _effective_entry_mode(gsp)
                if em in ("add", "add_norm_matched"):
                    v_nrm = float(torch.linalg.vector_norm(gsp.vector.detach().to(dtype=torch.float64)).item())
                    eff_s = (
                        gsp.strength * (_cv.NORM_MATCHED_TARGET / v_nrm)
                        if (em == "add_norm_matched" and v_nrm > 1e-12)
                        else gsp.strength
                    )
                    add_inter_entries.append((gsp.vector_name, gsp.vector, eff_s))
            if len(add_inter_entries) >= 2:
                inter_key = tuple((gsp.vector_name, gsp.strength, _effective_entry_mode(gsp)) for gsp in group_specs)
                if inter_key not in logged_interactions:
                    logged_interactions.add(inter_key)
                    inter = _cv.vector_interactions(add_inter_entries)
                    pair_notes = ", ".join(
                        f"{p['a']} x {p['b']}: raw_cos={p['cosine']:+.3f} push_cos={p['signed_cosine']:+.3f} 2*s_i*s_j*(v_i.v_j)={p['interaction_sq']:+.1f}"
                        for p in inter["pairs"]
                    )
                    logger.info(
                        "[h3] stacked control vectors combined_norm=%.2f (sum_individual=%.2f; %s)",
                        inter["combined_norm"],
                        inter["sum_individual_norms"],
                        pair_notes,
                    )

            cur = _cv.apply_post_refiner(
                cur,
                tags,
                batch_entries,
                scale_mode="raw",
                mask=custom_mask,
                mask_vision=mask_vision,
            )
            i = j

        return executor(x, timestep, cur, transformer_options, **kwargs)

    return wrapper


def resolve_vector_file_path(vector_name: str | Path) -> Path:
    raw = str(vector_name).strip()
    if not raw or raw == "none":
        raise ValueError("vector_name is 'none'")
    cand = Path(raw)
    if cand.is_file():
        return cand.resolve()
    try:
        full = folder_paths.get_full_path(CONTROL_VECTORS_FOLDER, raw)
        if full is not None and Path(full).is_file():
            return Path(full).resolve()
    except Exception:
        pass
    for folder in folder_paths.get_folder_paths(CONTROL_VECTORS_FOLDER):
        p = Path(folder) / raw
        if p.is_file():
            return p.resolve()
        if not raw.endswith(".safetensors"):
            p_st = Path(folder) / f"{raw}.safetensors"
            if p_st.is_file():
                return p_st.resolve()
    raise ValueError(
        f"control vector file {vector_name!r} not found in models/{CONTROL_VECTORS_FOLDER}/"
    )


def attach_control_vector(
    model: Any,
    vector_name: str | Path,
    *,
    strength: float = 1.0,
    mode: str = "add",
    scale_mode: str = "raw",
    mask: str = "non_vision",
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    allow_recipe_mismatch: bool = False,
    loaded_prefix: str | dict[str, Any] | None = None,
    prompt: str = "",
) -> Any:
    """Clone `model` with `vector_name` attached on `WrappersMP.DIFFUSION_MODEL`.

    Validates the model type, vector format, family, and recipe even when
    `strength == 0.0`."""
    dm = model.get_model_object("diffusion_model")
    if not _is_minimax_h3(dm):
        raise ValueError("MiniMaxH3ApplyControlVectors requires a MiniMax H3 diffusion model")
    if mode not in _cv.APPLY_MODES:
        raise ValueError(f"mode must be one of {_cv.APPLY_MODES}, got {mode!r}")
    if scale_mode not in _cv.SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {_cv.SCALE_MODES}, got {scale_mode!r}")
    if mask not in MASK_MODES:
        raise ValueError(f"mask must be one of {MASK_MODES}, got {mask!r}")
    if not math.isfinite(float(strength)):
        raise ValueError(f"strength must be finite, got {strength!r}")
    sp_f = float(start_percent)
    ep_f = float(end_percent)
    if not (math.isfinite(sp_f) and math.isfinite(ep_f)) or not (0.0 <= sp_f <= 1.0) or not (0.0 <= ep_f <= 1.0) or sp_f > ep_f:
        raise ValueError(
            f"start_percent and end_percent must satisfy 0.0 <= start_percent <= end_percent <= 1.0, "
            f"got start_percent={start_percent!r}, end_percent={end_percent!r}"
        )

    m = model.clone()
    raw_name = str(vector_name).strip()
    if raw_name == "none":
        return m

    vec_path = resolve_vector_file_path(vector_name)
    tensors, metadata = _cv.load_vector(vec_path)
    if "vector.post_refiner" not in tensors:
        raise ValueError(
            f"control vector {vec_path.name} has no 'vector.post_refiner' tensor"
        )

    loaded_family = _cv.detect_family(model)
    loaded_recipe = extract_model_recipe(model)
    # Compare checkpoint + loras (encoder is only known at Make time unless passed)
    recipe_to_check = {
        "checkpoint": loaded_recipe["checkpoint"],
        "loras": loaded_recipe["loras"],
        "encoder": metadata["recipe"].get("encoder", loaded_recipe["encoder"]),
    }
    advisories = _cv.check_vector_compatibility(
        metadata,
        loaded_family=loaded_family,
        loaded_recipe=recipe_to_check,
        loaded_prefix=loaded_prefix,
        allow_recipe_mismatch=allow_recipe_mismatch,
    )
    for msg in advisories:
        logger.warning("[h3] %s", msg)

    t_opts = m.model_options.setdefault("transformer_options", {})
    existing: tuple[_AppliedVectorSpec, ...] = tuple(t_opts.get(WRAPPER_KEY, ()))
    new_spec = _AppliedVectorSpec(
        vector_name=vec_path.name,
        vector=tensors["vector.post_refiner"],
        strength=float(strength),
        mode=mode,
        mask=mask,
        start_percent=sp_f,
        end_percent=ep_f,
        metadata=metadata,
        advisories=tuple(advisories),
        scale_mode=scale_mode,
        prompt=str(prompt or ""),
    )
    updated = existing + (new_spec,)
    t_opts[WRAPPER_KEY] = updated

    # Replace the wrapper list under WRAPPER_KEY so all stacked specs execute
    # in one float64 accumulation pass per DiT step.
    wrappers = m.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
    wrappers[WRAPPER_KEY] = [_make_control_vector_wrapper(updated)]
    logger.info(
        "[h3] control vector %s attached (family=%s, strength=%g, mode=%s, scale_mode=%s, mask=%s, window=[%.2f, %.2f))",
        vec_path.name,
        loaded_family,
        float(strength),
        mode,
        scale_mode,
        mask,
        sp_f,
        ep_f,
    )
    return m


class MiniMaxH3MakeControlVector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        ref_enc_mod = _repo("reference_encode")
        return io.Schema(
            node_id="MiniMaxH3MakeControlVector",
            display_name="MiniMax H3 Make Control Vector",
            category="model/conditioning/minimax",
            description=(
                "Measures a contrastive control vector from a pair TOML file under "
                "prompt_bank/control_vectors/, encoding each pair through the loaded "
                "Qwen3-VL encoder and the DiT's condition_proj + token_refiner, pooling "
                "the differing span rows ('diff_span' by default) in float64, projecting out "
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
                        "Pair TOML filename under prompt_bank/control_vectors/ "
                        "(for example 'film_16mm_1974.toml' or 'lettering.toml')."
                    ),
                ),
                io.Combo.Input(
                    "pooling",
                    options=list(_cv.VALID_POOLING),
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
                        "Optional simple words, short phrase, or 'pole_a -> pole_b' contrast "
                        "(for example 'erratic handheld camera shake' or 'awkward eye-contact avoidance'). "
                        "When non-empty, compiles 10 canonical H3 scene pairs in memory with zero .toml file required."
                    ),
                ),
                io.Combo.Input(
                    "slot",
                    options=["auto", *_pair_compiler.CANONICAL_SLOTS],
                    default="auto",
                    optional=True,
                    tooltip=(
                        "Structural H3 slot for compiled words ('auto' infers from the words across "
                        "opening_framing, subject_behavior, interpersonal_dynamics, camera_motion, "
                        "speech_delivery, and soundscape)."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="vector_name"),
                io.String.Output(display_name="report"),
            ],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, pair_file="film_16mm_1974.toml", words="", slot="auto", **kwargs):
        if isinstance(words, str) and words.strip():
            return ("words", words.strip(), str(slot))
        try:
            full = resolve_pair_file_path(pair_file)
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
        slot="auto",
    ) -> io.NodeOutput:
        dm = model.get_model_object("diffusion_model")
        if not _is_minimax_h3(dm):
            raise ValueError("MiniMaxH3MakeControlVector requires a MiniMax H3 diffusion model")
        out_path, report = make_control_vector_from_pairs(
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
            slot=str(slot or "auto"),
        )
        return io.NodeOutput(out_path.name, json.dumps(report, indent=2, sort_keys=True))


class MiniMaxH3ApplyControlVectors(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ApplyControlVectors",
            display_name="MiniMax H3 Apply Control Vectors",
            category="model/conditioning/minimax",
            description=(
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
                    options=_vector_combo_options(),
                    default=_APPLY_DEFAULTS["vector_name"],
                    tooltip=(
                        "Control vector file (<name>__<family>.safetensors) under "
                        "models/control_vectors/, or 'none'."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=_APPLY_DEFAULTS["strength"],
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
                    options=list(_cv.APPLY_MODES),
                    default=_APPLY_DEFAULTS["mode"],
                    optional=True,
                    tooltip=(
                        "'add' (default) adds strength * v; 'add_norm_matched' normalizes v to "
                        "67.5 post-refiner shift norm at strength=1.0 across all pooling modes; "
                        "'hold' projects out strength * (delta_r . u_hat) * u_hat from the "
                        "accumulated additive steering shift without modifying the base prompt; "
                        "'project_out' removes strength * ((r . v) / ||v||^2) * v from the prompt "
                        "tokens (E4.4); 'amplify' adds strength * ((r . v) / ||v||^2) * v."
                    ),
                ),
                io.Combo.Input(
                    "mask",
                    options=list(MASK_MODES),
                    default=_APPLY_DEFAULTS["mask"],
                    tooltip=(
                        "'non_vision' (default) steers all tag-1 text rows and leaves tag-0 "
                        "vision rows untouched; 'scene_only' also leaves <d>...</d> dialogue "
                        "and (S1) speaker tags untouched; 'description_only' steers only scene "
                        "tokens inside integrated_multimodal_description: (fl2va) or "
                        "detailed_description: (ref2va); 'soundscape_only' steers only scene "
                        "tokens inside overall_soundscape:; 'after_dialogue' steers only scene "
                        "tokens after the last </d>; 'all_rows' is the unmasked ablation (E2.3)."
                    ),
                ),
                io.Float.Input(
                    "start_percent",
                    default=_APPLY_DEFAULTS["start_percent"],
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "Start of the active step window as a fraction of schedule steps "
                        "(step_idx / N_steps, matching MiniMaxH3Sol and MiniMaxH3LoRABranch)."
                    ),
                ),
                io.Float.Input(
                    "end_percent",
                    default=_APPLY_DEFAULTS["end_percent"],
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "End of the active step window as a fraction of schedule steps "
                        "(step_idx / N_steps, matching MiniMaxH3Sol and MiniMaxH3LoRABranch)."
                    ),
                ),
                io.Boolean.Input(
                    "allow_recipe_mismatch",
                    default=_APPLY_DEFAULTS["allow_recipe_mismatch"],
                    optional=True,
                    tooltip=(
                        "When False (default), refuses a vector whose checkpoint or LoRA "
                        "recipe differs from the loaded model's; when True, logs the "
                        "difference and applies."
                    ),
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def fingerprint_inputs(cls, vector_name=_APPLY_DEFAULTS["vector_name"], **kwargs):
        raw = str(vector_name).strip()
        if not raw or raw == "none":
            return "none"
        try:
            full = resolve_vector_file_path(raw)
            st = os.stat(full)
            return (str(full), st.st_mtime_ns, st.st_size)
        except Exception:
            return raw

    @classmethod
    def validate_inputs(
        cls,
        vector_name=_APPLY_DEFAULTS["vector_name"],
        strength=_APPLY_DEFAULTS["strength"],
        mode=_APPLY_DEFAULTS["mode"],
        mask=_APPLY_DEFAULTS["mask"],
        start_percent=_APPLY_DEFAULTS["start_percent"],
        end_percent=_APPLY_DEFAULTS["end_percent"],
        allow_recipe_mismatch=_APPLY_DEFAULTS["allow_recipe_mismatch"],
        **kwargs,
    ) -> bool | str:
        raw = str(vector_name).strip()
        if raw == "none":
            return True
        try:
            resolve_vector_file_path(raw)
        except Exception as exc:
            return str(exc)
        return True

    @classmethod
    def execute(
        cls,
        model,
        vector_name=_APPLY_DEFAULTS["vector_name"],
        strength=_APPLY_DEFAULTS["strength"],
        mode=_APPLY_DEFAULTS["mode"],
        mask=_APPLY_DEFAULTS["mask"],
        start_percent=_APPLY_DEFAULTS["start_percent"],
        end_percent=_APPLY_DEFAULTS["end_percent"],
        allow_recipe_mismatch=_APPLY_DEFAULTS["allow_recipe_mismatch"],
    ) -> io.NodeOutput:
        m = attach_control_vector(
            model,
            vector_name,
            strength=float(strength),
            mode=mode,
            mask=mask,
            start_percent=float(start_percent),
            end_percent=float(end_percent),
            allow_recipe_mismatch=bool(allow_recipe_mismatch),
        )
        return io.NodeOutput(m)


def match_conditioning_row_norm(
    conditioning: list[Any],
    *,
    reference_conditioning: list[Any] | None = None,
    target_median_row_norm: float = 0.0,
) -> list[Any]:
    """Scale tag-1 text rows of `conditioning` so their median L2 norm (excluding
    row 0) matches `reference_conditioning` or `target_median_row_norm` (`E4.3`).

    Leaves `tag == 0` vision rows, `minimax_token_tags`, `minimax_keyframes`,
    `minimax_refs`, and sequence length `L` bitwise untouched."""
    if not isinstance(conditioning, list) or not conditioning:
        raise ValueError("conditioning must be a non-empty list")

    target_norm = float(target_median_row_norm)
    if reference_conditioning is not None:
        if not isinstance(reference_conditioning, list) or not reference_conditioning:
            raise ValueError("reference_conditioning must be a non-empty list when provided")
        ref_t = reference_conditioning[0][0]
        ref_d = reference_conditioning[0][1] if len(reference_conditioning[0]) > 1 else {}
        ref_tags = ref_d.get("minimax_token_tags") if isinstance(ref_d, dict) else None
        target_norm = _cv.row_norm_census(ref_t, ref_tags)["median_row_norm_ex_row0"]

    if not math.isfinite(target_norm) or target_norm <= 0.0:
        raise ValueError(
            "MiniMaxH3LayerNormMatch requires either reference_conditioning or target_median_row_norm > 0"
        )

    out_cond: list[Any] = []
    for entry in conditioning:
        t = entry[0]
        d = dict(entry[1]) if len(entry) > 1 and isinstance(entry[1], dict) else {}
        tags = d.get("minimax_token_tags")
        cur_norm = _cv.row_norm_census(t, tags)["median_row_norm_ex_row0"]
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
        matched = match_conditioning_row_norm(
            conditioning,
            reference_conditioning=reference_conditioning,
            target_median_row_norm=float(target_median_row_norm),
        )
        return io.NodeOutput(matched)
