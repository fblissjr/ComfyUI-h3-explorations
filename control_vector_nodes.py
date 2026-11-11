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

import copy
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
_j_lens = _repo("j_lens")
_rules = _repo("h3_rules")
_substrate = _repo("substrate")

REPO_ROOT = Path(__file__).resolve().parent
COMFY_ROOT = REPO_ROOT.parent.parent
PAIR_BANK_DIR = REPO_ROOT / "prompt_bank" / "control_vectors"
USER_PAIR_BANK_DIR = REPO_ROOT / "data" / "control_vectors"
PROMPT_BANK_DIR = REPO_ROOT / "prompt_bank"

CONTROL_VECTORS_FOLDER = "control_vectors"
WRAPPER_KEY = "h3_control_vectors"
MASK_MODES = _spans.INFERENCE_MASK_MODES
_APPLY_DEFAULTS = {
    **_rules.CONTROL_VECTOR_DEFAULTS,
    **_rules.CONTROL_VECTOR_OPTIONAL_DEFAULTS,
}


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
    dit_path_override: str | Path | None = None,
    unet_path_override: str | Path | None = None,
    encoder_path_override: str | Path | None = None,
    loras_override: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract the `CONTRACT.md` §3 recipe dict `{checkpoint, loras, encoder}` from a
    loaded `ModelPatcher` and optional `CLIP` (or explicit file paths)."""
    # 1. Diffusion (DiT) checkpoint
    resolved_dit_override = dit_path_override if dit_path_override is not None else unet_path_override
    ckpt_path: Path | None = (
        Path(resolved_dit_override) if resolved_dit_override is not None else _patcher_checkpoint_path(model)
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
    for base in (USER_PAIR_BANK_DIR, PAIR_BANK_DIR, REPO_ROOT):
        p = base / raw_toml
        if p.is_file():
            return p.resolve()
    raise ValueError(
        f"pair file {pair_file!r} not found under {USER_PAIR_BANK_DIR} or {PAIR_BANK_DIR}"
    )


def encode_pair_set(
    clip: Any,
    pair_set: _cv.PairSet,
    *,
    encoded_references: Any | None = None,
    with_holds: bool = True,
    on_pair: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, float]]]:
    """Encode every pair, then every hold, of `pair_set` through `clip` (Qwen3-VL `layers.49`) in one
    pass, so the encoder stays resident while it works. Returns `(encoded_pairs, encoded_holds,
    encoder_row_censuses)`; each encoded entry carries both sides' rows, tags, ids and spans and the
    differing-token bounds, which is what `control_vectors.pool_pair_set` reads. The node and
    `bench/control_vectors.py` both build from this. `on_pair(entry)` is called after each pair."""
    encoded_pairs: list[dict[str, Any]] = []
    enc_censuses: list[dict[str, float]] = []
    for idx, item in enumerate(pair_set.pairs):
        enc_a, tags_a, ids_a, dec_a = encode_prompt_for_vector(clip, item.a, encoded_references=encoded_references)
        enc_b, tags_b, ids_b, dec_b = encode_prompt_for_vector(clip, item.b, encoded_references=encoded_references)
        enc_censuses.append(_cv.row_norm_census(enc_a, tags_a))
        enc_censuses.append(_cv.row_norm_census(enc_b, tags_b))
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
    mode: str = "auto",
    target_prompt: str = "",
) -> tuple[Path, dict[str, Any]]:
    """Build and save a `.safetensors` control vector from `pair_path`, `pair_set`,
    or in-place `words` + `target_prompt` contrast clauses (zero `.toml` file required),
    returning `(saved_path, full_report_dict)`."""
    if pooling not in _cv.VALID_POOLING:
        raise ValueError(f"pooling must be one of {_cv.VALID_POOLING}, got {pooling!r}")

    family = _cv.detect_family(model)
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
        compile_mode = _pair_compiler.infer_compile_mode(
            req_mode,
            ref_images=encoded_references,
            target_prompt=tp_str,
            words=words.strip(),
        )
        pair_set = _pair_compiler.compile_words_to_pairset(
            words.strip(),
            name=output_name.strip() or None,
            mode=compile_mode,
            target_prompt=tp_str,
        )
    else:
        if pair_path is None:
            raise ValueError("either pair_path, pair_set, or words must be provided")
        resolved_pair_path = resolve_pair_file_path(pair_path)
        pair_set = _cv.load_pairs(resolved_pair_path, bank_dir=PROMPT_BANK_DIR)
    recipe = extract_model_recipe(model, clip)
    refiner_run = build_refiner_runner(model)

    # Encode every pair and hold through `clip` first, then refine them all: `clip` and `model`
    # (`ModelPatcher`) do not swap in and out of VRAM on every pair inside ComfyUI.
    encoded_pairs, encoded_holds, enc_censuses = encode_pair_set(
        clip, pair_set, encoded_references=encoded_references, with_holds=apply_holds
    )
    refined_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    ref_censuses: list[dict[str, float]] = []
    for enc in encoded_pairs:
        ref_a = refiner_run(enc["enc_a"])
        ref_b = refiner_run(enc["enc_b"])
        refined_pairs.append((ref_a, ref_b))
        ref_censuses.append(_cv.row_norm_census(ref_a, enc["tags_a"]))
        ref_censuses.append(_cv.row_norm_census(ref_b, enc["tags_b"]))
    refined_holds = [(refiner_run(h["enc_a"]), refiner_run(h["enc_b"])) for h in encoded_holds]

    # The one pair build, shared with `bench/control_vectors.py` (`control_vectors.pool_pair_set`).
    pooled = _cv.pool_pair_set(
        encoded_pairs,
        refined_pairs,
        encoded_holds,
        refined_holds,
        pooling=pooling,
        unit_mean=unit_mean,
        include_encoder=include_encoder,
    )
    a_mat, b_mat, w_vec = pooled.a_mat, pooled.b_mat, pooled.w_vec
    pair_spans = pooled.pair_spans
    final_vec = pooled.vector
    tensors_to_save: dict[str, torch.Tensor] = {}
    if pooled.hold_basis is not None:
        tensors_to_save["holds.post_refiner"] = pooled.hold_basis
    tensors_to_save["vector.post_refiner"] = final_vec

    gen = torch.Generator(device="cpu").manual_seed(0)
    q_report = _cv.quality_report(a_mat, b_mat, w_vec, generator=gen)
    if _j_lens is not None and _j_lens.bundle_available():
        try:
            q_report["j_lens_readout"] = _j_lens.decode_vector(
                final_vec, space="post_refiner", family=family
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
    if include_encoder and pooled.enc_raw_vec is not None:
        enc_vec, enc_raw_norm = _cv.encoder_point(pooled.enc_raw_vec)
        if enc_vec is not None:
            tensors_to_save["vector.encoder"] = enc_vec
            points_meta["encoder"] = {
                "unit": "median_row_norm",
                "raw_norm": enc_raw_norm,
                "median_row_norm": median_enc_row_norm_ex0,
                "row0_excluded": True,
            }

    pairs_dict_node = _cv.pairs_record(
        pair_set,
        path=rel_pair_path,
        pair_spans=pair_spans,
        n_holds=len(pair_set.holds) if apply_holds else 0,
        inline=resolved_pair_path is None,
        words_spec=words if isinstance(words, str) else None,
        inline_holds=apply_holds,
    )

    metadata: dict[str, Any] = {
        "format": _cv.FORMAT_VERSION,
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
    video_scale: float = 1.0
    audio_scale: float = 1.0
    split_end_percent: float = 1.0
    ref_vae_end_percent: float = 1.0
    ref_vae_aug: float = 0.999


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


def _stream_time(timestep: Any) -> float | None:
    """The video stream's time `t_v` at this call, read from the wrapper's `timestep` exactly as core
    reads it (`comfy/ldm/minimax/model.py::_forward`: the timestep is `sigma * 1000` and
    `t_v = 1 - sigma`). 0 is noise and 1 is clean. None when there is no timestep to read."""
    if timestep is None:
        return None
    if isinstance(timestep, torch.Tensor):
        raw = float(timestep.detach().flatten()[0].item())
    elif isinstance(timestep, (list, tuple)):
        raw = float(timestep[0])
    else:
        raw = float(timestep)
    return 1.0 - max(raw / 1000.0, 1e-6)


def _resolve_custom_inference_mask(
    sp: _AppliedVectorSpec,
    *,
    tags_raw: Any,
    tags_2d: torch.Tensor,
    payload: Mapping[str, Any],
    L: int,
    device: torch.device,
    kwarg_span_masks: Any = None,
) -> torch.Tensor | None:
    if sp.mask in ("non_vision", "all_rows"):
        return None
    span_masks = kwarg_span_masks
    if span_masks is None:
        span_masks = payload.get("h3_span_masks")
    if span_masks is None and tags_raw is not None:
        span_masks = getattr(tags_raw, "_h3_span_masks", None)
    if span_masks is None and sp.prompt and sp.prompt.strip():
        span_masks = _spans.span_masks_from_prompt(sp.prompt, L, tags=tags_2d[0])
    if span_masks is None:
        raise ValueError(
            f"control vector {sp.vector_name!r} uses mask={sp.mask!r}, which requires "
            "conditioning from an H3 conditioning node (h3_span_masks / _h3_span_masks) or prompt=..."
        )
    return _spans.inference_mask(span_masks, sp.mask).to(device=device)


def _validate_float_bounds(
    name: str,
    val: Any,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> float:
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite, got {val!r}")
    if lo is not None and fval < lo:
        raise ValueError(f"{name} must be >= {lo}, got {val!r}")
    if hi is not None and fval > hi:
        raise ValueError(f"{name} must be <= {hi}, got {val!r}")
    return fval


def _effective_entry_mode(sp: _AppliedVectorSpec) -> str:
    if sp.mode == "add" and sp.scale_mode == "norm_matched":
        return "add_norm_matched"
    return sp.mode


def _has_visual_cond(payload: Mapping[str, Any]) -> bool:
    if payload.get("cond_video_latents"):
        return True
    for r in payload.get("refs") or ():
        if isinstance(r, Mapping) and (
            "latent" in r
            or r.get("kind") in ("image", "video", "video_audio")
            or r.get("type") == "image"
        ):
            return True
    for kf in payload.get("keyframes") or ():
        if isinstance(kf, Mapping) and kf.get("latent") is not None:
            return True
    return False


def _strip_visual_cond_entries(
    keyframes: Any, refs: Any
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
    kept_kf: list[dict[str, Any]] = []
    for kf in keyframes or ():
        if isinstance(kf, Mapping) and kf.get("audio_latent") is not None:
            kept_kf.append({k: v for k, v in kf.items() if k != "latent"})

    kept_refs: list[dict[str, Any]] = []
    for blk in refs or ():
        if not isinstance(blk, Mapping):
            continue
        kind = blk.get("kind")
        if kind == "audio":
            kept_refs.append(dict(blk))
        elif (
            kind == "video_audio"
            and int(blk.get("ref_audio_t", 0) or 0) > 0
            and blk.get("audio_latent") is not None
        ):
            audio_only = {
                k: v
                for k, v in blk.items()
                if k not in ("latent", "latent_t", "latent_h", "latent_w")
            }
            audio_only["kind"] = "audio"
            kept_refs.append(audio_only)

    return (kept_kf or None), (kept_refs or None)


def _drop_visual_cond_from_layout(orig_layout: Any) -> tuple[Any, int]:
    """Slice `'cond'` and `'ref_img'` segments out of a `PackedLayout` while preserving
    every surviving token's 3D RoPE `position_ids` (including `_ref_t_span` cursor shifts
    and `EditGuider.align_positions` offsets) and `layout.signature`."""
    if not (
        hasattr(orig_layout, "segments")
        and hasattr(orig_layout, "position_ids")
        and hasattr(orig_layout, "signature")
    ):
        return orig_layout, 0

    dropped_rows = sum(
        int(b) - int(a)
        for a, b, kind in orig_layout.segments
        if kind in ("cond", "ref_img")
    )
    if dropped_rows == 0:
        return orig_layout, 0

    kept_pos: list[torch.Tensor] = []
    new_segments: list[tuple[int, int, str]] = []
    new_img_pos: list[torch.Tensor] = []
    new_img_update: list[torch.Tensor] = []
    new_audio_pos: list[torch.Tensor] = []
    new_audio_update: list[torch.Tensor] = []
    off = 0
    for a, b, kind in orig_layout.segments:
        if kind in ("cond", "ref_img"):
            continue
        n = int(b) - int(a)
        kept_pos.append(orig_layout.position_ids[int(a) : int(b)])
        new_segments.append((off, off + n, str(kind)))
        if kind == "video":
            new_img_pos.append(torch.arange(off, off + n))
            new_img_update.append(torch.ones(n, dtype=torch.bool))
        elif kind in ("cond_audio", "ref_audio"):
            new_audio_pos.append(torch.arange(off, off + n))
            new_audio_update.append(torch.zeros(n, dtype=torch.bool))
        elif kind == "audio":
            new_audio_pos.append(torch.arange(off, off + n))
            new_audio_update.append(torch.ones(n, dtype=torch.bool))
        off += n

    released = copy.copy(orig_layout)
    released.seq_len = off
    released.position_ids = torch.cat(kept_pos, dim=0)
    released.img_pos = (
        torch.cat(new_img_pos) if new_img_pos else torch.empty(0, dtype=torch.long)
    )
    released.img_update = (
        torch.cat(new_img_update) if new_img_update else torch.empty(0, dtype=torch.bool)
    )
    released.audio_pos = (
        torch.cat(new_audio_pos) if new_audio_pos else torch.empty(0, dtype=torch.long)
    )
    released.audio_update = (
        torch.cat(new_audio_update)
        if new_audio_update
        else torch.empty(0, dtype=torch.bool)
    )
    released.signature = orig_layout.signature
    released.segments = new_segments
    return released, dropped_rows


def _make_control_vector_wrapper(specs: tuple[_AppliedVectorSpec, ...]):
    logged_prefix_notes: set[tuple[str, str]] = set()
    logged_interactions: set[tuple[tuple[str, float, str], ...]] = set()
    logged_ref_notes: set[str] = set()
    logged_prompt_couplings: set[tuple[str, str]] = set()
    logged_steered_rows: set[tuple[int, str, int, int, str]] = set()
    resolved_prompt_vectors: dict[tuple[str, str], torch.Tensor] = {}
    last_resolved_by_spec: dict[str, torch.Tensor] = {}
    # Keyed by id(orig_layout) with a strong ref to orig_layout in the value so
    # released PackedLayouts are built once per run (~0 us overhead on later steps).
    released_layout_cache: dict[
        int, tuple[Any, Any, int, list[dict[str, Any]] | None, list[dict[str, Any]] | None]
    ] = {}

    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        if context is None or not specs:
            return executor(x, timestep, context, transformer_options, **kwargs)

        payload = kwargs.get("minimax_payload") or {}
        call_kwargs = dict(kwargs)
        ref_specs = [
            sp for sp in specs if sp.strength != 0.0 or sp.vector_name == "none"
        ]
        min_ref_end = min((sp.ref_vae_end_percent for sp in ref_specs), default=1.0)
        min_ref_aug = min((sp.ref_vae_aug for sp in ref_specs), default=0.999)
        if (min_ref_end < 1.0 or min_ref_aug < 0.999) and _has_visual_cond(payload):
            ref_active = _step_in_window(transformer_options, timestep, 0.0, min_ref_end)
            new_payload = dict(payload)
            if not ref_active:
                orig_layout = payload.get("layout")
                target_sig: tuple[int, int, int, int, int] | None = None
                if (
                    isinstance(x, (list, tuple))
                    and len(x) >= 2
                    and torch.is_tensor(x[0])
                    and x[0].ndim == 5
                    and torch.is_tensor(x[1])
                    and x[1].ndim >= 1
                ):
                    target_sig = (
                        int(context.shape[1]),
                        int(x[0].shape[2]),
                        (int(x[0].shape[3]) + 1) // 2 * 2,
                        (int(x[0].shape[4]) + 1) // 2 * 2,
                        int(x[1].shape[-1]),
                    )
                if target_sig is not None and (
                    orig_layout is None
                    or getattr(orig_layout, "signature", None) != target_sig
                ):
                    from comfy.ldm.minimax.model import PackedLayout

                    orig_layout = PackedLayout(
                        *target_sig,
                        keyframes=payload.get("keyframes"),
                        refs=payload.get("refs"),
                    )

                cache_entry = (
                    released_layout_cache.get(id(orig_layout))
                    if orig_layout is not None
                    else None
                )
                if cache_entry is not None and cache_entry[0] is orig_layout:
                    _, released_layout, dropped_rows, kept_kf, kept_refs = cache_entry
                else:
                    released_layout, dropped_rows = _drop_visual_cond_from_layout(orig_layout)
                    kept_kf, kept_refs = _strip_visual_cond_entries(
                        payload.get("keyframes"), payload.get("refs")
                    )
                    if orig_layout is not None and len(released_layout_cache) < 32:
                        released_layout_cache[id(orig_layout)] = (
                            orig_layout,
                            released_layout,
                            dropped_rows,
                            kept_kf,
                            kept_refs,
                        )

                new_payload["cond_video_latents"] = []
                if kept_kf is not None:
                    new_payload["keyframes"] = kept_kf
                else:
                    new_payload.pop("keyframes", None)
                if kept_refs is not None:
                    new_payload["refs"] = kept_refs
                else:
                    new_payload.pop("refs", None)
                if orig_layout is not None:
                    new_payload["layout"] = released_layout

                if "ref_gated" not in logged_ref_notes:
                    logged_ref_notes.add("ref_gated")
                    orig_seq = getattr(orig_layout, "seq_len", None)
                    rel_seq = getattr(released_layout, "seq_len", None)
                    logger.info(
                        "[h3] control vector ref_vae_end_percent=%.2f: released %d visual cond rows (PackedLayout seq_len %s -> %s, 3D RoPE positions preserved)",
                        min_ref_end,
                        dropped_rows,
                        orig_seq,
                        rel_seq,
                    )
            elif min_ref_aug < 0.999:
                # Core mixes each condition row as `aug` of the picture and the rest noise, and labels
                # the row `max(t_v, aug)` (`comfy/ldm/minimax/model.py`, `_cond_video_rows` and `seg_t`
                # in `_forward`). Sent as given, a weight below the stream's time leaves the rows noisier
                # than their label on the later steps: the model is told noise is picture. Sending
                # `max(t_v, weight)` keeps the share and the label equal on every step. The picture sits
                # at the weight until the stream is cleaner than that, and at the stream's own time after.
                stream_t = _stream_time(timestep)
                eff_aug = float(min_ref_aug) if stream_t is None else max(float(min_ref_aug), stream_t)
                new_payload["visual_cond_noise_aug"] = eff_aug
                if "ref_aug" not in logged_ref_notes:
                    logged_ref_notes.add("ref_aug")
                    logger.info(
                        "[h3] control vector ref_vae_aug=%.3f: condition rows held at max(weight, stream time), "
                        "so their noise matches their label on every step",
                        min_ref_aug,
                    )
            call_kwargs["minimax_payload"] = new_payload
            payload = new_payload

        active_specs = [
            sp
            for sp in specs
            if sp.strength != 0.0
            and _step_in_window(
                transformer_options, timestep, sp.start_percent, sp.end_percent
            )
        ]
        if not active_specs:
            return executor(x, timestep, context, transformer_options, **call_kwargs)

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
        kwarg_span_masks = kwargs.get("h3_span_masks")
        span_masks_top = kwarg_span_masks
        if span_masks_top is None:
            span_masks_top = payload.get("h3_span_masks")
        if span_masks_top is None and tags_raw is not None:
            span_masks_top = getattr(tags_raw, "_h3_span_masks", None)
        edit_pass = payload.get("h3_edit_pass")
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        is_pure_uncond = (
            edit_pass not in ("original", "edited")
            and isinstance(cond_or_uncond, (list, tuple))
            and 0 not in cond_or_uncond
        )
        if edit_pass in ("original", "edited"):
            ep_key = (",".join(sp.vector_name for sp in active_specs), str(edit_pass))
            if ep_key not in logged_prefix_notes:
                logged_prefix_notes.add(ep_key)
                logger.info(
                    "[h3] control vector (%s) running alongside MiniMaxH3EditGuider (%s pass): "
                    "applying 1x post-refiner shift on both passes so edit_scale amplifies only the prompt edit",
                    ep_key[0],
                    edit_pass,
                )
        live_tail_text = (
            ""
            if is_pure_uncond
            else (
                getattr(span_masks_top.get("scene"), "_h3_tail_text", "")
                if isinstance(span_masks_top, dict) and "scene" in span_masks_top
                else ""
            )
        )

        def _spec_effective_vector(sp: _AppliedVectorSpec) -> torch.Tensor:
            eff_prompt = (sp.prompt or live_tail_text or "").strip()
            vec_out = sp.vector
            if sp.metadata.get("deferred_against_prompt"):
                if eff_prompt:
                    cache_key = (sp.vector_name, eff_prompt)
                    if cache_key not in resolved_prompt_vectors:
                        if _j_lens.bundle_available():
                            res_synth = _j_lens.synthesize_from_words(
                                sp.metadata["words_raw"],
                                sp.metadata["against_raw"],
                                family=sp.metadata.get("synth_family", "fl2va"),
                                hold_words=sp.metadata.get("hold_raw") or None,
                                alpha=float(sp.metadata.get("hold_alpha", 1.0)),
                                prompt=eff_prompt,
                            )
                            resolved_prompt_vectors[cache_key] = res_synth["vector.post_refiner"]
                            logger.info(
                                "[h3] resolved %s against live prompt subject %s -> PR(+) %s | PR(-) %s",
                                sp.vector_name,
                                res_synth["neg_words_matched"],
                                res_synth["readout"].get("pos_summary", ""),
                                res_synth["readout"].get("neg_summary", ""),
                            )
                        else:
                            resolved_prompt_vectors[cache_key] = sp.vector
                    vec_out = resolved_prompt_vectors[cache_key]
                    last_resolved_by_spec[sp.vector_name] = vec_out
                else:
                    if (sp.vector_name, "missing_prompt_warn") not in logged_prompt_couplings:
                        logged_prompt_couplings.add((sp.vector_name, "missing_prompt_warn"))
                        logger.warning(
                            "[h3] control vector %r uses against_words=%r, but conditioning has no _h3_span_masks "
                            "(wire MiniMaxH3Conditioning or MiniMaxH3ReferenceConditioning); falling back to @replace_human",
                            sp.vector_name,
                            sp.metadata.get("against_raw", "@prompt_subject"),
                        )
                    vec_out = last_resolved_by_spec.get(sp.vector_name, sp.vector)
            if eff_prompt:
                pc_key = (sp.vector_name, eff_prompt[:120])
                if pc_key not in logged_prompt_couplings:
                    logged_prompt_couplings.add(pc_key)
                    if _j_lens.bundle_available():
                        try:
                            raw_w = str(sp.metadata.get("words_raw", ""))
                            pairs_m = sp.metadata.get("pairs") if isinstance(sp.metadata.get("pairs"), dict) else {}
                            b_side_w = str(pairs_m.get("b_side", "")) if isinstance(pairs_m, dict) else ""
                            seed_w = _j_lens._parse_word_list(raw_w or b_side_w)
                            seed_w = [w for w in seed_w if w not in _j_lens._PROMPT_RECEPTOR_STOPWORDS][:6]
                            pc = _j_lens.analyze_prompt_coupling(
                                vec_out,
                                eff_prompt,
                                pos_words=seed_w or None,
                                family=str(sp.metadata.get("family", "fl2va")),
                                already_contrasted=bool(sp.metadata.get("deferred_against_prompt")),
                            )
                            logger.info(
                                "[h3] control vector %s prompt coupling: %s",
                                sp.vector_name,
                                pc.get("summary", ""),
                            )
                        except Exception:
                            pass
            return vec_out

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

        def _spec_stream_scale(sp: _AppliedVectorSpec, stream_attr: str | None) -> float:
            if stream_attr is None:
                return 1.0
            if not _step_in_window(transformer_options, timestep, 0.0, sp.split_end_percent):
                return 1.0
            return float(getattr(sp, stream_attr, 1.0))

        def _effective_spec_strength(sp: _AppliedVectorSpec, stream_attr: str | None) -> float:
            s_val = float(sp.strength) * _spec_stream_scale(sp, stream_attr)
            if _effective_entry_mode(sp) == "hold":
                return max(0.0, min(2.0, s_val))
            return s_val

        def _build_steered_context(base_ctx: torch.Tensor, stream_attr: str | None) -> torch.Tensor:
            # Group consecutive specs with the same (mask, prompt) so stacked additive
            # vectors and inference-time 'hold' vectors accumulate in float64 before
            # casting back to context.dtype (`CONTRACT.md` §3 / paper §2.1).
            cur_ctx = base_ctx
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
                    device=base_ctx.device,
                    kwarg_span_masks=kwarg_span_masks,
                )
                mask_vision = sp.mask != "all_rows"
                group_specs: list[_AppliedVectorSpec] = [sp]
                s0 = _effective_spec_strength(sp, stream_attr)
                v0 = _spec_effective_vector(sp)
                batch_entries: list[tuple[torch.Tensor, float, str]] = [
                    (v0, s0, _effective_entry_mode(sp))
                ]
                j = i + 1
                while (
                    j < n
                    and (active_specs[j].mask, active_specs[j].prompt)
                    == group_key
                ):
                    gnext = active_specs[j]
                    group_specs.append(gnext)
                    sj = _effective_spec_strength(gnext, stream_attr)
                    vj = _spec_effective_vector(gnext)
                    batch_entries.append(
                        (
                            vj,
                            sj,
                            _effective_entry_mode(gnext),
                        )
                    )
                    j += 1

                if (
                    j < n
                    and _effective_entry_mode(active_specs[j]) == "hold"
                    and any(em in ("add", "add_norm_matched") for (_v, _s, em) in batch_entries)
                ):
                    raise ValueError(
                        f"mode='hold' on {active_specs[j].vector_name!r} has mask={active_specs[j].mask!r}, "
                        f"which differs from preceding additive control vector mask={sp.mask!r}; "
                        "a 'hold' node must use the same mask as the additive node(s) it projects out of"
                    )

                add_inter_entries: list[tuple[str, torch.Tensor, float]] = []
                for gsp, (gv, gs_eff, em) in zip(group_specs, batch_entries):
                    if em in ("add", "add_norm_matched") and gs_eff != 0.0:
                        v_nrm = float(torch.linalg.vector_norm(gv.detach().to(dtype=torch.float64)).item())
                        eff_s = (
                            gs_eff * (_cv.NORM_MATCHED_TARGET / v_nrm)
                            if (em == "add_norm_matched" and v_nrm > 1e-12)
                            else gs_eff
                        )
                        add_inter_entries.append((gsp.vector_name, gv, eff_s))
                if len(add_inter_entries) >= 2:
                    inter_key = tuple(
                        (gsp.vector_name, round(gs_eff, 6), em)
                        for gsp, (_v, gs_eff, em) in zip(group_specs, batch_entries)
                    )
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

                if any(s_eff != 0.0 for (_v, s_eff, _m) in batch_entries):
                    if not is_pure_uncond:
                        row_key = (i, sp.mask, L, int(has_vision), str(edit_pass or ""))
                        if row_key not in logged_steered_rows:
                            logged_steered_rows.add(row_key)
                            eff_rm = _cv.resolve_post_refiner_row_mask(
                                tags,
                                batch_size=cur_ctx.shape[0],
                                seq_len=L,
                                device=cur_ctx.device,
                                mask=custom_mask,
                                mask_vision=mask_vision,
                            )
                            n_steered = int(eff_rm[0].sum().item())
                            logger.info(
                                "[h3] control vector (%s) steered %d/%d context rows (mask=%s%s)",
                                ",".join(gsp.vector_name for gsp in group_specs),
                                n_steered,
                                L,
                                sp.mask,
                                f", {edit_pass} pass" if edit_pass in ("original", "edited") else "",
                            )
                            if n_steered == 0 and _spans.split_fields_mask(sp.mask) is not None:
                                logger.warning(
                                    "[h3] control vector (%s) has no effect: mask=%s selects the prose of the "
                                    "named field(s), and this prompt has none of them or they read N/A",
                                    ",".join(gsp.vector_name for gsp in group_specs),
                                    sp.mask,
                                )
                    cur_ctx = _cv.apply_post_refiner(
                        cur_ctx,
                        tags,
                        batch_entries,
                        scale_mode="raw",
                        mask=custom_mask,
                        mask_vision=mask_vision,
                    )
                i = j
            return cur_ctx

        need_split = any(
            _spec_stream_scale(sp, "video_scale") != _spec_stream_scale(sp, "audio_scale")
            for sp in active_specs
        )
        if not need_split:
            cur = _build_steered_context(context, "video_scale")
            return executor(x, timestep, cur, transformer_options, **call_kwargs)

        ctx_video = _build_steered_context(context, "video_scale")
        ctx_audio = _build_steered_context(context, "audio_scale")
        out_vid = executor(x, timestep, ctx_video, transformer_options, **call_kwargs)
        out_aud = executor(x, timestep, ctx_audio, transformer_options, **call_kwargs)
        if (
            isinstance(out_vid, list)
            and len(out_vid) >= 2
            and isinstance(out_aud, (list, tuple))
            and len(out_aud) >= 2
        ):
            return [out_vid[0], out_aud[1], *out_vid[2:]]
        if (
            isinstance(out_vid, tuple)
            and len(out_vid) >= 2
            and isinstance(out_aud, (list, tuple))
            and len(out_aud) >= 2
        ):
            return (out_vid[0], out_aud[1], *out_vid[2:])
        return out_vid

    return wrapper


def _match_sibling_or_stem_in_dir(
    directory: Path,
    stem: str,
    preferred_family: str | None,
    preferred_pooling: str | None = "diff_span",
    *,
    prefer_lora: bool = True,
) -> Path | None:
    if not directory.is_dir():
        return None
    fams = (
        [preferred_family]
        if preferred_family in ("fl2va", "ref2va", "fasth3")
        else ["fl2va", "ref2va", "fasth3"]
    )
    sort_key = lambda p: (0 if ("__lora_" in p.name) == prefer_lora else 1, p.name)
    for fam in fams:
        if preferred_pooling:
            cands_pool = sorted(
                directory.glob(f"{stem}__{fam}__{preferred_pooling}*.safetensors"),
                key=sort_key,
            )
            if cands_pool:
                return cands_pool[0].resolve()
        cands_fam = sorted(
            directory.glob(f"{stem}__{fam}*.safetensors"),
            key=sort_key,
        )
        if cands_fam:
            return cands_fam[0].resolve()
    return None


def resolve_vector_file_path(
    vector_name: str | Path,
    *,
    preferred_family: str | None = None,
    prefer_lora: bool | None = None,
    log_swap: bool = True,
) -> Path:
    raw = str(vector_name).strip()
    if not raw or raw == "none":
        raise ValueError("vector_name is 'none'")

    def _maybe_swap_family_sibling(found: Path) -> Path:
        if preferred_family not in ("fl2va", "ref2va"):
            return found
        other_fam = "fl2va" if preferred_family == "ref2va" else "ref2va"
        tag_other = f"__{other_fam}"
        if tag_other not in found.name:
            return found
        stem = found.name.split(tag_other, 1)[0]
        pooling_hint = "diff_span" if "__diff_span" in found.name else (
            "from_first_diff" if "__from_first_diff" in found.name else None
        )
        eff_prefer_lora = ("__lora_" in found.name) if prefer_lora is None else bool(prefer_lora)
        sib = _match_sibling_or_stem_in_dir(
            found.parent,
            stem,
            preferred_family,
            preferred_pooling=pooling_hint,
            prefer_lora=eff_prefer_lora,
        )
        if sib is not None and sib.is_file():
            if log_swap:
                logger.info(
                    "[h3] auto-resolved sibling control vector %s -> %s for loaded family=%s",
                    found.name,
                    sib.name,
                    preferred_family,
                )
            return sib
        return found

    cand = Path(raw)
    if cand.is_file():
        return _maybe_swap_family_sibling(cand.resolve())
    try:
        full = folder_paths.get_full_path(CONTROL_VECTORS_FOLDER, raw)
        if full is not None and Path(full).is_file():
            return _maybe_swap_family_sibling(Path(full).resolve())
    except Exception:
        pass
    folders = [Path(f) for f in folder_paths.get_folder_paths(CONTROL_VECTORS_FOLDER)]
    for folder in folders:
        p = folder / raw
        if p.is_file():
            return _maybe_swap_family_sibling(p.resolve())
        if not raw.endswith(".safetensors"):
            p_st = folder / f"{raw}.safetensors"
            if p_st.is_file():
                return _maybe_swap_family_sibling(p_st.resolve())

    # Bare stem resolution (e.g. 'cursed_cctv' -> 'cursed_cctv__fl2va__diff_span__*.safetensors')
    clean_stem = raw[:-len(".safetensors")] if raw.endswith(".safetensors") else raw
    if "__" not in clean_stem and "/" not in clean_stem and "\\" not in clean_stem:
        eff_prefer = True if prefer_lora is None else bool(prefer_lora)
        for folder in folders:
            matched = _match_sibling_or_stem_in_dir(
                folder,
                clean_stem,
                preferred_family,
                preferred_pooling="diff_span",
                prefer_lora=eff_prefer,
            )
            if matched is not None:
                return matched

    raise ValueError(
        f"control vector file {vector_name!r} not found in models/{CONTROL_VECTORS_FOLDER}/"
    )


def attach_control_vector(
    model: Any,
    vector_name: str | Path = "none",
    *,
    strength: float = 1.0,
    mode: str = "add",
    scale_mode: str = "raw",
    mask: str = "scene_only",
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    allow_recipe_mismatch: bool = False,
    loaded_prefix: str | dict[str, Any] | None = None,
    prompt: str = "",
    video_scale: float = 1.0,
    audio_scale: float = 1.0,
    split_end_percent: float = 1.0,
    ref_vae_end_percent: float = 1.0,
    ref_vae_aug: float = 0.999,
    words: str = "",
    against_words: str = "",
    hold_words: str = "",
    hold_alpha: float = 1.0,
) -> Any:
    """Clone `model` with `vector_name` and/or inline J-Lens `words` attached on
    `WrappersMP.DIFFUSION_MODEL`.

    Validates the model type, vector format, family, and recipe even when
    `strength == 0.0`."""
    dm = model.get_model_object("diffusion_model")
    if not _is_minimax_h3(dm):
        raise ValueError("MiniMaxH3ApplyControlVectors requires a MiniMax H3 diffusion model")
    if mode not in _cv.APPLY_MODES:
        raise ValueError(f"mode must be one of {_cv.APPLY_MODES}, got {mode!r}")
    if scale_mode not in _cv.SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {_cv.SCALE_MODES}, got {scale_mode!r}")
    if not _spans.is_inference_mask(mask):
        raise ValueError(
            f"mask must be one of {MASK_MODES} or prompt fields joined by '+' ({_spans.PROMPT_FIELDS}), got {mask!r}"
        )
    if not math.isfinite(float(strength)):
        raise ValueError(f"strength must be finite, got {strength!r}")
    sp_f = float(start_percent)
    ep_f = float(end_percent)
    if not (math.isfinite(sp_f) and math.isfinite(ep_f)) or not (0.0 <= sp_f <= 1.0) or not (0.0 <= ep_f <= 1.0) or sp_f > ep_f:
        raise ValueError(
            f"start_percent and end_percent must satisfy 0.0 <= start_percent <= end_percent <= 1.0, "
            f"got start_percent={start_percent!r}, end_percent={end_percent!r}"
        )
    vs_f = _validate_float_bounds("video_scale", video_scale, lo=-5.0, hi=5.0)
    as_f = _validate_float_bounds("audio_scale", audio_scale, lo=-5.0, hi=5.0)
    sep_f = _validate_float_bounds("split_end_percent", split_end_percent, lo=0.0, hi=1.0)
    rvep_f = _validate_float_bounds("ref_vae_end_percent", ref_vae_end_percent, lo=0.0, hi=1.0)
    rva_f = _validate_float_bounds("ref_vae_aug", ref_vae_aug, lo=0.0, hi=1.0)
    ha_f = _validate_float_bounds("hold_alpha", hold_alpha, lo=0.0, hi=2.0)

    m = model.clone()
    raw_name = str(vector_name).strip()
    words_clean = str(words or "").strip()
    against_clean = str(against_words or "").strip()
    hold_clean = str(hold_words or "").strip()

    if (not raw_name or raw_name == "none") and not words_clean:
        if hold_clean:
            raise ValueError(
                "hold_words requires vector_name != 'none' or non-empty words; "
                "cannot apply hold_words when no control vector or inline words are selected"
            )
        if rvep_f < 1.0 or rva_f < 0.999:
            t_opts = m.model_options.setdefault("transformer_options", {})
            existing_none: tuple[_AppliedVectorSpec, ...] = tuple(t_opts.get(WRAPPER_KEY, ()))
            ref_spec = _AppliedVectorSpec(
                vector_name="none",
                vector=torch.zeros((_cv.POST_REFINER_DIM,), dtype=torch.float32),
                strength=0.0,
                mode=mode,
                mask=mask,
                start_percent=sp_f,
                end_percent=ep_f,
                metadata={},
                advisories=(),
                scale_mode=scale_mode,
                prompt=str(prompt or ""),
                video_scale=vs_f,
                audio_scale=as_f,
                split_end_percent=sep_f,
                ref_vae_end_percent=rvep_f,
                ref_vae_aug=rva_f,
            )
            updated_none = existing_none + (ref_spec,)
            t_opts[WRAPPER_KEY] = updated_none
            wrappers = m.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
            wrappers[WRAPPER_KEY] = [_make_control_vector_wrapper(updated_none)]
        return m

    loaded_family = _cv.detect_family(model)
    t_opts = m.model_options.setdefault("transformer_options", {})
    updated: tuple[_AppliedVectorSpec, ...] = tuple(t_opts.get(WRAPPER_KEY, ()))

    if raw_name and raw_name != "none" and words_clean:
        logger.warning(
            "[h3] both vector_name=%r and inline words=%r are set on the same MiniMaxH3ApplyControlVectors node; "
            "both vectors will be stacked (set vector_name='none' to apply only inline words)",
            raw_name,
            words_clean,
        )

    if raw_name and raw_name != "none":
        loaded_recipe = extract_model_recipe(model)
        vec_path = resolve_vector_file_path(
            vector_name,
            preferred_family=loaded_family,
            prefer_lora=bool(loaded_recipe.get("loras")),
        )
        tensors, metadata = _cv.load_vector(vec_path)
        if "vector.post_refiner" not in tensors:
            raise ValueError(
                f"control vector {vec_path.name} has no 'vector.post_refiner' tensor"
            )

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

        vec_pr = tensors["vector.post_refiner"]
        if hold_clean and ha_f > 0.0:
            synth_fam = "ref2va" if "ref2va" in str(loaded_family).lower() else "fl2va"
            hold_synth = _j_lens.synthesize_from_words(
                hold_clean,
                family=synth_fam,
            )
            orig_nrm = float(torch.linalg.vector_norm(vec_pr.to(dtype=torch.float64)).item())
            vec_held = _j_lens.apply_soft_holds(
                vec_pr,
                [hold_synth["vector.post_refiner"]],
                alpha=ha_f,
            )
            res_nrm = float(torch.linalg.vector_norm(vec_held.to(dtype=torch.float64)).item())
            if res_nrm <= 1e-3 * max(1.0, orig_nrm):
                raise ValueError("hold_words completely cancelled control vector (residual norm is zero)")
            vec_pr = vec_held.to(dtype=vec_pr.dtype)

        new_spec = _AppliedVectorSpec(
            vector_name=vec_path.name,
            vector=vec_pr,
            strength=float(strength),
            mode=mode,
            mask=mask,
            start_percent=sp_f,
            end_percent=ep_f,
            metadata=metadata,
            advisories=tuple(advisories),
            scale_mode=scale_mode,
            prompt=str(prompt or ""),
            video_scale=vs_f,
            audio_scale=as_f,
            split_end_percent=sep_f,
            ref_vae_end_percent=rvep_f,
            ref_vae_aug=rva_f,
        )
        updated = updated + (new_spec,)
        logger.info(
            "[h3] control vector %s attached (family=%s, strength=%g, mode=%s, scale_mode=%s, mask=%s, window=[%.2f, %.2f), video_scale=%g, audio_scale=%g, ref_vae_end=%.2f)",
            vec_path.name,
            loaded_family,
            float(strength),
            mode,
            scale_mode,
            mask,
            sp_f,
            ep_f,
            vs_f,
            as_f,
            rvep_f,
        )

    if words_clean:
        synth_fam = "ref2va" if "ref2va" in str(loaded_family).lower() else "fl2va"
        prompt_str = str(prompt or "").strip()
        is_prompt_against = against_clean.lower().lstrip("@") in ("prompt", "prompt_subject")
        synth = _j_lens.synthesize_from_words(
            words_clean,
            against_clean if against_clean else None,
            family=synth_fam,
            hold_words=hold_clean if hold_clean else None,
            alpha=ha_f,
            prompt=prompt_str if prompt_str else None,
        )
        w_label = f"words:{words_clean[:36]}"
        w_meta = {
            "name": w_label,
            "family": loaded_family,
            "measured_under": {"prefix": "none", "pooling": "diff_span"},
            "j_lens_readout": synth["readout"],
            "deferred_against_prompt": bool(is_prompt_against and not prompt_str),
            "words_raw": words_clean,
            "against_raw": against_clean,
            "hold_raw": hold_clean,
            "hold_alpha": ha_f,
            "synth_family": synth_fam,
        }
        w_spec = _AppliedVectorSpec(
            vector_name=w_label,
            vector=synth["vector.post_refiner"],
            strength=float(strength),
            mode=mode,
            mask=mask,
            start_percent=sp_f,
            end_percent=ep_f,
            metadata=w_meta,
            advisories=(),
            scale_mode=scale_mode,
            prompt=str(prompt or ""),
            video_scale=vs_f,
            audio_scale=as_f,
            split_end_percent=sep_f,
            ref_vae_end_percent=rvep_f,
            ref_vae_aug=rva_f,
        )
        updated = updated + (w_spec,)
        rd_summary = (synth.get("readiness") or {}).get("summary", "")
        logger.info(
            "[h3] inline word control vector %r synthesized on CPU (family=%s, strength=%g, mode=%s, PR(+) %s, readiness=%s)",
            words_clean,
            synth_fam,
            float(strength),
            mode,
            synth["readout"].get("pos_summary", ""),
            rd_summary,
        )

    for idx_sp, sp_chk in enumerate(updated):
        if _effective_entry_mode(sp_chk) == "hold" and float(sp_chk.strength) != 0.0:
            prior = updated[:idx_sp]
            has_matching_add = any(
                _effective_entry_mode(prev) in ("add", "add_norm_matched")
                and float(prev.strength) != 0.0
                and (prev.mask, prev.prompt) == (sp_chk.mask, sp_chk.prompt)
                for prev in prior
            )
            if not has_matching_add:
                prior_add_masks = sorted(
                    {
                        prev.mask
                        for prev in prior
                        if _effective_entry_mode(prev) in ("add", "add_norm_matched")
                        and float(prev.strength) != 0.0
                    }
                )
                if prior_add_masks:
                    raise ValueError(
                        f"mode='hold' on {sp_chk.vector_name!r} uses mask={sp_chk.mask!r}, which does not match "
                        f"any preceding additive control vector's mask ({prior_add_masks}); "
                        "a 'hold' node must share the same mask as the additive node(s) it projects out of"
                    )

    t_opts[WRAPPER_KEY] = updated
    # Replace the wrapper list under WRAPPER_KEY so all stacked specs execute
    # in one float64 accumulation pass per DiT step.
    wrappers = m.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
    wrappers[WRAPPER_KEY] = [_make_control_vector_wrapper(updated)]
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
                        "or 'lettering.toml')."
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
                        "Optional in-place 'pole_a -> pole_b' clause contrast(s) "
                        "(plus optional 'soundscape: ...' and 'hold: ...' clauses) "
                        "applied directly to 'target_prompt' with zero .toml file required."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["auto", *_pair_compiler.SUPPORTED_FRAME_MODES],
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
                        "Target H3 scene prompt when compiling from 'words'. Either pole_a or pole_b "
                        "in 'words' must match a substring in target_prompt for exact in-place compilation."
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
        mode="auto",
        target_prompt="",
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
            mode=str(mode or "auto"),
            target_prompt=str(target_prompt or ""),
        )
        return io.NodeOutput(out_path.name, json.dumps(report, indent=2, sort_keys=True))


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
    """Resize `image_a` and `image_b` to an identical 32-px-snapped `(target_w, target_h)`
    RGB tensor `[N, target_h, target_w, 3]`.

    In Qwen3-VL (`comfy/text_encoders/qwen_vl.py::qwen2vl_mrope_position_ids`),
    `<Picture 1>` vision tokens receive 3D `(t, h, w)` RoPE coordinates from
    `grid_thw` and set every trailing text token's M-RoPE start position to
    `start_next = (int(grid.max()) // 2) + start`. Matching `(target_w, target_h)`
    on the 32-px grid guarantees identical `grid_thw`, identical vision token
    count `N_vis`, and bitwise-identical M-RoPE `position_ids` between `A` and `B`."""
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
        va = ref_cond_mod._view_or_source(
            image_a[i : i + 1], w_a, h_a, target_w, target_h
        )
        vb = ref_cond_mod._view_or_source(
            image_b[i : i + 1], w_b, h_b, target_w, target_h
        )
        frames_a.append(va)
        frames_b.append(vb)
    out_a = torch.cat(frames_a, dim=0)
    out_b = torch.cat(frames_b, dim=0)
    return out_a, out_b, (target_w, target_h)


def _encode_media_frame_and_prompt(
    clip: Any,
    frame_1hwc: torch.Tensor,
    text: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int], Any]:
    """Encode a single `[1, H, W, 3]` image reference + `text` through `clip`."""
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
    spans_obj: _spans.H3Spans,
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
        m = torch.ones(( int(tags_1d.numel()), ), dtype=torch.bool, device="cpu")
        if not bool(m.any()):
            raise ValueError("media pooling='all_rows' found empty sequence")
        return m
    raise ValueError(
        f"media pooling must be one of {_cv.VALID_MEDIA_POOLING}, got {pooling!r}"
    )


def make_control_vector_from_media_pairs(
    *,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    model: Any,
    clip: Any,
    output_name: str = "media_contrast",
    pooling: str = "scene_text",
    short_edge: int = 512,
    scene_prompt: str = "",
    include_encoder: bool = True,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    made_by: str = "MiniMaxH3MakeMediaControlVector/1",
    loras: Sequence[dict[str, Any]] | None = None,
    encoder_path_override: str | Path | None = None,
    refiner_device: str | torch.device | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build and save a `.safetensors` control vector from contrasting `<Picture 1>`
    reference images (`image_a` baseline vs. `image_b` positive/steered, computing
    `B - A`) with matched `(target_w, target_h)` 32-px grid geometry so Qwen3-VL
    3D M-RoPE `position_ids` stay identical between `A` and `B`."""
    if pooling not in _cv.VALID_MEDIA_POOLING:
        raise ValueError(
            f"pooling must be one of {_cv.VALID_MEDIA_POOLING}, got {pooling!r}"
        )
    stem = str(output_name or "media_contrast").strip()
    if not _cv.NAME_RE.match(stem):
        raise ValueError(f"output_name must match [a-z0-9_]+, got {stem!r}")

    matched_a, matched_b, (target_w, target_h) = match_media_pair_geometry(
        image_a, image_b, short_edge=short_edge
    )
    n_frames = int(matched_a.shape[0])
    custom_prompt = str(scene_prompt or "").strip()
    if not custom_prompt:
        raise ValueError(
            "scene_prompt is required when building a media-contrast control vector; "
            "provide a scene prompt matching your target setup"
        )
    prompts = [custom_prompt for _ in range(n_frames)]
    frame_indices = list(range(n_frames))

    family = _cv.detect_family(model)
    if isinstance(model, (str, Path)):
        recipe = extract_model_recipe(
            None,
            clip,
            dit_path_override=model,
            encoder_path_override=encoder_path_override,
            loras_override=loras,
        )
        refiner_run = build_refiner_runner(
            model, device=refiner_device or "cpu", loras=loras
        )
    else:
        recipe = extract_model_recipe(
            model,
            clip,
            encoder_path_override=encoder_path_override,
            loras_override=loras,
        )
        refiner_run = build_refiner_runner(
            model, device=refiner_device, loras=loras
        )

    a_pooled_list: list[torch.Tensor] = []
    b_pooled_list: list[torch.Tensor] = []
    enc_a_pooled_list: list[torch.Tensor] = []
    enc_b_pooled_list: list[torch.Tensor] = []
    weights_list: list[float] = []
    enc_censuses: list[dict[str, float]] = []
    ref_censuses: list[dict[str, float]] = []
    pair_spans: list[dict[str, Any]] = []

    pair_hash = hashlib.sha256()
    pair_hash.update(_cv._tensor_bf16_bytes(matched_a))
    pair_hash.update(_cv._tensor_bf16_bytes(matched_b))
    for pr in prompts:
        pair_hash.update(pr.encode("utf-8"))

    # Pass 1: Encode all media frame pairs through `clip` while `clip` stays
    # resident on GPU so `clip` and `model` (`ModelPatcher`) do not swap in and
    # out of VRAM on every pair inside ComfyUI.
    encoded_media_pairs: list[tuple[int, int, torch.Tensor, torch.Tensor, list[int], Any, torch.Tensor, torch.Tensor, list[int], Any]] = []
    for idx, (f_idx, prompt_text) in enumerate(zip(frame_indices, prompts)):
        fa = matched_a[f_idx : f_idx + 1]
        fb = matched_b[f_idx : f_idx + 1]
        enc_a, tags_a, ids_a, dec_a = _encode_media_frame_and_prompt(
            clip, fa, prompt_text
        )
        enc_b, tags_b, ids_b, dec_b = _encode_media_frame_and_prompt(
            clip, fb, prompt_text
        )
        if tags_a.shape != tags_b.shape or not bool((tags_a == tags_b).all()):
            raise RuntimeError(
                f"media pair {idx} produced mismatched token_tags despite identical "
                f"{target_w}x{target_h} geometry: {tags_a.shape} vs {tags_b.shape}"
            )
        encoded_media_pairs.append(
            (idx, f_idx, enc_a, tags_a, ids_a, dec_a, enc_b, tags_b, ids_b, dec_b)
        )

    # Pass 2: Run `condition_proj + token_refiner` across all cached media encodings.
    for idx, f_idx, enc_a, tags_a, ids_a, dec_a, enc_b, tags_b, ids_b, dec_b in encoded_media_pairs:
        ref_a = refiner_run(enc_a)
        ref_b = refiner_run(enc_b)

        enc_censuses.append(_cv.row_norm_census(enc_a, tags_a))
        enc_censuses.append(_cv.row_norm_census(enc_b, tags_b))
        ref_censuses.append(_cv.row_norm_census(ref_a, tags_a))
        ref_censuses.append(_cv.row_norm_census(ref_b, tags_b))

        spans_a = _spans.spans(ids_a, tags_a.tolist(), dec_a)
        spans_b = _spans.spans(ids_b, tags_b.tolist(), dec_b)
        mask_a = _media_pooling_mask(spans_a, tags_a, pooling)
        mask_b = _media_pooling_mask(spans_b, tags_b, pooling)

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
        weights_list.append(1.0)
        n_vis = int((tags_a == 0).sum().item())
        pair_spans.append(
            {
                "index": idx,
                "weight": 1.0,
                "first_diff": 0,
                "end_a": int(enc_a.shape[0]),
                "end_b": int(enc_b.shape[0]),
                "a_span": f"<Picture 1: image_a[{f_idx}] {target_w}x{target_h} ({n_vis} vision tokens)>",
                "b_span": f"<Picture 1: image_b[{f_idx}] {target_w}x{target_h} ({n_vis} vision tokens)>",
                "len_a": int(enc_a.shape[0]),
                "len_b": int(enc_b.shape[0]),
                "pooled_rows_a": int(mask_a.sum().item()),
                "pooled_rows_b": int(mask_b.sum().item()),
                "vision_tokens": n_vis,
                "qwen_view_size": [target_w, target_h],
            }
        )

    a_mat = torch.stack(a_pooled_list, dim=0)
    b_mat = torch.stack(b_pooled_list, dim=0)
    w_vec = torch.tensor(weights_list, dtype=torch.float32)
    final_vec = _cv.mean_difference(a_mat, b_mat, w_vec)
    tensors_to_save: dict[str, torch.Tensor] = {"vector.post_refiner": final_vec}

    gen = torch.Generator(device="cpu").manual_seed(0)
    q_report = _cv.quality_report(a_mat, b_mat, w_vec, generator=gen)
    if _j_lens is not None and _j_lens.bundle_available():
        try:
            q_report["j_lens_readout"] = _j_lens.decode_vector(
                final_vec, space="post_refiner", family=family
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

    prefix_meta = {"media_contrast_grid": f"{target_w}x{target_h}"}
    metadata: dict[str, Any] = {
        "format": _cv.FORMAT_VERSION,
        "name": stem,
        "quality": f"media-contrast ({target_w}x{target_h}, pooling={pooling})",
        "family": family,
        "pairs": {
            "path": f"<media_pairs:{stem}>",
            "sha256": pair_hash.hexdigest(),
            "n_pairs": len(prompts),
            "n_holds": 0,
            "a_side": f"image_a ({target_w}x{target_h})",
            "b_side": f"image_b ({target_w}x{target_h})",
            "pair_spans": pair_spans,
        },
        "recipe": recipe,
        "measured_under": {
            "mode": "ref2va",
            "frame": "scene",
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
        prefix="none",
        apply_holds=True,
        n_holds=0,
        loras=recipe.get("loras"),
    )
    out_path = _cv.save_vector(
        target_dir / out_filename, tensors_to_save, metadata, overwrite=overwrite
    )
    full_report = {
        "vector_file": out_filename,
        "path": str(out_path),
        "metadata": metadata,
        "qwen_view_size": [target_w, target_h],
        "pair_spans": pair_spans,
    }
    return out_path, full_report


class MiniMaxH3MakeMediaControlVector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MakeMediaControlVector",
            display_name="MiniMax H3 Make Media Control Vector",
            category="model/conditioning/minimax",
            description=(
                "Extracts a post-refiner control vector (B - A) from contrasting reference "
                "images (image_a baseline vs. image_b positive/steered) with zero .toml "
                "file required. Resizes both sides to an identical 32-px-snapped "
                "(target_w, target_h) grid so Qwen3-VL 3D M-RoPE position_ids and "
                "vision token counts match exactly."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Image.Input(
                    "image_a",
                    tooltip="Baseline/negative pole reference image(s) [N, H, W, C] (Pole A in B - A).",
                ),
                io.Image.Input(
                    "image_b",
                    tooltip="Positive/steered pole reference image(s) [N, H, W, C] carrying the target look or attribute (Pole B in B - A).",
                ),
                io.String.Input(
                    "output_name",
                    default="media_contrast",
                    optional=True,
                    tooltip="Output vector stem (<output_name>__<family>.safetensors under models/control_vectors/).",
                ),
                io.Combo.Input(
                    "pooling",
                    options=list(_cv.VALID_MEDIA_POOLING),
                    default="scene_text",
                    optional=True,
                    tooltip=(
                        "Which post-refiner rows to mean-pool per pair side: 'scene_text' "
                        "(default, steerable visual scene prose on sectioned H3 prompts), "
                        "'look_only' (summary: + subject_definitions: + visual retention_analysis:), "
                        "'description_only' (detailed_description: / integrated_multimodal_description:), "
                        "'soundscape_only' (overall_soundscape: + non-'N/A' non_diegetic_music:), "
                        "'all_tag1', 'vision_tokens' (tag-0 <Picture 1> rows), or 'all_rows'."
                    ),
                ),
                io.Int.Input(
                    "short_edge",
                    default=512,
                    min=32,
                    max=2048,
                    step=32,
                    optional=True,
                    tooltip=(
                        "Target shorter edge in pixels (snapped to 32) applied to both "
                        "image_a and image_b so Qwen3-VL 3D M-RoPE grid_thw matches."
                    ),
                ),
                io.String.Input(
                    "scene_prompt",
                    default="",
                    optional=True,
                    tooltip=(
                        "Required H3 ref2va scene prompt used as the text context when "
                        "encoding image_a and image_b."
                    ),
                ),
                io.Boolean.Input(
                    "overwrite",
                    default=False,
                    optional=True,
                    tooltip="When False (default), refuses to overwrite an existing vector file.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="vector_name"),
                io.String.Output(display_name="report"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        image_a,
        image_b,
        output_name="media_contrast",
        pooling="scene_text",
        short_edge=512,
        scene_prompt="",
        overwrite=False,
    ) -> io.NodeOutput:
        dm = model.get_model_object("diffusion_model")
        if not _is_minimax_h3(dm):
            raise ValueError("MiniMaxH3MakeMediaControlVector requires a MiniMax H3 diffusion model")
        out_path, report = make_control_vector_from_media_pairs(
            image_a=image_a,
            image_b=image_b,
            model=model,
            clip=clip,
            output_name=str(output_name or "media_contrast"),
            pooling=str(pooling or "scene_text"),
            short_edge=int(short_edge),
            scene_prompt=str(scene_prompt or ""),
            overwrite=bool(overwrite),
            made_by="MiniMaxH3MakeMediaControlVector/1",
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
                    options=list(MASK_MODES),
                    default=_APPLY_DEFAULTS["mask"],
                    tooltip=(
                        "'scene_only' (default) steers all visual prose (look_only | description_only: "
                        "subject_definitions:, summary:, visual retention_analysis:, and "
                        "integrated_multimodal_description: / detailed_description:) while leaving "
                        "tag-0 vision rows, <d>...</d> dialogue, (S1) speaker tags, "
                        "<Subject N>/<Picture N>/<Video N>/<Audio N> tags, [Shot N], section headers, "
                        "overall_soundscape:, non_diegetic_music:, Part One keyframe alignment lines, "
                        "and <0.0 seconds> timestamps untouched; "
                        "'look_only' steers only subject_definitions:, summary:, and visual "
                        "retention_analysis: (falling back to description_only on 3-field t2va/fl2va prompts); "
                        "'description_only' steers only events/actions prose inside "
                        "integrated_multimodal_description: (fl2va/t2va) or detailed_description: (ref2va); "
                        "'soundscape_only' steers overall_soundscape:, non-'N/A' non_diegetic_music:, and "
                        "<Audio N>: lines in retention_analysis:; "
                        "'non_vision' steers all tag-1 text rows and leaves tag-0 vision rows untouched; "
                        "'after_dialogue' steers only scene tokens after the last </d>; "
                        "'all_rows' is the unmasked ablation (E2.3). "
                        "A field's own name steers the prose of that one field and nothing else: "
                        "'subject_definitions', 'summary', 'retention_analysis', 'overall_soundscape', "
                        "'non_diegetic_music' (the description field is 'description_only'). A field the "
                        "prompt does not have, or one that reads N/A, steers no rows and the log says so. "
                        "To steer several fields at once, tick them in 'fields' below; that takes the place of "
                        "this choice."
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
                io.Float.Input(
                    "video_scale",
                    default=_APPLY_DEFAULTS["video_scale"],
                    min=-5.0,
                    max=5.0,
                    step=0.05,
                    optional=True,
                    tooltip=(
                        "Stream multiplier applied to the video output velocity within "
                        "[0.0, split_end_percent). At (1.0, 1.0) runs a single DiT pass; "
                        "at (1.0, 0.0) steers video while keeping audio unsteered."
                    ),
                ),
                io.Float.Input(
                    "audio_scale",
                    default=_APPLY_DEFAULTS["audio_scale"],
                    min=-5.0,
                    max=5.0,
                    step=0.05,
                    optional=True,
                    tooltip=(
                        "Stream multiplier applied to the audio output velocity within "
                        "[0.0, split_end_percent). At (0.0, 1.0) steers audio while "
                        "keeping video unsteered."
                    ),
                ),
                io.Float.Input(
                    "split_end_percent",
                    default=_APPLY_DEFAULTS["split_end_percent"],
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "End of the step window [0.0, split_end_percent) where decoupled "
                        "video_scale != audio_scale runs."
                    ),
                ),
                io.Float.Input(
                    "ref_vae_end_percent",
                    default=_APPLY_DEFAULTS["ref_vae_end_percent"],
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "For ref2va / keyframe / guide-latent renders: fraction of steps [0.0, ref_vae_end_percent) "
                        "where clean 3D Video VAE conditioning rows ('ref_img' / 'cond') stay packed in the DiT "
                        "sequence (default 1.00 keeps them active on every step; lower values drop visual cond rows "
                        "on later steps while preserving 3D RoPE positions and any speaker/audio references)."
                    ),
                ),
                io.Float.Input(
                    "ref_vae_aug",
                    default=_APPLY_DEFAULTS["ref_vae_aug"],
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    optional=True,
                    tooltip=(
                        "For renders with a reference, keyframe or guide picture: the picture's share of its "
                        "condition rows, the rest noise (core's visual_cond_noise_aug; the default is core's, an "
                        "almost clean picture). Lower gives the model a noisier copy of the picture on the steps "
                        "where it is held. The rows are never noisier than the frame being made: once the frame is "
                        "cleaner than this share, the picture follows it."
                    ),
                ),
                io.String.Input(
                    "words",
                    default=_APPLY_DEFAULTS["words"],
                    optional=True,
                    tooltip=(
                        "Optional inline positive words (e.g. 'blizzard, snowdrift, freezing, snow' "
                        "or '8-bit, LED, steampunk') synthesized in <2 ms on CPU via J-Lens with "
                        "zero .safetensors file required. Auto-matches loaded model family (fl2va/ref2va)."
                    ),
                ),
                io.String.Input(
                    "against_words",
                    default=_APPLY_DEFAULTS["against_words"],
                    optional=True,
                    tooltip=(
                        "Optional negative words when 'words' is used. Leave empty for neutral "
                        "baseline, pass '@prompt_subject' (or '@prompt') to auto-extract the scene "
                        "prompt's medium & subject on Step 0, or pass '@replace_human'."
                    ),
                ),
                io.String.Input(
                    "hold_words",
                    default=_APPLY_DEFAULTS["hold_words"],
                    optional=True,
                    tooltip=(
                        "Optional nuisance words to project out of 'vector_name' or inline 'words' on CPU (<2 ms)."
                    ),
                ),
                io.Float.Input(
                    "hold_alpha",
                    default=_APPLY_DEFAULTS["hold_alpha"],
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    optional=True,
                    tooltip=(
                        "Attenuation strength alpha in [0.0, 2.0] for 'hold_words' (default 1.0)."
                    ),
                ),
                io.MultiCombo.Input(
                    "fields",
                    options=list(_spans.PROMPT_FIELDS),
                    default=[],
                    optional=True,
                    tooltip=(
                        "Tick one or more fields of the prompt to steer exactly those and nothing else: "
                        "'subject_definitions', 'summary', 'retention_analysis', 'description' "
                        "(detailed_description: or integrated_multimodal_description:, whichever the prompt has), "
                        "'overall_soundscape', 'non_diegetic_music'. With any field ticked, 'mask' above is not "
                        "used. With none ticked, 'mask' decides. A ticked field the prompt does not have, or one "
                        "that reads N/A, adds no rows."
                    ),
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        vector_name=_APPLY_DEFAULTS["vector_name"],
        words=_APPLY_DEFAULTS["words"],
        against_words=_APPLY_DEFAULTS["against_words"],
        hold_words=_APPLY_DEFAULTS["hold_words"],
        hold_alpha=_APPLY_DEFAULTS["hold_alpha"],
        **kwargs,
    ):
        raw = str(vector_name).strip()
        w_str = str(words or "").strip()
        h_str = str(hold_words or "").strip()
        vec_fp: Any = "none"
        if raw and raw != "none":
            resolved_stats: list[tuple[str, int, int]] = []
            for pref_fam in (None, "fl2va", "ref2va"):
                try:
                    full = resolve_vector_file_path(
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
        vector_name=_APPLY_DEFAULTS["vector_name"],
        strength=_APPLY_DEFAULTS["strength"],
        mode=_APPLY_DEFAULTS["mode"],
        mask=_APPLY_DEFAULTS["mask"],
        start_percent=_APPLY_DEFAULTS["start_percent"],
        end_percent=_APPLY_DEFAULTS["end_percent"],
        allow_recipe_mismatch=_APPLY_DEFAULTS["allow_recipe_mismatch"],
        video_scale=_APPLY_DEFAULTS["video_scale"],
        audio_scale=_APPLY_DEFAULTS["audio_scale"],
        split_end_percent=_APPLY_DEFAULTS["split_end_percent"],
        ref_vae_end_percent=_APPLY_DEFAULTS["ref_vae_end_percent"],
        ref_vae_aug=_APPLY_DEFAULTS["ref_vae_aug"],
        words=_APPLY_DEFAULTS["words"],
        against_words=_APPLY_DEFAULTS["against_words"],
        hold_words=_APPLY_DEFAULTS["hold_words"],
        hold_alpha=_APPLY_DEFAULTS["hold_alpha"],
        **kwargs,
    ) -> bool | str:
        raw = str(vector_name).strip()
        if raw and raw != "none":
            try:
                resolve_vector_file_path(raw)
            except Exception as exc:
                return str(exc)
        picked = kwargs.get("fields")
        if picked:
            try:
                _spans.fields_mask(picked if isinstance(picked, (list, tuple)) else [picked])
            except ValueError as exc:
                return str(exc)
        w_str = str(words or "").strip()
        h_str = str(hold_words or "").strip()
        if h_str and (not raw or raw == "none") and not w_str:
            return (
                "hold_words requires vector_name != 'none' or non-empty words; "
                "cannot apply hold_words when no control vector or inline words are selected"
            )
        if w_str or h_str:
            try:
                ha_val = float(hold_alpha)
                if not math.isfinite(ha_val) or not (0.0 <= ha_val <= 2.0):
                    return f"hold_alpha must be in [0.0, 2.0], got {hold_alpha!r}"
                if _j_lens.bundle_available():
                    if w_str:
                        _j_lens.synthesize_from_words(
                            w_str,
                            str(against_words or "").strip() or None,
                            family="fl2va",
                            hold_words=h_str or None,
                            alpha=ha_val,
                        )
                    elif h_str:
                        _j_lens.synthesize_from_words(
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
        vector_name=_APPLY_DEFAULTS["vector_name"],
        strength=_APPLY_DEFAULTS["strength"],
        mode=_APPLY_DEFAULTS["mode"],
        mask=_APPLY_DEFAULTS["mask"],
        start_percent=_APPLY_DEFAULTS["start_percent"],
        end_percent=_APPLY_DEFAULTS["end_percent"],
        allow_recipe_mismatch=_APPLY_DEFAULTS["allow_recipe_mismatch"],
        video_scale=_APPLY_DEFAULTS["video_scale"],
        audio_scale=_APPLY_DEFAULTS["audio_scale"],
        split_end_percent=_APPLY_DEFAULTS["split_end_percent"],
        ref_vae_end_percent=_APPLY_DEFAULTS["ref_vae_end_percent"],
        ref_vae_aug=_APPLY_DEFAULTS["ref_vae_aug"],
        words=_APPLY_DEFAULTS["words"],
        against_words=_APPLY_DEFAULTS["against_words"],
        hold_words=_APPLY_DEFAULTS["hold_words"],
        hold_alpha=_APPLY_DEFAULTS["hold_alpha"],
        fields=None,
    ) -> io.NodeOutput:
        # Ticked fields take the place of `mask`: the selection is itself a mask name (`a+b`).
        if fields:
            mask = _spans.fields_mask(fields if isinstance(fields, (list, tuple)) else [fields])
        m = attach_control_vector(
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
