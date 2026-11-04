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
from typing import Any, Sequence

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
_rules = _repo("h3_rules")
_substrate = _repo("substrate")

REPO_ROOT = Path(__file__).resolve().parent
COMFY_ROOT = REPO_ROOT.parent.parent
PAIR_BANK_DIR = REPO_ROOT / "prompt_bank" / "control_vectors"
PROMPT_BANK_DIR = REPO_ROOT / "prompt_bank"

CONTROL_VECTORS_FOLDER = "control_vectors"
WRAPPER_KEY = "h3_control_vectors"
MASK_MODES = ("non_vision", "all_rows")
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
        f for f in folder_paths.get_filename_list(CONTROL_VECTORS_FOLDER) if f.endswith(".safetensors")
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
            if isinstance(obj_patches, dict):
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


def build_refiner_runner(
    source: str | Path | Any,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    """Return a callable `run_refiner(encoder_rows_2d: Tensor[L, 5120]) -> Tensor[L, 5376]`
    that runs `condition_proj` + `token_refiner` at `dtype` (default `bfloat16`,
    matching `MiniMaxH3.extra_conds`).

    Accepts either a `.safetensors` checkpoint path (loading only `condition_proj.*`
    and `token_refiner.*` in < 0.2s without loading the 50 DiT blocks) or a loaded
    ComfyUI `ModelPatcher` / `MiniMaxH3Model`."""
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
    )
    if ckpt_path is not None and ckpt_path.is_file() and not has_loras:
        return build_refiner_runner(ckpt_path, device=device, dtype=dtype)

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
        return cond_list, cond_tensor, tags, token_ids, decode_fn

    tokens = clip.tokenize(clean_text)
    cond_list = clip.encode_from_tokens_scheduled(tokens)
    cond_tensor = cond_list[0][0][0].detach().to(device="cpu", dtype=torch.float32)
    tags = torch.as_tensor(
        cond_list[0][1]["minimax_token_tags"], dtype=torch.long, device="cpu"
    ).flatten()
    entries = _extract_token_entries(tokens)
    token_ids = _spans.expand_token_ids(entries, tags)
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
    pair_path: str | Path,
    model: Any,
    clip: Any,
    pooling: str = "scene_text",
    apply_holds: bool = True,
    output_name: str = "",
    output_dir: str | Path | None = None,
    encoded_references: Any | None = None,
    made_by: str = "MiniMaxH3MakeControlVector/1",
) -> tuple[Path, dict[str, Any]]:
    """Build and save a `.safetensors` control vector from `pair_path`, returning
    `(saved_path, full_report_dict)`."""
    if pooling not in _cv.VALID_POOLING:
        raise ValueError(f"pooling must be one of {_cv.VALID_POOLING}, got {pooling!r}")

    resolved_pair_path = resolve_pair_file_path(pair_path)
    pair_set = _cv.load_pairs(resolved_pair_path, bank_dir=PROMPT_BANK_DIR)
    family = _cv.detect_family(model)
    recipe = extract_model_recipe(model, clip)
    refiner_run = build_refiner_runner(model)

    a_pooled_list: list[torch.Tensor] = []
    b_pooled_list: list[torch.Tensor] = []
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

        first_diff = _spans.first_differing_index(ids_a, ids_b)
        spans_a = _spans.spans(ids_a, tags_a.tolist(), dec_a)
        spans_b = _spans.spans(ids_b, tags_b.tolist(), dec_b)
        mask_a = _spans.pooling_mask(
            spans_a, pooling, first_diff=min(first_diff, len(ids_a) - 1)
        )
        mask_b = _spans.pooling_mask(
            spans_b, pooling, first_diff=min(first_diff, len(ids_b) - 1)
        )

        pool_a = ref_a.to(dtype=torch.float64)[mask_a].mean(dim=0).to(dtype=torch.float32)
        pool_b = ref_b.to(dtype=torch.float64)[mask_b].mean(dim=0).to(dtype=torch.float32)
        a_pooled_list.append(pool_a)
        b_pooled_list.append(pool_b)
        weights_list.append(float(item.weight))
        pair_spans.append(
            {
                "index": idx,
                "weight": float(item.weight),
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

    raw_vec = _cv.mean_difference(a_mat, b_mat, w_vec)
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
            h_diff = _spans.first_differing_index(h_ids_a, h_ids_b)
            h_spans_a = _spans.spans(h_ids_a, h_tags_a.tolist(), h_dec_a)
            h_spans_b = _spans.spans(h_ids_b, h_tags_b.tolist(), h_dec_b)
            h_mask_a = _spans.pooling_mask(
                h_spans_a, pooling, first_diff=min(h_diff, len(h_ids_a) - 1)
            )
            h_mask_b = _spans.pooling_mask(
                h_spans_b, pooling, first_diff=min(h_diff, len(h_ids_b) - 1)
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

    try:
        rel_pair_path = str(resolved_pair_path.relative_to(REPO_ROOT))
    except ValueError:
        rel_pair_path = resolved_pair_path.name

    stem = (output_name or pair_set.name).strip()
    if not _cv.NAME_RE.match(stem):
        raise ValueError(f"output_name must match [a-z0-9_]+, got {stem!r}")

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
        },
        "recipe": recipe,
        "measured_under": {
            "mode": "ref2va" if encoded_references is not None else pair_set.mode,
            "frame": pair_set.frame,
            "prefix": prefix_meta,
            "pooling": pooling,
        },
        "points": {
            "post_refiner": {
                "unit": "mean_pair_difference",
                "raw_norm": raw_norm,
                "median_row_norm": median_ref_row_norm,
            }
        },
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
    family = family.replace(":", "_")
    out_filename = f"{stem}__{family}.safetensors"
    out_path = _cv.save_vector(target_dir / out_filename, tensors_to_save, metadata)

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
    mask: str
    metadata: dict[str, Any]
    advisories: tuple[str, ...]


def _make_control_vector_wrapper(specs: tuple[_AppliedVectorSpec, ...]):
    logged_prefix_notes: set[tuple[str, str]] = set()

    def wrapper(executor, x, timestep, context, transformer_options={}, **kwargs):
        if context is None or not specs or all(sp.strength == 0.0 for sp in specs):
            return executor(x, timestep, context, transformer_options, **kwargs)
        payload = kwargs.get("minimax_payload") or {}
        tags_raw = payload.get("text_token_tags")
        B, L, _D = context.shape
        if tags_raw is not None:
            tags = torch.as_tensor(tags_raw, device=context.device, dtype=torch.long)
            if tags.ndim == 1:
                tags = tags.unsqueeze(0).expand(B, -1)
        else:
            if any(sp.mask != "all_rows" for sp in specs):
                raise ValueError(
                    "minimax_payload is missing 'text_token_tags'; cannot apply "
                    "non_vision mask without token tags"
                )
            tags = torch.ones((B, L), device=context.device, dtype=torch.long)

        has_vision = bool((tags == 0).any().item())
        render_prefix_kind = "references_or_keyframes" if has_vision else "none"
        for sp in specs:
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

        # Group consecutive specs with the same mask mode so stacked vectors
        # sum in float64 before a single cast to context.dtype (`CONTRACT.md` §3 / paper §2.1).
        cur = context
        i = 0
        n = len(specs)
        while i < n:
            sp = specs[i]
            mask_vision = sp.mask == "non_vision"
            batch_entries: list[tuple[torch.Tensor, float]] = [(sp.vector, sp.strength)]
            j = i + 1
            while j < n and (specs[j].mask == "non_vision") == mask_vision:
                batch_entries.append((specs[j].vector, specs[j].strength))
                j += 1
            cur = _cv.apply_post_refiner(
                cur,
                tags,
                batch_entries,
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
    mask: str = "non_vision",
    allow_recipe_mismatch: bool = False,
    loaded_prefix: str | dict[str, Any] | None = None,
) -> Any:
    """Clone `model` with `vector_name` attached on `WrappersMP.DIFFUSION_MODEL`.

    Validates the model type, vector format, family, and recipe even when
    `strength == 0.0`."""
    dm = model.get_model_object("diffusion_model")
    if not _is_minimax_h3(dm):
        raise ValueError("MiniMaxH3ApplyControlVectors requires a MiniMax H3 diffusion model")
    if mask not in MASK_MODES:
        raise ValueError(f"mask must be one of {MASK_MODES}, got {mask!r}")
    if not math.isfinite(float(strength)):
        raise ValueError(f"strength must be finite, got {strength!r}")

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
        mask=mask,
        metadata=metadata,
        advisories=tuple(advisories),
    )
    updated = existing + (new_spec,)
    t_opts[WRAPPER_KEY] = updated

    # Replace the wrapper list under WRAPPER_KEY so all stacked specs execute
    # in one float64 accumulation pass per DiT step.
    wrappers = m.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
    wrappers[WRAPPER_KEY] = [_make_control_vector_wrapper(updated)]
    logger.info(
        "[h3] control vector %s attached (family=%s, strength=%g, mask=%s)",
        vec_path.name,
        loaded_family,
        float(strength),
        mask,
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
                "scene-text rows in float64, projecting out any [[hold]] directions, and "
                "saving <name>__<family>.safetensors under models/control_vectors/."
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
                    default="scene_text",
                    tooltip=(
                        "Which tag-1 rows after the token refiner are mean-pooled per pair side: "
                        "'scene_text' (default, excludes reference labels and timestamps), "
                        "'all_tag1', or 'from_first_diff'."
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
            ],
            outputs=[
                io.String.Output(display_name="vector_name"),
                io.String.Output(display_name="report"),
            ],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, pair_file="film_16mm_1974.toml", **kwargs):
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
        pooling="scene_text",
        apply_holds=True,
        output_name="",
        encoded_references=None,
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
            made_by="MiniMaxH3MakeControlVector/1",
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
                        "mean pair difference; negative subtracts; 0.0 is an exact no-op)."
                    ),
                ),
                io.Combo.Input(
                    "mask",
                    options=list(MASK_MODES),
                    default=_APPLY_DEFAULTS["mask"],
                    tooltip=(
                        "'non_vision' (default) applies only to tag-1 text rows and leaves "
                        "tag-0 vision rows bitwise untouched; 'all_rows' is the unmasked "
                        "ablation (E2.3 / MAINodes-style)."
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
        mask=_APPLY_DEFAULTS["mask"],
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
        mask=_APPLY_DEFAULTS["mask"],
        allow_recipe_mismatch=_APPLY_DEFAULTS["allow_recipe_mismatch"],
    ) -> io.NodeOutput:
        m = attach_control_vector(
            model,
            vector_name,
            strength=float(strength),
            mask=mask,
            allow_recipe_mismatch=bool(allow_recipe_mismatch),
        )
        return io.NodeOutput(m)
