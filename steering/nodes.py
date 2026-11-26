"""Primary ComfyUI steering nodes (`MiniMaxH3Steer`, `MiniMaxH3ReleaseReferences`,
`MiniMaxH3WordVector`, `MiniMaxH3MakeMediaControlVector`) and runtime model wrapper."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
import logging
import math
import os
import sys
import types
import weakref
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import folder_paths
import torch
from comfy.patcher_extension import CallbacksMP, WrappersMP
from comfy_api.latest import io

from . import hooks as _steer_hooks
from . import xray as _j_lens
from .constants import import_repo_module as _repo
from .encode import (
    _attach_refiner_lora_branches,
    _encode_media_frame_and_prompt,
    _extract_token_entries,
    _media_pooling_mask,
    _patcher_checkpoint_path,
    _validate_short_edge,
    build_refiner_runner,
    encode_conditioning_for_vector,
    encode_pair_set,
    encode_prompt_for_vector,
    extract_model_recipe,
    match_media_pair_geometry,
    resolve_lora_file_path,
)

logger = logging.getLogger("control_vector_nodes")

REPO_ROOT = Path(__file__).resolve().parent.parent
COMFY_ROOT = REPO_ROOT.parent.parent
PAIR_BANK_DIR = REPO_ROOT / "prompt_bank" / "control_vectors"
USER_PAIR_BANK_DIR = REPO_ROOT / "data" / "control_vectors"
PROMPT_BANK_DIR = REPO_ROOT / "prompt_bank"

_cv = _repo("control_vectors")
_spans = _repo("h3_spans")
_pair_compiler = _repo("h3_pair_compiler")
_rules = _repo("h3_rules")
_substrate = _repo("substrate")

CONTROL_VECTORS_FOLDER = "control_vectors"
WRAPPER_KEY = "h3_control_vectors"
MASK_MODES = _spans.INFERENCE_MASK_MODES
STEER_MODES: tuple[str, ...] = ("add_norm_matched_erasure", "add_norm_matched", "add", "project_out")
STEER_TARGETS: tuple[str, ...] = (
    "scene_only",
    "description_only",
    "look_only",
    "soundscape_only",
    "after_dialogue",
    "non_vision",
    "vision_tokens",
)
STEER_STREAMS: tuple[str, ...] = ("both", "video_only", "audio_only")
STEER_EDIT_PASSES: tuple[str, ...] = ("both", "edited_only", "original_only")
H3ControlVector = io.Custom("MINIMAX_H3_CONTROL_VECTOR")
H3EditPlan = io.Custom("MINIMAX_H3_EDIT_PLAN")
_APPLY_DEFAULTS = {
    **_rules.CONTROL_VECTOR_DEFAULTS,
    **_rules.CONTROL_VECTOR_OPTIONAL_DEFAULTS,
}


def _self_mod():
    return sys.modules[__name__]


@dataclasses.dataclass(frozen=True)
class H3VectorSpec:
    """Lazy, family-agnostic vector specification carried over `MINIMAX_H3_CONTROL_VECTOR` wires."""

    kind: str  # "file" or "words"
    vector_name: str = "none"
    words: str = ""
    against_words: str = ""
    hold_words: str = ""
    hold_alpha: float = 1.0
    holds: tuple[tuple[H3VectorSpec, float], ...] = ()


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


def _is_hidden_vector_relpath(rel_path: str | Path) -> bool:
    parts = Path(str(rel_path).replace("\\", "/")).parts
    return any(p == "j_lens" or p.startswith(("_", ".")) for p in parts[:-1])


def _vector_combo_options() -> list[str]:
    files = sorted(
        f
        for f in folder_paths.get_filename_list(CONTROL_VECTORS_FOLDER)
        if f.endswith(".safetensors") and not _is_hidden_vector_relpath(f)
    )
    return ["none", *files]


def _extract_vector_stem_and_family(filename: str) -> tuple[str | None, str | None]:
    norm = filename.replace("\\", "/")
    base = norm.rsplit("/", 1)[-1]
    for fam in ("fl2va", "ref2va", "fasth3", "h3_image"):
        tag = f"__{fam}"
        if tag in base:
            cand = base.split(tag, 1)[0]
            if cand and "__" not in cand:
                return cand, fam
    return None, None


def _vector_stem_options(*, include_none: bool = True) -> list[str]:
    """Return deduplicated bare vector stems for `MiniMaxH3Steer` / `MiniMaxH3LoadControlVector`."""
    raw_files = [f for f in _vector_combo_options() if f != "none"]
    by_stem_fam: dict[tuple[str, str], list[str]] = {}
    entries: set[str] = set()
    for fname in raw_files:
        stem, fam = _extract_vector_stem_and_family(fname)
        if stem is not None and fam is not None:
            entries.add(stem)
            by_stem_fam.setdefault((stem, fam), []).append(fname)
        else:
            entries.add(fname)
    for (_stem, _fam), fnames in by_stem_fam.items():
        if len(fnames) > 1:
            entries.update(fnames)
    ordered = sorted(entries)
    return ["none", *ordered] if include_none else (ordered or ["none"])


def _is_minimax_h3(diffusion_model: Any) -> bool:
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
    except ImportError:
        return False
    return isinstance(diffusion_model, MiniMaxH3Model)


def _is_minimax_h3_image(diffusion_model: Any) -> bool:
    try:
        from comfy.ldm.minimax.model import MiniMaxH3ImageModel
    except ImportError:
        return False
    return isinstance(diffusion_model, MiniMaxH3ImageModel)


def resolve_pair_file_path(pair_file: str | Path) -> Path:
    raw = str(pair_file).strip()
    if not raw:
        raise ValueError("pair_file must be a non-empty string")
    cand = Path(raw)
    if cand.is_file():
        return cand.resolve()
    raw_toml = raw if raw.endswith(".toml") else f"{raw}.toml"
    for base in (USER_PAIR_BANK_DIR, PAIR_BANK_DIR, REPO_ROOT):
        p = base / raw_toml
        if p.is_file():
            return p.resolve()
    raise ValueError(
        f"pair file {pair_file!r} not found under {USER_PAIR_BANK_DIR} or {PAIR_BANK_DIR}"
    )


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
    audio_cond_end_percent: float = 1.0
    audio_cond_aug: float = 1.0
    video_ref_end_percent: float = 1.0
    guide_cond_end_percent: float = 1.0
    edit_pass: str = "both"
    token_focus: float = 1.0
    erasure_vr: torch.Tensor | None = None
    erasure_diag: torch.Tensor | None = None
    erasure_mu_d: torch.Tensor | None = None
    erasure_holds: tuple[tuple[torch.Tensor, float], ...] = ()
    erasure_mu_a: torch.Tensor | None = None
    block_band: str = "all"
    split_backend: str = "auto_1x_kv"


def _step_in_window(
    transformer_options: dict[str, Any] | None,
    timestep: Any,
    start_percent: float,
    end_percent: float,
) -> bool:
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
            if cur is timestep and cur_val > 1.0 and float(sched_t.max().item()) <= 1.0001:
                cur_val /= 1000.0
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
        if cur is timestep and 1.0 < cur_val <= 1000.0:
            cur_val /= 1000.0
        if 0.0 <= cur_val <= 1.0:
            frac = 1.0 - cur_val
            return bool(start_percent <= frac < end_percent)
    return True


def _stream_time(timestep: Any) -> float | None:
    if timestep is None:
        return None
    if isinstance(timestep, torch.Tensor):
        raw = float(timestep.detach().flatten()[0].item())
    elif isinstance(timestep, (list, tuple)):
        raw = float(timestep[0])
    else:
        raw = float(timestep)
    return 1.0 - max(raw / 1000.0, 1e-6)


def _audio_stream_time(
    timestep: Any,
    transformer_options: Mapping[str, Any] | None = None,
    *,
    default_shift_video: float = 12.0,
    default_shift_audio: float = 3.0,
) -> float | None:
    if timestep is None:
        return None
    if isinstance(timestep, torch.Tensor):
        raw = float(timestep.detach().flatten()[0].item())
    elif isinstance(timestep, (list, tuple)):
        raw = float(timestep[0])
    else:
        raw = float(timestep)
    sigma_v = max(raw / 1000.0, 1e-6)
    t_opts = transformer_options or {}
    shift_v = float(t_opts.get("minimax_h3_sigma_shift_video", default_shift_video))
    shift_a = float(t_opts.get("minimax_h3_sigma_shift_audio", default_shift_audio))
    denom_from = shift_v + sigma_v * (1.0 - shift_v)
    base = sigma_v / denom_from if denom_from != 0.0 else sigma_v
    denom_to = 1.0 + (shift_a - 1.0) * base
    sigma_a = (shift_a * base / denom_to) if denom_to != 0.0 else base
    return 1.0 - sigma_a


def _resolve_custom_inference_mask(
    sp: _AppliedVectorSpec,
    *,
    tags_raw: Any,
    tags_2d: torch.Tensor,
    payload: Mapping[str, Any],
    L: int,
    device: torch.device,
    kwarg_span_masks: Any = None,
    is_h3_image: bool = False,
) -> torch.Tensor | None:
    if sp.mask in ("non_vision", "all_rows"):
        return None
    if sp.mask == "vision_tokens":
        return (tags_2d[0] == 0).to(device=device)
    span_masks = kwarg_span_masks
    if span_masks is None:
        span_masks = payload.get("h3_span_masks")
    if span_masks is None and tags_raw is not None:
        span_masks = getattr(tags_raw, "_h3_span_masks", None)
    if span_masks is None and sp.prompt and sp.prompt.strip():
        span_masks = _spans.span_masks_from_prompt(sp.prompt, L, tags=tags_2d[0])
    if span_masks is None and is_h3_image:
        row_tags = tags_2d[0]
        vis_idxs = (row_tags == 0).nonzero(as_tuple=False).flatten()
        if vis_idxs.numel() > 0:
            last_vis = int(vis_idxs[-1].item())
            caption_rows = torch.arange(L, device=device) > last_vis
        else:
            caption_rows = (row_tags != 0).to(device=device)
        sel = _spans.split_fields_mask(sp.mask)
        if sel is not None:
            if any(f in ("description", "summary", "subject_definitions") for f in sel):
                return caption_rows
            return torch.zeros((L,), dtype=torch.bool, device=device)
        if sp.mask in ("scene_only", "description_only", "look_only", "after_dialogue"):
            return caption_rows
        if sp.mask == "soundscape_only":
            return torch.zeros((L,), dtype=torch.bool, device=device)
    if span_masks is None:
        raise ValueError(
            f"control vector {sp.vector_name!r} uses mask={sp.mask!r}, which requires "
            "conditioning from an H3 conditioning node (h3_span_masks / _h3_span_masks) or prompt=..."
        )
    if isinstance(span_masks, Mapping) and "scene" in span_masks:
        scene_t = span_masks["scene"]
        if isinstance(scene_t, torch.Tensor) and not bool(getattr(scene_t, "_h3_has_sections", False)):
            vis_t = span_masks.get("vision")
            prose_rows = (
                (scene_t & (~vis_t)).to(device=device)
                if isinstance(vis_t, torch.Tensor) and vis_t.shape == scene_t.shape
                else (scene_t & (tags_2d[0] != 0)).to(device=device)
            )
            sel = _spans.split_fields_mask(sp.mask)
            if sel is not None:
                if any(f in ("description", "summary", "subject_definitions") for f in sel):
                    return prose_rows
                return torch.zeros((L,), dtype=torch.bool, device=device)
            if sp.mask == "soundscape_only":
                return torch.zeros((L,), dtype=torch.bool, device=device)
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


def _has_video_ref_cond(payload: Mapping[str, Any]) -> bool:
    for r in payload.get("refs") or ():
        if isinstance(r, Mapping) and r.get("kind") in ("video", "video_audio"):
            return True
    return False


def _total_frames_for_latent_t(latent_t: int) -> int:
    if latent_t <= 0:
        return 0
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN

    return sum(int(FRAME_PER_TOKEN[k % 5]) for k in range(int(latent_t)))


def _is_midclip_guide_kf(kf: Mapping[str, Any], total_frames: int) -> bool:
    if kf.get("latent") is None or total_frames <= 2:
        return False
    idx = int(kf.get("resolved_frame_index", 0))
    return 0 < idx < (total_frames - 1)


def _has_midclip_guide_cond(payload: Mapping[str, Any], total_frames: int) -> bool:
    for kf in payload.get("keyframes") or ():
        if isinstance(kf, Mapping) and _is_midclip_guide_kf(kf, total_frames):
            return True
    return False


def _cond_keep_mask(
    keyframes: Any, total_frames: int, *, drop_midclip_guide: bool
) -> tuple[bool, ...]:
    return tuple(
        not (drop_midclip_guide and _is_midclip_guide_kf(kf, total_frames))
        for kf in (keyframes or ())
        if isinstance(kf, Mapping) and kf.get("latent") is not None
    )


def _ref_img_segment_kinds(refs: Any) -> list[str]:
    return [
        str(r.get("kind"))
        for r in (refs or ())
        if isinstance(r, Mapping) and r.get("kind") in ("image", "video", "video_audio")
    ]


def _keyframe_audio_count(keyframes: Any) -> int:
    return sum(
        1
        for kf in (keyframes or ())
        if isinstance(kf, Mapping) and kf.get("audio_latent") is not None
    )


def _has_guide_audio_cond(payload: Mapping[str, Any]) -> bool:
    if _keyframe_audio_count(payload.get("keyframes")) > 0:
        return True
    layout = payload.get("layout")
    if hasattr(layout, "segments"):
        return any(kind == "cond_audio" for _a, _b, kind in layout.segments)
    return False


def _has_any_audio_cond(payload: Mapping[str, Any]) -> bool:
    if payload.get("cond_audio_latents") or _has_guide_audio_cond(payload):
        return True
    for r in payload.get("refs") or ():
        if isinstance(r, Mapping) and (
            r.get("audio_latent") is not None or r.get("kind") in ("audio", "video_audio")
        ):
            return True
    return False


def _strip_cond_entries(
    keyframes: Any,
    refs: Any,
    *,
    drop_visual: bool,
    drop_guide_audio: bool,
    drop_video_ref: bool = False,
    drop_midclip_guide: bool = False,
    total_frames: int = 0,
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
    kept_kf: list[dict[str, Any]] = []
    for kf in keyframes or ():
        if not isinstance(kf, Mapping):
            continue
        drop_this_vis = bool(
            drop_visual or (drop_midclip_guide and _is_midclip_guide_kf(kf, total_frames))
        )
        has_vis = (not drop_this_vis) and (kf.get("latent") is not None)
        has_aud = (not drop_guide_audio) and (kf.get("audio_latent") is not None)
        if has_vis or has_aud:
            entry = dict(kf)
            if drop_this_vis:
                entry.pop("latent", None)
            if drop_guide_audio:
                entry.pop("audio_latent", None)
            kept_kf.append(entry)

    if not drop_visual and not drop_video_ref:
        kept_refs = [dict(blk) for blk in (refs or ()) if isinstance(blk, Mapping)]
        return (kept_kf or None), (kept_refs or None)

    kept_refs = []
    for blk in refs or ():
        if not isinstance(blk, Mapping):
            continue
        kind = blk.get("kind")
        if kind == "audio":
            kept_refs.append(dict(blk))
        elif kind == "image" and not drop_visual:
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
        elif not drop_visual and not drop_video_ref:
            kept_refs.append(dict(blk))

    return (kept_kf or None), (kept_refs or None)


def _strip_visual_cond_entries(
    keyframes: Any, refs: Any
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
    return _strip_cond_entries(keyframes, refs, drop_visual=True, drop_guide_audio=False)


def _drop_cond_from_layout(
    orig_layout: Any,
    *,
    drop_visual: bool,
    drop_guide_audio: bool,
    drop_video_ref: bool = False,
    ref_img_kinds: Sequence[str] | None = None,
    drop_midclip_guide: bool = False,
    cond_keep_mask: Sequence[bool] | None = None,
) -> tuple[Any, int, int]:
    if not (
        hasattr(orig_layout, "segments")
        and hasattr(orig_layout, "position_ids")
        and hasattr(orig_layout, "signature")
    ):
        return orig_layout, 0, 0

    eff_drop_vid_ref = bool(drop_video_ref and not drop_visual)
    eff_drop_guide = bool(drop_midclip_guide and not drop_visual)
    kinds_seq = tuple(ref_img_kinds or ())
    cond_keep = tuple(cond_keep_mask) if cond_keep_mask is not None else ()

    dropped_vis = 0
    dropped_aud = 0
    ref_idx_scan = 0
    cond_idx_scan = 0
    for a, b, kind in orig_layout.segments:
        n = int(b) - int(a)
        if kind == "ref_img":
            rk = kinds_seq[ref_idx_scan] if ref_idx_scan < len(kinds_seq) else None
            ref_idx_scan += 1
            if drop_visual or (eff_drop_vid_ref and rk in ("video", "video_audio")):
                dropped_vis += n
        elif kind == "cond":
            keep_cond = cond_keep[cond_idx_scan] if cond_idx_scan < len(cond_keep) else True
            cond_idx_scan += 1
            if drop_visual or (eff_drop_guide and not keep_cond):
                dropped_vis += n
        elif kind == "cond_audio" and drop_guide_audio:
            dropped_aud += n

    if dropped_vis == 0 and dropped_aud == 0:
        return orig_layout, 0, 0

    kept_pos: list[torch.Tensor] = []
    new_segments: list[tuple[int, int, str]] = []
    new_img_pos: list[torch.Tensor] = []
    new_img_update: list[torch.Tensor] = []
    new_audio_pos: list[torch.Tensor] = []
    new_audio_update: list[torch.Tensor] = []
    off = 0
    ref_idx = 0
    cond_idx = 0
    for a, b, kind in orig_layout.segments:
        if kind == "ref_img":
            rk = kinds_seq[ref_idx] if ref_idx < len(kinds_seq) else None
            ref_idx += 1
            if drop_visual or (eff_drop_vid_ref and rk in ("video", "video_audio")):
                continue
        elif kind == "cond":
            keep_cond = cond_keep[cond_idx] if cond_idx < len(cond_keep) else True
            cond_idx += 1
            if drop_visual or (eff_drop_guide and not keep_cond):
                continue
        elif drop_guide_audio and kind == "cond_audio":
            continue
        n = int(b) - int(a)
        kept_pos.append(orig_layout.position_ids[int(a) : int(b)])
        new_segments.append((off, off + n, str(kind)))
        if kind in ("cond", "ref_img"):
            new_img_pos.append(torch.arange(off, off + n))
            new_img_update.append(torch.zeros(n, dtype=torch.bool))
        elif kind == "video":
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
    return released, dropped_vis, dropped_aud


def _drop_visual_cond_from_layout(orig_layout: Any) -> tuple[Any, int]:
    released, dropped_vis, _dropped_aud = _drop_cond_from_layout(
        orig_layout, drop_visual=True, drop_guide_audio=False
    )
    return released, dropped_vis


def _specs_need_attention_hooks(
    specs: Sequence[_AppliedVectorSpec],
    dm: Any,
    *,
    is_h3_image: bool,
) -> bool:
    if not _steer_hooks.is_hookable_dit(dm):
        return False
    for sp in specs:
        if getattr(sp, "block_band", "all") != "all":
            return True
        if (
            not is_h3_image
            and float(getattr(sp, "video_scale", 1.0)) != float(getattr(sp, "audio_scale", 1.0))
            and getattr(sp, "split_backend", "auto_1x_kv") == "auto_1x_kv"
        ):
            return True
    return False


def _make_control_vector_wrapper(
    specs: tuple[_AppliedVectorSpec, ...],
    *,
    is_h3_image: bool = False,
    default_shift_video: float = 12.0,
    default_shift_audio: float = 3.0,
    dm_ref: weakref.ref[Any] | None = None,
    workspace: Any = None,
):
    logged_prefix_notes: set[tuple[str, str]] = set()
    logged_interactions: set[tuple[tuple[str, float, str], ...]] = set()
    logged_ref_notes: set[str] = set()
    logged_prompt_couplings: set[tuple[str, str]] = set()
    logged_steered_rows: set[tuple[int, str, int, int, str, float]] = set()
    logged_stack_warnings: set[tuple[str, int, str, str]] = set()
    resolved_prompt_vectors: OrderedDict[tuple[str, str], Any] = OrderedDict()
    last_resolved_by_spec: dict[str, Any] = {}
    released_layout_cache: weakref.WeakKeyDictionary[
        Any,
        dict[tuple[bool, bool, bool, bool], tuple[Any, int, int]],
    ] = weakref.WeakKeyDictionary()

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
        min_aud_end = min((sp.audio_cond_end_percent for sp in ref_specs), default=1.0)
        min_aud_aug = min((sp.audio_cond_aug for sp in ref_specs), default=1.0)
        min_vid_ref_end = min((sp.video_ref_end_percent for sp in ref_specs), default=1.0)
        min_guide_end = min((sp.guide_cond_end_percent for sp in ref_specs), default=1.0)
        live_policy = (
            transformer_options.get("_h3_live_policy")
            if isinstance(transformer_options, dict)
            else None
        )
        pol_vid_end = getattr(live_policy, "recommended_video_end_percent", None) if live_policy is not None else None
        is_av_x = isinstance(x, (list, tuple)) and len(x) >= 2
        eff_ref_end = (
            min(min_ref_end, max(0.05, float(pol_vid_end)))
            if (is_av_x and pol_vid_end is not None and float(pol_vid_end) < 0.40)
            else min_ref_end
        )
        eff_vid_end = (
            min(eff_ref_end, min_vid_ref_end, max(0.05, min(1.0, float(pol_vid_end))))
            if pol_vid_end is not None
            else min(eff_ref_end, min_vid_ref_end)
        )
        eff_guide_end = min(eff_ref_end, min_guide_end)

        latent_t = 0
        if is_av_x and torch.is_tensor(x[0]) and x[0].ndim == 5:
            latent_t = int(x[0].shape[2])
        elif torch.is_tensor(x) and x.ndim == 5:
            latent_t = int(x.shape[2])
        else:
            lay_pre = payload.get("layout")
            sig_pre = getattr(lay_pre, "signature", None)
            if sig_pre is not None and len(sig_pre) >= 2:
                latent_t = int(sig_pre[1])
        total_frames = _total_frames_for_latent_t(latent_t) if eff_guide_end < 1.0 else 0

        has_vis_cond = _has_visual_cond(payload) if (eff_ref_end < 1.0 or min_ref_aug < 0.999) else False
        has_vid_ref_cond = _has_video_ref_cond(payload) if eff_vid_end < 1.0 else False
        has_guide_cond = _has_midclip_guide_cond(payload, total_frames) if eff_guide_end < 1.0 else False
        has_guide_aud = _has_guide_audio_cond(payload) if min_aud_end < 1.0 else False
        has_any_aud = _has_any_audio_cond(payload) if min_aud_aug < 1.0 else False

        vis_active = (
            _step_in_window(transformer_options, timestep, 0.0, eff_ref_end)
            if (has_vis_cond and eff_ref_end < 1.0)
            else True
        )
        drop_vis = bool(has_vis_cond and not vis_active)
        vid_ref_active = (
            _step_in_window(transformer_options, timestep, 0.0, eff_vid_end)
            if (has_vid_ref_cond and not drop_vis and eff_vid_end < 1.0)
            else True
        )
        guide_active = (
            _step_in_window(transformer_options, timestep, 0.0, eff_guide_end)
            if (has_guide_cond and not drop_vis and eff_guide_end < 1.0)
            else True
        )
        aud_active = (
            _step_in_window(transformer_options, timestep, 0.0, min_aud_end)
            if (has_guide_aud and min_aud_end < 1.0)
            else True
        )
        drop_vid_ref = bool(has_vid_ref_cond and not drop_vis and not vid_ref_active)
        drop_guide = bool(has_guide_cond and not drop_vis and not guide_active)
        drop_aud = bool(has_guide_aud and not aud_active)
        want_vis_aug = bool(has_vis_cond and not drop_vis and min_ref_aug < 0.999 and is_av_x)
        want_aud_aug = bool(has_any_aud and min_aud_aug < 1.0 and is_av_x)

        if drop_vis or drop_vid_ref or drop_guide or drop_aud or want_vis_aug or want_aud_aug:
            new_payload = dict(payload)
            if drop_vis or drop_vid_ref or drop_guide or drop_aud:
                orig_layout = payload.get("layout")
                target_sig: tuple[int, int, int, int, int] | None = None
                if (
                    is_av_x
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
                elif torch.is_tensor(x) and x.ndim == 5:
                    target_sig = (
                        int(context.shape[1]),
                        int(x.shape[2]),
                        (int(x.shape[3]) + 1) // 2 * 2,
                        (int(x.shape[4]) + 1) // 2 * 2,
                        0,
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

                ref_img_kinds = _ref_img_segment_kinds(payload.get("refs")) if drop_vid_ref else ()
                cond_keep = (
                    _cond_keep_mask(
                        payload.get("keyframes"), total_frames, drop_midclip_guide=True
                    )
                    if drop_guide
                    else ()
                )
                flags_key = (drop_vis, drop_vid_ref, drop_guide, drop_aud)
                per_layout = None
                if orig_layout is not None:
                    try:
                        per_layout = released_layout_cache.get(orig_layout)
                    except TypeError:
                        per_layout = None
                cache_entry = per_layout.get(flags_key) if per_layout is not None else None
                if cache_entry is not None:
                    cached_layout, dropped_vis_rows, dropped_aud_rows = cache_entry
                    released_layout = orig_layout if cached_layout is None else cached_layout
                else:
                    released_layout, dropped_vis_rows, dropped_aud_rows = _drop_cond_from_layout(
                        orig_layout,
                        drop_visual=drop_vis,
                        drop_guide_audio=drop_aud,
                        drop_video_ref=drop_vid_ref,
                        ref_img_kinds=ref_img_kinds,
                        drop_midclip_guide=drop_guide,
                        cond_keep_mask=cond_keep,
                    )
                    if orig_layout is not None:
                        try:
                            if per_layout is None:
                                per_layout = {}
                                released_layout_cache[orig_layout] = per_layout
                            cached_layout = None if released_layout is orig_layout else released_layout
                            per_layout[flags_key] = (
                                cached_layout,
                                dropped_vis_rows,
                                dropped_aud_rows,
                            )
                        except TypeError:
                            pass
                kept_kf, kept_refs = _strip_cond_entries(
                    payload.get("keyframes"),
                    payload.get("refs"),
                    drop_visual=drop_vis,
                    drop_guide_audio=drop_aud,
                    drop_video_ref=drop_vid_ref,
                    drop_midclip_guide=drop_guide,
                    total_frames=total_frames,
                )

                if drop_vis:
                    new_payload["cond_video_latents"] = []
                elif drop_vid_ref or drop_guide:
                    orig_vid_lats = list(payload.get("cond_video_latents") or ())
                    if orig_vid_lats:
                        n_kf_vis = sum(
                            1
                            for kf in (payload.get("keyframes") or ())
                            if isinstance(kf, Mapping) and kf.get("latent") is not None
                        )
                        if n_kf_vis == 0 and orig_layout is not None:
                            n_kf_vis = sum(
                                1
                                for _a, _b, k in getattr(orig_layout, "segments", ())
                                if k == "cond"
                            )
                        kf_lats = orig_vid_lats[:n_kf_vis]
                        if drop_guide:
                            kept_kf_lats = [
                                lat
                                for idx_k, lat in enumerate(kf_lats)
                                if (cond_keep[idx_k] if idx_k < len(cond_keep) else True)
                            ]
                        else:
                            kept_kf_lats = kf_lats
                        ref_lats = orig_vid_lats[n_kf_vis:]
                        if drop_vid_ref:
                            ref_lat_kinds = [
                                str(r.get("kind"))
                                for r in (payload.get("refs") or ())
                                if isinstance(r, Mapping) and "latent" in r
                            ]
                            kept_ref_lats = [
                                lat
                                for lat, rk in zip(ref_lats, ref_lat_kinds)
                                if rk not in ("video", "video_audio")
                            ]
                        else:
                            kept_ref_lats = ref_lats
                        new_payload["cond_video_latents"] = kept_kf_lats + kept_ref_lats
                if drop_aud:
                    n_drop_aud = _keyframe_audio_count(payload.get("keyframes"))
                    if n_drop_aud == 0 and orig_layout is not None:
                        n_drop_aud = sum(
                            1
                            for _a, _b, k in getattr(orig_layout, "segments", ())
                            if k == "cond_audio"
                        )
                    orig_aud_lats = list(payload.get("cond_audio_latents") or ())
                    new_payload["cond_audio_latents"] = (
                        orig_aud_lats[n_drop_aud:] if n_drop_aud > 0 else []
                    )
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

                if drop_vis and "ref_gated" not in logged_ref_notes:
                    logged_ref_notes.add("ref_gated")
                    orig_seq = getattr(orig_layout, "seq_len", None)
                    rel_seq = getattr(released_layout, "seq_len", None)
                    logger.info(
                        "[h3] control vector ref_vae_end_percent=%.2f: released %d visual cond rows (PackedLayout seq_len %s -> %s, 3D RoPE positions preserved)",
                        min_ref_end,
                        dropped_vis_rows,
                        orig_seq,
                        rel_seq,
                    )
                elif not drop_vis:
                    if drop_vid_ref and "vid_ref_gated" not in logged_ref_notes:
                        logged_ref_notes.add("vid_ref_gated")
                        orig_seq = getattr(orig_layout, "seq_len", None)
                        rel_seq = getattr(released_layout, "seq_len", None)
                        logger.info(
                            "[h3] control vector video_end_percent=%.2f: released %d <Video N> visual rows while keeping <Picture N> active (PackedLayout seq_len %s -> %s, 3D RoPE positions preserved)",
                            eff_vid_end,
                            dropped_vis_rows,
                            orig_seq,
                            rel_seq,
                        )
                    if drop_guide and "guide_gated" not in logged_ref_notes:
                        logged_ref_notes.add("guide_gated")
                        orig_seq = getattr(orig_layout, "seq_len", None)
                        rel_seq = getattr(released_layout, "seq_len", None)
                        logger.info(
                            "[h3] control vector guide_end_percent=%.2f: released %d mid-clip guide cond rows while preserving endpoints & refs (PackedLayout seq_len %s -> %s, 3D RoPE positions preserved)",
                            eff_guide_end,
                            dropped_vis_rows,
                            orig_seq,
                            rel_seq,
                        )
                if drop_aud and "aud_gated" not in logged_ref_notes:
                    logged_ref_notes.add("aud_gated")
                    orig_seq = getattr(orig_layout, "seq_len", None)
                    rel_seq = getattr(released_layout, "seq_len", None)
                    logger.info(
                        "[h3] control vector audio_end_percent=%.2f: released %d guide-audio cond rows (PackedLayout seq_len %s -> %s, ref_audio & 3D RoPE preserved)",
                        min_aud_end,
                        dropped_aud_rows,
                        orig_seq,
                        rel_seq,
                    )

            if want_vis_aug and _has_visual_cond(new_payload):
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

            if want_aud_aug and _has_any_audio_cond(new_payload):
                aud_t = _audio_stream_time(
                    timestep,
                    transformer_options,
                    default_shift_video=default_shift_video,
                    default_shift_audio=default_shift_audio,
                )
                eff_aud_aug = float(min_aud_aug) if aud_t is None else max(float(min_aud_aug), aud_t)
                new_payload["audio_cond_noise_aug"] = eff_aud_aug
                if "aud_aug" not in logged_ref_notes:
                    logged_ref_notes.add("aud_aug")
                    logger.info(
                        "[h3] control vector audio_clean_level=%.3f: audio condition rows held at max(weight, audio stream time), "
                        "so their noise matches their shifted label on every step",
                        min_aud_aug,
                    )

            call_kwargs["minimax_payload"] = new_payload
            payload = new_payload

        edit_pass = payload.get("h3_edit_pass")

        def _spec_matches_edit_pass(sp: _AppliedVectorSpec) -> bool:
            ep = getattr(sp, "edit_pass", "both")
            if ep == "edited_only":
                return edit_pass == "edited"
            if ep == "original_only":
                return edit_pass == "original"
            return True

        active_specs = [
            sp
            for sp in specs
            if sp.strength != 0.0
            and _spec_matches_edit_pass(sp)
            and _step_in_window(
                transformer_options, timestep, sp.start_percent, sp.end_percent
            )
        ]
        if not active_specs:
            return executor(x, timestep, context, transformer_options, **call_kwargs)

        dm_exec = (
            getattr(executor, "class_obj", None)
            if _steer_hooks.is_hookable_dit(getattr(executor, "class_obj", None))
            else None
        )
        if dm_exec is None:
            dm_exec = getattr(getattr(executor, "__self__", None), "diffusion_model", None)
        if dm_exec is None and dm_ref is not None:
            dm_exec = dm_ref()

        if context.ndim == 3 and context.shape[-1] != _cv.POST_REFINER_DIM:
            if dm_exec is None or not (
                hasattr(dm_exec, "token_refiner") and hasattr(dm_exec, "condition_proj")
            ):
                raise ValueError(
                    f"control vector received context with hidden_size={context.shape[-1]} "
                    f"(expected {_cv.POST_REFINER_DIM}) and could not access diffusion_model.token_refiner"
                )
            with _steer_hooks.pause_malloc_graph_ctx(), torch.no_grad():
                context = torch.stack(
                    [
                        dm_exec.token_refiner(
                            dm_exec.condition_proj(context[b_idx]),
                            transformer_options=transformer_options,
                        )
                        for b_idx in range(int(context.shape[0]))
                    ],
                    dim=0,
                )

        tags_raw = payload.get("text_token_tags")
        B, L, _D = context.shape
        with _steer_hooks.pause_malloc_graph_ctx():
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
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        is_pure_uncond = (
            edit_pass not in ("original", "edited")
            and isinstance(cond_or_uncond, (list, tuple))
            and 0 not in cond_or_uncond
        )
        if edit_pass in ("original", "edited"):
            ep_key = (
                ",".join(f"{sp.vector_name}[{getattr(sp, 'edit_pass', 'both')}]" for sp in active_specs),
                str(edit_pass),
            )
            if ep_key not in logged_prefix_notes:
                logged_prefix_notes.add(ep_key)
                logger.info(
                    "[h3] control vector (%s) running alongside MiniMaxH3EditGuider (%s pass)",
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

        jl_mod = _self_mod()._j_lens
        cv_mod = _self_mod()._cv

        def _spec_effective_vector_and_erasure(
            sp: _AppliedVectorSpec,
        ) -> tuple[torch.Tensor, tuple[Any, ...] | None]:
            eff_prompt = (sp.prompt or live_tail_text or "").strip()
            vec_out = sp.vector
            er_out: tuple[Any, ...] | None = (
                (sp.erasure_vr, sp.erasure_diag, sp.erasure_mu_d, sp.erasure_holds, sp.erasure_mu_a)
                if (sp.erasure_vr is not None and sp.erasure_diag is not None)
                else None
            )
            if sp.metadata.get("deferred_against_prompt"):
                if eff_prompt:
                    cache_key = (sp.vector_name, eff_prompt)
                    if cache_key not in resolved_prompt_vectors:
                        if jl_mod.bundle_available():
                            res_synth = jl_mod.synthesize_from_words(
                                sp.metadata["words_raw"],
                                sp.metadata["against_raw"],
                                family=sp.metadata.get("synth_family", "fl2va"),
                                hold_words=sp.metadata.get("hold_raw") or None,
                                alpha=float(sp.metadata.get("hold_alpha", 1.0)),
                                prompt=eff_prompt,
                            )
                            res_er = (
                                (
                                    res_synth["erasure.vr"],
                                    res_synth["erasure.diag"],
                                    res_synth.get("erasure.mu_d"),
                                    sp.erasure_holds,
                                )
                                if ("erasure.vr" in res_synth and "erasure.diag" in res_synth)
                                else er_out
                            )
                            resolved_prompt_vectors[cache_key] = (res_synth["vector.post_refiner"], res_er)
                            logger.info(
                                "[h3] resolved %s against live prompt subject %s -> PR(+) %s | PR(-) %s",
                                sp.vector_name,
                                res_synth["neg_words_matched"],
                                res_synth["readout"].get("pos_summary", ""),
                                res_synth["readout"].get("neg_summary", ""),
                            )
                        else:
                            resolved_prompt_vectors[cache_key] = (sp.vector, er_out)
                        while len(resolved_prompt_vectors) > 64:
                            resolved_prompt_vectors.popitem(last=False)
                    else:
                        resolved_prompt_vectors.move_to_end(cache_key)
                    vec_out, er_out = resolved_prompt_vectors[cache_key]
                    last_resolved_by_spec[sp.vector_name] = (vec_out, er_out)
                else:
                    if (sp.vector_name, "missing_prompt_warn") not in logged_prompt_couplings:
                        logged_prompt_couplings.add((sp.vector_name, "missing_prompt_warn"))
                        logger.warning(
                            "[h3] control vector %r uses against_words=%r, but conditioning has no _h3_span_masks "
                            "(wire MiniMaxH3Conditioning or MiniMaxH3ReferenceConditioning); falling back to @replace_human",
                            sp.vector_name,
                            sp.metadata.get("against_raw", "@prompt_subject"),
                        )
                    vec_out, er_out = last_resolved_by_spec.get(sp.vector_name, (sp.vector, er_out))
            if eff_prompt:
                pc_key = (sp.vector_name, eff_prompt[:120])
                if pc_key not in logged_prompt_couplings:
                    logged_prompt_couplings.add(pc_key)
                    if jl_mod.bundle_available():
                        try:
                            raw_w = str(sp.metadata.get("words_raw", ""))
                            pairs_m = sp.metadata.get("pairs") if isinstance(sp.metadata.get("pairs"), dict) else {}
                            b_side_w = str(pairs_m.get("b_side", "")) if isinstance(pairs_m, dict) else ""
                            seed_w = jl_mod._parse_word_list(raw_w or b_side_w)
                            seed_w = [w for w in seed_w if w not in jl_mod._PROMPT_RECEPTOR_STOPWORDS][:6]
                            pc = jl_mod.analyze_prompt_coupling(
                                vec_out,
                                eff_prompt,
                                pos_words=seed_w or None,
                                family=str(sp.metadata.get("synth_family") or sp.metadata.get("family", "fl2va")),
                                already_contrasted=bool(sp.metadata.get("deferred_against_prompt")),
                            )
                            logger.info(
                                "[h3] control vector %s prompt coupling: %s",
                                sp.vector_name,
                                pc.get("summary", ""),
                            )
                        except Exception:
                            pass
            return vec_out, er_out

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
            if live_policy is not None:
                raw_ps = getattr(live_policy, "strength_scale", 1.0)
                if raw_ps is not None:
                    pol_scale = max(0.35, min(1.25, float(raw_ps)))
                    s_val *= pol_scale
            return s_val

        def _build_steered_context(
            base_ctx: torch.Tensor,
            stream_attr: str | None,
            specs_subset: Sequence[_AppliedVectorSpec] = active_specs,
        ) -> torch.Tensor:
            if not specs_subset:
                return base_ctx
            cur_ctx = base_ctx
            applied_add_names: list[str] = []
            groups: list[dict[str, Any]] = []

            for sp in specs_subset:
                custom_mask = _resolve_custom_inference_mask(
                    sp,
                    tags_raw=tags_raw,
                    tags_2d=tags,
                    payload=payload,
                    L=L,
                    device=cur_ctx.device,
                    kwarg_span_masks=kwarg_span_masks,
                    is_h3_image=is_h3_image,
                )
                mask_vision = sp.mask not in ("all_rows", "vision_tokens")
                eff_rm = cv_mod.resolve_post_refiner_row_mask(
                    tags,
                    batch_size=cur_ctx.shape[0],
                    seq_len=L,
                    device=cur_ctx.device,
                    mask=custom_mask,
                    mask_vision=mask_vision,
                )
                resolved_key = (eff_rm.detach().cpu().numpy().tobytes(), sp.prompt)
                s_eff = _effective_spec_strength(sp, stream_attr)
                vec_out, er_out = _spec_effective_vector_and_erasure(sp)
                em = _effective_entry_mode(sp)
                tf_val = float(sp.token_focus)

                merged = False
                if cv_mod.is_additive_mode(em) and tf_val >= 1.0 - 1e-9:
                    for gi in range(len(groups) - 1, -1, -1):
                        cand = groups[gi]
                        if (
                            cand["allow_add_merge"]
                            and not cand["locked"]
                            and cand["resolved_key"] == resolved_key
                        ):
                            has_intervening_overlap = any(
                                bool((groups[gk]["eff_rm"] & eff_rm).any().item())
                                for gk in range(gi + 1, len(groups))
                            )
                            if not has_intervening_overlap:
                                cand["specs"].append(sp)
                                cand["entries"].append((vec_out, s_eff, em))
                                cand["erasure_specs"].append(er_out)
                                merged = True
                            break
                    if not merged:
                        for prev in groups:
                            if bool((prev["eff_rm"] & eff_rm).any().item()):
                                prev["locked"] = True
                        groups.append(
                            {
                                "resolved_key": resolved_key,
                                "eff_rm": eff_rm,
                                "custom_mask": custom_mask,
                                "mask_vision": mask_vision,
                                "token_focus": tf_val,
                                "specs": [sp],
                                "entries": [(vec_out, s_eff, em)],
                                "erasure_specs": [er_out],
                                "is_add": True,
                                "allow_add_merge": True,
                                "allow_hold_merge": True,
                                "locked": False,
                            }
                        )
                elif em == "hold":
                    for gi in range(len(groups) - 1, -1, -1):
                        cand = groups[gi]
                        if (
                            cand["allow_hold_merge"]
                            and not cand["locked"]
                            and cand["resolved_key"] == resolved_key
                        ):
                            has_intervening_overlap = any(
                                bool((groups[gk]["eff_rm"] & eff_rm).any().item())
                                for gk in range(gi + 1, len(groups))
                            )
                            if not has_intervening_overlap:
                                cand["specs"].append(sp)
                                cand["entries"].append((vec_out, s_eff, em))
                                cand["erasure_specs"].append(er_out)
                                merged = True
                            break
                    if not merged:
                        has_diff_add = any(
                            g["is_add"] and g["resolved_key"] != resolved_key
                            for g in groups
                        )
                        has_same_add = any(
                            g["is_add"] and g["resolved_key"] == resolved_key
                            for g in groups
                        )
                        if has_diff_add and not has_same_add:
                            first_add_mask = next(
                                g["specs"][0].mask for g in reversed(groups) if g["is_add"]
                            )
                            raise ValueError(
                                f"mode='hold' on {sp.vector_name!r} has mask={sp.mask!r}, "
                                f"which differs from preceding additive control vector mask={first_add_mask!r}; "
                                "a 'hold' node must use the same mask as the additive node(s) it projects out of"
                            )
                        for prev in groups:
                            if bool((prev["eff_rm"] & eff_rm).any().item()):
                                prev["locked"] = True
                        groups.append(
                            {
                                "resolved_key": resolved_key,
                                "eff_rm": eff_rm,
                                "custom_mask": custom_mask,
                                "mask_vision": mask_vision,
                                "token_focus": tf_val,
                                "specs": [sp],
                                "entries": [(vec_out, s_eff, em)],
                                "erasure_specs": [er_out],
                                "is_add": False,
                                "allow_add_merge": False,
                                "allow_hold_merge": False,
                                "locked": True,
                            }
                        )
                else:
                    for prev in groups:
                        if bool((prev["eff_rm"] & eff_rm).any().item()):
                            prev["locked"] = True
                    is_add_focused = cv_mod.is_additive_mode(em)
                    groups.append(
                        {
                            "resolved_key": resolved_key,
                            "eff_rm": eff_rm,
                            "custom_mask": custom_mask,
                            "mask_vision": mask_vision,
                            "token_focus": tf_val,
                            "specs": [sp],
                            "entries": [(vec_out, s_eff, em)],
                            "erasure_specs": [er_out],
                            "is_add": is_add_focused,
                            "allow_add_merge": False,
                            "allow_hold_merge": is_add_focused,
                            "locked": not is_add_focused,
                        }
                    )

            for gi, grp in enumerate(groups):
                group_specs = grp["specs"]
                batch_entries = grp["entries"]
                batch_erasure_specs = grp["erasure_specs"]
                custom_mask = grp["custom_mask"]
                mask_vision = grp["mask_vision"]
                eff_rm = grp["eff_rm"]
                sp0 = group_specs[0]
                v0 = batch_entries[0][0]
                mask_desc = "+".join(dict.fromkeys(gsp.mask for gsp in group_specs))

                add_inter_entries: list[tuple[str, torch.Tensor, float]] = []
                for gsp, (gv, gs_eff, em) in zip(group_specs, batch_entries):
                    if cv_mod.is_additive_mode(em) and gs_eff != 0.0:
                        eff_v = (
                            gsp.erasure_mu_d
                            if (em == "add_norm_matched_erasure" and gs_eff > 0.0 and gsp.erasure_mu_d is not None)
                            else gv
                        )
                        eff_v64 = eff_v.detach().to(dtype=torch.float64)
                        v_nrm = float(torch.linalg.vector_norm(eff_v64).item())
                        eff_s = (
                            gs_eff * (cv_mod.NORM_MATCHED_TARGET / v_nrm)
                            if (cv_mod.is_norm_matched_mode(em) and v_nrm > 1e-12)
                            else gs_eff
                        )
                        if em == "add_norm_matched_erasure" and gs_eff > 0.0 and gsp.erasure_holds:
                            eff_v = cv_mod._apply_holds_to_shift64(eff_s * eff_v64, gsp.erasure_holds, eff_v64.device)
                            eff_s = 1.0
                        add_inter_entries.append((gsp.vector_name, eff_v, eff_s))
                if len(add_inter_entries) >= 2:
                    inter_key = tuple(
                        (gsp.vector_name, round(gs_eff, 6), em)
                        for gsp, (_v, gs_eff, em) in zip(group_specs, batch_entries)
                    )
                    if inter_key not in logged_interactions:
                        logged_interactions.add(inter_key)
                        inter = cv_mod.vector_interactions(add_inter_entries)
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

                for gsp, (_v, gs_eff, em) in zip(group_specs, batch_entries):
                    if cv_mod.is_additive_mode(em) and gs_eff != 0.0:
                        applied_add_names.append(gsp.vector_name)
                if any(s_eff != 0.0 for (_v, s_eff, _m) in batch_entries):
                    if not is_pure_uncond:
                        row_key = (gi, mask_desc, L, int(has_vision), str(edit_pass or ""), round(float(sp0.token_focus), 4))
                        if row_key not in logged_steered_rows:
                            logged_steered_rows.add(row_key)
                            n_steered = int(eff_rm[0].sum().item())
                            focus_note = ""
                            if float(sp0.token_focus) < 1.0 - 1e-9 and n_steered > 0:
                                aim_shift64 = torch.zeros(cur_ctx.shape[-1], dtype=torch.float64, device=cur_ctx.device)
                                aim_holds: list[tuple[torch.Tensor, float]] = []
                                has_aim_add = False
                                for gsp_a, (gv, gs_eff, em) in zip(group_specs, batch_entries):
                                    if gs_eff == 0.0:
                                        continue
                                    if cv_mod.is_additive_mode(em):
                                        has_aim_add = True
                                        eff_v = (
                                            gsp_a.erasure_mu_d
                                            if (em == "add_norm_matched_erasure" and gs_eff > 0.0 and gsp_a.erasure_mu_d is not None)
                                            else gv
                                        )
                                        gv64 = eff_v.detach().to(device=cur_ctx.device, dtype=torch.float64)
                                        eff_gs = (
                                            gs_eff * (cv_mod.NORM_MATCHED_TARGET / max(float(torch.linalg.vector_norm(gv64).item()), 1e-12))
                                            if cv_mod.is_norm_matched_mode(em)
                                            else gs_eff
                                        )
                                        term_aim64 = eff_gs * gv64
                                        if em == "add_norm_matched_erasure" and gs_eff > 0.0 and gsp_a.erasure_holds:
                                            term_aim64 = cv_mod._apply_holds_to_shift64(term_aim64, gsp_a.erasure_holds, cur_ctx.device)
                                        aim_shift64 = aim_shift64 + term_aim64
                                    elif em == "hold":
                                        aim_holds.append((gv, gs_eff))
                                if has_aim_add and aim_holds:
                                    aim_shift64 = cv_mod._apply_holds_to_shift64(aim_shift64, aim_holds, cur_ctx.device)
                                aim_vec_log = aim_shift64 if has_aim_add else v0
                                foc_rm, min_abs_cos = cv_mod.focus_row_mask(
                                    cur_ctx,
                                    eff_rm,
                                    aim_vec_log,
                                    token_focus=float(sp0.token_focus),
                                )
                                n_focused = int(foc_rm[0].sum().item())
                                focus_note = f", token_focus={float(sp0.token_focus):.2f} ({n_focused}/{n_steered} rows, |cos|>={min_abs_cos:.3f})"
                            logger.info(
                                "[h3] control vector (%s) steered %d/%d context rows (mask=%s%s%s)",
                                ",".join(gsp.vector_name for gsp in group_specs),
                                n_steered,
                                L,
                                mask_desc,
                                focus_note,
                                f", {edit_pass} pass" if edit_pass in ("original", "edited") else "",
                            )
                            if n_steered == 0 and _spans.split_fields_mask(sp0.mask) is not None:
                                logger.warning(
                                    "[h3] control vector (%s) has no effect: mask=%s selects the prose of the "
                                    "named field(s), and this prompt has none of them or they read N/A",
                                    ",".join(gsp.vector_name for gsp in group_specs),
                                    sp0.mask,
                                )
                    cur_ctx = cv_mod.apply_post_refiner(
                        cur_ctx,
                        tags,
                        batch_entries,
                        scale_mode="raw",
                        mask=custom_mask,
                        mask_vision=mask_vision,
                        warn_if_stacked_norm=False,
                        token_focus=float(sp0.token_focus),
                        erasure_specs=batch_erasure_specs,
                    )
            if not is_pure_uncond and len(applied_add_names) >= 2:
                diff64 = (cur_ctx - base_ctx).detach().to(dtype=torch.float64)
                shift_norms = torch.linalg.vector_norm(diff64, dim=-1)
                steered_rows_mask = shift_norms > 1e-6
                if bool(steered_rows_mask.any().item()):
                    peak_shift = float(shift_norms[steered_rows_mask].max().item())
                    if peak_shift >= cv_mod.STACK_SHIFT_WARN_NORM:
                        stack_key = (
                            ",".join(applied_add_names),
                            L,
                            str(stream_attr or "all"),
                            str(edit_pass or ""),
                        )
                        if stack_key not in logged_stack_warnings:
                            logged_stack_warnings.add(stack_key)
                            text_rows_b0 = base_ctx[0, tags[0] == 1] if bool((tags[0] == 1).any().item()) else base_ctx[0]
                            base_med = float(
                                torch.linalg.vector_norm(text_rows_b0.detach().to(dtype=torch.float64), dim=-1)
                                .median()
                                .item()
                            )
                            pct = (100.0 * peak_shift / base_med) if base_med > 1e-9 else 0.0
                            n_over = int((shift_norms[0] >= cv_mod.STACK_SHIFT_WARN_NORM).sum().item())
                            logger.warning(
                                "[h3] stacked control vectors (%s) peak net row shift=%.2f on %d/%d rows "
                                "(%.0f%% of base median row norm %.1f; threshold=%.1f) — use disjoint targets "
                                "(e.g. description_only + look_only on ref2va) or lower per-dial strengths to 0.8..1.0",
                                stack_key[0],
                                peak_shift,
                                n_over,
                                L,
                                pct,
                                base_med,
                                cv_mod.STACK_SHIFT_WARN_NORM,
                            )
            return cur_ctx

        has_audio_stream = isinstance(x, (list, tuple)) and len(x) >= 2
        need_split = has_audio_stream and any(
            _spec_stream_scale(sp, "video_scale") != _spec_stream_scale(sp, "audio_scale")
            for sp in active_specs
        )
        dit_replaces = (
            transformer_options.get("patches_replace", {}).get("dit", {})
            if isinstance(transformer_options, dict)
            else {}
        )
        has_block_hooks = bool(
            workspace is not None
            and _steer_hooks.is_hookable_dit(dm_exec)
            and ("double_block", 0) in dit_replaces
            and _steer_hooks.has_steer_block_patch(dit_replaces[("double_block", 0)])
        )
        n_blocks_exec = len(dm_exec.blocks) if _steer_hooks.is_hookable_dit(dm_exec) else 50
        all_blocks_fset = frozenset(range(n_blocks_exec))

        vid_non_all_bands: dict[str, frozenset[int]] = {}
        aud_non_all_bands: dict[str, frozenset[int]] = {}
        vid_all_band_specs: list[_AppliedVectorSpec] = []
        aud_all_band_specs: list[_AppliedVectorSpec] = []
        for sp in active_specs:
            bb_raw = getattr(sp, "block_band", "all")
            v_sc = _spec_stream_scale(sp, "video_scale")
            a_sc = _spec_stream_scale(sp, "audio_scale")
            if bb_raw == "all":
                if v_sc != 0.0:
                    vid_all_band_specs.append(sp)
                if a_sc != 0.0:
                    aud_all_band_specs.append(sp)
            else:
                bset = _steer_hooks.parse_block_band(bb_raw, n_blocks_exec)
                if has_block_hooks and bset == all_blocks_fset:
                    if v_sc != 0.0:
                        vid_all_band_specs.append(sp)
                    if a_sc != 0.0:
                        aud_all_band_specs.append(sp)
                else:
                    if v_sc != 0.0:
                        vid_non_all_bands[bb_raw] = bset
                    if a_sc != 0.0:
                        aud_non_all_bands[bb_raw] = bset

        if len(vid_non_all_bands) > 1:
            raise ValueError(
                "MiniMaxH3Steer cannot combine multiple distinct non-'all' block_band ranges on the video stream "
                f"in the same step: {sorted(vid_non_all_bands.keys())}"
            )
        if len(aud_non_all_bands) > 1:
            raise ValueError(
                "MiniMaxH3Steer cannot combine multiple distinct non-'all' block_band ranges on the audio stream "
                f"in the same step: {sorted(aud_non_all_bands.keys())}"
            )

        distinct_bands = {**vid_non_all_bands, **aud_non_all_bands}
        has_band_limit = bool(distinct_bands)
        any_legacy_split = need_split and any(
            getattr(sp, "split_backend", "auto_1x_kv") == "legacy_2x_pass"
            for sp in active_specs
            if _spec_stream_scale(sp, "video_scale") != _spec_stream_scale(sp, "audio_scale")
        )
        if has_band_limit and not has_block_hooks:
            raise ValueError(
                f"MiniMaxH3Steer(block_band={next(iter(distinct_bands))!r}) requires a hookable "
                "MiniMaxH3Model with H3SteerBlockPatch entries in patches_replace['dit']"
            )

        def _merge_stream_outputs(out_v: Any, out_a: Any) -> Any:
            if (
                isinstance(out_v, list)
                and len(out_v) >= 2
                and isinstance(out_a, (list, tuple))
                and len(out_a) >= 2
            ):
                return [out_v[0], out_a[1], *out_v[2:]]
            if (
                isinstance(out_v, tuple)
                and len(out_v) >= 2
                and isinstance(out_a, (list, tuple))
                and len(out_a) >= 2
            ):
                return (out_v[0], out_a[1], *out_v[2:])
            return out_v

        def _invoke_with_steer_runtime(
            *,
            mode_id: int,
            main_ctx: torch.Tensor,
            shadow_a_init: torch.Tensor,
            shadow_b_init: torch.Tensor | None,
            swap_blocks: frozenset[int],
            aud_split_blocks: frozenset[int],
            dual_shadow: bool,
            fork_b_at_block: int | None,
            aud_uses_shadow_b: bool,
            aud_unswapped_blocks: frozenset[int] = frozenset(),
        ) -> Any:
            if isinstance(transformer_options, dict) and transformer_options.get("_h3_steer_runtime") is not None:
                raise ValueError(
                    "MiniMaxH3Steer 1x hook mode (streams!='both' or block_band!='all') cannot stack with another "
                    "active 1x H3SteerRuntime (e.g. MiniMaxH3QueryRouter); use streams='both' & block_band='all', "
                    "or split_backend='legacy_2x', or disable one of the two 1x attention hooks."
                )
            if isinstance(transformer_options, dict) and transformer_options.get("_h3_spatial_qr_rt") is not None:
                raise ValueError(
                    "MiniMaxH3Steer 1x hook mode (streams!='both' or block_band!='all') cannot share the same "
                    "sampler step with MiniMaxH3SpatialQueryRouter; use streams='both' & block_band='all', or "
                    "separate their start/end_percent windows."
                )
            with _steer_hooks.pause_malloc_graph_ctx():
                rt = _steer_hooks.H3SteerRuntime(
                    dm=dm_exec,
                    workspace=workspace,
                    mode=mode_id,
                    main_ctx=main_ctx,
                    shadow_a_init=shadow_a_init,
                    shadow_b_init=shadow_b_init,
                    swap_blocks=swap_blocks,
                    aud_split_blocks=aud_split_blocks,
                    dual_shadow=dual_shadow,
                    fork_b_at_block=fork_b_at_block,
                    aud_uses_shadow_b=aud_uses_shadow_b,
                    aud_unswapped_blocks=aud_unswapped_blocks,
                )
                to_next = dict(transformer_options or {})
                to_next["_h3_steer_runtime"] = rt
                to_next["optimized_attention_override"] = _steer_hooks.make_steer_attention_override(
                    to_next.get("optimized_attention_override")
                )
            out_res = executor(x, timestep, rt.main_ctx, to_next, **call_kwargs)
            sink = to_next.get("_h3_steer_stats_sink")
            if callable(sink):
                sink(dict(rt.stats))
            elif isinstance(sink, list):
                sink.append(dict(rt.stats))
            return out_res

        if not has_band_limit and not need_split:
            with _steer_hooks.pause_malloc_graph_ctx():
                cur = _build_steered_context(context, "video_scale", active_specs)
            return executor(x, timestep, cur, transformer_options, **call_kwargs)

        use_1x_split = bool(need_split and has_block_hooks and not any_legacy_split)
        if not has_band_limit and need_split and not use_1x_split:
            with _steer_hooks.pause_malloc_graph_ctx():
                ctx_video = _build_steered_context(context, "video_scale", active_specs)
                ctx_audio = _build_steered_context(context, "audio_scale", active_specs)
            out_vid = executor(x, timestep, ctx_video, transformer_options, **call_kwargs)
            out_aud = executor(x, timestep, ctx_audio, transformer_options, **call_kwargs)
            return _merge_stream_outputs(out_vid, out_aud)

        vid_band_set = next(iter(vid_non_all_bands.values())) if vid_non_all_bands else all_blocks_fset
        aud_band_set = next(iter(aud_non_all_bands.values())) if aud_non_all_bands else all_blocks_fset
        if not need_split and len(distinct_bands) == 1:
            common_band = next(iter(distinct_bands.values()))
            vid_band_set = common_band
            aud_band_set = common_band

        with _steer_hooks.pause_malloc_graph_ctx():
            outside_vid = (
                _build_steered_context(context, "video_scale", vid_all_band_specs)
                if (vid_non_all_bands and vid_all_band_specs)
                else context
            )
            inside_vid = _build_steered_context(context, "video_scale", active_specs)

        if not need_split:
            band_set = vid_band_set
            is_prefix = bool(band_set) and (0 in band_set) and ((n_blocks_exec - 1) not in band_set)
            if len(band_set) == 0:
                return _invoke_with_steer_runtime(
                    mode_id=1,
                    main_ctx=outside_vid,
                    shadow_a_init=inside_vid,
                    shadow_b_init=None,
                    swap_blocks=frozenset(),
                    aud_split_blocks=frozenset(),
                    dual_shadow=False,
                    fork_b_at_block=None,
                    aud_uses_shadow_b=False,
                )
            if is_prefix:
                return _invoke_with_steer_runtime(
                    mode_id=1,
                    main_ctx=inside_vid,
                    shadow_a_init=outside_vid,
                    shadow_b_init=None,
                    swap_blocks=all_blocks_fset - band_set,
                    aud_split_blocks=frozenset(),
                    dual_shadow=False,
                    fork_b_at_block=None,
                    aud_uses_shadow_b=False,
                )
            return _invoke_with_steer_runtime(
                mode_id=1,
                main_ctx=outside_vid,
                shadow_a_init=inside_vid,
                shadow_b_init=None,
                swap_blocks=band_set,
                aud_split_blocks=frozenset(),
                dual_shadow=False,
                fork_b_at_block=None,
                aud_uses_shadow_b=False,
            )

        with _steer_hooks.pause_malloc_graph_ctx():
            outside_aud = (
                _build_steered_context(context, "audio_scale", aud_all_band_specs)
                if (aud_non_all_bands and aud_all_band_specs)
                else context
            )
            inside_aud = _build_steered_context(context, "audio_scale", active_specs)

        if not use_1x_split:
            def _run_mode1_pass(o_ctx: torch.Tensor, i_ctx: torch.Tensor, b_set: frozenset[int]) -> Any:
                if len(b_set) == 0:
                    return executor(x, timestep, o_ctx, transformer_options, **call_kwargs)
                if b_set == all_blocks_fset:
                    return executor(x, timestep, i_ctx, transformer_options, **call_kwargs)
                is_pref = (0 in b_set) and ((n_blocks_exec - 1) not in b_set)
                if is_pref:
                    return _invoke_with_steer_runtime(
                        mode_id=1, main_ctx=i_ctx, shadow_a_init=o_ctx, shadow_b_init=None,
                        swap_blocks=all_blocks_fset - b_set, aud_split_blocks=frozenset(),
                        dual_shadow=False, fork_b_at_block=None, aud_uses_shadow_b=False,
                    )
                return _invoke_with_steer_runtime(
                    mode_id=1, main_ctx=o_ctx, shadow_a_init=i_ctx, shadow_b_init=None,
                    swap_blocks=b_set, aud_split_blocks=frozenset(),
                    dual_shadow=False, fork_b_at_block=None, aud_uses_shadow_b=False,
                )

            return _merge_stream_outputs(
                _run_mode1_pass(outside_vid, inside_vid, vid_band_set),
                _run_mode1_pass(outside_aud, inside_aud, aud_band_set),
            )

        if not vid_non_all_bands and not aud_non_all_bands:
            return _invoke_with_steer_runtime(
                mode_id=2,
                main_ctx=inside_vid,
                shadow_a_init=inside_aud,
                shadow_b_init=None,
                swap_blocks=frozenset(),
                aud_split_blocks=all_blocks_fset,
                dual_shadow=False,
                fork_b_at_block=None,
                aud_uses_shadow_b=False,
            )

        early_vid_eq_aud = bool(outside_vid is outside_aud or torch.equal(outside_vid, outside_aud))
        if (vid_non_all_bands and aud_non_all_bands) and (vid_band_set != aud_band_set):
            if early_vid_eq_aud:
                return _invoke_with_steer_runtime(
                    mode_id=3,
                    main_ctx=outside_vid,
                    shadow_a_init=inside_vid,
                    shadow_b_init=inside_aud,
                    swap_blocks=vid_band_set,
                    aud_split_blocks=aud_band_set,
                    dual_shadow=True,
                    fork_b_at_block=None,
                    aud_uses_shadow_b=True,
                    aud_unswapped_blocks=vid_band_set - aud_band_set,
                )

            def _run_mode1_mixed(o_ctx: torch.Tensor, i_ctx: torch.Tensor, b_set: frozenset[int]) -> Any:
                if len(b_set) == 0:
                    return executor(x, timestep, o_ctx, transformer_options, **call_kwargs)
                is_pref = (0 in b_set) and ((n_blocks_exec - 1) not in b_set)
                if is_pref:
                    return _invoke_with_steer_runtime(
                        mode_id=1, main_ctx=i_ctx, shadow_a_init=o_ctx, shadow_b_init=None,
                        swap_blocks=all_blocks_fset - b_set, aud_split_blocks=frozenset(),
                        dual_shadow=False, fork_b_at_block=None, aud_uses_shadow_b=False,
                    )
                return _invoke_with_steer_runtime(
                    mode_id=1, main_ctx=o_ctx, shadow_a_init=i_ctx, shadow_b_init=None,
                    swap_blocks=b_set, aud_split_blocks=frozenset(),
                    dual_shadow=False, fork_b_at_block=None, aud_uses_shadow_b=False,
                )

            return _merge_stream_outputs(
                _run_mode1_mixed(outside_vid, inside_vid, vid_band_set),
                _run_mode1_mixed(outside_aud, inside_aud, aud_band_set),
            )

        band_set = vid_band_set if vid_non_all_bands else aud_band_set
        vid_const_band = bool(outside_vid is inside_vid or torch.equal(outside_vid, inside_vid))
        aud_const_band = bool(outside_aud is inside_aud or torch.equal(outside_aud, inside_aud))

        if len(band_set) == 0:
            if early_vid_eq_aud:
                return _invoke_with_steer_runtime(
                    mode_id=3, main_ctx=outside_vid, shadow_a_init=inside_vid, shadow_b_init=None,
                    swap_blocks=frozenset(), aud_split_blocks=frozenset(),
                    dual_shadow=False, fork_b_at_block=None, aud_uses_shadow_b=False,
                )
            return _invoke_with_steer_runtime(
                mode_id=2, main_ctx=outside_vid, shadow_a_init=outside_aud, shadow_b_init=None,
                swap_blocks=frozenset(), aud_split_blocks=all_blocks_fset,
                dual_shadow=False, fork_b_at_block=None, aud_uses_shadow_b=False,
            )

        if early_vid_eq_aud and vid_const_band:
            return _invoke_with_steer_runtime(
                mode_id=3,
                main_ctx=outside_vid,
                shadow_a_init=inside_aud,
                shadow_b_init=None,
                swap_blocks=frozenset(),
                aud_split_blocks=band_set,
                dual_shadow=False,
                fork_b_at_block=None,
                aud_uses_shadow_b=False,
            )

        is_prefix = (0 in band_set) and ((n_blocks_exec - 1) not in band_set)
        if is_prefix and early_vid_eq_aud and aud_const_band:
            return _invoke_with_steer_runtime(
                mode_id=3,
                main_ctx=inside_vid,
                shadow_a_init=outside_vid,
                shadow_b_init=None,
                swap_blocks=all_blocks_fset - band_set,
                aud_split_blocks=band_set,
                dual_shadow=False,
                fork_b_at_block=None,
                aud_uses_shadow_b=False,
            )

        min_b = min(band_set)
        is_suffix = band_set == frozenset(range(min_b, n_blocks_exec))
        if early_vid_eq_aud and aud_const_band:
            return _invoke_with_steer_runtime(
                mode_id=3,
                main_ctx=outside_vid,
                shadow_a_init=inside_vid,
                shadow_b_init=None if is_suffix else outside_aud,
                swap_blocks=band_set,
                aud_split_blocks=band_set,
                dual_shadow=True,
                fork_b_at_block=min_b if is_suffix else None,
                aud_uses_shadow_b=True,
            )

        if aud_const_band:
            return _invoke_with_steer_runtime(
                mode_id=3,
                main_ctx=outside_vid,
                shadow_a_init=inside_vid,
                shadow_b_init=outside_aud,
                swap_blocks=band_set,
                aud_split_blocks=all_blocks_fset,
                dual_shadow=True,
                fork_b_at_block=None,
                aud_uses_shadow_b=True,
            )

        def _run_mode1_pass_fb(o_ctx: torch.Tensor, i_ctx: torch.Tensor) -> Any:
            if is_prefix:
                return _invoke_with_steer_runtime(
                    mode_id=1, main_ctx=i_ctx, shadow_a_init=o_ctx, shadow_b_init=None,
                    swap_blocks=all_blocks_fset - band_set, aud_split_blocks=frozenset(),
                    dual_shadow=False, fork_b_at_block=None, aud_uses_shadow_b=False,
                )
            return _invoke_with_steer_runtime(
                mode_id=1, main_ctx=o_ctx, shadow_a_init=i_ctx, shadow_b_init=None,
                swap_blocks=band_set, aud_split_blocks=frozenset(),
                dual_shadow=False, fork_b_at_block=None, aud_uses_shadow_b=False,
            )

        return _merge_stream_outputs(
            _run_mode1_pass_fb(outside_vid, inside_vid),
            _run_mode1_pass_fb(outside_aud, inside_aud),
        )

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
        if preferred_family in ("fl2va", "ref2va", "fasth3", "h3_image")
        else ["fl2va", "ref2va", "fasth3", "h3_image"]
    )

    def _pick(pat: str) -> Path | None:
        direct = sorted(
            directory.glob(pat),
            key=lambda p: (0 if ("__lora_" in p.name) == prefer_lora else 1, p.name),
        )
        if direct:
            return direct[0].resolve()
        nested = sorted(
            (
                p
                for p in directory.rglob(pat)
                if not _is_hidden_vector_relpath(p.relative_to(directory))
            ),
            key=lambda p: (
                0 if ("__lora_" in p.name) == prefer_lora else 1,
                len(p.relative_to(directory).parts),
                p.relative_to(directory).as_posix(),
            ),
        )
        return nested[0].resolve() if nested else None

    for fam in fams:
        if preferred_pooling:
            hit_pool = _pick(f"{stem}__{fam}__{preferred_pooling}*.safetensors")
            if hit_pool is not None:
                return hit_pool
        hit_fam = _pick(f"{stem}__{fam}*.safetensors")
        if hit_fam is not None:
            return hit_fam
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
        if preferred_family not in ("fl2va", "ref2va", "h3_image"):
            return found
        other_fam = next(
            (
                fam
                for fam in ("fl2va", "ref2va", "h3_image")
                if fam != preferred_family and f"__{fam}" in found.name
            ),
            None,
        )
        if other_fam is None:
            return found
        tag_other = f"__{other_fam}"
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

    norm_raw = raw.replace("\\", "/")
    clean_stem = norm_raw[:-len(".safetensors")] if norm_raw.endswith(".safetensors") else norm_raw
    rel_subdir = ""
    leaf_stem = clean_stem
    if "/" in clean_stem:
        rel_subdir, leaf_stem = clean_stem.rsplit("/", 1)
    if leaf_stem and "__" not in leaf_stem and not Path(rel_subdir).is_absolute():
        eff_prefer = True if prefer_lora is None else bool(prefer_lora)
        if preferred_family == "fl2va":
            fam_tries: tuple[str | None, ...] = ("fl2va", "ref2va", None)
        elif preferred_family == "ref2va":
            fam_tries = ("ref2va", "fl2va", None)
        elif preferred_family is not None:
            fam_tries = (preferred_family, None)
        else:
            fam_tries = (None,)
        for fam_try in fam_tries:
            for folder in folders:
                search_dir = (folder / rel_subdir) if rel_subdir else folder
                matched = _match_sibling_or_stem_in_dir(
                    search_dir,
                    leaf_stem,
                    fam_try,
                    preferred_pooling="diff_span",
                    prefer_lora=eff_prefer,
                )
                if matched is not None:
                    return matched

    raise ValueError(
        f"control vector file {vector_name!r} not found in models/{CONTROL_VECTORS_FOLDER}/"
    )


def _project_encoder_vector_to_h3_image(v_enc_scaled: torch.Tensor, dm: Any) -> torch.Tensor:
    """Project a `5120`-d scaled `vector.encoder` direction into `5376`-d `h3_image` post-refiner space."""
    v32 = v_enc_scaled.detach().to(device="cpu", dtype=torch.float32).flatten()
    jl_mod = _self_mod()._j_lens
    if jl_mod.bundle_available():
        b = jl_mod.load_j_lens_bundle()
        if "J_ref_h3_image" in b:
            return (b["J_ref_h3_image"] @ v32).to(dtype=torch.float32).contiguous()
    proj_mod = getattr(dm, "condition_proj", None)
    ref_mod = getattr(dm, "token_refiner", None)
    if callable(proj_mod) and callable(ref_mod):
        w_param = getattr(proj_mod, "weight", None)
        dev = w_param.device if isinstance(w_param, torch.Tensor) else torch.device("cpu")
        dt = w_param.dtype if isinstance(w_param, torch.Tensor) else torch.float32
        with torch.no_grad():
            half = (0.5 * v32).view(1, -1).to(device=dev, dtype=dt)
            r_pos = ref_mod(proj_mod(half)).detach().to(device="cpu", dtype=torch.float32).flatten()
            r_neg = ref_mod(proj_mod(-half)).detach().to(device="cpu", dtype=torch.float32).flatten()
            return (r_pos - r_neg).contiguous()
    w_proj = getattr(proj_mod, "weight", None)
    if isinstance(w_proj, torch.Tensor):
        return (w_proj.detach().to(device="cpu", dtype=torch.float32) @ v32).contiguous()
    raise ValueError(
        "cannot project vector.encoder to 'h3_image': neither J_ref_h3_image nor condition_proj is available"
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
    allow_recipe_mismatch: bool = True,
    loaded_prefix: str | dict[str, Any] | None = None,
    prompt: str = "",
    video_scale: float = 1.0,
    audio_scale: float = 1.0,
    split_end_percent: float = 1.0,
    ref_vae_end_percent: float = 1.0,
    ref_vae_aug: float = 0.999,
    audio_cond_end_percent: float = 1.0,
    audio_cond_aug: float = 1.0,
    video_ref_end_percent: float = 1.0,
    guide_cond_end_percent: float = 1.0,
    words: str = "",
    against_words: str = "",
    hold_words: str = "",
    hold_alpha: float = 1.0,
    edit_pass: str = "both",
    token_focus: float = 1.0,
    extra_hold_words: Sequence[tuple[str, float]] = (),
    block_band: str = "all",
    split_backend: str = "auto_1x_kv",
) -> Any:
    """Clone `model` with `vector_name` and/or inline J-Lens `words` attached on `WrappersMP.DIFFUSION_MODEL`."""
    mod = _self_mod()
    cv_mod = mod._cv
    jl_mod = mod._j_lens
    is_h3_fn = mod._is_minimax_h3
    is_h3_img_fn = mod._is_minimax_h3_image

    dm = model.get_model_object("diffusion_model")
    if not is_h3_fn(dm):
        raise ValueError("MiniMaxH3ApplyControlVectors requires a MiniMax H3 diffusion model")
    if mode not in cv_mod.APPLY_MODES:
        raise ValueError(f"mode must be one of {cv_mod.APPLY_MODES}, got {mode!r}")
    if scale_mode not in cv_mod.SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {cv_mod.SCALE_MODES}, got {scale_mode!r}")
    if edit_pass not in STEER_EDIT_PASSES:
        raise ValueError(f"edit_pass must be one of {STEER_EDIT_PASSES}, got {edit_pass!r}")
    norm_block_band = _steer_hooks.validate_block_band(block_band)
    norm_split_backend = str(split_backend if split_backend is not None else "auto_1x_kv").strip() or "auto_1x_kv"
    if norm_split_backend not in _steer_hooks.STEER_SPLIT_BACKENDS:
        raise ValueError(
            f"split_backend must be one of {_steer_hooks.STEER_SPLIT_BACKENDS}, got {split_backend!r}"
        )
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
    acep_f = _validate_float_bounds("audio_cond_end_percent", audio_cond_end_percent, lo=0.0, hi=1.0)
    aca_f = _validate_float_bounds("audio_cond_aug", audio_cond_aug, lo=0.0, hi=1.0)
    vrep_f = _validate_float_bounds("video_ref_end_percent", video_ref_end_percent, lo=0.0, hi=1.0)
    gcep_f = _validate_float_bounds("guide_cond_end_percent", guide_cond_end_percent, lo=0.0, hi=1.0)
    is_h3_img = is_h3_img_fn(dm)
    if rva_f < 0.999 and is_h3_img:
        raise ValueError(
            "MiniMax-H3-Image hardcodes reference AdaLN timestep labels to 0.999 in ComfyUI core, "
            "so clean_level / ref_vae_aug < 0.999 would inject noise without updating the timestep "
            "label; use end_percent to release references and keep clean_level at 0.999"
        )
    if (acep_f < 1.0 or aca_f < 1.0) and is_h3_img:
        raise ValueError(
            "MiniMax-H3-Image has no audio stream, so audio_end_percent < 1.0 and "
            "audio_clean_level < 1.0 cannot be applied; keep both at 1.0"
        )
    if vrep_f < 1.0 and is_h3_img:
        raise ValueError(
            "MiniMax-H3-Image only accepts image references (<Picture N>), so "
            "video_end_percent < 1.0 cannot be applied; use end_percent to release "
            "image references and keep video_end_percent at 1.0"
        )
    if gcep_f < 1.0 and is_h3_img:
        raise ValueError(
            "MiniMax-H3-Image has no timeline guide keyframes, so "
            "guide_end_percent < 1.0 cannot be applied; use end_percent to release "
            "image references and keep guide_end_percent at 1.0"
        )
    ha_f = _validate_float_bounds("hold_alpha", hold_alpha, lo=0.0, hi=2.0)
    tf_f = _validate_float_bounds("token_focus", token_focus, lo=0.05, hi=1.0)
    def_shift_v = float(getattr(dm, "sigma_shift_video", 12.0))
    def_shift_a = float(getattr(dm, "sigma_shift_audio", 3.0))

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
        if rvep_f < 1.0 or rva_f < 0.999 or acep_f < 1.0 or aca_f < 1.0 or vrep_f < 1.0 or gcep_f < 1.0:
            t_opts = m.model_options.setdefault("transformer_options", {})
            existing_none: tuple[_AppliedVectorSpec, ...] = tuple(t_opts.get(WRAPPER_KEY, ()))
            ref_spec = _AppliedVectorSpec(
                vector_name="none",
                vector=torch.zeros((cv_mod.POST_REFINER_DIM,), dtype=torch.float32),
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
                audio_cond_end_percent=acep_f,
                audio_cond_aug=aca_f,
                video_ref_end_percent=vrep_f,
                guide_cond_end_percent=gcep_f,
            )
            updated_none = existing_none + (ref_spec,)
            t_opts[WRAPPER_KEY] = updated_none
            wrappers = m.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
            wrappers[WRAPPER_KEY] = [
                _make_control_vector_wrapper(
                    updated_none,
                    is_h3_image=is_h3_img,
                    default_shift_video=def_shift_v,
                    default_shift_audio=def_shift_a,
                )
            ]
        return m

    loaded_family = cv_mod.detect_family(model)
    loaded_recipe = mod.extract_model_recipe(model)
    synth_family_arg = (
        "fl2va_base"
        if (str(loaded_family) == "fl2va" and not loaded_recipe.get("loras"))
        else str(loaded_family)
    )
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
        vec_path = mod.resolve_vector_file_path(
            vector_name,
            preferred_family=loaded_family,
            prefer_lora=bool(loaded_recipe.get("loras")),
        )
        tensors, metadata = cv_mod.load_vector(vec_path)
        if "vector.post_refiner" not in tensors:
            raise ValueError(
                f"control vector {vec_path.name} has no 'vector.post_refiner' tensor"
            )

        vec_family = str(metadata.get("family", ""))
        erasure_vr: torch.Tensor | None = None
        erasure_diag: torch.Tensor | None = None
        erasure_mu_d: torch.Tensor | None = None
        erasure_mu_a: torch.Tensor | None = None
        if (
            loaded_family == "h3_image"
            and vec_family in ("fl2va", "ref2va", "fasth3")
            and "vector.encoder" in tensors
        ):
            cv_mod._validate_metadata_dict(metadata)
            v_enc = tensors["vector.encoder"].detach().to(device="cpu", dtype=torch.float32).flatten()
            v_enc_nrm = float(torch.linalg.vector_norm(v_enc.to(dtype=torch.float64)).item())
            if v_enc_nrm <= 1e-12:
                raise ValueError(f"control vector {vec_path.name} has zero 'vector.encoder' norm")
            u_enc = v_enc / v_enc_nrm
            pts_enc = (metadata.get("points") or {}).get("encoder")
            enc_raw_norm = (
                float(pts_enc["raw_norm"])
                if isinstance(pts_enc, Mapping) and float(pts_enc.get("raw_norm", 0.0) or 0.0) > 1e-9
                else float(cv_mod.NORM_MATCHED_TARGET)
            )
            vec_pr = _project_encoder_vector_to_h3_image(u_enc * enc_raw_norm, dm)
            proj_msg = (
                f"projected {vec_path.name} (family={vec_family!r}) from 5120-d vector.encoder "
                f"(raw_norm={enc_raw_norm:.2f}) into 5376-d 'h3_image' post-refiner space "
                f"(||v_pr||={float(torch.linalg.vector_norm(vec_pr.to(dtype=torch.float64)).item()):.2f})"
            )
            advisories = [proj_msg]
            logger.info("[h3] %s", proj_msg)
            if mode == "add_norm_matched_erasure" and "erasure.vr" in tensors:
                skip_msg = (
                    f"skipping {vec_family!r} post-refiner erasure.vr on cross-family 'h3_image' "
                    f"vector.encoder projection for {vec_path.name}; applying projected 1D direction"
                )
                advisories.append(skip_msg)
                logger.info("[h3] %s", skip_msg)
        else:
            recipe_to_check = {
                "checkpoint": loaded_recipe["checkpoint"],
                "loras": loaded_recipe["loras"],
                "encoder": metadata["recipe"].get("encoder", loaded_recipe["encoder"]),
            }
            advisories = cv_mod.check_vector_compatibility(
                metadata,
                loaded_family=loaded_family,
                loaded_recipe=recipe_to_check,
                loaded_prefix=loaded_prefix,
                allow_recipe_mismatch=allow_recipe_mismatch,
            )
            for msg in advisories:
                logger.warning("[h3] %s", msg)
            vec_pr = tensors["vector.post_refiner"]
            if "erasure.vr" in tensors and "erasure.diag" in tensors:
                erasure_vr = tensors["erasure.vr"].detach().to(device="cpu", dtype=torch.float32)
                erasure_diag = tensors["erasure.diag"].detach().to(device="cpu", dtype=torch.float32)
                if "erasure.mu_d" in tensors:
                    erasure_mu_d = tensors["erasure.mu_d"].detach().to(device="cpu", dtype=torch.float32)
                if "erasure.mu_a" in tensors:
                    erasure_mu_a = tensors["erasure.mu_a"].detach().to(device="cpu", dtype=torch.float32)

        erasure_holds: tuple[tuple[torch.Tensor, float], ...] = ()
        word_hold_pairs: list[tuple[str, float]] = []
        if hold_clean and ha_f > 0.0:
            word_hold_pairs.append((hold_clean, ha_f))
        for ew_raw, ea_raw in (extra_hold_words or ()):
            ew_clean = str(ew_raw or "").strip()
            ea_f = _validate_float_bounds("hold_alpha", ea_raw, lo=0.0, hi=2.0)
            if ew_clean and ea_f > 0.0:
                word_hold_pairs.append((ew_clean, ea_f))
        if word_hold_pairs:
            synth_fam = jl_mod._canonical_synth_family(synth_family_arg)
            orig_nrm = float(torch.linalg.vector_norm(vec_pr.to(dtype=torch.float64)).item())
            vec_held = vec_pr
            synth_holds: list[tuple[torch.Tensor, float]] = []
            for wh_text, wh_alpha in word_hold_pairs:
                hold_synth = jl_mod.synthesize_from_words(
                    wh_text,
                    family=synth_fam,
                )
                hold_vec_pr = hold_synth["vector.post_refiner"]
                vec_held = jl_mod.apply_soft_holds(
                    vec_held,
                    [hold_vec_pr],
                    alpha=wh_alpha,
                )
                synth_holds.append((hold_vec_pr.detach().to(device="cpu", dtype=torch.float32), wh_alpha))
            res_nrm = float(torch.linalg.vector_norm(vec_held.to(dtype=torch.float64)).item())
            if res_nrm <= 1e-3 * max(1.0, orig_nrm):
                raise ValueError("hold_words completely cancelled control vector (residual norm is zero)")
            if mode == "add_norm_matched_erasure" and erasure_vr is not None and float(strength) > 0.0:
                erasure_holds = tuple(synth_holds)
            else:
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
            audio_cond_end_percent=acep_f,
            audio_cond_aug=aca_f,
            video_ref_end_percent=vrep_f,
            guide_cond_end_percent=gcep_f,
            edit_pass=edit_pass,
            token_focus=tf_f,
            erasure_vr=erasure_vr,
            erasure_diag=erasure_diag,
            erasure_mu_d=erasure_mu_d,
            erasure_holds=erasure_holds,
            erasure_mu_a=erasure_mu_a,
            block_band=norm_block_band,
            split_backend=norm_split_backend,
        )
        updated = updated + (new_spec,)
        logger.info(
            "[h3] control vector %s attached (family=%s, strength=%g, mode=%s, scale_mode=%s, mask=%s, window=[%.2f, %.2f), video_scale=%g, audio_scale=%g, ref_vae_end=%.2f, edit_pass=%s, token_focus=%.2f, block_band=%s, split_backend=%s)",
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
            edit_pass,
            tf_f,
            norm_block_band,
            norm_split_backend,
        )

    if words_clean:
        synth_fam = jl_mod._canonical_synth_family(synth_family_arg)
        prompt_str = str(prompt or "").strip()
        is_prompt_against = against_clean.lower().lstrip("@") in ("prompt", "prompt_subject")
        synth = jl_mod.synthesize_from_words(
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
        w_erasure_vr = synth.get("erasure.vr")
        w_erasure_diag = synth.get("erasure.diag")
        w_erasure_mu_d = synth.get("erasure.mu_d")
        w_erasure_holds: tuple[tuple[torch.Tensor, float], ...] = ()
        if extra_hold_words:
            w_synth_holds: list[tuple[torch.Tensor, float]] = []
            for ew_raw, ea_raw in extra_hold_words:
                ew_clean = str(ew_raw or "").strip()
                ea_f = _validate_float_bounds("hold_alpha", ea_raw, lo=0.0, hi=2.0)
                if ew_clean and ea_f > 0.0:
                    hs = jl_mod.synthesize_from_words(ew_clean, family=synth_fam)
                    w_synth_holds.append((hs["vector.post_refiner"].detach().to(device="cpu", dtype=torch.float32), ea_f))
            if w_synth_holds:
                w_erasure_holds = tuple(w_synth_holds)
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
            audio_cond_end_percent=acep_f,
            audio_cond_aug=aca_f,
            video_ref_end_percent=vrep_f,
            guide_cond_end_percent=gcep_f,
            edit_pass=edit_pass,
            token_focus=tf_f,
            erasure_vr=w_erasure_vr,
            erasure_diag=w_erasure_diag,
            erasure_mu_d=w_erasure_mu_d,
            erasure_holds=w_erasure_holds,
            block_band=norm_block_band,
            split_backend=norm_split_backend,
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
                cv_mod.is_additive_mode(_effective_entry_mode(prev))
                and float(prev.strength) != 0.0
                and (prev.mask, prev.prompt) == (sp_chk.mask, sp_chk.prompt)
                for prev in prior
            )
            if not has_matching_add:
                prior_add_masks = sorted(
                    {
                        prev.mask
                        for prev in prior
                        if cv_mod.is_additive_mode(_effective_entry_mode(prev))
                        and float(prev.strength) != 0.0
                    }
                )
                if not prior_add_masks:
                    logger.warning(
                        "[h3] control vector %r is in mode 'hold' with no additive control vector before it in "
                        "the chain: it changes nothing. A hold removes its direction from what earlier nodes add; "
                        "to take a direction out of the prompt itself use mode 'project_out', or hold_words on "
                        "the node that adds.",
                        sp_chk.vector_name,
                    )
                if prior_add_masks:
                    raise ValueError(
                        f"mode='hold' on {sp_chk.vector_name!r} uses mask={sp_chk.mask!r}, which does not match "
                        f"any preceding additive control vector's mask ({prior_add_masks}); "
                        "a 'hold' node must share the same mask as the additive node(s) it projects out of"
                    )

    t_opts[WRAPPER_KEY] = updated
    if norm_block_band != "all" and not _steer_hooks.is_hookable_dit(dm):
        raise ValueError(
            f"block_band={norm_block_band!r} requires a MiniMax H3 model with hookable DiT blocks"
        )
    if is_h3_img and norm_block_band != "all":
        _steer_hooks.parse_block_band(norm_block_band, len(dm.blocks))

    needs_hooks = _specs_need_attention_hooks(updated, dm, is_h3_image=is_h3_img)
    ws_for_wrapper = None
    if needs_hooks:
        dm_wrappers = m.wrappers.get(WrappersMP.DIFFUSION_MODEL, {})
        dit_replaces = t_opts.get("patches_replace", {}).get("dit", {})
        if (
            hasattr(model, "_h3_frozen_video_cache")
            or hasattr(m, "_h3_frozen_video_cache")
            or any("frozen_video_cache" in str(wk) for wk in dm_wrappers)
        ):
            raise ValueError(
                "MiniMaxH3Steer block_band / 1x stream split cannot be combined with MiniMaxH3FrozenVideoCache "
                "(cached steps skip double_block hooks)"
            )
        for b_patch in dit_replaces.values():
            b_mod = getattr(b_patch, "__module__", "") or ""
            b_qname = getattr(b_patch, "__qualname__", "") or ""
            if "nodes_sparse_attention" in b_mod or "make_h3_block_patch" in b_qname:
                raise ValueError(
                    "MiniMaxH3Steer block_band / 1x stream split requires dense optimized_attention or "
                    "MiniMaxH3Sol, not core chunked BlockSparseAttention"
                )
        ws_for_wrapper = _steer_hooks.H3SteerStepWorkspace()
        m.add_callback_with_key(
            CallbacksMP.ON_CLEANUP,
            f"{WRAPPER_KEY}_workspace",
            lambda *args, _ws=ws_for_wrapper, **kwargs: _ws.clear(),
        )
        for b_idx in range(len(dm.blocks)):
            existing = dit_replaces.get(("double_block", b_idx))
            if not _steer_hooks.has_steer_block_patch(existing):
                m.set_model_patch_replace(
                    _steer_hooks.H3SteerBlockPatch(b_idx, previous=existing),
                    "dit",
                    "double_block",
                    b_idx,
                )

    wrappers = m.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})
    wrappers[WRAPPER_KEY] = [
        _make_control_vector_wrapper(
            updated,
            is_h3_image=is_h3_img,
            default_shift_video=def_shift_v,
            default_shift_audio=def_shift_a,
            dm_ref=weakref.ref(dm),
            workspace=ws_for_wrapper,
        )
    ]
    return m


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
    scene_prompts: Sequence[str] | None = None,
    include_encoder: bool = True,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    made_by: str = "MiniMaxH3MakeMediaControlVector/1",
    loras: Sequence[dict[str, Any]] | None = None,
    encoder_path_override: str | Path | None = None,
    refiner_device: str | torch.device | None = None,
    min_erasure_pairs: int = _cv.MIN_ERASURE_PAIRS,
) -> tuple[Path, dict[str, Any]]:
    """Build and save a `.safetensors` control vector from contrasting `<Picture 1>` reference images."""
    mod = _self_mod()
    cv_mod = mod._cv
    jl_mod = mod._j_lens
    if pooling not in cv_mod.VALID_MEDIA_POOLING:
        raise ValueError(
            f"pooling must be one of {cv_mod.VALID_MEDIA_POOLING}, got {pooling!r}"
        )
    stem = str(output_name or "media_contrast").strip()
    if not cv_mod.NAME_RE.match(stem):
        raise ValueError(f"output_name must match [a-z0-9_]+, got {stem!r}")

    matched_a, matched_b, (target_w, target_h) = mod.match_media_pair_geometry(
        image_a, image_b, short_edge=short_edge
    )
    n_frames = int(matched_a.shape[0])
    if scene_prompts is not None:
        if isinstance(scene_prompts, (str, bytes)):
            raise ValueError(
                "scene_prompts must be a sequence of per-pair prompt strings, not a bare string"
            )
        prompts = [str(p or "").strip() for p in scene_prompts]
        if len(prompts) != n_frames:
            raise ValueError(
                f"scene_prompts length ({len(prompts)}) must match media pair count ({n_frames})"
            )
        if any(not p for p in prompts):
            raise ValueError("scene_prompts entries must all be non-empty strings")
    else:
        custom_prompt = str(scene_prompt or "").strip()
        if not custom_prompt:
            raise ValueError(
                "scene_prompt is required when building a media-contrast control vector; "
                "provide a scene prompt matching your target setup"
            )
        prompts = [custom_prompt for _ in range(n_frames)]
    frame_indices = list(range(n_frames))

    family = cv_mod.detect_family(model)
    if isinstance(model, (str, Path)):
        recipe = mod.extract_model_recipe(
            None,
            clip,
            dit_path_override=model,
            encoder_path_override=encoder_path_override,
            loras_override=loras,
        )
        refiner_run = mod.build_refiner_runner(
            model, device=refiner_device or "cpu", loras=loras
        )
    else:
        recipe = mod.extract_model_recipe(
            model,
            clip,
            encoder_path_override=encoder_path_override,
            loras_override=loras,
        )
        refiner_run = mod.build_refiner_runner(
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
    pair_hash.update(cv_mod._tensor_bf16_bytes(matched_a))
    pair_hash.update(cv_mod._tensor_bf16_bytes(matched_b))
    for pr in prompts:
        pair_hash.update(pr.encode("utf-8"))

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

    media_a_spans: list[torch.Tensor] = []
    media_b_spans: list[torch.Tensor] = []
    for idx, f_idx, enc_a, tags_a, ids_a, dec_a, enc_b, tags_b, ids_b, dec_b in encoded_media_pairs:
        ref_a = refiner_run(enc_a)
        ref_b = refiner_run(enc_b)

        enc_censuses.append(cv_mod.row_norm_census(enc_a, tags_a))
        enc_censuses.append(cv_mod.row_norm_census(enc_b, tags_b))
        ref_censuses.append(cv_mod.row_norm_census(ref_a, tags_a))
        ref_censuses.append(cv_mod.row_norm_census(ref_b, tags_b))

        spans_a = _spans.spans(ids_a, tags_a.tolist(), dec_a)
        spans_b = _spans.spans(ids_b, tags_b.tolist(), dec_b)
        mask_a = _media_pooling_mask(spans_a, tags_a, pooling)
        mask_b = _media_pooling_mask(spans_b, tags_b, pooling)

        media_a_spans.append(ref_a[mask_a].detach().to(device="cpu", dtype=torch.float32))
        media_b_spans.append(ref_b[mask_b].detach().to(device="cpu", dtype=torch.float32))
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
    final_vec = cv_mod.mean_difference(a_mat, b_mat, w_vec)
    tensors_to_save: dict[str, torch.Tensor] = {"vector.post_refiner": final_vec}
    min_er_media = max(1, int(min_erasure_pairs))
    if len(media_a_spans) < min_er_media:
        media_erasure = None
        logger.warning(
            "[h3] skipping Pole-A erasure on %r media build (n_pairs=%d < MIN_ERASURE_PAIRS=%d; "
            "provide >=%d contrasting frame pairs to build well-conditioned erasure.vr; saving 1D vector)",
            stem,
            len(media_a_spans),
            min_er_media,
            min_er_media,
        )
    elif pooling == "vision_tokens":
        media_erasure = cv_mod.build_spatial_vision_erasure(
            media_a_spans, media_b_spans, weights=weights_list
        )
        if media_erasure is not None:
            (
                tensors_to_save["erasure.mu_d"],
                tensors_to_save["erasure.vr"],
                tensors_to_save["erasure.diag"],
                tensors_to_save["erasure.mu_a"],
            ) = media_erasure
    else:
        media_erasure = cv_mod.build_pole_a_erasure(
            media_a_spans, media_b_spans, weights=weights_list
        )
        if media_erasure is not None:
            (
                tensors_to_save["erasure.mu_d"],
                tensors_to_save["erasure.vr"],
                tensors_to_save["erasure.diag"],
            ) = media_erasure

    gen = torch.Generator(device="cpu").manual_seed(0)
    q_report = cv_mod.quality_report(a_mat, b_mat, w_vec, generator=gen)
    if jl_mod is not None and jl_mod.bundle_available():
        try:
            jl_media_readout = jl_mod.decode_vector(
                final_vec, space="post_refiner", family=family, top_k=12
            )
            if media_erasure is not None and hasattr(jl_mod, "decode_erasure_payload"):
                jl_media_readout["erasure_readout"] = jl_mod.decode_erasure_payload(
                    final_vec,
                    media_erasure[1],
                    media_erasure[2],
                    erasure_mu_d=media_erasure[0],
                    v1d_readout=jl_media_readout,
                    family=family,
                )
            q_report["j_lens_readout"] = jl_media_readout
            logger.info(
                "[h3] built media control vector %r (%s): j_lens (+) %s | (-) %s",
                stem,
                family,
                jl_media_readout.get("pos_summary", ""),
                jl_media_readout.get("neg_summary", ""),
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
        enc_raw_vec = cv_mod.mean_difference(enc_a_mat, enc_b_mat, w_vec)
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
        "format": cv_mod.FORMAT_VERSION,
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
    out_filename = cv_mod.default_vector_filename(
        stem,
        family,
        pooling=pooling,
        prefix="none",
        apply_holds=True,
        n_holds=0,
        loras=recipe.get("loras"),
    )
    out_path = cv_mod.save_vector(
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
                H3ControlVector.Output(display_name="control_vector"),
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
        mod = _self_mod()
        dm = model.get_model_object("diffusion_model")
        if not mod._is_minimax_h3(dm):
            raise ValueError("MiniMaxH3MakeMediaControlVector requires a MiniMax H3 diffusion model")
        out_path, report = mod.make_control_vector_from_media_pairs(
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
        return io.NodeOutput(
            out_path.name,
            json.dumps(report, indent=2, sort_keys=True),
            H3VectorSpec(kind="file", vector_name=out_path.name),
        )


def _vector_file_fingerprint(raw_name: str) -> Any:
    raw = str(raw_name or "").strip()
    if not raw or raw == "none":
        return "none"
    resolved_stats: list[tuple[str, int, int]] = []
    for pref_fam in (None, "fl2va", "ref2va", "h3_image"):
        try:
            full = _self_mod().resolve_vector_file_path(raw, preferred_family=pref_fam, log_swap=False)
            st = os.stat(full)
            entry = (str(full), st.st_mtime_ns, st.st_size)
            if entry not in resolved_stats:
                resolved_stats.append(entry)
        except Exception:
            continue
    if len(resolved_stats) == 1:
        return resolved_stats[0]
    return tuple(resolved_stats) if resolved_stats else raw


def _flatten_hold_specs(
    upstream_holds: Sequence[tuple[H3VectorSpec, float]],
    extra_hold: H3VectorSpec | None,
    extra_alpha: float,
) -> list[tuple[H3VectorSpec, float]]:
    out: list[tuple[H3VectorSpec, float]] = []

    def _visit(spec: H3VectorSpec, alpha: float) -> None:
        base_spec = dataclasses.replace(spec, holds=())
        out.append((base_spec, float(alpha)))
        for sub_spec, sub_alpha in spec.holds:
            _visit(sub_spec, float(sub_alpha))

    for h_spec, h_alpha in upstream_holds:
        _visit(h_spec, float(h_alpha))
    if extra_hold is not None:
        _visit(extra_hold, float(extra_alpha))
    return out


def _attach_h3_vector_spec(
    model: Any,
    spec: H3VectorSpec,
    *,
    strength: float,
    mode: str,
    mask: str,
    start_percent: float,
    end_percent: float,
    allow_recipe_mismatch: bool,
    video_scale: float,
    audio_scale: float,
    hold_words_override: str | None = None,
    hold_alpha_override: float | None = None,
    edit_pass: str = "both",
    token_focus: float = 1.0,
    extra_hold_words: Sequence[tuple[str, float]] = (),
    block_band: str = "all",
    split_backend: str = "auto_1x_kv",
) -> Any:
    eff_hold_words = spec.hold_words if hold_words_override is None else hold_words_override
    eff_hold_alpha = spec.hold_alpha if hold_alpha_override is None else hold_alpha_override
    if spec.kind == "file":
        return _self_mod().attach_control_vector(
            model,
            spec.vector_name,
            strength=float(strength),
            mode=mode,
            mask=mask,
            start_percent=float(start_percent),
            end_percent=float(end_percent),
            allow_recipe_mismatch=bool(allow_recipe_mismatch),
            video_scale=float(video_scale),
            audio_scale=float(audio_scale),
            split_end_percent=1.0,
            hold_words=str(eff_hold_words or ""),
            hold_alpha=float(eff_hold_alpha),
            edit_pass=edit_pass,
            token_focus=float(token_focus),
            extra_hold_words=extra_hold_words,
            block_band=block_band,
            split_backend=split_backend,
        )
    if spec.kind == "words":
        return _self_mod().attach_control_vector(
            model,
            "none",
            strength=float(strength),
            mode=mode,
            mask=mask,
            start_percent=float(start_percent),
            end_percent=float(end_percent),
            allow_recipe_mismatch=bool(allow_recipe_mismatch),
            video_scale=float(video_scale),
            audio_scale=float(audio_scale),
            split_end_percent=1.0,
            words=str(spec.words or ""),
            against_words=str(spec.against_words or ""),
            hold_words=str(eff_hold_words or ""),
            hold_alpha=float(eff_hold_alpha),
            edit_pass=edit_pass,
            token_focus=float(token_focus),
            extra_hold_words=extra_hold_words,
            block_band=block_band,
            split_backend=split_backend,
        )
    raise ValueError(f"unsupported H3VectorSpec.kind={spec.kind!r}")


class MiniMaxH3WordVector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3WordVector",
            display_name="MiniMax H3 Word Vector (J-Lens)",
            category="model/conditioning/minimax",
            description=(
                "Synthesizes a post-refiner steering direction in <2 ms on CPU from single-token "
                "concept words via J-Lens, with zero .safetensors build required. Wire into "
                "MiniMaxH3Steer.custom_vector to steer, or MiniMaxH3Steer.hold_vector to protect "
                "a concept while another vector pushes."
            ),
            inputs=[
                io.String.Input(
                    "words",
                    default="",
                    tooltip=(
                        "Comma-separated positive single-token words (for example "
                        "'blizzard, snowdrift, freezing, snow' or 'cyberpunk, neon'). "
                        "Auto-matches the loaded model family (fl2va / ref2va) inside MiniMaxH3Steer."
                    ),
                ),
                io.String.Input(
                    "against_words",
                    default="",
                    optional=True,
                    tooltip=(
                        "Optional negative words to subtract. Leave empty for neutral baseline, "
                        "pass '@prompt_subject' (or '@prompt') to auto-extract and subtract the "
                        "live prompt's medium and subject on Step 0, or pass '@replace_human'."
                    ),
                ),
                io.String.Input(
                    "hold_words",
                    default="",
                    optional=True,
                    tooltip=(
                        "Optional comma-separated nuisance words to project out of the synthesized "
                        "direction on CPU (<2 ms)."
                    ),
                ),
                io.Float.Input(
                    "hold_alpha",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    optional=True,
                    tooltip=(
                        "Projection attenuation weight in [0.0, 2.0] for hold_words and any wired "
                        "hold_vector (default 1.0)."
                    ),
                ),
                H3ControlVector.Input(
                    "hold_vector",
                    optional=True,
                    tooltip=(
                        "Optional control vector or word vector to project out alongside this word "
                        "vector when applied in MiniMaxH3Steer."
                    ),
                ),
            ],
            outputs=[H3ControlVector.Output(display_name="control_vector")],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        words="",
        against_words="",
        hold_words="",
        hold_alpha=1.0,
        **kwargs,
    ):
        return (
            "words",
            str(words or "").strip(),
            str(against_words or "").strip(),
            str(hold_words or "").strip(),
            float(hold_alpha),
        )

    @classmethod
    def validate_inputs(
        cls,
        words="",
        against_words="",
        hold_words="",
        hold_alpha=1.0,
        **kwargs,
    ) -> bool | str:
        w_str = str(words or "").strip()
        if not w_str:
            return "MiniMaxH3WordVector requires non-empty 'words'"
        jl_mod = _self_mod()._j_lens
        try:
            ha_val = _validate_float_bounds("hold_alpha", hold_alpha, lo=0.0, hi=2.0)
            if jl_mod.bundle_available():
                jl_mod.synthesize_from_words(
                    w_str,
                    str(against_words or "").strip() or None,
                    family="fl2va",
                    hold_words=str(hold_words or "").strip() or None,
                    alpha=ha_val,
                )
        except Exception as exc:
            return str(exc)
        return True

    @classmethod
    def execute(
        cls,
        words="",
        against_words="",
        hold_words="",
        hold_alpha=1.0,
        hold_vector=None,
    ) -> io.NodeOutput:
        w_str = str(words or "").strip()
        if not w_str:
            raise ValueError("MiniMaxH3WordVector requires non-empty 'words'")
        a_str = str(against_words or "").strip()
        h_str = str(hold_words or "").strip()
        ha_f = _validate_float_bounds("hold_alpha", hold_alpha, lo=0.0, hi=2.0)
        jl_mod = _self_mod()._j_lens
        if jl_mod.bundle_available():
            jl_mod.synthesize_from_words(
                w_str,
                a_str or None,
                family="fl2va",
                hold_words=h_str or None,
                alpha=ha_f,
            )
        if hold_vector is not None and not isinstance(hold_vector, H3VectorSpec):
            raise ValueError(
                f"hold_vector must be an H3VectorSpec, got {type(hold_vector).__name__}"
            )
        holds = ((hold_vector, ha_f),) if hold_vector is not None else ()
        return io.NodeOutput(
            H3VectorSpec(
                kind="words",
                words=w_str,
                against_words=a_str,
                hold_words=h_str,
                hold_alpha=ha_f,
                holds=holds,
            )
        )


class MiniMaxH3Steer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Steer",
            display_name="MiniMax H3 Steer",
            category="model/conditioning/minimax",
            description=(
                "Steers a MiniMax H3 model's refined prompt conditioning using a saved control "
                "vector stem (or a wired custom_vector from MiniMaxH3WordVector / "
                "MiniMaxH3LoadControlVector / MiniMaxH3MakeControlVector). Leaves tag-0 vision "
                "tokens untouched and keeps sequence length L unchanged. Chain multiple "
                "MiniMaxH3Steer nodes to stack vectors in float64."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "vector",
                    options=_vector_stem_options(include_none=True),
                    default="none",
                    tooltip=(
                        "Saved control vector stem under models/control_vectors/ (auto-resolves "
                        "fl2va / ref2va family and LoRA sibling at run time), or 'none' when "
                        "wiring custom_vector."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=-5.0,
                    max=5.0,
                    step=0.05,
                    tooltip=(
                        "Steering multiplier (1.0 = 67.5 post-refiner shift norm in 'add_norm_matched_erasure' "
                        "/ 'add_norm_matched', one mean-pair-difference in 'add', or full projection removal "
                        "from prompt rows in 'project_out'; 0.0 is an exact no-op)."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=list(STEER_MODES),
                    default=_rules.STEER_DEFAULTS["mode"],
                    optional=True,
                    tooltip=(
                        "'add_norm_matched_erasure' (default) applies clamped prompt-centered Pole-A "
                        "subspace erasure (erasure.vr/diag/mu_d, RMS-matched to |s|*67.5, disabling "
                        "erasure for s<=0) and falls back cleanly to 1D 'add_norm_matched' when "
                        "erasure tensors are absent or projected cross-family onto 'h3_image'; "
                        "'add_norm_matched' applies pure 1D shift normalized to 67.5 at s=1.0; "
                        "'add' uses raw mean-pair-difference units; 'project_out' removes "
                        "strength * ((r . v) / ||v||^2) * v from the base prompt rows."
                    ),
                ),
                io.Combo.Input(
                    "target",
                    options=list(STEER_TARGETS),
                    default="scene_only",
                    optional=True,
                    tooltip=(
                        "Preset prompt rows to steer when 'fields' below has none ticked: "
                        "'scene_only' (default: all visual prose in subject_definitions, summary, and "
                        "description while leaving tag-0 vision, retention_analysis contracts, "
                        "<d> dialogue, soundscape, and music untouched), 'description_only' "
                        "(events/actions prose), 'look_only' (subject_definitions + summary), "
                        "'soundscape_only', 'after_dialogue', 'non_vision' (all text rows), or "
                        "'vision_tokens' (tag-0 visual reference rows, for media-contrast vectors)."
                    ),
                ),
                io.MultiCombo.Input(
                    "fields",
                    options=list(_spans.PROMPT_FIELDS),
                    default=[],
                    optional=True,
                    tooltip=(
                        "Tick one or more prompt fields ('subject_definitions', 'summary', "
                        "'retention_analysis', 'description', 'overall_soundscape', "
                        "'non_diegetic_music') to steer exactly those fields and override 'target' above."
                    ),
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "Start of the active step window [start_percent, end_percent) as step_idx / N_steps."
                    ),
                ),
                io.Float.Input(
                    "end_percent",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "End of the active step window [start_percent, end_percent) as step_idx / N_steps."
                    ),
                ),
                io.Combo.Input(
                    "streams",
                    options=list(STEER_STREAMS),
                    default="both",
                    optional=True,
                    tooltip=(
                        "Which generated stream(s) receive this steer: 'both' (default, single DiT pass "
                        "per step), 'video_only' (steers video and keeps audio unsteered across all active "
                        "steps; runs 2 DiT passes per active step), or 'audio_only' (steers audio and keeps "
                        "video unsteered; runs 2 DiT passes per active step)."
                    ),
                ),
                io.String.Input(
                    "hold_words",
                    default="",
                    optional=True,
                    tooltip=(
                        "Optional comma-separated single-token nuisance words to project out of this "
                        "steer on CPU (<2 ms) via J-Lens."
                    ),
                ),
                io.Float.Input(
                    "hold_alpha",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    optional=True,
                    tooltip=(
                        "Projection attenuation weight in [0.0, 2.0] for hold_words and any wired "
                        "hold_vector (default 1.0)."
                    ),
                ),
                io.Boolean.Input(
                    "allow_recipe_mismatch",
                    default=_rules.STEER_DEFAULTS["allow_recipe_mismatch"],
                    optional=True,
                    tooltip=(
                        "When True (default), logs checkpoint or LoRA recipe differences and applies; "
                        "set False only to hard-refuse mismatched recipes."
                    ),
                ),
                H3ControlVector.Input(
                    "custom_vector",
                    optional=True,
                    tooltip=(
                        "Optional vector wire from MiniMaxH3WordVector, MiniMaxH3LoadControlVector, "
                        "MiniMaxH3MakeControlVector, or MiniMaxH3MakeMediaControlVector. Leave 'vector' "
                        "at 'none' when wired."
                    ),
                ),
                H3ControlVector.Input(
                    "hold_vector",
                    optional=True,
                    tooltip=(
                        "Optional control vector or word vector whose direction is projected out of the "
                        "accumulated additive steering shift on the same target rows (leaving the base "
                        "prompt untouched)."
                    ),
                ),
                io.Combo.Input(
                    "edit_pass",
                    options=list(STEER_EDIT_PASSES),
                    default="both",
                    optional=True,
                    tooltip=(
                        "When paired with MiniMaxH3EditGuider: 'both' (default, steers both passes so only "
                        "prompt diffs scale), 'edited_only' (steers only the edited pass so painted mask=1 "
                        "gets this vector with zero global bleed), or 'original_only' (steers only the "
                        "original anchor pass so mask=0 keeps this look in a split-world render)."
                    ),
                ),
                io.Float.Input(
                    "token_focus",
                    default=1.0,
                    min=0.05,
                    max=1.0,
                    step=0.05,
                    optional=True,
                    tooltip=(
                        "Fraction of target prompt rows to steer, ranked by |cos(row, vector)| "
                        "(1.00 = all target rows, the validated default for add_norm_matched_erasure; "
                        "values < 1.0 are experimental and starve multi-row Pole-A subspace coverage)."
                    ),
                ),
                io.Combo.Input(
                    "block_band",
                    options=list(_steer_hooks.STEER_BLOCK_BANDS),
                    default="all",
                    optional=True,
                    tooltip=(
                        "DiT block band to steer: 'all' (default, all 50 blocks via zero-overhead prelude), "
                        "'mid_late_20_49' (blocks 20..49 only, keeping blocks 0..19 base identity/pose untouched), "
                        "'late_30_49' (blocks 30..49 late finish only), or 'early_0_19' (blocks 0..19 layout only)."
                    ),
                ),
                io.Combo.Input(
                    "split_backend",
                    options=list(_steer_hooks.STEER_SPLIT_BACKENDS),
                    default="auto_1x_kv",
                    optional=True,
                    tooltip=(
                        "Backend for 'video_only' / 'audio_only' stream splits: 'auto_1x_kv' (default, single 1x "
                        "DiT pass splicing only audio queries against unsteered/steered text K/V) or "
                        "'legacy_2x_pass' (reference 2x full DiT velocity split)."
                    ),
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        vector="none",
        hold_words="",
        hold_alpha=1.0,
        **kwargs,
    ):
        vec_fp = _vector_file_fingerprint(str(vector or ""))
        h_str = str(hold_words or "").strip()
        ep_str = str(kwargs.get("edit_pass", "both") or "both").strip()
        tf_f = float(kwargs.get("token_focus", 1.0))
        bb_str = str(kwargs.get("block_band", "all") or "all").strip()
        sb_str = str(kwargs.get("split_backend", "auto_1x_kv") or "auto_1x_kv").strip()
        if bb_str == "all" and sb_str == "auto_1x_kv":
            if not h_str and ep_str == "both" and abs(tf_f - 1.0) <= 1e-9:
                return vec_fp
            if ep_str == "both" and abs(tf_f - 1.0) <= 1e-9:
                return (vec_fp, h_str, float(hold_alpha))
            if abs(tf_f - 1.0) <= 1e-9:
                return (vec_fp, h_str, float(hold_alpha), ep_str)
            return (vec_fp, h_str, float(hold_alpha), ep_str, tf_f)
        return (vec_fp, h_str, float(hold_alpha), ep_str, tf_f, bb_str, sb_str)

    @classmethod
    def validate_inputs(
        cls,
        vector="none",
        strength=1.0,
        mode=_rules.STEER_DEFAULTS["mode"],
        target="scene_only",
        fields=[],  # noqa: B006
        start_percent=0.0,
        end_percent=1.0,
        streams="both",
        hold_words="",
        hold_alpha=1.0,
        edit_pass="both",
        token_focus=1.0,
        block_band="all",
        split_backend="auto_1x_kv",
        **kwargs,
    ) -> bool | str:
        mod = _self_mod()
        jl_mod = mod._j_lens
        raw = str(vector or "").strip()
        if raw and raw != "none":
            try:
                mod.resolve_vector_file_path(raw, log_swap=False)
            except Exception as exc:
                return str(exc)
        if mode not in STEER_MODES:
            return f"mode must be one of {STEER_MODES}, got {mode!r}"
        if streams not in STEER_STREAMS:
            return f"streams must be one of {STEER_STREAMS}, got {streams!r}"
        if edit_pass not in STEER_EDIT_PASSES:
            return f"edit_pass must be one of {STEER_EDIT_PASSES}, got {edit_pass!r}"
        if fields:
            try:
                _spans.fields_mask(fields if isinstance(fields, (list, tuple)) else [fields])
            except ValueError as exc:
                return str(exc)
        elif target not in STEER_TARGETS:
            return f"target must be one of {STEER_TARGETS}, got {target!r}"
        try:
            _steer_hooks.validate_block_band(block_band)
            sb_norm = str(split_backend if split_backend is not None else "auto_1x_kv").strip() or "auto_1x_kv"
            if sb_norm not in _steer_hooks.STEER_SPLIT_BACKENDS:
                return f"split_backend must be one of {_steer_hooks.STEER_SPLIT_BACKENDS}, got {split_backend!r}"
            if strength is not None and not math.isfinite(float(strength)):
                return f"strength must be finite, got {strength!r}"
            sp_f = _validate_float_bounds("start_percent", start_percent, lo=0.0, hi=1.0) if start_percent is not None else None
            ep_f = _validate_float_bounds("end_percent", end_percent, lo=0.0, hi=1.0) if end_percent is not None else None
            if sp_f is not None and ep_f is not None and sp_f > ep_f:
                return (
                    f"start_percent and end_percent must satisfy start_percent <= end_percent, "
                    f"got start_percent={start_percent!r}, end_percent={end_percent!r}"
                )
            ha_f = _validate_float_bounds("hold_alpha", hold_alpha, lo=0.0, hi=2.0) if hold_alpha is not None else 1.0
            _tf_f = _validate_float_bounds("token_focus", token_focus, lo=0.05, hi=1.0) if token_focus is not None else 1.0
            h_str = str(hold_words or "").strip()
            if h_str and jl_mod.bundle_available():
                jl_mod.synthesize_from_words(h_str, family="fl2va")
            _ = (ha_f, _tf_f)
        except Exception as exc:
            return str(exc)
        return True

    @classmethod
    def execute(
        cls,
        model,
        vector="none",
        strength=1.0,
        mode=_rules.STEER_DEFAULTS["mode"],
        target="scene_only",
        fields=[],  # noqa: B006 — read-only default matching io.MultiCombo schema default []
        start_percent=0.0,
        end_percent=1.0,
        streams="both",
        hold_words="",
        hold_alpha=1.0,
        allow_recipe_mismatch=_rules.STEER_DEFAULTS["allow_recipe_mismatch"],
        custom_vector=None,
        hold_vector=None,
        edit_pass="both",
        token_focus=1.0,
        block_band="all",
        split_backend="auto_1x_kv",
    ) -> io.NodeOutput:
        mod = _self_mod()
        cv_mod = mod._cv
        dm = model.get_model_object("diffusion_model")
        if not mod._is_minimax_h3(dm):
            raise ValueError("MiniMaxH3Steer requires a MiniMax H3 diffusion model")
        if mode not in STEER_MODES:
            raise ValueError(f"mode must be one of {STEER_MODES}, got {mode!r}")
        if streams not in STEER_STREAMS:
            raise ValueError(f"streams must be one of {STEER_STREAMS}, got {streams!r}")
        if edit_pass not in STEER_EDIT_PASSES:
            raise ValueError(f"edit_pass must be one of {STEER_EDIT_PASSES}, got {edit_pass!r}")
        if fields:
            eff_mask = _spans.fields_mask(fields if isinstance(fields, (list, tuple)) else [fields])
            logger.info(
                "[h3] MiniMaxH3Steer: using ticked fields -> mask=%s (overriding target=%r)",
                eff_mask,
                target,
            )
        else:
            if target not in STEER_TARGETS:
                raise ValueError(f"target must be one of {STEER_TARGETS}, got {target!r}")
            eff_mask = target

        s_val = float(strength)
        if not math.isfinite(s_val):
            raise ValueError(f"strength must be finite, got {strength!r}")
        sp_f = _validate_float_bounds("start_percent", start_percent, lo=0.0, hi=1.0)
        ep_f = _validate_float_bounds("end_percent", end_percent, lo=0.0, hi=1.0)
        if sp_f > ep_f:
            raise ValueError(
                f"start_percent and end_percent must satisfy 0.0 <= start_percent <= end_percent <= 1.0, "
                f"got start_percent={start_percent!r}, end_percent={end_percent!r}"
            )
        ha_f = _validate_float_bounds("hold_alpha", hold_alpha, lo=0.0, hi=2.0)
        tf_f = _validate_float_bounds("token_focus", token_focus, lo=0.05, hi=1.0)
        node_hold_words = str(hold_words or "").strip()

        if hold_vector is not None and not isinstance(hold_vector, H3VectorSpec):
            raise ValueError(
                f"hold_vector must be an H3VectorSpec, got {type(hold_vector).__name__}"
            )

        raw_vec = str(vector or "").strip()
        if custom_vector is not None:
            if not isinstance(custom_vector, H3VectorSpec):
                raise ValueError(
                    f"custom_vector must be an H3VectorSpec, got {type(custom_vector).__name__}"
                )
            if raw_vec and raw_vec != "none":
                raise ValueError(
                    f"MiniMaxH3Steer received both custom_vector and vector={raw_vec!r}; "
                    "set vector='none' when wiring custom_vector, or chain a second "
                    "MiniMaxH3Steer node to stack two vectors"
                )
            primary_spec = custom_vector
        else:
            if not raw_vec or raw_vec == "none":
                if node_hold_words or hold_vector is not None:
                    raise ValueError(
                        "MiniMaxH3Steer: hold_words and hold_vector require vector != 'none' "
                        "or a wired custom_vector"
                    )
                return io.NodeOutput(model.clone())
            primary_spec = H3VectorSpec(kind="file", vector_name=raw_vec)

        wired_holds = _flatten_hold_specs(primary_spec.holds, hold_vector, ha_f)
        eff_hold_words = primary_spec.hold_words
        eff_hold_alpha = primary_spec.hold_alpha
        extra_hw: list[tuple[str, float]] = []
        if node_hold_words:
            if not primary_spec.hold_words:
                eff_hold_words = node_hold_words
                eff_hold_alpha = ha_f
            elif abs(float(primary_spec.hold_alpha) - ha_f) <= 1e-9 or mode == "project_out":
                eff_hold_words = f"{primary_spec.hold_words}, {node_hold_words}"
            elif mode == "add_norm_matched_erasure" and primary_spec.kind == "file":
                extra_hw.append((node_hold_words, ha_f))
            else:
                wired_holds.append((H3VectorSpec(kind="words", words=node_hold_words), ha_f))

        if not cv_mod.is_additive_mode(mode) and wired_holds:
            raise ValueError(
                "MiniMaxH3Steer: hold_vector only applies to additive modes "
                "('add_norm_matched', 'add_norm_matched_erasure', or 'add'), not mode='project_out' "
                "(use hold_words to orthogonalize the vector before removing it from the prompt)"
            )

        if streams == "video_only":
            v_scale, a_scale = 1.0, 0.0
        elif streams == "audio_only":
            v_scale, a_scale = 0.0, 1.0
        else:
            v_scale, a_scale = 1.0, 1.0

        m = _attach_h3_vector_spec(
            model,
            primary_spec,
            strength=s_val,
            mode=mode,
            mask=eff_mask,
            start_percent=sp_f,
            end_percent=ep_f,
            allow_recipe_mismatch=bool(allow_recipe_mismatch),
            video_scale=v_scale,
            audio_scale=a_scale,
            hold_words_override=eff_hold_words,
            hold_alpha_override=eff_hold_alpha,
            edit_pass=edit_pass,
            token_focus=tf_f,
            extra_hold_words=tuple(extra_hw),
            block_band=block_band,
            split_backend=split_backend,
        )
        for h_spec, h_alpha in wired_holds:
            eff_h_strength = 0.0 if s_val == 0.0 else float(h_alpha)
            m = _attach_h3_vector_spec(
                m,
                h_spec,
                strength=eff_h_strength,
                mode="hold",
                mask=eff_mask,
                start_percent=sp_f,
                end_percent=ep_f,
                allow_recipe_mismatch=bool(allow_recipe_mismatch),
                video_scale=v_scale,
                audio_scale=a_scale,
                edit_pass=edit_pass,
                token_focus=tf_f,
                block_band=block_band,
                split_backend=split_backend,
            )
        return io.NodeOutput(m)


class MiniMaxH3ReleaseReferences(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReleaseReferences",
            display_name="MiniMax H3 Release References",
            category="model/conditioning/minimax",
            description=(
                "Controls when and how cleanly visual reference, keyframe, or guide-latent "
                "conditioning rows ('ref_img' / 'cond') and timeline audio-guide islands "
                "('cond_audio') hold the render. Independent of control vectors: works on "
                "plain prompts, MiniMaxH3EditGuider, and MiniMaxH3Steer alike."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "end_percent",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Fraction of steps [0.0, end_percent) where clean 3D Video VAE visual "
                        "conditioning rows ('ref_img' / 'cond') remain packed in the DiT sequence "
                        "(default 1.00 keeps them on every step). Lowering to 0.35-0.50 lets early "
                        "steps lock identity and composition from <Picture 1> while freeing later "
                        "steps to change wardrobe, props, or environment without dropping speaker/audio "
                        "references or shifting 3D RoPE."
                    ),
                ),
                io.Float.Input(
                    "clean_level",
                    default=0.999,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    optional=True,
                    tooltip=(
                        "Cleanliness share of the reference picture on the steps where it is held "
                        "(core's visual_cond_noise_aug; default 0.999 is an almost clean picture). "
                        "Lower values mix noise into the visual condition rows, clamped at "
                        "max(clean_level, stream_time) so condition rows are never noisier than "
                        "their timestep label. Across chained nodes, the lowest end_percent and "
                        "clean_level win."
                    ),
                ),
                io.Float.Input(
                    "audio_end_percent",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "Fraction of steps [0.0, audio_end_percent) where timeline audio-guide rows "
                        "('cond_audio' from MiniMaxH3GuideLatent) remain packed in the DiT sequence "
                        "(default 1.00 keeps them on every step; <Audio N> speaker/reference rows "
                        "'ref_audio' are always preserved). Lowering to 0.50 on 6-step Pika LCM "
                        "releases short foley islands after step 2 (before shift_audio=3.0 reaches "
                        "sigma_a=0.50) so they lock visual timing without muting inter-event ambience."
                    ),
                ),
                io.Float.Input(
                    "audio_clean_level",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    optional=True,
                    tooltip=(
                        "Cleanliness share of audio conditioning rows on the steps where they are held "
                        "(core's audio_cond_noise_aug; default 1.0 matches AUDIO_COND_TIMESTEP). "
                        "Lower values mix noise into audio condition rows, clamped at "
                        "max(audio_clean_level, audio_stream_time) under shift_video/shift_audio so "
                        "rows are never noisier than their seg_t['cond_audio'] label. Core applies "
                        "audio_cond_noise_aug to both guide cond_audio and <Audio N> ref_audio rows, "
                        "so leave at 1.0 when keeping speaker voice references clean."
                    ),
                ),
                io.Float.Input(
                    "video_end_percent",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "Fraction of steps [0.0, min(end_percent, video_end_percent)) where <Video N> "
                        "visual reference rows ('ref_img' from kind in ('video', 'video_audio')) remain "
                        "packed in the DiT sequence (default 1.00 keeps them through end_percent). "
                        "Setting end_percent=1.0 and video_end_percent=0.50 holds <Picture N> subject "
                        "identity on every step while releasing synthetic or motion-transfer <Video N> "
                        "latents halfway through so H3 animates natural limb gait on later steps without "
                        "shifting 3D RoPE positions."
                    ),
                ),
                io.Float.Input(
                    "guide_end_percent",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "Fraction of steps [0.0, min(end_percent, guide_end_percent)) where mid-clip "
                        "visual guide rows ('cond' from MiniMaxH3GuideLatent strictly between frame 0 "
                        "and frame total_frames - 1) remain packed in the DiT sequence (default 1.00 "
                        "keeps them through end_percent). First_frame (0), last_frame (total_frames - 1), "
                        "<Picture N> / <Video N> refs, and timeline audio guides remain untouched, "
                        "allowing coarse mid-clip layout/pose anchoring early without late-step freeze."
                    ),
                ),
                H3EditPlan.Input(
                    "edit_plan",
                    optional=True,
                    tooltip=(
                        "Optional 4-axis EditPlan from MiniMaxH3AutoEditPrompt. Lowers end_percent to "
                        "min(end_percent, edit_plan.ref_end_percent) (for example 0.33 on silhouette-"
                        "deforming edits) unless edit_plan.manual_widgets is True."
                    ),
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def validate_inputs(
        cls,
        end_percent=1.0,
        clean_level=0.999,
        audio_end_percent=1.0,
        audio_clean_level=1.0,
        video_end_percent=1.0,
        guide_end_percent=1.0,
        **kwargs,
    ) -> bool | str:
        try:
            if end_percent is not None:
                _validate_float_bounds("end_percent", end_percent, lo=0.0, hi=1.0)
            if clean_level is not None:
                _validate_float_bounds("clean_level", clean_level, lo=0.0, hi=1.0)
            if audio_end_percent is not None:
                _validate_float_bounds("audio_end_percent", audio_end_percent, lo=0.0, hi=1.0)
            if audio_clean_level is not None:
                _validate_float_bounds("audio_clean_level", audio_clean_level, lo=0.0, hi=1.0)
            if video_end_percent is not None:
                _validate_float_bounds("video_end_percent", video_end_percent, lo=0.0, hi=1.0)
            if guide_end_percent is not None:
                _validate_float_bounds("guide_end_percent", guide_end_percent, lo=0.0, hi=1.0)
        except Exception as exc:
            return str(exc)
        return True

    @classmethod
    def execute(
        cls,
        model,
        end_percent=1.0,
        clean_level=0.999,
        audio_end_percent=1.0,
        audio_clean_level=1.0,
        video_end_percent=1.0,
        guide_end_percent=1.0,
        edit_plan=None,
    ) -> io.NodeOutput:
        mod = _self_mod()
        dm = model.get_model_object("diffusion_model")
        if not mod._is_minimax_h3(dm):
            raise ValueError("MiniMaxH3ReleaseReferences requires a MiniMax H3 diffusion model")
        ep_f = _validate_float_bounds("end_percent", end_percent, lo=0.0, hi=1.0)
        vep_f = _validate_float_bounds("video_end_percent", video_end_percent, lo=0.0, hi=1.0)
        if edit_plan is not None:
            is_map = isinstance(edit_plan, Mapping)
            manual = bool(edit_plan.get("manual_widgets", False) if is_map else getattr(edit_plan, "manual_widgets", False))
            if not manual:
                plan_ep_raw = edit_plan.get("ref_end_percent", ep_f) if is_map else getattr(edit_plan, "ref_end_percent", ep_f)
                plan_ep = _validate_float_bounds("edit_plan.ref_end_percent", plan_ep_raw, lo=0.0, hi=1.0)
                ep_f = min(ep_f, plan_ep)
                plan_vep_raw = edit_plan.get("video_end_percent", vep_f) if is_map else getattr(edit_plan, "video_end_percent", vep_f)
                if plan_vep_raw is not None:
                    plan_vep = _validate_float_bounds("edit_plan.video_end_percent", plan_vep_raw, lo=0.0, hi=1.0)
                    vep_f = min(vep_f, plan_vep)
        cl_f = _validate_float_bounds("clean_level", clean_level, lo=0.0, hi=1.0)
        aep_f = _validate_float_bounds("audio_end_percent", audio_end_percent, lo=0.0, hi=1.0)
        acl_f = _validate_float_bounds("audio_clean_level", audio_clean_level, lo=0.0, hi=1.0)
        gep_f = _validate_float_bounds("guide_end_percent", guide_end_percent, lo=0.0, hi=1.0)
        m = mod.attach_control_vector(
            model,
            "none",
            ref_vae_end_percent=ep_f,
            ref_vae_aug=cl_f,
            audio_cond_end_percent=aep_f,
            audio_cond_aug=acl_f,
            video_ref_end_percent=vep_f,
            guide_cond_end_percent=gep_f,
        )
        return io.NodeOutput(m)


# Re-export legacy / companion nodes & helpers for Option B Zero Manifest Change
try:
    from .legacy_nodes import (  # noqa: E402, F401
        MiniMaxH3ApplyControlVectors,
        MiniMaxH3LayerNormMatch,
        MiniMaxH3LoadControlVector,
        MiniMaxH3MakeControlVector,
        make_control_vector_from_pairs,
        match_conditioning_row_norm,
    )
except ImportError:
    pass
