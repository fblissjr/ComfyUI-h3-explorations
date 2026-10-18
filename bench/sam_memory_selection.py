#!/usr/bin/env python3
"""Runtime Meta memory-selection patch and 4-arm scene measurement tool for SAM 3 / SAM 3.1.

Applies Meta's `cal_mem_score` + `frame_filter` memory selection (`mf_threshold = 0.01`)
at runtime to ComfyUI core's `SAM3Tracker` (`sam3_fp32.safetensors`) and `SAM31Tracker`
(`sam3.1_multiplex_fp32.safetensors`) via `ModelPatcher.add_object_patch` and
`CallbacksMP.ON_CLONE`, without editing any ComfyUI file.

Memory scores (`eff_iou_score`) are computed and stored per object (shape `[N_obj]`),
never averaged with `.mean()` across objects or multiplex slots, so a hidden target
tracked in a multi-object group is not admitted into memory on the strength of its
visible companions' scores.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
COMFY_ROOT = REPO_ROOT.parent.parent
for _p in (REPO_ROOT, COMFY_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import comfy.cli_args

if not torch.cuda.is_available():
    comfy.cli_args.args.cpu = True

from comfy.patcher_extension import CallbacksMP

# Marked starting value: inherited from Meta's SAM 3 / SAM 3.1 tracker default
# (`mf_threshold: float = 0.01` in `sam3_tracker_base.py` and `video_tracking_multiplex.py`).
MF_THRESHOLD: float = 0.01

CALLBACK_KEY = "h3_sam_memory_selection"
CONFIG_ATTR = "_h3_sam_memory_selection_cfg"
STATE_ATTR = "_h3_sam_memory_selection_state"

SAM31_CHECKPOINT = "sam3.1_multiplex_fp32.safetensors"
SAM3_BASE_CHECKPOINT = "sam3_fp32.safetensors"


def cal_mem_score_per_object(
    object_score_logits: torch.Tensor,
    iou_scores: torch.Tensor,
) -> torch.Tensor:
    """Compute per-object effective memory score (`shape [N_obj]`, CPU float32).

    For each object `o`:
      - If `object_score_logits[o] > 0`, `obj_score_norm[o] = 2 * sigmoid(logit[o]) - 1`
        (maps `(0, +inf)` to `(0, 1)`), else `0.0`.
      - `max_iou[o] = max_t(iou_scores[o, t])`.
      - `eff_iou_score[o] = obj_score_norm[o] * max_iou[o]`.

    Unlike Meta's reference `cal_mem_score`, this never calls `.mean()` across the batch
    or multiplex slots, preserving each object's independent visibility gate in both
    single-object (`alone`) and multi-object (`group`) tracking calls.
    """
    logits = object_score_logits.detach().float().reshape(-1)
    ious = iou_scores.detach().float()
    if ious.dim() == 1:
        ious = ious.unsqueeze(-1)
    elif ious.dim() > 2:
        ious = ious.reshape(logits.shape[0], -1)
    if ious.shape[0] != logits.shape[0]:
        raise ValueError(
            f"Object count mismatch between object_score_logits ({logits.shape}) "
            f"and iou_scores ({ious.shape})"
        )
    zeros = torch.zeros_like(logits)
    obj_score_norm = torch.where(logits > 0.0, 2.0 * torch.sigmoid(logits) - 1.0, zeros)
    max_iou = ious.max(dim=-1).values
    return (obj_score_norm * max_iou).cpu()


def frame_filter_per_object(
    frame_scores: dict[int, torch.Tensor],
    frame_idx: int,
    max_num: int,
    mf_threshold: float = MF_THRESHOLD,
    n_obj: int = 1,
    present_frames: set[int] | None = None,
) -> list[list[int]]:
    """Filter past non-conditioning frame indices per object by `eff_iou_score > mf_threshold`.

    Walks backward from `frame_idx - 1` down to `1`, collecting up to `max_num - 1`
    past frames per object whose per-object `eff_iou_score[o] > mf_threshold` in
    chronological order. Always appends `must_include = frame_idx - 1` at the end if
    it exists in `non_cond_frame_outputs` and was not already selected.
    """
    cap = max(max_num - 1, 0)
    valid_per_obj: list[list[int]] = []
    must_include = frame_idx - 1
    has_must = (
        (must_include in frame_scores)
        if present_frames is None
        else (must_include in present_frames or must_include in frame_scores)
    )

    must_scores = frame_scores.get(must_include)

    for obj_idx in range(n_obj):
        valid: list[int] = []
        if cap > 0:
            for i in range(frame_idx - 1, 0, -1):
                if present_frames is not None and i not in present_frames:
                    continue
                scores_i = frame_scores.get(i)
                if scores_i is None or obj_idx >= scores_i.numel():
                    continue
                if float(scores_i[obj_idx].item()) > mf_threshold:
                    valid.insert(0, i)
                if len(valid) >= cap:
                    break
        obj_has_must = has_must and (must_scores is None or obj_idx < must_scores.numel())
        if must_include >= 1 and must_include not in valid and obj_has_must:
            valid.append(must_include)
        valid_per_obj.append(valid)
    return valid_per_obj


class IoUTap:
    """Wraps `sam_mask_decoder.forward` to record the latest `iou_pred` and `object_score_logits`."""

    def __init__(self, orig_forward: Any) -> None:
        self.orig_forward = orig_forward
        self.last_iou_pred: torch.Tensor | None = None
        self.last_obj_score_logits: torch.Tensor | None = None

    def reset(self) -> None:
        self.last_iou_pred = None
        self.last_obj_score_logits = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self.orig_forward(*args, **kwargs)
        if isinstance(out, tuple) and len(out) >= 2:
            self.last_iou_pred = out[1].detach()
            if len(out) >= 4 and out[3] is not None:
                self.last_obj_score_logits = out[3].detach()
        return out


class MemorySelectionStep:
    """Wraps `tracker._prepare_memory_conditioned_features` with per-object memory selection."""

    def __init__(
        self,
        tracker: Any,
        orig_fn: Any,
        iou_tap: IoUTap,
        mf_threshold: float = MF_THRESHOLD,
        focus_obj_idx: int = 0,
    ) -> None:
        self.tracker = tracker
        self.orig_fn = orig_fn
        self.iou_tap = iou_tap
        self.mf_threshold = float(mf_threshold)
        self.focus_obj_idx = int(focus_obj_idx)
        self._completed_runs: list[tuple[dict[int, torch.Tensor], dict[int, list[list[int]]]]] = []
        self.reset_state(clear_runs=True)

    def reset_state(self, clear_runs: bool = True) -> None:
        if clear_runs:
            self._completed_runs = []
        elif self.frame_scores or self.selected_history:
            self._completed_runs.append(
                (dict(self.frame_scores), {k: [list(x) for x in v] for k, v in self.selected_history.items()})
            )
        self._output_dict_id: int | None = None
        self._last_frame_idx: int = -1
        # Maps frame_idx -> 1D CPU float32 tensor [N_obj] of eff_iou_score
        self.frame_scores: dict[int, torch.Tensor] = {}
        # Maps frame_idx -> shallow dict of memory tensors retained past core's 16-frame prune
        self.retained_bank: dict[int, dict[str, Any]] = {}
        # Maps frame_idx -> list[list[int]] of valid_indices per object
        self.selected_history: dict[int, list[list[int]]] = {}
        self.iou_tap.reset()

    def merged_frame_scores(self) -> dict[int, torch.Tensor]:
        """Return per-frame `eff_iou_score` across single- or multi-object `SAM3Tracker` passes."""
        if not self._completed_runs:
            return dict(self.frame_scores)
        runs = list(self._completed_runs)
        if self.frame_scores or self.selected_history:
            runs.append((self.frame_scores, self.selected_history))
        all_keys = sorted({k for r_scores, _ in runs for k in r_scores})
        return {
            k: torch.cat([r_scores[k] for r_scores, _ in runs if k in r_scores], dim=0)
            for k in all_keys
        }

    def merged_selected_history(self) -> dict[int, list[list[int]]]:
        """Return per-frame `valid_indices` across single- or multi-object `SAM3Tracker` passes."""
        if not self._completed_runs:
            return dict(self.selected_history)
        runs = list(self._completed_runs)
        if self.frame_scores or self.selected_history:
            runs.append((self.frame_scores, self.selected_history))
        all_keys = sorted({k for _, r_sel in runs for k in r_sel})
        return {
            k: [list(obj_list) for _, r_sel in runs if k in r_sel for obj_list in r_sel[k]]
            for k in all_keys
        }

    @staticmethod
    def _snapshot_entry(entry: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "maskmem_features",
            "maskmem_pos_enc",
            "image_features",
            "image_pos_enc",
            "obj_ptr",
            "object_score_logits",
            "eff_iou_score",
        )
        return {k: entry[k] for k in keys if k in entry}

    def _score_previous_frame(
        self,
        prev_idx: int,
        prev_out: dict[str, Any],
        multiplex_state: Any,
    ) -> torch.Tensor | None:
        if "eff_iou_score" in prev_out and isinstance(prev_out["eff_iou_score"], torch.Tensor):
            score = prev_out["eff_iou_score"].detach().float().cpu().reshape(-1)
            self.frame_scores[prev_idx] = score
            return score

        raw_iou = prev_out.get("ious", self.iou_tap.last_iou_pred)
        raw_logits = prev_out.get("object_score_logits", self.iou_tap.last_obj_score_logits)
        if raw_iou is None or raw_logits is None:
            return None

        iou_t = raw_iou.detach()
        logits_t = raw_logits.detach()
        if multiplex_state is not None:
            if iou_t.dim() == 3:
                iou_t = multiplex_state.demux(iou_t)
            if logits_t.dim() == 3:
                logits_t = multiplex_state.demux(logits_t)
        elif logits_t.dim() == 3 and iou_t.dim() == 3:
            # Fallback demux if multiplex_state was not passed
            n_obj = logits_t.shape[0] * logits_t.shape[1]
            iou_t = iou_t.reshape(n_obj, -1)
            logits_t = logits_t.reshape(n_obj, -1)

        score = cal_mem_score_per_object(logits_t, iou_t)
        prev_out["eff_iou_score"] = score
        self.frame_scores[prev_idx] = score
        return score

    def _lookup_frame(
        self,
        f_idx: int,
        non_cond_outputs: dict[int, dict[str, Any]],
    ) -> dict[str, Any] | None:
        out = non_cond_outputs.get(f_idx)
        if out is not None:
            return out
        return self.retained_bank.get(f_idx)

    def _build_remapped_non_cond(
        self,
        frame_idx: int,
        max_obj_ptrs: int,
        valid_per_obj: list[list[int]],
        non_cond_outputs: dict[int, dict[str, Any]],
        multiplex_state: Any,
    ) -> dict[int, dict[str, Any]]:
        n_obj = len(valid_per_obj)
        if n_obj == 0:
            return {}

        focus_obj = min(max(self.focus_obj_idx, 0), n_obj - 1)
        focus_valid = valid_per_obj[focus_obj]
        max_k = max((len(v) for v in valid_per_obj), default=0)
        max_k = min(max_k, max(max_obj_ptrs - 1, 0))

        remapped: dict[int, dict[str, Any]] = {}
        num_mux = getattr(self.tracker, "num_multiplex", 16)

        for k in range(1, max_k + 1):
            target_key = frame_idx - k
            if target_key < 0:
                break

            # Single-object or all objects selected the exact same frame at rank -k
            per_obj_src = [
                valid_per_obj[o][-k] if k <= len(valid_per_obj[o]) else None
                for o in range(n_obj)
            ]
            spatial_src = focus_valid[-k] if k <= len(focus_valid) else None

            if n_obj == 1 or multiplex_state is None or (
                all(s == spatial_src for s in per_obj_src) and spatial_src is not None
            ):
                if spatial_src is None:
                    continue
                src_out = self._lookup_frame(spatial_src, non_cond_outputs)
                if src_out is not None:
                    remapped[target_key] = src_out
                continue

            # Multi-object multiplex where objects diverged in valid_indices at rank -k:
            # Spatial memory comes from `focus_obj` (if available at rank -k), while
            # each object slot (b = o // M, m = o % M) in `obj_ptr` comes from its own
            # `valid_per_obj[o][-k]`.
            entry: dict[str, Any] = {}
            if spatial_src is not None:
                spatial_out = self._lookup_frame(spatial_src, non_cond_outputs)
                if spatial_out is not None:
                    for key in (
                        "maskmem_features",
                        "maskmem_pos_enc",
                        "image_features",
                        "image_pos_enc",
                    ):
                        if key in spatial_out:
                            entry[key] = spatial_out[key]
            else:
                entry["maskmem_features"] = None

            template_ptr: torch.Tensor | None = None
            for s in per_obj_src:
                if s is not None:
                    cand = self._lookup_frame(s, non_cond_outputs)
                    if cand is not None and "obj_ptr" in cand:
                        template_ptr = cand["obj_ptr"]
                        break

            if template_ptr is not None:
                combined_ptr = torch.zeros_like(template_ptr)
                for obj_idx, src_f in enumerate(per_obj_src):
                    if src_f is None:
                        continue
                    src_out = self._lookup_frame(src_f, non_cond_outputs)
                    if src_out is None or "obj_ptr" not in src_out:
                        continue
                    ptr = src_out["obj_ptr"]
                    if ptr.dim() == 3:
                        b_idx = obj_idx // num_mux
                        m_idx = obj_idx % num_mux
                        if b_idx < ptr.shape[0] and b_idx < combined_ptr.shape[0]:
                            combined_ptr[b_idx, m_idx] = ptr[b_idx, m_idx]
                    elif ptr.dim() == 2 and obj_idx < ptr.shape[0]:
                        combined_ptr[obj_idx] = ptr[obj_idx]
                entry["obj_ptr"] = combined_ptr

            if entry:
                remapped[target_key] = entry

        return remapped

    def __call__(
        self,
        frame_idx: int,
        is_init_cond_frame: bool,
        current_vision_feats: Any,
        current_vision_pos_embeds: Any,
        feat_sizes: Any,
        output_dict: dict[str, Any],
        num_frames: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        multiplex_state = kwargs.get("multiplex_state", args[0] if args else None)

        if is_init_cond_frame or frame_idx <= 0 or id(output_dict) != self._output_dict_id or frame_idx <= self._last_frame_idx:
            if is_init_cond_frame or frame_idx <= 1 or id(output_dict) != self._output_dict_id or frame_idx < self._last_frame_idx:
                is_new_output_dict_in_run = self._output_dict_id is not None and (
                    id(output_dict) != self._output_dict_id or frame_idx < self._last_frame_idx
                )
                self.reset_state(clear_runs=not is_new_output_dict_in_run)
                self._output_dict_id = id(output_dict)

        self._last_frame_idx = frame_idx

        if is_init_cond_frame or self.tracker.num_maskmem == 0:
            return self.orig_fn(
                frame_idx,
                is_init_cond_frame,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
                output_dict,
                num_frames,
                *args,
                **kwargs,
            )

        non_cond_outputs: dict[int, dict[str, Any]] = output_dict["non_cond_frame_outputs"]
        prev_idx = frame_idx - 1
        if prev_idx >= 1 and prev_idx in non_cond_outputs:
            prev_out = non_cond_outputs[prev_idx]
            score = self._score_previous_frame(prev_idx, prev_out, multiplex_state)
            if score is not None and bool((score > self.mf_threshold).any().item()):
                self.retained_bank[prev_idx] = self._snapshot_entry(prev_out)

        # Also score any other unscored entries present in non_cond_outputs
        for f_idx, f_out in non_cond_outputs.items():
            if f_idx not in self.frame_scores and "eff_iou_score" in f_out:
                s = f_out["eff_iou_score"].detach().float().cpu().reshape(-1)
                self.frame_scores[f_idx] = s
                if bool((s > self.mf_threshold).any().item()):
                    self.retained_bank[f_idx] = self._snapshot_entry(f_out)

        if multiplex_state is not None and hasattr(multiplex_state, "total_valid_entries"):
            n_obj = int(multiplex_state.total_valid_entries)
        elif self.frame_scores:
            first_score = next(iter(self.frame_scores.values()))
            n_obj = int(first_score.numel())
        else:
            n_obj = 1

        max_obj_ptrs = min(num_frames, int(self.tracker.max_obj_ptrs_in_encoder))
        present_frames = set(non_cond_outputs.keys()) | set(self.retained_bank.keys())
        valid_per_obj = frame_filter_per_object(
            frame_scores=self.frame_scores,
            frame_idx=frame_idx,
            max_num=max_obj_ptrs,
            mf_threshold=self.mf_threshold,
            n_obj=n_obj,
            present_frames=present_frames,
        )
        self.selected_history[frame_idx] = valid_per_obj

        # Keep retained_bank bounded to frames currently in valid_per_obj for at least one object
        keep_frames: set[int] = set()
        for v in valid_per_obj:
            keep_frames.update(v)
        for old_f in list(self.retained_bank.keys()):
            if old_f not in keep_frames:
                del self.retained_bank[old_f]

        remapped_non_cond = self._build_remapped_non_cond(
            frame_idx=frame_idx,
            max_obj_ptrs=max_obj_ptrs,
            valid_per_obj=valid_per_obj,
            non_cond_outputs=non_cond_outputs,
            multiplex_state=multiplex_state,
        )
        remapped_output_dict = {
            "cond_frame_outputs": output_dict["cond_frame_outputs"],
            "non_cond_frame_outputs": remapped_non_cond,
        }
        return self.orig_fn(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            remapped_output_dict,
            num_frames,
            *args,
            **kwargs,
        )


def _install_patches(
    patcher: Any,
    mf_threshold: float,
    focus_obj_idx: int,
) -> MemorySelectionStep:
    tracker = patcher.model.diffusion_model.tracker
    # Always resolve the unpatched bound methods from tracker.__class__ so re-applying
    # or cloning never double-wraps a wrapper.
    orig_decoder_fwd = tracker.sam_mask_decoder.__class__.forward.__get__(
        tracker.sam_mask_decoder, tracker.sam_mask_decoder.__class__
    )
    orig_prep_mem = tracker.__class__._prepare_memory_conditioned_features.__get__(
        tracker, tracker.__class__
    )
    iou_tap = IoUTap(orig_decoder_fwd)
    step = MemorySelectionStep(
        tracker=tracker,
        orig_fn=orig_prep_mem,
        iou_tap=iou_tap,
        mf_threshold=mf_threshold,
        focus_obj_idx=focus_obj_idx,
    )
    patcher.add_object_patch(
        "diffusion_model.tracker.sam_mask_decoder.forward",
        iou_tap,
    )
    patcher.add_object_patch(
        "diffusion_model.tracker._prepare_memory_conditioned_features",
        step,
    )
    # Give every patched or cloned ModelPatcher a distinct attachment key so
    # ComfyUI's ModelPatcher.clone_has_same_weights (which checks attachments.keys()
    # before short-circuiting on empty weight patches) never skips patch_model().
    for k in [k for k in patcher.attachments if isinstance(k, str) and k.startswith(CALLBACK_KEY)]:
        del patcher.attachments[k]
    patcher.attachments[f"{CALLBACK_KEY}_{uuid.uuid4().hex}"] = step
    setattr(patcher, STATE_ATTR, step)
    return step


def _on_clone_reinstall(src: Any, dst: Any) -> None:
    cfg = getattr(src, CONFIG_ATTR, None)
    if cfg is None:
        return
    setattr(dst, CONFIG_ATTR, dict(cfg))
    _install_patches(
        dst,
        mf_threshold=cfg["mf_threshold"],
        focus_obj_idx=cfg["focus_obj_idx"],
    )


def apply_memory_selection(
    model: Any,
    mf_threshold: float = MF_THRESHOLD,
    focus_obj_idx: int = 0,
) -> Any:
    """Return a cloned `ModelPatcher` with runtime memory selection installed.

    Works on both `SAM31Model` (`SAM31Tracker`) and `SAM3Model` (`SAM3Tracker`).
    Never mutates `model` in place or edits ComfyUI core files.
    """
    m = model.clone()
    cfg = {
        "mf_threshold": float(mf_threshold),
        "focus_obj_idx": int(focus_obj_idx),
    }
    setattr(m, CONFIG_ATTR, cfg)
    _install_patches(m, mf_threshold=cfg["mf_threshold"], focus_obj_idx=cfg["focus_obj_idx"])
    existing = m.callbacks.get(CallbacksMP.ON_CLONE, {}).get(CALLBACK_KEY, [])
    if not existing:
        m.add_callback_with_key(CallbacksMP.ON_CLONE, CALLBACK_KEY, _on_clone_reinstall)
    return m


def get_memory_selection_state(model: Any) -> MemorySelectionStep | None:
    """Return the active `MemorySelectionStep` attached to `model` (or its live tracker)."""
    step = getattr(model, STATE_ATTR, None)
    if isinstance(step, MemorySelectionStep):
        return step
    try:
        fn = model.model.diffusion_model.tracker._prepare_memory_conditioned_features
        if isinstance(fn, MemorySelectionStep):
            return fn
    except AttributeError:
        pass
    return None


def track_masks_with_sam(
    model: Any,
    images: torch.Tensor,
    initial_masks: torch.Tensor,
) -> dict[str, Any]:
    """Run video tracking on `model` (`SAM3Tracker` or `SAM31Tracker`).

    Args:
        model: ComfyUI `ModelPatcher` wrapping `SAM3Model` or `SAM31Model`.
        images: `[N, H, W, 3]` float32 RGB frames in `[0, 1]`.
        initial_masks: `[N_obj, H, W]` or `[N_obj, 1, H, W]` binary seed masks on frame 0.

    Returns:
        dict with:
          - `masks`: bool numpy array `[N, N_obj, H, W]` at the input `(H, W)` resolution
          - `frame_scores`: dict[int, list[float]] per-frame `eff_iou_score` if memory selection ran
          - `selected_history`: dict[int, list[list[int]]] per-frame selected valid indices if memory selection ran
          - `elapsed_s`: wall time in seconds
    """
    import comfy.model_management
    from comfy.ldm.sam3.tracker import unpack_masks

    H, W = int(images.shape[1]), int(images.shape[2])
    images_nchw = images.permute(0, 3, 1, 2).contiguous()
    if initial_masks.dim() == 2:
        initial_masks = initial_masks.unsqueeze(0).unsqueeze(0)
    elif initial_masks.dim() == 3:
        initial_masks = initial_masks.unsqueeze(1)

    comfy.model_management.load_model_gpu(model)
    diff_model = model.model.diffusion_model

    model.patch_model(load_weights=False)
    try:
        active_step = get_memory_selection_state(model)
        if active_step is not None:
            active_step.reset_state(clear_runs=True)

        t0 = time.perf_counter()
        with torch.inference_mode():
            result = diff_model.forward_video(
                images_nchw,
                initial_masks=initial_masks,
                new_det_thresh=1.0,
                max_objects=int(initial_masks.shape[0]),
                detect_interval=1,
                pbar=None,
                target_device=model.load_device,
                target_dtype=model.model.manual_cast_dtype or model.model_dtype(),
            )
        elapsed_s = time.perf_counter() - t0

        frame_scores_out: dict[int, list[float]] = {}
        selected_out: dict[int, list[list[int]]] = {}
        if active_step is not None:
            frame_scores_out = {
                int(k): [round(float(x), 6) for x in v.tolist()]
                for k, v in sorted(active_step.merged_frame_scores().items())
            }
            selected_out = {
                int(k): [[int(i) for i in obj_list] for obj_list in v]
                for k, v in sorted(active_step.merged_selected_history().items())
            }
    finally:
        model.unpatch_model(unpatch_weights=False)

    if isinstance(result, dict):
        packed = result["packed_masks"]
        if packed is None:
            masks_out = np.zeros((images.shape[0], initial_masks.shape[0], H, W), dtype=bool)
        else:
            binary = unpack_masks(packed)  # [N, N_obj, 1008, 1008]
            n_f, n_o = int(binary.shape[0]), int(binary.shape[1])
            flat = binary.reshape(n_f * n_o, 1, binary.shape[2], binary.shape[3]).float()
            resized = F.interpolate(flat, size=(H, W), mode="bilinear", align_corners=False) > 0.5
            masks_out = resized.reshape(n_f, n_o, H, W).cpu().numpy()
    elif isinstance(result, torch.Tensor):
        # SAM3Tracker.track_video returns float mask logits [N, N_obj, 1008, 1008]
        n_f, n_o = int(result.shape[0]), int(result.shape[1])
        flat = result.reshape(n_f * n_o, 1, result.shape[2], result.shape[3]).float()
        resized = F.interpolate(flat, size=(H, W), mode="bilinear", align_corners=False) > 0.0
        masks_out = resized.reshape(n_f, n_o, H, W).cpu().numpy()
    else:
        raise TypeError(f"Unexpected forward_video return type: {type(result)!r}")

    return {
        "masks": masks_out,
        "frame_scores": frame_scores_out,
        "selected_history": selected_out,
        "elapsed_s": round(elapsed_s, 3),
    }


def compute_mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute binary IoU between two 2D boolean masks."""
    a_b = np.asarray(a, dtype=bool)
    b_b = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(a_b, b_b).sum())
    union = int(np.logical_or(a_b, b_b).sum())
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def compute_mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Return `(cx, cy)` centroid in pixel coordinates, or `None` if mask is empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return (round(float(xs.mean()), 2), round(float(ys.mean()), 2))


def summarize_track_series(
    masks: np.ndarray,
    obj_idx: int = 0,
    first_source_frame: int = 0,
    stride: int = 1,
    companion_masks: dict[str, np.ndarray] | None = None,
    baseline_masks: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Build per-frame diagnostics (area, centroid, IoU vs baseline and companions).

    `masks` has shape `[N, N_obj, H, W]` (or `[N, H, W]`). When `stride == -1`
    (backward propagation from a later `pick_frame`), frame `i` maps to
    `first_source_frame - i`; when `stride == 1`, `first_source_frame + i`.
    """
    if masks.ndim == 4:
        series = masks[:, obj_idx]
    else:
        series = masks
    rows: list[dict[str, Any]] = []
    for i in range(series.shape[0]):
        src_f = first_source_frame + i * stride
        m = series[i]
        area = int(m.sum())
        centroid = compute_mask_centroid(m)
        row: dict[str, Any] = {
            "step_idx": i,
            "source_frame": src_f,
            "area_px": area,
            "centroid_xy": list(centroid) if centroid is not None else None,
        }
        if baseline_masks is not None and i < baseline_masks.shape[0]:
            row["iou_vs_off"] = round(compute_mask_iou(m, baseline_masks[i]), 4)
        if companion_masks:
            comp_ious: dict[str, float] = {}
            for cname, carr in companion_masks.items():
                if i < carr.shape[0]:
                    comp_ious[cname] = round(compute_mask_iou(m, carr[i]), 4)
            row["iou_vs_companions"] = comp_ious
        rows.append(row)
    return rows


def track_bidirectional_from_pick(
    model: Any,
    images: torch.Tensor,
    seed_masks: torch.Tensor,
    pick_frame: int,
) -> dict[str, Any]:
    """Track forward (`pick_frame..N-1`) and backward (`pick_frame..0`), mirroring `SubjectTrack`."""
    n_total = int(images.shape[0])
    n_obj = int(seed_masks.shape[0])
    H, W = int(images.shape[1]), int(images.shape[2])
    full_masks = np.zeros((n_total, n_obj, H, W), dtype=bool)
    full_scores: dict[int, list[float]] = {}
    full_selected: dict[int, list[list[int]]] = {}
    elapsed = 0.0

    # Forward pass: pick_frame .. n_total - 1
    fwd_imgs = images[pick_frame:]
    fwd_out = track_masks_with_sam(model, fwd_imgs, seed_masks)
    full_masks[pick_frame:] = fwd_out["masks"]
    elapsed += fwd_out["elapsed_s"]
    for k, v in fwd_out["frame_scores"].items():
        full_scores[pick_frame + k] = v
    for k, v in fwd_out["selected_history"].items():
        full_selected[pick_frame + k] = [
            [pick_frame + idx for idx in obj_list] for obj_list in v
        ]

    # Backward pass: pick_frame .. 0 (reversed)
    if pick_frame > 0:
        bwd_imgs = images[: pick_frame + 1].flip(0)
        bwd_out = track_masks_with_sam(model, bwd_imgs, seed_masks)
        # bwd_out["masks"][0] is pick_frame; [1:] is pick_frame - 1 down to 0
        full_masks[:pick_frame] = bwd_out["masks"][1:][::-1]
        elapsed += bwd_out["elapsed_s"]
        for k, v in bwd_out["frame_scores"].items():
            full_scores[pick_frame - k] = v
        for k, v in bwd_out["selected_history"].items():
            full_selected[pick_frame - k] = [
                [pick_frame - idx for idx in obj_list] for obj_list in v
            ]

    return {
        "masks": full_masks,
        "frame_scores": dict(sorted(full_scores.items())),
        "selected_history": dict(sorted(full_selected.items())),
        "elapsed_s": round(elapsed, 3),
    }


def _load_video_window(
    video_path: Path,
    first_frame: int,
    num_frames: int,
    width: int,
    height: int,
) -> torch.Tensor:
    """Decode `[num_frames, height, width, 3]` float32 RGB frames in `[0, 1]` using PyAV."""
    import av

    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if idx < first_frame:
                continue
            if len(frames) >= num_frames:
                break
            rgb = frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
            frames.append(rgb)
    if len(frames) != num_frames:
        raise RuntimeError(
            f"Expected {num_frames} frames starting at {first_frame} from {video_path.name}, "
            f"decoded {len(frames)}"
        )
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _load_video_window_by_seconds(
    video_path: Path,
    start_sec: float,
    num_frames: int,
    fps: float,
    width: int,
) -> torch.Tensor:
    """Decode `num_frames` at `fps` starting at `start_sec` via `LoadVideoFFmpegPath`, matching `subject_alone_or_in_a_group.py`."""
    import types
    import server

    if getattr(server.PromptServer, "instance", None) is None:
        server.PromptServer.instance = types.SimpleNamespace(prompt_queue=None, routes=None)
    vhs_root = COMFY_ROOT / "custom_nodes" / "ComfyUI-VideoHelperSuite"
    if str(vhs_root) not in sys.path:
        sys.path.insert(0, str(vhs_root))
    from videohelpersuite.load_video_nodes import LoadVideoFFmpegPath

    loaded = LoadVideoFFmpegPath().load_video(
        video=str(video_path),
        force_rate=float(fps),
        custom_width=int(width),
        custom_height=0,
        frame_load_cap=int(num_frames),
        start_time=float(start_sec),
        format="AnimateDiff",
    )[0][..., :3]
    u8 = (loaded.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return torch.from_numpy(u8.astype(np.float32) / 255.0)


def _resolve_input_dir() -> Path:
    env_dir = os.environ.get("H3_INPUT_DIR")
    if env_dir:
        return Path(env_dir)
    try:
        import folder_paths

        fp_in = Path(folder_paths.get_input_directory())
        if fp_in.is_dir():
            return fp_in
    except Exception:
        pass
    candidate = COMFY_ROOT / "input"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError("Could not locate H3_INPUT_DIR or ComfyUI input directory")


def _resolve_data_dir() -> Path:
    local_data = REPO_ROOT / "data"
    if local_data.is_dir():
        return local_data
    shared_data = COMFY_ROOT / "custom_nodes" / "ComfyUI-h3-explorations" / "data"
    if shared_data.is_dir():
        return shared_data
    return local_data


def build_default_scenes() -> dict[str, dict[str, Any]]:
    """Define the 4 benchmark scenes (kitchen 604-1009, hall 1306-1418, hall regain 1419-1465, lotsofpeopledance)."""
    input_dir = _resolve_input_dir()
    data_dir = _resolve_data_dir()
    fun_video = input_dir / "mrpop" / "fun_0000_6394_24p_sound_in_step.mov"
    dance_video = input_dir / "lotsofpeopledance_0414_0720.mkv"

    return {
        "kitchen_604_1009": {
            "description": (
                "Kitchen 604-1009 (`lead` and companions `second`, `third`): "
                "focus windows 684-760 (head-bend occlusion) and 998-1009 (`lead` occluded "
                "behind `second` at 999-1002 and raw track jumps onto `second` at 1003-1009)."
            ),
            "video_path": fun_video,
            "first_source_frame": 604,
            "num_frames": 406,
            "width": 1024,
            "height": 768,
            "pick_frame": 202,  # source frame 806
            "seed_npz": data_dir / "2026-10-15_second_hfp_0604e" / "masks__shot_c_0604b.npz",
            "seed_frame_in_npz": 202,
            "focus_windows": [[684, 760], [998, 1009]],
            # Secondary sub-probe on 680-760 seeded on lead's face (`parts[0]`) at source frame 680
            # where `lead`'s face is occluded on 687-707 (21 frames) and recovers at 708..760.
            "face_subprobe": {
                "first_source_frame": 680,
                "num_frames": 81,
                "pick_frame": 0,  # source frame 680
                "seed_npz": data_dir / "2026-10-17_lead_fp_0613g" / "masks__shot_c_0604b.npz",
                "seed_frame_in_npz": 67,  # 613 + 67 = 680
                "focus_windows": [[684, 760]],
            },
        },
        "hall_1306_1418": {
            "description": (
                "Hall 1306-1418 (`lead` and `third`/dad, plus `second`): "
                "seeded at pick_frame=4 (source 1310); `third` is occluded behind `lead` "
                "and `second` in the middle of the shot and steps out on 1392-1418."
            ),
            "video_path": fun_video,
            "first_source_frame": 1306,
            "num_frames": 113,
            "width": 1024,
            "height": 768,
            "pick_frame": 4,  # source frame 1310
            "seed_npz": data_dir / "2026-10-12_lead_fp_1306_span_1306_1418_blocked" / "masks__lead_c_1306.npz",
            "seed_frame_in_npz": 4,
            "focus_windows": [[1392, 1418]],
        },
        "regain_1419_1465": {
            "description": (
                "Hall regain 1419-1465 (`second`, companion 0): "
                "seeded at pick_frame=12 (source 1431), occluded behind `lead` and `third` "
                "on 1437-1452 (16 frames), and re-latched on 1453-1465."
            ),
            "video_path": fun_video,
            "first_source_frame": 1419,
            "num_frames": 47,
            "width": 1024,
            "height": 768,
            "pick_frame": 12,  # source frame 1431
            "seed_npz": data_dir / "2026-10-15_lead_fp_1419d" / "masks__shot_c_1419.npz",
            "seed_frame_in_npz": 12,
            "focus_windows": [[1435, 1465]],
        },
        "lotsofpeople": {
            "description": (
                "`lotsofpeopledance_0414_0720.mkv` (144 frames at 24 fps from 138.0s, width 1344, "
                "seeded on frame 4): tests both the `most central` dancer (lost in group of 16 "
                "around frame 95/115) and the `largest` front-row dancer (jumps to far-right "
                "dancer on frame 20 when tracked alone)."
            ),
            "video_path": dance_video,
            "start_sec": 138.0,
            "num_frames": 144,
            "fps": 24.0,
            "width": 1344,
            "seed_frame": 4,
            "focus_windows": [[15, 45], [85, 143]],
        },
    }


def load_sam_model(checkpoint_name: str) -> Any:
    """Load `SAM3Model` or `SAM31Model` via ComfyUI's `load_state_dict_guess_config`."""
    import comfy.sd
    import comfy.utils
    import folder_paths
    from bench._lib.sam3_core import TEXT_PROJECTION_KEY

    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint_name)
    sd, md = comfy.utils.load_torch_file(ckpt_path, return_metadata=True)
    sd.pop(TEXT_PROJECTION_KEY, None)
    model, _, _, _ = comfy.sd.load_state_dict_guess_config(
        sd,
        output_vae=False,
        output_clip=False,
        output_model=True,
        metadata=md,
    )
    return model


def run_scene_four_arms(
    scene_name: str,
    models: dict[str, Any],
    out_dir: Path,
    mf_threshold: float = MF_THRESHOLD,
) -> dict[str, Any]:
    """Run the 4 arms (SAM 3.1 off/on, SAM 3 base off/on; alone and group) on `scene_name`."""
    scenes = build_default_scenes()
    if scene_name not in scenes:
        raise KeyError(f"Unknown scene {scene_name!r}; expected one of {sorted(scenes)}")
    cfg = scenes[scene_name]
    out_dir.mkdir(parents=True, exist_ok=True)

    if scene_name == "lotsofpeople":
        return _run_lotsofpeople_scene(cfg, models, out_dir, mf_threshold=mf_threshold)

    images = _load_video_window(
        cfg["video_path"],
        first_frame=cfg["first_source_frame"],
        num_frames=cfg["num_frames"],
        width=cfg["width"],
        height=cfg["height"],
    )
    npz = np.load(cfg["seed_npz"])
    pf_npz = int(cfg["seed_frame_in_npz"])
    pick_frame = int(cfg["pick_frame"])
    first_src = int(cfg["first_source_frame"])

    # Build seed masks at pick_frame:
    # Object 0: `lead` (`raw[pf_npz] > 0.5`)
    # Companions: `companions[pf_npz, c] > 0.5` for each non-empty companion c
    lead_seed = torch.from_numpy((npz["raw"][pf_npz] > 0.5).astype(np.float32))
    comp_seeds: list[torch.Tensor] = []
    if "companions" in npz and npz["companions"].ndim == 4:
        for c_idx in range(npz["companions"].shape[1]):
            cm = (npz["companions"][pf_npz, c_idx] > 0.5).astype(np.float32)
            if cm.sum() > 0:
                comp_seeds.append(torch.from_numpy(cm))

    group_seeds = torch.stack([lead_seed] + comp_seeds, dim=0)  # [1 + C, H, W]
    alone_lead_seed = lead_seed.unsqueeze(0)  # [1, H, W]

    scene_report: dict[str, Any] = {
        "scene": scene_name,
        "description": cfg["description"],
        "first_source_frame": first_src,
        "num_frames": int(cfg["num_frames"]),
        "pick_frame": pick_frame,
        "pick_source_frame": first_src + pick_frame,
        "mf_threshold": mf_threshold,
        "n_group_objects": int(group_seeds.shape[0]),
        "focus_windows": cfg["focus_windows"],
        "arms": {},
    }

    # Determine which subjects to track:
    # - In kitchen_604_1009: target is `lead` (obj 0 in group, and alone), plus face_subprobe
    # - In hall_1306_1418: target is `lead` (obj 0) AND `third`/dad (companion 1 -> obj 2 in group, and alone)
    # - In regain_1419_1465: target is `second` (companion 0 -> obj 1 in group, and alone)
    targets: list[dict[str, Any]] = [
        {
            "name": "lead",
            "alone_seed": alone_lead_seed,
            "group_obj_idx": 0,
        }
    ]
    if scene_name == "hall_1306_1418" and len(comp_seeds) >= 2:
        targets.append(
            {
                "name": "third_dad",
                "alone_seed": comp_seeds[1].unsqueeze(0),
                "group_obj_idx": 2,
            }
        )
    elif scene_name == "regain_1419_1465" and len(comp_seeds) >= 1:
        targets.append(
            {
                "name": "second",
                "alone_seed": comp_seeds[0].unsqueeze(0),
                "group_obj_idx": 1,
            }
        )

    for model_key, base_model in models.items():
        model_arms: dict[str, Any] = {}
        for mem_on in (False, True):
            arm_label = f"{model_key}__mem_{'on' if mem_on else 'off'}"
            # 1) Alone calls for each target subject
            alone_results: dict[str, Any] = {}
            for tgt in targets:
                tname = tgt["name"]
                active_model = (
                    apply_memory_selection(base_model, mf_threshold=mf_threshold, focus_obj_idx=0)
                    if mem_on
                    else base_model.clone()
                )
                res = track_bidirectional_from_pick(
                    active_model, images, tgt["alone_seed"], pick_frame=pick_frame
                )
                alone_results[tname] = res

            # 2) Group call (all subjects together)
            # Note: For SAM31Tracker, group multiplexes all objects together.
            # When mem_on is True, we set focus_obj_idx to the primary target of the scene
            # (or run per-target focus_obj_idx if multiple targets are evaluated).
            group_results: dict[str, Any] = {}
            is_sam31 = "sam31" in model_key
            if len(comp_seeds) > 0:
                if is_sam31:
                    for tgt in targets:
                        tname = tgt["name"]
                        g_idx = int(tgt["group_obj_idx"])
                        active_model = (
                            apply_memory_selection(
                                base_model, mf_threshold=mf_threshold, focus_obj_idx=g_idx
                            )
                            if mem_on
                            else base_model.clone()
                        )
                        res = track_bidirectional_from_pick(
                            active_model, images, group_seeds, pick_frame=pick_frame
                        )
                        group_results[tname] = res
                else:
                    # SAM3Tracker.track_video loops _track_single_object per mask independently,
                    # so group object `g_idx` is identical to `alone` for that object; we still
                    # run or record it explicitly so the 4-arm x (alone, group) table is complete.
                    for tgt in targets:
                        tname = tgt["name"]
                        alone_res = alone_results[tname]
                        # Synthesize the group-indexed view around the single-object run
                        # (and verify with a group call on SAM3Tracker if needed).
                        group_results[tname] = {
                            "masks": alone_res["masks"],
                            "frame_scores": alone_res["frame_scores"],
                            "selected_history": alone_res["selected_history"],
                            "elapsed_s": alone_res["elapsed_s"],
                            "note": "SAM3Tracker.track_video loops _track_single_object per object independently.",
                        }

            model_arms["mem_on" if mem_on else "mem_off"] = {
                "alone": alone_results,
                "group": group_results,
            }
        scene_report["arms"][model_key] = model_arms

    # Optional kitchen face sub-probe (680-760, 21-frame occlusion 687-707)
    if "face_subprobe" in cfg:
        sub = cfg["face_subprobe"]
        sub_imgs = images[
            sub["first_source_frame"] - first_src : sub["first_source_frame"] - first_src + sub["num_frames"]
        ]
        sub_npz = np.load(sub["seed_npz"])
        face_seed = torch.from_numpy(
            (sub_npz["parts"][sub["seed_frame_in_npz"], 0] > 0.5).astype(np.float32)
        ).unsqueeze(0)
        sub_arms: dict[str, Any] = {}
        for model_key, base_model in models.items():
            sub_arms[model_key] = {}
            for mem_on in (False, True):
                active_model = (
                    apply_memory_selection(base_model, mf_threshold=mf_threshold, focus_obj_idx=0)
                    if mem_on
                    else base_model.clone()
                )
                sub_arms[model_key]["mem_on" if mem_on else "mem_off"] = track_masks_with_sam(
                    active_model, sub_imgs, face_seed
                )
        scene_report["face_subprobe"] = {
            "first_source_frame": sub["first_source_frame"],
            "num_frames": sub["num_frames"],
            "arms": _format_subprobe_summary(sub_arms, first_src=sub["first_source_frame"]),
        }

    summary = _compile_scene_summary(scene_report, targets, npz)
    out_json = out_dir / f"summary__{scene_name}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    return summary


def _format_subprobe_summary(sub_arms: dict[str, Any], first_src: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model_key, mdict in sub_arms.items():
        off_masks = mdict["mem_off"]["masks"][:, 0]
        on_masks = mdict["mem_on"]["masks"][:, 0]
        diffs: list[dict[str, Any]] = []
        for i in range(off_masks.shape[0]):
            iou = compute_mask_iou(on_masks[i], off_masks[i])
            off_area = int(off_masks[i].sum())
            on_area = int(on_masks[i].sum())
            if iou < 0.999 or off_area != on_area:
                diffs.append(
                    {
                        "source_frame": first_src + i,
                        "off_area_px": off_area,
                        "on_area_px": on_area,
                        "off_centroid": compute_mask_centroid(off_masks[i]),
                        "on_centroid": compute_mask_centroid(on_masks[i]),
                        "iou_on_vs_off": round(iou, 4),
                    }
                )
        out[model_key] = {
            "mem_off_elapsed_s": mdict["mem_off"]["elapsed_s"],
            "mem_on_elapsed_s": mdict["mem_on"]["elapsed_s"],
            "frame_scores_on": mdict["mem_on"]["frame_scores"],
            "selected_history_on": {
                str(first_src + k): [[first_src + x for x in o] for o in v]
                for k, v in mdict["mem_on"]["selected_history"].items()
            },
            "changed_frames_count": len(diffs),
            "changed_frames": diffs,
        }
    return out


def _compile_scene_summary(
    scene_report: dict[str, Any],
    targets: list[dict[str, Any]],
    npz: Any,
) -> dict[str, Any]:
    first_src = int(scene_report["first_source_frame"])
    n_frames = int(scene_report["num_frames"])
    focus_windows = scene_report["focus_windows"]

    # Build reference companion masks across the window from the saved capture npz if shapes match
    ref_companions: dict[str, np.ndarray] = {}
    if "raw" in npz and npz["raw"].shape[0] == n_frames:
        ref_companions["saved_lead"] = npz["raw"] > 0.5
    if "companions" in npz and npz["companions"].shape[0] == n_frames:
        for c_i in range(npz["companions"].shape[1]):
            ref_companions[f"saved_comp_{c_i}"] = npz["companions"][:, c_i] > 0.5

    compiled_arms: dict[str, Any] = {}
    for model_key, mdict in scene_report["arms"].items():
        compiled_arms[model_key] = {}
        for mode in ("alone", "group"):
            compiled_arms[model_key][mode] = {}
            for tgt in targets:
                tname = tgt["name"]
                if tname not in mdict["mem_off"].get(mode, {}):
                    continue
                off_res = mdict["mem_off"][mode][tname]
                on_res = mdict["mem_on"][mode][tname]
                obj_idx = (
                    0
                    if mode == "alone" or off_res["masks"].shape[1] == 1
                    else int(tgt["group_obj_idx"])
                )
                off_m = off_res["masks"][:, obj_idx]
                on_m = on_res["masks"][:, obj_idx]

                # Companion masks from the group run itself (if group) or saved reference
                live_companions: dict[str, np.ndarray] = dict(ref_companions)
                if mode == "group" and on_res["masks"].shape[1] > 1:
                    for o_i in range(on_res["masks"].shape[1]):
                        if o_i != obj_idx:
                            live_companions[f"group_obj_{o_i}_off"] = off_res["masks"][:, o_i]
                            live_companions[f"group_obj_{o_i}_on"] = on_res["masks"][:, o_i]

                off_series = summarize_track_series(
                    off_m, obj_idx=0, first_source_frame=first_src, companion_masks=live_companions
                )
                on_series = summarize_track_series(
                    on_m,
                    obj_idx=0,
                    first_source_frame=first_src,
                    companion_masks=live_companions,
                    baseline_masks=off_m,
                )

                changed_frames = [
                    {
                        "source_frame": on_series[i]["source_frame"],
                        "off_area_px": off_series[i]["area_px"],
                        "on_area_px": on_series[i]["area_px"],
                        "off_centroid_xy": off_series[i]["centroid_xy"],
                        "on_centroid_xy": on_series[i]["centroid_xy"],
                        "iou_vs_off": on_series[i]["iou_vs_off"],
                        "off_iou_vs_companions": off_series[i].get("iou_vs_companions", {}),
                        "on_iou_vs_companions": on_series[i].get("iou_vs_companions", {}),
                    }
                    for i in range(n_frames)
                    if on_series[i]["iou_vs_off"] < 0.999
                    or on_series[i]["area_px"] != off_series[i]["area_px"]
                ]

                focus_rows: list[dict[str, Any]] = []
                for w_start, w_end in focus_windows:
                    for i in range(n_frames):
                        sf = first_src + i
                        if w_start <= sf <= w_end:
                            score_val = on_res["frame_scores"].get(i)
                            sel_val = on_res["selected_history"].get(i)
                            focus_rows.append(
                                {
                                    "source_frame": sf,
                                    "off_area_px": off_series[i]["area_px"],
                                    "on_area_px": on_series[i]["area_px"],
                                    "off_centroid_xy": off_series[i]["centroid_xy"],
                                    "on_centroid_xy": on_series[i]["centroid_xy"],
                                    "iou_vs_off": on_series[i]["iou_vs_off"],
                                    "eff_iou_score": score_val,
                                    "selected_source_frames": (
                                        [[first_src + x for x in o] for o in sel_val]
                                        if sel_val is not None
                                        else None
                                    ),
                                    "off_iou_vs_companions": off_series[i].get("iou_vs_companions", {}),
                                    "on_iou_vs_companions": on_series[i].get("iou_vs_companions", {}),
                                }
                            )

                # Which frames had eff_iou_score[obj_idx] <= mf_threshold?
                filtered_out_frames: list[dict[str, Any]] = []
                for f_i, s_list in on_res["frame_scores"].items():
                    s_idx = min(obj_idx, len(s_list) - 1) if s_list else 0
                    if s_list and s_list[s_idx] <= scene_report["mf_threshold"]:
                        filtered_out_frames.append(
                            {
                                "source_frame": first_src + int(f_i),
                                "eff_iou_score": s_list[s_idx],
                                "all_object_scores": s_list,
                            }
                        )

                compiled_arms[model_key][mode][tname] = {
                    "mem_off_elapsed_s": off_res["elapsed_s"],
                    "mem_on_elapsed_s": on_res["elapsed_s"],
                    "total_frames": n_frames,
                    "changed_frames_count": len(changed_frames),
                    "filtered_out_frames_count": len(filtered_out_frames),
                    "filtered_out_frames": filtered_out_frames,
                    "changed_frames": changed_frames,
                    "focus_window_frames": focus_rows,
                }

    out = {
        "scene": scene_report["scene"],
        "description": scene_report["description"],
        "first_source_frame": first_src,
        "num_frames": n_frames,
        "pick_frame": scene_report["pick_frame"],
        "pick_source_frame": scene_report["pick_source_frame"],
        "mf_threshold": scene_report["mf_threshold"],
        "focus_windows": focus_windows,
        "arms": compiled_arms,
    }
    if "face_subprobe" in scene_report:
        out["face_subprobe"] = scene_report["face_subprobe"]
    return out


def _run_lotsofpeople_scene(
    cfg: dict[str, Any],
    models: dict[str, Any],
    out_dir: Path,
    mf_threshold: float = MF_THRESHOLD,
) -> dict[str, Any]:
    """Run the 4 arms on `lotsofpeople` (alone and 16-dancer group)."""
    import types
    from comfy_extras.nodes_sam3 import SAM3_Detect
    from bench._lib.sam3_core import load_core
    from workflows.h3_config import SEGMENTER

    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO_ROOT)]
    sys.modules.setdefault("_h3pack", pkg)
    import _h3pack.subject_track as st

    images = _load_video_window_by_seconds(
        cfg["video_path"],
        start_sec=float(cfg["start_sec"]),
        num_frames=int(cfg["num_frames"]),
        fps=float(cfg["fps"]),
        width=int(cfg["width"]),
    )
    seed_frame = int(cfg["seed_frame"])

    # Detect all people on the seed frame once using stock SAM3_Detect, matching subject_alone_or_in_a_group.py
    stock_m, stock_clip = load_core(SEGMENTER)
    cond = stock_clip.encode_from_tokens_scheduled(stock_clip.tokenize("person:16"))
    with torch.inference_mode():
        det_out = SAM3_Detect.execute(
            stock_m,
            images[seed_frame : seed_frame + 1],
            conditioning=cond,
            threshold=0.5,
            individual_masks=True,
        )
        all_seed_masks = getattr(det_out, "args", det_out)[0].float().cpu()

    flat = (all_seed_masks > 0.5).flatten(1).float()
    area = all_seed_masks.flatten(1).sum(1)

    def _overlap_of(i: int, j: int) -> float:
        inter = float((flat[i] * flat[j]).sum())
        u = max(float(flat[i].sum() + flat[j].sum()) - inter, 1.0)
        m = max(min(float(flat[i].sum()), float(flat[j].sum())), 1.0)
        return max(inter / u, inter / m)

    def _distinct(order: list[int], k: int = 16) -> list[int]:
        chosen: list[int] = []
        for idx in order:
            if any(_overlap_of(idx, j) >= 0.5 for j in chosen):
                continue
            chosen.append(idx)
            if len(chosen) == k:
                break
        return chosen

    targets_cfg = [
        ("most_central", int(st.choose(all_seed_masks, [], "most central"))),
        ("largest", int(st.choose(all_seed_masks, [], "largest"))),
    ]

    # Track from seed_frame forward (frames seed_frame .. N-1)
    track_images = images[seed_frame:]
    n_track = int(track_images.shape[0])

    compiled_arms: dict[str, Any] = {}
    for model_key, base_model in models.items():
        compiled_arms[model_key] = {"alone": {}, "group": {}}
        is_sam31 = "sam31" in model_key

        for tname, tidx in targets_cfg:
            by_size = [tidx] + [i for i in area.argsort(descending=True).tolist() if i != tidx]
            ordered_indices = _distinct(by_size, 16)
            group_seed = all_seed_masks[ordered_indices]  # [<=16, H, W], target is index 0
            alone_seed = all_seed_masks[tidx : tidx + 1]  # [1, H, W]

            # Companion reference masks on seed frame for jump detection
            for mode, seed_tensor in (("alone", alone_seed), ("group", group_seed)):
                if mode == "group" and not is_sam31:
                    # SAM3Tracker tracks each object independently in a loop
                    alone_entry = compiled_arms[model_key]["alone"][tname]
                    compiled_arms[model_key]["group"][tname] = dict(
                        alone_entry,
                        note="SAM3Tracker.track_video loops _track_single_object per object independently.",
                    )
                    continue

                mode_runs: dict[str, Any] = {}
                for mem_on in (False, True):
                    active_model = (
                        apply_memory_selection(base_model, mf_threshold=mf_threshold, focus_obj_idx=0)
                        if mem_on
                        else base_model.clone()
                    )
                    mode_runs["mem_on" if mem_on else "mem_off"] = track_masks_with_sam(
                        active_model, track_images, seed_tensor
                    )

                off_res = mode_runs["mem_off"]
                on_res = mode_runs["mem_on"]
                off_m = off_res["masks"][:, 0]
                on_m = on_res["masks"][:, 0]

                changed_frames: list[dict[str, Any]] = []
                focus_rows: list[dict[str, Any]] = []
                for i in range(n_track):
                    clip_f = seed_frame + i
                    iou = round(compute_mask_iou(on_m[i], off_m[i]), 4)
                    off_area = int(off_m[i].sum())
                    on_area = int(on_m[i].sum())
                    off_c = compute_mask_centroid(off_m[i])
                    on_c = compute_mask_centroid(on_m[i])
                    row = {
                        "clip_frame": clip_f,
                        "off_area_px": off_area,
                        "on_area_px": on_area,
                        "off_centroid_xy": list(off_c) if off_c else None,
                        "on_centroid_xy": list(on_c) if on_c else None,
                        "iou_vs_off": iou,
                        "eff_iou_score": on_res["frame_scores"].get(i),
                    }
                    if iou < 0.999 or off_area != on_area:
                        changed_frames.append(row)
                    for w_s, w_e in cfg["focus_windows"]:
                        if w_s <= clip_f <= w_e:
                            focus_rows.append(row)

                filtered_out_frames = [
                    {
                        "clip_frame": seed_frame + int(f_i),
                        "eff_iou_score": s_list[0],
                        "all_object_scores": s_list,
                    }
                    for f_i, s_list in on_res["frame_scores"].items()
                    if s_list and s_list[0] <= mf_threshold
                ]

                compiled_arms[model_key][mode][tname] = {
                    "target_seed_idx": int(tidx),
                    "n_seeded_objects": int(seed_tensor.shape[0]),
                    "mem_off_elapsed_s": off_res["elapsed_s"],
                    "mem_on_elapsed_s": on_res["elapsed_s"],
                    "total_tracked_frames": n_track,
                    "changed_frames_count": len(changed_frames),
                    "filtered_out_frames_count": len(filtered_out_frames),
                    "filtered_out_frames": filtered_out_frames,
                    "changed_frames": changed_frames,
                    "focus_window_frames": focus_rows,
                }

    summary = {
        "scene": "lotsofpeople",
        "description": cfg["description"],
        "start_sec": cfg["start_sec"],
        "num_frames": cfg["num_frames"],
        "seed_frame": seed_frame,
        "n_detected_people_on_seed": int(all_seed_masks.shape[0]),
        "mf_threshold": mf_threshold,
        "arms": compiled_arms,
    }
    out_json = out_dir / "summary__lotsofpeople.json"
    out_json.write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["kitchen_604_1009", "hall_1306_1418", "regain_1419_1465", "lotsofpeople"],
        help="Scene names to measure.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["sam31", "sam3"],
        default=["sam31", "sam3"],
        help="Which SAM checkpoints to run (`sam31`, `sam3`).",
    )
    parser.add_argument(
        "--mf-threshold",
        type=float,
        default=MF_THRESHOLD,
        help="Memory selection effective IoU threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_resolve_data_dir() / "2026-10-17_sam_memory_selection",
        help="Untracked data directory for JSON measurement outputs.",
    )
    args = parser.parse_args(argv)

    ckpt_map = {
        "sam31": SAM31_CHECKPOINT,
        "sam3": SAM3_BASE_CHECKPOINT,
    }
    loaded_models: dict[str, Any] = {}
    for mkey in args.models:
        ckpt = ckpt_map[mkey]
        print(f"[sam_memory_selection] loading {mkey} ({ckpt}) ...", flush=True)
        loaded_models[mkey] = load_sam_model(ckpt)

    all_summaries: dict[str, Any] = {}
    for sname in args.scenes:
        print(f"[sam_memory_selection] running scene {sname} ...", flush=True)
        summary = run_scene_four_arms(
            sname,
            models=loaded_models,
            out_dir=args.out_dir,
            mf_threshold=args.mf_threshold,
        )
        all_summaries[sname] = summary
        print(
            f"[sam_memory_selection] finished {sname} -> {args.out_dir / f'summary__{sname}.json'}",
            flush=True,
        )

    combined_path = args.out_dir / "summary__all_scenes.json"
    combined_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"[sam_memory_selection] wrote {combined_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
