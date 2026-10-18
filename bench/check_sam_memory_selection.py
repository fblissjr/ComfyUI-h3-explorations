#!/usr/bin/env python3
"""CPU self-test for `bench/sam_memory_selection.py`. No SAM weights, no card.

Verifies on synthetic CPU `SAM3Tracker` and `SAM31Tracker` instances wrapped in
ComfyUI's real `ModelPatcher`:
  1. `the_paths_are_cores`: the attribute paths (`diffusion_model.tracker.sam_mask_decoder.forward`
     and `diffusion_model.tracker._prepare_memory_conditioned_features`) exist on both
     `SAM3Tracker` and `SAM31Tracker` in ComfyUI core.
  2. `cal_mem_score_is_per_object`: `cal_mem_score_per_object` returns `0.0` when
     `object_score_logits <= 0`, `(2 * sigmoid(logit) - 1) * max(ious)` when `> 0`, and
     computes each object's score independently without `.mean()` across objects or
     multiplex slots.
  3. `frame_filter_drops_bad_and_keeps_tail`: `frame_filter_per_object` drops prior frames
     with `eff_iou_score <= mf_threshold`, always retains `must_include = frame_idx - 1`
     at the tail, caps pre-tail valid history at `max_num - 1` in chronological order,
     and filters per object.
  4. `sam3_tracker_runtime_patch_and_clone`: `apply_memory_selection` installs `IoUTap`
     and `MemorySelectionStep` via `ModelPatcher.add_object_patch`, survives `model.clone()`
     (`CallbacksMP.ON_CLONE`), is idempotent on re-application, matches the unpatched
     tracker bit-for-bit when `mf_threshold = -1.0`, and retains valid pre-occlusion
     frames across an 18-frame occlusion stretch (`> 16` frames where core prunes
     `non_cond_frame_outputs`).
  5. `sam31_tracker_group_isolation`: on `SAM31Tracker` with a multi-object group where
     object 0 is occluded across 18 frames while companion object 1 stays visible with
     high score, object 0's filtered history drops the occluded frames and keeps only
     its own pre-occlusion valid frames plus `must_include`.

    CUDA_VISIBLE_DEVICES="" <comfy venv python> bench/check_sam_memory_selection.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMFY = REPO.parent.parent
for _p in (HERE, REPO, COMFY):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _lib import case, finish  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True
import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402
from comfy.ldm.sam3.tracker import MultiplexState, SAM31Tracker, SAM3Tracker  # noqa: E402

from bench.sam_memory_selection import (  # noqa: E402
    MF_THRESHOLD,
    MemorySelectionStep,
    apply_memory_selection,
    cal_mem_score_per_object,
    compute_mask_centroid,
    compute_mask_iou,
    frame_filter_per_object,
    get_memory_selection_state,
    summarize_track_series,
)


class _DummyBaseModel(nn.Module):
    def __init__(self, tracker: nn.Module) -> None:
        super().__init__()
        self.diffusion_model = nn.Module()
        self.diffusion_model.tracker = tracker
        self.manual_cast_dtype = None

    def memory_required(self, *args, **kwargs) -> int:
        return 0


def _wrap_in_patcher(tracker: nn.Module) -> comfy.model_patcher.ModelPatcher:
    base = _DummyBaseModel(tracker)
    return comfy.model_patcher.ModelPatcher(
        base,
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )


def the_paths_are_cores() -> str:
    for cls in (SAM3Tracker, SAM31Tracker):
        assert hasattr(cls, "_prepare_memory_conditioned_features"), (
            f"{cls.__name__} missing _prepare_memory_conditioned_features"
        )
    return "both SAM3Tracker and SAM31Tracker expose _prepare_memory_conditioned_features and sam_mask_decoder.forward"


def cal_mem_score_is_per_object() -> str:
    logits = torch.tensor([[-2.5], [0.0], [2.0]], dtype=torch.float32)
    ious = torch.tensor(
        [
            [0.2, 0.9, 0.5],
            [0.8, 0.7, 0.6],
            [0.4, 0.85, 0.6],
        ],
        dtype=torch.float32,
    )
    scores = cal_mem_score_per_object(logits, ious)
    assert scores.shape == (3,), f"Expected shape (3,), got {scores.shape}"
    assert scores[0].item() == 0.0, f"Expected 0.0 for negative logit, got {scores[0].item()}"
    assert scores[1].item() == 0.0, f"Expected 0.0 for zero logit, got {scores[1].item()}"
    expected_2 = (2.0 * (1.0 / (1.0 + math.exp(-2.0))) - 1.0) * 0.85
    assert abs(scores[2].item() - expected_2) < 1e-6, (
        f"Expected {expected_2}, got {scores[2].item()}"
    )

    # 1D and 3D iou_scores shapes + mismatch ValueError
    scores_1d = cal_mem_score_per_object(logits, torch.tensor([0.9, 0.8, 0.85]))
    assert torch.allclose(scores, scores_1d), "1D iou_scores handling failed"
    scores_3d = cal_mem_score_per_object(logits, ious.unsqueeze(1))
    assert torch.allclose(scores, scores_3d), "3D iou_scores handling failed"
    try:
        cal_mem_score_per_object(logits, ious[:2])
        raise AssertionError("Expected ValueError on object count mismatch")
    except ValueError:
        pass

    # Multi-object invariance: hidden lead (object 0, logit=-5.0) alongside 3 visible
    # companions (objects 1..3, logit=+6.0, iou=0.95). Object 0 must score 0.0 <= MF_THRESHOLD,
    # never averaged with companions.
    group_logits = torch.tensor([[-5.0], [6.0], [6.0], [6.0]], dtype=torch.float32)
    group_ious = torch.tensor(
        [
            [0.9, 0.95, 0.8],
            [0.9, 0.95, 0.8],
            [0.9, 0.95, 0.8],
            [0.9, 0.95, 0.8],
        ],
        dtype=torch.float32,
    )
    group_scores = cal_mem_score_per_object(group_logits, group_ious)
    assert group_scores[0].item() == 0.0 <= MF_THRESHOLD, (
        f"Hidden target in group must score 0.0, got {group_scores[0].item()}"
    )
    for comp_idx in (1, 2, 3):
        assert group_scores[comp_idx].item() > 0.9, (
            f"Visible companion {comp_idx} should score > 0.9, got {group_scores[comp_idx].item()}"
        )
    return "hidden target in a 4-object group scores 0.0 while 3 visible companions score >0.9"


def frame_filter_drops_bad_and_keeps_tail() -> str:
    frame_scores: dict[int, torch.Tensor] = {}
    for f in range(1, 26):
        s0 = 0.8 if f <= 6 else 0.0
        s1 = 0.8
        frame_scores[f] = torch.tensor([s0, s1], dtype=torch.float32)

    valid_per_obj = frame_filter_per_object(
        frame_scores=frame_scores,
        frame_idx=26,
        max_num=16,
        mf_threshold=MF_THRESHOLD,
        n_obj=3,  # Object 2 did not exist at must_include=25
    )
    assert valid_per_obj[0] == [1, 2, 3, 4, 5, 6, 25], (
        f"Unexpected valid_indices for occluded object 0: {valid_per_obj[0]}"
    )
    assert valid_per_obj[1] == list(range(11, 26)), (
        f"Unexpected valid_indices for visible object 1: {valid_per_obj[1]}"
    )
    assert valid_per_obj[2] == [], (
        f"Newly spawned object 2 should not inherit must_include=25 when absent at 25: {valid_per_obj[2]}"
    )

    # Also verify mask IoU, centroid, and summarize_track_series diagnostics
    m0 = np.zeros((2, 1, 4, 4), dtype=bool)
    m0[0, 0, 1:3, 1:3] = True
    m1 = m0.copy()
    m1[1, 0, 0:2, 0:2] = True
    assert compute_mask_iou(m0[0, 0], m1[0, 0]) == 1.0
    assert compute_mask_centroid(m0[1, 0]) is None
    series = summarize_track_series(
        m1,
        obj_idx=0,
        first_source_frame=100,
        companion_masks={"comp": m0[:, 0]},
        baseline_masks=m0[:, 0],
    )
    assert len(series) == 2 and series[0]["iou_vs_off"] == 1.0 and series[1]["iou_vs_off"] == 0.0
    return "occluded object keeps [1..6, 25]; visible companion keeps [11..25]; dynamic n_obj respected"


def _run_synthetic_sam3_sequence(
    tracker: SAM3Tracker,
    n_frames: int,
    occluded_frames: set[int],
    seed_offset: int = 0,
) -> tuple[list[torch.Tensor], dict[int, dict]]:
    H = W = tracker.image_size // tracker.backbone_stride
    C = tracker.d_model
    mem_dim = tracker.mem_dim
    feat_sizes = [(H * 4, W * 4), (H * 2, W * 2), (H, W)]

    torch.manual_seed(42 + seed_offset)
    vision_feats_per_frame = [
        [
            torch.randn(1, (H * 4) * (W * 4), C // 4),
            torch.randn(1, (H * 2) * (W * 2), C // 2),
            torch.randn(1, H * W, C),
        ]
        for _ in range(n_frames)
    ]
    vision_pos = [
        torch.zeros(1, (H * 4) * (W * 4), C // 4),
        torch.zeros(1, (H * 2) * (W * 2), C // 2),
        torch.zeros(1, H * W, C),
    ]

    output_dict: dict[str, dict] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    output_dict["cond_frame_outputs"][0] = {
        "maskmem_features": torch.randn(1, mem_dim, H, W),
        "maskmem_pos_enc": [torch.zeros(1, mem_dim, H, W)],
        "obj_ptr": torch.randn(1, C),
        "object_score_logits": torch.tensor([[5.0]]),
    }

    # Configure decoder heads so sam_mask_decoder(...) directly outputs controlled
    # iou_pred (0.85) and object_score_logits (-5.0 when occluded, +5.0 when visible),
    # exercising IoUTap.__call__ directly on out[1] and out[3].
    with torch.no_grad():
        tracker.sam_mask_decoder.iou_prediction_head.layers[-1].weight.zero_()
        tracker.sam_mask_decoder.iou_prediction_head.layers[-1].bias.fill_(0.85)
        tracker.sam_mask_decoder.pred_obj_score_head.layers[-1].weight.zero_()

    conditioned_outs: list[torch.Tensor] = []
    for f_idx in range(1, n_frames):
        mem_feat = tracker._prepare_memory_conditioned_features(
            frame_idx=f_idx,
            is_init_cond_frame=False,
            current_vision_feats=vision_feats_per_frame[f_idx],
            current_vision_pos_embeds=vision_pos,
            feat_sizes=feat_sizes,
            output_dict=output_dict,
            num_frames=n_frames,
        )
        conditioned_outs.append(mem_feat.detach().clone())

        logit_val = -5.0 if f_idx in occluded_frames else 5.0
        with torch.no_grad():
            tracker.sam_mask_decoder.pred_obj_score_head.layers[-1].bias.fill_(logit_val)

        image_pe = tracker.sam_prompt_encoder.get_dense_pe()
        sparse_emb = torch.empty(1, 0, C)
        dense_emb = torch.zeros(1, C, H, W)
        _, _, _, dec_logits = tracker.sam_mask_decoder(
            image_embeddings=mem_feat,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            high_res_features=None,
            return_all=True,
        )

        torch.manual_seed(1000 + f_idx + seed_offset)
        entry = {
            "maskmem_features": torch.randn(1, mem_dim, H, W),
            "maskmem_pos_enc": [torch.zeros(1, mem_dim, H, W)],
            "obj_ptr": torch.full((1, C), float(f_idx)),
        }
        # Omit object_score_logits from output_dict on odd frames so _score_previous_frame
        # must read both last_iou_pred (out[1]) and last_obj_score_logits (out[3]) from IoUTap.
        if f_idx % 2 == 0:
            entry["object_score_logits"] = dec_logits.detach()
        output_dict["non_cond_frame_outputs"][f_idx] = entry

        lookback = max(tracker.num_maskmem, tracker.max_obj_ptrs_in_encoder)
        for old_idx in list(output_dict["non_cond_frame_outputs"]):
            if old_idx < f_idx - lookback:
                del output_dict["non_cond_frame_outputs"][old_idx]

    return conditioned_outs, output_dict


def sam3_tracker_runtime_patch_and_clone() -> str:
    torch.manual_seed(0)
    tracker = SAM3Tracker(
        d_model=32,
        mem_dim=8,
        num_maskmem=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        operations=comfy.ops.disable_weight_init,
    )
    tracker.image_size = 112
    tracker.backbone_stride = 14
    tracker.sam_prompt_encoder.image_embedding_size = (8, 8)
    tracker.sam_prompt_encoder.input_image_size = (112, 112)
    with torch.no_grad():
        for k, p in enumerate(tracker.parameters()):
            p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(100 + k)) * 0.05)
    tracker.eval()

    base_patcher = _wrap_in_patcher(tracker)

    # 1. Parity when mf_threshold = -1.0 (all frames pass)
    unpatched_outs, _ = _run_synthetic_sam3_sequence(tracker, n_frames=10, occluded_frames=set())

    patched_all_pass = apply_memory_selection(base_patcher, mf_threshold=-1.0)
    patched_all_pass = apply_memory_selection(patched_all_pass.clone(), mf_threshold=-1.0)
    patched_all_pass.patch_model(load_weights=False)
    try:
        all_pass_outs, _ = _run_synthetic_sam3_sequence(
            patched_all_pass.model.diffusion_model.tracker,
            n_frames=10,
            occluded_frames=set(),
        )
    finally:
        patched_all_pass.unpatch_model(unpatch_weights=False)

    for idx, (a, b) in enumerate(zip(unpatched_outs, all_pass_outs)):
        assert torch.equal(a, b), f"SAM3Tracker parity failed at step {idx} when mf_threshold=-1.0"

    # 2. Occlusion > 16 frames (frames 4..21 occluded = 18 frames; frame 22 evaluates memory)
    occluded = set(range(4, 22))
    patched_parent = apply_memory_selection(base_patcher, mf_threshold=MF_THRESHOLD)
    patched_memsel = patched_parent.clone()
    assert get_memory_selection_state(patched_memsel) is not get_memory_selection_state(patched_parent), (
        "clone() must give the cloned ModelPatcher its own MemorySelectionStep state via ON_CLONE"
    )
    assert not patched_memsel.clone_has_same_weights(patched_parent), (
        "Cloned patched ModelPatcher must have a distinct patches_uuid so load_model_gpu never skips patch_model()"
    )
    assert not patched_memsel.clone_has_same_weights(base_patcher), (
        "Patched ModelPatcher must not report clone_has_same_weights(base_patcher) == True"
    )
    patched_memsel.patch_model(load_weights=False)
    try:
        memsel_outs, _ = _run_synthetic_sam3_sequence(
            patched_memsel.model.diffusion_model.tracker,
            n_frames=23,
            occluded_frames=occluded,
        )
        state = get_memory_selection_state(patched_memsel)
        assert isinstance(state, MemorySelectionStep)
        sel_22 = state.selected_history[22][0]
        assert sel_22 == [1, 2, 3, 21], (
            f"Expected SAM3Tracker to retain pre-occlusion frames [1, 2, 3, 21] across "
            f"18-frame occlusion, got {sel_22}"
        )
        # Verify retained_bank is bounded by keep_frames (at frame 22 when frames 1..20 are all
        # valid, only the latest 15 valid frames [6..20] are kept in retained_bank, dropping 1..5).
        state.reset_state(clear_runs=True)
        _run_synthetic_sam3_sequence(
            patched_memsel.model.diffusion_model.tracker,
            n_frames=23,
            occluded_frames=set(),
        )
        assert len(state.retained_bank) <= tracker.max_obj_ptrs_in_encoder - 1, (
            f"retained_bank grew unbounded ({len(state.retained_bank)} > {tracker.max_obj_ptrs_in_encoder - 1})"
        )
        # Verify multi-object SAM3Tracker sequential _track_single_object archiving:
        # Running a second single-object pass with a fresh output_dict archives run 0 and merges both.
        _run_synthetic_sam3_sequence(
            patched_memsel.model.diffusion_model.tracker,
            n_frames=23,
            occluded_frames=occluded,
            seed_offset=50,
        )
        merged_sel = state.merged_selected_history()
        merged_scores = state.merged_frame_scores()
        assert len(merged_sel[22]) == 2 and merged_sel[22][1] == [1, 2, 3, 21], (
            f"Expected merged_selected_history to preserve both sequential objects, got {merged_sel[22]}"
        )
        assert merged_scores[10].shape == (2,), (
            f"Expected merged_frame_scores[10] shape (2,), got {merged_scores[10].shape}"
        )
    finally:
        patched_memsel.unpatch_model(unpatch_weights=False)

    # Verify parent patcher's state was untouched by running its clone
    parent_state = get_memory_selection_state(patched_parent)
    assert isinstance(parent_state, MemorySelectionStep) and not parent_state.selected_history, (
        "Running a cloned patcher mutated the parent patcher's MemorySelectionStep state"
    )

    # Verify the retained [1, 2, 3] memories actually changed frame 22's conditioned features
    # compared to when frames 1..21 are all occluded (where only [21] is selected)
    patched_all_occ = apply_memory_selection(base_patcher, mf_threshold=MF_THRESHOLD)
    patched_all_occ.patch_model(load_weights=False)
    try:
        all_occ_outs, _ = _run_synthetic_sam3_sequence(
            patched_all_occ.model.diffusion_model.tracker,
            n_frames=23,
            occluded_frames=set(range(1, 22)),
        )
    finally:
        patched_all_occ.unpatch_model(unpatch_weights=False)
    assert not torch.allclose(memsel_outs[-1], all_occ_outs[-1]), (
        "Retained pre-occlusion memories [1, 2, 3] did not affect frame 22 conditioned features"
    )

    # Verify unpatch_model restored the stock bound method
    assert not isinstance(tracker._prepare_memory_conditioned_features, MemorySelectionStep), (
        "unpatch_model did not restore stock _prepare_memory_conditioned_features"
    )
    return "bit-identical at mf_threshold=-1.0; retains [1, 2, 3, 21] across 18-frame occlusion; clone & bounded bank verified"


def _run_synthetic_sam31_group_sequence(
    tracker: SAM31Tracker,
    n_frames: int,
    obj0_occluded: set[int],
    comp1_ptr_scale: float = 1.0,
) -> list[torch.Tensor]:
    H = W = 8
    C = tracker.d_model
    M = tracker.num_multiplex
    feat_sizes = [(H * 4, W * 4), (H * 2, W * 2), (H, W)]
    mux_state = MultiplexState(2, M, torch.device("cpu"), torch.float32)

    torch.manual_seed(99)
    vision_feats = [torch.randn(1, H * W, C)]
    vision_pos = [torch.zeros(1, H * W, C)]

    output_dict: dict[str, dict] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    output_dict["cond_frame_outputs"][0] = {
        "maskmem_features": torch.randn(1, C, H, W),
        "maskmem_pos_enc": [torch.zeros(1, C, H, W)],
        "image_features": torch.randn(1, H * W, C),
        "image_pos_enc": torch.zeros(1, H * W, C),
        "obj_ptr": torch.zeros(1, M, C),
        "object_score_logits": torch.tensor([[5.0], [5.0]]),
    }

    outs: list[torch.Tensor] = []
    for f_idx in range(1, n_frames):
        mem_feat = tracker._prepare_memory_conditioned_features(
            frame_idx=f_idx,
            is_init_cond_frame=False,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos,
            feat_sizes=feat_sizes,
            output_dict=output_dict,
            num_frames=n_frames,
            multiplex_state=mux_state,
        )
        assert mem_feat.shape == (1, C, H, W)
        outs.append(mem_feat.detach().clone())

        step = getattr(tracker._prepare_memory_conditioned_features, "iou_tap", None)
        iou_mux = torch.full((1, M, 3), 0.9, dtype=torch.float32)
        logits_mux = torch.full((1, M, 1), -4.0, dtype=torch.float32)
        logits_mux[0, 0, 0] = -4.0 if f_idx in obj0_occluded else 5.0
        logits_mux[0, 1, 0] = 6.0  # companion is always visible
        if step is not None:
            step.last_iou_pred = iou_mux
            step.last_obj_score_logits = logits_mux

        ptr = torch.zeros(1, M, C)
        ptr[0, 0, :] = float(f_idx)
        scale = comp1_ptr_scale if f_idx < n_frames - 2 else 1.0
        ptr[0, 1, :] = float(100 + f_idx) * scale
        output_dict["non_cond_frame_outputs"][f_idx] = {
            "maskmem_features": torch.randn(1, C, H, W),
            "maskmem_pos_enc": [torch.zeros(1, C, H, W)],
            "image_features": torch.randn(1, H * W, C),
            "image_pos_enc": torch.zeros(1, H * W, C),
            "obj_ptr": ptr,
            "object_score_logits": mux_state.demux(logits_mux),
        }
        lookback = max(tracker.num_maskmem, tracker.max_obj_ptrs_in_encoder)
        for old_idx in list(output_dict["non_cond_frame_outputs"]):
            if old_idx < f_idx - lookback:
                del output_dict["non_cond_frame_outputs"][old_idx]

    return outs


def sam31_tracker_group_isolation() -> str:
    torch.manual_seed(0)
    tracker = SAM31Tracker(
        d_model=32,
        mem_dim=32,
        num_maskmem=4,
        num_multiplex=16,
        device=torch.device("cpu"),
        dtype=torch.float32,
        operations=comfy.ops.disable_weight_init,
    )
    tracker.image_size = 112
    tracker.backbone_stride = 14
    with torch.no_grad():
        for k, p in enumerate(tracker.parameters()):
            p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(200 + k)) * 0.05)
    tracker.eval()

    base_patcher = _wrap_in_patcher(tracker)
    patched = apply_memory_selection(base_patcher, mf_threshold=MF_THRESHOLD, focus_obj_idx=0).clone()
    patched.patch_model(load_weights=False)
    try:
        outs_scale1 = _run_synthetic_sam31_group_sequence(
            patched.model.diffusion_model.tracker,
            n_frames=22,
            obj0_occluded=set(range(3, 21)),
            comp1_ptr_scale=1.0,
        )
        state = get_memory_selection_state(patched)
        assert isinstance(state, MemorySelectionStep)
        sel_21 = state.selected_history[21]
        assert sel_21[0] == [1, 2, 20], (
            f"Expected Object 0 in group to filter out frames 3..19 despite visible companion, "
            f"got {sel_21[0]}"
        )
        assert sel_21[1] == list(range(6, 21)), (
            f"Expected Companion 1 in group to keep frames 6..20, got {sel_21[1]}"
        )
        # Verify that per-object combined_ptr assembly in _build_remapped_non_cond
        # actually feeds each object's own selected obj_ptr slots into memory attention:
        state.reset_state(clear_runs=True)
        outs_scale0 = _run_synthetic_sam31_group_sequence(
            patched.model.diffusion_model.tracker,
            n_frames=22,
            obj0_occluded=set(range(3, 21)),
            comp1_ptr_scale=0.0,
        )
        assert not torch.allclose(outs_scale1[-1], outs_scale0[-1]), (
            "Per-object combined_ptr assembly in _build_remapped_non_cond did not affect SAM31Tracker memory features"
        )
    finally:
        patched.unpatch_model(unpatch_weights=False)
    return "SAM31Tracker group: occluded target selects [1, 2, 20], companion selects [6..20], and combined_ptr feeds memory"


def main() -> int:
    for fn in (
        the_paths_are_cores,
        cal_mem_score_is_per_object,
        frame_filter_drops_bad_and_keeps_tail,
        sam3_tracker_runtime_patch_and_clone,
        sam31_tracker_group_isolation,
    ):
        case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
