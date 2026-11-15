"""Corrections for ComfyUI's departures from Meta's SAM 3.1 code, attached as ComfyUI object patches.

ComfyUI's SAM 3.1 (`comfy/ldm/sam3`, `comfy_extras/nodes_sam3.py`) is a port. Run beside Meta's code on equal inputs it
computes the same thing from the trunk on, and departs in places before/in the detector and in the video tracker:

1. **The image range.** ComfyUI's SAM nodes resize a frame and hand it to the trunk in 0 to 1. Meta's image processor
   and its list-of-images route hand the trunk -1 to 1, and the patch embedding has no bias, so nothing absorbs it.
2. **The text encoder's activation.** ComfyUI's config runs `quick_gelu`; Meta built the encoder with exact GELU.
3. **The tracker's memory** (`memory_selection`, on by default). ComfyUI's tracker remembers the last six frames
   whatever they hold; Meta's keeps only past frames where the object was seen (`eff_iou_score > 0.01`). For
   single-object tracking (`N_obj == 1`, as Subject Track runs), per-object selection is already at the `One-by-one`
   SAM3-DMS upper bound. For multi-object SAM 3.1 Object Multiplex (`N_obj >= 2`), Decoupled Multiplex Memory Selection
   (`MemoryEncodeFilter` write-time slot suppression + any-visible spatial union + per-object `obj_ptr`) decouples each
   target's reliability without stripping visible companions' spatial memory. Applied to `SAM31Tracker` only.
4. **The propagation pointer token** (`correct_multimask_ptr`, optional, off by default). `SAM31Tracker._forward_propagation`
   projects token 0 even when `argmax(iou_obj)` picked multimask branch 1 or 2; this projects the winning branch's token.
5. **The detector presence head** (`correct_detector_presence`, optional, off by default). `SAM3Detector._detect` drops
   or ignores global `presence`; this folds `sigmoid(presence)` into detection logits once in `fp32` (preventing `fp16` `Inf`).

**One node, and not a loader.** It takes the MODEL and CLIP any checkpoint loader returned (ComfyUI's own, or another
pack's) and returns clones of them carrying object patches. Nothing is loaded, cached, moved to a device or accounted for
here: ComfyUI's patcher applies the patches when it loads the model for a call and puts the stock attributes back when
it unloads it, so the pair that was wired in stays stock, ComfyUI's SAM nodes take the corrected pair like any other, and
its memory management owns both. Wired twice in a row it is a no-op: what is attached is read off the patchers
themselves (`corrections`), never guessed from a frame's values.

**What it leaves as ComfyUI computes it**, even with all optional switches on: the resize kernel, query overlap removal
in `SAM3_Detect`, `SAM3_Detect`'s 2-pass refinement, the hole-fill value, the multi-object occlusion suppression rules,
and threshold-before-resize ordering on output.

**The range patch, and why it is shaped as it is.** It replaces the trunk's `patch_embed.forward`, the one door every
image passes: the detect node, its refinement crops, the track node and a direct trunk call. It clamps to 0..1 first,
because ComfyUI's own bicubic resize overshoots a little and Meta's routes saturate, then maps to -1..1. It calls the
module's class `forward` on the module it is attached to, never a bound method captured earlier, and it is bound again
for every clone (`CallbacksMP.ON_CLONE`): ComfyUI can give a clone a second instance of the model, and a patch holding
the first instance's layer would then compute with the wrong one.

**A stock model and its corrected clone in one graph** share one model instance, and ComfyUI's
`ModelPatcher.clone_has_same_weights` does not compare object patches. On the plain patcher the two were measured to
switch correctly in both directions (2026-10-07, a CPU probe through ComfyUI's detect node: evidence about the
switching, and about detections on the CPU path only, which differ from the card's). On the server's dynamic
patcher that is this node's acceptance test, run as a graph on the queue (`bench/sam31_corrections_queue_test.py`;
`bench/results/2026-10-07_sam31_corrections_on_a_queue.md`).

**It reads what is there and never corrects twice.** No correction is tied to a ComfyUI version. Memory selection is
attached once: a model that already carries it keeps the one it has, and so does a clone (bound again to the clone's
own tracker, with a state of its own). The activation is replaced only in MLPs that run the shipped one, so an encoder
ComfyUI already builds with exact GELU is left alone and the report says "already exact". The range patch looks at what arrives: a frame with values far below zero is already
in the trained range (ComfyUI's nodes changed, or something upstream in the graph mapped it), so from then on the patch
passes every frame through untouched, and `corrections` says so to whoever asks after frames have gone through (the
node's own report is written before any have). That test is best effort before the first such frame: a
bright frame that was already mapped has nothing low enough to give it away. `bench/check_sam31_corrections_on_card.py`
reads the stock pair's first layer, so a ComfyUI that starts mapping the range turns that check red.

**Offload and reload.** The patches live on the patcher (`object_patches`), not on the model: ComfyUI puts the stock
attributes back when it unloads the model and applies the patcher's again on every load, the plain class in
`patch_model` and the dynamic class in `partially_load`, so an offload does not lose them. That is read in ComfyUI's
code and is what the on-card check and the on-queue acceptance exercise. The pack's attention nodes attach through
`model_options` instead because the sampler passes those options into the transformer on every call; ComfyUI's SAM
nodes call the model directly and read no options, so for SAM an object patch is the one hook ComfyUI's own code path
honours.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from comfy_api.latest import io

logger = logging.getLogger(__name__)

#: read: the path of the trunk's patch embedding under ComfyUI's SAM 3 model (`comfy/ldm/sam3/sam.py`, `ViTDet.patch_embed`;
#: `comfy/ldm/sam3/detector.py` builds `detector.backbone.vision_backbone.trunk`). `bench/check_sam31_corrections.py` reads it
#: off core's classes so a rename there is red here.
PATCH_EMBED = "diffusion_model.detector.backbone.vision_backbone.trunk.patch_embed"
RANGE_PATCH = PATCH_EMBED + ".forward"
#: reasoned: ComfyUI's bicubic resize was measured to undershoot 0 by about a tenth on real frames
#: (`bench/results/2026-10-07_sam3_core_against_meta.md`'s runs) and cannot undershoot a 0..1 picture by a half; a frame
#: mapped to -1..1 has values down to -1. A half is clear of the first and catches a mapped frame with anything darker
#: than a quarter in it.
ALREADY_MAPPED_BELOW = -0.5
#: read: the activation ComfyUI's SAM 3 text encoder is configured with (`comfy/text_encoders/sam3_clip.py`).
SHIPPED_ACTIVATION = "quick_gelu"
NOT_CORRECTED = (
    "the resize",
    "query overlap removal in the detect node",
    "the detect node's refinement passes",
    "the hole-fill value",
    "the rules between tracked objects",
    "threshold before resize on output",
)


class NotSAM3(ValueError):
    """The loaded checkpoint has no SAM 3 trunk or text encoder where this node attaches."""


class CorrectedRange:
    """`forward` for the trunk's patch embedding: the frame clamped to 0..1, mapped to -1..1, then the stock forward."""

    def __init__(self, module, seen: dict | None = None):
        self.module = module
        self.stock = type(module).forward      # the class's, so a forward patched on the instance is never wrapped twice
        # what the patch has seen, shared by every clone's copy of it (a clone gets its own `CorrectedRange`, bound to
        # its own module): `arrives_mapped` is set by the first frame that is clearly in -1..1 already; then nothing is mapped
        self.seen = {"arrives_mapped": False} if seen is None else seen

    @property
    def arrives_mapped(self) -> bool:
        return bool(self.seen["arrives_mapped"])

    def __call__(self, x, *args, **kwargs):
        if not self.arrives_mapped and float(x.amin()) < ALREADY_MAPPED_BELOW:
            self.seen["arrives_mapped"] = True
            logger.warning("[h3] SAM 3.1 Corrections: a frame reached the model already in -1..1 (lowest value %.2f), so the "
                           "image range is left as it arrives from here on. ComfyUI's SAM nodes may map it themselves now, or "
                           "something upstream in the graph does.", float(x.amin()))
        if self.arrives_mapped:
            return self.stock(self.module, x, *args, **kwargs)
        return self.stock(self.module, x.clamp(0.0, 1.0) * 2.0 - 1.0, *args, **kwargs)


def _attach_range(patcher, seen: dict | None = None) -> None:
    import comfy.utils
    try:
        module = comfy.utils.get_attr(patcher.model, PATCH_EMBED)
    except AttributeError as exc:
        raise NotSAM3(f"SAM 3.1 Corrections: this checkpoint's model has no `{PATCH_EMBED}`; it is not a SAM 3 checkpoint.") from exc
    patcher.add_object_patch(RANGE_PATCH, CorrectedRange(module, seen))


def _rebind_range(parent, clone) -> None:
    """`CallbacksMP.ON_CLONE`: bind the range patch to the clone's own model instance, which need not be the parent's."""
    patch = clone.object_patches.get(RANGE_PATCH)
    if isinstance(patch, CorrectedRange):
        _attach_range(clone, patch.seen)


# ---------------------------------------------------------------- memory selection (SAM 3.1's tracker)
#
# Meta's tracker builds each frame's memory from past frames where the object was seen; ComfyUI's port takes the last
# six frames whatever they hold (`comfy/ldm/sam3/tracker.py`, `collect_memory_tokens` with `num_maskmem = 7`: six past
# frames and the conditioning frame). Meta scores a past frame by `cal_mem_score` and keeps it by `frame_filter` above
# `mf_threshold` (`sam3_tracker_base.py`, `video_tracking_multiplex.py`). Here the score is kept per object, never
# averaged across the objects or multiplex slots of one call, so a hidden object tracked beside visible ones is not
# admitted to memory on their scores. Two object patches on the tracker carry it: `IoUTap` on the mask decoder's
# `forward` (it records the decoder's IoU prediction and object score), and `MemorySelectionStep` on
# `_prepare_memory_conditioned_features` (it scores the frame before, filters the past frames per object, and hands
# core's own method an `output_dict` holding only the frames kept). Both are built from the CLASS's methods bound to
# the instance they are attached to, and are built again for every clone (`CallbacksMP.ON_CLONE`), as the range patch is.
# Measured on SAM 3.1 on three `fun` scenes and a group dance, 2026-10-17: with it on no arm was worse, and the lead's
# face over 680-760 of `lead_fp_0613g` came back at 710 instead of 715 (`data/2026-10-17_sam_memory_selection/`;
# `bench/sam_memory_selection.py` runs it). On SAM 3 it was not measured as a win, so the node leaves SAM 3 as ComfyUI
# builds it and says so. Three things changed after that measurement, on a blind read the same day: an object a
# detection adds mid-video no longer raises (`_score_previous_frame`), nothing is held on the card after a track
# (`end_of_pass`), and an object's pointers are read as Meta reads them (`meta_pointers`). The face probe was run again
# through this code (`bench/sam_memory_selection.py --face-probe`): back at 710 with it on and 715 off, as before, with
# the mask areas of the memory-on arm changed on some frames; the other scenes were not measured again. On a second
# blind read the node took selection to one-object calls only (`one_object_only`) and a track that follows another on
# a reused output_dict now starts a pass of its own (`end_of_pass`); the face probe, one object, was unchanged by both.

#: inherited: Meta's SAM 3 and SAM 3.1 tracker default (`mf_threshold: float = 0.01` in `sam3_tracker_base.py` and
#: `video_tracking_multiplex.py`). A past frame is kept for an object when its score is above this.
MF_THRESHOLD = 0.01
#: read: the tracker's path under ComfyUI's SAM 3 and SAM 3.1 models (`comfy/ldm/sam3/detector.py`, `self.tracker`), and
#: the attributes patched on it (`comfy/ldm/sam3/tracker.py`). `bench/check_sam31_corrections.py` reads them off core.
TRACKER = "diffusion_model.tracker"
DECODER_PATCH = TRACKER + ".sam_mask_decoder.forward"
MEMORY_PATCH = TRACKER + "._prepare_memory_conditioned_features"
ENCODE_MEM_PATCH = TRACKER + "._encode_new_memory"
TRACK_LOOP_PATCH = TRACKER + ".track_video_with_detection"
MEMORY_CALLBACK = "h3_sam31_memory_selection"
NO_OBJ_SCORE = -1024.0
#: reasoned: the input is optional, so a graph queued before it existed (a saved job graph, a re-queued render) still
#: validates (found by the server queue test on 2026-10-18: as a required input, every graph without it was refused,
#: "Required input is missing"). Absent means the widget's default, on, so the UI and the API agree
#: (`bench/check_schema_defaults.py`); an old graph re-queued therefore tracks with it on, not as it was first run.
MEMORY_SELECTION_WHEN_ABSENT = True
#: measured: the tracker class memory selection was measured as a win on (2026-10-17, above). The node attaches it to
#: this class only; `attach_memory_selection` takes any SAM 3 tracker, for the bench tool that measures both.
MEMORY_SELECTION_TRACKERS = ("SAM31Tracker",)
#: read: `MultiplexMaskDecoder` (`comfy/ldm/sam3/tracker.py:856`) returns `mask_tokens_2d[:, :, 0:1]` only; tapping
#: `sam_mask_decoder.transformer.forward` recovers all `T` mask tokens so `_forward_propagation` can project the
#: winning `argmax(iou_obj)` branch's token (`MULTIMASK_PTR_TRACKERS` only).
MULTIMASK_TAP_PATCH = TRACKER + ".sam_mask_decoder.transformer.forward"
PROPAGATION_PATCH = TRACKER + "._forward_propagation"
MULTIMASK_CALLBACK = "h3_sam31_multimask_ptr"
MULTIMASK_PTR_TRACKERS = ("SAM31Tracker",)
MULTIMASK_PTR_WHEN_ABSENT = False
#: read: `SAM3Detector._detect` (`comfy/ldm/sam3/detector.py:422-470`) returns `scores` without multiplying in
#: `dec_out["presence"]`. Wrapping `_detect` fixes both `SAM3_Detect` and `forward_from_trunk` (`SAM3_VideoTrack`).
DETECTOR = "diffusion_model.detector"
PRESENCE_PATCH = DETECTOR + "._detect"
PRESENCE_CALLBACK = "h3_sam31_detector_presence"
DETECTOR_PRESENCE_WHEN_ABSENT = False


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

    A frame scored for fewer objects than `n_obj` (it was tracked before a detection added objects,
    core's `_add_detected_objects`) is KEPT for the objects it has no score for: an object is
    filtered only on its own scores, and until it has them its memory is what ComfyUI's would be.
    """
    cap = max(max_num - 1, 0)
    valid_per_obj: list[list[int]] = []
    must_include = frame_idx - 1
    has_must = (
        (must_include in frame_scores)
        if present_frames is None
        else (must_include in present_frames or must_include in frame_scores)
    )

    for obj_idx in range(n_obj):
        valid: list[int] = []
        if cap > 0:
            for i in range(frame_idx - 1, 0, -1):
                if present_frames is not None and i not in present_frames:
                    continue
                scores_i = frame_scores.get(i)
                if scores_i is None:
                    continue
                if obj_idx >= scores_i.numel() or float(scores_i[obj_idx].item()) > mf_threshold:
                    valid.insert(0, i)
                if len(valid) >= cap:
                    break
        if must_include >= 1 and must_include not in valid and has_must:
            valid.append(must_include)
        valid_per_obj.append(valid)
    return valid_per_obj


def _slot_rows(x: torch.Tensor) -> torch.Tensor:
    """Flatten `[num_buckets, M, ...]` or `[N, ...]` per-slot outputs into `[slots, D]`."""
    x = x.detach()
    return x.reshape(-1, x.shape[-1]) if x.dim() >= 3 else x.unsqueeze(-1) if x.dim() == 1 else x


class IoUTap:
    """Wraps `sam_mask_decoder.forward` to record the latest `iou_pred`, `object_score_logits`, and `decode_seq`."""

    def __init__(self, orig_forward: Any) -> None:
        self.orig_forward = orig_forward
        self.last_iou_pred: torch.Tensor | None = None
        self.last_obj_score_logits: torch.Tensor | None = None
        self.decode_seq: int = 0
        # off after a pass's last frame (`MemorySelectionStep.end_of_pass`), so the decoder's output on that frame, which
        # no later frame reads, is not held on the card; on again when the next pass starts (`reset_state`)
        self.armed = True

    def reset(self) -> None:
        self.last_iou_pred = None
        self.last_obj_score_logits = None
        self.decode_seq = 0

    def record(self, iou_pred: torch.Tensor, obj_score_logits: torch.Tensor | None) -> None:
        if not self.armed:
            return
        self.last_iou_pred = iou_pred.detach()
        if obj_score_logits is not None:
            self.last_obj_score_logits = obj_score_logits.detach()
        self.decode_seq += 1

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self.orig_forward(*args, **kwargs)
        if isinstance(out, tuple) and len(out) >= 2:
            self.record(out[1], out[3] if len(out) >= 4 else None)
        return out


class MemoryEncodeFilter:
    """Write-time per-slot reliability gate on `SAM31Tracker._encode_new_memory` for multi-object calls (`N_obj >= 2`).

    Leaves `N_obj == 1` (`SAM31Tracker` single-object) completely untouched. When `N_obj >= 2`, suppresses corrupted
    memory writes for unreliable/occluded slots (`eff <= mf_threshold`) by setting their encoded mask to `NO_OBJ_SCORE`
    (`-1024.0`, mapping via `sigmoid * 20 - 10` to background `-10.0`) and `object_score_logits` to `-10.0` (activating
    `no_obj_embed_spatial`), while exempting conditioning masks, Call #2 reconditioned slots, and Call #3 newborn slots.
    """

    def __init__(self, orig_fn: Any, iou_tap: IoUTap, mf_threshold: float = MF_THRESHOLD) -> None:
        self.orig_fn = orig_fn
        self.iou_tap = iou_tap
        self.mf_threshold = float(mf_threshold)
        self.reset()

    def reset(self) -> None:
        self._last_decode_seq: int = -1
        self._last_slot_means: torch.Tensor | None = None
        self._expected_n_obj: int | None = None
        self._seq_n_scored: int | None = None
        self._reconditioned_slots: set[int] = set()
        self._had_birth: bool = False
        self.suppressed_last_call: list[int] = []

    def begin_frame(self, n_obj: int) -> None:
        """Called by `MemorySelectionStep` after scoring `frame_idx - 1` and before running `frame_idx`."""
        self._reconditioned_slots = set()
        self._had_birth = False
        self._expected_n_obj = int(n_obj)

    def __call__(
        self,
        pix_feat: Any,
        pred_masks_high_res: torch.Tensor,
        object_score_logits: torch.Tensor,
        is_mask_from_pts: bool = False,
        multiplex_state: Any = None,
        is_conditioning: bool = False,
        cond_obj_mask: torch.Tensor | None = None,
    ) -> Any:
        def _call_orig(masks_t: torch.Tensor, scores_t: torch.Tensor) -> Any:
            return self.orig_fn(
                pix_feat,
                masks_t,
                scores_t,
                is_mask_from_pts=is_mask_from_pts,
                multiplex_state=multiplex_state,
                is_conditioning=is_conditioning,
                cond_obj_mask=cond_obj_mask,
            )

        decode_seq = self.iou_tap.decode_seq
        if decode_seq != self._last_decode_seq:
            self._reconditioned_slots = set()
            self._had_birth = False
            self._last_slot_means = None
            self._last_decode_seq = decode_seq
            self._seq_n_scored = self._expected_n_obj
            self._expected_n_obj = None

        if is_conditioning or multiplex_state is None:
            self.suppressed_last_call = []
            return _call_orig(pred_masks_high_res, object_score_logits)

        n_now = int(getattr(multiplex_state, "total_valid_entries", 1))
        if n_now < 2:
            # Never touch N_obj == 1: single-object tracking must stay bit-for-bit identical.
            self.suppressed_last_call = []
            return _call_orig(pred_masks_high_res, object_score_logits)

        raw_iou = self.iou_tap.last_iou_pred
        raw_logits = self.iou_tap.last_obj_score_logits
        if raw_iou is None or raw_logits is None:
            self.suppressed_last_call = []
            return _call_orig(pred_masks_high_res, object_score_logits)

        slot_means = pred_masks_high_res.detach().float().flatten(1).mean(dim=1).cpu()
        if cond_obj_mask is None and self._last_slot_means is not None:
            for o in range(min(int(slot_means.numel()), int(self._last_slot_means.numel()))):
                if abs(float(slot_means[o].item() - self._last_slot_means[o].item())) > 1e-5:
                    self._reconditioned_slots.add(o)
        self._last_slot_means = slot_means

        logits_r, iou_r = _slot_rows(raw_logits), _slot_rows(raw_iou)
        n_masks = int(pred_masks_high_res.shape[0])
        if self._seq_n_scored is None:
            self._seq_n_scored = min(n_masks, int(logits_r.shape[0]), int(iou_r.shape[0]))
        n = min(self._seq_n_scored, n_masks, int(logits_r.shape[0]), int(iou_r.shape[0]))
        if n_masks > n or (cond_obj_mask is not None and bool(cond_obj_mask.any().item())):
            self._had_birth = True
        if n <= 0:
            self.suppressed_last_call = []
            return _call_orig(pred_masks_high_res, object_score_logits)

        eff = cal_mem_score_per_object(logits_r[:n], iou_r[:n])
        suppress_idx: list[int] = []
        for o in range(n):
            if o in self._reconditioned_slots:
                continue
            if cond_obj_mask is not None and o < int(cond_obj_mask.shape[0]) and bool(cond_obj_mask[o].item()):
                continue
            if float(eff[o].item()) <= self.mf_threshold:
                suppress_idx.append(o)

        self.suppressed_last_call = suppress_idx
        if not suppress_idx:
            return _call_orig(pred_masks_high_res, object_score_logits)

        masks_mod = pred_masks_high_res.clone()
        masks_mod[suppress_idx] = NO_OBJ_SCORE
        scores_mod = object_score_logits.clone()
        scores_mod[suppress_idx] = -10.0
        return _call_orig(masks_mod, scores_mod)


class MemoryTrackLoopGuard:
    """Wraps `SAM31Tracker.track_video_with_detection` so `end_of_pass()` always runs when a loop returns or raises."""

    def __init__(self, orig_fn: Any, step: MemorySelectionStep) -> None:
        self.orig_fn = orig_fn
        self.step = step

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self.orig_fn(*args, **kwargs)
        finally:
            self.step.end_of_pass()


class MemorySelectionStep:
    """Wraps `tracker._prepare_memory_conditioned_features` with per-object memory selection."""

    def __init__(
        self,
        tracker: Any,
        orig_fn: Any,
        iou_tap: IoUTap,
        mf_threshold: float = MF_THRESHOLD,
        focus_obj_idx: int | None = 0,
        record: bool = False,
        one_object_only: bool = False,
        encode_filter: MemoryEncodeFilter | None = None,
    ) -> None:
        self.tracker = tracker
        self.orig_fn = orig_fn
        self.iou_tap = iou_tap
        self.mf_threshold = float(mf_threshold)
        self.focus_obj_idx: int | None = None if focus_obj_idx is None else int(focus_obj_idx)
        # `record`: keep each pass's scores and kept frames after the next pass starts (`merged_*`), for the bench tool,
        # which clears them before every track. Off (the node): a pass's record is dropped when the next one starts, so
        # nothing grows over the patcher's life.
        self.record = bool(record)
        # `one_object_only` (legacy opt-out / explicit bench switch): selection on one-object calls only; stock core
        # memory on multi-object calls. The node uses `focus_obj_idx=None, one_object_only=False` (Multiplex DMS).
        self.one_object_only = bool(one_object_only)
        self.encode_filter = encode_filter
        # Meta reads an object's pointers from every kept frame but its oldest (`for t_diff ...: if -t_diff <=
        # -len(valid_indices): break`, coderef/sam3/sam3/model/video_tracking_multiplex.py:1517, and
        # sam3_tracker_base.py:723), so with only the frame before kept no non-conditioning pointer is read. Matched
        # here by leaving `obj_ptr` out of that frame's entry, which core's SAM31Tracker skips (`if out is not None and
        # "obj_ptr" in out`). Core's SAM3Tracker indexes `out["obj_ptr"]` unguarded, so on SAM 3 (the bench tool only;
        # the node does not attach there) every kept frame's pointer is still fed. No measured reason to differ.
        self.meta_pointers = type(tracker).__name__ == "SAM31Tracker"
        self._completed_runs: list[tuple[dict[int, torch.Tensor], dict[int, list[list[int]]]]] = []
        self.reset_state(clear_runs=True)

    def reset_state(self, clear_runs: bool = True) -> None:
        if clear_runs or not self.record:
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
        # Mid-video birth or reconditioned frames kept unconditionally in spatial union & retained_bank during a pass
        self.cond_frames: set[int] = set()
        # the frame from which this pass tracks several objects and keeps core's memory (`one_object_only`), else None
        self.several_from: int | None = None
        # set by `end_of_pass`: the next call starts a new pass whatever its output_dict's id or first frame
        self._pass_ended = False
        self.iou_tap.reset()
        self.iou_tap.armed = True
        if self.encode_filter is not None:
            self.encode_filter.reset()

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

        # Read per object by slot, never demuxed with the state as it is now. Core's MultiplexState puts object i in slot
        # i (`MultiplexState._build`: slot = (i // M) * M + i % M), so a multiplexed tensor of any bucket count reads per
        # object as its slots in order. The frame before may have been tracked before a detection added objects
        # (`_add_detected_objects` grows the state and leaves that frame's `object_score_logits` at the old count, and
        # the tap's IoU in the old bucket count): it is scored for the objects it was tracked with, the count its own
        # stored scores give, and an object added since has no score there (`frame_filter_per_object` keeps the frame
        # for it). Until 2026-10-17 this demuxed with the grown state and raised.
        logits_r, iou_r = _slot_rows(raw_logits), _slot_rows(raw_iou)
        if raw_logits.dim() == 2:
            n = int(raw_logits.shape[0])
        elif multiplex_state is not None:
            n = int(multiplex_state.total_valid_entries)
        else:
            n = int(logits_r.shape[0])
        n = min(n, int(logits_r.shape[0]), int(iou_r.shape[0]))
        if n <= 0:
            return None
        score = cal_mem_score_per_object(logits_r[:n], iou_r[:n])

        if self.encode_filter is not None:
            if self.encode_filter._reconditioned_slots:
                score = score.clone()
                for o in self.encode_filter._reconditioned_slots:
                    if 0 <= o < score.numel():
                        score[o] = max(float(score[o].item()), 1.0)
                self.cond_frames.add(prev_idx)
            if self.encode_filter._had_birth:
                self.cond_frames.add(prev_idx)
        if multiplex_state is not None and hasattr(multiplex_state, "total_valid_entries"):
            if int(multiplex_state.total_valid_entries) > score.numel():
                self.cond_frames.add(prev_idx)
        if "pred_masks" in prev_out and isinstance(prev_out["pred_masks"], torch.Tensor):
            if int(prev_out["pred_masks"].shape[0]) > score.numel():
                self.cond_frames.add(prev_idx)

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

    def _spatial_union_frames(
        self,
        frame_idx: int,
        max_num: int,
        n_obj: int,
        present_frames: set[int] | None = None,
    ) -> list[int]:
        """Select up to `max_num - 1` past frames for spatial memory where ANY valid object was seen or conditioned."""
        cap = max(max_num - 1, 0)
        valid: list[int] = []
        must_include = frame_idx - 1
        has_must = (
            (must_include in self.frame_scores or must_include in self.cond_frames)
            if present_frames is None
            else (must_include in present_frames or must_include in self.frame_scores)
        )
        if cap > 0:
            for i in range(frame_idx - 1, 0, -1):
                if present_frames is not None and i not in present_frames:
                    continue
                if i in self.cond_frames:
                    valid.insert(0, i)
                else:
                    scores_i = self.frame_scores.get(i)
                    if scores_i is None:
                        continue
                    if any(
                        o >= scores_i.numel() or float(scores_i[o].item()) > self.mf_threshold
                        for o in range(n_obj)
                    ):
                        valid.insert(0, i)
                if len(valid) >= cap:
                    break
        if must_include >= 1 and must_include not in valid and has_must:
            valid.append(must_include)
        return valid

    def _build_remapped_non_cond(
        self,
        frame_idx: int,
        max_obj_ptrs: int,
        valid_per_obj: list[list[int]],
        non_cond_outputs: dict[int, dict[str, Any]],
        multiplex_state: Any,
        spatial_valid: list[int] | None = None,
    ) -> dict[int, dict[str, Any]]:
        n_obj = len(valid_per_obj)
        if n_obj == 0:
            return {}

        if spatial_valid is None:
            focus_obj = 0 if self.focus_obj_idx is None else min(max(self.focus_obj_idx, 0), n_obj - 1)
            focus_valid = valid_per_obj[focus_obj]
        else:
            focus_valid = spatial_valid

        max_k = max(max((len(v) for v in valid_per_obj), default=0), len(focus_valid))
        max_k = min(max_k, max(max_obj_ptrs - 1, 0))

        remapped: dict[int, dict[str, Any]] = {}
        num_mux = getattr(self.tracker, "num_multiplex", 16)

        for k in range(1, max_k + 1):
            target_key = frame_idx - k
            if target_key < 0:
                break

            per_obj_src = [
                valid_per_obj[o][-k] if k <= len(valid_per_obj[o]) else None
                for o in range(n_obj)
            ]
            # each object's pointer from all its kept frames but the oldest (`self.meta_pointers`, Meta's rule)
            ptr_src = [
                src if src is not None and (k < len(valid_per_obj[o]) or not self.meta_pointers) else None
                for o, src in enumerate(per_obj_src)
            ]
            spatial_src = focus_valid[-k] if k <= len(focus_valid) else None
            spatial_out = self._lookup_frame(spatial_src, non_cond_outputs) if spatial_src is not None else None

            if all(src is None for src in ptr_src):
                # no object reads a pointer at this rank: spatial memory alone
                if spatial_out is not None:
                    remapped[target_key] = {key: v for key, v in spatial_out.items() if key != "obj_ptr"}
                continue
            if n_obj == 1 or multiplex_state is None or (
                spatial_out is not None and all(src == spatial_src for src in ptr_src)
            ):
                # one object, or every object reads this rank's pointer from the same spatial source frame
                if spatial_out is not None:
                    remapped[target_key] = spatial_out
                continue

            # Multi-object multiplex where objects or spatial union diverged at rank -k:
            # Spatial memory comes from `spatial_valid[-k]` (any-visible union when `focus_obj_idx is None`, else
            # `focus_obj`), while each object slot `(b = o // M, m = o % M)` in `obj_ptr` comes from its own
            # `valid_per_obj[o][-k]`, zero where that object reads no pointer at this rank.
            entry: dict[str, Any] = {}
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

            cand_ptrs: list[tuple[int, torch.Tensor]] = []
            for obj_idx, src_f in enumerate(ptr_src):
                if src_f is None:
                    continue
                src_out = self._lookup_frame(src_f, non_cond_outputs)
                if src_out is not None and "obj_ptr" in src_out:
                    cand_ptrs.append((obj_idx, src_out["obj_ptr"]))

            if cand_ptrs:
                first_ptr = cand_ptrs[0][1]
                if first_ptr.dim() == 3:
                    state_buckets = int(getattr(multiplex_state, "num_buckets", 1))
                    need_buckets = (n_obj + num_mux - 1) // num_mux
                    max_buckets = max(
                        state_buckets,
                        need_buckets,
                        max(int(p.shape[0]) for _, p in cand_ptrs),
                    )
                    combined_ptr = first_ptr.new_zeros((max_buckets, first_ptr.shape[1], first_ptr.shape[2]))
                    for obj_idx, ptr in cand_ptrs:
                        if ptr.dim() == 3:
                            b_idx = obj_idx // num_mux
                            m_idx = obj_idx % num_mux
                            if b_idx < ptr.shape[0] and b_idx < combined_ptr.shape[0] and m_idx < ptr.shape[1]:
                                combined_ptr[b_idx, m_idx] = ptr[b_idx, m_idx]
                else:
                    combined_ptr = torch.zeros_like(first_ptr)
                    for obj_idx, ptr in cand_ptrs:
                        if ptr.dim() == 2 and obj_idx < ptr.shape[0] and obj_idx < combined_ptr.shape[0]:
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
                is_new_output_dict_in_run = self._pass_ended or (self._output_dict_id is not None and (
                    id(output_dict) != self._output_dict_id or frame_idx < self._last_frame_idx
                ))
                self.reset_state(clear_runs=not is_new_output_dict_in_run)
                self._output_dict_id = id(output_dict)

        self._last_frame_idx = frame_idx
        n_now = int(getattr(multiplex_state, "total_valid_entries", 1)) if multiplex_state is not None else 1

        if is_init_cond_frame or self.tracker.num_maskmem == 0:
            if self.encode_filter is not None:
                self.encode_filter.begin_frame(n_now)
            out = self.orig_fn(
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
            if frame_idx >= num_frames - 1:
                self.end_of_pass()
            return out

        if self.one_object_only and (self.several_from is not None or n_now >= 2):
            if self.several_from is None:
                self.several_from = frame_idx
                self.retained_bank.clear()
            if self.encode_filter is not None:
                self.encode_filter.begin_frame(n_now)
            out = self.orig_fn(frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes,
                               output_dict, num_frames, *args, **kwargs)
            if frame_idx >= num_frames - 1:
                self.end_of_pass()
            return out

        non_cond_outputs: dict[int, dict[str, Any]] = output_dict["non_cond_frame_outputs"]
        prev_idx = frame_idx - 1
        if prev_idx >= 1 and prev_idx in non_cond_outputs:
            prev_out = non_cond_outputs[prev_idx]
            score = self._score_previous_frame(prev_idx, prev_out, multiplex_state)
            if (score is not None and bool((score > self.mf_threshold).any().item())) or prev_idx in self.cond_frames:
                self.retained_bank[prev_idx] = self._snapshot_entry(prev_out)

        # Also score any other unscored entries present in non_cond_outputs
        for f_idx, f_out in non_cond_outputs.items():
            if f_idx not in self.frame_scores and "eff_iou_score" in f_out:
                s = f_out["eff_iou_score"].detach().float().cpu().reshape(-1)
                self.frame_scores[f_idx] = s
                if bool((s > self.mf_threshold).any().item()) or f_idx in self.cond_frames:
                    self.retained_bank[f_idx] = self._snapshot_entry(f_out)

        if self.encode_filter is not None:
            self.encode_filter.begin_frame(n_now)

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

        if self.focus_obj_idx is None:
            spatial_valid = (
                valid_per_obj[0]
                if n_obj == 1
                else self._spatial_union_frames(frame_idx, max_obj_ptrs, n_obj, present_frames)
            )
        else:
            focus_idx = min(max(self.focus_obj_idx, 0), max(n_obj - 1, 0))
            spatial_valid = valid_per_obj[focus_idx] if n_obj > 0 else []

        # Keep retained_bank bounded to frames currently referenced by any object or spatial union
        keep_frames: set[int] = set(spatial_valid)
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
            spatial_valid=spatial_valid,
        )
        remapped_output_dict = {
            "cond_frame_outputs": output_dict["cond_frame_outputs"],
            "non_cond_frame_outputs": remapped_non_cond,
        }
        out = self.orig_fn(
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
        if frame_idx >= num_frames - 1:
            self.end_of_pass()
        return out

    def end_of_pass(self) -> None:
        """The pass's last frame is done: let go of what is on the card, keep the small record on the CPU.

        `retained_bank` holds memory tensors of up to `max_obj_ptrs - 1` frames (masks' memory, image features, their
        positions) and the tap the decoder's last output, all on the device and outside ComfyUI's memory accounting; a
        patch lives as long as its patcher, through offloads. The scores (`frame_scores`, CPU) and the kept frames
        (`selected_history`, ints) stay until the next pass starts, for whoever reads them."""
        self.retained_bank.clear()
        self.cond_frames.clear()
        self.iou_tap.reset()
        self.iou_tap.armed = False
        if self.encode_filter is not None:
            self.encode_filter.reset()
        # the next call is a new pass: core may hand it an output_dict at the same address (a freed dict's id is reused)
        # starting on a later frame, which the id and frame tests alone read as this pass going on, with the tap off
        self._output_dict_id = None
        self._last_frame_idx = -1
        self._pass_ended = True


def tracker_kind(model) -> str | None:
    """The class name of the video tracker under a SAM 3 patcher's model, or None when it has none."""
    import comfy.utils
    try:
        return type(comfy.utils.get_attr(model.model, TRACKER)).__name__
    except AttributeError:
        return None


def _install_memory(
    patcher,
    mf_threshold: float,
    focus_obj_idx: int | None,
    record: bool = False,
    one_object_only: bool = False,
) -> MemorySelectionStep:
    """Install tracker memory patches built from the class's methods bound to THIS patcher's tracker."""
    import comfy.utils
    try:
        tracker = comfy.utils.get_attr(patcher.model, TRACKER)
    except AttributeError as exc:
        raise NotSAM3(f"SAM 3.1 Corrections: this checkpoint's model has no `{TRACKER}`; it is not a SAM 3 checkpoint.") from exc
    decoder = tracker.sam_mask_decoder
    tap = IoUTap(type(decoder).forward.__get__(decoder, type(decoder)))
    encode_filter: MemoryEncodeFilter | None = None
    has_mux_loop = hasattr(type(tracker), "track_video_with_detection")
    if not one_object_only and has_mux_loop and hasattr(type(tracker), "_encode_new_memory"):
        encode_filter = MemoryEncodeFilter(
            orig_fn=type(tracker)._encode_new_memory.__get__(tracker, type(tracker)),
            iou_tap=tap,
            mf_threshold=mf_threshold,
        )
    step = MemorySelectionStep(
        tracker=tracker,
        orig_fn=type(tracker)._prepare_memory_conditioned_features.__get__(tracker, type(tracker)),
        iou_tap=tap,
        mf_threshold=mf_threshold,
        focus_obj_idx=focus_obj_idx,
        record=record,
        one_object_only=one_object_only,
        encode_filter=encode_filter,
    )
    patcher.add_object_patch(DECODER_PATCH, tap)
    patcher.add_object_patch(MEMORY_PATCH, step)
    if encode_filter is not None:
        patcher.add_object_patch(ENCODE_MEM_PATCH, encode_filter)
    if has_mux_loop:
        patcher.add_object_patch(
            TRACK_LOOP_PATCH,
            MemoryTrackLoopGuard(type(tracker).track_video_with_detection.__get__(tracker, type(tracker)), step),
        )
    return step


def _rebind_memory(parent, clone) -> None:
    """`CallbacksMP.ON_CLONE`: the clone gets patches of its own, bound to its own tracker and with a state of its own."""
    step = clone.object_patches.get(MEMORY_PATCH)
    if isinstance(step, MemorySelectionStep):
        _install_memory(clone, step.mf_threshold, step.focus_obj_idx, step.record, step.one_object_only)


def get_memory_selection_state(model) -> MemorySelectionStep | None:
    """The `MemorySelectionStep` a patcher carries (or, failing that, the one live on its tracker), else None."""
    step = model.object_patches.get(MEMORY_PATCH) if hasattr(model, "object_patches") else None
    if isinstance(step, MemorySelectionStep):
        return step
    try:
        live = model.model.diffusion_model.tracker._prepare_memory_conditioned_features
    except AttributeError:
        return None
    return live if isinstance(live, MemorySelectionStep) else None


def attach_memory_selection(
    patcher,
    mf_threshold: float = MF_THRESHOLD,
    focus_obj_idx: int | None = 0,
    record: bool = False,
    one_object_only: bool = False,
) -> MemorySelectionStep:
    """Attach memory selection to a SAM 3 or SAM 3.1 patcher, in place. Idempotent: a patcher that carries it keeps it.

    `focus_obj_idx`: `None` enables Decoupled Multiplex Memory Selection (any-visible spatial union + per-slot `obj_ptr` +
    write-time slot gate via `MemoryEncodeFilter`); an integer pins spatial memory to that focus object index for
    ablation/bench runs. A patcher that already carries memory selection with another threshold or focus is refused, never
    silently kept or replaced. `record`: keep every pass's scores and kept frames until `reset_state(clear_runs=True)`
    (the bench tool's tracks); the node keeps none. `one_object_only`: legacy opt-out forcing unfiltered core memory
    whenever two or more objects are tracked in one call."""
    from comfy.patcher_extension import CallbacksMP
    norm_focus = None if focus_obj_idx is None else int(focus_obj_idx)
    step = patcher.object_patches.get(MEMORY_PATCH)
    if isinstance(step, MemorySelectionStep):
        if (step.mf_threshold, step.focus_obj_idx) != (float(mf_threshold), norm_focus):
            raise ValueError(
                f"this model already carries memory selection at mf_threshold {step.mf_threshold}, focus "
                f"{step.focus_obj_idx}; asked for {float(mf_threshold)}, {norm_focus}. Attach to the model "
                "before it was given one."
            )
        return step
    step = _install_memory(patcher, float(mf_threshold), norm_focus, bool(record), bool(one_object_only))
    if MEMORY_CALLBACK not in patcher.callbacks.get(CallbacksMP.ON_CLONE, {}):
        patcher.add_callback_with_key(CallbacksMP.ON_CLONE, MEMORY_CALLBACK, _rebind_memory)
    return step


def apply_memory_selection(
    model,
    mf_threshold: float = MF_THRESHOLD,
    focus_obj_idx: int | None = 0,
    record: bool = True,
    one_object_only: bool = False,
):
    """A clone of a SAM 3 or SAM 3.1 patcher with memory selection attached; the patcher handed in is not touched.

    Records every pass by default: the bench tool reads the merged record of a track's passes after it returns."""
    m = model.clone()
    attach_memory_selection(
        m,
        mf_threshold=mf_threshold,
        focus_obj_idx=focus_obj_idx,
        record=record,
        one_object_only=one_object_only,
    )
    return m


# ---------------------------------------------------------------- propagation pointer token (SAM 3.1 multimask)


class MultimaskTokenTap:
    """Wraps `sam_mask_decoder.transformer.forward` to expose all `T` mask tokens during `_forward_propagation`."""

    def __init__(self, orig_forward: Any) -> None:
        self.orig_forward = orig_forward
        self.last_hs: torch.Tensor | None = None
        self.armed = False

    def reset(self) -> None:
        self.last_hs = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self.orig_forward(*args, **kwargs)
        if self.armed and isinstance(out, tuple) and len(out) >= 1 and isinstance(out[0], torch.Tensor):
            self.last_hs = out[0]
        return out


class CorrectedPropagationPtr:
    """Wraps `SAM31Tracker._forward_propagation` to project `tokens_obj[obj, best_idx]` instead of `tokens_obj[:, 0]`."""

    def __init__(self, tracker: Any, orig_fn: Any, token_tap: MultimaskTokenTap) -> None:
        self.tracker = tracker
        self.orig_fn = orig_fn
        self.token_tap = token_tap

    def __call__(self, backbone_features: Any, high_res_features: Any = None, multiplex_state: Any = None) -> Any:
        self.token_tap.reset()
        self.token_tap.armed = True
        try:
            low_res_masks, high_res_masks, obj_ptr_muxed, score_obj = self.orig_fn(
                backbone_features,
                high_res_features=high_res_features,
                multiplex_state=multiplex_state,
            )
            hs = self.token_tap.last_hs
            if hs is None or multiplex_state is None:
                return low_res_masks, high_res_masks, obj_ptr_muxed, score_obj

            decoder = self.tracker.sam_mask_decoder
            M = int(decoder.num_multiplex)
            T = int(getattr(decoder, "num_mask_output_per_object", getattr(decoder, "num_multimask_outputs", 3)))
            B_hs = int(hs.shape[0])
            iou_pred = decoder.iou_prediction_head(hs[:, M : 2 * M]).view(B_hs, M, T)
            iou_obj = multiplex_state.demux(iou_pred)
            best_idx = torch.argmax(iou_obj, dim=-1)
            if not bool((best_idx != 0).any().item()):
                return low_res_masks, high_res_masks, obj_ptr_muxed, score_obj

            mask_tokens_2d = hs[:, 2 * M : 2 * M + M * T].reshape(B_hs, M, T, -1)
            tokens_obj = multiplex_state.demux(mask_tokens_2d)
            obj_range = torch.arange(tokens_obj.shape[0], device=tokens_obj.device)
            sam_token = tokens_obj[obj_range, best_idx]
            obj_ptr = self.tracker.obj_ptr_proj(sam_token)
            is_obj = (score_obj > 0).float()
            no_obj = self.tracker.no_obj_ptr_linear(obj_ptr)
            obj_ptr = is_obj * obj_ptr + (1.0 - is_obj) * no_obj
            obj_ptr_muxed = multiplex_state.mux(obj_ptr)
            return low_res_masks, high_res_masks, obj_ptr_muxed, score_obj
        finally:
            self.token_tap.reset()
            self.token_tap.armed = False


def _install_multimask_ptr(patcher) -> CorrectedPropagationPtr:
    import comfy.utils
    try:
        tracker = comfy.utils.get_attr(patcher.model, TRACKER)
        decoder_xfmr = tracker.sam_mask_decoder.transformer
    except AttributeError as exc:
        raise NotSAM3(
            f"SAM 3.1 Corrections: this checkpoint's model has no `{MULTIMASK_TAP_PATCH}`; it is not a SAM 3.1 checkpoint."
        ) from exc
    if not hasattr(type(tracker), "_forward_propagation"):
        raise NotSAM3(
            f"SAM 3.1 Corrections: tracker `{type(tracker).__name__}` has no `_forward_propagation`."
        )
    tap = MultimaskTokenTap(type(decoder_xfmr).forward.__get__(decoder_xfmr, type(decoder_xfmr)))
    prop = CorrectedPropagationPtr(
        tracker=tracker,
        orig_fn=type(tracker)._forward_propagation.__get__(tracker, type(tracker)),
        token_tap=tap,
    )
    patcher.add_object_patch(MULTIMASK_TAP_PATCH, tap)
    patcher.add_object_patch(PROPAGATION_PATCH, prop)
    return prop


def _rebind_multimask_ptr(parent, clone) -> None:
    """`CallbacksMP.ON_CLONE`: bind multimask token tap and propagation wrapper to the clone's own tracker."""
    if isinstance(clone.object_patches.get(PROPAGATION_PATCH), CorrectedPropagationPtr):
        _install_multimask_ptr(clone)


def _attach_multimask_ptr(patcher) -> None:
    from comfy.patcher_extension import CallbacksMP
    if not isinstance(patcher.object_patches.get(PROPAGATION_PATCH), CorrectedPropagationPtr):
        _install_multimask_ptr(patcher)
    if MULTIMASK_CALLBACK not in patcher.callbacks.get(CallbacksMP.ON_CLONE, {}):
        patcher.add_callback_with_key(CallbacksMP.ON_CLONE, MULTIMASK_CALLBACK, _rebind_multimask_ptr)


# ---------------------------------------------------------------- detector presence head


class CorrectedDetectorPresence:
    """Wraps `SAM3Detector._detect` to fold `sigmoid(presence)` into detection logits once in `float32`."""

    def __init__(self, orig_fn: Any) -> None:
        self.orig_fn = orig_fn

    def __call__(
        self,
        features: Any,
        positions: Any,
        text_embeddings: Any = None,
        text_mask: Any = None,
        points: Any = None,
        boxes: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        has_prompts = text_embeddings is not None or points is not None or boxes is not None
        boxes_xyxy, scores, masks, dec_out = self.orig_fn(
            features, positions, text_embeddings, text_mask, points, boxes, *args, **kwargs
        )
        if has_prompts and isinstance(dec_out, dict):
            presence = dec_out.get("presence")
            if presence is not None:
                pres = presence.float().reshape(scores.shape[0], -1).sigmoid()
                prob = scores.float().sigmoid() * pres
                scores = torch.logit(prob, eps=1e-6).clamp(-12.0, 12.0).to(dtype=scores.dtype)
        return boxes_xyxy, scores, masks, dec_out


def _install_detector_presence(patcher) -> CorrectedDetectorPresence:
    import comfy.utils
    try:
        detector = comfy.utils.get_attr(patcher.model, DETECTOR)
    except AttributeError as exc:
        raise NotSAM3(
            f"SAM 3.1 Corrections: this checkpoint's model has no `{DETECTOR}`; it is not a SAM 3 checkpoint."
        ) from exc
    if not hasattr(type(detector), "_detect"):
        raise NotSAM3(
            f"SAM 3.1 Corrections: detector `{type(detector).__name__}` has no `_detect`."
        )
    patch = CorrectedDetectorPresence(type(detector)._detect.__get__(detector, type(detector)))
    patcher.add_object_patch(PRESENCE_PATCH, patch)
    return patch


def _rebind_detector_presence(parent, clone) -> None:
    """`CallbacksMP.ON_CLONE`: bind the detector presence patch to the clone's own detector instance."""
    if isinstance(clone.object_patches.get(PRESENCE_PATCH), CorrectedDetectorPresence):
        _install_detector_presence(clone)


def _attach_detector_presence(patcher) -> None:
    from comfy.patcher_extension import CallbacksMP
    if not isinstance(patcher.object_patches.get(PRESENCE_PATCH), CorrectedDetectorPresence):
        _install_detector_presence(patcher)
    if PRESENCE_CALLBACK not in patcher.callbacks.get(CallbacksMP.ON_CLONE, {}):
        patcher.add_callback_with_key(CallbacksMP.ON_CLONE, PRESENCE_CALLBACK, _rebind_detector_presence)


def text_mlps(clip, shipped_only: bool = True) -> list[str]:
    """The names, under the CLIP patcher's model, of the text encoder's MLPs: those BUILT with the shipped activation, or all.

    "Built with" is read under any patch: the encoder is one module shared by every clone, and while a corrected
    clone is loaded its patch is what `mlp.activation` shows. The activation the module was built with is then in the
    patchers' shared backup, which is read first.
    """
    import comfy.clip_model
    shipped = comfy.clip_model.ACTIVATIONS[SHIPPED_ACTIVATION]
    backup = clip.patcher.object_patches_backup
    return [name for name, m in clip.patcher.model.named_modules()
            if isinstance(m, comfy.clip_model.CLIPMLP) and (backup.get(name + ".activation", m.activation) is shipped or not shipped_only)]


def attach(
    model,
    clip,
    image_range: bool = True,
    text_activation: bool = True,
    memory_selection: bool = True,
    multimask_ptr: bool = False,
    detector_presence: bool = False,
) -> None:
    """Attach the chosen corrections to a SAM 3 pair's patchers, in place. Idempotent: a pair that has one keeps it.

    Memory selection and multimask propagation pointer token selection go on SAM 3.1's tracker only
    (`MEMORY_SELECTION_TRACKERS` / `MULTIMASK_PTR_TRACKERS`); any other tracker, or none, is left as ComfyUI builds
    it, and `corrections` and the report say which.
    """
    from comfy.patcher_extension import CallbacksMP
    if image_range and not corrections(model, None)["image_range"]:
        _attach_range(model)
        model.add_callback_with_key(CallbacksMP.ON_CLONE, "h3_sam31_range", _rebind_range)
    if memory_selection and tracker_kind(model) in MEMORY_SELECTION_TRACKERS:
        attach_memory_selection(model, focus_obj_idx=None, one_object_only=False)
    if multimask_ptr and tracker_kind(model) in MULTIMASK_PTR_TRACKERS:
        _attach_multimask_ptr(model)
    if detector_presence and model is not None:
        _attach_detector_presence(model)
    if text_activation and clip is not None:
        if not text_mlps(clip, shipped_only=False):
            raise NotSAM3("SAM 3.1 Corrections: this checkpoint's text encoder has none of ComfyUI's CLIP MLPs; it is not "
                          "ComfyUI's SAM 3 text encoder.")
        for name in text_mlps(clip):      # none when ComfyUI already builds the encoder with exact GELU: nothing to do
            clip.patcher.add_object_patch(name + ".activation", F.gelu)


def corrections(model, clip) -> dict:
    """What the patchers of a SAM 3 pair will apply, read off them.

    `image_range`: the range patch is attached. `frames_arrive_mapped`: it is attached and has seen frames that were in
    -1..1 already, so it maps nothing. `text_activation_layers`: MLPs it switches to exact GELU.
    `text_activation_already_exact`: MLPs the encoder already runs with exact GELU, as built. `memory_selection`: the
    tracker's memory patch is attached. `multimask_ptr`: the winning multimask branch token pointer patch is attached.
    `detector_presence`: the detector presence gating patch is attached. Every value is false on a stock pair.
    """
    out = {
        "image_range": False,
        "frames_arrive_mapped": False,
        "text_activation_layers": 0,
        "text_activation_already_exact": 0,
        "memory_selection": False,
        "multimask_ptr": False,
        "detector_presence": False,
    }
    if model is not None:
        patch = model.object_patches.get(RANGE_PATCH)
        out["image_range"] = isinstance(patch, CorrectedRange)
        out["frames_arrive_mapped"] = bool(out["image_range"] and patch.arrives_mapped)
        out["memory_selection"] = isinstance(model.object_patches.get(MEMORY_PATCH), MemorySelectionStep)
        out["multimask_ptr"] = isinstance(model.object_patches.get(PROPAGATION_PATCH), CorrectedPropagationPtr)
        out["detector_presence"] = isinstance(model.object_patches.get(PRESENCE_PATCH), CorrectedDetectorPresence)
    if clip is not None:
        out["text_activation_layers"] = sum(1 for k, v in clip.patcher.object_patches.items() if k.endswith(".activation") and v is F.gelu)
        out["text_activation_already_exact"] = len(text_mlps(clip, shipped_only=False)) - len(text_mlps(clip))
    return out


def report(model, clip) -> str:
    """What the pair will do, in the words a user reads."""
    got = corrections(model, clip)
    lines = ["SAM 3.1, as loaded by the checkpoint loader wired in:"]
    lines.append("image range: frames arrive already in -1..1, so nothing is mapped (ComfyUI's nodes or the graph map them)" if got["frames_arrive_mapped"]
                 else "image range: corrected at the model's first layer (clamped to 0..1, mapped to -1..1)" if got["image_range"]
                 else "image range: NOT corrected (the model gets 0..1, as ComfyUI's SAM nodes hand it over)")
    lines.append(f"text encoder activation: corrected to exact GELU in {got['text_activation_layers']} layers" if got["text_activation_layers"]
                 else f"text encoder activation: already exact GELU as ComfyUI builds it ({got['text_activation_already_exact']} layers); nothing to correct"
                 if got["text_activation_already_exact"]
                 else f"text encoder activation: NOT corrected ({SHIPPED_ACTIVATION}, ComfyUI's setting)")
    if model is not None:
        kind = tracker_kind(model)
        lines.append(f"memory selection: on (threshold {MF_THRESHOLD:g}; {kind}); single-object tracks keep frames where "
                     f"the object was seen, and multi-object tracks run Decoupled Multiplex Memory Selection "
                     f"(write-time slot gate + any-visible spatial union + per-object pointers)" if got["memory_selection"]
                     else f"memory selection: NOT applied to {kind}: it was measured as a win on SAM 3.1 only; the tracker "
                          "keeps ComfyUI's memory of the last six frames" if kind and kind not in MEMORY_SELECTION_TRACKERS
                     else "memory selection: NOT applied (the model has no video tracker)" if not kind
                     else "memory selection: off (the video tracker keeps ComfyUI's memory of the last six frames, seen or not)")
        lines.append(f"propagation pointer token: corrected to winning argmax(iou) branch ({kind})" if got["multimask_ptr"]
                     else f"propagation pointer token: NOT applied to {kind} (SAM 3.1 only)" if kind and kind not in MULTIMASK_PTR_TRACKERS
                     else "propagation pointer token: NOT applied (the model has no video tracker)" if not kind
                     else "propagation pointer token: NOT corrected (ComfyUI projects branch 0 token)")
        lines.append("detector presence head: corrected (folds sigmoid(presence) into logits in float32)" if got["detector_presence"]
                     else "detector presence head: NOT corrected (ComfyUI drops global presence score)")
    lines.append("left as ComfyUI computes them: " + "; ".join(NOT_CORRECTED))
    return "\n".join(lines)


def corrected(
    model,
    clip,
    image_range: bool = True,
    text_activation: bool = True,
    memory_selection: bool = True,
    multimask_ptr: bool = False,
    detector_presence: bool = False,
):
    """Clones of a SAM 3 pair with the chosen corrections attached; the pair handed in is not touched. Returns (model, clip)."""
    model, clip = model.clone(), clip.clone()
    attach(
        model,
        clip,
        image_range=image_range,
        text_activation=text_activation,
        memory_selection=memory_selection,
        multimask_ptr=multimask_ptr,
        detector_presence=detector_presence,
    )
    return model, clip


class MiniMaxH3SAM31Corrections(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SAM31Corrections",
            display_name="MiniMax H3 SAM 3.1 Corrections",
            category="model/latent/minimax",
            description=(
                "Corrects ComfyUI's departures from Meta's SAM 3.1: image range, text encoder activation, decoupled "
                "per-object / multiplex tracker memory selection, and optional multimask pointer token and detector "
                "presence gating. Wire a SAM 3.1 checkpoint's model and text encoder in, and use the outputs wherever "
                "a SAM 3 node takes them. The report says what is corrected."),
            inputs=[
                io.Model.Input("segmenter", tooltip="The SAM 3.1 checkpoint's model, from any checkpoint loader."),
                io.Clip.Input("segmenter_clip", tooltip="The SAM 3.1 checkpoint's text encoder, from the same loader."),
                io.Boolean.Input("correct_image_range", default=True,
                                 tooltip=("Give the model each frame in the range it was trained on. Off: frames reach "
                                          "it as ComfyUI's SAM nodes hand them over.")),
                io.Boolean.Input("correct_text_activation", default=True,
                                 tooltip=("Run the text encoder with the activation Meta built it with. It changes "
                                          "little for one-word phrases such as `person`. Off: ComfyUI's setting.")),
                io.Boolean.Input("memory_selection", default=True, optional=True,
                                 tooltip=("Keep as memory only past frames where objects were reliably seen (Meta's "
                                          "rule, threshold 0.01): single-object tracks keep Meta's per-object selection "
                                          "unchanged, and multi-object calls apply Decoupled Multiplex Memory Selection "
                                          "(write-time slot gate + any-visible spatial union + per-object pointers). "
                                          "SAM 3.1 only; off keeps ComfyUI's last six frames. Absent defaults to on.")),
                io.Boolean.Input("correct_multimask_ptr", default=False, optional=True,
                                 tooltip=("Project the winning argmax(iou) mask token into SAM 3.1's `obj_ptr` instead "
                                          "of hardcoded branch 0 during propagation. Optional, off by default.")),
                io.Boolean.Input("correct_detector_presence", default=False, optional=True,
                                 tooltip=("Multiply `sigmoid(presence)` into SAM3Detector prompt logits once in "
                                          "float32 to gate absent concepts in `SAM3_Detect` and `SAM3_VideoTrack`. "
                                          "Optional, off by default.")),
            ],
            outputs=[io.Model.Output(display_name="segmenter", tooltip="The SAM 3.1 model, for any SAM 3 node."),
                     io.Clip.Output(display_name="segmenter_clip", tooltip="The SAM 3.1 text encoder, for any SAM 3 node."),
                     io.String.Output(display_name="report", tooltip="What is corrected and what is not.")],
        )

    @classmethod
    def execute(
        cls,
        segmenter,
        segmenter_clip,
        correct_image_range=True,
        correct_text_activation=True,
        memory_selection=MEMORY_SELECTION_WHEN_ABSENT,
        correct_multimask_ptr=MULTIMASK_PTR_WHEN_ABSENT,
        correct_detector_presence=DETECTOR_PRESENCE_WHEN_ABSENT,
    ) -> io.NodeOutput:
        model, clip = corrected(
            segmenter,
            segmenter_clip,
            image_range=bool(correct_image_range),
            text_activation=bool(correct_text_activation),
            memory_selection=bool(memory_selection),
            multimask_ptr=bool(correct_multimask_ptr),
            detector_presence=bool(correct_detector_presence),
        )
        text = report(model, clip)
        logger.info("[h3] MiniMaxH3SAM31Corrections: %s", text.replace("\n", "; "))
        return io.NodeOutput(model, clip, text)
