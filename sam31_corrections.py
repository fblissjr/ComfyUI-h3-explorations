"""Three of ComfyUI's departures from Meta's SAM 3.1 code, corrected as ComfyUI patches on a loaded model and text encoder.

ComfyUI's SAM 3.1 (`comfy/ldm/sam3`, `comfy_extras/nodes_sam3.py`) is a port. Run beside Meta's code on equal inputs it
computes the same thing from the trunk on, and departs in two places before the model
(`bench/results/2026-10-07_sam3_core_against_meta.md`; `docs/research/masking/2026-10-07_sam31_in_comfyui.md` walks through both),
and in one in the video tracker:

1. **The image range.** ComfyUI's SAM nodes resize a frame and hand it to the trunk in 0 to 1. Meta's image processor
   and its list-of-images route hand the trunk -1 to 1, and the patch embedding has no bias, so nothing absorbs it.
2. **The text encoder's activation.** ComfyUI's config runs `quick_gelu`; Meta built the encoder with exact GELU.
3. **The tracker's memory** (`memory_selection`, on by default, at the owner's word of 2026-10-17). ComfyUI's tracker
   remembers the last six frames whatever they hold; Meta's keeps only past frames where the object was seen. Applied
   to SAM 3.1's tracker only: on SAM 3 it was not measured as a win, and the report says it is left off there. And
   only to a call that tracks one object (the chain's Subject Track tracks one at a time): a call tracking several, from
   the frame a second object is there, keeps ComfyUI's memory. How it works and what it was measured on is with its
   code, below ("memory selection").

**One node, and not a loader.** It takes the MODEL and CLIP any checkpoint loader returned (ComfyUI's own, or another
pack's) and returns clones of them carrying object patches: one for each correction, two for the memory. Nothing is
loaded, cached, moved to a device or accounted for here: ComfyUI's patcher applies the patches when it loads the model for a call and puts the stock
attributes back when it unloads it, so the pair that was wired in stays stock, ComfyUI's SAM nodes take the corrected
pair like any other, and its memory management owns both. Wired twice in a row it is a no-op: what is attached is read
off the patchers themselves (`corrections`), never guessed from a frame's values.

**What it does not correct**, so "corrected" is not read as "the same as Meta's": the resize (ComfyUI's detect node
shrinks without smoothing), the detection rule (ComfyUI keeps on the class score alone, Meta on class times presence,
with overlapping detections removed), the detect node's refinement passes, the hole-fill value, the pointer token, the
rules between tracked objects and the order of threshold and resize on output.

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
NOT_CORRECTED = ("the resize", "the detection rule (class score alone, no overlap removal)", "the detect node's refinement passes",
                 "the hole-fill value", "the pointer token", "the rules between tracked objects", "threshold before resize on output")


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
#: the two attributes patched on it (`comfy/ldm/sam3/tracker.py`). `bench/check_sam31_corrections.py` reads them off core.
TRACKER = "diffusion_model.tracker"
DECODER_PATCH = TRACKER + ".sam_mask_decoder.forward"
MEMORY_PATCH = TRACKER + "._prepare_memory_conditioned_features"
MEMORY_CALLBACK = "h3_sam31_memory_selection"
#: reasoned: the input is optional, so a graph queued before it existed (a saved job graph, a re-queued render) still
#: validates (found by the server queue test on 2026-10-18: as a required input, every graph without it was refused,
#: "Required input is missing"). Absent means the widget's default, on, so the UI and the API agree
#: (`bench/check_schema_defaults.py`); an old graph re-queued therefore tracks with it on, not as it was first run.
MEMORY_SELECTION_WHEN_ABSENT = True
#: measured: the tracker class memory selection was measured as a win on (2026-10-17, above). The node attaches it to
#: this class only; `attach_memory_selection` takes any SAM 3 tracker, for the bench tool that measures both.
MEMORY_SELECTION_TRACKERS = ("SAM31Tracker",)


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


class IoUTap:
    """Wraps `sam_mask_decoder.forward` to record the latest `iou_pred` and `object_score_logits`."""

    def __init__(self, orig_forward: Any) -> None:
        self.orig_forward = orig_forward
        self.last_iou_pred: torch.Tensor | None = None
        self.last_obj_score_logits: torch.Tensor | None = None
        # off after a pass's last frame (`MemorySelectionStep.end_of_pass`), so the decoder's output on that frame, which
        # no later frame reads, is not held on the card; on again when the next pass starts (`reset_state`)
        self.armed = True

    def reset(self) -> None:
        self.last_iou_pred = None
        self.last_obj_score_logits = None

    def record(self, iou_pred: torch.Tensor, obj_score_logits: torch.Tensor | None) -> None:
        if not self.armed:
            return
        self.last_iou_pred = iou_pred.detach()
        if obj_score_logits is not None:
            self.last_obj_score_logits = obj_score_logits.detach()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self.orig_forward(*args, **kwargs)
        if isinstance(out, tuple) and len(out) >= 2:
            self.record(out[1], out[3] if len(out) >= 4 else None)
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
        record: bool = False,
        one_object_only: bool = False,
    ) -> None:
        self.tracker = tracker
        self.orig_fn = orig_fn
        self.iou_tap = iou_tap
        self.mf_threshold = float(mf_threshold)
        self.focus_obj_idx = int(focus_obj_idx)
        # `record`: keep each pass's scores and kept frames after the next pass starts (`merged_*`), for the bench tool,
        # which clears them before every track. Off (the node): a pass's record is dropped when the next one starts, so
        # nothing grows over the patcher's life.
        self.record = bool(record)
        # `one_object_only` (the node): selection runs on a call that tracks ONE object. Where a call tracks two or more
        # (several seeds, or a detection adding one mid-video, from that frame to the end of the pass), core's memory is
        # used as it is, unfiltered: the multi-object remap takes every object's spatial memory from one focus object,
        # so a hidden object 0 would strip the others' memory to its own few kept frames (a blind read, 2026-10-17).
        # The bench tool measures with an explicit focus and leaves this off.
        self.one_object_only = bool(one_object_only)
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
        # the frame from which this pass tracks several objects and keeps core's memory (`one_object_only`), else None
        self.several_from: int | None = None
        # set by `end_of_pass`: the next call starts a new pass whatever its output_dict's id or first frame
        self._pass_ended = False
        self.iou_tap.reset()
        self.iou_tap.armed = True

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
        def rows(x: torch.Tensor) -> torch.Tensor:
            x = x.detach()
            return x.reshape(-1, x.shape[-1]) if x.dim() >= 3 else x.unsqueeze(-1) if x.dim() == 1 else x

        logits_r, iou_r = rows(raw_logits), rows(raw_iou)
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
                # no object reads a pointer at this rank: the focus object's spatial memory alone
                if spatial_out is not None:
                    remapped[target_key] = {key: v for key, v in spatial_out.items() if key != "obj_ptr"}
                continue
            if n_obj == 1 or multiplex_state is None or (
                spatial_out is not None and all(src == spatial_src for src in ptr_src)
            ):
                # one object, or every object reads this rank's pointer from the focus object's frame
                if spatial_out is not None:
                    remapped[target_key] = spatial_out
                continue

            # Multi-object multiplex where objects diverged in valid_indices at rank -k:
            # Spatial memory comes from `focus_obj` (if available at rank -k), while
            # each object slot (b = o // M, m = o % M) in `obj_ptr` comes from its own
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

            template_ptr: torch.Tensor | None = None
            for s in ptr_src:
                if s is not None:
                    cand = self._lookup_frame(s, non_cond_outputs)
                    if cand is not None and "obj_ptr" in cand:
                        template_ptr = cand["obj_ptr"]
                        break

            if template_ptr is not None:
                combined_ptr = torch.zeros_like(template_ptr)
                for obj_idx, src_f in enumerate(ptr_src):
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
                is_new_output_dict_in_run = self._pass_ended or (self._output_dict_id is not None and (
                    id(output_dict) != self._output_dict_id or frame_idx < self._last_frame_idx
                ))
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

        n_now = int(getattr(multiplex_state, "total_valid_entries", 1)) if multiplex_state is not None else 1
        if self.one_object_only and (self.several_from is not None or n_now >= 2):
            if self.several_from is None:
                self.several_from = frame_idx
                self.retained_bank.clear()
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
        self.iou_tap.reset()
        self.iou_tap.armed = False
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


def _install_memory(patcher, mf_threshold: float, focus_obj_idx: int, record: bool = False,
                    one_object_only: bool = False) -> MemorySelectionStep:
    """The two object patches, built on the class's methods bound to THIS patcher's tracker, never on a wrapper."""
    import comfy.utils
    try:
        tracker = comfy.utils.get_attr(patcher.model, TRACKER)
    except AttributeError as exc:
        raise NotSAM3(f"SAM 3.1 Corrections: this checkpoint's model has no `{TRACKER}`; it is not a SAM 3 checkpoint.") from exc
    decoder = tracker.sam_mask_decoder
    tap = IoUTap(type(decoder).forward.__get__(decoder, type(decoder)))
    step = MemorySelectionStep(tracker=tracker, orig_fn=type(tracker)._prepare_memory_conditioned_features.__get__(tracker, type(tracker)),
                               iou_tap=tap, mf_threshold=mf_threshold, focus_obj_idx=focus_obj_idx, record=record,
                               one_object_only=one_object_only)
    patcher.add_object_patch(DECODER_PATCH, tap)
    patcher.add_object_patch(MEMORY_PATCH, step)
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


def attach_memory_selection(patcher, mf_threshold: float = MF_THRESHOLD, focus_obj_idx: int = 0,
                            record: bool = False, one_object_only: bool = False) -> MemorySelectionStep:
    """Attach memory selection to a SAM 3 or SAM 3.1 patcher, in place. Idempotent: a patcher that carries it keeps it.

    `focus_obj_idx`: in a call tracking several objects, the object whose kept frames give the spatial memory where the
    objects' kept frames differ (each object's pointer always comes from its own). A patcher that already carries
    memory selection with another threshold or focus is refused, never silently kept or replaced. `record`: keep every
    pass's scores and kept frames until `reset_state(clear_runs=True)` (the bench tool's tracks); the node keeps none.
    `one_object_only`: selection on a call tracking one object only, core's memory where a call tracks several (the node)."""
    from comfy.patcher_extension import CallbacksMP
    step = patcher.object_patches.get(MEMORY_PATCH)
    if isinstance(step, MemorySelectionStep):
        if (step.mf_threshold, step.focus_obj_idx) != (float(mf_threshold), int(focus_obj_idx)):
            raise ValueError(f"this model already carries memory selection at mf_threshold {step.mf_threshold}, focus "
                             f"{step.focus_obj_idx}; asked for {float(mf_threshold)}, {int(focus_obj_idx)}. Attach to the model "
                             "before it was given one.")
        return step
    step = _install_memory(patcher, float(mf_threshold), int(focus_obj_idx), bool(record), bool(one_object_only))
    if MEMORY_CALLBACK not in patcher.callbacks.get(CallbacksMP.ON_CLONE, {}):
        patcher.add_callback_with_key(CallbacksMP.ON_CLONE, MEMORY_CALLBACK, _rebind_memory)
    return step


def apply_memory_selection(model, mf_threshold: float = MF_THRESHOLD, focus_obj_idx: int = 0, record: bool = True):
    """A clone of a SAM 3 or SAM 3.1 patcher with memory selection attached; the patcher handed in is not touched.

    Records every pass by default: the bench tool reads the merged record of a track's passes after it returns."""
    m = model.clone()
    attach_memory_selection(m, mf_threshold=mf_threshold, focus_obj_idx=focus_obj_idx, record=record)
    return m


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


def attach(model, clip, image_range: bool = True, text_activation: bool = True, memory_selection: bool = True) -> None:
    """Attach the chosen corrections to a SAM 3 pair's patchers, in place. Idempotent: a pair that has one keeps it.

    Memory selection goes on SAM 3.1's tracker only (`MEMORY_SELECTION_TRACKERS`); any other tracker, or none, is left
    as ComfyUI builds it, and `corrections` and the report say which."""
    from comfy.patcher_extension import CallbacksMP
    if image_range and not corrections(model, None)["image_range"]:
        _attach_range(model)
        model.add_callback_with_key(CallbacksMP.ON_CLONE, "h3_sam31_range", _rebind_range)
    if memory_selection and tracker_kind(model) in MEMORY_SELECTION_TRACKERS:
        attach_memory_selection(model, one_object_only=True)
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
    tracker's memory patch is attached. Every value is false on a stock pair (`tracker_kind` names the tracker).
    """
    out = {"image_range": False, "frames_arrive_mapped": False, "text_activation_layers": 0, "text_activation_already_exact": 0,
           "memory_selection": False}
    if model is not None:
        patch = model.object_patches.get(RANGE_PATCH)
        out["image_range"] = isinstance(patch, CorrectedRange)
        out["frames_arrive_mapped"] = bool(out["image_range"] and patch.arrives_mapped)
        out["memory_selection"] = isinstance(model.object_patches.get(MEMORY_PATCH), MemorySelectionStep)
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
        lines.append(f"memory selection: on for one-object tracks (the video tracker keeps as memory only past frames where "
                     f"the object was seen, Meta's rule, threshold {MF_THRESHOLD:g}; {kind}); a call tracking several objects keeps "
                     "ComfyUI's memory" if got["memory_selection"]
                     else f"memory selection: NOT applied to {kind}: it was measured as a win on SAM 3.1 only; the tracker "
                          "keeps ComfyUI's memory of the last six frames" if kind and kind not in MEMORY_SELECTION_TRACKERS
                     else "memory selection: NOT applied (the model has no video tracker)" if not kind
                     else "memory selection: off (the video tracker keeps ComfyUI's memory of the last six frames, seen or not)")
    lines.append("left as ComfyUI computes them: " + "; ".join(NOT_CORRECTED))
    return "\n".join(lines)


def corrected(model, clip, image_range: bool = True, text_activation: bool = True, memory_selection: bool = True):
    """Clones of a SAM 3 pair with the chosen corrections attached; the pair handed in is not touched. Returns (model, clip)."""
    model, clip = model.clone(), clip.clone()
    attach(model, clip, image_range=image_range, text_activation=text_activation, memory_selection=memory_selection)
    return model, clip


class MiniMaxH3SAM31Corrections(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SAM31Corrections",
            display_name="MiniMax H3 SAM 3.1 Corrections",
            category="model/latent/minimax",
            description=(
                "Corrects three things ComfyUI's SAM 3.1 does differently from Meta's code: the range of the image the "
                "model is given, the text encoder's activation, and which past frames the video tracker remembers. Wire a "
                "SAM 3.1 checkpoint's model and text encoder in, and use the outputs wherever a SAM 3 node takes them. "
                "The report says what is corrected."),
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
                                 tooltip=("When tracking one object through a video, remember only past frames where "
                                          "it was seen (Meta's rule), so frames where it was hidden do not fill the "
                                          "tracker's memory. On for one-object tracks; a call tracking several objects "
                                          "keeps ComfyUI's memory. SAM 3.1 only; a SAM 3 model is left as it is. Off: "
                                          "ComfyUI's memory of the last six frames, seen or not. A graph saved before "
                                          "this input existed runs with it on.")),
            ],
            outputs=[io.Model.Output(display_name="segmenter", tooltip="The SAM 3.1 model, for any SAM 3 node."),
                     io.Clip.Output(display_name="segmenter_clip", tooltip="The SAM 3.1 text encoder, for any SAM 3 node."),
                     io.String.Output(display_name="report", tooltip="What is corrected and what is not.")],
        )

    @classmethod
    def execute(cls, segmenter, segmenter_clip, correct_image_range=True, correct_text_activation=True,
                memory_selection=MEMORY_SELECTION_WHEN_ABSENT) -> io.NodeOutput:
        model, clip = corrected(segmenter, segmenter_clip, image_range=bool(correct_image_range),
                                text_activation=bool(correct_text_activation), memory_selection=bool(memory_selection))
        text = report(model, clip)
        logger.info("[h3] MiniMaxH3SAM31Corrections: %s", text.replace("\n", "; "))
        return io.NodeOutput(model, clip, text)
