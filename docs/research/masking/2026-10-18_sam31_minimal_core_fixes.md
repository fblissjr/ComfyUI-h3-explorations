# 2026-10-18: Minimal Upstream Fixes for SAM 3 / 3.1 in ComfyUI Core

**In short (as of 2026-10-18).** If SAM 3 and 3.1 were fixed in ComfyUI core itself, the smallest set is six fixes of about 25 lines in four files: map input images to -1 to 1 (the largest single error), gate detector scores by presence, use exact GELU in the text encoder (submitted as Comfy-Org/ComfyUI#16855), stop the single-prompt path tokenizing a literal count suffix, turn a boolean attention mask into an additive one, and gate tracker memory by quality as Meta does. They would not cover multi-shot and multi-person tracking, which this pack handles in its own nodes. This repo still patches nothing in core; the pack's runtime corrections are mapped in [`../../masked_v2v/sam31_core.md`](../../masked_v2v/sam31_core.md).

Last updated: 2026-10-18. Written against ComfyUI core (`comfy/text_encoders/sam3_clip.py`, `comfy_extras/nodes_sam3.py`, `comfy/ldm/sam3/{sam,detector,tracker}.py`) and Meta's reference (`coderef/sam3`). Companion to [`docs/masked_v2v/sam31.md`](../../masked_v2v/sam31.md), [`2026-10-07_mryolk.md`](2026-10-07_mryolk.md), and [`2026-10-07_mryolk_stage_table.md`](2026-10-07_mryolk_stage_table.md).

---

## 1. Scope and Goal

In this repository, we leave ComfyUI core untouched (`bench/check_vsa_core_patch.py`) and apply SAM 3.1 fixes through two layers:
1. **Runtime `ModelPatcher` corrections** in `sam31_corrections.py::MiniMaxH3SAM31Corrections` (`sam31_corrections.py:455-530`) and `bench/sam_memory_selection.py` (`bench/sam_memory_selection.py:1-306`).
2. **A shot-aware detection, box-prompted propagation, and track arbitration pipeline** in `shot_table.py`, `subject_track.py`, `subject_tracks.py`, and `subject_boxes.py`.

If we instead wanted to fix SAM 3 / 3.1 **directly in ComfyUI core with the smallest possible footprint**—no new nodes and no architectural rewrite, just making core's existing `SAM3_Detect`, `SAM3_Segment`, `SAM3_VideoTrack`, and `SAM3_VideoSegment` match Meta's reference math and weights—there are **6 surgical fixes (~25 lines across 4 files)**.

---

## 2. Summary of the 6 Minimal Core Fixes

| Rank | Issue in ComfyUI Core | Core File & Lines | Meta Reference File & Lines | Measured Effect / Why It Matters |
|---|---|---|---|---|
| **1** | Input images fed in `[0, 1]` (plus unclamped bicubic overshoot on video) instead of `[-1, 1]` into bias-free `ViTDet.patch_embed` | `comfy_extras/nodes_sam3.py:65, 118`, `comfy/ldm/sam3/tracker.py:207-209`, `comfy/ldm/sam3/sam.py:217`, `comfy/ldm/sam3/detector.py:113` | `coderef/sam3/sam3/model_builder.py:628`, `coderef/sam3/sam3/model/sam3_video_base.py:869`, `coderef/sam3/sam3/model/sam3_image_processor.py:26` | **Largest single error**: `0.65` relative L2 (`max_abs = 8.15`) on vision trunk features ([`2026-10-07_sam3_core_against_meta.md`](../../../bench/results/2026-10-07_sam3_core_against_meta.md)); shifts detection scores by up to `0.63` on multi-word prompts (`human hair`, [`2026-10-10_sam31_corrections_on_one_stretch.md`](../../../bench/results/2026-10-10_sam31_corrections_on_one_stretch.md)). |
| **2** | Detector `presence` logit (`pred_logit.sigmoid()`) ignored when scoring queries | `comfy/ldm/sam3/detector.py:213, 492, 495-509`, `comfy_extras/nodes_sam3.py:217`, `comfy/ldm/sam3/tracker.py:1697` | `coderef/sam3/sam3/model/sam3_image_processor.py:168-169`, `coderef/sam3/sam3/model/sam3_video_base.py:854` | Raw `pred_logits.sigmoid()` fires false-positive boxes when the prompted concept is absent from the frame (`presence = 0.032` on `"woman in dark dress"`, [`2026-10-07_sam3_core_against_meta.md`](../../../bench/results/2026-10-07_sam3_core_against_meta.md)). |
| **3** | `SAM3_CLIP_CONFIG` uses `"quick_gelu"` instead of `"gelu"` across all 24 text encoder layers | `comfy/text_encoders/sam3_clip.py:6` | `coderef/sam3/sam3/model/text_encoder_ve.py:130, 198` | Accumulates `4.8%` to `26%` relative L2 drift (`0.056` after `resizer`, `max_abs = 0.453`) across the 24-layer `VETextEncoder`; `"gelu"` brings text features to float32 parity (`2.0e-7`). Submitted upstream in Comfy-Org/ComfyUI#16855. |
| **4** | Single-prompt fast path in `SAM3TokenizerWrapper.tokenize_with_weights` passes raw `text` instead of `parsed[0][0]` | `comfy/text_encoders/sam3_clip.py:30-44, 51-54` | `coderef/sam3/sam3/model/tokenizer_ve.py:239-257` | Writing `"person:1"` or `"(person)"` satisfies `len(parsed) <= 1 and parsed[0][1] == 1` and tokenizes literal `:1` (`2533, 281, 272`) or `()` into the prompt instead of `"person"`. |
| **5** | `SplitMHA.forward` passes a boolean `text_mask` (`True` = valid, `False` = pad) into `optimized_attention` | `comfy/ldm/sam3/detector.py:54-57, 478` | `coderef/sam3/sam3/model/decoder.py:224-273` | Non-SDPA ComfyUI attention backends (`sub_quadratic_attention`, `split_attention`, `--cpu` fallback) add `bool` masks directly to attention weights (`+1` on valid, `+0` on pad) or invert them, letting vision queries attend to all 29 padding tokens. |
| **6** | `SAM31Tracker` and `SAM3Tracker` feed the last 6 consecutive non-conditioning frames (`t_pos = 1..6`) unconditionally and evict `non_cond_frame_outputs` after 16 frames without `cal_mem_score` / `frame_filter` quality gating | `comfy/ldm/sam3/tracker.py:227-269, 1217-1296, 1755-1758` | `coderef/sam3/sam3/model/video_tracking_multiplex.py:2109-2142`, `coderef/sam3/sam3/model/sam3_multiplex_base.py:482-502` | Filtering past frames by `eff_iou_score > mf_threshold` (`0.01`) and retaining valid pre-occlusion memories across 16+ frame occlusions recovers the lead 5 frames earlier after occlusion (`710` vs `715` on `lead_fp_0613g`) and holds through occlusions at `95-116` and `130-140` on `lotsofpeople` (`data/2026-10-17_sam_memory_selection/`, [`docs/state/decisions.md`](../../state/decisions.md)). |

---

## 3. Detailed Breakdown & Minimal Diffs

### Fix 1: Scale input images to `[-1, 1]` and clamp bicubic overshoot

- **What ComfyUI core does**:
  - `SAM3_PrepareImage.execute` (`comfy_extras/nodes_sam3.py:65`) and `SAM3_PrepareVideo.execute` (`comfy_extras/nodes_sam3.py:118`) resize ComfyUI's `[0, 1]` `IMAGE` tensor to `1008x1008` and pass it directly to the vision backbone without shifting/scaling to `[-1, 1]`.
  - `Sam3TrackerModel.prepare_video` (`comfy/ldm/sam3/tracker.py:207-209`) resizes video frames to `1008x1008` with `F.interpolate(..., mode="bicubic", align_corners=False)` and no clamp or range shift, producing pixel values in `[-0.08, 1.09]`.
  - `ViTDet.patch_embed` (`comfy/ldm/sam3/detector.py:113`, called in `Sam3Image.forward_image` at `comfy/ldm/sam3/sam.py:217`) is a `Conv2d(3, 1024, kernel_size=14, stride=14, bias=False)` with **no bias term** to absorb a mean shift.
- **What Meta's reference does**:
  - Normalizes `[0, 1]` images with `mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]` (`(x - 0.5) / 0.5 = x * 2.0 - 1.0`) in `coderef/sam3/sam3/model_builder.py:628`, `coderef/sam3/sam3/model/sam3_video_base.py:869`, and `coderef/sam3/sam3/model/sam3_image_processor.py:26`.
- **Minimal core change**:
  In `comfy/ldm/sam3/sam.py:217` (`Sam3Image.forward_image`, or at the prepare call sites `comfy_extras/nodes_sam3.py:65, 118` and `comfy/ldm/sam3/tracker.py:207-209`):
  ```python
  img = img.clamp(0.0, 1.0) * 2.0 - 1.0
  ```
- **Why**:
  - Removes the `0.65` relative L2 (`max_abs = 8.15`) error on vision trunk features ([`2026-10-07_sam3_core_against_meta.md`](../../../bench/results/2026-10-07_sam3_core_against_meta.md)).
  - In our runtime node (`sam31_corrections.py:267-318`), this is implemented via a `register_forward_pre_hook` on `vision_backbone.trunk.patch_embed`.

---

### Fix 2: Gate detector query scores by `presence = pred_logit.sigmoid()`

- **What ComfyUI core does**:
  - `Sam3Detector.forward` computes `presence_out` at `comfy/ldm/sam3/detector.py:213` and returns `"presence": presence_out[-1]` at `comfy/ldm/sam3/detector.py:492`.
  - However, `SAM3_Detect.execute` (`comfy_extras/nodes_sam3.py:217`) computes `probs = outputs["pred_logits"].sigmoid()` and ignores `outputs["presence"]`.
  - Worse, `Sam3Detector.detect` (`comfy/ldm/sam3/detector.py:495-509`, called by `Sam3VideoTracker.track_video_with_detection` at `comfy/ldm/sam3/tracker.py:1697`) calls `self.forward(...)` without using `"presence"` at all:
    ```python
    probs = out["pred_logits"].sigmoid()
    ```
- **What Meta's reference does**:
  - `Sam3Processor._forward_grounding` (`coderef/sam3/sam3/model/sam3_image_processor.py:168-169`) and `Sam3VideoBase.forward_grounding` (`coderef/sam3/sam3/model/sam3_video_base.py:854`) multiply per-query classification probabilities by the image-level presence probability:
    ```python
    presence_score = pred_logit.sigmoid().unsqueeze(1)
    scores = (pred_logits.sigmoid() * presence_score).squeeze(-1)
    ```
- **Minimal core change**:
  In `comfy/ldm/sam3/detector.py:498-500` (`Sam3Detector.detect`) and `comfy_extras/nodes_sam3.py:217` (`SAM3_Detect.execute`):
  ```python
  probs = out["pred_logits"].sigmoid() * out["presence"].sigmoid().unsqueeze(-1)
  ```
- **Why**:
  - `presence` is the learned gate that answers whether the text prompt is in the frame at all. Without it, a prompt for a concept absent from the shot can still produce individual queries with `pred_logits.sigmoid() >= 0.5` (e.g. `"woman in dark dress"` in [`2026-10-07_sam3_core_against_meta.md`](../../../bench/results/2026-10-07_sam3_core_against_meta.md) had a top query logit of `0.50` while `presence` was `0.032`).
  - Furthermore, the largest benefit of fixing `quick_gelu -> gelu` on descriptive multi-token prompts is cutting false-positive `presence` in half (`0.061 -> 0.032`), which only takes effect when `presence` is multiplied into `probs`.

---

### Fix 3: Switch `SAM3_CLIP_CONFIG["hidden_act"]` from `"quick_gelu"` to `"gelu"` (PR #16855)

- **What ComfyUI core does**:
  - `comfy/text_encoders/sam3_clip.py:6` sets `"hidden_act": "quick_gelu"` (`x * sigmoid(1.702 * x)`, `comfy/clip_model.py:78`).
- **What Meta's reference does**:
  - `coderef/sam3/sam3/model/text_encoder_ve.py:130, 198` constructs every `ResidualAttentionBlock` in `VETextEncoder` with `act_layer: Callable = nn.GELU` (exact erf GELU).
- **Minimal core change**:
  In `comfy/text_encoders/sam3_clip.py:6`:
  ```python
  "hidden_act": "gelu",
  ```
- **Why**:
  - Across 24 transformer layers, `quick_gelu` accumulates `4.8%` relative L2 drift on 1-word prompts (`person`) and up to `26%` on 11-token prompts (`0.056` after `language_backbone.resizer` in [`2026-10-07_sam3_core_against_meta.md`](../../../bench/results/2026-10-07_sam3_core_against_meta.md)). Switching to `"gelu"` matches Meta's `VETextEncoder` to float32 noise (`2.0e-7`).

---

### Fix 4: Pass `parsed[0][0]` instead of raw `text` on the single-prompt fast path in `SAM3TokenizerWrapper`

- **What ComfyUI core does**:
  - `comfy/text_encoders/sam3_clip.py:30-44` (`_parse_prompts`) strips parentheses and parses `prompt:N` max-detection counts into `(text_part, max_det)` tuples.
  - `SAM3TokenizerWrapper.tokenize_with_weights` (`comfy/text_encoders/sam3_clip.py:51-54`) then checks:
    ```python
    parsed = _parse_prompts(text)
    if len(parsed) <= 1 and (not parsed or parsed[0][1] == 1):
        return super().tokenize_with_weights(text, return_word_ids, **kwargs)
    ```
  - When `text` is `"person:1"` or `"(person)"`, `_parse_prompts(text)` returns `[("person", 1)]`, which takes the fast path at line 54 and passes raw `text` (`"person:1"`) instead of `"person"`.
- **Minimal core change**:
  In `comfy/text_encoders/sam3_clip.py:53-54`:
  ```python
  if len(parsed) <= 1 and (not parsed or parsed[0][1] == 1):
      return super().tokenize_with_weights(parsed[0][0] if parsed else text, return_word_ids, **kwargs)
  ```
- **Why**:
  - A user who writes `"person:1"` to cap `max_detections` at 1 (as documented by `_parse_prompts`) currently gets `:1` (`2533, 281, 272`) tokenized into the prompt, whereas `"person:2"` goes down lines 56–63 and strips `:2`.

---

### Fix 5: Convert boolean `attn_mask` in `SplitMHA.forward` to an additive float mask

- **What ComfyUI core does**:
  - `Sam3Detector.forward` (`comfy/ldm/sam3/detector.py:478`) builds `text_mask = text_attention_mask.ne(0)` (`dtype=torch.bool`, `True` = valid token, `False` = padding).
  - `SplitMHA.forward` (`comfy/ldm/sam3/detector.py:54-57`) passes `mask=attn_mask` directly to `optimized_attention(q, k, v, self.num_heads, mask=attn_mask, skip_reshape=True, low_precision_attention=False)`.
  - While `optimized_attention_pytorch` (`F.scaled_dot_product_attention`) interprets a `bool` mask as `True = keep, False = mask out`, ComfyUI's non-SDPA attention backends (`optimized_attention_sub_quad`, `optimized_attention_split`, and `--cpu` fallback) add `mask` directly to `sim` (`sim = sim + mask`) or invert it—giving valid tokens `+1` and leaving all 29 padding tokens unmasked (`+0`).
- **Minimal core change**:
  In `comfy/ldm/sam3/detector.py:54-57`:
  ```python
  if attn_mask is not None and attn_mask.dtype == torch.bool:
      attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill_(~attn_mask, -torch.finfo(q.dtype).max)
  out = optimized_attention(q, k, v, self.num_heads, mask=attn_mask, skip_reshape=True, low_precision_attention=False)
  ```
- **Why**:
  - An additive float mask (`0.0` for valid tokens, `-finfo.max` for padding) has unambiguous semantics across SDPA, xFormers, split attention, and sub-quadratic attention on both GPU and CPU.

---

### Fix 6: Filter non-conditioning spatial memory and object pointers with `cal_mem_score` and `frame_filter` (`mf_threshold = 0.01`)

- **What ComfyUI core does**:
  - `collect_memory_tokens` (`comfy/ldm/sam3/tracker.py:227-269`) and `SAM31Tracker._prepare_memory_conditioned_features` (`comfy/ldm/sam3/tracker.py:1217-1296`) unconditionally take the last `num_maskmem - 1` (`6`) consecutive frames `frame_idx - 6 .. frame_idx - 1` (`t_pos = 1..6`) for spatial memory and the last `15` frames for object pointers.
  - `track_video_with_detection` (`comfy/ldm/sam3/tracker.py:1755-1758`) evicts `non_cond_frame_outputs[old_idx]` once `old_idx < frame_idx - 16`.
  - When a subject is occluded or out of frame for 6+ frames, all 6 spatial memory slots fill with empty/occluded masks, and after 16 frames the last good pre-occlusion views are deleted.
- **What Meta's reference does**:
  - `cal_mem_score` (`coderef/sam3/sam3/model/video_tracking_multiplex.py:2109-2142`, `coderef/sam3/sam3/model/video_tracking_multiplex_demo.py:2038-2071`) scores each frame's memory quality from `object_score_logits > 0`, the winning multimask branch's `iou_pred` scaled from `[0.5, 1.0]` to `[0.0, 1.0]`, and an overlap dilution penalty at $126\times 126$.
  - `frame_filter` (`coderef/sam3/sam3/model/sam3_multiplex_base.py:482-502`) walks backward from `frame_idx - 1` to `1` and keeps only past frames whose `eff_iou_score > mf_threshold` (`0.01`, up to `6` spatial memories), and drops the oldest kept frame's object pointer (`pos_and_ptrs[:-1]`) when `len(pos_and_ptrs) > 6`.
- **Minimal core change**:
  In `comfy/ldm/sam3/tracker.py:1217-1296, 1375-1398, 1755-1758`, compute per-object `eff_iou_score` (`cal_mem_score_per_object` in `sam31_corrections.py:222-304`) before discarding `iou_predictions`, retain frames with `eff_iou_score > 0.01` across the 16-frame eviction loop, and filter past frames in `_prepare_memory_conditioned_features` (`sam31_corrections.py:363-745`, `bench/sam_memory_selection.py:67-169`).
- **Why**:
  - Prevents empty or low-IoU occlusion frames (`eff_iou_score <= 0.01`) from filling all 6 spatial memory slots and evicting the last good views of the subject.
  - Measured in `data/2026-10-17_sam_memory_selection/` ([`docs/state/decisions.md`](../../state/decisions.md), 2026-10-17): recovers the lead 5 frames earlier after occlusion (`710` vs `715` on `lead_fp_0613g`) and holds the tracked dancer through occlusions at `95–116` and `130–140` on `lotsofpeople` with 0 identity jumps and no measurable wall-time cost.

---

## 4. What These 6 Minimal Core Fixes Do Not Cover (and Why That Stays in Custom Nodes)

Even with all 6 fixes above applied to ComfyUI core (and even in Meta's own `Sam3VideoPredictor` in `coderef/sam3`), the built-in video tracking loop (`comfy/ldm/sam3/tracker.py:1636-1802`) assumes a **single continuous shot**:
- It has no concept of hard camera cuts (`ShotTable` in `shot_table.py:34-420`), so memory features from before a cut bleed across the cut.
- It enforces a hard 64-object lifetime cap across the entire video (`INTERNAL_MAX_OBJECTS = 64` at `comfy/ldm/sam3/tracker.py:1555`), silently dropping new detections once 64 objects have appeared across cuts.
- Its `keep_alive` counter (`MAX_KEEP_ALIVE = 8`, `MIN_KEEP_ALIVE = -4` at `comfy/ldm/sam3/tracker.py:1602-1603`) and `HIGH_CONF = 0.8` mask overwrite (`comfy/ldm/sam3/tracker.py:1600, 1720-1732`) drop subjects during brief occlusions or overwrite multi-part masks with single-part detections.

Those multi-shot and multi-person problems are architectural rather than single-line parity bugs, which is why our repo solves them in `subject_track.py:1-2033`, `subject_tracks.py:1-709`, and `subject_boxes.py:1-122` (documented in [`docs/masked_v2v/sam31.md`](../../masked_v2v/sam31.md)).
