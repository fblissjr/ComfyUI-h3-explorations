# SAM 3.1 in this pack: the code map, and how each layer differs from ComfyUI core

**Our SAM 3.1 stack is two layers above an unpatched ComfyUI core**: runtime patches on cloned model objects
(`sam31_corrections.py`), and a shot-aware tracker built on core's detector and tracker as primitives
(`subject_track.py`, `subject_tracks.py`, `shot_table.py`, `subject_boxes.py`). This page is the file and line map of
both, what each change did and did not improve, and what is left as core computes it. What to ask SAM and the
overview are [`sam31.md`](sam31.md); the dated studies are [`docs/research/masking/`](../research/masking/).

## Where our SAM 3.1 code lives, and how it relates to ComfyUI core

Nothing in this pack edits or monkey-patches ComfyUI core on disk
(`bench/check_vsa_core_patch.py`). Instead, our SAM 3.1 stack sits in two
layers above core's `comfy/text_encoders/sam3_clip.py`,
`comfy/ldm/sam3/{sam,detector,tracker}.py`,
`comfy/supported_models.py:2510-2559` (`SAM3`, `SAM31`), and
`comfy_extras/nodes_sam3.py`:

| file in this pack | what it owns | what it touches in ComfyUI core |
|---|---|---|
| [`bench/convert_sam3_checkpoint.py`](../../bench/convert_sam3_checkpoint.py) (`bench/convert_sam3_checkpoint.py:1-116`) | Repacks Meta's pickled `sam3.1_multiplex.pt` into `sam3.1_multiplex_fp32.safetensors` (`workflows/h3_config.py::SEGMENTER`, `workflows/h3_config.py:491`) with zero weight changes, verified by `torch.equal` readback (`bench/convert_sam3_checkpoint.py:98-103`). | Drops only the 32 `complex64` `*.attn.freqs_cis` buffers that safetensors cannot store and that `comfy/supported_models.py:2529-2530` (`SAM3.process_unet_state_dict`) deletes at load anyway. |
| [`sam31_corrections.py`](../../sam31_corrections.py) (`sam31_corrections.py:1-873`, `MiniMaxH3SAM31Corrections`) | Runtime patches on cloned `MODEL` and `CLIP` `ModelPatcher` objects (`add_object_patch` + `CallbacksMP.ON_CLONE`) for three departures of ComfyUI core from Meta's SAM 3.1 reference: vision input range, text MLP activation, and per-object tracker memory selection. Guarded by [`bench/check_sam31_corrections.py`](../../bench/check_sam31_corrections.py). | `ViTDet.patch_embed` (`comfy/ldm/sam3/sam.py:214-220`), `SAM3TextModel` MLPs (`comfy/text_encoders/sam3_clip.py:5-19`), and `SAM31Tracker` (`comfy/ldm/sam3/tracker.py:1217-1802`). |
| [`bench/sam_memory_selection.py`](../../bench/sam_memory_selection.py) (`bench/sam_memory_selection.py:1-250`) | Standalone measurement harness for per-object tracker memory selection on both `SAM31Tracker` (including multi-object per-bucket remapping) and `SAM3Tracker`. | `SAM31Tracker` and `SAM3Tracker` (`comfy/ldm/sam3/tracker.py:497-641, 1217-1802`). |
| [`subject_track.py`](../../subject_track.py) (`subject_track.py:1-2033`, `MiniMaxH3SubjectTrack`) | Shot-aware single-person tracker across cuts: cut detection, `:1` tokenizer bypass, direct detector and `ViTDet.trunk` calls, two-view trunk feature matching (`top_third` + `head`), bidirectional seed-only `SAM3_VideoTrack` (`max_objects=1`), `ScoreTap`, mid-shot `regain`, and multi-tracker exclusion (`others`). Guarded by [`bench/check_subject_track.py`](../../bench/check_subject_track.py). | Calls `segmenter.model.detector` (`comfy/ldm/sam3/detector.py:472-493`), `vision_backbone.trunk` (`comfy/ldm/sam3/sam.py:223-295`), `SAM3_VideoTrack.execute` (`comfy_extras/nodes_sam3.py:282-314`), and taps `SAM31Tracker._forward_propagation` (`comfy/ldm/sam3/tracker.py:1351-1398`). |
| [`subject_tracks.py`](../../subject_tracks.py) (`subject_tracks.py:1-709`) | Pure-tensor shared helpers imported by `subject_track.py` without loading ComfyUI nodes: nested-detection deduplication (`one_each`), speck removal (`drop_specks`), track plausibility/trust checks (`plausible`, `doubted`, `trusted`), gap bridging (`unbroken`), pre-loss trim (`take_back`, `take_back_by_place`), occlusion check (`in_the_way`), and note parsing (`parse_notes`, `hand_out`). | Replaces the missing query NMS in `comfy_extras/nodes_sam3.py:217-226` and replaces core's `keep_alive` / `HIGH_CONF` refresh rules in `comfy/ldm/sam3/tracker.py:1578-1643`. |
| [`shot_table.py`](../../shot_table.py) (`shot_table.py:1-250`, `MiniMaxH3SaveShotTable`) | Formats the per-shot JSON audit (`build`, `shot_table.py:86-154`) and saves it beside the numbered preview PNGs (`shot_table.py:188-250`). | Replaces blind index selection in `SAM3_TrackPreview` / `SAM3_TrackToMask` (`comfy_extras/nodes_sam3.py:317-518`). |
| [`subject_boxes.py`](../../subject_boxes.py) (`subject_boxes.py:1-122`, `MiniMaxH3SubjectBoxes`) | Converts a `[N, H, W]` subject mask into `[N, 5]` `(batch_idx, x1, y1, x2, y2)` `BBOX_LIST` for `MiniMaxH3BodyPose`, dropping masks smaller than `SMALLEST_MASK_PX = 2048` (`subject_boxes.py:30-45`). | Feeds `MiniMaxH3BodyPose` (`body_pose.py`) in masked job previews (`workflows/masked_job_graph.py:318-323`). |
| [`video_mask.py`](../../video_mask.py) (`video_mask.py:2028-2041, 2506-2585`, `detect_part`) | When `MiniMaxH3MaskedSource` runs with `replace="head and hair"`, calls `SAM3_Detect.execute` per phrase on frames where the subject is present (`video_mask.py:2028-2041`). | `comfy_extras/nodes_sam3.py::SAM3_Detect.execute` (`comfy_extras/nodes_sam3.py:114-248`). |

---

## Layer 1: Runtime patches on ComfyUI core (`sam31_corrections.py`)

`MiniMaxH3SAM31Corrections` (`sam31_corrections.py:837-873`) sits between
`CheckpointLoaderSimple` and every SAM consumer. It clones the incoming
`MODEL` and `CLIP` `ModelPatcher`s (`sam31_corrections.py:766-816`), registers
its patches via `add_object_patch`, and registers a `CallbacksMP.ON_CLONE`
callback (`sam31_corrections.py:806-812`) because `SAM3_Detect.execute`
(`comfy_extras/nodes_sam3.py:121`), `_refine_mask`
(`comfy_extras/nodes_sam3.py:74`), and `SAM3_VideoTrack.execute`
(`comfy_extras/nodes_sam3.py:293`) each clone the incoming `ModelPatcher` and
call `add_object_patch("diffusion_model", model)` on the clone—which shadows
any sub-attribute patch registered on `diffusion_model.*` unless re-attached
on clone.

### 1. Vision backbone input range (`correct_image_range`)

- **How ComfyUI core does it**:
  - `SAM3_Detect.execute` (`comfy_extras/nodes_sam3.py:118`) resizes `[B, H, W, C]`
    images in `[0, 1]` with `comfy.utils.common_upscale(..., "bilinear", "disabled")`
    and passes them directly to `model(img_resized, ...)`.
  - `_refine_mask` (`comfy_extras/nodes_sam3.py:65`) resizes cropped `[0, 1]`
    regions with `"bilinear"` and passes them to `model.forward_segment(...)`.
  - `_prep_frame` (`comfy/ldm/sam3/tracker.py:207-209`) resizes `[0, 1]` video
    frames to `1008x1008` with `"bicubic"` (which overshoots below `0.0` and
    above `1.0`) and passes them directly into `ViTDet.patch_embed`
    (`comfy/ldm/sam3/sam.py:214-220`), whose `Conv2d(3, 1024, kernel_size=14, stride=14, bias=False)`
    has **no bias** (`comfy/ldm/sam3/sam.py:217`).
- **How Meta's reference does it**:
  - `Sam3ImageProcessor` (`coderef/sam3/sam3/model/sam3_image_processor.py:55-63`)
    and the image-list video loader (`coderef/sam3/sam3/model/io_utils.py:309-316`)
    convert uint8 to `[0, 1]` and normalize with `v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])`
    to `[-1, 1]`. *(Quirk in Meta's own code: the `cv2` MP4 video loader at
    `coderef/sam3/sam3/model/io_utils.py:348-365` converts uint8 to float
    `0..255` without dividing by 255 before `(x - 0.5)/0.5`, feeding `[-1, 509]`
    into the trunk; see `sam31_corrections.py:35-43` and
    [`docs/research/masking/2026-10-07_sam31_in_comfyui.md`](../research/masking/2026-10-07_sam31_in_comfyui.md), finding 7.)*
- **How our patch works (`sam31_corrections.py::CorrectedRange`, `sam31_corrections.py:93-131`)**:
  - Wraps `diffusion_model.detector.backbone.vision_backbone.trunk.patch_embed.forward`.
  - Checks `x.amin() < ALREADY_MAPPED_BELOW` (`-0.5`, `sam31_corrections.py:90, 121-125`):
    if a future ComfyUI update already maps inputs to `[-1, 1]`, it leaves
    `x` untouched; otherwise it runs `x.clamp(0.0, 1.0).mul(2.0).sub(1.0)`
    (clamping bicubic overshoot first, then mapping `[0, 1]` to `[-1, 1]`).
- **Why we did it, and what it changed / improved / don't know**:
  - **Why**: Feeding `[0, 1]` into a bias-free `patch_embed` halves the AC
    signal and injects a `+0.5` DC offset, producing a `0.65` relative L2
    error on `ViTDet` trunk features (`cosine = 0.793` vs `0.9999`,
    [`2026-10-07_sam3_core_against_meta.md`](../../bench/results/2026-10-07_sam3_core_against_meta.md)).
  - **What it improved**: Reduced trunk relative L2 error from `0.65` to
    `0.015` (the residual `0.015` is core's resize kernel vs. `torchvision`'s).
    On the 32-object seeded video tracking test in
    [`2026-10-07_sam3_core_against_meta.md`](../../bench/results/2026-10-07_sam3_core_against_meta.md),
    mapping frames to `[-1, 1]` held all 32 seeded objects where stock core
    lost object 10 (`IoU 0.000`). On `SAM3_Detect` on the live server queue
    ([`2026-10-07_sam31_corrections_on_a_queue.md`](../../bench/results/2026-10-07_sam31_corrections_on_a_queue.md)),
    it moved mask pixel counts by ~3–5% (`IoU 0.946–0.962` against stock).
  - **What it did not improve / nuance to watch**: On the 447-frame kitchen
    stretch of `fun` (`shots 1–6`, frames 604–1050,
    [`2026-10-10_sam31_corrections_on_one_stretch.md`](../../bench/results/2026-10-10_sam31_corrections_on_one_stretch.md)),
    mapping to `[-1, 1]` shifted `MiniMaxH3SubjectTrack`'s pooled trunk
    cosine similarities upward for **both** the right person and wrong
    candidates in later shots (e.g. shot 6's wrong candidate rose from
    `0.720` to `0.761`, narrowing the margin under `MATCH_FLOOR = 0.78` from
    `0.060` to `0.019`, and shot 4's `regain` accepted a wrong candidate at
    frame 832 for 6 frames before `take_back` dropped it).

### 2. Text encoder MLP activation (`correct_text_activation`)

- **How ComfyUI core does it**:
  - `SAM3_CLIP_CONFIG` (`comfy/text_encoders/sam3_clip.py:5-19`, line 6) sets
    `"hidden_act": "quick_gelu"`. ComfyUI's `CLIPMLP` layers in all 24
    transformer blocks of `SAM3TextModel` therefore apply
    `x * torch.sigmoid(1.702 * x)`.
- **How Meta's reference does it**:
  - `VETextEncoder` (`coderef/sam3/sam3/model/text_encoder_ve.py:130, 198`)
    constructs `ResidualAttentionBlock` with default `act_layer=nn.GELU`
    (`F.gelu`, exact Gaussian error function).
- **How our patch works (`sam31_corrections.py::text_mlps`, `sam31_corrections.py:747-778`)**:
  - Iterates over the 24 `layer.mlp` modules of
    `clip.patcher.model.sam3_clip.transformer.text_model.encoder.layers` and
    attaches `clip_patcher.add_object_patch(f"...layers.{i}.mlp.activation", F.gelu)`.
  - Reads `clip_patcher.object_patches_backup` before checking `mlp.activation`
    (`sam31_corrections.py:772-774`) so that when the text encoder is already
    loaded on the GPU from an earlier corrected run, a second run still
    registers the patch on its fresh `ModelPatcher` clone instead of being
    fooled by the in-place attribute on the shared `nn.Module`.
- **Why we did it, and what it changed / improved / don't know**:
  - **Why**: Across 24 layers, `quick_gelu` accumulates a `0.056` relative L2
    drift (`max_abs = 0.453`, cosine `0.9985`) on valid token features after
    `language_backbone.resizer`
    ([`2026-10-07_sam3_core_against_meta.md`](../../bench/results/2026-10-07_sam3_core_against_meta.md)).
  - **What it improved**: Restores exact numerical parity with Meta's
    `VETextEncoder` (`2.0e-7` relative L2, `max_abs = 1.9e-6`).
  - **What it did not change**: On the one-word prompts our pipeline actually
    sends (`person`, `head`), changing `quick_gelu` to `gelu` changed **zero**
    detections across 447 frames (`IoU = 1.0000` on every frame,
    [`2026-10-10_sam31_corrections_on_one_stretch.md`](../../bench/results/2026-10-10_sam31_corrections_on_one_stretch.md)).
    On a 7-word descriptive prompt (`man on the right in dark jacket`,
    [`2026-10-07_sam3_core_against_meta.md`](../../bench/results/2026-10-07_sam3_core_against_meta.md)),
    switching to `gelu` halved the detector's global `presence` probability
    (`0.061` $\to$ `0.032`), which helps under Meta's `scores.sigmoid() * presence.sigmoid()`
    gating, but ComfyUI core's `SAM3_Detect` (`comfy_extras/nodes_sam3.py:217`)
    ignores `presence` and thresholds `scores.sigmoid()` alone, where
    borderline ~0.50 query scores nudge by ~0.02 in either direction.

### 3. Per-object tracker memory selection (`memory_selection`)

- **How ComfyUI core does it**:
  - In `SAM31Tracker._prepare_memory_conditioned_features`
    (`comfy/ldm/sam3/tracker.py:1217-1271`) and `collect_memory_tokens`
    (`comfy/ldm/sam3/tracker.py:221-256`), every propagation step
    unconditionally feeds the conditioning frame (`cond_frame_outputs`) plus
    the immediately preceding `num_maskmem - 1` (`6`) frames from
    `non_cond_frame_outputs` and up to `max_obj_ptrs_in_encoder - 1` (`15`)
    recent `obj_ptr` tokens into decoupled memory cross-attention.
  - In `track_video_with_detection` (`comfy/ldm/sam3/tracker.py:1755-1758`),
    core deletes `non_cond_frame_outputs[old_idx]` as soon as
    `old_idx < frame_idx - lookback` (`lookback = max(7, 16) = 16`).
  - Consequently, if the tracked person is occluded or out of frame for 6+
    frames, all 6 non-conditioning spatial memory slots fill with empty or
    occluded frames, and after 16 frames the last good non-conditioning
    views are evicted from `non_cond_frame_outputs` altogether.
- **How Meta's reference does it**:
  - `cal_mem_score` (`coderef/sam3/sam3/model/video_tracking_multiplex.py:2109-2142`,
    `coderef/sam3/sam3/model/video_tracking_multiplex_demo.py:2038-2071`)
    scores each frame's memory quality from `object_score_logits > 0`,
    predicted mask `iou_pred` scaled from `[0.5, 1.0]` to `[0, 1]`, and a
    dilution penalty from other objects whose masks overlap $> 0.1$ at
    $126\times 126$.
  - `frame_filter` (`coderef/sam3/sam3/model/sam3_multiplex_base.py:482-502`)
    walks backward from `frame_idx - 1` to `1` and keeps only past frames
    whose `eff_iou_score > mf_threshold` (`0.01`, up to `max_mem_frames = 6`),
    and drops the oldest kept frame's object pointer (`pos_and_ptrs[:-1]`)
    when `len(pos_and_ptrs) > max_mem_frames`.
  - *(Quirks in Meta's reference: Meta's SAM 3.1 builder
    `coderef/sam3/sam3/model_builder.py:1141` sets `mf_threshold=0.0` while
    SAM 3.0's builder sets `0.01`; and Meta's multiplex `cal_mem_score`
    reduces `eff_iou_score` with `.mean()` across all active slots in a
    bucket, so one visible companion in a multiplex bucket can keep an
    occluded object's frame above `0.01`.)*
- **How our patch works (`sam31_corrections.py:134-745`, `IoUTap` + `MemorySelectionStep`)**:
  - Attaches to `SAM31Tracker` only (`sam31_corrections.py:786-805`; a
    checkpoint with `SAM3Tracker` is left unpatched).
  - `IoUTap` (`sam31_corrections.py:162-219`) wraps `tracker.sam_mask_decoder`
    to capture raw `iou_pred` (`[B, M, 3]`) before `_forward_propagation`
    (`comfy/ldm/sam3/tracker.py:1375-1398`) discards it after `argmax`.
  - `cal_mem_score_per_object` (`sam31_corrections.py:222-304`) computes the
    effective memory score **per object** (`[N_obj]` float32, without Meta's
    bucket-wide `.mean()`), using `object_score_logits > 0`, the winning
    multimask branch's `iou_pred` mapped from `[0.5, 1.0]` to `[0.0, 1.0]`,
    and the overlap dilution penalty at $126\times 126$.
  - `MemorySelectionStep` (`sam31_corrections.py:363-745`) wraps `track_step`,
    `_prepare_memory_conditioned_features`, and `track_video_with_detection`:
    - Stores frames whose `eff_iou_scores > MF_THRESHOLD` (`0.01`) in a
      per-pass `retained_bank` (`sam31_corrections.py:397, 502-520`) so they
      survive core's 16-frame eviction loop (`comfy/ldm/sam3/tracker.py:1755-1758`),
      and frees `retained_bank` in a `try / finally` at the end of
      `track_video_with_detection` (`sam31_corrections.py:729-744`) so no GPU
      tensors leak onto the cached `ModelPatcher`.
    - Before `_prepare_memory_conditioned_features` runs, builds a remapped
      `output_dict` (`sam31_corrections.py:553-623`) where the selected past
      frames (up to 6 spatial memories and 15 object pointers) are placed at
      synthetic keys `frame_idx - 1`, `frame_idx - 2`, ... so core's
      `collect_memory_tokens` (`comfy/ldm/sam3/tracker.py:250-256`) assigns
      contiguous temporal positional encodings `t_pos = 1..6`, matching
      Meta's `enumerate(valid_Shortcut)` indexing.
    - When `len(ptr_frames) > num_maskmem - 1` (`6`), drops the oldest kept
      non-conditioning `obj_ptr` (`ptr_frames[:-1]`, `sam31_corrections.py:592-607`),
      matching Meta's `pos_and_ptrs[:-1]`.
    - **Single-object safety guard (`one_object_only=True`, `sam31_corrections.py:424, 548-549, 725-727`)**:
      In `MiniMaxH3SAM31Corrections`, the filter activates only when
      `N_obj == 1` (which is how `MiniMaxH3SubjectTrack` always calls the
      tracker: `max_objects=1`, one subject per pass). In a multi-object call
      (`N_obj > 1`), a single remapped `output_dict` driven by `focus_obj=0`
      would strip visible companions' spatial memory whenever object 0 was
      occluded (`sam31_corrections.py:541-547`). The standalone bench tool
      `bench/sam_memory_selection.py:67-169` implements per-bucket/per-object
      remapping for multi-object experiments.
- **Why we did it, and what it changed / improved / don't know**:
  - **Why**: When a person bends down, turns away, or passes behind another
    dancer for more than 6 frames, stock `SAM31Tracker` fills all 6 spatial
    memory slots with empty masks and loses or delays re-acquiring the person.
  - **What it improved** (`docs/state/decisions.md`, 2026-10-17, measured in
    `data/2026-10-17_sam_memory_selection/` across three `fun` scenes and a
    group dance clip):
    - **0 frames** jumped to another person on any tested scene (same as stock).
    - On the group dance clip (`lotsofpeople`), the tracked dancer held
      through occlusions at frames `95–116` and `130–140` that stock
      `SAM31Tracker` lost.
    - On the lead's kitchen scene (`lead_fp_0613g`), after the lead bent
      down and her face was hidden (`687–708`), the tracker recovered her at
      frame `710` (1 frame after the owner's visual mark at `709`) vs. frame
      `715` on stock `SAM31Tracker`—5 frames earlier, with no measurable
      wall-time cost.
    - Turned **on by default** in the masked job chain on 2026-10-17
      (`workflows/h3_config.py::MASKED_JOB_SAM_CORRECTIONS` at
      `workflows/h3_config.py:643-644`; `workflows/masked_job_graph.py::memory_selection_of`
      at `workflows/masked_job_graph.py:205-215` refuses turning it off
      without a recorded `why` and `said_by`).
  - **Why SAM 3.0 (`SAM3Tracker`) is not patched**: On SAM 3.0 base, enabling
    memory selection brought the mask back at frame `702`, which on inspection
    was the back of the lead's hair rather than her face (`docs/state/decisions.md`,
    2026-10-17).

---

## What `MiniMaxH3SAM31Corrections` leaves as ComfyUI computes it (`NOT_CORRECTED`)

`sam31_corrections.py::NOT_CORRECTED` (`sam31_corrections.py:85-86`) lists
seven other places where ComfyUI core differs from Meta's SAM 3.1 reference
that the corrections node deliberately does **not** patch on the model:

| # | stage | ComfyUI core (`comfy/` & `comfy_extras/`) | Meta reference (`coderef/sam3`) | why we leave it unpatched on the model (or handle it in our nodes) |
|---|---|---|---|---|
| 1 | **Resize kernel** | `SAM3_Detect` (`comfy_extras/nodes_sam3.py:118`) uses un-antialiased bilinear (`common_upscale(..., "bilinear")`); `_prep_frame` (`comfy/ldm/sam3/tracker.py:209`) uses bicubic. | `v2.Resize((1008, 1008), bilinear)`: `antialias=False` in `coderef/sam3/sam3/model/sam3_image_processor.py:55`, `antialias=True` in video `coderef/sam3/sam3/model/io_utils.py:309`. | Lives inside node/tracker frame-prep helpers rather than `nn.Module` forwards; accounts for only `0.015` rel L2 on trunk features once `[-1, 1]` range is fixed ([`2026-10-07_sam3_core_against_meta.md`](../../bench/results/2026-10-07_sam3_core_against_meta.md)). |
| 2 | **Detection score formula & NMS** | `SAM3_Detect` (`comfy_extras/nodes_sam3.py:217-226`) thresholds `scores.sigmoid() > threshold` with no `presence` factor and no NMS; `track_video_with_detection` (`comfy/ldm/sam3/tracker.py:1697-1701`) ignores `presence` (`comfy/ldm/sam3/detector.py:495-509` drops `presence`) and runs `_nms_masks` (`max(IoU, IoM) >= 0.5`). | Multiplies `scores.sigmoid() * presence.sigmoid()` (twice in `coderef/sam3/sam3/model/sam3_image_processor.py:124`, once in video `coderef/sam3/sam3/model/sam3_video_base.py`) and runs greedy IoM NMS at `0.1` (`coderef/sam3/sam3/model/sam3_multiplex_detector_utils.py:58-127`). | On single-word nouns (`person`, `head`), `presence` is near `1.0` whenever a person is visible. Nested/duplicate queries from `detector` are deduplicated directly in `subject_tracks.py::one_each` (`subject_tracks.py:107-152`). |
| 3 | **Detect node's 2-pass `_refine_mask`** | `SAM3_Detect` (`comfy_extras/nodes_sam3.py:42-85, 232-235`) crops each detected box (+10% pad), runs `model.forward_segment` twice at `1008x1008`, and unions with the coarse mask (`torch.maximum(binary, refined)` at line 85). | Not in Meta's `Sam3ImageProcessor`; Meta uses `forward_Lock` / `forward_segment` only for interactive point/box prompts. | `MiniMaxH3SubjectTrack` bypasses `SAM3_Detect.execute` and calls `segmenter.model.detector(...)` directly (`subject_track.py:1945`), avoiding both the extra 2 backbone passes per person and the union-only mask growth. |
| 4 | **Hole-fill / sprinkle area (`max_area=16`)** | `track_video_with_detection` (`comfy/ldm/sam3/tracker.py:116-136, 1747-1748`) runs `fill_holes_in_mask_scores(current_out["pred_masks"], max_area=16)` on low-res `288x288` masks before memory encoding, leaving `pred_masks_high_res` (`1008x1008`, used for output) untouched. | SAM 3.1's builder (`coderef/sam3/sam3/model_builder.py:1138`) sets `fill_hole_area=0` (disabled); SAM 3.0 used `16`. | Only touches low-res masks fed into `_deferred_memory_encode`, not the output masks; our tracker post-processes output masks with `subject_tracks.py::drop_specks` (`subject_tracks.py:186-217`). |
| 5 | **Object pointer token in `SAM31Tracker`** | `_forward_propagation` (`comfy/ldm/sam3/tracker.py:1391`) always projects `sam_token = tokens_obj[:, 0]` through `obj_ptr_proj` even though line 1380 picked `best_idx = torch.argmax(iou_obj, dim=-1)` for the mask. | `coderef/sam3/sam3/model/multiplex_utils.py:183-199` selects `sam_tokens_out` at `best_iou_inds` so the object pointer matches the winning mask branch. | Inside `_forward_propagation` between demux and mux; not yet isolated as a standalone A/B on our shots. |
| 6 | **Multi-object occlusion suppression** | `_suppress_recently_occluded` (`comfy/ldm/sam3/tracker.py:1486-1507, 1749-1751`) suppresses at `max(IoU, IoM) >= 0.3` on `pred_masks` (memory) only, and `_deferred_memory_encode` (`comfy/ldm/sam3/tracker.py:1514-1524`) clamps non-winning pixels when a mask shrinks below `0.3`. | `suppress_overlapping_based_on_recent_occlusion` (`coderef/sam3/sam3/model/video_tracking_multiplex.py:2178-2231`) suppresses at `IoU > 0.7` on both low-res and high-res output masks. | Inoperative when `N_obj == 1` (`comfy/ldm/sam3/tracker.py:1490`, `1514`), which is how `MiniMaxH3SubjectTrack` runs every tracking pass (`max_objects=1`). |
| 7 | **Threshold & resize order on output** | `track_video_with_detection` (`comfy/ldm/sam3/tracker.py:194-198, 1769-1776`) thresholds `masks_out > 0` at `1008x1008` and bit-packs into uint8; `SAM3_TrackPreview` (`comfy_extras/nodes_sam3.py:437-439`) resizes unpacked bools with `nearest`, while `SAM3_TrackToMask` (`comfy_extras/nodes_sam3.py:516-517`) resizes unpacked `0/1` floats with `bilinear` to `(H, W)`. | Interpolates float logits to `(H, W)` with `bilinear` and thresholds `> 0.0` once (`coderef/sam3/sam3/model/video_tracking_multiplex.py:1478-1485`). | `MiniMaxH3SubjectTrack` re-thresholds the mask returned by `SAM3_TrackToMask` at `> 0.5` (`subject_track.py:1968`), and `MiniMaxH3MaskedSource` grows and feathers the mask on the render canvas (`video_mask.py:747-762, 1990-2025`). |

---

## Layer 2: Shot-aware orchestration and shared helpers (`subject_track.py`, `subject_tracks.py`, `shot_table.py`, `subject_boxes.py`)

### Why we do not run ComfyUI core's text-prompted `SAM3_VideoTrack` across a clip

ComfyUI core's `SAM3_VideoTrack` (`comfy_extras/nodes_sam3.py:251-314`) drives
`SAM31Tracker.track_video_with_detection` (`comfy/ldm/sam3/tracker.py:1647-1802`).
When given text `conditioning` on a multi-shot video, four behaviours in
`track_video_with_detection` break single-subject replacement:

1. **No concept of scene cuts**: SAM 3's video datasets contain single-camera
   takes without cuts. Across a hard cut, spatial memory and object pointers
   from the old shot do not match the new layout, or match the wrong person
   ([`2026-10-04_masked_v2v_band.md`](../../bench/results/2026-10-04_masked_v2v_band.md)).
2. **A hard lifetime ceiling of 64 object slots (`INTERNAL_MAX_OBJECTS = 64`, `comfy/ldm/sam3/tracker.py:1645, 1651-1652, 1691-1694`)**:
   On every detection frame (`frame_idx % detect_interval == 0`),
   `_match_and_add_detections` (`comfy/ldm/sam3/tracker.py:1578-1643`) adds any
   detection whose overlap (`max(IoU, IoM)`) with existing tracks is `< 0.5` as
   a brand-new object in `MultiplexState` (`_add_detected_objects`,
   `comfy/ldm/sam3/tracker.py:1541-1556`). Slots are **never freed or reused**.
   Once `mux_state.total_valid_entries >= max_objects` (`8` by default in
   `comfy_extras/nodes_sam3.py:271`, clamped to `64` at
   `comfy/ldm/sam3/tracker.py:1651-1652`), `run_det` (`comfy/ldm/sam3/tracker.py:1691-1694`)
   becomes `False` for the remainder of the video—so no new person is ever
   detected and no lost track is ever refreshed by detection again
   ([`2026-10-08_one_person_track_against_core_multi_object.md`](../../bench/results/2026-10-08_one_person_track_against_core_multi_object.md)).
3. **Keep-alive blanking (`comfy/ldm/sam3/tracker.py:1585-1606, 1770-1773`)**:
   Each track starts at `keep_alive = 1` (or `8` for `initial_masks`,
   `comfy/ldm/sam3/tracker.py:1712, 1727`), increments by `+1` (up to `8`) on detection frames
   where a detection overlaps it by `>= 0.5`, and decrements by `-1` (down to
   `-4`) when unmatched. Whenever `keep_alive[i] <= 0`, lines `1770-1773`
   overwrite `masks_out[i] = NO_OBJ_SCORE` (`-1024.0`), blanking the output
   mask even if propagation still had the person—except that once `max_objects`
   is hit and `run_det` turns off (`comfy/ldm/sam3/tracker.py:1762`), `keep_alive` stops
   updating and freezes at whatever sign it held on the frame the cap was
   exhausted!
4. **High-confidence detection overwrite (`HIGH_CONF = 0.8`, `comfy/ldm/sam3/tracker.py:1607-1626`)**:
   Whenever a detection scores `>= 0.8` and overlaps a tracked mask by
   `max(IoU, IoM) >= 0.5`, `_match_and_add_detections` overwrites
   `current_out["pred_masks"][best_trk]` with the detector's mask and
   re-encodes memory (`_deferred_memory_encode`). Because `_compute_mask_overlap`
   (`comfy/ldm/sam3/tracker.py:72-81`) uses `max(IoU, IoM)` (intersection over
   the *smaller* mask's area), a small partial detection (or a neighbour whose
   mask overlaps half of the subject) with score `>= 0.8` can overwrite the
   tracked mask.

### How `MiniMaxH3SubjectTrack` (`subject_track.py`) and `subject_tracks.py` work instead

`MiniMaxH3SubjectTrack` (`subject_track.py:1387-1869`, wired by default in
`workflows/h3_config.py::SUBJECT_TRACK` at `workflows/h3_config.py:500-501`
and in job previews at `workflows/masked_job_graph.py:300-307`) replaces
core's text-conditioned video loop with a shot-by-shot pipeline that uses
ComfyUI core's `SAM3Detector` and `SAM31Tracker` as low-level primitives:

| stage | our function & lines | what it does and how it differs from ComfyUI core | why we did it, and what we observed |
|---|---|---|---|
| **1. Direct detector & trunk callables** | `subject_track.py::_sam_callables` (`subject_track.py:1906-1970`) | Loads `segmenter` onto the GPU via `comfy.model_management.load_models_gpu([segmenter])` and calls `model.detector(img_resized, text_embeddings=..., text_mask=..., orig_size=(H, W))` (`comfy/ldm/sam3/detector.py:472-493`) directly, returning all raw query masks and `scores.sigmoid()` above `threshold` without running `SAM3_Detect`'s 2-pass `_refine_mask` loop (`comfy_extras/nodes_sam3.py:232-235`). Also exposes `trunk_of(image)` calling `vision_backbone.trunk(x)` (`comfy/ldm/sam3/sam.py:223-295`) to get `[N, 72, 72, 1024]` features at `1008x1008`. | Avoids 2 extra `1008x1008` backbone passes per detected person on every probe frame, and exposes the `72x72x1024` ViTDet trunk features needed to match the subject's identity across scene cuts. |
| **2. Tokenizer `:1` fast-path bypass** | `subject_track.py::counted` (`subject_track.py:395-415`) | Strips any existing `:N` suffix and returns bare `name` when `limit <= 1`, or `f"{name}:{int(limit)}"` when `limit >= 2`. | Fixes `comfy/text_encoders/sam3_clip.py:51-54` (`SAM3TokenizerWrapper.tokenize_with_weights`), which passes the raw string `"person:1"` to the BPE tokenizer when `len(parsed) == 1` and `weight == 1`, encoding literal `:1` into the text tokens (`subject_track.py:397-404`). |
| **3. Deduplicating un-NMS'd detector queries** | `subject_tracks.py::one_each` (`subject_tracks.py:107-152`), `subject_track.py::one_detection_each` (`subject_track.py:1294-1336`) | Sorts detections on a frame by area (largest first) and drops any mask that lies `>= 0.8` (`SAME_THING = 0.8`, `subject_tracks.py:101`) inside a kept larger mask—**unless** the highest-scoring detection in that group has area `< 0.5` (`A_PART_UNDER = 0.5`, `subject_tracks.py:104`) of the largest mask, in which case the largest mask is a background/merged superset and is dropped first (`subject_tracks.py:124-138`). | Core's `SAM3_Detect` (`comfy_extras/nodes_sam3.py:217-226`) runs no NMS across its 200 queries, so one person is regularly returned 2–3 times (full body + torso + head). Measured on `fun` and `lotsofpeople` ([`2026-10-08_one_person_track_against_core_multi_object.md`](../../bench/results/2026-10-08_one_person_track_against_core_multi_object.md)): collapses nested duplicate queries to 1 mask per person while keeping a smaller foreground person standing in front of a larger background figure. |
| **4. Letterbox-aware shot cut detection** | `subject_track.py::flat_borders`, `cut_scores`, `find_cuts`, `auto_cuts` (`subject_track.py:424-518`) | Strips constant top/bottom letterbox and left/right pillarbox bars (`flat_borders`, `subject_track.py:424-454`), downsamples to `192x108` (`SCORE_SIZE`), computes $1 - r$ of adjacent-frame spatial gradients (`cut_scores`), and thresholds at `median + 6 * MAD` clamped to `[0.15, 0.35]` (`auto_cuts`, with `MIN_SHOT = 6` frames). | SAM 3's tracker cannot cross a scene cut without either losing the person or jumping to whoever now stands in the old spot. Stripping flat borders (`flat_borders`) fixed missed cuts on letterboxed sources where constant black bars diluted frame-to-frame gradient differences ([`2026-10-06_bordered_source_cuts.md`](../../bench/results/2026-10-06_bordered_source_cuts.md)). |
| **5. Multi-tracker exclusion (`others`)** | `subject_track.py::held_people` (`subject_track.py:1349-1384`), `subject_tracks.py::in_the_way` (`subject_tracks.py:562-574`) | When a second or third tracker runs after an earlier one (`workflows/masked_job_graph.py:288-303`), `others` carries the union of earlier trackers' masks. `held_people` removes any candidate detection on a pick, match, or regain frame that overlaps `others` by `>= 0.8` (`SAME_THING`), and `in_the_way` prevents `regain` from firing when `others` covers `>= 0.3` (`COVERED_BY_OTHER`) of the lost subject's last known mask. | Prevents two trackers in a multi-subject preview from picking or switching onto the same person during a cross-over ([`2026-10-08_one_person_track_against_core_multi_object.md`](../../bench/results/2026-10-08_one_person_track_against_core_multi_object.md)). |
| **6. Picking the main subject & two-view trunk signature** | `subject_track.py::choose`, `top_third`, `head_of`, `signature`, `relative`, `main_subject` (`subject_track.py:536-685, 1094-1140`) | On the pick frame (`pick_on="automatic"` scans the first shot's sample frames for the highest detector score, or uses `a frame I name`), filters `person` detections to those containing a detected `head` (`head_of`, `overlap >= 0.5`), chooses by `largest`, `most central` (default in `workflows/h3_config.py:495-497`), `leftmost`, `rightmost`, or `person N`, and pools L2-normalized $72\times 72$ ViTDet trunk features over **two views**: `top_third` of the person mask and the `head` mask. When other people are on the pick frame, `relative` (`subject_track.py:617-633`) subtracts the mean signature of those other people and re-normalizes. | Requiring a `head` detection stopped non-person false positives (such as a hanging microphone at the top of the frame, [`2026-10-06_subject_track_defaults.md`](../../bench/results/2026-10-06_subject_track_defaults.md)) from winning `pick`. Two-view (`top_third` + `head`) contrastive trunk matching (`relative`) raised the separation between the target subject and bandmates across cuts (`MATCH_FLOOR = 0.78`, `PLAIN_FLOOR = 0.60`, `SOLO_MARGIN = 0.05` in `auto_match`, `subject_track.py:706-736`). **Known limit**: ViTDet trunk features are appearance/pose features, not a face-identity embedding; in a crowd or after a wardrobe/lighting change, cosine scores cluster tightly (`0.72–0.79`) and shots need `seeds` or `corrections` ([`2026-10-07_subject_track_under_nudge.md`](../../bench/results/2026-10-07_subject_track_under_nudge.md), [`who_is_who.md`](who_is_who.md)). |
| **7. Bidirectional single-object tracking per shot & `ScoreTap`** | `subject_track.py::follow` (`subject_track.py:1183-1291`), `ScoreTap` (`subject_track.py:1872-1903`), `_sam_callables.track` (`subject_track.py:1953-1968`) | Within one shot `[start, end)`, calls `SAM3_VideoTrack.execute(segmenter, frames, conditioning=None, initial_mask=seed[None], max_objects=1)` (`comfy_extras/nodes_sam3.py:282-314`). If the seed frame `at` is inside the shot (`at > start`), runs one pass forward on `shot[at:]` and a second pass forward on the time-reversed slice `shot[:at + 1].flip(0)` (`subject_track.py:1236-1239`). `ScoreTap` wraps `SAM31Tracker._forward_propagation` (`comfy/ldm/sam3/tracker.py:1351-1398`) to record per-frame `score_obj` (`object_score_logits`) into `shot_table`. | Passing `conditioning=None, max_objects=1` bypasses core's per-frame detector, slot cap (`INTERNAL_MAX_OBJECTS = 64`), `keep_alive` blanking, and `HIGH_CONF = 0.8` mask overwrites altogether: the tracker follows the one seeded mask bidirectionally through the shot using propagation + memory attention alone ([`2026-10-08_one_person_track_against_core_multi_object.md`](../../bench/results/2026-10-08_one_person_track_against_core_multi_object.md)). |
| **8. Gallery expansion & mid-shot `regain`** | `subject_track.py::gallery_frames` (`subject_track.py:1143-1180`), `regain` (`subject_track.py:1572-1677`), `subject_tracks.py::plausible`, `doubted`, `trusted`, `take_back`, `take_back_by_place` (`subject_tracks.py:259-363, 465-550`) | After each accepted shot, `gallery_frames` adds up to `GALLERY_PER_SHOT = 4` trusted frames (`trusted`, where the mask is between `0.25x` and `2.5x` of the shot median and hasn't jumped by `> 1.0` mask-widths) to the reference gallery. Inside a shot, if the tracked mask vanishes for `LOST_RUN = 6` frames (`subject_track.py:338`) or jumps (`doubted`), `regain` searches forward at `REGAIN_STRIDE = 6` for a candidate matching the gallery (`>= 0.78`, or the sole person when `alone`), tracks forward from that new seed, trims pre-loss partial mask collapse (`take_back`, `< 0.5` of running median area) and box drift (`take_back_by_place`, shift `> 0.5` of mask span), and splices the regained tail in if it adds net tracked frames. | Recovers subjects who leave the frame or are fully occluded mid-shot and return later, and strips the 1–5 frames where SAM's mask collapses onto a sleeve or a bystander's shoulder just before losing the subject (`take_back`, `take_back_by_place`, measured on `fun` shots in [`2026-10-08_one_person_track_against_core_multi_object.md`](../../bench/results/2026-10-08_one_person_track_against_core_multi_object.md)). |
| **9. Gap bridging & speck removal** | `subject_tracks.py::unbroken` (`subject_tracks.py:387-419`), `drop_specks` (`subject_tracks.py:186-217`) | `drop_specks` removes connected components smaller than `SPECK_SHARE = 0.05` of the mask's area on a frame unless those pixels also appear on an adjacent frame (`t - 1` or `t + 1`). `unbroken` bridges empty runs of up to `FILL_GAP = 2` frames by copying the nearest tracked frame when the bounding boxes before and after the gap are plausible. | Core's `fill_holes_in_mask_scores(..., max_area=16)` (`comfy/ldm/sam3/tracker.py:1747-1748`) only cleans low-res `288x288` memory masks, leaving transient 1-frame background specks in `pred_masks_high_res` (`1008x1008`). Because `MiniMaxH3MaskedSource` pools over 4-frame latent steps (`FRAME_PER_TOKEN = 4`), a single-frame speck punches a 4-frame hole in the plate; `drop_specks` removes transient 1-frame islands while keeping small real features (like an ear or hand) that persist across adjacent frames. |
| **10. Seed boxes, typed corrections, shot table & pose boxes** | `subject_track.py::parse_seeds` (`subject_track.py:791-943`), `parse_corrections` (`subject_track.py:946-1063`), `shot_table.py::build` & `MiniMaxH3SaveShotTable` (`shot_table.py:86-250`), `subject_boxes.py::MiniMaxH3SubjectBoxes` (`subject_boxes.py:79-122`) | `seeds` accepts `shot N: frame F x0,y0,x1,y1` (written from the owner's clicks on the who-is-who page by `bench/page_answers_to_cast.py` and `bench/cast_file.py::seeds`), picks the detector mask with highest IoU (`>= 0.1`) against the box (or uses the box itself when no detection overlaps), and adds that frame to the gallery as ground truth. `corrections` accepts `shot N: person K [@F]`, `shot N: none`, `cut F`, `no cut F`, `hold F..G`, `drop F..G` (`subject_tracks.py:577-709`). `MiniMaxH3SaveShotTable` writes `*_shots.json` and numbered candidate tiles (`*_shots_00001_.png`); `MiniMaxH3SubjectBoxes` turns `[N, H, W]` masks into `[N, 5]` boxes for `MiniMaxH3BodyPose`. | Replaces ComfyUI core's trial-and-error `object_indices` string in `SAM3_TrackToMask` (`comfy_extras/nodes_sam3.py:484-509`) with a machine-readable `shot_table` JSON and numbered candidate tiles that drive `bench/who_page.py`, `bench/shot_to_load.py`, and `MiniMaxH3MaskedSource`'s `cut_gate` (`video_mask.py:813-860`). |
