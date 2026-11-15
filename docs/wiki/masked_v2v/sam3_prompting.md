# SAM 3.1 in this pack: prompting, runtime corrections, shot-aware tracking, and how our code differs from ComfyUI core

last updated: 2026-10-18 (merged the full map of our SAM 3.1 nodes, runtime patches, shared helpers, ComfyUI core code links, and what each change did or did not improve; corrected the 2026-10-07 line that said no graph wires `MiniMaxH3SAM31Corrections`); 2026-10-07 (first written, from the day's reading of Meta's code and benchmark annotations; later the same day, what each ComfyUI difference does to a plain noun with one modifier)

Written by hand. This page is the operative form: one rule a line, and a
pointer to the code or record that holds the count behind it. Where it and a
record disagree, the record is right; where it and the code disagree, the
code is. H3's own prompt is a different subject and lives in
[`docs/wiki/prompting/prompting.md`](../prompting/prompting.md).

## What SAM 3.1 takes

A short noun phrase, or an example box, or a click. Meta's own description
is "a short text phrase or exemplars"
(`coderef/sam3/README.md`), and the text side did not change from SAM 3 to
SAM 3.1: 3.1 changed the tracker (`coderef/sam3/RELEASE_SAM3p1.md`). The
detector answers "where is every one of these", not "which one do you mean"
and not "what is this".

In this pack every phrase SAM is given is a visible, editable input, by the
owner's rule (2026-10-04). The shipped ones are in
`workflows/h3_config.py::SUBJECT_TRACK` (`subject_phrase`, `head_phrase`).

## The rules

| to find | ask for | why, and where it is counted |
|---|---|---|
| a person | the bare word, `person` | It is the commonest form in Meta's benchmarks and the one the benchmark almost never expects to be refused. [`2026-10-07_sam3_benchmark_phrases.md`](../../../bench/results/2026-10-07_sam3_benchmark_phrases.md), "People" |
| a part of a person | the part's noun alone (`head`, `hand`, `hair`) | The bare part noun far outnumbers "a person's hand" in every dataset. Same record, "People". For parts inside one tracked person this pack uses the part model, not a phrase: `sapiens2_parts.py` |
| a thing | its plain singular noun, at most an article and one colour or modifier | That is the shape of nearly every phrase the benchmarks ask. Same record, "What a phrase looks like" |
| a small thing | the same, and do not depend on it | Things that small are in the benchmark by size, but a given small noun can be nearly untested (the record's "Size" names one). A phrase found the lane's test prop for the first seconds of a window and then not at all, by this pack's tracker and by ComfyUI's own per-frame text tracker alike (2026-10-07, not yet in a record) |
| what a person is holding | no phrase: the part node's `held` output | Inside the subject's mask, not body or clothing to the part model, near the lips or a hand. Nothing small is tracked on its own. `sapiens2_parts.py`, "What the subject holds" |
| one particular person among several | `person`, then choose | See the next section |

## Choosing one among several: not with a longer phrase

**Do not describe which one** ("the man in the red shirt", "the leftmost
woman", "the object in his hand").

- Meta's benchmarks barely contain such phrases, so the model is untested on
  them, which is not the same as failing: the benchmark record, "What a
  phrase looks like" and its fourth line to act on. Meta's own examples are
  one or two plain words throughout:
  [`2026-10-07_mryolk_phrases.md`](../research/masking/2026-10-07_mryolk_phrases.md).
- Under Meta's own scoring (the next section) the descriptive phrases an
  independent session tried returned nothing at all
  ([`2026-10-07_sam3_precision_arms.md`](../../../bench/results/2026-10-07_sam3_precision_arms.md),
  the third departure).
- Meta's way to choose is to find all with the plain word and then select:
  a box or a click on top of the phrase, an object id in video, or a
  vision-language model that looks and picks
  (`coderef/sam3/examples/sam3.1_video_predictor_example.ipynb`,
  `sam3_agent.ipynb`; the phrase guide's "What Meta's own examples do").
- This pack's way today: the Subject Track's pick rule, its match across a
  cut, and typed corrections (`shot N: person K`) or seed boxes (`shot N: frame F x0,y0,x1,y1`): [`docs/wiki/masked_v2v.md`](../masked_v2v.md).
  ComfyUI's detect node accepts boxes and points; nothing here passes them
  yet (read, not run).

## What ComfyUI does differently from Meta with a phrase

Each matters more the longer or rarer the phrase.

- **The count is ComfyUI's syntax, not the model's.** `person:16` asks the
  node for up to sixteen detections (`comfy/text_encoders/sam3_clip.py:48-84`,
  `SAM3TokenizerWrapper.tokenize_with_weights`); for counts above one, the
  `:N` is stripped before the text reaches the model. For `:1` (`person:1`),
  core's fast path (`comfy/text_encoders/sam3_clip.py:51-54`) forgets to strip
  `:1` and tokenizes the literal characters `:` and `1` into the prompt;
  `subject_track.py::counted` (`subject_track.py:395-415`) works around that
  by sending bare `name` when the count is one. The Subject Track's
  `max_people` sets the count. On a frame with more people than the count,
  the subject may not be among those returned, which decides whether a pick
  or a regain can see them at all:
  [`2026-10-07_subject_alone_or_in_a_group.md`](../../../bench/results/2026-10-07_subject_alone_or_in_a_group.md).
- **The presence score is ignored by default.** Meta multiplies every detection's score
  by a score for "is this thing in the picture at all" (`presence` from
  `comfy/ldm/sam3/detector.py:213-214`) and removes overlapping detections;
  ComfyUI's `SAM3_Detect` (`comfy_extras/nodes_sam3.py:217-226`) and
  `track_video_with_detection` (`comfy/ldm/sam3/tracker.py:1697-1701`)
  threshold `scores.sigmoid()` alone unless opted in via
  `MiniMaxH3SAM31Corrections(correct_detector_presence=True)`
  (`sam31_corrections.py:1153-1180`, `default=False`; query overlap removal
  remains `NOT_CORRECTED`).
  On a frame where the thing is barely present ComfyUI returns detections
  Meta's rule drops. **Read, not run (2026-10-07): Meta's released image
  code appears to apply the presence score twice**, once inside the model
  where the class score is written (`coderef/sam3/sam3/model/sam3_image.py`,
  under `supervise_joint_box_scores`, which both builders set) and again in
  the processor before its threshold
  (`coderef/sam3/sam3/model/sam3_image_processor.py`). That is Meta's
  single-image path. Its video pipeline thresholds the model's joint score
  as written, with no second factor
  (`coderef/sam3/sam3/model/sam3_video_base.py`, where detections are read),
  and that is the path the records below compare with, so their "Meta's
  rule", one multiplication, is Meta's video rule. No record here ran the
  image processor. A proposal to follow Meta has to say which path it
  means. The
  precision record's third departure;
  [`2026-10-07_sam3_core_against_meta.md`](../../../bench/results/2026-10-07_sam3_core_against_meta.md)
  has both rules over the same detections.
- **The text encoder runs another activation than Meta built it with.** Core
  sets `"hidden_act": "quick_gelu"` in `comfy/text_encoders/sam3_clip.py:6`
  where Meta's `VETextEncoder` uses `nn.GELU()`
  (`coderef/sam3/sam3/model/text_encoder_ve.py:130`). Nil
  to small for `person` and `head` and for a plain noun with one modifier
  (the shape the rules above ask for; a session preparing the upstream fix,
  2026-10-07, on public images, reported and not in a tracked record),
  larger for longer phrases. The corrections node
  (`sam31_corrections.py::MiniMaxH3SAM31Corrections`, `sam31_corrections.py:1345-1407`)
  sets it right at run time for whatever loader feeds it, corrects the
  image's value range, filters tracker memory (`N_obj == 1` gating plus
  Multiplex `SAM3-DMS` when `N_obj >= 2`), and optionally enables Meta's
  propagation multimask pointer token and detector presence head (`default=False`):
  [`2026-10-07_sam31_corrections_on_a_queue.md`](../../../bench/results/2026-10-07_sam31_corrections_on_a_queue.md),
  [`2026-10-10_sam31_corrections_on_one_stretch.md`](../../../bench/results/2026-10-10_sam31_corrections_on_one_stretch.md),
  [`2026-11-15_sam31_dms_and_fixes.md`](../../../bench/results/2026-11-15_sam31_dms_and_fixes.md).
  Every masked job preview graph wires it with the three default-on corrections
  (`workflows/masked_job_graph.py:280-287`,
  `workflows/h3_config.py::MASKED_JOB_SAM_CORRECTIONS` at
  `workflows/h3_config.py:1441`), while the single-pass shipped templates in
  `workflows/build_workflows.py:2259-2276` wire `CheckpointLoaderSimple`
  directly. *(Until 2026-10-18 this line still said "No shipped graph wires
  it yet", written on 2026-10-07 before `masked_job_graph.py` landed.)* Even
  with all corrections active, query overlap removal (NMS) in `SAM3_Detect`
  is still scored by ComfyUI's rule and not Meta's.
- **On the CPU, and wherever ComfyUI does not use PyTorch's attention, the
  detector does not hide the padding of a phrase.** The detector hands its
  text mask to ComfyUI's attention as a true/false mask
  (`comfy/ldm/sam3/detector.py:54-57`, `SplitMHA.forward`, called from
  `EncoderLayer.forward` at `comfy/ldm/sam3/detector.py:101`,
  `DecoderLayer.forward` at `comfy/ldm/sam3/detector.py:139`, and
  `SegmentationHead.forward` at `comfy/ldm/sam3/detector.py:352`); PyTorch's
  attention reads that as "attend where true", while the sub-quadratic and
  split attention functions add it to the scores, which hides nothing
  (`comfy/ldm/modules/sub_quadratic_attention.py`). ComfyUI picks those two
  when PyTorch attention is not enabled (`comfy/ldm/modules/attention.py`,
  where `optimized_attention` is chosen), which is the default in a `--cpu`
  process. Found and localised by a session reviewing the upstream fix,
  2026-10-07, on public images: with `--use-pytorch-cross-attention` a CPU
  process gives the card's detections. Not reported upstream and not yet in
  a tracked record. It is not only long phrases: the same session reports
  a plain modifier-and-noun phrase gaining detections of other things on
  that path. The server log line "Using pytorch attention" is the
  observable for which path a process is on. **A detection score from a CPU
  process run without that flag is evidence about the faulty path only**;
  text features are not affected.

---

## Where our SAM 3.1 code lives, and how it relates to ComfyUI core

Nothing in this pack edits or monkey-patches ComfyUI core on disk
(`bench/check_vsa_core_patch.py`). Instead, our SAM 3.1 stack sits in two
layers above core's `comfy/text_encoders/sam3_clip.py`,
`comfy/ldm/sam3/{sam,detector,tracker}.py`,
`comfy/supported_models.py:2510-2559` (`SAM3`, `SAM31`), and
`comfy_extras/nodes_sam3.py`:

| file in this pack | what it owns | what it touches in ComfyUI core |
|---|---|---|
| [`bench/convert_sam3_checkpoint.py`](../../../bench/convert_sam3_checkpoint.py) (`bench/convert_sam3_checkpoint.py:1-116`) | Repacks Meta's pickled `sam3.1_multiplex.pt` into `sam3.1_multiplex_fp32.safetensors` (`workflows/h3_config.py::SEGMENTER`, `workflows/h3_config.py:1248`) with zero weight changes, verified by `torch.equal` readback (`bench/convert_sam3_checkpoint.py:98-103`). | Drops only the 32 `complex64` `*.attn.freqs_cis` buffers that safetensors cannot store and that `comfy/supported_models.py:2529-2530` (`SAM3.process_unet_state_dict`) deletes at load anyway. |
| [`sam31_corrections.py`](../../../sam31_corrections.py) (`sam31_corrections.py::MiniMaxH3SAM31Corrections`, `sam31_corrections.py:1345-1407`) | Runtime patches on cloned `MODEL` and `CLIP` `ModelPatcher` objects (`add_object_patch` + `CallbacksMP.ON_CLONE`) for three default-on corrections (vision input range, text MLP activation, single-object + Multiplex `SAM3-DMS` tracker memory selection) plus two opt-in fixes (`correct_multimask_ptr`, `correct_detector_presence`, `default=False`). Guarded by [`bench/check_sam31_corrections.py`](../../../bench/check_sam31_corrections.py). | `ViTDet.patch_embed` (`comfy/ldm/sam3/sam.py:214-220`), `SAM3TextModel` MLPs (`comfy/text_encoders/sam3_clip.py:5-19`), `SAM31Tracker` (`comfy/ldm/sam3/tracker.py:1217-1802`), and `SAM3Detector._detect`. |
| [`bench/sam_memory_selection.py`](../../../bench/sam_memory_selection.py) (`bench/sam_memory_selection.py:1-250`) | Standalone measurement harness for per-object tracker memory selection on both `SAM31Tracker` (including multi-object per-bucket remapping) and `SAM3Tracker`. | `SAM31Tracker` and `SAM3Tracker` (`comfy/ldm/sam3/tracker.py:497-641, 1217-1802`). |
| [`subject_track.py`](../../../subject_track.py) (`subject_track.py::MiniMaxH3SubjectTrack`, `subject_track.py:2756`) | Shot-aware single-person tracker across cuts: cut detection, `:1` tokenizer bypass, `SAM3_Detect.execute` and `ViTDet.trunk` calls, two-view trunk feature matching (`top_third` + `head`), bidirectional seed-only `SAM3_VideoTrack` (`max_objects=1`), `ScoreTap` (`track_step`), mid-shot `regain`, and multi-tracker exclusion (`others`). Guarded by [`bench/check_subject_track.py`](../../../bench/check_subject_track.py). | Calls `SAM3_Detect.execute` (`subject_track.py:2636`), `vision_backbone.trunk` (`comfy/ldm/sam3/sam.py:223-295`), `SAM3_VideoTrack.execute` (`comfy_extras/nodes_sam3.py:282-314`), and taps `SAM31Tracker.track_step` (`subject_track.py:2254-2306`). |
| [`subject_tracks.py`](../../../subject_tracks.py) (`subject_tracks.py:1-709`) | Pure-tensor shared helpers imported by `subject_track.py` without loading ComfyUI nodes: nested-detection deduplication (`one_each`), speck removal (`drop_specks`), track plausibility/trust checks (`plausible`, `doubted`, `trusted`), gap bridging (`unbroken`), pre-loss trim (`take_back`, `take_back_by_place`), occlusion check (`in_the_way`), and note parsing (`parse_notes`, `hand_out`). | Replaces the missing query NMS in `comfy_extras/nodes_sam3.py:217-226` and replaces core's `keep_alive` / `HIGH_CONF` refresh rules in `comfy/ldm/sam3/tracker.py:1578-1643`. |
| [`shot_table.py`](../../../shot_table.py) (`shot_table.py:1-250`, `MiniMaxH3SaveShotTable`) | Formats the per-shot JSON audit (`build`, `shot_table.py:86-154`) and saves it beside the numbered preview PNGs (`shot_table.py:188-250`). | Replaces blind index selection in `SAM3_TrackPreview` / `SAM3_TrackToMask` (`comfy_extras/nodes_sam3.py:317-518`). |
| [`subject_boxes.py`](../../../subject_boxes.py) (`subject_boxes.py:1-122`, `MiniMaxH3SubjectBoxes`) | Converts a `[N, H, W]` subject mask into `[N, 5]` `(batch_idx, x1, y1, x2, y2)` `BBOX_LIST` for `MiniMaxH3BodyPose`, dropping masks smaller than `SMALLEST_MASK_PX = 2048` (`subject_boxes.py:30-45`). | Feeds `MiniMaxH3BodyPose` (`body_pose.py`) in masked job previews (`workflows/masked_job_graph.py:318-323`). |
| [`video_mask.py`](../../../video_mask.py) (`video_mask.py:2028-2041, 2506-2585`, `detect_part`) | When `MiniMaxH3MaskedSource` runs with `replace="head and hair"`, calls `SAM3_Detect.execute` per phrase on frames where the subject is present (`video_mask.py:2028-2041`). | `comfy_extras/nodes_sam3.py::SAM3_Detect.execute` (`comfy_extras/nodes_sam3.py:114-248`). |

---

## Layer 1: Runtime patches on ComfyUI core (`sam31_corrections.py`)

`MiniMaxH3SAM31Corrections` (`sam31_corrections.py:1345-1407`) sits between
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
    [`2026-10-07_mryolk.md`](../research/masking/2026-10-07_mryolk.md), finding 7.)*
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
    [`2026-10-07_sam3_core_against_meta.md`](../../../bench/results/2026-10-07_sam3_core_against_meta.md)).
  - **What it improved**: Reduced trunk relative L2 error from `0.65` to
    `0.015` (the residual `0.015` is core's resize kernel vs. `torchvision`'s).
    On the 32-object seeded video tracking test in
    [`2026-10-07_sam3_core_against_meta.md`](../../../bench/results/2026-10-07_sam3_core_against_meta.md),
    mapping frames to `[-1, 1]` held all 32 seeded objects where stock core
    lost object 10 (`IoU 0.000`). On `SAM3_Detect` on the live server queue
    ([`2026-10-07_sam31_corrections_on_a_queue.md`](../../../bench/results/2026-10-07_sam31_corrections_on_a_queue.md)),
    it moved mask pixel counts by ~3–5% (`IoU 0.946–0.962` against stock).
  - **What it did not improve / nuance to watch**: On the 447-frame kitchen
    stretch of `fun` (`shots 1–6`, frames 604–1050,
    [`2026-10-10_sam31_corrections_on_one_stretch.md`](../../../bench/results/2026-10-10_sam31_corrections_on_one_stretch.md)),
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
    ([`2026-10-07_sam3_core_against_meta.md`](../../../bench/results/2026-10-07_sam3_core_against_meta.md)).
  - **What it improved**: Restores exact numerical parity with Meta's
    `VETextEncoder` (`2.0e-7` relative L2, `max_abs = 1.9e-6`).
  - **What it did not change**: On the one-word prompts our pipeline actually
    sends (`person`, `head`), changing `quick_gelu` to `gelu` changed **zero**
    detections across 447 frames (`IoU = 1.0000` on every frame,
    [`2026-10-10_sam31_corrections_on_one_stretch.md`](../../../bench/results/2026-10-10_sam31_corrections_on_one_stretch.md)).
    On a 7-word descriptive prompt (`man on the right in dark jacket`,
    [`2026-10-07_sam3_core_against_meta.md`](../../../bench/results/2026-10-07_sam3_core_against_meta.md)),
    switching to `gelu` halved the detector's global `presence` probability
    (`0.061` $\to$ `0.032`), which helps under Meta's `scores.sigmoid() * presence.sigmoid()`
    gating, but ComfyUI core's `SAM3_Detect` (`comfy_extras/nodes_sam3.py:217`)
    ignores `presence` and thresholds `scores.sigmoid()` alone, where
    borderline ~0.50 query scores nudge by ~0.02 in either direction.

### 3. Per-object & Multiplex `SAM3-DMS` tracker memory selection (`memory_selection`)

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
  - Consequently, if a tracked person is occluded or out of frame for 6+
    frames, all 6 non-conditioning spatial memory slots fill with empty or
    occluded frames, and after 16 frames the last good non-conditioning
    views are evicted from `non_cond_frame_outputs` altogether.
- **How Meta's reference does it**:
  - `cal_mem_score` (`coderef/sam3/sam3/model/video_tracking_multiplex.py:2740-2747`,
    `coderef/sam3/sam3/model/sam3_tracker_base.py:508-515`) computes
    `object_score_norm = where(logit > 0, logit.sigmoid() * 2 - 1, 0)` and
    averages `(object_score_norm * iou_score).mean()` across active objects in
    the bucket. Neither `cal_mem_score` nor `cal_mem_score_per_object` rescales
    `iou_pred` from `[0.5, 1.0]` to `[0, 1]` or applies a `126x126` overlap
    dilution penalty.
  - `frame_filter` (`coderef/sam3/sam3/model/sam3_multiplex_base.py:482-502`)
    walks backward from `frame_idx - 1` to `1` and keeps only past frames
    whose `eff_iou_score > mf_threshold` (`0.01`, up to `max_mem_frames = 6`),
    and drops the oldest kept frame's object pointer (`pos_and_ptrs[:-1]`)
    when `len(pos_and_ptrs) > max_mem_frames`.
  - Under Meta's `SAM3-DMS` (`arXiv:2601.09699v1`), multiplex multi-target
    memory decouples per-object eligibility at write time so an occluded or
    low-confidence object's slot does not write noisy memory while visible
    companion objects continue updating theirs.
- **How our patch works (`sam31_corrections.py:200-1021`, `IoUTap` + `MemoryEncodeFilter` + `MemorySelectionStep` + `MemoryTrackLoopGuard`)**:
  - Attaches to `SAM31Tracker` only (`attach_memory_selection`,
    `sam31_corrections.py:992-1021`; a checkpoint with `SAM3Tracker` is left
    unpatched).
  - `IoUTap` (`sam31_corrections.py:286-315`) wraps `tracker.sam_mask_decoder`
    to capture raw `iou_pred` before `_forward_propagation` discards it after
    `argmax`.
  - `cal_mem_score_per_object` (`sam31_corrections.py:200-231`) computes
    `(2 * sigmoid(logit) - 1) * max_t(iou_scores[o, t])` **per object**
    (`float32`, without Meta's bucket-wide `.mean()`), and
    `frame_filter_per_object` (`sam31_corrections.py:233-277`) filters each
    object slot independently against `MF_THRESHOLD` (`0.01`).
  - `MemoryEncodeFilter` (`sam31_corrections.py:318-430`, `ENCODE_MEM_PATCH`
    `sam31_corrections.py:172`) intercepts `SAM31Tracker._encode_new_memory`:
    strictly gated to `N_obj >= 2` (single-object `N_obj == 1` passes are
    unaffected), suppressing mask-memory encodes for slots whose score is
    `<= MF_THRESHOLD` so empty/occluded objects do not pollute shared
    multiplex spatial memory.
  - `MemorySelectionStep` (`sam31_corrections.py:447-916`) wraps `track_step`,
    `_prepare_memory_conditioned_features`, and `track_video_with_detection`:
    - When `N_obj == 1` (`focus_obj_idx` set / single slot), preserves the
      byte-identical single-object bank behavior (`diff_px == 0` verified in
      [`2026-11-15_sam31_dms_and_fixes.md`](../../../bench/results/2026-11-15_sam31_dms_and_fixes.md)).
    - When `N_obj >= 2` (`focus_obj_idx=None`), `_spatial_union_frames`
      (`sam31_corrections.py:614-649`) takes the spatial union of retained
      frames where *any* tracked object scored `> MF_THRESHOLD`, and
      `_build_remapped_non_cond` (`sam31_corrections.py:651-758`) assembles
      per-slot `obj_ptr` tokens across multi-bucket multiplex tensors.
    - Frees per-pass state in `end_of_pass` (`sam31_corrections.py:899-916`),
      guarded by `MemoryTrackLoopGuard` (`sam31_corrections.py:433-444`,
      `TRACK_LOOP_PATCH` `sam31_corrections.py:173`) around
      `track_video_with_detection` so no GPU tensors leak onto the cached
      `ModelPatcher`.
- **Why we did it, and what it changed**:
  - **Single-object (`N_obj == 1`)**: Turned **on by default** in the masked
    job chain on 2026-10-17 (`workflows/h3_config.py::MASKED_JOB_SAM_CORRECTIONS`
    at `workflows/h3_config.py:1441`; `workflows/masked_job_graph.py::memory_selection_of`
    at `workflows/masked_job_graph.py:205-215`): recovers occluded subjects
    (`first_frame_back_from_708 = 709` vs `714` on kitchen face probe with
    range fixed), with `diff_px = 0` parity preserved across the 2026-11-15
    upgrade.
  - **Multi-object (`N_obj >= 2`)**: Replaces the earlier `one_object_only=True`
    guard with Multiplex `SAM3-DMS` ([`2026-11-15_sam31_dms_and_fixes.md`](../../../bench/results/2026-11-15_sam31_dms_and_fixes.md)):
    write-time `MemoryEncodeFilter` + read-time spatial union recovers
    post-occlusion `strangers` focus IoU on `kitchen_occ_960_1009` (`0.0000`
    $\to$ `0.8333`) with `0` companion jumps and holds `fun_crowd_2200` /
    `regain_1419_1465` / `hall_1306_1418`, while trading similarity-vs-alone
    on low-scoring tiny seeds (`kitchen_face_group_680_760` `strangers`).
  - **Why SAM 3.0 (`SAM3Tracker`) is not patched**: On SAM 3.0 base, enabling
    memory selection brought the mask back at frame `702`, which on inspection
    was the back of the lead's hair rather than her face (`docs/wiki/decisions.md`,
    2026-10-17).

### 4. Propagation multimask pointer token (`correct_multimask_ptr`, `default=False`, `optional=True`)

- **How ComfyUI core does it**: `SAM31Tracker._forward_propagation` selects
  `best_idx = torch.argmax(iou_obj, dim=-1)` for the output mask, but projects
  `tokens_obj[:, 0]` through `obj_ptr_proj` instead of the winning branch's
  token.
- **How our patch works (`sam31_corrections.py:1048-1109`)**:
  `MultimaskTokenTap` (`MULTIMASK_TAP_PATCH` `sam31_corrections.py:187`, lines
  `1048-1063`) taps `sam_mask_decoder.transformer.forward`, and
  `CorrectedPropagationPtr` (`PROPAGATION_PATCH` `sam31_corrections.py:188`,
  lines `1066-1109`) wraps `SAM31Tracker._forward_propagation` so
  `tokens_obj[obj_range, best_idx]` is projected for the object pointer.
- **Status (`default=False`)**: Non-zero branches win on `28.75%–87.50%` of
  propagation object-steps in
  [`2026-11-15_sam31_dms_and_fixes.md`](../../../bench/results/2026-11-15_sam31_dms_and_fixes.md),
  nudging mask boundaries slightly (`IoU 0.9967–0.9996`) with mixed small
  moves across windows, so it stays opt-in (`optional=True, default=False`).

### 5. Detector global presence gating (`correct_detector_presence`, `default=False`, `optional=True`)

- **How ComfyUI core does it**: `SAM3Detector._detect` returns prompt logits
  without multiplying by Meta's global image-level `presence` probability
  `sigmoid(presence)`.
- **How our patch works (`sam31_corrections.py:1153-1180`)**:
  `CorrectedDetectorPresence` (`PRESENCE_PATCH` `sam31_corrections.py:195`,
  lines `1153-1180`) wraps `SAM3Detector._detect`, multiplying
  `sigmoid(presence)` into prompt probabilities in `float32` via
  `torch.logit(prob, eps=1e-6).clamp(-12.0, 12.0)`.
- **Status (`default=False`)**: Suppresses absent-prompt false positives to
  `0` detections (`>0.3`) and `0` spawned `SAM3_VideoTrack` tracks while
  preserving 100% of present-prompt counts in
  [`2026-11-15_sam31_dms_and_fixes.md`](../../../bench/results/2026-11-15_sam31_dms_and_fixes.md).
  Left opt-in (`optional=True, default=False`) because `presence` modulation
  slightly lowers present-prompt scores and would shift calibrated
  `MiniMaxH3SubjectTrack` detection thresholds on marginal frames.

---

## What `MiniMaxH3SAM31Corrections` leaves as ComfyUI computes it (`NOT_CORRECTED`)

`sam31_corrections.py::NOT_CORRECTED` (`sam31_corrections.py:86-93`) lists
six remaining places where ComfyUI core differs from Meta's SAM 3.1 reference
that the corrections node deliberately does **not** patch on the model:

| # | stage | ComfyUI core (`comfy/` & `comfy_extras/`) | Meta reference (`coderef/sam3`) | why we leave it unpatched on the model (or handle it in our nodes) |
|---|---|---|---|---|
| 1 | **Resize kernel** | `SAM3_Detect` (`comfy_extras/nodes_sam3.py:118`) uses un-antialiased bilinear (`common_upscale(..., "bilinear")`); `_prep_frame` (`comfy/ldm/sam3/tracker.py:209`) uses bicubic. | `v2.Resize((1008, 1008), bilinear)`: `antialias=False` in `coderef/sam3/sam3/model/sam3_image_processor.py:55`, `antialias=True` in video `coderef/sam3/sam3/model/io_utils.py:309`. | Lives inside node/tracker frame-prep helpers rather than `nn.Module` forwards; accounts for only `0.015` rel L2 on trunk features once `[-1, 1]` range is fixed ([`2026-10-07_sam3_core_against_meta.md`](../../../bench/results/2026-10-07_sam3_core_against_meta.md)). |
| 2 | **Query overlap removal (NMS)** | `SAM3_Detect` (`comfy_extras/nodes_sam3.py:217-226`) thresholds `scores.sigmoid() > threshold` with no NMS; `track_video_with_detection` (`comfy/ldm/sam3/tracker.py:1697-1701`) runs `_nms_masks` (`max(IoU, IoM) >= 0.5`). | Runs greedy IoM NMS at `0.1` (`coderef/sam3/sam3/model/sam3_multiplex_detector_utils.py:58-127`). *(Global `presence` gating is now available opt-in via `correct_detector_presence=True`.)* | Nested/duplicate queries from `detector` are deduplicated directly in `subject_tracks.py::one_each` (`subject_tracks.py:107-152`). |
| 3 | **Detect node's 2-pass `_refine_mask`** | `SAM3_Detect` (`comfy_extras/nodes_sam3.py:42-85, 232-235`) crops each detected box (+10% pad), runs `model.forward_segment` twice at `1008x1008`, and unions with the coarse mask (`torch.maximum(binary, refined)` at line 85). | Not in Meta's `Sam3ImageProcessor`; Meta uses `forward_Lock` / `forward_segment` only for interactive point/box prompts. | `MiniMaxH3SubjectTrack` routes detection and tracking calls through `subject_track.py::_sam_callables` (`subject_track.py:2614-2742`, calling `SAM3_Detect.execute` at `subject_track.py:2636` and `SAM3_VideoTrack.execute` at `subject_track.py:2696`). |
| 4 | **Hole-fill / sprinkle area (`max_area=16`)** | `track_video_with_detection` (`comfy/ldm/sam3/tracker.py:116-136, 1747-1748`) runs `fill_holes_in_mask_scores(current_out["pred_masks"], max_area=16)` on low-res `288x288` masks before memory encoding, leaving `pred_masks_high_res` (`1008x1008`, used for output) untouched. | SAM 3.1's builder (`coderef/sam3/sam3/model_builder.py:1138`) sets `fill_hole_area=0` (disabled); SAM 3.0 used `16`. | Only touches low-res masks fed into `_deferred_memory_encode`, not the output masks; our tracker post-processes output masks with `subject_tracks.py::drop_specks` (`subject_tracks.py:186-217`). |
| 5 | **Multi-object occlusion suppression** | `_suppress_recently_occluded` (`comfy/ldm/sam3/tracker.py:1486-1507, 1749-1751`) suppresses at `max(IoU, IoM) >= 0.3` on `pred_masks` (memory) only, and `_deferred_memory_encode` (`comfy/ldm/sam3/tracker.py:1514-1524`) clamps non-winning pixels when a mask shrinks below `0.3`. | `suppress_overlapping_based_on_recent_occlusion` (`coderef/sam3/sam3/model/video_tracking_multiplex.py:2178-2231`) suppresses at `IoU > 0.7` on both low-res and high-res output masks. | Inoperative when `N_obj == 1` (`comfy/ldm/sam3/tracker.py:1490`, `1514`), which is how `MiniMaxH3SubjectTrack` runs every tracking pass (`max_objects=1`). |
| 6 | **Threshold & resize order on output** | `track_video_with_detection` (`comfy/ldm/sam3/tracker.py:194-198, 1769-1776`) thresholds `masks_out > 0` at `1008x1008` and bit-packs into uint8; `SAM3_TrackPreview` (`comfy_extras/nodes_sam3.py:437-439`) resizes unpacked bools with `nearest`, while `SAM3_TrackToMask` (`comfy_extras/nodes_sam3.py:516-517`) resizes unpacked `0/1` floats with `bilinear` to `(H, W)`. | Interpolates float logits to `(H, W)` with `bilinear` and thresholds `> 0.0` once (`coderef/sam3/sam3/model/video_tracking_multiplex.py:1478-1485`). | `MiniMaxH3SubjectTrack` re-thresholds the returned mask at `> 0.5`, and `MiniMaxH3MaskedSource` grows and feathers the mask on the render canvas (`video_mask.py:747-762, 1990-2025`). |

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
   ([`2026-10-04_masked_v2v_band.md`](../../../bench/results/2026-10-04_masked_v2v_band.md)).
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
   ([`2026-10-08_one_person_track_against_core_multi_object.md`](../../../bench/results/2026-10-08_one_person_track_against_core_multi_object.md)).
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

`MiniMaxH3SubjectTrack` (`subject_track.py:2756+`, wired by default in
`workflows/h3_config.py::SUBJECT_TRACK` at `workflows/h3_config.py:1265-1267`
and in job previews at `workflows/masked_job_graph.py:300-307`) replaces
core's text-conditioned video loop with a shot-by-shot pipeline that uses
ComfyUI core's `SAM3Detector` and `SAM31Tracker` as low-level primitives:

| stage | our function & lines | what it does and how it differs from ComfyUI core | why we did it, and what we observed |
|---|---|---|---|
| **1. Detector & trunk callables** | `subject_track.py::_sam_callables` (`subject_track.py:2614-2742`) | Calls `SAM3_Detect.execute` (`subject_track.py:2636`), exposes `trunk_of(image)` calling `vision_backbone.trunk(x)` (`comfy/ldm/sam3/sam.py:223-295`) to get `[N, 72, 72, 1024]` features at `1008x1008`, and delegates single-object tracking to `SAM3_VideoTrack.execute` (`subject_track.py:2696`). | Exposes the `72x72x1024` ViTDet trunk features needed to match the subject's identity across scene cuts while keeping node calls compatible with `ModelPatcherDynamic`. |
| **2. Tokenizer `:1` fast-path bypass** | `subject_track.py::counted` (`subject_track.py:427+`) | Strips any existing `:N` suffix and returns bare `name` when `limit <= 1`, or `f"{name}:{int(limit)}"` when `limit >= 2`. | Fixes `comfy/text_encoders/sam3_clip.py:51-54` (`SAM3TokenizerWrapper.tokenize_with_weights`), which passes the raw string `"person:1"` to the BPE tokenizer when `len(parsed) == 1` and `weight == 1`, encoding literal `:1` into the text tokens. |
| **3. Deduplicating un-NMS'd detector queries** | `subject_tracks.py::one_each` (`subject_tracks.py:107-152`), `subject_track.py::one_detection_each` (`subject_track.py:1294-1336`) | Sorts detections on a frame by area (largest first) and drops any mask that lies `>= 0.8` (`SAME_THING = 0.8`, `subject_tracks.py:101`) inside a kept larger mask—**unless** the highest-scoring detection in that group has area `< 0.5` (`A_PART_UNDER = 0.5`, `subject_tracks.py:104`) of the largest mask, in which case the largest mask is a background/merged superset and is dropped first (`subject_tracks.py:124-138`). | Core's `SAM3_Detect` (`comfy_extras/nodes_sam3.py:217-226`) runs no NMS across its 200 queries, so one person is regularly returned 2–3 times (full body + torso + head). Measured on `fun` and `lotsofpeople` ([`2026-10-08_one_person_track_against_core_multi_object.md`](../../../bench/results/2026-10-08_one_person_track_against_core_multi_object.md)): collapses nested duplicate queries to 1 mask per person while keeping a smaller foreground person standing in front of a larger background figure. |
| **4. Letterbox-aware shot cut detection** | `subject_track.py::flat_borders`, `cut_scores`, `find_cuts`, `auto_cuts` (`subject_track.py:424-518`) | Strips constant top/bottom letterbox and left/right pillarbox bars (`flat_borders`, `subject_track.py:424-454`), downsamples to `192x108` (`SCORE_SIZE`), computes $1 - r$ of adjacent-frame spatial gradients (`cut_scores`), and thresholds at `median + 6 * MAD` clamped to `[0.15, 0.35]` (`auto_cuts`, with `MIN_SHOT = 6` frames). | SAM 3's tracker cannot cross a scene cut without either losing the person or jumping to whoever now stands in the old spot. Stripping flat borders (`flat_borders`) fixed missed cuts on letterboxed sources where constant black bars diluted frame-to-frame gradient differences ([`2026-10-06_bordered_source_cuts.md`](../../../bench/results/2026-10-06_bordered_source_cuts.md)). |
| **5. Multi-tracker exclusion (`others`)** | `subject_track.py::held_people` (`subject_track.py:1349-1384`), `subject_tracks.py::in_the_way` (`subject_tracks.py:562-574`) | When a second or third tracker runs after an earlier one (`workflows/masked_job_graph.py:288-303`), `others` carries the union of earlier trackers' masks. `held_people` removes any candidate detection on a pick, match, or regain frame that overlaps `others` by `>= 0.8` (`SAME_THING`), and `in_the_way` prevents `regain` from firing when `others` covers `>= 0.3` (`COVERED_BY_OTHER`) of the lost subject's last known mask. | Prevents two trackers in a multi-subject preview from picking or switching onto the same person during a cross-over ([`2026-10-08_one_person_track_against_core_multi_object.md`](../../../bench/results/2026-10-08_one_person_track_against_core_multi_object.md)). |
| **6. Picking the main subject & two-view trunk signature** | `subject_track.py::choose`, `top_third`, `head_of`, `signature`, `relative`, `main_subject` (`subject_track.py:536-685, 1094-1140`) | On the pick frame (`pick_on="automatic"` scans the first shot's sample frames for the highest detector score, or uses `a frame I name`), filters `person` detections to those containing a detected `head` (`head_of`, `overlap >= 0.5`), chooses by `largest`, `most central` (default in `workflows/h3_config.py:1265`), `leftmost`, `rightmost`, or `person N`, and pools L2-normalized $72\times 72$ ViTDet trunk features over **two views**: `top_third` of the person mask and the `head` mask. When other people are on the pick frame, `relative` (`subject_track.py:617-633`) subtracts the mean signature of those other people and re-normalizes. | Requiring a `head` detection stopped non-person false positives (such as a hanging microphone at the top of the frame, [`2026-10-06_subject_track_defaults.md`](../../../bench/results/2026-10-06_subject_track_defaults.md)) from winning `pick`. Two-view (`top_third` + `head`) contrastive trunk matching (`relative`) raised the separation between the target subject and bandmates across cuts (`MATCH_FLOOR = 0.78`, `PLAIN_FLOOR = 0.60`, `SOLO_MARGIN = 0.05` in `auto_match`, `subject_track.py:706-736`). **Known limit**: ViTDet trunk features are appearance/pose features, not a face-identity embedding; in a crowd or after a wardrobe/lighting change, cosine scores cluster tightly (`0.72–0.79`) and shots need `seeds` or `corrections` ([`2026-10-07_subject_track_under_nudge.md`](../../../bench/results/2026-10-07_subject_track_under_nudge.md), [`docs/wiki/masked_v2v/who_is_who.md`](who_is_who.md)). |
| **7. Bidirectional single-object tracking per shot & `ScoreTap`** | `subject_track.py::follow`, `ScoreTap` (`subject_track.py:2254-2306`), `_sam_callables.track` (`subject_track.py:2696`) | Within one shot `[start, end)`, calls `SAM3_VideoTrack.execute(segmenter, frames, conditioning=None, initial_mask=seed[None], max_objects=1)` (`comfy_extras/nodes_sam3.py:282-314`). If the seed frame `at` is inside the shot (`at > start`), runs one pass forward on `shot[at:]` and a second pass forward on the time-reversed slice `shot[:at + 1].flip(0)`. `ScoreTap` wraps `SAM31Tracker.track_step` (avoiding collisions with `_forward_propagation` patches) to record per-frame `score_obj` (`object_score_logits`) into `shot_table`. | Passing `conditioning=None, max_objects=1` bypasses core's per-frame detector, slot cap (`INTERNAL_MAX_OBJECTS = 64`), `keep_alive` blanking, and `HIGH_CONF = 0.8` mask overwrites altogether: the tracker follows the one seeded mask bidirectionally through the shot using propagation + memory attention alone ([`2026-10-08_one_person_track_against_core_multi_object.md`](../../../bench/results/2026-10-08_one_person_track_against_core_multi_object.md)). |
| **8. Gallery expansion & mid-shot `regain`** | `subject_track.py::gallery_frames` (`subject_track.py:1143-1180`), `regain` (`subject_track.py:1572-1677`), `subject_tracks.py::plausible`, `doubted`, `trusted`, `take_back`, `take_back_by_place` (`subject_tracks.py:259-363, 465-550`) | After each accepted shot, `gallery_frames` adds up to `GALLERY_PER_SHOT = 4` trusted frames (`trusted`, where the mask is between `0.25x` and `2.5x` of the shot median and hasn't jumped by `> 1.0` mask-widths) to the reference gallery. Inside a shot, if the tracked mask vanishes for `LOST_RUN = 6` frames (`subject_track.py:338`) or jumps (`doubted`), `regain` searches forward at `REGAIN_STRIDE = 6` for a candidate matching the gallery (`>= 0.78`, or the sole person when `alone`), tracks forward from that new seed, trims pre-loss partial mask collapse (`take_back`, `< 0.5` of running median area) and box drift (`take_back_by_place`, shift `> 0.5` of mask span), and splices the regained tail in if it adds net tracked frames. | Recovers subjects who leave the frame or are fully occluded mid-shot and return later, and strips the 1–5 frames where SAM's mask collapses onto a sleeve or a bystander's shoulder just before losing the subject (`take_back`, `take_back_by_place`, measured on `fun` shots in [`2026-10-08_one_person_track_against_core_multi_object.md`](../../../bench/results/2026-10-08_one_person_track_against_core_multi_object.md)). |
| **9. Gap bridging & speck removal** | `subject_tracks.py::unbroken` (`subject_tracks.py:387-419`), `drop_specks` (`subject_tracks.py:186-217`) | `drop_specks` removes connected components smaller than `SPECK_SHARE = 0.05` of the mask's area on a frame unless those pixels also appear on an adjacent frame (`t - 1` or `t + 1`). `unbroken` bridges empty runs of up to `FILL_GAP = 2` frames by copying the nearest tracked frame when the bounding boxes before and after the gap are plausible. | Core's `fill_holes_in_mask_scores(..., max_area=16)` (`comfy/ldm/sam3/tracker.py:1747-1748`) only cleans low-res `288x288` memory masks, leaving transient 1-frame background specks in `pred_masks_high_res` (`1008x1008`). Because `MiniMaxH3MaskedSource` pools over 4-frame latent steps (`FRAME_PER_TOKEN = 4`), a single-frame speck punches a 4-frame hole in the plate; `drop_specks` removes transient 1-frame islands while keeping small real features (like an ear or hand) that persist across adjacent frames. |
| **10. Seed boxes, typed corrections, shot table & pose boxes** | `subject_track.py::parse_seeds` (`subject_track.py:791-943`), `parse_corrections` (`subject_track.py:946-1063`), `shot_table.py::build` & `MiniMaxH3SaveShotTable` (`shot_table.py:86-250`), `subject_boxes.py::MiniMaxH3SubjectBoxes` (`subject_boxes.py:79-122`) | `seeds` accepts `shot N: frame F x0,y0,x1,y1` (written from the owner's clicks on the who-is-who page by `bench/page_answers_to_cast.py` and `bench/cast_file.py::seeds`), picks the detector mask with highest IoU (`>= 0.1`) against the box (or uses the box itself when no detection overlaps), and adds that frame to the gallery as ground truth. `corrections` accepts `shot N: person K [@F]`, `shot N: none`, `cut F`, `no cut F`, `hold F..G`, `drop F..G` (`subject_tracks.py:577-709`). `MiniMaxH3SaveShotTable` writes `*_shots.json` and numbered candidate tiles (`*_shots_00001_.png`); `MiniMaxH3SubjectBoxes` turns `[N, H, W]` masks into `[N, 5]` boxes for `MiniMaxH3BodyPose`. | Replaces ComfyUI core's trial-and-error `object_indices` string in `SAM3_TrackToMask` (`comfy_extras/nodes_sam3.py:484-509`) with a machine-readable `shot_table` JSON and numbered candidate tiles that drive `bench/who_page.py`, `bench/shot_to_load.py`, and `MiniMaxH3MaskedSource`'s `cut_gate` (`video_mask.py:813-860`). |

---

## How many, and for how long

- Sixteen people for one phrase is inside what Meta's benchmarks cover;
  thirty-two is at their edge, in images and in video. The benchmark record,
  its fifth line and "People". Sixteen is also the size of one group in SAM
  3.1's tracker (`RELEASE_SAM3p1.md`, Object Multiplex).
- Benchmark clips are short. A phrase held over minutes, across cuts, is
  this pack's own machinery and not something the model was tested on.
- **A subject who shrinks a great deal through one long shot is outside what
  the video benchmark covers.** A wide spread between a track's largest and
  smallest annotated frame is common there, but most of it is an object
  entering, leaving or being cut by the frame's edge; a sustained shrink of
  that order with no gap, over a long track, is all but absent for people
  and for vehicles, and growing is commoner than shrinking (counted from
  the annotation files by a helper of the lead session, 2026-10-07; the
  script and its aggregates are not yet tracked). Untested is not failing:
  on the one such window measured here the shipped track held as one mask
  ([`2026-10-07_subject_track_calls_on_masks.md`](../../../bench/results/2026-10-07_subject_track_calls_on_masks.md)).
  The annotation files keep slivers of a few pixels and do not say why a
  frame is unannotated.
- A small object in the video benchmark leaves and returns often, and is
  usually gone only briefly in third-person footage and for much longer in
  egocentric footage. A regain should expect short dropouts as normal:
  [`2026-10-07_sam3_benchmark_targets.md`](../../../bench/results/2026-10-07_sam3_benchmark_targets.md),
  which also has how many instances a pair holds for the words this lane
  asks (`person`, `head`, `hand`, `hair`).

## What is not known

- Whether a box or a click through ComfyUI's detect node (`SAM3_Detect`'s
  `points` / `boxes` path in `comfy/ldm/sam3/detector.py:225-312`) selects
  one person more cleanly than IoU-matching `person` detections against a
  seed box in `subject_track.py::parse_seeds`. Read in the code, not run.

## Where the evidence is

| record or note | what it holds |
|---|---|
| [`2026-10-07_sam3_benchmark_phrases.md`](../../../bench/results/2026-10-07_sam3_benchmark_phrases.md) | Meta's four datasets counted from their annotation files: phrase shapes, how often each is asked where nothing is annotated, sizes, people, video. `bench/sam3_dataset_phrases.py` reproduces it. The datasets are gated and are not in this repository |
| [`2026-10-07_mryolk_phrases.md`](../research/masking/2026-10-07_mryolk_phrases.md) | what Meta's README, notebooks, tests and eval code do with a phrase, by file and line |
| [`2026-10-07_sam3_precision_arms.md`](../../../bench/results/2026-10-07_sam3_precision_arms.md) | precision, and three departures of ComfyUI from Meta measured blind |
| [`2026-10-07_sam3_core_against_meta.md`](../../../bench/results/2026-10-07_sam3_core_against_meta.md) | ComfyUI against Meta's code stage by stage on equal inputs (`[0, 1]` vs `[-1, 1]`, `quick_gelu` vs `gelu`, `presence` gating, 32-object seeded tracking) |
| [`2026-10-07_sam31_corrections_on_a_queue.md`](../../../bench/results/2026-10-07_sam31_corrections_on_a_queue.md) | `MiniMaxH3SAM31Corrections` (`correct_image_range` and `correct_text_activation`) accepted on a server's queue |
| [`2026-11-15_sam31_dms_and_fixes.md`](../../../bench/results/2026-11-15_sam31_dms_and_fixes.md) | Multiplex `SAM3-DMS` (`memory_selection`), `correct_multimask_ptr`, and `correct_detector_presence` benchmarked on `sam3.1_multiplex_fp32.safetensors` across `N_obj == 1`, `N_obj >= 2`, `SAM3_Detect`, and `SAM3_VideoTrack` |
| [`2026-10-10_sam31_corrections_on_one_stretch.md`](../../../bench/results/2026-10-10_sam31_corrections_on_one_stretch.md) | `MiniMaxH3SAM31Corrections` (`image_range` and `text_activation` alone and together) on the 447-frame kitchen stretch of `fun` (`shots 1–6`): why `text_activation` alone changes 0 detections on `person`/`head` and how `image_range` shifts trunk cosine similarity |
| [`2026-10-08_one_person_track_against_core_multi_object.md`](../../../bench/results/2026-10-08_one_person_track_against_core_multi_object.md) | `MiniMaxH3SubjectTrack` (seeded per shot, `max_objects=1`, with `one_each`, `take_back`, `take_back_by_place`, `drop_specks`, and `others`) against ComfyUI core's text-prompted `SAM3_VideoTrack` (`max_objects=8` and `64`) on `fun` and `lotsofpeople` |
| [`2026-10-06_bordered_source_cuts.md`](../../../bench/results/2026-10-06_bordered_source_cuts.md), [`2026-10-06_subject_track_defaults.md`](../../../bench/results/2026-10-06_subject_track_defaults.md), [`2026-10-07_subject_track_under_nudge.md`](../../../bench/results/2026-10-07_subject_track_under_nudge.md), [`2026-10-07_subject_alone_or_in_a_group.md`](../../../bench/results/2026-10-07_subject_alone_or_in_a_group.md) | border stripping for cut detection (`flat_borders`), requiring a detected `head` on the pick frame (`head_of`), how `pick`, `match` and `regain` hold under a 1-level nudge, and what following a subject alone or in a group does |
| [`docs/wiki/research/masking/2026-10-07_mryolk.md`](../research/masking/2026-10-07_mryolk.md) | SAM 3.1 in ComfyUI end to end |
| [`2026-10-07_mryolk_stage_table.md`](../research/masking/2026-10-07_mryolk_stage_table.md) | the same, one row per stage with the file and lines on ComfyUI's side and on Meta's: the place to look up where a stage lives, the rules between followed objects among them (its tracker rows) |
| [`2026-10-18_sam31_minimal_core_fixes.md`](../research/masking/2026-10-18_sam31_minimal_core_fixes.md) | the 6 minimal surgical fixes (~25 lines across 4 files) that would bring ComfyUI core's existing SAM 3 / 3.1 detector, tokenizer, and single-shot tracker to numerical and functional parity with Meta's reference |
| `docs/research/masking/2026-10-06_mrsun.md` (no longer in the tree since 2026-10-11) | Meta's video pipeline for one phrase, frame by frame, and what ComfyUI's port does instead at each step and at a cut. A reading; nothing was run for it |

