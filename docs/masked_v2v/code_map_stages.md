# Masked V2V Code Map — Part 1: End-to-End Mask & Tensor State Mutation Ledger

This companion document to [2026-10-26_code_map_and_simplification_plan.md](../research/masking/2026-10-26_code_map_and_simplification_plan.md)
traces every mask, tensor, `.npz`, `.json`, and `.mkv` state mutation from Stage 1 (pre-capture tracking and
parts segmentation) through Stage 5 (final delivery assembly).

For exact `StartLine-EndLine` coordinates and `cannot_see` boundaries, see [CODE_MAP_INDEX.md](CODE_MAP_INDEX.md)
or run `$H3_PYTHON bench/masked_code_map.py <symbol>`. For contracts, window padding, and duplicate logic, see
[code_map_contracts.md](code_map_contracts.md).

---

## Stage 1 — Pre-Capture Tracking, Parts, Strangers, Depth, References & Job Creation

### Stage 1.1a — SAM 3.1 Subject & Head Tracking (`subject_track.py`)

- **Symbols**: `subject_track.py::MiniMaxH3SubjectTrack.execute`, `subject_track.py::follow`, `subject_track.py::_track_head_span`, `subject_track.py::_span_head`, `subject_track.py::name_drawn`, `subject_track.py::auto_match`
- **Reads**:
  - Source clip `<shot>.mp4`, `<video>/cast.jsonl` (`bench/cast_file.py::read`), and `<video>/shots.json`.
  - Per-shot `shot.gallery` crop paths (`da6921dc`) or explicit `--seed` boxes / `corrections.json`.
- **Transforms & Constants**:
  - Runs SAM 3.1 propagation across the shot to produce boolean person masks `masks` (`(N, H, W)`) and `boxes` (`(N, 4)`).
  - When `head_mode` is active (`parse_head_mode`), derives `head_masks`, `head_boxes`, `head_scores`, and `head_source`.
  - Post-`0121b793` / `cf244cb1`: `name_drawn` enforces the 3-group `cast.jsonl` taxonomy (`named` vs `unnamed_recurring` vs `incidentals`, refusing ungrouped or overlapping entries) and writes `empty_runs` on both `subject_track` and `head_track` JSONs (`ca8fdcc9`).
- **Writes & Overwrites**:
  - `<preview>_<subj>_track.mp4` (`8px` mask preview), `<preview>_<subj>_track.json` (`boxes`, `scores`, `empty_runs`), `<preview>_<subj>_head_track.mp4`, and `<preview>_<subj>_head_track.json`.
- **Audit & Traps**:
  - **Forearm Bridge vs Empty Head Track (`4682-4685`)**: SAM 3.1 person tracking can stay alive on an occluded subject's arm/sleeve while `head_track` records `empty_runs` (`0 px`). Downstream `sapiens2_parts.py` crops around the forearm box unless gated by `head_track`.
  - **Gallery vs Replacement Still (`da6921dc`)**: `auto_match` must match against `shot.gallery` (the filmed person in the source clip), never the replacement identity still in `cast.jsonl`.

### Stage 1.1b — Sapiens-2B 28-Class Body-Part Segmentation (`sapiens2_parts.py`)

- **Symbols**: `sapiens2_parts.py::subject_parts`, `sapiens2_parts.py::MiniMaxH3SubjectParts.execute`
- **Reads**:
  - Source frames `(N, H, W, 3)` and person track `masks` / `boxes` from Stage 1.1a.
- **Transforms & Constants**:
  - Crops a `1.25x` padded bounding box (`pad_ratio=0.25`) around each tracked person box, resizes to `1024x768`, runs Sapiens-2B (`torch.bfloat16`), and argmaxes 28 classes (`0=background`, `1=Apparel`, `2=Face_Neck`, `3=Hair`, `23=Upper_Lip`, `24=Lower_Lip`, `25=Lower_Teeth`, `26=Upper_Teeth`, `27=Tongue`).
  - Zeroes out predictions outside `person_mask` (`cls_full[~person_mask] = 0`).
- **Writes & Overwrites**:
  - `<preview>_<subj>_parts.mp4`, `<preview>_<subj>_parts.npz` (`classes` uint8 `(N, H, W)`), and `<preview>_<subj>_parts.json`.
- **Audit & Traps**:
  - **Boundary Hair Loss (`4912`)**: Because `cls_full` is clipped to SAM 3.1's `person_mask`, any hair strands SAM 3.1 left outside its body silhouette are zeroed before Stage 2 sees them; Stage 2.4 (`head_tube_parts.py`) and Stage 2.8 (`her_hair_in_hole`) recover them via whole-frame/head-tube pass.
  - **Occluded Forearm Crop Hallucination (`4682-4685`)**: When a subject's head is behind another person (`head_track` empty) and only her forearm is visible (`82x204 px`), cropping `1024x768` around the forearm causes Sapiens-2B to hallucinate `Face_Neck=541 px` and `Hair=427 px` on skin/cloth.

### Stage 1.2 — Whole-Frame Strangers & Unclaimed People Pass (`bench/strangers.py`)

- **Symbols**: `bench/strangers.py::run`, `bench/strangers.py::people_pass`, `bench/strangers.py::sam_people`, `bench/strangers.py::handed_back`, `bench/strangers.py::others_of`, `bench/strangers.py::held_others`
- **Reads**:
  - Source clip `<shot>.mp4`, all tracked subjects' `_track.mp4` and `_parts.npz`, and `<preview>_people.npz` (SAM 3.1 whole-shot person discovery).
- **Transforms & Constants**:
  - Runs whole-frame Sapiens-2B (`people_pass`) to get `classes` `(N, H, W)` and `person` mask (`c > 0`).
  - Subtracts owned subject tracks to form `other_people` and `core` (high-confidence stranger body/face core), holds stranger masks across short dropouts (`hold_frames=8`), and hands back (`handed_back`) stranger regions that vanish for `> 8` frames while touching a tracked subject (`handed_back_on_no_other_person`).
- **Writes & Overwrites**:
  - `<preview>_strangers.mp4`, `<preview>_strangers_classes.npz` (`classes`, `handed_back`), and `<preview>_strangers.json` (with 13-item `cannot_see` array).
- **Audit & Traps**:
  - **Unowned SAM `people.npz` Dropped Where `c == 0` + `core[i]` `NO_CORE` Trap (`1458`)**: In `people_pass` (`534-548`),
    unowned SAM 3.1 detections (`o[3] is None`) are NOT unioned into `people[i]` where whole-frame Sapiens2 returns `c == 0`,
    AND `core[i]` (`564`) requires Sapiens2 classes (`NO_CORE = {0, 1, 17, 18, 19, 20}`). Even if `people[i]` were unioned,
    `handed_back` (`201-233`) checks `(comp & core[i]).any()` and drops any connected component touching a subject box when
    `core[i]` is empty (`43,190 px` of man's torso on `1458`).

### Stage 1.3 — Monocular Metric Depth (`bench/measure_moge2_depth.py`)

- **Symbols**: `bench/measure_moge2_depth.py::run_measurement`
- **Reads**:
  - Source clip `<shot>.mp4`.
- **Transforms & Constants**:
  - Evaluates MoGe-2 (`Ruicheng/moge-2-vitl-normal`) per frame and downsamples by `stride=4` to float16 metric depth `(N, H//4, W//4)`.
- **Writes & Overwrites**:
  - `<preview>_depth.npz` (`depth` float16, `stride=4` plus `npz_writer_fields` provenance) and `<preview>_depth.json`.
- **Audit & Traps**:
  - Downsampled `stride=4` depth is bilinearly or nearest-neighbor upsampled in Stage 2.1 (`read_depth_npz`); thin hair strands against background take background depth unless rescued by Stage 2.4 (`sam31_head_tube`).

### Stage 1.4 — 308-Keypoint Whole-Body Pose (`bench/pose_along_shot.py`)

- **Symbols**: `bench/pose_along_shot.py::run`, `bench/pose_along_shot.py::read`
- **Reads**:
  - Source clip `<shot>.mp4` and `<preview>_<subj>_track.json` boxes.
- **Transforms & Constants**:
  - Runs Sapiens-1B-308-pose (`torch.bfloat16`) inside each subject's tracked box; extracts 68-point face/ear/brow/mouth landmarks, pose confidence scores, and left/right ear visibility.
- **Writes & Overwrites**:
  - `<preview>_<subj>_pose.npz` (`keypoints` `(N, 308, 2)`, `scores` `(N, 308)`) and `<preview>_<subj>_pose.json`.
- **Audit & Traps**:
  - Pose landmarks on profile/back-of-head frames (`< 0.30` score) still project onto the 2D head silhouette; `bench/_lib/face_points.py::face_on_mask` and `bench/head_tube_parts.py::cross_check_face_points` threshold by confidence.

### Stage 1.5 — Head Reference Video & 308-Point Face Drawing (`bench/head_reference.py`, `bench/crop_reference_to_head.py`, `bench/draw_face_points.py`)

- **Symbols**: `bench/head_reference.py::run`, `bench/crop_reference_to_head.py::run`, `bench/draw_face_points.py::run`, `bench/draw_face_points.py::draw`
- **Reads**:
  - Replacement identity still `<cast_ref>.png`, `<preview>_<subj>_pose.npz`, `<preview>_<subj>_parts.npz`, and `<preview>_<subj>_head_track.mp4`.
- **Transforms & Constants**:
  - `bench/head_reference.py::run` warps/crops the replacement identity still onto the tracked head trajectory (`_head_ref.mp4`).
  - `bench/draw_face_points.py::run` renders synthetic 3D-projected face landmarks, sculpted eyes (`sculpted_eye`), and mouth state (`anchored_mouth`, `anchored_closed_mouth`, `bend_mouth`) onto `<job>_<subj>_drawn.mp4`.
- **Writes & Overwrites**:
  - `<preview>_<subj>_head_ref.mp4`, `<job>_<subj>_drawn.mp4`, `<job>_<subj>_drawn_sheet.jpg`, and `<job>_<subj>_drawn.json`.
- **Audit & Traps**:
  - **Double-Write of `_sheet.jpg` & `.json`**: `bench/draw_face_points.py::run` writes `_sheet.jpg` and `.json` at `1622-1623`, and `bench/shot_to_load.py::drawn_form` immediately overwrites both at `4035-4041`.
  - **`4647h` `<Video 1>` Bare Frames (`76533a22` / `9c5a6841`)**: When `head_track` had gaps (`4682-4685`, `4702-4714`, `4721-4746`), `second_hfp_4647h_drawn.mp4` had `43/79` bare black frames while the prompt told H3 `<Video 1>` was a continuous guide (`4647i` remake queued under Build 3).

### Stage 1.6a — Part-on-Track Filtering (`bench/shot_to_load.py`)

- **Symbols**: `bench/shot_to_load.py::part_on_track`, `bench/shot_to_load.py::hold_to_track`, `bench/shot_to_load.py::make_part_on_track`
- **Reads**:
  - `<preview>_<subj>_parts.mp4` (or `.npz`), `<preview>_<subj>_track.mp4`, and `<preview>_<subj>_head_track.mp4`.
- **Transforms & Constants**:
  - Filters part masks to pixels within `8px` (`_on_track_8px`) of the subject's SAM 3.1 track and runs `hold_to_track` (`1280-1391`), which carries forward the last seen head/part pixels across frames where the part mask dropped out (`hold_frames=8`).
- **Writes & Overwrites**:
  - `<preview>_<subj>_part_head_hair_on_track_8px.mkv` and `<preview>_<subj>_part_head_hair_on_track_8px.json`.
- **Audit & Traps**:
  - **Stale `hold_to_track` Smearing (`4682-4685`, `4702-4714`)**: `hold_to_track` freezes frame `4681`'s `16,580 px` head mask onto `4682-4685` when the woman walks behind the man, punching a hole through the occluding man's chest, and smears a stale `737 px` spot on `4702-4709` while her real head moves `62 px` right. Settled for retirement in Morning Build 1 (`Part 5e`).

### Stage 1.6b — Load Job JSON & Graph Compilation (`bench/shot_to_load.py`, `bench/masked_job.py`, `workflows/masked_job_graph.py`)

- **Symbols**: `bench/shot_to_load.py::step_job`, `bench/shot_to_load.py::load_job`, `bench/shot_to_load.py::carry_words_of`, `bench/shot_to_load.py::head_marks_of`, `bench/masked_job.py::build`, `workflows/masked_job_graph.py::load`
- **Reads**:
  - `<video>/shots.json`, `<video>/cast.jsonl`, preview `.json` / `.npz` artifacts, and CLI overrides.
- **Transforms & Constants**:
  - Enforces settled `shot.recipe` (`cbd39fbe`: `hold_frames=0`, `feather_px=8`, `grow_px=0`, `floor_head_scale=1.1`, `mask_fill=hole_and_hair`, `min_Component_pixels=150`, `protect_ears="off"`, `protect_facial_hair="keep"`) unless listed in `shot.shot_overrides`.
  - Computes `head_marks_of` (`2636-2691`, scoped strictly to own subject's masks after `9c5a6841`) and builds `<job>.json` and `<job>_api.json`.
- **Writes & Overwrites**:
  - `data/<video>/jobs/<job>.json` and `data/<video>/jobs/<job>_api.json`.
- **Audit & Traps**:
  - Before `9c5a6841`, `head_marks_of` read `keep_out_masks` (which included other subjects' tracks), causing a bystander's face points to trigger `head_marks` on the target subject.

---

## Stage 2 — Capture Pre-Pass & Mask Settlement (`bench/capture_masked_run.py`, `bench/head_tube_parts.py`)

### Stage 2.1 — Depth Mediations & Pixel Ownership (`bench/capture_masked_run.py`)

- **Symbols**: `bench/capture_masked_run.py::_files`, `bench/capture_masked_run.py::read_depth_npz`, `bench/capture_masked_run.py::_depth_medians_and_owner`, `bench/capture_masked_run.py::owner_map`
- **Reads**:
  - `<preview>_depth.npz`, subject `_track.mp4` / `_parts.npz`, and `<preview>_strangers.mp4` / `_strangers_classes.npz`.
- **Transforms & Constants**:
  - Upsamples MoGe-2 depth to `(N, H, W)` and computes per-frame median depth per tracked subject and stranger component (`_depth_medians_and_owner`).
  - Assigns per-pixel ownership (`owner_map`: `0=unowned`, `1..K=subject index`, `stranger_id`) using depth ordering (`depth_margin=0.05m`) when silhouettes overlap.
- **Writes & Overwrites**:
  - In-memory `owner`, `front_other_mask`, and `depth_settled` arrays; later persisted into `owners.npz` (`2875`).
- **Audit & Traps**:
  - If a stranger is missing from `<preview>_strangers.mp4` (Stage 1.2 `NO_CORE` trap on `1458`), `owner_map` never sees that person as `front_other_mask` and awards overlapping pixels to the background subject.

### Stage 2.2 — Raw Input Reading & Keep-Out Carving (`bench/capture_masked_run.py`)

- **Symbols**: `bench/capture_masked_run.py::_files`, `bench/capture_masked_run.py::_keep_in_front`, `bench/capture_masked_run.py::carve_keep_out`, `bench/capture_masked_run.py::_bridge_severed`, `bench/capture_masked_run.py::grid_carve`
- **Reads**:
  - Subject's raw part video (`_part_head_hair_on_track_8px.mkv`), `keep_out` videos (other subjects' tracks + strangers), and `front_other_mask`.
- **Transforms & Constants**:
  - Stores uncarved input in `as_read[by]` (`2286`).
  - Subtracts `keep_out` masks that are in front of `by` (`_keep_in_front`), runs `_bridge_severed` (`45f3eaf1`, reconnecting thin severed neck/hair bridges when safe), and applies `16x16` latent grid-aware carving (`grid_carve`).
- **Writes & Overwrites**:
  - Mutates `masks_by[by]` in memory (`as_read[by]` remains the uncarved snapshot).
- **Audit & Traps**:
  - **Uncarved `as_read` Escape Into Stage 2.8 (`3820654d`)**: `as_read[by]` is NOT carved by `keep_out`. Before `3820654d`, `hole_of_the_load` read `as_read`, re-admitting keep-out pixels into the hole union.

### Stage 2.3 — Off-Head Disconnected Blob Filter (`bench/capture_masked_run.py`)

- **Symbols**: `bench/capture_masked_run.py::off_head`, `bench/capture_masked_run.py::_files`
- **Reads**:
  - `masks_by[by]`, `<preview>_<subj>_head_track.mp4`, and `<preview>_<subj>_pose.npz`.
- **Transforms & Constants**:
  - Removes connected components of `masks_by[by]` that neither touch the SAM 3.1 `head_track` tube nor contain valid face/head keypoints (`off_head`, `5662-5716`).
- **Writes & Overwrites**:
  - Updates `masks_by[by]` in memory and records `off_head_dropped` counts in `part_grades.json`.
- **Audit & Traps**:
  - When `hold_to_track` smeared a stale head mask onto an occluder (`4682-4685`), `off_head` could not tell the held mask was stale until `head_track` empty-run gating was added.

### Stage 2.4 — SAM 3.1 Head Tube, Dip Bridging & Temporal Smoothing (`bench/head_tube_parts.py`)

- **Symbols**: `bench/head_tube_parts.py::head_tube_for_capture`, `bench/head_tube_parts.py::sam31_head_tube`, `bench/head_tube_parts.py::_head_seed_from_person_mask`, `bench/head_tube_parts.py::bridge_short_tube_dips`, `bench/head_tube_parts.py::smooth_seg_in_tube`, `bench/head_tube_parts.py::face_piece_kept_whole`
- **Reads**:
  - `<preview>_<subj>_head_track.mp4`, `<preview>_<subj>_parts.npz`, `<preview>_<subj>_pose.npz`, and `front_other_mask`.
- **Transforms & Constants**:
  - Builds the whole-head tube `ht_res["tube"]` (`sam31_head_tube`), bridges short tube area dips (`<= 6` frames,
    `bridge_short_tube_dips`), temporally smooths Sapiens-2B logits inside the tube (`smooth_seg_in_tube`), expands
    attached hair (`_expand_track_with_attached_parts`), fills unclassified internal holes
    (`_fill_unclassified_in_head_tube`), and protects facial hair (`face_piece_kept_whole`, `FACIAL_HAIR_CANNOT_SEE`).
- **Writes & Overwrites**:
  - Returns `ht_res` dict (`tube`, `head_hair`, `classes_smoothed`, `dip_bridges`, `facial_hair_kept`) to `bench/capture_masked_run.py::_files`.
- **Audit & Traps**:
  - **Flat `0.333` Horizontal Slice Seed Trap (`4791284f`)**: When re-seeding after an occlusion gap,
    `_head_seed_from_person_mask` (`58-92`) took the top `0.333` (`_HEAD_BOX_TOP_FRAC`) of the full-body person mask
    (`y=318..470`), slicing across a bending/turning subject's shoulder and back in `lead_fp_2050d` (`f=33..45`).
    Fixed in `4791284f` by intersecting with `Face_Neck`/`Hair` classes and 308-pose head points.

### Stage 2.5 — Front-Occluder Depth Clip (`bench/capture_masked_run.py`)

- **Symbols**: `bench/capture_masked_run.py::_files`
- **Reads**:
  - `masks_by[by]`, `ht_res["tube"]`, and `front_other_mask` from Stage 2.1.
- **Transforms & Constants**:
  - Clips `masks_by[by]` and `ht_res["tube"]` against `front_other_mask` (`masks_by[by] &= ~front_other_mask`) so an occluding person closer to the camera is never overwritten by the occluded subject's head tube.
- **Writes & Overwrites**:
  - Mutates `masks_by[by]` and `ht_res["tube"]` in memory (`2538-2590`).
- **Audit & Traps**:
  - Depends on Stage 1.2 (`strangers.py`) and Stage 2.1 (`_depth_medians_and_owner`) having detected the front person.

### Stage 2.6 — Largest Piece Guard, Floor & Early `masks__<by>.npz` Write (`bench/capture_masked_run.py`)

- **Symbols**: `bench/capture_masked_run.py::largest_piece`, `bench/capture_masked_run.py::refuse_pieces_on_a_head`, `bench/capture_masked_run.py::starting_floor`, `bench/capture_masked_run.py::floor_on_the_canvas`
- **Reads**:
  - `masks_by[by]`, `ht_res["tube"]`, and job recipe (`pieces`, `floor_head_scale=1.1`, `min_component_pixels=150`).
- **Transforms & Constants**:
  - Enforces `refuse_pieces_on_a_head` (`PIECES_CANNOT_SEE`), drops small islands (`< 150 px`), and unions the head-scale floor (`floor_on_the_canvas`, `1.1x` head box floor when enabled).
- **Writes & Overwrites**:
  - **Writes `masks__<by>.npz` at line `2626` BEFORE `grade_holds` (Stage 2.7) and BEFORE `her_hair_in_hole` (Stage 2.8)**.
- **Audit & Traps**:
  - **Stale `masks__<by>.npz` Trap**: Because `masks__<by>.npz` is written at `2626` before Stage 2.7 (`grade_holds`) and Stage 2.8 (`her_hair_in_hole`), any tool reading `masks__<by>.npz` sees the pre-grade, pre-hair-fill mask rather than the settled mask written to `parts_held__<by>.mkv` at `5557`.

### Stage 2.7 — Hold Grading & First `parts_held__<by>.mkv` Write (`bench/capture_masked_run.py`)

- **Symbols**: `bench/capture_masked_run.py::grade_holds`, `bench/capture_masked_run.py::part_sizes`, `bench/capture_masked_run.py::held_parts`, `bench/capture_masked_run.py::_files`
- **Reads**:
  - `masks_by[by]`, `ht_res["tube"]`, `<preview>_<subj>_pose.npz`, and `SIZES_CANNOT_SEE`.
- **Transforms & Constants**:
  - `grade_holds` (`915-1035`) grades each frame's mask (`kept`, `held`, `dropped_small`, `dropped_off_track`) against minimum pixel floors (`1,024 px` default face/head floor) and sets `hair_pass["head"] = ht_res["tube"]` (`2830`, `b461584e`).
- **Writes & Overwrites**:
  - Writes `part_grades.json` (`2816`), **first write of `parts_held__<by>.mkv` (`2843`)**, and `owners.npz` (`2875`).
- **Audit & Traps**:
  - **Small Verified Head Erasure (`4702-4714`, `9c5a6841`)**: On distant shots where a real, pose-verified head is `< 1,024 px` (`488-796 px` on `4702-4714`), `grade_holds` emptied the frame (`0 px`) until `9c5a6841` exempted frames with verified SAM 3.1 `head_track` + 308-pose face landmarks.
  - **Stale `part_grades.json` (`2816`)**: `part_grades.json` is written at `2816` and is NEVER updated after Stage 2.8 (`her_hair_in_hole`, `5557`) or Stage 3.2 (`on-plan`, `2101`) mutates `parts_held__<by>.mkv`.

### Stage 2.8 — Her Hair in the Hole (`bench/capture_masked_run.py`)

- **Symbols**: `bench/capture_masked_run.py::her_hair_in_hole`, `bench/capture_masked_run.py::hole_of_the_load`, `bench/capture_masked_run.py::whole_frame_hair`, `bench/capture_masked_run.py::hair_in_hole_files`
- **Reads**:
  - `hair_pass["head"]` (`ht_res["tube"]`), `<preview>_<subj>_parts.npz`, `<preview>_strangers_classes.npz`, `keep_out` masks, and `HAIR_IN_HOLE_CANNOT_SEE`.
- **Transforms & Constants**:
  - Computes 3 additive hair-recovery parts inside the regeneration hole (`hole_of_the_load`):
    - **Part 1 (`in_hole_part1`)**: subject's own Sapiens-2B `Hair` (`class == 3`) inside `hole(f)`.
    - **Part 2 (`in_hole_part2`)**: whole-frame hair (`<preview>_strangers_classes.npz` `class == 3`) connected to the subject's head tube and not owned by another person (`~taken[k]`, `b461584e`).
    - **Part 3 (`in_hole_part3`)**: dark/hair-colored pixels inside the head-tube neighborhood bridging `Part 1`/`Part 2`.
- **Writes & Overwrites**:
  - **Overwrites `parts_held__<by>.mkv` in place (`5557`)** and **overwrites `track__<by>.mkv` in place (`5559`, `00a3f1d8`)**; writes `hair_in_hole.json` and `hair_in_hole__<by>.npz`.
- **Audit & Traps**:
  - **`hole_of_the_load` Pre-Carve/Pre-Feather Trap (`3820654d`)**: `hole_of_the_load` (`5165-5275`) reconstructed `hole(f)` from uncarved `as_read` without subtracting `keep_out` and without replicating `video_mask.py`'s `grow_px`/`feather_px`/`16x16` token rounding. Fixed in `3820654d` to subtract `keep_out` and match node mask footprint.
  - **Part 3 `~track[f]` & `strangers_classes.npz` Traps (`9c5a6841`)**: Part 3 excluded `track[f]` (the full-body SAM 3.1 silhouette), blocking hair recovery on frames where hair lay inside the body silhouette (`4912`), while reading whole-frame `strangers_classes.npz` outside the head box pulled in background objects (`4682-4685`).

### Stage 2.9 — Planned Region Prediction (`bench/capture_masked_run.py`)

- **Symbols**: `bench/capture_masked_run.py::planned_region`, `bench/capture_masked_run.py::predict_regions`, `bench/capture_masked_run.py::read_planned_regions`
- **Reads**:
  - Settled `parts_held__<by>.mkv`, `keep_out` masks, and `<job>_api.json` node `104` parameters.
- **Transforms & Constants**:
  - Simulates `video_mask.py::window` / `token_mask` per planned window to predict the exact `(N, H, W)` boolean region and `16x16` token grid H3 will regenerate.
- **Writes & Overwrites**:
  - Writes `planned_region__<node_id>.npz` (`2899`).
- **Audit & Traps**:
  - Replaced in Stage 3.2 (`on-plan`) once node `74` writes `plan.json` and `window_NNN_region.npz` on the server.

---

## Stage 3 — Plan Execution (`plan_only`), On-Plan Re-Carving & Pre-Render Gate

### Stage 3.1 — Server-Side `plan_only` Execution (`video_mask.py`, `audio_freeze_song.py`, `bench/load_chain.py`)

- **Symbols**: `bench/load_chain.py::step_plan`, `video_mask.py::MiniMaxH3MaskedSource.execute`, `audio_freeze_song.py::MiniMaxH3AudioFreezeSong.execute`, `audio_freeze_song.py::write_plan`
- **Reads**:
  - `<job>_api.json` submitted to ComfyUI with `plan_only=True` on node `74`, reading `parts_held__<by>.mkv` and `track__<by>.mkv`.
- **Transforms & Constants**:
  - Node `104` (`MiniMaxH3MaskedSource.execute`) computes per-window latent masks (`token_mask`), context padding (`pad_context`, `hold_context`), and zoom boxes (`zoom_plan`), saving `window_NNN_region.npz` (`save_window_region`).
  - Node `74` (`write_plan`) writes `<prefix>_plan.json` and exits before calling the DiT sampler (`eb94f9df`).
- **Writes & Overwrites**:
  - `$H3_OUTPUT_DIR/<prefix>_plan.json` and `$H3_OUTPUT_DIR/<prefix>_window_NNN_region.npz`.
- **Audit & Traps**:
  - If `parts_held__<by>.mkv` is later mutated by Stage 3.2 (`on-plan`), the server's `window_NNN_region.npz` from Stage 3.1 reflects the pre-`on-plan` mask unless re-verified by `bench/load_chain.py::masks_as_planned`.

### Stage 3.2 — On-Plan Keep-Out Re-Carving (`bench/capture_masked_run.py`, `bench/load_chain.py`)

- **Symbols**: `bench/load_chain.py::step_on_plan`, `bench/capture_masked_run.py::step_keepout_carves`, `bench/capture_masked_run.py::node_taken_inside`, `bench/capture_masked_run.py::grid_carve`, `bench/capture_masked_run.py::_bridge_severed`
- **Reads**:
  - `$H3_OUTPUT_DIR/<prefix>_plan.json`, `$H3_OUTPUT_DIR/<prefix>_window_NNN_region.npz`, and `parts_held__<by>.mkv`.
- **Transforms & Constants**:
  - Checks whether `16x16` token pooling in `window_NNN_region.npz` expanded the regeneration region into `keep_out` / `front_other_mask` (`node_taken_inside`).
  - Carves back offending `16x16` grid cells (`grid_carve`), repairs thin bridges (`_bridge_severed`), and re-runs `her_hair_in_hole` (`2118-2167`).
- **Writes & Overwrites**:
  - **Overwrites `parts_held__<by>.mkv` in place (`2101`, third write; `2165`, fourth write)** and **overwrites `track__<by>.mkv` in place (`2167`)**; writes `keepout_carves.json`.
- **Audit & Traps**:
  - **Silent 4x In-Place Overwrite of `parts_held__<by>.mkv`**: Across Stages 2.7 (`2843`), 2.8 (`5557`), and 3.2 (`2101`, `2165`), `parts_held__<by>.mkv` is overwritten up to 4 times in place while `masks__<by>.npz` (`2626`) and `part_grades.json` (`2816`) remain frozen at Stage 2.6/2.7.

### Stage 3.3 — Pre-Render Verification, Flags & Frame Table (`bench/capture_masked_run.py`, `bench/inspect_mask_steps.py`, `bench/flags_before_render.py`, `bench/what_h3_got.py`, `bench/frame_table.py`)

- **Symbols**: `bench/capture_masked_run.py::preflight`, `bench/inspect_mask_steps.py::inspect_capture`, `bench/flags_before_render.py::check_flags_read`, `bench/what_h3_got.py::audit_capture`, `bench/frame_table.py::build_frame_table`
- **Reads**:
  - Capture directory artifacts (`parts_held__<by>.mkv`, `part_grades.json`, `hair_in_hole.json`, `owners.npz`, `<job>_api.json`, `<prefix>_plan.json`).
- **Transforms & Constants**:
  - `preflight` (`4050-4960`) evaluates 24 flag functions (`flag_shots` through `flag_text`, bounded by `OTHER_CANNOT_SEE`).
  - `bench/inspect_mask_steps.py::inspect_capture` (`b461584e`) audits Stage 2 step-by-step pixel deltas (`as_read` -> `carved` -> `head_tube` -> `parts_held` -> `window_region`).
  - `bench/what_h3_got.py::audit_capture` (`ac803f92`) checks what H3 actually received across windows, references, and prompts.
  - `bench/frame_table.py::build_frame_table` writes `frame_table.json` and `frame_table.md`.
- **Writes & Overwrites**:
  - `preflight.json`, `mask_steps.json`, `flags_read.json`, `frame_table.json`, and `frame_table.md`.
- **Audit & Traps**:
  - **`frame_table.py` Stale `part_grades.json` Mismatch**: `frame_table.py` reads `part_grades.json` (written at Stage 2.7) alongside `parts_held__<by>.mkv` (mutated by Stage 2.8 and Stage 3.2), reporting `"held"` or `"dropped"` grades that no longer match the pixels in `parts_held__<by>.mkv`.

---

## Stage 4 — DiT Sampling, Token Masking & Window Lay-Back (`video_mask.py`, `audio_freeze_song.py`)

### Stage 4.1 — Window Slicing, Context Padding & Zoom Crop (`video_mask.py`)

- **Symbols**: `video_mask.py::MiniMaxH3MaskedSource.execute`, `video_mask.py::window`, `video_mask.py::pad_context`, `video_mask.py::hold_context`, `video_mask.py::zoom_plan`, `video_mask.py::zoomed_source`
- **Reads**: Source frames, `parts_held__<by>.mkv`, `keep_out` masks, and window bounds from `loop_plan.py::plan_windows`.
- **Transforms & Constants**: Slices each window (`4k+1` via `h3_rules.py::snap_length`), prepends/appends held context padding (`pad_context`, `hold_context`), and crops to `zoom_box` (`zoomed_source`) when `zoom` is active.
- **Writes & Overwrites**: Passes windowed source tensor, pixel mask, and `token_mask` into the ComfyUI conditioning dict (see [code_map_contracts.md](code_map_contracts.md) Section `2b`).

### Stage 4.2 — Latent `16x16` Token Mask & Step Schedule (`video_mask.py`)

- **Symbols**: `video_mask.py::token_mask`, `video_mask.py::grow`, `video_mask.py::margins`, `video_mask.py::drop_specks`, `video_mask.py::close_lone_holes`
- **Reads**: Window pixel mask `(T, H, W)` and recipe parameters (`grow_px=0`, `feather_px=8`, `keep_out`).
- **Transforms & Constants**: Pools pixel masks onto the `(t_lat, H//16, W//16)` spatiotemporal token grid (`1` latent frame = `4` video frames after frame 0), drops lone specks (`drop_specks`), and closes single-cell holes (`close_lone_holes`).
- **Writes & Overwrites**: Saves `window_NNN_region.npz` (`save_window_region`) with `token_mask`, `pixel_alpha`, `zoom_box`, and `pad_spec`.
- **Audit & Traps**: Any masked pixel inside a `4x16x16` cell marks the entire latent cell for DiT generation until pixel-space `lay_window` composites with `pixel_alpha`.

### Stage 4.3 — Pixel Alpha Feathering, Cut Gate & Window Lay (`video_mask.py`)

- **Symbols**: `video_mask.py::pixel_alpha`, `video_mask.py::changed_alpha`, `video_mask.py::cut_gate`, `video_mask.py::unlaid_frames`, `video_mask.py::lay_window`, `video_mask.py::lay_back`
- **Reads**: DiT-decoded window frames, original source frames, `pixel_alpha` (`feather_px=8`), and `cut_gate`.
- **Transforms & Constants**: Zeros alpha where `unlaid_frames` or `cut_gate` refuses laying, composites generated pixels over source (`lay_window`), and inverse-warps zoomed windows onto the full canvas (`lay_back`).
- **Writes & Overwrites**: Produces composited window tensor and `unlaid` frame indices for Node `74`.

### Stage 4.4 — Cross-Window Overlap Blending & Audio Silencing (`audio_freeze_song.py`, `loop_plan.py`)

- **Symbols**: `audio_freeze_song.py::MiniMaxH3AudioFreezeSong.execute`, `loop_plan.py::frames_kept`, `loop_plan.py::frames_written`, `audio_freeze_song.py::heard_under_held`, `audio_freeze_song.py::silenced`
- **Reads**: Composited windows from Stage 4.3, `loop_plan.py` overlap weights, and source audio waveform.
- **Transforms & Constants**: Strips held padding (`frames_kept`, `frames_written`), blends overlapping window tails/leads, and silences audio under held padding (`heard_under_held`, `silenced`).
- **Writes & Overwrites**: Assembles the full-load output frame sequence and restores original audio on the final MP4.

### Stage 4.5 — Server-Side Output Artifact Writes (`audio_freeze_song.py`)

- **Symbols**: `audio_freeze_song.py::_write_pieces_mp4`, `audio_freeze_song.py::_write_frames_mp4`, `audio_freeze_song.py::_write_review_mp4`, `audio_freeze_song.py::zoomed_record`
- **Reads**: Assembled load frames, raw uncomposited DiT pieces, and per-window metadata.
- **Writes & Overwrites**: Writes `$H3_OUTPUT_DIR/<prefix>.mp4`, `<prefix>_pieces.mp4`, `<prefix>_frames.mp4`, `<prefix>_review.mp4`, `<prefix>.json`, and `<prefix>_zoomed.json` (collected into capture folder by `bench/masked_job.py::collect`; see Section `2a.2` of [code_map_contracts.md](code_map_contracts.md)).

---

## Stage 5 — Delivery Table, Plane Compositing, Worn Layer & Final Assembly (`bench/assemble_delivery.py`)

### Stage 5.1 — Delivery Table & Matte Resolution (`bench/assemble_delivery.py`, `bench/delivery_table.py`)

- **Symbols**: `bench/assemble_delivery.py::read_table`, `bench/assemble_delivery.py::Matte`, `bench/assemble_delivery.py::laid_by_of`, `bench/assemble_delivery.py::_laid_by_of`, `bench/delivery_table.py::decide`
- **Reads**: `data/<video>/delivery_table.json` and each kept run's capture folder (`window_NNN_region.npz`, `parts_held__<by>.mkv`, `<prefix>_zoomed.json`).
- **Transforms & Constants**: Resolves which run lays each `(subject, frame)` span and constructs `Matte` objects (`laid_by_of`) from saved window regions and held part masks (never reading early Stage 2.6 `masks__<by>.npz`).

### Stage 5.2 — Multi-Subject Depth Ordering & Restore Weights (`bench/assemble_delivery.py`)

- **Symbols**: `bench/assemble_delivery.py::Captures`, `bench/assemble_delivery.py::owners`, `bench/assemble_delivery.py::weight_of`, `bench/assemble_delivery.py::restore_weight`
- **Reads**: Per-capture `owners.npz`, `<preview>_depth.npz`, `<preview>_strangers_classes.npz`, and `delivery_table.json` (`restore_words`).
- **Transforms & Constants**: Orders overlapping subject planes back-to-front per frame (`owners`), computes soft blend weights (`weight_of`, `feather_px=8`), and builds `restore_weight` masks where original source pixels are restored over generated planes.

### Stage 5.3 — Worn Layer (Glasses/Jewelry/Apparel) & Plane Compositing (`bench/assemble_delivery.py`)

- **Symbols**: `bench/assemble_delivery.py::placements_of`, `bench/assemble_delivery.py::worn_layer`, `bench/assemble_delivery.py::lay_plane`, `bench/assemble_delivery.py::laid_over`, `bench/assemble_delivery.py::carry_edge_rows`
- **Reads**: Source frames, kept render frames (`piece_frames`), 308-keypoint face shifts (`shifts_of`, `moved_by`), and Sapiens-2B `worn_classes`.
- **Transforms & Constants**: Tracks worn items (`placements_of`, `worn_layer`) across face shifts, composites each subject plane (`lay_plane`, `laid_over`), and clamps zoom-box edge rows (`carry_edge_rows`) to prevent border seams.

### Stage 5.4 — Final MP4 Encoding, Audit & Sidecar (`bench/assemble_delivery.py`, `bench/stack_delivery.py`, `bench/rebuild_delivery.py`)

- **Symbols**: `bench/assemble_delivery.py::build`, `bench/assemble_delivery.py::flags_of`, `bench/assemble_delivery.py::check`, `bench/assemble_delivery.py::write_sidecar`, `bench/stack_delivery.py::compare`, `bench/rebuild_delivery.py::rebuild`
- **Writes & Overwrites**: Muxes video and original audio into `$H3_OUTPUT_DIR/Video/<video>/<delivery>.mp4` and `<delivery>.json` sidecar (`write_sidecar`); `bench/rebuild_delivery.py::verify_reused_clip` and `verify_poc_on_disk` refuse delivery if `table_laid_sha` or upstream mtime (`bench/_lib/code_stamp.py::check_upstream_mtime`) has drifted.
