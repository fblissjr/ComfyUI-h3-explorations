# Masked V2V Code Map — Part 1: End-to-End Mask & Tensor State Mutation Ledger

<!-- narrative_stamp: doc_ast=#6db067c9 sections=29 (run `bench/masked_code_map.py --stamp-narrative` after reviewing prose) -->

This companion document to [2026-10-26_code_map_and_simplification_plan.md](../research/masking/2026-10-26_code_map_and_simplification_plan.md)
traces every mask, tensor, `.npz`, `.json`, and `.mkv` state mutation from Stage 1 (pre-capture tracking and
parts segmentation) through Stage 5 (final delivery assembly).

All line numbers live in [CODE_MAP_INDEX.md](CODE_MAP_INDEX.md) (or via `$H3_PYTHON bench/masked_code_map.py <symbol>`)
so narrative prose never drifts when lines shift. For contracts, window padding, and duplicate logic, see
[code_map_contracts.md](code_map_contracts.md).

---

## Stage 1 — Pre-Capture Tracking, Parts, Strangers, Depth, References & Job Creation

### Stage 1.1a — SAM 3.1 Subject & Head Tracking (`subject_track.py`)

- **Symbols** (`#714ffc12`): `subject_track.py::MiniMaxH3SubjectTrack.execute`, `subject_track.py::follow`, `subject_track.py::parse_head_mode`, `subject_track.py::_track_head_span`, `subject_track.py::_span_head`, `subject_track.py::name_drawn`, `subject_track.py::auto_match`, `bench/shot_to_load.py::preview_job`, `bench/cast_file.py::drawn`
- **Reads**:
  - Source clip `<shot>.mp4`, `<video>/cast/cast.jsonl` (`bench/cast_file.py::read`, `bench/cast_file.py::seeds`), and `<video>/shots.json`.
  - Confirmed shot gallery looks or verified seed boxes from `cast.jsonl` (Group 1 `/who/` page clicks `1-174` and Group 3 session-verified SAM detection boxes `197-209`). `bench/cast_file.py::drawn` (`6b0ea9a6`) now requires `--people <people.npz>` on `box`, verifies `IoU >= 0.500` against SAM 3.1 detections on that frame, and snaps the box to the best SAM detection (`drawn=None`).
- **Transforms & Constants**:
  - Runs SAM 3.1 propagation across the shot to produce boolean person masks `masks` (`(N, H, W)`) and `boxes` (`(N, 4)`).
  - Derives `head_masks`, `head_boxes`, `head_scores`, and `head_from` via `subject_track.py::parse_head_mode`, which defaults to `"per stretch"` (`HEAD_PER_STRETCH`, `head_per_stretch=True` in `follow` and `bench/shot_to_load.py::preview_job` since `6b0ea9a6`; whole-span mode requires explicit `head: span: WHY` / `--head-span WHY`).
- **Writes & Overwrites**:
  - `<preview>_<subj>_track.mp4`, `<preview>_<subj>_track.json` (`boxes`, `scores`, `empty_runs`), `<preview>_<subj>_head_track.mp4`, and `<preview>_<subj>_head_track.json`.
- **Audit & Traps**:
  - **Forearm Bridge vs Empty Head Track (`4682-4685`)**: SAM 3.1 person tracking can stay alive on an occluded subject's bare forearm (`y=297-359`) while `head_track` is `0 px` (`4682-4685`, or `4682-4695` on previews built before `6b0ea9a6` when `--head-per-stretch` was off). Downstream `sapiens2_parts.py` crops around the forearm box unless gated by `head_track`.

### Stage 1.1b — Sapiens-2B 28-Class Body-Part Segmentation (`sapiens2_parts.py`)

- **Symbols** (`#4f3942b8`): `sapiens2_parts.py::subject_parts`, `sapiens2_parts.py::MiniMaxH3SubjectParts.execute`
- **Reads**:
  - Source frames `(N, H, W, 3)` and person track `masks` / `boxes` from Stage 1.1a.
- **Transforms & Constants**:
  - Crops a padded bounding box around each tracked person box, resizes to `1024x768`, runs Sapiens-2B (`torch.bfloat16`), and argmaxes 28 classes (`0=background`, `1=Apparel`, `2=Face_Neck`, `3=Hair`, `23=Upper_Lip`, `24=Lower_Lip`, `25=Lower_Teeth`, `26=Upper_Teeth`, `27=Tongue`).
- **Writes & Overwrites**:
  - `<preview>_<subj>_parts.mp4`, `<preview>_<subj>_parts.npz` (`classes` uint8 `(N, H, W)`), and `<preview>_<subj>_parts.json`.
- **Audit & Traps**:
  - **Occluded Forearm Crop Hallucination (`4682-4685`)**: When a subject's head is behind another person (`head_track == 0 px`) and her person track holds only her forearm (`y=297-359`), cropping `1024x768` around the forearm causes Sapiens-2B to hallucinate `Face_Neck` (`608-2,123 px`) on skin/cloth (suppressed in Stage 2.4 by `head_tube` with `0` face points).

### Stage 1.2 — Whole-Frame Strangers & Unclaimed People Pass (`bench/strangers.py`)

- **Symbols** (`#24b3ce68`): `bench/strangers.py::people_pass`, `bench/strangers.py::sam_people`, `bench/strangers.py::others_of`, `bench/strangers.py::held_others`, `bench/strangers.py::handed_back`, `bench/strangers.py::run`
- **Reads**:
  - Source clip `<shot>.mp4`, all tracked subjects' `_track.mp4` and `_parts.npz`, and `<preview>_people.npz` (written by `people_pass` / `sam_people` and unpacked by `people_on` / `others_of`).
- **Transforms & Constants**:
  - `bench/strangers.py::others_of` matches `<preview>_people.npz` detections against tracked subjects and marks unclaimed SAM detections with `o[3] is None`.
  - `bench/strangers.py::run` runs whole-frame Sapiens-2B (`predict(pic)`) to get `classes[i] = c`, and since `426638a6`
    unions unclaimed SAM 3.1 detections where Sapiens-2B finds nobody (`det |= m & (c == 0)` for `who is None` in `others[i]`)
    into both `people[i] = (c > 0) | det` and `core[i] = np.isin(c, core_ids) | det` over the 18 `CORE_CLASSES`
    (`Face_Neck`, `Hair`, `Torso`, arms, clothing, legs, lips/teeth/tongue).
  - Subtracts grown owned subject tracks (`strangers.grown(kept, px)`), holds stranger masks across short dropouts, and runs
    `bench/strangers.py::handed_back`, which checks `held = piece & core[i]` and hands back (`out[i] &= ~piece`) any stranger
    component touching a subject where `held.any()` is false (`NO_CORE = "no body core of its own"`) or where no unowned SAM
    person overlaps after `hold_frames` (`NO_OTHER = "handed_back_on_no_other_person"`).
- **Writes & Overwrites**:
  - `<preview>_people.npz` (from `people_pass`), `<preview>_strangers.mkv`, `<preview>_strangers_classes.npz`, and `<preview>_strangers.json` (with 13-item `cannot_see` array).
- **Audit & Traps**:
  - **Resolved `c == 0` / `NO_CORE` Trap (`426638a6`) vs Stale Previews**: Before `426638a6`, unowned SAM 3.1 detections
    (`who is None`) were not unioned into `people[i]` or `core[i]` where `c == 0` (`4686-4693`, `1458`). Any
    `<preview>_strangers.mkv` written before `426638a6` lacks those SAM-only stranger pixels until `strangers` is re-run
    (`bench/strangers.py::held_to_its_record` checks `taken_out == OWN_FORM`).

### Stage 1.3 — Monocular Metric Depth (`bench/measure_moge2_depth.py`, `bench/shot_to_load.py::make_depth`)

- **Symbols** (`#58b38c82`): `bench/measure_moge2_depth.py::run_measurement`, `bench/shot_to_load.py::make_depth`
- **Reads**:
  - Source clip `<shot>.mp4`.
- **Transforms & Constants**:
  - Evaluates MoGe-2 (`vendor/meta_moge2/hyden_moge2.py`) per frame on raw `[0, 1]` float32 RGB and downsamples by `stride=4` to float16 metric depth `(N, H//4, W//4)`. `bench/shot_to_load.py::make_depth` (`1e91de26`) unions `mine_subj["hair_hers"]` frames into `ov_frames` when selecting frames for load depth measurement.
- **Writes & Overwrites**:
  - `<preview>_depth.npz` (`depth` float16, `stride=4` plus `bench/_lib/code_stamp.py::npz_writer_fields` provenance) and `<preview>_depth.json`.
- **Audit & Traps**:
  - Downsampled `stride=4` depth is upsampled in Stage 2.1 (`bench/capture_masked_run.py::read_depth_npz`); thin hair strands against background take background depth unless rescued by Stage 2.4 (`sam31_head_tube`).

### Stage 1.4 — 308-Keypoint Whole-Body Pose (`bench/pose_along_shot.py`)

- **Symbols** (`#715ae0e0`): `bench/pose_along_shot.py::run`, `bench/pose_along_shot.py::read`
- **Reads**:
  - Source clip `<shot>.mp4` and `<preview>_<subj>_track.mp4` (masks outside grown track greyed out so another brighter face off-track is not read).
- **Transforms & Constants**:
  - Runs Sapiens-1B-308-pose inside each subject's tracked box; extracts 308 keypoints, scores, and face/ear/lid measurements.
- **Writes & Overwrites**:
  - `<preview>_<subj>_pose.npz` (`keypoints` `(N, 308, 2)`, `scores` `(N, 308)`) and `<preview>_<subj>_pose.json`.
- **Audit & Traps**:
  - When `person_track` holds only a forearm or drifts (`4676-4685`, `4706-4716` on `4647`), the pose crop follows the drifted/forearm box and misses the emerging head, leaving `<Video 1>` without a body model on those frames.

### Stage 1.5 — Head Reference Video & 308-Point Face Drawing (`bench/head_reference.py`, `bench/crop_reference_to_head.py`, `bench/draw_face_points.py`)

- **Symbols** (`#0ff4cca6`): `bench/head_reference.py::run`, `bench/crop_reference_to_head.py::run`, `bench/draw_face_points.py::run`, `bench/draw_face_points.py::draw`, `bench/shot_to_load.py::step_reference`, `bench/shot_to_load.py::drawn_form`
- **Reads**:
  - Source clip, `<preview>_<subj>_pose.npz`, `<preview>_<subj>_parts.npz`, and `<preview>_<subj>_head_track.mp4`.
- **Transforms & Constants**:
  - `bench/crop_reference_to_head.py::run` crops the stabilized head/cranium box (`one_box`, `grey_head`, `cranium_box`).
  - `bench/draw_face_points.py::run` renders 308-point facial landmarks, irises (`dark_irises`, `open_lids`, `sculpted_eye`), and mouth geometry (`anchored_mouth`, `anchored_closed_mouth`, `bend_mouth`).
  - `bench/head_reference.py::run` orchestrates the crop and drawing pass to build `<Video 1>`.
- **Writes & Overwrites**:
  - `<ref_folder>/<subj>_head_reference_<first>_<last>.mp4`, `.json`, `_sheet.jpg`, and `_points_filled.npz`.
- **Audit & Traps**:
  - **`4647h` `<Video 1>` Bare Frames (`76533a22` / `9c5a6841`)**: On `second_4647_4725_j`, `<Video 1>` had `43/79` bare frames (`4673-4686` and `4697-4725` from a carried `--no-draw-face-on` word, on top of no body model on `4676-4685` and `4706-4716` from forearm/drifted track crops). `second_hfp_4647i` must remake `reference` after Build 3 with no drawing word carried.

### Stage 1.6a — Part-on-Track Clipping (`bench/shot_to_load.py`)

- **Symbols** (`#89067dde`): `bench/shot_to_load.py::part_on_track`, `bench/shot_to_load.py::hold_to_track`, `bench/shot_to_load.py::make_part_on_track`, `bench/shot_to_load.py::part_of`, `bench/shot_to_load.py::make_part`
- **Reads**:
  - `<preview>_<subj>_parts.mp4`, `<preview>_<subj>_track.mp4`, and `<preview>_strangers.json`.
- **Transforms & Constants**:
  - When `strangers` is active on a load, `bench/shot_to_load.py::hold_to_track` computes `held = a & strangers.grown(kept, px)` (`px = h3_config.MASKED_STRANGERS["grow_px"] = 8`), where `kept, gone, empty = strangers.held_track(track)` holds the **person track** across short empty person-track runs and **intersects** (`&`) the per-frame part mask `a` with that grown person track.
- **Writes & Overwrites**:
  - `<preview>_<subj>_part_<kind>_on_track_8px.mkv` and `.json`.
- **Audit & Traps**:
  - **Clipping Valid Hair Outside `person_track` (`4912`) & Drifted Track Clipping (`4699-4702`)**: Because `hold_to_track`
    intersects `a` with `grown(kept, 8)`, any hair Sapiens found outside SAM 3.1's `person_track` is clipped away (`4912`),
    and if `person_track` drifts onto another person (`4699-4702`), `a` is clipped to the wrong person's box. Settled for
    retirement in Morning Build 1 (`Part 5e`).

### Stage 1.6b — Load Job JSON & Graph Compilation (`bench/shot_to_load.py`, `bench/masked_job.py`, `workflows/masked_job_graph.py`)

- **Symbols** (`#baea9276`): `bench/shot_to_load.py::step_job`, `bench/shot_to_load.py::load_job`, `bench/shot_to_load.py::carry_words_of`, `bench/shot_to_load.py::head_marks_of`, `bench/masked_job.py::build`, `workflows/masked_job_graph.py::load`
- **Reads**:
  - `<video>/shots.json`, `<video>/plan.json`, `<video>/cast/cast.jsonl`, preview `.json` / `.npz` artifacts, and CLI flags.
- **Transforms & Constants**:
  - Enforces settled `subjects.<subject>.recipe` and `shot_overrides` in `plan.json` (`cbd39fbe`: hard-refusing differing CLI `--margin` or `--zoom-up-to` values).
  - `bench/shot_to_load.py::head_marks_of` parses `plan.json["head_marks"]` for `subject`, clips spans to the load's `[a, b]` frame range, and emits `head_hidden`, `hair_hers`, and `hair_not_hers` onto the subject's job entry.
- **Writes & Overwrites**:
  - `data/<video>/jobs/<job>.json` and `<job_folder>/<job>_api.json`.

---

## Stage 2 — Capture Pre-Pass & Mask Settlement (`bench/capture_masked_run.py`, `bench/head_tube_parts.py`)

### Stage 2.1 — Depth Medians & Pixel Ownership (`bench/capture_masked_run.py`)

- **Symbols** (`#ba1e4ff5`): `bench/capture_masked_run.py::_files`, `bench/capture_masked_run.py::read_depth_npz`, `bench/capture_masked_run.py::_depth_medians_and_owner`, `bench/capture_masked_run.py::owner_map`
- **Reads**:
  - `<preview>_depth.npz`, subject `track` / `parts` masks, and `other` keep-out masks (`strangers.mkv`, other subjects' tracks).
- **Transforms & Constants**:
  - Upsamples MoGe-2 depth to `(N, H, W)` and computes per-frame median depth per subject and keep-out (`_depth_medians_and_owner`).
  - Assigns per-pixel ownership (`owner_map`) and `front_other_mask` where another person is closer to the camera than the replaced subject by `> DEPTH_ORDER_MIN_M` (`0.05` m).
- **Writes & Overwrites**:
  - In-memory `owners`, `front_other_mask`, and `depth_settled`; persisted at the end of `_files` into `owners.npz`.
- **Audit & Traps**:
  - If an unowned SAM person is missing from `<preview>_strangers.mkv` (Stage 1.2 `c == 0` / `NO_CORE` trap on `4686-4693` and `1458`), `_depth_medians_and_owner` never sees that bystander in `other` keep-outs.

### Stage 2.2 — Raw Input Reading & Grid Keep-Out Carving (`bench/capture_masked_run.py`)

- **Symbols** (`#92211333`): `bench/capture_masked_run.py::_files`, `bench/capture_masked_run.py::_keep_in_front`, `bench/capture_masked_run.py::carve_keep_out`, `bench/capture_masked_run.py::_bridge_severed`, `bench/capture_masked_run.py::grid_carve`, `bench/capture_masked_run.py::node_taken_inside`
- **Reads**:
  - Subject's `track` and `parts` mask videos, `other` keep-out mask videos, and `planned` regions (when running `on-plan`).
- **Transforms & Constants**:
  - Keeps uncarved `as_read` snapshot of keep-out tracks; when `planned` regions exist (`step_on_plan`), carves keep-out
    `track` masks on the `16x16` grid (`grid_carve`), carves a 1-cell 4-connected bridge (`_bridge_severed`, `45f3eaf1`) when
    keep-outs pinch a `hers` unit into a diagonal island that `video_mask.py::drop_islands` would drop, and (`39e1f29a`) passes
    `keeps`/`depth`/`valid` to `node_taken_inside` and `cross_rows` so foreground keep-outs preserved by `_keep_in_front` are
    excluded from `keep_out_inside_own_track` faults.
- **Writes & Overwrites**:
  - Updates keep-out `track` arrays in memory and writes `carved__<by>.mkv` when `carved > 0`.

### Stage 2.3 — Off-Head Disconnected Blob Filter (`bench/capture_masked_run.py`)

- **Symbols** (`#d1aa1dce`): `bench/capture_masked_run.py::off_head`, `bench/capture_masked_run.py::_files`
- **Reads**:
  - Subject `parts` mask, `head_track`, and reference face points (`points=`, `off_head_margin=16`).
- **Transforms & Constants**:
  - Drops connected components of `parts` that lie far outside the head/reference points (`off_head`).
- **Writes & Overwrites**:
  - Mutates `parts` in memory before Stage 2.4/2.6.

### Stage 2.4 — SAM 3.1 Head Tube, Dip Bridging & Temporal Smoothing (`bench/head_tube_parts.py`)

- **Symbols** (`#699cc22e`): `bench/head_tube_parts.py::head_tube_for_capture`, `bench/head_tube_parts.py::sam31_head_tube`, `bench/head_tube_parts.py::_head_seed_from_person_mask`, `bench/head_tube_parts.py::bridge_short_tube_dips`, `bench/head_tube_parts.py::smooth_seg_in_tube`, `bench/head_tube_parts.py::face_piece_kept_whole`
- **Reads**:
  - `<preview>_<subj>_head_track.mp4`, `<preview>_<subj>_parts.npz`, `<preview>_<subj>_pose.npz`, `person_track`, and `head_hidden` marks.
- **Transforms & Constants**:
  - Builds the whole-head tube `ht_res["tube"]` (`sam31_head_tube`), bridges short tube dips (`<= 6` frames,
    `bridge_short_tube_dips`, skipping `head_hidden` frames), temporally smooths Sapiens-2B probabilities inside the tube
    (`smooth_seg_in_tube`), expands attached hair (`_expand_track_with_attached_parts`), fills internal holes
    (`_fill_unclassified_in_head_tube`), and protects facial hair (`face_piece_kept_whole`, `FACIAL_HAIR_CANNOT_SEE`).
  - Since `1e91de26`, `_head_seed_from_person_mask` intersects `own_head_2d & person_mask_2d` (the subject's own Sapiens
    head+hair classes inside `person_track`) when non-empty instead of taking a flat horizontal `head_rel_y1` slice, and on
    frames where `head_track` was empty and `sam31_head_tube` regained/bridged from `person_track`, `head_tube_for_capture`
    replaces `tube[t]` with `parts_bool[t]` and restricts hair relabeling to `ht_in_track | (tube_full & parts_bool)` so
    `Upper_Clothing`/`Torso` outside `head_track` are never relabeled as `Hair`.
- **Writes & Overwrites**:
  - Returns `ht_res` dict (`tube`, `mask`, `summary`) to `bench/capture_masked_run.py::_files` and writes `subjects/<label>/head_tube.json`.
- **Audit & Traps**:
  - **Fallback Flat Slice When `own_head_2d` Is Empty**: `_head_seed_from_person_mask` (`1e91de26`) uses `own_head_2d & person_mask_2d` whenever crop-Sapiens found head/hair pixels inside `person_track`, falling back to `head_rel_y1` only if that intersection is empty.

### Stage 2.5 — Front-Occluder Depth Clip (`bench/capture_masked_run.py`)

- **Symbols** (`#c4e854f5`): `bench/capture_masked_run.py::_files`
- **Reads**:
  - `parts`, `held`, and `front_other_mask` from Stage 2.1.
- **Transforms & Constants**:
  - Subtracts `front_other_mask` (`parts = parts & ~front_other_mask` and `held = held & ~front_other_mask`) so a closer person is never overwritten by the background subject's mask.

### Stage 2.6 — Largest Piece Guard, Floor & Early `masks__<by>.npz` Write (`bench/capture_masked_run.py`)

- **Symbols** (`#916848dd`): `bench/capture_masked_run.py::largest_piece`, `bench/capture_masked_run.py::refuse_pieces_on_a_head`, `bench/capture_masked_run.py::starting_floor`, `bench/capture_masked_run.py::floor_on_the_canvas`
- **Reads**:
  - `track`, `parts`, and subject options (`pieces`, `least`).
- **Transforms & Constants**:
  - Enforces `refuse_pieces_on_a_head` (`PIECES_CANNOT_SEE`), applies `largest_piece` if set, and resolves `starting_floor` (`floor_on_the_canvas`).
- **Writes & Overwrites**:
  - **Writes `subjects/<label>/masks__<by>.npz` BEFORE `held_parts` (Stage 2.7), `grade_holds` (Stage 2.7), `--drop`, and `her_hair_in_hole` (Stage 2.8)**.
- **Audit & Traps**:
  - **Pre-Grade / Pre-Fill `masks__<by>.npz` Snapshot Trap**: Because `masks__<by>.npz` is written before `held_parts`, `grade_holds`, `--drop`, and `her_hair_in_hole`, its `parts` bitpack never reflects Stage 2.7 grading/holding or Stage 2.8 hair recovery.

### Stage 2.7 — Hold Grading & First `parts_held__<by>.mkv` Write (`bench/capture_masked_run.py`)

- **Symbols** (`#d015e89f`): `bench/capture_masked_run.py::held_parts`, `bench/capture_masked_run.py::grade_holds`, `bench/capture_masked_run.py::part_sizes`, `bench/capture_masked_run.py::frame_spans`, `bench/capture_masked_run.py::_files`
- **Reads**:
  - `track`, `parts`, `ht_res["tube"]` (`ht_for_grade`), class maps, `least` floor (`1,024 px` default on canvas), and `SIZES_CANNOT_SEE`.
- **Transforms & Constants**:
  - **In-Memory Call Sequence Inside `_files`**: `off_head` -> `head_tube_for_capture` -> front-occluder depth clip
    (`parts & ~front_other_mask`) -> `refuse_pieces_on_a_head` / `largest_piece` / `starting_floor` / `masks__<by>.npz`
    write -> `subject_rows` (`mine`) -> `held_parts` & `under_doubted` -> `grade_holds` -> `hair_in_hole_files`.
  - `held_parts` holds across empty `parts` frames on a non-empty `track`; `grade_holds` grades each frame (`kept`,
    `held`, `emptied`) against the `least` floor (`1,024 px`) and since `1e91de26` accepts `head_track=ht_for_grade`
    (`ht_res["tube"]` when `head_tube` ran), exempting frames where `int((out[f] & head_track[f]).sum()) >= SEGMENT_PX`
    (`64 px`) from being emptied as fragments under `least_px` or `COLLAPSE_SHARE` (`4702-4705, 4708-4714`). Applies
    `--drop` spans, clips against `front_other_mask`, and sets `hair_pass["head"] = ht_res["tube"]` (`b461584e`).
- **Writes & Overwrites**:
  - Writes `subjects/<label>/part_grades.json` (Write #1) and `subjects/<label>/doubted_frames.json`, appends
    `manifest["subjects"][...]["sightings"]` (including `parts_doubted_and_left_on_source_frames` and
    `parts_emptied_by_grade_on_source_frames`), and performs **Write #1 of `subjects/<label>/parts_held__<by>.mkv`**.
- **Audit & Traps**:
  - **`doubted_frames.json` Pre-Fill Snapshot vs `part_grades.json` Sync (`1e91de26`, `8a420f89`)**: Since `1e91de26` and
    `8a420f89`, `hair_in_hole_files` (Stage 2.8) updates `part_grades.json` on disk (Write #2) and syncs
    `sighting["parts_doubted_and_left_on_source_frames"]` (`frame_spans(..., join=1)`) and
    `sighting["parts_emptied_by_grade_on_source_frames"]` when `her_hair_in_hole` restores a doubted/emptied frame,
    though `doubted_frames.json` and `masks__<by>.npz` remain pre-Stage-2.8 snapshots.
  - **`frame_spans(..., join=2)` Default Merge Trap**: `bench/capture_masked_run.py::frame_spans` (and its duplicate
    `bench/assemble_delivery.py::frame_spans`) defaults to `join=2` (`f - out[-1][1] <= join`), which merges
    non-adjacent frames `102` and `104` across `103` into `[102, 104]` unless `frame_spans(..., join=1)` is passed explicitly (`8a420f89`).

### Stage 2.8 — Her Hair in the Hole (`bench/capture_masked_run.py`)

- **Symbols** (`#ef42c261`): `bench/capture_masked_run.py::her_hair_in_hole`, `bench/capture_masked_run.py::hole_of_the_load`, `bench/capture_masked_run.py::whole_frame_hair`, `bench/capture_masked_run.py::hair_in_hole_files`, `bench/capture_masked_run.py::frame_spans`
- **Reads**:
  - `hair_pass["head"]` (`ht_res["tube"]`), `<preview>_<subj>_parts.npz`, `<preview>_strangers_classes.npz`, `other` keep-out masks, `head_hidden` / `hair_hers` / `hair_not_hers` marks, and `HAIR_IN_HOLE_CANNOT_SEE`.
- **Transforms & Constants**:
  - Computes 3 hair-recovery stages around `hole_of_the_load`:
    - Stage 1: own Sapiens-2B `Hair` connected to `head_tube` inside `hole(f)`.
    - Stage 2: whole-frame hair not on another person (`~taken[k]`, skipping `head_hidden` frames, `b461584e`).
    - Stage 3: remaining hair outside `hole(f)` (`~hole(f) & ~track[f]`, flagging undecided frames).
- **Writes & Overwrites**:
  - **Overwrites `subjects/<label>/parts_held__<by>.mkv` in place (Write #2)**, **overwrites `subjects/<label>/track__<by>.mkv`
    in place (`00a3f1d8`)** when `held` extends beyond `track`, **overwrites `subjects/<label>/part_grades.json` in place
    (Write #2, `1e91de26`)** when `her_hair_in_hole` changes pixel counts or restores an emptied frame, writes
    `subjects/<label>/hair_in_hole__<by>.json`, and updates `manifest["subjects"][...]["sightings"]`
    (`8a420f89`: `frame_spans(doubted_left, join=1)`) before `manifest.json` is written in Stage 2.9.
- **Audit & Traps**:
  - **`hole(f)` Pre-Carve/Pre-Feather Trap & `~track[f]` Trap (`3820654d`)**: In
    `her_hair_in_hole`, `hole(f)` still tests `out[f]` grown by `margin` (`30` canvas px) **before** the keep-out carve and feather
    (`4686`: `head[39] & ~hole(39) == 0 px`, leaving `130 px` of `head_tube[39]` in the feathered rim at `matte <= 127`), and
    Stage 3 still restricts by `~track[f]`, ignoring hair that lies inside the full-body `person_track` silhouette (`4912`).

### Stage 2.9 — Planned Region Prediction & `step_carves.json` (`bench/capture_masked_run.py`)

- **Symbols** (`#9ea8eb97`): `bench/capture_masked_run.py::planned_region`, `bench/capture_masked_run.py::predict_regions`, `bench/capture_masked_run.py::read_planned_regions`, `bench/capture_masked_run.py::step_keepout_carves`
- **Reads**:
  - Settled `parts_held__<by>.mkv`, `track__<by>.mkv`, and `other` keep-out masks.
- **Transforms & Constants**:
  - Simulates `video_mask.py::token_mask` per run (`predict_regions`) to write `runs/<name>/region.npz`, and calls `step_keepout_carves` to compute per-latent-step keep-out carve statistics (`runs/<name>/step_carves.json`), then writes `owners.npz` and `manifest.json`.
- **Writes & Overwrites**:
  - Writes `runs/<name>/region.npz`, `runs/<name>/step_carves.json`, `owners.npz`, and `manifest.json`.

---

## Stage 3 — Plan Execution (`plan_only`), On-Plan Re-Capture & Pre-Render Gate

### Stage 3.1 — Server-Side `plan_only` Execution (`video_mask.py`, `audio_freeze_song.py`, `bench/load_chain.py`)

- **Symbols** (`#32244193`): `bench/load_chain.py::step_plan`, `video_mask.py::MiniMaxH3MaskedSource.execute`, `audio_freeze_song.py::MiniMaxH3AudioFreezeSong.execute`, `audio_freeze_song.py::write_plan`
- **Reads**:
  - `<job>_api.json` submitted to ComfyUI with `plan_only=True` on node `74`, reading `parts_held__<by>.mkv` and `track__<by>.mkv`.
- **Transforms & Constants**:
  - Node `104` (`MiniMaxH3MaskedSource.execute`) computes per-window latent masks (`token_mask`), context padding (`pad_context`, `hold_context`), and zoom boxes (`zoom_plan`), saving `window_NNN_region.npz` (`save_window_region`).
  - Node `74` (`write_plan`) writes `<prefix>_plan.json` and exits before calling the DiT sampler.
- **Writes & Overwrites**:
  - `$H3_OUTPUT_DIR/<prefix>_plan.json` and `$H3_OUTPUT_DIR/<prefix>_window_NNN_region.npz`.

### Stage 3.2 — On-Plan Re-Capture (`bench/load_chain.py::step_on_plan`, `bench/capture_masked_run.py::_files`)

- **Symbols** (`#e2bd66e8`): `bench/load_chain.py::step_on_plan`, `bench/capture_masked_run.py::_files`, `bench/capture_masked_run.py::grid_carve`, `bench/capture_masked_run.py::_bridge_severed`, `bench/capture_masked_run.py::step_keepout_carves`
- **Reads**:
  - `$H3_OUTPUT_DIR/<prefix>_plan.json` and `$H3_OUTPUT_DIR/<prefix>_window_NNN_region.npz`.
- **Transforms & Constants**:
  - `bench/load_chain.py::step_on_plan` copies the initial capture folder to `<capture>_first` and invokes `bench/capture_masked_run.py files ... --overwrite` a **second time** on `<capture>` with the planned region files available.
  - During this second `_files` pass, keep-out `track` masks are grid-carved on the node's `16x16` grid (`grid_carve`, `_bridge_severed`), and Stages 2.6-2.9 run again.
- **Writes & Overwrites**:
  - **Overwrites `masks__<by>.npz`, `part_grades.json` (Writes #3 and #4 across the load), `parts_held__<by>.mkv` (Writes #3 and #4 across the load), `track__<by>.mkv`, `runs/<name>/region.npz`, `runs/<name>/step_carves.json`, and `owners.npz` in place**.
- **Audit & Traps**:
  - Even after `step_on_plan` re-runs `_files`, `masks__<by>.npz` and `doubted_frames.json` inside that second pass are still written before `her_hair_in_hole` performs Write #4 to `parts_held__<by>.mkv` and `part_grades.json` (`1e91de26`).

### Stage 3.3 — Pre-Render Verification, Flags & Frame Table (`bench/capture_masked_run.py`, `bench/inspect_mask_steps.py`, `bench/flags_before_render.py`, `bench/what_h3_got.py`, `bench/frame_table.py`)

- **Symbols** (`#27512bb8`): `bench/capture_masked_run.py::verify`, `bench/capture_masked_run.py::preflight`, `bench/inspect_mask_steps.py::inspect_capture`, `bench/flags_before_render.py::check_flags_read`, `bench/what_h3_got.py::audit_capture`, `bench/frame_table.py::build_frame_table`
- **Reads**:
  - Capture directory artifacts (`parts_held__<by>.mkv`, `part_grades.json`, `hair_in_hole__<by>.json`, `owners.npz`, `runs/<name>/region.npz`).
- **Transforms & Constants**:
  - `preflight` evaluates 24 flag functions (`flag_shots` through `flag_text`, bounded by `OTHER_CANNOT_SEE`).
  - `bench/inspect_mask_steps.py::inspect_capture` (`b461584e`, `ac803f92`) audits `[0]`-`[7]` forensic categories.
  - `bench/flags_before_render.py::check_flags_read` (`cbd39fbe`) enforces recipe agreement, coverage, and unresolved `hair_in_hole` gates.
  - `bench/frame_table.py::build_frame_table` builds the per-frame outcome table.
- **Writes & Overwrites**:
  - `verify.json`, `flags.json`, `flags_read.json`, and `frame_table__<by>.jsonl`.
- **Audit & Traps**:
  - **`part_grades.json` Sync (`1e91de26`) vs `masks__<by>.npz` Pre-Fill Snapshot**: Since `1e91de26`, `hair_in_hole_files` updates `part_grades.json` on disk when `her_hair_in_hole` changes pixel counts or restores an emptied frame, so `bench/frame_table.py` sees post-Stage-2.8 grades; however, `masks__<by>.npz` and `doubted_frames.json` remain pre-Stage-2.8 snapshots.

---

## Stage 4 — DiT Sampling, Token Masking & Window Lay-Back (`video_mask.py`, `audio_freeze_song.py`)

### Stage 4.1 — Window Slicing, `17n+5` Snapping, Context Padding & Zoom Crop (`video_mask.py`)

- **Symbols** (`#5115405f`): `video_mask.py::MiniMaxH3MaskedSource.execute`, `video_mask.py::window`, `video_mask.py::pad_context`, `video_mask.py::hold_context`, `video_mask.py::held_edge_source`, `video_mask.py::zoom_plan`, `video_mask.py::zoomed_source`
- **Reads**: Source frames, `parts_held__<by>.mkv`, `keep_out` masks, and window bounds from `workflows/masked_job_graph.py::window_for` / `window_pad` / `loop_plan.py::plan_windows`.
- **Transforms & Constants**: Slices each window (`length` on H3's `17n + 5` 17-frame VAE clip grid via `h3_rules.py::snap_length` / `on_both_clocks`), applies all-after context padding (`pad_context`) and outer-edge held frame repetition (`hold_context`, `held_edge_source`), and crops to `zoom_box` (`zoomed_source`) when `zoom` is active.
- **Writes & Overwrites**: Passes windowed source tensor, pixel mask, and `token_mask` into the ComfyUI conditioning dict (see [code_map_contracts.md](code_map_contracts.md) Section `2b`).

### Stage 4.2 — Latent `16x16` Token Mask & `(1, 4, 4, 4, 4)` Step Schedule (`video_mask.py`)

- **Symbols** (`#225af39d`): `video_mask.py::run_lengths`, `video_mask.py::token_mask`, `video_mask.py::grow`, `video_mask.py::margins`, `video_mask.py::drop_specks`, `video_mask.py::close_lone_holes`, `video_mask.py::drop_islands`
- **Reads**: Window pixel mask `(T, H, W)` and recipe parameters (`grow_pixels`, `feather_pixels`, `keep_out`).
- **Transforms & Constants**: Pools pixel masks onto the `(t_lat, H//16, W//16)` spatiotemporal token grid using the 17-frame VAE clip cycle `1, 4, 4, 4, 4` (`run_lengths`: 5 latent steps per 17 video frames, `5n + 2` steps for `17n + 5` frames), drops lone keep-out specks (`drop_specks`), closes 1-cell holes (`close_lone_holes`), and drops disconnected islands (`drop_islands`).
- **Writes & Overwrites**: Saves `window_NNN_region.npz` (`save_window_region`) with `tokens`, `matte`, `kept`, `zoom_box`, and padding metadata.

### Stage 4.3 — Pixel Alpha Feathering, Cut Gate & Window Lay (`video_mask.py`)

- **Symbols** (`#d578991a`): `video_mask.py::pixel_alpha`, `video_mask.py::changed_alpha`, `video_mask.py::cut_gate`, `video_mask.py::unlaid_frames`, `video_mask.py::lay_window`, `video_mask.py::lay_back`
- **Reads**: DiT-decoded window frames, original source frames, `pixel_alpha`, and `cut_gate`.
- **Transforms & Constants**: Zeros alpha where `unlaid_frames` or `cut_gate` refuses laying, composites generated pixels over source (`lay_window`), and maps zoomed windows back onto the full canvas (`lay_back`, saving the exact canvas-space `matte`).
- **Writes & Overwrites**: Produces composited window frames, `matte`, and `unlaid` frame indices for Node `74`.

### Stage 4.4 — Padding Trim, Multi-Window Join & Audio Silencing (`audio_freeze_song.py`, `loop_plan.py`)

- **Symbols** (`#7108e5ad`): `audio_freeze_song.py::MiniMaxH3AudioFreezeSong.execute`, `loop_plan.py::frames_kept`, `loop_plan.py::frames_written`, `audio_freeze_song.py::heard_under_held`, `audio_freeze_song.py::silenced`
- **Reads**: Composited windows from Stage 4.3, window padding/trim metadata, `loop_plan.py` counts, and source audio.
- **Transforms & Constants**: Trims padded context frames beyond the shot's kept span, joins multi-window stretches (`frames_kept`, `frames_written`), and silences audio under held padding (`heard_under_held`, `silenced`) during DiT sampling.
- **Writes & Overwrites**: Assembles the output frame sequence and muxes original shot audio onto the final MP4.

### Stage 4.5 — Server-Side Output Artifact Writes (`audio_freeze_song.py`)

- **Symbols** (`#c54e1f76`): `audio_freeze_song.py::_write_pieces_mp4`, `audio_freeze_song.py::_write_frames_mp4`, `audio_freeze_song.py::_write_review_mp4`, `audio_freeze_song.py::zoomed_record`
- **Reads**: Assembled load frames, raw uncomposited DiT pieces, and per-window metadata.
- **Writes & Overwrites**: Writes `$H3_OUTPUT_DIR/<prefix>.mp4`, `<prefix>_pieces.mp4`, `<prefix>_frames.mp4`, `<prefix>_review.mp4`, `<prefix>.json`, and `<prefix>_zoomed.json` (collected by `bench/masked_job.py::collect`; see Section `2a.2` of [code_map_contracts.md](code_map_contracts.md)).

---

## Stage 5 — Delivery Table, Plane Compositing, Worn Layer & Final Assembly (`bench/assemble_delivery.py`)

### Stage 5.1 — Delivery Table & Matte Resolution (`bench/assemble_delivery.py`, `bench/delivery_table.py`)

- **Symbols** (`#9c78b08a`): `bench/assemble_delivery.py::read_table`, `bench/assemble_delivery.py::Matte`, `bench/assemble_delivery.py::laid_by_of`, `bench/assemble_delivery.py::_laid_by_of`, `bench/delivery_table.py::decide`
- **Reads**: `data/<video>/delivery_table.json` and each kept run's `_region.npz` (`matte`, `kept`, `tokens`).
- **Transforms & Constants**: Resolves which run lays each `(subject, frame)` span and constructs `Matte` objects (`laid_by_of`, `rule: "matte"` when saved `matte` is present in `_region.npz`).

### Stage 5.2 — Multi-Subject Depth Ordering & Restore Weights (`bench/assemble_delivery.py`)

- **Symbols** (`#663a75fe`): `bench/assemble_delivery.py::Captures`, `bench/assemble_delivery.py::owners`, `bench/assemble_delivery.py::weight_of`, `bench/assemble_delivery.py::restore_weight`
- **Reads**: Per-capture `owners.npz`, `<preview>_depth.npz`, `<preview>_strangers_classes.npz`, and `delivery_table.json` (`restore_words`).
- **Transforms & Constants**: Orders overlapping subject planes per frame (`owners`), computes blend weights (`weight_of`), and builds `restore_weight` masks where original source pixels are restored over generated planes.

### Stage 5.3 — Worn Layer (Glasses/Jewelry/Apparel) & Plane Compositing (`bench/assemble_delivery.py`)

- **Symbols** (`#b43c56aa`): `bench/assemble_delivery.py::placements_of`, `bench/assemble_delivery.py::worn_layer`, `bench/assemble_delivery.py::lay_plane`, `bench/assemble_delivery.py::laid_over`, `bench/assemble_delivery.py::carry_edge_rows`
- **Reads**: Source frames, kept render frames (`piece_frames`), 308-keypoint face shifts (`shifts_of`, `moved_by`), and Sapiens-2B `worn_classes`.
- **Transforms & Constants**: Tracks worn items (`placements_of`, `worn_layer`) across face shifts, composites each subject plane (`lay_plane`, `laid_over`), and clamps zoom-box edge rows (`carry_edge_rows`) to prevent border seams.

### Stage 5.4 — Final MP4 Encoding, Audit & Sidecar (`bench/assemble_delivery.py`, `bench/stack_delivery.py`, `bench/rebuild_delivery.py`)

- **Symbols** (`#3dca7621`): `bench/assemble_delivery.py::build`, `bench/assemble_delivery.py::flags_of`, `bench/assemble_delivery.py::check`, `bench/assemble_delivery.py::write_sidecar`, `bench/stack_delivery.py::compare`, `bench/rebuild_delivery.py::rebuild`
- **Writes & Overwrites**: Muxes video and original audio into `$H3_OUTPUT_DIR/Video/<video>/<delivery>.mp4` and `<delivery>.json` sidecar (`write_sidecar`); `bench/rebuild_delivery.py::verify_reused_clip` and `verify_poc_on_disk` refuse delivery if `table_laid_sha` or upstream mtime (`bench/_lib/code_stamp.py::check_upstream_mtime`) has drifted.
