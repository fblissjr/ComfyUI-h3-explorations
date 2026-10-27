# Masked V2V Code Map, Traps & Phased Simplification Plan (`2026-10-26` / `2026-10-27`)

This document is the primary architecture router, forensic postmortem ledger, and phased simplification plan
for the masked video-to-video (`masked_v2v`) lane.

---

## 1. Modular Code Map Router & Anti-Drift Runbook

To prevent Gemini/AGY sessions from stalling on `read_file` (`46,080`-byte per-call cap) or `edit_file`
(`3,500`-char table rows), and to ensure line numbers never silently drift as code lands, the code map is
split into 4 focused files (`< 35 KB` and `< 400` chars/line each) backed by `bench/masked_code_map.py`:

| Document | Scope & Contents | Typical Size |
|---|---|---|
| **[docs/research/masking/2026-10-26_code_map_and_simplification_plan.md](2026-10-26_code_map_and_simplification_plan.md)** (this file) | Router, anti-drift runbook, the 6 high-cost overwrite/stale-record traps, `2026-10-26` commit ledger (`b461584e..d083ede0`), and Part `5a-5g` refactoring + morning build specs | `~20 KB` |
| **[docs/masked_v2v/CODE_MAP_INDEX.md](../../masked_v2v/CODE_MAP_INDEX.md)** | **Auto-generated AST Jump Index (Part 3) & `cannot_see` Boundaries (Part 2d)** across all 6 core monoliths (`3.1-3.6`) and 37 supporting files (`3.7`), with live `StartLine-EndLine` coordinates and 8-char AST body digests | `~31 KB` |
| **[docs/masked_v2v/code_map_stages.md](../../masked_v2v/code_map_stages.md)** | **Part 1 — End-to-End Mask & Tensor State Mutation Ledger (`Stage 1.1a` -> `Stage 5.4`)**: symbols, inputs, transforms, writes/overwrites, and traps at every sub-stage | `~18 KB` |
| **[docs/masked_v2v/code_map_contracts.md](../../masked_v2v/code_map_contracts.md)** | **Part 2 (`2a-2c`) & Part 4 (`4.1-4.5`)**: ComfyUI prompt JSON contracts, `$H3_OUTPUT_DIR` artifact writes, 4-way window & `4k+1` math diagram, dynamic imports, shared-name file collisions, and duplicate logic inventory | `~12 KB` |

### Anti-Drift Maintenance Workflow (For Any Session Landing Masked-Lane Code)

1. **Instant Symbol Lookup**:
   ```bash
   $H3_PYTHON bench/masked_code_map.py <symbol_or_file>
   ```
2. **Detect Narrative Stage Drift Before Refreshing the Index**:
   ```bash
   $H3_PYTHON bench/masked_code_map.py --drift
   ```
   Compares each function's current AST body digest against the digest recorded in `docs/masked_v2v/CODE_MAP_INDEX.md`
   and prints the exact `### Stage X.Y` / `### Section` headings in `code_map_stages.md` and `code_map_contracts.md`
   whose referenced functions changed.
3. **Update Narrative Stage Bullets & Regenerate the AST Index**:
   - Edit only the flagged `### Stage X.Y` block in `docs/masked_v2v/code_map_stages.md` or `docs/masked_v2v/code_map_contracts.md`.
   - Run `$H3_PYTHON bench/masked_code_map.py --write` to update `docs/masked_v2v/CODE_MAP_INDEX.md`.
4. **Verify All Code-Map Invariants (`--check`)**:
   ```bash
   $H3_PYTHON bench/masked_code_map.py --check
   ```
   Verifies in `< 0.2s` that `CODE_MAP_INDEX.md` matches the live AST, every `<path>.py::<symbol>` in `code_map_stages.md`
   and `code_map_contracts.md` exists on disk, every file is `< 35,000` bytes and `< 400` chars/line, and zero
   machine-specific `/home/` or `/Users/` paths appear.

---

## 2. Executive Summary — The 6 High-Cost Overwrite, Stale-Record & Blind-Spot Traps

Across the `2026-10-26` and `2026-10-27` forensic audits of `lead_fp_2050d`/`e`, `second_hfp_4647e`/`f`/`g`/`h`,
`lead_hfp_4902b`, and `lead_hfp_1420`, six structural patterns repeatedly cost hours of debugging or wasted DiT
renders:

1. **Silent 4x In-Place File Overwrites + Frozen Early Snapshots (`bench/capture_masked_run.py`)**:
   - `parts_held__<by>.mkv` is written at Stage 2.7 (`_files`), overwritten in place at Stage 2.8 (`her_hair_in_hole`),
     and overwritten in place twice more at Stage 3.2 (`step_keepout_carves`).
   - Meanwhile, `masks__<by>.npz` is written once at Stage 2.6 (before `grade_holds` and `her_hair_in_hole`) and
     `part_grades.json` is written once at Stage 2.7, leaving downstream readers (`bench/frame_table.py::build_frame_table`)
     comparing stale Stage 2.7 grades against Stage 3.2 masks.
2. **Stale `hold_to_track` Smearing Across Real Occlusions (`bench/shot_to_load.py::hold_to_track`)**:
   - In Stage 1.6a, `hold_to_track` (`_on_track_8px.mkv`) blindly freezes the last non-empty part mask for up to
     `hold_frames=8` whenever the part mask drops out, even when the subject walked behind an occluder (`4682-4685`,
     punching a `16,580 px` hole in the foreground man's chest) or moved `62 px` laterally (`4702-4709`).
3. **Pre-Carve / Pre-Feather Hole Reconstruction (`bench/capture_masked_run.py::hole_of_the_load`)**:
   - Before `3820654d`, `hole_of_the_load` reconstructed `hole(f)` from uncarved `as_read` without subtracting
     `keep_out` and without applying `video_mask.py`'s `grow_px`/`feather_px`/`16x16` token rounding, while Part 3 of
     `her_hair_in_hole` excluded `~track[f]` (blocking hair recovery inside the body silhouette on `4912`) and read
     unscoped `strangers_classes.npz` (`9c5a6841`).
4. **Post-Occlusion Flat Horizontal Slice Seed (`bench/head_tube_parts.py::_head_seed_from_person_mask`)**:
   - When SAM 3.1 `head_track` dropped out and re-seeded from the full-body mask, `_head_seed_from_person_mask`
     sliced the top `0.333` (`_HEAD_BOX_TOP_FRAC`) of the person bounding box (`y=318..470`), cutting horizontally
     across a bending/turning subject's shoulder and back in `lead_fp_2050d` (`f=33..45`, fixed in `4791284f`).
5. **Unowned SAM 3.1 `people.npz` Dropped Where Whole-Frame Sapiens2 Sees `c == 0` (`bench/strangers.py`)**:
   - In `bench/strangers.py::people_pass`, unowned SAM 3.1 person masks (`o[3] is None`) are ignored wherever
     whole-frame Sapiens-2B returns background (`c == 0`), AND `core[i]` requires non-apparel Sapiens-2B classes
     (`NO_CORE = {0, 1, 17, 18, 19, 20}`). When `core[i]` is empty (`1458`), `handed_back` drops the entire
     `43,190 px` stranger component because it touches a tracked subject's box.
6. **4-Way Window, Context Padding & `4k+1` Math Split Across 4 Modules**:
   - Window bounds, held-context padding, and `4k+1` length snapping are computed independently in
     `workflows/masked_job_graph.py`, `loop_plan.py`, `video_mask.py`, and `bench/capture_masked_run.py::predict_regions`
     (see [code_map_contracts.md](../../masked_v2v/code_map_contracts.md) Section `2b`).

---

## 3. Complete `2026-10-26` Commit Ledger (`b461584e..d083ede0`, 20 Commits)

Every commit landed on `2026-10-26` that shaped the current masked-lane code on disk:

- **`b461584e`** (`bench/capture_masked_run.py`, `bench/inspect_mask_steps.py`): Added `bench/inspect_mask_steps.py`;
  set `hair_pass["head"] = ht_res["tube"]`; gated `her_hair_in_hole` Part 2 by `~taken[k]`.
- **`3820654d`** (`bench/capture_masked_run.py`): Fixed `hole_of_the_load` to subtract `keep_out` and match node mask
  footprint instead of reading uncarved `as_read`.
- **`00a3f1d8`** (`bench/capture_masked_run.py`): Updated `her_hair_in_hole` to also write updated `track__<by>.mkv`
  alongside `parts_held__<by>.mkv`.
- **`45f3eaf1`** (`bench/capture_masked_run.py`): Added `_bridge_severed` to reconnect thin neck/hair bridges severed
  by `keep_out` carving when not occluded.
- **`4791284f`** (`bench/head_tube_parts.py`): Fixed `_head_seed_from_person_mask` flat `0.333` top-of-box slice on
  bending/turning postures (`lead_fp_2050d` `f=33..45`) by intersecting with `Face_Neck`/`Hair` and 308-pose head points.
- **`9c5a6841`** (`bench/capture_masked_run.py`, `bench/shot_to_load.py`): Exempted pose-verified small heads
  (`< 1,024 px` on `4702-4714`) from `grade_holds` erasure; scoped `head_marks_of` to own subject's masks; tightened
  `her_hair_in_hole` Part 3.
- **`cbd39fbe`** (`bench/shot_to_load.py`, `bench/flags_before_render.py`): Enforced settled `shot.recipe` defaults and
  `shot.shot_overrides` validation in `shot_to_load.py` and `bench/flags_before_render.py::recipe_check_of`.
- **`eb94f9df`** (`audio_freeze_song.py`, `bench/load_chain.py`): Added `plan_only` fast path (`write_plan`) in
  `MiniMaxH3AudioFreezeSong.execute`.
- **`ac803f92`** (`bench/what_h3_got.py`): Extended `audit_capture` to verify rendered window regions, references, and
  prompt text against plan.
- **`76533a22`** (`bench/head_reference.py`, `bench/draw_face_points.py`): Audited `<Video 1>` bare-frame gaps on
  `second_hfp_4647h` (`43/79` bare frames when `head_track` dropped out).
- **`cf244cb1`** (`bench/cast_file.py`, `subject_track.py`): Introduced 3-group `cast.jsonl` taxonomy (`named`,
  `unnamed_recurring`, `incidentals`).
- **`0121b793`** (`bench/cast_file.py`, `subject_track.py`): Enforced non-overlapping cast group validation in
  `bench/cast_file.py::read` and `subject_track.py::name_drawn`.
- **`ca8fdcc9`** (`subject_track.py`, `bench/facts_pass.py`): Recorded `empty_runs` on `subject_track` and `head_track`
  JSONs and surfaced them in `facts_pass.py`.
- **`da6921dc`** (`subject_track.py`, `bench/shot_to_load.py`): Routed tracker `auto_match` through `shot.gallery`
  (filmed person in the shot) instead of replacement identity stills.
- **`e41908c3`** (`bench/flags_before_render.py`): Added `unlaid_while_tracked_of` and `unresolved_hair_in_hole_of`
  pre-render gate checks.
- **`f18290a4`** (`bench/frame_table.py`, `bench/load_chain.py`): Added `verify_table_fresh` gate in `frame_table.py`
  and `bench/load_chain.py::step_table`.
- **`a934c112`** (`bench/assemble_delivery.py`): Added `carry_edge_rows` and `restore_weight` refinements for zoomed
  delivery planes.
- **`5209b7e1`** (`bench/rebuild_delivery.py`, `bench/timeline_page.py`): Added `table_laid_sha` and upstream mtime
  provenance checks across delivery rebuilds and timeline pages.
- **`83d4190b`** (`bench/review_pages.py`, `bench/said_on_shot.py`): Linked per-shot owner verdicts and answers into
  review pages and preflight summaries.
- **`d083ede0`** (`docs/wiki/handover.md`, `docs/wiki/decisions.md`): Recorded end-of-day handover and settled the 3
  morning builds (`Part 5e-5g` below).

---

## 4. Phased Simplification & Refactoring Plan (`Part 5a-5d`)

### Part 5a — Pure Capture Stage Pipeline & Single-Write Artifact Naming (`bench/capture_masked_run.py`)

- **Problem**: `bench/capture_masked_run.py::_files` (`~670` lines) mutates `masks_by[by]` across 8 stages, writes
  `masks__<by>.npz` early at Stage 2.6, writes `part_grades.json` at Stage 2.7, and overwrites `parts_held__<by>.mkv`
  in place 4 times across Stages 2.7, 2.8, and 3.2.
- **Target Architecture**:
  1. Refactor Stage 2 into pure functions taking and returning an explicit `CaptureStageState` dataclass (`as_read`,
     `carved`, `off_head_filtered`, `head_tube_merged`, `depth_clipped`, `floored`, `graded`, `hair_filled`).
  2. Write `masks__<by>.npz` and `part_grades.json` **after** Stage 2.8 (`her_hair_in_hole`) and update both at the end
     of Stage 3.2 (`step_keepout_carves`) so `masks__<by>.npz`, `part_grades.json`, `parts_held__<by>.mkv`, and
     `frame_table.json` always agree on the settled mask.
  3. Record each stage's pixel count and SHA-256 in `mask_steps.json` so `bench/inspect_mask_steps.py` reads the ledger
     directly instead of re-deriving intermediate masks.

### Part 5b — Unified `window_spec.py` for 4-Way Window, Padding & `4k+1` Math

- **Problem**: `workflows/masked_job_graph.py`, `loop_plan.py`, `video_mask.py`, and
  `bench/capture_masked_run.py::predict_regions` each compute window bounds, held-context padding, and `4k+1` length
  snapping separately (Section `2b` & `4.3` of [code_map_contracts.md](../../masked_v2v/code_map_contracts.md)).
- **Target Architecture**:
  1. Extract a pure-Python, zero-ComfyUI module `window_spec.py` at repo root exporting `WindowSpec` and
     `plan_window_specs(shot_span, mask_active_frames, cut_frames, max_length=362)`.
  2. Route `workflows/masked_job_graph.py::window_for` / `window_pad`, `loop_plan.py::plan_windows`,
     `video_mask.py::window`, and `bench/capture_masked_run.py::predict_regions` through `window_spec.py`.

### Part 5c — `bench/_lib/` Consolidation & Monolith Decomposition

- **Problem**: Duplicate helpers across 5-6 files (Part 4 of [code_map_contracts.md](../../masked_v2v/code_map_contracts.md))
  and three `4,300-6,800`-line monoliths (`bench/capture_masked_run.py`, `bench/shot_to_load.py`,
  `bench/assemble_delivery.py`).
- **Target Architecture**:
  1. Move bit-packed `.npz` mask helpers (`_pack`, `_unpack_mask`) into `bench/_lib/frames.py`.
  2. Create `bench/_lib/sapiens_classes.py` as the single source of truth for Sapiens-2B 28-class indices (`FACE_NECK`,
     `HAIR`, `MOUTH_CLASSES`, `NO_CORE`, `WORN_CLASSES`).
  3. Split `bench/capture_masked_run.py` into `bench/capture_stages.py` (Stages 2.1-2.9),
     `bench/capture_preflight.py` (the 24 `flag_*` checks), and `bench/capture_look.py` (`look`, `video`, diagnostic
     strips), keeping CLI subcommands in `bench/capture_masked_run.py` unchanged.

### Part 5d — Byte-for-Byte Replay Gate Before Any Refactor Lands

- **Rule**: Every refactoring commit in Parts `5a-5c` must verify bit-identical `.npz` masks (`masks__<by>.npz`,
  `planned_region__<node_id>.npz`, `owners.npz`) and identical `preflight.json` flags on a reference capture directory
  before landing on `main`.

---

## 5. Settled Morning Builds (`Part 5e-5g`, Ready for Worktree Execution)

These three builds are settled in `docs/wiki/decisions.md` and `docs/wiki/handover.md` and are specified down to exact
functions and files:

### Part 5e — Build 1: Retire `hold_to_track` & Unify Head-Room Part Definition (`bench/shot_to_load.py`, `bench/capture_masked_run.py`)

- **Root Cause**:
  - `bench/shot_to_load.py::hold_to_track` (`L1280-L1391`) freezes the last seen head/part mask across dropout frames
    when building `_part_head_hair_on_track_8px.mkv`. On `second_hfp_4647h` (`4682-4685`), this held frame `4681`'s
    `16,580 px` head mask over the foreground man's chest while the woman was completely behind him, and on `4702-4709`
    smeared a stale `737 px` spot while her real head moved `62 px` right.
- **Both Ends (Writers & Readers)**:
  - **Writers**: `bench/shot_to_load.py::part_on_track`, `hold_to_track`, `make_part_on_track` (`L1242-L1433`);
    `bench/head_tube_parts.py::head_tube_for_capture` (`L1035-L1200`); `bench/capture_masked_run.py::_files`,
    `grade_holds`, `her_hair_in_hole`.
  - **Readers**: `bench/capture_masked_run.py::_files` (reads `_part_head_hair_on_track_8px.mkv`),
    `bench/inspect_mask_steps.py::inspect_capture`, `bench/flags_before_render.py::check_flags_read`,
    `bench/frame_table.py::build_frame_table`.
- **Implementation Spec**:
  1. In `bench/shot_to_load.py::part_on_track` / `make_part_on_track`, retire temporal holding (`hold_to_track`) so
     `_on_track_8px.mkv` contains only real per-frame pixels present on that frame (`hold_frames=0` at the pre-capture
     part stage).
  2. Gate `sapiens2_parts.py` / `part_on_track` head-hair emission by `head_track` presence or verified 308-pose face
     landmarks so occluded forearm-only crops (`4682-4685`) cannot emit hallucinated `Face_Neck`/`Hair` pixels.
  3. Settle downstream floor and hole-fill in `bench/capture_masked_run.py` (`starting_floor`, `grade_holds`,
     `her_hair_in_hole`) on top of the un-smeared part input and `ht_res["tube"]`.

### Part 5f — Build 2: Include Unowned SAM 3.1 `people.npz` Detections & Fix `core[i]` Hand-Back in `bench/strangers.py`

- **Root Cause**:
  - On shot `1420` (`frame 1458`), whole-frame Sapiens-2B (`bench/strangers.py::people_pass`, `L493-L566`) missed the
    foreground man's torso (`c == 0`), while SAM 3.1 `<preview>_people.npz` detected him (`43,190 px`) as unowned
    (`o[3] is None`). Because `people_pass` only labeled pixels where `c > 0` and defined `core[i]` (`L564`) strictly
    from Sapiens-2B non-apparel classes (`NO_CORE = {0, 1, 17, 18, 19, 20}`), the man was absent from both
    `people[i]` and `core[i]`. Even if unioned into `people[i]`, `handed_back` (`L171-L234`) drops any component
    touching a subject box where `(comp & core[i]).any()` is false.
- **Both Ends (Writers & Readers)**:
  - **Writer**: `bench/strangers.py::people_pass` (`L493-L566`), `handed_back` (`L171-L234`), `run` (`L593-L645`).
  - **Readers**: `bench/shot_to_load.py::others_of_load` / `run_strangers` (`L1788-L1850`),
    `bench/capture_masked_run.py::_depth_medians_and_owner` (`L181-L246`), `bench/capture_masked_run.py::her_hair_in_hole`
    (`L5278-L5565`).
- **Implementation Spec**:
  1. In `bench/strangers.py::people_pass`, union unowned SAM 3.1 `people.npz` detections (`o[3] is None` after
     subtracting owned subject tracks) into `people[i]` even where whole-frame Sapiens-2B has `c == 0`.
  2. Include interior (eroded) unowned SAM 3.1 person masks in `core[i]` (`L564`) so `handed_back` (`L171-L234`)
     recognizes SAM-backed strangers as having a valid core and does not hand them back when they overlap a tracked
     subject's bounding box.
  3. Update `<preview>_strangers.json["cannot_see"]` in `bench/strangers.py::run` (`L617-L642`) and run
     `$H3_PYTHON bench/masked_code_map.py --write`.

### Part 5g — Build 3: Tracker Post-Occlusion Re-Finding (`--head-per-stretch`), `shot.gallery` Match & `second_hfp_4647i` `<Video 1>` Remake

- **Root Cause**:
  - In `subject_track.py`, single-seed propagation across a full-occlusion gap (`4682-4685` in shot `4647`) failed to
    re-acquire the woman's small head when she re-emerged (`4702-4714`, `4721-4746`), leaving `43/79` frames of
    `second_hfp_4647h_drawn.mp4` (`<Video 1>`) bare black (`76533a22`).
- **Both Ends (Writers & Readers)**:
  - **Writers**: `subject_track.py::parse_head_mode`, `_track_head_span`, `_span_head`, `follow`, `auto_match`,
    `name_drawn`; `bench/draw_face_points.py::run`; `bench/shot_to_load.py::drawn_form`.
  - **Readers**: `bench/head_tube_parts.py::sam31_head_tube`, `bench/capture_masked_run.py::_files`,
    `bench/what_h3_got.py::audit_capture`.
- **Implementation Spec**:
  1. Make `--head-per-stretch` (`per-stretch` re-seeding on each contiguous visible stretch bounded by `empty_runs` /
     pose landmarks) the default head-tracking mode in `subject_track.py::parse_head_mode` / `_track_head_span`.
  2. Verify `auto_match` uses `shot.gallery` (`da6921dc`) and `name_drawn` enforces the 3-group `cast.jsonl` taxonomy
     (`cf244cb1`, `0121b793`).
  3. Re-track shot `4647` `second`, regenerate `second_hfp_4647i` `<Video 1>` (`_drawn.mp4`) and `<Video 2>`
     (`_head_ref.mp4`), and verify via `bench/what_h3_got.py` that `<Video 1>` has continuous landmark coverage across
     all visible stretches before queueing `second_hfp_4647i`.
