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
| **[docs/masked_v2v/code_map_stages.md](../../masked_v2v/code_map_stages.md)** | **Part 1 — End-to-End Mask & Tensor State Mutation Ledger (`Stage 1.1a` -> `Stage 5.4`)**: symbols, inputs, transforms, writes/overwrites, and traps at every sub-stage | `~32 KB` |
| **[docs/masked_v2v/code_map_contracts.md](../../masked_v2v/code_map_contracts.md)** | **Part 2 (`2a-2c`) & Part 4 (`4.1-4.5`)**: ComfyUI prompt JSON contracts, `$H3_OUTPUT_DIR` artifact writes, 4-way window & `17n+5` math diagram, dynamic imports, shared-name file collisions, and duplicate logic inventory | `~14 KB` |

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
   - Keep exact line numbers exclusively in `docs/masked_v2v/CODE_MAP_INDEX.md` (cite `<path>.py::<symbol>` in narrative docs).
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

1. **Silent 4x In-Place File Overwrites + Pre-Fill Snapshot Lag (`bench/capture_masked_run.py`, `bench/load_chain.py`)**:
   - During a single `bench/capture_masked_run.py::_files` pass, `masks__<by>.npz` is written at Stage 2.6 (before
     `held_parts`, `grade_holds`, `--drop`, and `her_hair_in_hole`), `part_grades.json` and `doubted_frames.json` are
     written at Stage 2.7, and `parts_held__<by>.mkv` is written at Stage 2.7 and overwritten in place at Stage 2.8
     (`bench/capture_masked_run.py::hair_in_hole_files`, which also overwrites `track__<by>.mkv` and, since `1e91de26` /
     `8a420f89`, overwrites `part_grades.json` in place and syncs `sighting["parts_doubted_and_left_on_source_frames"]`
     with `frame_spans(..., join=1)`).
   - Then in Stage 3.2 (`bench/load_chain.py::step_on_plan`), `capture_masked_run.py files --overwrite` runs a **second**
     time on the same directory, overwriting `masks__<by>.npz`, `part_grades.json` (4 writes total),
     `parts_held__<by>.mkv` (4 writes total), and `track__<by>.mkv` in place. Within both passes, `masks__<by>.npz` and
     `doubted_frames.json` remain pre-Stage-2.8 snapshots that do not reflect `her_hair_in_hole`.
2. **`hold_to_track` Clipping to Drifted/Narrow `person_track` + Downstream `grade_holds` Floor (`bench/shot_to_load.py`, `bench/capture_masked_run.py`)**:
   - In Stage 1.6a, `bench/shot_to_load.py::hold_to_track` computes `held = a & strangers.grown(kept, px)`, clipping the
     subject's per-frame Sapiens part `a` to her SAM 3.1 `person_track` grown by `8px` (`_on_track_8px.mkv`). When
     `person_track` misses outer hair strands (`4912`) or drifts onto another person (`4699-4702`), valid head/hair pixels
     are clipped away before Stage 2 starts (still active on `main`).
   - Downstream in `bench/capture_masked_run.py::grade_holds`, valid small heads `< 1,024 px` (`4702-4705, 4708-4714`) were
     previously emptied; since `1e91de26`, `_files` passes `head_track=ht_for_grade` (`ht_res["tube"]`) to `grade_holds`,
     exempting frames where `int((out[f] & head_track[f]).sum()) >= SEGMENT_PX` (`64 px`) from floor/collapse emptying.
3. **Pre-Carve / Pre-Feather `hole(f)` & `~track[f]` in `her_hair_in_hole` (`bench/capture_masked_run.py`)**:
   - In `bench/capture_masked_run.py::her_hair_in_hole`, `hole(f)` tests `out[f]` grown by `margin` (`30` canvas px)
     **before** the keep-out carve and feather (`4686`: `head[39] & ~hole(39) == 0 px`, leaving `130 px` of
     `head_tube[39]` in the feathered rim at `matte <= 127`), while Part 3 (`in_hole_part3`) checks `~track[f]`,
     blocking hair recovery wherever hair lies inside the full-body `person_track` silhouette (`4912`) (still active on `main`).
4. **Post-Occlusion Flat Horizontal Slice Seed (`bench/head_tube_parts.py::_head_seed_from_person_mask` — Fixed on `main` in `1e91de26`)**:
   - Before `1e91de26`, when `head_track == 0 px` (`4682-4695` in `second_hfp_4647h`), a regained `head_tube` frame in
     `bench/head_tube_parts.py::_head_seed_from_person_mask` (with `tracker=None`) took a flat horizontal slice
     (`head_rel_y1`) of `person_track`, slicing `945-1,354 px` off crop-Sapiens's valid head/hair mask (`4686`).
   - Fixed in `1e91de26`: `_head_seed_from_person_mask` now takes `own_head_2d & person_mask_2d` when non-empty, and
     `head_tube_for_capture` replaces regained/bridged `person_track` tube frames (`~ht_has_any`) with `parts_bool[t]` and
     restricts hair relabeling to `ht_in_track | (tube_full & parts_bool)`.
5. **Unowned SAM 3.1 `people.npz` Dropped Where Whole-Frame Sapiens2 Sees `c == 0` (`bench/strangers.py` — Fixed on `main` in `426638a6`)**:
   - Before `426638a6`, in `bench/strangers.py::run`, unowned SAM 3.1 person detections (`who is None` in
     `<preview>_people.npz`) were NOT unioned into `people[i]` (`c > 0`) or `core[i]` (`np.isin(c, core_ids)`) where
     whole-frame Sapiens-2B missed a person (`c == 0`, as on `4686-4693` in `4647k` and `1458`).
   - Fixed in `426638a6`: `bench/strangers.py::run` now computes `det |= m & (c == 0)` for `who is None` in `others[i]`
     and sets `people[i] = (c > 0) | det` and `core[i] = np.isin(c, core_ids) | det` so `handed_back` preserves
     SAM-backed strangers (`held.any()` is true) instead of dropping them as `NO_CORE`.
6. **4-Way Window, Context Padding & `17n+5` Math Split Across 4 Modules**:
   - Window bounds, all-after context padding (`before = 0, after = extra`), outer-edge held context, and `17n + 5`
     causal-VAE length snapping (`5n + 2` latent steps via `1, 4, 4, 4, 4` per 17-frame clip) are computed across
     `workflows/masked_job_graph.py`, `h3_rules.py`, `loop_plan.py`, `video_mask.py`, and
     `bench/capture_masked_run.py::predict_regions` (see [code_map_contracts.md](../../masked_v2v/code_map_contracts.md)
     Section `2b`).

---

## 3. Verbatim `2026-10-26` Commit Ledger (`b461584e..d083ede0`, 20 Commits on `main`)

Every commit landed on `main` on `2026-10-26`, separating actual code commits (Builds A, C, D) from forensic
documentation commits:

### Code Commits Landed on `2026-10-26` (6 Commits)
- **`b461584e`** (`fix(bench): pass head_tube to her_hair_in_hole and inspector, skip hidden frames, and restore full others subtraction`):
  Updated `bench/capture_masked_run.py` (`hair_pass["head"] = ht_res["tube"]`, `hidden` frame skip in `her_hair_in_hole`
  Stage 2/3, full `~taken[k]` subtraction) and `bench/inspect_mask_steps.py`.
- **`ac803f92`** (`fix(bench): use _iso_epoch in git-less check fake and test active-step 0px directly`):
  Updated `bench/inspect_mask_steps.py`, `bench/what_h3_got.py`, and `bench/check_inspect_mask_steps.py` (`[2]` flags
  active latent steps where some frame has `0 px` mask).
- **`00a3f1d8`** (`fix(bench): include held pixels in track mkv when her_hair_in_hole extends beyond track`):
  Updated `bench/capture_masked_run.py::hair_in_hole_files` to rewrite `track__<by>.mkv` when `held` extends beyond
  `given["track"]` (`4943-4946` crown pixels on `4912`).
- **`45f3eaf1`** (`fix(bench): carve 4-connected bridge in grid_carve when keep-out pinches hers unit into drop_islands`):
  Added `bench/capture_masked_run.py::_bridge_severed` inside `grid_carve`.
- **`cbd39fbe`** (`bench/shot_to_load, flags_before_render: refuse CLI overrides against settled plan.json recipes`):
  Enforced `subjects.<subject>.recipe` and `shot_overrides` in `bench/shot_to_load.py` and
  `bench/flags_before_render.py::recipe_check_of`.
- **`d083ede0`** (`add incomplete 2026-10-26_code_map_and_simplification_plan to complete another day`):
  Checked in the initial work-in-progress draft of this document.

### Forensic & Handover Documentation Commits Landed on `2026-10-26` (14 Commits)
- **`b375d53e`** (`docs(masked_v2v): record 4647 cut-vs-raw and cast-seed findings in forensics, verifying, AGENTS, and decisions`).
- **`7c195e68`** (`docs(masked_v2v): cite assemble_delivery rule 'matte' and cast.jsonl line 186 in AGENTS.md and decisions.md`).
- **`68e8086d`** (`Claude docs edits and annotations`).
- **`f6028d64`** (`docs(wiki): record cast box rule, 4647h postmortem, Build D hash, and 2026-10-26 handover`).
- **`ee8c1d8e`** (`changelog: the entry for 2026-10-26`).
- **`70df42cd`** (`docs: annotate masked_v2v & handover docs with close-of-2026-10-26 findings`).
- **`3820654d`** (`docs: note her_hair_in_hole pre-carve/pre-feather hole(f) test on 4686` — **documented** the `hole(f)` bug in prose; code fix is in Build 1 / `Part 5e` below).
- **`4791284f`** (`docs: record head_tube regained flat-slice mechanism on 4686-4695` — **documented** the `_head_seed_from_person_mask` flat-slice bug in prose; code fix is in Build 1 / `Part 5e` below).
- **`9c5a6841`** (`docs: record 4647h reference bare frames & downstream grade_holds floor in §0.9 and handover` — **documented** the `grade_holds` `< 1,024 px` floor and `4647h` bare reference frames in prose; code fixes are in Builds 1 & 3 / `Part 5e, 5g` below).
- **`76533a22`** (`docs: specify no drawing word carried when remaking 4647i reference`).
- **`ce601a92`** (`docs: add cold-start dry-run file:line pointers and code-gate targets to handover`).
- **`cf244cb1`** (`docs: clarify Group 1 and Group 3 cast boxes stand while only Group 2 screenshot boxes withdraw`).
- **`0121b793`** (`docs: define Group 2 owner visual cues and confirmed words as recorded ground truth driving pipeline logic`).
- **`da6921dc`** (`docs: scrap stale runbook/pipeline duplicates and align masked_v2v links with Group 1/2/3 & recipe gates`).

---

## 4. Phased Simplification & Refactoring Plan (`Part 5a-5d`)

### Part 5a — Pure Capture Stage Pipeline & Single-Write Artifact Naming (`bench/capture_masked_run.py`)

- **Problem**: `bench/capture_masked_run.py::_files` (`664` lines, `54.5 KB`) mutates masks across 8 stages, writes
  `masks__<by>.npz` early at Stage 2.6, writes `part_grades.json` at Stage 2.7, and overwrites `parts_held__<by>.mkv`
  in place at Stage 2.7 and Stage 2.8 (`hair_in_hole_files`), then runs all over again during `bench/load_chain.py::step_on_plan`.
- **Target Architecture**:
  1. Refactor Stage 2 into pure functions taking and returning an explicit `CaptureStageState` dataclass (`as_read`,
     `carved`, `off_head_filtered`, `head_tube_merged`, `depth_clipped`, `floored`, `graded`, `hair_filled`).
  2. Write `masks__<by>.npz` and `part_grades.json` **after** Stage 2.8 (`her_hair_in_hole`) so `masks__<by>.npz`,
     `part_grades.json`, `parts_held__<by>.mkv`, and `bench/frame_table.py::build_frame_table` always agree on the
     settled mask.
  3. Record each stage's pixel count and SHA-256 in `mask_steps.json` so `bench/inspect_mask_steps.py::inspect_capture`
     reads the ledger directly instead of re-deriving intermediate masks.

### Part 5b — Unified `window_spec.py` for 4-Way Window, Padding & `17n+5` Math

- **Problem**: `workflows/masked_job_graph.py`, `loop_plan.py`, `video_mask.py`, and
  `bench/capture_masked_run.py::predict_regions` each compute window bounds, all-after context padding, held-edge
  context, and `17n + 5` length snapping separately (Section `2b` & `4.3` of
  [code_map_contracts.md](../../masked_v2v/code_map_contracts.md)).
- **Target Architecture**:
  1. Extract a pure-Python, zero-ComfyUI module `window_spec.py` at repo root exporting `WindowSpec` and
     `plan_window_specs(shot_span, mask_active_frames, cut_frames, max_length=362)`.
  2. Route `workflows/masked_job_graph.py::window_for` / `window_pad` / `held_context`, `loop_plan.py::plan_windows`,
     `video_mask.py::window`, and `bench/capture_masked_run.py::predict_regions` through `window_spec.py`.

### Part 5c — `bench/_lib/` Consolidation & Monolith Decomposition

- **Problem**: Duplicate helpers across 5-6 files (Part 4 of [code_map_contracts.md](../../masked_v2v/code_map_contracts.md))
  and three `4,300-6,800`-line monoliths (`bench/capture_masked_run.py`, `bench/shot_to_load.py`,
  `bench/assemble_delivery.py`).
- **Target Architecture**:
  1. Move bit-packed `.npz` mask helpers (`_pack`, `_unpack_mask`) into `bench/_lib/frames.py`.
  2. Create `bench/_lib/sapiens_classes.py` as the single source of truth for Sapiens-2B 28-class indices (`FACE_NECK`,
     `HAIR`, `MOUTH_CLASSES`, `CORE_CLASSES`, `WORN_CLASSES`).
  3. Split `bench/capture_masked_run.py` into `bench/capture_stages.py` (Stages 2.1-2.9),
     `bench/capture_preflight.py` (the 24 `flag_*` checks), and `bench/capture_look.py` (`look`, `video`, diagnostic
     strips), keeping CLI subcommands in `bench/capture_masked_run.py` unchanged.

### Part 5d — Byte-for-Byte Replay Gate Before Any Refactor Lands

- **Rule**: Every refactoring commit in Parts `5a-5c` must verify bit-identical `.npz` masks (`masks__<by>.npz`,
  `runs/<name>/region.npz`, `owners.npz`) and identical `flags.json` / `verify.json` output on a reference capture
  directory before landing on `main`.

---

## 5. Morning Builds Status (`Part 5e-5g`, Landed on `main` in `426638a6`, `1e91de26`, `6b0ea9a6`, `8a420f89`)

These three builds were settled in `docs/wiki/decisions.md` and `docs/wiki/handover.md` and landed on `main` on
`2026-10-27`. Below is the exact ledger of what each commit changed on `main` vs. what remains open:

### Part 5e — Build 1 (`1e91de26`, `8a420f89`): `head_tube` Seed & Regained-Frame Fix, `grade_holds` `head_track` Exemption & `part_grades.json` Sync

- **Landed on `main` (`1e91de26`, `8a420f89`)**:
  1. `bench/head_tube_parts.py::_head_seed_from_person_mask` & `sam31_head_tube`: takes `own_head_2d & person_mask_2d`
     when non-empty instead of a flat horizontal `head_rel_y1` slice of `person_track`.
  2. `bench/head_tube_parts.py::head_tube_for_capture`: on frames where `head_track` was empty and `sam31_head_tube`
     regained/bridged from `person_track`, replaces `tube[t]` with `parts_bool[t]` and restricts hair relabeling to
     `ht_in_track | (tube_full & parts_bool)` so `Upper_Clothing`/`Torso`/`Hands` outside `head_track` and `parts_bool`
     are never relabeled as `Hair`.
  3. `bench/capture_masked_run.py::grade_holds` & `_files`: passes `head_track=ht_for_grade` (`ht_res["tube"]` when
     `head_tube` ran) and exempts frames where `int((out[f] & head_track[f]).sum()) >= SEGMENT_PX` (`64 px`) from being
     emptied as fragments under `least_px` (`1,024 px`) or `COLLAPSE_SHARE`.
  4. `bench/capture_masked_run.py::hair_in_hole_files`: rewrites `part_grades.json` on disk (Write #2) and syncs
     `sighting["parts_doubted_and_left_on_source_frames"]` (`8a420f89`: `frame_spans(doubted_left, join=1)`) and
     `sighting["parts_emptied_by_grade_on_source_frames"]` after `her_hair_in_hole`.
  5. `bench/shot_to_load.py::make_depth`: unions `mine_subj["hair_hers"]` frames into `ov_frames` when measuring MoGe-2 depth.
- **Still Open on `main`**:
  - `bench/shot_to_load.py::hold_to_track` (`_on_track_8px.mkv`) is still active on `main`.
  - `bench/capture_masked_run.py::her_hair_in_hole` still tests `hole(f)` against pre-carve/pre-feather `out[f]` and
    restricts Part 3 by `~track[f]`.
- **Both Ends (Writers & Readers)**:
  - **Writers**: `bench/shot_to_load.py::part_on_track`, `bench/shot_to_load.py::hold_to_track`,
    `bench/shot_to_load.py::make_part_on_track`, `bench/shot_to_load.py::make_depth`;
    `bench/head_tube_parts.py::_head_seed_from_person_mask`, `bench/head_tube_parts.py::sam31_head_tube`,
    `bench/head_tube_parts.py::head_tube_for_capture`;
    `bench/capture_masked_run.py::grade_holds`, `bench/capture_masked_run.py::her_hair_in_hole`,
    `bench/capture_masked_run.py::hair_in_hole_files`, `bench/capture_masked_run.py::_files`.
  - **Readers**: `bench/load_chain.py::mask_specs`, `bench/capture_masked_run.py::_files`,
    `bench/inspect_mask_steps.py::inspect_capture`, `bench/flags_before_render.py::check_flags_read`,
    `bench/frame_table.py::build_frame_table`.

### Part 5f — Build 2 (`426638a6`): Unowned SAM 3.1 `people.npz` Detections & `core[i]` in `bench/strangers.py`

- **Landed on `main` (`426638a6`)**:
  - In `bench/strangers.py::run`, computes `det |= m & (c == 0)` for every unclaimed SAM 3.1 detection (`who is None` in
    `others[i]`), setting `people[i] = (c > 0) | det` and `core[i] = np.isin(c, core_ids) | det` so `handed_back`
    preserves SAM-backed strangers (`held = piece & core[i]` is non-empty) instead of dropping them as `NO_CORE`
    (`4686-4693` on `4647`, `1458`). Updated `OWN_FORM` and `bench/strangers.py::held_to_its_record`.
- **Both Ends (Writers & Readers)**:
  - **Writers**: `bench/strangers.py::others_of`, `bench/strangers.py::held_others`, `bench/strangers.py::handed_back`,
    `bench/strangers.py::run`.
  - **Readers**: `bench/shot_to_load.py::others_of_load`, `bench/shot_to_load.py::run_strangers`,
    `bench/capture_masked_run.py::_depth_medians_and_owner`, `bench/capture_masked_run.py::her_hair_in_hole`,
    `bench/inspect_mask_steps.py::inspect_capture`.

### Part 5g — Build 3 (`6b0ea9a6`): Tracker Default `HEAD_PER_STRETCH` & SAM `--people` Verification on `cast_file.py box`

- **Landed on `main` (`6b0ea9a6`)**:
  1. `subject_track.py::parse_head_mode`, `subject_track.py::follow`, and `bench/shot_to_load.py::preview_job`: flipped
     `head_mode` default to `"per stretch"` (`HEAD_PER_STRETCH`, `head_per_stretch=True`); whole-span mode requires
     explicit `head: span: WHY` (`HEAD_SPAN = "span"`) or `--head-span WHY`.
  2. `bench/cast_file.py::drawn`: requires `--people <people.npz>` on `box`, verifies `IoU >= 0.500` against SAM 3.1
     detections on that frame, snaps the stored box to the best SAM detection box, and clears `drawn=None` so
     `subject_track.py::parse_seeds` treats it as a verified detection seed rather than `APPROXIMATE`.
- **Both Ends (Writers & Readers)**:
  - **Writers**: `subject_track.py::parse_head_mode`, `subject_track.py::_track_head_span`, `subject_track.py::_span_head`,
    `subject_track.py::follow`, `subject_track.py::auto_match`, `subject_track.py::name_drawn`, `subject_track.py::parse_seeds`;
    `bench/shot_to_load.py::preview_job`, `bench/shot_to_load.py::step_reference`; `bench/cast_file.py::read`,
    `bench/cast_file.py::drawn`, `bench/cast_file.py::seeds`.
  - **Readers**: `bench/head_tube_parts.py::sam31_head_tube`, `bench/capture_masked_run.py::_files`,
    `bench/what_h3_got.py::audit_capture`.
