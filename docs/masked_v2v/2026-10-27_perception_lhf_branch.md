# Perception, Stage-Split & Modularization Branch (`gem/perception-lhf`, 2026-10-27)

All 13 commits of the 2026-10-27 perception quick-wins (`LHF-0..LHF-6`), strategic multipliers (`SM-1..SM-3`),
modularization (`Mod-1..Mod-4`), and opportunistic helpers (`OD-1..OD-3`) are built, unit-tested, and ready to be
tested on real capture data on an isolated branch and worktree without touching `main`'s active render pipeline.

- **Branch name**: `gem/perception-lhf`
- **Worktree path**: `../../h3_worktrees/gem-perception-lhf` (`<comfy>/h3_worktrees/gem-perception-lhf`)
- **Commit range**: `920c38cd..27838b55` (13 commits cut from `996b5368`, `57` files changed)
- **Verification state on branch**:
  - All 5 new check suites (`check_wevis_seed.py`, `check_perception_contracts.py`, `check_capture_stages.py`,
    `check_hyden_da2.py`, `check_sam_audio_speaker.py`) plus updated existing checks (`check_strangers.py`,
    `check_head_tube_parts.py`, `check_subject_track.py`) pass on CPU and report `SEEN` under `bench/break_checks.py`.
  - `bench/masked_code_map.py --check`, `bench/check_doc_inventory.py`, `bench/check_imports.py`,
    `bench/check_no_owner_paths.py`, `bench/check_node_ids.py`, and `bench/check_schema_defaults.py` all pass on the
    branch.

---

## 1. What Exists on `gem/perception-lhf` (By Commit)

### Quick Wins (`LHF-0..LHF-6`)

1. **`LHF-0` (`920c38cd`) — Worktree `COMFY` Root Resolution**:
   - `bench/_lib/__init__.py` and `workflows/h3_config.py` resolve `<comfy>` via `H3_COMFY_ROOT` or by walking ancestor
     directories for `comfy/cli_args.py` before falling back to `parents[2]`, so scripts and custom nodes import `comfy`
     cleanly from `../../h3_worktrees/<branch>` even when `custom_nodes/ComfyUI-h3-explorations` is a symlink.
2. **`LHF-1` (`8303fd3f`; also landed on `main` as `426638a6`) — Unowned SAM 3.1 Detections in `bench/strangers.py`**:
   - Unions unclaimed SAM 3.1 person detections (`who is None` in `<preview>_people.npz`) into `people[i]` and `core[i]`
     where whole-frame Sapiens-2B sees `c == 0`, preventing `handed_back` from dropping SAM-backed strangers as
     `NO_CORE`.
3. **`LHF-2` (`8417d6b6`; also landed on `main` as `6b0ea9a6`) — Per-Stretch Head Tracking & Cast Box Verification**:
   - Flips `subject_track.py::parse_head_mode` default to `"per stretch"` (`HEAD_PER_STRETCH`) and verifies drawn boxes
     in `bench/cast_file.py` against SAM 3.1 detections (`IoU >= 0.5`).
4. **`LHF-3a` & `LHF-3b` (`f717d23f`, `9bcdfcb3`, `a32741dc`) — Connected Head + Hair Preservation Across Regain & SAM Head Tracks**:
   - `bench/head_tube_parts.py::head_tube_for_capture` preserves connected Sapiens `parts_bool` head/hair components on
     active SAM `head_track` frames as well as regained frames (up to `MAX_HAIR_EXTENSION_RATIO`), rather than clipping
     long hair at the SAM `head_track` jaw/skull boundary.
   - `bench/shot_to_load.py` (`part_on_track`, `make_part_on_track`) and `sapiens2_parts.py`
     (`MiniMaxH3Sapiens2PartsOnTrack`) accept an optional `head_mask` (`head_track`) and union it with `person_track`
     before the `8px` dilation so outer hair and emerging heads are not clipped by `_on_track_8px.mkv`.
5. **`LHF-4` & `LHF-5` (`0af05732`) — Auto MoGe-2 Depth on Multi-Person Shots & Depth-Gated Hair Extension**:
   - `bench/shot_to_load.py` automatically runs MoGe-2 depth (`depth.npz`) when a shot has `>= 2` cast tracks or
     non-empty `strangers.mkv`.
   - `bench/head_tube_parts.py` and `bench/capture_masked_run.py` gate hair/head extension pixels by `depth.npz` where an
     overlapping stranger is closer to the camera (`stranger_z < subject_z - 0.10`), populating `depth_gated_frames` and
     `depth_gated_px`.
6. **`LHF-6a` & `LHF-6b` (`7b6eca71`, `a32741dc`, `a1725457`) — `pose_along_shot` Head-Track Union/Interpolation & MoGe-2 Shot FOV**:
   - `bench/pose_along_shot.py` (`pose_boxes` / `run`) unions `head_track` into the SAM 3D Body crop box and linearly
     interpolates across `<= 4`-frame occlusion gaps.
   - `body_pose.py` (`shot_fov_from_depth_npz`, `MiniMaxH3BodyPose`), `subject_boxes.py`, `workflows/masked_job_graph.py`,
     and `bench/shot_to_load.py` wire optional `depth_npz` median shot FOV into SAM 3D Body projection.

### Strategic Multipliers (`SM-1..SM-3`) & Modularization (`Mod-1..Mod-4`)

7. **`SM-1a` & `SM-1b` (`52796600`, `107664f4`) — WeVisOne-VL-4B Referring Grounder (`bench/wevis_seed.py`)**:
   - Standalone loader in `vendor/tencent_wevisone/wevisone.py` and CLI tool `bench/wevis_seed.py` (tested by
     `bench/check_wevis_seed.py`) for text-described subject grounding on keyframes.
   - Handles `<box>none</box>` abstention, snaps predicted `[0, 1000)` boxes to `<preview>_people.npz` SAM 3.1 detections
     (`IoU >= 0.50`), and supports `--with-masks` to decode `mask_decoder` (`1024 -> 256 -> 64`) sigmoid alpha maps.
   - **Wall-time & workflow fit**: Cold-loading the `8.2 GB` bf16 checkpoint takes `~4.5–6.0s` (`~8.6 GB` VRAM) plus
     `~0.18–0.25s/frame`, so it belongs in **Stage 0 cast discovery / post-cut re-ID** (`1–3` keyframes per dark shot),
     never in the per-frame Stage 1 loop.
8. **`Mod-1` (`9f2a70f7`) — Unified Sapiens-2B Class & Keypoint Contracts (`bench/_lib/perception_contracts.py`)**:
   - Centralizes all Sapiens-2B 28-class indices (`FACE_NECK`, `HAIR`, `CORE_CLASSES`, `ACCESSORY_CLASSES`, `PART_MAP`,
     etc.) and 308-keypoint ranges in `bench/_lib/perception_contracts.py` (tested by
     `bench/check_perception_contracts.py`), eliminating duplicated class tuples across `sapiens2_parts.py`,
     `bench/strangers.py`, `bench/head_tube_parts.py`, `bench/capture_masked_run.py`, `bench/head_reference.py`, and
     `bench/assemble_delivery.py`.
9. **`SM-2` & `Mod-2/3/4` (`8864c607`) — Non-Destructive Per-Stage Capture Mattes (`bench/_capture/stages.py`)**:
   - Adds `bench/_lib/capture_io.py`, `bench/_lib/window_spec.py`, and `bench/_capture/stages.py` (tested by
     `bench/check_capture_stages.py`).
   - Wires `bench/capture_masked_run.py` to write separate per-stage mattes (`parts_raw__<by>.mkv`,
     `parts_tube__<by>.mkv`, `parts_depth__<by>.mkv`, `parts_held__<by>.mkv`) and a per-frame provenance ledger
     `stage_chain__<by>.json` while preserving `parts_held__<by>.mkv` for downstream compatibility.
10. **`SM-3` (`9c5db878`) — Multi-Subject Logit-Margin Pixel Arbitration (`subject_track.py`)**:
    - Adds `arbitrate_multiplex_masks` (assigns contested pixels to the cast subject with higher SAM 3.1 logit margin
      and cuts frames where `> 50%` of a subject's area is lost to a higher-logit competitor) and `track_cast_multiplex`
      in `subject_track.py` (tested by `bench/check_subject_track.py`).

### Opportunistic / Deferred Helpers (`OD-1..OD-3`, `5bda6f3b`)

11. **`OD-1` — Soft Boundary Alpha for Accessories / Hair (`sapiens2_parts.py`, `bench/assemble_delivery.py`)**:
    - Adds optional `boundary_feather_px` on `sapiens2_parts.py` (`MiniMaxH3Sapiens2PartsOnTrack`) and preserves float
      `[0.0, 1.0]` alpha in `bench/assemble_delivery.py` when compositing soft mattes.
12. **`OD-2` — Hyden DA2 Depth Edge-Refinement Helper (`vendor/meta_moge2/hyden_da2.py`)**:
    - Sobel depth-discontinuity boundary refinement and `378x378` ViT-L/14 preprocessing helper (tested by
      `bench/check_hyden_da2.py`).
13. **`OD-3` — SAM-Audio Active-Speaker Attribution Helper (`bench/sam_audio_speaker.py`)**:
    - Per-frame `[0, 1]` vocal-energy envelope and mouth-motion correlation helper in
      `vendor/meta_sam_audio/sam_audio_tv.py` and `bench/sam_audio_speaker.py` (tested by
      `bench/check_sam_audio_speaker.py`).

---

## 2. Highest-Value / Lowest-Hanging Next Steps for Real-Data Testing

When ready to validate and cherry-pick from `gem/perception-lhf` onto `main`, run these in order:

1. **Rank #1 — Non-Destructive Real-Clip A/B Test on `fun` Window `4679-4703` (~15 mins, CPU-only, zero GPU risk)**:
   - Copy the existing capture inputs (`_people.npz`, `seg29.mkv`, `depth.npz`, `subject_track__*.mkv`) from `fun`'s
     window `4679-4703` into a scratch `/tmp` directory.
   - Run the branch's `bench/head_tube_parts.py`, `bench/strangers.py`, and `bench/capture_masked_run.py` from
     `../../h3_worktrees/gem-perception-lhf` against the scratch copy, then run
     `bench/inspect_mask_steps.py --strips /tmp/ab_4679.jpg` to compare before/after pixel counts on the classic failure
     frames (`4698` stranger self-swallow and `4693-4702` hair-extension vs. depth-gate trade-off).
2. **Rank #2 — Wire the Reader Side of `SM-2` (`stage_chain__<by>.json`) into `bench/inspect_mask_steps.py` (~20 mins, CPU-only)**:
   - `bench/capture_masked_run.py` on `gem/perception-lhf` already writes `parts_raw__<by>.mkv`, `parts_tube__<by>.mkv`,
     `parts_depth__<by>.mkv`, and `stage_chain__<by>.json`.
   - Updating `bench/inspect_mask_steps.py` (`inspect_capture`) to read `stage_chain__<by>.json` when present lets its
     `[0]`/`[2]`/`[5]`/`[7]` forensic report name the exact stage (`raw` -> `tube` -> `depth` -> `held`) that dropped a
     frame's pixels.
3. **Rank #3 — Wire `arbitrate_multiplex_masks` (`SM-3`) into `subject_track.py::_cut_on_others` & `WindowSpec.merge_reload` into `bench/capture_masked_run.py` (~30 mins, CPU-only)**:
   - Call `arbitrate_multiplex_masks` inside `subject_track.py::_cut_on_others` (or `bench/strangers.py::people_pass`)
     so multi-cast shots get pixel-level logit-margin carving in addition to whole-frame `IoU >= 0.5` crossover cuts.
   - Call `WindowSpec.merge_reload` inside `bench/capture_masked_run.py` when reloading an existing `manifest.json` by
     hand (`bench/shot_to_load.py::carry_words_of` already carries `head_tube` and `depth` across `--remake`).
4. **Rank #4 (Defer) — Full GPU Checkpoint Loaders for `OD-2` (`hyden_da2.py`) and `OD-3` (`sam_audio_speaker.py`)**:
   - Loading the `1.6 GB` ViT-L/14 and `14.8 GB` `sam-audio-large` checkpoints onto GPU is not needed for finishing
     `fun` (MoGe-2 `--resolution-level 9` + `LHF-5` already covers depth gating).
