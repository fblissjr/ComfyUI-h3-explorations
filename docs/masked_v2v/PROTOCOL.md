# Masked Video-to-Video Execution & Debugging Protocol

This protocol is the step-by-step operational manual for building, auditing, rendering, debugging, and landing masked video-to-video loads. Read [`docs/masked_v2v/AGENTS.md`](AGENTS.md) first, above all "The owner's rules for passes, prep and judging", which every step below follows, then [`data/AGENTS.md`](../../data/AGENTS.md); see [`docs/masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](2026-10-25_pass_contamination_and_mask_step_forensics.md), [`docs/wiki/masked_v2v_pipeline.md`](../wiki/masked_v2v_pipeline.md), [`docs/wiki/masked_v2v.md`](../wiki/masked_v2v.md), and [`docs/wiki/runbook.md`](../wiki/runbook.md) for the broader design and card procedures.

---

## Stage 0: Environment, Ports, & Pre-Flight Checks

[`docs/comfy_notes.md`](../comfy_notes.md) ("Running things") is the single authoritative home for environment setup, `<comfy>/start.sh`, `comfy` imports, `agy` subshell behavior, and build worktrees. Summary of what every session needs before its first command:

1. **Never run `uv` (`uv run`, `uv sync`, `uv add`)**:
   - Invoking `uv` with the ComfyUI venv active empties that venv and dirties `uv.lock` ([`AGENTS.md`](../../AGENTS.md), "Operative rules"; [`docs/comfy_notes.md`](../comfy_notes.md)).
   - Always invoke the ComfyUI venv Python directly (`$PY` = `<comfy>/.venv/bin/python`), prefix CPU-only tools and checks with `CUDA_VISIBLE_DEVICES=`, and call `bench/_lib/__init__.py::bootstrap` before importing `comfy`, `comfy_extras`, or `nodes` ([`docs/comfy_notes.md`](../comfy_notes.md), "Importing `comfy`, `comfy_extras`, and `nodes`").
2. **Environment keys (`H3_OUTPUT_DIR`, `H3_INPUT_DIR`, `H3_MODEL_STORE`, `H3_CAPTURE_ROOT`) and `agy` subshell non-persistence**:
   - Derived directly from `../../start.sh` and `../../models/sapiens2/` without hardcoding private paths ([`docs/comfy_notes.md`](../comfy_notes.md), "Where this machine keeps things"):
     ```bash
     COMFY="$(cd ../.. && pwd)"
     export PY="$COMFY/.venv/bin/python"
     export H3_OUTPUT_DIR="$(awk '/^[[:space:]]*--output-directory[[:space:]]/{print $2}' "$COMFY/start.sh")"
     export H3_INPUT_DIR="$(awk '/^[[:space:]]*--input-directory[[:space:]]/{print $2}' "$COMFY/start.sh")"
     export H3_MODEL_STORE="$(dirname "$(readlink -f "$COMFY"/models/sapiens2/* | head -1)")"
     export H3_CAPTURE_ROOT="$H3_MODEL_STORE/h3_captures"
     ```
   - In the Antigravity (`agy`) harness, each `run_command` call starts a fresh subshell (`cd` and `export` do not persist across calls): evaluate the block above in every `run_command` call that invokes a `bench/` tool.
   - ComfyUI nodes write through `--output-directory` and `--input-directory` passed by `<comfy>/start.sh`, whereas `bench/` tools read `H3_OUTPUT_DIR`, `H3_INPUT_DIR`, and `H3_MODEL_STORE` from the environment: never start bare `main.py` without `<comfy>/start.sh`.
3. **Inspecting the server process (`:8188`) and review server (`:8765`)**:
   - The server process is the resource, not the GPU ([`AGENTS.md`](../../AGENTS.md)). Before stopping or restarting ComfyUI on port `8188`, inspect the port owner and its environment:
     ```bash
     ss -ltnp | grep -E ':8188|:8765'
     tr '\0' '\n' < /proc/<pid>/environ | grep '^H3_'
     ```
   - If any arming key (`H3_CAPTURE`, `H3_SOL_OBSERVE`, `H3_SOL_PROBE`, `H3_TELEMETRY`) is set in `/proc/<pid>/environ`, ask before killing the server.
   - Never use `bench/restart_comfy.sh` (disabled; exits 2). Follow the manual recipe in [`docs/comfy_notes.md`](../comfy_notes.md) ("The recipe, by hand", using `<comfy>/start.sh`).
4. **Code stamps, clean node code, and build worktrees (`../../h3_worktrees/<branch>`)**:
   - Cut build worktrees two levels below `<comfy>` at `$(git rev-parse --show-toplevel)/../../h3_worktrees/<builder>-<item>`, never under the home folder and never inside `custom_nodes/`, and never run pipeline commands (`shot_to_load.py`, `masked_job.py`, `load_chain.py`, `rebuild_delivery.py`) from inside a worktree ([`docs/comfy_notes.md`](../comfy_notes.md), "Build worktrees").
   - Derived `.json` and `.npz` files carry `writer_stamp` (`bench/_lib/code_stamp.py::writer_stamp`, `bench/_lib/code_stamp.py::npz_writer_fields`) and check upstream mtimes via `bench/_lib/code_stamp.py::check_upstream_mtime` (`bench/_lib/code_stamp.py::MTIME_TOLERANCE_S` = `1.0` s).
   - [`bench/masked_job.py`](../../bench/masked_job.py) refuses to queue a graph beside uncommitted node code unless given `--override "WHO: WHY"`, which it writes into the record (`bench/masked_job.py::before_queue`; [`AGENTS.md`](AGENTS.md) §8, stage 0). `bench/load_chain.py render` also refuses, with no override, a render graph whose build record lists uncommitted node code (`bench/load_chain.py::render_against_plan`).
5. **When to restart ComfyUI (`:8188`), and what a change forces**: [`docs/masked_v2v/AGENTS.md`](AGENTS.md) §8 is the one
   copy. It gives each stage in trace order, its command, what it reads and writes, and what a change to it forces. The
   default remake is a fresh preview and every stage after it; a restart goes by the recipe in
   [`docs/comfy_notes.md`](../comfy_notes.md).

---

## Stage 1: Choosing `--source` (Single-Subject vs Multi-Subject Sequential Passes) & Running `shot_to_load.py preview`

### 1.1 Choosing `--source` (No Silent Default)

[`bench/shot_to_load.py`](../../bench/shot_to_load.py) `preview` (`bench/shot_to_load.py::step_preview`) requires `--source VIDEO` explicitly and has **no default**:

- **Which subject is pass 1**: the one already kept on the shot. A kept subject stays and is not remade to make the other, unless something broken changes the composite ([`AGENTS.md`](AGENTS.md) §1; the owner, 2026-10-21). No subject goes first by rule.
- **Single-subject shot, or pass 1 of a shot of two**:
  - Pass the 24.0 fps working-copy source video registered in `data/<video>/manifest.json` (for `fun`, `fun_0000_6394_24p_sound_in_step.mov`).
  - The source video must be an exact `24.0` fps every-frame working copy (`bench/clip_analysis.py` `init --copy`, sound stretched in step via [`bench/copy_sound.py`](../../bench/copy_sound.py)). `bench/shot_to_load.py::working_copy` refuses any source whose manifest rate is not `24`.
- **Pass 2 (where pass 1 is already laid in `data/<video>/laid.jsonl`, and the other subject has something to replace: a subject recorded `not seen` with `bench/shots_look.py` owes no load, [`AGENTS.md`](AGENTS.md) §1)**:
  - **NEVER** run `preview` or `job` against the raw unswapped source video, a stale cut older than another subject's laid row, or a source cut that already holds the load's own subject's piece (`bench/shot_to_load.py::check_pass2_source`, `_cut_pieces_on_frames`, `check_job_pass2_source`).
  - Use the newest cut of the whole video that holds pass 1 ([`bench/rebuild_delivery.py`](../../bench/rebuild_delivery.py) `--video <video> --clip`, producing `fun_..._from_laid.mp4` where pass 1's subject is already replaced) and register it in `<share>/input/` and `data/<video>/manifest.json`. A cut of one span (5.3) begins on that span's first frame and is not a source ([`AGENTS.md`](AGENTS.md) §1).
  - Run a **fresh** `bench/shot_to_load.py preview --source <post_pass1_video>` so SAM 3.1 and Sapiens segment pass 1's **actual replaced appearance** (new hair/head silhouette) and the other subject's track against the post-Pass-1 picture, then make pass 2's whole chain again on it. Only the owner's own words carry from the original video's preview: the cast boxes and `data/<video>/plan.json`.

### 1.2 Mandatory `--cast` Verification

Always verify `data/<video>/cast/cast.jsonl` exists on disk (and pass `--cast data/<video>/cast/cast.jsonl` to `bench/shot_to_load.py job`; `preview` defaults `--cast` to `data/<video>/cast/cast.jsonl` when omitted, `bench/shot_to_load.py::step_preview`):
- `bench/shot_to_load.py` checks `cast_path.is_file()` up front and raises `bench/shot_to_load.py::Refused` if the cast file is missing.
- How `bench/shot_to_load.py::seed_of` and `bench/shot_to_load.py::others_of_load` seed SAM 3.1 from `cast.jsonl`, and what counts as a valid tracker seed (Group 1 `/who/` page clicks `lines 1–174` and Group 3 session-verified exact SAM detection boxes `lines 197–209` vs. Group 2 owner visual cues in `plan.json` `head_marks` and `owner_sheets/`): see [`AGENTS.md`](AGENTS.md) §3 (the one home).

### 1.3 Running `preview`

```bash
<comfy-venv>/bin/python bench/shot_to_load.py preview \
  --shots data/<video>/shots.json \
  --plan data/<video>/plan.json \
  --cast data/<video>/cast/cast.jsonl \
  --shot <first_frame> \
  --folder <folder the preview's job file is written in> \
  --source <24p_working_copy_or_post_pass1_cut> \
  --for <subject> \
  --by <session> \
  --remake \
  --queue
```

`--shots`, `--plan`, `--shot`, `--folder` and `--by` are required (`bench/shot_to_load.py::main`); there is no `--video`
or `--run`. `--source` is required by the step, and `--for` (the replacement the preview is made for, repeatable) where
anything is laid on the shot ([`AGENTS.md`](AGENTS.md) §1). `--remake` takes the next free letter when a preview of the shot exists. Without `--queue` the job file is
written and the two commands that build and queue it are printed (`bench/masked_job.py build preview <job file> --date
<day>`, then `bench/masked_job.py queue data/<day>_shot_c_<first>[<letter>]_job preview --host <host> --wait`). A
preview that was built and never queued has no `preview.queued.json`, and `job` refuses it until it is queued.

What `preview` produces in `data/<day>_shot_c_<first>[<letter>]_job/` and on the share:
- One SAM 3.1 person tracker (`MiniMaxH3SubjectTrack` with `TRACK_HEAD = True` by default, producing both the person track and the SAM 3.1 head track), Sapiens2-Seg part and 29-class maps, Sapiens2-Pose keypoints (`pose.npz`), and SAM 3D Body mesh video (`body.mp4`) per subject listed in `shots.json` for that shot.

---

## Stage 2: Building the Head Tube, Parts Mask, & Reference

### 2.1 Head Tube & Part Masks (`head_tube_parts.py`)

[`bench/head_tube_parts.py`](../../bench/head_tube_parts.py) (`bench/head_tube_parts.py::sam31_head_tube` and `bench/head_tube_parts.py::head_tube_for_capture`) combines the SAM 3.1 head track, Sapiens2-Seg class maps, and Sapiens2-Pose facial landmarks:

- **Unified head track (`TRACK_HEAD = True`)**: SAM 3.1 tracks `"head"` directly, preventing Sapiens misclassifications (such as labeling a cheek or forehead as `Upper_Clothing` or `Background` during turns) from punching holes inside the head.
- **Attached profile rim expansion**: `bench/head_tube_parts.py::head_tube_for_capture` expands `head_track_bool` into attached part components (`parts_bool & track_bool`) within the head's vertical span `[hy0, hy1]` so side-profile face rims are not clipped.
- **A lost head is re-seeded only on the subject's own face points**: where the head track loses her head and her person track still holds her, `bench/head_tube_parts.py::sam31_head_tube` re-seeds a head from the top of the person track only when her face points over the line lie on her own replaced classes (`bench/head_tube_parts.py::own_part_face_points`, at least `bench/head_tube_parts.py::REGAIN_FACE_POINTS_MIN`). Otherwise her head is hidden there and the frame stays empty, listed with its count under `regain_refused` in the head tube's provenance. On `second_hfp_4912e` the top of her person track was her raised hand, and the re-seed had made it her head. `[2026-10-26: in the capture where tracker=None, _head_seed_from_person_mask's flat horizontal head_rel_y1 slice cut 945–1,354 px off her hair on 4686 and 4693–4695; Build 1 in docs/wiki/handover.md retires hold_to_track and that flat slice in favor of Sapiens face+hair on her crop unioned with her head track in a room anchored to her head.]`
- **`Hair` only for a subject who replaces hair**: a hair-and-face part takes every non-face tube pixel as `Hair`; a part without `Hair` (a face-only subject) keeps the part model's own classes, and `bench/head_tube_parts.py::_fill_unclassified_in_head_tube` writes only the face. What the part replaces is the classes the preview cut from the plan's parts, read back by `bench/capture_masked_run.py::part_classes`.
- **Bridging short class-map dips (`bridge_short_tube_dips`)**:
  - During rapid head turns or motion blur, Sapiens `Face_Neck` can dip below 60% of its flanking anchor area for a few frames even while the person track remains continuous.
  - `bench/head_tube_parts.py::bridge_short_tube_dips` (`bench/head_tube_parts.py::SHORT_DIP_MAX_FRAMES` = `6`, `bench/head_tube_parts.py::SHORT_DIP_RATIO` = `0.6`) automatically bridges any contiguous dip of `<= 6` frames (`0.25` s at `24` fps) where `tube_px < 0.6 * min(pre_px, post_px)` by taking the union of the flanking anchor frames' head tubes clipped to the current frame's dilated person track, recording bridged spans under `bridged_short_dips` in provenance.

### 2.2 Building the Load Job (`shot_to_load.py job`)

```bash
<comfy-venv>/bin/python bench/shot_to_load.py job \
  --shots data/<video>/shots.json \
  --plan data/<video>/plan.json \
  --cast data/<video>/cast/cast.jsonl \
  --shot <first_frame> \
  --folder <folder the load's job file is written in> \
  --preview data/<day>_shot_c_<first>[<letter>]_job \
  --subject <subject> \
  --by <session> \
  --remake --no-carry-words "<why>"
```

- `--shots`, `--plan`, `--shot`, `--folder`, `--subject`, `--preview` and `--by` are required; there is no `--video`.
  `--preview` is the preview's job FOLDER, which must hold `job.json` and a `preview.queued.json` that ended in
  success (`bench/shot_to_load.py::step_job`), never its `job.json`. `--remake` with an earlier load of the shot and
  subject takes `--no-carry-words "<why>"` or `--carry-words-from <old job file>`. Run it with `--dry` first: it
  prints the load's name and its defaults and writes nothing.
- Prints and records `THE DEFAULTS THIS LOAD CARRIES` (`bench/shot_to_load.py::defaults_carried`).
- **Whole-shot windows (`<= 345` frames), `[shot.first .. shot.last]` input locking, and `h3_rules.on_both_clocks` context/overlap padding**:
  - [`AGENTS.md`](AGENTS.md) §6 is the single authoritative home for the 24.0 fps end-to-end rule, one shot = one render (`workflows/masked_job_graph.py::window_for`, `h3_rules.on_both_clocks`, `h3_rules.chain_lengths`), interior occlusions (`capture_masked_run.py verify --no-mask`, `left_as_the_source_with_no_mask`, `head_marks` `hidden` in [`AGENTS.md`](AGENTS.md) §10), how `preview`, `strangers.py`, `reference`, `text`, and `capture_masked_run.py verify` lock every input to `[shot.first .. shot.last]`, and the owner's 2026-10-25 three-path ruling on window padding (`workflows/masked_job_graph.py::window_pad`, `held_context`, `video_mask.hold_context`, `video_mask.py::held_edge_falls_back`, `video_mask.py::zoomed_source`, `video_mask.py::zoom_box`, and `audio_freeze_song.py::MiniMaxH3AudioFreezeSong.execute`).
- **Remake span words and settled `plan.json` recipe**:
  - When remaking an existing load (`--remake`), pass `--carry-words-from <old_job.json>` or `--no-carry-words "<why>"` so span words (full list in [`AGENTS.md`](AGENTS.md) §8, "What only a person writes") are never dropped by accident. A seed is never carried, and settled `margin` and `zoom_up_to` are read from `data/<video>/plan.json` (`recipe` and `shot_overrides`, enforced in code by `cbd39fbe` with no `--*-why` CLI bypass).
- **Span flags (`--no-face-on`, `--no-draw-face-on`, `--no-head-on`, `--hold-part`) vs. never `--drop` on visible faces**:
  - Never `--drop` interior frames where the original actor's face or beard is visible; use `--no-face-on A-B=WHY` (`bench/capture_masked_run.py::grade_holds`), `--no-draw-face-on A-B=WHY`, or `--no-head-on A-B=WHY [--no-head-form hold|blank]` (`bench/head_reference.py`) instead. Full rules in [`AGENTS.md`](AGENTS.md) §2 and §8.
- **No stale pose or mask data**: `job` chooses the load's reference folder (`bench/shot_to_load.py::reference_folder`), and `bench/_lib/code_stamp.py::check_upstream_mtime` only warns. What each stage reads, what a change forces, and what no tool enforces: [`AGENTS.md`](AGENTS.md) §8.

### 2.3 Cropping & Drawing the Head Reference (`shot_to_load.py reference`)

```bash
<comfy-venv>/bin/python bench/shot_to_load.py reference <load job file> --by <session>
<comfy-venv>/bin/python bench/shot_to_load.py text <load job file> --by <session> --plan data/<video>/plan.json
```

The job file is positional for both. `reference` reads the pose model's store from `H3_MODEL_STORE` (or
`--model-store`); `text` refuses without `--plan`.

[`AGENTS.md`](AGENTS.md) §7 and §8 (stages 3–4) are the single authoritative home for the reference crop, mouth/eye drawing rules, and `<Subject 2>` / `<Subject 1>` text rules. Summary of what the two commands invoke:

1. **Head-box crop (`bench/crop_reference_to_head.py`)**:
   - `step_reference` invokes [`bench/head_reference.py`](../../bench/head_reference.py) → [`bench/crop_reference_to_head.py`](../../bench/crop_reference_to_head.py) (`bench/crop_reference_to_head.py::one_box`), sizing one constant box across the load according to `plan.json` `reference.crop` (`head`, `face`, or `cranium` in `bench/head_reference.py::CROP_FORMS`, `CROWN_ROOM`, `HEAD_TRACK_ROOM`, `HEAD_ROOM`, `CRANIUM_ROOM`, `CRANIUM_NECK`, `CRANIUM_MOST`; every subject of `fun` uses `cranium`). Cuts `<stem>_plain_cut.mp4` and `<stem>_source_cut.mp4` with `writer_stamp` ([`AGENTS.md`](AGENTS.md) §7).
2. **Facial landmark drawing (`bench/draw_face_points.py`)**:
   - Runs [`bench/draw_face_points.py`](../../bench/draw_face_points.py) (`bench/shot_to_load.py::REFERENCE_DEFAULT`: `iris="contrast"`, `mouth="auto"`, `closed_lip_line=True`, `--darken-head` off), clipping marks to the SAM 3D Body head silhouette.
   - Rules for `sculpted` vs `outlined` closed mouths (`bench/draw_face_points.py::CLOSED_MOUTH_FORMS`, `INNER_GAP_CLOSED_PX`, `MOUTH_BEND`), `inner outline` vs `outlined` open mouths (`OPEN_MOUTH_FORMS`, `fit_mouth_to_head`), retired `smooth_mouth_area`, and downcast/shut eyes (`DOWNCAST_RING_OUTSIDE_MIN`, `dark_irises`, `EYE_OPEN`, `--eyes-shut-on` / `EYES_SHUT_FORMS`): see [`AGENTS.md`](AGENTS.md) §7.
3. **Generating the Prompt Text (`shot_to_load.py text` — `bench/load_text.py`)**:
   - Invokes [`bench/load_text.py`](../../bench/load_text.py) (`NO_SCENE`, `NO_CLOTHING`, `subject_2`, `FACE_EYES`, `MOUTH_CLAUSE`, `mouth_drawn_open`, `mouth_drawn_closed`, `mouth_drawn`, `TRANSFERRED_DRAWN`, `FOLLOWS_DRAWN`, `TRANSFERRED_DRAWN_NO_MOUTH`, `FOLLOWS_DRAWN_NO_MOUTH`) to generate `<load>.txt` from the reference record, `data/<video>/plan.json` (`text.picture_hair`, `text.hair_words`), and the voice table. Full prompt rules in [`AGENTS.md`](AGENTS.md) §7 and `bench/load_text.py`'s docstring.

---

## Stage 3: Running `load_chain.py` & Pre-Render Telemetry (`facts_pass.py`, `step_carves.json`)

### 3.1 `first`, `plan`, `on-plan`, and `table`

Run [`bench/load_chain.py`](../../bench/load_chain.py) steps in strict order from the load's job file:

```bash
<comfy-venv>/bin/python bench/load_chain.py first   data/<day>_<load>_job/<load>.json --by <session>
<comfy-venv>/bin/python bench/load_chain.py plan    data/<day>_<load>_job/<load>.json --by <session>
<comfy-venv>/bin/python bench/load_chain.py on-plan data/<day>_<load>_job/<load>.json --by <session>
<comfy-venv>/bin/python bench/load_chain.py table   data/<day>_<load>_job/<load>.json --by <session>
```

- `first`: builds the initial capture at `data/<day>_<load>/` via [`bench/capture_masked_run.py`](../../bench/capture_masked_run.py), runs `verify`, evaluates the preflight gate (`flags.json`), and renders the strip under `<capture>_first`.
- `plan`: builds and queues the no-sampling plan graph (`workflows/masked_job_graph.py`), which writes `<load>_plan.json` and `<load>_window_N_planned_region.npz`.
- `on-plan`: copies the initial capture to `<capture>_first`, rebuilds `data/<day>_<load>/` in place on the node's actual planned regions, runs `verify` and `preflight --gate`, writes `runs/<run>/step_carves.json` (`bench/capture_masked_run.py::step_keepout_carves`), draws the plan strip under `Video/<video>/looks/<day>_<load>/`, and writes `frame_table.jsonl` ([`bench/frame_table.py`](../../bench/frame_table.py)).
- Blocking preflight flags in `flags.json` include `dropped_frame_has_visible_face` (`bench/capture_masked_run.py::dropped_or_unmasked_face_in_context`, triggered when any frame with `region_px == 0` inside a tracked span has `s_face_px >= 4000`), `reference_shows_no_figure` (the subject tracked or carried on a run of `bench/capture_masked_run.py::FIGURE_RUN_LEAST` frames where `<Video 1>` is black; no override clears it; [`AGENTS.md`](AGENTS.md) §8, stage 3), `her_head_lost_in_view` and `unclaimed_hair_touches_another` ([`AGENTS.md`](AGENTS.md) §10).

### 3.2 Running `facts_pass.py` and Auditing `face_outside_region_px` & `step_carves.json`

Before requesting a look or queueing a render, run [`bench/facts_pass.py`](../../bench/facts_pass.py) on the on-plan capture directory:

```bash
CUDA_VISIBLE_DEVICES= <comfy-venv>/bin/python bench/facts_pass.py \
  data/<day>_<load> \
  --job <load job file> \
  --output-json data/<day>_<load>/facts_pass.json \
  --output-md data/<day>_<load>/facts_pass.md
```

With neither `--output-json` nor `--output-md` it prints the Markdown and writes nothing.

Inspect three critical signals before spending GPU render minutes:
1. **`face_outside_region_px`** (`bench/facts_pass.py`):
   - Counts pixels where `(s_face | (head_track & ~s_hair)) & ~region` is non-zero on each frame.
   - Any frame with `face_outside_region_px > 0` means part of the subject's face or head track lies outside the 16-px token region that H3 will regenerate — guaranteeing a chin, cheek, or profile cutout where the original video shows through.
2. **`runs/<run>/step_carves.json` and `other_overlap_px`** (`bench/capture_masked_run.py::step_keepout_carves`):
   - Reports per-window, per-latent-step (`video_mask.py::run_lengths`) grown token units, final token units, total carved units, units carved by each other subject or keep-out across the 4-frame latent step, and **carved units touching the subject's `head_track` or carried `part`**.
   - Can also be inspected on specific frames with:
     ```bash
     CUDA_VISIBLE_DEVICES= <comfy-venv>/bin/python bench/capture_masked_run.py diagnose \
       data/<day>_<load> --frames <start>-<end>
     ```
3. **`mouth_mode` and `parts_held_px` dips**:
   - Confirm `mouth_mode` matches the window's vocal activity (`no_mouth` on silent windows, drawn mouth on singing windows) and check `notable_spans` for any `parts_held_drop_below_half_rolling_12f_median`.
4. **End-to-end mask-step & provenance forensics (`bench/inspect_mask_steps.py`)**:
   ```bash
   CUDA_VISIBLE_DEVICES= H3_INPUT_DIR=<share>/input H3_OUTPUT_DIR=<share>/output \
     <comfy-venv>/bin/python bench/inspect_mask_steps.py \
     --capture data/<day>_<load> --problems-only --strips /tmp/<load>_strips.jpg
   ```
   - Audits `[0]`–`[7]` in ~5 seconds on CPU; see [`AGENTS.md`](AGENTS.md) §11 and [`docs/masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](2026-10-25_pass_contamination_and_mask_step_forensics.md) for the full description of each check.

### 3.3 The Look, Flags Read (`flags_before_render.py`), Full Prompt Read (`what_h3_got.py`), & `render`

1. **The look the tool requires**: `bench/capture_masked_run.py looked`, signed by the lead on the owner's word with "not inspected" in its note; a plan strip may be sanity-checked. (The landed render is read before `page` by a reader who did not make it: 5.1 below; the owner's word of 2026-10-28 in `decisions.md` reversed the 2026-10-22 "no session looks at the render before the owner does".) Or the maker runs `bench/load_chain.py no-look <load job file> --by <session> --not-first WORDS.txt` where the gate is clear and no forcing rule in `bench/load_chain.py::FORCING_FLAGS` or `bench/load_chain.py::FORCING_FIGURES` stands; `--not-first` is a file with the maker's statement that the load is not the first of its kind, and which earlier load it follows).
2. **Flags read before render (`bench/flags_before_render.py`)**:
   ```bash
   CUDA_VISIBLE_DEVICES= <comfy-venv>/bin/python bench/flags_before_render.py \
     data/<day>_<load> --load <load> --initial <session>
   ```
   Prints and records the five diagnostic blocks (`[0]` recipe gate against `data/<video>/plan.json` (`recipe` and `shot_overrides`, landed in `cbd39fbe` on `2026-10-26` after `third_fp_5114d` ran at margin `64` against `16` via a written `why`: both `bench/shot_to_load.py job` and `bench/flags_before_render.py::recipe_check_of` now hard-refuse any differing `margin` or `zoom_up_to` value with no `--*-why` bypass; [`docs/wiki/decisions.md`](../wiki/decisions.md) `2026-10-26`), `[1]` `flags.json`, `[2]` `marks_against_the_body_model` + `face_on_the_track` + `mouthless_reference`, `[3]` `runs/<run>/step_carves.json`, and `[4]` held/emptied frames + `unlaid_while_tracked` + `unresolved_hair_in_hole`) to `<capture>/looks/<load>.flags_read.json` (`bench/load_chain.py::the_flags_read` refuses `render` without it).
3. **Full prompt read (`bench/what_h3_got.py`)**:
   ```bash
   CUDA_VISIBLE_DEVICES= <comfy-venv>/bin/python bench/what_h3_got.py \
     --job-folder data/<day>_<load>_job --load <load> --stage plan
   ```
   Read the entire prompt text in full from the queued graph ([`AGENTS.md`](../../AGENTS.md)).
4. **Queue the render (`load_chain.py render`)**:
   ```bash
   <comfy-venv>/bin/python bench/load_chain.py render \
     data/<day>_<load>_job/<load>.json --by <session> \
     --grounds GROUNDS.txt --show SHOW.txt
   ```

---

## Stage 4: Multi-Subject Crossing & Keep-Out Auditing

When two subjects cross paths in the same shot (e.g., `lead` walking behind `second` on Kitchen `604-1009`), keep-out rules operate on 4-frame latent steps (`16x16` pixel cells) and can carve boxy holes into the back subject's face if misconfigured:

1. **Why latent-step keep-outs carve 4 frames at once**:
   - In `video_mask.py`, token regions are quantized to latent steps (`1` frame on step `0`, `4` frames per step thereafter) and `16x16` pixel cells.
   - A subject's own part mask wins every cell it touches on any frame of the latent step (`video_mask.py`), so another subject's keep-out only carves **margin cells** or cells where the subject's own part mask dropped out across all frames of that step.
2. **The rule for Pass 2 on a crossing shot: segment it on the post-Pass-1 video.** Run `shot_to_load.py preview --source <post_pass1_video>` where pass 1 (the subject already kept, [`AGENTS.md`](AGENTS.md) §1) is laid at 24.0 fps. Pass 1's subject is then kept out by its own track on that preview, as another subject is on any shot (`bench/load_chain.py::mask_specs`), so the keep-out follows the replaced subject's silhouette. Never segment Pass 2 on the raw source video, or Pass 1's keep-out will follow the raw actor's old hair/head silhouette instead. Where pass 1's replacement covers less than the original did (on the kitchen, the original's taller hair), what is left of the original shows in the cut, and pass 2's region may need to cover it.
3. **Auditing before rendering**:
   - Check `runs/<run>/step_carves.json` and `facts_pass.json` (`face_outside_region_px` and `other_overlap_px`) on the crossing frames (`755-774`). Verify that `face_outside_region_px == 0` across the entire crossing before queueing `load_chain.py render`.

---

## Stage 5: Review Clips, `items.json`, `TO_REVIEW.html`, `owner_answers.jsonl`, and Landing a Kept Load (`laid.jsonl`)

### 5.1 Landing a Finished Render (`load_chain.py land` & `page`)

When the ComfyUI queue finishes:

```bash
<comfy-venv>/bin/python bench/load_chain.py land data/<day>_<load>_job/<load>.json --by <session>
<comfy-venv>/bin/python bench/load_chain.py page data/<day>_<load>_job/<load>.json --by <session> --spec SPEC.json
```

- `land` frees the GPU (`POST /free`), runs the Sapiens pose pass on the rendered video (`POSE_PASS_FAILED` if incomplete), records `/history` cost lines, and builds the frame-numbered stacked review clip (`Video/<video>/review_clips/<load>_<first>_<last>_stacked.mp4`), whose record goes under `<capture>/strips/`. `land` writes no sidecar: [`bench/sidecar.py`](../../bench/sidecar.py) is called by the assembler, `bench/stack_delivery.py` and `bench/recomposite_piece.py`.
- **Between `land` and `page`, the reading** (the owner's word of 2026-10-28 and 2026-10-29, [`docs/wiki/decisions.md`](../wiki/decisions.md); never `land && page` in one command): a reader who did not make the load, a session or a subagent, reads the landed `.1fps.<first>-<last>.mp4` strip and the `24 fps` clip (the native path in `.gemini/skills/video-understanding/SKILL.md`, "Who reads a landed render"), the reference `_sheet.jpg`, and the measured shift of the replaced part inside `parts_held` against the source frames and the kept loads (per-frame luma and `B-R`). It writes a short reading beside the capture (`<capture>/looks/<load>.reading.md`): what it sees with frame numbers, what it cannot see, the figures, and no verdict and no hold. Leave reader files as the reader wrote them; add a dated note beside a slip rather than editing the reading (the owner, 2026-10-29). The lead reads the reading, not the video, and the reading's words go on the review card beside the clip (`working_with_the_owner.md`, "A reader's take sits beside each review clip"). The owner's eye on playback accepts the piece.
- `page` prepends the candidate clip with `state: "open"` to `data/<video>/review/items.json` and creates the media symlinks under `data/<video>/review/media/`. Question cards on a shot (`kind: "look"`, built by `bench/load_chain.py::look_page_item` and rendered by `bench/review_pages.py`) also live in `data/<video>/review/items.json` with a still or short strip, the owner's chronological prior words on the shot from [`bench/said_on_shot.py`](../../bench/said_on_shot.py) (`--for-question`), and custom choice buttons.

### 5.2 Updating Both the Interactive Review Queue (`items.json`) and Static `TO_REVIEW.html`

The owner uses two views served by `data/<video>/poc/serve.py` on port `8765`:
1. **Interactive Review Page (`http://<host>:8765/`, served from `data/<video>/poc/index.html`)**:
   - Dynamically fetches `data/<video>/review/items.json` (`poc/review/items.json`) and `data/<video>/review/start_here.json`.
   - Renders items with `"state": "open"` under `"1. Look at these next"` with interactive verdict buttons (`keep`, `remake`, etc.) and a frame-note box.
   - Clicking a verdict button sends `POST /api/answer` to `serve.py`, which appends a timestamped line to `data/<video>/owner_answers.jsonl` (append-only, never hand-edited).
   - A playback reader writes `data/<day>_<load>/playback_reading_<reader>.md` only after the owner has answered the clip; no session looks at a render or reads it before the owner (decisions.md, 2026-10-22). Where one is referenced via `reader_file` in `data/<video>/review/start_here.json`, [`bench/review_pages.py`](../../bench/review_pages.py) still renders the 4-field reader card (`Sees`, `Cannot see`, `Verdict`, `Why`) folded closed.
2. **Static `TO_REVIEW.html` and `PROVENANCE.html` (`Video/<video>/TO_REVIEW.html` and `data/<video>/poc/TO_REVIEW.html`)**:
   - Built by [`bench/rebuild_delivery.py`](../../bench/rebuild_delivery.py) (`--video <video>` or `--video <video> --clip`), which runs:
     1. `laid_verify` ([`bench/laid.py`](../../bench/laid.py) `verify`)
     2. `ledger` ([`bench/render_ledger.py`](../../bench/render_ledger.py))
     3. `delivery_table` ([`bench/delivery_table.py`](../../bench/delivery_table.py))
     4. `assemble` (when `--clip` is passed: [`bench/assemble_delivery.py`](../../bench/assemble_delivery.py))
     5. `gap_check` ([`bench/frame_coverage.py`](../../bench/frame_coverage.py))
     6. `stack` (when `--clip` is passed: [`bench/stack_delivery.py`](../../bench/stack_delivery.py))
     7. `pages` ([`bench/covered_items.py`](../../bench/covered_items.py) and [`bench/review_pages.py`](../../bench/review_pages.py) `to-review` and `provenance`, plus `bench/rebuild_delivery.py::verify_poc_on_disk`)
     8. `snapshot` (`bench/rebuild_delivery.py::snapshot_owner_answers` into `data/<video>/owner_answers_snapshots/`)

### 5.3 Recording Verdicts & Laying a Kept Piece (`laid.jsonl`)

- Translate new lines in `data/<video>/owner_answers.jsonl` into `data/<video>/verdicts/verdicts.jsonl` with [`bench/page_answers_to_verdicts.py`](../../bench/page_answers_to_verdicts.py) (and inspect uncited answers with [`bench/page_answers_ledger.py`](../../bench/page_answers_ledger.py)).
- **Laying a piece is a person's act** ([`bench/laid.py`](../../bench/laid.py); its docstring has every flag). `lay` and `replace` write a line, `remove` takes one out with its reason (kept in `laid.removed.jsonl` beside the list), and `set-owner` changes a line's owner state in place. `--owner` is required on `lay`, `replace` and `set-owner`, and `kept` or `kept in a joined clip` takes `--owner-word FILE:LINE`, the owner's own line. Save the piece's own matte if needed (`bench/recomposite_piece.py --save-own-matte`).
- **One rebuild a batch.** Every `laid.py` write runs `bench/rebuild_delivery.py --video <video>` without `--clip`, and that run refuses once the list has changed since the clip was assembled (`bench/rebuild_delivery.py::verify_reused_clip`). Pass `--no-rebuild WHY` on each write of a batch, then run one `bench/rebuild_delivery.py --video <video> --clip`. The full rebuild runs only when the cut and the pages must move, after landings or removals (the owner, 2026-10-21).
- **A look at one shot** assembles and stacks that span alone, or uses the load's own stacked clip from `land` where nobody else is laid on the shot (the owner, 2026-10-21). It never takes a whole-video build. The current table's JSON (how to tell which is current: [`data/AGENTS.md`](../../data/AGENTS.md), "Where the lane's state lives") records its own assembly and stack commands (`assembly.command`, `assembly.stack.build`); take them with the shot's span:
  ```bash
  # the table's assembly.command, with --span set to the shot and an --out of its own
  <comfy-venv>/bin/python bench/assemble_delivery.py --source <working copy> --table <table .txt> \
    --span FIRST-LAST --out <span clip>.mp4 <the rest of the table's assembly.command: --size, --shots, every --capture>
  # stack it: --first is the source frame the span clip begins on
  <comfy-venv>/bin/python bench/stack_delivery.py build --source <working copy> --table <table .txt> \
    --joined <span clip>.mp4 --first FIRST --only FIRST-LAST --out <stacked>.mp4 --work <scratch folder>
  ```

---

## Appendix: Case Studies from `fun`

### Case Study 1: Kitchen `604-1009` Bouffant Hair Dome & Crossing Cutout (2026-10-19)

- **Symptom**: On Kitchen `604-1009` (`lead` walking behind `second` on frames `755-774`), `lead`'s face had a dark helmet-like cutout and chin smudge during the crossing, even after `second_hfp_0604h` (blonde mom) had been laid into the cut.
- **Root Cause**:
  1. The original 1980s actress playing `second` had a tall dark bouffant hairdo rising ~60 px higher (`ymin=188..209`) than the replaced blonde mom in `second_hfp_0604h` (`ymin=211..221`).
  2. Pass 2 (`lead`) reused a preview segmented on the **raw 1980s source video** rather than the post-Pass-1 video where `second_hfp_0604h` was already laid. That gave `second`'s keep-out `5,843–9,276 px` of non-existent bouffant hair dome on frames `755-774`, which carved an invisible helmet out of `lead`'s head as she walked behind `second`.
  3. In addition, on frames `759-761`, Sapiens2-Seg mislabeled `lead`'s cheek and forehead as `Upper_Clothing` (`13,948–16,899 px`) and `Background` (`20,133 px`), leaving only a `2,037 px` crescent of `s_face`. Because a keep-out carves any 4-frame latent cell where the subject's own mask has 0 pixels, `second`'s keep-out carved whole 16x16 squares out of `lead`'s chin and cheek.
- **Fix**:
  - Require `--source` explicitly on `shot_to_load.py preview` and run Pass 2's preview on the 24.0 fps post-Pass-1 cut (`fun_..._from_laid.mp4`).
  - Enable the SAM 3.1 head track (`TRACK_HEAD = True`) by default so `lead`'s face mask stays whole (`10,662–13,505 px` on `759-761`, `face_outside_region_px == 0` in `facts_pass.py`).
  - Report latent-step keep-out carves touching the head in `runs/<run>/step_carves.json` (`bench/capture_masked_run.py::step_keepout_carves`).

### Case Study 2: Hallway `1306-1418` Bidirectional Beard Leak & Short-Dip Bridging (2026-10-19)

- **Symptom**: In `third_fp_1306d` (replacing the bearded dad `third` with a clean-shaven reference), the rendered subject grew a dark/gray beard and eyeglasses across the window, whereas control sub-window `1340-1360` (`third_fp_1340b`) had `0/21` beard and `0/21` glasses.
- **Root Cause & Fix**: Two mechanisms left `third`'s beard unmasked in the bidirectional DiT context (`1402-1404` / `1408-1411` Sapiens `Face_Neck` dips and `--drop 1411-1413` on frames with `16k–48k px` of visible face). Guarded by `bench/head_tube_parts.py::bridge_short_tube_dips` (`SHORT_DIP_MAX_FRAMES = 6`, `SHORT_DIP_RATIO = 0.6`) and `bench/capture_masked_run.py::dropped_or_unmasked_face_in_context` (`dropped_frame_has_visible_face`, `DROPPED_FACE_MIN_PX = 4000`); see [`docs/masked_v2v/AGENTS.md`](AGENTS.md) §2.

### Case Study 3: Singing vs Silent Mouth Drawing Rules (`0278f/i` vs `1340c/d`, 2026-10-18)

- **Symptom & Fix (`8ac7e928`, `bench/shot_to_load.py::REFERENCE_DEFAULT`)**: On silent shot `278-293` (`0278d/e/h`), drawing a closed mouth line sat `8–12 px` above the gray head's sculpted lips and read as parted lips, while on singing window `1340-1360` (`1340c`) suppressing the mouth left lips shut. Keep `--darken-head` off (`1.0x`) and decide the mouth per window from the voice table (`mouth="auto"`, `--closed-lip-line` on by default; see [`reference_and_prompt_rules.md`](reference_and_prompt_rules.md) for current `open_mouth`, `closed_mouth`, and `MOUTH_BEND` rules).

### Case Study 4: 24.0 fps Source Timing, the `17n + 5` Grid, & Dual-Clock Audio Sync (`lead_fp_0751`, 2026-10-18)

- **Symptom & Fix (`e44af676`, [`docs/h3_resolutions.md`](../h3_resolutions.md))**: `lead_fp_0751` (`751-785` on a `23.976` fps cut) sought 1 frame early (`floor(751 * 23.976024 / 24.0) = 750`). Every working copy, preview, reference, render, and assembled cut is forced to exact `24.0` fps (`24/1`), and every window length sits on both clocks (`h3_rules.on_both_clocks`: `39, 90, 141, 192, 243, 294, 345`); see [`docs/masked_v2v/AGENTS.md`](AGENTS.md) §6 and [`frames_rules.md`](frames_rules.md).
