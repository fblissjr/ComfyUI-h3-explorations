# Masked Video-to-Video Execution & Debugging Protocol

This protocol is the step-by-step operational manual for building, auditing, rendering, debugging, and landing masked video-to-video loads. Read [`docs/masked_v2v/AGENTS.md`](AGENTS.md) first, above all "The owner's rules for passes, prep and judging", which every step below follows, then [`data/AGENTS.md`](../../data/AGENTS.md); see [`docs/wiki/masked_v2v_pipeline.md`](../wiki/masked_v2v_pipeline.md), [`docs/wiki/masked_v2v.md`](../wiki/masked_v2v.md), and [`docs/wiki/runbook.md`](../wiki/runbook.md) for the broader design and card procedures.

---

## Stage 0: Environment, Ports, & Pre-Flight Checks

1. **Never run `uv` (`uv run`, `uv sync`, `uv add`)**:
   - Invoking `uv` with the ComfyUI venv active empties that venv and dirties `uv.lock` ([`AGENTS.md`](../../AGENTS.md), "Operative rules"; [`docs/comfy_notes.md`](../comfy_notes.md)).
   - Always invoke the ComfyUI venv Python directly (`<comfy-venv>/bin/python`), and prefix CPU-only tools and checks with `CUDA_VISIBLE_DEVICES=`.
2. **Environment keys (`H3_OUTPUT_DIR`, `H3_INPUT_DIR`, `H3_MODEL_STORE`, `H3_CAPTURE_ROOT`)**:
   - `H3_OUTPUT_DIR`: the server's output share (`<share>/output`, holding `Video/<video>/`). Read by `workflows/h3_config.py::output_dir`, [`bench/shot_to_load.py`](../../bench/shot_to_load.py), [`bench/load_chain.py`](../../bench/load_chain.py), and [`bench/rebuild_delivery.py`](../../bench/rebuild_delivery.py).
   - `H3_INPUT_DIR`: the server's input directory (`<share>/input`, holding working copies and reference stills). Read by `bench/masked_job.py::input_dir`.
   - `H3_MODEL_STORE`: local model directory holding `workflows/h3_config.py::POSE_MODEL` and perception weights.
   - `H3_CAPTURE_ROOT`: transient capture directory when captures are stored outside `data/`.
3. **Inspecting the server process (`:8188`) and review server (`:8765`)**:
   - The server process is the resource, not the GPU ([`AGENTS.md`](../../AGENTS.md)). Before stopping or restarting ComfyUI on port `8188`, inspect the port owner and its environment:
     ```bash
     ss -ltnp | grep -E ':8188|:8765'
     tr '\0' '\n' < /proc/<pid>/environ | grep '^H3_'
     ```
   - If any arming key (`H3_CAPTURE`, `H3_SOL_OBSERVE`, `H3_SOL_PROBE`, `H3_TELEMETRY`) is set in `/proc/<pid>/environ`, ask before killing the server.
   - Never use `bench/restart_comfy.sh` (disabled; exits 2). Follow the manual recipe in [`docs/comfy_notes.md`](../comfy_notes.md) ("The recipe, by hand").
4. **Code stamps and clean node code**:
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
  - **NEVER** run `preview` or `job` against the raw unswapped source video.
  - Use the newest cut of the whole video that holds pass 1 ([`bench/rebuild_delivery.py`](../../bench/rebuild_delivery.py) `--video <video> --clip`, producing `fun_..._from_laid.mp4` where pass 1's subject is already replaced) and register it in `<share>/input/` and `data/<video>/manifest.json`. A cut of one span (5.3) begins on that span's first frame and is not a source ([`AGENTS.md`](AGENTS.md) §1).
  - Run a **fresh** `bench/shot_to_load.py preview --source <post_pass1_video>` so SAM 3.1 and Sapiens segment pass 1's **actual replaced appearance** (new hair/head silhouette) and the other subject's track against the post-Pass-1 picture, then make pass 2's whole chain again on it. Only the owner's own words carry from the original video's preview: the cast boxes and `data/<video>/plan.json`.

### 1.2 Mandatory `--cast` Verification

Always pass `--cast data/<video>/cast/cast.jsonl` to both `bench/shot_to_load.py preview` and `bench/shot_to_load.py job`:
- `bench/shot_to_load.py` checks `Path(a.cast).is_file()` up front and raises `bench/shot_to_load.py::Refused` if the cast file is missing.
- Without `--cast`, the builder cannot seed SAM 3.1 from confirmed cast boxes or auto-identify `lead` vs `second` vs `third` from reference embeddings (`bench/shot_to_load.py::seed_of`, `bench/shot_to_load.py::others_of_load`).

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
- **A lost head is re-seeded only on the subject's own face points**: where the head track loses her head and her person track still holds her, `bench/head_tube_parts.py::sam31_head_tube` re-seeds a head from the top of the person track only when her face points over the line lie on her own replaced classes (`bench/head_tube_parts.py::own_part_face_points`, at least `bench/head_tube_parts.py::REGAIN_FACE_POINTS_MIN`). Otherwise her head is hidden there and the frame stays empty, listed with its count under `regain_refused` in the head tube's provenance. On `second_hfp_4912e` the top of her person track was her raised hand, and the re-seed had made it her head.
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
- **One Shot = One Prompt = One Render (Never Split a Shot `<= 345` Frames)**:
  - Every shot in `data/<video>/shots.json` of `<= 345` frames (`max(h3_rules.chain_lengths())`) renders as **one single-window H3 load** (`workflows/masked_job_graph.py::window_for`) on both clocks (`h3_rules.on_both_clocks`: `39, 90, 141, 192, 243, 294, 345`) with zero mid-shot seams ([`docs/wiki/decisions.md`](../wiki/decisions.md), 2026-10-15, "One shot is never split"; 2026-10-21, windows on both clocks).
  - **Handling interior occlusions inside a single whole-shot load**: When a foreground dancer or hand completely occludes a subject for a few frames inside a shot (`track_share == 0`, `parts_share == 0`), **never** split the shot into two sub-loads before and after the occlusion. Keep the entire shot in one load, account for the occluded frames with `bench/capture_masked_run.py verify <capture> --no-mask "<subject>:<first>-<last>=<why>" --by <session>`, and let the node's composite leave those frames as the source (`left_as_the_source_with_no_mask`, `bench/capture_masked_run.py::verify_checks`) only where she is truly hidden, on the owner's word, marked `hidden` in the plan's `head_marks` ([`AGENTS.md`](AGENTS.md) §10). A frame where she shows and the mask is empty is a fault to fix, never an occlusion to pass (decisions.md, 2026-10-22).
  - **How every preprocessed input is locked to `[shot.first .. shot.last]`**:
    1. **Preview (`shot_to_load.py preview --shot <first>`)**: Looks up `[first, last]` (`N = last - first + 1` frames) in `data/<video>/shots.json`, seeds SAM 3.1 from the confirmed box inside `[first, last]` in `data/<video>/cast/cast.jsonl`, and writes `track`, `head_track`, `parts`, `classes`, `pose`, `mesh`, and `strangers` starting at frame `0 == shot.first` for `N` frames (`shot.first .. shot.last`).
    2. **Head Reference & Pose (`shot_to_load.py reference`)**: Runs `pose_along_shot.py` and `head_reference.py` on `--first <shot.first> --count <N>` with `--mesh-first <shot.first>` and `--classes-first <shot.first>`, so `<Video 1>` (`<subject>_head_reference_<first>_<last>.mp4`) has frame `0 == shot.first` and exact length `N`.
    3. **Prompt & Voice (`shot_to_load.py text`)**: `load_text.py` reads the clothing span and vocal activity (`voice_spans.json`) over `[shot.first, shot.last]` and records `"frames": [shot.first, shot.last]` in the job JSON (`masked_job.py` refuses any prompt whose frame range does not match `[first, last]`).
    4. **Hard Verification Gate (`capture_masked_run.py verify`)**: Checks `at_is_the_previews_first_frame` (`at == shot.first`), `inputs_reach_the_span` (every file covers `shot.first .. shot.last`), and `inputs_at_the_canvas` (`1024x768` at `24.0` fps) before any `plan` or `render` can run.
    5. **A shot of two subjects**: both use the **same shot boundaries `[shot.first, shot.last]`**. Pass 1 is the subject already kept ([`AGENTS.md`](AGENTS.md) §1), made on the working copy with the other subjects tracked on `[shot.first, shot.last]` as keep-outs (`others`); pass 2 is made on the newest whole-video cut that holds pass 1 (`--source <post_pass1_video>`), from a fresh preview and every stage after it, so pass 1's keep-out follows its new silhouette frame for frame.
  - **How inputs align with H3's both-clocks window (`h3_rules.on_both_clocks`) and what context/overlap frames are thrown away**:
    - **Single-window shot (`N <= 345` frames, `[first .. last]`)**: `track`, `parts`, `others`, and `<Video 1>` `head_reference` load `N` frames at `24.0` fps starting at `first`. `window_for(N)` chooses the shortest length `W` on **both clocks** (`h3_rules.on_both_clocks`: `39, 90, 141, 192, 243, 294, 345`) that holds `N`. When `N < W`, `workflows/masked_job_graph.py::window_pad` pads the source loader and audio with `after = W - N` source frames after `[first .. last]` and none before (before it only at the clip's end; the VAE's 17-frame clips, 2026-10-22), `video_mask.py::MiniMaxH3MaskedSource.execute` pads the masks with zeros (under the default `as filmed`; `face covered` holds the edge masks, [`AGENTS.md`](AGENTS.md) §6) on those context frames where the shot goes on and, across a cut, puts held copies of the shot's edge frame there with its masks (`held_before`, `held_after` from `workflows/masked_job_graph.py::held_context`; `video_mask.hold_context`; the track silent under them, 2026-10-22), unless that side's edge frame has no region or one under `video_mask.HELD_EDGE_LEAST` of the shot's middle region, when it keeps real frames (`held_fell_back`), `video_mask.py::zoomed_source` reads the shot's one zoom box (`video_mask.py::zoom_box`) over the whole window, and `audio_freeze_song.py::MiniMaxH3AudioFreezeSong.execute` **throws away the `before` and `after` context frames** after VAE decode (`trim = pad_before`, `kept_frames = total - pad_after`) so the output MP4 and saved region/matte are exact frames `[first .. last]` (`N` frames).
    - **Multi-window chained shot (`N > 345` frames, e.g. Kitchen `606–998`, `393` frames on `[345, 141]`)**: Only shots exceeding `345` frames run as chained windows inside one load, using lengths on the dual video+audio clock grid (`h3_rules.chain_lengths`: `141, 192, 243, 294, 345`, `h3_rules.on_both_clocks`) with `context_frames = 90` overlap. Window 2 takes the last `90` frames of Window 1 as overlap context, generates `141` frames (`90` overlap + `51` new), and the stitcher **throws away the `90` overlap context frames from Window 2** plus any tail frames past `N`, emitting exact `N` frames (`393` frames = `606–998`).
- When remaking an existing load on the same shot and subject (`--remake`), you must pass either `--carry-words-from <old_job.json>` or `--no-carry-words "<why>"` so span words (`--no-face-on`, `--no-mouth-on`, `--mouth-on`, `--hold-part`, `--eyes-shut-on` and `--eyes-shut-form`; the full list is [`AGENTS.md`](AGENTS.md) §8, "What only a person writes") are never dropped by accident. A seed and a zoom asked past the default are never carried.
- **Never `--drop` frames where the original actor's face or beard is visible**: H3 runs full attention over all `5n + 2` latent steps of the `17n + 5` window regardless of `tokens_sum == 0`. Dropping interior frames saves zero compute and exposes the unmasked source face to bidirectional temporal attention. Use `--no-face-on A-B=WHY` on back/side turns instead: the reference draws no face lines there, and on a hair-and-face part the capture drops only the face classes from her part, the mask the load carries, on those frames, so her hair stays in (`bench/capture_masked_run.py::grade_holds`; a face-only part is emptied there). The node grows the region from that part, so its margin may still cover some of the face.
- **No stale pose or mask data**: `job` chooses the load's reference folder (`bench/shot_to_load.py::reference_folder`), and `bench/_lib/code_stamp.py::check_upstream_mtime` only warns. What each stage reads, what a change forces, and what no tool enforces: [`AGENTS.md`](AGENTS.md) §8.

### 2.3 Cropping & Drawing the Head Reference (`shot_to_load.py reference`)

```bash
<comfy-venv>/bin/python bench/shot_to_load.py reference <load job file> --by <session>
<comfy-venv>/bin/python bench/shot_to_load.py text <load job file> --by <session> --plan data/<video>/plan.json
```

The job file is positional for both. `reference` reads the pose model's store from `H3_MODEL_STORE` (or
`--model-store`); `text` refuses without `--plan`.

Under the hood, `step_reference` invokes [`bench/head_reference.py`](../../bench/head_reference.py), which orchestrates [`bench/crop_reference_to_head.py`](../../bench/crop_reference_to_head.py) and [`bench/draw_face_points.py`](../../bench/draw_face_points.py):

1. **Head-box crop (`bench/crop_reference_to_head.py`)**:
   - Sizes one constant box in the canvas's shape across the load (`bench/crop_reference_to_head.py::one_box`). By default (`head`) it is the union of the Sapiens face box and the SAM 3.1 head track (`bench/crop_reference_to_head.py::CROWN_ROOM`, `bench/crop_reference_to_head.py::HEAD_TRACK_ROOM`) so tilted crowns are not cut. A subject whose plan says `reference: {crop: "face", why}` (`bench/head_reference.py::CROP_FORMS`, 2026-10-22) is cut to the faces alone with `bench/crop_reference_to_head.py::HEAD_ROOM`, the head track not read: the face larger in `<Video 1>`, the crown possibly cut, the box 0278f was drawn at. A subject whose plan says `reference: {crop: "cranium", why}` (2026-10-22) is cut round the body model's grey head: each frame's face with the grey joined to it above mid-face, `bench/crop_reference_to_head.py::CRANIUM_ROOM` above and beside, `CRANIUM_NECK` below the chin, enlarged at most `CRANIUM_MOST`, the head track not read; the source is cut by that box as given (`--box`). The job carries the word as `job.reference.crop` and the step passes `--crop face|cranium --crop-why`; the record's `crop` says the form and the enlargement, and under `cranium` the frames the grey head reaches the box's edge and the travel's cost (`crop.cranium`). Every subject of fun takes `cranium` (the plan; the owner, 2026-10-22). A frame whose face has no grey head joined to it does not size the cranium box; a span with no grey head at all keeps those frames (`crop.cranium.those_frames_size_the_box`). A frame with a wrongly fitted grey body still sizes it.
   - Cuts both the SAM 3D Body mesh video (`<stem>_plain_cut.mp4`) and the source video (`<stem>_source_cut.mp4`) to that box and writes each one's record and carried points beside it with `writer_stamp`.
2. **Facial landmark drawing (`bench/draw_face_points.py`)**:
   - Default form (`bench/shot_to_load.py::REFERENCE_DEFAULT`): `iris="contrast"`, `mouth="auto"`, `closed_lip_line=True` (`--closed-lip-line` on by default; `--darken-head` off).
   - **Silhouette clipping**: all drawn marks are clipped to the gray SAM 3D Body head silhouette so profile eyes shrink cleanly into the head edge instead of floating in background space (which H3 can misread as eyeglasses).
   - **Mouth mode rules (Closed = 3D Gray Mesh Lips; Open = Black Interior + White Outline)**:
     - **No mouth smoothing (`smooth_mouth_area` retired, 2026-10-20)**: `bench/draw_face_points.py::smooth_mouth_area` is no longer called on the drawing path (still defined, and exercised only by `bench/check_draw_face_points.py`). Painting a flat gray oval over the mouth region created a bright gray splotch across the lips, chin, and hands (`0415e`, `0484d`, `1039b`) that H3 interpreted as an open mouth.
     - **Closed-mouth frames stay as natural 3D gray mesh lips (`mouths[f] = None`)**: Whenever Sapiens2 (`sapiens2-seg-1b` + `sapiens2-pose-1b`) detects a closed mouth (`(not has_hole) or (gap <= INNER_GAP_CLOSED_PX)`, `bench/draw_face_points.py::INNER_GAP_CLOSED_PX`), `bench/draw_face_points.py::run` sets `mouths[f] = None` and draws no mouth there, leaving the 3D gray mesh's own sculpted closed lips untouched.
     - **Open-mouth frames draw black interior + white outline (`inner_lip_ring` / `mouth_inner`)**: Whenever Sapiens2 detects an open mouth (a teeth/tongue hole in the 29-class map and an inner-lip gap over `bench/draw_face_points.py::INNER_GAP_CLOSED_PX`), the black interior + white outline is drawn at the mouth, and `bench/draw_face_points.py::fit_mouth_to_head` keeps the drawn mouth inside the gray head's mouth-to-chin space (its docstring says how). That is the default form, `inner outline`. A subject's plan may set `reference.open_mouth` (`bench/draw_face_points.py::OPEN_MOUTH_FORMS`); the lead's is `outlined`: outer and inner lip outlines on the class map's lips, capped to the upper face's size ([`AGENTS.md`](AGENTS.md) §7).
   - **A lowered eye**: an eye whose iris-ring share outside the lids is at or over `bench/draw_face_points.py::DOWNCAST_RING_OUTSIDE_MIN` is drawn dark (lid lines, no white, the iris a black disc clipped to the lids; `bench/draw_face_points.py::dark_irises`), only under `--iris contrast` with the closed lip line; a person's `--eyes-shut-on` with `--eyes-shut-form` (`bench/draw_face_points.py::EYES_SHUT_FORMS`) draws the eyes shut (`lids`, `none`, or `closed`: painted over in the skin grey round them with one lid line) or dark (`lowered`) on its frames. A dark iris is drawn in lids no thinner than `bench/draw_face_points.py::EYE_OPEN`'s `least`: thinner lids have the lower lid opened down to it, so a piece of black always shows. A shut stretch takes `--eyes-shut-form closed`, or it is drawn open. Why, and the closed-eye rule taken back out on 2026-10-21: [`AGENTS.md`](AGENTS.md) §7.
3. **Generating the Prompt Text (`shot_to_load.py text` — Unified `<Subject 2>` Sentence)**:
   - Invokes [`bench/load_text.py`](../../bench/load_text.py) (`bench/load_text.py::NO_SCENE`, `bench/load_text.py::NO_CLOTHING`) to generate the load's text from the reference record, `data/<video>/plan.json`, and the voice table.
   - All face-drawn loads use a **single unified `<Subject 2>` sentence** (`bench/load_text.py::subject_2`) that describes the reference video ("with its face drawn as thin white outlines: the eyes, each with a dark iris, open, close and look where the person's eyes do", `bench/load_text.py::FACE_EYES`), and names the mouth only for a subject whose plan draws its open mouth outlined, on a load whose drawing opens the mouth on some frame (`bench/load_text.py::MOUTH_CLAUSE`, `mouth_drawn_open`; [`AGENTS.md`](AGENTS.md) §7).
   - The rest of the text is the owner's form of 2026-10-22: the definition names the picture's hair (`text.picture_hair`) and `[Shot 1]` the shot's (`text.hair_words`), both required of every subject; no `looks` words; a face-only retention line "the girl's face from <Picture 1> is retained."; a face-and-hair subject "the face and the <picture_hair> of <Picture 1> are retained.", with the summary "with the face and the <picture_hair> of <Picture 1>" and `[Shot 1]` "has the face and the hair of <Picture 1>" (2026-10-23; `text.hair_change` retired and refused); where a face is drawn, `<Subject 2>`'s retention line transfers "where the eyes look and how far the mouth opens" and `[Shot 1]` has her eyes and mouth follow the figure's (`bench/load_text.py::TRANSFERRED_DRAWN`, `FOLLOWS_DRAWN`, the owner's approval of 2026-10-22), the mouth's words only where the drawing opens a mouth on some frame (the closed-lip line's `open_frames`, `bench/load_text.py::mouth_drawn_open`; else `TRANSFERRED_DRAWN_NO_MOUTH`, `FOLLOWS_DRAWN_NO_MOUTH`; the owner's go of 2026-10-22 after second_hfp_0278l, case study 3; "and eye movements" on a plain cut), the music `N/A` ([`AGENTS.md`](AGENTS.md) §7; `bench/load_text.py`'s docstring).
   - **What each form did on a render** is the owner's to judge on playback (`data/<video>/owner_answers.jsonl`, `data/<video>/verdicts/verdicts.jsonl`); what was decided and corrected is in [`docs/wiki/decisions.md`](../wiki/decisions.md) (2026-10-20, item 6, corrected on 2026-10-21).

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

### 3.3 The Look, Full Prompt Read (`what_h3_got.py`), & `render`

1. **The look the tool requires**: `bench/capture_masked_run.py looked`, signed by the lead on the owner's word with "not inspected" in its note; no session looks at the render before the owner does (decisions.md, 2026-10-22). A plan strip may be sanity-checked. Or the maker runs `bench/load_chain.py no-look <load job file> --by <session> --not-first WORDS.txt` where the gate is clear and no forcing rule in `bench/load_chain.py::FORCING_FLAGS` or `bench/load_chain.py::FORCING_FIGURES` stands; `--not-first` is a file with the maker's statement that the load is not the first of its kind, and which earlier load it follows).
2. **Full prompt read (`bench/what_h3_got.py`)**:
   ```bash
   CUDA_VISIBLE_DEVICES= <comfy-venv>/bin/python bench/what_h3_got.py \
     --job-folder data/<day>_<load>_job --load <load> --stage plan
   ```
   Read the entire prompt text in full from the queued graph ([`AGENTS.md`](../../AGENTS.md)).
3. **Queue the render (`load_chain.py render`)**:
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
- `page` prepends the candidate clip with `state: "open"` to `data/<video>/review/items.json` and creates the media symlinks under `data/<video>/review/media/`.

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

- **Symptom**: In `third_fp_1306d` (replacing the bearded dad `third` with a clean-shaven reference), the rendered subject grew a dark/gray beard and eyeglasses across the window, whereas a control render on clean sub-window `1340-1360` (`third_fp_1340b`) had `0/21` beard and `0/21` glasses.
- **Root Cause**:
  - H3's DiT uses full bidirectional temporal attention across all frames in the window. Any frame where the original actor's gray beard is left unmasked in the frozen context acts as an in-context reference that H3 copies onto the regenerated hole across the entire window.
  - Two distinct mechanisms left `third`'s beard unmasked in `1306-1418`:
    1. **Transient Sapiens class dips during fast motion (`1402-1404` and `1408-1411`)**: While SAM 3.1's person track stayed solid (`96k–146k px`), Sapiens `Face_Neck` dipped to `20,129 px` on `1403` and `2,413–5,736 px` on `1409-1410` (`< 60%` of the `36,585 px` and `47,200 px` flanking frames), exposing `51,735 px` of raw beard outside the region.
    2. **Manual `--drop 1411-1413` on frames with a visible face**: Dropping frames `1411-1413` set `region_px = 0` while `1411-1413` still had `48,938 px`, `16,409 px`, and `21,341 px` of visible `Face_Neck` + beard in the frozen context.
- **Fix**:
  - `bench/head_tube_parts.py::bridge_short_tube_dips` (`SHORT_DIP_MAX_FRAMES = 6`, `SHORT_DIP_RATIO = 0.6`) automatically bridges `<= 6`-frame head-tube dips inside a continuous person track (`1402-1404` and `1408-1411` bridged to `35k–45k px`).
  - `bench/capture_masked_run.py::dropped_or_unmasked_face_in_context` raises the blocking preflight flag `dropped_frame_has_visible_face` (`DROPPED_FACE_MIN_PX = 4000`) whenever a dropped/empty frame inside a tracked span has `>= 4,000 px` of visible `Face_Neck`.

### Case Study 3: Singing vs Silent Mouth Drawing Rules (`0278f/i` vs `1340c/d`, 2026-10-18)

- **Symptom**:
  - On silent shot `278-293` (`second_hfp_0278d/e/h`), drawing a closed mouth line caused `second`'s lips to hang open, because the pose model's mouth line sat `8–12 px` above the SAM 3D Body gray head's own sculpted lips and read as two parted lips. Darkening the gray head (`0278i`, `--darken-head`) made the filled white-and-black contrast eyes pop as cartoon "googly eyes".
  - Conversely, on singing window `1340-1360` (`third_fp_1340c`), suppressing the drawn mouth caused `third`'s lips to stay shut on all 21 frames while he sang.
- **Fix (`8ac7e928`, `bench/shot_to_load.py::REFERENCE_DEFAULT`)**:
  - Keep `--darken-head` off (head brightness `1.0x`) so filled contrast eyes blend naturally (`0278f`).
  - Decide the mouth per window from the voice table (`mouth="auto"`, `--closed-lip-line` on by default): draw **no mouth** on silent windows (`0278f`, lips stays closed 16/16 frames) and draw the **repaired mouth** (covering the gray head's sculpted lips and fitted above its chin) on voiced/singing windows (`1340d`, full singing articulation).

### Case Study 4: 24.0 fps Source Timing, the `17n + 5` Grid, & Dual-Clock Audio Sync (`lead_fp_0751`, 2026-10-18)

- **Symptom**: `lead_fp_0751` (frames `751-785` rendered against an assembled cut) started 1 frame early (`750` loaded for `751`) and showed boundary flicker against the plate; additionally, ultra-short sub-window test clips (`35` frames = `1.458` s, rounded to `39` frames with `4` held tail frames) can feel brief or slightly off-cadence in browser loop playback compared to full-shot renders.
- **Root Cause**:
  1. **23.976 vs 24.0 fps seek drift**: The original video's native rate was `24000/1001` (`23.976024` fps). While the raw working copy `fun_0000_6394_24p_sound_in_step.mov` had been retimed to exact `24.0` fps (`bench/clip_analysis.py` `init --copy`), the assembled cut was written at `23.976` fps, and `workflows/masked_job_graph.py::seek` divides the start frame by `24.0` (`751 / 24.0`), which seeks to frame `750` on a `23.976` fps stream (`floor(751 * 23.976024 / 24.0) = 750`).
  2. **Video VAE grid (`17n + 5`) vs Audio latent clock (`h3_rules.on_both_clocks`)**: Video frame counts round up to `17n + 5` (`5n + 2` latent steps, up to `h3_rules.MAX_LENGTH` = `362`; see [`docs/h3_resolutions.md`](../h3_resolutions.md), "The length axis"), where attention scales with the square of the sequence. Meanwhile, H3 lays audio at `h3_rules.AUDIO_LATENT_RATE` latents/s against `h3_rules.FPS` (`24`), giving `round(frames / 24 * AUDIO_LATENT_RATE)` audio latents. The two clocks agree **only on every third `17n + 5` length (`39, 90, 141, 192, 243, 294, 345`, `h3_rules.on_both_clocks`)**; any other grid length (`22`, `362`, etc.) leaves a `< 1/3` audio-latent rounding remainder at the final frame.
- **Fix & Standing Rule (`e44af676`, [`docs/h3_resolutions.md`](../h3_resolutions.md))**:
  - **24.0 fps end-to-end**: Every working copy, preview, reference, render, and multi-pass composite delivery ([`bench/rebuild_delivery.py`](../../bench/rebuild_delivery.py), [`bench/assemble_delivery.py`](../../bench/assemble_delivery.py)) is forced to `24.0` fps (`24/1`), and `bench/shot_to_load.py::working_copy` and [`bench/masked_job.py`](../../bench/masked_job.py) refuse any source whose manifest rate is not `24`.
  - **Dual-clock chained seams & whole-shot delivery**: Chained multi-window seams (`h3_rules.chain_lengths`, e.g. `[345, 141]` with `90`-frame context on Kitchen `606-998`) **must** meet on lengths that are whole on both clocks (`h3_rules.on_both_clocks`) so audio never starts a chained window off-picture; short sub-window clips (`35f` / `39f`) are used only as fast diagnostic probes before rendering the full shot (`docs/wiki/masked_v2v.md`, "A shot, made whole").
