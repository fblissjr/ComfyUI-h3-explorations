# docs/masked_v2v/: the masked video-to-video lane

Loaded when a session opens a file here; [`docs/AGENTS.md`](../AGENTS.md) and the root [`AGENTS.md`](../../AGENTS.md) still govern.

- **Start here**:
  - Execution & debugging protocol: [`docs/masked_v2v/PROTOCOL.md`](PROTOCOL.md)
  - Lane rules and where state lives: [`data/AGENTS.md`](../../data/AGENTS.md) (and each project's untracked `data/<video>/AGENTS.md`)
  - End-to-end pipeline, design, and card runbook: [`docs/wiki/masked_v2v_pipeline.md`](../wiki/masked_v2v_pipeline.md), [`docs/wiki/masked_v2v.md`](../wiki/masked_v2v.md), and [`docs/wiki/runbook.md`](../wiki/runbook.md)
- **This lane is one use of H3.** A finding about common code (reference conditioning, the shared nodes) is written where that code's page is and linked from [`docs/wiki/index.md`](../wiki/index.md).

## The rules that bite when missed

### 1. What `--source` to pass in, and which subject is made first on a shot of two

- [`bench/shot_to_load.py`](../../bench/shot_to_load.py) `preview` (`bench/shot_to_load.py::step_preview`) requires `--source VIDEO`, a path, with **no silent default**. Never assume `data/<video>/shots.json`'s original raw video is the right source for the pass in hand.
- **A kept subject stays** (the owner, 2026-10-21; [`docs/wiki/decisions.md`](../wiki/decisions.md)). A subject already kept on the shot is not remade to make the other, unless something broken changes the composite, which is the owner's look to say. The assembler's `two_pieces_change_the_same_pixels` flag ([`bench/assemble_delivery.py`](../../bench/assemble_delivery.py)) names the frames where two pieces change the same pixels: a place to look, not proof that the composite is broken. No subject goes first by rule: the kept one is pass 1.
- **A single-subject shot, or pass 1 of a shot of two**: pass the video's working copy, the file `data/<video>/manifest.json` registers (`bench/shot_to_load.py::working_copy` maps a record's copy to it).
- **Pass 2, where the other subject has something to replace.** A shot is done when every subject who can be swapped there is kept (the owner, 2026-10-21; [`docs/wiki/decisions.md`](../wiki/decisions.md)). A subject with nothing to replace, such as one whose face never shows, is recorded as `not seen` with [`bench/shots_look.py`](../../bench/shots_look.py) on the owner's word or a look, and `bench/shot_coverage.py::nothing_to_make` reads that line, so no load is owed for that subject. Otherwise the other subject is made on the newest cut that holds the kept one. Pass 1's output is pass 2's source, and pass 2's whole chain is made again on it, starting from a fresh `bench/shot_to_load.py preview --source <that cut>`: no track, mask, keep-out, reference or text is reused from a preview of the original video. Only the owner's own words carry: the cast boxes (`data/<video>/cast/cast.jsonl`) and `data/<video>/plan.json`. The cut is one of the whole video (`bench/rebuild_delivery.py --video <video> --clip`): a load finds its frames by number from the start of its source (`workflows/masked_job_graph.py::seek`), and a cut of one span begins on that span's first frame, so it is for looking at, not a source. Register the cut as [`PROTOCOL.md`](PROTOCOL.md) 1.1 says. `bench/shot_to_load.py::check_pass2_source` refuses the raw source on `preview` and `job` where another subject is laid on the frames (`--allow-raw-source WHY` lets it through); it knows the raw source by its name and path.
- **Why (Kitchen `604-1009` `second` + `lead` crossing on `755-774`, 2026-10-19)**: The original 1980s mom (`second`) had a tall dark bouffant hairdo rising ~60 px higher (`ymin=188..209`) than the replaced blonde mom (`second_hfp_0604h`, `ymin=211..221`). Reusing a preview segmented on the raw 1980s video gave `second`'s keep-out `5,843–9,276 px` of old bouffant hair dome and carved an invisible dark helmet out of `lead`'s face as `lead` walked behind her.

### 2. Bidirectional Temporal Attention & Unmasked Context Leaks (The Beard Rule)

- H3's DiT uses full bidirectional temporal attention across the entire window (`W` frames, `5n + 2` latent steps for `17n + 5` frames). Even 3–4 frames where an original subject's salient feature (such as a gray beard or eyeglasses) is left unmasked in the frozen context will propagate that feature onto the regenerated subject across the entire window.
- **Two causes caught on Hallway `1306-1418` (`third`, 2026-10-19)**:
  1. **Transient Sapiens class-map dips (`<= 6` frames) during fast motion**: on `1402-1404` and `1408-1411`, Sapiens `Face_Neck` dipped below 50% of flanking frames while the person track stayed solid (`51,735 px` of exposed beard). Fixed automatically by `bench/head_tube_parts.py::bridge_short_tube_dips` (`bench/head_tube_parts.py::SHORT_DIP_MAX_FRAMES` = `6`, `bench/head_tube_parts.py::SHORT_DIP_RATIO` = `0.6`).
  2. **Dropping interior frames where the face is still visible**: `--drop 1411-1413` zeroed the regeneration mask on frames that still had `16,409–48,938 px` of visible `Face_Neck` + beard. Guarded by `bench/capture_masked_run.py::dropped_or_unmasked_face_in_context` preflight blocking flag `dropped_frame_has_visible_face` (`bench/capture_masked_run.py::DROPPED_FACE_MIN_PX` = `4000`). **Never `--drop` a frame where the subject's face is still visible to the camera.** On back- or side-of-head turns, pass `--no-face-on` to [`bench/shot_to_load.py`](../../bench/shot_to_load.py) `job` instead: `<Video 1>` draws no face lines there, and on a hair-and-face part the capture drops only the face classes from her part, the mask the load carries, on those frames, so her hair stays in (`bench/capture_masked_run.py::grade_holds`). The node grows the region from that part, so its margin may still cover some of the face. A face-only part is emptied on those frames.

### 3. Always pass `--cast data/<video>/cast/cast.jsonl` and verify it exists

- [`bench/shot_to_load.py`](../../bench/shot_to_load.py) `preview` and `job` refuse missing `--cast` files cleanly (`bench/shot_to_load.py::Refused`). Without `--cast data/<video>/cast/cast.jsonl`, the builder cannot auto-identify `lead` vs `second` vs `third` from confirmed reference embeddings.

### 4. Pre-render & Post-capture Telemetry (`facts_pass.py`, `step_carves.json`, `code_stamp.py`)

- Run [`bench/facts_pass.py`](../../bench/facts_pass.py) on the capture directory to inspect `face_outside_region_px` (catches chin/cheek cutouts where `(s_face | (head_track & ~s_hair))` falls outside `region`), `other_overlap_px`, and `mouth_mode` before spending GPU render minutes.
- Inspect `runs/<run>/step_carves.json` (`bench/capture_masked_run.py::step_keepout_carves`) for latent-step keep-out cell carves touching the subject's head track or carried part.
- All derived `.npz` / `.json` artifacts carry `writer_stamp` (`bench/_lib/code_stamp.py::writer_stamp`, `bench/_lib/code_stamp.py::npz_writer_fields`). `bench/_lib/code_stamp.py::check_upstream_mtime` compares file times and returns warnings: [`bench/draw_face_points.py`](../../bench/draw_face_points.py) prints them, and [`bench/head_reference.py`](../../bench/head_reference.py) and [`bench/measure_moge2_depth.py`](../../bench/measure_moge2_depth.py) add them to their refusal to write over an output that is there. Nothing refuses a stale input on its own (§8).

### 5. Review Server (`items.json`), Verdicts (`owner_answers.jsonl`), and Static `poc/TO_REVIEW.html`

- The owner reviews clips and submits verdicts and frame notes on the interactive review page (`http://<host>:8765/`, served by `data/<video>/poc/serve.py` from `data/<video>/poc/index.html`), which reads `data/<video>/review/items.json` (`state: "open"` renders under `"1. Look at these next"`) and appends clicks to `data/<video>/owner_answers.jsonl`.
- Every new review clip **must** be added to `data/<video>/review/items.json` (`state: "open"`, via `bench/load_chain.py page`) **and** synced to `data/<video>/poc/TO_REVIEW.html` ([`bench/rebuild_delivery.py`](../../bench/rebuild_delivery.py) rebuilds `TO_REVIEW.html` and `PROVENANCE.html` and verifies `data/<video>/poc/` on disk via `bench/rebuild_delivery.py::verify_poc_on_disk` during the `pages` step; [`bench/build_cut_map.py`](../../bench/build_cut_map.py) builds the 2-tab chronological cut table, frame-level gap audit, and 6-stage hierarchical data flow lineage DAG in `data/<video>/poc/PROVENANCE.html`, `data/<video>/poc/CUT_MAP.html`, and `data/<video>/CUT_MAP.md`).

### 6. 24.0 fps End-to-End, One Shot = One Render (`<= 362` Frames), the `17n + 5` Grid, and What Held/Overlap Frames Are Thrown Away

- **Exact `24.0` fps (`24/1`) everywhere**: Every working copy, preview, reference, render, and assembled multi-pass delivery cut ([`bench/rebuild_delivery.py`](../../bench/rebuild_delivery.py), [`bench/assemble_delivery.py`](../../bench/assemble_delivery.py)) is forced to `24.0` fps (`bench/shot_to_load.py::working_copy` and [`bench/masked_job.py`](../../bench/masked_job.py) refuse any non-24 fps source).
- **One Shot = One Prompt = One Render (Never Split a Shot `<= 362` Frames)**:
  - A cut must always align 1:1 with a shot, its prompt, and its render ([`docs/wiki/decisions.md`](../wiki/decisions.md), 2026-10-15, "One shot is never split"). Every shot of `<= 362` frames (`h3_rules.MAX_LENGTH`) renders as **one single-window H3 load** (`workflows/masked_job_graph.py::window_for`) with zero mid-shot seams.
  - **Interior occlusions stay inside the single whole-shot load**: When a foreground dancer or hand completely occludes a background subject for a few frames inside a shot (`track_share == 0`, `parts_share == 0`), **never** split the shot into two sub-loads before and after the occlusion. Keep the entire shot in one load, account for the occluded frames with `capture_masked_run.py verify --no-mask`, and let the node's composite leave those `0 px` mask frames as the source (`left_as_the_source_with_no_mask`, `bench/capture_masked_run.py::verify_checks`).
  - **The only exception (`N > 362` frames)**: When a single camera shot in `data/<video>/shots.json` exceeds H3's `362`-frame ceiling (in `fun`, only Kitchen `604–1009` at `406` frames and Guitar Solo `2050–2622` at `573` frames), it still runs as **one load** (`shot_to_load.py job [--seam ...]`), which the song node chains across windows on the dual video+audio clock grid (`h3_rules.chain_lengths`: `39, 90, 141, 192, 243, 294, 345`, `h3_rules.on_both_clocks`) with `context_frames = 90` (`workflows/h3_config.py::MASKED_JOB_CONTEXT`) overlap frames.
- **How Every Preprocessed Input Is Locked to `[shot.first .. shot.last]` (Why Starting Mid-Shot Breaks Tracking/Masking Alignment)**:
  1. **Preview (`shot_to_load.py preview --shot <first>`)**: Looks up the exact shot `[first, last]` (`N = last - first + 1` frames) in `data/<video>/shots.json`, seeds SAM 3.1 from the confirmed frame box inside `[first, last]` in `data/<video>/cast/cast.jsonl`, and writes every tracker and perception stream (`track`, `head_track`, `parts`, `classes`, `pose`, `mesh`, and `strangers` keep-out) starting at frame `0 == shot.first` for `N` frames (`shot.first .. shot.last`).
  2. **Head Reference & Pose (`shot_to_load.py reference`)**: Runs `pose_along_shot.py` and `head_reference.py` on `--first <shot.first> --count <N>` with `--mesh-first <shot.first>` and `--classes-first <shot.first>`, so `<Video 1>` (`<subject>_head_reference_<first>_<last>.mp4`) has frame `0 == shot.first` and exact length `N`.
  3. **Prompt & Voice (`shot_to_load.py text`)**: `load_text.py` reads the clothing span and vocal activity (`voice_spans.json`) over `[shot.first, shot.last]` and records `"frames": [shot.first, shot.last]` in the job JSON (`masked_job.py` refuses any prompt whose frame range does not match the load's `[first, last]`).
  4. **Hard Verification Gate (`capture_masked_run.py verify`)**: Before any `plan` or `render` can run, `verify` checks three hard assertions on every mask, part, class map, and keep-out file: `at_is_the_previews_first_frame` (`at == shot.first`), `inputs_reach_the_span` (every file covers `shot.first .. shot.last`), and `inputs_at_the_canvas` (`1024x768` at `24.0` fps).
  5. **A shot of two subjects**: both use the **same shot boundaries `[shot.first, shot.last]`**. Pass 1 is the subject already kept (§1), made on the working copy with the other subjects tracked on the same span as keep-outs (`others`). Pass 2 is made on the newest cut that holds pass 1, from a fresh preview of that cut (`--source <the cut>`) and every stage after it, so pass 1's new appearance is in the picture and pass 1's keep-out follows its new silhouette frame for frame.
- **How Data Inputs Align with H3's Generated Frame Count (`17n + 5`) & What Overlap/Tail Frames Are Thrown Away**:
  - **Single-window shot (`N <= 362` frames, `[first .. last]`)**:
    1. Every input (`source` video, SAM `track`, held `parts`, `others` keep-outs, and `<Video 1>` `head_reference`) is loaded at exact `24.0` fps starting at `first` for `N` frames (`workflows/masked_job_graph.py::load`), and audio is trimmed to `extent.seconds = N / 24.0`.
    2. `window_for(N)` snaps `N` up to the next `17n + 5` H3 length `W` (e.g., `N = 31 -> W = 39`, `N = 45 -> W = 56`, `N = 344 -> W = 345`; `5n + 2` latent steps).
    3. `video_mask.py::held_mask` pads frames `N .. W-1` (`W - N` tail frames) by holding frame `N - 1` with the regeneration region open so H3 samples a valid `17n + 5` window, and `MiniMaxH3Song` (`extent = "first_seconds"`, `extent.seconds = N / 24.0`) **throws away the `W - N` held tail frames** after VAE decode so the emitted MP4 is exact frames `[first .. last]` (`N` frames).
  - **Multi-window chained shot (`N > 362` frames, e.g. Kitchen `606–998`, `393` frames on `[345, 141]` with `context_frames = 90`)**:
    1. Window 1 generates `W_1` frames (`345`).
    2. Window 2 overlaps Window 1 by taking the last `90` frames of Window 1 as latent context (`context_frames = 90`), and generates `W_2` frames (`141` = `90` overlap context + `51` new frames).
    3. The stitcher **throws away the `90` overlap context frames from Window 2** plus the `W_total - N` held tail frames (`396 - 393 = 3` frames), emitting exact `N` frames (`393` frames = `606–998`) with zero audio or visual seam.
- **No stale pose or mask data**: never reuse masks, SAM tracks, Sapiens2 part or class maps, or pose files across a change to what made them. What each stage reads and writes, what a change forces, and what no tool enforces: §8.

### 7. Head Reference Mouth Drawing & Unified `<Subject 2>` Prompt (Closed = 3D Gray Mesh Lips; Open = Black Interior + White Outline)

- **No mouth smoothing (`smooth_mouth_area` retired, 2026-10-20)**: `bench/draw_face_points.py::smooth_mouth_area` is no longer called on the drawing path (still defined, and exercised only by `bench/check_draw_face_points.py`). Painting a flat gray oval over the mouth region created a bright gray splotch across the lips, chin, and hands (`second_hfp_0415e`, `second_hfp_0484d`, `second_hfp_1039b`) that H3 read as an open mouth.
- **Closed-mouth frames stay as natural 3D gray mesh lips (`mouths[f] = None`)**: Whenever Sapiens2 (`sapiens2-seg-1b` + `sapiens2-pose-1b`) detects a closed mouth (`(not has_hole) or (gap <= INNER_GAP_CLOSED_PX)`, `bench/draw_face_points.py::INNER_GAP_CLOSED_PX`), `bench/draw_face_points.py::run` sets `mouths[f] = None` and draws no mouth there, leaving the 3D gray mesh's sculpted closed lips untouched.
- **Open-mouth frames draw black interior + white outline (`inner_lip_ring` / `mouth_inner`)**: Whenever Sapiens2 detects an open mouth (a teeth/tongue hole in the 29-class map and an inner-lip gap over `bench/draw_face_points.py::INNER_GAP_CLOSED_PX`), the black interior + white outline is drawn at the mouth, and `bench/draw_face_points.py::fit_mouth_to_head` keeps the drawn mouth inside the gray head's mouth-to-chin space (its docstring says how).
- **A lowered eye is drawn as its iris arc between the lids, with no white** (2026-10-20). Under `--iris contrast` with the closed lip line, the chain's default form (`bench/shot_to_load.py::REFERENCE_DEFAULT`), an eye whose iris ring lies `bench/draw_face_points.py::DOWNCAST_RING_OUTSIDE_MIN` or more outside its lids gets no disc, so no white or black fill, and its iris is drawn as an arc between the lids; the other eye keeps its contrast disc (`shapes` and `discs` in [`bench/draw_face_points.py`](../../bench/draw_face_points.py)). A person's word (`job --eyes-shut-on`) draws the eyes shut on its frames. The text is not changed for it. Drawing lowered eyes shut by rule was tried on 2026-10-21 and taken back out the same day: H3 drew her eyes open and toward the camera on second_hfp_0484g and second_hfp_1039e ([`docs/wiki/decisions.md`](../wiki/decisions.md)).
- **Single unified `<Subject 2>` prompt sentence (`bench/load_text.py::subject_2`)**: All face-drawn loads use a single unified `<Subject 2>` sentence ("with facial features drawn on its face as thin white outlines, where the eyes open, close and look where the person's eyes do") without mentioning mouth/lips, so the text never biases H3 toward opening or closing the mouth while preserving eye-gaze direction.
- **What each form did on a render** is the owner's to judge on playback: their verdicts are in `data/<video>/owner_answers.jsonl` and `data/<video>/verdicts/verdicts.jsonl`, and what was decided and corrected is in [`docs/wiki/decisions.md`](../wiki/decisions.md) (2026-10-20, item 6, corrected on 2026-10-21).

### 8. What a change forces: the chain in trace order

A load is made in the order below, and each stage reads what the stages above it wrote. A change to a stage's code or
inputs therefore forces that stage and every stage after it to be made again (root [`AGENTS.md`](../../AGENTS.md),
"Nothing an earlier run made is reused by a key"; [`docs/wiki/decisions.md`](../wiki/decisions.md), 2026-10-18,
"Derived data is made again whenever the code that makes it changes"). Trace a fault the same way: from the stage that
shows it, up through what that stage read.

- **The default remake is a fresh preview and every stage after it** (the owner, 2026-10-21: when unsure, make every
  input again; decisions.md). The exception is a test of one change, which may reuse a preview when `git log` from the
  preview's commit to `HEAD` shows no change to the code stage 1 reads (decisions.md, 2026-10-21). The preview's commit
  is `commit` in its job folder's `preview.queued.json`, which [`bench/masked_job.py`](../../bench/masked_job.py) writes
  when it queues the preview.
- **No remake starts later than `job --remake`, and none starts at `first`.** `first`, `plan` and `render` each run once
  under a load's name ([`bench/load_chain.py`](../../bench/load_chain.py), "The steps, and what each refuses"), so a
  remake is a new load with a new name, its own reference step and its own text.
- **Nothing enforces this today.** No step of the load chain compares the code that made an input with the code now.
  `bench/_lib/code_stamp.py::check_upstream_mtime` compares file times and only warns; the strangers record carries no
  writer stamp ([`bench/strangers.py`](../../bench/strangers.py)); and `shot_to_load.py reference --existing` takes a
  drawing already in its folder without asking what code drew it (`bench/shot_to_load.py::step_reference`). The
  session that makes a remake holds the rule.

**What only a person writes.** These are the chain's inputs, made by no stage. A remake carries the first four; on
pass 2, only the owner's own words among them carry (§1).

- `data/<video>/shots.json`: who is on each shot, a shot's `order`, the frames a subject was marked or boxed on, and a
  person's lines on which parts of a subject are seen there (`who.<subject>.seen_looks`, written by
  [`bench/shots_look.py`](../../bench/shots_look.py)).
- `data/<video>/cast/cast.jsonl`: the owner's boxes.
- `data/<video>/plan.json`: for each subject, the region, parts, keep-outs, restore, hair band, pieces, reference
  form, zoom, clothing, voice and text.
- The span words given to `job` (`--no-face-on`, `--drop`, `--hold-part`, `--no-mouth-on`, `--mouth-on`,
  `--eyes-shut-on`, `--parts-on`, `--track-not`). A remake carries them with `--carry-words-from OLD_JOB`, or says why
  not with `--no-carry-words WHY`.
- A look (stage 6), and an override of a gate flag (stage 5). Each belongs to one capture. A new capture takes its own
  look, or a person rests one on an earlier capture's with [`bench/look_carry.py`](../../bench/look_carry.py), which
  compares the two captures first.

**The stages.**

0. **The source, and the code the server runs.**
   - The source is the working copy, or on pass 2 the newest cut that holds the kept subject (§1).
   - The server runs the node pack: `nodes.py`, every module it imports, and `workflows/h3_config.py`. The preview,
     the plan and the render are graphs, and a graph runs the code the server imported when it started. A change to
     that code takes a restart by the recipe in [`docs/comfy_notes.md`](../comfy_notes.md), on an empty queue and
     after every queued load has landed: the queue and its history live in the server's memory (ComfyUI's
     `execution.py`, `PromptQueue`), and `land` reads the history.
   - The bench steps also import pack modules from disk on every run, with no restart
     (`bench/capture_masked_run.py::_pack`, for `video_mask`, `sapiens2_parts` and `loop_plan`). A capture made after an
     edit and before the restart reads other code than the graphs it is checked against.
   - [`bench/masked_job.py`](../../bench/masked_job.py) refuses to queue a graph beside uncommitted node code unless
     given `--override "WHO: WHY"`.
   - A change forces: the restart, then a fresh preview and every stage below.
1. **Preview**: `bench/shot_to_load.py preview` (the command: [`PROTOCOL.md`](PROTOCOL.md) 1.3).
   - Reads: the source (`--source`, required); `shots.json` (who is on the shot, its `order`, and the frame an
     unseeded tracker picks on, `bench/shot_to_load.py::pick_of`); the cast file (`--cast`): the shot takes one seed, the
     owner's box latest in the file among those inside it, and the other boxes are listed as unused
     (`bench/shot_to_load.py::seed_of`, `bench/cast_file.py::seeds`). With `--seed-cuts-of JOB.json=TABLE.json` (the
     shots table an earlier preview of the same frames saved), each of the tracker's cuts takes its own
     (`bench/shot_to_load.py::cuts_of_table`). Without `--cast` the tracker picks the largest person
     (`bench/shot_to_load.py::PICK`); `plan.json` (each subject's parts and still); `workflows/h3_config.py` (the
     canvas, SAM 3.1's memory selection, the head track).
   - Writes: its job file in `--folder`. With `--queue`, its job folder `data/<day>_shot_c_<first>[<letter>]_job/`
     and, on the share, each subject's track, head track, parts, class map, shot table, pose table and body-model
     video. Without `--queue` it writes the job file and prints the two commands that build and queue it.
   - A change forces (the source, a new owner box, the shots record, the tracker, part or pose code):
     `preview --remake`, which takes the next free letter, then every stage below.
2. **Job**: `bench/shot_to_load.py job` (the command: [`PROTOCOL.md`](PROTOCOL.md) 2.2).
   - Reads: the preview's job folder and the files it saved, `plan.json`, the cast file and the span words.
   - Writes: the load's job file `<folder>/<load>.json`, which fixes the load's capture folder (`data/<day>_<load>`)
     and its reference folder (`bench/shot_to_load.py::reference_folder`: the span's plain folder until a queued graph
     has read it or it is not of today's form, then the next of `bench/shot_to_load.py::REFERENCE_SUFFIXES`). It also
     cuts the files the load reads into the preview's folders on the share, named by the preview
     (`bench/shot_to_load.py::PARTS_AT`, `KEEP_OUTS_AT`, `TRACKS_AT`): the load's own part cut, the part held to the
     track, the keep-outs, the stay, the `--track-not` copies and the strangers file.
   - **The preview's files are made once, for every load built on it.** The strangers file is made only when it is
     missing (`bench/shot_to_load.py::run_strangers`), so a change to `bench/strangers.py` does not reach a
     `job --remake`, and nothing says so. A cut that `job` would make differently now is refused as "of another
     making", to be moved aside by a person. A change to the code that makes these files forces a fresh preview.
   - A change forces (the job step's code, a plan field `job` reads, a span word): a fresh preview by default (above),
     and at the least `job --remake`, then every stage below.
3. **Reference**: `bench/shot_to_load.py reference <job file>` (the command: [`PROTOCOL.md`](PROTOCOL.md) 2.3).
   - Reads: the source, with the preview's track as the box for the pose pass; the body-model video; the class map
     when the mouth is `auto`; the head track; the job's reference form and span words
     (`bench/shot_to_load.py::drawn_form`).
   - Runs [`bench/pose_along_shot.py`](../../bench/pose_along_shot.py) for the pose points, then
     [`bench/head_reference.py`](../../bench/head_reference.py), which crops
     ([`bench/crop_reference_to_head.py`](../../bench/crop_reference_to_head.py)) and draws
     ([`bench/draw_face_points.py`](../../bench/draw_face_points.py)).
   - Writes, in the folder `job` named: the points (`<subject>_<first>_<last>_points.npz`), `<Video 1>`, the plain cut,
     the record and the sheet. It also writes into the job file: `job.reference`, which names the points, and the
     load's `motion_file`.
   - **The pose points have two readers**: the drawing, and the capture, as its `off_head` (`points=`) and the gate's
     `--chin-points` (`bench/load_chain.py::mask_specs`, `bench/load_chain.py::told_to_the_gate`). The capture's
     `pose=` is the preview's own pose table, not these.
   - Where `job` gave the new load a folder that already holds a reference, `reference` refuses, and
     `reference --existing` takes the old drawing as it is: held to its frames, its form and the span words, not to
     the code that drew it. Once a queued graph has read the folder, the new load gets a new folder, and the pose pass
     and the drawing run again on today's code: a second change in an A/B.
   - A change forces (the pose, crop or drawing code, the reference form, a span word the drawing reads): a fresh
     preview by default (above), and at the least `job --remake`, then every stage below.
4. **Text**: `bench/shot_to_load.py text <job file> --plan data/<video>/plan.json` (the command:
   [`PROTOCOL.md`](PROTOCOL.md) 2.3).
   - Reads: the reference's record ([`bench/load_text.py`](../../bench/load_text.py)), `plan.json` (the subject's
     words, clothing and voice by span), and the voice tables the video's manifest names, found by the source's file
     name ([`bench/clip_analysis.py`](../../bench/clip_analysis.py)).
   - Writes: `<load>.txt` beside the job file, and `texts.<label>` (`by`, `generated_from`) in the job file. A text is
     written once: a different text for the same load is refused, and so is any text once the plan is built.
   - **The text is the capture's parent.** Each `load_chain.py` step refuses a load whose text is not written
     (`bench/load_chain.py::read_job`), and the gate is handed the text (`preflight --text`).
   - A change forces (`bench/load_text.py`, the plan's words, clothing or voice, the voice tables): a fresh preview by
     default (above), and at the least `job --remake`, then every stage below.
5. **Capture and plan**: `bench/load_chain.py first`, `plan`, `on-plan`, `table` (the commands:
   [`PROTOCOL.md`](PROTOCOL.md) 3.1).
   - `first` reads the preview's files (track, head track, parts, class maps, shot tables, pose tables), the job's
     keep-outs and span words, the reference's points, the text and the voice table. It writes the capture
     `data/<day>_<load>/`, `<load>.who_is_who.json` and the first strip, and runs verify, the gate and the strip.
   - **One span word, two readers.** `--no-face-on` goes to the drawing and to the capture (`noface=`), where a
     hair-and-face part loses only its face classes on those frames and a face-only part is emptied
     (`bench/capture_masked_run.py::grade_holds`). `--drop` goes to the capture (`drop=`) and to the gate's
     `--named-empty`. `--hold-part` goes to the capture (`word=`).
   - `plan` is a run on the server: it builds and queues the plan graph and writes `<load>.plan.json`, `job.plan.json`,
     `<load>.plan.queued.json`, the band file on a hair-band load, and the node's plan of the windows on the share.
     `on-plan` copies the capture to `<capture>_first`, builds it again in place on the node's plan, writes
     `<load>.plan.masks.json`, makes the band again and writes the frame table, which `table` can also make again.
   - **The zoom is the node's**: it is set from the region's boxes at plan and render (`video_mask.py::zoom_box`).
     The capture works it out again for its part floor (`bench/capture_masked_run.py::floor_zoom`), which scales
     `bench/capture_masked_run.py::PART_LEAST_PX`. A change to a mask moves the zoom, the floor, and which frames are
     emptied.
   - **An override is a parent too.** `plan` refuses a blocked gate. A person clears a flag by name
     (`bench/capture_masked_run.py outcome <capture> <flag> overridden --by WHO --note WHY`), which records it
     in `outcomes.json` and writes the gate's verdict in `flags.json` again; `plan` and `render` read that verdict. An
     override written before `plan` or `render` lets that step run on this load; one changed after the step ran is a new
     load (`job --remake`). An override is a person's act after a look at the frames, never a loop in a script
     (decisions.md, 2026-10-21).
   - A change forces (`bench/capture_masked_run.py`, [`bench/head_tube_parts.py`](../../bench/head_tube_parts.py),
     [`bench/frame_table.py`](../../bench/frame_table.py), a span word the capture reads): a fresh preview by default
     (above), and at the least `job --remake`, then every stage below. These run per load, so a new load gets a new
     capture.
6. **Look, render, land and page**: a look by a session that did not make the capture
   (`bench/capture_masked_run.py looked`), or the maker's `bench/load_chain.py no-look --not-first FILE` where no
   forcing rule stands; then `load_chain.py render`, `land` and `page` (the commands: [`PROTOCOL.md`](PROTOCOL.md) 3.3
   and 5.1).
   - `render` reads the look, the gate, the frame table, the job file as planned and every mask the plan read. It
     writes the go beside the job file and the render under the job's output folder on the share.
   - `land` writes the render's points beside the capture, the stacked clip under `Video/<video>/review_clips/`, and
     the clip's record under `<capture>/strips/`. `page` adds the clip to `data/<video>/review/items.json`.
7. **Laid, and the cut**: `bench/laid.py`, then one `bench/rebuild_delivery.py --video <video> --clip` for a batch
   ([`PROTOCOL.md`](PROTOCOL.md) 5.3). Laying a piece is a person's act. The cut is pass 2's source (stage 0), so a
   change to a kept piece of pass 1 forces a new cut, and pass 2 made again on it from a fresh preview.

### 9. Frames to look at, and a subject who leaves the frame (`second_hfp_5889e`)

- **Frames to look at are picked where something changes, not evenly spaced** (the owner, 2026-10-21;
  [`docs/wiki/decisions.md`](../wiki/decisions.md)). [`bench/pick_frames.py`](../../bench/pick_frames.py) (with
  `bench/_lib/pick_frames.py`) lists a span's frames where a recorded signal or the picture changes, keeps one a second
  as a floor, and never drops what was written down or an island. Past those, it picks the strongest change of each
  second and lists the rest under `also_changed_not_picked`, with their scores. The reference sheets choose their frames
  with it (01eb9031, which replaced sheets of a fixed count of evenly spaced frames). Evenly spaced samples had missed
  the frames on `second_hfp_5889e` where the second subject leaves the picture. Watch the continuous clip as well.
- **A subject who leaves the frame before the shot ends is not a fault to trim** (the owner, on playback, 2026-10-21).
  `second_hfp_5889e` was right as rendered: after the subject leaves, the render equals the source, as it should. It is laid on
  its kept span (`data/fun/laid.jsonl`), and the assembler's `ends` failure there read a correct tail as a fault
  (decisions.md, 2026-10-21).
