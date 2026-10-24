# docs/masked_v2v/: the masked video-to-video lane

Loaded when a session opens a file here; [`docs/AGENTS.md`](../AGENTS.md) and the root [`AGENTS.md`](../../AGENTS.md) still govern.

- **Start here**:
  - How the masked render path works today, and what to simplify (Gemini, 2026-10-21; local, not in git; one claim
    corrected in its mailbox's C61: chained shots are live): [`internal/gemini/2026-10-21_render_path_report.md`](../../internal/gemini/2026-10-21_render_path_report.md)
  - Execution & debugging protocol: [`docs/masked_v2v/PROTOCOL.md`](PROTOCOL.md)
  - Verifying raw frames vs. `.npz` / JSON records (`laid.py keeps`, zoom boxes, and reference crops): [`docs/masked_v2v/verifying_raw_frames_and_keeps.md`](verifying_raw_frames_and_keeps.md)
  - The frame math (window lengths, the VAE's 17-frame clips, padding after the shot, what the text encoder samples): [`docs/masked_v2v/frames_rules.md`](frames_rules.md)
  - Lane rules and where state lives: [`data/AGENTS.md`](../../data/AGENTS.md) (and each project's untracked `data/<video>/AGENTS.md`)
  - End-to-end pipeline, design, and card runbook: [`docs/wiki/masked_v2v_pipeline.md`](../wiki/masked_v2v_pipeline.md), [`docs/wiki/masked_v2v.md`](../wiki/masked_v2v.md), and [`docs/wiki/runbook.md`](../wiki/runbook.md)
- **This lane is one use of H3.** A finding about common code (reference conditioning, the shared nodes) is written where that code's page is and linked from [`docs/wiki/index.md`](../wiki/index.md).

## The owner's rules for passes, prep and judging (2026-10-21)

The owner's words of 2026-10-21, each with its reasons in [`docs/wiki/decisions.md`](../wiki/decisions.md) on that date. Where a tool does not enforce a rule yet, the rule still holds and the session follows it by hand; the build that enforces it is named in `decisions.md`.

**Passes and the chain**
- **Every replacement is a pass.** A replacement is anything H3 makes in a hole: a person's face, her hair, an object such as a table. H3 makes one replacement a render, so on a shot with several, each pass is made on the newest whole-video cut that holds every other replacement laid there. Each replacement adds a link to that chain. No tool names a subject in this rule: it reads the labels in `shots.json`, `plan.json` and `laid.jsonl`.
- **A kept piece stays** unless something broken changes the composite, which is the owner's look to say (§1).
- **A re-render is not a pass.** Redoing a replacement that is alone on its shot, or the last pass with nothing laid on top of it, is one fresh preview against its source and one render. Redoing pass N of a chain breaks every pass made on its output, and those are made again, in order, on the new chain. Which cut a piece was made on is read from its render graph's source (`bench/what_h3_got.py`); no tool records the chain or flags a stale one yet. A stale kept piece is shown to the owner, not remade by rule.
- **A shot is done when every subject who can be swapped there is kept.** A subject with nothing to replace is recorded `not seen` with [`bench/shots_look.py`](../../bench/shots_look.py), on the owner's word or a look, and owes no load (§1).

**Prep**
- **Prep is not reused between loads, at any stage of the chain** (§8). Each load makes its own preview, strangers file, job, reference, text and capture; only the owner's own words carry (the cast boxes and `plan.json`). A fresh preview costs little card time next to a render made on stale prep.
- **Any render with a dependency gets its whole prep made fresh** on today's chain.
- **The one exception**, a one-variable test of a single replacement with nothing else laid on its shot, may keep its own preview, with a written reason in the job. It holds only where no code that feeds that stage has changed since it ran (masks, the part and pose models, SAM tracking, the drawing), and the window and its start and end frames have not moved. When in doubt, regenerate.
- **Moving a render's true start or end frame invalidates its prep**, unless the prep was made upfront for the whole video and the render is a single pass.

**Windows**
- **A render window is only a length on both clocks** (`h3_rules.on_both_clocks`). Where a shot is shorter than its window, the extra frames go after it (before it only at the clip's end; the owner, 2026-10-22, "just add buffer frames after", §6). By the owner's ruling of 2026-10-25 ([`docs/research/masking/2026-10-25_context_frames_and_identity_on_2050.md`](../research/masking/2026-10-25_context_frames_and_identity_on_2050.md)), the original subject is never shown unmasked in a window: where the shot's own footage continues past the piece, the load runs to the window's length masked end to end and the extra frames are chopped off at the lay; across a cut, the padding frames are held copies of the shot's own first or last frame with its region open, never the other shot's frames (2026-10-22, the owner's "seam problem"; §6), unless its edge frame's region is empty or under `video_mask.HELD_EDGE_LEAST` of the shot's middle region (`held_fell_back`, §6). This reverses the 2026-10-21 "as filmed" unmasked context default and the "best practice, not a rule" of [`docs/h3_resolutions.md`](../h3_resolutions.md), "Whole on the audio clock".

**Judging a render**
- **A render is measured against the change being made, per subject.** A part the plan replaces for that subject (`plan.json` `subjects.<s>.parts`) is compared to the subject's reference still, never to the source: dark hair that matches the original is a fault for a subject who replaces her hair, and right for one who replaces only the face. A part not replaced is compared to the source.
- **Tools show what they see:** the raw figures and frames, with what they were measured on and what they cannot see, and no rule's verdict. A tool may stop before a step that cannot be undone. A look or override the tool requires is signed by the lead on the owner's word, and its note says "not inspected". No session looks at a render before the owner does; a plan strip may be sanity-checked (decisions.md, 2026-10-22).
- **A look at one shot assembles and stacks that span alone**, or uses the load's own stacked clip where nobody else is laid on the shot. The whole-video rebuild runs only when the cut and pages must move, once a batch ([`PROTOCOL.md`](PROTOCOL.md) 5.3).

## The rules that bite when missed

### 1. What `--source` to pass in, and which subject is made first on a shot of two

- [`bench/shot_to_load.py`](../../bench/shot_to_load.py) `preview` (`bench/shot_to_load.py::step_preview`) requires `--source VIDEO`, a path, with **no silent default**. Never assume `data/<video>/shots.json`'s original raw video is the right source for the pass in hand.
- **A kept subject stays** (the owner, 2026-10-21; [`docs/wiki/decisions.md`](../wiki/decisions.md)). A subject already kept on the shot is not remade to make the other, unless something broken changes the composite, which is the owner's look to say. The assembler's `two_pieces_change_the_same_pixels` flag ([`bench/assemble_delivery.py`](../../bench/assemble_delivery.py)) names the frames where two pieces change the same pixels: a place to look, not proof that the composite is broken. No subject goes first by rule: the kept one is pass 1.
- **A single-subject shot, or pass 1 of a shot of two**: pass the video's working copy, the file `data/<video>/manifest.json` registers (`bench/shot_to_load.py::working_copy` maps a record's copy to it).
- **Pass 2, where the other subject has something to replace.** A shot is done when every subject who can be swapped there is kept (the owner, 2026-10-21; [`docs/wiki/decisions.md`](../wiki/decisions.md)). A subject with nothing to replace, such as one whose face never shows, is recorded as `not seen` with [`bench/shots_look.py`](../../bench/shots_look.py) on the owner's word or a look, and `bench/shot_coverage.py::nothing_to_make` reads that line, so no load is owed for that subject. Otherwise the other subject is made on the newest cut that holds the kept one. Pass 1's output is pass 2's source, and pass 2's whole chain is made again on it, starting from a fresh `bench/shot_to_load.py preview --source <that cut>`: no track, mask, keep-out, reference or text is reused from a preview of the original video. Only the owner's own words carry: the cast boxes (`data/<video>/cast/cast.jsonl`) and `data/<video>/plan.json`. The cut is one of the whole video (`bench/rebuild_delivery.py --video <video> --clip`): a load finds its frames by number from the start of its source (`workflows/masked_job_graph.py::seek`), and a cut of one span begins on that span's first frame, so it is for looking at, not a source. Register the cut as [`PROTOCOL.md`](PROTOCOL.md) 1.1 says. `bench/shot_to_load.py::check_pass2_source` refuses the raw source on `preview` and `job` where another replacement is laid on the frames (`--allow-raw-source WHY` lets it through), refuses a stale cut older than another subject's laid row (`bench/shot_to_load.py::check_pass2_source`), and refuses a source cut that already holds the load's own subject's piece on those frames (`bench/shot_to_load.py::_cut_pieces_on_frames`, `check_job_pass2_source`); it knows the raw source by its name and path, and the replacements by the labels `shots.json`, `laid.jsonl`, and `<cut>.json` hold. `preview` is told the replacement it is made for (`--for LABEL`, repeatable; required where anything is laid on the shot) and checks both that the other subjects' laid pieces are in the cut and that `--for`'s own subject is not already in the cut, so a preview made to redo a laid piece takes the raw source where nothing else is laid; `job` checks every label against the load's subject.
- **Why (Kitchen `604-1009` `second` + `lead` crossing on `755-774`, 2026-10-19)**: The original 1980s mom (`second`) had a tall dark bouffant hairdo rising ~60 px higher (`ymin=188..209`) than the replaced blonde mom (`second_hfp_0604h`, `ymin=211..221`). Reusing a preview segmented on the raw 1980s video gave `second`'s keep-out `5,843–9,276 px` of old bouffant hair dome and carved an invisible dark helmet out of `lead`'s face as `lead` walked behind her.

### 2. Bidirectional Temporal Attention & Unmasked Context Leaks (The Beard Rule)

- H3's DiT uses full bidirectional temporal attention across the entire window (`W` frames, `5n + 2` latent steps for `17n + 5` frames). Even 3–4 frames where an original subject's salient feature (such as a gray beard or eyeglasses) is left unmasked in the frozen context will propagate that feature onto the regenerated subject across the entire window.
- **Two causes caught on Hallway `1306-1418` (`third`, 2026-10-19)**:
  1. **Transient Sapiens class-map dips (`<= 6` frames) during fast motion**: on `1402-1404` and `1408-1411`, Sapiens `Face_Neck` dipped below 50% of flanking frames while the person track stayed solid (`51,735 px` of exposed beard). Fixed automatically by `bench/head_tube_parts.py::bridge_short_tube_dips` (`bench/head_tube_parts.py::SHORT_DIP_MAX_FRAMES` = `6`, `bench/head_tube_parts.py::SHORT_DIP_RATIO` = `0.6`).
  2. **Dropping interior frames where the face is still visible**: `--drop 1411-1413` zeroed the regeneration mask on frames that still had `16,409–48,938 px` of visible `Face_Neck` + beard. Guarded by `bench/capture_masked_run.py::dropped_or_unmasked_face_in_context` preflight blocking flag `dropped_frame_has_visible_face` (`bench/capture_masked_run.py::DROPPED_FACE_MIN_PX` = `4000`). **Never `--drop` a frame where the subject's face is still visible to the camera.** On back- or side-of-head turns, pass `--no-face-on` to [`bench/shot_to_load.py`](../../bench/shot_to_load.py) `job` instead: `<Video 1>` draws no face lines there, and on a hair-and-face part the capture drops only the face classes from her part, the mask the load carries, on those frames, so her hair stays in (`bench/capture_masked_run.py::grade_holds`). The node grows the region from that part, so its margin may still cover some of the face. A face-only part is emptied on those frames. Where the subject's face is still in the capture's hole (`parts_held` stays open) and only `<Video 1>`'s 2D face marks or 3D grey head fit is wrong on a short span, use `--no-draw-face-on FIRST-LAST=WHY` (suppresses `<Video 1>`'s 2D face lines while keeping the grey head and the capture's face mask) or `--no-head-on FIRST-LAST=WHY [--no-head-form hold|blank]` (`bench/head_reference.py`: holds the last good cut body-model frame before the span or blanks `<Video 1>` with no face lines drawn, while leaving the capture and regeneration mask untouched).
- **A third way in: the context frames of a short load** (2026-10-22; updated 2026-10-25, see [`docs/research/masking/2026-10-25_context_frames_and_identity_on_2050.md`](../research/masking/2026-10-25_context_frames_and_identity_on_2050.md)). A shot shorter than its window is padded with context frames after it (§6): real source frames where the shot goes on, on which by default nothing is regenerated, so the original's beard or face is in the window (`lead_fp_2050c` had 21 unmasked singer frames on `2272-2292` and `lead_fp_2050e` had 45 on `2095-2139`, causing H3 to re-synthesise the original singer's face instead of `<Picture 1>`); across a cut, held copies of the shot's edge frame with its region open. By the owner's ruling of 2026-10-25 ([`docs/research/masking/2026-10-25_context_frames_and_identity_on_2050.md`](../research/masking/2026-10-25_context_frames_and_identity_on_2050.md)), the original subject is never shown unmasked in a window: where the shot's footage continues past the load, the load runs to the legal window length with every frame masked end to end and the extra frames chopped off at the lay; across a hard cut, held copies of the edge frame with its mask open stay. (Note: `context: "face covered"` via `video_mask.pad_context` only stamps a static copy of the shot's first/last mask onto moving context frames, leaving `41–59%` of a moving face uncovered on `2095-2139`, so end-to-end masked windows or held edge copies are required.) No gate reads unmasked context frames outside the capture's span: `bench/capture_masked_run.py::dropped_or_unmasked_face_in_context` reads the capture's own frames.
- **A fourth way in: under the jaw** (2026-10-23). The head tube cuts a face-only part to the SAM 3.1 head track, which ends at the jaw: the head track is widened only by a piece of the part that fits within its rows (`bench/head_tube_parts.py::head_tube_for_capture`), and his face, beard and neck are one piece of the part that runs below them, so the beard is cut even where the preview's part holds it (third_fp_5011b: 986-1607 px a frame on 12 of the 15 frames where a reader saw the goatee below the region). For a face-only subject whose source shows facial hair to replace, the plan says `source_facial_hair: true` with `source_facial_hair_by` (`bench/shot_to_load.py::source_facial_hair_of`); `job` writes `facial_hair: keep` on the subject, the chain passes `facial_hair=keep` to the capture, and the head tube gives back the piece of the part joined to the face that it cut, inside his person track and the face's columns (`FACE_COLUMN_PAD_PX` past them), less every other person's mask, each px with its own class; no pose point is read, so a head turned away keeps it too (`bench/head_tube_parts.py::face_piece_kept_whole`; per frame, with what the columns cut, in `head_tube__<by>.json`, `source_facial_hair`). It cannot see facial hair the part model calls clothing or Hair (5011b's 5020-5022 under his arm, 5461b's 5463-5464: Upper_Clothing; Hair is left out, for the reasons in `FACIAL_HAIR_CANNOT_SEE`), and a hand the part model calls Face_Neck where it touches the face is kept inside the face's columns. Without the word nothing changes.

### 3. Always pass `--cast data/<video>/cast/cast.jsonl` and verify it exists

- [`bench/shot_to_load.py`](../../bench/shot_to_load.py) `preview` and `job` refuse missing `--cast` files cleanly (`bench/shot_to_load.py::Refused`). `preview` defaults `--cast` to `data/<video>/cast/cast.jsonl` when omitted (`bench/shot_to_load.py::step_preview`). Without the cast file, the builder cannot seed SAM 3.1 from confirmed boxes or auto-identify `lead` vs `second` vs `third` from confirmed reference embeddings.

### 4. Pre-render & Post-capture Telemetry (`facts_pass.py`, `flags_before_render.py`, `step_carves.json`, `code_stamp.py`)

- Run [`bench/facts_pass.py`](../../bench/facts_pass.py) on the capture directory to inspect `face_outside_region_px` (catches chin/cheek cutouts where `(s_face | (head_track & ~s_hair))` falls outside `region`), `other_overlap_px`, and `mouth_mode` before spending GPU render minutes.
- Run [`bench/flags_before_render.py`](../../bench/flags_before_render.py) `data/<day>_<load> --initial <session>` to print and sign off the four diagnostic blocks (`flags.json`, `marks_against_the_body_model`, `runs/<run>/step_carves.json`, and held/emptied frames); `bench/load_chain.py::the_flags_read` refuses `render` unless `<capture>/looks/<load>.flags_read.json` is recorded for the current `flags.json`.
- Inspect `runs/<run>/step_carves.json` (`bench/capture_masked_run.py::step_keepout_carves`) for latent-step keep-out cell carves touching the subject's head track or carried part.
- All derived `.npz` / `.json` artifacts carry `writer_stamp` (`bench/_lib/code_stamp.py::writer_stamp`, `bench/_lib/code_stamp.py::npz_writer_fields`). `bench/_lib/code_stamp.py::check_upstream_mtime` compares file times and returns warnings: [`bench/draw_face_points.py`](../../bench/draw_face_points.py) prints them, and [`bench/head_reference.py`](../../bench/head_reference.py) and [`bench/measure_moge2_depth.py`](../../bench/measure_moge2_depth.py) add them to their refusal to write over an output that is there. Nothing refuses a stale input on its own (§8).

### 5. Review Server (`items.json`), Verdicts (`owner_answers.jsonl`), and Static `poc/TO_REVIEW.html`

- The owner reviews clips and look questions and submits verdicts and frame notes on the interactive review page (`http://<host>:8765/`, served by `data/<video>/poc/serve.py` from `data/<video>/poc/index.html`), which reads `data/<video>/review/items.json` (`state: "open"` renders under `"1. Look at these next"`) and appends clicks to `data/<video>/owner_answers.jsonl`. Question cards (`kind: "look"`, built by `bench/load_chain.py::look_page_item` and rendered by `bench/review_pages.py`) show a still or short strip, the owner's chronological prior words on the shot from [`bench/said_on_shot.py`](../../bench/said_on_shot.py) (`--for-question`), and custom choice buttons.
- Every new review clip **must** be added to `data/<video>/review/items.json` (`state: "open"`, via `bench/load_chain.py page`) **and** synced to `data/<video>/poc/TO_REVIEW.html` ([`bench/rebuild_delivery.py`](../../bench/rebuild_delivery.py) rebuilds `TO_REVIEW.html` and `PROVENANCE.html` and verifies `data/<video>/poc/` on disk via `bench/rebuild_delivery.py::verify_poc_on_disk` during the `pages` step; [`bench/build_cut_map.py`](../../bench/build_cut_map.py) builds the 2-tab chronological cut table, frame-level gap audit, and 6-stage hierarchical data flow lineage DAG in `data/<video>/poc/PROVENANCE.html`, `data/<video>/poc/CUT_MAP.html`, and `data/<video>/CUT_MAP.md`).

### 6. 24.0 fps End-to-End, One Shot = One Render (`<= 345` Frames), Lengths on Both Clocks, and What Context/Overlap Frames Are Thrown Away

- **Exact `24.0` fps (`24/1`) everywhere**: Every working copy, preview, reference, render, and assembled multi-pass delivery cut ([`bench/rebuild_delivery.py`](../../bench/rebuild_delivery.py), [`bench/assemble_delivery.py`](../../bench/assemble_delivery.py)) is forced to `24.0` fps (`bench/shot_to_load.py::working_copy` and [`bench/masked_job.py`](../../bench/masked_job.py) refuse any non-24 fps source).
- **One Shot = One Prompt = One Render (Never Split a Shot `<= 345` Frames)**:
  - A cut must always align 1:1 with a shot, its prompt, and its render ([`docs/wiki/decisions.md`](../wiki/decisions.md), 2026-10-15, "One shot is never split"; 2026-10-21, windows on both clocks). Every shot of `<= 345` frames (`max(h3_rules.chain_lengths())`) renders as **one single-window H3 load** (`workflows/masked_job_graph.py::window_for`) on both clocks (`h3_rules.on_both_clocks`: `39, 90, 141, 192, 243, 294, 345`) with zero mid-shot seams.
  - **Interior occlusions stay inside the single whole-shot load**: When a foreground dancer or hand completely occludes a background subject for a few frames inside a shot (`track_share == 0`, `parts_share == 0`), **never** split the shot into two sub-loads before and after the occlusion. Keep the entire shot in one load, account for the occluded frames with `capture_masked_run.py verify --no-mask`, and let the node's composite leave those frames as the source (`left_as_the_source_with_no_mask`, `bench/capture_masked_run.py::verify_checks`) only where she is truly hidden, on the owner's word, marked `hidden` in the plan's `head_marks` (§10). A frame where she shows and the mask is empty is a fault to fix, never an occlusion to pass (decisions.md, 2026-10-22).
  - **The only exception (`N > 345` frames)**: When a single camera shot in `data/<video>/shots.json` exceeds `345` frames (`max(h3_rules.chain_lengths())`; in `fun`, only Kitchen `604–1009` at `406` frames and Guitar Solo `2050–2622` at `573` frames), it still runs as **one load** (`shot_to_load.py job [--seam ...]`), which the song node chains across windows on the dual video+audio clock grid (`h3_rules.chain_lengths`: `141, 192, 243, 294, 345`, `h3_rules.on_both_clocks`) with `context_frames = 90` (`workflows/h3_config.py::MASKED_JOB_CONTEXT`) overlap frames.
- **How Every Preprocessed Input Is Locked to `[shot.first .. shot.last]` (Why Starting Mid-Shot Breaks Tracking/Masking Alignment)**:
  1. **Preview (`shot_to_load.py preview --shot <first>`)**: Looks up the exact shot `[first, last]` (`N = last - first + 1` frames) in `data/<video>/shots.json`, seeds SAM 3.1 from the confirmed frame box inside `[first, last]` in `data/<video>/cast/cast.jsonl`, and writes every tracker and perception stream (`track`, `head_track`, `parts`, `classes`, `pose`, `mesh`) starting at frame `0 == shot.first` for `N` frames (`shot.first .. shot.last`). The `strangers` keep-out is made for the preview at the job step, where a load names it, on an empty queue with the card freed first (`bench/strangers.py`: its people file, every person SAM 3.1 finds on each frame as the preview's tracker asks, then the part model on the whole frame; a piece touching a subject is a stranger only where it lies on another person the detector found, the owner's rule of 2026-10-23, and since one file serves every subject's load, another subject's own detection is another person).
  2. **Head Reference & Pose (`shot_to_load.py reference`)**: Runs `pose_along_shot.py` and `head_reference.py` on `--first <shot.first> --count <N>` with `--mesh-first <shot.first>` and `--classes-first <shot.first>`, so `<Video 1>` (`<subject>_head_reference_<first>_<last>.mp4`) has frame `0 == shot.first` and exact length `N`.
  3. **Prompt & Voice (`shot_to_load.py text`)**: `load_text.py` reads the clothing span and vocal activity (`voice_spans.json`) over `[shot.first, shot.last]` and records `"frames": [shot.first, shot.last]` in the job JSON (`masked_job.py` refuses any prompt whose frame range does not match the load's `[first, last]`).
  4. **Hard Verification Gate (`capture_masked_run.py verify`)**: Before any `plan` or `render` can run, `verify` checks three hard assertions on every mask, part, class map, and keep-out file: `at_is_the_previews_first_frame` (`at == shot.first`), `inputs_reach_the_span` (every file covers `shot.first .. shot.last`), and `inputs_at_the_canvas` (`1024x768` at `24.0` fps).
  5. **A shot of two subjects**: both use the **same shot boundaries `[shot.first, shot.last]`**. Pass 1 is the subject already kept (§1), made on the working copy with the other subjects tracked on the same span as keep-outs (`others`). Pass 2 is made on the newest cut that holds pass 1, from a fresh preview of that cut (`--source <the cut>`) and every stage after it, so pass 1's new appearance is in the picture and pass 1's keep-out follows its new silhouette frame for frame.
- **How Data Inputs Align with H3's Both-Clocks Window (`h3_rules.on_both_clocks`) & What Context/Overlap Frames Are Thrown Away**:
  - **Single-window shot (`N <= 345` frames, `[first .. last]`)**:
    1. SAM `track`, held `parts`, `others` keep-outs, and `<Video 1>` `head_reference` are loaded at exact `24.0` fps starting at `first` for `N` frames (`workflows/masked_job_graph.py::load`).
    2. `window_for(N)` chooses the shortest length `W` on **both clocks** (`h3_rules.on_both_clocks`: `39, 90, 141, 192, 243, 294, 345`) that holds `N` (`W >= N`; e.g., `N = 35 -> W = 39`, `N = 66 -> W = 90`, `N = 100 -> W = 141`).
    3. When `N < W`, padding goes after `[first .. last]` (`before = 0`, `after = W - N` in `workflows/masked_job_graph.py::window_pad`; `before > 0` only at the source clip's end; why: the video VAE encodes a window in 17-frame clips, `comfy/ldm/minimax/vae.py` `clip_length=17` and `encode_temporal`, and with 4 and 4 on `second_hfp_4479e` the shot's last frame sat alone at the start of the last clip with 16 frames of the next shot, the bright square). **By the owner's 2026-10-25 three-path ruling** ([`docs/research/masking/2026-10-25_context_frames_and_identity_on_2050.md`](../research/masking/2026-10-25_context_frames_and_identity_on_2050.md)), **the original subject is never shown unmasked in a window**:
       - **Where the footage continues past the piece** (e.g., `2050`'s mirror spin): leaving real padding frames unmasked (`as filmed` in `video_mask.py::MiniMaxH3MaskedSource.execute`, or `face covered` in `video_mask.py::pad_context`, which only repeats a static copy of the edge frame's mask and leaves `41–59%` of a moving face uncovered on `2095-2139`) exposes the original subject in the window (`lead_fp_2050c`, `lead_fp_2050e`). Instead, the load runs to the legal window length `W` masked end to end (`pad_after == 0`), and the extra denoised frames past the piece are chopped off at the lay via `laid.jsonl`'s `frames` field (`bench/laid.py`).
       - **Across a hard cut into another shot** (2026-10-22, the owner's "seam problem" on `second_hfp_4479c`): `workflows/masked_job_graph.py::held_context` counts, side by side, the context frames outside `shot_span` (`held_before`, `held_after`); the node puts copies of the shot's edge frame there with its masks (`video_mask.hold_context`), and the song node silences the track under them (`audio_freeze_song.silenced`), unless the shot's edge frame on that side has no region or one under `video_mask.HELD_EDGE_LEAST` (`0.25`) of the shot's middle region (`video_mask.py::held_edge_falls_back`, the `4912e` case, and `2050` at `2247` where `4,578 px < 5,457 px`), when that side falls back to real frames (`held_fell_back`). Every padded window's region file and plan row (`context_before_held`, `context_after_held`) say how many were held. The zoom box (`video_mask.py::zoomed_source`, `zoom_box`) is read from the region's boxes over the whole window. After VAE decode, `audio_freeze_song.py::MiniMaxH3AudioFreezeSong.execute` **throws away any `before` and `after` padding frames** (`trim = pad_before`, `kept_frames = total - pad_after`), and `laid.jsonl`'s `frames` field drops any extra denoised frames past the piece at the lay.
  - **Multi-window chained shot (`N > 345` frames, e.g. Kitchen `606–998`, `393` frames on `[345, 141]` with `context_frames = 90`)**:
    1. Window 1 generates `W_1` frames (`345`).
    2. Window 2 overlaps Window 1 by taking the last `90` frames of Window 1 as latent context (`context_frames = 90`), and generates `W_2` frames (`141` = `90` overlap context + `51` new frames).
    3. The stitcher **throws away the `90` overlap context frames from Window 2** plus any tail frames past `N`, emitting exact `N` frames (`393` frames = `606–998`) with zero audio or visual seam.
- **Historical context word in the code (`context_shows`, reversed on 2026-10-25)**: `workflows/masked_job_graph.py` and `video_mask.py::pad_context` still carry `as filmed` (zeros on real padding frames) and `face covered` (static copy of the shot's first/last mask expanded across real padding frames). By the 2026-10-25 ruling ([`docs/research/masking/2026-10-25_context_frames_and_identity_on_2050.md`](../research/masking/2026-10-25_context_frames_and_identity_on_2050.md)), neither is sufficient when the original subject remains in continuous padding frames (`as filmed` leaves 100% of the face unmasked; `face covered`'s static mask leaves `41–59%` of a moving face unmasked), so continuous footage runs masked end-to-end to the window length `W` and is chopped at the lay.
- **No stale pose or mask data**: never reuse masks, SAM tracks, Sapiens2 part or class maps, or pose files across a change to what made them. What each stage reads and writes, what a change forces, and what no tool enforces: §8.

### 7. Head Reference Mouth Drawing & Unified `<Subject 2>` Prompt (Closed = 3D Gray Mesh Lips; Open = Black Interior + White Outline)

- **No mouth smoothing (`smooth_mouth_area` retired, 2026-10-20)**: `bench/draw_face_points.py::smooth_mouth_area` is no longer called on the drawing path (still defined, and exercised only by `bench/check_draw_face_points.py`). Painting a flat gray oval over the mouth region created a bright gray splotch across the lips, chin, and hands (`second_hfp_0415e`, `second_hfp_0484d`, `second_hfp_1039b`) that H3 read as an open mouth.
- **Closed-mouth frames stay as natural 3D gray mesh lips (`mouths[f] = None`)** by default: Whenever Sapiens2 (`sapiens2-seg-1b` + `sapiens2-pose-1b`) detects a closed mouth (`(not has_hole) or (gap <= INNER_GAP_CLOSED_PX)`, `bench/draw_face_points.py::INNER_GAP_CLOSED_PX`), `bench/draw_face_points.py::run` sets `mouths[f] = None` and draws no mouth there, leaving the 3D gray mesh's sculpted closed lips untouched, unless the subject's plan says `reference.closed_mouth: "outlined"` (below).
- **Open-mouth frames draw black interior + white outline (`inner_lip_ring` / `mouth_inner`)**: Whenever Sapiens2 detects an open mouth (a teeth/tongue hole in the 29-class map and an inner-lip gap over `bench/draw_face_points.py::INNER_GAP_CLOSED_PX`), the black interior + white outline is drawn at the mouth, and `bench/draw_face_points.py::fit_mouth_to_head` keeps the drawn mouth inside the gray head's mouth-to-chin space (its docstring says how).
- **An open mouth's form is the plan's, a subject at a time** (`reference.open_mouth`, `bench/draw_face_points.py::OPEN_MOUTH_FORMS`; the owner, 2026-10-22, on lead_fp_1156d). Without it, `inner outline`: the whole inner-lip ring filled black. The lead's is `outlined`: the lips' outer outline and the inner one round a dark inside, placed on the Sapiens2 class map's lips (not the pose points, and not above them), the whole mouth no wider than `bench/draw_face_points.py::MOUTH_WIDTH_MOST` and no more open than `MOUTH_OPENING_MOST` of the upper face (glabella to the bottom of the nose, so a close-up and a far shot keep one proportion), the dark inside at most `MOUTH_OPEN_SHARE_MOST` of the outer lips, the body model's lips painted flat exactly inside the outer outline (`anchored_mouth`, `lips_paint_out`). A frame whose class map holds no lips takes its nearest neighbour's. `capped` and `outlined over the lips` are built, no subject's form. Closed frames are the same in every open form. The measures behind each limit are in the constants' comments.
- **A closed mouth's form is the plan's too, a subject at a time** (`reference.closed_mouth` with the reference's `why`, `bench/draw_face_points.py::CLOSED_MOUTH_FORMS`; the owner, 2026-10-23, on second_hfp_5114i: where the closed-lip line called every frame closed nothing was drawn at the mouth and the text named none, and the second subject took the still's smile, "If we dont have a drawing for mouth expressions or facial expressions like that we probably do need one"). Without it, `sculpted`: nothing drawn, as above. `outlined`: on each frame the line calls closed, the lips' outer outline and the line where they meet (`anchored_closed_mouth`), placed on the class map's lips by the outlined open mouth's own move (`onto_the_lips`, so not on the pose points, the 0278d/e/h fault), no wider than `MOUTH_WIDTH_MOST` nor taller than `MOUTH_CLOSED_HEIGHT_MOST` of the upper face, the body model's lips painted flat inside the outline (`lips_paint_out`), the meeting line with no dark edge (`CLOSED_MOUTHS`). It stays on the class map's lips: it is not fitted to the grey head (the build's blind read of 2026-10-23: the open form's `fit_mouth_shapes_to_head` moved it to a line under the nose or onto the cheek on the profile shots 4479-4509 and 5114-5129, the 0278 fault), only cut, as every mark is, to the head's silhouette above the chin `bench/draw_face_points.py::find_mesh_chin_y` finds; a frame where none of it is left is not drawn and the record lists it (`frames_not_drawn_nothing_on_the_head`). Under the same word every drawn mouth, open or closed, takes the source's expression as a bend (`bench/draw_face_points.py::MOUTH_BEND`, `bend_mouth`; the owner, 2026-10-23): its corners drawn above its middle (a smile) or below (a frown) by the corners' lift on that frame (`corner_lift`, on the pose model's points read at the canvas), flat inside a dead zone, and no bend at all where the corners are not read (a mouth narrower than `least_width_px` at the canvas, or a corner under `least_corner_score`: on 4479-4509 and 5114-5129 the figure was noise, on 0014-0167 it followed raised corners; the constant's comment has the measures). The pose points set only the bend: every point keeps its place across, the middle and the thickness stay where the class map's lips put them, an open mouth is bent after its fit to the grey head and its dark inside keeps its height; open frames are otherwise drawn in their own form. It reaches the drawing as `head_reference --closed-mouth` (`bench/shot_to_load.py::drawn_form`, under the closed-lip line only), and the record's `drawn.mouth.closed_lip_line.closed_mouth` lists `frames_drawn_closed` (only frames where some of it is left) with each frame's drawn size over the upper face, its box against the class map lips' box and how far the fit would have moved it, and under `bend` each frame's reading, the frames bent and how far the bend moved each. On a silent window (mouth `left out`) nothing is drawn at the mouth and the record says so. Whether it holds her mouth as the source's is a render's to decide. On a turned head `find_mesh_chin_y` can find the grey head's edge just under the nose, and there the cut leaves nothing (5115-5117 and 5120-5124 of 5114 in the build's proof).
- **A lowered eye is drawn with a dark iris, and a closed eye is painted over** (2026-10-22; the iris arc between the lids from 2026-10-20). Under `--iris contrast` with the closed lip line, the chain's default form (`bench/shot_to_load.py::REFERENCE_DEFAULT`), an eye whose iris ring lies `bench/draw_face_points.py::DOWNCAST_RING_OUTSIDE_MIN` or more outside its lids is drawn dark with no person's word: its lash ring, eyelid and crease lines, no white sclera, and the iris a solid black disc clipped to the lids, so a narrow eye shows a dark iris where the pose model puts it (`dark_irises` in [`bench/draw_face_points.py`](../../bench/draw_face_points.py)); an eye under the share keeps the white and black contrast form. A dark iris is drawn in lids no thinner than `bench/draw_face_points.py::EYE_OPEN` (its `least`, the lash ring's area over the eye's width squared, measured on frames whose gaze rendered right): where the pose model's lids are thinner, the drawn lower lid is opened down to it, the corners and the upper lid kept (`open_lids`), so a piece of the black iris always shows (after 0278k, a second seed, looked up as 0278j did); the record lists the frames (`drawn.eye_rings.frames_the_lids_are_opened_on`). The lid opens downward (`EYE_OPEN`'s `way`, the owner's choice of 2026-10-22). The cost: an eye shut in the source is drawn open too; name a shut stretch with `--eyes-shut-on ... --eyes-shut-form closed`. On 0278m the gaze held down, then opened as her head lifted; that is parked, not solved (decisions.md, 2026-10-22). The reference record says, eye by eye, which frames were drawn dark (`drawn.eye_rings.frames_an_iris_is_drawn_dark_on`). A person's word (`job --eyes-shut-on`, with `--eyes-shut-form` from `bench/draw_face_points.py::EYES_SHUT_FORMS`) draws its frames `none`, `lids`, `lowered` (every eye dark, whatever the share) or `closed` (the body model's whole sculpted eye, found on its frame round the lash ring by `bench/draw_face_points.py::sculpted_eye` and `SCULPTED_EYE`, painted flat in the skin grey round it, hard-edged and never past the head's edge, then one white line along the upper lash line, no iris; the brows stay). The text has no words of its own for `lowered` or `closed` (`lids` and `none` each add a clause): `<Subject 2>` says "the eyes, each with a dark iris, open, close and look where the person's eyes do" (`bench/load_text.py::FACE_EYES`), which the dark iris on downcast and `lowered` frames makes true there too, where the iris arc it replaced had no black. Drawing lowered eyes shut by rule was tried on 2026-10-21 and taken back out the same day: H3 drew her eyes open and toward the camera on second_hfp_0484g and second_hfp_1039e ([`docs/wiki/decisions.md`](../wiki/decisions.md)).
- **The reference is cut to the head's box, or to the face's or the grey head's on the plan's word** (`reference.crop`, `bench/head_reference.py::CROP_FORMS`; 2026-10-22, after 0278j/k looked the wrong way on the head box where 0278f, on the face's, did not). `head`, the default since `ba061ef3`, holds the head track with room above the crown and the faces, so her whole head and hair are in `<Video 1>` and the face is drawn smaller. `face` sizes the one box on the faces alone with `bench/crop_reference_to_head.py::HEAD_ROOM` and reads no head track: the face larger, the crown possibly cut. Larger is not enough on its own: on 278-293 the drawn iris stays under one Qwen3-VL 16 px patch tall on any of the three boxes (the cranium crop's iris is larger and still under one patch; [`docs/wiki/decisions.md`](../wiki/decisions.md), 2026-10-22). `cranium` (the Gemini reviewer's design, 2026-10-22) frames the body model's own bald grey head, since that is all `<Video 1>` holds, not the real hair the head track holds: each frame's face with the grey joined to it above mid-face (`bench/crop_reference_to_head.py::grey_head`; above the chin the neck and shoulders join raised arms and risen shoulders, seen on lead 1156 and second 0278), `CRANIUM_ROOM` above and beside, `CRANIUM_NECK` below the chin, one box over the load in the canvas's shape, enlarged at most `CRANIUM_MOST`; the record's `crop.cranium` says on which frames the grey head reaches the box's edge and what the subject's travel costs the enlargement, as figures. The record's `crop` says the form, the why and the enlargement; the text says "a closer view of the head and shoulders" in every form (`bench/load_text.py::subject_2`). Every subject of fun takes `cranium` (the plan; the owner, 2026-10-22). A frame whose face has no grey head joined to it does not size the cranium box; a span with no grey head at all keeps those frames (`crop.cranium.those_frames_size_the_box`). A frame with a wrongly fitted grey body still sizes it.
- **Single unified `<Subject 2>` prompt sentence (`bench/load_text.py::subject_2`)**: under the closed-lip line every face-drawn load's `<Subject 2>` sentence ends "with its face drawn as thin white outlines: the eyes, each with a dark iris, open, close and look where the person's eyes do" (`bench/load_text.py::FACE_EYES`; the owner's approval of 2026-10-22; until then "with facial features drawn on its face as thin white outlines, where the eyes open, close and look where the person's eyes do", which a load that draws no eye still says, `FACE_NO_EYES`). It names the mouth only for the outlined form (`bench/load_text.py::MOUTH_CLAUSE`) where the drawing opens a mouth on some frame of the load (`bench/load_text.py::mouth_drawn_open`: the closed-lip line's `open_frames`, the owner's go of 2026-10-22): "; where the mouth is open, the lips are outlined and it is dark inside, and the mouth opens and closes where the person's mouth does", tied to the open mouth because a closed frame draws nothing. Until 2026-10-22 no load's sentence named the mouth.
- **The rest of the text is the owner's form of 2026-10-22** (second_hfp_0415l, written by hand and kept): `<Subject 1>`'s definition names the hair `<Picture 1>` shows (the plan's `text.picture_hair`) and `[Shot 1]` the hair the shot ends up with (`text.hair_words`), never eye, eyebrow or skin words (the plan's `looks` is read by no line); a face-only subject's retention line is "the girl's face from <Picture 1> is retained."; a face-and-hair subject reads "the face and the <picture_hair> of <Picture 1> are retained.", its summary "with the face and the <picture_hair> of <Picture 1>" and `[Shot 1]` "has the face and the hair of <Picture 1>" (the plan's `text.picture_hair`; the owner's word of 2026-10-23, which also retired the plan's `text.hair_change`, refused if a plan still carries it); where a face is drawn, `<Subject 2>`'s retention line transfers "the movements of the head, the facial expressions, where the eyes look and how far the mouth opens" and `[Shot 1]` ends "..., her eyes looking where its eyes look and her mouth opening and closing where its mouth does." (`bench/load_text.py::TRANSFERRED_DRAWN`, `FOLLOWS_DRAWN`, the owner's approval of 2026-10-22), and where the drawing draws no mouth on any frame (`bench/load_text.py::mouth_drawn`: none opened, `mouth_drawn_open`, and none drawn closed under the plan's `closed_mouth`, `mouth_drawn_closed`, 2026-10-23) they name no mouth: "the movements of the head, the facial expressions and where the eyes look" and "..., her eyes looking where its eyes look." (`TRANSFERRED_DRAWN_NO_MOUTH`, `FOLLOWS_DRAWN_NO_MOUTH`, the owner's go of 2026-10-22 after second_hfp_0278l, whose drawing opened no mouth while its text said her mouth opens and closes, and she spoke; the singing line is the voice's and stays; "and eye movements" on a plain cut, the head's movements alone on a hair load); the music is `N/A`. The one home of the form is [`bench/load_text.py`](../../bench/load_text.py) (its docstring), held byte for byte to the owner's text by `bench/check_load_text.py`.
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
  drawing already in its folder without asking what code drew it (`bench/shot_to_load.py::step_reference`). Likewise, a
  capture's `.npz` and `.json` records are the chain's cache, and the node's own inputs and writes (`<load>_mask_00001.mp4`,
  `<load>_00001_zoomed.mp4`, `<load>.render.queued.json` `said`) are the artifact
  ([`docs/masked_v2v/verifying_raw_frames_and_keeps.md`](verifying_raw_frames_and_keeps.md)). The
  session that makes a remake holds the rule.

**What only a person writes.** These are the chain's inputs, made by no stage. A remake carries the first four; on
pass 2, only the owner's own words among them carry (§1).

- `data/<video>/shots.json`: who is on each shot, a shot's `order`, the frames a subject was marked or boxed on, and a
  person's lines on which parts of a subject are seen there (`who.<subject>.seen_looks`, written by
  [`bench/shots_look.py`](../../bench/shots_look.py)).
- `data/<video>/cast/cast.jsonl`: the owner's boxes.
- `data/<video>/plan.json`: for each subject, the region, parts, keep-outs, restore, hair band, pieces, reference
  form, zoom, what the context frames show, a person's marks on the head and hair (`head_marks`, §10), clothing, voice and text.
- The span words given to `job` (`--no-face-on`, `--no-draw-face-on`, `--no-head-on` and `--no-head-form`, `--drop`, `--hold-part`, `--no-mouth-on`, `--mouth-on`,
  `--eyes-shut-on` and `--eyes-shut-form`, `--parts-on`, `--track-not`). A remake carries them with `--carry-words-from OLD_JOB`, or says why
  not with `--no-carry-words WHY`. It never carries a seed (`--seed N --seed-why WHY`, written as `defaults.seed` and `defaults.seed_by`;
  the remake takes `h3_config.SEED` unless it is given its own), a zoom asked past the default (`--zoom-up-to N
  --zoom-up-to-why WHY`) or a margin asked in place of the region's (`--margin N --margin-why WHY`, written as `defaults.margin`
  and `defaults.margin_by`; the remake takes `h3_config.MASKED_LOAD_MARGIN` by its region unless it is given its own; a
  face-only load keeps the default: on 2026-10-23 a larger one let H3 copy the drawing's white outlines onto the face,
  `docs/wiki/decisions.md`); each is per load. The plan's reference form is read from the plan again, never carried.
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
   - Reads: the source (`--source`, required); the replacement(s) it is made for (`--for`, required where `laid.jsonl` lays
     anything on the shot) and `laid.jsonl`, against which the source is held (`bench/shot_to_load.py::check_pass2_source`,
     §1); `shots.json` (who is on the shot, its `order`, and the frame an
     unseeded tracker picks on, `bench/shot_to_load.py::pick_of`); the cast file (`--cast`): every owner box inside the
     shot is a seed on its frame (`bench/shot_to_load.py::seed_of`, `bench/cast_file.py::seeds`; boxes drawn on a sheet
     enter the file by `bench/cast_file.py box`, which takes `--encloses head|body`: a box round only her arm or body,
     where her head is hidden, seeds her person track and never her head track; every drawn box is recorded
     `drawn: approximate`, the owner's word of 2026-10-23, and names roughly WHICH person, never an outline:
     `subject_track.py::name_drawn` takes the one person lying mostly inside it, refuses a box across two people or on
     another tracked subject, and only where nobody qualifies asks SAM 3's box prompt, unclipped). The other subjects'
     boxes on each boxed frame ride along as `other` lines, so even the first tracker of a job knows who is somebody
     else there; a box-prompt result is taken only as one person a detection found too, else refused as doubtful
     (`subject_track.py::prompt_seed`). A tile's box is named on its frame (`subject_track.py::name_seed`:
     the detection whose box agrees, else the one whose mask overlaps the box, else what SAM 3 segments inside it, never
     the rectangle), tracks from each out to its neighbours' frames and keeps, frame by frame, the track its own score
     holds best (`subject_track.py::merge_tracks`), switching only to a mask continuous with the last kept and leaving
     empty, as `seed_tracks_disagree_on`, frames where two boxes' tracks hold different people; the shot table's `seeds`
     says what each box took, how, the frames it was kept on and why one was refused. The head track is seeded and
     main's span head track by default, exactly as before (from the shot's first head box or its own seed, never a top
     third from a body box); per stretch only on `preview --head-per-stretch WHY` (the owner, 2026-10-23: an opt-in, so no
     saved or rendered shot can lose a head), which seeds and tracks the head inside each stretch the person track holds
     her and fills a stretch with none from main's span head; the shot table's `head_mode` says which ran
     (`subject_track.py::_track_head_span`, `_span_head`). A box prompt is
     refused as doubtful on a frame where the tracker knows none of the job's other subjects (`job: N other subjects`,
     written by `bench/shot_to_load.py preview`). With `--seed-cuts-of JOB.json=TABLE.json` (the shots table an
     earlier preview of the same frames saved) the note counts the boxes by the tracker's cuts
     (`bench/shot_to_load.py::cuts_of_table`). `--cast` defaults to `data/<video>/cast/cast.jsonl` on `preview` when omitted (`bench/shot_to_load.py::step_preview`); without a cast box for the subject, the tracker picks the largest person
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
   - **The preview's files are made once, for every load built on it.** The strangers file (with its people file
     beside it) is made only when it is missing (`bench/shot_to_load.py::run_strangers`), so a change to
     `bench/strangers.py` does not reach a `job --remake`: a file whose record names another form
     (`strangers.OWN_FORM`), grow, hold or subject share is refused (`strangers.held_to_its_record`), and of any other
     change nothing says so. A cut that `job` would make differently now is refused as "of another
     making", to be moved aside by a person. A change to the code that makes these files forces a fresh preview.
   - A change forces (the job step's code, a plan field `job` reads, a span word): a fresh preview by default (above),
     and at the least `job --remake`, then every stage below.
3. **Reference**: `bench/shot_to_load.py reference <job file>` (the command: [`PROTOCOL.md`](PROTOCOL.md) 2.3).
   - Reads: the source, with the preview's track as the box for the pose pass; the body-model video; the class map
     when the mouth is `auto`; the head track (not read where the plan's `reference.crop` is `face` or `cranium`); the job's reference form and span words
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
   - **A black reference stops the load at the gate** (2026-10-22). The record's `figure` lists the frames where the
     body-model fit found no body and `<Video 1>` is black; the gate is told the load's reference video (`--reference`,
     `bench/load_chain.py::told_to_the_gate`) and refuses, with no override, where the subject is tracked or carried on
     a run of `FIGURE_RUN_LEAST` such frames with her part at or over the capture's floor
     (`bench/capture_masked_run.py::flag_reference_figure`); a shorter run, and frames under the floor, are said as data
     with their px, never a stop. Its way out: a reference that finds her
     there, or the job leaving those frames empty on a person's word (`drop=` with `drop_by`), only where she is truly hidden.
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
     (`bench/capture_masked_run.py::grade_holds`). `--no-draw-face-on` (`bench/shot_to_load.py::drawn_form`, `bench/head_reference.py`) suppresses `<Video 1>`'s 2D face marks on those frames (leaving the 3D grey head in `<Video 1>`) while keeping the capture's face mask open. `--no-head-on` and `--no-head-form hold|blank` (`bench/shot_to_load.py::drawn_form`, `bench/head_reference.py`) replace a wrong 3D grey head on `<Video 1>` with the last good cut body-model frame before the span (`hold`, default) or black (`blank`), with no face lines drawn, while leaving the capture and regeneration mask untouched. `--drop` goes to the capture (`drop=`) and to the gate's
     `--named-empty`. `--hold-part` goes to the capture (`word=`). Written `--hold-part FIRST-LAST@FRAME=WHY`, it names the
     frame the hold comes from: the capture places that frame's graded part where it was, not moved, cut to the track, and refuses
     it where that frame has no graded part (`bench/capture_masked_run.py::HELD_IN_PLACE`; 2026-10-23, third_fp_5011d's 5039).
     Where that part lies off her track, nothing is placed and no moved hold, the floor's included, takes the frame: it stays as
     the grade left it, recorded `held in place from F: missed` (`HELD_IN_PLACE_MISSED`), and the gate's empty-part flags ask.
   - **The head tube knows what its subject replaces** (`bench/head_tube_parts.py::head_tube_for_capture`, from the
     classes the preview cut from the plan's parts). A head the head track lost is re-seeded from the person track only
     where her own face points show on her own replaced classes (`sam31_head_tube`, `regain_refused` in its
     provenance), and tube pixels become `Hair` only on a part made of `Hair` ([`PROTOCOL.md`](PROTOCOL.md) 2.1).
   - **Her hair in the hole** (§10) runs after every mask is read and writes the part the load carries again.
   - `plan` is a run on the server: it builds and queues the plan graph and writes `<load>.plan.json`, `job.plan.json`,
     `<load>.plan.queued.json`, the band file on a hair-band load, and the node's plan of the windows on the share.
     `on-plan` copies the capture to `<capture>_first`, builds it again in place on the node's plan, writes
     `<load>.plan.masks.json`, makes the band again and writes the frame table, which `table` can also make again.
   - **The zoom is the node's**: it is set from the region's boxes at plan and render (`video_mask.py::zoom_box`), and
     read over the window's context frames too: real ones where the shot goes on, and across a cut the held copies of
     the shot's edge frame, which add no other box (`video_mask.py::zoomed_source`, `video_mask.hold_context`). By default at most `video_mask.ZOOM_LARGEST` (`2.0`); since 2026-10-23 (`data/fun/AGENTS.md`, "The zoom, by subject"), `second` loads pass `bench/shot_to_load.py job --zoom-up-to 4 --zoom-up-to-why WHY` (`video_mask.ZOOM_MOST_ASKED = 4.0`), enlarging a region too small to be `video_mask.ZOOM_HEAD_TARGET` (`300 px`) tall at `2.0x` until it is, up to `4.0x`, where the union box across the window allows (`set_by` and `up_to` in the plan record say what set it). Because `video_mask.py::_tracked_boxes` (`sapiens2_parts.mask_boxes`) reads the raw bounding box across all white pixels of the grown `region` on each frame, disconnected stray mask pieces inflate `region_height` (`video_mask.py::region_height`; e.g., `5935c`'s `323 px` raw vs `241 px` largest component) and widen the union box across the window, which can cap the zoom below `2.0x`; additionally, `video_mask.py::zoomed_source` crops and upscales frames already downscaled to the `1024x768` canvas by the loader ([`docs/research/masking/2026-10-25_context_frames_and_identity_on_2050.md`](../research/masking/2026-10-25_context_frames_and_identity_on_2050.md)).
     The capture works it out again for its part floor (`bench/capture_masked_run.py::floor_zoom`), which scales
     `bench/capture_masked_run.py::PART_LEAST_PX`; it takes the node's default (`2.0`), so on a load that asks past `2` the floor
     is held to the zoom at `2` and on-plan names the frames the run's zoom would change (`floor_against_the_run`). A change to a mask moves the zoom, the floor, and which frames are
     emptied.
   - **An override is a parent too.** `plan` refuses a blocked gate. A person clears a flag by name
     (`bench/capture_masked_run.py outcome <capture> <flag> overridden --by WHO --note WHY`), which records it
     in `outcomes.json` and writes the gate's verdict in `flags.json` again; `plan` and `render` read that verdict. An
     override written before `plan` or `render` lets that step run on this load; one changed after the step ran is a new
     load (`job --remake`). An override is a person's act on the owner's word, signed by the lead, never a loop in a script
     (decisions.md, 2026-10-21; 2026-10-22).
   - A change forces (`bench/capture_masked_run.py`, [`bench/head_tube_parts.py`](../../bench/head_tube_parts.py),
     [`bench/frame_table.py`](../../bench/frame_table.py), a span word the capture reads): a fresh preview by default
     (above), and at the least `job --remake`, then every stage below. These run per load, so a new load gets a new
     capture.
6. **Look, flags read, render, land and page**: the look the tool requires, signed by the lead on the owner's word with "not
   inspected" in its note (`bench/capture_masked_run.py looked`), or the maker's `bench/load_chain.py no-look
   --not-first FILE` where no forcing rule stands, plus `bench/flags_before_render.py <capture> --initial WHO` (`bench/load_chain.py::the_flags_read`). No session looks at the render before the owner does (decisions.md,
   2026-10-22). Then `load_chain.py render`, `land` and `page` (the commands: [`PROTOCOL.md`](PROTOCOL.md) 3.3
   and 5.1).
   - `render` reads the look, the flags-read record (`<capture>/looks/<load>.flags_read.json`), the gate, the frame table, the job file as planned and every mask the plan read. It
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

### 10. Her hair in the hole: no original hair of hers left in the window (2026-10-22)

- **The fault.** Where a replaced subject's original hair shows unregenerated on frames of the window, H3 draws her new hair dark to
  match it (measured over 61 renders of the second subject; second_hfp_5768e against 5768f is the controlled pair: the same load with
  her hair kept in the hole came out blonde and was kept). On second_hfp_4912e her crown above a passing hat brim on 4944-4946 was in
  no mask at all ([`docs/wiki/decisions.md`](../wiki/decisions.md), 2026-10-21).
- **The pass** (`bench/capture_masked_run.py::her_hair_in_hole`, at the capture's `files` step, on every load, for a subject whose
  part is made of Hair; no one looks). SAM 3.1 is the source of truth for her and her head; Sapiens2 decides only what is hair.
  1. Her SAM head track outside the hole (the part grown by the load's margin at the zoom the node will run it at, from the load's
     `render_region` and `zoom_up_to`, `hole_zoom`; on the capture built on the node's plan, the plan's own margin and zoom,
     `hole_of_the_load`; refused in words where a load names her and its margin cannot be read) is added to the part the load carries,
     less another person's pixels and her face on a `--no-face-on` frame.
  2. A run of frames with no head where SAM's own score still holds her is a lost head. The capture runs no model and cannot seed
     SAM again, so it refuses (`her_head_lost_in_view`, at the top level of the gate) with the frames.
  3. Sapiens2's Hair in her room (round her head, or her nearest whole head) that no one's mask holds and the hole does not is
     added; a piece touching another person is refused (`unclaimed_hair_touches_another`).
  What it added and why, a frame at a time, and what it cannot see: `subjects/<s>/hair_in_hole__<by>.json` and a line in the manifest.
- **It does not reopen the head tube's re-seed gate** (`bench/head_tube_parts.py`, `regain_refused`): nothing is taken from the top
  of her person track, and every addition leaves out every other person's mask as its file holds it, before the capture's carve. The
  carve gives keep-out cells that are mostly hers back to her for the node; it does not make another person's pixels hers. On
  4912e's 4940-4941 the carve on the node's grid had taken the passing woman's hat and brow out of the capture's strangers mask
  (5,243 of her nearest whole head's 5,356 px on 4941 are in the strangers file and 4 in the capture's copy); read from the file,
  `hers` there holds 113 px, not 5,352. A lost head's frames are asked about, and filled only on a person's `hers`.
- **The owner answers on the page; the answer is plan data.** `subjects.<s>.head_marks`: `[{first, last, mark, by}]` with `mark`
  one of `hidden` (her head is not in view there: nothing asked), `hers` (the unclaimed hair is hers; on a lost head, her nearest
  whole head track, less every other person's mask, is held there) or `not hers` (no unclaimed hair added there). `job` cuts the
  spans to the load (`bench/shot_to_load.py::head_marks_of`) and the chain hands them to the capture. A lost head can also be
  boxed in the cast file for a fresh preview, which seeds SAM on that frame.
- **Her hair is read from** the whole-frame class map the strangers run now keeps beside its file (`whole_frame_classes` in the
  strangers record, `bench/strangers.py`), else her own class map as the preview saved it (her crop alone). Sapiens2 labels her
  crown on 4912e's 4944 and 4945 Background even on the whole frame (run on those two frames, 2026-10-22), so there only part 2
  catches her.
- `bench/capture_masked_run.py hair-in-hole CAPTURE` works the same out on a capture that exists, read-only, and writes it elsewhere.
