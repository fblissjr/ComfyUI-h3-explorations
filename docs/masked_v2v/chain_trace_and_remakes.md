# What a Change Forces (The Chain in Trace Order) & `her_hair_in_hole`

Extracted from [`docs/masked_v2v/AGENTS.md`](AGENTS.md) §8 and §10 so both the router (`AGENTS.md`) and this specification stay under `35 KB` and fit in a single `read_file` call. Step-by-step CLI commands: [`docs/masked_v2v/PROTOCOL.md`](PROTOCOL.md).

---

## 1. What a change forces: the chain in trace order (§8)

A load is made in the order below, and each stage reads what the stages above it wrote. A change to a stage's code or
inputs therefore forces that stage and every stage after it to be made again (root [`AGENTS.md`](../../AGENTS.md),
"Nothing an earlier run made is reused by a key"; [`docs/wiki/decisions.md`](../wiki/decisions.md), 2026-10-18,
"Derived data is made again whenever the code that makes it changes"). Trace a fault the same way: from the stage that
shows it, up through what that stage read.

- **The default remake is a fresh preview and every stage after it** (the owner, 2026-10-21: when unsure, make every
  input again; `decisions.md`). The exception is a test of one change, which may reuse a preview when `git log` from the
  preview's commit to `HEAD` shows no change to the code stage 1 reads (`decisions.md`, 2026-10-21). The preview's commit
  is `commit` in its job folder's `preview.queued.json`, which [`bench/masked_job.py`](../../bench/masked_job.py) writes
  when it queues the preview.
- **No remake starts later than `job --remake`, and none starts at `first`.** `first`, `plan` and `render` each run once
  under a load's name ([`bench/load_chain.py`](../../bench/load_chain.py), "The steps, and what each refuses"), so a
  remake is a new load with a new name, its own reference step and its own text.
- **Nothing in `load_chain.py` enforces this on its own; `inspect_mask_steps.py` audits it explicitly.** No step of `load_chain.py` compares the code that made an input with the code now: `bench/capture_masked_run.py::verify_checks` (`inputs_written_together`) only checks that sibling files inside a preview share internal timestamps, so passing an older `--preview data/<old_preview_job>` to `shot_to_load.py job` passes `inputs_written_together` even when producer code in the dependency chain (`subject_track.py`, `sam31_corrections.py`, `sapiens2_parts.py`, `strangers.py`, `head_tube_parts.py`, `pose_along_shot.py`, `head_reference.py`, `draw_face_points.py`, `video_mask.py`) changed afterward (seen on `2026-10-25`: `lead_fp_2623b` reused `shot_c_2623d` from `2026-10-21T15:03`; `second_hfp_1039h` reused `shot_c_1039d` from `2026-10-22T12:12` with no `strangers`; and `second_hfp_5114k` reused `shot_c_5114f` from `2026-10-22T16:50` before `513783e6`, `14838f44`, and `79b7b6da`). `bench/_lib/code_stamp.py::check_upstream_mtime` compares file times and only warns; the strangers record carries no writer stamp ([`bench/strangers.py`](../../bench/strangers.py)); and `shot_to_load.py reference --existing` takes a drawing already in its folder without asking what code drew it (`bench/shot_to_load.py::step_reference`). Likewise, a capture's `.npz` and `.json` records are the chain's cache, and the node's own inputs and writes (`<load>_mask_00001.mp4`, `<load>_00001_zoomed.mp4`, `<load>.render.queued.json` `said`) are the artifact ([`docs/masked_v2v/verifying_raw_frames_and_keeps.md`](verifying_raw_frames_and_keeps.md)). Run [`bench/inspect_mask_steps.py`](../../bench/inspect_mask_steps.py) `--capture data/<day>_<load>` ([`AGENTS.md`](AGENTS.md) §11) to check every input file's mtime against both the capture's build time and its producer code's latest git commit. The session that makes a remake holds the rule.

**What only a person writes.** These are the chain's inputs, made by no stage. A remake carries the first four; on
pass 2, only the owner's own words among them carry ([`AGENTS.md`](AGENTS.md) §1).

- `data/<video>/shots.json`: who is on each shot, a shot's `order`, the frames a subject was marked or boxed on, and a
  person's lines on which parts of a subject are seen there (`who.<subject>.seen_looks`, written by
  [`bench/shots_look.py`](../../bench/shots_look.py)).
- `data/<video>/cast/cast.jsonl`: the owner's boxes.
- `data/<video>/plan.json`: for each subject, the region, parts, keep-outs, restore, hair band, pieces, reference
  form, zoom, what the context frames show, a person's marks on the head and hair (`head_marks`, §2 below), clothing, voice and text.
- The span words given to `job` (`--no-face-on`, `--no-draw-face-on`, `--no-head-on` and `--no-head-form`, `--drop`, `--hold-part`, `--no-mouth-on`, `--mouth-on`,
  `--eyes-shut-on` and `--eyes-shut-form`, `--parts-on`, `--track-not`). A remake carries them with `--carry-words-from OLD_JOB`, or says why
  not with `--no-carry-words WHY`. It never carries a seed (`--seed N --seed-why WHY`, written as `defaults.seed` and `defaults.seed_by`;
  the remake takes `h3_config.SEED` unless it is given its own). Settled `margin` and `zoom_up_to` live in `data/<video>/plan.json`
  (`subjects.<s>.recipe` and `shot_overrides`, enforced in code by `cbd39fbe` at `bench/shot_to_load.py` and `bench/flags_before_render.py::recipe_check_of`,
  which hard-refuse any differing CLI `--margin` or `--zoom-up-to` value with no `--*-why` bypass; `docs/wiki/decisions.md` 2026-10-26). The plan's reference form is read from the plan again, never carried.
- A look (stage 6), and an override of a gate flag (stage 5). Each belongs to one capture. A new capture takes its own
  look, or a person rests one on an earlier capture's with [`bench/look_carry.py`](../../bench/look_carry.py), which
  compares the two captures first.

### The stages

0. **The source, and the code the server runs.**
   - The source is the working copy, or on pass 2 the newest cut that holds the kept subject ([`AGENTS.md`](AGENTS.md) §1).
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
     [`AGENTS.md`](AGENTS.md) §1); `shots.json` (who is on the shot, its `order`, and the frame an
     unseeded tracker picks on, `bench/shot_to_load.py::pick_of`); the cast file (`--cast`): every owner box inside the
     shot is a seed on its frame (`bench/shot_to_load.py::seed_of`, `bench/cast_file.py::seeds`; boxes drawn on a sheet
     enter the file by `bench/cast_file.py box`, which takes `--encloses head|body`: a box round only her arm or body,
     where her head is hidden, seeds her person track and never her head track; every drawn box is recorded
     `drawn: approximate`, the owner's word of 2026-10-23, and names roughly WHICH person, never an outline
     [2026-10-26, the owner: Group 1 `/who/` page clicks (`lines 1-174`) and Group 3 session-verified exact SAM detection
     boxes (`lines 197-209`, guided by the owner's Group 2 visual cues) stand as tracker seeds; Group 2 rough hand-drawn
     rectangles from the 2026-10-23 pasted screenshots (`lines 175-192, 194-196`) are never used directly as raw box seeds
     and withdraw with the tracker build in the handover's order (`decisions.md`, 2026-10-26)]:
     `subject_track.py::name_drawn` takes the one person lying mostly inside it, refuses a box across two people or on
     another tracked subject, and only where nobody qualifies asks SAM 3's box prompt, unclipped). The other subjects'
     boxes on each boxed frame ride along as `other` lines, so even the first tracker of a job knows who is somebody
     else there; a box-prompt result is taken only as one person a detection found too, else refused as doubtful
     (`subject_track.py::prompt_seed`). A tile's box is named on its frame (`subject_track.py::name_seed`:
     the detection whose box agrees, else the one whose mask overlaps the box, else what SAM 3 segments inside it, never
     the rectangle), tracks from each out to its neighbours' frames and keeps, frame by frame, the track its own score
     holds best (`subject_track.py::merge_tracks`), switching only to a mask continuous with the last kept and leaving
     empty, as `seed_tracks_disagree_on`, frames where two boxes' tracks hold different people; the shot table's `seeds`
     says what each box took, how, the frames it was kept on and why one was refused. Since `6b0ea9a6` (2026-10-27, after `second_hfp_4647h` lost her head track on `4682-4695` under whole-span mode), the head track defaults to `"per stretch"` (`HEAD_PER_STRETCH` in `subject_track.py::parse_head_mode` and `bench/shot_to_load.py`), which seeds and tracks the head inside each stretch the person track holds
     her and fills a stretch with none from main's span head (`"main"` whole-span mode was the default until `6b0ea9a6`); the shot table's `head_mode` says which ran
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
3. **Reference**: `bench/shot_to_load.py reference <job file>` (the command: [`PROTOCOL.md`](PROTOCOL.md) 2.3; drawing rules: [`reference_and_prompt_rules.md`](reference_and_prompt_rules.md)).
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
   [`PROTOCOL.md`](PROTOCOL.md) 2.3; text rules: [`reference_and_prompt_rules.md`](reference_and_prompt_rules.md)).
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
     it where that frame has no graded part (`bench/capture_masked_run.py::HELD_IN_PLACE`; 2026-10-23, `third_fp_5011d`'s 5039).
     Where that part lies off her track, nothing is placed and no moved hold, the floor's included, takes the frame: it stays as
     the grade left it, recorded `held in place from F: missed` (`HELD_IN_PLACE_MISSED`), and the gate's empty-part flags ask.
   - **The head tube knows what its subject replaces** (`bench/head_tube_parts.py::head_tube_for_capture`, from the
     classes the preview cut from the plan's parts). A head the head track lost is re-seeded from the person track only
     where her own face points show on her own replaced classes (`sam31_head_tube`, `regain_refused` in its
     provenance), and tube pixels become `Hair` only on a part made of `Hair` ([`PROTOCOL.md`](PROTOCOL.md) 2.1).
   - **Her hair in the hole** (§2 below) runs after every mask is read and writes the part the load carries again.
   - `plan` is a run on the server: it builds and queues the plan graph and writes `<load>.plan.json`, `job.plan.json`,
     `<load>.plan.queued.json`, the band file on a hair-band load, and the node's plan of the windows on the share.
     `on-plan` copies the capture to `<capture>_first`, builds it again in place on the node's plan, writes
     `<load>.plan.masks.json`, makes the band again and writes the frame table, which `table` can also make again.
   - **The zoom is the node's**: it is set from the region's boxes at plan and render (`video_mask.py::zoom_box`), and
     read over the window's context frames too: real ones where the shot goes on, and across a cut the held copies of
     the shot's edge frame, which add no other box (`video_mask.py::zoomed_source`, `video_mask.hold_context`). Settled `zoom_up_to` and `margin` live in `data/<video>/plan.json` (`subjects.<s>.recipe` and `shot_overrides`, enforced in code by `cbd39fbe` at `bench/shot_to_load.py` and `bench/flags_before_render.py::recipe_check_of` with no `--*-why` CLI bypass; for `second`, `zoom_up_to = 4.0`, enlarging a region too small to be `video_mask.ZOOM_HEAD_TARGET` (`300 px`) tall at `2.0x` until it is, up to `4.0x`, where the union box across the window allows; `set_by` and `up_to` in the plan record say what set it). Because `video_mask.py::_tracked_boxes` (`sapiens2_parts.mask_boxes`) reads the raw bounding box across all white pixels of the grown `region` on each frame, disconnected stray mask pieces inflate `region_height` (`video_mask.py::region_height`; e.g., `5935c`'s `323 px` raw vs `241 px` largest component) and widen the union box across the window, which can cap the zoom below `2.0x`; additionally, `video_mask.py::zoomed_source` crops and upscales frames already downscaled to the `1024x768` canvas by the loader ([`docs/research/masking/2026-10-25_context_frames_and_identity_on_2050.md`](../research/masking/2026-10-25_context_frames_and_identity_on_2050.md)).
     The capture works it out again for its part floor (`bench/capture_masked_run.py::floor_zoom`), which scales
     `bench/capture_masked_run.py::PART_LEAST_PX`; it takes the node's default (`2.0`), so on a load that asks past `2` the floor
     is held to the zoom at `2` and on-plan names the frames the run's zoom would change (`floor_against_the_run`). A change to a mask moves the zoom, the floor, and which frames are
     emptied.
   - **An override is a parent too.** `plan` refuses a blocked gate. A person clears a flag by name
     (`bench/capture_masked_run.py outcome <capture> <flag> overridden --by WHO --note WHY`), which records it
     in `outcomes.json` and writes the gate's verdict in `flags.json` again; `plan` and `render` read that verdict. An
     override written before `plan` or `render` lets that step run on this load; one changed after the step ran is a new
     load (`job --remake`). An override is a person's act on the owner's word, signed by the lead, never a loop in a script
     (`decisions.md`, 2026-10-21; 2026-10-22).
   - A change forces (`bench/capture_masked_run.py`, [`bench/head_tube_parts.py`](../../bench/head_tube_parts.py),
     [`bench/frame_table.py`](../../bench/frame_table.py), a span word the capture reads): a fresh preview by default
     (above), and at the least `job --remake`, then every stage below. These run per load, so a new load gets a new
     capture.
6. **Look, flags read, render, land and page**: the look the tool requires, signed by the lead on the owner's word with "not
   inspected" in its note (`bench/capture_masked_run.py looked`), or the maker's `bench/load_chain.py no-look
   --not-first FILE` where no forcing rule stands, plus `bench/flags_before_render.py <capture> --initial WHO` (`bench/load_chain.py::the_flags_read`). No session looks at the render before the owner does (`decisions.md`,
   2026-10-22). Then `load_chain.py render`, `land` and `page` (the commands: [`PROTOCOL.md`](PROTOCOL.md) 3.3
   and 5.1).
   - `render` reads the look, the flags-read record (`<capture>/looks/<load>.flags_read.json`), the gate, the frame table, the job file as planned and every mask the plan read. It
     writes the go beside the job file and the render under the job's output folder on the share.
   - `land` writes the render's points beside the capture, the stacked clip under `Video/<video>/review_clips/`, and
     the clip's record under `<capture>/strips/`. `page` adds the clip to `data/<video>/review/items.json`.
7. **Laid, and the cut**: `bench/laid.py`, then one `bench/rebuild_delivery.py --video <video> --clip` for a batch
   ([`PROTOCOL.md`](PROTOCOL.md) 5.3). Laying a piece is a person's act. The cut is pass 2's source (stage 0), so a
   change to a kept piece of pass 1 forces a new cut, and pass 2 made again on it from a fresh preview.

---

## 2. Her hair in the hole: no original hair of hers left in the window (§10, 2026-10-22)

- **The fault.** Where a replaced subject's original hair shows unregenerated on frames of the window, H3 draws her new hair dark to
  match it (measured over 61 renders of the second subject; `second_hfp_5768e` against `5768f` is the controlled pair: the same load with
  her hair kept in the hole came out blonde and was kept). On `second_hfp_4912e` her crown above a passing hat brim on 4944-4946 was in
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
  identified on the who-is-who page in the cast file (`Group 1`) or via a session-verified exact SAM detection box guided by the owner's visual cues (`Group 3`) for a fresh preview, which seeds SAM on that detection (rough hand-drawn rectangles from pasted screenshots, `Group 2`, are never copied directly into `cast.jsonl` as raw seeds; [`docs/wiki/decisions.md`](../wiki/decisions.md), 2026-10-26).
- **Her hair is read from** the whole-frame class map the strangers run now keeps beside its file (`whole_frame_classes` in the
  strangers record, `bench/strangers.py`), else her own class map as the preview saved it (her crop alone). Sapiens2 labels her
  crown on 4912e's 4944 and 4945 Background even on the whole frame (run on those two frames, 2026-10-22), so there only part 2
  catches her.
- `bench/capture_masked_run.py hair-in-hole CAPTURE` works the same out on a capture that exists, read-only, and writes it elsewhere.
- **2026-10-25 / 2026-10-26 occlusion & `head_tube` handoff (`449b17f1` on 2026-10-25; items 1 and 4 landed on 2026-10-26 in `b461584e`, which also skips `hidden` frames in the fill and counts only asked frames at the gate; [`docs/masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](2026-10-25_pass_contamination_and_mask_step_forensics.md) §0.7)**:
  1. `head_tube` runs by default in `bench/capture_masked_run.py:2558` (`not args.no_head_tube`, `head_tube: true` in `manifest.json`) whenever a `head_track` exists, and its bridged output (`head_tube__<by>.npz["tube"]`) is handed into `her_hair_in_hole` and `inspect_mask_steps.py` when present so short head-track gaps already bridged by `head_tube` (`4686-4695` on `4647g`) are not seen as `0 px` by `her_hair_in_hole`.
  2. `clean_whole`: excludes edge-shrunk frames (`< 0.75 * neighbor`) from serving as `whole` gap-fill references (`4694` on `4647g`, `4920` and `4939` on `4912g`).
  3. `near_h`: widens the `saved_part` bridge reach to `abs(k - src) <= 2`.
  4. Lost-head fill subtracts every other person's mask (`~taken[k]`): commit `449b17f1` tried class-filtering `taken` down to `Apparel` + non-subject `face_ids` (`taken_for_hers`), which failed on `4912h` (`4943-4946`) because same-colour bystander hair has class `Hair` and was claimed as the subject's hair. Every other person's mask (`~taken[k]`) is subtracted in full, and occlusion gaps where `head_tube` is empty are settled by owner `head_marks` (`hers` / `hidden` / `not hers`) in `marks.json`.
  5. **What `second_hfp_4647h` and `HAIR_IN_HOLE_CANNOT_SEE` (`bench/capture_masked_run.py:5132–5407`) exposed at close of 2026-10-26 ([`docs/wiki/decisions.md`](../wiki/decisions.md); [`2026-10-25_pass_contamination_and_mask_step_forensics.md`](2026-10-25_pass_contamination_and_mask_step_forensics.md) §0.9)**: Stage 3 (`line 5384`, `cand = hair[f] & room & ~part[f] & ~hole[f] & ~track[f]`) subtracts the subject's own `person_track` (`~track[f]`), which forbade `her_hair_in_hole` from recovering the `945–1,354 px` of hair that `head_tube` had cut inside her track on `4686` and `4693–4695` when `--head-per-stretch` was off (`0` hair pixels added across all `79` frames of `4647h`), and reads whole-frame Sapiens (`strangers_classes.npz`, item 2) rather than her high-res crop class map (`classes__<preview>.npz`); meanwhile Stage 2 (`item 4`) triggers on arm-only `person_track` frames (`4682–4685`). On 2026-10-27 (`426638a6`, `1e91de26`, `6b0ea9a6`), `strangers.py` includes unowned SAM detections where whole-frame Sapiens finds nobody (`426638a6`), `grade_holds` takes `head_tube`'s tube when `head_tube` ran and `head_tube_parts.py` anchors to the head (`1e91de26`), and `head_mode` defaults to `"per stretch"` (`HEAD_PER_STRETCH`, `6b0ea9a6`).
  6. **Proven recipe that solved `second_hfp_4647i` (`4647–4725`, kept by the owner on 2026-10-27, `data/fun/owner_answers.jsonl` lines 1653–1654: *"Looks good! blonde hair was kept and face was kept all the way through"*)**:
     - **Why `4647h` had reverted to dark hair from `4685` onward**: In `bench/capture_masked_run.py::grade_holds`, `raw_px` is compared against `floor_px = round(PART_LEAST_PX / (floor_zoom ** 2)) = 1,024 px` (`PART_LEAST_PX = 4096`, `floor_zoom = 2.0`). In `4647h` (before `6b0ea9a6` and `1e91de26`), `head_track` was `0 px` on `4682–4695` and emerging masks under `1,024 px` were emptied (`"under the floor"`), leaving original dark hair unmasked in the window.
     - **What actually kept `second_hfp_4647i`'s emerging frames (`426638a6..6b0ea9a6`)**:
       1. **`1e91de26` taking `head_tube`'s tube in `grade_holds` at the default `1,024 px` floor (`data/2026-10-27_second_hfp_4647i/subjects/second/part_grades.json:1-15`)**: `second_hfp_4647i` ran with `least_px: 1024` (`least_px_set_by: "the starting value (PART_LEAST_PX)..."`, `head_tube: true`, not a lowered CLI floor). Because `6b0ea9a6` re-found `head_track` per stretch and `1e91de26` (`bench/capture_masked_run.py:4678-4695`) takes `head_tube`'s tube directly when `head_tube` ran, the 7 sub-`1,024 px` emerging frames on `4686–4702` (`4686, 4687, 4691, 4692, 4700, 4701, 4702` at `609–951 px`, plus `4673` at `622 px` and `4703–4715` at `332–694 px`) were retained (`did: "saved part kept"` or `"saved part kept, fill refused"`). Where a shot requires a lowered floor, the actual CLI flag in `bench/shot_to_load.py job` is `--least PX --least-why WHY` (not `--part-least-px`).
       2. **Suppress 2D face lines on hidden/turned-away frames (`--no-draw-face-on`) while keeping the open-mouth reference frame**: Pass `--no-draw-face-on` on the spans where her face is hidden or turned away (`4673–4687` and `4697–4725` on `4647i`) so `<Video 1>` has zero `MARKS AGAINST THE BODY MODEL` flags, while keeping 2D face/mouth lines active on the visible singing/speaking frames (`4647–4672`, preserving `open_frames` and the mouth prompt clause in `<Subject 2>`).
