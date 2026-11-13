# Masked video to video: router index & lane overview

last updated: 2026-11-10 (added "The pipeline, end to end", read off the code, with every step linked to its function and line; before that, 2026-10-27: converted to a router index pointing to `docs/wiki/masked_v2v/` for current operative rules, execution protocol, and code map; the full `2026-10-04..2026-10-15` design chronicle is preserved verbatim in [`archive/2026-10-15_masked_v2v_chronicle.md`](archive/2026-10-15_masked_v2v_chronicle.md)).

## Current Operative Rules, Protocol & Code Map (`docs/wiki/masked_v2v/`)

| Topic | Canonical Document |
|---|---|
| **Operative lane rules & index (`§1`–`§11`)** | [`../masked_v2v/AGENTS.md`](masked_v2v/AGENTS.md) |
| **Execution & debugging CLI commands (Stages 0–5)** | [`../masked_v2v/PROTOCOL.md`](masked_v2v/PROTOCOL.md) |
| **What a change forces (Stages 0–7 in trace order) & `her_hair_in_hole`** | [`../masked_v2v/chain_trace_and_remakes.md`](masked_v2v/chain_trace_and_remakes.md) |
| **Head reference (`<Video 1>`) mouth/eye drawing, crops (`cranium`), & `<Subject 2>` prompts** | [`../masked_v2v/reference_and_prompt_rules.md`](masked_v2v/reference_and_prompt_rules.md) |
| **AST jump index, 7-stage contracts, & simplification plan (`2026-10-26`)** | [`../masked_v2v/CODE_MAP_INDEX.md`](masked_v2v/CODE_MAP_INDEX.md), [`../masked_v2v/code_map_stages.md`](masked_v2v/code_map_stages.md), [`../masked_v2v/code_map_contracts.md`](masked_v2v/code_map_contracts.md), [`../research/masking/2026-10-26_code_map_and_simplification_plan.md`](research/masking/2026-10-26_code_map_and_simplification_plan.md) |
| **Simplification plan after `fun` (`2026-11-10`, a proposal): SQLite records, opt-ins as defaults, the branch test, one temporal stage** | [`../masked_v2v/2026-11-10_simplification_plan.md`](masked_v2v/2026-11-10_simplification_plan.md) |
| **Perception, stage-split & modularization branch (`gem/perception-lhf`, `2026-10-27`)** | [`../masked_v2v/2026-10-27_perception_lhf_branch.md`](masked_v2v/2026-10-27_perception_lhf_branch.md) |
| **Verifying raw frames vs. `.npz` / JSON records (`laid.py keeps`, zoom boxes)** | [`../masked_v2v/verifying_raw_frames_and_keeps.md`](masked_v2v/verifying_raw_frames_and_keeps.md) |
| **Pass contamination, perception provenance, & `inspect_mask_steps.py` (`2026-10-25`)** | [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md) |
| **Window lengths (`h3_rules.on_both_clocks`), 17-frame VAE clips, & padding** | [`../masked_v2v/frames_rules.md`](masked_v2v/frames_rules.md) |
| **Where per-video state lives (`data/<video>/`)** | [`../../data/AGENTS.md`](../../data/AGENTS.md) |
| **Historical chronicle (`2026-10-04` through `2026-10-15`)** | [`archive/2026-10-15_masked_v2v_chronicle.md`](archive/2026-10-15_masked_v2v_chronicle.md) |

## The pipeline, end to end

Read off the code on 2026-11-10. Each step links to the function that does it, at the line it starts on.
`bench/masked_code_map.py --check` holds every such link to the live code, and `--fix-links` moves an anchor
that has drifted (`bench/check_masked_code_map.py`, case `line_anchored_links`). Where this section and the code
disagree, the code is right. Which rules bind each stage and what a change forces are in
[`../masked_v2v/chain_trace_and_remakes.md`](masked_v2v/chain_trace_and_remakes.md); the commands are in
[`../masked_v2v/PROTOCOL.md`](masked_v2v/PROTOCOL.md).

### The perception models, and where each runs

| model | config constant | what it gives the lane | where it runs |
|---|---|---|---|
| SAM 3.1 (multiplex checkpoint) | `h3_config.SEGMENTER` | people found by a phrase, one person followed through a shot from a mask, a separate head mask | the preview graph's Subject Track node; the strangers pass |
| SAM 3.1 corrections | `h3_config.MASKED_JOB_SAM_CORRECTIONS` | three places ComfyUI's port departs from Meta's code, put right at runtime: the image range, the text encoder's GELU, and the tracker's memory (only frames where the person was seen) | [`sam31_corrections.py::MiniMaxH3SAM31Corrections`](../../sam31_corrections.py#L832), wired before every tracker |
| Sapiens2 segmentation 1B | `h3_config.SAPIENS2` | a class for every pixel of one person (`vendor/meta_sapiens2/seg_classes29.json`: face and neck, hair, lips, teeth, clothing, hands...) | the preview's Subject Parts node; the strangers pass on the whole frame; the head tube |
| Sapiens2 pose 1B | `h3_config.POSE_MODEL` | named body, hand and face points, the lids and lips among them | the reference step's pose pass, and again on the render at land |
| SAM 3D Body (DINOv3) | `h3_config.MASKED_JOB_BODY_MODEL` | a posed body mesh a frame, and which way body and head face | the preview graph's Body Pose node |
| HyDen MoGe-2 metric point | `vendor/meta_moge2/hyden_moge2.py::DEFAULT_CHECKPOINT` | metric depth a pixel, to say who is in front where two people overlap | the job step, on overlap frames only |

Two models on disk are off: Sapiens2 matting (`h3_config.SAPIENS2["matting"]` is `"none"`) and the ViT-H body model.
What each model can and cannot tell the lane is on [`meta_perception_models.md`](masked_v2v/meta_perception_models.md).

### 0. What a person writes, before any tool runs

- **The working copy**: every frame of the source at exactly 24 fps, registered in `data/<video>/manifest.json`. A
  job always renders from the present copy, whatever copy a record was made on
  ([`bench/shot_to_load.py::working_copy`](../../bench/shot_to_load.py#L294)), and a source at any other rate is
  refused ([`bench/masked_job.py::analysed`](../../bench/masked_job.py#L196)).
- **`data/<video>/shots.json`**: where each shot starts and ends and who is on it. Only a person writes it. The
  tracker's own cut finder ([`subject_track.py::cut_scores`](../../subject_track.py#L488), edge maps of neighbouring
  frames compared, so a lighting change is not a cut) proposes cuts inside a load; it does not write this file.
- **`data/<video>/cast/cast.jsonl`**: who is who, as a person answered it on the who page
  ([`bench/page_answers_to_cast.py::main`](../../bench/page_answers_to_cast.py#L449)) or as a box checked against
  SAM's detections on that frame ([`bench/cast_file.py::drawn`](../../bench/cast_file.py#L276)). Each answer becomes
  a seed for the tracker ([`bench/cast_file.py::seeds`](../../bench/cast_file.py#L383)).
- **`data/<video>/plan.json`**: per subject, the parts replaced, the reference still, the margin and zoom, the
  owner's head marks, and a person's word on depth. Depth is the chain's default (the owner, 2026-10-28; the code's
  own since 2026-11-10, [`bench/shot_to_load.py::make_depth`](../../bench/shot_to_load.py#L4435)); a plan turns it off only with a `why`.

### 1. Preview: who is where, on every frame of the shot

[`bench/shot_to_load.py::step_preview`](../../bench/shot_to_load.py#L4227) holds the source to what is already laid
on the shot ([`bench/shot_to_load.py::check_pass2_source`](../../bench/shot_to_load.py#L4080): the raw source is
refused where another subject's piece is laid, and so are a cut older than that piece and a cut that already holds
this subject's own replacement), writes the
preview job ([`bench/shot_to_load.py::preview_job`](../../bench/shot_to_load.py#L464)), and queues the graph
[`workflows/masked_job_graph.py::preview`](../../workflows/masked_job_graph.py#L306) on ComfyUI. For each subject,
in the job's order, with every earlier subject's mask passed in as someone else:

1. **Track** ([`subject_track.py::follow`](../../subject_track.py#L1400)): cut the load into shots; pick the subject
   from the cast seeds ([`subject_track.py::name_seed`](../../subject_track.py#L1832)) or, with none, by the pick rule
   ([`subject_track.py::main_subject`](../../subject_track.py#L995)); track each shot from that person's mask, never a
   text prompt; join several seeds' tracks ([`subject_track.py::merge_tracks`](../../subject_track.py#L1153)); look
   again for a person the tracker let go, against a gallery of the track's own frames
   ([`subject_track.py::gallery_scores`](../../subject_track.py#L1069)); and follow the head separately, per stretch
   the person is seen ([`subject_track.py::_track_head_span`](../../subject_track.py#L1252)).
2. **Parts** ([`sapiens2_parts.py::subject_parts`](../../sapiens2_parts.py#L373)): the subject alone (everything
   further than a margin from the track blanked), cropped to the model's portrait shape, labelled by Sapiens2,
   pasted back, cut to the subject's mask, and a missed or doubtful frame held from its nearest good one.
3. **Body** ([`body_pose.py::MiniMaxH3BodyPose`](../../body_pose.py#L869)): SAM 3D Body on Meta's own crop, given
   the subject's mask so it poses this person and not one behind; drawn as the grey body-model video
   ([`body_pose.py::MiniMaxH3BodyMeshVideo`](../../body_pose.py#L943)).

It writes each subject's track, head track, parts, class map, shot table, pose table and body-model video to the
share. A person looks at the tracker's tiles before anything is built on it.

### 2. Job: the load's own keep-outs and depth

[`bench/shot_to_load.py::step_job`](../../bench/shot_to_load.py#L4490) writes the load's job file and cuts its masks:

- **Strangers** ([`bench/shot_to_load.py::run_strangers`](../../bench/shot_to_load.py#L4352) runs
  [`bench/strangers.py::run`](../../bench/strangers.py#L525)): SAM 3.1 people on every frame
  ([`bench/strangers.py::people_pass`](../../bench/strangers.py#L362)) and Sapiens2 on the whole frame; every person
  who is not a subject becomes a keep-out, less what is the subject's own (a hat, a hand, flying hair:
  [`bench/strangers.py::handed_back`](../../bench/strangers.py#L174)).
- **The part held to the track** ([`bench/shot_to_load.py::hold_to_track`](../../bench/shot_to_load.py#L1183)) and the
  keep-outs cut ([`bench/shot_to_load.py::cut_keep_out`](../../bench/shot_to_load.py#L1112)).
- **Depth** ([`bench/shot_to_load.py::make_depth`](../../bench/shot_to_load.py#L4435)), on every load unless a person's
  words turn it off (`--no-depth WHY`, or the plan's `depth` with a `why`): the frames where two tracks or a keep-out overlap
  ([`bench/shot_to_load.py::overlap_frames_of`](../../bench/shot_to_load.py#L4376)) go through HyDen MoGe-2
  ([`bench/measure_moge2_depth.py::run_measurement`](../../bench/measure_moge2_depth.py#L280)), which writes
  `depth.npz` (full-canvas metric depth, a validity mask and the camera) into the capture folder. A load with no
  overlap frames runs no depth.

### 3. Reference: the video H3 is shown as `<Video 1>`

[`bench/shot_to_load.py::step_reference`](../../bench/shot_to_load.py#L4640) runs two tools:

- **The pose pass** ([`bench/pose_along_shot.py::run`](../../bench/pose_along_shot.py#L218)): Sapiens2 pose on every
  frame, shown only the tracked person (grey outside the track), so another face in the box is not read.
- **The head reference** ([`bench/head_reference.py::main`](../../bench/head_reference.py#L1223)): the body-model video
  cut to the head ([`bench/crop_reference_to_head.py::cranium_box`](../../bench/crop_reference_to_head.py#L267)), with
  the face drawn from the pose points ([`bench/draw_face_points.py::main`](../../bench/draw_face_points.py#L1906)). The
  drawing rules are in [`../masked_v2v/reference_and_prompt_rules.md`](masked_v2v/reference_and_prompt_rules.md).

### 4. Text

[`bench/shot_to_load.py::step_text`](../../bench/shot_to_load.py#L4993) writes the load's prompt from the plan and the
reference record ([`bench/load_text.py::main`](../../bench/load_text.py#L486)); whether a mouth is drawn decides the
mouth clause ([`bench/load_text.py::mouth_drawn`](../../bench/load_text.py#L214)).

### 5. Capture and plan: the masks the render will take, worked out and checked

[`bench/load_chain.py`](../../bench/load_chain.py) runs these in order:

1. **First** ([`bench/load_chain.py::step_first`](../../bench/load_chain.py#L1227)): who is who across the load's shots,
   as one row a shot of whose word each call is
   ([`bench/load_chain.py::who_is_who_file`](../../bench/load_chain.py#L1207)); then the capture
   ([`bench/capture_masked_run.py::files`](../../bench/capture_masked_run.py#L2525)), which builds every mask:
   - who owns each pixel two tracks claim ([`bench/capture_masked_run.py::owner_map`](../../bench/capture_masked_run.py#L615)),
     settled by depth where it ran ([`bench/capture_masked_run.py::read_depth_npz`](../../bench/capture_masked_run.py#L513),
     [`bench/capture_masked_run.py::_depth_medians_and_owner`](../../bench/capture_masked_run.py#L560): each person's
     median depth inside their own pixels, and the shared pixels to whoever is in front);
   - the head tube: SAM 3.1's head mask with Sapiens2 smoothed over three frames inside it
     ([`bench/head_tube_parts.py::head_tube_for_capture`](../../bench/head_tube_parts.py#L1919),
     [`bench/head_tube_parts.py::smooth_seg_in_tube`](../../bench/head_tube_parts.py#L789)), short dips bridged
     ([`bench/head_tube_parts.py::bridge_short_tube_dips`](../../bench/head_tube_parts.py#L661)) and a beard kept with
     the face where the plan says so ([`bench/head_tube_parts.py::face_piece_kept_whole`](../../bench/head_tube_parts.py#L1809));
   - parts held or emptied by size and face state ([`bench/capture_masked_run.py::grade_holds`](../../bench/capture_masked_run.py#L1006)),
     what lies off her head taken out ([`bench/capture_masked_run.py::off_head`](../../bench/capture_masked_run.py#L6026)),
     and her own hair outside the hole put back in ([`bench/capture_masked_run.py::her_hair_in_hole`](../../bench/capture_masked_run.py#L5390));
   - then the capture is held to what the nodes wrote ([`bench/capture_masked_run.py::verify_checks`](../../bench/capture_masked_run.py#L3394))
     and its risks written as flags ([`bench/capture_masked_run.py::preflight`](../../bench/capture_masked_run.py#L6586)).
2. **Plan** ([`bench/load_chain.py::step_plan`](../../bench/load_chain.py#L1245)): the render graph built and queued
   without sampling, so the node says what region it will regenerate
   ([`workflows/masked_job_graph.py::load`](../../workflows/masked_job_graph.py#L874) with `plan`).
3. **On plan** ([`bench/load_chain.py::step_on_plan`](../../bench/load_chain.py#L1261)): the capture made again on the
   plan's own grid ([`bench/capture_masked_run.py::predict_regions`](../../bench/capture_masked_run.py#L1559)), which is
   the region the render must match.
4. **Table** ([`bench/load_chain.py::step_table`](../../bench/load_chain.py#L1386)): the frame table, one row a frame.

### 6. Look, flags, render, land, page

- **Before a render**: a reader who did not make the capture looks through its strip; the flags printout is read
  and initialed ([`bench/flags_before_render.py::main`](../../bench/flags_before_render.py#L1378)), and render refuses
  without it ([`bench/load_chain.py::the_flags_read`](../../bench/load_chain.py#L696)).
  [`bench/inspect_mask_steps.py::main`](../../bench/inspect_mask_steps.py#L1314) audits the mask steps and the inputs'
  provenance.
- **Render** ([`bench/load_chain.py::step_render`](../../bench/load_chain.py#L1425)): the load's graph
  ([`workflows/masked_job_graph.py::load`](../../workflows/masked_job_graph.py#L874)) on the source's frames, one
  window on both clocks ([`workflows/masked_job_graph.py::window_for`](../../workflows/masked_job_graph.py#L559)),
  padded after the shot ([`workflows/masked_job_graph.py::window_pad`](../../workflows/masked_job_graph.py#L572)) with
  held copies of the edge frame across a cut ([`workflows/masked_job_graph.py::held_context`](../../workflows/masked_job_graph.py#L624)).
  Inside it, the Masked Source node ([`video_mask.py::MiniMaxH3MaskedSource`](../../video_mask.py#L2330)) turns the
  masks into the latent tokens H3 regenerates ([`video_mask.py::token_mask`](../../video_mask.py#L698)), zoomed in on
  a small subject ([`video_mask.py::zoom_box`](../../video_mask.py#L1932)), with the head reference, the still and the
  text as conditioning, and the song's own audio kept.
- **Land** ([`bench/load_chain.py::step_land`](../../bench/load_chain.py#L1502)): the render's region held to the
  plan's; the pose pass run on the render; the stacked review clip made.
- **Page** ([`bench/load_chain.py::step_page`](../../bench/load_chain.py#L1563)): the clip put on the owner's review
  page.

### 7. Laid, and the cut

- **Laid** ([`bench/laid.py::main`](../../bench/laid.py#L2763)): one line a kept piece in `data/<video>/laid.jsonl`, the
  only record of what stands.
- **The cut** ([`bench/rebuild_delivery.py::main`](../../bench/rebuild_delivery.py#L1023)): the ledger and table, then
  the assembler ([`bench/assemble_delivery.py::main`](../../bench/assemble_delivery.py#L3488)), which lays each piece by
  the node's own matte ([`bench/assemble_delivery.py::Matte.at`](../../bench/assemble_delivery.py#L948)) and puts back
  what a later pass must not cover ([`bench/assemble_delivery.py::restore_weight`](../../bench/assemble_delivery.py#L1415)),
  then the stack and the review pages. A second subject on the same shot is made as its own pass on this cut.

## The frame every masked text is written in

Every masked load prompt is generated from `plan.json` and the reference record by [`bench/load_text.py`](../../bench/load_text.py) (checked by [`bench/check_load_text.py`](../../bench/check_load_text.py)):
- **Current prompt specification (`<Subject 1>`, `<Subject 2>`, `[Shot 1]`, mouth & eye clauses)**: [`../masked_v2v/reference_and_prompt_rules.md`](masked_v2v/reference_and_prompt_rules.md).
- **General H3 prompting reference**: [`../prompting.md`](prompting/prompting.md) and [`../h3_references.md`](core/h3_references.md).
- **Historical `2026-10-04..2026-10-15` text experiments**: [`archive/2026-10-15_masked_v2v_chronicle.md`](archive/2026-10-15_masked_v2v_chronicle.md).

## A shot, made whole

Every shot of `<= 345` frames (`max(h3_rules.chain_lengths())`) renders as one single-window load on both clocks (`h3_rules.on_both_clocks`: `39, 90, 141, 192, 243, 294, 345`) at exact `24.0` fps (`24/1`), with each replacement pass built from a fresh preview on the newest whole-video cut:
- **Current whole-shot rules (`§1`, `§6`, `§8`)**: [`../masked_v2v/AGENTS.md`](masked_v2v/AGENTS.md) and [`../masked_v2v/chain_trace_and_remakes.md`](masked_v2v/chain_trace_and_remakes.md).
- **Step-by-step CLI commands (`shot_to_load.py`, `load_chain.py`)**: [`../masked_v2v/PROTOCOL.md`](masked_v2v/PROTOCOL.md).
- **Historical `2026-10-15` hall-shot recipe**: [`archive/2026-10-15_masked_v2v_chronicle.md`](archive/2026-10-15_masked_v2v_chronicle.md).

## A latent step is several frames, and its grid is fixed for a whole load

H3's video VAE encodes 17-frame clips into `(1, 4, 4, 4, 4)` latent steps (`video_mask.FRAME_PER_TOKEN`), and the DiT attends bidirectionally across the entire window:
- **Current frame math, latent-step boundaries, & the 2026-10-25 three-path context rule**: [`../masked_v2v/frames_rules.md`](masked_v2v/frames_rules.md), [`../masked_v2v/AGENTS.md`](masked_v2v/AGENTS.md) §2 & §6, and [`../research/masking/2026-10-25_context_frames_and_identity_on_2050.md`](research/masking/2026-10-25_context_frames_and_identity_on_2050.md).
- **Earlier window-context and edge studies**: [`window_context.md`](masked_v2v/window_context.md), [`window_edges_data.md`](masked_v2v/window_edges_data.md), and [`archive/2026-10-15_masked_v2v_chronicle.md`](archive/2026-10-15_masked_v2v_chronicle.md).

## A small subject, rendered zoomed in

`video_mask.py::zoom_box` (`render_region: "zoomed"`) crops and enlarges the window around the union of the subject's region boxes up to `plan.json`'s settled `zoom_up_to` (enforced by `cbd39fbe` at `bench/shot_to_load.py` and `bench/flags_before_render.py::recipe_check_of`):
- **Current zoom rules, `floor_zoom`, & `cranium` reference crop**: [`../masked_v2v/chain_trace_and_remakes.md`](masked_v2v/chain_trace_and_remakes.md) (Stage 5) and [`../masked_v2v/reference_and_prompt_rules.md`](masked_v2v/reference_and_prompt_rules.md).
- **Historical zoomed-render chronicle**: [`archive/2026-10-15_masked_v2v_chronicle.md`](archive/2026-10-15_masked_v2v_chronicle.md).
