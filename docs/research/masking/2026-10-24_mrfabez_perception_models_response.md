# 2026-10-24: a reviewer's account of the perception models, checked against the code

Last updated: 2026-10-24. Written by mrfabez (a Claude session advising the owner, not the day's lead) against main at
`988a03dc`. Nothing here was run on the card: every line is a read of the code, the dated records under `bench/results/`
and the wiki, with the file and line that holds it. The companion page is `docs/wiki/meta_perception_models.md`, which
carries dated notes where this reading overtook a sentence; `docs/wiki/decisions.md`, 2026-10-24, logs what the page
used to say.

## 1. What was reviewed

At the close of 2026-10-23 a Gemini session (Spark, not the day's reviewer) wrote a one-page account of how SAM 3.1,
Sapiens2 and SAM 3D Body relate to that day's bugs (`internal/gemini/2026-10-23_z_end_of_day_from_gemini_spark_perception_models-annotated-by-claude.md`, its annotated copy;
untracked). Its thesis: signals the pack already computes and saves are read by nothing, while the one place the part
model hit a hard limit (the third subject's grey goatee) was fought with a wider margin instead of a model that can
name a beard. It then gave one remedy per bug.

The thesis is right about the shape of the problem. Several of the remedies are not what they seem: two name what the
next-day plan already does, one names something that is already on, and the two headline model changes run against
decisions this repo made and recorded. The owner asked whether the account holds. This file is the answer, one claim a
row, and what follows from each.

## 2. The claims, one a row

Verdicts: **holds**, **partly** (right about the fact, wrong about what follows), **does not hold**. "Where" is the
line that settles it. "What follows" is the action, or the reason there is none.

| # | the claim | verdict | where | what follows |
|---|---|---|---|---|
| 1 | The part model's class table has no beard or facial-hair class, so a goatee is labelled as whatever lies behind it (clothing, background) | **holds** | `vendor/meta_sapiens2/seg_classes29.json`, `names` | The beard needs a rule that does not depend on a class. That is the next-day plan's replacement of the 0764f585 rule with a region grown downward inside the subject's own track (`internal/claude/2026-10-23_mrtx/tomorrow_plan.md`, section 1B), checked first on the previews' raw class videos. Another Sapiens checkpoint would not change this |
| 2 | `sam31_head_tube` prompts SAM 3.1 with the word "head", which stops at the jaw | **does not hold** as stated | `bench/head_tube_parts.py::sam31_head_tube` takes a `prompt` word and a `tracker` callable; the chain's call (`run_head_tube_parts`, the call at line 1516) passes neither, and no `tracker=` exists under `bench/`. The word is written into the record and reaches no model. The head track the parts are clipped to is the subject-track node's own SAM 3.1 propagation from a head mask seed (`subject_track.py::_span_head`; on by default since 2026-10-19, `h3_config.SUBJECT_TRACK["track_head"]`) | Nothing to change in a prompt. The jaw stop is the head track's rows (`docs/wiki/decisions.md`, 2026-10-23, "what was laid stopped above it"), and the fix lives where row 1 says |
| 3 | Hand the beard to SAM 3.1's open-vocabulary text prompt ("head, beard") | **does not hold** as a chain change | `subject_track.py`, the module docstring's first paragraphs: core's text-prompted video track was tried on 2026-10-04 and set aside for named behaviors (new objects across cuts, an object cap, a keep-alive that blanks a track, a detector mask that overwrites the tracked one); the chain tracks from a mask with no text prompt by design | A text prompt at one seed frame, to find a beard region once and hand it to the track as a mask, is a smaller idea than the reviewer's and untested. It is second behind row 1's geometric rule, and only if that rule fails on collar or hand cases. Not traced: whether core's detector finds "beard" at all at his face sizes |
| 4 | The pose model runs inside the whole-person box, so on 3969 it read her lit forearm as a face; crop it to the head tube instead | **partly** | The pass runs inside the subject's track (`bench/shot_to_load.py`, the "pose pass", `--box-mask` on the track file; `bench/pose_along_shot.py`, `HOLD_GROW_PX`). The head tube is itself what failed on 3969-3970 (the part went onto the lit arm; `docs/wiki/handover.md`, lesson D), so a crop to it can hand the pose the same wrong region. And the track hold exists because a read inside the track box once drew another woman's face (`bench/shot_to_load.py`, the paragraph on 2026-10-14 under the reference step) | A one-variable test, not a fix. The next-day plan's fix for 3967 is the neighbour fill and a drawing-only word for the named frames |
| 5 | Nothing reads the SAM 3D Body table's 3D points; cross-checking the 2D face points against it would have caught 3969 | **partly** | The table is written beside every preview (`body_pose.py::pose_table`). The capture reads its 3D points (`bench/capture_masked_run.py::pose_state`: which way the body, hips and head face, the chin's lift, each wrist's distance from the nose), and so does `bench/measure_subject_yaw.py`. What nothing reads is the grey head's own points against the marks drawn on its face | That printout is the plan's step, written before this review (`internal/claude/2026-10-23_mrtx/tomorrow_plan.md`, section 1C, "the grey head's own points are already on disk"). Do it as written: a printout, no gate, cutoffs read off kept drawings (0014g, 3909b, 4726b, 4602c) before any frame is flagged automatically |
| 6 | Find the chin from the rig's own vertex instead of scanning grey pixels; use the head's pitch or visibility to know when the face is turned down | **holds** as a direction, not as a day's work | `body_marks.json` indexes vertex groups (`marks`, `vertices`); `bench/draw_face_points.py::find_mesh_chin_y` scans luma for a dark gap, and is called twice: for the lit silhouette the mouth gate reads and for every drawn mouth's placement | The projected chin is the better long-term form of the chin guard. It changes every drawing with a mouth, so it goes behind the replay of `gem-chin-and-lips-guard` on kept drawings and gets a one-variable A/B first (darkening the head once gave googly eyes: `docs/wiki/decisions.md`, 2026-10-18) |
| 7 | The tracker's per-frame presence score is saved and nothing reads it; reading it would have flagged the drift onto other women | **holds** | `subject_track.py`, "The tracker's own score": each frame's score goes into the shot table, and "nothing reads it yet: whether it warns of a loss is a test still to run" | The one new item this review adds. A reader of the saved score along a span, bench only, CPU, no restart: run it first on the two drift cases (4053 from 4099; 2823's 3110-3166; `internal/claude/2026-10-23_mrtx/bugs_for_tomorrow.md`, the item on the lead's track moving onto other women) and on kept spans as controls. It prints; a person reads it. Section 4 has the shape |
| 8 | The vision-trunk signatures could be checked along a continuous track to catch a steal | **partly** | The signatures are kept per shot for comparing a shot's people with another tracker's subject (`subject_track.py`, "What a refused shot leaves behind"); nothing compares them along a track. The 2026-10-10 record says a likeness could not name a person across shots (`bench/results/2026-10-10_who_is_who_across_shots.md`) | Along one track the question is different (did this track change person), and untested. Second behind row 7, on the same test set, and only if the score alone does not separate the cases |
| 9 | Track every subject jointly with the multiplex model, or feed the other subjects' boxes as negative prompts, so a drawn box cannot take another subject | **does not hold** today | Core has the multiplex state (`comfy/ldm/sam3/tracker.py::MultiplexState`). The memory correction applies only to a call that tracks one object (`sam31_corrections.py`, docstring item 3), so joint tracking drops it or needs it extended. The corrections module holds three numeric corrections and no negative-prompt path | The case cited (4647's box on 4697) is closed by 513783e6 (a drawn box names one person who is no other subject) and the box's withdrawal (`data/fun/cast/cast.jsonl`). No new design is owed for it |
| 10 | Memory selection and depth awareness would have kept 1419's head across the occlusion | **does not hold** | Memory selection is on by default for one-object SAM 3.1 calls since 2026-10-17 (`sam31_corrections.py`; `bench/sam_memory_selection.py`) and did not prevent it. The loss was the pack's own seed loop breaking on the first occluded gallery head (`subject_track.py:1237`), fixed on 2026-10-24 (988a03dc). Depth is in the capture as the opt-in `--depth`, settling which subject owns a shared pixel (`bench/results/2026-10-18_moge2_depth_on_crossings.md`); it is not in the tracker | The reviewer recommended something already running. Depth in the tracker is a new design nobody has asked for |
| 11 | A normals model (Sapiens-Normal or MoGe normals) would separate a protruding beard from a collar | **not checked** | No normals model is on disk (`docs/wiki/meta_perception_models.md`, "Depth and geometry": "no use in this lane that anybody has named") | Parked. Row 1's rule is the test to run first |
| 12 | (A detail) The body model in use is the ViT-H release | **does not hold** | `h3_config.MASKED_JOB_BODY_MODEL` names the DINOv3 file; the ViT-H loader exists and no job names it | Minor, but a reader building on the review would load the wrong file |

## 3. What reads which model today

The state the review described, as the code has it on 2026-10-24. "Reads it" names the consumer of the saved output,
which is where the review's thesis bites.

| model | where it runs in the chain | what it saves | what reads that |
|---|---|---|---|
| SAM 3.1 (`h3_config.SEGMENTER`) | the subject track node, from a mask seed with no text prompt; the head track from a head mask seed (`subject_track.py`); three runtime corrections and memory selection (`sam31_corrections.py`; `docs/research/masking/2026-10-18_sam31_minimal_core_fixes.md`) | the track, the head track, the shot table with each frame's presence score and the per-shot signatures | the capture and every tool after it read the tracks; **nothing reads the score**; the signatures are read across shots only |
| Sapiens2 segmentation (`h3_config.SAPIENS2["segmentation"]`) | the part node on every load that takes a part (`sapiens2_parts.py`); clipped to the head track by `bench/head_tube_parts.py` | the class map; the parts | the capture, the mouth gate, the strangers and keep-out tools. No facial-hair class exists to read |
| Sapiens2 pose (`h3_config.POSE_MODEL`) | the pose pass of every load with a reference, inside the subject's track (`bench/pose_along_shot.py`) | the points file, the lid and lip figures | `bench/head_reference.py` and `bench/draw_face_points.py` (the face drawn on the body-model video, the chain's default layer); the eye figures of `bench/read_render_against_original.py` |
| SAM 3D Body, DINOv3 (`h3_config.MASKED_JOB_BODY_MODEL`) | the body pose node, fed the mask, for the motion video (`body_pose.py`) | the mesh video and the pose table (2D and 3D points a frame) | the capture's `pose_state` and `flag_pose`; `bench/measure_subject_yaw.py`. **Nothing compares its head's points with the marks drawn on its face** |
| HyDen MoGe-2 metric point (`vendor/meta_moge2/hyden_moge2.py`) | only on a job's opt-in `--depth`, on the frames where two tracks or a keep-out overlap (`bench/measure_moge2_depth.py`) | a depth file in the capture folder | the capture, to settle shared pixels. The field of view it recovers is measured and not fed to the body node (`body_pose.py`, "Not here") |
| Sapiens2 matting | nowhere (`h3_config.SAPIENS2["matting"]` is `"none"`) | | |
| Sapiens2 pointmap, HyDen relative depth, SAM-Audio, PE-AV | nowhere | | |

## 4. What to do, in order

Each item says who it waits on and what it costs. None needs a server restart.

1. **Already in the day's plan; nothing to add from the review.** The third subject back on the default margin; the
   beard rule off the class and inside his own track, checked on the raw class videos before one render of 5675; the
   pose-table printout per drawing; the drawing-only word for frames a reader names; the chin guard replayed on kept
   drawings before it lands.
2. **A reader of the tracker's saved score along a span** (row 7). Bench only, CPU. Input: a shot table; output: a
   line per frame with the score and its run-to-run change, and the frames where it falls below the span's own level,
   as data with no verdict. Test set: the two drift spans in row 7 and three kept spans with no known fault. It passes
   as a tool only if the drift spans show a fall the kept spans do not; otherwise the record says so and the idea
   rests. Owner's word needed: none until it has a record.
3. **The projected chin and head pitch from the rig** (row 6), as a replay against kept drawings first, with the
   current scan's chin beside it frame by frame, and no landing until one render decides it. Owner's word needed:
   none until the replay has a record.
4. **A seed-frame text prompt for "beard"** (row 3), only if item 1's geometric rule fails on a collar or hand case,
   and then as a measurement of what core's detector returns on his frames, not as a chain change.

Not to do, with the reason on the page: joint multi-object tracking (row 9, the one-object memory correction); depth in
the tracker (row 10, no case); swapping the Sapiens checkpoint (row 1, the class table is the limit); running the pose
inside the head tube as a fix rather than a test (row 4).

## 5. Not run

- Everything in this file. It is a read; the records it cites were run by others on the days they name.
- Whether core's detector finds a beard by text at his face sizes (row 3).
- Whether the saved presence score falls before either drift case (row 7). The 2026-10-04 postmortem says it was seen
  falling once (`docs/research/postmortems/2026-10-04_feature_lotsofpeopledance-end-to-end.md`, item 4); that is one
  sighting, not a test.
- Whether the rig's projected chin lands where the scan's does on kept drawings (row 6).

## 6. Where the corrections went

- `docs/wiki/meta_perception_models.md`: dated notes of 2026-10-24 where the chain overtook a line (the keypoint model
  as the chain's pose pass; depth as the opt-in ownership measure; the face drawing as the layer a load is shown; the
  preflight rows as the capture's flags; the part model's missing facial-hair class; SAM 3.1's corrections and unread
  score).
- `docs/wiki/decisions.md`, 2026-10-24: what the page used to say.
