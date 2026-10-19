# What is known from renders, by form

Which forms of masked load have rendered, on what, what each carried, and where each result is written. One shot or
one seed is said as that. A form with no row has not rendered, or its render was not read. Owner verdicts on a
video's renders are that video's state (`data/<video>/verdicts/verdicts.jsonl`), not this page's.

## What is known from renders, by form

| the form | what it was seen on | what it carried | what it did not show | where it is written |
|---|---|---|---|---|
| a whole subject on the fast chain, the generic prompt | the first clips, 2026-10-04 | the person replaced from the still, in the hole the mask gives | movement: the new subject faces the camera when the original turns, and a late start carries the turn with the original's hair and clothes | [`limits_and_directions.md`](limits_and_directions.md), "Replacing the person"; `../../bench/results/2026-10-04_masked_v2v_first_run.md`, `../../bench/results/2026-10-04_masked_v2v_band.md` |
| the subject's own frames on grey as `<Video 1>` (`motion_reference`), on ref2va | the arms of 2026-10-05 | the subject's movement, at `h3_config.MASKED_MOTION_STEPS` | fl2va ignores the reference at any step count; ref2va loses it at eight | [`overview.md`](overview.md), "What it is"; `../../bench/results/2026-10-05_masked_v2v_motion_arms.md` |
| the head and upper body on the ref2va motion graph | one window of one clip at two lengths and one seed; a typed text beside the node's wording | the owner called it solid on playback, and the two wordings both good and not told apart (stacked on the share: `Video/compare_stacked/body_345_p7_typed_beside_node_over_mask.mp4`) | the same region on the fast chain was called broken | `../../bench/results/2026-10-06_masked_v2v_body_window_arms.md` |
| `keep` inside a face-only region | one clip, one window, one seed, 2026-10-10 | the jewellery that was kept | the kept tokens held the skin beside it too, and the face moved back toward the original's | [`pieces.md`](pieces.md), `keep` |
| a body mesh as the motion video | one window, one shot and seed, as an appended video | the action, with no action words | things inside the region came back as other things until one sentence named them | [`pieces.md`](pieces.md), `motion_video` |
| a body model with lid marks as the motion video | the second subject's kitchen shot, face and hair, one seed a render; the lead's sung close-up, face only | on the kitchen shot her head following the source's, and her lids at the source's level with the marks enlarged | following in time; a second seed; on the close-up it did not repeat | "The motion video as a body model, with marks drawn on its face", below |
| the body model cut to the head with the face points drawn | the lead's hall shots, 2026-10-12 | judged better than the lid marks by the owner on playback | the standard is provisional on holding up across the video | [`working_rules.md`](working_rules.md), "Rules of the lane"; [`data/AGENTS.md`](../../data/AGENTS.md) |
| a small subject rendered zoomed in | one street shot, one load, one seed | it ran clean and the face is replaced where it shows, in stills | whether it reads as the reference at that zoom | [`pieces.md`](pieces.md), "A small subject, rendered zoomed in" |
| a continuation window after its context | renders of 2026-10-08 and 2026-10-10 | the context's pose | the motion video's pose where it had moved on | [`window_context.md`](window_context.md) |
| a short load with its held tail open | three loads that had faded to the original, 2026-10-10, one seed each | the still's person to the load's last frame | a second clip | [`window_context.md`](window_context.md), "The held tail of a short load" |

## The motion video as a body model, with marks drawn on its face

**How to read this section: patterns with their conditions, not a recipe** (the owner's ruling at the close of
2026-10-11). Each pattern did a named job under the conditions it was tried in. Begin from the pattern nearest your
case, say which of its conditions your shot shares and which it does not, and treat the first render as a test of the
pattern. When a pattern fails on a new shot, that is a row for the table, with the narrower condition it now has, not
a reason to delete one. The form of every head or head-and-hair replacement since 2026-10-12 is the body model cut to
the head with the face points drawn (`bench/head_reference.py`); the lid marks below are its forerunner.

| the pattern | what it was for | the conditions it was seen under | why we think it worked | where it stands |
|---|---|---|---|---|
| a body-model video of the subject as the load's motion video | her head following the source's | one shot; one person, seated, her face AND her hair redrawn | the model takes movement from a video reference, and with her hair redrawn nothing else in the picture fixes where her head is | held on that shot. On a load that redraws a face only, her head sat as near the source's without the video, so it is not shown there |
| the source's eye openings drawn LARGE on that body model's face | her eyelids at the source's level | the same shot; the source's eyes lowered for most of it; each mark several of the model's cells wide | marks that fit inside one cell did little and the same marks enlarged did, under the same words: a mark has to be large enough for the model to see | held on that shot as a level, never as following in time. Did not repeat on a second shot, which differed in several ways at once: another person, a face only, marks grown less against her face, eyes fully shut for a long stretch |
| a text that says what the reference video is and that she follows it, and nothing of what her face does | the model using the reference, with a render still able to show whether the reference or the words did it | the same shot, with the enlarged marks | the words license the reference and the picture carries the content: a longer text that related her eyes to the figure's did no better | held on that shot, on one seed and nearly one text; the owner picked it on playback ([`text.md`](text.md), "The text: the fragment") |
| the pose model's eye figure on the source and on the render, taken apart by where the source's eyes are open and where they are lowered | a first check of a render's eyes before a person looks | the renders of one shot, with one render made twice as the floor | on that shot it fell on the same sides as the owner's verdicts on playback | trusted at its two ends only; its mouth figure is not trusted |

The records hold every figure and caveat:
[`2026-10-11_kitchen_shot_head_and_eyes_two_tests.md`](../../bench/results/2026-10-11_kitchen_shot_head_and_eyes_two_tests.md),
[`2026-10-11_kitchen_shot_lid_marks_on_the_body_model.md`](../../bench/results/2026-10-11_kitchen_shot_lid_marks_on_the_body_model.md),
and, for the second shot,
[`2026-10-11_lead_closeup_lid_marks_did_not_carry.md`](../../bench/results/2026-10-11_lead_closeup_lid_marks_did_not_carry.md).
The owner's rulings behind it are in [`docs/state/decisions.md`](../state/decisions.md) under 2026-10-11.

**On the second shot it did not repeat.** On the lead's sung close-up (face only, one seed) her eyes read as in the
render of those frames with no reference video at all: open through the long stretch where the source's are shut.
Her mouth sings as before, and her head is as that earlier render's. From the tables the render with marks and the
render without sit at the same number on both stretches; that number lies in the middle the eye figure cannot read.
No render of this load was made twice, so no floor exists and "the same" is all that is said. **Why is not known**:
the person, a face only against a face and hair, the mask and its margin, how much the marks were grown, how a shut
eye comes out when drawn, and a sentence that she sings all differ together. The record names the render it puts to
the owner next: the same load with the marks grown as much as the kitchen's and nothing else changed.

### What the reference video was

One video, the load's motion video, the size of the canvas, one frame for each frame of the load:

- **A plain grey body model on a black ground**, posed on every frame as the person in the source is posed. Nothing
  of the source's picture is in it. Its own face is a neutral modelled face that never changes: the body model sets
  the jaw and every expression to zero ([`perception_models.md`](perception_models.md), "The face"). On the frames
  looked at its mouth is shut.
- **On that face, over each eye, a white shape with a thin black edge** filling the space between her upper and lower
  lash line on that frame: an almond where her eye is open, a sliver where it is lowered or shut. Lids only. Each
  shape is grown about its own centre by one factor for the whole shot; in the render the owner picked the factor was
  three (`--grow`), which makes each shape several cells of the model's grid wide.

The file is untracked: `data/2026-10-11_fun_gate_inputs/second_mesh_with_lids_0014_x3.mp4`.

### How it was made: the nodes and the code

1. **The body model.** A job's preview draws it from the track: `MiniMaxH3SubjectBoxes` (`subject_boxes.py`) gives one
   box a frame round the tracked person, and `MiniMaxH3BodyModelLoader`, `MiniMaxH3BodyPose` and
   `MiniMaxH3BodyMeshVideo` (`body_pose.py`) pose the model in it and draw it, at `h3_config.MASKED_JOB_BODY_MODEL`,
   `MASKED_JOB_BOXES`, `MASKED_JOB_BODY_POSE` and `MASKED_JOB_MESH`.
2. **The face points.** The Sapiens2 pose model on every frame of the source, fitted to the canvas as the loader fits
   it, in the tracked person's box: `bench/pose_along_shot.py` (`run` needs the card; `read` needs none and says what a
   shot's points can carry, beside `h3_config.POSE_READING["lid_gap_px_to_draw"]`). The model folder is
   `h3_config.POSE_MODEL`; the points' names are Meta's list, `vendor/meta_sapiens2/pose_keypoints308.json`, held to
   Meta's own file by `bench/check_meta_sapiens2.py` (the tool's docstring, "The names"). No node runs that model.
   The kitchen renders used the one-off it replaced, `data/2026-10-11_sapiens2_pose_consecutive/run.py`.
3. **The drawing.** `bench/draw_lid_marks.py` draws the lid shapes from those points (each grown by
   `h3_config.LID_MARKS["face_px"]` over the shot's median face height, a starting value from one shot, or a factor
   given by hand); it writes the marked video, a record and a sheet over the original, reads its own video back, and
   its `verify` is for a second reader. Frames where a head is turned away are named by a person. The kitchen shot's
   video was drawn by the one-off it was written from, `data/2026-10-11_fun_gate_inputs/draw_lids_on_mesh.py`.
   `bench/draw_face_points.py` draws the pose model's face points as lines at the face's true size
   (`bench/check_draw_face_points.py`), and `bench/crop_reference_to_head.py` cuts a reference video to one box round
   the shot's head and enlarges it (`bench/check_crop_reference_to_head.py`). **One command for the form the lane uses
   now**: `bench/head_reference.py` cuts the plain body model and the source by the same box, draws the face on the cut
   body model with no dots and the mouth left out, and writes the reference, its sheet and one record naming every
   input by path and the commits it ran on (`bench/check_head_reference.py`). A reference cut this way no longer lies
   over the source pixel for pixel, and the line of text that says what `<Video 1>` is has to say so.
4. **The load.** A job file's load names the video (`motion: mesh`, `motion_file`, `motion_first`;
   `../../workflows/masked_job_graph.py`) and goes through the chain like any load ([`running_a_job.md`](running_a_job.md)).
5. **The render's graph** as the builder wrote it: `MiniMaxH3MaskedSource` with `motion_reference` on `a video I wire`,
   `motion_vae` on, the video at the canvas's short edge, `start_from` on `noise`; `MiniMaxH3AudioFreezeSong`;
   `MiniMaxH3Sol`; `MiniMaxH3AppendRefImage` for the still. Saved beside the job:
   `data/2026-10-11_gate_p4_0014_job/second_hf_0014_p4.render.json`.

### What it carried on that shot

- **The body model carries her head.** With it her head follows the source's; without it, and without the sentences
  that go with it, her head holds nearly still (the two-tests record).
- **Nothing carried her lids until the marks did.** With the enlarged marks her lids come down to about the source's
  level where the source's are lowered.
- **It is a level, and following in time is not shown.** Where the source's eyes are open no load opened hers as far,
  and the shot has no blink.
- **On playback the owner picked the load with the enlarged marks and the text of [`text.md`](text.md)**, and called
  the other loads with enlarged marks as good; their saved lines are in the untracked `data/fun/owner_answers.jsonl`.

### Why we think it worked, and what is not shown

What the renders support, and no further; each is one shot and one seed.

- **The body model, shown as a video, is what her head follows** on the kitchen load, where her hair was redrawn with
  her face; not shown for a face-only load.
- **Larger marks did more, under the same words.** At true size each lay inside one cell of the model's grid and her
  lids moved a little; enlarged, they brought her lids to the source's level. Part of what size buys is the drawing
  itself: the dark edge and the encoder take a strip along a mark's outline, so a thin mark is mostly edge
  (`../../bench/draw_lid_marks.py`, the comment at `EDGE_PX`). The same true-size marks under the vendor guide's layout
  brought her lids most of the way.
- **The words say that the video is followed; the picture says how.** Saying what is retained of the video, or what
  the shapes are for, changed nothing by the measures of her lids, her head and her lip gap.
- **Her look comes from the still and not from the figure, as far as was looked**: no render shows a grey face or the
  shapes in a box round her eyes. One reader found a light streak on each upper lid over a run of frames in the load
  with enlarged marks (`data/2026-10-11_mrtie_gate_o_read/`); nobody has judged it on playback.

Not shown: that she follows in time; a second seed; any other shot. The form of the words moved her face once,
between two loads with the same true-size marks, for a reason nobody has.

### Limits known today

- **A small face has no lids to draw.** On the lead's street shots the gap between her lash lines at whole-frame size
  is too small to draw marks from or to read a render by. The pose model's own score for the lid points does not say
  this: it stays high on such a face and on a head turned away.
- **A load rendered zoomed in takes a motion video only as a file the load names, and the node does not cut that
  video to the box** (`video_mask.zoomed_source`; the builder's rule is `workflows/masked_job_graph.py::RENDER_REGIONS`).
  So the head-cut reference can go with the zoom; a whole-frame body model cannot. The preview stage has no zoom, so a
  face the pose model cannot read at whole-frame size gets no drawn points from the zoom alone.
- **Her head sits a little further from the source's under the plain text**, while its turn is the same in every
  render; the cause is not separated.
- **The figure's mouth never moves.** On the sung close-up her mouth sang as in the earlier render with the singing
  sentence and no reference video; the figure's still mouth did not shut hers (one shot, one seed).

### Reading a render of this form

The pose model on every frame of the source and of the render: the eye opening taken apart by where the source's eyes
are open and where they are lowered, and where the head sits in pixels beside any correlation, since a correlation
hid the head's offset on this shot. A level is stated with a floor beside it, the same render made twice. The eye
figure is trusted at its two ends and not in between, and the mouth figure is not trusted
([`perception_models.md`](perception_models.md), "The trials"). `../../bench/read_render_against_original.py` does
this reading from the points files of the source and each render; without two makes of one render handed in as a
floor it prints every difference and calls none a difference, and it writes the lip gap as a column and says nothing
of a mouth. Its figures are defined in `../../bench/_lib/face_points.py`, read against `h3_config.POSE_READING`, each a
starting value.

### What comes next for the video

By the rulings of 2026-10-11: segmentation and depth are shown to the model as further layers of this one video,
never as more reference videos, added one at a time and each read as a single render against the kept one: the pose
model's own drawing of the face points (now the standard form), then the part colours, then depth shading.
