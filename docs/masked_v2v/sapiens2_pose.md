# The Sapiens2 keypoint model in the masked lane: the face drawn on the reference, and the eye measure

**Sapiens2's keypoint model names 308 points a frame, and the lane uses its face points two ways: drawn as lines on the
body-model video to make the head reference of every face, head or hair load, and read back as figures, of which only
the eye opening at its two ends is trusted.** It runs along a load's frames in `bench/pose_along_shot.py`, called by the
chain at the reference step (on the source) and at the land step (on the render), on the folder `h3_config.POSE_MODEL`;
no node loads it. Its mouth figure failed, so a mouth is never measured from it; the reference's mouth is drawn from its
lip points only where they lie on the part model's mouth, and only on windows where the person sings or talks. The
overview of every perception model is [`perception_models.md`](perception_models.md).

## How it is wired

- **The pass.** `bench/pose_along_shot.py run` reads the frames it is told to, fits a source to the canvas as the lane's
  loader does, and gives the model one box a frame with every pixel outside the tracked mask (grown by
  `pose_along_shot.HOLD_GROW_PX`) set to grey (`HOLD_GREY`), so the face read is the tracked person's. A box alone
  drew another woman's face on 11 of 47 frames of a close group (3167-3213, 2026-10-14) and passed. The record lists
  the frames whose face middle is off the mask (`bench/_lib/face_points.py::face_middle_on_mask`), and no reference is
  made from those. Point names are Meta's, in `../../vendor/meta_sapiens2/pose_keypoints308.json` (2026-10-11).
- **Who calls it.** `bench/shot_to_load.py reference` on the load's own frames of the source; `bench/load_chain.py
  land` on the render (a land whose pose pass did not finish is a failed land, 2026-10-13, `503f9cf3`). The head tube
  reads the same points for its face state ([`sapiens2_parts.md`](sapiens2_parts.md), "The head tube").
- **One definition of each figure.** `bench/_lib/face_points.py` (`lid_rings`, `lip_gap`, `face_ids`). The lines a
  figure is read by are `h3_config.POSE_READING`, each a starting value from one or two clips, with its provenance
  beside it.

## The head reference: the face points drawn on the body model

**The form for every face, head or hair load since 2026-10-12** ([`data/AGENTS.md`](../../data/AGENTS.md), "Settled
for the masked lane"): `bench/head_reference.py` cuts the body-model video ([`sam3d_body.md`](sam3d_body.md)) to one box
round the head (`bench/crop_reference_to_head.py`, sized from the head track with room above the crown since
2026-10-19, `crop_reference_to_head.CROWN_ROOM`, `HEAD_TRACK_ROOM`, which keeps the crown and makes the face smaller in
the reference: [`docs/state/decisions.md`](../state/decisions.md), 2026-10-19), and `bench/draw_face_points.py` draws the source's
face on it at its true size: lids, creases, brows, the iris in the contrast form, and the mouth. What the drawing carries
by default is `bench/shot_to_load.py::REFERENCE_DEFAULT`, printed by every job as THE DEFAULTS THIS LOAD CARRIES
(`bench/shot_to_load.py::defaults_carried`):

- **the eye**: white sclera and black iris inside the lids (`contrast`, 2026-10-14, `264c0840`), iris and pupil rings
  only above `POSE_READING["eye_ring_score_at_least"]`;
- **the mouth**: `auto`, drawn where the lip lines lie on the class map's mouth (`bench/head_reference.py::MOUTH_CHECK`,
  2026-10-14 and 2026-10-15), decided per window by the plan's voice: drawn on every face frame of a window where the
  person sings or talks, left out on a silent one (`5ebaecfe`). A drawn mouth on a still, closed mouth parted the lips
  on every frame of 278-293, and with none they stayed closed ([`docs/state/decisions.md`](../state/decisions.md),
  2026-10-18);
- **closed lips as one line, an open mouth as an outline filled dark** (`--closed-lip-line`, the default since
  2026-10-19, `8ac7e928`; `draw_face_points.lip_shape`, `draw_face_points.mouth_shapes`);
- every mark clipped to the grey head's silhouette; a darkened head stays off (it made the filled eyes stand out on
  `second_hfp_0278i`).

What renders of this form carried, and its forerunner, the lid marks of 2026-10-11 (`bench/draw_lid_marks.py`,
`h3_config.LID_MARKS`: her lids at the source's level on the kitchen shot, not repeated on the sung close-up), are in
[`renders_by_form.md`](renders_by_form.md). How a load's text names the drawing is [`text.md`](text.md).

## Reading a render against its original

`bench/read_render_against_original.py` (2026-10-11) sets a render's face beside its original's from the points of
each: the eye opening split by where the original's eyes are open and shut, and where the head sits in canvas pixels.
It writes the lip gap as a column and says nothing of a mouth. A mouth is read by eye.

## Trial 1's verdict

The keypoint model's trial ([`perception_trials.md`](perception_trials.md), "The trials", row 1) was judged by a
session that did not run it and looked at every frame of the lead's sung close-up before reading the runner's account
([`2026-10-11_sapiens2_pose_first_frames.md`](../../bench/results/2026-10-11_sapiens2_pose_first_frames.md), the dated
addition headed "The same evening, the verdict on that run as trial 1", which holds the lines and the frames):

- **The eye figure passes on that clip and is trusted at its two ends**: it says shut and it says open, and between the
  two it is not readable. Shut is not one value across views of the head. Not tested: a blink, and a smaller face.
- **The mouth figure fails.** It jumped on a frame where the mouth did not move, read a shut mouth facing the camera as
  open at a high score, and read an open mouth in profile as nearly shut.
- **The score of the lid points does not say the lids were seen**: it stays high on a face a few pixels tall and on a
  head turned away.
- What was built on the pass is the eye reader above, whose lines are starting values read off two shots of one clip.

## Limits

- On a small, soft face its lip points sat above the lips the part model found (the same record), so the part
  classes, not the points, are what a mouth mask is cut from.
- Its pupil points scored 0.20 to 0.28 and lay hundreds of pixels off the face on one shot (the comment beside
  `h3_config.POSE_READING`); only the iris ring is drawn.
- The lid points sometimes miss a lowered or closing eye (2026-10-14, `018dd813`), and a laughing squint draws open
  lids (`da273a6f`). Gaze is a known limit, horizontal in the render and vertical in the points (`07e262ab`).
- The model places lid points on the back of a head and scores them well; frames to leave bare come from a person.
- Not run: a blink or a half-open mouth on consecutive frames; the model as a node.
