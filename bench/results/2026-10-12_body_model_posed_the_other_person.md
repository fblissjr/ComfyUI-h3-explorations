# The body model posed the other person for half a second, and what was changed

lane: masked video to video
verdict: no render; three previews of one shot. Given a box alone, the body model was posed as the person standing behind the subject on twelve frames. The model's own mask prompt, which both upstreams feed, put the two ends of that stretch right and left the middle wrong. Shown the subject alone it is on her on every frame, and where the box alone had been right her head moved a few pixels and the joints the picture does not show moved most. Run twice since (the dated note): two runs of one form agree to a fraction of a pixel, so the forms' differences are real. Not judged for a whole person's hands.

2026-10-12, session mroh; the fault and each form of the fix were looked
at by a second session, mrf, on its own cut of frames before it read
mroh's reading. One video (`fun_0000_1919_24p.mov`), one shot (source
frames 604 to 1009), one subject. No render was made: everything here is
a job's preview, which samples nothing. Nothing was run twice when this
was first written; the dated note under the tables has the floor, from
a second run made after the commit.

## What it buys

A masked load can show the model a body-model video of its subject, so
that a redrawn head or person moves as the original did
([`../../docs/wiki/masked_v2v.md`](../../docs/wiki/masked_v2v.md), "The
motion video as a body model, with marks drawn on its face"). That video
is only as good as the body it was posed from. This record says that on
frames where a second person stands in the subject's box the body model
was posed as the second person, how that was found before anything was
rendered from it, and what the node now does. It applies to every shot
where two people overlap. It is accepted for a load that takes a head
from the body model; for a whole person's hands it is not judged.

## What was found

The preview of the second subject on 604 to 1009 (job `second_k_0604`)
drew a body-model video that is a seated figure for the shot, and for
twelve frames a standing figure bent over the table: the lead, who
stands behind the second subject on those frames.

- **Seen first**, by mroh, on the sheet `bench/draw_lid_marks.py` lays
  over the original before its video is used: on source frame 766 the
  tile shows a hand and a chest where her face is. Then on a sheet of ten
  frames from 744 to 782.
- **Measured.** The body model's nose (the first of its 2D points in the
  preview's pose table) against the nose the pose model places on the
  original (`bench/pose_along_shot.py`), on the canvas, every frame of
  the shot.
- **Not the track.** Her tracked box is steady through the twelve frames
  (the capture's `subjects/second/per_frame.csv`, `track_box`). The
  body-model node was handed that box and nothing else, and the other
  person fills it.

## Three forms, one preview each

The same preview three times: the same span, subjects and picks. `t1`
and `t2` ran on uncommitted code under the builder's `--override`, which
their queue records hold.

| form | what the body-model node was given | job |
|---|---|---|
| box only | the subject's box a frame | `second_k_0604` |
| mask prompt | the box, and the subject's mask as the model's own mask prompt with a score of one, which is what Meta's estimator and ComfyUI's own node do when a track exists | `second_k_0604_t1` |
| alone | the box, the mask prompt, and the frame with the picture further than `body_pose.ALONE_MARGIN_PX` from her mask replaced by the model's mean colour | `second_k_0604_t2` |

### The fault: the body model's nose against the original's

Canvas pixels, 406 frames. "Far" is further than half the height of her
face on that frame, a line chosen to list frames by, not a threshold.

| form | median | nine frames in ten under | most | frames that are far |
|---|---|---|---|---|
| box only | 7.7 | 10.5 | 218.0 | 756 to 767 |
| mask prompt | 7.7 | 10.5 | 201.1 | 757 to 766 |
| alone | 10.0 | 12.2 | 17.2 | none |

Frame by frame on the stretch:

| source frame | box only | mask prompt | alone |
|---|---|---|---|
| 755 | 9.9 | 9.6 | 9.6 |
| 756 | 218.0 | 16.8 | 9.2 |
| 757 | 213.1 | 50.5 | 9.1 |
| 758 | 209.1 | 43.0 | 8.7 |
| 759 | 207.7 | 130.0 | 6.7 |
| 760 | 208.8 | 201.1 | 6.9 |
| 761 | 203.6 | 194.8 | 8.1 |
| 762 | 194.8 | 193.7 | 8.7 |
| 763 | 186.7 | 185.9 | 6.0 |
| 764 | 183.3 | 182.7 | 7.8 |
| 765 | 185.2 | 183.0 | 7.6 |
| 766 | 196.1 | 89.1 | 8.7 |
| 767 | 194.7 | 7.3 | 6.4 |
| 768 | 5.4 | 6.0 | 6.1 |

The mask prompt put the two ends of the stretch back on her and left the
middle on the lead. Shown her alone, the body model is on her on every
frame.

### The price: how far the body model moved where the box alone had it right

Source frames 604 to 1009 less 755 to 768, 392 frames. For each part,
the mean distance its points moved against the box-only preview, as the
median over frames, with nine frames in ten under the figure in
brackets. Canvas pixels.

| part of the body | mask prompt | alone |
|---|---|---|
| head (nose, eyes, ears) | 0.8 (1.3) | 2.5 (4.3) |
| shoulders | 0.8 (1.7) | 3.1 (5.1) |
| elbows and wrists | 1.4 (2.3) | 7.6 (11.2) |
| hips | 1.7 (3.9) | 21.9 (31.8) |
| knees and below | 6.2 (10.3) | 50.7 (116.1) |
| hands | 5.1 (14.7) | 15.4 (34.0) |

Her hips and legs are behind a table on every frame of this shot, so
those points are the model's guess in all three forms.

*Dated note, 2026-10-12, 14:19Z, mroh: the floor.* The alone form was
run a second time, as the preview the loads are built from (job
`second_k_0604_b`, on commit e5242d8a; the first run is `t2`, the same
code before it was committed). Between the two runs the tracks, the
part and the held-things mask decode equal on every frame, and the
body model's points agree to a fraction of a pixel: over the 392 control
frames the head's points moved 0.000 px at the median and 0.220 px at
most, the hands' 0.002 and 0.341, and the largest single move of any
part is 2.149 px, at the hips on one frame
(`data/2026-10-12_body_model_mask_test/second_0604_alone_two_runs.csv`).
So every figure in "The price" above is outside the floor: the forms
really differ by those amounts, and two runs of one form do not. This
is a floor for two makes of one preview on one machine; it says nothing
about which form is nearer the truth.

## What the figures cannot say

- **Which body model is nearer the truth** on the frames where the forms
  differ by a few pixels. The pose model's nose is another model's
  reading of the same frame, not the truth, and the alone form sits a
  little further from it at the median.
- **Anything about a whole-person load.** The hands moved more than the
  head, and nobody has looked at whether they moved toward her hands or
  away. This shot's load takes only a head from the body model.
- **Whether the margin is right.** `ALONE_MARGIN_PX` is the part node's
  own margin and it was enough here; no other value was tried.
- *When first written this list ended: no floor, two runs of one
  preview not compared. The dated note above has the floor.*

## What was changed

`MiniMaxH3BodyPose` takes an optional `subject_mask`, appended. Wired, it
gives the model its mask prompt and shows it the person alone (the form
`t2` tested); unwired, the node is as it was. A job's preview graph wires
the subject's own track to it (`workflows/masked_job_graph.py`). The
mask prompt is both upstreams' behaviour. **Showing the person alone is
this pack's and neither upstream's**: it is the part node's own function
(`sapiens2_parts.alone`), reused, and it rests on this one shot.
`bench/check_body_pose.py` has a case that reads the mask back from what
the model is handed and was red with the mask left out of the batch.

## What is now suspect

Every body-model video made before this change on a shot where two
people overlap, and the renders that were shown one. In this video that
is, at least, the third subject's whole-person renders on 1222 to 1631,
where he and the lead are together; nobody has measured them, and the
measure above needs a pose pass on the original of each shot. The
second subject's picked render on 14 to 154 is of a shot she is alone
in.

## A gap in what the tools show

Nothing printed by the preview, the capture or its gate said that a body
was on the wrong person: the pose table's own columns (how much of a
body is outside the frame, whether a hand was refined) read the same
either way. It was caught by looking at a sheet made for another
purpose. The figure that shows it needs only files a capture already
reads: the body model's nose against the subject's own part or track on
each frame. It is not built; a per-frame column in the capture's pose
rows, printed with what it was measured on, is where it would go.

## Where the files are

- The three previews, on the share under the server's output folder:
  `Video/fun/mroh/second_k_0604/`, `second_k_0604_t1/`,
  `second_k_0604_t2/`; each holds its body-model video
  (`..._second_mesh_00001.mp4`, frame 0 is source frame 604) and its pose
  table. Their job records: `data/2026-10-12_second_k_0604*_job/`.
- The pose model's points on the original:
  `data/2026-10-12_fun_second_inputs/second_0604_points.npz`, with the
  per-frame table beside it.
- The one-off that made the tables above, and one row a frame for each
  pair of forms: `data/2026-10-12_body_model_mask_test/`.
