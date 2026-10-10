# What was knowable before the render

Every fault of a masked job that was found after a render, set against the
data that existed, or was cheap to get, before anything sampled. The owner's
question (2026-10-10): for each of these, what did we have in advance that
would have told us it would happen. The answer for most of them is that the
data was there and nothing made anyone act on it.

**How this page is kept.** One row per fault, added the day it is found.
A row is not finished until its last column names a rule that raises it
before sampling. A row whose predictor does not exist says so. Dated lines
are added under the table when a row's state changes; rows are not rewritten.
The rules are `bench/capture_masked_run.py`'s (`preflight --gate`); the plan
they read is the song node's own (`<name>_plan.json` and the planned region
files a preview writes). The order a job runs these in is the masking
board's `guide-order-of-operations-masked-job`.

## The four sources, all available before sampling

| Source | What it costs | What it holds |
|---|---|---|
| the trackers' shot tables and tracks | a no-sampling preview | who was taken on each shot and at what likeness, how many people were detected, each subject's size and box per frame |
| the plan | nothing: arithmetic on the loads | each window's frames, where windows join, which frames are kept from the window before, which are held past the load's end, which latent step each frame is in, where the cuts fall |
| the class maps and the owner map | the same preview | which part is where, whose a contested pixel is, where a part is empty |
| a pose pass and the voice table on the SOURCE | a short pass on the card; the voice table needs none | the head's angle, the chin, each wrist against the nose, per frame; which frames are voiced; the mouth's opening |

## The faults of 2026-10-10

| Fault, as found after a render | What would have said so before | Did we have it | The rule that raises it now |
|---|---|---|---|
| a whole shot left as the source for both subjects; a viewer saw it | the shot table: "absent", people detected, the likeness a few hundredths under the line | yes, flagged "iffy", no outcome recorded | `absent_with_people_on_screen` and `load_has_a_shot_without_the_subject`, top level since 0.276.0 |
| a short load fades back to the original toward its end | the plan: most of the window is the load's last frame held as plate, and the subject is on that frame; the track: the subject is a small share of the frame | yes, both, joined by nothing | being built (mrdeer): no planned window shows the model, as a frame to keep, the subject it is replacing; the remedy is the Masked Source's `held_tail` |
| a head turn not made on a continuation, the mesh followed on a load with no kept frames | the pose pass on the source: a large change of the head's angle just after a window's join, behind kept frames that hold the old pose | no: the pose pass was run on the renders, hours later | not built: the pose at the first new frame against the pose on the last kept frame |
| the source's picture repainted on the far side of a cut | the plan: a cut inside a latent step that holds the subject on one side only | yes, as arithmetic nobody had written | `region_carried_across_a_cut`, `region_shared_across_a_cut`; the node's cut gate leaves those frames alone |
| a face drawn for one frame where the part was emptied | the plan: a frame with an empty mask inside a latent step whose other frames have one | yes | `region_on_a_frame_with_no_mask`, 0.279.0 |
| the original's face or hair back where pixels of the subject were kept | the masks: kept pixels inside the part | yes; the flag fired and the lead overrode it | `kept_pixels_inside_the_part` |
| a mouth that stays at its ordinary opening where the source's opens wide | the source's mouth opening against the voice table: an open run on unvoiced frames | yes, two columns nothing joined | the report in `mouth`; not yet a flag of the gate |
| a small subject that does not take its mesh's pose | the track: her share of the frame for the whole load, and the tokens a mesh at that size gives her head | yes | `subject_small_for_the_whole_load` (its line passed her by a hair: read the size) |
| two trackers on one person; one subject's region over another | the tracks' overlap, the owner map | yes | `two_tracks_on_one_person`, `region_over_another_subject`, `contested_by_class_too` |
| a filled part that drew a face on hair, a hat or an arm | the class map under the filled mask | yes | the graded fill (`part_grades.json`), 0.273.0 |
| a sentence written from one shot drawn in another | which shot's frames each load's text was written from | no: nothing records it | not built |
| held frames written past the shot because the audio outlasted the load | the audio's length against the load's | yes | the song node's report line for held frames inside the track (with `held_tail`); the session builder cuts the audio |
| the motion video at half the canvas | the graph: `motion_short_edge` against the canvas's short edge | yes | not built as a rule; a pose pass later showed the size was not what failed that render |
| the first window of a pass in the original's look, on one seed | nothing known | no | none: a short first render per subject is the only test (the order of operations, step 6) |
| long hair drawn over a face in profile for a long stretch | partly: the pose pass says how long the head is in profile; nothing measures the reference's hair | partly | none |

## What the table says

Of the fifteen, eleven were in data already on disk before the render that
showed them. In four of those the tool had raised a flag: one at a level
that did not block, one overridden, two read as good news. In the others the
two columns that predict the fault existed and no rule joined them. Two
needed a pass that is cheap and was not run up front (the pose pass on the
source) or a field nobody records (which shot a text was written from). Two
have no predictor known.

So the missing thing was seldom data. It was a rule that blocks, and running
the cheap passes on the source before the first render and not on the
renders after it.

## Dated lines

- **2026-10-10 (mrdeer): the short load that fades to the original has its
  rule.** `original_shown_in_the_held_tail`, top level, 0.281.0
  (`bench/capture_masked_run.py::flag_held_tail`), from the node's plan:
  the frames held past the load's end that reach the model clean, when the
  subject's mask is on the load's last frame. Read back through it, the two
  loads of the day raise it and it names the latent step that went first.
  It can only be checked on the node's own plan (`--run NAME:preview=`); a
  what-if `--plan` has no windows and the gate says so
  (`held_tail_not_checked`).
- **2026-10-10 (mrdeer): that fault was on a third render nobody had seen.**
  `changed` now reads a render's two ends against the source every time it
  runs (`ends_on_the_source`, 0.281.0). Run on the day's renders it named
  the two known loads and a third, a short face load whose last frames fall
  toward the source, which I had called clean (the frames are in that
  render's `changed.json`). The reading is the outcome
  that grades the rule, not a way to find the fault: by the time it reads,
  the render exists.
- **2026-10-10 (mrdeer): the head turn behind kept frames has its rule.**
  `kept_frames_far_from_the_pose_ahead`, iffy, 0.282.0 (`flag_kept`): the
  source's head over a continuation's new frames against the last kept
  frame, from the node's plan and a pose table with 3D keypoints, with
  `kept_frames_not_checked` when there is no such table. Not yet run on a
  real continuation's plan: the one case it is built from was measured
  after the renders (`bench/results/2026-10-10_kept_frames_against_the_mesh.md`).
- **2026-10-10 (mrdeer): the open mouth with no voice is a flag of the
  gate.** `mouth_open_with_no_voice`, iffy, 0.282.0 (`flag_mouth`), when the
  capture has the subject's class map and the voice table.
- **2026-10-10 (mrdeer): one cell of the table says more than happened.**
  "The flag fired and the lead overrode it", of the kept pixels: there was
  no gate and no override on that render. The flag was in the list, the
  render was queued with it standing, the original's hair came back on the
  frames it named, and the outcome was recorded afterwards
  (`KEPT_IN_PART` was lowered from that measurement, 96f204f6). The
  override exists since 0.266.0.
