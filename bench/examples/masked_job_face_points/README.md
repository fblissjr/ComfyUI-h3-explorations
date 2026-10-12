# An example masked job: a face replaced, with the face points drawn on the body model cut to the head

last updated: 2026-10-12

**An example of one load that rendered, not a recipe and not a text to
copy.** It is the shape of the load the owner judged on playback on
2026-10-12: the lead's face replaced on one hall shot (source frames 1222
to 1305) of one video, one seed, with the reference video made the way the
owner ruled that evening for every face, head or hair replacement: the
body model cut to one box round the head, the pose model's face points
drawn on it, and a text that says exactly what that video is. Beside the
same load with lid marks instead (`../masked_job_lid_marks/`), the owner's
words on playback were "fp_cv has the correct eyes", then on the other two
hall shots "more natural for mouth" and "eyes looking slightly better in
the direction the original is looking", and "keep those they're good to
go". The readings from frames had called the eyes a draw
(`data/2026-10-12_lead_g_1222_fp_cv/reading_mrforest.md` and
`reading_blind/reading.md`, untracked); what shows in motion is the
owner's to see. The decision and its figures are in
[`docs/wiki/decisions.md`](../../../docs/wiki/decisions.md) (2026-10-12,
"always cropped to the head, with the face points drawn") and the rule in
[`docs/wiki/masked_v2v.md`](../../../docs/wiki/masked_v2v.md), "Rules of
the lane". It is the owner's standard, provisional on holding up across
the rest of the video.

## What is here

| file | what |
|---|---|
| `job.example.json` | the job file of that load, with every path of that day's media replaced by a placeholder |
| `text.example.txt` | the whole text that load was given, byte for byte. An example prompt: written from the frames of that one window and true of that window only |
| `README.md` | this page |

Nothing here is read by a tool, a check or a graph. A job is built from a
copy under the untracked `data/`, with real paths and a text of its own.

## What differs from the lid-marks example

Everything the other example explains key by key holds here (read it
first: `../masked_job_lid_marks/README.md`, "The job file, key by key").
Three things differ:

1. **The reference video** (`loads[].motion_file`). There: the whole-frame
   body model with two white ovals drawn on its eyes. Here: the body model
   cut to one box round the subject's head over the whole load and
   enlarged to the canvas, with the pose model's face points drawn on the
   cut video as thin white lines (each eye an outline with the iris ring
   inside it, the brows, the nose; no dots, the mouth left out). One
   command makes it with its sheet and one record naming every input:
   `bench/head_reference.py` (its docstring has the three steps it runs:
   cut the plain body model with the subject's points, cut the source the
   same way for the sheet, draw on the cut video). It needs the subject's
   body model and points from the preview (`body_pose.py`,
   `bench/pose_along_shot.py run`). On this load the box was 284 by 213
   of the canvas, enlarged about 3.6 times, her face 212 px tall in it
   against 59 px in the frame.
2. **The two reference sentences of the text** say exactly what that video
   is: "a closer view of the head and shoulders of a plain grey body model
   of a person on a black ground, posed as the person moves, with its eyes,
   eyebrows and nose drawn on its face as thin white outlines; the eyes are
   outlined with the iris inside them, and they open, close and look where
   the person's eyes do", and in `retention_analysis` that "the drawn
   outlines themselves" are not used. Where the box holds more than the
   head and shoulders (a raised arm, the chest), the sentence names it: a
   text says exactly what the video holds (the owner's rule for texts,
   2026-10-12).
3. **The subject.** The lead, face only (`parts: ["face_and_neck"]`, margin
   16), one keep-out (the third subject). The other example is the second
   subject's face and hair at margin 64. The form is the same for both.

**The one exception.** A face too small or too turned for the lines takes
the plain body model with no drawn face, and the note says why: on
604-948 the lead's face is 65 px at the median and under the drawing line
(`h3_config.POSE_READING["lid_gap_px_to_draw"]`) on 379 of 406 frames, so
her load there has the plain body model and the text says so.

## What was and was not exercised

The job file is the load's own (made by `bench/sweep_loads.py` from the
lid-marks load in two sweeps, the reference and then the text), with the
`what` line added at the top and the same ten kinds of value replaced as
in the other example. **Nothing was built, planned or rendered from this
file**: the placeholders are not files. No check holds it; the builder's
docstring (`bench/masked_job.py`) is the reference for the keys.
