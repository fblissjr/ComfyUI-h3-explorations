# The Sapiens2 pose model, first run: eight frames where the answer is plain by eye

2026-10-11, session mrfire. A throwaway first run on one clip: eight frames,
one pass each, no flipped pass, looked at by the session that ran it and by
nobody else yet. Nothing is built on it.

## What it buys

The owner ruled that of the perception models on disk that no code loads,
this one is wired first, because it serves head movement and expression,
and that it is tried on a handful of frames where the answer is obvious by
eye before anything is built (`docs/state/decisions.md`, 2026-10-11). This is
that handful. It says whether the model's named points land on lips, lids
and brows on this film, at the sizes its faces have, and whether two plain
numbers made from them agree with what the picture shows. Accepted when a
person who did not run it has looked at the eight pictures and agrees.

## How it was run

`Sapiens2ForPoseEstimation` and `Sapiens2ImageProcessor` from the installed
`transformers`, the checkpoint in the owner's store
(`facebook_sapiens2-pose-1b`), on the card beside an idle server, under
bfloat16 autocast as the part model is run. One person a frame: for the
opening shot the box of the tracked person, for the close-up the whole
frame as the box (Meta's own demo does that when no person is detected).
The processor's own crop and its own mapping back to the frame were used,
not this pack's. Source frames are the working copy at its own size;
render frames are at the canvas.

The checkpoint names its points `LABEL_0` to `LABEL_307`. The names used
here were read as text out of Meta's
`sapiens/pose/configs/_base_/keypoints308.py` in the reference checkout,
with its teeth points removed as that file removes them. That the order is
the checkpoint's rests on two things: the left and right pairs of that
list are exactly the checkpoint's own `flip_pairs`, and the points landed
on the features their names say.

## What the pictures show (stills, one session's look)

On all eight the lip points ring the mouth, the lid points lie on the
lids, the brow points above each eye, and the midline points run down the
nose to the chin. That holds on a face about a hundred pixels tall in a
render and on the lead's close-up with her head tilted far over, eyes shut
and mouth wide open over a telephone handset. On the opening shot the brow
points of one side sit a little high, where her fringe covers the brow.

**The pupil points are not usable.** Of the eighteen (indices 290 to 307)
about half land off the face on every frame, on a hand or the table, and
five of them carry the same low score on every one of the eight frames.
The eighteen iris points (272 to 289) are on the eyes.

*Added the same hour: a second session looked at the same eight pictures
and the saved points before opening this record, and found more than the
paragraph above says. Where the two differ, its reading is the more
careful one.*

- *The lip points do NOT sit on the lips on a small, soft face. On the
  four source frames of the second subject the group lies several canvas
  pixels above the part model's lip pixels (most on frames 30 and 148,
  where her head is turned or tilted), on lips little more than a dozen
  pixels tall, and only a small share of the lip points land on a lip
  pixel. By eye on 30 and 148 the upper points are on the skin between
  nose and lip. So "the lip points ring the mouth on all eight" was too
  strong: it holds on the two close-up frames and looks right on the
  renders, where there was no second reading to test it against. A mouth
  mask should not be cut from these points on her. The gap between the
  lips is another matter and still separates open from shut.*
- *The iris points keep a high score on a shut or lowered eye. The score
  does not know the eye is shut, so it cannot be used to say an iris is
  seen.*
- *The brows cannot be judged on the second subject: her fringe and its
  shadow cover them. On the close-up with the head tilted, the lid and
  brow points of one eye lie on the same dark stroke and it could not
  tell which it is. On the renders the brow points sit a little under
  the brows.*
- *The pupil points: it reached the same verdict by its own test (the
  centres carry the same two scores on every picture, wide eye or shut).*
- *Its own numbers, made another way (each over the face's height, not
  the mouth's or the eye's width, which is a poor ruler on a turned
  head), order the mouths and the eyes as its eye did, with one pair of
  near-equal eyes swapped. Among the six shut mouths the spread is a
  pixel or less: that is the floor of the measure, and a shut mouth never
  reads as nothing. It found the same thing on the render: the eyes about
  twice as open as the source's on both frames.*
- *What it would add before trusting the eye number: a blink and a
  half-open mouth, where the answer is at neither end.*

*Added in the evening, the middle of the eye measure, which the second
session asked for: the model was run on every frame of the lead's
close-up (source frames 1575 to 1729, the original at the canvas, her
tracked box), where her eyes go wide, shut and part open again, and
twelve frames spread over the range of the eye opening were looked at
in order of the number (opening over the distance from glabella to chin,
the measure of the later records). Under about 0.02 her eyes are shut on
all eight frames; at 0.032 they are open and at 0.062 wide. But at 0.021
and 0.025, two frames near the shot's end where she smiles and the frame
blurs, they also look shut. So the two ends are right and the middle is
not shown to be: a shut eye in a blurred or smiling face can read part
open. A second session found the same on its own crops of source frames
1700 to 1729. The score of the lid points stays high on those frames, on
a face a few pixels tall, and on a frame where a head is turned away, so
it says nothing about whether the lids were seen. The sheet:
`data/2026-10-11_sapiens2_pose_consecutive/leadcanvas_1575_eyes_by_rank_sheet.jpg`.*

*The same evening, the verdict on that run as trial 1 of
`docs/masked_v2v/perception_models.md`, "The trials", by a session that
did not run it and looked at every one of the 155 frames before reading
the paragraph above (the figures are the opening over the distance from
glabella to chin):*

- ***The eye figure passes on this clip.** It rises as her eyes open,
  holds while they are wide, falls as they close, stays flat while they
  are shut and rises again where she looks up; the ten largest steps
  from one frame to the next all lie inside those events. What it could
  not test: a blink. The shot has none, every change takes three to five
  frames, and the one closing happens while her head turns away.*
- ***Where the eye figure is trusted: at its two ends.** Under about
  0.020 her eyes are shut and over about 0.045 they are open; between
  the two it is not readable. Shut is not one number (lower with the head
  tilted or in profile than facing the camera), and near the shot's end
  the figure climbs slowly into the middle on blurred frames where no
  opening can be seen. Both lines are read off this clip and the kitchen
  shot and are starting values.*
- ***The mouth figure fails**, by the row's own words. It jumps on one
  frame of a fast head turn where the mouth neither closes nor reopens
  (the lower line of points lands on her teeth). A shut mouth facing the
  camera does not read shut: the lower "inner" line lies on the lip's
  outer edge, so the gap is the lip's thickness, at a score of the lip
  points near one, so the score does not warn. And an open mouth in
  profile behind a handset reads nearly shut, at a low score. So one
  value means a shut mouth in one view and an open one in another.*
- ***What the mouth figure can still do:** more or less open between
  frames of one view, with the lip points' score at or above about 0.8
  (read off this clip, a starting value). Never open or shut from a
  line. A render's mouth is read by eye.*

*The judge's figures and sheets: `data/2026-10-11_mrtie_trial1_read/`.
Not done by anyone: a blink; a smaller face.*

## Two numbers against the eye

`lip gap`: the two centre points of the inner lips apart, over the width of
the mouth. `eye opening`: the centre points of the upper and lower lash
lines apart, over the length of the upper lash line; left and right.

| frame | face height, px | lip gap | eye opening, left / right | by eye |
|---|---|---|---|---|
| source 30 (second subject) | 149 | 0.08 | 0.41 / 0.28 | eyes open, mouth shut |
| source 72 | 144 | 0.06 | 0.13 / 0.15 | looking down at her hands, mouth shut |
| source 100 | 145 | 0.05 | 0.12 / 0.12 | lids lowered, mouth shut in a slight smile |
| source 148 | 150 | 0.08 | 0.15 / 0.12 | lids lowered, mouth shut |
| today's render, frame 100 | 100 | 0.05 | 0.24 / 0.25 | eyes open, looking down; mouth shut |
| today's render, frame 148 | 109 | 0.06 | 0.27 / 0.25 | eyes open; mouth shut |
| source 1600 (the lead, close) | 284 | 0.43 | 0.48 / 0.46 | eyes wide, mouth open, singing |
| source 1660 (the lead, close) | 189 | 0.47 | 0.01 / 0.06 | eyes shut, mouth wide open |

Both numbers order the frames as the eye does: shut, lowered and open lids
are three separate bands, and the two singing frames stand far from the
shut mouths.

Two things they say about the lane, each on one or two frames:

- **The render's eyes are more open than the source's.** On frames 100 and
  148 the source's lids are lowered and the face-and-hair render's are
  about twice as open. That is one measured piece of what the owner named
  as missing in that render (no facial expression).
- **The capture's flag for a mouth open with no voice was wrong on the
  frame looked at.** It marked source frames 92 to 113 from the part
  model's lip classes; on frame 100 the mouth is shut by these points and
  by eye. That measure's own note says a shut mouth in a smile reads
  large.

## What it does not say

Anything about a small face: the smallest here is about a hundred pixels
tall, and the street and telephone shots are far smaller. Anything about
steadiness from frame to frame: no two frames here are neighbours. Whether
this pack's own crop of a tracked person suits the model as well as the
processor's. What the model costs beside a loaded video model on the card.
And nothing here makes the video model follow an expression: it measures
one.

## Where things are

Untracked: the one-off script, the names as read, the points and scores of
each frame and the table, `data/2026-10-11_sapiens2_pose_first_frames/`.
On the share: the eight pictures, each the face enlarged beside the same
crop with the points drawn,
`Video/fun/pose_first_frames_2026-10-11/`.
