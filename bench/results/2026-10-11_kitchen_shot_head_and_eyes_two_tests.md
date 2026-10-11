# What carries her head and what does not carry her eyes: two tests on the kitchen shot, read with the pose model on every frame

2026-10-11, session mrfire; second reader mrcolor. One shot (the second
subject, source frames 14 to 154 of `fun_0000_1919_24p.mov`), one seed.
Each test is two changes from the reference load, said as two. The owner
had the two test clips on the local page when this was written and had
not judged them.

## What it buys

The owner called the face-and-hair render of this shot close and named
what it lacks: no facial expression or head movement. Two leads were on
the table for the cause, each from one clip: the faceless body-mesh motion
video, and the size of the redrawn area. The owner asked for both to be
tested before anything is built on pose data. This is those two renders,
and with them the first run of the pose model on consecutive frames
(trial 1 of `docs/wiki/meta_perception_models.md`, "The trials"), which
gives the question a measure. It applies to this shot. Accepted when the
owner has looked at the clips and agrees with what the figures say.

## The loads

All three go through `bench/masked_job.py`, nothing posted by hand; the
reference is the load of `2026-10-11_close_render_made_again.md`.

| | part redrawn | margin | motion video | text |
|---|---|---|---|---|
| reference ("gate") | face and hair | 64 px | the body mesh | that record's |
| test A | the same | 64 px | **none** | **its four statements about the motion video deleted** |
| test B | **face and neck only** (the largest piece of the face, eyeglass, lip and tongue classes) | **16 px** | the body mesh | **its statements that her hair is the picture's deleted** |

The text changes were the second reader's: with no video given, a text
that still tells the model to follow a figure in a video makes test A
unreadable for every outcome but "nothing changed", and the same holds for
a text that asks for hair a face-only region cannot draw. Deletions only
(in B, two words changed for agreement). The one-change forms were built
and not rendered. So neither test says which of its two changes did what
it shows, and neither is read against the other.

Before sampling, by the second reader from the saved plans: test A's
region equals the reference's on every latent step and frame; test B's
mask equals the face-only file on every frame, and its region is about a
third of the reference's share of the frame.

## The measure

The Sapiens2 pose model on every frame of the source and of each render,
the tracked person's box, the model's own processor. Per frame: an eye
opening and a mouth opening, each the mean distance from every point of
the upper line to the nearest point of the lower line, over the distance
from the glabella to the chin; and where the nose, the chin and the
glabella are. "Head lean" is the angle of the glabella-to-chin line in the
picture. A correlation is over the frames of the shot, against the
source's series. The second reader worked its own figures from the same
tables with its own formulas before it saw the first reader's, and they
agree to the second decimal where both computed the same thing.

| | head lean, correlation with the source | head lean, spread over the shot (degrees, 5th to 95th centile) | nose path, correlation (across, down) | eyes, median opening | eyes, correlation |
|---|---|---|---|---|---|
| source | | 23.9 | | 0.019 | |
| reference | +0.96 | 21.7 | +1.00, +0.99 | 0.031 | −0.41 |
| test A | +0.29 | 4.2 | +0.12, +0.62 | 0.036 | −0.08 |
| test B | +0.99 | 23.4 | +1.00, +1.00 | 0.031 | −0.10 |

The mouth is shut in the source and in all three renders, at the floor of
the measure; its correlations are not read.

For the lead's close-up of frames 1575 to 1729, two renders of 2026-10-10
that were given no motion video, the same measure with the whole frame as
the box:

| | mouth, median opening | mouth, correlation with the source | head lean, correlation | eyes, median | eyes, correlation |
|---|---|---|---|---|---|
| source | 0.105 | | | 0.016 | |
| text with the singing sentence | 0.094 | +0.60 | +0.72 | 0.037 | +0.26 |
| text without it (and with a sentence about a handset that the other lacks) | 0.005 | −0.63 | +0.72 | 0.046 | +0.33 |

## What the pictures show (stills; neither reader judged likeness)

The second reader, the head cropped on eight frames across the shot: the
source turns away at the start and bows to the bowl through the middle
with lowered eyes; the reference follows that pose on all eight with eyes
lowered but more open; test A faces the camera on all eight with a slight
smile; test B follows the pose on all eight under her own dark hair. It
found no seam, no second hairline and no switch back to the source in
either test on those frames, and from pixels that test B keeps her own
hair, with the band nearest her face drawn again a little lighter and
redder than her own. Both tests treat her hands as the reference did.

## What it says

- **Her head does follow the source's in the reference render**, closely.
  The owner's "no head movement" is not what the measure finds on this
  shot; what it finds missing is below.
- **The body-mesh video is what carries her head when her hair is
  redrawn.** Without it (and without the sentences that go with it) her
  head holds nearly still. The owner said the same on playback of a
  render of 2026-10-10 made without it. Two renders, one shot.
- **A face-and-neck region at a small margin keeps her head and her own
  hair**, with the mesh still given. Nothing here shows what that region
  does without a mesh on this shot; the lead's close-up shows a face-only
  region holding the head with none.
- **Neither test changed her eyes.** In every render they read wider open
  than the source's throughout. That is a steady offset: this shot has no
  blink to test the measure on, and the renders' faces are smaller than
  the source's, so part of it could be size.
- **The mouth follows the source only where the text says she sings.**

So the lead that the mesh holds her face still is wrong in its main part,
and the size of the redrawn area is not the cause either. The open
question is the eyes, and nothing the model is given says her lids are
lowered.

## What it does not say

Which of a test's two changes did what it shows. Anything on a second
seed or a second shot. Whether the eye measure holds on a blink or on a
face much smaller than these. Whether either test reads as the still: no
likeness was judged. And the measure is a score: it is given to no model.

## Where things are

Untracked: the job files and texts, `data/2026-10-11_fun_gate_inputs/`
(`gate_a2_0014`, `gate_b2_0014`); the captures with the second reader's
looks, `data/2026-10-11_gate_a2_0014/` and `data/2026-10-11_gate_b2_0014/`;
the one-off script and one table a clip,
`data/2026-10-11_sapiens2_pose_consecutive/`. On the share: the renders
under `Video/mrfire/gate_a2_0014/` and `Video/mrfire/gate_b2_0014/`, and
each stacked with its masks and frame numbers under
`Video/mrfire/review_2026-10-11/`.
