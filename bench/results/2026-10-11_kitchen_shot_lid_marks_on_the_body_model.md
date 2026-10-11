# Lid marks drawn on the body-model video, and the words that go with them: renders of the kitchen shot

2026-10-11, session mrfire; readers mrcolor and mrtie, each of whom wrote
its reading before seeing another's. One shot (the second subject, source
frames 14 to 154 of `fun_0000_1919_24p.mov`), one seed a load. Every load
went through `bench/masked_job.py`, nothing posted by hand, and each is a
change of the reference load of
[`2026-10-11_close_render_made_again.md`](2026-10-11_close_render_made_again.md)
("the gate"). The owner had watched one of them (o1x3) on playback when
this was written and had not picked among the last three.

## What it buys

[`2026-10-11_kitchen_shot_head_and_eyes_two_tests.md`](2026-10-11_kitchen_shot_head_and_eyes_two_tests.md)
left one thing open on this shot: her head follows the original's and her
eyes read wider open than the original's in every render, because nothing
the model is given carries her lids. The owner ruled out a sentence about
her eyes (it is written a shot at a time and hides whether the text or the
reference did it) and asked for the pose model's points on the body-model
video instead, with a text that says what the video is and that she
follows it (`docs/wiki/decisions.md`, 2026-10-11). These renders say
whether the model takes anything from marks drawn on that video, at what
size, and under which words. It applies to this shot. Accepted when the
owner has picked on playback and the same form has held on a second shot.

## The loads

The marks: on each frame, the opening of each eye of the original, as the
Sapiens2 pose model's lash-line points give it on the source fitted to the
canvas, filled white with a thin black edge on the body model's face
(`data/2026-10-11_fun_gate_inputs/draw_lids_on_mesh.py`). Lids only: the
record of the pose model's first run says why lips and pupils are not
drawn. At true size a mark is inside one cell of the model's grid; the
enlarged marks are each grown three times about the eye's own centre and
reach across the bridge of the nose.

| load | motion video | text |
|---|---|---|
| the gate | the body model | takes only the head from the video |
| text | the same | the gate's with its five passages of edit-talk deleted, nothing added |
| o1 | marks at true size | the gate's, widened to head and facial expression; says what the marks show |
| o2 | marks at true size | the same content laid out in the vendor guide's form (a second subject for what is taken) |
| o1x3 | marks enlarged | o1's, word for word |
| p1 | marks enlarged | the plain form: what the video is, then that she follows its movements and its facial expressions; nothing of head, eyes or what she does |
| p2 | marks enlarged | p1's, with the video's retention line saying what is retained and what is not used |
| p4 | marks enlarged | p1's, with the video's line saying the shapes on its face are part of its facial expression and are not drawn; no colour named |

Two things about the texts, found by the readers before any render was
read. o1, o2 and o1x3 each carry a half-sentence relating her eyes to the
reference's, so against the gate each changes two things together, the
marks and that clause; among the three the comparisons are single. And
the texts of all but `text` keep the gate's edit-talk, held constant on
purpose. p1, p2 and p4 have no word of her eyes.

Before sampling, for every load: the second reader read the text whole
against the frames, diffed the built graph against the one it is a change
of, and compared the planned region with the gate render's saved region
(equal, tokens and mask, on every load).

## The measure

The pose model on every frame, as in the two-tests record: an eye opening
(the mean distance between the lash lines over the distance from the
glabella to the chin), a mouth opening made the same way, and the lean of
the glabella-to-chin line. The original's eyes are open on source frames
14 to 33 and lowered from about 45 on; the shot has no blink. "The floor"
is the 10th's render of the gate load against today's: the same seed and
the same graph, so it bounds what the measure moves by between two makes
of one render, and nothing about a second seed.

| | eyes, median | eyes, frames 14 to 33 | eyes, frames 45 to 154 | eyes, mean difference from the gate | eyes, correlation with the original | mouth, median | head lean, correlation with the original |
|---|---|---|---|---|---|---|---|
| the original | 0.0192 | 0.0285 | 0.0190 | | | 0.0140 | |
| the gate | 0.0305 | 0.0251 | 0.0304 | | −0.42 | 0.0133 | +0.96 |
| the 10th's render (the floor) | 0.0318 | 0.0252 | 0.0319 | +0.0012 | −0.45 | 0.0133 | +0.96 |
| text | 0.0303 | 0.0255 | 0.0313 | +0.0008 | −0.53 | 0.0152 | +0.95 |
| o1 | 0.0266 | 0.0250 | 0.0270 | −0.0026 | −0.35 | 0.0303 | +0.95 |
| o2 | 0.0241 | 0.0257 | 0.0236 | −0.0053 | +0.19 | 0.0138 | +0.96 |
| o1x3 | 0.0202 | 0.0239 | 0.0204 | −0.0081 | +0.35 | 0.0195 | +0.95 |
| p1 | 0.0203 | 0.0255 | 0.0203 | −0.0078 | +0.63 | 0.0129 | +0.95 |
| p2 | 0.0200 | 0.0254 | 0.0202 | −0.0080 | +0.65 | 0.0131 | +0.94 |
| p4 | 0.0199 | 0.0260 | 0.0199 | −0.0081 | +0.66 | 0.0134 | +0.95 |

How far her head sits from the original's, which the correlation does not
show (the nose point's distance in canvas pixels, the mean of the shot and
of source frames 114 to 133): the gate 5.1 and 10.0; o1x3 6.0 and 12.1; p1
6.9 and 15.3; p2 6.9 and 15.3; p4 7.0 and 15.5. The blind reader found
it; the other two confirmed it in their own tables.

## What it says

- **The edit-talk does nothing that shows on this load.** `text` is the
  gate's piece again: the saved video latents are 0.9997 alike, the
  picture differs by about three grey levels of 255 on her face and hair,
  and the table reads as the floor does. One load, one seed; the sound
  line was left in.
- **The model takes her lids from the marks, as a level.** Where the
  original's lids are lowered hers are lower than the gate's in all three
  overlay renders, beyond the floor, in the order o1, o2, o1x3, and with
  the enlarged marks they sit at about the original's level. The step from
  o1 to o1x3 is the same words and larger marks, so that step is the
  picture's and no sentence's.
- **It is not shown to follow in time.** On frames 14 to 33, where the
  original's eyes are open, no load opened hers further; o1x3 is the
  lowest there. Its eyes fall from that stretch to the rest by about a
  third of the original's fall, where the gate's rise. So: a level
  reached, and a direction followed weakly. With no blink in the shot
  nothing here tests a quick change.
- **Her head follows as in the gate render** in the three overlay loads.
- **The form of the words moved her eyes too** (o2 against o1, same
  marks), by about half the step the larger marks gave. Whether a second
  seed of o1 would show the same is not known.
- **o1's mouth reads parted through the shot**, where the original's, the
  gate's and o2's are closed. Three readers found it three ways and none
  can say why. o1 and o2 were given the same video.
- **Nothing white from the marks was found on her face**, by a count on
  every frame or on enlarged stills. One reader found a pale fleck on each
  upper lid in the overlay renders, where the marks sit, which the gate
  render does not have; nobody has judged it on playback.
- **The plain form of text does at least what the longer one did for
  her eyes.** p1 against o1x3 is the same marks and different words: her
  lids sit at the same level from frame 45 on; on frames 14 to 33 they
  are as open as the gate's, where o1x3 had them lower; and they fall
  from that stretch to the rest by a little over half the original's
  fall. So the clause relating her eyes to the reference's, and the
  phrase saying what the marks show, were not what lowered her lids: p1
  has neither. Her mouth is closed, as the gate's.
- **Saying what is retained of the video changes nothing that can be
  measured** (p2 against p1): every figure is the same within the floor
  and the saved latents are 0.9995 alike.
- **Her head sits a few pixels further from the original's in p1, p2
  and p4**, most on source frames 114 to 133. It is where her head sits,
  not how it is turned: the turn and the tilt of the eye line match the
  original's equally in every render, the gate included. The word "head"
  left the text between o1x3 and p1, which now says she follows the
  figure's movements; but four sentences changed between those two, not
  that phrase alone, and the gate (the phrase, no marks) and o1x3 (the
  phrase and the marks) are already a pixel apart. So a text with the
  phrase back can be expected to win back about half at most. One seed.
- **Saying what the shapes are for changes nothing that can be measured
  either** (p4 against p1): the same lids, the same mouth, the same head
  a few pixels off, the saved latents 0.9995 alike. On this shot and seed
  the three plain wordings make one picture, by all three readers. They
  are one seed and nearly one text, so their agreeing is one result seen
  three times: that the eyes clause was not needed, and that her head
  sits a few pixels off without the word "head", each rest on that one
  result until a second shot or a second seed repeats it.

## Which other shots can take it (the pose model on the original only; nothing rendered)

The owner asked for other examples before more is read into this one. The
same pose pass on the original at the canvas, each person in their tracked
box (medians):

| shot | face, glabella to chin, canvas px | gap between the lash lines, canvas px | eyes along the shot |
|---|---|---|---|
| the kitchen shot, 14 to 154 (second subject) | 106 | 2.0 | open, then lowered |
| the lead, 168 to 185 | 24 | 0.7 | |
| the lead, 294 to 414 | 46 | 0.9 | |
| the lead, 438 to 483 | 56 | 1.2 | |
| the lead, 516 to 603 | 15 | 0.4 | |
| the lead's close-up, 1575 to 1729 | 175 | 2.5, to 10.6 where her eyes are wide | wide, shut, part open again |
| the third subject, 1466 to 1535 | 126 | | steady; his head is turned away on the last frames |

On the lead's street shots the gap between her lids is about a pixel:
there is nothing to draw marks from and nothing to read a render by, so
the form cannot carry her eyes there at whole-frame size. The pose
model's score for the lid points stays high on those frames, and on a
frame where a head is turned away, so the score does not say this; the
pixel gap does. A render zoomed in on what is replaced would give the
pixels, and a zoomed load takes no motion video (`video_mask.py`, the
note at `RENDER_ZOOM_REPLACED`). The close-up is the shot that tests
following in time. Put to the owner as one card.

## What it does not say

Anything on a second seed or a second shot. Whether the reference is
followed in time. Which of the last three reads best: that is the owner's
on playback. Likeness to the still, her hair, the light on her and the
sound were not read by anyone. On a face much smaller than this one the
marks would be smaller than a cell even enlarged three times.

A point of light the owner saw on playback is on the side of her nose on
every frame of every render of this shot, the gate and the 10th's
included; it is not from the marks or from any text here, and the owner
closed it.

## Where things are

Untracked: the job files, the texts, the script that draws the marks and
the two marked videos, `data/2026-10-11_fun_gate_inputs/` (`gate_t_0014`,
`gate_o_0014`, `gate_p_0014`, `gate_p4_0014`); the captures with the second
reader's looks, `data/2026-10-11_gate_t_0014/`, `data/2026-10-11_gate_o_0014/`,
`data/2026-10-11_gate_p_0014/`, `data/2026-10-11_gate_p4_0014/`; one table a
clip and one a source shot, `data/2026-10-11_sapiens2_pose_consecutive/`; the blind reader's
figures, `data/2026-10-11_mrtie_gate_o_read/`. On the share: the renders
under `Video/mrfire/gate_<job>/`, and each stacked with its masks, the
marked video and frame numbers under `Video/mrfire/review_2026-10-11/`.
