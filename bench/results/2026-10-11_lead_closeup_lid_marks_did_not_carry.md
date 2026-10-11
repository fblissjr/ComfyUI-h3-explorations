# The lid marks on a second shot: on the lead's singing close-up her eyes did not follow

2026-10-11, session mrfire; second reader mrcolor, who wrote its reading
before seeing the first reader's. One shot (the lead, source frames 1575
to 1729 of `fun_0000_1919_24p.mov`), one render, one seed. Nothing here
was made twice, so there is no floor for this load. The owner had not
seen the render when this was written.

## What it buys

[`2026-10-11_kitchen_shot_lid_marks_on_the_body_model.md`](2026-10-11_kitchen_shot_lid_marks_on_the_body_model.md)
found that marks of the original's eye openings, drawn on the body-model
video, bring a redrawn face's lids to the original's level on one shot,
and could not say whether they are followed in time: nobody's eyes open
and close in that shot. The owner asked for other examples before more
was read into it. This is the first: a shot where the original's eyes go
wide, shut and open again while she sings. It says whether the form
repeats, and it applies to this shot. Nothing is accepted by it: it is a
null, and what follows from it is one more render.

## The load

Through `bench/masked_job.py`, nothing posted by hand. The lead, face
only. Masks (hers, and the two other subjects' as keep-outs), sound (the
vocals stem), seed, margin and window are those of the render of these
frames made on 2026-10-10 with the sentence that she sings, which the
owner had praised; the mask files decode equal to that day's on every
frame, and the planned region equals that render's saved region on every
cell.

Three things differ from that render, together: a motion video where it
had none (a body model of her from a new preview); the marks on it; and
four additions to its text in the wording the owner picked on the kitchen
shot (what the video is, and that she follows its movements and its
facial expressions). That render was posted by hand and this one was
built, so against it this is a different make and not a single change.

The marks are drawn by the same rule as the kitchen's, each eye's opening
grown about its own centre by one factor for the shot, here 1.82 (the
kitchen's working marks were at 3): the rule keeps a frontal face's marks
one size on the canvas whatever the face's size, and her face here is
about one and two thirds the kitchen face's height.

Known before it sampled, and written down then: the body model's mouth is
closed on every frame while the text says she sings with her mouth
opening wide; the marks on the last dozen frames are drawn part open
where no opening can be seen in the blurred original; and the text's one
sentence about a telephone handset is true of about the last three fifths
of the window.

## The measure

The Sapiens2 pose model on every frame, each clip at the canvas in her
tracked box: the original, this render, and the two renders of
2026-10-10 (with the singing sentence, and without it). The eye opening
is the distance between the lash lines over the distance from glabella
to chin. It is trusted at its two ends only, under about 0.020 shut and
over about 0.045 open, and the mouth figure is not trusted at all:
`2026-10-11_sapiens2_pose_first_frames.md`, the evening's additions. So
her mouth was read by eye.

The original's eyes: wide on about frames 1593 to 1604, shut from 1608 to
1699, a short look up on 1700 to 1703.

| | eyes, 1593 to 1604 | eyes, 1608 to 1650 | eyes, 1651 to 1699 | eyes, 1700 to 1703 | frames of 1608 to 1699 under 0.020 | nose point from the original's, canvas px, mean |
|---|---|---|---|---|---|---|
| the original | 0.056 | 0.014 | 0.011 | 0.027 | all | |
| the 10th's render, singing sentence | 0.046 | 0.029 | 0.038 | 0.040 | 15% | 16.5 |
| the 10th's render, no singing sentence | 0.055 | 0.037 | 0.039 | 0.050 | 21% | 23.9 |
| this render, with the marks | 0.044 | 0.030 | 0.037 | 0.035 | 14% | 16.9 |

From the files: on her face this render differs from the 10th's singing
render by about seventeen grey levels of 255 and the saved video latents
are 0.994 alike, about as far apart as the 10th's two renders are from
each other; off her face the pictures are alike.

By eye, stills only, two sessions on their own crops: she sings, her
mouth open where the 10th's singing render has it open. Her eyes are open
through the stretch where the original's are screwed shut, in this render
as in the 10th's.

## What it says

- **Her eyes did not follow the marks.** On every stretch this render's
  eyes read as the 10th's singing render's do. Where the original's are
  shut hers are open. The form that lowered a face's lids on the kitchen
  shot did nothing that can be measured here.
- **The reference's closed mouth did not shut hers.** A text that says
  she follows the figure's facial expressions, beside a figure whose
  mouth never opens, still gave a mouth that sings where the sentence and
  the sound say so.
- **The reference was not ignored altogether**: the face is a different
  make of the same face. What it changed is not her lids.

## What differs from the kitchen load, none of it separated

- The marks are smaller against her face: grown 1.82 times on a face
  about 175 px tall, where the kitchen's were grown 3 times on a face
  about 106 px tall.
- A shut eye's mark is a sliver. The white the marks add to a frame of
  the reference, at the median: about 220 to 340 pixels on the shut
  stretch here (under 100 on 17 frames), against about 620 on the
  kitchen shot's lowered stretch. On the wide stretch here it is about
  3800, where the 10th's renders are already near wide, so that stretch
  cannot show the marks working or not. A session writing the marks as a
  tracked tool found the same day that a mark thinner than about its own
  black edge leaves no white at all.
- The redrawn part is her face only at a margin of 16 px; the kitchen
  load redrew face and hair at 64, and there the body model was also what
  carried the head.
- The text carries a sentence that she sings; the kitchen's has no
  sentence about her face's action.
- Another person, another still, a longer window.

## What it does not say

Why. Whether larger marks would carry on this shot. Whether a face-only
region takes an expression from a motion video at all: the one face-only
load of the kitchen shot had no marks. Anything about following in time,
which is still untested on every shot. What the owner sees on playback.
A third reader had the files and had not answered when this was written.

## What is next, put to the owner

One render: this load with the marks at three times and nothing else
changed. Built and held, not rendered: the same load with a shorter
wording the owner asked to try, and the third subject's shot; neither
would say why this one failed.

## Where things are

Untracked: the job, the text, the reference video and its sheet, the
script that drew it, `data/2026-10-11_fun_set_inputs/` (`set_close_1575`,
`set_lead_1575`, `lead_sings_1575_v1.txt`, `lead_mesh_with_lids_1575.mp4`);
the capture with the second reader's two looks,
`data/2026-10-11_set_close_1575/`; the four tables,
`data/2026-10-11_sapiens2_pose_consecutive/` (`leadcanvas_1575`,
`closeup_lids_1575`, `closeup_d10sings_1575`, `closeup_d10plain_1575`). On
the share: the render under `Video/mrfire/set_close_1575/`, and stacked
with its masks, the reference and frame numbers under
`Video/mrfire/review_2026-10-11/`.
