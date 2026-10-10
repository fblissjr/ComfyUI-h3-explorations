# Kept frames against the body mesh: which one a whole-subject render follows through a head turn (2026-10-10)

lane: masked video to video
verdict: not judged. One stretch of one clip, one seed a render, read by a pose model and not watched. Every render that carried frames of the subject turned away before the turn missed it by the same wide margin, at either size of mesh and with either Sol sink; the one load with no such frames made the turn, lowered the head and brought a hand to the face as the source and the mesh do. On another, shorter shot a load with no kept frames did not follow its mesh, so "no kept frames" is not the whole of it.

**Read this first.** These are a pose model's readings (ComfyUI core's body
node), every second frame, of renders nobody has played. One stretch
carries the headline, with one seed for each way of making it. The second
shot is there because it cuts against the headline and is the same day's
work: its subject is a small share of the frame behind another person, and
its window is mostly a held last frame.

**What was asked.** A whole-subject pass over a long stretch came back with
the subject's head still turned away where the source turns to the camera,
lowers the head and raises a hand to the face. The body mesh given as the
motion video shows all three. The pass's second window is a continuation:
it keeps frames from the window before, in which she is turned away. Was
it the mesh (too small in the frame), the window's place in the load, or
the kept frames?

## How

`bench/measure_subject_yaw.py` on the card for the source, the mesh video
and every render of the stretch, the subject's box from her tracked mask;
then `bench/measure_subject_motion.py`. The numbers are in the json beside
this file
([`2026-10-10_kept_frames_against_the_mesh.json`](2026-10-10_kept_frames_against_the_mesh.json)):
one entry a render with what it is, the head's distance from the source's
on the turn and after it, the chin's angle and the nearer wrist's distance
from the nose where the source lowers the head and raises the hand, and
the motion measure; the same head figure from the fresh load's first
frame; the opening shot's renders; and the shorter shot.

## What it shows

- **The mesh is a faithful reference.** Read by the same predictor it is
  within a couple of degrees of the source's head throughout and follows
  the source's joints closely.
- **The load with no kept frames follows it**: the head through the turn
  and after, the chin, the hand at the face, and the motion measure about
  as high as the mesh's own.
- **Every render with kept frames of her turned away fails alike**: the
  continuation, the same with the mesh at twice the size, the same with
  the cheaper Sol sink, and two one-window patches laid on the pass's own
  render. Tens of degrees off on the turn and after it, the chin up, the
  hand nowhere near the face.
- **The mesh's size changed nothing** on this stretch: the two
  continuations that differ only in it read the same to a degree.
- **The cheaper Sol sink changed nothing** here, and on the opening shot it
  is not behind the dearer one on any joint group.
- **The mesh to the text encoder alone follows far less** than the mesh in
  the video model (the opening shot).
- **A load with no kept frames can still not follow**
  (`a_load_with_no_kept_frames_that_did_not_follow`): the mesh has the
  hand at the face and the head turned away; the render faces the camera
  with the hands down.

## What it does not show

Why the short load failed: her size in the frame and the share of its
window that is a held last frame both differ from the load that followed,
and nothing here separates them. Whether kept frames that were noised, or
fewer of them, or a facing sentence, would let a continuation make the
turn: those renders were queued when this was written. Whether the
predictor reads a rendered face as it reads the source's: the mesh's own
row is the only check on that, and it is a grey figure, not a render.
Anything about how a render plays.
