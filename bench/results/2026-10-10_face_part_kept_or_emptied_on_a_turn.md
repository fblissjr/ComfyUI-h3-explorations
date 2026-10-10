# A face part kept or emptied where the head is turned away, on one spinning shot (2026-10-10)

lane: masked video to video
verdict: not judged. One shot, one seed, two renders that differ in the part mask on four frames, read from tables and from stills. Where the part was kept on the back of the head the render redrew the hair and drew no face; where it was emptied the frames are the source's. The emptied mask is the one to load; the harm it avoids on this shot is four frames of redrawn hair, not a face on the back of a head.

**Read this first.** Four frames of one shot. The prediction this record
tests was mine and was too strong: I had told the lead that a part left on
a turned-away head "drew a face shape on hair". The shape was the MASK's.
The render, given that mask on hair, drew hair. Whether a longer turn, a
larger head or another seed would draw a face there is not known.

**What was asked.** A face-only pass on a shot in which the subject spins
through a full turn. The part node holds the last face shape through the
frames where the face is gone; the capture's own fill does the same from
the other side. `bench/capture_masked_run.py::grade_holds` was written to
decide each such frame by the class map (the part is kept only where it
lies on the part's own classes). The first render of the shot was queued
from a mask I had cleaned by hand, which still carried the hold on four
frames; the second from the graded mask. That makes the first a control.

## How

Both renders are one load of the same frames, same still, text, seed and
settings; the masks differ on four frames only. The numbers are in the json
beside this file
([`2026-10-10_face_part_kept_or_emptied_on_a_turn.json`](2026-10-10_face_part_kept_or_emptied_on_a_turn.json)):
the grade's scores on the emptied frames; per frame around the turn, the
difference of each render from the source in the part's area and of the two
renders from each other; the floor far from any part; and the same over
every frame both masks have a part on.

## What it shows

- **The grade found what my hand missed.** On the four frames the kept
  part scores near nothing on the face class (the json's
  `grades_on_the_emptied_frames`); the head has turned and the hair is
  toward the camera. On the other frames the two masks are the same.
- **Kept, the render redraws the hair.** On those four frames the kept
  render is well off the source in the part's area and the emptied one is
  at the floor. As seen in stills the kept render's hair has strands that
  are not the source's, for four frames, and then the source's again: a
  flicker of texture, not a face.
- **Elsewhere the two renders are one render.** Over the frames both masks
  have a part on they are equally far from the source inside the part and
  close to each other.
- **The source's mouth is open with no voice on one run of the shot**
  (`mouth_open_with_no_voice`): the frames a face pass has no channel for,
  found by `mouth`'s report and not by eye. What each render's mouth does
  there is not in this record yet.

## What it does not show

Whether the redrawn hair is visible at speed. Whether a face is ever drawn
on a turned-away head (it was not here). Anything about a frame where the
class map itself is wrong: the grade cannot see those, and one such frame
on this shot was emptied by hand. Whether the render's mouth follows the
source's on the open run: that needs the part model's class map of each
render.
