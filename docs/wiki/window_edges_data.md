# Data for choosing a window's edges

Last updated: 2026-10-13.

**A direction with nothing built.** Written 2026-10-13 by the morning's lead at the owner's ask, after a day
on which every load began where the subject's face first showed and ended where it stopped showing, and the
sound was read for one thing only: whether a load's text may say the subject sings (the voiced runs,
`voice_per_frame.csv` and `voice_spans.json` beside the clip's stems). The owner's word that day: no rule to add;
capture the data and have it in front of whoever sets a window's first and last frame. What follows is the
list, general to any clip and any subject. Each item is figures a frame, made once a video into its analysis
folder, printed by a load's details for that load's edges, and drawn on the plan strip's header; none is a
threshold, none refuses anything ([`../../AGENTS.md`](../../AGENTS.md), "Green or red only for a hard
failure"; "A constant carries its provenance"). Where a tool now exists for one of them, the line that says so
goes here with its date; until then every item is a want.

Why the edges matter: the model is seeded by a window's first frame and holds its last frame with the region
open past the load's end ([`masked_v2v.md`](masked_v2v.md), "A latent step is several frames"), and a sung
phrase, a blink or a head turn that an edge cuts through is carried by nothing on the other side.

## Of the sound

1. **Phrase onsets and ends inside each voiced run.** Read from the vocals stem's energy envelope: the short
   gaps the span joiner closed, and an onset strength (librosa's, if it is in the environment; else a frame-rate
   RMS over a short window). A load's details print the frames from each edge to the phrase start and end before
   and after it. Cannot see: a held note from a phrase break; whose voice it is.
2. **A beat grid with downbeats.** Read from the rest stem (the drums are in it), a beat tracker with the tempo
   printed beside its grid; the clip's cuts printed once against the grid, so it is known whether the picture's
   cuts sit on beats at all. A load's details print the nearest beat and downbeat before and after each edge.
   Cannot see: rubato and tempo changes (the grid is a fit).
3. **Whose mouth opens while the voice is on**, a frame and a subject. Read from the pose passes on the original
   (`bench/pose_along_shot.py`: the lip gap and the lip score of each subject in each shot) against the voiced
   frames; per phrase, the subject whose mouth-open frames cover it. This is what the owner's cue ("when he
   opens his mouth, that is your cue") is read by eye today. Cannot see: a shout or a laugh from a sung line; a
   mouth in profile where the lip score is low; a small face at the back where the points are unreliable.
4. **The vocal contour** per frame: the vocals stem's RMS and a pitch track. Where a phrase is loud and sustained
   the mouth stays wide, and a load whose text says the subject sings is read against it. Cannot see: reverb
   tails and doubled vocals.
5. **The sound's offset to the picture**, measured once a copy, from the first audio packet's time against the
   first frame (a session noted on 2026-10-10 that the source's track starts a fraction of a frame before the
   picture; nobody measured it). Cannot see: nothing; it is one constant a copy.

## Of the picture, a frame and a subject, from the track, the part and the points

6. **Face readability**: the face part's size in pixels, the head's yaw (the nose against the ears from the
   points), the gap between the lash lines, and a sharpness figure in the face box (the Laplacian's variance on
   the source fitted as the loader fits it). The first frame seeds the model and the last is held, so when there
   is a choice an edge goes on a frame where the face is large, sharp and not in profile. Cannot see: soft focus
   from motion blur; a face the points misplace.
7. **Occlusion runs**: frames where the part shrinks under a hand or an arm (the part's size against the
   track's, which the capture already prints as flags) and where a keep-out overlaps the part, printed as runs
   so that an edge is not set inside one. Cannot see: an occluder nobody tracks.
8. **Events of the original** to carry and not to cut through: blinks (the lid gap's minima), mouth-open onsets,
   head-turn onsets (the yaw's velocity), and the subject's entry and exit frames (the track's first and last
   mask). A load's details list the events inside its frames and the nearest outside each edge. Cannot see:
   anything on a face under the drawing line (`h3_config.POSE_READING["lid_gap_px_to_draw"]`) or the back of a
   head.
9. **Camera motion** per frame: the frame difference outside every tracked mask (a pan or a whip), so that an
   edge is not set mid-whip and a reader knows a smear is the camera's. Cannot see: a subject moving against a
   still camera (that is the subjects' own figure, item 8).

## What is built

**2026-10-13, the same day: items 1, 2 and 5, from the two stems and the copies alone.**
[`bench/clip_signals.py`](../../bench/clip_signals.py) with
[`bench/check_clip_signals.py`](../../bench/check_clip_signals.py), on made sound with each break seen red. No
library was added: scipy, numpy and PyAV were in the environment; librosa and every ready-made beat tracker
were not, so the onset curve and the tracker are the tool's own short code, and what that loses is said in its
record. The files live in the video's audio folder, beside the voice table they are read with: one file a
signal, a row a source frame, and a record beside each with its inputs by path and hash, its settings, what it
cannot see, and the table's hash as a fingerprint a capture can name.

- **Item 1, phrases** (`phrases`): the vocals stem's level at several steps a frame; inside each voiced run
  every stretch under the voice table's own line is listed as a gap, and a gap of `BREAK` steps or more is a
  break. This is the level alone, not an onset strength: a break sung through with no drop is not found, and
  a run sung through is one phrase. The levels are in the table for a reader who would cut otherwise.
- **Item 2, a beat grid** (`beats`): the tempo, its three strongest candidates, how much stronger the onset
  curve is on the beats than off them, and the tempo in each stretch, so a tempo that moves shows as a spread.
  NO DOWNBEAT: a bar's first beat needs a trained model that is not here; the record gives the low band's
  level on each of the four ways of counting in fours, as figures. With `--shots`, the record's cuts against
  the grid, once, beside what chance would give.
- **Item 5, the sound's offset** (`offset`): the first sound packet against the first picture, as the
  container has it, a figure a copy. It cannot see priming samples or an edit list.
- **A reader's print** (`at`): for a load's first and last frame, the run and the phrase each falls in with
  the frames to their ends, the beats either side, how many frames earlier the load would begin on the
  phrase's start or the beat before and whether that is inside the same shot, and how far the phrase runs
  past the load's end. Printed, never stored, and it says nothing about where an edge should be.

**The first real run is a throwaway**, read end to end by a session that did not write the tool before
anything of it is called made (the lane's rule for a new tool); the figures a real clip gave are in that
run's records and its reader's words, not here.

**Not built:** the vocal contour's pitch track (item 4; the level is in the phrases table); every picture
item (3, 6, 7, 8, 9); the figures in a load's own details and on the plan strip, which change the chain's
files and wait for a day with no load between its plan and its render; the capture's manifest naming the
signals' fingerprints.

## One remark, not a rule

The held tail is where a phrase that runs past a load's end goes wrong. Items 1 and 8 printed for the frames
just after the load's end tell the planner whether the buffer belongs after the load rather than before it.
