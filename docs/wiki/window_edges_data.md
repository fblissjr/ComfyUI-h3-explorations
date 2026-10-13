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
  NO DOWNBEAT and no count in fours: a bar's first beat needs a trained model that is not here. With
  `--shots`, the record's cuts against the grid, once, beside what chance would give.
- **Item 5, the sound's offset** (`offset`): the first sound packet against the first picture, as the
  container has it, a figure a copy. It cannot see priming samples or an edit list.
- **A reader's print** (`at`): for a load's first and last frame, the run and the phrase each falls in with
  the frames to their ends, the beats either side, how many frames earlier the load would begin on the
  phrase's start or the beat before and whether that is inside the same shot, and how far the phrase runs
  past the load's end. Printed, never stored, and it says nothing about where an edge should be.

**The first real run was a throwaway and was read twice** by sessions that did not write the tool: the morning's
lead against this page's standard, and a blind reader who measured the same sound its own way first
(`reader_first_run/reading.md` in the clip's audio folder, with the script behind every figure). What held: the
tempo and the grid (the reader's own grid has the same period, and the tool's beats sit on its hits), and the
offset on the file's own timestamps and edit lists. What was the tool's fault, and is mended in the same day's
second commit, each with a case seen red:

- the vocal level was read from the two channels' MEAN, which reads low wherever they differ; it is now the
  power of both channels. The voice table's own level is still of the mean, and its record of that is a row;
- a beat's time and the per-frame onset column were half a window apart; they are on one clock;
- a place was rounded up onto the next frame; it is cut, never rounded up;
- the tracker lays beats before a pulse starts and after it stops; every beat now carries its own strength
  against the middle beat's and the frames since the beat before, and the record says where the steady run
  begins and ends, by two settings that are starting values from that one reading;
- the count in fours is gone: nothing supported it;
- what the first reading took for a half tempo was a hit of another band half-way between every pair of beats.
  Where the curve is also like itself at half the period, the place half-way is written beside the grid with
  the curve's strength there, as a place half-way and never as a beat; where it is as like itself two beats on,
  the record says every other beat may be the pulse and gives both ways' strength;
- `at` prints what lies inside a load as well: the phrases that start or end there, the beats, the half-way
  places, each with its place in the load.

**It is still not called made.** The run is made again as a second throwaway and read again before a load's
details or the plan strip read any of it.

**Not built, and written down so that it is not rediscovered:** a phrase break found by the fall of the level
against the sung level round it. The blind reader did this by hand (a dip a fixed depth under the level of the
singing near it, for half a frame or more) and found dips of several frames inside phrases the line passes over,
one of them inside a "phrase" a whole voiced run long. How deep and how long such a fall must be to be a break
between two sung lines is a setting for the owner's ear, on more than one track; the step levels it needs are in
the table.

**Not built:** the vocal contour's pitch track (item 4; the level is in the phrases table); every picture
item (3, 6, 7, 8, 9); the figures in a load's own details and on the plan strip, which change the chain's
files and wait for a day with no load between its plan and its render; the capture's manifest naming the
signals' fingerprints.

## One remark, not a rule

The held tail is where a phrase that runs past a load's end goes wrong. Items 1 and 8 printed for the frames
just after the load's end tell the planner whether the buffer belongs after the load rather than before it.

## The first run read, and the fixes (2026-10-13, the evening)

The throwaway first run of `bench/clip_signals.py` on this video's two stems was read by two readers who did
not write it (the morning's lead from the records and the printout; the evening lead's blind reader against
the sound itself, with a beat grid of its own built from hits it picked). What they found, and the fix for each,
in the order the fixes are made; every fix is held by a synthetic control in `bench/check_clip_signals.py`,
seen red first. Nothing here is called made until the run is made again and read again by a session that did
not write the tool, before any job reads the tables.

1. **The vocal level was read from the two channels averaged as waveforms**, which cancels whatever differs
   between them in phase, so the stem reads low exactly where it is widest; 10 of the run's 35 phrase breaks
   did not hold when the channels were read properly. FIX: the power of each channel on its own, the larger
   kept (or the powers summed), never the mean of the waveforms. CONTROL: a stereo tone with its channels out
   of phase reads as the tone's level, not as silence.
2. **A break the level does not cross the voiced line is invisible** (the tool finds gaps under the line only),
   so a breath inside the phrase 1388-1537 at about 1467-1478 was missed and the printout said the phrase ran
   96 frames past a load's end where the dip is about 25 after it. FIX, as data and not as a cut: a `dip`
   table beside the phrases, every frame's level against a local reference (the running maximum over about a
   second each side) and every dip listed with its frames, its depth in decibels and its length, down to a
   stated floor and with no threshold that acts; and one more column on the vocals stem, an onset strength
   (spectral flux), since a phrase that begins after a breath shows as a spike where the level barely moves.
   The relative-level cut a reader makes by ear is then a reading of this table, not a setting in the tool.
   CONTROL: a tone with a known six-decibel dip of ten frames above the line appears in the dips table at those
   frames and that depth; a gap of two frames under `BREAK` does not split a phrase and one of three does.
3. **Two clocks in one table** (the rows at the source's rate, a beat column at the copy's). FIX: one rate a
   table, the source's own, rows as source frames; the copies keep the frame index through the retime, so the
   same row serves them. CONTROL: a click track's beat column reads the same frames read at either rate.
4. **The tempo is right**: the reader's own grid gave the same period to a hundredth of a frame and 260 of the
   tool's 266 beats sit on it; what read as a double tempo is a low hit half-way between beats that the tool
   does not record. FIX: a column of the hits between the beats (the low band's onsets off the grid), not a
   doubled tempo; the candidates stay in the record. CONTROL: a click track with a softer click between each
   pair keeps its tempo and lists the between-hits.
5. **Beats placed before the music's pulse starts and after it ends.** FIX: the grid capped to the span where
   the onset curve rises above its own floor, the span said in the record, rows outside it empty. CONTROL: a
   click track with silence at both ends has no beat row in the silence.
6. **Two rows print a beat on the next frame** (a beat at 1419.08 printed under 1420). FIX: the floor of the
   fractional frame is the row, the fraction kept beside it, the same rule in the `at` print.
7. **The `at` print** gains the phrase and beat boundaries inside a load, between its first and last frame, not
   only at its edges; and the offset record's `cannot_see` says that the encoder's lead-in is unseen, so the
   2026-10-10 note that the sound starts a fraction of a frame early is neither confirmed nor refuted by it.
   The offset is confirmed at 0.0 on every copy on the files' own timestamps.

What the second reading then needs from the owner, once: the dips table on 1388-1537 against their ear on
1467-1478, and two of the ten breaks that did not hold. If the table lists the breath they hear at the depth it
shows, the phrase signal is data a job can print without anyone's ear inside the tool.
