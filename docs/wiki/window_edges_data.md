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
  (the tool's own, made from that level by the voice table's rule)
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
- **Dips** (`dips`), added the same evening on the readers' findings: every stretch of the vocal level that
  lies under the sung level round it, down to a stated floor, listed with its frames, its depth, how much of it
  is under the voice table's line, and the voice's own onset strength where it ends. A list and no cut: which
  dip is a breath between two sung lines is a reading of the table.

**The first real run was a throwaway and was read twice** by sessions that did not write the tool. What they
found, what was built for each finding and the control that holds it are the last section of this page. **It is
still not called made:** the run is made again and read again before a load's details or the plan strip read
any of it.

**Not built:** the vocal contour's pitch track (item 4; the level is in the phrases table); every picture
item (3, 6, 7, 8, 9); the figures in a load's own details and on the plan strip, which change the chain's
files and wait for a day with no load between its plan and its render; the capture's manifest naming the
signals' fingerprints.

## One remark, not a rule

The held tail is where a phrase that runs past a load's end goes wrong. Items 1 and 8 printed for the frames
just after the load's end tell the planner whether the buffer belongs after the load rather than before it.

## The first run read, and the fixes (2026-10-13, the evening)

The throwaway first run of `bench/clip_signals.py` on this video's two stems was read by two readers who did
not write it: the morning's lead from the records and the printout, and the evening lead's blind reader against
the sound itself, with a beat grid of its own built from hits it picked (`reader_first_run/reading.md` in the
clip's audio folder, with the script behind every figure). What they found and what was built for each, each
held by a made-sound control in `bench/check_clip_signals.py` that was seen red first. Nothing here is called
made until the run is made again and read again by a session that did not write the tool, before any job reads
the tables.

1. **The vocal level was read from the two channels averaged as waveforms**, which cancels whatever differs
   between them, so the stem reads low exactly where it is widest; a good part of the run's phrase breaks did
   not hold when the channels were read properly. BUILT: the level is the power of both channels' samples
   together, never of their mean. The voice table's own level is still of the mean, and its record of that is
   a row in [`next_steps.md`](next_steps.md): the table is not made again until it is known which frames
   change sides and which texts that touches. A reading of the code then found the runs and what is quiet
   inside them on two scales (the runs read from the voice table, the gaps and dips on the power of both
   channels). BUILT: the tool makes its own voiced runs, from its own level, by the voice table's own rule
   (imported: its line, its join, its least length), writes them in its records, and lists the frames on which
   the voice table's runs differ; runs, gaps and dips are on one scale. CONTROL: two channels that cancel in
   their mean read at the tone's level and their phrase is found; a stereo tone the voice table's reading
   calls silence comes back as its own on-stretches, with the voice table's file as it was.
2. **A fall of the level that does not cross the voiced line was invisible**, so a breath inside a phrase a
   whole voiced run long was missed, and the print said the phrase ran far past a load's end where a dip lay
   close after it. BUILT, as data and not as a cut: the `dips` table. Every step's level against the sung
   level round it (a high percentile over a window either side, the blind reader's own choice, a starting
   value), and every dip listed down to a stated floor with its frames, its depth, its length, how much of it
   is under the line, and the vocals stem's own onset strength where it ends, since a line that begins after
   a breath shows as a spike where the level barely moved. The floor bounds the list and cuts nothing: the
   cut a reader makes is a reading of this table, not a setting in the tool. CONTROL: a fall of twenty
   decibels for five frames inside unbroken singing, never under the line, leaves the phrases one phrase with
   no gap and is listed on its frames at its depth; a shallower swell and a fall of one step are in the
   columns and not in the list; a gap of exactly the break's length breaks a phrase and one a step short does
   not.
3. **Two clocks in one table**: a beat's time had half the tracker's window added and the per-frame onset
   column did not, so a beat's own onset sat as often in the row before as in its own. BUILT: one clock, the
   middle of the window, for both. CONTROL: on a click track every beat's row is the row whose onset is the
   strongest round it, and is its click's own frame. The same reading of the code found the onset curve still
   made from the channels' mean; it is the rise of each channel's spectrum, summed. CONTROL: clicks whose
   channels cancel in their mean keep their tempo and a beat on every click.
4. **The tempo is right**: the reader's own grid gave the same period and nearly all the tool's beats sit on
   it; what read as a half tempo is a hit of another band half-way between every pair of beats, which the
   table did not hold. BUILT: where a tempo candidate lies near twice the tempo, the place half-way between
   each pair of beats is written beside the grid with the onset curve's strength there, as a place half-way
   and never as a beat; where one lies near half the tempo, the record says every other beat may be the pulse
   and gives the strength of both ways of taking them. The tempo stands. CONTROL: a click track with a faint
   click between each pair keeps the accents' tempo and has the faint clicks on its half-way places; with the
   click between at a third of the strength every click is a beat and every other one is named; clicks with
   nothing between them get no half-way place.
5. **Beats were laid before the music's pulse starts and after it ends**, some on single sounds and two on
   nothing; and the count in fours started from the first of them. BUILT, not as mrfit first put it (rows
   outside the pulse left empty): every beat the tracker lays stays in the table with its own strength against
   the middle beat's and the frames since the beat before, and the record says where the steady run begins
   and ends and how many beats lie outside it, by two settings marked as starting values from this one
   reading. A beat is not dropped by a setting. The count in fours is taken out. CONTROL: clicks that start
   late after two stray sounds leave the beats before them in the table, outside the run.
6. **Two rows carried a beat's place rounded up onto the next frame.** BUILT: a place is cut to a hundredth and
   never rounded up, so its whole part is its row; the `at` print reads the same value. CONTROL: on a click
   track every beat's written place is in its own row.
7. **The `at` print** has the phrases that start or end inside a load, the dips, the beats and the half-way
   places, each with its place counted from the load's first frame, and the dips within a second after its
   last frame, where the held tail is. The offset record no longer speaks of rows it does not have, and its
   `cannot_see` says a figure of zero neither confirms nor refutes the note of 2026-10-10 that the sound
   starts a part of a frame early. The offset reads zero on every copy on the files' own timestamps and edit
   lists. CONTROL: a hundredth of a second of offset is read as that; a copy with no sound gives no figure.

What the second reading then needs from the owner, once: the dips table on 1388-1537 against their ear at about
1467-1478, and two of the breaks that did not hold on the first run. If the table lists the breath they hear,
at the depth it shows, the phrase signal is data a job can print without anyone's ear inside the tool.
