# Data for choosing a window's edges

**Figures a frame, of the sound and of the picture, in front of whoever sets a load's first and last frame; none is a
threshold and none refuses anything** (the owner, 2026-10-13: no rule to add, capture the data). The model is seeded
by a window's first frame and holds its last frame with the region open past the load's end
([`pieces.md`](pieces.md), "A latent step is several frames, and its grid is fixed for a whole load";
[`window_context.md`](window_context.md), the held tail), so a sung phrase, a blink or a head turn that an edge cuts
through is carried by nothing on the other side. Of the nine items below, items 1, 2 and 5 (and dips) are built
(`bench/clip_signals.py`), and are not yet called made: the run is made again and read again by a session that did
not write the tool before a load's details or the plan strip read any of it. Each item is made once a video into its
analysis folder, printed by a load's details for that load's edges, and drawn on the plan strip's header
([`../../AGENTS.md`](../../AGENTS.md), "Green or red only for a hard failure"; "A constant carries its provenance").

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

[`bench/clip_signals.py`](../../bench/clip_signals.py), held by [`bench/check_clip_signals.py`](../../bench/check_clip_signals.py)
on made sound with each break seen red. No library was added (scipy, numpy and PyAV were in the environment; librosa
and every ready-made beat tracker were not), so the onset curve and the tracker are the tool's own short code. One
file a signal, a row a source frame, in the video's audio folder beside the voice table, each with a record of its
inputs by path and hash, its settings, what it cannot see, and the table's hash as a fingerprint a capture can name.

| signal | what it holds | cannot see |
|---|---|---|
| `phrases` (item 1) | the vocals stem's level at several steps a frame, as the power of both channels together (never of their mean, which cancels what differs between them); the tool's own voiced runs, made by the voice table's rule (its line, its join, its least length, imported), with the frames on which the voice table's runs differ; inside each run every stretch under the line is a gap, and a gap of `BREAK` steps or more is a break | a break sung through with no drop; a run sung through is one phrase |
| `dips` | every stretch of the level under the sung level round it (a high percentile over a window either side, a starting value), down to a stated floor, with its frames, depth, length, share under the line, and the voice's onset strength where it ends | which dip is a breath: a reading of the table, not a setting |
| `beats` (item 2) | the tempo, its three strongest candidates, how much stronger the onset curve is on the beats than off, the tempo in each stretch; one clock, the window's middle, for beats and the onset column; where a candidate is near twice the tempo, the half-way places with their strength, never as beats; every beat kept with its strength and the frames since the one before, and where the steady run begins and ends (two settings, starting values from one reading); places cut to a hundredth and never rounded up; with `--shots`, the cuts against the grid beside what chance would give | NO DOWNBEAT and no count in fours: a bar's first beat needs a trained model that is not here |
| `offset` (item 5) | the first sound packet against the first picture, as the container has it; reads zero on every copy on the files' own timestamps and edit lists | priming samples or an edit list; a zero neither confirms nor refutes the note of 2026-10-10 that the sound starts a part of a frame early |
| `at` (a reader's print) | for a load's first and last frame: the run and phrase each falls in, the frames to their ends, the beats and half-way places either side, the dips, each counted from the load's first frame, and the dips within a second after its last frame, where the held tail is. Printed, never stored | where an edge should be |

**Not built**: the vocal contour's pitch track (item 4); every picture item (3, 6, 7, 8, 9); the figures in a load's
own details and on the plan strip, which change the chain's files and wait for a day with no load between its plan
and its render; the capture's manifest naming the signals' fingerprints. The voice table's own level is still of the
channels' mean; its record is a row in [`docs/state/next_steps.md`](../state/next_steps.md), and the table is not
made again until it is known which frames change sides and which texts that touches.

**The first run** was a throwaway, read by two sessions that did not write the tool (`reader_first_run/reading.md` in
the clip's audio folder, with the script behind every figure); every row above that changed since is held by a
made-sound control in `bench/check_clip_signals.py`. The tempo held against the reader's own grid. What the next
reading needs from the owner, once: the dips table on 1388-1537 against their ear at about 1467-1478, and two of the
breaks that did not hold on the first run.

## One remark, not a rule

The held tail is where a phrase that runs past a load's end goes wrong. Items 1 and 8 printed for the frames just
after the load's end tell the planner whether the buffer belongs after the load rather than before it.
