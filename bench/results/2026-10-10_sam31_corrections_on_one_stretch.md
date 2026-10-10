# The Subject Track with and without the pack's SAM 3.1 corrections, on one stretch of one clip (2026-10-10)

lane: masked video to video
verdict: one clip, one stretch of seven shots, one no-sampling preview each way. On it the corrected path made one wrong call right, made none worse, and left the masks the same where both paths had the subject. A default rests on this; it is a first read and says so.

**Read this first.** Seven shots is not a sample. Every similarity went up
on the corrected path, the one wrong candidate's as well as the right
takes', so the gap between the lowest right take and the wrong candidate
did not widen: the shot that changed its call crossed the match line
because everything moved up, and a right absence now sits nearer the line
than it did. That margin is the thing to watch on the next stretch, and a
clip whose two leads look alike is where it would bite.

**What was asked.** No shipped or session graph wired
`MiniMaxH3SAM31Corrections` (`sam31_corrections.py`), so every track,
similarity and "absent under the line" of the day's gating came from
ComfyUI's SAM 3.1 port uncorrected. The same preview was made twice on one
stretch, once with the corrections node between the loader and both
Subject Tracks, and the two captures compared shot by shot.

## How

Two no-sampling previews of the same frames, picks and settings, differing
only in the corrections node; each turned into a capture folder by
`bench/capture_masked_run.py files` (untracked `data/`), and the lead's
shot tables, per-frame mask rows and the rows across subjects read from
there. "What is there by eye" is from a sheet of eleven frames with both
tracks outlined. The numbers are in the json beside this file
([`2026-10-10_sam31_corrections_on_one_stretch.json`](2026-10-10_sam31_corrections_on_one_stretch.json)):
one row a shot with the call, the similarity, the detections and the mask's
median area on each path; the agreement of the two paths' masks; the
separation figures; and the second tracker.

## What it shows

- **The lead's tracker.** Six of seven calls right uncorrected, seven of
  seven corrected. The one that changed is a shot the lead is alone in,
  called absent at the line uncorrected and taken just above it corrected.
  The json's `separation` block is the honest reading of that.
- **The masks.** Where both paths have the lead, the two masks agree almost
  to the pixel (`lead_mask`); area per shot, the step from the frame before
  and the pieces do not differ. The corrected path has a mask on one more
  shot, the one above.
- **The second tracker is not fixed by it.** Meant for the second subject,
  it takes the lead on four shots on both paths and has nothing on the shot
  where the second subject is small at the back. It needs typed
  corrections either way; `bench/capture_masked_run.py`'s
  `two_tracks_on_one_person` rule was written from this stretch and names
  those frames on both captures.

## What it does not show

Whether the corrected path is better on a clip with more people, a crowd,
or two subjects who look alike; whether a render differs (nothing was
sampled); whether the kitchen stretch gated earlier the same day would
track differently (it was not redone: the masks agree where both exist
here, which is the reason given for leaving it).
