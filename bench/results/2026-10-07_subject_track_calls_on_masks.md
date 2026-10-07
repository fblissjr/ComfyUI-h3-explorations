# Each call the Subject Track makes to the tracker, judged on its own masks (2026-10-07)

lane: masked
verdict: on masks, not on recorded steps: following the largest person on a hard crowd stretch the pick's own tracker call is cut where the mask moves to another figure and a gallery stops one frame before, in both runs, and following the most central person on three windows every call is one unbroken run with nothing trimmed; on a fifth window, where the tracked mask shrinks to about a ninth of its area, the track is one continuous mask that a fresh detect reproduces at every look, so a refresh from the detector would change nothing there; on that window's pick frame the pick rules name different detections

**What was asked.** `subject_tracks.unbroken` and `gallery_span` were
accepted on recorded step values
([`2026-10-07_subject_regain_looks.md`](2026-10-07_subject_regain_looks.md),
"One jump or a creep"); the masks themselves had not been through them.
And, for a window of one shot in which the subject's mask becomes much
smaller, whether the track jumps or creeps, how its size changes, and
what a fresh detect returns beside it: the measurement a refresh of the
track at intervals was to be judged on.

**How.** [`bench/subject_regain_looks.py`](../subject_regain_looks.py)
`run --track-only`, at the commit that added the per-call recording.
The node's own `track` callable is wrapped, so every call the node makes
to ComfyUI's tracker (the pick's, and each one after a loss) has its own
frames put through `unbroken` and `gallery_span` as they come back,
before a later call writes over part of them. One tracked call each,
never the stitched piece, at frame size. Today's Subject Track at
`h3_config.SUBJECT_TRACK` with the pick named; SAM 3.1 as ComfyUI ships
it, on the card, outside the server; each window as fed and again with
every value moved one level of 255; one run per arm. The line is
`subject_tracks.MOVED_OFF`.

- hard, pick `largest`: `lotsofpeopledance_0414_0720.mkv` from 138.0 s, 144 frames at 24.0 a second, 1344x760 ([`2026-10-07_subject_track_calls_on_masks_largest_hard.json`](2026-10-07_subject_track_calls_on_masks_largest_hard.json))
- hard, pick `most central`: `lotsofpeopledance_0414_0720.mkv` from 138.0 s, 144 frames at 24.0 a second, 1344x760 ([`2026-10-07_subject_track_calls_on_masks_most_central_hard.json`](2026-10-07_subject_track_calls_on_masks_most_central_hard.json))
- calm, pick `most central`: `lotsofpeopledance_0414_0720.mkv` from 16.0 s, 144 frames at 24.0 a second, 1344x760 ([`2026-10-07_subject_track_calls_on_masks_most_central_calm.json`](2026-10-07_subject_track_calls_on_masks_most_central_calm.json))
- dense, pick `most central`: `vma.mp4` from 121.93 s, 72 frames at 24.0 a second, 1344x760 ([`2026-10-07_subject_track_calls_on_masks_most_central_dense.json`](2026-10-07_subject_track_calls_on_masks_most_central_dense.json))
- from 4 s, pick `largest`: `lotsofpeopledance_0414_0720.mkv` from 4.0 s, 360 frames at 24.0 a second, 1344x760 ([`2026-10-07_subject_track_calls_on_masks_largest_from_4s.json`](2026-10-07_subject_track_calls_on_masks_largest_from_4s.json))

**Written before the run.** Met for the largest person on the hard
stretch if the pick's call stops after frame 19 on "the mask moved off"
in both runs and a gallery may use frames up to 18. Met for the most
central person if all six pick calls are one run to the track's end with
nothing trimmed. The window from 4 s had no acceptance: a first
measurement, reported as it came out.

## Every tracked call

| window | pick | run | call: frames tracked [first, one past the last) | seeded on | unbroken | it stops, before; after | a gallery may use | least step inside | its last frame: overlap, mask area ratio, box ratio, centre step | the frame that stops it: the same four |
|---|---|---|---|---|---|---|---|---|---|---|
| hard | largest | as fed | [0, 144] | 4 | [0, 20] | the end of the track; the mask moved off | [0, 19] | 0.815 | 19: 0.886, 0.964, 5.271, 0.679 | 20: 0.0, 0.759, 0.147, 0.378 |
| hard | largest | as fed | [23, 144] | 23 | [23, 71] | the end of the track; an empty frame | [23, 71] | 0.695 | 70: 0.882, 0.968, 1.01, 0.007 |  |
| hard | largest | as fed | [71, 144] | 83 | [71, 92] | the end of the track; an empty frame | [71, 92] | 0.868 | 91: 0.912, 1.028, 1.016, 0.001 |  |
| hard | largest | as fed | [92, 144] | 92 | [92, 104] | the end of the track; the mask moved off | [92, 103] | 0.678 | 103: 0.891, 0.943, 0.983, 0.01 | 104: 0.0, 0.54, 0.589, 1.454 |
| hard | largest | as fed | [105, 144] | 141 | [125, 144] | an empty frame; the end of the track | [125, 144] | 0.839 | 143: 0.894, 1.028, 1.045, 0.005 |  |
| hard | largest | nudged | [0, 144] | 4 | [0, 20] | the end of the track; the mask moved off | [0, 19] | 0.815 | 19: 0.882, 0.964, 5.157, 0.672 | 20: 0.001, 0.77, 0.15, 0.378 |
| hard | largest | nudged | [23, 144] | 23 | [23, 68] | the end of the track; an empty frame | [23, 68] | 0.698 | 67: 0.826, 0.882, 0.881, 0.061 |  |
| hard | largest | nudged | [68, 144] | 92 | [68, 104] | the end of the track; the mask moved off | [68, 103] | 0.676 | 103: 0.888, 0.941, 0.983, 0.01 | 104: 0.0, 0.556, 0.589, 1.454 |
| hard | largest | nudged | [105, 144] | 141 | [126, 144] | an empty frame; the end of the track | [126, 144] | 0.846 | 143: 0.896, 1.033, 1.038, 0.005 |  |
| hard | most central | as fed | [0, 144] | 4 | [0, 144] | the end of the track; the end of the track | [0, 144] | 0.585 | 143: 0.851, 1.002, 0.916, 0.014 |  |
| hard | most central | nudged | [0, 144] | 4 | [0, 144] | the end of the track; the end of the track | [0, 144] | 0.581 | 143: 0.839, 1.017, 0.942, 0.01 |  |
| calm | most central | as fed | [0, 144] | 4 | [0, 144] | the end of the track; the end of the track | [0, 144] | 0.704 | 143: 0.963, 0.998, 0.996, 0.003 |  |
| calm | most central | nudged | [0, 144] | 4 | [0, 144] | the end of the track; the end of the track | [0, 144] | 0.704 | 143: 0.956, 1.014, 0.999, 0.003 |  |
| dense | most central | as fed | [0, 72] | 4 | [0, 72] | the end of the track; the end of the track | [0, 72] | 0.745 | 71: 0.941, 0.999, 1.015, 0.0 |  |
| dense | most central | nudged | [0, 72] | 4 | [0, 72] | the end of the track; the end of the track | [0, 72] | 0.747 | 71: 0.942, 1.0, 1.015, 0.0 |  |
| from 4 s | largest | as fed | [0, 360] | 4 | [0, 219] | the end of the track; an empty frame | [0, 219] | 0.854 | 218: 0.974, 0.988, 0.975, 0.001 |  |
| from 4 s | largest | as fed | [219, 360] | 219 | [219, 360] | the end of the track; the end of the track | [219, 360] | 0.663 | 359: 0.901, 1.002, 0.994, 0.012 |  |
| from 4 s | largest | nudged | [0, 360] | 4 | [0, 212] | the end of the track; an empty frame | [0, 212] | 0.854 | 211: 0.974, 0.989, 0.992, 0.004 |  |
| from 4 s | largest | nudged | [212, 360] | 212 | [212, 360] | the end of the track; the end of the track | [212, 360] | 0.622 | 359: 0.891, 0.971, 0.98, 0.01 |  |

- **The jump is cut on masks, as it was on recorded steps** (measured,
  both runs): the pick's call for the largest person is unbroken up to
  frame 19, stops at frame 20 with nothing shared, and a gallery may use
  frames up to 18. On frame 19 the mask's own area hardly changes while
  its box is several times the frame before's: the box sees a mask lying
  in two places, the area does not.
- **The most central person is never cut** (measured, three windows,
  both runs).
- **A second jump, with no frame of warning** (measured, both runs): a
  later call on the hard stretch stops at frame 104 with nothing shared,
  and the frame before it is ordinary by every measure. `gallery_span`
  leaves that frame out too, which costs one good frame. Its trim was
  reasoned from the first jump alone; this is a case it does not
  describe.
- **The other low steps of the stitched piece are seams**, where one
  call's frames meet another's, and not jumps: each call here ends on an
  empty frame or on the track's end except the two above.

## The window from 4 s

| run | each pick rule's detection on the pick frame, and its box (left, top, right, bottom) | frames with a mask | taken again at |
|---|---|---|---|
| as fed | `largest`: detection 0, [0.013, 0.087, 0.703, 0.98]; `most central`: detection 2, [0.502, 0.062, 0.799, 0.946]; `best match for the phrase`: detection 0, [0.013, 0.087, 0.703, 0.98] | 360 of 360 | [219] |
| nudged | `largest`: detection 0, [0.013, 0.087, 0.703, 0.98]; `most central`: detection 2, [0.504, 0.062, 0.799, 0.945]; `best match for the phrase`: detection 0, [0.013, 0.087, 0.703, 0.98] | 360 of 360 | [212] |

The mask's share of the frame and its box (width by height, as shares of the frame), every twenty-four frames from the pick frame:

| run | 4 | 28 | 52 | 76 | 100 | 124 | 148 | 172 | 196 | 220 | 244 | 268 | 292 | 316 | 340 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| as fed, share | 0.273 | 0.251 | 0.262 | 0.262 | 0.233 | 0.202 | 0.167 | 0.162 | 0.159 | 0.125 | 0.084 | 0.056 | 0.033 | 0.031 | 0.029 |
| as fed, box | 0.69 by 0.89 | 0.65 by 0.87 | 0.67 by 0.89 | 0.64 by 0.73 | 0.63 by 0.72 | 0.50 by 0.70 | 0.43 by 0.67 | 0.38 by 0.68 | 0.35 by 0.70 | 0.29 by 0.64 | 0.23 by 0.63 | 0.18 by 0.62 | 0.22 by 0.37 | 0.18 by 0.40 | 0.15 by 0.42 |
| nudged, share | 0.273 | 0.252 | 0.262 | 0.262 | 0.233 | 0.202 | 0.167 | 0.162 | 0.159 | 0.125 | 0.084 | 0.056 | 0.033 | 0.031 | 0.029 |
| nudged, box | 0.69 by 0.89 | 0.65 by 0.87 | 0.67 by 0.89 | 0.64 by 0.73 | 0.63 by 0.72 | 0.50 by 0.70 | 0.43 by 0.67 | 0.38 by 0.68 | 0.35 by 0.70 | 0.29 by 0.64 | 0.23 by 0.63 | 0.18 by 0.62 | 0.22 by 0.37 | 0.18 by 0.40 | 0.15 by 0.42 |

Each frame's mask against the frame before's, in the finished piece, and the steps where the box's area changes by a fifth or more:

| run | least step, frames 1 to 239 | least step, frames 240 to 359 | steps under a half | box area against the frame before's, where it changes by a fifth or more: frame (ratio, overlap) |
|---|---|---|---|---|
| as fed | 0.854 | 0.663 | 0 | 177 (1.22, 0.982), 274 (1.24, 0.779), 275 (0.70, 0.708), 278 (0.69, 0.837) |
| nudged | 0.854 | 0.622 | 0 | 231 (1.68, 0.954), 233 (0.57, 0.974), 237 (1.76, 0.976), 238 (0.56, 0.971), 274 (1.39, 0.894), 275 (0.63, 0.622), 278 (0.69, 0.837) |

A fresh detect, sixteen asked, on every twelfth frame, against the finished track's mask there (mask against mask, intersection over union; a detection is "on the track" from 0.3):

| run | looks | looks with exactly one detection on the track | its overlap, least and greatest | the next detection's overlap, greatest | detections returned, least and greatest | first look that returns all sixteen |
|---|---|---|---|---|---|---|
| as fed | 30 | 30 | 0.966, 0.996 | 0.008 | 10, 16 | 180 |
| nudged | 30 | 30 | 0.965, 0.996 | 0.008 | 10, 16 | 180 |


- **One continuous mask** (measured, both runs): no step under a half
  in the finished piece. The tracker let go once; the node looked on that
  same frame and took a detection overlapping the frame before's mask
  almost wholly, so the piece has no seam there. A one-level change of
  the input moves the frame it lets go on by a few frames.
- **The mask shrinks to about a ninth of its area** over the window, and
  near frame 275 its box loses about a third of its area in each of two
  steps.
- **A fresh detect reproduces the track's mask at every look**: exactly
  one detection on the track, nearly the same mask, and nothing else
  overlapping it to speak of. So on this window a refresh of the track
  from the detector would replace the mask with one almost identical to
  it. Nothing here shows a track that drifts from what a fresh detect
  gives.
- **The pick rules disagree on the pick frame**: `largest` and
  `most central` name different detections, in both runs. This window
  was followed with `largest`. On the hard stretch above the two rules
  also name different detections. A default is a first guess; which
  detection is the subject is settled per shot.

## What this does not show

- **Identity.** Continuity, and agreement with a detector that answers
  "where is every person", not "who". No labels.
- **That a refresh never helps.** One window, on which the track and the
  detector agree. A track that drifts, or a tracker that lets go and is
  not taken back, has not been recorded for a whole, in-frame subject.
- **Anything after the subject's mask**: what a part model takes inside
  it, what a margin regenerates around it, or a render.
- The window from 4 s with `most central`; more than one nudge; another
  clip of the kind.
- The detector's count: from the middle of that window every look
  returns all sixteen detections asked for, so who else stands near the
  subject is not fully listed.

## Reproducing it

```
<python> bench/subject_regain_looks.py run --clip lotsofpeopledance_0414_0720.mkv --second 138 --seconds 6 --width 1344 --rate 24 --pick largest --track-only --json J
<python> bench/subject_regain_looks.py run --clip lotsofpeopledance_0414_0720.mkv --second 138 --seconds 6 --width 1344 --rate 24 --pick "most central" --track-only --json J
<python> bench/subject_regain_looks.py run --clip lotsofpeopledance_0414_0720.mkv --second 16 --seconds 6 --width 1344 --rate 24 --pick "most central" --track-only --json J
<python> bench/subject_regain_looks.py run --clip vma.mp4 --second 121.93 --seconds 3 --width 1344 --rate 24 --pick "most central" --track-only --json J
<python> bench/subject_regain_looks.py run --clip lotsofpeopledance_0414_0720.mkv --second 4 --seconds 15 --width 1344 --rate 24 --pick largest --track-only --json J
<python> bench/subject_regain_looks.py render --json J
```

## Files

- The five json named above, one per window and pick: each call's
  verdict (`each_tracked_call`), which detection each pick rule names
  (`each_pick_rule_on_the_pick_frame`), every frame's box, step and
  share of the frame (`the_track_every_frame`) and the fresh detect at
  each look (`a_fresh_detect_at_each_look`).
