# The Subject Track after a loss inside a shot, and the subject handed from one run to the next (2026-10-06)

lane: masked
verdict: on one clip the track is seeded again 48 frames after the tracker lets go and a later load picks the same person from an earlier run's gallery where the pick rule takes somebody else; the line between the subject and another person is 0.88 with three hundredths to spare, and the subject is refused where the gallery shows them at a much larger size

**Result: both parts work on this clip and both rest on one thin number.**
With the search after a loss, a load whose track ended on frame 680 of 744
carries the mask on 737 frames. With an earlier run's gallery handed in, a
load on which the pick rule takes a person who is not the subject refuses
that person and picks the subject eight frames later, and the mask is on
every frame. A load with no loss gives the same mask as before, frame for
frame. The thin number: a person is taken at a plain similarity of 0.88 to
the gallery; on this clip the subject scores 0.89 to 0.97 when the gallery
is near their size and 0.83 to 0.87 when they are much smaller than it
shows them, and another person scores 0.86. Nothing was sampled. One clip.

**How to read it.** `subject_track.follow`, the node's own function, run in
a process of its own on core's SAM 3 nodes with the checkpoint the shipped
graphs load, on frames decoded with the loader's own ffmpeg arguments. It
was not run through the node on a server. A "score" is a person's best plain
similarity to a gallery of the subject, the lower of the two places a person
is compared (`gallery_scores`). Frame numbers called `span` count from
36.0 s of the clip at 24 fps; the others are a load's own. Every figure is
in [`2026-10-06_subject_track_regain_and_handover.json`](../../data/results/2026-10-06_subject_track_regain_and_handover.json),
one entry per run and per probed frame. What the frames show was looked at
on sheets for the day's notes and is not part of this record.

## What ran

- **Clip:** `vma.mp4`, loaded as the masked graphs' loader loads it (24 fps,
  width 1344, frames 1344x760), in loads of 744 frames from 36.0, 66.0, 96.0
  and 126.0 s. The cut finder finds no cut in any of them.
- **Tracker settings:** the node's defaults but the subject phrase, one that
  returns a single detection on most frames of this clip. The automatic pick
  lands on each load's frame 4.
- **Constants:** `REGAIN_SAME` 0.88, `REGAIN_MARGIN` 0.03, `GALLERY_MOST` 8,
  `REGAIN_MIN_RUN` and `PROBE_STRIDE` 12 (`subject_track.py`).
- **The tracker before this change**, for comparison: its masks of the same
  loads, saved lossless by the day's no-sampling passes on the server.

## The five runs on the committed code

| load | gallery handed in | pick | what happened | mask on | seconds |
|---|---|---|---|---|---|
| 36.0 s | no | frame 4, by the rule | let go on 681; probes on 681, 693 (one detection each, not comparable: no head), 705 (none), 717 (two, 0.74 and 0.70), 729 (one, 0.895): seeded again on 729 and tracked both ways | 737 of 744; empty on 681-683 and 690-693 | 72.5 |
| 66.0 s | no | frame 4, by the rule | no loss; nothing searched | 744 of 744 | 61.0 |
| 96.0 s | no | frame 4, by the rule | the track ends after 55 frames; 58 probes over frames 55-743 take nobody, the best 0.864 against this run's own gallery; the gap is reported | 55 of 744 | 166.3 |
| 96.0 s | from the 66.0 s run | frame 12, against the gallery, at 0.948 | frame 4's one detection refused at 0.856, frame 0 has none; seeded again once, on frame 565 at 0.974 | 744 of 744 | 75.4 |
| 126.0 s | from the 66.0 s run | frame 36, against the gallery, at 0.892, the next person there 0.844 | frames 4, 0, 12 and 24 refused at 0.830, 0.871, 0.844 and 0.843 | 744 of 744 | 67.5 |

Against the tracker before this change: on the 36.0 s load its mask is empty
on frames 681-743, 63 frames, of which this run fills 56. On the 66.0 s load
all 744 frames are identical. On the 96.0 s load without a gallery the first
55 frames are the same track and the rest empty in both. The probe decodes
the frames itself: on the 36.0 s load its mask differs from the saved one by
a median of 2 pixels a frame over the frames both carry and by 10,893 pixels
on the worst of them (frame 215), with the same last frame before the loss.

In the 96.0 s load the person the rule picks on frame 4 is not the subject:
that detection's mask shares no pixel with the subject's track on the same
frame (span 1444), measured in the earlier run named below. Alone, this
change stops that run carrying the wrong person past the end of their track
and cannot put the pick right: the run has no gallery but that person's.

## The scores the line was set from

Measured earlier the same day, before the line was moved from 0.93, with a
gallery of eight evenly spread frames.

| who | where | score |
|---|---|---|
| the subject, found again after the loss | span 725, 729, 741, against the gallery of span 0-680 | 0.907, 0.895, 0.890 |
| the subject on those frames, by a later load's own track | span 721-741, same gallery | 0.907 to 0.925 |
| the subject, gallery close in time and size | span 1436-1492, gallery of span 1200-1429 | 0.953 to 0.970 |
| a person who is not the subject, the phrase's one detection | span 1444, that gallery | 0.862 (top third 0.862, head 0.889); the subject on the same frame 0.960 |
| detections at the frame's edge after the loss | span 717 | 0.742 and 0.700; on 681, 693, 721, 733 not comparable, no head |

After the loss the phrase's one detection a frame was, on the eight frames
scored between span 681 and 741: a region at the frame's edge on four, two
edge regions on one, the subject on three. Around span 1440 it is the
subject on some frames, another person on one and nothing on one. A rule
that takes the only detection would have taken the wrong region on most of
those frames; the first draft of this change had such a rule and it was
removed on these numbers.

At 0.93 the subject was refused after the loss (best 0.895) and the run
reported the gap. The line was set to 0.88: above the other person's 0.862
and under the subject's 0.890.

## What it does not show

- **One clip, one subject, one phrase.** The line and the margin are set
  from it.
- **The score falls with the difference in size between a person and the
  gallery.** The subject's box is 268x477 px at span 0 and 69x117 at span
  680; against that shot's gallery he scores 0.89 to 0.91 at the small end.
  In the 126.0 s load, where the box is 44 to 92 px wide, the subject's
  first four probes scored 0.83 to 0.87 against the gallery of the 66.0 s
  load and were refused; the pick at 0.892 cleared the line by twelve
  thousandths. The gallery holds the frames of the smallest and the largest
  mask of its own track, which does not help a gallery from another stretch.
  Whether the four refused detections were the subject was not measured
  here; the day's passes on the server put the phrase's detection of frame 4
  of that load on a track that stays on for all 744 frames.
- **A first pick that is wrong stays wrong** without a gallery from a run
  before it.
- **A search that finds nobody costs** a probe every twelve frames to the
  shot's end: about a hundred seconds more on 689 frames here.
- **Not run:** the hand-over through the node on a server
  (`subject_from`); the first clip of the lane, with its cuts and
  corrections, beyond the checks' stand-ins; a render.

## Reproduce

The probe is a scratch script of the session and is not tracked; it calls
`subject_track.follow` with the callables of `subject_track._sam_callables`
and saves `shot_table.build`'s table. The checks that hold the behaviour
without a model: `bench/check_subject_track.py` items 10 and 11 and
`bench/check_shot_table.py` (`gaps_are_listed`,
`gallery_travels_in_the_table`).
