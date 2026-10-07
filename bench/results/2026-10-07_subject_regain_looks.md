# Today's Subject Track at each look after a loss: who is among the candidates, and what a rule by place would take (2026-10-07)

lane: masked
verdict: following the most central person on a six-second crowd stretch today's node never loses the mask and never looks again; following the largest, ComfyUI's tracker moves the mask to another figure with no empty frame between, the node's regain then takes the figure the track was on, and its later takes land in three places with every line passed; read off the data, a candidate over the likeness line that overlaps the picked figure's box is at that figure's place and no other candidate overlaps it, and such a candidate is among those returned on one look of eight when sixteen detections are asked and on three to five when thirty-two or sixty-four are; a rule by place was not run in a loop and its number rests on one stretch

**What was asked.** Earlier the same day
([`2026-10-07_subject_track_under_nudge.md`](2026-10-07_subject_track_under_nudge.md))
most looks the node made after a loss were refused on a crowd stretch, and
every look returned exactly the sixteen detections asked for. Two things
could be wrong at a refused look: the subject is not among the people
returned, or likeness cannot tell them from the next person. The owner
asked whether a region of the frame, or a higher count, would do.

**How.** [`bench/subject_regain_looks.py`](../subject_regain_looks.py)
runs today's node's own pass (`subject_track.py`: `follow`, at the
settings of `h3_config.SUBJECT_TRACK` with the `pick` named on the command
line) and at every look it made after a loss asks again six ways: sixteen,
thirty-two and sixty-four people and heads on the whole frame, sixty-four
people with sixteen heads, sixteen on a crop around the subject's last
place, sixteen on the middle half of the frame. "Sixteen on the frame"
must reproduce the node's own best and next at that look; it does at every
look of both runs. It runs as fed and again with every value moved one
level of 255. SAM 3.1 as ComfyUI ships it, on the card, outside the
server, one run per arm. The data:
[`2026-10-07_subject_regain_looks_largest.json`](2026-10-07_subject_regain_looks_largest.json),
[`2026-10-07_subject_regain_looks_most_central.json`](2026-10-07_subject_regain_looks_most_central.json);
every table here is printed from them by the tool's `render`, which also
prints the six ways at every look.

`lotsofpeopledance_0414_0720.mkv` from 138.0 s, 144 frames at 24.0 a second, 1344x760; SAM 3.1 as ComfyUI ships it, on cuda:0; pick `largest`; the regain line 0.88, the lead required 0.03; the crop is 3.0 times the subject's last box, the centre region the middle 0.5 of the frame in each direction. One run per arm.

## The track the node hands back, for each pick

| run | the pick's box on the pick frame (left, top, right, bottom) | frames with a mask | plausible | looks after a loss | taken again at | box centre every twelve frames | each mask against the frame before's: least, median, steps under a half |
|---|---|---|---|---|---|---|---|
| most central, as fed | [0.458, 0.446, 0.519, 0.679] | 144 of 144 | 144 | 0 | [] | {'0': [0.488, 0.547], '12': [0.487, 0.537], '24': [0.491, 0.532], '36': [0.487, 0.574], '48': [0.485, 0.532], '60': [0.487, 0.553], '72': [0.485, 0.635], '84': [0.482, 0.546], '96': [0.481, 0.548], '108': [0.488, 0.525], '120': [0.485, 0.555], '132': [0.461, 0.558]} | 0.585, 0.885, 0 |
| most central, nudged | [0.458, 0.446, 0.519, 0.676] | 144 of 144 | 144 | 0 | [] | {'0': [0.488, 0.547], '12': [0.487, 0.537], '24': [0.491, 0.532], '36': [0.487, 0.575], '48': [0.486, 0.53], '60': [0.487, 0.553], '72': [0.485, 0.635], '84': [0.482, 0.545], '96': [0.481, 0.548], '108': [0.488, 0.525], '120': [0.485, 0.555], '132': [0.461, 0.558]} | 0.581, 0.886, 0 |
| largest, as fed | [0.306, 0.633, 0.417, 0.913] | 124 of 144 | 122 | 8 | [23, 83, 92, 141] | {'0': [0.365, 0.756], '12': [0.362, 0.76], '24': [0.885, 0.761], '36': [0.893, 0.827], '48': [0.899, 0.77], '60': [0.908, 0.766], '72': [0.457, 0.809], '84': [0.458, 0.806], '96': [0.34, 0.796], '108': None, '120': None, '132': [0.469, 0.798]} |  |
| largest, nudged | [0.306, 0.633, 0.425, 0.913] | 123 of 144 | 121 | 8 | [23, 92, 141] | {'0': [0.367, 0.755], '12': [0.303, 0.76], '24': [0.885, 0.761], '36': [0.893, 0.827], '48': [0.899, 0.77], '60': [0.908, 0.766], '72': [0.341, 0.791], '84': [0.34, 0.799], '96': [0.339, 0.796], '108': None, '120': None, '132': [0.469, 0.797]} |  |

**The most central person is held**: a mask on every frame, in one place,
every step sharing well over a half with the frame before, and the node
never looks again. The stretch is not hard for a subject who is whole and
in the middle. This is the person the owner means on this clip; the
shipped default `pick` became `most central` the same day (0.220.0).

**The largest person is not**, and the box centres show how. The pick
stands at about 0.36 across. Twelve frames later the track is still
there. By frame 24 its box centre is at 0.885: the far end of the row.
The mask on the last frame before the first empty one, written by the
original track and by no take, has its box from 0.813 to 0.943 across (the
json's `the_place_before_the_first_loss`). So ComfyUI's tracker, following
one figure alone, moved the mask onto another figure half the frame away
with no empty frame between. The node then looks, on the next frame, and
takes the detection that overlaps where the track last was: the figure it
was already on. Its gallery is made of frames of the track
(`subject_track.py::gallery_frames`), so it now holds two people, and its
later takes land at 0.455, 0.336 and 0.461 across: three places, each with
a likeness over the line and a lead over the margin. The first take has
the highest likeness and the widest lead of the stretch and is the
farthest from the pick, because the gallery had already been taught that
figure. A contact strip of the frames, outlined, was looked at by two
sessions and agrees by position; a clip is not described here by what it
shows, and the owner's eye is the last word.

## One jump or a creep: the largest person's track, frame by frame

The same pass run again with `--track-only`, saving every frame of the
track the node hands back
([`2026-10-07_subject_regain_looks_largest_track.json`](2026-10-07_subject_regain_looks_largest_track.json),
`the_track_every_frame`). Each frame's mask against the frame before's, by
intersection over union:

| run | steps where both frames have a mask | least | median | steps under a half: frame (overlap) |
|---|---|---|---|---|
| as fed | 122 | 0.0 | 0.885 | 20 (0.0), 71 (0.0), 92 (0.0), 104 (0.0) |
| nudged | 121 | 0.0 | 0.884 | 20 (0.001), 68 (0.0), 104 (0.0) |

| run | frame | the mask's box (left, top, right, bottom) | overlap with the frame before's mask |
|---|---|---|---|
| as fed | 16 | [0.303, 0.587, 0.42, 0.925] | 0.906 |
| as fed | 17 | [0.301, 0.583, 0.419, 0.925] | 0.904 |
| as fed | 18 | [0.301, 0.579, 0.423, 0.926] | 0.906 |
| as fed | 19 | [0.301, 0.57, 0.923, 0.929] | 0.886 |
| as fed | 20 | [0.821, 0.651, 0.937, 0.937] | 0.0 |
| as fed | 21 | [0.383, 0.629, 0.946, 0.938] | 0.825 |
| as fed | 22 | [0.813, 0.612, 0.943, 0.939] | 0.839 |
| as fed | 23 | [0.816, 0.588, 0.944, 0.941] | 0.841 |
| nudged | 16 | [0.303, 0.587, 0.419, 0.925] | 0.903 |
| nudged | 17 | [0.301, 0.583, 0.423, 0.926] | 0.899 |
| nudged | 18 | [0.301, 0.579, 0.425, 0.928] | 0.903 |
| nudged | 19 | [0.301, 0.57, 0.923, 0.929] | 0.882 |
| nudged | 20 | [0.821, 0.647, 0.938, 0.937] | 0.001 |
| nudged | 21 | [0.816, 0.629, 0.946, 0.938] | 0.837 |
| nudged | 22 | [0.813, 0.612, 0.943, 0.939] | 0.838 |
| nudged | 23 | [0.816, 0.588, 0.943, 0.941] | 0.832 |

**One jump, with one frame of warning, the same in both runs.** Up to
frame 18 the box is the picked figure's and every step shares most of the
mask. On frame 19 the box runs from the picked figure to the far end of
the row: the mask is on two figures at once, and the step still passes,
because the far piece is small. On frame 20 the mask is on the far figure
alone and shares nothing with frame 19's. A step test fires there by a
wide margin: the most central person's track on the same stretch never
steps under 0.58 (the table above). The frame before the jump, already on
two figures, passes a step test; a test on the box's growth would see it.
The later steps under a half are where a take's fill meets what was
there, or further jumps; they are not separated here.

### By place, read off the data

For each look and each count asked on the whole frame: the candidates over the likeness line, each as `likeness (overlap of its box with the picked subject's box on the pick frame)`. `at the place` is the best of those overlapping by 0.3 or more. This anchor is fair only where the subject stays where they stood.

| run | look, frame | asked | today's rule takes | candidates over the line | at the place |
|---|---|---|---|---|---|
| as fed | 23 | 16, the frame | somebody (0.0) | 0.9632 (0.0) | nobody |
| as fed | 23 | 32, the frame | somebody (0.0) | 0.9632 (0.0) | nobody |
| as fed | 23 | 64, the frame | somebody (0.0) | 0.9632 (0.0) | nobody |
| as fed | 71 | 16, the frame | nobody | none | nobody |
| as fed | 71 | 32, the frame | nobody | 0.899 (0.592), 0.8948 (0.107) | 0.899 |
| as fed | 71 | 64, the frame | nobody | 0.899 (0.592), 0.8948 (0.107) | 0.899 |
| as fed | 83 | 16, the frame | somebody (0.092) | 0.9244 (0.092) | nobody |
| as fed | 83 | 32, the frame | somebody (0.092) | 0.9244 (0.092) | nobody |
| as fed | 83 | 64, the frame | somebody (0.092) | 0.9244 (0.092) | nobody |
| as fed | 92 | 16, the frame | somebody (0.456) | 0.9227 (0.456), 0.8923 (0.054) | 0.9227 |
| as fed | 92 | 32, the frame | somebody (0.456) | 0.9227 (0.456), 0.8923 (0.054) | 0.9227 |
| as fed | 92 | 64, the frame | somebody (0.456) | 0.9227 (0.456), 0.8923 (0.054) | 0.9227 |
| as fed | 105 | 16, the frame | nobody | 0.8974 (0.0), 0.8885 (0.0), 0.8885 (0.0) | nobody |
| as fed | 105 | 32, the frame | nobody | 0.8982 (0.477), 0.8974 (0.0), 0.8885 (0.0), 0.8885 (0.0) | 0.8982 |
| as fed | 105 | 64, the frame | nobody | 0.8982 (0.477), 0.8974 (0.0), 0.8885 (0.0), 0.8885 (0.0) | 0.8982 |
| as fed | 117 | 16, the frame | nobody | none | nobody |
| as fed | 117 | 32, the frame | nobody | 0.8846 (0.426) | 0.8846 |
| as fed | 117 | 64, the frame | nobody | 0.8846 (0.426) | 0.8846 |
| as fed | 129 | 16, the frame | nobody | 0.8967 (0.0), 0.8875 (0.029), 0.8849 (0.0) | nobody |
| as fed | 129 | 32, the frame | nobody | 0.9049 (0.0), 0.8967 (0.0), 0.8875 (0.029), 0.8849 (0.0) | nobody |
| as fed | 129 | 64, the frame | nobody | 0.9123 (0.445), 0.9049 (0.0), 0.8967 (0.0), 0.8875 (0.029), 0.8849 (0.0) | 0.9123 |
| as fed | 141 | 16, the frame | somebody (0.035) | 0.9308 (0.035), 0.8975 (0.0), 0.887 (0.0) | nobody |
| as fed | 141 | 32, the frame | somebody (0.035) | 0.9308 (0.035), 0.8975 (0.0), 0.887 (0.0) | nobody |
| as fed | 141 | 64, the frame | somebody (0.035) | 0.9308 (0.035), 0.8975 (0.0), 0.887 (0.0) | nobody |
| nudged | 23 | 16, the frame | somebody (0.0) | 0.9641 (0.0) | nobody |
| nudged | 23 | 32, the frame | somebody (0.0) | 0.9641 (0.0) | nobody |
| nudged | 23 | 64, the frame | somebody (0.0) | 0.9641 (0.0) | nobody |
| nudged | 68 | 16, the frame | nobody | 0.8898 (0.0) | nobody |
| nudged | 68 | 32, the frame | nobody | 0.8898 (0.0) | nobody |
| nudged | 68 | 64, the frame | nobody | 0.8958 (0.0), 0.8898 (0.0) | nobody |
| nudged | 80 | 16, the frame | nobody | none | nobody |
| nudged | 80 | 32, the frame | somebody (0.426) | 0.9202 (0.426) | 0.9202 |
| nudged | 80 | 64, the frame | somebody (0.426) | 0.9202 (0.426) | 0.9202 |
| nudged | 92 | 16, the frame | somebody (0.392) | 0.9197 (0.392) | 0.9197 |
| nudged | 92 | 32, the frame | somebody (0.392) | 0.9197 (0.392) | 0.9197 |
| nudged | 92 | 64, the frame | nobody | 0.9197 (0.392), 0.8926 (0.0), 0.8881 (0.082) | 0.9197 |
| nudged | 105 | 16, the frame | nobody | 0.8927 (0.0), 0.8901 (0.0), 0.8849 (0.0) | nobody |
| nudged | 105 | 32, the frame | nobody | 0.8927 (0.0), 0.8901 (0.0), 0.8849 (0.0), 0.8832 (0.425) | 0.8832 |
| nudged | 105 | 64, the frame | nobody | 0.8927 (0.0), 0.8901 (0.0), 0.8849 (0.0), 0.8832 (0.425) | 0.8832 |
| nudged | 117 | 16, the frame | nobody | none | nobody |
| nudged | 117 | 32, the frame | nobody | none | nobody |
| nudged | 117 | 64, the frame | nobody | none | nobody |
| nudged | 129 | 16, the frame | nobody | 0.9023 (0.0), 0.8878 (0.062), 0.8855 (0.0) | nobody |
| nudged | 129 | 32, the frame | nobody | 0.9058 (0.0), 0.9023 (0.0), 0.8878 (0.062), 0.8855 (0.0) | nobody |
| nudged | 129 | 64, the frame | nobody | 0.9124 (0.428), 0.9058 (0.0), 0.9023 (0.0), 0.8878 (0.062), 0.8855 (0.0) | 0.9124 |
| nudged | 141 | 16, the frame | somebody (0.066) | 0.9315 (0.066), 0.8968 (0.0), 0.8851 (0.0) | nobody |
| nudged | 141 | 32, the frame | somebody (0.066) | 0.9315 (0.066), 0.8968 (0.0), 0.8851 (0.0) | nobody |
| nudged | 141 | 64, the frame | nobody | 0.9315 (0.066), 0.9076 (0.0), 0.8968 (0.0), 0.8851 (0.0) | nobody |

| run | asked | looks | looks with a candidate over the line at the place | today's rule takes somebody | of them at the place |
|---|---|---|---|---|---|
| as fed | 16, the frame | 8 | 1 | 4 | 1 |
| as fed | 32, the frame | 8 | 4 | 4 | 1 |
| as fed | 64, the frame | 8 | 5 | 4 | 1 |
| nudged | 16, the frame | 8 | 1 | 3 | 1 |
| nudged | 32, the frame | 8 | 3 | 4 | 2 |
| nudged | 64, the frame | 8 | 4 | 2 | 1 |

**What the by-place tables show** (read off the data; no rule was run in a
loop). The candidates over the likeness line fall in two groups with
nothing between, in both runs: at the picked figure's place, and
elsewhere. Today's rule takes somebody at three or four looks of eight and
is at the place at one or two of them. A candidate at the place is among
those returned at ONE look of eight when sixteen are asked for, and at
three to five when thirty-two or sixty-four are. Under today's rule a
higher count gains one take and, at sixty-four, loses two (more people
over the line, a smaller lead); judged by place it is what puts the figure
among the candidates at all.

## What this does not show

- **A rule by place working.** It was read off recorded candidates against
  one anchor, the pick's own box, which is fair only because this figure
  stays where they stand for the six seconds. The rule's shape is
  supported; its number (`AT_THE_PLACE` in the tool, for the table only)
  is not: four labelled takes, one stretch. A tracker's anchor would be
  the last frame before its track broke, and that needs a test on every
  frame that the mask has not moved off, which today's node does not have.
- **Anything about the crop around the last place or the centre region.**
  For the largest person the node's last place is the far figure's, so the
  crop looks in the wrong place, and the middle half of the frame excludes
  the front row; for the most central person there is no loss to repair.
  Both columns are in the data and say nothing about the idea. A first run
  of this tool read the last place from the finished track, which a take
  fills backward as well as forward (`subject_track.py`, the regain's
  `piece[a:b] = track(a, b, f, ...)`), so it knew the future; that run's
  data is not kept.
- **Identity.** Position on a stretch where the figure does not move, and
  a strip looked at. No labels.
- A stretch where a whole, in-frame subject is lost. None was found on
  this footage (see
  [`2026-10-07_subject_alone_or_in_a_group.md`](2026-10-07_subject_alone_or_in_a_group.md)).

## Reproducing it

```
<python> bench/subject_regain_looks.py run --clip lotsofpeopledance_0414_0720.mkv --second 138 --seconds 6 --width 1344 --rate 24 --pick largest --json L
<python> bench/subject_regain_looks.py run --clip lotsofpeopledance_0414_0720.mkv --second 138 --seconds 6 --width 1344 --rate 24 --pick "most central" --json C
<python> bench/subject_regain_looks.py run --clip lotsofpeopledance_0414_0720.mkv --second 138 --seconds 6 --width 1344 --rate 24 --pick largest --track-only --json T
<python> bench/subject_regain_looks.py render --json L
```

## Files

- `2026-10-07_subject_regain_looks_largest.json`,
  `2026-10-07_subject_regain_looks_most_central.json`: one run each, as fed
  and nudged, every look with its six ways and every candidate over the
  line.
- `2026-10-07_subject_regain_looks_largest_track.json`: the largest
  person's track on every frame, as fed and nudged.
