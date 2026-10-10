# How much of the tracked subject a part mask covers: the figure, on the clip that showed it was missing (2026-10-06)

lane: masked
verdict: on one arm of the second test clip the part mask covered under a quarter of its own median share of the tracked subject on 40 of 360 frames, all of them in the first 51, while the part node reported the part "found on every frame"; the new per-frame figure names those frames from the two masks alone, and on 37 of them most of the part lies off the subject's mask

**What was asked.** `MiniMaxH3SubjectParts` (`sapiens2_parts.py`) calls a
frame found when one pixel of a taken class lies on the subject, and its
report counted frames. On the arm below the render kept the original through
the opening of the window and nothing in any report said why. The fix is a
count (`part_coverage.py`): per frame, the share of the subject's own mask the
parts cover, and the share of the parts that lies off it. This record is the
evidence for its two constants.

**How to read it.** One arm, one clip, one choice of parts (hair, face and
neck, upper clothing, hands). The two masks are the ones the arm's own graph
saved lossless: the Masked Source's mask output, which on a parts graph is the
part mask before the margin, and the tracker's mask. Nothing was looked at for
this record: every figure is a count of pixels in those two files, made by
`part_coverage.coverage` and `summarise`. The same counts were made
independently by the session that ran the arm, and the two agree frame for
frame on all three (subject pixels, part pixels, part pixels on the subject).
Every number here is in
[`2026-10-06_part_coverage.json`](../../data/results/2026-10-06_part_coverage.json), with the
per-frame values.

## What ran

- **Clip and arm:** `vma.mp4`, one 345-frame window with 360 frames loaded,
  arm `vma36_upper_s1`, 1344 wide. The masks are 1344x760.
- **Files:** `usedmask_vma36_upper_s1_00001.mkv` and
  `trackmask_vma36_upper_s1_00001.mkv`, in the server's output folder. Not in
  the repo.
- The subject is on all 360 frames.

## The figures

| | |
|---|--:|
| median share of the subject the parts cover | 90% |
| lowest | 0.2%, frame 4 |
| frames with no part on the subject at all | 0 |
| frames under a quarter of the median | 40: 0-26, 28, 32-35, 43-50 |
| frames where more than half of the part lies off the subject | 37: 0-25, 32-33, 35, 43-50 |

So "found on every frame" was true and useless: no frame is empty, and forty
are a sliver.

## The two constants

**`LOW_OF_MEDIAN`.** How many frames are marked as the share moves:

| under this share of the median | frames | which |
|---|--:|---|
| 0.1 | 36 | 0-26, 32-33, 44-50 |
| 0.25 | 40 | 0-26, 28, 32-35, 43-50 |
| 0.5 | 57 | 0-29, 32-37, 40, 43-53, 55-56, 61-63, 276-278, 283 |
| 0.75 | 68 | 0-29, 31-38, 40-56, 58-64, 68-69, 276-278, 283 |

Between 0.1 and 0.25 the set barely moves, and it is the opening stretch. At
0.5 it starts to take single frames late in the window. The constant is 0.25.
It is judged against the clip's own median because a choice of parts sets
what share is normal: the head alone covers little of a whole person, and a
fixed share would mark every frame of such a clip.

**`OUTSIDE_MOST`.** The share of the part that lies off the subject's own
mask is between 0.47 and 0.98 on frames 0 to 26 (median 0.82), and has a
median of 0.12 and a largest value of 0.23 on frames 80 to 344. Half
separates the two with room on both sides on this clip.

## What this does and does not show

- One clip, one arm, one choice of parts. The low rule is not measured on a
  choice of the head alone, which is the case it was shaped for.
- A clip where the part is wrong on most frames has a low median, and the low
  rule then marks little. The other two rules do not depend on the median,
  and the part node also prints the share of the subject its model labelled
  as a person at all. That last figure is not in this record: the label maps
  of this arm were not saved.
- The measured mask is the Masked Source's, not the part node's `parts`
  output. On a parts graph the first is made from the second.
- Nothing here changes a mask. It is a report.
