# The part model shown the subject alone: two windows, with and without (2026-10-06)

lane: masked
verdict: with everything further than 8 pixels from the tracked subject's mask replaced by flat colour before the part model sees it, the parts cover a median of 96% of the subject over the window and never under 70%, from the first frame; without it the same node covers under 3% through the first 27 frames; on a second clip's window, where the parts were already on the subject, coverage is within about two points frame by frame and the part reaches further into the margin around the subject; counted, not looked at, on one window of each of two clips

**What was asked.** On this window the part model's labels lay off the
tracked subject through the opening, so the part mask covered a sliver of
them and the render kept the original there
(`2026-10-06_part_coverage.md`). The part model is trained on one person in a
frame. The proposed fix (`sapiens2_parts.alone`): before the crop is taken,
replace the picture further than `ALONE_MARGIN` from the tracker's mask by
the colour that is zero to the model, so there is nobody else to label. This
record is the test of it on the real model, run before the code was applied.

**How to read it.** One no-sampling run by the session that owns the card,
with the fix stood in for by core nodes in front of the unchanged part node:
the tracker's mask grown by 8 pixels (`GrowMask`, tapered corners), the
frames composited through it over a flat image. The part node ran twice in
that one graph, on the frames as they are and on the greyed ones, with the
same tracker's mask as its subject, and both `parts` outputs were saved
lossless with the tracker's mask. Every figure is a count of pixels in those
three files, made by `part_coverage.coverage`; the "plain" half reproduces
the figures of `2026-10-06_part_coverage.md`, the same forty frames named.
**Nothing was looked at for this record.** Per-frame values:
[`2026-10-06_part_model_shown_subject_alone.json`](../../data/results/2026-10-06_part_model_shown_subject_alone.json).

## What ran

- **Clip and window:** `vma.mp4` from 36.0 s, 360 frames loaded, 1344x760.
- **Parts taken:** hair, face and neck, upper clothing, hands.
- **Files**, in the server's output folder, not in the repo:
  `trackmask_vma36_parts_grey_00001.mkv`,
  `partsplain_vma36_parts_grey_00001.mkv`,
  `partsgrey_vma36_parts_grey_00001.mkv`.
- **Two differences from the code as written:** the fill is the nearest
  8-bit colour to the model's mean (124, 116, 104), under one step a channel
  from it; and the grown edge is `GrowMask`'s tapered one where the code
  grows square, a pixel or two at corners.
- **No matting model was loaded.**

## The share of the subject the parts cover

Median over each stretch of the window, with the lowest frame in it:

| frames | as it is | lowest | shown alone | lowest |
|---|--:|--:|--:|--:|
| 0-26 | 1.3% | 0.1% | 96.8% | 95.3% |
| 27-50 | 17.5% | 2.7% | 95.0% | 93.2% |
| 51-79 | 67.3% | 26.4% | 95.7% | 94.5% |
| 80-344 | 93.5% | 24.1% | 96.1% | 74.5% |
| 345-359 | 76.4% | 69.6% | 78.3% | 70.1% |

| over the window | as it is | shown alone |
|---|--:|--:|
| median | 90% | 96% |
| lowest | 0.2%, frame 4 | 70%, frame 348 |
| frames in doubt (`part_coverage.summarise`) | 40: 0-26, 28, 32-35, 43-50 | none |

Frame by frame, shown alone is higher by more than five points on 115 frames
and lower by more than five on none; its largest loss is 4.3 points, on
frame 244.

Where shown alone is lowest it is as low with the picture as it is: its
lowest frame between 66 and 344 is frame 294, at 74.5% against 73.7%, and
on every one of the 69 frames of that stretch where it is under 90%
(213-228, 231-236, 244, 251-267, 273-275, 290-308, 334-336, 340-343) the
picture as it is gives under 90% too. So what is low there is how much of
the subject the taken classes are on those frames, not the change.

## The share of the part that lies off the subject's own mask

| frames | as it is | shown alone |
|---|--:|--:|
| 0-26, median | 82.0% | 14.5% |
| 80-344, median | 11.9% | 16.5% |

On the stretch that was wrong it falls. Where the part was already on the
subject it rises a little: with a flat surround the part reaches further
into the ring between the tracker's mask and the node's `subject_margin`,
which is where the node already lets a label count.

## A second window, where the picture as it is already worked

The same pass on `thrill_2160.mkv` from 67.0 s (360 frames loaded, 1024x768,
the same parts taken; the subject's mask is empty on frames 345 to 359), run
to see that the change takes nothing away. Files, same folder convention:
`grey_trackmask_thrill67_parts_grey_00002.mkv`,
`grey_partsplain_thrill67_parts_grey_00002.mkv`,
`grey_partsgrey_thrill67_parts_grey_00001.mkv`. The session that ran it
counted the same three series and reports the same figures.

| frames | covered, as it is | shown alone | off the subject, as it is | shown alone |
|---|--:|--:|--:|--:|
| 0-59 | 61.1% | 62.2% | 7.0% | 19.4% |
| 60-119 | 58.5% | 59.4% | 7.0% | 16.8% |
| 120-179 | 61.4% | 61.8% | 6.5% | 16.4% |
| 180-239 | 56.6% | 57.6% | 6.9% | 18.8% |
| 240-299 | 99.3% | 99.4% | 5.8% | 6.5% |
| 300-344 | 59.1% | 59.8% | 6.2% | 15.5% |

Medians per stretch. Over the window the median coverage is 60% as it is and
61% shown alone, and neither has a frame in doubt. Frame by frame, shown
alone minus as it is has a median of 0.8 points; it is lower by more than
five points on no frame, its largest loss is 2.1 points (frame 71) and its
largest gain 5.2 (frame 17).

**What does move here is the part's reach past the subject's own mask**: the
share of the part lying off the tracker's mask goes from about 7% to between
15% and 19% on five of the six stretches. Those pixels can only be in the
margin the node already allows a label to count in. So on a clip where the
plain picture worked, the mask is the same on the subject and a little
fatter at the subject's edge. Whether that band is the subject's own edge,
labelled where it used to be called background, or flat colour taking a
label, the counts cannot say. The session leading the lane read the plain
and the greyed class picture for this window and reported, as seen: on every
tile of each the classes are the subject's own and in the same places, and
nothing is labelled on the flat surround; the greyed edges reach a little
further into the ring. That reading is theirs. The band is inside
`subject_margin` either way, and it is why `MASK_VERSION` moves: a mask kept
from before this change is not the mask the node makes now.

## What this does and does not show

- One window of each of two clips, one margin, one run each. No other margin was tried.
- **The counts do not say the labels are the right classes.** A model shown
  one person on flat ground may label nearly everything inside the mask as a
  body part, including what the tracker's mask takes in of anyone in front
  of the subject; coverage near the whole mask is what that would look like
  too. The node's class pictures are the place to read that. The session
  leading the lane read the greyed class picture for frames 0 to 63 (eight
  tiles) beside the plain one and reported, as seen: on every greyed tile
  the labels are on the subject and each class is where that part of the
  body is, and on the plain picture the first three tiles carry no label on
  the subject. That reading is theirs; it covers those eight frames.
- Not tested: the matte with a flat surround (no matting model was loaded;
  the masked graphs wire `parts`, not `matte`), a subject who holds
  something the tracker's mask leaves out, and a render.
- The test stood the fix in with core nodes. The code itself has run on the
  check's stand-in models only.
