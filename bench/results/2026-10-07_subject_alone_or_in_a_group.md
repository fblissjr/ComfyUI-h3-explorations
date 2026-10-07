# One subject in a crowd: followed alone, or seeded together with the people around them (2026-10-07)

lane: masked
verdict: a whole, in-frame subject (the most central person) is held without a break when followed alone on all three test windows, and company costs frames on the hard one; the largest person on the hard stretch, a front-row figure, is held for sixteen frames alone before the mask jumps to another figure and for twenty to twenty-eight in a group of sixteen before it moves off, so the group keeps a mask on, not the figure; with either of ComfyUI's two rules between followed objects switched off, or both, the central subject in a group is lost at the same frame, so those rules are not the cause there (one subject, one stretch, a group of sixteen; every arm applies the rule where ComfyUI applies it, so Meta's rule whole was not run); "held" is continuity of the mask and no label confirms identity

**Corrected 2026-10-07, the same evening; this replaces the record's first
form.** As first written this record's verdict was that a subject keeps a
plausible mask several times longer in a group than alone. Three things
were wrong with it, and each is kept visible below. (1) "The subject" was
the LARGEST person on the seed frame, the shipped `pick` at the time: a
front-row figure low in the frame on the crowd clip and a figure cut by
the left and bottom edges on the dense window, not the most central
person. (2) It counted plausible masks, and a plausible mask can be on
another figure: ComfyUI's tracker moves a mask to a neighbour with no
empty frame between
([`2026-10-07_subject_regain_looks.md`](2026-10-07_subject_regain_looks.md)).
(3) Its rule, written before the run, was on "trusted" frames, the watch
of `subject_tracks.py`, which gives false alarms on a figure that is held
and is in no acceptance now. The first form's tables are the last section.

**What was asked.** Tracking a subject in a crowded scene comes first in
this lane. Today's Subject Track seeds ComfyUI's tracker with its one
subject alone (`subject_track.py::_sam_callables`). Does seeding the
people around them as well hold the subject better?

**How.** [`bench/subject_alone_or_in_a_group.py`](../subject_alone_or_in_a_group.py).
ONE subject, chosen on the seed frame by the node's own rule
(`subject_track.choose`, named on the command line and written into the
data), and ONE set of seed masks for every arm, found once on the frames
as fed by SAM as it ships. The subject is followed from the seed frame
alone, with the three nearest people, and in a group of up to sixteen, by
ComfyUI's SAM 3.1 as it ships and through
[`sam31_corrections.py`](../../sam31_corrections.py); every arm again with
every input value moved one level of 255. On the card, outside the
server, one run per arm.

**What is counted.** `held`: frames from the seed frame up to the first
that is empty or whose mask shares under 0.2 with the frame before's
(intersection over union; the line is reasoned, not measured, and the
tables print each track's least and median step beside it so the line can
be judged). This is continuity of the mask. It is not identity: no frame
here is labelled, and a mask can stay continuous while taking in part of
a neighbour. `plausible` (the shape test) and `trusted` (the watch) are
printed and decide nothing: on a figure held on every frame the shape
test fails a frame or two and the watch doubts a sixth of the frames. For
the hard and calm windows of the most central person `held` was
recomputed, exactly, from the first empty frame and the steps under a
half that the run saved; the run itself stored a count that also broke at
one frame the shape test failed, and the json keeps that number under its
own name. Data:
[`2026-10-07_subject_alone_or_in_a_group.json`](2026-10-07_subject_alone_or_in_a_group.json);
every table is printed from it by the tool's `render`.

## The most central person

### The crowd clip, the hard stretch

`lotsofpeopledance_0414_0720.mkv` from 138.0 s, 144 frames at 24.0 a second, 1344x760; the subject is the most central person on frame 4, box [0.457, 0.446, 0.519, 0.679] (left, top, right, bottom as shares of the frame) (16 detections; one detect on the seed frame as fed, by SAM as it ships; the same masks seed every arm, the subject's first), followed from there; the detector is asked again every 12 frames. One run per arm.

Seed sets: {'alone': {'seeded': 1, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}, 'with the three nearest': {'seeded': 4, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}, 'in a group of up to sixteen': {'seeded': 16, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}}.

| SAM 3.1 | the subject is seeded | seeded | frames | with a mask | plausible | trusted | first frame lost | the same, nudged: with a mask, plausible, trusted | overlap with its nudged twin | against the alone arm where both have a mask: frames, mean overlap, frames under a half |
|---|---|---|---|---|---|---|---|---|---|---|
| as ComfyUI ships it | alone | 1 | 140 | 140 | 139 | 116 | None | 140, 140, 117 | 0.991 |  |
| as ComfyUI ships it | with the three nearest | 4 | 140 | 134 | 134 | 111 | 137 | 132, 131, 109 | 0.977 | [134, 0.968, 0] |
| as ComfyUI ships it | in a group of up to sixteen | 16 | 140 | 97 | 96 | 73 | 95 | 96, 95, 72 | 0.957 | [97, 0.944, 1] |
| corrected | alone | 1 | 140 | 140 | 140 | 117 | None | 140, 139, 116 | 0.992 |  |
| corrected | with the three nearest | 4 | 140 | 140 | 137 | 116 | None | 140, 138, 117 | 0.987 | [140, 0.965, 0] |
| corrected | in a group of up to sixteen | 16 | 140 | 129 | 122 | 99 | 115 | 124, 118, 95 | 0.945 | [129, 0.932, 3] |

Does the mask stay on one figure? `held` counts frames from the seed frame up to the first that is empty or shares under 0.2 of its mask with the frame before's.

| SAM 3.1 | the subject is seeded | held from the seed without a break | nudged | each mask against the frame before's: least, median | frames under a half (frame, overlap) | nudged: least, frames under a half |
|---|---|---|---|---|---|---|
| as ComfyUI ships it | alone | 140 | 140 | 0.62, 0.889 | [] | 0.614, [] |
| as ComfyUI ships it | with the three nearest | 133 | 132 | 0.388, 0.89 | [[136, 0.388]] | 0.638, [] |
| as ComfyUI ships it | in a group of up to sixteen | 91 | 91 | 0.561, 0.89 | [] | 0.562, [] |
| corrected | alone | 140 | 140 | 0.61, 0.885 | [] | 0.603, [] |
| corrected | with the three nearest | 140 | 140 | 0.522, 0.886 | [] | 0.528, [] |
| corrected | in a group of up to sixteen | 111 | 107 | 0.0, 0.887 | [[137, 0.441], [138, 0.0]] | 0.691, [] |

### The crowd clip, a calm stretch

`lotsofpeopledance_0414_0720.mkv` from 16.0 s, 144 frames at 24.0 a second, 1344x760; the subject is the most central person on frame 4, box [0.316, 0.247, 0.531, 0.62] (left, top, right, bottom as shares of the frame) (16 detections; one detect on the seed frame as fed, by SAM as it ships; the same masks seed every arm, the subject's first), followed from there; the detector is asked again every 12 frames. One run per arm.

Seed sets: {'alone': {'seeded': 1, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}, 'with the three nearest': {'seeded': 4, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}, 'in a group of up to sixteen': {'seeded': 16, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}}.

| SAM 3.1 | the subject is seeded | seeded | frames | with a mask | plausible | trusted | first frame lost | the same, nudged: with a mask, plausible, trusted | overlap with its nudged twin | against the alone arm where both have a mask: frames, mean overlap, frames under a half |
|---|---|---|---|---|---|---|---|---|---|---|
| as ComfyUI ships it | alone | 1 | 140 | 140 | 140 | 140 | None | 140, 140, 117 | 0.997 |  |
| as ComfyUI ships it | with the three nearest | 4 | 140 | 140 | 140 | 140 | None | 140, 140, 117 | 0.998 | [140, 0.989, 0] |
| as ComfyUI ships it | in a group of up to sixteen | 16 | 140 | 140 | 140 | 140 | None | 140, 140, 117 | 0.998 | [140, 0.978, 0] |
| corrected | alone | 1 | 140 | 140 | 140 | 140 | None | 140, 140, 140 | 0.995 |  |
| corrected | with the three nearest | 4 | 140 | 140 | 140 | 140 | None | 140, 140, 140 | 0.998 | [140, 0.989, 0] |
| corrected | in a group of up to sixteen | 16 | 140 | 140 | 140 | 140 | None | 140, 140, 140 | 0.997 | [140, 0.979, 0] |

Does the mask stay on one figure? `held` counts frames from the seed frame up to the first that is empty or shares under 0.2 of its mask with the frame before's.

| SAM 3.1 | the subject is seeded | held from the seed without a break | nudged | each mask against the frame before's: least, median | frames under a half (frame, overlap) | nudged: least, frames under a half |
|---|---|---|---|---|---|---|
| as ComfyUI ships it | alone | 140 | 140 | 0.731, 0.939 | [] | 0.731, [] |
| as ComfyUI ships it | with the three nearest | 140 | 140 | 0.723, 0.939 | [] | 0.725, [] |
| as ComfyUI ships it | in a group of up to sixteen | 140 | 140 | 0.734, 0.94 | [] | 0.733, [] |
| corrected | alone | 140 | 140 | 0.732, 0.937 | [] | 0.731, [] |
| corrected | with the three nearest | 140 | 140 | 0.722, 0.939 | [] | 0.722, [] |
| corrected | in a group of up to sixteen | 140 | 140 | 0.73, 0.94 | [] | 0.731, [] |

### The dense window

`vma.mp4` from 121.93 s, 72 frames at 24.0 a second, 1344x760; the subject is the most central person on frame 4, box [0.468, 0.297, 0.569, 0.576] (left, top, right, bottom as shares of the frame) (16 detections; one detect on the seed frame as fed, by SAM as it ships; the same masks seed every arm, the subject's first), followed from there; the detector is asked again every 12 frames. One run per arm.

Seed sets: {'alone': {'seeded': 1, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}, 'with the three nearest': {'seeded': 4, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.009}, 'in a group of up to sixteen': {'seeded': 16, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.009}}.

| SAM 3.1 | the subject is seeded | seeded | frames | with a mask | plausible | trusted | first frame lost | the same, nudged: with a mask, plausible, trusted | overlap with its nudged twin | against the alone arm where both have a mask: frames, mean overlap, frames under a half |
|---|---|---|---|---|---|---|---|---|---|---|
| as ComfyUI ships it | alone | 1 | 68 | 68 | 68 | 68 | None | 68, 68, 68 | 0.992 |  |
| as ComfyUI ships it | with the three nearest | 4 | 68 | 68 | 68 | 68 | None | 68, 68, 68 | 0.994 | [68, 0.96, 0] |
| as ComfyUI ships it | in a group of up to sixteen | 16 | 68 | 68 | 62 | 62 | None | 68, 59, 59 | 0.994 | [68, 0.952, 0] |
| corrected | alone | 1 | 68 | 68 | 68 | 49 | None | 68, 68, 33 | 0.993 |  |
| corrected | with the three nearest | 4 | 68 | 68 | 68 | 49 | None | 68, 68, 33 | 0.987 | [68, 0.959, 0] |
| corrected | in a group of up to sixteen | 16 | 68 | 68 | 58 | 47 | None | 68, 58, 25 | 0.99 | [68, 0.954, 0] |

Does the mask stay on one figure? `held` counts frames from the seed frame up to the first that is empty or shares under 0.2 of its mask with the frame before's.

| SAM 3.1 | the subject is seeded | held from the seed without a break | nudged | each mask against the frame before's: least, median | frames under a half (frame, overlap) | nudged: least, frames under a half |
|---|---|---|---|---|---|---|
| as ComfyUI ships it | alone | 68 | 68 | 0.741, 0.92 | [] | 0.745, [] |
| as ComfyUI ships it | with the three nearest | 68 | 68 | 0.636, 0.917 | [] | 0.64, [] |
| as ComfyUI ships it | in a group of up to sixteen | 68 | 68 | 0.745, 0.914 | [] | 0.742, [] |
| corrected | alone | 68 | 68 | 0.647, 0.915 | [] | 0.645, [] |
| corrected | with the three nearest | 68 | 68 | 0.633, 0.919 | [] | 0.64, [] |
| corrected | in a group of up to sixteen | 68 | 68 | 0.725, 0.914 | [] | 0.723, [] |

**Alone holds on all three windows**, as shipped and corrected, plain and
nudged: every frame, no step under a half. **Company costs frames on the
hard stretch** and more company costs more; the nudged twins agree within
a few frames. On the other two windows company makes no difference. The
two corrections help the subject in company and make no difference alone.
So "a subject followed alone is lost within a few frames on a dense
frame", from the morning's comparison with Meta's tracker
([`2026-10-07_sam3_core_against_meta.md`](2026-10-07_sam3_core_against_meta.md)),
is the edge-cut figure's result only.

## ComfyUI's two rules between followed objects, switched off

Why does company lose a subject who is held alone? The first suspect was
the pair of rules ComfyUI's tracker applies between followed objects,
first read, with file and line for both sides, in
[`docs/research/masking/2026-10-07_mryolk_stage_table.md`](../../docs/research/masking/2026-10-07_mryolk_stage_table.md),
row 26. (1) Of two tracks whose masks overlap, the one more recently
empty is blanked: `comfy/ldm/sam3/tracker.py`,
`_suppress_recently_occluded`, at 0.3 of the larger of intersection over
union and intersection over the smaller mask; Meta's full pipeline has
the rule at 0.7 of intersection over union alone
(`meta_sam3/sam3/model_builder.py`,
`suppress_overlapping_based_on_recent_occlusion_threshold`;
`meta_sam3/sam3/model/sam3_video_base.py`), and Meta's also blanks the
returned mask where ComfyUI's changes only the stored one. Both fire only
between two tracks that have each been empty before. (2) A track that
keeps under 0.3 of its area once every pixel is given to one track gets
an empty memory for that frame: `_deferred_memory_encode` in ComfyUI,
`_suppress_object_pw_area_shrinkage` in Meta's code, the same number.
That reading stands; this run tests whether the difference costs this
subject anything. A different rule, the suppression of a track the
detector stops matching, which an earlier reading found computed and
never applied in Meta's 3.1 code
([`docs/research/masking/2026-10-06_mrsun.md`](../../docs/research/masking/2026-10-06_mrsun.md)),
is not either of these.

The tool's `--rules` switches each off in memory on the tracker object
for the length of one arm (nothing on disk is touched). No arm is Meta's
rule whole.

`lotsofpeopledance_0414_0720.mkv` from 138.0 s, 144 frames at 24.0 a second, 1344x760; the subject is the most central person on frame 4, box [0.457, 0.446, 0.519, 0.679], seeded with the group of up to sixteen; each rule switched off in memory on core's tracker for one arm; the subject in the group of up to sixteen. One run per arm.

| SAM 3.1 | core's rules between followed objects | the subject held from the seed without a break | nudged | with a mask | nudged | pairs of tracks sharing over a half of their masks on some frame | frames with such a pair | of them with the subject | nudged: pairs, frames, with the subject |
|---|---|---|---|---|---|---|---|---|---|
| as ComfyUI ships it | as core has them | 91 | 91 | 97 | 96 | 11 | 21 | 0 | [13, 21, 0] |
| as ComfyUI ships it | the occlusion rule off | 91 | 91 | 97 | 96 | 11 | 21 | 0 | [13, 17, 0] |
| as ComfyUI ships it | the occlusion rule at 0.7, core's measure | 91 | 91 | 97 | 96 | 10 | 22 | 0 | [12, 19, 0] |
| as ComfyUI ships it | the occlusion rule at 0.7 of intersection over union (Meta's threshold and measure) | 91 | 91 | 97 | 96 | 11 | 21 | 0 | [13, 17, 0] |
| as ComfyUI ships it | the memory rule off | 91 | 90 | 97 | 94 | 10 | 23 | 0 | [9, 18, 0] |
| as ComfyUI ships it | both off | 91 | 75 | 97 | 92 | 10 | 25 | 0 | [10, 13, 0] |
| corrected | as core has them | 111 | 107 | 129 | 124 | 18 | 27 | 0 | [17, 26, 0] |
| corrected | the occlusion rule off | 111 | 107 | 129 | 124 | 18 | 25 | 0 | [17, 24, 0] |
| corrected | the occlusion rule at 0.7, core's measure | 111 | 107 | 129 | 123 | 17 | 25 | 0 | [17, 24, 0] |
| corrected | the occlusion rule at 0.7 of intersection over union (Meta's threshold and measure) | 111 | 107 | 129 | 124 | 18 | 23 | 0 | [17, 25, 0] |
| corrected | the memory rule off | 110 | 107 | 128 | 123 | 14 | 24 | 0 | [14, 27, 0] |
| corrected | both off | 111 | 107 | 129 | 123 | 11 | 19 | 0 | [13, 26, 0] |

The prediction written before the run: an arm that changes only the occlusion rule equals core as it is on every track up to the first frame on which any track is empty; an earlier difference voids the arm.

| SAM 3.1 | core's rules between followed objects | first frame on which any track is empty (core as it is) | first frame differing from core as it is | nudged | the prediction holds | nudged |
|---|---|---|---|---|---|---|
| as ComfyUI ships it | as core has them | 12 |  |  |  |  |
| as ComfyUI ships it | the occlusion rule off |  | 36 | 29 | True | True |
| as ComfyUI ships it | the occlusion rule at 0.7, core's measure |  | 41 | 39 | True | True |
| as ComfyUI ships it | the occlusion rule at 0.7 of intersection over union (Meta's threshold and measure) |  | 36 | 29 | True | True |
| as ComfyUI ships it | the memory rule off |  | 17 | 14 |  |  |
| as ComfyUI ships it | both off |  | 17 | 14 |  |  |
| corrected | as core has them | 15 |  |  |  |  |
| corrected | the occlusion rule off |  | 42 | 42 | True | True |
| corrected | the occlusion rule at 0.7, core's measure |  | 46 | 56 | True | True |
| corrected | the occlusion rule at 0.7 of intersection over union (Meta's threshold and measure) |  | 42 | 42 | True | True |
| corrected | the memory rule off |  | 22 | 20 |  |  |
| corrected | both off |  | 22 | 20 |  |  |

**Not the cause.** In every arm the subject is lost at the same frame:
with the occlusion rule off, with it at Meta's threshold on either
measure, with the memory rule off, with both off. The prediction written
before the run held in every arm that changes only the occlusion rule, so
the patches do what they say and the rule does fire later in the run; it
does not touch the subject. About a dozen pairs of the sixteen tracks
coincide on some frame with the rules or without them, never with the
subject, so switching the rules off neither costs merges nor buys the
subject back. `rules-selftest` in the tool is a model-free case for the
two forms of the occlusion rule, with its control.

Not tested: the session logic around the rules (the stage table's row
26b: which detections become new objects, a track refreshed by a
detection, the counter that blanks a track in the output); the
multiplexed pass the followed objects share; Meta's rule whole; these
rules on the morning's sixteen-and-thirty-two comparison with Meta's
tracker.

## The largest person on the hard stretch, read again by continuity

### The crowd clip, the hard stretch, the largest person

`lotsofpeopledance_0414_0720.mkv` from 138.0 s, 144 frames at 24.0 a second, 1344x760; the subject is the largest person on frame 4, box [0.306, 0.633, 0.417, 0.913] (left, top, right, bottom as shares of the frame) (16 detections; one detect on the seed frame as fed, by SAM as it ships; the same masks seed every arm, the subject's first), followed from there; the detector is asked again every 12 frames. One run per arm.

Seed sets: {'alone': {'seeded': 1, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}, 'with the three nearest': {'seeded': 4, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.001}, 'in a group of up to sixteen': {'seeded': 16, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.001}}.

| SAM 3.1 | the subject is seeded | seeded | frames | with a mask | plausible | trusted | first frame lost | the same, nudged: with a mask, plausible, trusted | overlap with its nudged twin | against the alone arm where both have a mask: frames, mean overlap, frames under a half |
|---|---|---|---|---|---|---|---|---|---|---|
| as ComfyUI ships it | alone | 1 | 140 | 19 | 18 | 1 | 23 | 19, 18, 1 | 0.99 |  |
| as ComfyUI ships it | with the three nearest | 4 | 140 | 15 | 15 | 1 | 19 | 14, 14, 1 | 0.894 | [15, 0.935, 0] |
| as ComfyUI ships it | in a group of up to sixteen | 16 | 140 | 74 | 54 | 3 | 73 | 69, 52, 2 | 0.785 | [19, 0.715, 3] |
| corrected | alone | 1 | 140 | 20 | 19 | 19 | 24 | 20, 19, 19 | 0.985 |  |
| corrected | with the three nearest | 4 | 140 | 20 | 18 | 18 | 24 | 20, 17, 17 | 0.973 | [20, 0.928, 1] |
| corrected | in a group of up to sixteen | 16 | 140 | 100 | 92 | 49 | 104 | 99, 95, 70 | 0.863 | [20, 0.774, 4] |

Does the mask stay on one figure? `held` counts frames from the seed frame up to the first that is empty or shares under 0.2 of its mask with the frame before's.

| SAM 3.1 | the subject is seeded | held from the seed without a break | nudged | each mask against the frame before's: least, median | frames under a half (frame, overlap) | nudged: least, frames under a half |
|---|---|---|---|---|---|---|
| as ComfyUI ships it | alone | 16 | 16 | 0.006, 0.9 | [[20, 0.006]] | 0.011, [[20, 0.011]] |
| as ComfyUI ships it | with the three nearest | 15 | 14 | 0.801, 0.875 | [] | 0.75, [] |
| as ComfyUI ships it | in a group of up to sixteen | 20 | 20 | 0.043, 0.662 | [[23, 0.455], [24, 0.08], [25, 0.288], [26, 0.086], [27, 0.483], [28, 0.187], [29, 0.067], [30, 0.228], [31, 0.378], [34, 0.065], [35, 0.178], [36, 0.271], [54, 0.395], [55, 0.396], [60, 0.043], [62, 0.068], [63, 0.471], [64, 0.42], [65, 0.19], [66, 0.268], [67, 0.082], [68, 0.127], [69, 0.245], [87, 0.134], [88, 0.112], [89, 0.342]] | 0.03, [[23, 0.465], [24, 0.071], [25, 0.282], [26, 0.084], [28, 0.111], [29, 0.044], [30, 0.414], [31, 0.349], [32, 0.203], [33, 0.265], [34, 0.097], [35, 0.147], [36, 0.272], [37, 0.47], [59, 0.39], [60, 0.03], [62, 0.057], [63, 0.467], [64, 0.334], [65, 0.084], [66, 0.144], [67, 0.044], [68, 0.114], [87, 0.115], [88, 0.116], [89, 0.117]] |
| corrected | alone | 16 | 16 | 0.001, 0.892 | [[20, 0.001]] | 0.003, [[20, 0.003]] |
| corrected | with the three nearest | 20 | 17 | 0.253, 0.88 | [[21, 0.253]] | 0.096, [[21, 0.096]] |
| corrected | in a group of up to sixteen | 28 | 28 | 0.132, 0.82 | [[31, 0.253], [32, 0.16], [34, 0.45], [35, 0.384], [36, 0.183], [37, 0.132], [44, 0.401], [45, 0.449], [65, 0.261], [67, 0.48], [68, 0.432], [101, 0.402], [102, 0.317]] | 0.04, [[31, 0.302], [32, 0.04], [33, 0.5], [35, 0.397], [65, 0.295], [66, 0.396], [67, 0.254], [68, 0.314], [69, 0.449], [101, 0.25], [102, 0.265]] |

**The group keeps a mask on, not the figure.** Alone, the mask is
continuous for sixteen frames and then shares nothing with the frame
before: the jump the regain-looks record shows frame by frame. In a group
of sixteen the mask stays on for most of the stretch and is plausible on
most of those frames, but it is continuous from the seed for twenty
frames as shipped and twenty-eight corrected, and then moves off over
three or four low steps. One effect of company is real: with neighbours
seeded the mask does not make the half-frame jump (the far figure has its
own track).

## What this does not show

- Identity, anywhere. Continuity of a mask from a seed that was on the
  subject is the strongest statement here.
- A stretch where a whole, in-frame subject is lost when followed alone.
  The three windows have none.
- Why company loses a held subject.
- More than one subject, a regain (every arm is one seeded call forward),
  a cut, or which neighbours make good company.

## Reproducing it

```
<python> bench/subject_alone_or_in_a_group.py run --clip lotsofpeopledance_0414_0720.mkv --second 138 --seconds 6 --width 1344 --rate 24 --subject "most central" --key "crowd, hard, the most central" --json J
<python> bench/subject_alone_or_in_a_group.py run --clip lotsofpeopledance_0414_0720.mkv --second 16 --seconds 6 --width 1344 --rate 24 --subject "most central" --key "crowd, calm, the most central" --json J
<python> bench/subject_alone_or_in_a_group.py run --clip vma.mp4 --second 121.93 --seconds 3 --width 1344 --rate 24 --subject "most central" --key "dense, the most central" --json J
<python> bench/subject_alone_or_in_a_group.py run --clip lotsofpeopledance_0414_0720.mkv --second 138 --seconds 6 --width 1344 --rate 24 --subject "most central" --rules --key "crowd, hard, the most central, core's rules" --json J
<python> bench/subject_alone_or_in_a_group.py run --clip lotsofpeopledance_0414_0720.mkv --second 138 --seconds 6 --width 1344 --rate 24 --subject largest --key "crowd, hard, the largest" --json J
<python> bench/subject_alone_or_in_a_group.py rules-selftest
<python> bench/subject_alone_or_in_a_group.py render --json J
```

The first form's three keys were run the same way with `--subject
largest` and the keys `crowd, hard`, `crowd, calm` and `dense`, by the
tool as it stood before the per-frame measure.

## The first form's tables (the largest person; plausible masks; kept as measured)

### crowd, hard

`lotsofpeopledance_0414_0720.mkv` from 138.0 s, 144 frames at 24.0 a second, 1344x760; the subject is the largest person on frame 4, box None (left, top, right, bottom as shares of the frame) (16 detections; one detect on the seed frame as fed, by SAM as it ships; the same masks seed every arm, the subject's first), followed from there; the detector is asked again every 12 frames. One run per arm.

Seed sets: {'alone': {'seeded': 1, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}, 'with the three nearest': {'seeded': 4, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.001}, 'in a group of up to sixteen': {'seeded': 16, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.001}}.

| SAM 3.1 | the subject is seeded | seeded | frames | with a mask | plausible | trusted | first frame lost | the same, nudged: with a mask, plausible, trusted | overlap with its nudged twin | against the alone arm where both have a mask: frames, mean overlap, frames under a half |
|---|---|---|---|---|---|---|---|---|---|---|
| as ComfyUI ships it | alone | 1 | 140 | 19 | 18 | 1 | 23 | 19, 18, 1 | 0.99 |  |
| as ComfyUI ships it | with the three nearest | 4 | 140 | 15 | 15 | 1 | 19 | 14, 14, 1 | 0.894 | [15, 0.935, 0] |
| as ComfyUI ships it | in a group of up to sixteen | 16 | 140 | 74 | 54 | 3 | 73 | 69, 52, 2 | 0.785 | [19, 0.715, 3] |
| corrected | alone | 1 | 140 | 20 | 19 | 19 | 24 | 20, 19, 19 | 0.985 |  |
| corrected | with the three nearest | 4 | 140 | 20 | 18 | 18 | 24 | 20, 17, 17 | 0.973 | [20, 0.928, 1] |
| corrected | in a group of up to sixteen | 16 | 140 | 100 | 92 | 49 | 104 | 99, 95, 70 | 0.863 | [20, 0.774, 4] |

### crowd, calm

`lotsofpeopledance_0414_0720.mkv` from 16.0 s, 144 frames at 24.0 a second, 1344x760; the subject is the largest person on frame 4, box None (left, top, right, bottom as shares of the frame) (16 detections; one detect on the seed frame as fed, by SAM as it ships; the same masks seed every arm, the subject's first), followed from there; the detector is asked again every 12 frames. One run per arm.

Seed sets: {'alone': {'seeded': 1, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}, 'with the three nearest': {'seeded': 4, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.002}, 'in a group of up to sixteen': {'seeded': 16, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.002}}.

| SAM 3.1 | the subject is seeded | seeded | frames | with a mask | plausible | trusted | first frame lost | the same, nudged: with a mask, plausible, trusted | overlap with its nudged twin | against the alone arm where both have a mask: frames, mean overlap, frames under a half |
|---|---|---|---|---|---|---|---|---|---|---|
| as ComfyUI ships it | alone | 1 | 140 | 140 | 140 | 140 | None | 140, 140, 140 | 0.996 |  |
| as ComfyUI ships it | with the three nearest | 4 | 140 | 140 | 140 | 140 | None | 140, 140, 140 | 0.998 | [140, 0.992, 0] |
| as ComfyUI ships it | in a group of up to sixteen | 16 | 140 | 140 | 140 | 140 | None | 140, 140, 140 | 0.997 | [140, 0.984, 0] |
| corrected | alone | 1 | 140 | 140 | 140 | 140 | None | 140, 140, 140 | 0.997 |  |
| corrected | with the three nearest | 4 | 140 | 140 | 140 | 140 | None | 140, 140, 140 | 0.998 | [140, 0.994, 0] |
| corrected | in a group of up to sixteen | 16 | 140 | 140 | 140 | 140 | None | 140, 140, 140 | 0.996 | [140, 0.986, 0] |

### dense

`vma.mp4` from 121.93 s, 72 frames at 24.0 a second, 1344x760; the subject is the largest person on frame 4, box None (left, top, right, bottom as shares of the frame) (16 detections; one detect on the seed frame as fed, by SAM as it ships; the same masks seed every arm, the subject's first), followed from there; the detector is asked again every 12 frames. One run per arm.

Seed sets: {'alone': {'seeded': 1, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}, 'with the three nearest': {'seeded': 4, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}, 'in a group of up to sixteen': {'seeded': 16, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}}.

| SAM 3.1 | the subject is seeded | seeded | frames | with a mask | plausible | trusted | first frame lost | the same, nudged: with a mask, plausible, trusted | overlap with its nudged twin | against the alone arm where both have a mask: frames, mean overlap, frames under a half |
|---|---|---|---|---|---|---|---|---|---|---|
| as ComfyUI ships it | alone | 1 | 68 | 5 | 5 | 5 | 9 | 5, 5, 5 | 0.995 |  |
| as ComfyUI ships it | with the three nearest | 4 | 68 | 17 | 17 | 1 | 21 | 17, 17, 1 | 0.992 | [5, 0.988, 0] |
| as ComfyUI ships it | in a group of up to sixteen | 16 | 68 | 68 | 36 | 12 | None | 68, 37, 2 | 0.993 | [5, 0.982, 0] |
| corrected | alone | 1 | 68 | 6 | 6 | 0 | 10 | 6, 6, 6 | 0.994 |  |
| corrected | with the three nearest | 4 | 68 | 27 | 27 | 15 | 31 | 27, 27, 27 | 0.99 | [6, 0.991, 0] |
| corrected | in a group of up to sixteen | 16 | 68 | 68 | 33 | 16 | None | 68, 36, 25 | 0.99 | [6, 0.984, 0] |

## Files

- `2026-10-07_subject_alone_or_in_a_group.json`: one key per run; each
  with its subject, seed sets and arms.
