# One subject in a crowd: followed alone, or seeded together with the people around them (2026-10-07)

lane: masked
verdict: on a hard crowd stretch the subject keeps a plausible mask three times longer with SAM as shipped, and five times longer with the two corrections on, when seeded in a group with the fifteen most prominent people than when followed alone as today's Subject Track does; three neighbours are no better than alone there; on a calm stretch every arm holds every frame; each arm's twin with the input moved one level agrees within a few frames; not shown: that the later plausible frames are still the subject and not a neighbour

**What was asked.** The owner, 2026-10-07: "Being able to confidently
track objects/subjects/segmented subjects in crowded scenes and across
shots is super important and we have to get that right first", with
`lotsofpeopledance` named as the clip to test on. Today's Subject Track
seeds ComfyUI's tracker with its one subject alone
(`subject_track.py::_sam_callables`). Earlier the same day a subject
followed alone on a dense frame was lost within a few frames, in ComfyUI's
port and in Meta's own tracker alike
([`2026-10-07_sam3_core_against_meta.md`](2026-10-07_sam3_core_against_meta.md)).

**How.** [`bench/subject_alone_or_in_a_group.py`](../subject_alone_or_in_a_group.py).
ONE subject (the largest person on the seed frame, as the node picks) and
ONE set of seed masks for every arm, found once on the frames as fed by
SAM as it ships; a candidate that is a second mask of an already chosen
person is left out. The subject is followed from the seed frame alone,
with the three nearest people, and in a group of up to sixteen (the most
prominent, the subject first), by ComfyUI's SAM 3.1 as it ships and
through [`sam31_corrections.py`](../../sam31_corrections.py); every arm
again with every input value moved one level of 255. Counted, for the
SUBJECT's track only: frames with a mask; frames whose mask passes the
shape test of [`subject_tracks.py`](../../subject_tracks.py) ("plausible":
not a strip along the frame's border, not a fragment); frames that are
"trusted" (plausible, and not contradicted by the detector on the frames
it is asked again); and the first frame lost. Each group arm also reports
its overlap with the alone arm where both have a mask. One run per arm;
outside the server, the card to itself. Every number is in
[`2026-10-07_subject_alone_or_in_a_group.json`](2026-10-07_subject_alone_or_in_a_group.json)
and the tables are printed from it by the tool's `render`.

**The reading, written before the first run** (the tool's docstring). "In
a group" is the tracker node's design only if, on the hard crowd stretch,
the subject's trusted frames in a group exceed alone by more than the
larger of the two arms' own nudged-against-plain differences, in both the
plain and the nudged pair, and it is not worse by that much on the calm
stretches. Otherwise alone, since alone costs least.

### crowd, hard

`lotsofpeopledance_0414_0720.mkv` from 138.0 s, 144 frames at 24.0 a second, 1344x760; the subject is the largest person on frame 4 (16 detections; one detect on the seed frame as fed, by SAM as it ships; the same masks seed every arm, the subject's first), followed from there; the detector is asked again every 12 frames. One run per arm.

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

`lotsofpeopledance_0414_0720.mkv` from 16.0 s, 144 frames at 24.0 a second, 1344x760; the subject is the largest person on frame 4 (16 detections; one detect on the seed frame as fed, by SAM as it ships; the same masks seed every arm, the subject's first), followed from there; the detector is asked again every 12 frames. One run per arm.

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

`vma.mp4` from 121.93 s, 72 frames at 24.0 a second, 1344x760; the subject is the largest person on frame 4 (16 detections; one detect on the seed frame as fed, by SAM as it ships; the same masks seed every arm, the subject's first), followed from there; the detector is asked again every 12 frames. One run per arm.

Seed sets: {'alone': {'seeded': 1, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}, 'with the three nearest': {'seeded': 4, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}, 'in a group of up to sixteen': {'seeded': 16, 'second_masks_of_a_chosen_person_dropped': 0, 'largest_overlap_with_the_subject': 0.0}}.

| SAM 3.1 | the subject is seeded | seeded | frames | with a mask | plausible | trusted | first frame lost | the same, nudged: with a mask, plausible, trusted | overlap with its nudged twin | against the alone arm where both have a mask: frames, mean overlap, frames under a half |
|---|---|---|---|---|---|---|---|---|---|---|
| as ComfyUI ships it | alone | 1 | 68 | 5 | 5 | 5 | 9 | 5, 5, 5 | 0.995 |  |
| as ComfyUI ships it | with the three nearest | 4 | 68 | 17 | 17 | 1 | 21 | 17, 17, 1 | 0.992 | [5, 0.988, 0] |
| as ComfyUI ships it | in a group of up to sixteen | 16 | 68 | 68 | 36 | 12 | None | 68, 37, 2 | 0.993 | [5, 0.982, 0] |
| corrected | alone | 1 | 68 | 6 | 6 | 0 | 10 | 6, 6, 6 | 0.994 |  |
| corrected | with the three nearest | 4 | 68 | 27 | 27 | 15 | 31 | 27, 27, 27 | 0.99 | [6, 0.991, 0] |
| corrected | in a group of up to sixteen | 16 | 68 | 68 | 33 | 16 | None | 68, 36, 25 | 0.99 | [6, 0.984, 0] |

## What this shows

- **On the hard stretch, a group of sixteen holds the subject several
  times longer than alone** (measured; the nudged twin of every arm agrees
  within three frames on plausible frames). Met by the rule written first,
  for the corrected model: the trusted frames in a group exceed alone by
  more than the group arm's own nudged difference, in both pairs.
- **Three neighbours are not enough there**, and help on the dense window.
- **On the calm stretch company costs nothing**: every arm holds every
  frame, and the group arms' masks are the alone arm's.
- **The two corrections matter for the subject in a group on the hard
  stretch** (measured, plausible frames; the shape test does not use the
  detector and the seeds are shared, so this is the tracker alone).
- **A mask that stays on is not a mask that stays a person**: on the dense
  window the group arm has a mask on every frame and a plausible one on
  about half.

## What it does not show

- **That the later plausible frames are still the subject.** A plausible
  mask on a neighbour counts. Where both arms have a mask on the hard
  stretch, the group arm's is not always on the alone arm's pixels (the
  last column), and after the alone arm is lost there is nothing to
  compare with. A likeness test of the tracked mask against the subject,
  and an eye on the stacked masks, are what would say.
- **Anything steady from "trusted" on dense footage.** The watch behind it
  moves with what the detector returns under a one-level change; and with
  SAM as shipped it trusted almost nothing on the hard stretch, where
  every look returns exactly the sixteen detections asked for and the
  frame holds more people than that.
- "As ComfyUI ships it" against "corrected" on TRUSTED frames: the
  detector behind the looks changes too. Alone against group within one of
  them is the clean comparison.
- More than one subject, one window per kind of footage, a regain (every
  arm is one seeded call), or which neighbours make good company.

## Reproducing it

```
<python> bench/subject_alone_or_in_a_group.py run --clip lotsofpeopledance_0414_0720.mkv --second 138 --seconds 6 --width 1344 --rate 24 --subject largest --key "crowd, hard" --json J
<python> bench/subject_alone_or_in_a_group.py run --clip lotsofpeopledance_0414_0720.mkv --second 16 --seconds 6 --width 1344 --rate 24 --subject largest --key "crowd, calm" --json J
<python> bench/subject_alone_or_in_a_group.py run --clip vma.mp4 --second 121.93 --seconds 3 --width 1344 --rate 24 --subject largest --key "dense" --json J
<python> bench/subject_alone_or_in_a_group.py render --json J
```

## Files

- `2026-10-07_subject_alone_or_in_a_group.json`: one key per window, each
  with its seed sets and twelve arms.
