bump: patch

### Added

- **Today's Subject Track at each look after a loss.**
  `bench/subject_regain_looks.py` and
  `bench/results/2026-10-07_subject_regain_looks.md`: following the most
  central person on a crowd stretch the node never loses the mask;
  following the largest, ComfyUI's tracker moves the mask to another
  figure in one frame with no empty frame between, the node takes that
  figure back, and its later takes land in three places. Read off the
  data, a candidate that overlaps where the subject stood is the one at
  the subject's place, and it is among the candidates far more often when
  thirty-two or sixty-four detections are asked than sixteen. The track
  is saved frame by frame.
- **Is a track still on the subject: five measures of likeness.**
  `bench/subject_likeness_in_a_group.py` and
  `bench/results/2026-10-07_subject_likeness_in_a_group.md`: rank against
  the other tracks works and the node's absolute line does not; the
  lightness of the pixels under the mask separates most cleanly where the
  subject differs from the people around; the node's own measure has no
  head to compare on some looks. One pass.
- **`bench/subject_alone_or_in_a_group.py --rules` and `rules-selftest`**:
  ComfyUI's two rules between followed objects switched off in memory,
  one at a time and both, with a count of tracks that coincide; and a
  model-free case for the two forms of the occlusion rule.

### Changed

- **`bench/results/2026-10-07_subject_alone_or_in_a_group.md` is replaced
  by its corrected form.** Its first verdict, that a subject is held
  several times longer in a group than alone, was the largest person's,
  counted in plausible masks, which count a mask that has moved to
  another figure. Counted by continuity from the seed: the most central
  person is held alone on all three windows and company costs frames on
  the hard one; the largest is held a few frames longer in a group, not
  several times. With either of ComfyUI's two rules between followed
  objects off, or both, the central person in a group is lost at the same
  frame. The first form's tables are kept in the record.
- **Dated corrections in three places that read as general and were the
  largest person's**: the verdict and top of
  `2026-10-07_subject_track_under_nudge.md` (the arms agree; that is not
  "no regain lands on another person"), the lone-subject sentence of
  `2026-10-07_sam3_core_against_meta.md` and of
  `docs/research/masking/2026-10-07_mryolk.md` (a figure cut by two edges
  of the frame), and the comparison record's sentence on a CPU process,
  which now points at the cause.
