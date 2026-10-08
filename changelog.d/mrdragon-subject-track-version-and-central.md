bump: minor

### Fixed

- **The Subject Track's kept-mask version is bumped, so 0.224.0's fix runs.**
  `MiniMaxH3SubjectTrack.MASK_VERSION` 9 to 10. 0.224.0 changed how a
  corrected shot is followed and left the version alone, so a render with
  the same inputs loaded the mask kept before the fix and tracked nothing;
  the first confirmation render showed the old result for that reason.

### Changed

- **`most central` is measured in the frame's own proportions.**
  `subject_track.choose` scaled each axis to the same range, which measures
  a wide frame as if it were square and counts a step up or down about 1.8
  times a step sideways on 16:9. Both axes are now in units of the frame's
  width. Tested first on the two frames of one clip where the subject's
  detection is known, every detection's mask saved from the server: he is
  second of twelve on one frame and first of sixteen on the other under the
  old measure and the new alike, and third or fourth of sixteen on the
  second frame when the centre is taken from the head and shoulders, which
  is why that part is not in. So this corrects the measure; it does not fix
  a known miss.
- `docs/wiki/masked_v2v.md` says how each pick rule measures, that no rule
  is right everywhere, and what a correction does and does not do.
