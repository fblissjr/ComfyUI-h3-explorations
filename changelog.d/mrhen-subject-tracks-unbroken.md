bump: patch

### Added

- **A test, on every frame, that a tracked mask has stayed on one figure**
  (`subject_tracks.unbroken`). Each frame's mask is compared with the last
  mask before it, walking away from the frame the track was seeded on, and
  the track is cut at the first empty frame or the first step that shares
  too little; what reaches the seed without a cut is the only part a gallery
  may be taken from. It answers the fault of
  `bench/results/2026-10-07_subject_regain_looks.md`: a mask moved onto
  another figure with no empty frame between, which a count of masks and the
  shape test both pass. The line is an argument (`moved_off`), its default
  reasoned on one clip at one frame rate. **It is a test of continuity, not
  of identity**: a mask that grows over two figures and shrinks onto the
  other is not cut, and `bench/check_subject_tracks.py` asserts that as a
  known limit beside the cases it does catch, with the area ratio and the
  box centre's step returned for a report. It takes one tracked call, not a
  piece stitched from several. Model-free, and imported by no node: the
  Subject Track is unchanged.
