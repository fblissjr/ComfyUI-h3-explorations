bump: patch

### Added

- **The frames of an unbroken run a gallery may be taken from**
  (`subject_tracks.gallery_span`): the run, less its outermost frame on a
  side where it stopped because the mask moved off. On the one jump recorded
  so far the frame before it already lay on both figures at once while
  sharing most of its pixels with the frame before
  (`bench/results/2026-10-07_subject_regain_looks.md` and the per-frame
  track beside it), so the cut keeps it and a gallery must not: the Subject
  Track's `gallery_frames` takes the frame with the largest mask by rule. A
  rule with no number, reasoned from that one case. The seed frame is never
  left out, and a run that stopped at an empty frame or the track's end
  loses nothing. `unbroken` cuts as it did and now also returns the seed and
  each frame's box area ratio, which sees a far piece of a mask that the
  mask's own area barely moves; nothing is decided on it.
  `bench/check_subject_tracks.py` has the case, with the node's own gallery
  taking that frame as its control. Imported by no node.
