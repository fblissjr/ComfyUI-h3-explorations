bump: patch

### Added

- `bench/results/2026-10-09_whole_frame_text_and_copy_size.md` and its json:
  the whole-frame video-to-video renders of 2026-10-09 on one clip. Matched
  pairs in which a window's text decided whether the render kept its
  source's framing; the short edges at which the video model's copy of the
  source held; the lead's face with and without a still; Sol twins; a held
  frame and a timing lead with what was ruled out for each. Stills and one
  timing measure, not judged on playback.

### Changed

- `docs/h3_references.md`, "Edit a source video": when the source already
  shows the person and the video model has its own copy of it, the still is
  left out. A rule the owner had recorded; no default changed.
  `docs/wiki/decisions.md` has the line.
