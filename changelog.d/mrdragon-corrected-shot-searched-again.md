bump: minor

### Fixed

- **A shot corrected by hand is looked for again when its track lets go.**
  `subject_track.follow` tracked a corrected shot once from the corrected
  seed and left it empty from the frame the tracker let the subject go; the
  report said so ("none on N to the end") and the render showed the original
  person from there. It now runs the same search on a corrected shot as on
  any other, with the corrected track's own frames as the gallery, and the
  report names where the subject was found again. Found on a window that
  starts with the subject small, where a correction is the only way to
  follow him and the tracker let go at a burst of movement around him
  (`bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md`, section
  5). `bench/check_subject_track.py` has the case, red against the node
  before this. The search is still by likeness with a margin.
