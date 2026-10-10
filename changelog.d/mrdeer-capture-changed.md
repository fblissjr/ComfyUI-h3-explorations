bump: minor

### Added

- `bench/capture_masked_run.py changed`: what a run redrew, per segment and per subject inside its region, as the mean grey difference from the source against a floor taken outside the region; one segment's difference frame by frame (`--series`); and the change from the frame before inside the region, render beside source. It replaces five session scripts that the first day's findings rested on (the other subject redrawn under a pass's region, a face drawn back toward the original beside kept pixels, a face pass level across a window boundary, no step at a seam) and gives the same figures as they did on the render it was checked against. `frame_changes` is the function; the check holds it on a frame whose differences are known.

### Fixed

- The cut rule took its cuts only from the run's own subject's shot table and said nothing when that subject had none; it now uses any subject's table in the capture, since cuts are the source's.
