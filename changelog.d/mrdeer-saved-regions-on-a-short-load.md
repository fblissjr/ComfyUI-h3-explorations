bump: minor

### Added

- **`bench/capture_masked_run.py verify <capture>`: does the capture read
  what the nodes wrote.** The first check of a job, before a plan is
  trusted. It reads the capture folder, writes `verify.json` and exits 0 or
  `VERIFY_FAILED`; a check that cannot be made is not a pass. The checks:
  every mask video of a sighting was written in one run
  (`WRITTEN_TOGETHER_S`; a folder holding two previews under one name is how
  a morning's mask gets loaded in the afternoon), is at the capture's canvas
  and reaches the span; no track is empty inside a shot its table says it
  was taken in; a plan that carries a part has that subject's class map;
  and, for every rendered run, the region its windows saved against the
  region read back from the review, cell for cell, with the carried mask's
  overlap (`READERS_CELLS`, `READERS_CARRIED`, each set a little under the
  one measurement there is). `files` now records each input's write time,
  size and length, and reads both regions when a render has both.

- **A fill of a doubted part is graded by the class map** (`grade_holds`,
  `part_grades.json`). The capture's fill takes a neighbour's shape to a
  frame the gate doubted; on a fast turn the doubted part was right and the
  neighbour's shape landed on an arm, a hat or the back of a head. Each of
  the saved part and the fill is now scored by the share of its pixels on
  the part's own classes: the fill is taken only where it scores at least
  `HOLD_INSIDE` and above the saved part, the saved part stays where it
  scores that, and a frame where neither does is emptied. `grade=off` on
  `--mask` leaves the fill ungraded. It cannot see a frame where the class
  map itself is wrong; `drop=` is still the caller's.
- **What a body is doing, per frame, from a pose table with 3D keypoints**
  (`pose_state`, columns of `pose__<by>`): which way the body, the hips and
  the head face and the chin's lift, by `bench/measure_subject_yaw.py`'s own
  arithmetic, and each wrist's distance from the nose in torso units.

### Fixed

- The reader of the regions a render's windows saved refused the first
  two-window render it met: the last window of a load runs past the load's
  last frame (its tail is held frames the render does not write), and the
  reader wanted the windows to end exactly on the render's last frame.
  Frames past the render's end are now dropped and counted in the manifest
  (`held_frames_past_the_render`), a window file after the one that reaches
  the render's end is an earlier, longer run's and is left unread and named,
  and windows that leave a gap or stop short are refused as before.

Both pinned in `bench/check_capture_masked_run.py`. Run on three real
renders, the saved region and the region un-tinted from the review were the
same cell for cell on every frame: the first evidence that the review
reader, which every capture made before the saved files rests on, read the
region right.
