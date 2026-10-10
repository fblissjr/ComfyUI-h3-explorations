bump: minor

### Changed

- **An absence with people on screen blocks the gate until it is answered
  in writing** (`bench/capture_masked_run.py preflight`,
  `absent_with_people_on_screen`). It was "iffy" whatever it scored. On
  2026-10-10 a shot that both trackers had called absent, each with its
  subject on screen, went through a day of renders with nothing laid on it
  and reached a viewer as the original. It is now at the top level unless
  the tracker's table carries a typed correction for the shot (a person, or
  none) or `--not-in` covers the whole shot; half a shot does not answer it.
  A corrected shot's state (`taken (corrected)`) is read as taken, and a
  typed take under the match line is no longer flagged as a guess.
- `verify` judges the two region readers' carried masks by the pixels of
  either that lie over a pixel from the other (`READERS_CARRIED_OFF`), not
  by their overlap, which falls with the mask's size for the same edge and
  had failed a face that the two readers agree on. The overlap stays in the
  record.

### Added

- `load_has_a_shot_without_the_subject`, top level: a planned load holds a
  shot on which its subject has no mask at all, and `--not-in` does not say
  that is right. `subject_small_for_the_whole_load`, iffy: the subject's
  mask never reaches `SMALL_SUBJECT` of the frame in a load.
- `mouth` reports the source's mouth open with no voice (`open_runs`): the
  runs of `OPEN_RUN` frames or more on which the source's opening is in the
  top `OPEN_SHARE` of the shot and above its median while the voice table
  says unvoiced, with each render's opening over the run as a share of the
  source's. `--voice` gives the clip's table to a capture made without one.
  It is the join `docs/wiki/state_signals.md` asks for under "A mouth open
  with no voice".

All pinned in `bench/check_capture_masked_run.py`.
