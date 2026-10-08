bump: patch

### Added

- **A record of this pack's one-person track beside core's text-prompted
  multi-object track on one load**
  (`bench/results/2026-10-08_one_person_track_against_core_multi_object.md`,
  figures in the `.json` beside it). The owner asked to see every object
  core's `SAM3_VideoTrack` holds, with our subject marked. On the one load
  it was run on, this pack's Subject Track held the subject on every frame;
  core's pass held him and the person in front under their own ids for the
  first part of the load, then ran out of object slots as the crowd moved
  and tracked nobody to the end. Why it stops is read in core's code
  (slots are never freed and detection runs only while one is free), and
  the figures are consistent with it. One run of each, not the same task
  (core was given a phrase, ours was told which person), and no claim about
  which tracker is better in general. No code changes.
