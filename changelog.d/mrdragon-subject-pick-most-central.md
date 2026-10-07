bump: minor

### Changed

- **The Subject Track picks the most central person by default; it was the
  largest.** The owner's decision of 2026-10-07 on a measured difference: on
  the lane's crowd clip the two rules name different people on all three
  windows tested, `largest` a figure at the frame's edge and `most central`
  the person meant, and with `most central` today's node holds the subject
  on every frame of the stretch that had been called hard
  (`bench/results/2026-10-07_subject_track_under_nudge.md`). Moved together:
  `MiniMaxH3SubjectTrack`'s `pick` default (`subject_track.py`),
  `h3_config.SUBJECT_TRACK` and every generated masked graph, rebuilt.
  `largest` is still a choice on the node. **An arm recorded before this
  change was picked with `largest`**: reproducing one needs `pick` set to
  that, since the recorded patches did not name it.
  `docs/wiki/decisions.md` has the line and `docs/wiki/masked_v2v.md` says
  which default a render had.
