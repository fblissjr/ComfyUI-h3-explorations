bump: minor

### Added

- `bench/capture_masked_run.py preflight`, two rules that were readings
  after a render and are flags before one:
  `kept_frames_far_from_the_pose_ahead` (`flag_kept`, `turn_from_kept`,
  iffy): for a planned continuation, how far the source's head turns over
  the window's new frames from where it faces on the last kept frame
  (`KEPT_TURN`, inherited from the facing measure's tolerance), with the
  chin's change and a wrist's approach to the nose as figures, from the
  node's plan and the subject's pose table; `kept_frames_not_checked` when
  the subject has no pose table with 3D keypoints.
  `mouth_open_with_no_voice` (`flag_mouth`, iffy): the runs `mouth` reports,
  from the source's class map and the capture's voice table, for every
  subject a run replaces. Pinned in `bench/check_capture_masked_run.py`.
