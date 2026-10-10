bump: minor

### Added

- `bench/capture_masked_run.py files` takes a subject's body pose table
  (`pose=TABLE.json[,pose_at=N]` on `--mask`: what `MiniMaxH3BodyPose`
  writes, schema `h3_body_pose_table/1`, another version refused). The
  capture keeps the table and writes a row a frame for the subject: whose
  box the body was fitted to, which hands the hand decoder refined, the
  share of keypoints outside the frame, how many bodies carry the subject's
  name. `preflight` reads it before the mesh is used as a motion video
  (`flag_pose`): `pose_fitted_to_the_whole_frame`, at the top level when
  another subject of the capture is in the frame, since the body returned
  can then be theirs; `several_bodies_under_one_name`;
  `no_pose_where_the_subject_is`; `hand_not_refined`, one flag a side, only
  for a table made with refinement on; and `pose_mostly_outside_the_frame`
  over `POSE_OUTSIDE` of the keypoints. Pinned on a hand-written table in
  `bench/check_capture_masked_run.py`, and read once against a table the
  node wrote from a public sample.
