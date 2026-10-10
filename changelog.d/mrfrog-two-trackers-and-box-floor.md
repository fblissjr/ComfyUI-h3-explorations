bump: minor

### Added

- `MiniMaxH3SubjectTrack` gains `others`, an optional mask of the people
  other Subject Track nodes already follow. A detection lying mostly on it
  (`subject_track.py::ON_OTHERS`, measured on one clip) is never this node's
  subject, so two nodes in one graph cannot follow the same person; a shot
  showing only those people has no subject. The report and the shot table
  say what was left out and on which frames this node's own mask sits on
  such a person. Nothing is cut on that.
- The shot table carries the tracker's own score per frame of each shot
  (`track_score`), read by wrapping one method of the loaded tracker for the
  length of a track call. A record that could be out of step with its frames
  is not written, and the report says why. Nothing reads the score yet.
- `MiniMaxH3SubjectBoxes` gains `smallest_mask`: a frame whose mask covers
  fewer pixels than that has no box (`subject_boxes.py::SMALLEST_MASK_PX`,
  measured on one clip), and the report tells such a frame from an empty one.
- `MiniMaxH3BodyPose`'s table gains each person's 3D keypoints in camera
  space, the parts of the body that lie outside the frame and on which side,
  the boxes it refused, and whether one box list served every frame.

### Changed

- `MiniMaxH3BodyPose` gives no body to a box with no area, off the frame or
  not a number, where it drew one; frames are rounded to 8 bits, not
  truncated; its report names which parts of the body are outside the frame
  in place of a count over the clip, says when one box list was used for
  every frame, and ends with the seconds the pass took. Found by mrcorn's
  cold read, whose three stand-in cases are in `bench/check_body_pose.py`.
- `MiniMaxH3SubjectTrack.MASK_VERSION` is 12.
