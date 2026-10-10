bump: minor

### Added

- `MiniMaxH3BodyMeshVideo` gains a third `style`, `marked`: the lit grey
  mesh with the face side of the head, the back of it and each hand in flat
  colours (`body_pose.py::MARK_COLOURS`), so which way a head faces is an
  area of colour at a small size and the hands can be told apart: the face
  side yellow, the back of the head blue, the person's right hand red, the
  left green. The node gains a second output, `legend`, the one sentence
  that names those colours for a prompt using the frames (empty for the
  other styles); a text takes the colours from there, never from memory.
  Which vertex is which part is a fact about the rig, so it is in a new
  file, `body_marks.json`, written from the rig's own keypoints at rest by
  `bench/check_body_pose.py --write-marks` and read only when the style is
  drawn: a pose pass runs nothing for it and its pose data carries nothing
  for it (mrcorn's cold read and mrpop: a pass must not be lost to a
  decoration), and the check holds the file against the rig. A signal to test, not a
  default. `bench/check_body_pose.py` holds that with nothing marked the
  draw is ComfyUI's public one pixel for pixel, that marks change only
  where a part is and only to the flat colours, and that a head turned to
  the camera is nearly all face side and turned away nearly none.

### Changed

- From mrcorn's second cold read. The Subject Track's report says, per
  shot, when people on the tile's frame belong to another tracker and are
  not numbered. A frame a track was seeded on has no tracker score in the
  shot table: the tracker was told there, and its number is not a reading.
  `ScoreTap` puts back whatever the tracker object already carried under
  the name it wraps.
