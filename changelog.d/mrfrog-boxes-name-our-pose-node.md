bump: patch

### Changed

- `MiniMaxH3SubjectBoxes`' description names `MiniMax H3 Body Pose` as the
  node its boxes go into, where it named ComfyUI's own.
- `bench/check_body_pose.py --card` runs the four weights cases on the card,
  in the precision the server's loader picks there. A sweep never passes it.
