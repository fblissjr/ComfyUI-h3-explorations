bump: patch

### Changed

- `MiniMaxH3BodyModelLoader` loads SAM 3D Body in float32 always, where it
  took the half precision ComfyUI's own loader takes on the card. Measured
  with `bench/check_body_pose.py --card`: in half precision the keypoints
  were outside Meta's own precision floor on four boxes of six, in float32
  inside it on all six
  (`bench/results/2026-10-10_sam3d_body_core_against_meta.md`, "On the
  card"). The node's memory estimate is doubled to match. No input moved.
- `bench/check_body_pose.py`: the keypoint case measures every box before it
  judges any, so a red line carries all six figures; the silhouette's box
  may sit two pixels from the projection's, with the reason beside the
  constant; on the card the first line prints the computed and the stored
  types.
