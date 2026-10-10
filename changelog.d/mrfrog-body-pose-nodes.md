bump: minor

### Added

- `body_pose.py`: three nodes that put a body mesh in a graph with no ComfyUI
  SAM node in it. `MiniMaxH3BodyModelLoader` loads either release of SAM 3D
  Body. `MiniMaxH3BodyPose` takes frames and `MiniMaxH3SubjectBoxes`' boxes
  and returns the pose data and a table per person per frame (the box, the
  crop, each hand's crop and whether its decoder was used, the keypoints),
  as text and, given a prefix, as a file beside the render.
  `MiniMaxH3BodyMeshVideo` draws the pose as frames on black. ComfyUI's model
  code and rasteriser are called as a library; the crop, for the body and
  for both hands, is Meta's own, transcribed, where ComfyUI's node samples it
  another way (`bench/results/2026-10-10_sam3d_body_core_against_meta.md`).
- `bench/check_body_pose.py` and `bench/fixtures/sam3d_body_meta_reference.json`:
  our crop against the crop Meta's own code made, bit for bit, with ComfyUI's
  crop as the control; with the weights on disk, our keypoints inside Meta's
  own precision floor on every box of two public samples and each hand's
  decoder used exactly where Meta's code used it.
  `bench/compare_sam3d_body_core_against_meta.py fixture` writes the fixture.
- `opencv-python-headless` is declared in `pyproject.toml` (it went in with
  0.259.3's commit): `body_pose.py` samples its crops with OpenCV, as Meta's
  code does. The pack loads without it; the node says what it needs.

### Changed

- `docs/wiki/meta_perception_models.md`, `docs/wiki/h3_uses.md`,
  `docs/checks.md` and the comparison's record say what replaces ComfyUI's
  SAM 3D Body nodes and what the new table gives the preflight.
