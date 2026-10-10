bump: minor

### Added

- `bench/compare_sam3d_body_core_against_meta.py`: ComfyUI core's SAM 3D
  Body prediction against Meta's own inference code on the same image, boxes
  and camera. Core's nodes run in process; Meta's code runs in a child
  process from `coderef/sam-3d-body`, as a one-off numeric reference, with a
  Python outside this repo. It carries a control (core's model with the crop
  sampled as Meta samples it), a floor (Meta's two precisions against each
  other) and a direct reading of whether each hand's decoder was used.
- `bench/results/2026-10-10_sam3d_body_core_against_meta.md` and its data
  file: two public samples, on the CPU in float32. Core computes what Meta's
  code computes once both are given the same crop; as shipped it is off by a
  small amount that is all the crop's sampling; the camera is the same on
  both sides; expression is zero on both.

### Changed

- `docs/wiki/meta_perception_models.md`: what the run says of each
  difference between core's port and Meta's inference, a section of
  candidate preflight flags for a mesh-driven render, and which weights are
  now on disk.
