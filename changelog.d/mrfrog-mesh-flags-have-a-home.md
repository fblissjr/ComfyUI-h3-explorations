bump: patch

### Changed

- `docs/wiki/meta_perception_models.md`: each candidate preflight flag for a
  mesh-driven render names the tracked tool it belongs in and the input that
  tool still lacks; one of them is closed for a graph that takes its boxes
  from `MiniMaxH3SubjectBoxes`. A section says where every probe behind the
  page is, so nothing rests on a session folder.
- `bench/compare_sam3d_body_core_against_meta.py`'s docstring holds the
  recipe for the Python that runs Meta's side, and its record
  (`bench/results/2026-10-10_sam3d_body_core_against_meta.md`) the commands
  that reproduce it.
