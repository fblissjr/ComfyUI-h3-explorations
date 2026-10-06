bump: patch

### Fixed

- **`pyproject.toml`'s `version` is the changelog's newest version again.**
  It is what ComfyUI's registry and manager read as the pack's version, and
  it had sat at 0.13.0 since f66fa9d1 while the changelog went past 0.213.
  `bench/build_changelog.py` now writes it on every build
  (`sync_pyproject`), so it cannot drift again.

### Changed

- **`docs/wiki/masked_v2v.md` says what each thing called a version is**:
  the pack's version (the changelog's newest heading, mirrored in
  `pyproject.toml`); `MASK_VERSION`, one integer per node class that goes
  into the kept mask's key and is bumped only when that node would make a
  different mask from the same inputs, so the numbers on different nodes
  are unrelated and never compared, and it leaves with the kept mask;
  `TABLE_VERSION`, the shot table's file format. A node's identity is its
  `node_id` in the manifest, which has no number. Owner, 2026-10-06:
  "might need to make sure what mask version vs node versions mean in this
  repo".
