bump: patch

### Fixed

- **A place in a clip is written and read back by one rule.**
  `subject_tracks.py`: a time that rounded up to the minute was printed
  as `0:60.0`, which the reader of the same text refuses; and a load
  starting on a half frame could put a typed place one frame off, since
  the two directions rounded separately. Both go through one helper now,
  and `bench/check_subject_tracks.py`'s round-trip case covers both.
  Found by a review of the tracker's pieces; nothing served uses them yet.

### Added

- **`subject_tracks.take_back(..., leader_must_be_near=True)`**: a clear
  leader by likeness is refused when it is not near the subject's last
  place or not of a like size, with its distance in the report. Off by
  default until it is measured. Its check case has the control: without
  the option the far leader is taken.

### Changed

- **The Subject Track bench tools name who they follow.**
  `bench/subject_track_under_nudge.py` takes a required `--pick` and
  `bench/subject_alone_or_in_a_group.py` a required `--subject`, written
  into each json, and neither reads the pick from `h3_config`, whose
  default changed in 0.220.0. The two records' reproduce lines say
  `largest`, which is what their runs followed.
- **`bench/results/2026-10-07_sam3_core_against_meta.md` says which device
  it ran on.** Every model rung ran on the card; a CPU process is a
  different path for ComfyUI's SAM 3.1 and the record says nothing about
  it. `sam31_corrections.py`'s docstring marks its one CPU probe the same
  way and points at the node's acceptance record (comments only).
