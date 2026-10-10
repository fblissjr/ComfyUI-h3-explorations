bump: patch

### Fixed

- The two records of 0.265.2 are reworded to name subjects by label and
  areas by where they lie, and nothing else: a tracked record does not say
  what a clip shows. Their numbers are unchanged; two json keys are renamed
  to match and the per-joint list becomes a count. A made-up sentence in
  `bench/check_capture_masked_run.py`'s text case loses a setting it did
  not need.
