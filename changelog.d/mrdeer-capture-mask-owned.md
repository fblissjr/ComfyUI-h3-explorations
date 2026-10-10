bump: patch

### Added

- `bench/capture_masked_run.py mask --classes-owned`: classes taken only where the capture's owner map gives the subject the pixel, joined with `--classes` (taken as labelled). It is the `keep` mask for a pass on another subject: a face and hair as the class map has them, apparel and hands only where they are that subject's. Against a session's hand-built keep mask for the same rule it differs on no pixel in 447 frames.
