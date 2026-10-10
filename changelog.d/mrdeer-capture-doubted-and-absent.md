bump: patch

### Changed

- `bench/capture_masked_run.py preflight`: a shot the tracker calls absent with somebody on screen is a shot to look at whatever it scored, unless the whole shot lies inside frames the caller gave with `--not-in`. The score could not settle it: on one clip the subject was on screen in two such shots, at 0.052 and 0.133 under the match line, and truly absent in three at 0.20 to 0.37 under. `NEAR_UNDER` now only words the reason.
- A finished capture folder is not rebuilt in place without `--overwrite`: a queued render may be loading its mask videos.

### Added

- With a class map, each doubted part frame is named by what lies under the shape a fill would put there: the part itself, the subject's own hair, another of their classes, another subject, or nothing labelled. `subjects/<label>/doubted_frames.json` has the shares per frame and whether `parts_held` filled it, and the preflight raises one flag a kind; only "nothing labelled" is a frame a fill is likely right for. Provenance, said beside the constant: sixteen frames of one preview read by eye, fourteen named rightly.
