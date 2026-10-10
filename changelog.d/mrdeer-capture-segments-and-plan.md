bump: minor

### Added

- `bench/capture_masked_run.py` reads the part node's class map (`classes=` on a `--mask`, by the node's own `class_indices`, from one colour channel) and what a subject holds (`held=`). A segment is `<label>.<class>`, the same id in every table: `subjects/<label>/segments.csv` has its pixels per frame, and `runs/<run>/segments_in_region.csv` has every segment that lies inside a run's region without being the mask it carries, in pixels and cells, the subject's own and anybody else's. Two preflight rules read them: `part_not_visible` (the classes a part is made of are gone or under half their recent size while the subject is there) and `segment_inside_region` (one flag a segment, with its frames and the most cells it took).
- `drop=FIRST-LAST+FIRST-LAST` on a `--mask` empties the held part on source frames where it should take nothing, and a plan with `carried=held` works its region out from the held part.

### Fixed

- A plan's region followed the wrong rule for a kept-out mask: it took the others' pixels out and then counted any cell with a pixel left. It now follows `video_mask.window`: a token the others touch is given up unless the subject's own mask, before any margin, has a pixel in it. Set against one render's own region (447 frames, a whole subject, another kept out) the plan agrees at 0.99 intersection over union at the median; the docstring no longer says the real region is never smaller, since the others' tokens are shared across a latent step's frames too.
