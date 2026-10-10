bump: minor

### Added

- `bench/capture_masked_run.py files` writes `owners.npz` when a capture has more than one subject: whose each pixel is (`owner`, an index into `labels`, 255 for nobody, 254 for contested). A pixel one track claims is that subject's; a pixel several claim is the one's whose class map names it something, when exactly one does; named by more than one or by none it is contested and not guessed. `frames.csv` gains the pixels claimed by more than one track and the contested ones per frame, and the preflight raises `contested_by_class_too` on frames with a contested patch. On the two-subject span the rule came from it settles 13,793 of 14,728 pixels both tracks claim, which is the count a session's one-off had found.
