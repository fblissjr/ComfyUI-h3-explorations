bump: minor

### Added

- `bench/capture_masked_run.py look`: for a whole-subject render, per frame, where it sits between the source (0) and a render of the same subject that held the new one (1), read as the mean grey level over the top of the subject's mask. Written for a fault no mask flag predicts: one window of a long render drew a look-alike of the original and the next the new subject, with identical regions. It reads only the frames the render regenerated, writes `look__<render>.json` in the capture folder, and refuses when the held render is no lighter or darker than the source over that area.
- A preflight rule, `kept_pixels_inside_the_part`: a run that wires the Masked Source's `keep` and leaves cells of the subject's own part as the original's pixels. Provenance, written beside the rule: one pair of renders in which a `keep` on an earring brought the original's face back.
- `diagnose` prints the flags `bench/assemble_delivery.py` wrote into the capture folder beside the preflight's own.
