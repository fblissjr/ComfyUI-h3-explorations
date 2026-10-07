bump: patch

### Added

- **A record of the reference still given to the text encoder alone in a
  masked render** (`bench/results/2026-10-07_masked_still_encoder_only.md`
  and its json): `MiniMaxH3AppendRefImage.use_vae` off on two windows of one
  clip, with the owner's verdicts on the stacked renders. Worse in both
  places, at the 512 view and at the large shared view; the masked graphs
  keep giving the still to both models. The sampling saved at the 512 view
  is recorded and not taken.

### Changed

- A comment in `sapiens2_parts.py` beside `HELD_ANCHORS` no longer names an
  object; a missing blank line in `docs/wiki/sam3_prompting.md`. No
  behaviour changes.
