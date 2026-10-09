bump: patch

### Added

- **`docs/research/masking/2026-10-09_mrfetch.md`: three questions from the
  whole-frame video-to-video lane, answered from core's code and this
  pack's.** What core's first sigmas are for a late start and what it mixes
  there, with why a small share of the source still holds the coarse
  picture; why the last latent frame of a window can let go of the source's
  look when the reference video reaches the text encoder only (a candidate
  cause in how the encoder's samples are paired, with the test that would
  settle it); and what one fraction as the mask over the whole frame does
  on each step in core. A reading: nothing was rendered, and each section
  says what would settle it. Listed in the folder's README and pointed at
  from `docs/wiki/references.md`.
