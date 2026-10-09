bump: minor

### Added

- **`MiniMaxH3MaskedSource` takes an `edge`: `whole tokens` (the default, the
  node as it was) or `latent cells`.** With `latent cells` the region's mask
  reaches the sampler per latent cell, half a token's side. Core labels a
  token by the most regenerated of its cells and puts the source back cell by
  cell (`comfy/model_base.py`, `MiniMaxH3.scale_latent_inpaint`), so a kept
  cell inside a regenerated token is the source's in the result; vllm-omni's
  mask editing does the same. It is a trial aimed at one measured limit:
  rounding out to whole tokens keeps the region near twice a small subject's
  own area with no margin at all (the masking board, finding mhi-03). The
  input is optional and appended last, so a saved graph is unchanged, and it
  stays out of the kept mask's key. No graph sets it, nothing has rendered
  with it, and it is not a default until the owner has judged a pair
  (2026-10-09). `bench/check_video_mask.py` item 15: no masked pixel is lost,
  the cells are inside the default's tokens and fewer on a small mask, core
  labels the same rows from either mask, the window reads the choice from
  the record, and the node refuses an unknown one. The check that `others`
  is the node's last input now holds its place before `edge`.
  `docs/wiki/masked_v2v.md` has the line.
