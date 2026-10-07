bump: minor

### Added

- **`MiniMaxH3MaskedSource` takes an optional `keep` mask**: one mask per
  source frame of what must stay the original even inside the region
  (something the subject holds, a person standing close, anything passing in
  front). It is taken out of the token mask after `grow_pixels`, in whole
  tokens, so the margin cannot run back over it; the model is given those
  tokens clean and the composite shows the source there. Refused together
  with `paint_out` or a softened start. Unwired, a render is unchanged, and
  no generated graph wires it. Asked for by the owner on 2026-10-07 for a
  prop the original holds, which is otherwise under the noise and comes
  back as a guess. Not yet rendered.
- **`check_video_mask.py`, the keep cases**: unwired changes no token;
  wired, every token it touches is kept and no other changes, with the same
  keep taken out before the grow as the control that must not keep them;
  one kept frame keeps exactly its own latent step; the refusals. Seen red
  with the keep ignored and with it applied before the grow.
