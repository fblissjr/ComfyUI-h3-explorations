bump: patch

### Fixed

- **The record of the three masked renders says what its control is not**
  (`bench/results/2026-10-09_masked_text_and_edge_one_window.md`, and the
  json's `changed_from_base`): the 2026-10-08 final it was cut from carries
  a hand-written text per window that says which way the lead faces, and the
  control drops those for the node's own text. The record did not say so, and
  read as if the final had rendered with the node's text. It also gains the
  owner's one playback note on the control.
