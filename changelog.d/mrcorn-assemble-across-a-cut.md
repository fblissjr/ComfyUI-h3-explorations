bump: minor

### Added

- **`bench/assemble_delivery.py` flags a piece that changes pixels just
  across a cut of the source** (`piece_changes_across_a_cut`). A cut is a
  frame where the fitted source moves by more than `CUT` levels from the
  frame before; a run of `SPILL` changed frames or fewer on one side of it,
  joined to changed frames on the other, is a pass whose region ran over the
  cut and redrew the next shot for a moment. It needs no capture and no shot
  table. Measured 2026-10-10: one whole-person render, given as a single
  row, changed about a fifth of the frame on five frames of other shots at
  four cuts, every proof passed, and no flag said so; the area flag named
  four frames that were not them. Rows cut on the source's cuts keep such
  frames out of a delivery, and the docstring and the masked lane's page
  say to cut them so.

### Fixed

- A piece given as several rows raised each of its flags once per row.
  A piece is one piece.

`bench/check_assemble_delivery.py` has an eleventh case, on a second clip
with a hard cut and no audio, which also covers a delivery with no track.
