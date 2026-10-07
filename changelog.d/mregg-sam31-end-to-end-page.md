bump: patch

### Added

- **SAM 3.1 in ComfyUI from input to output, as a page and a stage table.**
  `docs/research/masking/2026-10-07_mryolk.md` follows one request through
  core's SAM 3.1 nodes stage by stage, for someone who knows ComfyUI and
  not SAM: what goes in and comes out, which file does it, what Meta's code
  does at the same point, and a verdict for each stage (same by reading,
  same by running, different as a choice or as a departure, not
  established). `2026-10-07_mryolk_stage_table.md` beside it has one row
  per stage with the lines on both sides and the tensor to compare.

### Fixed

- **The record of core's SAM 3.1 against Meta's code says less where it
  said too much.** `bench/results/2026-10-07_sam3_core_against_meta.md`
  called the float32-against-bf16 difference "the floor" without measuring
  it, and now says it is read as the floor and names the control; it said
  "Meta's `io_utils.py` maps to -1..1" where that holds for the
  list-of-images route its rungs drive and for Meta's image processor, and
  the other routes were not compared; it now calls out the resize as a
  departure larger than that floor, and says mapping the range and
  switching the activation correct those two departures and no other.
  `bench/sam3_parity_ladder.py`'s docstring and one comment say the same.
