bump: patch

### Added

- **ComfyUI's SAM 3.1 compared with Meta's code stage by stage, as a tool
  and the record of its first run.** `bench/sam3_parity_ladder.py` runs
  core's port beside the copy under `meta_sam3/` (imported there as a
  reference; no node depends on it) on equal inputs: tokens, the two
  weights files, the text encoder, the image range and trunk, the detector's
  raw outputs, and both trackers started from the same masks.
  `bench/results/2026-10-07_sam3_core_against_meta.md` is what it found and
  what it could not: two departures before the model (the range of the
  image core's nodes hand over, and the text encoder's activation), the
  rest the same to the size of float32 against bf16, and no difference
  between the trackers beyond sixteen people that survives a small change
  of the input. Nothing in the pack's nodes changes.
