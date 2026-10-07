bump: patch

### Added

- **`bench/sam3_precision_arms.py`, and the record of its first run**
  (`bench/results/2026-10-07_sam3_precision_arms.md` and its json): SAM 3.1
  through ComfyUI's own detect and track nodes at float16, bf16 and true
  float32 on the same frames and the same seeds, with a repeated arm as the
  run-to-run floor, a one-level nudge of the input as the sensitivity floor,
  a larger nudge as the control the harness must see, a prediction written
  before the comparison, and stacked mask videos with a row for where the
  arms disagree. Written and run by an independent session working from the
  primary sources only. The record's verdict is that precision is not a
  lever for the tracker on the stretches tried, so the lane keeps ComfyUI's
  default; it also carries that session's own measurement of three places
  where ComfyUI's SAM 3.1 departs from Meta's code. No node, graph or
  default changes.
