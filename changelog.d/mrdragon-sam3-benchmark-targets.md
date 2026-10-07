bump: patch

### Added

- **`bench/sam3_dataset_phrases.py targets`, and its record**
  (`bench/results/2026-10-07_sam3_benchmark_targets.md` and its json of
  aggregates): for the words this lane asks SAM for and for small things as
  a class, how many instances a pair holds in Meta's benchmarks, and in the
  video benchmark how often an object leaves and returns and for how long.
  By the same independent session, with the same handling of the gated
  datasets: counts only.

### Changed

- **`docs/wiki/sam3_prompting.md`**: a regain should expect short dropouts as
  normal, with a pointer to the new record; and the open question about
  descriptive phrases on the CPU and on the card now says what was
  separated the same day (it is not precision; a detection score measured in
  a CPU process is evidence about the CPU path only).
