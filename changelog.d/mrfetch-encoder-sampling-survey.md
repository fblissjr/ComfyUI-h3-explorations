bump: patch

### Changed

- `docs/research/masking/2026-10-09_mrfetch.md`, section 2: how a reference
  video is sampled for the text encoder, read in sglang, FastVideo,
  vllm-omni, LightX2V and core. All five use one rule, the odd count padded
  by repeating the last sample included, so the repeated block the section
  names as a candidate is the release's own presentation and not a fault in
  this pack's sampling. Two results mrship reported that day are added,
  marked as reported and not judged.
