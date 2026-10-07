bump: patch

### Fixed

- **`docs/wiki/sam3_prompting.md` says which of Meta's paths applies the
  presence score twice.** The entry before this one left it open whether
  today's records had under-applied Meta's rule. They had not: the second
  multiplication is in Meta's single-image processor, the video pipeline
  thresholds the joint score once, and the records compare with the video
  pipeline. Read at the three sites by two sessions; the image processor
  itself was not run.
