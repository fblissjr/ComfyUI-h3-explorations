bump: patch

### Changed

- **`docs/wiki/sam3_prompting.md` flags that Meta's image code appears to
  apply the presence score twice**: inside the model where the class score
  is written, and again in the processor before its threshold. Read at both
  sites and not run. Today's records computed Meta's rule as one
  multiplication; the page says what that leaves open.
