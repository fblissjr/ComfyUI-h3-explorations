bump: minor

### Changed

- **The masked ref2va motion graphs run sixteen steps, the base count.**
  `h3_config.MASKED_MOTION_STEPS` was 12, the lowest count seen to carry the
  source's movement on 2026-10-05 and chosen for speed; it now reads
  `SAMPLING["steps"]`. The owner's decision of 2026-10-07, made before the
  twelve-against-sixteen pair then rendering had been judged: quality first,
  speed later. Six generated graphs change (the two motion graphs, their
  daily copies and the two cache probes). Sampler, scheduler and shifts are
  unchanged. A masked ref2va render takes about a third longer; every one
  before this ran twelve steps.
