bump: patch

### Added

- **A record of the frozen video cache on a masked window with its cached
  step on the kitchen kernel**
  (`bench/results/2026-10-07_frozen_cache_masked_window_fixed.md` and its
  json): the 2026-10-06 pair repeated on a scratch server with one function
  changed, the cached arm's sampling against the stock arm's, the build and
  step rates, and what was not looked at. The change itself is kept as
  `archive/frozen_cache_masked/dense_options_fix_2026-10-07.diff`. No node
  code changes: the masked use stays retired until the owner says
  otherwise. `archive/frozen_cache_masked/README.md` and
  `docs/wiki/next_steps.md` point at the record.
