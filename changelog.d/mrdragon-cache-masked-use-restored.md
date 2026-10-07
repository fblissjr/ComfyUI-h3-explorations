bump: minor

### Changed

- **The frozen video cache's masked use is back, with its cached step on the
  attention backend the graph installs.** The owner restored
  `frozen_video_cache.py`, `bench/check_frozen_video_cache.py`, the
  generator's `masked_cache`, `h3_config.FROZEN_VIDEO_CACHE`'s `halo` and the
  node manifest as they stood before the retirement of 0.214.0, after
  watching the corrected pair
  (`bench/results/2026-10-07_frozen_cache_masked_window_fixed.md`). One
  change on top: the cached block no longer removes
  `optimized_attention_override` from the options it calls attention with.
  That override is where core's Model Attention Backend node puts the
  kitchen kernel, and with it gone the call ran on torch's kernel
  (`bench/results/2026-10-07_frozen_cache_rectangle_kernel.md`). Sol-Attn,
  where it is wired, declines the call and hands it to the backend it was
  installed on. `MiniMaxH3FrozenVideoCache` has its `halo` input again, the
  probe graph `h3_probe_v2v_masked_song_ref2va_motion_cache_api.json` is
  generated again, and the four audio-refine probe graphs carry `halo` at
  the node's default. No shipped or daily graph carries the cache.
- **The audio-refine pass's cached steps run on the graph's backend too**,
  through the same line. Not timed or graded again since; its records
  (`bench/results/2026-09-25_frozen_cache_s1.md`,
  `bench/results/2026-10-06_frozen_cache_stage_split.md`) were measured on
  torch's kernel.

### Added

- **`check_frozen_video_cache.py` item 17**: a counting backend installed
  the way core installs one, alone and under the real Sol-Attn override,
  must see a cached step's call. Red against the module as retired.

### Fixed

- **The fixed-pair record described its own change wrongly.** It said the
  change walked the override chain to the backend under Sol-Attn; on Sol's
  real override it walked nowhere and left the override in place. The
  record carries a dated correction, the times are unaffected, and
  `docs/wiki/decisions.md` has the line. `docs/wiki/masked_v2v.md`,
  `docs/wiki/next_steps.md` and `archive/frozen_cache_masked/README.md` no
  longer say the masked use is retired.
