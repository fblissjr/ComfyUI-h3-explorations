bump: patch

### Added

- `bench/results/2026-10-06_frozen_cache_masked_window.md`, with its data file: the first run of `MiniMaxH3FrozenVideoCache` on a masked video-to-video window. Four arms on one window of the ref2va motion graph and its probe twin: the cache with `verify`, an uncached baseline, the cache, and the baseline again. The cache engages cleanly (one build, every later step cached, no rebuild or decline), moves no kept row, and saves no sampling time: a cached step is slower than the Sol-Attn stock step it replaces and faster only than a dense one. The model that predicted a saving was wrong about the attention call for the live rows against every row, which is most of a cached step. `verify`'s per-step departure from the stock step is in the record; no picture was looked at for it. The stock render is bit-identical across a cold and a warm run, and frame-identical to the same arm rendered that morning.

### Changed

- On that record the owner retired the cache's masked use and kept the node for the audio-refine pass, where it is measured to pay; `docs/wiki/decisions.md` has the line. The retirement is not in this commit: the masked gate, the generator's `masked_cache` and the probe graph are still in the tree, and `docs/wiki/next_steps.md` lists moving them under `archive/` as owed.
- Two comments the record made false are corrected: `frozen_video_cache.LIVE_SHARE_LIMIT` no longer says it sits under break-even, and the generator's note on the probe graph no longer says the cache waits to be timed. No behaviour, graph, input or default changes.
