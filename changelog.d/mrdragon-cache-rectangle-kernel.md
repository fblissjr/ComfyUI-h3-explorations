bump: patch

### Fixed

- **The frozen-row cache's first masked run is corrected: its attention ran
  on torch's kernel, not kitchen's.** The cached block removes the attention
  override that carries the kitchen int8 backend, and the checkpoint names no
  backend of its own, so the call fell to core's default. Timed alone on the
  card, torch's kernel reproduces the attention stage that run measured, and
  the kitchen kernel runs the same rectangle at the square's rate per pair
  (`bench/results/2026-10-07_frozen_cache_rectangle_kernel.md`,
  `bench/bench_rect_attention.py`). So "saves no time" described a cache on
  the wrong kernel, and the cost model was never tested. No code path is
  changed and the retirement stands until the owner reopens it: dated notes
  in the 2026-10-06 record, `archive/frozen_cache_masked/README.md`,
  `docs/wiki/next_steps.md`, `docs/wiki/decisions.md` and the docstring of
  `frozen_video_cache._dense_options`, which the audio-refine pass still
  runs through.
- `INDEX.md` lists the four files the retirement put under
  `archive/frozen_cache_masked/`; it was built before they were tracked and
  `bench/check_doc_inventory.py` was red on the committed tree.
