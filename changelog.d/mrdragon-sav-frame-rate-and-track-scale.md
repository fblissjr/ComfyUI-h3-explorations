bump: patch

### Fixed

- **SA-V's durations were a quarter of what they are.**
  `bench/sam3_dataset_phrases.py::VEVAL_FPS` held 24 frames a second for
  SA-V; its annotation is at six (the file names step by four inside a
  24-frames-a-second folder). The constant is corrected with how it was
  measured, and `bench/results/2026-10-07_sam3_benchmark_targets.md` and
  `2026-10-07_sam3_benchmark_phrases.md` carry dated corrections: SA-V's
  seconds times four, its "within N seconds" shares read as within 4N. The
  tables are as first printed and are not regenerated. Found by a recount of
  the same files.

### Changed

- **`docs/wiki/sam3_prompting.md` says a subject who shrinks a great deal
  through one long shot is outside what Meta's video benchmark covers**,
  with what was counted and that the count is not yet in a tracked record.
