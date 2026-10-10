bump: minor

### Added

- `bench/capture_masked_run.py mouth` reads a mouth against the voice's
  level (`with_the_voice`: the agreement of the opening with the vocal
  stem's level per frame, at no shift and at the shift that fits best), for
  the reference and for every arm, and takes `--frames FIRST-LAST` to score
  one stretch of source frames. It answers whether a subject who does not
  sing moves the mouth with the song.

### Fixed

- `bench/capture_masked_run.py changed` stopped with no floor when a subject
  lay wholly inside its region and the capture had no class map, because the
  floor wanted labelled pixels outside the region. With nothing labelled out
  there the floor is now the picture's own difference outside the region; a
  region over the whole frame still has none.
  `bench/check_capture_masked_run.py` pins both, and the voice reading.
