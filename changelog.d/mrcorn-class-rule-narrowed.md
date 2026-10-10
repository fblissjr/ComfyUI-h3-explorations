bump: patch

### Fixed

- **`bench/assemble_delivery.py`: the class-restore rule of 0.278.1 covers
  the harm it is for and no more.** It kept a class off whatever an owner
  map gave any other subject, so a row giving back its own subject's class
  left its change standing on a third person's pixels. Now a class is kept
  off pixels only when it is ANOTHER subject's class than the row's own and
  the map gives those pixels to the row's own subject; everything else the
  class map names is given back. The record's count is renamed to say so
  (`class_restores_kept_off_the_rows_own_subject`). The owner-map case of
  `bench/check_assemble_delivery.py` holds both sides.
