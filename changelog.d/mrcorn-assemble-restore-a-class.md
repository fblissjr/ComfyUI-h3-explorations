bump: minor

### Added

- **`bench/assemble_delivery.py`: a row can give a class of a subject back
  to the source.** `restore=<subject>.<Class>[+<Class>...]` after a row's
  three fields, the class one of `sapiens2_parts.CLASS_NAMES` or its index,
  read from a capture folder's class map (`classes__<by>.npz`). Wherever
  that class is on the source frame, grown by `RESTORE_GROW` and feathered,
  the piece is not laid, at either size: the source's own pixels show, or an
  earlier row's. It is for a thing inside a region that should not have been
  redrawn, given back at assembly because a kept area changes what the
  sampler draws beside it. The per-frame table gains the pixels given back,
  and two flags: `restored_pixels_beside_a_large_change` (that edge is a
  join between two pictures) and `restore_has_no_class_map`. Asked for by
  the session leading the masked lane, with the measurement behind it.

### Fixed

- **`bench/check_assemble_delivery.py` built a different clip on every
  run.** ffmpeg's `gradients` source draws its colours and its line at
  random whatever `seed` is given, so the check's margins moved between runs
  and one could have gone red on nothing. The picture's colours and line are
  named now, the clip is the same every time, and the first case prints its
  margin against the tool's change threshold. A ninth case covers the
  restore.
