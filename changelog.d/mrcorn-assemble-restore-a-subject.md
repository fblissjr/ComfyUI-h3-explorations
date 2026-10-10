bump: minor

### Added

- **`bench/assemble_delivery.py`: `restore=<subject>` with no class gives
  back the whole of another subject.** Wherever that subject's tracked mask
  is and the piece's own subject's is not (the piece's subject is its run's,
  read from the capture folder), the piece is not laid and the source's own
  pixels show. It is for a pass on one person whose region took in part of
  another: the rule for pixels two pieces both changed only settles those,
  and what one pass alone changed of the other person stayed in the file.
  Where both masks claim a pixel the piece keeps it. Measured 2026-10-10 on
  one whole-person render of one clip, a stretch where the two people are
  close: the other person's tracked pixels more than twelve levels from the
  source fell to about a seventh of what they were, and to about a fortieth
  outside the piece's own subject's mask. Asked for by the session leading
  the masked lane. `bench/check_assemble_delivery.py` has a tenth case.
- The check record lists, per frame, the edge pixels beside a restore
  (`restored_beside_a_large_change_px_per_frame`), so a piece with no run in
  a capture still says where its joins are longest.

- The check record now holds the table's rows with their restores, the
  capture folders given, when it ran, and a sentence from `--note`, so a
  file rebuilt in place says that it was and why.

### Changed

- The flag `restore_has_no_class_map` is `restore_has_no_mask`: it now also
  covers a subject with no tracked mask on a frame.
