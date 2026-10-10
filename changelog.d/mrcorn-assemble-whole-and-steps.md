bump: minor

### Added

- **`bench/assemble_delivery.py`: `restore=<subject>:whole`.** The piece is
  laid nowhere inside that subject's tracked mask, with nothing taken out
  for the piece's own subject: for a pass that has no business inside
  another person's mask whoever is in front. The plain form still leaves a
  pixel both masks claim to the piece. Asked for by the session leading the
  masked lane after looking at the both-claimed pixels on eight frames:
  which person is in front differs from frame to frame.

### Changed

- **The flag for a render's changed area is a step, not a distance from a
  median.** `piece_changes_far_more_than_it_usually_does` compared every
  frame with one median for the whole piece and lit a hundred and ten
  frames of a render whose framing gets closer partway. It is
  `changed_area_steps` now: the area before and after a frame, over
  `STEP_FRAMES` either side, differing by `JUMP`, named at the frame where
  the area moves most. On that render it names five frames; on the render
  that redrew the wrong person past a cut it names the cut.
- **What a restore gave back is not counted as changed away from the
  subject.** `piece_changes_away_from_its_subject` counted what a piece
  changed, so it kept lighting frames whose pixels the row had given back.
  It counts what is laid.

`bench/check_assemble_delivery.py` covers all three.
