bump: patch

### Added

- `bench/results/2026-10-10_face_part_kept_or_emptied_on_a_turn`: a record,
  with its numbers in a json, of one face-only shot rendered twice, the part
  mask kept on four frames where the head is turned away in one render and
  emptied by `grade_holds` in the other. Kept, the render redrew the hair
  and drew no face; emptied, the frames are the source's. It also carries
  the frames of the shot's one run with the source's mouth open on unvoiced
  frames. One shot, one seed; its first lines say so.
- Three dated lines in `docs/wiki/state_signals.md` under entries that named
  a session message as their source: the face part's rule is code and its
  harm was overstated; head lift and a hand at the face are columns of the
  capture; the open-mouth runs are found by `mouth`.
