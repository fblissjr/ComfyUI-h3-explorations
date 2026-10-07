bump: patch

### Fixed

- **The default-pick decision no longer cites a record that does not hold
  its evidence.** The 0.220.0 entry, `docs/wiki/decisions.md` and the
  comment beside the default in `subject_track.py` said `largest` took "a
  figure at the frame's edge" on "the lane's crowd clip" for all three
  windows, and cited `bench/results/2026-10-07_subject_track_under_nudge.md`,
  every arm of which was picked with `largest`. The windows are two
  stretches of the crowd clip and one of a second clip; `largest` is low in
  the frame on the first and touches the frame's edges on the second; and
  the evidence is not yet in a tracked record. `decisions.md` now says so
  and keeps what it used to claim. Found by a fresh session reading the
  lane. Comment and prose only: no behaviour changes.
