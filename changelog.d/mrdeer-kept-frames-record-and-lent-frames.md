bump: minor

### Added

- `bench/capture_masked_run.py preflight`: `region_on_a_frame_with_no_mask`
  (`flag_lent`, iffy), for a rendered run or a plan read from the node's
  files: frames that have no mask of their own and are regenerated all the
  same, because their latent step holds frames that have one. Found on two
  face-only renders whose part had been emptied on turned-away frames: six
  such frames, exactly the ones the delivery's record showed something laid
  on, one with a face drawn where the source shows none. A plan worked out
  from the masks cannot show it. Pinned in
  `bench/check_capture_masked_run.py`.
- `bench/results/2026-10-10_kept_frames_against_the_mesh`: a record, numbers
  in a json, of a pose model's reading of every render of one head turn
  against the source and the body mesh. Every render that kept frames of
  the subject turned away missed the turn alike, at either mesh size and
  with either Sol sink; the one load with none followed the mesh; and a
  shorter load with none did not. One stretch, one seed a render; its first
  lines say so. `docs/wiki/state_signals.md` gains a dated line pointing at
  it.
