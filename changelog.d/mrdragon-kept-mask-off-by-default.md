bump: minor

### Changed

- **The kept mask on disk is disabled in code.** `video_mask.MASK_REUSE_ENABLED`
  is False, and while it is the Masked Source reads and writes no kept mask
  whatever its `reuse_mask` input says; the input's tooltip says it has no
  effect, and its default is off in the node, in `h3_config.MASKED_SOURCE`
  and in every generated masked graph. The owner's decision, until the lane
  is ready for production: the store cannot tell that the code which makes
  a mask changed. `mask_store.py` stays, and `bench/check_mask_store.py`
  still exercises it with the switch on and has a case for the shipped
  state. A change to how
  a subject is followed that forgets to bump a node's `MASK_VERSION` is
  served the old mask with no sign of it, and the first confirmation render
  of 0.224.0's fix showed the unfixed result for that reason. Every run now
  tracks afresh, which costs the tracker and the part model each time.
- **Reuse of stored rendered windows is disabled in code the same way.**
  `audio_freeze_song.WINDOW_REUSE_ENABLED` is False; `reuse_windows` on the
  song node defaults to off in the node and in every generated song graph,
  and while the switch is False no stored window is reused whatever the
  input says. A stored window's key is built from the queued graph, the
  track, the text and the seed and from nothing of this pack's code, so it
  has the kept mask's blind spot. Windows are still written, and what a
  session keeps in memory (`window_keep.py`) is unaffected. Every window of
  every song render now renders on every run. Both switches wait on one
  thing: a key that changes when the code does.
