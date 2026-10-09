bump: patch

### Fixed

- **The record of the three masked renders states the frames it really
  rendered and audits what each model was given**
  (`bench/results/2026-10-09_masked_text_and_edge_one_window.md` and its
  json): measured, the window is one frame earlier than the record said;
  the frame rate, the single resize and the absence of joins are set down;
  the owner's playback of the other two renders is added, with the missing
  half of the cap in the text render and a reading of it that is marked as
  not shown; and the track is said to have held where the redraw did not.
