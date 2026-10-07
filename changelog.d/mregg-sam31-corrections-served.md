bump: minor

### Added

- **`MiniMaxH3SAM31Corrections` is served.** The node of 0.216.4 (the
  model and text encoder of any checkpoint loader in, clones carrying the
  image-range and text-activation corrections out, with a report) is in
  the pack's node list, appended at the end, after its acceptance on a
  server's queue:
  `bench/results/2026-10-07_sam31_corrections_on_a_queue.md`. A stock pair
  and its corrected clone sharing one model each got their own first
  layer in both orders; the corrections held after a real H3 sample
  evicted SAM from the card and after all models were freed. Not covered:
  SAM called while the video model holds most of the card. No shipped
  graph wires the node yet.
- **`bench/sam31_corrections_queue_test.py` and
  `bench/sam31_probe_node.py`**: the acceptance as a tool (a scratch
  server serving a scratch copy of the pack, so nothing unaccepted is on
  the main server) and the test-only probe it registers there. The probe
  is in no node list of the pack.
- **Today's Subject Track under a one-level change of the input.**
  `bench/results/2026-10-07_subject_track_under_nudge.md`: the automatic
  pass alone, without typed corrections. Its decisions between shots are
  the same as fed and nudged on a many-cut stretch and on a calm crowd
  stretch; on a hard crowd stretch the nudge makes it skip one of four
  regains and never take another person, and moves a regain's lead by
  about the lead the node requires.
- **What phrase to give SAM 3.1**,
  `docs/research/masking/2026-10-07_mryolk_phrases.md`: from Meta's
  examples, eval configs and dataset cards.

### Changed

- **`bench/sam3_parity_ladder.py` measures its floors and has a tracker
  rung that can tell a difference from noise.** The trunk and detector
  rungs carry FLOOR rows (ComfyUI against itself at bf16, and with the
  frame moved one level of 255) and a TF32 self-test; the detector rung
  counts ComfyUI's detection rule and Meta's over the same queries on
  several frames; a new `there-and-back` rung plays a window forward and
  back, repeats every arm nudged, corrects the range at the first layer
  as the node does, and counts only masks that pass a shape test.
- **`bench/results/2026-10-07_sam3_core_against_meta.md` restated on those
  floors.** The trunk is the same to a measured bf16 floor. The range
  fault moves the trunk's features by one and a half to two times what a
  one-level change of the frame does, and the resize by less than that
  change: the first version called the resize "about four times" the
  equal-tensor difference, which was true and overstated its weight. From
  the same seeds, with a floor, Meta's tracker keeps more plausible masks
  than ComfyUI's at sixteen subjects and holds a second group where
  ComfyUI's does not, which the first version could not establish. A
  subject who "ends on their own seed" was shown not to mean a subject
  followed.
