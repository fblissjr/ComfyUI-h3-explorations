bump: patch

### Fixed

- **The SAM 3.1 Corrections node could skip the text-encoder correction and
  say nothing needed correcting.** `sam31_corrections.py` chose which text
  MLPs to patch by reading each one's live activation. The encoder is one
  module shared by every clone, and while a corrected clip is loaded its
  patch is what the module shows; so when the node ran again on the stock
  encoder while an earlier corrected clip was still loaded, it attached no
  activation patch and its report said "already exact", while the new clip
  would run ComfyUI's shipped activation. Under-correcting, silently, with
  a false report, and only in that state; the image-range correction was
  not affected and no shipped graph wires the node. Found by a code review
  the day the node was served (0.218.0). It now reads what an MLP was
  built with from the patchers' shared backup first. The acceptance on a
  queue had not taken that path
  (`bench/results/2026-10-07_sam31_corrections_on_a_queue.md`, the dated
  note); `bench/sam31_corrections_queue_test.py` now does, and
  `bench/check_sam31_corrections.py` has a case that is red on the node as
  first served.
- **A clone of the corrected model forgot that frames had arrived already
  mapped**, and would have mapped them again. Every clone's range patch now
  shares what the patch has seen.

### Changed

- **`subject_tracks.py`'s watch says what it cannot see.** It doubts a
  track no detection overlaps; a track that swapped onto a detected
  neighbour is not doubted, and its docstring claimed otherwise. The check
  now asserts that limit. A doubt reaches back to the look before the one
  that disagreed. New: `take_back`, the rule for taking a subject back
  after a loss (a clear leader by likeness; a call too close by likeness
  goes to the candidate where the subject last was, if near, like-sized
  and clearly nearer; nobody otherwise).
- **`bench/sam3_parity_ladder.py`'s there-and-back rung counts seeds that
  are a second mask of the same subject** and can leave them out: core's
  detect node removes no overlapping detections, and core's tracker lets
  one of two overlapping masks lose its memory where Meta's tracker-only
  path does not, so such seeds could pass for subjects lost.
  `bench/subject_track_under_nudge.py` records every look after a loss,
  taken or not.
- **The comparison record's first detector table** is the same frame run
  again in the tool's present form (the same numbers, with a floor and
  both detection rules), and **the node's record** has its run on the main
  server.
