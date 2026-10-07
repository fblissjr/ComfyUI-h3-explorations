bump: patch

### Added

- **One subject in a crowd, followed alone or in a group.**
  `bench/subject_alone_or_in_a_group.py` and
  `bench/results/2026-10-07_subject_alone_or_in_a_group.md`: on a hard
  crowd stretch the subject keeps a plausible mask several times longer
  when seeded with the fifteen most prominent people than when followed
  alone, as today's Subject Track follows them; three neighbours are no
  better than alone; on a calm stretch it makes no difference. Each arm
  was run again with the input moved one level. Not shown: that the later
  frames are still the subject.
- **`docs/checks.md`, a rule in the standard: a node tested on the queue
  has to be made to run, in the state under test.** The executor serves a
  node from its cache when its class and inputs are unchanged, whatever
  its id, and runs it once per prompt, usually before any model is
  loaded; with the two tests that could not fail because of it.

### Changed

- **The Corrections node's queue test reaches the fault it was extended
  for.** `bench/sam31_corrections_queue_test.py`'s fifth step, as first
  written (0.218.1), passed on the faulty node: its second node had the
  same inputs as the first and was served from the cache. It now ends the
  first prompt on a text-only probe and gives the second node a different
  input; the probe (`bench/sam31_probe_node.py`) reports what the shared
  text module showed before its call. On a scratch server the step is red
  on the node as first served and green on the fix; the record has the
  run and the control.
- **`bench/results/2026-10-07_subject_track_under_nudge.md`'s hard stretch
  lists every look after a loss**, taken or not: the two arms look at
  different frames, most refusals are on the lead with the best person
  over the line, and every look returns the sixteen detections asked for.
