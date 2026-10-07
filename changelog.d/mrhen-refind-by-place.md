bump: patch

### Added

- **A re-find judged by where the subject last was**
  (`subject_tracks.take_back_by_place`), beside `take_back`, which asks
  likeness first. A candidate has to stand where the subject stood: its box
  overlapping the subject's box on the last frame a gallery may use
  (`gallery_span`). Likeness only says who may be asked, in one of two ways
  the caller names and the result reports: over a line, or first by a
  margin. Nobody is taken when none stands there, when more than one who may
  be asked does, or when there is no last place. The count asked of the
  detector is an argument, and the result says whether as many came back as
  were asked for, since place only helps when the subject is among them.
  Place is not identity: whoever stands there is taken, and
  `bench/check_subject_tracks.py` asserts that as a known limit. The overlap
  that counts as "there" is an argument, its default read off one stretch.
- **The recorded looks through that function**
  (`bench/subject_regain_looks.py replay`, no model): on the takes a strip
  labels it takes the one that stands at the place and refuses the others
  today's rule makes, in both runs, and anchored on the frame beside the
  jump it takes nobody at all.
  `bench/results/2026-10-07_subject_regain_looks.md` has the tables in a
  dated section. The same tool now puts each call the node makes to the
  tracker through `unbroken` and `gallery_span` as it comes back; that part
  has not run yet. Imported by no node.
- **A second thing the step test's default line does not catch, named in
  the check**: a mask that leaves its figure in steps each a little over the
  line and then goes empty (`a_slow_move_off_passes_the_default`, with the
  arm of `bench/results/2026-10-07_subject_alone_or_in_a_group.json` it was
  seen in). The line is not moved on one arm of one figure; a caller's
  higher line cuts it.
