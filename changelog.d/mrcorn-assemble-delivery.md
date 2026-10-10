bump: minor

### Added

- **`bench/assemble_delivery.py`: one delivery file from masked renders and
  the untouched original.** A table of renders by source frame range becomes
  one file at the source's own rate, with one encode, the frames no render
  covers taken from the original through the loader's fit and the writer's
  conversion (so a cut between the two is not a step in colour), and the
  source's audio packets copied. Passes over the same frames, each made from
  the original, are merged by what each changed; where two changed the same
  pixel, a capture folder's masks say whose it is before the table's order
  does. It checks its own file by decode (frame count, timestamps, every
  frame nearest the frame fed for its place, no bias in any plane, all four
  colour fields, the audio a run of the source's packets and in place to the
  sample) and exits 1 when that fails. It raises flags from the per-frame
  table of what each render changed: pixels changed where the subject has no
  tracked mask, far from the subject's mask, far more than the render's own
  median, the same pixels as another render, or none at all; with
  `--capture` the table and the flags are written into the capture folder.
  `--soften` is an optional luma blur inside what the renders changed.
  Three things it was built around, each measured 2026-10-10 on one clip: a
  render's kept pixels equal the fitted original to within its codec noise,
  which is what lets a threshold stand in for the mask the node does not
  save (`data/CAPTURE_GAPS.md`); a track cut straight from a source starts
  at the picture's keyframe before the span and hides up to a third of a
  second behind an edit list, so the track is copied alone first and cut one
  packet early; and frames piped to an encode with no matrix label are
  converted on the way to a BT.709 file, which every proof but a bias test
  let through.
- **`bench/check_assemble_delivery.py`**: the tool against a small clip and
  pieces made with the song node's own writer. A piece that kept every pixel
  reads no changed pixel, which fails if the tool's copy of the writer's
  conversion drifts; a painted rectangle is found where it is; three files
  that must fail its check do (a frame dropped and one doubled, audio
  encoded again, a track cut straight from the source); a mask outranks the
  table's order where two pieces share pixels; and each flag sits beside the
  case that must not raise it. `docs/checks.md` has the row, and
  `docs/wiki/masked_v2v.md` points at the tool.
