bump: minor

### Added

- **`bench/assemble_delivery.py`: the check record says what the file did
  to each subject, and `--compare` prints two records side by side.** With
  captures given, the delivered file is read back at the canvas and set
  against the fitted source on the frames a piece covers: for every subject
  a capture knows, the pixels of its tracked mask (and of what it owns)
  more than `OFF` levels from the source, in all and by class of its class
  map, inside its track and outside it; the pixels more than one track
  claims, how many an owner map settled and how many it left contested; and
  the floor, the same share on pixels no track claims and no piece changed.
  A pass on one person should leave the others at the floor. These were
  one-off measurements asked for three times in a day ("the other person's
  pixels off the source, by class, before and after"); the same stretch
  built with and without a restore, then `--compare`, is that answer.
- A thirteenth case in `bench/check_assemble_delivery.py`, with the reading
  pinned on a picture that runs from black to white.

### Fixed

- Found while building it, never committed: read through ffmpeg as `gray`,
  a tv-range file is widened to full range, bright and dark pixels move by
  up to sixteen levels, and a quarter of untouched footage read as off the
  source. A mid-toned test picture did not show it. The file is read as
  yuv420p (`luma_at_canvas`). Hand measurements of the same day that read
  both sides as `gray` compared like with like and stand as comparisons;
  their twelve levels were about ten of the file's own.
