bump: minor

### Fixed

- **The Subject Track's mask no longer carries stray specks far from the
  subject** (`subject_tracks.drop_specks`, called on the assembled mask
  before the tiles, the shot table and the output). The tracker's mask can
  hold a few pixels on somebody else on a frame. Everything after it widens
  what it is given: the part node shows the model the mask widened, the
  Masked Source grows the part by its margin, rounds it out to whole tokens
  and holds it for a latent step's frames. So a speck of a few pixels became
  a block of the crowd regenerated for several frames, nowhere near the
  subject (found in a render's region overlay on 2026-10-08; the masking
  board, finding `mhi-08`). Per frame the largest connected piece is the
  subject, and another piece is dropped only when it is both under
  `SPECK_SHARE` of that piece's area and farther than `SPECK_REACH` x the
  square root of that area from its box. Either test alone keeps a piece,
  because a subject is often in several: an arm past the head of the person
  in front is kept by its size however far it is, a sliver of sleeve by
  being near however small. The report says how many pixels went on which
  frames. `MASK_VERSION` on the node is 11. The mask is read a frame at a
  time and the node lets the function work in place, so a clip's mask is not
  copied on every run (found in review).
- `bench/check_subject_tracks.py` holds the rule (a far speck goes; a large
  detached piece, a small near piece, a frame's only piece and an empty
  frame stay; each test alone removes nothing; the reach follows the
  subject's size; a speck straight above goes; a diagonal tail off the
  subject's corner stays theirs; the input is written to only when asked),
  and `bench/check_subject_track.py` pins the node's call.

### Not done

- The two thresholds are measured on one load's mask. Not run through the
  node on the server: the function was run on that load's saved mask.
