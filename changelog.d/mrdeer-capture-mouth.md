bump: minor

### Added

- `bench/capture_masked_run.py mouth`: a render's mouth against the source's, frame by frame. `mouth_openings` reads how open a mouth is from a mouth mask or from a class map's lips, teeth and tongue (the largest piece's two axes over the face's size); `score_mouth` gives the agreement of two series with the render shifted a few frames either way, the same for the source against itself as the control, and with a voice table whether either mouth rests where the voice does. It replaces two session scripts that the lip-sync results of 2026-10-10 rested on and reproduces their figures on the pass it was checked against; from class maps alone, with no separate mouth preview, it reads the same to within 0.02. The first form of the measure and why it was thrown out are in the function's docstring.
