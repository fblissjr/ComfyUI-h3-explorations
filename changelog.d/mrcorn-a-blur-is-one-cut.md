bump: patch

### Fixed

- **`bench/assemble_delivery.py`: a blur is one cut, not a shot a frame.**
  The shots proof cut the span wherever the fitted source moved over `CUT`
  from the frame before. On its first real file five consecutive frames of a
  blur inside one shot were five cuts, three "shots" one frame long read
  "tracked, nothing laid" where the face part is emptied on purpose, and the
  build failed. Consecutive frames over the line are now one cut at the
  first of them and the run is listed in the record; a capture's shot table
  is compared with these cuts on the frames it covers, and a disagreement is
  reported (the tracker's cut finder is the better one). One more case in
  the shots case of `bench/check_assemble_delivery.py`.
- `bench/results/2026-10-10_delivery_files_and_their_records.md`: the dated
  lines for that, and for a new opening file built two ways and compared by
  the join flag.
