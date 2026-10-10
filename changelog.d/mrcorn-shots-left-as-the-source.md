bump: minor

### Added

- **`bench/assemble_delivery.py`: a shot where a named subject is the
  source's own no longer passes in silence.** A locked delivery showed the
  source's own picture for a whole shot: no row laid a pixel there, both
  trackers having called their subject absent, so no proof had anything to
  be wrong about, and a viewer found it. The record now has `shots`: the
  span cut by the source's own cuts and, for every subject a row names, how
  many frames of each shot have anything of theirs laid, how many they are
  tracked on, and what the captures say of people there (a shot table's
  count, and the capture's flag `absent_with_people_on_screen`, which the
  assembler had never read). A shot with nothing laid fails the build when
  the subject is tracked on it, or when people are detected who are not
  other named subjects laid or tracked there; otherwise it is listed and
  counted. The table answers it in a new line,
  `source first-last subject=<label> words`, kept in the record; such a line
  over a shot that has something laid, or over part of a shot, fails. Runs
  inside a laid shot where the subject is tracked and nothing is laid are
  listed with the longest. `verdict_line` carries the count, and the frames
  when any fail.
- `bench/check_assemble_delivery.py`: the case (17), and the flags case now
  holds that a piece which changes nothing while its subject is tracked is a
  failure and not only a flag.
- `bench/results/2026-10-10_delivery_files_and_their_records.md`: the dated
  lines for the fault, the proof, and the locked file's record read again.
