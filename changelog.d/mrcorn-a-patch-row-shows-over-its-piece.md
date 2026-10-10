bump: minor

### Added

- **`bench/assemble_delivery.py` refuses to build over a locked file.** A
  folder's `LOCKED.md` lists the files the owner has accepted (the name in
  backticks on a list line, then `md5 <sum>`). A build whose `--out` is one
  of them stops before anything is read and names a free name beside it;
  `--check-only` still reads a locked file, writes only its record, and
  fails if the file is no longer the one that was locked. The join of
  `bench/patch_render_window.py` goes through the same command and so has
  the same refusal. One case in `bench/check_assemble_delivery.py`.

### Fixed

- **`bench/assemble_delivery.py` laid a patch under the render it patches
  when a capture was given.** Where two rows changed a pixel it went to the
  row whose subject held it; a render had a run in the capture and so a
  subject, a patch of it had neither, and the earlier row took the patched
  pixels back inside its subject's mask. Every proof passed. Now the rows
  whose subject holds a pixel are the ones that can take it and the latest
  of them does; a row whose subject is not known can take whatever it
  changed; and a row can say whose it is with `subject=<label>`, which also
  gets it the "away from its subject" flag.
- **Two proofs that would have failed that file.** `rows_shown`: every row
  must show on the pixels it was meant to (what it changed, less what its
  restores give back, what a later row takes and what another known
  subject's earlier row holds), or the build fails naming the row. And the
  tie between the file and the frames the table makes is local: no
  `BLOCK`-pixel square of a decoded frame may sit over `APART` levels from
  the frame fed. It was a whole-frame nearest-neighbour test, which a wrong
  region of a frame does not move: the faulty file passed a re-check under
  the corrected rule until this was added. The figures that set `APART` are
  beside the constant.
- `bench/check_assemble_delivery.py`: a patch row with a capture (it fails
  under the old rule), and the control for the local proof (a file read
  against the table without its patch row fails on the patched frames, with
  every frame in order and no other proof failing).

### Changed

- `bench/patch_render_window.py`: its docstring says `build` is the
  2026-10-08 route, is not the one in use and has no check on purpose; it
  goes when the tracked job builder lands. `join` stays and is covered.
