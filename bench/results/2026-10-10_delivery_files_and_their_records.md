# Delivery files built with the assembler, and what their records prove (2026-10-10)

lane: masked video to video
verdict: not judged. Every figure here is read from a file or from the frames a table makes; no session played a clip. One source clip, two subjects, one day.

**Read this first.** This is the standing account of `bench/assemble_delivery.py`'s
files and proofs, kept by date. When a proof, a rule or a file's standing
changes, a dated line is ADDED under "Changes, by date"; nothing above it is
rewritten. The tool's docstring is the method; this is what happened with it.
Pieces are named by what they are (a whole-person render, its patch, a face
render) and subjects as the lead and the second subject: the exact file names
and labels are in each file's `<out>.check.json` and in `LOCKED.md`, on the
share beside the files (`Video/mrpop/delivery/`).

## The files

All at the source's own size and rate, the source's audio packets copied,
built from masked renders made at the canvas. "Record" is `<file>.check.json`.

| file | source frames | rows | captures | standing |
|---|---|---|---|---|
| `fun_0604_1050.mp4` | 604-1050 | whole-person render; its patch on 752-780; face render | two runs and an owner map | **locked** by the owner, 2026-10-10; never rebuilt |
| `fun_0604_1050_patch_not_laid_do_not_use.mp4` | 604-1050 | the same table, built by the rule as it was that morning | the same | wrong on 752-780; kept as evidence until the session closes |
| `fun_0000_0603.mp4` | 0-603 | one whole-person render on its own four shots | none | standing, not locked |
| `fun_0000_0603_cut_spill_do_not_use.mp4` | 0-603 | the same render as one row | none | wrong on five frames at cuts; kept as evidence until the session closes |
| `fun_0604_1050_face_only.mp4`, `..._face_only_b.mp4` | 604-1050 | face render alone, without and with a class given back | one | interim, superseded by the locked file |

## What the locked file's record shows

`fun_0604_1050.mp4`, read on the placed file after it was copied from the
name it was built under:

- 447 frames, all nearest their own fed frame, none misplaced; timestamps in
  place; tagged BT.709, tv range.
- Largest 16-pixel square against the frame fed: 2.27 levels (frame 1018);
  squares over the bound: 0.
- Colour bias against the frames fed: 0.035 levels at most (Y, U, V).
- Audio: 806 packets, byte for byte a run of the source's; sync error under a
  sample.
- `rows_shown`: each of the three rows on a share of 1 of the pixels it was
  meant to show on.
- Shared pixels: 3,705,707 settled by order, 10,202 by a mask.
- On 758-768, on the pixels of the second subject where the render and its
  patch differ by over 12 levels (50,786 px a frame): the file is nearer the
  patch on 99.6% of them, 0.7 to 1.2 levels from the patch, 37 to 44 from the
  render (a reading made outside the tool, on the built file).
- The lead's own pixels given back from the source on 752-780: all 29 frames,
  median 1,989 px, most 9,472 (frame 766). The join flag is raised on all 29,
  longest 1,084 px (frame 761).
- No change carried across a cut: the whole-person render changes 222,512 px
  on frame 1009, none on 1010-1038, 132,283 on 1039.
- The same table without the lead given back on the whole-person rows (built
  to scratch, records compared with `--compare`): the lead's tracked pixels
  more than 12 levels off the source fall from 3.24% to 3.15%.
- md5 `33a335acb9ab194cf9bee964407f184a`, unchanged after two later re-reads.

## Two faults of the tool that day, both found on built files

1. **A patch laid under the render it patches.** With a capture given, a pixel
   two rows changed went to the row whose subject held it; the render had a
   run in the capture, the patch had none, so the render took the patched
   pixels back. Every proof passed. Found from one figure that was not the one
   expected (2,041,570 px "settled by a mask" where order was expected) and
   then measured on the file: on the pixels above, 11.5 to 13.8 levels from
   the render and 24.8 to 32.3 from the patch. Fixed: the latest row whose
   subject holds a pixel takes it; an unknown subject can take what it
   changed; a row can say whose it is.
2. **The wrong file passed a re-check under the fixed rule.** The new proof
   that every row shows reads the frames the table makes, and the tie between
   the file and those frames was a whole-frame figure: 0.49 on the wrong file
   against 0.42 on the right one. Made local: on the wrong file the largest
   square is 141.7 levels (frame 760) and 9,731 squares are over the bound,
   all on 752-780; on the right one, the figures above.

An earlier fault of the same day, of a table and not of the tool: one render
given as one row across several shots changed five frames of other shots at
the cuts (294, 414, 438, 482, 483). Rows are cut on the source's cuts, and a
flag reads the source's own frame-to-frame change for it.

## What a record proves and what it does not

- Read from the frames the tool makes from the table: `rows_shown`, the
  flags, the regions, the subjects' tables. They say the rule did what the
  table asked.
- Read from the file: the count, the order, the squares, the colour, the
  audio. They say the file is those frames.
- Not proved by anything here: that the picture looks right; that a join
  between source pixels and redrawn ones holds in motion (the join flag gives
  the frames to watch); that a piece's changed region is the region its
  render kept (it is rebuilt from a threshold measured on footage).
- The lock (`LOCKED.md` in the output folder) stops this tool building over a
  listed name and makes a re-read fail when the file's md5 has changed. It
  does not stop a copy, a move or another encoder.

## Changes, by date

- 2026-10-10, 0.269.0 (`b71b9024`): the rule for shared pixels, `rows_shown`,
  the local tie between file and table (`BLOCK`, `APART`), the refusal to
  build over a locked name. Session mrcorn.
- 2026-10-10: `fun_0604_1050.mp4` locked by the owner. A later version of the
  span is a new file beside it.
- 2026-10-10: the records of `fun_0000_0603.mp4`, its do-not-use twin and the
  two face-only interims were written before `rows_shown` and the local tie
  existed, and do not carry them. `fun_0000_0603.mp4` is re-read under them
  the same day; the line below says what it showed.
- 2026-10-10, later: `fun_0000_0603.mp4` re-read (`--check-only`, the video
  untouched, md5 the same before and after): passes; each of its four rows on
  a share of 1; largest square 2.79 levels (frame 579), none over the bound. Its record
  now carries the present proofs. Built with no capture, so no mask rule is
  read for it.
