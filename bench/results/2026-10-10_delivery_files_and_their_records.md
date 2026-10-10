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
- 2026-10-10, later still: **the locked file shows the source's own picture
  for a whole shot, and every proof had passed.** On source frames 1010-1038
  (29 frames) nothing is laid for either subject: both trackers had called
  their subject absent there, with two people detected. A viewer found it.
  The data had said so twice and nothing made anyone answer it: the record's
  own flag `piece_changes_nothing` named those frames for the whole-person
  row, and the capture's flag `absent_with_people_on_screen` named them for
  the lead (the closest person at 0.667 against a line of 0.80). I had read
  "0 px changed on 1010-1038" as the cut gate holding. A second witness,
  measured by the lead session on the file against the source: about 51 dB
  on every frame of 1010-1038 (one more encode of the same picture), against
  about 22 dB before the cut and about 24 dB on 1039-1050.
- 2026-10-10, the same change: **a proof for it**, `shots` in the record
  ("shots where a named subject is the source's own"). The span is cut into
  shots by the source's own cuts; for every subject a row names, a shot with
  nothing laid fails the build when the subject is tracked there or when
  people are detected who are not other named subjects laid or tracked
  there, and is listed and counted otherwise (the people are another
  subject's, nobody was detected, or nothing is known). The table answers it
  with a `source first-last subject=<label> words` line, whose words are
  kept. Runs of frames inside a laid shot where the subject is tracked and
  nothing is laid are listed with the longest. `verdict_line` in the record
  carries the count, and the frames when any fail.
- 2026-10-10, the locked file read again under that proof (`--check-only`;
  md5 unchanged): **FAILS**, on 1010-1038 for both subjects (not tracked,
  people detected: 2); 1039-1050 is by intent for the lead (a `source` line
  added to the table read, in the lead session's words: not in this shot).
  Its other proofs read as before. Also in that record, said and not failed:
  inside the first shot the lead is tracked with nothing laid on 45 frames,
  the longest run 25 (974-998; the other is 606-625). The file, its md5 and
  its lock stand; `LOCKED.md` carries a dated note of the fault; a version
  with rows on that shot is to be built beside it as `fun_0604_1050_b.mp4`.
- 2026-10-10, what the new proof cannot know, as its record says: a shot's
  people count is one frame of it; a subject with no shot table borrows
  another subject's count (only the lead has a shot table in that day's
  kitchen captures); a dissolve is not a cut; with no capture no subject is
  named and nothing is asked.
- 2026-10-10, a rule changed: **a class given back keeps off what an owner
  map gives another subject.** A class restore read the named subject's
  class map and nothing else. Measured on one two-person shot of 29 frames
  (a capture of source frames 1010-1038): one subject's class map put 19.7%
  of its lower clothing class, 18.7% of a hand and 24.9% of its upper
  clothing on pixels the owner map gives the other subject; the other
  subject's map put 0.1% and 1.2% of the two classes its rows give back on
  the first's. Where an owner map covers the frame the class is now kept
  off those pixels; pixels no track claims and contested ones are still
  given back; with no owner map nothing changes. The locked file's bytes
  still match its table under the changed rule (a re-read of the same bytes
  in scratch: no square over the bound, largest 2.27 as before): the 6,695
  class pixels the rule now leaves, on 51 frames, are pixels its face row
  gives back anyway as the other subject's whole.
- 2026-10-10, the shots proof failed its first real file and the fault was
  the proof's: two scratch builds of the opening stretch read "tracked,
  nothing laid" on three shots one frame long (316, 317, 318). The source
  moves over the cut line on five consecutive frames there, 315-319, a blur
  inside one shot; the tracker's own shot table for that load has no cut in
  294-414; and the face part is emptied on those frames on purpose.
  Consecutive frames over the line are now one cut at the first of them, the
  run is listed in the record (`runs_of_frames_over_the_line_read_as_one_cut`),
  and a capture's shot table that covers the frames and puts its cuts
  elsewhere is reported against this tool's cuts on those frames only. I had
  predicted "passes, the other shots not known" before reading the records;
  the difference is what found it.
- 2026-10-10, a new file: **`fun_0000_0603_b.mp4`**, beside
  `fun_0000_0603.mp4` and differing from it by two rows, the lead's face on
  two street shots (294-414 and 438-483), each a per-shot load. Standing: not
  locked; for the owner to watch. Built two ways to scratch names and
  compared by the join flag (source pixels given back that sit against a
  change of over 25 levels), per row:

  | row | given back | px a frame given back (median) | join: frames, median px, total, longest |
  |---|---|---|---|
  | 294-414 | the Apparel class | 860 | 81, 52, 8,051, 323 (frame 314) |
  | 294-414 | Apparel and Hair | 2,046 | 101, 75, 11,780, 323 (frame 314) |
  | 438-483 | the Apparel class | 1,171 | 36, 76, 4,588, 361 (frame 462) |
  | 438-483 | Apparel and Hair | 3,163 | 38, 142, 6,860, 520 (frame 462) |

  Placed: Apparel only (the hat's edge is under that class; the hair stays
  as the render drew it). The lead session's reason, kept here because the
  capture session's reading (the hair is redrawn inside the region at about
  ten levels against a floor of three) pointed the other way: a seam along
  the whole hairline is on every frame, and hair redrawn with the face
  inside a sixteen-pixel margin is the scene's own hair. The other build is
  kept in a session scratchpad and is one build away.
  The placed file's record: passes; six rows each on a share of 1; no square
  over the bound (largest 2.84, frame 561); 604 frames, 1,085 audio packets
  of the source's; the lead tracked with nothing laid on 316-319, 350-357,
  414 and 451-455 (the part emptied on purpose); the across-a-cut flag on 315
  is the blur above; a known limit in its note: on 366-390 the source's
  mouth opens wide with no voice and the render's does not follow (0.61 of
  the source's opening by the capture session's measure). Eight other shots
  of the span have nothing laid for the lead and read "not known": no
  capture given covers them. md5 `31edf7f411809c8841906d58fe7ade5d`.
- 2026-10-10, the class rule of that day narrowed, an hour after it went in:
  it kept a class off whatever an owner map gave ANY other subject. On a row
  giving back its OWN subject's class that left the piece's change standing
  on a third person's pixels, where the source is the right picture and was
  what the row showed before the rule. The harm the rule is for is another
  subject's class map reaching onto the ROW'S OWN subject, so that is what
  it now covers and nothing else. No file was built under the wider rule
  with an owner map given (the street file's captures hold none).
- 2026-10-10, `fun_0000_0603_b.mp4`'s note amended (`--check-only`, md5
  unchanged): the eight shots that read "not known" for the lead are parts
  of the span where she is on screen and no pass has been made yet; that is
  the job's remaining work and not a fault of the file.
