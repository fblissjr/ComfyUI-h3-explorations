# Handover: read this before anything else

**Rewritten at the end of each working day by the session that led it; the
day before is in git.** This one was written on 2026-10-13 by `mrtower`, the
lead from 19:07Z (after `mrfit`, `mrpin` and `mrtan`), with `mrpen` on the
card, `mrohmy` on tooling and `mrlongtake` building the two-shot tool. It was
written at 22:45Z by the clock, BEFORE the card's day had ended, at the
owner's word, so that the day's state did not depend on one session's
context: the block "In flight when this was written" says what to check
against the record. Where this page and a record disagree, the record is
right (`decisions.md`, 2026-10-13; `data/fun/verdicts/verdicts.jsonl`; the
render ledger and the delivery table; the Open List).

## State when written (2026-10-13, 22:45Z by the clock)

- **The owner's target for the day's last hours: something made that day on
  everything up to frame 2049 of `fun.mp4`**, and one joined video of the
  WHOLE clip in `Video/fun/day_2026-10-13/` on the share, with an index page
  and, for every standing piece, the graph it was queued with
  (`bench/day_pieces.py`).
- **Kept by the owner on playback that day** (their clicks, in
  `verdicts.jsonl`): the second subject on 949-998; the lead on 460-483,
  852-965, 1010-1038, the hall face 1425-1441, and EVERY phone and oval
  inset from 1730 to 2037 (the first pieces rendered zoomed in; the shot
  1898-1943 as one piece, `lead_fp_1898`, which stands over the half
  `lead_fp_1920c` kept earlier).
- **Every load now samples a window fitted to its own length**
  (`masked_job_graph.window_for`, `h3_rules.snap_length`), by the owner's
  word; what that did to a render's time is in the server's log lines
  quoted in the day's page items and in the lead's notes, not here.
- **The shots record was mended at five places** where it and the picture
  disagreed (`bench/shots_record.py`; the looks are
  `data/fun/cuts_look_2026-10-13/reading.md`): one false cut, three missing,
  one two frames early. Every preview of 1920-6394 was made again from the
  owner's who-is-who; `bench/shot_parts.py` prints what each shot and
  subject needs.
- **Built that day, each with its check**: the render ledger and the
  delivery table generated from it; the chain from a shot to a page item
  (`shot_to_load.py`, `load_chain.py`) with a gate-alone path for ordinary
  loads (`no-look`); a preview made to ask the owner; a seed a tracker cut;
  the day's folder of pieces; a tool for two neighbouring shots in one load,
  opt-in (`bench/two_shot_load.py`); the first sound signals for choosing a
  window's edges (`bench/clip_signals.py`, NOT yet called made).

### Known gaps up to frame 2049

- **The lead**: the kitchen outside 852-965 (604-851: mostly the back of her
  head; where her face shows it is small and in short runs); 516-603 (her
  face a few pixels: nothing makeable at the node's largest zoom); and, IN
  FLIGHT when this was written, 294-414, 168-185 and 1134-1155.
- **The second subject**: 1010-1038 (the owner's answer to the Open List's
  row 24: face and hair, her new hair alone where her head is bowed; it
  needs a load that names a subset of its subject's parts, not built);
  1419-1465 (at the size floor even zoomed); 155-167 and 999-1009; IN
  FLIGHT, 1093-1133.
- **The third subject**: IN FLIGHT, 1419-1465; not started, what shows of
  him on 1397-1418 and after 1531.
- Not gaps: frames where no face of a subject shows (`data/fun/plan.json`,
  `departures`, and each job file's `left_out`).

### In flight when this was written (check each against the record)

- The card was working down the gap list above; the last new load was to
  start by 00:15Z. A piece is done when it has a page item and a row in the
  day's final delivery table.
- The joined clip of the whole video, the day's folder and its page items
  were to be made after the card's last piece: if `Video/fun/day_2026-10-13/`
  holds no placed clip, they were not, and the lead's handoff under
  `internal/claude/` (untracked) has the eleven steps in order.
- The sound signals were being mended after two readings found faults; a
  second throwaway run is owed a reader before anything reads the tables.

## What not to assume

- That a piece on the page was read frame by frame: since the owner's
  go-fast word an ordinary load renders on the gate alone, and its item's
  think is empty unless a reading exists.
- That `AGENTS.md` still asks for a look on every load: the owner removed
  that sentence; the chain holds where a look is owed.
- That the tracker's own pick is the owner's person: build only on a row
  `bench/shot_parts.py` marks as seeded.
- That a record shot is a cut, or that a tracker's cut is one: five were
  wrong, and most of the tracker's cuts in the last shot were a head
  dropping or an arm crossing.
- That the phrases or beats of `clip_signals.py` may be read yet.
- That a two-shot load has been rendered: the tool is in main and its test
  is not run.
- That the server is the one of the morning: it was restarted for the
  window change; find the port's owner before trusting a pid.

## Tomorrow, in order

Start in three seats (a card session, a tooling session, a lead that reads
and rules) and change none of `bench/masked_job.py`, `load_chain.py`,
`shot_to_load.py`, `capture_masked_run.py` until the first batch of renders
has landed.

1. Two small things left for a gap: `no-look` refusing a job that names
   `two_shots`; a sentence in `two_shots_of`'s docstring.
2. The two-shot test (the hall shots 1225-1305 and 1306-1396 as two loads
   and as one), all three renders or none, with one blind reading.
3. The second subject on 1010-1038, once a load can name a subset of its
   subject's parts (tooling's first job).
4. `lead_fp_1898` made once more: whether its second half is softer than the
   half by chance or by the longer load. A whole shot wins over a seam.
5. The hall face from the cut (1419-1441) beside `lead_fp_1425b`.
6. The long shot 2050-2341, the first window near the planner's longest: a
   plan look, its render read, the owner's eye, before the two halves of
   the last shot go that way.
7. What of the gap list did not land; then 2050 onward in frame order.

For the owner's first batch: the third subject's beard (row 23); folder
names on the share (row 21); whether her face on the spinning ball
(2218-2341) is replaced; a skill for the lane and a pointer in `AGENTS.md`
to the who-is-who procedure.

## Where things are

- The day's rulings: [`decisions.md`](decisions.md), 2026-10-13. The lane:
  [`masked_v2v.md`](masked_v2v.md). Rows owed: [`next_steps.md`](next_steps.md).
  The data wanted for a window's edges: [`window_edges_data.md`](window_edges_data.md).
- What each shot needs: the newest `data/fun/shot_parts_*.md`. Who is in
  each shot: `data/fun/shots.json`, `data/fun/cast/cast.jsonl`.
- The owner's pages: the H3 Open List; the local review page (the page
  server under `data/fun/poc/`).
- Each session's notes and handoff of the day: `internal/claude/` (untracked).
