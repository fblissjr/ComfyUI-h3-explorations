# Handover: read this before anything else

**Rewritten at the end of each working day by the session that led it; the
day before is in git.** This one was written on 2026-10-13 by `mrtower`, the
lead from 19:07Z (after `mrfit`, `mrpin` and `mrtan`), with `mrpen` on the
card, `mrohmy` on tooling and `mrlongtake` building the two-shot tool. Where
this page and a record disagree, the record is right (`decisions.md`,
2026-10-13; `data/fun/verdicts/verdicts.jsonl`; the render ledger and the
delivery table; the Open List).

## State when closed (2026-10-13, the owner's evening)

- **The day's end-of-day clip is placed**: the WHOLE of `fun.mp4` in one
  file, `Video/fun/day_2026-10-13/fun_0000_6394_joined_2026-10-13T2319Z.mp4`
  on the share, with its stacked form, a text file beside it (how it was
  made, the known gaps, the table), an index page and one small folder a
  standing piece holding the graph it was queued with (`bench/day_pieces.py`).
  Its build record passes and was read by the lead; nobody had watched it
  on playback when this was written. It is on the local review page as two
  items.
- **What is in it**: every piece up to frame 2037 that has a record of how
  it was made, most rendered that day and the rest the day before, standing
  only where nothing was rendered for that subject that day; the list by
  day with why each stands is
  `data/2026-10-13_fun_delivery_tables/pieces_by_day_fun_0000_6394_2026-10-13T2319Z.md`.
  From 2050 to the end the clip is the original: nothing is made there yet.
- **Kept by the owner on playback that day** (their clicks, in
  `verdicts.jsonl`): the second subject on 949-998; the lead on the street
  walk 294-413, on 460-483, 852-965, 1010-1038, 1061-1079, 1134-1155, the
  hall face 1425-1441, and EVERY phone and oval inset from 1730 to 2037
  (the first pieces rendered zoomed in; the shot 1898-1943 as one piece,
  `lead_fp_1898`, which stands over the half `lead_fp_1920c` kept earlier);
  the third subject's face on 1452-1465, his first piece through the chain.
- **Every load now samples a window fitted to its own length**
  (`masked_job_graph.window_for`, `h3_rules.snap_length`), by the owner's
  word; what that did to a render's time is in the server's log lines
  quoted in the day's page items and in the card session's handoff.
- **The shots record was mended at five places** where it and the picture
  disagreed (`bench/shots_record.py`; the looks are
  `data/fun/cuts_look_2026-10-13/reading.md`). Every preview of 1920-6394
  was made again from the owner's who-is-who; `bench/shot_parts.py` prints
  what each shot and subject needs.
- **Built that day, each with its check**: the render ledger and the
  delivery table generated from it; the chain from a shot to a page item
  (`shot_to_load.py`, `load_chain.py`) with a gate-alone path for ordinary
  loads (`no-look`); a preview made to ask the owner; a seed a tracker cut;
  the day's folder of pieces; a tool for two neighbouring shots in one load,
  opt-in (`bench/two_shot_load.py`); the first sound signals for choosing a
  window's edges (`bench/clip_signals.py`, NOT yet called made).

### Known gaps up to frame 2049

Where one of the three subjects is in the picture and the clip shows the
original. The text file beside the clip has the same list; what every shot
still needs is `coverage_fun_0000_6394_2026-10-13T2319Z.txt` in the day's
delivery tables folder, regenerated and never typed.

- **The lead**: the kitchen before her piece (604-851: mostly the back of
  her head; where her face shows it is small and in short runs); 516-603
  (her face a few pixels: nothing makeable at the node's largest zoom);
  168-185 (the node refused the zoom: she walks across more than half the
  frame and a load has one box; a row in `next_steps.md`). Left as the
  original on purpose inside her pieces: three stretches of the street walk
  and the middle of 1134-1155.
- **The second subject**: 1010-1038 (the owner's answer to the Open List's
  row 24: face and hair, her new hair alone where her head is bowed; it
  needs a load that names a subset of its subject's parts, not built);
  1419-1465 and 1093-1133 (too small even zoomed: on 1093-1133 the first
  capture's verify and gate refused the load, kept under
  `data/2026-10-13_second_hfp_1093_job/chain/`); 155-167 and 999-1009.
- **The third subject**: 1419-1451 (his head bowed, left as the original on
  purpose) and 1397-1418 (his face shows on a few frames in two runs; not
  started).
- **Not looked at as frames**, each the original because that day's piece
  covers fewer frames than the older clip's did: the lead on 438-459,
  966-1009, 1080-1097, 1397-1424 and 1442-1465; the third subject on
  1306-1309 and 1397-1410. Whether each is a gap is a look for the next day.
- Not gaps: frames where no face of a subject shows (`data/fun/plan.json`,
  `departures`, and each job file's `left_out`); the third subject on
  1532-1729, his back to the camera by the card session's sampled look.

### Left open when closed

- Pieces in the clip the owner has not clicked on (the review page shows
  each as open; `verdicts.jsonl` is the record). The owner does not yet
  know which look of the third subject they like best, with a beard or
  without: three of his pieces have stubble and the newest a drawn beard.
  Settle that with them before building more of him.
- The sound signals: mended twice after three readings; the run to read is
  the tooling session's third throwaway (its notes say how to make it
  again). A reader who did not build it reads that run before anything
  reads the tables.
- The servers: ComfyUI was stopped at the close, its queue empty; the page
  server was left running because the owner was still reviewing on it. It
  is to be stopped by whoever finds it idle.
- Three previews of the mended shots are still to be made seeded from the
  owner's answers (4602, 5889, 2623); the standing previews of 4602 and
  5889 have the second and third subjects the wrong way round.

## What not to assume

- That a piece in the clip was kept by the owner: where there is no keep
  the table lays the newest render in the form, and says which is which.
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
- That the phrases, beats or dips of `clip_signals.py` may be read yet.
- That a two-shot load has been rendered: the tool is in main and its test
  is not run.
- That a server is running or is the one of the morning: find the port's
  owner before trusting a pid (`docs/comfy_notes.md`).

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
7. A look at the stretches listed above as not looked at; the third subject
   on 1397-1418 by the recipe of his first piece; then 2050 onward in frame
   order.

For the owner's first batch: the third subject with or without a beard
(they answered row 23 as rendered, his beard gone; that evening they wrote
on the page that they may like him better with it, and his newest piece
came out with a drawn one); folder names on the share (row 21); whether her
face on the spinning ball (2218-2341) is replaced; a skill for the lane and
a pointer in `AGENTS.md` to the who-is-who procedure; one listening check
for the sound signals (whether a dip inside one long sung run is a breath
between two lines).

## Where things are

- The day's rulings: [`decisions.md`](decisions.md), 2026-10-13. The lane:
  [`masked_v2v.md`](masked_v2v.md). Rows owed and tooling in order:
  [`next_steps.md`](next_steps.md). The data wanted for a window's edges:
  [`window_edges_data.md`](window_edges_data.md).
- The day's clip, its notes and a workflow for every piece:
  `Video/fun/day_2026-10-13/` on the share. Its ledger, table, reading,
  coverage and records: `data/2026-10-13_fun_delivery_tables/`, the files
  named `2026-10-13T2319Z`.
- What each shot needs: the newest `data/fun/shot_parts_*.md`. Who is in
  each shot: `data/fun/shots.json`, `data/fun/cast/cast.jsonl`.
- The owner's pages: the H3 Open List; the local review page (the page
  server under `data/fun/poc/`).
- Each session's notes and handoff of the day: `internal/claude/` (untracked).
