# Handover: read this before anything else

**Rewritten at the end of each working day by the session that led it; the
day before is in git.** This one was written at the close of 2026-10-13 (the
owner's evening; the clock had passed midnight UTC) by `mrtower`, the lead
from 19:07Z, with `mrpen` on the card and `mrohmy` on tooling. Where this
page and a record disagree, the record is right (`decisions.md`,
2026-10-13; `data/fun/verdicts/verdicts.jsonl`; the render ledger and the
delivery table; the Open List).

## State when closed

- **The owner watched the day's joined clip of the whole of `fun.mp4`**
  (`Video/fun/day_2026-10-13/` on the share: the clip, its stacked form, a
  text file with how it was made, an index page and a workflow folder for
  every piece in it). Up to frame 2037 it is pieces, each laid from its own
  build record; from 2050 on it is the original: nothing is made there yet.
- **After watching, the owner rated what they had not judged.** Kept on
  playback that day: the second subject on 949-998; the lead on 294-413,
  460-483, 852-965, 1010-1038, 1061-1079, 1134-1155, 1306-1396, the hall
  face 1425-1441 and every phone and oval inset from 1730 to 2037; the third
  subject on 1310-1396 and 1466-1531, and by their word 1222-1305. Rated
  wrong, on one fault (her eyes do not follow the original's lowered look):
  the second subject's pieces of the 12th on 14-154, 415-437 and 484-515.
  Not sure: her 278-293 and the lead's 1466-1534 and 1575-1729, all of the
  12th. `verdicts.jsonl` is the record.
- **The third subject's look is settled by the owner: no beard**, as on
  1222-1305 and 1466-1531. His 1452-1465, which had drawn one, was made
  again without (`third_fp_1452`, not yet judged).
- **A table made after those ratings stands, and no clip is built from it**
  (`t_fun_0000_6394_2026-10-14T0033Z` in
  `data/2026-10-13_fun_delivery_tables/`): the three pieces rated wrong are
  out and the original is laid there, so the whole-video clip is NOT built
  again until they are remade. Six per-shot clips with every piece laid are
  built from it (`Video/fun/day_2026-10-13/shots/`).
- **What the owner reviews from**, both local files in the day's folder on
  the share, generated from the table, the ledger and the verdicts, never
  typed: `TO_REVIEW.html` (the one place with review links, at their word:
  none in a hosted page or a tracked file) and `PROVENANCE.html` (a timeline
  a subject, every stretch and gap sorted by frame, where each piece came
  from, and what changes in the next clip and why).
- **The gaps are printed by a tool at the end of every delivery table**
  (`bench/piece_gaps.py`): frames no piece covers, frames a piece covers and
  leaves as the original, and a subject in the shot under another's piece.
  The ledger carries the frames each render leaves as the original.
- **Built that day, each with its check**: the ledger and the generated
  table; the chain from a shot to a page item with a gate-alone path
  (`no-look`); a window fitted to a load's own length; the shots record's
  tool (five mends); the day's folder; two shots in one load, opt-in, its
  test not run (`bench/two_shot_load.py`); a load that names some of its
  subject's parts (`shot_to_load.py job --parts`); the sound signals for a
  window's edges (`bench/clip_signals.py`, NOT called made).
- **Both servers were stopped at the close.** How the review page's server
  is started by hand is in [`../comfy_notes.md`](../comfy_notes.md).

### Known gaps up to frame 2049

The list is generated: run `bench/piece_gaps.py` on the newest table, or
read the tail of the table's own print. What the evening found, in short:

- **Where the owner saw a subject go back to the original**, none a loss of
  tracking: the third subject on 1397-1451 (his arm, a bowed head, then his
  back; what returns is his beard); the lead on 516-603 (her face a few
  pixels); the second subject on 999-1009, where her own hair returns.
- **Tried that evening and stopped, each on a named fault with a row in
  [`next_steps.md`](next_steps.md)**: the second subject on 998-1009 (the
  lead's track lies on her head though the lead has left the picture) and
  on 1010-1038 (her keep-outs take most of a small region; the reference
  follows another body where the lead crosses in front of her); the third
  subject on 1397-1418 (the fill on his arm and on the lead's hair).
- **Not looked at as frames**: the lead on 438-459, 966-998, 1080-1097,
  1397-1424 and 1442-1465; the third subject on 1306-1309.
- **The shots record is unsure** the second subject is in 1156-1188 and
  1466-1729: the owner's click on the who-is-who page before anything is
  built there.

## What not to assume

- That a piece in a clip was kept by the owner: where there is no keep the
  table lays the newest render in the form, and says which is which.
- That a load's own frames are all replaced: a load leaves frames as the
  original, and a reason in its job file may rest on a look at every fifth
  frame. Look at every frame of a stretch before ruling or building on it.
- That another try will fix a blocked load: every stop of the evening was a
  fault in a record or a step (a track, a reference, a keep-out).
- That a window may hold frames of the subject left as the original at no
  cost: the one load that did drew the original's beard onto the new face.
- That the owner can judge one load's clip of a two-person shot: they are
  shown the shot with every piece laid, each marked with its render day.
- That the tracker's pick is the owner's person, or a record shot a cut:
  build only on a row `bench/shot_parts.py` marks as seeded.
- That the phrases, beats or dips of `clip_signals.py` may be read yet.
- That a server is running: find the port's owner before trusting a pid.

## Tomorrow, in order

Three seats from the start (a card session, a tooling session, a lead that
reads and rules). The owner reviews first, from `TO_REVIEW.html`.

1. **The owner's answers**, which decide the card's order: whether every
   piece of the 12th is made again with the present tooling; whether the
   three short runs of the third subject on 1397-1418 are built; what they
   want to send to the sessions from the provenance page.
2. **Tooling first**, since the gaps the owner saw are tooling rows away and
   not renders away: a part that changes inside one load (`--parts-on`);
   the reference step held to the subject's own track; a way to say a
   track is not that person on named frames; the gate's reading of
   keep-outs on a small head. Before any tool for a new kind of load, one
   dry run of that load to its gate on the card.
3. **The card meanwhile**: the second subject's eyes, as a test on her
   shortest shot (415-437) before her three pieces are made again; then the
   two-shot test on the hall shots; the long shot at 2050 with a look, a
   reading and the owner's eye; then 2050 onward in frame order.
4. **When tooling lands**: the second subject on 998-1009 and 1010-1038;
   the third subject on 1397-1418 if the owner says so; then the
   whole-video clip built again.
5. Carried: `lead_fp_1898` once more (softness); the hall face from the cut
   beside `lead_fp_1425b`; a look at the stretches not looked at.

Still the owner's, none blocking: folder names on the share (row 21);
whether her face on the spinning ball (2218-2341) is replaced; a skill for
the lane and a pointer in `AGENTS.md` to the who-is-who procedure; one
listening check for the sound signals; the unplaced copies and throwaway
tables of the day, kept until they say.

## Where things are

- The day's rulings: [`decisions.md`](decisions.md), 2026-10-13. The lane:
  [`masked_v2v.md`](masked_v2v.md). Rows owed and tooling in order:
  [`next_steps.md`](next_steps.md).
- The day's clip, the per-shot clips, the two local files and a workflow
  for every piece: `Video/fun/day_2026-10-13/` on the share. Ledgers,
  tables, readings and gap prints: `data/2026-10-13_fun_delivery_tables/`.
- What each shot needs: the newest `data/fun/shot_parts_*.md`. Who is in
  each shot: `data/fun/shots.json`, `data/fun/cast/cast.jsonl`.
- Each session's notes, handoff and feedback on the day: `internal/claude/`
  (untracked), `2026-10-13_what_we_did_and_feedback.md` among them.
