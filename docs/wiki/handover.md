# Handover: read this before anything else

**Rewritten at the end of each working day by the session that led it; the day before is in git.** Written at the
owner's close of 2026-10-16 (their evening, about 22:45Z) by `mrtony`, the lead from 17:46Z, with `mrtaco` on the card
and still building. Where this page and a record disagree, the record is right (`decisions.md`, newest first;
`data/fun/verdicts/verdicts.jsonl`; `data/fun/laid.jsonl`; the newest table and gap check; `git log`). The lead's own
notes: `internal/claude/2026-10-16_mrtony/HANDOFF.md`; the card's: `internal/claude/2026-10-16_mrtaco/HANDOFF.md`.

## State when closed

- **The owner is off for the night.** Unattended, on their recorded go only: `second_hfp_1010c` (the mom, 1010-1038)
  renders once the exact matte has landed, its checks pass and ComfyUI is restarted by the recipe; then ComfyUI is
  stopped. Nothing else renders: every other load stops at its plan strip or diff.
- **Landed today** (`git log fab71024..HEAD`): the sidecar `<name>.mp4.json`; the one view (kept squares in yellow,
  within H.264 level 5.1, index at the front); gates 1-4 (no largest piece on a face; own restores only as an exception;
  keep-outs carved out of her own track, on the node's own grid, route C with the land step holding the render's region
  to the prediction; a "whole region" piece laid whole); the gap check in the chain, its new kinds (barely changed,
  region too small, corrected scale) and the remake/capture split; the frame table (`bench/frame_table.py`, a Gemini
  builder's); the hair band made again after the carve; the reference-folder rule (`_b`, `_c`); the clothing line in
  load texts.
- **The cut** (`data/fun/laid.jsonl`, 43 lines): the owner kept today the street `lead_fp_0438c`, the mom's
  `second_hfp_0014f` (whole region; it settles her margin and composite), `lead_fp_0294d`, and by look every piece on
  1419-2049, `lead_fp_2625` and `lead_fp_2823b`. Four lines have no keep, all in the kitchen being remade.
- **The newest whole video** is `fun_0000_6394_joined_2026-10-16T2003Z_from_laid` (gate 4 in; before 0014f's and
  0294d's keeps were laid). The app plays `..._VIEW3_web.mp4`. Rebuild after the mom's render.

## What not to assume

- **Most laid pieces were rendered "only what changed"** (all but the street and 0014f): the original can show where
  they barely changed the picture. The gap check flags it only under its line (806); the owner's eye is the check.
- **A piece is not laid on most of the song**: 2182 of 6395 frames carry one; 48 shots (3449 frames) need a capture
  before any load. Nothing is made after 5564.
- **The 0613f kitchen plan is NOT to render**: Sapiens labels her skirt and arm as her face on many frames (Gemini D16);
  the off-head rule and floor-hold remake it as `lead_fp_0613g`.
- **Captures from before c3b53645 are refused** wherever their load names a keep-out; a load planned before the carve is
  made again under a new name (the refusal says so).
- **Every time written today came from `date -u`** after three estimated ones were wrong; keep doing that.

## Tomorrow, in order

1. Read the night's outbox (`internal/gemini/2026-10-16_outbox.md`) and mrtaco's HANDOFF; land what passed its read.
   Then write the 2026-10-16 changelog entry from all of that day's commits, the night's included
   (`changelog.d/README.md`; `bump: minor`: the day added tools).
2. The mom's 1010 render (`second_hfp_1010c`) is a CANDIDATE beside `second_hfp_1010j`, which the owner kept on 10-15
   and which is laid: put its stacked view on Start here beside 1010j's; it is laid only on their keep. The whole video
   was rebuilt from the laid list at the close (`rebuild.sh --clip`, the stamp in the pages' table) with 0014f and 0294d.
3. The owner's looks, on Start here in this order: the plan strips (`lead_fp_1010g` with the hat square gone,
   `lead_fp_0613g` with its zoom figures and 626-629 / 803-805 checked, 516-603, the whole-shot loads 1061-1092,
   1093-1133, 1306-1418 and the dad's 1310-1418); the re-composite before/afters (0604d, 0278c, 0415d, 0484c); the four
   late pieces 4653-4677, 5114-5129, 5461-5497, 5533-5564.
4. Land Gemini's D19 (pages read the gap check) and D20 (`--carry-words-from`) after a Claude read each.
5. Open with the owner: a new kitchen load for the mom (604-1009) under today's rules (the Open List's r16_01, the lead's
   pick: see the re-laid 0604d first); the
   per-frame matte trial's result; re-compositing kept pieces one by one on their word.

## Where things are

- Decisions of the day: `docs/wiki/decisions.md` 2026-10-16 (newest first). The owner's notes:
  `data/fun/verdicts/verdicts.jsonl` (today's from 21:50Z on are the whole-video watch).
- Requests to and answers from the Gemini sessions: `internal/gemini/2026-10-16_requests.md` (D1-D20) and
  `..._outbox.md`; reads by the lead's blind readers: `internal/claude/2026-10-16_mrtony/review*.md`.
- Pages: `Video/fun/TO_REVIEW.html` and `PROVENANCE.html` on the share, made by
  `internal/claude/2026-10-16_mrtoon/pages.sh` from `start_here.json`; the app at the review server
  (`data/fun/poc/serve.py`, restarted by the lead at about 18:20Z).
- Re-composite trials: `data/2026-10-16_recomposite_trials/` and `Video/fun/day_2026-10-16/recomposite_trials/`.
