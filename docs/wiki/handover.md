# Handover: read this before anything else

**Rewritten at the end of each working day by the session that led it; the
day before is in git.** This one was written at the close of 2026-10-14 (about
5:45pm the owner's time) by `mrtwo`, the lead, with `mrcircle` on the card,
`mrdrill` on tooling and `mrpre` on the region and look spec. Where this page
and a record disagree, the record is right (`decisions.md`, 2026-10-14;
`data/fun/verdicts/verdicts.jsonl`; the newest ledger and table; the Open List).

## State when closed

- **The ledger and table stand at `t_fun_0000_6394_2026-10-14T2242Z`**
  (`data/2026-10-14_fun_delivery_tables/`), rebuilt by
  `internal/claude/2026-10-14_mrtwo_scripts/rebuild.sh` with the prefers in
  `prefers.txt` beside it; the owner's `TO_REVIEW.html` and `PROVENANCE.html`
  (`Video/fun/day_2026-10-13/` on the share) are made from it by `pages.sh`.
- **Kept by the owner today, on playback**: the second subject's 484c, 278c,
  415d ("until the apron strap is fixed"), 14d (two small hair flashes on her
  right at 57 and 163, in the record) and the 604-1009 cut; the lead's 1097c,
  1156c and the 1466-1729 cut; the third subject's 4653b, 5533b and the
  1222-1305 cut. Rated wrong: 484b, 1156, 4479b (beard), 1222's 18:49Z cut.
- **The chain's defaults** are every lane tool, never a per-load pick
  (`bench/shot_to_load.py::defaults_carried`); a kept piece is not remade for a
  better form. The text is generated in one form (`bench/load_text.py`).
- **Gaze is a recorded limit** (`decisions.md`, 2026-10-14): four renders, the
  last zoomed, did not follow a sideways look. The drawn mouth does carry.
- **Landed at the close, not yet seen by the owner**: `lead_fp_0751b` (751-778,
  the owner's 758; 761 a one-frame flicker) and `second_hfp_5675b` (5675-5715).
  751b sits in the two-subject shot 604-1009: its joined clip is to build
  before the owner sees it.
- **Both servers stopped at the close** (ComfyUI pid 10995 at 22:41:51Z by the
  recipe; the review page on 8765 at 22:43Z). `docs/comfy_notes.md` has how to
  start them.

### Planned and not rendered (`internal/claude/2026-10-14_mrcircle_night_notes.md` has each)

- The lead's 630-680 and 805-833 and the second subject's 1010-1031 go as
  `lead_fp_0630c`, `lead_fp_0805c` and `second_hfp_1010e`: their b/d loads
  drew the mouth off her face; the references were made again under the mend
  (eed4acb9, no mouth on the readers' frames, checked) and each needs a new
  letter on that reference, then `look_carry.py` against its earlier load
  (1010e needs its two overrides again).
- The second subject past 2049: 5461c (the empty clothing keep-out dropped,
  69cfe18e; its reader looks at her apron and straps) and 5114c (the
  zoom-held floor; its span 5114-5129 is in the plan; its reader names the
  third subject's face or hands in her region).
- The five job steps are scripted, not run:
  `internal/claude/2026-10-14_mrcircle_scripts/chain_2236.sh`.

## What not to assume

- That a look carries to a remade load whose reference changed: `look_carry.py`
  compares masks, not the reference video (an open question for the reviewer).
- That a flag on today's captures still reads as written: the specks floor,
  the held split, the voice mend and the zoom-held floor all landed after some
  captures were flagged. Flag again before ruling.
- That the hair coming and going has one cause. The composite row in
  `next_steps.md` holds three, read on 604d and 14d: the composite keeping the
  original where the drawing nears its colours (604, bottom left); the drawing
  itself (14d, 50-65); and a keep-out speck on one frame holing its whole
  latent step of four frames (14d, 159-166: held on 161-162, strangers on
  164-165, so 163 is holed with no mask on it). mrpre's first check tomorrow:
  which grid cell each speck falls in.
- That a load renders at full frame: the strip's header says which.

## Tomorrow, in order

1. **The card**: 630c, 805c and 1010e, then 5461c and 5114c (above); build the
   604-1009 joined clip with 751b for the owner; then the remaining 0-2049 gaps (run
   `bench/propose_runs.py` first, then `bench/piece_gaps.py` on the newest
   table); then the joined clip of 0-2049 for the owner.
2. **Past 2049**: the queue in `internal/claude/2026-10-14_mrdrill_runs_past_2049.md`
   (gaps longest first: the lead's 2823-3097 and 2050-2248), with the second
   subject's 5114b, 5461b and 5675b.
3. **Tooling** (mrdrill's order): the zoom-held floor's first real capture
   read, and a case running the files step on made mask videos; overlapping
   keeps ordered by the keep's time; the gaps print in `shot_parts` and the job
   step; the strip drawing a zoomed box; the queue past 2049 as a tracked tool
   re-cut from each newest table. The region seat (mrpre): the step-shared
   hole, measured on 14d and 604d before any build. Read the reviewer's notes
   on today's landings first (`internal/gemini/`).
4. **The owner, none blocking**: who is in the shot for the second subject on
   1156-1188, 1466-1729 and 1535-1574 (a who page); row z19.

## Where things are

- Rulings: [`decisions.md`](decisions.md), 2026-10-14. Rows:
  [`next_steps.md`](next_steps.md). The lane: [`masked_v2v.md`](masked_v2v.md).
- The lead's scripts (rebuild, pages, prefers): `internal/claude/2026-10-14_mrtwo_scripts/`.
  Briefs for each seat: `internal/claude/2026-10-14_mrtwo_brief_*.md`.
- The outside reviewer's notes: `internal/gemini/`; its brief:
  `internal/claude/2026-10-14_for_the_auditor/START_HERE.md`.
- Clips: `Video/fun/review_clips/` and `Video/fun/day_2026-10-14/` on the share.
