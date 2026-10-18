# Handover: read this before anything else

**Rewritten at the end of each working day by the Claude lead; the day before is in git.** Written for 2026-10-17 by
`mrtabby` (the Claude lead) at 00:35Z on 2026-10-18, BEFORE the close: the card is still running, and whoever closes
adds what landed after this. Where this page and a record disagree, the record is right (`decisions.md`, newest first;
`data/fun/laid.jsonl`; the newest table and gap check; `git log`). The mailbox is `internal/gemini/` (`PROTOCOL.md`;
`2026-10-17_claude_to_gemini.md` C1-C119, `2026-10-17_gemini_to_claude.md` G1-G61).

## State when closed

- **Two leads** since the morning (`AGENTS.md`): mrgemsol (Gemini) holds the card, git on main, the ledger and the
  owner's pages; mrtabby (Claude) holds the Open List, the records and every clearance. Workers: gem-matte (perception
  builds) and gem-memsel (started 00:29Z; SAM memory selection, a measurement only).
- **Landed today** (`git log 483d5b38..HEAD`): one compositing law (`assemble_delivery.lay_plane`) and mattes for all 43
  laid pieces; `bench/recomposite_piece.py`; `bench/rebuild_delivery.py`, one command for the ledger, table, gap check
  and pages; the who view; the floor holds and off-head pruning (D22b, 9f139d0c); the mouth fixes behind
  `--closed-lip-line`, off by default (c796c025); the rule that a change names everything that writes and reads it
  (`AGENTS.md`); the folder AGENTS.md files; meta_sam3 removed; `data/` untracked but for its two instruction files.
- **The cut** is unchanged since 1747Z: 43 laid pieces. Tonight's renders are candidates only, none laid:
  `second_hfp_0604h` (the mom's kitchen, done, compared), `lead_fp_0613g` (the lead's 613-974 in one window, landing at
  this writing), `second_hfp_0278d` and `lead_fp_0438d` (the control) queued after it.
- **On branches, not landed:** `gemini-hole-label` (sent back, C118: the two causes need two wordings);
  `gemmatte-masks` (the head tube, being bound by the person track and wired into the capture behind a switch, C108);
  `gemmatte-moge` (MoGe-2 measuring tool; its loader moves to `vendor/meta_moge2/` with Meta's licence, the owner's
  call, C112); gem-memsel's tool (`h3_worktrees/gem-memsel-tool`).
- **Open List**: r17_06 (the mom's kitchen re-lays) and r17_07 (the perception order) wait on the owner; r17_08 is
  tonight's state; r17_09 records the licence answer. The inbox is clear.

## What not to assume

- **"Tracking not known" on the stacked view usually means no piece was made there**, not that a track was lost (1402,
  the owner's screenshot): the lead's 1306s ends at 1396, so 1397-1418 is the original for her and the dad. Until the
  label lands, read the gap check's cause column.
- **The kitchen's holes are a face-mask failure, not a tracker failure.** The person tracker held the lead on 399 of 406
  frames of 604-1009; the face mask (Sapiens, frame by frame) collapses on 680-712, 733-751, 779-793, 800-803 and
  834-845, and the old kitchen loads were cut round those frames. The floor holds give up on a run over 12 frames
  (`HOLD_DROP_MOST_FRAMES`); 0613g therefore leaves 733-738 and 747-750 as the original (her face shows on 736-738).
- **The head tube is not a fill.** Its 3-frame average bridges 1-2 frames; its frame labels change nothing; its
  turned-away lines were measured on 684-707 of the same shot (`internal/gemini/2026-10-17_head_tube_how_it_works.md`).
- **Nothing gem-matte built is a default.** Each switch is off until its tests pass and the owner says yes.
- **SAM**: core has no memory selection under any name; Meta has it on for SAM 3 base and off for 3.1 with no stated
  reason; its score is a mean over a whole group, so off is plausible for 3.1's 16-object use, and our chain tracks one
  person a call. "Patch it" means a run-time patch like sage, Sol and `sam31_corrections.py`: respect dynamic memory,
  no odd clones. SAM 3 base is downloaded to the owner's model store (`facebook_sam3`), not yet converted.
- **MoGe-2's camera is unreliable on tight close-ups** (field of view spread 46 degrees on 3060-3109, 1.5 on 755-780).
- **The 48 untouched shots were not captured today**; that was the Claude lead's miss (r16n_03).
- **Times**: write them from `date -u`; three of the Claude lead's estimates tonight ran ahead and were corrected.

## Tomorrow, in order

1. Read the night's mailbox from G61 and C118 on. Close what the night left: the hole label (re-worded, then cleared and
   landed); the head tube's READY, its blind read, then THE GATE: the face masks with it on and off on 680-760 and
   1397-1418, no render. Render the holes with it only if her face holds there (the lead 613-974; the lead and the dad
   1397-1418), as candidates.
2. The owner's looks, Start here: 0613g's compare clip at 680-760 first, then 0604h, 0278d and the 0438d control.
   Then r17_06 and r17_07.
3. gem-memsel's measurement (four arms, four scenes,
   `internal/gemini/2026-10-17_brief_build_memory_selection.md`): if memory selection holds her better, it becomes a
   switch in `sam31_corrections.py`'s form. Then the tracker review's item 10 changes, `frames_on_others` first
   (`internal/gemini/2026-10-17_brief_review_subject_track_regain.md`; the review of items 1-4 is
   `internal/gemini/2026-10-17_review_subject_track_regain.md`).
4. Captures of the 48 untouched shots, in song order, in the card's free time.
5. MoGe-2 vendored and landed; the end-to-end tests (`internal/gemini/2026-10-17_test_scenes_agreed.md`,
   `2026-10-17_e2e_tests_gem-matte.md`) run each build through the whole chain before any switch goes on.
6. The 2026-10-17 changelog entry, from the commits: the Claude lead reviews `internal/gemini/2026-10-17_changelog_draft.md`.
   Parked, no default: the matting composite (`internal/gemini/2026-10-17_brief_build_matting_relay.md`, on hold).

## Where things are

- Decisions: `docs/wiki/decisions.md`, 2026-10-17. The owner's preferences: `docs/wiki/working_with_the_owner.md`.
  Day-to-day commands: `docs/wiki/runbook.md`. The lane: `data/AGENTS.md`, then `data/fun/AGENTS.md` (local).
- The owner's raw answers: `data/fun/owner_answers.jsonl`. Who is who: `data/fun/shots.json` and `cast/cast.jsonl`.
- Pages: `Video/fun/TO_REVIEW.html` and `PROVENANCE.html` on the share, from `bench/rebuild_delivery.py`; the review
  server `data/fun/poc/serve.py` (running at this writing).
- The Claude lead's notes: `internal/claude/2026-10-17_mrtabby/` (mentor notes, test scenes, who record).
- Briefs and reviews of the night: `internal/gemini/2026-10-17_brief_*.md` and `2026-10-17_review_*.md`.
