# Handover: read this before anything else

**Rewritten at the end of each working day by the session that led it; the
day before is in git.** This one was written at the close of 2026-10-15 (the
evening, the owner's time) by `mrtall`, the lead, with `mrgo` on the card and
its tooling from the evening (`mrarc`, `mrlol` and `mrvid` held the day and
closed with notes). Where this page and a record disagree, the record is right
(`decisions.md`, 2026-10-15; `data/fun/poc/answers.jsonl`; the newest ledger and
table; the Open List).

## State when closed

- **Two Claude sessions from now on** (the owner): a lead, and one that holds the card and builds its tools
  (`AGENTS.md`, "How the owner and the sessions talk"). The evening's card seat's brief:
  `internal/claude/2026-10-15_brief_mrgo.md`.
- **What stands:** the newest ledger and table under `data/2026-10-15_fun_delivery_tables/` (2145Z at writing, newer if
  the card landed after), and the coverage reading beside it. The owner's pages: `Video/fun/PROVENANCE.html` and
  `TO_REVIEW.html` on the share, served at `http://100.87.141.65:8765/fun_share/`; the interactive review page is
  `http://100.87.141.65:8765/`.
- **Kept today on playback:** the lead's 2823b, 0630d, 0805d, 2625, 751c; the second subject's 5461c, 5114d, 1010j;
  the hall shot 1419-1465 made whole (lead_fp_1419d, third_fp_1419d), which replaced the short pieces there.
- **On the card tonight (mrgo), in order:** the 2823-3109 remake in one window (or to 3166); the context-noise change
  landed, then the second subject's whole kitchen load 604-1009 (seam 898, context 90, noise 0.8, eyes shut drawn on
  938-953 and 968-1009 only); then what time allows. Whatever landed is on the review page; its job note says what.
- **Servers:** ComfyUI was restarted by the lead at 21:52:56Z (pid 3933887) and belongs to the lead's session; the
  review server is the lead's (pid 22533). Both are stopped at the close unless the card is still sampling, in which case
  the card seat stops ComfyUI when it finishes, and says so in its notes.
- **The owner reviewed nothing after about 22:15Z.** Nothing rendered tonight is judged.

## What not to assume

- **One load a subject over the whole shot.** Never fresh loads split inside a shot; chained windows inside one load
  only, on `h3_rules.chain_lengths()`. A load up to `h3_rules.MAX_LENGTH` (362) is one window.
- **The recipe is "A shot, made whole"** (`masked_v2v.md`). Pieces the owner kept are not remade for a better form;
  their explicit calls are (the kitchen, 2823-3109).
- **A text never describes the frozen picture, and says "sings" only where the person sings.** The generator
  (`bench/load_text.py`) writes today's form; a two-subject text is hand-written and the lane is closed.
- **Two subjects in one masked pass is closed** (`../roadmap.md`, "Closed lanes"); multi-person H3 generation is not.
- **A stored window off the chained grid cannot be continued** (2823b's is 277 frames). Check a parent's window before
  offering a continuation.
- **`--no-face-on` blanks the reference only; it does not empty the part** (mrgo's queue).
- **A reviewer's frame list or citation is a pointer, never data.** Today the reviewer's eye spans and four of its
  quoted rules were wrong; read the frames and grep the quote.
- **The owner's eye on playback is the only acceptance**, and a problem the owner has not seen is not theirs to decide
  (the lead's hair, dropped).

## Tomorrow, in order

1. Show the owner what landed overnight: the 2823 remake beside 2823b, the second subject's kitchen shot (its inner
   seam named, 898), and anything else, at the top of the review page.
2. The kitchen's lead: `--no-face-on` emptying the part on face-only loads first; then one hole a pass (a load given the
   second subject's render as its video, the assembler judging it against that base); then the lead over 613-974, one
   window.
3. The rest of `next_steps.md`, "The queue of 2026-10-15": the hall's far-table mother, 1093-1133 hair alone, 4310b,
   5935-6104 (why its reference failed first), the 2050 shot to about 2247, `--continue-from` after a fresh cold read
   (the dad over 1397-1418 only if 1310c's window is on the grid), then the gaps and the shots after 3166.
4. A clean-up job on the wiki: `internal/claude/2026-10-15_wiki_audit.md`.
5. Tooling queued for the card seat, in order after the kitchen's lead: `--mouth-on FIRST-LAST=WHY` (draw the lip lines on
   named frames where the gate dropped them; 2823c's 3098-3103 wait on it, with `--no-face-on 3104-3109`); the ledger
   counting every frame a piece leaves as the original, not only `drop=` frames; the assembler protecting an earlier
   row by what it changed (for one hole a pass); `load_text` writing "sings" span-wise inside one load instead of refusing
   a voice change; whether the Masked Prompt node's voice and frozen-picture wording still renders anywhere.
6. The outside reviewer's C5 (provenance, `internal/gemini/2026-10-15_provenance_c5.md`): two faults fixed at the close
   (today's verdicts taken into `data/fun/verdicts/verdicts.jsonl`; the lead's rebuild now reads the newest table);
   the ledger's "left as the original" is item 5; the kitchen's old pieces stand as placeholders until her whole-shot
   load replaces them.

## Where things are

- Today's rulings: `decisions.md`, 2026-10-15. The postmortem:
  `../research/postmortems/2026-10-15_span_fun-kitchen-and-whole-shot.md`.
- The close-out notes of the day's seats: `internal/claude/2026-10-15_{mrarc,mrlol,mrvid}_closeout.md`; in-flight
  patches copied out of temp folders: `internal/claude/2026-10-15_handoff/`.
- Unlanded worktrees: `mrlol-no-face-on` (`--continue-from`, 71ee3c8c, cold read owed), `mrlol-label` (parked),
  `mrvid-hold` (the hold form, 97512bda, cold read owed), `mrgo-context-noise` (landed or not: its commit says).
- The kitchen's facts: `internal/claude/2026-10-15_mrarc_scripts/kitchen_plan_facts.md`; the reviewer's C4:
  `internal/gemini/2026-10-15_kitchen_approach_c4_mrgem.md`.
- The Open List: `https://claude.ai/artifact/7Jcf7G7jeB9v9kdqRHs1Fm` (today's rows r15_01 to r15_08, all answered or
  closed).
- The lead's scripts (rebuild, pages, joined clips, prefers): `internal/claude/2026-10-15_mrtall_scripts/`.
