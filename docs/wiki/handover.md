# Handover: read this before anything else

**Rewritten at the end of each working day by the session that led it; the day before is in git.** Written at the close
of 2026-10-15 (the owner's evening; past midnight UTC) by `mrtall`, the lead, with `mrgo` on the card; reconciled with
three reviews (`internal/claude/2026-10-16_{mrgo,mrnight,coldreader}_whats_broken.md`) and a cold read of this page as
tomorrow's lead. Where this page and a record disagree, the record is right (`decisions.md`;
`data/fun/verdicts/verdicts.jsonl`; the newest ledger and table; the Open List).

## State when closed

- **The owner closed everything for the night** ("id rather close everything off and shutdown for the night"). ComfyUI
  (pid 3933887) is stopped by mrgo with the queue empty, and the review server (pid 22533) by the lead; if either
  still runs, its owner's note says why.
- **Landed on the card:** `second_hfp_0604g`, the mom's whole kitchen load (604-1009, two windows meeting at 898,
  context 90, noise 0.8 on the inner window), at 00:38:10Z. Its blind render reader
  (`data/2026-10-15_second_hfp_0604g/look_render/reading.md`) and the outside reviewer's watch (a summary to the owner,
  no file) agree: one person across 898 and through her turn and bow; hair short at her shoulders. They name: her
  likeness drifting between the halves (fuller and older on 604-771, younger on 952-1009); eyes NOT closed where the
  reference draws them shut on 938-953 (she opens them turning to camera on 947-953); 752-771 eyes open where the
  source is bowed (not drawn shut there); 955-967 and 676-747 not following the source's turn. Stills and a reviewer's
  watch, never the owner's acceptance.
- **NOT rendered:** `lead_fp_2823d` (2823-3109, one window). Planned, its text read in full (sha256 b91721257e72b4a1),
  its overrides f001 and f002 written by mrgo; its plan reader stopped at the close. mrgo's note says where it reached
  and the commands to render it: `internal/claude/2026-10-15_mrgo_closeout.md`.
- **Landed in code today** (each read cold, cases seen red): context_noise per load (3e5ea8f1); `--track-not` held to
  its preview track (cf3ab288); `--mouth-on` and the read-back fix (1e4207c4); every run's record keeps the server's
  cached nodes (1ff12a36). Every check on main read at the close: 131 green, 13 not graded, 1 red (a stale INDEX row,
  rebuilt in 3647a1b2). The changelog entry of 2026-10-15 is in (1ed9da30, 0.288.0).
- **Kept on playback on 2026-10-15:** the lead's 2823b, 0630d, 0805d, 2625, 751c; the mom's 5461c, 5114d, 1010j; the
  hall 1419-1465 made whole (lead_fp_1419d, third_fp_1419d).
- **The owner's pages** were rebuilt at the close from table `2026-10-16T0043Z`; Start here lists 0604g's candidate
  clip (516-1038, the mom's new piece laid alone; `prefers.txt` still lays the old pieces) and its stacked render.
- **The outside reviewer was stopped by the owner** before answering C17 (what is broken) and C18 (the restore
  carry's diff). Its C13, C14 and C16 answers are in `internal/gemini/`.

## What not to assume

- **PROVENANCE understates where the original shows**: `render_ledger.left_by_the_load` (`render_ledger.py:365-380`)
  reads only `drop=`, not frames the floor emptied (`parts_emptied_by_grade_on_source_frames`; 15 in the kept hall cut).
- **What the cut lays is decided in four places**: the owner's keeps; `internal/claude/2026-10-15_mrtall_scripts/
  prefers.txt` (there is no `holds.txt`; `holds.pending.txt` holds two eye tests, never laid); two lines typed into
  that folder's `rebuild.sh`; and the newest table by file time. `rebuild.sh` writes into
  `data/2026-10-15_fun_delivery_tables/` whatever the day.
- **A gap in the table is counted against the whole shot, not the frames she is tracked in**, and two renders of one
  person meet unflagged. In the kitchen the lead flips to the original on 681-750, 779-804 and 834-851 (visible); its
  gaps 604-629 and 966-1009 are frames she is barely in.
- **Four job records point at a reference they never read** (0751b, 1097b, 1156, 5114d; originals in
  `references/*_SUPERSEDED_*`, 5114d's proven by sha256).
- **`--no-face-on` blanks the reference only; it does not empty the part.** Until the gate branch lands, every such span
  on a face-only load also takes `drop=` by hand, which bypasses the job step's checks: say it in the job's note.
- **Two job-step lines mislead**: the track-over-track advice repeats after `--track-not`, and "hair band: none" prints
  on loads that carry one in the clothing keep-out.
- **The full read of a render's text is of the QUEUED PLAN graph** (`bench/what_h3_got.py --stage plan`); the render
  step builds and queues in one step and is held to the plan graph.
- **A check's colour is read in context, never as the verdict.** What breaks is reverted; what lands is an obvious win
  to the lead, another Claude session and the outside reviewer (`decisions.md`, 2026-10-16).
- **The face points say "face" on the back of a head and through hands** (the fill dry run); never gate on them alone.
- **A text says what is drawn, never what is not, and never the frozen picture**; "sings" only where the person sings.
- **One load a subject over the whole shot**, chained windows on `h3_rules.chain_lengths()`, ceiling 362.
- **A reviewer's quote or figure is a pointer, never data**; copy quotes from the file by line.
- **`git add -A` in a worktree sweeps in `INDEX.md`** (a check rewrites it there). Pathspec commits only.

## Tomorrow, in order

1. **Read, whole, before acting:** this page; `AGENTS.md` (two sessions; the text rules); the entries at the top of
   `decisions.md` (2026-10-15 late and 2026-10-16); the three reviews named above and
   `internal/gemini/2026-10-16_c16_fill_rule.md`; `internal/claude/2026-10-15_gaze_pieces.md`;
   `internal/claude/2026-10-15_mrgo_closeout.md`; `internal/claude/2026-10-15_mrnight_fixes_note.md`, "STATE FOR
   TOMORROW".
2. **Start the servers**: ComfyUI by the recipe in `docs/comfy_notes.md`; the review server from the repo root,
   `python3 data/fun/poc/serve.py` (port 8765).
3. **The outside reviewer**: in `internal/gemini/2026-10-16_requests.md`, re-post C18 (`internal/claude/
   2026-10-16_restore_hold_all.diff`) and C17 rescoped to THIS page; a watch of each render the owner will see.
4. **Write the card seat's brief** from items 5-11, with the landing bar.
5. **The owner reviews first**, from Start here on `TO_REVIEW.html`: (a) 0604g's candidate (516-1038), with the
   readers' places to look above; keep it or not; (b) once rendered, `lead_fp_2823d` beside the kept 2823b
   (`bench/stack_rows.py`: masked input, 2823d, 2823b). Say on the page that 3098-3104 lose her glasses in the render
   and get them back only at assembly (item 9). Pages: `rebuild.sh`, then `pages.sh STAMP
   data/2026-10-15_fun_delivery_tables/t_fun_0000_6394_2026-10-16T0043Z.json`.
6. **The card renders 2823d first**, by the commands in mrgo's note, after its plan reader is finished and checked.
7. **First tooling (the owner's yes): one per-frame coverage tool and one source of truth.** Per shot, subject and
   frame: tracked or not (SAM 3.1), which piece lays it, whether the original shows, and where one tracker lost her
   while the others had her (the flip survey's method, `internal/claude/2026-10-15_flip_survey/`). Three lists: visible
   gaps, seams between renders (with both render days), frames emptied inside a piece; the ledger's
   `left_by_the_load` fixed by it. With it `data/fun/laid.jsonl` as the ONLY thing the rebuild reads for what is laid
   (span, subject, load, sha256 as laid, why), both pages saying what they were made from, and a kept piece whose
   bytes change put back on TO_REVIEW as "changed since you kept it".
8. **The gate's one rule** (branch `mrnight-fixes`, e72e85af + 6ffd21ac). FIRST rerun the old/new comparison with the
   branch's OWN job step, since every "neck from behind" frame was a `--no-face-on` frame on a capture made before the
   branch's trim: the harness is `internal/claude/2026-10-15_gate_compare/` (`run_gate.py`, `compare.py`, README);
   remake each compared job with the worktree's `bench/shot_to_load.py job` into a scratch data folder, then run it.
   Only then the chin cut (mrnight's note, "Next commands"). Rerun the comparison after every change. Two more from
   mrnight's review, before it lands: (a) reasoned, not run: the capture's own hold may refill her face at the edge
   of the trimmed hair-only span, undoing the trim; (b) a DECISION for the owner and the lead: a frame held by a
   person's word with no face drawn (2823d 3104-3105, the lead's ruling B) is flagged a hole by the rule's letter, so
   every remake of such a load needs an override, the routine the rule replaced. Agreement by word, or a standing
   override?
9. **The restore carried across a hold** (branch `mrgo-restore-hold`, read cold three times, assembly only), after the
   reviewer's C18; its first rebuild prints which laid pieces change, and each goes back to the owner.
10. **References never renamed once read** (unbuilt): in `shot_to_load`'s reference step a remake takes the next free
    name unless `--existing`, and verify fails where a recorded sha256 no longer matches. Optionally `SHORT_FILL` from
    2 to 4 keeping "nothing labelled" (the reviewer's C16), after 8.
11. **The kitchen's lead (613-974, one window)** only after: the gate branch (8), one hole a pass (her load made on the
    mom's render; `next_steps.md`), and the owner keeping 0604g in 5(a).
12. Then the rest of `next_steps.md`, "The queue of 2026-10-15"; the gaze test set (`2026-10-15_gaze_pieces.md`, the
    touch-nothing test first, every other gaze piece set aside; the gaze branch's red `check_body_pose` is the check's
    to fix, not the code, and its `mesh_file=True` preview default is dropped from any landing until a cut is ruled the
    default: mrnight's review); the two misleading job-step lines; the wiki clean-up
    (`internal/claude/2026-10-15_wiki_audit.md`).

## Where things are

- Rulings: `decisions.md` (the late entries at its top). Postmortem:
  `../research/postmortems/2026-10-15_span_fun-kitchen-and-whole-shot.md`.
- Tonight's investigations: `internal/claude/2026-10-15_cache_check/` (no stale cache found),
  `2026-10-15_flip_survey/` (every flip is the part model plus our floor), `2026-10-16_fill_dry_run/`,
  `2026-10-15_gate_compare/`.
- Worktrees (beside ComfyUI's `custom_nodes`; "ahead" of main may mean landed by cherry-pick; the notes say):
  `mrnight-fixes` (the gate; item 8) and `no-face-on` (mrlol's `--continue-from`, cold read owed) both edit
  `shot_to_load.py` and `load_chain.py`; `mrgo-restore-hold` (item 9) and `mrvid-hold` (the hold form, cold read owed)
  both edit `assemble_delivery.py`; `mrnight-gaze` (gaze, not landed); `ref-label` (parked); `mrgo-context-noise`,
  `mrgo-mouth-on`, `mrgo-track-from` (landed).
- The Open List: `https://claude.ai/artifact/7Jcf7G7jeB9v9kdqRHs1Fm`. The lead's scripts:
  `internal/claude/2026-10-15_mrtall_scripts/`. The outside reviewer: `internal/gemini/`.
