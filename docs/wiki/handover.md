# Handover: read this before anything else

**Rewritten at the end of each working day by the session that led it; the day before is in git.** This one was written
at the close of 2026-10-15 (the owner's evening; past midnight UTC) by `mrtall`, the lead, with `mrgo` on the card, and
reconciled with four reviews of the owner's question "what is broken, and what should the handover drop, add and
change": `internal/claude/2026-10-16_{mrgo,mrnight,coldreader}_whats_broken.md` and the outside reviewer's
`internal/gemini/2026-10-16_c17_whats_broken.md` (if it is not there, it was still being written; read it first).
Where this page and a record disagree, the record is right (`decisions.md`; `data/fun/verdicts/verdicts.jsonl`; the
newest ledger and table; the Open List).

> **Gaze: the touch-nothing test first; every other gaze piece is set aside until it answers** (the owner, 2026-10-16).
> Nothing of the gaze build landed. The pieces, why each, and the test set: `internal/claude/2026-10-15_gaze_pieces.md`.

## State when closed

- **Two Claude sessions** (the owner): a lead, and one on the card and its tools (`AGENTS.md`). The evening's card seat:
  `mrgo`, brief `internal/claude/2026-10-15_brief_mrgo.md`, notes `internal/claude/2026-10-15_mrgo_closeout.md`.
- **On the card overnight, in order:** `second_hfp_0604g` (the second subject's whole kitchen load, 604-1009, seam 898,
  context 90, noise 0.8 on the inner window, eyes shut drawn on 938-953 and 968-1009, `--track-not lead:1003-1009`),
  rendering at writing; then `lead_fp_2823d` (2823-3109 in one window; 3098-3103 mouth drawn on a word, 3104-3105 held
  with no face lines, 3106-3109 dropped) on its existing text (sha256 b91721257e72b4a1, read in full), after its blind
  reader. Both texts read in full from the queued plan graphs. Whether each landed: mrgo's note and the newest ledger.
- **Landed today in code** (each read cold, its cases seen red): context_noise per load (3e5ea8f1); a `--track-not`
  copy held to its preview track (cf3ab288); `--mouth-on` and the read-back's edge fix (1e4207c4). The cached-node
  record (branch `mrtall-cached-nodes`, 1e3f95c8) lands after 0604g's land step on mrgo's clear; if it did not, it is
  item 6 below.
- **Kept on playback on 2026-10-15:** the lead's 2823b, 0630d, 0805d, 2625, 751c; the second subject's 5461c, 5114d,
  1010j; the hall shot 1419-1465 made whole (lead_fp_1419d, third_fp_1419d).
- **The owner's pages:** `Video/fun/PROVENANCE.html` and `TO_REVIEW.html` on the share, served at
  `http://100.87.141.65:8765/fun_share/`; the interactive review page `http://100.87.141.65:8765/`. Both servers are
  stopped at the close (ComfyUI is the lead's, started 21:52:56Z; the review server is `python3 data/fun/poc/serve.py`
  from the repo root); if the card was still sampling, mrgo stops ComfyUI and says so.
- **The owner reviewed nothing rendered after about 22:15Z.**

## What not to assume

- **PROVENANCE understates where the original shows.** `render_ledger.left_by_the_load` (`render_ledger.py:365-380`)
  reads only `drop=` frames, not frames the part floor emptied (`parts_emptied_by_grade_on_source_frames`): the kept hall
  cut has 15 such frames (1419d 1421, 1449-1459, 1464-1465; third 1419d). Its docstring claims more than it reads.
- **What the cut lays is decided in four places**, not one: the owner's keeps, a session's `prefers.txt`/`holds.txt`
  (five dated copies under `internal/claude/`), two lines typed into the lead's `rebuild.sh`, and the newest table by
  file time. Until item 2 lands, read all four before trusting a page.
- **Four job records point at a reference they never read:** 0751b, 1097b, 1156, 5114d (each remade under the same
  name after its render; originals in `references/*_SUPERSEDED_*`; 5114d's proven by sha256). Any reading that paired
  1097b's render with today's file at that path used the wrong file.
- **A gap in the table is counted against the whole shot, not the frames she is tracked in**, and the table lays two
  renders of one person side by side without flagging the seam between them (the kitchen's 949).
- **`--no-face-on` blanks the reference only; it does not empty the part.** Until item 3 lands, every `--no-face-on`
  span on a face-only load also takes `drop=` by hand (and a hand-written `drop=` bypasses the job step's checks and is
  not printed among the defaults; say it in the job's note).
- **Two job-step lines mislead** (mrgo's review): the track-over-track advice repeats after `--track-not` is given, and
  "hair band: none" prints on loads that carry one in the clothing keep-out.
- **"Read what_h3_got before it samples" cannot be met as written** (the render step builds and queues in one step):
  the full read is of the QUEUED PLAN graph; the render step holds the render graph to it.
- **A check's colour is read in context, never as the verdict** (the owner, 2026-10-16). What breaks is reverted; what
  lands is an obvious win to the lead, another Claude session and the outside reviewer.
- **The face points do not tell a face from a turned-away head or hands** (the fill dry run: "face shows" on 28 of 29
  turned-head frames and all 74 hands frames). Never gate anything on their presence alone.
- **A text says what is drawn, never what is not** (the owner, 2026-10-16), and never the frozen picture; "sings" only
  where the person sings.
- **One load a subject over the whole shot**, chained windows inside it on `h3_rules.chain_lengths()`, ceiling 362; two
  subjects in one masked pass is closed; continuing a piece only from one the owner kept and on the grid.
- **A reviewer's frame list, figure or quote is a pointer, never data** (the outside reviewer misquoted code three times
  on 2026-10-15). Copy quotes from the file by line.
- **Running `git add -A` in a worktree sweeps in `INDEX.md`**, which a check rewrites there. Pathspec commits only.

## Tomorrow, in order

1. **The owner reviews first.** Check what landed overnight (mrgo's note, the newest ledger), rebuild (`internal/claude/
   2026-10-15_mrtall_scripts/rebuild.sh`, then `pages.sh`), and put at the top of the review page and Start here:
   (a) the second subject's kitchen 604-1009 as a joined clip with the new piece laid alone for her (a candidate; the
   table changes only on the owner's keep), with what to watch: her identity across 898, her turn and bow 945-1009, her
   eyes on 950-970 (her face is about 87 px in the reference, so shut eyes may not carry); (b) `lead_fp_2823d` beside
   the kept 2823b (`bench/stack_rows.py`: masked input, 2823d, 2823b), keep one. Give the owner the direct links.
2. **First tooling, the owner's yes of 2026-10-16: one per-frame coverage tool, and one source of truth.** For every
   shot and subject, per frame: tracked or not (SAM 3.1), which piece lays it, whether the original shows, and the
   frames where one tracker lost her while the others had her (the part model: present, under the floor, emptied,
   held, dropped by a word; the face points; the body model), from the flip survey's method
   (`internal/claude/2026-10-15_flip_survey/`). From it three lists: real gaps (tracked and the original shows), seams
   (two renders of her meeting, with both render days), frames emptied inside a piece. The ledger's
   `left_by_the_load` is fixed by it. With it, `data/fun/laid.jsonl` as the ONLY thing the rebuild reads for what is
   laid (one line a piece: span, subject, load, sha256 as laid, and why), and both pages saying at the top what they
   were made from. An assembly change that alters a kept piece's bytes puts it back on TO_REVIEW as "changed since you
   kept it" (the owner's yes).
3. **The gate's one rule** (branch `mrnight-fixes`, e72e85af + 6ffd21ac; with `--no-face-on` emptying her face).
   FIRST rerun the old/new comparison with the branch's OWN job step (the cold reader: every "neck from behind" frame
   was a `--no-face-on` frame on a capture made before the branch's trim, so the blocker may come from the test);
   only then the chin cut, if still needed. State and commands: `internal/claude/2026-10-15_mrnight_fixes_note.md`,
   "STATE FOR TOMORROW"; the comparison `internal/claude/2026-10-15_gate_compare/`. Rerun it after every change below.
4. **The hold rule: NOT the widened one** (the outside reviewer's C16, `internal/gemini/2026-10-16_c16_fill_rule.md`,
   on the dry run): the face points cannot gate it (true on 28 of 29 turned-away frames and all 74 hands frames), and
   they do not exist yet when the capture runs (the pose pass is the reference step, `shot_to_load.py:2033`). What
   already tells a real part-model dropout from a turned head or hands is main's own `under_doubted` read of the class
   map: every real dropout on file (751b 761; 2823d 3098-3104; 5935b 6008, 6012) reads "nothing labelled", a turned
   head "their own hair", hands "their other classes". Main's `SHORT_FILL = 2` broke nothing in the dry run. If wanted:
   raise it to 4 keeping "nothing labelled" and standing neighbours inside the load, with 2823d 3101-3104 as its case,
   after 3, rerunning the comparison. The tracking count (item 2) is what shows how often a longer run happens.
5. **A reference is never renamed once a render has read it:** a remake takes the next free name unless `--existing`;
   verify fails where a recorded sha256 no longer matches the file.
6. The cached-node record, if it did not land overnight; and the restore carried across a hold (branch
   `mrgo-restore-hold`, read cold three times, assembly only), after the outside reviewer's C18. Its first rebuild
   prints which laid pieces change; each goes back to the owner.
7. The kitchen's lead (613-974, one window) once 3 lands, then one hole a pass.
8. The rest of `next_steps.md`, "The queue of 2026-10-15", and the gaze test set (`2026-10-15_gaze_pieces.md`).
9. The two misleading job-step lines above; `load_text` writing "sings" span-wise; the Masked Prompt node's old wording;
   the wiki clean-up (`internal/claude/2026-10-15_wiki_audit.md`).

## Where things are

- Rulings: `decisions.md`, 2026-10-15 (the late entries are at its top). The postmortem:
  `../research/postmortems/2026-10-15_span_fun-kitchen-and-whole-shot.md`.
- Tonight's investigations: the cache check `internal/claude/2026-10-15_cache_check/README.md` (no stale cache found;
  `server_history_cached_2026-10-16T0013Z.json` holds the server's own list for the runs it remembered); the flip
  survey `internal/claude/2026-10-15_flip_survey/README.md` (every flip is the part model plus our floor; not SAM, not
  reused masks, not the other person); the fill dry run `internal/claude/2026-10-16_fill_dry_run/`; the gate comparison
  `internal/claude/2026-10-15_gate_compare/`.
- Worktrees, in the worktrees folder beside ComfyUI's `custom_nodes` (a branch "ahead" of main may have landed by
  cherry-pick; its note says):

  | branch | what | state |
  |---|---|---|
  | `mrnight-fixes` | the gate's one rule, `--no-face-on` empties her face | built, not landed (item 3) |
  | `mrnight-gaze` | the gaze pieces | built in part, not landed; touch-nothing test first |
  | `mrgo-restore-hold` | the restore carried across a hold | read cold three times; waits on C18 (item 6) |
  | `mrtall-cached-nodes` | cached nodes in every run's record | read cold; lands after 0604g (item 6 if not) |
  | `no-face-on` (mrlol) | `--continue-from` | cold read owed |
  | `mrvid-hold` | the hold form | cold read owed |
  | `ref-label` (mrlol) | the written label | parked |
  | `mrgo-context-noise`, `mrgo-mouth-on`, `mrgo-track-from` | | landed |

- The Open List: `https://claude.ai/artifact/7Jcf7G7jeB9v9kdqRHs1Fm`. The lead's scripts (rebuild, pages, joined clips,
  prefers): `internal/claude/2026-10-15_mrtall_scripts/` (date-pinned; item 2 retires their prefers and holds).
- The outside reviewer's files: `internal/gemini/` (requests and outbox by date).
