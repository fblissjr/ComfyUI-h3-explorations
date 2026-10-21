# Handover: read this before anything else

Rewritten 2026-10-21 at about 20:45Z by mrturtle (the Claude lead after mrtent), with mract on the card (after mrgordon)
and Gemini (mrgemx) building and reviewing. Where this page and a record disagree, the record is right:
- `data/fun/laid.jsonl`, `data/fun/owner_answers.jsonl`, `data/fun/verdicts/verdicts.jsonl`, `data/fun/shots.json`,
  `data/fun/cast/cast.jsonl`;
- [`decisions.md`](decisions.md), 2026-10-21 at the top;
- the owner's rules at the top of [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md);
- `git log`.
Running notes: `internal/claude/2026-10-21_mrturtle/notes.md`. Mailboxes: `internal/claude/2026-10-21_mrtent_to_mrgordon.md`
(the lead's T-entries, now to mract) and `..._mrgordon_to_mrtent.md` (the card's S-entries); `internal/gemini/2026-10-21_*`.

## State at this writing

- **The card is on hold** (T46): no reference, job or render. The owner's go while they are away (T55): renders may queue
  once four gating builds are cleared and landed, the prep is remade once, fresh, on main's code, and each load passes
  its full prompt read, a second reader and the gate. Nothing is laid without the owner's keep.
- **Five builds, each through the gate** (its own branch, checks with breaks shown red, a blind read, the lead's
  clearance, the card lands it by pathspec from the merge base):
  - `claude-headtube-hair` (a lead's builder): the head tube re-seeds a head only on the subject's own evidence; the
    Hair relabel only for a subject who replaces hair; a subject's replaced parts are never a stranger's. Gates the prep.
    Brief `internal/claude/2026-10-21_mrturtle/headtube_strangers_brief.md`; Gemini blind-reads it.
  - `claude-seed-every-box` (a lead's builder), trimmed by the owner to its core: every owner box seeds its tracker, and
    the prep step knows which replacement it is for (5011 needs no override). If it touches `subject_track.py`, its
    landing needs a ComfyUI restart. Gates the prep. The stale-cut check, the chain record and the no-reuse refusals
    move to the next day; tonight the prep is remade fresh by hand.
  - `mrgemx-eyes-down` (Gemini): a person's word that draws lowered eyes as the 2026-10-20 arc on chosen frames (for
    0415). Gates the prep. Brief C38.
  - `mrgemx-window-overlap` (Gemini): windows only on both clocks, real context frames, zoom across a cut. Node code
    (`video_mask.py`): land it back to back with the seeding build, one restart. Gates the prep. Brief C38.
  - `claude-change-check`: parked unlanded at df2bf603 (the owner). The fix instead corrects the existing measurement that
    compares a replaced part to the source; the lead traces which tool says it.
- **Today's previews are provisional** (T52, T53): remade once after the four gating builds land.
- **Kept today:** second_hfp_5768f (the hair fix holds); 4602c, 0484f, 5675c, 5461e, 4726c. Out today: see decisions.md.
- **Servers:** ComfyUI :8188 (mract's; stop when the card work is done, by ear). The review server :8765 (pid 17874) is
  the lead's to stop at 22:00Z (the owner's 5 pm), by the recipe.

## What not to assume

- **The cut on the share (1714Z) is behind laid.jsonl.** The PROVENANCE page says so. No `--clip` rebuild until a batch of
  keeps.
- **The closed-eye drawing is reverted** (c076fd87). A lowered eye is the 2026-10-20 arc again. 0415 and 1039 stay
  unsolved: 0415 waits on the lowered-eye span; 1039's body-model head stays upright while she bows (parked).
- **4912's dark hair** is her own hair crown handed to the strangers keep-out on 4939-4946, plus the held tail frames of
  4946; H3 redrew her hair in a new shape but dark. Its remake waits on the head-tube build.
- **The beard-era checks look at the face only, inside the subject's outline.** They cannot see a replaced hair part
  outside it. The follow-up is a flag over every part the plan replaces.
- **A reader's read is never the owner's.** And the lead's figures compared hair to the source: wrong for a replaced part.

## Next up, in order

1. Clear each build as its READY lands (`internal/claude/2026-10-21_ready_*.md`, Gemini's G-entries): read the diff,
   re-run a check from its worktree, read its blind review, then a CLEARED T-entry; mract lands it.
2. After the four gating builds: the prep remade once, then renders in T55's order: the 0415 eye test; 4912; the mom's
   remakes (4479, 5114, 5011, 0278 on the raw copy; 5533, 4647 on the 1714Z cut). The owner trimmed tonight to those;
   the dad's second passes (4602, 4726, 5461, 5675) and the lead's shots from the gap page (data/fun/poc/gaps.html;
   answers at owner_answers.jsonl:1402-1465) move to the next day.
3. At the close: the changelog from the commits (changelog.d/README.md); decisions.md; archive today's
   `data/fun/laid.jsonl.before_*` backups to `data/fun/archive/2026-10-21/`; remove landed worktrees.
4. Builds queued after these: the seeding build's trimmed parts (the stale-cut check, the chain record, the no-reuse
   refusals) and W7 (reuse and staleness made visible, reading that record), the per-part flag above,
   `laid.py set-owner --recite`, whole-video prep for first-pass subjects, 1039's head pose, the bigger zoom (parked).

## Where things are

- Read-ins: `internal/claude/2026-10-21_readin_next_lead_2.md` (the next lead), `..._readin_next_card.md` (the card).
- The gap census and page: `internal/claude/2026-10-21_mrturtle/gaps.md`, `data/fun/poc/gaps.html`.
- Today's traces and figures: `internal/claude/2026-10-21_mrturtle/` (d4912_hair_vs_face, eyes_mean_vs_main_*,
  third_face_list), `internal/claude/2026-10-21_g10_verify.md`, Gemini's G23 and G24.
- The archive of stale data docs: `data/fun/archive/2026-10-21/INDEX.md`.
- Worktrees to remove at the close, once each is checked landed and clean: mrgordon-cut-map-a, claude-docs-w6,
  claude-noface-hair, mrgemx-eyes-mean, claude-eyes-closed (never landed), and the earlier sessions' list in
  `internal/claude/2026-10-21_mrgordon/handover_card.md`; mrgem-cut-map is Gemini's: ask it.
