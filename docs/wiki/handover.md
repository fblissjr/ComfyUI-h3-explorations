# Handover: read this before anything else

Written for 2026-10-21 by mrtusk (the Claude lead, holding both sides after mrsim handed off). Where this page and a
record disagree, the record is right:
- `data/fun/laid.jsonl`, `data/fun/verdicts/verdicts.jsonl`, `data/fun/owner_answers.jsonl`, `data/fun/shots.json`,
  `data/fun/cast/cast.jsonl`;
- [`decisions.md`](decisions.md), 2026-10-21 at the top;
- `git log`.
The full state of the video, shot by shot: `internal/claude/2026-10-21_final_stretch_status.md`. Its runbook pointer
is `data/fun/FINAL_STRETCH_RUNBOOK.md`, rewritten today from records.

## State when closed

- **Landed today on main:**
  - 7d194514 and cd94b32d: protocol commands and the closed-mouth rule corrected;
  - 10b1939f: `laid.py --owner` required, `set-owner`, and every kept citing the owner's line;
  - 7b7354d2: laid `frames` honored, and `remove --by` keeps a record;
  - d7068b36: the assembler's audio proof compares decoded PCM, which unblocks `--clip`;
  - plus this page and decisions.md.
- **The ledger:**
  - 278-293 is 0278c again. 0278i, which the owner had turned down, is out.
  - The five early second pieces are unconfirmed; their `kept` had come from the old default.
  - 5889e is kept (owner_answers.jsonl:1352), laid 5889-5917.
- **Not yet synced:** the cut, the gap check, TO_REVIEW and PROVENANCE. The rebuild refuses while laid pieces the
  owner called wrong stay in (5768e at :1361 and 4912e at :1366 so far). The interactive review page
  (`http://<tailnet>:8765/`) is current; it reads items.json live.
- **The owner reviewed** the second subject's new pieces today. Six kept her original dark hair: 1093c, 1419c, 4647d,
  4912e, 5011e and 5768e (the readers' read, and the owner's on 5768e and 4912e).
- **In flight:** branch claude-pick-frames (`bench/pick_frames.py`, the adaptive frame picker), one last small round
  after its third blind read cleared it with notes. Worktree `h3_worktrees/claude-pick-frames`.
- **Servers:** the review server on :8765 is up (mrtusk's; the log is internal/claude/serve_2026-10-21.log). ComfyUI is
  down; nothing rendered today.

## What not to assume

- **A tool's "ok", "none" or "PASSES" is not a look at the picture.** Today's audit
  (internal/claude/2026-10-21_control_audit.md and its verification) lists checks that pass while blind: facts_pass on
  an empty track, the dropped-face guard off `--drop` frames, the no-mask excuse.
- **An override is a person's act after a look.** On 10-20 a batch script overrode every blocking flag. That is how
  5768e's own warning was cleared unseen.
- **`--no-face-on` on a second-subject load empties her hair as well as her face.** Where her hair shows, the beard rule
  applies to the hair. Never leave her original head visible inside a window.
- **A reader's verdict is not the owner's.**
  - Gemini reads video at about 1 fps.
  - Claude readers read stills.
  - Gemini's 10-20 cards called all 12 new pieces "clean blonde".
- **"Present" comes from `shots.json`; who is who from `cast.jsonl`.** The who page matches both exactly
  (internal/claude/2026-10-21_who_page_trace.md). Previews seed on one owner box per cut and drop the rest (11
  subject-shots), so read `internal/claude/2026-10-21_mrsim_preview_seeds.md` before trusting a preview's track.
- **The 0110Z cut** ("2026-10-20T0110Z", made 2026-10-21T01:46Z) failed its own check and holds 0278i. Do not use it
  as a Pass-2 source. Rebuild first.

## Next up, in order

1. **When the owner's review pauses, one pass:**
   - every laid piece they called wrong comes out (`laid.py remove --by <answers line>`);
   - `page_answers_to_verdicts.py` takes the new lines (`--dry-run` first);
   - then `bench/rebuild_delivery.py --video fun --clip`, `bench/build_cut_map.py`, and the copy of poc's PROVENANCE
     and CUT_MAP to the share with backups.
2. **Land claude-pick-frames** once its last round passes a quick re-check. Then point docs/masked_v2v/AGENTS.md §9 at it,
   and correct §9's 5889e account (she is visible 5916-5917, gone from 5918).
3. **On the owner's go, one test render:** second_hfp_5768e at seed 730451892, with her hair kept in the region on
   5768-5773 (no `--no-face-on` there, no overrides). If blonde, remake the six the same way; 4912e needs its own look.
   Small faces (4647d: 19.7 px on the canvas, about 40 px at the 2.0x cap) also need more zoom. The lead's insets
   worked at 2.0x only because their faces were 56-76 px. `video_mask.py::ZOOM_LARGEST` = 2.0 has no per-load
   input, so a per-load zoom is a build. The owner (2026-10-21): barely visible; note it as an issue, and build only
   if it is easy. Low priority. For 4647d the owner's word is to replace the hair only, with no face work and no zoom
   build (owner_answers.jsonl, 2026-10-21T16:14:59Z): remake it with the hair fix alone, alongside the others.
4. **Open List rows, from internal/claude/2026-10-21_tools_to_diagnostics_plan.md:**
   - stop-and-show or print-only;
   - one seed per cut (R8);
   - who may override (R10);
   - the no-override rules (R3-R5).
   Then the owner's picks among builds W3d (pages and lineage), W3g (print what the plan replaces, and the prompt),
   W3h (who page), W3b, W3c and W6 (docs).
5. **Then Pass 2 and the remaining lead shots,** per the final-stretch status file: 23 unblocked single-subject lead
   previews, three re-seeds (0168, 0516, 6105), and 1189-1221, which has no preview.

## Where things are

- The day's working records: `internal/claude/2026-10-21_*`, plus the mailboxes `..._mrtusk_to_mrsim.md` /
  `..._mrsim_to_mrtusk.md` and `internal/gemini/2026-10-21_*`.
- The lane's rules: [`docs/masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`PROTOCOL.md`](../masked_v2v/PROTOCOL.md)
  and [`data/AGENTS.md`](../../data/AGENTS.md); `data/fun/AGENTS.md` (local).
- Mentor notes on the solo days and today: `internal/claude/2026-10-21_mrtusk/mentor_notes.md`.
- Worktrees to remove at the close, once each is checked landed and clean: `h3_worktrees/claude-control-audit`
  (scratch), mrsim-laid-owner, mrsim-laid-frames and mrsim-audio-pcm.
