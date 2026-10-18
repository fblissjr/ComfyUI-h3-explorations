# Handover: read this before anything else

**Rewritten at the end of each working day by the Claude lead; the day before is in git.** Written for 2026-10-18 by
`mrted` (the Claude lead) at 22:05Z, while the card still ran; the Gemini lead `mrgemson` adds the close's state at the
end of "State when closed" and hands off to a new Gemini lead, who starts from this page. Where this page and a record
disagree, the record is right (`decisions.md`, newest first; `data/fun/laid.jsonl`; the newest gap check; `git log`).
Today's mailbox: `internal/gemini/2026-10-18_claude_to_gemini.md` (C1-C127) and `2026-10-18_gemini_to_claude.md`
(G1-G21); orders to the drawing builder: `internal/gemini/2026-10-18_mrted_to_gem-mouth.md` (M1-M21).

## State when closed

- **Kept and laid today** (the owner's keeps; `data/fun/laid.jsonl`, 38 lines): the mom's whole kitchen
  `second_hfp_0604h` (604-1009; recipe frozen in `bench/results/2026-10-18_second_hfp_0604h_recipe.md`, why it worked
  in `adf51858`'s postmortem), the lead's kitchen `lead_fp_0606`, the lead's hall `lead_fp_1306t`, the lead's street
  spin `lead_fp_0438e`.
- **Landed** (`git log --since=2026-10-18T04:15Z`): MoGe-2 depth for who is in front where tracks overlap, behind
  `--depth`, off by default (`1c213f41`, `3980ed5c`, `01b75141`); the tracker cuts frames that sit on another person
  before regain (10a, `f6281edc`, MASK_VERSION 15); the head tube no longer takes back off-head islands, the dad's bicep
  patch among them (`ffca8acd`); the pages rebuild themselves when `laid.jsonl` is written, with a stale-page banner
  (`6508f64e`); the reader's four-field card on Start here (`d3911162`); the mouth shape behind `--closed-lip-line`, off
  (`6ead729f`); the comfy-kitchen kernel fix for nanobind 2.10.x (`2e6f0d04`, `fd34e1aa`; upstream PR
  Comfy-Org/comfy-kitchen#256); the read-in order and one-lead rule (`bcf73037`).
- **Two root causes found, each with its control render** (`decisions.md`, 2026-10-18):
  1. The dad's glasses and grey beard in `third_fp_1306d`: his original face showed inside the window (under his fist
     on 1306-1309; the dropped 1409-1413), and H3 copies what the window holds. `third_fp_1340b`, the same reference on
     clean frames, came out 0/21 glasses and 0/21 beard. The floating contrast eye in profile then shaped the glasses.
  2. The mom's open mouth on 278-293: the drawn closed mouth sat 8-12 px above the grey head's own sculpted lips, which
     reads as parted lips. `second_hfp_0278f`, with no mouth drawn, kept her lips closed on all 16 frames.
  gem-dad measured the drift between the drawn points and the grey head on 45 references (3,503 frames;
  `internal/gemini/2026-10-18_feature_drift_finding.md`): the zoom, the camera focal and Meta's box enlargement each
  move it by 2 px or less. It is the grey head's generic face and pose, so the fix is in the drawing.
- **The drawing is the main line** (the owner, about 21:50-22:00Z; M10-M21): every mark clipped to the grey head's
  silhouette (a far eye shrinks into the edge, never switches off); the grey head a little darker with its shading
  kept; filled eyes; lines at today's width; no dots; no colour per part (the owner sided with gem-smear: cleaner, not
  pose-model colours); the mouth decided per WINDOW by voice, none when silent, drawn on every face frame when the
  person sings or talks, with the grey head's own lips covered and the mouth kept above its chin.
- **In flight at writing:** the lead's kitchen chin 751-785 rendered against the cut that has 0604h laid (C106; the
  smear was 0606 rendered beside the original mom's dark hair); gem-mouth's drawing build (the owner sees the reference
  sheet before any of its renders); `gem-review-kept-today` (C111, Kept today on the review page, old sections out);
  gem-track's Sapiens2 Pointmap 1B/5B and MoGe-2 check on 1306-1309, CPU only
  (`internal/gemini/2026-10-18_pointmap_finding.md` when done). `third_fp_1306e` rendered before the owner's skip
  reached the card: not to be laid.
- **Closed** (`docs/roadmap.md`, "Closed lanes"): Sapiens matting, SAM 3 base. Not again: a reference driven by core's
  SAM 3D Body face expression (broke the mouth, eyes and iris before).
- **Dropped by the owner today:** the dad's beard work (the region grown over his beard, the full hall remake, the
  2050-2622 window test); wider lines; colour per part. The dad's hall stays as laid (`third_fp_1310c`, with its gaps).
- **Open List:** r18_05 (which kept pieces to remake with the old mouth) waits on the drawing test; r18_02 and r18_03
  are answered with conditions (below); r18_06 is tonight's state.

## What not to assume

- **A drawn mouth is not neutral.** Drawing a closed mouth opened hers in 0278d, 0278e and 0278h; taking the mouth out
  of the text alone (0278h) did not close it. The 24 older kept pieces drew no mouth and never had this.
- **A leak check that counts face labels only is blind.** Sapiens labels a beard under a fist as hand and a turned
  beard as hair (1306-1309, 1408-1411); gem-track's shrinking-face measure (`pinched_head_face`) catches it.
- **A dropped stretch is not the back of the head because a label says so**: 1412-1413 show the dad's bearded profile.
- **Old tracks predate today's tracker fixes.** A shot captured before today gets a fresh first preview, never a
  re-run on its saved .mkv (10a and regain run only in the preview).
- **The head tube's default needs one occluded shot.** The drawing tests are unoccluded faces and show nothing about
  it; one hand-or-mic-over-face shot (2050-2622 or 516-603, gem-track) must read clean first.
- **Off by default until the owner says yes:** the head tube, `--closed-lip-line`, `--depth`, and every part of the
  drawing build. THE DEFAULTS THIS LOAD CARRIES in each job step is the observable.
- **The kernel venv still has nanobind 2.10.x.** The fork's `.none()` guards work around it; take 2.11 at the next
  kernel rebuild (`docs/comfy_notes.md`, "After a ComfyUI update").
- **Times from `date -u`.** The Claude lead wrote five stamps ahead of the clock today; each was corrected.

## Tomorrow, in order

1. Read in by `AGENTS.md`'s order. The new Gemini lead starts on the new mailbox tool, `internal/comms/`
   (`README.md`, `post.py`, `wait.py`); today's files stay as the record.
2. The owner's looks, batched on Start here: 751-785 against the cut, and the drawing build's reference sheet if it
   did not reach them tonight, then its renders: the mom's 278-293 (silent, no mouth) and the dad's 1340-1360 twice
   (no mouth against the repaired mouth), each against today's drawing, same seed. If the two 1340 arms read equal, the
   same pair on 1156-1182 before the mouth leaves singing windows (M19).
3. Defaults, each a row on the Open List for the owner's yes: the new drawing and the per-window mouth (replaces
   r18_03's rule) once the tests read; the head tube (r18_02) after one occluded shot; depth after one more real load.
4. r18_05: remake the kept pieces with the old mouth, once the drawing is proven.
5. The second-half push, worst first (`internal/gemini/2026-10-18_next_shots_track_check.md`): fresh first previews
   for every shot from an old capture; the lost tracks (the dad 1632-1729, the mom 1437-1452); the shots that need depth
   (1466-1729, 4647-4725, 1419-1465); 2050-2622; then the uncaptured shots in song order (the newest gap check).
6. Read gem-track's Pointmap finding; it decides nothing tomorrow, but hand-over-face shots recur across the second half.

## Where things are

- Decisions and why: `docs/wiki/decisions.md`, 2026-10-18. The owner's preferences: `docs/wiki/working_with_the_owner.md`.
- The cut and gaps: `data/fun/laid.jsonl`; the newest gap check under `data/2026-10-18_fun_delivery_tables/`
  (`frame_coverage_*`); verdicts `data/fun/verdicts/verdicts.jsonl`; the owner's pages `Video/fun/TO_REVIEW.html` and
  `PROVENANCE.html` on the share, rebuilt by `bench/rebuild_delivery.py`.
- Today's findings, each one page: `internal/gemini/2026-10-18_*_finding.md` (dad_glasses, mouth, mouth_0278e,
  feature_drift), `_window_leaks_options.md`, `_drawing_map.md`, `_open_reference_view.md`, `_next_shots_track_check.md`,
  `_depth_crossings_map.md`, `_mouth_kept_audit.md`; each worker's notes `_worker_<name>.md`.
- The Claude lead's notes: `internal/claude/2026-10-18_mrted/` (`mentor_notes.md` for the next Claude lead,
  `built_2026-10-17.md`, `handover_notes.md`). The Gemini lead's: its last G entry, headed "End of day".
- Commands: `docs/wiki/runbook.md`. The lane: `data/AGENTS.md`, then `data/fun/AGENTS.md` (local).
