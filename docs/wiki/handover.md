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
- **The newest whole video** is `fun_0000_6394_joined_2026-10-16T2250Z_from_laid` (with 0014f and 0294d laid; ledger
  and table 2250Z); the app plays its `_STACKED.mp4`, Start here 1. The pages were made from it at about 23:45Z;
  Start here 2 is 1010c beside 1010j.

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

1. Read the night's outbox (`internal/gemini/2026-10-16_outbox.md`), mrtaco's HANDOFF
   (`internal/claude/2026-10-16_mrtaco/HANDOFF.md`) and the lead's (`internal/claude/2026-10-16_mrtony/HANDOFF.md`).
   Then write the 2026-10-16 changelog entry from all that day's commits (`changelog.d/README.md`; `bump: minor`).
2. **Builds waiting, each on its own branch; none lands before its blind read passes, landed from the merge base.**
   As of 00:24Z (the lead updates this on each landing; each branch's newest commit is the truth): the matte ring
   (one pass) and `reach` are BUILT AND TESTED on `mrtaco-matte4` at 1e072b55, read not yet done; the plain-row blur
   and the thin 2.0 margins are NOT started. The re-composite round-2 fixes and the `held` measurement are NOT
   started (`mrtaco-recomposite` stands at 6c2437b4). D19c was building on `gemini-d19b`; D20b and D22 in their
   copies; D23 (a Gemini read of mrtaco's two branches and the brightness) running. mrtaco stopped for the night.
   - **The matte at the source's size (mrtaco, `mrtaco-matte4`).** Two passes leave a ring of the original inside
     every matte piece's feather (up to 34 of 130 levels; `internal/claude/2026-10-16_mrtony/review2_matte_d18.md`).
     The fix is one weight scaled once and laid in one pass (written down: `docs/wiki/masked_v2v.md`, "Laying a piece
     by its matte at the source's size"), plus `reach` and the plain-row blur. MUST land before any matte-laid piece
     (1010c, a re-laid piece, any new render) goes into the cut; the third read is first. Gemini's D23
     (`internal/gemini/2026-10-16_review_D23.md`, on 55c6783a) found the same ring independently; its third read must also
     check D23's two further claims against 1e072b55: the double update at the canvas size in chroma at odd pixel
     positions (`merged`, old :1182, :1190), and 12.5% of the source leaking where a later full edge crosses an earlier
     full interior.
   - **The re-composite tool (mrtaco, `mrtaco-recomposite`).** Control passes: own composite reproduces the original,
     last step included, plate offset gone (`review2_recomposite_piece.md`). Left: a stale decode reused by file name
     (key on the latent's and VAE's sha256), a typed `--vae` overriding the graph's, six green mutations, 24/1 assumed,
     and the encoder quality: `--crf` defaults to 12 (`recomposite_piece.py:321`) instead of the render graph's own
     `crf` (checked by the lead: all 121 render graphs on disk used 12, so no piece today changes; read it from the graph
     and refuse a typed value that disagrees, as for the source and the VAE).
     Also from D23: the default `--device cpu` cannot decode with the render's int8 VAE (CUDA only; the trial agent saw
     the same), so default to cuda or refuse with why; and a re-laid piece lacks `<stem>.png`, `_with_mask.mp4` and
     `kept` in its region file, which `stack_delivery`, `render_ledger` and `capture_masked_run` read.
   - **D19c, the pages read the gap check (Gemini, `gemini-d19b`):** the last round; then its read.
   - **D20b, `--carry-words-from` (Gemini, its branch):** keep each word's author; tonight's 1010c lost 1010j's
     `no_mouth_on 1010-1018` (both job files show it), which is what it is for. Until it lands, a remake's words are
     copied by hand and listed in the job's note.
   - **D22, floor holds and off-head (Gemini):** on branch `gemini-d22` at fc2aba16 (from `mrtaco-floor` 70976d50),
     NOT landed: its second read (`internal/claude/2026-10-16_mrtony/review2_floor_off_head.md`) found its note describing
     code the commit does not hold, a bent-head guard that can keep a hand, four vacuous cases; D22b in the requests file.
     Good: the frame table accepts floor holds; every held frame reports its source and face points. Then 0613g's plan.
   - Settled tonight, no build: the brightness (and D23 watched the street and 0014f at 24 fps: no box). `assemble_delivery` lays onto a 16-bit read at the canvas and the
     source's own yuv at full size; no step in the cut (`review2_recomposite_piece.md` item 3). Other tools still read
     8 bits (`recomposite_window`, `patch_render_window`, `masked_render_against_source`, `strangers`): a list, no rush.
3. **The owner's two questions (the Open List):** r16_02, what the `held` keep-out keeps. It, not `clothes`, cut the
   mom's neck and hair at 287, 422, 483-489 (measured, `data/2026-10-16_recomposite_trials/*_keepout_cause.txt`): her
   white apron is Background to the part model, so near her lips and hands it becomes "held". The carve does not touch
   `held`, so today's rules do not fix it; the lead's pick is held only where a look says she holds something. Those
   three shots (278-293, 415-437, 484-515; one window each) are remade after that answer. And r16_01, the mom's
   kitchen: see 0604d re-laid first (clips in `Video/fun/day_2026-10-16/recomposite_trials/`).
4. **The owner's looks, Start here:** 1, the whole video (2250Z): the four late pieces 4653-4677, 5114-5129,
   5461-5497, 5533-5564. 2, 1010c beside the kept 1010j (readers: she sits upright facing the camera, a pop at
   1036-1037, the wrong clothing words, the lost mouth word; cause of the pose NOT known), with the call on 1032-1038
   (1010j stops at 1031: a short load, or the owner's blonde fill). Plan strips come as made: 0613g, 1010g, 516-603,
   the whole-shot loads 1061-1092, 1093-1133, 1306-1418 and the dad's 1310-1418.
5. Owed: delete `data/2026-10-16_recomposite_trials/**/decoded_*.npy` (about 3.2 GB; the owner's yes) once the
   trial agent is done; list what went.

## Where things are

- Decisions of the day: `docs/wiki/decisions.md` 2026-10-16 (newest first). The owner's notes:
  `data/fun/verdicts/verdicts.jsonl` (today's from 21:50Z on are the whole-video watch).
- Requests to and answers from the Gemini sessions: `internal/gemini/2026-10-16_requests.md` (D1-D20) and
  `..._outbox.md`; reads by the lead's blind readers: `internal/claude/2026-10-16_mrtony/review*.md`.
- Pages: `Video/fun/TO_REVIEW.html` and `PROVENANCE.html` on the share, made by
  `internal/claude/2026-10-16_mrtoon/pages.sh` from `start_here.json`; the app at the review server
  (`data/fun/poc/serve.py`, restarted by the lead at about 18:20Z).
- Re-composite trials: `data/2026-10-16_recomposite_trials/` and `Video/fun/day_2026-10-16/recomposite_trials/`.
