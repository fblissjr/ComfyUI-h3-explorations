# Handover: read this before anything else

Rewritten 2026-10-25 at the close of the 2026-10-24 working day by the Gemini lead (`gemlead`), paired with the Claude
lead (`mrfabez`). Where this page and a record disagree, the record is right: `git log`, [`decisions.md`](decisions.md)
(2026-10-24 and 2026-10-25), the lane rules in [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) and
[`../masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md), the research
record [`../research/masking/2026-10-25_context_frames_and_identity_on_2050.md`](../research/masking/2026-10-25_context_frames_and_identity_on_2050.md),
and `data/fun/`. Read next, and check before acting on it:
`internal/claude/2026-10-24_plan_for_gemini_from_the_reviewer.md` and
`internal/gemini/2026-10-24_begin_of_day_plan_by_gemini-revision3_status_at_1800Z.md`.

## State when closed

- **Landed on `main` today (`988a03dc..dc671ad7`, 16 commits)**:
  - Tracker & bench Phase 1A/1B: gallery head search when nearest head is occluded (`988a03dc`), unified
    `bench/_lib/name_taken.py` across `data/*_jobs` and per-shot pass-2 cut check (`e4d3555b`), `--no-draw-face-on`,
    reference fill union, and `marks_against_the_body_model` diagnostic (`310416d9`), downward beard growth by
    `JAW_BEARD_PAD_FRAC = 0.25` in `face_piece_kept_whole` (`4cebb0aa`), and frontal-pose guard in `find_mesh_chin_y`
    (`39b9899f`).
  - Evening tooling & docs: own-subject pass-2 refusal and default `--cast` in `preview` (`829fc56b`, restored in
    `7a17459b`), `bench/flags_before_render.py` initial gate and `bench/flag_census.py` (`314d83dc`), `--no-head-on`
    and `--no-head-form` (`8b01e869`), `bench/said_on_shot.py` and `bench/review_server.py look` (`0eb91797`), the
    2026-10-25 three-path window rule, `345`-frame masked ceiling, and held-edge nearest-good-frame rule (`72ea3582`),
    the raw-frame and laid-keeps verification procedure (`2ae4b318`), and the kept verdicts and seam MAD table in the
    2050 research record (`755e0402`, `dc671ad7`).
- **ComfyUI (`:8188`) and the review server (`:8765`)**: both stopped at the close, unarmed, ports free.
- **The cut**: the `2026-10-24T2341Z` whole-video rebuild
  (`data/2026-10-24_fun_delivery_tables/rebuild_fun_2026-10-24T2341Z.json`), built from `data/fun/laid.jsonl` at `64`
  rows (`sha256: 2dbbc8388f815599`), `0` assembly failures, all `64` rows at `share: 1.0`, `laid_but_barely_changed: 0`
  frames, and `seams: 1` (only the historical `3097/3098` split between `lead_fp_2823b` and `lead_fp_3098b`; `0` seams
  on any piece laid today). Copied byte-equal (`cmp`) to `input/` and registered in `data/fun/manifest.json` as
  `fun_0000_6394_joined_2026-10-24T2341Z_from_laid.mp4`. After the build, line `64` (`lead_fp_2050g`) in
  `data/fun/laid.jsonl` (`sha256: dcf05f22d96d72f1`) was extended from `[[2050, 2247]]` to `[[2050, 2248]]` with
  `--no-rebuild` so the next cut carries frame `2248` (where `2248 -> 2249` moves by `1.1` luma units).
- **Renders kept and laid today** (`data/fun/owner_answers.jsonl`): `third_fp_5675g` (`5675-5762`), `third_fp_5011g`
  (`5011-5043`), `third_fp_5461e` (`5461-5525`), `third_fp_4647e` (`4647-4725`, replacing split piece `third_fp_4653b`),
  `lead_fp_4310c` (`4310-4374`), and `lead_fp_2050f` (`2050-2139`, 90f proof) superseded by `lead_fp_2050g`
  (`2050-2248`). Removed: `third_fp_1306g` (`1306-1384`, marked `wrong` on `1330-1348`). Standalone `bench/laid.py keeps`
  verified all `64` laid rows (`82` kept loads verified, `25` historical keeps superseded, `0` missing, `0` marked wrong).
- **Three worktrees ready to land before tomorrow's server start** (`git worktree list`, none holding uncommitted work):
  1. `h3_worktrees/builder-retire-versions` (`70279b26`, `gemlead/retire-versions`): retires hand-bumped `MASK_VERSION`
     (`nodes/subject_track.py`, `nodes/sapiens2_parts.py`), `REGION_VERSION` (`nodes/video_mask.py`), and
     `bench/masked_job.py::tracker_mask_version` in favor of `bench/_lib/code_stamp.py` and `ran_code`. Cleared by the
     Claude lead; held because it touches `nodes/*.py`.
  2. `h3_worktrees/builder-hold-nearest-good` (`33fcaf24`, `gemlead/hold-nearest-good`): the padding rule's node side in
     `nodes/video_mask.py` (`held_edge_source` and `hold_context`), stepping inward from the shot edge to the nearest
     frame inside the shot whose mask clears `HELD_EDGE_LEAST`, retiring the guard's fallback to unmasked real frames
     (`bench/check_video_mask.py` 7/7 ok). Held because it touches `nodes/video_mask.py`.
  3. `h3_worktrees/builder-laid-keeps` (`a733e355`, `gemlead/laid-keeps`): `bench/laid.py keeps` subcommand and
     `bench/check_laid.py` (6/6 ok); wire into `bench/rebuild_delivery.py` and add its [`../checks.md`](../checks.md)
     index row when landing.

## What not to assume

- **Padding frames never show the original subject unmasked in a window** ([`decisions.md`](decisions.md) 2026-10-25;
  [`../research/masking/2026-10-25_context_frames_and_identity_on_2050.md`](../research/masking/2026-10-25_context_frames_and_identity_on_2050.md)).
  Even when the hole is 100% blanked and `<Video 1>` drives the expression, unmasked frames of the same person anywhere
  in the window override `<Picture 1>`'s identity (`lead_fp_2050c` and `lead_fp_2050e` vs `lead_fp_2050f` and
  `lead_fp_2050g`). Follow the three-path window rule: (1) a shot up to `345` frames runs masked end to end across its
  full legal window and is chopped at the lay where the subject is off screen or at a hard cut; (2) a shot longer than
  `345` frames runs as chained windows inside one load (`90`-frame overlap re-denoised on the previous window's render);
  (3) across a hard cut, held copies take the nearest in-shot frame whose mask clears the floor (`gemlead/hold-nearest-good`).
- **Verify the node's own written video files, not just `.npz` or `.json` caches**
  ([`../masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md)). Before any render,
  check `in_frames`, `ref_frames`, and the 4 `mask_frames` `.mp4` files and run `bench/flags_before_render.py` (`one`
  initial required by the gate, two readers in practice). After any render, inspect the node's own `mask_review` and
  `raw` canvas videos alongside the assembled cut.
- **Never build a subject's pass on a cut that already holds that same subject's piece on the shot.** `shot_to_load.py`
  now refuses this (`829fc56b` / `7a17459b`), because any frame where the new mask is smaller leaks the old render
  around the rim.
- **Never vote on the review page on the owner's behalf, and never cite from memory or a compaction summary.** Read the
  exact `file:line` on disk before stating a figure, prompt (`bench/what_h3_got.py`), or cause.

## Tomorrow, in order

From `internal/claude/2026-10-24_plan_for_gemini_from_the_reviewer.md` (§5) and
`internal/gemini/2026-10-24_begin_of_day_plan_by_gemini-revision3_status_at_1800Z.md`:

1. **Before starting ComfyUI (`:8188`)**: blind-read and land the two node worktrees on `main`
   (`gemlead/retire-versions` `70279b26` and `gemlead/hold-nearest-good` `33fcaf24`), plus `gemlead/laid-keeps`
   (`a733e355`, wired into `bench/rebuild_delivery.py` with its [`../checks.md`](../checks.md) row). Then start `:8188`
   unarmed and `:8765` once.
2. **Whole-shot chained load on `2050-2622` (`573` frames, Path 2)**: `lead_fp_2050g` stands in the cut for `2050-2248`,
   and the singer re-enters the mirror from `2271` onward inside the same recorded shot (`2050-2622`). Run one chained-window
   load over the full `573`-frame shot (`--drop 2249-2270`), masked throughout, to replace `lead_fp_2050g` once kept.
3. **Multi-pass & dad remakes on the `2341Z` cut (`fun_0000_6394_joined_2026-10-24T2341Z_from_laid.mp4`)**:
   - `1306h` (`third` on `1306-1384`, `--source` the `2341Z` cut holding `lead` and `second`, replacing removed `1306g`).
   - `4647` pass 2 for `second` (`4647-4725`, on the `2341Z` cut holding `third_fp_4647e`), then rebuild and run pass 3
     for `lead` (`4647-4725`).
   - `3967` (`lead` remake with `--hold-part 3967-3970@3971` and `--no-draw-face-on 3969-3970`), `4479` (`third`),
     `1419` (`second` pass 3: read `1419g`'s seeded head track on `1431-1436`), `4912` and `1039` (`second`), and
     `5114` (`second_hfp_5114i` closed-mouth remake, then `third` on the rebuilt cut).
4. **Remaining coverage shots** (`6105`, `5935`, `3759`, `0516`, `3595`, `5044`, `4774`, `3476`, `4227`, `3240`), each
   checked against the three-path window rule so no window carries unmasked frames of the subject.
5. **Remaining §5 tooling items** (with the card held between steps):
   - **0b**: reader's take 5th line reading the node's own `mask_review` and `raw` canvas writes, plus a seam-MAD table
     helper printing both render and source step MADs across every laid piece's edges.
   - **3 & 4**: `bench/render_ledger.py` repo-relative paths for `bench/check_doc_links.py`; `laid.jsonl` `replaces`
     normalisation and per-load CLI word census.
   - **5b, 5c, 5d**: eye-line roll rule in `marks_against_the_body_model`; largest-connected-component per-frame boxes
     and native-resolution `1424x1080` zoom crop in `nodes/video_mask.py`; `bench/crop_reference_to_head.py::cranium_box`
     skipping bare and dropped frames.

## Where things are

- Today's research record & seam table:
  [`../research/masking/2026-10-25_context_frames_and_identity_on_2050.md`](../research/masking/2026-10-25_context_frames_and_identity_on_2050.md).
- Verification procedure: [`../masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md).
- Today's plans, case study, and storage inventory: `internal/claude/2026-10-24_plan_for_gemini_from_the_reviewer.md`,
  `internal/claude/2026-10-24_stop_start_pivot_for_gemini.md`,
  `internal/claude/2026-10-24_subagent_briefs.md`,
  `internal/gemini/2026-10-24_begin_of_day_plan_by_gemini-revision3_status_at_1800Z.md`,
  `internal/gemini/2026-10-24_case_study_what_worked_and_what_didnt.md`, and
  `internal/gemini/2026-10-24_storage_inventory_and_cleanup_plan.md`.
- The `2341Z` cut and stacked views: `data/2026-10-24_fun_delivery_tables/rebuild_fun_2026-10-24T2341Z.json` and the
  share's `Video/fun/day_2026-10-24/fun_0000_6394_joined_2026-10-24T2341Z_from_laid.mp4` (`_STACKED.mp4` beside it).
