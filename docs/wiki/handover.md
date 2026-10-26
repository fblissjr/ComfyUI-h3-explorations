# Handover: read this before anything else

Rewritten 2026-10-25 at the close of the 2026-10-25 working day by the Gemini lead (`gemlead`). Where this page and a
record disagree, the record is right: `git log`, [`decisions.md`](decisions.md), the lane rules in
[`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), and
[`../masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md), the pass-contamination
and mask-step forensic reference [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md),
and `data/fun/`. Read [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md)
(especially §3 on Pass-2/Pass-3 source vs. perception contamination, §4 on environment variables and data paths, and §6 on
every tool and rule built today) before running any preview, capture, or render tomorrow. *(The step-by-step plan this
line pointed at, `internal/gemini/2026-10-26_step_by_step_fix_plan_4912_and_4647.md`, is superseded as of 2026-10-26: it
rests on the pass-1 contamination premise and on `--hold-part` spans and the `shot_c_4647j` preview, each overturned in
[`decisions.md`](decisions.md) 2026-10-26 and the forensics record §0; what was run instead is in this page's "What not
to assume" and in the record. Do not run its commands.)*

## State when closed

- **Landed on `main` today (`dc671ad7..449b17f1`, 10 commits)**:
  - Morning worktree landings & rulings: retired hand-bumped `MASK_VERSION` and `REGION_VERSION` in favor of
    `bench/_lib/code_stamp.py` (`96f9989e`), nearest-good-frame held-edge rule in `video_mask.py` (`f5e0d66d`),
    `bench/laid.py keeps` verification and unrendered source intent check (`62bc4469`), eye-line roll check in
    `marks_against_the_body_model` and bare/dropped-frame skip in `bench/crop_reference_to_head.py::cranium_box`
    (`86e65d1f`), and 2026-10-25 rulings and 64-row remake inventory (`f355d420`, `b370a50a`).
  - Timeline, coverage & recipe gates: `bench/inventory.py`, `bench/timeline_page.py`, and frame-pinned note matching in
    `bench/said_on_shot.py` (`ec9a113f`); partial-shot, unlaid-while-tracked, and partial-lay coverage gates (`e5be4a20`);
    per-subject recipe defaults (`h3_config.SUBJECT_RECIPE_DEFAULTS`), pre-render recipe diff (a diff, not a gate: a
    difference with any written reason passes; `decisions.md`, 2026-10-26, third_fp_5114d), and pass-2 cut/lay
    guards (`79b7b6da`).
  - Mask-step forensics & latent-step/keep-out fixes (`449b17f1`): `bench/inspect_mask_steps.py` and
    `bench/check_inspect_mask_steps.py` (`[0]` pass-dependency/creation-date/MoGe-2 depth/prior-pass DiT overlap provenance,
    `[1]` `PAD` bleed, `[2]` blue-only `0 px` frames, `[3]` speck-in-blue frames, `[4]` `< 75%` edge-shrunk `whole` frames,
    `[5]` `strangers` self-swallow, `[6]` missed SAM 3.1 strangers inside the blue region, `[7]` gap recovery table +
    `--strips`); introduced `video_mask.py::HELD_EDGE_EMPTY_RUN = max(FRAME_PER_TOKEN)` (`4`, one 4-frame latent step; no
    empty-run cap existed after `f5e0d66d`); filtered `whole` reference frames against `< 0.75 * neighbor`
    edge-shrink (`clean_whole`), widened `near_h` to `abs(k - src) <= 2`, and added class-filtered (`Apparel` + non-subject
    `face_ids`) `taken_for_hers` in `bench/capture_masked_run.py::her_hair_in_hole` (subsequently reverted in favor of
    subtracting every other person's mask `~taken[k]` after `4912h` showed same-color bystander hair has class `Hair`); added
    `unmasked_head_hair_context` signal in `bench/capture_masked_run.py` and `bench/flags_before_render.py`.
- **The cut**: the `2026-10-25T2355Z` whole-video rebuild
  (`data/2026-10-25_fun_delivery_tables/rebuild_fun_2026-10-25T2355Z.json`), built from `data/fun/laid.jsonl` at `68`
  rows across `52` shots (`sha256: 7174f8a9c62aefc9`), `0` assembly failures (`verdict: passes`), all `68` rows verified by
  `bench/laid.py keeps` (`0` missing keeps, `0` laid marked wrong) and `bench/laid.py verify` (`68/68` `bytes as laid`).
  Copied to `input/` and registered in `data/fun/manifest.json` as `fun_0000_6394_joined_2026-10-25T2355Z_from_laid.mp4`.
- **Renders kept and laid today** (`data/fun/owner_answers.jsonl`): `third_fp_4479e` (`4479-4509`), `third_fp_1306h`
  (`1306-1418`, replacing removed `third_fp_1306g`), `lead_fp_1061b` (`1061-1092`, replacing `lead_fp_1061p`),
  `lead_fp_2623b` (`2623-2711`, replacing `lead_fp_2625`), `second_hfp_5114k` (`5114-5129`, replacing `second_hfp_5114i`),
  `second_hfp_1039h` (`1039-1060`), and `lead_fp_3967c` (`3967-4052`, laid on its active face-visible span `3970-4035`).

## What not to assume

- **Pass 1's keep-out silhouette (`others`) and Pass 2's preview always come from the post-Pass-1 cut (The Kitchen `604-1009`
  Bouffant Hair Rule, [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) §1), and never graft tracks across different source
  plates ([`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md) §2.1)**:
  - Never use a raw-source keep-out mask for an already-replaced Pass-1 subject: in Kitchen `604-1009` (`lead_fp_0606d` over
    `second_hfp_0604h`), `second`'s 1980s bouffant hair shrunk by ~60 px in Pass 1 (`second_hfp_0604h`), so a raw-source
    keep-out for `second` kept the old 60 px taller bouffant mask (`5843-9276 px`) and made `lead` disappear into `second`'s
    old hair mask when passing behind her on `755-774`. Pass 2's preview **always** runs on the newest post-Pass-1 cut so
    Pass 1's keep-out tracks the new replaced subject.
  - On `second_hfp_4647g` (`4647-4725`, Pass 2 on the post-`third_fp_4647e` cut), `third_fp_4647e`'s regenerated region
    overlapped `second` on `32` frames (`4647-4689`, up to `1,076 px`), while `second` lost her head track on `4674-4685`
    (`0 px`) and ran without `--hold-part` or `depth.npz`. On `4696-4703`, `second`'s head was tracked at `1,416-6,652 px`
    (`0 px` only on `4697`), and `head_tube` was already enabled by default (`head_tube: true` in `manifest.json`, bridging
    `4686-4695` at `907-2,867 px`), but `capture_masked_run.py:2797` handed raw `head_track_arr` (`0 px` on `4674-4695`) to
    `her_hair_in_hole` instead of `head_tube`'s bridged mask. Do **not** graft `second`'s tracks from the raw preview
    (`shot_c_4647f`): `4647j` already tracks `second`'s body at `22,521-51,297 px` across `4647-4673`, and the `4674-4685`
    gap is resolved on the post-Pass-1 preview (`shot_c_4647j`) by feeding `head_tube` into `her_hair_in_hole`, owner
    `head_marks` (or single-span `--hold-part 4674-4685`), and `depth.npz`.
  - **Overturned 2026-10-26, kept here so the paragraph above is not trusted** ([`decisions.md`](decisions.md), 2026-10-26;
    the forensics record §0): the pass-1 cut does not alter her (single-digit changed pixels inside her track on every
    frame); the `4697` drop and the `4698-4706` drift were the owner's cast box on `4698` (`cast.jsonl` line 186, withdrawn
    at line 210), taken on the cut's preview by a share of 0.56, which put the woman in the black vest in her track, so
    `shot_c_4647j` is not reused: `second_hfp_4647h` was built on the fresh preview `shot_c_4647k`; the tube hand-off
    landed in `b461584e`; `4674-4685` is `hidden` on the owner's reading, so nothing is held there and `--hold-part` is
    withdrawn; the owner's marks are `hidden` 4674-4685 and 4716-4725, `hers` 4697-4698.
- **`head_tube` runs by default in `bench/capture_masked_run.py:2558`** whenever a `head_track` exists (`head_tube: true` in
  `manifest.json` unless `--no-head-tube` is passed); `--head-tube` on `bench/shot_to_load.py job` records the word in
  `job.json`. Also note that `--hold-part` accepts only a single `A-B` span (`capture_masked_run.py:2643` refuses `+`); for
  multiple occlusion spans (such as `4921-4926` and `4940-4946` on `4912`), use owner `head_marks` in `marks.json`.
- **Always run `bench/measure_moge2_depth.py` (`facebook_hyden-mogev2-metric-point`) when `contested_px > 0` or `strangers`
  carves the target subject** (`[0]` in `bench/inspect_mask_steps.py`). None of today's 10 captures had `depth.npz`.
- **Restart ComfyUI (`:8188`) before the first render tomorrow** if it was started before `449b17f1` (`21:08 CDT`), so
  `video_mask.py` loads `HELD_EDGE_EMPTY_RUN = 4` (fixes the 4-frame latent-step `PAD` bleed seen on `second_hfp_4647g`).
- **Always export `H3_INPUT_DIR`, `H3_OUTPUT_DIR`, and `H3_MODEL_STORE`** in every subshell and subagent command, and pass
  `--share "$H3_OUTPUT_DIR"` (not `$H3_OUTPUT_DIR/Video/fun`) to `bench/laid.py`.

## Tomorrow, in order

1. **Verify `:8188` is running on current `main`** (restart unarmed if its process started before the latest `video_mask.py`
   or node change so ComfyUI executes the current code).
2. **Remake `second_hfp_4912` (`4912-4946`) and `second_hfp_4647` (`4647-4725`) on the `2355Z` cut**:
   - Audit `--depth` (`MoGe-2` metric depth, built `2026-10-18` and last run on `2026-10-20` `lead_fp_0606d`) and `head_tube`
      end-to-end as part of standard input creation (`bench/shot_to_load.py job --depth` → `depth.npz` →
      `bench/capture_masked_run.py` `owners.npz` → `bench/assemble_delivery.py::restore_weight` →
      `bench/inspect_mask_steps.py` `[0]`), so every capture and assembly explicitly records whether `depth.npz` was created
      and consumed, its impact on `carve_keep_out` and `restore_weight`, and its code/data freshness.
   - On `4912-4946`: rebuild capture/job after feeding `head_tube` into `her_hair_in_hole`, reverting class-filtered
     `taken_for_hers` to `~taken[k]`, running `measure_moge2_depth.py` (`depth.npz`), and answering the two occlusion gaps
     (`4921-4925` and `4941-4946`) via owner `head_marks` in `marks.json`; verify `[0]-[7]` with
     `bench/inspect_mask_steps.py --capture <dir> --problems-only` and `bench/what_h3_got.py` before rendering.
   - On `4647-4725`: keep the `2355Z` post-Pass-1 cut as `--source` and use the post-Pass-1 preview `shot_c_4647j` (never
     grafting tracks from raw `shot_c_4647f`), with `head_tube` fed into `her_hair_in_hole`, `depth.npz` to settle the
     `second` vs `third` / `strangers` overlap (`contested_px = 186`), and owner `head_marks` (or single-span
     `--hold-part 4674-4685`) on `4674-4685`; render Pass 2 (`second`), then rebuild and run Pass 3 (`lead`).
     *(As run on 2026-10-26: the `4698` cast box withdrawn first, a fresh preview `shot_c_4647k` on the current cut, the
     job with the owner's marks and `--depth`, no `--hold-part`; `decisions.md` 2026-10-26.)*
3. **Second half of shot `2050-2622` (`2271-2598`)**: per the owner's 2026-10-25 ruling in [`decisions.md`](decisions.md)
   ("How the rest of shot `2050-2622` is rendered"), `lead_fp_2050g` stands in the cut on `2050-2248`, and the second half
   (`2271-2598`; `2599-2622` is back of head after `lead` turns at `2598`) renders as a single `345`-frame window at
   `2254-2598` (`20*17+5`, under `h3_rules.MAX_LENGTH = 362`, with `2254-2270` as runway) and is laid on `2271-2598`
   (inspecting `2445-2598` where `lead` dances with the spinning disco ball).
4. **Remaining multi-pass & coverage shots**: `5114` Pass 2 (`third` on `5114-5129` over `second_hfp_5114k`), `1419` Pass 3
   (`second` on `1419-1465`), `3967` tail/head coverage if needed, and remaining unrendered shots (`6105`, `5935`, `3759`,
   `0516`, `3595`, `5044`, `4774`, `3476`, `4227`, `3240`), checking every capture with `bench/inspect_mask_steps.py` and
   `bench/flags_before_render.py` before queueing.

## Where things are

- Today's root-cause analysis, 10-capture matrix, and tool reference guide:
  [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md).
- Tomorrow's step-by-step fix & verification plan for `second_hfp_4912h`, `second_hfp_4647h`, and `--depth` / `--head-tube`
  end-to-end input-creation provenance: `internal/gemini/2026-10-26_step_by_step_fix_plan_4912_and_4647.md`.
- Forensic mask & latent-step inspector and check: [`../../bench/inspect_mask_steps.py`](../../bench/inspect_mask_steps.py)
  and [`../../bench/check_inspect_mask_steps.py`](../../bench/check_inspect_mask_steps.py).
- Verification procedure & lane rules: [`../masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md),
  [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) (§10–§11), and [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md) (§3.2–§3.3).
- The `2355Z` cut and stacked views: `data/2026-10-25_fun_delivery_tables/rebuild_fun_2026-10-25T2355Z.json` and the
  share's `Video/fun/day_2026-10-25/fun_0000_6394_joined_2026-10-25T2355Z_from_laid.mp4` (`_STACKED.mp4` beside it).
