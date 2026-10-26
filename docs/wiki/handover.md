# Handover: read this before anything else

Rewritten 2026-10-25 at the close of the 2026-10-25 working day by the Gemini lead (`gemlead`). Where this page and a
record disagree, the record is right: `git log`, [`decisions.md`](decisions.md), the lane rules in
[`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), and
[`../masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md), the pass-contamination
and mask-step forensic reference [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md),
and `data/fun/`. Read [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md)
(especially §3 on Pass-2/Pass-3 source vs. perception contamination, §4 on environment variables and data paths, and §6 on
every tool and rule built today) before running any preview, capture, or render tomorrow.

## State when closed

- **Landed on `main` today (`dc671ad7..449b17f1`, 10 commits)**:
  - Morning worktree landings & rulings: retired hand-bumped `MASK_VERSION` and `REGION_VERSION` in favor of
    `bench/_lib/code_stamp.py` (`96f9989e`), nearest-good-frame held-edge rule in `video_mask.py` (`f5e0d66d`),
    `bench/laid.py keeps` verification and unrendered source intent check (`62bc4469`), eye-line roll check in
    `marks_against_the_body_model` and bare/dropped-frame skip in `bench/crop_reference_to_head.py::cranium_box`
    (`86e65d1f`), and 2026-10-25 rulings and 64-row remake inventory (`f355d420`, `b370a50a`).
  - Timeline, coverage & recipe gates: `bench/inventory.py`, `bench/timeline_page.py`, and frame-pinned note matching in
    `bench/said_on_shot.py` (`ec9a113f`); partial-shot, unlaid-while-tracked, and partial-lay coverage gates (`e5be4a20`);
    per-subject recipe defaults (`h3_config.SUBJECT_RECIPE_DEFAULTS`), pre-render recipe diff gate, and pass-2 cut/lay
    guards (`79b7b6da`).
  - Mask-step forensics & latent-step/keep-out fixes (`449b17f1`): `bench/inspect_mask_steps.py` and
    `bench/check_inspect_mask_steps.py` (`[0]` pass-dependency/creation-date/MoGe-2 depth/prior-pass DiT overlap provenance,
    `[1]` `PAD` bleed, `[2]` blue-only `0 px` frames, `[3]` speck-in-blue frames, `[4]` `< 75%` edge-shrunk `whole` frames,
    `[5]` `strangers` self-swallow, `[6]` missed SAM 3.1 strangers inside the blue region, `[7]` gap recovery table +
    `--strips`); lowered `video_mask.py::HELD_EDGE_EMPTY_RUN` from `len(FRAME_PER_TOKEN)` (`5`, `17` frames) to
    `max(FRAME_PER_TOKEN)` (`4`, one 4-frame latent step); filtered `whole` reference frames against `< 0.75 * neighbor`
    edge-shrink (`clean_whole`), widened `near_h` to `abs(k - src) <= 2`, and added class-filtered (`Apparel` + non-subject
    `face_ids`) `taken_for_hers` in `bench/capture_masked_run.py::her_hair_in_hole`; added `unmasked_head_hair_context`
    signal in `bench/capture_masked_run.py` and `bench/flags_before_render.py`.
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

- **Pass 1's keep-out silhouette (`others`) always comes from the post-Pass-1 cut (The Kitchen `604-1009` Bouffant Hair Rule,
  [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) §1), whereas Pass 2's own target subject track/parts must be checked
  for Pass-1 DiT margin blur (`4647e -> 4647g`, [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md) §2.1)**:
  - Never use a raw-source keep-out mask for an already-replaced Pass-1 subject: in Kitchen `604-1009` (`lead_fp_0606d` over
    `second_hfp_0604h`), `second`'s 1980s bouffant hair shrunk by ~60 px in Pass 1 (`second_hfp_0604h`), so a raw-source
    keep-out for `second` kept the old 60 px taller bouffant mask (`5843-9276 px`) and made `lead` disappear into `second`'s
    old hair mask when passing behind her on `755-774`. Pass 2's preview **always** runs on the newest post-Pass-1 cut so
    Pass 1's keep-out tracks the new replaced subject.
  - Conversely, when `inspect_mask_steps.py` `[0]` flags that Pass 1's regenerated region (`third_fp_4647e`) altered the boundary
    ring beside Pass 2's subject and collapsed Pass 2's SAM/Sapiens track (`second_hfp_4647g` on `4696-4703`, plus `second`
    running without `--head-tube`, `--hold-part`, or `depth.npz` on `4674-4685`), keep the post-Pass-1 cut as `--source` and
    keep **Pass 1's (`third`'s) keep-out silhouette from the post-Pass-1 preview**, and restore **only Pass 2's own target
    subject (`second`)** tracks/parts/pose from the un-erased raw preview on the drifted frames (alongside `--head-tube`,
    `--hold-part`, and `measure_moge2_depth.py`).
- **Always run `bench/measure_moge2_depth.py` (`facebook_hyden-mogev2-metric-point`) when `contested_px > 0` or `strangers`
  carves the target subject** (`[0]` in `bench/inspect_mask_steps.py`). None of today's 10 captures had `depth.npz`.
- **Restart ComfyUI (`:8188`) before the first render tomorrow** if it was started before `449b17f1` (`21:08 CDT`), so
  `video_mask.py` loads `HELD_EDGE_EMPTY_RUN = 4` (fixes the 4-frame latent-step `PAD` bleed seen on `second_hfp_4647g`).
- **Always export `H3_INPUT_DIR`, `H3_OUTPUT_DIR`, and `H3_MODEL_STORE`** in every subshell and subagent command, and pass
  `--share "$H3_OUTPUT_DIR"` (not `$H3_OUTPUT_DIR/Video/fun`) to `bench/laid.py`.

## Tomorrow, in order

1. **Verify `:8188` is running on `449b17f1+`** (restart unarmed if its process started before `449b17f1` so `video_mask.py`
   uses `HELD_EDGE_EMPTY_RUN = 4`).
2. **Remake `second_hfp_4912h` (`4912-4946`) and `second_hfp_4647h` (`4647-4725`) on the `2355Z` cut**:
   - On `4912-4946`: rebuild capture/job with `449b17f1` (`HELD_EDGE_EMPTY_RUN = 4` + class-filtered `taken_for_hers` +
     `clean_whole`), run `bench/measure_moge2_depth.py` if `contested_px > 0`, and verify `[0]-[7]` with
     `bench/inspect_mask_steps.py --capture <dir> --problems-only` (`0` held-padding bleed, `0` speck-in-blue, `0`
     self-swallow) before rendering.
   - On `4647-4725`: keep the `2355Z` post-Pass-1 cut as `--source` and keep `third`'s post-Pass-1 keep-out silhouette, while
     restoring **only `second`'s own** un-erased tracks/parts/pose from the raw preview on the span contaminated by
     `third_fp_4647e`'s margin (`4647-4705`), run `bench/measure_moge2_depth.py` (`depth.npz`) to settle the `second` vs
     `third` overlap, render Pass 2 (`second`), then rebuild and run Pass 3 (`lead` — whose preview on the post-Pass-2 cut
     will then segment `second`'s new blonde hair silhouette just like Kitchen `604-1009`).
3. **Whole-shot chained load on `2050-2622` (`573` frames, Path 2)**: `lead_fp_2050g` stands in the cut on `2050-2248`;
   finish the chained load over `2050-2622` (`--drop 2249-2270`, and inspect `2445-2598` where `lead` dances with the
   spinning disco ball) to cover the second half of the shot (`2271-2622`).
4. **Remaining multi-pass & coverage shots**: `5114` Pass 2 (`third` on `5114-5129` over `second_hfp_5114k`), `1419` Pass 3
   (`second` on `1419-1465`), `3967` tail/head coverage if needed, and remaining unrendered shots (`6105`, `5935`, `3759`,
   `0516`, `3595`, `5044`, `4774`, `3476`, `4227`, `3240`), checking every capture with `bench/inspect_mask_steps.py` and
   `bench/flags_before_render.py` before queueing.

## Where things are

- Today's root-cause analysis, 10-capture matrix, and tool reference guide:
  [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md).
- Forensic mask & latent-step inspector and check: [`../../bench/inspect_mask_steps.py`](../../bench/inspect_mask_steps.py)
  and [`../../bench/check_inspect_mask_steps.py`](../../bench/check_inspect_mask_steps.py).
- Verification procedure & lane rules: [`../masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md),
  [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) (§10–§11), and [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md) (§3.2–§3.3).
- The `2355Z` cut and stacked views: `data/2026-10-25_fun_delivery_tables/rebuild_fun_2026-10-25T2355Z.json` and the
  share's `Video/fun/day_2026-10-25/fun_0000_6394_joined_2026-10-25T2355Z_from_laid.mp4` (`_STACKED.mp4` beside it).
