# Handover: read this before anything else

Rewritten 2026-10-26 at the close of the 2026-10-26 working day by the Gemini lead (`gemlead`). Where this page and a
record disagree, the record is right: `git log`, [`decisions.md`](decisions.md), the lane rules in
[`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), and
[`../masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md), the pass-contamination
and mask-step forensic reference [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md),
and `data/fun/`. *(The step-by-step plan `internal/gemini/2026-10-26_step_by_step_fix_plan_4912_and_4647.md` is superseded as
of 2026-10-26: it rested on the disproved pass-1 contamination premise, `--hold-part` spans, and the `shot_c_4647j` preview,
each overturned in [`decisions.md`](decisions.md) 2026-10-26 and the forensics record §0. Do not run its commands.)*

## State when closed

- **Landed on `main` today (`b461584e..76533a22`, 15 commits on 2026-10-26)**:
  - **Build A (`b461584e`, `ac803f92`)**: fed `head_tube` (`ht_res["tube"]`) into `bench/capture_masked_run.py::her_hair_in_hole`
    and `bench/inspect_mask_steps.py` instead of raw `head_track_arr`; skipped `hidden` frames in `her_hair_in_hole` Stage 2/3;
    restored full `~taken[k]` other-person subtraction; and made `inspect_mask_steps.py` `[2]` flag active latent steps where
    some frame has `0 px` mask.
  - **Build B (`b375d53e`, `7c195e68`, `68e8086d`)**: recorded the verified `4647-4725` cut-vs-raw pixel measurements
    (`bench/assemble_delivery.py` `rule: "matte"` lays `0` on `second_track` across all `79` frames of `third_fp_4647e`, so the
    Pass-1 cut did not contaminate `second`) and the `cast.jsonl` line `186` (`frame: 4698`) approximate box root cause in
    [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md) §0,
    [`../masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md) §7–§8,
    [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), and [`decisions.md`](decisions.md).
  - **Build C (`00a3f1d8`, `45f3eaf1`)**: included `held_masks` (`her_hair_in_hole`) in `track__<by>.mkv` so `video_mask.py`'s
    window union box covers `hers` crown pixels (`4943-4946` on `4912`), and carved a 1-cell 4-connected bridge in `grid_carve`
    when keep-outs pinch a `hers` unit into a diagonal island dropped by `drop_islands`.
  - **Build D (`cbd39fbe`)**: moved settled `margin` and `zoom_up_to` recipes into `data/fun/plan.json` (`subjects.<subject>.recipe`
    and `shot_overrides`, recording `second_hfp_1093k` at `margin = 128`), and made `bench/shot_to_load.py job` and
    `bench/flags_before_render.py::recipe_check_of` hard-refuse any differing CLI `--margin` or `--zoom-up-to` value with no
    `--*-why` reason bypass.
- **Renders kept and laid today** (`data/fun/owner_answers.jsonl` & `data/fun/laid.jsonl`):
  - **Kept and laid**: `lead_fp_5935d` (`5935-6104`, `170f`, `owner_answers.jsonl:1652`, laid as row `69` across `53` shots in
    `data/fun/laid.jsonl` with `--no-rebuild` pending the next cut rebuild; `bench/laid.py keeps`: `69` laid rows, `0` missing
    keeps, `0` laid marked wrong).
  - **Rejected with root causes recorded in [`decisions.md`](decisions.md) (2026-10-26)**: `third_fp_5114d` (`5114-5129`, ran at
    `margin = 64` via session-written `--margin-why`, giving `regenerating_share = 0.20` and white reference eye lines; closed
    by Build D `cbd39fbe`) and `second_hfp_4647h` (`4647-4725`, swapped her hair to blonde on `4647-4673` at `+36.9` BGR luma,
    then reverted to dark hair from `4684-4685` onward).
- **The server (`:8188`)**: stopped cleanly at close of day per [`../comfy_notes.md`](../comfy_notes.md).

## What not to assume

- **Hand-drawn boxes on pasted screenshots (`cast.jsonl` lines `175-192` and `194-196`) are not tracker seeds; `/who/` page clicks (`1-174`) and session-verified SAM detection boxes (`197-209`) stand (`decisions.md`, 2026-10-26)**:
  - `data/fun/cast/cast.jsonl` holds three groups:
    1. **Group 1 (`lines 1-174`)**: the owner's `who-is-who` clicks on the HTML page — **valid and stand** (each identifies which detected person is the subject on that frame).
    2. **Group 2 (owner visual cues: annotated screenshots, rough sketches, and confirmed words saying a person or segment such as `hair + face` or `hair` is visible on a frame, once read back and confirmed by the session)**: always recorded as ground truth (`plan.json` `head_marks`, shot records, `owner_sheets/`) so the owner's eyes and word are never discarded, and used to drive tracker improvements, SAM 3.1 prompting/labeling, part/segment rules, and Group 3 verification (preferring general pipeline fixes over manual frame rules). The rough hand-drawn rectangles from `data/fun/owner_sheets/2026-10-23_4647_who_is_who_owner.webp` and `2026-10-23_4912_who_is_who_owner.webp` that were copied directly into `cast.jsonl` (`lines 175-192` and `194-196`, `2` already withdrawn at `193` and `210`) are **never included directly as raw box seeds**.
    3. **Group 3 (`lines 197-209`)**: `13` exact SAM person-detection boxes on `2050` (`2292-2571`, e.g. `2292 P6`, `2325 P1`, `2436 P3`) verified by a session on the SAM detection masks using the owner's Group 2 visual cues — **valid and stand**.
  - Do **not** withdraw Group 2 (`cast.jsonl` lines `175-192` and `194-196`) before tomorrow's tracker build lands: on `4647-4725` the owner's only
    page click on `second` is frame `4666` (`line 128`, before the `4677-4681` occlusion), so withdrawing Group 2 (`175-192` and `194-196`)
    goes together with tomorrow's tracker build that re-finds her after an occlusion (or replaces any needed seed with an exact verified SAM detection box, Group 3).
- **What both sides verified on `second_hfp_4647h` (`4647-4725`)**:
  - `4647-4673` replaced her face and hair with blonde hair cleanly, proving `2.17x` zoom in the single 79-frame window swaps
    her to blonde once unmasked dark hair is removed. Never split a shot `<= 345` frames across an interior occlusion
    ([`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) §6).
  - On `4682-4685` (`t=11`, `0` regenerated tokens), her person track (`765-1,082 px`) held only her bare forearm (`y=297-359`),
    so crop-Sapiens cropped only her forearm (`y=289-367`, false-positive `Face_Neck` on her arm, suppressed by `head_tube`
    with `0` face points) while her emerging head edge on `4684-4685` had no track or mask and the `hidden` mark (`4674-4685`)
    zeroed the step. On `4686` and `4693-4695`, because `--head-per-stretch` was off (`head_track == 0 px` on `4682-4695`),
    a regained `head_tube` frame in the capture (`bench/head_tube_parts.py:335-350, 545-568`, `_head_seed_from_person_mask` with
    `tracker=None`) was a flat horizontal slice of her upper-body person track at a proportion measured when she stood tall
    (`4673`), slicing `945-1,354 px` off crop-Sapiens's valid head/hair mask (`1,271 px` lost in `parts_held` on `4686`), while
    on `4688, 4691-4693` `parts_held` held `726-1,460 px` of `Hair` (class `4` in `classes__shot_c_4647k.npz`).
  - On `4686-4693` (`t=12-13`), the laid matte reached `40-70 px` above her head into the dark-haired bystander woman SAM had
    detected (`2,000-3,500 px` in `shot_c_4647k_people.npz` at `0.54-0.81`) while `strangers.mkv` held `0 px` there because
    whole-frame `Sapiens2` missed her in `bench/strangers.py::run`.
  - In `her_hair_in_hole` (`bench/capture_masked_run.py:5303-5312, 5384`), `hole(f)` judges "inside the hole" against `out[f]`
    grown by the margin (`30` canvas px) before the keep-out carve and the feather (`4686`: `head[39] & ~hole(39) == 0 px`,
    leaving `130 px` of `head_tube[39]` in the feathered rim at `matte <= 127`), and `grade_holds` (`part_grades.json`) still
    runs after the job step (`4702-4705, 4708-4714` emptied under the `1,024 px` floor and `4706-4707` held from `4701`).
  - `<Video 1>` (`second_4647_4725_j/second_head_reference_4647_4725.json`) had `43` of `79` frames bare (`4673-4686` and
    `4697-4725` from a `--no-draw-face-on` word, on top of no body model on `4676-4685` and `4706-4716` because the pose crop
    followed the forearm/drifted track); `second_hfp_4647i` cannot reuse `second_4647_4725_j` and must remake `reference` after
    the tracker build with no drawing word carried.
- **Build days and render days do not mix**: tomorrow morning is a build session in worktrees with `:8188` idle until all three
  builds are replayed on kept/existing captures (`4647k`, `4912h`, `1093`) and the owner inspects the `4682-4696` strip.

## Tomorrow, in order

1. **Morning builds (built in worktrees with `:8188` off; Builds 1 & 2 replayed offline on saved `4647k`, `4912h`, and `1093` captures first, then Build 3 run for the fresh `4647` preview + `reference` remake and owner strip check before queuing any render)**:
   - **Build 1 — Part definition & downstream floor/fill (`bench/shot_to_load.py:1102-1198, 3088-3090, 4518-4525`, `bench/load_chain.py:374-380`, `bench/head_tube_parts.py:335-350, 521-569`, `bench/capture_masked_run.py:1098-1112, 1191-1208, 2778-2828, 5177-5432`)**:
     retire `hold_to_track` (`_on_track_8px.mkv`) and `_head_seed_from_person_mask`'s flat `head_rel_y1` slice; define her part at
     the job step from her Sapiens face and hair on her crop unioned with her SAM head track within a room anchored to her head
     (never unioning `Upper_Clothing`/`Torso` from `person_track` into a face+hair part, and never anchoring the head room to an
     arm-only crop), less every other person's SAM detection; and settle the downstream `grade_holds` (`1,024 px` floor at
     `1098-1112`) vs. `her_hair_in_hole` (`~hole(f)` at `5303-5312` and `~track[f]` at `5384`) so a valid small head is not
     emptied or left in the feathered rim.
   - **Build 2 — Strangers keep-out (`bench/strangers.py:395-449, 479-502, 542-570`)**: include SAM 3.1's other-person detections
     (`people.npz`, `others[i]` where `o[3] is None`) in `strangers.py::run` (`561-570`) where whole-frame Sapiens2 finds nobody.
   - **Build 3 — Tracker post-occlusion re-finding, Group 2 screenshot-box withdrawal & `4647i` reference remake (`subject_track.py:829-840, 1200-1359, 1403, 1489, 1604-1654`, `bench/shot_to_load.py:485-494, 556-561`, `bench/cast_file.py:276-314, 364`, `data/fun/cast/cast.jsonl`)**:
     flip `head_mode` default to `"per stretch"` (`HEAD_PER_STRETCH`) in `subject_track.py:833-840` and `shot_to_load.py:485-494`
     and re-find a subject after an occlusion via the tracker's own head detection per stretch and the still signature match;
     withdraw the Group 2 hand-drawn screenshot boxes (`cast.jsonl` lines `175-192` and `194-196`, citing `data/fun/owner_sheets/*.webp`) while keeping Group 1 (`/who/` page clicks `1-174`) and Group 3 (verified SAM detection boxes `197-209`), and update `cast_file.py` so `box` requires/records a verified SAM detection rather than raw hand-drawn screenshot coordinates; and remake `<Video 1>` (`shot_to_load.py reference`) for
     `second_hfp_4647i` from the new preview's track with no drawing word (`--no-draw-face-on`, `--no-carry-words`) carried. If
     the match cannot re-find `second` on `4647` after `4681`, one page click on her outline on `4690` (or a Group 3 verified SAM detection box guided by the owner's sheet) is the fallback.
   - **Replay & owner strip check**: replay Builds 1–2 offline on `4647k`, `4912h`, and `1093` captures (and verify on the fresh
     `4647` preview/capture once Build 3 runs) and show the owner the `4682-4696` strip drawn from the result before queuing any
     H3 render on `:8188`.
2. **Afternoon renders on the fixed chain**: `second_hfp_4647i` (`4647-4725`), `second_hfp_4912k` (`4912-4946`), and
   `third_fp_5114e` (`5114-5129` at settled `margin = 16`).
3. **The lead's owed single-subject spans**: `6105-6394`, `3759-3908`, `516-603`, and the second half of `2050-2622`
   (`2254-2598` laid on `2271-2598`), followed by the `604` gaps and `lead`'s Pass 3 on `4647`.

## Where things are

- **Reading order for tomorrow's session before writing code**:
  1. [`handover.md`](handover.md) (this file, for the day's order and what not to assume).
  2. [`decisions.md`](decisions.md) (`2026-10-26`: `5114d` margin rule `cbd39fbe`, `second_hfp_4647h` postmortem, the `cast.jsonl` box rule, and the two guardrails for tomorrow's builds).
  3. [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md) (`§0`–`§0.9`: verified `4647` and `4912` measurements).
  4. `bench/capture_masked_run.py` line `5132` (`HAIR_IN_HOLE_CANNOT_SEE`: the author's list of what `her_hair_in_hole` cannot see).
  5. `internal/gemini/2026-10-26_code_map_and_simplification_plan.md` (Parts 1–4 for exact `file:line` jump coordinates and state-mutation trace; note that `5e`, `5f`, and `5g` in Part 5 are superseded by this handover).
- Verification procedure & lane rules: [`../masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md),
  [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), and [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md).
- The laid ledger (`69` rows across `53` shots, including `lead_fp_5935d`): `data/fun/laid.jsonl`.
