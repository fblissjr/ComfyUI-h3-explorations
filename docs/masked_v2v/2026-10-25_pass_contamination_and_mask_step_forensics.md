# Multi-Pass Contamination, Perception Provenance, and Mask-Step Forensics (`2026-10-25`)

This record documents the root-cause analysis, file creation (`%w`) vs. modification (`%y`) audit, perception/depth model status, environment/folder rules, and step-by-step mask diagnostic tooling (`bench/inspect_mask_steps.py`) built on `2026-10-25` to explain why multi-subject (`4647–4725`, `1306–1418`, `4479–4560`) and occluded single-subject (`4912–4946`, `2050–2292`) loads failed while unoccluded single-subject loads (`5114–5129`, `1039–1060`) succeeded cleanly.

Related rules and procedures:
- [`docs/masked_v2v/AGENTS.md`](AGENTS.md)
- [`docs/masked_v2v/PROTOCOL.md`](PROTOCOL.md)
- [`docs/masked_v2v/verifying_raw_frames_and_keeps.md`](verifying_raw_frames_and_keeps.md)
- [`docs/masked_v2v/frames_rules.md`](frames_rules.md)
- [`docs/wiki/handover.md`](../wiki/handover.md)

---

## 1. Executive Summary: Why `5114k` and `1039h` Succeeded While `4647g` and `4912g` Failed

In `data/fun/owner_answers.jsonl` on `2026-10-25`, the owner kept `second_hfp_5114k` (`5114–5129`) and `second_hfp_1039h` (`1039–1060`), while `second_hfp_4647g` (`4647–4725`) and `second_hfp_4912g` (`4912–4946`) showed severe hair/head loss, face-on-face paste-over, or tail flashes. Running [`bench/inspect_mask_steps.py`](../../bench/inspect_mask_steps.py) across all seven captures exposes the exact structural difference:

| Capture | Subject & Span | Pass # & Source | `part_sizes.least` | `[1]` Pad Bleed | `[2]` Blue-Only (`0 px`) | `[3]` Specks (`1..63 px`) | `[4]` Edge-Shrunk Whole | `[5]` `strangers` Swallow | `[6]` Missed SAM 3.1 |
|---|---|---|---|---|---|---|---|---|---|
| `2026-10-25_second_hfp_5114k` | `second` (`5114..5129`, 16f) | Pass #1 (raw 24p) | `11,043 px` | `0` | `0` | `0` | `0` | `0` | `0` |
| `2026-10-25_second_hfp_1039h` | `second` (`1039..1060`, 22f) | Pass #1 (raw 24p) | `21,993 px` | `0` | `0` | `0` | `0` | `0` | `0` |
| `2026-10-25_second_hfp_4647g` | `second` (`4647..4725`, 79f) | Pass #2 (`1903Z` cut holding `third_fp_4647e`) | `0 px` | `PAD_79`, `PAD_80` (`23 cells`) | `6` (`4674..4676, 4697, 4724..4725`) | `1` (`4699`, `8 px`) | `1` (`4694`) | `0` | `33` (`person#2`, `lead`) |
| `2026-10-25_second_hfp_4912g` | `second` (`4912..4946`, 35f) | Pass #1 (raw 24p) | `0 px` | `0` | `2` (`4925, 4941`) | `1` (`4926`, `8 px`) | `2` (`4920, 4939`) | `13` (`4921..4926, 4940..4946`) | `24` (`person#4`, arm) |
| `2026-10-24_lead_fp_2050g` | `lead` (`2050..2292`, 243f) | Pass #1 (raw 24p) | `0 px` | `0` | `1` (`2249`) | `0` | `2` (`2099, 2244`) | `11` (`2251..2257, 2262..2265`) | `123` (`person#3`, mirror) |
| `2026-10-25_third_fp_1306h` | `third` (`1306..1418`, 113f) | Pass #2 (`1903Z` cut holding `lead_fp_1306t`) | `0 px` | `0` | `0` | `0` | `5` (`1309, 1320, 1353, 1397, 1400`) | `0` | `14` (`person#15`) |
| `2026-10-25_lead_fp_2623b` | `lead` (`2623..2711`, 89f) | Pass #1 (raw 24p, **stale 10-21 tracks**) | `16,580 px` | `0` | `0` | `0` | `1` (`2624`) | `0` | `0` |

**Why `5114k` and `1039h` succeeded**:
- Neither shot has a second pass laid ahead of it, and neither has another person's arm or head crossing in front of the subject's face.
- `part_sizes.least` never drops below `11,043 px` (`5114k`) or `21,993 px` (`1039h`), so none of the occlusion-recovery or edge-padding branches (`her_hair_in_hole`, `strangers.py`, `held_edge_source`) are ever triggered.

---

## 2. Two Compounding Failure Chains on `4647–4725` and `4912–4946`

### 2.1 Multi-Subject Sequential Pass Contamination (`third_fp_4647e` → `second_hfp_4647g`)

On shot `4647–4725` (`79` frames), three subjects appear (`lead` on the left, `second` in the middle, `third` on the right).
1. **Pass 1 (`third_fp_4647e`) baked an uncarved wide region into the `1903Z` cut**:
   - In `data/2026-10-24_third_fp_4647e_job/third_fp_4647e.render.queued.json`, Pass 1 ran with `keep_out_files: []` (0 keep-out tracks for `second` or `lead`).
   - Its `16x16` token grid (`third_fp_4647e_window_0_region.npz`) extended left to canvas `x = 468`, covering `64–100%` of `second`'s face and hair on frames `4647–4689` (`100%` on `4673–4681`) and `100%` of `lead`'s face on `4662–4667`.
   - Inside that uncarved region on frames `4670–4685`, the DiT regenerated `second` as a bald head facing away. When `third_fp_4647e` was laid into `fun_0000_6394_joined_2026-10-25T1903Z_from_laid.mp4`, its region matte pasted that synthetic bald head over `second` in the `1903Z` video.
2. **Pass 2 (`shot_c_4647e` → `second_hfp_4647g`) ran perception on the contaminated `1903Z` cut**:
   - Running `preview` on `fun_0000_6394_joined_2026-10-25T1903Z_from_laid.mp4` meant Sapiens2-Seg and Sapiens2-Pose looked at the synthetic bald head baked into `1903Z` on frames `4670–4685` and found `0 px` of hair (`parts_held` dropped from `11,013 px` on `4647` to `966 px` on `4672` and `0 px` on `4674–4676`).
   - Meanwhile, comparing `third`'s SAM 3.1 `track.npz` on Pass 2 (`shot_c_4647e`) against Pass 1 (`third_fp_4647e`) shows `7` frames with `> 20%` IoU drift (`IoU = 0.000` on `4703–4708` where Pass 1 tracked `third` at `9,426–12,922 px` and Pass 2 lost him completely).
3. **Why `second`'s mask dropped out on `4694–4700` and bled into `PAD_79–80` (`4724–4725`)**:
   - **Speck on `4699` (`[3]`)**: Frame `4699` had `saved_part = 8 px` (`> 0`). Because `her_hair_in_hole()` in `bench/capture_masked_run.py` checked `not p.any()` (`== 0 px`) instead of `< PART_SPECK_MAX_PX` (`64 px`), frame `4699` was skipped by gap recovery (`0` blue cells, `0` orange cells) while `4698` (`0 px`) and `4700` (`0 px`) were recovered.
   - **Edge-shrunk `whole` reference on `4694` (`[4]`)**: Frame `4694` (`whole = 1,667 px`, `40%` of `4693`'s `4,209 px` as her head crossed behind `third`) was chosen as the nearest non-empty `whole` reference for `4695–4696`, shrinking the orange box to `5` cells (`4694–4695`) and `0` cells (`4696`, leaving a `[2]` blue-only frame).
   - **Missed SAM 3.1 `lead` on `4693–4725` (`[6]`)**: `lead` (`person#2` in SAM 3.1 `masks.npz`) stood inside `second`'s blue region on `33` frames (`4693–4725`, up to `6,526 px`), but `shot_c_4647e` only wrote a keep-out track for `third`, leaving `lead` uncarved (`green = 0`).
   - **Tail pad-bleed on `4724–4725` into `PAD_79, PAD_80` (`[1]`)**: On `4724–4725` (`sf 77–78`), `second` had already exited the frame (`saved_part = 0 px`, `2` trailing empty frames). Because `video_mask.py::HELD_EDGE_EMPTY_RUN` was `8` (`> 2`), `held_edge_source()` stepped backward to `4714` (`sf = 67`, `2,089 px` > `HELD_EDGE_LEAST = 1,024`) and copied `4714`'s mask into `PAD_79` and `PAD_80`, re-opening `23` orange cells on the tail after `second` had left.

### 2.2 Single-Subject Self-Swallow Under Occlusion (`second_hfp_4912g` & `lead_fp_2050g`)

On `second_hfp_4912g` (`4912–4946`, 35 frames), `second` is occluded twice by another person's arm (`person#4` in SAM 3.1): on `4921–4926` and `4940–4946`.
1. **Why `strangers` swallowed `second`'s own head (`[5]`)**:
   - In `bench/strangers.py::strangers_of` (`:163-169`) and `bench/strangers.py::run` (`:521-654`), any Sapiens person pixel outside the grown subject tracks (`people & ~tracks`) is marked `True` in `strangers.npz`.
   - When `second`'s SAM 3.1 `track` dropped to `0 px` on `4921–4926` and `4940–4946`, `~track` became the **entire `1024x768` frame**. Sapiens2-Seg still saw `second`'s own hair, face, and torso (`149,544–172,718 px`), so `Sapiens_person & ~track` dumped **her own body and head** into `strangers.npz`.
   - When `her_hair_in_hole()` (`bench/capture_masked_run.py:553`) computed `whole & ~strangers`, subtracting `strangers` erased `98–100%` of `whole` (`3,795 / 3,795 px` on `4925`, `4,009 / 4,009 px` on `4941`), leaving `0` orange cells (`[2]` blue-only frames on `4925` and `4941`) and starving `4921–4926` and `4940–4946`.
   - The exact same self-swallow hit `2026-10-24_lead_fp_2050g` on `11` frames (`2251–2257` and `2262–2265`, where `strangers` swallowed `4,530 / 4,530 px` = `100%` of `lead`'s `whole` head when the mirror rim interrupted her track).
2. **Speck on `4926` (`[3]`) and edge-shrunk `whole` on `4920, 4939` (`[4]`)**:
   - Frame `4926` had `saved_part = 8 px`, so `not p.any()` skipped it (`0` blue, `0` orange).
   - Frames `4920` (`whole = 1,419 px` vs `3,004 px` on `4919`) and `4939` (`whole = 928 px` vs `3,103 px` on `4938`) were edge-shrunk transition frames just before occlusion, poisoning the neighbor copy.

---

## 3. Audit of File Creation (`%w`) vs. Modification (`%y`) Dates Across `2026-10-25` Captures

Checking both creation (`stat -c '%w'`) and modification (`stat -c '%y'`) timestamps across `data/2026-10-25_*` (`bench/inspect_mask_steps.py` check `[0]`) surfaced three distinct provenance patterns:

1. **Captures built on `2026-10-25` that reused `2026-10-21` or `2026-10-22` preview tracks**:
   - `2026-10-25_lead_fp_2623b`: `parts.npz` was created and modified on `2026-10-25 14:43:35 -0500`, whereas `track.npz`, `head_track.npz`, and `classes.npz` were created on `2026-10-25 14:43:35` with preserved upstream mtimes (`%y`) from **`2026-10-21 17:17:04 -0500`** (`4` days older, from `2026-10-21_shot_c_2623_job`).
   - `2026-10-25_second_hfp_1039g/h` and `2026-10-25_second_hfp_5114j/k`: `parts.npz` was written on `2026-10-25` (`15:56` / `16:15`), while `track.npz` and `classes.npz` carried mtimes from **`2026-10-22`** (`14:15` / `14:21`). Because `1039` and `5114` had no prior laid pass on their spans, reusing the `2026-10-22` Pass-1 preview was benign there.
2. **Pass-2 capture built without running `strangers.py` (`2026-10-25_third_fp_4479e`)**:
   - `2026-10-25_third_fp_4479e` (`4479–4560`, Pass 2 on `1903Z` where `lead_fp_4479d` is laid) has **no `strangers.npz` on disk** and a `[3]` speck frame on `4538` (`saved_part = 8 px`).
3. **Why checking both `%w` (birth) and `%y` (mtime) matters**:
   - When `shutil.copy2` copies an older preview's `.npz` or `.mkv` into a new capture directory, `%y` preserves the old preview's mtime while `%w` records when the file was placed in the capture directory. `bench/inspect_mask_steps.py` now inspects both `birth` and `mtime` on every capture file and flags any `> 24h` skew (`stale_inputs`) between `track.npz`/`classes.npz` and `parts.npz`.

---

## 4. Audit of Parked / Forgotten Perception Models (`MoGe-2` Metric Depth & `Sapiens2` Matting)

### 4.1 `facebook_hyden-mogev2-metric-point` (`bench/measure_moge2_depth.py` → `depth.npz`)

- **What exists in the repo**:
  - `workflows/h3_config.py:137` defines `MOGE2_MODEL = "facebook_hyden-mogev2-metric-point/model.pt"`.
  - [`bench/measure_moge2_depth.py`](../../bench/measure_moge2_depth.py) runs MoGe-2 (`Ruicheng/moge-2-vitl-normal`) on the shot frames and writes `depth.npz` (`depth`, `mask`, `intrinsics`) into the capture directory.
  - [`bench/capture_masked_run.py`](../../bench/capture_masked_run.py) (`:236-267`, `:1383-1410`) **already wires `depth.npz` into `carve_keep_out`**: when `depth.npz` is present in the capture directory, pixels contested between the target subject (`part`) and a keep-out track (`other`) are compared in metric depth (`Z_target < Z_other - 0.02m`), and wherever the target subject is closer to the camera (`front`), the subject's mask is **kept** instead of carved out (`contested & ~front`).
- **What happened in practice**:
  - Across all `79` capture directories in `data/`, `depth.npz` was generated on **only two captures** (`data/2026-10-18_lead_fp_0751d/depth.npz` and `data/2026-10-19_lead_fp_0606c/depth.npz`), and was never invoked by `shot_to_load.py` or `load_chain.py` after `2026-10-19`:
    - On `2026-10-18_lead_fp_0751d`: `contested_px = 1,670`, `front_kept_px = 1,670` (`100%` of contested overlap settled by MoGe-2 depth; `0 px` wrongly carved).
    - On `2026-10-19_lead_fp_0606c`: `contested_px = 1,995`, `front_kept_px = 1,717` (`86%` kept in front), `carved_keep_out_px = 278`.
    - On `2026-10-25_second_hfp_4647g` (`depth_used = False`): `17,792` contested pixels between `second` and `third` were blindly carved out (`front_kept_px = 0`) because `measure_moge2_depth.py` was not run.
    - On `2026-10-25_third_fp_1306h` (`depth_used = False`): `149,666` contested pixels between `third` and `lead` were blindly carved out (`front_kept_px = 0`).
- **Rule going forward**:
  - `bench/inspect_mask_steps.py` `[0]` now reports `depth_used`, `contested_px`, `front_kept_px`, and `carved_keep_out_px` on every capture and flags any capture where `contested_px > 0` and `depth_used == False`.

### 4.2 `facebook_hyden-da2-relative-depth` & `Sapiens2ForImageMatting`

- **`facebook_hyden-da2-relative-depth`**: Not present in the repo (`0` references in code or `workflows/h3_config.py`); MoGe-2 (`facebook_hyden-mogev2-metric-point`) is the metric depth model used here.
- **`Sapiens2ForImageMatting` (`sapiens2_parts.py:36-39, 55-60`)**:
  - `sapiens2_parts.py` imports `Sapiens2ForImageMatting` alongside `Sapiens2ForSemanticSegmentation`, and defines `MATTING_WEIGHTS`, with `NO_MATTING = "none"` as the default when matting weights are not selected.
  - Every masked preview graph built by `workflows/masked_job_graph.py` passes `matting: "none"`, so alpha matting has never been active in the masked lane.

---

## 5. Environment Variables, Share Paths, Worktree Safety, and the Stray `./Video/` Directory

### 5.1 Canonical Environment Variables (`H3_OUTPUT_DIR`, `H3_INPUT_DIR`, `H3_MODEL_STORE`, `H3_CAPTURE_ROOT`)

| Variable | Role | Readers (`file:line`) | Fallback When Unset |
|---|---|---|---|
| `H3_OUTPUT_DIR` | Root output directory on the share (`<share>/output`, holding `Video/<video>/`) | `workflows/h3_config.py::output_dir` (`:2077-2085`), `bench/shot_to_load.py`, `bench/load_chain.py`, `bench/rebuild_delivery.py` | ComfyUI's local `output/` directory (`<comfy-root>/output`) |
| `H3_INPUT_DIR` | Root input directory on the share (`<share>/input`, holding working copies, `*_from_laid.mp4` cuts, and reference stills) | `bench/masked_job.py::input_dir` (`:160-171`), `bench/rebuild_delivery.py` | ComfyUI's local `input/` directory (`<comfy-root>/input`) |
| `H3_MODEL_STORE` | Local perception/pose model store | `bench/load_chain.py` (`os.environ.get("H3_MODEL_STORE")`) | `<comfy-root>/models` |
| `H3_CAPTURE_ROOT` | Optional capture root override | `subject_track.py`, `sapiens2_parts.py`, `video_mask.py` | Repo's `data/` directory (`Path(__file__).parent / "data"`) |

### 5.2 Why `./Video/fun/gemlead/` Appeared Inside the Repo on `2026-10-24`

- On `2026-10-24 20:51`, a session invoked `bench/shot_to_load.py` with a relative `--output Video/fun/gemlead` without `H3_OUTPUT_DIR` resolved against the share, creating `./Video/fun/gemlead/lead_fp_2050f` and `./Video/fun/gemlead/lead_fp_2050g` inside the repo root (`ComfyUI-h3-explorations/Video/`).
- Those two folders held only stale `2026-10-24` intermediate `.mkv` masks (`4.7 MB`), while the real renders and `.mkv` masks used by the server live on `<share>/output/Video/fun/gemlead/`. The stray `./Video/` directory has been removed.
- **Worktree rule**: Never run `shot_to_load.py`, `masked_job.py`, `load_chain.py`, or `rebuild_delivery.py` from inside a worktree (`h3_worktrees/<branch>`) without pointing `--shots`, `--plan`, and capture paths at the main repo's `data/` and setting `H3_INPUT_DIR` and `H3_OUTPUT_DIR` to the share. Worktrees are strictly for editing and testing code (`bench/check_*.py`) before landing on `main`.

---

## 6. Architectural Fixes Landed (`2026-10-25`)

### 6.1 `video_mask.py`: `HELD_EDGE_EMPTY_RUN = max(FRAME_PER_TOKEN)` (`4`)

- **File**: [`video_mask.py`](../../video_mask.py) (`:74-79`), tested in [`bench/check_video_mask.py`](../../bench/check_video_mask.py).
- **Change**: Lowered `HELD_EDGE_EMPTY_RUN` from `8` (`2 * max(FRAME_PER_TOKEN)`) to `4` (`max(FRAME_PER_TOKEN)`, one DiT temporal token).
- **Effect**: When a subject is genuinely absent at the start or end of a shot for `>= 4` frames (or at the tail where `saved_part == 0`), `held_edge_source()` holds the empty edge frame (`0 px`) instead of reaching back `11` frames (`4714 -> 4724..4725` on `4647g`) and copying an earlier mask into `PAD` frames.

### 6.2 `bench/capture_masked_run.py`: Class-Filtered `strangers`, Edge-Shrunk `whole` Filter, and `abs(k - src) <= 2`

- **File**: [`bench/capture_masked_run.py`](../../bench/capture_masked_run.py) (`:500-632`), tested in [`bench/check_capture_masked_run.py`](../../bench/check_capture_masked_run.py) (`60/60` cases passing).
- **Changes**:
  1. **Filter `strangers` by `whole_classes` (`Apparel` + non-subject `face_ids`) when recovering `hers`**:
     - Instead of subtracting raw `strangers` (`Sapiens_person & ~track`, which swallows the subject's own head whenever `track` drops to `0 px`), `her_hair_in_hole()` restricts `strangers` inside `hers` to pixels that Sapiens2-Seg classifies as **clothing/limbs (`Apparel`, IDs `8..22` plus limbs `4..7, 13..17, 21..22`)** or **another person's face outside `track` (`face_ids & ~track`)**.
     - On `second_hfp_4912g`, this recovers all `13` swallowed frames (`4921–4926` and `4940–4946`, restoring `3,795 px` on `4925` and `4,009 px` on `4941` while still carving the occluding arm's `Apparel`/limb pixels).
  2. **Exclude edge-shrunk `whole` frames (`< 0.75 * neighbor`) from serving as gap-fill references**:
     - `clean_whole` rejects any frame whose `whole` area is `< 75%` of an immediate neighbor (`4694` on `4647g`, `4920` and `4939` on `4912g`), so adjacent gap frames pull from the full unshrunk neighbor (`4693` or `4919`/`4938`).
  3. **Widen `near_h` to `abs(k - src) <= 2`**:
     - Allows `saved_part` to bridge across 2-frame transition gaps around a speck or occlusion boundary.

### 6.3 `bench/inspect_mask_steps.py` & `bench/check_inspect_mask_steps.py`

- **Files**: [`bench/inspect_mask_steps.py`](../../bench/inspect_mask_steps.py) and [`bench/check_inspect_mask_steps.py`](../../bench/check_inspect_mask_steps.py) (indexed in [`docs/checks.md`](../checks.md)).
- **What it checks on any capture directory**:
  - `[0]` **Input & Pass-Dependency Provenance**: Pass order vs. `laid.jsonl`, creation (`%w`) + modification (`%y`) dates and `> 24h` skew (`stale_inputs`), `strangers.npz` presence, `depth.npz` (`MoGe-2`) usage vs. contested keep-out pixels (`contested_px`, `front_kept_px`, `carved_keep_out_px`), and IoU drift between a prior laid pass's `track.npz` and the current pass's keep-out `track.npz`.
  - `[1]` **Tail / Lead Pad-Bleed**: `PAD_*` frames in `mask_steps_frames.json` carrying `orange > 0` while adjacent in-shot edge frames have `saved_part == 0`.
  - `[2]` **Blue-Only In-Shot Frames**: Frames where `blue > 0` and `orange == 0` (`saved_part == 0`), where the DiT receives no hole to regenerate.
  - `[3]` **Speck-in-Blue Frames**: Frames where `0 < saved_part < 64 px` (`PART_SPECK_MAX_PX`) inside a blue region, which bypass `not p.any()` gap recovery.
  - `[4]` **Edge-Shrunk Whole-Head Frames**: Frames where `whole < 0.75 * neighbor` poisons neighbor gap recovery.
  - `[5]` **`strangers` Self-Swallow**: Frames where `track == 0` causes `strangers` (`Sapiens_person & ~track`) to swallow `>= 50%` of the subject's own `whole` head.
  - `[6]` **Missed SAM 3.1 Strangers Inside Blue Region**: Non-target SAM 3.1 `masks.npz` objects (`>= 256 px`) inside the blue token region that are not covered by `green` (`keep_out`).
  - `[7]` **Per-Frame Gap Recovery & Visual Diagnostic Strip**: Runs `her_hair_in_hole()` before/after comparison and renders `--out-image` overlaying `blue`, `orange`, `green`, `whole`, and `strangers` on every flagged frame.
- **Hard gate (`--gate`)**: Exits `1` if any of `[1]` pad bleed, `[2]` blue-only `0 px` frame, `[3]` speck frame, or `[5]` `strangers` self-swallow is present.

### 6.4 Split-Provenance Rule for Multi-Subject Pass 2+

When running Pass 2 (or Pass 3) on a shot where Pass 1 is already laid in `data/<video>/laid.jsonl`:
1. **Target subject's own track and parts (`subject`)**: Should come from the **clean raw 24p working copy** whenever Pass 1's wide `16x16` token grid overlapped the target subject (`4647–4725`), so Pass 1's composite does not erase the target subject's hair or face in Sapiens/SAM 3.1.
2. **Already-laid subject's keep-out (`other`)**: Must cover the laid pass's actual rendered silhouette (union of the post-Pass-1 preview track and Pass 1's own capture track), and whenever `other` and `subject` overlap (`contested_px > 0`), run `bench/measure_moge2_depth.py` on the raw source frames so `depth.npz` settles front/back occlusion by metric depth (`Z_subject < Z_other - 0.02m`) instead of blindly carving the front subject.
