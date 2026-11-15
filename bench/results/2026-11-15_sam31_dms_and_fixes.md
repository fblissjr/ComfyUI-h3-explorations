# SAM 3.1 Decoupled Multiplex Memory Selection (`SAM3-DMS`), multimask pointer fix, and detector presence head (2026-11-15)

lane: masked video to video
verdict: verified on CPU checks (`24` unit cases + `7` memory-selection cases) and on the RTX 4090 with ComfyUI's `ModelPatcherDynamic` (`sam3.1_multiplex_fp32.safetensors`, `MF_THRESHOLD=0.01`). On `N_obj == 1` (`kitchen_face_680_760`), the updated `attach(..., memory_selection=True)` matches legacy single-object behavior at `diff_px = 0` (`first_frame_back_from_708 = 709`). Across 5 multi-object (`N_obj >= 2`) windows vs per-object `alone` tracks: on late-window occlusion `kitchen_occ_960_1009`, full Multiplex DMS (`MemoryEncodeFilter` + spatial union) recovers `strangers` post-occlusion focus IoU (`998..1009`) from `0.0000` (`mem_off` / `focus0_legacy` / `dms_read_only`) to `0.8333` (`overall mean_iou 0.6888 -> 0.7608`, `focus_iou 0.6637 -> 0.9415`, `0` companion jumps); on `fun_crowd_2200` (`9` people), it maintains `0.8860` (`0.8899` with `multimask_ptr`) vs `0.8843` (`mem_off`) with `0` jumps; on `regain_1419_1465` and `hall_1306_1418`, it holds/slightly improves focus IoU with `0` degraded frames; and on `kitchen_face_group_680_760`, write-time gating on `79` low-score `strangers` slot-frames trades `strangers` similarity-vs-alone (`0.7147 -> 0.6343`, `degraded: 2 -> 14`) while slightly improving `lead_face` (`0.7832 -> 0.7843`, `0.7940` with `multimask_ptr`). `correct_multimask_ptr=True` triggers `argmax(iou) != 0` on `28.75%–87.50%` of propagation steps (`IoU 0.9967–0.9996` vs `off`). `correct_detector_presence=True` drops false-positive `>0.3` detections on 4 absent prompts (`19/23/32/26 -> 0`) and absent `SAM3_VideoTrack` tracks (`5 -> 0`) while keeping 100% of present-prompt detection counts (`20/16/20`). Both `correct_multimask_ptr` and `correct_detector_presence` default to `False` for pipeline backwards compatibility.

## 1. What was changed and measured

- **Node**: [`MiniMaxH3SAM31Corrections`](../../sam31_corrections.py) (`sam31_corrections.py`):
  1. **Multiplex `SAM3-DMS` under `memory_selection=True` (default `True`)**:
     - Preserves the single-object (`N_obj == 1`) path identically (`diff_px = 0`).
     - For `N_obj >= 2` on `SAM31Tracker`, adds write-time slot gating (`MemoryEncodeFilter` on `_encode_new_memory`), read-time any-visible spatial union (`focus_obj_idx=None`, `_spatial_union_frames`), per-slot `obj_ptr` assembly across multiplex buckets (`_build_remapped_non_cond`), and state reset (`MemoryTrackLoopGuard` on `track_video_with_detection`).
  2. **Propagation multimask pointer selection (`correct_multimask_ptr=False`, `optional=True`)**:
     - Installs `MultimaskTokenTap` + `CorrectedPropagationPtr` on `SAM31Tracker._forward_propagation` to select `tokens_obj[obj_range, argmax(iou_obj)]` instead of ComfyUI's fixed branch `0`.
  3. **Detector global presence gating (`correct_detector_presence=False`, `optional=True`)**:
     - Installs `CorrectedDetectorPresence` on `SAM3Detector._detect` to multiply `sigmoid(presence_logit)` into prompt scores before thresholding (`torch.logit(prob, eps=1e-6).clamp(-12.0, 12.0)` in `float32`).
- **Hardware & server state** (via `bench/hwinfo.py`):
  - **CPU**: `13th Gen Intel(R) Core(TM) i9-13900K`, `125.5 GiB` RAM.
  - **GPU**: `NVIDIA GeForce RTX 4090` (`24564 MiB`), driver `620.70.06`, power limit `360.00 W` (vs `450.00 W` stock — wall-clock seconds below are internal to this run only).
  - **Model**: `sam3.1_multiplex_fp32.safetensors`, `dynamic_vram=true`, patcher `ModelPatcherDynamic`, memory score cutoff `MF_THRESHOLD = 0.01`.
- **Scripts & raw captures**:
  - Benchmark script: [`bench/measure_sam31_dms_and_fixes.py`](../measure_sam31_dms_and_fixes.py) (`total_elapsed_s = 659.36 s`).
  - GPU card patcher test: [`bench/check_sam31_corrections_on_card.py`](../check_sam31_corrections_on_card.py).
  - Face probe script: [`bench/sam_memory_selection.py`](../sam_memory_selection.py) `--face-probe`.
  - Raw JSON outputs (untracked under `data/`): `data/2026-11-15_sam31_dms_and_fixes/benchmark_summary.json` and `data/2026-11-15_sam31_dms_and_fixes/face_probe_through_node.json`.

---

## 2. Commands run

```bash
CUDA_VISIBLE_DEVICES="" "$PY" bench/check_sam31_corrections.py
CUDA_VISIBLE_DEVICES="" "$PY" bench/check_sam_memory_selection.py
"$PY" bench/check_sam31_corrections_on_card.py
"$PY" bench/sam_memory_selection.py --face-probe --out-dir data/2026-11-15_sam31_dms_and_fixes
"$PY" bench/measure_sam31_dms_and_fixes.py --out-dir data/2026-11-15_sam31_dms_and_fixes
```

---

## 3. Single-object (`N_obj == 1`) bit-identity gate & `--face-probe`

Source window `680..760` (`81` frames, `lead_face` seed `812 px` from `fun/2026-10-17_lead_fp_0613g/subjects/lead/masks__shot_c_0604b.npz` `parts[67]`).

- **In-process `N_obj == 1` parity gate** (`data/2026-11-15_sam31_dms_and_fixes/benchmark_summary.json`, `n_obj_1_bit_identity_gate`):
  - Comparing current `attach(..., image_range=True, memory_selection=True)` against legacy `N_obj == 1` (`focus_obj_idx=0, one_object_only=True`, no `ENCODE_MEM_PATCH` or `TRACK_LOOP_PATCH`): `diff_px = 0`, `byte_identical = true`, `first_frame_back_from_708 = 709`.
- **`--face-probe` 4-arm comparison against `2026-10-17`**:
  - `parts_sha256 = 337afd3ecb01215f477d916ca946910528951b8ffcda3213389702416e0ffd9a` (matches `data/fun/2026-10-17_sam_memory_selection/face_probe_through_node.json`).

| Arm | `first_frame_back_from_708` (`2026-10-17`) | `first_frame_back_from_708` (`2026-11-15`) | `elapsed_s` (`2026-11-15`) |
|---|---|---|---|
| `range_as_comfyui__memory_off` | `715` | `715` | `5.249` |
| `range_as_comfyui__memory_on` | `710` | `710` | `4.785` |
| `range_corrected__memory_off` | `714` | `714` | `4.850` |
| `range_corrected__memory_on` | `709` | `709` | `4.824` |

Small `1–4 px` boundary area differences (`face_area_px`) appear equally on the unpatched stock `range_as_comfyui__memory_off` arm vs the `2026-10-17` JSON (e.g., `1745` vs `1746`, `2042` vs `2044`), reflecting upstream ComfyUI/CUDA runtime drift between dates while all recovery frames match identically.

---

## 4. Experiment A: Multi-object Multiplex `SAM3-DMS` vs per-object `alone`

Four scene windows from `bench/sam_memory_selection.py::build_default_scenes()` plus an auto-detected crowd window from `$H3_INPUT_DIR/mrpop/fun_0000_6394_24p_sound_in_step.mov` (`SAM3_Detect("person:12", threshold=0.45)` + `subject_tracks.one_each` on candidate frames, picking source frame `2200` with `9` people sorted smallest-first: `36500`–`90487` px). All arms run with `image_range=True, text_activation=False` and compare per-object masks against running each object alone with `dms_full` (`N_obj == 1`).

### Summary across all 5 windows

| Window (frames / `N_obj` / `pick`) | Arm | `mean_iou` | `focus_iou` | `min_obj_iou` | `degraded (<0.5)` | `jumps` | `low_scores (<=0.01)` | `elapsed_s` |
|---|---|---|---|---|---|---|---|---|
| `regain_1419_1465` (`47f`, `N=3`, `pick=12`, focus `1435..1465`) | `mem_off` | `0.9461` | `0.9844` | `0.8642` | `0` | `0` | `0` | `2.872` |
| | `focus0_legacy` | `0.9452` | `0.9854` | `0.8642` | `0` | `0` | `21` | `2.828` |
| | `dms_read_only` | `0.9460` | `0.9862` | `0.8647` | `0` | `0` | `21` | `2.842` |
| | `dms_full` | `0.9460` | `0.9862` | `0.8647` | `0` | `0` | `21` | `2.847` |
| | `dms_full_plus_ptr` | `0.9460` | `0.9865` | `0.8650` | `0` | `0` | `21` | `2.853` |
| `hall_1306_1418` (`113f`, `N=2`, `pick=4`, focus `1392..1418`) | `mem_off` | `0.9905` | `0.9716` | `0.9876` | `0` | `0` | `0` | `6.883` |
| | `focus0_legacy` | `0.9906` | `0.9717` | `0.9876` | `0` | `0` | `0` | `6.859` |
| | `dms_read_only` | `0.9906` | `0.9717` | `0.9876` | `0` | `0` | `0` | `6.865` |
| | `dms_full` | `0.9906` | `0.9717` | `0.9876` | `0` | `0` | `0` | `6.872` |
| | `dms_full_plus_ptr` | `0.9905` | `0.9713` | `0.9874` | `0` | `0` | `0` | `6.903` |
| `kitchen_occ_960_1009` (`50f`, `N=3`, `pick=0`, focus `998..1009`) | `mem_off` | `0.6888` | `0.6637` | `0.0852` | `0` | `0` | `0` | `3.016` |
| | `focus0_legacy` | `0.6904` | `0.6635` | `0.0903` | `0` | `0` | `55` | `3.012` |
| | `dms_read_only` | `0.6904` | `0.6636` | `0.0903` | `0` | `0` | `55` | `3.013` |
| | **`dms_full`** | **`0.7608`** | **`0.9415`** | **`0.3034`** | `0` | `0` | `58` | `3.020` |
| | `dms_full_plus_ptr` | `0.7541` | `0.9417` | `0.2834` | `0` | `0` | `58` | `3.016` |
| `kitchen_face_group_680_760` (`81f`, `N=3`, `pick=0`, focus `684..760`) | `mem_off` | `0.8305` | `0.8266` | `0.7147` | `3` | `0` | `0` | `4.915` |
| | `focus0_legacy` | `0.8129` | `0.8079` | `0.6614` | `6` | `0` | `83` | `4.908` |
| | `dms_read_only` | `0.8265` | `0.8222` | `0.6947` | `4` | `0` | `78` | `4.907` |
| | `dms_full` | `0.8037` | `0.7985` | `0.6343` | `16` | `0` | `101` | `4.914` |
| | `dms_full_plus_ptr` | `0.8071` | `0.8021` | `0.6347` | `16` | `0` | `101` | `4.918` |
| `fun_crowd_2200` (`60f`, `N=9`, `pick=0`, focus `2200..2259`) | `mem_off` | `0.8843` | `0.8843` | `0.7232` | `30` | `0` | `0` | `3.678` |
| | `focus0_legacy` | `0.8848` | `0.8848` | `0.7428` | `28` | `0` | `88` | `3.670` |
| | `dms_read_only` | `0.8876` | `0.8876` | `0.7428` | `27` | `0` | `86` | `3.663` |
| | `dms_full` | `0.8860` | `0.8860` | `0.7414` | `29` | `0` | `91` | `3.665` |
| | `dms_full_plus_ptr` | `0.8899` | `0.8899` | `0.7393` | `28` | `0` | `87` | `3.669` |

`alone` reference totals across objects: `regain_1419_1465` `8.429 s`; `hall_1306_1418` `13.675 s`; `kitchen_occ_960_1009` `8.769 s`; `kitchen_face_group_680_760` `14.598 s`; `fun_crowd_2200` `32.541 s` (`~8.9x` slower than `3.665 s` multiplex).

### Per-object breakdown on `kitchen_occ_960_1009` and `kitchen_face_group_680_760`

- **`kitchen_occ_960_1009`** (seed areas: `lead 192728 px`, `second 97813 px`, `strangers 1350 px`):
  - Without `MemoryEncodeFilter`, all three arms lose `strangers` in the post-occlusion focus window `998..1009`:
    - `mem_off`: `lead 0.9938` (`focus 0.9962`), `second 0.9875` (`0.9948`), `strangers 0.0852` (**`focus 0.0000`**).
    - `focus0_legacy`: `lead 0.9938`, `second 0.9870`, `strangers 0.0903` (**`focus 0.0000`**).
    - `dms_read_only`: `lead 0.9938`, `second 0.9871`, `strangers 0.0903` (**`focus 0.0000`**).
  - With write-time slot gating:
    - **`dms_full`**: `lead 0.9941` (`focus 0.9968`), `second 0.9848` (`0.9945`), **`strangers 0.3034` (`focus 0.8333`)**.
    - `dms_full_plus_ptr`: `lead 0.9940` (`0.9968`), `second 0.9848` (`0.9949`), `strangers 0.2834` (`focus 0.8333`).
  - `dms_read_only` vs `dms_full` shows write-time `MemoryEncodeFilter` is the load-bearing change for post-occlusion recovery on `SAM31Tracker`.
- **`kitchen_face_group_680_760`** (seed areas: `lead_face 812 px`, `second 66954 px`, `strangers 734 px`):
  - `mem_off`: `lead_face 0.7832` (`degraded 1`), `second 0.9937`, `strangers 0.7147` (`degraded 2`: `[742, 744]`).
  - `dms_read_only`: `lead_face 0.7912` (`degraded 1`), `second 0.9937`, `strangers 0.6947` (`degraded 3`, `low_scores 60`).
  - `dms_full`: `lead_face 0.7843` (`degraded 2`: `[686, 748]`), `second 0.9926`, `strangers 0.6343` (`degraded 14`, `low_scores 79`).
  - `dms_full_plus_ptr`: `lead_face 0.7940` (`degraded 2`), `second 0.9925`, `strangers 0.6347` (`degraded 14`, `low_scores 79`).
  - Trade-off: on a tiny background `strangers` seed (`734 px`) whose per-frame score falls `<= 0.01` on `79/81` frames, write-time slot suppression drops `strangers` similarity to the single-object `alone` run (`0.7147 -> 0.6343`) while keeping `lead_face` and `second` intact and causing `0` companion jumps.

---

## 5. Experiment B: `correct_multimask_ptr` (`False` vs `True`)

Run on top of `memory_selection=True, image_range=True, text_activation=False`. Records how often propagation steps pick a non-zero multimask branch (`argmax(iou_pred) != 0`), resulting mask IoU between `off` and `on`, and frames shifted (`IoU < 0.999`).

| Case | Prop steps | Branch counts `[b0, b1, b2]` | `argmax != 0` | Mean IoU (`off` vs `on`) | Changed frames (`< 0.999`) | Time (`off -> on`) |
|---|---|---|---|---|---|---|
| `kitchen_face_680_760_N1` (`81f`, `1` obj) | `80` | `[57, 1, 22]` | `23` (`28.75%`) | `0.9977` | `37` | `4.851 -> 4.864 s` |
| `hall_third_1306_1418_N1` (`113f`, `1` obj) | `112` | `[14, 98, 0]` | `98` (`87.50%`) | `0.9980` | `9` | `6.842 -> 6.874 s` |
| `hall_group_1306_1418_N2` (`113f`, `2` objs) | `224` | `[33, 103, 88]` | `191` (`85.27%`) | `0.9996` (`0.9998 / 0.9993`) | `3 / 10` | `6.871 -> 6.886 s` |
| `regain_second_1419_1465_N1` (`47f`, `1` obj) | `46` | `[13, 4, 29]` | `33` (`71.74%`) | `0.9996` | `6` | `2.830 -> 2.834 s` |
| `regain_group_1419_1465_N3` (`47f`, `3` objs) | `138` | `[24, 60, 54]` | `114` (`82.61%`) | `0.9967` (`0.9972 / 0.9956 / 0.9974`) | `21 / 13 / 23` | `2.865 -> 2.865 s` |

On `kitchen_face_680_760_N1`, `first_frame_back_from_708` stays `709` both `off` and `on`. Because the fix nudges mask contours (`IoU 0.9967–0.9996`) with mixed small moves across windows (`fun_crowd_2200` `0.8860 -> 0.8899`, `kitchen_occ_960_1009` `0.7608 -> 0.7541`), `correct_multimask_ptr` is exposed as `optional=True, default=False`.

---

## 6. Experiment C: `correct_detector_presence` (`False` vs `True`)

Both arms run with `image_range=True, text_activation=True, memory_selection=True`.

### `SAM3_Detect` across 8 frames (`620, 700, 1000, 1320, 1440, 2000, 3000, 4000`, `threshold=0.0, refine_iterations=0`)

| Group | Prompt | `presence_off` (`>0.3 / >0.4 / >0.5`) | `presence_off` `max / mean_top1` | `presence_on` (`>0.3 / >0.4 / >0.5`) | `presence_on` `max / mean_top1` |
|---|---|---|---|---|---|
| Absent | `astronaut:8` | `19 / 15 / 11` | `0.7837 / 0.6484` | **`0 / 0 / 0`** | `0.0079 / 0.0024` |
| Absent | `bicycle:8` | `23 / 8 / 2` | `0.5674 / 0.4551` | **`0 / 0 / 0`** | `0.0158 / 0.0053` |
| Absent | `dinosaur:8` | `32 / 17 / 10` | `0.8198 / 0.5723` | **`0 / 0 / 0`** | `0.0369 / 0.0102` |
| Absent | `umbrella:8` | `26 / 16 / 8` | `0.7861 / 0.6490` | **`0 / 0 / 0`** | `0.2344 / 0.0367` |
| Present | `person:8` | `20 / 19 / 17` | `0.9883 / 0.9694` | `20 / 19 / 17` | `0.9727 / 0.9504` |
| Present | `woman:8` | `16 / 16 / 15` | `0.9863 / 0.9578` | `16 / 16 / 15` | `0.9717 / 0.9285` |
| Present | `head:8` | `20 / 18 / 17` | `0.9512 / 0.9303` | `20 / 18 / 17` | `0.9458 / 0.9191` |

### `SAM3_VideoTrack` (`24` frames `620..643`, `detection_threshold=0.4`, `max_objects=16`, `detect_interval=6`)

- `"astronaut, dinosaur"`: `presence_off` spawned `5` tracks (`[0.6738, 0.6299, 0.6265, 0.5215, 0.5713]`); `presence_on` spawned **`0` tracks (`[]`)**.
- `"person, astronaut, dinosaur"`: `presence_off` spawned `5` tracks (`[0.9512, 0.9131, 0.6738, 0.6558, 0.5713]`); `presence_on` spawned **`4` tracks (`[0.9302, 0.8931, 0.6411, 0.4658]`)**.

`correct_detector_presence` is exposed as `optional=True, default=False` so existing masked jobs (especially `MiniMaxH3SubjectTrack` threshold calibration on marginal `person` crops) remain unchanged unless opted in.

---

## 7. What it does not show

- **No H3 diffusion renders**: no sampled `fun` / `fun2` videos were generated or regression-tested against this branch.
- **No human-labeled multi-object ground truth**: Experiment A measures agreement against per-object `alone` tracks (`N_obj == 1` with `dms_full`), not manual pixel annotations.
- **No threshold retuning under `correct_detector_presence=True`**: present-prompt `mean_top1` scores shift down slightly (`person:8` `0.9694 -> 0.9504`, `woman:8` `0.9578 -> 0.9285`), and borderline detection shots in `subject_track.py` were not swept.
- **Non-stock GPU power limit**: timings were captured under `360.00 W` cap (`hwinfo.py`) and compare only arms within this run.
