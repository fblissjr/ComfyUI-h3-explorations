# Zero-Code GPU Trio + Kitchen Ensemble `Ref2VA` Verification

**Date:** `2026-11-23`  
**Canvas / Length / Backend:** `1152x768`, `141` frames (`5.88 s @ 24 fps`, dual-clock aligned), `MiniMax-H3` `fl2va` & `ref2va`, `MiniMaxH3Sol`, `Pika SVD995` (`6` steps) + `AudioRefine` undistilled base pass  
**Runner & Outputs:** [`run_zero_code_gpu_trio_2026_11_23.py`](../run_zero_code_gpu_trio_2026_11_23.py) · untracked metrics under `data/zero_code_trio_2026_11_23/` (`summary.json`, `trio3_dinov3.json`, `trio3_yin_f0.json`, `trio3_ensemble_dinov3.json`, `trio3_ensemble_yin_f0.json`) · renders under `$H3_OUTPUT_DIR/Video/zero_code_trio_2026_11_23/` (`verdict: verified`)

---

## Summary of Findings

1. **Trio 1 (`MiniMaxH3FreezeAudio` `mask=0.0` vs. `mask=0.25` vs. 2-Pass Joint `FrozenVideoCache(int4)` + `AudioRefineMask`):** **VERIFIED.**
   - `trio1b` (`audio_mask = 0.25`) preserved the `118 BPM` kick/snare/hi-hat envelope cleanly inside decoded model audio (`env_corr_vs_groove = 0.9985`, `-21.87 dBFS`) with zero tempo drift or noise bursts while adapting visual arm strike motion (`trio1a_vs_1b_video_mae = 19.9379`).
   - `trio1c` paired 6-step Pika joint sampling (`74.99 s` total) with undistilled Pass-2 `MiniMaxH3FrozenVideoCache(int4)` + `MiniMaxH3AudioRefineMask`: Pass-1 vs. Pass-2 video stayed pixel-locked at the `crf=19` H.264 floor (`trio1c_pass1_vs_pass2_video_mae = 1.8641`, `interframe_mae_mean = 9.145` on both), while Pass-2 replaced the Pika audio stream with crisp street bucket, pot, and splash-cymbal rimshots (`RMS -11.07 dBFS`, `peak +2.03 dBFS` vs. Pass-1 `+0.68 dBFS`).
2. **Trio 2 (`Tier 2` Zero-Swap `P-H3-Image LoRA` + `MiniMaxH3GuideLatent` @ `f68`, `guide_end_percent = 1.00` vs. `0.33`):** **VERIFIED.**
   - Generated pixel-locked `f0` morning conservatory (`trio2_t2i_first_frame_00001_.png`) and `f68` stormy indigo twilight + overhead amber grow-lamp (`trio2_edit_guide_f068_00001_.png`, cross-anchor `MAE = 83.128`) without model reloads (`114.47 s` including anchor stills; `54.95 s` cached second video arm).
   - Both arms tracked the lighting shift and clean glass atomizer pickup (`00:00..00:06`), with `guide_end_percent = 0.33` (`trio2b`) moving more smoothly through `f68` (`f64_to_68_mae = 3.650` vs. `1.639` hesitation in `trio2a`) and ending cleanly (`second_half MAE` vs. `2a = 14.402`) without a duplicated still pose.
3. **Trio 3 & Trio 3 Ensemble (`M3` `MiniMaxH3ReleaseReferences` + `attach_lse_reference_balance` across Single Sprinter & Three-Cook Restaurant Pass):** **VERIFIED.**
   - **Single-subject track sprint (`woman_blonde.jpg`, `141f`, spoken line `<d>[English] Clear the lane, coming through!</d>`):** All three arms (`end=1.00/clean=0.999`, `end=0.33/clean=0.999`, `end=0.50/clean=0.900`) launched cleanly out of starting blocks into a front close-up sprint with `0` copy-freeze warnings (`ref_cos_mean = 0.140..0.143`) and negligible paired pitch shift (`+0.034 st` at `0.33`, `-0.106 st` at `0.50`).
   - **Three-cook two-speaker restaurant pass (`$H3_INPUT_DIR/t2i_30_photoreal_kitchen_three_cooks_seed11.png`, `141f`, `(S1)` + `(S2)` dialogue + bell):** Tested on a multi-subject composite still, `end_percent = 1.00` (`trio3d`, `99.08 s`) tightly locked `f0` onto the input still pose (`DINOv3 ref_cos_f0 = 0.9763`), whereas `end_percent = 0.33` (`trio3e`, `57.55 s`) dropped rigid `f0` pose lock (`ref_cos_f0 = 0.8796`, `-9.9%`) so `<Subject 1>` opened naturally into plating/speech at `f0` while keeping all three subjects' likenesses, both spoken lines, overhead wok flare, brass bell chime, and paired F0 (`-0.061 st`, `0` triage notes) intact.

---

## 1. Trio 1: `audio_freeze` Loose Mask (`0.25`) & `AudioRefineMask` + `FrozenVideoCache(int4)`

Prompt: `TRIO1_PROMPT` (`141f`, `1152x768`, `seed = 4409`), input WAV `$H3_INPUT_DIR/h3_trio_groove_20261123.wav` (`32 kHz` stereo `118 BPM` drum+bass groove).

| Arm | Execution | `audio` `RMS / Peak` (`dBFS`) | `env_corr` vs. Groove | `interframe_mae_mean` (`h1 / h2`) | Visual & Audio Inspection (`read_file`) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `trio1a_freeze_mask000` (`mask=0.0`) | `62.17 s` | `-21.93 / -1.92` | `0.9999` | `9.644` (`8.661 / 10.627`) | Strict muxed groove; crisp orbital camera past wet stone arch (`00:00..00:06`) |
| `trio1b_freeze_mask025` (`mask=0.25` muxed) | `51.41 s` | `-21.93 / -1.92` | `0.9999` | `10.065` (`8.862 / 11.268`) | Re-adjusted stick arc (`trio1a_vs_1b MAE = 19.9379`) to fit drum groove |
| `trio1b_freeze_mask025_decoded` (`mask=0.25` decoded) | *(same)* | `-21.87 / -0.52` | **`0.9985`** | `10.065` | AudioVAE decode preserved `118 BPM` kick/snare/hat timing with warmer room blend |
| `trio1c_pass1_pika6` (Joint 6-step Pika) | `74.99 s` *(both)* | `-11.05 / +0.68` | `-0.0411` | `9.145` (`8.433 / 9.857`) | Coherent street percussion video + synthesized acoustic bucket hits |
| `trio1c_pass2_audiorefine` (`FrozenVideoCache(int4)` + `AudioRefineMask`) | *(incl. above)* | `-11.07 / +2.03` | `+0.0319` | `9.145` (`pass1_vs_pass2 MAE = 1.8641`) | Video pixel-locked to Pass-1; Pass-2 refiner sharpened metallic pot & cymbal transients |

---

## 2. Trio 2: Embedded `P-H3-Image LoRA` Tier 2 + `GuideLatent` (`f68`)

Scene: Victorian glasshouse botanist trimming an orchid in daylight -> stormy twilight rain + overhead tungsten grow lamp (`141f`, `1152x768`, `seed = 5817`). Anchor stills: `trio2_t2i_first_frame_00001_.png` and `trio2_edit_guide_f068_00001_.png` (`anchors_mae = 83.128`).

| Arm (`guide_end_percent`) | Execution | `f0` vs `T2I` MAE | `f68` vs `Guide` MAE | `f64->f68` / `f68->f72` MAE | `f140` vs `Guide` MAE | `interframe_mae_mean` | Visual Inspection (`read_file`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `trio2a_guide_end_100` (`1.00`) | `114.47 s` *(incl. anchors)* | `12.984` | `9.124` | `1.639 / 4.963` | `35.601` | `3.268` | Lamps/storm arrive crisp by `f48`; slight pause around `f64..68` before grabbing spray bottle |
| `trio2b_guide_end_033` (`0.33`) | `54.95 s` *(cached)* | `13.288` | `16.898` | **`3.650`** `/ 5.526` | `35.563` | `3.203` | Smoother motion through `f68` (`2a_vs_2b`: `h1=10.054`, `h2=14.402`); same clean orchid misting finish |

---

## 3. Trio 3 + Kitchen Ensemble: `M3` `ReleaseReferences` & `DINOv3` / `YIN F0` Probes

All `5` `ref2va` runs used `size_policy="tiered"`, `qwen_view="separate"`, `MiniMaxH3Sol`, and `141` frames at `1152x768`.

| Arm (`ref`, `end_percent / clean_level`) | Exec | `audio` `RMS / Peak` | `DINOv3` `ref_cos` (`f0 / mean`) | `DINOv3` `temp_self` (`mean / min`) | `YIN F0` `global / paired` (`voiced`) | Visual & Audio Inspection (`read_file`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `trio3a_ref_end100_clean999` (`woman_blonde.jpg`, `1.00 / 0.999`) | `80.83 s` | `-24.26 / -4.41` | `0.123 / 0.142` | `0.937 / 0.791` | `493.4 Hz` ref (`0.172`) | Clean block explosion + `"Clear the lane, coming through!"` (`00:04`) |
| `trio3b_ref_end033_clean999` (`woman_blonde.jpg`, `0.33 / 0.999`) | `56.55 s` | `-21.85 / -1.51` | `0.103 / 0.143` | `0.940 / 0.824` | `-0.143 / +0.034 st` (`0.186`) | Matches likeness/stride without statue lock; `0` DINOv3/YIN triage notes |
| `trio3c_ref_end050_clean900` (`woman_blonde.jpg`, `0.50 / 0.900`) | `59.73 s` | `-23.00 / -3.42` | `0.107 / 0.140` | `0.930 / 0.771` | `-0.580 / -0.106 st` (`0.189`) | Strong motion blur + coach reappears edge-right at `00:06`; `0` warnings |
| `trio3d_kitchen_end100` (`t2i_30_photoreal_kitchen_three_cooks_seed11.png`, `1.00 / 0.999`) | `99.08 s` | `-10.96 / +2.32` | **`0.976` / `0.842`** | `0.933 / 0.877` | `354.7 Hz` ref (`0.431`) | Holds `f0` tightly to composite still; clean `(S1)` + `(S2)` dialogue & bell ding |
| `trio3e_kitchen_end033` (`t2i_30_photoreal_kitchen_three_cooks_seed11.png`, `0.33 / 0.999`) | `57.55 s` | `-10.97 / +1.71` | **`0.880`** `/ 0.793` | `0.925 / 0.827` | `-0.311 /` **`-0.061 st`** (`0.402`) | Relaxes `f0` pose-freeze while keeping all 3 chefs, speech, flames & bell aligned |
