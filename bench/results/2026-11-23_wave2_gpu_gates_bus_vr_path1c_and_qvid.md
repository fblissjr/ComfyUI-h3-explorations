# Wave 2 GPU Gates: `Bus-Scoped V_r` A/B, `4`-Winner `192f`/`345f` Showcases, `Path 1C ColQwen`, and `1x Q_vid[mask]` vs. `2x Sol`

**Date:** `2026-11-23`  
**Seed / Canvas / Backend:** `seed = 730451892`, `1152x768`, `6`-step Pika SVD995 (`MiniMax-H3`, `MiniMaxH3Sol`, `split_backend = "auto_1x_kv"`)  
**Ledgers & Outputs:** [`data/exp_unpooled_sliders/wave2_2026_11_23/`](../../data/exp_unpooled_sliders/wave2_2026_11_23/) (`gate2a_results.json`, `gate2b_path1c_results.json`, `gate2c_qvid_results.json`) · `$H3_OUTPUT_DIR/Video/wave2_2026_11_23/` (`gate2a/`, `gate2b/`, `gate2c/`)

---

## Summary of Empirical Decisions

1. **Gate 2A (`Bus-Scoped V_r`):** **REJECTED for `bank/` promotion.** Projecting cross-bus erasure `V_r` orthogonal to top Look PCA directions (`pc8`/`pc16` via `compute_bus_erasure_basis` and `project_erasure_orthogonal_to_basis` in [`control_vectors.py`](../../control_vectors.py#L1513-L1552)) caused severe white-magenta exposure blowout in `blacklight_uv_rave + reverb_cathedral` (`mid_laplacian_var 43.44 -> 7.95 / 17.38`) at `f96`, whereas shipped weights under runtime `peer_ros=True` kept both `uv+reverb` and `chrome+zeroG` visually intact. `../../models/control_vectors/bank/` (`137` `.safetensors`) stays unmodified.
2. **Gate 2A (`4`-Winner `192f` Adult Matrix, `1x` Stream/Band Routing, & `345f` Sol Parity):** **VERIFIED** across bright [`prompt_bank/t2va_restaurant_kitchen.txt`](../../prompt_bank/t2va_restaurant_kitchen.txt) and dim [`prompt_bank/t2va_stairwell_dialogue.txt`](../../prompt_bank/t2va_stairwell_dialogue.txt), including the **first live GPU execution of `streams="audio_only" + block_band="late_30_49"` in `1x`** (`rgb_mae_vs_base = 7.461`, spectral centroid `2505.2 -> 2687.3 Hz`) and clean `345f` `MiniMaxH3Sol` completion without graph breaks or seam collapse. Visual inspection notes scene-dependent partiality on dim `stairwell` for `charcoal_sketch_animated` (`rgb_mae = 39.27`, eye/cheek hatching vs. full stylization on `kitchen` at `77.30`) and `x_ray_fluoroscope` (half-jaw skeletal overlay vs. full internal skull/throat glow on `kitchen`).
3. **Gate 2B (`Path 1C ColQwen` `target="vision_tokens"` Dual-Lock Probe):** **EMPIRICAL NULL / FAIL.** Tag-0 `<Picture 1>` `vision_tokens` steering + `MiniMaxH3ReleaseReferences(0.17, clean_level=0.90/0.85)` left warm skin, blonde hair, and green awning saturated (`hsv_sat = 0.2432` vs. base `0.2477`) because DiT self-attention still reads raw `reference_latent` VAE pixels. Runtime `target="vision_tokens"` remains cleanly wired and CPU-tested in [`control_vector_nodes.py`](../../control_vector_nodes.py#L93) and [`h3_spans.py`](../../h3_spans.py#L81); the scratch vector (`noir_colqwen_p1c`) is **not** promoted to `bank/`.
4. **Gate 2C (`1x Q_vid[mask]` vs. `2x MiniMaxH3Sol EditGuider`):** **CLEAR FAIL (`temporal_pass = false`, `portal_pass = false`).** Dense `Q_vid[mask]` splicing delivered only `1.047x–1.090x` portal contrast (`vs. 1.421x` crisp localized charcoal inside `2x EditGuider`) with severe bidirectional `K_vid, V_vid` washout and empty-stove layout drift, and `1.335x–1.381x` temporal contrast at only `1.07x–1.11x` wall-clock speed (`1836.1 MB` peak VRAM vs. `263.8 MB` `2x`). **Do not promote `Split-Text Online-Softmax Merge` or `1x Q_vid[mask]`**; keep `1x` query slicing on `Q_aud` (`130–810` rows) and route spatial portals / temporal flips through `2x` `MiniMaxH3EditGuider`.

---

## 1. Gate 2A: `Bus-Scoped V_r` A/B, `4`-Winner `192f` Showcases, `1x` Stream/Band Routing, & `345f` Sol Parity

Ledgers & contact sheets: [`gate2a_results.json`](../../data/exp_unpooled_sliders/wave2_2026_11_23/gate2a_results.json), `$H3_OUTPUT_DIR/Video/wave2_2026_11_23/gate2a/sheets/{sheet_bus_scoped_vr,sheet_4winners_192f,sheet_streams_and_345f}.png`. All runs at `1152x768`, `s = 1.35` (except `reverb_cathedral = 1.25` in the UV+Reverb stack).

### 1.1 `Bus-Scoped V_r` Multi-Bus Stacks (`192f`, `streams="both"`, `block_band="all"`)

| Arm | Scene | `hsv_sat` | `luma` | `mid_lap_var` | `rgb_mae` | `audio_dBFS` | `centroid_hz` | `YIN` `F0` Δ (global / paired) | `exec_s` | Visual Verdict (`f96`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `base_kitchen_192f` | `kitchen` | `0.3730` | `93.00` | `157.49` | `0.00` | `-10.99` | `2088.2` | `320.42 Hz` ref | — | Natural chef at gas burner flame |
| `stack_uv_reverb_shipped` | `kitchen` | `0.3917` | `95.58` | `43.44` (`h2=313.88`) | `71.904` | `-16.78` | `2317.1` | `+1.712 / +1.954 st` | `78.47 s` | **PASS:** Crisp UV purple/blue apron + flame plume |
| `stack_uv_reverb_pc8` | `kitchen` | `0.3419` | `98.14` | **`7.95`** | `72.424` | `-17.85` | `2230.2` | `+1.683 / +2.421 st` | `72.88 s` | **FAIL:** Blinding white-pink bloom obscures chef & stove |
| `stack_uv_reverb_pc16` | `kitchen` | `0.3506` | `99.39` | **`17.38`** | `73.449` | `-15.70` | `2269.0` | `+3.377 / +2.430 st` | `72.75 s` | **FAIL:** Severe white-magenta washout + `+3.38 st` pitch shift |
| `base_stairwell_192f` | `stairwell` | `0.3050` | `103.93` | `35.57` | `0.00` | `-25.25` | `2505.2` | `157.91 Hz` ref | `77.81 s` | Woman speaking in dim stairwell |
| `stack_chrome_zerog_shipped` | `stairwell` | `0.3815` | `72.36` | `38.99` | `64.228` | `-24.92` | `2345.1` | `+0.099 / +0.469 st` | `77.68 s` | **PASS:** Clean metallic chrome collar/lapels & cage lamp |
| `stack_chrome_zerog_pc8` | `stairwell` | `0.3750` | `72.60` | `48.86` | `63.734` | `-24.92` | `2265.2` | `+0.289 / +0.680 st` | `72.32 s` | Similar metallic lapels (`tail_sustain 0.196`) |
| `stack_chrome_zerog_pc16` | `stairwell` | `0.3828` | `73.47` | `53.12` | `63.590` | `-26.20` | `2429.4` | `-0.297 / +0.225 st` | `72.27 s` | Similar metallic lapels; unnecessary vs. `shipped` |

### 1.2 `4` Verified `t=1` Winners at `192f` + `1x` Stream/Band Routing + `345f` Sol Parity

| Arm (`length`, `streams`, `block_band`) | Scene | `hsv_sat` | `luma` | `mid_lap_var` | `rgb_mae` | `audio_dBFS` | `centroid_hz` | `YIN` `F0` Δ (global / paired) | `exec_s` | Visual Observation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `charcoal_kitchen_192f` (`both`, `all`) | `kitchen` | `0.1681` | `118.23` | `599.11` | `77.298` | `-24.09` | `2522.6` | `-0.512 / +0.205 st` | `77.97 s` | Full black-and-white charcoal/ink drawing w/ orange stove flame |
| `xray_kitchen_192f` (`both`, `all`) | `kitchen` | `0.3481` | `73.92` | `25.54` | `64.362` | `-17.85` | `2252.5` | `+0.511 / +0.535 st` | `72.64 s` | Luminous cyan X-ray teeth/jaw/throat cavity on chef |
| `chrome_kitchen_192f` (`both`, `all`) | `kitchen` | `0.3510` | `86.25` | `54.96` | `68.452` | `-11.00` | `2509.0` | `+1.170 / +1.097 st` | `72.68 s` | Golden-amber metallic kitchenware & twin pan flames |
| `uv_kitchen_192f` (`both`, `all`) | `kitchen` | `0.3869` | `81.91` | `25.13` | `68.263` | `-14.61` | `2420.0` | `+1.794 / +1.898 st` | `72.77 s` | Vivid cyan UV pan plasma plume & violet lighting |
| `charcoal_stairwell_192f` (`both`, `all`) | `stairwell` | `0.3459` | `90.48` | `67.64` | `39.270` | `-23.07` | `2184.4` | `-0.617 / +0.763 st` | `77.83 s` | **Partial:** charcoal hatching around right eye/cheek; cream light |
| `xray_stairwell_192f` (`both`, `all`) | `stairwell` | `0.3549` | `70.88` | `37.77` | `52.262` | `-24.23` | `2143.9` | `-0.188 / +0.799 st` | `72.59 s` | **Partial:** glowing skeletal mandible/teeth over lower-right face |
| `chrome_stairwell_192f` (`both`, `all`) | `stairwell` | `0.3693` | `90.47` | `44.52` | `49.670` | `-20.58` | `2134.3` | `+0.404 / +0.528 st` | `72.27 s` | Metallic liquid-chrome jacket lapels & high-gloss highlights |
| `uv_stairwell_192f` (`both`, `all`) | `stairwell` | `0.3560` | `93.63` | `41.73` | `42.775` | `-21.52` | `2039.0` | `-0.816 / +0.642 st` | `72.53 s` | Crisp cyan blacklight face rim & deep magenta studio bounce |
| `charcoal_192f_vidonly_1x` (`video_only`, `all`) | `stairwell` | `0.3253` | `89.32` | `81.67` | `37.062` | `-25.86` | `2188.4` | `-1.397 /`**`-0.039 st`** | **`76.55 s`** | `0.984x` base wall-clock parity with near-zero paired pitch shift |
| `reverb_192f_audonly_late30` (`audio_only`, `late_30_49`) | `stairwell` | `0.3115` | `104.02` | `41.01` | **`7.461`** | `-23.49` | **`2687.3`** | `-1.021 / +0.163 st` | **`75.25 s`** | First live GPU `1x audio_only + late_30_49`; pixel-matched to base |
| `base_stairwell_345f` (`345f`, unsteered) | `stairwell` | `0.3239` | `98.94` | `32.38` | `0.00` | `-28.20` | `2101.1` | `144.28 Hz` ref | `165.85 s` | Multi-shot stairwell (`f48` -> `f168` -> `f312` -> `f344`), clean Sol |
| `charcoal_stairwell_345f` (`345f`, `both`, `all`) | `stairwell` | `0.3934` | `91.79` | `139.29` | `48.857` | `-25.58` | `1800.0` | `+0.769 / +0.007 st` | `159.79 s` | Clean `345f` `MiniMaxH3Sol` completion; textured grit on jacket/window (`half2 MAE=60.183`) |

*(Baseline note: `uv_kitchen_192f` and `stack_uv_reverb_*` triggered `probe_yin_f0_drift` triage notes (`+1.79 / +1.71 / +1.68 / +3.38 st`) on female speech in `t2va_restaurant_kitchen.txt`, confirming expected cross-modal pitch drift when steering `streams="both"`.)*

---

## 2. Gate 2B: Track 2 `Path 1C ColQwen` `target="vision_tokens"` Dual-Lock Probe (`90f` `ref2va`)

Ledger & contact sheet: [`gate2b_path1c_results.json`](../../data/exp_unpooled_sliders/wave2_2026_11_23/gate2b_path1c_results.json), `$H3_OUTPUT_DIR/Video/wave2_2026_11_23/gate2b/sheet_gate2b_path1c.png`. Scene: `prompt_bank/ref2va_woman_blonde_street.txt` + `<Picture 1>` `woman_blonde.jpg` (`1152x768`, `90f`).

- **ColQwen Build Metrics (`noir_colqwen_p1c`, `N=6` aligned `512x512` pairs):** Built with `erasure.vr [8, 5376]`, `erasure.mu_a` norm `80.6324`, `erasure.mu_d` norm `16.5322`, `vector_norm = 16.5322` (`norm_before = 16.5577 -> 16.5322` under `--hold-words "face, eyes, nose, mouth, identity, portrait" --hold-alpha 0.65`), `diag = [0.8695, 0.8288, 0.8019, 0.7709, 0.7361, 0.7295, 0.7052, 0.7003]`, `pairwise_cosine = 0.50692`, `split_half_cosine = 0.75350`.
- **Unit & Schema Support Retained (No Bank Promotion):** Appended `"vision_tokens"` cleanly to the end of `STEER_TARGETS` ([`control_vector_nodes.py`](../../control_vector_nodes.py#L93)) and `INFERENCE_MASK_MODES` ([`h3_spans.py`](../../h3_spans.py#L81)) with mock DiT execution verification in [`bench/check_control_vectors.py`](../../bench/check_control_vectors.py#L8350-L8450); `noir_colqwen_p1c` stays in scratch only.

| Arm | Steering (`scale`, `target`) | `end_percent` / `clean_level` | `hsv_sat` / `mid_sat` | `luma` / `lap_var` | `rgb_mae` | `DINOv3` `color` / `gray` cos | `exec_s` / Peak VRAM (`breaks`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ref2va_base` | Unsteered | `1.00 / 0.999` | `0.2477 / 0.2549` | `102.19 / 74.40` | `0.000` | `0.39429 / 0.37122` | `37.17 s` / `175.9 MB` (`99`) |
| `ref2va_text_only_rel100` | `noir` (`1.5`, `scene_only`) | `1.00 / 0.999` | **`0.1919 / 0.1913`** | `99.41 / 76.03` | `45.486` | `0.43019 / 0.42522` | `36.12 s` / `312.8 MB` (`99`) |
| `ref2va_text_only_rel17` | `noir` (`1.5`, `scene_only`) | `0.17 / 0.90` | `0.2140 / 0.2123` | `110.20 / 60.85` | `45.040` | `0.46978 / 0.43832` | `29.19 s` / `312.8 MB` (`149`) |
| `ref2va_p1c_vis_only_rel17` | `noir_colqwen_p1c` (`1.35`, `vision_tokens`) | `0.17 / 0.90` | **`0.2432 / 0.2485`** (NULL) | `100.81 / 98.54` | `32.592` | `0.34639 / 0.32364` | `29.16 s` / `254.1 MB` (`149`) |
| `ref2va_p1c_dual_rel17` | `p1c` (`1.35`) + `noir` (`1.35`, `scene_only`) | `0.17 / 0.90` | `0.2351 / 0.2398` | `98.94 / 76.54` | `42.789` | `0.41040 / 0.38390` | `29.19 s` / `323.3 MB` (`149`) |
| `ref2va_p1c_dual_rel17_cl85` | `p1c` (`1.5`) + `noir` (`1.35`, `scene_only`) | `0.17 / 0.85` | `0.2225 / 0.2248` | `97.18 / 66.22` | `42.512` | `0.40572 / 0.39208` | `29.21 s` / `323.3 MB` (`149`) |

---

## 3. Gate 2C: `1x Q_vid[mask]` Bleed & Wall-Clock Probe vs. `2x MiniMaxH3Sol EditGuider` (`90f`)

Ledger & contact sheets: [`gate2c_qvid_results.json`](../../data/exp_unpooled_sliders/wave2_2026_11_23/gate2c_qvid_results.json), `$H3_OUTPUT_DIR/Video/wave2_2026_11_23/gate2c/{sheet_gate2c_temporal,sheet_gate2c_portal}.png`. Scene: `prompt_bank/t2va_restaurant_kitchen.txt`, `charcoal_sketch_animated` (`s = 1.35`), `1152x768 @ 90f` (`23,328` video tokens: `temporal_half = 12,096` / `51.9%`, `portal_20pct = 5,713` / `24.5%` at `R = 237.33 px`). Overall verdict: **`FAIL` (`temporal_pass = false`, `portal_pass = false`)**.

| Mask Family | Arm (`alpha`) | `exec_s` (Speed vs. `2x`) | Peak VRAM (`breaks`, `splices`) | `inside` / `outside` MAE vs. Base | Contrast Ratio | `inside` / `outside` `hsv_sat` | Visual Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| None | `base_90f` | `29.01 s` | `144.1 MB` (`99`, `0`) | — | — | — | Unsteered baseline |
| Full | `full_steered_90f` | `25.39 s` | `249.2 MB` (`99`, `0`) | `82.060 / 0.000` | `inf` | `0.1094 / —` | Full-clip charcoal reference |
| Temporal `51.9%` | `editguider_2x_temporal` (`1.0`) | `49.28 s` (`1.00x`) | `263.8 MB` (`199`, `0`) | `79.183 / 53.667` | **`1.475x`** | `0.2225 / 0.3127` | Crisp mid-clip switch to charcoal |
| Temporal `51.9%` | `qvid_1x_temporal_a10` (`1.0`) | `44.25 s` (`1.11x`) | `570.0 MB` (`99`, `300`) | `72.731 / 54.202` | `1.342x` | `0.2757 / 0.3087` | Incomplete temporal boundary |
| Temporal `51.9%` | `qvid_1x_temporal_a13` (`1.3`) | `45.69 s` (`1.08x`) | `1836.1 MB` (`99`, `300`) | `77.907 / 56.412` | `1.381x` | `0.2542 / 0.3066` | Best `1x` temporal ratio; still trails `2x` |
| Temporal `51.9%` | `qvid_1x_temporal_a15` (`1.5`) | `46.22 s` (`1.07x`) | `1836.1 MB` (`99`, `300`) | `75.366 / 56.457` | `1.335x` | `0.2193 / 0.3490` | Dark-lip smudge (`f44`) & blank counter (`f89`) |
| Portal `24.5%` | `editguider_2x_portal` (`1.0`) | `49.75 s` (`1.00x`) | `264.2 MB` (`199`, `0`) | `62.412 / 43.915` | **`1.421x`** | `0.3084 / 0.3635` | Localized charcoal inside moving circle (`00:02`–`00:03`) |
| Portal `24.5%` | `qvid_1x_portal_a10` (`1.0`) | `35.27 s` (`1.41x`) | `395.8 MB` (`99`, `300`) | `64.733 / 59.381` | **`1.090x`** | `0.3175 / 0.3640` | `K_vid, V_vid` bleed washes out portal boundary |
| Portal `24.5%` | `qvid_1x_portal_a13` (`1.3`) | `35.92 s` (`1.39x`) | `963.8 MB` (`99`, `300`) | `63.963 / 60.773` | **`1.052x`** | `0.3337 / 0.3623` | Scene layout drifts into empty stove |
| Portal `24.5%` | `qvid_1x_portal_a15` (`1.5`) | `36.17 s` (`1.38x`) | `963.8 MB` (`99`, `300`) | `64.884 / 61.957` | **`1.047x`** | `0.3206 / 0.3552` | No localized charcoal portal; bleed dominates |
