# 4-Sensor H3 Silicon Calibration Gate (`2026-11-26`)

Live GPU calibration of all four read-only internal H3 sensors ([`bench/probe_h3_sensors.py`](../probe_h3_sensors.py), CPU gate [`bench/check_probe_h3_sensors.py`](../check_probe_h3_sensors.py)) across both **`MiniMax-H3-Image` (`4 MP` `2368x1760`, 6-step `euler`)** and **joint AV `MiniMax-H3` (`1344x768`, `192f` / `8.0 s`, 6-step Pika H3 Flex `MiniMaxH3Sol` + `MiniMaxH3SpatialQueryRouter`)** on `HEAD = a31fb461`.

- **Visual contact sheets & JSON telemetry:** `$H3_OUTPUT_DIR/Video/beat2_v2_2026-11-26/sensor_gate/` (`h3_image_table_still.png`, `h3_image_table_step{0,1}_sheet.png`, `h3_image_table_report.json`, `ferry_binaural_midframe.png`, `ferry_binaural_step{0,1}_sheet.png`, `ferry_binaural_report.json` — all stills and contact sheets inspected natively via `read_file`).
- **Attachment order (verified in `0e674e2d`):** Always attach `MiniMaxH3SpatialQueryRouter -> attach_h3_sensor_probe` (**probe last**) so attention peeks run ahead of `SpatialQR`'s `iso`-block fast path (`0..24`) and `_SensorBlockPatch` reads post-`finish_block` `h_shadow_b` across all 50 blocks.

---

## 1. Go / No-Go Summary for Beat 3B Self-Steering

| Sensor | Signal Measured | Target A (`H3-Image` `4 MP`, `3.68 s` samp) | Target B (`192f` AV Ferry `binaural_split`, `124.9 s` samp) | Beat 3B Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Sensor 1: `Q_vid @ K_txt[noun]^T`** | Step-`0/1` zero-coordinate object localization | **`7/7` (`100%`) `peak_in_box=True`** on Step `0` & Step `1` (`B35`/`B40` mean `box_mass = 0.805–0.834`, **`3.54x–3.67x`** enrichment) | **`4/4` (`100%`) `peak_in_box=True`** on Steps `0, 1, 2` (`B40` mean `box_mass = 0.789–0.816`, **`2.11x–2.14x`** enrichment) | **STRONG GO (`B35 + B40`, Step `0/1`)** — zero-coordinate Self-Masked Routing works without `SAM 3.1`. Avoid `B45` (horizontal RoPE stripes). |
| **Sensor 2: `Q_vid @ K_aud[t]^T`** | Stereo `L/R` & speaker-face visual grounding | N/A (`audio_t = 0`, clean no-op) | **`B25` stereo split `lr_x_sep = +0.140`** (`L_x=0.419`, `R_x=0.558`); **`B40` `aud:L` / `aud:R` / `aud:stereo` locks razor-sharp hotspots directly onto the left/right speakers' faces** | **STRONG GO (`B25` L/R split + `B40` zero-training speaker spotlight)** |
| **Sensor 3: Pure-latent `audio_x0`** | `< 1 ms` Step-`0/1/2` onset peaks & `L/R` correlation (`zero` Audio VAE decode) | N/A (`audio_t = 0`, clean no-op) | `S0`: `lr_corr = 0.2705`, `14` peaks; top flux peaks (`1.18 s`, `2.85–3.68 s`, `5.03–5.95 s`, `7.30–7.48 s`, `z = 2.8–4.5`) align within `~0.2 s` of all `4` spoken lines | **GO** — fast zero-decode Acoustic Thermostat & turn-timing triage |
| **Sensor 4: Block `J-Lens` (`h_txt_a`, `h_shadow_b`)** | Per-block `decode_vector` semantic readout (`B00..B48`) | `B00` norm `48.4K` -> `B48` `662.0K` | **`B00..B10` accurate** (`B-A+ = ['rain', 'mist', 'glowing', 'snow', 'smoke']`); **past `B20`, norms explode (`29K -> 725K`)** and Block-0 `J_wiener` saturates on generic lighting/adverb axes | **HONEST NO-GO past `B10`** — keep `J-Lens` strictly at `B00..B10` input/early diagnostics; do not build a deep-block drift controller on uncalibrated Block-0 `J_wiener`. |

---

## 2. Sensor 1: Sharp Phase Transition at Blocks `30..40` (`Q_vid @ K_txt[noun]^T`)

### 2a. Target A — `MiniMax-H3-Image` (`4 MP`, `vendor_guides/h3_image/examples/revised_prompt.t2i.txt`, `seed = 42`, `txt_len = 326`)

Seven quadrant-pinned nouns (`bread` center, `carrots` left, `cheeses` right, `olive oil` upper-right, `grapes` upper-left, `knife` lower-left, `potatoes` lower-center; uniform-random box mass `~0.228`):

| Step / Block | `peak_in_box` Hits | Mean `box_mass` | Mean Enrichment | Mean Norm Entropy | Visual Read (`step0_sheet.png` / `step1_sheet.png`) |
| :--- | :-: | :-: | :-: | :-: | :--- |
| `S0 B10` | `0/7` | `0.257` | `1.10x` | `0.988` | Near-uniform diffuse activation |
| `S0 B15` | `4/7` | `0.326` | `1.41x` | `0.978` | Coarse broad quadrants beginning |
| `S0 B20` | `3/7` | `0.334` | `1.44x` | `0.960` | Diffuse background layout |
| `S0 B25` | `4/7` | `0.373` | `1.63x` | `0.905` | Coarse cluster formation |
| **`S0 B30`** | **`7/7`** | **`0.706`** | **`3.10x`** | **`0.711`** | **Sharp phase transition** — all 7 nouns snap into target quadrants |
| **`S0 B35`** | `6/7` | **`0.805`** | **`3.54x`** | `0.716` | **Pixel-sharp object silhouettes** (`bread` basket, `cheeses` board, `grapes`, `knife`, `olive oil`) |
| **`S0 B40`** | **`7/7`** | **`0.805`** | **`3.54x`** | `0.739` | **Peak Step-0 mask quality** (`S1 B40` reaches `0.834` / `3.67x`) |
| `S0 B45` | `7/7` | `0.762` | `3.36x` | `0.625` | **Degrades into periodic horizontal RoPE stripes** across all 7 nouns — exclude from mask extraction |

Per-noun best Step-`0` / Step-`1` locks:

| Noun | Matched BPE Indices | Best `S0` Block | `S0 box_mass` (`peak_xy`) | `S1 box_mass` (`peak_xy`) |
| :--- | :--- | :-: | :-: | :-: |
| `bread` (center) | `[52, 163, 305]` | `B30` | `0.684` (`0.49, 0.55`) | `0.678` (`0.66, 0.55`) |
| `carrots` (left) | `[90]` | `B40` | `0.819` (`0.24, 0.61`) | `0.920` (`0.22, 0.45`) |
| `cheeses` (right) | `[126]` | `B35` | `0.822` (`0.76, 0.63`) | `0.726` (`0.98, 0.61`) |
| `grapes` (upper-left) | `[246]` | `B35` | `0.865` (`0.07, 0.28`) | `0.910` (`0.07, 0.39`) |
| `knife` (lower-left) | `[231]` | `B35` | `0.902` (`0.26, 0.77`) | `0.918` (`0.21, 0.83`) |
| `olive oil` (upper-right) | `[203, 204]` | `B40` | `0.808` (`0.79, 0.08`) | `0.882` (`0.87, 0.23`) |
| `potatoes` (lower-center) | `[179]` | `B40` | `0.881` (`0.56, 0.92`) | `0.911` (`0.43, 0.75`) |

### 2b. Target B — Joint AV `b1_192f_stairwell_ferry_binaural` (`1344x768`, `192f`, `seed = 730451892`, `SpatialQR early_0_24 + binaural_split`)

Identical phase transition across `57` video latent steps (`[57, 24, 42]`):

| Step / Block | `peak_in_box` Hits | Mean `box_mass` | Mean Enrichment | Mean Norm Entropy |
| :--- | :-: | :-: | :-: | :-: |
| `S0 B10` | `4/4` | `0.451` | `1.09x` | `0.988` |
| `S0 B20` | `3/4` | `0.506` | `1.25x` | `0.942` |
| `S0 B25` | `3/4` | `0.527` | `1.31x` | `0.906` |
| `S0 B30` | `4/4` | `0.641` | `1.68x` | `0.763` |
| `S0 B35` | `4/4` | `0.707` | `1.85x` | `0.704` |
| **`S0 B40`** | **`4/4`** | **`0.789`** | **`2.11x`** | **`0.688`** |
| **`S1 B40`** | **`4/4`** | **`0.816`** | **`2.14x`** | **`0.679`** |

At `B40`, `bulb` (`[68, 193]`) hits `box_mass = 0.847` (`3.56x` enrichment, `peak_xy = (0.42, 0.15)`), `woman` (`[30, 72]`) climbs `0.757 -> 0.851 -> 0.891` across `S0 -> S2`, `man` (`[43, 95]`) hits `0.718 -> 0.785`, and `stairwell` (`[58, 59, 173, 174]`) isolates the left concrete walls (`0.833 -> 0.869`) while rejecting the night harbor water on the right.

---

## 3. Sensor 2 (`Q_vid @ K_aud[t]^T`) & Sensor 3 (`audio_x0` Transients)

- **Sensor 2 (`ferry_binaural_step{0,1}_sheet.png`):**
  - **`B25`:** Peak stereo horizontal separation — `S0 L_xy = (0.457, 0.521)` vs `R_xy = (0.597, 0.515)` (`lr_x_sep = +0.140`), `S1 L_xy = (0.419, 0.514)` vs `R_xy = (0.558, 0.519)` (`lr_x_sep = +0.139`).
  - **`B35` / `B40`:** Collapses background clutter into **two tight facial hotspots directly over the speaking characters' heads/mouths**: `B40 aud:L` isolates the woman on the left (`x ≈ 0.32`), `B40 aud:R` isolates the man on the right (`x ≈ 0.72`), and `B40 aud:stereo` lights both speakers cleanly with zero face detector or `SAM 3.1`.
- **Sensor 3 (`audio_x0`, `sampler_post_cfg_function`, `< 1 ms`, zero Audio VAE decode):**
  - `S0 (sigma = 1.000)`: `rms = 2.101`, `L/R_dB = +0.161`, `lr_latent_corr = 0.2705`, `14` peaks; top flux peaks `1.175 s (f28, z = 4.36)`, `2.85–3.675 s (f68–88, z = 3.29–3.33)`, `5.025–5.95 s (f121–143, z = 2.17–2.81)`, `7.30–7.475 s (f175–179, z = 2.98–3.05)` match all four dialogue lines observed in `b1_192f_stairwell_ferry_binaural_00001-audio.mp4` (`00:01`, `00:03`, `00:05`, `00:07`).
  - `S1 (sigma = 0.973)`: `rms = 2.427`, `L/R_dB = -0.291`, `lr_latent_corr = 0.3060`, `16` peaks (top `3.025 s / f73 z = 4.46`, `0.625 s / f15 z = 4.36`, `5.85 s / f140 z = 3.75`, `6.80 s / f163 z = 3.45`).
  - `S2 (sigma = 0.923)`: `rms = 2.284`, `L/R_dB = -0.178`, `lr_latent_corr = 0.3207`, `18` peaks (top `1.175 s / f28 z = 4.34`, `7.00 s / f168 z = 4.19`, `5.875 s / f141 z = 4.11`).

---

## 4. Sensor 4 (`J-Lens` Across Blocks `0..48` — Honest Calibration Limit)

On `ferry_binaural`:
- **`B00` (`normA = 29.1K`, `normB = 30.4K`) & `B10` (`55.8K` / `57.9K`):** `J-Lens(mean(h_shadow_b) - mean(h_txt_a))` decodes the exact semantic difference between the concrete stairwell (`A`) and foggy night ferry (`B`): `B-A+ = ['rain', 'mist', 'glowing', 'snow', 'smoke']`.
- **`B20..B48` (`normA` grows `97.6K -> 115K -> 306K -> 629K` on `S0`, up to `725K` on `S1`):** Because `J_wiener_fl2va` (`steering/xray.py`) was fitted on Block-0 post-refiner space, projecting `20x`-amplified deep-block residual vectors onto Block-0 `J_wiener` saturates onto generic lighting/adverb axes (`['bright', 'soft', 'light', 'likely', 'effectively']`). **Action:** Keep `J-Lens` at `B00..B10` only; drive Beat 3B self-steering exclusively from Sensors 1 (`B35 + B40`), 2 (`B25` / `B40`), 3 (`audio_x0`), and `ModelPreviewOverrideKJ` (`kj_wrapper.frames[-1]`).
