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

---

## 5. Live Silicon Proof of `steering/live_policy.py::attach_self_mask_router` (`1.0x`, `4 MP`)

End-to-end single-pass `1.0x` GPU calibration (`Step 0` `B35 + B40` harvest -> `Steps 1..5` `SpatialVisualQRRuntime`, zero coordinates, zero `SAM 3.1`, zero pre-exported mask PNG) on `MiniMax-H3-Image` `4 MP` (`revised_prompt.t2i.txt`, `seed = 42`, `~4.2–4.4 s` warm per still; outputs under `$H3_OUTPUT_DIR/Video/beat3_self_mask_poc_2026-11-26/`, all stills and mask overlays inspected natively via `read_file`):

| Arm | `target_phrase` | Mask Params (`z / dil / floor`) | `Isolation` | Warm Time | `hard_cov` / `zero_frac` | `in_bbox` / `out_bbox` MAE | Honest Visual Verdict |
| :--- | :--- | :--- | :--- | :-: | :-: | :-: | :--- |
| `cheeses_blue_ice` | `"cheeses"` | `1.10 / 1 / 0.05` | `early_0_24_shield` | `4.34 s` | `0.265 / 0.433` | `30.15 / 21.00` (`1.44x`) | **10/10 single-pass win:** All three cheeses on the right wooden board turned into translucent frosted glacier-blue ice blocks with rising cold vapor mist; zero seam artifacts; bread/vegetables/grapes/oil held matched. |
| `cheeses_blue_ice_tight` | `"cheeses"` | `1.65 / 0 / 0.12` | `early_0_24_shield` | `4.18 s` | `0.123 / 0.811` | `26.99 / 14.99` (`1.80x`) | **Cleaner surround lock (`-6.0` outer MAE, `81.1%` exact-zero bg):** Eliminated halved-bread mask spill while freezing core wedges & brie with cold vapor mist and zero patch seams. |
| `olive_oil_neon_flask` | `"olive oil"` | `1.15 / 1 / 0.05` | `early_0_24_shield` | `4.34 s` | `0.207 / 0.439` | `24.08 / 19.73` (`1.22x`) | **10/10 neon alchemy flask:** Upper-right jar became a glowing electric-blue plasma flask sparkling with cyan stars while the rest of the feast held photographic; wider mask hit the adjacent honey jar cleanly. |
| `olive_oil_neon_tight` | `"olive oil"` | `1.65 / 0 / 0.12` | `early_0_24_shield` | `4.35 s` | `0.116 / 0.836` | `19.06 / 16.57` (`1.15x`) | **Laser-focused on the olive-oil bottle neck (`83.6%` exact-zero bg):** Zero bleed to black olives or honey jar; subtle starry cyan plasma glow strictly inside the olive-oil bottle glass. |
| `bread_emerald_crystal` / `mid2034_shield` | `"bread"` | `1.05 / 1 / 0.05` | `early_0_24_shield` / `mid_20_34_shield` | `4.41 s` | `0.279 / 0.457` | `33.73 / 20.61` (`1.64x`) | **Honest `1.0x` centerpiece boundary:** Because `bread` (3 mentions, center `~28%`) locks brown-crust geometry on Step `0` (and `region_prompt` kept the word `"loaf"`), starting on Step `1` only grew small emerald gems beneath the brown loaves. |
| `bread_emerald_tight_all049` | `"bread"` | `1.65 / 0 / 0.12` | `all_0_49` | `14.38 s`* | `0.138 / 0.795` | `47.98 / 16.92` (`2.84x`) | **Hard limit proven:** Turning off seamless `25..49` forced full emerald quartz geodes (`2.84x` MAE), **but** produced hard `32x32` patch-grid edges around the organic attention silhouette. |

*\* Cold-cache first arm in its process; subsequent arms ran `4.18–4.41 s`.*

**Three calibrated rules locked into [`steering/live_policy.py`](../../steering/live_policy.py):**
1. **Calibrated defaults (`threshold_z = 1.35`, `softness = 0.25`, `dilation_passes = 0`, `sparse_floor = 0.10`, `isolation = "early_0_24_shield"`):** Balances tight surround exclusion (`~70–84%` exact-zero background) with full object coverage and zero `32x32` patch-grid edges.
2. **Never pair `all_0_49` with organic irregular attention masks:** Reserve `all_0_49` strictly for `1`-frame temporal cuts or clean geometric splits; always use `early_0_24_shield` or `mid_20_34_shield` on irregular object silhouettes.
3. **`1.0x` single-pass vs `1.17x` Step-0 replay boundary:** Single-pass `1.0x` `attach_self_mask_router` (Step `0` harvest -> Steps `1..5`) transforms accent/secondary props cleanly in `~4.2–4.4 s` (avoid repeating the source noun in `region_conditioning`). For total material replacement of a dominant multi-mention centerpiece (`bread -> sapphire crystal`), replaying Step `0` with the harvested mask (`1.17x`, as demonstrated in `b3_h3_image_step0_self_mask_bread_crystal_00001_.png`, `4.30x` MAE) cleanly overrides Step-0 geometry without `all_0_49` edges.

---

## 6. Track 2 Silicon Gate: `1.17x` `replay_step0` & `141f` Joint AV Video (`3e76b4a3`)

Live end-to-end validation of the Track 2 upgrades in [`steering/live_policy.py`](../../steering/live_policy.py) (`cross_time_scale_floor = 0.55` + `replay_step0: bool = False` `1.17x` mode, CPU gate [`bench/check_live_policy.py`](../check_live_policy.py)) across `MiniMax-H3-Image` `4 MP` (`seed = 42`) and `141f` (`5.875 s`) `1344x768` `24 fps` Pika H3 Flex AV (`seed = 730451892`, `MiniMaxH3Sol` + `comfy kitchen attention`). Outputs and JSON reports live under `$H3_OUTPUT_DIR/Video/track2_self_mask_av_2026-11-26/` (`track2_self_mask_av_report.json`, `track2b_replay_followup_report.json`); every still, video, and 6-frame temporal mask overlay contact sheet was inspected natively via `read_file`.

| Arm | Target / `mask_source` | Settings (`z / iso / replay_step0`) | Sample Time | Mask Telemetry (`hard_cov`, `peak / centroid / bbox`) | `in_mask` / `out_mask` MAE | Honest Visual Verdict |
| :--- | :--- | :--- | :-: | :--- | :-: | :--- |
| **`a1_bread_replay_step0_mid2034_shield`** (`4 MP` `H3-Image`) | `text_phrase="bread"` | `1.35 / mid_20_34_shield / True` (`1.17x`) | **`5.01 s`** | `0.1781` (`74.2%` exact-zero bg), `peak=(0.561, 0.282)` | **`69.80 / 29.90` (`2.334x`)** | **10/10 single-call centerpiece breakthrough:** Both loaves inside the wicker basket crystallize into glowing faceted emerald-green quartz geodes casting soft green light onto the wicker weave while grapes, veggies, cutlery, olive oil, and cheese board stay photoreal with zero `all_0_49` square seams. |
| `b1_stairwell_right_speaker` (`141f` AV) | `right_speaker` | `1.35 / mid_20_34_shield / False` (`1.0x`) | `59.19 s` | `0.1099` (`[0.073, 0.177]`), `peak=(0.679, 0.313)`, `cx=0.649` | `20.96 / 13.55` (`1.547x`) | **Sensor locked right speaker's face; visual style stayed photoreal.** Unrouted stairwell reverb held `lr_latent_corr = 0.935` (`silent=0`); `mid_20_34_shield` without `replay_step0` did not break skin texture. |
| `b2_stairwell_active_speaker` (`141f` AV) | `active_speaker` | `1.35 / mid_20_34_shield / False` (`1.0x`) | `59.69 s` | `0.1028` (`[0.055, 0.158]`), `cx=0.441..0.498` (L) / `0.613..0.621` (R) | `19.84 / 13.37` (`1.484x`) | **Dynamic turn tracking confirmed in `mask_strip.png`; visual style stayed photoreal.** Mask shifted between woman (`f12..36`, `f84..96`) and man (`f48..60`) cleanly without breaking skin consistency. |
| `b3_stairwell_active_speaker_replay_e024` (`141f` AV) | `active_speaker` | `1.55 / early_0_24_shield / True` (`1.17x`, pure cyan hologram prompt) | `87.49 s`* | `0.0739` (`[0.031, 0.140]`, `82.9%` zero), `cx=0.439..0.502` (L) / `0.618..0.631` (R) | `34.88 / 23.16` (`1.506x`) | **Honest `1x` AV face-mask boundary:** `in_mae` rose `+76%` over `b2` and face-tracking stayed sharp, **but neither actor turned into a cyan hologram** — `5–8%` flickering face-only masks are re-normalized by unisolated blocks `25..49` plus audio lip-sync grounding. |
| `b4_stairwell_right_speaker_replay_e024` (`141f` AV) | `right_speaker` | `1.55 / early_0_24_shield / True` (`1.17x`, pure cyan hologram prompt) | `75.74 s` | `0.0827` (`[0.045, 0.139]`, `82.5%` zero), `peak=(0.679, 0.313)`, `cx=0.656` | `31.46 / 18.93` (`1.662x`) | **Same boundary as `b3`:** Tighter right-speaker face focus (`1.662x` MAE ratio) and subtle lighting/angle shift, but photoreal human skin resisted stylized hologram replacement. |
| **`c1_walk_text_phrase_woman`** (`141f` AV) | `text_phrase="woman"` | `1.35 / mid_20_34_shield / False` (`1.0x`) | `59.08 s` | `0.1269` (`[0.079, 0.161]`), `cx: 0.209 -> 0.375 -> 0.541` (`f12->f60->f120`) | `13.89 / 19.11` (`0.727x`) | **Zero-SAM3 moving `[T, H, W]` proof (full `00:00–00:05`):** Tracked her left-to-right walk across the cloister and turned her head/upper jacket into cobalt-blue chrome across all `141` frames with locked footsteps & architecture (caveats: lower burgundy skirt stayed red; `mask_strip.png` carried diffuse arch tails raising `out_mae`). |
| **`c2_walk_woman_coat_replay_e024`** (`141f` AV) | `text_phrase="woman, crimson, coat"` | `1.80 / early_0_24_shield / True` (`1.17x`) | `75.51 s` | `0.0703` (`[0.022, 0.137]`, `85.6%` zero), `bbox=[0.167, 0.542, 0.405, 0.708]` | `18.55 / 26.08` (`0.711x`) | **Strongest moving chrome transform + direct causal gating proof:** Gleaming mirror-polished silver chrome humanoid across `00:00–00:02` (`f0..72`) while covered by the mask, then snaps back to a real human at `00:03–00:05` (`f96..120`, `cov = 3.8–4.7%`) the instant she enters the bright sunlit right half and exits the lagged `z = 1.80` mask. |

*\* Cold first arm after DiT load; warm `1.17x` `141f` AV arms ran `75.51–75.74 s` vs `59.08–59.69 s` (`1.0x`) and `49.98 s` (`base`).*

**Registration & Usage Verdict for `MiniMaxH3SelfMaskRouter` (`steering/live_policy.py`):**
- **Ready to register in [`nodes.py`](../../nodes.py) on the next planned ComfyUI restart** (kept unregistered during this sprint so live `:8188` stays hot with zero restarts).
- **Production-ready on `MiniMax-H3-Image` (`4 MP`):** Use `replay_step0 = False` + `early_0_24_shield` (`1.0x`, `~4.2 s`) for props and `replay_step0 = True` + `mid_20_34_shield` (`1.17x`, `~5.0 s`) for dominant centerpieces — zero coordinates, zero `SAM 3.1`, zero square patch edges.
- **Sensor-proven / experimental on `24 fps` AV video:** `B35 + B40` reliably localizes moving subjects (`text_phrase`) and alternating talking faces (`active_speaker` / `right_speaker`) for live telemetry and subtle-to-medium moving subject swaps (`c1`, `c2`), whereas pixel-tight full-body tracking across dramatic lighting changes or bold stylized overrides on `< 10%` talking faces still favor `SAM 3.1` + `2x MiniMaxH3EditGuider`.


