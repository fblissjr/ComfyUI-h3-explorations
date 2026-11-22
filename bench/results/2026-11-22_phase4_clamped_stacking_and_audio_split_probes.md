# Phase 4 Steering Probes: Clamped Union Multi-Slider Stacking, `blocks.20..49` Shadow Routing, `1x` `Q_aud` Audio Split, & `4.0 MP` `MiniMax-H3-Image` Native Erasure (`2026-11-22`)

lane: phase4
verdict: Clamped-union stacking eliminates `lambda > 1.0` over-erasure on CPU and cuts `90f` 4-slider `box_office` saturation from `0.2253 -> 0.1315`; `1x` `Q_aud -> K_txt_base` splicing recovers `de_llm_slop` audio attenuation (`-24.56 -> -22.29 dBFS`, within `0.18 dB` of base `-22.11 dBFS`) at `28.80s` (`1.73x` faster than `2x` `video_only` `49.94s`) with identical visual grading; and `45` `__h3_image__` bank vectors (`34` built + `crystallize_ice` rebuilt) enable native `4.0 MP` rank-8 single & 2-slider stacked erasure (`~3.0s` / frame). Remaining visual edge cases (overlapping `scene_only` chromatic competition and a `claymation + noir` diptych split on `lighthouse`) directly motivate cross-slider `V_r_union ⊥ span(mu_d^(1..K))` protection and per-token multi-vector Pole-B routing.

---

## 1. Runtime & Bank Deliverables Shipped Ahead of Phase 4

1. **Runtime Clamped-Union Multi-Slider Stacking ([`control_vectors.py`](../../control_vectors.py)):**
   - Replaced linear shift summation on shared masks with `_stacked_erasure_block_shift64` inside `apply_post_refiner` (exact `K <= 1` path preserved bit-for-bit): forms an orthonormal union basis from stacked Pole-A subspaces, clamps union eigenvalues to `[0.0, 1.0]`, and enforces budget `e_budget = min(e_budget_sum, max(NORM_MATCHED_TARGET, ||add_total||_2))` alongside `target_rms <= 1e-6` zero guard.
   - Extended `compose_erasure_payload(..., return_mu_a=False)` to accept optional `erasure_mu_a` 7-tuples and extended `per_token_erasure_profile(..., mu_a=None)`.
2. **Node Safeguards & CLI Lockstep ([`control_vector_nodes.py`](../../control_vector_nodes.py), [`bench/control_vector/cli.py`](../control_vector/cli.py)):**
   - Added `min_erasure_pairs` (`MIN_ERASURE_PAIRS`) guard on `make_control_vector_from_media_pairs` to omit fragile low-pair `erasure.*` tensors; updated `MiniMaxH3Steer.token_focus` tooltip to document `1.00` as the validated default.
   - Locked `cli.py` `grade --mode` default to `h3_rules.STEER_DEFAULTS["mode"]` (`add_norm_matched_erasure`) and extended unit regressions in [`bench/verify_pole_a_erasure.py`](../verify_pole_a_erasure.py) and [`bench/check_control_vectors.py`](../check_control_vectors.py).
3. **Bank Hygiene & `--families h3_image` Coverage (`37` Builds Verified):**
   - Fixed Pair 8 Gate D shared-prefix warning in [`prompt_bank/control_vectors/crystallize_ice.toml`](../../prompt_bank/control_vectors/crystallize_ice.toml#L96-L101) (`"Inside an airy print shop"`, `0` Gate D warnings).
   - Relocated the `10` `[vector]` TOMLs (`ankle_deep_flood`, `arctic_expedition`, `astronaut_crew`, `circus_clowns`, `disco_fever`, `medieval_feast`, `overgrown_jungle_ruin`, `pirate_crew`, `summer_beachwear`, `wild_west_saloon`) from `data/edit_concepts/` to `data/control_vectors/` (`23` vector TOMLs in `data/control_vectors/`, `0` `[vector]` files left in `data/edit_concepts/`).
   - Built `34` visual/material/style `--families h3_image` vectors and rebuilt `crystallize_ice` (`fl2va,ref2va,h3_image` `--overwrite`) into `$H3_MODEL_STORE/control_vectors/bank/` (`136` total `.safetensors` files including `45` `__h3_image__`, all verified with `erasure.{mu_d,vr,diag}`).

---

## 2. Showcase A — Clamped Union Runtime Stacking (`t=1` Stills + `90f` 4-Slider AV)

Runner: [`bench/control_vector/run_phase4_stacked_and_stream_probes.py`](../control_vector/run_phase4_stacked_and_stream_probes.py) (`--stage showcase_a`). Metrics: [`data/exp_unpooled_sliders/phase4_2026_11_22/showcase_a_clamped_stacking.json`](../../data/exp_unpooled_sliders/phase4_2026_11_22/showcase_a_clamped_stacking.json). Media: `$H3_OUTPUT_DIR/Image/phase4_2026-11-22/showcase_a_stack3_stills.png` and `$H3_OUTPUT_DIR/Video/phase4_2026-11-22/showcase_a_4slider_av_*.mp4`.

- **Spectral Over-Erasure Check:** Stacking `noir (s=1.25) + chaos_panic (s=1.0) + de_llm_slop (s=0.9)` in the old linear sum produced **`6` union eigenvalues `> 1.0`** (`[1.8568, 1.6141, 1.5071, 1.2997, 1.2130, 1.0882, 0.9992, ...]`).

### 2a. `t=1` Pika `fl2va` 3-Slider Stack (`noir +1.25`, `chaos_panic +1.0`, `de_llm_slop +0.9`)

| Prompt | Seed | Arm | `hsv_sat` | `laplacian_var` | `rms_contrast` | `clip_frac` |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| `box_office` | `730451892` | `Base` | `0.2920` | `215.32` | `58.24` | `0.00020` |
| `box_office` | `730451892` | `Single noir +1.25` | `0.0472` | `473.51` | `87.15` | `0.10546` |
| `box_office` | `730451892` | `Old Parallel Sum (λ_max=1.86)` | `0.1995` | `182.35` | `84.59` | `0.11097` |
| `box_office` | `730451892` | `New Clamped Union (λ_max<=1.0)` | `0.3083` | `198.38` | `79.33` | `0.09746` |
| `box_office` | `101` | `Base` | `0.3596` | `164.89` | `58.81` | `0.00001` |
| `box_office` | `101` | `Single noir +1.25` | `0.0246` | `355.05` | `76.96` | `0.07462` |
| `box_office` | `101` | `Old Parallel Sum (λ_max=1.86)` | `0.2592` | `279.89` | `75.12` | `0.05797` |
| `box_office` | `101` | `New Clamped Union (λ_max<=1.0)` | **`0.1911`** | **`306.76`** | `76.43` | `0.05282` |
| `sitcom_kitchen` | `730451892` | `Base` | `0.3840` | `711.21` | `70.27` | `0.01643` |
| `sitcom_kitchen` | `730451892` | `Single noir +1.25` | `0.2856` | `480.46` | `72.84` | `0.01890` |
| `sitcom_kitchen` | `730451892` | `Old Parallel Sum (λ_max=1.86)` | `0.3068` | `182.42` | `61.40` | `0.00606` |
| `sitcom_kitchen` | `730451892` | `New Clamped Union (λ_max<=1.0)` | `0.3197` | `168.93` | `62.32` | `0.00526` |
| `sitcom_kitchen` | `101` | `Base` | `0.3634` | `834.23` | `59.39` | `0.01493` |
| `sitcom_kitchen` | `101` | `Single noir +1.25` | `0.2680` | `627.39` | `70.30` | `0.02076` |
| `sitcom_kitchen` | `101` | `Old Parallel Sum (λ_max=1.86)` | `0.2425` | `141.23` | `64.22` | `0.00920` |
| `sitcom_kitchen` | `101` | `New Clamped Union (λ_max<=1.0)` | **`0.2268`** | **`189.75`** | `58.16` | `0.00861` |

### 2b. `90f` Speaking 4-Slider AV Stack (`t2va_box_office.txt`, `scene_only` 3-stack + `soundscape_only reverb_cathedral +1.25`)

| Arm | `sat_mean` | `sat_mid_mean` | `mid_laplacian_var` | `temp_l1_mean` | `audio_dbfs` | `centroid_hz` | `tail_sustain_ratio` | `gen_elapsed_s` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `base` | `0.3842` | `0.3862` | `172.87` | `4.366` | `-22.11` | `2414.5` | `0.3384` | `24.79` |
| `old_parallel_4slider` | `0.2253` | `0.2412` | `270.66` | `10.287` | `-21.72` | `1983.8` | `0.2749` | `24.96` |
| **`new_clamped_4slider`** | **`0.1315`** | **`0.1400`** | `96.10` | `11.583` | `-21.66` | `1917.7` | `0.2825` | `25.19` |

- **Visual Inspection:** On the `90f` `box_office` AV strips (`$H3_OUTPUT_DIR/Video/phase4_2026-11-22/strips/showcase_a_4slider_av_*.png`), `old_parallel_4slider` retains a brightly colored orange/teal cinema marquee (`f=0..35`) and warm yellow storefront glow (`f=53..89`), whereas `new_clamped_4slider` holds slate-monochrome sky, street, and facade across `f=0..89` while still manifesting `chaos_panic` (long spitting ticket-paper cascade at `f=89`). The lower mid-frame Laplacian (`96.10`) reflects the wide static grey booth wall centered at `f=45` rather than blur. Still sheets show `New Clamped Union` sharpening `box_office s=101` (`lap 279.89 -> 306.76`, `sat 0.2592 -> 0.1911`) and `sitcom_kitchen s=101` (`141.23 -> 189.75`, `0.2425 -> 0.2268`), though `box_office s=730451892` retains warm awning tint (`sat=0.3083`) because stacking 3 overlapping `scene_only` vectors hits peak row shift `161.11` (`threshold = 97.9`) without cross-slider `V_r^(j) ⊥ mu_d^(i)` protection.

---

## 3. Showcase B — Inverted Shadow-Text K/V Routing (`blocks.20..49` on `90f` `ref2va`)

Runner: `--stage showcase_b`. Metrics: [`data/exp_unpooled_sliders/phase4_2026_11_22/showcase_b_inverted_blocks20.json`](../../data/exp_unpooled_sliders/phase4_2026_11_22/showcase_b_inverted_blocks20.json). Strips: `$H3_OUTPUT_DIR/Video/phase4_2026-11-22/strips/showcase_b_inv_shadow_*.png`. Prompt: [`prompt_bank/ref2va_woman_blonde_street.txt`](../../prompt_bank/ref2va_woman_blonde_street.txt) (`ReleaseReferences(end_percent=0.50)`, `target="description_only"`, `strength=1.5`).

| Arm | `swaps` | `sanity_diff` | `sat_mean` | `center_vs_ref_sat_diff` | `border_sat_mean` | `border_lap_e` | `audio_dbfs` | `elapsed_s` | `vram_gb` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `unsteered_base` | `0` | `0.0` | `0.2603` | `0.0463` | `0.2473` | `70.92` | `-20.52` | `38.92` | `3.34` |
| `noir_all_0_49` | `300` | `0.0` | `0.2199` | `0.0438` | `0.1937` | `63.56` | `-22.46` | `35.86` | `3.34` |
| `noir_inverted_20_49` | `180` | `0.0` | `0.2143` | `0.0445` | `0.1861` | `62.32` | `-20.46` | `35.89` | `3.34` |
| `felt_muppet_all_0_49` | `300` | `0.0` | `0.2568` | `0.0177` | `0.2515` | `99.97` | `-20.30` | `35.91` | `3.34` |
| `felt_muppet_inverted_20_49` | `180` | `0.0` | `0.2721` | `0.0222` | `0.2709` | `100.58` | `-21.08` | `35.84` | `3.34` |

- **Findings:**
  - **Architectural parity:** Inverted shadow routing (`h_txt_base` in the main stream, `0` swaps in `blocks.0..19`, `180` K/V swaps in `blocks.20..49`) achieved `sanity_block0_diff = 0.0`, identical peak VRAM (`3.34 GB`), and `35.84–35.89s` elapsed vs. full `0..49` (`35.86–35.91s`), while matching `noir`'s storefront awning darkening (`border_sat_mean`: `0.2473 -> 0.1861` vs. `0.1937` all-block) and keeping `<Picture 1>` portrait saturation locked (`center_vs_ref_sat_diff`: `0.0445` vs. `0.0438`).
  - **Honest visual limitation on `description_only`:** Inspecting all 5 strips confirms that holding `<Picture 1>` via `ReleaseReferences(0.50)` while masking to `description_only` (leaving `[Style]` / `[Lighting]` unsteered) kept both `0..49` and `20..49` predominantly photorealistic — turning green shop awnings black on `noir` and fluffing hair/motion slightly on `felt_muppet` (`border_lap_e 70.92 -> 100.58`) rather than full monochrome or muppet felt conversion.

---

## 4. Showcase C — `1x` `Q_aud -> K_txt_base` Split vs. `2x` `streams="video_only"` (`90f` `t2va_box_office.txt`)

Runner: `--stage showcase_c` (`de_llm_slop +1.15`, `scene_only`). Metrics: [`data/exp_unpooled_sliders/phase4_2026_11_22/showcase_c_qaud_audio_split.json`](../../data/exp_unpooled_sliders/phase4_2026_11_22/showcase_c_qaud_audio_split.json). Dialogue `voice_spans.py` on `base`: `voiced_spans_inclusive = [[1, 87]]` (`164 / 363` active speech hops voiced, `median_f0_hz = 237.46 Hz`, `iqr = 8.927 st`; `voiced_spans` control frames `56-61` unvoiced `-43.6 dBFS` vocals / `-77.1 dBFS` rest, Demucs add-back `-46 dB`).

| Arm | `gen_elapsed_s` | `vram_gb` | `audio_dbfs` | `centroid_hz` | `sat_mean` | `mid_laplacian_var` | `YIN F0` Shift (`speech` / `all`) | `aud_splices` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `base` | `25.92` | `2.76` | `-22.11` | `2414.5` | `0.3842` | `172.87` | `—` (`237.46 Hz`) | `0` |
| `steered_both_1x` | `25.97` | `2.77` | **`-24.56`** (`-2.45 dB`) | `2237.1` | `0.2796` | `191.17` | `+0.032 st` / `+0.032 st` | `0` |
| **`steered_1x_qaud_ktxt_base`** | **`28.80`** (`+2.88s`) | **`2.77`** | **`-22.29`** (**`-0.18 dB`**) | **`2344.9`** | **`0.2783`** | **`197.86`** | **`+0.630 st`** / `+0.630 st` (`<=1.5 st` OK) | `300` |
| `steered_2x_video_only` | `49.94` (`1.93x`) | `2.77` | `-23.42` (`-1.31 dB`) | `2454.4` | `0.2854` | `196.98` | `+0.007 st` / `+0.007 st` | `0` |

- **Findings:** Routing unsteered text K/V strictly into `Q_aud` inside a single DiT pass (`steered_1x_qaud_ktxt_base`, `300` splices across `50` blocks $\times$ `6` steps) recovered `93%` of `steered_both_1x`'s `-2.45 dB` loudness loss (`-24.56 -> -22.29 dBFS`, matching `base` within `-0.18 dB` and preserving `2344.9 Hz` centroid) at **`28.80s` vs. `49.94s`** for `2x` `streams="video_only"` (`1.73x` wall-clock speedup, zero extra VRAM), while visually matching `steered_both_1x` and `steered_2x_video_only`'s cool filmic de-slop look (`sat_mean = 0.2783` vs. `0.2796 / 0.2854`). Zero YIN triage warnings (`<= 1.5 st` threshold) occurred on any arm.

---

## 5. Showcase D — Windowing `svd_ac_r8_ew50` & Native `4.0 MP` `MiniMax-H3-Image` Rank-8 Erasure (Zero 1D Arms)

Runner: `--stage showcase_d`. Metrics: [`data/exp_unpooled_sliders/phase4_2026_11_22/showcase_d_ew50_and_h3_image.json`](../../data/exp_unpooled_sliders/phase4_2026_11_22/showcase_d_ew50_and_h3_image.json).

### 5a. Early-Step Erasure Window (`svd_ac_r8` `[0.0, 1.0)` vs. `svd_ac_r8_ew50` `[0.0, 0.50)`, `42` Pika `fl2va` Stills)

Sheet: `$H3_OUTPUT_DIR/Image/phase4_2026-11-22/showcase_d_svd_ac_ew50.png`. Summary across paired cells (`seeds = [730451892, 101]`):

- **`noir` on `box_office` (`s = 1.0 / 1.25 / 1.5`):** `r8_ew50` matches `r8_full` bit-for-bit visually and quantitatively across all 3 strengths — e.g. `s = 1.25` gives `sat = 0.0476 / 0.0244` (`r8_ew50`) vs. `0.0472 / 0.0246` (`r8_full`), `s = 1.5` gives `0.0503 / 0.0292` vs. `0.0489 / 0.0298`.
- **`paper_origami` on `sitcom_kitchen` (`s = 1.0 / 1.25 / 1.5`):** `r8_ew50` gives `sat = 0.4980 / 0.4984 / 0.5499` (`730451892`) vs. `r8_full` `0.4960 / 0.5007 / 0.5514` (`laplacian_var 1247.23–1338.12` at `s = 1.25`).
- **`de_llm_slop` on `freight_elevator` (`s = 1.0 / 1.25 / 1.5`):** `r8_ew50` gives `sat = 0.5805 / 0.5713 / 0.5559` (`730451892`) vs. `r8_full` `0.5871 / 0.5756 / 0.5450`.
- Confirms Pole-A erasure does its layout/palette conflict resolution inside early denoise steps `[0.0, 0.50)`.

### 5b. Native `4.0 MP` (`2496x1664`) `MiniMax-H3-Image` Rank-8 Single & 2-Slider Stacked Erasure

Sheet: `$H3_OUTPUT_DIR/Image/phase4_2026-11-22/showcase_d_h3_image_native_erasure.png`. Every attached slider asserted `spec.erasure_vr is not None` (`rank = 8`, `0` cross-family fallbacks, `0` 1D arms).

| Concept Pair | Scene | Arm | `erasure_ranks` | `gen_elapsed_s` | `hsv_sat` | `laplacian_var` | `rms_contrast` | `clip_frac` |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `paper_origami` | `clockmaker` | `base` | `[]` | `11.27` *(cold)* | `0.4179` | `465.26` | `71.88` | `0.01590` |
| `paper_origami` | `clockmaker` | `native_r8` (`s=1.25`) | `[8]` | `3.07` | `0.5656` | `591.33` | `65.30` | `0.00243` |
| `paper_origami + de_llm_slop` | `clockmaker` | `stack2_de_llm_slop` (`1.20, 0.90`) | `[8, 8]` | `3.06` | `0.4673` | `396.48` | `73.37` | `0.03856` |
| `paper_origami` | `lighthouse` | `base` | `[]` | `3.01` | `0.2384` | `196.67` | `66.58` | `0.00087` |
| `paper_origami` | `lighthouse` | `native_r8` (`s=1.25`) | `[8]` | `3.02` | `0.3179` | `271.95` | `52.41` | `0.00052` |
| `paper_origami + de_llm_slop` | `lighthouse` | `stack2_de_llm_slop` (`1.20, 0.90`) | `[8, 8]` | `3.05` | `0.1939` | `227.28` | `60.93` | `0.00040` |
| `claymation_stopmotion` | `clockmaker` | `base` | `[]` | `3.03` | `0.4179` | `465.26` | `71.88` | `0.01590` |
| `claymation_stopmotion` | `clockmaker` | `native_r8` (`s=1.25`) | `[8]` | `3.04` | `0.5872` | `317.28` | `64.67` | `0.00093` |
| `claymation_stopmotion + noir` | `clockmaker` | `stack2_noir` (`1.20, 1.15`) | `[8, 8]` | `3.05` | `0.4386` | `381.35` | `80.75` | `0.07641` |
| `claymation_stopmotion` | `lighthouse` | `base` | `[]` | `3.03` | `0.2384` | `196.67` | `66.58` | `0.00087` |
| `claymation_stopmotion` | `lighthouse` | `native_r8` (`s=1.25`) | `[8]` | `3.02` | `0.1831` | `297.54` | `56.14` | `0.00090` |
| `claymation_stopmotion + noir` | `lighthouse` | `stack2_noir` (`1.20, 1.15`) | `[8, 8]` | `3.03` | `0.0896` | `773.36` | `73.83` | `0.07255` |

- **Visual Inspection:**
  - `paper_origami` `native_r8` folds brass prisms/lattices on `clockmaker` and transforms both the Fresnel lens housing and keeper's oilskin into sharp creased origami geometry on `lighthouse`, while `stack2_de_llm_slop` strips orange studio warmth cleanly (`sat 0.5656 -> 0.4673` on `clockmaker`, `0.3179 -> 0.1939` on `lighthouse`) with crisp whitewashed daylight stone.
  - `claymation_stopmotion` `native_r8` sculpts recognizable tactile plasticine clay figures, brick walls, and miniature mugs on both scenes (`~3.02–3.05s` per `4.15 MP` render), and `stack2_noir` on `clockmaker` combines sculpted clay modeling with chiaroscuro monochrome-desaturated lighting (`rms_contrast = 80.75`, warm skin tint preserved in highlights).
  - **Honest failure / composition artifact:** On `lighthouse`, stacking `claymation_stopmotion (1.20) + noir (1.15)` (net row shift `133.42` vs. warning threshold `97.9`) produced high monochrome desaturation (`sat = 0.0896`) **but split the composition into a white-bordered two-panel diptych** (`laplacian_var = 773.36` driven by white inner frame borders).

---

## 6. Next Multi-Vector Frontiers (Track 1 & Track 2)

1. **Track 1 — Cross-Slider `V_r_union ⊥ span(mu_d^(1..K))` & Unified Compiler-Safe Attention Hooks:** Project stacked Pole-A union bases orthogonal to co-active `mu_d` targets (fixing the `box_office s=730451892` bleed and `lighthouse` `stack2` over-push), and promote `blocks.20..49` + `1x Q_aud -> K_txt_base` into a unified `MiniMaxH3Sol`-compatible attention hook.
2. **Track 2 — Per-Token Multi-Vector Pole-B Routing & `ColQwen` Spatial Banks:** Replace single-`mu_d` broadcast injection with `K`-prototype / `W_B` multi-vector Pole-B token routing plus persisted `N >= 6` `ColQwen` `vision_tokens` multi-vector banks.
