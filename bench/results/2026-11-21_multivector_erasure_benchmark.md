# 2026-11-21 Multi-Vector Pole-A Subspace Erasure Benchmark & 38-Concept Bank Census

Date: `2026-11-21`  
Runner: [`bench/control_vector/eval_multivector_erasure.py`](../control_vector/eval_multivector_erasure.py) + [`bench/verify_pole_a_erasure.py`](../verify_pole_a_erasure.py)  
Owning paper: [`docs/wiki/research/control_vectors/2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md`](../../docs/wiki/research/control_vectors/2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md)  
Practitioner guide: [`docs/wiki/control_vectors/tutorials_progressive_guide.md`](../../docs/wiki/control_vectors/tutorials_progressive_guide.md)

## 1. Reproducer Commands

```bash
COMFY="$(cd ../.. && pwd)" && PY="$COMFY/.venv/bin/python"
set -a; . ./h3.env; set +a

# CPU algebraic proofs (< 5 s)
CUDA_VISIBLE_DEVICES="" "$PY" bench/verify_pole_a_erasure.py --cpu-only
CUDA_VISIBLE_DEVICES="" "$PY" bench/control_vector/eval_multivector_erasure.py --stage census
CUDA_VISIBLE_DEVICES="" "$PY" bench/control_vector/eval_multivector_erasure.py --stage cpu_math

# GPU benchmark stages (writes JSON to data/exp_unpooled_sliders/paper_2026_11_21/
# and renders to $H3_OUTPUT_DIR/{Image,Video}/paper_erasure_2026-11-21/)
"$PY" bench/control_vector/eval_multivector_erasure.py --stage ablations
"$PY" bench/control_vector/eval_multivector_erasure.py --stage word_media
"$PY" bench/control_vector/eval_multivector_erasure.py --stage audio_video
"$PY" bench/control_vector/eval_multivector_erasure.py --stage base50
```

---

## 2. Production 38-Concept Bank Census (`76` `.safetensors` Files)

Source: `data/exp_unpooled_sliders/paper_2026_11_21/census_38_bank.json`.
- **Files audited:** `76` (`38` stems $\times$ `{fl2va, ref2va}` in `models/control_vectors/bank/`).
- **Regime split:** `10` `multi_facet_bundle`, `21` `intermediate`, `7` `monolithic_override`.
- **Summary medians:** `median_mpc = 0.6763`, `median_cos_mu_d_v1d = 0.9935` (`min = 0.9479` on `film_16mm_1974_neutral`), `median_cos_fl2va_ref2va_1d = 0.9892`, **`median_vr_grassmann_overlap_fl2va_ref2va = 0.9323`**.
- **Target-word rescues (`erasure.mu_d` top-8 vs. `v_1d` top-8):** **`28 / 38` concepts (`73.7%`)** rescue domain-specific words via asymmetric `tau = 0.15` row-softmax matching.

| Concept Stem | Regime | Pairs | `mpc` | `1D_E%` | `cos(mu_d, v_1d)` | `fl2va↔ref2va` 1D / `V_r` Grassmann | Words Rescued in `mu_d` | Top Pole-A Words Erased (`V_r`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `infomercial_fail` | `multi_facet_bundle` | 10 | `0.4333` | — | `0.9935` | `0.9910` / `0.9294` | `panicked` | `motion, movement, quiet, paper` |
| `wes_anderson_symmetry` | `multi_facet_bundle` | 10 | `0.4531` | `22.62%` | `0.9951` | `0.9888` / `0.9545` | — | `miserable, helpless, voice, opposite` |
| `handheld_documentary_zoom` | `multi_facet_bundle` | 10 | `0.5255` | — | `0.9932` | `0.9796` / `0.9065` | — | `natural, dark, neat, stationary` |
| `elastic_neck_stretch` | `multi_facet_bundle` | 10 | `0.5721` | — | `0.9952` | `0.9907` / `0.9323` | `transformation` | `prominent, visibly, realistic` |
| `noir` | `multi_facet_bundle` | 12 | `0.5742` | `21.93%` | `0.9702` | `0.9876` / `0.9468` | **`black, footage, archival`** | `warm, natural, lighting, pattern` |
| `balloon_head_inflate` | `multi_facet_bundle` | 10 | `0.5819` | — | `0.9958` | `0.9896` / `0.9326` | — | `human, main, humanoid, earnest` |
| `head_swell` | `multi_facet_bundle` | 10 | `0.5882` | — | `0.9934` | `0.9886` / `0.9482` | — | `human, humanoid, male, your` |
| `zero_gravity` | `multi_facet_bundle` | 10 | `0.5895` | — | `0.9937` | `0.9903` / `0.9533` | `desert` | `horizontal, vertical, depth, scattered` |
| `smoke_dissolve` | `multi_facet_bundle` | 10 | `0.6091` | — | `0.9982` | `0.9906` / `0.9445` | — | `ambient, sunlight, lighting, clear` |
| `chaos_panic` | `multi_facet_bundle` | 10 | `0.7110` | `29.61%` | `0.9916` | `0.9920` / `0.9356` | — | `wearing, hands, aged, focused` |
| `magnetic_attraction` | `intermediate` | 10 | `0.6202` | — | `0.9934` | `0.9918` / `0.9681` | `explosive, tiny` | `motion, near, position, natural` |
| `wind_tunnel_gale` | `intermediate` | 10 | `0.6275` | — | `0.9935` | `0.9907` / `0.8808` | `wind, frantic` | `silent, quiet, motion, exactly` |
| `cursed_cctv` | `intermediate` | 10 | `0.6295` | — | `0.9933` | `0.9897` / `0.9173` | `grey, overhead, viewed` | `live, animated, solo, dead` |
| `pyro_klaxon_audio` | `intermediate` | 10 | `0.6352` | — | `0.9975` | `0.9856` / `0.8617` | — | `quiet, silent, silence, gentle` |
| `sunken_coral_reef` | `intermediate` | 10 | `0.6362` | — | `0.9969` | `0.9901` / `0.9657` | — | `dryer, repetitive, noise, normal` |
| `fever_melt` | `intermediate` | 10 | `0.6398` | — | `0.9950` | `0.9900` / `0.9487` | `phantom` | `live, animated, scene, presumably` |
| `underwater_submerged` | `intermediate` | 10 | `0.6449` | — | `0.9983` | `0.9891` / `0.8889` | `floating` | `dryer, scary, desert, flattering` |
| `whisper_asmr` | `intermediate` | 10 | `0.6499` | — | `0.9929` | `0.9854` / `0.8777` | `whisper` | `contemplation, quietly, outdoors` |
| `vocal_strain` | `intermediate` | 10 | `0.6670` | — | `0.9923` | `0.9812` / `0.9235` | `giant` | `injured, distressed, tent, pale` |
| `crystallize_ice` | `intermediate` | 10 | `0.6733` | — | `0.9984` | `0.9911` / `0.9549` | `freezing` | `tone, natural, indoor, sunlight` |
| `auctioneer_rapid_patter` | `intermediate` | 10 | `0.6793` | — | `0.9913` | `0.9851` / `0.8347` | `streak` | `compared, gradual, altogether, hands` |
| `strobe_club_pulse` | `intermediate` | 10 | `0.6840` | — | `0.9982` | `0.9866` / `0.9302` | `black` | `smooth, smoothly, movement, daylight` |
| `pyro_chain_explosions` | `intermediate` | 10 | `0.6868` | — | `0.9963` | `0.9920` / `0.8940` | — | `focus, focused, dusty, surrounded` |
| `miniature_tiltshift` | `intermediate` | 10 | `0.6892` | — | `0.9925` | `0.9896` / `0.9062` | `puppet, animate` | `realistic, soft, effectively` |
| `lettering` | `intermediate` | 10 | `0.6922` | — | `0.9895` | `0.9892` / `0.9284` | `feminine` | `blank, empty, male, glitch` |
| `paper_origami` | `intermediate` | 10 | `0.6987` | `36.34%` | `0.9976` | `0.9887` / `0.9323` | `sharply` | `live, video, scene, lily` |
| `claymation_stopmotion` | `intermediate` | 10 | `0.7108` | — | `0.9909` | `0.9885` / `0.9450` | `puppet` | `video, live, alive, outdoors` |
| `bioluminescent_glow` | `intermediate` | 10 | `0.7110` | — | `0.9984` | `0.9915` / `0.9603` | `swirling, magical` | `male, female, mart, clean` |
| `felt_muppet` | `intermediate` | 10 | `0.7147` | — | `0.9947` | `0.9906` / `0.9675` | `tent` | `slight, markedly, poised, realistic` |
| `cyberpunk_neon_rain` | `intermediate` | 10 | `0.7159` | — | `0.9976` | `0.9892` / `0.9547` | `noir` | `male, boring, nice, canvas` |
| `thermal_infrared` | `intermediate` | 10 | `0.7185` | — | `0.9949` | `0.9902` / `0.9456` | `patches, convex` | `live, video, animated, outdoors` |
| `molten_lava_golem` | `monolithic_override` | 10 | `0.7091` | `44.82%` | `0.9987` | `0.9892` / `0.9374` | `glowing` | `worn, occasion, street, realistic` |
| `operatic_bellow` | `monolithic_override` | 10 | `0.7504` | — | `0.9933` | `0.9835` / `0.9155` | — | `warm, surreal, candid, natural` |
| `reverb_cathedral` | `monolithic_override` | 12 | `0.7721` | — | `0.9919` | `0.9839` / `0.8497` | `majestic, mystical, ancient` | `exterior, minimal, clear` |
| `vhs_1991_camcorder` | `monolithic_override` | 10 | `0.7899` | — | `0.9914` | `0.9868` / `0.8264` | `jittery` | `characters, nighttime, foreground, bright` |
| `rubber_hose_1930s` | `monolithic_override` | 10 | `0.8039` | — | `0.9926` | `0.9896` / `0.9374` | `drawn` | `video, scene, live, outdoors` |
| `film_16mm_1974` | `monolithic_override` | 14 | `0.8072` | — | `0.9529` | `0.9851` / `0.9239` | **`grain, tape`** | `professional, humanoid, meticulously` |
| `film_16mm_1974_neutral` | `monolithic_override` | 14 | `0.8072` | — | `0.9479` | `0.9846` / `0.9270` | **`grain, tape`** | `professional, humanoid, model` |

---

## 3. Controlled GPU Ablations 1–7 (Matched `RMS = 101.25` @ `s = 1.5`, Non-Market Prompts)

Sources: `data/exp_unpooled_sliders/paper_2026_11_21/{cpu_math_ablations,gpu_ablations_1_to_5,gpu_word_and_media_ablations}.json`, `$H3_OUTPUT_DIR/Image/paper_erasure_2026-11-21/ablation{1..7}_*.png`.

### 3.1 Ablation 1 — Soft-MaxSim Temperature $\tau_{\text{attn}}$ & De-Coning (`noir` on `indoor_neutral`, 3 Seeds)

| Arm | `s=730451892` `hsv_sat` | `s=101` `hsv_sat` | `s=2026` `hsv_sat` | B&W Rate (`< 0.06`) |
| :--- | :---: | :---: | :---: | :---: |
| `1D (tau=inf)` | `0.1896` | `0.1592` | `0.2260` | `0 / 3` |
| `tau=0.50` | **`0.0251`** | **`0.0223`** | **`0.0303`** | **`3 / 3`** |
| `tau=0.15 (shipped)` | **`0.0264`** | **`0.0223`** | **`0.0223`** | **`3 / 3`** |
| `tau=0.07` | **`0.0260`** | **`0.0236`** | **`0.0261`** | **`3 / 3`** |
| `MaxSim (tau=0.03)` | `0.1992` | **`0.0245`** | `0.1846` | `1 / 3` |
| `tau=0.15 deconed` | `0.1702` | **`0.0211`** | `0.1943` | `1 / 3` |

### 3.2 Ablation 2 — `V_r` Construction (`svd_ac` vs. `skiplist_svd` vs. `contested_svd`)

| Concept (Scene) | Seed | `1D (r=0)` | `svd_ac (r=8)` | `skiplist_svd (r=8)` | `contested_svd (r=8)` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `noir` (`indoor_neutral`) | `730451892` | `0.1896` *(color)* | **`0.0264` *(B&W)*** | `0.1187` *(wood leak)* | **`0.0233` *(B&W)*** |
| `noir` (`indoor_neutral`) | `101` | `0.1592` *(color)* | **`0.0223` *(B&W)*** | **`0.0239` *(B&W)*** | **`0.0244` *(B&W)*** |
| `paper_origami` (`sitcom_kitchen`) | `730451892` | `0.6371` *(daughter deleted)* | **`0.5706` *(father+daughter+dog)*** | **`0.5418` *(father+daughter+dog)*** | **`0.5826` *(father+daughter+dog)*** |
| `paper_origami` (`sitcom_kitchen`) | `101` | `0.5904` *(daughter deleted)* | **`0.5687` *(father+daughter+dog)*** | **`0.5983` *(father+daughter+dog)*** | **`0.5317` *(father+daughter+dog)*** |

### 3.3 Ablation 3 — Rank $r$ & Ridge $\lambda_{\text{rel}}$ (`seed = 730451892`)

- **Rank sweep ($\lambda_{\text{rel}} = 0.15$):**
  - `noir` `hsv_sat`: `r=0: 0.1896`, `r=1: 0.0258`, `r=2: 0.0287`, `r=4: 0.0249`, `r=8: 0.0264`, `r=16: 0.0245`.
  - `paper_origami` `hsv_sat` (+ visual): `r=0: 0.6371` *(missing daughter)*, `r=1: 0.5875` *(human skin)*, `r=2: 0.5736`, `r=4: 0.5675`, `r=8: 0.5706` *(both figures crisp origami)*, `r=16: 0.5395` *(both crisp origami)*.
  - `watercolor_wash` `hsv_sat`: `r=0: 0.4153`, `r=1: 0.4021`, `r=2: 0.3867`, `r=4: 0.3922`, `r=8: 0.4138`, `r=16: 0.4095`.
- **Ridge sweep ($r = 8$):**
  - `noir` `hsv_sat`: `0.01 -> 0.0326`, `0.05 -> 0.0241`, `0.15 -> 0.0264`, `0.50 -> 0.0247`, `1.50 -> 0.1643` *(color leak when over-damped)*.

### 3.4 Ablation 4 — Pole-A Pair Count Coverage Curve (`cpu_math M3` + GPU Renders)

| Concept | Pairs $N$ | Monte-Carlo `V_r` Grassmann Overlap vs. Full | `cos(mu_d, full)` | Rendered `hsv_sat` (`s=730451892`) | Visual Observation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `noir` | `1` | `0.4061 ± 0.0221` | `0.7933` | `0.1654` | Warm color room |
| `noir` | `2` | `0.5207 ± 0.0277` | `0.8888` | `0.1404` | Warm amber wood + shadows |
| `noir` | `5` | `0.6856 ± 0.0411` | `0.9632` | `0.1210` | Muted sepia-tinted |
| `noir` | **`12`** | **`1.0000`** | **`1.0000`** | **`0.0264`** | **Pure silver-gelatin B&W** |
| `paper_origami` | `1` | `0.4879 ± 0.0236` | `0.8577` | `0.6522` | Live-action humans, paper table props |
| `paper_origami` | `2` | `0.6019 ± 0.0292` | `0.9276` | `0.5859` | Live-action skin, geometric backdrop |
| `paper_origami` | `5` | `0.7452 ± 0.0407` | `0.9796` | `0.5737` | Folded-paper father & daughter emerge |
| `paper_origami` | **`10`** | **`1.0000`** | **`1.0000`** | **`0.5706`** | **Sharp folded-paper father, daughter, dog, props** |

### 3.5 Ablation 5 — Denoising Step Windows (`noir` & `paper_origami`)

| Step Schedule Arm | `noir` `hsv_sat` / `red_excess` | `paper_origami` `hsv_sat` / `red_excess` |
| :--- | :---: | :---: |
| `1D [0..1]` | `0.1896` / `+9.59` | `0.6371` / `+43.96` |
| `Both [0..1] (default)` | **`0.0264` / `+0.16`** | **`0.5706` / `+42.46`** |
| `Erase [0..0.34] + mu_d [0..1]` | `0.0679` / `+2.69` | `0.5676` / `+42.45` |
| `Erase [0..0.50] + mu_d [0..1]` | **`0.0281` / `+0.35`** | **`0.5699` / `+42.61`** |
| `Both [0..0.50]` | **`0.0439` / `+1.15`** | **`0.5682` / `+42.68`** |
| `Late Erase [0.34..1]` | `0.2974` / `+20.40` *(fails)* | `0.3825` / `+26.30` *(live-action)* |

### 3.6 Ablations 6 & 7 — `MiniMaxH3WordVector` (`Model2Vec` + `against`-SVD) & `MiniMaxH3MakeMediaControlVector` (`ColQwen` Patch Erasure)

- **Ablation 6 (`WordVector` vs. 10-pair GPU):** `WordVec 1D` (`noir: 0.3155`, `origami: 0.4535`) vs. `WordVec Model2Vec+Erase` (`noir: 0.3234`, `origami: 0.4539`, `used_model2vec_lexicon = true`) vs. `10-Pair GPU+Erase` (`noir: 0.0264`, `origami: 0.5706`).
- **Ablation 7 (`MakeMediaControlVector`, `<Picture 1>` `ref2va_scene_kitchen`, `rank = 8`, `diag = [0.87, 0.686, 0.651, 0.626, 0.608, 0.568, 0.542, 0.52]`):** `s=730451892`: `MakeMedia 1D = 0.3457` vs. `ColQwen+Erase = 0.3446`; `s=101`: `0.3371` vs. `0.3435` (single-pair `N=1` ~`40%` subspace bound).

---

## 4. Joint Audio-Video (`192f` & `90f`), Stratified Bank, & Undistilled 50-Step Base H3

Sources: `data/exp_unpooled_sliders/paper_2026_11_21/{gpu_audio_video_ablations,gpu_base50_and_stratified}.json`, `$H3_OUTPUT_DIR/Image/paper_erasure_2026-11-21/ablation{8c,9}_*.png`, `$H3_OUTPUT_DIR/Video/paper_erasure_2026-11-21/*.mp4`.

### 4.1 Ablation 8a — Clean Qwen-BPE `pieces` `192f` (`8.00s`) AV Retrieval Comparison (`t2va_hardware_aisle_short`, `31/268` Non-Alnum Clamped)

| Arm (`seed = 730451892`, `1152x768 @ 192f`) | `sat_mean` ± `sat_std` | `temp_l1_mean` | `audio_dbfs` | `centroid_hz` | Elapsed | Native `.mp4` Visual + Audio Inspection |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `A2_1d_clean_sif_fixed_pieces` | `0.2393 ± 0.0434` | `6.190` | `-27.55` | `3708.4` | `82.3s` | **Full color** across all 3 shots (blue shirt, pink skin, red signs); synced speech + bolt foley. |
| `M5_ward_warp_clean_sif_fixed_pieces` | **`0.1341 ± 0.0100`** | `6.674` | `-25.49` | `3172.2` | `77.7s` | **Crisp 3-shot monochrome**; synced 2-speaker dialogue + bolt/bag foley (requires corpus `-2PC` + SIF). |
| **`S1_pole_a_erasure`** | **`0.1362 ± 0.0093`** | `7.094` | `-25.93` | `2874.5` | `78.2s` | **Crisp 3-shot silver-gelatin monochrome**, lowest flicker (`0.0093`), 100% synced dialogue + bolt scoop, **zero corpus PCA / zero SIF**. |

### 4.2 Ablation 8b — `40 Hz` Stereo Audio-Stream Erasure (`90f` `t2va_sitcom_kitchen`)

| Concept | Mask | Mode | `sat_mean` | `audio_dbfs` | `spectral_centroid_hz` | `tail_sustain_ratio` |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `base` | `none` | `none` | `0.4108` | `-20.98` | `2403.1` | `0.3856` |
| `reverb_cathedral` | `soundscape_only` | `add_norm_matched_1d` | `0.3857` | `-23.22` | `1912.4` | `0.2631` |
| `reverb_cathedral` | `soundscape_only` | **`add_norm_matched_erasure`** | **`0.3972`** | `-21.62` | **`2409.4`** | **`0.3217`** *(+22.3% tail vs 1D)* |
| `whisper_asmr` | `scene+soundscape` | `add_norm_matched_1d` | `0.4146` | `-13.98` | `3232.8` | `0.4898` |
| `whisper_asmr` | `scene+soundscape` | **`add_norm_matched_erasure`** | `0.4359` | `-13.98` | **`3425.4`** | `0.4785` |
| `underwater_submerged` | `scene+soundscape` | `add_norm_matched_1d` | `0.4007` | `-14.62` | `2656.3` | `0.4158` |
| `underwater_submerged` | `scene+soundscape` | **`add_norm_matched_erasure`** | `0.3951` | `-13.99` | **`3201.3`** | **`0.4217`** *(buoyant floating father)* |

### 4.3 Ablation 8c — Stratified Bank Concepts (`Pika 6-Step`) & Ablation 9 — Undistilled 50-Step Base H3 (`loras=[]`)

- **Stratified Bank (`ablation8c_stratified_bank_concepts.png`):**
  - `film_16mm_1974` (`hardware_aisle`): `1D = 0.3010` vs. `Pole-A Erasure = 0.4107` (richer 1974 Kodak grain/warmth).
  - `cursed_cctv` (`indoor_neutral`): `1D = 0.2346` vs. `Pole-A Erasure = 0.2521` (overhead dim greenish CCTV cast).
  - `bioluminescent_glow` (`indoor_neutral`): `1D = 0.2883` vs. `Pole-A Erasure = 0.3040`.
  - `felt_muppet` (`sitcom_kitchen`): `1D = 0.5212` vs. `Pole-A Erasure = 0.5250` (fuzzy Muppet-style family).
- **Undistilled 50-Step Base `fl2va` (`ablation9_base50_undistilled_h3.png` + `base50_39f_noir_*.mp4`, Euler 50 steps, `cfg = 1.0`):**
  - `noir` (`indoor_neutral`): `seed 730451892`: `Base50 1D = 0.2222` *(color)* $\to$ **`Base50 Pole-A Erasure = 0.0402` *(crisp B&W)***; `seed 101`: `Base50 1D = 0.2398` *(color)* $\to$ **`Base50 Pole-A Erasure = 0.0469` *(crisp B&W)***.
  - `paper_origami` (`sitcom_kitchen`): `seed 730451892`: `0.5864` vs. **`0.5536`**; `seed 101`: `0.4139` vs. **`0.4068`** (cleaner folded kitchen surfaces and intact percolator).
  - `39f` speaking AV (`sitcom_kitchen`, `cfg = 1.0`): `base50_39f_noir_1d` `sat_mean = 0.4138` (bright green cabinets & yellow chairs) $\to$ `base50_39f_noir_erasure` **`sat_mean = 0.3599`** (all green cabinets, yellow chairs, and walls stripped to white/grey monochrome while preserving spoken dialogue).
