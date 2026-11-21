# Phase 3 Steering & Elicitation Benchmark: `A5` Guardrails, `de_llm_slop`, `V_r` Centering (`Exp 1`), Same-Modality Vision-Token Erasure (`Exp 2`), and Shadow-Text K/V Block-Band Routing (`Exp 3`)

- **Date:** 2026-11-21
- **Runner:** [`run_phase3_steering_experiments.py`](../control_vector/run_phase3_steering_experiments.py)
- **Prior benchmark:** [`2026-11-21_multivector_erasure_benchmark.md`](2026-11-21_multivector_erasure_benchmark.md) · [`how_pole_a_erasure_works.md`](../../docs/wiki/steering/how_pole_a_erasure_works.md)
- **Raw JSON logs:** [`stage_a_de_llm_slop.json`](../../data/exp_unpooled_sliders/phase3_2026_11_21/stage_a_de_llm_slop.json) · [`stage_b_surgical_erasure_v2.json`](../../data/exp_unpooled_sliders/phase3_2026_11_21/stage_b_surgical_erasure_v2.json) · [`stage_c_vision_erasure.json`](../../data/exp_unpooled_sliders/phase3_2026_11_21/stage_c_vision_erasure.json) · [`stage_d_shadow_text_kv.json`](../../data/exp_unpooled_sliders/phase3_2026_11_21/stage_d_shadow_text_kv.json)
- **Media directories (machine-local):** `$H3_OUTPUT_DIR/Image/phase3_2026-11-21/` and `$H3_OUTPUT_DIR/Video/phase3_2026-11-21/` (zero media committed to git; every `.mp4` has a verified `.1fps.*.mp4` companion inspected natively).

---

## 1. Summary Verdict

1. **`A5` Silent-Failure Guardrails (`control_vectors.py`, `bench/control_vector/{cli,build_cmd,make_cmds}.py`):** Added `_BASELINE_STATE_WORDS` + `_DIALOGUE_SPAN_STRIP_RE` (`Gate D` shared-prefix contradiction warning across non-`<d>` prose), raised `Gate B` from `med_fd < 4` to `med_fd < 10`, and gated compiled `<compiled:...>` low-pair builds (`len(pairs) < MIN_ERASURE_PAIRS = 6`) from minting rank-deficient `erasure.vr` unless `--force-low-pair-erasure` is passed. All 51 tracked/bank TOMLs pass Gate D cleanly except intentional real bug catch `crystallize_ice.toml` Pair 8 (`'bright'`).
2. **Stage A (`de_llm_slop` Master EQ Strip, [`de_llm_slop.toml`](../../prompt_bank/control_vectors/de_llm_slop.toml)):** Rewriting v2 with strictly optical/exposure/texture tokens inside `[first_diff:end_diff]` raised `fl2va` pairwise agreement to `mpc = 0.5828`, `split_half = 0.8910` (`ref2va`: `0.5885` / `0.8936`; `h3_image`: `0.6078` / `0.9021`), with J-Lens `mu_d (+)` rescuing `shot, grain, indoor` and `vr[0..2]` erasing `cheesy, mush, smooth, glossy, blur`. Across 4 `heylook`-expanded non-market stills + synthetic gloss + a `90f` speaking `t2va_box_office.txt` clip, **`add_norm_matched_erasure` at `+1.15` (`sat_mean = 0.3615`) stripped neon/HDR bloom into flat practical daylight** where `Base` (`0.5964`) and `1D` (`0.5452`) kept warm glowing signage, while staying non-destructive on already-flat tungsten `freight_elevator` (`0.4690 -> 0.4517`).
3. **Stage B (`Exp 1` — Ramp, `r=8` `V_r` Centering Variants, & 3-Slider Stack):** On `noir` (`indoor_neutral`, `s = +1.5`), per-pair-centered `contested_svd_r8` (`hsv_sat = 0.0233`) and `svd_ac_r8` (`0.0264`) both hit crisp monochrome where `global_contested_svd_r8` (`0.1337` full / `0.1404` `ew50`) and `1D` (`0.1896`) failed to desaturate cleanly — showing per-pair baseline subtraction is critical when contrastively decomposing `A_p`. Simultaneously summing 3 un-orthogonalized shifts (`noir +1.25 + chaos_panic +1.0 + de_llm_slop +0.9`) confirmed strong multi-trait motion/action composition alongside a mild saturation dilution (`box_office`: `0.0649/0.0264` standalone `noir` vs `0.2603/0.2121` stacked).
4. **Stage C (`Exp 2` — Same-Modality `vision_tokens` Erasure with `erasure.mu_a`):** Implemented [`build_spatial_vision_erasure`](../../control_vectors.py) (`1:1` `B_p - A_p` + optional `"erasure.mu_a"` dataset Pole-A mean in `control_vectors.py` / `control_vector_nodes.py`) so uniform color casts on `tags == 0` vision patches are preserved rather than canceled by per-prompt centering. On `ref2va_woman_90f`, Gear 1 text-only erasure (`description_only`) cleanly yields **Selective-Color Subject Lock** (`center_sat = 0.3099`, `border_sat = 0.1726`, darkening green shop awnings to black), while Gear 2b + `noir_vis` (`rel = 0.17`, `clean = 0.85`) lowers subject saturation (`0.3099 -> 0.2763`) and `fl2va_noodle_141f` drops `-10.4%` (`0.4346 -> 0.3894`, `mid_sat = 0.4350 -> 0.3896`) without full monochrome conversion because MiniMax-H3 also pins raw VAE `cond_video_latents` / `ref_img` chroma in denoising steps `[0, 0.33]` ([`conditioning.py#L169-L185`](../../conditioning.py#L169-L185), `comfy/ldm/minimax/model.py:601-608, 749`).
5. **Stage D (`Exp 3` — Shadow-Text K/V Block-Band Routing, `blocks.0..49`):** Verified that evolving an unsteered shadow text stream `h_txt_base` (disabling Comfy's per-block `malloc_graph` arena via `disable_comfy_compiler=True` + `pause_malloc_graph()`, zero `patches_replace["dit"]`, chaining cleanly above `MiniMaxH3Sol`) matches `unsteered_base` cleanly when `0` blocks are active (`sanity_block0_diff = 0.0`, `300` swaps). Routing steered `paper_origami +1.55` text K/V ONLY to **`blocks.20..49`** (`mid_late_20_49`) achieves **folded-paper emergence (`border_lap = 140.66` vs `69.39` base)** while giving the **lowest portrait color drift (`center_vs_ref_sat_diff = 0.0192`)** and visible lip-sync, whereas `early_only_0_19` (`69.31`) and `late_only_30_49` (`77.94`) produce near-zero origami transformation — empirically isolating `blocks.20..29` as the critical text-to-video style breakout window.

---

## 2. Stage A — `de_llm_slop` Build Metrics & Non-Market Stills + `90f` AV (`t2va_box_office`)

Built across `fl2va`, `ref2va`, and `h3_image` into `$COMFY/models/control_vectors/bank/` (`de_llm_slop.toml`, `12` single-clause pairs, `0` Gate A–D lint warnings):

| Family | `raw_norm` | `mean_pairwise_cosine` | `split_half_cosine` | `erasure` Payload | Recommended Strength (`add_norm_matched_erasure`) |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `fl2va` | `53.38` | `0.5828` | `0.8910` | `r=8`, `cos(mu_d, v_1d)=0.9719`, `diag=[0.870..0.714]` | `0.74 .. 1.26` (sweet spot **`1.0 .. 1.2`**) |
| `ref2va` | `53.54` | `0.5885` | `0.8936` | `r=8` | `0.74 .. 1.26` |
| `h3_image` | `55.11` | `0.6078` | `0.9021` | `r=8` | `0.74 .. 1.26` |

### Still & `90f` Speaking AV Metrics (`stage_a_de_llm_slop.json`, `$H3_OUTPUT_DIR/Image/phase3_2026-11-21/stage_a_de_llm_slop_sheet.png`)

| Scene (`heylook` / Synthetic) | Arm (`s = +1.15`, `scene_only`) | Mean `hsv_sat` (`s=730451892` / `s=101`) | Laplacian Var (`s=730451892` / `s=101`) | Visual / Audio Observation |
| :--- | :--- | :---: | :---: | :--- |
| `box_office` (`heylook=True`) | `Base (0.0)` | `0.6732` / `0.6562` | `164.2` / `225.2` | Strong amber wall glow + glowing red/blue neon signage |
| `box_office` (`heylook=True`) | `1D (+1.15)` | `0.6511` / `0.5735` | `145.6` / `149.3` | Slightly dimmer interior, neon/orange tint persists |
| `box_office` (`heylook=True`) | `Erasure (+1.15 [0..1])` | **`0.5648` / `0.5106`** | `177.8` / `171.0` | Drops glowing neon sign & syrupy HDR cast to flat overcast glass |
| `hardware_aisle` (`heylook=True`) | `Base (0.0)` -> `Erasure (+1.15)` | `0.4974->0.3189` / `0.4869->0.3940` | `921.6->559.8` / `1314.8->818.3` | Strips orange LUT glow & creamy defocus to plain daylight tubes |
| `freight_elevator` (`heylook=True`) | `Base (0.0)` -> `Erasure (+1.15)` | `0.4690->0.4517` / `0.3763->0.3819` | `232.4->253.8` / `263.1->246.6` | Already neutral tungsten — stays non-destructive |
| `synthetic_gloss_bike_shop` | `Base` -> `1D` -> `Erasure (+1.15)` | `0.5862->0.3688->0.3570` (`s=101`: `0.5089->0.3343->0.3629`) | `1334->1354->1051` | Eliminates teal-orange rim halo & shallow bokeh polish |
| **`90f` AV (`t2va_box_office`)** | `base` -> `1d_s1p15` -> `erasure_s1p15` | `0.5964` -> `0.5452` -> **`0.3615`** | `158.7` -> `195.2` -> `150.8` | Audio RMS `-14.55` -> `-15.67` -> `-27.93 dBFS` (quieter score/ambience, exact `"Enjoy the feature"` lip-sync at `00:01`); `base`/`1d` retain orange neon bloom vs. `erasure` overcast documentary glass |

---

## 3. Stage B (`Exp 1`) — Strength Ramp, `r=8` `V_r` Centering Variants, & 3-Slider Stack

From [`stage_b_surgical_erasure_v2.json`](../../data/exp_unpooled_sliders/phase3_2026_11_21/stage_b_surgical_erasure_v2.json) (`90` ramp + `16` stack renders, `r = 8` preserved throughout — no reduction to `r = 4`):

### 3.1 Strength Ramp & `V_r` Construction Comparison at `s = +1.5` (`SEED_MAIN = 730451892`)

| Concept / Scene | `Base (s=0)` `sat / lap` | `1D (r=0)` `sat / lap` | `svd_ac_r8` `sat / lap` | `contested_svd_r8` `sat / lap` | `global_contested_svd_r8` `[0..1]` / `ew50` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `noir` (`indoor_neutral`) | `0.3201 / 281.8` | `0.1896 / 340.7` (colored) | **`0.0264 / 686.6`** (crisp B&W) | **`0.0233 / 541.8`** (crisp B&W) | `0.1337 / 357.7` (stays warm) · `ew50: 0.1373` |
| `paper_origami` (`sitcom_kitchen`) | `0.3840 / 711.2` | `0.6371 / 1454.3` (legs/cloth human) | **`0.5706 / 1521.4`** (full paper fold) | `0.5826 / 1394.6` | `0.5513 / 1322.5` · `ew50: 0.5487` |
| `chaos_panic` (`hardware_aisle`) | `0.4072 / 654.4` | `0.3804 / 366.9` | `0.3370 / 503.8` | `0.3471 / 360.0` | `0.3200 / 369.4` · `ew50: 0.3215` |

### 3.2 Simultaneous 3-Slider Stack (`noir +1.25 + chaos_panic +1.0 + de_llm_slop +0.9`)

| Prompt (`seed=730451892` / `s=101`) | `Base` `sat / lap` | `Single noir +1.25` `sat / lap` | `Stack3 1D (n+c+d)` `sat / lap` | `Stack3 Erasure (n+c+d)` `sat / lap` |
| :--- | :---: | :---: | :---: | :---: |
| `box_office` | `0.2920 / 215.3` · `0.3596 / 164.9` | **`0.0649 / 449.4` · `0.0264 / 438.9`** | `0.1637 / 208.8` · `0.1068 / 346.2` | `0.2603 / 185.4` · `0.2121 / 301.3` |
| `sitcom_kitchen` | `0.3840 / 711.2` · `0.3634 / 834.2` | `0.2975 / 494.8` · `0.2678 / 611.3` | `0.2715 / 193.2` · `0.2627 / 338.5` | `0.3269 / 124.9` · `0.2311 / 124.3` |

- **Observed tradeoff:** Stacking `noir` alongside high-strength `chaos_panic` and `de_llm_slop` produces vigorous frantic motion blur (`lap = 124`) and clean daylight tones, **but naive linear summation of three un-orthogonalized erasure vectors partially washes out `noir`'s monochrome** (`box_office` `0.0649 -> 0.2603`). Pair `de_llm_slop` with realistic/color looks or attenuate competing looks cleanly rather than assuming additive orthogonality across 3 strong `add_norm_matched_erasure` blocks.

---

## 4. Stage C (`Exp 2`) — Same-Modality `vision_tokens` Erasure (`erasure.mu_a`)

Built `10`-pair `color -> greyscale` `vision_tokens` vectors via `make_control_vector_from_media_pairs(..., pooling="vision_tokens")` into `$H3_OUTPUT_DIR/Video/phase3_2026-11-21/scratch_vectors/`, verifying `"erasure.mu_a"` (`[5376]`) is written and applied via `_erasure_row_shift64` on `tags == 0` vision patches ([`stage_c_vision_erasure.json`](../../data/exp_unpooled_sliders/phase3_2026_11_21/stage_c_vision_erasure.json)):

### 4.1 `<Picture 1>` `ref2va` (`woman_blonde.jpg`, `90f`, `vis=338`, `desc=426`)

| Arm (`noir +1.5` text + optional `noir_vis +1.25`) | `sat_mean` | `center_sat_mean` | `border_sat_mean` | `audio_dbfs` | Visual Result (Native Inspection) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `gear0_1d_text` (`1D`, `end_percent=1.0`) | `0.2099` | `0.3069` | `0.1899` | `-22.45` | Green awning, warm bricks & colored street stay saturated |
| **`gear1_selective_color_erasure`** (`Erasure`, `end_percent=1.0`) | `0.1966` | **`0.3099`** | **`0.1726`** | `-22.47` | **Selective-Color Subject Lock:** Green awning turns matte charcoal/black & street cools silver-grey while subject stays vibrant blonde/skin |
| `gear2a_vis_erasure_rel033` (`+noir_vis`, `rel=0.33`) | **`0.1919`** | `0.2868` | `0.1758` | `-22.59` | Street desaturated; subject blonde slightly muted |
| `gear2b_vis_erasure_rel017_clean085` (`rel=0.17`, `clean=0.85`) | `0.1924` | **`0.2763`** | `0.1799` | `-21.25` | Drops portrait chroma (`-10.8%` vs Gear 1), though warm skin/hair persists via raw VAE `ref_img` channel injection in step 0 |

### 4.2 `fl2va` First+Last Frame Noodle Counter (`fl2va_noodle_f000 -> f123`, `141f` `on_both_clocks`, `vis=2020`, `scene=240`)

| Arm (`noir +1.55` text + optional `noir_vis +1.35`) | `sat_mean` | `sat_mid_mean` | `mid_laplacian_var` | `audio_dbfs` | Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `text_only_erasure_rel033` | `0.4346` | `0.4350` | `530.98` | `-24.17` | Confirms Showcase 3 (`0.4328`): red neon `"老张牛肉面"` & cyan stalls stay bright |
| **`dual_text_plus_vis_rel033`** | **`0.3894`** (`-10.4%`) | **`0.3896`** | `533.93` | `-24.98` | **`vision_tokens` (`tags==0`) `erasure.mu_a` measurably cools background saturation (`-10.4%`)**, while raw VAE `cond_video_latents` in steps `[0, 0.33]` (`b in [0.50, 1.0]`) keep endpoint neon red chroma locked ([`conditioning.py#L169-L185`](../../conditioning.py#L169-L185)) |
| `dual_text_plus_vis_rel017_clean085` | `0.4013` | `0.4027` | `537.08` | `-24.80` | Earlier release + `0.85` noise yields less chroma cooling than `0.33` at `1.35` vis strength |

---

## 5. Stage D (`Exp 3`) — Shadow-Text K/V Block-Band Routing (`blocks.0..49`)

Tested on `<Picture 1>` `woman_blonde.jpg` (`90f`, `MiniMaxH3ReleaseReferences(end_percent=0.50)`, `paper_origami +1.55` `Erasure` on `description_only`, chaining cleanly above `MiniMaxH3Sol`, zero `patches_replace["dit"]`; [`stage_d_shadow_text_kv.json`](../../data/exp_unpooled_sliders/phase3_2026_11_21/stage_d_shadow_text_kv.json)). Running cross-block shadow states required disabling Comfy's per-block `malloc_graph` arena cleanup (`disable_comfy_compiler = True` + `pause_malloc_graph()`, since `comfy/ldm/minimax/model.py:658` destroys per-block arenas between `blk i` and `blk i+1`):

| Arm | Active Steered Blocks | K/V Swaps (`h_txt_base`) | `sat_mean` | `border_laplacian_energy` | `center_vs_ref_sat_diff` | `audio_dbfs` | Visual Result (Native Inspection) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `unsteered_base` (`s=0.0`, sanity diff `0.0`) | `0` | `300` | `0.2583` | `69.39` | `0.0436` | `-20.35` | Realistic walk & speech; zero origami — confirms exact shadow parity |
| `early_only_0_19` | `0..19` (`20`) | `180` | `0.2479` | `69.31` | `0.0638` | `-20.92` | **Zero origami fold:** virtually indistinguishable from `unsteered_base` |
| `late_only_30_49` | `30..49` (`20`) | `180` | `0.2522` | `77.94` | `0.0387` | `-20.84` | **Zero origami fold:** misses critical `20..29` transition band |
| **`mid_late_20_49`** | **`20..49` (`30`)** | `120` | `0.1871` | **`140.66`** (`+103%`) | **`0.0192`** (best steered) | `-21.13` | **Clean origami star eyewear + visible lower face/mouth lip-sync** + lowest identity color drift |
| `all_blocks_0_49` | `0..49` (`50`) | `0` | `0.1858` | `180.88` | `0.0264` | `-22.00` | Full 8-point white origami burst covering eyes & upper nose |
| `all_plus_lse_ref_balance` (`Sol dense=30..49`, `w=1.65`, `ref=[0]`) | `0..49` (`50`) | `0` | `0.1852` | **`186.37`** | `0.0326` | `-21.30` | Sharper folded starburst & warmer clothing (`+3.0%` border energy over un-balanced `all_blocks`), though full `0..49` text still masks eyes |

**Mechanistic takeaway:** Keeping `blocks.0..19` on unsteered `h_txt_base` and releasing steered text K/V ONLY in **`blocks.20..49`** hits the `blk 20` video emergence boundary directly: `blocks.0..19` alone produce `0` style change (`border_lap = 69.31` vs `69.39`), `blocks.30..49` alone produce near-zero (`77.94`), and `blocks.20..49` captures both structural origami shift (`+103%` border Laplacian energy) and tighter `<Picture 1>` portrait color lock (`center_vs_ref_sat_diff = 0.0192`).
