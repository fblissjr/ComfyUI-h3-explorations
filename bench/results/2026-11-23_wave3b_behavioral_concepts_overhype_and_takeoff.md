# Wave 3 Part B Behavioral Concepts: `youtube_overhype_influencer` & `sudden_takeoff_flight` (`192f`)

**Date:** `2026-11-23`  
**Seed / Canvas / Backend:** `seed = 730451892`, `length = 192`, `1152x768`, `6`-step Pika SVD995 (`MiniMax-H3`, `MiniMaxH3Sol`, `split_backend = "auto_1x_kv"`)  
**Ledger, Prompts, & Outputs:** [`data/exp_unpooled_sliders/wave3b_2026_11_23/wave3b_results.json`](../../data/exp_unpooled_sliders/wave3b_2026_11_23/wave3b_results.json) · [`prompt_bank/t2va_stairwell_dialogue.txt`](../../prompt_bank/t2va_stairwell_dialogue.txt) · [`prompt_bank/t2va_motorcycle_highway.txt`](../../prompt_bank/t2va_motorcycle_highway.txt) · `$H3_OUTPUT_DIR/Video/wave3b_2026_11_23/` (`overhype/`, `takeoff/`, `sheets/`)

---

## Summary & Promotion Decision

Both behavioral concepts passed empirical visual and acoustic evaluation on `:8188` at `1152x768 @ 192f` and were **promoted** into [`prompt_bank/control_vectors/`](../../prompt_bank/control_vectors/) alongside [`prompt_bank/t2va_motorcycle_highway.txt`](../../prompt_bank/t2va_motorcycle_highway.txt), bringing the active repository bank to **`50` active TOMLs (`50 fl2va`, `50 ref2va`, `43 h3_image` = `143` `.safetensors` in `../../models/control_vectors/bank/`)** with **`0` collisions** (`bench/check_steering_parity.py`: `47/50 = 94.0%` lexicon coverage on the existing `204`-term lexicon; rebuild deferred as optional polish).

1. **`youtube_overhype_influencer` (`PROMOTED` — `fl2va`, `ref2va`, `h3_image`):** Authored with `N = 8` diverse adult dialogue scenes keeping `<d>[English] ...</d>` lines 100% verbatim across poles (`0` lint warnings in `control_vectors.lint_pair_set`, `first_diff_tok = 35..42`, median `24`). Delivered `3.37x` temporal motion (`temp_l1 3.328 -> 11.207`), `+10.88 dB` vocal/room gain (`-25.25 -> -14.37 dBFS`), `+17.43 st` median `YIN` `F0` shift (`157.91 -> 432.1 Hz`, voiced fraction `0.2412 -> 0.5867`), wide-eyed shock expressions, live `1x audio_only + block_band="late_30_49"` (`+1.29 dB`, `+1.995 st` global / `+3.446 st` paired `YIN` `F0` with calm coats preserved), and zero-bleed `f0..f24` `2x` Spatial Director portal activation. *Honest observation:* `1x` full passes also style stairwell cinderblock walls into graffiti murals and streetwear fur/bomber jackets, whereas the `2x` portal keeps baseline walls clean until the ellipse ramps open.
2. **`sudden_takeoff_flight` (`PROMOTED` — `fl2va`, `ref2va`, `h3_image`):** Authored with `N = 8` adult ground-motion scenes (`0` lint warnings, `first_diff_tok = 35..42`, median `25`) and paired against the new clean adult prompt [`prompt_bank/t2va_motorcycle_highway.txt`](../../prompt_bank/t2va_motorcycle_highway.txt). `1x` steering (`s = 1.50`) launched the grounded motorcycle vertically skyward into open canyon air (`f96..f191`) while raising spectral centroid `+1359.9 Hz` (`925.0 -> 2284.9 Hz`, `rgb_mae = 74.028`). `2x` rising/expanding HUD portals (`rx: 0.35 -> 0.58`, `f48 -> f168`) preserved identical grounded highway riding through `f24` before lifting the rider off the bars (`f72`) and shooting vertically skyward (`f96..f191`); adding `MiniMaxH3SoundCues` (`48: impact @1.1\n51: snap @0.9` + `MiniMaxH3ReleaseReferences audio_end_percent=0.60`) lifted overall audio `+1.89 dB` (`-12.87 -> -10.98 dBFS`) and half-2 airborne wind/roar `+9.39 dB` (`-21.77 -> -12.38 dBFS`). *Honest observation:* the `2x` portal at `scale = 2.2` adds a stylized sky-beam/planet portal glow above the horizon compared to the naturalistic dust-plume launch of the `1x` render.

---

## 1. Promoted Build & J-Lens Metrics (`diff_span`, `r = 8`, `N = 8` Pairs)

| Stem | Family | `raw_norm` | `mpc` | `split_half` | `1D_E` / `\|mu_d\|/E\|d\|` | Top `j_lens (+)` / `(-)` | Built File |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| `youtube_overhype_influencer` | `fl2va` | `73.83` | `0.7393` | `0.9213` | `75.8%` / `0.872` | `explosive, erupt, frantic` / `quiet, gentle` | `youtube_overhype_influencer__fl2va__diff_span__lora_2a8c2b52.safetensors` |
| `youtube_overhype_influencer` | `ref2va` | `70.38` | `0.7262` | `0.9167` | `74.5%` / `0.864` | `explosive, erupt, frantic` / `gentle, quiet` | `youtube_overhype_influencer__ref2va__diff_span__lora_865ee189.safetensors` |
| `youtube_overhype_influencer` | `h3_image` | `71.01` | `0.7326` | `0.9195` | `78.8%` / `0.890` | `erupt, explosive, chaotic` / `quiet, gentle` | `youtube_overhype_influencer__h3_image__diff_span.safetensors` |
| `sudden_takeoff_flight` | `fl2va` | `69.81` | `0.6593` | `0.8880` | `68.2%` / `0.828` | `aerial, futuristic, explosive` / `smooth, walking, steady` | `sudden_takeoff_flight__fl2va__diff_span__lora_2a8c2b52.safetensors` |
| `sudden_takeoff_flight` | `ref2va` | `67.16` | `0.6523` | `0.8852` | `67.5%` / `0.824` | `futuristic, aerial, explosive` / `smooth, walking, steady` | `sudden_takeoff_flight__ref2va__diff_span__lora_865ee189.safetensors` |
| `sudden_takeoff_flight` | `h3_image` | `70.33` | `0.6608` | `0.8879` | `67.0%` / `0.820` | `aerial, explosive, dramatic` / `steady, flat, walking` | `sudden_takeoff_flight__h3_image__diff_span.safetensors` |

*(Advisory `[j_lens eyeball note]` on `youtube_overhype_influencer__h3_image`: only `1/12` `b_side` tokens matched `top_k` due to generic `"hand"` across `a_side`/`b_side` metadata descriptions; GPU `1x` & `2x` renders verified strong expression, gesture, and vocal transfer.)*

---

## 2. Serialized `:8188` GPU Results (`1152x768 @ 192f`, `seed = 730451892`)

Contact sheets: `$H3_OUTPUT_DIR/Video/wave3b_2026_11_23/sheets/{sheet_overhype,sheet_takeoff}.png`.

| Suite / Arm | Config (`streams`, `block_band`, `scale`) | `hsv_sat` | `luma` | `temp_l1` | `rgb_mae` (`h1 / h2`) | `audio_dBFS` (`h1 / h2`) | `centroid_hz` | `YIN` `F0` Δ (`global / paired`) | `exec_s` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `overhype_base_192f` | Unsteered (`stairwell`) | `0.3050` | `103.93` | `3.328` | `0.000` | `-25.25` (`-24.35 / -26.40`) | `2505.2` | `157.91 Hz` ref | `82.90 s` |
| `overhype_full_both_192f` | `1x`, `both`, `all`, `s=1.45` | `0.4304` | `94.97` | **`11.207`** | `48.931` (`46.994 / 50.868`) | **`-14.37`** (`-14.11 / -14.65`) | `2056.1` | **`+17.427 / +19.312 st`** (`vf=0.587`) | `72.27 s` |
| `overhype_vidonly_1x_192f` | `1x`, `video_only`, `all`, `s=1.45` | `0.4493` | `95.17` | `11.438` | `49.418` (`46.726 / 52.109`) | `-19.31` (`-18.96 / -19.69`) | `2327.7` | `+5.235 / +5.215 st` (`vf=0.330`) | `76.20 s` |
| `overhype_audonly_late30_192f` | `1x`, `audio_only`, `late_30_49`, `s=1.45` | `0.2943` | `106.56` | `3.705` | `16.004` (`18.354 / 13.654`) | **`-23.96`** (`-23.88 / -24.04`) | `2509.2` | `+1.995 / +3.446 st` (`vf=0.293`) | `75.16 s` |
| `overhype_hud_portal_2x_192f` | `2x` portal (`eg=2.0, aud=1.35, s=1.55`) | `0.4319` | `105.88` | `10.768` | `42.460` (`37.899 / 47.021`) | **`-13.67`** (`-13.08 / -14.36`) | `2618.3` | **`+16.555 / +18.585 st`** (`vf=0.489`) | `151.63 s` |
| `moto_base_192f` | Unsteered (`t2va_motorcycle_highway.txt`) | `0.3279` | `106.33` | `12.578` | `0.000` | `-13.04` (`-14.29 / -12.08`) | `925.0` | `81.15 Hz` ref (`vf=0.194`) | `78.57 s` |
| `moto_takeoff_full_1x_192f` | `1x`, `both`, `all`, `s=1.50` | `0.3432` | `117.38` | `11.017` | `74.028` (`71.397 / 76.660`) | `-15.62` (`-13.14 / -22.01`) | **`2284.9`** | `+4.228 / +4.187 st` (`vf=0.018`) | `72.14 s` |
| `moto_takeoff_hud_portal_2x_192f` | `2x` rising portal (`eg=2.2, aud=1.2, s=1.60`) | `0.3914` | `121.77` | `15.994` | `65.070` (`49.664 / 80.476`) | `-12.87` (`-10.15 / -21.77`) | `1920.1` | `-2.131 / n/a st` (`vf=0.008`) | `151.95 s` |
| `moto_takeoff_hud_portal_soundcue_2x_192f` | `+ SoundCues (48: impact@1.1, 51: snap@0.9)`, `audio_end=0.60` | `0.3964` | `122.26` | **`16.182`** | `65.287` (`50.566 / 80.008`) | **`-10.98`** (`-9.93 /` **`-12.38`**) | `1774.1` | `-1.894 / +2.671 st` (`vf=0.016`) | `145.82 s` |
