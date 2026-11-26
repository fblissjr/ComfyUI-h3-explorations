lane: steering
verdict: verified

# Spatial Query Routing `v2-slim` & Beat 1–3 Showcase Suites (`2026-11-26`)

Verification and diverse-scene stress record for [`steering/spatial_query_router.py`](../../steering/spatial_query_router.py) (`Spatial QR v2-slim`: `0d3398af`, `a31fb461`), `H3 Studio` `v2-slim` + live preview telemetry (`app/` `6dd8ab3`), and the `22` live GPU showcases across Beat 1 Diverse (`6`), Beat 2 `v2-slim` Validation (`8`), and Beat 3 Breakthrough Showcases (`8`), companion to the 4-Sensor Silicon Calibration in [`bench/results/2026-11-26_h3_4sensor_calibration.md`](2026-11-26_h3_4sensor_calibration.md).

All renders ran on Pika H3 Flex 6-step (`ManualSigmas`, `seed = 730451892` unless noted) or `MiniMax-H3-Image` (`6` steps, `euler`, `generate_megapixels = 4.0`, `seed = 42` / `730451892`). Every clip and still was inspected visually via native `read_file` before writing metrics.

---

## 1. Beat 1 — Diverse Scenes (`b510ba6b`, pre-`v2-slim` Baseline & Diagnostic Finds)

Outputs under `$H3_OUTPUT_DIR/Video/beat1_diverse_scenes_2026-11-26/` (`seed = 730451892`):

| Clip | Exec | Resolution / Length | Config | Visual & Audio Finding (Native Inspection + CPU Metrics) |
| :--- | :-: | :-: | :--- | :--- |
| `b1_141f_cooking_claymation_lens_00001-audio.mp4` | `81.77 s` | `1344x768`, `141f` | `Steer(claymation_stopmotion, +1.5)` + `SpatialQR(same_prompt, invert=True, audio_zone="region")` | **Strong 1.1x Vector Reality Lens.** Steered half shifts toward sculpted clay (`sat 0.249` vs `0.331`, `sharp 20.1` vs `26.1`) while unsteered half and TV-studio soundtrack (`-26.03 dBFS`) hold live-action texture. |
| `b1_192f_stairwell_ferry_binaural_00001-audio.mp4` | `125.01 s` | `1344x768`, `192f` | `early_0_24` + `audio_zone="binaural_split"` | **Decorrelated stereo (`lr_corr = 0.071`, `8.38 dB` L/R spectral divergence), quiet mix (`-34.48 dBFS`), & visual leak diagnostic.** Pre-`v2-slim`, left stairwell speaker (`woman_wool`) took a foggy ferry backdrop and right speaker bled visual cues cross-ear. **Led directly to `v2-slim` Visual-to-Ear Firewall.** |
| `b1_192f_silent_to_rock_all_0_49_00001-audio.mp4` | `111.72 s` | `1344x768`, `192f` | `[0..84]` Silent Film -> `[85..191]` Arena Rock, `all_0_49`, `audio_zone="base"` | **Razor-sharp 1-frame visual cut (`f84 -> f85` `mae_prev = 89.42`), audio pinned to solo piano (`audio_h2 = -29.60 dBFS`).** Motivated adding `audio_zone="follow_mask"`. |
| `b1_192f_silent_to_rock_early_0_24_00001-audio.mp4` | `121.27 s` | `1344x768`, `192f` | Same step cut, `early_0_24` | Pre-cut transition (`f84` `mae_prev = 11.5`) and minor neon bleed confirming `all_0_49` is strictly cleaner for temporal hard cuts (`-9.55 s` faster too). |
| `b1_192f_silent_to_rock_expand_wipe_00001-audio.mp4` | `117.28 s` | `1344x768`, `192f` | Monotonic expanding portal `f48..f191`, `early_0_24` | Smooth zero-cut atmospheric transition (`cut_max_mae = 1.34`, `shot1_sat = 0.135 -> shot2_sat = 0.337`) vs hard jump-cut. |
| `b1_192f_silent_to_rock_plain_ctrl_00001-audio.mp4` | `94.74 s` | `1344x768`, `192f` | Plain `BasicGuider` `[Shot 1]..[Shot 2]` | Plain control held sepia/dim blend across `f85` (`cut_max_mae_f83_90 = 2.01`, failed to snap cleanly on schedule). |

---

## 2. Beat 2 — `Spatial QR v2-slim` GPU Validation (`0d3398af` / `a31fb461`)

Outputs under `$H3_OUTPUT_DIR/Video/beat2_v2_2026-11-26/`:

| Output | Exec | What It Tested | Observed Verdict |
| :--- | :-: | :--- | :--- |
| **`b2_h3_image_t2i_clockmaker_split_00001_.png`** (`2368x1760`) | `33.68 s` | Unlocked `MiniMax-H3-Image` `latent_t=1` `t2i` split (`early_0_24`) | **Showstopper.** Left warm brass workbench (`luma = 107.13`, `R/B = 1.324`, `R-B = +29.84`) merges organically into right cybernetic neon workbench (`luma = 82.94`, `R/B = 0.653`, `R-B = -38.70`); **`seam_1px_col_mae = 4.45` (`0.74x` natural background `6.01`)**. |
| `b2_h3_image_dual_ref_oldman_woman_00001_.png` (`2368x1760`) | `41.99 s` | `<Picture 1>` + `<Picture 2>` edit split (`early_0_24`, generic prompt framing) | **Diagnostic finding:** Left/right environments split cleanly (`R-B = +13.61` vs `-18.97`, `seam_1px_col_mae = 4.50`), but omitting `"on the left/right half"` in prompt prose caused `<Subject 1>` (`old_man.jpg`) to center across the seam and omit `<Subject 2>`. Fixed in Beat 3. |
| `b2_141f_luthier_beam_mid2049_00001-audio.mp4` vs `..._mid2034_00001-audio.mp4` | `78.23 s` vs `71.08 s` | `"mid_20_49"` strict late-block isolation vs `"mid_20_34"` beam `[0.28..0.62]` | **Sharp behavioral split:** `mid_20_49` confines claymation texture tightly inside `[0.28, 0.62]` (`f100..130` `in_lap_var = 143.92` vs `out = 59.79`, `+26.49` luma diff), whereas `mid_20_34` (`35..49` global K/V) morphs the whole actress + shop into unified claymation (`in_lap_var = 221.0`). |
| **`b2_141f_sam3_chrome_mid2034_shield_00001-audio.mp4`** | `153.91 s` | `10%` `SAM 3.1` dancer mask under `"mid_20_34_shield"` (`protect` background via asymmetric K/V) | **Broke the `10%` small-mask barrier in one `1x` pass.** White-brick loft pixels stay locked (`bg_margin_mae_mean = 0.45`, `median = 0.32`, `max = 1.44`) while the dancer renders as mirror-polished liquid chrome (`center_sat_all = 0.175`, brief 1-leg trouser flicker around `00:01`). |
| `b2_90f_vector_lens_charcoal_shield_00001-audio.mp4` | `49.88 s` | `Steer(charcoal_sketch_animated)` + `mid_20_34_shield` with `invert=True` | **Polarity lesson:** `*_shield` shields `region` while letting `base` read `region` K/V (`box_luma_in = 138.11`, `out = 108.62`); combining `invert=True` put `Steer` on `base` (`30%` center box) instead of `region`, washing back to photoreal (`border_drift_mae = 4.13..5.65`). |
| `b2_141f_tenenbaum_mirror_seance_00001-audio.mp4` | `90.53 s` | `i2va` Keyframe border (`tenenbaum_mirror.jpg`) + `[0.25..0.75]` séance + `MiniMaxH3SoundCues("0.5s: slam")` | **Keyframe audio-visual lock early, multi-shot cut later:** `0.5s: slam` lands cleanly at **`0.51 s` (`-13.34 dBFS`, `+52.5 dB` over `0.2..1.2 s` median `-65.81 dBFS`)**, seamless mirror apparition `00:00–00:02.8`, then `t2va_victorian_seance.txt` `[Shot 2]` cuts the border away (`border_drift_post_f68 = 79.07`). Fixed in Beat 3 with a single-shot prompt. |
| **`b2_192f_silent_to_rock_follow_mask_00001-audio.mp4`** | `116.21 s` | `192f` Silent Film -> Arena Rock cut at `f85` with **`audio_zone="follow_mask"`** | **100/10 AV Multi-Shot Cut.** `f85` video cut (`mae_prev = 88.61`, `shot1_sat = 0.205 -> shot2_sat = 0.285`) triggers a **`+10.56 dB` audio jump (`-29.48 -> -18.92 dBFS`)** and **`+410.7 Hz` spectral centroid surge (`550.4 -> 961.1 Hz`)** cleanly from solo piano into belted arena rock vocals + pyrotechnics (vs Beat 1 `audio_zone="base"` `+4.96 dB` / `+51.4 Hz`). |

---

## 3. Beat 3 — Sensor-Driven & Zero-Code Showcases (Completed after 4-Sensor Gate)

Outputs under `$H3_OUTPUT_DIR/Video/beat3_showcases_2026-11-26/`:

| Output | Exec | Innovation Demonstrated | Observable Result |
| :--- | :-: | :--- | :--- |
| **`b3_h3_image_step0_self_mask_bread_crystal_00001_.png`** (`2368x1760`, `seed=42`) | `17.13 s` | **Zero-Coordinate, Zero-SAM Self-Masked Routing:** Step-0 `B35+B40` `Q_vid @ K_txt["bread"]^T` (`$H3_INPUT_DIR/sensor_step0_bread_mask.png`, `11.15% >= 0.35`, `71.8% <= 0.05`) + `mid_20_34_shield` | **10/10 target surgical material swap.** Loaves inside the self-extracted mask turn into glowing translucent sapphire crystal (`inside_mae_rgb = 65.48` vs `outside_mae_rgb = 15.23`, `4.30x`; `B-R` flips **`-60.75 -> +44.96`**) while the woven basket, grapes, carrots, cheeses, potatoes, and olive oil stay matched to `h3_image_table_still.png`. |
| **`b3_h3_image_dual_ref_lr_e024_00001_.png`** vs **`..._all049_00001_.png`** (`2368x1760`) | `41.73 s` vs `22.49 s` | Explicit `"on the left/right half"` spatial framing fixing Beat 2's dual-ref omission (`<Picture 1>` `old_man.jpg` L \| `<Picture 2>` `woman_blonde.jpg` R) | **100% dual-identity fix, with crisp `early_0_24` vs `all_0_49` seam lesson:** Both place `<Subject 1>` cleanly on Mediterranean terrace (L, `R-B = +32.66 / +25.26`) and `<Subject 2>` in midnight neon Tokyo (R, `R-B = -47.23 / -45.83`). **`early_0_24` produces a `3.1x` smoother center join (`seam_1px_mae = 4.49`, `1.69x` bg)** than `all_0_49` (`seam_1px_mae = 26.28`, `5.21x` hard edge). |
| `b3_90f_vector_lens_claymation_portal_00001-audio.mp4` | `52.18 s` | `Steer(claymation_stopmotion, +1.6)` + `invert=True` + `[0.25..0.75]` portal box (`all_0_49`) on wide cobble street | Warmer/stylized interior tint on `[0.25..0.75]` alley (`tactile shift milder` at medium-wide distance than close-up cooking studio). |
| `b3_141f_tenenbaum_mirror_locked_00001-audio.mp4` | `83.08 s` | Single-shot séance prompt replacing `[Shot 2]` + `tenenbaum_mirror.jpg` border lock (`mid_20_49`, `0.5s: slam`) | **Locked bathroom border (`141/141` frames):** `border_drift_post_f68_mae` dropped **`79.07 -> 3.18`** (`center_motion_post_f68_mae = 10.64`) with crisp `0.5s: slam` knock + séance speech. Honest nuance: strict `mid_20_49` at `x=0.25..0.75` etched an inner rectangular sub-border inside the real gilt mirror (`early_0_24` blends cleanly when box bounds don't hug the exact mirror frame). |
| **`b3_192f_stairwell_ferry_binaural_v2_00001-audio.mp4`** | `124.52 s` | Post-Visual-to-Ear-Firewall `192f` `binaural_split` (`early_0_24`) w/ synchronized `[Shot 1]` 4-turn dialogue | **10/10 Split-World Stereo Dialogue:** Seamless wall/rail bulkhead separating sunlit stairwell woman (L) and foggy night harbor ferry man (R) with 4 crisp non-overlapping dialogue turns (`"You said tomorrow"` / `"It moved"` / `"To when?"` / `"Tonight"`, `lr_corr = 0.1406`, `L = -33.41 dBFS`, `R = -34.75 dBFS`). |
| **`b3_141f_midas_walk_kinematic_wake_00001-audio.mp4`** | `110.52 s` | **"Midas Walk" Kinematic Wake (`1x` `ref2va`):** `<Picture 1>` tortoise + `SpriteTrajectory` (`<Video 1>` released `0.33`) + `SpatialDirector` trailing vertical split (`x = 0.04 -> 0.92`, `feather = 0.08`) + `SpatialQR(mid_20_34, follow_mask)` + `SoundCues` | **100/10 showstopper in ONE `1x` pass.** As the tortoise walks steadily left-to-right with synced thuds (`0.5..4.5 s`), frozen winter cloister flagstones burst into lush emerald moss, golden sunbeams, buttercups, lupines, and roses strictly behind its wake (**`left_zone_sat` doubles `0.203 -> 0.433`** while unvisited right zone remains frosted `0.156`). Shipped as [`workflows/h3_r2v_midas_walk_spatial_wake_pika_api.json`](../../workflows/h3_r2v_midas_walk_spatial_wake_pika_api.json). |
| **`b3_141f_moge2_depth_wave_00001-audio.mp4`** | `93.92 s` | **`MoGe-2` 3D Metric Depth Wave (`i2va`):** Metric depth (`p02 = 50.6 m`, `median = 184.0 m`, `p98 = 722.6 m`) converted into a advancing `[141, 768, 1344]` sweep mask (`$H3_INPUT_DIR/moge2_alpine_depth_wave_141f.mp4`) + `SpatialQR(mid_20_34, follow_mask)` | **100/10 depth-aware 3D transformation.** Near foreground pines freeze snow-white by `f35..f54` (**`fg B-R = +12.75`**) while `500..720 m` sunlit peaks hold warm golden hour (**`far B-R = -10.52`**), before the glacier wave + green-cyan aurora reaches the summit (`late_f110_140` **`B-R = +36.34`**) and `follow_mask` shifts the soundscape from warm pastoral guitar (`-16.92 dBFS`, `722.2 Hz`) to rushing crystalline sub-bass aurora (`-13.23 dBFS`, `1061.6 Hz`). |

---

## 4. Shipped Curated Workflows (`workflows/build_workflows.py`)

Promoted the verified Beat 2–3 mechanisms into zero-manual-wiring shipped workflows (in addition to `h3_text_to_video_pika_spatial_query_router_api.json`):
1. [`workflows/h3_text_to_video_pika_multishot_spatial_cut_api.json`](../../workflows/h3_text_to_video_pika_multishot_spatial_cut_api.json) — `192f` `all_0_49 + follow_mask` 1-frame multi-shot hard cut with procedural `SolidMask` step mask.
2. [`workflows/h3_text_to_video_pika_binaural_dialogue_split_api.json`](../../workflows/h3_text_to_video_pika_binaural_dialogue_split_api.json) — `192f` `binaural_split` synchronized stairwell \| harbor ferry stereo dialogue split.
3. [`workflows/h3_text_to_video_pika_vector_lens_claymation_api.json`](../../workflows/h3_text_to_video_pika_vector_lens_claymation_api.json) — `141f` single-prompt `MiniMaxH3Steer(claymation_stopmotion)` + `MiniMaxH3SpatialQueryRouter(invert=True, audio_zone="region")` Vector Reality Lens.
4. [`workflows/h3_r2v_midas_walk_spatial_wake_pika_api.json`](../../workflows/h3_r2v_midas_walk_spatial_wake_pika_api.json) — `141f` Midas Walk kinematic wake combining `<Picture 1>`, `<Video 1>` sprite release (`0.33`), `SoundCues`, `MiniMaxH3SpatialDirector`, and `MiniMaxH3SpatialQueryRouter(mid_20_34, follow_mask)`.
5. [`workflows/image/h3_image_t2i_spatial_query_router_api.json`](../../workflows/image/h3_image_t2i_spatial_query_router_api.json) — `MiniMax-H3-Image` `t2i` (`16 -> 4 MP`) seamless split-world clockmaker (`early_0_24`).
