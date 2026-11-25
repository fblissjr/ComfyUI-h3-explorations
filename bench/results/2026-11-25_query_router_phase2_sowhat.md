# Query router Phase 2 gates, six bug fixes, & "so what?" generalization sweep (`2026-11-25`)

lane: query routing
verdict: Late-block intra-audio cross-masking (`audio_block_band="late_30_49"`) unlocks zero-echo `1x` binaural split across two scenes (`stairwell` `pic_mae=7.926`, `lr_corr=0.1122`; `t2va_radio_drama` `pic_mae=16.014`, `lr_corr=0.1167`, balanced `L=-13.68 / R=-14.35 dBFS` across 3 speakers). Six router bugs were fixed and verified (`9e5db27c`, `3bb7a3c9`, `13/13` tests): closing the 1-hop `K_vid -> shadow_text -> Q_aud` leak rescued `aud_to_vid_gain=0.0` (`+9.43 dB` over `g=1.0`, `+9.63 dB` over pre-fix `g=0.0`, restoring `-22.85 dBFS` dry close-mic stairwell speech over submerged underwater bubbles when `prompt_aud` has zero conflicting visual words), and multi-run `SEG_TAG` `_tag_runs` unlocked ensemble `ref2va` (`ref2va_scene_kitchen` w/ `woman_blonde.jpg`: `plain_ja=40.033` recast all unpinned cooks & erased `"FIRE"` tickets vs `router_ja=28.552` locking cast, dishes, and tickets). Across `18` Phase 2 renders plus Gates 1b/2/1b-probes, Cross-Modal Query Routing's payoff follows an asymmetric Cross-Modal Prior Conflict Law: massive `1x` wins on binaural splits, soundscape-to-architecture leaks (`cooking_zen` brick studio vs plain garden windows), unpinned `EN -> JA` dubs (`stairwell`, `police_interrogation`, `ref2va_scene_kitchen`), and clean-text `aud_to_vid_gain=0.0` physical decouples, versus honest ties where plain prompting already succeeds (`samurai` `JA -> EN`, `delta_blues` `EN -> ES`, tight single-headshot `ref2va`, inner-monologue voiceover, and full-prompt `ref2va` motion-transfer restyle).

**Date:** `2026-11-25`
**Code & shipped graphs:** [`steering/query_router.py`](../../steering/query_router.py), [`steering/hooks.py`](../../steering/hooks.py), [`steering/nodes.py`](../../steering/nodes.py), [`bench/check_query_router.py`](../../bench/check_query_router.py) (`9e5db27c`, `3bb7a3c9`); shipped `fl2va`/`ref2va` query-router workflows and zero-visual-edit `heylook` diff presets (`e4417d17`); Studio 2.0 integration (`app/.git` `5141838`).
**Media folders:** `$H3_OUTPUT_DIR/Video/query_router_crossmask_gate_2026-11-25/`, `$H3_OUTPUT_DIR/Video/query_router_phase1_2026-11-25/`, `$H3_OUTPUT_DIR/Video/query_router_phase1b_2026-11-25/`, `$H3_OUTPUT_DIR/Video/query_router_phase2a_2026-11-25/`, and `$H3_OUTPUT_DIR/Video/query_router_phase2b_2026-11-25/`.
**Setup:** Pika H3 Flex `6`-step `lcm`, `1344x768`, seed `730451892`, `MiniMaxH3Sol(dense_blocks="49", quantizer="rotated")`, `1x` `BasicGuider`. Every rendered `.mp4` (and split mono Left/Right `.wav` on every binaural clip) was directly watched and listened to before scoring.

---

## Part A — Gate 1 & Gate 1b: Intra-Audio Block-Diagonal Cross-Masking (`blocks.0..49` vs `late_30_49`)

Yesterday's `f0918dac` added block-diagonal intra-audio cross-masking (`L <-> R` or `H1 <-> H2`) in [`steering/hooks.py`](../../steering/hooks.py). Live GPU testing on `stairwell` (`192` frames, `prompt_aud` = thunderstorm) exposed a sharp depth split:

| Arm | `audio_block_band` | Pic MAE vs Base | RMS (dBFS) | L / R (dBFS) | `lr_corr` | Observed Audio & Visual Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `stairwell_binaural_right_storm_v2` | `all` (`0..49`) | `12.668` | `-21.04` | `-22.03 / -20.23` | `-0.0018` | Blocked rain bleed (`L` dry, `R` storm), **but** desynced L/R speech timing (double-voice echo) |
| `stairwell_temporal_half2_storm_v2` | `all` (`0..49`) | `19.044` | `-21.99` | `-21.98 / -22.00` | `0.9945` | Isolated storm to H2, **but** H1 crammed all 8 lines into `0..4s` and H2 restarted from line 1 |
| `stairwell_binaural_right_storm_midlate20` | `mid_late_20_49` | `11.131` | `-22.65` | `-22.58 / -22.71` | `0.0443` | Reduced desync vs `all`, slight timing wobble remained |
| **`stairwell_binaural_right_storm_late30`** | **`late_30_49`** | **`7.926`** | **`-23.21`** | **`-22.59 / -23.93`** | **`0.1122`** | **Clean breakthrough:** `blocks.0..29` lock L/R speech phase & lip-sync; `30..49` keep `L` bone-dry and `R` storm-heavy with **zero echo** and tighter visual lock than `8.41` all-blocks identity |
| `stairwell_temporal_half2_storm_late30` | `late_30_49` | `15.330` | `-24.46` | `-24.46 / -24.46` | `0.9966` | Eliminated H1/H2 dialogue restart; clean `0..4s` dry -> `4..8s` storm switch, with minor 1-phrase stutter where `t=0.5` cut mid-sentence (*"Who else knows?"*) |
| `stairwell_temporal_half2_storm_midlate20` | `mid_late_20_49` | `13.819` | `-23.92` | `-23.94 / -23.90` | `0.9964` | Similar to `late_30_49`, slightly noisier seam |

---

## Part B — Gate 2 & Phase 1b Test-Before-Code Probes (What Earned Code vs. What Got Cut)

Before writing new kernel paths, five targeted live probes tested whether proposed features were actually needed beyond plain prompting:

| Probe | Scene / Mode | Metrics | Observable Verdict & Decision |
| :--- | :--- | :--- | :--- |
| **Gate 2a — Second T2V `EN -> JA` dub** | `t2va_police_interrogation` (`192f`) | `plain_ja` `54.035` vs `router_ja` `21.675` MAE (`plain_vs_router=52.102`; spec `28.51` vs `21.171` dB; RMS `base=-28.75`, `plain=-18.40`, `router=-30.95`) | **Huge `1x` win (`2/2` T2V scenes):** Plain JA turned a moody American crime precinct into a bright Japanese TV drama and recast all 3 actors; router held the precinct and cast cleanly. |
| **Gate 2b — Tight single-headshot `ref2va` `EN -> JA`** | `t2i_29_edit_reference_portrait_seed7.png` (`141f`) | `plain_ja` `27.772` vs `router_ja` `11.476` MAE (spec `17.441` vs `17.534` dB; RMS `-16.74` / `-15.59` / `-14.67`) | **Honest neutral:** When `<Picture 1>` fills a tight single-subject portrait frame, the image reference itself blocks ethnicity/wardrobe drift in `plain_ja`. Motivated testing *ensemble* `ref2va` with unpinned co-stars in Batch 2B. |
| **Gate 2c — Plain `ref2va` Dual-Lock falsification** | `ref2va_pika_motion_transfer` -> `V5_neon_wet_midnight` (`192f`, plain prompt, NO router, NO `ReleaseReferences`) | `rms = -13.98 dBFS` | **Cut `K`-logit `ref_img` masking:** Plain full-prompt rewrite cleanly shattered flat pop-art Dual-Lock into photoreal wet cobalt-midnight neon while keeping `<Picture 1>` Desmond, `<Picture 2>` record shop, and `<Video 1>` dance. Dual-Lock was a *control-vector* ceiling, not a prompt problem. |
| **Phase 1b — Inner monologue / voiceover** | `voiceover_plain_en` vs `voiceover_router_en` (`141f`) | `pic_mae = 28.561`, RMS `-29.38` vs `-28.83 dBFS` | **Cut `vid_to_aud_gain`:** Plain prompting already kept the woman's mouth 100% sealed at a rain-streaked window while delivering crisp off-screen narration. |
| **Phase 1b — Arbitrary-length `video_block_band="mid_late_20_49"`** | `stairwell` -> equal-token-length `Blueprint Ink` (`192f`) | `mid_late_20_49` `81.218` vs `all` `88.261` MAE (`mid_vs_all = 53.21`) | **Cut arbitrary-length `video_block_band`:** Mid-late routing failed to lock Shot 1 blocking/actors and yielded a semi-photoreal/ink hybrid. |

---

## Part C — Six Verified Query-Router Bug Fixes (`9e5db27c`, `3bb7a3c9`)

Direct code audit of [`steering/hooks.py`](../../steering/hooks.py), [`steering/query_router.py`](../../steering/query_router.py), and [`steering/nodes.py`](../../steering/nodes.py) surfaced six bugs, all fixed and covered by `13/13` tests in [`bench/check_query_router.py`](../../bench/check_query_router.py):

1. **1-hop `K_vid -> shadow_text -> Q_aud` leak (`9e5db27c`):** When `aud_to_vid_gain < 1.0`, `bufs.q_a`/`bufs.q_b` previously ran with `attn_mask=None`, letting shadow text read `K_vid` and pass visual acoustics into `Q_aud` on the next block. Fixed via a dedicated `txt_vid_bias` masking `vid_spans` only (never `au_comp_spans`).
2. **Unequal-length `ref2va`/`fl2va` multi-run `SEG_TAG` collapse (`9e5db27c`):** Shadow text of unequal length previously collapsed `mod_txt_a` onto `mod_txt_main[0][2]` (`tag=0` vision AdaLN). Fixed by carrying `shadow_a_tags`/`shadow_b_tags` and rebuilding `_tag_runs`.
3. **Partial `audio_block_band` + `section_split="sections_by_modality"` (`9e5db27c`):** Blocks outside `aud_band` previously fed `Q_aud` soundscape-stripped `ctx_vid`. Fixed via `aud_outside_uses_shadow_a`.
4. **Disarmed broken in-place `V[ref_*]` mutations (`9e5db27c`):** Safely retired `ref_img_gain` and `ref_audio_stream` mutations (`v_bh` in-place edits that corrupted reference self-attention in Step 2) while preserving schema widget slots (`AGENTS.md` Rule 5).
5. **`2x` `EditGuider` reference cache thrash (`9e5db27c`):** Raised `_REF_CACHE_MAX` from `1` to `4` so dual-pass `original`/`edited` steps stop evicting each other.
6. **Silent nested `1x` `_h3_steer_runtime` clobber guard (`3bb7a3c9`, `Test 13`):** Raises a clear `ValueError` if hook-mode `MiniMaxH3Steer` (`streams != "both"` or `block_band != "all"`) stacks with `MiniMaxH3QueryRouter`, while permitting plain `MiniMaxH3Steer(streams="both", block_band="all")` + `MiniMaxH3QueryRouter`.

---

## Part D — Phase 2 Batch A: Singing Dub, Reverse Dub, & Chaos-Visuals/Zen-Audio (`10` Renders, `3bb7a3c9`)

All `10` renders ran post-restart on `3bb7a3c9` (`192` frames, `1344x768`, seed `730451892`, `$H3_OUTPUT_DIR/Video/query_router_phase2a_2026-11-25/`):

| Arm | Scene | Pic MAE vs Base | RMS (dBFS) | `HF>2k` (`H2`) | Observed Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `p2_blues_base_en` | `t2va_delta_blues_porch` | `0.000` | `-27.44` | `0.0767` (`0.0559`) | English blues singing + steel resonator guitar + `"CLARKSVILLE 5 MILES"` tin sign |
| `p2_blues_plain_es` | `t2va_delta_blues_porch` | `35.266` | `-30.31` | `0.1239` (`0.1212`) | **Honest tie:** Plain Spanish sang cleanly over resonator guitar while keeping porch, elderly woman, singer, and `"CLARKSVILLE 5 MILES"` intact |
| `p2_blues_router_es` | `t2va_delta_blues_porch` | `31.948` *(vs plain `27.704`)* | `-26.98` | `0.1389` (`0.1337`) | On par with `plain_es` (most `31.95` MAE comes from syllable-duration cut timing via `Q_vid -> K_aud`) |
| `p2_samurai_base_ja` | `t2va_samurai_bamboo_duel` | `0.000` | `-24.38` | `0.1630` (`0.1707`) | Two Japanese samurai in bamboo mist + vertical `"関ヶ原街道"` kanji trail marker |
| `p2_samurai_plain_en` | `t2va_samurai_bamboo_duel` | `28.402` | `-23.41` | `0.0941` (`0.1610`) | **Honest tie (asymmetric prior proof):** Plain English kept both Japanese actors, armor, taiko/shakuhachi score, and crisp `"関ヶ原街道"` kanji intact |
| `p2_samurai_router_en` | `t2va_samurai_bamboo_duel` | `25.229` *(vs plain `25.993`)* | `-27.37` | `0.1115` (`0.1787`) | Matches `plain_en`; proves `JA -> EN` on period pieces has near-zero visual recasting prior |
| `p2_cooking_base` | `t2va_cooking_show_meltdown` | `0.000` | `-14.05` | `0.0738` (`0.0797`) | Windowless white-brick TV studio, copper pots, `"MORNING BITES"` neon, beetroot blender explosion |
| `p2_cooking_plain_zen` | `t2va_cooking_show_meltdown` | **`47.988`** | `-14.00` | `0.0333` (`0.0373`) | **Severe soundscape-to-architecture leak:** `"Zen tea pavilion"` in `overall_soundscape` ripped open sunlit French garden windows with green trees + pink floral vase behind the chef |
| **`p2_cooking_router_zen_g1`** | `t2va_cooking_show_meltdown` | **`39.503`** *(vs plain `46.530`)* | **`-16.89`** | **`0.0247` (`0.0298`)** | **10/10 win:** Kept windowless brick TV studio 100% locked while syncing serene solo cello + hushed meditative whisper to the purple beetroot explosion |
| `p2_cooking_router_zen_g0` | `t2va_cooking_show_meltdown` | `43.053` *(vs g1 `33.270`)* | `-15.16` | `0.0649` (`0.0709`) | **Mechanism catch:** Because `prompt_aud` still contained `"blender lid flies off like a rocket"`, `g=0.0` lost lip/event grounding (sharp `"HAAH!"` gasp doubling `t=3..4s` `HF>2k` `0.0935 -> 0.1889`) and leaked a green window back into Shots 2–3 via open `Q_vid -> K_aud` |

---

## Part E — Phase 2 Batch B: Leak-Free `a2v=0.0`, Ensemble `ref2va`, & `late_30_49` Binaural/Temporal (`8` Renders, `3bb7a3c9`)

All `8` renders ran on `3bb7a3c9` (`192` frames, `1344x768`, seed `730451892`, `$H3_OUTPUT_DIR/Video/query_router_phase2b_2026-11-25/`):

| Arm | Scene / Mode | Pic MAE vs Base | RMS (dBFS) | `L / R` (`lr_corr`) | `HF>2k` (`H1 / H2`) | Observed Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `prev_uwvid_unrouted` *(ref, 11-24)* | `stairwell` UW | `0.000` | `-32.14` | `-32.12 / -32.13` (`0.9935`) | `0.0346` (`0.0388 / 0.0310`) | Muffled submerged underwater speech |
| `prev_uwvid_plain_dry` *(ref, 11-24)* | `stairwell` UW+Dry plain | — | `-32.02` | `-31.99 / -31.99` (`0.9896`) | `0.0297` (`0.0340 / 0.0263`) | Plain dry soundscape still muffled underwater |
| `uwvid_dryaud_a2v_100` *(ref, 11-24)* | `stairwell` router `a2v=1.0` | — | `-32.28` | — | `0.0280` | `g=1.0` still leaked underwater muffling from `K_vid -> Q_aud` |
| `prev_uwvid_a2v_000_leaked` *(ref, 11-24)* | `stairwell` pre-`9e5db27c` `a2v=0.0` | — | `-32.48` | `-32.46 / -32.47` (`0.9961`) | `0.0311` (`0.0289 / 0.0329`) | Stayed muffled due to Bug #1 (`bufs.q_a` 1-hop `K_vid` leak) |
| **`p2b_uwvid_dryaud_a2v_000_post9e5`** | **`stairwell` post-`9e5db27c` `a2v=0.0`** | **`23.397`** *(vs prev UW)* | **`-22.85`** | **`-22.84 / -22.83` (`0.9947`)** | `0.0331` (`0.0313 / 0.0365`) | **`+9.43 dB` over `g=1.0` / `+9.63 dB` over pre-fix `g=0.0`:** Crystal-clear close-mic dry stairwell dialogue over 100% submerged turquoise caustics & rising bubbles (`prompt_aud` has zero underwater words) |
| `p2b_kitchen_ref_base_en` | `ref2va_scene_kitchen` (`woman_blonde.jpg`) | `0.000` | `-13.99` | `-13.97 / -13.99` (`0.9978`) | `0.1076` (`0.0999 / 0.1182`) | `<Subject 1>` blonde + Black pan-tossing cook + curly-haired flat-top cook + Western dishes + `"FIRE"` tickets |
| `p2b_kitchen_ref_plain_ja` | `ref2va_scene_kitchen` (`plain_ja`) | `40.033` | `-15.03` | `-15.02 / -15.03` (`0.9957`) | `0.1094` (`0.1400 / 0.0843`) | Kept `<Subject 1>` blonde **while recasting ALL unpinned cooks to East Asian**, swapping plated dishes, and erasing `"FIRE"` tickets |
| **`p2b_kitchen_ref_router_ja`** | **`ref2va_scene_kitchen` (`router_ja`)** | **`28.552`** *(vs plain `38.947`)* | `-15.27` | `-15.25 / -15.26` (`0.9958`) | `0.1221` (`0.1653 / 0.0887`) | **10/10 `ref2va` win & live `_tag_runs` proof:** Locked `<Subject 1>`, Black cook, curly-haired cook `(S2)`, Western plates, and `"FIRE"` tickets across all 4 shots with synced overlapping + sung Japanese & `<\|cutoff\|>` |
| `p2b_radio_base` | `t2va_radio_drama` | `0.000` | `-15.85` | `-15.86 / -15.74` (`0.9757`) | `0.0631` (`0.1122 / 0.0240`) | Deadpan 1948 padded broadcast booth, `"ON AIR"`, gravel foley, 3 actors + `"I can hear him breathing."` subtitle |
| `p2b_radio_plain_temporal_horror` | `t2va_radio_drama` (`plain`) | `41.702` | `-14.93` | `-14.81 / -14.70` (`0.9217`) | `0.0864` (`0.1628 / 0.0192`) | Scrambled all 4 actors/wardrobe/lighting AND barely moved Shot 2 horror audio (`H2 HF = 0.0192`) |
| `p2b_radio_router_temporal_horror_late30` | `t2va_radio_drama` (`t>=0.5`, `late_30_49`) | `21.927` *(vs plain `42.510`)* | `-14.17` | `-14.14 / -13.98` (`0.9518`) | `0.1150` (`0.1187 / 0.1121`) | Locked exact studio & cast (`21.93` vs `41.70`) and hit **`5.8x`** H2 horror HF (`0.1121` vs `0.0192`), **with honest boundary artifact:** 1-phrase `"put out the lamp"` repeat where `t=0.5` landed flush on the line |
| **`p2b_radio_router_binaural_horror_late30`** | **`t2va_radio_drama` (`right`, `late_30_49`)** | **`16.014`** *(vs plain `41.588`)* | `-16.54` | **`-13.68 / -14.35` (`0.1167`)** | `0.1085` (`0.2094 / 0.0340`) | **Headline binaural showcase:** Exact 1948 studio visuals; split mono ears confirm **Left = dry padded broadcast booth**, **Right = storm-lashed Gothic mansion + rain + horror strings**, with all 3 actors' voices sample-locked to lips and **zero echo** |

---

## Part F — Unified Law of Asymmetric Cross-Modal Prior Conflict

Across `18` Phase 2 renders plus Gates 1b/2/1b-probes (`2026-11-24`..`2026-11-25`), `1x` Cross-Modal Query Routing has a sharp, predictable value boundary:

1. **Where `1x` Query Routing delivers unmistakable wins over plain prompting:**
   - **Zero-echo `1x` Binaural Split (`audio_channel="right"`, `audio_block_band="late_30_49"`):** `2/2` scenes (`stairwell` `7.926` MAE / `lr_corr=0.1122`; `t2va_radio_drama` `16.014` MAE / `lr_corr=0.1167`), synthesizing two distinct acoustic worlds (room reverb, foley, score) with sample-locked voices in one pass.
   - **Soundscape-to-Architecture Leaks:** Any scene where `overall_soundscape` / `non_diegetic_music` invokes a contrasting environment (`cooking_zen` Zen pavilion windows vs indoor brick studio, `stairwell` sitcom/storm, `tortoise` cathedral).
   - **Asymmetric `EN -> JA` Dubbing with Unpinned Degrees of Freedom:** Modern unspecified T2V interiors (`stairwell`, `police_interrogation`) and ensemble `ref2va` scenes (`ref2va_scene_kitchen` — saves unpinned co-stars, plated dishes, and English tickets).
   - **Clean-Text Physical Decoupling (`aud_to_vid_gain=0.0`, post-`9e5db27c`):** Strips visual-physics-to-audio bleed (`+9.43 dB` dry speech over submerged underwater video) strictly when `prompt_aud` contains zero conflicting visual-action prose; keep default `1.0` otherwise.
2. **Where plain prompting already works (honest non-wins — do not oversell the router here):**
   - **`JA -> EN` reverse dubs on visually over-specified period pieces** (`samurai_bamboo_duel`), **heavily visual-anchored `EN -> ES`** (`delta_blues_porch`), **tight single-headshot `ref2va`** (`t2i_29`), **sealed-mouth voiceover**, and **full-prompt `ref2va` restyles** (`motion_transfer` -> wet midnight neon).
   - **Temporal audio switching (`audio_time_start=0.5`, `late_30_49`)** cleanly locks visuals and flips H2 acoustics (`5.8x` HF) but repeats a spoken phrase if `t0` bisects dialogue; place `t0` only on silence/music transitions.
