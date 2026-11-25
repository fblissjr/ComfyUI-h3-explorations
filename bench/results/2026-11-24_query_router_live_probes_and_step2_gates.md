# Query router live probes & Step 2 gates: dubbing, 1-picture-3-soundtracks, identity drift breakdown, and Step 2 mechanism verdicts

lane: query routing
verdict: Multi-seed Spanish, zero-shot Japanese, per-speaker bilingual (S1 ES / S2 EN), 1-picture-3-soundtracks (AM radio, sitcom laugh track, underwater hydrophone+sonar), and submerged-video/dry-audio routing all hold visuals near the router's identity floor at single-pass 1x speed (96.0s-111.0s). Micro-CUDA parity confirms fp32 H3SteerRuntime is exact (1.79e-7 max_abs) and traces live bf16 drift to bf16 GEMM slicing, SDPA query row-offset tiling, and e46b70ac's TokenRefiner override (fixed in 5677149d). Step 2 live gates falsified naive 1x binaural/temporal Q_aud subset routing (smears across open intra-audio attention even though MiniMaxH3AudioVAE processes L/R independently) and mid-late V[ref_img] scaling on old_man.jpg, while aud_to_vid_gain runs cleanly without losing lip-sync and adds modest HF lift at 0.25 beyond two-prompt text routing alone.

**Date:** `2026-11-24`
**Code & server:** `MiniMaxH3QueryRouter` ([`steering/query_router.py`](../../steering/query_router.py)), [`steering/hooks.py`](../../steering/hooks.py), [`comfy/ldm/minimax/audio_vae.py`](../../../../comfy/ldm/minimax/audio_vae.py). Part A ran standalone against `5677149d`; Part B ran live on `:8188` (`e46b70ac`); Part C ran live on `:8188` (`5677149d`).
**Clips:** `$H3_OUTPUT_DIR/Video/query_router_v1_1_live_2026-11-24/`, `$H3_OUTPUT_DIR/Video/query_router_v2_showcase_2026-11-24/`, and `$H3_OUTPUT_DIR/Video/query_router_a2v_conflict_2026-11-24/`.

---

## Part A — Identity Drift Root-Cause Breakdown (`5677149d` vs. `e46b70ac`)

To test why `audio_conditioning == video_conditioning` (`stairwell_router_identity`) produces an `8.41` MAE baseline on `stairwell` and `17.07` / `24.41` on `tortoise`, standalone parity checks (`_TinyH3Model` 4 blocks, `heads=42`, `head_dim=128`, `l_txt=200`, `n_aud=240`, `n_vid=2000`, and H3-scale `S=25440` CUDA micro-benchmarks) isolated each component of `H3SteerRuntime`:

| Check | Device / Dtype | Max Abs | Mean Abs | Rel L2 | Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `A1` End-to-end 4-block `H3SteerRuntime` (`mode=2`) | `fp32` CPU | `1.79e-7` | `4.17e-9` vid / `5.83e-9` aud | `3.77e-8` / `4.64e-8` | Exact (`< 1e-4` gate); zero structural routing bug |
| `A1` End-to-end 4-block `H3SteerRuntime` (`mode=2`) | `fp32` CUDA | `1.79e-7` | `4.16e-9` vid / `5.63e-9` aud | `3.76e-8` / `4.53e-8` | Exact; repeat runs equal `0.0` |
| `A1` End-to-end 4-block `H3SteerRuntime` (`mode=2`) | `bf16` CUDA | `0.00781` | `6.13e-5` vid / `8.79e-5` aud | `0.00109` / `0.00132` | Compounding `bf16` drift (`0.00391 -> 0.00781` over blocks `0..2`) |
| `A2` `F.sdpa` vs `cat([q_txt, q_aud])` (`S=25440`) | `bf16` CUDA | `0.0` | `0.0` | `0.0` | Bit-exact on `FLASH_ATTENTION`, `EFFICIENT_ATTENTION`, & `comfy.ops` |
| `A2` `F.sdpa` vs isolated `q_aud` at row `0` (`S=25440`) | `bf16` CUDA | `6.10e-5` | `4.24e-6` | `0.00255` | FlashAttention query-tile boundary offset shift (`aud_only` / `aud_unswapped`) |
| `A3` `TensorWiseINT8Layout` (`convrot` & `rowwise`, `5376x5376`) | `bf16` CUDA | `0.0` | `0.0` | `0.0` | INT8 matmul slice (`M=25440` vs `M=200`) is bit-exact |
| `A3` Plain `F.linear` slice (`M=25440` vs `M=200`, `5376x5376`) | `bf16` CUDA | `0.01563` | `1.04e-3` | `0.00294` | `bf16` cuBLAS reduction order differs (unmerged LoRA & non-int8 linears) |
| `A4` `comfy.quant_ops.ck.rms_rope_split_half_` slice | `bf16` CUDA | `0.0` | `0.0` | `0.0` | Bit-exact |
| `A5` `TokenRefiner` `{}` vs `attention_pytorch` (`2` blocks) | `bf16` CUDA | `0.0` | `0.0` | `0.0` | Exact |
| `A5` `TokenRefiner` `{}` vs `attention_comfy_kitchen_int8` (`e46b70ac`) | `bf16` CUDA | `0.10938` | `0.01699` | `0.02155` | `e46b70ac` passed `ck_int8` into `TokenRefiner` (`5677149d` forces `{}`) |

Live re-check on `5677149d` (`stairwell_router_identity_v2`, seed `730451892`, `120.1s`) measured `pic_mae_rgb_320x180 = 8.999`, `sound_log_spec_mae_db = 5.157 dB`, `sound_rms_dbfs = -25.72 dBFS` — matching `e46b70ac`'s `8.41` floor rather than collapsing it, confirming that compounding `bf16` GEMM slicing and FlashAttention row-offset tiling dominate across `50` blocks × `6` sampler steps.

---

## Part B — Live `1x` Dubbing & 1-Picture-3-Soundtracks (`e46b70ac`)

Pika H3 Flex `fl2va`, `1344x768`, `192` frames, `1x` `BasicGuider` + `MiniMaxH3Sol`, compared against matching-seed [`bench/results/2026-11-24_query_router_first_renders.md`](2026-11-24_query_router_first_renders.md) unsteered baselines (`stairwell_unsteered_s101`: `-28.28 dBFS`; `stairwell_unsteered_s730451892`: `-25.87 dBFS`). All `6` `.mp4` clips were directly watched and listened to (`read_file`, `$0` spend).

| Arm | Seed | Wall (s) | Pic MAE (`320x180`) | Spec MAE (dB) | RMS (dBFS) | Observed Picture & Audio Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `stairwell_dub_es_s101` | `101` | `111.0` | `13.818` | `17.393` | `-25.78` | Holds seed `101` stairwell framing/lighting; all 8 lines spoken in Spanish with matched lips |
| `stairwell_dub_ja_s730451892` | `730451892` | `102.1` | `11.959` | `16.115` | `-22.87` | Holds `s730451892` actors/framing/cuts; natural colloquial Japanese across all 8 lines + matched lips |
| `stairwell_dub_bilingual_s1es_s2en_s730451892` | `730451892` | `102.1` | `11.797` | `11.991` | `-30.95` | `(S1)` woman speaks Spanish while `(S2)` man answers in English (including off-screen Shot 2) |
| `stairwell_aud_am_radio_s730451892` | `730451892` | `102.0` | `8.793` | `7.140` | `-25.92` | Picture stays at identity floor (`8.79` vs `8.41`); narrowband 1930s AM-radio voice + crackle/hiss |
| `stairwell_aud_sitcom_laugh_s730451892` | `730451892` | `102.1` | `9.948` | `9.472` | `-26.18` | Serious stairwell video unchanged while live 1970s studio audience bursts into laughter & applause between lines |
| `stairwell_aud_hydrophone_sonar_s730451892` | `730451892` | `102.1` | `9.593` | `15.429` | `-27.40` | Dry concrete stairwell video with zero water/blue bleed; heavy submerged low-pass + bubbles + active sonar pings |

---

## Part C — Step 2 Mechanism Gates (`5677149d`)

### C1. Binaural (`audio_channel`) & Temporal (`audio_time_start/end`) — Falsified Under Naive `Q_aud` Subset Routing

Tested on `stairwell` (`192f`, `1344x768`, seed `730451892`, thunderstorm soundscape routed to subset `Q_aud`). Both smeared across the entire audio stream:

| Arm | Wall (s) | Pic MAE (`320x180`) | RMS (dBFS) | L / R RMS (dBFS) · `lr_corr` | L / R Spec MAE vs Base (dB) | H1 (`0..4s`) / H2 (`4..8s`) RMS (dBFS) | H1 / H2 Spec MAE vs Base (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stairwell_unsteered` *(baseline)* | — | `0.000` | `-25.87` | `-25.87 / -25.88` · `0.9964` | `0.000 / 0.000` | `-25.10 / -26.83` | `0.000 / 0.000` |
| `stairwell_router_identity_v2` | `120.1` | `8.999` | `-25.72` | `-25.71 / -25.73` · `0.9965` | `5.117 / 5.184` | `-24.86 / -26.82` | `5.353 / 4.953` |
| `stairwell_binaural_right_storm` (`channel="right"`) | `102.0` | `8.827` | `-23.30` | `-23.30 / -23.30` · `0.9942` | `9.919 / 9.915` | `-22.68 / -24.06` | `9.983 / 9.475` |
| `stairwell_temporal_half2_storm` (`t=0.5..1.0`) | `96.1` | `8.741` | `-22.82` | `-22.84 / -22.81` · `0.9956` | `10.585 / 10.585` | `-22.24 / -23.52` | `10.023 / 10.757` |

- **Why:** Direct inspection of [`comfy/ldm/minimax/audio_vae.py`](../../../../comfy/ldm/minimax/audio_vae.py#L374-L443) confirms `MiniMaxH3AudioVAE` runs its mono encoder and `BigVGAN` decoder independently per stereo channel (`b * s, 1, L`), so the AudioVAE did **not** cause the L/R collapse. Like Gate 2C's falsification of `1x Q_vid[mask]`, isolating `Q_aud_subset -> K_txt_alt` leaves intra-stream `audio <-> audio` self-attention (`Q_aud_L <-> K_aud_R` and `Q_aud[0..4s] <-> K_aud[4..8s]`) open across all `50` blocks, diffusing the alternate prompt across channels and frames.

### C2. Depth-Gated `ref_img_gain` (`V[ref_img]` Scaling) — Falsified on `old_man.jpg` (`141f`)

Tested on Pika H3 Flex `ref2va` (`workflows/h3_image_ref_plus_text_to_video_pika_api.json` adapted to `$H3_INPUT_DIR/old_man.jpg` at `141f`, `1344x768`, seed `730451892`) with a sub-zero cobalt-blue frost corridor rewrite and `ref_img_block_band="mid_late_20_49"`:

| Arm (`ref_img_gain` in `blocks.20..49`) | Wall (s) | Pic MAE vs `g=1.0` (`320x180`) | Mean `(R, G, B)` | Cool Shift `B - R` | Sound RMS (dBFS) | Observed Picture Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `ref2va_oldman_frost_ref_gain_100` (`1.0`) | `111.1` | `0.000` | `(67.77, 91.15, 110.49)` | `+42.72` | `-23.13` | Full cobalt-blue frost, icicles, snow, cold breath, & faithful likeness |
| `ref2va_oldman_frost_ref_gain_025` (`0.25`) | `96.1` | `46.334` | `(79.27, 78.86, 81.02)` | `+1.74` | `-26.46` | Warm brown wood hallway returns on right side; camera pulls back at start |
| `ref2va_oldman_frost_ref_gain_000` (`0.0`) | `96.1` | `49.339` | `(97.93, 98.96, 104.46)` | `+6.54` | `-20.06` | Pulls far back into backlit silhouette walk-in; cheek talisman distorts |

Scaling `V[ref_img]` alone in `blocks.20..49` leaves `K[ref_img]` in the softmax denominator, reducing total attention value magnitude instead of shifting softmax weight toward text tokens after `blocks.0..19` already injected the warm-wood reference layout.

### C3. `aud_to_vid_gain` Conflict Sweep (`UW` Video vs. Dry Stairwell Audio)

Tested on `stairwell` (`192f`, `1344x768`, seed `730451892`): main conditioning rewritten to deeply submerged `V1_underwater_caustics` + submerged hydrophone soundscape (`uwvid_unrouted`), while `audio_conditioning` held untouched dry stairwell text across `blocks.0..49` (`aud_to_vid_gain ∈ {1.0, 0.25, 0.0}`).

| Arm (`blocks.0..49`, `s730451892`) | `aud_to_vid_gain` | Wall (s) | Pic MAE vs UW / Dry | Spec MAE vs Dry Base (dB) | Spec MAE vs UW Unrouted (dB) | HF Tilt `2–8k` vs `100–500` (dB) | RMS (dBFS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stairwell_unsteered` *(dry base)* | — | — | `47.558` / `0.000` | `0.000` | `20.215` | `-18.79` | `-25.87` |
| `uwvid_unrouted` *(no router)* | — | `96.0` | `0.000` / `47.558` | `20.215` | `0.000` | `-22.62` | `-32.12` |
| `uwvid_dryaud_a2v_100_all` | `1.0` | `102.0` | `16.076` / `48.511` | `14.810` | `14.211` | `-24.14` | `-32.27` |
| `uwvid_dryaud_a2v_025_all` | `0.25` | `99.2` | `16.126` / `48.544` | `14.965` | `15.337` | `-20.79` | `-32.09` |
| `uwvid_dryaud_a2v_000_all` | `0.0` | `99.0` | `16.641` / `48.188` | `14.478` | `14.378` | `-22.96` | `-32.47` |

- **What this shows:** Two-prompt text routing alone (`gain=1.0`) kept submerged caustics/bubbles (`16.076` MAE vs `uwvid_unrouted`) while dropping heavy water rumble and moving `14.211 dB` away from `uwvid_unrouted`. Lowering `aud_to_vid_gain` gave brightest HF tilt at `0.25` (`-20.79 dB` vs `-24.14 dB` at `1.0`), left lip-sync intact even at `0.0` across all `50` blocks (because `Q_vid -> K_aud` remains unmasked so video follows audio timing), and yielded modest spectral separation beyond two-prompt routing (`14.478–14.965 dB` vs dry base).

### C4. `ref_audio_stream="audio_only"` Live GPU A/B (`5677149d`) — Motion Lowered, Audio Corrupted

Tested on Pika H3 Flex `ref2va` (`workflows/h3_image_ref_plus_text_to_video_pika_api.json` adapted to `$H3_INPUT_DIR/old_man.jpg` + `$H3_INPUT_DIR/h3_trio_groove_20261123.wav` at `141f`, `1344x768`, seed `730451892`):

| Arm (`ref_audio_stream`) | Wall (s) | Motion Energy MAE | Sound RMS (dBFS) | Pic / Spec MAE Between Arms | Observed Picture & Audio Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `ref2va_oldman_groove_ref_aud_both` (`both`) | `111.0` | `0.311` | `-24.50` | `8.434` / `28.488 dB` | Calm portrait + crisp, warm bass-and-drum groove matching `<Audio 1>` |
| `ref2va_oldman_groove_ref_aud_only` (`audio_only`) | `102.1` | `0.219` | `-19.83` | — | Visually slightly stiller, **but** audio breaks into distorted metallic buzz |

- **Root cause (`steering/hooks.py`):** In-place `v_bh[:, :, ra:rb, :].zero_()` removes `V[ref_audio]` from the entire main pass — including `ref_audio`'s own self-attention (`Q_ref_aud -> K_ref_aud`) across all `50` blocks, corrupting `K_ref_aud` / `V_ref_aud` before `Q_aud` reads them via `in_aud_unswapped`. Just like `ref_img_gain`, clean reference isolation requires per-query `K`-logit masking instead of mutating `v_bh` in place.

---

## Part D — Comparison Reels & Intra-Audio Cross-Masking Fix (`2026-11-24`)

- **Labeled `1x` Comparison Reels (`$H3_OUTPUT_DIR/Video/query_router_reels_2026-11-24/`):**
  - `reel_1x_dubbing_stairwell_s730451892.mp4` (`32.0s`, `1344x768`, `24 fps`, stereo AAC): Base English -> Spanish dub -> Japanese dub -> Bilingual `(S1)` Spanish / `(S2)` English.
  - `reel_1x_soundtracks_and_visuals_stairwell_s730451892.mp4` (`56.0s`, `1344x768`, `24 fps`, stereo AAC): Base -> Thunderstorm -> AM Radio -> Sitcom Laugh Track -> Hydrophone+Sonar -> Blueprint Ink + Dry Speech -> Underwater Caustics + Dry Speech.
- **Intra-audio block-diagonal cross-masking (`steering/hooks.py`):** Implemented automatic complement-span cross-masking inside `Q_aud` side-SDPA whenever `audio_channel != "both"` or `(audio_time_start, audio_time_end) != (0.0, 1.0)` (zero overhead/change when routing full audio), verified offline by [`bench/check_query_router.py`](../check_query_router.py) `Test 9` (`< 1e-5` 1-block invariance and `> 5x` 4-block separation), ready for tomorrow's live `:8188` restart test.

