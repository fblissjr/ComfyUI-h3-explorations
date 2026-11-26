# Closed-Loop Self-Steering Rescues, `MiniMaxH3DepthBeatMask` (`70` Nodes), H3 Studio 2.0 Cleanup, & Showcase Reel (`2026-11-26`)

> **Commits covered:** `app/.git` [`edb13a9`](../../app/README.md) (`H3 Studio 2.0`), main repo [`650b4e46`](../../steering/live_policy.py) – [`fab1e94e`](../../steering/live_policy.py) (`H3LivePolicy` hardening, Dynamic Step-1 Auto-Shield, Remote `heylook` Step-0 Visual Critic + OOV noun mapping, calibrated burn ceilings), [`MiniMaxH3DepthBeatMask`](../../depth_beat_mask.py) (`70` registered node IDs in [`bench/node_id_manifest.json`](../node_id_manifest.json)), and `3` curated workflows (`136` total validated API graphs).
> **Observable media locations:** `$H3_OUTPUT_DIR/Video/closed_loop_rescues_2026-11-26/`, `$H3_OUTPUT_DIR/Video/phase2_showcases_2026-11-26/`, `$H3_INPUT_DIR/portal_tear_sunny_study_room.png`, and Showcase Reel `$H3_OUTPUT_DIR/Video/h3_closed_loop_and_portal_showcase_reel_2026-11-26.mp4` (`805f / 33.54 s`, `1344×888 @ 24 fps`, `48 kHz` stereo AAC).

---

## 1. Track Summary (Phase 1 Half-Baked Cleanup + Phase 2 Extensions)

### 1.1 H3 Studio 2.0 Cleanup (`app/.git` `edb13a9`)
- **Explicit Manual-Sigma Gating:** Exposed `GuideRecipe.has_manual_sigmas` on `POST /api/recipe` (`app/recipes.py`, `app/contract/openapi.json`, `app/web/src/api-types.gen.ts`), disabling `Scout 4x` in `ComposePane.tsx` whenever custom manual sigmas override the scheduler.
- **Full `MiniMaxH3SelfMaskRouter` Control Exposure:** Surfaced all `6` `SELF_MASK_SOURCES` (`text_phrase`, `active_speaker`, `left_speaker`, `right_speaker`, `audio_transients`, `thermostat_only`), `threshold_z`, `dilation_passes`, and `auto_thermostat` (`app/studio_db.py`, `app/graph_builder.py`, `app/web/src/{types,document,request,store,takeSummary}.ts`, `SpatialSplitCard.tsx`, `SplitStudioSheet.tsx`), wiring standalone `Auto Thermostat` (`thermostat_only`) cleanly alongside `MiniMaxH3QueryRouter`, `SelfMask -> 2x EditGuider` blank-`region_prompt` surgical bridging, and server-side `freeze_audio` `audio_zone="base"` pinning.
- **Live Spatial Lock-On Telemetry:** Added `peak_xy` to `h3_self_steer` (`app/contract/ws_events.json`, `app/web/src/takes.ts`, `LivePreview.tsx`, `app.css`) rendering a normalized SVG `.live-steer-reticle` crosshair during live runs (zero inline CSS styles).
- **Verified (`edb13a9`):** `cd app/web && bunx tsc -b` (`0`), `bun test src` (passed), `bun run build` (built), `app/test_app.py` (Tier 1 passed), `pytest app/tests -q --isolated-browser` (`165 passed, 1 skipped` live GPU), and `bench/check_no_owner_paths.py app` (`79` artifacts scanned, exit `0`).

### 1.2 `H3LivePolicy` Hardening, Dynamic Shield, & Remote `heylook` Step-0 Critic (`650b4e46`..`fab1e94e`)
- **Hardening (`650b4e46`):** Phase-matched `[1, 4, 4, 4, 4]` `evaluate_video_x0_motion` inside `_post_cfg`, Step-1 copy-freeze guard auto-lowering `<Picture 1>` `eff_ref_end = 0.33` alongside `<Video 1>`, fast-attack `0.35` highlight-burn taper floor, auto-populated `prompt_id` from `PromptServer.instance.last_prompt_id`, capped `2`-step smooth decay `[1.0, 0.58, 0.28]` on `synthesize_audio_transient_mask`, and zero-hook `"thermostat_only"` mode coexisting with `SpatialQueryRouter`, `QueryRouter`, `SageAttention`, and `MiniMax-H3-Image`.
- **Phase 2 Extensions (`65cc7cac`, `4c615a3e`, `fab1e94e`):** Dynamic Step-1 Auto-Shield (`early_0_24 -> *_shield` on `< 32%` low-contrast washout masks), opt-in remote `heylook` Step-0 Visual Critic (`0 MB` local VRAM -> CPU `WordVector(mode="project_out")` across non-vision text rows with OOV artifact expansion e.g. `letterbox -> ["border", "bars", "letterbox", "frame", "widescreen"]`), calibrated `evaluate_preview_frame` ceilings to live `taeh3` Step-0 previews, and `check_live_policy.py` (`0..12` green).

### 1.3 Composable `MiniMaxH3DepthBeatMask` (`70`th Node) & `3` New Curated Workflows (`136` Total)
- **New Node ([`depth_beat_mask.py`](../../depth_beat_mask.py)):** Converts any image + optional audio WAV into a `[F, H, W]` ComfyUI `MASK` (`depth_band`, `depth_wave`, `audio_beat_cut`, `audio_beat_pulse`, `depth_band_on_beat`), using `vendor/meta_moge2` metric depth (unloaded off CUDA in `finally:`) and `17*m` / onset WAV beat snapping ([`bench/check_depth_beat_mask.py`](../check_depth_beat_mask.py) `0..4/5` OK).
- **Shipped Curated Workflows (`136` validated graphs):**
  1. [`workflows/h3_text_to_video_pika_self_mask_audio_transients_api.json`](../../workflows/h3_text_to_video_pika_self_mask_audio_transients_api.json) (`141f` Pika `t2va`, emergent `audio_transients` lightning without external WAV)
  2. [`workflows/h3_i2v_pika_moge2_portal_tear_api.json`](../../workflows/h3_i2v_pika_moge2_portal_tear_api.json) (`141f` Pika `fl2va` `i2v`, `MiniMaxH3DepthBeatMask` `depth_band_on_beat` `0.65..1.0` @ `cut_frame=12` + `SoundCues("0.5s: slam")` + `SpatialQueryRouter(follow_mask)`)
  3. [`workflows/h3_r2v_dual_ref_spatial_split_pika_api.json`](../../workflows/h3_r2v_dual_ref_spatial_split_pika_api.json) (`141f` Pika `ref2va`, 2-photo `<Picture 1>` + `<Picture 2>` spatial split with `ReleaseReferences(0.50)` + `early_0_24` + `binaural_split`)

---

## 2. Live Silicon Results — GPU Batch A & A2 (`$H3_OUTPUT_DIR/Video/closed_loop_rescues_2026-11-26/`)

All Batch A & A2 renders were co-audited natively via `read_file` against `closed_loop_rescues_report.json`:

| Arm / Pair | Mechanism Tested | Key Observable Measurements | Honest Silicon Verdict |
| :--- | :--- | :--- | :--- |
| **`r5b_critic_open` vs `r5b_critic_s0_replay_scrub`** | Remote `heylook` `gemma-4-31B-it-8bit` (`5.45 s`, `0 MB` local VRAM) + OOV CPU `WordVector` Step-0 rollback (`1.17x`) | `top_10pct_luma`: **`0.0000 -> 0.7692`**; `bot_10pct_luma`: **`0.0000 -> 0.5715`**; `mean_luma`: `0.3674 -> 0.5420`; scrubbed `["border", "bars", "letterbox", "frame", "widescreen"]` | **Knockout rescue.** Open-loop locked thick black letterbox bars top & bottom; Step-0 critic detection + Step-0 rollback completely obliterated both bars into full-bleed `1344x768` Utah salt flats & blue sky while preserving the exact vintage red coupe and V8 engine track. |
| **`r5b_critic_scrub` / `r5b_critic_s1_negscrub`** | Remote Step-0 critic scrubbing Steps `1..5` without Step-0 replay | `r5b_critic_scrub`: `rgb_mae = 4.021`; `r5b_critic_s1_negscrub`: `top_10pct_luma = 0.0001`, `bot_10pct_luma = 0.0004` | **Architectural proof.** Proves Steps `1..5` modulate fine detail (`mae=4.021`) without removing Step-0 (`sigma=1.0`) layout geometry; **Step-0 replay (`1.17x`) is what erases borders**. |
| **`r4_a2v_open` vs `r4_a2v_autoclamp`** | Step-1 `audio_x0` `stereo_collapse_risk` live `a2v=0.0` clamp | `rgb_mae = 4.138`; `hi_clip_frac 0.0028 -> 0.0018`; `audio_lr_corr 0.9793 -> 0.9025`; `audio_hf_6khz_share 0.0001 -> 0.0`; `audio_rms_dbfs -24.58 -> -23.28` | **Confirmed live firewall rescue.** Step-1 `audio_x0` clamped `recommended_aud_to_vid_gain = 0.0` mid-sample (`lr_latent_corr = 0.9129`), opening L/R stereo width, zeroing `>= 6 kHz` hiss, and keeping the molten glassblowing furnace & harp clean. |
| **`r3_tb3_v2_smooth_decay`** | Zero-WAV `audio_transients` with capped `2`-step smooth decay | `active_latent_steps = [4,5,6, 11,12,13, 16,17,18,19, 26]`, `mean_luma = 0.1372`, `audio_rms_dbfs = -13.98` | Crisp cyan lightning bolt across the arched rain window at `00:01`, parchment sheets flying into mid-air at `00:02`, clean return to warm oil lamps by `00:03`. |
| **`r1_burn_v2_open` vs `r1_burn_v2_thermostat` & `r1` / `r2` / `r5a`** | Thresholded Step-`0/1` governors (`burn` `v1`/`v2`, `video_x0` freeze, Auto-Shield) | `r1_v1`: `rgb_mae = 0.0`; `r1_v2`: `rgb_mae = 10.957` (`hi_clip 0.0592 -> 0.0591` gentle `0.9952` Step-0 taper); `r2_freeze`: `0.0` (`norm_temporal_delta = 0.159..0.438`, zero false freeze triggers); `r5a`: `0.0` (`56.2` clean portal contrast) | **Honest calibration telemetry.** High-contrast scenes & active-motion scenes correctly stayed untouched (`rgb_mae = 0.0`), while `r1_burn_v2` verified gentle live Step-0 burn tapering when `hi_clip = 0.1035` barely crossed `0.10`. |

---

## 3. Live Silicon Results — GPU Batch B (`$H3_OUTPUT_DIR/Video/phase2_showcases_2026-11-26/`)

| Showcase Clip | Spec & Runtime | Measured Metrics | Observable Native Video/Audio Verdict |
| :--- | :--- | :--- | :--- |
| **`b1_real_room_portal_tear.mp4`** *(+ `$H3_INPUT_DIR/portal_tear_sunny_study_room.png` & `b1_room_moge2_depth_and_mask_sheet.png`)* | `141f`, `1344x768`, Pika `fl2va` `i2v`, `MiniMaxH3DepthBeatMask(depth_band_on_beat, near=0.65, far=1.0, cut=12)` + `SoundCues("0.5s: slam")` + `SpatialQR(early_0_24, follow_mask)`, `90.32 s` | Depth `0.078m..1.295m` (`median=1.234m`); `mask f0..11 = 0.0`, `f12 = 0.0975`, `f34 cov = 39.0%`; `pre_beat_f8_vs_f0_mad = 1.80`; `post_beat_f60_fg_mad = 6.60` vs `portal_mad = 37.47` (**`5.68×` fg/bg isolation**); `audio rms = -22.24 dBFS` | Calm sunlit mid-century study (`f0..f11`) cracks open on `0.5s slam` into swirling molten-orange lava & jagged electric-cyan arcs **strictly behind** the unmoved foreground cognac leather armchair, walnut credenza, lamp, and coffee table, casting warm fire rim reflections across the furniture. |
| **`b2_dual_ref_video_split.mp4`** | `141f`, `1344x768`, Pika `ref2va`, `<Picture 1>` `old_man.jpg` + `<Picture 2>` `woman_blonde.jpg` (`dit_short_edge=768`) + `ReleaseRefs(0.50)` + `SpatialQR(early_0_24, binaural_split)`, `92.93 s` | `f70 left RGB = [52.9, 37.7, 28.4]` (`R-B = +24.5` warm amber); `f70 right RGB = [63.8, 69.8, 80.1]` (`B-R = +16.3` neon cool); stereo `-21.30 dBFS` (L) vs `-23.49 dBFS` (R), `stereo_diff = -21.17 dBFS` | Two real reference portraits sit across one seamless continuous dark walnut foreground table in distinct opposite worlds (1940s candlelit study left vs rain-streaked Tokyo neon booth right) with lip-synced S1/S2 stereo dialogue (*"The package crossed borders."* / *"Then open the gate."*) and zero identity bleed. |
| **`b3_thunder_transients_smoothed.mp4`** *(+ `b3_thunder_transients_mask_strip.png` & `b3_tb3_old_vs_new_decay_sheet.png`)* | `90f`, `1344x768`, Pika `fl2va` `t2va` @ matched seed `730451892`, `MiniMaxH3SelfMaskRouter(audio_transients, all_0_49)`, `34.99 s` | `active_latent_steps = [4,5, 9,10,11, 16..19, 26]`; mask tail `f38..f46`: `0.747 -> 0.433 -> 0.209 -> 0.029`; `f35..f46 luma`: OLD `[31.34, 68.65, 90.08, 107.91, 72.83]` vs NEW `[35.25, 62.48, 76.63, 76.93, 44.29]` -> `91.76 -> 84.14 -> 66.45`; `audio rms = -13.98 dBFS` | Direct matched-seed win over `tb3_emergent_thunder_transients_00001-audio.mp4`: removes `OLD f35/f38` ghost double-exposure & empty-room jump cut, holding the archivist solid as he turns toward intense blue plasma window sparks before easing smoothly back into candlelight. |

---

## 4. Annotated Showcase Reel (`$H3_OUTPUT_DIR/Video/h3_closed_loop_and_portal_showcase_reel_2026-11-26.mp4`)

- **Specs:** `805` frames (`33.54 s`), `1344×888` (`60 px` header bar + `1344×768` viewport + `60 px` `"WATCH / LISTEN FOR:"` footer), `24.0 fps` H.264 `yuv420p` + `48 kHz` stereo AAC (`192 kbps`), `7.81 MB`.
- **Chapters:** Intro Title Card -> `r4_a2v_open` vs `r4_a2v_autoclamp` A/B -> `r5b_critic_open` vs `r5b_critic_s0_replay_scrub` A/B -> `tb3` vs `b3_thunder_transients_smoothed` A/B -> `b1_room_moge2_depth_and_mask_sheet` + `b1_real_room_portal_tear` -> `b2_dual_ref_video_split` (`<Pic 1>`/`<Pic 2>` corner badges) -> `tb1_right_speaker_2x_editguider_bridge_00001-audio` -> Summary Card.

---

## 5. Offline Verification Suite

All checks executed on CPU (`CUDA_VISIBLE_DEVICES=""`) after final doc & workflow edits:
- `workflows/build_workflows.py` -> `validated 136 graphs ok`
- `bench/check_node_ids.py` -> `70 nodes ok`
- `bench/check_schema_defaults.py`, `bench/check_literal_widgets.py`, `bench/check_widget_deviations.py`, `bench/check_prompt_guide_conformance.py`, `bench/check_ref_prompt_labels.py`, `bench/check_live_policy.py` (`0..12`), `bench/check_depth_beat_mask.py` (`0..4/5`), `bench/check_imports.py`, `bench/check_doc_links.py`, `bench/check_doc_inventory.py`, `bench/check_no_owner_paths.py`, `bench/check_no_owner_paths.py app`, and `bench/check_commit_ready.py` -> all green.
