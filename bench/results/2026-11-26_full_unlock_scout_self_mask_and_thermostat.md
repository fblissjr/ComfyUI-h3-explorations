# Full Unlock: H3 Studio 2.0 `Scout 4x`, Stem Choreography -> Cello Re-Score, Dual-Sprite D-Pad, & Closed-Loop `MiniMaxH3SelfMaskRouter` + `H3LivePolicy` (`2026-11-26`)

> **Commits covered:** `app/.git` [`61b69aa`](../../app/README.md) (`H3 Studio 2.0`), main repo [`eaa19c99`](../../steering/live_policy.py) (`69` registered nodes via [`MiniMaxH3SelfMaskRouter`](../../nodes.py)), plus curated workflows (`133` total validated API graphs).
> **Observable media locations:** `$H3_OUTPUT_DIR/Video/stem_choreography_rescore_2026-11-26/`, `$H3_OUTPUT_DIR/Video/stage2b_zero_code_2026-11-26/`, and `$H3_OUTPUT_DIR/Video/trackb_closed_loop_2026-11-26/`.

---

## 1. Track A — H3 Studio 2.0 Full Unlock (`app/.git` `61b69aa`) & Stage 2B Showcases

### 1.1 H3 Studio 2.0 Additions (`app/` `61b69aa`)
- **Causal Audio Beat Snapping (`app/media.py`, `app/server.py`, `app/studio_db.py`, `app/graph_builder.py`):**
  - Added `analyze_wav_beats(pcm_bytes, sample_rate, channels)` inside `clean_and_analyze_wav()`, returning `beats_s`, `causal_cut_frames` (snapped to `17 * m` standalone clip starts on H3's `[1, 4, 4, 4, 4]` causal VAE grid), and `recommended_cut_frame` from `POST /api/upload_audio`.
- **Unified `SpatialSplitCard.tsx` + `Send Split to Video` (`app/web/`):**
  - Added `shape_mode` (`box | beat_cut | wipe_lr | self_mask`), `cut_frame`, `target_phrase`, and `replay_step0` in [`app/graph_builder.py::_wire_spatial_router`](../../app/graph_builder.py), wiring `MiniMaxH3SpatialDirector` (for causal `beat_cut` and L->R `wipe_lr` splits) and `MiniMaxH3SelfMaskRouter` (for Type-a-Noun `self_mask` + `Centerpiece 1.17x`).
  - Added Image -> Video **Send Split to Video** handoff carrying prompt pairs + spatial split configuration directly into Video compose.
- **Zero-Wasted-Step Video `Scout 4x` (`ComposePane.tsx`, `TakeInspector.tsx`, `GalleryPane.tsx`, `takeSummary.ts`):**
  - Queues 4 parallel 2-step drafts (`ManualSigmas = "1.0, 0.972973, 0.0"`, `baseSeed + 0..3`, zero `/api/interrupt`), tags each draft with a `SCOUT 2-STEP` badge, and unlocks 1-tap **Render Full 6-Step** (`sigmas: ""`) at the winning seed.
- **Closed-Loop WS Telemetry (`app/contract/ws_events.json`, `App.tsx`, `RunStrip.tsx`, `store.ts`):**
  - Streams `h3_self_steer` (`self_mask_locked`, `preview_thermostat`, `audio_x0_thermostat`) directly into active run badges and saved take provenance.
- **CPU & Contract Verification (`61b69aa`):** `cd app/web && bunx tsc -b` (exit `0`), `bun test src` (`75 pass, 0 fail`, `240` assertions), `bun run build` (exit `0`), `app/test_app.py` (Tier 1 all passed), `pytest app/tests -q --isolated-browser` (`164 passed, 1 skipped` live GPU), and `bench/check_no_owner_paths.py app` (`79` artifacts scanned, exit `0`).

### 1.2 Co-Audited GPU Slot 1 — `#5` Drum/Click Stem Choreography -> Clean Cello Re-Score
- **Location:** `$H3_OUTPUT_DIR/Video/stem_choreography_rescore_2026-11-26/` (`stem_choreography_report.json`, `elapsed_s = 198.08`, click stem `h3_click_heartbeat_stem_141f.wav`, run by peer session; all 3 MP4s co-audited natively via `read_file`).

| Pass | Mode | `rms_dbfs` | Spectral Centroid | `click_env_corr` | `cello_body_100_900_share` | `video_mae_vs_pass1` | Honest AV Verdict |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :--- |
| `s5_pass1_click_choreography_00001-audio.mp4` | Click choreography (`141f`) | `-20.35 dBFS` | `121.3 Hz` | `1.0` | `0.8033` | `0.0` | Candlelit cellist bows crisply in tight tempo with the dry click track. |
| `s5_pass2a_rescore_g10_00001-audio.mp4` | Video-frozen re-score (`g=1.0`) | `-20.20 dBFS` | `545.9 Hz` | **`-0.0518`** | `0.9048` | **`0.0`** | Pixel-identical video; click completely erased and replaced by rich solo cello synced to every bow stroke. |
| `s5_pass2b_rescore_a2v0_00001-audio.mp4` | Video-frozen re-score (`a2v=0.0`) | `-27.92 dBFS` | `538.1 Hz` | **`-0.0779`** | **`0.9508`** | **`0.0`** | Identical video lock with drier, tighter mid-body chamber cello tone and lower room bloom. |

### 1.3 GPU Slot 2 — Zero-Code Dual-Sprite D-Pad (`zc1_dual_sprite_dpad_141f`)
- **Location:** `$H3_OUTPUT_DIR/Video/stage2b_zero_code_2026-11-26/zc1_dual_sprite_dpad_141f_00001-audio.mp4` (`174.11 s` on `:8188`, `141f`, `1344x768`, Pika H3 Flex `ref2va`).
- **Recipe:** Two chained `MiniMaxH3AppendSpriteTrajectory` actors (`<Picture 1>` `ref_signal/tortoise_pic1.png` walking L->C + `<Picture 2>` `old_man.jpg` walking R->C with parabolic `hop 78..102: lift=0.22`), `MiniMaxH3ReleaseReferences(video_end_percent=0.33)`, and `MiniMaxH3SoundCues("78: hop @0.85\n102: thud @0.80")`.
- **Honest AV Verdict:** Zero identity crossover between `<Picture 1>` and `<Picture 2>`: the lit-candle tortoise strides steadily right from the left edge while the elderly station attendant enters from the right, executes a clean high leap with arms up around `00:03–00:04`, lands with a synced boot thud, and leans down warmly toward the tortoise at `00:05`.

---

## 2. Track B — Closed-Loop `MiniMaxH3SelfMaskRouter` + `H3LivePolicy` (`eaa19c99` & GPU Slot 3)

Commit [`eaa19c99`](../../steering/live_policy.py) promoted [`MiniMaxH3SelfMaskRouter`](../../nodes.py) (`69`th registered node in [`bench/node_id_manifest.json`](../../bench/node_id_manifest.json), `15` inputs with optional `region_conditioning=None`), added `"audio_transients"` Step-0 `audio_x0` burst masking (`flux_z >= 2.2`), attached `policy` across `m._h3_live_policy` and `transformer_options["_h3_live_policy"]` (cloned safely downstream), bridged [`ModelPreviewOverrideKJ`](../../steering/live_policy.py) frames into `H3LivePolicy` Visual Thermostat (`strength_scale` & `0.33` anti-copy-freeze release), wired live actuators into [`MiniMaxH3Steer`](../../steering/nodes.py), [`MiniMaxH3ReleaseReferences`](../../steering/nodes.py), [`MiniMaxH3QueryRouter`](../../steering/query_router.py) (`recommended_aud_to_vid_gain = 0.0` firewall), and [`MiniMaxH3EditGuider`](../../edit_guidance.py) (`_effective_raw_mask()`), verified offline by [`bench/check_live_policy.py`](../check_live_policy.py) (`0..9`).

### 2.1 Live GPU Slot 3 Matrix (`seed = 730451892`, `1344x768`, Pika H3 Flex 6-step `fl2va` + `MiniMaxH3Sol`)
Outputs, 6-frame mask strips, and `trackb_closed_loop_report.json` live under `$H3_OUTPUT_DIR/Video/trackb_closed_loop_2026-11-26/`.

| Arm (`frames`) | Architecture / Sensors Active | Sample Time | Key Telemetry | Quantitative Pixel / Luma Result | Honest AV Verdict |
| :--- | :--- | :-: | :--- | :--- | :--- |
| **`tb1_right_speaker_2x_editguider_bridge`** (`141f`) | `SelfMaskRouter(right_speaker, z=1.45, dilation=1, region_conditioning=None)` -> `MiniMaxH3EditGuider(scale=1.95, first_step=1, last_step=3, streams='video_only')` | `98.71 s` | `hard_cov=0.227` (`[0.1617..0.2907]`), `peak=[0.6786, 0.3125]`, `centroid=[0.6311, 0.3221]` | `vs b0`: `in_mask_mae = 38.497`, `out_mask_mae = 21.605` (**`1.782x`**) | **10/10 breakthrough on `< 9%` faces:** Solves `b3/b4`'s `1x` limit with zero coords / zero `SAM 3.1`. Right man's head & neck become articulated cyan-cobalt crystal android plates + glowing sapphire optics while keeping exact lip-sync (`"It moved."` / `"Tonight."`) and photoreal left woman. |
| `tb2_stairwell_openloop_noir` vs **`tb2_stairwell_thermostat_noir`** (`90f`) | `SelfMaskRouter(text_phrase="detective", region=None)` -> `MiniMaxH3Steer(noir, +1.15)` -> `ModelPreviewOverrideKJ(taeh3.safetensors)` Visual Thermostat | `29.15 s` vs `29.16 s` (`+0.01 s`) | `preview_history=6` steps (`mean_luma 0.1894..0.2225`), `dim_understeer_risk` fired on all `6` steps, `applied_strength_scale: 1.06 -> 1.1181`, `rec_end=0.5` | `mae_vs_openloop = 9.361`; `mean_luma 0.2083 -> 0.2124`; `audio -31.05 -> -30.68 dBFS` | **`0.01 s` real-time closed-loop modulation:** Streamed `taeh3` steps via `_preview_bridge_outer` cleanly and modulated the live `noir` trajectory (`mae = 9.361`) without oversaturating highlights (`hi_clip <= 0.006`). |
| **`tb3_emergent_thunder_transients`** (`90f`, zero `.wav`) | `SelfMaskRouter(mask_source="audio_transients", isolation="all_0_49")` + electric-cyan plasma lightning storm `region_conditioning` | `36.2 s` | Step-0 `audio_x0` bursts at `active_latent_steps = [4, 5, 9, 11, 18, 26]` (`silent=21/27`, `hard_cov=0.2222`, `audio_rms = -13.98 dBFS`) | `active_frames_24fps = [13..17, 30..33, 35..38, 60..63, 86..89]`, `max_luma_frame = 38` (`0.3998` vs `0.1682` mean) | **True zero-WAV audio-driven visual burst:** Cozy candlelit study at `00:00` erupts into blinding cyan lightning + whipping papers on the emergent thunderclap (`00:01–00:02`, `f17/f35/f89`) and snaps back to warm candlelight between bursts (with minor 1-frame transient strobe where Step-0 noise peaked early). |

---

## 3. Curated Workflows Shipped (`workflows/`)

Generated via [`workflows/build_workflows.py`](../../workflows/build_workflows.py) and validated against `--object-info` (`1119` entries) in [`bench/check_widget_deviations.py`](../check_widget_deviations.py):

1. [`workflows/h3_text_to_video_pika_self_mask_speaker_edit_guider_api.json`](../../workflows/h3_text_to_video_pika_self_mask_speaker_edit_guider_api.json) — `141f` Pika H3 Flex `SelfMaskRouter(mask_source="right_speaker", threshold_z=1.45, dilation_passes=1)` -> `MiniMaxH3EditGuider(scale=1.95, first_step=1, last_step=3, streams="video_only")` paired with [`prompt_bank/edit_guidance/t2va_stairwell_right_speaker_android.txt`](../../prompt_bank/edit_guidance/t2va_stairwell_right_speaker_android.txt).
2. [`workflows/image/h3_image_t2i_self_mask_router_api.json`](../../workflows/image/h3_image_t2i_self_mask_router_api.json) — `MiniMax-H3-Image` `16 -> 4 MP` zero-coordinate noun replacement (`MiniMaxH3SelfMaskRouter(mask_source="text_phrase", target_phrase="bread", isolation="mid_20_34_shield", replay_step0=True)` transforming loaves into emerald quartz geodes while preserving surrounding foods photoreal).
