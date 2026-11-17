# Pika H3 Flex Sampling, `taeh3`/`taeh3_image` Preview Gate, & MiniMax-H3-Image Empty-Vision Probes (2026-11-17)

Date: 2026-11-17  
Server: ComfyUI `:8188` (`comfyui.service`, RTX 4090, `comfy-kitchen 0.2.41+sol.b63d8c7.up.7d6a22a`)  
Runner: `bench/run_with_preview_gate.py` (`ModelPreviewOverrideKJ` in-memory injection, `vae=None`, `audio_vae=None`, `max_resolution=672`, `jpeg_quality=80`)

## Summary

Two live GPU probes on idle `:8188` plus an instrumented `taeh3_image.safetensors` capture established four points:

1. **`MiniMax-H3-Image` T2I needs zero vision tokens (Probe B, `seed=42`, `16 -> 4 MP`):** Injecting a `1024x1024` solid `#808080` (`b1`) or `#000000` (`b2`) `EmptyImage` with `vae` unwired (Qwen3-VL vision tokens without DiT `minimax_refs`) alters composition (`PSNR = 12.18 dB` / `11.80 dB` vs baseline `b0`) and drops Laplacian sharpness by **`-17.0%`** (`1392.79` vs `1678.04`) and **`-24.3%`** (`1269.68` vs `1678.04`). The upstream consensus (`sglang`, `vllm-omni`, `diffusers`, `LightX2V`) and our shipped `workflows/image/h3_image_t2i_api.json` (0 vision tokens) hold empirically.
2. **KJNodes tiny-VAE preview gating works end to end at interactive overhead:**
   - `taeh3_image.safetensors` (`preview_frames=1`, `672x498` JPEG) captured all `7` WS messages (`step 0` Gaussian noise init + `steps 1..6`) at `~2.67s/step` on warm steps `2..6` (`38.71s` total including step-1 torch compile vs `28.52s–39.22s` uninstrumented), revealing final bread/plate composition clearly at **step 1** (`~4.9s`).
   - `taeh3.safetensors` (`preview_frames=8`, `672x384` H.264 MP4) captured all `7–8` WS messages across all 5 Pika `t2va` arms (`141` frames, `1344x768`) at `~7.85–7.93s/step` on warm steps `2..N`, locking macro shot cuts and figure blocking at **step 1** (`~12.1–17.4s`).
3. **Pika H3 Flex `lcm` beats `SamplerEulerAncestral(eta=0.5)` on `t2va` micro-detail (`seed=730451892`, `141f`):** Switching `KSamplerSelect("lcm")` (`a0`: `lap=85.25`, `luma_std=52.94`) to `SamplerEulerAncestral(eta=0.5, s_noise=1.0)` on the identical `PIKA_MANUAL_SIGMAS` grid (`a1`: `lap=72.69`, `luma_std=48.68`) dropped sharpness by **`-14.7%`** and noticeably blurred cloth weave, facial features, and background market lettering.
4. **Adding one low-sigma `shift=12` tail knot on `lcm` (`a4`) was sharpest in this 1-seed `t2va` test:** Splitting Pika's final tail drop (`0.571429 -> 0.333333 -> 0.0`) with one extra knot (`[..., 0.609272, 0.387097, 0.180301, 0.0]`) under `lcm` raised mean Laplacian variance to **`91.28`** (`+7.1%` over `a0`), preserved temporal smoothness (`flicker_ratio=1.490` vs `a0`'s `1.488`), and kept clean `-26.94 dBFS` audio with `0` clips at `68.98s` (`~7.85s` incremental cost for 1 warm step). Shipped repo defaults remain pinned to `PIKA_MANUAL_SIGMAS` (`6` steps, `lcm`) per `bench/check_distill_settings.py`; the 7-step tail curve is available for manual interactive tuning (e.g., via `@app`'s sigma editor).

---

## Probe B: `MiniMax-H3-Image` T2I Empty/Neutral Vision-Token A/B

- **Base graph:** `workflows/image/h3_image_t2i_api.json` (`16 -> 4 MP`, `4736x3552 -> 2368x1760`, `H3_IMAGE_SAMPLER="euler"`, `H3_IMAGE_MANUAL_SIGMAS`, `seed=42`)
- **Output root:** `$H3_OUTPUT_DIR/Image/nov17_probes/probe_b_t2i_vision/`
- **Condition:** In `b1` and `b2`, node `"99"` `EmptyImage(width=1024, height=1024, batch_size=1, color=...)` wired to `MiniMaxH3ImageConditioning["images.image_0"]` with optional `"vae"` removed so Qwen3-VL encodes vision tokens while `minimax_refs` stays empty (isolating Qwen vision tokens from VAE latent anchoring).
- *(Note: Initial `b0`–`b2` comparison runs ran before fixing V3 `["COMBO", {"options": ...}]` parsing in `bench/run_with_preview_gate.py::_combo_choices`, so their runs had `injected=false`; `b0_taeh3_steps/` then verified `taeh3_image.safetensors` capture cleanly at `seed=42`.)*

| Arm | Conditioning Input | Duration (s) | Size | Luma Mean / Std | RGB Means | Laplacian Var (Sharpness) | MAE vs `b0` | PSNR vs `b0` |
|---|---|---:|---|---:|---|---:|---:|---:|
| `b0_baseline` | Prompt only (`0` vision tokens) | 39.22 | 2368x1760 | 100.14 / 67.60 | 117.79, 95.98, 75.26 | **1678.04** | — | — |
| `b1_grey_vision_only` | `EmptyImage(1024, 0x808080)`, `vae` unwired | 29.19 | 2368x1760 | 101.29 / 68.37 | 125.06, 94.34, 74.71 | 1392.79 (-17.0%) | 42.936 | 12.18 dB |
| `b2_black_vision_only` | `EmptyImage(1024, 0x000000)`, `vae` unwired | 28.52 | 2368x1760 | 97.95 / 69.24 | 121.56, 91.01, 71.75 | 1269.68 (-24.3%) | 45.521 | 11.80 dB |

**`b0_taeh3_steps` (`taeh3_image.safetensors` WS capture, `injected=true`, `38.71s`):**

| WS Step | Sigma | `step_ms` | RMS Delta (`delta`) | Preview Size | Output File |
|---:|---:|---:|---:|---|---|
| 0 (init) | 1.0000 | — | — | 672x499 | `b0_taeh3_steps/steps/step_000_s1.0000.jpg` |
| 1 | 1.0000 | 4877.8 | 1.338 | 672x498 | `b0_taeh3_steps/steps/step_001_s1.0000.jpg` |
| 2 | 0.9675 | 2664.1 | 0.469 | 672x498 | `b0_taeh3_steps/steps/step_002_s0.9675.jpg` |
| 3 | 0.8875 | 2676.6 | 0.287 | 672x498 | `b0_taeh3_steps/steps/step_003_s0.8875.jpg` |
| 4 | 0.7500 | 2676.4 | 0.182 | 672x498 | `b0_taeh3_steps/steps/step_004_s0.7500.jpg` |
| 5 | 0.5328 | 2677.1 | 0.155 | 672x498 | `b0_taeh3_steps/steps/step_005_s0.5328.jpg` |
| 6 | 0.2320 | 2680.5 | 0.104 | 672x498 | `b0_taeh3_steps/steps/step_006_s0.2320.jpg` |

Visual inspection (`probe_b_comparison_3up.png`, `probe_b_detail_crops.png`, `probe_b_step_trajectories.png`): `b0_baseline` renders crisp sourdough crumb structure, basket reeds, wood grain, and silverware placement. Both neutral-image arms rearrange items, wash slightly warmer/redder, and soften micro-contrast on crust, beets, and table grain.

---

## Probe A: Pika H3 Flex `t2va` Sampler & Sigma Curve A/B

- **Base graph:** `workflows/h3_text_to_video_pika_api.json` (`1344x768`, `length=141` frames / `5.875s` dual-clock `39 + 51*2`, `seed=730451892`, Pika SVD995 `fl2va` LoRA)
- **Output root:** `$H3_OUTPUT_DIR/Video/nov17_probes/probe_a_pika_sampling/`
- **Preview capture:** Every arm had `injected=true` (`taeh3.safetensors`, `preview_frames=8`, `max_resolution=672`, `vae`/`audio_vae` unwired).
- **Tested schedules (exact `shift=12` transform `12*t / (1 + 11*t)`):**
  - `PIKA_MANUAL_SIGMAS` (`a0`, `a1`): `"1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"` (`t = [1.0, 0.75, 0.50, 0.25, 0.10, 0.04, 0.0]`)
  - Rebalanced 6-step (`a2`): `"1.0, 0.962264, 0.896797, 0.77193, 0.557457, 0.270677, 0.0"` (`t = [1.0, 0.68, 0.42, 0.22, 0.095, 0.03, 0.0]`)
  - 7-step tail knot (`a3`, `a4`): `"1.0, 0.972973, 0.923077, 0.8, 0.609272, 0.387097, 0.180301, 0.0"` (`t = [1.0, 0.75, 0.50, 0.25, 0.115, 0.05, 0.018, 0.0]`)

| Arm | Sampler | Steps | Duration (s) | Mean `step_ms` | Luma Mean / Std | Laplacian Var | Motion Delta | Flicker Ratio | Audio RMS / Peak (dBFS) | Clips |
|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| `a0_lcm_contract` | `KSamplerSelect("lcm")` | 6 | 73.69 | 9450.1 | 79.89 / 52.94 | 85.25 | 6.885 | 1.488 | -27.35 / -10.56 | 0 |
| `a1_rf_ancestral_eta05` | `SamplerEulerAncestral(0.5, 1.0)` | 6 | 80.65 | 8983.5 | 77.18 / 48.68 | 72.69 (-14.7%) | 6.280 | 1.507 | -28.05 / -11.40 | 0 |
| `a2_rf_ancestral_eta05_rebalanced6` | `SamplerEulerAncestral(0.5, 1.0)` | 6 | 61.07 | 8607.5 | 77.00 / 49.66 | 78.40 (-8.0%) | 6.551 | 1.502 | -27.65 / -10.34 | 0 |
| `a3_rf_ancestral_eta05_7step_tail` | `SamplerEulerAncestral(0.5, 1.0)` | 7 | 69.03 | 8512.8 | 77.28 / 48.71 | 75.68 (-11.2%) | 6.391 | 1.506 | -27.86 / -11.31 | 0 |
| `a4_lcm_7step_tail` | `KSamplerSelect("lcm")` | 7 | 68.98 | 8505.3 | 80.18 / 53.16 | **91.28 (+7.1%)** | **7.057** | 1.490 | -26.94 / -10.31 | 0 |

### Native Media Inspection (`probe_a_keyframes_5arm.png`, `probe_a_taeh3_step_grid.png`, `probe_a_grid.1fps.00000-00120.mp4`, and per-arm `.mp4`s)

- **`a0_lcm_contract` vs `a1_rf_ancestral_eta05`:** All 3 market cuts (`[Shot 1]` stallholder sets down orange crate, `[Shot 2]` porter lifts crate and drops coins in metal tin, `[Shot 3]` porter carries crates away between stalls), dialogue (`"Last of the good ones. After this it is all imports."` / `"Then I will take two."`), lip sync, and ambient foley track cleanly. `a0` shows higher contrast and crisper crate woodwork, produce sheen, faces, and background shutter signs than `a1`, which exhibits mild ancestral haze (`-14.7%` Laplacian variance).
- **`a2` (`rebalanced6`):** Recovering lower-sigma budget lifted ancestral sharpness over `a1` (`78.40` vs `72.69`), kept the coin tin cleanly on the counter at `f=84` (`00:03–00:04`), and avoided any clipping, though pulling middle knots off `PIKA_MANUAL_SIGMAS` kept micro-detail softer than `a0` and garbled the background stall fascia glyphs slightly.
- **`a3` vs `a4` (`7-step tail`):** Adding the single low-sigma `0.180301` refinement knot helped both samplers vs their 6-step counterparts on `PIKA_MANUAL_SIGMAS` upper knots (`a3`: `75.68` vs `a1` `72.69`; `a4`: `91.28` vs `a0` `85.25`). Visually in `a4`, knit vest texture, crate slat seams, orange peel pores, and background shoppers at `00:04–00:05` are sharpest of the 5 arms without added flicker (`1.490` vs `1.488`).
- **Limits:** Single scene (`h3_text_to_video_pika`), single seed (`730451892`), and `t2va` only; `ref2va`/`fl2va` and multi-seed averages are needed before considering any change to Pika's default contract.

### Inspected Files (`$H3_OUTPUT_DIR`)

- `$H3_OUTPUT_DIR/Image/nov17_probes/probe_b_t2i_vision/{probe_b_comparison_3up.png,probe_b_detail_crops.png,probe_b_step_trajectories.png,metrics.json}`
- `$H3_OUTPUT_DIR/Video/nov17_probes/probe_a_pika_sampling/{probe_a_keyframes_5arm.png,probe_a_taeh3_step_grid.png,probe_a_grid.mp4,probe_a_grid.1fps.00000-00120.mp4,metrics.json}`
- Per-arm outputs & WS captures under `{arm}/steps/` and `{arm}/outputs/`
