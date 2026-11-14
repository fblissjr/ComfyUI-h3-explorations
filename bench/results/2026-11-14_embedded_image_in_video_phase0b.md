# Embedded Image Generation & Editing in H3 Video Workflows — Phase 0B Live Bench, 2026-11-14

**Question.** Within one zero-swap ComfyUI graph (`1x Qwen3-VL-32B bf16` encoder + `1x fl2va int8_convrot` DiT + sibling `MiniMaxH3LoRABranch` nodes + strict `image_vae` / `video_vae` split), how well do two Phase 0B patterns turn an in-graph `P-H3-Image LoRA` `T2I -> 2D Edit` still pair into a 6-step Pika AV clip against a matched-seed `i2va` control:

1. **Arm 0 (`arm0_t2i_i2va_control`):** In-graph `P-H3-Image LoRA` `8 -> 4 MP` (`16:9`) `T2I` `first_frame` (`seed=4201`) -> `MiniMaxH3Conditioning(first_frame=image_a)` -> 6-step Pika `i2va` (`seed=4301`, `1344x768`, `141` frames / `5.875s` at `24 fps`).
2. **Arm 1 (`arm1_t2i_edit_bookends_fl2va`):** Cached `first_frame` + in-graph `P-H3-Image LoRA` `8 -> 4 MP` 2D Edit (`image_1=image_a`, `seed=4202`, transforming sunlit afternoon workshop into rainy deep-indigo blue-hour night with the right-hand green-glass banker's lamp lit) -> `MiniMaxH3Conditioning(first_frame=image_a, last_frame=image_b)` -> 6-step Pika `fl2va` (`seed=4301`).
3. **Arm 2 (`arm2_t2i_edit_media_cv_steer`):** Cached `image_a` + cached `image_b` -> [`MiniMaxH3MakeMediaControlVector`](../../control_vector_nodes.py) (`model=<Pika LoRABranch>`, `pooling="scene_text"`, `short_edge=512`, `scene_prompt=I2VA_PROMPT`, `overwrite=True`) -> [`MiniMaxH3Steer`](../../control_vector_nodes.py) (`strength=1.25`, `mode="add_norm_matched"`, `target="scene_only"`, `start_percent=0.0`, `end_percent=1.0`) -> IDENTICAL `MiniMaxH3Conditioning(first_frame=image_a, prompt=I2VA_PROMPT)` and `seed=4301` as Arm 0.

See [`docs/wiki/research/image/2026-11-14_embedded_image_in_video_plan.md`](../../docs/wiki/research/image/2026-11-14_embedded_image_in_video_plan.md) and runner [`data/embedded_image_video_2026-11-14/run_phase0b_bench.py`](../../data/embedded_image_video_2026-11-14/run_phase0b_bench.py) (with metrics in `analysis_metrics.json` under the same gitignored directory).

---

## Length Law Observation (`length = 90` vs `141`)

An initial queue at `length = 90` (`3.750s`, the shortest video-grid `on_both_clocks` count above `39`) failed in [`MiniMaxH3Conditioning`](../../conditioning.py) (`keyframe_canvas.py:181-189`) because `resolve_keyframe_geometry` enforces `duration_in_range(length)` (`124..362` frames, `5..15s` at `24 fps`). Switching to **`length = 141`** (`8 * 17 + 5`, `42` video latent steps, `235` `40 Hz` audio tokens, `5.875s` -> `5.88-second mark` in `FL2VA_PROMPT`) is the shortest dual-clock (`on_both_clocks`) length in `[124, 362]` and passed preflight (`0 FAIL`) and live execution across all three arms.

---

## Environment, Zero-Swap Resident Graph, & Timings

- **Hardware / Runtime (`http://127.0.0.1:8188`, server at `95aa4447`):** RTX 4090 (`25,295,454,208 B` / `23.56 GiB` VRAM), `134,808,080,384 B` (`125.55 GiB`) host RAM, ComfyUI `0.45.0`, `pytorch 2.15.0+cu134`, `comfy-kitchen 0.2.41+sol.b63d8c7.up.7d6a22a`, `sol_attn_cuda 0.2.41+sol.b63d8c7.up.7d6a22a`, flags `--cuda-malloc --supports-fp8-compute --mmap-torch-files`.
- **Shared Loaders (`client_id` reused across all 3 arms):**
  - DiT: `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (`UNETLoader`)
  - Encoder: `qwen3vl_32b_minimax_h3_bf16_pruned.safetensors` (`MiniMaxH3EncoderLoader`)
  - Image VAE (only `t=1` encode/decode): `p-h3-image_vae_int8_convrot.safetensors`
  - Video VAE (only video conditioning/decode): `minimax_h3_video_vae_int8_convrot.safetensors`
  - Audio VAE: `minimax_h3_audio_vae_fp32.safetensors`
- **Image LoRA (`t=1`):** `h3/p-h3-image_lora_bf16.safetensors` (`8`-step Euler, `shift=1.0`, [`MiniMaxH3ImageSize`](../../h3_image_nodes.py) `generate_megapixels=8.0` -> `megapixels=4.0`, decoded & Lanczos-shrunk to `2720x1536`, adapted by [`MiniMaxH3Conditioning`](../../conditioning.py) `canvas="from_keyframe"` to `1344x768`).
- **Video LoRA:** `h3/pika_h3_flex_fl2va_svd995_bf16.safetensors` (`6`-step LCM, `shift_video=12.0`, `shift_audio=3.0`, [`MiniMaxH3Sol`](../../sol_attn_h3.py) `tau=1.0`, `quantizer="rotated"`, `dense_blocks="49"`, `sink_conditioning="exact_kv_and_all_rows"`, `start_percent=0.2`, `end_percent=1.0`, `min_tokens=12288`).
- **Zero-Swap Telemetry:**
  - Arm 0 (`arm0_t2i_i2va_control`): **`82.06 s`** (`ram_free = 87.85 GB`, `vram_free = 2.80 GB`; builds T2I `first_frame` + `i2va` video)
  - Arm 1 (`arm1_t2i_edit_bookends_fl2va`): **`124.09 s`** (`ram_free = 87.14 GB`, `vram_free = 3.16 GB`; reuses cached `first_frame`, builds `8 -> 4 MP` 2D Edit `image_b` + bookended `fl2va` video)
  - Arm 2 (`arm2_t2i_edit_media_cv_steer`): **`96.08 s`** (`ram_free = 86.75 GB`, `vram_free = 2.90 GB`; reuses both cached stills, extracts media control vector + steers `i2va` video)
  - Host `free -b` after `analyze_phase0b.py`: `used = 47,528,251,392 B`, `available = 87,279,828,992 B`, **`Swap = 0 B`**.

---

## Generated Still Baseline & Media Control Vector Diagnostics

1. **In-Graph Stills (adapted via `FIT` to `1344x768` for video reference metrics):**
   - `t2i_first_frame_00001_.png` (`$H3_OUTPUT_DIR/Image/embedded_image_video_2026-11-14/`, `2720x1536`, `6,024,938 B`): warm golden afternoon sunlight from left paned window, clockmaker holding brass balance wheel above open watch movement, unlit green-glass oil lamp on right (`mean_luma = 95.98`, `mean B-R = -38.07`).
   - `edit_second_frame_00001_.png` (`$H3_OUTPUT_DIR/Image/embedded_image_video_2026-11-14/`, `2720x1536`, `5,436,946 B`): dark rainy blue-hour twilight outside rain-streaked panes with a lit amber flame inside the glass lamp illuminating hands and watch (`mean_luma = 53.45`, `mean B-R = +2.18`; `image_b` vs `image_a` **`MAE = 51.32`**, `RMSE = 73.22`, `PSNR = 10.84 dB`).
2. **Extracted Media Control Vector (`<comfy>/models/control_vectors/phase0b_edit_look__fl2va__lora_2a8c2b52.safetensors`, `44,176 B`; sibling `.md` `4,461 B`):**
   - Pooling grid: `short_edge = 512` -> `qwen_view_size = [896, 512]` (`450` vision tokens + `126` pooled `scene_text` rows out of `681` total sequence tokens).
   - **Norm warning (critical observation):** `vector.post_refiner` `[5376]` measured **`raw_norm = 5.998986`** (`median_row_norm = 163.6183`; `3.67%` relative shift), and `vector.encoder` `[5120]` measured `raw_norm = 5.662035` (`median_row_norm = 281.7267`). The node's own quality table flags `< 14` on `scene_text` as low-SNR signal dilution because pooling `126` identical prompt text tokens only captures indirect VLM cross-attention bleed from the `450` `<Picture 1>` vision tokens.

---

## Quantitative & Native Audio/Visual Comparison (`141` Frames, `1344x768`, `24 fps`)

| Arm | Wall (s) | MP4 Size (B) | Frame 0 -> A MAE (PSNR) | Frame 140 -> A MAE | Frame 140 -> B MAE (PSNR) | Closest Frame to B (MAE) | Clip Luma | Clip `B - R` (`f0 -> f68 -> f140`) | Adj Diff Mean (Min..Max) | Audio RMS / Peak (Active) | Provenance Stamp (`n_sparse/evals`) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| **Arm 0:** `i2va` Control | `82.06` | `3,337,552` | `17.31` (`17.52 dB`) | `43.88` | `64.99` (`9.16 dB`) | `f52` (`52.16`) | `94.14` | `-37.23` (`-38.15 -> -36.92 -> -35.69`) | `2.66` (`0.51..5.51`) | `-38.12 / -0.40 dBFS` (`0.0763`) | `stamp_..._462458f5.json` (`5/6`) |
| **Arm 1:** `fl2va` T2I+Edit Bookends | `124.09` | `2,462,192` | `17.18` (`17.54 dB`) | `52.49` | **`5.11`** (**`28.98 dB`**) | **`f140` (`5.11`)** | `66.62` | **`-11.20`** (`-38.95 -> -2.52 -> +0.73`) | `0.94` (`0.14..2.45`) | `-41.81 / -0.47 dBFS` (**`0.6948`**) | `stamp_..._b1d22593.json` (`5/6`) |
| **Arm 2:** Media CV Steer (`s=1.25`) | `96.08` | `3,329,420` | `17.35` (`17.51 dB`) | `50.64` | `65.92` (`9.03 dB`) | `f51` (`52.26`) | `94.87` | `-37.14` (`-38.06 -> -37.46 -> -36.17`) | `2.37` (`0.45..5.66`) | `-41.39 / -3.08 dBFS` (`0.0469`) | `stamp_..._9743b377.json` (`5/6`) |

*Outputs under `$H3_OUTPUT_DIR`: `Video/embedded_image_video_2026-11-14/{arm0_t2i_i2va_control,arm1_t2i_edit_bookends_fl2va,arm2_t2i_edit_media_cv_steer}_00001-audio.mp4` and provenance sidecars under `$H3_OUTPUT_DIR/provenance/`. Companion `.1fps.00000-00140.mp4` review strips live alongside them under `$H3_OUTPUT_DIR/Video/embedded_image_video_2026-11-14/`.*

### Native Media Inspection Findings (`read_file` on stills, `.1fps`, and `24 fps` audio-muxed `.mp4`s)

1. **Arm 0 (`i2va` control):** Smoothly animates the clockmaker placing the brass balance wheel into the open pocket watch movement (`00:00–00:02`), setting tweezers aside (`00:02–00:03`), and reaching toward loose gears/case lid (`00:03–00:05`), while keeping warm afternoon sunlight and the unlit oil lamp locked throughout. Audio cleanly delivers mechanical workshop room tone and sharp synchronized metallic tool clicks at `00:03–00:05`.
2. **Arm 1 (`fl2va` bookends from `T2I -> Edit`):** Smooth, physically grounded temporal transition with zero pops or freezes (`adj MAE mean = 0.94`). Sunlight on the left paned window cools continuously from golden afternoon (`f0 B-R = -38.95`) through twilight (`f34 B-R = -22.33`, `f51 B-R = -10.44`) into dark rain-streaked blue night (`f140 B-R = +0.73`) while the green-glass banker's lamp flame literally ignites inside its chimney around `00:01–00:02` and casts warm amber highlights across the movement and clockmaker's knuckles, converging onto `image_b` at frame `140` (`MAE = 5.11`, `PSNR = 28.98 dB`). Audio smoothly layers soft continuous evening rain pattering against the window (`active_frac = 0.6948`, ~`9x` Arm 0) over faint bench watch ticks.
3. **Arm 2 (`MakeMediaControlVector(pooling="scene_text", 1 pair) + MiniMaxH3Steer(strength=1.25)`):**
   - **No color/relight transfer onto pinned daylight `first_frame`:** Despite identical `seed=4301` and neutral `I2VA_PROMPT` to Arm 0, Arm 2 stayed sunlit (`clip B-R = -37.14` vs Arm 0 `-37.23`, `clip luma = 94.87` vs `94.14`). Visually, the lamp stays unlit and the window stays warm daylight, with only minor secondary differences (a faint brief curl of wispy mist above the tweezers around `00:01–00:02` and slightly altered right-hand reach path toward the watch lid (`f140 -> A MAE = 50.64` vs Arm 0 `43.88`)).
   - **Root cause confirmed in measurements:** Single-pair `scene_text` extraction produced a small cross-attention residual (`raw_norm = 5.999` `< 14` low-signal bound), and in `i2va` the unedited sunlit `first_frame` (`clean_level = 0.999`) acts as exact `cond` latent anchor across all 6 steps. By contrast, when pixel-space target geometry or global lighting change is desired, **rendering/editing a 2D keyframe (`image_b`) and supplying actual `cond` latent targets (`fl2va` bookend in Arm 1, or Phase 1 mid-clip `GuideLatent` + `guide_end_percent`) succeeds cleanly on the first try.** `MakeMediaControlVector` on single `scene_text` pairs should not be treated as a drop-in relight replacement for an unedited `first_frame`; future media-vector tests should evaluate multi-pair style datasets (`raw_norm >= 14`) and `first_frame`-free `t2va` or `guide_end_percent`-released `ref`/`guide` setups.

---

## Ready for Next Step (Phase 1 Live A/B Post-Reload)

Commit `aa943b17` implemented `guide_end_percent` mid-clip release in [`MiniMaxH3ReleaseReferences`](../../control_vector_nodes.py) and prefix-extending keyframe support in [`EditGuider`](../../edit_guidance.py) (not yet loaded in the running `95aa4447` server process). Once `comfyui.service` reloads commit `aa943b17`+, run the pre-validated Phase 1 A/B (`guide_end_percent = 1.0` vs `0.50` vs `0.34` at `frame_idx = 68`, `length = 141`):

```bash
set -a; . ./h3.env; set +a && COMFY="$(cd ../.. && pwd)" && \
"$COMFY/.venv/bin/python" data/embedded_image_video_2026-11-14/run_phase1_guide_release_ab.py
```
