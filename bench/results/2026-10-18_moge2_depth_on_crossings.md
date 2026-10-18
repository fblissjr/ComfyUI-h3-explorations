# HyDen MoGe-2 depth on crossing frames: ownership, camera field of view, and cost (2026-10-18)

lane: masked video to video
verdict: no render; measured on saved preview masks and four spans of the `fun` source clip on CPU (`CUDA_VISIBLE_DEVICES=""`) while the NVIDIA GeForce RTX 4090 card was held by the ComfyUI server. Across all 16 frames of the four spans, MoGe-2's metric depth puts the people and hands in the visible front-to-back order; on `shot_c_1306b` (`1409-1417`) it drops contested shared pixels from `186,923 / 188,179` (`99.33%`) to `3 / 188,179` (`0.0016%`) and clips `53,938` px of the back subject's part spill off the front subject (`22,738` px on frame `1415`), and on `shot_c_0604c` (`755-780`) it drops contested shared pixels from `1,520 / 1,670` (`91.02%`) to `0 / 1,670` (`0.00%`). Single-frame recovered camera field of view holds within `1.54`–`3.17` deg on medium shots (`755-780`, `1022-1038`) and swings by `46.62` deg on a tight head close-up (`3060-3109`) and `74.86` deg when a foreground torso fills the frame (`1409-1417`), without affecting within-frame depth ordering.

## 1. What was measured, when, and on what hardware

- **Dates**: `2026-10-17` (four-span survey, session `gem-matte`) and `2026-10-18` (end-to-end chain ownership tests on `shot_c_1306b` and `shot_c_0604c`, session `gem-depth`).
- **Machine and card state**:
  - **GPU in the machine**: `NVIDIA GeForce RTX 4090` (`24 GiB`), held by the running ComfyUI server during both sessions.
  - **Device used for all runs in this record**: **CPU** (`13th Gen Intel(R) Core(TM) i9-13900K`, 24 cores / 48 threads, `CUDA_VISIBLE_DEVICES=""`, `--device cpu`, `torch.float32`). **GPU wall time and peak VRAM on the RTX 4090 were not measured in these runs.**
- **Model and checkpoint**:
  - `vendor/meta_moge2/hyden_moge2.py::load_hyden_moge2`
  - Checkpoint: `hyden_mogev2_metric_point_vitl_fp32_f1066593896.pth` under `$H3_MODEL_STORE/facebook_hyden-mogev2-metric-point/` (`1,498,228,429` bytes / `1.50 GB` `fp32` on disk, `sha256: 0ca6e0db7eb3510ef43f14a562a9d221be746b3007f54d1a0fa5bf80fd9bf6f3`, `953` `state_dict` keys = `738` parameter tensors + `215` buffers).
  - Input resolution: `1024x768` canvas (`h3_config.MASKED_JOB_CANVAS`), `float32` RGB in `[0.0, 1.0]`; internal `DINOv2 ViT-L/14` backbone token grid is `518x518` (`37x37` patches) coupled with the multi-scale `CNN_Large` pyramid at `1024x768`.

---

## 2. Commands run

### Four-span depth and camera intrinsics survey (`2026-10-17`)

```bash
CUDA_VISIBLE_DEVICES="" <python> bench/measure_moge2_depth.py \
  --source "$H3_INPUT_DIR/mrpop/fun_0000_6394_24p_sound_in_step.mov" \
  --frames 755,764,772,780 \
  --size 1024x768 \
  --device cpu \
  --out-dir data/fun/moge2/span_0755_0780
```
(Repeated with `--frames 1022,1027,1032,1038` into `data/fun/moge2/span_1022_1038`, `--frames 1398,1405,1411,1418` into `data/fun/moge2/span_1398_1418`, and `--frames 3060,3076,3092,3109` into `data/fun/moge2/span_3060_3109`.)

### Chain `--depth` ownership and clipping runs (`2026-10-18`)

```bash
CUDA_VISIBLE_DEVICES="" <python> bench/shot_to_load.py job \
  --shots data/fun/shots.json \
  --plan data/fun/plan.json \
  --preview data/2026-10-18_shot_c_1306b_job \
  --subject lead \
  --load lead_fp_1393 \
  --first 1393 --last 1417 \
  --depth "dad crosses in front of lead on 1409-1417" \
  --by gem-depth
```
Followed by the `files` command produced by `bench/load_chain.py::capture_args` (`bench/capture_masked_run.py files ... --depth <capture>/depth.npz`), and repeated on `data/2026-10-18_shot_c_0604c_job` (`second`, frames `755-780`, overlap frames `759-764, 775`).

---

## 3. Cost: measured on CPU, and what is reasoned (not yet measured) for the RTX 4090

| Metric | Value | Provenance |
|---|---|---|
| **Checkpoint size on disk** | `1,498,228,429` bytes (`1.50 GB` / `1.40 GiB` in `fp32`; `0.75 GB` / `0.70 GiB` in `fp16`) | **Measured** (`stat` on `hyden_mogev2_metric_point_vitl_fp32_f1066593896.pth`). |
| **CPU weight load (`load_hyden_moge2`, `device="cpu"`)** | `1.57 s` | **Measured** (`2026-10-18`, `i9-13900K`, `CUDA_VISIBLE_DEVICES=""`, warm disk cache). |
| **CPU inference per `1024x768` frame (`float32`)** | `3.97 s/frame` (`3.90`–`4.04 s` across 3 frames; `0.25 fps`) | **Measured** (`2026-10-18`, `i9-13900K`, `CUDA_VISIBLE_DEVICES=""`, `model.infer` on `1x3x768x1024`). |
| **CPU peak RSS (`float32`, `1024x768`)** | `3,842.8 MiB` (`3.75 GiB`) | **Measured** (`resource.getrusage(RUSAGE_SELF).ru_maxrss`). |
| **GPU inference and VRAM on the machine's RTX 4090** | **Not yet measured on the RTX 4090** (estimated at `~50 ms/frame` / `~18 s` per 362 frames and `~1.8 GiB` VRAM in `fp16`, or `~3.1 GiB` in `fp32`) | **Reasoned** from the `0.70 GiB` `fp16` (`1.40 GiB` `fp32`) weights plus `DINOv2 ViT-L/14` `518x518` and `CNN_Large` `1024x768` activations (`internal/gemini/2026-10-17_perception_design.md`); card was held by the ComfyUI server during both sessions. |
| **`depth.npz` size on disk per overlap frame (`1024x768`, `float16`)** | `~1.57 MB/frame` (`14.3 MB` for `shot_c_1306b`'s 9 overlap frames `1409-1417`) | **Measured** on `<capture>/depth.npz` written by `bench/measure_moge2_depth.py`. |

Because `bench/shot_to_load.py::overlap_frames_of` runs `bench/measure_moge2_depth.py` **only on frames in the load where two subject tracks or a keep-out overlap**, a load with `0` overlap frames runs `0` forward passes, and a crossing load runs only its overlap frames (`9` frames on `shot_c_1306b`, `7` frames on `shot_c_0604c` `755-780`, `18` frames across all of `604-1009`).

---

## 4. Pixel ownership and part-spill clipping on `shot_c_1306b` and `shot_c_0604c` (`2026-10-18`)

Constants in `bench/capture_masked_run.py`:
- `DEPTH_CORE_ERODE_PX = 5` (reasoned: strips 2D mask edge before reading a subject's interior median depth).
- `DEPTH_CORE_MIN_PX = 64` (reasoned: falls back to un-eroded exclusive pixels when eroded core is smaller).
- `DEPTH_WIN_M = 0.05` (reasoned from the four-span survey below: minimum median depth separation in meters for one subject to beat another on a frame; smallest true front-to-back separation measured was `0.0515 m` on frame `3076`).
- `DEPTH_BOUNDARY_SLACK_M = 0.02` (reasoned: boundary tie-break slack toward the front subject on shared silhouette pixels).

### 1. Hall crossing `shot_c_1306b` (`lead` and `third`, overlap frames `1409-1417`, `1024x768`)

- **Shared pixels across `1409-1417` (`9` frames)**: `188,179` px.
- **Without `--depth`**:
  - `OWNER_CONTESTED`: **`186,923 / 188,179` px (`99.33%`)**.
  - `third` (dad) owned: `1,040` px (`0.55%`); `lead` owned: `216` px (`0.11%`).
  - `lead`'s Sapiens2 face-part mask (`shot_c_1306b_lead_part_saved_on_track_8px.mkv`) spilled onto `third`'s foreground back/shoulder inside the `8` px track halo (`22,738` px clipped on frame `1415`, `53,938` px across `1409-1417`).
- **With `--depth`** (`z_third = 0.131`–`0.538 m` in front of `z_lead = 0.237`–`0.966 m` on every frame `1409-1417`):
  - `OWNER_CONTESTED`: **`3 / 188,179` px (`0.0016%`)** (`settled_by_depth_px = 186,920`).
  - `third` (dad) owned: **`187,960 / 188,179` px (`99.88%`)**.
  - `lead` `parts_clipped_by_depth`: **`53,938` px clipped across `8` frames** (`22,738` px on frame `1415`), while preserving `15,413` px of `lead`'s own visible face/head behind `third` on `1415`.

### 2. Kitchen crossing `shot_c_0604c` (`second` and `lead`, frames `755-780`, overlap frames `759-764, 775`, `1024x768`)

- **Shared pixels across `759-764, 775` (`7` frames)**: `1,670` px.
- **Without `--depth`**:
  - `OWNER_CONTESTED`: **`1,520 / 1,670` px (`91.02%`)**.
  - Because `second`'s delivery row carries `restore=lead:whole`, `bench/assemble_delivery.py::restore_weight` previously bypassed `owners.npz` on `whole` restores and punched `lead`'s entire 2D track out of `second`'s rendered hair/head as `lead` walked behind her.
- **With `--depth`** (`z_second = 1.077`–`1.648 m` in front of `z_lead = 1.662`–`2.573 m` by `0.55`–`0.93 m` on every overlap frame):
  - `OWNER_CONTESTED`: **`0 / 1,670` px (`0.00%`)** (`settled_by_depth_px = 1,670`, `100.00%` to `second`).
  - In `bench/assemble_delivery.py::restore_weight`, `restore=lead:whole` is gated by `captures.owned` and `captures.in_front`, restoring `0` pixels of `lead` over `second`'s foreground head/hair on `759-764, 775`.

---

## 5. Camera field of view (`fov_x_deg`) stability across shots (`data/fun/moge2/*/record.json`)

| Span (`1024x768`) | Frames | `fov_x_deg` `[least, median, most]` | Spread (`max - min`) | Std (`deg`) | What the picture shows |
|---|---|---|---:|---:|---|
| `span_0755_0780` | `755, 764, 772, 780` | `[39.18°, 39.73°, 40.72°]` | `1.54°` | `0.57°` | Kitchen medium two-shot with visible walls and cabinets; single-frame FOV is steady. |
| `span_0748_0786` | `755, 759-764, 772, 775, 780` | `[38.75°, 39.75°, 41.87°]` | `3.12°` | `0.90°` | Same kitchen shot across all 10 sampled crossing frames. |
| `span_1022_1038` | `1022, 1027, 1032, 1038` | `[43.19°, 44.81°, 46.36°]` | `3.17°` | `1.22°` | Medium close-up of `lead` with `second` seated behind (`0.97`–`1.35 m` behind `lead`). |
| `span_1398_1418` | `1398, 1405, 1411, 1418` | `[43.50°, 48.93°, 57.88°]` | `14.38°` | `5.88°` | Hall crossing: as `third` crosses close to the lens (`1398 -> 1411`), single-frame FOV shifts by `14.38°` (`74.86°`, `41.86°`–`116.72°`, across `1409-1417` when `third`'s back fills the frame on `1416`). |
| `span_3060_3109` | `3060, 3076, 3092, 3109` | `[36.66°, 48.99°, 83.28°]` | **`46.62°`** | `17.76°` | Tight head close-up of `lead` putting on sunglasses against a flat out-of-focus wall: single-frame `recover_focal_shift` swings from `83.28°` (`3060`) to `36.66°` (`3092`). |

### Why single-frame `fov_x_deg` swings on close-ups do not affect within-frame depth ordering

In `vendor/meta_moge2/hyden_moge2.py::HyDenMoGe.infer` (`lines 988-1009`), `recover_focal_shift(points, mask)` solves for two frame-wide scalars (`focal` and `shift`) and adds `shift` uniformly to every pixel's $Z$ coordinate on that frame (`depth = points[..., 2] + shift[..., None, None]`, `line 999`). Within any single frame $t$, `shift` cancels identically in $D_t(y_2, x_2) - D_t(y_1, x_1)$ and in $z_t(B) - z_t(A)$. Thus single-frame camera FOV cannot be fed raw frame-by-frame to a 3D mesh camera on tight close-ups, while within-frame front-to-back depth ordering is unaffected.
