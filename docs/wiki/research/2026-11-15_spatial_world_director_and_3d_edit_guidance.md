# 3D Semantic Velocity Steering, Modality Decoupling, & Spatial World Director (2026-11-15)

Status: **Verified Architecture & Showcase Study (`49c0d57a`)** — documents the 3D `(x, y, t)` semantic velocity steering, joint audio-video modality decoupling, procedural `3f` sound cues, 2D+time sprite trajectory kinematics, and interactive ComfyUI Spatial World Director + HUD telemetry stack, compared strictly against **native ComfyUI core** (`BasicGuider`, `CFGGuider`, `SetLatentNoiseMask`, `ConditioningSetTimestepRange`, `MiniMaxH3AddGuide`, `MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`) and backed by paired `141`-frame (`5.875 s` @ `24.0 fps`) 6-step Pika H3 Flex GPU measurements in `$H3_OUTPUT_DIR/Video/spatial_director_2026-11-15/metrics.json`.

For daily node parameter tuning and recipe workflows, start at [`../edit_guidance.md`](../edit_guidance.md) and [`../control_vectors.md`](../steering.md). For the desktop web-app architecture, full masking compatibility audit, sprite D-Pad analysis, and end-to-end UX/code diagrams, see [`2026-11-16_spatial_director_desktop_app_architecture.md`](2026-11-16_spatial_director_desktop_app_architecture.md).

---

## 1. Executive Summary: What Native ComfyUI Core Gives You vs. What We Built

MiniMax H3 denoises **joint 3D video (`[B, 24, T_lat, H/16, W/16]`) and `40 Hz` audio (`[B, 32, T_a]`)** inside a single packed DiT sequence (`comfy/ldm/minimax/model.py::PackedLayout`), governed by two asymmetric rectified-flow schedules (`shift_video = 12.0`, `shift_audio = 3.0`) under 6-step Pika H3 Flex `sample_lcm`. Native ComfyUI core exposes only scalar single-branch guidance (`BasicGuider` / `CFGGuider`), flat latent masks (`SetLatentNoiseMask`), and `t >= 0` pinned reference media (`MiniMaxH3AddGuide`, `MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`).

Commit `49c0d57a` completed the full 3D spatiotemporal and interactive authoring loop atop our custom steering stack:

1. **Zero-extra-pass 3D `(x, y, t)` semantic velocity steering (`MiniMaxH3EditGuider` + `MiniMaxH3PromptEdit`)**: Evaluates `[original, edited]` at matched `(x_t, sigma)` inside a `CFGGuider` subclass (`edit_guidance.py:215`), splits `(v, a)` across both `NestedTensor` and packed 3D `[B, 1, C_flat]` latents, and applies a `[F, H, W]` velocity mask pooled over H3's `(16 px, 4-frame causal)` latent cell geometry. Outside the mask at step `0`, velocity leak from `[edited]` is **strictly `0.0`**; on later steps, unmasked tokens still attend bidirectionally to edited tokens so cast light, reflections, and shadows stay physically coherent across the boundary.
2. **Modality-decoupled steering & `40 Hz` temporal audio gating**: Supports `streams = "both" | "video_only" | "audio_only" | "independent"` (`audio_scale`) with zero extra DiT forward passes, plus linear temporal envelope projection (`t_env = raw.amax(dim=(1, 2))` -> `40 Hz` `t_aud`) when driven by a 3D `[F, H, W]` (`F > 1`) mask (`edit_guidance.py:326-335`).
3. **1-pass `3f`-snapped acoustic impulses (`MiniMaxH3SoundCues`)**: Synthesizes clean `cond_audio` impulse islands aligned to exact `3`-frame (`125 ms` = `5` audio tokens) boundaries over a `-28.0 dBFS` noise floor and drops them mid-denoise (`audio_end_percent = 0.60`, `MiniMaxH3ReleaseReferences`) so native foley blossoms cleanly without synthetic waveform bleed.
4. **2D+time actor kinematics (`MiniMaxH3AppendSpriteTrajectory`)**: Renders `<Picture 1>` as a lightweight kinematic `<Video 1>` sprite path (waypoints, hops, locked `face=left|right` backpedaling) released at `video_end_percent = 0.33` (crossing the unshifted flow midpoint `b = 0.50` under `shift = 12.0`) while keeping `<Picture 1>` active through `end_percent = 1.0`.
5. **Interactive Viewport Stage & Telemetry HUD (`MiniMaxH3SpatialDirector`, `MiniMaxH3DirectorHUD`, `web/h3_spatial_director.js`)**: Provides an in-node ComfyUI stage (`Reality Portal`, `Genie + Targets`, `Split Wipe`, `24 fps` frame scrubber, draggable cursor/portal reticles, Genie D-pad `◀ ▲HOP ▼ ▶`) compiling straight to `[F, H, W]` masks, `3f` sound cues, sprite trajectories, and synchronized HUD / Split-A/B diagnostic videos.

---

## 2. Native ComfyUI Core vs. Spatial World Director: 5-Axis Architectural Comparison

Every contrast below targets stock ComfyUI primitives verified directly on disk.

| Architectural Axis | Native ComfyUI Core | This Pack (`MiniMaxH3EditGuider` + `MiniMaxH3SpatialDirector`) |
| :--- | :--- | :--- |
| **1. Prompt Steering & Anchored Composition** | `BasicGuider` (`comfy_extras/nodes_custom_sampler.py:801`) runs a single positive prompt (`cfg = 1.0` on Pika LCM); `CFGGuider` (`comfy/samplers.py:1188`) contrasts positive vs. negative (`uncond`) globally. Editing prompt clauses changes `Qwen3-VL` token offsets at step `0`, re-rolling actors, camera, and layout. `ConditioningSetTimestepRange` (`ComfyUI/nodes.py:305`) hard-swaps whole conditioning dicts across step spans with no semantic difference vector or `0 < scale < 1` interpolation. | `EditGuider` (`edit_guidance.py:215-465`) computes `x0_orig + eff_scale * (x0_edit - x0_orig)` at matched `(x_t, sigma)` with `first_step` structural anchoring (`0`, `1`, `2`) and `last_step` 1-pass tail release. `MiniMaxH3PromptEdit` (`edit_guidance.py:480`, `h3_prompt_edit.py:145`) rewrites only unique `old -> new` phrases while refusing to corrupt `<d>...</d>` dialogue, `(S1)` tags, or `<Picture N>` / `<Subject N>` markers. |
| **2. Spatial + Temporal Edit Localization (`(x, y, t)`)** | Stock `SetLatentNoiseMask` (`ComfyUI/nodes.py:1567`) overwrites `noise_mask` with a flat 5D tensor: on H3 `NestedTensor(video, audio)` latents this forces the sampler to pad `1`s onto the audio stream (silently regenerating the soundtrack) and trilinearly interpolates masks across H3's `[1, 4, 4, 4, 4]` causal clip boundaries ([`h3_audio_freeze.md`](../core/h3_audio_freeze.md)). Even when properly chunked, `noise_mask` freezes pixels mechanically instead of guiding semantic creation inside moving masks. | `EditGuider._prepare_masks_for_step` (`edit_guidance.py:277-340`) center-crops continuous masks (`_fit_crop_hw`, `edit_guidance.py:164`), max-pools space onto `(H/16, W/16)` (`_pool_spatial_continuous`, `edit_guidance.py:186`), and causally `amax`-pools time over `FRAME_PER_TOKEN` (`_pool_causal_temporal`, `edit_guidance.py:205`). Unmasked regions follow `[original]` directly on step `0` (`v_m = 0`), while bidirectional DiT self-attention on steps `1..5` lets unmasked surroundings catch real cast light and reflections without paste seams. |
| **3. Joint Video / Audio Modality Control** | `CFGGuider` scales the packed joint AV prediction with one scalar. Calm visual room tokens actively muffle `soundscape:` changes, while adding storm/explosion text to `detailed_description` alters the visuals. Stock `MiniMaxH3AddGuide` (`comfy_extras/nodes_minimax_h3.py:182`) pins `cond_audio` at `AUDIO_COND_TIMESTEP = 1.0` (`comfy/ldm/minimax/model.py:35, 747-755`) on a `40 Hz` 1D timeline (zero `(x, y)` spatial coordinates), and holding synthetic `.wav` guides late bleeds synthetic transients into output audio. | `EditGuider.predict_noise` splits `(v, a)` on both `NestedTensor` (`edit_guidance.py:409-429`) and `pack_latents` 3D `[B, 1, C_flat]` tensors via `inner_model.latent_shapes` (`edit_guidance.py:394-399, 431-465`) in **0 extra DiT passes**, supporting `"both"`, `"video_only"`, `"audio_only"`, and `"independent"` (`audio_scale`). 3D `[F, H, W]` (`F > 1`) masks project `t_env = raw.amax(dim=(1, 2))` onto `t_aud` (`40 Hz`) so audio deltas swell only during active spatial events (`edit_guidance.py:331-335`). `MiniMaxH3SoundCues` (`sound_cues.py:337`) emits zero-file `3f`-snapped transients released at `audio_end_percent = 0.60`; `MiniMaxH3FreezeAudio` (`audio_freeze.py:263`) preserves external tracks via `NestedTensor(video=1, audio=0)`. |
| **4. Actor Locomotion & Reference Conditioning** | `MiniMaxH3ImageToVideo` (`comfy_extras/nodes_minimax_h3.py:132`) anchors full `first_frame` / `last_frame` bookends; `MiniMaxH3ReferenceToVideo` (`comfy_extras/nodes_minimax_h3.py:259`) takes static stills or pre-rendered external videos held across the whole schedule (`1.0`). | `MiniMaxH3AppendSpriteTrajectory` (`sprite_trajectory.py:492`) builds a zero-file `<Video 1>` kinematic guide directly from `<Picture 1>` (position, scale, parabolic `hop`, `face=left|right` backpedal) released at `video_end_percent = 0.33` (`MiniMaxH3ReleaseReferences`, `steering/nodes.py:2940`) so H3 renders 3D gait and limb articulation without cutout bleed. Shared `[original, edited]` prefixes encode once via [`reference_encode.py`](../../../reference_encode.py). |
| **5. Authoring UI & Verification Telemetry** | Disconnected float/int widgets and JSON coordinates; bare `VAEDecode` output with no timeline overlay of active masks, sound triggers, or matched-seed controls. | `MiniMaxH3SpatialDirector` (`spatial_director.py:456`) + `web/h3_spatial_director.js` embed an interactive HTML5 canvas stage (`Reality Portal`, `Genie + Targets`, `Split Wipe`, Genie D-Pad, `24 fps` scrubber). `MiniMaxH3DirectorHUD` (`spatial_director.py:1271`) burns crisp reticle rings, cursor clicks, Genie key caps, multi-lane sequencer bars, and optional same-seed `baseline_images` wipes directly into verification MP4s. |

---

## 3. Under the Hood: Math, Causal Pooling, Modality Split, & Flow Timing

### 3.1 Dual-Branch Velocity Extrapolation & Zero-Extra-Pass AV Split

Because Pika H3 Flex uses rectified flow (`x0_hat = x - sigma * v` inside `comfy.samplers.CONST.calculate_denoised`), linear extrapolation between `[original]` (`conds["negative"]`) and `[edited]` (`conds["positive"]`) clean predictions at identical `(x_t, sigma)` is algebraically equivalent to steering the velocity field `v` (with H3's shifted audio carry cancelling identically in `out[1] - out[0]`):

```text
v_out = v_orig + (eff_v * v_m) * (v_edit - v_orig)
a_out = a_orig + (eff_a * a_m) * (a_edit - a_orig)
```

- **Matched `(x_t, sigma)`**: `EditGuider.predict_noise` (`edit_guidance.py:408`) calls `comfy.samplers.calc_cond_batch(self.inner_model, [original, edited], x, timestep, model_options)`. Only `MiniMaxH3Model`'s packed layout (`PackedLayout`) differs between the two branches (`original` and `edited` are tagged via `h3_edit_pass` in `EditGuider._prepare`, `edit_guidance.py:342-378`).
- **Packed 3D AV splitting (`edit_guidance.py:393-465`)**: Live `CFGGuider.sample` calls `comfy.utils.pack_latents` to flatten `(video, audio)` into `[B, 1, C_flat]`. Using `inner_model.latent_shapes`, `EditGuider` slices `packed_split = math.prod(latent_shapes[0][1:])`, unpacks `v` as `(B, 24, T_lat, H/16, W/16)` and `a` as `(B, 32, T_a)` (`edit_guidance.py:436-465`), applies distinct scalar gains (`eff_v`, `eff_a`) and continuous masks (`v_m`, `a_m`), and repacks without launching a third DiT pass.

### 3.2 Continuous `[F, H, W]` Spatial-Temporal Pooling & `40 Hz` Audio Envelope Projection

Inside `EditGuider._prepare_masks_for_step` (`edit_guidance.py:277-340`), an incoming pixel mask `[F, H, W]` (normalized clamped `[0.0, 1.0]`, `_normalize_edit_mask` at `edit_guidance.py:134`) is lowered once per latent geometry onto H3's native video and audio token grids:

1. **Spatial alignment (`_fit_crop_hw` + `_pool_spatial_continuous`, `edit_guidance.py:164-203`)**: Aspect-matches (`center` crop if aspect differs) and runs `F.adaptive_max_pool2d(..., (h_lat, w_lat))` so even thin `16 px` spatial portals or target rings reach full `1.0` peak gain inside their active DiT patch cells (`2x2` `32 px` DiT patches over `16x16` VAE cells).
2. **Causal temporal pooling (`_pool_causal_temporal`, `edit_guidance.py:205-212`)**: H3's causal Video VAE packs frames into repeating `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)` latent steps (`edit_guidance.py:37`, [`h3_frame_rules.md`](../core/h3_frame_rules.md)). Step `0` covers `[f0]`, step `1` covers `[f1..f4]`, step `2` covers `[f5..f8]`, and every `17`-frame block repeats `1 + 4 + 4 + 4 + 4`. `_pool_causal_temporal` takes `amax` over each latent bucket's exact pixel span (`max`), preserving instantaneous onset inside interactive click targets without smearing across `17`-frame standalone boundaries.
3. **40 Hz temporal audio projection (`edit_guidance.py:318-335`)**:
   - **Static `[1, H, W]` masks (`t_in == 1`)**: `a_m` is set to `None` (`edit_guidance.py:318-325`) because a static region crop should not automatically gate clip-wide audio.
   - **Temporal `[F, 1, 1]` envelopes (`h_in == 1 and w_in == 1`, `edit_guidance.py:309-315`)**: Linearly interpolated from `t_in` video frames to `t_aud` (`40 Hz`) audio tokens.
   - **Dynamic 3D `[F, H, W]` masks (`t_in > 1`, `edit_guidance.py:326-335`)**: Computes per-frame spatial peak activity `t_env = raw.amax(dim=(1, 2)).reshape(1, 1, t_in)` (`raw` clamped pre-downsample) and linearly interpolates `t_env` onto `(1, ..., t_aud)` (`F.interpolate(..., size=t_aud, mode="linear", align_corners=False)`). Result: when an interactive target only activates on frames `[24..66]` and `[78..126]`, audio velocity steering under `"independent"` or `"audio_only"` turns on during those windows and drops to `0.0` elsewhere, letting `[original]` room tone play undisturbed between events.

### 3.3 Why Step `0` Self-Attention Creates Realistic Spill Without Unmasked Drift

At step `0` (`sigma = 1.0`), pure Gaussian noise is partitioned by `v_m`: tokens where `v_m == 0` execute `v0` (zero direct injection of `[edited]`), while tokens where `v_m > 0` execute `v0 + scale * (v1 - v0)`. Unlike post-render image compositing or frozen latent replacement, **both regions coexist inside H3's full-sequence DiT self-attention from step `0` through step `5`**. Once the edited portal or target generates luminous ice crystals or molten embers at coarse steps `0..2`, surrounding unmasked stone flagstones attend to those bright keys on steps `1..5` and naturally render bounce light, specular sheen, and heat haze while unedited architecture outside the mask stays anchored to the baseline layout.

### 3.4 Asymmetric Schedules (`shift_video = 12.0`, `shift_audio = 3.0`) & Reference Release (`0.33` vs. `0.60`)

Under 6-step Pika H3 Flex (`sigmas = [1.0, 0.972973, 0.923077, 0.80, 0.571429, 0.333333, 0.0]`):

- **`first_step = 0`**: Shapes silhouette, geometry, and hard portal/eruption boundaries from initial noise. Best for Reality Portals and Point-and-Click World Targets where spatial masks (`v_m`) already lock outside structure.
- **`first_step = 1`**: Step `0` (`2.70%` `x0_hat_0`) locks global camera, actor pose, and props; steps `1..5` re-light the scene or transform weather without mask geometry.
- **`first_step = 2` (or `streams = "audio_only"`)**: At step `2`, `sigma_video = 0.923` has committed global scene composition while `sigma_audio = 0.750` leaves soundscape and music steerable (`edit_guidance.py`, [`edit_guidance.md`](../edit_guidance.md)).
- **`MiniMaxH3ReleaseReferences` thresholds (`steering/nodes.py:2940`)**:
  - `video_end_percent = 0.33`: With `shift_video = 12.0`, steps `[0, 1]` move `t_v` from `1.0 -> 0.973 -> 0.923`, crossing the unshifted flow midpoint `b = 0.50` (`step 2` is evaluated at fraction `2 / 6 = 0.3333...`). Dropping the synthetic `<Video 1>` sprite card at `0.33` transfers trajectory, facing, and jump apex while giving steps `2..5` back to live 3D photorealism ([`2026-11-12_sound_cues_dpad_and_033_release_law.md`](../steering/research/2026-11-12_sound_cues_dpad_and_033_release_law.md)).
  - `audio_end_percent = 0.60`: Combined with `MiniMaxH3EditGuider(streams="video_only")` and `0`-vs-`K` `MiniMaxH3SoundCues`, holding the synthetic impulse cue slightly past midpoint focuses early motion kinematics onto exact `3f` timestamps while releasing `x_audio` onto clean native foley synthesis (`r ~ 0` waveform bleed vs. synthetic trigger).

---

## 4. Measured `141`-Frame GPU Showcases (`$H3_OUTPUT_DIR/Video/spatial_director_2026-11-15/metrics.json`)

Both shipped graphs compile on H3's exact **dual-clock grid** (`length = 141 = 39 + 51 * 2`, `h3_rules.py:118::on_both_clocks`, `5.875 s` @ `24.0 fps`), at `1344x768`, 6-step Pika H3 Flex (`RandomNoise = 730451892`, `shift_video = 12.0`, `shift_audio = 3.0`, `scale = 2.0` shipped default; measured showcase batch rendered at `scale = 2.2`, `first_step = 0` against same-seed `scale = 0.0` baselines).

### 4.1 Showcase 01 — Draggable Reality Portal (`portal_path`)

- **Workflow & prompt**: [`workflows/h3_spatial_portal_pika_api.json`](../../../workflows/h3_spatial_portal_pika_api.json), [`prompt_bank/t2va_spatial_reality_portal.txt`](../../../prompt_bank/t2va_spatial_reality_portal.txt).
- **Director configuration (`h3_rules.py:382-410`)**: Circular `REALITY LENS` sweeping left-to-right across `(0.18, 0.52) -> (0.50, 0.50) -> (0.82, 0.52)` over frames `0 -> 70 -> 140` (`rx = 0.24..0.26`, `feather = 0.08`, `streams = "independent"`, `audio_scale = 1.0`).
- **Observed visual transformation**: Same-seed baseline maintains sunlit golden-hour Mediterranean limestone arches and tranquil fountain throughout. The directed render opens a traveling bioluminescent midnight ice cavern inside the sweeping cyan lens (`in_mask` arches frosted with glowing cyan crystals and cold mist) and closes smoothly back to dry sunlit sandstone behind the portal's wake.
- **Measured localization & acoustic delta (`metrics.json`)**:
  - **Pixel RGB L1 (`0..255` vs. same-seed `scale = 0.0`)**: `in_mask_mae = 48.086` (`m >= 0.5`, `11.22%` of spatiotemporal pixels), `out_mask_mae = 27.883` (`m <= 0.05`, `86.07%` of pixels), `all_mae = 30.167`, **`localization_ratio = 1.72x`** (clean background stays sunlit while cast blue/cyan portal rim-light reflects on nearby pavement).
  - **Decoded `32 kHz` audio (`5.856 s`)**: Baseline overall `-40.37 dBFS` RMS / `-15.89 dBFS` peak (`0–2 s: -38.44 dBFS`, `3–5 s: -41.31 dBFS`) vs. Directed overall **`-33.30 dBFS` RMS** / `-16.42 dBFS` peak (`+7.07 dB` crystal-chime and sub-zero wind sweep across the courtyard take; `0–2 s: -33.55 dBFS`, `3–5 s: -33.21 dBFS`).
- **Deliverables (`$H3_OUTPUT_DIR/Video/spatial_director_2026-11-15/`)**: `01_draggable_reality_portal_hud.mp4`, `01_draggable_reality_portal_split_ab.mp4`, `sheet_01_portal.png`, `raw/01_portal_baseline_00001-audio.mp4`, `raw/01_portal_directed_00001-audio.mp4`.

### 4.2 Showcase 02 — Genie D-Pad + Point-and-Click World Targets (`point_targets`)

- **Workflow & prompt**: [`workflows/h3_world_targets_genie_pika_api.json`](../../../workflows/h3_world_targets_genie_pika_api.json), [`prompt_bank/ref2va_genie_world_targets.txt`](../../../prompt_bank/ref2va_genie_world_targets.txt).
- **Director + multi-node chain (`h3_rules.py:411-445`)**:
  1. `<Picture 1>` = `ref_signal/tortoise_pic1.png` held at `end_percent = 1.0`.
  2. `<Video 1>` = `MiniMaxH3AppendSpriteTrajectory` (`canvas = 384x224`, `sprite_scale = 0.31`) driving waypoints `f0 (0.24, 0.76) -> f54 (0.50, 0.76) -> f96 (0.68, 0.76) -> f140 (0.54, 0.76)` (`face = right` backpedal on `f96..140`) plus an airborne jump `hop` on `f63..78` (`lift = 0.25`), released at `video_end_percent = 0.33`.
  3. `MiniMaxH3SoundCues` (`snap_to_3f = True`, `floor_dbfs = -28.0`) fires `24: slam @1.20` and `78: hop @1.20` on the edited branch only (`audio_end_percent = 0.60`).
  4. `MiniMaxH3EditGuider` (`streams = "video_only"`, `scale = 2.2`, `first_step = 0`) triggers glowing arcane/crystal pavement eruptions inside Target A (`f24..66`, `(0.24, 0.54)`, `radius = 0.19`, `ramp = 5`) and Target B (`f78..126`, `(0.76, 0.54)`, `radius = 0.19`, `ramp = 5`).
- **Observed visual transformation**: Both baseline and directed renders share identical seed, `<Picture 1>`, and `<Video 1>` locomotion—walking right, hopping cleanly in mid-air at `f63..78`, and backpedaling leftward while facing right—proving sprite locomotion stays intact under velocity masking. In the directed pass, Target A erupts golden-amber embers on `f24..66` (`f42` peak) and Target B erupts cyan-emerald crystal pillars on `f78..126` (`f102` peak), while intervening cobblestone slabs remain undisturbed.
- **Measured localization & acoustic delta (`metrics.json`)**:
  - **Pixel RGB L1 (`0..255` vs. same-seed baseline with identical `<Video 1>` trajectory, `scale = 0.0`, and zero SoundCues)**: `in_mask_mae = 26.677` (`m >= 0.5`, `3.77%` of pixels), `out_mask_mae = 9.210` (`m <= 0.05`, `94.53%` of pixels), `all_mae = 10.003`, **`localization_ratio = 2.90x`**.
  - **Decoded `32 kHz` audio (`5.856 s`)**: Baseline overall `-42.76 dBFS` RMS / `-6.68 dBFS` peak vs. Directed overall `-36.02 dBFS` RMS / `-7.39 dBFS` peak. Locally during active target cue windows: **Target A (`1.00–1.80 s`, `f24..43`) rises `+11.09 dB`** (`-59.37 -> -48.28 dBFS`) and **Target B (`3.25–4.05 s`, `f78..97`) rises `+4.22 dB`** (`-35.21 -> -30.99 dBFS`) from native foley synthesis reacting to the released `3f` impulses.
- **Deliverables (`$H3_OUTPUT_DIR/Video/spatial_director_2026-11-15/`)**: `02_genie_point_and_click_targets_hud.mp4`, `02_genie_point_and_click_targets_split_ab.mp4`, `sheet_02_targets.png`, `raw/02_targets_baseline_00001-audio.mp4`, `raw/02_targets_directed_00001-audio.mp4`.

---

## 5. Shipped Code Inventory & Verification Suite

| Component | Tracked File & Symbol / Entrypoint | Role |
| :--- | :--- | :--- |
| **Guider & `[F, H, W]` + Audio Gating** | [`edit_guidance.py::EditGuider`](../../../edit_guidance.py#L215), [`edit_guidance.py::MiniMaxH3EditGuider`](../../../edit_guidance.py#L519), [`edit_guidance.py::MiniMaxH3PromptEdit`](../../../edit_guidance.py#L480) | Zero-extra-pass AV velocity differential, `streams` split, causal `[1, 4, 4, 4, 4]` mask pooling, `40 Hz` `t_env` audio gating |
| **Interactive Compiler & Telemetry HUD** | [`spatial_director.py::MiniMaxH3SpatialDirector`](../../../spatial_director.py#L456), [`spatial_director.py::MiniMaxH3DirectorHUD`](../../../spatial_director.py#L1271), [`web/h3_spatial_director.js`](../../../web/h3_spatial_director.js), [`h3_rules.py`](../../../h3_rules.py#L380) | Interactive ComfyUI 2D+time canvas UI, cosine `[F, H, W]` mask / `3f` sound-cue / sprite-trajectory compilation, HUD + Split-A/B overlays |
| **1-Pass `3f` Audio & 2D Sprite Kinematics** | [`sound_cues.py::MiniMaxH3SoundCues`](../../../sound_cues.py#L337), [`sprite_trajectory.py::MiniMaxH3AppendSpriteTrajectory`](../../../sprite_trajectory.py#L492), [`steering/nodes.py::MiniMaxH3ReleaseReferences`](../../../control_vector_nodes.py#L4666) | Zero-file `125 ms` audio impulse islands, `<Picture 1>` silhouette trajectory rendering, `0.33` / `0.60` midpoint reference release |
| **Generated API Workflows** | [`workflows/h3_spatial_portal_pika_api.json`](../../../workflows/h3_spatial_portal_pika_api.json), [`workflows/h3_world_targets_genie_pika_api.json`](../../../workflows/h3_world_targets_genie_pika_api.json) | Reproducibly generated by [`workflows/build_workflows.py`](../../../workflows/build_workflows.py) from [`workflows/h3_config.py`](../../../workflows/h3_config.py) |

Offline and graph-generation regression suite:

```bash
<comfy-venv-python> bench/check_edit_guidance.py
<comfy-venv-python> bench/check_spatial_director.py
<comfy-venv-python> bench/check_widget_deviations.py
<comfy-venv-python> bench/check_node_ids.py
```
