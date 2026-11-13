# MiniMax H3 Frame, Latent-Step, and Audio-Clock Rules

Generalized architectural rules governing **all** MiniMax H3 lanes in this repo (`t2va`, `i2va`/`fl2va`, `ref2va`, guide-latents, procedural sound cues and sprite trajectories, frozen-audio song loops, single-frame image models, and `masked_v2v`). When prose and code disagree, the code is right ([`AGENTS.md`](../AGENTS.md), [`prose_measurements.md`](prose_measurements.md)).

- **Legal resolutions, 95-canvas table, & token counts**: [`h3_resolutions.md`](h3_resolutions.md) and [`resolution.py`](../resolution.py).
- **Prompt duration formatting & timestamps (`At MM:SS.mmm`)**: [`prompting.md`](prompting.md) and [`bench/grade_prompt_text.py`](../bench/grade_prompt_text.py).
- **Reference media policies & sizing**: [`h3_references.md`](h3_references.md) and [`reference_conditioning.py`](../reference_conditioning.py).
- **`masked_v2v`-specific extensions** (zoom boxes, `<Video 1>` landmark drawing, `BLUE-ONLY` step cleanup, Clip 3 hair-margin guards): [`masked_v2v/frames_rules.md`](masked_v2v/frames_rules.md).

---

## Quick Decision Table: Which `length` Should I Use?

| Workload / Lane | Legal `length` Rule | Canonical Values (`24 fps`) | Where Enforced |
| :--- | :--- | :--- | :--- |
| **Single-frame still / edit** (`MiniMax-H3-Image`, `P-H3-Image`, Pika `single_frame=True`) | **`1`** only (`h3_rules.py::SINGLE_FRAME`); bypasses all video duration rules and `17n + 5` | `1` frame | `h3_rules.py::is_single_frame`, [`guide_latent.py`](../guide_latent.py), [`ph3_image_nodes.py`](../ph3_image_nodes.py), [`h3_image_nodes.py`](../h3_image_nodes.py) |
| **Standalone 1-window video** (`t2va`, `i2va`, `fl2va`, `ref2va`, 1-window frozen song with no seam) | Any `17n + 5` in `5..362` (`h3_rules.py::snap_length`, `h3_rules.py::MAX_LENGTH`); trained vendor floor is `5.0 s` (`124`), though ComfyUI renders shorter `17n + 5` clips | `5, 22, 39, 56, 73, 90, 107, 124, ..., 345, 362` (`workflows/h3_config.py::LONG_LENGTH` = `362`; `h3_rules.py::reference_would_emit` stops at `345`) | `h3_rules.py::snap_length`, `loop_plan.py::stands_alone`, `comfy_extras/nodes_minimax_h3.py::align_frame_count` |
| **Audio-synced, chained, looped, or `masked_v2v` video** (multi-window loops, `audio_freeze_song`, all `masked_v2v` loads) | **Dual-clock `39 + 51m`** (`frames % 17 == 5` **and** `frames % 3 == 0`), where video VAE clips and `40 Hz` audio tokens both land on exact integers | `39, 90, 141, 192, 243, 294, 345` (`chain_lengths`: `141..345`; `context_frames`: `39` or `90`) | `h3_rules.py::on_both_clocks`, `h3_rules.py::chain_lengths`, `loop_plan.py::GRID`, `workflows/h3_config.py::LONG_WINDOW` |

---

## 1. Two Clocks, One Shared RoPE Timeline (`24 fps` Video vs. `40 Hz` Audio)

H3 is a joint audio-video Diffusion Transformer that packs text, visual/audio conditioning, audio latents, and video latents into **one unified 1D attention sequence** (`comfy/ldm/minimax/model.py::PackedLayout`), sharing a single temporal Rotary Position Embedding (`t`) axis:

- **Video pixel clock — fixed `24 fps`**: `h3_rules.py::FPS` (`24`), `comfy_extras/nodes_minimax_h3.py::FPS`. Core ComfyUI conditioning nodes (`MiniMaxH3ImageToVideo`, `MiniMaxH3ReferenceToVideo`) expose **no `fps` input** and assume every video input is `24.0 fps`. Passing un-resampled `30`/`60`/`23.976` fps footage (`force_rate=0`) shifts both DiT temporal RoPE coordinates and Qwen3-VL `<T.T seconds>` labels ([`h3_references.md`](h3_references.md); `bench/shot_to_load.py` and [`reference_conditioning.py`](../reference_conditioning.py) force/resample inputs to `24.0`).
- **Audio latent clock — fixed `40 Hz` (`25 ms` per token)**: Stereo `32,000 Hz` PCM encoded with a stride of `800` samples yields **`40` audio latent steps per second** (`25 ms` per step; `h3_rules.py::AUDIO_LATENT_RATE`, `comfy_extras/nodes_minimax_h3.py::AUDIO_LATENT_FPS`, `sound_cues.py::SAMPLES_PER_AUDIO_TOKEN`). Note: `25 ms` is the *step period*, corresponding to **`40 Hz`** (never `25 Hz`).
- **Clock conversion factor (`FRAME_RESCALE = 5 / 3`)**: `comfy/ldm/minimax/model.py::FRAME_RESCALE` (`5.0 / 3.0 = 40 / 24`) converts video frame indices into `25 ms` audio-token units (`1` audio token = `1.0` temporal RoPE unit; `1` pixel frame = `5/3` RoPE units; `3` pixel frames = exact `5` audio tokens = `125 ms`).

---

## 2. Video VAE: Independent `17`-Frame Causal Clips & `[1, 4, 4, 4, 4]` Latent Steps

H3 does not encode or mask a video tensor uniformly across its duration. Three nested temporal scopes act simultaneously in every render:

| Temporal Scope | Extent in Pixel Frames | Code & Exact Mechanism | What Is Coupled Here | Practical Consequence Across All Lanes |
| :--- | :--- | :--- | :--- | :--- |
| **1. Latent Step (`1` or `4` frames)** | Inside every `17`-frame clip: step `0` = **`1` frame**; steps `1..4` = **`4` frames each** | `comfy/ldm/minimax/model.py::FRAME_PER_TOKEN` = `(1, 4, 4, 4, 4)` (`1 + 4 + 4 + 4 + 4 = 17`; `loop_plan.py::STEP_CYCLE`). `comfy/ldm/minimax/model.py::mask_row_values` and `comfy/model_base.py::MiniMaxH3.scale_latent_inpaint` operate per `(latent_step, 32x32_patch)` | All `4` pixel frames packed inside a `4`-frame step share one noisy latent tensor `z` and one `denoise_mask` value | A temporal mask or inpaint blend **cannot** change mid-step across frames `[1..4]`. Any `1`-frame change or outlier inside a `4`-frame step shares tokens with its 3 neighbors (`video_mask.py::token_mask`). |
| **2. Video VAE Clip (`17` frames = `5` steps)** | **Encoder**: independent `17`-frame chunks (`0..16`, `17..33`, `34..50`, ...).<br>**Decoder**: overlapping `7`-step (`28`-frame) chunks. | `comfy/ldm/minimax/vae.py::encode_temporal` slices input `[0:T]` into independent `17`-frame clips (`clip_length`), repeats the final frame to pad a short tail clip to `17`, runs `EncoderFCN3D` with front-zero-padded `comfy/ldm/minimax/vae.py::CausalConv3d`, and drops `token_drop = 3` (`12` frames). `comfy/ldm/minimax/vae.py::decode_temporal` runs `ViT3DDecoder` over `5 + 2 = 7` latent steps. | **Zero cross-clip encoder memory** (`16 -> 17`, `33 -> 34`); strictly causal (`<= t`) inside each `17`-frame encode clip; locally bidirectional inside each `28`-frame decode chunk. | Every `17`th pixel frame (`0, 17, 34, 51, ...` = latent steps `0, 5, 10, 15, ...`) starts a fresh causal clip as a **standalone `1`-frame token** (identified as natural anchor/sink frames in [`archive/2026-09-19_in_context_reference_and_sinks.md`](archive/2026-09-19_in_context_reference_and_sinks.md)). The tail clip drops `3` padded steps (`12` frames), leaving `2` steps (`1 + 4 = 5` frames): **`17n + 5` frames -> `5n + 2` latent steps** (`comfy_extras/nodes_minimax_h3.py::video_latent_t`). |
| **3. DiT Sampling Window (`W` frames)** | Whole active render (`5..362` standalone; `39..345` dual-clock; `> 345` chained with `39` or `90` overlap) | `comfy/ldm/minimax/model.py::PackedLayout` and `comfy/ldm/minimax/model.py::DiTBlock` run unmasked packed self-attention over `[text | refs/cond | audio | video]` (with `MiniMaxH3Sol` applying sparse attention on later blocks/steps and dense self-attention on early steps + `dense_blocks`). | Fully bidirectional across all `5n + 2` video latent steps and all audio steps of the active window | Any unmasked source pixels, noisy frames, or stray reference/guide frames anywhere inside the active window propagate bidirectionally across the whole window. Chained windows only see their own `W` frames (`90` overlap context). |

### Key Implementation Laws From Section 2
- **Loop up, never closed-form (`h3_rules.py::snap_length`)**: Valid video frame counts follow `17n + 5` (`5, 22, 39, 56, 73, 90, 107, 124, ..., 345, 362`). Both `h3_rules.py::snap_length` and `comfy_extras/nodes_minimax_h3.py::align_frame_count` step upward with a loop rather than DiffSynth's broken closed-form expression (which overshoots `length % 17 in {1..4}` by `+17`). Note that `snap_length` returns `1` for `length <= 1` (`SINGLE_FRAME`), then `5` (`FRAME_REMAINDER`) for `2..4`.
- **Spatial & nested AV mask rules**: Spatial DiT tokens are **`32x32` px** (`16x16` VAE cells x `2x2` DiT patchify in `comfy/ldm/minimax/model.py::patchify_video`). Always store AV inpaint masks as a `NestedTensor(video, audio)`; **never wire stock `SetLatentNoiseMask` onto H3 AV latents** — it writes a flat 5D mask, forcing the sampler to pad `1`s onto the audio stream (silently regenerating audio) and trilinearly subsampling masks across 4-frame steps (`audio_freeze.py`, `video_mask.py`, [`h3_audio_freeze.md`](h3_audio_freeze.md)).
- **`MAX_LENGTH = 362` vs. `REFERENCE_MAX_DURATION = 15.0s`**: `362` frames (`15.083s` at `24 fps`) is the maximum legal video length (`h3_rules.py::MAX_LENGTH`); diffusers stops one grid step earlier at `345` (`14.375s`) because of its `15.0s` cap (`h3_rules.py::reference_would_emit`). `MIN_DURATION = 5.0s` (`124` frames) is the trained vendor duration floor, though ComfyUI core and our pack render shorter `17n + 5` clips (`5..107`) for tests, `masked_v2v` (`>= 39`), and short standalone tracks.

---

## 3. Dual-Clock Alignment (`39 + 51m`) and Audio Seams

For `length > 1`, `comfy_extras/nodes_minimax_h3.py::temporal_shape` allocates:

```text
video_t = 5 * ((frames - 5) // 17) + 2       # from align_frame_count / video_latent_t
audio_t = round(frames / 24 * 40)            # 5/3 audio tokens per video frame, rounded
```

The two clocks land on **exact integers simultaneously** iff both hold:
1. `frames % 17 == 5` (whole Video VAE steps: `5n + 2`), and
2. `(frames * 40) % 24 == 0`, i.e. **`frames % 3 == 0`** (whole Audio VAE tokens: `5 * frames / 3`).

Setting `n = 3m + 2` gives **`frames = 39 + 51m`** (`h3_rules.py::on_both_clocks`, `loop_plan.py::GRID` = `51` frames = `3` `17`-frame clips = exact `85` audio tokens):

| `frames` | Duration (`24 fps`) | `17`-Frame Clips | Video Latent Steps (`5n + 2`) | Audio Latent Tokens (`5/3 * frames`) | Typical Role |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **`39`** | `1.625 s` | `2` + `5f` tail | `12` | `65` | Minimum dual-clock window; short context / short `masked_v2v` shots |
| **`90`** | `3.750 s` | `5` + `5f` tail | `27` | `150` | Standard chained overlap (`workflows/h3_config.py::MASKED_JOB_CONTEXT = 90`) |
| **`141`** | `5.875 s` | `8` + `5f` tail | `42` | `235` | `h3_rules.py::SHORTEST_CHAINED` (`>= 5.0 s` floor on both clocks) |
| **`192`** | `8.000 s` | `11` + `5f` tail | `57` | `320` | Mid-length chain / `8.0 s` standalone |
| **`243`** | `10.125 s` | `14` + `5f` tail | `72` | `405` | Chained window (`10.1 s`) |
| **`294`** | `12.250 s` | `17` + `5f` tail | `87` | `490` | Chained window (`12.3 s`) |
| **`345`** | `14.375 s` | `20` + `5f` tail | `102` | `575` | `workflows/h3_config.py::LONG_WINDOW` (max chained/masked window; max diffusers `<= 15.0s` grid) |

### What Goes Wrong Off `39 + 51m`
- Any off-dual-clock `17n + 5` length (`5, 22, 56, 73, 107, 124, 158, 175, ..., 328, 362`) has `(5 * frames) % 3` in `{1, 2}`, forcing `temporal_shape` to round `audio_t` by **`±1/3` audio token (`±8.33 ms`, `±267` samples at `32 kHz`)** (community observation credited to seitanism, verified in `h3_resolutions.md`, "Whole on the audio clock" and `audio_freeze.py`). For example, `362` requests `603.33` audio tokens (`rounded to 603`, `-8.33 ms`), while `22` requests `36.67` (`rounded to 37`, `+8.33 ms`).
- **Audio seam rule**: A **standalone 1-window generation** (`loop_plan.py::stands_alone`) with nothing concatenated before or after it has no seam, so any `17n + 5 <= 362` (`MAX_LENGTH` / `LONG_LENGTH`) is legal. By contrast, **every chained loop (`h3_rules.py::chain_lengths`), every context overlap (`c % 17 == 5 and (c * 5) % 3 == 0` in `loop_plan.py::check_window_settings`), every `audio_freeze_song` multi-window render, and every `masked_v2v` load (`workflows/masked_job_graph.py::window_for`) MUST sit on `on_both_clocks`** so `window - context = 51m` advances an exact integer number of both video clips and audio tokens with zero cross-cut drift.

---

## 4. Timeline Keyframes (`t >= 0`), `3`-Frame Audio Cues, and RoPE Collisions

`PackedLayout` inserts `keyframes` (`minimax_keyframes`) immediately after the prompt/reference prefix (`cursor`) at positive timeline offsets relative to target frame `0`:

```text
cond_t = cursor + FRAME_RESCALE * resolved_frame_index      # FRAME_RESCALE = 5 / 3
```

- **Visual keyframes & guide clips ([`conditioning.py`](../conditioning.py), [`guide_latent.py`](../guide_latent.py))**:
  - Anchored at `VISUAL_COND_TIMESTEP = 0.999` (`img_update = False`, clean conditioning rows that are never denoised).
  - Stock `MiniMaxH3ImageToVideo` picks crop vs. stretch from *which socket* (`first_frame` vs `last_frame`) is wired, causing a lone `last_frame` to cover-crop away borders; `MiniMaxH3Conditioning` fixes this via semantic `frame_index` resolution ([`h3_geometry_and_nodes.md`](h3_geometry_and_nodes.md)).
  - Multi-frame guide clips (`MiniMaxH3GuideLatent`, `guide_latent.py::resolve_guide_frames`): `< 5` input frames anchor **`1` frame** at `resolved_frame_index`; `>= 5` input frames snap **down** to `17k + 5` (`5, 22, 39, ...`) and must fit inside `[0, frame_count)`.
- **Timeline audio keyframe islands (`cond_audio`, [`sound_cues.py`](../sound_cues.py), [`guide_latent.py`](../guide_latent.py))**:
  - Placed along 1D audio RoPE at `floor(5/3 * f) + k` (`AUDIO_COND_TIMESTEP = 1.0`).
  - **`3`-frame (`125 ms`) snap rule (`snap_to_3f = True`)**: `FRAME_RESCALE * f` is an exact integer audio token **only when `f % 3 == 0`** (`3` frames at `24 fps` = `5` audio tokens at `40 Hz`). Off-`3f` frame indices suffer up to `16.7 ms` fractional RoPE offset / truncation jitter.
  - **Non-overlapping `cond_audio` RoPE law**: Two separate `cond_audio` islands whose intervals `[start_tok, start_tok + rt)` intersect assign **duplicate 1D temporal RoPE indices** inside `PackedLayout`. `guide_latent.py::validate_audio_keyframe_spans` raises immediately on overlap, and `sound_cues.py::merge_sound_cue_islands` merges cues closer than the previous cue's duration into one continuous waveform at the leading anchor (`-28.0 dBFS` continuous rumble floor avoids digital-silence muting between impacts; [`research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md`](research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md)).

---

## 5. Negative-RoPE References (`t < 0`), Qwen3-VL `2 fps` Pairs, & `0.33` Release

Reference stills (`<Picture N>`), motion/character clips (`<Video N>`), and speaker/sound clips sit in the packed prefix **before** `cursor` (`t < target_origin`), conditioning appearance and kinematics without pinning target frame `0`:

- **Reference video clipping & snapping ([`reference_conditioning.py`](../reference_conditioning.py), [`h3_references.md`](h3_references.md))**: Reference videos are resampled to `24 fps`, truncated to target `frame_count`, and snapped **down** to `17n + 5`. Under `comfy` policy, `>= 5` frames (`2` latent steps) is accepted; under `release` / sglang parity (`video_policy="release"`), **`>= 22` frames** is required so Qwen3-VL receives at least two `2 fps` pairs. Note that `MiniMaxH3AppendSpriteTrajectory` renders exact `17n + 5` kinematic cutout frames (`workflows/h3_config.py::SPRITE_TRAJECTORY_DEFAULTS`).
- **Reference audio capping & hop alignment**: Stock `MiniMaxH3ReferenceToVideo` fails to truncate reference audio to `frame_count / 24`, inflating `ref_audio` token rows and pushing target `cursor` forward via `comfy/ldm/minimax/model.py::_ref_t_span` (`max(ref_audio_t, sum(video_t_spans))`). Our `MiniMaxH3ReferenceConditioning` caps reference audio duration to `frame_count / FPS` and hop-aligns audio VAE input (`reference_conditioning.py::_encode_ref_audio_aligned`).
- **Qwen3-VL `2 fps` pair sampling (`0, 12, 24, 36, ...`)**: Core's text encoder samples `<Video N>` at `2.0 fps` (every `12`th frame at `24 fps`) and merges consecutive pairs `(0, 12)`, `(24, 36)` into **one fused visual token block per second** labeled `<T.T seconds>`. Motion that happens purely between `12`-frame multiples is visible to the 3D Video VAE (`ref_img`) only, never Qwen3-VL. Avoid blank/solid-black frames landing on indices `0, 12, 24, ...` ([`masked_v2v/frames_rules.md`](masked_v2v/frames_rules.md) §3).
- **Selective `0.33` step release on 6-step Pika (`MiniMaxH3ReleaseReferences`)**: Under `MiniMaxH3SigmaShift(shift=12.0)` and `workflows/h3_config.py::PIKA_MANUAL_SIGMAS`, steps `0..1` (`end_percent = 0.33`) already carry the trajectory to unshifted flow midpoint `b = 0.50` (`comfy/ldm/minimax/model.py::time_shift_sigma`). Dropping synthetic reference scaffolding mid-denoise with [`control_vector_nodes.py`](../control_vector_nodes.py) (`MiniMaxH3ReleaseReferences` `video_end_percent = 0.33` or `audio_end_percent = 0.33` on `BasicGuider`, `0.60` under `MiniMaxH3EditGuider(streams="video_only")`) lets `<Video N>` sprites or `cond_audio` cues steer early motion/impact frames while freeing later steps for realistic physics and clean audio, keeping `<Picture 1>` anchored at `end_percent = 1.0` ([`wiki/control_vectors.md`](wiki/control_vectors.md), [`wiki/edit_guidance.md`](wiki/edit_guidance.md)).

---

## 6. The Single-Frame (`length = 1`) Image Exception

`length = 1` (`h3_rules.py::SINGLE_FRAME`, `h3_rules.py::is_single_frame`) is the sole legal exception outside `17n + 5`. ComfyUI's VAE carries a `t == 1` path (`comfy/ldm/minimax/vae.py::PH3_IMAGE_FRAME_POSITION` on `PH3ImageDecoder`), and video duration/seam checks do **not** apply to single-frame image workflows (`workflows/image/`):

1. **Official `MiniMax-H3-Image`** ([`h3_image.md`](h3_image.md), [`h3_image_nodes.py`](../h3_image_nodes.py)): 1-stream `MiniMaxH3ImageModel` (`comfy/model_base.py::MiniMaxH3Image`, no audio stream), 6-step `workflows/h3_config.py::H3_IMAGE_MANUAL_SIGMAS`, `workflows/h3_config.py::PH3_IMAGE_VAE` default (promoted encoder via `MiniMaxH3VAEPrecision`), `MiniMaxH3ImageSize` enforcing a **`4.0 MP` generation floor** + `MiniMaxH3ImageDownscale` Lanczos downscale back to requested sub-4MP dimensions, and up to `16` `<Picture N>` stills (`0`-indexed `images.image_0..N-1` on core `MiniMaxH3ImageConditioning`; `1`-indexed on split KV-cached `MiniMaxH3EncodeImageReferences` + `MiniMaxH3PromptOnImageReferences`).
2. **PrunaAI `P-H3-Image`** ([`wiki/p_h3_image.md`](wiki/p_h3_image.md), [`h3_image_editing.md`](h3_image_editing.md), [`ph3_image_nodes.py`](../ph3_image_nodes.py)): 8-step `workflows/h3_config.py::PH3_IMAGE_MANUAL_SIGMAS` with `MiniMaxH3PH3ImageConditioning` (`[4 MP, 8 MP]` output canvas and `image_1` clamp, `[1 MP, 8 MP]` `image_2..image_8` clamp, decoupled `canvas="explicit"`, `float16` `seed=42` latent rounding, 8-image cap).
3. **6-step Pika single-frame** ([`h3_image_editing.md`](h3_image_editing.md)): `MiniMaxH3Conditioning(single_frame=True)`, `MiniMaxH3EncodeReferences(single_frame=True)`, or `MiniMaxH3GuideLatent(single_frame=True)` collapses the AV latent to `[B, 24, 1, H/16, W/16]` video (+ minimal audio stub) under `MiniMaxH3SigmaShift(shift=12.0)` + `PIKA_MANUAL_SIGMAS`.

---

## 7. Spatial Canvas Grid & the `int32` QKV Stride Ceiling

- **Trained canvas family (`comfy_extras/nodes_minimax_h3.py::adapt_canvas`, [`h3_resolutions.md`](h3_resolutions.md))**: `BASE_SHORT_EDGE = 768`, `MAX_PIXELS = 768 * 1344` (`1,032,192` px), `CANVAS_MULTIPLE = 32` (`16x` VAE spatial downscale x `2x2` DiT patchify), across aspect ratios `1:4 .. 4:1`. Note `1344x768` is **`7:4`** (1.75:1), not `16:9`, and `768x768` (`1:1`) uses roughly a third of `1344x768` attention FLOPs. Core conditioning nodes allow any multiple of `32` without calling `adapt_canvas`; [`resolution.py`](../resolution.py) (`MiniMaxH3Resolution`) and [`preflight.py`](../preflight.py) (`MiniMaxH3Preflight`) report when a canvas leaves the trained family.
- **`int32` fused-QKV stride threshold (`99,864` tokens)**: H3 stores Q, K, and V as three interleaved views of one fused buffer (`stride = 3 * 56 * 128 = 21,504`). Signed 32-bit element indexing overflows past `2^31 / 21,504 = 99,864` total packed tokens (`>= 328` frames at `1344x768`, and earlier when references add tokens). Renders above `99,864` tokens require `int64`-safe / rotated attention (`comfy kitchen attention`, `MiniMaxH3SageAttention` on our int64 fork, or `MiniMaxH3Sol`); KJNodes' `MiniMaxH3TokenCounter` computes against a contiguous `7,168` stride and misreports the threshold as `299,593` ([`h3_resolutions.md`](h3_resolutions.md)).

---

## 8. Lane Specializations (Where Each Lane Adds Tighter Rules)

| Lane | Additional Frame / Mask / Clock Constraint Beyond This Document | Authoritative Doc / Code |
| :--- | :--- | :--- |
| **`masked_v2v` (face/hair/body replacement)** | Must stay on `on_both_clocks` (`39..345`); zero `BLUE-ONLY` (`0 px`) frames inside active 4-frame steps; guard standalone Clip 3 frame `34` (`Step 10` on `39`-frame loads); never split `< 39`-frame shots inside continuous footage; never show original subject unmasked in context padding; strip `c == 0` `strangers` spill over hair margins; couple `<Video 1>` render/cranium zoom and mouth/eye drawings to `bench/load_text.py` clauses | [`masked_v2v/frames_rules.md`](masked_v2v/frames_rules.md) (`§0.2` Rules L1–L6, `§6` Zoom, `§7` Mouth/Eye) |
| **`audio_freeze` / `audio_freeze_song`** | Every multi-window chain segment must stay on `h3_rules.py::chain_lengths` (`141..345`) with `context_frames` on `on_both_clocks` (`39` or `90`), snapping waveform slices directly to `40 Hz` audio hops (`audio_freeze.py::slice_window`); standalone 1-window songs may use any `17n + 5 <= 362` (`loop_plan.py::stands_alone`) | [`h3_audio_freeze.md`](h3_audio_freeze.md), [`loop_plan.py`](../loop_plan.py) |
| **Control vectors & `MiniMaxH3EditGuider`** | On 2-stream video+audio runs, high visual guidance scales (`scale >= 1.65`) clip audio unless `streams="independent"` (`audio_scale=1.0`) or `"video_only"` is set; timeline audio cue edits wire `0`-keyframe `original` vs `edited` with `MiniMaxH3ReleaseReferences(audio_end_percent=0.60)` | [`wiki/edit_guidance.md`](wiki/edit_guidance.md), [`wiki/control_vectors.md`](wiki/control_vectors.md) |
