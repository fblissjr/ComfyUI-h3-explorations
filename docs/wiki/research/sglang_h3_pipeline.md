# sglang's MiniMax-H3 pipeline, end to end

last updated: 2026-11-04 (line citations re-anchored against the clone's `c07df5a055`, a symbol where one encloses the text; a dated note on SubBlock's SM120 support and `compute_mode`, section 11); 2026-11-04 (dated notes against sglang origin/main `8884b4b421`: step count now counts forwards, sections 0, 2 and 8 and insight 20; reference stills now have three resize modes, section 2 and insight 11; a short reference video, section 3; curve AdaLN stays fp32 and the fused norm-modulate at `lossless`, section 7 and insight 17; the Euler scheduler file deleted and Pika's re-noise path, section 8 and insight 3; the VAE fast path and tile batching, section 9; the output writer, section 9; the renamed quality tiers, sections 10 and 14.1; the ComfyUI app, section 10; FastH3's step count, VSA-H3 on SM80 and `kitchen_int8`'s removal; cookbook recipe cited by heading; drifted citations repointed where a note touches them); 2026-10-02 (dated notes on the ComfyUI app's H3 integrated mode, rollout, Spectrum and `convrot_int8`, from the eighth read); 2026-09-25 (one dated note on FastH3's registration)

Written 2026-08-25 from a source read of `coderef/sglang` at commit
`6569125e3a` (2026-08-25). Nothing was run: no server, no GPU, no model
load. Five scoped readers each took one slice (admission and media; the
conditioning encode; the packed sequence and DiT; the denoise loop, VAEs and
output; the runtime layer), read the files whole, and reported with line
citations they had verified. This file is the synthesis. It does **not**
compare anything to ComfyUI; [`sglang_comparison.md`](sglang_comparison.md)
owns that and should be read after this one, not instead of it.

**Re-verified 2026-08-30 against `coderef/sglang` at commit `5ab97c4f44`.**
Every citation below was mechanically re-anchored: the cited span was pulled
from both commits and repointed where its text had moved. Line numbers in this
file are therefore the *current* checkout's, not 2026-08-25's, and one path was
simply wrong from the start — the indexed-modulation Triton kernel lives under
`python/sglang/kernels/`, never under `multimodal_gen/runtime/kernels/`, and the
old citation named the latter. What changed in substance since the first read
is listed at the head of section 13; the largest item is section 11, which did
not exist then.

*Note 2026-11-04: those line numbers are `5ab97c4f44`'s. Most of them no
longer point at the same text in the clone's checkout `16dd97df61`, although
`bench/check_doc_links.py` still finds every cited line. They were not
re-anchored on this pass; the claims were checked against sglang origin/main
`8884b4b421`, and where one moved, a dated note beside it says what is true
there and cites a file and symbol. Find the symbol before quoting a line
number from this page.
Later the same day the line citations were re-anchored against the clone's
checkout `c07df5a055` (upstream tip of 2026-11-05). A citation now names a
symbol where one encloses the cited text; a span inside a large function, or
across several, carries that checkout's line numbers; cookbook and docs pages
are cited by heading. Each was traced from the commit it was written at, which
was not always `5ab97c4f44`: section 7's weights citation still held
`6569125e3a` numbers, section 14.9's are `593c7a900d`'s (its `sol_attn.py`
numbers stand, that file being unchanged since), and short `:line` citations
whose paragraph named another file first now name their own. Where a claim
no longer holds at `c07df5a055` and no note said so, one does now (section
11).*

**Third read, 2026-09-04, against `coderef/sglang` at commit `320bdd1ee2`
(2026-09-05).** A diff read, not a re-anchor: every H3-touching commit between
`5ab97c4f44` and that tip was read as a diff, with its PR body and the
cookbook, and section 14 records what landed by stage. **Line citations in
sections 1 to 12 were not re-anchored on this pass**; the DiT model file
moved by several hundred lines when its AdaLN cache was relocated into its own
module, so section 7's line numbers into `runtime/models/dits/minimax_h3.py`
point at text that has moved. Section 14 cites files and symbols, which do
not rot the same way.

**Section 14.9 added 2026-09-11**, from a source read at `593c7a900d`: the
Sol-Attn backend, which sglang has shipped since 2026-08-09 and which no
earlier read of this file described.

Citations are repo-relative paths with the line range the reader verified.
Every claim is SOURCE (read at the cited lines) unless marked
INFERENCE. Numbers are what the code says at that commit; the code moves,
and `bench/check_doc_links.py` confirms only that cited lines still exist,
not that they still say this.

---

## 0. The map

```
HTTP /v1/videos ─ lower request (task, conditions, target) ─ validate canonical request
   │  seconds/size/num_frames/fps are INERT for H3; only `target` carries geometry and time
   ▼
PRE-QUEUE  localise every media URI, ffprobe it, resolve the plan (canvas, frames, per-material shapes)
   │  extra["minimax_h3_resolved_plan"] frozen; multi-output expansion -> seed+i per output
   ▼
QUEUE  one request at a time (H3 never merges requests)
   ▼
1 InputValidation      seeds, generators
2 PartitionAdmission   task in the checkpoint's partition; steps >= 2; quality gate  (AFTER download)
3 TextEncoding         Qwen3-VL-32B, 50 layers, raw presentation -> hidden [L,5120] bf16 + tags
4 VisualEncoding       video VAE fp32, posterior SAMPLED at seed 42 -> keyframe / ref rows [n,96]
5 AudioEncoding        audio VAE fp32, posterior MEAN, cuDNN off -> ref audio rows [2T,32]
6 LatentPreparation    noise: video [T*(H/32)(W/32),96], audio [2*audio_t,32], same seed, two generators
7 TimestepPreparation  two sigma schedules (video shift 12, audio shift 3), same step count
8 Denoising            packed [text|cond|audio|video|pad] to a multiple of 64; 50 DiT blocks; Euler, no noise
9 Decoding             video VAE decode fp16 autocast, tiled; audio VAE decode fp32
   ▼
ffmpeg  h264 yuv420p 24 fps + aac 32 kHz stereo -> mp4, then re-probed and REJECTED on any mismatch
```

*Note 2026-11-04: at sglang origin/main stage 2 admits `num_inference_steps >= 1`
(`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::MiniMaxH3SamplingParams.min_num_inference_steps`),
and the step count is the number of DiT forwards, since `32f34e898b` (#38671);
section 2's note has the schedule.*

Stage composition: `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py::MiniMaxH3Pipeline.create_pipeline_stages`.
The constructor refuses to start without `ffmpeg` and `ffprobe` on PATH
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py::MiniMaxH3Pipeline.__init__`).

---

## 1. Request and admission

**Tasks.** Exactly `t2va`, `fl2va`, `ref2va`, case-sensitive, never inferred
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/video_adapter.py::MiniMaxH3VideoModelAdapter.supported_tasks`; `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/video_adapter.py::MiniMaxH3VideoModelAdapter.validate_task_gate`; `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py:290`). There is no
`i2va`: a first frame is `fl2va` with `frame_index: 0`; video-to-video is
`ref2va` with a `video` or `video_audio` reference. Keyframe signatures are
the ordered tuple of `frame_index` values and must be one of `(0,)`, `(-1,)`,
`(0,-1)` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py:72-76`), enforced at validation, plan
resolution, canvas preparation and text encoding alike.

**Canonical request** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py::minimax_h3_validate_canonical_request`):
`{schema, task, prompt, conditions, target[, flow_shift, audio_flow_shift, seed]}`.
`target` is `{short_edge, aspect_ratio, duration_seconds}`; unknown `target`
keys are **silently dropped** before validation (`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::MiniMaxH3SamplingParams._validate`),
while an unknown key in a `conditions[]` entry is an error
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py::_validate_conditions`, against `_ALLOWED_CONDITION_KEYS`).
`seed` is in `[0, 2^63-1]`.

*2026-10-09: these three citations were line ranges from the 2026-08-25 read.
sglang's `17b2f35ae7` shortened the file and the first fell off its end, so
they name symbols now. The other line citations on this page are still that
read's and drift the same way: find the symbol before quoting one.*

**What the HTTP layer refuses outright** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/video_adapter.py:124-172`):
explicit `num_frames` or `fps`; `guidance_scale`, `guidance_scale_2`,
`true_cfg_scale`, a non-None `negative_prompt`; `audio_guidance_scale`;
`enable_frame_interpolation`, `enable_upscaling`; any `output_mode` other than
decoded files. The sampling params also reject TeaCache, rollout and
trajectory output (`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::MiniMaxH3SamplingParams._validate`).
*Corrected 2026-10-02: since `0931a72eb2` rollout is accepted for t2va only;
trajectory output still raises. The cited line range no longer holds the
check; it is the `rollout` branch of the same file.*

**Per task** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py:148-242`):

| task | conditions | duration | aspect ratio |
|---|---|---|---|
| `t2va` | none | required | `auto` -> `16:9`; explicit only from `21:9, 16:9, 4:3, 1:1, 3:4, 9:16` |
| `fl2va` | 1..2 keyframes (`image`, `frame_index`) | required | `auto` deferred to the first keyframe's probed ratio; explicit `W:H` with any positive integers |
| `ref2va` | >= 1 of keyframe/image, reference/image, reference/video, reference/video_audio, reference/audio | optional when exactly one audio-bearing condition | `auto` -> `16:9`; explicit from the six; references never bind geometry |

The finite aspect list is enforced only for t2va and ref2va
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py::_validate_target`); the ref2va profile's own comment claims
`7:4` is allowed and is contradicted by that check (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py:231-233`).

**Material chains and who sees what** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py:80-88,165-226`;
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/resolved_plan.py:370-387`):

| (role, type) | chain | video VAE | audio VAE | Qwen3-VL |
|---|---|---|---|---|
| keyframe / image | `image.target_canvas` | yes | no | fl2va yes; **ref2va no** |
| reference / image | `image.reference_preserve` | yes | no | yes |
| reference / video | `video.reference_preserve` | yes | if a soundtrack exists | yes |
| reference / video_audio | `video_audio.reference_preserve` | yes | required | yes |
| reference / audio | `audio` | no | yes | label only |

**Partition gate.** The checkpoint declares itself in
`model_index.json._minimax_h3` (`schema_version 1`, `partition`, `tasks`,
`task_aliases`, `sigma_shift_scales`; `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py::MiniMaxH3ReleaseMetadata`). The
static map is `t2va -> fl2va, fl2va -> fl2va, ref2va -> ref2va`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py:32-38`). One partition per process: `--model-variant`
selects the subfolder and is cross-checked against the loaded partition
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py::MiniMaxH3Pipeline._load_config`). The gate runs as
pipeline stage 2 (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py::MiniMaxH3PartitionAdmissionStage`), which is **after** the
media were localised, probed and queued.
*Note 2026-11-04: at sglang origin/main the block also takes optional
`dmd_denoising_steps`, `pdd_fused_heads`, `stochastic_sampling` and
`lora_adapter`, which distilled and adapter releases use; `schema_version`
is still 1
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py::MiniMaxH3ReleaseMetadata`).*

**Seeds.** Per-output seed is `seed + output_index`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/request_utils.py::normalize_output_seeds`); the latent stage falls back to 42
only when the plan carries none (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/latent_preparation.py::MiniMaxH3LatentPreparationStage._prepare_denoise_state_from_plan`).

---

## 2. Time grid and canvas

Constants: `fps = 24`; duration `[4.0, 15.0]` s checked on the **request**
value (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py:29-31`; `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py::_validate_target`).

**Frames** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/time_request.py:5-29`):
```
align(fc)        = fc + (5 - fc) % 17            # snaps UP to 17n+5
video_latent_t   = ((fc - 5) // 17) * 5 + 2      # 5n+2
audio_latent_t   = round(seconds * 40)           # 40 Hz audio latents
frame_count      = align(round(duration * 24)); duration := frame_count / 24
```
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/resolved_plan.py:185-224`). So 4.0 s -> 107 frames / 4.458 s / latent 32;
5.0 s -> 124 / 5.167 s / 37 / audio 207; 15.0 s -> 362 / 15.083 s / 107. The
aligned duration can exceed the 15 s bound and nothing re-checks it.

**Sigmas** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/time_request.py::minimax_h3_time_shift_sigmas`): `base = linspace(1, 0, n)`,
`sigma = s*base / (1 + (s-1)*base)`, `unique_consecutive`, terminal 0
guaranteed. `num_inference_steps` counts grid points including the terminal
zero, so the loop runs `n-1` model calls; `n >= 2` is enforced at admission
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py::MiniMaxH3PartitionAdmissionStage.forward`). Shift priority per modality: request >
`model_index` `sigma_shift_scales` > profile default, and the defaults are
`12.0` video, `3.0` audio for every task (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/timestep_preparation.py::MiniMaxH3TimestepPreparationStage._generate_sigmas_from_plan`;
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py:153-154,175-176,228-229`). Default 50 steps. `t = 1 - sigma`.
*Note 2026-11-04: the grid-point count is no longer true at sglang origin/main.
Since `32f34e898b` (#38671, 2026-10-10) `num_inference_steps` counts DiT
forwards: `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/time_request.py::minimax_h3_time_shift_sigmas`
builds `linspace(1, 0, N+1)`, so the default 50 is 50 model calls (it was 49),
and the floor is 1, not 2
(`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::MiniMaxH3SamplingParams.min_num_inference_steps`).
The shift priority and the 12 / 3 defaults hold.*

**Target canvas, `adapt_shape_v1`** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/resolved_plan.py:114-182`; constants
`:42-47`): base short edge 768, area cap `768 * 1344 = 1,032,192`, multiple 32,
aspect in `[1/4, 4]`.
```
nominal = (768*ratio, 768) if ratio >= 1 else (768, 768/ratio)
if area(nominal) > cap: nominal *= sqrt(cap / area)        # size_mode "area"
w, h = max(32, round(nominal/32)*32) per axis               # nearest, independently
```
Worked: 16:9 -> 1344x768; 21:9 -> 1536x672; 4:3 -> 1024x768; 1:1 -> 768x768;
3:4 -> 768x1024; 9:16 -> 768x1344. `latent_h = height // 16`,
`latent_w = width // 16`; packed video rows `= latent_t * (latent_h//2) * (latent_w//2)`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/latent_preparation.py::MiniMaxH3LatentPreparationStage._prepare_denoise_state_from_plan`). Any positive integer short edge
is accepted and merely warns once when not 768
(`coderef/sglang/python/sglang/multimodal_gen/test/unit/test_minimax_h3_short_edge.py:41-102`).

**Keyframes** take the target shape verbatim. The first keyframe in the list
(the geometry anchor, including a lone `-1`) is **stretched** to the canvas
with LANCZOS, no aspect preservation; only the second keyframe of a `(0,-1)`
pair is cover-cropped with upscale allowed (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/canvas.py:84-97,214-230`).
Under `auto` aspect the stretch is near-identity; under an explicit ratio it
distorts.

**Reference stills** are sized independently of the target: `scale = 2048 / min(w, h)`
**always, upscaling included**, no area cap, each axis rounded to 32 on its
own (so a slight stretch), ratio in `[1/4, 4]`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::minimax_h3_resolve_reference_image_shape`). The prepared PIL feeds both Qwen3-VL
and the VAE (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::minimax_h3_prepared_reference_image`).
*Note 2026-11-04: at sglang origin/main this is one of three modes. Since
`5727f08932` (#35839, 2026-10-13) the request field
`reference_resize_mode` picks `match` (scale to the target canvas's pixel
area, never up), `max` (down to a 2048 short edge only when the still is
larger) or `diffusers` (always to a 2048 short edge, upscaling included,
which is the behavior above), and `diffusers` is the default
(`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::MiniMaxH3SamplingParams`;
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::minimax_h3_resolve_reference_image_shape`;
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py::MINIMAX_H3_REFERENCE_RESIZE_MODES`).
Every mode keeps the display ratio, rounds each axis to 32 on its own and
resamples with LANCZOS; only stills change, keyframes and reference videos
do not. The constants' comment says the ModelTC Ref2VA Turbo LoRA was trained
with `match`.*

**Reference videos** go through `adapt_shape_v1` with the base short edge
fixed at 768, ignoring `target.short_edge` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/prequeue.py:290-302`), decoded
with a direct non-cropping `scale=W:H` filter.

---

## 3. Media ingestion

URIs (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/material_io.py::minimax_h3_localize_material_uri`): `file://` (localhost only), bare paths,
`http(s)` (120 s timeout, 1 MiB chunks, suffix from URL or Content-Type),
`data:`/`base64://` with a streaming validating decoder, `tar+offset://`;
`s3://` raises `NotImplementedError`. Sources and probe facts are cached per
request in a temp dir registry keyed by owner (`:61-93`), which pre-queue
renames from `material` to `prequeue_material` so multi-output expansion does
not delete the shared source after the first output (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/prequeue.py:227-238`).

Probe (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/material_io.py::_validate_localized_media`): images via PIL, **JPEG/PNG/WEBP only**,
display size after `exif_transpose`; audio/video via `ffprobe` with a
container allow-list (`mov, mp4, m4a, 3gp, 3g2, mj2, matroska, webm, wav,
mp3, flac, ogg`), stream requirements per type, duration = max of container
and stream durations, display geometry corrected for SAR, DAR and rotation
(`:251-293`).

Decoding for the encoders:

- **Reference video frames**: one ffmpeg pass
  `[-ss start] -i path -map 0:v:0 -an -vf fps=24,scale=W:H:flags=lanczos,setsar=1 -frames:v N -f rawvideo -pix_fmt rgb24`
  -> `uint8 [T, H, W, 3]` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::minimax_h3_decode_reference_video_frames`), where `N` is the
  **target's** aligned frame count (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::_reference_video_target_frame_count`), so a reference video is
  bounded by the target, not by itself. The same array serves Qwen3-VL
  (every 12th frame, i.e. 2 fps) and the VAE.
  *Note 2026-11-04, traced at sglang origin/main in code this read did not
  follow: `-frames:v` only caps the decode. A reference shorter than the
  target comes back with its own frame count, unpadded and unlooped; the
  video VAE encode then trims it down to the nearest `17n+5` with `n >= 1`,
  and a reference under 22 frames raises "Cannot trim"
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/processor.py::VAEProcessor.get_suitable_video_length`,
  through `align_video_length` in trim mode). The ref2va cursor then
  advances by `max(T_audio, span(video))` of what was encoded (section 6).*
- **Audio**: `ffmpeg [-ss] -i -map 0:a:0 -vn -ac 2 [-ar 44100 for video chains] [-t target_duration] -f f32le`
  -> `[2, N]` float32 (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::_load_waveform`), then `torchaudio.Resample(src, 32000)` when
  needed (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::_audio_resampler`). A video soundtrack is therefore resampled twice
  (ffmpeg to 44.1 kHz, torchaudio to 32 kHz) despite the docstring's "single
  resample", and it is **truncated to the target duration**
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/audio_encoding.py:147-152`), contradicting the profile comment that
  says the full track is kept (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py:208-210`).
- **The RGB decode is shared across ranks, and the sharing is a mapped file
  rather than a collective.** On Linux, `ffmpeg` writes its rawvideo stream
  straight into an anonymous `memfd` and the result is `mmap`ed, which avoids
  aggregating several hundred MiB through `communicate()`
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:488-538`). When more than one rank is in the
  request's replica, one rank per *host* runs that decode and the others open
  its descriptor through `/proc` and map the same pages copy-on-write
  (`:552-661`), gated by a `share_across_replicas` flag the text-encoding stage
  passes only when the replica is wider than one and this is not the
  single-copy encoder-DP path (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py:401-418`;
  `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::minimax_h3_decode_reference_video_frames`). Every failure mode falls
  back **together**, all ranks agreeing from an `all_gather_object` of the
  leader's state, so a `hidepid` or container policy that blocks the `/proc`
  traversal degrades to a per-rank decode rather than to a failed request or to
  different RGB bytes on different ranks (`:647-657`).
- A silent `video` reference yields a zero-length audio entry `[0, 32]` to
  keep block order (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/audio_encoding.py:175-184`); `video_audio`
  without an audio stream is refused at probe.

---

## 4. The Qwen3-VL conditioning encode

**Presentation** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/presentation.py`). No chat template, no system prompt,
no `<|im_start|>`, no BOS or EOS: the stage builds raw ids with
`tokenizer(text, add_special_tokens=False)` and hand-made vision blocks
`[<|vision_start|>] + [pad]*n + [<|vision_end|>]` (`:30-39,109-113`). The
prompt passes through verbatim. Per task:

| task | stream |
|---|---|
| t2va | `prompt` |
| fl2va | for each keyframe `"<Picture i>: "` + image block; then `prompt` |
| ref2va | per condition in request order: image -> `"<Picture i>: "` + image block; audio -> `"<Audio j>: "` **label only**; video -> `"<Video k>: "` then per temporal block `"<t seconds>"` (`.1f`) + video block; then `prompt` |

(`:84-106,116-158,262-294`). Ordinals are per type and 1-based. A video with a
soundtrack emits its `<Audio j>: ` label **before** its `<Video k>: `
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py:453-470`). ref2va keyframes are deliberately
omitted from the presentation (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py::MiniMaxH3TextEncodingStage._encode_ref2va`). Verified in this repo's copy of
the release tokenizer: its chat template writes `Picture N: ` without angle
brackets under `add_vision_id`; H3 writes `<Picture i>: ` with them and never
invokes the template.

**Token tags** travel with the ids: text, labels and timestamps are tag 1;
vision blocks **including the start and end sentinels** are tag 0, the video
modality (`:42-81`). The tags later overwrite the packed sequence's text-slot
tags (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py:710-713`), so vision tokens inside the text
segment are modulated with the video AdaLN branch while sitting at text
positions.

**Vision blocks.** The release processor files set image bounds
`65,536..16,777,216` px, video bounds `4,096..25,165,824`, patch 16, temporal
patch 2, merge 2, mean and std 0.5 (this repo's `vendor_config/preprocessor_config.json`
and `video_preprocessor_config.json`). The stage calls `image_processor` and
`video_processor` directly, never the combined processor, and passes
`do_sample_frames=False` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py:347,485,516-521`).
Pixels become `(x/255 - 0.5)/0.5` in `[-1, 1]`; a still is **duplicated to
T=2** for the Conv3d patch embed; tokens per still `= (H/32)*(W/32)`, so a
768x1344 canvas is 1,008 tokens and a 2048x2048 reference 4,096. Video: frames
sampled at stride 12, timestamps `i/2` padded to even by repeating the last,
block timestamps are pair midpoints (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:734-769`;
pinned at 25 frames -> `[0.25, 1.0]`, `coderef/sglang/python/sglang/multimodal_gen/test/unit/test_minimax_h3_media.py::test_video_transform_runs_once_and_qwen_samples_shared_rgb`).
`pixel_values` are cast to bf16 on the device (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/encoders/minimax_h3_qwen3vl.py:347-380`).

**The encoder** (`coderef/sglang/python/sglang/multimodal_gen/configs/models/encoders/minimax_h3_qwen3vl.py:11,21-53`;
`coderef/sglang/python/sglang/multimodal_gen/runtime/models/encoders/minimax_h3_qwen3vl.py:46-52,258-263,274-281`):
Qwen3-VL-32B with `num_hidden_layers` forced to 50, `use_cache` off, hidden
5120, 64 heads, 8 KV heads, head dim 128; `language_model.norm` replaced by
`Identity`; layers >= 50, `lm_head` and the final norm dropped at load. The
tap is the residual after decoder layer index 49, **unnormalised**
(equivalent to HF `hidden_states[50]`). Precision bf16
(`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py:70`). Batch of one; M-RoPE
positions computed on CPU by `get_rope_index` when any grid is present, else
plain `arange` on all three axes for text-only (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/encoders/minimax_h3_qwen3vl.py:358-365`). DeepStack
features from vision blocks `[8, 16, 24]` are added after decoder layers 0, 1
and 2 (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/encoders/qwen3vl.py:632-642`, `coderef/sglang/python/sglang/multimodal_gen/runtime/models/encoders/qwen3vl.py::Qwen3VLTextModel._deepstack_process`). Output:
`hidden_states [L, 5120]` bf16, `text_len`, `text_token_tags [L]` into
`extra["minimax_h3_text_embeddings"]` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py::MiniMaxH3TextEncodingStage._encode_from_plan`).
Encoding is deduplicated by `(task, prompt, materials, shape)` within a
grouped request (`:87-113`), and requests are distributed one whole request
per encoder copy under encoder DP (`:115-196`).

---

## 5. Reference and keyframe encodes

**Video VAE** (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3.py::MiniMaxH3VideoVAE`;
`coderef/sglang/python/sglang/multimodal_gen/configs/models/vaes/minimax_h3_video.py:11-24`): 24 latent channels,
spatial 16x (`space_down [2,2,2,2,1,1]`), temporal 4x (`time_down [1,2,2,1,1,1]`),
clip length 17, token drop 3, causal CNN encoder with reflect padding and
temporally isolated GroupNorm, **non-causal ViT decoder** (dim 2048, 36
layers, 32 heads x 64, 3-D RoPE), ImageNet pixel normalisation, top-left crop
to a multiple of 16, encoder tiling on at 256 px with >= 64 px overlap. The
VAE's own `shift_factor` and `scaling_factor` are 0 and 1; all normalisation is
the stage's `(z - latents_mean) / latents_std` with 24 values from the VAE
config.

Encode: `quant_conv(encoder(x))` -> `DiagonalGaussianDistribution.sample()`,
**always sampled, never the mean**; the noise is drawn on the CPU default
generator then moved (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/klvae.py::DiagonalGaussianDistribution`, `coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/klvae.py::AutoencoderKLLegacy.encode_base`). So every keyframe and
reference-video encode forks the RNG and seeds it to **42**, independent of
the request seed (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/keyframe_encoding.py:31-32`, `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/keyframe_encoding.py::minimax_h3_scoped_encode_rng`; `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::MINIMAX_H3_REFERENCE_VIDEO_ENCODE_SEED`;
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/visual_encoding.py:243-244`), with `use_fp16_latent=True`, which
rounds the latent to fp16 before normalisation (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/klvae.py:1023-1024`). Rows are
patchified `[1,2,2]` to `[n, 96]` fp32 on CPU (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_tokens.py:23-41`).
A 17n+5-frame video encodes as n+1 clips of 17 (last padded by repeating the
final frame), 5 tokens each, minus 3 dropped, giving 5n+2 (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/klvae.py::AutoencoderKL.encode_temporal`).

**Audio VAE** (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3.py::MiniMaxH3AudioVAE`): DAC-style
encoder with rates `[2,4,4,5,5]`, hop 800 samples at 32 kHz, i.e. 40 latent
frames per second; latent dim 2048 projected by a causal attention block to
32 channels; BigVGAN decoder with rates `[5,5,2,2,2,2,2]`, output
`clamp(-1, 1)` rather than tanh. Encode takes the **posterior mean**
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::minimax_h3_encode_reference_audio_rows`) under a determinism context that disables
TF32 and cuDNN and pins SDP to the math kernel (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::_AudioVAEDeterminismContext`); rows are
`[2T, 32]` channel-major (all left, then all right).

**Condition noise augmentation** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/condition_noise.py`; defaults in
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py:30-32`): visual conditions at `0.999`, audio references at
`1.0`, overridable per request in `[0, 1]`. Applied only below 1.0: for each
visual condition a fresh CPU generator seeded with the request seed draws
`randn(1, 24, target_latent_t + n_conditions, H, W)` and keeps the `[:T_c]`
prefix, so the anchor noise depends on the target length and on how many
conditions there are, and two same-shape conditions get identical noise
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/condition_noise.py::minimax_h3_imgvid_cond_noise_aug_rows`); audio uses `seed + 1` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/condition_noise.py::minimax_h3_audio_cond_noise_aug_rows`). The mix is
`t*clean + (1-t)*noise`. At the defaults, keyframes carry 0.1 % noise and audio
references are untouched.

**Neither encode necessarily runs on every rank.** The audio encode runs on
replica rank 0 only, and its rows reach the other ranks by
`broadcast_tensor_dict`; a failure on the owner is broadcast as a string first
so that no rank is left waiting in the payload collective
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/audio_encoding.py:91-129`). The visual encode takes whichever of
two shapes applies: with a replica wider than one **and** the video VAE's
`parallel_tiling` on, every rank encodes and nothing is broadcast, because the
tile loop is itself split by tile-parallel rank
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/klvae.py::AutoencoderKL.tiled_encode`); otherwise rank 0
encodes and every produced `extra` key is broadcast
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/visual_encoding.py:88-124`). So the seed-42
posterior sample is drawn once and distributed in the broadcast shape, and
drawn per rank over a tile subset in the other. **Whether those two shapes
produce the same rows is not established by this read** — the seed is pinned
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/keyframe_encoding.py:31-32`) but nothing here says the
tiled draw order is invariant to how many ranks the tiles were split over, and
nothing was executed to find out.

---

## 6. The packed sequence

File `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py`. Constants (`:30-35`): spatial scale 32, temporal
group 5, `frame_per_token (1,4,4,4,4)`, frame rescale `5/3`, patch 2x2;
alignment 64 (`coderef/sglang/python/sglang/multimodal_gen/configs/models/dits/minimax_h3.py::MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT`).

**Layout.** t2va/fl2va: `[text | keyframes | target audio | target video | pad]`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py::minimax_h3_packed_sequence`). ref2va: `[text | keyframes | reference blocks in request order | target audio | target video | pad]`,
where a video reference contributes its audio rows first, then its visual
rows (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py::minimax_h3_packed_sequence_ref2va_blocks`, `:412-417`). `seq_len = ceil(used / 64) * 64`. `update_mask` is
False on every conditioning row; `audio_update_mask` is False on reference
audio rows. `cu_seqlens = [0, used, seq_len]`: one attention segment for the
live rows and one for the padding tail (`:210,542`).

**Position ids** `(t, h, w)` per row (`:170-203,258-279`): text `(i, 0, 0)`;
spatial axes are `linspace` over `[left, left+ratio)` with
`ratio = D / sqrt(latent_h * latent_w)`, `endpoint=False`, scaled by 32, so
every media item is normalised by its own area; video temporal coordinates
advance by `(5/3) * (1,4,4,4,4)[k % 5]` from an origin at `text_len`; audio
rows sit at `t = text_len + arange(audio_t)`, `h = 0`, `w` at the target
w-grid's two extremes, one per channel; a first-frame keyframe sits at the
origin and a last-frame keyframe at `origin + span(latent_t) - 5/3`. In
ref2va a cursor walks the blocks: an image reference costs one integer slot,
an audio reference costs its `T`, a video reference costs
`max(T_audio, span(video))`, and the keyframes, though packed before the
references, are timed at the target origin (`:433-523`). Standalone audio
references borrow the **target** w-grid extremes; a video reference's audio
uses its own grid (`:460-461,486-490`).

**Tags and timesteps.** Tags: pad -1, text 1, audio 2, image/video 0
(`:205-208`). Per denoise step at most four timestep values exist
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py:404-467`): `t_video` for video, text **and padding** rows;
`max(t_video, 0.999)` for keyframe and visual reference rows; `t_audio` for
target audio; `max(t_audio, 1.0)` for audio references. They are deduplicated
by exact fp32 equality on the host, and each row's modulation index is
`tag + 3 * timestep_index` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py::MiniMaxH3DenoiseBranch._expand_step_timesteps`; modality count 3,
`coderef/sglang/python/sglang/multimodal_gen/configs/models/dits/minimax_h3.py::MINIMAX_H3_ADALN_MODALITY_NUM`). Widths: t2va 2, fl2va 3, ref2va 4.

Tests pin the row counts, the keyframe anchors to twelve places, the
rejection of any other signature, and the ref2va block order
(`coderef/sglang/python/sglang/multimodal_gen/test/unit/test_minimax_h3_packed_sequence.py::TestMiniMaxH3PackedSequence`).

---

## 7. The DiT forward

**Architecture** (`coderef/sglang/python/sglang/multimodal_gen/configs/models/dits/minimax_h3.py:106-126`;
`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py`): 50 blocks, hidden 5376, 56 heads x
128, FFN 14336 (SwiGLU, `fc1` fused `[gate, up]`), a 2-block token refiner
with no AdaLN and no RoPE, `condition_proj 5120 -> 5376`,
`video_patch_proj 96 -> 5376` and `audio_patch_proj 32 -> 5376`, time embedder
(256-wide sinusoid, cos before sin, `proj_in 256->5376`, SiLU, `proj_out 5376->2688`),
per-block AdaLN `Linear(2688 -> 18*5376)` viewed as `[M*3, 6*5376]` and
chunked into shift/scale/gate for attention and MLP, final layer
`Linear(2688 -> 2*5376)` plus `video_out 5376 -> 96` and `audio_out 5376 -> 32`.
RMSNorm everywhere at eps 1e-5, no biases inside blocks. **fp32 island**: the
patch projections, time embedder, output heads and `rope.inv_freq`; everything
else bf16 (`:211-226`); the island is never quantised. Pruned checkpoints
(`MiniMaxH3PrunedTransformer3DModel`) replace the time embedder with a curve
table `adaln_t_table [grid, rank]` read by linear interpolation
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3DiTModel._time_embedding`; test pins 8 / 1025 / 2688).
*Note 2026-11-04: "everything else bf16" missed the curve path, in code that
predates this read (`21c88f8625`, #35370). A curve checkpoint also keeps
each block's `adaln_proj` in fp32 and refuses a non-fp32 one after load, so
its modulation tables are fp32 at every quality tier
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3AdalnProj`,
`params_dtype = _FP32_DTYPE if arch.adaln_curve_grid ...`;
`::MiniMaxH3DiTModel.post_load_weights`, "must stay fp32 with curve
AdaLN"); `1fd46ddd56` (#44180) gave those fp32 tables their own fused
kernels, pinned equal to the eager ops by its tests. The class name
`MiniMaxH3PrunedTransformer3DModel` is now an alias the base model accepts.*

**RoPE** (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:567-606`): `inv_freq [16]` is a persistent buffer allocated empty
and loaded from the checkpoint (INFERENCE: nothing in the tree computes it);
per axis `pos * inv_freq` gives `[S, 3, 16]`, concatenated to 48 and doubled
to 96, so rotation covers head dims `[0, 96)` in neox rotate-half pairing and
dims `[96, 128)` pass through. Q and K are RMS-normalised per head and rounded
to bf16 **before** rotation (`:1531-1557`).

**Forward** (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3DiTModel.forward`). `_embed` (eager under graph capture) scatters
text rows (already refined once per request), video rows through
`video_patch_proj` and audio rows through `audio_patch_proj` into a bf16
`[local_seq_len, 5376]` buffer; padding rows are zero. `adaln_input = silu(t_emb).bf16`.
Tags are clamped at 0, so **padding rows are modulated as video rows at the
video timestep** (`:3150`). Each block, for every row identically:
```
h = norm1(x); h = h*(1+scale_msa) + shift_msa; x = x + gate_msa * attn(h)
h = norm2(x); h = h*(1+scale_mlp) + shift_mlp; x = x + gate_mlp * mlp(h)
```
with the CUDA kernels rounding `1+scale`, `x*(1+scale)`, `gate*other` and the
SwiGLU product to bf16 (`coderef/sglang/python/sglang/kernels/ops/diffusion/modulate/indexed_modulation_triton.py:39-45,78-82`).
*Note 2026-11-04: at sglang origin/main that chain is the `exact` tier's. Since
`121aa8dcc7` (#43327, 2026-10-09) the default `lossless` tier, and `high`,
run each block's `norm1` / `norm2` plus `x*(1+scale) + shift` as one fp32
pass with a single bf16 rounding
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::_modulate_rmsnorm_scale_shift`,
`fused=`; the gate is `MINIMAX_H3_NORM_MODULATE` in
`coderef/sglang/python/sglang/kernels/ops/diffusion/sites/minimax_h3_norm_modulate_site.py`).
The final layer's modulation and the gated residual are not fused. Since
`65386cb95a` (#43501) a dynamic LoRA delta is likewise added inside the
second GEMM at `lossless`, while `exact` keeps the separate add
(`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/lora/linear.py::request_allows_lossless`).*
The first gated residual is in place unless Cache-DiT input preservation is
armed (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:1892-1899`, `coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3DiTModel.set_cache_dit_input_preservation`).

**Attention** is one FlashAttention varlen call over the whole packed
sequence: `cu_seqlens_q = cu_seqlens_k = [0, used, seq_len]`, `max_seqlen = used`,
non-causal, scale `128^-0.5` (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/flash_attn.py::FlashAttentionImpl.forward_varlen`).
**Every live row attends to every other live row with no modality mask**; the
padding tail attends only to itself. That is a statement about the *dense*
backends — since 2026-08-26 one backend deliberately breaks it, and section 11
is the whole of that exception. Backend on CUDA: FA (fa3; fa4 on
Blackwell), TORCH_SDPA on SM12.x, cuDNN SDPA on Blackwell when FA is present
(`coderef/sglang/python/sglang/multimodal_gen/runtime/platforms/cuda.py:724-832`); any backend implementing
`forward_varlen` is admissible. Under Ulysses the all-to-all happens inside
the attention core, so each rank holds the full sequence with `56/ulysses`
heads (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::_minimax_h3_attention_core_impl`; `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/usp.py::_usp_input_all_to_all_packed_qkv`, `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/usp.py::_usp_output_all_to_all`); under ring,
K/V rotate and each hop attends only the remote chunk's real prefix, FA only
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:1060-1076`; `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/usp.py:961-1077`).

**Final layer** modulates by timestep only, casts to fp32, and computes
`video_out` and `audio_out` on **all** rows; the video logits are then
`index_select`ed to the target rows and the audio logits cover every audio
row (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3FinalLayer.forward`, `:3245-3296`). Under sequence parallel the rows are gathered
after selection, and under TP the columns after that.

**AdaLN plans and cache** (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_adaln_cache.py::MiniMaxH3AdalnCache`; `coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3DiTModel.prepare_adaln_plans`). The per-request set of
distinct timestep tuples is computed once before the loop; the cache stores
`block_params [P, W, 50, 96768]` bf16 and `final_params [P, W, 10752]` and
looks them up by exact fp32 match. Building it means one `F.linear` per layer
at the plan's own row count and the TP-sharded width, because cuBLAS picks
kernels by shape and any other shape "silently perturbs the output"
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3_adaln_cache.py:561-572,627-631`); the online rebuild reads all of `adaln_proj`,
which the flag's help text prices at 24.2 GiB. Unquantised, non-curve
checkpoints only. The standalone tool `coderef/sglang/python/sglang/multimodal_gen/tools/build_minimax_h3_adaln_cache.py`
re-implements the time embedder and hard-codes the model dimensions.

**Weights** (`coderef/sglang/python/sglang/multimodal_gen/runtime/loader/minimax_h3_weights.py`; `coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:1161-1188`, `coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3Attention._install_qkv_weight_loader`):
the official safetensors interleave Q, K and V **per head** (56 groups of
`[q_h | k_h | v_h]`, 21,504 rows); the runtime reorders to `[q_all | k_all | v_all]`
on load, and installs the same row permutation on every row-indexed `qkv_proj`
parameter whose leading dimension is 21,504, so per-row quantisation scales
follow their rows; the unit test states that without it the model "loads and
runs, and renders noise" (`coderef/sglang/python/sglang/multimodal_gen/test/unit/test_minimax_h3_qkv_scale_reorder.py:21-27`).
Diffusers-named checkpoints are mapped, with `fc1`'s `[value, gate]` swapped
to `[gate, value]`. Comfy-format quantised DiTs are recognised by their
per-layer markers; NVFP4 files keep the native layout with swizzled scales.

**Breakable CUDA graphs** (`coderef/sglang/python/sglang/multimodal_gen/runtime/breakable_cuda_graph/model_padders/minimax_h3.py`):
the padder buckets only the **text** length (default buckets 64..1024; the
cookbook's 1344x768 ref2va needs 5504) and never grows the packed sequence;
`cu_seqlens` becomes `[0, used, x.shape[1]]` with `max_seqlen = x.shape[1]`.
`_embed` and every block's attention core stay eager; projections, norms,
RoPE, modulation and MLPs are captured.

---

## 8. The denoise loop

`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py`. Persistent buffers `x [1, seq_len, 96]` and
`audio_x [1, seq_len, 32]` in fp32 are primed once with every conditioning
row, and afterwards only the target rows are re-copied each step
(`:264-269`, `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py::MiniMaxH3DenoiseBranch.forward_kwargs`). The static kwargs handed to the model include the
position ids (cast to fp32 on the host), both update masks, the rank-local
tags, `skip_mask_out_condition=True`, the refined prompt embeddings, the
position infos, and the packed-sequence params (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py:297-334`).

**Update.** The scheduler class `MiniMaxH3EulerAncestralEta0SchedulerAdapter`
exists as a validated reference but is not wired; the loop uses a fused
in-place function pinned bit-equal to it (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py::_minimax_h3_update_target_rows_`; `coderef/sglang/python/sglang/multimodal_gen/test/unit/test_minimax_h3_denoise_loop.py::test_inplace_target_update_matches_scheduler_math`):
```
denoised = state + sigma * v
state    = (sigma_next / sigma) * state + (1 - sigma_next / sigma) * denoised
```
which is plain Euler on the rectified-flow line. **No noise is injected at
any step**; the class name's "ancestral" carries eta = 0. Video and audio
target rows update separately with their own sigma ratios; conditioning rows
are never touched after priming. There is no early exit; every `n-1` step
runs. *Note 2026-11-04: at sglang origin/main the scheduler class is gone:
`329072d99a` (#44690, 2026-10-19) deleted
`runtime/models/schedulers/scheduling_minimax_h3_euler_ancestral.py`, and the
update is
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py::_minimax_h3_update_target_rows_`,
the same math, pinned by `test_inplace_target_update_matches_scheduler_math`
in `coderef/sglang/python/sglang/multimodal_gen/test/unit/test_minimax_h3_denoise_loop.py`.
With the step change in section 2's note, the loop runs
`num_inference_steps` forwards, not `n-1`. Base H3 still injects no noise; the
one exception is a release whose `model_index.json` sets
`stochastic_sampling` (Pika H3 Flex, `37fc17b8d6`, #47378), which switches
both modalities to `::_minimax_h3_renoise_target_rows_`, drawing fresh noise
each step.* One positive branch, no guidance arithmetic (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py:56`;
`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py:73`).

**Cache-DiT** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py:469-623`; `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py::MINIMAX_H3_HIGH_QUALITY_CACHE_DIT_PHASES`):
mounted only for `quality="high"` with `(warmup 4, residual threshold 0.04,
max consecutive 1)`, or manually through `SGLANG_CACHE_DIT_*` when `quality`
was not explicit. Before mounting, every block's in-place residual is
disabled, because Cache-DiT snapshots the block input and an in-place rewrite
makes every residual read as zero, silently. Off under graph capture and
during warmup. The audited numbers quoted in the constant's comment are
SSIM 0.931 / PSNR 28.16 dB against lossless (reported, not verified here).
*Note 2026-11-04: the comment now names that reference `quality="exact"`,
the same path under its new name (section 10's note). The comment also
calls `exact` the default, which the sampling params contradict.*

Text refinement (the two-block refiner on the projected prompt) runs once per
request before the loop, with `cu = [0, text_len, text_len]` and no
positions; running it at a bucketed length is "not bitwise equivalent"
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py::_precompute_refined_prompt_embeds`; `coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:2850-2851`).

Refinement is not the only thing lifted out of the loop. The **RoPE tables are
built once per request too** and stashed in the branch's static kwargs, because
the position grid is step-invariant; the stage calls the model's
`build_rope_cache` when it has one and silently does nothing when it does not
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py::_precompute_rope_cache`). So the cost model for a step is the blocks plus the AdaLN lookup,
with neither the refiner nor the rotary construction inside it.

---

## 9. Decode and output

**Video** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py::MiniMaxH3DecodingStage.forward`): reverse-normalise `z*std + mean`,
decode under **fp16 autocast** by contract (`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::MiniMaxH3PipelineConfig.vae_decode_precision`)
although the VAE is fp32-resident because it also encodes keyframes
(the comment above `vae_precision` in the same class); the 36 decoder blocks' weights are cast to fp16 persistently, the
embedder, output projection and norms stay fp32 (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/vae_vit.py::ViT3DDecoder.prepare_autocast_linear_weights`, `coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/vae_vit.py:373-376,453-456`). Decode is tiled at 256 px
with 64 px overlap and, across decode ranks, whole tiles are dealt round-robin
and gathered; `spatial`, `spatial_shard` and `patch` modes are refused as
having "failed the released quality contract" (`coderef/sglang/python/sglang/multimodal_gen/configs/models/vaes/minimax_h3_video.py::MiniMaxH3VideoVAEConfig`).
Temporal decode takes chunks of 7 tokens (5 plus overlap 2) to 28 frames,
drops 3, blends the overlap, and streams into a preallocated output. Pixels
are inverse-ImageNet-normalised and clamped to `[0, 1]`, then cropped to
`latent_h*16 x latent_w*16`.
*Note 2026-11-04: at sglang origin/main the decode has a fast path gated by
the request's quality tier. At `lossless` (the default) and `high` on CUDA,
each decoder attention block fuses its QK RMSNorm and NeoX RoPE into one
in-place launch and runs on cuDNN SDPA
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_vae_cuda_opt.py`,
module docstring), and since `0498ab7b6e` (#45179, 2026-10-24) up to
`decoder_tile_batch = 64` tiles and `window_batch = 1` temporal window share
one decoder forward
(`coderef/sglang/python/sglang/multimodal_gen/configs/models/vaes/minimax_h3_video.py::MiniMaxH3VideoVAEConfig`,
whose comment says batching "moves rounding, not the math"). `exact`,
`--performance-mode memory` and non-CUDA devices decode one tile per forward
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py`,
`use_vae_fast_path(..., quality_allows(quality, "lossless"))`). Not gated by
tier: since `284cda01fb` (#41906) each block's scaled residual add is fused
into the next RMSNorm on CUDA (off on ROCm), with the residual stream kept in
fp32 to match the dtype eager autocast promotes to
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/vae_vit.py::_run_fused_transformer_blocks`;
`coderef/sglang/python/sglang/kernels/ops/diffusion/norm/h3_vae_fused_norm.py::h3_vae_scale_add_rmsnorm`).*

**Audio** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py::MiniMaxH3DecodingStage._decode_audio`): reverse-normalise `[2, 32, T]`, decode in fp32 on
**replica** rank 0, broadcast within that replica; output `[1, 2, 800*T]` at
32 kHz (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py::MiniMaxH3DecodingStage._decode_replica_audio`). This paragraph used to say "world rank 0"; that was
correct against the code as read on 2026-08-25 and is now withdrawn — sglang
moved every H3 collective off the world group on 2026-08-26, and the reason is
in section 10.

The decode also runs inside a **deterministic-algorithms scope** added the same
day (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py::_deterministic_audio_decode_context`, applied at `:351`). Its stated defect: without it cuDNN picks
convolution algorithms from free-workspace state, so the same audio latent
decodes to different bytes on a server process's *first* request than on every
later one. The scope turns off TF32 and cuDNN benchmarking and sets
`cudnn.deterministic`, but — unlike the encode-side context in section 5 — it
leaves cuDNN itself enabled, and the comment names escalating to the encode's
context as the fallback if first-request divergence returns. So there are now
three distinct determinism scopes in play rather than the two insight 4 names:
the audio encode with cuDNN off, the audio decode with cuDNN on but pinned, and
the video path with neither.

**File** (`coderef/sglang/python/sglang/multimodal_gen/runtime/entrypoints/utils.py::CudaVideoEncoder`,
`::_cuda_video_ffmpeg_command`; this read cited a line range that now holds
other code): frames `(x*255)` to
uint8 `[T, H, W, 3]` streamed in 128 MiB chunks through a CUDA-registered
memfd into
`ffmpeg -f rawvideo -pix_fmt rgb24 -r 24 ... -vcodec libx264 -pix_fmt yuv420p -crf 25 -acodec aac`;
audio through a temporary WAV after `clamp(-1, 1)`. No trimming of audio to
video: both lengths derive from the same aligned duration, so drift is bounded
by the 40 Hz rounding. The written MP4 is then re-probed and the request
**fails after the file exists** unless it has exactly one h264 yuv420p stream
at 24 fps with the resolved size and frame count, one AAC stream at 32 kHz
stereo, and audio-video duration drift within 0.25 s
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/video_adapter.py::_probe_minimax_h3_output_fields`; tolerance `coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py:74-76`).
*Note 2026-11-04, what the writer does at sglang origin/main. The command
above omits `-preset`: since `492f346d88` (#38657) it passes
`-preset fast` (`X264_PRESET` in the same file, which a request may
override since `ba68306256`, #41797). Since `4f320fb85a` (#37662)
`_resolve_ffmpeg_exe` always returns `imageio_ffmpeg`'s bundled binary,
never the `ffmpeg` on PATH, with the comment "keep the pinned encoder
binary: x264 upgrades change lossy video pixels"; the PATH `ffmpeg` and
`ffprobe` the pipeline still requires are for probing and media input. The
CRF is still 25 at the default `output_compression`, the pixel format
`yuv420p` with no color tags, and the uint8 cast is `clamp(0, 255)`
then truncation. Since `9101e895ea` (#41819) the MP4 is encoded while the
VAE decodes: a writer thread stages each chunk in registered host memory
and pipes it to ffmpeg's stdin
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/encode_while_decoding.py::EncodeWhileDecoding`).
The re-probe and its refusals hold, now read in-process with PyAV when it
imports and with `ffprobe` otherwise
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/video_adapter.py::_probe_minimax_h3_output_fields`).*

---

## 10. The runtime around the model

**Components** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py::MiniMaxH3Pipeline._required_config_modules`):
processor, tokenizer, text encoder (bf16), transformer (bf16 with the fp32
island), video VAE (fp32 resident, fp16 decode), audio VAE (fp32). No
scheduler component. Weight loads are ordered by descending size, so the DiT
(the deployment config prices it at 61.73 GB) loads before the 46 GB encoder.
Under `--performance-mode auto`, H3 opts into DiT layerwise offload
(`dit_layerwise_offload_modes=("auto","memory")`) on any card with less than
120 GiB free, and everything is kept resident above that
(`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::MiniMaxH3PipelineConfig.get_model_deployment_config`; `coderef/sglang/python/sglang/multimodal_gen/runtime/server_args/auto_tune.py::ServerArgsAutoTuner._should_auto_enable_dit_layerwise_offload`).
Layerwise-offloadable units: the DiT's refiner blocks and blocks; the video
VAE's decoder transformer blocks (the CNN encoder stays resident); the audio
VAE's encoder and decoder blocks; the text encoder's text layers plus the
vision blocks.

**Parallelism.** TP must divide 56 heads; Ulysses must divide the TP-local
heads and, with ring, 64; the packed `seq_len` must divide by
`ulysses * ring` (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3DiTModel._validate_sequence_parallel_config`, `coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:3072-3099`).
Ring is the outer contiguous row split and Ulysses the inner. CFG parallel is
refused. Since 2026-08-26 ring also **requires the FA backend specifically**:
admission raises when `ring_degree > 1` and the resolved transformer backend is
anything else (`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py:258-266`). That is an admission
gate, so it fires before any weight is downloaded rather than at the first ring
hop. Encoder folding across the replica engages when hidden >= 4096
(5120 qualifies) on one node with peer access; cross-node runs need
`--encoder-parallel replicate`. On one GPU there is no collective anywhere
(INFERENCE from the same code with `sp_ws = 1`).

Disaggregation is refused outright: the pipeline accepts only the monolithic
role (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py::MiniMaxH3Pipeline.validate_disagg_role`). The text encoder, the
transformer and both VAEs are declared `native_only`, so none
of them can fall back to a stock Transformers or Diffusers implementation
(`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::MiniMaxH3PipelineConfig.native_only_components`). MPS is admissible but only under
synchronous layerwise offload for every one of those, and with `torch.compile`
off (`:235-257`).

**Replica, not world.** Every H3 collective — the resolved-plan and encode-row
broadcasts, the shared reference-video decode, the audio decode broadcast —
targets `get_replica_group()`. It used to target `get_world_group()`, and the
comments that justified that said sglang rejected `dp_size > 1` so the world
group *was* one replica. That is no longer true: `--dp-size` is refused only
outside the monolithic role
(`coderef/sglang/python/sglang/multimodal_gen/runtime/server_args/server_args.py:4155-4167`), so a world-group collective
would now block on replicas that are not serving this request. The
replacement is `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/replica_broadcast.py:1-60`, and it
is used by the visual, audio and decode stages alike. Nothing about the single
GPU or single-replica case changes; what changes is that a multi-replica
deployment is now a supported shape rather than an impossible one.

**Batching.** H3 is `TI2V`, which does not support cross-request dynamic
batching, so the scheduler pops one request at a time; only a single
request's multi-output fan-out reaches the grouped-stage path, where text
encoding is shared by fingerprint (`coderef/sglang/python/sglang/multimodal_gen/runtime/managers/scheduler.py:1297-1329`;
`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/base.py::PipelineConfig.supports_dynamic_batching`). Warmup is off by default;
`--warmup-resolutions` turns it on but H3 substitutes its own `768 / 16:9 / 5 s`
target regardless of the values given (`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::MiniMaxH3SamplingParams.prepare_synthetic_warmup_request_for_queue`).

**Quality levels** (`coderef/sglang/python/sglang/multimodal_gen/configs/sample/sampling_params.py::QUALITY_LEVELS`, `coderef/sglang/python/sglang/multimodal_gen/configs/sample/sampling_params.py::SamplingParams.quality`):
`lossless` (default, bit-exact against CI goldens, no caching) and `high`.
`high` is a fail-closed exact match on two things: the workload
`{t2va, 1344x768, 24 fps, 124 frames, 50 steps, shift 12/3}`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py::_MINIMAX_H3_QUALITY_WORKLOAD`, `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py::MiniMaxH3PartitionAdmissionStage.forward`) and the deployment, which
`validate_quality_deployment` (`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::MiniMaxH3PipelineConfig.validate_quality_deployment`)
pins to `num_gpus 4`, `sp 4`, `ulysses 4`, `ring 1`, `tp 1`, `fa` attention,
`performance_mode speed`, no compile, no graphs, no quantisation of DiT or
encoder, no FSDP, no DiT layerwise offload, variant `fl2va`, and a device
whose name contains `H200` at compute capability 9.0. Everything the gate
requires off is therefore something sglang ships but does not stand behind
for quality.
*Note 2026-11-04: the tier names changed. Since `7a4d6dfd2f` (#42370,
2026-10-07) the tiers are `exact`, `lossless` and `high`
(`coderef/sglang/python/sglang/multimodal_gen/configs/sample/sampling_params.py::QUALITY_LEVELS`,
each including the fast paths of the ones before it, `::quality_allows`).
The bit-exact path described above as `lossless` is now `exact`; `lossless`
is the old `extra-high` (still accepted as an alias) and is still the
default, and it now mounts rounding-level fast paths for H3: the fused norm
and modulation (section 7's note), the VAE decode fast path (section 9's
note) and the in-GEMM LoRA delta. The `high` gate above holds, with the same
workload and the same 4xH200 deployment.*

**Fast paths, as shipped for H3.** torch.compile: available, off in every
mode. Breakable CUDA graphs: allowlisted, opt-in, attention stays eager, the
cookbook records a B200 ref2va run "without a measured speedup". Cache-DiT:
only through `high` or the env knobs. Attention: `fa` default; `sage_attn`,
`sol_attn`, `laser_attn`, `torch_sdpa` and others admissible because they
implement `forward_varlen`; flashinfer is not in the backend enum.
`subblock_sparse_attn` is different in kind from that list — H3 builds request
metadata for it that no other backend receives, and section 11 is its record.
Online
quantisation options include `fp8` and `kitchen_int8`; pre-quantised DiTs via
`--transformer-weights-path` cover GGUF, Comfy `fp8_scaled` and
`int8_convrot`, MXFP8, ConvRot W4A8/W4A4 and NVFP4; the cookbook lists
FP8, ConvRot, NVFP4-AWQ, Quanto int8 and GGUF overlays for the text encoder.
FSDP is a capacity option. Frame interpolation, upscaling and TeaCache are
rejected at validation. *2026-10-02: two more since `ae47bcd4da` and
`e667ab10d5`. Spectrum is a second, opt-in skip-step for H3, and
`kitchen_int8` is a deprecated alias of `convrot_int8`;
[`sglang_comparison.md`](sglang_comparison.md), "Eighth read", has both.*
*Note 2026-11-04: `3472e16f96` (#45706, 2026-10-22) removed the
`kitchen_int8` alias; at sglang origin/main only `convrot_int8` is accepted.*

**Their ComfyUI app** (`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/`): a custom-node pack
with an integrated mode for FLUX, Qwen-Image and Z-Image, and a **server
mode only** node for H3, `SGLDiffusionGenerateH3` (`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/nodes.py::SGLDiffusionGenerateH3`), which
posts to `/v1/videos` on a running `sglang serve` and polls. The README's
rule is that models needing conditioning ComfyUI cannot supply (audio,
references, task routing) go server-mode. Its inputs mirror the request
schema: task, keyframes as frame indices 0 / -1, reference image, video,
audio, seed, steps 50, short edge 768, aspect from four choices, duration
4..15 s, both shifts. It forwards `negative_prompt` if set, which the server
side fixes to `None` (INFERENCE: ignored).

*Corrected 2026-10-02: "server mode only" stopped being true at
`f1e62e3a2e` (#35990). The app now also has an integrated mode for H3: core
builds the model and runs the encoder, VAEs, conditioning nodes and sampler,
and the DiT runs one step per call in an sglang worker process. The server
node is now at
`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/nodes.py::SGLDiffusionGenerateH3`.
[`sglang_comparison.md`](sglang_comparison.md), "Eighth read", has what it
means for our patches.*
*Note 2026-11-04, the app at sglang origin/main. Integrated mode now refuses
a batch other than one for H3
(`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/minimax_h3.py::MiniMaxH3Adapter.pack`)
and, since `5730345717` (#43495), any ComfyUI model patch, block patch or
attention override, because the DiT runs in the worker and would drop it
silently
(`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/base.py::_reject_model_patches`);
so this pack's runtime patches cannot ride it. `b4822bae03` (#43504) applies
the denoise masks to the velocities before the carried-audio transform
(`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/minimax_h3.py::MiniMaxH3Adapter.unpack`)
and passes the ComfyUI step index to the DiT
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/comfyui_step.py`,
`current_timestep=step`), so sglang's Sol `dense_steps` counts ComfyUI
steps there. The worker-side H3 parity profile added by `45de4adde9`
(#44841) refuses to install unless the DiT source hashes to its reviewed
`SOURCE_SHA256`
(`coderef/sglang/python/sglang/multimodal_gen/apps/comfyui_worker_patches/h3.py`),
and that hash matches the DiT file at `b931b2c8bd`, not at origin/main. The
server node above is unchanged in kind: no reference-resize input, so it
always gets the `diffusers` default, and its `steps` default of 50 now means
50 forwards.*

---

## 11. SubBlock sparse attention

New since the 2026-08-25 read. This section exists because it is the only place
in the pipeline where H3 hands the attention layer information about *what a row
is*, and because it is the single exception to section 7's "no modality mask".
Everything here is SOURCE from a read of the current checkout; none of it was
run, and none of the upstream measurements quoted below were reproduced here.

**The backend, and where it applies.** `subblock_sparse_attn` is a
training-free block-sparse backend: a router scores 64x64 query/key block pairs
by a log-sum-exp over sub-block pairs and keeps a top-k budget, then dispatches
to SGLang's CuTe-DSL SM90 kernel or FlashInfer's `sm_100a` blk64 kernel
(`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py:1-34`, `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py::_get_subblock_sparse_attention_runner`;
router rationale at `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:1-77`). The
resolver **fails closed** on anything that is not compute capability 9.0 or
10.0, rather than falling back
(`coderef/sglang/python/sglang/multimodal_gen/runtime/platforms/cuda.py::_SubBlockSparseAttentionBackendResolver`). Inside the DiT it is scoped further
still: only modules whose prefix matches `blocks.<n>.` are eligible, so the
token refiner is always dense; head size must be 128; and it goes dense for the
first `skip_first_steps` denoise steps and any sequence shorter than
`min_seq_len` (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py:488-493,544-553`). The schedule is
`--attention-backend-config` JSON, validated on construction —
`sparsity` in `[0, 1)`, `n_k` and `n_q` each in `{1, 2, 4, 8}` (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py::SubBlockSparseSchedule`).
The shipped defaults and the arguments for them are in the module docstring
(`:68-102`), including the claim that the step cutoff and the layer cutoff behave
nothing alike: dropping the layer cutoff to zero is inside the run-to-run noise
floor while dropping the step cutoff to zero leaves the sample "essentially
uncorrelated with dense". **Reported by upstream, not verified here**, and every
one of those numbers is a rendered-clip cosine, which is the comparison class
this repo treats as a distribution rather than a pair.
*Note 2026-11-04: the fail-closed set is no longer 9.0 and 10.0 at the
clone's `c07df5a055`. Since `4e37882a93` (#37332, 2026-09-03) the resolver
also admits compute capability 12.0 and loads FlashInfer's SM120 blk64 kernel
for it; anything else still fails closed
(`coderef/sglang/python/sglang/multimodal_gen/runtime/platforms/cuda.py::_SubBlockSparseAttentionBackendResolver.supported_capabilities`).
`ffe98a4279` (#37982) and `2a0cb2f04e` (#40116) added a `compute_mode`
schedule key, `bf16` by default, whose `sage_fp8` value runs SageAttention2 on
SM90 and FlashInfer's Sage kernel on SM120, both with INT8 QK and FP8 PV; SM100
refuses it
(`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py::_get_subblock_sparse_attention_runner`).*

**What H3 adds on top: the heterogeneous call.** The generic backend would apply
the sparse budget to every query block in a long enough document. H3 does not
want that — the packed sequence's first segment holds text, keyframes,
references, audio and target video in one document, and only the *video* rows
are supposed to be approximated — the target's always, a reference video's too,
but never a reference image's and never the text. So the mask is assembled
across the pipeline and threaded to the kernel as a fifth argument:

- **Presentation** optionally tracks a per-token `video_mask`, true only for
  `<|video_pad|>` content tokens and explicitly false for the
  `<|vision_start|>` / `<|vision_end|>` delimiters, which stay dense
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/presentation.py:42-81,84-106,239-241`). Image blocks
  are not marked: an image reference in the text segment is dense.
- **The packed-sequence builders** optionally emit `video_pos`, which for
  t2va/fl2va is exactly the target video rows and for ref2va is reference
  *video* rows plus the target — reference image blocks excluded
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py:235-238,596-599`).
- **The denoise branch** unions those two, then reduces to 64-row query blocks
  with a rule that is the interesting part: a block qualifies **only if every
  one of its real rows is video**, counted against `used_len` rather than
  `seq_len` so the final partial block is judged on its real rows alone
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py:40-99`). Mixed boundary blocks
  stay dense. The result rides in the static kwargs as
  `subblock_sparse_query_block_mask` (`:335-338`).
- **The DiT's attention core** raises if sparsity is going to run and the mask
  is absent — it predicts that from `cu_seqlens_host`, the impl's own
  `_sparse_ready`, and whether any segment clears `min_seq_len`, rather than
  discovering it inside the kernel
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:1078-1106`).

Inside the kernel call the mask is not a second launch. The impl expands the
router's compact top-k index to full width, writes the router's picks into the
prefix of the selected rows and a complete `arange` into the unselected ones,
and sets a per-row `block_counts` of `topk` or `num_blocks`; one heterogeneous
call then serves both
(`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py:630-661`). SM90 additionally
requires each active prefix sorted ascending and SM100 does not, which the
dispatcher handles by sorting only for the SM90 runner (`:624-627`).

**Admission.** `MiniMaxH3PipelineConfig.uses_subblock_attention` is what decides
whether any of the above metadata is built, and it resolves the transformer's
backend by the selector's own precedence — a process-wide forced backend first,
then a `--component-attention-backends transformer=...` entry, then the global
`--attention-backend`
(`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::MiniMaxH3PipelineConfig.resolve_transformer_attention_backend`; the precedence order is pinned by
`coderef/sglang/python/sglang/multimodal_gen/test/unit/test_minimax_h3_admission.py::test_resolve_transformer_attention_backend_uses_selector_precedence`). The text-encoding and denoising
stages each call it independently, so the mask and the `video_pos` set are
produced or skipped together
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py:54-64`;
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py:702-723`). A presentation
that predates this — the code comment's own words are "legacy/precomputed" —
carries no `text_video_token_mask`, and the helper then keeps every Qwen row
dense rather than failing.

---

## 12. Insights

Numbered so they can be cited. Each is SOURCE unless marked.

1. **Attention is fully joint and unmasked.** Text, vision tokens in the text
   segment, keyframes, references, audio and video rows all attend to each
   other in one varlen segment; only padding is fenced off. Every "rows do
   not interact" intuition is wrong here. **One backend now breaks this on
   purpose** and it is the only one that does — section 11.
2. **Padding rows are real rows.** Zero-embedded, modulated as video at the
   video timestep, attending among themselves, and simply not selected at the
   output. The 64-row alignment therefore changes the sequence the model
   sees, not just its shape.
3. **The scheduler is deterministic Euler.** eta = 0 throughout; nothing
   injects noise after the initial draw. "Euler ancestral" in the class name
   is the whole of the ancestral part.
   *Note 2026-11-04: holds for base H3 at sglang origin/main, where the class
   is deleted and the update lives in `denoise_loop.py`; a release that sets
   `stochastic_sampling` (Pika H3 Flex) re-noises each step instead (section
   8's note).*
4. **Four RNG facts share one number.** Initial video noise and initial audio
   noise come from two CPU generators seeded with the same request seed
   (INFERENCE: their first values coincide in draw order); the video VAE
   encode of every keyframe and reference video is a **posterior sample at
   seed 42** regardless of the request seed, and is rounded to fp16 before
   normalisation; the audio VAE encode is the posterior **mean** in fp32 with
   cuDNN disabled. Two modalities, two determinism strategies — and, since
   2026-08-26, a third scope on the audio *decode* (section 9).
5. **Conditioning rows are noised at 0.999 with a draw that depends on the
   target.** The noise tensor is drawn at `target_latent_t + n_conditions`
   frames and prefix-sliced, so the same seed gives different anchor noise at
   a different duration or condition count, and two same-shape conditions get
   identical noise.
6. **Keyframe timestep is `max(t_video, 0.999)`**, a formula, not a
   constant; at the shipped schedules it is always 0.999.
7. **Vision tokens in the text segment take the video AdaLN branch** while
   sitting at text positions; the `<|vision_start|>` and `<|vision_end|>`
   sentinels do too. Timestamps and `<Audio j>: ` labels are text.
8. **Audio never enters Qwen3-VL**; only its label does. A silent video
   reference gets no audio label but still a zero-length audio block in the
   packed sequence.
9. **The encoder tap is unnormalised**, and DeepStack only ever touches
   decoder layers 0 to 2.
10. **The first keyframe is stretched to the canvas**, not cover-cropped;
    only the second of a first/last pair is cropped. Under explicit aspect
    ratios the first frame is distorted.
11. **Reference geometry is asymmetric.** Stills always go to a 2048 short
    edge with no area cap, upscaling included, rounded per axis (a slight
    stretch); reference videos go to a 768 short edge under the target's area
    cap and ignore `target.short_edge`; keyframes take the canvas verbatim.
    *Note 2026-11-04: at sglang origin/main the 2048 short edge is the
    default `diffusers` mode of `reference_resize_mode`; `match` and `max`
    never upscale (section 2's note). Reference videos and keyframes are
    unchanged.*
12. **The reference video is bounded by the target**, not by itself: decoded
    to the target's aligned frame count, and its soundtrack truncated to the
    target duration, against the profile comment that says otherwise.
13. **Every duration is rounded up to 17n+5 frames**, the bound `[4, 15]` s is
    checked on the request, and 15 s ships as 15.083 s.
14. **Row widths and rates are hard-coded**: 96 and 32 per row, 40 Hz audio
    latents from hop 800 at 32 kHz, 2 fps Qwen video sampling, temporal patch
    2, a 5/3-scaled `(1,4,4,4,4)` temporal grid, a 32-scaled area-normalised
    spatial grid, packed alignment 64.
15. **Position ids are per-item area-normalised**, so a reference still and
    the target never share a spatial grid; audio rows live at `h = 0` on the
    w-grid extremes.
16. **The ref2va temporal cursor** prices an image reference at one slot, an
    audio reference at `T`, a video reference at `max(T_audio, span)`, and
    times the keyframes at the target origin even though they are packed
    first.
17. **bf16 rounding points are part of the numerical contract**: Q/K norm
    output before RoPE, `1+scale`, `x*(1+scale)`, `gate*other`, the SwiGLU
    product, the final-layer input after modulation, patch-projection outputs
    on scatter.
    *Note 2026-11-04: at sglang origin/main this list is the `exact` tier's.
    At the default `lossless` the norm and `x*(1+scale) + shift` round once
    (section 7's note), and curve checkpoints modulate with fp32 tables.*
18. **AdaLN exactness depends on GEMM shape.** The cache must be built at the
    plan's own row count and TP width; a cached plan is matched by exact fp32
    equality; a novel schedule raises.
19. **The qkv row order is a load-time hazard**: official files interleave per
    head, the runtime concatenates, and per-row scales must follow or the model
    "renders noise". The 21,504-row gate is the only thing separating
    row-indexed scales from swizzled ones.
20. **`num_inference_steps` counts sigma grid points**, so a 4-step turbo is
    requested as 5 and an 8-step as 9.
    *Note 2026-11-04: no longer true at sglang origin/main. Since `32f34e898b`
    (#38671) `num_inference_steps` counts DiT forwards:
    `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/time_request.py::minimax_h3_time_shift_sigmas`
    builds `linspace(1, 0, N+1)`, the FastH3 and VDN defaults are 8
    (`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::FastH3SamplingParams`,
    `coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3_vdn.py::VDNH3SamplingParams`),
    and the PDD stage checks that `pdd_config.json`'s `num_inference_steps` equals the fused head count
    (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/timestep_preparation.py`, in `__init__`).*
21. **Video output is computed on every row and then selected**; audio output
    includes the reference rows, which the loop slices off. Communication
    volume and numerics depend on gathering after the selection.
22. **The "high" quality gate is a hardware fingerprint** (device name
    contains `H200`, capability 9.0) plus one exact flag set and one exact
    workload. An H100 fails on the name; a B200 on both. Under `auto` a card
    below 120 GiB free selects DiT layerwise offload on its own and thereby
    fails the gate without any offload flag having been passed.
23. **Partition admission runs after localisation and queueing**; the HTTP
    boundary only checks that the task name is one of three.
24. **H3 never batches distinct requests**, yet the cookbook offers encoder DP
    with `--batching-max-size 2`; only multi-output fan-out of one request
    reaches the grouped path (INFERENCE on the cookbook's intent).
25. **Output validation is post hoc and strict**: the file is written, then
    re-probed, and codec, pixel format, fps, size, frame count, sample rate,
    channels and 0.25 s drift can each fail the request.
26. **Cache-DiT silently never fires** unless the blocks' in-place residual is
    disabled first; the stage does this before mounting.
27. **Doc drift inside sglang's own tree**: the performance skill says ring
    attention and CFG parallel are incompatible and that SageAttention is
    rejected for packed multi-segment attention, while the cookbook and the
    code support ring and Sage (CFG parallel is genuinely rejected); the ref2va
    profile comment permits `7:4` and the validator does not; the audio
    profile comment says the full track is kept and the stage truncates it;
    the reference docstring says "single resample" and there are two.
    - **Half of the first clause is withdrawn as of 2026-08-30.** It used to
      read that "the cookbook and the code support ring and Sage". Ring and
      Sage *together* are now refused: admission raises when `ring_degree > 1`
      and the resolved transformer backend is not FA (section 10). Sage
      without ring is still admissible, and CFG parallel is still genuinely
      rejected, so only the conjunction changed.
28. **The 64-row alignment is load-bearing twice.** Section 6 records it as
    the packed-sequence pad; it is also exactly the SubBlock query-block size,
    so the sparsity mask is a statement about the same grid the padding is
    computed on. A block qualifies for the sparse budget only if **every real
    row in it is video**, and the count is taken against `used`, not
    `seq_len` — so the last partial block is judged on its live rows alone and
    the padding tail is never a reason to keep a block dense (section 11).
29. **H3 tells the attention layer what a row is, and only this one backend
    listens.** `video_pos` and the presentation's per-token video mask exist
    for no other purpose, are built only when the resolved transformer backend
    is `subblock_sparse_attn`, and reach the kernel as a fifth argument the
    other backends never see. A missing mask when sparsity would have run is a
    raise, not a silent dense fallback.
30. **Sparse and dense share one kernel launch.** The impl expands the
    router's compact top-k index to full width, writes a complete `arange`
    into the unselected query rows, and varies `block_counts` per row — so
    "some blocks sparse, some dense" is one heterogeneous call rather than
    two passes (section 11).
31. **The reference-video RGB is decoded once per host and mapped, not
    gathered.** ffmpeg writes into a `memfd`; the other ranks in the replica
    open that descriptor through `/proc` and map the same pages copy-on-write.
    Every failure mode falls back to per-rank decoding *together*, agreed by an
    all-gather, so the optimisation can never change the bytes one rank sees
    relative to another (section 3).
32. **The VAE encodes do not all run on all ranks.** Audio is always replica
    rank 0 plus a broadcast. Video is either every rank (parallel tiling on,
    replica wider than one) or rank 0 plus a broadcast — which means the
    seed-42 posterior sample is drawn once in one shape and once per rank in
    the other, and is identical across ranks only because the seed is pinned
    (section 5).
33. **Every H3 collective moved from the world group to the replica group on
    2026-08-26**, because `--dp-size > 1` became a supported shape and a
    world-group collective would wait on replicas that are not serving the
    request. The code comments that used to justify the world group did so by
    asserting sglang rejected DP; `server_args` rejects it only outside the
    monolithic role (section 10).

---

## 13. Coverage

### What moved between the two reads

The 2026-08-30 pass diffed `6569125e3a..5ab97c4f44` over
`python/sglang/multimodal_gen/` and read every H3-touching hunk. What it found,
and where each landed:

| change | where it is recorded |
|---|---|
| SubBlock sparse attention gets H3-specific request metadata: a per-token video mask, `video_pos`, and a 64-row query-block mask | section 11 (new); insights 28-30 |
| Ring parallelism now requires the FA backend at admission | section 10; the amendment under insight 27 |
| Backend resolution for the DiT gained an explicit precedence (forced > per-component > global) | section 11 |
| Every H3 collective moved from the world group to the replica group | sections 9 and 10; insight 33 |
| The fp32 audio decode gained a deterministic-algorithms scope | section 9 |
| The Ulysses staging buffers became one grow-only allocation per (role, dtype, device) rather than one per shape | not written: it changes allocator behaviour, not the numerics or the layout this file traces |

Mechanisms the first read missed entirely — present in both commits, absent
from the doc until now — are the shared reference-video decode (section 3), the
rank-0-plus-broadcast ownership of both VAE encodes (section 5), the
once-per-request RoPE cache (section 8), and the monolithic-role,
`native_only_components` and MPS admission gates (section 10).

Deliberately not re-derived here: the sigma-schedule construction and the
low-step / distillation-LoRA path, which are another lane's.

### What was read

Read whole by the readers: every file under `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/`, the pipeline, the sample
and pipeline configs, the DiT model file (2,600 lines), the encoder wrapper
and Qwen3-VL model files, both VAE model directories, the scheduler, the
weight loader, the BCG padder, the AdaLN cache tool, the ten H3 unit tests,
the cookbook page, the performance and fast-path skill docs, and the ComfyUI
app's node and transport code. Read in part: the generic denoising, decoding
and input-validation stages, `server_args.py` and `auto_tune.py` (cited
ranges), `usp.py`, the attention backends, `scheduler.py`, `gpu_worker.py`,
the layerwise-offload manager, and the transformers Qwen2-VL/Qwen3-VL
processors (this box's transformers 5.15.0 against sglang's pin of 5.12.1;
INFERENCE that the processor code is equivalent; *note 2026-11-04: sglang
origin/main pins `transformers==5.19.0` in `coderef/sglang/python/pyproject.toml`
since `11972e5207` (#42784), and the ComfyUI venv's
`transformers.__version__` read 5.18.0 that day*). Not read: the LoRA
machinery beyond adapter preparation, the Cache-DiT adapter internals, the
disaggregation mixin, the host-memory budget manager, MPS-only branches.

Added on the 2026-08-30 pass: the SubBlock sparse backend and its router read
whole; the CuTe-DSL SM90 and FlashInfer blk64 kernels themselves **not** read,
so every claim about what those kernels do with a routing plan is taken from
the calling code. The upstream sparsity, cutoff and estimator measurements
quoted in section 11 are **reported, not verified here** — they are rendered-clip
cosines against a dense render, which is the comparison class this repo requires
a distribution for, and the module docstring is the only record of them.

## 14. Third read, 2026-09-04: what landed between `5ab97c4f44` and `320bdd1ee2`

Fifteen commits touch the H3 path in those five days, and another ten touch
the diffusion runtime it runs on. Nothing here was run. Where a number is
given it is the PR author's, on the PR author's hardware, and is quoted only
so a reader knows one exists; none of it is comparable to a measurement here.

| change | where it lands |
|---|---|
| FastH3 as a pipeline of its own, refusing everything it was not distilled for | 14.1 |
| a third request quality tier, `extra-high`, which is a no-op for H3 | 14.1 |
| explicit per-component attention backends must be consumed or the server refuses; explicit choices are admitted by capability, not by a model allowlist | 14.1 |
| warmup renders at the served clip shape | 14.1 |
| block-FP8 checkpoints load correctly: the per-head qkv reorder now moves scale rows in blocks | 14.2 |
| third-party component bundles: partial snapshots, LoRA over runtime-quantized bases, hybrid FL2VA/Ref2VA files, native VAE overrides | 14.2 |
| the AdaLN cache moved to its own module and gained a pinned-host tier and per-plan LRU; adaln LoRA deltas and layout mismatches fail closed | 14.3 |
| a `to_gate_compress` branch per block when the checkpoint carries one | 14.3 |
| per-step forward context carries attention metadata into the DiT | 14.4 |
| three attention policies new to H3: cube sparse, VSA-H3, SpargeAttention; SubBlock gains SM120 | 14.5 |
| video VAE decoder: unfused bias on SM12.x only | 14.6 |
| profiler spans, quieter hot path, nightly performance dashboard, BCG warmup diagnostics, key masks on the replicated-prefix Ulysses path | 14.7 |

### 14.1 Request and admission (extends section 1)

**FastH3 is a separate pipeline, not a mode.**
`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::FastH3PipelineConfig` and
`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::FastH3SamplingParams` subclass the H3 ones,
set `has_gate_compress` on the arch config, and refuse: `--model-variant`,
`quality="high"`, any task but `t2va`, any step count but five grid points
(*note 2026-11-04: at sglang origin/main, since the FastH3 V2 checkpoint
`4f320fb85a` (#37662) and the forward-count change `32f34e898b` (#38671),
it accepts exactly `num_inference_steps=8`,
`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::FastH3SamplingParams`*),
and under VSA-H3 also `--ring-degree` above one, `torch.compile` and the
breakable CUDA graph. The FastVideo repository's flat Diffusers layout is
materialised into the base-H3 layout through a model overlay
(`runtime/utils/model_overlay.py`), and the `-LoRA` sibling is rejected by
`--lora-path` because it carries full-rank `.diff` and `.set_weight` tensors.
Registered under `registry.py` by repository id. *2026-09-25: half stale.
Since `ce06a14444` the config registers from
`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::register`; `registry.py` keeps the
repository-id alias.*

**Quality is now three cumulative tiers**
(`coderef/sglang/python/sglang/multimodal_gen/configs/sample/sampling_params.py::QUALITY_LEVELS`): `lossless`,
`extra-high` (request-gated kernel fusions only), `high` (plus model-owned
approximate paths such as H3's audited Cache-DiT). H3 has no request-gated
fusion site, so `extra-high` runs the `lossless` denoise path; the unit test
`test_extra_high_quality_does_not_enable_h3_cache_dit` pins that. Any explicit
`quality` takes H3 off the process-wide Cache-DiT defaults.
*Note 2026-11-04: both halves are stale at sglang origin/main. Since
`7a4d6dfd2f` (#42370) the tiers are `exact`, `lossless` (the old
`extra-high`, still an alias, and still the default) and `high`; the old
`lossless` is now `exact`. H3 now has request-gated fusion sites at
`lossless`: the DiT's fused norm and modulation (`121aa8dcc7`, #43327), the
video VAE decode fast path and tile batching, and the in-GEMM LoRA delta
(sections 7, 9 and 10's notes). The test is now
`test_lossless_quality_does_not_enable_h3_cache_dit` in
`coderef/sglang/python/sglang/multimodal_gen/test/unit/test_minimax_h3_admission.py`;
Cache-DiT is still `high` only.*

**Attention backend selection became a contract.** Two commits: explicit
per-component requests are tracked per exact component and must be consumed
at construction or through a declared deferred first use, else the server
refuses rather than falling back silently
(`runtime/models/dits/minimax_h3.py` now calls
`claim_deferred_component_attn_backend` for the DiT's lazy resolution); and
a model's `_supported_attention_backends` set is an automatic-selection hint
only, so an explicit global, component or per-request backend is admitted
after platform and layer capability checks even when the model never listed
it. Section 11's precedence (forced, then per-component, then global) stands.

**Warmup at the served shape.** `MiniMaxH3SamplingParams._synthetic_warmup_target`
derives the warmup clip's duration from `--warmup-num-frames` and its aspect
ratio from `--warmup-resolutions`; before this the synthetic warmup always
rendered the default five-second 768p clip and the first served forward at
another length paid allocator growth.

**Admission also rejects** a step count that would overflow the online AdaLN
GPU plan slab (14.3), naming the environment variable that resizes it.

### 14.2 Loading (extends section 10)

**Block-FP8 checkpoints.** H3 stores qkv grouped per head; sglang permutes
the rows to `[q_all, k_all, v_all]` on load and permutes row-indexed scales
the same way (section 12 already records this). A block-FP8 scale is
row-indexed in blocks, so `_qkv_scale_block_rows` now scales both the
permutation and the row-count gate down by the block height for a
`BlockQuantScaleParameter`, and a block that straddles two heads' rows raises
instead of silently mis-scaling. The PR's motivation is the failure class this
repo's section on the refuted qkv hypothesis describes: the model loaded
without error and rendered blank frames.

**Third-party bundles** (`loader/component_loaders/vae_loader.py`,
`loader/minimax_h3_weights.py`, `layers/lora/linear.py`): a partially cached
Hub snapshot is completed file by file; a LoRA stays dynamic when a runtime
quantization method owns the base weight, fp8 storage included; explicitly
selected hybrid FL2VA/Ref2VA filenames are accepted; native VAE weight
overrides load with folded weight norm and checkpoint-carried latent
statistics. Companion commits preserve exact component identity through
loading, add exact per-component precision overrides, detect quantized
transformer replacements, and filter duplicate precision variants across
custom loaders.

### 14.3 The DiT forward (extends section 7)

**The AdaLN cache is its own module now**,
`runtime/models/dits/minimax_h3_adaln_cache.py`, a pure relocation of the
class section 7 describes, with two new tiers under `--minimax-h3-adaln-online`:
the GPU slab evicts per plan by LRU with in-flight protection instead of
resetting whole, and a pinned-host tier
(`--minimax-h3-adaln-host-cache-gb`) holds plan sets by schedule so a
schedule seen before is served without re-reading the checkpoint's
`adaln_proj` weights. Plan slots are resolved on the host in
`prepare_adaln_plans` and passed to the forward as a device scalar, which
removes a per-step device lookup and its capture-breaking host sync. Four
traps now fail closed: adaln LoRA deltas in either cache mode
(`_reject_adaln_lora`, `validate_lora_layers`), Diffusers-layout checkpoints
as a rebuild source, and weight updates the cache cannot follow
(`validate_weight_update_source`, `refresh_weight_derived_caches`). The
default numerics path is stated to stay bit-exact with resident weights; the
env-gated fp32 rebuild changes bits by construction.

**A compression gate per block.** When `arch.has_gate_compress` is set (the
FastH3 checkpoints), each main block owns a `to_gate_compress`
`ColumnParallelLinear`, bf16 and unquantized, whose output is threaded
through the Ulysses all-to-all beside q, k and v and handed to the attention
backend. Base H3 has no such weights, so the gate loads as zeros and the
VSA-H3 backend runs pure sparse. The two token-refiner blocks are constructed
with `cube_sparse_capable=False`, so the cube backend refuses to attach there.
The fused `qk_norm` path is now gated to NVIDIA devices.

### 14.4 The denoise loop (extends section 8)

`denoise_loop.py` sets a forward context per step
(`coderef/sglang/python/sglang/multimodal_gen/runtime/managers/forward_context.py::set_forward_context`, `current_timestep` plus
`attn_metadata`) when a sparse backend supplied metadata, and passes the AdaLN
plan slot as a device tensor. `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py::_build_cube_attn_metadata`
builds the cube metadata once per request from the packed sequence's new
`stream_layout` (target shape; per-condition image shapes with a role of
`joint_cube` for FL2VA keyframes on the target timeline, `dense_prefix` for
standalone reference images, `independent_cube` for reference videos; event
order; audio stream lengths), while
`_maybe_prepare_vsa_h3_step_metadata` builds VSA-H3 metadata per step and
refuses the ref2va layout outright. Both hang off
`packed_sequence.py`, which is where the cube geometry of a request is decided.

### 14.5 Attention policies (new; section 11 covers SubBlock)

Four now exist for H3 beyond the dense backends, sage, and Sol. Sol itself
is described in 14.9, added 2026-09-11; this section named it and did not
read it.

**Cube sparse attention**
(`layers/attention/backends/cube_sparse_attn/{mask,backend}.py`; merged
2026-09-02 with two MiniMax engineers as co-authors). FlexAttention over
`(T, H, W)` cubes of latent tokens; the mask module is kernel-agnostic and
separates three index spaces it documents at the top of `CubeLayout`
(semantic label, physical block, token), because a semantic cube can own more
than one physical block when a keyframe shares a target coordinate. Per
denoise update a `topk_ratio_list` entry, one per update, where a ratio of
one runs the native dense path. Text, audio, standalone reference images and
the token refiner stay dense; reference videos and the target share one
global top-k pool. The PR's recommended schedule for the base fifty-step
grid is dense for the first two updates and then decays through the run to a
small floor, which is a step policy stated by the vendor's own engineers and
is the first such schedule published for H3. Ring parallelism refused;
FlexAttention's routing overhead can exceed the saving on short sequences,
by their own warning.

**VSA-H3** (`layers/attention/backends/video_sparse_attn_h3.py` with
`vsa_h3_kernels.py`; adapted from FastVideo). Segment-pure prefix tiles plus
`(4, 4, 4)` video tiles over the packed `[text | keyframes | audio | video]`
sequence; an in-tree Triton tile-64 kernel gated to SM90, SM100 and SM103.
*Note 2026-11-04: `465ddf9758` (#43823) added SM80 through a raw-pointer
Triton path; `coderef/sglang/python/sglang/multimodal_gen/runtime/platforms/cuda.py::VSA_H3_CAPABILITIES`
still excludes compute capability 8.9.*
Non-video queries are always dense; non-video keys are either always kept
(`vsa_mode=exempt`) or compete in the top-k under a FLOP-matched budget
(`compete`). `vsa_dense_first_n_steps` and `vsa_dense_layers` are the step
and layer escapes. Only the DiT runs sparse.

**SpargeAttention** (`layers/attention/backends/sparge_attn.py`). Wraps the
upstream `spas_sage2_attn_meansim_topk_cuda` with one knob, `topk` in (0, 1];
SM80 through SM90, fp16 or bf16, head dim 64 or 128, no GQA; anything not
square self-attention over at least one 128-row block falls back to dense
SDPA. The PR states it is approximate even at `topk=1` because the kernel
includes SageAttention2's quantization, and its own LTX measurement found no
single-GPU speed or memory win over FlashAttention. Registered for generic
DiTs; whether H3's packed-varlen path reaches it is not established by this
read.

**SubBlock on SM120** (`layers/attention/backends/subblock_sparse/`):
FlashInfer's SM120 blk64 operator joins the SM90 CuTe and SM100 paths, the
resolver accepts exactly 9.0, 10.0 and 12.0, and the README now says to put
`torch_sdpa` on the text encoder on SM12.x. The PR's numbers are on eight
RTX PRO 5000 cards.

### 14.6 Decode (extends section 9)

`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/base_module.py::_unfused_bias_linear`
runs the decoder's `w2` as a plain matmul plus a separate bias add on SM12.x
only, because cuBLAS on a GB10 picks a slow kernel for the fused epilogue at
that shape. Every other architecture takes the unchanged fused path.

### 14.7 The runtime around the model (extends section 10)

`maybe_record_function` profiler spans now wrap request phases in the
scheduler, GPU worker and pipeline executor; the hot-path logs in the H3 and
generic denoising stages were quieted; the breakable-CUDA-graph runner gained
frame-count diagnostics for video warmup; the nightly performance dashboard
was made robust; and `USPAttention` learned to apply a 2D key mask on the
replicated-prefix Ulysses path, which is what let breakable CUDA graphs and
sequence parallelism coexist for Qwen-Image and is multi-GPU-only.

### 14.8 What was and was not read

Read as diffs with their PR bodies: every commit in the table above. Read
whole: the cookbook page as it stands, the attention-backends and quality-tier
docs, the SubBlock README, `sparge_attn.py`, the cube mask module's
dataclass docstrings and the backend's head, the VSA-H3 backend's head, the
AdaLN cache module's head, the FastH3 config and sampling classes, the
admission test additions. Not read: the Triton VSA kernel, FlexAttention's
internals, the FlashInfer SM120 operator, the body of the cube mask module
past its docstrings, the cookbook's Ascend and AMD sections, and the
consumer-card sections, which the diff shows unchanged since 2026-08-30 and
which [`sglang_comparison.md`](sglang_comparison.md) already prices.

### 14.9 The Sol-Attn backend, read 2026-09-11 at `593c7a900d`

`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/sol_attn.py`
(a bare `:line` below means this file), added by `51470b376f` (#33702, 2026-08-09)
and reworked by `63d783bbe0` (#34581, 2026-08-18), which added the Sage dense
path; `28262c20df` (2026-09-02) is formatting only. It wraps NVLabs'
`sol_attn` package, installed from Sana's `sol-engine` branch
(`coderef/sglang/docs/docs/sglang-diffusion/attention_backends.mdx`, the `sol_attn` row of "Backend options"), and
the backend resolves only when that package imports, and raises otherwise
(`_SolAttnBackendResolver`,
`coderef/sglang/python/sglang/multimodal_gen/runtime/platforms/cuda.py::_SolAttnBackendResolver`).

**Opt-in.** Selected with `--attention-backend sol_attn`, or for the DiT
alone with `--component-attention-backends ... transformer=sol_attn`. The
default DiT backend stays `fa`
(`coderef/sglang/docs/docs/sglang-diffusion/attention_backends.mdx`, "Sage then Sol hybrid": "The default DiT backend remains `fa`.").

**Configuration**, read by `_get_sol_attn_runtime_config` from
`--attention-backend-config` (`:59-83`; the documented table at
`coderef/sglang/docs/docs/sglang-diffusion/attention_backends.mdx`, the "Sol-Attn (`sol_attn`)" table under "Supported Configuration Parameters",
gives the same defaults):

| key | default | where |
|---|---|---|
| `tau` | 1.0 | `:74` |
| `thresh_type` | `diag` (the other mode is `exact`) | `:75` |
| `kv_splits` | `auto`, resolved by `_resolve_kv_splits` (`:44-56`) | `:76` |
| `sink_tokens` | 0 | `:77` |
| `sink_start` | 0 | `:72` |
| `dense_steps` | 10 | `:79` |
| `dense_layers` | `"0,1"`, parsed as indices and ranges | `:80`, parser `:26-41` |
| `dense_backend` | `fa`; `sage_attn` is the other, with `sage` and `sageattention` as aliases | `:62-71` |

**When a call runs dense.** `_should_use_dense` (`:128-140`)
returns true when the step index is below `dense_steps` or the module's layer
index is in `dense_layers`. The step is the denoise loop's index, not a
sigma: H3's loop passes `current_timestep=step` for `step in range(num_steps)`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py::minimax_h3_denoise_loop`;
the line numbers this read cited have drifted).
So `dense_steps` is a count, the same at any step count or shift. There is no
end gate: past `dense_steps`, every step through the last runs sparse.
*Note 2026-11-04: holds at sglang origin/main, where `sol_attn.py` is
unchanged since `593c7a900d` and the loop index now runs over
`num_inference_steps` forwards (section 8's note); so at the default
`dense_steps` of 10, all six steps of a Pika H3 Flex request are dense.* The
dense path is FlashAttention varlen (`_dense_fa`, `:142-162`) or
SageAttention per sequence (`_dense_sage`, `:164-202`).

**The layer index is a name match.** `_parse_layer_idx` takes the first
`blocks\.(\d+)` in the module prefix (`:124-126`). H3's DiT file names its
blocks `blocks.<i>`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:2487`)
and its token-refiner blocks `token_refiner.blocks.<i>` (the same file at
lines 1769 and 2480), two of them
(`coderef/sglang/python/sglang/multimodal_gen/configs/models/dits/minimax_h3.py::MiniMaxH3DiTArchConfig.token_refiner_num_layers`).
The refiner's attention resolves its backend the same way the DiT's does;
only the cube backend is sent back to FlashAttention for it (the DiT file at
lines 1232-1241 and 1727-1733). So refiner block `i` is gated by the same
`dense_layers` entry as DiT block `i`, and at the default `"0,1"` both
refiner blocks run dense at every step. Read from source, not run; nothing in
the backend file excludes the refiner or a short sequence on its own.

**The sparse call** (`_run_sol_attn_thd`, `:204-235`) requires
bf16 activations (`:216-217`) and head dim 128 (`:22`, checked when the impl
is built), and passes `tau`, `thresh_type`, `kv_splits`, `sink_start` and
`sink_tokens`, keeping only the arguments the installed package's signature
accepts. `int8_qk=True` is added when the package has that parameter and the
device is SM89 or newer; the comment beside it calls that the Wan2GP Ada
port and says NVLabs' official API has no such argument (`:229-233`).

**What it does not have.** No token routing: `token_aug` and `extra_tokens`
appear nowhere in the file. No sink derived from H3's layout: `sink_start`
and `sink_tokens` are fixed numbers from the config, and a grep of
`multimodal_gen` for `sink_tokens` finds only the SANA-WM refiner and the
causal KV cache, neither on this path. So by default no conditioning rows
are held exact. No minimum sequence length.

**The documented H3 recipe** puts the dense prefix on Sage and everything
after it on Sol: `--attention-backend-config dense_backend=sage_attn,dense_steps=10`
with `--component-attention-backends text_encoder=torch_sdpa,transformer=sol_attn`
(`coderef/sglang/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx`, "9. Feature contracts and advanced recipes";
*Note 2026-11-04: this read cited lines 1116-1119; the cookbook has grown and that line range now holds other
text. At sglang origin/main `8884b4b421` the recipe is in "9. Feature contracts
and advanced recipes", in the paragraph after "On a single 24 GB card, use
`convrot_int8` instead of FP8", starting "`fa` keeps exact attention. For a
faster, approximate DiT path"*;
the same example at
`coderef/sglang/docs/docs/sglang-diffusion/attention_backends.mdx`, "Sage then Sol hybrid",
which calls both
`sage_attn` and `sol_attn` approximate; *note 2026-11-04: that page grew too (this read cited lines 868-885),
and the example is now under its "Sage then Sol hybrid" heading, which still
says both are approximate and the default DiT backend remains `fa`*).

Tests: `coderef/sglang/python/sglang/multimodal_gen/test/unit/test_sol_attn_backend.py`
has four, for the layer-range parser, the dense-backend aliases, the
early-step dense gate, and a sparse layer after it.
