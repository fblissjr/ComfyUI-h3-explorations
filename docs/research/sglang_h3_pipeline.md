# sglang's MiniMax-H3 pipeline, end to end

Written 2026-08-25 from a source read of `coderef/sglang` at commit
`6569125e3a` (2026-08-25). Nothing was run: no server, no GPU, no model
load. Five scoped readers each took one slice (admission and media; the
conditioning encode; the packed sequence and DiT; the denoise loop, VAEs and
output; the runtime layer), read the files whole, and reported with line
citations they had verified. This file is the synthesis. It does **not**
compare anything to ComfyUI; [`sglang_comparison.md`](sglang_comparison.md)
owns that and should be read after this one, not instead of it.

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

Stage composition: `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py:122-170`.
The constructor refuses to start without `ffmpeg` and `ffprobe` on PATH
(`:52-67`).

---

## 1. Request and admission

**Tasks.** Exactly `t2va`, `fl2va`, `ref2va`, case-sensitive, never inferred
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/video_adapter.py:68,80-84`; `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py:300`). There is no
`i2va`: a first frame is `fl2va` with `frame_index: 0`; video-to-video is
`ref2va` with a `video` or `video_audio` reference. Keyframe signatures are
the ordered tuple of `frame_index` values and must be one of `(0,)`, `(-1,)`,
`(0,-1)` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py:72-76`), enforced at validation, plan
resolution, canvas preparation and text encoding alike.

**Canonical request** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py:278-378`):
`{schema, task, prompt, conditions, target[, flow_shift, audio_flow_shift, seed]}`.
`target` is `{short_edge, aspect_ratio, duration_seconds}`; unknown `target`
keys are **silently dropped** before validation (`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py:189-198`),
while an unknown key in a `conditions[]` entry is an error (`:41-43,190-192`).
`seed` is in `[0, 2^63-1]`.

**What the HTTP layer refuses outright** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/video_adapter.py:119-167`):
explicit `num_frames` or `fps`; `guidance_scale`, `guidance_scale_2`,
`true_cfg_scale`, a non-None `negative_prompt`; `audio_guidance_scale`;
`enable_frame_interpolation`, `enable_upscaling`; any `output_mode` other than
decoded files. The sampling params also reject TeaCache, rollout and
trajectory output (`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py:212-226`).

**Per task** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py:148-242`):

| task | conditions | duration | aspect ratio |
|---|---|---|---|
| `t2va` | none | required | `auto` -> `16:9`; explicit only from `21:9, 16:9, 4:3, 1:1, 3:4, 9:16` |
| `fl2va` | 1..2 keyframes (`image`, `frame_index`) | required | `auto` deferred to the first keyframe's probed ratio; explicit `W:H` with any positive integers |
| `ref2va` | >= 1 of keyframe/image, reference/image, reference/video, reference/video_audio, reference/audio | optional when exactly one audio-bearing condition | `auto` -> `16:9`; explicit from the six; references never bind geometry |

The finite aspect list is enforced only for t2va and ref2va
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py:102-106`); the ref2va profile's own comment claims
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
`task_aliases`, `sigma_shift_scales`; `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py:46-119`). The
static map is `t2va -> fl2va, fl2va -> fl2va, ref2va -> ref2va`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py:32-38`). One partition per process: `--model-variant`
selects the subfolder and is cross-checked against the loaded partition
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py:87-113`). The gate runs as
pipeline stage 2 (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py:140-213`), which is **after** the
media were localised, probed and queued.

**Seeds.** Per-output seed is `seed + output_index`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/entrypoints/utils.py:352-359`); the latent stage falls back to 42
only when the plan carries none (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/latent_preparation.py:94-96`).

---

## 2. Time grid and canvas

Constants: `fps = 24`; duration `[4.0, 15.0]` s checked on the **request**
value (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py:50-52`; `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py:127-136`).

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

**Sigmas** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/time_request.py:32-59`): `base = linspace(1, 0, n)`,
`sigma = s*base / (1 + (s-1)*base)`, `unique_consecutive`, terminal 0
guaranteed. `num_inference_steps` counts grid points including the terminal
zero, so the loop runs `n-1` model calls; `n >= 2` is enforced at admission
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py:150-154`). Shift priority per modality: request >
`model_index` `sigma_shift_scales` > profile default, and the defaults are
`12.0` video, `3.0` audio for every task (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/timestep_preparation.py:118-153`;
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py:153-154,175-176,228-229`). Default 50 steps. `t = 1 - sigma`.

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
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/latent_preparation.py:84-85,97`). Any positive integer short edge
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
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:125-178`). The prepared PIL feeds both Qwen3-VL
and the VAE (`:840-915`).

**Reference videos** go through `adapt_shape_v1` with the base short edge
fixed at 768, ignoring `target.short_edge` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/prequeue.py:290-302`), decoded
with a direct non-cropping `scale=W:H` filter.

---

## 3. Media ingestion

URIs (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/material_io.py:761-878`): `file://` (localhost only), bare paths,
`http(s)` (120 s timeout, 1 MiB chunks, suffix from URL or Content-Type),
`data:`/`base64://` with a streaming validating decoder, `tar+offset://`;
`s3://` raises `NotImplementedError`. Sources and probe facts are cached per
request in a temp dir registry keyed by owner (`:57-89`), which pre-queue
renames from `material` to `prequeue_material` so multi-output expansion does
not delete the shared source after the first output (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/prequeue.py:227-238`).

Probe (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/material_io.py:352-529`): images via PIL, **JPEG/PNG/WEBP only**,
display size after `exif_transpose`; audio/video via `ffprobe` with a
container allow-list (`mov, mp4, m4a, 3gp, 3g2, mj2, matroska, webm, wav,
mp3, flac, ogg`), stream requirements per type, duration = max of container
and stream durations, display geometry corrected for SAR, DAR and rotation
(`:247-289`).

Decoding for the encoders:

- **Reference video frames**: one ffmpeg pass
  `[-ss start] -i path -map 0:v:0 -an -vf fps=24,scale=W:H:flags=lanczos,setsar=1 -frames:v N -f rawvideo -pix_fmt rgb24`
  -> `uint8 [T, H, W, 3]` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:368-438`), where `N` is the
  **target's** aligned frame count (`:730-748`), so a reference video is
  bounded by the target, not by itself. The same array serves Qwen3-VL
  (every 12th frame, i.e. 2 fps) and the VAE.
- **Audio**: `ffmpeg [-ss] -i -map 0:a:0 -vn -ac 2 [-ar 44100 for video chains] [-t target_duration] -f f32le`
  -> `[2, N]` float32 (`:207-284`), then `torchaudio.Resample(src, 32000)` when
  needed (`:287-291`). A video soundtrack is therefore resampled twice
  (ffmpeg to 44.1 kHz, torchaudio to 32 kHz) despite the docstring's "single
  resample", and it is **truncated to the target duration**
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/audio_encoding.py:147-152`), contradicting the profile comment that
  says the full track is kept (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py:208-210`).
- A silent `video` reference yields a zero-length audio entry `[0, 32]` to
  keep block order (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/audio_encoding.py:175-184`); `video_audio`
  without an audio stream is refused at probe.

---

## 4. The Qwen3-VL conditioning encode

**Presentation** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/presentation.py`). No chat template, no system prompt,
no `<|im_start|>`, no BOS or EOS: the stage builds raw ids with
`tokenizer(text, add_special_tokens=False)` and hand-made vision blocks
`[<|vision_start|>] + [pad]*n + [<|vision_end|>]` (`:30-39,85-89`). The
prompt passes through verbatim. Per task:

| task | stream |
|---|---|
| t2va | `prompt` |
| fl2va | for each keyframe `"<Picture i>: "` + image block; then `prompt` |
| ref2va | per condition in request order: image -> `"<Picture i>: "` + image block; audio -> `"<Audio j>: "` **label only**; video -> `"<Video k>: "` then per temporal block `"<t seconds>"` (`.1f`) + video block; then `prompt` |

(`:64-82,92-134,231-262`). Ordinals are per type and 1-based. A video with a
soundtrack emits its `<Audio j>: ` label **before** its `<Video k>: `
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py:428-445`). ref2va keyframes are deliberately
omitted from the presentation (`:358-364`). Verified in this repo's copy of
the release tokenizer: its chat template writes `Picture N: ` without angle
brackets under `add_vision_id`; H3 writes `<Picture i>: ` with them and never
invokes the template.

**Token tags** travel with the ids: text, labels and timestamps are tag 1;
vision blocks **including the start and end sentinels** are tag 0, the video
modality (`:42-61`). The tags later overwrite the packed sequence's text-slot
tags (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py:643-646`), so vision tokens inside the text
segment are modulated with the video AdaLN branch while sitting at text
positions.

**Vision blocks.** The release processor files set image bounds
`65,536..16,777,216` px, video bounds `4,096..25,165,824`, patch 16, temporal
patch 2, merge 2, mean and std 0.5 (this repo's `vendor_config/preprocessor_config.json`
and `video_preprocessor_config.json`). The stage calls `image_processor` and
`video_processor` directly, never the combined processor, and passes
`do_sample_frames=False` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py:329,460,491-496`).
Pixels become `(x/255 - 0.5)/0.5` in `[-1, 1]`; a still is **duplicated to
T=2** for the Conv3d patch embed; tokens per still `= (H/32)*(W/32)`, so a
768x1344 canvas is 1,008 tokens and a 2048x2048 reference 4,096. Video: frames
sampled at stride 12, timestamps `i/2` padded to even by repeating the last,
block timestamps are pair midpoints (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:683-718`;
pinned at 25 frames -> `[0.25, 1.0]`, `coderef/sglang/python/sglang/multimodal_gen/test/unit/test_minimax_h3_media.py:64-105`).
`pixel_values` are cast to bf16 on the device (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/encoders/minimax_h3_qwen3vl.py:347-380`).

**The encoder** (`coderef/sglang/python/sglang/multimodal_gen/configs/models/encoders/minimax_h3_qwen3vl.py:11,21-53`;
`coderef/sglang/python/sglang/multimodal_gen/runtime/models/encoders/minimax_h3_qwen3vl.py:46-52,258-263,274-281`):
Qwen3-VL-32B with `num_hidden_layers` forced to 50, `use_cache` off, hidden
5120, 64 heads, 8 KV heads, head dim 128; `language_model.norm` replaced by
`Identity`; layers >= 50, `lm_head` and the final norm dropped at load. The
tap is the residual after decoder layer index 49, **unnormalised**
(equivalent to HF `hidden_states[50]`). Precision bf16
(`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py:67`). Batch of one; M-RoPE
positions computed on CPU by `get_rope_index` when any grid is present, else
plain `arange` on all three axes for text-only (`:358-365`). DeepStack
features from vision blocks `[8, 16, 24]` are added after decoder layers 0, 1
and 2 (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/encoders/qwen3vl.py:631-641,666-677`). Output:
`hidden_states [L, 5120]` bf16, `text_len`, `text_token_tags [L]` into
`extra["minimax_h3_text_embeddings"]` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py:289-298`).
Encoding is deduplicated by `(task, prompt, materials, shape)` within a
grouped request (`:81-107`), and requests are distributed one whole request
per encoder copy under encoder DP (`:109-190`).

---

## 5. Reference and keyframe encodes

**Video VAE** (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3.py:20-87`;
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
generator then moved (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/klvae.py:73-87,1294-1308`). So every keyframe and
reference-video encode forks the RNG and seeds it to **42**, independent of
the request seed (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/keyframe_encoding.py:31-32,36-72`; `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:613-614`;
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/visual_encoding.py:243-244`), with `use_fp16_latent=True`, which
rounds the latent to fp16 before normalisation (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/klvae.py:994-995`). Rows are
patchified `[1,2,2]` to `[n, 96]` fp32 on CPU (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_tokens.py:23-41`).
A 17n+5-frame video encodes as n+1 clips of 17 (last padded by repeating the
final frame), 5 tokens each, minus 3 dropped, giving 5n+2 (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/klvae.py:545-581`).

**Audio VAE** (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3.py:90-113`): DAC-style
encoder with rates `[2,4,4,5,5]`, hop 800 samples at 32 kHz, i.e. 40 latent
frames per second; latent dim 2048 projected by a causal attention block to
32 channels; BigVGAN decoder with rates `[5,5,2,2,2,2,2]`, output
`clamp(-1, 1)` rather than tanh. Encode takes the **posterior mean**
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:294-362`) under a determinism context that disables
TF32 and cuDNN and pins SDP to the math kernel (`:53-118`); rows are
`[2T, 32]` channel-major (all left, then all right).

**Condition noise augmentation** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/condition_noise.py`; defaults in
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py:24-26`): visual conditions at `0.999`, audio references at
`1.0`, overridable per request in `[0, 1]`. Applied only below 1.0: for each
visual condition a fresh CPU generator seeded with the request seed draws
`randn(1, 24, target_latent_t + n_conditions, H, W)` and keeps the `[:T_c]`
prefix, so the anchor noise depends on the target length and on how many
conditions there are, and two same-shape conditions get identical noise
(`:91-119`); audio uses `seed + 1` (`:168-188`). The mix is
`t*clean + (1-t)*noise`. At the defaults, keyframes carry 0.1 % noise and audio
references are untouched.

---

## 6. The packed sequence

File `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py`. Constants (`:30-35`): spatial scale 32, temporal
group 5, `frame_per_token (1,4,4,4,4)`, frame rescale `5/3`, patch 2x2;
alignment 64 (`coderef/sglang/python/sglang/multimodal_gen/configs/models/dits/minimax_h3.py:6`).

**Layout.** t2va/fl2va: `[text | keyframes | target audio | target video | pad]`
(`:118-219`). ref2va: `[text | keyframes | reference blocks in request order | target audio | target video | pad]`,
where a video reference contributes its audio rows first, then its visual
rows (`:274-528,391-396`). `seq_len = ceil(used / 64) * 64`. `update_mask` is
False on every conditioning row; `audio_update_mask` is False on reference
audio rows. `cu_seqlens = [0, used, seq_len]`: one attention segment for the
live rows and one for the padding tail (`:209,517`).

**Position ids** `(t, h, w)` per row (`:169-202,238-259`): text `(i, 0, 0)`;
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
references, are timed at the target origin (`:411-498`). Standalone audio
references borrow the **target** w-grid extremes; a video reference's audio
uses its own grid (`:438-439,461-465`).

**Tags and timesteps.** Tags: pad -1, text 1, audio 2, image/video 0
(`:204-207`). Per denoise step at most four timestep values exist
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py:290-353`): `t_video` for video, text **and padding** rows;
`max(t_video, 0.999)` for keyframe and visual reference rows; `t_audio` for
target audio; `max(t_audio, 1.0)` for audio references. They are deduplicated
by exact fp32 equality on the host, and each row's modulation index is
`tag + 3 * timestep_index` (`:320-327`; modality count 3,
`coderef/sglang/python/sglang/multimodal_gen/configs/models/dits/minimax_h3.py:7`). Widths: t2va 2, fl2va 3, ref2va 4.

Tests pin the row counts, the keyframe anchors to twelve places, the
rejection of any other signature, and the ref2va block order
(`coderef/sglang/python/sglang/multimodal_gen/test/unit/test_minimax_h3_packed_sequence.py:13-184`).

---

## 7. The DiT forward

**Architecture** (`coderef/sglang/python/sglang/multimodal_gen/configs/models/dits/minimax_h3.py:79-99`;
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
else bf16 (`:140-155`); the island is never quantised. Pruned checkpoints
(`MiniMaxH3PrunedTransformer3DModel`) replace the time embedder with a curve
table `adaln_t_table [grid, rank]` read by linear interpolation
(`:2040-2051`; test pins 8 / 1025 / 2688).

**RoPE** (`:402-441`): `inv_freq [16]` is a persistent buffer allocated empty
and loaded from the checkpoint (INFERENCE: nothing in the tree computes it);
per axis `pos * inv_freq` gives `[S, 3, 16]`, concatenated to 48 and doubled
to 96, so rotation covers head dims `[0, 96)` in neox rotate-half pairing and
dims `[96, 128)` pass through. Q and K are RMS-normalised per head and rounded
to bf16 **before** rotation (`:909-922`).

**Forward** (`:2311-2590`). `_embed` (eager under graph capture) scatters
text rows (already refined once per request), video rows through
`video_patch_proj` and audio rows through `audio_patch_proj` into a bf16
`[local_seq_len, 5376]` buffer; padding rows are zero. `adaln_input = silu(t_emb).bf16`.
Tags are clamped at 0, so **padding rows are modulated as video rows at the
video timestep** (`:2478`). Each block, for every row identically:
```
h = norm1(x); h = h*(1+scale_msa) + shift_msa; x = x + gate_msa * attn(h)
h = norm2(x); h = h*(1+scale_mlp) + shift_mlp; x = x + gate_mlp * mlp(h)
```
with the CUDA kernels rounding `1+scale`, `x*(1+scale)`, `gate*other` and the
SwiGLU product to bf16 (`coderef/sglang/python/sglang/multimodal_gen/runtime/kernels/ops/diffusion/modulate/indexed_modulation_triton.py:39-45,78-82`).
The first gated residual is in place unless Cache-DiT input preservation is
armed (`:1521-1528,1971-1985`).

**Attention** is one FlashAttention varlen call over the whole packed
sequence: `cu_seqlens_q = cu_seqlens_k = [0, used, seq_len]`, `max_seqlen = used`,
non-causal, scale `128^-0.5` (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/flash_attn.py:451-474`).
**Every live row attends to every other live row with no modality mask**; the
padding tail attends only to itself. Backend on CUDA: FA (fa3; fa4 on
Blackwell), TORCH_SDPA on SM12.x, cuDNN SDPA on Blackwell when FA is present
(`coderef/sglang/python/sglang/multimodal_gen/runtime/platforms/cuda.py:531-639`); any backend implementing
`forward_varlen` is admissible. Under Ulysses the all-to-all happens inside
the attention core, so each rank holds the full sequence with `56/ulysses`
heads (`:555-620`; `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/usp.py:323-372,527-586`); under ring,
K/V rotate and each hop attends only the remote chunk's real prefix, FA only
(`:588-604`; `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/usp.py:791-903`).

**Final layer** modulates by timestep only, casts to fp32, and computes
`video_out` and `audio_out` on **all** rows; the video logits are then
`index_select`ed to the target rows and the audio logits cover every audio
row (`:1595-1653,2539-2590`). Under sequence parallel the rows are gathered
after selection, and under TP the columns after that.

**AdaLN plans and cache** (`:1084-1362,1732-1745`). The per-request set of
distinct timestep tuples is computed once before the loop; the cache stores
`block_params [P, W, 50, 96768]` bf16 and `final_params [P, W, 10752]` and
looks them up by exact fp32 match. Building it means one `F.linear` per layer
at the plan's own row count and the TP-sharded width, because cuBLAS picks
kernels by shape and any other shape "silently perturbs the output"
(`:1229-1240,1278-1282`); the online rebuild reads all of `adaln_proj`,
which the flag's help text prices at 24.2 GiB. Unquantised, non-curve
checkpoints only. The standalone tool `coderef/sglang/python/sglang/multimodal_gen/tools/build_minimax_h3_adaln_cache.py`
re-implements the time embedder and hard-codes the model dimensions.

**Weights** (`coderef/sglang/python/sglang/multimodal_gen/runtime/loader/minimax_h3_weights.py`; `:648-671,712-759`):
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
(`:168-173,232-267`). The static kwargs handed to the model include the
position ids (cast to fp32 on the host), both update masks, the rank-local
tags, `skip_mask_out_condition=True`, the refined prompt embeddings, the
position infos, and the packed-sequence params (`:193-230`).

**Update.** The scheduler class `MiniMaxH3EulerAncestralEta0SchedulerAdapter`
exists as a validated reference but is not wired; the loop uses a fused
in-place function pinned bit-equal to it (`:33-50`; `coderef/sglang/python/sglang/multimodal_gen/test/unit/test_minimax_h3_denoise_loop.py:122-149`):
```
denoised = state + sigma * v
state    = (sigma_next / sigma) * state + (1 - sigma_next / sigma) * denoised
```
which is plain Euler on the rectified-flow line. **No noise is injected at
any step**; the class name's "ancestral" carries eta = 0. Video and audio
target rows update separately with their own sigma ratios; conditioning rows
are never touched after priming. There is no early exit; every `n-1` step
runs. One positive branch, no guidance arithmetic (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py:56`;
`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py:70`).

**Cache-DiT** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py:417-571`; `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py:58-64`):
mounted only for `quality="high"` with `(warmup 4, residual threshold 0.04,
max consecutive 1)`, or manually through `SGLANG_CACHE_DIT_*` when `quality`
was not explicit. Before mounting, every block's in-place residual is
disabled, because Cache-DiT snapshots the block input and an in-place rewrite
makes every residual read as zero, silently. Off under graph capture and
during warmup. The audited numbers quoted in the constant's comment are
SSIM 0.931 / PSNR 28.16 dB against lossless (reported, not verified here).

Text refinement (the two-block refiner on the projected prompt) runs once per
request before the loop, with `cu = [0, text_len, text_len]` and no
positions; running it at a bucketed length is "not bitwise equivalent"
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py:338-376`; `coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:2186-2187`).

---

## 9. Decode and output

**Video** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py:324-393`): reverse-normalise `z*std + mean`,
decode under **fp16 autocast** by contract (`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py:62`)
although the VAE is fp32-resident because it also encodes keyframes
(`:58-61`); the 36 decoder blocks' weights are cast to fp16 persistently, the
embedder, output projection and norms stay fp32 (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/vae_vit.py:231-262,273-276,348-351`). Decode is tiled at 256 px
with 64 px overlap and, across decode ranks, whole tiles are dealt round-robin
and gathered; `spatial`, `spatial_shard` and `patch` modes are refused as
having "failed the released quality contract" (`coderef/sglang/python/sglang/multimodal_gen/configs/models/vaes/minimax_h3_video.py:36-57`).
Temporal decode takes chunks of 7 tokens (5 plus overlap 2) to 28 frames,
drops 3, blends the overlap, and streams into a preallocated output. Pixels
are inverse-ImageNet-normalised and clamped to `[0, 1]`, then cropped to
`latent_h*16 x latent_w*16`.

**Audio** (`:274-322`): reverse-normalise `[2, 32, T]`, decode in fp32 on
world rank 0, broadcast; output `[1, 2, 800*T]` at 32 kHz.

**File** (`coderef/sglang/python/sglang/multimodal_gen/runtime/entrypoints/utils.py:489-655`): frames `(x*255)` to
uint8 `[T, H, W, 3]` streamed in 128 MiB chunks through a CUDA-registered
memfd into
`ffmpeg -f rawvideo -pix_fmt rgb24 -r 24 ... -vcodec libx264 -pix_fmt yuv420p -crf 25 -acodec aac`;
audio through a temporary WAV after `clamp(-1, 1)`. No trimming of audio to
video: both lengths derive from the same aligned duration, so drift is bounded
by the 40 Hz rounding. The written MP4 is then re-probed and the request
**fails after the file exists** unless it has exactly one h264 yuv420p stream
at 24 fps with the resolved size and frame count, one AAC stream at 32 kHz
stereo, and audio-video duration drift within 0.25 s
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/video_adapter.py:357-552`; tolerance `coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py:71-73`).

---

## 10. The runtime around the model

**Components** (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines/minimax_h3_pipeline.py:39-50`):
processor, tokenizer, text encoder (bf16), transformer (bf16 with the fp32
island), video VAE (fp32 resident, fp16 decode), audio VAE (fp32). No
scheduler component. Weight loads are ordered by descending size, so the DiT
(the deployment config prices it at 61.73 GB) loads before the 46 GB encoder.
Under `--performance-mode auto`, H3 opts into DiT layerwise offload
(`dit_layerwise_offload_modes=("auto","memory")`) on any card with less than
120 GiB free, and everything is kept resident above that
(`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py:85-98`; `coderef/sglang/python/sglang/multimodal_gen/runtime/server_args/auto_tune.py:548-563`).
Layerwise-offloadable units: the DiT's refiner blocks and blocks; the video
VAE's decoder transformer blocks (the CNN encoder stays resident); the audio
VAE's encoder and decoder blocks; the text encoder's text layers plus the
vision blocks.

**Parallelism.** TP must divide 56 heads; Ulysses must divide the TP-local
heads and, with ring, 64; the packed `seq_len` must divide by
`ulysses * ring` (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:1788-1818,2400-2427`).
Ring is the outer contiguous row split and Ulysses the inner. CFG parallel is
refused. Encoder folding across the replica engages when hidden >= 4096 (5120
qualifies) on one node with peer access; cross-node runs need
`--encoder-parallel replicate`. On one GPU there is no collective anywhere
(INFERENCE from the same code with `sp_ws = 1`).

**Batching.** H3 is `TI2V`, which does not support cross-request dynamic
batching, so the scheduler pops one request at a time; only a single
request's multi-output fan-out reaches the grouped-stage path, where text
encoding is shared by fingerprint (`coderef/sglang/python/sglang/multimodal_gen/runtime/managers/scheduler.py:968-1000`;
`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/base.py:411-416`). Warmup is off by default;
`--warmup-resolutions` turns it on but H3 substitutes its own `768 / 16:9 / 5 s`
target regardless of the values given (`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py:107-156`).

**Quality levels** (`coderef/sglang/python/sglang/multimodal_gen/configs/sample/sampling_params.py:54-57,132-146`):
`lossless` (default, bit-exact against CI goldens, no caching) and `high`.
`high` is a fail-closed exact match on two things: the workload
`{t2va, 1344x768, 24 fps, 124 frames, 50 steps, shift 12/3}`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py:22-31,160-213`) and the deployment, which
`validate_quality_deployment` (`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py:104-195`)
pins to `num_gpus 4`, `sp 4`, `ulysses 4`, `ring 1`, `tp 1`, `fa` attention,
`performance_mode speed`, no compile, no graphs, no quantisation of DiT or
encoder, no FSDP, no DiT layerwise offload, variant `fl2va`, and a device
whose name contains `H200` at compute capability 9.0. Everything the gate
requires off is therefore something sglang ships but does not stand behind
for quality.

**Fast paths, as shipped for H3.** torch.compile: available, off in every
mode. Breakable CUDA graphs: allowlisted, opt-in, attention stays eager, the
cookbook records a B200 ref2va run "without a measured speedup". Cache-DiT:
only through `high` or the env knobs. Attention: `fa` default; `sage_attn`,
`sol_attn`, `laser_attn`, `torch_sdpa` and others admissible because they
implement `forward_varlen`; flashinfer is not in the backend enum. Online
quantisation options include `fp8` and `kitchen_int8`; pre-quantised DiTs via
`--transformer-weights-path` cover GGUF, Comfy `fp8_scaled` and
`int8_convrot`, MXFP8, ConvRot W4A8/W4A4 and NVFP4; the cookbook lists
FP8, ConvRot, NVFP4-AWQ, Quanto int8 and GGUF overlays for the text encoder.
FSDP is a capacity option. Frame interpolation, upscaling and TeaCache are
rejected at validation.

**Their ComfyUI app** (`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/`): a custom-node pack
with an integrated mode for FLUX, Qwen-Image and Z-Image, and a **server
mode only** node for H3, `SGLDiffusionGenerateH3` (`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/nodes.py:579-792`), which
posts to `/v1/videos` on a running `sglang serve` and polls. The README's
rule is that models needing conditioning ComfyUI cannot supply (audio,
references, task routing) go server-mode. Its inputs mirror the request
schema: task, keyframes as frame indices 0 / -1, reference image, video,
audio, seed, steps 50, short edge 768, aspect from four choices, duration
4..15 s, both shifts. It forwards `negative_prompt` if set, which the server
side fixes to `None` (INFERENCE: ignored).

---

## 11. Insights

Numbered so they can be cited. Each is SOURCE unless marked.

1. **Attention is fully joint and unmasked.** Text, vision tokens in the text
   segment, keyframes, references, audio and video rows all attend to each
   other in one varlen segment; only padding is fenced off. Every "rows do
   not interact" intuition is wrong here.
2. **Padding rows are real rows.** Zero-embedded, modulated as video at the
   video timestep, attending among themselves, and simply not selected at the
   output. The 64-row alignment therefore changes the sequence the model
   sees, not just its shape.
3. **The scheduler is deterministic Euler.** eta = 0 throughout; nothing
   injects noise after the initial draw. "Euler ancestral" in the class name
   is the whole of the ancestral part.
4. **Four RNG facts share one number.** Initial video noise and initial audio
   noise come from two CPU generators seeded with the same request seed
   (INFERENCE: their first values coincide in draw order); the video VAE
   encode of every keyframe and reference video is a **posterior sample at
   seed 42** regardless of the request seed, and is rounded to fp16 before
   normalisation; the audio VAE encode is the posterior **mean** in fp32 with
   cuDNN disabled. Two modalities, two determinism strategies.
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
18. **AdaLN exactness depends on GEMM shape.** The cache must be built at the
    plan's own row count and TP width; a cached plan is matched by exact fp32
    equality; a novel schedule raises.
19. **The qkv row order is a load-time hazard**: official files interleave per
    head, the runtime concatenates, and per-row scales must follow or the model
    "renders noise". The 21,504-row gate is the only thing separating
    row-indexed scales from swizzled ones.
20. **`num_inference_steps` counts sigma grid points**, so a 4-step turbo is
    requested as 5 and an 8-step as 9.
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

---

## 12. Coverage

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
INFERENCE that the processor code is equivalent). Not read: the LoRA
machinery beyond adapter preparation, the Cache-DiT adapter internals, the
disaggregation mixin, the host-memory budget manager, MPS-only branches.
