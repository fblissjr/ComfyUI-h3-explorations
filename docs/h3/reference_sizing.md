# Sizing a reference: two copies, the knobs, and the vendor's path

The detail under [`references.md`](references.md). Every still is read twice, and each copy has its own size:

| copy | becomes | sized by | default |
|---|---|---|---|
| the video model's | DiT reference rows, attended every step | `size_policy` (`max` with `dit_short_edge` and `allow_upscale`, or `match`) | `max`, a 2048 short edge, upscaling on: vendor parity |
| the text encoder's | vision tokens ahead of the prompt | `qwen_view` (`separate` with `qwen_short_edge`, or `shared`) | `separate` at `h3_rules.REF_QWEN_SHORT_EDGE`: the owner's choice of 2026-10-03, a departure from the vendor |

Read the values from `MiniMaxH3AppendRefImage`'s `define_schema` (`reference_conditioning.py`), not from this page.
Rows go as the square of the scale, so the knobs are the largest per-reference cost lever. Nothing here says the 2048
copy looks better: that is `docs/state/open_experiments.md` #1, unmeasured, and upscaling adds tokens rather than
detail. `bench/refview2_arms.json` is the reference-view ablation that would move the defaults (six arms on five
scene graphs, `h3_config.REFVIEW2_SCENES`); it was rendered and never scored.

## Sizing a reference image: the video model's copy

**sglang's rule**: a fixed 2048 short edge, upscaling allowed, nearest 32 per axis, no area cap
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:125-177`:
`scale = 2048 / min(w, h)`, `"allow_upscale": True`, and a docstring saying reference images have no area-cap branch).
sglang, diffusers and DiffSynth all feed that one prepared still to both the VAE and Qwen
([`docs/research/2026-08-25_sglang_h3_pipeline.md`](../research/2026-08-25_sglang_h3_pipeline.md), "Reference stills").

**Core never upscales.** Its `ref_image_size` clamps with `min(1.0, ...)` in both modes
(`comfy_extras/nodes_minimax_h3.py:297-301`):

```python
if ref_image_size == "match":
    scale = min(1.0, math.sqrt((width * height) / (w * h)))
else:  # "max"
    scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))   # 2048
```

`max` picks the ceiling (a 2048 short edge instead of the generation's pixel area), not the direction; core's tooltip
says "never upscaled". So under core a reference smaller than 2048 on its short side reaches the DiT under-sized, and
its default `match` is off-vendor the other way too: it sizes to the generation's area (near 1 MP at 1344x768) where
sglang's 16:9 reference lands near 7.5 MP. `ref_image_size` applies to every image reference at once and to nothing
else; video references take `adapt_canvas` from their own aspect.

**This pack's typed path does one resize, per reference, with the canvas in scope**
([`reference_geometry.py::fit_reference_image`](../../reference_geometry.py)):

```python
if size_policy == "match":
    # allow_upscale is never read here
    scale = min(1.0, math.sqrt((canvas_w * canvas_h) / (source_w * source_h)))
else:
    full  = short_edge / min(source_w, source_h)
    scale = full if allow_upscale else min(1.0, full)
```

`size_policy` is a `DynamicCombo`: `dit_short_edge` and `allow_upscale` exist only under `max`, so a contradictory
setting cannot be expressed. One image per append node, so five references are five independent decisions. Prefer
`max` for a vendor-matching arm because `match`'s ceiling moves with the canvas. (`MiniMaxH3ReferenceFit`, the
two-node form of this on core's socket node, was retired in 0.173.0.)

### Worked examples

Five stills at `max`, a 2048 short edge. Rows are `(w/32) x (h/32)` after the round to 32; computed from the formulas,
not measured.

| source | short edge | upscale off | rows | upscale on | rows | factor |
|---|---|---|---|---|---|---|
| 1024x512 | 512 | 1024x512 | 512 | 4096x2048 | 8,192 | 16.00x |
| 960x1280 | 960 | 960x1280 | 1,200 | 2048x2720 | 5,440 | 4.53x |
| 2048x2612 | 2048 | 2048x2624 | 5,248 | 2048x2624 | 5,248 | 1.00x |
| 768x512 | 512 | 768x512 | 384 | 3072x2048 | 6,144 | 16.00x |
| 1920x1080 | 1080 | 1920x1088 | 2,040 | 3648x2048 | 7,296 | 3.58x |

| scenario | reference rows | vs none upscaled |
|---|---|---|
| none upscaled | 9,384 | 1.00x |
| only the two 512-short-edge stills | 22,824 | 2.43x |
| only the two largest | 18,880 | 2.01x |
| all upscaled | 32,320 | 3.44x |

**The smaller the reference, the more upscaling costs it**, because the factor is `2048 / short_edge`, squared: the
two smallest add 13,440 rows, the two largest 9,496. These are the video model's rows only; the text encoder's copy is
priced separately (below), and `bench/preflight_graph.py` or the conditioner's preview prices both.

## The text encoder's copy: `qwen_view`

- **`shared`**: Qwen sees the same tensor the VAE encodes. What every serving implementation does, and the default
  from 2026-09-13 to 2026-10-03.
- **`separate`** with `qwen_short_edge` (pre-filled from `h3_rules.REF_QWEN_SHORT_EDGE`): the encoder is shown the
  source scaled so its shorter side reaches N (nearest 32, one Lanczos resample, either direction), while the VAE keeps
  the `size_policy` copy. The default since 2026-10-03, on the owner's look at three clips of the finish graph at one
  seed, not blind; on the one dialogue scene the mix came out louder (`h3_rules.REF_QWEN_SHORT_EDGE` and
  `docs/state/decisions.md` have the history and what the evidence does not cover).

It breaks no contract: nothing indexes a Qwen token against a latent patch ([`conditioning.md`](conditioning.md),
section 1b). What it changes is the cost split: a 640x480 still at `qwen_short_edge=2048` costs 300 reference rows
and 5,440 Qwen tokens, the `noup_q2048` arm of the ablation. Under `image_policy=release` the release's still bounds are
pre-applied to the Qwen view alone.

**After `qwen_view`, the loaded encoder's own processor bounds apply**, and on the shipped encoder they bind nothing a
reference node produces. `h3_config.MODELS["clip"]` is `ENCODER_INT8`, loaded through core's `CLIPLoader` path, so
core's own `process_qwen2vl_images` bounds apply, and they give the same merged-token counts as the release's at every
aspect and short edge swept (448 / 1,824 / 7,296 at 16:9;
[`../../data/results/2026-08-29_qwen_view_under_snapshot.json`](../../data/results/2026-08-29_qwen_view_under_snapshot.json)).
An encoder swap between those two therefore changes the weights and not the reference view. Read the bounds from what
`MiniMaxH3EncoderLoader` stamps (`h3_encoder_loader.native_encoder_contract`, read out of core), never from prose.
Controlled by `bench/check_reference_runtime.py::qwen_view_is_separate_from_the_vae_view` and
`preflight_prices_the_two_views`.

(History, closed with the AWQ encoder lane on 2026-09-13: the W4 artifact's 200,704-301,056-pixel snapshot crushed
the encoder's copy to about 265 tokens whatever the first stage did, which is why no-upscale was the sound default
while it shipped; an owner decision of 2026-08-25 calibrated the v2 candidate at the 2048 upscale, recorded in
`canonical/active_plan.md`, with the processor-policy benchmark in `2026-08-24_layer50_processor_policy_benchmark.md`.
`docs/state/decisions.md` has the lane.)

## The vendor image path, stage by stage

Read from source 2026-08-21 against `coderef/sglang` at `a41da991c8`, beside `comfy_extras/nodes_minimax_h3.py`,
`comfy/text_encoders/minimax.py` and `comfy/ldm/minimax/model.py`. The two are the same pipeline written twice. The
divergences are the size a still reaches the two towers, the VAE posterior (sampled against mean), the condition-noise
draw, and the encoder's pixel bounds.

| stage | sglang | ComfyUI | same? |
|---|---|---|---|
| admission | validates `minimax_h3.request/v1`, resolves the release partition from `model_index.json._minimax_h3`, refuses a task outside it (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py:60-135`) | no partition concept; any checkpoint takes any graph | **no**, see below |
| geometry | resolved before queueing and frozen (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/prequeue.py:303-315`) | computed inline in the node at execute time | no consequence found |
| sizing | 2048 short edge, upscale included, nearest-32 per axis, ratio 1:4..4:1 enforced, no area cap (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:125-203`) | `min(1.0, ...)` in both modes; default `match` targets the generation area | **no**: the sizing divergence |
| resampler | PIL LANCZOS (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:181-203`) | `common_upscale(..., "lanczos")` on the tensor (`comfy_extras/nodes_minimax_h3.py:64-68`) | same intent |
| orientation | `ImageOps.exif_transpose` on open (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:886-887`) | none; a tensor arrives decoded | not applicable |
| one image, two towers | the identical prepared image feeds Qwen and the visual tokenizer (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:840-846`) | the same `resized` tensor goes to `vae.encode` and into `ref_items` (`comfy_extras/nodes_minimax_h3.py:304-306`) | **yes, until 3.0625:1** (below); on the typed path `qwen_view=separate` parts them on purpose |
| VAE encode | the video posterior **sampled** under a pinned seed 42 (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/keyframe_encoding.py:7-30`, and `MINIMAX_H3_REFERENCE_VIDEO_ENCODE_SEED` at `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:613`) | the posterior **mean**, never sampled (`comfy/ldm/minimax/vae.py:685`) | **no** |
| patchify | the HF `image_processor`, reading the release's `processor/preprocessor_config.json` | reimplemented, geometry right, bounds left on the shared helper's defaults (`comfy/text_encoders/minimax.py:35-66`, `comfy/text_encoders/qwen3vl.py:62-68`) | **algorithm yes, constants no** |
| presentation | `<Picture i>: ` then a vision block, no chat template (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/presentation.py:104`) | the same string, the same place (`comfy/text_encoders/minimax.py:164`) | **yes** |
| packing | ref blocks in request order, target timeline past their spans (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/packed_sequence.py:274-320`) | `_ref_t_span` does the same (`comfy/ldm/minimax/model.py:335-338`) | **yes** |
| condition timestep | 0.999, applied as `max(t_video, aug)` (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py:24`) | 0.999, as `max(t_v, vis_aug)` (`comfy/ldm/minimax/model.py:32`) | **yes** |
| condition noise | the same 0.999 as the mixing weight, `aug * clean + (1 - aug) * noise`, per-condition CPU generator (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/condition_noise.py:93-117`) | the same recipe and constant, a fresh generator per condition (`comfy/ldm/minimax/model.py:499-511`) | **recipe yes, draw no** |

**What follows from the table:**

- **The two VAEs disagree in opposite directions.** sglang samples the **video** posterior at a pinned seed and takes
  the **audio** posterior mean
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/audio_encoding.py:83`).
  ComfyUI takes the mean for both, so it diverges on every reference still, reference video and keyframe and matches
  on reference audio, precision included (`comfy/sd.py` pins the H3 audio VAE to fp32). The latent differs before any
  noise is mixed in. A parity test must separate this from the video VAE's precision, and must be made at the latent:
  a rendered clip cannot A/B it.
- **The condition-noise draw differs; the recipe does not.** sglang and DiffSynth draw in latent space over the full
  conditioned length, slice, then patchify
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/condition_noise.py:94-119`);
  ComfyUI draws in row space after patchify at the condition's own size (`comfy/ldm/minimax/model.py:505-511`). Same
  distribution, a different sample at one seed. At the shipped level the noise carries a thousandth of the weight, so
  this changes which sample you get, not fidelity. *Source read, not measured*; sglang is the vendor's serving path,
  not a specification, and some of its choices follow from hardware this box does not have.
- **The condition noise level is a setting on both sides.** sglang takes it per request
  (`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::MiniMaxH3SamplingParams`,
  `imgvid_cond_noise_aug_for_inference`; default in
  `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/denoising.py::minimax_h3_condition_noise_aug`).
  Core reads `minimax_visual_cond_noise_aug` and `minimax_audio_cond_noise_aug` from the conditioning
  (`comfy/model_base.py::MiniMaxH3`, `extra_conds`; `comfy/model_base.py:2175-2179`) and uses the one value as the
  mixing weight and the timestep label of every visual condition (`comfy/ldm/minimax/model.py::MiniMaxH3Model`,
  `_cond_video_rows` and the `seg_t` table of `_forward`). This pack's one node that sets it is
  `reference_noise.py::MiniMaxH3ReferenceNoise` (since `c7fcae2f`), on the model side, through a wrapper; the
  conditioning key stays unwritten. A lower value mixes more noise into the reference rows and labels them less clean,
  a way to weaken a visual reference without touching the prompt (reasoned from the two recipes, not rendered).
- **ref2va is not a text-to-video model, by its own release metadata.** MiniMax ships two partitioned entry points,
  each with a `_minimax_h3` block in its `model_index.json`: fl2va declares `tasks: ["t2va", "fl2va"]`, ref2va
  `tasks: ["ref2va"]` (confirmed 2026-08-21 from the official weights). sglang refuses a task outside the partition;
  ComfyUI has no such gate, so a plain t2v graph loads ref2va and renders. Read a ref2va loss at plain t2v that way, not
  as a defect.

### The encoder's pixel bounds

The patch geometry matches: `patch_size=16`, `temporal_patch_size=2`, `merge_size=2`, 0.5 mean/std, and core passes 16
explicitly rather than inheriting Qwen2-VL's 14 (`comfy/text_encoders/qwen3vl.py:62-68`). The **pixel bounds** do not:
the release ships still bounds in `processor/preprocessor_config.json` and video bounds in
`processor/video_preprocessor_config.json`; core reads neither and leaves `process_qwen2vl_images` on the shared
helper's defaults, applying the same pair to each two-frame video block. Pixel counts, not edges:

| bound | H3 release | ComfyUI |
|---|---|---|
| image min pixels | 65,536 | 3,136 |
| image max pixels | 16,777,216 | 12,845,056 |
| video min pixels | 4,096 | 3,136 |
| video max pixels | 25,165,824 | 12,845,056 |

- **The floor**: the release enlarges a still under 65,536 pixels to reach it; core's floor is twenty times lower, so
  a thumbnail or a small face crop reaches Qwen under-tokenized. A second way, beside core's `min(1.0, ...)` clamp,
  that a small reference arrives smaller than the release intends. Preflight reports it; nothing enforces it.
- **The ceiling, measured** (`bench/measure_qwen_bounds_bite.py`, calling the real `process_qwen2vl_images` with the
  arguments `comfy/text_encoders/qwen3vl.py:65` passes;
  [`../../data/results/2026-08-21_qwen_bounds_bite.json`](../../data/results/2026-08-21_qwen_bounds_bite.json)). At a
  2048 short edge it bites past `12,845,056 / 2048² = 3.0625:1`, where the release carries the still untouched to 4:1.
  Past that, the VAE gets the full copy and Qwen a smaller one, silently:

| prepared | what Qwen sees | release |
|---|---|---|
| 6144x2048 (3:1) | 6144x2048, untouched | untouched |
| 6272x2048 | 6272x2048, untouched | untouched |
| 6656x2048 (3.25:1) | 6432x1984, shrunk | untouched |
| 7168x2048 (3.5:1) | 6688x1888, shrunk | untouched |
| 8192x2048 (4:1) | 7168x1792, shrunk | untouched |

- **A keyframe is never resized here.** Every legal H3 canvas is a multiple of 32, the helper's rounding factor, and
  inside both implementations' bounds; the same script's control arm goes loud if a canvas is ever resized, which
  would make the inert claim in [`docs/research/2026-08-21_official_weights_metadata.md`](../research/2026-08-21_official_weights_metadata.md)
  wrong.
- **Reference video has a different shape**: the release's video maximum is clip-wide over the sampled frames, core's
  per two-frame block. The boundary is source- and length-dependent ([`comfyui_vendor_gaps.md`](comfyui_vendor_gaps.md),
  gap 2).
- Being seen in patches is not a mechanism for anything: patchified images are what the model was trained on. What can
  cost you is arriving at the tokenizer a different size than the release would have chosen.

## Video and audio

**Audio: capped at this pack's typed boundary.** Core encodes the whole waveform where the vendor cuts it to the
generated duration, at 80 rows per second of excess. One compiler-derived cap replaced the explicit
`TrimAudioDuration` nodes of 2026-08-22 and cannot drift from `length`. The same boundary duplicates mono to stereo
(core does too) and refuses more than two channels, where core silently keeps the first two. Preflight reports native
socket graphs separately, including "unreadable" when ffprobe cannot answer.

**Video: release parity is opt-in.** `MiniMaxH3ReferenceConditioning.video_policy` takes `comfy` (default: core's
no-upscale VAE view and its per-pair Qwen processor) or `release`, which owns two inseparable stages: it upscales the
full-rate view to `adapt_canvas` for the video VAE, and sends the raw 2 fps samples through the release's
`Qwen3VLVideoProcessor`, whose clip-wide budget may shrink Qwen's view while the VAE keeps the full canvas. Upscale
alone would overshoot the release's Qwen rows, so it is not an option. `image_policy` takes the same two values for
stills. Generated graphs use `comfy` on every graph but the `release` probe arm (walk `h3_config.graph_paths`). Core's
own node remains no-upscale and duration-unaware; the 39-frame live smoke shows the local policy runs, not that
upscaling improves a clip.
