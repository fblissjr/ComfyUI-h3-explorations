# MiniMax-H3-Image inference, stage by stage, across every implementation

last updated: 2026-11-17 (off-grid Qwen copy now snapped with 8-bit bicubic via `reference_encode.py::_snap_h3_image_qwen_grid`; initial comparison and record [`../sweeps/2026-11-17_h3_image.md`](../sweeps/2026-11-17_h3_image.md))

How each implementation runs MiniMax-H3-Image, text to image and edit, from loading to the decoded picture.

**What is compared:**
- **The ranked engines:** sglang (rank 1), vllm-omni (rank 2) and diffusers (rank 3).
- **ComfyUI core** (trusted least as evidence).
- **This pack's shipped graphs:** `workflows/image/h3_image_t2i_api.json` and `h3_image_edit_api.json`, written by
  `workflows/build_workflows.py::build_h3_image_api`.
- **The periphery:** DiffSynth-Studio and LightX2V.

**Narrower pages own single questions:**
- [`h3_image_sampling_step.md`](h3_image_sampling_step.md): the Euler step, the sigmas and the draw order.
- [`qwen3vl_encoder.md`](qwen3vl_encoder.md): the text encoder.
- [`../../image/h3_image.md`](../../image/h3_image.md), "Edit references, point by point": the reference table, with
  its reasons.
- [`../../research/image/2026-11-14_h3_image_vae_inspection.md`](../../research/image/2026-11-14_h3_image_vae_inspection.md):
  the decoder and tile stitching.

All of it is a source read; nothing here was rendered. The record says which lines were checked by hand and which
are an agent's reading.

## The short answer

- **Text to image gives the text encoder no vision tokens in any implementation.** This pack's text-to-image graph
  sends the same tokens the engines do ([`qwen3vl_encoder.md`](qwen3vl_encoder.md)).
- **Alignment sits on what the model is shown and how it steps.** The token sequence, its tags, the packed layout,
  the positions, the reference level, the sigma grid and the output heads are the same everywhere.
- **Differences sit at four edges:**
  - **The request:** canvas size, where the size comes from, prompt edges, empty prompts.
  - **The random draws:** their order, the device, and the anchor noise stream.
  - **Precision and files:** VAE encode dtype, DiT quantization, attention kernel, decoder.
  - **One numerical form:** the Euler step.
- **Core is the outlier on reference preparation and the edit canvas.**
- **This pack is the outlier on size and files, by the owner's decisions.** It now matches the ranked engines on
  reference preparation.
- **The periphery's outliers are its own:** DiffSynth on precision; LightX2V and diffusers on draw order.

## What this pack's graphs run

| stage | text to image | edit |
|---|---|---|
| files | `H3_IMAGE_UNET` (Comfy-Org's pruned int8 DiT) behind `ModelAttentionBackend` (comfy-kitchen INT8 attention), `H3_IMAGE_ENCODER` (bf16), `IMAGE_GRAPH_VAE` (P-H3-Image's VAE, int8 decoder) | the same |
| size | `MiniMaxH3ImageSize`: generates at `H3_IMAGE_MEGAPIXELS` at the chosen shape, delivers `H3_IMAGE_DELIVER_MEGAPIXELS` | the same, at the picture's shape (`H3_IMAGE_ASPECT_OF_PICTURE`) |
| conditioning | core's `MiniMaxH3ImageConditioning` with no pictures | `MiniMaxH3VAEPrecision(encoder="fp32")`, then `MiniMaxH3EncodeImageReferences` (`canvas="explicit"`, `H3_IMAGE_REFERENCES`), then `MiniMaxH3PromptOnImageReferences` reading the seed from `RandomNoise` |
| sampling | `BasicGuider`, `SamplerCustomAdvanced` with `KSamplerSelect(H3_IMAGE_SAMPLER)`, `ManualSigmas(H3_IMAGE_MANUAL_SIGMAS)`, `RandomNoise` | the same |
| decode | core's `VAEDecode`, then `MiniMaxH3ImageDownscale` (8-bit by truncation, PIL Lanczos), then `PreviewImage` | the same |

## Where every implementation agrees

Read in sglang, vllm-omni, diffusers, DiffSynth, LightX2V, core and the pack, except where a line says otherwise.

1. **The encoder.** Qwen3-VL-32B in bf16. The conditioning is the state after layer 50, before the final norm.
2. **The presentation.**
   - Text to image is the prompt's tokens alone.
   - An edit is `Picture N: ` and that picture's vision block for each picture in order, then the prompt.
   - No chat template, no special tokens.
3. **The token tags and the text timestep.**
   - A vision block, start and end markers included, is tagged as image for the DiT's AdaLN; labels and the prompt
     are tagged as text.
   - Text rows, the vision rows among them, carry the current step's timestep. This is core's `seg_t` read by hand;
     the engines' are the agents' reading.
4. **The bounds of the two copies of a picture.**
   - The text encoder's copy has its short edge put on one size: sglang's `edit_reference_qwen_*_short_edge`,
     vllm-omni's `MINIMAX_H3_IMAGE_QWEN_SHORT_EDGE`, core's `IMAGE_REF_SHORT_EDGE`, the pack's
     `H3_IMAGE_QWEN_SHORT_EDGE`.
   - The VAE's copy has its short edge kept inside a range: `edit_reference_vae_*_short_edge`,
     `MINIMAX_H3_IMAGE_VAE_SHORT_EDGE`, `IMAGE_REF_VAE_MAX_SHORT_EDGE`, the pack's `H3_IMAGE_VAE_*_SHORT_EDGE`.
   - Outside its bounds, a copy is resized with Lanczos, each side on the 32 px grid.
5. **Normalization differs by consumer:**
   - the VAE's copy of a picture uses ImageNet mean and standard deviation (core's
     `comfy/ldm/minimax/vae.py::IMAGENET_MEAN`);
   - the text encoder's copy uses mean and standard deviation `0.5`, mapping pixels to `[-1, 1]`
     (`vendor_config/preprocessor_config.json`, core's `comfy/text_encoders/qwen3vl.py::Qwen3VL.preprocess_embed`);
   - latents use the release's latent statistics.

   *Corrected 2026-11-17: this line first gave ImageNet statistics for all pixels.*
6. **A reference's latent is a posterior sample, never the mean.**
7. **The packed sequence and its positions.**
   - The order is text, then each reference, then the target.
   - Text sits at time positions `0..L-1`. Reference `k` sits at `L + k`, and the target comes after the last
     reference.
   - Each picture gets its own area-normalized spatial grid.
8. **References are held at one level for every step,** mixed as `level * clean + (1 - level) * noise`, and their
   outputs are discarded. The level is core's `VISUAL_COND_TIMESTEP`, sglang's `MINIMAX_H3_IMGVID_COND_TIMESTEP`, and
   diffusers' `reference_noise_aug`.
9. **The schedule.**
   - The release's six sigmas, bit for bit, with no shift and no guidance (`h3_image_sampling_step.md`).
   - One output head per step, run in float32 after the final norm and AdaLN (core's `FinalLayer._modulate`).
     DiffSynth's examples are the exception: they load the heads, like everything else, in bf16.
10. **The target noise** is a float32 normal draw at the canvas's latent shape.
11. **The VAE decodes in tiles** at the release's tile size and minimum overlap. Where the encode was read
    (vllm-omni, diffusers, DiffSynth, LightX2V, core), it is tiled the same way.
12. **Output pixels are cut to 8 bits by truncation.** When the canvas is larger than the delivered size, the
    8-bit picture is resized with PIL Lanczos.
    - The pack's `MiniMaxH3ImageDownscale` does the same.
    - `PreviewImage` gets back those exact bytes (the check is in the record).
    - diffusers calls the truncation the released pipeline's.

## Where they differ

The pack's column is what the shipped graphs run. "Verbatim" means used as given.

| aspect | sglang | vllm-omni | diffusers | core | this pack | DiffSynth | LightX2V | the outlier |
|---|---|---|---|---|---|---|---|---|
| **prompt edges** | verbatim | verbatim | verbatim | verbatim | edit stripped (`h3_rules.py::normalize_prompt`); text to image verbatim (core's node) | verbatim | verbatim | the pack, by its own rule |
| **empty prompt** | refused | refused | refused (a blank one passes) | one pad token, runs | text to image runs (core's node); edit refused | encoded as `" "` | refused | core, DiffSynth |
| **text segment cut** | at 4096 tokens each | `_TEXT_SEGMENT_MAX_TOKENS` | `max_sequence_length` | none | none (core's tokenizer) | `MINIMAX_H3_IMAGE_MAX_SEQUENCE_LENGTH` | `max_sequence_length` | core; no prompt here comes near it |
| **Qwen copy whose short edge is already the bound, long edge off the grid** | untouched to the HF processor, which resizes with its default, bicubic | the same | the same | Lanczos to the grid first | snapped with 8-bit bicubic (`reference_encode.py::_snap_h3_image_qwen_grid`) | the same as sglang | the same as sglang | core |
| **VAE copy inside its bounds** | cropped top left to the 16 px cell, latent cut to an even grid | the same | the same | resampled to the 32 px grid | cropped (`in_bounds_align="crop_32"`) | the same as sglang | the same as sglang | core |
| **reference posterior sample** | request's generator, rounded to fp16 | the same | the same | fixed `IMAGE_ENCODE_SEED`, not rounded | render's seed, rounded to fp16 (`reference_latent="release"`) | request's generator, rounded to fp16 (noise drawn in the VAE's dtype) | the same as sglang | core |
| **VAE encoder precision** | fp32, enforced | fp32 | fp32 (pinned) | as loaded (cast to the VAE dtype on load) | fp32 (`MiniMaxH3VAEPrecision`) | bf16 in its examples | fp32 (`_keep_in_fp32_modules`) | core, DiffSynth |
| **anchor noise per reference** | one draw per reference from the request's generator | the same | the same | a stream restarted at the sampler's seed for every reference (`_cond_video_rows`) | core's (model code) | the same as sglang | the same as sglang | core and the pack |
| **draw order** (T target, A anchors, P posteriors) | T, A, P | T, A, P | P, T, A | separate seeds | T, A, P (T and A only advance the stream) | T, then P and A per reference | P, T, A (its comment: Modular Diffusers') | split: the ranked two against diffusers and LightX2V |
| **generator device** | platform device unless `generator_device` is set; the release's examples set `cpu` | the same | the caller's; its docs pass a CPU one | CPU | CPU | CPU | CPU | sglang and vllm-omni by default |
| **Euler step** | velocity, float32 | velocity, float32 | velocity (`euler_update_form`) | stock `euler` (denoised form) | core's | velocity | velocity, float32 | core and the pack; tested 2026-11-15, kept |
| **text-to-image canvas** | the request (no pipeline default; its OpenAI protocol says `1024x1024`), raised to `min_generation_area`, resized back | default `MINIMAX_H3_IMAGE_DEFAULT_SIZE`, the same floor and resize back | default 2048x2048, the same floor and resize back | `IMAGE_SIZE` square, no floor | `H3_IMAGE_MEGAPIXELS` generated, shrunk to `H3_IMAGE_DELIVER_MEGAPIXELS` | as diffusers | as diffusers | the pack, by decision (2026-11-14) |
| **edit canvas** | the request; never a picture's size | the same | the same | image_1's VAE-copy size, no floor | the picture's shape at `H3_IMAGE_MEGAPIXELS`, shrunk to the delivered size | the request (default 2048x2048) | the request | core and the pack (the pack by decision) |
| **DiT precision and attention** | bf16 with fp32 islands, enforced; dense attention | bf16 with fp32 islands | bf16 with fp32 islands; SiLU in fp32 before the AdaLN cast | the checkpoint's; fp32 heads | Comfy-Org's pruned int8 file with comfy-kitchen INT8 attention (owner, 2026-11-11) | every module at bf16, the fp32 islands too | bf16 with fp32 islands; `sage_attn2` in its shipped configs; AdaLN cache | the pack, by decision; DiffSynth; LightX2V |
| **decoder** | the release's 2D ViT, bf16 weights under bf16 autocast | the same, with its 2D RoPE bases kept fp32 | the release's, at the load dtype; a tile blends against its neighbours' raw decodes | the release's, at the load dtype; a tile blends against the already blended canvas | P-H3-Image's int8 decoder (`IMAGE_GRAPH_VAE`; owner, 2026-11-12 and 2026-11-14) | the release's, bf16 | the release's, bf16 with fp32 norms | the pack, by decision |
| **reference cap** | none in the edit profile (agent) | `MINIMAX_H3_IMAGE_MAX_REFERENCES`; aspect capped at `MINIMAX_H3_IMAGE_MAX_REFERENCE_ASPECT` | not read | the node's slot count | `H3_IMAGE_MAX_REFS` | not read | 16 | none to speak of |

## The pattern

**The model's contract is copied faithfully everywhere.** Every implementation follows the release's code: the
tokens, tags, layout, positions, level, grid and heads. That holds for core and for the periphery too. The release
fixes all of it, and getting any of it wrong produces a visibly broken picture.

**The differences fall into four kinds, and each kind carries different weight:**

1. **The request edge** is a serving choice, not the model's. It covers the default size, the canvas floor, where
   an edit's size comes from, prompt edges, empty prompts, and caps. The engines agree with each other on the floor
   and on taking the size from the request; core takes it from the picture.
2. **The random draws** matter only for seed-exact reproduction. They change the noise a reference or the target
   starts from, not what the model is shown. No two implementations draw the same way except sglang and vllm-omni.
   This pack follows those two on order, but not on the anchor stream.
3. **Precision and files** decide fine detail. The pack's choices here are the owner's, each with its record
   ([`../../image/h3_image.md`](../../image/h3_image.md)).
4. **The Euler form** is one row, tested ([`h3_image_sampling_step.md`](h3_image_sampling_step.md)).

**The outliers, by implementation:**
- **sglang and vllm-omni** agree on every stage read. They differ only in request defaults and caps.
- **diffusers** agrees with them except on draw order: it draws the posteriors first. LightX2V copies diffusers here.
- **Core** agrees on everything the model is shown. It is the outlier on reference preparation (resample, fixed
  seed, no fp16 rounding, encode precision), on the edit canvas, on empty prompts and on the Euler form.
- **This pack** matches the ranked engines on reference preparation (since 2026-11-13) and draw order. It is the
  outlier by decision on size, the DiT file, attention and the decoder, and by its own rule on prompt edges in
  edits. It inherits two things from core: the anchor stream and the Euler form.
- **DiffSynth** is the outlier on precision (everything bf16), draw order and the empty prompt.
- **LightX2V** is the outlier on draw order, the attention kernel and the AdaLN cache.

## Open

- **The anchor noise stream** is in core's model code (`_cond_video_rows`). Matching the engines needs a runtime
  patch, and nothing shows it is worth one ([`../../image/h3_image.md`](../../image/h3_image.md)). It is also why
  an edit cannot be reproduced seed for seed against any engine. Text to image with a CPU generator starts from the
  same noise everywhere.
- **Prompt edges.** The pack strips edit prompts while its text-to-image graph, on core's node, does not. The
  engines strip neither. Making the two paths agree is a choice, not a fix.
- **An empty text-to-image prompt** runs on core's node, where every engine refuses it.
- **From the encoder page** ([`qwen3vl_encoder.md`](qwen3vl_encoder.md), "Open", narrowed 2026-11-17):
  - Core's encoder runs PyTorch SDPA with an explicit causal mask, and which kernel SDPA then picks is not checked.
  - Core's vision RoPE table is built on the CPU, where sglang rebuilds it on the GPU for this release.
- **Measured since, by a peer session** (`bench/results/2026-11-17_pika_sampler_and_t2i_vision_probes.md`, Probe B):
  grey or black pictures shown only to the text encoder, in text to image, gave a different picture at the same
  seed from the plain prompt. So such pictures are not neutral to this model.
  - The record has sharpness figures and a session's reading of the stills.
  - It is one seed, and the arms differ in layout, so its sharpness figures compare different pictures. They cannot
    by themselves show whether the vision tokens soften a picture.
