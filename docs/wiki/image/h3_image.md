# MiniMax-H3-Image: MiniMax's own still-image model

Last updated: 2026-11-13 (the graphs load the bf16 text encoder; "Edit references, point by point" added under "What the vendor's servers do"); 2026-11-12.

MiniMax-H3-Image makes a picture from text, or edits one or more pictures, on
one checkpoint, in six steps with no guidance. It is a different model from
the two image paths this repo already had: PrunaAI's P-H3-Image
([`wiki/p_h3_image.md`](p_h3_image.md)) and a single frame of H3 video on
Pika ([`h3_image_editing.md`](h3_image_editing.md)). This page is what the
release's files are, what we checked about them, and how the workflows here run
it. The release's model card is the authority on the model; it is read from the
download, not kept in this repo.

## What it is, from the model card and the files

- **The DiT is the H3 video DiT without its audio layers**, the same depth and
  width. Its tensor names are the fl2va release's minus the audio projection
  and the audio output head (the two releases' shard indexes, compared
  2026-11-11).
- **The text encoder is H3's, unchanged.** The release's shards carry the same
  hub hashes as the H3 release's, and its tokenizer and processor files are
  byte for byte the same (checked 2026-11-11 by comparing the two downloads).
  The graphs load it unquantized since 2026-11-13 (the owner's decision):
  `H3_IMAGE_ENCODER`, the release's bf16 weights cut after the layer H3 reads,
  where the video graphs load the int8 file. What is on record about the int8
  file is H3 video's ([`../decisions/2026-09-25_to_2026-09-29.md`](../decisions/2026-09-25_to_2026-09-29.md)):
  its error sits mostly on picture tokens. Nothing has measured it on this model.
- **The VAE is its own file.** Its encoder weights are the H3 video VAE's
  (equal to fp16 rounding on every encoder tensor compared, 2026-11-11); its
  decoder is a smaller single-picture decoder with different shapes. The video
  VAE cannot stand in for it.
- **It samples on a fixed grid of six steps** with one output head per step,
  and the grid is part of the checkpoint (`workflows/h3_config.py::H3_IMAGE_MANUAL_SIGMAS`).
  There is no step count, guidance scale or negative prompt to set.
- **It is built for large pictures.** The card gives a working range in
  megapixels and a best size (`H3_IMAGE_MEGAPIXELS_RANGE`, `H3_IMAGE_MEGAPIXELS`
  in `h3_config.py`), with both sides a multiple of 32.

## Is it fl2va or ref2va underneath?

Neither, as released. `bench/measure_h3_image_weights.py` compared every block
of the three DiTs, and the record is
[`bench/results/2026-11-11_h3_image_weights.md`](../../../bench/results/2026-11-11_h3_image_weights.md):

- The image weights are far from both video checkpoints, many times the
  distance the two video checkpoints are from each other, and about equally
  far from each.
- Along the one direction that separates fl2va from ref2va, the image weights
  sit between the two through most of the depth, nearer fl2va, and at fl2va's
  end in the last blocks. Their own move away from fl2va is nearly at right
  angles to that direction.

That is what a model trained on from the family's shared starting point looks
like, not a fine-tune of either released video checkpoint. In how it is *run*
it is ref2va's design: reference pictures ride in the sequence ahead of the
picture being made, each named `<Picture N>`, held just short of clean, and
text to image is the same thing with no pictures. Which checkpoint MiniMax
actually started from is not something the weights can say.

## The files on this box

| file | what it is | made by |
|---|---|---|
| `H3_IMAGE_UNET` | the DiT, int8 linears, pruned AdaLN; the graphs' default | Comfy-Org |
| `H3_IMAGE_UNET_BF16` | the DiT as released, nothing quantized or pruned. Not kept on disk: written, rendered and removed on 2026-11-11 | `bench/convert_h3_image_dit.py` writes it again from the release |
| `H3_IMAGE_ENCODER` | the H3 text encoder in bf16, nothing quantized, cut after the layer H3 reads; the graphs' default since 2026-11-13. Larger than the card, so core places part of it off the card | `bench/convert_h3_bf16_encoder.py`, from the release |
| `PH3_IMAGE_VAE` | H3 encoder (stored fp32 in file; cast to fp16 by `comfy/sd.py` on load, `max_abs_diff = 2.98e-8`, then promoted in place by `MiniMaxH3VAEPrecision`) + 36-layer 32-head (`dim=2048`) still-image decoder; the graphs' default since 2026-11-12 | PrunaAI (`P-H3-Image`) |
| `H3_IMAGE_VAE` | the release's VAE in fp16 (16-head `dim=1024` decoder) | Comfy-Org, a plain cast of the release's |
| `H3_IMAGE_VAE_FP32` | the release's VAE as released | the release |

The names are constants in `workflows/h3_config.py`. The files on disk are
links from ComfyUI's model folders into the model store.

## Is Comfy-Org's int8 file a faithful copy? Yes

The same record holds the evidence:

- **Its int8 weights are the public recipe run on the release.** Quantizing the
  release's weights with comfy-kitchen's own quantizer, in the format the file
  declares, gives the file's int8 codes exactly, in every linear of every
  block. Converting it again ourselves the same way would produce the same
  bytes.
- **Everything it does not quantize is the release's bytes**, the fused
  attention rows reordered the way core needs them and the six output heads
  stacked the way core reads them.
- **The one approximation besides int8 is the pruned AdaLN table**, and at the
  timesteps this model is ever run at its error is well below the int8 error.
- **It also carries a request for INT8 attention** in every block: on a graph
  with no attention node, core runs comfy-kitchen's INT8 attention on this
  file. That is the pack's own dense kernel (`h3_config.DENSE_BACKEND_NODE`),
  and the default graphs name it. The owner approved the file's request on
  2026-11-11 (`bench/results/model_contents_baseline.json`), and
  `bench/check_model_contents.py` now fingerprints this file and the VAE, so
  a file replaced under the same name shows.

So there was nothing to gain from redoing the same conversion. What we did
write is the opposite end: the release whole in bf16, as the reference the int8
file can be judged against. The record has the comparison renders. On the
release's two worked cases every arm gives the same picture with different
fine detail, and a session reading crops at full size found no defect in one
arm that the others lack. That does not rank them, and for text to image a
pixel comparison cannot: MiniMax's own servers report a text-to-image picture
moving at a fixed seed with any small numerical change, while an edit holds
("What the vendor's servers do" below). On the edit the arms were compared by
numbers too, and the record has them.

## The workflows

In [`workflows/image/`](../../../workflows/image/), written by
`workflows/build_workflows.py::build_h3_image_api`:

| graph | what it does |
|---|---|
| `h3_image_t2i_api.json` | text to image |
| `h3_image_edit_api.json` | an edit of a picture, at the picture's shape |
| `h3_image_t2i_heylook_api.json` | short text -> `MiniMax H3 Image Prompt Expander (heylook)` -> text to image |
| `h3_image_edit_heylook_api.json` | picture + short edit instruction -> `MiniMax H3 Image Prompt Expander (heylook)` -> edited image |

Four probe graphs rendered the same two cases on the bf16 file, with the same
INT8 attention and with core's exact attention, so the int8 linears and INT8
attention could each be judged alone. They and the bf16 file were removed on
2026-11-11 at the owner's word once the comparison was recorded; the builder
still takes `unet=H3_IMAGE_UNET_BF16` and `attention=None` to make them again.

The chain follows the vendor's diffusers pipeline
(`coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3_image/`) on
core's nodes. Core's conditioning node was read against that pipeline and
agrees with it on what matters: the sequence layout, the picture labels, the
level references are held at, the reference sizes for the VAE and for the text
encoder, and a sampled rather than mean VAE encoding of each reference. Three
small things differ and were left alone: `H3ImageVAE.encode` draws each
reference's VAE posterior sample from fixed `IMAGE_ENCODE_SEED = 42` where the
pipeline uses the request's seed (`MiniMaxH3._cond_video_rows` still seeds the
`0.001` DiT reference noise from the sampler seed), core rounds an in-bounds
reference to a multiple of 32 where the pipeline crops it, and core skips the
pipeline's rounding of reference latents to fp16. The chain differs from
Comfy-Org's shipped templates in four places:

1. **The six-step grid is handed to the sampler as written.** The templates
   ask `BasicScheduler(simple, 6)` over `MiniMaxH3ImageModelSampling`'s
   1,000-entry discrete table; because `6` does not divide `1,000`, `simple`
   drifts off `pdd_config.json`'s knots by up to `8.69e-4` (`sgm_uniform` by
   `2.09e-3`, `beta` by `0.104`), whereas `ManualSigmas` (`H3_IMAGE_MANUAL_SIGMAS`)
   lands with `0.0` float32 error. The heads are one per step of this grid under
   plain Euler; no other sampler, step count or grid is valid.
2. **The size is megapixels and a shape** (`MiniMax H3 Image Size`,
   [`h3_image_nodes.py`](../../../h3_image_nodes.py)), at the card's best size (`16.0`
   megapixels) by default. Requests below `4.0` megapixels (`enforce_4mp_floor=True`
   default) follow MiniMax's `sglang` (`image_canvas.py`) and `vllm-omni`
   (`pipeline_minimax_h3_image.py`) servers: the DiT samples on the vendor's
   `4.0`-megapixel canvas floor (`width`, `height`), and `MiniMax H3 Image Downscale`
   (`MiniMaxH3ImageDownscale`, wired between `VAEDecode` and `SaveImage`) resizes
   the decoded pixels to `(target_width, target_height)` with 8-bit PIL Lanczos
   (passing `>= 4.0`-megapixel outputs through untouched). The templates start at
   the bottom of the working range without a floor guard.
3. **A picture being edited reaches the model untouched.** The edit template
   resamples it first. The vendor's pipeline keeps a reference's own pixels
   whenever its short side is within bounds, and core's node already does the
   bounding. Note that core's `MiniMaxH3ImageConditioning` declares
   `io.Autogrow.TemplatePrefix(prefix="image_")` (`0`-indexed `images.image_0..N-1`
   in API JSON; `1`-indexed `images.image_1` works over direct `/prompt` HTTP
   posts because `_image_refs` sorts by trailing integer, but ComfyUI's browser
   frontend drops `images.image_1` when opening the API JSON and silently runs
   text-to-image with zero references; fixed 2026-11-12).
4. **The default VAE (`PH3_IMAGE_VAE`, `p-h3-image_vae_bf16.safetensors`) pairs
   the identical H3 encoder with P-H3-Image's larger 36-layer, 32-head
   (`dim=2048`) still-image decoder** (2026-11-12). All 118 encoder +
   `quant_conv` tensors in `PH3_IMAGE_VAE` are bit-for-bit identical (`max_diff = 0`)
   to `H3_IMAGE_VAE_FP32`, with identical `LATENTS_MEAN`/`LATENTS_STD` and the
   same `H3ImageVAE.encode` posterior sample (`IMAGE_ENCODE_SEED = 42`), so
   reference latents and the 6-step DiT trajectory are unchanged. On decode,
   the release's lightweight 24-layer, 16-head (`dim=1024`) `MiniMaxH3ImageVAE`
   exhibits orange/green shadow stipple and fine-lettering breakup (`29.91 dB`
   PSNR on `reference1.png` at `1024x1024`, whether fp16 or fp32: `69.27 dB`
   between the two), whereas `PH3ImageVAE` decodes the exact same latent at
   `31.47 dB` (`+1.56 dB`) and cleanly removes the dark-shadow grain on live
   `MiniMax-H3-Image` DiT latents (`build_h3_image_api(..., vae=H3_IMAGE_VAE)`
   keeps the release decoder available).

Attention is the same as the templates': comfy-kitchen's INT8 attention, named
in the graph. The vendor's pipeline computes exact attention, which is what
the bf16 reference arm ran.

The two default graphs carry the release's own worked cases at the release's
seeds. Rendered here on 2026-11-11, each gives the composition of the
corresponding "local" picture on the model card, as a session read the stills:
the farm table, and the two podcast hosts with the headphones gone and fists
touching over the sign.

Both graphs, and the four bf16 probes while they existed, were run as they
were on disk on 2026-11-11, after the server was restarted to load
`MiniMax H3 Image Size`.

## What the vendor's servers do

MiniMax's release names sglang and vllm-omni as the servers to run it on, and
the sglang code is the closest thing to the vendor's own reference. Both were
read on 2026-11-11 (the `coderef/sglang` and `coderef/vllm-omni` clones, by a
subagent, the lines below then checked by hand). They agree with the diffusers
pipeline on everything the chain above takes from it. What they add:

- **A text-to-image picture is not stable at a fixed seed; an edit is.** The
  vllm-omni recipe for this model
  (`coderef/vllm-omni/recipes/MiniMaxAI/MiniMax-H3-Image-H200.md`,
  "Correctness guard") says text-to-image outputs of its different GPU
  layouts differ from each other and that the same spread appeared inside the
  sglang reference; its quantization page
  (`coderef/vllm-omni/docs/user_guide/quantization/fp8.md`) says the same of
  fp8 against bf16, with edits staying close. So a text-to-image pair at one
  seed says nothing about a quantized file. Compare edits, or many seeds.
- **Precision.** sglang runs the DiT in bf16 with its fp32 islands kept,
  encodes references with the VAE's fp32 weights and decodes in bf16
  (`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3_image.py`).
  The edit graphs here do the same since 2026-11-11 (the owner's decision):
  `MiniMaxH3VAEPrecision` promotes the encoder to fp32 for the picture being
  edited and leaves the decoder in half precision (clearing ComfyUI's cached
  `VAE.size` so `model_management` sees the promoted footprint). Although
  `PH3_IMAGE_VAE` (`p-h3-image_vae_bf16.safetensors`) stores its 118 encoder +
  `quant_conv` tensors in fp32 on disk (`max_diff = 0` against `H3_IMAGE_VAE_FP32`),
  `comfy/sd.py` casts the entire loaded `PH3ImageVAE` `state_dict` to `vae_dtype`
  (`fp16`) during `VAE.__init__` (`max_abs_diff = 2.98e-8` across all 118 encoder
  tensors); `MiniMaxH3VAEPrecision(encoder="fp32")` therefore provides **FP32
  Conv3d/GroupNorm activation accumulation** without loading a second VAE file.
  On the release's edit case the change moves the
  render about as much as the int8 file does against bf16
  (`edit_graphs_with_fp32_reference_encode` in the renders JSON), which is to
  say within the spread the vendor's servers show between their own layouts.
  It is the vendor's way of encoding; nothing here shows it looks better.
- **On the small points where core differs from the diffusers pipeline,
  the two servers side with the pipeline**: the reference's VAE posterior sample
  comes from the request's seed (`H3ImageVAE.encode` uses fixed `seed=42` while
  `MiniMaxH3._cond_video_rows` seeds the `0.001` reference noise from the
  sampler seed, differing only in RNG stream advancement across multiple
  references), the sampled latent is rounded to
  fp16, and an in-bounds reference is cropped rather than resized. sglang and
  diffusers draw those in a different order from each other, so there is no
  single bit-exact edit to match; the differences are in the noise a
  reference carries at its conditioning level, not in what the model is shown.
- **The noise is drawn on the CPU in the release's own example requests**
  (`generator_device=cpu`), which is what ComfyUI does. That is why the default
  text-to-image graph, at the release's seed, lands on the model card's
  "local" picture.
- **vllm-omni's advice for edits** (same recipe): give references at the
  canvas size of the requested shape, and make several changes in one edit
  rather than chaining edits. The first is not what the release's own example
  does (a small picture, a large output) and has not been tried here.
- **Neither server contains the prompt expander.** It is its own model
  (`MiniMax-H3-Image-PE`), served as a language model; its repository's README,
  kept at [`../vendor_guides/h3_image/README.md`](../../../vendor_guides/h3_image/README.md),
  says how each request is put together and how the reply is used.

### Edit references, point by point (read 2026-11-13)

How a reference picture reaches the model in each implementation. A source
read, nothing rendered: ComfyUI core `f11d16096`, `coderef/sglang`
`084a5e9aae`, `coderef/vllm-omni` `2c148ceb1`, and this pack's working tree.
The two servers agree with each other on every row.

| what | sglang and vllm-omni | core `MiniMaxH3ImageConditioning` | this pack's `MiniMaxH3EncodeImageReferences` |
|---|---|---|---|
| label ahead of each picture in the text encoder | `Picture N: ` (sglang `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/presentation.py`, vllm-omni `coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3_image.py::minimax_h3_image_presentation`) | same (`picture_label` in `comfy_extras/nodes_minimax_h3.py`) | same (`reference_encode.py::H3_IMAGE_PICTURE_LABEL`) |
| a role for the first picture | none; references are an ordered list | none in the encode; the first sizes the canvas when width or height is unset | none in the encode; the first sizes the canvas at `canvas="from_image_1"` |
| size bounds on the short edge, text-encoder copy and VAE copy | `edit_reference_qwen_*_short_edge`, `edit_reference_vae_*_short_edge` (sglang `coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3_image.py`); `MINIMAX_H3_IMAGE_QWEN_SHORT_EDGE`, `MINIMAX_H3_IMAGE_VAE_SHORT_EDGE` (vllm-omni) | same values (`IMAGE_REF_SHORT_EDGE`, `IMAGE_REF_VAE_MAX_SHORT_EDGE`) | same values by default (`H3_IMAGE_QWEN_SHORT_EDGE`, `H3_IMAGE_VAE_MIN_SHORT_EDGE`, `H3_IMAGE_VAE_MAX_SHORT_EDGE`), each an input |
| a picture outside the bounds | scaled so the short edge lands on the nearer bound, each side to the nearest grid multiple, Lanczos | same (`_size_short_edge`, `_resize_to`) | same (`_h3_image_ref_size`) |
| **a picture inside the bounds** | the original pixels; the VAE copy is cropped from the top left to the grid (sglang `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/keyframe_encoding.py::_encode_keyframe_cond_rows`, its `encode_images` crop then `crop_to_even`; vllm-omni `coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/image_vae.py::encode_image_latent`, `minimax_h3_image_reference_grid`) | both sides rounded to the grid and the picture **resampled** with Lanczos | resampled at `in_bounds_align="resize_32"`, the default; cropped as the servers do at `"crop_32"` |
| **the reference latent's posterior sample** | drawn from the request's generator, after the target noise and the references' anchor noise (sglang `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/image_noise.py`, vllm-omni `coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3_image.py`, `forward`) | a fixed seed (`comfy/ldm/minimax/vae.py::IMAGE_ENCODE_SEED`, in `H3ImageVAE.encode`) | the same fixed seed: the node calls the same `vae.encode` |
| **the sampled latent** | rounded to fp16 before it is normalized (`use_fp16_latent`; vllm-omni `encode_image_latent`) | not rounded | not rounded |
| **the noise mixed into each reference at its conditioning level** | one draw per reference, in order, from the request's generator | every reference restarts one stream at the sampler's seed (`comfy/ldm/minimax/model.py::_cond_video_rows`), so two references of one shape carry the same noise | the same: it is the model's code, not the node's |
| encoder precision | fp32 in sglang (the same `_encode_keyframe_cond_rows`); vllm-omni's encoder weights not read | whatever the VAE was loaded as | the same; the shipped edit graphs put `MiniMaxH3VAEPrecision(encoder="fp32")` ahead of either node |

So for this model core is close to the servers, which was not true of H3
video's references. Four rows differ (bold). The pack's node closes one of
them, the crop, and only when `in_bounds_align` is `"crop_32"`. No shipped
graph sets that: the plain edit graphs use core's node, and
`workflows/build_workflows.py::_wire_edit_guidance` swaps the pack's node in
without naming `in_bounds_align`, so it runs at the default.

The crop row is the one a picture's own size decides. It applies when a
picture's short edge is already inside the VAE bounds and its sides are not
grid multiples, which is the common case for a pasted or uploaded picture:
core resamples the whole picture by a fraction of a grid cell, the servers
trim the right and bottom edges and leave every remaining pixel as it was. A
picture outside the bounds is resampled by all four.

What the pack's node would need to match both servers, and where each
change lives:

1. `in_bounds_align="crop_32"`, as the default or written by the builder.
   Nothing else in the sizing differs: trimming to the 32-pixel grid from the
   top left is the servers' 16-pixel crop followed by their even latent grid.
2. Sample the reference latent itself, from the render's seed, instead of
   calling `vae.encode`: a seed input on the node, the encoder's moments, and
   a generator at that seed. This makes the sample follow the seed as the
   servers' does. It cannot make a same-seed render match theirs: they draw
   target noise, anchor noise and posterior samples from one generator in that
   order, and here the first two are drawn by `RandomNoise` and by
   `_cond_video_rows`, outside any conditioning node.
3. Round that sample to fp16 before normalizing. One line once the node
   samples for itself.
4. The anchor-noise row is in the model, not a node. Matching it means a
   runtime patch of `_cond_video_rows`, as the pack's other model patches are
   done; nothing here shows it is worth one.

With 1 to 3 done the pack's node would differ from core's in exactly those
three ways, plus the split it already has (pictures encoded once, any number
of prompts put on top).

The conditioning level itself is the same number in core and sglang
(`comfy/ldm/minimax/model.py::VISUAL_COND_TIMESTEP`, sglang
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py::MINIMAX_H3_IMGVID_COND_TIMESTEP`); only where its noise
comes from differs.

Not compared: the picture preprocessing inside each text encoder's own
processor, the servers' padding of the token ids, and vllm-omni's refusal
of a very long reference (`MINIMAX_H3_IMAGE_MAX_REFERENCE_ASPECT`). Whether
any row is visible in a render is not known; the fp32 encode note above is
the only one of these that has been rendered.

## Prompts

The model is given one long, literal English description, used as it is.
MiniMax write that description with a language model, and published the
instructions they give it in [`../vendor_guides/h3_image/`](../../../vendor_guides/h3_image/)
alongside the release's worked examples.
[`wiki/system_prompts/h3_image/`](../system_prompts/h3_image/README.md) holds
our standalone, English-only system-prompt suite for local thinking LLMs/VLMs
(general scenes vs. design/typography/infographics for T2I, and `1–16`
`<Picture N>` edits, each available as plain prose or with a trailing
`aspect_ratio:` line parsed by `h3_image_nodes.py::parse_expander_reply`).

- **Text to image**: concise, length-proportional framing and scene prose
  (around `45–90` words for focused requests; `"double-quoted"` visible text
  when requested). The ratio variant appends `aspect_ratio: <10 presets>`.
- **An edit**: one continuous 3-beat natural paragraph (`"double-quoted"` visible
  text) anchoring attached images as `<Picture 1>`, `<Picture 2>`, ..., stating
  what remains unchanged, what changes or transfers, and how the finished target
  image looks without synthetic `<Subject N>` tags. The ratio variant appends
  `aspect_ratio: <same as the picture | W:H>`.

The two default graphs (`h3_image_t2i_api.json`, `h3_image_edit_api.json`) take
a finished description directly. `h3_image_t2i_heylook_api.json` and
`h3_image_edit_heylook_api.json` wire `MiniMax H3 Image Prompt Expander (heylook)`
(`h3_image_nodes.py::MiniMaxH3ImagePromptExpander`) ahead of conditioning and
`MiniMax H3 Image Size.shape_from_expander`, expanding a short request or
`1..16`-picture edit instruction over `POST /v1/messages` on a local or LAN
`heylookitsanllm` server (`base_url` defaults to `http://127.0.0.1:8080`; model
ids are resolved at runtime from `GET /v1/models`, gating on `vision` when
pictures are wired). Do not feed `general_image_edit_system_prompt.md` (`[Shot 1]`
4-section schema) to `MiniMax-H3-Image`: `[Shot N]` splits the canvas into panels
and unanchored attributes override `<Picture 1>`.

### Prompts written for H3 video

Tried on 2026-11-11 at the owner's ask: six cases with prompts in the two H3
video guides' formats, unchanged, each on the int8 workflow and on the bf16
reference at one seed (`videoprompt_cases` in the renders JSON; side-by-side
sheets `videoprompt_sheet_*.jpg` beside the renders). What a session saw in
the stills, for the owner to check:

- **The model takes the format.** Every case gave a coherent picture of the
  scene the prompt describes. No field name, speaker tag or line of dialogue
  was drawn as lettering, and the sound and music fields left no visible
  trace.
- **`[Shot N]` is read as panels.** A one-location prompt with three shots
  (the covered market) came out as one frame. Two prompts whose three shots are
  different framings (a diner, a stairwell) came out as one image divided into
  panels, one per shot, in the prompt's order.
- **The reference format works for an edit.** With one picture and the six
  sections of the reference guide, the woman in the picture stands in the
  diner the prompt asks for, with her face, hair and freckles. With two
  pictures and a three-shot script, the result is a grid of four panels, the
  first shot drawn twice across the top and the two close-ups below, the
  elderly man taken from his picture without his cap or the mark on his
  cheek. The same script as text to image gave three panels.
- **Words in the prompt outrank the picture where they disagree.** That
  two-picture prompt describes the woman with dark shoulder-length hair; the
  picture wired for her is of a woman with long blonde hair and freckles. The
  render gave her dark hair, and her face does not read as the picture's
  either (crops of the picture, the diner edit and the stairwell edit side by
  side: the diner edit is plainly the same person, the stairwell woman is
  not). The man, whose description agrees with his picture, kept his face.
- **Picture numbering did not decide who was who in that case.** Wired in
  either order, the two pictures gave nearly the same image: each subject
  followed its description to the matching picture.
- **int8 and bf16 agree in layout** on all six. Within it they can differ in
  more than detail: on the diner prompt both drew two interior panels over a
  wide exterior, but which of the couple shares the first panel with the cook
  is swapped between them, and the exterior shows both people on one and the
  woman alone on the other.

One seed each, so none of this is a rule. It is consistent with the image
model having been trained on the family's prompt conventions, which is what
the shared text encoder and design would lead one to expect.

## Optional add-on nodes & remaining items

Everything below is strictly optional and is **not** wired into the four default
`h3_image_*_api.json` workflows:

- **1–16 reference pictures & split KV-prefix reference nodes (`reference_encode.py`):**
  `build_h3_image_api` now supports up to `16` reference pictures (`_H3_IMAGE_REF_NODES`,
  matching the model card). For multi-prompt iteration or `MiniMaxH3EditGuider` runs,
  experimental `MiniMaxH3EncodeImageReferences` and `MiniMaxH3PromptOnImageReferences`
  split reference encoding from prompt continuation: `MiniMaxH3EncodeImageReferences`
  caches the Qwen3-VL causal KV prefix and 1-stream `[1, 24, 1, H//16, W//16]`
  reference latents while exposing `qwen_short_edge` (`1024`), `vae_min_short_edge`
  (`1024`), `vae_max_short_edge` (`2048` default, configurable up to `8192` for
  ultra-sharp 4K edits), `in_bounds_align` (`"resize_32"` vs vendor `"crop_32"`),
  and `use_vae` (`True` vs encoder-only `False`). When `edit_guidance` is wired onto
  an `h3_image_edit` graph (`workflows/build_workflows.py::_wire_edit_guidance` and Studio),
  `MiniMaxH3ImageConditioning` at node `5` automatically splits into `1x`
  `MiniMaxH3EncodeImageReferences(canvas="explicit")` (`132`) feeding `2x`
  `MiniMaxH3PromptOnImageReferences` (`5`, `131`), encoding vision and FP32 VAE
  reference latents once across both edit-guidance branches.
- **Exact LoRA branch loading (`lora_branch.py::MiniMaxH3LoRABranch`):**
  `MiniMaxH3LoRABranch` supports `MiniMaxH3ImageModel` (`1`-stream, pruned `t_dim=8`
  AdaLN, `pdd_steps=6` `576`-row `final_layer.video_out`), cleanly skips audio-only
  keys and `2688`-column video `adaln_proj.linear` keys when `t_dim=8`, adds
  `modules="attention + mlp"` for spatial/style video LoRAs whose AdaLN modulation
  is pruned, and supports `576`-row PDD-6 `final_layer.video_out` LoRAs.
- **Control vectors, J-Lens & reference release (`control_vectors.py`, `j_lens.py`, `control_vector_nodes.py`):**
  `MiniMax-H3-Image` is registered as first-class family `"h3_image"`
  (`KNOWN_FAMILIES` hash `("0c92b9024acc", "6f9a10f4d0ed", "036c13e58478")`),
  `h3_spans.py` classifies core's pre-vision `"Picture N: "` labels as `label`
  without catching post-vision caption prose, `models/j_lens/j_lens_h3.safetensors`
  carries calibrated `J_ref_h3_image` and `J_wiener_h3_image` operators for
  zero-VRAM `MiniMaxH3WordVector` / `MiniMaxH3Steer`, existing `fl2va`/`ref2va`
  `.safetensors` vectors with `vector.encoder` auto-project into `h3_image`
  post-refiner space, and `MiniMaxH3Steer` (`scene_only` / `description`) steers
  post-vision caption tokens even when core's `MiniMaxH3ImageConditioning` (no
  `h3_span_masks`) is wired. `MiniMaxH3ReleaseReferences` (`end_percent < 1.0`,
  `clean_level=0.999`) also works on 1-stream 5D `MiniMaxH3ImageModel`.
- **Not yet rendered by eye:** Multi-picture (`> 2`) edits, transferred video
  LoRAs (`modules="attention + mlp"`), and `h3_image` control vectors / word
  vectors have been verified numerically on GPU/CPU but have not yet been
  compared visually across multi-seed sheets; the release's standalone fp32 VAE
  file (`H3_IMAGE_VAE_FP32`) remains unreferenced by default graphs.
