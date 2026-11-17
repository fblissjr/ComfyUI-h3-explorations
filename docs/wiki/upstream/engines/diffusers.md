# diffusers: the library port (rank 3)

last updated: 2026-11-17 (MiniMax-H3-Image path read stage by stage, [`../sweeps/2026-11-17_h3_image.md`](../sweeps/2026-11-17_h3_image.md)); 2026-11-15 (page created; state as of [`../sweeps/2026-11-15.md`](../sweeps/2026-11-15.md))

**Rank 3.** Its modular pipelines are the clearest statement of each model's tensor namespace and sampler, and the
release's Hugging Face layout follows it. Where it disagrees with sglang or vllm-omni, they win.

## Where its H3 code lives (`coderef/diffusers/src/diffusers/`)

| area | path |
|---|---|
| H3 video | `modular_pipelines/minimax_h3/` |
| MiniMax-H3-Image | `modular_pipelines/minimax_h3_image/`, `models/transformers/transformer_minimax_h3_image.py`, `models/autoencoders/autoencoder_kl_minimax_h3_image.py` |
| P-H3-Image | `modular_pipelines/p_h3_image/` |
| Pika H3 Flex | `modular_pipelines/pika_h3_flex/` |
| scheduler | `schedulers/scheduling_minimax_h3.py` |
| converter | `scripts/convert_minimax_h3_to_diffusers.py` (repo root) |

## Deep pages in this wiki

- [`../../image/p_h3_image.md`](../../image/p_h3_image.md): the P-H3-Image contract, read at `c55ec279`.
- [`../../image/h3_image.md`](../../image/h3_image.md): MiniMax-H3-Image.
- `../../references.md`, "Pika H3 Flex".

## State as of 2026-11-15

- **References.** Stills are scaled to a 2048 short edge, "2048 for the released checkpoint"
  (`modular_pipelines/minimax_h3/before_encoder.py`); see [`../topics/reference_sizing.md`](../topics/reference_sizing.md).
- **MiniMax-H3-Image** (`875373ac7`, #15230).
  - Modular only, with no CFG, and an empty prompt is refused.
  - Uses the velocity Euler form ([`../topics/h3_image_sampling_step.md`](../topics/h3_image_sampling_step.md)).
  - The reference noise draws are in an order of their own, listed on the same page.
- **Prompt enhancement.** Custom Hub blocks run `MiniMaxAI/MiniMax-H3-Image-PE` and return `prompt` and
  `aspect_ratio`. Their code is on the Hub and was not read.
- **Encoder.** `hidden_states[50]` of a full stack ([`../topics/qwen3vl_encoder.md`](../topics/qwen3vl_encoder.md)).
- **Tensor parallelism** for the H3 transformer (`7e46c94ca`): multi-GPU only.

## MiniMax-H3-Image as of 2026-11-17

Read stage by stage at the tip on 2026-11-17. Every implementation is compared on
[`../topics/h3_image_pipeline.md`](../topics/h3_image_pipeline.md). "By hand" marks a line checked against the file
this session. Paths are under `modular_pipelines/minimax_h3_image/` unless given in full.

- **It agrees with sglang and vllm-omni on every stage read except the order of the random draws.** It draws every
  posterior, then the target, then every anchor; the ranked two draw target, anchors, posteriors. LightX2V copies
  diffusers' order.
- **Text to image** (by hand, `encoders.py`):
  - the prompt's tokens only, verbatim and not stripped, cut at `max_sequence_length`, with no padding;
  - every row tagged text;
  - an empty prompt refused, though a blank one passes.
- **Edit references** (by hand for the encode, `encoders.py`):
  - **The Qwen copy** is bounded in `coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3_image/before_encoder.py::bound_short_edge`, Lanczos through `VaeImageProcessor`.
  - **The VAE copy** keeps its pixels inside its bounds. It is cropped top left to the 16 px cell, the posterior is
    sampled from the request's generator and rounded to fp16, and then it is cut to an even grid.
  - **The VAE encoder is pinned to fp32** (`_keep_in_fp32_modules` in
    `models/autoencoders/autoencoder_kl_minimax_h3_image.py`).
  - **The mix at the reference level** uses a float32 tensor so that `1 - t` is taken in float32.
- **Canvas** (by hand, `before_denoise.py`): the request (default 2048x2048), raised to `min_generation_area`. The
  edit size comes from the request, never a picture.
- **Decode** (by hand, `decoders.py`): the decode is cut to 8 bits by truncation, "the released pipeline quantizes
  its images to 8 bits by truncation", then resized back with PIL Lanczos when the canvas was raised.
  - Tiles blend against their neighbours' raw decodes (agent; the 2026-11-14 VAE inspection read the same).
- **Step:** velocity, from the converter's `euler_update_form: "velocity"`. The class default is `blend`, so a
  scheduler config without that key would step differently (agent).
- **Agent's inference, not run:** under a bf16 load `decoder.proj_in` is not kept in fp32, despite a comment saying
  `proj_in` is pinned. The loader matches whole dotted name parts.

## Open upstream work

On [`../horizon.md`](../horizon.md): loading the pruned ComfyUI checkpoints (#15299), Comfy int8 ConvRot through
comfy-kitchen (#15209), and ref2va soundtrack resampling (#15199 / #15245).
