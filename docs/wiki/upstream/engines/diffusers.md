# diffusers: the library port (rank 3)

last updated: 2026-11-15 (page created; state as of [`../sweeps/2026-11-15.md`](../sweeps/2026-11-15.md))

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

## Open upstream work

On [`../horizon.md`](../horizon.md): loading the pruned ComfyUI checkpoints (#15299), Comfy int8 ConvRot through
comfy-kitchen (#15209), and ref2va soundtrack resampling (#15199 / #15245).
