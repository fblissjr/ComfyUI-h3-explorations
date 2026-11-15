# Qwen3-VL-32B: how each implementation uses the shared encoder

last updated: 2026-11-15 (page created from the 2026-11-15 sweep)

One encoder serves every model the pack runs: H3 video, MiniMax-H3-Image, and P-H3-Image when it runs with our 32B
teacher. A change here touches all of them. The two-model trap applies: "attention" and "capture" also name parts
of the DiT, so a fact here is about the encoder only (`../../references.md`, "The trap this page exists to
prevent").

## Where they agree (as of 2026-11-15)

- **The conditioning is the layer-50 state before the final norm.** sglang replaces the final `norm` with
  `nn.Identity` (`runtime/models/encoders/minimax_h3_qwen3vl.py`). vllm-omni outputs `hidden_states[50]`
  unnormalized (`vllm_omni/diffusion/models/minimax_h3/encoder.py`). diffusers takes `hidden_states[50]` of a full
  stack, and refuses a stack of 50 layers or fewer because that last state is post-norm
  (`modular_pipelines/minimax_h3_image/encoders.py::get_qwen3vl_prompt_embeds`). DiffSynth truncates to 50 and sets
  the norm to Identity. Core truncates to 50 (`Qwen3VL_32BConfig`). Pika keeps 51 of 64 layers for speed (`6cb647e`).
- **Precision is bf16.** No chat template, no special tokens.
- **H3-Image presentation is the same in sglang, vllm-omni, DiffSynth, LightX2V and core:** `Picture i: `, then the
  vision block for each reference, then the prompt. Vision copies are at a 1024 short edge. The details are in
  [`../../image/h3_image.md`](../../image/h3_image.md).
- **No engine rewrites or expands H3 prompts in its serving path.** Expanders are separate: diffusers' Hub blocks for
  `MiniMaxAI/MiniMax-H3-Image-PE`, the templates' `TextGenerateV2`, and LightX2V's rewriters.

## Where they differ, or added something

| what | who | where | checked against core? |
|---|---|---|---|
| The vision RoPE table is rebuilt on the GPU after offload for **image** releases only. A CPU build computes `pow` differently, and video releases keep the CPU-built table their consistency references were made with | sglang (`6be0d75ae0`, #49574) | `runtime/pipelines/minimax_h3_pipeline.py` (the `recompute_on_device_change` flag) | no |
| `text_encoder_cudnn_sdp`: cuDNN SDPA is on by default during encode; off makes the same seed give the same image across restarts, "at a small cost in agreement with the reference" | vllm-omni | `vllm_omni/diffusion/models/minimax_h3/encoder.py::cudnn_sdp` | no |
| A `position_ids` argument, so the image release can pass mRoPE positions | sglang | `minimax_h3_qwen3vl.py` | |
| `CombinedLinear` fuses q/k/v and gate/up when quant layouts match; `MLP.forward` changed (not pinned by `bench/check_reference_encode.py`) | core (`48b090cb6`, #17361) | `comfy/ops.py`, `comfy/text_encoders/llama.py` | it is core |
| Video input for Qwen 3.x counts `type == "video"` spans in mRoPE; the H3 tokenizer still tags its video blocks `"image"` | core (`c85957ad7`, #17582) | `comfy/text_encoders/qwen_vl.py`, `comfy/text_encoders/minimax.py` | it is core |
| An INT4 ConvRot 32B in the H3-Image 50-layer spec; abliterated 32B encoders | periphery (Hugging Face) | [`../sweeps/2026-11-15.md`](../sweeps/2026-11-15.md), "New repos, weights and papers" | |

## Open

- Where core builds the vision RoPE table, and whether that differs between our video and image paths.
- Whether core's encoder attention reaches cuDNN SDPA. If it does, whether that is a seed-reproducibility issue
  across restarts here.
