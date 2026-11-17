# Qwen3-VL-32B: how each implementation uses the shared encoder

last updated: 2026-11-17 (added `reference_encode.py::_snap_h3_image_qwen_grid` 8-bit bicubic snap for off-grid Qwen copies; MiniMax-H3-Image read across sglang, vllm-omni, DiffSynth, LightX2V, core and this pack:
text to image carries no vision tokens anywhere; prompt handling, truncation, padding, empty prompts, Qwen-copy
resampling and mRoPE added; record [`../sweeps/2026-11-17_h3_image.md`](../sweeps/2026-11-17_h3_image.md)); 2026-11-15 (page created from the 2026-11-15 sweep)

One encoder serves every model the pack runs: H3 video, MiniMax-H3-Image, and P-H3-Image when it runs with our 32B
teacher. A change here touches all of them. The two-model trap applies: "attention" and "capture" also name parts
of the DiT, so a fact here is about the encoder only (`../../references.md`, "The trap this page exists to
prevent").

## Where they agree (as of 2026-11-15)

- **The conditioning is the layer-50 state before the final norm.** sglang replaces the final `norm` with
  `nn.Identity` (`runtime/models/encoders/minimax_h3_qwen3vl.py`). vllm-omni outputs `hidden_states[50]`
  unnormalized (`vllm_omni/diffusion/models/minimax_h3/encoder.py`). diffusers takes `hidden_states[50]` of a full
  stack, and refuses a stack of 50 layers or fewer because that last state is post-norm
  (`coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3_image/encoders.py::get_qwen3vl_prompt_embeds`). DiffSynth truncates to 50 and sets
  the norm to Identity. Core truncates to 50 (`Qwen3VL_32BConfig`). Pika keeps 51 of 64 layers for speed (`6cb647e`).
- **Precision is bf16.** No chat template, no special tokens.
- **H3-Image presentation is the same in sglang, vllm-omni, DiffSynth, LightX2V and core:** `Picture i: `, then the
  vision block for each reference, then the prompt. Vision copies are at a 1024 short edge. The details are in
  [`../../image/h3_image.md`](../../image/h3_image.md).
- **H3-Image text to image carries no vision tokens in any implementation** (2026-11-17). The text encoder gets the
  prompt's own tokens and nothing else: no picture of any kind, no `<|vision_start|>` / `<|image_pad|>`, no
  `Picture` label, no template, no special tokens. Read by hand in sglang
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py::_compute_image_conditioning`, `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/presentation.py::minimax_h3_image_presentation`),
  vllm-omni (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3_image.py::minimax_h3_image_presentation`), DiffSynth
  (`diffsynth/pipelines/minimax_h3_image.py`, the prompt embedder) and core
  (`comfy/text_encoders/minimax.py::MiniMaxH3Tokenizer.tokenize_with_weights` with no images). LightX2V is the
  agent's reading. This pack's `h3_image_t2i` graphs run core's `MiniMaxH3ImageConditioning` with no pictures, so
  they send the same tokens. The community "neutral vision patches for t2i" idea ([`../periphery.md`](../periphery.md))
  is something no engine does.
- **The vision block is tagged as image for the DiT's AdaLN, its start and end markers included; the labels and the
  prompt are tagged as text** (sglang, vllm-omni, DiffSynth, LightX2V, core's `token_tags_from_embeds_info`).
  **Inside Qwen's mRoPE only the image-pad tokens take 2D positions; the start and end markers count as text**
  (vllm-omni's encoder, core's `comfy/text_encoders/qwen_vl.py::qwen2vl_mrope_position_ids`; sglang and LightX2V use the HF-style
  `get_rope_index`).
- **The Qwen copy is bounded the same way everywhere:** Lanczos to a 1024 short edge with both sides on the 32 px grid,
  and a picture whose short edge is already exactly 1024 is passed through untouched to the processor (sglang,
  vllm-omni, DiffSynth, LightX2V, this pack's `reference_encode.py::_prepare_h3_image_ref`, then snapped onto the 32 px
  grid with 8-bit bicubic via `reference_encode.py::_snap_h3_image_qwen_grid` when its long edge is off-grid). Core is
  the exception: it resamples that picture with Lanczos as well. The image release's `processor/preprocessor_config.json`
  is byte-for-byte the H3 video release's (`vendor_config/preprocessor_config.json`, compared 2026-11-17).
- **No engine rewrites or expands H3 prompts in its serving path.** Expanders are separate: diffusers' Hub blocks for
  `MiniMaxAI/MiniMax-H3-Image-PE`, the templates' `TextGenerateV2`, and LightX2V's rewriters.

## Where they differ, or added something

| what | who | where | checked against core? |
|---|---|---|---|
| The vision RoPE table is rebuilt on the GPU after offload for **image** releases only. A CPU build computes `pow` differently, and video releases keep the CPU-built table their consistency references were made with | sglang (`6be0d75ae0`, #49574) | `runtime/pipelines/minimax_h3_pipeline.py` (the `recompute_on_device_change` flag) | no |
| `text_encoder_cudnn_sdp`: cuDNN SDPA is on by default during encode; off makes the same seed give the same image across restarts, "at a small cost in agreement with the reference" | vllm-omni | `coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/encoder.py::cudnn_sdp` | no |
| A `position_ids` argument, so the image release can pass mRoPE positions | sglang | `minimax_h3_qwen3vl.py` | |
| `CombinedLinear` fuses q/k/v and gate/up when quant layouts match; `MLP.forward` changed (not pinned by `bench/check_reference_encode.py`) | core (`48b090cb6`, #17361) | `comfy/ops.py`, `comfy/text_encoders/llama.py` | it is core |
| Video input for Qwen 3.x counts `type == "video"` spans in mRoPE; the H3 tokenizer still tags its video blocks `"image"` | core (`c85957ad7`, #17582) | `comfy/text_encoders/qwen_vl.py`, `comfy/text_encoders/minimax.py` | it is core |
| An INT4 ConvRot 32B in the H3-Image 50-layer spec; abliterated 32B encoders | periphery (Hugging Face) | [`../sweeps/2026-11-15.md`](../sweeps/2026-11-15.md), "New repos, weights and papers" | |
| **The prompt is tokenized as given.** No engine strips it. This pack strips both ends (`h3_rules.py::normalize_prompt`) in every node of its own, which includes the H3-Image edit path, but not in text to image, which runs core's node | sglang (its `presentation.py` header: "Prompt text passes through verbatim"), vllm-omni, DiffSynth, LightX2V, core | `presentation.py`; `pipeline_minimax_h3_image.py` | core agrees with the engines; the pack is the outlier, by its own rule |
| **Each text segment is cut at a fixed token count** (`_TEXT_SEGMENT_MAX_TOKENS` in vllm-omni; `[:4096]` in sglang; `MINIMAX_H3_IMAGE_MAX_SEQUENCE_LENGTH` in DiffSynth; LightX2V's `max_sequence_length`). The label and the prompt are cut separately; there is no overall cap | sglang, vllm-omni, DiffSynth, LightX2V | the presentation functions | core does not cut (`Qwen3VLSDTokenizer` `max_length`); no prompt here comes near the limit |
| **The ids are padded to a multiple of 8 with id 0, and the padded rows are dropped before the DiT** (`hidden[:text_len]`). The encoder is causal, so the padding never changes the kept rows | sglang (`text_encoding.py`), vllm-omni | the same | core pads nothing; the DiT sees the same rows |
| **An empty prompt** is refused by sglang (`request_validation.py`), vllm-omni and LightX2V. DiffSynth encodes `" "` | the four | | core encodes a single pad token (`151643`) and runs, so this pack's text-to-image graph (core's node) accepts an empty prompt. The pack's own nodes refuse one |
| **The vision tower's pixels are cast to bf16** before the vision encoder | sglang, vllm-omni (agent's reading) | `minimax_h3_qwen3vl.py`; `encoder.py` | core passes float32 pixels to `self.visual` (`comfy/text_encoders/qwen3vl.py::preprocess_embed`), which then runs at its weights' dtype; not traced further |
| **Q, K and V are three separate bf16 GEMMs "to keep the numerics identical"** | vllm-omni (agent's reading) | `encoder.py` | core fuses them (`CombinedLinear`, row above) |
| **The processor's own resize.** When the Qwen copy is not on the 32 px grid, the HF processor (`Qwen2VLImageProcessorFast`, named in the release's config with no resample set) resizes it with its class default, bicubic | sglang, vllm-omni, DiffSynth, LightX2V | the release's `processor/preprocessor_config.json`; transformers' `image_processing_qwen2_vl` | core's preprocessor resizes with bilinear and no antialias (`comfy/text_encoders/qwen_vl.py::process_qwen2vl_images`). Reached only by a picture whose short edge is already exactly 1024 and whose long edge is off the grid. Core resamples it to the grid with Lanczos first. Since 2026-11-17 this pack snaps it with `comfy/text_encoders/qwen_vl.py::resize_8bit` (`reference_encode.py::_snap_h3_image_qwen_grid`), matching `Qwen2VLImageProcessorFast` within `5.91e-08` (`max_abs < 1e-6`, vs `0.2307` unfixed) and leaving core alone as the outlier |
| **The processor's pixel limits** are the release config's `size` (`shortest_edge` and `longest_edge`, `vendor_config/preprocessor_config.json`). Core's `comfy/text_encoders/qwen_vl.py::process_qwen2vl_images` uses Qwen2-VL's defaults instead, which differ from the release's on both limits (read 2026-11-17) | sglang, vllm-omni, diffusers, DiffSynth, LightX2V (the HF processor reads the config) | the release's `processor/preprocessor_config.json` | `reference_encode.py::_snap_h3_image_qwen_grid` passes the release config's `(65536, 16777216)` bounds into `comfy/text_encoders/qwen_vl.py::smart_resize` and resizes off-grid copies via `comfy/text_encoders/qwen_vl.py::resize_8bit` (as core does for another model in `comfy/text_encoders/wevisone.py`) |
| **A conditioning cache keyed on the prompt, the raw pictures and the bounds** | sglang (agent's reading) | `.../minimax_h3/stages/text_encoding.py` | this pack keeps the reference prefix in `MiniMaxH3EncodeImageReferences` for as long as its output is wired |

## Open

- **Where core builds the vision RoPE table: narrowed 2026-11-17** (read by hand, after a peer session's review).
  - `comfy/text_encoders/qwen35.py::Qwen35VisionRotaryEmbedding` computes `inv_freq` once, in its constructor, on
    the device the encoder is built on. `forward` then runs `torch.outer` wherever the buffer has moved.
  - Core builds the encoder on `comfy/model_management.py::text_encoder_initial_device`. That is the offload device
    (the CPU) whenever dynamic memory is on, or the encoder is larger than the card's free memory, as the bf16 32B
    is. So for this pack the table is the CPU build.
  - sglang rebuilds the table on the GPU for image releases only. It keeps the CPU build for video, whose
    consistency references were made with it.
  - Still open: whether the difference is visible on an edit's vision tokens. Text to image has none.
- **Which attention core's encoder runs: narrowed 2026-11-17** (read by hand).
  - The language model and the vision tower both ask for `optimized_attention_for_device(..., small_input=True)`
    (`comfy/text_encoders/llama.py`, `comfy/text_encoders/qwen35.py`).
  - That is PyTorch SDPA (`comfy/ldm/modules/attention.py::attention_pytorch`) whenever PyTorch attention is
    enabled, which on an NVIDIA card with torch 2 is the default. It holds whatever the server's sage or kitchen
    flags say; those choose the DiT's attention, not this.
  - The language model passes an explicit additive causal mask with `is_causal=False`, so SDPA's flash kernel is
    not eligible, and it chooses among memory-efficient, cuDNN and math; all three are enabled in the venv's torch.
    vllm-omni calls SDPA with `is_causal=True` (agent's reading), so the two may run different kernels.
  - Still open: which kernel core actually gets, and whether it reproduces across restarts. vllm-omni says its
    cuDNN default does not.
