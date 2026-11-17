# sglang: the vendor's serving path (rank 1)

last updated: 2026-11-17 (MiniMax-H3-Image path read stage by stage, [`../sweeps/2026-11-17_h3_image.md`](../sweeps/2026-11-17_h3_image.md)); 2026-11-15 (page created; state as of [`../sweeps/2026-11-15.md`](../sweeps/2026-11-15.md))

**Rank 1** ([`../index.md`](../index.md), "When upstreams disagree"). sglang is the closest thing to ground truth for
what MiniMax intends at each stage. A sglang default can still move for reasons other than fidelity: the
2026-11-05 reference-sizing change was a speed PR ([`../topics/reference_sizing.md`](../topics/reference_sizing.md)).
Read the PR's motive before treating a moved default as the vendor's word.

## Where its H3 code lives (`coderef/sglang/python/sglang/multimodal_gen/`)

| area | path |
|---|---|
| stages (references, keyframes, packing, noise, denoise loops, canvas, presentation) | `runtime/pipelines_core/stages/model_specific_stages/minimax_h3/` and its `stages/` |
| constants and defaults | `.../minimax_h3/constants.py`, `configs/sample/minimax_h3.py`, `configs/sample/minimax_h3_image.py`, `.../minimax_h3/task_profiles.py` |
| pipelines (video, Pika, FastH3, image) | `runtime/pipelines/minimax_h3_pipeline.py`, `configs/pipeline_configs/minimax_h3*.py` |
| DiT, VAEs, encoder | `runtime/models/dits/minimax_h3*.py`, `runtime/models/vaes/minimax_h3_*`, `runtime/models/encoders/minimax_h3_qwen3vl.py` |
| Sol attention | `runtime/layers/attention/backends/sol_attn.py` |
| ComfyUI checkpoint loading, ComfyUI app | `runtime/loader/comfyui_checkpoints/`, `apps/ComfyUI_SGLDiffusion/` |
| docs | `docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx` (repo root) |

## Deep pages in this wiki

- [`../../research/sglang_h3_pipeline.md`](../../research/sglang_h3_pipeline.md): what it does, stage by stage.
- [`../../research/sglang_comparison.md`](../../research/sglang_comparison.md): what it does that we do not. Reads one
  to eleven, plus the owner's 2026-09-11 adopt-upstream decision.
- [`../../attention/sol_upstream.md`](../../attention/sol_upstream.md): Sol.
- [`../../image/h3_image.md`](../../image/h3_image.md): MiniMax-H3-Image, "What the vendor's servers do".

## State as of 2026-11-15

- **Base H3 sampling.** Shifts 12/3 and the visual condition level `MINIMAX_H3_IMGVID_COND_TIMESTEP` are unchanged
  since the 2026-11-04 reads, as are Euler with one positive forward and the video VAE's fp32 encode and fp16 decode.
- **References.** Small stills are no longer enlarged by default (`max`, since `92a6910ef2`); see
  [`../topics/reference_sizing.md`](../topics/reference_sizing.md). There is a per-image cache of reference and
  keyframe rows (`b7ccc7e2f9`).
- **Encoder.** Layer-50 pre-norm, bf16. The vision RoPE is rebuilt on the GPU for image releases only
  ([`../topics/qwen3vl_encoder.md`](../topics/qwen3vl_encoder.md)).
- **Pika.** TF32 matmul off, LoRA factors at bf16 unmerged, and separate seeded RNG streams
  ([`../topics/pika_parity.md`](../topics/pika_parity.md)).
- **MiniMax-H3-Image.** One positive branch, with a float32 velocity Euler step
  ([`../topics/h3_image_sampling_step.md`](../topics/h3_image_sampling_step.md)).
- **LoRA.** Stacked dynamic adapters, each with its own rank, alpha and strength (`b966c2798e`).
- **Comparisons.** The cookbook says `exact` and `lossless`, which differ only in in-op rounding, grow into different
  samples over a long schedule. Same-seed cross-engine A/Bs are read with that in mind.
- **Not in sglang.** No edit, continuation or window path for video (the ninth to eleventh reads), and no H3 prompt
  rewriting.

## MiniMax-H3-Image as of 2026-11-17

Read stage by stage, at the tip on 2026-11-17. Every implementation is compared on
[`../topics/h3_image_pipeline.md`](../topics/h3_image_pipeline.md). "By hand" marks a line checked against the file
this session. The rest is the agent's reading. Paths are under `.../minimax_h3/` (the stages folder above) unless
given in full.

- **Precisions are enforced, not defaults** (by hand). `coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3_image.py::validate_server_args`
  refuses anything but a bf16 DiT, a bf16 text encoder, an fp32 VAE encode and a bf16 VAE decode, with no flow
  shift and no scheduler override.
- **Text to image** (by hand): the prompt's tokens and nothing else (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py::_compute_image_conditioning`,
  `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/presentation.py::minimax_h3_image_presentation`).
  - The prompt is used verbatim, and each segment is cut at 4096 tokens.
  - The ids are padded with id 0 to a multiple of 8, and the padded rows are dropped before the DiT (`hidden[:text_len]`).
  - An empty or blank prompt is refused (`request_validation.py`).
- **Canvas** (by hand). `min_generation_area` (in `minimax_h3_image.py`, `2048 * 2048`) is a floor: a smaller request
  is generated larger at its aspect ratio (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/image_canvas.py::minimax_h3_image_generation_canvas`, nearest 32, ties
  to even). Then it is decoded, cut to 8-bit by truncation and resized back to the request with PIL Lanczos
  (`minimax_h3_resize_image_output`).
  - The pipeline has no default size; its OpenAI protocol defaults to `1024x1024`.
  - The output size never comes from a reference.
- **Edit references:** unchanged since the 2026-11-13 read ([`../../image/h3_image.md`](../../image/h3_image.md),
  "Edit references, point by point"), re-read by hand.
  - **Bounds:** `edit_reference_qwen_*_short_edge` and `edit_reference_vae_*_short_edge`.
  - **Encode:** an in-bounds picture keeps its pixels, and its encode runs on fp32 weights
    (`keyframe_encoding.py`). The posterior is drawn from the request's generator over the latent before its even
    crop, with `use_fp16_latent=True`.
  - **Anchor level:** reference rows are pinned at `MINIMAX_H3_IMGVID_COND_TIMESTEP` (`pin_imgvid_cond=True` in
    `denoise_loop.py`).
- **Noise** (by hand): one generator per request, drawn in this order: target, anchors, posteriors (`image_noise.py`,
  header). It sits on the platform device unless `generator_device` is set.
- **Decode** (agent): the release's 2D ViT decoder, bf16 weights under bf16 autocast, tiled.
- **Not covered by the 2026-11-15 notes above:**
  - a Qwen conditioning cache keyed on the prompt, the raw pictures and the bounds (agent);
  - no reference cap in the edit profile (agent; vllm-omni and LightX2V cap at 16).

## Open upstream work

On [`../horizon.md`](../horizon.md): Open-H3-Regenerate-2K (#47668), the AdaLN LoRA dtype on pruned files (#50611),
and the Q4 roadmap with RTX 4090 defaults (#43313).
