# vllm-omni: the second H3 serving engine (rank 2)

last updated: 2026-11-22 (state moved to [`../sweeps/2026-11-22.md`](../sweeps/2026-11-22.md): `keep_steps` and Heun merged, SM89 VAE fused ops, a ComfyUI control node); 2026-11-17 (MiniMax-H3-Image path read stage by stage, [`../sweeps/2026-11-17_h3_image.md`](../sweeps/2026-11-17_h3_image.md)); 2026-11-15 (page created; state as of [`../sweeps/2026-11-15.md`](../sweeps/2026-11-15.md))

**Rank 2.** It is the most active H3 engine and covers the most ground that sglang does not: distill contracts
(FastH3, Regenerate), long-clip machinery (timeline guides, continuation, windows), masks and edits, a ComfyUI app,
and agent skills for H3 prompting and character replacement.

## Where its H3 code lives (`coderef/vllm-omni/`)

| area | path |
|---|---|
| DiT pipeline, sampling, denoise loop, continuation, timeline guides, latent masks, Regenerate, H3-Image | `vllm_omni/diffusion/models/minimax_h3/` |
| request preprocessing, conditioning, encoder processing, reference video | `vllm_omni/model_executor/models/minimax_h3/` |
| encoder | `vllm_omni/diffusion/models/minimax_h3/encoder.py` |
| recipes | `recipes/MiniMaxAI/`, `recipes/models/MiniMaxAI/` |
| ComfyUI app | `apps/ComfyUI-vLLM-Omni/` |
| agent skills | `.agents/skills/vllm-omni-h3-character-replacement/`, `.agents/skills/h3-prompt-writing/` |
| attention schedules | `vllm_omni/diffusion/attention/schedule.py`, `docs/user_guide/diffusion/attention_backends/schedules.md` |

## Deep pages in this wiki

- `../../references.md`: "What moved by" sections through 2026-11-04, "FastH3 Omni", "Long clips upstream".
- [`../../image/h3_image.md`](../../image/h3_image.md) and
  [`../../research/image/2026-11-14_h3_image_vae_inspection.md`](../../research/image/2026-11-14_h3_image_vae_inspection.md):
  its H3-Image path.

## State as of 2026-11-22

- **Base H3 defaults** are unchanged since 2026-11-04 (shifts, condition level, Qwen tap). The 50-step fallback is
  the named constant `coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/time_request.py::MINIMAX_H3_DEFAULT_NUM_INFERENCE_STEPS`.
- **Reference release** (`bbe6e7c85`, #9762). `keep_steps` conditions on a ref2va video reference for the first
  k steps, then moves to an image-only layout with the prompt re-encoded without the video and positions rebuilt.
  `MiniMaxH3ReleaseReferences` keeps positions and the text context instead. Off by default; refuses audio
  references and Pika ([`../sweeps/2026-11-22.md`](../sweeps/2026-11-22.md), item 6).
- **References.** A still keeps its own size rounded to 32 and is never enlarged
  (`coderef/vllm-omni/vllm_omni/model_executor/models/minimax_h3/preprocessing.py::resolve_minimax_h3_reference_image_shape`). See
  [`../topics/reference_sizing.md`](../topics/reference_sizing.md).
- **Long clips (ComfyUI app).**
  - Defaults are `continuation_window_frames=311`, step 17, with overlap 39 (`apps/ComfyUI-vLLM-Omni/comfyui_vllm_omni/nodes.py`).
  - 311 is off our `h3_rules.on_both_clocks`, so this is where their chain rule differs from ours.
- **Character replacement** (`b8f128171`, #9194).
  - The method is reference conditioning only, with no pixel lock, using the Character Swap LoRA and six-section
    `<Subject N>` prompts.
  - The skill claims, without measurements, that segments of about five seconds hold identity better and that a
    segment can be generated backward.
  - Read it against [`../../masked_v2v.md`](../../masked_v2v.md).
- **Attention schedules by step or by sigma window** (`62876f40a`, `e25c9682a`): Sol's dense-or-sparse choice,
  keyed on noise level.
- **Samplers.** The sampler owns the loop, and `euler`, `res_multistep` or `heun` can be chosen per request
  (`7465c0f52`; Heun `6492091b4`, #10400). Heun runs on each stream's own grid, which core's `heun` does not
  (item 7 of the 2026-11-22 sweep). DPM++ 2M is an open PR.
- **Video VAE on SM89.** The decoder runs fused ops by default on SM89, held to bitwise equality with their eager
  path (`f8e650153`, #10516).
- **ComfyUI app.** A control node, `VLLMOmniMiniMaxH3Control`, for canny, depth, HED, MLSD, pose and inpaint
  control, prepared on the client (`0a883471e`, #7492).
- **Encoder.** `text_encoder_cudnn_sdp` ([`../topics/qwen3vl_encoder.md`](../topics/qwen3vl_encoder.md)).
- **Fixes:** the shifted first sigma is clamped to 1.0 (`d3414a3b0`; does not arise at shifts 12/3); Regenerate's
  missing control fields (`61aef20b7`).

## MiniMax-H3-Image as of 2026-11-17

Read stage by stage, at the tip on 2026-11-17. Every implementation is compared on
[`../topics/h3_image_pipeline.md`](../topics/h3_image_pipeline.md). "By hand" marks a line checked against the file
this session. Paths are under `vllm_omni/diffusion/models/minimax_h3/`; the pipeline is
`pipeline_minimax_h3_image.py`.

- **It agrees with sglang on every stage read except the defaults at the request edge.**
- **Text to image** (by hand, `minimax_h3_image_presentation`):
  - the prompt's tokens only, verbatim;
  - each segment cut at `_TEXT_SEGMENT_MAX_TOKENS`;
  - the ids padded to a multiple of 8, with the padding dropped before the DiT;
  - an empty or blank prompt refused.
  - cuDNN SDPA is on during the encode by default (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/encoder.py::cudnn_sdp`,
    [`../topics/qwen3vl_encoder.md`](../topics/qwen3vl_encoder.md)).
- **Canvas** (by hand):
  - The default request is `MINIMAX_H3_IMAGE_DEFAULT_SIZE` square.
  - `MINIMAX_H3_IMAGE_DEFAULT_MIN_GENERATION_AREA` is a floor (`minimax_h3_image_generation_canvas`). A request may
    set its own floor within `MINIMAX_H3_IMAGE_MIN_GENERATION_AREA_RANGE`.
  - The decode is quantized to 8-bit by truncation and resized back with Lanczos.
  - The output size never comes from a reference.
- **Edit references** (by hand):
  - **Bounds:** `MINIMAX_H3_IMAGE_QWEN_SHORT_EDGE` and `MINIMAX_H3_IMAGE_VAE_SHORT_EDGE`; a short edge already
    inside keeps the original pixels (`minimax_h3_image_reference_size` returns `None`).
  - **Limits:** at most `MINIMAX_H3_IMAGE_MAX_REFERENCES` pictures, each refused past
    `MINIMAX_H3_IMAGE_MAX_REFERENCE_ASPECT`.
  - **Encode:** the VAE is loaded and kept in fp32 for the encode (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/image_vae.py::from_pretrained`). The picture is
    cropped to the 16 px cell and encoded tiled. The posterior is drawn from the request's generator over the full
    grid, then rounded to fp16 (`encode_image_latent`), and the latent is cut to an even grid
    (`minimax_h3_image_reference_grid`).
- **Noise** (by hand): the same order as sglang (target, anchors, posteriors; the comment at the top of `forward`),
  on the platform device unless `generator_device=cpu`.
- **Step** (by hand): velocity, float32, on the target rows only (`_euler_target_step`), on the release's
  `pdd_config.json` grid after the fp32 `1 - sigma` round trip.
- **Decode** (by hand for the cast): `prepare_decoder_weights` casts the decoder to bf16 at first decode and keeps
  its 2D RoPE bases in fp32. It decodes tiled.
- **Recipes** (agent): the H200 recipe's correctness guard and edit advice are on
  [`../../image/h3_image.md`](../../image/h3_image.md). The XPU recipe adds that an edit prompt should name
  `<Picture N>` and say what is kept and what changes, and that "a one-line instruction gives visibly worse edits".
  This pack's edit system prompts already ask for that form
  ([`../../system_prompts/h3_image/README.md`](../../system_prompts/h3_image/README.md)).
- **Not present:** an H3-Image node in `apps/ComfyUI-vLLM-Omni` (agent).

## Open upstream work

On [`../horizon.md`](../horizon.md): Pika parity (#9777), reference isolation (#9689), DPM++ 2M (#10405), the Sol
backend (#10513, approved), ComfyUI single-file and int8 ConvRot checkpoints (#9721, #6894), startup LoRA fusion
(#10742), an H3-Image prompt-embedding cache (#10369), kernel-choice bimodality (#6673), VAE tile batching bits
(#10256), non-24 fps (#9398), the P-H3-Image request (#9863), and the parity RFC (#7380).
