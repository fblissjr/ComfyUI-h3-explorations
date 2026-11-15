# vllm-omni: the second H3 serving engine (rank 2)

last updated: 2026-11-15 (page created; state as of [`../sweeps/2026-11-15.md`](../sweeps/2026-11-15.md))

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

## State as of 2026-11-15

- **Base H3 defaults** are unchanged since 2026-11-04 (shifts, condition level, Qwen tap).
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
- **Samplers.** The sampler owns the loop, and `euler` or `res_multistep` can be chosen per request (`7465c0f52`).
  Heun and DPM++ 2M are open PRs.
- **Encoder.** `text_encoder_cudnn_sdp` ([`../topics/qwen3vl_encoder.md`](../topics/qwen3vl_encoder.md)).
- **Fixes:** the shifted first sigma is clamped to 1.0 (`d3414a3b0`; does not arise at shifts 12/3); Regenerate's
  missing control fields (`61aef20b7`).

## Open upstream work

On [`../horizon.md`](../horizon.md): Pika parity (#9777), `keep_steps` reference release (#9762), reference isolation
(#9689), samplers (#10400, #10405), VAE tile batching bits (#10256), non-24 fps (#9398), the P-H3-Image request
(#9863), and the parity RFC (#7380).
