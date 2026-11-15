# sglang: the vendor's serving path (rank 1)

last updated: 2026-11-15 (page created; state as of [`../sweeps/2026-11-15.md`](../sweeps/2026-11-15.md))

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

## Open upstream work

On [`../horizon.md`](../horizon.md): Open-H3-Regenerate-2K (#47668), the AdaLN LoRA dtype on pruned files (#50611),
and the Q4 roadmap with RTX 4090 defaults (#43313).
