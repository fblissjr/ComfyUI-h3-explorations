# MiniMax-H3-Image: the Euler step, the sigmas and the noise draws across implementations

last updated: 2026-11-15 (page created from the 2026-11-15 sweep)

**Status: open, untested here.** Nothing in the pack was changed. Each row says what an implementation does and
where. Which differences show in a picture is a render's question.

## The Euler step

| implementation | rank | form | where |
|---|---|---|---|
| sglang | 1 | velocity, float32: `x + (sigma - sigma_next) * v` | `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py::minimax_h3_image_denoise_loop` (`target_rows + sigma_deltas[step] * v.float()`) |
| vllm-omni | 2 | not read for this page | `coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3_image.py` |
| diffusers | 3 | velocity (`euler_update_form="velocity"`, written by the converter for H3-Image; video, Pika and P-H3 keep `blend`) | `coderef/diffusers/src/diffusers/schedulers/scheduling_minimax_h3.py` (`875373ac7`, #15230) |
| ComfyUI core | trusted least | stock `euler`: `d = (x - denoised) / sigma`, then `x + d * dt`, with `denoised` from the model's velocity | `comfy/k_diffusion/sampling.py::sample_euler`; `comfy/ldm/minimax/sampling.py::MiniMaxH3ImageModelSampling` |
| this pack | | core's `euler` through `KSamplerSelect` with `H3_IMAGE_MANUAL_SIGMAS` | `workflows/h3_config.py` |

The forms are the same update in exact arithmetic and differ in float32 rounding. diffusers' docstring claims the
H3-Image heads "were distilled with the velocity form", and that the bf16 model amplifies a last-bit difference
into a visibly different image. That is their claim, and it was not tested here.

## The sigmas

Both diffusers (`modular_pipelines/minimax_h3_image/before_denoise.py`) and DiffSynth
(`FlowMatchScheduler.set_timesteps_minimax_h3_image`) build timesteps as `1 - sigma` in float32, then take sigma back
as `1 - timestep`. That moves the sigmas below 0.5 by one ulp, "as the release does". Our `H3_IMAGE_MANUAL_SIGMAS`
values were not compared against that.

## The noise draws for edit references (one CPU generator)

| implementation | order |
|---|---|
| this pack (`reference_encode.py::sample_release_references`) | target, then every reference's anchor noise, then every posterior |
| DiffSynth | target, then per reference: posterior, anchor |
| diffusers | every posterior, then target, then every anchor |

This matters only for seed-exact reproduction against one of them. sglang's order and the release's were not read
for this page.

## The AdaLN time embedding

DiffSynth (`coderef/DiffSynth-Studio/diffsynth/models/minimax_h3_image_dit.py::MiniMaxH3ImageAdalnProj`) and diffusers (`coderef/diffusers/src/diffusers/models/transformers/transformer_minimax_h3_image.py`) run
SiLU on the float32 time embedding and cast afterwards. Core's non-curve branch casts first
(`comfy/ldm/minimax/model.py::_time_embedding`, `self.time_embedder(t_vals).to(dtype)`). The owner's
`H3_IMAGE_UNET` is a pruned file on core's curve branch, so only `H3_IMAGE_UNET_BF16` would meet the difference
(reasoned from the code). Release issue #85 traces 17-frame brightness dips on unpruned video files to the same cast
([`../horizon.md`](../horizon.md)).

## What would settle it

A velocity-form Euler sampler (a small `SAMPLER` node, or a `ManualSigmas` + custom step) against stock `euler` at one
seed on one t2i and one edit prompt, stacked for the owner.
