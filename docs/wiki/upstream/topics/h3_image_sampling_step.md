# MiniMax-H3-Image: the Euler step, the sigmas and the noise draws across implementations

last updated: 2026-11-17 (vllm-omni's step read; the fp32 sigma round trip shown to leave this grid unchanged, and the "moves by one ulp" line corrected; sglang's, vllm-omni's and LightX2V's draw orders and every implementation's generator device added; DiffSynth and LightX2V step rows added; record [`../sweeps/2026-11-17_h3_image.md`](../sweeps/2026-11-17_h3_image.md)); 2026-11-15 (page created from the 2026-11-15 sweep; same-seed `2048x2048` T2I + Edit Euler update test added the same day)

**Status: velocity vs. stock `euler` tested 2026-11-15 on `2048x2048` T2I + Edit (stock `euler` kept). The sigma
grid is the same in every implementation, bit for bit, as of 2026-11-17. The draw order still differs between
implementations, which matters only for cross-engine seed-exact reproduction.** Each row says what an implementation
does and where. The whole H3-Image pipeline, stage by stage, is on [`h3_image_pipeline.md`](h3_image_pipeline.md).

## The Euler step

| implementation | rank | form | where |
|---|---|---|---|
| sglang | 1 | velocity, float32: `x + (sigma - sigma_next) * v` | `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py::minimax_h3_image_denoise_loop` (`target_rows + sigma_deltas[step] * v.float()`) |
| vllm-omni | 2 | velocity, float32: `target + delta * velocity[n_cond:].float()` with `delta = sigma_k - sigma_{k+1}`, on the target rows only; its docstring keeps this form "because it is the released sampler's arithmetic" (read by hand 2026-11-17) | `coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3_image.py::_euler_target_step` |
| diffusers | 3 | velocity (`euler_update_form="velocity"`, written by the converter for H3-Image; video, Pika and P-H3 keep `blend`) | `coderef/diffusers/src/diffusers/schedulers/scheduling_minimax_h3.py` (`875373ac7`, #15230) |
| ComfyUI core | trusted least | stock `euler`: `d = (x - denoised) / sigma`, then `x + d * dt`, with `denoised` from the model's velocity | `comfy/k_diffusion/sampling.py::sample_euler`; `comfy/ldm/minimax/sampling.py::MiniMaxH3ImageModelSampling` |
| this pack | | core's `euler` through `KSamplerSelect` with `H3_IMAGE_MANUAL_SIGMAS` | `workflows/h3_config.py` |
| DiffSynth | periphery | velocity: `FlowMatchScheduler.step`, `sample + model_output * (sigma_next - sigma)`, on the pipeline's negated velocity, float32 latents (agent's reading, 2026-11-17) | `coderef/DiffSynth-Studio/diffsynth/diffusion/flow_match.py`, `diffsynth/pipelines/minimax_h3_image.py` |
| LightX2V | periphery | velocity, float32: `target_latents + delta * video_noise_pred[start:].float()`, target rows only (read by hand 2026-11-17) | `coderef/LightX2V/lightx2v/models/schedulers/minimax_h3_image/scheduler.py` |

Every implementation but core and this pack takes the velocity form. The forms are the same update in exact arithmetic and differ in `float32` rounding (and `bf16` input rounding in `calculate_denoised`).
`diffusers`' docstring claims the H3-Image heads "were distilled with the velocity form" and that low-bit rounding differences grow across steps. Measured at `seed=42` on `2048x2048` (`H3_IMAGE_UNET` + `IMAGE_GRAPH_VAE`), unconstrained T2I shows small micro-detail shifts (`PSNR = 29.24 dB`) while `<Picture 1>`-anchored Editing stays nearly identical (`PSNR = 49.95 dB`), with zero composition, sharpness, or visual quality gap.

## The sigmas

Every implementation reads the same six-step grid, and none applies a shift:
- The engines read it from the release's `pdd_config.json`: sglang validates it (`.../minimax_h3/time_request.py`),
  and so does vllm-omni (`pipeline_minimax_h3_image.py`). LightX2V's scheduler also takes it.
- DiffSynth hard-codes it (`FlowMatchScheduler.set_timesteps_minimax_h3_image`).
- Core reads it from the checkpoint (`MiniMaxH3ImageModelSampling.grid`).
- This pack writes it out (`H3_IMAGE_MANUAL_SIGMAS`).

The `pdd_config.json` in the release download (under `$H3_MODEL_STORE`) lists exactly `H3_IMAGE_MANUAL_SIGMAS`
(compared 2026-11-17).

The fp32 round trip:
- **Who does it:** diffusers (`modular_pipelines/minimax_h3_image/before_denoise.py`), DiffSynth, sglang
  (`time_request.py`), vllm-omni (`pipeline_minimax_h3_image.py`) and LightX2V (`scheduler.py`) build timesteps as
  `1 - sigma` in float32, then take sigma back as `1 - timestep`. sglang, vllm-omni and LightX2V were read by hand
  on 2026-11-17.
- **What it can do:** a round trip like this can move a sigma below 0.5 by one ulp.
- **What it does here:** nothing. On this grid it leaves every value unchanged, so `H3_IMAGE_MANUAL_SIGMAS` is what
  the engines step on, bit for bit. The command is in [`../sweeps/2026-11-17_h3_image.md`](../sweeps/2026-11-17_h3_image.md).
  *Corrected 2026-11-17: this section used to say the round trip "moves the sigmas below 0.5 by one ulp" and that
  our values were not compared.*

## The noise draws (one generator per request)

| implementation | order | generator device |
|---|---|---|
| sglang | target, then every reference's anchor noise, then every posterior (read by hand 2026-11-17: the header of `.../minimax_h3/image_noise.py`) | the platform device (CUDA) unless the request or the pipeline config sets `generator_device`; the release's own example requests set `cpu` |
| vllm-omni | the same as sglang (read by hand: the comment at the top of `forward`, `pipeline_minimax_h3_image.py`) | the same as sglang |
| diffusers | every posterior, then target, then every anchor (re-read 2026-11-17) | the caller's generator; its docs pass a CPU one, and `randn_tensor` then draws on CPU |
| this pack (`reference_encode.py::sample_release_references`) | sglang's order. The target and anchor draws only advance the stream; this graph's own target noise is `RandomNoise`'s, and the anchor noise is core's (below) | CPU |
| DiffSynth | target, then per reference: posterior, anchor | CPU (`rand_device` default) |
| LightX2V | every posterior, then target, then every anchor. Its comment says it follows Modular Diffusers (read by hand: `coderef/LightX2V/lightx2v/models/runners/minimax_h3_image/minimax_h3_image_runner.py::_sample_inputs`) | CPU |
| ComfyUI core | target from `RandomNoise`; each reference's posterior at fixed `IMAGE_ENCODE_SEED`; anchor noise from a stream restarted at the sampler's seed for every reference (`comfy/ldm/minimax/model.py::_cond_video_rows`) | CPU |

Three facts follow from the table:
- **The ranked engines agree** (sglang and vllm-omni), and this pack's draw follows them.
- **Seed-exact reproduction of an edit is not available here,** against any engine. The pack's anchor noise is still
  core's restarted stream, not a draw from the request's generator, so two references of one shape carry the same
  anchor noise.
- **A text-to-image render at a given seed** draws its target noise first on CPU in every implementation, so with
  `generator_device=cpu` it starts from the same noise everywhere.

## The AdaLN time embedding

DiffSynth (`coderef/DiffSynth-Studio/diffsynth/models/minimax_h3_image_dit.py::MiniMaxH3ImageAdalnProj`) and diffusers (`coderef/diffusers/src/diffusers/models/transformers/transformer_minimax_h3_image.py`) run
SiLU on the float32 time embedding and cast afterwards. Core's non-curve branch casts first
(`comfy/ldm/minimax/model.py::_time_embedding`, `self.time_embedder(t_vals).to(dtype)`). The owner's
`H3_IMAGE_UNET` is a pruned file on core's curve branch, so only `H3_IMAGE_UNET_BF16` would meet the difference
(reasoned from the code). Release issue #85 traces 17-frame brightness dips on unpruned video files to the same cast
([`../horizon.md`](../horizon.md)).

## Same-seed `2048x2048` test (2026-11-15)

Direct `float32` velocity Euler (`x_f32 + (sigma_next - sigma) * v_f32`) vs. ComfyUI stock `sample_euler` (`d = (x - denoised) / sigma; x + d * dt`), run in-process on `seed=42`, `H3_IMAGE_MANUAL_SIGMAS`, `H3_IMAGE_UNET`, and `IMAGE_GRAPH_VAE` (`p-h3-image_vae_int8_convrot.safetensors`; output grid at `$H3_OUTPUT_DIR/Image/nov15_lab/exp4c_euler_step_ab/exp4c_euler_step_ab.png`):

- **Edit (`2048x2048`, `<Picture 1>`)**: `latent_l2_diff = 4.021` (`max_abs = 0.176`), decoded `PSNR = 49.95 dB`, `RGB MAE = 0.345 / 255` (`4.70 s` vs. `4.71 s`) — visually indistinguishable when anchored by `<Picture 1>`.
- **T2I (`2048x2048`)**: `latent_l2_diff = 61.883` (`max_abs = 2.636`), decoded `PSNR = 29.24 dB`, `RGB MAE = 3.683 / 255` (`3.05 s` vs. `3.05 s`) — 6-step amplification of `float32`/`bf16` rounding shifts fine bokeh highlights and ink stroke phase slightly while keeping identical composition, sharpness, and quality.
- **Outcome**: Keep stock `euler` in `workflows/` and `app/`; no custom velocity-Euler sampler node is needed unless chasing bit-exact cross-engine T2I reproduction against `sglang`/`diffusers`.
