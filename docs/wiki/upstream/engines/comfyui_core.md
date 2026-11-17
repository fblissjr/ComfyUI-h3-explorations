# ComfyUI core: what the pack runs on (trusted least as evidence)

last updated: 2026-11-17 (MiniMax-H3-Image path read against the engines, [`../sweeps/2026-11-17_h3_image.md`](../sweeps/2026-11-17_h3_image.md); `bench/check_reference_encode.py` passed at HEAD); 2026-11-15 (page created; state as of [`../sweeps/2026-11-15.md`](../sweeps/2026-11-15.md))

**The owner trusts core least as evidence of how H3 should work (2026-11-15).** It is watched more closely than
anything else, though, because the pack runs on it, copies parts of it and patches it at runtime. A change in a
function the pack copies or pins is the most urgent kind of finding here. Where core and sglang differ, sglang is the
reference. The GitHub repo is `Comfy-Org/ComfyUI`, and the checkout is `../..`.

## Where its H3 code lives (ComfyUI root)

| area | path |
|---|---|
| DiT, packing, sampling, VAEs | `comfy/ldm/minimax/` (`model.py`, `sampling.py`, `vae.py`) |
| nodes | `comfy_extras/nodes_minimax_h3.py` |
| encoder and tokenizer | `comfy/text_encoders/minimax.py`, `llama.py`, `qwen_vl.py`, `qwen35.py` |
| model detection and VAE choice | `comfy/model_detection.py`, `comfy/sd.py` |
| quantized ops and dynamic VRAM | `comfy/ops.py`, `comfy/model_patcher.py`, `comfy/model_management.py` |

## What the pack depends on, and what holds it

- `bench/check_reference_encode.py` pins core's encoder functions that `reference_encode.py` copies. It passed at
  HEAD on 2026-11-15. `MLP.forward` changed (`48b090cb6`) and is not pinned.
- `lora_branch.py` follows core's `FinalLayer` / `_pdd_head` signatures, which changed in `ede4e5fe8` and are already
  followed. Re-read on 2026-11-17: `comfy/ops.py::resolve_cast_module_with_vbar` still performs `requantize_from_float`
  in `post_cast` (so `lora_branch.py`'s line-5 docstring reference remains accurate; only `_v_relayout` moved into
  `comfy/model_prefetch.py::pin_modules`).
- `sol_attn_h3.py` uses the `("double_block", i)` patch contract, unchanged.
- `reference_encode.py` imports `IMAGE_ENCODE_SEED`, renamed from `PH3_IMAGE_ENCODE_SEED` in `ede4e5fe8`, and already
  uses the new name.
- `ph3_image_nodes.py` citations (`lines 87, 290, 344` for `comfy_extras/nodes_minimax_h3.py`,
  `comfy/text_encoders/qwen35.py`, and `comfy/ldm/minimax/vae.py::H3ImageVAE.encode`) were checked at HEAD on 2026-11-17
  and are already current.

## State as of 2026-11-15

- **Its own MiniMax-H3-Image path** (`ede4e5fe8`, #17655): `MiniMaxH3ImageModel`, `MiniMaxH3ImageModelSampling` on
  the PDD grid, `H3ImageVAE`, and the node `MiniMaxH3ImageConditioning`.
- **References.** Never enlarged ([`../topics/reference_sizing.md`](../topics/reference_sizing.md)).
- **Time embedding.** The non-curve branch casts before SiLU. That is the cause release issue #85 traces brightness
  dips to on unpruned files. The pack's shipped graphs load pruned files (curve branch)
  ([`../topics/h3_image_sampling_step.md`](../topics/h3_image_sampling_step.md)).
- **Encoder.** `CombinedLinear` fusion and video input for Qwen 3.x
  ([`../topics/qwen3vl_encoder.md`](../topics/qwen3vl_encoder.md)).

## MiniMax-H3-Image against the engines, 2026-11-17

Read by hand at HEAD on 2026-11-17. Every implementation is compared on
[`../topics/h3_image_pipeline.md`](../topics/h3_image_pipeline.md). `bench/check_reference_encode.py` passed at that
HEAD.

**Core agrees with the engines on everything the model is shown:**
- **The tokens.** Text to image sends the prompt alone (`comfy/text_encoders/minimax.py`). An edit sends
  `Picture N: ` and a vision block per picture, then the prompt.
- **The tags.** A vision block is tagged as image, its markers included (`token_tags_from_embeds_info`).
- **The packed layout and its positions** (`comfy/ldm/minimax/model.py::PackedLayout`).
- **The reference level** (`VISUAL_COND_TIMESTEP`).
- **The grid and the heads.** The grid is read from the checkpoint (`MiniMaxH3ImageModelSampling`), and the heads
  run in float32 after the final norm and AdaLN (`FinalLayer._modulate`).
- **The VAE tiles,** at the release's tile settings.

**Where core is the outlier:**
- **An in-bounds reference is resampled to the 32 px grid;** the engines crop it (`comfy_extras/nodes_minimax_h3.py::_image_refs`).
- **The reference posterior is drawn at a fixed seed and not rounded to fp16** (`comfy/ldm/minimax/vae.py::H3ImageVAE.encode`,
  `IMAGE_ENCODE_SEED`).
- **The reference encode runs at whatever dtype the VAE was loaded at.** `comfy/sd.py` casts the image VAE to its
  working dtype on load; the engines encode in fp32.
- **The anchor noise restarts one stream at the sampler's seed for every reference** (`_cond_video_rows`). The
  engines take one draw per reference from the request's generator.
- **The edit canvas defaults to image_1's VAE-copy size, with no floor and no resize back.** The engines take the
  size from the request, raise it to a minimum generation area, and resize the decode back.
- **The step is stock `euler`** (the denoised form); the engines use the velocity form. Tested and kept in this pack
  ([`../topics/h3_image_sampling_step.md`](../topics/h3_image_sampling_step.md)).
- **An empty prompt is encoded as one pad token, and the run goes ahead.** Every engine refuses one. No text segment
  is cut; the engines cut each at a fixed token count.
- **A text-encoder copy whose short edge is already `IMAGE_REF_SHORT_EDGE` is resampled to the grid anyway.** When
  the copy is off the grid, core's own preprocessor resizes with bilinear (`comfy/text_encoders/qwen_vl.py::process_qwen2vl_images`),
  where the engines' HF processor resizes with bicubic.

**What this pack takes from core and what it replaces:**
- **Text to image** runs core's `MiniMaxH3ImageConditioning`.
- **An edit** replaces core's node with `MiniMaxH3EncodeImageReferences` and `MiniMaxH3PromptOnImageReferences`.
  These match the engines on the crop, the draw, the fp16 rounding, the fp32 encode, and (since 2026-11-17,
  `reference_encode.py::_snap_h3_image_qwen_grid`) 8-bit bicubic snapping of off-grid Qwen copies
  ([`../../image/h3_image.md`](../../image/h3_image.md), "Edit references, point by point").
- **The anchor stream and the step** stay core's.

## Open upstream work

On [`../horizon.md`](../horizon.md):
- #17715, the token refiner run on every step. If it merges, `lora_branch.py`, `nodes.py` and the control-vector seam
  need a re-read.
- #17705 / #17692, the full time-embedder layout warning.
- #17649, the encoder reference cycle.
- #17701, the malloc-graph arena.
- #16712, mask-driven packing.
