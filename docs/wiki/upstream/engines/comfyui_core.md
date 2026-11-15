# ComfyUI core: what the pack runs on (trusted least as evidence)

last updated: 2026-11-15 (page created; state as of [`../sweeps/2026-11-15.md`](../sweeps/2026-11-15.md))

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
  followed. Its docstring cites `resolve_cast_module_with_vbar`, whose relayout moved into
  `model_prefetch.pin_modules`, so the docstring needs a re-read.
- `sol_attn_h3.py` uses the `("double_block", i)` patch contract, unchanged.
- `reference_encode.py` imports `IMAGE_ENCODE_SEED`, renamed from `PH3_IMAGE_ENCODE_SEED` in `ede4e5fe8`, and already
  uses the new name.
- Stale line citations in `ph3_image_nodes.py` (three, listed in the sweep record). They are comment-only and left for
  a worktree edit.

## State as of 2026-11-15

- **Its own MiniMax-H3-Image path** (`ede4e5fe8`, #17655): `MiniMaxH3ImageModel`, `MiniMaxH3ImageModelSampling` on
  the PDD grid, `H3ImageVAE`, and the node `MiniMaxH3ImageConditioning`.
- **References.** Never enlarged ([`../topics/reference_sizing.md`](../topics/reference_sizing.md)).
- **Time embedding.** The non-curve branch casts before SiLU. That is the cause release issue #85 traces brightness
  dips to on unpruned files. The pack's shipped graphs load pruned files (curve branch)
  ([`../topics/h3_image_sampling_step.md`](../topics/h3_image_sampling_step.md)).
- **Encoder.** `CombinedLinear` fusion and video input for Qwen 3.x
  ([`../topics/qwen3vl_encoder.md`](../topics/qwen3vl_encoder.md)).

## Open upstream work

On [`../horizon.md`](../horizon.md):
- #17715, the token refiner run on every step. If it merges, `lora_branch.py`, `nodes.py` and the control-vector seam
  need a re-read.
- #17705 / #17692, the full time-embedder layout warning.
- #17649, the encoder reference cycle.
- #17701, the malloc-graph arena.
- #16712, mask-driven packing.
