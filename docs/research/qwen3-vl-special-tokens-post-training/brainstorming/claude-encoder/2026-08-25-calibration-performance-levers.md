# Performance levers for the v2 calibration run, ranked by what is measured

**Date:** 2026-08-25
**Status:** Brainstorming for the lane executing the plan. Not authority; no
lever here is authorised, and none may be adopted without the numerical
acceptance the canonical plan already requires for its axis.
**Substrate:** torch `2.13.0+cu132`, cuDNN 92000, RTX 4090 (sm_89), the
pinned `llm-compressor` checkout at `8357e94`. Evidence classes follow
[`canonical/README.md`](../../canonical/README.md).

## Where the cost actually is

**MEASURED**, from the accepted Gate 2A record
([`2026-08-25-gate2a-corrected-floor.md`](2026-08-25-gate2a-corrected-floor.md)):

- **Weight movement dominates wall-clock.** Every sequential step onloads
  every layer once per subgraph. At FP32 storage that is about 125 GB per
  step, part of it from the disk tier. The identical 1,100-token step took
  64.5 s, 61.9 s and 213.4 s across three processes with bit-identical CUDA
  figures; the forwards were the same, the host I/O was not.
- **One attention op governs the GPU boundary.** Grouped-query attention at
  FP32 dispatches `aten::_scaled_dot_product_attention_math` and materialises
  the `[64, L, L]` logit tensor: a 19.4 GiB transient on a 5,857-token row,
  a 19.23 GiB allocator request on the 8,981-token row.

**SOURCE**, `src/llmcompressor/modifiers/transform/awq/base.py` at the
pinned commit, which is what turns the floor into the Gate 2B cost:

- `_apply_smoothing` runs, for every resolved mapping in every layer, one
  unquantised forward of the mapping's *parent module* over every cached
  batch (`_run_samples`), then `_compute_best_scale` runs the same parent
  forward once per grid point. `n_grid` defaults to 20; `duo_scaling="both"`
  splits it into 10 + 10. For the q/k/v mapping the parent is the attention
  block, so the math-backend attention on each row runs roughly 21 times per
  decoder layer in the AWQ pass. Gate 2A's floor is two forwards per layer
  per row. **INFERENCE:** the AWQ increment on attention time is therefore
  about an order of magnitude, and every one of those calls re-allocates the
  logit tensor.
- Cached parent inputs live where `offload_device` says. The default is
  `None` (on the accelerator) except for MoE; the Gate 2B contract places it
  on CPU. The modifier pins that cache before each grid search, so the
  per-grid-point onload is the pinned-transfer cost, not a pageable one.

## Levers, in order of payoff

| # | lever | what it buys | what it costs | status |
|---|---|---|---|---|
| 1 | **BF16 storage, FP32 active compute** | halves bytes moved per step; the whole model fits in host RAM, so the disk tier and its timing variance go away; returns the host memory a modifier needs | none: **MEASURED** bit-identical to FP32-stored `comfy_exact` at layer 49 on all four Gate 1B fixtures (relative L2 0.0); loads through the bridge in 0.2 s with 62 GiB on CPU and `MemAvailable` holding at 119 GiB against 5 GiB under FP32 storage | **closed by measurement**, commits `471ab2d` and `1c8fe43`; `bench/results/2026-08-25_storage_axis_layer49_*.json`, `2026-08-25_gate2a_primary_bf16_store.json` |
| 2 | **expanded-KV memory-efficient attention** | removes the quadratic allocation; because of the AWQ multiplier above it is the dominant *time* lever for Gate 2B, not only the memory lever | about 0.6 GiB more resident K/V per layer call at the longest row (NOMINAL); changes the attention arithmetic `comfy_exact` was accepted with | **MEASURED feasibility, acceptance pending.** The composed path (BF16 storage plus expanded KV) completes the entire five-row primary population, 25,250 tokens with a 10,358-token row, at 6.4 GiB allocated / 7.7 GiB reserved, and the 8,981-token stress row at 5.7 GiB, where grouped-query math OOMed at 22.5 GiB on the fourth row (`2026-08-25_gate2a_primary_bf16_store_expanded_kv.json`, `_stress_bf16_store_expanded_kv.json`). **Not numerically free:** grouped-query math versus expanded-KV math at layer 49 differs by 4.4e-4 (vision rows) on the multi-image fixture and 3.4e-3 on keyframe-only, the band of the accepted `comfy_exact` residual, but by 8 to 10 percent on the two fixtures Gate 1B already flagged as sensitive (single 44x40, mixed), where every kernel arm scatters by that much around deployed ComfyUI. Expanded-efficient versus ComfyUI: 3.5e-4 / 2.9e-3 / 0.090 / 0.058 across the four, against accepted grouped-math residuals of 4.7e-4 / 2.9e-3 / 0.059 / 0.051 (`2026-08-25_kernel_axis_{expansion,kernel,vs_comfy}_*.json`). **The early-tap control settles which:** at decoder layer 24, past the DeepStack injections, the three arms (grouped math / expanded math / expanded efficient) sit at 1.0e-3 / 1.4e-3 / 1.0e-3 (single) and 1.2e-3 / 2.0e-3 / 1.2e-3 (mixed) from deployed ComfyUI on vision rows, expanded-efficient indistinguishable from grouped math; by layer 49 the same arms are at percent level with their order changed. The percent-level spread on the sensitive fixtures is compounding through depth, not a kernel ranking (`2026-08-25_kernel_axis_tap24_{single_image,mixed_keyframe_reference}.json`; write-up [`2026-08-25-gate2b-prerequisites.md`](2026-08-25-gate2b-prerequisites.md)). Instrument caveat: tap 0 is not a valid vision-row comparison point, since Transformers' hook reads layer 0 before the DeepStack injection and ComfyUI's truncated stack reads after it; any early tap must sit past layer 2. Acceptance runs through Codex on those numbers, not on feasibility |
| 3 | **`sequential_prefetch=True`** (`DatasetArguments`, default off) | overlaps the next batch's onload with the current forward, in the pipeline and in AWQ's `_run_samples` | two batches resident on device; the per-row cache is 46 to 74 KB per token, so the overlap is small | untested here; try once numerics are settled |
| 4 | **`n_grid`** | linear in AWQ runtime | changes the chosen scales, therefore the result | a last-resort resource lever, not a free one |
| 5 | **`sequential_targets_per_subgraph > 1`** | fewer cache round-trips per row | more resident weights on the axis that is already binding | not recommended on this card |
| 6 | **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** | avoids fragmentation OOMs in a pass with many transient allocations | nothing | harmless, but both Gate 2A OOMs showed 188 MiB reserved-but-unallocated, so fragmentation was not the failure and this does not move the boundary |

## What does not apply, and why

- **`torch.compile`.** llm-compressor's `enable_compile` compiles only the
  MSE observer's inner loop (`observers/mse_quant.py`); it never touches the
  model. Compiling the model itself is not viable on this path: the
  subgraphs are fx-traced and executed as generated Python under
  compressed-tensors offload hooks and the storage policy's functional-level
  precision patches, and every H3 row has a different sequence length, so
  recompilation would eat any gain on a run whose cost is bandwidth and one
  large attention op.
- **TF32 matmul.** Off on both sides, so it is consistent and not a lever:
  ComfyUI sets no matmul precision flag, and the calibration venv reports
  `allow_tf32=False`, `float32_matmul_precision=highest`. Turning it on
  would roughly double matmul throughput and silently change the accepted
  `comfy_exact` arithmetic against deployed ComfyUI.
- **TF32 convolution, a note for the precision lane.** `cudnn.allow_tf32`
  is True by default in both processes, and both vision patch embeds are
  `Conv3d` (`comfy/text_encoders/qwen35.py`, `ops.Conv3d`; Transformers'
  patch embed). So the first op of the tower runs TF32 under both
  implementations. Consistent, so nothing to fix, but "FP32 active compute"
  is not literally FP32 at that op and the policy record should say so.
  SOURCE, not measured.
- **Flash and cuDNN attention.** "No available kernel" at FP32 for every
  real shape probed, forced or automatic. They are fp16/bf16 paths; reaching
  them is a BF16-attention arm, a further precision deviation rather than a
  free speedup.
- **Torch 2.13 / CUDA 13.2 / sm_89 specifics.** Not verified from release
  notes here and not asserted. What is measured on this build and card: no
  fused FP32 grouped-query kernel exists; the memory-efficient kernel
  handles FP32 non-GQA shapes up to 16,512 patches; cuDNN SDPA is
  unavailable at FP32 at every shape.

## Update, same day, from the storage/kernel lane

The lane executing the plan checked the three items above against source and
its own runs and confirmed them; the table now carries its measurements. One
nuance on the TF32 note: under `comfy_exact_bf16_store` the patch-embed conv
weight is kept FP32 at load, so the functional conv3d patch sees it as already
FP32 and never casts it; the TF32 point stands regardless, and the lane is
adding it to the storage-policy record.

**Owner decision, later the same day (`f2f6d5a`).** Reference stills now
calibrate at the vendor's 2048 short edge with upstream upscaling, as the
primary policy. The 8,981-token "stress" row is therefore a typical row of
the population, not a stress case, and a 16:9 still is 7,296 visual tokens:
levers 1 and 2 above are the condition for the run to exist at all, not
optimisations, and the population that fits a token budget shrinks
accordingly.

## Bounds

- No lever was executed for this note. The peer's process held the card
  while it was written; every number above is from the Gate 2A record or a
  source read of the pinned library and the installed ComfyUI.
- Lever 2's time claim is an inference from the grid-search structure and
  the measured per-call cost; the AWQ increment itself is Gate 2B's to
  measure, and this note must not be read as pre-empting it.
