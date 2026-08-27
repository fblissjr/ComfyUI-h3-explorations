# What sglang, vllm, flashinfer and nanobind have that we do not, for the encoder

**Date:** 2026-08-25
**Status:** Brainstorming survey for the lane executing the v2 plan. Not
authority; nothing here is authorised. Companion to
[`2026-08-25-calibration-performance-levers.md`](2026-08-25-calibration-performance-levers.md).
**Method:** three read-only source surveys of `coderef/sglang`,
`coderef/vllm` (at `bc11ecaf4e`) and `coderef/flashinfer` (0.6.18), plus
direct checks of `coderef/comfy-kitchen-sol` against the installed wheel and
the shipped graphs. Everything is SOURCE with a path unless marked MEASURED;
nothing was benchmarked. The DiT-side sglang comparison (CUDA graphs, step
caching, AdaLN cache, sequence parallel) is already priced in
[`sglang_comparison.md`](../../../sglang_comparison.md) and is not repeated.

## Verdicts up front

| stack | for the v2 calibration run | for serving the W4 artifact in ComfyUI |
|---|---|---|
| sglang | nothing to borrow; one framing fact (below) | nothing: bf16 end to end, NVFP4 by dequant, no AWQ kernel |
| vllm | one candidate kernel arm for FP32 grouped-query attention (Triton / FlexAttention), unmeasured | one candidate experiment: Marlin W4A16 versus kitchen's dequant-plus-cuBLAS at H3's prefill M, unmeasured |
| flashinfer | **none**: no FP32 input dtype exists in any attention path, and vision head_dim 72 is silently unsupported | FA2 on Ada with no Ada CI runner; nothing for W4A16 |
| nanobind | nothing: it is the binding layer kitchen already builds with | same |
| comfy-kitchen-sol | AWQ op identical to upstream; no interaction with calibration | two integration items for v2, both about Sol's sink (below) |

## 1. sglang: the vendor's own encoder path is bf16, and that is the finding

`coderef/sglang/python/sglang/multimodal_gen/runtime/models/encoders/minimax_h3_qwen3vl.py` wraps sglang's own
`Qwen3VLModel`, not Transformers. Precision is declared once:
`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py:67` `text_encoder_precisions =
("bf16",)`; the vision tower has no separate knob and casts its input to its
patch-embed dtype; pixel values are cast to bf16 at the stage boundary; the
output is cast to bf16 and shape-checked `[L, 5120]`. Only 50 decoder layers
are constructed, layers 50 and up and `lm_head` are never materialised, and
the final norm is replaced by `nn.Identity()` with the comment "H3 and
ClipProj consume an unnormalized intermediate residual stream". Language
attention is sglang's `LocalAttention` constrained to FlashAttention or torch
SDPA, causal, grouped-query heads taken natively (SDPA gets `enable_gqa=True`);
the all-ones mask is built but consumed only by M-RoPE index construction and
never reaches the kernel. No `torch.compile`, no CUDA graph, no cross-request
embedding cache; one complete request per encoder copy with no padding,
deliberately, because "H3 presentations have variable multimodal layouts".

**Consequence for the precision lane.** The FP32 active compute that
`comfy_exact` reproduces is ComfyUI's manual-cast choice. MiniMax's own
serving stack runs the encoder in bf16 with FP32 accumulation inside the
attention kernel. That does not change the accepted policy, whose target is
deployed ComfyUI, but it removes any idea that FP32 attention has a vendor
authority behind it: the vendor never ran it.

Two mechanisms transfer:

- **Checkpoint stays mmapped when a host copy would not fit**
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/loader/component_loaders/text_encoder_loader.py:913-923`, `_keep_this_checkpoint_mapped`; the
  comment names the 62.13 GiB H3 encoder). Parameters point at the mapped
  weights with no copy. The Gate 2A load peak of 122 GiB RSS was the opposite
  arrangement; the storage-axis work has since reached 0.2 s loads with 62
  GiB on CPU, which is this mechanism by another route.
- **Vision packing**: all images and video blocks in one `cu_seqlens` varlen
  batch with the bounds cached host-side to avoid a device sync per block
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/encoders/qwen3vl_vision.py:179-183`).

Quantised encoders exist in sglang (FP8, kitchen int8 / W4A8 / W4A4, Comfy
`nvfp4`, GGUF, Quanto int8) but every path except Quanto keeps the vision
tower bf16, the NVFP4 path dequantises each active linear to a bf16 matmul,
and there is no AWQ or GPTQ kernel: "NVFP4-AWQ" is NVFP4 with AWQ input
pre-scales. `quality="high"` requires an unquantised encoder.

## 2. vllm: Marlin for the artifact, Triton for FP32 attention, no calibration

### Serving the W4 artifact

For our exact scheme (compressed-tensors pack-quantized, group 128, symmetric
4-bit, bf16 activations) vllm's selection on sm_89 is deterministic:
CUTLASS W4A8 and Machete are rejected at compute capability 90; AllSpark
rejects group-128 int4 below sm_90; **Marlin is selected**
(`coderef/vllm/vllm/model_executor/kernels/linear/mixed_precision/marlin.py:35-101`). Facts that matter:

- Marlin is a fused dequant-GEMM at every M; it tiles M in 64-row blocks with
  no fallback (`coderef/vllm/csrc/libtorch_stable/quantization/marlin/marlin.cu:423-438`). Kitchen's CUDA
  `gemv_awq_w4a16`, by its own tiering, is a naive GEMV at M ≤ 8, a fused
  int4×bf16 MMA at 8 < M ≤ 512, and **dequant plus cuBLAS bf16 above M =
  512**. **MEASURED:** the installed `0.2.31+sol.23d1a66` registry selects the
  CUDA backend for that op at M = 1, 64, 1,100 and 5,857, so every H3 encode
  lands in the dequant-plus-cuBLAS tier.
- Whether Marlin beats that at M = 1k to 10k on sm_89 is unmeasured
  anywhere in vllm's tree. vllm's own AllSpark W8A16 kernel switches to
  dequant plus cuBLAS above M = 1024 (`coderef/vllm/vllm/model_executor/layers/quantization/utils/allspark_utils.py:9`), which is the
  same regime and hints the fused advantage fades there.
- The ops are plain `torch.ops` in `vllm._C_stable_libtorch`
  (`marlin_gemm`, `gptq_marlin_repack`), callable without the engine, built
  against torch 2.13.0 with 8.9 resolving to `8.0+PTX`. The
  compressed-tensors to Marlin repack is a transpose plus a permutation with
  a pure-torch reference that vllm's tests assert bit-identical to the CUDA
  repack (`coderef/vllm/vllm/model_executor/layers/quantization/utils/marlin_utils_test.py:33-125`, `coderef/vllm/tests/kernels/quantization/test_marlin_gemm.py:248-265`), so
  it can be done once offline; only the workspace depends on the device.
- An Ada-specific **W4A8-FP8 Marlin** variant exists
  (`VLLM_MARLIN_INPUT_DTYPE=fp8`) and is documented as fast only on
  SM89/SM12x (`coderef/vllm/vllm/model_executor/layers/quantization/utils/marlin_utils.py:666-679`, `coderef/vllm/CMakeLists.txt:624-631`). It
  quantises activations per token to fp8, which is a second numerical change.
- **No vllm W4A16 kernel accepts FP32 activations** (`coderef/vllm/csrc/libtorch_stable/quantization/marlin/marlin.cu:37-40`
  static-asserts half or bf16; Exllama fp16 only; Triton W4A16 fp16/bf16).
  The config-time gate would let an fp32 run through to the C++ failure.

Our loader already casts to the kernel dtype across the quantised matmul
(`h3_awq_encoder.py::H3AWQOperations.Linear._forward`), so Marlin would slot
into the same place. That makes it an experiment, not a plan change: a
kitchen-versus-Marlin sweep at M in {512, 1,100, 5,857, 10,358} on real H3
shapes, with the numerical delta reported beside the time.

### FP32 grouped-query attention, a third candidate arm

vllm lists `torch.float32` in the supported dtypes of `TRITON_ATTN`
(`coderef/vllm/vllm/v1/attention/backends/triton_attn.py:289-293`, any compute capability) and
`FLEX_ATTENTION` (`coderef/vllm/vllm/v1/attention/backends/flex_attention.py:91-95`), both with native grouped-query
heads. FlashAttention (FA2 on Ada) and FlashInfer are fp16/bf16 only. So an
FP32 fused kernel with no KV expansion and no materialised logits exists,
which torch SDPA does not offer for the grouped-query shape. It is a candidate
for the kernel matrix beside expanded-KV efficient attention, with two
cautions: the explorer found no test asserting fp32 correctness for the
Triton path, and its numerics against deployed ComfyUI are unmeasured. Note
FlexAttention is in torch itself, so the arm would not need vllm.

### Precision notes worth carrying

- vllm executes the Conv3d patch embed as `unfold` plus `F.linear` when the
  kernel equals the stride (`coderef/vllm/vllm/model_executor/layers/conv.py:77-81`, `:167-200`). With
  `VLLM_FLOAT32_MATMUL_PRECISION` defaulting to `highest` that is true FP32
  where ComfyUI's `ops.Conv3d` runs a cuDNN TF32 convolution. Same weights,
  different arithmetic at the tower's first op.
- vllm forces `torch.set_float32_matmul_precision("highest")` by default
  (`coderef/vllm/vllm/v1/worker/gpu_worker.py:161-162`), the same setting our two processes
  already have.
- No AWQ calibration exists in vllm; it defers to llm-compressor.

## 3. flashinfer: nothing, for a precise reason

- **No FP32 input dtype anywhere.** `coderef/flashinfer/flashinfer/jit/utils.py:33-44`'s
  `dtype_map` has no `torch.float32` entry, so the JIT raises `KeyError`
  before nvcc runs; the FA2 kernel `static_assert(sizeof(DTypeQ) == 2)` at
  `coderef/flashinfer/include/flashinfer/attention/prefill.cuh:2192`, `:2809`, `:3600`;
  `FLASHINFER_ENABLE_F32` is defined nowhere in the build. The best it offers
  is bf16 in with FP32 accumulation, which is less input precision than the
  SDPA math backend we are trying to replace.
- **Vision head_dim 72 is unsupported and not rejected**: kernel traits round
  `72 / 16 * 16` to 64 while the generated stride setters keep 72, and
  `coderef/flashinfer/include/flashinfer/utils.cuh:449-452` records that neither `plan()` nor
  the JIT validates head dims. The suite never tests 72.
- On sm_89 everything resolves to the FA2 template; FA3 is sm_90, the cutlass
  and cute-dsl paths are Blackwell. There is **no Ada CI runner**
  (`coderef/flashinfer/CONTRIBUTING.md:102-114`), so "supported" means the arch flag is emitted.
- Not installed in either venv; the `egg-info` in the tree is a stale 0.5.3
  build; declared CUDA support is 12.9 / 13.0 / 13.4, and cu132 would resolve
  to the cu130 jit-cache wheel, ABI unverified.
- No int4 AWQ GEMM is exposed (the TRT-LLM `fpA_intB` kernels are vendored
  with no binding); `mm_bf16_fp4` is NVFP4 on Blackwell only; RoPE and norm
  kernels are fp16/bf16 in, and there is no M-RoPE kernel at all.

## 4. nanobind: the binding layer, already in use

comfy-kitchen's CUDA and HIP backends bind through nanobind; every kitchen op
call, `gemv_awq_w4a16` included, passes tensors through `_wrap_for_dlpack`
(`coderef/comfy-kitchen-sol/comfy_kitchen/backends/cuda/__init__.py:491-506`, a per-tensor `__dlpack__(stream=-1)`
export working around a PyTorch sync issue). That is microseconds per call
against a workload of one large attention op and gigabytes of weight movement
per layer. `nanobind-backend` is its split-mode runtime with no user-facing
API. The sol fork's newest commit (`bd3fc78 nanobind`) touches only the wheel
workflow. If the lane ever writes a fused kernel, this is the toolchain it
would use; that is a build convenience, not a lever.

## 5. comfy-kitchen-sol: the AWQ op is untouched; Sol's sink is the interaction

**MEASURED, on the checkouts and the installed wheel:**

- `coderef/comfy-kitchen-sol/comfy_kitchen/tensor/awq_w4a16.py` and `coderef/comfy-kitchen-sol/comfy_kitchen/backends/eager/awq.py` are
  byte-identical between `comfy-kitchen-sol` and upstream `comfy-kitchen`;
  the CUDA backend diff contains only `sol_attn` additions. The installed
  `0.2.31+sol.23d1a66` serves both `gemv_awq_w4a16` and `sol_attn` from one
  wheel, and `bench/check_h3_awq_encoder.py --gpu` already forces kitchen's
  CUDA backend across the real H3 call as a control.
- The AWQ encoder's `custom_operations` live on the CLIP; `sol_attn` patches
  the DiT. They share no state beyond the wheel; `_install_quant_format`
  refuses if `h3_awq_w4a16` is already registered with a different contract.

**The interaction that v2 changes is Sol's exact sink.**
`custom_nodes/ComfyUI-SolAttn-cuda/sol_attn_minimax.py::_sink_blocks` covers rows
`[0, video_start)`: text, keyframe cond and cond audio, reference image,
reference audio, and the target audio. [`SOLATTN.md`](../../../../SOLATTN.md)
already records that "images and videos pay in two segments, and the text
segment is inside the sink as well, so a reference grows the exact region
twice". A reference still costs about 264 to 294 text-segment rows under the
v1 artifact's 301k-pixel cap and 2,040 to 7,296 under v2's release bounds
(both MEASURED in the layer-49 policy benchmark). Sol's exact work scales
with the sink, so v2 will shrink Sol's speedup on reference-heavy graphs by
an amount nobody has priced (and with the owner's later decision that stills
calibrate at the 2048 upscale, `f2f6d5a`, a parity graph sits at the top of
that range: 7,296 rows per 16:9 still, plus `video_policy=release` for video), and the shipped `min_tokens` / `sink_conditioning`
findings were all measured at v1 row counts.

Two integration items follow, neither of which exists today:

1. **`image_policy=encoder` and `video_policy=encoder` read the v1 artifact's
   snapshot regardless of the loaded CLIP** (`reference_conditioning.py::_qwen_video_settings`;
   `bench/preflight_graph.py` prices Qwen rows through the same policy).
   Under a v2 encoder both would report v1 geometry and the preflight would
   under-price the packed sequence, the sink, and the Sol split by roughly
   eightfold per reference. **Done the same day**, commit `489b259`: the
   loader stamps its artifact's contract on the CLIP, the conditioner and the
   preflight resolve `encoder` from it, and a CLIP that declares nothing
   resolves to native. A v2 directory carrying the release processor files
   declares itself through `snapshot_contract(directory)` with no table row.
2. **Re-measure the Sol frontier at v2 row counts** before any v2 render is
   compared to a v1 one; the exact/sparse split moves with the sink, so a
   timing comparison across encoders is otherwise confounded.

## What I could not determine

- Marlin versus kitchen dequant-plus-cuBLAS crossover at M = 1k to 10k on
  sm_89; no benchmark in either tree covers it.
- Whether `import vllm._custom_ops` is side-effect-free inside a ComfyUI
  process, or whether the precompiled `_C_stable_libtorch` loads against a
  torch other than 2.13.0.
- Whether vllm's fp32 Triton attention path is exercised by any test.
- Whether cuDNN's own SDPA accepts FP32 through flashinfer's cudnn backend;
  moot for vision, which its head_dim constraint (128 or 192) rules out.
- Anything about render quality; nothing here touches it.
