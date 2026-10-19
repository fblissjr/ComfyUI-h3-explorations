# What upstream says: the paper, Sol-Engine, Sol-H3, core, kitchen, sglang and the other packs

This page states what other people ship and claim, once per upstream, with the revision it was read at. It asserts
none of our numbers and takes no decision except in its last section, "What we would adopt, and what would decide
it". Where it and a page about our configuration disagree about ours, that page is right. Every pass over
comfy-kitchen and ComfyUI core, with what each one found, is logged in
[`sol_upstream_reads.md`](sol_upstream_reads.md).

| source | how it was read | revision | when |
|---|---|---|---|
| arXiv 2607.24027, the Sol-Attn paper | abstract, ablation summary, HTML v1; not a full read | v1 | 2026-08-16 |
| Sol-Engine's per-hardware H3 cells | source, `coderef/Sana` on `sol-engine` | first read at `6fb7eb1`, re-verified at `757d902` | 2026-08-15, 2026-09-10 |
| Sol-H3, Sol-H3-Spark, `super_acceleration`, `RTX4090` | source; added by `9791888` (RTX4090, 2026-08-17), `2936c47` (Sol-H3, 2026-09-08), `8249b28` (its MXFP8, 2026-09-09), `757d902` (Spark) | `757d902` | 2026-09-10 |
| [`Efficient-Large-Model/H3-to-LTX-Latent-Adapter`](https://huggingface.co/Efficient-Large-Model/H3-to-LTX-Latent-Adapter) | the Hugging Face model card only | | 2026-09-10 |
| ComfyUI core's `BlockSparseAttention` | source in the ComfyUI checkout | `1f641fd9`, line citations re-read at `65787d66` | 2026-09-10, 2026-10-02 |
| core running Sol with no node | the pass logged in `sol_upstream_reads.md` | `08ff3c11` | 2026-10-09 |
| comfy-kitchen | fetch, `gh` and the fork clone | the installed build ([`sol_kernel.md`](sol_kernel.md)) | latest pass 2026-10-09 |
| sglang's SubBlock router | source | `ffe98a4279`; `sage_fp8`'s arch set at `2f5c9ac43d` | 2026-09-10, 2026-09-25 |
| sglang's Sol-Attn backend | source, its attention-backend doc and H3 cookbook page | `593c7a900d` | 2026-09-11 |
| Saganaki22 and sumeetprashant ComfyUI packs | their READMEs only | | 2026-08-16 |
| xmarre's ComfyUI-Sol-H3 | its README and the body of its PR 9, via `gh` | | 2026-09-11 |

Every `coderef/Sana/...` pointer below resolves against a checkout at `757d902`. The branch `release/sol-h3-spark` has
the same tree as that tip.

## The paper

**"Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification."** Read 2026-08-16:
the abstract, the ablation summary and the HTML, not the whole PDF. Everything below is at that depth.

**The contribution is the correction, not the sparsity.** The framing is that
existing block-sparse methods fail two ways: routing is rigid and costly to
materialize, and unselected blocks are dropped outright, which degrades
accuracy under aggressive sparsity. Sol-Attn thresholds during online softmax
and **reuses the proxy scores of unselected blocks to approximate their
contribution** instead of discarding them. The ablation that matters compares
exact-only against exact-or-approx, and the correction's advantage **widens as
sparsity rises**.

What follows from it:

- **The pooled tail is the method, not a side knob.** Sol-Engine's public API
  has no toggle for it at all, and neither does the copy Sol-H3 vendors (both
  signatures are cited below). Our node has no switch for it either ([`sol_attn.md`](sol_attn.md)).
- **`tau` is the paper's beta, and it is never swept.** The threshold is
  `t_i = mu_i + beta * sigma_i` over the row-wise mean and standard deviation,
  with beta described as a shared standardized cutoff. Sol-Engine defaults 1.0.
- **H3 is not evaluated anywhere in it.** Tested models are Wan 2.1-14B,
  HunyuanVideo-13B, LTX 2.3-22B, Bernini-14B for editing, a SANA-WM refiner,
  Ideogram 4 for 2K text-to-image, and Wan 2.1-1.3B on a 5090. The headline
  speedups are in the abstract, against dense.
- **No token reordering appears in it.** No Morton, no Z-order, no permutation, no spatial block layout ([`morton.md`](morton.md)).
- **sm89 was not an official target** when the paper and its docs were written. That changed on 2026-08-15 ("The SM89 CuTe kernel" below).




## Sol-Engine's per-hardware H3 cells

Sol-Engine **supports MiniMax-H3 as a first-class model** with validated,
pinned configurations. The paper does not mention H3. The framework does.

The per-hardware cells run at **1344x768, 124 frames**, 50 steps, on the
released BF16 weights. The packages added since run other geometries
(Sol-H3 at 124, 243 or 362 frames, Spark's draft at 672x384, the
`super_acceleration` first stage at 896x512), each cited in its own section.

**The recipe is per-hardware-profile, not one policy.**

| cell | attention | thresholding | cache |
|---|---|---|---|
| 4xA100 (sm80) | **Triton reference**, tau 1.0 | `exact` | FirstBlockCache |
| 4xH100 (sm90) | CuTe, tau 1.0 | `exact` | FirstBlockCache |
| 8xGB200 (sm100) | CuTe, tau 1.0 | `diag` | FirstBlockCache |
| 1x RTX 5090 (sm120) | CuTe, tau 1.0 | `diag` | TeaCache |
| **1x RTX 4090 (sm89)**, added 2026-08-17 | SM89 Sol-Attn, tau 1.0 | `diag` | TeaCache |

Sources: `coderef/Sana/models/minimax_h3/A100/adapter.py:550` and the H100
cell's same line, `coderef/Sana/models/minimax_h3/GB200/gpu_infer.py:144`,
`coderef/Sana/models/minimax_h3/RTX5090/adapter.py:477`,
`coderef/Sana/models/minimax_h3/RTX4090/adapter.py:476-477`, and each cell's
README.

**A100 is the Triton reference path**, because sm80 has no CuTe kernel.

**Every per-hardware cell runs the first two blocks and the first 10 of 50
steps dense.** `SOL_ATTN_FIRST_DENSE_STEPS` and `SOL_ATTN_FIRST_DENSE_LAYERS`
(`coderef/Sana/models/minimax_h3/RTX5090/adapter.py:457-460`, the same
function in the RTX4090 cell at `:457-461`), `H3_SOL_DENSE_LAYERS`
(`coderef/Sana/models/minimax_h3/GB200/gpu_infer.py:146`), `dense_blocks: int = 2`
(`coderef/Sana/models/minimax_h3/GB10/fusion_install.py:84`). This is the per-hardware cells' policy, not NVLabs' H3 policy: the packages added since do not follow it. Sol-H3 runs Ref2VA with no dense steps
or layers, and Spark's Ref2VA draft keeps only step 0 and layer 0 dense, both
cited below.

**None of the per-hardware cells reorders tokens.** Stated explicitly in the
A100 README: "pinned SGLang BF16 execution without `torch.compile` or token
reordering" (`coderef/Sana/models/minimax_h3/A100/README.md:24`).

### The single-card cells: RTX 5090 and RTX 4090

`models/minimax_h3/RTX5090/` runs H3 on **one consumer GPU** with SGLang
layerwise component offload, and it ships an isolated Sol-only arm,
`config/minimax_h3/rtx5090_sol.toml` against `config/minimax_h3/rtx5090_dense.toml`,
with tau 1.0, `diag`, and the dense steps and blocks above. **No number is
published for that arm**; the RTX 5090 README publishes only the full-opt
figure (`coderef/Sana/models/minimax_h3/RTX5090/README.md:13`), and full-opt
includes TeaCache and `torch.compile`.

`models/minimax_h3/RTX4090/` is the same stack on **this repo's card class**:
the released BF16 FL2VA checkpoint on one 24 GiB RTX 4090, the 33B DiT, the
Qwen3-VL conditioner and the VAEs under SGLang layerwise component offload
(`coderef/Sana/models/minimax_h3/RTX4090/README.md:5-7`). Its full-opt recipe
(`:33-40`): SM89 Sol-Attn at tau 1.0 with `diag`, an exact prefix KV sink and
dense prefix queries, the first 10 steps and first two blocks dense, TeaCache
(`coderef/Sana/models/minimax_h3/RTX4090/teacache.py`), and regional
`torch.compile`. It imports the `sol_attn` package
(`coderef/Sana/models/minimax_h3/RTX4090/adapter.py:337`, `:474`), whose
dispatch table in Sana maps compute capability 8.9 to `cute_sm89`
(`coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:11`).

**It publishes a Sol-increment arm, and that arm sits on top of TeaCache, not
against dense.** The controlled attribution table
(`coderef/Sana/models/minimax_h3/RTX4090/README.md:20-31`) goes lossless opt,
then plus TeaCache, then plus TeaCache and Sol-Attn, from one warm timing
sample per arm with no perceptual or embedding similarity metric, which the
README says itself. Its correctness evidence is a real-QKV gate against SDPA
(`:83-85`, the gate at `coderef/Sana/models/minimax_h3/RTX4090/adapter.py:347-362`,
`tau=-1000` so every block is routed). The README records the evidence as
collected at Sana `6fb7eb1` before the cell was split into its own directory
(`:81`, `:88-91`).


## Sol-H3: the multi-GPU production package

Added by `2936c47` (2026-09-08). A production inference package for text-to-video,
first-frame image-to-video and Ref2VA at 1344x768 and 24 FPS, for 1, 2, 4 or 8
GPUs, **validated on 8x NVIDIA B300** (`coderef/Sana/models/minimax_h3/Sol-H3/README.md:5-12`).
Kernel validation is limited to SM103; the SM90 and SM120 paths are declared
unvalidated (`:32-34`). One GPU runs dense by contract: the engine refuses SOL
attention below two processes
(`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/engine.py:147-148`).

It runs four DiT forwards (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/engine.py:22`: five scheduler points), at shift
12 for video and 3 for audio (`:214-215`), with the FastH3 4-step preview
adapter for T2V and I2V and the LightX2V ref2v Turbo 4-step LoRA for Ref2VA
(`coderef/Sana/models/minimax_h3/Sol-H3/README.md:115`, `:140`). Attention
backends are `dense`, `sol` and `sol_bsa` (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/engine.py:24`); Ref2VA accepts
only `dense` or `sol_bsa` (`:149-150`), and `sol_bsa` is the default
(`coderef/Sana/models/minimax_h3/Sol-H3/README.md:148`).

### Policy per task

One call sets it (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/engine.py:265-272`):

| task | tau | dense steps | dense layers | sink mode |
|---|---|---|---|---|
| T2V, I2V | 1.0 | 1 (the first of four forwards) | 2 | `prefix` |
| Ref2VA | 1.0 | 0 | 0 | `text_audio` |

- **No dense tail.** The only step gate is `step < dense_steps`
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py:345`);
  the layer gate is beside it (`:349`).
- **The diagonal band is forced exact**, one block either side
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/bsa_metadata.py:84`).
  comfy-kitchen forces the same band (the `<= 1` line in
  `coderef/comfy-kitchen/comfy_kitchen/backends/eager/sol_attn.py`).
- **No top-k list, no per-head or per-block tau.** tau is one value per call.

### How the sink is built

- **`prefix`** marks everything before the target-video tail as an exact KV
  range (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py:296`) and then recomputes every prefix query row
  with dense attention (`:418-420`). The module docstring gives the reason: the
  audio rows are generated, not only conditioned on, and one prompt's dialogue
  fell apart under the text-only policy (`:26-38`).
- **`text_audio`** leaves the Sol-kernel sink empty (`:287-288`) and instead
  builds a per-block mask from the text and audio indices, **excluding Ref2VA
  visual references** (`:298-318`); the BSA route then forces those key blocks
  and those query blocks (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/bsa_metadata.py:85-88`). So text and audio are exact
  both ways while reference rows stay sparse. It does this through cuDNN
  block-sparse attention's per-block selection, not through the Sol kernel's
  contiguous sink.

### The kernel it carries

- **`sol_bsa`** computes the routed blocks with cuDNN block-sparse attention
  and merges the pooled term for the omitted blocks through a log-sum-exp in a
  Triton kernel (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sol_residual.py:146-160`).
- **`relayout.py`** is the Ulysses all-to-all layout pair, with an optional
  int8 wire path after QK-norm and RoPE; it moves heads between GPUs and does
  not reorder tokens (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/relayout.py:1-27`).
- **The vendored `sol_attn`** is pinned to Sana `cee25847a` as "released Sol-Attn
  integration contract" (`coderef/Sana/models/minimax_h3/Sol-H3/SOURCE_SNAPSHOT.json:14-16`).
  Same algorithm as comfy-kitchen's: 64-token blocks, threshold `mean + tau*std`
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn/preprocess.py:232-240`),
  and the pooled tail unconditionally on in the Triton reference
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn/triton_ref/fwd.py:108-131`). Differences: it is
  BF16 throughout (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn/sm120/mainloop.py:727-729`), and its
  signature has **no top-k, no `sink_q`, no tail toggle**
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn/interface.py:421-433`).
  **Its architecture table omits sm89**: it lists (9,0), (10,0), (10,3) and
  (12,0) (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn/interface.py:10-19`), and anything else selects the Triton
  reference (`:95`).
- **Its arithmetic gate** runs block-sparse attention with every block selected
  against dense SDPA on the real Q/K/V, once per shape
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py:470-500`).
- The module docstring repeats the "H3 needs no reordering" statement word for
  word from the GB200 cell (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py:8-12`); see the Morton section.

### MXFP8 and the transport quantization

- **MXFP8 is linears only, and compute capability 10 only.** It replaces the
  fused QKV, the output projection and the FFN up and down projections in
  blocks 2 to 46 (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/compute_quant.py:11-12`,
  `:96-130`), with E4M3 values and one E8M0 scale per 32 along K
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/mxfp8.py:3-4`), and
  refuses any device whose major capability is not 10 (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/compute_quant.py:27-35`).
  Attention's QK and PV products are not touched.
- **The multi-GPU exchange is quantized separately**: group-scaled int8 for the
  Q/K/V packet and FP8 for the returned output
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/comm_quant.py:1`, `:14-24`,
  `:130-139`). This exists only when there is an exchange.

### What it validates

The README's latency tables and its own caveat that the Sol-H3 rows are a
four-forward distilled adapter plus approximate attention, so "the difference
from either 49-forward baseline is not a runtime-only speedup"
(`coderef/Sana/models/minimax_h3/Sol-H3/README.md:36-65`). Its one quality
comparison is MXFP8 against BF16 at one prompt and seed, as decoded-RGB PSNR
and SSIM (`:67-83`).


## Sol-H3-Spark: one DGX Spark, two stages

Added by `757d902`. A two-stage video-and-audio pipeline for **one GB10
(SM121)**; every stage refuses any other capability
(`coderef/Sana/models/minimax_h3/Sol-H3-Spark/README.md:1-5`, `:71`;
`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage2.py:181-182`;
`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage1_ops/sol.py:101-102`).
The recipe is a frozen record whose hash is checked at load
(`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/config.py:22-27`), so the
config file is the authority (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/configs/default.json:4-42`):

1. **Stage 1, an H3 draft** at 672x384x124: FastH3 VSA DataFree LoRA, W8A8 FP8
   DiT, four updates, VSA at 90% video sparsity on 64-token tiles through cuDNN
   BSA, no video decode (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/configs/default.json:4-14`). The VSA profile keeps the prefix
   exempt and has no dense layers or steps
   (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage1_ops/vsa.py:19-40`).
2. **Latent transfer**: a learned x2 H3 latent upscaler, then the H3-to-LTX
   latent adapter, which **replaces an H3 VAE decode followed by an LTX VAE
   encode** (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/README.md:30`, `:47-51`; `coderef/Sana/models/minimax_h3/Sol-H3-Spark/configs/default.json:15-22`). The adapter is a
   small Conv3D residual network
   (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage2_ops/h3_ltx_adapter/model.py:27-42`).
   Its [model card](https://huggingface.co/Efficient-Large-Model/H3-to-LTX-Latent-Adapter)
   gives the boundary as a normalized H3 latent `[B, 24, T, H/16, W/16]` in and
   a normalized LTX-2.5 latent `[B, 128, T, H/32, W/32]` out, and says the
   checkpoint holds neither model's weights and generates nothing alone.
3. **Stage 2, an LTX-2.5 refiner**: BF16 LTX-2.5 dev with a distilled LoRA,
   three joint audio-video updates, Triton Sol with
   `"taus": [1.0, 1.25, 1.5]`, `diag`, layer 0 dense (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/configs/default.json:23-37`,
   `coderef/Sana/models/minimax_h3/Sol-H3-Spark/README.md:33`). **This tau ramp runs on the LTX refiner, not on H3.**
4. **Output**: LTX's own Conv VideoVAE at 1344x768, 121 frames, with H3's
   generated audio muxed in (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/configs/default.json:38-42`).

### Ref2VA's opt-in Sol draft: per-step tau and a permuted sink

Ref2VA swaps in its own partition and the LightX2V Ref2VA 4-step LoRA, and
its draft attention is **dense FA4 by default, Sol only when selected**
(`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/config.py:12`, `:36-51`).
When Sol is selected the policy is (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/config.py:52-59`):

- dense for step 0 and for body layer 0 of the later steps;
- `tau_by_step` `[None, 1.0, 1.25, 1.5]`, i.e. **sparser as denoising
  proceeds** (higher tau keeps fewer blocks); the same tuple is `TAUS` in
  `coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage1_ops/sol.py:12`,
  routed by `route()` (`:15-18`);
- text and audio tokens are the sink and their queries run dense; reference
  image and Qwen-visual tokens are **not** in the sink.

This is the only place in the tree where **H3** runs a per-step tau.

**How the Sol kernel is made to express "text and audio exact, references
sparse".** The kernel's sink is one contiguous range, and H3 packs reference
rows between text and audio. `sink_plan` builds a permutation that puts every
visual row first and every text or audio row last, each part in its native
order, and records `"sparse_block_grouping_changed": True` in its receipt
(`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage1_ops/sol.py:21-47`). Q, K and V are permuted after RoPE, the kernel runs with the
sink over the trailing text-audio range, the sink query rows are recomputed
dense, and the output is permuted back (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage1_ops/sol.py:134-148`). The route counts per
request are asserted, not assumed (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage1_ops/sol.py:83-92`). Its own validation notes
describe the same policy (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/docs/validation.md:92-98`).

### Reference matching

"Reference matching" is a **per-image pixel budget**, taken from the draft
canvas or the final canvas (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/config.py:43-50`), applied with each image's own
aspect ratio and nearest-32 alignment, Lanczos resampling
(`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/qwen_ops/media.py:46-73`;
`coderef/Sana/models/minimax_h3/Sol-H3-Spark/docs/validation.md:87-90`). It is not an identity matcher.

### What it validates

Functional coverage, stated as such: bitwise context and latent checks, call
counts, and final media produced, for T2VA, three FL2VA endpoint cases, and a
small set of Ref2VA image and image-plus-audio cases
(`coderef/Sana/models/minimax_h3/Sol-H3-Spark/docs/validation.md:25-40`, `:49-106`). The same file says it gives no
identity, accessory or voice fidelity guarantee and does not establish that
reducing reference detail preserves quality (`:104-106`).


## `super_acceleration`: the same split across two GB200s

H3 FL2VA as stage 1 on one GB200 at 896x512 with the LightX2V four-step LoRA,
then a tensor handoff to an LTX-2.5 stage 2 on a second GB200
(`coderef/Sana/models/minimax_h3/super_acceleration/README.md:14-31`). Stage 2
keeps LTX layer 0 dense and runs layers 1-47 on Sol with taus 1.0, 1.25, 1.5
(`coderef/Sana/models/minimax_h3/super_acceleration/stage2/sol_attention.py:18`,
`STAGE2_TAUS`). **That ramp is on the LTX transformer, not H3.** The README
makes no speedup claim and says its validation "must not be described as a
quality pass" (`coderef/Sana/models/minimax_h3/super_acceleration/README.md:52-56`). Spark runs the same H3-draft-then-LTX-refiner
shape on one box.

## ComfyUI core's own Sol node, `BlockSparseAttention`

Merged 2026-09-06 (Comfy-Org/ComfyUI #16072, kijai; core commit `e308cc73`),
display name "Model Sparse Attention", experimental
(`comfy_extras/nodes_sparse_attention.py:353-365`). Its defaults are read from
`BlockSparseAttention.define_schema` (`:355-411`), not restated here.

- **Three methods**: `sol-attn` (adaptive tau), `sla` (a fixed keep percent,
  for LoRAs distilled against that pattern) and `vsa` (FastVideo's cube tiling
  with the learned coarse branch, for FastH3 weights) (`:368-389`).
- **Token routing is on for every block by default.** `extra_tokens`
  (`:398-401`) is passed as kitchen's `token_aug` on both paths (`:208`,
  `:296`), and is ignored under VSA (`:326-329`). The Python `execute`
  signature defaults it to 0 (`:414-415`); the schema default is what a graph
  built in the UI carries.
- **The sigma window** is `start_percent` / `end_percent` (`:390-393`), turned
  into sigmas at patch time (`:331-332`); with its default end the window runs
  to the last step. `dense_blocks` and `min_tokens` gate as their names say
  (`:68-82`).
- **Sink modes** `exact_kv`, `exact_kv_and_rows` (the default) and `off`
  (`:402-406`): the packed prefix is an exact KV range for every query, and the
  second mode also runs the target-audio query rows dense (`sinks()`, `:84-100`).
- **Two execution paths.** Generic models go through the attention override,
  which calls `ck.sol_attn` on full Q/K/V and falls back to whatever override
  was on the hook before it (`:171-226`). **H3 goes through block patches**
  (`set_model_patch_replace(..., "dit", "double_block", i)`, `:343-345`) into
  kitchen's chunked producer: Q/K/V are projected in 4096-row slices straight
  into the kernel's int8 carriers and never built in full (`:248-307`), and
  each block's pooled `kmean` / `vscale` statistics are **carried over from the
  previous step**, per conditioning branch (`:263-270`, `:292-301`). The CUDA
  backend's `sol_attn_chunked` docstring at comfy-kitchen tag `v0.2.33`
  describes those two tensors as the last step's statistics, computed fresh
  when absent. The V scale is last step's V absmax times a clip margin
  (`_SOL_VSCALE_MARGIN` in the installed backend's `vscale_of`), so a V
  louder than last step's by more than the margin clips; the routing
  threshold's key variance is taken from the current call.
- **What core's model gained for it**: a `to_gate_compress` layer created when
  the checkpoint carries one (`comfy/ldm/minimax/model.py:169-172`, detected at
  `comfy/model_detection.py:415`, i.e. FastH3 VSA weights), and two
  `transformer_options` keys for attention patches, `minimax_h3_layout`
  (`comfy/ldm/minimax/model.py:667`) and `block_index` (`:760`). Line numbers read at core `65787d66` (2026-10-02).

**Core can also run kitchen's Sol-Attn with no node in the graph** (read 2026-10-09 at `08ff3c11`), on a model
file's or a LoRA's say-so (`b26625f2`, #16831; `f49c531e`, #16880), at kitchen's defaults but for the threshold.
That is a different door to the same kernel; `BlockSparseAttention` did not change. Before that, a model file could
already name a block's attention through a `<module>.comfy_attention.config` tensor (`c194dd00`, #16419, kitchen INT8
only); an `optimized_attention_override` keeps priority, so Sol's override still wins, and no file in
`workflows/h3_config.py::MODELS` carries the key.

## comfy-kitchen

**The Sol kernel is upstream's.** Everything once carried from kijai's branches is merged: PR 117 (int8 Sol-Attn,
2026-08-29), 143 (its HIP port), 150 (`4950f16`: full-range P quantisation in the exact branch, fp16 q/k/v/out through
an `elem` code, and a public `sol_attn_chunked`), 156 (`b678fdf`, token routing; the identical tree to the PR head
`1128df6` that the 2026-09-04 grade was taken against, so the grade transferred,
[`data/results/2026-09-08_kitchen_0233_blk_cnt_rebase.json`](../../data/results/2026-09-08_kitchen_0233_blk_cnt_rebase.json))
and 158 (its HIP port). By content, kijai's `sol_exact_pquant`, `sol_token_aug_main` and `minimax_vae` branches hold
nothing the tags lack. Comfy-Org squash-merges, so commit counts against upstream overstate what is unmerged. A branch
named `sol_fp16_pv` in the kijai mirror is kijai's 2026-08-14 `sol_attn` head, not a 16-bit PV variant.

**What we carry**, on the owner's fork (`coderef/comfy-kitchen`, branch `h3-frontier`, upstream main plus our
commits since 2026-09-27; [`sol_kernel.md`](sol_kernel.md) has how to read the installed build):

| commits | what | upstream status |
|---|---|---|
| `blk_cnt` | the routed-count out-parameter on `sol_attn` and `sol_attn_chunked`; Python-only, since the route stage already writes the count | PR 168 carries the two `sol_attn` commits; open, no activity since 2026-09-11. The chunked commit stays a fork delta even if it merges |
| `qk_balance` | per-head q/k channel rebalancing inside the INT8 quantizers ([`block49_quant_error.md`](block49_quant_error.md)) | branch `sol-qk-balance-pr`: rebased by hand onto `e5e0d02` (the v0.2.34 tag) with the `blk_cnt`-adjacent lines dropped and the two route tests restated on the output; built for sm_89, the ten new cases pass and the suite fails only the pre-existing `test_topk_ties_over_select`. Not pushed, no PR. The PR text is [`sol_upstream_qk_balance_pr.md`](sol_upstream_qk_balance_pr.md); the owner pushes and opens it, with `gh pr create --repo Comfy-Org/comfy-kitchen --base main --head <fork>:sol-qk-balance-pr` spelled out |
| `rotate` | Hadamard rotation of q/k before Sol's INT8 quantizers | ours only |

**Carried for good (owner, 2026-09-11).** The carried commits stay whether or not their PRs merge, and everything
else tracks upstream; `vendor/rebuild_kernel.sh` refuses any other source and `--check` reports whether the source is
current without building. What would make the carry change: PR 168 merging (drop those commits on the next rebase),
or a kitchen change to `quant_k_rows` / `quant_q_rows`, where `qk_balance` and `rotate` live.

**Upstream's own Sol suite does not pass on this card**, on the stock PyPI wheel or on a local build of the same
commit: three groups of nanobind binding-validation cases whose expected error only the HIP bindings raise, and
`test_topk_ties_over_select` against its own cosine bar. One binding case leaves a sticky CUDA error, so a single gap
reads as dozens of failures in the same process. Attribution and the stock control are in the 2026-09-08 rebase
record above. Whether these fail on upstream's CI is not established.

**Open kitchen PRs that matter here**, as of the 2026-10-02 pass: 179 (`draft_attention`, with core half 16362) is the
one open kernel that would run on this card; kijai graded it against `sol_attn` on real H3 captures and found it no
better at matched speed, with a larger workspace, so the instrument if it merges is a capture grader, not a render.
The H3-targeted stack 217-231 either dispatches only on SM120 or adds opt-in APIs core does not call yet. The rest of
each pass is in [`sol_upstream_reads.md`](sol_upstream_reads.md).

## sglang's SubBlock router: two notes, read 2026-09-10

[`docs/research/2026-08-25_sglang_h3_pipeline.md`](../research/2026-08-25_sglang_h3_pipeline.md) §11
owns SubBlock. Two things in the source it does not yet record:

- **A reserved sink block and a forced diagonal were tried and left out.** The
  router's docstring says both were measured on real H3 attention cells, that
  the diagonal barely moved the error and the sink helped only in a middle band
  of DiT layers that did not survive to the pixels
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:212-215`).
- **The step and layer cutoffs**, already recorded at
  `docs/research/sglang_h3_pipeline.md` lines 722-729 at `901bee1`, now sit beside a second
  compute mode: `sage_fp8`, which on SM90 quantizes Q/K to INT8 and V and the
  softmax probabilities to E4M3
  (`coderef/sglang/python/sglang/kernels/ops/attention/subblock_sage_fp8_sm90.py:2-8`),
  added by `ffe98a4279` (2026-09-09) and not the default
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py:106`;
  the cutoff defaults and their stated sweep are at `:76-85` of the same file).
  *2026-09-25: `2a0cb2f04e` (#40116) extends `sage_fp8` to SM120 through a
  FlashInfer CuTe-DSL kernel and makes its default key sub-block count per
  arch. `DEFAULT_COMPUTE_MODE` is still `bf16`, and it now sits at `:105`.
  SubBlock still accepts only SM90, SM100 and SM120, so none of it runs on
  this card.*


## sglang's own Sol-Attn backend, read 2026-09-11

sglang has had a `sol_attn` DiT attention backend since `51470b376f` (#33702,
2026-08-09). It wraps NVLabs' `sol_attn` package from Sana's `sol-engine`
branch and is opt-in: sglang's default DiT backend is `fa`.
[`docs/research/2026-08-25_sglang_h3_pipeline.md`](../research/2026-08-25_sglang_h3_pipeline.md)
section 14.9 walks the source with line citations, and
[`docs/h3/sglang_comparison.md`](../h3/sglang_comparison.md) sets it
beside core's node and ours.

- **Defaults**
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/sol_attn.py:59-83`):
  tau 1.0, `thresh_type` `diag`, `kv_splits` `auto`, `sink_tokens` 0,
  `sink_start` 0, `dense_steps` 10, `dense_layers` `"0,1"`, dense backend
  `fa` or `sage_attn`.
- **Step policy.** Dense for the first `dense_steps` step indices, sparse
  from there through the last step, with no end gate (`:128-140` of the same
  file).
  Tau, threshold, dense prefix and dense layers take the shape of the per-hardware cells' policy ("Upstream H3 policies beside ours" below); the sink does not.
- **Sink.** A fixed `sink_start`/`sink_tokens` range from the config.
  Nothing in sglang's H3 pipeline derives it from the packed layout, so by
  default no conditioning rows are exact.
- **Absent:** token routing, and a minimum sequence length.
- **INT8 QK** is passed only when the installed package accepts `int8_qk`
  and the device is SM89 or newer; the comment beside it says NVLabs'
  official API has no such argument (`:229-233` of the same file).
- **The documented H3 recipe** is `dense_backend=sage_attn,dense_steps=10`,
  with the text encoder on `torch_sdpa`
  (`coderef/sglang/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx:1116-1119`).

## Sol-Engine's shared kernel package, and what separates it from comfy-kitchen's

Re-derived from source on 2026-08-16, pointers re-verified 2026-09-10.

### `thresh_type`: two modes, and comfy-kitchen implements one

Their two modes, in `coderef/Sana/techniques/sparse_backends/sol_attn/preprocess.py`:

- **`diag`** (`_compute_diag_threshold`): variance estimated as
  `sum_d q_d^2 * var(kc_d)` from the per-dimension variance of the key-block
  centroids, which assumes independence across the 128 dims.
- **`exact`** (`_compute_exact_threshold`): the real score row's
  `E[s^2] - E[s]^2` over all key blocks for that query.

comfy-kitchen's CUDA kernel computes `kcvar[d]` as the per-dimension variance of
the key-block centroids
(`coderef/comfy-kitchen-kijai/comfy_kitchen/backends/cuda/sage_attention/sol_attn_preprocess.cu:106`),
reduces `c_d^2 * kcvar[d]` (`:148`), and thresholds at
`tau * sqrt(var + 1e-6)` (`:54`), against mean-centred key centroids so the
mean term is already absorbed. **That is Sol-Engine's `diag` formula.** `exact`
is therefore a threshold kernel comfy-kitchen does not have, not an input it
leaves unexposed.

The single-consumer-card cells (RTX 5090, RTX 4090) both run `diag`.

### `kv_splits`: SM90 only

`coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:116` raises
`"kv_splits=2/4 is currently available on SM90 only"`. A 4090 gets 1 in
Sol-Engine too.

### The SM89 CuTe kernel

PR #464, merged 2026-08-15 (the dispatch commit is `9cfdd07`): "a general BF16
SM89 CuTe DSL Sol-Attn forward kernel using M64/N64 tiles, cp.async, and warp
MMA". That is the 4090, and it makes NVLabs' own kernel a **second independent
implementation** beside comfy-kitchen's.

The code is at `coderef/Sana/techniques/sparse_backends/sol_attn/sm89/`, with
dispatch at `coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:11`
(`(8, 9): "cute_sm89"`). The RTX 4090 cell above is its first published H3
use. Sol-H3's vendored copy does not carry it (see "The kernel it carries").

**It is installed and it runs on this box.** `vendor/build_sana_sol_sm89.sh`
installs the runtime, builds the `sol-attn` wheel out of the checkout, and then
compiles and exercises the kernel on this card; run it for the versions and the
deviations, which are printed rather than written down here.

Two things that section of their docs does not say.

**The requirements list is incomplete.** It names PyTorch, CUDA, Triton,
`cuda-python` and the CuTe DSL. Installing the DSL is not sufficient:
`apache-tvm-ffi` is named in no Sol-Attn requirements list and is not a
dependency of the DSL wheel either, yet `sol_attn/common/runtime.py` passes
`enable_tvm_ffi=True` on every tensor and `_compile_sm89` compiles with
`--enable-tvm-ffi`, so the first SM89 call dies on
`ModuleNotFoundError: No module named 'tvm_ffi'`. Found by running it,
2026-08-19.

**The compile-time failure discipline does not cover the import.** Their docs
warn that a DSL *version* mismatch fails at compile time rather than falling
back, "so that a dense run is never reported as a sparse one". An *absent*
runtime is a different path: backend selection returns `"triton"` whenever the
CuTe runtime is unavailable
(`coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:84-91`). That is
why the build script asserts `get_sol_attn_backend() == "cute_sm89"` before it
measures anything.

**The API surface.** Their public entry point
(`coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:413-424`) is
`sol_attn(q, k, v, *, scale, tau, thresh_type, kv_splits, sink_tokens, sink_start)`.
There is **no `sink_q`**, no `dense_blocks`, no tail toggle, no `max_blocks`,
no `key_bias`. Their integration contract puts dense query rows at the caller:
"an MMDiT integration should still compute valid text query rows with dense
attention" (quoted in
`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py:21-24`).

### Token pruning and NVFP4

Two of Sol-Engine's five methods, and **not applied to their H3 stacks**.

- **NVFP4 needs Blackwell** (sm_100+) plus `transformer_engine`.
- **Token pruning** is model-agnostic scaffolding in
  `coderef/Sana/techniques/methods/token_prune.py` whose runtime lives in SGLang,
  and it needs a `ModelSpec` seam H3 does not have there. It ships on LTX-2.3,
  not H3.

## Morton in Sol-Engine, which does exist

Correcting a claim this repo made three times on 2026-08-15: **Sol-Engine does
ship Morton.** It is absent from the paper, absent from the published docs, and
present and on by default for Wan.

- `coderef/Sana/techniques/sparse_backends/sol_attn_backend.py` has
  `_morton3d_perm` (`:176`, "canonical x/y/z-interleaved") and
  `install_wan_morton_forward` (`:209`).
- The Wan Sol configs describe themselves as Sol-Attn "at tau 1.0 with
  **global Morton3D order** and dense guards"
  (`coderef/Sana/config/wan21_t2v_1_3b/wan21_sol_only.toml:4`, and the same
  phrase in the other Wan fullstack and Sol-only configs).
- HunyuanVideo has it **off**: `HUNYUAN_SOL_MORTON = "0"`
  (`coderef/Sana/config/hunyuan_video/full.toml:20`).
- H3 has it absent entirely.
- **They ship only the 3D curve.** There is no `2d_frame` variant anywhere in
  Sol-Engine; that is kijai's addition for H3's non-uniform temporal axis.

Their stated reason for H3 needing none, from
`coderef/Sana/models/minimax_h3/GB200/sol_attn_h3.py:8`, repeated word for word
in Sol-H3 (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py:8-12`):

> **Morton reordering per attention call.** The released
> `install_wan_morton_forward` docstring says why not: only self-attention is
> order-sensitive, so the permutation belongs once at the block stack, and
> doing it per call cost more than the kernel saved. What the H3 integration
> then shows is that H3 needs *no* reordering at all -- the packed video tail
> is already a contiguous grid-ordered block, and the routing works on it
> directly.

And the ablation control they ship for Wan, `config/wan21_t2v_14b/reorder_only.toml`:

> "reorder-only control: global Morton3D order with every layer forced dense
> through the Sol-Attn adapter."

**One H3 permutation now exists, and it is not a spatial curve.** Spark's
Ref2VA Sol draft permutes Q, K and V so visual rows come first and text-audio
rows last, keeping native order inside each part, to make the sink contiguous;
it records that block grouping changes (see the Spark section). So no Sol-Engine H3 path reorders tokens spatially, and one reorders them by segment.

Whether any of that transfers to H3 is argued in [`morton.md`](morton.md).

## FirstBlockCache and TeaCache

`coderef/Sana/models/minimax_h3/A100/first_block_cache.py`: about 30 lines of
policy around a threshold, gated on `H3_FIRSTBLOCKCACHE=1`, threshold from
`H3_CACHE_THRESHOLD`, with decisions synchronized across ranks so a multi-GPU
run cannot diverge. The RTX 5090 and RTX 4090 cells use TeaCache instead
(`coderef/Sana/models/minimax_h3/RTX4090/teacache.py`; the retained and
cooldown steps in `coderef/Sana/models/minimax_h3/RTX4090/README.md:39`).

The idea, which is not theirs alone and not new: run the first transformer
block, look at how much its output changed against the previous step, and if
the change is below the threshold, reuse the step's residual instead of running
the remaining blocks. A step-skipping cache, orthogonal to attention.

Their thresholds are tuned at **50 steps**, and every cache cell above runs 50.

## Other ComfyUI Sol-Attn packs

Found 2026-08-16, READMEs only.

**[Saganaki22/ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn)**
is H3-specific: zero-copy MiniMax H3 nodes, published benchmarks, and
**scheduled tau** -- ramping the threshold across sampling steps, sparse early
to dense late, with selectable interpolation curves. Spark's Ref2VA draft
ramps the other way, dense first and sparser late.

- They default **tau to 1.3, "tuned locally", against the paper's 1.0**.
- Their quality evaluation is **correctness against a reference** (L2 against
  SDPA, bit-identical strided views), not perceptual.
- Their speed numbers are on a 5090 with `torch.randn` tensors, which is both a
  different card and the synthetic-input trap ([`sol_instruments.md`](sol_instruments.md)).

**[sumeetprashant/ComfyUI-SolAttn](https://github.com/sumeetprashant/ComfyUI-SolAttn)**
exists and was not read past its README.

**[xmarre/ComfyUI-Sol-H3](https://github.com/xmarre/ComfyUI-Sol-H3)**, added
2026-09-11 from its README and the body of its PR 9 only, packages the Sol-H3
CuTe kernel from the author's own Sana fork and does not substitute
comfy-kitchen's `sol_attn` for it. Its supported kernel target is SM120 on
Linux or WSL2, so its sparse route does not run on this card. It composes
with a family of the same author's H3 packs (VDN, Spectrum,
Flow-Aligned-Regenerate and others), and PR 9 is the provider half of the
mixed-resolution work behind kitchen PR 171.

**Neither of the two 2026-08-16 packs reorders tokens.** With the paper and the Sol-Engine cells, no
searched source outside kijai's packs applies a spatial token order to H3.

## Published numbers, and why not to quote them against ours

Where upstream's H3 figures live, each with the conditions beside it:

- the per-hardware cells: each cell's README, e.g.
  `coderef/Sana/models/minimax_h3/RTX5090/README.md:13`;
- the RTX 4090 cell, including its TeaCache-then-Sol attribution:
  `coderef/Sana/models/minimax_h3/RTX4090/README.md:9-31`;
- Sol-H3 on B300: `coderef/Sana/models/minimax_h3/Sol-H3/README.md:36-65`;
- `super_acceleration` on GB200: `coderef/Sana/models/minimax_h3/super_acceleration/README.md:40-66`.

**None of these is a Sol-Attn-only number.** Full-opt stacks a cache,
compilation and, on the multi-GPU cells, context parallelism on top of Sol,
against a 50-step dense baseline; Sol-H3 compares a four-forward distilled
adapter against 49-forward baselines and says so; `super_acceleration`
publishes latency with no baseline. Ours is Sol-Attn against sage at our own
step count on one 4090. Different stack, different baseline, different step
count, and on all but one cell different hardware.

This is the unit trap ([`sol_measured_arms.md`](sol_measured_arms.md)) at a larger scale. The RTX 4090 cell's
Sol increment is the closest thing to comparable, and it is an increment over
TeaCache from one sample. The one arm that *would* be comparable,
`rtx5090_sol` against `rtx5090_dense`, still has no published result.




## What we would adopt, and what would decide it

The sections above say what upstream ships; this one is the decision side, for one sm89 card with 24 GB. Read
2026-09-10 and 2026-09-11. **Nothing below has been run here.** Each row says what would decide
it under this repo's rules: a numerical knob is graded on captured activations
first, and a perceptual claim needs a blind distribution under
[`docs/method/eval_comparison.md`](../method/eval_comparison.md) section 3.

### Upstream H3 policies beside ours

Every upstream cell is cited above; the sglang column is
read from its `sol_attn` backend on 2026-09-11 and walked in
`docs/research/2026-08-25_sglang_h3_pipeline.md`. Our column is a pointer: the values
live in `workflows/h3_config.py::SOL_RECOMMENDED_CUDA`.

| knob | Sol-Engine per-hardware cells | Sol-H3, T2V / Ref2VA | Spark Ref2VA draft (opt-in) | sglang `sol_attn` (opt-in) | core `BlockSparseAttention` | ours |
|---|---|---|---|---|---|---|
| tau | 1.0 | 1.0 / 1.0 | none, 1.0, 1.25, 1.5 by step | 1.0 | its `define_schema` | `SOL_RECOMMENDED_CUDA` |
| threshold | `exact` on A100/H100, `diag` elsewhere | `diag` | `diag` | `diag` | kitchen's, which is `diag` | kitchen's, `diag` |
| dense at the start | first 10 of 50 steps | first forward / none | step 0 | first `dense_steps` (10), a step count | `start_percent` | `start_percent` |
| dense at the end | none | none | none | none | none at its default `end_percent` | none since 2026-09-11 (`end_percent` 1.0); the last step before that |
| dense layers | first two | two / none | layer 0 | `dense_layers`, 0 and 1 | `dense_blocks`, empty | `dense_blocks`, `SOL_DENSE_TAIL` (the middle ridge and block 49 since 2026-10-02; every upstream that keeps layers dense keeps the first ones) |
| sink | prefix KV exact, prefix query rows dense (our `exact_kv_and_all_rows`) | same / text and audio exact both ways, references sparse | text and audio exact both ways, references sparse, by permutation | none by default (`sink_tokens` 0, a manual count) | `exact_kv_and_rows` | `sink_conditioning`, `exact_kv_and_all_rows` since 2026-10-02 |
| token routing | absent | absent | absent | absent | on for every block | `token_routing`, off |
| token order | native | native | segment permutation | native | native | native (Morton closed 2026-09-27) |
| cache | FirstBlockCache or TeaCache | none | none | none in the backend | none | none shipped |

**The end of the window is the one knob where sglang and core agree against
what we shipped**, so it was adopted on 2026-09-11 under the owner's rule that
this repo takes upstream's default when those two agree
(`docs/h3/sglang_comparison.md`, the 2026-09-11 defaults section). The
start agrees only at 50 steps, because sglang counts steps and core uses a
fraction; every other row has the two disagreeing with each other.

### What could be adopted, and what would decide it

| technique, and where upstream runs it | runs here | what it would change | decided by | cheapest first test |
|---|---|---|---|---|
| **Per-step tau.** Spark's opt-in Ref2VA draft, four updates. A tau ramp also runs on Spark's and `super_acceleration`'s LTX refiners, which are not H3 | yes; the kernel takes tau per call | `tau_table` is per head and block, with no step axis; a per-step profile needs the step, which the override can take from the current sigma the way the window already does | captured activations first, then a blind distribution. The only H3 instance is a four-update Ref2VA draft, so whether it transfers to a 16-step base schedule is itself the question | give `bench/measure_sol_exact_variants.py --capture` a tau list (it grades fixed arms today) and read, per captured step, whether the error against exact at a higher tau late is no worse than at the shipped tau early: the premise the ramp rests on |
| **Text and audio exact, references sparse, by permutation.** Spark's Ref2VA Sol draft | yes; an `index_select` on Q/K/V around the kernel call and its inverse on the output, no kernel change | a fourth `sink_conditioning` mode: permute so the text and audio rows are contiguous, pass that range as `sink_blocks` and `sink_q`, permute the output back. It changes block membership at the boundary, so routing moves too. The `rows` input's `tau_map` route ([`sol_sink.md`](sol_sink.md)) now expresses the same split without a permutation, ungraded | captured activations on a reference capture against exact, beside `exact_kv_and_rows` and `exact_kv_and_all_rows`; then cost, from the route observer | call `ck.sol_attn` directly on permuted Q/K/V from a capture taken with `workflows/h3_probe_capture_ref3_api.json`, all three sink modes against fp32 dense |
| **Dense layers.** Two in the per-hardware cells and Sol-H3 T2V, none in Sol-H3 Ref2VA, layer 0 in Spark | yes; `dense_blocks` exists | a string in `SOL_RECOMMENDED_CUDA` | [`sol_dense_blocks.md`](sol_dense_blocks.md). Upstream disagrees with itself, so copying one of them settles nothing | as that page says |
| **An all-routed arithmetic gate.** Sol-H3's `_run_bsa_gate`; the RTX 4090 cell's `tau=-1000` gate against SDPA | already here | nothing | nothing | `bench/check_solattn_correctness.py` calibrates the eager oracle against SDPA with every block exact; `bench/measure_sol_exact_variants.py`'s `exact_all_routed` arm and `bench/read_sol_probe.py:334` run the kernel with every block routed against fp32 dense on captured Q/K/V |
| **Sana's `cute_sm89` kernel.** Sana's shared `sol_attn` package; the RTX 4090 cell imports it | yes; `vendor/build_sana_sol_sm89.sh` builds and runs it here | a second kernel behind the same override. It is BF16 throughout with no INT8 route, so it is better placed as a second, closer-to-exact implementation than as a production path (reasoned, not measured); their API has no `sink_q` | kernel-level grade on captured Q/K/V, both kernels against fp32 dense at the shipped tau with the same KV sink | one Base16 capture cell, KV sink only so both APIs can express it ([`sol_attn.md`](sol_attn.md), "What is open") |
| **TeaCache.** The RTX 5090 and RTX 4090 cells | yes; the RTX 4090 cell runs it on this card class | a model-level step cache, orthogonal to Sol and not an attention knob; it changes the trajectory | a blind distribution against the true baseline (`CLAUDE.md`). At few steps a skipped step is a larger share of the schedule, and their thresholds are tuned at 50 (reasoned) | render the existing EasyCache probe (`workflows/h3_probe_cache_easy_api.json`), a cache of the same family, before porting anything |

**What does not transfer, and why:**

- **MXFP8 compute.** It refuses any device whose major compute capability is
  not 10; sm89's is 8.
- **The multi-GPU exchange** (`relayout.py` and the int8 and FP8 transport).
  It exists only when heads move between GPUs; one card has no exchange.
- **`sol_bsa`, cuDNN block-sparse attention.** Upstream validates it on SM103
  only and declares SM90 and SM120 unvalidated. Nothing says whether the cuDNN
  block-sparse API runs on sm89, and nothing here has tried; not a candidate
  until that is known.
- **The two-stage H3-then-LTX pipelines** (Spark, `super_acceleration`). They
  keep both DiTs resident on a GB10's full memory or across two GB200s; on a
  24 GB card that is offload of two large DiTs, and the LTX refiner is outside
  this repo's H3 scope. The H3-to-LTX latent adapter is a separate question
  from Sol.
- **VSA** (Spark's T2VA and FL2VA drafts). It needs the FastH3 VSA weights; it
  is already probed here (`workflows/h3_probe_vsa_api.json`, exempt from Sol in
  `bench/check_attention_defaults.py::SOL_EXEMPT_STEMS`), and core's
  `BlockSparseAttention` runs it too.

### Core's `BlockSparseAttention` against ours: the implementation is not a variable (2026-09-10)

*Measured on captured activations*
([`data/results/2026-09-10_sol_impl_capture_grade.json`](../../data/results/2026-09-10_sol_impl_capture_grade.json), graded by
`bench/grade_sol_impl_on_capture.py` on the `pre=` capture recorded in
[`data/results/2026-09-10_capture_inventory_sol_impl_courtroom.json`](../../data/results/2026-09-10_capture_inventory_sol_impl_courtroom.json)). At
matched knobs (`h3_config.SOL_CORE_DEFAULTS`), core's chunked producer --
q/k-norm and RoPE inside the kernel, K/V statistics carried from the previous
step -- and our node's call on ComfyUI's own normed, roped tensors land the
same distance from exact attention in every cell. Against each other they are
close but not within the floor: the record's `core_fresh.vs_ours` sits above
our all-routed floor (`control_ours_all_routed`) in most cells, and
`core_carried.vs_ours` sits above it in every carried cell. Carrying the
statistics moves core's output away from ours but not its distance from
exact. So at matched knobs the two nodes are equally far from exact and their
defaults (the knob column above) are the larger difference. Against core's
defaults, ours at the policy shipped on 2026-09-10 (the record's
`knobs.shipped`; qk_balance and the dense tail came later) is not uniformly
closer to exact: it loses on all four block-0 cells and on the per-row mean,
and wins the whole-tensor mean because of block 49. The rendered pairs from
that day were never scored. Ours as shipped since (the `rotated` quantizer, the kitchen fallback, `SOL_DENSE_TAIL`, `exact_kv_and_all_rows`) has not been compared with core
(`bench/results/2026-09-27_attention_parity.md`). One capture, one scene,
three blocks, four steps: a statement about these cells, not every layer.