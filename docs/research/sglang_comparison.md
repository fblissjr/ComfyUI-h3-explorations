# sglang's H3 serving path against ours

last updated: 2026-09-19 (closing section "Sixth read" added, and a dated
note under the Sol-Attn defaults table); 2026-09-11 (subsection "Sol-Attn
defaults: sglang against core's and ours" added under "What we do that they
do not", and closing section "Fifth read" added; "Fourth read" added 2026-09-10; one subsection under "What
they do that we do not" and the section "Third read" added 2026-09-04;
everything else is the 2026-08-29 read)

What the vendor-side serving implementation does that this install does not,
what both do where ours may be the weaker version, and what looks like a gap
until you check the record and find it already priced.

**Read from source 2026-08-21** against `coderef/sglang` at commit
`a41da991c8`, alongside ComfyUI's `comfy_extras/nodes_minimax_h3.py`,
`comfy/ldm/minimax/model.py`, `comfy/model_management.py` and this repo's own
bench records. No renders were made for any of it. Every claim below says
which file it came from; nothing here is a measurement on this card unless it
cites a record in `bench/results/`.

**The clone has moved twice since that read, and this is the file's standing
hazard.** `coderef/sglang` was at `a7ec6b97f7` on 2026-08-22 and is at
`97781eb7f3` (2026-08-29) now. The prose sections below were read at
`a41da991c8`; only the sections dated later were read at a later commit.
`bench/check_doc_links.py` confirms every cited line still exists, which is not
the same as confirming it still says the same thing. Re-read before quoting an
older section as current.

**Why this file is not merged into the pipeline walk, asked 2026-08-29.**
Because the two drift for different reasons and on different clocks.
[`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) is a source read of somebody
else's tree: it goes stale when **their** code moves, and the fix is to re-read
at a new commit. This file goes stale when **ours** moves — a ComfyUI upgrade,
a node change, a new measurement here — and the fix is to re-derive against our
side. Merging them would produce one file that needs both kinds of maintenance
and signals neither, and would put a 700-line vendor description in front of
every reader who only wanted the delta. The split is kept. What was actually
drifting is fixed below and in
[`../comfyui_vendor_gaps.md`](../comfyui_vendor_gaps.md).

**The pipeline itself, before the comparison.** Since 2026-08-25
[`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) is the stage-by-stage walk of
sglang's H3 path: request, time grid and canvas, media ingestion, the Qwen3-VL
encode, the VAE encodes, the packed sequence, the DiT forward, the denoise
loop, decode and output, the runtime around it, and a numbered insights
section. It compares nothing; this file does. Read it first when the question
is "what does sglang actually do", and this one when it is "how does that
differ from here".

**Scope, and what the index is for.** This file owns the *optimization and
runtime* comparison. The reference-conditioning comparison — sizing, patchify,
presentation, packing, condition timestep — is owned by
[`h3_references.md`](../h3_references.md), section "The vendor image path,
stage by stage", and its detail must not be restated here.

That split kept every divergence documented and left none of them listed
together. The consolidated snapshot that fixes it is
[`comfyui_vendor_gaps.md`](../comfyui_vendor_gaps.md), which defers to this
file and to [`h3_references.md`](../h3_references.md) rather than competing
with them.

---

## Every known divergence, in one place

[`comfyui_vendor_gaps.md`](../comfyui_vendor_gaps.md) is the consolidated
report: every gap between this install and the release, with practical impact,
a priority by what it costs a working user, and what is enforced by an
assertion versus by nothing. **It is a dated snapshot and this file is still
the authority** for everything below; where the two disagree, this one is
right and the snapshot is stale.

That file exists because the ownership rule below is correct and had a cost:
the gaps were all documented, across three files, and none of them were listed
together.

---

## The filter: most of sglang's speed is four cards

sglang's audited deployment for `quality="high"` is a 4xH200 fl2va server with
`sp_degree=4` and `ulysses_degree=4`, and `validate_quality_deployment`
(`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py:98-186`)
raises unless the resident server matches it exactly — down to asserting the
device is an H200 at compute capability 9.0. That same gate requires
`enable_torch_compile`, `enable_breakable_cuda_graph`,
`is_dit_layerwise_offload_selected` and `quantization` all to be **off**.

Two consequences worth holding on to. Their headline performance is sequence
parallelism we cannot copy on one card. And the knobs they ship but require
off for their quality claim are knobs they do not stand behind for quality
either, so "sglang has it and we don't" is not on its own an argument.

---

## What they do that we do not

### An exact AdaLN cache

`coderef/sglang/python/sglang/multimodal_gen/tools/build_minimax_h3_adaln_cache.py`
precomputes the modulation parameters for every timestep the schedule will
actually visit, and the DiT then drops `adaln_proj` entirely
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:1412-1414`).
There is an in-process prepass that builds the same thing by reading all the
`adaln_proj` layers once
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:1186-1265`),
sized by a plan-width knob whose per-task requirement is stated at
`:1206` — t2va, fl2va and ref2va need different numbers of distinct timesteps
per step, which is the same fact our packing already encodes.

**Why this is interesting here.** It is an exact answer to the problem the
Comfy-Org "pruned" checkpoints answer approximately. The pruned file replaces
those weights with a rank-8 SVD of the time curve
([`evidence.md`](../evidence.md) owns that measurement); the cache keeps full
accuracy and pays a cache tensor sized by the step count instead of by the
weight matrix.

**What it does and does not buy on this box.** No speed. On the pruned file we
run, the AdaLN projection is already tiny, so there is no matmul left worth
eliminating. What it buys is *unpruned accuracy at pruned memory*.

**#22 has since reported, and it half-opened the gate**
([`bench/results/2026-08-21_pruning_sensitivity.json`](../../bench/results/2026-08-21_pruning_sensitivity.json)).
The pruning is **not** invisible at the output: the first-step velocity moves
5.6-9.4% against a determinism floor of exactly zero. So "the residual does not
matter" — the outcome that would have killed this outright — did not happen.
What did happen is that the effect is the same size on both checkpoints and
**smaller than the `fp8_scaled`-vs-`int8_convrot` difference this repo already
ships**, which is why it is still not urgent: an exact AdaLN would remove an
error that is not the largest one in the stack.

**What would make it worth building** is evidence that the difference is
*visible*, which no measurement here can supply — the arms are one forward, not
a render. That is a blind session on pruned against unpruned under
[`eval_comparison.md`](../eval_comparison.md) section 3, and it has not been run
or scheduled.


### Two more sparse-attention backends, read 2026-09-04

Neither needs the FastH3 weights to run on base H3, and neither is a serving
feature; both are attention policies, which is the axis this repo works on.

**Cube sparse attention** (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/cube_sparse_attn/`,
merged 2026-09-02 with two MiniMax engineers as co-authors). FlexAttention
over `(T, H, W)` cubes of latent tokens, so a block is a spatial-temporal
neighbourhood rather than a run of Morton-ordered rows; a per-step
`topk_ratio_list` with one entry per denoise update, where a ratio of one
means that step runs dense; text, audio, standalone reference images and the
token refiner stay dense; reference videos and the target compete in one
global top-k pool. No pooled correction term, so it is the SLA shape of the
idea rather than the Sol shape. Their own cookbook calls it approximate and
says FlexAttention's routing overhead can outweigh the saving on short
sequences. It is the vendor's engineers choosing cube geometry for this
model, which [`../morton.md`](../morton.md) has no measurement against.

**VSA-H3** (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/video_sparse_attn_h3.py`, merged 2026-09-02): the
trained sparse policy for the FastH3 weights, an in-tree Triton kernel gated
to SM90 and above, so it does not run on this card at all. Its knobs are
what to read: `vsa_mode` exempt/compete (whether non-video keys are always
kept or compete in the top-k), `vsa_dense_first_n_steps`, `vsa_dense_layers`.
Ours: sink ranges derived from the segment table play the role of `exempt`,
`dense_blocks` is `vsa_dense_layers`, and the `sigma_start`/`sigma_end`
window on `MiniMaxH3SolAttn` is the per-step knob in sigma rather than in
step index. [`../SOLATTN.md`](../SOLATTN.md) owns those.

### Breakable CUDA graphs over a packed sequence

`coderef/sglang/python/sglang/multimodal_gen/runtime/breakable_cuda_graph/model_padders/minimax_h3.py`
pads the packed prompt to the model's sequence alignment and rewrites
`cu_seqlens_q`, `max_seqlen_q` and the position ids so one captured graph
serves varying lengths. That padding layer is the part that makes graph
capture possible for a packed-sequence model at all. Note again that their own
quality gate requires this off, and that under it H3's attention core and
`_embed` are marked `eager_on_graph(True)`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py`,
read 2026-08-25): the captured region excludes the phase that dominates
sampling time on this box.

**Corrected 2026-08-25: ComfyUI does have an equivalent, and it is off for H3
by construction.** Source read, not run: `comfy/model_prefetch.py` captures one
`torch.cuda.CUDAGraph` per block inside the dynamic-VRAM prefetch queue and
replays it only when the allocator's placement signature for that block's
weights matches the capture (`vbar_signature_compare`), which is the
streaming-weights problem answered per block rather than avoided. It is wired
for LLaMA-family decode (`fixed_kv_decode` only, so never for a prompt encode),
Gemma4 decode and MiniMax Music; the H3 DiT loop
(`comfy/ldm/minimax/model.py`, the `prefetch_queue_pop` calls) passes neither
`core` nor `enable_graph`, so no H3 block is captured. `TorchCompileModel`'s
`cudagraphs` backend is the other route and it clones the model with
`disable_dynamic=True`, which on a card the DiT does not fit
([`hardware.md`](../hardware.md), measured 2026-08-17) is not a route.
What is left to gain is bounded by
`bench/results/2026-08-18_phase0_instrument.json`: sampling at 1024x768 with
three references ran at 100% SM occupancy with power pegged at the limit, and
graph replay removes launch gaps only. Untested and the one place it could
still pay: a single-frame `workflows/image/` render at a small canvas, where
the same instrument reading SM occupancy well under 100% would be the signal.
The technique is also available outside sglang as `meta-pytorch/breakable-cuda-graphs`
(BSD-3; README read 2026-08-25: `@no_graph` regions may not return CUDA
tensors, and it says nothing about weights that move between replays).

### An enforced fp32 island, which our int8 load silently collapses

**Found 2026-08-29 from the ComfyUI side**, and it is the sharpest new entry in
this file because the vendor names the exact set we lose.

sglang keeps a **named, enforced** list of tensors that stay fp32 while the rest
of the DiT is bf16 (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:144-159`):
`MINIMAX_H3_FP32_PARAM_NAMES` covers both patch projections, the time embedder
and both output heads, and `MINIMAX_H3_FP32_BUFFER_NAMES` covers
`rope.inv_freq`. It even handles the pruned case, dropping `time_embedder.*`
from the list when `adaln_t_table` is present (`:2060-2066`).
[`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) §7 records that the island is
never quantised.

ComfyUI declares the same intent and does not keep it on a quantized
checkpoint. `comfy/ldm/minimax/model.py` constructs those layers with an
explicit `dtype=torch.float32` and calls them "the checkpoint's fp32 island" in
a comment at `:302` — but `MixedPrecisionOps.Linear.__init__`
(`comfy/ops.py:1300-1303`) discards the `dtype=` its caller passed and uses the
compute dtype, so on `int8_convrot` every one of them loads bf16.
[`comfyui_h3_t2va_trace.md`](comfyui_h3_t2va_trace.md) §1.5 has the mechanism
and the verification-by-execution; the magnitudes are 3.7e-4 to 1.7e-3
relative, against the 8.8e-3 the same checkpoint's int8 blocks already carry.

**Priced, not urgent, and the reason is the same as the AdaLN cache above**: it
removes an error that is not the largest one in the stack. What makes it worth
recording anyway is that it is a *stated intention this install does not meet*,
it is a **strict** regression for `adaln_proj` (F16 on disk to bf16 in memory),
and it silently confounds any bf16-against-int8 checkpoint comparison, which
changes four things at once rather than one. **Enforced by nothing** — no check
asserts our DiT's fp32 set against the vendor's named list, and nothing would
notice if core changed it again.

### Text-encoder precision, where ours is the more conservative one

Recorded so this file is not read as a list of places we are behind. sglang runs
the Qwen3-VL encode in **bf16**
([`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) §4). ComfyUI upcasts the
embedding to fp32 (`comfy/sd1_clip.py:213`) and never comes back down, and
`comfy/sd.py:269-270` sets the patcher's compute dtype to match with the comment
"Match torch.float32 hardcode upcast in TE implemention". So the whole 50-layer
stack runs fp32 activations here and bf16 there.

That is also why an int8 encoder never reaches an int8 GEMM in ComfyUI, which is
upstream policy rather than an oversight — `25022e0b` (2025-11-24) replaced an
explicit `fp8_matrix_mult=False` with today's `full_precision_mm=True`. Measured
consequence on this box: int8 costs no more time than bf16 per resident layer
(0.78 ms dequant against a 0.87 ms cast) and halves the PCIe transfer that
actually dominates (10.87 ms against 21.75 ms), for 0.88% weight error.
[`comfyui_h3_t2va_trace.md`](comfyui_h3_t2va_trace.md) §2.4 owns those numbers.

### Refusals at admission rather than degraded service

Not speed, but the design difference that shows up most often:

- The release partition gate
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py:60-135`)
  reads `partition` from `model_index.json._minimax_h3` and raises when the
  requested task does not belong to it. A `t2va` request on the `ref2va`
  partition is refused, not served badly.
- `resolved_parallel_decode_mode()`
  (`coderef/sglang/python/sglang/multimodal_gen/configs/models/vaes/minimax_h3_video.py:42-56`)
  refuses `spatial`, `spatial_shard` and `patch` VAE decode as outside the
  released quality contract, before any large component downloads.

ComfyUI has no partition concept: a plain t2v graph loads `ref2va` and
renders. **This is the frame for the checkpoint-swap arm** — a `ref2va` loss at
plain t2v reads as *not a t2v model by its own release metadata*, not as a
defect in the checkpoint.

---

## What we both do, where ours may be the weaker version

### Video VAE precision, and why our existing note does not settle it

sglang keeps the video VAE **fp32-resident and decodes in fp16 autocast**, and
the config says why in as many words: the VAE also encodes keyframes
(`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py:59-62`,
`vae_precision: "fp32"`, `vae_decode_precision: "fp16"`).

Ours runs fp16 throughout. `comfy/model_management.py:1258-1273` returns fp16
on this card unless `--fp32-vae` is passed, and the default launch mode does
not pass it.

`<comfy>/start.sh` already carries a considered decision against `--fp32-vae`,
added and reverted 2026-08-10: measured 2-3x decode cost, no established
benefit, and the reference's own fp32 evidence was an *audio* VAE measurement
that ComfyUI already honours. That reasoning stands for what it addressed.

**What the sglang read adds is that the decision was scoped to decode, and the
vendor does not decode in fp32 either.** The vendor splits the two: fp32 for
residency and encode, fp16 for decode. So the 2-3x decode cost we measured and
rejected was never the vendor's behaviour, and the open half of the question is
the *encode* side — reference images, reference videos and keyframes, whose
whole job is identity fidelity, computed once per render rather than per step.

`--fp32-vae` cannot express that split; it forces both. So the flag is the
wrong instrument for the remaining question, and the start.sh note should not
be read as having closed it. **Enforced by nothing** — no check asserts our VAE
encode precision against the vendor's, and nothing would notice if it changed.

### VAE tiling is silent here

ComfyUI tiles under memory pressure and records nothing about having done so;
sglang treats decode mode as part of the quality contract and refuses the modes
it considers inexact (cited above). We cannot currently distinguish a tiled
decode from an untiled one after the fact.

---

## Refuted here, and worth keeping refuted

**Hypothesis, raised and killed on 2026-08-21: that the fp8-vs-int8 fidelity
gap is a qkv row-permutation defect in the repack.**

The hypothesis had a real source behind it. sglang treats the H3 qkv layout as
a load-time hazard: the release interleaves each head's Q, K and V rows, sglang
wants them concatenated, so it permutes on load
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:662-689`)
and then permutes **row-indexed quantization metadata the same way** via
`_install_qkv_row_reorder`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:179-209`),
using the row count as the gate so swizzled and per-tensor scales pass through
untouched. Its unit test states the failure mode: get it wrong and the model
"loads and runs, and renders noise"
(`coderef/sglang/python/sglang/multimodal_gen/test/unit/test_minimax_h3_qkv_scale_reorder.py:22-26`).

It dies on two independent grounds, both already in this repo:

1. **The fp8 file has no row-indexed scale to misalign.** Its `weight_scale` is
   a scalar (`bench/analyze_quant_delta.py`, the format description at the top
   of the file). No permutation of output rows can put a per-tensor scalar on
   the wrong row.
2. **The gap is uniform across module kinds, so it is not qkv-specific.**
   [`bench/results/2026-08-21_quant_delta_fl2va.json`](../../bench/results/2026-08-21_quant_delta_fl2va.json):
   `fp8_vs_bf16` reads the same for `attn.qkv_proj` as for `attn.out_proj`,
   `mlp.fc1` and `mlp.fc2`. A layout defect confined to the fused qkv weight
   would single that module out and it does not.

The reordering itself was never taken on faith: `probe_qkv_layout()` in
`bench/analyze_quant_delta.py` refuses to measure unless reordering is the
better reading, and the same record carries its verdict.

So the remaining explanation for the gap is the one the script was built to
measure: a per-tensor scalar scale against a per-output-row scale, which differ
exactly in the per-channel error distribution and nowhere in a whole-tensor
norm. **No new work is owed here.**

---

## Looks like a gap, already priced

**Step caching.** sglang ships a Cache-DiT configuration for `quality="high"`
with a measured SSIM and PSNR against lossless
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py:57-64`),
so it reads as an obvious gap. It is not. [`roadmap.md`](../roadmap.md) records
the owner decision of 2026-08-20 that step caching dies here: the speedup came
from skipping steps on a 16-step schedule, and at 4 steps there is nothing to
skip. Their number is at 50 steps on four H200s. Different regime.

**Quantization, torch.compile, layerwise offload.** All present in sglang and
all required *off* by its own quality gate, so none of them is evidence for
anything. We already run quantized weights by necessity on 24 GB.

---

## What we do that they do not

Recorded so the comparison is not read one-directionally: the SLA router has no
counterpart in sglang's H3 path. [`SOLATTN.md`](../SOLATTN.md) owns those
numbers. Their speedups and ours are not comparable and must not be put in the
same table.

**Sol-Attn and the sage kernels no longer belong in that sentence, corrected
2026-08-28 against `803b4fb31c`.** This section read "Sol-Attn, the sage kernels
and the SLA router have no counterpart ... which runs dense FlashAttention",
which was true when written and is not now: sglang ships a `sol_attn` attention
backend (`coderef/sglang/python/sglang/multimodal_gen/test/unit/test_sol_attn_backend.py`)
and a `kitchen_int8` linear path dispatching the same `comfy_kitchen.int8_linear`
our checkpoints load through.

**What that convergence is, and is not.** Their published consumer-card table's
fastest row is an int8 DiT with a sage-to-Sol hybrid — which is the stack our
graphs already wire. That is two projects arriving at one answer, and it is
worth more as corroboration than as an action item, because there is nothing
here to adopt. **Do not import their numbers.** Their PSNR column compares
different samples by our own 2026-08-18 measurement, and their own quality
section says as much; their seconds came off a host that page-caches the whole
checkpoint, so the mechanism carries and the figures do not.

### Sol-Attn defaults: sglang against core's and ours (2026-09-11)

Read from source: sglang at `593c7a900d`
([`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) section 14.9 has the walk
and the lines); core's `BlockSparseAttention` at ComfyUI `1d48d9cf`
(`comfy_extras/nodes_sparse_attention.py:368-407` for the schema,
`:310-318` for what an H3 block does outside the window). Ours is
`workflows/h3_config.py::SOL_RECOMMENDED_CUDA`, and core's defaults are
copied into `workflows/h3_config.py::SOL_CORE_DEFAULTS`.

| knob | sglang `sol_attn` | core `BlockSparseAttention` | ours, before this change |
|---|---|---|---|
| on by default | no; the DiT default is `fa` and Sol is opt-in | only in a graph that adds the node | on in every shipped video graph, except the stems in `bench/check_attention_defaults.py::SOL_EXEMPT_STEMS` |
| dense at the start | the first 10 steps (`dense_steps`), a count | the first fifth of the sigma schedule (`start_percent` 0.2) | same as core |
| dense at the end | none: sparse through the last step | none at its default `end_percent` 1.0 | the last step, through `end_percent` 0.9, `SOL_END_PERCENT_BY_STEPS` for the distilled step counts, and `SOL_PDD_OVERRIDES` for PDD |
| tau | 1.0 | 1.3 | 1.0 |
| dense blocks | 0 and 1 (and, by the name match, both token-refiner blocks) | none | none |
| sink | none by default (`sink_tokens` 0) | `exact_kv_and_rows` | `exact_kv_and_rows` |
| token routing | none | `extra_tokens` 256 on every block | off |
| outside the window | FlashAttention, or Sage in the documented recipe | the block's own attention, whatever the model already runs | Sage, which Sol chains onto |

*2026-09-19: the last cell is no longer true. Since 2026-09-15 the shipped
graphs chain Sol onto kitchen's dense attention through core's Model
Attention Backend, not onto Sage; `workflows/h3_config.py::DEFAULT_DENSE_CHAIN`
names the chain and its comment block says what each chain runs. The other
cells of our column still match `SOL_RECOMMENDED_CUDA`, the knobs added to it
since (`qk_balance`, `rotate`, `morton`) aside.*

The start row agrees only on a fifty-step grid, where 10 steps is a fifth.
At our sixteen-step base, sglang's count would keep 10 steps dense against
the 4 that `start_percent` 0.2 keeps (the `start_percent` comment in
`workflows/h3_config.py::SOL_RECOMMENDED_CUDA` has that arithmetic).

**Decision, owner, 2026-09-11.** This is a tinkering repo: where sglang and
ComfyUI's own node agree on a default and ours differs, ours takes theirs,
without waiting on an eval of our own. The one knob where both agree against
us is the end of the window, so ours moves to sparse through the last step,
`end_percent` 1.0, for base, distilled and PDD graphs alike. It lands with the
next graph rebuild, and `CHANGELOG.md` carries it. The other rows are not
adopted: on tau, dense blocks, the sink and token routing, sglang and core
disagree with each other, and on the start they agree only at a step count
we do not run.

---

## The 768 cap, and what it did to our code

The open release is 768p on the short edge. Establishing what *kind* of limit
that is changed one doc claim and found one defect; the finding itself is owned
by [`h3_resolutions.md`](../h3_resolutions.md) and is not restated here. The
short version for anyone arriving from a performance question: the area cap is
hard upstream, the short edge only warns, and neither is a property of the
weights.

Auditing this repo against it (2026-08-21, source read) came out mostly clean.
`resolution.py` already grades a canvas by whether `adapt_canvas` is a fixed
point on it and labels anything outside the trained family rather than refusing,
which is the right posture given what the cap turns out to be. `reference_fit.py`
keeps the 2048 reference short edge and the 768 target cap properly separate — the two knobs this repo has confused
before — and `h3_config.py` has no target short-edge constant to confuse.

**One real defect, found and fixed.** `bench/preflight_graph.py` priced a
reference image that is *not* fed through `MiniMaxH3ReferenceFit` as if it were
upscaled to a 2048 short edge. Core does the opposite: it clamps with
`min(1.0, ...)` in both sizing modes (`comfy_extras/nodes_minimax_h3.py:297-301`)
and never enlarges. The over-count is the square of a scale the reference never
gets — a 1024x1024 reference priced at four times its real row count. No shipped
API graph reaches that branch, because they all wire the fit node; the exposure
is exactly the hand-built graph that `CLAUDE.md` promises preflight can price.
Fixed by defaulting the no-fit case to no upscale, and confirmed non-inert
against a graph rewired to bypass the fit node.

---

## Open after this read

- The VAE **encode** precision question, above. **Half of this closed on
  2026-08-21, after the sentence below was written**: the instrument exists.
  `bench/grade_vae_encoder_precision.py` grades the encoder at the call rather
  than at a rendered clip, and
  [`bench/results/2026-08-21_vae_encoder_precision.json`](../../bench/results/2026-08-21_vae_encoder_precision.json)
  records fp16 bit-identical to itself against fp32 moving the latent, with
  bf16 as the far control. What stays open is whether that delta is *visible*,
  and the variable below, which the instrument does not separate.
  `--fp32-vae` remains the wrong flag because it forces both halves. **It now has a second
  variable tangled with it, found 2026-08-21**: sglang *samples* the released
  posterior under a seed pinned at 42 for keyframes and reference video
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/keyframe_encoding.py:30`,
  `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:613`) where ComfyUI takes the mean
  (`comfy/ldm/minimax/vae.py:685`). Any encode instrument has to separate
  precision from mean-versus-sample or it measures neither.
  [`h3_references.md`](../h3_references.md) carries it in the vendor image
  path table.
- Whether an AdaLN cache is worth building, which is downstream of
  [`open_experiments.md`](../open_experiments.md) #22 and should not be
  started before it.
- Nothing left on the reference path: video, audio, frame rate, soundtrack
  pairing and condition noise were all re-derived on 2026-08-21 against sglang,
  diffusers and DiffSynth-Studio, and [`h3_references.md`](../h3_references.md)
  carries the results. One item there is marked unverified and stays that way:
  what sglang does when a reference video is shorter than its aligned target.
  The mono-upmix question that sat beside it was closed on 2026-08-21 — no
  upmix happens and the packed layout raises — and `h3_references.md` carries
  it under Known limitations.

## Third read, 2026-09-04

[`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) section 14 records what
landed in sglang between 2026-08-31 and 2026-09-05. This section says what
each item is against what we do, in the order a reader deciding what to borrow
would want, and it changes no earlier verdict on this page.

**FastH3 is a different model, and sglang treats it as one.** Its pipeline
config refuses every mode, task, step count and quality tier the student was
not distilled for. That is the discipline any FastH3 rung here would need to
inherit: a t2va-only arm at five grid points, its own dense pair, never pooled
with base-H3 rungs. [`vsa/vsa_node.md`](vsa/vsa_node.md) holds the blockers.

**Their quality tiers are our ladder, stated as a contract instead of
measured.** `lossless`, `extra-high` and `high` are request-time promises
enforced at admission; our dense, sage and Sol rungs are renders judged blind.
For H3 the middle tier is empty on their side, which says the same thing the
ladder verdict says on ours: on this model the kernel arithmetic is not where
the quality goes. Their definition of "lossless" also names the exception
explicitly, `torch.compile`, and refuses it as ground truth; ours names the
Comfy Compiler's malloc graph as the regime and checked one rung against it
(`bench/results/2026-09-04_stairwell_dense_retime.jsonl`).

**Silent fallback is now refused on their side too.** An explicit
per-component attention backend must be consumed or the server does not
start, and the SubBlock README's warning that `transformer=subblock_sparse_attn`
"appears to work and silently does nothing" is the failure this repo's chain
assert and `bench/check_attention_defaults.py` exist for. Two projects
arriving at the same guard; nothing to import.

**The AdaLN cache verdict stands, with a new hazard beside it.** The tiered
host cache saves a checkpoint re-read on a plan miss, which a single-graph
ComfyUI render never pays, so "no speed on the pruned file, unpruned
accuracy at pruned memory" is unchanged. What is new is that they now fail
closed on a LoRA that touches `adaln_proj` under either cache mode, after
finding such deltas were silently dropped. Whether any LoRA this repo loads
touches those weights is a question for `bench/check_pdd_head_selection.py`'s
owner, not settled here.

**The cube schedule is a step policy from the vendor's own engineers.** The
recommended `topk_ratio_list` for the fifty-step grid keeps the first two
updates dense and decays from there; roadmap step 5 (the window's start) has
so far had only our own probe trend to reason from. It is a prior, not a
measurement on our stack, and cube geometry is not Morton order.

**SpargeAttention is adaptable and not urgent.** It runs on this card's
architecture with one knob and no per-model tuning, so it could be an arm
against Sol; their own measurement on another model found no single-GPU
speed or memory win and a large perceptual change, and they call it
approximate even at full keep. Below Sol's block policy in priority.

**Block-FP8 is the qkv reorder hazard again.** A standard block-quantized
export loaded clean and rendered blank until the scale-row permutation was
made block-aware. Our refuted-hypothesis section above already established
that our fp8 file has no row-indexed scale to misalign; this is corroboration
that the hazard is real for formats that do, not evidence about ours.

**Everything else is serving or other silicon**: warmup at the served shape
(our runner's `--warmup` row is the same idea), profiler spans, SM120 paths,
the SM12.x decoder workaround, key masks under Ulysses, third-party bundle
loading. Read, priced, no action.

## Fourth read, 2026-09-10

What landed in `coderef/sglang` between `320bdd1ee2` (the third read) and
`887c401e15`, against what we do. It changes no earlier verdict on this page.
The two Sol-side items, SubBlock's new Sage compute and its router's note on
reserved sink blocks and the forced diagonal, are recorded in
[`../sol_upstream.md`](../sol_upstream.md) and not repeated here.

**"Singularity" is a third-party checkpoint, not a vendor variant.**
`65400bb420` (#38455) serves `WarmBloodAban/Minimax-h3_Singularity`, an FL/Ref
fusion fine-tune shipped full or pruned INT8, through `--model-variant hybrid`
plus an explicit transformer-weights path: one pipeline for t2va, fl2va and
ref2va, with the partition gate relaxed (sglang's H3 cookbook page, its
Singularity section). ComfyUI has no partition gate
([`../comfyui_vendor_gaps.md`](../comfyui_vendor_gaps.md) gap 10), so such a
file loads here as it is. Nothing to borrow.

**Two fixes that do not apply to core.** `b83f1bdd21` (#38225) removed
`@torch.jit.script` from the audio VAE's snake activation, because the
profiling JIT changed its rounding after the first call and a repeated
reference-audio request stopped matching the first. Core's `snake` is plain
eager code (`comfy/ldm/minimax/audio_vae.py`). `a8e45f16cc` (#38506) adds a
shared INT8 embedding lookup for INT8, W4A8 and NVFP4 encoders; the encoder
this install loads keeps `model.embed_tokens.weight` in BF16 (read from the
header of `h3_config.MODELS["clip"]`, 2026-09-10).

**Skip-softmax is other silicon.** `0ea8378085` (#37959) adds a
request-scoped skip-softmax attention through FlashInfer's TRTLLM kernels,
dispatched only for compute capability 9.0 and 10.x
(`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/skip_softmax.py`).
It is one global threshold with a dense warm-up, the same blunt shape as a
single tau, and nothing of it runs on this card.

**Everything else is serving or other silicon**: XPU (#33366) and Xeon CPU
(#35147) enablement, GB300/GB200 and DGX Spark recipes (#38296, #37456),
per-phase warmup memory for residency calibration (#37916), and a docs sync
(#38784). Read, priced, no action.

## Fifth read, 2026-09-11

What landed in `coderef/sglang` between `887c401e15` and `593c7a900d`. It
changes no earlier verdict on this page, and none of it touches an H3 file.

**The RoPE refactor does not reach H3.** `d0035da34e` (#33555) moves DiT
RoPE onto a shared `RotaryEmbedding` custom op and edits the shared qk-norm
helpers (`runtime/layers/rotary_embedding/base.py`,
`runtime/layers/layernorm.py`) for Flux, Qwen-Image, GLM-Image and others.
H3's DiT imports neither: it keeps its own `MiniMaxH3Rope` and
`_apply_qk_norm` and calls `sgl_kernel`'s rotary directly
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py`).

**Everything else is serving**: in-place pinning for layerwise-offload host
stores (#39021), IPC JIT recovery after an interrupted build (#39034), and
LLM-side work. Read, priced, no action.

## Sixth read, 2026-09-19

What landed in `coderef/sglang` between `593c7a900d` and `993d1fccba`. It
changes no earlier verdict on this page. Of the commits touching H3 files,
three matter here.

**The RTX 5090 recipe (`a25f213bc4`, #39373) is documentation, and it sets
memory placement only.** The command in
`coderef/sglang/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx` (section on
the RTX 5090) picks the allocator, the fl2va variant, the `memory`
performance preset, what is offloaded and how many DiT layers stay resident.
The preset sets offload and residency defaults only
(`coderef/sglang/python/sglang/multimodal_gen/runtime/server_args/auto_tune.py`);
it chooses no attention backend, quantisation, compile or step cache. The
sampler settings on that page describe the benchmark run on both engines, not
a new sglang default. So the adopt-upstream rule has nothing new to act on:
the Sol table above still holds (sglang's `sol_attn` defaults and core's node
schema have not moved since 2026-09-11), and on canvas, steps, sampler and
text-encoder precision sglang and core still disagree with each other. Two
things on that page are easy to misread:

- **"kitchen int8" in its 24 GB row means INT8 for the DiT's Linear layers**
  (the page says it changes Linear numerics only). This repo's "kitchen" chain
  is an attention kernel. Same word, different stage.
- **The recipe's SDPA attention is a property of SM120**, where sglang
  defaults to SDPA (`coderef/sglang/python/sglang/multimodal_gen/runtime/platforms/cuda.py`);
  the same command on SM89 gets FlashAttention. Its resident-layer count and
  bf16 DiT are for the 32 GB card.

The same page withdraws its earlier physical-4090 step time as an artifact of
an old pinning path and replaces it with a derived figure for a machine it
has not re-measured. Nothing in this repo quoted the withdrawn figure.

**VDN-H3 (`ff1ce11348`, #37903) is a different checkpoint, not a backend for
ours.** OpenVDN's `vdn-minimax-h3` adds a trained linear-attention branch and
an 8-step DMD2 LoRA to the H3 backbone, t2va and fl2va only. Its
`hybrid_window_attn_h3` backend replaces video-to-video self-attention with
an exact gated softmax over a window of neighbouring frame chunks (first and
last frames, text and audio rows dense) plus a bidirectional linear branch
for the rest
(`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/hybrid_window_attn_h3.py`,
`coderef/sglang/python/sglang/multimodal_gen/configs/models/dits/minimax_h3_vdn.py`).
The linear branch is trained, so it does not transfer to the base weights.
The 2026-09-17 rotation survey already lists it. `42b5af8c62` refreshes its
cookbook numbers.

**Everything else is serving, other models or docs**: out-of-tree platform
support, CFG and tracing docs, DSV4 and Qwen work, ROCm and XPU. Read at the
title, priced, no action.
