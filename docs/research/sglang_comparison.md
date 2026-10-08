# sglang's H3 serving path against ours

last updated: 2026-10-08 (closing section "Ninth read" added: frozen audio
and video inputs, with a dated note in the eighth read); 2026-10-02 (closing section "Eighth read" added, with dated
notes on the seventh read's ComfyUI sentence, the `kitchen_int8` name and our
dense-blocks cell); 2026-09-25 (closing section "Seventh read" added); 2026-09-19 (closing section "Sixth read" added, and a dated
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
our checkpoints load through. *2026-10-02: `kitchen_int8` is now a deprecated
alias of `convrot_int8` (`e667ab10d5`), which still dispatches to kitchen on
this card; see "Eighth read".*

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

*2026-10-02: the dense-blocks cell of our column is no longer "none". The Sol
node's `dense_blocks` default is `sol_attn_h3.py::SOL_DENSE_TAIL`, mirrored by
`workflows/h3_config.py::SOL_DENSE_TAIL`, so on that row sglang, core and ours
now all differ. sglang's and core's cells were re-read at `89f21671bb` and
`65787d66` and have not moved.*

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

## Seventh read, 2026-09-25

What landed in `coderef/sglang` between `993d1fccba` and `2f5c9ac43d`. Four
commits touch H3 paths. One of them is news.

**sglang now implements PDD (`973fb44471`, #40568).** Until this commit no
engine did ([`pdd/pdd_implementations.md`](pdd/pdd_implementations.md), section
1, corrected in place). How it works:

- Two offline tools, then an environment variable at serve time. The first
  merges the alibaba-pai adapter into the release's unpruned native
  transformer and copies the raw head stacks out
  (`coderef/sglang/python/sglang/multimodal_gen/tools/build_minimax_h3_pdd_weights.py`).
  The second fuses each block of heads into one head per step with weights
  proportional to each sub-interval's `dsigma` on the shifted grid, computing
  in fp32 and storing bf16
  (`coderef/sglang/python/sglang/multimodal_gen/tools/fuse_minimax_h3_pdd_heads.py::fuse`).
  `SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS` points the final layer at the fused
  file (`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3FinalLayer.load_pdd_fused_heads`).
- The step index is the denoise loop's counter, set per step, not recovered
  from `t` or sigma. The timestep stage raises unless the request's sigma grid
  equals the grid the heads were fused for
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/timestep_preparation.py`,
  `_apply_pdd_schedule`).
- The cookbook recipe
  (`coderef/sglang/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx`, "PDD
  acceleration LoRAs") counts sigma points, so its step count is eight
  evaluations. It uses the shifts the fusion tool defaults to and the default
  attention backend, which is not Sol. The fl2va adapter serves t2va and
  fl2va; ref2va has its own.

**Where it agrees with ours:** the fusion formula (`pdd_math.fusion_plan`),
the shifts (`h3_config.PDD_SHIFT`), the evaluation count (`h3_config.PDD_STEPS`), Euler at eta 0, reading alibaba-pai's raw
head stack as absolute heads rather than deltas, and failing closed on an
off-grid request. **Where it differs:** one uniform partition fixed when the
heads are fused, where ours fuses lazily from the sampler's sigmas and takes
any schedule. Also a loop counter for the step index where ours matches
`t_emb`, no strength or head-off control, the backbone merged offline into
unpruned BF16 weights where ours patches the pruned INT8 checkpoint or uses
the bake, and dense attention where our PDD graphs run Sol. That last one is
one upstream alone, so the adopt rule does not fire.

**A probable defect in its offline builder: the fc1 merge does not swap gate
and value.** The builder maps diffusers `ff.net.0.proj` onto native
`mlp.fc1` as a plain rename. The reader compared the two layouts in the
local release. Native `fc1` equals the diffusers weight with its halves
swapped, and does not equal it in the same order. sglang's own runtime LoRA
path swaps (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/lora/pipeline.py::_swap_peft_swiglu_fc1_lora_b`),
and so do its checkpoint loader and our converter. The builder's only test
merges `to_out` alone. The layouts were measured; the merge was not run. So
the fc1 delta landing swapped is an inference, not an observation.

**The other three:**

- `abef3efb64` (#40378) fuses SwiGLU for quantized H3 MLPs with a kernel that
  rounds SiLU to bf16 before the multiply, the same rounding as its eager path
  and as core's eager `comfy/ops.py::_swiglu_eager`. It is a fusion, not a new
  convention. Whether kitchen's fused INT8 SwiGLU rounds SiLU internally is
  not established here: its CUDA source is not in the wheel.
- `2a0cb2f04e` (#40116) extends SubBlock's `sage_fp8` to SM120; the note is
  under "sglang's SubBlock router" in
  [`../sol_upstream.md`](../sol_upstream.md). No SM89 path.
- `e6931ca889`, reverted by `f702a0be29`, re-landed as `ce06a14444`: each
  model's pipeline config now registers from its own file
  (`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::register`).
  Every `coderef/sglang/...` path and `::symbol` cited under `docs/` still
  resolves. The FastH3 sentence in
  [`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) that says "Registered under
  `registry.py`" is half stale and has a dated note.

**Sol-Attn defaults have not moved**: no new commit touches `sol_attn.py`,
and it is still opt-in, so the table under "Sol-Attn defaults: sglang against
core's and ours" holds. `50ec9702d0` adds a "Run in ComfyUI" section to the H3
cookbook naming a server-mode node, which agrees with
[`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) that sglang has no in-process
ComfyUI path for H3. The rest is serving, MiniMax-M3, HiSparse, ROCm and XPU,
read at the title.

*Corrected 2026-10-02: since `f1e62e3a2e` sglang has a ComfyUI path for H3
that runs the DiT under a ComfyUI graph, one step at a time in a separate
worker process. "No in-process path" is still literally true; "server mode
only" is not. See "Eighth read".*

## Eighth read, 2026-10-02

What landed in `coderef/sglang` between `2f5c9ac43d` and `89f21671bb`. Six
commits touch H3 paths. One retires a sentence in the seventh read, and the
re-read retires a sentence in [`../wiki/references.md`](../wiki/references.md)
that was already wrong when written.

**No default moved, and the adopt-upstream rule does not fire.**
`sol_attn.py` is unchanged, so the table under "Sol-Attn defaults" holds
(with the dated note on our dense-blocks cell). The H3 sample config still
runs fifty steps with guidance pinned at 1.0. The integrated mode's shifts
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/comfyui_step.py`,
`DEFAULT_SIGMA_SHIFT_VIDEO` and `DEFAULT_SIGMA_SHIFT_AUDIO`)
equal core's `comfy/supported_models.py::MiniMaxH3.sampling_settings` and
`h3_config.SIGMA_SHIFT`, which already agree. Its example graphs use
`res_multistep` on `simple`. They copy core's templates, so they echo one
upstream rather than adding a second one.

**sglang now runs the H3 DiT under a ComfyUI graph (`f1e62e3a2e`, #35990).**
The loader builds core's model object from the safetensors header, then swaps
its diffusion model for an executor that ships each sampler step over local
ZMQ, with CUDA IPC handles, to an sglang worker
(`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/core/generator.py::load_model`,
`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/minimax_h3.py::MiniMaxH3Executor`).
The worker runs one forward per call
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/comfyui_step.py::MiniMaxH3ComfyUIStepStage`).
ComfyUI keeps the encoder, both VAEs, core's H3 conditioning nodes, the packed
layout and the sampler. Two consequences:

- **None of our DiT-side patches would reach it** (reasoned). The diffusion
  model is replaced, so the Sol node, the dense attention chain, PDD heads in
  core's `final_layer` and core's LoRA loader all patch a model that no longer
  runs. sglang ships its own LoRA loader node for this mode.
- **Two places read as departures from core's `MiniMaxH3Model.forward`**
  (reasoned, not run). `MiniMaxH3Adapter.unpack` scales the audio velocity by
  `time_shift_slope` on top of the carry correction, which core does not, and
  it does not multiply the output by the denoise mask, which core does. Its
  only audio test checks shapes. If anyone compares this mode against a stock
  graph, these are the first two places to look.
  *2026-10-08: this bullet says what the adapter leaves out and not what the
  worker does with a mask, which is in the ninth read below. Both
  departures were read again at `214347891a` and stand.*

**Spectrum skip-step (`ae47bcd4da`, #35684) is opt-in**
(`SamplingParams.enable_spectrum` is False). It forecasts the
pre-final-layer hidden state of the target audio and video rows with a
Chebyshev ridge fit, holds audio at its last real value, blends video toward
the forecast, and runs the embed step and the final layer every step
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py::MiniMaxH3DiTModel._h3_spectrum_predict_targets`,
`coderef/sglang/python/sglang/multimodal_gen/runtime/cache/spectrum.py::SpectrumMixin.begin_spectrum_step`).
The skip schedule is a rule: real warm-up forwards, then a widening window
with no cooldown. That is DPCache's insertion point with a rule where DPCache
has a calibration. It already forecasts only the target rows, which
[`2026-09-25_step_caching_survey.md`](2026-09-25_step_caching_survey.md)
said no upstream did, and holds audio rather than weighting the two streams.
The survey's verdict stands (reasoned): nothing to skip at distilled step
counts, a second candidate at the base step count. Its code comment credits
"Comfy / Wan2GP", probably xmarre's Spectrum pack
([`../sol_upstream.md`](../sol_upstream.md), "Other ComfyUI Sol-Attn
packs"); not verified.

**The same commit's "fused RMSNorm/AdaLN" landed unfused.**
`_modulate_rmsnorm_scale_shift` is the norm followed by `_modulate_scale_shift`,
and its docstring says the fused kernel was removed for drift. On the block
path, bf16 on CUDA,
`coderef/sglang/python/sglang/kernels/ops/diffusion/modulate/indexed_modulation_triton.py::_indexed_scale_shift_bf16_kernel`
rounds to bf16 at `1+scale`, at the product and at the stored sum. Core's
`comfy/ldm/minimax/model.py::_mod_scale_shift` is three in-place bf16 ops at
the same three points. That kernel has not changed since 2026-08-18
(`git log` on the file), so the seventh-read-era claim in
[`../wiki/references.md`](../wiki/references.md) that sglang keeps the affine
in fp32 and core rounds at more points was wrong when written; it is corrected
there. vllm-omni's fp32 affine is the outlier of the three.

**`kitchen_int8` is now a deprecated alias of `convrot_int8` (`e667ab10d5`,
#38040).** Same quantization as kitchen's, with interchangeable weight and
scale tensors
(`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/quantization/configs/convrot_int8_config.py`).
`auto` picks its own JIT CUTLASS kernels on compute capability 9.0, 10.0, 12.0
and 12.1 and `comfy_kitchen.int8_linear` everywhere else, so on this card it is
still kitchen, and the `int8_convrot` files in `h3_config.MODELS` load
auto-detected. Nothing to adopt.

**The rest:**

- `dc1bd46802` (#40470): an exact, bounded conditioning cache, on by default,
  that also stores the H3 VAE encode's moments before posterior sampling, so
  seeded sampling stays exact
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/cache/conditioning.py`).
  ComfyUI's node cache covers the same ground here.
- `0931a72eb2` (#34365): RL rollout for t2va only, a stochastic video-target
  update that returns log-probs, audio deterministic, off by default.
- `84622ce9d5` (#41272) is a crash fix for LoRA-wrapped linears and
  `bf8adf9602` puts int64 offsets in the SubBlock router kernels.
- Everything else is Qwen-Image, Flux 3, other models, NPU, MiniMax-M3 and
  docs, read at the title.

## Ninth read, 2026-10-08

A scoped read, not a walk of commits. The owner asked whether sglang does
anything with frozen audio, or anything special with a video that is edited,
continued or rendered in windows. Read at `214347891a`, the clone's head on
the day, in the H3 stage folder and the files named below; what was not read
is listed at the end. It is not a read of what landed since the eighth
read's `89f21671bb`. No renders.

**The answer is no, in sglang's own pipeline.** Each of these was already on
record and holds at this head:

- **A source track that is reused is a reference audio block and nothing
  else.** The target's audio starts as noise
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/latent_preparation.py::MiniMaxH3LatentPreparationStage`),
  is updated like any target row while the reference rows stay pinned
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/denoise_loop.py::minimax_h3_denoise_loop`),
  and the file's track is the decoded latent, with no mux of the source's
  audio (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py::MiniMaxH3DecodingStage`).
  [`../h3_audio_freeze.md`](../h3_audio_freeze.md) section 8 owns this; its
  dated note says which of its items were checked again.
- **There is no edit, continuation, window or partial-strength path.** One
  request is one clip inside the duration limits. The newest file in the
  folder with a promising name, `minimax_h3_rollout.py`, is the t2va RL
  rollout of the eighth read.
- **Sampling does not differ by task.** Every task row carries the same
  shifts
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py::MINIMAX_H3_TASK_PROFILES`),
  and guidance and the negative prompt are refused at the door
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/video_adapter.py::MiniMaxH3VideoModelAdapter`).
  [`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) sections 1 and 8 have the
  detail.

**What moved: the stage that runs a DiT step for ComfyUI handles denoise
masks.** The eighth read described that stage and left this out. ComfyUI's
side forwards the video mask and the audio mask with each step
(`coderef/sglang/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/executors/minimax_h3.py::MiniMaxH3Adapter`),
and the worker turns them into per-row timesteps with core's own formula, on
target rows only
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/comfyui_step.py::comfyui_payload_to_branch_inputs`,
`::_overlay_per_row_timesteps`). An audio mask that is one everywhere is
ignored (`::_audio_mask_values`), as in core. So a row frozen by a mask is
labelled clean there as it is under core's
`comfy/ldm/minimax/model.py::MiniMaxH3Model`. What the adapter still does not
do is scale the returned velocity by the mask, which core does; the eighth
read's bullet on that stands, so the labels agree and the output scaling
does not (reasoned, not run). This is core's rule carried over so a ComfyUI
sampler can drive sglang's DiT. Nothing in sglang's request, task table or
native loop reaches it, so it is no evidence about training.
[`../h3_audio_freeze.md`](../h3_audio_freeze.md) section 8 said no mask path
existed anywhere in the serving code; it is corrected there.

**How its edit-shaped request differs from the whole-frame edit rendered
here** (a reference video at the canvas size, the source's audio frozen in
the target rows by a mask, a long clip in windows whose head frames are
frozen):

- **A sounded reference video always brings its soundtrack as an audio
  block, with its own label ahead of the video's.** The task table routes
  the video into the audio encoder with no switch
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py::MINIMAX_H3_TASK_PROFILES`),
  and the label is emitted whenever the file has an audio stream
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py::MiniMaxH3TextEncodingStage`).
  Here the soundtrack is a block only when it is wired. A reference video
  with no audio block beside a frozen target track is therefore a state
  sglang cannot produce. The arm that carries both is the hybrid row of
  [`../h3_audio_freeze.md`](../h3_audio_freeze.md) section 2.
- **Its only way to pin a frame of the target is a still keyframe at the
  first frame, the last, or both**, and ref2va admits those beside its
  references
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/task_profiles.py::MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES`).
  Core's layout places a keyframe or guide at any frame index
  (`comfy/ldm/minimax/model.py::PackedLayout`), and the window carry here
  freezes frames by mask, which is ComfyUI's mechanism and not the
  vendor's.
- **A reference video is resized to its own shape at the vendor's short
  edge, whatever the target's, and cut to the target's frame count.** The
  shape comes from the same resolver the canvas uses, called with the
  constant and not the request's short edge
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/prequeue.py::minimax_h3_prepare_for_queue`,
  `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/resolved_plan.py::minimax_h3_resolve_spatial_shape`);
  one ffmpeg pass sets the rate, the scale, the start offset and the frame
  cap
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py::minimax_h3_decode_reference_video_frames`).
  A source with the canvas's aspect ratio, handed in at the canvas size and
  the window's own length, is what that path would have made of it.
- **A request may leave the duration out and take it from its one
  audio-bearing reference**, snapped to the frame grid
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/prequeue.py::_resolve_deferred_temporal_shape`).
  Length is set by hand here.

The rest belongs to the reference comparison and is written up where that
lives, [`../h3_references.md`](../h3_references.md): where a reference
video's rows sit against the target's in time and space ("Edit a source
video"), and the condition noise level as a setting that both sides have,
set here by one node since `c7fcae2f` ("The vendor image path, stage by
stage").

**No default moved, and the adopt-upstream rule does not fire.** Nothing in
this read is a default of ours that sglang and core agree against.

**Not read:** `stages/visual_encoding.py`, `stages/replica_broadcast.py`,
`keyframe_encoding.py`, `packed_tokens.py`, `release_metadata.py`, most of
`video_adapter.py` and `resolved_plan.py`, the DiT forward beyond its mask
lines, the scheduler, the release's video processor config, and the ComfyUI
app beyond the H3 executor's pack and unpack. What happens to a reference
video shorter than the target is still untraced
([`../h3_references.md`](../h3_references.md), "Known limitations,
collected"). Line-number citations in the older sections of this file and
of [`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) were not re-verified;
several have moved.
