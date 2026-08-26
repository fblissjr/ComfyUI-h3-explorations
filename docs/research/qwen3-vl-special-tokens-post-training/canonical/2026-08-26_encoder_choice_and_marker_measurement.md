# What the AWQ lane could never have moved, what an encoder costs to hold, and how the marker question gets answered without a trainer

**Status:** Authoritative facts; the two recommendations at the end are open for
the owner
**Recorded:** 2026-08-26
**Reading it against:** [`2026-08-25_v2_launch_record.md`](2026-08-25_v2_launch_record.md)
owns the Gate 5 verdict and the four-encoder table and is not restated here.
This record answers three questions that came after it and were not in it.

## 1. The v2 calibration could not have moved either thing the lane is named for

The lane is `qwen3-vl-special-tokens-post-training`. The v2 run was a W4A16 AWQ
calibration. Those are not the same object, and the gap is structural rather
than a matter of the metric Gate 5 chose.

**SOURCE.** The quantization scheme targets `["Linear"]`.
[`bench/h3_awq_recipe.py`](../../../../bench/h3_awq_recipe.py) says why the
embedding needs no ignore entry -- it is an `nn.Embedding`, never a target --
and its boundary assertion checks that the embedding and the output head carry
no scheme rather than trusting the pattern list. The vision tower, patch merger
and DeepStack mergers are excluded the same way, recovered by module class.

**MEASURED.** The assertion is not what settles it; the holdout captures are.
In [`2026-08-25_v2_holdout_layer50.json`](../../../../bench/results/2026-08-25_v2_holdout_layer50.json)
the `layer0_input` relative L2 is exactly **0.0** on every row, both geometries,
both W4 arms. The token embedding table and the entire vision path are
byte-identical across BF16, v1 and v2. Whatever v2 was, it was not a different
vision encoder and it did not touch the seven H3 marker rows.

**INFERENCE, from those two.** The only thing that differed between v1 and v2
was the 4-bit approximation of the decoder linears. Calibration data can shift
where that error lands; it cannot teach the model anything, and it cannot reach
the untrained marker rows or the vision tower at all. Grading it against BF16
was correct for the object that was built -- a quantization's ceiling is the
full-precision model -- and irrelevant to the objective in the lane's name. Both
hold at once.

**MEASURED, and it closes the one lever that did exist.** Calibration data
could in principle bias the scale search toward H3's schema positions. The pool
cannot supply them: in
[`2026-08-25_v3_selection_max_no_upscale.json`](../../../../bench/results/2026-08-25_v3_selection_max_no_upscale.json),
`<d>` and `</d>` occur in 505 of 1,028 pool rows and the other five H3 markers
in **zero**, while the whole label scaffold -- `<Picture i>:`, `<Video k>:`,
timestamps -- is 952 of 265,922 tokens. A scale search cannot be biased toward a
schema that is a third of a percent of the token mass and missing five of its
seven markers.

**Consequence.** Do not propose calibration data as a route to marker,
prompt-structure or vision-encoding alignment. `active_plan.md` already scoped
the v2 run narrowly ("do not... begin special-token training as part of this
work"); what nobody wrote down before the run is that no calibration population
could have reached those questions in principle. That is the sentence this
section exists to make available.

## 2. What each encoder costs to hold, by component

**MEASURED, 2026-08-26.** Producer
[`bench/measure_encoder_footprint.py`](../../../../bench/measure_encoder_footprint.py),
record [`2026-08-26_encoder_footprints.json`](../../../../bench/results/2026-08-26_encoder_footprints.json).
Safetensors headers only; no tensor data read. The **H3 path** is decoder layers
0--49 plus the embedding table and the vision tower, with later layers and the
output head dropped as `comfy/text_encoders/minimax.py` drops them on load. It
is weights only: activations, offload staging and allocator overhead are not in
it.

| encoder | on disk | H3 path | decoder | embeddings | vision tower |
|---|---:|---:|---:|---:|---:|
| BF16 release | 62.13 | 47.97 | 45.41 | 1.45 (BF16) | 1.11 (BF16) |
| INT8 ConvRot | 25.28 | 25.28 | 22.72 (I8) | 1.45 (BF16) | 1.11 (BF16) |
| NVFP4 AWQ | 14.61 | 14.61 | 12.78 | 0.73 (I8) | 1.11 (BF16) |
| W4A16 AWQ v1 | 18.99 | 14.27 | 11.71 | 1.45 (BF16) | 1.11 (BF16) |
| W4A16 AWQ v2 | 19.00 | 14.27 | 11.71 | 1.45 (BF16) | 1.11 (BF16) |

GiB throughout. Read beside the fidelity column of the four-encoder table,
which the launch record owns.

Three things this table says that file size did not:

- **The encoder of record does not fit a 24 GB card.** INT8 ConvRot's H3 path is
  the whole file: it already carries only layers 0--49, so there is no depth
  slack to find.
- **The vision tower is 1.11 GiB, under five percent of every artifact.** There
  is nothing to win by compressing it and it is the part the lane cares about.
- **The embedding table is BF16 in three of the four artifacts.** NVFP4 already
  ships it at I8 for 0.73 GiB, so the format is proven in a file already on the
  box.

**MEASURED, and it bounds what fitting is worth.** The occupancy instrument on a
real arm-B render
([`2026-08-25_refview_b_qwen2048_int8_occupancy.json`](../../../../bench/results/2026-08-25_refview_b_qwen2048_int8_occupancy.json))
put the INT8 encode at 59.6 s of a 691 s render, peaking at 20,470 MiB with
95.8% mean GPU utilization and 22.3% mean memory-interface. ComfyUI streams the
oversized file layer by layer and the per-layer compute hides the transfer, so
the encode is compute-bound, not streaming-bound. A perfect fit therefore wins
at most that 59.6 s.

## 3. Recommendation: no further own-calibration quants for this card

**RECOMMENDATION, owner's call open.** For a 24 GB target, stop building AWQ
candidates. The reading:

- The 4-bit grid is the floor and the launch record measured it: both W4
  artifacts sit at 0.33--0.38 median relative L2 on every population and
  calibration data moves that by about a tenth either way. The subsection below
  gives the mechanism for that ceiling: AWQ's objective is local to each
  mapping, so no calibration population can aim it at the layer-49 state H3
  consumes.
- NVFP4 already dominates W4A16 at the policy that ships. `active_plan.md` makes
  the 2048 short edge the primary reference-still policy; NVFP4 reads 0.147
  there against W4's 0.312, at 14.61 GiB against 14.27. It loses marginally at
  `max_no_upscale`. If a small encoder is wanted, that file exists.
- There is nothing to calibrate on the INT8 path. ConvRot stores `W @ H^T` in a
  **constructed** Hadamard basis (`_build_hadamard` in
  [`bench/analyze_quant_delta.py`](../../../../bench/analyze_quant_delta.py)) --
  a data-free transform. No calibration set can improve or harm it.

What to do instead, in the order I would do it:

1. **Cache the conditioning.** If the working loop is seed iteration on a fixed
   prompt and reference set, the encode happens once rather than per render and
   the whole VRAM question stops mattering for that workflow. Design note:
   [`conditioning_cache.md`](../../conditioning_cache.md).
2. **Mixed precision chosen by per-module sensitivity measured on captured
   activations**, not on the weights -- see the subsection below for why the
   distinction decides whether the allocation is worth anything. The format
   supports the artifact already: group size is read from each tensor's own
   `comfy_quant` marker, so one file can carry I8 on some linears and 4-bit on
   others. Shaving INT8's H3 path to fit needs about 2.8 GiB, of which the
   embedding table at I8 is 0.72.
3. **Leave the vision tower alone**, per the table above.

### The measurement that is missing, and why it must be activation-aware

No encoder-side per-module sensitivity record was found under `bench/results/`
on 2026-08-26. `analyze_quant_delta.py`'s per-module work is on the **DiT**
(`blocks.N.attn.qkv_proj`, `mlp.fc1/fc2`), and the four-encoder table is
whole-encoder layer-50 only. So step 2 currently has nothing to allocate bits
with.

**Measure it on captured activations, not on the weights.** The cheap version
of this measurement ranks modules by weight error, `||W - W_hat||`. That is the
wrong quantity and it is worth saying why before somebody builds it: a large
perturbation in a direction the real activations never excite costs nothing
downstream, and a small one in a heavily excited direction costs a lot. The
quantity that predicts output error is `||Wx - W_hat x||` on the `x` that
actually occurs. Same bit budget, sensitivity ranked on the distribution that
occurs rather than on the weights in isolation.

This repo already has the pattern:
[`bench/grade_sage_on_capture.py`](../../../../bench/grade_sage_on_capture.py)
grades a kernel against an exact reference on captured activations, which
`CLAUDE.md` names as the only controlled comparison available here about a
numerical knob. The same shape applied to a quantized linear instead of an
attention kernel is the instrument step 2 needs. It requires captured
activations and matrix multiplies -- no calibration run, no `llm-compressor`,
no host budget, no ComfyUI downtime.

**A second use for the same trace data: the objective.** Relative L2 over the
raw layer-50 state weights all 5,120 dimensions and every position equally, and
the DiT does not:
[`bench/measure_dit_prefix_attention.py`](../../../../bench/measure_dit_prefix_attention.py)
measured 13% of attention on the prefix at block 49 against 0.2% at block 0,
with section keys over-read and cut timestamps under-read. The launch record
already lists the metric as its own third suspect for the Gate 5 result. Trace
data is what would let encoder error be weighted by what the DiT actually
reads, which is a better objective for the knapsack than undifferentiated L2 --
and the same criticism applies to every number in the four-encoder table, which
is why that table decides *which file*, not *which module*.

Whoever picks this up should check the widest MLP input (`down_proj`, the same
tensor family that dominated AWQ's host cache) first, because that is where the
bytes are.

**What this does not change: AWQ already calibrates per layer on propagated
real activations.** Raised as a proposal on 2026-08-26 and checked rather than
assumed. `propagate_error` defaults to true
(`coderef/llm-compressor/src/llmcompressor/args/dataset_arguments.py`), and the
sequential pipeline runs each subgraph twice -- once with modifier hooks live
to collect statistics and quantize, then again with hooks disabled to capture
the **quantized** outputs and feed those forward as the next subgraph's inputs.
The v2 run's activations were real: the bundle carries genuine H3 presentation
built through ComfyUI's own tokenizer and reference-conditioning path, and the
forward ran the real BF16 32B weights. (`build_native_h3_calibration_batch.py`
loads no 32B weights and runs a reduced hidden width, but it is only the bundle
*builder*, and its provenance says no presentation field depends on that width.)
The only thing not drawn from a real render is prompt provenance -- H3-IR rows
rather than prompts written against the owner's own references -- and that is a
lever on the ten-percent term the overfit test already sized, in an unknown
direction. Use trace data for the allocation above, not for another
calibration.

**The shape was the encoding, and it was checked, not assumed.** **SOURCE.**
ComfyUI's H3 encode is one causal forward over the whole packed sequence:
`comfy/text_encoders/llama.py` builds a `triu_(1)` causal mask whenever
`seq_len > 1`, and `MiniMaxH3ClipModel` sets `enable_attention_masks=False`, so
there is no padding mask over it. It is not a bidirectional encoder and it is
not a decode loop; the single-token KV-cache branch is never reached.
**MEASURED.** The calibration was matched to that deliberately: Gate 1's
effective attention-mask rule
([`2026-08-24_gate1_seam_acceptance.md`](2026-08-24_gate1_seam_acceptance.md))
found keeping versus omitting an all-ones mask bit-identical at the raw
layer-49 state, and its decision asserts every element is one before omitting
the mask from the traced graph, with a red control that inserts a zero and
proves omission is refused. No generation happened at any point -- AWQ collects
statistics from forward passes and the pilot calls no `generate()`. v1's
`add_generation_prompt=True` put `<|im_start|>assistant\n` in the token stream
and nothing was ever decoded from it; v2 has no chat template. The cross-stack
check exists for exactly this question and read relative L2 0.0020 against a
wrong-tap control at 0.0506.

**What was *not* the encoding is the objective, and this is the finding that
matters.** **SOURCE**, `AWQModifier._compute_loss`: the loss compares the fp16
output against the quantized-weight output of **one mapping's parent module**,
MSE, and `_error_metrics` is recorded per `smooth_name`/`parent_name`. Four
mappings per layer, each grid-searched independently. **There is no term
anywhere for the error at layer 49.** So the compounding half of the question
is satisfied -- `propagate_error` gives each layer inputs from the already
quantized layers above it -- while the output half is not: the run minimised
four-by-sixty-four local reconstruction errors and never knew which layer's
state is the deliverable. Fourteen of those sixty-four are layers H3 never
reads, which does not corrupt layers 0--49 because they are upstream of them.
**INFERENCE, from the layer count and not from a timing:** if AWQ's per-layer
cost is near-uniform -- the decoder layers are architecturally identical -- that
is about a fifth of the grid search spent on outputs nothing reads. The v2 run
record carries no per-layer modifier timing to confirm it.

**Consequence, and it is why section 1's ten-percent ceiling is not surprising.**
No choice of calibration data can make a local per-mapping objective into a
layer-49 objective. Data decides which activations the local errors are
measured on; it cannot change what is being minimised. An allocation measured
as `||Wx - W_hat x||` on captured activations **can** be propagated to layer
49's output and ranked on that, which is the objective AWQ structurally cannot
express -- a second reason to spend trace data there rather than on another
calibration population. **Section 5 records the owner's trace-calibration idea
against this same boundary**; it was filed before this paragraph existed, and
the annotation there carries what this finding does to its GPTQ argument.

## 4. Recommendation: no encoder fine-tune or LoRA before the marker arms run

**RECOMMENDATION, owner's call open.** Two of the three things usually named
together here are not weights problems at all:

- **2048 short edge** is a serving policy. The vision tower patches arbitrary
  resolution; no weight encodes a short-edge target. Whether the policy helps is
  the Gate 6 blind-pair question on the three `h3_probe_refview_*` graphs, and
  that has not run.
- **Prompt structure** already matches the vendor.
  [`h3_references.md`](../../../h3_references.md) grades ComfyUI's presentation
  against sglang's stage: same string, same place, no chat template.

**Special tokens are the only genuine weights question, and the frozen DiT is
the argument against training them.** **MEASURED**
([`2026-08-25_released_encoder_is_stock.json`](../../../../bench/results/2026-08-25_released_encoder_is_stock.json)):
the released `text_encoder/` is byte-identical to stock Qwen3-VL-32B-Instruct
across all 14 shards. So the release pairs the DiT with an encoder whose seven
marker rows are untrained. **INFERENCE:** training those rows moves the encoder
off the distribution the DiT was paired with -- drift, not alignment.

**UNKNOWN, and it must stay labelled that way: we do not know the DiT was
trained on stock Qwen3-VL.** The measurement above is about the *released*
encoder. What MiniMax used during DiT training is not observable from the
release, and [`baseline.md`](baseline.md) records it as unknown.

### How the question gets answered with no trainer

Not by teaching -- by **selecting among representations that already exist** and
measuring the DiT's response. Four arms, declared in
[`bench/marker_corpus/compiled.json`](../../../../bench/marker_corpus/compiled.json)
with the runtime in [`marker_arms.py`](../../../../marker_arms.py):

| arm | what changes | trained rows? |
|---|---|---|
| `release_id` | nothing; the seven dedicated IDs | no, untrained stock tail rows |
| `legacy_bpe` | tokenizer: marker text as ordinary BPE, via an unpatched tokenizer **selected by vocabulary, not by constructor name** | yes |
| `stripped` | prompt text: marker strings removed, every other character kept | n/a |
| `mean_init_rows` | the seven rows replaced by the embedding-table mean, as an offset-keyed `set` patch on a clone -- never the shared module, never a file | n/a |

`legacy_bpe` is the only arm carrying trained representations for the markers.
So the question is not which arm is better in the abstract but **which
distribution the frozen DiT learned to read**.

Three instruments, all built:

- **Encoder level, deterministic, no renders.** Capture layer-50 per arm and
  compare on aligned positions. Done once already
  ([`2026-08-21_h3_marker_token_states.json`](../../../../bench/results/2026-08-21_h3_marker_token_states.json)):
  marker representation moves the encoding by relative L2 0.034 on a short voice
  line to 0.16 on a dialogue-heavy prompt. Against the four-encoder table's
  0.021--0.027 for INT8 and 0.31--0.37 for W4, marker representation is a larger
  perturbation than choosing INT8 over BF16, and free.
- **DiT level.** The encoder result proves the arms differ, not that the DiT
  cares. That needs renders as a distribution, never a pair:
  `run_graph_arms.py` -> `blind_batch.py` -> the scoring app ->
  `score_session.py`, under `eval_comparison.md` section 3.
- **Structure level**, for "does this piece of prompt structure reach the
  model", with no perceptual judgement:
  [`bench/measure_dit_prefix_attention.py`](../../../../bench/measure_dit_prefix_attention.py).
  Its ref2va version needs the capture path to dump token tags and vision spans
  first.

### Why the arms are also the discriminator for section 4's UNKNOWN

The seven rows are untrained but not zero -- they are some initialisation
vector. **INFERENCE**, stated before the arms run so it cannot be fitted to
them:

- `release_id` reliably beating `mean_init_rows` means the DiT learned to read
  those *specific* untrained vectors, which is only possible if it trained
  against an encoder carrying exactly those rows. Evidence that the released
  encoder is the training encoder.
- The two being indistinguishable means the dedicated IDs carry no learned
  meaning to the DiT, and the special-token lane closes cheaply.
- `legacy_bpe` winning means training used the pre-patch spelling, and the fix
  is a tokenizer choice rather than anything in the weights.

**Readiness caveat.** The corpus is currently five scenes and says so of itself
-- "a seed set, not the frozen evaluation corpus; freezing is the owner's act"
-- and declares two missing cells (standalone reference audio with a referenced
speaker; reference video with a soundtrack). The arms can run on what exists; a
quotable claim wants the coverage the brief names.

## 5. Calibrating on a real render's own activations

**Owner's idea, 2026-08-26, recorded before anything was built or measured.**
Calibrate each layer against the activations that actually occur during a real
H3 render, rather than against a constructed pool.

### What it cannot reach, so it is not proposed as one

Section 1 stands unchanged and is not weakened by this. `layer0_input` relative
L2 is exactly 0.0 across BF16, v1 and v2, so **no** calibration population --
traced, constructed, or otherwise -- reaches the token embedding table, the
seven marker rows or the vision path. This idea is aimed at the other target
section 1 names: where the 4-bit approximation error *lands* on the decoder
linears. Those two objectives were conflated once already in this lane, which is
why the distinction is restated rather than assumed.

### The gap it would actually close, and its size

**MEASURED**, already in section 1 and repeated here because it is this idea's
whole case: the pool's label scaffold is **952 of 265,922 tokens**, and five of
the seven H3 markers occur in **zero** rows. A trace is by construction 100%
H3-schema-shaped -- the real packed sequence, the real `<Picture i>:` scaffold,
the real reference stills at the sizes the graph feeds them.

**MEASURED, and it argues the other way on modality.** The pool is not
text-heavy: `2026-08-25_v3_selection_max_no_upscale.json` records **188,754
visual against 76,216 text** tokens, so it already approximates a render's
composition. The gap is prompt *structure*, not modality. Any argument for this
idea that rests on "calibration never sees images" is wrong.

**PRIOR, stated before the run so it cannot be fitted to the result.** v1 to v2
was two different constructed pools and moved the error by about a tenth either
way (Gate 5, `2026-08-25_v2_launch_record.md`). Trace-versus-constructed is a
larger distribution shift than pool-versus-pool, but it is the same *kind* of
lever, so a tenth is the order to expect and a result far outside it should be
distrusted before it is believed.

**The prior now rests on a mechanism and not only on that analogy.** Section 3
establishes that the objective being minimised is layer-local: AWQ's loss
compares one mapping's parent module against its quantized self, with no term
anywhere for the layer-49 state H3 actually reads. GPTQ is the same shape --
[`bench/h3_gptq_recipe.py`](../../../../bench/h3_gptq_recipe.py) describes it as
compensating against "the inverse Hessian of the layer's own input covariance",
which is a *different* solver for the same local objective, not a wider one.
So changing the calibration distribution changes **where** each layer's local
error lands, and no calibration change of any kind introduces a term for the
state the DiT consumes. That bounds this idea and the one-arm GPTQ
consolidation below by the same ceiling, and it is the reason a tenth is the
order to expect rather than merely the last thing that happened.

### The cheapest form of it is not a new mechanism

llm-compressor already runs the model over the calibration set and collects each
layer's activations; that is how AWQ's scale search and GPTQ's Hessian are both
fed. So "calibrate on trace data" reduces, in its cheapest form, to **replacing
the pool with real H3 render inputs** -- our own prompts and reference stills,
in schema, at graph geometry -- and changing no code in the recipe at all.
[`bench/select_v2_calibration_rows.py`](../../../../bench/select_v2_calibration_rows.py)'s stratified draw would be replaced by an enumeration
of shipped graphs rather than a stratified draw from a public dataset.

The stronger reading -- per-layer solve against that layer's own observed input
statistics -- is **already what GPTQ does**, and
[`bench/h3_gptq_recipe.py`](../../../../bench/h3_gptq_recipe.py) exists. GPTQ
compensates rounding against the layer's own input covariance, which is the
traced quantity. So this idea and the GPTQ card on the 2026-08-26 handoff are
closer to the same experiment than they look, and running them as one arm --
GPTQ **on** an in-schema pool -- costs one run rather than two and separates
cleanly from the v1/v2 pair, which varied neither.

**Added after 52d5916, and it bounds the paragraph above rather than refuting
it.** GPTQ's objective is layer-local too. Its own recipe module says so:
[`bench/h3_gptq_recipe.py`](../../../../bench/h3_gptq_recipe.py) describes GPTQ
as pushing each column's error into the columns it has not reached yet "using
the inverse Hessian of **the layer's own input covariance**". So GPTQ changes
*how* a layer's local error is minimised where AWQ only moves it between
channels -- a real difference, and the axis Gate 5 never varied -- but neither
carries a term for the layer-49 state H3 reads. See section 3's objective
paragraph. The consequence for the one-arm proposal: an in-schema pool and a
compensating solver both act on the same bounded term, which is consistent with
the tenth-order prior stated above and is a second, independent reason for it.

### What would confirm or refute it

The first measurement is free of the card and should come first: **how far apart
are the two activation distributions?** Capture the encoder's per-layer input
statistics once on a pool row and once on a shipped graph's real prompt, and
compare. If they are close, the idea cannot move much and the reasoning above is
wrong somewhere. If they are far apart at the layers W4 hurts most, that
locates the gain before any calibration run is spent.

**UNMEASURED.** Nothing here has been run.

**Corrected in place, 2026-08-26.** This paragraph claimed the encoder-side
hook did not exist, reasoning from `h3_capture.py` -- which does only capture
DiT activations -- to the whole repo. That inference was wrong and the
annotation below, written independently within the hour, contradicted it three
lines later. Reasoning from one file to a repo is how the claim was made; the
correction is left visible because a section that argues both ways is worse
than either. [`bench/capture_h3_encoder_states.py`](../../../../bench/capture_h3_encoder_states.py)
already runs the encoder arm and taps two points -- the layer-0 input and the
raw state after language layer index 49 -- with the presentation hashing and the
refusal-on-mismatch its comparator needs. What it does not have is a tap on
every layer's *input*. Adding one is an extension of an existing arm rather than
a new capture path, and
[`compare_transformers_comfy_layer50.py`](../../../../bench/compare_transformers_comfy_layer50.py)'s
`_embedding_tap` is the shape to copy.

**Which graph to trace, added 2026-08-26 after a peer proposed the four
no-attention-patching PDD arms as the cleanest source.** **SOURCE, verified:**
no attention patching in this repo reaches the Qwen3-VL encoder at all.
`MiniMaxH3SageAttention` and `MiniMaxH3SolAttnCurve` both take
`io.Model.Input("model")` and the sage node's own description says to connect it
between the model loader and the sampler; its `patch_token_refiner` option
patches the DiT's text token-refiner blocks, not the encoder. The encoder
resolves its own `optimized_attention_for_device` inside
`comfy/text_encoders/llama.py`'s decoder forward, independent of
`transformer_options`, and `attention.py` contains no reference to a CLIP or
text encoder. So sage, Sol, the SLA router and PDD arms all produce **the same
encoder activations** for the same prompt and references, and a graph's
attention arm is not a selection criterion for this measurement. What is: schema
coverage -- markers, the `<Picture i>:` and timestamp scaffold, reference roles,
still policy -- which is the gap this section already names. That widens the
candidate set from four arms to every shipped graph. (The peer's reasoning does
hold one stage down: for a **DiT**-side capture, such as the ref2va prefix
attention on the handoff's card, a graph wiring no attention patching is the
cleanest source and the PDD arms are the only ones.)

## What this record does not establish

- Anything perceptual. Every recommendation here rests on weights and encoder
  states. Gate 6 has not run.
- That mixed precision recovers INT8's fidelity. Section 3 step 2 is a proposal
  with a named missing measurement, not a result.
- What the DiT was trained against. Section 4 says how that becomes answerable,
  and answers it nowhere.
- Any change to the deployed artifact, the encoder of record, or the card order
  in the 2026-08-26 point handoff. Nothing here supersedes an owner decision.
- Anything about section 5. It is an idea with a stated prior and a named first
  measurement, and not one line of it has been run.
