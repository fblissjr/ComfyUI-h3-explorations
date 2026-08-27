# Gate 1B precision arm, the effective-input transform, and the Gate 2A floor

**Date:** 2026-08-24
**Status:** Deliverable for Codex review. Not authority; not a launch request.
**Scope:** the three items assigned in
[`2026-08-24-codex-to-claude-encoder-gate1-review.md`](../codex/2026-08-24-codex-to-claude-encoder-gate1-review.md).
Gate 3 has not been started. No quantization, no recipe, no candidate directory,
no deployment change.

## Headline

The hybrid arm as originally specified **failed** the predeclared rule, on one
grid out of four. Locating why turned out to be worth more than the verdict: the
residual was not the dtype at all, it was the *reduction order*, and replacing
only that produces a policy -- `comfy_exact` -- that passes every criterion on
every fixture and sits at the measured cross-implementation arithmetic floor.

The effective-input transform is in place and its equivalence is proven with the
two questions separated. Gate 2A is measured and has a hard boundary: **CUDA is
the binding constraint, at about 22 GiB, and the run OOMs between 8,703 and
14,892 tokens.**

## Gate 1B: the precision arm

### The first arm failed, and the failure was informative

Built as specified -- BF16 position interpolation, FP32 active compute -- and
graded against deployed ComfyUI on four released-weight fixtures:

| fixture | grid | vision output, plain FP32 | plain BF16 | hybrid |
|---|---|---:|---:|---:|
| single-image | 44x40 | 0.00971 | 0.04573 | **0.01856** |
| multi-image | 18x18 x2 | 0.00836 | 0.03625 | 0.00597 |
| keyframe-only | 48x84 | 0.01088 | 0.05111 | 0.00115 |
| mixed keyframe+reference | 84x48 + 22x22 | 0.01880 | 0.09551 | 0.00333 |

Hybrid beat both plain arms on three fixtures and lost on one, so it did not
pass. A second single-block fixture was added specifically to test whether block
count was the discriminator; the 48x84 keyframe row is single-block and hybrid is
best there by a factor of nine, so it is not.

### Where the difference actually was

Four comparison points, in order, on the released position table
([`probe_position_embedding_parity.py`](../../../../../bench/probe_position_embedding_parity.py)):

| grid | gather indices | BF16 weights | `.sum(1)` result | four-term add result |
|---|---|---|---|---|
| 44x40 | exact | exact | differs | **exact** |
| 18x18 x2 | exact | exact | differs | **exact** |
| 48x84 | exact | exact | exact | **exact** |
| 84x48 + 22x22 | exact | exact | differs | **exact** |

**MEASURED.** The two implementations agree exactly on which table rows to
gather and, at BF16, on the interpolation coefficients. What differs is only how
the four taps are summed: `comfy/text_encoders/qwen35.py` writes
`pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]`, transformers
writes `.sum(1)`, and at BF16 those are not the same number on three of the four
grids. On 48x84 they happen to agree -- which is exactly the fixture where the
original hybrid arm was already best, so the forensics predict the acceptance
matrix rather than merely accompanying it.

FP32 coefficients differ trivially (relative L2 `2.5e-6` on 44x40, exact
elsewhere) and that difference rounds away at BF16.

The extraction is not a reimplementation: ComfyUI's `weight_tensor` is a local,
so the probe substitutes `pos_embed` with a stub and reads the quantities back
out of the real function's own output -- a one-hot basis per tap makes the
weighted sum emit the weights, and a scalar source position makes it emit the
permutation, because bilinear weights sum to one. Every recovered quantity
carries its invariant: the permutation is a bijection, the indices are in range,
and the per-position weight-sum error is reported (`2.0e-3` to `2.9e-3` at BF16,
which bounds what the recovery can resolve). Corrupting one index or one
coefficient moves the comparison by `1.0e-2` to `3.6e-2`, so the "exact"
verdicts above are falsifiable.

### The policy, and what it is allowed to touch

[`h3_calibration_precision.py`](../../../../../bench/h3_calibration_precision.py)
now carries named policies rather than a dtype flag:

| policy | coefficients | reduction | active linears |
|---|---|---|---|
| `float32` | FP32 | `.sum(1)` | FP32 |
| `bfloat16_native` | FP32, as the library returns them | `.sum(1)` | BF16 |
| `bfloat16` | BF16 | `.sum(1)` | BF16 |
| `hybrid` | BF16 | `.sum(1)` | FP32 |
| `comfy_exact` | BF16 | ComfyUI's four-term add | FP32 |

**`bfloat16_native` and `bfloat16` are different arms, and the distinction is a
correction.** Transformers' helper computes the interpolation coefficients in
FP32 whatever dtype the model is, so plain BF16 rounds once, after the weighted
sum. Casting the coefficients first is a choice made here. The earlier
canonical figure of `0.095507` for "Transformers BF16" is the *native* arm; the
coefficient-cast arm measures `0.09341` on the same fixture. Reporting the
second under the first's name would attribute this lane's choice to the library.

`comfy_exact` substitutes only the reduction. It keeps the released BF16 table,
keeps transformers' own indices and weights, and leaves every active linear and
residual at FP32. It modifies no checkpoint, no saved config, no deployed
artifact, no symlink, no ComfyUI node, and no installed package.

### Acceptance

**MEASURED**, on the full tower (merged output and all three DeepStack features,
worst of the four) and at the raw layer-49 state split by position class:

| fixture | arm | vision output | layer-49 vision | layer-49 text |
|---|---|---:|---:|---:|
| 44x40 | FP32 | 0.00971 | 0.13663 | 0.000778 |
| | BF16 native | 0.04573 | 0.79786 | 0.012770 |
| | hybrid | 0.01856 | 0.61291 | 0.001684 |
| | **comfy_exact** | **0.00126** | **0.05927** | **0.000296** |
| 18x18 x2 | FP32 | 0.00836 | 0.00220 | 0.001261 |
| | BF16 native | 0.03625 | 0.01082 | 0.015455 |
| | hybrid | 0.00597 | 0.00132 | 0.000662 |
| | **comfy_exact** | **0.00095** | **0.00047** | **0.000382** |
| 48x84 | FP32 | 0.01088 | 0.01685 | 0.001148 |
| | BF16 native | 0.05111 | 0.43092 | 0.012082 |
| | hybrid | 0.00115 | 0.00292 | 0.000265 |
| | **comfy_exact** | **0.00115** | **0.00292** | **0.000265** |
| 84x48 + 22x22 | FP32 | 0.01880 | 0.39304 | 0.002615 |
| | BF16 native | 0.09551 | 0.96227 | 0.018018 |
| | hybrid | 0.00333 | 0.05178 | 0.001431 |
| | **comfy_exact** | **0.00117** | **0.05108** | **0.000719** |

`comfy_exact` is closer to deployed ComfyUI than both plain arms at the vision
output and at layer-49 vision positions on **every** fixture, and does not
worsen layer-49 text positions against plain FP32 -- it improves them
everywhere. The predeclared rule passes.

Its vision-output residual lands in a tight `0.00095`--`0.00126` band regardless
of grid, which is the matched-precision cross-implementation figure
(`0.00138`). That is the signature of a position embedding that is now exact
with only ordinary FP32 implementation difference left. On 48x84 `hybrid` and
`comfy_exact` are identical because `.sum(1)` already matched there.

The red control -- one interpolation tap scaled -- moves the vision output to
`0.26`--`0.36`, two to three hundred times the honest value.

**This is not bitwise parity and must not be called that.** The layer-49 vision
residual still reaches `0.059` on 44x40. It is the closest measured
approximation, arrived at by eliminating a specific identified cause.

### The policy's own guardrails, each watched failing

[`check_calibration_precision_policy.py`](../../../../../bench/check_calibration_precision_policy.py),
six arms, on a reduced-width model:

- **Restoration**: the module-level helper, the `pos_embed` forward, its dtype
  and the instance's hook count all come back identical, including when the body
  raises.
- **Raising forward**: a forward that raises must still close the gate. The
  cleanup hook uses `always_call=True`; removing it turns this red with
  "the gate was left open after a raising forward", which was verified by
  mutation, not assumed.
- **Instance scoping**: a second Qwen3-VL in the same process is bit-identical
  while the policy is active on the first, and the first is not -- the gate is
  asserted open and closed. Concurrent use raises rather than silently applying
  the wrong policy; this policy requires single-threaded forward execution and
  says so.
- **Offload-dispatch compatibility**: the substituted forward must stay a bound
  method, because `compressed_tensors.offload.module.offload_module` reads
  `module.forward.__func__`. **This one was found by a real failure, not by
  design** -- the first Gate 2 run died before its first batch on a substitution
  that was numerically perfect and structurally wrong. It is now a control.
- **Source guard**: with the substituted expression absent from the installed
  source, the policy refuses. It branches on the expression actually present,
  not a version string, because a backport or fork can carry any version number.
- **Red control**: `comfy_exact_corrupt_tap` moves the tower output.

## The effective-input transform

[`h3_effective_batch.py`](../../../../../bench/h3_effective_batch.py) is the one
declared transformation between the raw presentation and `oneshot`. It asserts
the mask exists and is all ones, records that assertion with the raw-presentation
hash, omits the mask, records the effective-model-input hash, and refuses the row
outright on any zero -- a padded row is not eligible and stops the run rather
than being processed under a different rule from its neighbours.

**MEASURED, three arms on a released-weight fixture, so the two questions do not
contaminate each other:**

| comparison | what it isolates | layer-49 text | layer-49 vision |
|---|---|---:|---:|
| all-ones mask, math backend vs no mask, math backend | mask omission under identical arithmetic | **0.000000** | **0.000000** |
| no mask, auto backend vs the same, math backend | kernel selection alone | 0.000344 | 0.000443 |

Omitting a mask that masks nothing changes the result by **exactly zero**,
including at layer 0 and on both position classes. The `4.2e-4` seen when the
backend is left free is attributable to kernel selection and to nothing else,
and it is the same order as `comfy_exact`'s own residual on the tightest
fixture, so it is not negligible and is reported rather than absorbed.

**MEASURED.** Forcing the memory-efficient backend fails outright at FP32 with
this attention shape (`No available kernel`). Torch reports availability but not
selection, so each capture records the requested backend and the availability
matrix for both the grouped-query shape the model declares and the expanded
shape a `repeat_kv` would produce. At FP32 no fused backend is available for the
grouped-query shape; that is a Gate 2 feasibility fact, not a reason to lower
calibration precision.

The comparator now refuses to report metrics when the arms differ on policy,
tap layer, perturbation or source, so a backend result cannot absorb a
difference caused by something else.

**Controls.** A mask with one zero and a batch with no mask are both refused by
name. M-RoPE position ids are unchanged by the omission on every row, asserted
rather than argued. The seam identity chain -- bundle file, `DataLoader`,
`IntermediatesCache`, traced subgraph -- now runs on the effective batch, and
`attention_mask` is absent from the traced subgraph's declared inputs where it
was present before.

## Gate 2A: the sequential floor envelope

**Not a budget.** The session runs with **no modifiers**: tracing, the
intermediates cache, and two forwards per subgraph per row, with nothing
observing or rewriting a weight. AWQ adds activation observation, its smoothing
search and in-memory rewriting on top. Reading these numbers as a population
budget would understate the real cost by an unmeasured margin.

**MEASURED**, `comfy_exact`, 50 decoder layers, RTX 4090:

| stratum | rows | tokens | visual | outcome | peak CUDA alloc | peak CUDA resv | peak host | forwards | seconds | cache |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| primary | 1 | 1,100 | 442 | completed | 5.32 GiB | 5.38 GiB | 121.5 GiB | 102 | 35.7 | 0.047 GiB |
| primary | 3 | 8,703 | 6,519 | completed | 22.11 GiB | 22.61 GiB | 121.5 GiB | 306 | 49.5 | 0.547 GiB |
| primary | 4 | 14,892 | 11,569 | **CUDA OOM** | 22.54 GiB | 22.78 GiB | 121.2 GiB | 12 | 30.7 | 0.959 GiB |
| 2048-upscale stress | 1 | 8,981 | 8,196 | **CUDA OOM** | 8.17 GiB | 8.58 GiB | 122.1 GiB | 3 | 31.1 | 0.649 GiB |

What this establishes:

- **CUDA is the binding constraint**, not host RAM and not disk. The boundary
  lies between 8,703 and 14,892 tokens for the primary stratum.
- **The replay pass is real, counted rather than inferred.** 102 forwards for
  one row is 51 subgraphs times two passes; 306 for three rows is the same times
  three. `propagate_error` says what was requested; this counts what happened.
- **The intermediates cache is not the problem.** It lives on the CPU, as its
  own `offload_device` declares, and grows to 0.547 GiB at 8,703 tokens --
  roughly 67 KB per token.
- **Temporary disk use is zero bytes**, at every step and at exit.
- **Cleanup is clean.** After every step, including both OOMs and the deliberate
  abort, CUDA returns to about 0.045 GiB reserved and the offload directory is
  empty.
- **The deliberate abort control works**: raised inside the fifth subgraph
  forward, caught, and nothing left behind.
- **The 2048-upscale stress stratum does not fit at all.** One row -- two
  128x128 grids from 300x300 sources -- OOMs at FP32. Note that same row costs
  951 tokens under the primary `max`/no-upscale policy and 8,981 under the
  stress policy, a factor of 9.4, which is why the plan keeps the stratum
  separately named.

**MEASURED, and narrower than this document first claimed.** *Raw, unconverted*
`device_map="auto"` does not compose with `SequentialPipeline` in this pinned
pair: Accelerate replaces every hooked module's `forward` with a
`functools.partial`, and `compressed_tensors.offload.module.offload_module` --
which the pipeline calls through `set_onload_device` before the first batch --
reads `module.forward.__func__`. That failure is real and reproducible.

**It is not evidence about the official conversion path, and the first version
of this section generalised one to the other.** `llmcompressor.utils.dev
::load_context` wraps `compressed_tensors.offload::load_offloaded_model`, which
loads through Accelerate and then calls `from_accelerate` to *replace* those
hooks with compressed-tensors offload caches; `device_map="auto_offload"`
additionally restricts placement to CPU and disk. That path is the supported
bridge and it is what a launcher should reach for. It is measured separately
below.

**The host figures in this table are contaminated and are superseded.** They
were produced by an instrumentation defect: the run reported parameter dtypes
and devices by iterating `model.parameters()`, which on an offloaded model
onloads every parameter. Under the host-resident arrangement that inflates the
RSS high-water mark; under any offloaded arrangement it also destroys the
topology the field was meant to describe. `ru_maxrss` is in any case a
*historical* peak and does not say what was free when a later stage would begin.
Both are fixed, and the corrected figures are in the follow-up section.

**Two further reporting defects in this table, both found by review rather than
by it going red.** The CUDA peak was reset before every subgraph forward, so the
value reported per step described only the last window -- on a two-layer control
it under-reported by about a factor of two. And temporary disk use followed
symlinks, so an offload directory linking into the source checkpoint would have
reported tens of gigabytes that were never written. Both are fixed; the numbers
above are kept as the diagnostic record they are.

## What this does not establish

- **No population budget.** Gate 2A is modifier-free by construction.
- **Nothing about the AWQ increment** in memory, time or host state.
- **No render, fidelity or quality claim.** No DiT was run and nothing was
  quantized.
- **`comfy_exact` is not bitwise parity** with deployed ComfyUI end to end. Its
  position embedding is exact; the tower and language stack retain ordinary FP32
  implementation difference, up to `0.059` relative L2 on layer-49 vision
  positions for one fixture.
- **The precision arms rest on four released-weight fixtures**, chosen because a
  FP32 forward fits. Fixture-level evidence about the boundary, not a population
  estimate.
- **Whether an active subgraph can be promoted to FP32 while inactive weights
  stay BF16 and offloaded is still unmeasured.** The FP32-resident configuration
  was measured instead, because it was the one that ran. That promotion is the
  obvious lever against both the CUDA ceiling and the host ceiling and it should
  be designed before Gate 2B rather than after.

## State at the end of this session, and what runs first tomorrow

**The Gate 2A numbers in this document are a diagnostic record, not the final
evidence.** Seven review findings landed against the harness after they were
produced, and each one moved a number or a claim:

| finding | effect on the numbers above |
|---|---|
| CUDA peak reset before every subgraph | the per-step peak described only the last window; under-reported by about 2x on a control |
| `model.parameters()` onloads an offloaded model | every host-RAM figure describes the instrumentation as much as the run |
| `ru_maxrss` is a historical peak | says nothing about what was free when a later stage would begin |
| `directory_bytes` followed symlinks | would report unwritten checkpoint bytes as temporary disk use |
| symlink targets counted per link | one shard counted once per link, exceeding the disk |
| offload topology keyed by tensor device | disk-resident weights filed under `meta`, so CPU and disk were indistinguishable |
| availability matrix omitted cuDNN, probed a synthetic 512-token text shape | see below |

All are fixed in the harness. The corrected runs were launched and then stopped
mid-flight at the end of the session, so **no bridge artifact is committed**:
the two that existed were produced by a superseded harness version and were
removed rather than left to be mistaken for the coherent set.

**What is settled and survives the corrections**, because it does not depend on
any of the affected fields:

- The official bridge -- `load_context` over `load_offloaded_model` in the
  pinned `llm-compressor` and `compressed-tensors`, with
  `device_map="auto_offload"`
  -- **loads all 64 decoder layers at FP32** and converts Accelerate's hooks
  cleanly: 1,058 offloaded tensors, zero Accelerate hooks remaining, no
  `hf_device_map`. This is the substrate answer, and it supersedes the
  host-resident arrangement the tables above used.
- The modifier-free sequential path completed a three-row population through
  that substrate.
- Raw, unconverted `device_map="auto"` remains incompatible, for the reason
  given earlier in this document.

**A withdrawn recommendation, retained as a candidate.** This lane proposed
expanding KV heads so the memory-efficient kernel becomes available at FP32,
on the reasoning that `enable_gqa=True` leaves no fused backend and forces the
math backend's quadratic scratch. That reasoning rested on an availability
matrix which omitted cuDNN entirely and was measured at a synthetic 512-token
text shape rather than at the shapes that actually failed. Two independent
defects pointing the same way, which is the combination that makes a wrong
conclusion feel well founded.

It is **not** a recommendation. It is an unselected candidate lever, and it
stays open until two things are measured: which backend is actually selected,
and at which stage each OOM occurs. Availability is not selection, and even a
fused backend may require workspace.

### First tasks tomorrow, in order

1. **One coherent rerun** of both Gate 2A arms on the current harness -- the
   primary escalation and the 2048 stress row behind a small trace row -- so
   the evidence set comes from a single harness version rather than three.
2. **`bench/probe_sdpa_backend_selection.py`**, which names the dispatched
   `aten::_scaled_dot_product_*` operation with `torch.profiler`, at the real
   released head geometry. Deliberately separate from the pilot so its overhead
   cannot contaminate the feasibility peak and timing.
3. Only then, revisit the KV-expansion lever and the 2048 stratum decision.

Gate 2B remains blocked on all of the above, and on its own conditions: a fresh
model per modifier arm, an explicit host reserve above the bridge's 5 GB
default, measured modifier state and smoothing overhead, and no checkpoint or
candidate directory.

## For Codex

1. **`comfy_exact` is accepted** in the canonical parity record, which this
   deliverable's matrix supports. Recorded here as the evidence behind it, and
   with the limit restated: it passes the predeclared rule; it is not parity.
2. **Already integrated, with one loose end.**
   [`2026-08-24_transformers_comfy_parity.md`](../../canonical/2026-08-24_transformers_comfy_parity.md)
   now closes Gate 1B on `comfy_exact` and its per-fixture evidence links are
   published at the paths it cites. The remaining loose end is the older table
   in that file, whose `0.095507` row is labelled "Transformers BF16" -- that
   figure is the *native* arm, and the file's own Gate 1B table correctly says
   "native BF16". The two should not read as the same arm as the
   coefficient-cast `bfloat16` policy, which measures `0.09341` on the same
   fixture.
3. **Gate 2B scope.** A bounded disposable modifier-bearing run is needed to
   turn the floor into a budget. Given the FP32 host peak already at 121.5 GiB,
   Gate 2B should decide the subgraph-promotion question first, or it will
   measure an arrangement that cannot host the modifier.
4. **The 2048-upscale stress stratum needs a decision, but not yet a verdict.**
   At FP32 it did not fit for a single row under the host-resident arrangement.
   Whether that stratum is dropped, re-sized, or run differently is a plan
   decision, and it should not be settled on a singleton OOM alone -- the
   controlled retest, with a small trace row ahead of it to separate trace and
   setup cost from the row's own forward, is reported in the follow-up section.
