# Gate 2A, corrected: the no-modifier sequential floor from one harness version

**Date:** 2026-08-25
**Status:** Deliverable for Codex acceptance. Not authority; not a launch
request; not a population budget.
**Scope:** the rerun specified in
[`2026-08-24_gate2_readiness.md`](../../canonical/2026-08-24_gate2_readiness.md):
both planned Gate 2A arms on one committed harness, and the separate
backend-dispatch probe at the real row and block shapes. Gate 2B was not
started. No recipe, no modifier, no candidate directory, no deployment change.

## Headline

**MEASURED.** Through the supported `load_context` / `auto_offload` bridge, all
64 decoder layers at FP32, the modifier-free sequential path completes three
primary rows (8,703 tokens, longest row 5,857) at a cumulative CUDA peak of
22.1 GiB allocated / 23.0 GiB reserved, and fails on the fourth. Both failures
in this record are the same mechanism, named by the allocator and by the
profiler rather than inferred: **decoder layer 0's attention on the longest row
dispatches `aten::_scaled_dot_product_attention_math` and materialises the
full `[64 heads, L, L]` FP32 logit tensor.** On the 8,981-token stress row the
allocator asked for 19.23 GiB, which is that tensor to the hundredth of a
gibibyte.

**MEASURED, and a correction to the canonical "no automatic backend
identified".** Selection is now identified at every real shape. With the
grouped-query call the model declares, no fused kernel exists at FP32 on this
card and `auto` is the math backend. With the key/value heads expanded to 64,
`auto` selects `aten::_scaled_dot_product_efficient_attention` at every real
length and completes at 8,981 and 10,358 tokens where math cannot allocate.
That is the KV-expansion lever the previous session withdrew: it now has
measured selection and a measured OOM stage on both sides. It is a candidate
with evidence, not a recommendation; see the numerical caveat below.

**MEASURED.** The FP32-resident arrangement consumes the host. Load-time RSS
peaks at 122.6 GiB on a 125 GiB box, settles near 108 GiB, and climbs to 117
GiB across steps; `MemAvailable` ends at 5.05 GiB. There is no host reserve to
derive from this arrangement for a modifier. The bridge's "disk" share is
symlinks into two source shards, with zero bytes staged.

## Provenance

Every result file names the producer that wrote it (commit, file SHA-256,
dirty flag). Three harness commits exist since the readiness record's
`745d916`, all before any accepted number was produced:

| commit | change | reports it produced |
|---|---|---|
| `01a81d6` | producer provenance in every report; mask-omission proof read from the dataloader, cache and traced graph; cache growth sampled per forward instead of read as the residual; trace as its own timed stage; host memory before and after the bridge load; probe keeps the dispatched op name when a call fails | none: its first run crashed in a progress print after the measurement |
| `0ce0e82` | that print | all three pilot reports |
| `1505355` | probe sweeps the expanded-KV text shape beside the grouped-query one | both probe reports (the earlier pair from `0ce0e82` were overwritten by the rerun and differ only by the added shapes) |

Reports, all under `bench/results/`:

- [`2026-08-25_gate2a_abort_control.json`](../../../../../bench/results/archive/v2_encoder/2026-08-25_gate2a_abort_control.json)
- [`2026-08-25_gate2a_primary_escalation.json`](../../../../../bench/results/archive/v2_encoder/2026-08-25_gate2a_primary_escalation.json)
- [`2026-08-25_gate2a_stress_2048.json`](../../../../../bench/results/archive/v2_encoder/2026-08-25_gate2a_stress_2048.json)
- [`2026-08-25_sdpa_backend_selection_primary.json`](../../../../../bench/results/archive/v2_encoder/2026-08-25_sdpa_backend_selection_primary.json)
- [`2026-08-25_sdpa_backend_selection_stress.json`](../../../../../bench/results/archive/v2_encoder/2026-08-25_sdpa_backend_selection_stress.json)

Producers: [`pilot_sequential_feasibility.py`](../../../../../bench/pilot_sequential_feasibility.py),
[`probe_sdpa_backend_selection.py`](../../../../../bench/probe_sdpa_backend_selection.py),
[`build_native_h3_calibration_batch.py`](../../../../../bench/build_native_h3_calibration_batch.py).

Bundles were rebuilt on the current tree from the accepted pool at H3-IR
`460db32`, under the release processor declaration, and are disposable; each
report carries the bundle provenance and every row's raw-presentation and
effective-input hashes. The population is the diagnostic's, so the two records
are directly comparable:

| row | role | tokens | visual | blocks (grid) | still policy |
|---|---|---:|---:|---|---|
| `train-00973` | single-image | 1,100 | 442 | 44x40 | max, no upscale |
| `train-00500` | keyframe + reference | 1,746 | 1,133 | 84x48, 22x22 | max, no upscale |
| `train-00012` | multi-image 2--3 | 5,857 | 4,944 | 156x104, 52x68 | max, no upscale |
| `train-00002` | video reference | 6,189 | 5,050 | 5 x 48x84 | (video) |
| `train-00006` | keyframe + reference | 10,358 | 9,270 | 2 x 96x172, 48x84 | max, no upscale |
| `train-00302` | multi-image 2--3, 300x300 sources | 8,981 | 8,196 | 2 x 128x128 | **2048 upscale, stress** |

## Substrate

**MEASURED**, identical across all three pilot loads:

- `Qwen3VLForConditionalGeneration` with 64 decoder layers at FP32 through
  `load_context` over `load_offloaded_model`, `device_map="auto_offload"`:
  1,061 offloaded tensors, 0 resident, all `float32`; **0 Accelerate hooks
  remaining, no `hf_device_map`, conversion clean**. Read from
  `OffloadCache.offloaded_values` without onloading.
- Placement: 114.1 GiB on CPU, 10.16 GiB on disk (108.7 / 15.6 on the first
  load, when less host memory was available; the bridge sizes the CPU share
  from `MemAvailable` minus its 5 GB default reserve at load time).
- Load: 15.3--16.1 s.
- **Staging: 0 files, 4,002 physical bytes, 46 symlinks resolving to 2 unique
  targets totalling 8.15 GB that were never copied.** The "disk" tier reads the
  source safetensors shards in place. This is the case the symlink-dedup fix
  was written for, and it means temporary-disk use is not a Gate 2 resource
  axis under this bridge.
- Host around the load: `MemAvailable` 121.9 GiB before; **peak RSS 122.6
  GiB**, current RSS 107.9 GiB, `MemAvailable` 14.8 GiB after. The load
  transiently overshoots its settled footprint by about 15 GiB.

## The two arms

**MEASURED**, `comfy_exact`, 64 layers, RTX 4090 with nothing else on the card
(23.54 GiB usable). CUDA figures are cumulative over cache construction and
every subgraph forward. "Warm transient" is the largest growth inside a single
subgraph call whose weights were already onloaded.

| arm / step | rows | tokens | longest row | outcome | CUDA alloc / reserved | warm entry / transient | host current / avail | forwards | trace / cache / forwards / total s |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| abort control, step 1 | 1 | 1,100 | 1,100 | completed | 5.32 / 5.38 | 5.18 / 0.82 | 110.7 / 10.7 | 130 | 2.0 / 0.0 / 56.9 / 64.5 |
| abort control, step 2 | 2 | 2,846 | 1,746 | **deliberate abort** after 5 forwards | 5.62 / 5.73 | 5.30 / 0.32 | 110.8 / 11.3 | 5 | 1.3 / 0.1 / 1.6 / 3.7 |
| primary, step 1 | 1 | 1,100 | 1,100 | completed | 5.32 / 5.38 | 5.18 / 0.82 | 113.8 / 8.8 | 130 | 1.6 / 0.0 / 207.1 / 213.4 |
| primary, step 3 | 3 | 8,703 | 5,857 | completed | **22.11 / 23.00** | 5.73 / **19.40** | 116.2 / 5.8 | 390 | 1.0 / 0.3 / 48.6 / 88.8 |
| primary, step 4 | 4 | 14,892 | 6,189 | **CUDA OOM** | 22.54 / 22.72 (before the failed request) | 6.16 / 19.40 | 116.9 / 5.05 | 12 | 0.3 / 0.5 / 10.4 / 12.0 |
| stress, step 1 | 1 | 1,100 | 1,100 | completed | 5.32 / 5.38 | 5.18 / 0.82 | 113.5 / 9.1 | 130 | 1.2 / 0.0 / 56.1 / 61.9 |
| stress, step 2 | 2 | 10,081 | 8,981 | **CUDA OOM** | 8.23 / 8.70 (before the failed request) | 6.01 / 2.22 | 112.2 / 10.5 | 6 | 1.9 / 0.4 / 10.8 / 13.7 |

Read across the arms:

- **The replay is real and counted**: 130 forwards per row is 65 subgraphs (64
  decoder layers plus the vision/embedding head) times two passes; 390 for
  three rows.
- **The 1,100-token step is bit-for-bit repeatable on CUDA** (5.315 / 5.379
  three times) and **not repeatable in time**: 64.5, 61.9 and 213.4 s for the
  identical step. The variance is in the forwards, not the trace or cache, and
  the disk tier is the only thing that differs between processes. Wall-clock
  from this arrangement is host I/O, not compute.
- **The host does not recover between steps.** Current RSS rises 107.9 to
  116.9 GiB over the primary escalation and `MemAvailable` falls to 5.05 GiB.
  Whether that is page cache for the disk tier or retained onload copies is
  not distinguished by this record; either way it is memory a modifier would
  need.
- **Cleanup is clean every time**: CUDA returns to 0.008--0.009 GiB allocated
  and 0.043--0.045 GiB reserved after completion, after the deliberate abort,
  and after both OOMs; the offload directory holds only its 46 symlinks at exit
  and is removed.
- **The intermediates cache is on the CPU as declared**, 46--74 KB per token
  at peak, and the residual equals the peak because the widest thing it ever
  holds is the last subgraph's inputs. It is not a Gate 2 constraint on this
  population; it would be at population scale, and that is a host figure.

### Both OOMs, attributed

**MEASURED**, the allocator's own message, preserved at the subgraph-forward
stage on a warm call in both cases:

| arm | failing call | allocator request | already allocated | nominal `[64, L, L]` FP32 logits for that row |
|---|---|---:|---:|---:|
| primary, step 4 | call 12: decoder layer 0, calibrate pass, batch 3 (`train-00002`, 6,189 tokens) | 2.28 GiB | 22.54 GiB | 9.13 GiB |
| stress, step 2 | call 6: decoder layer 0, calibrate pass, batch 1 (`train-00302`, 8,981 tokens) | **19.23 GiB** | 5.10 GiB | **19.23 GiB** |

The stress request equals the nominal tensor exactly; the primary request is a
later, smaller allocation made after the 19.4 GiB transient of the 5,857-token
row had already set the high-water mark and the 6,189-token row's larger
logits were in flight. Both land on the first language layer, on the longest
row in the step, on a warm call, so weight onloading is not the mechanism.
**The vision tower is not the mechanism either**: the stress row's two
128x128 blocks (16,384 patches, nominal 16.0 GiB under math) completed both
passes of the vision subgraph before the language stack failed, and the
16,224-patch block of `train-00012` completed inside the 22.1 GiB step.

Why the vision tower survives what the language stack cannot is the next
section.

## Backend selection, measured

Producer: `probe_sdpa_backend_selection.py`, separate from the pilot so the
profiler's overhead is outside every number above. Two arms at FP32, torch
`2.13.0+cu132`, released head geometry (64 query heads, 8 KV heads, head
dimension 128; 16 vision heads).

**Direct arm, every real shape.** The `aten::_scaled_dot_product_*` op the
profiler recorded under `auto`, and which forced backend reproduces `auto` bit
for bit:

| shape | lengths probed | availability (flash / efficient / cuDNN) | `auto` dispatched | reproduced bit-for-bit by | forced math |
|---|---|---|---|---|---|
| text, grouped-query (`enable_gqa`), causal | 1,100 / 1,746 / 5,857 / 6,189 / 8,981 / 10,358 | none / none / none | `_attention_math` at every length | math (where it completes) | OOM at 8,981 (19.23 GiB) and 10,358 (25.58 GiB) |
| text, expanded KV (64 heads), causal | same six | none / **yes** / none | `_efficient_attention` at every length | efficient, delta 0.0 | completes to 6,189; OOM at 8,981 and 10,358 |
| vision, full attention, non-causal | 484 / 1,760 / 3,536 / 4,032 / 16,224 / 16,384 / 16,512 patches | none / **yes** / none | `_efficient_attention` at every size | efficient, delta 0.0 | completes to 4,032; OOM at 16,224 (15.69 GiB), 16,384 (16.00), 16,512 (16.25) |

Flash and cuDNN report "No available kernel" at every FP32 shape, forced or
not. The availability API and the profiler agree at every shape probed, which
is the first time in this lane they have been checked against each other.

**In-situ arm**, two released decoder layers at released width, real
calibration batches (`train-00500`, two blocks; `train-00973`, one block),
under `comfy_exact`: the model's own forward dispatches
`_scaled_dot_product_attention_math` and
`_scaled_dot_product_efficient_attention`, one for each stack, matching the
direct arm.

**What this settles.**

- The Gate 2A OOM boundary is the math backend's quadratic logit tensor on the
  longest row's language attention. Population size does not enter it.
- Under expanded KV the efficient kernel is selected at FP32 and completes at
  every real length in the population, including the two that OOM today. The
  KV-expansion lever is therefore a measured candidate. Its numerical standing
  is not measured here: the probe records a max-absolute difference against
  the math result of `1.5e-3` to `1.7e-3` on unit-scale random inputs, which
  is not the Gate 1B metric and was not measured against deployed ComfyUI at
  layer 49. It also changes the arithmetic `comfy_exact` was accepted with, so
  it would need its own Gate 1B-style row before it could carry the policy.
- **INFERENCE, consistent with the above and offered as a reconciliation.**
  The Gate 1 effective-mask record measured relative L2 `4.2e-4` between
  "auto" and "forced math" with the mask omitted and attributed it to kernel
  policy without naming the kernel. On this record the language stack is math
  under both; the only kernel that changes under forced math is the vision
  tower's, efficient to math. That difference is the plausible source of the
  `4.2e-4`, which would make it a vision-tower figure. Not re-measured here.

## Effective-input proof through the real path

**MEASURED**, every step of every arm: each row's raw mask was all ones and
was omitted with both hashes recorded; the `IntermediatesCache` at the first
subgraph forward held exactly `image_grid_thw, input_ids, mm_token_type_ids,
pixel_values`; `attention_mask` was a placeholder of none of the 65 traced
subgraphs; `pixel_values` and `image_grid_thw` were placeholders of the first.
`omission_survives_dataloader_cache_and_trace` is `true` in all seven steps,
read from the dataloader, the cache and the graph rather than from the
transform's own record.

## What was corrected, and what deviated from the assignment

- The harness was changed before the accepted run rather than run as
  committed at `745d916`. Each change is a field the readiness record requires
  and the committed version could not report; all are above the accepted
  numbers in the commit history and named in the reports.
- The deliberate-abort control is its own invocation. The harness aborts only
  on the last prefix, and the last prefix is the one that OOMs, so the two
  controls cannot share a run.
- The first run of `01a81d6` crashed in a progress print after step 1 had
  completed and its cleanup had run; nothing from that process is quoted.
- The first bridge load ran while the owner's ComfyUI server was up, idle,
  with its models freed. The server exited through its `KeyboardInterrupt`
  path at 08:46:56, thirty seconds into the load, after a logging error, with
  this process at its 122 GiB RSS peak. The kernel log is not readable from
  this account, so the cause is not proven. The server was restarted in its
  `default` mode after the last probe and its queue was empty on both sides.
  Future Gate 2 loads should start on an empty host, by arrangement with the
  owner, not after a `/free`.

## What this record does not establish

- **No population budget**, and no total-token budget either: the GPU
  boundary is the single longest row, and the host boundary is the model's
  own storage.
- **Nothing about the AWQ increment** in memory, time, observer state or host
  RAM.
- **No numerical acceptance of the efficient kernel** as a calibration path.
- **No render, fidelity or quality claim.**
- **No measurement of the BF16-stored / FP32-active arrangement**, which is the
  one that would leave host room for a modifier. The FP32-resident
  configuration was measured because it is what the accepted policy and the
  committed harness load; its host figures show it cannot host Gate 2B.

## For Codex: decisions this record puts on the table

1. **Accept or reject this as the Gate 2A record.** Every item in the
   readiness record's required list is present in the report files, read from
   the object that owns it.
2. **The arrangement, before Gate 2B.** The FP32-resident load leaves 5 GiB
   of host memory after three rows and cannot take an explicit modifier
   reserve. Gate 2B's entry contract asks for a reserve "chosen from the
   corrected Gate 2A record"; the record's answer is that this arrangement has
   none to give. The BF16-offloaded, FP32-promoted-per-subgraph arrangement
   needs to be designed and measured first, or Gate 2B measures something that
   cannot run.
3. **The kernel, before Gate 2B.** Under grouped-query math at FP32, the
   full sequential arrangement completed a 5,857-token row and failed on the
   6,189-token video row with 6.16 GiB already resident at entry; the direct
   kernel alone completes 6,189 and fails outright at 8,981 and 10,358. So
   the full-model threshold for this arrangement lies between 5,857 and
   6,189 tokens and may move when the storage arrangement changes; it is not
   a rule that every video-reference row is excluded, since their lengths
   vary. Expanded-KV efficient attention is measured to complete every real
   length in this population at the kernel. Whether it may replace the math path under `comfy_exact`
   is a numerical question for a Gate 1B-style fixture against deployed
   ComfyUI at layer 49, and it should be asked before a modifier is
   instantiated on top of either.
4. **The 2048-upscale stress stratum.** Its single row asks for a 19.23 GiB
   logit tensor from 300x300 sources; under the primary policy the same row is
   951 tokens. Under the math path it cannot run; under the efficient path it
   is one of the two longest rows in the population. Its cost is entirely
   manufactured pixels. A decision to drop it from calibration, or to keep it
   as a declared exclusion, is a plan decision this record can now inform.

## Response to Codex's review of this record

Codex's 2026-08-25 review, relayed by the owner, accepted the record and made
corrections. Each is accepted here so the review does not have to re-derive
them:

- **Population size is one axis among several, not the wrong axis.** The
  GPU boundary in this record is the longest language row under its selected
  backend, with vision-block size and the vision backend as independent GPU
  axes; total population tokens govern the host cache (measured here at 46 to
  74 KB per token at peak), runtime, and calibration coverage. The final
  budget states row envelope, visual-block envelope, population/cache budget,
  runtime budget, and host reserve as separate dimensions.
- **All 64 layers remain the artifact target.** A 50-layer package can be
  derived from a 64-layer artifact and not the reverse; removing layers 50 to
  63 would not touch a layer-0 attention failure; and the depth ablation
  needs the deeper layers calibrated. Reopen only if Gate 2B shows all-64
  execution is materially infeasible.
- **The 6,189-token failure is arrangement-specific**, as corrected in
  decision 3 above. One video fixture failing does not exclude the family.
- **Superseded 2026-08-25, later the same day, by owner decision** (canonical
  `active_plan.md`, commit `f2f6d5a`): the 2048-short-edge upscale is now the
  *primary* reference-still policy, so the interpolated row is a calibration
  row and the evaluation-only ruling below is withdrawn. The measurement it
  rests on stands: that row is a 19.23 GiB logit tensor under grouped-query
  math, which is why the kernel and storage axes stopped being optional.
- **The interpolated 300x300-to-2048 row is evaluation-only.** (withdrawn, above) This is a
  decision about manufactured pixels, not about genuine high-resolution
  references or the 2048 processor ceiling, both of which stay.
- **Two producer commits are acceptable**: the three pilot reports name
  `0ce0e82` consistently and the separate dispatch instrument names
  `1505355`, whose only change was the probe's added shapes. No rerun to force
  unrelated instruments onto one commit.
- **Storage and kernel are tested as separate axes before they are
  composed**: BF16 inactive storage with FP32 active promotion against
  full-FP32 storage, holding grouped-query math fixed on rows that fit;
  expanded-KV efficient attention against grouped-query math, holding storage
  fixed, on the released-weight Gate 1B fixtures (all of which fit under
  either kernel); then the combined path on the long rows, which can only run
  under expanded KV, against deployed ComfyUI at layer 49 split by position
  class. Otherwise an improvement on the long rows could not be attributed.
- **Operational rule.** Future bridge loads happen with ComfyUI fully
  stopped by arrangement with the owner and restarted afterwards. `/free` is
  not a host-memory boundary when the load peaks near total system memory,
  whatever the precise cause of the 08:46:56 exit was.

**Kernel-axis matrix, as corrected by Codex.** An earlier draft of this
response said grouped-query math and explicitly expanded-KV math "should be
exact". That is a hypothesis until measured: the expansion is semantically a
copy, but implementation ordering can change floating-point results, which is
the same class of difference Gate 1B found in a reduction order. The matrix:

1. grouped-query math versus explicitly expanded-KV math, at every real
   length that fits: isolates the expansion transformation;
2. expanded-KV math versus expanded-KV efficient: isolates the kernel
   arithmetic;
3. both expanded arms versus deployed ComfyUI at the raw layer-49 state,
   split by position class: downstream numerical acceptability;
4. BF16-stored / FP32-active storage plus expanded efficient attention on the
   long rows: acceptance of the composed path where the grouped-math
   reference cannot run.

**The expansion has its own memory cost, to be measured beside the gain.**
NOMINAL, not measured: eight KV heads become 64, so each of K and V grows
eightfold. At FP32 and head dimension 128 that is `L x 64 x 128 x 4` bytes per
tensor: 0.34 GiB each at 10,358 tokens against 0.04 GiB grouped, so about
0.6 GiB more resident K/V per layer call at the longest row in this
population, in exchange for not allocating the 25.58 GiB logit tensor. The
probe allocated the expanded tensors but did not record their memory; the
matrix above should.

Gate 2A is accepted on that review. Gate 2B is not started. The next two
prerequisites are the storage/promotion arrangement and the numerical plus
full-sequential acceptance of expanded-KV efficient attention, in the order
above.
