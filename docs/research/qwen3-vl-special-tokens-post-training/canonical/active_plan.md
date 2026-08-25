# Canonical active plan

**Status:** Owner-accepted execution plan
**Last updated:** 2026-08-25

## Outcome and next milestone

The next model artifact to build is a new native-H3-calibrated W4A16 AWQ
candidate. It is a multimodal conditioning experiment: reference stills,
keyframes, mixed reference requests, and genuine reference-video blocks are
first-class inputs. A text-only quantization result would not satisfy the
owner's objective.

The current W4 checkpoint remains deployed while the candidate is built and
evaluated. Do not edit its processor snapshot, repoint its symlink, publish a
candidate, or begin special-token training as part of this work.

The immediate milestone was acceptance of one corrected no-modifier Gate 2A
record, followed by a bounded real-AWQ-modifier Gate 2B; both are done, and
the corrected v2 quantization launched on 2026-08-25
([`2026-08-25_v2_launch_record.md`](2026-08-25_v2_launch_record.md)). The
next milestone is Gate 5's numerical acceptance of that candidate.
The no-training marker render evaluation follows on the accepted candidate; it
is not a prerequisite for building the candidate.

## Locked decisions

### Calibration source and split

- The accepted source is the H3-IR candidate pool in
  [`calibration_data_pool.md`](calibration_data_pool.md). Do not reuse the
  rejected Gemini manifests, Malcolmrey generated outputs, avatar_500 images,
  or another source without a separately accepted rights and semantics review.
- The pool's primary-role partition is the target distribution. Dialogue,
  wide/tall geometry, audio labels, and small-source inputs are overlays, not
  competing buckets. Preserve their natural coverage where the component-safe
  split permits it and report the achieved distribution.
- Calibration and holdout are assigned by whole exact-media connected
  component, never by row. Reserve at least two small-source components for
  holdout. Verify every declared local media hash and complete a near-duplicate
  review before accepting the split.
- Absolute calibration population size is deliberately open. Set it from the
  measured modifier-bearing pilot as separate budget dimensions -- row
  envelope, visual-block envelope, population/cache, runtime, host reserve --
  not from a preselected row count or a single total-token figure. Every
  eligible and excluded row receives a reason.
- Superseded 2026-08-25 by the owner's still-policy decision above: the
  2048-short-edge upscale is now the primary reference-still policy, so
  interpolated rows are calibration rows. The earlier evaluation-only ruling
  on the 300x300-to-2048 row is withdrawn with it.
- Text-only T2VA rows are excluded from the vision-traced `oneshot` calibration
  run. They remain a deterministic held-out regression population. Dummy visual
  inputs and silent media dropping are forbidden. A second text-only trace is a
  future experiment only if v2 materially regresses the held-out T2VA arm.

### Execution arrangement

Decided 2026-08-25 on the measurements in
[`2026-08-25_gate2_arrangement.md`](2026-08-25_gate2_arrangement.md), which
also records who decided what and under which authority:

- **Storage:** `comfy_exact_bf16_store` -- weights BF16 where stored and
  offloaded, every parameterised op computed with a transient FP32 copy of
  its weight. Bit-identical to the accepted `comfy_exact` on the released
  weights; leaves the host free for a modifier.
- **Kernel:** expanded-KV memory-efficient attention, accepted as a
  calibration execution policy under the mid-stack reading, with a
  kernel-sensitivity control on the AWQ observer required in Gate 2B.
- **Depth:** all 64 decoder layers remain the artifact target.
- **Host:** ComfyUI is stopped by the owner's arrangement before any bridge
  load and restarted afterwards.
- **Host budget (measured 2026-08-25, after the first launch died):** AWQ
  costs about 430 KB of anonymous host memory per population token on top of
  whatever weights are resident, so the calibration runs with the weights on
  the bridge's disk tier under a cgroup cap, and the population is sized to
  that. A 128 GB swap file is the safety net against a kill, not a budget;
  a larger population needs the modifier's parent cache off host memory
  (deferred work below). Record and numbers:
  [`2026-08-25_v2_launch_record.md`](2026-08-25_v2_launch_record.md).

### Role-aware geometry and processor contract

The candidate owns a composition of upstream role sizing and the Qwen
processor. Both stages must be recorded per media item; a row-wide image policy
cannot represent mixed keyframe-plus-reference requests.

- **Keyframes:** use the resolved H3 target-canvas geometry. Retain the original
  temporal-role prose because Qwen does not receive the resolved frame index.
- **Ordinary Ref2VA stills, primary policy:** the vendor serving convention
  sglang implements: a 2048-pixel short edge with upstream upscaling
  **enabled**, nearest-32 geometry, no area cap (`upscale_2048` in the
  builder). **OWNER-DECISION, 2026-08-25**, superseding the earlier
  `max`/no-upscale primary: the DiT was trained on encoder states the vendor
  pipeline produced, so the calibration distribution follows that geometry
  whatever the information content of interpolated pixels. `max` with no
  upscale remains available as a comparison stratum. The cost is accepted:
  a 16:9 still is 7,296 visual tokens, so a token budget buys fewer rows.
- **Calibration population, OWNER-DECISION 2026-08-25:** a mixed set, rows
  under the 2048-upscale policy beside rows under `max`/no-upscale, in one
  bundle with the policy recorded per row, component-disjoint from a holdout
  built under both geometries. AWQ's scales are aggregates over everything the
  calibration shows the encoder, and both geometries are real serving modes
  (most shipped graphs run no-upscale; the Qwen-only upscale below is the
  experiment), so the quant must not favour either arm. The holdout check
  reports each geometry separately.
- **Serving parity and the first render experiment:** the vendor pipeline
  gives the same 2048 image to the encoder and to the VAE, which on a 24 GiB
  card means reference-latent rows and a VAE encode that do not fit for
  multi-reference graphs. The two branches share no geometry contract
  ([`h3_conditioning_end_to_end.md`](../../../h3_conditioning_end_to_end.md)
  section 1b), so the reference-node lane is building a per-reference Qwen
  view size independent of the VAE view. The first Gate 6 experiment on v2 is
  that ablation: (A) no-upscale on both paths, (B) Qwen view at 2048 with the
  VAE at source, (C) full parity where the card allows, matched seeds, judged
  blind as a distribution per [`eval_comparison.md`](../../../eval_comparison.md),
  with the per-arm row and VRAM cost recorded beside the verdict. Whether the
  DiT takes more identity from a finer Qwen view is **UNKNOWN** until then.
- **Reference video:** use the release role policy: 768-pixel short edge,
  1,032,192-pixel upstream area budget, duration-aware Qwen processing, and the
  native two-frame block/timestamp presentation.
- **Qwen still processor:** calibrate and serve v2 with the release-declared
  processor contract and snapshotted release bounds. Do not inherit the current
  W4 artifact's constrained snapshot or ComfyUI function defaults.
- **Qwen video processor:** snapshot and use the release video processor
  contract. Keep its ownership separate from the still processor even where
  current values happen to agree. Do not substitute stock ComfyUI's
  per-two-frame-block defaults: the release/encoder maximum is clip-wide over
  the sampled reference. The measured distinction activates only in its
  source/length regime (311+ target frames at 1344x768; outside the legal range
  for the measured 960x544 source), but calibration must preserve the declared
  policy rather than infer whether a selected row will trigger it.

The stock-versus-release resize kernel and float-versus-uint8 boundary have not
been isolated at layer 49 independently of pixel bounds. v2 follows the release
processor path because that is the accepted serving contract, not because a
separate fidelity benefit from that numerical preprocessing has been measured.

Control feasibility by manifest composition and total visual tokens. Do not
silently shrink a row into the old constrained band, drop media from an accepted
row, or use one universal cap to make the run fit.

## Execution sequence

### Gate 0 — freeze the substrate

Codex owns canonical integration and review. Before a new Claude starts:

1. preserve the current checkpoint, symlink, BF16 source, and `llm-compressor`
   checkout;
2. record the repository, ComfyUI, Transformers, `llm-compressor`, CUDA, Torch,
   and adapter revisions used by the preflight;
3. keep machine-local home paths out of committed scripts, manifests, results,
   and documentation; and
4. consume the completed M-RoPE and vision/DeepStack arithmetic-parity evidence
   rather than repeating it without a new escaped defect.

### Gate 1 — prove the exact calibration seam

**Status:** Accepted for presentation, media, full-checkpoint mapping, the raw
layer-49 tap, and hashed identity through the traced graph. The acceptance and
its bounds are recorded in
[`2026-08-24_gate1_seam_acceptance.md`](2026-08-24_gate1_seam_acceptance.md).

The new Claude owns the v2 implementation. Its first deliverable is a compact
audit plus an executable red/green seam proof, not a launcher.

The accepted path must instrument or reuse the installed native-H3 presentation
path and pass the exact validated batches to a preconstructed dataloader that
`oneshot` actually consumes. It must prove, on real decoded media:

- raw prompt bytes, token IDs, seven-token handling, image/audio/video labels,
  ordered media, H3 token tags, and `mm_token_type_ids`;
- image and two-frame video patch tensors, hashes, grids, vision spans,
  timestamps, attention masks, and M-RoPE inputs;
- keyframe/reference role provenance and both effective geometry stages;
- DeepStack injection and the raw unnormalized state after decoder layer 49;
- full-checkpoint strict loading into the Transformers calibration model; and
- identity between the validated batch, the dataloader batch, and the object
  consumed by the traced graph.

Mutation controls must detect chat wrapping, first-image slicing, media reorder,
timestamp drift, a missing temporal repeat, a grid change, media dropping,
wrong token tags, missing multimodal token types, and a builder that is not the
one handed to `oneshot`.

Token-tag mutations belong to the presentation gate. They are expected not to
change the Qwen/`oneshot` batch because H3 consumes those tags downstream; the
seam proof must record that specificity rather than pretend the traced graph can
observe them.

### Gate 1B — match the deployed precision configuration

**Status:** Accepted. The calibration-only `comfy_exact` arm preserves released
BF16 position-embedding values and coefficients, uses ComfyUI's explicit
four-term reduction order, and uses FP32 active compute. It passed the
predeclared comparison against deployed ComfyUI and the plain Transformers
FP32/BF16 arms at the released vision output and raw layer-49 state, split into
text and vision positions. The measurement and remaining residual are in
[`2026-08-24_transformers_comfy_parity.md`](2026-08-24_transformers_comfy_parity.md).

This is a calibration execution policy, not an embedding/vision storage change:
the candidate still preserves the BF16 embedding, vision tower, and DeepStack
tensors. Do not patch the deployed encoder or source checkpoint.

Normalize attention masks only through the rule in the Gate 1 acceptance
record: preserve and hash the raw mask, require it to be all ones, omit it from
the effective model input, and refuse omission if any zero appears. All later
dataloader/cache/trace identity assertions apply to that effective input.

### Gate 2A — measure the no-modifier sequential floor

**Status:** Accepted by Codex on 2026-08-25 from one committed harness version.
The record, the two prerequisite axes measured after it, and the corrections
consumed are in
[`2026-08-25_gate2_arrangement.md`](2026-08-25_gate2_arrangement.md). The
requirements below remain the standard the record was held to.

Run the smallest no-modifier experiment through the real pinned
`llm-compressor` sequential path that reveals cache, replay, offload, cleanup,
and baseline resource behavior without producing a launchable candidate.

Use the supported `load_context`/`load_offloaded_model` conversion after the
Accelerate load and prove that no Accelerate hooks or `hf_device_map` remain.
Do not use a raw `device_map="auto"` failure to characterize the converted
path.

Measure cumulative peak allocated and reserved VRAM, current and high-water
process RSS, system memory available, activation/intermediate-cache placement
and growth, replay behavior, time by observable stage, physical temporary-disk
use, and interruption/OOM cleanup. Report total population tokens separately
from the longest row and individual visual-block grids. Preserve the allocator
message and failure stage. Include representative single-, multi-,
mixed-keyframe/reference-, reference-video-, and 2048-upscale stress rows. The
pilot must have a deliberate abort/failure control and mark every partial
output non-launchable.

Read offload placement from the offload mechanism without enumerating model
parameters in a way that onloads them. Probe Flash, memory-efficient, and cuDNN
availability at each real row/block shape. If automatic backend selection is
used to explain a result, establish it with a separate profiler or
forced-backend probe; availability alone is not selection.

Use the accepted `comfy_exact` policy. Plain FP32, native BF16, and the earlier
generic-sum hybrid remain comparison arms; lower cost alone cannot promote one
to the candidate policy. The pilot must also prove that the effective
all-ones-mask omission survives the real sequential dataloader and cache path.

This no-modifier run is a floor measurement. It does not instantiate AWQ
observers or modifier state and therefore cannot set the final token budget or
absolute calibration population. A prefix's total token count is not a GPU
budget: rows execute sequentially, so the longest row and largest visual block
must be kept distinct from population-wide host-cache growth.

### Gate 2B — measure a bounded real AWQ modifier

**Status:** Measured 2026-08-25 in sprint form (smoke, one-row and three-row
AWQ arms, the observer kernel-sensitivity control, then two 2-layer
110k-token host-budget probes). Its population/cache dimension was dropped by
the sprint decision and reinstated the same day after the first launch was
OOM-killed; see [`2026-08-25_v2_launch_record.md`](2026-08-25_v2_launch_record.md).
Entry contract, item by item, in
[`2026-08-25_gate2_arrangement.md`](2026-08-25_gate2_arrangement.md).

Using the Gate 2A harness, the `comfy_exact_bf16_store` storage arrangement
and expanded-KV attention, run the smallest real AWQ-modifier experiment that
exposes observer state, sequential layer calibration, cache/replay behavior,
peak VRAM and host RAM, temporary disk use, and cleanup on intentional abort.
It must remain incapable of producing or publishing a candidate checkpoint.

Every modifier arm starts from a fresh full model load of all 64 layers. Use
the supported converted offload bridge with an explicit host-memory reserve
declared in the run record, explicitly place AWQ observer offload on CPU, and
verify the decoder-only target boundary before execution. Include a control
proving that the modifier path was entered, and the kernel-sensitivity control
on the observer's scales. Do not create an output directory, serializer,
symlink action, or publishing step.

Gate 2B sets the budget as separate dimensions -- row envelope, visual-block
envelope, population/cache, runtime, host reserve -- and fixes an absolute
calibration population only if its measured arm covers the accepted workload
shapes and leaves a declared operating reserve. Otherwise it narrows the next
pilot. Neither Gate 2 arm changes the accepted role partition.

### Gate 3 — freeze the executable split and launch package

**Status:** Executed 2026-08-25 in sprint form: component-disjoint split by
the selector, mr_data's independent review of the calibration set and the
rebuilt holdout, the corrected family map, the committed recipe with its
boundary asserted, a new output path. Codex's independent review is replaced
by that harness and the acting-point record; the owner approved the launch.
Items 2 (rejection manifests) and the pool-wide near-duplicate remainder are
open and named in the launch record.

Using Gate 2B's measured budget:

1. deterministically assign whole media components to calibration and holdout;
2. emit calibration, holdout, and rejection manifests with achieved role and
   overlay distributions;
3. recompute every local media hash and complete near-duplicate review;
4. capture the exact post-policy row-level presentation and geometry trace;
5. instantiate the complete AWQ/quantization recipe in the pinned environment;
6. prove that all 64 decoder layers are targeted while the embedding table,
   vision tower, and DeepStack remain BF16; and
7. make the output path new, explicit, and unable to overwrite or repoint the
   deployed artifact.

Codex independently reviews this package against
[`2026-08-24_awq_v2_preflight_review.md`](2026-08-24_awq_v2_preflight_review.md).
The full run begins only after that review is green and the owner explicitly
approves the launch. Gemini is paused; the rejected launcher is not a starting
point.

### Gate 4 — quantize and audit the candidate

The technical lead supervises the full run and records emitted row counts,
errors, layer progression, resource observations, and recovery state. The
candidate must be a new all-64-layer Hugging Face-compatible artifact with its
processor/tokenizer contract, source/checkpoint hashes, recipe, run environment,
and row-level calibration provenance bundled beside it.

Structural loading, tensor cardinality, and format checks are necessary but are
not fidelity evidence. A partial, interrupted, or fallback-modified result is
rejected rather than renamed as the candidate.

### Gate 5 — numerical acceptance

Codex owns independent capture and comparison. On the component-disjoint
holdout, compare at minimum:

1. BF16 versus current W4 with the current W4 policy forced into both arms;
2. BF16 versus v2 W4 with the v2 policy forced into both arms;
3. BF16 native deployed path versus each artifact's declared deployed path; and
4. current W4 versus v2 only where one explicit common policy and semantic
   alignment make that comparison valid.

Report results by primary role and relevant overlays, including text, vision,
marker, and ordinary-label positions. Reject rowwise metrics when token IDs,
grids, tags, spans, or semantic alignment differ. Numerical fidelity can accept
or reject an encoder artifact; it cannot establish H3 render quality.

### Gate 6 — H3 and marker evaluation

If v2 passes structural and numerical acceptance, run blinded distributional
H3 evaluations with matched seed sets. The first purpose is to determine
whether v2 improves the real reference-conditioning workload, not merely its
text-only behavior.

Then evaluate the no-training release-ID, true unpatched-BPE, and stripped
marker arms using the frozen owner-authored evaluation corpus described in
[`owner_authored_marker_corpus.md`](owner_authored_marker_corpus.md). Authorize
conclusions separately for dialogue, caption, lyrics, and cutoff. A result for
one family does not authorize training another.

Same-seed renders are different samples after an encoder change. Judge a
distribution blind; do not interpret a clip pair as pixel-aligned degradation.

## Deferred work

### Special-token post-training

No trainer is authorized by this plan. Post-training receives its own plan only
if a marker family shows a reproducible task-relevant deficit, the frozen
evaluation and disjoint development data support that family, the target
representation/objective is explicit, and a differentiable substrate passes
gradient checks plus measured RTX 4090 feasibility.

If release IDs are already best, preserve them. If true legacy BPE is better,
serving with that tokenizer is the lowest-risk first repair. If the arms overlap
at the experiment's sensitivity, stop rather than train into noise.

### A larger calibration population

Deferred 2026-08-25. The host, not the card, bounds the population: AWQ's
parent-argument cache costs about 430 KB per population token and is
re-streamed on every grid pass, so swap does not help and a bigger box only
moves the line. Growing past the launched 214k tokens means changing
`AWQModifier` to keep that cache off anonymous memory or to make fewer passes
over it. Take it up only if Gate 5 shows the candidate wanting more
calibration mass; the measurement is in
[`2026-08-25_v2_launch_record.md`](2026-08-25_v2_launch_record.md).

### Encoder-depth and structural-pruning ablations

Layer 50 remains the released interface. Depths below or above it are a separate
distribution-shift experiment against H3's learned projection and token refiner,
not part of v2 calibration acceptance. Design that ablation after the v2
artifact is stable, holding tokenizer, preprocessing, weights, DiT, prompt,
sampler, and seed population fixed.

Removing decoder layers 50--63, the final LM norm, and LM head from an H3-only
file is packaging. It does not prune embedding rows and, with the current
adapter, does not change the resident H3 model. Embedding precision is another
independent axis and stays outside this v2 run.

## Roles

- **Encoder Claude, v2 technical lead:** exact seam, full-load validation,
  feasibility pilot, split/trace integration, recipe and launcher, run
  supervision, and candidate artifact audit.
- **Codex:** retired from this effort on 2026-08-25 by the owner's
  decision; its accepted records stand. Its former duties (canonical
  authority, independent review, capture and comparison, the go/no-go
  recommendation) pass to the technical point below, with independent review
  supplied by separately briefed sessions where a second reader is needed.
- **Encoder Claude, technical point (session "v2-lead"):** owns the plan and
  its acceptances, labelled in canonical and reversible by the owner. The
  owner's launch approval is unchanged.
- **Repository owner:** final launch approval, GPU scheduling, and perceptual
  evaluation decisions.
- **Gemini:** paused. It may later receive a bounded tactical task with an
  independently checkable output; it does not own the v2 launcher.
- **Reference-node lane:** outside the v2 critical path. Consume its role-aware
  sizing contract; do not mix its compatibility redesign into this run.

## Immediate stop conditions

Stop and report rather than continue if:

- the validated presentation is not the object consumed by `oneshot`;
- chat framing appears, media is sliced/reordered/dropped, or a text trace
  silently replaces a vision trace;
- calibration and holdout overlap by exact or near-duplicate media or prompt;
- the actual role/geometry distribution cannot fit the measured budget without
  undeclared resizing or replacement;
- full-checkpoint loading, recipe construction, or an intentional failure
  control does not behave as specified;
- a candidate path could overwrite or repoint the deployed artifact;
- a numerical comparison lacks a matching BF16 arm or semantic alignment; or
- memory, time, fidelity, or quality is inferred rather than measured.
