# Canonical active plan

**Status:** Owner-accepted execution plan
**Last updated:** 2026-08-24

## Outcome and next milestone

The next model artifact to build is a new native-H3-calibrated W4A16 AWQ
candidate. It is a multimodal conditioning experiment: reference stills,
keyframes, mixed reference requests, and genuine reference-video blocks are
first-class inputs. A text-only quantization result would not satisfy the
owner's objective.

The current W4 checkpoint remains deployed while the candidate is built and
evaluated. Do not edit its processor snapshot, repoint its symlink, publish a
candidate, or begin special-token training as part of this work.

The first expensive milestone is the corrected v2 quantization, after a bounded
feasibility pilot and a reviewed launch preflight. The no-training marker render
evaluation follows on the accepted candidate; it is not a prerequisite for
building the candidate.

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
  measured feasibility pilot and a total-token budget, not from a preselected
  row count. Every eligible and excluded row receives a reason.
- Text-only T2VA rows are excluded from the vision-traced `oneshot` calibration
  run. They remain a deterministic held-out regression population. Dummy visual
  inputs and silent media dropping are forbidden. A second text-only trace is a
  future experiment only if v2 materially regresses the held-out T2VA arm.

### Role-aware geometry and processor contract

The candidate owns a composition of upstream role sizing and the Qwen
processor. Both stages must be recorded per media item; a row-wide image policy
cannot represent mixed keyframe-plus-reference requests.

- **Keyframes:** use the resolved H3 target-canvas geometry. Retain the original
  temporal-role prose because Qwen does not receive the resolved frame index.
- **Ordinary Ref2VA stills, primary policy:** use `max` with upstream upscaling
  disabled. This preserves real source detail up to the serving ceiling without
  manufacturing pixels.
- **Ordinary Ref2VA stills, stress policy:** keep a separately named
  2048-short-edge, upscale-allowed stratum. It measures the serving convention
  without allowing interpolated large references to dominate the primary mix.
- **Reference video:** use the release role policy: 768-pixel short edge,
  1,032,192-pixel upstream area budget, duration-aware Qwen processing, and the
  native two-frame block/timestamp presentation.
- **Qwen still processor:** calibrate and serve v2 with the release-declared
  processor contract and snapshotted release bounds. Do not inherit the current
  W4 artifact's constrained snapshot or ComfyUI function defaults.
- **Qwen video processor:** snapshot and use the release video processor
  contract. Keep its ownership separate from the still processor even where
  current values happen to agree.

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

### Gate 2 — run the bounded one-4090 feasibility pilot

After Gate 1 passes, run the smallest experiment through the real pinned
`llm-compressor` sequential path that reveals the relevant resource behavior
without producing a launchable candidate.

Measure peak allocated and reserved VRAM, peak host RAM, activation/intermediate
cache placement and growth, replay behavior, time by observable stage, temporary
disk use, and interruption/OOM cleanup. Include representative single-, multi-,
mixed-keyframe/reference-, reference-video-, and 2048-upscale stress rows. The
pilot must have a deliberate abort/failure control and mark every partial output
non-launchable.

The pilot sets the total-token budget and therefore the absolute calibration
population. It does not change the accepted role partition.

### Gate 3 — freeze the executable split and launch package

Using Gate 2's measured budget:

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

- **New Claude, v2 technical lead:** exact seam, full-load validation,
  feasibility pilot, split/trace integration, recipe and launcher, run
  supervision, and candidate artifact audit.
- **Codex:** canonical authority, independent preflight review, privacy and
  provenance gates, BF16/W4 capture and comparison, and the go/no-go
  recommendation to the owner.
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
