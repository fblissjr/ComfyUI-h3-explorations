# Codex to new Claude: native-H3 AWQ v2 technical lead

**Date:** 2026-08-24
**Status:** Current assignment handoff. Canonical records override this memo.

You are the technical lead for one bounded lane: prove, measure, and prepare the
replacement native-H3 AWQ v2 quantization path. Do not take on the
reference-node redesign, marker-corpus authoring, layer-depth ablation, or
special-token trainer. Those are separate lanes or deferred experiments.

Work from the repository root. Do not write machine-local home paths into code,
manifests, results, or documentation.

## Read before acting

Read `CLAUDE.md`, then every file in
[`canonical/`](../../canonical/README.md). The execution authority is
[`active_plan.md`](../../canonical/active_plan.md); the accepted data boundary
is [`calibration_data_pool.md`](../../canonical/calibration_data_pool.md).

Inspect primary code rather than relying on prior agent summaries:

- [`h3_awq_encoder.py`](../../../../../h3_awq_encoder.py)
- [`capture_h3_encoder_states.py`](../../../../../bench/capture_h3_encoder_states.py)
- [`compare_h3_encoder_captures.py`](../../../../../bench/compare_h3_encoder_captures.py)
- [`probe_calibration_input_seam.py`](../../../../../bench/probe_calibration_input_seam.py)
- [`build_h3_calibration_pool.py`](../../../../../bench/build_h3_calibration_pool.py)
- [`coderef/llm-compressor`](../../../../../coderef/llm-compressor)
- [`coderef/sglang`](../../../../../coderef/sglang)
- [`coderef/DiffSynth-Studio`](../../../../../coderef/DiffSynth-Studio)
- [`coderef/MiniMax-H3`](../../../../../coderef/MiniMax-H3)
- installed ComfyUI MiniMax/Qwen3-VL encoder code; and
- the local BF16 source plus deployed AWQ artifact metadata.

Record full revisions for every moving checkout and installed runtime package.
A symlink name is not a version.

## Your scope

Own Gates 1--3 of the canonical plan:

1. prove the exact installed-native-H3-to-`llm-compressor` calibration seam;
2. run the bounded one-4090 feasibility pilot;
3. construct the deterministic component-safe split and reviewed launch
   package; and
4. only after Codex review and explicit owner approval, supervise the full
   candidate run and audit its artifact.

Your first deliverable is the Gate 1 audit and executable seam proof. It is not
a quantization launcher and must not create a candidate checkpoint.

## Closed findings to consume

Do not repeat these probes unless a new revision or escaped defect changes the
boundary:

- M-RoPE position IDs match exactly between the tested Transformers and ComfyUI
  implementations.
- Vision merged output and DeepStack arithmetic match to float32 rounding in
  the seeded parity fixtures.
- The current W4 artifact cannot be repaired by widening its processor config:
  held-out BF16-versus-W4 layer-50 fidelity worsened under the release bounds.
- One `oneshot` sequential trace can accept variable image counts, grids, and
  native two-frame video blocks, but it cannot safely mix vision-bearing and
  text-only rows.
- The first Gemini preflight and launcher are rejected. They may be read as a
  failure record, not used as implementation substrate.
- MiniMax's released H3 interface is the raw unnormalized state after decoder
  layer 49. The output depth is not an open v2 parameter.
- The W4 candidate keeps the full input embedding, vision tower, and DeepStack
  in BF16 while quantizing all 64 decoder layers on disk.

## Locked input and serving decisions

- Use only the accepted H3-IR candidate pool for this run.
- Preserve the accepted mutually exclusive role partition and treat dialogue,
  aspect, audio-label, and small-source properties as overlays.
- Split whole exact-media connected components. The absolute row count remains
  open until your feasibility pilot establishes the total-token budget.
- Exclude text-only rows from the vision trace and retain them as a held-out
  T2VA regression arm.
- Size keyframes at the target H3 canvas and preserve their temporal prose.
- Size ordinary Ref2VA stills primarily with `max` and no upstream upscale.
- Keep a separately named 2048-short-edge upscale stress stratum.
- Size reference video with the release 768-short-edge / 1,032,192-pixel role
  policy and native duration-aware two-frame presentation.
- Use the release-declared Qwen still and video processor contracts for both
  calibration and v2 serving. Snapshot the configs with the candidate.
- Record upstream and Qwen-effective geometry per media item. Mixed
  keyframe-plus-reference rows require per-picture roles and policies.

Do not make a universal still cap substitute for the role policy. Do not shrink,
duplicate, replace, or silently exclude an accepted media item to make the run
fit.

## Gate 1: exact seam proof

Use a preconstructed dataloader to bypass library formatting, sampling, and
collation. Instrument or reuse the installed native-H3 presentation path; do
not independently reimplement labels, timestamp formatting, patch geometry,
reference ordering, or resize arithmetic and then validate that implementation
against itself.

For real decoded still, multi-reference, mixed keyframe/reference, and
reference-video fixtures, demonstrate identity for:

- raw prompt bytes and input IDs;
- released special-token IDs;
- ordered `<Picture i>`, `<Video i>`, `<Audio i>`, and timestamp labels;
- H3 token tags and Transformers `mm_token_type_ids`;
- decoded media hashes, processed patch hashes, grids, and vision spans;
- attention masks, M-RoPE inputs, and DeepStack injection;
- upstream and effective processor geometry;
- the raw layer-50 tap; and
- the exact object consumed by the traced `oneshot` graph.

Also prove strict full-checkpoint mapping and loading into the actual
Transformers calibration model. Small random-model arithmetic parity does not
cover this boundary.

Every claim needs a control that fails for the intended defect. Required
mutations include chat framing, first-image slicing, reference reorder,
timestamp change, missing temporal repeat, grid change, media drop, tag/type
change, and disconnecting the validated builder from the dataloader consumed by
`oneshot`.

## Gate 2: feasibility pilot

After Gate 1 is green, design the smallest real sequential-path pilot that
reveals the 32B run's resource behavior without emitting a launchable
checkpoint. It must include representative role families and the 2048-upscale
stress case.

Measure:

- peak allocated/reserved VRAM;
- peak host RAM and growth with calibration tokens;
- activation/intermediate cache placement and size;
- graph replay or error-propagation passes;
- observable stage timing;
- temporary disk use; and
- cleanup/recovery after a deliberate abort or controlled failure.

Record the exact environment and make partial output unmistakably
non-launchable. Do not estimate the final population or runtime from source
reading.

## Gate 3: split and launch package

Use the measured total-token budget to produce deterministic calibration,
holdout, and rejection manifests. Assign whole media components, reserve the
required small-source holdout, report achieved role/overlay shares, verify all
local media hashes, and perform the required near-duplicate review.

The row trace must come from the exact post-policy dataloader the launcher will
consume. Instantiate the complete AWQ plus quantization recipe in the pinned
environment before requesting a 32B launch. Prove all 64 decoder layers are
targeted and BF16 tensors remain outside the quantized target set.

The launcher must write only to a new explicit directory, refuse an existing or
deployed target, leave the current symlink untouched, and stamp portable logical
identifiers rather than machine-local paths.

## Collaboration and authority

- Write proposals and status under
  `docs/research/qwen3-vl-special-tokens-post-training/brainstorming/claude-encoder/`.
- Promote a finding to `canonical/` only after Codex reviews its evidence and
  bounds.
- Codex owns the independent preflight review and later BF16/W4 numerical
  evaluation.
- The repository owner owns GPU scheduling and final launch approval.
- Gemini is offline and paused; no task or dependency should be assigned to
  him.
- The existing reference-node Claude lane is outside your critical path.
  Consume its role-aware geometry contract without changing its nodes.

## Preservation and stop conditions

Do not launch full quantization, edit or repoint the deployed artifact, publish
anything, patch the BF16 source, or build a special-token trainer before the
canonical gates say so.

Stop and report if the validated batches differ from what `oneshot` consumes;
media is silently altered; the component-safe split cannot fit the measured
budget; strict loading or recipe construction fails; a failure control stays
green; output can collide with the deployed artifact; or a memory, time,
fidelity, or quality claim would be inferred rather than measured.

## First response and first deliverable

After reading the required material, reply with:

1. the exact scope you accept;
2. which closed findings you will consume without repeating;
3. the smallest Gate 1 implementation you intend to build;
4. the mutation controls that make it capable of failing; and
5. any concrete blocker that the canonical plan does not already resolve.

Then write the Gate 1 audit and executable seam proof. Do not begin the
feasibility pilot until Codex has reviewed that first deliverable.
