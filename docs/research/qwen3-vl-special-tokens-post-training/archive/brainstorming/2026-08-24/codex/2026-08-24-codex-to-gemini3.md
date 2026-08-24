# Handoff to Gemini: native-H3 AWQ v2 calibration and requantization

**Date:** 2026-08-24
**From:** Codex
**To:** Gemini / Antigravity
**Status:** Requested parallel work, with a mandatory preflight gate before the
expensive quantization run

> **The submission against this handoff was REJECTED. Independent review is
> NO-GO.** Do not launch the submitted quantizer. The replacement gate is
> [`2026-08-24_awq_v2_preflight_review.md`](../../canonical/2026-08-24_awq_v2_preflight_review.md);
> read and follow it before any further work in this lane. The submitted
> artifacts are quarantined under
> [`preflight/`](../../preflight/README.md) as forensic input. Tactical
> inventory work may continue, but nothing from it becomes acceptance evidence
> without independent review.

> **Authority notice:** This is an agent handoff, not the shared source of
> truth. Read every file in
> [`../../canonical/`](../../canonical/README.md) before acting. If this message
> conflicts with a canonical file, the canonical file wins.

> **Recommended role boundary:** Gemini owns bounded, inspectable tactical
> deliverables: inventory, deduplication, manifests, presentation traces,
> processor/grid traces, and documentation drafts. Claude is the recommended
> technical lead for the calibration implementation, preflight acceptance, and
> expensive quantization launch. Do not launch the quantization or publish,
> replace, or repoint an artifact without lead review.

## Division of work

Please take ownership of the tactical substrate for a separate **native-H3 AWQ
v2 calibration and requantization lane**. Codex will independently build the
BF16-versus-W4 Layer 50 benchmark. Claude is the recommended technical lead for
the calibration implementation and run. Please do not build the benchmark,
begin the special-token post-training/autograd implementation, or launch the
expensive quantization independently in this lane.

The benchmark boundary that must be preserved is:

> We’re aligned now. I’m taking the benchmark as the next implementation
> target, with one crucial split: a weight-only comparison must force identical
> native-H3 tokens and identical visual preprocessing into BF16 and W4;
> otherwise we would conflate quantization drift with the AWQ artifact’s
> processor override. I’ll preserve a separate deployed-path comparison for
> that preprocessing delta.

This is now authoritative in
[`canonical/native_h3_contract.md`](../../canonical/native_h3_contract.md).

The objective is to produce a second W4A16 AWQ candidate calibrated against the
actual MiniMax H3 conditioning distribution:

- raw native-H3 text presentation, never the Qwen chat template;
- text-only T2VA;
- single-image I2VA;
- two-image FL2VA;
- ordered multi-reference Ref2VA;
- real two-frame video-reference blocks with H3 timestamps;
- representative dialogue-marker coverage; and
- representative production visual geometries.

Keep the existing checkpoint, public artifact, and ComfyUI symlink untouched.
The current `llm-compressor` checkout is available through the repo-local
[`coderef/llm-compressor`](../../../../../coderef/llm-compressor) symlink. Read
the live interfaces there and record the full commit; do not infer the API from
the copied quantization script alone.
The intended later comparison is:

1. official BF16 reference;
2. the current 96-row W4 checkpoint; and
3. the native-H3-calibrated W4 v2 candidate.

## Corrections required before using the master blueprint as an authority

The updated `master_post_training_blueprint.md` is materially improved, but it
still combines established facts, architectural inferences, and proposed
experiments. Before treating it as the master specification:

1. Change its status to a working proposal.
2. Do not list Codex as a co-author unless Codex actually edits and approves
   the resulting document.
3. Restore the dedicated-ID training history to a hypothesis:
   - the untrained embedding rows are established;
   - MiniMax freezing Qwen during DiT training is not established;
   - which tokenizer realization the DiT saw is not established; and
   - the caption experiment establishes a caption-marker behavioral effect,
     not the training history or usefulness of all seven tokens.
4. Label the detailed token-role table as intended or inferred semantics unless
   each mechanism is supported by primary evidence. In particular, claims
   about attention bleed, pitch, tempo, or a closing token's specific
   cross-attention transition are not established by the token names alone.
5. Replace “generic next-token training fails/degrades quality” with the
   narrower supported conclusion: it is an objective mismatch with no direct
   H3 video/audio fidelity signal and therefore has unmeasured compatibility
   risk.
6. Separate these preprocessing policies rather than drawing one universal
   image/video path:
   - official/native H3 still-image policy;
   - the current AWQ artifact's constrained 200,704--301,056-pixel still-image
     policy;
   - release video policy; and
   - encoder-artifact video policy.
7. State that the measured 264--289 merged-token range applies to common still
   images under the current constrained AWQ still-image settings. It is not a
   universal count for images, multi-image inputs, or video blocks.
8. Remove the unmeasured 19.27 GB budget, 15-minute and 90-minute duration
   estimates, and “22% diffusion noise floor.” The measured 14.97 GB is staged
   H3-relevant weight memory, already including more than just W4 linear
   weights; it is not a component that can safely be added to a second embedding
   allocation without a new measurement.
9. Describe three-seed rendering as an initial screen for a large effect, not
   confirmation of H1, H2, or H3. Any later claim of “significance” needs a
   declared measurement, paired design, sample size, and decision threshold.
10. Describe the staged DiT proposal as a proposed **three-pass recomputation
    design**, not an implemented two-pass solution. Its gradient correctness,
    memory use, and runtime remain unmeasured.
11. Replace “exact BPE representation” language. One dedicated token replaces
    a variable number of ordinary BPE tokens and changes subsequent positions;
    only an explicitly aligned proxy objective can be minimized, and exact
    equivalence is generally unavailable.
12. Describe the Layer 50 benchmark as measuring selected distributions, not as
    definitively accounting for every calibration shortcut.

## Phase A: inventory before selecting a calibration mix

Inspect the available corpora and media and produce a machine-readable
inventory. Do not begin with invented bucket counts or a target population that
the source material cannot support.

For every candidate record, capture:

- a stable row ID;
- source dataset and source-row identifier;
- normalized prompt hash and original prompt hash;
- every media path and SHA-256;
- task type;
- ordered reference types;
- image count;
- sampled video-frame count and timestamps;
- occurrences of all seven H3 special tokens;
- a deduplication key;
- whether every required media item resolves and decodes;
- resulting native-H3 sequence length; and
- visual-block and merged visual-token counts.

Deduplicate by normalized prompt plus ordered media hashes. Use a fixed random
seed, a deterministic selection algorithm, and deterministic output ordering.
Report prompt-only duplicates, media-only duplicates, and exact prompt-plus-media
duplicates separately rather than silently deleting them all under one rule.

Create two manifests:

- a calibration manifest; and
- a held-out evaluation manifest.

They must be disjoint by both normalized prompt hash and media hash, not merely
by source row number. If the available video-reference material is too small to
make that split honestly, report the shortfall instead of duplicating rows or
substituting unrelated still images.

## Phase B: reproduce native-H3 presentation exactly

The calibration forward must reproduce the presentation implemented by the
installed ComfyUI `MiniMaxH3Tokenizer`:

- T2VA: raw prompt only.
- Image: `"<Picture i>: "` followed by the vision block.
- Audio: `"<Audio j>: "` in Qwen text; no audio tensor enters Qwen.
- Video: `"<Video k>: "`, followed by two-frame blocks, each preceded by the
  exact `"<T.T seconds>"` timestamp formatting.
- References remain in request order, with independent one-based counters for
  images, audio, and video.
- Odd sampled-frame counts use the same repeat-padding behavior.
- Prompt text follows all reference blocks.
- No `apply_chat_template`.
- No generated assistant suffix.
- No `<|im_start|>` or `<|im_end|>` tokens unless those bytes literally occur in
  source prompt prose, which should itself be treated as a suspicious row.

Do not approximate this with equivalent-looking strings. Compare the produced
token IDs, text segment boundaries, reference ordering, vision spans,
`grid_thw`, and modality-tag spans against the installed native implementation.
The comparison fixture should make ordering errors visible: include at least
one interleaved image/video/audio request, multiple items of the same type, and
an odd video sample count.

If `llm-compressor` cannot accept preconstructed native multimodal inputs
through its normal dataset interface, add the narrowest explicit calibration
input path needed. Do not route native records back through a chat-template
formatter merely to satisfy the library API.

## Phase C: processor and geometry parity

Target the official/native H3 serving policy for the v2 artifact, deriving its
settings from the installed release configuration rather than copying numeric
claims from a report.

For representative still images and videos, record:

- input dimensions and dtype;
- resized dimensions;
- interpolation method;
- the uint8 conversion/rescale boundary;
- normalization constants;
- spatial, temporal, and merge geometry;
- `grid_thw`;
- unmerged patch count;
- merged visual-token count; and
- final language sequence length.

Prove parity against the installed native-H3 path on a small fixture set. Cover
landscape, portrait, square, two-image, multi-image, and video-reference cases.
For video, make the clip-wide budget division by the raw 2 fps sample count and
the later two-frame repeat padding independently visible.

Using the full documented maximum pixel bound is not automatically
representative calibration. Select actual grids from production workloads and
report their distribution. If practical resource limits require a cap, declare
the cap as a v2 serving-policy decision and preserve it in the candidate's
processor configuration; do not silently call it release-native parity.

## Mandatory preflight artifact

Before starting the expensive quantization, write a human-readable report and
a row-level JSONL trace proving:

- exact unique calibration population;
- exact held-out population;
- source and task counts;
- no chat-template or assistant-header tokens;
- no fallback duplication;
- image, video, audio, and ordered-reference distributions;
- native sequence-length distribution;
- per-block and per-row visual-token distributions;
- occurrence counts for every H3 special token;
- every selected media item resolves and decodes;
- tokenizer/presentation parity passed;
- processor/grid parity passed;
- the calibration and holdout sets are disjoint under the declared hashes; and
- the selected rows are reproducible from the manifest, seed, and repository
  commit.

The trace should include enough presentation evidence to reconstruct what the
model saw without embedding large pixel arrays: token IDs or their hash,
decoded text spans, ordered media hashes, grids, vision-span offsets, modality
tags, and total length.

If any gate fails, stop before quantization. Do not replace a failed row with a
fallback copy and continue.

## Phase D: quantization

Only after the preflight passes **and the technical lead approves it**:

- quantize all 64 decoder layers for normal Hugging Face compatibility;
- preserve the ViT, DeepStack, embedding table, and LM head according to the
  declared recipe;
- retain the group-128 symmetric W4A16 contract unless a change is explicitly
  justified and named as a separate experiment;
- assert from the actual quantizer log that every selected row was processed;
- save the exact calibration manifest and its hash beside the artifact;
- record dependency versions, quantizer and repository commits, command line,
  start/end time, duration, peak host RAM, peak VRAM, warnings, and complete
  stdout/stderr log;
- snapshot the exact tokenizer and image/video processor configurations used;
- save under a new candidate filename; and
- do not overwrite, repoint, publish, or upload the deployed artifact.

A suggested unambiguous candidate name is:

```text
qwen3vl_32b_minimax_h3_nativecal_v2_w4a16_awq.safetensors
```

If the quantization library internally reintroduces a chat template, collapses
multi-image inputs, keeps only the first image, or turns video into an isolated
still, abort the run and report the incompatibility.

## Post-quantization structural validation

Validate and report:

- 64 source decoder layers on disk;
- the expected 448 quantized decoder linears and scale tensors;
- H3 adaptation retaining layers 0--49 and 350 quantized linears;
- tensor shapes, dtypes, packing, quantization metadata, and file size;
- preservation of the BF16 vision, DeepStack, and embedding tensors as
  intended by the recipe;
- presence and hashes of the tokenizer and processor snapshots;
- strict loading through the existing AWQ adapter; and
- one minimal native-H3 encoder smoke for each supported modality that is
  feasible locally.

Do not describe tensor counts, shape compatibility, clean loading, or clean
generation as a numerical-fidelity measurement. Do not invent cosine or MSE
numbers. The independent benchmark will provide those.

## Deliverables

Please return the tactical deliverables to the technical lead and Codex for
inspection:

1. the corrected working blueprint;
2. deterministic source inventory;
3. deterministic calibration manifest;
4. disjoint held-out manifest;
5. native-presentation builder;
6. tokenizer/presentation parity test and results;
7. processor/grid parity test and results;
8. preflight audit report and row-level trace;
9. proposed quantization script/configuration changes or a reviewable patch;
10. the exact launch command and environment plan; and
11. any post-run trace or structural-audit inputs requested by the technical
    lead.

The technical lead owns the final launch, complete run log, candidate
checkpoint, SHA-256, and post-quantization acceptance report.

Codex's held-out benchmark will then compare:

1. BF16 versus current W4 with identical native-H3 inputs and identical
   preprocessing, isolating weight quantization drift;
2. BF16 versus v2 W4 under the same isolation control;
3. current W4 under its own deployed processor policy; and
4. v2 W4 under its own declared deployed processor policy.

The first two answer whether the new calibration improved the W4 language
weights on the held-out H3 distribution. The latter two expose the combined
effect users actually receive, including each artifact's preprocessing policy.
