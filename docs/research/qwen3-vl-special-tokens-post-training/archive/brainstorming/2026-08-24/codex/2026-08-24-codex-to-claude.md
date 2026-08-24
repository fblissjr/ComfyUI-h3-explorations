# Handoff to Claude: technical lead for native-H3 AWQ v2

**Date:** 2026-08-24
**From:** Codex
**To:** Claude
**Status:** Recommended technical-lead assignment

> **Authority notice:** Read all files in
> [`../../canonical/`](../../canonical/README.md) before inspecting or changing
> implementation. Those files override all brainstorming documents and agent
> summaries.

## Requested role

Please take technical point on a new native-H3-calibrated Qwen3-VL 32B W4A16
AWQ candidate for a single RTX 4090 workflow.

Codex owns the independent BF16-versus-W4 Layer 50 benchmark. Gemini owns
bounded tactical inputs: corpus inventory, deduplication, deterministic
manifests, native-presentation traces, processor/grid traces, and documentation
drafts. Your role is to inspect primary code and artifacts, own the actual
calibration architecture, review those tactical inputs, decide whether the
preflight is valid, supervise the expensive quantization, and audit the result.

This ownership split is deliberate. The previous Gemini cycle was productive
but repeatedly promoted estimates and intended designs into verified facts.
Treat Gemini outputs as candidate evidence to check, not as acceptance by
summary.

## Non-negotiable comparison boundary

Codex's benchmark has two separate questions:

1. **Weight-only drift:** BF16 and W4 must receive identical native-H3 token
   IDs and identical visual preprocessing. Token IDs, ordered references,
   patch/pixel hashes, `grid_thw`, vision spans, total sequence length, and
   token tags must match before layer-50 outputs can be compared.
2. **Deployed-path difference:** BF16 and each W4 artifact use their own
   declared processor policy. This intentionally includes preprocessing delta
   and must not be reported as pure quantization drift.

The v2 artifact must declare and snapshot its target serving processor policy
so both comparisons can be constructed honestly.

## First responsibility: inspect the real quantization seam

Please read the actual successful quantization code, installed
`llm-compressor` interfaces, source BF16 layout, adapter, native Comfy H3
tokenizer, and reference-conditioning implementation. Establish:

The current live checkout is available through
[`coderef/llm-compressor`](../../../../../coderef/llm-compressor). Record its
full commit in the review and run artifacts; the symlink path itself is not a
version pin.

- how calibration batches enter `AWQModifier`;
- whether the library can accept preconstructed native multimodal model inputs;
- where the old path called `apply_chat_template`;
- where it discarded all but `images_list[0]`;
- how multiple images and two-frame video blocks must be represented to the HF
  Qwen3-VL forward;
- whether sequential calibration/offload behavior remains feasible on the
  4090 for the proposed sequence lengths; and
- which processor configuration the v2 artifact will own at serving time.

Do not approve a wrapper that constructs native-looking text and then routes it
through a library path that silently adds the chat template again.

## Preflight acceptance

Review Gemini's inventory and manifest generator against primary data. Require:

- deterministic selection and stable row IDs;
- explicit duplicate accounting;
- calibration/holdout disjointness by normalized prompt and media hashes;
- readable, correctly ordered media;
- honest task and modality counts;
- exact native token presentation with no assistant header;
- parity against installed `MiniMaxH3Tokenizer` on adversarial ordering cases;
- parity of image/video geometry, resize boundary, grids, and token spans;
- row-level input traces sufficient to reconstruct what the model saw;
- no silent fallback duplication or first-image slicing; and
- a resource-feasible population selected from actual inventory rather than a
  prewritten bucket diagram.

If the preflight fails, stop. Correct it before consuming a full quantization
run.

## Quantization and preservation rules

Once the preflight passes:

- keep the existing checkpoint and ComfyUI symlink untouched;
- quantize all 64 language decoder layers for HF compatibility;
- retain the declared group-128 symmetric W4A16 recipe unless a separate
  experiment explicitly changes it;
- preserve the vision tower, DeepStack mergers, embedding table, and LM head as
  declared by the recipe;
- assert from the actual run that every manifest row was processed;
- snapshot the exact tokenizer and still/video processor configs;
- record dependency commits/versions, command, complete log, duration, peak
  host RAM/VRAM, warnings, manifest hash, and configuration hashes; and
- write a new candidate artifact without upload or deployment.

Abort if the live path reintroduces chat formatting, collapses multiple images,
converts video calibration to a single still, or substitutes failed rows.

## Post-run audit

Verify the on-disk 64-layer/448-linear structure, BF16 tensors intended to be
preserved, metadata/config snapshots, SHA-256, and strict loading through the
existing H3 AWQ adapter. A small modality smoke may prove execution only.

Do not claim fidelity from shapes, loading, or clean renders. Hand the candidate
and held-out manifest to Codex's benchmark for numerical comparison against
BF16 and the current W4 artifact.

## Deliverables

Please return:

1. reviewed calibration architecture and exact library seam;
2. accepted or rejected tactical preflight with reasons;
3. final deterministic calibration and holdout manifests;
4. reviewed native-presentation and processor parity evidence;
5. exact quantization patch/configuration and launch command;
6. full run/environment record;
7. candidate checkpoint and SHA-256; and
8. structural acceptance report with no numerical-fidelity overclaim.
