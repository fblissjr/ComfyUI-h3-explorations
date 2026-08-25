# Canonical H3 special-token research record

**Status:** Authoritative index
**Created:** 2026-08-24
**Maintainers:** repository owner and the technical point (session
"v2-lead") since 2026-08-25; Codex's accepted records stand

This directory is the shared source of truth for the Qwen3-VL special-token and
AWQ calibration work. A statement is not authoritative merely because it
appears elsewhere in this research tree.

## Authority boundary

Files belong here only when they are one of the following:

- a measured or source-verified fact that both work lanes may rely on;
- an explicit owner decision;
- an accepted execution plan or interface contract;
- a generated evidence artifact whose provenance and limitations are recorded;
  or
- a correction to a previously overstated claim.

Keep these outside `canonical/`:

- agent-to-agent handoffs and status messages;
- brainstorming and candidate ideas;
- pseudocode and unimplemented designs;
- unapproved training or quantization recipes;
- synthetic examples presented for discussion;
- timing, memory, fidelity, or quality estimates that have not been measured;
  and
- forensic drafts that still contain retracted claims.

Outside documents may be useful evidence or historical context, but they must
not override a conflicting statement in this directory.

## Canonical files

| File | Authority |
|---|---|
| [`baseline.md`](baseline.md) | Established facts, bounded owner observations, and explicitly open questions. |
| [`native_h3_contract.md`](native_h3_contract.md) | Presentation, preprocessing, and comparison rules that implementations must preserve. |
| [`active_plan.md`](active_plan.md) | Accepted execution sequence, locked decisions, ownership, gates, and stop conditions. |
| [`2026-08-24_awq_v2_preflight_review.md`](2026-08-24_awq_v2_preflight_review.md) | Independent rejection of the first Gemini v2 preflight artifact set and the replacement gate. |
| [`2026-08-24_calibration_input_seam.md`](2026-08-24_calibration_input_seam.md) | Which native-H3 modalities can reach AWQ calibration, the trace constraint that decides it, and the preprocessing policies a v2 artifact must choose between. |
| [`encoder_depth_and_embedding.md`](encoder_depth_and_embedding.md) | The released layer-50 conditioning contract, verified local variant inventories, embedding precision, and the boundary for depth/pruning ablations. |
| [`2026-08-24_still_policy_token_cost.md`](2026-08-24_still_policy_token_cost.md) | Measured raw-input Qwen sequence/cache cost and cap frontier, plus the source-verified distinction between the generated canvas and the 2048-short-edge Ref2VA image envelope. |
| [`2026-08-24_keyframe_vs_reference_positioning.md`](2026-08-24_keyframe_vs_reference_positioning.md) | How the DiT packs keyframes versus references, why that frees a reference from canvas geometry, and why the calibration strata are geometry strata rather than presentation strata. |
| [`2026-08-24_serving_geometry_composes.md`](2026-08-24_serving_geometry_composes.md) | Upstream reference sizing and the encoder cap compose; what that requires of the v2 manifest schema and decision order, and how the encoder cap would be changed if that is decided. |
| [`2026-08-24_layer50_processor_policy_benchmark.md`](2026-08-24_layer50_processor_policy_benchmark.md) | Measured BF16-versus-current-W4 layer-50 baseline and the decision not to widen the deployed artifact's image budget as a config-only repair. |
| [`2026-08-24_transformers_comfy_parity.md`](2026-08-24_transformers_comfy_parity.md) | Measured M-RoPE and vision/DeepStack arithmetic parity, the released-weight precision gap, and the accepted calibration-only `comfy_exact` policy. |
| [`2026-08-24_gate1_seam_acceptance.md`](2026-08-24_gate1_seam_acceptance.md) | Accepted native-H3-to-`llm-compressor` Gate 1/1B evidence, the fail-closed effective attention-mask rule, and the Gate 2 boundary. |
| [`2026-08-24_gate2_readiness.md`](2026-08-24_gate2_readiness.md) | End-of-day Gate 2 boundary: closed substrate findings, superseded diagnostics, corrected Gate 2A acceptance requirements. Its pending part is closed by the next row. |
| [`2026-08-25_gate2_arrangement.md`](2026-08-25_gate2_arrangement.md) | Gate 2A accepted; the storage and kernel axes measured; the arrangement Gate 2B runs on, its entry contract, and which decisions were made under acting-point authority while Codex is offline. |
| [`2026-08-25_v2_launch_record.md`](2026-08-25_v2_launch_record.md) | The first v2 launch's host OOM and the measured per-token host budget; the disk-tier arrangement and the control it exposed; the rebuilt split and the family map's catch; what launched. |
| [`owner_authored_marker_corpus.md`](owner_authored_marker_corpus.md) | Accepted construction contract for new multimodal marker evaluation and candidate training/development corpora compiled from semantically fixed scene specifications. |
| [`calibration_data_pool.md`](calibration_data_pool.md) | Accepted H3-IR candidate-pool partition, overlay coverage, exact-media component constraint, rights boundary, and remaining split checks. |

No post-training recipe is canonical yet. The first proposed v2 calibration
manifest and launcher were independently rejected. Gate 1, Gate 1B and Gate
2A are accepted; Gate 2B and Gate 3 were executed in sprint form and the v2
calibration launched on 2026-08-25
([`2026-08-25_v2_launch_record.md`](2026-08-25_v2_launch_record.md)). Until
Gate 5 accepts the candidate it is a new directory and nothing else: the
deployed artifact and its symlink are unchanged.

## Evidence labels

Canonical prose uses these labels when the evidence class matters:

- **MEASURED:** produced by an inspected script, artifact, manifest, or run.
- **SOURCE:** read directly from installed code or released configuration.
- **OWNER-OBSERVED:** supplied by the owner from a completed human review whose
  scope is stated, but whose full measurement record may not yet be written.
- **OWNER-DECISION:** an explicit project objective or choice made by the
  repository owner; it governs subsequent plans but is not a measured result.
- **INFERENCE:** the evidence supports the interpretation but does not directly
  record it.
- **UNKNOWN:** not established and must not be silently promoted to fact.
- **PLAN:** accepted work to perform, not a completed result.

## Editing protocol

1. Read all canonical files before changing a canonical claim or implementing
   work that relies on one.
2. Cite the concrete source, artifact, or owner decision next to a new fact.
3. State the population, comparison boundary, and limitations of a measurement.
4. Never replace `UNKNOWN` with a historical narrative merely because that
   narrative is architecturally plausible.
5. Put proposed text outside this directory until it is accepted. Once
   accepted, promote only the supported part rather than copying an entire
   brainstorming document.
6. When a canonical claim changes, correct or clearly supersede conflicting
   non-canonical summaries that could still mislead a reader.
7. Preserve the current deployed checkpoint and symlink unless the owner
   explicitly approves replacement after evaluation.
