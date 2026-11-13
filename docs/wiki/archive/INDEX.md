# Archived Docs (`docs/wiki/archive/`)

> **Frozen / superseded documents.** Pages here were moved out of active wiki lanes so historical reasoning and retraction ledgers stay intact without confusing current work. Do not treat filenames or rules here as live unless verified against the codebase and [`../index.md`](../index.md).

## Retracted Claims & Historical Ledgers

| File | Historical Title | Superseded / Owned By |
| :--- | :--- | :--- |
| [`2026-08-22_retracted_claims_part1.md`](2026-08-22_retracted_claims_part1.md) | Retracted Claims Ledger — Part 1 (Rows 1–20, August 2026) | [`../ops/evidence.md`](../ops/evidence.md) |
| [`2026-09-05_retracted_claims_part2.md`](2026-09-05_retracted_claims_part2.md) | Retracted Claims Ledger — Part 2 (Rows 21–40, August–October 2026) | [`../ops/evidence.md`](../ops/evidence.md) |
| [`rules_history.md`](rules_history.md) | `CLAUDE.md` as it stood before the 2026-09-03 cut | [`../../../AGENTS.md`](../../../AGENTS.md), [`../index.md`](../index.md) |
| [`comfyui_vendor_gaps.md`](comfyui_vendor_gaps.md) | Where ComfyUI's H3 path differs from the vendor's | [`../core/custom_node_gaps.md`](../core/custom_node_gaps.md) |
| [`config_drift.md`](config_drift.md) | Why `h3_config.py` drifts, and what would actually stop it | [`../ops/evidence.md`](../ops/evidence.md), [`../ops/checks.md`](../ops/checks.md) |
| [`check_postmortems.md`](check_postmortems.md) | Check postmortems: what each defect taught | [`../ops/checks.md`](../ops/checks.md), [`../ops/checks/standard_and_gaps.md`](../ops/checks/standard_and_gaps.md) |
| [`sustainability.md`](sustainability.md) | Keeping this repo worth working in | [`../../AGENTS.md`](../../AGENTS.md), [`../projects/index.md`](../projects/index.md) |

## Historical Handovers, Queues & Plans

| File | Historical Title | Superseded / Owned By |
| :--- | :--- | :--- |
| [`2026-08-20_roadmap_regime_and_ordering.md`](2026-08-20_roadmap_regime_and_ordering.md) | Historical Regime Question, Token Ordering & August 2026 Queue | [`../ops/roadmap.md`](../ops/roadmap.md) |
| [`2026-09-05_roadmap_forward_plans.md`](2026-09-05_roadmap_forward_plans.md) | Historical Forward Plans (2026-09-03 to 2026-09-05) | [`../ops/roadmap.md`](../ops/roadmap.md) |
| [`2026-10-15_masked_v2v_chronicle.md`](2026-10-15_masked_v2v_chronicle.md) | Masked video to video: how it works, what it cannot do, where to go next | [`../masked_v2v.md`](../masked_v2v.md), [`../masked_v2v/2026-11-10_simplification_plan.md`](../masked_v2v/2026-11-10_simplification_plan.md) |
| [`2026-10-20_next_steps_history.md`](2026-10-20_next_steps_history.md) | Next Steps — Historical Queues (`2026-09-05` through `2026-10-20`) | [`../next_steps.md`](../next_steps.md), [`../decisions.md`](../decisions.md) |

## Sparse Attention (`Sol`), Quantization & Kernel Surveys

| File | Historical Title | Superseded / Owned By |
| :--- | :--- | :--- |
| [`bench_plan.md`](bench_plan.md) | Sol-Attn bench plan, from 2026-08-14 | [`../attention/SOLATTN.md`](../attention/SOLATTN.md) |
| [`morton.md`](morton.md) | Morton ordering in Sol-Attn: what it does, and what we actually know | [`../attention/SOLATTN.md`](../attention/SOLATTN.md) |
| [`2026-09-01_session_sol-route-observability_explainer.html`](2026-09-01_session_sol-route-observability_explainer.html) | Sol Routing, Observed (standalone interactive explainer) | [`../attention/SOLATTN.md`](../attention/SOLATTN.md) |
| [`2026-09-03_sol_exact_pquant_and_base_capture.md`](2026-09-03_sol_exact_pquant_and_base_capture.md) | Sol exact-P-quantization and the Base16 capture | [`../attention/h3_quant_policy.md`](../attention/h3_quant_policy.md) |
| [`2026-09-04_sol_token_aug_grade.md`](2026-09-04_sol_token_aug_grade.md) | Sol-Attn token routing (`token_aug`) graded on the Base16 capture | [`../attention/SOLATTN.md`](../attention/SOLATTN.md) |
| [`2026-09-05_token_aug_plan.md`](2026-09-05_token_aug_plan.md) | Plan: token routing (`token_aug`) from graded lever to shippable per-block policy | [`../attention/SOLATTN.md`](../attention/SOLATTN.md) |
| [`2026-09-17_rotation_and_lowbit_attention_survey.md`](2026-09-17_rotation_and_lowbit_attention_survey.md) | Rotation and low-bit attention for MiniMax H3 on Ada sm89 — research survey | [`../attention/h3_quant_policy.md`](../attention/h3_quant_policy.md) |
| [`2026-09-19_dsv41_hierarchical_indexer.md`](2026-09-19_dsv41_hierarchical_indexer.md) | DeepSeek V4.1's hierarchical sparse indexer and layer modes, read for MiniMax H3 | [`../attention/SOLATTN.md`](../attention/SOLATTN.md) |
| [`2026-09-19_token_selection_scorers.md`](2026-09-19_token_selection_scorers.md) | Cheap scorers that choose what attention computes exactly, finer than a 64-token block | [`../attention/SOLATTN.md`](../attention/SOLATTN.md) |
| [`2026-09-19_video_sparse_attention_2026.md`](2026-09-19_video_sparse_attention_2026.md) | Four training-free video sparse-attention methods of 2026, read for MiniMax H3 | [`../attention/SOLATTN.md`](../attention/SOLATTN.md) |
| [`2026-09-19_where_approximation_is_tolerated.md`](2026-09-19_where_approximation_is_tolerated.md) | Where approximation is tolerated: steps, blocks and heads chosen by profiling | [`../attention/SOLATTN.md`](../attention/SOLATTN.md) |
| [`2026-09-27_sol_node_redesign.md`](2026-09-27_sol_node_redesign.md) | The Sol node redesign: plan, reasons, and the tests that decide it | [`../attention/SOLATTN.md`](../attention/SOLATTN.md) |
| [`2026-11-04_krea3_report_read_for_h3.md`](2026-11-04_krea3_report_read_for_h3.md) | The Krea 3 technical report, read for H3 | [`../core/h3_distills.md`](../core/h3_distills.md) |

## PDD, Continuation, Distills & Retired Experiments

| File | Historical Title | Superseded / Owned By |
| :--- | :--- | :--- |
| [`h3_pdd.md`](h3_pdd.md) | Parallel Decoding Distillation on MiniMax H3 (PDD retired except Pika) | [`../core/h3_distills.md`](../core/h3_distills.md) |
| [`h3_ref2v_distillation.md`](h3_ref2v_distillation.md) | Why ref2v resists step distillation | [`../core/h3_distills.md`](../core/h3_distills.md) |
| [`pdd_artifacts.md`](pdd_artifacts.md) | PDD artifacts: the master inventory | [`../core/h3_distills.md`](../core/h3_distills.md) |
| [`2026-09-25_continuation_guide_rows.md`](2026-09-25_continuation_guide_rows.md) | Continuing a clip by guide rows: six designs and one to build | [`../core/h3_audio_freeze.md`](../core/h3_audio_freeze.md) |
| [`2026-09-25_step_caching_survey.md`](2026-09-25_step_caching_survey.md) | Step caching for H3, re-examined | [`../core/h3_distills.md`](../core/h3_distills.md) |
| [`2026-09-25_temporal_offset_and_adaln_rounding.md`](2026-09-25_temporal_offset_and_adaln_rounding.md) | A global temporal offset, and AdaLN rounding on the INT8 DiT | [`../core/h3_frame_rules.md`](../core/h3_frame_rules.md) |
| [`2026-09-26_distill_routing.md`](2026-09-26_distill_routing.md) | Routing distills within a clip: six routes, three of them H3's own | [`../core/h3_distills.md`](../core/h3_distills.md) |
| [`2026-09-26_flashgen.md`](2026-09-26_flashgen.md) | FlashGen: what it is, where our settings come from, how upstream runs it | [`../core/h3_distills.md`](../core/h3_distills.md) |
| [`2026-09-19_evaluation_one_judge.md`](2026-09-19_evaluation_one_judge.md) | Evaluation for one judge and one GPU: accelerated video+audio | [`../ops/eval_comparison.md`](../ops/eval_comparison.md) |
| [`2026-09-19_in_context_reference_and_sinks.md`](2026-09-19_in_context_reference_and_sinks.md) | In-context conditioning, and an open sweep | [`../core/h3_references.md`](../core/h3_references.md) |
| [`drift_frontier.md`](drift_frontier.md) | Drift work: the open frontier | [`../core/h3_audio_freeze.md`](../core/h3_audio_freeze.md) |
| [`prompt_length_experiment.md`](prompt_length_experiment.md) | Prompt length, paired render: pre-registration | [`../prompting/prompting.md`](../prompting/prompting.md) |
| [`scene_arm_renders.md`](scene_arm_renders.md) | The two scene arms: what to look for, and what these renders cannot tell you | [`../ops/eval_comparison.md`](../ops/eval_comparison.md) |
