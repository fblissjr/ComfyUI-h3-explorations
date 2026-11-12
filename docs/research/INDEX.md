# Research & Postmortems Router (`docs/research/`)

Router index for every research memo, masked-v2v architecture note, postmortem, PDD study, and sparse-attention investigation under `docs/research/`. For active repo rules and settled findings, start at [`docs/wiki/index.md`](../wiki/index.md) and [`docs/evidence.md`](../evidence.md). Also see [`docs/drift_frontier.md`](../drift_frontier.md), [`docs/sol_upstream_qk_balance_pr.md`](../sol_upstream_qk_balance_pr.md), and [`docs/CLAUDE.md`](../CLAUDE.md) ([`docs/AGENTS.md`](../AGENTS.md)).

## Masked V2V Research (`docs/research/masking/`)

| Document | Topic |
|---|---|
| [`masking/2026-10-07_mryolk.md`](masking/2026-10-07_mryolk.md) | SAM 3.1 in ComfyUI, from input to output |
| [`masking/2026-10-07_mryolk_phrases.md`](masking/2026-10-07_mryolk_phrases.md) | Phrases for SAM 3.1: what to give it and how specific to be |
| [`masking/2026-10-07_mryolk_stage_table.md`](masking/2026-10-07_mryolk_stage_table.md) | Stage table: is ComfyUI's SAM 3.1 code the same computation as Meta's? |
| [`masking/2026-10-09_mrfetch.md`](masking/2026-10-09_mrfetch.md) | 2026-10-09, mrfetch: three questions from the whole-frame lane, answered from code |
| [`masking/2026-10-18_sam31_minimal_core_fixes.md`](masking/2026-10-18_sam31_minimal_core_fixes.md) | 2026-10-18: Minimal Upstream Fixes for SAM 3 / 3.1 in ComfyUI Core |
| [`masking/2026-10-24_mrfabez_perception_models_response.md`](masking/2026-10-24_mrfabez_perception_models_response.md) | 2026-10-24: a reviewer's account of the perception models, checked against the code |
| [`masking/2026-10-25_context_frames_and_identity_on_2050.md`](masking/2026-10-25_context_frames_and_identity_on_2050.md) | 2026-10-25: the lead's face did not change on 2050, and what the window showed the model |
| [`masking/2026-10-26_code_map_and_simplification_plan.md`](masking/2026-10-26_code_map_and_simplification_plan.md) | Masked V2V Code Map, Traps & Phased Simplification Plan (`2026-10-26` / `2026-10-27`) |

## Control vectors (`docs/research/control_vectors/`)

For the active index of code, `.toml` pair files, LLM authoring, and readout tools, start at [`docs/wiki/control_vectors.md`](../wiki/control_vectors.md).

| Document | Topic |
|---|---|
| [`control_vectors/2026-11-04_h3_control_vectors.md`](control_vectors/2026-11-04_h3_control_vectors.md) | 2026-11-04: control vectors for H3 (contrastive mean-difference directions in the refined text conditioning): hypothesis, the gate, experiments, build plan, and the code facts they rest on |
| [`control_vectors/2026-11-05_control_vector_system_prompt.md`](control_vectors/2026-11-05_control_vector_system_prompt.md) | 2026-11-05: self-contained system prompt for generating multimodal video+audio scene prompts and contrastive-pair `.toml` files for activation control vectors |
| [`control_vectors/2026-11-06_detangle_comparison_and_borrowable_concepts.md`](control_vectors/2026-11-06_detangle_comparison_and_borrowable_concepts.md) | 2026-11-06: `DeTangle` (`matlowai/DeTangle`, by Dylan Matlow) compared against H3 control vectors: what transfers (soft ridge hold attenuation, paired two-number baseline-subtracted evaluation, side-by-side comparison UI) and what does not (full-token covariance whitening in 5376-d, unsteered test-prompt cosine) |
| [`control_vectors/2026-11-06_mainodes_comparison_and_pika_roadmap.md`](control_vectors/2026-11-06_mainodes_comparison_and_pika_roadmap.md) | 2026-11-06: `ComfyUI-MAINodes` (`dials/`, `h3_cond_strength.py`, `concept_lab/`) compared against H3 control vectors, side-effect analysis of what adopting its methods would break, and the 6-item Pika H3 Flex roadmap (single-clause pair compiler, inference-time cross-vector holds, velocity-space audio/video split, `ref2va` 3D VAE reference gate, picture/video media contrast vectors, and `1,4,4,4,4`-aligned temporal curves) |
| [`control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md`](control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md) | 2026-11-06: Anthropic's Jacobian Lens (`J-Lens`, Gurnee, Lindsey et al. July 2026) adapted to H3 post-refiner control vectors: token-0 attention sink rules, broadcast sequence attention value path ($\beta$), `TokenRefiner` singular value spectrum, empirical-covariance Wiener inversion ($J_{\text{Wiener}}$), 7-prefix cloze baseline `logsumexp` readout (`< 15 ms` on CPU), $\alpha \in \{0.0, 0.5, 1.0\}$ hold sweeps, and why naive $\Sigma_{49}^{1.0}$ pullback (`cos = 0.27–0.38`) was replaced by spectral-filtered $\Sigma_{49, \text{filt}}^{0.30}$ (`cos = 0.52–0.64`) |
| [`control_vectors/2026-11-06_fresh_eye_audit_and_holistic_resolution.md`](control_vectors/2026-11-06_fresh_eye_audit_and_holistic_resolution.md) | 2026-11-06: fresh-eye audit of H3 control vectors (`2026-11-05`–`2026-11-06`) and holistic resolution blueprint across hold norms, `vector.encoder` holds, multi-span diff caching, and runtime nodes |
| [`control_vectors/2026-11-06_holistic_resolution_proposal.md`](control_vectors/2026-11-06_holistic_resolution_proposal.md) | 2026-11-06: holistic resolution proposal for H3 control vectors, `J-Lens`, and active worktree consolidation |
| [`control_vectors/2026-11-06_holistic_stack_review_and_invariants.md`](control_vectors/2026-11-06_holistic_stack_review_and_invariants.md) | 2026-11-06: disk-verified review of the H3 control-vector stack and the 5 system-wide invariants |
| [`control_vectors/2026-11-09_control_vector_simplification_plan.md`](control_vectors/2026-11-09_control_vector_simplification_plan.md) | 2026-11-09: control-vector simplification plan, open-problem inventory, and 2-pass velocity `MiniMaxH3EditGuider` design and gate records |
| [`control_vectors/2026-11-10_control_vector_state_history_and_tradeoffs.md`](control_vectors/2026-11-10_control_vector_state_history_and_tradeoffs.md) | 2026-11-10: state history (`2026-11-04`..`2026-11-10`), over-correction audit (why control-vector magic felt reduced and why `EditGuider` sampling is slow), what to keep vs. remove vs. restore, half-baked seams, research directions inventory, and prioritized recommendations |
| [`control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md`](control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md) | 2026-11-12: a reading of "The Chain Lens [interim]" (vgel, Raymond Douglas) and WorkspaceBench with their references (reading notes in [`control_vectors/2026-11-12_chain_lens_refs/`](control_vectors/2026-11-12_chain_lens_refs/)), what on disk can and cannot run the method, and six experiments for H3 readouts and steering: blind readout scoring, the word-readable share of a vector, a whitened whole-word rows table, lens-against-tail agreement, a chain readout through `layers.50..63`, and a lens-coordinate swap |
| [`control_vectors/2026-11-12_new_vectors_three_ways.md`](control_vectors/2026-11-12_new_vectors_three_ways.md) | 2026-11-12: eight control vectors built fresh from one phrase each (the five bank scenes and neutral phrases each pair was made from, rows pooled, lengths, readout words, and what the batch shares), the `MiniMaxH3Steer` settings every clip was rendered with, and the same concepts as a prompt rewrite and by `MiniMaxH3EditGuider` on one prompt and seed; clips not yet judged |
| [`control_vectors/exl3_qwen3vl/runbook.md`](control_vectors/exl3_qwen3vl/runbook.md) | Runbook for quantizing and running `Qwen3-VL-32B-Instruct` (`5.37 bpw` and `4.65 bpw` EXL3) for H3 control-vector batch labelling and `layers.49` activation-steering readout (`label_h3_control_vectors.py`, `build_h3_cal_data.py`) |

## Postmortems (`docs/research/postmortems/`)

| Document | Topic |
|---|---|
| [`postmortems/2026-10-04_feature_lotsofpeopledance-end-to-end.md`](postmortems/2026-10-04_feature_lotsofpeopledance-end-to-end.md) | Postmortem: the lotsofpeopledance masked render, end to end |
| [`postmortems/2026-10-10_span_fun-three-people.md`](postmortems/2026-10-10_span_fun-three-people.md) | Postmortem: 2026-10-10, three people replaced in one clip, eight sessions |
| [`postmortems/2026-10-11_span_fun-lid-marks-one-shot.md`](postmortems/2026-10-11_span_fun-lid-marks-one-shot.md) | Postmortem: 2026-10-11, one shot read as a result, six sessions |
| [`postmortems/2026-10-15_span_fun-kitchen-and-whole-shot.md`](postmortems/2026-10-15_span_fun-kitchen-and-whole-shot.md) | Postmortem, 2026-10-15: the kitchen came apart, and the whole shot came back |
| [`postmortems/2026-10-18_24fps_joined_cut.md`](postmortems/2026-10-18_24fps_joined_cut.md) | 2026-10-18: the joined cut was 23.976 fps, and H3 needs 24 fps end to end |
| [`postmortems/2026-10-18_mom_kitchen_flicker.md`](postmortems/2026-10-18_mom_kitchen_flicker.md) | 2026-10-18: Why `second_hfp_0604h` Fixed the Second Subject's Kitchen Flicker (`fun` `604–1009`) |

## Sparse Attention (`docs/research/sparse/`)

| Document | Topic |
|---|---|
| [`sparse/2026-10-02_sol_sparse_attention_hyperparameter_campaign.md`](sparse/2026-10-02_sol_sparse_attention_hyperparameter_campaign.md) | Sol-Attn Sparse Attention Hyperparameter Campaign & Empirical Optimization Report |
| [`sparse/sol_capture_and_optimization_guide.md`](sparse/sol_capture_and_optimization_guide.md) | End-to-End Guide: Sol-Attn Data Capture & Hyperparameter Optimization |
| [`sparse/sol_dense_blocks_reanalysis.md`](sparse/sol_dense_blocks_reanalysis.md) | Sol-Attn Dense Blocks Re-Analysis and 5-Test Experimental Plan |

## Parallel Decoding Distillation (`docs/research/pdd/`)

| Document | Topic |
|---|---|
| [`pdd/2026-08-27_handoff.md`](pdd/2026-08-27_handoff.md) | Handoff, end of 2026-08-27 |
| [`pdd/2026-08-28_audio_plan.md`](pdd/2026-08-28_audio_plan.md) | Audio under PDD: the plan, for the session that picks this up |
| [`pdd/2026-08-28_handoff.md`](pdd/2026-08-28_handoff.md) | Handoff, 2026-08-28 |
| [`pdd/2026-08-28_scene_complexity.md`](pdd/2026-08-28_scene_complexity.md) | What makes a PDD clip degrade: three accounts refuted, one standing |
| [`pdd/2026-08-29_market_scene_arms.md`](pdd/2026-08-29_market_scene_arms.md) | Market scene, 2026-08-29: three arms, two clean pairs |
| [`pdd/2026-08-30_handoff.md`](pdd/2026-08-30_handoff.md) | Handoff, 2026-08-30 |
| [`pdd/2026-08-31_handoff.md`](pdd/2026-08-31_handoff.md) | 2026-08-31 — what merging a PDD LoRA onto int8 costs, and what still is not known |
| [`pdd/2026-09-05_bake_plan.md`](pdd/2026-09-05_bake_plan.md) | The backbone bake: plan, 2026-09-05 |
| [`pdd/2026-09-25_upstream_pdd_comparison.md`](pdd/2026-09-25_upstream_pdd_comparison.md) | PDD upstream: can we do better than core's and our own? |
| [`pdd/README.md`](pdd/README.md) | PDD diagrams and handoffs |
| [`pdd/artifact_axis.md`](pdd/artifact_axis.md) | What actually makes a distilled clip fall apart |
| [`pdd/arxiv_2607.26004v1_Parallel_Decoding_Distillationfor_Fast_Image_and_Video_Generation.md`](pdd/arxiv_2607.26004v1_Parallel_Decoding_Distillationfor_Fast_Image_and_Video_Generation.md) | arxiv_2607.26004v1_Parallel_Decoding_Distillationfor_Fast_Image_and_Video_Generation |
| [`pdd/audio_under_pdd.md`](pdd/audio_under_pdd.md) | The sigma schedule is what governs PDD quality |
| [`pdd/depth_and_axes.md`](pdd/depth_and_axes.md) | Where in the DiT does PDD matter, and what else can be turned |
| [`pdd/pdd_implementations.md`](pdd/pdd_implementations.md) | PDD: five implementations, and how this lane got where it is |
| [`pdd/queued_arms.md`](pdd/queued_arms.md) | Queued arms, 2026-08-27 |
| [`pdd/tier1_gate.md`](pdd/tier1_gate.md) | Tier 1: what must be true before the observer gets a render |

## Core H3, Sol-Attn, Conditioning & Quantization (`docs/research/`)

| Document | Topic |
|---|---|
| [`2026-09-03_sol_exact_pquant_and_base_capture.md`](2026-09-03_sol_exact_pquant_and_base_capture.md) | Sol exact-P-quantization and the Base16 capture |
| [`2026-09-04_sol_token_aug_grade.md`](2026-09-04_sol_token_aug_grade.md) | Sol-Attn token routing (`token_aug`) graded on the Base16 capture |
| [`2026-09-05_token_aug_plan.md`](2026-09-05_token_aug_plan.md) | Plan: token routing (`token_aug`) from graded lever to shippable per-block policy |
| [`2026-09-17_rotation_and_lowbit_attention_survey.md`](2026-09-17_rotation_and_lowbit_attention_survey.md) | Rotation and low-bit attention for MiniMax H3 on Ada sm89 — research survey |
| [`2026-09-19_dsv41_hierarchical_indexer.md`](2026-09-19_dsv41_hierarchical_indexer.md) | DeepSeek V4.1's hierarchical sparse indexer and layer modes, read for MiniMax H3 |
| [`2026-09-19_evaluation_one_judge.md`](2026-09-19_evaluation_one_judge.md) | Evaluation for one judge and one GPU: accelerated video+audio |
| [`2026-09-19_in_context_reference_and_sinks.md`](2026-09-19_in_context_reference_and_sinks.md) | In-context conditioning, and an open sweep: what 2025-2026 work says about the reference rows, the prefix and the sinks |
| [`2026-09-19_token_selection_scorers.md`](2026-09-19_token_selection_scorers.md) | Cheap scorers that choose what attention computes exactly, finer than a 64-token block (2026 sweep) |
| [`2026-09-19_video_sparse_attention_2026.md`](2026-09-19_video_sparse_attention_2026.md) | Four training-free video sparse-attention methods of 2026, read for MiniMax H3 |
| [`2026-09-19_where_approximation_is_tolerated.md`](2026-09-19_where_approximation_is_tolerated.md) | Where approximation is tolerated: steps, blocks and heads chosen by profiling (2026) |
| [`2026-09-25_continuation_guide_rows.md`](2026-09-25_continuation_guide_rows.md) | Continuing a clip by guide rows: six designs and one to build |
| [`2026-09-25_step_caching_survey.md`](2026-09-25_step_caching_survey.md) | Step caching for H3, re-examined |
| [`2026-09-25_temporal_offset_and_adaln_rounding.md`](2026-09-25_temporal_offset_and_adaln_rounding.md) | A global temporal offset, and AdaLN rounding on the INT8 DiT |
| [`2026-09-26_distill_routing.md`](2026-09-26_distill_routing.md) | Routing distills within a clip: six routes, three of them H3's own |
| [`2026-09-26_flashgen.md`](2026-09-26_flashgen.md) | FlashGen: what it is, where our settings come from, how upstream runs it |
| [`2026-09-27_sol_node_redesign.md`](2026-09-27_sol_node_redesign.md) | The Sol node redesign: plan, reasons, and the tests that decide it |
| [`2026-11-04_pika_h3_flex_weights_and_loras.md`](2026-11-04_pika_h3_flex_weights_and_loras.md) | Pika H3 Flex: the weights against MiniMax-H3, how kijai's ComfyUI LoRAs were made, and which to run on int8 |
| [`2026-11-04_pika_h3_flex_plan.md`](2026-11-04_pika_h3_flex_plan.md) | Pika H3 Flex: the plan in the owner's priority order, the ledger of what holds, and the prompt, frame and reference rules |
| [`2026-11-04_krea3_report_read_for_h3.md`](2026-11-04_krea3_report_read_for_h3.md) | The Krea 3 technical report, read for H3: what could apply, what is already here, what does not |
| [`calibration_checkpoint_resume.md`](calibration_checkpoint_resume.md) | Checkpoint and resume for a sequential AWQ calibration run |
| [`comfyui_h3_t2va_trace.md`](comfyui_h3_t2va_trace.md) | ComfyUI's H3 t2va pipeline, traced end to end |
| [`conditioning_cache.md`](conditioning_cache.md) | A persistent conditioning cache: design note |
| [`conditioning_node_redesign.md`](conditioning_node_redesign.md) | Reference node redesign: fold the fit, and move the ceiling to the conditioner |
| [`conditioning_nodes.md`](conditioning_nodes.md) | Conditioning against the release: what was built, and what was not |
| [`h3_dit_implementations.md`](h3_dit_implementations.md) | The H3 DiT: five implementations of one model, compared |
| [`h3_partition_distance.md`](h3_partition_distance.md) | How far apart fl2va and ref2va actually are |
| [`m3_h3_bridge.md`](m3_h3_bridge.md) | The MiniMax-M3 to H3 conditioning bridge: what, why, how, and why it probably will not work |
| [`marker_arm_binding.md`](marker_arm_binding.md) | Where each marker-corpus arm binds at render time |
| [`merge_requantisation.md`](merge_requantisation.md) | What happens to a LoRA when it is merged onto an int8 module |
| [`official_weights_metadata.md`](official_weights_metadata.md) | What the official release ships, against what ComfyUI does with it |
| [`quant_levers.md`](quant_levers.md) | What can be changed about H3's quantisation, and what each lever needs |
| [`sglang_comparison.md`](sglang_comparison.md) | sglang's H3 serving path against ours |
| [`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) | sglang's MiniMax-H3 pipeline, end to end |
| [`smoothquant_for_attention_qk.md`](smoothquant_for_attention_qk.md) | SmoothQuant's trick, pointed at attention's q and k |
| [`technique_transfer.md`](technique_transfer.md) | What transfers from LLM and ViT serving to H3, and what does not |
