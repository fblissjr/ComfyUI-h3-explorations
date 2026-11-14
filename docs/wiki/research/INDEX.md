# Research & Postmortems Router (`docs/wiki/research/`)

Router index for dated studies, experiment logs, and postmortems under `docs/wiki/research/`. These records state what was measured on their date; for active rules, workflows, and current project ownership, start at [`../index.md`](../index.md), [`../projects/index.md`](../projects/index.md), and [`../ops/evidence.md`](../ops/evidence.md). Retired notes live in [`../archive/INDEX.md`](../archive/INDEX.md).

## Control Vectors, Readouts, Sound Cues & 2D D-Pad (`docs/wiki/research/control_vectors/`)

For active nodes and recipes, start at [`../control_vectors.md`](../control_vectors.md) and [`../edit_guidance.md`](../edit_guidance.md).

| Document | Topic |
|---|---|
| [`control_vectors/2026-11-04_h3_control_vectors.md`](control_vectors/2026-11-04_h3_control_vectors.md) | Control vectors for H3 (contrastive mean-difference directions in refined text conditioning): hypothesis, gate, build plan, and code facts |
| [`control_vectors/2026-11-05_control_vector_system_prompt.md`](control_vectors/2026-11-05_control_vector_system_prompt.md) | Self-contained system prompt for generating multimodal video+audio scene prompts and contrastive `.toml` pairs |
| [`control_vectors/2026-11-06_detangle_comparison_and_borrowable_concepts.md`](control_vectors/2026-11-06_detangle_comparison_and_borrowable_concepts.md) | `DeTangle` compared against H3 control vectors: what transfers (soft ridge holds, paired subtraction, comparison UI) vs. what does not |
| [`control_vectors/2026-11-06_mainodes_comparison_and_pika_roadmap.md`](control_vectors/2026-11-06_mainodes_comparison_and_pika_roadmap.md) | `ComfyUI-MAINodes` compared against H3 control vectors and the 6-item Pika H3 Flex steering roadmap |
| [`control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md`](control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md) | Adapting Anthropic's Jacobian Lens (`J-Lens`) to H3 post-refiner control vectors ($J_{\text{Wiener}}$, token-0 sink, spectral-filtered $\Sigma_{49,\text{filt}}^{0.30}$) |
| [`control_vectors/2026-11-06_fresh_eye_audit_and_holistic_resolution.md`](control_vectors/2026-11-06_fresh_eye_audit_and_holistic_resolution.md) | Fresh-eye audit of control vectors (`2026-11-05`–`11-06`) across hold norms, encoder holds, and runtime nodes |
| [`control_vectors/2026-11-06_holistic_resolution_proposal.md`](control_vectors/2026-11-06_holistic_resolution_proposal.md) | Holistic resolution proposal for control vectors, `J-Lens`, and worktree consolidation |
| [`control_vectors/2026-11-06_holistic_stack_review_and_invariants.md`](control_vectors/2026-11-06_holistic_stack_review_and_invariants.md) | Disk-verified review of the control-vector stack and 5 system-wide invariants |
| [`control_vectors/2026-11-09_control_vector_simplification_plan.md`](control_vectors/2026-11-09_control_vector_simplification_plan.md) | Control-vector simplification plan, open-problem inventory, and `MiniMaxH3EditGuider` design/gates |
| [`control_vectors/2026-11-10_control_vector_state_history_and_tradeoffs.md`](control_vectors/2026-11-10_control_vector_state_history_and_tradeoffs.md) | State history (`11-04`..`11-10`), over-correction audit, half-baked seams, and prioritized recommendations |
| [`control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md`](control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md) | Chain Lens & WorkspaceBench digest, 6 readout/steering experiments, and updated `2026-11-12` execution order (§15) |
| [`control_vectors/2026-11-12_chain_lens_refs/jlens_workspace_paper.md`](control_vectors/2026-11-12_chain_lens_refs/jlens_workspace_paper.md) | Reading notes: J-lens / global workspace paper |
| [`control_vectors/2026-11-12_chain_lens_refs/lens_variants.md`](control_vectors/2026-11-12_chain_lens_refs/lens_variants.md) | Reading notes: J-lens variants and extensions |
| [`control_vectors/2026-11-12_chain_lens_refs/mtp_and_decoding.md`](control_vectors/2026-11-12_chain_lens_refs/mtp_and_decoding.md) | Reading notes: MTP heads, contrastive decoding, and substitutes for a missing MTP head |
| [`control_vectors/2026-11-12_chain_lens_refs/workspacebench_and_verbalizers.md`](control_vectors/2026-11-12_chain_lens_refs/workspacebench_and_verbalizers.md) | Reading notes: WorkspaceBench internals, BPE bug, and trained verbalizers |
| [`control_vectors/2026-11-12_new_vectors_three_ways.md`](control_vectors/2026-11-12_new_vectors_three_ways.md) | Eight fresh single-phrase vectors vs. prompt rewrite vs. `MiniMaxH3EditGuider`, audio cross-check, and batch-share findings |
| [`control_vectors/2026-11-12_exp5_ladder_and_temporal_guide_synthesis.md`](control_vectors/2026-11-12_exp5_ladder_and_temporal_guide_synthesis.md) | Forensic audit of `exp2`–`exp4`, `exp5` 1:1 ladder (`heavy_blizzard` & `watercolor_wash`), temporal/audio guide mechanics, and next priorities |
| [`control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md`](control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md) | Guide, failure modes, and RoPE/flow math for `MiniMaxH3SoundCues`, `MiniMaxH3AppendSpriteTrajectory`, and the `0.33` (`b = 0.50`) release rule |
| [`control_vectors/exl3_qwen3vl/runbook.md`](control_vectors/exl3_qwen3vl/runbook.md) | EXL3 quantization (`5.37` / `4.65 bpw`) runbook for `Qwen3-VL-32B-Instruct` control-vector labelling and `layers.49` steering |
| [`control_vectors/exl3_qwen3vl/h3_control_vector_report.md`](control_vectors/exl3_qwen3vl/h3_control_vector_report.md) | H3 control-vector batch-labelling & `layers.49` readout report |
| [`control_vectors/exl3_qwen3vl/control_vector_viewer.html`](control_vectors/exl3_qwen3vl/control_vector_viewer.html) | Interactive standalone HTML viewer for EXL3 `Qwen3-VL` control-vector labels and activations |

## Timeline Signal & Guide Studies (`2026-11-12`) & Eval Flywheel

| Document | Topic |
|---|---|
| [`2026-11-10_the_eval_flywheel.md`](2026-11-10_the_eval_flywheel.md) | How a month of masked-V2V visual inspection turned into automated rules and checks, and which survived |
| [`2026-11-12_guide_signal_full_length_bars.md`](2026-11-12_guide_signal_full_length_bars.md) | Full-length bar signal given as a guide video on text-to-video: visual bleed vs. audio-window locking across five arms |
| [`2026-11-12_guide_sound_islands_and_kinematic_dpad.md`](2026-11-12_guide_sound_islands_and_kinematic_dpad.md) | Timeline `cond_audio` islands (`d19ba65c`, `e0a5bb35`) vs. 2D kinematic D-pad sprite trajectories and `0.33` selective release |

## Masked V2V Research (`docs/wiki/research/masking/`) & Postmortems (`postmortems/`)

| Document | Topic |
|---|---|
| [`masking/2026-10-07_mryolk.md`](masking/2026-10-07_mryolk.md) | SAM 3.1 in ComfyUI, from input to output |
| [`masking/2026-10-07_mryolk_phrases.md`](masking/2026-10-07_mryolk_phrases.md) | Phrases for SAM 3.1: what to give it and how specific to be |
| [`masking/2026-10-07_mryolk_stage_table.md`](masking/2026-10-07_mryolk_stage_table.md) | Stage table: is ComfyUI's SAM 3.1 computation identical to Meta's? |
| [`masking/2026-10-09_mrfetch.md`](masking/2026-10-09_mrfetch.md) | Three questions from the whole-frame lane answered from code |
| [`masking/2026-10-18_sam31_minimal_core_fixes.md`](masking/2026-10-18_sam31_minimal_core_fixes.md) | Minimal upstream fixes for SAM 3 / 3.1 in ComfyUI core |
| [`masking/2026-10-24_mrfabez_perception_models_response.md`](masking/2026-10-24_mrfabez_perception_models_response.md) | Perception-models review checked claim by claim against code |
| [`masking/2026-10-25_context_frames_and_identity_on_2050.md`](masking/2026-10-25_context_frames_and_identity_on_2050.md) | Why lead identity failed on `2050` and what window context showed the model |
| [`masking/2026-10-26_code_map_and_simplification_plan.md`](masking/2026-10-26_code_map_and_simplification_plan.md) | Masked V2V code map, traps, and phased simplification plan |
| [`postmortems/2026-10-04_feature_lotsofpeopledance-end-to-end.md`](postmortems/2026-10-04_feature_lotsofpeopledance-end-to-end.md) | Postmortem: `lotsofpeopledance` masked render end to end |
| [`postmortems/2026-10-10_span_fun-three-people.md`](postmortems/2026-10-10_span_fun-three-people.md) | Postmortem: three people replaced in one clip |
| [`postmortems/2026-10-11_span_fun-lid-marks-one-shot.md`](postmortems/2026-10-11_span_fun-lid-marks-one-shot.md) | Postmortem: one shot read as a result across six sessions |
| [`postmortems/2026-10-15_span_fun-kitchen-and-whole-shot.md`](postmortems/2026-10-15_span_fun-kitchen-and-whole-shot.md) | Postmortem: whole-shot vs. split-shot boundary seams |
| [`postmortems/2026-10-18_24fps_joined_cut.md`](postmortems/2026-10-18_24fps_joined_cut.md) | Why the joined cut was `23.976 fps` and why H3 requires exact `24.0 fps` |
| [`postmortems/2026-10-18_mom_kitchen_flicker.md`](postmortems/2026-10-18_mom_kitchen_flicker.md) | Why `second_hfp_0604h` fixed the second subject's kitchen flicker (`604–1009`) |

## Sparse Attention & VSA (`sparse/`, `vsa/`)

| Document | Topic |
|---|---|
| [`2026-11-13_e_means_clustering_assessment.md`](2026-11-13_e_means_clustering_assessment.md) | Evaluation of Meta FAIR `e-means` (`arXiv:2611.09501v1`) for H3/Sol-Attn and offline pipelines (**declined** for live attention; revisit only at `N >= 10^7`) |
| [`sparse/2026-10-02_sol_sparse_attention_hyperparameter_campaign.md`](sparse/2026-10-02_sol_sparse_attention_hyperparameter_campaign.md) | Sol-Attn sparse-attention hyperparameter campaign & empirical optimization report |
| [`sparse/sol_capture_and_optimization_guide.md`](sparse/sol_capture_and_optimization_guide.md) | End-to-end guide: Sol-Attn data capture & hyperparameter optimization |
| [`sparse/sol_dense_blocks_reanalysis.md`](sparse/sol_dense_blocks_reanalysis.md) | Sol-Attn dense-blocks reanalysis & 5-test experimental plan |
| [`sparse/sol_duckdb_dashboard.html`](sparse/sol_duckdb_dashboard.html) | Interactive Sol DuckDB metrics dashboard (HTML) |
| [`sparse/sol_tau_visual_explainer.html`](sparse/sol_tau_visual_explainer.html) | Visual explainer for Sol `tau` sparse threshold (HTML) |
| [`sparse/token_routing_comparison.html`](sparse/token_routing_comparison.html) | Interactive token routing comparison report (HTML) |
| [`vsa/fastvideo_vsa_checkpoint.md`](vsa/fastvideo_vsa_checkpoint.md) | FastVideo VSA-distilled H3 checkpoint analysis |
| [`vsa/vsa_node.md`](vsa/vsa_node.md) | Running VSA on H3: what the node does and what still blocks it |

## Parallel Decoding Distillation (`pdd/`)

| Document | Topic |
|---|---|
| [`pdd/README.md`](pdd/README.md) | PDD diagrams and handoffs index |
| [`pdd/2026-08-27_handoff.md`](pdd/2026-08-27_handoff.md) | Handoff, 2026-08-27 |
| [`pdd/2026-08-28_audio_plan.md`](pdd/2026-08-28_audio_plan.md) | Audio under PDD plan |
| [`pdd/2026-08-28_handoff.md`](pdd/2026-08-28_handoff.md) | Handoff, 2026-08-28 |
| [`pdd/2026-08-28_scene_complexity.md`](pdd/2026-08-28_scene_complexity.md) | What makes a PDD clip degrade |
| [`pdd/2026-08-29_market_scene_arms.md`](pdd/2026-08-29_market_scene_arms.md) | Market scene arms, 2026-08-29 |
| [`pdd/2026-08-30_handoff.md`](pdd/2026-08-30_handoff.md) | Handoff, 2026-08-30 |
| [`pdd/2026-08-31_handoff.md`](pdd/2026-08-31_handoff.md) | Cost of merging a PDD LoRA onto int8 |
| [`pdd/2026-09-05_bake_plan.md`](pdd/2026-09-05_bake_plan.md) | Backbone bake plan |
| [`pdd/2026-09-25_upstream_pdd_comparison.md`](pdd/2026-09-25_upstream_pdd_comparison.md) | Upstream PDD comparison |
| [`pdd/artifact_axis.md`](pdd/artifact_axis.md) | What actually makes a distilled clip fall apart |
| [`pdd/arxiv_2607.26004v1_Parallel_Decoding_Distillationfor_Fast_Image_and_Video_Generation.md`](pdd/arxiv_2607.26004v1_Parallel_Decoding_Distillationfor_Fast_Image_and_Video_Generation.md) | Paper reference notes (`arXiv:2607.26004v1`) |
| [`pdd/audio_under_pdd.md`](pdd/audio_under_pdd.md) | Sigma schedule & audio under PDD |
| [`pdd/depth_and_axes.md`](pdd/depth_and_axes.md) | Where in the DiT PDD matters |
| [`pdd/pdd_implementations.md`](pdd/pdd_implementations.md) | Five PDD implementations compared |
| [`pdd/queued_arms.md`](pdd/queued_arms.md) | Queued arms, 2026-08-27 |
| [`pdd/tier1_gate.md`](pdd/tier1_gate.md) | Tier 1 gate before observer renders |
| [`pdd/thirty_two_intervals.html`](pdd/thirty_two_intervals.html) | Standalone PDD diagram: 32 intervals |
| [`pdd/three_pdd_implementations.html`](pdd/three_pdd_implementations.html) | Standalone diagram: three PDD implementations |
| [`pdd/two_pdd_paths.html`](pdd/two_pdd_paths.html) | Standalone diagram: two PDD paths |

## Image Models (`docs/wiki/research/image/`)

For the active MiniMax-H3-Image and P-H3-Image setup, start at [`../image/h3_image.md`](../image/h3_image.md) and [`../image/p_h3_image.md`](../image/p_h3_image.md).

| Document | Topic |
|---|---|
| [`image/2026-11-14_h3_image_vae_inspection.md`](image/2026-11-14_h3_image_vae_inspection.md) | The MiniMax-H3-Image VAE: weights, config, Comfy-Org's fp16 file and core's decoder checked against diffusers, vllm-omni and sglang (nothing wrong found); tile stitching and geometry; release vs. P-H3 vs. HDR decoders on one latent; lettering breakup is in the latent |

## Core H3, Pika, Conditioning, & Upstream Stack Studies

| Document | Topic |
|---|---|
| [`2026-11-04_pika_h3_flex_weights_and_loras.md`](2026-11-04_pika_h3_flex_weights_and_loras.md) | Pika H3 Flex weights vs. MiniMax-H3 and LoRA compatibility on int8 |
| [`2026-11-04_pika_h3_flex_plan.md`](2026-11-04_pika_h3_flex_plan.md) | Pika H3 Flex integration plan, frame/ref ledger, and priority order |
| [`calibration_checkpoint_resume.md`](calibration_checkpoint_resume.md) | Checkpoint & resume for sequential AWQ calibration runs |
| [`comfyui_h3_t2va_trace.md`](comfyui_h3_t2va_trace.md) | ComfyUI's H3 `t2va` pipeline traced end to end |
| [`conditioning_cache.md`](conditioning_cache.md) | Persistent conditioning cache design note |
| [`conditioning_node_redesign.md`](conditioning_node_redesign.md) | Reference node redesign |
| [`conditioning_nodes.md`](conditioning_nodes.md) | Conditioning against the release: what was built vs. not built |
| [`h3_dit_implementations.md`](h3_dit_implementations.md) | Five implementations of the H3 DiT compared |
| [`h3_partition_distance.md`](h3_partition_distance.md) | Weight distance between `fl2va` and `ref2va` |
| [`m3_h3_bridge.md`](m3_h3_bridge.md) | MiniMax-M3 to H3 conditioning bridge feasibility study |
| [`marker_arm_binding.md`](marker_arm_binding.md) | Where each marker-corpus arm binds at render time |
| [`merge_requantisation.md`](merge_requantisation.md) | What happens when a LoRA is merged onto an int8 module |
| [`official_weights_metadata.md`](official_weights_metadata.md) | Official release metadata vs. ComfyUI weights |
| [`quant_levers.md`](quant_levers.md) | H3 quantization levers and constraints |
| [`sglang_comparison.md`](sglang_comparison.md) | sglang's H3 serving path vs. ComfyUI pack |
| [`sglang_h3_pipeline.md`](sglang_h3_pipeline.md) | sglang's MiniMax-H3 pipeline end to end |
| [`smoothquant_for_attention_qk.md`](smoothquant_for_attention_qk.md) | SmoothQuant applied to attention Q and K (`MiniMaxH3ChannelBalance`) |
| [`technique_transfer.md`](technique_transfer.md) | What transfers from LLM/ViT serving to H3 and what does not |
