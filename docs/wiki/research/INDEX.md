# Research & Postmortems Router (`docs/wiki/research/`)

Router index for dated studies, experiment logs, and postmortems under `docs/wiki/research/`. These records state what was measured on their date; for active rules, workflows, and current project ownership, start at [`../index.md`](../index.md), [`../projects/index.md`](../projects/index.md), and [`../ops/evidence.md`](../ops/evidence.md). Retired notes live in [`../archive/INDEX.md`](../archive/INDEX.md).

## Cross-Pillar API Workflows & Perception Decoupling (`2026-11-14`)

| Document | Topic |
|---|---|
| [`2026-11-14_cross_pillar_api_workflows_and_sam31_decoupling.md`](2026-11-14_cross_pillar_api_workflows_and_sam31_decoupling.md) | Six-pillar API composition synthesis (`P-H3-Image` + Phase 1 `guide_end_percent` + `EditGuider` + `SoundCues` + `SpriteTrajectory` + `AudioRefine` + decoupled `SAM 3.1` / `Sapiens2`), Phase 2B `H3MaskedSource -> AudioFreezeSong` boundary audit, two-phase zero-swap execution, and 3-node extraction plan off [`audio_freeze_song.py`](../../../audio_freeze_song.py) |
| [`2026-11-16_geiru_keystone_audit_and_roadmap.md`](2026-11-16_geiru_keystone_audit_and_roadmap.md) | Clean-room technical audit of `geiru_context_suite` / `NerdNotes`, `ComfyUI-MMH3Tools`, and `ComfyUI-MiniMaxH3-Context-Loop` (`4,629f` `30`-clip analysis, RTX 4090 `return_lse` `88.2 ms` vs. `519.0 ms` benchmark, `c % 5 == 2` VAE seam-lock proof, and `5`-bucket holistic roadmap) |

## Control Vectors, Readouts, Sound Cues & 2D D-Pad (`docs/wiki/steering/research/`)

For active nodes and recipes, start at [`../steering.md`](../steering.md) and [`../edit_guidance.md`](../edit_guidance.md).

| Document | Topic |
|---|---|
| [`../steering/research/2026-11-04_h3_control_vectors.md`](../steering/research/2026-11-04_h3_control_vectors.md) | Control vectors for H3 (contrastive mean-difference directions in refined text conditioning): hypothesis, gate, build plan, and code facts |
| [`../steering/research/2026-11-05_control_vector_system_prompt.md`](../steering/research/2026-11-05_control_vector_system_prompt.md) | Self-contained system prompt for generating multimodal video+audio scene prompts and contrastive `.toml` pairs |
| [`../steering/research/2026-11-06_detangle_comparison_and_borrowable_concepts.md`](../steering/research/2026-11-06_detangle_comparison_and_borrowable_concepts.md) | `DeTangle` compared against H3 control vectors: what transfers (soft ridge holds, paired subtraction, comparison UI) vs. what does not |
| [`../steering/research/2026-11-06_mainodes_comparison_and_pika_roadmap.md`](../steering/research/2026-11-06_mainodes_comparison_and_pika_roadmap.md) | `ComfyUI-MAINodes` compared against H3 control vectors and the 6-item Pika H3 Flex steering roadmap |
| [`../steering/research/2026-11-06_j_lens_and_refiner_jacobian.md`](../steering/research/2026-11-06_j_lens_and_refiner_jacobian.md) | Adapting Anthropic's Jacobian Lens (`J-Lens`) to H3 post-refiner control vectors ($J_{\text{Wiener}}$, token-0 sink, spectral-filtered $\Sigma_{49,\text{filt}}^{0.30}$) |
| [`../steering/research/2026-11-06_fresh_eye_audit_and_holistic_resolution.md`](../steering/research/2026-11-06_fresh_eye_audit_and_holistic_resolution.md) | Fresh-eye audit of control vectors (`2026-11-05`–`11-06`) across hold norms, encoder holds, and runtime nodes |
| [`../steering/research/2026-11-06_holistic_resolution_proposal.md`](../steering/research/2026-11-06_holistic_resolution_proposal.md) | Holistic resolution proposal for control vectors, `J-Lens`, and worktree consolidation |
| [`../steering/research/2026-11-06_holistic_stack_review_and_invariants.md`](../steering/research/2026-11-06_holistic_stack_review_and_invariants.md) | Disk-verified review of the control-vector stack and 5 system-wide invariants |
| [`../steering/research/2026-11-09_control_vector_simplification_plan.md`](../steering/research/2026-11-09_control_vector_simplification_plan.md) | Control-vector simplification plan, open-problem inventory, and `MiniMaxH3EditGuider` design/gates |
| [`../steering/research/2026-11-10_control_vector_state_history_and_tradeoffs.md`](../steering/research/2026-11-10_control_vector_state_history_and_tradeoffs.md) | State history (`11-04`..`11-10`), over-correction audit, half-baked seams, and prioritized recommendations |
| [`../steering/research/2026-11-12_chain_lens_digest_and_experiment_plan.md`](../steering/research/2026-11-12_chain_lens_digest_and_experiment_plan.md) | Chain Lens & WorkspaceBench digest, 6 readout/steering experiments, and updated `2026-11-12` execution order (§15) |
| [`../steering/research/2026-11-12_chain_lens_refs/jlens_workspace_paper.md`](../steering/research/2026-11-12_chain_lens_refs/jlens_workspace_paper.md) | Reading notes: J-lens / global workspace paper |
| [`../steering/research/2026-11-12_chain_lens_refs/lens_variants.md`](../steering/research/2026-11-12_chain_lens_refs/lens_variants.md) | Reading notes: J-lens variants and extensions |
| [`../steering/research/2026-11-12_chain_lens_refs/mtp_and_decoding.md`](../steering/research/2026-11-12_chain_lens_refs/mtp_and_decoding.md) | Reading notes: MTP heads, contrastive decoding, and substitutes for a missing MTP head |
| [`../steering/research/2026-11-12_chain_lens_refs/workspacebench_and_verbalizers.md`](../steering/research/2026-11-12_chain_lens_refs/workspacebench_and_verbalizers.md) | Reading notes: WorkspaceBench internals, BPE bug, and trained verbalizers |
| [`../steering/research/2026-11-12_new_vectors_three_ways.md`](../steering/research/2026-11-12_new_vectors_three_ways.md) | Eight fresh single-phrase vectors vs. prompt rewrite vs. `MiniMaxH3EditGuider`, audio cross-check, and batch-share findings |
| [`../steering/research/2026-11-12_exp5_ladder_and_temporal_guide_synthesis.md`](../steering/research/2026-11-12_exp5_ladder_and_temporal_guide_synthesis.md) | Forensic audit of `exp2`–`exp4`, `exp5` 1:1 ladder (`heavy_blizzard` & `watercolor_wash`), temporal/audio guide mechanics, and next priorities |
| [`../steering/research/2026-11-12_sound_cues_dpad_and_033_release_law.md`](../steering/research/2026-11-12_sound_cues_dpad_and_033_release_law.md) | Guide, failure modes, and RoPE/flow math for `MiniMaxH3SoundCues`, `MiniMaxH3AppendSpriteTrajectory`, and the `0.33` (`b = 0.50`) release rule |
| [`../steering/research/2026-11-20_unpooled_multivector_sliders.md`](../steering/research/2026-11-20_unpooled_multivector_sliders.md) | Verified **Clamped Pole-A Subspace Erasure** (`add_norm_matched_erasure`, `193.5 KB` `float32`, row-softmax `tau=0.15` `mu_d` + rank-8 `V_r` / rank-16 composed), `5/5` multi-seed, `192f` AV (`fl2va` + `ref2va` `<Picture 1>`), `P-H3-Image` `4.1 MP`, and `2026-11-21` showcases |
| [`../steering/research/2026-11-20_multivector_retrieval_sota_to_dit_steering.md`](../steering/research/2026-11-20_multivector_retrieval_sota_to_dit_steering.md) | Oct–Nov 2026 multi-vector retrieval SOTA survey (**Minish Lab** `reach`/`Model2Vec`/`POTION`, **LightOn** `PyLate`/`LateOn` + **Clavié** Ward pooling, **Khattab** `ColBERTv2`/`PLAID`/`WARP_SELECT`, **Mixedbread** *Latent Terms* + `mxbai-edge-colbert-v0`, **Google** `MUVERA`, Balanced Sinkhorn OT vs. row-softmax `tau=0.15`) -> H3 `5376`-D steering + [`bench/verify_pole_a_erasure.py`](../../../bench/verify_pole_a_erasure.py) |
| [`../steering/research/2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md`](../steering/research/2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md) | Standalone research monograph (v4): *Breaking the Frozen-Anchor Steering Paradox: From Late-Interaction Multi-Vector Retrieval to Parametric Subspace EQ in Diffusion and Input-Injected Iterative Transformers* (Autoregressive Self-Steering vs. Prelude Input Injection across DiTs, Looped-DiTs, Masked DLMs & Looped LLMs; 4 impossibility proofs; Exact SVD Ridge Split & Three-Subspace EQ Theorems 1–2 + Algorithms 1–2; $2\times 2$ synergy factorial + 6-arm specificity; `38`-bank census; `50`-block DiT trace + causal `blocks.20..49` Shadow-Text K/V proof; Dual-Lock `fl2va` law; `YIN` $F_0$ probes + 6 `.svg` figures; [`bench/results/2026-11-21_multivector_erasure_benchmark.md`](../../../bench/results/2026-11-21_multivector_erasure_benchmark.md), [`bench/results/2026-11-21_phase3_de_slop_and_vision_block_steering.md`](../../../bench/results/2026-11-21_phase3_de_slop_and_vision_block_steering.md)) |
| [`../steering/research/2026-11-21_reproducing_late_interaction_to_dit_steering.md`](../steering/research/2026-11-21_reproducing_late_interaction_to_dit_steering.md) | Zero-data reproduction runbook & section-to-code/workflow/wiki reference map for the `2026-11-21` monograph (Tiers 1–5 across `verify_pole_a_erasure.py`, `check_control_vectors.py`, `eval_multivector_erasure.py`, `run_phase3_steering_experiments.py`, and `probe_yin_f0_drift.py`) |
| [`../steering/research/2026-11-22_paper_to_codebase_promotion_spec.md`](../steering/research/2026-11-22_paper_to_codebase_promotion_spec.md) | **9-Item Paper-to-Codebase Promotion Spec & Gap-Closure Ledger (`2026-11-22`)**: code-anchored specs & lane ownership across Specs A–E (`blocks.20..49` production socket, `1x` `Q_aud` split gate, clamped union stacking `e_budget` theorem, Dual-Lock `fl2va` unlock, local `Qwen-Image-2.1` validation), plus CPU `YIN` $F_0$ Phase 3 table, `app/cv_lab.py` `-2PC` Vibe Search falsification (`88.9%` vs. `77.8%`), and exact `32 + 4 + 6 = 42` missing `h3_image` census |
| [`../steering/research/2026-11-22_unified_steer_attention_hook_spec.md`](../steering/research/2026-11-22_unified_steer_attention_hook_spec.md) | **Compiler-Safe Unified Attention-Hook Architecture Spec (`MiniMaxH3Steer`, `2026-11-22`)**: eliminates Phase 3/4 prototype patchwork (`disable_comfy_compiler = True`, shared `register_forward_pre_hook` mutation, full `k`/`v` concatenation) via call-local `H3SteerStepWorkspace` + `H3SteerBlockPatch` (`patches_replace["dit"]["double_block"]`), zero-clone in-place `[tx_a:tx_b]` save/restore + fused `cat([q_shadow, q_aud])`, dual-path Sol + standalone Sage + `ExactBlocks` coverage, Mode 0–3 state machine (fixing combined band + `Q_aud` block-20 fork), and `--assert-graph-breaks` verification gate |
| [`../steering/research/2026-11-22_track1_track2_multivector_erasure_blueprint.md`](../steering/research/2026-11-22_track1_track2_multivector_erasure_blueprint.md) | **Track 1 vs. Track 2 Multi-Vector & Subspace Erasure Blueprint (`2026-11-22`)**: `52`-vector (`2,652` pair) cross-interference census (up to `42.4%` raw cross-hit + `15.9%` mean self-overlap trap) proving per-slider RASteer `ROS` (`V_r^(j) ⊥ mu_d^(-j)`), `7`-bank LOPO CV audit comparing Track 1 `mu_d` against prompt-centered Ward/Ridge `WARP_SELECT` & Theorem-1 `W_B` (`-14.34%` `look_direct_1990s_flash`, `-5.03%` `noir`, `-3.79%` `molten_lava_golem`), `N >= 6` `ColQwen` `vision_tokens` banks with RASteer face-retain protection, MUTE/RASteer/Looped-DiT routing, and Track 1 vs. Track 2 showcase matrix |
| [`../steering/research/2026-11-24_outside_review_erasure_novelty_evidence_and_commits.md`](../steering/research/2026-11-24_outside_review_erasure_novelty_evidence_and_commits.md) | **Outside review (`2026-11-24`)**: whether Pole-A erasure and its math are novel (prior-art table), what the gate records do and do not show claim by claim, what "multivector" means at each place it is used, an overhyped / underhyped / overlooked ledger, and a read of the `2026-11-20`..`2026-11-23` commits. Carries pointers to records, not values. |
| [`../steering/research/exl3_qwen3vl/runbook.md`](../steering/research/exl3_qwen3vl/runbook.md) | EXL3 quantization (`5.37` / `4.65 bpw`) runbook for `Qwen3-VL-32B-Instruct` control-vector labelling and `layers.49` steering |
| [`../steering/research/exl3_qwen3vl/h3_control_vector_report.md`](../steering/research/exl3_qwen3vl/h3_control_vector_report.md) | H3 control-vector batch-labelling & `layers.49` readout report |
| [`../steering/research/exl3_qwen3vl/control_vector_viewer.html`](../steering/research/exl3_qwen3vl/control_vector_viewer.html) | Interactive standalone HTML viewer for EXL3 `Qwen3-VL` control-vector labels and activations |

## Timeline Signal & Guide Studies (`2026-11-12`–`2026-11-15`) & Eval Flywheel

| Document | Topic |
|---|---|
| [`2026-11-10_the_eval_flywheel.md`](2026-11-10_the_eval_flywheel.md) | How a month of masked-V2V visual inspection turned into automated rules and checks, and which survived |
| [`2026-11-12_guide_signal_full_length_bars.md`](2026-11-12_guide_signal_full_length_bars.md) | Full-length bar signal given as a guide video on text-to-video: visual bleed vs. audio-window locking across five arms |
| [`2026-11-12_guide_sound_islands_and_kinematic_dpad.md`](2026-11-12_guide_sound_islands_and_kinematic_dpad.md) | Timeline `cond_audio` islands (`d19ba65c`, `e0a5bb35`) vs. 2D kinematic D-pad sprite trajectories and `0.33` selective release |
| [`2026-11-15_spatial_world_director_and_3d_edit_guidance.md`](2026-11-15_spatial_world_director_and_3d_edit_guidance.md) | Native ComfyUI core (`BasicGuider` / `CFGGuider`, `SetLatentNoiseMask`, `MiniMaxH3AddGuide`) vs. 3D `[F, H, W]` velocity steering, `40 Hz` temporal audio gating, `MiniMaxH3SpatialDirector`, `MiniMaxH3DirectorHUD`, and measured `141`-frame showcase benchmarks |

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
| [`image/2026-11-14_h3_image_generate_big_deliver_small.md`](image/2026-11-14_h3_image_generate_big_deliver_small.md) | Generate at 16 MP and shrink, against generate at the delivered size: 14 cases, phone and desktop, content slips, time and file size; the decision to deliver 4 MP from a 16 MP canvas |
| [`image/2026-11-14_h3_image_vae_inspection.md`](image/2026-11-14_h3_image_vae_inspection.md) | The MiniMax-H3-Image VAE: weights, config, Comfy-Org's fp16 file and core's decoder checked against diffusers, vllm-omni and sglang (nothing wrong found); tile stitching and geometry; release vs. P-H3 vs. HDR vs. the H3 video decoder and its `t1` fine-tune on encoded pictures and on real renders; no decode node beats what ships |
| [`image/2026-11-14_embedded_image_in_video_plan.md`](image/2026-11-14_embedded_image_in_video_plan.md) | Phases 0–4 build & verification plan for embedding H3-family image generation and editing (`Pika` `t = 1`, `P-H3-Image`, `MiniMax-H3-Image`) inside H3 video workflows: dual-VAE hygiene, continuous `MakeMediaControlVector` dialing, Phase 1 `guide_end_percent` mid-clip release guards, scout-edit-fork & `MiniMaxH3MaskedSource` diff masking, anti-burnout loops, and explicit non-starters |

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
| [`2026-11-15_upstream_synthesis.md`](2026-11-15_upstream_synthesis.md) | Everything the 2026-11-15 upstream sweep found (sglang, vllm-omni, diffusers, core, periphery), grouped by area with what each could do for the pack and a depth tag per item (hand-checked, agent-read code, body only, title only); standing pages in [`../upstream/`](../upstream/index.md) |
