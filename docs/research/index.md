# Research: dated studies and postmortems

What was true when each was written. The current answer is on the topic page named in each section.

## The model and its implementations (current pages: `docs/h3/`)

| record | what it is |
|---|---|
| [`2026-08-29_comfyui_h3_t2va_trace.md`](2026-08-29_comfyui_h3_t2va_trace.md) | what ComfyUI's own code does, call by call, for one t2va render |
| [`2026-08-29_h3_dit_implementations.md`](2026-08-29_h3_dit_implementations.md) | the DiT across every implementation available here; diffusers is the reference of record |
| [`2026-08-25_sglang_h3_pipeline.md`](2026-08-25_sglang_h3_pipeline.md) | sglang's H3 pipeline stage by stage, at source level; compares nothing |
| [`2026-08-21_official_weights_metadata.md`](2026-08-21_official_weights_metadata.md) | what the release declares against what ComfyUI assumes. Read before using a marker |
| [`2026-08-29_h3_partition_distance.md`](2026-08-29_h3_partition_distance.md) | how far apart fl2va and ref2va are, per component |
| [`2026-08-21_conditioning_nodes.md`](2026-08-21_conditioning_nodes.md) | what was built against the two conditioning defects, and the contracts in `MiniMaxH3ReferenceToVideo` that live only in comments |
| [`2026-08-24_conditioning_node_redesign.md`](2026-08-24_conditioning_node_redesign.md) | the reference node redesign: fold the fit, move the ceiling to the conditioner |
| [`2026-08-25_conditioning_cache.md`](2026-08-25_conditioning_cache.md) | a persistent conditioning cache: design note |
| [`2026-09-15_m3_h3_bridge.md`](2026-09-15_m3_h3_bridge.md) | the MiniMax-M3 to H3 conditioning bridge, and the case against it |
| [`2026-08-25_marker_arm_binding.md`](2026-08-25_marker_arm_binding.md) | where each marker-corpus arm binds at render time |
| [`2026-09-19_in_context_reference_and_sinks.md`](2026-09-19_in_context_reference_and_sinks.md) | in-context references, the prefix and attention sinks; the every-fifth-latent-frame finding starts here |
| [`2026-09-25_continuation_guide_rows.md`](2026-09-25_continuation_guide_rows.md) | continuing a clip by guide rows: six designs and one to build |
| [`2026-09-25_temporal_offset_and_adaln_rounding.md`](2026-09-25_temporal_offset_and_adaln_rounding.md) | a global temporal offset, and AdaLN rounding on the INT8 DiT |
| [`2026-09-25_step_caching_survey.md`](2026-09-25_step_caching_survey.md) | step caching for H3, re-examined |

## Attention and quantization (current pages: `docs/attention/`)

| record | what it is |
|---|---|
| [`2026-09-27_sol_node_redesign.md`](2026-09-27_sol_node_redesign.md) | the Sol node redesign the owner approved: what goes, what merges, the Morton plan, the tests in order |
| [`2026-09-01_session_sol-route-observability_explainer.html`](2026-09-01_session_sol-route-observability_explainer.html) | Sol routing, observed: an explainer page |
| [`2026-08-14_sol_bench_plan.md`](2026-08-14_sol_bench_plan.md) | Sol-Attn runs pre-registered with predictions from 2026-08-14, and the outcomes that scored them |
| [`2026-09-03_sol_exact_pquant_and_base_capture.md`](2026-09-03_sol_exact_pquant_and_base_capture.md) | Sol exact-P quantization and the Base16 capture |
| [`2026-09-04_sol_token_aug_grade.md`](2026-09-04_sol_token_aug_grade.md) | token routing (`token_aug`) graded on the Base16 capture |
| [`2026-09-05_token_aug_plan.md`](2026-09-05_token_aug_plan.md) | token routing from graded lever to a per-block policy: the plan |
| [`2026-09-17_rotation_and_lowbit_attention_survey.md`](2026-09-17_rotation_and_lowbit_attention_survey.md) | rotation and low-bit attention on Ada sm89: a survey |
| [`2026-09-19_dsv41_hierarchical_indexer.md`](2026-09-19_dsv41_hierarchical_indexer.md) | DeepSeek V4.1's hierarchical sparse indexer, read for Sol routing |
| [`2026-09-19_token_selection_scorers.md`](2026-09-19_token_selection_scorers.md) | cheap scorers that choose what attention computes exactly, finer than a block |
| [`2026-09-19_video_sparse_attention_2026.md`](2026-09-19_video_sparse_attention_2026.md) | four training-free video sparse-attention methods of 2026, with a frozen-route test stated in advance |
| [`2026-09-19_where_approximation_is_tolerated.md`](2026-09-19_where_approximation_is_tolerated.md) | which layers, heads and steps tolerate approximate attention |
| [`2026-08-31_quant_levers.md`](2026-08-31_quant_levers.md) | what can be changed about H3's quantization and which levers are closed |
| [`2026-09-15_smoothquant_for_attention_qk.md`](2026-09-15_smoothquant_for_attention_qk.md) | SmoothQuant's channel balance, pointed at attention's q and k |
| [`2026-08-25_calibration_checkpoint_resume.md`](2026-08-25_calibration_checkpoint_resume.md) | checkpoint and resume for a sequential AWQ calibration run |
| [`sparse/2026-10-02_sol_dense_blocks_reanalysis.md`](sparse/2026-10-02_sol_dense_blocks_reanalysis.md) | Sol-Attn dense blocks re-analyzed under rotated INT8, the middle-cluster error spike, and the 5-test matrix |
| [`sparse/2026-10-02_sol_sparse_attention_hyperparameter_campaign.md`](sparse/2026-10-02_sol_sparse_attention_hyperparameter_campaign.md) | the Sol-Attn sparse hyperparameter campaign of 2026-10-02 |
| [`sparse/2026-10-02_sol_capture_and_optimization_guide.md`](sparse/2026-10-02_sol_capture_and_optimization_guide.md) | Sol-Attn data capture and hyperparameter optimization, end to end; with three explainer pages beside it in `sparse/` |
| [`vsa/`](vsa/2026-08-30_vsa_node.md) | running VSA on H3, and the FastVideo VSA-distilled checkpoint |

## Distills (current pages: `docs/distills/`)

| record | what it is |
|---|---|
| [`pdd/`](pdd/README.md) | the PDD lane's handoffs, plans and diagrams; its README orders them, newest first |
| [`2026-09-26_flashgen.md`](2026-09-26_flashgen.md) | FlashGen: where each setting comes from, how upstream runs it, why a LoRA merged into int8 loses most of a distill |
| [`2026-09-26_distill_routing.md`](2026-09-26_distill_routing.md) | routing distills within a clip: six routes, three of them H3's own |
| [`2026-08-31_merge_requantisation.md`](2026-08-31_merge_requantisation.md) | what happens to a LoRA merged onto an int8 module; stored weights only |

## Evaluation and prompts (current pages: `docs/method/`, `docs/prompting/`)

| record | what it is |
|---|---|
| [`2026-08-16_prompt_length_experiment.md`](2026-08-16_prompt_length_experiment.md) | a paired render on prompt length, pre-registered before either clip was watched, and its verdict |
| [`2026-08-28_scene_arm_renders.md`](2026-08-28_scene_arm_renders.md) | the two scene arms of 2026-08-28: what to look for and what those renders could not tell |
| [`2026-09-19_evaluation_one_judge.md`](2026-09-19_evaluation_one_judge.md) | evaluation for one judge and one GPU: decoys, a low anchor, stop rules, small-panel arithmetic. Read before building a blind panel |

## The docs themselves (current pages: `docs/index.md`)

| record | what it is |
|---|---|
| [`2026-10-19_docs_consolidation.md`](2026-10-19_docs_consolidation.md) | what the 2026-10-19 rewrite of every topic folder cut, and where each piece went; the text before it is in git at `901bee1` |

## Masked video to video (current pages: `docs/masked_v2v/`)

| record | what it is |
|---|---|
| [`masking/2026-10-07_mryolk.md`](masking/2026-10-07_mryolk.md) | SAM 3.1 in ComfyUI, from input to output |
| [`masking/2026-10-07_mryolk_phrases.md`](masking/2026-10-07_mryolk_phrases.md) | phrases for SAM 3.1: what to give it and how specific to be |
| [`masking/2026-10-07_mryolk_stage_table.md`](masking/2026-10-07_mryolk_stage_table.md) | is ComfyUI's SAM 3.1 the same computation as Meta's, stage by stage |
| [`masking/2026-10-09_mrfetch.md`](masking/2026-10-09_mrfetch.md) | three questions from the whole-frame lane, answered from code |
| [`masking/2026-10-18_sam31_minimal_core_fixes.md`](masking/2026-10-18_sam31_minimal_core_fixes.md) | the minimal upstream fixes for SAM 3 and 3.1 in ComfyUI core |
| [`postmortems/2026-10-04_feature_lotsofpeopledance-end-to-end.md`](postmortems/2026-10-04_feature_lotsofpeopledance-end-to-end.md) | the lotsofpeopledance masked render, end to end |
| [`postmortems/2026-10-10_span_fun-three-people.md`](postmortems/2026-10-10_span_fun-three-people.md) | three people replaced in one clip, eight sessions |
| [`postmortems/2026-10-11_span_fun-lid-marks-one-shot.md`](postmortems/2026-10-11_span_fun-lid-marks-one-shot.md) | one shot read as a result, six sessions |
| [`postmortems/2026-10-15_span_fun-kitchen-and-whole-shot.md`](postmortems/2026-10-15_span_fun-kitchen-and-whole-shot.md) | the kitchen came apart, and the whole shot came back |
| [`postmortems/2026-10-18_24fps_joined_cut.md`](postmortems/2026-10-18_24fps_joined_cut.md) | the joined cut was 23.976 fps, and H3 needs 24 fps end to end |
| [`postmortems/2026-10-18_mom_kitchen_flicker.md`](postmortems/2026-10-18_mom_kitchen_flicker.md) | why one render fixed the second subject's kitchen flicker |
