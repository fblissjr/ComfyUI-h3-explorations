# Deliberately Absent Citations Ledger (`doc-link-absent`)

> **Sharded from [`docs/wiki/ops/checks.md`](../checks.md) on `2026-11-25`:** Active machine-read ledger parsed by [`bench/check_doc_links.py`](../../../../bench/check_doc_links.py). Full historical multi-paragraph `WHY:` notes live in [`doc_link_absent_history.md`](doc_link_absent_history.md).

```doc-link-absent
PATH: meta_sam3/EDITS.diff
WHY: the declared edits on the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model_builder.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/decoder.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/io_utils.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/maskformer_segmentation.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/model_misc.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/multiplex_mask_decoder.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/sam3_image.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/sam3_multiplex_base.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/sam3_multiplex_detector_utils.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/sam3_multiplex_tracking.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/sam3_tracker_utils.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/sam3_video_base.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/text_encoder_ve.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/tokenizer_ve.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/video_tracking_multiplex_demo.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/video_tracking_multiplex.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: meta_sam3/sam3/model/vl_combiner.py
WHY: part of the local copy of Meta's SAM 3.1 inference code

PATH: docs/research/qwen3-vl-special-tokens-post-training/canonical/owner_authored_marker_corpus.md
WHY: the brief the marker corpus was compiled to, in the research folder of...

PATH: docs/research/qwen3-vl-special-tokens-post-training/canonical/2026-08-24_layer50_processor_policy_benchmark.md
WHY: a note on the encoder's processor policies at one layer, in the...

PATH: docs/research/qwen3-vl-special-tokens-post-training/canonical/2026-08-24_serving_geometry_composes.md
WHY: a note on how the serving geometry composes, in the research folder of...

PATH: docs/research/qwen3-vl-special-tokens-post-training/canonical/2026-08-24_keyframe_vs_reference_positioning.md
WHY: a note on how keyframes and references are positioned, in the research...

PATH: docs/research/qwen3-vl-special-tokens-post-training/canonical/active_plan.md
WHY: the plan of the encoder calibration work, in the research folder of the...

PATH: docs/research/qwen3-vl-special-tokens-post-training/canonical/2026-08-24_still_policy_token_cost.md
WHY: a note on what each still policy costs in tokens, in the research...

PATH: bench/check_calibration_selector.py
WHY: a check of the calibration row selector's selection paths

PATH: bench/check_marker_corpus.py
WHY: a check of the compiled marker corpus against its brief

PATH: bench/check_pool_media_integrity.py
WHY: a check of the calibration pool's media verification

PATH: bench/check_native_h3_presentation.py
WHY: a check of a native-H3 calibration bundle against two reference arms

PATH: bench/check_released_encoder_is_stock.py
WHY: a check that compared the released text encoder's shards with stock...

PATH: bench/check_calibration_model_mapping.py
WHY: a check that the released checkpoint loads strictly into the...

PATH: bench/select_v2_calibration_rows.py
WHY: the tool that selected calibration rows from the pool

PATH: bench/build_h3_calibration_pool.py
WHY: the tool that built the calibration candidate pool

PATH: bench/review_pool_near_duplicates.py
WHY: the tool that reviewed the pool for near-duplicate media

PATH: bench/review_v2_calibration_bundle.py
WHY: the tool that reviewed a calibration bundle before a run

PATH: docs/research/masking/2026-10-04_mrblue.md
WHY: a session's research note on the masked lane...

PATH: docs/research/masking/2026-10-04_mrhf.md
WHY: a session's research note on the masked lane...

PATH: docs/research/masking/2026-10-04_mrpink.md
WHY: a session's research note on the masked lane...

PATH: docs/research/masking/2026-10-05_mryellow.md
WHY: a session's research note on the masked lane...

PATH: docs/research/masking/2026-10-06_mrsun.md
WHY: a session's research note on the masked lane...

PATH: vendor/sol_attn_minimax.py
WHY: upstream's Sol-Attn node, kept as a read-only reference since...

PATH: void_conditioning.py
WHY: the module of MiniMaxH3VoidConditioning

PATH: bench/convert_void_checkpoint.py
WHY: the VOID checkpoint converter

PATH: bench/check_void_conversion.py
WHY: the converter's check

PATH: bench/check_void_conditioning.py
WHY: the conditioning node's check

PATH: bench/prove_calibration_seam.py
WHY: needed llmcompressor, which the owner removed as a dependency on...

PATH: bench/pilot_sequential_feasibility.py
WHY: needed llmcompressor, which the owner removed as a dependency on...

PATH: bench/probe_calibration_input_seam.py
WHY: needed llmcompressor, which the owner removed as a dependency on...

PATH: bench/h3_awq_recipe.py
WHY: needed llmcompressor, which the owner removed as a dependency on...

PATH: bench/h3_gptq_recipe.py
WHY: needed llmcompressor, which the owner removed as a dependency on...

PATH: bench/check_calibration_checkpoint.py
WHY: needed llmcompressor, which the owner removed as a dependency on...

PATH: assert_chain.py
WHY: the module of SageChainAssert, retired 0.173.0 (owner, 2026-09-29)

PATH: reference_fit.py
WHY: the module of MiniMaxH3ReferenceFit, retired 0.173.0 (owner, 2026-09-29)

PATH: bench/analyze_morton.py
WHY: Morton block-geometry analysis, deleted 2026-09-27 with the Morton lane

PATH: sol_curves.py
WHY: the Hilbert/2d curve helpers behind the Sol node's morton_curve, deleted

PATH: _morton.py
WHY: the Triton pack's Wan Morton file, deleted with the pack on 2026-08-16

PATH: _morton_h3.py
WHY: same pack, same deletion.

PATH: 2026-08-23_caption_marker_arms.md
WHY: owner removed the whole internal/prompts/*.md set on 2026-09-01.

PATH: 2026-08-22_office_refs_fr_subs.md
WHY: same set, same removal, same reason.

PATH: h3_awq_encoder.py
WHY: the AWQ adapter, deleted 2026-09-13 with the closed encoder lane

PATH: h3_awq_encoder.md
WHY: the adapter's own document, deleted the same day for the same reason.

PATH: qwen3vl_32b_minimax_h3_w4a16_awq
WHY: the W4 artifact's config snapshot directory under config/, deleted with

PATH: check_h3_awq_encoder.py
WHY: the adapter's check, deleted with it.

PATH: build_native_h3_calibration_batch.py
WHY: the calibration-batch builder of the closed lane, deleted 2026-09-13.

PATH: capture_h3_encoder_states.py
WHY: the layer-50 capture tool for the W4 artifacts, deleted 2026-09-13.

PATH: measure_still_policy_token_cost.py
WHY: swept the AWQ snapshot's still bounds against core's

PATH: custom_nodes/ComfyUI-H3-Motion-Context-MultiRef/h3_song_audio_context.py
WHY: seitanism's sibling pack, read 2026-09-12 and no longer installed under

PATH: custom_nodes/ComfyUI-H3-Motion-Context-MultiRef/existing_video_extension.py
WHY: the same sibling pack's extension node, cited beside it in

PATH: pdd_lora.py
WHY: the module of MiniMaxH3PDDLoRA, retired 2026-11-06 with the PDD lane

PATH: pdd_math.py
WHY: the PDD 32-point grid and fusion helpers, retired 2026-11-06 with the PDD lane

PATH: pdd_observe.py
WHY: the PDD step observer, retired 2026-11-06 with the PDD lane

PATH: bench/convert_pdd_lora.py
WHY: the PDD LoRA sidecar converter, retired 2026-11-06 with the PDD lane

PATH: convert_pdd_lora.py
WHY: same converter, linked by relative path in docs/wiki/archive/h3_pdd.md and docs/research/pdd/

PATH: bench/check_pdd_sigmas.py
WHY: the PDD sigma check, retired 2026-11-06 with the PDD lane

PATH: bench/check_pdd_sidecar_contract.py
WHY: the PDD sidecar bake-contract check, retired 2026-11-06 with the PDD lane

PATH: check_pdd_sidecar_contract.py
WHY: same check, linked by relative path in docs/wiki/archive/h3_pdd.md

PATH: bench/compare_pdd_conversions.py
WHY: the PDD conversion comparison tool, retired 2026-11-06 with the PDD lane

PATH: bench/analyze_pdd_unmerge_curve.py
WHY: the PDD unmerged-block curve analysis, retired 2026-11-06 with the PDD lane

PATH: analyze_pdd_unmerge_curve.py
WHY: same tool, linked by relative path in docs/wiki/archive/h3_pdd.md and docs/research/pdd/

PATH: bench/bake_pdd_checkpoint.py
WHY: the PDD8 backbone checkpoint baker, retired 2026-11-06 with the PDD lane

PATH: prompt_bank/control_vectors/film_16mm_1974_bare.toml
WHY: the early E2.3 bare-frame twin of film_16mm_1974.toml, retired 2026-11-06

PATH: film_16mm_1974_bare.toml
WHY: same file, linked by relative path in historical research docs

PATH: prompt_bank/control_vectors/lettering_bare.toml
WHY: the early E2.3 bare-frame twin of lettering.toml, retired 2026-11-06

PATH: lettering_bare.toml
WHY: same file, linked by relative path in historical research docs

PATH: sol_chunked_h3.py
WHY: MiniMaxH3SolChunked module retired 2026-11-13 to archive/sol_chunked_h3.py

PATH: bench/check_sol_chunked.py
WHY: check retired 2026-11-13 to archive/bench/check_sol_chunked.py

PATH: check_sol_chunked.py
WHY: same file, linked by relative filename in historical docs

PATH: overlay_loader.py
WHY: MiniMaxH3OverlayLoader retired 2026-11-25 to archive/overlay_loader.py

PATH: denoise_mask_probe.py
WHY: MiniMaxH3DenoiseMaskProbe retired 2026-11-25 to archive/denoise_mask_probe.py

PATH: bench/check_denoise_mask_probe.py
WHY: check retired 2026-11-25 to archive/bench/check_denoise_mask_probe.py

```
