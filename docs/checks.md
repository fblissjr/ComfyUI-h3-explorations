# What this repo checks, and why

Index of every check in `bench/`. One row per script: what it defends, what it
needs to run, and whether it has earned trust.

There is no test suite. Each script is standalone, prints its own `ok` /
`FAIL` lines, and returns a non-zero exit code on failure.
`bench/run_checks.py` runs them all with the card masked and compares what is
red with `bench/checks_baseline.json` ("Running them").

**Per-check narrative and the run logs live in
[`docs/check_postmortems.md`](check_postmortems.md)**, not here. This file is
the index and the standard. It carries no run history and no counts in prose:
the table below is the authority for what exists, and a count typed into a
sentence here has been wrong every time it was tried.

**Two entries are not a `check_*.py`**: `bench/preflight_graph.py` and
`bench/build_wiki_index.py`, listed last. The first grades and prices rather
than gating, and a person runs it before a render rather than the suite running
it. The second reports the documents no link reaches.
Listing both is deliberate -- each defends real ground -- and it is why
"every check" and "every row" are not the same set.

## The standard

A check reads the thing under test, not its own copy of the answer (see
"Drive the expression, never restate it" below), and compares it with an
independent answer where one exists: a vendor file, a committed baseline, a
second implementation. One that reports red while the state is correct is
worse than no check. Where the expected value is itself the measurement, as
with `check_correctness.py`'s error against the stock forward, the check
records it rather than inventing a threshold.

Until 2026-09-11 this section also required every check over mechanically
specified behaviour to be shown red before it was trusted (break the thing it
guards, watch it fail, put it back), with rules for building those mutation
controls. The owner retired that requirement, and nothing here asks for it
now. The index's `shown red` column went with it on 2026-09-11; git history has
what each cell recorded.


For `break_checks.py` history, metric-ranking, expression-driving, and queue-execution case studies, as well as retired checks, uncontrolled requirements, and gaps, see [`docs/checks/standard_and_gaps.md`](checks/standard_and_gaps.md).

## Running them

Most need neither CUDA nor a model and finish in about a second.

```bash
# from the repo root, with the ComfyUI venv python
python bench/check_keyframe_canvas.py
```

**Before a commit that changes a node schema, a default or a loader, run the
sweep:**

```bash
<comfy venv python> bench/run_checks.py            # every check_*.py, masked
<comfy venv python> bench/run_checks.py --only sol --logs <dir>
<comfy venv python> bench/run_checks.py --changed  # what the changed files can turn red
```


For `--changed`, `CUDA_VISIBLE_DEVICES=` masking rules, and `bench/_lib::bootstrap` details, see [`docs/checks/standard_and_gaps.md`](checks/standard_and_gaps.md#running-sweep--masking-details).

## The index

Full per-check history, 2026-08-23 amendments, and `"claims block"` notes are sharded under `docs/checks/` (each `< 35 KB`; see [`docs/checks/index_15.md`](checks/index_15.md) for the closing notes):

| Shard | Checks covered |
|---|---|
| [`docs/checks/standard_and_gaps.md`](checks/standard_and_gaps.md) | Standard case studies, sweep/masking notes, retired checks, uncontrolled requirements & gaps |
| [`docs/checks/doc_link_absent_history.md`](checks/doc_link_absent_history.md) | Full historical `WHY:` entries for deliberately absent citations |
| [`docs/checks/index_01.md`](checks/index_01.md) | `check_correctness.py` .. `check_pdd_sidecar_contract.py` (25 rows) |
| [`docs/checks/index_02.md`](checks/index_02.md) | `check_graph_values.py` .. `check_lora_alpha.py` (24 rows) |
| [`docs/checks/index_03.md`](checks/index_03.md) | `check_marker_arms.py` .. `check_shot_table.py` (13 rows) |
| [`docs/checks/index_04.md`](checks/index_04.md) | `check_sam31_corrections.py` .. `check_subject_parts.py` (9 rows) |
| [`docs/checks/index_05.md`](checks/index_05.md) | `check_capture_masked_run.py` .. `check_draw_lid_marks.py` (5 rows) |
| [`docs/checks/index_06.md`](checks/index_06.md) | `check_draw_face_points.py` .. `check_render_ledger.py` (9 rows) |
| [`docs/checks/index_07.md`](checks/index_07.md) | `check_render_history.py` .. `check_smoke_h3.py` (6 rows) |
| [`docs/checks/index_08.md`](checks/index_08.md) | `check_load_chain.py` .. `check_shot_to_load.py` (3 rows) |
| [`docs/checks/index_09.md`](checks/index_09.md) | `check_load_text.py` .. `check_shot_coverage.py` (13 rows) |
| [`docs/checks/index_10.md`](checks/index_10.md) | `check_shot_parts.py` .. `check_review_pages.py` (10 rows) |
| [`docs/checks/index_11.md`](checks/index_11.md) | `check_covered_items.py` .. `check_clip_signals.py` (11 rows) |
| [`docs/checks/index_12.md`](checks/index_12.md) | `check_assemble_delivery.py` .. `check_sweep_loads.py` (8 rows) |
| [`docs/checks/index_13.md`](checks/index_13.md) | `check_plate_restore.py` .. `check_dit_prefix_attention.py` (26 rows) |
| [`docs/checks/index_14.md`](checks/index_14.md) | `measure_dit_prefix_attention.py` .. `build_prompt_bank.py` (23 rows) |
| [`docs/checks/index_15.md`](checks/index_15.md) | `masked_code_map.py` .. `smoke_h3.py` (4 rows) |

| check | defends | needs | claims block |
|---|---|---|---|
| `check_correctness.py` | the patched H3 forward against the... ([detail](checks/index_01.md)) | CUDA | no |
| `check_clone_v_wiring.py` | clone_v reaches the forward, and... ([detail](checks/index_01.md)) | CUDA | no |
| `check_override_routing.py` | which calls the attention override... ([detail](checks/index_01.md)) | - | yes |
| `check_lowvram_handoff.py` | more than its name says. ([detail](checks/index_01.md)) | - | yes |
| `check_vendor_config.py` | that vendor_config/ still is what... ([detail](checks/index_01.md)) | vendor_config/... | no |
| `check_schema_defaults.py` | every node's io.Schema defaults... ([detail](checks/index_01.md)) | PYTHONPATH... | yes |
| `check_ref_prompt_labels.py` | every ref graph's prompt names... ([detail](checks/index_01.md)) | - | yes |
| `check_prompt_guide_conformance.py` | every shipped prompt against the... ([detail](checks/index_01.md)) | the guides in... | yes |
| `check_distill_settings.py` | every shipped graph runs its... ([detail](checks/index_01.md)) | - | yes |
| `check_distill_grid.py` | that a distilled graph samples on... ([detail](checks/index_01.md)) | no CUDA | yes |
| `check_solattn_correctness.py` | Sol-Attn's CUDA kernel against the... ([detail](checks/index_01.md)) | CUDA | yes |
| `check_bench_matches_shipped.py` | that bench_e2e_h3.py's shipped arm... ([detail](checks/index_01.md)) | - | yes |
| `check_attention_defaults.py` | that every graph carries the... ([detail](checks/index_01.md)) | - | yes |
| `count_packed_rows.py` | not a check ([detail](checks/index_01.md)) | no CUDA | n/a — reports, does not assert |
| `check_node_ids.py` | the rule CLAUDE.md opens with, and... ([detail](checks/index_01.md)) | no CUDA | yes |
| `check_pdd_head_selection.py` | the PDD node's runtime guards, on... ([detail](checks/index_01.md)) | no CUDA | yes |
| `check_output_dir_resolution.py` | that no script reaches the render... ([detail](checks/index_01.md)) | nothing; pure... | yes |
| `check_pdd_bank_encoding.py` | that the published per-interval... ([detail](checks/index_01.md)) | nothing. Pure... | yes |
| `check_pdd_sigmas.py` | 2026-09-25 ([detail](checks/index_01.md)) | no CUDA | yes |
| `check_pdd_unmerged.py` | that the un-merged backbone path... ([detail](checks/index_01.md)) | nothing. Pure... | yes |
| `check_pdd_sidecar_contract.py` | that the PDD sidecar's bake... ([detail](checks/index_01.md)) | no CUDA | yes |
| `pdd_artifact_inventory.py` | with --check, that... ([detail](checks/index_01.md)) | no CUDA | yes |
| `pdd_artifact_inventory.py` | with --check, that... ([detail](checks/index_01.md)) | no CUDA | yes |
| `pdd_artifact_inventory.py` | with --check, that the generated... ([detail](checks/index_01.md)) | no CUDA | yes |
| `check_pdd_sidecar_contract.py` | that the PDD sidecar's bake... ([detail](checks/index_01.md)) | no CUDA | yes |
| `check_graph_values.py` | that a check can read a value a... ([detail](checks/index_02.md)) | no CUDA | yes |
| `check_retraction_consumers.py` | that a retracted claim has not... (`zero DiT calls`) ([detail](checks/index_02.md)) | - | yes |
| `check_audio_carry_inversion.py` | that MiniMaxH3AudioCarryProbe's... ([detail](checks/index_02.md)) | pure torch on a... | no |
| `check_reload_invariance.py` | whether a render is bit-identical... ([detail](checks/index_02.md)) | a GPU, a live... | no |
| `check_doc_links.py` | that a doc does not point at a file... ([detail](checks/index_02.md)) | - | yes |
| `check_provenance_stamp.py` | that provenance.py records the... ([detail](checks/index_02.md)) | - | yes |
| `check_exact_blocks.py` | that MiniMaxH3ExactBlocks still... ([detail](checks/index_02.md)) | - | yes |
| `check_channel_balance.py` | that MiniMaxH3ChannelBalance's fold... ([detail](checks/index_02.md)) | - | new 2026-09-14 |
| `check_token_routing.py` | that MiniMaxH3Sol's token_routing... ([detail](checks/index_02.md)) | - | new 2026-09-17, rewritten 2026-09-27 |
| `check_sol_kernel.py` | that the installed comfy_kitchen... ([detail](checks/index_02.md)) | - | yes |
| `check_literal_widgets.py` | that a numeric node input means the... ([detail](checks/index_02.md)) | - | no |
| `check_widget_deviations.py` | that every shipped widget value... ([detail](checks/index_02.md)) | a live... | no |
| `check_sol_chunked.py` | that the chunked-producer forward... ([detail](checks/index_02.md)) | CUDA | no |
| `check_sol_node_equivalence.py` | that sol_attn_h3.py, the fork taken... ([detail](checks/index_02.md)) | CUDA | no (kernel cases); yes (sink cases, OOM control) |
| `build_sidecar_node.py` | with --check, that the Hub bundle... ([detail](checks/index_02.md)) | no CUDA | no |
| `build_sidecar_examples.py` | with --check, that the example... ([detail](checks/index_02.md)) | no CUDA | no |
| `check_sol_observe.py` | that the Sol route recorder... ([detail](checks/index_02.md)) | CUDA | no |
| `check_pipeline_telemetry.py` | that the pipeline telemetry... ([detail](checks/index_02.md)) | the ComfyUI... | yes |
| `check_step_x0_observer.py` | that MiniMaxH3StepX0Observer... ([detail](checks/index_02.md)) | the ComfyUI... | yes |
| `check_core_sparse_capture.py` | that MiniMaxH3CoreSparseCapture... ([detail](checks/index_02.md)) | the ComfyUI... | yes |
| `check_denoise_mask_probe.py` | that MiniMaxH3DenoiseMaskProbe... ([detail](checks/index_02.md)) | the ComfyUI... | yes |
| `check_vae_precision_quantized.py` | that MiniMaxH3VAEPrecision refuses... ([detail](checks/index_02.md)) | the ComfyUI... | yes |
| `check_vsa_core_patch.py` | that this ComfyUI's VSA support is... ([detail](checks/index_02.md)) | - | no |
| `check_lora_alpha.py` | that every *_LORA in h3_config.py... ([detail](checks/index_02.md)) | - | yes |
| `check_marker_arms.py` | that a marker-corpus arm which was... ([detail](checks/index_03.md)) | the marker... | no |
| `check_model_files.py` | that no shipped graph and no... ([detail](checks/index_03.md)) | live server | yes |
| `check_model_contents.py` | that the bytes behind each model... ([detail](checks/index_03.md)) | the model files | yes **Since 2026-10-09 it also reads every `.safetensors` a constant or a generated graph names for an attention preference** (a key ending in `.config`, which core's model and LoRA loaders turn into the attention a module runs when no node has set an override): a file that carries one is red unless the baseline names it under `attention_preference_approved`. The graph it guards is the one with no attention node, the stamped dense baseline. `scan_can_fail` writes such a file on every run, so the green is a scan that can see one. |
| `check_h3_encoder_loader.py` | that MiniMaxH3EncoderLoader guards... ([detail](checks/index_03.md)) | no CUDA | no |
| `archive/bench/check_single_frame.py` | ARCHIVED 2026-08-27 with the... ([detail](checks/index_03.md)) | CUDA | yes |
| `check_keyframe_canvas.py` | canvas derivation, plus the aspect... ([detail](checks/index_03.md)) | PYTHONPATH... | yes |
| `measure_audio_video_coupling.py` | an instrument, not a gate ([detail](checks/index_03.md)) | OpenCV, PyAV | no |
| `audit_audio_freeze_control.py` | an audit, not a gate ([detail](checks/index_03.md)) | audio VAE on... | no |
| `check_audio_freeze.py` | the audio-freeze node's contract... ([detail](checks/index_03.md)) | PYTHONPATH... | yes |
| `check_frozen_video_cache.py` | MiniMaxH3FrozenVideoCache... ([detail](checks/index_03.md)) | PYTHONPATH... | yes |
| `check_video_mask.py` | masked video-to-video... ([detail](checks/index_03.md)) | PYTHONPATH... | no |
| `check_subject_track.py` | following one person across cuts... ([detail](checks/index_03.md)) | PYTHONPATH... | no |
| `check_shot_table.py` | the per-shot table (shot_table.py)... ([detail](checks/index_03.md)) | - | yes |
| `check_sam31_corrections.py` | the SAM 3.1 Corrections node... ([detail](checks/index_04.md)) | PYTHONPATH... | yes |
| `check_sam31_corrections_on_card.py` | the same node on the real SAM 3.1... ([detail](checks/index_04.md)) | CUDA | yes |
| `check_subject_tracks.py` | the model-free judgements about a... ([detail](checks/index_04.md)) | - | yes |
| `check_break_checks.py` | that bench/break_checks.py tells a... ([detail](checks/index_04.md)) | - | yes |
| `check_cast_file.py` | that bench/cast_file.py, the one... ([detail](checks/index_04.md)) | - | yes |
| `check_masked_prompt.py` | the masked lane's prompt node... ([detail](checks/index_04.md)) | - | yes |
| `check_prompt_bank.py` | the prompt bank's gate under a name... ([detail](checks/index_04.md)) | - | no |
| `check_subject_yaw.py` | the arithmetic of the turn metric... ([detail](checks/index_04.md)) | - | yes |
| `check_subject_parts.py` | body parts on the tracked subject... ([detail](checks/index_04.md)) | bench/_lib... | no |
| `check_capture_masked_run.py` | the tables... ([detail](checks/index_05.md)) | bench/_lib... | no |
| `check_frame_sheet.py` | the frame reader every... ([detail](checks/index_05.md)) | ffmpeg and... | no |
| `check_pick_frames.py` | the frame picker... ([detail](checks/index_05.md)) | ffmpeg and... | yes |
| `check_contact_sheets.py` | the contact sheets of an input... ([detail](checks/index_05.md)) | ffmpeg and... | no |
| `check_draw_lid_marks.py` | the motion video... ([detail](checks/index_05.md)) | bench/_lib... | no |
| `check_draw_face_points.py` | the motion video... ([detail](checks/index_06.md)) | bench/_lib... | no |
| `check_crop_reference_to_head.py` | the video... ([detail](checks/index_06.md)) | bench/_lib... | no |
| `check_head_reference.py` | the one command... ([detail](checks/index_06.md)) | bench/_lib... | no |
| `check_page_answers_to_verdicts.py` | that... ([detail](checks/index_06.md)) | nothing beyond... | no |
| `check_picks_against_answers.py` | that... ([detail](checks/index_06.md)) | numpy, ffmpeg | no |
| `check_who_page.py` | that bench/who_page.py, the static... ([detail](checks/index_06.md)) | PIL; ffmpeg for... | no |
| `check_who_view.py` | that bench/who_view.py, which... ([detail](checks/index_06.md)) | the standard... | yes |
| `check_renders_page.py` | that bench/renders_page.py, the... ([detail](checks/index_06.md)) | nothing beyond... | no |
| `check_render_ledger.py` | that bench/render_ledger.py, which... ([detail](checks/index_06.md)) | Pillow, to... | no |
| `check_render_history.py` | that bench/render_history.py, which... ([detail](checks/index_07.md)) | ffprobe... | no |
| `check_delivery_table.py` | that bench/delivery_table.py, which... ([detail](checks/index_07.md)) | the delivery... | no |
| `check_stack_delivery.py` | that bench/stack_delivery.py, which... ([detail](checks/index_07.md)) | ffmpeg and... | yes |
| `check_stack_rows.py` | that bench/stack_rows.py, which... ([detail](checks/index_07.md)) | ffmpeg; no... | no |
| `check_what_h3_got.py` | that bench/what_h3_got.py, the one... ([detail](checks/index_07.md)) | ffmpeg and... | no |
| `check_smoke_h3.py` | that bench/smoke_h3.py's readings... ([detail](checks/index_07.md)) | the pack's... | no |
| `check_load_chain.py` | that bench/load_chain.py, which... ([detail](checks/index_08.md)) | bench/_lib | no |
| `check_look_rooms.py` | that bench/look_rooms.py, which... ([detail](checks/index_08.md)) | bench/_lib | no |
| `check_shot_to_load.py` | that bench/shot_to_load.py, which... ([detail](checks/index_08.md)) | bench/_lib | no |
| `check_load_text.py` | that bench/load_text.py, which... ([detail](checks/index_09.md)) | the standard... | no |
| `check_strangers.py` | that bench/strangers.py, the... ([detail](checks/index_09.md)) | ffmpeg; no... | no |
| `check_propose_runs.py` | that bench/propose_runs.py, the... ([detail](checks/index_09.md)) | ffmpeg, torch | no |
| `check_runs_left.py` | that bench/runs_left.py, the queue... ([detail](checks/index_09.md)) | torch; no... | no |
| `check_laid.py` | that bench/laid.py says who chose a... ([detail](checks/index_09.md)) | no ffmpeg, no... | no |
| `check_build_cut_map.py` | that bench/build_cut_map.py's... ([detail](checks/index_09.md)) | no ffmpeg, no... | no |
| `check_look_carry.py` | that bench/look_carry.py, which... ([detail](checks/index_09.md)) | no card, no... | no |
| `check_others_present.py` | that bench/others_present.py, who... ([detail](checks/index_09.md)) | no card, no... | no |
| `check_two_shot_load.py` | that bench/two_shot_load.py, which... ([detail](checks/index_09.md)) | bench/_lib, the... | no |
| `check_page_answers_ledger.py` | that bench/page_answers_ledger.py... ([detail](checks/index_09.md)) | nothing beyond... | no |
| `check_read_hair_band.py` | the figures bench/read_hair_band.py... ([detail](checks/index_09.md)) | bench/_lib... | no |
| `check_page_answers_to_cast.py` | that bench/page_answers_to_cast.py... ([detail](checks/index_09.md)) | the cast file... | no |
| `check_shot_coverage.py` | that bench/shot_coverage.py, which... ([detail](checks/index_09.md)) | the delivery... | no |
| `check_shot_parts.py` | that bench/shot_parts.py, which... ([detail](checks/index_10.md)) | numpy, ffmpeg | no |
| `check_shots_look.py` | that bench/shots_look.py, which... ([detail](checks/index_10.md)) | the standard... | no |
| `check_shots_record.py` | that bench/shots_record.py, which... ([detail](checks/index_10.md)) | the standard... | no |
| `check_day_pieces.py` | that bench/day_pieces.py, which... ([detail](checks/index_10.md)) | the standard... | no |
| `check_piece_gaps.py` | that bench/piece_gaps.py, which... ([detail](checks/index_10.md)) | the standard... | no |
| `check_frame_coverage.py` | that bench/frame_coverage.py, which... ([detail](checks/index_10.md)) | numpy; no... | no |
| `check_frame_table.py` | One table per load of what happened... ([detail](checks/index_10.md)) | bench/_lib | no |
| `check_sidecar.py` | that bench/sidecar.py, which writes... ([detail](checks/index_10.md)) | ffmpeg, Pillow | no |
| `check_shot_review_clips.py` | that bench/shot_review_clips.py... ([detail](checks/index_10.md)) | the standard... | no |
| `check_review_pages.py` | that bench/review_pages.py, which... ([detail](checks/index_10.md)) | ffmpeg and... | no |
| `check_covered_items.py` | that bench/covered_items.py, which... ([detail](checks/index_11.md)) | the tool alone | no |
| `check_rebuild_delivery.py` | that bench/rebuild_delivery.py runs... ([detail](checks/index_11.md)) | ffmpeg... | no |
| `check_length_ceiling.py` | that the longest length anything... ([detail](checks/index_11.md)) | the tool alone | no |
| `check_read_render_against_original.py` | what... ([detail](checks/index_11.md)) | bench/_lib... | no |
| `check_loader_fit.py` | that bench/_lib/frames.py::FIT... ([detail](checks/index_11.md)) | bench/_lib... | no |
| `check_pose_along_shot.py` | everything round the model in... ([detail](checks/index_11.md)) | bench/_lib... | no |
| `check_meta_sapiens2.py` | that vendor/meta_sapiens2/ is... ([detail](checks/index_11.md)) | bench/_lib... | no |
| `check_recomposite_window.py` | the tool that lays a saved window... ([detail](checks/index_11.md)) | ffmpeg and... | no |
| `check_recomposite_piece.py` | the tool that lays a whole masked... ([detail](checks/index_11.md)) | ffmpeg and... | no |
| `check_voice_spans.py` | bench/voice_spans.py's frame... ([detail](checks/index_11.md)) | bench/_lib... | no |
| `check_clip_signals.py` | that bench/clip_signals.py, which... ([detail](checks/index_11.md)) | numpy, scipy... | no |
| `check_assemble_delivery.py` | the delivery file... ([detail](checks/index_12.md)) | bench/_lib... | no |
| `check_region_against_plate.py` | the figures... ([detail](checks/index_12.md)) | bench/_lib... | no |
| `check_patch_render_window.py` | the join of... ([detail](checks/index_12.md)) | bench/_lib... | no |
| `check_clip_analysis.py` | the tool that makes an input... ([detail](checks/index_12.md)) | bench/_lib... | yes |
| `check_clip_folder.py` | the shared rules of an input... ([detail](checks/index_12.md)) | bench/_lib... | yes |
| `check_copy_sound.py` | that bench/copy_sound.py, which... ([detail](checks/index_12.md)) | bench/_lib... | no |
| `check_masked_job.py` | the masked job builder... ([detail](checks/index_12.md)) | bench/_lib, the... | yes |
| `check_sweep_loads.py` | the copies bench/sweep_loads.py... ([detail](checks/index_12.md)) | bench/_lib... | no |
| `check_plate_restore.py` | a masked render split across two... ([detail](checks/index_13.md)) | PYTHONPATH... | no |
| ~~`check_mask_store.py`~~ | RETIRED 2026-10-10 with what it... ([detail](checks/index_13.md)) | n/a | no |
| `check_lora_branch.py` | MiniMaxH3LoRABranch... ([detail](checks/index_13.md)) | PYTHONPATH... | yes |
| `check_checkpoint_overlay.py` | checkpoint_overlay.py ([detail](checks/index_13.md)) | synthetic half | yes |
| `check_mutant_parity.py` | the standalone H3ExactLoRA... ([detail](checks/index_13.md)) | self-bootstrappe... | yes |
| `check_prompt_lists.py` | the __name__ prompt list rules the... ([detail](checks/index_13.md)) | no CUDA | no |
| `check_prompt_rule_controls.py` | controls for the prompt grader's... ([detail](checks/index_13.md)) | no CUDA | no |
| `check_generator_constants.py` | the workflow generator reads... ([detail](checks/index_13.md)) | PYTHONPATH... | no |
| `run_marker_stock_tail.py` | not a check ([detail](checks/index_13.md)) | a live server... | no |
| `grade_vae_encoder_precision.py` | not a gate ([detail](checks/index_13.md)) | no CUDA | no |
| `check_conditioning_behaviour.py` | that MiniMaxH3Conditioning still... ([detail](checks/index_13.md)) | - | no |
| `check_reference_contracts.py` | the load-bearing contracts in... ([detail](checks/index_13.md)) | - | no |
| ~~`check_mono_ref_audio.py`~~ | RETIRED 2026-08-29. ([detail](checks/index_13.md)) | audio VAE on CPU | no |
| `check_doc_inventory.py` | that this index answers "what does... ([detail](checks/index_13.md)) | - | no |
| `check_changelog.py` | that CHANGELOG.md's newest entries... ([detail](checks/index_13.md)) | - | no |
| `check_no_owner_paths.py` | that repository text never stores a... ([detail](checks/index_13.md)) | git | no |
| `check_capture_manifest.py` | that every activation capture... ([detail](checks/index_13.md)) | the capture... | no |
| `check_capture_manifest_controls.py` | that the manifest contract can go... ([detail](checks/index_13.md)) | torch, the... | no |
| `check_graph_discovery.py` | that no check finds graphs by... ([detail](checks/index_13.md)) | - | no |
| `check_reference_order.py` | the ordered-reference plan... ([detail](checks/index_13.md)) | no CUDA | no |
| `check_reference_runtime.py` | the executable... ([detail](checks/index_13.md)) | no CUDA | no |
| `check_reference_encode.py` | the split reference path's pieces... ([detail](checks/index_13.md)) | no CUDA | yes |
| ~~`check_window_keep.py`~~ | RETIRED 2026-10-10 with what it... ([detail](checks/index_13.md)) | n/a | no |
| `check_typed_reference_consumers.py` | label discovery and preflight both... ([detail](checks/index_13.md)) | - | no |
| `check_release_qwen_grid.py` | that the release's duration-aware... ([detail](checks/index_13.md)) | no CUDA | no |
| `check_dit_prefix_attention.py` | that... ([detail](checks/index_13.md)) | PYTHONPATH, CPU... | yes |
| `measure_dit_prefix_attention.py` | not a check ([detail](checks/index_14.md)) | a capture... | n/a |
| `build_wiki_index.py` | not a gate ([detail](checks/index_14.md)) | - | no |
| `check_prompt_docs_sync.py` | first, that the two vendor guides... ([detail](checks/index_14.md)) | - | no |
| `check_skill_routes.py` | that every repo-relative path a... ([detail](checks/index_14.md)) | - | no |
| `check_speaker_id_control.py` | that... ([detail](checks/index_14.md)) | - | yes |
| `check_camera_vocabulary.py` | that every shipped prompt's camera... ([detail](checks/index_14.md)) | the graphs | no |
| `check_sam3d_body_conversion.py` | that... ([detail](checks/index_14.md)) | no CUDA | no |
| `check_sam3d_body_vith.py` | that sam3d_body_vith.py is Meta's... ([detail](checks/index_14.md)) | - | yes |
| `check_body_pose.py` | that body_pose.py gives SAM 3D Body... ([detail](checks/index_14.md)) | - | yes |
| `check_imports.py` | that every top-level... ([detail](checks/index_14.md)) | bench/_lib | no |
| `check_commit_ready.py` | that bench/check_commit_ready.py... ([detail](checks/index_14.md)) | bench/_lib | no |
| `check_head_tube_parts.py` | that bench/head_tube_parts.py... ([detail](checks/index_14.md)) | no CUDA | no |
| `check_measure_moge2_depth.py` | that... ([detail](checks/index_14.md)) | no CUDA | yes |
| `check_sam_memory_selection.py` | that bench/sam_memory_selection.py... ([detail](checks/index_14.md)) | bench/_lib... | no |
| `check_measure_blocks.py` | that bench/measure_blocks.py... ([detail](checks/index_14.md)) | no CUDA | yes |
| `check_facts_pass.py` | that bench/facts_pass.py reads a... ([detail](checks/index_14.md)) | no CUDA | yes |
| `check_code_stamp.py` | that bench/_lib/code_stamp.py... ([detail](checks/index_14.md)) | no CUDA | yes |
| `check_said_on_shot.py` | that bench/said_on_shot.py... ([detail](checks/index_14.md)) | no CUDA | no |
| `check_inventory.py` | that bench/inventory.py... ([detail](checks/index_14.md)) | no CUDA | no |
| `check_timeline_page.py` | that bench/timeline_page.py... ([detail](checks/index_14.md)) | no CUDA | no |
| `check_flags_before_render.py` | that bench/flags_before_render.py... ([detail](checks/index_14.md)) | no CUDA | no |
| `check_inspect_mask_steps.py` | that bench/inspect_mask_steps.py... ([detail](checks/index_14.md)) | no CUDA | no |
| `build_prompt_bank.py` | a generator ([detail](checks/index_14.md)) | - | no |
| `masked_code_map.py` | a generator and AST drift gate ([detail](checks/index_15.md)) | no CUDA | no |
| `check_masked_code_map.py` | that bench/masked_code_map.py... ([detail](checks/index_15.md)) | no CUDA | no |
| `preflight_graph.py` | not a gate ([detail](checks/index_15.md)) | no CUDA | no |
| `smoke_h3.py` | the H3 chain composes and runs... ([detail](checks/index_15.md)) | live ComfyUI... | no |

## What is deliberately not checked

`docs/open_experiments.md` is the other half of this document: seven things
this repo has decided **not** to measure, each with its cost, the decision it
would change, and the actual blocker. Read it before proposing a new check --
several obvious ideas are already there with a reason attached.

A suite of twelve **render** scenes is designed but not run. They are quality
gates for output, not code checks: each carries a pre-registered binary claim
and a predicted per-arm direction, and a human watches and listens. Nothing
executes them, and nothing here can -- judging whether a third shot happened
or two voices stayed distinct is not a `check_*.py`.

> Those live in `internal/`, which is **gitignored and not distributed**. If
> you cloned this repo you do not have them, and that is deliberate: they are
> the owner's working research notes, not shipped content. Everything this
> document describes is in `bench/` and is present in the clone.

One thing there has no code check and cannot get one until the graphs exist:

- **Two-stage split graphs do not exist** (the lightx2v ones were retired
  2026-09-26). When one does, the checks worth writing are: both stages read
  one schedule, stage 2 carries `DisableNoise`, the split point is inside the
  step range, and the two stages' shift nodes agree.


### Citations that cannot resolve, for `check_doc_links.py`

Full multi-line `WHY:` histories are preserved in [`docs/checks/doc_link_absent_history.md`](checks/doc_link_absent_history.md).

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

```
