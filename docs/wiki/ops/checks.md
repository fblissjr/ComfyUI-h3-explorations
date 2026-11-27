# What this repo checks, and why

Index of every check in `bench/`. One row per script: what it defends, what it
needs to run, and whether it has earned trust.

There is no test suite. Each script is standalone, prints its own `ok` /
`FAIL` lines, and returns a non-zero exit code on failure.
`bench/run_checks.py` runs them all with the card masked and compares what is
red with `bench/checks_baseline.json` ("Running them").

**Per-check narrative and the run logs live in
[`docs/wiki/archive/check_postmortems.md`](../archive/check_postmortems.md)**, not here. This file is
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


For `break_checks.py` history, metric-ranking, expression-driving, and queue-execution case studies, as well as retired checks, uncontrolled requirements, and gaps, see [`docs/wiki/ops/checks/standard_and_gaps.md`](checks/standard_and_gaps.md).

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
<comfy venv python> bench/run_checks.py --jobs 1   # one at a time; the default is DEFAULT_JOBS at once
<comfy venv python> bench/run_checks.py --help     # every flag, from its argparse
```

The sweep is a command-line tool with an argparse `--help` that lists every
flag; its module docstring (`bench/run_checks.py`) has the rules behind them.
`--jobs N` (`bench/run_checks.py::DEFAULT_JOBS` by default) runs that many
checks at once, each in its own process; rows still print in check order and
the verdict does not depend on N while every check keeps to its own temporary
directory (the module docstring, "Running checks side by side").


For `--changed`, `CUDA_VISIBLE_DEVICES=` masking rules, and `bench/_lib::bootstrap` details, see [`docs/wiki/ops/checks/standard_and_gaps.md`](checks/standard_and_gaps.md#running-sweep--masking-details).

## The index

Full per-check history, 2026-08-23 amendments, and `"claims block"` notes are sharded under `docs/wiki/ops/checks/` (each `< 35 KB`; see [`docs/wiki/ops/checks/index_15.md`](checks/index_15.md) for the closing notes):

| Shard | Checks covered |
|---|---|
| [`docs/wiki/ops/checks/standard_and_gaps.md`](checks/standard_and_gaps.md) | Standard case studies, sweep/masking notes, retired checks, uncontrolled requirements & gaps |
| [`docs/wiki/ops/checks/doc_link_absent.md`](checks/doc_link_absent.md) | Active machine-read `doc-link-absent` ledger parsed by `bench/check_doc_links.py` |
| [`docs/wiki/ops/checks/doc_link_absent_history.md`](checks/doc_link_absent_history.md) | Full historical `WHY:` entries for deliberately absent citations |
| [`docs/wiki/ops/checks/index_01.md`](checks/index_01.md) | `check_correctness.py` .. `check_pdd_sidecar_contract.py` (25 rows) |
| [`docs/wiki/ops/checks/index_02.md`](checks/index_02.md) | `check_graph_values.py` .. `check_lora_alpha.py` (24 rows) |
| [`docs/wiki/ops/checks/index_03.md`](checks/index_03.md) | `check_marker_arms.py` .. `check_shot_table.py` (13 rows) |
| [`docs/wiki/ops/checks/index_04.md`](checks/index_04.md) | `check_sam31_corrections.py` .. `check_subject_parts.py` (9 rows) |
| [`docs/wiki/ops/checks/index_05.md`](checks/index_05.md) | `check_capture_masked_run.py` .. `check_draw_lid_marks.py` (5 rows) |
| [`docs/wiki/ops/checks/index_06.md`](checks/index_06.md) | `check_draw_face_points.py` .. `check_render_ledger.py` (9 rows) |
| [`docs/wiki/ops/checks/index_07.md`](checks/index_07.md) | `check_render_history.py` .. `check_smoke_h3.py` (6 rows) |
| [`docs/wiki/ops/checks/index_08.md`](checks/index_08.md) | `check_load_chain.py` .. `check_shot_to_load.py` (3 rows) |
| [`docs/wiki/ops/checks/index_09.md`](checks/index_09.md) | `check_load_text.py` .. `check_shot_coverage.py` (13 rows) |
| [`docs/wiki/ops/checks/index_10.md`](checks/index_10.md) | `check_shot_parts.py` .. `check_review_pages.py` (10 rows) |
| [`docs/wiki/ops/checks/index_11.md`](checks/index_11.md) | `check_covered_items.py` .. `check_clip_signals.py` (11 rows) |
| [`docs/wiki/ops/checks/index_12.md`](checks/index_12.md) | `check_assemble_delivery.py` .. `check_sweep_loads.py` (8 rows) |
| [`docs/wiki/ops/checks/index_13.md`](checks/index_13.md) | `check_plate_restore.py` .. `check_dit_prefix_attention.py` (26 rows) |
| [`docs/wiki/ops/checks/index_14.md`](checks/index_14.md) | `measure_dit_prefix_attention.py` .. `build_prompt_bank.py` (23 rows) |
| [`docs/wiki/ops/checks/index_15.md`](checks/index_15.md) | `masked_code_map.py` .. `check_h3_env.py` (16 rows) |

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
| ~~`check_pdd_head_selection.py`~~ | **RETIRED 2026-11-06 with the PDD lane.** the PDD node's runtime guards, on... ([detail](checks/index_01.md)) | no CUDA | yes |
| `check_output_dir_resolution.py` | that no script reaches the render... ([detail](checks/index_01.md)) | nothing; pure... | yes |
| ~~`check_pdd_bank_encoding.py`~~ | **RETIRED 2026-11-06 with the PDD lane.** that the published per-interval... ([detail](checks/index_01.md)) | nothing. Pure... | yes |
| ~~`check_pdd_sigmas.py`~~ | **RETIRED 2026-11-06 with the PDD lane.** 2026-09-25 ([detail](checks/index_01.md)) | no CUDA | yes |
| ~~`check_pdd_unmerged.py`~~ | **RETIRED 2026-11-06 with the PDD lane.** that the un-merged backbone path... ([detail](checks/index_01.md)) | nothing. Pure... | yes |
| ~~`check_pdd_sidecar_contract.py`~~ | **RETIRED 2026-11-06 with the PDD lane.** that the PDD sidecar's bake... ([detail](checks/index_01.md)) | no CUDA | yes |
| ~~`pdd_artifact_inventory.py`~~ | **RETIRED 2026-11-06 with the PDD lane.** with --check, that... ([detail](checks/index_01.md)) | no CUDA | yes |
| `check_graph_values.py` | that a check can read a value a... ([detail](checks/index_02.md)) | no CUDA | yes |
| `check_retraction_consumers.py` | that a retracted claim has not... (`zero DiT calls`) ([detail](checks/index_02.md)) | - | yes |
| ~~`check_audio_carry_inversion.py`~~ | **RETIRED 2026-11-06 with the PDD lane.** that MiniMaxH3AudioCarryProbe's... ([detail](checks/index_02.md)) | pure torch on a... | no |
| ~~`check_reload_invariance.py`~~ | **RETIRED 2026-11-06 with the PDD lane.** whether a render is bit-identical... ([detail](checks/index_02.md)) | a GPU, a live... | no |
| `check_doc_links.py` | that a doc does not point at a file... ([detail](checks/index_02.md)) | - | yes |
| `check_provenance_stamp.py` | that provenance.py records the... ([detail](checks/index_02.md)) | - | yes |
| `check_exact_blocks.py` | that MiniMaxH3ExactBlocks still... ([detail](checks/index_02.md)) | - | yes |
| `check_channel_balance.py` | that MiniMaxH3ChannelBalance's fold... ([detail](checks/index_02.md)) | - | new 2026-09-14 |
| `check_token_routing.py` | that MiniMaxH3Sol's token_routing... ([detail](checks/index_02.md)) | - | new 2026-09-17, rewritten 2026-09-27 |
| `check_sol_kernel.py` | that the installed comfy_kitchen... ([detail](checks/index_02.md)) | - | yes |
| `check_literal_widgets.py` | that a numeric node input means the... ([detail](checks/index_02.md)) | - | no |
| `check_widget_deviations.py` | that every shipped widget value... ([detail](checks/index_02.md)) | a live... | no |
| `archive/bench/check_sol_chunked.py` | ARCHIVED 2026-11-13 (`MiniMaxH3SolChunked` retired to `archive/sol_chunked_h3.py`): that the chunked-producer forward... ([detail](checks/index_02.md)) | CUDA | no |
| `check_sol_node_equivalence.py` | that sol_attn_h3.py, the fork taken... ([detail](checks/index_02.md)) | CUDA | no (kernel cases); yes (sink cases, OOM control) |
| ~~`build_sidecar_node.py`~~ | **RETIRED 2026-11-06 with the PDD lane.** with --check, that the Hub bundle... ([detail](checks/index_02.md)) | no CUDA | no |
| ~~`build_sidecar_examples.py`~~ | **RETIRED 2026-11-06 with the PDD lane.** with --check, that the example... ([detail](checks/index_02.md)) | no CUDA | no |
| `check_sol_observe.py` | that the Sol route recorder... ([detail](checks/index_02.md)) | CUDA | no |
| `check_pipeline_telemetry.py` | that the pipeline telemetry... ([detail](checks/index_02.md)) | the ComfyUI... | yes |
| `check_step_x0_observer.py` | that MiniMaxH3StepX0Observer... ([detail](checks/index_02.md)) | the ComfyUI... | yes |
| `check_core_sparse_capture.py` | that MiniMaxH3CoreSparseCapture... ([detail](checks/index_02.md)) | the ComfyUI... | yes |
| `archive/bench/check_denoise_mask_probe.py` | ARCHIVED 2026-11-25 (`MiniMaxH3DenoiseMaskProbe` retired to `archive/denoise_mask_probe.py`): that MiniMaxH3DenoiseMaskProbe... ([detail](checks/index_02.md)) | the ComfyUI... | yes |
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
| ~~`check_checkpoint_overlay.py`~~ | **RETIRED 2026-11-25 with the FastH3 overlay lane (`archive/checkpoint_overlay.py`, `archive/overlay_loader.py`).** ([detail](checks/index_13.md)) | n/a | no |
| `check_h3_image.py` | MiniMax-H3-Image: the release-to-ComfyUI layout, the size rule, `MiniMaxH3ImagePromptExpander` (`concept` + `@extend:prepend\|append` overlays), and the shipped graphs' chain ([detail](checks/index_13.md)) | no CUDA | no |
| ~~`check_mutant_parity.py`~~ | RETIRED 2026-11-04 with the package it... ([detail](checks/index_13.md)) | n/a | no |
| `check_prompt_lists.py` | the __name__ prompt list rules the... ([detail](checks/index_13.md)) | no CUDA | no |
| `check_prompt_rule_controls.py` | controls for the prompt grader's... ([detail](checks/index_13.md)) | no CUDA | no |
| `check_generator_constants.py` | the workflow generator reads... ([detail](checks/index_13.md)) | PYTHONPATH... | no |
| `run_marker_stock_tail.py` | not a check ([detail](checks/index_13.md)) | a live server... | no |
| `grade_vae_encoder_precision.py` | not a gate ([detail](checks/index_13.md)) | the card for `--fp16-accumulation` | no |
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
| `check_control_vectors.py` | that h3_spans.py, control_vectors.py, control_vector_nodes.py, build_workflows.py, and bench/control_vectors.py (including `add_norm_matched_erasure`, `erasure.mu_a` vision centering, Gate B `med_fd < 10`, Gate D contradiction lint, and `MIN_ERASURE_PAIRS = 6`)... ([detail](checks/index_15.md)) | no CUDA | yes |
| ~~`check_control_vector_studio.py`~~ | retired 2026-11-22 alongside standalone Control Vector Studio (`:8196`, superseded by `app/` `/lab` and `bench/control_vectors.py`) | — | — |
| `check_steering_parity.py` | that active `prompt_bank/control_vectors/*.toml`, `_retired/`, `models/control_vectors/bank/` (`fl2va`/`ref2va`/`h3_image`), `data/control_vectors/`, and `model2vec_h3_lexicon.safetensors` stay in parity... ([detail](checks/index_15.md)) | no CUDA | yes |
| `check_control_vector_readout_score.py` | that bench/control_vector_readout_score.py builds blind judge packets that leak nothing (one option list per group of near concepts) and counts answers as it says... ([detail](checks/index_15.md)) | no CUDA | yes |
| `check_guide_latent.py` | that MiniMaxH3GuideLatent replaces ReferenceToVideo + AddGuide for Ref2VA guide-latent workflows... ([detail](checks/index_15.md)) | no CUDA | yes |
| `check_edit_guidance.py` | that edit_guidance.py places a prompt edit exactly and refuses what it cannot place, and that the edit guider computes original + scale * (edited - original)... ([detail](checks/index_15.md)) | no CUDA | yes |
| `check_config_names.py` | that every h3_config name a tracked Python file reads is one workflows/h3_config.py defines... ([detail](checks/index_15.md)) | no CUDA | yes |
| `check_lane_db.py` | that bench/lane_db/, the lane's record in SQLite, keeps its schema's rules, refuses a field no mapping line covers, and gives back the files it was built from... ([detail](checks/index_15.md)) | no CUDA | no |
| `check_sound_and_sprite_nodes.py` | that MiniMaxH3SoundCues merges close cues onto non-overlapping 40 Hz RoPE spans over a -28 dBFS floor and MiniMaxH3AppendSpriteTrajectory decouples (x, y) from facing polarity without upscaling 384x224... ([detail](checks/index_15.md)) | no CUDA | yes |
| `check_image_studio.py` | that the standalone H3 Pocket Console (`app/test_app.py`) passes all offline Tier 1 contract, security, workflow-parity, and `execution.validate_prompt` checks against live pack schemas (`app/tests/` pytest suites run separately inside `app/`) ([detail](checks/index_15.md)) | `app/test_app.py`, no CUDA | yes |
| `check_spatial_director.py` | that `MiniMaxH3SpatialDirector` compiles `portal_path` / `point_targets` / `combined` `[F, H, W]` masks, `include_mask` / `exclude_mask`, `3f` SoundCues, D-pad sprite trajectories, and refusals, `MiniMaxH3DirectorHUD` renders even-clamped overlays/PiP/split-A/B deltas, and all four shipped spatial director workflows (`t2v`, `world_targets`, `r2v`, `i2v`) stay wired | no CUDA | yes |
| `check_director_studio.py` | that the standalone Spatial Director Studio (`director/`) compiles `r2v_portal` / `i2v_portal` (`last_frame`) / `t2v_portal` (`combined`, `audio_refine`) / `world_targets` + `pika_t2i` stills + two-recipe `compile_audio_redub` (`polish` `0.5x` nested `fit_audio` vs. `fresh` `1.0x` `ManualSigmas`) with `save_output=False`, dual `VHS_VideoCombine` (`13`+`162`), conditional `51`/`70`, and dynamic `165` `ModelPreviewOverrideKJ` (`preview_fps=7` + `taeh3.safetensors`), protects `$H3_OUTPUT_DIR/Video/` and active uploads in `compute_forget_plan`, supports `/api/segment` (SAM 3.1/GrabCut), `/api/extract_frame`, `/api/redub`, MP4 take import in `/api/upload`, and `/api/escalate` on `on_both_clocks`, enforces zero CDN/inline/hardcoded presets across 19 `/api/*` routes in `director/static/`, and passes Tier 2 headless Chrome (`1280x800`) CDP against isolated `--cpu` ComfyUI (`:8288`) + `director/server.py` (`:8295`) (exits `2` via `bench/_lib::needs` when `director/` is absent) | `director/`, no CUDA; Tier 2 needs ComfyUI + Chrome | yes |
| `check_lse_balance.py` | that `attention.py` (`lse_subset_balance_bhnd`, `install_lse_balance_override`, `attach_lse_reference_balance`) matches an explicit `float64` re-weighted softmax oracle within `1e-13` while preserving non-target query rows bitwise (`0.0`) and outside-subset weights within `1e-14`, composes beneath `MiniMaxH3Sol` (`route("lse_balance", ...)` in `sol_attn_h3.py` and `sol_observe.py`) and `provenance.py` in either node order, and matches live CUDA FlashAttention-2 (`D = 128`) and `sage-fork` `return_lse=True` when unmasked on a GPU | no CUDA (runs additional CUDA FlashAttention-2 + `sage-fork` parity case when unmasked) | no |
| `replay_cranium_references.py` | 3-way replay of all cranium references ([detail](checks/index_15.md)) | no CUDA | no |
| `preflight_graph.py` | not a gate ([detail](checks/index_15.md)) | no CUDA | no |
| `smoke_h3.py` | the H3 chain composes and runs... ([detail](checks/index_15.md)) | live ComfyUI... | no |
| `check_h3_image_vae_convert.py` | bench/convert_h3_image_vae_from_diffusers.py writes a diffusers MiniMax-H3-Image VAE in core's layout ([detail](checks/index_15.md)) | no CUDA | no |
| `check_h3_env.py` | this machine's locations stay in the gitignored h3.env; the template, loader and writer ([detail](checks/index_15.md)) | git | no |
| `check_preview_gate.py` | that `bench/run_with_preview_gate.py` parses V1 and V3 `COMBO` choices, injects in-memory `ModelPreviewOverrideKJ` (`taeh3.safetensors` for multi-frame video, `taeh3_image.safetensors` for single-frame stills, leaving `vae`/`audio_vae` unwired) and safely bypasses when absent, records WebSocket `kj_preview_override` frames (including `step=0` init noise) to `steps/` and `steps.jsonl`, cancels early via `--stop-after-step` + `POST /interrupt`, and collects completed `/history` outputs against an in-process `aiohttp` mock server | `aiohttp`, no CUDA, no model weights | no |
| `check_data_results_lookup.py` | that `bench/_lib/results.py` finds archived run data where the archive move put it (the path as written, then the same path in each dated `<first>_to_<last>` folder) and never by a bare name elsewhere, refuses a path two archive folders hold, globs across the archive folders; and that `bench/resolve_data_citations.py` reports from the same lookup and parses citations without templates or `--out` paths. Runs on a temporary tree; five deliberate breaks of the lookup and the parser were each seen (`bench/break_checks.py`, 2026-11-18) | - | yes |
| `check_auto_edit_prompt.py` | that `auto_edit_prompt.py`, `h3_prompt_expander.py` (`ref_end_percent`), `h3_prompt_contract.py`, and `heylook_client.py` (`@extend:prepend\|append` and `<!-- H3_EXTEND:prepend\|append -->` runtime overlays) discover `.toml` concepts in priority order, round-trip H3 prompt sections losslessly, enforce dialogue/shot/speaker/media-tag/special-token/`[task]`/`<Audio N>` retention invariants across `section_edits` and `edit_lines` (with 1-shot APC repair), compile 4-axis `EditPlan` overrides and `[F, 1, 1]` temporal curves into `MiniMaxH3EditGuider` and `MiniMaxH3ReleaseReferences`, hit the in-process LRU cache with `(path, mtime_ns, size)` and preset invalidation, gate `reasoning_effort` by `/v1/models` capabilities, handle `503` retries / `event: error` / `DELETE` cancellation against a loopback SSE mock, and keep both shipped `*_auto_edit_guidance_api.json` + 1x `prompt_expander` + split-cached `r2v` graphs valid and free of host IPs | no CUDA, loopback HTTP mock | yes |
| `check_index_steering_renders.py` | verifies `bench/index_steering_renders.py` PNG metadata parsing (`MiniMaxH3Steer`, `MiniMaxH3EditGuider`, `fast_1x`), `--since` filtering, and SQLite views (`v_problem_summary`, `v_renders_flat`, `v_vector_health`) | `PIL`, `sqlite3`, no CUDA | no |
| `check_m1_m2_probes.py` | verifies `M1` (`bench/probe_dinov3_copy_freeze.py`) and `M2` (`bench/probe_yin_f0_drift.py`) offline CPU triage math (`m2_yin_sine_and_harmonic_shift`, `m2_silence_gate_and_voice_spans`, `m1_score_clip_embeddings`, and edge guards) | `torch`, no CUDA | no |
| `check_sync_heylook_presets.py` | verifies `bench/sync_heylook_presets.py` 21-preset catalog & fenced-block extraction, exact remote matching, param preservation/`--reset-params`, `--prefix`, auth forwarding, downstream inheritance, and idempotency against an in-process `httpx` mock | `httpx`, no CUDA | no |
| `check_heylook_prompts.py` | verifies zero internal lore, zero crutch examples, positive-only optical restraint, closed vendor camera/reference vocabularies, preset prefix precedence/inheritance, and 14-key `HeylookProvenance` across runtime system prompts, callers, and `llm_pairs.py` | no CUDA, loopback HTTP mock | yes |
| `check_steering_v2.py` | offline CPU verification gate for the clean-slate `steering/` v2 package: bank tensor/Point C prelude parity, direct unpooled span extraction + `build_pole_a_erasure` parity, `MixerCompiler` & `director/compiler.py` routing across non-film concept families, 14-key `[provenance]` round-trip, and stratified pair-compiler diversity | `torch`, no CUDA | yes |
| `check_query_router.py` | offline CPU verification gate for `MiniMaxH3QueryRouter` (`steering/query_router.py`), arbitrary-length shadow K/V in `steering/hooks.py`, and mixed-band `MiniMaxH3Steer`: exact `audio_only` parity, shorter/longer dual-prompt routing, pre-`TokenRefiner` `section_split` (without sectionless fallback), mixed `video_block_band` + `audio_block_band`, refusals, and schema registration | `torch`, no CUDA | yes |
| `check_spatial_query_router.py` | offline CPU verification gate for `MiniMaxH3SpatialQueryRouter` (`steering/spatial_query_router.py`): 12-input schema registration, box/invert/static+video `MASK` and `ref2va`/`fl2va` segment partitioning, `early_0_24`/`seamless_off`/block-band runtime isolation, `audio_zone`, mismatched text lengths, `AttentionTensorContainer` lifecycle, and empty/full mask short-circuits | `torch`, no CUDA | yes |
| `check_probe_h3_sensors.py` | offline CPU verification gate for the read-only 4-sensor H3 calibration probe (`bench/probe_h3_sensors.py`): BPE noun span matching, zero-mutation bitwise parity (`max_abs == 0.0`), Sensor 1 (`Q_vid @ K_txt[noun]^T`) quadrant/sink localization, Sensor 2 (`Q_vid @ K_aud[t]^T`) stereo L/R separation, Sensor 3 pure-latent `audio_x0` transient detection, `MiniMax-H3-Image` (`latent_t=1, audio_t=0`) parity, Sensor 4 `MiniMaxH3SpatialQueryRouter` dual-order stacking + synthetic J-Lens `decode_vector`, and repo `data/` output refusal + PNG/JSON export | `torch`, `PIL`, no CUDA | yes |
| `check_live_policy.py` | offline CPU verification gate for Beat 3B `steering/live_policy.py` (`H3LivePolicy` & `MiniMaxH3SelfMaskRouter`, tests `0..15`): BPE phrase matching, sparse-zero 4D attention mask synthesis (`cross_time_scale_floor=0.55`) + lossless `SpatialVisualQRRuntime` frame expansion, Visual & Acoustic (`audio_x0`) + `video_x0` thermostats, Step-0 `B35+B40` harvest (`0.0` parity) -> Step-1 `early_0_24_shield` + `replay_step0=True` (`1.17×`) routing, `left_speaker`/`right_speaker`/`active_speaker` & transparent `audio_transients` masks, `SelfMask -> 2x EditGuider` bridge (`edit_guidance.py`), Run-#2-safe KJ preview / `MiniMaxH3Steer` / `0.33` release / wire-order-independent `QueryRouter` `a2v=0.0` actuators, Dynamic Step-1 Auto-Shield, true `1.17×` Step-0 Visual Critic rollback replay + CPU `taeh3` decode, `MiniMax-H3-Image` parity, `ON_CLEANUP` + `reset_for_new_run()` isolation, refusal/16-input schema guards, and `harvest_heads` / `reharvest` (named heads only; a peek-only second reading on the routed pass) | `torch`, `numpy`, no CUDA | yes |
| `check_depth_beat_mask.py` | offline CPU verification gate for composable `MiniMaxH3DepthBeatMask` (`depth_beat_mask.py`, `0..5/6`): synthetic depth bands (`depth_band`, `depth_wave`), `17*m` WAV beat snapping (`audio_beat_cut`, `audio_beat_pulse`), `depth_band_on_beat` temporal gating (`f0..cut-1` zero), `invert`, `[F, H, W]` shape, `resolve_moge2_checkpoint` systemd `$HOME` override regression, and 13-input schema invariants | `torch`, `numpy`, no CUDA | yes |

## What is deliberately not checked

`docs/wiki/ops/open_experiments.md` is the other half of this document: seven things
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

Sharded on `2026-11-25` so this router stays under `45,000 B`:

- **Active machine-read `doc-link-absent` ledger** (parsed by [`bench/check_doc_links.py`](../../../bench/check_doc_links.py)): [`docs/wiki/ops/checks/doc_link_absent.md`](checks/doc_link_absent.md)
- **Full multi-line historical `WHY:` notes:** [`docs/wiki/ops/checks/doc_link_absent_history.md`](checks/doc_link_absent_history.md)
