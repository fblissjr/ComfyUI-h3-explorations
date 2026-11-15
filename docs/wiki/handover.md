# Handover: read this before anything else

Updated `2026-11-15` after adding interactive 2D+time `MiniMaxH3SpatialDirector` / `MiniMaxH3DirectorHUD`, multi-frame `[F, H, W]` `40 Hz` audio envelope gating, `MiniMaxH3Steer.edit_pass` + `fl2va_base` J-Lens, Multiplex `SAM3-DMS` (`sam31_corrections.py`), and upstream subfolder tracking (`docs/wiki/upstream/index.md`); built on the `2026-11-13` LLM Wiki layout (`docs/wiki/index.md`).
Where this page and code/verdicts disagree, the code and `bench/results/` records are right: check [`../../AGENTS.md`](../../AGENTS.md), [`index.md`](index.md), [`decisions.md`](decisions.md), and [`projects/2026-11-13_repo_audit_and_atlas/index.md`](projects/2026-11-13_repo_audit_and_atlas/index.md).

## Current State (`2026-11-15`)

0. **3D Semantic Velocity Steering, `MiniMaxH3SpatialDirector`, & `edit_pass` (`2026-11-14`..`2026-11-15`)**:
   - Extended `MiniMaxH3EditGuider.mask` ([`../../edit_guidance.py`](../../edit_guidance.py)) across `[H, W]`, causal `[F, H, W]` (`[1, 4, 4, 4, 4]` video pooling + `40 Hz` `max(H, W)` temporal audio envelope gating when `F > 1`), and `[F, 1, 1]` shapes across `NestedTensor`, `packed_split` (`latent_shapes[0]`), and unpacked `MiniMax-H3-Image` latents, paired with `MiniMaxH3Steer.edit_pass` (`both | edited_only | original_only`; [`../../control_vector_nodes.py`](../../control_vector_nodes.py)) for regional vector painting, Split-World dual vectors, and `first_step = 0` regional `project_out` scrubs.
   - Shipped `MiniMaxH3SpatialDirector` & `MiniMaxH3DirectorHUD` ([`../../spatial_director.py`](../../spatial_director.py), [`../../web/h3_spatial_director.js`](../../web/h3_spatial_director.js), `66` registered node IDs) with interactive `portal_path` (`circle`, `ellipse`, `vertical_split`) and Genie + `point_targets` workflows ([`../../workflows/h3_spatial_portal_pika_api.json`](../../workflows/h3_spatial_portal_pika_api.json), [`../../workflows/h3_world_targets_genie_pika_api.json`](../../workflows/h3_world_targets_genie_pika_api.json)). Write-up: [`research/2026-11-15_spatial_world_director_and_3d_edit_guidance.md`](research/2026-11-15_spatial_world_director_and_3d_edit_guidance.md).
   - Aligned `MiniMaxH3AppendRefImage.allow_upscale` default to `False` (`sglang#47907`, `docs/wiki/upstream/topics/reference_sizing.md`) and rebuilt `j_lens_h3.safetensors` with `fl2va_base` (`P-H3-Image`) operators.
   - Upgraded `MiniMaxH3SAM31Corrections` ([`../../sam31_corrections.py`](../../sam31_corrections.py), [`../../bench/results/2026-11-15_sam31_dms_and_fixes.md`](../../bench/results/2026-11-15_sam31_dms_and_fixes.md)) with Multiplex `SAM3-DMS` (`N_obj >= 2`, preserving `N_obj == 1` at `diff_px = 0`) plus default-off `correct_multimask_ptr` and `correct_detector_presence` (`745f2193`, `11c0a443`).

1. **LLM-Wiki-First Documentation & Repo Audit (`2026-11-13`)**:
   - All active documentation now lives under [`index.md`](index.md) (`core/`, `image/`, `attention/`, `prompting/`, `control_vectors/`, `system_prompts/`, `masked_v2v/`, `ops/`, `projects/`, `research/`, `upstream/`, `decisions/`, `archive/`).
   - Full cross-session portfolio atlas, code-vs-prose truth ledger, and Top 14 high-leverage unfinished/forgotten ideas live in [`projects/2026-11-13_repo_audit_and_atlas/index.md`](projects/2026-11-13_repo_audit_and_atlas/index.md) ([`ledger.json`](projects/2026-11-13_repo_audit_and_atlas/ledger.json), [`report.html`](projects/2026-11-13_repo_audit_and_atlas/report.html)).

2. **Sound Cues, 2D Sprite D-Pad, `0.33` Release & `EditGuider` (`2026-11-12`)**:
   - Shipped zero-file `MiniMaxH3SoundCues` (`3`-frame Audio VAE grid snap, `-28 dBFS` ambient rumble floor, RoPE overlap merging; [`../../sound_cues.py`](../../sound_cues.py)), `MiniMaxH3AppendSpriteTrajectory` (2D `(x, y, scale, face)` cutout trajectory + hop arcs; [`../../sprite_trajectory.py`](../../sprite_trajectory.py)), and `MiniMaxH3ReleaseReferences` independent `video_end_percent = 0.33` / `audio_end_percent = 0.33..0.60` release (`5407673f`, `1af989e0`, `0fbac0a3`).
   - Fixed `EditGuider.predict_noise` 3D `[B, 1, C_flat]` packed AV latent splitting (`e0a5bb35`) and keyframe-differential `_release_to_original_after` (`d19ba65c`).
   - Shipped [`../../workflows/h3_image_ref_dpad_sound_pika_api.json`](../../workflows/h3_image_ref_dpad_sound_pika_api.json) and [`../../workflows/h3_text_to_video_pika_sound_guide_api.json`](../../workflows/h3_text_to_video_pika_sound_guide_api.json). Walkthrough and limits: [`research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md`](research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md).

3. **Control Vectors, J-Lens / Chain Lens Readouts & `exp5` Ladder (`2026-11-11`..`2026-11-12`)**:
   - Shipped composable `MiniMaxH3Steer`, `MiniMaxH3WordVector`, `MiniMaxH3LoadControlVector`, `MiniMaxH3ReleaseReferences`, field-named section masks, Control Vector Studio (`bench/control_vector/ui_server.py`), and blind readout scoring ([`../../bench/control_vector_readout_score.py`](../../bench/control_vector_readout_score.py)).
   - Settled tool boundary (`fedcdff1`, `exp5` synthesis): Control Vectors are broadcast scene/world/mood dials; `MiniMaxH3EditGuider` handles subtle, clause-bound, continuous `0 < scale < 1`, audio/video split, and contradiction-free look swaps; plain prompt edits stay `1x` default when sufficient.
   - Next CPU & GPU steps (`§15` of [`research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md`](research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md) + [`research/control_vectors/2026-11-12_exp5_ladder_and_temporal_guide_synthesis.md`](research/control_vectors/2026-11-12_exp5_ladder_and_temporal_guide_synthesis.md)) sit ready to run.

4. **`MiniMax-H3-Image`, `P-H3-Image` & Mobile Studio (`2026-11-11`..`2026-11-12`)**:
   - Unified `MiniMax-H3-Image` (`4.0 MP` floor, `MiniMaxH3ImageDownscale`, `PH3_IMAGE_VAE` default, `heylook` SSE prompt expander, split KV-prefix ref cache `MiniMaxH3EncodeImageReferences` / `MiniMaxH3PromptOnImageReferences`, 1-stream `MiniMaxH3LoRABranch`) across [`../../h3_image_nodes.py`](../../h3_image_nodes.py) and `workflows/image/*`. Phone UI lives in `app/` (`app/AGENTS.md` / [`../../app/README.md`](../../app/README.md)).

5. **`comfy-kitchen` & `MiniMaxH3Sol` (`2026-11-11`)**:
   - `SOL_DENSE_TAIL` updated to `"49"` after the blind 6-step Pika Sol panel (`5d293b5f`, [`../../bench/results/2026-11-11_sol_pika_panel.md`](../../bench/results/2026-11-11_sol_pika_panel.md)). [`../../archive/sol_chunked_h3.py`](../../archive/sol_chunked_h3.py) (`MiniMaxH3SolChunked`) was retired `2026-11-13` (unused by any shipped graph, closing the chunked `quantizer` gap); the `comfy-kitchen` wheel still awaits rebuild for the rotated `top-k` fix (`e0b1c0ae`, [`attention/comfy_kitchen.md`](attention/comfy_kitchen.md)).

6. **Masked V2V & `bench/lane_db` (`2026-11-10`, Parked `'fun'`)**:
   - `'fun'` shot-by-shot rendering is **closed/parked**. Reusable perception/mask/plate nodes and SQLite Tier-1 `bench/lane_db/` (`migrations/0001..0003`, `importer.py`, `prove.py`, [`../../bench/check_lane_db.py`](../../bench/check_lane_db.py)) remain intact under [`masked_v2v/2026-11-10_simplification_plan.md`](masked_v2v/2026-11-10_simplification_plan.md).

## Reading Order

1. [`../../AGENTS.md`](../../AGENTS.md) — rules, H3 frame/clock invariants, and 66-node pack replacement map.
2. [`index.md`](index.md) — master LLM Wiki router (`docs/wiki/`).
3. [`projects/2026-11-13_repo_audit_and_atlas/index.md`](projects/2026-11-13_repo_audit_and_atlas/index.md) — full audit of what was built, verified renders, code-vs-prose gaps, and prioritized unfinished ideas.
4. [`next_steps.md`](next_steps.md) & [`decisions.md`](decisions.md) — active next queue and recent decisions.
