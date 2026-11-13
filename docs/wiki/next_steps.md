# Next Steps

**Short router to active queues (`2026-11-13`).** Full code-backed context across all 8 tracks lives in [`projects/2026-11-13_repo_audit_and_atlas/index.md`](projects/2026-11-13_repo_audit_and_atlas/index.md); current state sits in [`handover.md`](handover.md).

## Active Queues by Priority

1. **Steering, `EditGuider`, & Interactive Controls (Low-Hanging / High-Leverage)**:
   - Add per-frame temporal envelopes & spatial token masks to `MiniMaxH3EditGuider` (`~40` lines in [`../../edit_guidance.py`](../../edit_guidance.py); [`research/control_vectors/2026-11-12_exp5_ladder_and_temporal_guide_synthesis.md`](research/control_vectors/2026-11-12_exp5_ladder_and_temporal_guide_synthesis.md)).
   - Build a local drag-and-drop canvas UI over [`../../sprite_trajectory.py`](../../sprite_trajectory.py) + [`../../sound_cues.py`](../../sound_cues.py) (reusing Studio patterns from `bench/control_vector/ui_server.py`) and test 2-subject chained trajectories + SAM 3 cutout masks ([`research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md`](research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md)).
2. **Control Vector Readouts & Clean Pair Hygiene ([`research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md`](research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md) §15)**:
   - **CPU Items 1 & 2**: Build whitened whole-word rows table (`layers.49` / template-lens pooling) + score `project_out` neighbor separation blind with [`../../bench/control_vector_readout_score.py`](../../bench/control_vector_readout_score.py).
   - **Build Hygiene (Item 4 / `exp5`)**: Replace short generic Pole-A prompts (`cos = 0.72` "anti-mundane" shared direction) with length-matched, contradiction-free Pole-A + foley pairs and test holding the batch shared direction out.
   - **GPU Item 5**: Grow short contrastive chains through real `layers.50..63` and verify CPU J-Lens correlation.
3. **`MiniMaxH3SolChunked` & `comfy-kitchen` Wheel Rebuild ([`attention/comfy_kitchen.md`](attention/comfy_kitchen.md))**:
   - Forward `quantizer` in [`../../sol_chunked_h3.py`](../../sol_chunked_h3.py) and rebuild the `comfy-kitchen` wheel so the rotated `top-k` threshold fix (`e0b1c0ae`) runs in installed wheels.
4. **Audio Freeze & Multimodal Reference Release (`0.33`)**:
   - Run the high-signal unrendered `audio_freeze` experiments ([`core/h3_audio_freeze.md`](core/h3_audio_freeze.md): stem freeze + full mix mux, `0.25` loose audio mask, `MiniMaxH3AudioRefineMask` + `MiniMaxH3FrozenVideoCache`), and evaluate multimodal `0.33..0.50` release on grey-plate motion refs ([`research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md`](research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md)).
5. **Codebase Modularization & Masked V2V / `lane_db` (`'fun'` Parked)**:
   - `'fun'` shot-by-shot rendering is **closed/parked**. When ready to resume masked V2V (`'fun2'`), follow [`masked_v2v/2026-11-10_simplification_plan.md`](masked_v2v/2026-11-10_simplification_plan.md) (`lane_db` DuckDB layer + `perception-lhf` A/B).
   - Plan internal module grouping for the 75 root `.py` files and 511 `bench/` scripts in a dedicated refactor lane without touching registered `node_id` schemas ([`projects/2026-11-13_repo_audit_and_atlas/index.md`](projects/2026-11-13_repo_audit_and_atlas/index.md)).

## Historical Queues

- **Archived chronological queues (`2026-09-05` through `2026-10-20`)**: [`archive/2026-10-20_next_steps_history.md`](archive/2026-10-20_next_steps_history.md).
