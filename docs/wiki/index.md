# H3 LLM Wiki (`docs/wiki/`)

Master router for `ComfyUI-h3-explorations`. Root [`../../AGENTS.md`](../../AGENTS.md) holds the operative rules, pre-render H3 invariants, and the 65-node pack replacement map. Where any wiki page disagrees with the code on disk, the code wins — fix the page and record the correction in [`decisions.md`](decisions.md).

---

## 1. Start Here & Live Repo State

| Page | Role |
| :--- | :--- |
| [`../../AGENTS.md`](../../AGENTS.md) | Mandatory session rules, H3 clock/frame/mask invariants, 65-node map |
| [`projects/2026-11-13_repo_audit_and_atlas/index.md`](projects/2026-11-13_repo_audit_and_atlas/index.md) | **Repo Atlas (`2026-11-13`)**: 8-track status, Top 14 unfinished/forgotten ideas, code-vs-prose ledger, Nov 10–12 renders, offline HTML + `ledger.json` |
| [`projects/index.md`](projects/index.md) | Project directory index & structured dossier standard ([`projects/AGENTS.md`](projects/AGENTS.md)) |
| [`handover.md`](handover.md) | Current cross-track handover state (`2026-11-13`; `'fun'` parked) |
| [`next_steps.md`](next_steps.md) | Active queues by track, ordered next experiments, and pending checks |
| [`decisions.md`](decisions.md) | Dated decisions & reversals (`2026-11-11`..`2026-11-13` active + shards `decisions/*.md`) |
| [`h3_uses.md`](h3_uses.md) | End-to-end modes (`t2va`, `fl2va`, `ref2va`, `P-H3-Image`, `MiniMax-H3-Image`, audio freeze, masked V2V) |
| [`stages.md`](stages.md) | Stage-by-stage code -> owner -> guard map across conditioning, sampling, attention, and decode |
| [`known_before_a_render.md`](known_before_a_render.md) | Facts resolved in code/weights without spending card time |
| [`runbook.md`](runbook.md) | Quick operational checklists before launching a render or server session |
| [`working_with_the_owner.md`](working_with_the_owner.md) | Collaboration, evidence standards, and plain-English writing rules |
| [`../../README.md`](../../README.md) & [`../README.md`](../README.md) | Repository overview & `docs/` LLM-wiki directory map |

---

## 2. Core H3 Video, Conditioning, References & Audio (`core/`)

| Page | Owning Code | What It Owns |
| :--- | :--- | :--- |
| [`core/h3_frame_rules.md`](core/h3_frame_rules.md) | `h3_rules.py`, `loop_plan.py` | Fixed `24 fps` video & `40 Hz` audio clocks (`FRAME_RESCALE = 5/3`), VAE `17n + 5`, dual-clock `39 + 51m` seam grid, `t = 1` exception, keyframe/`3`-frame sound cues |
| [`core/h3_resolutions.md`](core/h3_resolutions.md) | `resolution.py`, `preflight.py` | Trained-family `768` short edge / `1344x768` cap / `32`-px multiples, `PackedLayout` token counts, `99,864` `int32` QKV threshold |
| [`core/h3_geometry_and_nodes.md`](core/h3_geometry_and_nodes.md) | `conditioning.py`, `h3_encoder_loader.py` | `MiniMaxH3Conditioning` geometry fixes, lone `last_frame` crop fix, empty-prompt refusal |
| [`core/h3_conditioning_end_to_end.md`](core/h3_conditioning_end_to_end.md) | `conditioning.py`, `reference_conditioning.py` | End-to-end conditioning trace from Qwen3-VL tokens and VAE latents into `TokenRefiner` and DiT |
| [`core/h3_references.md`](core/h3_references.md) | `reference_conditioning.py`, `reference_encode.py`, `reference_noise.py` | Typed `<Picture N>` / `<Video N>` / `<Audio>` builders, split-cache `MiniMaxH3EncodeReferences`, `2 fps` Qwen view, clean levels |
| [`core/h3_input_impacts.md`](core/h3_input_impacts.md) | `preflight.py`, `h3_rules.py` | Sequence-row cost and VRAM/compute impact of resolution, length, references, and keyframes |
| [`core/h3_distills.md`](core/h3_distills.md) | `lora_branch.py` | Pika H3 Flex 6-step distill vs. base 50-step H3, unmerged `MiniMaxH3LoRABranch` for int8, retired PDD/FlashGen policy |
| [`core/h3_audio_freeze.md`](core/h3_audio_freeze.md) | `audio_freeze.py`, `audio_freeze_song.py`, `audio_refine.py`, `frozen_video_cache.py`, `plate_restore.py` | `NestedTensor(video=1, audio=0)` mask freeze, `MiniMaxH3AudioFreezeSong`, `MiniMaxH3AudioRefineMask`, `MiniMaxH3RestorePlate`, why stock `SetLatentNoiseMask` breaks |
| [`core/custom_node_gaps.md`](core/custom_node_gaps.md) | `nodes.py` | Inventory of ComfyUI core gaps closed by custom pack nodes (`MiniMaxH3EncoderLoader`, `LoraLoader`, `ImageToVideo`, `SetLatentNoiseMask`, `SAM 3`) |

---

## 3. Single-Frame Image Generation, Editing & Phone Studio (`image/`, `app/`)

| Page | Owning Code | What It Owns |
| :--- | :--- | :--- |
| [`image/h3_image.md`](image/h3_image.md) | `h3_image_nodes.py`, `vae_precision.py`, `reference_encode.py` | `MiniMax-H3-Image`: `4.0 MP` floor + `MiniMaxH3ImageDownscale`, `PH3_IMAGE_VAE` default, `fp32` encoder precision, `MiniMaxH3ImagePromptExpander` (`heylook` SSE), 16-still split cache |
| [`image/p_h3_image.md`](image/p_h3_image.md) | `ph3_image_nodes.py` | `P-H3-Image`: `[4 MP, 8 MP]` canvas/`image_1` clamp, `[1 MP, 8 MP]` refs, `seed = 42` latent encoding, 8-step `ManualSigmas`, lazy `Qwen3.5-9B` upsampler |
| [`image/h3_image_editing.md`](image/h3_image_editing.md) | `conditioning.py`, `guide_latent.py`, `edit_guidance.py` | Pika `single_frame = True` (`t = 1`) generation/editing, spatial `mask` guide edits, `EditGuider` & `MiniMaxH3Steer` on stills |
| [`../../app/README.md`](../../app/README.md) | `app/server.py`, `app/routes.py`, `app/h3_image_tools.py` | iPhone Safari Obsidian Glass studio front end (`AGENTS.md`: managed via dedicated studio workflow) |

---

## 4. Steering, Control Vectors, Edit Guidance, Sound Cues & 2D D-Pad

Start at [`control_vectors.md`](control_vectors.md) and [`edit_guidance.md`](edit_guidance.md). Settled `2026-11-12` boundary: Control Vectors are broadcast `1x`-pass scene/world/mood dials; `MiniMaxH3EditGuider` (`2x` / partial `1.5x`) handles subtle, clause-bound, continuous `0 < scale < 1`, audio/video split (`streams`), and keyframe-differential steering; plain prompt rewrites stay first choice for straightforward swaps.

| Page | Topic |
| :--- | :--- |
| [`control_vectors.md`](control_vectors.md) | Unified steering hub: Control Vectors vs. Edit Guidance vs. J-Lens readout, shipped workflows, `exp5` tool-choice update |
| [`edit_guidance.md`](edit_guidance.md) | `MiniMaxH3EditGuider` + `MiniMaxH3PromptEdit` (`streams`, `audio_scale`, `first_step`, `last_step`, 3D `[B, 1, C_flat]` AV split) |
| [`control_vectors/which_tool_and_what_to_call_it.md`](control_vectors/which_tool_and_what_to_call_it.md) | Plain-English guide comparing Prompt Edit, Control Vectors, Edit Guidance, and Word Readout (`J-Lens`) |
| [`control_vectors/how_the_word_readout_works.md`](control_vectors/how_the_word_readout_works.md) | Step-by-step explainer of J-Lens cloze log-odds lexical readouts |
| [`control_vectors/worked_examples.md`](control_vectors/worked_examples.md) | Five common steering questions, exact recipes, and verified runs |
| [`control_vectors/building_pairs_and_vectors.md`](control_vectors/building_pairs_and_vectors.md) | Two-stage H3 text geometry, `diff_span` pooling, `[[hold]]` projections, `from-prompt`, `from-toml`, `make-media` |
| [`control_vectors/comfyui_nodes_and_controls.md`](control_vectors/comfyui_nodes_and_controls.md) | `MiniMaxH3Steer`, `MiniMaxH3LoadControlVector`, `MiniMaxH3WordVector`, `MiniMaxH3ReleaseReferences` (`video_end_percent = 0.33`), field masks |
| [`control_vectors/j_lens_diagnostics_and_word_lab.md`](control_vectors/j_lens_diagnostics_and_word_lab.md) | J-Lens Jacobian + spectral Wiener inverse ($J_{\text{Wiener}}$), `hold_words`, `words` synthesis, blind scoring (`bench/control_vector_readout_score.py`) |
| [`control_vectors/examples_and_recipes.md`](control_vectors/examples_and_recipes.md) | Copy-paste recipes across film grades, weather, claymation, wardrobe, and sound-guided choreography |
| [`control_vectors/preset_catalog_and_cli_reference.md`](control_vectors/preset_catalog_and_cli_reference.md) | `bench/control_vectors.py` CLI subcommands, `.safetensors` contract, prompt-bank catalog |
| [`control_vectors/llm_prompting_guide.md`](control_vectors/llm_prompting_guide.md) | Authoring balanced contrastive `.toml` pairs with single-clause & length discipline |
| [`control_vectors/studio_ui_guide.md`](control_vectors/studio_ui_guide.md) | Control Vector Studio (`bench/control_vectors.py ui`, `ui_server.py`, cookie auth, folder confinement) |
| [`research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md`](research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md) | **Sound Cues + 2D Sprite D-Pad + `0.33` Release Law**: `MiniMaxH3SoundCues`, `MiniMaxH3AppendSpriteTrajectory`, `MiniMaxH3ReleaseReferences`, `h3_image_ref_dpad_sound_pika_api.json` |
| [`research/control_vectors/2026-11-12_exp5_ladder_and_temporal_guide_synthesis.md`](research/control_vectors/2026-11-12_exp5_ladder_and_temporal_guide_synthesis.md) | **`exp2`–`exp5` Forensic Audit & Roadmap**: `heavy_blizzard` / `watercolor_wash` 1:1 ladder, Pole-A shared direction, `EditGuider` spatiotemporal mask proposal |
| [`research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md`](research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md) | **Chain Lens & Readout Plan (§15)**: blind readout score, whitened whole-word table, neighbor holds, upper-layer chain rollout |

---

## 5. Attention, Sparse Routing (`Sol`), `comfy-kitchen` & Quantization (`attention/`)

| Page | Owning Code | What It Owns |
| :--- | :--- | :--- |
| [`attention/SOLATTN.md`](attention/SOLATTN.md) | `sol_attn_h3.py` | `MiniMaxH3Sol` (`SOL_DENSE_TAIL = "49"` default since `2026-11-11`), sink conditioning, capture/DuckDB analysis (`MiniMaxH3SolChunked` archived `2026-11-13`) |
| [`attention/comfy_kitchen.md`](attention/comfy_kitchen.md) | `sol_attn_h3.py` | Upstream `comfy-kitchen` vs. local fork (`#124`, `#242`), rotation/RoPE handoffs, open wheel rebuild & retired `MiniMaxH3SolChunked` notes |
| [`attention/h3_quant_policy.md`](attention/h3_quant_policy.md) | `channel_balance.py`, `exact_blocks.py`, `attention.py` | INT8/FP8 quantization policy, SmoothQuant Q/K channel fold, exact-block bypass |
| [`attention/h3_block49_quant_error.md`](attention/h3_block49_quant_error.md) | `attention.py`, `sol_attn_h3.py` | Block 49 int8 K-outlier mechanics & why rotated attention (`qk_rotate`) / dense terminal block 49 eliminates vertical-stripe artifacts |
| [`attention/sol_upstream.md`](attention/sol_upstream.md) | `sol_attn_h3.py` | Upstream Sol integration notes & kernel alignment |
| [`attention/sol_upstream_qk_balance_pr.md`](attention/sol_upstream_qk_balance_pr.md) | `channel_balance.py` | Upstream Q/K balance PR scope and design |

---

## 6. Prompting, System Prompts & Graded Banks (`prompting/`, `system_prompts/`)

| Page | What It Owns |
| :--- | :--- |
| [`prompting/prompting.md`](prompting/prompting.md) | Master H3 prompting guide & section router (`grade_prompt_text.py`), sharded into [`06`](prompting/06_singing_text_markers_and_audio.md), [`09`](prompting/09_references_ref2va.md), [`10`](prompting/10_worked_examples.md), [`11`](prompting/11_rule_matrix_and_gaps.md), [`14`](prompting/14_sources_and_internals.md), [`16`](prompting/16_specificity.md), [`17_p_h3_image_prompting.md`](prompting/17_p_h3_image_prompting.md) |
| [`prompting/prompt_audit.md`](prompting/prompt_audit.md) | Prompt bank compliance & quality audit against `grade_prompt_text.py` |
| [`prompting/prompt_bank.md`](prompting/prompt_bank.md) & [`prompting/prompt_catalogue.md`](prompting/prompt_catalogue.md) | Generated text prompt corpus & workflow prompt index (`bench/build_prompt_bank.py`, `bench/build_prompt_catalogue.py`) |
| [`system_prompts/README.md`](system_prompts/README.md) | Runtime system prompts (`general_image_edit_system_prompt.md`, control-vector pair prompts, `h3_image/README.md`, `h3_p_h3_image_system_prompt.md`, `h3_system_prompt.md`) |

---

## 7. Masked V2V Pipeline & `lane_db` SQLite Architecture (`masked_v2v/`)

> **Project status (`2026-11-13`):** `'fun'` shot-by-shot rendering is **closed/parked**. Active architectural value in this subtree is the reusable perception/mask/restore node stack, `bench/lane_db/` SQLite record (`migrations/0001..0003`, `importer.py`, `prove.py`), and `2026-11-10_simplification_plan.md`.

| Page | What It Owns |
| :--- | :--- |
| [`masked_v2v.md`](masked_v2v.md) | End-to-end pipeline overview with symbol-anchored code links |
| [`masked_v2v/AGENTS.md`](masked_v2v/AGENTS.md) | Mandatory masked-V2V session rules & invariants (`'fun'` parked) |
| [`masked_v2v/PROTOCOL.md`](masked_v2v/PROTOCOL.md) | Shot table, load layout, delivery/composite rules |
| [`masked_v2v/CODE_MAP_INDEX.md`](masked_v2v/CODE_MAP_INDEX.md) | Generated code-map router (`bench/masked_code_map.py`) -> [`code_map_stages.md`](masked_v2v/code_map_stages.md), [`code_map_contracts.md`](masked_v2v/code_map_contracts.md) |
| [`masked_v2v/2026-11-10_simplification_plan.md`](masked_v2v/2026-11-10_simplification_plan.md) & [`masked_v2v/2026-11-10_lane_record_first_tier.md`](masked_v2v/2026-11-10_lane_record_first_tier.md) | Post-`'fun'` simplification plan & SQLite Tier-1 `lane_db` specification |
| [`masked_v2v/frames_rules.md`](masked_v2v/frames_rules.md), [`masked_v2v/reference_and_prompt_rules.md`](masked_v2v/reference_and_prompt_rules.md) | Whole-shot frame boundaries, dual-clock rules, `<Subject N>` prompting |
| [`masked_v2v/meta_perception_models.md`](masked_v2v/meta_perception_models.md), [`masked_v2v/subject_and_drawing_recipes.md`](masked_v2v/subject_and_drawing_recipes.md) | SAM 3.1, Sapiens2, SAM 3D Body (+ ViT-H), pose-mesh drawing recipes |
| **Detailed forensics, tables & historical track notes (`masked_v2v/`)** | [`masked_v2v_pipeline.md`](masked_v2v/masked_v2v_pipeline.md), [`sam3_prompting.md`](masked_v2v/sam3_prompting.md), [`state_signals.md`](masked_v2v/state_signals.md), [`who_is_who.md`](masked_v2v/who_is_who.md), [`who_is_who_evidence.md`](masked_v2v/who_is_who_evidence.md), [`window_context.md`](masked_v2v/window_context.md), [`window_edges_data.md`](masked_v2v/window_edges_data.md), [`chain_trace_and_remakes.md`](masked_v2v/chain_trace_and_remakes.md), [`delivery_table_and_restore_architecture.md`](masked_v2v/delivery_table_and_restore_architecture.md), [`upstream_anomaly_and_alignment_checks.md`](masked_v2v/upstream_anomaly_and_alignment_checks.md), [`verifying_raw_frames_and_keeps.md`](masked_v2v/verifying_raw_frames_and_keeps.md), [`2026-10-25_pass_contamination_and_mask_step_forensics.md`](masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md), [`2026-10-27_perception_lhf_branch.md`](masked_v2v/2026-10-27_perception_lhf_branch.md), [`2026-10-28_session_handover.md`](masked_v2v/2026-10-28_session_handover.md), [`2026-10-29_findings_0278n_text_and_frame_table.md`](masked_v2v/2026-10-29_findings_0278n_text_and_frame_table.md), [`fun_render_backlog.md`](masked_v2v/fun_render_backlog.md) *(parked)* |

---

## 8. Operations, Checks, Evidence & Hardware (`ops/`)

| Page | What It Owns |
| :--- | :--- |
| [`ops/comfy_notes.md`](ops/comfy_notes.md) | ComfyUI `:8188` server flags, `$H3_*` discovery, `.venv/bin/python`, `bootstrap`, worktree conventions |
| [`ops/checks.md`](ops/checks.md) | Check standard & sharded inventory ([`standard_and_gaps.md`](ops/checks/standard_and_gaps.md), [`doc_link_absent_history.md`](ops/checks/doc_link_absent_history.md), [`index_01.md`](ops/checks/index_01.md)..[`index_15.md`](ops/checks/index_15.md)) |
| [`ops/evidence.md`](ops/evidence.md) & [`ops/prose_measurements.md`](ops/prose_measurements.md) | Evidence standards, retracted-claim allowlist, prose measurement audit rules |
| [`ops/eval_comparison.md`](ops/eval_comparison.md) | Paired evaluation discipline, blind judges, control arms, audio+video checks |
| [`ops/capture_manifest_schema.md`](ops/capture_manifest_schema.md) & [`ops/pipeline_telemetry.md`](ops/pipeline_telemetry.md) | Provenance capture schema & Sol/sampler step telemetry |
| [`ops/hardware.md`](ops/hardware.md) | RTX 4090 `sm89` 24 GB VRAM/compute budgets and host RAM constraints |
| [`ops/roadmap.md`](ops/roadmap.md) & [`ops/open_experiments.md`](ops/open_experiments.md) | Research roadmap (closed PDD/FlashGen lanes) and unmeasured experiment ledger |

---

## 9. Generated Inventories, Research, References & Archive

- **Generated repo indexes**: [`../../INDEX.md`](../../INDEX.md), [`../../workflows/INDEX.md`](../../workflows/INDEX.md), [`../../bench/INDEX.md`](../../bench/INDEX.md), [`../../bench/results/INDEX.md`](../../bench/results/INDEX.md).
- **Research index**: [`research/INDEX.md`](research/INDEX.md) (all dated studies, Control Vectors, Chain Lens, Masking, PDD, Sparse, VSA, and postmortems).
- **External references**: [`references.md`](references.md) (upstream `sglang`, `FastVideo`, `LightX2V`, `vllm-omni`, `TaoMate-H3`, ComfyUI core comparisons).
- **Archive & history**: [`decisions.md`](decisions.md) (`decisions/*.md` shards) and [`archive/INDEX.md`](archive/INDEX.md) (`37` frozen/superseded files).
