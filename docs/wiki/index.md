# H3 LLM Wiki (`docs/wiki/`)

Master router for `ComfyUI-h3-explorations`. Root [`../../AGENTS.md`](../../AGENTS.md) holds the operative rules, peer collaboration norms, and H3 clock/frame/mask cheat sheet; [`../../nodes.py`](../../nodes.py) holds the grouped pack node list. Where any wiki page disagrees with the code on disk, the code wins — fix the page and record the correction in [`decisions.md`](decisions.md).

---

## 1. Start Here & Live Repo State

| Page | Role |
| :--- | :--- |
| [`README.md`](README.md) | **What Works Right Now (`2026-11-26` Rewritten)**: Structured capability-first tour (`70` nodes, `136` API graphs, `50` vector stems) with tool chooser, Self-Steering, Query/Spatial Routing, `MiniMaxH3DepthBeatMask`, Stage 2 Music-Video / Re-Score, H3 Studio 2.0, & visual proof sheets |
| [`../../AGENTS.md`](../../AGENTS.md) | Mandatory session rules, peer collaboration norms, and H3 clock/frame/mask cheat sheet |
| [`self_steering.md`](self_steering.md) | **Closed-Loop Self-Steering (`2026-11-26`, `MiniMaxH3SelfMaskRouter` + `H3LivePolicy`)**: 4 internal sensors (Step-0 `B35+B40` noun/speaker masks, `audio_transients` smoothed bursts, `0.0 ms` `taeh3` preview thermostat + Step-0 `heylook` Visual Critic) driving `1x SpatialQR`, `2x EditGuider` bridge, `MiniMaxH3Steer`/`ReleaseRefs` governors, & `QueryRouter` firewall clamp |
| [`query_routing.md`](query_routing.md) | **Cross-Modal Query Routing (`2026-11-25`, Wrapped Phase 1–2 + Step 1 Bidirectional)**: Single-pass `1x` `Q_vid` / `Q_aud` / depth routing thesis, `a2v=0.0` firewall, `86 s` showcase reel, recipes ([`query_routing_recipes.md`](query_routing_recipes.md)), and 10-scene matrix ([`steering/query_routing_test_matrix.md`](steering/query_routing_test_matrix.md)) |
| [`spatial_query_routing.md`](spatial_query_routing.md) | **Spatial Visual Query Routing (`2026-11-26` `v2-slim`)**: Single-pass `1x` split-screen, `4 MP` `MiniMax-H3-Image` stills, `*_shield` + Dynamic Step-1 Auto-Shield, `17*m` hard cuts, L→R wipes, `MoGe-2` Depth+Beat portal tears (`MiniMaxH3DepthBeatMask`), `2`-photo `ref2va` splits, and 4 audio zones |
| [`h3_steering_and_architecture.md`](h3_steering_and_architecture.md) | **Canonical Master Guide (`2026-11-22`)**: One consolidated plain-English + technical guide covering Subspace Erasure math (`PCRR`, Stage 1 `ROS`/smooth knee), H3's 50-block AV brain (`blocks.20..29`, Stage 2 `1x Q_aud`), Zero-Rewrite HUD Portals, 4-tool choice, 5 falsification proofs, `4`-bus bank design, and honest boundaries (with 4 embedded SVGs) |
| [`projects/2026-11-13_repo_audit_and_atlas/index.md`](projects/2026-11-13_repo_audit_and_atlas/index.md) | **Repo Atlas (`2026-11-13`)**: 8-track status, Top 14 unfinished/forgotten ideas, code-vs-prose ledger, Nov 10–12 renders, offline HTML + `ledger.json` |
| [`research/2026-11-14_cross_pillar_api_workflows_and_sam31_decoupling.md`](research/2026-11-14_cross_pillar_api_workflows_and_sam31_decoupling.md) | **Cross-Pillar API Unlocks & SAM 3.1 Decoupling (`2026-11-14`)**: Two-phase `:8188` perception+generation recipes, Phase 1 + `EditGuider` + `SoundCues` + `SpriteTrajectory` compound workflows, and clean extraction off `MiniMaxH3AudioFreezeSong` |
| [`projects/index.md`](projects/index.md) | Project directory index & structured dossier standard ([`projects/AGENTS.md`](projects/AGENTS.md)) |
| [`handover.md`](handover.md) | Current cross-track handover state (`2026-11-26`: Closed-Loop Rescues + `heylook` Step-0 Critic, `MiniMaxH3DepthBeatMask`, `2`-photo `ref2va` split, Studio 2.0 `app/` `edb13a9`, `70` nodes / `136` API graphs / `50` active bank stems) |
| [`next_steps.md`](next_steps.md) | Active queues by track, ordered next experiments, and pending checks |
| [`decisions.md`](decisions.md) | Dated decisions & reversals (`2026-11-11`..`2026-11-26` active + shards `decisions/*.md`) |
| [`h3_uses.md`](h3_uses.md) | End-to-end modes (`t2va`, `fl2va`, `ref2va`, `P-H3-Image`, `MiniMax-H3-Image`, audio freeze, masked V2V) |
| [`stages.md`](stages.md) | Stage-by-stage code -> owner -> guard map across conditioning, sampling, attention, and decode |
| [`known_before_a_render.md`](known_before_a_render.md) | Facts resolved in code/weights without spending card time |
| [`runbook.md`](runbook.md) | Quick operational checklists before launching a render or server session |
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
| [`image/workflows_matrix.md`](image/workflows_matrix.md) | `bench/build_index.py` (reads `workflows/image/`) | Auto-generated variant architecture, data flow, and per-graph defaults matrix across all tracked image workflows |
| [`research/image/2026-11-14_embedded_image_in_video_plan.md`](research/image/2026-11-14_embedded_image_in_video_plan.md) | `control_vector_nodes.py`, `edit_guidance.py`, `video_mask.py`, `audio_freeze.py` | Phases 0–4 plan for embedded image gen/edit inside video pipelines: dual image/video VAE split, `guide_end_percent` mid-clip release, diff auto-masking, and anti-burnout loop hooks |
| [`../../app/README.md`](../../app/README.md) | `app/server.py`, `app/cv_lab.py`, `app/recipes.py`, `app/graph_builder.py`, `app/heylook_bridge.py` | H3 Studio 2.0 ([`app/VISION.md`](../../app/VISION.md), [`app/DESIGN.md`](../../app/DESIGN.md)): React SPA + `/lab`, `1x` Cross-Modal Query Router (`Instant Audio Twists` / Genre Roulette / Heylook Smart Twists), collapsible `Re-Dub Audio` `Isolate Visuals (a2v=0.0)` drawer, 4-Bus mixer, `< 3 ms` Vibe Search, Server-SQLite Composer Document, SSRF-locked `heylook` |

---

## 4. Prompt Expansion, Query Routing, Steering, Control Vectors, Edit Guidance, Sound Cues & 2D D-Pad

Start at **[`query_routing.md`](query_routing.md)** (Cross-Modal Query Routing — `MiniMaxH3QueryRouter` design & `2026-11-25` wrapped findings), **[`h3_steering_and_architecture.md`](h3_steering_and_architecture.md)** (plain-English & technical steering overview), [`steering.md`](steering.md) (workflow & empirical rules hub), and [`edit_guidance.md`](edit_guidance.md). Settled `2026-11-22` boundary: [`MiniMaxH3PromptExpander`](../../h3_prompt_expander.py) (`1x` `heylook` + `@extend` overlays + `bench/sync_heylook_presets.py` + `ref_end_percent` slot 4 + [`h3_prompt_contract.py`](../../h3_prompt_contract.py), `1152x768 @ 192f` default) handles fresh scenes, mutually exclusive lighting, wardrobe, and multi-shot dialogue; `MiniMaxH3Steer` (`token_focus = 1.0`, `--families fl2va` or `--families h3_image`, `add_norm_matched_erasure` default, Stage 1 `peer_ros=True` + smooth norm knee `95.5 -> 98.5`, Stage 2 `block_band` + `split_backend="auto_1x_kv"`, active `4`-bus bank stems) handles orthogonal film/medium/material/timbre dials; `MiniMaxH3AutoEditPrompt` + `MiniMaxH3EditGuider` (`2x`) handles `lock_take`, `audio_only`, continuous `0 < scale < 1`, and `[F, H, W]` spatial masks; the Jacobian-lens word readout (`j_lens.py`) is an advisory `< 15 ms` CPU eyeball check.

| Page | Topic |
| :--- | :--- |
| [`self_steering.md`](self_steering.md) | **Closed-Loop Self-Steering (`2026-11-26`)**: `MiniMaxH3SelfMaskRouter` + `H3LivePolicy` 4-sensor architecture (`text_phrase`, speaker spotlight, `audio_transients`, `taeh3` preview thermostat), `tb1..tb3` proofs, `15`-input reference, and calibrated limits |
| [`query_routing.md`](query_routing.md) | **Cross-Modal Query Routing (`2026-11-25` Wrapped)**: Single-pass `1x` `Q_vid` / `Q_aud` / block-depth routing thesis, `MiniMaxH3QueryRouter`, Phase 1–2 + Step 1 Bidirectional + `a2v=0.0` firewall findings, and `86 s` showcase reel |
| [`query_routing_recipes.md`](query_routing_recipes.md) | **Query Routing Recipes & Exact Prompt Diffs**: Reproducible setup for `1x` `MiniMaxH3QueryRouter` (`t2va`, [`fl2va`](../../workflows/h3_first_last_frame_to_video_pika_query_router_api.json), split-cached [`ref2va`](../../workflows/h3_image_ref_plus_text_to_video_pika_query_router_api.json)), `AudioRefineMask(isolate_visuals=True)`, `86 s` showcase reel chapters, and verified same-seed plain-vs-routed prompt diffs |
| [`spatial_query_routing.md`](spatial_query_routing.md) | **Spatial Visual Query Routing (`MiniMaxH3SpatialQueryRouter`)**: Plain-English explainer, visual diagrams & proofs, 5 copy-paste recipes (Left/Right split, Portal Box, Sky/Ground split, Custom `MASK`, Audio Zone), and standalone [`h3_text_to_video_pika_spatial_query_router_api.json`](../../workflows/h3_text_to_video_pika_spatial_query_router_api.json) workflow |
| [`steering/query_routing_test_matrix.md`](steering/query_routing_test_matrix.md) | **Query Routing Test Matrix (Sharded `2026-11-25`)**: "Unmistakable in 1 Second" `V1..V7` / `A1..A7` concepts, 10-scene generalization matrix, and lab prompts |
| [`h3_steering_and_architecture.md`](h3_steering_and_architecture.md) | **Canonical Master Guide**: Plain-English + visual overview (`assets/h3_multivector_erasure_explainer.svg`), PCRR + Stage 1/2 formulas, 50-block AV brain, Zero-Rewrite HUD Portals, 4-tool choice, 5 falsification proofs, `4`-bus bank design, and Track 2 horizon |
| [`steering.md`](steering.md) | Unified steering hub (`<= 45 KB`): `MiniMaxH3PromptExpander` (`1x`) vs. orthogonal Control Vectors (`MiniMaxH3Steer`) vs. `2x` `MiniMaxH3AutoEditPrompt` + `MiniMaxH3EditGuider` vs. word readout, shipped workflows ([`steering/shipped_workflows.md`](steering/shipped_workflows.md)), empirical rules |
| [`steering/shipped_workflows.md`](steering/shipped_workflows.md) | **Shipped Steering Workflows (Sharded `2026-11-25`)**: Canonical API workflow table across Control Vectors, Edit Guidance, Query Routing, and Spatial Director |
| [`edit_guidance.md`](edit_guidance.md) | `MiniMaxH3PromptExpander` (`1x` authoring) vs. `MiniMaxH3AutoEditPrompt` + `MiniMaxH3EditGuider` (`2x` surgical edits), `MiniMaxH3PromptEdit`, `MiniMaxH3SpatialDirector`, `MiniMaxH3DirectorHUD`, Spatial Director Studio (`director/`): `combined`, SAM 3.1 `/api/segment`, `streams`, `audio_scale`, `.mask` `[F, H, W]` + 40 Hz audio gating, `edit_pass`, `/api/redub` |
| [`spatial_director_guide.md`](spatial_director_guide.md) | **Spatial Director User Guide & 6-Level Tutorial**: How the studio works end to end, what it can vs. cannot do, mouse/hotkey cheatsheet, `Preview in Circle`, `Target Entity`, `Set Path`, `Roll Forward`, and `Re-Dub Audio` |
| [`research/2026-11-15_spatial_world_director_and_3d_edit_guidance.md`](research/2026-11-15_spatial_world_director_and_3d_edit_guidance.md) | **3D Semantic Velocity Steering & Spatial World Director (`2026-11-15`)**: Native ComfyUI core comparison, `[F, H, W]` causal pooling + `40 Hz` audio gating, `MiniMaxH3SpatialDirector` / `MiniMaxH3DirectorHUD`, and `141`-frame showcase benchmarks |
| [`steering/which_tool_and_what_to_call_it.md`](steering/which_tool_and_what_to_call_it.md) | Tool comparison matrix & public naming rules for the Jacobian-lens word readout |
| [`steering/tutorials_progressive_guide.md`](steering/tutorials_progressive_guide.md) | **4-Lab Progressive Guide**: Multi-vector retrieval intuition -> `1x` H3 DiT steering across 4 cumulative hands-on labs |
| [`steering/how_pole_a_erasure_works.md`](steering/how_pole_a_erasure_works.md) | Deep-dive & historical `2026-11-21` showcase companion to [`h3_steering_and_architecture.md`](h3_steering_and_architecture.md) (detailed TV Cooking Show, Disco Wipe, `fl2va`/`ref2va` boundary transcripts) |
| [`steering/sound_cues_and_timed_masks.md`](steering/sound_cues_and_timed_masks.md) | Plain-English explainer: Sound Cues (`MiniMaxH3SoundCues`), timed animated mask "blooms" (`MiniMaxH3EditGuider.mask`), and what the tortoise, elevator (`1x`), & forge experiments proved |
| [`steering/how_the_word_readout_works.md`](steering/how_the_word_readout_works.md) | Step-by-step explainer of J-Lens cloze log-odds lexical readouts |
| [`archive/2026-11-12_control_vector_worked_examples.md`](archive/2026-11-12_control_vector_worked_examples.md) | *(Archived `2026-11-21`)* Early common steering questions, recipes, and verified runs |
| [`steering/building_pairs_and_vectors.md`](steering/building_pairs_and_vectors.md) | Two-stage H3 text geometry, `diff_span` pooling, `[[hold]]` projections, `from-prompt`, `from-toml`, `--families h3_image`, `make-media` |
| [`steering/comfyui_nodes_and_controls.md`](steering/comfyui_nodes_and_controls.md) | `MiniMaxH3Steer` (`edit_pass` + `h3_image` fingerprint parity), `MiniMaxH3LoadControlVector`, `MiniMaxH3WordVector`, `MiniMaxH3ReleaseReferences` (`video_end_percent = 0.33`, `guide_end_percent`), field masks |
| [`steering/j_lens_diagnostics_and_word_lab.md`](steering/j_lens_diagnostics_and_word_lab.md) | J-Lens Jacobian + spectral Wiener inverse ($J_{\text{Wiener}}$), `hold_words`, `words` synthesis, blind scoring (`bench/control_vector_readout_score.py`) |
| [`steering/examples_and_recipes.md`](steering/examples_and_recipes.md) | Copy-paste recipes across film grades, weather, claymation, wardrobe, native `MiniMax-H3-Image` steering, and `1x` / `2x` sound-guided choreography |
| [`steering/preset_catalog_and_cli_reference.md`](steering/preset_catalog_and_cli_reference.md) | `bench/control_vectors.py` CLI subcommands (`--families h3_image`, `1D_E%`, `--from-concept/--repair`), `.safetensors` contract, prompt-bank catalog |
| [`steering/llm_prompting_guide.md`](steering/llm_prompting_guide.md) | Authoring balanced contrastive `.toml` pairs with single-clause & length discipline |
| [`steering/studio_ui_guide.md`](steering/studio_ui_guide.md) | H3 Studio 2.0 (`app/` `/` + `/lab`, `app/.git` `edb13a9`: `1x` Query Router Twist Bar, World Split / `17*m` Beat Cuts / L→R Wipe / full `SelfMask` + `thermostat_only`, `Scout 4x` gated by `has_manual_sigmas`, `Freeze Audio`, SVG `.live-steer-reticle`, `Re-Dub Isolate Visuals`, 4-Bus Mixer, `< 3 ms` Vibe Search) & `director/` |
| [`steering/heylook_auto_edit_guidance_design.md`](steering/heylook_auto_edit_guidance_design.md) | Shipped `heylook` architecture: `1x` `MiniMaxH3PromptExpander` (`ref_end_percent`, `@extend` overlays, `bench/sync_heylook_presets.py`) / `MiniMaxH3ImagePromptExpander` + `h3_prompt_contract.py` + `data/edit_concepts/` and `2x` `MiniMaxH3AutoEditPrompt` (plain Markdown `section_edits` + lazy `MiniMaxH3EditGuider`) |
| [`steering/research/2026-11-12_sound_cues_dpad_and_033_release_law.md`](steering/research/2026-11-12_sound_cues_dpad_and_033_release_law.md) | **Sound Cues + 2D Sprite D-Pad + `0.33` Release Law**: `MiniMaxH3SoundCues`, `MiniMaxH3AppendSpriteTrajectory`, `MiniMaxH3ReleaseReferences`, `h3_image_ref_dpad_sound_pika_api.json` |
| [`steering/research/2026-11-12_exp5_ladder_and_temporal_guide_synthesis.md`](steering/research/2026-11-12_exp5_ladder_and_temporal_guide_synthesis.md) | **`exp2`–`exp5` Forensic Audit & Roadmap**: `heavy_blizzard` / `watercolor_wash` 1:1 ladder, Pole-A shared direction, `EditGuider` spatiotemporal mask proposal |
| [`steering/research/2026-11-12_chain_lens_digest_and_experiment_plan.md`](steering/research/2026-11-12_chain_lens_digest_and_experiment_plan.md) | **Chain Lens & Readout Plan (§15)**: blind readout score, whitened whole-word table, neighbor holds, upper-layer chain rollout |
| [`steering/research/2026-11-20_unpooled_multivector_sliders.md`](steering/research/2026-11-20_unpooled_multivector_sliders.md) | **Clamped Pole-A Subspace Erasure & Unpooled `1x` Sliders (`2026-11-20`)**: Peer-verified `add_norm_matched_erasure` (`193.5 KB` `float32`, row-softmax `tau=0.15` `mu_d` + rank-8 `V_r` / rank-16 composed), `5/5` multi-seed, `192f` AV (`fl2va` + `ref2va` `<Picture 1>`), `P-H3-Image` `4.1 MP`, and `2026-11-21` showcases |
| [`steering/research/2026-11-20_multivector_retrieval_sota_to_dit_steering.md`](steering/research/2026-11-20_multivector_retrieval_sota_to_dit_steering.md) | **Oct–Nov 2026 Multi-Vector Retrieval SOTA -> H3 `5376`-D Steering**: Minish Lab (`reach`/`Model2Vec`/`POTION`), LightOn (`PyLate`/`LateOn`) + Clavié Ward pooling, Khattab (`ColBERTv2`/`PLAID`/`WARP_SELECT`), Mixedbread (*Latent Terms* + `mxbai-edge-colbert-v0`), Google `MUVERA`, Balanced Sinkhorn OT vs. row-softmax `tau=0.15`, and [`bench/verify_pole_a_erasure.py`](../../bench/verify_pole_a_erasure.py) |
| [`steering/research/2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md`](steering/research/2026-11-21_from_late_interaction_retrieval_to_av_dit_steering.md) | **Research Monograph (v4) & Reproduction Map ([`reproducing`](steering/research/2026-11-21_reproducing_late_interaction_to_dit_steering.md))**: *Breaking the Frozen-Anchor Steering Paradox: From Late-Interaction Multi-Vector Retrieval to Parametric Subspace EQ in Diffusion and Input-Injected Iterative Transformers* (Theorems 1–2, `38`-bank census, `50`-block DiT + causal `blocks.20..49` K/V proof, Dual-Lock `fl2va` law, `YIN` $F_0$ probes, 50-step Base H3; [`2026-11-21_multivector_erasure_benchmark.md`](../../bench/results/2026-11-21_multivector_erasure_benchmark.md), [`2026-11-21_phase3_de_slop_and_vision_block_steering.md`](../../bench/results/2026-11-21_phase3_de_slop_and_vision_block_steering.md)) |
| [`steering/research/2026-11-22_track1_track2_multivector_erasure_blueprint.md`](steering/research/2026-11-22_track1_track2_multivector_erasure_blueprint.md) | **Stage 1/2 Verification, Track 1/2 Blueprint & Prior-Art Audit (`2026-11-22`)**: Multi-vector stacking equilibrium (`ROS` + smooth knee `dd45600e`), unified compiler-safe `block_band` + `1x Q_aud` hooks (`340e8ded`, [`2026-11-22_stage1_stage2_ros_norm_cap_and_unified_steer_hooks.md`](../../bench/results/2026-11-22_stage1_stage2_ros_norm_cap_and_unified_steer_hooks.md)), Track 2 `ColQwen` `Path 1C` + Bus-Scoped `V_r` + `Path 1A` blueprint, Promotion Spec ([`promotion_spec`](steering/research/2026-11-22_paper_to_codebase_promotion_spec.md)), and Paper v4 Critical Review ([`prior_art_review`](steering/research/2026-11-22_paper_v4_critical_review_prior_art_and_revision_plan.md)) |
| [`steering/paper/README.md`](steering/paper/README.md) | **Paper v5 Home & Reproducible Build Stub**: *Causal 4D Prelude Steering in a Single-Stream Joint Audio-Video Diffusion Transformer* — [`manuscript.md`](steering/paper/manuscript.md), [`main.tex`](steering/paper/main.tex), [`figures/README.md`](steering/paper/figures/README.md), and promoted CPU audit/LaTeX/figure scripts |
| [`steering/h3_cross_domain_visual_guide.html`](steering/h3_cross_domain_visual_guide.html) & [`2026-11-23_cross_domain_llm_dit_vgel_and_strategic_audit.md`](steering/research/2026-11-23_cross_domain_llm_dit_vgel_and_strategic_audit.md) | **Interactive Cross-Domain Visual Field Guide (`2026-11-23`)**: `Qwen3-VL L49` cliff, `50`-block DiT & `1x Q_aud`, LLM KV vs. frozen DiT anchor, interactive `1D`/Regime A/B/C & `de_llm_slop` simulator, `vgel.me` `ReLU` vs. `silu(z)*z` gate, and strategic scorecard |

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
| [`system_prompts/README.md`](system_prompts/README.md) | Runtime system prompts (`bench/sync_heylook_presets.py` & `@extend` overlays; `general_image_edit_system_prompt.md`, control-vector pair prompts, `h3_masked_edit_and_steering_system_prompt.md`, `h3_image/README.md`, `h3_video/README.md`, `h3_director/README.md`, `h3_query_router/README.md`, `h3_p_h3_image_system_prompt.md`, `h3_system_prompt.md`, `h3_auto_edit/` (`fresh_prompt`, `section_edits`, `edit_lines`)) |

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

- **Generated repo indexes**: [`../../INDEX.md`](../../INDEX.md), [`../../workflows/INDEX.md`](../../workflows/INDEX.md), [`image/workflows_matrix.md`](image/workflows_matrix.md), [`../../bench/INDEX.md`](../../bench/INDEX.md), [`../../bench/results/INDEX.md`](../../bench/results/INDEX.md).
- **Research index**: [`research/INDEX.md`](research/INDEX.md) (all dated studies, Control Vectors, Chain Lens, Masking, PDD, Sparse, VSA, and postmortems).
- **Upstream wiki (start here for anything upstream)**: [`upstream/index.md`](upstream/index.md): our trust ranking (sglang, vllm-omni, diffusers; core trusted least; the rest periphery), a page per engine (`upstream/engines/`), [`upstream/periphery.md`](upstream/periphery.md), cross-implementation topic pages (`upstream/topics/`), how a sweep is run (`bench/upstream_watch.py`, [`upstream/watchlist.json`](upstream/watchlist.json)), open upstream work in [`upstream/horizon.md`](upstream/horizon.md), and dated records in `upstream/sweeps/`.
- **External references (history to 2026-11-07)**: [`references.md`](references.md) (upstream `sglang`, `FastVideo`, `LightX2V`, `vllm-omni`, `TaoMate-H3`, ComfyUI core comparisons).
- **Archive & history**: [`decisions.md`](decisions.md) (`decisions/*.md` shards) and [`archive/INDEX.md`](archive/INDEX.md) (`37` frozen/superseded files).
