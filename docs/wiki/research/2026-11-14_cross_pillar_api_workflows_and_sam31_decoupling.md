# Cross-Pillar Code-Driven Workflows, Base-Model Leverage, & SAM 3.1 Decoupling (2026-11-14)

Status: **Architecture & Experiment Synthesis (`308b168f`)** — integrates completed Phase 0B & Phase 1 live `:8188` telemetry ([Phase 0B record](../../../bench/results/2026-11-14_embedded_image_in_video_phase0b.md), [Phase 1 record](../../../bench/results/2026-11-14_embedded_image_in_video_phase1.md)), existing perception decoupling in [`workflows/masked_job_graph.py`](../../../workflows/masked_job_graph.py), parked `EditGuider` masking in [`docs/wiki/next_steps.md`](../next_steps.md), and the decoupling boundary between [`video_mask.py`](../../../video_mask.py) and [`audio_freeze_song.py`](../../../audio_freeze_song.py).

**Core thesis.** Because the repo is run overwhelmingly through Python/API automation on `:8188` rather than interactive ComfyUI UI clicking, the highest-leverage gains come from composing **base models (`MiniMax-H3`, Pika H3 Flex `fl2va`/`ref2va`, `P-H3-Image`, `MiniMax-H3-Image`) + creative inference controls + decoupled perception (`SAM 3.1` / `Sapiens2`)**, while keeping `'fun'` / `'fun2'` untouched on [`MiniMaxH3AudioFreezeSong`](../../../audio_freeze_song.py) and **skipping external task/VFX LoRAs** whose training data and objectives narrow diversity before base inference has been pushed.

---

## 1. The Six Active Pillars & What Phase 1 (`308b168f`) Just Proved

Every capability below already runs in tracked Python code on disk across `64` registered nodes ([`bench/node_id_manifest.json`](../../../bench/node_id_manifest.json)) and `108` workflow graphs ([`docs/wiki/projects/2026-11-13_repo_audit_and_atlas/index.md`](../projects/2026-11-13_repo_audit_and_atlas/index.md)):

| Pillar | Key Modules / Nodes | Verified Capabilities & Measured Constraints |
| :--- | :--- | :--- |
| **1. Zero-Swap Still Engine (`t = 1`)** | [`ph3_image_nodes.py`](../../../ph3_image_nodes.py) (`MiniMaxH3PH3ImageConditioning`, `MiniMaxH3PH3ImagePromptUpsampler`), [`h3_image_nodes.py`](../../../h3_image_nodes.py) (`MiniMaxH3ImageSize`, `MiniMaxH3ImagePromptExpander`, `MiniMaxH3ImageDownscale`), [`reference_encode.py`](../../../reference_encode.py) (`MiniMaxH3EncodeImageReferences`, `MiniMaxH3PromptOnImageReferences`), [`conditioning.py`](../../../conditioning.py) (`single_frame=True`) | Shares one `Qwen3-VL-32B bf16` [`MiniMaxH3EncoderLoader`](../../../h3_encoder_loader.py) and `fl2va` DiT across image + video stages (`P-H3-Image LoRA` `8 -> 4 MP` + 6-step Pika ran at `Swap = 0 B` in Phase 0B & Phase 1). `MiniMax-H3-Image` (`16 -> 4 MP`, 16-still KV prefix cache) sits cleanly as Tier 3 when peak image fidelity justifies a DiT swap. Strict split: `IMAGE_GRAPH_VAE` (`t = 1` only) vs. `MODELS["video_vae"]` ([plan](image/2026-11-14_embedded_image_in_video_plan.md)). |
| **2. Mid-Clip Guide & Bookend Steering** | [`guide_latent.py`](../../../guide_latent.py) (`MiniMaxH3GuideLatent`), [`control_vector_nodes.py`](../../../control_vector_nodes.py) (`MiniMaxH3ReleaseReferences` `guide_end_percent`) | Phase 1 (`308b168f`) verified `guide_end_percent` drops `1008` mid-clip `cond` rows (`46063 -> 45055`) on `141`-frame (`on_both_clocks`) Pika `fl2va` while preserving endpoints and `<Picture N>` refs. **Cutoff law:** `control_vector_nodes.py:1152-1188` (`_step_in_window`) evaluates step `2` at `2 / 6 = 0.3333...`, so **`0.33`** (not `0.34`) cleanly isolates `2/6` steps (`[0, 1]`). Smooth `6/6 -> 3/6 -> 2/6 -> 0/6` `f68 -> B` dose (`MAE 5.03 / 10.54 / 16.68 / 53.31`), zero `67->68->69` boundary pop (`1.29–1.60` adj MAE), and `4.3x` audio lift (`active_frac 0.3712` vs `0.0864`) when prompt soundscape aligns. |
| **3. Contrastive & Timeline Steering** | [`edit_guidance.py`](../../../edit_guidance.py) (`MiniMaxH3EditGuider`, `MiniMaxH3PromptEdit`), [`control_vector_nodes.py`](../../../control_vector_nodes.py) (`MiniMaxH3Steer`, `MiniMaxH3MakeMediaControlVector`), [`j_lens.py`](../../../j_lens.py), [`sound_cues.py`](../../../sound_cues.py) (`MiniMaxH3SoundCues`), [`sprite_trajectory.py`](../../../sprite_trajectory.py) (`MiniMaxH3AppendSpriteTrajectory`) | **Boundary clarified:** `MiniMaxH3EditGuider` (`packed_split`, `streams = "both" \| "video_only" \| "audio_only" \| "independent"`) excels at localized/causal edits & early transition pull (Phase 1 Arm 5 pulled dusk/rain earlier via `_keyframes_extend_prefix` auto-release in `112.08 s`). By contrast, single-pair `MakeMediaControlVector(pooling="scene_text")` produced low SNR (`raw_norm = 5.999 < 14`) against a pinned `first_frame` (Phase 0B Arm 2). `SoundCues` (`3`-frame = `5` `40 Hz` audio tokens, `-28 dBFS` rumble floor) + `EditGuider(streams="video_only")` eliminates `.wav` bleed (`r = -0.0812` vs `+0.8414`, [`2026-11-12_guide_sound_islands_and_kinematic_dpad.md`](2026-11-12_guide_sound_islands_and_kinematic_dpad.md)). `SpriteTrajectory` already accepts optional `mask` + `video_end_percent = 0.33`. |
| **4. Already-Decoupled Perception Stack** | [`sam31_corrections.py`](../../../sam31_corrections.py) (`MiniMaxH3SAM31Corrections`), [`subject_track.py`](../../../subject_track.py) (`MiniMaxH3SubjectTrack`), [`sapiens2_parts.py`](../../../sapiens2_parts.py) (`MiniMaxH3SubjectParts`), [`subject_boxes.py`](../../../subject_boxes.py) (`MiniMaxH3SubjectBoxes`), branch `gem/perception-lhf` (`../../h3_worktrees/gem-perception-lhf`) | Runs completely standalone without H3 DiT or [`MiniMaxH3AudioFreezeSong`](../../../audio_freeze_song.py) ([`workflows/masked_job_graph.py`](../../../workflows/masked_job_graph.py)). `MiniMaxH3SubjectTrack` supports any `subject_phrase`, `seeds`, gradient `cut_scores`, `shot_table`, and `head_mask`. `MiniMaxH3SubjectParts` outputs standard `MASK`s (`parts`, `matte`, zero-prompt `held`, lossless 29-class `classes`). Unmerged `gem/perception-lhf` (`16b0805c`) adds `WeVisOne-VL-4B` seeding (`bench/wevis_seed.py`), `arbitrate_multiplex_masks`, and `OD-1` soft edge alpha ([branch note](../masked_v2v/2026-10-27_perception_lhf_branch.md)). |
| **5. Audio Freeze, Refine & Mask Plumbing** | [`audio_freeze.py`](../../../audio_freeze.py) (`MiniMaxH3FreezeAudio`, `MiniMaxH3FreezeAudioWindow`), [`audio_refine.py`](../../../audio_refine.py) (`MiniMaxH3AudioRefineMask`), [`frozen_video_cache.py`](../../../frozen_video_cache.py) (`MiniMaxH3FrozenVideoCache`), [`plate_restore.py`](../../../plate_restore.py), [`video_mask.py`](../../../video_mask.py), [`audio_freeze_song.py`](../../../audio_freeze_song.py) | `MiniMaxH3FreezeAudio` (`audio_freeze.py:234-260`, `freeze_masks`) **already preserves an incoming nested video `noise_mask`** while setting `audio_mask`. `MiniMaxH3FrozenVideoCache` wins on **AudioRefine** (video `0.0`, only audio+text live) and is **retired for masked video windows** after [`2026-10-06_frozen_cache_masked_window.md`](../../../bench/results/2026-10-06_frozen_cache_masked_window.md) measured `463.8 s` cached vs. `462.9 s` Sol-Attn stock at `23%` live rows. |
| **6. Code/API Orchestration & Guardrails** | [`workflows/build_workflows.py`](../../../workflows/build_workflows.py), [`workflows/h3_config.py`](../../../workflows/h3_config.py), [`bench/preflight_graph.py`](../../../bench/preflight_graph.py), [`provenance.py`](../../../provenance.py) (`MiniMaxH3ProvenanceStamp`) | Programmatic graph assembly/mutation, pre-render `grade()` validation (`0 FAIL` gates), `frame` / `on_both_clocks` / token-budget enforcement, native `.1fps` review inspection, and zero-swap persistent-`client_id` multi-arm batches. |

---

## 2. Critical Architectural Catch: Why SAM 3.1 Is Already Free, While Phase 2B Hits `H3MaskedSource`

Inspecting the actual schema and consumer graph of [`video_mask.py`](../../../video_mask.py) and [`audio_freeze_song.py`](../../../audio_freeze_song.py) resolves both how to leverage SAM 3.1 today and a hidden wiring trap in [`image/2026-11-14_embedded_image_in_video_plan.md`](image/2026-11-14_embedded_image_in_video_plan.md) (**Phase 2B**):

1. **Perception (`MiniMaxH3SAM31Corrections` -> `MiniMaxH3SubjectTrack` -> `MiniMaxH3SubjectParts` / `MiniMaxH3SubjectBoxes`) has zero dependency on [`MiniMaxH3AudioFreezeSong`](../../../audio_freeze_song.py).**
   - Every node in [`workflows/masked_job_graph.py`](../../../workflows/masked_job_graph.py#L275-L356) (`preview(job)`) outputs standard ComfyUI types (`MASK`, `IMAGE`, `STRING` JSON) and runs with **no DiT and no song node**.
2. **`H3MaskedSource` is what actually locks masked v2v to `MiniMaxH3AudioFreezeSong`:**
   - [`H3MaskedSource`](../../../video_mask.py#L248) is a custom dict wrapper returned by [`MiniMaxH3MaskedSource`](../../../video_mask.py#L2330). Across the entire repository, its only consumers are [`audio_freeze_song.py:598`](../../../audio_freeze_song.py#L598) and optional prompt-text assembly in [`masked_prompt.py:89`](../../../masked_prompt.py#L89).
   - Meanwhile, the actual math inside [`video_mask.py`](../../../video_mask.py) is **already cleanly factored into pure functions**: [`motion_reference`](../../../video_mask.py#L1760), [`window`](../../../video_mask.py#L2247) (including `token_mask`, `drop_specks`, `close_lone_holes`, `drop_islands`, `paint_out`, `soften_subject`), and [`lay_window`](../../../video_mask.py#L976) (`changed_alpha` + `unlaid_frames` cut gate + `lay_back`).
3. **Why Phase 2B cannot wire `|edited - orig|` into `MiniMaxH3MaskedSource` on `fl2va`/`EditGuider` as written:**
   - [`image/2026-11-14_embedded_image_in_video_plan.md`](image/2026-11-14_embedded_image_in_video_plan.md) rightly notes that [`MiniMaxH3GuideLatent`](../../../guide_latent.py#L225-L233) `mask` encodes an RGB control keyframe for paired guide+mask LoRAs rather than attaching a `noise_mask`.
   - However, routing `|edited - orig|` into [`MiniMaxH3MaskedSource`](../../../video_mask.py#L2330) forces the graph onto [`MiniMaxH3AudioFreezeSong`](../../../audio_freeze_song.py#L575-L641) — which has **no `first_frame`, no `last_frame`, no external `GuideLatent` / `ReleaseReferences` hook, and hard-codes `Guider_Basic` ([`audio_freeze_song.py:947`](../../../audio_freeze_song.py#L947)) instead of [`MiniMaxH3EditGuider`](../../../edit_guidance.py)**.
   - In addition, thresholding `|edited - orig|` on a single frame gives a **static 2D region**, whereas any moving subject across a `141`-frame scout pass needs a **`[T, H, W]` tracked mask** (`MiniMaxH3SubjectTrack`) — and two clean non-song ways exist to apply it (`MiniMaxH3EditGuider.mask` softly today/Stage 1, or extracting `video_mask.py`'s single-window functions onto standard `LATENT`/`IMAGE` sockets in Stage 2).

---

## 3. Leverage Today Over `:8188` API (Zero Changes to `fun`/`fun2` or Existing Nodes)

### 3.1 Two-Phase Zero-Swap Execution Topology on 24 GB VRAM

Core's `load_models_gpu` automatically evicts the DiT when `SAM 3.1` or `Sapiens2` execute ([`sapiens2_parts.py`](../../../sapiens2_parts.py)). Running perception inline inside an interactive single-clip render loop thrashes DiT <-> SAM/Sapiens across PCIe on every iteration.

Because the workflows are **Python-driven via `/prompt` + `client_id`**, batch perception and generation separately:

```text
PASS 1 — Perception Batch (SAM 3.1 + Sapiens2 hot in VRAM; zero DiT loaded)
  Inputs: generated stills, reference plates, or scout clips
  Graph : Load -> MiniMaxH3SAM31Corrections -> MiniMaxH3SubjectTrack (any subject_phrase)
               -> MiniMaxH3SubjectParts (parts / matte / held / classes) + MiniMaxH3SubjectBoxes
  Output: lossless masks (.mkv via MaskToImage+VHS_VideoCombine or PNG) + shot_table + boxes JSON

PASS 2 — Zero-Swap Generation / Edit Loop (P-H3-Image LoRA + Pika fl2va/ref2va hot in VRAM)
  Loop  : P-H3-Image (8->4 MP) -> SpriteTrajectory(mask=...) / GuideLatent(17m) / ReleaseReferences
               -> EditGuider -> Pika 6-step Sampler (Swap = 0 B across N arms/seeds)

PASS 3 (optional, after batch) — Automated Perception QC Gate
  Graph : Run SubjectTrack(subject_phrase=...) + SubjectBoxes on output MP4s
  Checks: unintended hard cuts (shot_table/cuts), subject drop frames, trajectory deviation
```

### 3.2 Four Immediate API-Driven Recipes (Existing `64` Nodes Only)

1. **Upgrade [`MiniMaxH3AppendSpriteTrajectory`](../../../sprite_trajectory.py#L533-L539) from GrabCut to SAM 3.1 + Sapiens2 `held` cutouts:**
   - `MiniMaxH3AppendSpriteTrajectory` falls back to deterministic border-seeded GrabCut whenever `mask` is unwired. Passing a `[1, H, W]` mask from `MiniMaxH3SubjectTrack` (`subject_phrase="person"` or `"red vintage bicycle"`, `"glass potion bottle"`, etc.), optionally unioned with `MiniMaxH3SubjectParts.held` (auto-detected held props near hands/lips with zero prompt) and refined via `.matte` ([`sapiens2_parts.py:772-783`](../../../sapiens2_parts.py#L772-L783)), eliminates background halo bleed on textured stills before `MiniMaxH3ReleaseReferences(video_end_percent=0.33)` unleashes full `ref2va` articulation.
2. **Programmatic semantic cutouts for `P-H3-Image` / `MiniMax-H3-Image` multi-reference conditioning:**
   - Run `SubjectTrack` -> `SubjectParts` (`upper_clothing`, `lower_clothing`, `hair`, `face_and_neck`, `held`, lossless `classes`) on a source frame/clip, crop and isolate the clean garment/prop onto neutral plates in numpy/PIL inside Python, and supply them as `<Picture 2>`, `<Picture 3>`, ... to [`MiniMaxH3PH3ImageConditioning`](../../../ph3_image_nodes.py#L395) or [`MiniMaxH3EncodeImageReferences`](../../../reference_encode.py#L714). Replaces external outfit/swap LoRAs with base-model multi-ref grounding.
3. **Closed-loop API retry / parameter sweep without human babysitting:**
   - After `Pika` writes candidate clips, queue `MiniMaxH3SubjectTrack` + `MiniMaxH3SubjectBoxes` in a QC sweep: automatically score cut count (`shot_table`), mask continuity (`SMALLEST_MASK_PX = 2048`), and centroid path (`x(t), y(t)` vs. intended keyframes), auto-retrying with adjusted seed, `guide_end_percent` (`0.33` vs `0.50`), or [`MiniMaxH3SoundCues`](../../../sound_cues.py) anchor spacing.
4. **Offline pre-GPU verification stack:**
   - Gate every API graph before `:8188` submission using [`bench/preflight_graph.py`](../../../bench/preflight_graph.py) (`on_both_clocks`, `duration_in_range` `[124..362]`, `<= 99,864` packed tokens) and run CPU `J-Lens` ([`j_lens.py`](../../../j_lens.py), `~15 ms`) whenever testing [`MiniMaxH3MakeControlVector`](../../../control_vector_nodes.py#L2752) / multi-pair vectors.

---

## 4. Cross-Pillar Compound Unlocks (No Opaque Task LoRAs Needed)

Bringing **high-res `t = 1` stills + SAM 3.1/Sapiens2 perception + `0.33` guide/ref release + `SoundCues` + `SpriteTrajectory` + `EditGuider` + `AudioRefine`** together in Python-orchestrated workflows unlocks five high-value capabilities on base/Pika weights alone:

### Unlock A — Directed Multi-Actor Physics & Foley From Any Single Still
- **Pillars:** Tier 1/2/3 Still + `SAM 3.1` (`SubjectTrack` + `SubjectParts.held`) + `SpriteTrajectory` (`<Video 1>`) + `SoundCues` (`cond_audio`) + `ReleaseReferences(video_end_percent=0.33, audio_end_percent=0.60)` + `EditGuider(streams="video_only")`.
- **How it beats motion LoRAs:** Instead of baking a narrow motion pattern into an uninspected external LoRA, you get **exact 2D trajectory (`(x, y, face, scale, hop)`), exact prop preservation (`held` mask attached to subject cutout), and exact `3`-frame (`125 ms`) impact timing (`SoundCues` with `-28 dBFS` rumble floor)** simultaneously—releasing rigid cutouts at step `2` (`0.33`) so base H3/Pika synthesizes natural stride physics and keeping `streams="video_only"` so organic room tone and foley emerge cleanly (`r = -0.0812` `.wav` bleed).
- **Ready extension already queued:** 2-subject chained `<Video 1>` + `<Video 2>` trajectories ([`docs/wiki/next_steps.md`](../next_steps.md), Atlas Item `#3`), directly aided by `gem/perception-lhf`'s `arbitrate_multiplex_masks` for clean multi-object non-overlapping cutouts.

### Unlock B — Soft Regional Evolution (`Scout -> SAM 3.1 Moving Mask -> Parked EditGuider.mask + Released 17m Guide`)
- **Pillars:** `Pika` Scout + `SAM 3.1` `[T, H, W]` `MASK` + `P-H3-Image` `8->4 MP` + `MiniMaxH3GuideLatent(17m, guide_end_percent=0.33..0.50)` + Parked **Path B `MiniMaxH3EditGuider.mask`** ([`docs/wiki/next_steps.md`](../next_steps.md)).
- **Why this cleanly solves Phase 2A + Phase 2B without touching `AudioFreezeSong`:**
  1. Run a 6-step Pika scout clip (`141` frames, `on_both_clocks`).
  2. Edit frame `17m` (`f68`) via `P-H3-Image` (relight, creature/wardrobe metamorphosis, prop arrival), optionally keeping clean outside pixels via dilated `|edited - orig|` (`CHANGE_THRESHOLD = 0.05`).
  3. Run `MiniMaxH3SubjectTrack` on the **scout video** to obtain a frame-accurate moving `[T, H, W]` mask of the target region (`subject`, `held`, or inverted background).
  4. Apply [`EditGuider`](../../../edit_guidance.py) (`v0 + (v1 - v0) * (eff_v * mask_packed)` on `packed_split`, strictly video-only when spatial) alongside `GuideLatent(17m)` + `guide_end_percent = 0.33..0.50`.
- **Payoff vs. hard `noise_mask`:** Preserves original environment motion and allows ambient lighting interaction across the mask boundary without hard `lay_window` seams—at 6-step Pika speed (`~68–112 s`). Open scientific test noted in [`docs/wiki/next_steps.md`](../next_steps.md): measure attention cross-talk outside `mask` on early steps (e.g., blizzard edit with foreground subject masked out).

### Unlock C — Fast Distilled Video + Pixel-Locked Undistilled Cinematic Audio Pass
- **Pillars:** `Pika` 6-step Video (`or SAM-masked video pass`) + `MiniMaxH3AudioRefineMask(video_mask=0.0, audio_mask=1.0)` ([`audio_refine.py:32-69`](../../../audio_refine.py#L32-L69)) + `MiniMaxH3FrozenVideoCache` ([`frozen_video_cache.py:914`](../../../frozen_video_cache.py#L914)) + `MiniMaxH3SoundCues` + pre-distill Base H3 model.
- **Why this avoids the retired trap:** [`2026-10-06_frozen_cache_masked_window.md`](../../../bench/results/2026-10-06_frozen_cache_masked_window.md) proved `MiniMaxH3FrozenVideoCache` loses to Sol-Attn when `23%+` of video tokens are live (`463.8 s` vs `462.9 s`). **When `video_mask = 0.0` (`AudioRefine`), only audio + text tokens are live**, so cached base-model refinement keeps every single generated video pixel bit-identical while recovering undistilled acoustic richness (foley textures, rain ambience, vocal timber) guided by aligned `overall_soundscape` / `SoundCues`!

### Unlock D — Scriptable Narrative Beat Sheets (`Phase 1` + `EncodeReferences` Split KV + Counterfactual A/B)
- **Pillars:** Tier 2/3 Still + `MiniMaxH3EncodeReferences` ([`reference_encode.py:468`](../../../reference_encode.py#L468)) / `MiniMaxH3PromptOnReferences` ([`reference_encode.py:541`](../../../reference_encode.py#L541)) + `MiniMaxH3GuideLatent(17m)` + `guide_end_percent = 0.33..0.50` + programmatic prompt beat alignment.
- **Why API control changes the game:** Phase 1 Arm 4 proved `guide_end_percent = 0.50` + soundscape-aligned prompt delivered a `4.3x` audio jump (`active_frac 0.3712` vs `0.0864`) over generic prompts on identical keyframe latents, while `MiniMaxH3EncodeReferences` caches `<Picture N>` / `<Video N>` Qwen3-VL vision/KV prefixes once. Python scripts can place visual state beats on `17m` boundaries, synchronized sound beats on `3n` frames (`SoundCues`), and iterate prompts/timing sweeps over the cached Qwen3-VL prefix with `0` reference re-encoding.

### Unlock E — Principled Alternatives to Unverified External LoRAs (`Base / Pika` Ladder)
Before training a custom LoRA (or reaching for an external one with opaque training objectives), code-driven H3 offers a strict capability ladder:
1. **Explicit `1x` visual & audio prompting** (still strongest for direct swaps; zero extra DiT cost).
2. **`P-H3-Image` / `MiniMax-H3-Image` `t = 1` anchor generation -> `fl2va` bookends / `17m` released `GuideLatent`** (verified `MAE 5.11` / `28.98 dB` convergence in Phase 0B Arm 1).
3. **`MiniMaxH3EditGuider` (`1.5x..2x` early steps `0..2`, auto-release via `_keyframes_extend_prefix`)** for continuous dose (`0 < scale < 1`), early transition pull (Phase 1 Arm 5), and `video_only` / `independent` stream isolation.
4. **`MiniMaxH3Steer` ONLY with multi-pair datasets (`raw_norm >= 14`)** + rotated [`h3_pair_compiler.py`](../../../h3_pair_compiler.py#L59-L107) (`DEFAULT_AUX_BASELINES`) + CPU `J-Lens` readout ([`j_lens.py`](../../../j_lens.py))—never single-pair `scene_text` against pinned `first_frame` (`raw_norm = 5.999 < 14`).
5. **Custom own-data LoRA (`MiniMaxH3LoRABranch` unmerged at compute dtype)** *only* once Steps 1–4 hit a verifiable distribution gap on the base/Pika model.

---

## 5. Clean Future Extraction: Decoupling Hard `MaskedSource` Off `AudioFreezeSong`

When you are ready **after `'fun2'`** to run hard `noise_mask` video-to-video directly inside modern single-window Pika / `EditGuider` / `GuideLatent` / `SamplerCustomAdvanced` pipelines (fixing the **Phase 2B** bottleneck cleanly without ever breaking `'fun'` or `'fun2'`), keep [`MiniMaxH3MaskedSource`](../../../video_mask.py#L2330) + [`MiniMaxH3AudioFreezeSong`](../../../audio_freeze_song.py#L598) frozen as-is for multi-window song jobs, and expose **three thin adapter nodes** wrapping the three places [`audio_freeze_song.py`](../../../audio_freeze_song.py) calls [`video_mask.py`](../../../video_mask.py):

```text
Existing Pure Helpers in video_mask.py         New Thin Single-Window Nodes           Standard Graph Consumers
──────────────────────────────────────         ──────────────────────────────         ────────────────────────
window_references / motion_reference() ──────► MiniMaxH3MaskedMotionRef  ───────────► MiniMaxH3EncodeReferences /
  (L1760; zoom_plan, widen, wired)             outputs: H3_REFERENCES (<Video N>)     ReleaseReferences (0.33)

video_mask.window() + vae.encode()     ──────► MiniMaxH3MaskedEncode     ───────────► MiniMaxH3FreezeAudio (keeps
  + start_zero_tokens() + sigmas[late:]        outputs: LATENT (NestedTensor +         video noise_mask via L243-260)
  (audio_freeze_song.py:880-959)               nested noise_mask), SIGMAS, META        -> EditGuider -> SamplerCustomAdvanced

video_mask.lay_window() + lay_back()   ──────► MiniMaxH3MaskedComposite  ───────────► VHS_VideoCombine
  (audio_freeze_song.py:979-1003)              outputs: IMAGE, MASK (matte)            (+ SaveImage / QC pass)
```

### Why This Split Works Cleanly
- **Zero duplicate logic:** Every mask-dilation (`grow_by`), `[1, 4, 4, 4, 4]` x `32x32` pooling (`video_mask.py::token_mask`), `others`/`keep` anti-speck rules (`drop_specks`, `close_lone_holes`, `drop_islands`), `paint_out` / softened late starts (`start_knots`), and `COMPOSITE_CHANGED` / `cut_gate` (`lay_window`) rule stays single-sourced in [`video_mask.py`](../../../video_mask.py) (`MiniMaxH3AudioFreezeSong` calls the exact same helpers).
- **Audio freeze ALREADY fits:** [`audio_freeze.py:234-260`](../../../audio_freeze.py#L234-L260) (`freeze_masks`) explicitly validates and preserves an incoming `NestedTensor` video `noise_mask` while overwriting only `audio_mask`. Wiring `MiniMaxH3MaskedEncode -> MiniMaxH3FreezeAudio -> SamplerCustomAdvanced` works natively without modifying [`audio_freeze.py`](../../../audio_freeze.py).
- **Phase 3A anti-burnout loops compose cleanly too:** Running [`MiniMaxH3FreezeAudioWindow`](../../../audio_freeze.py#L368) (`on_both_clocks` grids `141..345`, contexts `39`/`90`) from an API controller lets Python extract tail frames of window $k$, re-edit via `P-H3-Image`, run SAM 3.1 cut/mask checks, and seed window $k+1$ with `GuideLatent(17m)` + `guide_end_percent` without ever bloating [`MiniMaxH3AudioFreezeSong`](../../../audio_freeze_song.py)'s widget list first.

---

## 6. Phased Execution Order (Zero Waste, Zero Regressions)

| Stage | Scope | Code Impact | Immediate Payoff |
|---|---|---|---|
| **Stage 0 (Now)** | Python `:8188` two-phase perception -> generation script | `0` node changes | Batch `SAM 3.1` + `Sapiens2` -> wire `mask`/`held` into `SpriteTrajectory` + `P-H3-Image` multi-refs -> Pass 2 `Swap = 0 B` Pika loops -> automated `SubjectTrack`/`SubjectBoxes` QC gate. Evaluate `gem/perception-lhf` (`WeVisOne-VL-4B`, `arbitrate_multiplex_masks`) offline if needed. |
| **Stage 1 (`S`)** | Implement parked Path B `mask` on [`MiniMaxH3EditGuider`](../../../edit_guidance.py) ([`docs/wiki/next_steps.md`](../next_steps.md)) | `edit_guidance.py` only (~40 LOC; append optional input last) | Connects SAM 3.1 `[T, H, W]` moving masks straight into `fl2va`/Pika/`P-H3-Image` regional `EditGuider` steering—unlocking **Phase 2B** cleanly without `AudioFreezeSong`. |
| **Stage 2 (`M`, post-`fun2`)** | Expose `MiniMaxH3MaskedMotionRef`, `MiniMaxH3MaskedEncode`, `MiniMaxH3MaskedComposite` over [`video_mask.py`](../../../video_mask.py) pure helpers | New thin nodes appended to [`nodes.py`](../../../nodes.py); zero edits to `H3MaskedSource` / `AudioFreezeSong` | Unlocks hard `noise_mask` single-window masked v2v (`paint_out`, `others`/`keep`, `only what changed`) directly on Pika, `EditGuider`, `GuideLatent`, `AudioRefineMask`, and `SamplerCustomAdvanced` while `'fun'`/`'fun2'` remain untouched. |
| **Stage 3 (`M`)** | API-orchestrated chained `MiniMaxH3FreezeAudioWindow` anti-burnout loops (Phase 3A) | Python orchestration + existing `on_both_clocks` nodes | Tail-frame `P-H3-Image` refresh + `SAM 3.1` cut continuity verification across long multi-window renders before considering any permanent song-node schema extensions. |
