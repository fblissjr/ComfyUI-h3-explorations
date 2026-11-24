# Cross-Modal Query Routing in MiniMax H3 (`1x` Single-Pass Attention Routing)

Last updated: 2026-11-24.

**Cross-Modal Query Routing** controls which text conditioning and inter-modal key/value weights each query group (`Q_vid`, `Q_aud`, `Q_aud_L/R`, `Q_aud[t0:t1]`) reads per block depth inside MiniMax H3's single-stream `50`-block audio-video DiT ([`steering/hooks.py`](../../steering/hooks.py)).

> **Name disambiguation:** Three other features in this repo use the word "routing". This page owns **Cross-Modal Query Routing** (`Q_vid` / `Q_aud` per-stream and per-block attention routing). It is distinct from:
> 1. `MiniMaxH3Sol`'s sparse attention `token_routing` widget ([`attention/SOLATTN.md`](attention/SOLATTN.md));
> 2. Control-vector `[routing]` metadata tables ([`steering/data.py`](../../steering/data.py)); and
> 3. The **permanently pinned** multivector push-routing lane (`Path 1A/1B/1C`; [`decisions.md`](decisions.md)).

**Companion pages:** [`steering.md`](steering.md) · [`edit_guidance.md`](edit_guidance.md) · [`steering/research/2026-11-24_outside_review_erasure_novelty_evidence_and_commits.md`](steering/research/2026-11-24_outside_review_erasure_novelty_evidence_and_commits.md) · [`steering/paper/manuscript.md`](steering/paper/manuscript.md)

---

## 1. The Fresh Look ("So What?") — Beyond Control Vectors

Control vectors and subspace erasure are always a lossy approximation of a prompt, competing in a crowded activation-steering literature. **Cross-Modal Query Routing** exploits something structurally specific to **single-stream joint audio-video DiTs** like MiniMax H3 (`[text | cond | cond_audio | ref_img | ref_audio | audio | video]` packed into one flat self-attention sequence in `comfy/ldm/minimax/model.py::PackedLayout`), whereas models like `LTX-2` (asymmetric dual-stream) and `Ovi` (twin-backbone) already separate video and audio streams architecturally.

Normally, H3's single shared self-attention sequence is an all-or-nothing handcuff: one prompt drives every block and modality, so restyling the video warps the voice, rewriting the audio moves the actors, and `<Picture 1>` locks the lighting.

**Scoped mixing-console model:** Cross-Modal Query Routing turns a single-pass forward evaluation into a live mixing console across **whole-video queries (`Q_vid`), audio query groups (`Q_aud`, `Q_aud_L/R`, `Q_aud[t0:t1]`), inter-modal attention edges (`Q_aud -> K_vid`, `Q_vid -> K_ref_img`, `Q_vid -> K_aud`), and block depth** — with no second pass (`2x` NFE) and **zero control-vector training required**. **Per-region spatial video masks** stay on two-pass `MiniMaxH3EditGuider` + `MiniMaxH3SpatialDirector` ([`edit_guidance.md`](edit_guidance.md)), because Gate 2C empirically falsified `1x Q_vid[mask]` splicing ([`2026-11-23_wave2_gpu_gates_bus_vr_path1c_and_qvid.md`](../../bench/results/2026-11-23_wave2_gpu_gates_bus_vr_path1c_and_qvid.md)).

### Redefining What a "Concept" Even Is

When a full `32B` Qwen3-VL text encoder sits in front of `PackedLayout`, we should stop pressing ideas into `5376`-D control vectors whenever English already expresses them cleanly:

- **Audio prompts need no control vectors:** A dedicated `Q_aud` prompt (thunderstorm on tin roof, vintage AM radio bandpass, live studio laugh track, underwater hydrophone + sonar) is cleaner, sharper, and zero-maintenance compared with contrastive audio stems.
- **Stop building vague mood/film stems:** Claude's bank cosine audit ([`data/query_routing_2026_11_24/bank_shared_direction.json`](../../data/query_routing_2026_11_24/bank_shared_direction.json)) confirmed that older "high energy" stems (`chaos_panic`, `youtube_overhype_influencer`, `vocal_strain`, `auctioneer_rapid_patter`, `pyro_chain_explosions`) all share one dominant text-conditioning direction, while film-stock stems sit nearly parallel to each other.
- **Keep control vectors only where discrete prompts fall short:** continuous scalar scrubbing (when linear output blending between two routed prompts is not used), sub-symbolic kinematic or two-still visual transfers (`MiniMaxH3MakeMediaControlVector`), and single-prompt moving `2x` spatial portals (`MiniMaxH3EditGuider`).

---

## 2. What Ships Today & Evidence Status

| Component | Status | Owning Code & Current Behavior |
| :--- | :--- | :--- |
| **`1x` Audio/Video Text Split (`auto_1x_kv`) & Single-Band `block_band`** | **Shipped (behind `MiniMaxH3Steer`)** | [`steering/hooks.py::H3SteerRuntime`](../../steering/hooks.py) (`H3SteerStepWorkspace`, `H3SteerBlockPatch`) + [`steering/nodes.py`](../../steering/nodes.py) (`streams="video_only"` / `"audio_only"`, `split_backend="auto_1x_kv"`, `block_band` presets or `A-B` ranges). Compiler-safe across all four attention backends without breaking `malloc_graph`. |
| **Matched audio/video split evidence** | **Measured (single-seed per pair)** | Matched against both-streams steering on four concept-and-scene pairs across two checkpoint families ([`showcase_c_qaud_audio_split.json`](../../data/exp_unpooled_sliders/phase4_2026_11_22/showcase_c_qaud_audio_split.json), [`stage2_verification.json`](../../data/exp_unpooled_sliders/stage2_2026_11_22/stage2_verification.json), [`2026-11-23_wave2_gpu_gates_bus_vr_path1c_and_qvid.md`](../../bench/results/2026-11-23_wave2_gpu_gates_bus_vr_path1c_and_qvid.md), [`wave3b_results.json`](../../data/exp_unpooled_sliders/wave3b_2026_11_23/wave3b_results.json)). Loudness stays closer to the unsteered clip in all four; paired-frame pitch improves on two of the three pitch-measured pairs (global median improves on one) and worsens on one, leaving residual `Q_aud -> K_vid` pitch coupling. |
| **Late-block `audio_only` isolation** | **Measured (single-seed on 2 concepts)** | `streams="audio_only"` + `late_30_49` on `reverb_cathedral` and `youtube_overhype_influencer` changes the picture far less than steering both streams. |
| **Block-depth routing (`mid_late_20_49` / `late_30_49`)** | **Working control; boundaries unestablished** | Works reliably as a visual/identity preservation control across several runs, while the exact `blocks.20..29` boundary comes from a single-seed probe on one retired concept and the `50`-block trace ([`dit_blocks_50_trace.json`](../../data/exp_unpooled_sliders/paper_2026_11_21/dit_blocks_50_trace.json)) covers one prompt across three concepts. |
| **Known current gaps** | **Open** | (1) Zero shipped `workflows/*.json` graphs wire `auto_1x_kv` or `block_band` yet; (2) [`steering/nodes.py`](../../steering/nodes.py) raises `ValueError` if two active steers combine distinct non-`"all"` `block_band` ranges in the same step; (3) [`bench/edit_guidance_arms.py`](../../bench/edit_guidance_arms.py) does not pass stem `[routing]` `streams`/`block_band` yet; (4) multi-seed repeats are pending. |

---

## 3. Next to Build — Standalone `MiniMaxH3QueryRouter` (Zero-Concept Capabilities)

*(Design target — not yet implemented or GPU-verified.)*

Because node IDs and input lists cannot be renamed or inserted mid-list ([`AGENTS.md`](../../AGENTS.md) Rule 5) and dual-prompt routing needs **no control vector at all**, Cross-Modal Query Routing gets its own clean node — **`MiniMaxH3QueryRouter`** — built on top of [`steering/hooks.py::H3SteerRuntime`](../../steering/hooks.py):

- **Planned inputs:** `model`, `conditioning` (Prompt A), optional `routed_conditioning` (Prompt B of **arbitrary token length**), `route_target` (`audio`, `video`, `audio_time_window`, `audio_left`, `audio_right`), `block_band`, `audio_to_video_coupling` (damps `Q_aud -> K_vid` in `blocks.30..49`), `ref_img_late_scale` (scales `ref_img` contribution in `blocks.20..49`), and `t_start` / `t_end`.

### Five Core Zero-Concept Mechanisms to Build & Probe

1. **`1x` Split-Brain Dual-Prompt Routing (`Q_vid -> Prompt A`, `Q_aud -> Prompt B`, arbitrary token lengths):**
   - Lift `H3SteerRuntime`'s equal-`l_txt` in-place `[tx_a:tx_b]` assumption so `Q_aud` (or `Q_vid`) can attend to concatenated `K_b, V_b` from any separate `CONDITIONING` input against shared non-text keys (`refs`, `audio`, `video`).
   - **Full visual rewrite + untouched audio:** rewrite wardrobe, lighting, weather, or medium in rich natural language on `Q_vid` while `Q_aud` reads the untouched original prompt in a single pass. Keep `<d>` spoken lines verbatim across both prompts when on-screen speakers appear so lip motion and dialogue stay aligned.
   - **Impossible Kuleshov cross-modal splits:** pair deadpan calm visuals with an incompatible soundscape (or swap multiple soundtracks onto one locked visual seed) without text-encoder averaging.
2. **`1x` Depth-Layered Zero-LoRA Style & Material Transfer (`blocks.0..19` = Prompt A, `blocks.20..49` = Prompt B):**
   - Feed Prompt A (clean blocking, anatomy, camera, subject count) through `blocks.0..19` and Prompt B (e.g., blueprint ink, porcelain diorama, sunlit underwater caustics, frosted winter) into `blocks.20..49` from step `0` (hitting H3's early-step palette clock while locking spatial geometry in early blocks), while `Q_aud` reads Prompt A.
   - *Open probe question:* determine via a fast single-shot probe whether `blocks.0..19` locks color/texture too strongly before block `20` or cleanly separates layout from medium/lighting without a LoRA or control vector.
3. **`1x` Direct Cross-Modal Attention Mixing (`Q <-> K` edges between latent modalities):**
   - **Sight-to-Sound Coupling Dial (`Q_aud -> K_vid` in `blocks.30..49`):** Leave `blocks.0..29` open for lip-sync and pass a `[1, 1, 1, L]` additive `attn_mask` in [`steering/hooks.py`](../../steering/hooks.py) to damp `K_vid` logits in late blocks (`{1.0, 0.5, 0.25, 0.0}`), mapping the **lip-sync vs. pitch-invariance** trade-off curve directly.
   - **Depth-Gated Dual-Lock Dial (`Q_vid -> K_ref_img` & `Q_aud -> K_ref_audio`):** Keep `ref_img` full in `blocks.0..19` to lock `<Picture 1>` face and pose, and attenuate `ref_img` only in `blocks.20..49` across all attention backends — testing whether depth-gating breaks Dual-Lock more cleanly than the timestep hammer of `MiniMaxH3ReleaseReferences`.
   - **Depth-Targeted Sound-Driven Visuals (`Q_vid -> K_aud`):** Selective `blocks.0..19` vs. `blocks.20..49` `K_aud` gain ([`attention.py::apply_audio_gain`](../../attention.py#L280)) to test beat-locked camera/body motion vs. audio-reactive lighting.
4. **`1x` Temporal & Binaural Stereo Audio Routing (`Q_aud[t0:t1]` and `Q_aud_L` vs `Q_aud_R`):**
   - Built directly on `comfy/ldm/minimax/model.py::{pack_audio, _audio_grid}`, which packs stereo audio channel-major (`[L, R]` on the `40 Hz` clock) and pins Left to `w_low` and Right to `w_high`.
   - Enables **mid-clip acoustic teleports** (`0..t0` dry street -> `t0..T` vacuum, underwater hydrophone, or cathedral organ) and **binaural L/R splits** (different acoustic prompts in Left vs. Right ear).
   - *Open probe question:* verify in fast `141f` probes whether audio-to-audio self-attention preserves crisp time/stereo separation or smears splits.
5. **Simultaneous Multi-Band `1x` Mixing (prereq unlock):**
   - Lift the `len(non_all_bands) > 1` `ValueError` in [`steering/nodes.py`](../../steering/nodes.py) so `video_only` (`mid_late_20_49`) and `audio_only` (`late_30_49`) can run together in one `1x` forward pass using the existing dual-shadow workspace.

---

## 4. Additional Existing-Tool Workflows & Evaluation Practices to Test

Capabilities ready to test using nodes already in the pack:

1. **Ship `1x` Query Routing workflows (`workflows/build_workflows.py`):** ship clean `1x` `video_only`, late-block `audio_only`, and combined `video_only + audio_only` graphs (plus wire `streams` and `block_band` into [`bench/edit_guidance_arms.py`](../../bench/edit_guidance_arms.py) so automated tests match routed configs).
2. **Beat-synced music-video reality portals (`MiniMaxH3FreezeAudio` + `MiniMaxH3SpatialDirector` + `MiniMaxH3Steer`):** lock a real music track while driving moving wipes/portals across the video; because audio is frozen, visual steering runs with zero vocal pitch drift.
3. **Motion transfer + single-prompt moving portal (`MiniMaxH3AppendRefVideo` + `<Picture 1>` + `MiniMaxH3SpatialDirector`):** transfer real choreography onto a reference character and sweep a zero-rewrite portal across the performance.
4. **Fast multi-take audio redub (`MiniMaxH3AudioRefineMask` + `MiniMaxH3FrozenVideoCache`):** freeze video K/V once and audition multiple voice/foley/acoustic takes (`audio_only` steering or rewritten audio prompts) with truly locked pixels — zero shipped workflows use `MiniMaxH3AudioRefineMask` today.
5. **Single-prompt regional mask steering on `MiniMax-H3-Image` (`length = 1`):** combine `MiniMaxH3EditGuider` + spatial mask + `MiniMaxH3Steer` at single-frame speed for rapid multi-seed region checks.
6. **`[Shot N]` shot-scoped text targeting:** [`h3_spans.py`](../../h3_spans.py) already parses `[Shot N]` headers; test whether targeting one shot's rows steers only that shot (and whether `TokenRefiner` self-attention bleeds across cuts).
7. **Forced-choice blind concept identification gate:** adapt [`bench/control_vector_readout_score.py`](../../bench/control_vector_readout_score.py)'s blind packet format to clips — show a blind judge the unsteered and routed clips and ask it to name what changed from a short list. Any concept a viewer cannot identify within one watch is cut.

---

## 5. Fresh From-Scratch Concepts ("Unmistakable in 1 Second", Non-Destructive)

Every test concept below is designed to pass the **Unmistakable in 1 Second** bar (binary visual/audible recognition, zero vague film stocks or "high energy" synonyms) while keeping faces, blocking, camera moves, and speech intact — replacing stale crutches (`noir`, `film_16mm_1974`, `golden_hour_glow`, `knitted_wool_world`, `chaos_panic`, `pyro_klaxon_audio`):

| Concept ID | Stream | Replaces Crutch | 1-Second Binary Visual / Audible Signature | Non-Destructive Invariant |
| :--- | :---: | :--- | :--- | :--- |
| `V1_underwater_caustics` | Visual | murky `underwater_submerged` / `film_16mm` | Crisp bright turquoise sunlight caustics rippling across skin, walls, and props + steady streams of rising silver bubbles | Subjects hold exact pose, wardrobe, and lip-sync — no green fog hiding faces |
| `V2_frost_snowfall` | Visual | `heavy_blizzard` whiteout | Sharp crystalline rime frost coating metal/wood edges, visible cold breath plumes, and clean falling snowflakes against crisp backgrounds | High-clarity lighting; actors keep acting normally in intact blocking |
| `V3_blueprint_ink` | Visual | `charcoal_sketch` / `noir` | Crisp white architectural technical-ink linework and dimension ticks on saturated Prussian-blue drafting paper | Every silhouette, gesture, and architectural edge remains razor-legible |
| `V4_zero_g_props` | Visual | `chaos_panic` / explosions | Tabletop/handheld props (cups, folders, cutlery, loose papers, water droplets) lift gently and tumble mid-air in weightlessness | Actors remain grounded/calm and deliver exact dialogue |
| `V5_neon_wet_midnight` | Visual | `cyberpunk_neon_rain` / `noir` | Daylight flips cleanly to deep cobalt midnight with rain-beaded wet surfaces reflecting crisp magenta and cyan specular strips | Faces stay evenly readable under clean rim/key light; zero neon washout |
| `V6_bioluminescent_moonlight` | Visual | `bioluminescent_glow` bloom | Dark moonlit surfaces laced with sharp glowing cyan-teal bioluminescent veins along foliage, stone joints, or water ripples | Clean dark-to-glow contrast; no blown-out halos on faces |
| `V7_tiltshift_porcelain_diorama` | Visual | `knitted_wool_world` / `claymation` | Razor-narrow shallow-focus miniature tilt-shift plane + glossy hand-painted glazed porcelain texture across set and costumes | Staging, camera path, and proportions stay faithful like a stop-motion miniature |
| `A1_hydrophone_sonar` | Audio | vague "eerie ambience" | Muffled low-pass underwater hydrophone acoustics + crisp metallic active sonar ping every ~2 s + deep bubble rumble | Speech cadence stays rhythmic underneath the underwater filter |
| `A2_am_radio_1930s` | Audio | `cursed_cctv` / generic static | Tight narrowband vintage AM radio bandpass + crackly vinyl/tube hiss + warm mid-range saturation | Every spoken word remains 100% intelligible as an archival radio broadcast |
| `A3_sitcom_laugh_track` | Audio | `youtube_overhype` | Unmistakable 1970s live studio audience burst laughter and warm applause reacting right after deadpan pauses/lines | Original vocal pitch stays dry while audience beds sit around speech |
| `A4_tin_roof_thunderstorm` | Audio | vague "rain ambience" | Heavy percussive rain drumming loudly on a corrugated tin roof + crisp rolling thunderclaps | Strong broadband percussive contrast vs. quiet original room tone |
| `A5_grand_clock_silence` | Audio | generic "tense drone" | Total room-tone strip-out to bone-dry silence punctured only by a loud cavernous mechanical grandfather clock `tick-tock` (`1 Hz`) | Maximum contrast against dense/busy original soundscapes |
| `A6_cathedral_pipe_organ` | Audio | `reverb_cathedral` alone | Long stone nave reverb tail + deep sustained church pipe-organ pedal chord swelling under the scene | Instant spatial expansion without muddying transient articulation |
| `A7_subway_tunnel_pass` | Audio | `pyro_klaxon_audio` | Resonant tiled subway tunnel slapback echo + steel-on-rail express train rumble roaring past and fading | Instant acoustic location teleport for any visual scene |

---

## 6. Curated 10-Scene Diverse Generalization Matrix

Every scene below is adult-only, excludes bank crutches listed in [`steering/constants.py`](../../steering/constants.py) (`COMPILER_EXCLUDE_SHOTS`, `DEFAULT_EXCLUDE_SHOTS`), uses its bank-declared length ([`bench/run_graph_arms.py`](../../bench/run_graph_arms.py)), and keeps spoken `<d>` dialogue verbatim across visual rewrites:

| # | Scene (`prompt_bank/`) | Mode / Grid | Profile | Routing Capability & Playable Variations | Pass / Falsification Signal |
| :-: | :--- | :---: | :--- | :--- | :--- |
| **1** | [`t2va_pika_balcony_orbit.txt`](../../prompt_bank/t2va_pika_balcony_orbit.txt) | `fl2va` · `328f` (standalone `17n+5`, 1 shot) | Sparse · Bright pastel daylight · 3 silent people | **Cap 1a — `1x` Dual-Prompt (1 Picture, 3 Soundtracks):** Keep the sweeping balcony orbit pixel-locked on `Q_vid`; swap `Q_aud` to `A1_hydrophone_sonar`, `A7_subway_tunnel_pass`, or `A3_sitcom_laugh_track` + `A5_grand_clock_silence` timed to the pedestal drops. | All three renders hold baseline balcony geometry/camera path while each soundtrack is unmistakable within one second and free of visual underwater/subway bleed. |
| **2** | [`t2va_delta_blues_porch.txt`](../../prompt_bank/t2va_delta_blues_porch.txt) | `fl2va` · `345f` (`on_both_clocks`, 3 shots) | Medium · Warm sunset · 1 sung (`EN`) + resonator guitar | **Cap 1b — `1x` Dual-Prompt (Full Visual Rewrite + Untouched Audio vs. `2x` `EditGuider`):** Keep untouched acoustic blues + baritone vocal on `Q_aud`; rewrite `Q_vid` to `V2_frost_snowfall` (alpine frosted porch + breath plumes) or `V3_blueprint_ink` (Prussian-blue architectural ink blueprint). | Frame 1 flips cleanly to winter/blueprint across all three shots with tight lip-sync while keeping guitar groove and baritone F0 intact at single-pass speed. |
| **3** | [`t2va_cooking_show_meltdown.txt`](../../prompt_bank/t2va_cooking_show_meltdown.txt) | `fl2va` · `345f` (3 shots) | Dense · Bright TV studio · 1 expressive speaker (`EN`) | **Cap 2a — `1x` Sight-to-Sound Coupling Dial (`Q_aud -> K_vid` in `blocks.30..49`):** Keep Shot 2's chaotic beetroot blender eruption on `Q_vid`; route `Q_aud` to deadpan whisper-calm 1930s AM radio (`A2_am_radio_1930s` + `A5_grand_clock_silence`), sweeping `K_vid` gain `{1.0, 0.5, 0.25, 0.0}`. | Reveals a clear Pareto curve: mid damping suppresses Shot 2 vocal panic/pitch spikes and blender screech while keeping close-up lip-sync; `0.0` shows where lip-sync decouples. |
| **4** | [`t2va_police_interrogation.txt`](../../prompt_bank/t2va_police_interrogation.txt) | `fl2va` · `345f` (3 shots) | Sparse · Dim lamp · 2 speakers (`EN` baritone + tenor) | **Cap 2b — `1x` Coupling Dial (2-Speaker Lab):** Keep bone-dry steel interrogation acoustics on `Q_aud`; flip `Q_vid` to `V4_zero_g_props` (floating folders/ashtray/cuff chain) or `V1_underwater_caustics` (submerged room + bubbles) across `{1.0, 0.5, 0.25, 0.0}`. | Tests whether `blocks.30..49` damping strips visual underwater muffling/bubbles out of both `(S1)` baritone and `(S2)` tenor voices across cuts without losing sync. |
| **5** | [`t2va_orbital_airlock.txt`](../../prompt_bank/t2va_orbital_airlock.txt) | `fl2va` · `345f` (3 shots) | Dense · Dim strobes · 2 speakers (`EN`) | **Cap 3a — `1x` Temporal Audio Routing (`Q_aud[t0:t1]`):** Route `0..9.0s` to loud alarms/air rush and `9.0..14.38s` (`f216` blast-door lock) to dead helmet-vacuum isolation + heartbeat; or dry cabin (`0..5.0s`) -> `A6_cathedral_pipe_organ` bloom (`5.0..14.38s`, `f120`). | Crisp audible drop in loudness and brightness right at `t0` with zero visual glitch or stutter across the three shots. |
| **6** | [`t2va_samurai_bamboo_duel.txt`](../../prompt_bank/t2va_samurai_bamboo_duel.txt) | `fl2va` · `345f` (3 shots) | Medium · Cool mist · 2 speakers (`JA`, Left vs. Right) | **Cap 3b — `1x` Binaural Stereo L/R Routing (`Q_aud_L` `w_low` vs. `Q_aud_R` `w_high`):** Left ear = bone-dry close-mic wood knocks + dry `(S1)` + `A5_grand_clock_silence`; Right ear = `A4_tin_roof_thunderstorm` + echoing `(S2)` (or live reality Left vs. `A2_am_radio_1930s` Right). | On headphones, Left and Right channels separate unmistakably within one second (dry/close Left vs. storm or AM-radio Right) without stereo flanging or mono collapse. |
| **7** | [`ref2va_pika_motion_transfer.txt`](../../prompt_bank/ref2va_pika_motion_transfer.txt) | `ref2va` · `243f` (`on_both_clocks`, 1 shot) | Dense flat pop-art · 1 silent dancer · 3 verified donor files in `$H3_INPUT_DIR` | **Cap 4a — `1x` Depth-Gated Dual-Lock (`ref_img` `1.0` in `0..19`, swept `{1.0, 0.5, 0.25, 0.0}` in `20..49` vs. `ReleaseReferences(0.33)`):** Uses `$H3_INPUT_DIR/motion-transfer-input-{1.mp4,2.png,3.png}`. Flip flat risograph into `V5_neon_wet_midnight` or `V1_underwater_caustics` + `A1_hydrophone_sonar`. | Keeps Desmond's face, suit silhouette, vinyl LP, and `<Video 1>` 5-spin dance intact while actually replacing the flat illustration style with wet midnight neon or underwater caustics. |
| **8** | [`ref2va_scene_kitchen.txt`](../../prompt_bank/ref2va_scene_kitchen.txt) | `ref2va` · `345f` (4 fast shots) | Very dense · Hard stainless glare · 2 overlapping speakers + `<\|cutoff\|>` | **Cap 4b — Depth-Gated Dual-Lock on Photoreal Face + Fast Cuts:** Pair `<Picture 1>` with an existing `$H3_INPUT_DIR` portrait (`t2i_29_edit_reference_portrait_seed7.png` or `old_man.jpg`); keep `ref_img` full in `0..19` and attenuated in `20..49` while `Q_vid` flips the kitchen to `V2_frost_snowfall` (ice-house pass + cold breath) or `V4_zero_g_props`. | `<Picture 1>` face holds cleanly across Shots 1, 3, and 4 alongside crisp overlapping dialogue + `<\|cutoff\|>`, while lighting/environment flip unambiguously in one second. |
| **9** | [`t2va_opera_grand_stage.txt`](../../prompt_bank/t2va_opera_grand_stage.txt) | `fl2va` · `345f` (3 shots) | Dense period · Gaslight · Soprano aria (`IT`) + pit orch | **Cap 5a — `1x` Multi-Band 4D Mix (`mid_late_20_49` + `late_30_49`) + `AudioRefineMask` Crossover:** `Q_vid(20..49)` transforms theatre into `V7_tiltshift_porcelain_diorama` while `Q_aud(30..49)` routes the Italian aria into `A2_am_radio_1930s` (or `V6_bioluminescent_moonlight` + `A6_cathedral_pipe_organ`), benchmarked against two-pass `AudioRefineMask`. | Single-pass output delivers both stylized porcelain miniature / bioluminescent visuals and AM-radio / pipe-organ acoustics while keeping the 3-shot framing and Italian aria melody intact. |
| **10** | [`t2va_spatial_reality_portal.txt`](../../prompt_bank/t2va_spatial_reality_portal.txt) | `fl2va` · `141f` (`on_both_clocks`, 1 static shot) | Sparse · Bright golden-hour stone courtyard · Silent | **Cap 5b — `1x` 4D Mix + `2x` Moving Reality Portal Crossover:** `1x` whole-scene `V1_underwater_caustics` + `A1_hydrophone_sonar`, or `2x` moving circular portal revealing `V6` + `V2` glowing frost synced with `1x` temporal `Q_aud[t0:t1]` crossfading birdsong into ice chimes as the portal crosses the fountain. | Static stone arches and flagstones hold rock-steady (`blocks.0..19`) while underwater caustics + sonar read cleanly in `1x`, or portal edges and mid-clip acoustic shift align tightly in the crossover. |

---

## 7. Two Tiny Fresh `141f` Lab Prompts to Add & Render-Time Discipline

Auditing the full `180`-prompt bank surfaced two clean single-shot gaps (plus missing donor stills for several older `ref2va` scripts):

1. **`t2va_binaural_split_counter.txt` (`141f`, 1 static shot, `fl2va`):** Permanent `5.88s` spatial L/R separation with zero camera cuts — Left third of frame shows an open rain-streaked window onto a passing thunderstorm streetcar, Right third shows a brass rotary phone ringing beside a spinning vinyl turntable and seated speaker. Gives a pure, unconfounded testbed for `Q_aud_L` (`w_low`) vs. `Q_aud_R` (`w_high`). *(Optional `l2va` anchor runner-up already in the bank: [`l2va_chess_clock.txt`](../../prompt_bank/l2va_chess_clock.txt), `141f`, two-shot Russian Left vs. English Right.)*
2. **`t2va_threshold_walk_single_take.txt` (`141f`, 1 continuous tracking shot, `fl2va`):** Adult sound engineer walks without cuts from a loud rain-lashed loading dock straight through a heavy padded steel studio door swinging shut at mid-clip (`f72`) into a dead-quiet acoustic booth. Isolates pure `Q_aud[t0:t1]` temporal routing without `[Shot N]` hard-cut resets confounding it (unlike Scene 5 or [`t2va_ww2_sub_sonar.txt`](../../prompt_bank/t2va_ww2_sub_sonar.txt)).

**Render-time discipline:** Because eight of the ten bank prompts declare `328f–345f`, run initial multi-seed kill-or-keep mechanism probes on `90f` or `141f` single-shot prompts (and `length = 1` stills for visual-only checks), then promote proven mechanisms to the full `243f–345f` multi-shot hero renders.
