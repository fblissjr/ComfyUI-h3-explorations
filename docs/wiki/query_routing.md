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

Control vectors and subspace erasure are always a lossy approximation of a prompt, competing in a crowded activation-steering literature. **Cross-Modal Query Routing** exploits something structurally specific to **single-stream joint audio-video DiTs** like MiniMax H3 (`[text | cond | cond_audio | ref_img | ref_audio | audio | video]` packed into one flat self-attention sequence in `comfy/ldm/minimax/model.py::PackedLayout`), whereas models like `LTX-2` (`arXiv:2601.03233`, asymmetric dual-stream) and `Ovi` (`arXiv:2510.01284`, twin-backbone) already separate video and audio streams architecturally.

Normally, H3's single shared self-attention sequence is an all-or-nothing handcuff: one prompt drives every block and modality, so restyling the video warps the voice, rewriting the audio moves the actors, and `<Picture 1>` locks the lighting.

**Scoped mixing-console model:** Cross-Modal Query Routing turns a single-pass forward evaluation into a live mixing console across **whole-video queries (`Q_vid`), audio query groups (`Q_aud`, `Q_aud_L/R`, `Q_aud[t0:t1]`), inter-modal attention edges (`Q_aud -> K_vid`, `Q_vid -> K_ref_img`, `Q_vid -> K_aud`), and block depth** — with no second pass (`2x` NFE) and **zero control-vector training required**. **Per-region spatial video masks** stay on two-pass `MiniMaxH3EditGuider` + `MiniMaxH3SpatialDirector` ([`edit_guidance.md`](edit_guidance.md)), because Gate 2C empirically falsified `1x Q_vid[mask]` splicing ([`2026-11-23_wave2_gpu_gates_bus_vr_path1c_and_qvid.md`](../../bench/results/2026-11-23_wave2_gpu_gates_bus_vr_path1c_and_qvid.md)).

### Redefining What a "Concept" Even Is

Separating **content** (what is said: look, sound, behavior) from **address** (who reads it: stream, channel, time span, speaker, block depth) lets most creative ideas stay as plain English prompts:

- **Audio prompts need no control vectors:** A dedicated `Q_aud` prompt (thunderstorm on tin roof, vintage AM radio bandpass, live studio laugh track, underwater hydrophone + sonar) is cleaner, sharper, and zero-maintenance compared with contrastive audio stems.
- **Stop building vague mood/film stems:** Claude's bank cosine audit ([`data/query_routing_2026_11_24/bank_shared_direction.json`](../../data/query_routing_2026_11_24/bank_shared_direction.json)) confirmed that older "high energy" stems (`chaos_panic`, `youtube_overhype_influencer`, `vocal_strain`, `auctioneer_rapid_patter`, `pyro_chain_explosions`) all share one dominant text-conditioning direction, while film-stock stems sit nearly parallel to each other. *(Note: this clustering is measured in conditioning space; a direct benchmark comparing routed English prompts against stems is planned in the first `MiniMaxH3QueryRouter` probes.)*
- **Keep control vectors only where discrete prompts fall short:** continuous scalar scrubbing (when linear output blending between two routed prompts is not used), ideas where plain one-clause prompt rewrites stall (as observed on `knitted_wool_world` and `zero_gravity` in Phase 1 `tortoise`), sub-symbolic kinematic or two-still visual transfers (`MiniMaxH3MakeMediaControlVector`), and single-prompt moving `2x` spatial portals (`MiniMaxH3EditGuider`).

---

## 2. How & Why It Works Inside MiniMax H3

1. **Why cross-modal bleed happens by default (`comfy/ldm/minimax/model.py`):**
   - `Qwen3-VL-32B` (`layers.0..49`, width `5120`) encodes `[vision | text]` rows (`conditioning_token_tags`), then `MiniMaxH3Model._embed_and_pack` projects through `condition_proj` (`5120 -> 5376`) and a `2`-block bidirectional `TokenRefiner` (`comfy/ldm/minimax/model.py:619-622`).
   - `PackedLayout` packs `[text | cond | cond_audio | ref_img | ref_audio | audio | video]` into one contiguous sequence where every `50` `DiTBlock` runs full joint self-attention, distinguished only by `3D` RoPE `(t, h, w)` and AdaLN segment tags (`SEG_TAG`). Stereo audio is packed **channel-major** (`pack_audio`: `[L(0..T-1), R(0..T-1)]` on H3's `40 Hz` clock) with Left pinned to `w_low` and Right pinned to `w_high` (`_audio_grid`).
2. **How `1x` query routing breaks the coupling ([`steering/hooks.py`](../../steering/hooks.py)):**
   - `H3SteerStepWorkspace` pre-allocates tiny reusable buffers for the text tokens (`l_txt`) strictly under `pause_malloc_graph()` (`+0` `malloc_graph_breaks` across all four attention backends).
   - `H3SteerBlockPatch` wraps each `DiTBlock`: `prepare_block` runs `norm1` / `qkv_proj` / fused RoPE on the shadow text stream (`shadow_a` / `shadow_b`); `pre_attention` swaps text K/V in place inside `swap_blocks` (when lengths match) or concatenates `[K_b, K_non_txt]` in a dedicated side `SDPA` strictly for the tiny `Q_aud` slice (`au_spans`) and shadow text queries; `post_attention` writes `aud_out` back into `au_spans` before `out_proj`; `finish_block` advances the shadow text stream through `out_proj` + `mlp` across blocks `0..49`.
3. **Why `1x Q_aud` stays cheap and clean while `1x Q_vid[mask]` washed out:**
   - `au_spans` holds only `130..810` stereo rows (`~1–3%` of sequence length), with distinct `SEG_TAG` and `(h=0, w ∈ {w_low, w_high})` coordinates — so its side-SDPA adds negligible overhead and stays isolated.
   - Masking `Q_vid[mask]` (`~23,328` video rows in `90f`) failed Gate 2C (`1.047x–1.090x` contrast vs. `1.421x` `2x EditGuider`, `1836.1 MB` VRAM) because unmasked and masked video tokens still attend bidirectionally to each other (`Q_vid -> K_vid, V_vid`) across `50` blocks, washing out regional boundaries inside a single forward pass.

---

## 3. Known Verified Findings Today (Strictly Tiered)

| Tier | Finding & Observable Evidence | Honest Caveat / Boundary |
| :--- | :--- | :--- |
| **Tier 1 — Reproduced / Multi-Seed** | **`noir` Pole-A erasure flips sparse stills & `192f` `covered_market`:** reproduces `5/5` B&W on `indoor_neutral` stills (`positive_control_stills_metrics.json` vs `gate2b1/metrics_gate2b1.json`, `5` seeds) and `192f` `covered_market` (`positive_control_metrics.json` vs `gate2b2/metrics_gate2b2.json`). | Fails at default strength (`s=1.0`, `stage_b_surgical_erasure_v2.json`), on `39f` `indoor_neutral` video (`positive_control_metrics.json`), and only partially mutes prompts naming bright object colors (`tortoise_noir`). |
| **Tier 1 — Reproduced / Multi-Seed** | **Low-rank `V_r` preserves dense scenes where equal-RMS isotropic prompt shrink (`S4_iso_shrink`) collapses them:** verified on `gate2b1/metrics_gate2b1.json` and contact sheets (`5` seeds on `covered_market`). | On `watercolor_wash`, `S1_own_Vr` keeps the scene like `S0_1d` rather than beating `1D` on concept strength. |
| **Tier 1 — Checked (`1` blind reader)** | **70-pair blind stills matrix (`stills_matrix_unsealed.json`):** `7` visual concepts × `2` prompts × `5` seeds (`1D` vs `erasure` at high strength): `46` tied/same, `15` neither, `6` erasure (`5/6` sparse `noir`), `3` `1D`. | Read blind by one session on small still tiles before unsealing; shows `add_norm_matched_erasure` ties `1D` elsewhere and wins sparse `noir`, so keeping it as default costs nothing while erasure R&D pauses. |
| **Tier 1 — Multi-Run Architecture Laws** | **Dual-Lock raw VAE boundary** (`Gate 2B`, `gate2b2`), **Gate 2C `1x Q_vid[mask]` falsification** (`2026-11-23_wave2_gpu_gates_bus_vr_path1c_and_qvid.md`), **early-step palette clock `[0.0, 0.50)`** (`2` seeds × `3` concepts, `2026-11-22_phase4...md`), **`0.33` synthetic-ref release** (`1` seed across multiple configs, `2026-11-12_sound_cues_dpad_and_033_release_law.md`), and **bank shared-direction collapse** (`bank_shared_direction.json`). | Dual-Lock requires gating/releasing `ref_img` itself; `Q_vid[mask]` stays `2x`. |
| **Tier 2 — Working Control (`1` seed / pair)** | **`1x` `Q_aud`/`Q_vid` text split (`auto_1x_kv`):** matched vs steering both streams on `4` concept-scene pairs across `2` checkpoints (`showcase_c`, `stage2_verification`, `wave2`, `wave3b`) and `+0` added `malloc_graph_breaks` across `4` backends. | Loudness stays closer to base in `4/4`; paired-frame pitch improves in `2/3` pitch-measured pairs (global median in `1/3`) and worsens in `1/3`, exposing residual `Q_aud -> K_vid` pitch bleed. Zero shipped workflows wire `auto_1x_kv` yet. |
| **Tier 2 — Working Lead (`1` seed)** | **Routing stops cross-modal sound-to-picture leaks (`routed_as_bank_metrics.json` & `followup_bands_metrics.json`):** on `reverb_cathedral`, `audio_only` keeps the London street intact (`pic_mae` cut roughly in half vs forced `both`); on `operatic_bellow`, opening `blocks.20..49` (`audio_only`) unlocks pitch/loudness shifts vs. zero movement in `late_30_49`, still leaving part of the shift coupled to visual performance. | **Direct counter-signal on `golden_hour_glow`:** routing `video_only` + `mid_late_20_49` did *not* prevent cardigan/hat/candle distortion in `routed_as_bank` (`1` seed). Block boundaries (`20..29` vocal delivery vs `30..49` room reverb) remain **single-seed leads** pending 3-seed confirmation. |
| **Tier 2 — Working Control (`1` seed)** | **`2x` audio-view thunderstorm stand-in & 10 moving portals:** swapping soundscape via `2x` `audio_only` `EditGuider` holds stairwell video cleanly while delivering storm acoustics (`probe_metrics.json`, ran at `scale=1.5`); moving single-prompt `2x` portals verified across `10` concepts × `10` scenes (`2026-11-24_outside_review...md` Addendum). | Needs `1x` arbitrary-length replacement in `MiniMaxH3QueryRouter` and multi-seed repeats. |

---

## 4. Cross-Domain Prior-Art Map (DiT vs. LLM/VLM, `2026-11-24`)

*(Sources verified `2026-11-24`; retrieval omitted from split-text routing because no faithful analogue was found — "query routing" alone in IR refers to LLM/index model routing.)*

| Capability (Shipped or Proposed) | Exists in DiT World? | Closest LLM / VLM / Audio Analogue | What Specifically Remains Unclaimed in Single-Stream AV DiTs (`2026-11-24`) |
| :--- | :---: | :--- | :--- |
| **`1x` Split-Text K/V Routing (`Q_vid -> Prompt A`, `Q_aud -> Prompt B`, shadow text evolution)** | **Not found** (`LTX-2` `2601.03233` & `Ovi` `2510.01284` split via separate model towers; `Soundwich` `2610.00691` targets H3/LTX-2/Ovi via multi-stem carriers + Carrier Replay; `Regional-Prompting-FLUX` `2411.02395` is image-only spatial) | **None found** in retrieval; closest LLM pattern is runtime KV-bank swapping / prefix isolation | **Training-free `1x` single-pass cross-modal text K/V routing** (arbitrary-length dual prompts or steered/unsteered split + section isolation) inside a frozen single-stream AV DiT. **Strongest unclaimed piece.** |
| **`1x` Sight-to-Sound Attention Logit Damping (`Q_aud -> K_vid` in `blocks.30..49`)** | **Not found** (binaural/AV models like `AV-Link` use trained gating modules; `Soundwich` uses SAM entity masks) | **`PASTA`** (`2311.02262`, ICLR 2024: inference-time attention-logit reweighting) & VLM **`PAI`** (`2407.21771`) / **`OPERA`** (`2311.17911`: layer-targeted visual logit scaling) | **Training-free late-block `Q_aud -> K_vid` logit-damping dial** mapping lip-sync vs. vocal-pitch-invariance in a frozen AV DiT. **Cleanly unclaimed.** |
| **`1x` Temporal (`Q_aud[t0:t1]`) & Binaural Stereo (`Q_aud_L` / `Q_aud_R`) Routing** | **Not found** (stereo V2A models train position-aware encoders; `Soundwich` replays stem carriers) | Spatial audio beamforming & time-frequency source masks; `Regional-Prompting-FLUX` applied to 1D `(channel, time)` rows | **Training-free `1x` prompt routing onto channel-major stereo (`w_low`/`w_high`) and `40 Hz` time slices** inside a frozen AV DiT (needs fast `141f` probe first). |
| **Depth-Gated Reference Attenuation (`ref_img` `0..19` vs `20..49`) & Single-Band Block Prompting** | **Yes / Partial in T2I** (`P+` `2303.09522`, `InstantStyle` `2404.02733`, `B-LoRA` `2403.14572`, `SplitFlux` `2511.15258`, `TADA!` `2602.11910`, `MAINodes` per-block H3 deltas) | VLM `PAI` (`2407.21771`) layer-specific visual-token scaling; layer-selective LLM steering | Standalone block-band routing is **not** novel; unclaimed parts are **simultaneous distinct `Q_vid` + `Q_aud` bands in one `1x` pass** and **late-block `Q_vid -> K_ref_img` attenuation for raw-VAE Dual-Lock** (both unbuilt probes; note `golden_hour_glow` counter-signal on `0..19` vs `20..49`). |
| **Pole-A Subspace Erasure (`add_norm_matched_erasure`) & Moving Portals (`2x`)** | **Partial** (`SAFREE` `2410.12761`, `CURE` `2505.12677`, `RASteer` `2610.01969`, `MAINodes` 1D H3 dials; `Prompt-to-Prompt` `2208.01626`) | `ACE` (`2411.09003`), `MiMiC` (`2402.09631`), Conceptors (`2410.16314`), VLM `VCD` (`2311.16922`) | Only the exact erasure slider recipe (empirically ties `1D` outside sparse `noir`) and the single-prompt `2x` AV portal composition. |

---

## 5. Content vs. Address: Multimodal References, Masking, & Top 10 "SO WHAT?"s

Once control vectors stop driving everything, **the prompt's own structure plus multimodal references and masks form one unified Content + Address system**:

- **References are non-text content (`ref_img`, `ref_video`, `ref_audio`, `vision_tokens`):** Reference blocks are contiguous segments inside `PackedLayout.segments`, so routing them by reader (`Q_vid` vs `Q_aud`) and block depth requires simple per-segment scaling/masking rather than dual text shadow evolution.
- **Masks are addresses (reusing the pack's existing grammar):**
  1. **Text-span / Section / `[Shot N]` / Speaker masks ([`h3_spans.py`](../../h3_spans.py)):** Choose which conditioning rows each modality reads from a single prompt (or address `(S1)` alone while leaving `(S2)` untouched).
  2. **Temporal & Stereo audio masks (`1x` `Q_aud`):** Choose which `40 Hz` time frames or Left/Right stereo channels attend to Prompt B.
  3. **Spatial `[F, H, W]` video masks (`2x` `EditGuider` + `MiniMaxH3SpatialDirector` + SAM 3.1 / Sapiens2):** Stay two-pass for per-region video edits, composing directly with `1x` audio/reference routing so spatial visual portals never warp soundtracks.

### Top 10 "SO WHAT?"s & Two-Step Build Plan

| # | Goal ("So What?") | Current Status (`2026-11-24`) | Where It Lives / Next Milestone |
| :-: | :--- | :--- | :--- |
| **1** | **Two scripts, one `1x` pass (`Q_vid` Prompt A, `Q_aud` Prompt B)** | Working `2x` stand-in (`pic_mae=5.003`); arbitrary-`l_txt` `1x` in **Build Step 1** | `MiniMaxH3QueryRouter` + parity & 3-seed probe |
| **2** | **Zero sound-to-picture leak via section auto-routing** | Proven manually on `reverb_cathedral` (`routed_as_bank_metrics.json`); auto-split in **Build Step 1** | `section_split` (`soundscape_to_audio` / `sections_by_modality`) |
| **3** | **Sight-to-sound coupling knob (`Q_aud -> K_vid` in `30..49`)** | Residual pitch bleed measured; dial planned for **Build Step 2** | Additive `[1, 1, 1, L]` `attn_mask` in [`steering/hooks.py`](../../steering/hooks.py) |
| **4** | **Zero-LoRA depth-layered style (`0..19` Prompt A, `20..49` Prompt B)** | **Hypothesis with counter-signal** (`golden_hour_glow` `20..49` still shifted jacket/hat in `routed_as_bank`) | Keep as probe (`video_block_band` in **Build Step 1**) — do not promise geometry lock |
| **5** | **Depth-gated `ref_img` Dual-Lock dial (`0..19` vs `20..49`)** | Hypothesis sharing the same `0..19` premise; queued for **Build Step 2** | Per-depth `ref_img` gain probe after Step 1 parity |
| **6** | **Multimodal reference routing (`ref_img`/`ref_video`/`ref_audio` per reader)** | Design target for **Build Step 2** (reuses `PackedLayout.segments`) | `Q_vid -> ref_img/ref_video`, `Q_aud -> ref_audio` isolation |
| **7** | **Sub-clip `1x` temporal (`Q_aud[t0:t1]`) & binaural L/R (`Q_aud_L/R`)** | Planned for **Build Step 2** (`141f` bleed-vs-crispness kill-or-keep probe) | `pack_audio` / `_audio_grid` slice routing |
| **8** | **Per-speaker voice directing & `1x` dubbing probe** | Partial reach noted (`operatic_bellow` `followup_bands`); needs Step 1 + speaker span routing | `(S1)`-row audio routing with `(S2)` in-clip control |
| **9** | **Unified masking across `1x` router & `2x` spatial portals / `AudioRefineMask`** | Integration track after Step 1 parity (`h3_spans` + `SpatialDirector`) | Beat-synced portal & `AudioRefineMask` redub workflows |
| **10** | **Off the concept-vector treadmill without dropping what works** | Settled: `add_norm_matched_erasure` stays default (`noir` win, zero cost elsewhere) | Full prompts in `MiniMaxH3QueryRouter` handle expressiveness; vectors handle continuous dose / edge cases |

**Two-Step Build Discipline:**
- **Build Step 1 (`v1`, landing now):** Standalone **`MiniMaxH3QueryRouter`** node + arbitrary-length shadow K/V support in `steering/hooks.py` + lifting the single non-`"all"` `block_band` error in `steering/nodes.py`, delivering: (a) parity with `MiniMaxH3Steer(streams="audio_only")`, (b) arbitrary-length `audio_conditioning` / `video_conditioning`, (c) independent per-stream `video_block_band` + `audio_block_band`, and (d) single-prompt `section_split` (`soundscape_to_audio`, `sections_by_modality`).
- **Build Step 2 (`v2`, after Step 1 parity + 3-seed probes):** `Q_aud -> K_vid` late-block coupling dial, depth-gated `ref_img` / `ref_video` / `ref_audio` routing, temporal `Q_aud[t0:t1]` and binaural `Q_aud_L` / `Q_aud_R` audio routing, and single-speaker addressing.

---

## 6. Additional Existing-Tool Workflows & Evaluation Practices to Test

Capabilities ready to test using nodes already in the pack:

1. **Ship `1x` Query Routing workflows (`workflows/build_workflows.py`):** ship clean `1x` `video_only`, `mid_late_20_49` / `late_30_49` `audio_only`, and combined `video_only + audio_only` graphs (plus wire `streams` and `block_band` into [`bench/edit_guidance_arms.py`](../../bench/edit_guidance_arms.py) so automated tests match routed configs).
2. **Beat-synced music-video reality portals (`MiniMaxH3FreezeAudio` + `MiniMaxH3SpatialDirector` + `MiniMaxH3Steer`):** lock a real music track while driving moving wipes/portals across the video; because audio is frozen, visual steering runs with zero vocal pitch drift.
3. **Motion transfer + single-prompt moving portal (`MiniMaxH3AppendRefVideo` + `<Picture 1>` + `MiniMaxH3SpatialDirector`):** transfer real choreography onto a reference character and sweep a zero-rewrite portal across the performance.
4. **Fast multi-take audio redub (`MiniMaxH3AudioRefineMask` + `MiniMaxH3FrozenVideoCache`):** freeze video K/V once and audition multiple voice/foley/acoustic takes (`audio_only` steering or rewritten audio prompts) with truly locked pixels — zero shipped workflows use `MiniMaxH3AudioRefineMask` today.
5. **Single-prompt regional mask steering on `MiniMax-H3-Image` (`length = 1`):** combine `MiniMaxH3EditGuider` + spatial mask + `MiniMaxH3Steer` at single-frame speed for rapid multi-seed region checks.
6. **`[Shot N]` & single-speaker targeting:** [`h3_spans.py`](../../h3_spans.py) already parses `[Shot N]` headers; test whether targeting one shot's rows or one speaker's rows stays localized despite `TokenRefiner` self-attention.
7. **Forced-choice blind concept identification gate:** adapt [`bench/control_vector_readout_score.py`](../../bench/control_vector_readout_score.py)'s blind packet format to clips — show a blind judge the unsteered and routed clips and ask it to name what changed from a short list. Any concept a viewer cannot identify within one watch is cut.

---

## 7. Fresh From-Scratch Concepts ("Unmistakable in 1 Second", Non-Destructive)

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

## 8. Curated 10-Scene Diverse Generalization Matrix

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

## 9. Two Tiny Fresh `141f` Lab Prompts to Add & Render-Time Discipline

Auditing the full `180`-prompt bank surfaced two clean single-shot gaps (plus missing donor stills for several older `ref2va` scripts):

1. **`t2va_binaural_split_counter.txt` (`141f`, 1 static shot, `fl2va`):** Permanent `5.88s` spatial L/R separation with zero camera cuts — Left third of frame shows an open rain-streaked window onto a passing thunderstorm streetcar, Right third shows a brass rotary phone ringing beside a spinning vinyl turntable and seated speaker. Gives a pure, unconfounded testbed for `Q_aud_L` (`w_low`) vs. `Q_aud_R` (`w_high`). *(Optional `l2va` anchor runner-up already in the bank: [`l2va_chess_clock.txt`](../../prompt_bank/l2va_chess_clock.txt), `141f`, two-shot Russian Left vs. English Right.)*
2. **`t2va_threshold_walk_single_take.txt` (`141f`, 1 continuous tracking shot, `fl2va`):** Adult sound engineer walks without cuts from a loud rain-lashed loading dock straight through a heavy padded steel studio door swinging shut at mid-clip (`f72`) into a dead-quiet acoustic booth. Isolates pure `Q_aud[t0:t1]` temporal routing without `[Shot N]` hard-cut resets confounding it (unlike Scene 5 or [`t2va_ww2_sub_sonar.txt`](../../prompt_bank/t2va_ww2_sub_sonar.txt)).

**Render-time discipline:** Because eight of the ten bank prompts declare `328f–345f`, run initial multi-seed kill-or-keep mechanism probes on `90f` or `141f` single-shot prompts (and `length = 1` stills for visual-only checks), then promote proven mechanisms to the full `243f–345f` multi-shot hero renders.
