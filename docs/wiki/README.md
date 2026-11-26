# Top Research Efforts & What Works Right Now

Landing summary of what is working reliably in `ComfyUI-h3-explorations`, when to use it instead of plain prompting, and where to look next. For the full wiki map across all subsystems, see [`index.md`](index.md).

---

## 1. Active Top Priority: `1x` Cross-Modal Query Routing (`MiniMaxH3QueryRouter`)

In a single-stream joint audio-video DiT like MiniMax H3, every token sits in one shared `50`-block self-attention sequence. Changing one language tag, vocal cue, or `overall_soundscape:` sentence in a plain prompt therefore leaks into faces, unpinned co-stars, props, and room architecture — and writing an underwater visual scene drags the soundtrack underwater even when `overall_soundscape` asks for dry speech.

**How `1x` Query Routing fixes it ([`query_routing.md`](query_routing.md)):**
[`MiniMaxH3QueryRouter`](../../steering/query_router.py) lets video queries (`Q_vid`) attend to Prompt A and audio queries (`Q_aud`, including `late_30_49` binaural L/R splits) attend to Prompt B inside the **same single-pass `1x` forward evaluation** — zero second NFE pass, zero trained control vectors, shipped across `T2V`, `fl2va`, and split-cached `ref2va` workflows, `director/compiler.py::compile_audio_redub(isolate_visuals=True)` (`7a7693d0`), Studio 2.0 1-Click Genre Roulette, Smart Twists (`Wrong Voice` calibrated to `a2v=0.0`), & Re-Dub `Isolate Visuals` drawer (`app/` `b49de03`, `97e07e7`), and zero-visual-edit `heylook` diff presets (`h3-query-router-dub`, `h3-query-router-soundscape`).

- **Full technical note & architecture:** [`query_routing.md`](query_routing.md)
- **Step-by-step recipe book & exact prompt diffs:** [`query_routing_recipes.md`](query_routing_recipes.md)
- **Shipped `1x` workflows & diff presets:** [`workflows/h3_text_to_video_pika_query_router_api.json`](../../workflows/h3_text_to_video_pika_query_router_api.json) · [`workflows/h3_first_last_frame_to_video_pika_query_router_api.json`](../../workflows/h3_first_last_frame_to_video_pika_query_router_api.json) · [`workflows/h3_image_ref_plus_text_to_video_pika_query_router_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_query_router_api.json) · [`docs/wiki/system_prompts/h3_query_router/README.md`](system_prompts/h3_query_router/README.md)
- **Verified results & controls:** [`bench/results/2026-11-25_query_router_step1_bidirectional.md`](../../bench/results/2026-11-25_query_router_step1_bidirectional.md) · [`bench/results/2026-11-25_query_router_phase2_sowhat.md`](../../bench/results/2026-11-25_query_router_phase2_sowhat.md) · [`bench/results/2026-11-24_query_router_first_renders.md`](../../bench/results/2026-11-24_query_router_first_renders.md) · [`bench/results/2026-11-24_query_router_live_probes_and_step2_gates.md`](../../bench/results/2026-11-24_query_router_live_probes_and_step2_gates.md)
- **Watch/listen renders:** `$H3_OUTPUT_DIR/Video/h3_explorations_best_of_2026-11-25/reel_h3_explorations_best_of_2026-11-25.mp4` (`88 s` all-tracks best-of showcase reel: Query Routing, Control Vectors, Spatial Portal + 3f SoundCues, Tracked-Mask `EditGuider`, Video-Frozen Dry Redub, & Binaural Closer), `$H3_OUTPUT_DIR/Video/query_router_showcase_2026-11-25/reel_h3_query_router_showcase_2026-11-25.mp4` (`86 s` Query Router reel), `$H3_OUTPUT_DIR/Video/query_router_step1_2026-11-25/`, `$H3_OUTPUT_DIR/Video/query_router_phase2{a,b}_2026-11-25/`, and `$H3_OUTPUT_DIR/Video/query_router_reels_2026-11-24/`

### With Router vs. Plain Prompting Only (Matched Seed & Base Prompt)

The key point is **what did not change in the prompts** ([exact prompt diffs](query_routing_recipes.md#the-golden-rule-clean-prompt-diffs)), and **where cross-modal priors actually collide**:

- **Sight-to-Sound Firewall (`aud_to_vid_gain=0.0`, post-`9e5db27c`) Across Four Domains:** Closing direct `Q_aud -> K_vid` plus the 1-hop `K_vid -> shadow_text -> Q_aud` relay breaks visual-physics and close-up physiology dominance while keeping native lip-sync (`Q_vid -> K_aud` open):
  - **Shell-Shock World-Mute (`shellshock`):** Drops roaring warehouse flames by **`-32.66 dB`** (`-55.75` vs `-23.09` dBFS at `g=1.0`, **`55x` less `> 2 kHz` crackle**) into eerie anechoic stillness/soft breathing while keeping flames, sparks, and the `"DANGER"` sign blazing.
  - **Wrong-Voice Casting (`biker`):** Monotonic 4-rung pitch release (`plain 168.70 Hz` -> `g=1.0 187.37 Hz` -> `g=0.15 216.94 Hz` -> **`g=0.0 260.21 Hz`, `+7.5 st`**) gives a scarred 6'5" outlaw biker a squeaky elderly grandmother voice synced cleanly to his lips across all 3 shots.
  - **Pixel-Locked Underwater Dry Redub (`audiorefine_uwvid`, `7a7693d0`):** Uncached `MiniMaxH3AudioRefineMask` + `a2v=0.0` holds exact source pixels (`pic_mae = 2.337`) while cutting frozen-`K_vid` water hiss **`5.8x`** (`HF>2k 0.1458 -> 0.0250`), raising voiced speech **`+39%`**, and running `26 s` faster (`98.1 s` vs `124.1 s`).
  - **Underwater Video + Dry Speech (`uwvid_dryaud`):** **`+9.43 dB` over `g=1.0`** (`-22.85` vs `-32.28` dBFS) close-mic dry speech over submerged turquoise caustics & bubbles (strictly when `prompt_aud` has zero conflicting visual words).
- **Soundscape-to-Architecture Leaks (`cooking_zen` / `Sitcom` / `Thunderstorm` / `Cathedral`):** Writing `"tranquil Zen tea pavilion"` inside `overall_soundscape:` on `t2va_cooking_show_meltdown` literally tore open sunlit French garden windows with green trees and pink blossoms inside a windowless indoor TV studio (`47.99` MAE). `1x` routing kept the brick studio 100% locked while syncing serene solo cello + meditative whisper to the purple beetroot explosion (`-16.89` dBFS, `HF>2k = 0.0247`).
- **Asymmetric `EN -> JA` Dubbing on Unpinned Cast/Sets (`T2V` & Ensemble `ref2va`):** Translating `<d>` lines alone recast all actors on `stairwell` (`26.79 -> 11.96`), turned `t2va_police_interrogation` into a bright Japanese TV drama (`54.04 -> 21.68`), and on `ref2va_scene_kitchen` (`woman_blonde.jpg`) kept `<Subject 1>` blonde while firing/recasting every unpinned cook to East Asian cooks and erasing `"FIRE"` thermal tickets (`40.03 -> 28.55` MAE). `1x` routing locked every unpinned co-star, plated dish, and ticket across all 4 shots.
- **Depth-Gated `1x` Binaural Split (`late_30_49`, `2/2` scenes):** Keeping `blocks.0..29` open locks speech phase and lip-sync across Left and Right while cross-masking `blocks.30..49` separates two acoustic worlds in one pass (`t2va_radio_drama`: Left = dry 1948 padded broadcast booth, Right = storm-lashed Gothic mansion + horror strings, `lr_corr = 0.1167`, `pic_mae = 16.01`, zero echo across 3 speakers; `stairwell`: `pic_mae = 7.93`, `lr_corr = 0.1122`).

On identical base scenes and seeds (`192` frames at `1344x768`, seed `730451892`, plus `3`-seed confirmation on Thunderstorm and `tortoise` Cathedral Reverb):

| Showcase Case | Same-Seed Plain Prompt Only (No Router) | With `1x` Query Router | Picture Drift vs Base (Plain / Routed) | Audio Signal |
| :--- | :--- | :--- | :---: | :--- |
| **Shell-Shock World-Mute (`shellshock`, `a2v=0.0`)** | Leaked fire roar, phantom *"Danger"* whisper, & broke sign; `g=1.0` still pumped loud fire (`-23.09` dBFS) | Blazing warehouse flames, sparks, & `"DANGER"` sign locked over dead-quiet anechoic breathing | `0.00` / `38.86` *(vs plain)* | **`-55.75` dBFS (`-32.66 dB` vs `g=1.0`, `55x` less crackle)** |
| **Wrong-Voice Casting (`biker`, `a2v=0.0`)** | Deep male voice (`168.70 Hz`) ignored grandma tags; `g=1.0` stayed male (`187.37 Hz`) | Scarred 6'5" biker & `"NO CREDIT"` locked; squeaky elderly grandma voice synced to lips across 3 shots | `0.00` / `32.08` *(vs plain)* | **`260.21 Hz` (`+7.5 st` monotonic pitch curve)** |
| **Pixel-Locked Dry Redub (`audiorefine_uwvid`, `7a7693d0`)** | Frozen underwater pixels at `g=1.0` pumped `4.2x` rushing water hiss (`HF>2k = 0.1458`) over speech | Exact pixel lock (`2.337` MAE) + crisp dry stairwell dialogue, `26 s` faster uncached | `2.34` / `2.34` *(vs UW src)* | **`5.8x` lower HF (`0.0250`), `+39%` voiced speech** |
| **Chaos Visuals + Zen Audio (`cooking_show_meltdown`)** | `"Zen tea pavilion"` soundscape tears sunlit garden windows & blossoms into TV studio brick wall | Windowless brick TV studio & hood locked while cello + Zen whisper sync to beetroot explosion | `47.99` / `39.50` | `-16.89` dBFS, `HF>2k = 0.0247` (`g=1.0`) |
| **Ensemble `ref2va` JA Dub (`ref2va_scene_kitchen`)** | Keeps `<Picture 1>` blonde while recasting every unpinned cook as East Asian & erasing `"FIRE"` tickets | Locks `<Subject 1>`, Black cook, curly-haired cook, Western dishes & `"FIRE"` tickets across 4 shots | `40.03` / `28.55` | Synced overlapping + sung JA + `<\|cutoff\|>` |
| **Zero-Shot Japanese Dub (`police_interrogation` / `stairwell`)** | Recasts all actors as Japanese & shifts precinct/stairwell lighting, wardrobe, and window views | Original cast, wardrobe, lighting & set stay locked while speaking synced Japanese | `54.04` / `21.68` · `26.79` / `11.96` | `21.17` / `16.12` dB spec shift |
| **Spanish & Bilingual `(S1)` ES / `(S2)` EN (`stairwell`)** | Translating `<d>` lines alone drifts both actors' faces (even when `(S2)` stays English) | Original actors & framing stay locked (`ES` verified on `2` seeds) | `28.17` / `9.84` · `25.41` / `11.80` | `11.43` / `11.99` dB spec shift |
| **1970s Sitcom / Thunderstorm / Cathedral (`3` seeds)** | `overall_soundscape:` alone leaks retro 70s hair/lamps, shifted Shot 2 lamp, or church facades | Gritty stairwell / London street stay locked under audience laughs, storm, or reverb | `35.66` / `9.95` · `21.76` / `9.91` · `41.28` / `19.67` | Clean acoustic swap (`3/3` seeds on storm & cathedral) |
| **Underwater Video + Dry Speech (`aud_to_vid_gain=0.0`, `9e5db27c`)** | Submerged visuals drag even dry-soundscape plain prompt & `g=1.0` underwater (`-32.02` / `-32.28` dBFS) | Keeps turquoise caustics & bubbles while restoring crisp close-mic dry stairwell speech | `23.40` *(vs UW base)* | **`-22.85` dBFS (`+9.43 dB` over `g=1.0`)** |
| **`1x` Binaural Split (`radio_drama`, `late_30_49`)** | Impossible in plain stereo (`plain_temporal_horror` scrambled cast `41.70` & flat `H2 HF = 0.0192`) | Exact 1948 studio locked; Left ear = dry booth, Right ear = haunted storm mansion, 0 echo | `41.70` / `16.01` | `lr_corr = 0.1167` (`L -13.7 / R -14.4` dBFS) |

### Where Plain Prompting Already Works (Honest Boundaries)

- **Pure music-genre clashes (`s1_victorian_trap_*`) & explicit voice-quality descriptors (`s1_android_*`):** Plain H3 already kept an 1885 Victorian drawing room + `"LONDON 1885"` mantel plaque 100% period-authentic over booming 808 trap with zero neon bleed (`-14.25` vs `-14.42` dBFS), and already produced clean natural human voices on chrome androids (`147.55` vs `149.50 Hz`). Plain H3 leaks visuals mainly when `overall_soundscape:` invokes a contrasting **physical place or era**, not a pure music genre.
- **`JA -> EN` reverse dub on visually over-specified period pieces (`t2va_samurai_bamboo_duel`):** English carries near-zero visual recasting prior against explicitly described samurai armor and `"関ヶ原街道"` kanji (`28.40` plain vs `25.23` routed MAE — tie).
- **Heavily visual-anchored `EN -> ES` (`t2va_delta_blues_porch`):** Plain Spanish sang cleanly over the steel resonator guitar while preserving the porch, elderly woman, and `"CLARKSVILLE 5 MILES"` tin sign (`35.27` plain vs `31.95` routed MAE — tie).
- **Tight single-headshot `ref2va` (`t2i_29`), sealed-mouth voiceover, & full-prompt `ref2va` restyles (`motion_transfer` -> neon):** Plain prompting already locks tight portraits, seals lips on narration, and shatters flat pop-art Dual-Lock natively (`K`-logit `ref_img` masking and `vid_to_aud_gain` were cut after live falsification).
- **No router needed when visual rewrites agree with sound (`Blueprint Ink`) or calm portrait + `<Audio 1>` groove (`old_man.jpg`):** Plain prompting already matches (`65.45` vs `65.27` MAE on Blueprint Ink).
- **Sub-clip temporal audio switching (`audio_time_start=0.5`, `late_30_49`) caveat:** Locks visuals (`21.93` vs `41.70`) and delivers `5.8x` H2 horror HF on `t2va_radio_drama`, but repeats a spoken phrase whenever `t0` lands flush on dialogue (`"put out the lamp"` / `"Who else knows?"`); use only on silence/music gaps and kept off Studio 2.0's default UI.

---

## 2. `1x` Spatial Visual Query Routing (`MiniMaxH3SpatialQueryRouter`) — Two Worlds, One Shot

Put two completely different visual worlds inside **one** seamless video shot (left/right split-screen, sky/ground split, floating center portal box, or any 2D/`[F, H, W]` `MASK`) in a **single `1x` render (`+4.2 s` overhead)** with **100% stock 3D RoPE** (zero causal Video VAE shimmer, zero hard paste seams).

- **How it works ([`spatial_query_routing.md`](spatial_query_routing.md)):** During early structure blocks (`0..24`), [`MiniMaxH3SpatialQueryRouter`](../../steering/spatial_query_router.py) splits video tokens into two isolated `comfy_kitchen.int8_attention` islands (Base World A vs. Region World B, zero dense `attn_mask`) so each side builds its own opposite lighting, weather, and architecture without 1-hop text bleed. On late blocks (`25..49`), it reopens shared `K_vid_all` so dry cobblestones morph seamlessly into wet neon-reflecting asphalt and pedestrians walk cleanly across the portal boundary.
- **Verified proof (`seed = 730451892`, `90f 1344×768`, 6-step Pika Kitchen+Sol):** Sunlit European cobblestone street (`t2va_spatial_summer_street`) vs. midnight cyberpunk neon rain (`t2va_spatial_cyberpunk_rain`) — Left/Right split (`43.1 s`, `+38.3` L−R luma contrast) and free-form Center Portal Box `x∈[0.28,0.72], y∈[0.18,0.85]` (`42.1 s` vs `37.9 s` stock baseline, zero rectangular border, `f16→17` pop `2.43/2.25`).
- **Guide, visual proofs & recipes:** [`spatial_query_routing.md`](spatial_query_routing.md) · [`assets/h3_spatial_query_routing_proofs.jpg`](assets/h3_spatial_query_routing_proofs.jpg)
- **Standalone workflow & benchmark:** [`workflows/h3_text_to_video_pika_spatial_query_router_api.json`](../../workflows/h3_text_to_video_pika_spatial_query_router_api.json) · [`bench/results/2026-11-25_spatial_query_router.md`](../../bench/results/2026-11-25_spatial_query_router.md)

---

## 3. Other Proven Capabilities in This Repository

| Track | What Works Best | Canonical Entry Points |
| :--- | :--- | :--- |
| **`2x` Spatial Portals & World Directing** | Regional `[F, H, W]` moving masks, interactive portal drawing, `audio_only`/`video_only` `EditGuider`, & SAM 3.1 auto-retargeting | [`edit_guidance.md`](edit_guidance.md) · [`spatial_director_guide.md`](spatial_director_guide.md) |
| **`1x` Orthogonal Control Vectors & H3 Studio 2.0** | Continuous `Look`/`Motion`/`Audio`/`Hold` sliders (`MiniMaxH3Steer` + `add_norm_matched_erasure` on `noir` & `V_r` scene locks) + mobile/desktop studio UI | [`h3_steering_and_architecture.md`](h3_steering_and_architecture.md) · [`steering.md`](steering.md) · [`../../app/README.md`](../../app/README.md) |
| **Audio Freeze & Multi-Window Loops** | `NestedTensor(video=1, audio=0)` audio lock, real song/speech locking (`MiniMaxH3FreezeAudio`), & chained `on_both_clocks` (`39 + 51m`) song windows | [`core/h3_audio_freeze.md`](core/h3_audio_freeze.md) |
| **High-Res Single-Frame Images** | `MiniMax-H3-Image` (`16 MP -> 4 MP`, 16-still split cache, `heylook` expander) & `P-H3-Image` (`4–8 MP`, lazy `Qwen3.5-9B` upsampler) | [`image/h3_image.md`](image/h3_image.md) · [`image/p_h3_image.md`](image/p_h3_image.md) |
| **Masked V2V & Perception Stack** | Dual-clock window contracts, SAM 3.1 / Sapiens2 / SAM 3D Body nodes (`'fun'` campaign parked) | [`masked_v2v/AGENTS.md`](masked_v2v/AGENTS.md) · [`core/h3_frame_rules.md`](core/h3_frame_rules.md) |

---

## 4. Navigation

- **Full wiki router & subsystem index:** [`index.md`](index.md)
- **Session rules & H3 clocks:** [`../../AGENTS.md`](../../AGENTS.md) · **68-node pack registry:** [`../../nodes.py`](../../nodes.py)
- **Repository overview & quick install/usage:** [`../../README.md`](../../README.md)
