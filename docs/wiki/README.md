# Top Research Efforts & What Works Right Now

Landing summary of what is working reliably in `ComfyUI-h3-explorations`, when to use it instead of plain prompting, and where to look next. For the full wiki map across all subsystems, see [`index.md`](index.md).

---

## 1. Active Top Priority: `1x` Cross-Modal Query Routing (`MiniMaxH3QueryRouter`)

In a single-stream joint audio-video DiT like MiniMax H3, every token sits in one shared `50`-block self-attention sequence. Changing one language tag, vocal cue, or `overall_soundscape:` sentence in a plain prompt therefore leaks into faces, wardrobe, props, and room lighting, and writing an underwater visual scene drags the soundtrack underwater even when `overall_soundscape` asks for dry speech.

**How `1x` Query Routing fixes it ([`query_routing.md`](query_routing.md)):**
[`MiniMaxH3QueryRouter`](../../steering/query_router.py) lets video queries (`Q_vid`) attend to Prompt A and audio queries (`Q_aud`) attend to Prompt B inside the **same single-pass `1x` forward evaluation** (zero second NFE pass, zero trained control vectors, shipped in [`workflows/h3_text_to_video_pika_query_router_api.json`](../../workflows/h3_text_to_video_pika_query_router_api.json)).

- **Full technical note & architecture:** [`query_routing.md`](query_routing.md)
- **Step-by-step recipe book & exact prompt diffs:** [`query_routing_recipes.md`](query_routing_recipes.md)
- **Raw live probe metrics & controls:** [`bench/results/2026-11-24_query_router_first_renders.md`](../../bench/results/2026-11-24_query_router_first_renders.md) · [`bench/results/2026-11-24_query_router_live_probes_and_step2_gates.md`](../../bench/results/2026-11-24_query_router_live_probes_and_step2_gates.md)
- **Watch/listen comparison reels:** `$H3_OUTPUT_DIR/Video/query_router_reels_2026-11-24/` (including `reel_plain_vs_1x_router_stairwell_s730451892.mp4`)

### With Router vs. Plain Prompting Only (Matched Seed & Base Prompt)

On identical base scenes and seeds (`stairwell` `192` frames at `1344x768`, seed `730451892`, plus `3`-seed confirmation on Thunderstorm and `tortoise` Cathedral Reverb), swapping **only** dialogue `<d>` tags or **only** `overall_soundscape:` in a plain prompt produces clear cross-modal drift that `1x` routing eliminates:

| Showcase Case | Same-Seed Plain Prompt Only (No Router) | With `1x` Query Router | Picture Drift vs Base (Plain / Routed) | Spec Shift vs Base (Plain / Routed) |
| :--- | :--- | :--- | :---: | :---: |
| **Zero-Shot Japanese Dub** | Translating `<d>` lines alone recasts both actors into Japanese actors, swaps wardrobe & conduit, and changes Shot 3 brick window to misty hills | Original actors, clothes, lighting & brick window stay locked while speaking synced Japanese | `26.79` / `11.96` | `16.65` / `16.12` dB |
| **Zero-Shot Spanish Dub** | Translating `<d>` lines alone drifts both actors' facial identities and Shot 1 framing | Original actors & framing stay locked while speaking synced Spanish (`2` seeds verified) | `28.17` / `9.84` | `11.04` / `11.43` dB |
| **Bilingual `(S1)` ES / `(S2)` EN** | Translating only `(S1)`'s four lines still drifts both faces (including English `(S2)` in Shot 3) | Original actors lock while `(S1)` speaks Spanish and `(S2)` answers in English | `25.41` / `11.80` | `12.22` / `11.99` dB |
| **1970s Sitcom Laugh Track** | Changing `overall_soundscape:` alone gives woman a retro flipped bob, adds Shot 2 wall lamp, and flattens lighting | Serious concrete stairwell stays locked while live studio laughter & applause play between lines | `35.66` / `9.95` | `16.07` / `9.47` dB |
| **Thunderstorm (`stairwell`, `3` seeds)** | Changing `overall_soundscape:` alone moves Shot 2 wall lamp next to woman & darkens Shot 3 window | Stairwell geometry stays locked under violent storm audio (`3/3` seeds) | `21.76` / `9.91` | `11.15` / `12.06` dB |
| **Cathedral Reverb (`tortoise`, `3` seeds)** | Writing cathedral reverb in `overall_soundscape:` replaces London street facades with church doors | London street & tortoise stay locked (`3/3` seeds; `section_split` stops leak too) | `41.28` / `19.67` | `5.47` / `3.95` dB |
| **Underwater Visuals + Dry Speech** | Writing submerged visuals + dry concrete `overall_soundscape:` still muffles & bubbles speech (`-23.60` dB HF tilt) | Keeps turquoise caustics & bubbles while snapping speech brighter (`-20.79` dB HF tilt at `aud_to_vid_gain=0.25`) | `47.10` / `48.54` | `20.38` / `14.97` dB |

### Where Plain Prompting Already Works or Concepts Stayed Too Subtle

- **No router needed when visual rewrites agree with sound (`Blueprint Ink`):** Rewriting `stairwell` visuals to architectural blueprint ink while leaving dry stairwell acoustics in place yielded almost identical visuals (`65.45` plain vs `65.27` routed) and speech (`19.44` vs `17.50` dB) — plain prompting already handles that cleanly.
- **No router needed on calm portrait + `<Audio 1>` groove (`old_man.jpg`):** Plain prompting (`ref_audio_stream="both"` passthrough) already kept the man calm (`0.6143` head motion; background orbs drifted gently without beat-locking), whereas in-place `V[ref_audio]` zeroing (`"audio_only"`) broke audio into metallic buzz (`28.49` dB).
- **Audio concepts cut as too subtle for blind demos:** Although `Hydrophone + Sonar` also prevented a Shot 2 lamp leak (`24.00` vs `9.59`), both `Hydrophone + Sonar` and `1930s AM Radio` were too subtle by ear on casual watch without priming.
- **Still paused / under construction:** Naive `1x` sub-stream audio splits (`audio_channel`, `audio_time_start/end`) smeared across open intra-audio attention (block-diagonal masking fix is committed in [`steering/hooks.py`](../../steering/hooks.py) and awaits live `:8188` verification); `ref_img_gain` and `ref_audio_stream` require query-scoped `K`-logit masking instead of in-place `v_bh` mutation.

---

## 2. Other Proven Capabilities in This Repository

| Track | What Works Best | Canonical Entry Points |
| :--- | :--- | :--- |
| **`2x` Spatial Portals & World Directing** | Regional `[F, H, W]` moving masks, interactive portal drawing, `audio_only`/`video_only` `EditGuider`, & SAM 3.1 auto-retargeting | [`edit_guidance.md`](edit_guidance.md) · [`spatial_director_guide.md`](spatial_director_guide.md) |
| **`1x` Orthogonal Control Vectors & Pocket Console** | Continuous `Look`/`Motion`/`Audio`/`Hold` sliders (`MiniMaxH3Steer` + `add_norm_matched_erasure` on `noir` & `V_r` scene locks) + mobile/desktop studio UI | [`h3_steering_and_architecture.md`](h3_steering_and_architecture.md) · [`steering.md`](steering.md) · [`../../app/README.md`](../../app/README.md) |
| **Audio Freeze & Multi-Window Loops** | `NestedTensor(video=1, audio=0)` audio lock, real song/speech locking (`MiniMaxH3FreezeAudio`), & chained `on_both_clocks` (`39 + 51m`) song windows | [`core/h3_audio_freeze.md`](core/h3_audio_freeze.md) |
| **High-Res Single-Frame Images** | `MiniMax-H3-Image` (`16 MP -> 4 MP`, 16-still split cache, `heylook` expander) & `P-H3-Image` (`4–8 MP`, lazy `Qwen3.5-9B` upsampler) | [`image/h3_image.md`](image/h3_image.md) · [`image/p_h3_image.md`](image/p_h3_image.md) |
| **Masked V2V & Perception Stack** | Dual-clock window contracts, SAM 3.1 / Sapiens2 / SAM 3D Body nodes (`'fun'` campaign parked) | [`masked_v2v/AGENTS.md`](masked_v2v/AGENTS.md) · [`core/h3_frame_rules.md`](core/h3_frame_rules.md) |

---

## 3. Navigation

- **Full wiki router & subsystem index:** [`index.md`](index.md)
- **Session rules, H3 clocks, & 68-node pack map:** [`../../AGENTS.md`](../../AGENTS.md)
- **Repository overview & quick install/usage:** [`../../README.md`](../../README.md)
