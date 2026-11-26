# Spatial Visual Query Routing (`1x` Split-Screen & Portal Worlds)

**Put two completely different worlds inside one seamless MiniMax H3 video shot — in a single render pass (`+4 s`, `1.1×` speed), with zero cut lines, zero paste-in edges, and zero flicker.**

Want the left half of a street to be a **golden sunlit summer afternoon in Italy** while the right half is a **pouring midnight cyberpunk neon alley** — or want a **floating cyberpunk portal right down the center** of a sunny cobblestone street where people with glowing umbrellas walk straight out into daylight? [`MiniMaxH3SpatialQueryRouter`](../../steering/spatial_query_router.py) routes two prompts to different parts of the frame inside **one** pass, while keeping MiniMax H3's 3D RoPE 100% stock.

![1x Spatial Visual Query Routing — Same-Seed Visual Proofs](assets/h3_spatial_query_routing_proofs.jpg)

**Quick links:** [Standalone Split-World Workflow (`h3_text_to_video_pika_spatial_query_router_api.json`)](../../workflows/h3_text_to_video_pika_spatial_query_router_api.json) · [Shipped Curated Workflows (`7` Spatial + `1` Freeze Portal)](#shipped-curated-workflows) · [Camera ↔ Mic Query Routing](query_routing.md) · [Stage 2 Music-Video & Track 2 Co-Audit (`2026-11-26`)](../../bench/results/2026-11-26_stage2_music_video_and_studio_unlock.md) · [`v2-slim` & Beat 1–3 Showcases (`2026-11-26`)](../../bench/results/2026-11-26_spatial_qr_v2_and_showcases.md) · [4-Sensor Silicon Gate (`2026-11-26`)](../../bench/results/2026-11-26_h3_4sensor_calibration.md) · [2× Moving Portals & Existing-Take Edits](spatial_director_guide.md)

---

## 1. The "So What?" (Why Regular Users Care)

Normally, if you try to describe two opposite worlds in one plain video prompt (*"sunny summer left, dark rainy cyberpunk right"*), diffusion models average the lighting together into muddy overcast twilight — or if you composite two videos in post, you get a harsh cut line where shadows, reflections, and walking people break at the seam.

`MiniMaxH3SpatialQueryRouter` solves both problems in **one normal generation**:
- **True opposite lighting in one shot:** Bright sunlit warm limestone on one side (`96.0` luma) right next to pitch-black neon midnight rain (`57.8` luma) on the other — zero muddy averaging.
- **Organic seam merging:** Dry sunlit cobblestones literally turn wet and catch hot-pink neon reflections as they cross the center, and pedestrians walk cleanly across the portal boundary.
- **Works on any box or mask:** Left/Right split, Top/Bottom sky split, floating center portal box, or any arbitrary 2D / `[F, H, W]` ComfyUI `MASK` (including `SAM 3.1` subject masks!).
- **Fast & razor-sharp:** Uses `comfy_kitchen.int8_attention` and leaves H3's 3D RoPE 100% untouched — **`42.1 s` vs. `37.9 s` stock baseline** (`+4.2 s` on `90f 1344×768`), with zero `17`-frame causal VAE shimmer (`f16→17` pop `1.94–3.33` matching baseline `1.71–2.69`), and unlocks **`MiniMax-H3-Image`** (`latent_t=1`, `2368×1760` `4 MP` still splits in `33.7 s` with `seam_1px_col_mae = 4.45`).

---

## 2. How It Works (Progressively, Visually)

![How 1x Spatial Visual Query Routing Works Across 50 DiT Blocks](assets/h3_spatial_query_routing_explainer.svg)

MiniMax H3 generates a video across **50 Transformer blocks** (`blocks 0..49`), where early blocks decide global layout, weather, and lighting, and late blocks refine textures, reflections, and local continuity.

1. **Step 1 — Pick Two Prompts & a Region:**
   - **Base Prompt (World A)** feeds everything *outside* the region (e.g., sunlit summer street).
   - **Region Prompt (World B)** (`region_conditioning`, any token length!) feeds everything *inside* the region — defined either by a simple normalized rectangle (`x_start..x_end`, `y_start..y_end`) or any optional `MASK`.
2. **Step 2 — Early Blocks `0..24` ("Build Two Separate Worlds"):**
   - Video tokens are split into two disjoint index lists (**Base** and **Region**) with **zero dense `attn_mask`**.
   - **Base** attends only to `[Text A, Video Base, Audio/Refs]`; **Region** attends only to `[Text B, Video Region, Audio/Refs]`.
   - Because each island uses half-length K/V on `comfy_kitchen.int8_attention`, early blocks actually take *half* the attention FLOPs while blocking 1-hop cross-world text bleed (`Video A → Text B → Video B`).
3. **Step 3 — Late Blocks `25..49` ("Open the Door & Blend the Seam"):**
   - Both prompts stay split (`Text A` vs. `Text B`), **but** both sides now attend over the **entire** shared video canvas (`K_vid_all`).
   - H3 naturally unifies perspective, ground reflections, cast shadows, and moving figures right across the border — zero hard seam!

---

## 3. Practical Recipes (Copy-Paste Settings)

Load the shipped standalone workflow **[`workflows/h3_text_to_video_pika_spatial_query_router_api.json`](../../workflows/h3_text_to_video_pika_spatial_query_router_api.json)** (Pika H3 Flex 6-step, `1344×768`, `141f`). Node `5` is **Base Prompt (World A)**, Node `148` is **Region Prompt (World B)**, and Node `149` is **`MiniMaxH3SpatialQueryRouter`**.

### The Golden Rule for Seamless Blends
> **Match your camera, framing, and horizon sentence between Prompt A and Prompt B.**
> Put the exact same shot setup in both prompts (e.g., *"Live-action, crisp 35mm wide static locked-off tripod shot looking straight down a wide street..."*) and diverge **only** the world, weather, era, lighting, materials, and wardrobe. Matching camera geometry lets H3 align walls, horizons, and ground planes naturally across any split or portal!

### Recipe 1 — Left / Right Split-Screen (Two Worlds, One Street) *(Workflow Default)*
- **Use for:** Summer vs. Winter, Ancient Rome vs. Cyberpunk, Bright Day vs. Noir Night down the center of one street or room.
- **Settings on `MiniMaxH3SpatialQueryRouter`:**
  - `x_start = 0.50`, `x_end = 1.00`, `y_start = 0.00`, `y_end = 1.00`
  - `invert = False` *(flip to `True` to swap Left ↔ Right instantly!)*
  - `isolation = "early_0_24"` *(best contrast + seamless ground merge; try `"seamless_off"` for an even softer atmospheric fade)*
  - `audio_zone = "base"` *(summer birds/footsteps; switch to `"region"` for heavy neon rain!)*

### Recipe 2 — Floating Dimension Portal Box
- **Use for:** A sci-fi / fantasy rift opening down the middle of a normal room, corridor, or street while both foreground edges stay grounded in the real world.
- **Settings on `MiniMaxH3SpatialQueryRouter`:**
  - `x_start = 0.28`, `x_end = 0.72`, `y_start = 0.18`, `y_end = 0.85`
  - `invert = False`, `isolation = "early_0_24"`, `audio_zone = "base"`
- **Verified result (`spatial_box_portal_iso_0_24`):** Sunlit limestone street & sidewalk tables on both outer sides while the center opens into towering neon towers with umbrella pedestrians emerging into daylight (`42.1 s`, zero border artifact).

### Recipe 3 — Sky / Ground Horizon Split
- **Use for:** Replacing just the sky with a glowing alien nebula, aurora borealis, or storm cloud deck while keeping the foreground desert/ocean bright and untouched.
- **Settings on `MiniMaxH3SpatialQueryRouter`:**
  - `x_start = 0.00`, `x_end = 1.00`, `y_start = 0.00`, `y_end = 0.45`
  - `isolation = "early_0_24"` *(or `"early_0_16"` for softer horizon lighting bleed onto the ground)*

### Recipe 4 — Custom Shape, Feathered `[0..1]` Mask, `1×` Temporal Multi-Shot (`all_0_49`), or `SAM 3.1` Background Mask
- **Use for:** Non-rectangular portals, feathered reality halos, **`1×` hard multi-shot scene cuts (`all_0_49`)**, or swapping the full background behind a tracked `SAM 3.1` subject mask at Step 0.
- **Settings:** Connect any ComfyUI `MASK` (`[H, W]` static or `[F, H, W]` video mask matching the render frame count `F`) to `MiniMaxH3SpatialQueryRouter.mask`. When connected, `mask` overrides `(x_start..y_end)` and downsamples onto H3's `32×32` causal patch grid (hard-isolating at `>= 0.5` on `iso` blocks and blending smoothly with continuous `[0..1]` weights on non-`iso` blocks, respecting `invert`).
- **Temporal Multi-Shot (`f0..84` -> `f85..191`):** Build a step mask (`0.0` on `f0..84` -> `1.0` on `f85..191` aligned on a `17f` VAE clip boundary) and **set `isolation = "all_0_49"` + `audio_zone = "follow_mask"`** (`h3_text_to_video_pika_multishot_spatial_cut_api.json`) for a true 1-frame AV hard cut (`116.2 s`, `f84->85` `mae = 88.61`, `+10.56 dB` audio jump & `+410.7 Hz` spectral jump from solo piano into belted rock vocals with zero cross-cut bleed).
- **Small `~10%` `SAM 3.1` Subject Masks / Step-`0` `B35+B40` Self-Masks (`mid_20_34_shield`):** For small foreground regions where full symmetric isolation isolates only `10%` of tokens, use **`mid_20_34_shield`** (shields `region` during `20..34` while letting `base` read all visual context; verified at `153.91 s` turning a tracked dancer into mirror chrome while locking white-brick loft `bg_margin_mae = 0.45`, and on `4 MP` `MiniMax-H3-Image` turning bread extracted straight from Step-`0` `Q_vid @ K_txt["bread"]^T` into glowing sapphire crystal at `4.30×` inside/outside MAE with zero `SAM` or coordinates!). Keep `*_shield` with **`invert = False`** (put the transformed target inside `region`).
- **Continuous Depth Waves (`MoGe-2`) & Kinematic Wakes ("Midas Walk"):** Feed any smooth metric depth ramp (`MoGe-2`) or `MiniMaxH3SpatialDirector` trailing wipe into `.mask` with **`mid_20_34` + `audio_zone = "follow_mask"`** so frozen winter environments bloom strictly in a walking `<Picture 1>` subject's wake (`sat 0.203 -> 0.433`) or freeze depth-layer by depth-layer (`50.6..722.6 m` foreground-to-peaks).
- **`MiniMax-H3-Image` `<Picture 1>|<Picture 2>` Dual-Ref Splits:** Include explicit **`"on the left half"`** and **`"on the right half"`** spatial framing in Prompt A and Prompt B so `early_0_24` places both identities into their respective halves cleanly (`seam_1px_mae = 4.49`, `3.1×` smoother than `all_0_49`'s `26.28`).

### Recipe 5 — Swap, Follow-Mask, or Split the Soundtrack (`region` / `follow_mask` / `binaural_split`)
- Set **`audio_zone = "region"`** so stereo audio queries attend to **Region Prompt (World B)** instead of Base Prompt (World A) (e.g., hear midnight cyberpunk rain while watching the split world, or keep unsteered studio sound on `Vector Reality Lens` with `invert = True`).
- Set **`audio_zone = "follow_mask"`** on temporal step cuts or traveling spatial wipes so `40 Hz` audio frames automatically attend to Prompt B as `mask(t)` crosses active frames (`+10.56 dB` & `+410.7 Hz` jump on `f85` cut).
- Set **`audio_zone = "binaural_split"`** for a **`1×` 3D Binaural Split-World**: blocks `0..29` keep both ears shared for phase-locked lip-sync, and blocks `30..49` split Left/Right ears via the **`v2-slim` Visual-to-Ear Firewall** (`txt_aud_l` sees `vid_l`, `txt_aud_r` sees `vid_r`, blocking cross-ear visual leak). Align shared `[Shot 1]` `<d>` lines between Prompt A and Prompt B for crisp non-overlapping stereo dialogue (`124.52 s`, `lr_corr = 0.1406`).

---

## 4. Which Tool Should I Use? (`1×` Spatial Router vs. `2×` EditGuider)

| Goal | Best Tool | Speed | Why |
| :--- | :--- | :-: | :--- |
| **Generate a brand-new split-screen, sky split, `SAM 3.1` background swap, or static portal from scratch** | **`MiniMaxH3SpatialQueryRouter`** (`1×`) | **~`1.1–1.35×`** | Builds both worlds natively at Step 0 with soft `[0..1]` feather blending & `binaural_split` audio (`lr_corr 0.0152`, `4.1×` L/R spectral split) |
| **Split Camera vs. Microphone** (dubbing, wrong voice, Genre Roulette, Shell-Shock Mute) | **[`MiniMaxH3QueryRouter`](query_routing.md)** (`1×`) | **`1.0×`** | Routes `Q_vid` vs. `Q_aud` with the `aud_to_vid_gain = 0.0` Sight-to-Sound Firewall |
| **Edit an *existing* video take**, **re-skin a small tracked subject**, or animate a **moving circular portal** (`lock_take`) | **[`MiniMaxH3EditGuider`](edit_guidance.md) + [`MiniMaxH3SpatialDirector`](spatial_director_guide.md)** (`2×`, `last_step=3`) | `2.0×` | Anchors Step 0 and applies contrast-amplified `[F, H, W]` velocity deltas (`last_step=3` on 6-step Pika avoids overcook; moving `1×` `t2va` portals park in the horizon) |
| **Continuous style/mood dials** (`noir`, `claymation`, `reverb`) | **[`MiniMaxH3Steer`](steering.md)** (`1×`) | **`1.0×`** | Smooth orthogonal sliders with zero prompt rewrites (context-mode `streams="both", block_band="all"` verified at `42.6 s` stacked with `MiniMaxH3SpatialQueryRouter`) |

*(Requirements & limits: `MiniMaxH3SpatialQueryRouter` works on `t2va`, `fl2va`, `ref2va`, **and `MiniMax-H3-Image` (`latent_t=1`)** models — preserving per-segment-`t` `cond` keyframes, `ref_img` / `ref_audio` tokens across isolated islands, and stacking cleanly with context-mode `MiniMaxH3Steer(streams="both", block_band="all")`. Use **`ModelAttentionBackend("comfy kitchen attention")`** with optional `MiniMaxH3Sol`; it refuses `MiniMaxH3SageAttention` — even under `MiniMaxH3Sol` — and refuses stacking on the same active step with `MiniMaxH3QueryRouter`, hook-mode `MiniMaxH3Steer`, `MiniMaxH3FrozenVideoCache`, `BlockSparseAttention`, or a second `MiniMaxH3SpatialQueryRouter`. Attach `attach_h3_sensor_probe` AFTER `MiniMaxH3SpatialQueryRouter` when reading attention heatmaps.)*

---

## 5. `MiniMaxH3SpatialQueryRouter` Controls Reference

| Input | Default | What It Controls |
| :--- | :---: | :--- |
| **`model`** | *required* | Base H3 / `MiniMax-H3-Image` model (`ModelAttentionBackend("comfy kitchen attention")` + optional `MiniMaxH3Sol`) |
| **`region_conditioning`** | *required* | Second prompt (`CONDITIONING`, any token length; wire `["5", 0]` for `Vector Reality Lens`) |
| **`x_start` / `x_end`** | `0.50` / `1.00` | Normalized horizontal bounds `[0.0, 1.0]` of the region (`0.50..1.00` = right half) |
| **`y_start` / `y_end`** | `0.00` / `1.00` | Normalized vertical bounds `[0.0, 1.0]` of the region (`0.18..0.85` for center portal) |
| **`invert`** | `False` | `False`: inside box/mask gets World B (`region`); `True`: outside gets World B (keep `False` on `*_shield`) |
| **`isolation`** | `early_0_24` | 9 modes: `early_0_24`, `seamless_off`, `early_0_16`, `early_0_32`, `all_0_49` (1-frame cuts), `mid_20_34` (0..19 warmup + 35..49 blend), `mid_20_49` (0..19 warmup + strict 20..49), `early_0_24_shield`, `mid_20_34_shield` (asymmetric small-mask `region` shield) |
| **`audio_zone`** | `base` | `base`: stereo audio reads main prompt; `region`: reads `region_conditioning`; `binaural_split`: shared `0..29`, visual-firewalled `30..49` L/R split; `follow_mask`: `40 Hz` per-frame blend following `mask(t)` |
| **`start_percent` / `end_percent`** | `0.0` / `1.0` | Denoising step window where spatial routing is active |
| **`mask`** | *optional* | Any 2D `[H, W]` or video `[F, H, W]` `MASK` (`1` or `F` frames; hard-isolates `>= 0.5` on `iso` blocks and feathers smoothly `[0..1]` outside); overrides `x/y` box when connected |

---

## Shipped Curated Workflows

- [`workflows/h3_text_to_video_pika_spatial_query_router_api.json`](../../workflows/h3_text_to_video_pika_spatial_query_router_api.json) — `141f` Left/Right split-world street (`early_0_24`, base workflow).
- [`workflows/h3_text_to_video_pika_multishot_spatial_cut_api.json`](../../workflows/h3_text_to_video_pika_multishot_spatial_cut_api.json) — `192f` `all_0_49 + follow_mask` `f85` AV multi-shot hard cut (`SolidMask` step mask).
- [`workflows/h3_text_to_video_pika_binaural_dialogue_split_api.json`](../../workflows/h3_text_to_video_pika_binaural_dialogue_split_api.json) — `192f` `binaural_split` synchronized stairwell \| ferry stereo dialogue split.
- [`workflows/h3_text_to_video_pika_vector_lens_claymation_api.json`](../../workflows/h3_text_to_video_pika_vector_lens_claymation_api.json) — `141f` single-prompt `MiniMaxH3Steer(claymation_stopmotion)` + `MiniMaxH3SpatialQueryRouter(invert=True, audio_zone="region")` Vector Reality Lens.
- [`workflows/h3_r2v_midas_walk_spatial_wake_pika_api.json`](../../workflows/h3_r2v_midas_walk_spatial_wake_pika_api.json) — `141f` Midas Walk kinematic wake (`<Picture 1>`, `<Video 1>` sprite release `0.33`, `SoundCues`, `MiniMaxH3SpatialDirector`, `mid_20_34`, `follow_mask`).
- [`workflows/image/h3_image_t2i_spatial_query_router_api.json`](../../workflows/image/h3_image_t2i_spatial_query_router_api.json) — `MiniMax-H3-Image` `t2i` (`16 -> 4 MP`) split-world vintage \| cybernetic clockmaker.
- [`workflows/h3_text_to_video_pika_audio_freeze_beat_cut_api.json`](../../workflows/h3_text_to_video_pika_audio_freeze_beat_cut_api.json) — `141f` `1× SpatialQR(all_0_49, audio_zone="base")` + `MiniMaxH3FreezeAudio` causal `f85` (`3.54 s`) music-video beat-drop smash cut.
- **2× Moving Circular Portal Partner ([`spatial_director_guide.md`](spatial_director_guide.md)):** [`workflows/h3_text_to_video_pika_audio_freeze_spatial_portal_api.json`](../../workflows/h3_text_to_video_pika_audio_freeze_spatial_portal_api.json) — `141f` `MiniMaxH3FreezeAudio` + `MiniMaxH3SpatialDirector(portal_path)` + `2× EditGuider(streams="video_only", first_step=1, last_step=3, scale=1.65)` (avoiding `1×` `32×32` curved-edge stair-stepping).

## Further Reading & Verification

- **Measured GPU gates:** [`bench/results/2026-11-26_stage2_music_video_and_studio_unlock.md`](../../bench/results/2026-11-26_stage2_music_video_and_studio_unlock.md) (`8` Music-Video renders + Track 2 `SelfMaskRouter` AV co-audit) · [`bench/results/2026-11-26_spatial_qr_v2_and_showcases.md`](../../bench/results/2026-11-26_spatial_qr_v2_and_showcases.md) (`22` Beat 1–3 renders) · [`bench/results/2026-11-26_h3_4sensor_calibration.md`](../../bench/results/2026-11-26_h3_4sensor_calibration.md) (`4-Sensor Gate` & `SelfMaskRouter` live PoC) · [`bench/results/2026-11-26_bucket3_spatial_qr_and_sam3.md`](../../bench/results/2026-11-26_bucket3_spatial_qr_and_sam3.md) (`17` Bucket 3 + `mid_20_34` clips) · [`bench/results/2026-11-25_spatial_query_router.md`](../../bench/results/2026-11-25_spatial_query_router.md) (`4` Phase 1 clips)
- **Implementation & CPU test suite:** [`steering/spatial_query_router.py`](../../steering/spatial_query_router.py) · [`bench/check_spatial_query_router.py`](../../bench/check_spatial_query_router.py) · [`steering/live_policy.py`](../../steering/live_policy.py)
- **Related pages:** [`query_routing.md`](query_routing.md) (`1x` Camera ↔ Mic Split) · [`spatial_director_guide.md`](spatial_director_guide.md) (`2x` Moving Portals) · [`README.md`](README.md) (Top Research Highlights)
