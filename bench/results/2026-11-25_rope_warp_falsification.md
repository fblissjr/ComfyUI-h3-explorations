# Core-Free 3D RoPE Time/Space Warper Falsification (`2026-11-25`)

Standalone `16`-clip, `3`-round GPU falsification (`1344×768`, 6-step Pika H3 Flex `MiniMaxH3Sol`, `seed = 730451892`, zero ComfyUI core edits, zero new `node_id` minted) probing H3's unified 3D `(t, h, w)` RoPE grid (`comfy/ldm/minimax/model.py`) via a core-free `("double_block", i)` hook under `steering.hooks.pause_malloc_graph_ctx()`:

- **Part A (`13` clips, `141f` / `5.875 s`, `fl2va`):** Locked-off static cafe scene with continuous left-half motion (barista raising a silver carafe and pouring steaming espresso into a clear glass, silent) and rhythmic right-half motion + speech (man in a red jacket tossing a yellow tennis ball while saying *"Watch the ball, don't blink, one, two, three!"*).
- **Part B (`3` clips, `90f` / `3.75 s`, `ref2va`):** Symmetric blank white art gallery with zero left/right words in the prompt, testing `<Picture 1>` (`$H3_INPUT_DIR/old_man.jpg`) `(h, w)` RoPE box placement while leaving target `video` and `audio` RoPE 100% untouched.
- **Rendered clips:** `$H3_OUTPUT_DIR/Video/rope_warp_lab_2026-11-25/` (all `16` clips independently watched and listened to across three sessions plus owner eye-test).
- **Decision:** **Do NOT mint a permanent `MiniMaxH3RopeWarp` node or wire RoPE controls into Studio 2.0 (`app/`).** Keep these mechanistic findings documented here so future work never re-runs the falsified paths.

---

## 1. Results Table (`seed = 730451892`, `1344×768`, 6-Step Pika `MiniMaxH3Sol`)

### Part A — Spatial Time, Slow-Mo & Audio 3D RoPE (`141f`, `fl2va`)

| Round / Arm | Scope (`blocks`) | `L / R` MAE | `macro` `f20→100` (`L / R`) | `L / R` Sharpness | `f16→17` Pop (`L`) | Audio RMS (dBFS) | Honest Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| R1 `rope_base` (identity) | `0–49` | `3.07 / 6.59` | `22.49 / 19.76` | `225.9 / 151.0` | `3.54` (`max 3.89`) | `-27.15` | **Exact `0.000e+00` `max_abs` RoPE parity** across all `6` steps; crisp baseline (`lr_corr = 0.999`, `pan = -0.08 → -0.01 dB`). |
| R1 `rope_bullet_time_left` (raw `t = t_v0` left) | `0–49` | `8.60 / 6.54` | `4.17 / 17.15` | `33.5 / 84.8` | `31.53` (`max 32.46`) | `-29.23` | **Falsified (hazy strobe):** Locked left pose, **but** collapsing `t` across `[1,4,4,4,4]` VAE chunks caused heavy temporal blur and `31.5` pixel pops every `17` frames (`f16→17`). |
| R1 `rope_time_gradient_x` (`0.0× → 1.5×` across `w`) | `0–49` | `7.09 / 9.52` | — | — | — | `-29.95` | **Broken past `1.0×`:** Temporal stretch up to `1.5×` sheared the moving tennis ball and hand into triple green smeared ghosts (`00:03–00:05`). |
| R1 `rope_vid_slowmo_045` (raw `0.45×` video `t`) | `0–49` | `7.38 / 10.25` | `24.91 / 17.17` | `149.0 / 134.5` | `33.37` (`max 39.57`) | `-29.53` | **Broken (shimmer/pop):** Damped motion while keeping `1.0×` audio (`env_corr = 0.863`), but triggered `33–43` pixel `17f` VAE boundary pops. |
| R1 `rope_tenet_aud_rev` (audio `t` reversed) | `0–49` | `3.64 / 7.18` | `24.02 / 21.11` | `288.5 / 158.2` | `4.30` (`max 4.38`) | `-19.66` | **Mechanistically real (niche novelty):** Forward video stayed razor-sharp (sharper than base) while H3 natively synthesized reversed-phoneme speech + backwards pour; reversing audio via `ffmpeg -af areverse` recovers clear English *"Watch the ball, don't blink, one, two, three!"*. |
| R1 `rope_aud_w_sweep` (both ears swept `w_min→w_max`) | `0–49` | `3.79 / 7.82` | — | — | — | `-21.05` | **Script bug in R1 (stereo collapse):** Overwrote `ch0` and `ch1` with the same `w(t)` (`lr_corr 0.999 → 0.943`, `pan_shift = -0.01 dB`), though video stayed crisp and mix jumped `+6.1 dB`. |
| R2 `rope_bt_b0_24` (raw `t = t_v0` left) | `0–24` | `9.23 / 7.72` | `16.48 / 16.78` | `66.5 / 128.5` | `32.83` (`max 34.46`) | `-28.70` | **Still broken:** Early-block gating alone did NOT fix raw `t_0` collapse — `f16→17` pop (`32.83`) and vertical center seam persisted. |
| R2 `rope_bt_chunk_b0_24` (chunk-phase `clip_idx = 0` left) | `0–24` | `1.79 / 6.54` | `16.92 / 16.78` | `122.4 / 131.1` | `1.86` (`max 2.80`) | `-28.63` | **Pop eliminated (`17×` drop) & zero seam:** Preserving `[0, 5/3, 25/3, 45/3, 65/3]` intra-chunk VAE phase killed the `17f` pop completely; left side became silky slow-mo while right stayed full speed. |
| R2 `rope_slowmo_chunk_b0_24` (`0.45×` `clip_idx` + intra-phase) | `0–24` | `5.08 / 8.36` | `35.81 / 35.64` | `103.0 / 83.4` | `6.99` (`max 8.49`) | `-27.31` | **Flawed (chunk overlap math):** Because `0.45 × 28.333 = 12.75 < 21.667` (`max(intra_phase)`), adjacent `17f` chunks overlapped backward by `9` frames, causing hand ghosting at `00:01`. |
| R2 `rope_aud_lr_swap` (exact `w_low ↔ w_high`) | `0–49` | `3.11 / 6.79` | `22.78 / 19.78` | `224.1 / 146.0` | `3.37` (`max 4.20`) | `-28.95` | **Clean on-grid swap:** Flipped start pan sign (`-0.08 → +0.08 dB`) with 100% crisp video, though barely audible on a near-mono (`lr_corr = 0.999`) baseline. |
| R2 `rope_aud_stereo_pan` (fixed `ch0 < ch1` sweep across `w`) | `0–49` | `3.34 / 7.28` | `21.95 / 17.91` | `240.7 / 203.8` | `3.69` (`max 4.14`) | `-19.01` | **No L→R pan (`+0.02 dB`), unexpected `+8.1 dB` proximity boost:** Keeping `ch0 < ch1` separated still produced zero L→R stereo pan, **but** pulling audio `w` inward from frame edges made audio `+8.1 dB` hotter/closer with sharper video than baseline. |
| R3 `rope_bt_chunk_b0_39` (chunk-phase `clip_idx = 0` left) | `0–39` | `1.89 / 5.75` | `9.66 / 15.93` | `136.7 / 129.6` | `3.04` (`max 3.29`) | — | **Crispest Bullet-Time arm — still NOT a true Matrix freeze:** Zero seam and zero `17f` pop, barista posture/arm/glass locked (`macro_L 22.49 → 9.66`), **but** fast continuous pouring espresso still flows at `1.0×` inside every `0.71 s` (`17f`) chunk (owner eye-test: too subtle to ship as "Bullet-Time"). |
| R3 `rope_bt_chunk_all` (chunk-phase `clip_idx = 0` left) | `0–49` | `4.81 / 6.09` | `18.03 / 15.50` | `105.8 / 122.9` | `20.97` (`max 21.05`) | — | **Proves blocks `40..49` must stay stock:** Warping the final `10` decoder blocks brought the `f16→17` pop right back (`3.04 → 20.97`). |

### Part B — Zero-Shimmer `<Picture 1>` Spatial `(h, w)` RoPE Placement (`90f`, `ref2va`)

| Round / Arm | `<Picture 1>` `(x0..x1, y0..y1)` Box | `L / R` Sharpness | `f16→17` Pop (`L`) | Dark-Subject `x` Centroid | Honest Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| R3 `ref_place_base` | stock (centered) | `58.8 / 26.1` | `0.85` | `0.409` | Clean white gallery baseline; `<Subject 1>` stands dead-center. |
| R3 `ref_place_left` | `x=[0.02..0.42], y=[0.08..0.92]` | `58.1 / 29.7` | `0.64` | `0.431` | **Falsified:** `<Subject 1>` still stands dead-center (slight camera framing shift only). |
| R3 `ref_place_right` | `x=[0.58..0.98], y=[0.08..0.92]` | `57.3 / 29.7` | `0.76` | `0.473` | **Falsified:** `<Subject 1>` still stands dead-center; H3 `ref2va` is translation-invariant to `ref_img` `(h, w)` RoPE offsets. |

---

## 2. Mechanistic Takeaways Worth Keeping

1. **The Causal `[1, 4, 4, 4, 4]` Video VAE Hard Boundary (`comfy/ldm/minimax/model.py:102-109`):**
   - Every `5`-token VAE clip (`17` pixel frames = `0.71 s`) encodes `t_rel = clip_idx * (85/3) + [0, 5/3, 25/3, 45/3, 65/3]`.
   - Any warp that breaks the intra-chunk offsets (raw `t_0` freeze, raw `0.45×` scaling, `> 1.0×` stretch) or touches decoder blocks `40..49` (`rope_bt_chunk_all`) causes `21–43` pixel luminance/alignment pops at `f16→17` (`17k - 1 → 17k`).
   - Preserving `[0, 5/3, 25/3, 45/3, 65/3]` while leaving `blocks 40..49` stock (`rope_bt_chunk_b0_39`) completely eliminates pops (`3.04 < base 3.89`) and seams, **but** creates a physical ceiling: because each `0.71 s` chunk still runs forward at `1.0×`, continuous sub-second textures (falling liquid, rising steam) keep animating normally while only multi-second macro progression freezes. Non-overlapping `clip_idx` scaling also requires `scale >= (65/3) / (85/3) = 0.765` (below `0.765`, adjacent chunks overlap backward in RoPE time).
2. **Audio Temporal Axis (`t`) Is Cleanly Reversible (`rope_tenet_aud_rev`):**
   - Unlike the asymmetric `[1, 4, 4, 4, 4]` Video VAE, H3's `40 Hz` Audio VAE uses uniform `Δt = 1.0` stride (`_audio_grid`, `comfy/ldm/minimax/model.py:124-130`), so reversing audio `t` (`t_min + (t_max - t)`) stays 100% on-grid, leaves forward video razor-sharp (`288.5 vs 225.9`), and natively synthesizes true backwards-phoneme dialogue and reverse foley.
3. **Audio Spatial `w` & `ref_img` `(h, w)` Findings:**
   - `_audio_grid` pins `ch0` to `w_low` and `ch1` to `w_high`. Even when preserving `ch0 < ch1` separation, sweeping audio `w` across the frame did not pan stereo (`+0.02 dB` shift on a near-mono baseline), though pulling audio `w` inward from the outer frame edges raised mix loudness by `+8.1 dB` (`-19.01 vs -27.15 dBFS`) with sharper video (`240.7 / 203.8`).
   - Remapping `<Picture 1>` `ref_img` `(h, w)` RoPE coordinates into left/right target boxes produced virtually identical centered subjects (`x = 0.409 / 0.431 / 0.473`), confirming `ref2va` is trained spatially translation-invariant to `ref_img` `(h, w)` positions.
4. **Core-Free Stacking Note (If Any RoPE Node Is Ever Built Later):**
   - `WrappersMP.DIFFUSION_MODEL` pre-modifying `payload["layout"]` gets silently discarded at `comfy/ldm/minimax/model.py:731` if `MiniMaxH3QueryRouter` subsequently swaps `context` to a different `text_len` (`layout.signature` mismatch). Hooking `("double_block", i)` under `pause_malloc_graph_ctx()` runs post-layout-build, supports per-block gating (`0..39` vs stock `40..49`), and composes safely in both orders with `H3SteerBlockPatch` (`steering/hooks.py:1276-1287`, which only reads the untouched `text` slice of `rope_freqs`).
