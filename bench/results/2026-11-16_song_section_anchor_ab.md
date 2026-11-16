# Bucket 3 Verification — Timeline-Scoped DiT Frame-0 `section_anchor` & `anchor_end_percent = 0.33` Release

Date: 2026-11-16
Model: Pika H3 6-step (`MINIMAX_H3_MODEL_PIKA` + `PH3_VIDEO_VAE` + `MINIMAX_H3_ENCODER`)
Resolution / Grid: `1344x768`, `context_frames = 39` (`c = 12` latent steps), `context_noise = 0.6`, `context_profile = "seam_taper"`, `seed = 730451892`
Attention Chain: `MiniMaxH3SageAttention(mode="auto", head_chunks=1)` + `MiniMaxH3Sol(dense_blocks="49", quantizer="rotated", sink_conditioning="exact_kv_and_all_rows", start_percent=0.2)`
Script: [`bench/probe_bucket3_section_anchor.py`](../probe_bucket3_section_anchor.py)
Unit Suite: [`bench/check_audio_freeze.py`](../check_audio_freeze.py) (`check_section_anchor`)
Summary & Outputs: `data/bucket3_probe_2026-11-16/probe_summary.json`, `$H3_OUTPUT_DIR/Video/bucket3_probe_2026-11-16/` (zero media committed to `data/`)

---

## Code Changes Verified in Bucket 3

1. **Timeline-scoped DiT-only Frame-0 anchor ([`audio_freeze_song.py`](../../audio_freeze_song.py#L515-L536), [`MiniMaxH3AudioFreezeSong`](../../audio_freeze_song.py#L828)):**
   - Appended optional `section_anchor` (`"off"` default | `"per_section"` | `"whole_track"`) and `anchor_end_percent` (`0.33` default, range `[0.05, 1.0]`) at the end of `MiniMaxH3AudioFreezeSong.define_schema().inputs` (`bench/node_id_manifest.json` updated via [`bench/check_node_ids.py`](../check_node_ids.py); clean-room adaptation crediting `roycho87/geiru_context_suite` & `ckinpdx/ComfyUI-MMH3Tools`).
   - Encodes a section's post-trim opening pixel frame `images[:1]` (`zoomed_images[:1]` when zoomed) once per needed section via `vae.encode` (`~0.2 s` overhead on `W1`/`W3`, producing a `vt = 1` standalone latent `(1, 24, 1, 48, 84)` that avoids `c % 5 == 2` 4-frame latent slicing) and attaches it via DiT-only `minimax_refs` (`"section_anchor": True`, `"ref_img"` segment at `t < cursor`), avoiding both Qwen3-VL re-encoding and the 3D RoPE collision at `t = cursor` with Window `2..N`'s carried head context.
   - Seeds Window `1` from `prev[:, :, :1]` when continuing from a stored `continue_from` window, and resets `active_anchor` on `--- label` timeline entry transitions under `section_anchor = "per_section"` (`line 1269`).
2. **Selective `WrappersMP.DIFFUSION_MODEL` release & Bucket 4 hook ([`attach_section_anchor_release`](../../audio_freeze_song.py#L682-L815)):**
   - Strips **only** `section_anchor = True` `"ref_img"` segments from `PackedLayout`, `refs`, and `cond_video_latents` once denoising crosses `anchor_end_percent` (`frac_step >= 0.33`, i.e., active on Steps `0–1` and released cleanly for Steps `2–5` on 6-step Pika), while leaving any user-wired `<Picture N>` / `<Video N>` refs and every surviving token's 3D RoPE `position_ids` untouched.
   - While active (`Steps 0–1`), publishes `transformer_options["h3_section_anchor_spans"]` (exception-safe `finally` cleanup) for Bucket 4's `return_lse=True` subset attention balance.
3. **Unit & contract coverage ([`bench/check_audio_freeze.py`](../check_audio_freeze.py#L1358)):**
   - Offline CPU tests verify `off` / `per_section` / `whole_track` capture/reset across a 4-window 2-section timeline (asserting `W3` captures post-trim `f243`, never pre-trim context), `continue_from` seeding, selective `PackedLayout` slicing with a coexisting user `<Picture 1>` ref and bitwise 3D RoPE invariance across 6-step Pika sigmas, and invalid-knob refusals.

---

## Section 1 — 4-Window 2-Section Timeline (`447 f = 18.625 s`, `studio -> stage` at `00:10.125`)

Windows planned: `4 x 141 f` (`window_frames = 192`, `context_frames = 39`, written spans `W1: f0..140` `studio`, `W2: f141..242` `studio`, `W3: f243..344` `stage`, `W4: f345..446` `stage`), driven by [`prompt_bank/t2va_studio_dancer_close.txt`](../../prompt_bank/t2va_studio_dancer_close.txt) (`--- studio`) followed by a low-key dark midnight-blue theater spotlight block (`--- stage`) + 120 BPM drum track (`rms_dbfs = -23.36`, `peak_dbfs = -0.25`). All pixel means reduced in `float64`.

| Arm (`section_anchor`, `end%`) | Wall Time (s) | `W1` Studio Luma | `W2` Studio Luma (`W2 vs W1`) | `W3` Stage Luma (`B - R`) | `W4` Stage Luma (`B - R`) | `W4 vs W3` Drift | `W3` Sharpness (`Lap Var`) | `W4` Sharpness (`Lap Var`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `b3_4win_off` (`off`, `0.33`) | `276.29` | `141.604` | `130.612` (`-7.76%`) | `29.223` (`+32.99`) | `50.251` (`+35.94`) | **`+71.96%`** | `86.34` | `111.45` |
| `b3_4win_whole_track` (`whole_track`, `0.33`) | `268.94` | `141.604` | `130.959` (`-7.52%`) | `35.555` (`+32.89`) | `34.638` (`+34.19`) | `-2.58%` | `50.52` | `114.39` |
| `b3_4win_per_section` (`per_section`, `0.33`) | `267.94` | `141.604` | `130.959` (`-7.52%`) | `39.312` (`+32.21`) | `45.306` (`+33.52`) | **`+15.25%`** (`4.7x` lower) | `60.50` | **`115.61`** |

- **Anchor capture telemetry:** `[1] captured section anchor (studio frame 0, latent (1, 24, 1, 48, 84))` took **`0.2 s`**. Under `per_section`, `[3] section anchor reset at timeline entry 'stage'` cleared the `studio` anchor at Window 3 and `[3] captured section anchor (stage frame 0, latent (1, 24, 1, 48, 84))` took **`0.2 s`** (`0.4 s` total across the `268 s` 4-window render, `< 0.15%` overhead).

### Visual Findings (`bucket3_4win_stacked.mp4` & `bucket3_4win_stacked.1fps.00000-00446.mp4`)

- **Identical `studio` opening (`f000..f140`) & clean `W1 -> W2` seam (`f140 -> f141`):** All 3 arms share pixel-identical `W1` frames; `seam_12` jumps remain well below local frame-to-frame motion (`seam_l1 = 3.54–3.70`, `seam_jump_ratio = 0.6361–0.8273`, `seam_luma_jump <= 0.65`).
- **`whole_track` cross-section interference (`f251..f446`):** Carrying Window 1's bright neutral studio anchor into `--- stage` washes out `W3`'s wide spotlight silhouette (`f251..f290`, dancer facing away with pale grey trousers and shadowed face, dropping `W3` sharpness `86.34 -> 50.52`) and leaves `W4` under-lit in the key beam (`34.638` mean luma).
- **`off` (`W3..W4` stage) vs. `per_section` (`W3..W4` stage):**
  - Under `off`, `W3` starts dark (`29.223`), shows open-mouth expression drift despite `no vocal sound` around `f297..f305`, and then **jumps `+71.96%` brighter into `W4` (`50.251`)**.
  - Under `per_section`, resetting at `--- stage` and anchoring `W4` onto `W3`'s post-trim `f243` stage frame holds coherent cyan overhead key + cobalt shadow lighting across `W3 -> W4` (cutting `W4 vs W3` luma drift **`4.72x`**, `+71.96% -> +15.25%`), sharpens `W4` detail (`115.61`), and closes lips cleanly across `W4`.
  - *(Boundary detail visible in `.1fps.` `04:03..04:11` / `f243..f250`: `off` switches from studio to stage on `f243`, whereas both `whole_track` and `per_section` — whose anchored `W2` differed on its last `39` carried context frames — hold the studio close-up for `8` written frames `f243..f250` before cutting to stage at `f251`, explaining `W3` luma `39.312` including `8/102` bright studio transition frames.)*

---

## Section 2 — 2-Window Release Duration Check (`243 f = 10.125 s`, Single `studio` Section)

Direct single-variable check on `t2va_studio_dancer_close.txt` (`window_frames = 141`, `context_frames = 39`, `seam_taper` at `context_noise = 0.6`, `rms_dbfs = -22.06`, `peak_dbfs = -0.10`), alongside Bucket 2's unanchored `b2_song_taper_cn060` reference (`$H3_OUTPUT_DIR/Video/bucket2_probe_2026-11-16/b2_song_taper_cn060_00001.mp4`):

| Arm | `section_anchor` / `end%` | Wall Time (s) | `W1 -> W2` Luma (`% Delta`) | Seam `L1` (`f140 -> f141`) | Seam Jump Ratio | `W2` Motion `L1` | `W2` Sharpness (`Lap Var`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `b2_taper_off_ref` (Bucket 2 ref) | `off` / `—` | `132.14` | `140.170 -> 126.341` (`-9.87%`) | `3.0608` | `0.6725` | `6.6940` | `52.73` |
| `b3_2win_rel033` | `per_section` / `0.33` | `133.15` | `140.577 -> 128.175` (**`-8.82%`**) | `2.8223` (`-7.8%`) | `0.5302` | `7.2808` | `42.62` |
| `b3_2win_hold100` | `per_section` / `1.00` | `134.74` | `140.577 -> 128.300` (**`-8.73%`**) | `2.7498` | `0.5150` | `7.2256` | `47.77` |

### Visual Findings (`bucket3_2win_stacked.mp4` & `bucket3_2win_stacked.1fps.00000-00242.mp4`) & Bridge to Bucket 4

- **Default `anchor_end_percent = 0.33` confirmed (`Steps 0–1` of 6):** Compared to `1.00` full hold, `0.33` early release preserves **`91%`** of the `W2` luma improvement (`-9.87% -> -8.82%` vs `-8.73%`) and actually lowers `W2` seam jump ratio (`0.6725 -> 0.5302`, `seam_l1 = 2.8223`) without the mild chin-pull / edge-contrast tightening (`sharpness 47.77` vs `42.62`) visible when holding through `100%` of steps (`f144..f160`).
- **Why Bucket 4 follows directly:** In `studio` Window 2 at `1344x768`, an unweighted Frame-0 anchor contributes only `1,008` `ref_img` tokens out of `~46,000` packed tokens (`~2.2%` of total KV), competing against `12,096` drifted context tokens — so without explicit attention weight on `h3_section_anchor_spans`, `W2` studio brightness still sags **`-8.82%`**. **Bucket 4** solves this cleanly by adding mass-conserving `return_lse=True` subset attention balance across `h3_section_anchor_spans` so Window `2..N` queries can allocate a user-controlled attention mass share to the Frame-0 anchor without changing softmax entropy or requiring Qwen3-VL re-encodes.
