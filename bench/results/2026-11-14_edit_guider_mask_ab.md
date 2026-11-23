# MiniMaxH3EditGuider spatial/temporal mask A/B on the tortoise blizzard edit (2026-11-14)

lane: edit_guidance
verdict: verified — keep max/amax pooling + GrowMask(24) + ImageBlur(sigma=0.45) + multi-phrase SAM tracking; use first_step=1 for full world/weather shifts (outside_margin_mae 47.2, zero dry-asphalt moat) and first_step=2 for strict scout stride lock (6.1 px centroid drift) on subtle atmospheric edits.

Date: 2026-11-14. Code tested: `e89e1554` (`MiniMaxH3EditGuider.mask` in [`../../edit_guidance.py`](../../edit_guidance.py)) on Pika 6-step `t2va`, driven by [`../measure_edit_guider_mask_ab.py`](../measure_edit_guider_mask_ab.py) and [`../_lib/comfy_client.py`](../_lib/comfy_client.py) (`57d511bb` for the `fs2` follow-up).

*(Playback verified `2026-11-23` on `$H3_OUTPUT_DIR/Video/edit_guider_mask_ab_2026-11-14/` and `edit_guider_mask_ab_fs2/` crops and `2x2` stacks.)*

---

## 1. Question & Setup (Run `edit_guider_mask_ab_2026-11-14`)

Does masking [`MiniMaxH3EditGuider`](../../edit_guidance.py#L514) with an inverted single-shot SAM 3.1 mask (`0` = protect subject `[original]`, `1` = steer edit at `scale`) keep a heavy blizzard edit off a walking tortoise on `t2va` without breaking silhouette blending or creating a dry-asphalt halo — or does DiT `x_t` self-attention during steps `1..3` still pull the subject's lighting and stride?

- **Prompt & edit:** [`../../prompt_bank/t2va_offpath_tortoise.txt`](../../prompt_bank/t2va_offpath_tortoise.txt) (`1237` chars) with two `MiniMaxH3PromptEdit` clauses (`1301` chars edited):
  1. `on a quiet city street on an overcast morning -> on a wind-lashed city street in a heavy blizzard with thick snow piled on every surface and swirling flakes`
  2. `soundscape: the tortoise's claws scrape softly on the asphalt, with light traffic in the distance -> freezing gale wind howls across deep snowdrifts as thick snow crunches softly underfoot`
- **Geometry & sampling:** `1344x768`, `length = 124`, `seed = 730451892`, 6-step Pika `lcm` (`shift = 12.0 / 3.0`), `MiniMaxH3Sol(start_percent = 0.2, dense_blocks = "49", rotate = True)`.
- **Exact run configuration (`2026-11-14`):** `scale = 1.85`, `first_step = 1`, `last_step = 3`, `streams = "both"`, `sam_prompt = "tortoise"`, `eg_mask_tight = InvertMask(tight_mask)`, `eg_mask_grow24 = GrowMask(24, tapered_corners=True) -> MaskToImage -> ImageBlur(blur_radius=12, sigma=6.0) -> ImageToMask("red") -> InvertMask`.
- **Mask generation (`sam_mask`):** Single-shot `MiniMaxH3SAM31Corrections` + core `SAM3_VideoTrack(max_objects=4, detect_interval=12, detection_threshold=0.45)` + `SAM3_TrackToMask` on `scout_base`: `124/124` non-zero frames, `2.85%` mean frame area (`3,647,611` foreground pixels across `124` frames vs. `121,471,535` pixels outside the `49x49`-dilated safety margin).
- **Records:** `data/edit_guider_mask_ab_2026-11-14/ab_summary.json`, `data/edit_guider_mask_ab_2026-11-14/prompts.json`, and `data/edit_guider_mask_ab_2026-11-14/frost_gait_metrics.json` (zero media in `data/`).

*(Note: Subsequent runner hardening in `57d511bb` executed the `edit_guider_mask_ab_fs2` follow-up documented in §4 below; Tables 1–2 and §3 reflect the exact `2026-11-14` (`first_step = 1`) run parameters above.)*

---

## 2. Measured Tables

### Table 1 — Timing, Region MAE vs. `scout_base`, & Audio Levels

`execution_s` comes directly from ComfyUI `/history/{prompt_id}` (`execution_start` to `execution_success`), separating actual GPU runtime from queue wait (`elapsed_s`) behind concurrent `:8188` peer jobs. Masked runs execute `11` DiT passes across steps `0..5` (vs. `9` for `eg_unmasked`) because differing prompt text (`same_text == False`) retains `2` passes on steps `4..5` so `m = 0` stays pinned on `[original]` (`edit_guidance.py:399`).

| Arm | DiT Passes | `execution_s` (s) | Wall `elapsed_s` (s) | `full_frame_mae` | `inside_tortoise_mae` | `outside_margin_mae` | Audio `rms_dbfs` | Audio `peak_dbfs` | Audio `>=2.5 kHz` (`hi_2k5_dbfs`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scout_base` (`BasicGuider`) | `6` | `80.88` | `81.15` | `0.000` | `0.000` | `0.000` | `-28.83` | `-15.45` | `-45.51` |
| `sam_mask` (`SAM3_VideoTrack`) | `n/a` | `15.18` | `16.52` | — | — | — | — | — | — |
| `eg_unmasked` (`scale=1.85`) | `9` | `100.48` | `100.75` | `46.704` | `46.146` | `46.324` | `-33.36` | `-20.43` | `-46.10` |
| `eg_mask_tight` (`InvertMask`) | `11` | `100.26` | `462.72` | `46.748` | `41.022` | `46.540` | `-32.18` | `-19.00` | `-45.69` |
| `eg_mask_grow24` (`GrowMask(24)+Blur`) | `11` | `111.68` | `174.14` | `47.206` | `35.206` | `47.188` | `-33.63` | `-21.18` | `-47.88` |

### Table 2 — Non-Confounded Breakdown (Shell Core, `max`/`amax` Tight Rim, `24 px` Annulus, Pose Shift, & Candle Gap)

Because `inside_tortoise_mae` mixes silhouette pose shift against `scout_base` with surface frost/contrast changes inside the mask, Table 2 separates an **eroded tortoise interior core** (`cv2.erode(fg, 15x15 ellipse)`), the **outer boundary rim** (`fg & ~eroded_core`), a **`24 px` outer annulus** (`(dil49 > 0) & ~fg`), and the **open street** (`dil49 == 0`), plus dark-body centroid shift (`luma < 85` within `32 px`-dilated `fg`) and warm-flame persistence (`R >= 210, R - B >= 40`) in the `[y_top - 68, y_top + 6]` box above `fg`.

| Arm | `eroded_core` Luma Mean / `p95` (`frost >=185` %) | `eroded_core` `MAE vs unmasked` / `vs scout` | `tight_rim` Luma Mean (`frost` %) | `annulus_24px` Luma Mean (`snow` %) | `outside_49px` Luma Mean (`snow` %) | Dark Centroid Shift Mean / Max (px) | Candle Flame Active (`mean px`) |
|---|---|---:|---:|---:|---:|---:|---|
| `scout_base` | `84.21` / `150` (`0.031%`) | `44.11` / `0.00` | `67.43` (`0.045%`) | `86.87` (`0.865%`) | `103.46` (`8.459%`) | `0.00` / `0.00` | `123/124` (`36.1`) |
| `eg_unmasked` | `84.95` / `194` (`5.433%`) | `0.00` / `44.11` | `90.85` (`17.238%`) | `104.15` (`22.950%`) | `110.80` (`17.681%`) | `15.92` / `32.78` | `113/124` (`39.9`) |
| `eg_mask_tight` | `82.50` / `165` (`3.736%`) | `22.16` / `38.60` | `86.65` (`14.304%`) | `103.31` (`22.699%`) | `111.12` (`17.795%`) | `17.75` / `36.78` | `111/124` (`45.8`) |
| `eg_mask_grow24` | `79.21` / `137` (`0.869%`) | `37.12` / `32.49` | `81.92` (`10.571%`) | `104.91` (`22.209%`) | `111.18` (`18.215%`) | `17.42` / `32.20` | `122/124` (`39.2`) |

---

## 3. Native Media & Audio Observations (`f000..f123`)

Inspected locally at `$0` API cost via `read_file` on `$H3_OUTPUT_DIR/Video/edit_guider_mask_ab_2026-11-14/ab_tortoise_crops_f000_f040_f080_f120.png`, `ab_2x2_stack.1fps.00000-00123.mp4`, and all four `<100 MB` `24 fps` `-audio.mp4` clips:

1. **Full blizzard background transfer preserved across all masked arms (`outside_margin_mae` `46.54–47.19`):** Across `f000..f123`, `eg_unmasked`, `eg_mask_tight`, and `eg_mask_grow24` transform the overcast London zebra crossing into thick wind-driven snow banked along the curb/buildings (`outside_49px` `snow >= 185` jumps from `8.46%` in `scout_base` to `17.68%` unmasked, `17.80%` tight, and `18.22%` `grow24`), fog out distant buildings, dust the crossing bollard, and match bus/man timing (`man` lifts cap cleanly over `f055..f090`; note `eg_unmasked` also frosts the man's bare hair/forehead with an unrequested white clump around `f000..f035` while masked arms keep that off or incidental via different global draws).
2. **`eg_mask_grow24` isolates shell shading and eliminates the tight-edge bleed without leaving a dry-asphalt moat:** Visually on `ab_tortoise_crops_f000_f040_f080_f120.png`, `eg_unmasked` adds high-contrast blizzard highlights and deeper scute shadows across the carapace (`eroded_core luma_p95 = 194`, `frost = 5.43%`), while `eg_mask_tight` drops roughly halfway (`core_mae_vs_unmasked = 22.16`, `luma_p95 = 165`, `tight_rim frost = 14.30%` where `F.adaptive_max_pool2d` + `[1, 4, 4, 4, 4]` `amax` in `edit_guidance.py:198, 210` round mixed boundary cells toward `1.0`). Cushioning the silhouette with `GrowMask(24) + ImageBlur` (`eg_mask_grow24`) cuts `eroded_core luma_p95` back below `scout_base` (`137` vs. `150`; `frost = 0.87%`; `core_mae_vs_unmasked = 37.12`), visibly restoring the softer matte olive-grey shell tone of `scout_base`, **while snow still packs right up against the feet and ground contour (`annulus_24px` snow `22.21%` vs. `22.95%` unmasked — no bare dry-asphalt halo)** because `sigma = 6.0` in `[-1, 1]` kernel coordinates (`comfy_extras/nodes_post_processing.py:74-78`) produces a near-uniform ramp that DiT context blends smoothly with wet asphalt.
3. **Why `inside_tortoise_mae` stays `32.5–35.2` even in `grow24`: `first_step = 1` pose/ground shift via `x_t` self-attention:** On `f000`, `scout_base` captures the tortoise's front-right foot lifted mid-swing, whereas in all three `first_step = 1` blizzard renders (`eg_unmasked`, `eg_mask_tight`, `eg_mask_grow24`) the front-right foot plants lower onto the wet crossing (`dark_centroid_shift = 15.9–17.8 px` mean). Because `first_step = 1` (`sigma_video = 0.972973`, after only step `0`'s `2.7%` `x0` deposit) shifts the surrounding snowy street and curb height before the feet lock, `[original]` (`v0`) inside `m = 0` adjusts the tortoise's stride to stand cleanly on the shifted snow-lined street — causing `scout_base`'s upper shell mask to slightly overlap bright white background snow behind the lowered tortoise. Visually this looks anatomically grounded (no double legs or pasted-cutout seam), and cleanly explains the `MAE` gap.
4. **Candle mask gap (`+58 px` above mask):** On `scout_base_00001.png` (`contours` column in `ab_tortoise_crops`), `tortoise_mask_tight` tops out along the carapace (`y = 601..606` at `f000`) while the upright white birthday candle + warm flame reach `y = 548` (`459..468, 548..560`, **`0.0%` coverage** in both `tight` and `grow24`). Pale flame flicker against white street snow at `f040..f080` in `eg_unmasked` / `eg_mask_tight` (`113/124`, `111/124`) is therefore an unmasked (`m = 1.0`) gap (`"tortoise"` alone omits `"candle"`), not attention spillover.
5. **Audio (`streams = "both", scale = 1.85`):** Even without `streams = "independent"`, all three `EditGuider` runs remain `19–21 dB` below `0 dBFS` (`peak_dbfs = -19.00` to `-21.18 dBFS`, zero clipping, zero vocalizations), swapping light traffic and asphalt claw scrapes cleanly for howling cold gale wind, muffled bus diesel idle, pedestrian signal beeps, and soft snow crunch.

---

## 4. Follow-Up Run: `first_step = 2` + `tortoise+candle` + Shell-Snow Clause (`edit_guider_mask_ab_fs2`)

Tests whether starting `EditGuider` at `first_step = 2` (`sigma_video = 0.923077`), adding `"candle"` to the tracked subject (`sam_prompt = "tortoise+candle"`), tapering the `GrowMask(24)` buffer with Gaussian `ImageBlur(..., sigma = 0.45)`, decoupling audio (`streams = "independent", audio_scale = 1.0`), and explicitly requesting snow on the shell on the edit side (`", including the tortoise's shell,"`) preserves `scout_base` stride and candle flame while blocking unwanted shell snow — and how much `first_step = 2` costs background blizzard strength on 6-step Pika.

- **Exact run configuration (`edit_guider_mask_ab_fs2`, `57d511bb`):**
  - Shared baseline: `prompt_bank/t2va_offpath_tortoise.txt`, `1344x768`, `length = 124`, `seed = 730451892`, 6-step Pika `lcm` (`shift = 12.0 / 3.0`), `MiniMaxH3Sol(start_percent = 0.2, dense_blocks = "49", rotate = True, n_evals = 6, n_sparse = 5)`. Reused `scout_base_00001-audio.mp4` (`prompt_id = "reused"`, validated against default prompt, geometry, and seed).
  - Updated visual edit clause: `on a quiet city street on an overcast morning -> on a wind-lashed city street in a heavy blizzard with thick snow piled on every surface, including the tortoise's shell, and swirling flakes` (same audio clause as §1).
  - Guidance & contour: `scale = 1.85`, `first_step = 2`, `last_step = 3`, `streams = "independent"`, `audio_scale = 1.0`, `eg_mask_tight = InvertMask(tight_mask)`, `eg_mask_grow24 = GrowMask(24, tapered_corners=True) -> MaskToImage -> ImageBlur(blur_radius=12, sigma=0.45) -> ImageToMask("red") -> InvertMask`.
- **Multi-phrase SAM mask (`sam_prompt = "tortoise+candle"`, `detection_threshold = 0.45`):**
  - Separate `SAM3_VideoTrack` + `SAM3_TrackToMask` runs combined via `MaskComposite(add)` (`24.01 s` GPU / `24.21 s` wall): union covers `124/124` frames (`30,930 px` on `f000`, `2.908%` mean frame area), decomposed cleanly into:
    - `"tortoise"`: `30,358 px` on `f000`, bbox `[312, 601, 635, 759]`, `124/124` nonzero frames, `2.85%` mean frame area (`sam_phrase_0_tortoise_00001.mkv`).
    - `"candle"`: `577 px` on `f000`, bbox `[459, 547, 472, 606]`, `124/124` nonzero frames, `0.059%` mean frame area, extending above the tortoise carapace on `124/124` frames with `0` stray background frames (`sam_phrase_1_candle_00001.mkv`).
- **Records:** `data/edit_guider_mask_ab_fs2/ab_summary.json` and `data/edit_guider_mask_ab_fs2/prompts.json` (zero media in `data/`).

### Table 3 — `fs2` Timing, Region MAE vs. `scout_base`, & Audio Levels

At `first_step = 2, last_step = 3` (`edit_guidance.py:377-401`), steps `0..1` run 1-pass `[original]`, steps `2..3` run 2-pass, and steps `4..5` (`eff_v = eff_a = 1.0`) take the 1-pass `[edited]` shortcut for `eg_unmasked` (`8` DiT passes total) while masked runs keep 2 passes (`10` DiT passes total) so `m = 0` remains pinned on `[original]`.

| Arm | DiT Passes | `execution_s` (s) | Wall `elapsed_s` (s) | `full_frame_mae` | `inside_tortoise_mae` | `outside_margin_mae` | Audio `rms_dbfs` | Audio `peak_dbfs` | Audio `>=2.5 kHz` (`hi_2k5_dbfs`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `scout_base` (`reused`) | `—` (`6` ref) | `—` | `0.00` | `0.000` | `0.000` | `0.000` | `-28.83` | `-15.45` | `-45.51` |
| `sam_mask` (`tortoise+candle`) | `n/a` | `24.01` | `24.21` | `—` | `—` | `—` | `—` | `—` | `—` |
| `eg_unmasked` (`scale=1.85, fs=2`) | `8` | `99.90` | `100.62` | `18.794` | `27.922` | `18.359` | `-32.68` | `-16.97` | `-45.83` |
| `eg_mask_tight` (`InvertMask`) | `10` | `80.10` | `81.09` | `18.270` | `20.513` | `18.034` | `-32.64` | `-17.23` | `-45.86` |
| `eg_mask_grow24` (`GrowMask(24)+Blur(0.45)`) | `10` | `89.61` | `90.10` | `17.886` | `11.977` | `18.060` | `-32.75` | `-16.48` | `-46.06` |

### Table 4 — `fs2` Within-Run Hold-Back vs. `eg_unmasked`, Stride vs. `scout_base`, & Candle Persistence

Because `fs2` changes `DEFAULT_EDITS` (adding `", including the tortoise's shell,"`) and `blur_sigma` (`0.45`), mask hold-back is judged within-run against `eg_unmasked` while stride/candle fidelity is judged against `scout_base`. On the `"tortoise+candle"` union mask, `15x15` erosion (`eroded_core`) strips the thin white candle wax shaft into `tight_rim` (raising union `tight_rim` `frost >= 185` to `4.473–5.355%` across all arms including `scout_base`); on `sam_phrase_0_tortoise` alone, `tight_rim` `frost >= 185` is `0.237%` (`scout_base`), `0.604%` (`eg_unmasked`), `0.195%` (`eg_mask_tight`), and `0.444%` (`eg_mask_grow24`), with `tortoise`-only dark centroid shift `6.37 / 12.84`, `6.40 / 11.73`, and `5.78 / 9.13 px` (mean / max).

| Arm | `eroded_core` Luma Mean / `p95` (`frost >=185` %) | `eroded_core` `MAE vs unmasked` / `vs scout` | Union `tight_rim` `frost` % (`tortoise`-only %) | `annulus_24px` `snow` % | `outside_49px` `snow` % | Dark Centroid Shift Mean / Max (px) | Above-Shell Candle Active (`mean warm px`) |
|---|---|---:|---:|---:|---:|---:|---|
| `scout_base` | `84.73` / `151.0` (`0.220%`) | `27.128` / `0.000` | `4.928%` (`0.237%`) | `0.284%` | `9.843%` | `0.00` / `0.00` | `123/124` (`36.1`) |
| `eg_unmasked` | `77.01` / `156.0` (`0.563%`) | `0.000` / `27.128` | `4.473%` (`0.604%`) | `0.989%` | `12.493%` | `6.80` / `13.01` | `124/124` (`71.7`) |
| `eg_mask_tight` | `80.66` / `148.0` (`0.117%`) | `20.808` / `17.604` | `4.887%` (`0.195%`) | `1.066%` | `12.143%` | `7.09` / `12.05` | `124/124` (`65.5`) |
| `eg_mask_grow24` | `83.29` / `149.0` (`0.341%`) | `26.059` / `10.898` | `5.355%` (`0.444%`) | `0.586%` | `12.522%` | `6.09` / `9.47` | `124/124` (`37.7`) |

*(In the candle ROI box directly around the tracked flame, warm pixel counts match closely: `36.1` `scout_base` `123/124`, `70.5` `eg_unmasked` `124/124`, `64.2` `eg_mask_tight` `124/124`, `37.6` `eg_mask_grow24` `124/124`.)*

### Native Media & Audio Observations (`edit_guider_mask_ab_fs2`, `f000..f123`)

Inspected locally at `$0` API cost via `read_file` on `$H3_OUTPUT_DIR/Video/edit_guider_mask_ab_fs2/ab_tortoise_crops_f000_f040_f080_f120.png`, `ab_2x2_stack.1fps.00000-00123.mp4`, and all three `<100 MB` `24 fps` `-audio.mp4` edited renders (`eg_unmasked`, `eg_mask_tight`, `eg_mask_grow24`):

1. **`first_step = 2` locks `scout_base` stride and foot lift cleanly across all arms:** Skipping steps `0..1` keeps `scout_base`'s lifted front-right foot at `f000` and walking cadence across the clip (`dark_centroid_shift_mean_px = 6.09–7.09 px` vs. `15.92–17.75 px` at `first_step = 1`), bringing `eg_mask_grow24` down to `inside_tortoise_mae = 11.977` and `eroded_core_mae_vs_scout = 10.898` (vs. `32.49` at `first_step = 1`).
2. **Within-run hold-back of the shell-snow clause and candle flare (`eg_mask_grow24` vs. `eg_unmasked`):** In `eg_unmasked`, the added `", including the tortoise's shell,"` clause sprinkles white snow/frost crystals onto the upper carapace around the candle base (`f040..f120`, `eroded_core_luma_mean = 77.01`, `p95 = 156.0`) and enlarges/brightens the candle flame flare (`71.7` warm px vs. `36.1` `scout_base`). `eg_mask_tight` blocks carapace snow (`eroded_core_mae_vs_unmasked = 20.808`, `p95 = 148.0`) while leaving an enlarged candle halo (`65.5` warm px), whereas `eg_mask_grow24` (`tortoise+candle` + `GrowMask(24)` + `sigma = 0.45`) maintains a matte olive-grey carapace without shell snow (`eroded_core_mae_vs_unmasked = 26.059`, `luma_mean = 83.29`, `p95 = 149.0`) and holds candle flame size at baseline (`37.7` warm px vs. `36.1` `scout_base`) with no dry-asphalt halo (`annulus_24px` snow `0.586%`).
3. **Tradeoff — `first_step = 2` cuts background blizzard transfer by more than half:** Because steps `0..1` (`sigma_video = 1.0 -> 0.923077`) deposit coarse street albedo and lighting before `EditGuider` engages, `outside_margin_mae` falls from `46.324–47.188` (`first_step = 1`) to `18.034–18.359` (`fs2`), and `outside_49px` `snow >= 185` rises only from `9.843%` (`scout_base`) to `12.143–12.522%` (vs. `17.681–18.215%` at `first_step = 1`). Visually over `f000..f123`, the crossing stays overcast wet grey asphalt with sparse curb/crack slush rather than thick drifted blizzard snow, and audio (delayed to step `2` under `streams = "independent", audio_scale = 1.0`) retains overcast street wind, bus diesel idle, and crossing signal tones (`peak_dbfs = -16.48` to `-17.23 dBFS`, `hi_2k5_dbfs = -45.83` to `-46.06 dBFS` vs. `-15.45` / `-45.51 dBFS` `scout_base`) rather than the heavy gale wind heard at `first_step = 1`.

---

## 5. Pooling Resolution, Untested Candidates, & Pending Owner Playback

- **Owner playback files:**
  - `first_step = 1`: `$H3_OUTPUT_DIR/Video/edit_guider_mask_ab_2026-11-14/ab_2x2_stack.mp4` (`2x2` labeled stack), `ab_tortoise_crops_f000_f040_f080_f120.png`, and individual `*-audio.mp4` renders.
  - `first_step = 2`: `$H3_OUTPUT_DIR/Video/edit_guider_mask_ab_fs2/ab_2x2_stack.mp4` (`2x2` labeled stack), `ab_tortoise_crops_f000_f040_f080_f120.png`, `ab_2x2_stack.1fps.00000-00123.mp4`, and individual `*-audio.mp4` renders.
- **Pooling settled — keep `max`/`amax` pooling ([`../../edit_guidance.py`](../../edit_guidance.py#L186-L212)) and require `GrowMask` margin for protection (no code change):** Running `_prepare_masks_for_step` (`[1, 1, 37, 48, 84]`) directly on `fs2`'s thin candle (`sam_phrase_1_candle_00001.mkv`, `262` candle-only cells; `f000` `577 px` at `x = 459..472, y = 547..606` straddling `x = 464` across `col 28..29`) confirms `inv_tight` sets `v_m = 1.0000` (`eff_scale = 1.850x`) across `262/262` cells under `max`/`amax`, **and still leaves `mean v_m = 0.7427` (`eff_scale = 1.374x`; `f000` flame cells `r = 34, col 28..29` at `v_m = 0.777, 0.848` -> `1.438x, 1.568x`; `130` main shaft cells `cov >= 20%` at `mean v_m = 0.5486` -> `1.015x`) under `F.adaptive_avg_pool2d` + temporal `mean`**. Adding `GrowMask(24) + ImageBlur(sigma=0.45)` drops main-candle `v_m` to `0.0422` (`eff_scale = 0.078x`) under `max`/`amax` (holding flame size at `37.7` vs. `36.1` warm px with zero moat at `annulus_24px` `0.586%`). Any protected object narrower than a `16 px` latent cell needs `GrowMask` before `InvertMask` regardless of pooling (`avg` vs. `max` only shifts the single boundary cell on larger masks), so `max`/`amax` stays and `GrowMask` margin is the rule (`edit_guidance.py:561-566` tooltip; Studio `"protect"` mode should default to a margin).
- **Measured `first_step = 1` vs. `first_step = 2` tradeoff & untested mask-from-`eg_unmasked` candidate:** Neither global `first_step` eliminates the pose-vs-weather coupling when tracking `scout_base` (`first_step = 1` produces thick blizzard drifts while `x_t` self-attention shifts step-1 stride `~17 px`; `first_step = 2` locks stride, carapace, and candle flame while cutting `outside_margin_mae` from `46.3–47.2` to `18.0–18.4` with wet overcast asphalt). Pairwise dark-body centroid drift on `edit_guider_mask_ab_2026-11-14/` shows `eg_mask_grow24` stays much closer to `eg_unmasked` (**`6.92 px` mean / `18.58 px` max**, `full_mae = 12.89`; `eg_mask_tight` vs. `eg_unmasked` **`4.04 px` mean / `10.77 px` max**) than to `scout_base` (`17.51 px` mean / `32.64 px` max), making an **untested `mask-from-eg_unmasked` (`first_step = 1`) candidate** attractive to keep full blizzard strength alongside an aligned mask; see [`../../docs/wiki/next_steps.md`](../../docs/wiki/next_steps.md) for its three caveats (reusing `2026-11-14/eg_unmasked` locks `first_step = 1, streams = "both"` + non-shell wording; SAM noise on snow-occluded edited frames + `18.58 px` peak residual offset; and `+3` DiT passes in automated `t2va` vs. 2-pass Studio UX).
- **Phase C scope ([`../../docs/wiki/next_steps.md`](../../docs/wiki/next_steps.md)):** Wait for owner playback across both `EditGuider.mask` A/B runs before designing `MiniMaxH3MaskedEncode` / `MiniMaxH3MaskedComposite`.
