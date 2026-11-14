# MiniMaxH3EditGuider spatial/temporal mask A/B on the tortoise blizzard edit (2026-11-14)

lane: edit_guidance

Date: 2026-11-14. Code tested: `e89e1554` (`MiniMaxH3EditGuider.mask` in [`../../edit_guidance.py`](../../edit_guidance.py)) on Pika 6-step `t2va`, driven by [`../measure_edit_guider_mask_ab.py`](../measure_edit_guider_mask_ab.py) and [`../_lib/comfy_client.py`](../_lib/comfy_client.py).

*(No `verdict:` yet — generated clips await owner playback under `$H3_OUTPUT_DIR/Video/edit_guider_mask_ab_2026-11-14/`.)*

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

*(Note: After analyzing this run, [`../measure_edit_guider_mask_ab.py`](../measure_edit_guider_mask_ab.py) and [`../_lib/comfy_client.py`](../_lib/comfy_client.py) were updated with execution-start timeouts, `h3_prompt_edit` zero-VRAM import, `--resume`, multi-phrase `sam_prompt = "tortoise+candle"` unioned via `MaskComposite`, `streams = "independent"`, `audio_scale = 1.0`, and `ImageBlur(..., sigma = 0.45)` for future reruns; the measurements below reflect the exact `2026-11-14` run parameters above.)*

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

## 4. Open Questions & Next Candidates Pending Owner Playback

- **Owner playback files:** `$H3_OUTPUT_DIR/Video/edit_guider_mask_ab_2026-11-14/ab_2x2_stack.mp4` (`2x2` labeled stack), `ab_tortoise_crops_f000_f040_f080_f120.png`, and individual `*-audio.mp4` renders.
- **Hold `max`/`amax` vs. `avg` pooling in [`../../edit_guidance.py`](../../edit_guidance.py#L186-L212) open until playback:** On this `grow24` arm, `max`/`amax` outward expansion of the edit (`1.0`) eroded `eg_mask_tight`'s rim (`14.30%` `tight_rim` frost vs. `10.57%` in `grow24`) without hurting `grow24`; switching to `avg` would restore `InvertMask` symmetry (`avg(1 - m) == 1 - avg(m)`) at the trade-off of partial edit strength on `1`-cell features and small boundary bleed into `m = 0`. Wait for playback review before changing `edit_guidance.py`.
- **Optional rerun with updated defaults (`measure_edit_guider_mask_ab.py`):** If a locked `scout_base` stride and protected candle are needed on `t2va`, run a clean rerun tag with `--sam-prompt "tortoise+candle"` (unioned via `MaskComposite`), `--first-step 2` (`sigma_video = 0.923077`), `--streams independent --audio-scale 1.0`, and `--blur-sigma 0.45`.
- **Phase C scope ([`../../docs/wiki/next_steps.md`](../../docs/wiki/next_steps.md)):** Wait for owner playback of this `EditGuider.mask` A/B before designing `MiniMaxH3MaskedEncode` / `MiniMaxH3MaskedComposite`.
