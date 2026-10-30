# Upstream Data-Dependency, Outlier, and Alignment Diagnostic Reference

## 1. Trace & Align Every Upstream Input in the DAG (Explicit & Hidden Dependencies)

Every masked V2V render depends on a multi-stage upstream chain (documented in [`docs/masked_v2v/chain_trace_and_remakes.md`](chain_trace_and_remakes.md) §1):
`source video` -> `preview (shot_c_*)` -> `strangers.py` + `part_saved_on_track` + `measure_moge2_depth.py (--depth)` -> `pose_along_shot.py` -> `head_reference.py` -> `load_text.py` -> `capture_masked_run.py (first)` -> `plan` -> `on-plan` -> `render` -> `land`.

### Hidden / Unsynced Cross-Model Dependencies You Must Check (`2026-10-28` Audit)

Several upstream inputs come from **separate models that do not automatically lock to each other**, or write diagnostic JSON fields marked `"decides": "nothing"` unless you explicitly act on them:

| Upstream Input / JSON Record | Producer -> Consumer | What Can Go Out of Sync or Be Ignored | Action Required (and Reprocessing Scope) |
|---|---|---|---|
| **3D Body Mesh (`_mesh_00001.mp4`) vs. 2D Face Landmarks (`_points.npz`) & `_pose_table.json`** | `preview` (`SAM3DBody`) + `bench/pose_along_shot.py` (`Sapiens2-Pose`) -> `bench/head_reference.py` (`<Video 1>`) | `bench/head_reference.py` draws 2D `Sapiens2-Pose` white face lines at film `(x, y)` directly on top of the 3D `SAM3DBody` grey mesh, and records 3D/2D collisions in `marks_against_the_body_model.flagged_frames` (`nose_from_body_model_fh >= 0.5`, `eye_roll_diff_deg >= 15°`, `head_pitch_down_3d >= 0.8`; see the replay and marks scans under `data/fun/cranium_crown_and_marks_scan_fun_2026-10-29T0301Z.json` and `data/fun/`), where the 3D head lags/flips or 2D lines land on the neck/air. Yet `bench/head_reference.py` sets `"decides": "nothing"`. | Inspect `marks_against_the_body_model.flagged_frames` and `_sheet.jpg`. Pass `--no-head-on FIRST-LAST=WHY` (where the 3D mesh flips/jumps) or `--no-draw-face-on FIRST-LAST=WHY` (where 2D lines drift off the 3D head), then **reprocess `reference` -> `text` -> `first` -> `plan` -> `on-plan` -> `render`**. |
| **3D Grey Arms/Hands Inside `--crop cranium` (`grey_head` in `bench/crop_reference_to_head.py`)** | `bench/crop_reference_to_head.py` -> `bench/head_reference.py` | Under `dc07ccee`, `cranium_box()` bounds `grey_head` to `win[k]` (SAM 3.1 `head_track` grown by `0.50 * face` on each side) when sizing the `[x0, y0, x1, y1]` crop box, so raised 3D arms outside `win[k]` no longer stretch the box. However, `<Video 1>` still crops the full-body `_mesh_00001.mp4`, so 3D grey hands directly in front of the face inside `win[k]` (`f312–314`, `f377` on `0294`) remain visible in the crop alongside the white 2D mouth/eye lines. | Check `hand_not_refined` / raised-hand frames on `_sheet.jpg`. If a 3D grey hand covers the mouth/eyes on `<Video 1>`, check whether it blocks transfer in the render. |
| **Off-Track or No-Track Frames in `--crop cranium` (`bench/crop_reference_to_head.py::cranium_box`)** | `bench/pose_along_shot.py` (`_points.npz`) + SAM 3.1 `head_track` -> `bench/crop_reference_to_head.py::cranium_box` | Pre-`dc07ccee` references checked only `seen = np.any(hd)`, allowing an occlusion that placed 2D face landmarks on a passer-by or background (`104.5 px` vs `48.8 px` median on `0294` frame `317`; `436.8 px` vs `49.7 px` on `5044` frame `5044`) to blow out the whole-shot reference crop (`1.517x` instead of `2.674x` on `0294`; `1.111x` instead of `2.798x` on `5044`). From `dc07ccee` onward, `cranium_box()` requires 4-corner `face_box` containment inside `win[k]` and records `frames_off_the_head_track`, `frames_with_no_head_track`, and `frames_over_the_face_scale`. | Check `head_track_used`, `frames_off_the_head_track`, `frames_with_no_head_track`, and `frames_the_grey_head_reaches_the_box_edge` in the reference JSON; pass `--no-head-on` / `--no-face-on` / `--drop` for occluded spans so `<Video 1>` holds the last clean 3D mesh and omits bogus 2D lines. |
| **MoGe-2 Per-Pixel Depth (`depth.npz` via `bench/measure_moge2_depth.py`)** | `shot_to_load.py job --depth` -> `capture_masked_run.py files --depth` | `--depth` was historically an opt-in CLI flag (`has_depth = False` on pre-`2026-10-28` runs; `data/fun/plan.json` now sets `"depth": {"on": true}` for `lead`, `second`, and `third`). Moreover, `shot_to_load.py::overlap_frames_of()` only runs MoGe-2 on frames where bounding boxes overlap by `>= 50 px`, which can omit adjacent `hers`-marked frames unless checked. | Ensure `depth.on = true` in `plan.json` (or pass `--depth` on `shot_to_load.py job`). Check `depth.npz` `measured_frames` against `head_marks.hers` and `step_carves_summary` in `capture_manifest.json`. Adding `--depth` or changing `measured_frames` requires **reprocessing `job` -> `first` -> `plan` -> `on-plan` -> `render`**. |
| **`strangers` Raw SAM2 `c==0` Halo vs. Subject Hair Margin & `hers` (`m == 1`) Marks** | `bench/strangers.py` (`426638a6`) + `data/fun/plan.json` (`head_marks`) -> `bench/capture_masked_run.py` | Commit `426638a6` switched `strangers` from Facer class-filtered masks (`0%` `c==0`) to the raw SAM2 union (`38.2%` `c==0` background halo on `4912–4950`). Before `ba86fbe9`, `c==0` was not in `HAIR_CLASSES = (4, 14)` and `head_marks` (`m == 1` `hers`) only gated `[1] head_track`, not `[3] strangers` subtract—so `c==0` halo dilated by `strangers_margin` (`14 px`) and carved the subject's `10 px` hair margin across unoccluded frames plus her visible crown behind foreground hats. | Fixed globally in `ba86fbe9`: `keep_c0` on `strangers` is restricted to `(subject_classes == 0) | ~np.isin(subject_classes, [2, 3, 4])`, and `m == 1` (`hers`) frames with MoGe-2 depth retain subject `[2, 3, 4]` where `strangers_depth >= subject_depth - 0.015`. Always inspect `step_carves_summary` in `capture_manifest.json` (`strangers.carved_units_total` and per-step `S0..Sn` carves) to catch unexpected perimeter or crown carving before rendering. |
| **Interior `not seen` Spans in `shots.json` (`who[subject].seen_looks`)** | `shots.json` (`shots_look.py`) -> `shot_to_load.py job` | `shot_to_load.py` (`line 2359`) only checks `seen_looks` for start/end partial-shot bounds; it does **not** automatically populate `--drop` or `--no-face-on` for interior `"is": "not seen"` spans. | Always check `shots.json` `who[subject].seen_looks` before running `job` and pass every interior `not seen` span to `--drop` and `--no-face-on` (or `--no-head-on`). |

1. **Stale Cache & Missing Dependency Links**:
   - Check every input file's `writer_stamp` (`git_commit`, `stamped_at`) or filesystem mtime against the latest commit touching its writer script.
   - As of `2026-10-28`, any preview predating `2026-10-27` (`6b0ea9a6`, `head_mode = "per stretch"` default + SAM `--people` verification) or `strangers` predating `2026-10-27` (`426638a6`, unowned SAM people union) is stale cache and must be regenerated via `bench/shot_to_load.py preview ... --remake --queue`.
   - On multi-subject shots where an earlier pass is kept in `data/fun/laid.jsonl`, `--source` **must** be the newest whole-video cut containing all kept earlier passes on that shot, never the raw source or an older cut.
2. **Exact Frame-Window & Start-Frame Alignment**:
   - Older loads on a shot frequently covered a slightly narrower span (e.g., `lead_fp_0294d` on `2026-10-13` covered `120` frames `294–413` with `414` left out, whereas the shot is `121` frames `294–414`; `lead_fp_4053c` covered `4053–4097` of `4053–4126`).
   - Verify that every `.mkv`, `.mp4`, and `.npz` file feeding a job has the exact same start frame (`first`) and frame count (`frames`):
   ```bash
   python3 -c '
   import cv2, glob, json, os
   j = json.load(open("data/<day>_shot_c_<shot>_job/job.json"))["job"]
   print("Preview job first/frames:", j["first"], j["frames"])
   for f in sorted(glob.glob("<output_dir>/Video/fun/*/shot_c_<shot>/*.m*") +
                   glob.glob("<output_dir>/Video/fun/keep_outs/shot_c_<shot>/*.mkv") +
                   glob.glob("<output_dir>/Video/fun/parts/shot_c_<shot>/*.mkv")):
       cap = cv2.VideoCapture(f)
       print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "frames |", os.path.basename(f))
   '
   ```

---

## 2. Interrogate Summary Statistics for Outliers (`Least / Median / Most`)

Even when `reference` and `first` exit `0` with `0 likely to fail`, summary statistics reveal when a few bad frames are corrupting whole-shot conditioning:

### Case Study A — Reference Crop Blown Out by a 3-Frame Occlusion (`lead_fp_0294c`, `294–414`)
- **What the numbers showed**:
  - `pose_along_shot.py` printed: `face {'least': 34.3, 'median': 48.8, 'most': 389.3} px tall`.
  - `head_reference.py` (`--crop cranium`) printed:
    - `each_frame_alone_would_enlarge_least_median_most`: `[1.613, 5.721, 5.721]`
    - `the_one_box_enlarges`: **`1.517`** (`the_travel_keeps_of_the_median_frame_enlargement: 0.265`)
    - `sheet.why`: `317: inner_lip_gap_px briefly over 3 of 1 1.149>50.625>0.236`
- **What visual inspection on disk (`0294_ref_312_359.jpg`) revealed**:
  - On frames `317–319` (`3` frames), a foreground passer-by in a white baseball jersey walked directly in front of the lead.
  - Because `317–319` was not excluded, Sapiens Pose placed a `389 px` bogus face across the passer-by (`inner_lip_gap_px = 50.6 px`) and the 3D body mesh flipped upside down on `317` (feet in the air!) and jumped on `318–319`.
  - To fit that 3-frame glitch into one static crop box across all `121` frames, `crop_reference_to_head.py` zoomed the reference out from **`2.592x` to `1.517x`**, shrinking her head and mouth in `<Video 1>` across the entire shot!
- **The fix (`bench/head_reference.py`)**:
  - Passing `--no-head-on "317-319=<who> looked at <sheet.jpg>: ..."` and `--no-face-on "350-357=..."` + `--drop 317-319+350-357` held the last clean 3D body mesh (`316`) across `317–319` and added `317–319` and `350–357` to `crop_leave_out`, immediately restoring `the_one_box_enlarges` from **`1.517x` to `2.592x`** (`BOX: [431, 64, 826, 360]`).

### Case Study B — Stale Carried `--drop` Spans Discarding Good Frames (`lead_fp_0294d` -> `lead_fp_0294c`)
- **What happened**:
  - Wave 0 `lead_fp_0294d` (`2026-10-13`, built before SAM `head_track` and `strangers.py`) dropped `28` frames (`311–319`, `349–360`, `408–414`), leaving her real face unswapped on `28` of `121` frames.
- **What re-evaluating with `--no-carry-words` on fresh `shot_c_0294b` showed**:
  - On `311–316`, `349`, `358–360`, and `408–414` (`17` frames), her face is clearly visible and cleanly segmented once `head_track` (`per stretch`), `head_tube_parts.py`, and `strangers.py` are active. Only `11` frames (`317–319` behind the passer-by and `350–357` turned away) are genuinely hidden.

### Case Study C — Whole-Shot Zoom Suppressed by Lateral Subject Travel (`516–599`)
- **What the numbers showed**:
  - On `516–599` (`84` frames), `lead`'s face is small (`~35 px` tall whole-frame, `1.1` tokens). Yet a single continuous load (`lead_fp_0516e`) rendered at **`1.0x` whole-frame** (`[0, 0, 1024, 768]`) even with `render_region: part`.
- **Why**:
  - Because `video_mask.zoom_box` computes a single bounding box over all active frames of the load, and the lead descends the stoop (`516–558`, X `428..532`), runs rightward across the street (`559–578`, X `526..782`), and spins at the right edge (`579–599`, X `756..944`), the union X span (`428..944 = 516 px`) exceeded half the canvas (`512 px`), forcing `1.0x` zoom if `zoom_up_to == 2.0`.
- **The fix**:
  - Setting `zoom_up_to: 4.0` for `lead` in `data/fun/plan.json` and splitting `516–599` at its 3 natural movement phases (`lead_fp_0516f` `516–558` at **`3.88x`**, `lead_fp_0559` `559–578` at **`2.51x`**, `lead_fp_0579` `579–599` at **`2.67x`**) raised her rendered part height from `35 px` (`1.1` tokens) to `108–120 px` (`3.4–3.8` tokens), and all 3 sub-loads were kept on first review.

### Case Study D — Worn Accessories Across Rapid Head Tosses (`lead_fp_3909b` -> `lead_fp_3909c`)
- **What happened**:
  - `lead_fp_3909b` (`3909–3966`) relied on post-render pixel give-back (`restore=lead.Eyeglass` in `plan.json`) without naming her white sunglasses in `lead_fp_3909b.txt`.
  - During her rapid head toss (`3915–3921`, `7` frames) and dip (`3928–3931`, `4` frames), Sapiens `Eyeglass` missed her motion-blurred frames, and `--worn-fill` (`FILL_REACH = 2`) cannot bridge a 7-frame gap, leaving `18` frames (`d129: worn_thing_not_laid`) where her sunglasses vanished and reappeared.
- **The fix**:
  - Adding `"<Subject 1> wears white sunglasses."` in `[Shot 1]` of `lead_fp_3909c.txt` (`--by-hand`, matching `lead_fp_3098b` and `lead_fp_3967c`) caused H3 to draw her white cat-eye sunglasses directly across every single frame (`3909`, `3915–3921`, `3928–3931`, `3960`).

### Case Study E — Dark Background Hair Anchored into 4-Frame Latent Steps by `c==0` Strangers Halo & Missing `hers` Depth Recovery (`second_hfp_4912q/r` -> `second_hfp_4912t`, `4912–4946`)
- **What the numbers showed**:
  - Across `4912q` and `4912r`, `<Subject 1>`'s rendered hair was dark brown (`4912q` CIE `L*a*b*` over `(second_classes == 4) & parts_held`: `median L* = 6.3, mean = 12.8, b* = +9.0`; `4912r`: `median L* = 11.4, mean = 17.8, b* = +16.0`) despite `<Picture 1>` (`second.png`) and the text prompt specifying golden-blonde waves.
  - Comparing `step_carves_summary` in `capture_manifest.json` across generations exposed a **`2.7x` jump** in `strangers` latent-cell carves after commit `426638a6`:
    - Pre-`426638a6` (`shot_c_4912h` on `4912g/j/k`, Facer class-filtered `strangers`, `has_depth = False`): `1,100` cells carved across `7/12` steps (`S0, S1, S6, S7, S11` had **`0` cells carved**).
    - Post-`426638a6` (`shot_c_4912i` on `4912m..4912s`, raw SAM2 union `strangers` with `38.2%` `c==0` background halo): **`2,986` cells carved across all `12/12` steps** (`S0`: `131`, `S1`: `210`, `S6`: `226`, `S7`: `202` carved on completely unoccluded frames!).
- **Why**:
  1. **Raw SAM2 `c==0` Halo Carved the `10 px` Hair Dilation Margin**: Because `c==0` on `strangers` was not in `HAIR_CLASSES = (4, 14)`, `is_hair_only` evaluated to `False` and `strangers_margin` (`14 px`) dilated that background halo right through `second`'s `10 px` hair dilation margin (`[2]`) on every frame (`4912–4920`, `4927–4938`). That left a ring of original dark-haired video pixels around her head inside the latent grid on every single step.
  2. **4-Frame Latent Temporal Packing (`1 + 4k`)**: Because 4 video frames pack into 1 temporal latent step (`S2 = 4917–4920`, `S4 = 4925–4928`, `S8 = 4938–4941`, `S9 = 4942–4945`), any frame where her visible crown was carved by `strangers` (`4920`, `4939–4940`, `4943–4946`) or where a `59 px` sliver leaked onto the foreground dancer's neck (`4926` in `S4`) anchored the step's latent hair features to the dark source video.
  3. **`hers` (`m == 1`) Only Applied to `head_track` (`[1]`), Not `strangers` (`[3]`)**: Before `ba86fbe9`, `head_marks.hers` only prevented `head_track` (`[1]`) from clipping the subject; `[3] strangers` subtract still wiped `4939–4940` and `4943–4946` because the foreground man's SAM2 mask unioned with his white hat extended up to `y=267` over `second`'s visible crown (`y=302..346`), even though MoGe-2 depth showed her crown was above the hat (`y=347..526`).
- **The fix (`ba86fbe9` + `data/fun/plan.json`)**:
  - In `bench/capture_masked_run.py` (`ba86fbe9`), restricted `keep_c0` on `strangers` to `(subject_classes == 0) | ~np.isin(subject_classes, [2, 3, 4])` so `c==0` SAM2 halo never carves the subject's hair margin, and added depth-gated `hers` (`m == 1`) crown recovery (`strangers_depth >= subject_depth - 0.015` on `[2, 3, 4]`).
  - Updated `data/fun/plan.json` on `4912–4946` (`hidden = 4921-4926, 4941-4942`, `hers = 4920, 4939-4940, 4943-4946`).
  - `strangers` carves dropped from **`2,986` (`12/12` steps) to `1,613` (`10/12` steps, `-1,373` cells)** (`S0` and `S3` `0` carved; `S6` `226 -> 15`; `S7` `202 -> 37`), and rendered hair CIE `L*a*b*` in `second_hfp_4912t` jumped to **`median L* = 25.5, mean = 30.6, p90 = 58.8, b* = +30.0`** across the entire shot.

---

## 3. Pre-Render & Post-Render Visual Verification Checklist

1. **At `reference`**:
   - Open `<subject>_head_reference_<first>_<last>_sheet.jpg` and check `the_one_box_enlarges` vs `each_frame_alone_would_enlarge_least_median_most` in `<subject>_head_reference_<first>_<last>.json`.
   - Confirm open mouths have white lip outlines + dark interior on singing/open-mouth frames (`open_mouth: outlined`), and confirm no occluded frame has a flipped/jumping 3D mesh or face lines drawn on hair/strangers.
2. **At `first` / `on-plan`**:
   - Extract a contact sheet of the `first`/`on-plan` strip on all boundary frames, `doubted_part_*` frames, and `strangers` overlap frames.
   - Run `bench/inspect_mask_steps.py --capture data/<day>_<load> --problems-only` to check `[0]`–`[7]` provenance and latent-step hygiene.
   - Inspect `step_carves_summary` in `capture_manifest.json` (`strangers.carved_units_total` and per-step `S0..Sn` carves): unoccluded latent steps should have near-zero `strangers` carves, and fully hidden steps (`m == 0`) should have `0` `final_units`.
3. **At `land` (Before `page`)**:
   - Extract a close-up contact sheet comparing `ORIG`, prior render (if remaking), and new `CUT` across 8–18 key frames (including any previously flagged frames) and inspect via `read_file` before paging to `data/fun/review/items.json`.
   - When hair or wardrobe color transfer is in question, measure CIE `L*a*b*` over `(subject_classes == 4) & parts_held` across pre-occlusion, mid-occlusion, and crown-overlap spans to verify quantitative color retention against `<Picture 1>`.
