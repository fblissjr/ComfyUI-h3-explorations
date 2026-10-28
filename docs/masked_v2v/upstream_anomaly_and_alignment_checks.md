# Upstream Data-Dependency, Outlier, and Alignment Diagnostic Reference (Best Known as of `2026-10-28`)

> [!IMPORTANT]
> **Point-in-Time Reference — Always Check for Code Drift or Live Code Bugs First**
> - **Snapshot Date**: `2026-10-28` (compiled after a full-day production & debugging session on `fun`).
> - **Drift & Code-Bug Warning**:
>   1. **Documentation Drift**: Stage writers (`subject_track.py`, `sapiens2_parts.py`, `bench/strangers.py`, `bench/head_tube_parts.py`, `bench/pose_along_shot.py`, `bench/crop_reference_to_head.py`, `bench/head_reference.py`, `bench/load_text.py`, `bench/capture_masked_run.py`), CLI flags, defaults in `workflows/h3_config.py`, or `data/fun/plan.json` may have changed since `2026-10-28`, or a new dependency link may exist that is not listed here.
>   2. **Live Bugs in Pipeline Code**: An anomaly in a capture or reference may be caused by an **actual bug or unhandled edge case in the pipeline code itself** (for example, `crop_reference_to_head.py` sizing a whole-shot `cranium` box over frames where a foreground occlusion flipped the 3D body mesh upside down, or `head_tube_parts.py` clipping upper hair crowns above SAM's head box, or `load_chain.py::page_item` accepting non-string fields).
>   3. **Mandatory Downstream Reprocessing When Stage `X` Changes**: Fixing a code bug or input parameter at Stage `X` invalidates every artifact downstream of `X`. You **must** reprocess from Stage `X` forward through the DAG:
>      - **Bug/change in `subject_track.py`, `sapiens2_parts.py`, `cast.jsonl`, or `--source` cut** -> Reprocess **`preview`** -> `strangers` (`job`) -> `pose` + `reference` -> `text` -> `first` -> `plan` -> `on-plan` -> `render`.
>      - **Bug/change in `bench/strangers.py` or `MASKED_STRANGERS`** -> Reprocess **`job` (`strangers` + `part_saved_on_track`)** -> `first` -> `plan` -> `on-plan` -> `render`.
>      - **Bug/change in `bench/pose_along_shot.py`, `bench/crop_reference_to_head.py`, `bench/draw_face_points.py`, `bench/head_reference.py`, or `--no-head-on` / `--no-face-on`** (as in the `lead_fp_0294c` `317–319` mesh-flip/crop anomaly today) -> Delete old reference outputs and reprocess **`reference`** -> `text` -> `first` -> `plan` -> `on-plan` -> `render`.
>      - **Bug/change in `bench/head_tube_parts.py`, `bench/capture_masked_run.py`, `--drop`, `--hold-part`, or `--least`** -> Delete old capture/job folders and reprocess **`first`** -> `plan` -> `on-plan` -> `render`.

---

## 1. Trace & Align Every Upstream Input in the DAG

Every masked V2V render depends on a multi-stage upstream chain:
`source video` -> `preview (shot_c_*)` -> `strangers.py` + `part_saved_on_track` -> `pose_along_shot.py` -> `head_reference.py` -> `load_text.py` -> `capture_masked_run.py (first)` -> `plan` -> `on-plan` -> `render` -> `land`.

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
- **The fix (`head_reference.py:782-783`)**:
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

---

## 3. Pre-Render & Post-Render Visual Verification Checklist

1. **At `reference`**:
   - Open `<subject>_head_reference_<first>_<last>_sheet.jpg` and check `the_one_box_enlarges` vs `each_frame_alone_would_enlarge_least_median_most` in `<subject>_head_reference_<first>_<last>.json`.
   - Confirm open mouths have white lip outlines + dark interior on singing/open-mouth frames (`open_mouth: outlined`), and confirm no occluded frame has a flipped/jumping 3D mesh or face lines drawn on hair/strangers.
2. **At `first` / `on-plan`**:
   - Extract a contact sheet of the `first`/`on-plan` strip on all boundary frames, `doubted_part_*` frames, and `strangers` overlap frames.
   - Run `bench/inspect_mask_steps.py --capture data/<day>_<load> --problems-only` to check `[0]`–`[7]` provenance and latent-step hygiene.
3. **At `land` (Before `page`)**:
   - Extract a close-up contact sheet comparing `ORIG`, prior render (if remaking), and new `CUT` across 8–18 key frames (including any previously flagged frames) and inspect via `read_file` before paging to `data/fun/review/items.json`.
