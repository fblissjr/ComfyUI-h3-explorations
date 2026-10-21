# Handover: read this before anything else

> **PINNED SOURCE OF TRUTH FOR FINISHING `fun` (`0000–6394`)**:
> 1. **Clean End-to-End Cut, Frame-Level Gap Audit & Hierarchical Data Flow Lineage Map (CST timestamps, MP4 paths, per-frame coverage barcodes, and 6-stage DAG)**:
>    - **Browser**: `http://100.87.141.65:8765/PROVENANCE.html` (also at `http://100.87.141.65:8765/CUT_MAP.html` / `data/fun/poc/PROVENANCE.html`)
>    - **Markdown**: **`data/fun/CUT_MAP.md`** (regenerated in ~1.4s anytime via `<comfy-venv>/bin/python bench/build_cut_map.py --video fun`)
> 2. **Execution Runbook & Pre-Built Previews**:
>    - Read **`data/fun/FINAL_STRETCH_RUNBOOK.md`** (and [`data/2026-10-20_fun_phase2_inputs/README.md`](../../data/2026-10-20_fun_phase2_inputs/README.md) / [`data/2026-10-20_fun_phase3_inputs/README.md`](../../data/2026-10-20_fun_phase3_inputs/README.md) beside the per-window input files) FIRST. Keep this block pinned at the top of `handover.md` throughout the day until `fun` is 100% finished.

**Rewritten at the end of each working day by the designated lead; the day before is in git.** Written for 2026-10-20 by
`mrgem` (designated lead holding both sides' roles) at 2026-10-21T01:45Z (`8:45 PM CDT`, end of day). Where this page and a record disagree, the record is
right ([`decisions.md`](decisions.md), newest first; `data/fun/laid.jsonl`; `data/fun/CUT_MAP.md`; `data/fun/FINAL_STRETCH_RUNBOOK.md`; the newest gap check under
`data/2026-10-20_fun_delivery_tables/`; `git log`).

## State when closed

- **Servers stopped cleanly at close**: ComfyUI (`:8188`) and the review server (`:8765`, `data/fun/poc/serve.py`) were both stopped at close after all renders, landings, delivery assembly, and page rebuilds completed.
- **Kept and laid in `data/fun/laid.jsonl` (`50 lines`, `sha256 80f694bba3639c1f`, `0 missing or changed`)**:
  - **Post-Pass-1 24.0 fps Joined Cut Built & Registered for Tomorrow's Multi-Subject Pass 2/3**:
    - Output on share: `Video/fun/day_2026-10-20/fun_0000_6394_joined_2026-10-20T0110Z_from_laid.mp4` (`410,679,346` bytes, 6,395 frames at `24.0 fps`).
    - Input copy for ComfyUI / `shot_to_load.py --source`: `fun_0000_6394_joined_2026-10-20T0110Z_from_laid.mp4` under `H3_INPUT_DIR` (registered in `data/fun/manifest.json`, `head_sha256 8d6c19a1cf612ad5`).
  - **Phase 1 Full Kitchen Pass-2 (`lead_fp_0606d` on `606–998`, 393f / 2 windows `[345, 141]`)**: rendered (`1879 s`), landed (`Video/fun/mrgem/lead_fp_0606d/lead_fp_0606d/lead_fp_0606d_00001.mp4`), stacked review clip generated (`Video/fun/review_clips/lead_fp_0606d_0606_0998_stacked.mp4`), laid on `604–1009` (`lead`) in `data/fun/laid.jsonl` (replacing `lead_fp_0751e` and `lead_fp_0606c`), and added to `data/fun/review/items.json`.
  - **Phase 2 Subject 2 (`second`) Remakes Completed, Validated (`mrgemread`), & Laid in `data/fun/laid.jsonl` (`All 19 of 19` `second` shots across `fun` `0000–6394` now rendered, landed, validated, and laid in `data/fun/laid.jsonl`)**:
    - **Prior / Single-Subject `second` shots laid (`8` shots)**:
      1. `second_hfp_0014f` (`14–167`, 154f, owner kept)
      2. `second_hfp_0278i` (`278–293`, 16f)
      3. `second_hfp_0604h` (`604–1009`, 406f, owner kept)
      4. `second_hfp_1010g` (`1010–1038`, 29f, 0 dropped frames)
      5. `second_hfp_4479b` (`4479–4509`, 31f, single-subject `second`)
      6. `second_hfp_5114f` (`5114–5129`, 16f, single-subject `second`)
      7. `second_hfp_5461e` (`5461–5497`, 37f, single-subject `second`)
      8. `second_hfp_5533b` (`5533–5564`, 32f) and `second_hfp_5675c` (`5675–5715`, 41f).
    - **All `12` Multi-Subject Pass-1 `second` loads rendered, landed, paged, validated (`playback_reading_mrgemread.md`, all 12 `keep`), and laid today (`b270be7e` 3D mesh closed lips + `a0dddf18` downcast-eye arc + unified `<Subject 2>` prompt)**:
      9. `second_hfp_0415g` (`415–437`, 23f, replacing `second_hfp_0415e`)
      10. `second_hfp_0484f` (`484–515`, 32f, replacing `second_hfp_0484d`)
      11. `second_hfp_1039d` (`1039–1060`, 22f, replacing `second_hfp_1039b`)
      12. `second_hfp_1093c` (`1093–1133`, 41f whole-shot, interior `1124–1127` hand occlusion verified via `verify --no-mask`)
      13. `second_hfp_1419c` (`1419–1465`, 47f whole-shot, interior `1419–1427, 1437–1452` occlusions verified via `verify --no-mask`)
      14. `second_hfp_4602c` (`4602–4646`, 45f, interior `4635–4636` occlusion verified via `verify --no-mask`)
      15. `second_hfp_4647d` (`4647–4725`, 79f, interior `4683–4688, 4716–4725` occlusions verified via `verify --no-mask`)
      16. `second_hfp_4726c` (`4726–4773`, 48f, interior `4752–4755` occlusion verified via `verify --no-mask`)
      17. `second_hfp_4912e` (`4912–4946`, 35f, interior `4924–4926, 4946` occlusions verified via `verify --no-mask`)
      18. `second_hfp_5011e` (`5011–5043`, 33f, interior `5032` occlusion verified via `verify --no-mask`)
      19. `second_hfp_5768e` (`5768–5791`, 24f, `5768–5773` bowed behind `third`'s shoulder, emerging on `5774–5791`)
      20. `second_hfp_5889e` (`5889–5934`, 46f; laid on `5889–5915` where `second`'s face and hair are in frame behind `third` before the door opens and the conga line marches out, pushing her face/hair off the right edge at `5916` with only her arm/dress remaining on `5916–5934`).
  - **All 15 Remaining `second` Clothing Spans Added to `data/fun/plan.json`**: verified from source frames (`415–437`, `484–515`, `1093–1133`, `1419–1465`, `4479–4509`, `4602–4646`, `4647–4725`, `4726–4773`, `4912–4946`, `5011–5043`, `5114–5129`, `5533–5564`, `5675–5767`, `5768–5791`, `5889–5934`).
  - **Interactive 2-Tab Cut, Frame-Level Gap Audit & Lineage Explorer (`PROVENANCE.html` / `CUT_MAP.md`)**:
    - Built by [`bench/build_cut_map.py`](../../bench/build_cut_map.py) (local-disk-first, runs in ~1.4s). Syncs status badges (`IN CUT 100%` vs `GAP / RERUN`) with `frame_coverage_fun_0000_6394_*.json` + Phase 2/3/4 queued inputs, and provides a per-window **Frame-Level Coverage Barcode & Drill-Down** alongside the **6-Stage Hierarchical Data Flow Lineage DAG**.
- **Landed on `main` today (`93572357`, `35fb81fd`, `647e443d`, `5b56c7ef`, `b270be7e`, `a0dddf18`, `825d2f65`, `b47a7852`, `6828f77f`, `427154bb`)**:
  0. **Whole-Shot Part Load `on-plan` Verification Fix (`6828f77f`, [`bench/capture_masked_run.py`](../../bench/capture_masked_run.py)) & Parent -> Child Invalidation Docs (`825d2f65`, `b47a7852`, `427154bb`)**:
     - `bench/capture_masked_run.py::no_mask_statement_against_a_load` now recognizes `"the mask each window was given"` (written by `on-plan` and `land`) in addition to `"the part"` and `"the held part"`, so whole-shot part loads with verified interior occlusions (`verify --no-mask`) pass `on-plan` and `land` without splitting the shot.
     - Documented the 6-stage Parent -> Child invalidation DAG and ComfyUI (`:8188`) server-restart rules in [`docs/masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) ("8. Parent -> Child Invalidation DAG"), [`docs/masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md) ("Stage 0.5"), and [`data/AGENTS.md`](../../data/AGENTS.md).
  1. **Clean 3D Mesh Closed Lips, Retired `smooth_mouth_area` Smudge, Fixed `fit_mouth_to_head`, Downcast Eye Arc (`DOWNCAST_RING_OUTSIDE_MIN = 0.65`), & Unified `<Subject 2>` Prompt (`b270be7e`, `a0dddf18`, [`bench/draw_face_points.py`](../../bench/draw_face_points.py), [`bench/head_reference.py`](../../bench/head_reference.py), [`bench/load_text.py`](../../bench/load_text.py))**:
     - `smooth_mouth_area` is a no-op (retiring the flat grey oval smudge over the lips/chin/hands).
     - Closed-mouth frames (`not has_hole` and `gap <= INNER_GAP_CLOSED_PX`) stay as natural 3D grey mesh lips (`mouths[f] = None`, no white parting line); open-mouth frames (`inner_lip_ring` / `mouth_inner`) draw a black interior + white outline at the mouth.
     - `fit_mouth_to_head` only shifts vertically above a real bottom-of-frame dark neck gap under the jaw, never on a hand/arm gap above the pose mouth.
     - On downcast or narrowed eyes (`between_the_lids` `off >= DOWNCAST_RING_OUTSIDE_MIN = 0.65`, where `>= 65%` of the iris ring lies outside the eyelids and the black lash-ring border covers `100%` of the clipped black iris disc), `discs()` leaves the sclera unpainted and `shapes()` draws the between-the-lids iris arc (`_iris_ring`) instead of flooding the narrowed eye slit with `255,255,255` white sclera.
     - `load_text.py` uses one unified `<Subject 2>` sentence across all face-drawn loads ("with facial features drawn on its face as thin white outlines, where the eyes open, close and look where the person's eyes do") so text never biases mouth state while preserving eye-gaze transfer.
     - **Verified on `second_hfp_0415g` (`415–437`) and `second_hfp_0484f` (`484–515`)**: `second_hfp_0415g` completely eliminated the wide camera-staring eyes and toothy smile on frame `437` (compared to `0415f`) and kept the mouth closed and gaze lowered across `415–437`; `second_hfp_0484f` eliminated the dark open `"O"` mouth of `0484d` and softened/lowered the gaze compared to `0484e` (with one noted difference from `0484c`: `0484c` rendered on `render_region: full` at `zoom: 1.0x`, whereas `0484f` rendered on `render_region: part` at `zoom: 1.63x`, where `<Picture 1>`'s frontal portrait has stronger weight on early frames `484–496`).
  2. **Mandatory `--source` & Multi-Subject Pass-2 Refusal (`5b56c7ef`, [`bench/shot_to_load.py`](../../bench/shot_to_load.py))**:
     `shot_to_load.py preview` requires `--source <video>` with no default (`Refused` if omitted), and both `preview`
     and `job` check `data/<video>/laid.jsonl` (`check_pass2_source`), refusing the raw original source video when
     another subject on the shot is already laid on overlapping frames (unless `--allow-raw-source WHY` is passed).
  3. **Automatic Short-Dip Temporal Bridging & Dropped-Face Preflight Gate (`647e443d`)**:
     [`bench/head_tube_parts.py`](../../bench/head_tube_parts.py) (`bridge_short_tube_dips`, `SHORT_DIP_MAX_FRAMES = 6`,
     `SHORT_DIP_RATIO = 0.6`) automatically bridges transient `<= 6`-frame Sapiens class-map dips inside a continuous
     person track; [`bench/capture_masked_run.py`](../../bench/capture_masked_run.py)
     (`dropped_or_unmasked_face_in_context`, `DROPPED_FACE_MIN_PX = 4000`) blocks preflight (`dropped_frame_has_visible_face`)
     when a `--drop`ped frame still has visible `Face_Neck`; `shot_to_load.py` refuses missing `--cast` files cleanly.
  4. **Pre-Render Telemetry (`facts_pass.py`), Step Keep-Out Carves (`step_keepout_carves`), & Code Stamps (`code_stamp.py`)
     (`93572357`, `35fb81fd`)**:
     [`bench/facts_pass.py`](../../bench/facts_pass.py) emits per-frame `facts.json`/`facts.csv` (`face_outside_region_px`,
     `other_overlap_px`, `mouth_mode`, `ref_crop_box`, `zoom_box`); `capture_masked_run.py` emits
     `runs/<run>/step_carves.json`; [`bench/_lib/code_stamp.py`](../../bench/_lib/code_stamp.py) stamps `writer_stamp`
     into derived `.json`/`.npz` files across 6 tools and refuses stale upstream inputs in `head_reference.py`.
  5. **Git-Tracked Masked V2V Protocol & Automatic `poc/` Sync (`5b56c7ef`)**:
     Added [`docs/masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) and [`docs/masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md),
     and wired [`bench/rebuild_delivery.py`](../../bench/rebuild_delivery.py) to sync `TO_REVIEW.html` and
     `PROVENANCE.html` into `data/<video>/poc/` on every `pages` step.

## What not to assume

- **Never run Pass 2 of a multi-subject shot on a preview or source from the raw unswapped video.** On Kitchen `604-1009`,
  the raw 1980s mom had a tall dark bouffant hairdo (`ymin=188..209`) ~60 px higher than the replaced blonde mom
  (`second_hfp_0604h`, `ymin=211..221`). Segmenting the raw video gave `second`'s keep-out `5,843–9,276 px` of old
  bouffant hair that carved a dark helmet out of `lead`'s face on `758-771`. Always pass the post-Pass-1 joined cut to
  `shot_to_load.py preview --source`.
- **Never split a shot `<= 362` frames (`h3_rules.MAX_LENGTH`) around an interior occlusion.** Keep the entire shot in one whole-shot load, account for the occluded frames with `capture_masked_run.py verify <capture> --no-mask "<subject>:<first>-<last>=<why>"`, and let the node's composite leave those `0 px` mask frames as the source (`left_as_the_source_with_no_mask`, fixed for `on-plan` and `land` in `6828f77f`).
- **H3's bidirectional temporal attention propagates even 3–4 unmasked frames across the whole window.** On Hallway
  `1306-1418`, `1402-1404` and `1408-1411` Sapiens dips plus `--drop 1411-1413` exposed the original actor's grey beard
  in the frozen context, making H3 render a beard from frame `1306` onward. Never `--drop` a frame where a face/beard is
  visible; use `--no-face-on` only on genuine back-of-head frames.
- **Every review clip must be added to `data/<video>/review/items.json` (`state: "open"`).** Static `TO_REVIEW.html` has
  no verdict buttons; the owner submits verdicts and frame notes on `http://<host>:8765/` (`data/<video>/poc/index.html`),
  which reads `items.json` and appends to `data/<video>/owner_answers.jsonl`.
- **Never validate a clip from 4 evenly-spaced frame samples (`start, 1/3, 2/3, end`), and never lay tail frames where a part-load subject's head/hair has already exited the frame (`second_hfp_5889e`, `5889–5934`).** On `second_hfp_5889e`, `second` is behind `third` on `5889–5915`, then at `5916` the door opens and a conga line marches out while `second`'s face and hair exit the right frame edge, leaving only her arm/dress at the edge on `5916–5934`. Evenly-spaced 4-frame samples missed the `5914 -> 5916` exit completely, whereas the stacked clip's own flags (`pose_mostly_outside_the_frame (second) on 5915-5934`, `part_not_visible (['second.Face_Neck', 'second.Hair', ...])`) and `assemble_delivery.py`'s `ends` check (`ends on the source's own picture under its subject: frames 5930-5933`) caught that nothing of her face/hair was in frame on the tail. Always watch the continuous clip across every `pose_mostly_outside_the_frame` / `part_not_visible` transition, and `--drop` (or lay only `--frames 5889-5915` in `laid.jsonl`) tail spans where the replaced parts (`face_and_neck` + `hair`) have left the frame.

## Next up, in order

1. **Read in by `AGENTS.md`'s order**, plus [`docs/masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md),
   [`docs/masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), `data/fun/AGENTS.md`, `data/fun/CUT_MAP.md`, and `data/fun/FINAL_STRETCH_RUNBOOK.md` before lane work. Start ComfyUI (`:8188`) and the review server (`:8765`, `data/fun/poc/serve.py`) per [`docs/comfy_notes.md`](../comfy_notes.md).
2. **Owner Review Check of the 12 New `second` Pass-1 Clips & Joined Cut**:
   - All 12 new `second` Pass-1 clips (`0415g`, `0484f`, `1039d`, `1093c`, `1419c`, `4602c`, `4647d`, `4726c`, `4912e`, `5011e`, `5768e`, `5889e`) are open in `data/fun/review/items.json` and `data/fun/review/start_here.json` with folded reader cards (`playback_reading_mrgemread.md`, all 12 `keep`).
3. **Run Multi-Subject Pass 2 / Pass 3 (`lead` and `third`) on the Post-Pass-1 24.0 fps Cut (`fun_0000_6394_joined_2026-10-20T0110Z_from_laid.mp4`)**:
   - All 19 `second` shots across `fun` (`0000–6394`) are laid in `data/fun/laid.jsonl` (`50 lines`, `sha256 80f694bba3639c1f`), and the 24.0 fps post-Pass-1 cut is registered in `data/fun/manifest.json` as `fun_0000_6394_joined_2026-10-20T0110Z_from_laid.mp4` under `H3_INPUT_DIR`.
   - For the 2-subject shots sharing a frame with `second` (`278–293` `lead`, `415–437` `lead`, `484–515` `lead`, `1010–1038` `lead`, `1039–1060` `lead`, `4602–4646` `third`, `4726–4773` `third`, `4912–4946` `third`, `5533–5564` `third`, `5675–5715` `third`, `5889–5934` `third`), run `shot_to_load.py preview --source "$H3_INPUT_DIR/fun_0000_6394_joined_2026-10-20T0110Z_from_laid.mp4" --cast data/fun/cast/cast.jsonl --remake` -> `job` -> `reference` -> `text` -> `first` -> `plan` -> `on-plan` -> `table` -> `render` -> `land` -> `page`.
   - For the 3-subject shots (`1093–1133`, `1419–1465`, `4647–4725`, `5011–5043`, `5768–5791`), run Pass 2 (`third`) on `"$H3_INPUT_DIR/fun_0000_6394_joined_2026-10-20T0110Z_from_laid.mp4"`, lay `third` into `laid.jsonl`, rebuild the post-Pass-2 cut (`rebuild_delivery.py --video fun --clip`), and run Pass 3 (`lead`) on that post-Pass-2 cut.
4. **Run Remaining Single-Subject `lead` / `third` Shots (`data/2026-10-20_fun_phase3_inputs/`)**:
   - The 32 pre-built single-subject / non-`second` previews (`shot_c_0168b`..`6105b`, including `2823–3166` tail gap `shot_c_2823c.json` and `5935–6104` `shot_c_5935c.json`) are already built in `data/2026-10-20_shot_c_*_job/preview.json` on the 24.0 fps working copy (`fun_0000_6394_24p_sound_in_step.mov`, tracker mask version 15, `track_head=True`, seeded from `data/fun/cast/cast.jsonl`). Queue their previews and run their loads through `load_chain.py` with `bench/facts_pass.py` before every render.

## Where things are

- **Lane rules & execution protocol**: [`docs/masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md),
  [`docs/masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), [`data/AGENTS.md`](../../data/AGENTS.md), and
  `data/fun/AGENTS.md` (local).
- **Decisions & why**: [`docs/wiki/decisions.md`](decisions.md) (`## 2026-10-20` at top).
- **The cut, ledger, & gap check**: `data/fun/laid.jsonl` (`50 lines`, `sha256 80f694bba3639c1f`, with all 19 `second` shots laid plus `lead_fp_0606d` on `606–998` and `third_fp_1306f` on `1306–1418`); newest rebuild
  manifest and coverage table under `data/2026-10-20_fun_delivery_tables/`; interactive review page at
  `http://100.87.141.65:8765/` (`data/fun/review/items.json`, `data/fun/owner_answers.jsonl`) and static
  `Video/fun/TO_REVIEW.html` / `data/fun/poc/TO_REVIEW.html` and `PROVENANCE.html`.
- **Commands & environment keys**: [`docs/wiki/runbook.md`](runbook.md) and [`docs/comfy_notes.md`](../comfy_notes.md).
