# Handover: read this before anything else

> **PINNED SOURCE OF TRUTH FOR FINISHING `fun` (`0000–6394`)**:
> 1. **Clean End-to-End Cut, Frame-Level Gap Audit & Hierarchical Data Flow Lineage Map (CST timestamps, MP4 paths, per-frame coverage barcodes, and 6-stage DAG)**:
>    - **Browser**: `http://100.87.141.65:8765/PROVENANCE.html` (also at `http://100.87.141.65:8765/CUT_MAP.html` / `data/fun/poc/PROVENANCE.html`)
>    - **Markdown**: **`data/fun/CUT_MAP.md`** (regenerated in ~1.4s anytime via `<comfy-venv>/bin/python bench/build_cut_map.py --video fun`)
> 2. **Execution Runbook & Pre-Built Previews**:
>    - Read **`data/fun/FINAL_STRETCH_RUNBOOK.md`** (and [`data/2026-10-20_fun_phase2_inputs/README.md`](../../data/2026-10-20_fun_phase2_inputs/README.md) / [`data/2026-10-20_fun_phase3_inputs/README.md`](../../data/2026-10-20_fun_phase3_inputs/README.md) beside the per-window input files) FIRST. Keep this block pinned at the top of `handover.md` throughout the day until `fun` is 100% finished.

**Rewritten at the end of each working day by the designated lead; the day before is in git.** Written for 2026-10-20 by
`mrgem` (designated lead holding both sides' roles) at 2026-10-20T19:20Z (`2:20 PM CST`). Where this page and a record disagree, the record is
right ([`decisions.md`](decisions.md), newest first; `data/fun/laid.jsonl`; `data/fun/CUT_MAP.md`; `data/fun/FINAL_STRETCH_RUNBOOK.md`; the newest gap check under
`data/2026-10-19_fun_delivery_tables/`; `git log`).

## State when closed

- **Kept and laid in `data/fun/laid.jsonl` (`41` subject-shots in cut out of `90`: `24` clean 100% carried, `16` with frame gaps or queued re-run, `50` pending)**:
  - **Phase 1 Full Kitchen Pass-2 (`lead_fp_0606d` on `606–998`, 393f / 2 windows `[345, 141]`)**: rendered (`1879 s`), landed (`Video/fun/mrgem/lead_fp_0606d/lead_fp_0606d/lead_fp_0606d_00001.mp4`), stacked review clip generated (`Video/fun/review_clips/lead_fp_0606d_0606_0998_stacked.mp4`), laid on `604–1009` (`lead`) in `data/fun/laid.jsonl` (replacing `lead_fp_0751e` and `lead_fp_0606c`), and added to `data/fun/review/items.json`.
  - **Phase 2A & 2B Subject 2 (`second`) Remakes Completed & Laid Today (`10` of `19` `second` shots now done as whole-shot renders with `TRACK_HEAD=True`, `"whole region"`, and retired `held`/`hands`/`clothes`/`hair_band` keep-outs)**:
    1. `second_hfp_0278i` (`278–293`, 16f, replacing `second_hfp_0278f`)
    2. `second_hfp_0415e` (`415–437`, 23f, replacing `second_hfp_0415d`)
    3. `second_hfp_0484d` (`484–515`, 32f, replacing `second_hfp_0484c`)
    4. `second_hfp_1010g` (`1010–1038`, 29f, 0 dropped frames, replacing `second_hfp_1010j`)
    5. `second_hfp_1039b` (`1039–1060`, 22f, replacing `second_hf_1039_g` — restoring 100% live `.render.built.json` records across all rows in `laid.jsonl`)
    6. `second_hfp_4479b` (`4479–4509`, 31f)
    7. `second_hfp_5114f` (`5114–5129`, 16f, replacing `second_hfp_5114d`)
    8. `second_hfp_5461e` (`5461–5497`, 37f, replacing `second_hp_5461c`)
    9. `second_hfp_5533b` (`5533–5564`, 32f)
    10. `second_hfp_5675c` (`5675–5715`, 41f).
  - **All 15 Remaining `second` Clothing Spans Added to `data/fun/plan.json`**: verified from source frames (`415–437`, `484–515`, `1093–1133`, `1419–1465`, `4479–4509`, `4602–4646`, `4647–4725`, `4726–4773`, `4912–4946`, `5011–5043`, `5114–5129`, `5533–5564`, `5675–5767`, `5768–5791`, `5889–5934`).
  - **Interactive 2-Tab Cut, Frame-Level Gap Audit & Lineage Explorer (`PROVENANCE.html` / `CUT_MAP.md`)**:
    - Built by `data/fun/build_cut_map.py` (local-disk-first, runs in ~1.4s). Syncs status badges (`IN CUT 100%` vs `GAP / RERUN`) with `frame_coverage_fun_0000_6394_*.json` + Phase 2/3/4 queued inputs, and provides a per-window **Frame-Level Coverage Barcode & Drill-Down** alongside the **6-Stage Hierarchical Data Flow Lineage DAG**.
- **Landed on `main` today (`93572357`, `35fb81fd`, `647e443d`, `5b56c7ef`, `b270be7e`)**:
  0. **Clean 3D Mesh Closed Lips, Retired `smooth_mouth_area` Smudge, Fixed `fit_mouth_to_head`, & Unified `<Subject 2>` Prompt (`b270be7e`, [`bench/draw_face_points.py`](../../bench/draw_face_points.py), [`bench/head_reference.py`](../../bench/head_reference.py), [`bench/load_text.py`](../../bench/load_text.py))**:
     - `smooth_mouth_area` is a no-op (retiring the flat grey oval smudge over the lips/chin/hands).
     - Closed-mouth frames (`not has_hole` and `gap <= INNER_GAP_CLOSED_PX`) stay as natural 3D grey mesh lips (`mouths[f] = None`, no white parting line); open-mouth frames (`inner_lip_ring` / `mouth_inner`) draw a black interior + white outline at the mouth.
     - `fit_mouth_to_head` only shifts vertically above a real bottom-of-frame dark neck gap under the jaw, never on a hand/arm gap above the pose mouth.
     - `load_text.py` uses one unified `<Subject 2>` sentence across all face-drawn loads ("with facial features drawn on its face as thin white outlines, the white of the eye white and the iris black") so text never biases mouth state and `<Video 1>` alone drives frame-by-frame mouth articulation. Verified on `second_hfp_0415f` (`415–437`) and `second_hfp_0484e` (`484–515`).
  1. **Mandatory `--source` & Multi-Subject Pass-2 Refusal (`5b56c7ef`, [`bench/shot_to_load.py`](../../bench/shot_to_load.py))**:
     `shot_to_load.py preview` requires `--source <video>` with no default (`Refused` if omitted), and both `preview`
     and `job` check `data/<video>/laid.jsonl` (`check_pass2_source`), refusing the raw original source video when
     another subject on the shot is already laid on overlapping frames (unless `--allow-raw-source WHY` is passed).
  2. **Automatic Short-Dip Temporal Bridging & Dropped-Face Preflight Gate (`647e443d`)**:
     [`bench/head_tube_parts.py`](../../bench/head_tube_parts.py) (`bridge_short_tube_dips`, `SHORT_DIP_MAX_FRAMES = 6`,
     `SHORT_DIP_RATIO = 0.6`) automatically bridges transient `<= 6`-frame Sapiens class-map dips inside a continuous
     person track; [`bench/capture_masked_run.py`](../../bench/capture_masked_run.py)
     (`dropped_or_unmasked_face_in_context`, `DROPPED_FACE_MIN_PX = 4000`) blocks preflight (`dropped_frame_has_visible_face`)
     when a `--drop`ped frame still has visible `Face_Neck`; `shot_to_load.py` refuses missing `--cast` files cleanly.
  3. **Pre-Render Telemetry (`facts_pass.py`), Step Keep-Out Carves (`step_keepout_carves`), & Code Stamps (`code_stamp.py`)
     (`93572357`, `35fb81fd`)**:
     [`bench/facts_pass.py`](../../bench/facts_pass.py) emits per-frame `facts.json`/`facts.csv` (`face_outside_region_px`,
     `other_overlap_px`, `mouth_mode`, `ref_crop_box`, `zoom_box`); `capture_masked_run.py` emits
     `runs/<run>/step_carves.json`; [`bench/_lib/code_stamp.py`](../../bench/_lib/code_stamp.py) stamps `writer_stamp`
     into derived `.json`/`.npz` files across 6 tools and refuses stale upstream inputs in `head_reference.py`.
  4. **Git-Tracked Masked V2V Protocol & Automatic `poc/` Sync (`5b56c7ef`)**:
     Added [`docs/masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) and [`docs/masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md),
     and wired [`bench/rebuild_delivery.py`](../../bench/rebuild_delivery.py) to sync `TO_REVIEW.html` and
     `PROVENANCE.html` into `data/<video>/poc/` on every `pages` step.

## What not to assume

- **Never run Pass 2 of a multi-subject shot on a preview or source from the raw unswapped video.** On Kitchen `604-1009`,
  the raw 1980s mom had a tall dark bouffant hairdo (`ymin=188..209`) ~60 px higher than the replaced blonde mom
  (`second_hfp_0604h`, `ymin=211..221`). Segmenting the raw video gave `second`'s keep-out `5,843–9,276 px` of old
  bouffant hair that carved a dark helmet out of `lead`'s face on `758-771`. Always pass the post-Pass-1 joined cut to
  `shot_to_load.py preview --source`.
- **H3's bidirectional temporal attention propagates even 3–4 unmasked frames across the whole window.** On Hallway
  `1306-1418`, `1402-1404` and `1408-1411` Sapiens dips plus `--drop 1411-1413` exposed the original actor's grey beard
  in the frozen context, making H3 render a beard from frame `1306` onward. Never `--drop` a frame where a face/beard is
  visible; use `--no-face-on` only on genuine back-of-head frames.
- **When testing a short crossing sub-window (e.g., `751-785`), match the reference's framing to the load's zoom.** A
  full-shot whole-frame reference (`1024x768`) paired with a zoomed-in sub-window (`render_region: part`) mismatches the
  plate scale unless the reference is cropped to the head or the sub-window's `zoom_box` matches the reference.
- **Every review clip must be added to `data/<video>/review/items.json` (`state: "open"`).** Static `TO_REVIEW.html` has
  no verdict buttons; the owner submits verdicts and frame notes on `http://<host>:8765/` (`data/<video>/poc/index.html`),
  which reads `items.json` and appends to `data/<video>/owner_answers.jsonl`.

## Next up, in order

1. **Read in by `AGENTS.md`'s order**, plus [`docs/masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md),
   [`docs/masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), `data/fun/AGENTS.md`, `data/fun/CUT_MAP.md`, and `data/fun/FINAL_STRETCH_RUNBOOK.md` before lane work.
2. **Phase 2B `second` Previews & Load Jobs (Currently Running on `:8188` in `task-700`)**:
   - Previews `shot_c_4479c`, `4602c`, `4647c`, `4726c`, `4912c`, `5114c`, `5461c`, `5533c`, `5675c`, `5768c`, `5889c` are completed on `:8188`; `shot_c_5011c` + `reference -> text -> first -> plan -> on-plan -> table` for all Phase-2B `second` jobs (`second_hfp_4479b`..`5889b`) are running now.
   - Note on `1093–1133` and `1419–1465` (`second`): on `1093–1133`, `lead`'s gloved hand occludes `second` on `1124–1127` (use `--first 1093 --frames 31 --why ...` or `--who-is-who`); on `1419–1465`, `second` is only visible in the far background on `1455–1465` (11f).
   - Once Phase-2B `second` jobs reach `table`, verify their full prompts (`what_h3_got.py`), record `looked`, render & land them on `:8188`, lay them in `data/fun/laid.jsonl`, and assemble a fresh post-Pass-1 cut MP4 (`bench/assemble_delivery.py` + `bench/frame_coverage_report.py`) before running Pass-2 (`lead`/`third`) on multi-subject shots (`1010–1038`, `4479`, `4602`, `4647`, `4726`, `4912`, `5011`, `5533`, `5675`, `5768`, `5889`).
3. **Phase 3 & 4 — Remaining Unfinished / Gap Shots (`data/2026-10-20_fun_phase3_inputs/`)**:
   - Includes the `2823–3166` tail gap (`3098–3109`, pre-built in `shot_c_2823c.json` to render the continuous `2823–3109` 287f span) and all remaining `PENDING` / `GAP` windows shown in `PROVENANCE.html`.

## Where things are

- **Lane rules & execution protocol**: [`docs/masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md),
  [`docs/masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), [`data/AGENTS.md`](../../data/AGENTS.md), and
  `data/fun/AGENTS.md` (local).
- **Decisions & why**: [`docs/wiki/decisions.md`](decisions.md) (`## 2026-10-19` at top).
- **The cut, ledger, & gap check**: `data/fun/laid.jsonl` (`third_fp_1306f` laid on `1306-1418`); newest rebuild
  manifest and coverage table under `data/2026-10-19_fun_delivery_tables/`; interactive review page at
  `http://100.87.141.65:8765/` (`data/fun/review/items.json`, `data/fun/owner_answers.jsonl`) and static
  `Video/fun/TO_REVIEW.html` / `data/fun/poc/TO_REVIEW.html`.
- **Commands & environment keys**: [`docs/wiki/runbook.md`](runbook.md) and [`docs/comfy_notes.md`](../comfy_notes.md).
