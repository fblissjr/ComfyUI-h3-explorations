# Handover: read this before anything else

**Rewritten at the end of each working day by the designated lead; the day before is in git.** Written for 2026-10-19 by
`mrgem` (designated lead holding both sides' roles) at 22:32Z. Where this page and a record disagree, the record is
right ([`decisions.md`](decisions.md), newest first; `data/fun/laid.jsonl`; the newest gap check under
`data/2026-10-19_fun_delivery_tables/`; `git log`).

## State when closed

- **Kept and laid today** (`data/fun/laid.jsonl`, 38 lines):
  - **Hallway `1306-1418` (`third_fp_1306f`, replacing `third_fp_1310c`)**: reviewed and kept by the owner ("fantastic
    job on that one"). Covers the entire 113-frame hallway shot (`1306-1418`, zero dropped frames) clean-shaven (`0/113`
    beard, `0/113` glasses) with `lead_fp_1306t` + `third_fp_1306f` assembled in
    `Video/fun/mrgem/hall_1306_1418_lead_fp_1306t_third_fp_1306f_assembled.mp4`.
  - **Kitchen crossing `751-785` (`lead_fp_0751e`)**: rendered on the 24.0 fps post-Pass-1 cut (`second_hfp_0604h`
    already laid) using the post-Pass-1 preview `shot_c_0604e`. Eliminated the 1980s mom's tall dark bouffant hair dome
    (`5,843–9,276 px` on `758-771`), reducing `face_outside_region_px` from `9,665–14,189 px` to `0–17 px` on `758-759`
    and `0–6 px` on `771-785` (confirmed by the owner: "solved the mystery of the mask hiding the lead's face in the
    kitchen scene"). Live at the top of `data/fun/review/items.json` (`http://<host>:8765/`).
- **Landed on `main` today (`93572357`, `35fb81fd`, `647e443d`, `5b56c7ef`)**:
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

## Tomorrow, in order

1. **Read in by `AGENTS.md`'s order**, plus [`docs/masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) and
   [`docs/masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md) before lane work.
2. **Full Kitchen Pass-2 Load (`lead_fp_0606d` on `606-998`) or Splice of `lead_fp_0751e` (`751-785`)**:
   - Check the owner's verdict on `lead_fp_0751e` in `data/fun/owner_answers.jsonl` / `items.json`.
   - Either lay the full `606-998` Pass-2 load on `shot_c_0604e` (rendered against the 24.0 fps post-Pass-1 cut with
     `shot_c_0604e`'s clean blonde hair keep-out + orphaned bouffant bridge) or splice `lead_fp_0751e` (`751-785`, where
     `747-750` is already a zero-mask back-of-head drop boundary) into `lead_fp_0606c` (`606-998`).
3. **Remaining `fun` Gap Shots (from `data/2026-10-19_fun_delivery_tables/frame_coverage_fun_0000_6394_*.md`)**:
   - **Hallway far table `1419-1465` (`47f`)**: `third` (`fun_dad.jpg`, clean-shaven) and `second` (`fun_mom_edit.jpg`,
     `--no-face-on` on `1419-1427, 1437-1452`), sequential Pass 1 -> Pass 2 using the post-Pass-1 video for Pass 2.
   - **Mid-song multi-subject and single-subject shots**: `1093-1133`, `1134-1155`, `1466-1729`, `1802-1848`,
     `2013-2049`, `2050-2622` (windowed `<= 362` frames). Every load runs `bench/facts_pass.py` before rendering and
     goes to `data/fun/review/items.json` (`load_chain.py page`) + `rebuild_delivery.py` afterward.

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
