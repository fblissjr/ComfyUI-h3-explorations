# docs/masked_v2v/: the masked video-to-video lane

Loaded when a session opens a file here; [`docs/AGENTS.md`](../AGENTS.md) and the root [`AGENTS.md`](../../AGENTS.md) still govern.

- **Start here**:
  - Execution & debugging protocol: [`docs/masked_v2v/PROTOCOL.md`](PROTOCOL.md)
  - Lane rules and where state lives: [`data/AGENTS.md`](../../data/AGENTS.md) (and each project's untracked `data/<video>/AGENTS.md`)
  - End-to-end pipeline, design, and card runbook: [`docs/wiki/masked_v2v_pipeline.md`](../wiki/masked_v2v_pipeline.md), [`docs/wiki/masked_v2v.md`](../wiki/masked_v2v.md), and [`docs/wiki/runbook.md`](../wiki/runbook.md)
- **This lane is one use of H3.** A finding about common code (reference conditioning, the shared nodes) is written where that code's page is and linked from [`docs/wiki/index.md`](../wiki/index.md).

## Top six rules that bite when missed

### 1. What `--source` to pass in (Single-Subject vs Multi-Subject Pass 2)

- [`bench/shot_to_load.py`](../../bench/shot_to_load.py) `preview` (`bench/shot_to_load.py::step_preview`) requires `--source VIDEO` with **no silent default**. Never assume `data/<video>/shots.json`'s original raw video is the right source for the pass in hand.
- **Single-subject shot or Pass 1 of a multi-subject shot**: pass the raw 24.0 fps working-copy source video (`fun_0000_6394_24p_sound_in_step.mov`).
- **Pass 2+ of a multi-subject shot (where Pass 1 is already laid in `data/<video>/laid.jsonl`)**: **never** run `preview` or `job` against the raw source video. Always build or use a post-Pass-1 joined cut ([`bench/assemble_delivery.py`](../../bench/assemble_delivery.py) / `fun_..._from_laid.mp4` where Subject A is already replaced at 24.0 fps) and run a fresh `bench/shot_to_load.py preview --source <post_pass1_video>` for Pass 2.
- **Why (Kitchen `604-1009` `second` + `lead` crossing on `755-774`, 2026-10-19)**: The original 1980s mom (`second`) had a tall dark bouffant hairdo rising ~60 px higher (`ymin=188..209`) than the replaced blonde mom (`second_hfp_0604h`, `ymin=211..221`). Reusing a preview segmented on the raw 1980s video gave `second`'s keep-out `5,843–9,276 px` of old bouffant hair dome and carved an invisible dark helmet out of `lead`'s face as `lead` walked behind her.

### 2. Bidirectional Temporal Attention & Unmasked Context Leaks (The Beard Rule)

- H3's DiT uses full bidirectional temporal attention across the entire window (`W` frames, `5n + 2` latent steps for `17n + 5` frames). Even 3–4 frames where an original subject's salient feature (such as a grey beard or eyeglasses) is left unmasked in the frozen context will propagate that feature onto the regenerated subject across the entire window.
- **Two causes caught on Hallway `1306-1418` (`third`, 2026-10-19)**:
  1. **Transient Sapiens class-map dips (`<= 6` frames) during fast motion**: on `1402-1404` and `1408-1411`, Sapiens `Face_Neck` dipped below 50% of flanking frames while the person track stayed solid (`51,735 px` of exposed beard). Fixed automatically by `bench/head_tube_parts.py::bridge_short_tube_dips` (`bench/head_tube_parts.py::SHORT_DIP_MAX_FRAMES` = `6`, `bench/head_tube_parts.py::SHORT_DIP_RATIO` = `0.6`).
  2. **Dropping interior frames where the face is still visible**: `--drop 1411-1413` zeroed the regeneration mask on frames that still had `16,409–48,938 px` of visible `Face_Neck` + beard. Guarded by `bench/capture_masked_run.py::dropped_or_unmasked_face_in_context` preflight blocking flag `dropped_frame_has_visible_face` (`bench/capture_masked_run.py::DROPPED_FACE_MIN_PX` = `4000`). **Never `--drop` a frame where the subject's face is still visible to the camera.** On back- or side-of-head turns, keep the regeneration mask open and pass `--no-face-on` to [`bench/head_reference.py`](../../bench/head_reference.py) so `<Video 1>` draws no facial marks there.

### 3. Always pass `--cast data/<video>/cast/cast.jsonl` and verify it exists

- [`bench/shot_to_load.py`](../../bench/shot_to_load.py) `preview` and `job` refuse missing `--cast` files cleanly (`bench/shot_to_load.py::Refused`). Without `--cast data/<video>/cast/cast.jsonl`, the builder cannot auto-identify `lead` vs `second` vs `third` from confirmed reference embeddings.

### 4. Pre-render & Post-capture Telemetry (`facts_pass.py`, `step_carves.json`, `code_stamp.py`)

- Run [`bench/facts_pass.py`](../../bench/facts_pass.py) on the capture directory to inspect `face_outside_region_px` (catches chin/cheek cutouts where `(s_face | (head_track & ~s_hair))` falls outside `region`), `other_overlap_px`, and `mouth_mode` before spending GPU render minutes.
- Inspect `runs/<run>/step_carves.json` (`bench/capture_masked_run.py::step_keepout_carves`) for latent-step keep-out cell carves touching the subject's head track or carried part.
- All derived `.npz` / `.json` artifacts carry `writer_stamp` (`bench/_lib/code_stamp.py::writer_stamp`, `bench/_lib/code_stamp.py::npz_writer_fields`), and [`bench/head_reference.py`](../../bench/head_reference.py), [`bench/draw_face_points.py`](../../bench/draw_face_points.py), and [`bench/measure_moge2_depth.py`](../../bench/measure_moge2_depth.py) check upstream mtimes via `bench/_lib/code_stamp.py::check_upstream_mtime` and refuse stale upstream inputs.

### 5. Review Server (`items.json`), Verdicts (`owner_answers.jsonl`), and Static `poc/TO_REVIEW.html`

- The owner reviews clips and submits verdicts and frame notes on the interactive review page (`http://<host>:8765/`, served by `data/<video>/poc/serve.py` from `data/<video>/poc/index.html`), which reads `data/<video>/review/items.json` (`state: "open"` renders under `"1. Look at these next"`) and appends clicks to `data/<video>/owner_answers.jsonl`.
- Every new review clip **must** be added to `data/<video>/review/items.json` (`state: "open"`, via `bench/load_chain.py page`) **and** synced to `data/<video>/poc/TO_REVIEW.html` ([`bench/rebuild_delivery.py`](../../bench/rebuild_delivery.py) rebuilds `TO_REVIEW.html` and `PROVENANCE.html` and verifies `data/<video>/poc/` on disk via `bench/rebuild_delivery.py::verify_poc_on_disk` during the `pages` step; [`bench/build_cut_map.py`](../../bench/build_cut_map.py) builds the 2-tab chronological cut table, frame-level gap audit, and 6-stage hierarchical data flow lineage DAG in `data/<video>/poc/PROVENANCE.html`, `data/<video>/poc/CUT_MAP.html`, and `data/<video>/CUT_MAP.md`).

### 6. 24.0 fps End-to-End, the `17n + 5` Grid, and Dual-Clock Audio Sync (`on_both_clocks`)

- **Exact `24.0` fps (`24/1`) everywhere**: Every working copy, preview, reference, render, and assembled multi-pass delivery cut ([`bench/rebuild_delivery.py`](../../bench/rebuild_delivery.py), [`bench/assemble_delivery.py`](../../bench/assemble_delivery.py)) is forced to `24.0` fps (`bench/shot_to_load.py::working_copy` and [`bench/masked_job.py`](../../bench/masked_job.py) refuse any non-24 fps source).
- **Video grid (`17n + 5`) vs. Audio clock (`h3_rules.on_both_clocks`)**: Video frame counts round up to `17n + 5` (`5n + 2` latent steps, up to `h3_rules.MAX_LENGTH` = `362`; see [`docs/h3_resolutions.md`](../h3_resolutions.md), "The length axis"). Audio latents run at `h3_rules.AUDIO_LATENT_RATE` (`round(frames / 24 * AUDIO_LATENT_RATE)`), so the video and audio clocks agree **only on every third `17n + 5` length (`39, 90, 141, 192, 243, 294, 345`, `h3_rules.on_both_clocks`)**. Chained multi-window seams (`h3_rules.chain_lengths`, e.g. `[345, 141]` with `90`-frame context on Kitchen `606-998`) **must** meet on dual-clock lengths so audio timing never drifts across windows, and final laid pieces should render the whole shot (`docs/wiki/masked_v2v.md`, "A shot, made whole") rather than relying on ultra-short sub-window diagnostic clips.
