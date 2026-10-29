---
name: masked-v2v-audit
description: General upstream data-dependency, statistical outlier, frame-alignment, and visual verification workflow for H3 masked V2V pipelines. Always invoke before queuing renders or remakes, or whenever debugging unexpected behavior across multi-stage video pipelines.
---

# Masked V2V Upstream & Pre-Render Diagnostic Skill

Passing an automated preflight gate (`0 likely to fail`) is necessary but not sufficient. Whenever working on masked V2V loads, apply the general diagnostic principles below and consult the point-in-time reference in [`docs/masked_v2v/upstream_anomaly_and_alignment_checks.md`](../../../docs/masked_v2v/upstream_anomaly_and_alignment_checks.md) (linked from [`docs/masked_v2v/AGENTS.md`](../../../docs/masked_v2v/AGENTS.md) §11).

> [!WARNING]
> **Assume Drift or a Live Code Bug Until Verified Against Code and Disk**:
> 1. **Docs & Cutoffs Drift**: Every document, cache cutoff, dependency diagram, and threshold in `docs/masked_v2v/` reflects what was best known as of its snapshot date. Code, CLI flags, stage writers, model defaults, or upstream dependencies **may have changed or drifted since the docs were written**, or a new dependency link may exist that is not yet documented.
> 2. **It May Be a Bug in the Pipeline Code Itself**: When you spot a statistical outlier or visual anomaly (e.g., an upside-down 3D body mesh on an occluded frame shrinking the whole-shot `--crop cranium` box in `head_reference.py`, or SAM's head track clipping an upper hair crown in `head_tube_parts.py`), do not assume the code is right. Inspect the stage-writer source code (`file:line`) — it may be a code bug or unhandled edge case in a stage writer.
> 3. **Reprocess Everything Downstream of the Fix**: Whenever you fix a code bug or configuration input at Stage `X`, **every downstream stage in the DAG that consumes Stage `X`'s outputs is immediately stale and must be reprocessed** (for example: fixing `subject_track.py` requires reprocessing from `preview` -> `strangers` -> `pose` -> `reference` -> `first` -> `plan` -> `render`; fixing `head_reference.py` / `--no-head-on` requires reprocessing from `reference` -> `text` -> `first` -> `plan` -> `render`; fixing `head_tube_parts.py` requires reprocessing from `first` -> `plan` -> `render`).

## Core Diagnostic Habits (Always Apply)

1. **Trace Every Explicit and Hidden Dependency in the DAG (Never Assume Inputs Are Synced)**:
   - Audit every model stream and JSON/NPZ table feeding the stage — not just the primary track mask. In H3 masked V2V, multiple independent models feed the same stage without automatic cross-locking (e.g., 3D `SAM3DBody` `_mesh` vs. 2D `Sapiens2-Pose` `_points.npz` vs. `_pose_table.json` in `head_reference.py`; opt-in `MoGe-2` `--depth` in `shot_to_load.py job`; `seen_looks` in `shots.json`).
   - Check whether upstream diagnostic tables (`marks_against_the_body_model`, `face_score_mean`, `part_grades.json`, `seen_looks`) are actually acted on by the code or merely logged as `"decides: nothing"`. If the code logs an anomaly without acting on it, you must pass the corresponding override/exclusion flags (`--no-head-on`, `--no-face-on`, `--no-draw-face-on`, `--drop`, `--depth`) or fix the stage writer and reprocess downstream.
   - Verify that every input video, mask, and array feeding a job has the exact same start frame, frame count, and source cut.
2. **Interrogate Summary Statistics Against Per-Frame Reality (`Least / Median / Most`)**:
   - Whenever any stage prints summary metrics (`least / median / most`, whole-shot bounding box vs. per-frame box, area steps, or score floors), compare the extremes and whole-shot aggregates against the median frame.
   - A large gap between the aggregate/extremes and the median means a small number of outlier frames (occlusions, motion blur, tracker jumps) are distorting whole-shot behavior. Inspect those frames directly on disk.
3. **Visually Audit Intermediate Conditioning Against the Source Film**:
   - Never treat intermediate stages (masks, keep-outs, 3D mesh renders, pose landmarks, crop boxes, or generated prompt text) as black boxes.
   - Extract side-by-side contact sheets comparing the raw source frames against the intermediate conditioning across transitions, rapid motion, occlusions, and flagged spans.
4. **Verify Differential Results on Close-Up Contact Sheets Before Paging**:
   - After rendering and landing, extract high-resolution close-up contact sheets comparing the original source, the prior attempt (if any), and the new output on the exact frames targeted before paging to the review server.

## Handover Double-Check Required (`2026-10-29` Session)

> [!CAUTION]
> **NEW CODE, CONFIG & PROTOCOL CHANGES LANDED LATE `2026-10-28` — DOUBLE-CHECK FOR EDGE CASES OR REGRESSIONS AT HANDOVER**:
> 1. **`bench/crop_reference_to_head.py::cranium_box` (`CRANIUM_SCORE_FLOOR = 0.30`, `CRANIUM_FACE_OUTLIER = 2.50`)**:
>    - New automatic filter in `cranium_box()` (`lines 132–139, 272–290`) that excludes frames whose 2D face keypoints score `< 0.30` (median of `face_ids`) OR whose single-frame 2D face height exceeds `2.50x` the whole-shot median `face` height when deciding which frames (`good = seen & scored & in_scale`) size the whole-shot `--crop cranium` box (with fallback to `seen & kept` if no frame passes).
>    - **Handover check**: Verify on new `reference` runs that `cranium_box` does not over-exclude legitimate rapid close-ups or extreme expressions, and check `the_one_box_enlarges` vs `each_frame_alone_would_enlarge_least_median_most` on every new reference JSON.
> 2. **`data/fun/plan.json` Default `depth.on = true` for `lead`, `second`, and `third`**:
>    - Enabled `"depth": {"on": true, ...}` on all three subjects in `data/fun/plan.json` so `shot_to_load.py::make_depth` automatically runs MoGe-2 per-pixel depth (`bench/measure_moge2_depth.py`) on overlap frames (`overlap_frames_of`) whenever `strangers` or another subject's track overlaps the target subject (and skips in `0s` when `0` frames overlap).
>    - **Handover check**: Verify on both single-subject shots (with `strangers` overlap) and multi-subject crossing shots that `depth.npz` is generated cleanly, does not cause unexpected mask carving on non-overlapping frames, and does not break any `check_shot_to_load.py` or `capture_masked_run.py` invariants.
> 3. **End-to-End Dependency Freshness Protocol (`docs/masked_v2v/upstream_anomaly_and_alignment_checks.md`)**:
>    - Double-check the 5-row Hidden/Unsynced Cross-Model Dependencies table (`SAM3DBody` 3D `_mesh` vs `Sapiens2-Pose` 2D `_points.npz` collisions in `_pose_table.json`, 3D grey arms/hands inside `--crop cranium`, `cranium_box` low-score/outlier frames, `depth.npz`, and interior `not seen` spans in `shots.json`) against actual code and new renders tomorrow.
