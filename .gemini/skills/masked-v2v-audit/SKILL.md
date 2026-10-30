---
name: masked-v2v-audit
description: Upstream data-dependency, statistical outlier, frame-alignment, and visual verification workflow for H3 masked V2V pipelines. Invoke when auditing upstream inputs before queuing renders or remakes, or when debugging unexpected behavior across multi-stage video pipelines.
reviewed: bbe72995
---

# Masked V2V Upstream & Pre-Render Diagnostic Skill

Passing an automated preflight gate (`0 likely to fail`) is necessary but not sufficient. Whenever working on masked V2V loads, apply the general diagnostic principles below and consult [`docs/masked_v2v/upstream_anomaly_and_alignment_checks.md`](../../../docs/masked_v2v/upstream_anomaly_and_alignment_checks.md) and [`docs/masked_v2v/chain_trace_and_remakes.md`](../../../docs/masked_v2v/chain_trace_and_remakes.md) §1 for the stage-by-stage reprocessing DAG.

## Core Diagnostic Habits (Always Apply)

1. **Trace Every Explicit and Hidden Dependency in the DAG (Never Assume Inputs Are Synced)**:
   - Audit every model stream and JSON/NPZ table feeding the stage — not just the primary track mask. In H3 masked V2V, multiple independent models feed the same stage without automatic cross-locking (e.g., 3D `SAM3DBody` `_mesh` vs. 2D `Sapiens2-Pose` `_points.npz` vs. `_pose_table.json` in `head_reference.py`; `MoGe-2` `depth.npz` in `shot_to_load.py job`; `seen_looks` in `shots.json`).
   - Check whether upstream diagnostic tables (`marks_against_the_body_model`, `face_score_mean`, `part_grades.json`, `seen_looks`) are actually acted on by the code or merely logged as `"decides: nothing"`. If the code logs an anomaly without acting on it, you must pass the corresponding override/exclusion flags (`--no-head-on`, `--no-face-on`, `--no-draw-face-on`, `--drop`, `--depth`) or fix the stage writer and reprocess downstream.
   - Verify that every input video, mask, and array feeding a job has the exact same start frame, frame count, and source cut.
2. **Interrogate Summary Statistics Against Per-Frame Reality (`Least / Median / Most`)**:
   - Whenever any stage prints summary metrics (`least / median / most`, whole-shot bounding box vs. per-frame box, area steps, or score floors), compare the extremes and whole-shot aggregates against the median frame.
   - A large gap between the aggregate/extremes and the median means a small number of outlier frames (occlusions, motion blur, tracker jumps) are distorting whole-shot behavior. Inspect those frames directly on disk.
3. **Visually Audit Intermediate Conditioning Against the Source Film Using Native `read_file` First**:
   - Never treat intermediate stages (masks, keep-outs, 3D mesh renders, pose landmarks, crop boxes, or generated prompt text) as black boxes.
   - Use `agy`'s built-in `read_file` (`view_file`) tool directly on the pre-built `.1fps.<first>-<last>.mp4` look strips (`first` and `on-plan`), `<Video 1>` reference `.mp4` and `_sheet.jpg`, and `bench/inspect_mask_steps.py --strips <out.jpg>` JPEGs first, and extract custom side-by-side contact sheets when zooming into transitions, rapid motion, occlusions, or flagged spans.
4. **Verify Landed Results With Native `read_file` on `.1fps.mp4` & `24 fps` `.mp4` (Plus Close-Up Sheets Where Needed) Before Paging**:
   - After rendering and landing, inspect the landed `<load>_<first>_<last>_stacked.1fps.<first>-<last>.mp4` strip and `24 fps` `.mp4` clip directly via `read_file` (plus close-up contact sheets comparing `ORIG`, prior attempt, and `CUT` on targeted frames) before paging to the review server.

