---
name: masked-v2v-audit
description: Upstream data-dependency, statistical outlier, frame-alignment, and visual verification workflow for H3 masked V2V pipelines. Invoke when auditing upstream inputs before queuing renders or remakes, or when debugging unexpected behavior across multi-stage video pipelines.
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
3. **Visually Audit Intermediate Conditioning Against the Source Film**:
   - Never treat intermediate stages (masks, keep-outs, 3D mesh renders, pose landmarks, crop boxes, or generated prompt text) as black boxes.
   - Extract side-by-side contact sheets comparing the raw source frames against the intermediate conditioning across transitions, rapid motion, occlusions, and flagged spans.
4. **Verify Differential Results on Close-Up Contact Sheets Before Paging**:
   - After rendering and landing, extract high-resolution close-up contact sheets comparing the original source, the prior attempt (if any), and the new output on the exact frames targeted before paging to the review server.

