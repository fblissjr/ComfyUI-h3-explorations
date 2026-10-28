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

1. **Trace the Live Dependency Chain & Frame Alignment**:
   - Inspect what stage writers actually produce and consume in code right now. Do not assume an existing preview, keep-out, pose file, reference video, or carried parameter (`--drop`, `--hold-part`, `--least`, `--left-out`) is still valid or aligned.
   - Verify that every input video, mask, and array feeding a job has the exact same start frame, frame count, and source cut.
2. **Interrogate Summary Statistics Against Per-Frame Reality (`Least / Median / Most`)**:
   - Whenever any stage prints summary metrics (`least / median / most`, whole-shot bounding box vs. per-frame box, area steps, or score floors), compare the extremes and whole-shot aggregates against the median frame.
   - A large gap between the aggregate/extremes and the median means a small number of outlier frames (occlusions, motion blur, tracker jumps) are distorting whole-shot behavior. Inspect those frames directly on disk.
3. **Visually Audit Intermediate Conditioning Against the Source Film**:
   - Never treat intermediate stages (masks, keep-outs, 3D mesh renders, pose landmarks, crop boxes, or generated prompt text) as black boxes.
   - Extract side-by-side contact sheets comparing the raw source frames against the intermediate conditioning across transitions, rapid motion, occlusions, and flagged spans.
4. **Verify Differential Results on Close-Up Contact Sheets Before Paging**:
   - After rendering and landing, extract high-resolution close-up contact sheets comparing the original source, the prior attempt (if any), and the new output on the exact frames targeted before paging to the review server.
