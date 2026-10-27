# The masked video-to-video pipeline: router to the trace-order specification

Last updated: 2026-10-26 (replaced the 2026-10-16 stage table and flowchart—which predated `shot_to_load.py job` before `reference`/`text`, `strangers.py`, `on-plan`, `inspect_mask_steps.py`, `flags_before_render.py`, and `rebuild_delivery.py`—with direct pointers to the authoritative trace-order specification; root [`AGENTS.md`](../../AGENTS.md), "One home per kind of fact").

- **Authoritative stage order, inputs, outputs, and what a change to any stage forces**:
  - [`docs/masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) §8 ("What a change forces: the chain in trace order", stages `0` through `7`).
- **Step-by-step commands and pre-render / post-render audit gates**:
  - [`docs/masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md) (Stages `0` through `5`).
- **Window lengths (`h3_rules.on_both_clocks`), 17-frame VAE clips, held-edge padding, and post-decode frame slicing**:
  - [`docs/masked_v2v/frames_rules.md`](../masked_v2v/frames_rules.md) and [`docs/masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) §6.
- **Verifying raw frames vs. `.npz` / JSON caches and Pass-1 → Pass-2 provenance**:
  - [`docs/masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md) and [`docs/masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md).
- **Exact `file:line` call graphs, data contracts, and stage-by-stage mask mutations**:
  - [`docs/research/masking/2026-10-26_code_map_and_simplification_plan.md`](../research/masking/2026-10-26_code_map_and_simplification_plan.md), [`docs/masked_v2v/CODE_MAP_INDEX.md`](../masked_v2v/CODE_MAP_INDEX.md), [`docs/masked_v2v/code_map_stages.md`](../masked_v2v/code_map_stages.md), and [`docs/masked_v2v/code_map_contracts.md`](../masked_v2v/code_map_contracts.md).
