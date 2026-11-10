# Next steps

**A written router page, not generated. It says what to do next and where the reasoning lives; it restates none of it.** Kept short on purpose: when it disagrees with [`handover.md`](handover.md) or [`../roadmap.md`](../roadmap.md), those files are right.

## Starting a day: read [`handover.md`](handover.md) first

- **Daily boundary & active execution order (`2026-11-09`)**: [`handover.md`](handover.md) ("State when closed", "What not to assume", "Tomorrow, in order", and "Where things are").
- **Repo-level roadmap & open/closed lanes**: [`../roadmap.md`](../roadmap.md).
- **Active decisions log (`2026-10-23` through `2026-11-09` + date router)**: [`decisions.md`](decisions.md).
- **Steering & edit-guidance docs and plan**: [`edit_guidance.md`](edit_guidance.md), [`control_vectors.md`](control_vectors.md), and [`../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../research/control_vectors/2026-11-09_control_vector_simplification_plan.md).
- **Masked lane operative rules, live backlog, protocol, & code map**: [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), [`../masked_v2v/CODE_MAP_INDEX.md`](../masked_v2v/CODE_MAP_INDEX.md), and [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md).

## Current Order (`2026-11-09`)

Follow [`handover.md`](handover.md) ("Tomorrow, in order") and [`decisions.md`](decisions.md) (`2026-11-09`):
1. **Steering / repo hygiene follow-ups ([`../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../research/control_vectors/2026-11-09_control_vector_simplification_plan.md) §26)**:
   - Wire `bench/control_vector/llm_pairs.py::_extract_system_prompt_text` to [`system_prompts/h3_control_vector_standalone_system_prompt.md`](system_prompts/h3_control_vector_standalone_system_prompt.md) and update `bench/check_prompt_docs_sync.py`.
   - Fix or triage the three pre-existing red checks on `main` (`bench/check_reference_encode.py`, `bench/check_audio_freeze.py`, `bench/check_two_shot_load.py`).
2. **When resuming `data/fun/` masked V2V ([`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md) §6)**:
   - Land `../../h3_worktrees/agy-laid-carve` (`91ad4237`) before laying the first remake, and tighten `valid_owner_word` in [`bench/capture_masked_run.py`](../../bench/capture_masked_run.py) before signing a hard-flag override.
   - Run the three-pass remake on `1419–1465` (`second` Pass 1 `second_hfp_1419g` on `shot_c_1419i` -> `lead` Pass 2 -> `third` Pass 3), inspect the `1702Z` whole-video stacked cut and `lead_fp_6125c` tail `6367–6372` on `:8765`, and continue the `< 2026-10-28` remake queue.

## Historical Queues (`2026-09-05` through `2026-10-20`)

- **Archived chronological queues (`2026-09-05` through `2026-10-20`)**: [`archive/2026-10-20_next_steps_history.md`](archive/2026-10-20_next_steps_history.md).
