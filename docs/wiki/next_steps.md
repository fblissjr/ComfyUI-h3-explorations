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
1. **Open steering / control-vector items ([`../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../research/control_vectors/2026-11-09_control_vector_simplification_plan.md) §17, §19, §24, §25, §26, §27)**:
   - **Owner look at the two vector-round stacked clips (§25)**: `Video/edit_guidance/stacked/diner16_s<seed>_stacked.mp4` (`data/results/2026-11-09_edit_guidance/vector_round/`), which decides the `mask` default and whether in-place vectors (`from-prompt`) are worth keeping alongside edit guidance.
   - **J-Lens lexical-half builder (§17.2, §20.1)**: Build the lexical half from the full 64-layer weights in the model store, move the word map into tracked code, and stop `bench/control_vector/j_lens_builder.py` from overwriting its own source bundle.
   - **Apply-node reshape and CLI cut-down (§9, §15.2, §19, §24, §27)**: Reshape `MiniMaxH3ApplyControlVectors` and cut down `bench/control_vector/cli.py`, then revisit `comfyui_nodes_and_controls.md` and `preset_catalog_and_cli_reference.md`.
   - **System prompt wiring & pre-existing red checks (§15.5, §26.1)**: Wire `bench/control_vector/llm_pairs.py::_extract_system_prompt_text` to [`system_prompts/h3_control_vector_standalone_system_prompt.md`](system_prompts/h3_control_vector_standalone_system_prompt.md) (with `bench/check_prompt_docs_sync.py`), and fix or triage `bench/check_reference_encode.py`, `bench/check_audio_freeze.py`, and `bench/check_two_shot_load.py`.
2. **Masked V2V: `fun` is closed, `fun2` is next, in the order of [`../masked_v2v/2026-11-10_simplification_plan.md`](../masked_v2v/2026-11-10_simplification_plan.md)** (corrected 2026-11-10: this item used to send the next session to resume `fun`'s remake queue and the three-pass remake on `1419–1465`, which was laid on 2026-10-31):
   - Step 1, the lane's records in SQLite, is the session `mrsql`'s; the state and the rest of the order are in [`handover.md`](handover.md), "Masked V2V".

## Historical Queues (`2026-09-05` through `2026-10-20`)

- **Archived chronological queues (`2026-09-05` through `2026-10-20`)**: [`archive/2026-10-20_next_steps_history.md`](archive/2026-10-20_next_steps_history.md).
