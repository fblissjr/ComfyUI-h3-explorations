# Masked video to video: router index & lane overview

last updated: 2026-10-27 (converted to a `< 5 KB` router index pointing to `docs/masked_v2v/` for current operative rules, execution protocol, and code map; the full `2026-10-04..2026-10-15` design chronicle is preserved verbatim in [`archive/2026-10-15_masked_v2v_chronicle.md`](archive/2026-10-15_masked_v2v_chronicle.md)).

## Current Operative Rules, Protocol & Code Map (`docs/masked_v2v/`)

| Topic | Canonical Document |
|---|---|
| **Operative lane rules & index (`§1`–`§11`)** | [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) |
| **Execution & debugging CLI commands (Stages 0–5)** | [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md) |
| **What a change forces (Stages 0–7 in trace order) & `her_hair_in_hole`** | [`../masked_v2v/chain_trace_and_remakes.md`](../masked_v2v/chain_trace_and_remakes.md) |
| **Head reference (`<Video 1>`) mouth/eye drawing, crops (`cranium`), & `<Subject 2>` prompts** | [`../masked_v2v/reference_and_prompt_rules.md`](../masked_v2v/reference_and_prompt_rules.md) |
| **AST jump index, 7-stage contracts, & simplification plan (`2026-10-26`)** | [`../masked_v2v/CODE_MAP_INDEX.md`](../masked_v2v/CODE_MAP_INDEX.md), [`../masked_v2v/code_map_stages.md`](../masked_v2v/code_map_stages.md), [`../masked_v2v/code_map_contracts.md`](../masked_v2v/code_map_contracts.md), [`../research/masking/2026-10-26_code_map_and_simplification_plan.md`](../research/masking/2026-10-26_code_map_and_simplification_plan.md) |
| **Perception, stage-split & modularization branch (`gem/perception-lhf`, `2026-10-27`)** | [`../masked_v2v/2026-10-27_perception_lhf_branch.md`](../masked_v2v/2026-10-27_perception_lhf_branch.md) |
| **Verifying raw frames vs. `.npz` / JSON records (`laid.py keeps`, zoom boxes)** | [`../masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md) |
| **Pass contamination, perception provenance, & `inspect_mask_steps.py` (`2026-10-25`)** | [`../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md`](../masked_v2v/2026-10-25_pass_contamination_and_mask_step_forensics.md) |
| **Window lengths (`h3_rules.on_both_clocks`), 17-frame VAE clips, & padding** | [`../masked_v2v/frames_rules.md`](../masked_v2v/frames_rules.md) |
| **Where per-video state lives (`data/<video>/`)** | [`../../data/AGENTS.md`](../../data/AGENTS.md) |
| **Historical chronicle (`2026-10-04` through `2026-10-15`)** | [`archive/2026-10-15_masked_v2v_chronicle.md`](archive/2026-10-15_masked_v2v_chronicle.md) |

## The frame every masked text is written in

Every masked load prompt is generated from `plan.json` and the reference record by [`bench/load_text.py`](../../bench/load_text.py) (checked by [`bench/check_load_text.py`](../../bench/check_load_text.py)):
- **Current prompt specification (`<Subject 1>`, `<Subject 2>`, `[Shot 1]`, mouth & eye clauses)**: [`../masked_v2v/reference_and_prompt_rules.md`](../masked_v2v/reference_and_prompt_rules.md).
- **General H3 prompting reference**: [`../prompting.md`](../prompting.md) and [`../h3_references.md`](../h3_references.md).
- **Historical `2026-10-04..2026-10-15` text experiments**: [`archive/2026-10-15_masked_v2v_chronicle.md`](archive/2026-10-15_masked_v2v_chronicle.md).

## A shot, made whole

Every shot of `<= 345` frames (`max(h3_rules.chain_lengths())`) renders as one single-window load on both clocks (`h3_rules.on_both_clocks`: `39, 90, 141, 192, 243, 294, 345`) at exact `24.0` fps (`24/1`), with each replacement pass built from a fresh preview on the newest whole-video cut:
- **Current whole-shot rules (`§1`, `§6`, `§8`)**: [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) and [`../masked_v2v/chain_trace_and_remakes.md`](../masked_v2v/chain_trace_and_remakes.md).
- **Step-by-step CLI commands (`shot_to_load.py`, `load_chain.py`)**: [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md).
- **Historical `2026-10-15` hall-shot recipe**: [`archive/2026-10-15_masked_v2v_chronicle.md`](archive/2026-10-15_masked_v2v_chronicle.md).

## A latent step is several frames, and its grid is fixed for a whole load

H3's video VAE encodes 17-frame clips into `(1, 4, 4, 4, 4)` latent steps (`video_mask.FRAME_PER_TOKEN`), and the DiT attends bidirectionally across the entire window:
- **Current frame math, latent-step boundaries, & the 2026-10-25 three-path context rule**: [`../masked_v2v/frames_rules.md`](../masked_v2v/frames_rules.md), [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md) §2 & §6, and [`../research/masking/2026-10-25_context_frames_and_identity_on_2050.md`](../research/masking/2026-10-25_context_frames_and_identity_on_2050.md).
- **Earlier window-context and edge studies**: [`window_context.md`](window_context.md), [`window_edges_data.md`](window_edges_data.md), and [`archive/2026-10-15_masked_v2v_chronicle.md`](archive/2026-10-15_masked_v2v_chronicle.md).

## A small subject, rendered zoomed in

`video_mask.py::zoom_box` (`render_region: "zoomed"`) crops and enlarges the window around the union of the subject's region boxes up to `plan.json`'s settled `zoom_up_to` (enforced by `cbd39fbe` at `bench/shot_to_load.py` and `bench/flags_before_render.py::recipe_check_of`):
- **Current zoom rules, `floor_zoom`, & `cranium` reference crop**: [`../masked_v2v/chain_trace_and_remakes.md`](../masked_v2v/chain_trace_and_remakes.md) (Stage 5) and [`../masked_v2v/reference_and_prompt_rules.md`](../masked_v2v/reference_and_prompt_rules.md).
- **Historical zoomed-render chronicle**: [`archive/2026-10-15_masked_v2v_chronicle.md`](archive/2026-10-15_masked_v2v_chronicle.md).
