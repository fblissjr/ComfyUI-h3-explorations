# Handover: read this before anything else

Rewritten `2026-11-09T22:00Z` by the Gemini lead (coordinated with Claude in `../../h3_worktrees/claude-ph3-te32b`).
Where this page and a record disagree, the record is right: `git log`, [`decisions.md`](decisions.md), [`edit_guidance.md`](edit_guidance.md), [`control_vectors.md`](control_vectors.md), [`../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../research/control_vectors/2026-11-09_control_vector_simplification_plan.md), the lane rules in [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), and `data/fun/`.

## State when closed

- **Edit Guidance & Shipped Workflows (`2026-11-09`: `eea0b84b`..`77610090`)**:
  - Added `MiniMaxH3PromptEdit` (`original`, `edits`) and `MiniMaxH3EditGuider` (`model`, `original`, `edited`, `negative`, `cfg`, `edit_scale`, `start_percent`, `end_percent`, `edit_stream`, `rescaled`) in [`../../h3_edit_guidance.py`](../../h3_edit_guidance.py), pure-stdlib [`../../h3_prompt_edit.py`](../../h3_prompt_edit.py), [`../../h3_rules.py`](../../h3_rules.py) (`EDIT_GUIDANCE_DEFAULTS`), and [`../../bench/check_edit_guidance.py`](../../bench/check_edit_guidance.py).
  - Generated two shipped Pika 6-step edit-guidance workflows via `workflows/build_workflows.py::_wire_edit_guidance` and `prompt_bank/edit_guidance/*.txt`: [`../../workflows/h3_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_text_to_video_pika_edit_guidance_api.json) and [`../../workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json) (the first shipped workflow using `MiniMaxH3EncodeReferences` + `MiniMaxH3PromptOnReferences`, with the prompt written once on a single `PrimitiveStringMultiline` node `"133"` wired to both the original conditioning node `"5"` and `MiniMaxH3PromptEdit` `"130"`).
  - Updated graph walkers and static checks (`reference_order.py::plan_for`, `workflows/prompts.py::_follow` and `describe`, `bench/preflight_graph.py`, `bench/check_prompt_guide_conformance.py`, `bench/check_ref_prompt_labels.py`, `bench/audit_shipped_reference_bounds.py`, `bench/clip_recipe_coverage.py`, `bench/grade_prompt_text.py`) to follow `MiniMaxH3EncodeReferences`, `MiniMaxH3PromptEdit`, and `PrimitiveStringMultiline` links cleanly and fail loudly if a linked prompt cannot be resolved.
- **Control Vectors, Row Cache, Pooling & Wiki Rewrite (`2026-11-09`: `ed5f53fa`..`77610090`)**:
  - Retired `canonical_frames.json`; `MiniMaxH3MakeControlVector` and `bench/control_vectors.py build --words` contrast `old -> new` edits in-place against `target_prompt` (`diff_span` default, `min_shared_run = 8`, `remove_common_direction = True` across `>= 4` pairs).
  - Added token-ID `scene_only` span masking (`h3_spans.py`, `bench/control_vector_rows.py`), reference-image row caching (`--ref-image`), pooling comparison (`bench/compare_pooling.py`), label comparison (`bench/compare_ref_labels.py`), edit-guidance/vector render arm runner (`bench/run_edit_guidance_arms.py`), and `recommended_strength` for `add_norm_matched` (`strength = round(raw_norm / 67.5, 2)`).
  - Rewrote [`control_vectors.md`](control_vectors.md) and all 7 sub-pages under `docs/wiki/control_vectors/` from the code (`77610090`), and added [`edit_guidance.md`](edit_guidance.md).
- **PrunaAI `P-H3-Image` Default Encoder (`842a4108`)**:
  - The four shipped `workflows/image/h3_p_h3_image_*` graphs now default to the 32B `MiniMaxH3EncoderLoader`; the two probe graphs (`h3_probe_p_h3_image_*_te4b_api.json`) hold the 4B `PH3ImageTEModel` + `TextConnector` path.
- **Standing Masked V2V State (`data/fun/`, unchanged since `2026-10-31T1702Z`)**:
  - Full-video delivery rebuild `2026-10-31T1702Z` stands (`87` rows in `data/fun/laid.jsonl`, `sha256 b2894f97f43a24fc`, `0` failed proofs in `Video/fun/day_2026-10-31/fun_0000_6394_joined_2026-10-31T1702Z_from_laid.mp4`).
  - ComfyUI (`:8188`) is running and idle on the updated node schema; worktrees `../../h3_worktrees/claude-ph3-te32b` (`77610090`, clean) and `../../h3_worktrees/agy-laid-carve` (`328ca0bc`, clean) remain on disk.

## What not to assume

- **Three pre-existing red checks on `main` (unrelated to `2026-11-09` work)**:
  - `bench/check_reference_encode.py` (upstream ComfyUI encoder snapshot drift), `bench/check_audio_freeze.py` (`song_allows_over_max_with_window` case), and `bench/check_two_shot_load.py` (`second_pass_from_cut` test fixture trips `no_mask_statement_has_mask`). Do not mask them in `bench/checks_baseline.json` without fixing or getting an explicit owner decision.
- **Generator validation on `:8188` when placeholder input files are absent**:
  - `workflows/build_workflows.py` validates against `:8188` by default (`--no-validate` skips it). On this machine, `H3_INPUT_DIR` currently lacks the placeholder reference media (`1-man.png`, etc.), so a plain `build_workflows.py` run fails validation with 89 missing-file errors unless `--no-validate` is passed or the placeholders are present.
- **Pair bank and browser UI status (owner, `2026-11-09`)**:
  - The 32 `.toml` files in `prompt_bank/control_vectors/` are kept as a starter set and are not being grown (`decisions.md`). The control-vector browser UI (`bench/control_vectors.py ui`) was left for another day and is not maintained at present (`docs/wiki/control_vectors/studio_ui_guide.md`).
- **Standing `data/fun/` precautions**:
  - Do not render hand-patched `second_hfp_1419f`, do not remake kept pieces from `>= 2026-10-28` unless flagged on playback, and do not touch `data/fun/laid.jsonl` row 87 (`lead_fp_6125c`) until the owner decides on `6367–6372`.

## Tomorrow, in order

1. **Follow-ups from [`../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../research/control_vectors/2026-11-09_control_vector_simplification_plan.md) (§26), if continuing steering / repo hygiene**:
   - Wire `bench/control_vector/llm_pairs.py::_extract_system_prompt_text` to the standalone pair-authoring prompt (`docs/wiki/system_prompts/h3_control_vector_standalone_system_prompt.md`) and update `bench/check_prompt_docs_sync.py`.
   - Fix or triage the 3 pre-existing red checks on `main` (`check_reference_encode.py`, `check_audio_freeze.py`, `check_two_shot_load.py`).
2. **When resuming `data/fun/` masked V2V ([`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md) §6)**:
   - Land `../../h3_worktrees/agy-laid-carve` (`328ca0bc`, after fixing `_source_holds_other_subject` load-name matching and `replaces` handling) before laying the first remake, and tighten `valid_owner_word` in `bench/capture_masked_run.py` before signing a hard-flag override.
   - Run the 3-pass remake on `1419–1465` (`second_hfp_1419g` on `shot_c_1419i` first, citing `data/fun/owner_answers.jsonl:1739` on `1419–1423`), update row 87 (`lead_fp_6125c`, `6125–6372`) once the owner decides on `6367–6372`, and continue the `< 2026-10-28` remake queue.

## Where things are

- **Start-of-day reading order** ([`../../AGENTS.md`](../../AGENTS.md)):
  1. [`handover.md`](handover.md) (this file), [`next_steps.md`](next_steps.md), and [`decisions.md`](decisions.md) (`2026-11-09`).
  2. [`working_with_the_owner.md`](working_with_the_owner.md) and [`../comfy_notes.md`](../comfy_notes.md).
  3. For steering & edit guidance: [`edit_guidance.md`](edit_guidance.md), [`control_vectors.md`](control_vectors.md), and [`../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../research/control_vectors/2026-11-09_control_vector_simplification_plan.md).
  4. For masked V2V (`data/fun/`): [`../../data/AGENTS.md`](../../data/AGENTS.md), [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), [`../masked_v2v/CODE_MAP_INDEX.md`](../masked_v2v/CODE_MAP_INDEX.md), and [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md).
