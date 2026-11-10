# Handover: read this before anything else

Rewritten `2026-11-09T22:05Z` by the Gemini lead (coordinated with Claude in `../../h3_worktrees/claude-ph3-te32b`).
Where this page and a record disagree, the record is right: `git log`, [`decisions.md`](decisions.md), [`edit_guidance.md`](edit_guidance.md), [`control_vectors.md`](control_vectors.md), [`../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../research/control_vectors/2026-11-09_control_vector_simplification_plan.md), the lane rules in [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), and `data/fun/`.

## State when closed

- **Edit Guidance & Shipped Workflows (`2026-11-09`: `eea0b84b`..`77610090`)**:
  - Added `MiniMaxH3PromptEdit` (`original`, `edits`) and `MiniMaxH3EditGuider` (`model`, `original`, `edited`, `scale`, `first_step`, `align_positions`) in [`../../edit_guidance.py`](../../edit_guidance.py), pure-stdlib [`../../h3_prompt_edit.py`](../../h3_prompt_edit.py), [`../../h3_rules.py`](../../h3_rules.py) (`EDIT_GUIDANCE_DEFAULTS`: `scale=1.5`, `first_step=0`, `align_positions=False`), and [`../../bench/check_edit_guidance.py`](../../bench/check_edit_guidance.py).
  - Generated two shipped Pika 6-step edit-guidance workflows via `workflows/build_workflows.py::_wire_edit_guidance` and `prompt_bank/edit_guidance/*.txt`: [`../../workflows/h3_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_text_to_video_pika_edit_guidance_api.json) and [`../../workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json) (the first shipped workflow using `MiniMaxH3EncodeReferences` + `MiniMaxH3PromptOnReferences`, with the prompt written once on a single `PrimitiveStringMultiline` node `"133"` wired to both the original conditioning node `"5"` and `MiniMaxH3PromptEdit` `"130"`).
  - Updated graph walkers and static checks (`reference_order.py::plan_for`, `workflows/prompts.py::_follow` and `describe`, `bench/preflight_graph.py`, `bench/check_prompt_guide_conformance.py`, `bench/check_ref_prompt_labels.py`, `bench/audit_shipped_reference_bounds.py`, `bench/clip_recipe_coverage.py`, `bench/grade_prompt_text.py`) to follow `MiniMaxH3EncodeReferences`, `MiniMaxH3PromptEdit`, and `PrimitiveStringMultiline` links cleanly and fail loudly if a linked prompt cannot be resolved.
- **Control Vectors, Row Cache, Pooling & Wiki Rewrite (`2026-11-09`: `ed5f53fa`..`77610090`)**:
  - Retired `canonical_frames.json`; `MiniMaxH3MakeControlVector` and `bench/control_vectors.py` (`from-prompt`, `from-toml`, `from-words`, `from-images`) contrast `old -> new` edits in-place against `target_prompt` (`diff_span` default stays and no shared bank mean is subtracted per §20.2; `min_shared_run = 8` and `allow_multi_span` in `[vector]`).
  - Added token-ID `scene_only` span masking (`h3_spans.py`, [`../../bench/control_vector_rows.py`](../../bench/control_vector_rows.py) with `--ref-image` and `--only-picture-1`), pooling comparison ([`../../bench/control_vector_pooling.py`](../../bench/control_vector_pooling.py)), label comparison ([`../../bench/control_vector_lexicon.py`](../../bench/control_vector_lexicon.py)), edit-guidance/vector render arm runner ([`../../bench/edit_guidance_arms.py`](../../bench/edit_guidance_arms.py)), and `control_vectors.strength_ranges` from `NORM_SHIFT_WINDOW` (printing the constant default-mode `add_norm_matched` range first and the stored-length `add` range second).
  - Rewrote [`control_vectors.md`](control_vectors.md) and all 7 sub-pages under `docs/wiki/control_vectors/` from the code (`77610090`), and added [`edit_guidance.md`](edit_guidance.md).
- **PrunaAI `P-H3-Image` Default Encoder (`842a4108`)**:
  - The six generated `workflows/image/h3_p_h3_image_*` graphs (`t2i`, `edit`, `lora_t2i`, `lora_edit`, `t2i_control_vector`, `edit_control_vector`) now default to the 32B `MiniMaxH3EncoderLoader`; the two probe graphs (`h3_probe_p_h3_image_*_te4b_api.json`) hold the 4B `PH3ImageTEModel` + `TextConnector` path.
- **Standing Masked V2V State (`data/fun/`, unchanged since `2026-10-31T1702Z`)**:
  - Full-video delivery rebuild `2026-10-31T1702Z` stands (`87` rows in `data/fun/laid.jsonl`, `sha256 b2894f97f43a24fc`, `0` failed proofs in `Video/fun/day_2026-10-31/fun_0000_6394_joined_2026-10-31T1702Z_from_laid.mp4`).
  - ComfyUI (`:8188`) is running and idle on the updated node schema; worktrees `../../h3_worktrees/claude-ph3-te32b` (`77610090`, clean), `../../h3_worktrees/agy-laid-carve` (`91ad4237`, unmerged masked-lane `bench/laid.py` build), and `../../h3_worktrees/gem-perception-lhf` (`16b0805c`, unmerged masked-lane perception branch) remain on disk.

## What not to assume

- **Three pre-existing red checks on `main` (unrelated to `2026-11-09` work)**:
  - `bench/check_reference_encode.py` (upstream ComfyUI encoder snapshot drift), `bench/check_audio_freeze.py` (`song_allows_over_max_with_window` case), and `bench/check_two_shot_load.py` (`second_pass_from_cut` test fixture trips `no_mask_statement_has_mask`). Do not mask them in `bench/checks_baseline.json` without fixing or getting an explicit owner decision.
- **Generator validation on `:8188` when placeholder input files are absent**:
  - `workflows/build_workflows.py` validates against `:8188` by default (`--no-validate` skips it). On this machine, `H3_INPUT_DIR` currently lacks the placeholder reference media (`1-man.png`, etc.), so a plain `build_workflows.py` run fails validation with 89 missing-file errors unless `--no-validate` is passed or the placeholders are present.
- **J-Lens builder overwrites its source bundle, and pair bank / UI status**:
  - `bench/control_vector/j_lens_builder.py` (`build-j-lens`) recomputes only the refiner half, copies the lexical half out of the existing bundle on disk, and overwrites that file by default (§17.1). The 32 `.toml` files in `prompt_bank/control_vectors/` stay as a starter set and are not being grown (`decisions.md`); the browser UI (`bench/control_vectors.py ui`) is not maintained at present (`docs/wiki/control_vectors/studio_ui_guide.md`).
- **Standing `data/fun/` precautions**:
  - Do not render hand-patched `second_hfp_1419f`, do not remake kept pieces from `>= 2026-10-28` unless flagged on playback, and do not touch `data/fun/laid.jsonl` row 87 (`lead_fp_6125c`) until the owner decides on `6367–6372`.

## Tomorrow, in order

1. **Open items from [`../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../research/control_vectors/2026-11-09_control_vector_simplification_plan.md) (§17, §19, §24, §25, §26, §27)**:
   - **Owner playback of the two vector-round stacked clips (§25)**: `Video/edit_guidance/stacked/diner16_s<seed>_stacked.mp4` (`data/results/2026-11-09_edit_guidance/vector_round/`), which decides the `mask` default (`scene_only`, `non_vision`, `all_rows`) and whether in-place vectors (`from-prompt`) are worth keeping alongside edit guidance.
   - **J-Lens lexical-half builder (§17.2, §20.1)**: Build the lexical half from the full 64-layer weights in the model store, move the word map into tracked code, and make `bench/control_vector/j_lens_builder.py` write to a new path instead of overwriting its source bundle.
   - **Apply-node reshape and CLI cut-down (§9, §15.2, §19, §24, §27)**: Reshape `MiniMaxH3ApplyControlVectors` and simplify `bench/control_vector/cli.py` + merge `build_cmd.py`/`make_cmds.py`, then revisit `comfyui_nodes_and_controls.md` and `preset_catalog_and_cli_reference.md` (§27).
   - **System prompt wiring & pre-existing red checks (§15.5, §26.1)**: Wire `bench/control_vector/llm_pairs.py::_extract_system_prompt_text` to `docs/wiki/system_prompts/h3_control_vector_standalone_system_prompt.md` (with `bench/check_prompt_docs_sync.py`), and fix or triage `check_reference_encode.py`, `check_audio_freeze.py`, and `check_two_shot_load.py`.
2. **When resuming `data/fun/` masked V2V ([`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md) §6)**:
   - Land `../../h3_worktrees/agy-laid-carve` (`91ad4237`) before laying the first remake, tighten `valid_owner_word` in `bench/capture_masked_run.py` before signing a hard-flag override, run the 3-pass remake on `1419–1465` (`second_hfp_1419g` on `shot_c_1419i` first), update row 87 (`lead_fp_6125c`, `6125–6372`) once the owner decides on `6367–6372`, and continue the `< 2026-10-28` remake queue.

## Where things are

- **Start-of-day reading order** ([`../../AGENTS.md`](../../AGENTS.md)):
  1. [`handover.md`](handover.md) (this file), [`next_steps.md`](next_steps.md), and [`decisions.md`](decisions.md) (`2026-11-09`).
  2. [`working_with_the_owner.md`](working_with_the_owner.md) and [`../comfy_notes.md`](../comfy_notes.md).
  3. For steering & edit guidance: [`edit_guidance.md`](edit_guidance.md), [`control_vectors.md`](control_vectors.md), and [`../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../research/control_vectors/2026-11-09_control_vector_simplification_plan.md).
  4. For masked V2V (`data/fun/`): [`../../data/AGENTS.md`](../../data/AGENTS.md), [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), [`../masked_v2v/CODE_MAP_INDEX.md`](../masked_v2v/CODE_MAP_INDEX.md), and [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md).
