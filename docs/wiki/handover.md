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
- **Masked V2V (corrected 2026-11-10 by a Claude session at the owner's ask; this block used to name the `1702Z` rebuild as standing, `87` rows, `b2894f97`)**:
  - **`fun` is closed** (the owner, 2026-11-10: it ended 2026-10-31 at about 17:30 to 18:00 Central). The final cut is
    `Video/fun/day_2026-10-31/fun_0000_6394_joined_2026-10-31T2231Z_from_laid.mp4`. Its record
    (`data/fun/2026-10-31_fun_delivery_tables/rebuild_fun_2026-10-31T2231Z.json`) and sidecar name the laid list
    `70ead45b...`, which is `data/fun/laid.jsonl` as it stands. The `day_2026-10-31` folder holds seven cuts of that day;
    `2231Z` is the last.
  - **Copies:** the source `fun.mp4` is back in the input folder. The 24 fps working copy
    `input/mrpop/fun_0000_6394_24p_sound_in_step.mov` was rebuilt by its recorded recipe, byte-identical to its record
    (`data/fun/copies_rebuilt_2026-11-10/README.md`). `fun`'s old lead still, `mrpop/fun_main_head.png`, is not on the
    share.
  - **`fun2` is next**, in `data/fun_lead2/`: the same footage, only the lead replaced (`img_lead_2.jpg`, face only), with
    `second` and `third` composited as in `fun`'s final laid list. Everything dated 2026-11-01 Central or later
    (`>= 2026-11-01T05:00Z`) is `fun2`'s, including work run under the old conditions inside `fun`'s folders. Its rules:
    `data/fun_lead2/AGENTS.md`.
  - **The plan**: [`../masked_v2v/2026-11-10_simplification_plan.md`](../masked_v2v/2026-11-10_simplification_plan.md);
    its "Next, in order" says where each step stands.
  - **Done on 2026-11-10** (each in [`decisions.md`](decisions.md) under that date):
    - **Step 1:** the lane's records are in `data/lane.sqlite`, `fun` and `fun2` as two projects on one video
      (`bench/lane_record.py`; [`../masked_v2v/2026-11-10_lane_record_first_tier.md`](../masked_v2v/2026-11-10_lane_record_first_tier.md)).
    - **Step 2:** depth is the code's default in `bench/shot_to_load.py`.
    - **`fun2`'s set-up:** `data/fun_lead2/laid.jsonl` is re-seeded from the record, and five lead renders of
      2026-11-01 are the owner's keeps (`data/fun_lead2/owner_answers.jsonl`).
    - **The capture folders** that moved under `data/fun/` on 2026-11-09 have a link at each old name.
  - Worktrees `../../h3_worktrees/agy-laid-carve` (`91ad4237`, unmerged masked-lane `bench/laid.py` build) and
    `../../h3_worktrees/gem-perception-lhf` (`16b0805c`, unmerged masked-lane perception branch) remain on disk.

## What not to assume

- **Three pre-existing red checks on `main` (unrelated to `2026-11-09` work)**:
  - `bench/check_reference_encode.py` (upstream ComfyUI encoder snapshot drift), `bench/check_audio_freeze.py` (`song_allows_over_max_with_window` case), and `bench/check_two_shot_load.py` (`second_pass_from_cut` test fixture trips `no_mask_statement_has_mask`). Do not mask them in `bench/checks_baseline.json` without fixing or getting an explicit owner decision.
- **Generator validation on `:8188` when placeholder input files are absent**:
  - `workflows/build_workflows.py` validates against `:8188` by default (`--no-validate` skips it). On this machine, `H3_INPUT_DIR` currently lacks the placeholder reference media (`1-man.png`, etc.), so a plain `build_workflows.py` run fails validation with 89 missing-file errors unless `--no-validate` is passed or the placeholders are present.
- **J-Lens builder overwrites its source bundle, and pair bank / UI status**:
  - `bench/control_vector/j_lens_builder.py` (`build-j-lens`) recomputes only the refiner half, copies the lexical half out of the existing bundle on disk, and overwrites that file by default (§17.1). The 32 `.toml` files in `prompt_bank/control_vectors/` stay as a starter set and are not being grown (`decisions.md`); the browser UI (`bench/control_vectors.py ui`) is not maintained at present (`docs/wiki/control_vectors/studio_ui_guide.md`).
- **`data/fun/` is closed; nothing more is laid there** (2026-11-10). Its laid list is `fun`'s final state. The remake
  queue in `fun_render_backlog.md` §6 is closed with `fun`, not owed. The items this block used to hold are done or
  closed: the three-pass remake on `1419–1465` is laid (`second_hfp_1419g`, `lead_fp_1419b`, `third_fp_1419c`), and
  `lead_fp_6125c` is kept on `6125–6372`.
- **No database file goes in git** (the owner, 2026-11-10; `.gitignore` ignores `*.sqlite`, `*.db`, `*.duckdb` and their
  WAL, shm and journal files anywhere).

## Tomorrow, in order

1. **Open items from [`../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../research/control_vectors/2026-11-09_control_vector_simplification_plan.md) (§17, §19, §24, §25, §26, §27)**:
   - **Owner playback of the two vector-round stacked clips (§25)**: `Video/edit_guidance/stacked/diner16_s<seed>_stacked.mp4` (`data/results/2026-11-09_edit_guidance/vector_round/`), which decides the `mask` default (`scene_only`, `non_vision`, `all_rows`) and whether in-place vectors (`from-prompt`) are worth keeping alongside edit guidance.
   - **J-Lens lexical-half builder (§17.2, §20.1)**: Build the lexical half from the full 64-layer weights in the model store, move the word map into tracked code, and make `bench/control_vector/j_lens_builder.py` write to a new path instead of overwriting its source bundle.
   - **Apply-node reshape and CLI cut-down (§9, §15.2, §19, §24, §27)**: Reshape `MiniMaxH3ApplyControlVectors` and simplify `bench/control_vector/cli.py` + merge `build_cmd.py`/`make_cmds.py`, then revisit `comfyui_nodes_and_controls.md` and `preset_catalog_and_cli_reference.md` (§27).
   - **System prompt wiring & pre-existing red checks (§15.5, §26.1)**: Wire `bench/control_vector/llm_pairs.py::_extract_system_prompt_text` to `docs/wiki/system_prompts/h3_control_vector_standalone_system_prompt.md` (with `bench/check_prompt_docs_sync.py`), and fix or triage `check_reference_encode.py`, `check_audio_freeze.py`, and `check_two_shot_load.py`.
2. **Masked V2V: the simplification plan, in its order** ([`../masked_v2v/2026-11-10_simplification_plan.md`](../masked_v2v/2026-11-10_simplification_plan.md), "The order, and why"):
   - **Steps 1 and 2 are done** (2026-11-10), and so is `fun2`'s set-up. This item used to list them as ahead, with
     31 field rulings owed.
   - **Next:** a `fun2` cut from what already exists (the kept `second` and `third` pieces and the five kept lead
     renders), on the owner's go: the first end-to-end rebuild for `fun2`, and the first since the links went in.
   - **Then, the owner's two calls** before a new lead load: the 2026-10-25 context rule, which the code does not
     enforce, and where a new load's folders go.
   - **Then step 3's loads,** lead-alone shots first, each load's masks compared with those of `fun`'s kept lead piece
     on the same frames.
   - **Alongside:** the record's later tiers (`mrsql`).

## Where things are

- **Start-of-day reading order** ([`../../AGENTS.md`](../../AGENTS.md)):
  1. [`handover.md`](handover.md) (this file), [`next_steps.md`](next_steps.md), and [`decisions.md`](decisions.md) (`2026-11-09`).
  2. [`working_with_the_owner.md`](working_with_the_owner.md) and [`../comfy_notes.md`](../comfy_notes.md).
  3. For steering & edit guidance: [`edit_guidance.md`](edit_guidance.md), [`control_vectors.md`](control_vectors.md), and [`../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../research/control_vectors/2026-11-09_control_vector_simplification_plan.md).
  4. For masked V2V (`fun` closed, `fun2` in `data/fun_lead2/`): [`../masked_v2v/2026-11-10_simplification_plan.md`](../masked_v2v/2026-11-10_simplification_plan.md), [`../wiki/masked_v2v.md`](masked_v2v.md) ("The pipeline, end to end"), [`../../data/AGENTS.md`](../../data/AGENTS.md), [`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md), [`../masked_v2v/CODE_MAP_INDEX.md`](../masked_v2v/CODE_MAP_INDEX.md), and [`../masked_v2v/fun_render_backlog.md`](../masked_v2v/fun_render_backlog.md).
