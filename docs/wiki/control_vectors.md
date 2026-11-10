# Control vectors for H3: start here

last updated: 2026-11-10

![Control vector pipeline overview](control_vectors/assets/control_vector_pipeline.svg)

Rewritten on 2026-11-09 from the code and from that day's measurements, and updated on 2026-11-10 for 3-lever section masking (`look_only`, `scene_only`, `description_only`, `soundscape_only`), `MiniMaxH3EditGuider.last_step`, and `film_16mm_1974` warmth restoration. What these pages used to claim and
no longer do is logged in [`decisions.md`](decisions.md). The working records behind this page are
[`../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../research/control_vectors/2026-11-09_control_vector_simplification_plan.md)
("the plan" below) and [`../research/control_vectors/2026-11-10_control_vector_state_history_and_tradeoffs.md`](../research/control_vectors/2026-11-10_control_vector_state_history_and_tradeoffs.md).

## What a control vector is

H3 reads a prompt as rows: the text encoder makes one row per token, a small refiner inside the DiT reworks
them, and the DiT attends to the result. A control vector is one direction in that refined space. Adding it
to a prompt's rows pushes the render toward a look or a quality without changing the prompt's words.

A vector is built from pairs of prompts that differ in one thing. Both sides are encoded, the rows of the
changed wording are averaged on each side, and the difference is the vector
(`control_vectors.py::mean_difference`). Nothing is trained.

## Three ways to steer, and which to use

| Way | What you do | What it needs | Where it stands |
|---|---|---|---|
| Edit guidance | Change a few words of your own prompt and turn a scale | Two nodes, no file | The owner passed it on 2026-11-09. Its page is [`edit_guidance.md`](edit_guidance.md) |
| A bank vector | Load a vector built ahead of time from the pair files in `prompt_bank/control_vectors/` | A built `.safetensors` file | Kept as a starter set and not grown (owner, 2026-11-09). Rendered on text-to-video and reference shots |
| An in-place vector | Build a vector from the prompt you are steering, with its reference picture when it has one | One build, a few seconds on the card | Ran for the first time with a picture on 2026-11-09. Not yet judged against the other two |

Start with edit guidance when steering a single shot's wording, and set `last_step` (for example `2` or `3`
on 6-step Pika) to hand late refinement steps back to 1-pass `[edited]`. Reach for a control vector when you
want zero extra DiT passes (1 pass per step) or one reusable dial across many prompts without editing each:

| Dimension | Control vector (`MiniMaxH3ApplyControlVectors`) | Edit guidance (`MiniMaxH3EditGuider`) |
|---|---|---|
| Sampling cost (6-step Pika) | **6 DiT passes** (1× per step; zero sampling overhead) | **12 DiT passes** at `0..1000`, or **8–9 passes** (`-25%` to `-33%`) with `last_step = 2` or `3` |
| Setup cost | Zero if loading a `.safetensors` bank/saved vector; ~7–8 s once when building with `from-prompt` | Zero offline build; two prompt encodings per run |
| Section scoping | Built-in **3-lever span masks** (`look_only`, `scene_only`, `description_only`, `soundscape_only`) | Scoped by which prompt section (`summary:`, `soundscape:`, `music:`, or description) `MiniMaxH3PromptEdit` edits |
| Multi-dial composition | Stackable (`MiniMaxH3ApplyControlVectors` chains in `float64` + orthogonal `hold` mode) | Single baseline-vs-edited prompt pair per guider |

One round on one reference shot compared all three on two seeds (the plan, section 25). It has not been
judged by the owner, so no way is ranked here.

## What is known, and what is not

Measured on 2026-11-09, each with where to read it in the plan:

- The text space is close to one across the families. The fl2va and ref2va text paths give nearly parallel
  rows for the same prompt, with or without the Pika LoRA, and the image model's text path is fl2va's
  (sections 23.1 and 23.3). Vectors are still built and named per family; dialing `MiniMaxH3LoRABranch.strength`
  away from `1.0` on a recognized same-family H3 LoRA is allowed with an advisory log note.
- A reference picture in front of a prompt moves the description and soundscape rows little, and the
  subject, summary and retention rows more (section 23.5).
- Every pair file is a text-only prompt with no picture (section 23).
- The bank's vectors share a common component (section 20). Nothing is subtracted for it.

Not known:

- Whether a vector built in place beats one from the bank on a render.
- Which visual rows a vector should reach by default across every task. Since 2026-11-10, `scene_only`
  reaches visual prose only (`look_only | description_only`) and leaves `overall_soundscape:` and
  `non_diegetic_music:` untouched; `look_only`, `description_only`, and `soundscape_only` isolate individual
  levers (`comfyui_nodes_and_controls.md`).
- How each DiT reads the same vector. The measurements above are of rows, not of pictures.

## The pieces

| Piece | What it is | Its page |
|---|---|---|
| Pair files and builds | Where pairs come from, pooling, holds, building with a reference picture | [`building_pairs_and_vectors.md`](control_vectors/building_pairs_and_vectors.md) |
| The nodes | The node that applies a vector, the two that build one, and their defaults | [`comfyui_nodes_and_controls.md`](control_vectors/comfyui_nodes_and_controls.md) |
| The command line | The commands, the file naming and the pair files in the bank | [`preset_catalog_and_cli_reference.md`](control_vectors/preset_catalog_and_cli_reference.md) |
| Worked examples | The few that a session has run, with their commands | [`examples_and_recipes.md`](control_vectors/examples_and_recipes.md) |
| J-Lens | The bundle that names a vector in words and builds one from words | [`j_lens_diagnostics_and_word_lab.md`](control_vectors/j_lens_diagnostics_and_word_lab.md) |
| Asking an LLM for pairs | The prompt to hand it, and where that stands | [`llm_prompting_guide.md`](control_vectors/llm_prompting_guide.md) |
| The browser UI | Not maintained at present | [`studio_ui_guide.md`](control_vectors/studio_ui_guide.md) |

## The graphs

Generated by `workflows/build_workflows.py`; `workflows/INDEX.md` lists each one's chain.

- A bank vector applied: `workflows/h3_text_to_video_pika_control_vector_api.json`,
  `workflows/h3_text_to_video_pika_dual_control_vectors_api.json`,
  `workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json`,
  `workflows/image/h3_p_h3_image_t2i_control_vector_api.json` and
  `workflows/image/h3_p_h3_image_edit_control_vector_api.json`.
- Edit guidance: the two graphs on [`edit_guidance.md`](edit_guidance.md).
- No shipped graph builds a vector. A vector is built from the command line, or by wiring
  `MiniMaxH3MakeControlVector` by hand.

## Defaults, and where they live

The apply node's defaults are `h3_rules.CONTROL_VECTOR_DEFAULTS` and
`h3_rules.CONTROL_VECTOR_OPTIONAL_DEFAULTS`, one copy read by the node and by the generator. The node page
says what each is for. Two things to know before changing a strength:

- The default mode rescales every vector to one length before it multiplies by strength
  (`control_vectors.NORM_MATCHED_TARGET`), so a strength means the same push for any vector.
- A build prints a recommended range for that mode and a separate one for mode `add`
  (`control_vectors.strength_ranges`).

## Code and records

- The math and the file format: `control_vectors.py`.
- The nodes and the run-time wrapper: `control_vector_nodes.py`.
- Which rows are which: `h3_spans.py`.
- The edit grammar shared with edit guidance: `h3_prompt_edit.py`, and the pair compiler
  `h3_pair_compiler.py`.
- J-Lens: `j_lens.py` and `bench/control_vector/j_lens_builder.py`.
- The command line: `bench/control_vectors.py` and the package `bench/control_vector/`.
- Checks: `bench/check_control_vectors.py`, `bench/check_edit_guidance.py`, `bench/check_config_names.py`.
- Tools for study: `bench/control_vector_rows.py` (a cache of text rows), `bench/control_vector_pooling.py`,
  `bench/control_vector_lexicon.py`, `bench/edit_guidance_arms.py` (comparison arms).
- Records: the plan above;
  [`../research/control_vectors/2026-11-04_h3_control_vectors.md`](../research/control_vectors/2026-11-04_h3_control_vectors.md);
  [`../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md`](../research/control_vectors/2026-11-06_j_lens_and_refiner_jacobian.md);
  [`../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md);
  [`../research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md`](../research/control_vectors/exl3_qwen3vl/h3_control_vector_report.md).
  Dated records say what was true when written.
