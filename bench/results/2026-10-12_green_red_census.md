# Green and red: a census of what goes red in this repo, and on what kind of failure

2026-10-12, session mrfern (the afternoon's lead on the fun video). A reading, no render, no
check changed. It applies the owner's ruling (`AGENTS.md`, "Green or red only for a hard
failure"; `docs/state/decisions.md`, 2026-10-11 and 2026-10-12: the scope is everywhere) to
the 110 `bench/check_*.py` scripts, the capture tool's gate and the masked lane's other stops,
and says which reds are of a hard failure and which are of a degree. The cuts the ruling owes
are listed at the end for the owner's word; none is made here, since each changes what a
check means and two of them sit on open Open List rows.

## How it was read, and what it cannot see

Every check's own first line (`bench/INDEX.md`), then the comparison lines of the twenty whose
first line says "grade", "within", "exact" or a tolerance (grep for `atol`, `tol`, `FLOOR`,
`BAR`, `threshold` and a numeric comparison inside a case or an assert), then the comment above
each bar. Not read: every case of every check. A check whose first line says it asserts a
known answer and whose comparison lines all compare to a fixture built in the check is taken
as a hard-failure check on that reading alone. If a case inside such a check grades a degree,
this census missed it.

## The kinds

**A hard failure, by the ruling**: the thing under test differs from an answer that exists
independently of it (a fixture whose answer is known by construction; a vendor's own file;
Meta's own code; the stock forward; a committed baseline), or a file, node or link that is not
there. A float-noise allowance on such a comparison (`1e-5`, `1e-9`, bf16's few `1e-3`) does
not make it a matter of degree: the question is "is this the same thing", and the allowance
is the width of "the same" in floating point, reasoned in the comment above each.

**Of a degree**: a bar that is a level of quality, fit or closeness that a person set, where a
value on the wrong side is not plainly broken and might be wanted. The ruling says such a
figure is printed with its context and left to the runner.

## The checks

Hard failure, by first line and comparison (the great majority): every check of a tool against
a fixture made for the purpose (`assemble_delivery`, `capture_masked_run`, `cast_file`,
`contact_sheets`, `draw_lid_marks`'s geometry cases, `frame_sheet`, `masked_job`,
`page_answers_to_*`, `patch_render_window`, `pose_along_shot`, `read_render_against_original`,
`recomposite_window`, `region_against_plate`, `shot_coverage`, `shot_table`, `subject_parts`,
`subject_track(s)`, `video_mask`, `voice_spans`, `who_page`, `renders_page`, `clip_analysis`,
`clip_folder`, `break_checks`); every check of a graph, a constant, a widget, a model file, a
doc link, a skill route, a changelog, an index row, a vendor copy, a licence text or a baseline
(`attention_defaults`, `bench_matches_shipped`, `camera_vocabulary`, `capture_manifest*`,
`changelog`, `distill_settings`, `doc_inventory`, `doc_links`, `generator_constants`,
`graph_discovery`, `graph_values`, `literal_widgets`, `lora_alpha`, `marker_arms`,
`meta_sam3_copy`, `meta_sapiens2`, `model_contents`, `model_files`, `no_owner_paths`,
`node_ids`, `output_dir_resolution`, `prompt_*`, `ref_prompt_labels`, `release_qwen_grid`,
`retraction_consumers`, `schema_defaults`, `skill_routes`, `vendor_config`,
`widget_deviations`, `sol_kernel`, `vsa_core_patch`); and every equivalence of our code to a
reference implementation up to float noise (`body_pose` and `sam3d_body_vith` to Meta's,
`FORWARD_TOLERANCE = 1e-4`, with the comment that the two known breaks land far outside it;
`channel_balance`'s fold; `checkpoint_overlay`; `lora_branch`; `mutant_parity`;
`pdd_unmerged`, `1e-6`; `sol_node_equivalence` and `sol_observe` to the eager reference;
`frozen_video_cache`'s cached step to stock, `FLOOR = 2e-2` reasoned from bf16's mantissa;
`reload_invariance`, bit-identical; `audio_carry_inversion`'s inversion, `1e-5`;
`reference_encode`; `dit_prefix_attention`'s partitions; `subject_yaw`'s arithmetic;
`distill_grid`'s sigmas).

Of a degree, where a red can stand on a value that is not plainly broken:

| check or tool | the red | the bar and its provenance | what the cut would be |
|---|---|---|---|
| `check_correctness.py` | `FAIL` when the patched forward's error against the stock one is over `TOLERANCE = 0.15` | the comment above it says only that the failures it catches are shape-independent; `docs/method/checks.md` says this check "records rather than inventing a threshold", which the code does not do | print the error with the sequence length and the dtype, no red; or keep the red and say in the comment what measured 0.15 |
| `check_solattn_correctness.py` | a case is red when the kernel's agreement with the eager reference is under a bar (`got > bar`) | the bars are per case in the script; not read here | the same question: is each bar float noise (then a hard failure) or a quality level |
| `check_sol_chunked.py` | `cos > BAR` against the direct path; `agree > 0.9`; "the chunked path did not reduce the peak" | the cosine bar is an agreement level; the peak comparison is a design claim (chunking saves memory), not a break | print the cosine and both peaks; red only when the chunked output has the wrong shape |
| `check_channel_balance.py` | red when any block's energy share sits within 2x of the selection threshold ("selection is fragile") | a warning about the design's margin, raised as a failure | print the shares and which are near the line, no red |
| `check_keyframe_canvas.py` | `distortion > 1.01` must equal "the aspect ratios differ" | a 1 percent distortion line, reasoned | keep: it asserts a relation, not a level; the 1.01 is the width of "undistorted" |
| `check_dit_prefix_attention.py` | `front > 5 * back` for a planted signal | a planted signal in a synthetic capture: the answer is known by construction | keep: a hard failure of the measure on its own fixture |
| `bench/draw_lid_marks.py` (the tool, not its check) | prints `PASSED` / `FAILED` on two things: the video holds marks at all (hard), and "no neighbouring frame's points fit the marks better than the frame's own" (a fit comparison) | `fit_by_lag` at lags -1, 0, +1 | print the three fits and say which is best, no PASSED/FAILED on the second; the first stays, as the owner's own rule: the figures, not the tool's PASSED (its handoff note of 2026-10-12) |
| `bench/capture_masked_run.py preflight` | the gate: `--gate` exits `GATE_BLOCKED` when a top-level flag stands with no override; every flag's threshold (lines 261 to 300) is a level measured or reasoned on one clip | by the tool's own words each flag is "a prompt to look at those frames, never a refusal"; the block is the builder's, so that a look is recorded before a render | this is Open List row c22 (refuse, show, or drop the look): the owner's, unchanged by this census |
| `bench/assemble_delivery.py` | the delivery record's `passes` / `FAILS` when a named subject has nothing laid on a shot and nothing says why | a structural fact (no row, no reason line), not a degree | keep, as row c27's pick says; the word is a hard failure's |
| `bench/run_checks.py` | fails on a red that `checks_baseline.json` does not name; prints `not graded` for a check that cannot run here | the runner itself grades nothing | keep |

## What this changes today

Nothing. The three cuts that are plainly inside the ruling (`check_correctness`'s bar,
`check_sol_chunked`'s cosine and peak, `check_channel_balance`'s fragile-selection red, and
`draw_lid_marks`' second verdict) are listed for the owner's word on the Open List; each is a
small change with its check's docstring and `docs/method/checks.md` row updated in the same commit.
`draw_lid_marks` is in use on the card today (the second subject's cutaways), so its change
waits for a day it is not.
