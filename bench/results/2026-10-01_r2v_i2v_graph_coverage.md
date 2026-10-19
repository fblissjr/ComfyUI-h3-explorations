# Ref2va and i2v distill experiments: which graph made each, 2026-10-01

The owner asked whether the ref2va and i2v experiments with FlashGen and FastH3
(2026-09-24 to 2026-09-29) are kept as workflows, and whether the top approaches
have one even where the mutant repos (`fbjr/h3-mutant-distill` on Hugging Face,
`h3-mutant-distill` on GitHub) do not ship them.

**Result: every recipe rendered in that window has a graph in the repo.** Nothing
was missing, so no graph was added. What was missing was a map from clip to graph
to verdict, and a way to ask the question again; both are below.

## How it was derived

`bench/clip_recipe_coverage.py` reads the graph each clip was rendered from (the
first-frame PNG saved beside it), reduces it to a recipe, groups the clips, and
matches each group to the graphs under `h3_config.graph_paths(include_bench=True)`.
The recipe is the task, the checkpoints, each LoRA with its strength and blocks,
the sigma schedules, the sigma shift and VSA's keep percent. It leaves out the
seed, prompt, images, canvas and filenames, and it leaves out Sol-Attn: the
2026-09-26 probes carry no Sol node and the graphs today do, so the tool prints
`sol` or `nosol` beside each group rather than let that hide a match.

Run: `H3_OUTPUT_DIR=<share> python bench/clip_recipe_coverage.py --since 2026-09-24
--task r2v i2v --recursive --skip private h3_linkedin`. The full output is
`2026-10-01_r2v_i2v_graph_coverage.txt`. Skipped by name and not opened: one
folder whose name starts with `private`, and the clips of an unrelated side job.
The two clips the tool reports with no readable sidecar are codec comparison
images, not renders.

*Rerun 2026-10-01, the same day, after a review found two blind spots in the
tool: the PDD node's `steps` was not part of a recipe, so the PDD8 clips also
matched the `_pdd_4step` graphs, and `LoraLoaderModelOnly`'s strength was read
from the wrong input. The recipe now carries `pdd:N` and any PDD or branch
input moved off its inert value. Output:
`2026-10-01_r2v_i2v_graph_coverage_rerun.txt`. The conclusion is unchanged: the
same two parity-harness groups have no graph, and every PDD8 group now matches
only 8-step graphs. It also reads clips rendered after the first run (today's
parity arms) and the promoted path.*

## What the output says

- Every group of clips has a graph except two.
- Both are the parity harness's own arms
  (`bench/check_mutant_parity.py graphs`, 2026-09-28; `2026-09-28_mutant_parity.md`):
  the i2v and ref2va PDD8 graphs with PDD8's sigmas pinned in a `ManualSigmas`
  node. The pack graphs take them from the PDD node, so the recipe differs in that
  one field and the pack has the graph.
- The tool can report a miss: it did, on those two. A run that also read the
  skipped side job reported a third, a clip naming a checkpoint file this install
  no longer has.

## The map

Graph paths are under `workflows/`. "Clips" are filename prefixes under `Video/`
on the output share. The mutant column says which of the seven mutant example
workflows (`standalone/h3_mutant_distill/example_workflows/`) carries the recipe.

### First frame (i2v)

| approach | graph | clips | where it stands | mutants |
|---|---|---|---|---|
| PDD8 alone | `distill_experiments/h3_first_frame_to_video_pdd` | `first_frame_to_video_pdd_*` | the owner's i2v pick over the FlashGen finish, 2026-09-27 (`2026-09-27_finisher_grid.md`, "The owner's review") | `h3_i2v_pdd8` |
| FlashGen, 4 steps | `distill_experiments/h3_probe_i2v_flashgen_4step` | `h3_probe_i2v_flashgen_4step_*` | a first look at one seed, held up by eye (`2026-09-26_flashgen_tasks_s1.md`); no record sets it against PDD8 on i2v. The `_fp16` and `_int8` clips are the VAE comparison (`2026-09-26_int8_vae_tasks_s1.md`) | no |
| PDD8, then FlashGen from sigma 0.8 | `distill_experiments/h3_probe_i2v_step_switch_pdd8_flashgen_h080` | `h3_probe_i2v_step_switch_pdd8_flashgen_h080_savelat_*` | not for i2v: the finish brightens the whole frame at once, away from the input (owner, 2026-09-27) | no |

### Reference images (ref2va)

| approach | graph | clips | where it stands | mutants |
|---|---|---|---|---|
| PDD8 alone | `h3_image_ref_plus_text_to_video_pdd`, and `h3_ref2v_market_pdd` on the market scene | `image_ref_plus_text_to_video_pdd_*` | the ref2va PDD8 we run; never set against FlashGen on ref2va | `h3_r2v_pdd8` |
| FlashGen, 4 steps | `distill_experiments/h3_probe_r2v_flashgen_4step` | `h3_probe_r2v_flashgen_4step_*` | a first look: both references and the likeness held by eye in one render, an untrained transfer (`2026-09-26_flashgen_tasks_s1.md`) | `h3_r2v_flashgen` |
| PDD8, then FlashGen from sigma 0.8 | `distill_experiments/h3_probe_r2v_step_switch_pdd8_flashgen_h080` | `h3_probe_r2v_step_switch_pdd8_flashgen_h080_r2v_flashgen_finish_*` | the control arm of the 2026-09-29 blind session: the reference held on every arm (`2026-09-29_blind_sessions_read.md`). One scene, one reference, and not set against PDD8 alone | no |
| PDD8, then FastH3 at 12/3 | `distill_experiments/h3_probe_r2v_step_switch_pdd8_fasth3_s12` | `h3_probe_r2v_step_switch_pdd8_fasth3_s12_*` | the same as the FlashGen finish by eye, and the reference held, though FastH3 has never seen a reference token. Parked by the owner, 2026-09-29 | no |
| PDD8, then FastH3 at 10/3 | `distill_experiments/h3_probe_r2v_step_switch_pdd8_fasth3_s10` | `h3_probe_r2v_step_switch_pdd8_fasth3_s10_*` | grainy and compressed-looking by eye; kept as the arm that says why `s12` is the one | no |

Every graph above has a `_savelat` twin where the experiment saved latents. The
reference finishers run the market scene (one reference image and its graded
prompt from the prompt bank), because that is what the blind session rendered.

## Not built, and why

- **FastH3 alone on i2v or ref2va, and PDD8 then FastH3 on i2v.** Never rendered,
  so not an experiment we did. The FastH3 finisher lane is parked
  (`docs/state/decisions.md`, 2026-09-29), and the owner rejected the FlashGen
  finish on i2v.
- **The reference finishers with the generic two-reference prompt.** Never
  rendered. That prompt is the thin default that `bench/preflight_graph.py` warns
  about on `h3_image_ref_plus_text_to_video_pdd` and `h3_probe_r2v_flashgen_4step`;
  the market graph does not draw that warning.

## Owner's calls

- `h3_first_frame_to_video_pdd` is the owner's i2v pick and sits in
  `distill_experiments/` because its generator entry carries
  `distill_experiment=True`. The t2v finisher was promoted to `workflows/` on
  request (2026-09-27); this one was not. Promote it, or leave it.
  *Resolved 2026-10-01, the same day: the owner said to promote it. It is
  `workflows/h3_first_frame_to_video_pdd_api.json` from 0.179.2; the table above
  names the path it had when this was written.*
- Which of the unshipped rows should become mutant example workflows. The
  `mutant-packaging` board row already names the ref2va PDD8 against FlashGen
  blind pair as what would give those two rows a verdict.
  *Resolved 2026-10-01: the owner said ok. It was read as the unshipped rows with
  positive evidence, so `h3_i2v_flashgen` and `h3_r2v_pdd8_flashgen_finish` were added
  (0.180.0, `2026-10-01_mutant_parity_flashgen_i2v_r2v_finish.md`). The FastH3 finishers
  (parked) and the i2v FlashGen finish (rejected) were left out.*

## A prose error found on the way

`bench/preflight_graph.py`'s docstring said a bare `workflows/*_api.json` glob
"currently misses nothing" because `GRAPH_DIRS` was `("",)`. `GRAPH_DIRS` has
held `distill_experiments` since 2026-09-27, so that glob misses every graph in
the map above. Corrected in the same change, with the old claim logged in
`docs/state/decisions.md`.
