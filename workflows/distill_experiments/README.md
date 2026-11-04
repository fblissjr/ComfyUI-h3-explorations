# distill_experiments: the distill research graphs

last updated: 2026-11-04 (the FlashGen and FastH3 graphs retired); 2026-10-05 (the PDMD graphs retired); 2026-10-01 (the PDMD graphs; the first-frame and reference map; the i2v PDD8 pick moved to the root)

Generated like every graph here by `workflows/build_workflows.py`; never
hand-edit the JSON. Moved here 2026-09-27 at the owner's request. The routing
rule is `build_workflows._is_distill_experiment`:
- every `_savelat` (saves its latents) and `_x0` (saves each step's x0
  prediction) twin;
- every `h3_probe_*` graph that runs a distill (PDD, Pika H3 Flex, a step
  switch, an audio refine pass);
- the entries marked `distill_experiment=True`.

The everyday distill graphs stay at `workflows/`: `h3_text_to_video_pdd`,
`h3_text_to_video_pdd_manual_sigmas` (PDD6), `h3_first_frame_to_video_pdd`
(the owner's i2v pick, moved to the root on 2026-10-01), and the PDD ref and
first/last-frame graphs.

**The FlashGen and FastH3 graphs were retired on 2026-11-04**, with every
step switch that had either as a pass (`docs/roadmap.md`, "Closed lanes"; the
list is in `docs/wiki/decisions.md`, 2026-11-04). The owner's 2026-09-27
review of them, and what each held, is in
[`bench/results/2026-09-27_evening_takeaways.md`](../../bench/results/2026-09-27_evening_takeaways.md)
and the render dataset's findings; the graphs are in git before that commit.

`h3_decode_saved_latent_api.json` decodes any `_savelat` twin's latents
again without sampling.

## First frame and reference images: what each graph is

The evidence for what was rendered on i2v or ref2va from 2026-09-24 to
2026-09-29 is
[`bench/results/2026-10-01_r2v_i2v_graph_coverage.md`](../../bench/results/2026-10-01_r2v_i2v_graph_coverage.md),
which also lists what was never built. "Mutants" says whether h3-mutant-distill
ships the recipe. The graphs are API format: drag the file onto ComfyUI.

| task | approach | graph | where it stands | mutants |
|---|---|---|---|---|
| i2v | PDD8 alone | `../h3_first_frame_to_video_pdd_api.json` | the owner's i2v pick | yes |
| ref2va | PDD8 alone | `../h3_image_ref_plus_text_to_video_pdd_api.json`, `../h3_ref2v_market_pdd_api.json` | the ref2va PDD8 we run | yes |

To find which graph made a clip, or whether a new experiment has one at all, run
`bench/clip_recipe_coverage.py` over the renders (its docstring has the command).
It reports a recipe with no graph instead of passing over it.
