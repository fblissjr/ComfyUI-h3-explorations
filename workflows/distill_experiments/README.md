# distill_experiments: the distill research graphs

last updated: 2026-11-04 (the FlashGen and FastH3 graphs retired, then every PDD probe and twin with PDD's primary use); 2026-10-05 (the PDMD graphs retired); 2026-10-01 (the PDMD graphs; the first-frame and reference map; the i2v PDD8 pick moved to the root)

Generated like every graph here by `workflows/build_workflows.py`; never
hand-edit the JSON. Moved here 2026-09-27 at the owner's request. The routing
rule is `build_workflows._is_distill_experiment`:
- every `_savelat` (saves its latents) and `_x0` (saves each step's x0
  prediction) twin;
- every `h3_probe_*` graph that runs a distill (PDD, Pika H3 Flex or an
  audio refine pass);
- the entries marked `distill_experiment=True`.

The five PDD graphs kept stay at `workflows/`: `h3_text_to_video_pdd`,
`h3_first_frame_to_video_pdd` (the owner's i2v pick, moved to the root on
2026-10-01), `h3_first_last_frame_to_video_pdd`,
`h3_image_ref_plus_text_to_video_pdd` and `h3_ref2v_market_pdd`.

What is left here: the four Pika H3 Flex showcase probes
(`h3_probe_r2v_pika_motion_transfer_6step` and
`h3_probe_t2v_pika_balcony_orbit_6step`, each with its `_merge` twin), the
base-model instrumentation `h3_text_to_video_savelat`,
`h3_probe_t2v_base_euler32_savelat` and `h3_probe_t2v_base_euler32_x0`, and
the decode graph below.

**The FlashGen and FastH3 graphs were retired on 2026-11-04**, with every
step switch that had either as a pass (`docs/wiki/ops/roadmap.md`, "Closed lanes"; the
list is in `docs/wiki/decisions.md`, 2026-11-04). The owner's 2026-09-27
review of them, and what each held, is in
[`bench/results/2026-09-27_evening_takeaways.md`](../../bench/results/2026-09-27_evening_takeaways.md)
and the render dataset's findings; the graphs are in git before that commit.

**Every PDD probe and twin here was retired on 2026-11-04**, when the owner
retired PDD from primary use and kept five core graphs (`docs/wiki/ops/roadmap.md`,
"Closed lanes"; the list is in `docs/wiki/decisions.md`, 2026-11-04): the PDD
`_savelat` and `_x0` twins, the 16- and 32-step schedule tests, the audio
refine probe, the merge, sage, dense and baked ladder rungs, the PDD8-then-base
step switches and the four `h3_probe_ref2v_pdd*` reference arms. They are in
git at `358e6199`.

`h3_decode_saved_latent_api.json` decodes any `_savelat` twin's latents
again without sampling. Its placeholder latent name still reads
`text_to_video_flashgen_savelat`, a retired twin's; set it to the latent to
decode.

## First frame and reference images: what each graph is

The evidence for what was rendered on i2v or ref2va from 2026-09-24 to
2026-09-29 is
[`bench/results/2026-10-01_r2v_i2v_graph_coverage.md`](../../bench/results/2026-10-01_r2v_i2v_graph_coverage.md),
which also lists what was never built. "Mutants" said whether h3-mutant-distill
shipped the recipe; that package left this tree on 2026-11-04 (its release is
untouched). The graphs are API format: drag the file onto ComfyUI.

| task | approach | graph | where it stands | mutants |
|---|---|---|---|---|
| i2v | PDD8 alone | `../h3_first_frame_to_video_pdd_api.json` | the owner's i2v pick; kept when PDD was retired from primary use | yes |
| ref2va | PDD8 alone | `../h3_image_ref_plus_text_to_video_pdd_api.json`, `../h3_ref2v_market_pdd_api.json` | kept when PDD was retired from primary use | yes |

To find which graph made a clip, or whether a new experiment has one at all, run
`bench/clip_recipe_coverage.py` over the renders (its docstring has the command).
It reports a recipe with no graph instead of passing over it.
