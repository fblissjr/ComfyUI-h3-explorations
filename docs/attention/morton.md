# Morton token ordering in Sol-Attn: closed, not refuted

Detail page of [`sol_attn.md`](sol_attn.md).

**The answer.** Morton is off and gone. The owner closed the lane on 2026-09-27, and the reorder, the curves
(`sol_curves.py`), the reorder check and the ordering arms were deleted with the old Sol node
([`../research/2026-09-27_sol_node_redesign.md`](../research/2026-09-27_sol_node_redesign.md), section "Morton";
[`../state/roadmap.md`](../state/roadmap.md), "Closed lanes"). It is closed, not refuted: the `3d` order has lower Sol
error than raster at equal routed density on every captured cell, and that stands as a recorded property. Git has the
code; the records cite it by commit.

## Why it was closed

| reason | evidence |
|---|---|
| no H3 implementation anywhere reorders tokens spatially | core's only Morton code sorts meshes; kitchen has none; NVLabs' GB200 H3 profile lists per-call Morton as deliberately not done; LightX2V refuses it for H3. It ships only for Wan and Hunyuan ([`sol_upstream.md`](sol_upstream.md), "Morton in Sol-Engine") |
| its one measured advantage could only buy speed on Sol's share of a render, and was never tested as such | `bench/results/2026-09-17_sol_orderings.md` |
| the blind panel at the same tau found no defect removed | `bench/results/2026-09-18_sol_reorder_panel.md`: five valid pairs, plain order better on two, `3d` on one, two ties (the kitchen pair is struck) |
| it cost the node's most intrusive plumbing | process-wide `PackedLayout` and `_forward` patching |

The one visible win for the reorder was a stress scene: on 2026-09-17 the owner found every plain-order noodle bar arm
morphing around four seconds and the `3d` arms clean (`bench/results/2026-09-17_sol_options_noodlebar_batch.md`). That
prompt was written for 107 frames and rendered at 345, so it was withdrawn as evidence about normal use
(`bench/results/2026-09-18_off_length_prompts.md`). The decisions are in [`../state/decisions.md`](../state/decisions.md),
2026-09-17 to 2026-09-27.

## What Morton is, and how it can act at all

A token is one 32x32-pixel square of one latent frame; the model lists tokens frame by frame, row by row (raster
order). Sol cuts that list into 64-token blocks and gives each block one centroid, used both to decide whether the
block is exact and as the stand-in for all 64 rows when it is not. Under raster order a block is a thin strip across
the frame. Morton (Z-order) interleaves coordinate bits so that 64 consecutive tokens form a compact tile instead:
an 8x8 tile inside one frame (`2d_frame`), a roughly 4x4x4 brick across four latent frames (`3d`), or a Hilbert run
inside one frame (`hilbert`).

- **It can act only through block membership.** Attention is permutation-equivariant and the reorder was undone
  after the last block, so under dense attention it is neutral in exact arithmetic.
- **It is not bit-neutral.** Permuting the keys changes the summation order: PyTorch SDPA, bf16, T=4096, permute and
  invert gives 44% of elements differing, max abs diff 9.77e-04, cosine 0.9999964. Over 16 steps that perturbation is
  amplified, so a dense Morton render and a dense plain one at the same seed are different takes, and "it feels like a
  different seed" needs no block-membership effect.
- **Holding `tau` fixed does not hold sparsity fixed.** The routing threshold is
  `tau * sqrt(sum_d c_d^2 * kcvar_d)` with `kcvar` the variance across block centroids
  (`coderef/comfy-kitchen-kijai/comfy_kitchen/backends/cuda/sage_attention/sol_attn_preprocess.cu`), so a reorder moves
  the threshold as well as the membership, and the routing test (`colmean > thr`,
  `coderef/comfy-kitchen-kijai/comfy_kitchen/backends/eager/sol_attn.py`) moves with it. Routing and the tail are both
  centroid quantities (`coderef/comfy-kitchen-kijai/comfy_kitchen/backends/cuda/sage_attention/sol_attn_route.cu:18-19`).
  The sign of the density change is a joint function of depth and sigma: at block 49 `3d` routed 0.987x raster on the
  124-frame capture and 0.994x / 1.029x / 1.130x across steps 3 / 8 / 14 on the 362-frame reference capture. So no
  tau-matched ordering A/B is controlled, and no per-block `tau` can make one.
- **Block membership also depends on `video_start mod 64`**, because blocks are cut from absolute row 0. The reorder
  rotated its permutation by `(-video_start) % 64` to cancel that; all 64 offsets gave identical geometry, so
  reference load never changed block shape.

## What was measured

**At equal cost, on full-length captures (2026-09-17).** The comparison that the threshold finding made runnable:
sweep each ordering over tau 0.6 to 2.0 and compare error against routed density as curves, using the kernel's own
routed count and the finding that a Sol call's time follows routed density
(`bench/results/2026-09-17_sol_orderings.md`, `bench/sweep_sol_orderings_on_capture.py`; base 16-step t2v, 104,361
rows, first 8 heads). At raster's tau-1.0 operating point:

| cell | raster density / error | `3d` | `2d_frame` | `hilbert` |
|---|---|---|---|---|
| block 0, step 15 | 0.212 / 0.1249 | err -11.1%, dens -20.2% | err -10.4%, dens -20.8% | err -8.4%, dens -17.2% |
| block 24, step 15 | 0.215 / 0.0958 | err -27.1%, dens -30.4% | err +8.6%, dens +9.9% | err -6.1%, dens -6.5% |
| block 32, step 15 | 0.227 / 0.0922 | err -7.7%, dens -10.2% | err +9.2%, dens +10.5% | err +1.7%, dens +2.0% |
| block 40, step 15 | 0.202 / 0.2614 | err -19.4%, dens -32.5% | err -0.3%, dens -0.6% | err -7.8%, dens -11.9% |
| block 49, step 15 | 0.223 / 0.0377 | err -19.8%, dens -9.5% | err +105.6%, out of range | err +44.6%, dens +33.1% |
| block 32, step 4 | 0.222 / 0.0925 | err -2.9%, dens -4.4% | err -1.4%, dens -2.0% | err -3.3%, dens -4.5% |

`3d` is below raster on every cell; the per-frame curves are not improvements and lose badly at block 49, where
ordering within a frame separates tokens that attend across frames. The 2026-09-03 capture set ranks the orderings
the same way as a 2026-09-19 set, and the kitchen rebuild between them changed nothing
(`bench/results/2026-09-19_sol_orderings_old_vs_new.md`).

**Centroid fidelity on a 124-frame capture (2026-08-15).** Mean cosine of each key to its block's centroid, all 56
heads, 1344x768, one step, below the token floor:

| block | raster | `2d_frame` | `3d` | `hilbert` |
|---|---|---|---|---|
| 0 | 0.7523 | 0.7565 (+0.6%) | 0.7830 (+4.1%) | not run |
| 24 | 0.7442 | 0.7665 (+3.0%) | 0.7915 (+6.4%) | 0.7748 (+4.1%) |
| 49 | 0.8678 | 0.8804 (+1.5%) | 0.9434 (+8.7%) | 0.8978 (+3.5%) |

A compact block's centroid represents its members better. Spatial compactness does not rank orderings, though:
geometrically identical arms differed on activations, and the geometrically best arm gained least. Choose on captures,
never on block geometry. Keys needed for 90% of a query's mass were 394 of 591 blocks at block 0 and 178 at block 24
under raster: attention on this workload is not very sparse.

**Block geometry** (`bench/analyze_morton.py`, `bench/analyze_canvas_geometry.py`): `2d_frame` makes whole 8x8 tiles
only where width and height are both multiples of 256 pixels (768x768, 1024x768, 1280x768 and portraits), 3 of 48
landscape canvases; `3d` wants a height divisible by 128 and is connected on 27 of 48, but its floor is 51.5% connected
at 362 frames (1952x544) and it is worse than plain `hilbert` on 14 of 48. The per-canvas and per-length tables, the
`latent_t % 4` effect and the block maps are in [`../h3/geometry_cost.md`](../h3/geometry_cost.md), "Sol block geometry by canvas (the closed Morton lane)".

**Speed, rendered 2026-08-16.** Six arms at 1344x768, 294 frames, 16 steps, one run each:

| arm | sampler |
|---|---|
| sparse, Morton off | 454.0 s |
| sparse, `2d_frame` | 452.8 s |
| sparse, `3d` | 454.8 s |
| sparse, `hilbert` | 453.2 s |
| dense, `2d_frame` | 861.6 s |
| dense, Morton off | 860.8 s |

The permutation is free: the two control pairs disagree in sign (+0.8 s dense, -1.2 s sparse), both at or under the
bench's spread. Quote "free", never 0.8 s. Peak VRAM from this run is an allocator artifact (Morton "saved" 3.7 GB
sparse and cost 2.1 GB dense). The dense pair is also the 1.896x Sol-against-dense figure
([`sol_measured_arms.md`](sol_measured_arms.md)). The retracted Triton figure it replaced, Morton "worth 1.16x alone,
94% GPU utilisation", was Triton at 362 frames stacked on int8.

## What upstream said the payoff was

The only stated payoff anywhere is in the Triton pack's Wan file: Z-ordering "lets the same quality be reached at
higher sparsity" (`coderef/ComfyUI-SolAttn_triton/_morton.py:1-11`, a file that opens "for Wan"). The H3 variant
(`coderef/ComfyUI-SolAttn_triton/_morton_h3.py`) makes no quality or sparsity claim, the CUDA node never uses the word
"sparsity", and the paper contains no reordering at all. The author's own word on 2026-08-14 was that Morton "may or
may not increase quality, that's something to test". NVLabs ship Morton3D on by default for Wan and leave it out of H3,
saying the packed video tail "is already a contiguous grid-ordered block"; that is true of raster order too, which is
what Morton fixes on Wan, and our captures disagreed at the centroid level.

## The method lessons

- **The one quality finding ever recorded here was retracted within the hour** for vanishing at a second seed (commit
  `3b86b21` sent it upstream, `440eea9` retracted it, with the method postmortem in the commit message).
- **A mechanism written to explain an unestablished effect makes it feel confirmed.** Two were drafted here for a
  clip-start impression and cut.
- **Three arms agreeing is not a control.** The copied "reorder-only" control (Morton on, every block dense) isolated
  nothing until the fourth arm, dense with Morton off, was run; that arm also killed the VRAM "saving".
- **Canvases come from the legal set in [`../h3/geometry.md`](../h3/geometry.md).** A first sweep used four canvases H3 cannot
  render.

## If the lane is ever reopened

The comparison to run is quality at matched wall clock, not matched `tau`, on more than one seed, with the curve
chosen on captures; the 2026-09-17 sweep is the instrument for picking the operating point. The deleted code is at the
commits the redesign record names.
