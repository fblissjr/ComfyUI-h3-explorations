# Old captures against new: the 2026-09-03 set still ranks orderings the same way; the kitchen rebuild changed nothing

Date: 2026-09-19. Script: `bench/sweep_sol_orderings_on_capture.py` (last
changed `6eeaac3`), pack at `80155ec`, comfy-kitchen `0.2.35+sol.8176242`,
RTX 4090 alone, first 8 heads, the script's defaults (tau 0.6 to 2.0, grid
102 x 24 x 42, video from row 1545, audio rows 395 to 1545), which are the
conditions of `2026-09-17_sol_orderings.md`. Data:
`2026-09-19_sol_orderings_base16_rerun.json` (the old cells) and
`2026-09-19_sol_orderings_covered_market.json` (the new cells).

Why: the capture plan (`internal/2026-09-18_capture_plan.md`, set A) asked
whether the 2026-09-03 set still represents the current stack, before anyone
recycles it or keeps grading on it. Two things changed since it was taken, and
this separates them.

## The two sets

| | 2026-09-03 (`2026-09-03_base16_t2v_1344x768`) | 2026-09-19 (`2026-09-19_covered_market_sage_chain_depth` and `_steps`) |
|---|---|---|
| prompt, seed, length, checkpoint | covered market, the same prompt bytes (sha256 prefix `8a94a547e5e7`), seed 730451892, 345 frames, int8 convrot fl2va pruned, 16 steps | the same |
| attention chain | sage `auto` (then the dispatcher, plain fp8++) on every step, no Sol | sage `fp8++ balanced` on steps 0 to 3, Sol `qk_balance` tau 1.0 from step 4 |
| pack, kitchen, core | as of 2026-09-03 | pack `9ad93db`, kitchen `0.2.35+sol.8176242` |

So the new cells differ in TRAJECTORY (Sol was routing for the steps before
each captured one) as well as in code. A disagreement could come from either;
agreement covers both.

## Step 1: the kitchen rebuild, isolated

The six old cells swept again today on the current kitchen build against the
same six swept on 2026-09-17 on `0.2.34+sol.b532e28`: all 144 points (six cells,
four orderings, six taus), density and error, are identical at the stored
precision. The rebuild onto the 0.2.35 pin is neutral for Sol's error on these
cells. Re-derive: compare the `cells` of the two json files point by point.

## Step 2: old cells against new, same kernel

Read at raster's shipped point (tau 1.0). "err" is the ordering's error at
raster's density, "dens" the density it needs to reach raster's error, both
log-interpolated along its own curve; negative is better. Computed by
`bench/compare_sol_orderings.py`, which reproduces the 2026-09-17
record's published deltas to within a few tenths of a point (the interpolation
there is not recorded, so the method is re-implemented, not copied).

| cell | set | raster density / error | `3d` err, dens | `2d_frame` err, dens | `hilbert` err, dens |
|---|---|---|---|---|---|
| block 0, step 15 | old | 0.212 / 0.1249 | -11.2%, -20.4% | -10.5%, -20.9% | -8.5%, -17.3% |
| | new | 0.212 / 0.1241 | -11.1%, -20.6% | -10.3%, -20.1% | -8.6%, -17.4% |
| block 24, step 15 | old | 0.215 / 0.0958 | -27.2%, -30.7% | +8.5%, +9.5% | -6.3%, -6.9% |
| | new | 0.213 / 0.1042 | -32.0%, -36.3% | +3.5%, +4.0% | -9.0%, -10.1% |
| block 32, step 15 | old | 0.227 / 0.0922 | -7.9%, -10.4% | +9.1%, +10.1% | +1.5%, +1.9% |
| | new | 0.234 / 0.0922 | -7.7%, -10.3% | +14.5%, +15.7% | +5.8%, +6.7% |
| block 40, step 15 | old | 0.202 / 0.2614 | -19.5%, -33.0% | -0.5%, -0.8% | -8.0%, -12.2% |
| | new | 0.201 / 0.2423 | -12.1%, -19.0% | +6.0%, +8.4% | -0.8%, -1.1% |
| block 49, step 15 | old | 0.223 / 0.0377 | -20.2%, -10.4% | +104.9%, out of range | +43.9%, +32.7% |
| | new | 0.223 / 0.0410 | -24.6%, -13.2% | +115.0%, out of range | +48.6%, +35.9% |
| block 32, step 4 | old | 0.222 / 0.0925 | -3.0%, -4.6% | -1.5%, -2.1% | -3.3%, -4.7% |
| | new | 0.222 / 0.1006 | -2.6%, -4.0% | -2.0%, -2.9% | -2.2%, -3.3% |

## What it says

- **The conclusions of 2026-09-17 hold on the new set.** `3d` is below raster
  on every cell of both sets; `2d_frame` loses on blocks 24, 32 and 49 on both;
  `hilbert` loses on 32 and 49 on both. One sign changes: `2d_frame` on block 40
  goes from a marginal gain on the old set (-0.5%, inside the spread seen
  elsewhere) to a loss on the new one (+6.0%). It moves further from being a
  candidate, not closer.
- **Raster's routed density at tau 1.0 is stable** across the two sets, which
  is what the cost side of every Sol decision rests on.
- **Magnitudes move.** Raster's error at tau 1.0 is within a few percent on
  blocks 0 and 32 at step 15 and differs by up to about a tenth on blocks 24,
  40, 49 and on block 32 at step 4, in both directions. `3d`'s margin on block
  40 is smaller on the new set, and on block 24 larger. Nothing here says
  whether that is the Sol trajectory or two weeks of code; the kitchen half is
  ruled out by step 1.
- **So the old set is representative for rankings and signs, not for
  magnitudes to better than about a tenth.** A record that quotes an absolute
  error from the 2026-09-03 cells should say it is from a dense-sage
  trajectory. Whether to recycle it is the owner's call: it is still the common
  yardstick of every capture-graded record since 2026-09-14, and its
  `retention.json` keeps it to 2026-10-31.

Limits: eight heads, six cells, one scene. The new set is the sage chain, not
the default kitchen chain (the capture hook lives in the sage node). Error
against exact attention is not a verdict on a clip.
