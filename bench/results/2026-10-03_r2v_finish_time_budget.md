# Where a ref2va finish render's time goes, and what is left to take (2026-10-03)

The owner asked for an end-to-end look at render time on ref2va with a 2048
short-edge reference, without a quality sacrifice. This record holds the day's
measurements. The running notes, with each early claim and its correction, are
`internal/claude/2026-10-03_ditman_render_time_notes.md`; the live board is the
"H3 Lever Map" artifact.

## What ran

- **Graph:** `workflows/distill_experiments/h3_probe_r2v_step_switch_pdd8_flashgen_h080_savelat_api.json`
  (PDD8 for 6 evaluations, FlashGen for 2), market scene, 1344x768, 345 frames,
  one reference at 2048 short edge, 119,485 tokens, Sol at the 0.185.0 defaults.
- **Server:** started for this, armed with `H3_TELEMETRY` and `H3_SOL_TIME`
  only. Kitchen `0.2.37+sol.6b42fab.up.be003b7` (the build record beside the
  venv; the same carried commits as the 2026-10-01 build, on upstream v0.2.37).
  RTX 4090 at its stock power limit.
- **Renders:** one warmup at seed 730451891, then 730451892 and 730451893.
  Rows: `2026-10-03_r2v_finish_e2e.jsonl`. Per attention call:
  `2026-10-03_r2v_finish_sol_call_times.jsonl`. Telemetry raw records stay
  under the capture root (`2026-10-03_telemetry_r2v_e2e/`).
- **Host, for whoever compares later:** a kernel package was installed during
  the warmup render; a reboot and a BIOS memory-speed change were pending. The
  DIMMs ran below the base speed the modules report.

## The render

| stage | warmup | timed 1 | timed 2 |
|---|--:|--:|--:|
| total, s | 389.2 | 358.2 | 358.4 |
| sampler, s | 338.8 | 335.2 | 335.5 |
| of which PDD stage / finisher | 254.6 / 84.2 | 251.1 / 84.1 | 251.3 / 84.2 |
| decode, s | 17.6 | 17.7 | 17.7 |
| conditioning, s | 25.6 | cached | cached |
| mux and save, s | 3.6 | 3.8 | 3.8 |

Arming cost nothing visible: the timed sampler matches the unarmed 337 s of
`2026-10-02_sol_dense_blocks_panel.md` (market, `optCall`).

## One evaluation

About 41.9 s (335.3 s over 8; the first carries the DiT load).

| piece | s per evaluation | how |
|---|--:|---|
| Sol attention, 44 calls | 22.6 | live rows, median 512 ms per call |
| dense attention, 6 calls | 5.5 | live rows, median 914 ms per call |
| int8 linear layers | 8.6 | one block alone, times 50 (`2026-10-03_block_ops_r2v.json`) |
| norms, modulation, RoPE, gates | 1.7 | same file |
| LoRA branch at the call | about 3.3 | `2026-10-01_lora_branch_profile.md` scaled; not re-measured |

The pieces sum to the evaluation, so weight streaming, the allocator and the
stage switch are not hiding time. Telemetry agrees: the card is at full
utilisation through both samplers, the DiT is about 0.45 resident, and the two
stages cost the same per evaluation. Core's allocation compiler logs 12 breaks
and 24 rogues on the first PDD stage after a start and 4 and 0 on every later
stage.

By MAC count against the int8 issue rate in `docs/state/open_experiments.md` #17
(arithmetic, not a measurement): dense attention runs at about two thirds of
it, the linear layers at roughly 0.7 to 0.85.

**Corrected here:** an earlier reading in this session put the linear layers
at about 3 s per evaluation. That was the saving in
`2026-10-01_kitchen_merge_aade8d5.md`, misread as the total.

## Sol's exact stage against the dense kernel, same tensors

`bench/profile_sol_stages.py` on three cells of the 2026-09-27 two-reference
ref2va capture (120,582 tokens) with the capture's own spans
(`--audio-span 16616,17766 --video-start 17766`), then
`bench/time_dense_on_capture.py` on the same files. Records:
`2026-10-03_sol_stages_ref2va.json`, `2026-10-03_dense_on_capture_ref2va.json`.

| cell | Sol call, ms | exact stage, ms | routed share of all pairs | dense kernel, ms | exact ms per unit share / dense |
|---|--:|--:|--:|--:|--:|
| block 24, step 2 | 494 | 475 | 0.410 | 918 | 1.26 |
| block 40, step 2 | 506 | 485 | 0.415 | 919 | 1.27 |
| block 0, step 2 | 601 | 584 | 0.496 | 924 | 1.27 |

- The exact stage is 0.97 of a Sol call at the shipped settings; preprocess
  about 0.02; route and the V transpose under 0.01 together.
- Per attended pair, Sol's exact stage costs about 1.26 times what kitchen's
  dense kernel does on this card. At the dense kernel's cost a Sol call would
  be about a fifth shorter.
- The `model` field in the stage record says "captures from a base 16-step
  t2v render". That is a fixed string in the tool and wrong for these cells.

Read from source, none of it timed: Sol's exact kernel has no pipelined P·V
(the dense kernel has one for sm_89), reads each key's scale and bias from
global memory inside the tile loop, syncs threads twice per key block around
a two-stage gathered copy, and sets no launch bounds on this card. It does
hold Q in registers. The build uses fast math.

## Probe builds: where the exact stage's extra cost is

Branch `sol-exact-sm89` of the kitchen fork (local, commit `e96705d`), built
from a worktree to a side directory; the installed build was not touched.
Compile-time switches remove one thing at a time from the exact kernel only,
so routing is unchanged. One cell (block 24, step 2), whole Sol call, median
of 5; the other stages are about 16 ms of every row. Rows:
`2026-10-03_sol_exact_probe_builds.json`. A side build of the unmodified
branch matches the installed kernel in time and in a bit digest of the output.

| build | call, ms | output |
|---|--:|---|
| unmodified | 487 to 491 | reference digest |
| no P·V tensor-core issue | 401 | changed, by design |
| no QK tensor-core issue | 385 | changed |
| neither | 302 | changed |
| staging only, no compute | 286 | changed |
| compute only, no staging | 406 | changed |
| key scale and bias not loaded | 466 | changed |
| one thread sync per key block / none | 486 / 480 | changed (racy) |
| P·V accumulate pipelined | 488 | bit-identical |
| three-deep staging pipeline | 505 | bit-identical |
| key scale and bias staged into shared memory | 493 | bit-identical |
| launch bounds of 3 blocks per SM | 495 | bit-identical |

What it says:

- **The compute is not the gap.** With staging removed the call is 406 ms; the
  dense kernel's cost for the same share of pairs is about 377 ms (918 ms at
  0.41). Sol's arithmetic per pair is about as cheap as the dense kernel's.
- **The gap is the gathered copy.** Staging alone, with no compute, takes 286
  ms. The exact kernel copies a 16 KB K and V^T tile for 4 warps and 64 keys;
  the dense kernel copies 32 KB for 8 warps and 128 keys, a quarter of the
  bytes per attended pair. The tensor-core work adds almost fully on top of
  the staging time rather than hiding behind it.
- **The local fixes do nothing.** Pipelining the P·V accumulate, a deeper
  pipeline (it costs occupancy, as the source comment says), moving the key
  scales into shared memory (the register count rises and a block per SM is
  lost) and launch bounds are all within noise or worse. Thread syncs are
  about 2 percent.
- **What would help is staging fewer bytes per pair:** two adjacent query
  blocks in one launch block sharing a staged tile wherever both route it.
  They always do on the conditioning prefix and on the dense query rows,
  which by arithmetic from the capture's layout is over half of all tile
  visits. Not built. The per-row arithmetic would be unchanged, so the bit
  digest is the test.

## The paired kernel, built: bit-identical and slower on this card

The restructure the probe builds pointed to exists upstream for AMD cards
(Comfy-Org/comfy-kitchen #184, local branch `pr184` of the fork). Ported to
CUDA on branch `sol-exact-pair` (local, `e059ba1`), behind an environment
switch so one build runs both kernels. Rows:
`2026-10-03_sol_exact_pair_kernel.json`.

| kernel | block 24 | block 40 | block 0 | output |
|---|--:|--:|--:|---|
| single block (shipped) | 493 | 501 | 599 | reference digest |
| paired, union merge: a tile only one of the pair routes leaves the other half idle | 516 | 526 | 623 | bit-identical |
| paired, no idling: each half takes its own next tile, copied once when both want it | 733 | 741 | 866 | bit-identical |

(ms per Sol call, step 2.) Both variants are correct and neither is faster.

What the three kernels say together, as arithmetic from these timings and
the probe builds (not a separate measurement):

- A launch block of 8 warps holds a whole SM (the paired kernels compile to
  the register ceiling), against three blocks of 4 warps today.
- Copying a tile costs warp time in proportion to the copy instructions each
  thread issues: 256 threads sharing one tile finish an iteration in about
  half the time 128 threads take, and a half that stages its own tile with
  128 threads takes as long as today's kernel does with three blocks on the
  SM. So the copy is not hidden behind the tensor-core work, and it is not
  an L2 bandwidth limit either.
- Sharing therefore pays only on tiles every block in the launch block wants,
  and the rest must run no worse than today. Two blocks per SM cannot do
  that; three blocks' worth of rows in one launch block could, in principle,
  and its ceiling is the "compute only" row above: about a sixth off the
  call if every copy were free, about an eighth with the shared two thirds.
  Not built: it has to fit 384 threads in the register budget the single
  kernel uses per thread.

## P·V that could be skipped

`bench/probe_pv_skip_on_capture.py` on all 20 cells of the same capture, 56
heads, fp32 emulation of the kernel's tile walk. Record:
`2026-10-03_pv_skip_ref2va.json`. A warp skips a key tile when every one of
its 16 rows has the tile more than T bits under the row's maximum. The oracle
rule knows the final maximum, so these are upper bounds.

| block (step 2) | share skippable at T=12 | at T=29 (no bit changes in Sol's kernel) | mean error at T=12 | u8-P rounding error, for scale |
|---|--:|--:|--:|--:|
| 0 | 0.107 | 0.000 | 3.0e-4 | 6.6e-4 |
| 8 | 0.359 | 0.006 | 8.8e-4 | 1.8e-3 |
| 24 | 0.208 | 0.001 | 6.0e-4 | 2.5e-3 |
| 40 | 0.168 | 0.000 | 1.3e-3 | 3.0e-3 |
| 44 | 0.382 | 0.028 | 3.3e-3 | 2.3e-3 |
| 46 | 0.437 | 0.036 | 3.1e-3 | 2.2e-3 |
| 48 | 0.387 | 0.062 | 2.7e-3 | 2.0e-3 |
| 49 | 0.513 | 0.225 | 2.0e-3 | 1.1e-3 |

Bit-exact skipping exists only at block 49. A bounded skip is depth-dependent
and strongest in the tail. Step 6 reads a little higher than step 2. In the
kernel's own tile order the shares are lower (the `causal` rows in the file).

## What a tau per head would buy

`bench/measure_per_head_tau_on_capture.py` on ten cells of the same capture
(`2026-10-03_per_head_tau_ref2va.json`). One Sol call per tau in a sweep gives
every head's error against the dense kernel and its routed share, because a
head's routing does not read another head's; the best per-head assignment is
then priced against the single shipped tau. Fitted and scored on the same
cell, so an upper bound for that cell.

| cell | single tau 1.0: error, routed share | per head, same error: share | per head, same share: error |
|---|---|--:|--:|
| block 0, step 2 | 0.058, 0.496 | 0.391 | 0.033 |
| block 8, step 2 | 0.054, 0.428 | 0.414 | 0.048 |
| block 24, step 2 | 0.107, 0.410 | 0.384 | 0.090 |
| block 40, step 2 | 0.191, 0.415 | 0.387 | 0.165 |
| block 44, step 2 | 0.073, 0.429 | 0.373 | 0.050 |
| block 46, step 2 | 0.053, 0.418 | 0.369 | 0.040 |
| block 47, step 2 | 0.056, 0.417 | 0.367 | 0.041 |
| block 49, step 2 | 0.076, 0.416 | 0.346 | 0.059 |
| block 24, step 6 | 0.091, 0.412 | 0.392 | 0.085 |
| block 44, step 6 | 0.068, 0.434 | 0.363 | 0.048 |

- The share includes the conditioning floor, which no tau moves, so the
  routed remainder shrinks by more than these ratios suggest.
- **A block's table carries across steps.** Block 24's step-2 table on step 6
  gives error 0.091 at share 0.386 (single tau: 0.091 at 0.412); block 44's
  gives 0.061 at 0.377 (single tau: 0.068 at 0.434).
- **It does not carry across blocks.** Block 44's table on block 46 more than
  doubles the error. The table is per block and head.
- One scene. Yesterday's probe records say per-head error keeps its ranking
  across a new prompt and seed (rank correlation 0.96 at equal settings, 0.996
  across seeds only; `data/sparse/sol_analysis.duckdb`, `sol_probe_heads`),
  with every run on one reference geometry. A different reference is untested.
- The reference is the dense INT8 kernel, not exact attention, and local
  error did not predict the owner's verdicts on 2026-10-02.

## Reference rows across steps

Same capture, steps 2 and 6, reference rows' K and V over all 56 heads
(relative L2; computed inline, no file): zero at block 0; 0.14 and 0.18 at
block 8; 0.28 and 0.36 at block 24; 0.51 and 0.76 at block 40. Reference K/V
cannot be reused across steps without changing the result. One scene, steps
four apart.

## A second decoder

`bench/compare_vae_decoders.py --extra light_int8=...` on the timed render's
video latent (`2026-10-03_vae_decoders_light_r2v.json`), no `--fast` flags:

| decoder | steady decode, s | PSNR against fp16, dB |
|---|--:|--:|
| fp16 | 42.1 | reference (repeat is bit-identical) |
| shipped INT8 ConvRot | 17.3 | 56.7 |
| LynnReal light, INT8 ConvRot | 13.0 | 40.3 |
| taeh3 | 2.0 | 29.0 |

The light VAE is a distilled shorter decoder, not a quantization of the stock
one. Nobody has looked at its output here.

## What this does and does not show

- One scene, one canvas, one length, one reference. The token mix changes
  every share above.
- The linear-layer and elementwise figures are one block's ops with the card
  to itself. That they sum to the live evaluation is the check.
- Nothing here was judged by eye. The only output-changing item measured is
  the light decoder.
