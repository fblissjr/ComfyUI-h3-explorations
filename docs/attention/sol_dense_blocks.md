# Which blocks to keep off Sol: the evidence behind `dense_blocks`

Detail page of [`sol_attn.md`](sol_attn.md). The shipped list is `h3_config.SOL_DENSE_TAIL`, mirrored by
`sol_attn_h3.py::SOL_DENSE_TAIL` and held equal by `bench/check_attention_defaults.py`. The candidate lists are the
`SOL_DENSE_OPTION_*` constants beside it.

## What the knob does

A block named in `dense_blocks` does not run exact attention. The override hands the call to the dense kernel under
Sol, which on the shipped graphs is kitchen's rotated `int8_attention` (sage on the `FLOOR_STEMS` arms, and sage on
every graph before 2026-09-15). So the knob swaps Sol's approximation for the dense kernel's, and what it buys is the
difference between the two errors, not Sol's error. Negative indices count from the end (`block_spec.py`), so `-1` is
block 49.

**Exact attention at a block** is `MiniMaxH3ExactBlocks` (added 2026-08-29): the blocks it names run ComfyUI's own
attention with the override stripped, this box's unaccelerated answer rather than a reference implementation. Place
it after the dense node; either side of Sol works, because both of Sol's composition sites skip a forward carrying
`_uses_optimized_attention`, a branch on a private attribute that `bench/check_exact_blocks.py` asserts. Cost is
roughly 1.7x the sage time on the blocks it names. No shipped graph wires it; `h3_probe_t2v_exact_tail` and
`h3_probe_t2v_policy` are the arms that do ([`quant_policy.md`](quant_policy.md)).

## The decision, 2026-10-02

The shipped list is **Option C, `"38,39,40,41,42,49"`**: the middle ridge where routing error peaks under the rotated
quantizer, plus block 49, which feeds `final_layer.video_out`. It was adopted on local probe error and **kept by the
owner's decision** when the output-level panel, stopped after one judged seed, found that the `dense_blocks` choice
does not move the verdict, and C is the list `SOL_SINK_DEFAULT` was judged with
(`bench/results/2026-10-02_sol_dense_blocks_panel.md`). The re-read of the campaign that produced Options A to C
found the per-option differences inside prompt-and-seed noise on the cells every run measured
(`bench/results/2026-10-02_sol_campaign_reanalysis.md`). The campaign itself is
[`../research/sparse/2026-10-02_sol_sparse_attention_hyperparameter_campaign.md`](../research/sparse/2026-10-02_sol_sparse_attention_hyperparameter_campaign.md)
and its re-analysis [`docs/research/sparse/2026-10-02_sol_dense_blocks_reanalysis.md`](../research/sparse/2026-10-02_sol_dense_blocks_reanalysis.md).
The sequence of lists is in [`../state/decisions.md`](../state/decisions.md).

**What the rotated telemetry showed.** Full 50-block telemetry on a Ref2VA capture (S = 119k) under
`quantizer="rotated"`: rotation disperses the K-channel outliers that made 45, 48 and 49 the worst INT8 blocks, so
block 48's error fell to 7.14% (max 9.41%) and block 45's to 9.55%, while routing error peaks across blocks 38 to 43
at 19% to 26% (block 42 25.67%, block 39 25.22%). That retired the 2026-09-25 tail list `"45,48,49"`
(`SOL_DENSE_HISTORICAL_TAIL`), which came from the unrotated K-norm scan ([`block49_quant_error.md`](block49_quant_error.md)).

## What was measured before, and what it still says

All of the following is on unrotated INT8 and against a sage fallback; read it for the mechanism and the instruments,
not for today's list.

**Sol's error at a block is not the block to protect.** Sol's sparsity error against exact attention at tau 1.0, all
56 heads, S = 98,498, a ref2va capture
([`data/results/2026-08-19_sol_error_per_head_tau1.0.json`](../../data/results/2026-08-19_sol_error_per_head_tau1.0.json)),
steps 3 / 11:

| block | sparsity rel L2 | quant rel L2 |
|---|---|---|
| 0 | 0.089 / 0.093 | 0.024 / 0.032 |
| 8 | 0.100 / 0.127 | 0.024 / 0.031 |
| 16 | 0.123 / 0.146 | 0.032 / 0.039 |
| 24 | 0.136 / 0.161 | 0.031 / 0.036 |
| 32 | 0.143 / 0.176 | 0.041 / 0.046 |
| 40 | 0.196 / 0.224 | 0.049 / 0.047 |
| 49 | 0.165 / 0.143 | 0.135 / 0.135 |

Subtracting the dense kernel's own error at the same block gives what keeping it dense removes
(`bench/rank_dense_blocks.py`,
[`data/results/2026-08-29_dense_block_ranking.json`](../../data/results/2026-08-29_dense_block_ranking.json)):
block 40 removes the most (0.209), block 0 the least (0.093), and block 49 sits third because sage's own error there
is 6.6x block 0's, with `cos_min` negative (-0.04 to -0.11) on some rows
([`data/results/2026-08-18_sage_accuracy_on_capture.json`](../../data/results/2026-08-18_sage_accuracy_on_capture.json)).

**Propagation reverses the local ranking.** `bench/probe_block_propagation.py` runs Sol at exactly one block, the
dense kernel everywhere else, and reads the output latent against an all-dense baseline, which folds in how much of a
block's error survives to the output
([`data/results/2026-08-29_block_propagation.json`](../../data/results/2026-08-29_block_propagation.json)):

| block | video rel L2 | audio rel L2 |
|---|---|---|
| 0 | 0.0306 | 0.0412 |
| 32 | 0.0272 | 0.0471 |
| 1 | 0.0272 | 0.0436 |
| 2 | 0.0254 | 0.0405 |
| 24 | 0.0247 | 0.0352 |
| 8 | 0.0243 | 0.0366 |
| 16 | 0.0239 | 0.0378 |
| 40 | 0.0177 | 0.0297 |
| 49 | 0.0128 | 0.0255 |
| 48 | 0.0110 | 0.0232 |
| 45 | 0.0109 | 0.0253 |

Block 0 is the block Sol approximates best and the one whose error matters most at the output; 32 is a second peak,
replicated at a second seed; the last five blocks move the output least. Controlled by construction (two baselines
bit-identical across a restart; Sol's window holds only the last of four steps, so no trajectory diverges). It could
not see two things, both against the tail: its baseline ran sage at block 49 too, and it ran on the base model, so
PDD's fused output head was not in the path. It sampled 11 of 50 blocks at one sigma, which is why `"0-2,32"` shipped
from 2026-08-29 only until the owner demoted it to an explicit arm on 2026-09-02.

**Cost is uniform per block**: 1.01 s per dense block at 4 steps, linear from 2 to 8 blocks, against a 0.9 s noise
floor and a 150.4 s baseline (`bench/time_dense_blocks.py`,
[`data/results/2026-08-29_dense_block_cost.json`](../../data/results/2026-08-29_dense_block_cost.json)). The harness
reads the server's own execution span and refuses an arm whose sampler was cached; its first run did not, and read
19.51 s per block off a 3.0 s cached reference.

**Block 0 routes 29.8% of its key blocks exact against about 11% deeper** ([`sol_error.md`](sol_error.md)), so it is
already the block where Sol saves least.

## What does not transfer

- **The paper's dense policy.** It runs the first 20% of steps and the first layer dense on Wan2.1, HunyuanVideo,
  Bernini and Ideogram 4 (where `start_percent` 0.2 comes from), none of them H3. Cite it for the mechanism, never
  for which H3 blocks want dense.
- **Upstream H3 lists disagree with each other**: Sol-Engine's per-hardware cells keep the first two, Sol-H3 keeps two
  for T2V and none for Ref2VA, Spark's Ref2VA draft keeps layer 0, sglang's backend keeps 0 and 1, core's node keeps
  none ([`sol_upstream.md`](sol_upstream.md)). Copying one settles nothing.
- **`h3_config.SOL_ARTIFACT_INSURANCE`** (`tau=1.3, dense_blocks="33-35,39-42"`) is a Triton-era guess at the
  object-dissolve artifact, never wired and never probed.

## Open

An all-50-block, actual-schedule judgement at the output, on more than one seed and scene, is what would replace the
decision with a measurement. The route observer supplies cost, not sensitivity.
