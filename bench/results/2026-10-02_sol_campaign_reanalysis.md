# Sol dense_blocks campaign, re-read from the captures (2026-10-02)

A second reading of the eleven Sol probe and observer captures behind
0.184.7 to 0.184.12 (Options A, B and C for `dense_blocks`), using the raw
records rather than the campaign report. The report is
`docs/research/sparse/2026-10-02_sol_sparse_attention_hyperparameter_campaign.md`;
this record corrects it where they disagree.

**Source.** `data/sparse/captures/<run>/sol_{probe,observe}_*.jsonl` (gitignored),
one directory per test; `diagnostic_run` is Test 1, `test9c_allrows` is Test 9C.
**Reproduce.** `uv run python bench/sol_duckdb_analyzer.py` rebuilds
`data/sparse/sol_analysis.duckdb` and the dashboard; every table below is a
query on its views (named per table). Ad-hoc queries are given inline.
**Conditions.** All runs: the PDD8 + FlashGen finisher graph (sampler nodes 10
and 123), 1344x768, 345 frames, `quantizer=rotated`, kitchen INT8 dense
fallback, probe `trajectory=sol` (Sol's output is returned; the reference is
the chained fallback on identical q/k/v, not exact attention). Tests 1-4 were
written before the probe gained p99.9/p99.99 and per-segment worst heads
(`6620c5de`); the whole-call, per-head and per-segment metrics are computed
identically before and after.

## What the runs actually were (`sol_runs`)

| run | settings digest | tau | dense_blocks | start | prompt | seed |
|---|---|--:|---|--:|---|---|
| diagnostic_run (T1) | 8c4b242c6d14 | 1.0 | none | 0.0 | not recorded | not recorded |
| test2_standard | 7e5921fcceb5 | 1.0 | 45,48,49 | 0.0 | b08ea6b56ecb | 234683138190301 |
| test3_middle_dense | 6861b3492bd1 | 1.0 | 39,41,42,49 | 0.0 | b08ea6b56ecb | 409322557935755 |
| test4_start_point2 | 0fd5bbf5d260 | 1.0 | 39,41,42,49 | 0.2 | b08ea6b56ecb | 575039763063539 |
| test5_final | c9941be7ddaf | 1.1 | 39-42,49 | 0.0 | b08ea6b56ecb | 540684111820806 |
| test6_synthesis | ed793b83f5ef | 1.1 | 39-42,49 | 0.2 | b08ea6b56ecb | 497767775523101 |
| test7_multishot | ed793b83f5ef | 1.1 | 39-42,49 | 0.2 | fa12a58a8435 | 1108509668405916 |
| test8_tau13 | 43b7e413f186 | 1.3 | 39-42,49 | 0.2 | fa12a58a8435 | 582282439186259 |
| test9a_shield38 | b1ee25c218b1 | 1.3 | 38-42,49 | 0.2 | fa12a58a8435 | 654721216857353 |
| test9b_tau12 | b32d6f5a9a29 | 1.2 | 39-42,49 | 0.2 | fa12a58a8435 | 733999149447537 |
| test9c_allrows | 43b7e413f186 | 1.3 | 39-42,49 | 0.2 | fa12a58a8435 | 359680615083832 |

Every run has `sink_conditioning=exact_kv_and_rows`. Token routing is `off`
in T1 and `measured` (blocks 0, 24, 32, 40) in T2-T9C; the shipped default
(`workflows/h3_config.py::SOL_RECOMMENDED_CUDA`) is `off`. T1's render record
has no seed or prompt hash: it was written before `fdbcce6f` fixed the
`h3_config` import in `sol_observe._describe_prompt`.

Corrections to the campaign report that this table settles:

- **Test 9C did not run `exact_kv_and_all_rows`.** Its settings digest equals
  Test 8's, and the owner confirmed `exact_kv_and_rows` was used for every
  run. 9C is a same-prompt replicate of Test 8 at a new seed. The report's
  9C finding ("sinking conditioning queries does not resolve Block 38") has
  no run behind it.
- **The seed was never held fixed.** Every run has its own seed, including
  the Test 9 arms the report calls single-variable.
- **Tests 6 and 7 share a settings digest.** Test 7 differs from Test 6 only
  in prompt and seed, so it measures prompt-plus-seed variation, not
  multi-shot robustness.
- **T1 was two-stage**, like the rest (nodes 10 and 123), not single-stage.
- **The lowest sigma evaluated is 0.679**, not about 0.1. In T1 the
  per-step mean error falls monotonically with sigma (ad hoc: `SELECT node,
  step, avg(rel_l2) FROM sol_probe_cells WHERE run='diagnostic_run' GROUP BY
  ALL`).

## 1. The headline improvements are selection, not error reduction

`dense_blocks` and `start_percent` remove cells from what the probe measures.
The all-cell average and maximum therefore fall when the worst blocks or the
first steps go dense, whether or not any measured cell changed. The matched
population is the (sigma, block) pairs every run measured: sigma at or below
0.973, all blocks except 38-42, 45, 48 and 49 (`sol_run_summary`, relative
L2 in %; block-equivalents from `sol_run_cost`).

| run | cells | all avg | all max | matched avg | matched max | matched pooled | block-equiv |
|---|--:|--:|--:|--:|--:|--:|--:|
| diagnostic_run | 400 | 11.33 | 27.20 | 9.72 | 20.76 | 12.29 | 127.8 |
| test2_standard | 376 | 11.23 | 31.22 | 9.50 | 20.46 | 11.86 | 144.4 |
| test3_middle_dense | 368 | 10.19 | 23.26 | 9.54 | 18.05 | 11.82 | 149.6 |
| test4_start_point2 | 276 | 9.75 | 22.96 | 9.42 | 18.01 | 11.65 | 212.2 |
| test5_final | 360 | 10.43 | 21.87 | 9.78 | 18.29 | 12.09 | 149.1 |
| test6_synthesis | 270 | 9.94 | 19.74 | 9.78 | 19.47 | 12.26 | 212.0 |
| test7_multishot | 270 | 8.71 | 18.71 | 8.51 | 16.16 | 10.50 | 213.7 |
| test8_tau13 | 270 | 10.27 | 22.09 | 10.07 | 19.33 | 12.13 | 204.9 |
| test9a_shield38 | 264 | 9.76 | 17.73 | 9.78 | 17.73 | 11.70 | 209.7 |
| test9b_tau12 | 270 | 9.53 | 19.49 | 9.35 | 17.09 | 11.22 | 209.2 |
| test9c_allrows | 270 | 10.18 | 21.60 | 9.97 | 18.73 | 12.09 | 204.9 |

Replicate noise (`sol_replicates`, per-block mean on matched steps, run B
minus run A):

| pair | same prompt | mean abs diff (pts) | max abs diff (pts) |
|---|---|--:|--:|
| test8_tau13 / test9c_allrows | yes | 0.17 | 0.60 |
| test6_synthesis / test7_multishot | no | 1.23 | 3.37 |

Readings:

- **Option C against Option B at tau 1.3** (9A against the 8/9C pair): the
  matched averages sit inside the replicate pair's spread. The report's
  "record-low peak error" is Block 38 leaving the measured set.
- **`start_percent=0.2`** (T3 against T4): T3 restricted to the sigmas T4
  measures already averages most of the way to T4 (ad hoc: T3 over all cells
  against T3 with `sigma_key <= 0.973`). The rest is inside seed noise. The
  probe compares Sol and the fallback on identical inputs, so it cannot see
  the "error compounding" the report credits warmup with preventing. On this
  8-evaluation schedule it costs 62 block-equivalents per render (T3 against
  T4 in `sol_run_cost`).
- **Prompt and seed move error more than any config here.** The cross-prompt
  replicate's block-level spread is larger than every config contrast in the
  campaign.
- **tau is the one lever with a visible dose-response.** On prompt fa12a58a,
  matched average rises T7 (1.1) < T9B (1.2) < T8/T9C (1.3), each step larger
  than the same-prompt replicate gap, for a few block-equivalents per 0.1 of
  tau. One seed per arm, so this is the shape, not the size.
- **9A does not dominate 9B on cost.** Counting dense calls at full cost, the
  two are within a block-equivalent of each other; on matched cells 9B has the
  lower error.

## 2. Within one forward pass, a dense block lowers the error of later blocks

The probe's per-cell number is local, but its inputs are not: with
`trajectory=sol`, a block's q/k/v carry every earlier block's Sol error. Two
instances, both larger than the replicate gap:

- Block 40's mean error falls by several points once 39 is dense (T2 against
  T3/T4 in `sol_block_profile`).
- Blocks 43-44 fall by about a point in 9A against the 8/9C pair, already at
  the first sparse step (sigma 0.973). At that step, steps 0-1 were dense in
  all three runs, so only the seed and the 38 shield differ, and blocks before
  38 show only seed-sized differences there.

So the "ridge" is not a fixed property of blocks 39-43. Part of each block's
error is induced by sparse error upstream in the same pass, and a
one-block-at-a-time sweep cannot see it. A set has to be chosen as a set, which
is what `SOL_RECOMMENDED_CUDA`'s 2026-09-02 note already requires.

I also first read 9A's lower error on blocks 30-37 as a trajectory effect of
the 38 shield. The step breakdown does not support that: those blocks already
differ by a seed-sized amount at the first sparse step, before any
trajectory difference exists.

## 3. The tail is quiet in the PDD stage and not in the finisher

Per block, mean relative L2 on matched steps, PDD stage (node 10) against the
FlashGen finisher (node 123), averaged over every run that measured the block
(ad hoc on `sol_probe_cells`, grouped by block and node):

- No block from 0 to 44 rises in the finisher; most fall and a few hold
  level.
- **Blocks 45-48 are the only ones that rise**, by about a quarter to two
  fifths. Within the PDD stage their error is flat or falling across sigma,
  then it steps up at the node boundary in every run. That points to the
  finisher's weights (the FlashGen LoRA), not to sigma.
- **The step up grows with tau.** It is small in the four tau-1.0 runs and
  several times larger in every run at tau 1.1 or above (ad hoc: per run,
  mean of blocks 45-48 per sigma). The finisher's tail is where raising tau
  costs most.

The report's "tail debunked" holds for the PDD stage only. Blocks 45, 48 and
49 are also the three blocks whose `k_norm.weight` is lopsided
(`docs/attention/block49_quant_error.md` section 1), so the finisher may be loading
those channels harder. Not measured.

## 4. Block 49's and block 0's error sits in a few fixed heads; the ridge's does not

The share of a block's squared error carried by its worst heads, T1 (nothing
dense), matched steps (ad hoc on `sol_probe_heads`: per block, sum
`head_numerator` over cells, rank heads):

- **Block 49**: three heads carry about two thirds, eight heads about nine
  tenths. Making those eight exact would leave roughly a third of the block's
  relative error at a seventh of a dense block's cost (the eight heads' cost
  is arithmetic, not measured). The top four (11, 9, 47, 30) are the same at
  every step of both stages. Only T1 measured block 49 sparse, so this is one
  prompt.
- **Block 0**: the top five (23, 33, 52, 20, 9) are the same in all eleven
  runs, across both prompts, every tau, and routing on and off.
- **Blocks 38-43**: the error is spread over many heads. The top eight carry
  about a third to two fifths, and exact heads would not fix these blocks.

Heads 9 and 11 at block 49 are on the K-rounding worst list in
`docs/attention/block49_quant_error.md` (peaky heads, a handful of effective keys).
Under rotation the quantization term is small (`2026-09-27_sol_redesign_test2.md`),
so their remaining error is plausibly routing: a pooled block mean can miss a
single sharp key. Hypothesis, not measured.

## 5. Text and reference queries are the highest-error segments, and none of the runs protected them

Pooled per-segment relative L2, matched steps, blocks sparse everywhere: text
is highest, then ref_img and video close together, and audio is near zero
(ad hoc on `sol_probe_segments`). `exact_kv_and_rows` keeps only the
target-audio query rows exact (`sol_attn_h3.py`, the `sink_conditioning`
docstring). Text and reference queries ran sparse in every test, and audio's
near-zero error is that protection working. `exact_kv_and_all_rows` extends
exact rows to text and references, which is the arm the report describes
under 9C but never ran.

## 6. Token routing on block 40

T2's head 47 at block 40 does reach the report's 111% on one cell, but head 47
was already that block's worst head with routing off (T1), and block 40's
whole-call error is the same in T1 and T2. One head on one cell, across
different seeds. It is not evidence that routing fails on diffuse blocks.

## The owner's look at the campaign's renders

On 2026-10-02 the owner watched the campaign's own renders and found them
"all pretty good. i couldnt say when one was better or not, but they were all
solid". Not blind, one render per config, every render at its own seed, and
no dense render beside them: it says no config was visibly broken, not that
the configs match dense. The output-level panel that followed,
`2026-10-02_sol_dense_blocks_panel.md`, has a dense reference at matched
seeds, and against it the owner did see Sol's effect.

## What would settle the dense_blocks question

1. **Fix the seed and the prompt within a contrast**, and run two seeds per
   arm, so a difference can be set against a replicate.
2. **Run with the shipped routing (`off`).**
3. **Score on the output, not the local cell.** The final latent's distance
   from a dense render on the same seed carries upstream propagation, warmup
   and the finisher. The probe stays the tool for proposing candidates.
4. **Run the `exact_kv_and_all_rows` arm**, since text and reference rows
   carry the highest segment error and no run protected them.
5. **Price a per-head exact path for block 49** (and block 0), if the kernel
   can take a per-head mask. Not checked.
6. **Measure the finisher separately.** Its tail blocks behave differently
   from the PDD stage's, and FlashGen graphs ship.

A rendered blind pair is still the bar for a shipped default
(`docs/method/eval_comparison.md`). None of the Option A, B or C adoptions had one.
