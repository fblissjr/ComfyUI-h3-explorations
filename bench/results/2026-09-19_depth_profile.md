# Depth profile: which blocks Sol hurts most, on two scenes, and what the dense kernels do by block

Date: 2026-09-19. Captures: `2026-09-19_covered_market_sage_chain_depth` and
`_steps` (covered market, 345 frames) and `2026-09-19_noodle_bar_sage_chain_107f`
(noodle bar at its declared 107 frames); both seed 730451892, sage `fp8++
balanced` + Sol `qk_balance` tau 1.0, plain order, pack `9ad93db`; inventories
`2026-09-19_capture_inventory_*.json`. Analysis at pack `f684211`, comfy-kitchen
`0.2.35+sol.8176242`, RTX 4090 alone, first 8 heads of 56.

- Sol: `bench/sweep_sol_orderings_on_capture.py` on every step-8 and step-15
  cell of both scenes (the noodle bar with `--grid 32,24,42 --video-start 660
  --audio-span 304,660`, read from its records' segments), read at raster order
  and tau 1.0. Data: `2026-09-19_sol_orderings_depth_covered_market.json`,
  `2026-09-19_sol_orderings_depth_noodle_bar_107f.json`.
- Dense kernels: `bench/grade_dense_kernels_on_captures.py --heads 8` on all
  three sets: kitchen INT8 (the default chain's dense kernel), plain sage fp8++
  (NOT the balanced or rotated modes), bf16 SDPA (the floor), each against fp32
  attention on the same inputs. Data: `2026-09-19_dense_kernels_*.json`.
- Tables and correlations: `bench/depth_profile_tables.py` (the command is in
  its docstring), output `2026-09-19_depth_profile_tables.json`. Every number
  below is in that json.

Why: the owner wants `dense_blocks` decided from data, and blocks 8, 16, 44, 46
and 47 had never been measured. `internal/2026-09-19_question_review.md`
section D asks for a one-number ranking and a check that it holds across
scenes before anything is built on it; B covers the same twelve blocks as the
covered market's depth run, which makes that check possible.

## Sol's error by block (raster, tau 1.0; rank 1 = worst of the twelve)

| block | market s8 | noodle s8 | market s15 | noodle s15 |
|---|---|---|---|---|
| 0 | 0.0934 (5) | 0.0947 (6) | 0.1241 (2) | 0.1437 (2) |
| 8 | 0.0615 (11) | 0.0942 (7) | 0.0560 (10) | 0.0738 (9) |
| 16 | 0.0707 (9) | 0.0934 (8) | 0.0638 (8) | 0.0771 (8) |
| 24 | 0.1276 (2) | 0.1964 (2) | 0.1042 (4) | 0.1401 (3) |
| 32 | 0.0942 (4) | 0.1241 (3) | 0.0922 (5) | 0.0955 (6) |
| 40 | 0.1775 (1) | 0.2118 (1) | 0.2423 (1) | 0.2432 (1) |
| 44 | 0.1040 (3) | 0.1089 (4) | 0.1129 (3) | 0.1125 (4) |
| 45 | 0.0740 (8) | 0.0931 (9) | 0.0894 (6) | 0.0957 (5) |
| 46 | 0.0790 (7) | 0.1021 (5) | 0.0554 (11) | 0.0839 (7) |
| 47 | 0.0618 (10) | 0.0622 (11) | 0.0709 (7) | 0.0500 (11) |
| 48 | 0.0814 (6) | 0.0789 (10) | 0.0635 (9) | 0.0603 (10) |
| 49 | 0.0505 (12) | 0.0540 (12) | 0.0410 (12) | 0.0372 (12) |

Routed density at tau 1.0 is between 0.20 and 0.25 on every cell (the json has
each). Spearman rank correlation of the per-block error between the two
scenes: +0.85 at step 8 and +0.87 at step 15, over the twelve blocks.

## The dense kernels by block

Kitchen INT8 is below plain sage fp8++ on every cell of both scenes at every
captured step (1, 3, 4, 8, 12, 15). Both rank blocks almost identically on the
two scenes (Spearman between +0.89 and +1.00 at every step). Plain sage fp8++
is worst on block 49 on every cell, at about three times kitchen's error there;
kitchen INT8 is worst on block 49 on every cell but one (the market at step 1,
where block 45 is above it). bf16
SDPA sits flat at the floor on every block, so its correlation is noise.

| block | kitchen s8 market / noodle | sage fp8++ s8 market / noodle | kitchen s15 market / noodle | sage fp8++ s15 market / noodle |
|---|---|---|---|---|
| 0 | 0.0027 / 0.0029 | 0.0058 / 0.0062 | 0.0021 / 0.0022 | 0.0036 / 0.0038 |
| 24 | 0.0085 / 0.0085 | 0.0122 / 0.0120 | 0.0092 / 0.0088 | 0.0143 / 0.0137 |
| 40 | 0.0105 / 0.0118 | 0.0135 / 0.0146 | 0.0141 / 0.0158 | 0.0193 / 0.0217 |
| 44 | 0.0139 / 0.0135 | 0.0199 / 0.0196 | 0.0147 / 0.0138 | 0.0214 / 0.0214 |
| 49 | 0.0166 / 0.0200 | 0.0502 / 0.0583 | 0.0165 / 0.0178 | 0.0470 / 0.0517 |

(The other blocks and steps 1, 3, 4 and 12 are in the json.)

## What it says

1. **The block ranking is a property of the model, not of the scene**, to the
   extent two scenes can say so. Two very different prompts, one at a third of
   the other's length, rank the twelve blocks nearly the same way for Sol and
   almost identically for the dense kernels. That is the precondition section
   D of the review set for using any ranking at all, and it is met.
2. **Block 40 is Sol's worst block on both scenes at both steps**; blocks 0,
   24 and 44 follow. These are the `dense_blocks` candidates the data
   nominates.
3. **Block 49 is Sol's BEST block of the twelve, on both scenes at both
   steps**, with `qk_balance` on. It is the dense kernels' worst block, which
   is the block-49 INT8 story (`docs/h3_block49_quant_error.md`), and it is
   what a dense-tail practice ("keep 47 to 49 dense") is aimed at. On these
   captures that practice sends to the dense kernel the block where Sol does
   the least damage and where the dense kernel does the most. Blocks 47 and 48
   sit in Sol's lower half in seven of their eight rankings.
4. **Inside Sol's window the routing error dominates the quantization error.**
   Sol's error is between about 2 and 65 times kitchen INT8's on the same cell
   (smallest on block 49, largest on block 0 at step 15). Choosing between the
   dense kernels matters for the steps before `start_percent` and for any block
   made dense; inside the window, what Sol routes is the larger term.
5. **The price of one dense block.** On the default chain a DiT call costs a
   median of 719.6 ms on the kitchen dense fallback against 219.5 ms on Sol
   (`2026-09-17_sol_live_call_times.jsonl`, 345 frames, 0.2.34 build). At
   `start_percent` 0.2 and 16 steps, Sol has twelve steps, so each block moved
   into `dense_blocks` costs about 6 s of a full-length render (arithmetic from
   that record, not measured as a render).

## What it does not say

- **What "Sol's error" is here, exactly.** Capture-side output error: the
  relative L2 of the real CUDA kernel's output against fp32 dense attention,
  over every row of the first eight heads, on captured inputs, at tau 1.0 with
  the shipped recipe as the sweep reads it from
  `h3_config.SOL_RECOMMENDED_CUDA` (`qk_balance` ON, rotation off, the
  `exact_kv_and_rows` sink, pooled tail; the json's `conditions` carry it). So
  "block 49 is Sol's best block" holds WITH `qk_balance` on. With it off,
  block 49's quantization term is the one `qk_balance` was built to cut
  (`docs/h3_block49_quant_error.md`), and the ranking there is unmeasured.
- **Nothing about what the owner will see.** Capture error and the eye have
  disagreed in this pack (the reorder won most capture cells and no panel,
  `2026-09-18_sol_reorder_panel.md`). This nominates candidates for a render
  check; it chooses no default. The review's C.11 screening test and a blind
  render (`dense_blocks` = 40 against empty, and against the 47-49 practice)
  come before any change.
- **Thirty-eight blocks are still unmeasured.** Twelve of fifty, at two steps.
  A block between the sampled ones could outrank block 40. The online
  profiler planned in `internal/COORDINATION.md` measures all fifty.
- **Eight heads of fifty-six**, the same prefix everywhere. Whether a block's
  error sits in a few heads is unmeasured here (the review's feature 7).
- **Sage chain only**, because that is where the capture hook lives. What is
  captured is the input to attention, so the Sol and dense-kernel numbers are
  functions of the trajectory, and the sage chain's trajectory differs from
  the kitchen chain's on the first four steps.
- **The sage modes the chain now uses are not graded here.** The grader's sage
  arm is plain fp8++; `fp8++ balanced` and `fp8++ rotated` are graded on other
  cells in `2026-09-17_channel_balance_vs_sage_balanced_*.json` and
  `2026-09-17_sage_qk_rotate_kernel.json`.
