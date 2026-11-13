# Sol's quantizer and the dense tail, graded on PDD8 captures (2026-09-27)

Test 2 of the Sol node redesign
(`docs/wiki/archive/2026-09-27_sol_node_redesign.md`). It follows test 1
(`2026-09-27_sol_redesign_test1.md`). That test found `quantizer` changes
every render, and gave a provisional grade on 2026-09-19 base-model
captures. This record grades what ships.

## What was captured

- **Graphs:** the shipped PDD8 graphs on main `1ec6c8ce`, with kitchen
  `0.2.35+sol.fc32da2.up.c8c7825` and `MiniMaxH3Sol` at its shipped settings
  (quantizer balanced, dense_blocks 45,48,49), seed 730451892:
  - t2v: `distill_experiments/h3_text_to_video_pdd_savelat_api.json` on
    slapstick_moving_piano at 345 frames.
  - ref2va: `h3_image_ref_plus_text_to_video_pdd_api.json` at its own
    prompt and references.
  - Manifest: `bench/sol_redesign_test2_capture_arms.json`. Rows:
    `2026-09-27_sol_redesign_test2_capture.jsonl`.
- **Cells:** blocks 0, 8, 24, 40 and 44 to 49, at PDD8 steps 2 and 6. Sol's
  window opens at step 2. That is 20 cells per render, recorded in
  `2026-09-27_capture_manifest_sol_test2_{t2v,ref2va}.json` and the matching
  inventories.
- **Sequence lengths:** t2v 104,565 rows, ref2va 120,582.

## The quantizer, on the Sol blocks

`bench/grade_sol_quantizer_on_capture.py` graded blocks 0, 8, 24, 40, 44, 46
and 47 (the Sol blocks) at both steps in both modes: 28 cells, every head.
Record: `2026-09-27_sol_quantizer_grade_pdd8.json`. On every cell,
balanced equals plain bit for bit on heads whose gate is shut.

**The quantization term** (Sol against its unquantized reference). Each
line compares the first quantizer against the second across the 28 cells:

| Comparison | Cells lower | Median | Range |
|---|---|---|---|
| rotated vs plain | 28/28 | -11.7% | -20.3 to -4.3% |
| rotated vs balanced | 28/28 | -10.6% | -19.5 to -3.7% |
| balanced vs plain | 23/28, 2 ties | -0.7% | -6.4 to +0.2% |
| balanced+rotated vs rotated | 21/28, 2 ties | -0.1% | -2.3 to +0.2% |

**Total error against fp32 dense.** Sol's sparsity error (median 0.098 on
plain) swamps the quantization term (about 0.01), so the choice barely
moves the total:

| Comparison | Cells lower | Median | Range |
|---|---|---|---|
| rotated vs balanced | 21/28 | -0.05% | -0.32 to +0.13% |
| balanced+rotated vs rotated | 17/28 | -0.003% | |

The two ties in the balanced rows are block 24 at step 2 in both modes,
where no head opens the gate.

**Kernel time.** `bench/time_sol_options_on_capture.py`, t2v cells, all 56
heads, 7 iterations. The card was alone, with no server. Median call time
against plain:

| Cell | balanced | rotated | both |
|---|---|---|---|
| b0 s6 | x1.018 | x1.010 | x1.020 |
| b44 s6 | x1.015 | x1.008 | x1.019 |

The base-model grade on 2026-09-19 captures (`2026-09-27_sol_quantizer_grade.json`)
ranked them the same way, so PDD8 does not change the ranking.

## The dense tail: kitchen int8 against sage

`bench/grade_dense_kernels_on_captures.py`, blocks 45, 48 and 49 at both
steps, all 56 heads, rel L2 against fp32 dense. The sage rotated kernel
(the Sage node's "fp8++ rotated" and "auto" modes) was added to the grader
for this. Records: `2026-09-27_dense_tail_grade_pdd8_{t2v,ref2va}.json`.

| Cell | kitchen int8 | sage fp8++ | sage fp8++ rotated | bf16 SDPA |
|---|---|---|---|---|
| t2v b45 s2 / s6 | 0.0125 / 0.0133 | 0.0310 / 0.0290 | 0.0184 / 0.0201 | 0.0017 |
| t2v b48 s2 / s6 | 0.0093 / 0.0113 | 0.0153 / 0.0185 | 0.0138 / 0.0174 | 0.0018 |
| t2v b49 s2 / s6 | 0.0140 / 0.0155 | 0.0494 / 0.0552 | 0.0210 / 0.0242 | 0.0016 / 0.0015 |
| ref2va b45 s2 / s6 | 0.0128 / 0.0129 | 0.0297 / 0.0275 | 0.0197 / 0.0199 | 0.0017 |
| ref2va b48 s2 / s6 | 0.0106 / 0.0113 | 0.0174 / 0.0185 | 0.0162 / 0.0174 | 0.0018 |
| ref2va b49 s2 / s6 | 0.0367 / 0.0360 | 0.1458 / 0.1407 | 0.0405 / 0.0405 | 0.0015 |

- **Kitchen int8 is the more accurate dense kernel on all 12 cells.**
- **Sage without rotation is the worst on every cell.**
- **ref2va's block 49 is the hardest cell** for every int8 kernel.
  *Note 2026-09-29:* largely a property of the grade, which counts every query row, and at block 49 the text and reference rows' output is never read; see `2026-09-29_ref2va_block49_hardest_cell.md`. *Confirmed on the real kernels, 2026-09-29:* over the video and audio rows only, ref2va's block 49 is close to t2v's, not several times worse; see `2026-09-29_grade_video_audio_real_kernels.md`.

This answers audit §9b's open question. On matched cells, the kitchen chain
serves the dense tail better than the sage chain.

## What it decides, and what it does not

- **`rotated` is the better quantizer.** It gives the lowest quantization
  error on every cell, and it costs less than `balanced` (about 1% of Sol's
  call time against about 1.5 to 2%).
- **`balanced+rotated` does not beat `rotated` by enough to justify its
  extra cost.**
- **The whole choice is small next to Sol's sparsity error.** At the total
  it moves error by a median of 0.05%. That is why test 1's renders differ
  (any rounding change diverges the trajectory), while the quality at stake
  is small.
- **The default is not flipped in this record.** Two reasons:
  - Flipping changes the output of every PDD8 graph. It would break bit
    comparability with every reference rendered today, for a gain this
    small. When and whether to flip is the owner's call.
  - The "all blocks" routing preset requires a balanced quantizer (bug #7).
    Test 3 re-derives that on the fixed kernel.
- **The dense tail still beats Sol on PDD8** (graded the same night on the
  same 12 tail cells; record `2026-09-27_sol_quantizer_grade_pdd8_tail.json`,
  set against the dense-tail kernel grade above):
  - Sol's total error against fp32 dense is 0.07 to 0.13 under `rotated`.
  - Kitchen int8 dense is 0.009 to 0.037 on the same cells, several times
    closer on all 12, in both modes.
  - So `dense_blocks` 45,48,49 stays.
  - These are also the cells where the quantizer matters most. On block 49
    `plain` is several times worse than `rotated`, and `balanced+rotated`
    is lower again. That doesn't bear on the default, because the tail runs
    dense.
