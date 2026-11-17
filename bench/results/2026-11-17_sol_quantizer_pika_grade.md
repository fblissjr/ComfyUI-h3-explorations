# Sol's quantizers re-graded on Pika H3 Flex captures, with today's dense tail (2026-11-17)

**Question (the owner's).** The fork's `rotate` (the `rotated` quantizer, on 83 of the 85 shipped Sol nodes) was
chosen on the 2026-09-27 grade (`2026-09-27_sol_redesign_test2.md`). That grade was taken on PDD8 graphs, which no
longer run, and while blocks 45, 48 and 49 all ran dense. Does `rotated` still beat `plain` (upstream's own Sol
quantizer, Sage-style smoothed K) on what runs today: Pika, with only block 49 dense?

**Answer.** Yes on quantization error, on every cell. The total error barely moves except at block 45:
- **Quantization error:** `rotated` is lower on all 14 cells, median −13.2%.
- **Total error against fp32 dense:** lower on 13 of 14 cells, but by a median of only −0.20%.
- **Block 45 is the exception:** its total falls by 2.25% and 1.54%. It is the block that moved from dense to Sol on
  2026-11-11, and where `plain` quantizes worst.
- **Whether any of this is visible in a render** is not measured here.

## How it was measured

**The capture.** `h3_capture.py` was armed on a throwaway ComfyUI started from a user shell on `127.0.0.1:8189`. The
service cannot write `$H3_CAPTURE_ROOT`: it runs with `ProtectHome=tmpfs` and read-only binds. That server ran the
shipped `workflows/h3_text_to_video_pika_api.json` once, at length 141, seed as shipped, Sol as shipped (`rotated`,
`dense_blocks="49"`, tau 1.0), with
`H3_CAPTURE="dir=$H3_CAPTURE_ROOT/2026-11-17_pika_sol_quantizer,blocks=0:24:40:44:45:47:48,steps=1:4,cycle=6"`.
- It wrote 14 cells, q/k/v bf16 `(1, 56, 43189, 128)`: 383 text, 470 audio and 42,336 video rows.
- The cells total 25 GB under `$H3_CAPTURE_ROOT/2026-11-17_pika_sol_quantizer/`. Deleting them is the owner's call.
- Step 1 is sigma 0.973 and step 4 is sigma 0.571.

**The grade.** `bench/grade_sol_quantizer_on_capture.py <the 14 cells> --tau 1.0`, on the installed kernel
`0.2.41+sol.b63d8c7.up.7d6a22a`. It reports rel L2 over the fp32 dense norm, pooled over all 56 heads:
- `quant`: Sol against its own unquantized reference;
- `total`: Sol against fp32 dense attention.

The sink is off, as in that tool's other uses.

## Per cell, all heads (`plain` → `rotated`)

| cell | quant plain | quant rotated | change | total plain | total rotated | change |
|---|---|---|---|---|---|---|
| block 0, step 1 | 0.00621 | 0.00478 | −23.1% | 0.10322 | 0.10321 | −0.01% |
| block 0, step 4 | 0.00400 | 0.00314 | −21.6% | 0.08434 | 0.08433 | −0.01% |
| block 24, step 1 | 0.01630 | 0.01214 | −25.5% | 0.16546 | 0.16565 | +0.12% |
| block 24, step 4 | 0.01273 | 0.01111 | −12.7% | 0.11568 | 0.11564 | −0.03% |
| block 40, step 1 | 0.01371 | 0.01260 | −8.1% | 0.16942 | 0.16927 | −0.09% |
| block 40, step 4 | 0.02031 | 0.01773 | −12.7% | 0.17747 | 0.17734 | −0.07% |
| block 44, step 1 | 0.01256 | 0.01098 | −12.6% | 0.09136 | 0.09119 | −0.18% |
| block 44, step 4 | 0.01420 | 0.01234 | −13.1% | 0.08486 | 0.08451 | −0.41% |
| block 45, step 1 | 0.02404 | 0.01399 | −41.8% | 0.08769 | 0.08572 | −2.25% |
| block 45, step 4 | 0.02203 | 0.01473 | −33.1% | 0.08233 | 0.08107 | −1.54% |
| block 47, step 1 | 0.01324 | 0.01219 | −7.9% | 0.07550 | 0.07534 | −0.22% |
| block 47, step 4 | 0.01259 | 0.01149 | −8.7% | 0.05765 | 0.05745 | −0.35% |
| block 48, step 1 | 0.01332 | 0.01155 | −13.3% | 0.07066 | 0.07051 | −0.21% |
| block 48, step 4 | 0.01342 | 0.01136 | −15.4% | 0.06230 | 0.06193 | −0.60% |

**Summaries across the 14 cells:**
- `plain` medians: total 0.0863, quant 0.0134. As in 2026-09, sparsity dominates the total.
- `balanced+rotated` against `rotated`: a further −0.4% on quant, median.
- The full record, with open and shut heads apart and `balanced` alongside, is `grade_pika_quantizer.json`. It was
  written in the session's scratchpad and not kept. Re-run the command above on the cells to get it again.

## What it does and does not decide

- **`rotated` still beats `plain` on Pika today, everywhere it was graded,** and most where the 2026-11-11 tail change
  put a block on Sol (45).
- **The quantity at stake is small next to Sol's own sparsity error,** as in 2026-09.
- **One variant was not graded:** `plain` with block 45 also dense (`dense_blocks="45,49"`) would cover the one block
  where `rotate` matters, on stock comfy-kitchen, at the cost of one more dense block. It is not graded here.
- **Separately, the fork stays needed if `video_reference_rows="routed"` becomes a default.** Routed rows between exact
  ones go to the kernel as the fork's `tau_map` (`sol_attn_h3.py::_tau_map`).
- **What this cannot see:**
  - one render, one seed, one prompt, one length (141), two steps, seven blocks;
  - numbers on attention outputs, not pictures.
