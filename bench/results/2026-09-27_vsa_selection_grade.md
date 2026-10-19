# #45: core's kitchen VSA against a FastVideo-exact reference, on captured FastH3 attention (2026-09-27)

`docs/state/open_experiments.md` #45 asks whether part of FastH3's over-polish is the
kitchen's VSA rather than FastH3's training. Core's VSA departs from FastVideo
three ways (`2026-09-27_attention_parity.md`):
- it keeps `round(0.2n)` video tiles where FastVideo keeps `ceil`;
- it forces the ±1 diagonal exact;
- it runs int8 where FastVideo runs bf16.

**The capture.** lookingdude's session (`3ea2e885`) rendered FastH3 V2 on its
contract (slapstick_moving_piano, seed 730451892, 345 frames) with
`MiniMaxH3CoreSparseCapture`.
- It captured the fused projection and the coarse gate at blocks 0, 24 and 49,
  steps 2 and 6: 12 records, manifest schema 1.8.0, kept to 2026-10-31.
- The armed render's final latent is bit-identical to the unarmed one
  (CHANGELOG 0.164.2).

**The grade.** `bench/grade_vsa_selection_on_capture.py` (`07d4885f`), GPU,
kitchen `0.2.35+sol.fc32da2.up.c8c7825`; the per-cell record is
`2026-09-27_vsa_selection_grade.json`. Per cell, all against exact fp32
attention:
- **kitchen:** core's `ck.sol_attn_chunked` call replayed on the captured
  projection, with core's own tile plan, rope, sinks and chunking. Statistics
  are fresh: the real calls carried the previous step's.
- **reference:** FastVideo's `video_sparse_attn_h3` semantics in fp32: tile-mean
  scores, `ceil` kept count, prefix tiles kept and prefix queries dense, no
  forced diagonal, and the coarse term `softmax(scores) @ pooled v` times the
  gate.
- **Control:** the reference with every tile kept equals exact attention to
  1.9e-3 (bf16 output rounding) on seven sampled query tiles.

| block | step | kitchen | kitchen fine | reference | reference fine | kitchen vs reference | coarse share (ref) |
|---|---|---|---|---|---|---|---|
| 0 | 2 | .1132 | .1117 | .1138 | .1124 | .0075 | .015 |
| 0 | 6 | .0992 | .0976 | .0998 | .0982 | .0049 | .016 |
| 24 | 2 | .1005 | .0976 | .1001 | .0972 | .0117 | .023 |
| 24 | 6 | .0912 | .0885 | .0909 | .0881 | .0137 | .022 |
| 49 | 2 | .1252 | .0830 | .1164 | .0691 | .0497 | .093 |
| 49 | 6 | .1228 | .0714 | .1157 | .0581 | .0440 | .098 |

(Relative L2 against exact attention, whole tensor. The kept video tiles are
344 by `ceil` and 343 by `round`, of 1716.)

By segment, block 49 against exact:

| block | step | rows | kitchen | reference |
|---|---|---|---|---|
| 49 | 2 | video | .124 | .116 |
| | | text | .289 | .040 |
| | | audio | .222 | .183 |
| 49 | 6 | video | .121 | .116 |
| | | text | .337 | .041 |
| | | audio | .147 | .124 |

## Reading

- **At blocks 0 and 24 the kitchen is the reference, to within about 1%.**
  - Both sit 9-11% from exact attention, which is the cost of keeping 20% of
    the video tiles and is FastH3's trained regime.
  - The three deviations (one tile fewer, the forced diagonal, int8) move the
    output by 0.5-1.4% of its size.
- **At block 49 the kitchen is 4.4-5% from the reference and further from
  exact.** The gap is in the prefix rows, not in selection:
  - on text rows the kitchen is 29-34% from exact where the reference is 4%;
  - on video rows the two are within 1% of each other.
  - Text queries are computed densely by both. So the gap is the kitchen's
    int8 arithmetic on block 49's outlier keys, the pattern
    `docs/attention/block49_quant_error.md` documents for the base model's last
    blocks. It is not VSA selection.
- **The coarse term matters most at block 49,** about 9.5% of the output
  there against 1.5-2.3% at blocks 0 and 24.
- **So #45's question is answered for selection.** The kitchen's selection
  deviations do not plausibly produce FastH3's over-polish: on video rows at
  all three blocks the kitchen matches the reference to within 1%.
- **Inference, not measured:** FastH3's high detail is trained. The one
  measurable gap is int8 precision on block 49's text and audio rows.

## What would decide the rest

Whether block 49's text-row error is visible needs a render with block 49's
attention in bf16 and everything else on the contract.
- The FastVideo-exact reference node (board direction
  "vsa-fastvideo-reference") would give that.
- Core's `dense_blocks="49"` would not. It drops the coarse term, about 9.5%
  of block 49's output, and runs dense where FastH3 trained sparse, so any
  change would be confounded.

## Limits

- One scene, one seed, one render: three blocks at two steps.
- The replay uses fresh statistics where the real calls carried the previous
  step's.
- The reference computes in fp32, which is stricter than FastVideo's bf16
  kernel.
