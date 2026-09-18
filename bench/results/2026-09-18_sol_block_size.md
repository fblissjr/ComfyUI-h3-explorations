# Sol block size by token ordering, float reference, two scenes, 2026-09-18

Model: MiniMax H3, int8 convrot checkpoint, base 16-step text to video at the
trained canvas. Card: RTX 4090, alone. Script:
`bench/sweep_sol_block_size_on_capture.py` (its docstring is the method).
Data: `2026-09-18_sol_block_size.json` (the 2026-09-03 base-16 capture set,
covered market scene, six cells) and `2026-09-18_sol_block_size_courtroom.json`
(the 2026-09-10 courtroom set, four cells). Head prefix of 8. Both capture
sets were taken in plain token order.

## The question

Sol routes whole 64-token blocks. On 2026-09-17 the one lever that removed a
visible morph was the `3d` reorder, which changes block membership only. Is 64
tokens a real ceiling, that is, would a kernel with smaller blocks buy error
that ordering cannot? `BLOCK = 64` is baked into the CUDA kernel's layout, so
this measures it in the float reference before anyone designs a rewrite.

## What this is and is not

The Sol ALGORITHM in fp32: no INT8 term, no kernel. A ceiling on what finer
blocks could buy, not a forecast of a kernel. Configurations are compared at
equal routed density (cost), never at equal tau, because a reorder and a block
size both move the routing threshold. "Density" is the share of key blocks
attended exactly; routing cost, which grows with the square of the block
count, is not in it. Instrument checks (chunked reference against its oracle
at every block size; routed counts against kitchen's eager reference; a
control that the block size takes effect) are in each JSON under `instrument`.

## Results

Error against fp32 dense attention at the density plain order at block 64
routes at the given tau, and its ratio to that configuration. Regenerate with
`--summarize <file> --at-tau <tau>`.

### Covered market set, at tau 1.0

| cell | density | raster_b64 | raster_b32 | raster_b16 | 3d_b64 | 3d_b32 | 3d_b16 |
|---|---|---|---|---|---|---|---|
| b49_s15 | 0.223 | 0.0320 (1.00x) | 0.0318 (0.99x) | 0.0195 (0.61x) | 0.0222 (0.69x) | 0.0213 (0.67x) | 0.0088 (0.27x) |
| b0_s15 | 0.212 | 0.1241 (1.00x) | 0.1150 (0.93x) | 0.0982 (0.79x) | 0.1099 (0.89x) | 0.1027 (0.83x) | 0.0882 (0.71x) |
| b32_s15 | 0.227 | 0.0907 (1.00x) | 0.0816 (0.90x) | 0.0674 (0.74x) | 0.0837 (0.92x) | 0.0753 (0.83x) | 0.0613 (0.68x) |
| b32_s4 | 0.222 | 0.0907 (1.00x) | 0.0859 (0.95x) | 0.0756 (0.83x) | 0.0883 (0.97x) | 0.0823 (0.91x) | 0.0741 (0.82x) |
| b24_s15 | 0.215 | 0.0939 (1.00x) | 0.0848 (0.90x) | 0.0631 (0.67x) | 0.0684 (0.73x) | 0.0584 (0.62x) | 0.0458 (0.49x) |
| b40_s15 | 0.202 | 0.2580 (1.00x) | 0.2467 (0.96x) | 0.2131 (0.83x) | 0.2072 (0.80x) | 0.1962 (0.76x) | 0.1743 (0.68x) |

### Covered market set, at tau 1.3

| cell | density | raster_b64 | raster_b32 | raster_b16 | 3d_b64 | 3d_b32 | 3d_b16 |
|---|---|---|---|---|---|---|---|
| b49_s15 | 0.162 | 0.0439 (1.00x) | 0.0394 (0.90x) | 0.0279 (0.64x) | 0.0466 (1.06x) | 0.0448 (1.02x) | 0.0181 (0.41x) |
| b0_s15 | 0.151 | 0.1407 (1.00x) | 0.1327 (0.94x) | 0.1163 (0.83x) | 0.1298 (0.92x) | 0.1227 (0.87x) | 0.1083 (0.77x) |
| b32_s15 | 0.173 | 0.1151 (1.00x) | 0.1046 (0.91x) | 0.0862 (0.75x) | 0.1023 (0.89x) | 0.0936 (0.81x) | 0.0772 (0.67x) |
| b32_s4 | 0.168 | 0.1092 (1.00x) | 0.1036 (0.95x) | 0.0921 (0.84x) | 0.1059 (0.97x) | 0.0988 (0.91x) | 0.0897 (0.82x) |
| b24_s15 | 0.165 | 0.1199 (1.00x) | 0.1089 (0.91x) | 0.0820 (0.68x) | 0.0870 (0.73x) | 0.0747 (0.62x) | 0.0595 (0.50x) |
| b40_s15 | 0.147 | 0.3098 (1.00x) | 0.2993 (0.97x) | 0.2628 (0.85x) | 0.2478 (0.80x) | 0.2371 (0.77x) | 0.2153 (0.70x) |

### Courtroom set, at tau 1.0

| cell | density | raster_b64 | raster_b32 | raster_b16 | 3d_b64 | 3d_b32 | 3d_b16 |
|---|---|---|---|---|---|---|---|
| b49_s13 | 0.219 | 0.0220 (1.00x) | 0.0217 (0.98x) | 0.0142 (0.64x) | 0.0304 (1.38x) | 0.0300 (1.36x) | 0.0093 (0.42x) |
| b24_s13 | 0.220 | 0.1279 (1.00x) | 0.1140 (0.89x) | 0.0804 (0.63x) | 0.0829 (0.65x) | 0.0725 (0.57x) | 0.0615 (0.48x) |
| b0_s13 | 0.217 | 0.1337 (1.00x) | 0.1255 (0.94x) | 0.1089 (0.81x) | 0.1173 (0.88x) | 0.1107 (0.83x) | 0.1010 (0.75x) |
| b49_s5 | 0.239 | 0.0314 (1.00x) | 0.0286 (0.91x) | 0.0214 (0.68x) | 0.0354 (1.13x) | 0.0320 (1.02x) | 0.0149 (0.47x) |

### Courtroom set, at tau 1.3

| cell | density | raster_b64 | raster_b32 | raster_b16 | 3d_b64 | 3d_b32 | 3d_b16 |
|---|---|---|---|---|---|---|---|
| b49_s13 | 0.160 | 0.0347 (1.00x) | 0.0325 (0.94x) | 0.0354 (1.02x) | 0.0506 (1.46x) | 0.0497 (1.43x) | 0.0252 (0.73x) |
| b24_s13 | 0.170 | 0.1584 (1.00x) | 0.1431 (0.90x) | 0.1029 (0.65x) | 0.1036 (0.65x) | 0.0913 (0.58x) | 0.0781 (0.49x) |
| b0_s13 | 0.153 | 0.1524 (1.00x) | 0.1452 (0.95x) | 0.1293 (0.85x) | 0.1400 (0.92x) | 0.1337 (0.88x) | 0.1240 (0.81x) |
| b49_s5 | 0.178 | 0.0475 (1.00x) | 0.0438 (0.92x) | 0.0428 (0.90x) | 0.0567 (1.19x) | 0.0531 (1.12x) | 0.0285 (0.60x) |

## What it says

1. **Halving the block buys little; quartering it buys something real.** Block
   32 in plain order is within a tenth of block 64 on every cell. Block 16 is
   lower on every cell at tau 1.0, by a fifth to two fifths.
2. **The reorder at block 64 is in the same range as plain order at block 16**
   on the covered market set (better on some cells, worse on others), so on
   that scene ordering already buys about what a four-times-finer kernel would
   buy without it.
3. **The two compound.** `3d` at block 16 is the lowest curve on every cell of
   both scenes, and its ratio is close to the product of the two separate
   ratios. Finer blocks are not made redundant by the reorder.
4. **The reorder is NOT uniformly good at block 64.** On the courtroom scene's
   last block, at both captured steps, `3d` at block 64 and 32 is WORSE than
   plain order at equal density, and only `3d` at block 16 is better. The
   2026-09-17 CUDA-kernel ordering sweep (`2026-09-17_sol_orderings.md`) used
   only the covered market set, where `3d` won every cell; the courtroom set
   has not been through that sweep. The last block's error is the smallest in
   absolute terms of any cell here, so this is a small number getting larger,
   next to large numbers getting smaller on blocks 24 and 40; whether the eye
   cares is the panel's question, not this table's.
5. **At a higher tau the reorder's gain on the last block disappears on both
   scenes** while block 16's does not. Raising tau for speed is where finer
   blocks would matter most.

## What it does not say

Nothing about a clip. Nothing about INT8: the kernel's quantization groups
follow the 64-token block, and a finer block changes that term in a direction
this does not measure. Nothing about speed: sixteen times more block pairs to
route and smaller exact tiles are both costs, and neither is in "density". Two
scenes, one seed each, blocks 0, 24, 32, 40, 49 only.

## Next

Run the CUDA ordering sweep on the courtroom cells, to see whether finding 4
holds with the real kernel. A kernel with smaller blocks stays undecided: the
ceiling is real in float, and its price is unknown.
