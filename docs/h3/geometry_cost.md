# What canvas and length cost: the tables

The detail under [`geometry.md`](geometry.md), which states the rules. Every figure here is arithmetic over
`adapt_canvas` and `temporal_shape` (core, `comfy_extras/nodes_minimax_h3.py`) and the token grid `(W/32) x (H/32)`,
unless a row says measured. Attention is quoted as the square of the token ratio, because attention is O(S^2) and
video tokens dominate S; it prices attention only, not a whole step. Exact packed rows for one graph, text and
references included: `bench/count_packed_rows.py --length N --canvas WxH`.

## The common ratios

What `adapt_canvas` returns for the fourteen ratios most likely wanted. Attention is against 1344x768 at the same
length.

| ask for | canvas | VAE latent | token grid | tokens/frame | attention |
|---|---|---|---|---|---|
| 21:9 | 1536x672 | 96x42 | 48x21 | 1008 | 1.00x |
| 2:1 | 1440x704 | 90x44 | 45x22 | 990 | 0.96x |
| 16:9 | 1344x768 | 84x48 | 42x24 | 1008 | 1.00x |
| 5:3 | 1280x768 | 80x48 | 40x24 | 960 | 0.91x |
| 3:2 | 1152x768 | 72x48 | 36x24 | 864 | 0.73x |
| 4:3 | 1024x768 | 64x48 | 32x24 | 768 | 0.58x |
| 5:4 | 960x768 | 60x48 | 30x24 | 720 | 0.51x |
| 1:1 | 768x768 | 48x48 | 24x24 | 576 | 0.33x |
| 4:5 | 768x960 | 48x60 | 24x30 | 720 | 0.51x |
| 3:4 | 768x1024 | 48x64 | 24x32 | 768 | 0.58x |
| 2:3 | 768x1152 | 48x72 | 24x36 | 864 | 0.73x |
| 9:16 | 768x1344 | 48x84 | 24x42 | 1008 | 1.00x |
| 1:2 | 704x1440 | 44x90 | 22x45 | 990 | 0.96x |
| 9:21 | 672x1536 | 42x96 | 21x48 | 1008 | 1.00x |

Going wider than 1344x768 is free and buys no speed: the area cap binds and trades width for height. The only cheap
direction is toward square, a smooth 32-pixel ramp, and because attention goes as the square of tokens, small width
steps move the cost a lot. The shipped tiers sit on that ramp (`workflows/h3_config.py::CANVAS_TIERS`, with the note
above `CANVAS_TIER` on which tier is for what).

## Every canvas in the trained family

`adapt_canvas` produces 95 distinct canvases over the trained aspect range. They are symmetric, so the 48 below cover
landscape and square and each has a portrait mirror at identical cost. "Re-derives: no" marks the 26 that are not
fixed points: typed back into `adapt_canvas` they return a neighbor, because the area cap scales the pair and the
rounding does not land where it started. They are reachable and legal. The latent column is the VAE latent; halve
each axis for the token grid.

| canvas | latent | tokens/frame | attention | re-derives |
|---|---|---|---|---|
| 2016x512 | 126x32 | 1008 | 1.00x |  |
| 1984x512 | 124x32 | 992 | 0.97x |  |
| 1952x512 | 122x32 | 976 | 0.94x | no |
| 1952x544 | 122x34 | 1037 | 1.06x | no |
| 1920x544 | 120x34 | 1020 | 1.02x |  |
| 1888x544 | 118x34 | 1003 | 0.99x |  |
| 1856x544 | 116x34 | 986 | 0.96x | no |
| 1856x576 | 116x36 | 1044 | 1.07x | no |
| 1824x576 | 114x36 | 1026 | 1.04x | no |
| 1792x576 | 112x36 | 1008 | 1.00x |  |
| 1760x576 | 110x36 | 990 | 0.96x |  |
| 1728x576 | 108x36 | 972 | 0.93x | no |
| 1728x608 | 108x38 | 1026 | 1.04x |  |
| 1696x608 | 106x38 | 1007 | 1.00x |  |
| 1664x608 | 104x38 | 988 | 0.96x | no |
| 1664x640 | 104x40 | 1040 | 1.06x | no |
| 1632x640 | 102x40 | 1020 | 1.02x |  |
| 1600x640 | 100x40 | 1000 | 0.98x |  |
| 1568x640 | 98x40 | 980 | 0.95x | no |
| 1568x672 | 98x42 | 1029 | 1.04x | no |
| 1536x672 | 96x42 | 1008 | 1.00x |  |
| 1504x672 | 94x42 | 987 | 0.96x |  |
| 1504x704 | 94x44 | 1034 | 1.05x | no |
| 1472x704 | 92x44 | 1012 | 1.01x |  |
| 1440x704 | 90x44 | 990 | 0.96x |  |
| 1440x736 | 90x46 | 1035 | 1.05x | no |
| 1408x736 | 88x46 | 1012 | 1.01x |  |
| 1376x736 | 86x46 | 989 | 0.96x |  |
| 1376x768 | 86x48 | 1032 | 1.05x | no |
| 1344x768 | 84x48 | 1008 | 1.00x |  |
| 1312x768 | 82x48 | 984 | 0.95x |  |
| 1280x768 | 80x48 | 960 | 0.91x |  |
| 1248x768 | 78x48 | 936 | 0.86x |  |
| 1216x768 | 76x48 | 912 | 0.82x |  |
| 1184x768 | 74x48 | 888 | 0.78x |  |
| 1152x768 | 72x48 | 864 | 0.73x |  |
| 1120x768 | 70x48 | 840 | 0.69x |  |
| 1088x768 | 68x48 | 816 | 0.66x |  |
| 1056x768 | 66x48 | 792 | 0.62x |  |
| 1024x768 | 64x48 | 768 | 0.58x |  |
| 992x768 | 62x48 | 744 | 0.54x |  |
| 960x768 | 60x48 | 720 | 0.51x |  |
| 928x768 | 58x48 | 696 | 0.48x |  |
| 896x768 | 56x48 | 672 | 0.44x |  |
| 864x768 | 54x48 | 648 | 0.41x |  |
| 832x768 | 52x48 | 624 | 0.38x |  |
| 800x768 | 50x48 | 600 | 0.35x |  |
| 768x768 | 48x48 | 576 | 0.33x |  |

The worst case in the set is 1856x576, 7.3% over 1344x768 for no extra pixels; nearby ratios do not do this.

## Every length, at 1344x768

The on-grid counts in the usable range, with attention against the ceiling (`h3_rules.MAX_LENGTH`, the last row).

| frames | latent frames | duration | video tokens | attention vs the ceiling |
|---|---|---|---|---|
| 124 | 37 | 5.167s | 37,296 | 0.12x |
| 209 | 62 | 8.708s | 62,496 | 0.34x |
| 243 | 72 | 10.125s | 72,576 | 0.45x |
| 260 | 77 | 10.833s | 77,616 | 0.52x |
| 277 | 82 | 11.542s | 82,656 | 0.59x |
| 294 | 87 | 12.250s | 87,696 | 0.66x |
| 311 | 92 | 12.958s | 92,736 | 0.74x |
| 328 | 97 | 13.667s | 97,776 | 0.82x |
| 345 | 102 | 14.375s | 102,816 | 0.91x |
| 362 | 107 | 15.083s | 107,856 | 1.00x |

Every bench figure in this repo was taken at the ceiling, and `workflows/h3_config.py` warns that moving the length
breaks the comparison.

## The Sol-Attn token floor

Below about 60,000 video tokens a Sol-Attn arm returns a null that reads as "this knob does nothing"
(`bench/bench_e2e_h3.py::SOL_TOKEN_FLOOR`, sourced from upstream's report of nothing measurable under roughly 250-300
frames at 1344x768). Canvas and length both count, so cheap choices can fall under it. Video tokens; bold is under the
floor:

| canvas | 243 frames | 311 frames | 362 frames |
|---|---|---|---|
| 1344x768 | 72,576 | 92,736 | 107,856 |
| 1280x768 | 69,120 | 88,320 | 102,720 |
| 1152x768 | 62,208 | 79,488 | 92,448 |
| 1024x768 | **55,296** | 70,656 | 82,176 |
| 896x768 | **48,384** | 61,824 | 71,904 |
| 768x768 | **41,472** | **52,992** | 61,632 |

So a Sol measurement may use the `fast` tier and must not use `draft`, and a square canvas is the wrong one to measure
Sol on at any length but the ceiling. A graph with references shifts the segment balance; read its preflight rather
than this table. [`../attention/sol_attn.md`](../attention/sol_attn.md) owns every Sol number.

## The int32 crossing, and why there is no 100,000-token budget

H3 builds q, k and v as three views of one fused buffer, so the stride between them is `3 x heads x head_dim`
(`preflight.py::_FUSED_STRIDE`), and a signed int32 offset in a quantization kernel crosses at `preflight.py::_INT32_FUSED`,
just under 100,000 tokens. That is the whole origin of the "about 100k" figure: a kernel's offset width, not a limit
of the model. At 1344x768 the crossing sits between 311 and 328 frames (sequence lengths, text included, from
`PackedLayout`):

| frames | sequence length | |
|---|---|---|
| 124 | 41,822 | |
| 260 | 82,594 | |
| 311 | 97,884 | under |
| 328 | 102,982 | over |
| 345 | 108,078 | over |
| 362 | 113,406 | over |

The crossing is a property of the sage quantization kernels behind `MiniMaxH3SageAttention`. Every long graph is
past it, which is safe there because `attention.py::build_kernel` refuses any sageattention without
`sageattn_consume`, so every user of that node is on the fork that quantizes q and k with the int64 fix (fixed in the
fork's v0.7.0). The next ceiling is a `uint32` wrap in the CUDA kernels that quantize v
(`preflight.py::_CSRC_FUSED`, about twice the longest legal sequence), unfixed upstream code and unreachable on 24 GB;
the fork's CHANGELOG, "The CUDA quant kernels form global offsets in uint32", has the arithmetic. Preflight reports
which side of each a graph is on.

KJNodes' `MiniMaxH3TokenCounter` computes its warning from the contiguous stride (`heads x head_dim`) and so places
the threshold about three times higher. At these lengths it stays silent past the fused crossing: its silence is not
clearance.

## Sol block geometry by canvas (the closed Morton lane)

Sol-Attn summarizes the sequence in blocks of 64 tokens. In raster order a block is a band a few rows tall and the
full frame wide. The Morton reorder that made blocks into compact 3-D bricks was retired on 2026-09-27
(`docs/state/roadmap.md`, "Closed lanes"), so nothing here runs in a shipped graph; what that lane measured about
canvas and length, kept because it is still true of the geometry:

- **Canvas**: under the `3d` curve the bricks came out solid only when both token-grid axes divide by 4, that is
  width and height both multiples of 128. Six landscape canvases qualify (1664x640, 1152x768, 1280x768, 1024x768,
  896x768, 768x768, each with a portrait mirror); 1344x768 does not (42 tokens wide) and scored 13th of 48.
- **Length**: `latent_t % 4 == 3` was the case to avoid (158, 226, 294 and 362 frames); alignment at
  `latent_t % 4 == 0` (175, 243, 311) improved the block radius but not connectivity.
- **None of it is output quality.** Block geometry describes the router's summaries, not the picture, and was never
  shown to reach the screen.

[`../attention/morton.md`](../attention/morton.md) is that lane's record; the block maps and the 48-canvas ranking
are in the git history of the page this section replaced, `docs/h3/geometry_cost.md`.
