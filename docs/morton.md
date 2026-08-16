# Morton ordering in Sol-Attn: what it does, and what we actually know

Last updated: 2026-08-15.

## The short version

There is no confirmed quality result yet. Morton has never been tested on the
axis its author says it acts on, so "does Morton improve anything" remains
open in both directions.

What this page does establish, by measurement rather than by reading:

1. Morton cannot change dense attention at all. It is a permutation that gets
   undone, and attention is permutation-equivariant. The only thing it can
   change is which tokens share a 64-token block, which is what the sparse
   router summarizes. That makes most of the question pure arithmetic on the
   latent grid, answerable with no GPU and no watching of clips.
2. Morton delivers the compact tiles it promises on 3 of the 48 legal
   landscape canvases. The default canvas here, 1344x768, is not one of them.
   On it a block is typically two disconnected fragments in different parts of
   the frame.
3. Every Morton arm this project has run held `tau` fixed. The author's own
   docstring says the payoff appears at higher sparsity. So those arms were
   designed in a way that could not find the effect, and "a different seed
   that saves ten seconds" is the result you should expect from them.

If you only take one thing: the next Morton experiment should vary `tau` and
Morton together, not Morton alone.

## What a block is, concretely

The rest of this page talks about tokens and blocks. Both are physical things
here, and sizing them makes everything that follows easier to hold.

One token is one 32x32 pixel patch of one latent frame. At 1344x768 a frame is
42 patches wide and 24 patches high, so 1008 tokens per frame.

One block is 64 tokens, and it is the unit Sol-Attn makes a decision about:
compute this block exactly, or replace it with one average vector. That
decision is good when the 64 tokens resemble each other, because one vector
can then stand for all of them. It is bad when they do not.

The model produces tokens in raster order, meaning frame, then row, then
column. So 64 tokens in a row are a strip 42 patches wide and less than 2
patches tall, running the full width of the frame. Morton is supposed to
replace that strip with an 8x8 square of patches.

That is the whole question this page answers: does it, and what changes when
it does not.

## What Morton order is

Morton order, also called Z-order, maps a multi-dimensional coordinate to a
single number by interleaving the bits of the coordinates. For a 2D point
`(x, y)` with bits `x2 x1 x0` and `y2 y1 y0`, the Morton code is
`y2 x2 y1 x1 y0 x0`. Sorting points by that code walks the plane in a
recursive Z pattern: it finishes one quadrant before moving to the next, and
inside each quadrant it does the same thing again.

Guy Macdonald Morton described it at IBM in 1966 for file sequencing. The
property that made it last is that points close together in space usually end
up close together in the sorted order. It is not perfect. Two neighbors can
sit far apart when they fall on opposite sides of a high-order bit boundary,
which is the same weakness that shows up later in this page.

The idea is used wherever a multi-dimensional neighborhood has to become a
contiguous run of memory:

- GPU texture layout. Graphics hardware stores textures Morton-swizzled rather
  than row-major, so a 2D filtering kernel touches one cache line instead of
  several rows.
- Bounding volume hierarchy construction in ray tracing. Sorting primitives by
  Morton code turns hierarchy building into a sort, which is why it is the
  standard GPU BVH builder.
- Spatial databases and quadtree indexing, where a Z-order key lets a
  one-dimensional B-tree answer range queries over two dimensions.
- N-body and particle simulation, for the same locality reason.

Hilbert curves preserve locality better than Z-order and cost more to compute.
Z-order wins on ubiquity because the encode is a handful of bit operations.

## Where the same idea shows up in attention

Block-sparse attention has a layout problem that Morton is one answer to. If
you split a sequence into fixed blocks and decide per block pair whether to
compute exactly, the decision quality depends on how coherent a block is. A
block holding one object is well summarized by one pooled vector. A block
holding a strip across four unrelated regions is not.

For text this is mostly a non-issue, because token order already carries the
locality. For video it matters a lot: the natural flattening of a 3D latent
into a sequence is raster order, so 64 consecutive tokens are a thin
horizontal strip spanning the full width of a frame.

Several lines of work attack this, and they split into two families.

Content-based reordering, which sorts tokens by what they contain:

- Reformer (Kitaev et al., 2020) buckets tokens by locality-sensitive hash and
  sorts by bucket, so block-local attention becomes content-local. This is the
  closest conceptual ancestor: same move, content instead of geometry.
- Routing Transformer (Roy et al., 2020) clusters tokens with k-means and
  attends within clusters.
- Sparse VideoGen 2 does the video version. Read in its source by a survey
  agent on 2026-08-15: per-layer, per-head k-means over Q and K, warm-started
  across sampling steps, permute by cluster label, then route by comparing
  centroids. Morton is the fixed geometric guess at what SVG2 computes at
  runtime.

Geometry-based reordering, which sorts by position:

- Morton, as used here.
- Kandinsky 5's `fractal_flatten`, which the same survey found in the
  diffusers tree: an 8x8 spatial tiling so that 64-token flex-attention blocks
  are spatially contiguous. Same motivation, hardcoded tile size, not tunable.
- PAROAttention, "Pattern-Aware ReOrdering" for visual generation, reorders
  tokens so attention patterns become hardware-friendly block structures.
  Cited from memory, not read in source.

What the comparison suggests, and it is a suggestion rather than a finding:
the content-based family treats the layout as something to be computed per
head per layer, and the geometry-based family treats it as a global on/off
switch. Sol-Attn's Morton is a global switch. Whether a fixed curve is close
enough to a learned clustering on this model is unmeasured by anyone we found.

Two negative results worth recording, both from source greps on 2026-08-15:

- diffusers contains no space-filling-curve reordering at all. Searching the
  whole repository for morton, hilbert, z-order and space-filling returns
  nothing. Its only block permutation is Kandinsky's `fractal_flatten`.
- LightX2V refuses Morton on MiniMax-H3 outright, raising rather than running,
  because H3 packs text, audio and video into one sequence and a pure-video 3D
  Morton is not valid over that pack. Kijai's H3 implementation solves this by
  reordering only the video span, which is a real difference between the two.

## Related papers, and the honest limit on this section

Before: Morton (1966). Sparse Transformer (2019), Reformer (2020), Routing
Transformer (2020), Longformer and BigBird (2020) for the block-sparse
vocabulary. Native Sparse Attention and MoBA (2025) for block routing in LLMs.

Around and after, on video diffusion specifically: Sparse VideoGen and its
successor SVG2, radial attention, SpargeAttn, Video Sparse Attention,
PAROAttention, DraftAttention. All 2025.

The limit: the Sol-Attn paper is arXiv 2607.24027, which postdates this
assistant's training data. Nothing here is drawn from it, and its related-work
section may already answer questions this page treats as open. Reading it is a
task nobody has done yet. Every Sol-Attn claim below comes from the node
source, the CUDA kernels, or the eager reference implementation.

## How Sol-Attn uses blocks

Sol-Attn splits the sequence into 64-token blocks. For each query block it
routes a subset of key blocks to an exact branch and covers everything else
with one pooled term per block, so the full sequence still contributes to the
softmax denominator. Nothing is dropped; most of it is approximated.

`tau` sets the routing threshold. Read in the CUDA preprocessing kernel by the
survey agent, `tau` multiplies the standard deviation of the proxy score row
for that head and query block. It is a z-score multiplier, already normalized
per head and per block, which makes it less arbitrary than a raw threshold
would be. Higher `tau` keeps fewer blocks exact.

The quantity that decides whether the pooled term is a good stand-in is how
similar the 64 tokens in a block are to each other. That is the entire hinge.

## Why Morton can only act through block membership

Attention is permutation-equivariant: permute the queries and you permute the
outputs the same way, and permute the keys and values together and the output
does not change at all. Every other operation in a transformer block is
per-token. Sol-Attn's Morton permutes the video span of the hidden states,
permutes the matching rows of the rope table so positions travel with their
tokens, and applies the inverse permutation after the last block.

So under dense attention, Morton is exactly neutral. The node's own tooltip
says this, and the argument above is why it is true rather than approximately
true.

Under block-sparse attention it is not neutral, for exactly one reason: the
blocks are cut at fixed 64-token boundaries in the permuted order, so the
permutation decides who shares a block. Everything Morton does, good or bad,
flows through that.

This is a useful fact for experiment design. It means most of the question
needs no render. It also means any observed Morton effect that cannot be
traced to block membership is a trajectory difference, not a Morton effect.

## How it works in the code

The implementation lives in `vendor/sol_attn_minimax.py`, kept byte-identical
to upstream. Four pieces:

`morton_perm(grid, device, curve)` builds the permutation and its inverse for
a grid of `(latent_t, h//32, w//32)`, and caches them. Two curves:

- `3d` interleaves t, h and w equally, producing roughly 4x4x4 bricks.
- `2d_frame` puts the frame index in the high bits so frames never mix, and
  Z-orders within each frame. This is the default for H3 and the reason is
  specific: H3's `FRAME_PER_TOKEN` is `(1, 4, 4, 4, 4)`, so index-adjacent
  latent frames are either 1 or 4 real frames apart. A 3D curve groups
  temporally distant tokens as if they were neighbors.

`_perm_for(grid, curve, device, start)` rotates the permutation by
`(-start) % 64`. This exists because the kernel counts blocks from absolute
row 0 of the packed sequence, not from the start of the video span. Reference
rows push the video span off a 64 boundary, and without the rotation every
Z-order cell would split across two blocks.

`install_h3_morton(model)` wraps the model's `_forward` and `rope_freqs`, and
registers hooks on the transformer blocks. The first block's pre-hook permutes
the hidden states and the rope table together, later blocks reuse the table,
and the last block's post-hook applies the inverse. The comment in the source
is worth repeating: doing this as one decision with both tensors in hand is
deliberate, because splitting it across `rope_freqs` and the hook is what
corrupts output when the two guards disagree.

The same hooks also publish the video and audio segment boundaries that the
conditioning sink uses, so they get installed whenever `sink_conditioning` is
on, even with Morton off.

`bench/analyze_morton.py` measures all of this. It imports the shipped
`morton_perm` rather than reimplementing it, and cross-checks it against an
independently written implementation before printing any number.

## What we measured

All figures at 294 frames, curve `2d_frame` unless stated. Landed in commit
`9ffe33e`.

### Morton produces whole tiles on 3 of 48 canvases

A Morton code tiles a padded power-of-two space. When the latent grid is not a
multiple of 8 in both dimensions, the 8x8 cells get clipped at the edges, and
the surviving fragments do not line up with 64-token block boundaries. The
clean canvases are the ones where `h//32` and `w//32` are both multiples of 8:

| canvas | latent h x w | tokens/frame | blocks/frame | verdict |
|---|---|---|---|---|
| 1280x768 | 24 x 40 | 960 | 15.00 | whole 8x8 tiles |
| 1024x768 | 24 x 32 | 768 | 12.00 | whole 8x8 tiles |
| 768x768 | 24 x 24 | 576 | 9.00 | whole 8x8 tiles |
| 1344x768 | 24 x 42 | 1008 | 15.75 | ragged |
| 1152x768 | 24 x 36 | 864 | 13.50 | ragged |
| 960x544 | 17 x 30 | 510 | 7.97 | ragged |

Each has a portrait mirror at identical cost, since token count per frame is
symmetric in the two dimensions.

Block geometry at 294 frames, where radius is the RMS distance of a block's
tokens from their own spatial centroid, in latent cells, and fill is 64
divided by the bounding box volume:

| canvas | ordering | radius | fill | neighbors kept in block |
|---|---|---|---|---|
| 1024x768 | raster | 9.25 | 1.00 | 76.2% |
| 1024x768 | morton 2d_frame | 3.24 | 1.00 | 90.8% |
| 1344x768 | raster | 12.12 | 0.60 | 66.7% |
| 1344x768 | morton 2d_frame | 5.54 | 0.60 | 86.5% |
| 1344x768 | morton 3d | 1.66 | 0.98 | 77.2% |

Reading it: on 1024x768, Morton does exactly what the tooltip claims. On
1344x768 it improves the average and leaves the shape ragged. The picture is
clearer than the table. Each character is one latent cell, and cells sharing a
character share a 64-token block:

1024x768, one latent frame, first 9 of 24 rows, trimmed to 26 of 32 columns:

```
  ooooooooppppppppsssssssstt
  ooooooooppppppppsssssssstt
  ooooooooppppppppsssssssstt
  ooooooooppppppppsssssssstt
  ooooooooppppppppsssssssstt
  ooooooooppppppppsssssssstt
  ooooooooppppppppsssssssstt
  ooooooooppppppppsssssssstt
  qqqqqqqqrrrrrrrruuuuuuuuvv
```

Every block is one solid 8x8 square. Now the same frame at 1344x768, first 9
of 24 rows, full 42 columns:

```
  vvvvvvvvwwwwwwwwzzzzzzzzAAAAAAAAHHHHHHHHII
  vvvvvvvvwwwwwwwwzzzzzzzzAAAAAAAAHHHHHHHHII
  vvvvvvvvwwwwwwwwzzzzzzzzAAAAAAAAHHHHHHHHII
  vvvvvvvvwwwwwwwwzzzzzzzzAAAAAAAAHHHHHHHHII
  wwwwwwwwxxxxxxxxAAAAAAAABBBBBBBBIIIIIIIIII
  wwwwwwwwxxxxxxxxAAAAAAAABBBBBBBBIIIIIIIIII
  wwwwwwwwxxxxxxxxAAAAAAAABBBBBBBBIIIIIIIIII
  wwwwwwwwxxxxxxxxAAAAAAAABBBBBBBBIIIIIIIIII
  xxxxxxxxyyyyyyyyBBBBBBBBCCCCCCCCIIIIJJJJJJ
```

No block is a square. Block `w` occupies rows 0-3 at columns 8-15 and rows 4-7
at columns 0-7, so it is two separate 8x4 halves in different parts of the
frame. Block `A` does the same one tile over. That is one pooled vector
standing in for two disconnected regions, which is the thing Morton exists to
prevent.

Reproduce with `python bench/analyze_morton.py --canvas 1344x768 --length 294
--map`.

### The tail is worse than raster, even where the mean is better

On 1344x768, 4.7% of Morton blocks have a larger radius than raster order's
single worst block: 20.3 against 16.1. So Morton is not uniformly better
there. It tightens most blocks a lot and makes a minority worse than anything
raster produces. On 1024x768 no block exceeds raster's worst.

This is the most plausible mechanism we have for "sometimes I like it,
sometimes I don't", and it is a mechanism rather than a finding. Nobody has
shown that those blocks are the ones producing a visible difference.

### The rotation for reference rows is correct

Block geometry is invariant to `video_start` once `_perm_for` rotates the
permutation, verified at seven offsets from 0 to 20,000. Without the rotation,
1024x768 with references falls from fill 1.00 to 0.42, which is worse than the
ragged canvas.

This nearly went into this document as the opposite finding. Grouping tokens
by their index within the video span rather than by their absolute row
measures a partition that exists on no graph with references, and it makes the
rotation look like the cause of the damage it prevents. The corrected grouping
is in `block_ids()` and the reason is in its docstring.

### The damage does not concentrate at one edge of the frame

An early guess was that the ragged blocks would cluster at the right edge,
since 42 is 5 tiles of 8 plus a remainder of 2. Measured, they do not. Mean
block radius on 1344x768 by the block's mean latent column:

| columns | blocks | mean radius | max radius |
|---|---|---|---|
| 0-8 | 241 | 4.32 | 5.9 |
| 8-16 | 303 | 6.55 | 18.1 |
| 16-24 | 327 | 5.99 | 20.3 |
| 24-32 | 217 | 6.52 | 17.9 |
| 32-42 | 282 | 4.23 | 8.9 |

The loose blocks are the ones straddling two tile columns, so their mean
column lands in the interior. The frame edges are the tightest region. There
is no single part of the frame to watch.

### The damage does not concentrate in time either

Under `2d_frame`, blocks covering the first latent frame have a mean radius of
4.31 against 5.55 for the clip as a whole, so the start is slightly tighter
than average rather than worse. Under `3d`, every block spans exactly 4 latent
frames, including the first.

This matters for one of the anecdotes this work started from: that the start
and the end of a clip seem to differ more than the middle, in a sample of
about three. Morton block geometry does not predict that under `2d_frame`,
which is the shipped curve. Two readings are available and this page cannot
choose between them:

- The observation is trajectory divergence. Once the first frames differ, a
  sampler carries that difference forward, which the same anecdote notes.
  Under this reading it says nothing about Morton.
- Something other than block geometry is responsible, for example the
  interaction between the sigma window and the schedule. Sol runs dense on the
  first four steps and the last one at the shipped settings, so the ends of the
  denoising schedule are not the same as the ends of the clip and should not be
  confused with them.

There is one place where the mechanism would predict a start effect, and it is
the curve nobody runs. `3d` mixes 4 latent frames per block, and H3's first
latent frame covers 1 real frame where later ones cover 4. So `3d` pools a
1-frame latent with 4-frame latents precisely at the clip start. If a start
artifact is ever reproduced, `3d` is where to look for it.

## What upstream says the payoff is, and why our arms could not find it

From the Morton docstring in the Triton pack, read in source on 2026-08-15:

> Z-ordering makes each block a roughly 4 x 4 x 4 neighbourhood, which
> concentrates the mass into fewer blocks and lets the same quality be reached
> at higher sparsity.

The payoff is stated as being at higher sparsity. At fixed `tau`, Morton can
only add work: it costs a permutation and some non-tensor-core time, which is
what this project measured when it recorded the Morton arm running at 94% GPU
utilization where every other arm hit 99%.

Every Morton arm run here has held `tau` at its shipped value and toggled
Morton alone. Under the mechanism as its author states it, that arm cannot
find the benefit. It measures the cost and none of the payoff.

The author has also said directly, on 2026-08-14, that Morton "may or may not
increase quality, that's something to test". So the speed result is settled
and the quality question is untouched.

## What we know and what we do not

Known, by measurement on this machine:

- Morton is exactly neutral for dense attention, by construction.
- It produces whole 8x8 tiles on 1280x768, 1024x768 and 768x768, and ragged
  fragments everywhere else, including the 1344x768 default.
- On 1344x768 it improves mean block compactness by roughly 2x over raster and
  makes 4.7% of blocks worse than raster's worst.
- The reference-offset rotation works and makes geometry independent of how
  many reference rows precede the video.
- Block quality does not concentrate at any one part of the frame, or at the
  start or end of the clip, under the shipped curve.

Not known:

- Whether any of this is visible. No quality comparison has survived scrutiny.
  The one finding recorded, that Morton dropped a reference feature, was judged
  from a single frame per clip, reached upstream, and failed to replicate at a
  second seed within the hour. It is retracted in commit `440eea9`, and the
  method failure is written up there.
- Whether Morton at a higher `tau` beats no Morton at a lower one at equal wall
  clock. This is the experiment the mechanism actually predicts, and it has
  never been run.
- Whether the clean canvases behave differently from the ragged ones in output,
  as opposed to in block geometry.
- What Morton does to the routing decision itself. Block membership is
  measured; which blocks the router then selects is not.
- How a fixed curve compares to the content-based clustering that SVG2 uses.
- What the Sol-Attn paper says about any of this.

## Tests run and not run

Run:

- `bench/analyze_morton.py`, all figures on this page. No GPU, no model, about
  a second. Cross-checked against an independent implementation of the
  permutation on both curves before printing.
- The full check suite, 17 scripts, green at commit `9ffe33e`.
- `bench/smoke_h3.py` on `h3_probe_sol_on_api.json`, green, with all three
  chain lines and the CUDA kernel tag confirmed in the log.

Not run, in the order they would answer the most:

- A `tau` by Morton grid at equal wall clock. Nothing blocks it but GPU time.
- The routing simulation. Capture real q and k from one forward, then compute
  the selected-block masks offline for both orderings. Because Morton is
  exactly a permutation of the same tensors at the first transformer block,
  one capture gives exact masks for both arms without a second render.
  `h3_capture.py` exists for this and has never been run.
- A clean canvas against a ragged canvas at matched settings.
- `morton_curve="3d"` on a long clip. Never rendered.
- Reading arXiv 2607.24027.

Prior commits for context: `3b86b21` records the Morton observation that was
sent upstream, `440eea9` retracts it, and `bd392c2` adds the Sol-enabled probe
graphs the reference arms use.

## What to try next, in order

1. Run the `tau` by Morton grid. Pick a Morton-on `tau` that matches Morton-off
   at the shipped `tau` on wall clock, so the comparison is quality at equal
   speed rather than quality at equal `tau`.
2. Run the routing simulation. It converts "block membership changes" into
   "this fraction of routed blocks changes, in these places", which is a number
   rather than a judgment.
3. Compare 1280x768 against 1344x768 at matched settings. 1280x768 is 5:3,
   costs 0.95x the tokens of 16:9, and is the only near-16:9 canvas where
   Morton produces the tiles it is supposed to. If Morton matters at all, the
   gap between those two canvases is where it should be largest.
4. If any of the above shows an effect, the interesting follow-up is not more
   Morton. It is whether a better layout exists. The clean-canvas result says
   the tile shape is a lever, and the SVG2 comparison says a content-based
   layout is the version of this idea that other people found worth computing
   per head. A cheap middle option nobody appears to have tried: pad the latent
   grid to a multiple of 8 so the curve is whole on every canvas, and see
   whether the ragged canvases catch up to the clean ones.
