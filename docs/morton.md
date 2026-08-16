# Morton ordering in Sol-Attn: what it does, and what we actually know

Last updated: 2026-08-15. Line numbers are valid at commit `7e5ba88`.

## Where everything is

Grouped by what you would open it for. Paths are repo-relative except the two
sibling node packs, which live beside this repo under ComfyUI's `custom_nodes/`.

### The implementation, none of which is ours

| file | what it is |
|---|---|
| `vendor/sol_attn_minimax.py` | The CUDA Sol-Attn node, kept byte-identical to upstream. All the Morton machinery: `morton_perm` (`:150-188`), the block-alignment rotation `_perm_for` (`:205-224`), the video-span resolver `_video_span` (`:227-245`), and the install plus hooks `install_h3_morton` (`:276-396`). The node's own `morton` and `morton_curve` inputs and their tooltips are at `:746-753`. |
| `ComfyUI-SolAttn_triton/_morton.py` | The Triton pack's Morton, for Wan. **Its docstring (`:1-11`) is the load-bearing quote on this page**: it says Z-ordering "lets the same quality be reached at higher sparsity", which is the axis nothing here has tested. |
| `ComfyUI-SolAttn_triton/_morton_h3.py` | The H3 variant of the same, reordering only the video span rather than the whole packed sequence. The CUDA node's copy is inlined from these two. |
| `ComfyUI-SolAttn-cuda/` | A two-line loader shim over `vendor/sol_attn_minimax.py`, plus a README on why the node id is provisional and why it is not vendored into this repo. |

### What we built to look at it

| file | what it does |
|---|---|
| `bench/analyze_morton.py` | Every number and ASCII map on this page. No GPU, no model, about a second. Imports the shipped `morton_perm` and cross-checks it against an independently written implementation (`:115-140`) before printing. **`block_ids` (`:152-166`) is the one to read if you touch this**: grouping by `j // 64` instead of `(video_start + j) // 64` measures a partition no reference graph has, and that mistake reverses the conclusion about `_perm_for`. |
| `bench/gen_morton_figures.py` | The SVG block maps for the shareable version of this page. Same permutation, drawn instead of printed; captions derived from the geometry rather than typed. |
| `h3_capture.py` | Captures real q/k/v from a live forward. Written for a different question and **never run**. It is the missing input to the routing simulation in "Tests run and not run". |

### Where the settings live

| file | what it holds |
|---|---|
| `workflows/h3_config.py` | `SOL_RECOMMENDED_CUDA` is the shipped Sol config, including `morton=False` and `morton_curve="2d_frame"`, each with its evidence in a comment above. Nothing in this repo may hold a second copy of these. |
| `docs/SOLATTN.md` | Everything else about Sol-Attn: the two backends, the sigma window, the reference-load tables, and a "do not rely on" list of its own retracted numbers. Morton is one knob there; this page is the deep dive. |
| `docs/h3_resolutions.md` | All 95 legal canvases. The source for the 3-of-48 count below. |

### The prior attempt, and why it is worth reading before starting a new one

| reference | what happened |
|---|---|
| commit `3b86b21` | Records a Morton observation, hedged as n=1, and sends it upstream. |
| commit `440eea9` | Retracts it the same day. It did not replicate at a second seed. The commit message is the method postmortem, and it is short. |
| `internal/postmortems/2026-08-14_span_tau-and-morton.md` | The long version, gitignored. Section 4 is the useful part: nothing in the repo checks whether a quality judgement used an appropriate instrument, and that is what failed. |
| commit `9ffe33e` | Adds the analyser and the geometry results on this page. |
| commit `7e5ba88` | Scopes this page's absence claims and cuts two mechanisms it had invented to explain an unestablished effect. |

### Not read

arXiv 2607.24027, the Sol-Attn paper. It postdates the assistant's training
data and nobody here has opened it. It is the most likely place for several
"not known" rows below to already be answered.

## The short version

We have no quality result. We also did not find one anywhere else, and that
second statement is much weaker than it sounds: it describes what we searched,
not what exists. See "What counts as evidence here" below before quoting any
absence claim on this page.

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
3. Every Morton arm **this project** has run held `tau` fixed, while the
   author's own docstring puts the payoff at higher sparsity. So this
   project's arms were designed in a way that could not find the effect, and
   "a different seed that saves ten seconds" is the result you should expect
   from them. This says nothing about arms anyone else has run.

If you only take one thing: the next Morton experiment should vary `tau` and
Morton together, not Morton alone.

## What counts as evidence here

This page mixes three kinds of statement and they are not interchangeable.

**Measured.** Block geometry, from `bench/analyze_morton.py`. Deterministic
arithmetic on the latent grid, cross-checked against a second implementation.
These figures are as solid as anything here gets, and they are also the least
interesting kind of claim: they describe the permutation, not the output.

**Read in source.** The kernel behaviour, the tooltip and docstring quotes, the
comparisons to other projects. Accurate transcription, but a docstring is what
the author wrote, not a measurement.

**Impressions.** The owner's reactions to rendered clips. These are n of about
three, not blind, not controlled, from part-time work on a single machine, with
no fixed viewing protocol. **They are motivation, not evidence.** They are worth
recording because they are what prompted the work. They are not worth building
a mechanism on top of, and this page did exactly that in an earlier draft: it
took "the start and end seem to differ" and went looking for geometry that
would explain it. Explaining an effect that has not been established is how the
retracted finding in `440eea9` happened, one level up.

So: where this page says nobody has done something, read "we did not find it".
Our search was a handful of local checkouts, one survey pass, and one comment
from the author. Notably absent from it is the Sol-Attn paper itself, which is
the most likely place for a quality result to already exist.

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
section may already answer questions this page treats as open. Nobody here has
read it. Every Sol-Attn claim below comes from the node source, the CUDA
kernels, or the eager reference implementation.

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
says this (`vendor/sol_attn_minimax.py:746-749`), and the argument above is why it is true rather than approximately
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

`morton_perm(grid, device, curve)` (`vendor/sol_attn_minimax.py:150-188`)
builds the permutation and its inverse for a grid of
`(latent_t, h//32, w//32)`, and caches them. The bit-interleave itself is
`part1by2` at `:171-178`; the two curves differ by one line, `:182` against
`:184`. Two curves:

- `3d` interleaves t, h and w equally, producing roughly 4x4x4 bricks.
- `2d_frame` puts the frame index in the high bits so frames never mix, and
  Z-orders within each frame. This is the default for H3 and the reason is
  specific: H3's `FRAME_PER_TOKEN` is `(1, 4, 4, 4, 4)`, so index-adjacent
  latent frames are either 1 or 4 real frames apart. A 3D curve groups
  temporally distant tokens as if they were neighbors.

`_perm_for(grid, curve, device, start)` (`:205-224`) rotates the permutation by
`(-start) % 64`. This exists because the kernel counts blocks from absolute
row 0 of the packed sequence, not from the start of the video span. Reference
rows push the video span off a 64 boundary, and without the rotation every
Z-order cell would split across two blocks.

`install_h3_morton(model)` (`:276-396`) wraps the model's `_forward` and
`rope_freqs`, and registers hooks on the transformer blocks. The permute is
`pre_hook` at `:329-372` and the inverse is `post_hook` at `:374-388`. The first block's pre-hook permutes
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

It is tempting to offer this as the explanation for Morton feeling
inconsistent. Resist that. There is no established inconsistency to explain:
the impression rests on a handful of unblinded viewings. A mechanism that
explains an unestablished effect is worse than no mechanism, because it makes
the effect feel confirmed. What the tail result licenses is narrower, and it
is enough: **if** Morton produces a visible difference on this canvas, these
are the blocks to look at first.

### The rotation for reference rows is correct

Block geometry is invariant to `video_start` once `_perm_for` rotates the
permutation, verified at seven offsets from 0 to 20,000. Without the rotation,
1024x768 with references falls from fill 1.00 to 0.42, which is worse than the
ragged canvas.

This nearly went into this document as the opposite finding. Grouping tokens
by their index within the video span rather than by their absolute row
measures a partition that exists on no graph with references, and it makes the
rotation look like the cause of the damage it prevents. The corrected grouping
is in `block_ids()` (`bench/analyze_morton.py:152-166`) and the reason is in
its docstring.

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

One impression prompted this measurement: that the start and the end of a clip
seem to differ more than the middle, across about three clips. Block geometry
does not predict that under `2d_frame`.

The honest conclusion is that there is nothing here to explain yet. An
impression at that sample size, unblinded, is consistent with no effect at all,
and the same impression notes that once the first frames differ the rest
follows, which is what any sampler does. Two candidate mechanisms were drafted
for this page and both were cut, because writing them down gave a
not-yet-established observation the shape of a finding.

What the measurement does buy is a cheap negative: **if** a start effect is
ever established, the shipped curve's block geometry is not where it comes
from, and that is one fewer place to look.

There is one place where the mechanism would predict a start effect, and it is
the curve nobody runs. `3d` mixes 4 latent frames per block, and H3's first
latent frame covers 1 real frame where later ones cover 4. So `3d` pools a
1-frame latent with 4-frame latents precisely at the clip start. If a start
artifact is ever reproduced, `3d` is where to look for it.

## What upstream says the payoff is, and why our arms could not find it

From the Morton docstring in the Triton pack, `_morton.py:1-11`, read in
source on 2026-08-15:

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
increase quality, that's something to test". That is good evidence the author
has not tested it, and no evidence at all about anyone else. Here, the speed
result is settled and the quality question is untouched.

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

- Whether any of this is visible. No quality comparison run **here** has
  survived scrutiny. The one finding recorded, that Morton dropped a reference
  feature, was judged from a single frame per clip, reached upstream, and
  failed to replicate at a second seed within the hour. It is retracted in
  commit `440eea9`, and the method failure is written up there.
- Whether anyone outside this project already has a quality result. We did not
  find one, in a search that did not include the paper. Absence of a result in
  our search is not absence of a result.
- Whether Morton at a higher `tau` beats no Morton at a lower one at equal wall
  clock. This is the experiment the mechanism predicts, and this project has
  not run it.
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

- Read arXiv 2607.24027. First because it is the cheapest thing on the list,
  and because several rows in "not known" above may already be answered there.
- A `tau` by Morton grid at equal wall clock. Nothing blocks it but GPU time.
- The routing simulation. Capture real q and k from one forward, then compute
  the selected-block masks offline for both orderings. Because Morton is
  exactly a permutation of the same tensors at the first transformer block,
  one capture gives exact masks for both arms without a second render.
  `h3_capture.py` exists for this and has never been run.
- A clean canvas against a ragged canvas at matched settings.
- `morton_curve="3d"` on a long clip. Not rendered here.

Prior commits for context: `3b86b21` records the Morton observation that was
sent upstream, `440eea9` retracts it, and `bd392c2` adds the Sol-enabled probe
graphs the reference arms use.

## What to try next, in order

1. Read the paper before running anything. An afternoon of GPU time costs more
   than a download, and the experiment below may already be in it.
2. Run the `tau` by Morton grid. Pick a Morton-on `tau` that matches Morton-off
   at the shipped `tau` on wall clock, so the comparison is quality at equal
   speed rather than quality at equal `tau`.
3. Run the routing simulation. It converts "block membership changes" into
   "this fraction of routed blocks changes, in these places", which is a number
   rather than a judgment.
4. Compare 1280x768 against 1344x768 at matched settings. 1280x768 is 5:3,
   costs 0.95x the tokens of 16:9, and is the only near-16:9 canvas where
   Morton produces the tiles it is supposed to. If Morton matters at all, the
   gap between those two canvases is where it should be largest.
5. If any of the above shows an effect, the interesting follow-up is not more
   Morton. It is whether a better layout exists. The clean-canvas result says
   the tile shape is a lever, and the SVG2 comparison says a content-based
   layout is the version of this idea that other people found worth computing
   per head. A cheap middle option nobody appears to have tried: pad the latent
   grid to a multiple of 8 so the curve is whole on every canvas, and see
   whether the ragged canvases catch up to the clean ones.
