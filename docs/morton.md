# Morton ordering in Sol-Attn: what it does, and what we actually know

Last updated: 2026-08-16. Line numbers are valid at commit `7e5ba88` and are
checked by `bench/check_doc_links.py`, which resolves every `path:line`
citation on this page and fails when one goes out of range.

**Scope.** This page owns token order: block geometry, the curves, the capture
analysis, and the assumption chain. It does not own Sol-Attn's knobs or our
measured Sol-Attn numbers -- [`docs/SOLATTN.md`](SOLATTN.md) does, and it is the
entry point. What upstream claims lives in
[`docs/sol_upstream.md`](sol_upstream.md).

> ## Read this before quoting any number here: the evidence base is one canvas
>
> **Every activation measurement on this page is 1344x768.** One capture, one
> canvas, one prompt, one step. That includes the centroid-fidelity table, the
> mass-concentration table, the routed-density figures, the curve matrix and the
> stopping rule. The geometry sweeps now cover all 48 legal canvases; nothing
> that touches real activations covers more than one.
>
> **And 1344x768 is the worst possible single choice for it**, twice over:
>
> - It is the **most expensive** canvas in the legal set -- 1008 video tokens
>   per frame, the 1.00x row in [`docs/h3_resolutions.md`](h3_resolutions.md).
>   Every renderable comparison costs the maximum, which is why so few of them
>   have been run. 1152x768 is 0.73x, 1024x768 is 0.58x, 768x768 is 0.33x. A
>   quality A/B that is unaffordable at 1.00x is affordable three times over at
>   0.58x, and `CANVAS_TIER` in `workflows/h3_config.py` makes it one edit.
> - It is **`2d_frame`'s worst canvas**, which is the whole reason this page
>   exists. So the shipped curve is being judged where it is weakest, and the
>   alternatives where they flatter best. `3d` scores 97.9% connected here and
>   51.5% at 1952x544; a page measured at the second number would tell a
>   different story about which curve to pin.
>
> **Aspect ratio, not resolution, is the lever.** The cost is
> `(W/32) x (H/32)` per frame and attention goes as its square, so 1:1 costs a
> third of 16:9 at the same frame count. Before running anything here, decide
> which canvas the answer is supposed to generalise to, and pick a cheap one to
> measure it at unless the canvas itself is the variable.
>
> ### And the capture is 124 frames, which is below Sol's token floor
>
> 37,826 tokens against the ~60k floor in
> [`docs/SOLATTN.md`](SOLATTN.md). Noted 2026-08-16. **Be precise about what
> that does and does not invalidate, because the floor is a *speed* floor** --
> it is defined as the length below which a knob's effect vanishes into a null
> that reads as "this does nothing", and `bench_e2e_h3.py` warns on it. Applying
> it to activation statistics is not automatic and must not be asserted as if it
> were.
>
> What genuinely does not transfer from 124 frames:
>
> - **Routed density**, most of all. `kcvar` is a variance over *every* block
>   centroid, and the capture has 591 blocks against roughly 1,700 at 362
>   frames. Different population, different variance, different threshold. The
>   pending density figures are the most length-sensitive numbers on this page.
> - **The conditioning share.** 530 rows is 1.4% of this sequence and about
>   0.5% at 362 frames, so the sink's weight in every block statistic differs.
>
> What does transfer, **and only for the per-frame curves**:
>
> - **`2d_frame`, `hilbert` and serpentine geometry.** latent_t 37 against 107
>   moves them by under half a point at every canvas checked. Block shape is a
>   per-frame property for these, and the frame count does not touch it.
> - **`3d` geometry does NOT transfer, and an earlier version of this box said
>   it did.** Corrected 2026-08-16. `3d` interleaves t with h and w, so its
>   blocks span four latent frames by construction and the frame count is one
>   of its inputs. At 1440x736 it reads **67.2% at latent_t 37 and 53.7% at
>   107** — a 13.5-point move on the axis that was claimed not to matter.
>
>   The claim was checked, at 1344x768, where `3d` moves 97.9% to 98.4% and the
>   test passes. So it was verified on the one canvas where the effect does not
>   show, then generalised to a curve family it was never measured on. That is
>   the same selection-effect shape as the `3d` pin two sections down, made by
>   the same person on the same day.
>
> Centroid fidelity and mass concentration sit in between and nobody has
> measured which side they fall on. Treat them as untested at length rather than
> as invalidated.

## Where everything is

Grouped by what you would open it for. Paths are repo-relative except the two
sibling node packs, which live beside this repo under ComfyUI's `custom_nodes/`.

### The implementation, none of which is ours

| file | what it is |
|---|---|
| `vendor/sol_attn_minimax.py` | The CUDA Sol-Attn node, kept byte-identical to upstream. All the Morton machinery: `morton_perm` (`:150-188`), the block-alignment rotation `_perm_for` (`:205-224`), the video-span resolver `_video_span` (`:227-245`), and the install plus hooks `install_h3_morton` (`:276-396`). The node's own `morton` and `morton_curve` inputs and their tooltips are at `:746-753`. |
| `coderef/ComfyUI-SolAttn_triton/_morton.py` | The Triton pack's Morton, **for Wan**. Its docstring (`:1-11`) carries the only stated payoff anywhere in either pack: Z-ordering "lets the same quality be reached at higher sparsity". That sentence appears exactly once in the pack, and it is in the Wan file. |
| `coderef/ComfyUI-SolAttn_triton/_morton_h3.py` | The H3 variant, reordering only the video span rather than the whole packed sequence. **Makes no quality or sparsity claim**; its docstring is purely mechanical. The CUDA node inlines from both files, and contains the word "sparsity" zero times. |
| `ComfyUI-SolAttn-cuda/` | A two-line loader shim over `vendor/sol_attn_minimax.py`, plus a README on why the node id is provisional and why it is not vendored into this repo. |

### What we built to look at it

| file | what it does |
|---|---|
| `bench/analyze_morton.py` | Every number and ASCII map on this page. No GPU, no model, about a second. Imports the shipped `morton_perm` and cross-checks it against an independently written implementation (`:115-140`) before printing. **`block_ids` (`:152-166`) is the one to read if you touch this**: grouping by `j // 64` instead of `(video_start + j) // 64` measures a partition no reference graph has, and that mistake reverses the conclusion about `_perm_for`. |
| `bench/gen_morton_figures.py` | The SVG block maps for the shareable version of this page. Same permutation, drawn instead of printed; captions derived from the geometry rather than typed. |
| `h3_capture.py` | Captures real q/k/v from a live forward. **First run 2026-08-16**, after sitting unrun since it was written; it produced the activation measurement below that settled link 5. Captures are durable at `$H3_CAPTURE_ROOT/2026-08-15_dense_124f_1344x768/`. Arm it with `H3_CAPTURE` in the environment **before** ComfyUI starts -- it is read at module import, so there is no way to arm it on a running server. |
| `bench/analyze_capture.py` | Grades a capture: per-block centroid fidelity and mass concentration, under raster and each curve. Deliberately does **not** reimplement the router -- replicating the threshold formula from `sol_attn_preprocess.cu` would put a fidelity risk between the measurement and the claim, and neither test needs it. |

### Where the settings live

| file | what it holds |
|---|---|
| `workflows/h3_config.py` | `SOL_RECOMMENDED_CUDA` is the shipped Sol config, including `morton=False` and, **since 2026-08-16, `morton_curve="3d"`** -- changed on the activation measurement below, and changing nothing today because Morton is off. Each carries its evidence in a comment above. Nothing in this repo may hold a second copy of these. |
| `docs/SOLATTN.md` | **The entry point, and the authority on everything that is not token order**: the backends, the sigma window, the reference-load tables, and the ledger of its own retracted numbers. Morton is one knob there and a verdict-plus-link; this page is the deep dive it links to. |
| `docs/h3_resolutions.md` | All 95 legal canvases. The source for the 3-of-48 count below. |
| `docs/sol_upstream.md` | What upstream says, and only that: the paper, Sol-Engine, and the other ComfyUI packs. **Every H3 profile NVLabs publishes runs no token reordering, and one of them says why**, which is the sharpest external check on everything here. Also records that Sol-Engine ships Morton for Wan, on by default, 3D only. The counter-argument to their reasoning is on this page, not that one. |

### The prior attempt, and why it is worth reading before starting a new one

| reference | what happened |
|---|---|
| commit `3b86b21` | Records a Morton observation, hedged as n=1, and sends it upstream. |
| commit `440eea9` | Retracts it the same day. It did not replicate at a second seed. The commit message is the method postmortem, and it is short. |
| `internal/postmortems/2026-08-14_span_tau-and-morton.md` | The long version, gitignored. Section 4 is the useful part: nothing in the repo checks whether a quality judgement used an appropriate instrument, and that is what failed. |
| commit `9ffe33e` | Adds the analyser and the geometry results on this page. |
| commit `7e5ba88` | Scopes this page's absence claims and cuts two mechanisms it had invented to explain an unestablished effect. |

### The paper, read 2026-08-16

arXiv 2607.24027 was fetched and read on 2026-08-16, after this page spent two
weeks listing it as the cheapest unrun item. It is summarised in
[`docs/sol_upstream.md`](sol_upstream.md), which owns it.

**It contains no token reordering at all** -- no Morton, no Z-order, no
permutation, no spatial block layout. Neither do the published Sol-Engine docs,
nor either third-party ComfyUI pack.

That does not weaken this page; it sharpens what it can say. The attribution
chain has now narrowed four times: from "upstream says the payoff is at higher
sparsity", to "that is the Wan file, not H3", to "the H3 file makes no quality
claim", to **"it is not part of the published method at all"**. Morton on H3 is
one implementer's addition, and the one sentence stating a payoff for it is
about a different model in a pack we no longer run. Read the sparsity rationale
below with that in front of it.

What the paper does settle for this page: `tau` is its `beta`, and it is never
swept, so nothing upstream adjudicates the tau-by-Morton experiment either.

## What a block is, and why the order matters

Read this first. The rest of the page uses these words as if they were
obvious, and they are only obvious once.

**A token is one 32 by 32 pixel square of one latent frame.** At 1344x768 a
frame is 42 tokens wide and 24 tokens high, so 1008 tokens per frame. The model
puts them in one list: frame by frame, then row by row, then left to right.
That order is called raster order.

**A block is 64 tokens taken in a row from that list**, and it is the unit
Sol-Attn makes a decision about. Sol-Attn gives each block one summary, the mean
of its 64 tokens, which the CUDA source calls the centroid. It uses that
centroid twice: to decide whether to compute the block exactly, and, if it does
not, as the stand-in for all 64 rows. So the centroid has to describe the block.
When it does, both uses are right. When it does not, both are wrong.

**Under raster order a block is a thin strip.** 64 tokens is one full row of 42
plus 22 of the next, so it runs the whole width of the frame. A strip can hold
sky, a face and a wall at once, and the mean of those three is none of them.

**Morton order exists to replace that strip with an 8x8 square**, since 8 times
8 is 64. A square holds one small area, and the mean of one small area is more
likely to describe it. Morton is also safe by construction: the code applies the
order before the first transformer block and removes it after the last, so under
dense attention the output is unchanged **in exact arithmetic**. Those last
three words are load-bearing and are measured below -- in floating point it is
not bit-identical, and that has consequences for how a Morton A/B reads.

That is the whole question this page answers. Does it produce the square, and
what follows when it does not.

### It produces the square on 3 of 48 canvases, and there are two reasons

Measured at 1344x768: 24.1% of blocks are one solid 8x8, 46% are two
disconnected pieces, 25% are three. At 1280x768, 1024x768 and 768x768 it is
100%.

Two separate things go wrong, and **the second one is the smaller of the two**,
which is the opposite of what this page said in its first draft:

1. **The frame does not hold a whole number of blocks.** 1008 / 64 = 15.75, so
   block boundaries and frame boundaries drift apart, and every frame starts the
   pattern somewhere new.
2. **The grid is not a multiple of 8.** 42 / 8 = 5.25, so the rightmost Z-order
   tile is 2 wide instead of 8, and the leftover shifts everything after it.

Reason 1 does most of the damage. Aligned to the frame start, 93% of blocks stay
in one piece; at the real alignment, 60% do.

At 768 height the two collapse into one test, because both latent dims divide by
8 exactly when tokens per frame divides by 64. **So the rule is: both `h/32` and
`w/32` must be multiples of 8.**

**The same rule in pixels, which is the form worth remembering.** A Morton tile
is 8x8 tokens and a token is 32x32 pixels, so a tile is a **256x256 pixel
square**. The canvas has to be a whole number of those squares in both
directions: **width and height both divisible by 256.**

> ### The canvas rule is a `2d_frame` rule. `3d` does not have this problem.
>
> **Stated unconditionally on this page until 2026-08-16, and that was wrong in
> a way that got repeated.** Everything above and below about ragged canvases
> describes `morton_curve="2d_frame"`. `SOL_RECOMMENDED_CUDA` has pinned
> **`3d`** since 2026-08-16, and `3d` is close to canvas-independent. Measured
> at 294 frames:
>
> | canvas | | `2d_frame` radius / fill | `3d` radius / fill |
> |---|---|---|---|
> | 1344x768 | ragged | 5.54 / 0.60 | **1.66 / 0.98** |
> | 1152x768 | ragged | 5.25 / 0.72 | **1.62 / 0.98** |
> | 1280x768 | clean | 3.24 / 1.00 | 1.62 / 0.98 |
> | 1024x768 | clean | 3.24 / 1.00 | 1.62 / 0.98 |
>
> `3d` lands at radius ~1.6 and fill 0.98 on every canvas tested, ragged or
> clean, and **0.0% of its blocks are looser than raster's worst** anywhere --
> against 4.7% for `2d_frame` at 1344x768. The reason is structural: a `3d`
> block is a 4x4x4 brick spanning four latent frames, so a leftover in the
> frame *width* has three other axes to absorb it. `2d_frame` never mixes
> frames, so a 64-token run has to close as an 8x8 tile inside one frame or not
> at all, and that is the constraint the divisibility rule expresses.
>
> **So "only 3 of 48 canvases work, do not judge Morton on the default" is a
> true statement about the curve we no longer ship.** On the shipped curve the
> default canvas is fine.
>
> `3d` pays for it elsewhere and the bill is not measured: **100% of its blocks
> span more than one latent frame**, which is exactly the `FRAME_PER_TOKEN`
> objection -- H3's first latent frame covers 1 real frame where later ones
> cover 4, so `3d` pools a 1-frame latent with 4-frame latents at the clip
> start. Trading a spatial problem for a temporal one is not obviously a win,
> and no render has been made with `3d` on a long clip.

Reproduce: `python bench/analyze_morton.py --canvas 1344x768 --length 294`,
which prints both curves side by side.

### The rule of thumb, per curve, swept over all 48 landscape canvases

Measured 2026-08-16, every legal landscape canvas at 294 frames. **`3d` has a
canvas rule too -- it is just a different and much weaker one, and it is about
height.**

A `3d` block is a 4x4x4 brick, so what it wants is the *token* grid divisible
by 4. Height in tokens is `h/32`, so the rule lands on **height divisible by
128 pixels**:

| height | `h/32` | `%4` | canvases | `3d` radius, min-max |
|---|---|---|---|---|
| **768** | 24 | 0 | 20 | **1.62 - 1.80** |
| **640** | 20 | 0 | 4 | **1.62 - 1.65** |
| **512** | 16 | 0 | 3 | **1.67 - 1.70** |
| 704 | 22 | 2 | 3 | 1.80 - 1.93 |
| 576 | 18 | 2 | 5 | 1.70 - 2.00 |
| 544 | 17 | 1 | 4 | 1.84 - 2.55 |
| 608 | 19 | 3 | 3 | 1.84 - 2.43 |
| 672 | 21 | 1 | 3 | 1.86 - 2.45 |
| 736 | 23 | 3 | 3 | 1.90 - 2.39 |

**Moving to `3d` widened the canvas choice by 9x; it did not remove it.** This
gets read backwards, so state it as counts: `2d_frame` is clean on **3 of the
48** landscape canvases, `3d` on **27 of 48** (20 at height 768, 4 at 640, 3 at
512). The strict rule belongs to the curve we stopped shipping. Resolution is
still a lever on `3d`, just a much looser one.

**Width barely matters on `3d`.** At height 768 the twenty canvases span 24 to
43 tokens wide and all land between 1.62 and 1.80. The odd-token heights are
where it degrades, worst case 2.55 at 1888x544 and 1952x544 with fill dropping
to 0.63.

So, as rules of thumb:

- **`3d` (shipped): use a height of 768, 640 or 512.** Pick any width. 768 is
  the most common height in the legal set -- 20 of the 48 landscape canvases --
  so most renders already satisfy this without trying.
- **`2d_frame`: both dimensions divisible by 256**, which is 768x768,
  1024x768, 1280x768 and the portrait 768x1024. Everything else is ragged, and
  ragged is the common case rather than the exception.

**"Pick any width" is the right first-order rule and it is not the whole
structure**, recorded 2026-08-17 so the two pages do not drift. The height-768
band above spans 1.62 to 1.80, and that spread is not noise: it is
`w/32 % 4`. The six canvases with **both** token axes divisible by 4 land at
1.61 with no spread at all, while the other fourteen at that height range 86.6%
to 99.0% connected. Height remains roughly four times the lever width is, so
the bullet stands as written. The per-canvas ranking, the grouping, and a third
axis this page does not test (`latent_t % 4`, which is what separates 311 and
243 frames from the rest) are in
[`h3_input_impacts.md`](h3_input_impacts.md), which owns them.

### Which rule applies to you depends on where your graph came from

**The CUDA node ships both curves, and its own default is the canvas-sensitive
one.** `io.Combo.Input("morton_curve", options=["3d", "2d_frame"],
default="2d_frame")` (`vendor/sol_attn_minimax.py:750`). Only **Sol-Engine**
ships 3D-only, and it does not point Morton at H3 at all -- see
[`docs/sol_upstream.md`](sol_upstream.md). Those two facts get conflated.

So there are two starting points and they land on different rules:

| you start from | `morton_curve` you get | rule of thumb |
|---|---|---|
| a graph from `build_workflows.py` | **`3d`**, baked in `widgets_values` | height 768 / 640 / 512, any width |
| a fresh `SolAttnMiniMax` dropped in your own graph | **`2d_frame`**, the node default | both dims divisible by 256 |

Verified rather than assumed: the four Sol-enabled probe graphs bake
`'3d'` at widget index 6, e.g. `h3_probe_sol_on.json`
`[1.3, 0.2, 0.9, 4096, 'exact_kv_and_rows', False, '3d', ...]`.

**The trap is that flipping `morton` on is one widget and choosing the curve is
another.** Turning Morton on in a hand-built graph silently selects the curve
with the strict canvas rule, on a canvas that probably does not satisfy it.
That is most of the distance between "Morton does nothing useful here" and the
geometry this page measures.

**What this is a rule about, and it is not output quality.** Every number here
is block geometry -- how compact the 64 tokens sharing a summary are. Whether
compactness reaches the screen is link 6 and is still unverified, so "optimal
Morton shape" means "the router's per-block summaries describe tighter regions"
and nothing more. Do not read the table as a quality ranking of canvases.

**And that is why it is 3 of 48 rather than something about Morton.** H3's
canvas ladder steps in 32-pixel increments; Morton needs 256, so only one rung
in eight lines up on each axis independently. Verified against the list in
`docs/h3_resolutions.md` rather than reasoned about:

- Only **two** heights in the whole legal landscape set are divisible by 256:
  512 and 768. The other seven -- 544, 576, 608, 640, 672, 704, 736 -- all miss.
- 768 is the common one: **20 of the 48** canvases sit at that height, and all
  20 pass on height. Of those, only three widths are multiples of 256: 768,
  1024 and 1280.
- The next rung, 1536x768, is 1,179,648 pixels against the 1,032,192 cap. So the
  list stops at three because **the ladder runs out of room**, not because of
  anything Morton does.

One near-miss worth knowing, because it looks like a bug if you rediscover it:
**2048x512 tiles perfectly** and is 1,048,576 pixels -- **16,384 over the cap**.
A shape the curve would handle cleanly and the model will not render.

## The short version

We have no quality result. We also did not find one anywhere else, and that
second statement is much weaker than it sounds: it describes what we searched,
not what exists. See "What counts as evidence here" below before quoting any
absence claim on this page.

What this page does establish, by measurement rather than by reading:

1. Morton cannot change what dense attention *computes*. It is a permutation
   that gets undone, and attention is permutation-equivariant. The only thing it
   can change is which tokens share a 64-token block, which is what the sparse
   router summarizes. That makes most of the question pure arithmetic on the
   latent grid, answerable with no GPU and no watching of clips. **It is not
   bit-identical, though**, and a dense Morton render will still diverge from a
   dense non-Morton one -- see "Neutral in exact arithmetic, not in floating
   point" below before reading any A/B as a Morton effect.
2. Under `2d_frame`, Morton delivers the compact tiles it promises on 3 of
   the 48 legal landscape canvases. 1344x768 -- the canvas this repo's t2v
   graphs happen to bake, `CANVAS_TIERS["full"]`, and one option among 95 --
   is not one of them. On it a block is typically two disconnected fragments in
   different parts of the frame. **Under `3d`, the shipped curve, this does not
   apply**: see the callout under the canvas rule.
3. Every Morton arm **this project** has run held `tau` fixed, while the one
   place upstream states a payoff puts it at higher sparsity. So this
   project's arms were designed in a way that could not find the effect, and
   "a different seed that saves ten seconds" is the result you should expect
   from them. This says nothing about arms anyone else has run. Read the next
   section carefully before repeating the attribution: that statement is in
   the **Wan** Morton file, and the H3 path makes no quality claim at all.

If you only take one thing: the next Morton experiment should vary `tau` and
Morton together, not Morton alone.

## The one assumption everything downstream rests on

Stated plainly, because the rest of this page is more persuasive than it has
earned. Six links, and only the first four are verified.

| # | link | status |
|---|---|---|
| 1 | Sol cuts the sequence into 64-token blocks, counted from index 0 | verified, `coderef/comfy-kitchen-sol/comfy_kitchen/backends/cuda/ops/sol_attn_route.cu:18`, `:288`, `:435` |
| 2 | Routing and the pooled tail are centroid quantities | verified, `coderef/comfy-kitchen-sol/comfy_kitchen/backends/cuda/ops/sol_attn_route.cu:20-21`: "Both the routing decision and the tail VALUES are centroid quantities" |
| 3 | Morton changes which tokens share a block | verified, node source and `bench/analyze_morton.py` |
| 4 | The partition computed here is the partition the kernel uses | deterministic given grid and `video_start`; not an estimate, not a sample |
| 5 | **A fragmented block's centroid represents its members worse than a compact block's** | **MEASURED 2026-08-15 on captured activations. True.** See below |
| 6 | Therefore Morton's canvas dependence matters for output | still rests on 5 holding at a size that shows; untested |

**Link 5 was a story until 2026-08-15. It is now measured, and it is true.**
See "What the captured activations say" below. The short version: Morton does
raise centroid fidelity on real q/k, at every depth sampled. The reasoning was
that neighbouring latents are correlated, so a spatially compact block has a
mean that stands in better for its members, and that is what the numbers show.

What is left is link 6. A real improvement in centroid fidelity is not the same
as a visible improvement in output, and the shipped curve's improvement is
small: +0.6% to +3.0% depending on depth.

There was a specific reason to doubt link 5 as a universal rather than just to
flag it as unproven, and it is worth keeping now that the measurement has gone
the other way. Spatial compactness is a **proxy** for feature similarity, and
proxies fail at the interesting places: a tight 8x8 tile straddling a hard
object boundary may have a worse centroid than a scattered block sitting
entirely inside uniform sky. That objection predicted the tail result below --
`2d_frame` improves the mean while making a minority of blocks worse than
raster's worst -- so it was right about the shape and wrong about the average.

**The test that settled it** was cheap, exactly as predicted: capture `k` from
one forward, then for each block compute the mean cosine of its members to their
own centroid, under Morton and under raster. One number per ordering, no renders,
no clips. It ran on 2026-08-16 and it supported link 5 rather than killing it.
See "What the captured activations say".

The claim that no longer needs hedging is therefore the centroid one. The claim
that still does is link 6: on 1344x768, 24.1% of blocks are a solid 8x8 and the
rest are two or three disconnected fragments; on 1280x768 and 1024x768 it is
100%. That the tighter grouping summarises better is now measured. Whether it
reaches the screen is open.

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

The limit, and it narrowed on 2026-08-16: the Sol-Attn paper was read that day
at the depth recorded in [`docs/sol_upstream.md`](sol_upstream.md) -- abstract,
ablation summary and HTML, not the full PDF. It contains no token reordering and
no related work on space-filling curves, so it does not answer the questions
this section raises. Everything below still comes from the node source, the CUDA
kernels, or the eager reference implementation, and that is now a choice rather
than a gap.

## How Sol-Attn uses blocks

The routing mechanism, `tau`, and the sink belong to
[`docs/SOLATTN.md`](SOLATTN.md); this page does not restate them. One sentence
is needed here because everything below hinges on it:

**Sol-Attn gives each 64-token block a single pooled summary, and the quantity
that decides whether that summary is a good stand-in is how similar the 64
tokens in a block are to each other.** That is the entire hinge, and token
order is what decides which 64 tokens those are.

## Why Morton can only act through block membership

Attention is permutation-equivariant: permute the queries and you permute the
outputs the same way, and permute the keys and values together and the output
does not change at all. Every other operation in a transformer block is
per-token. Sol-Attn's Morton permutes the video span of the hidden states,
permutes the matching rows of the rope table so positions travel with their
tokens, and applies the inverse permutation after the last block.

So under dense attention, Morton is neutral in exact arithmetic. The node's own
tooltip says this (`vendor/sol_attn_minimax.py:746-749`), and the argument above
is why it is true as mathematics rather than approximately true.

Under block-sparse attention it is not neutral, for exactly one reason: the
blocks are cut at fixed 64-token boundaries in the permuted order, so the
permutation decides who shares a block. Everything Morton does, good or bad,
flows through that.

### Holding `tau` fixed does not hold sparsity fixed

**Read from the kernel source 2026-08-16, and it undercuts an argument this page
makes three times.** The routing threshold is not a constant that `tau` scales.
It is derived from the block partition, so **Morton moves the threshold itself.**

The chain, in
`coderef/comfy-kitchen-sol/comfy_kitchen/backends/cuda/ops/sol_attn_preprocess.cu`:

1. `kcvar[d]` is the variance **across the NTB block centroids** of dimension
   `d` -- `prep_pooled_stats` means over blocks, then sums `(kc - mean)^2` over
   blocks (`:107-123`).
2. The threshold for a query block is
   `tau * sqrt(sum_d c_d^2 * kcvar_d * log2s^2)`, where `c` is the query
   centroid (`:192-199`).

Block membership determines `kc`. Morton changes membership. So `kcvar` changes,
so the threshold changes, so **the number of blocks routed exact changes at a
fixed `tau`.**

**Direction is not derivable, and this page asserted for a day that it was.**
Corrected 2026-08-16. What stood here argued that coherent blocks push centroids
away from the global mean, so `kcvar` rises, so the threshold rises, so fewer
blocks clear it, so Morton at fixed `tau` is **more** approximate. That
reasoning moves the threshold and holds the scores fixed, and both move.

**Read from source, not measured.** The routing test is `colmean > thr`
(`coderef/comfy-kitchen-sol/comfy_kitchen/backends/eager/sol_attn.py:112-142`).
`thr` is the quantity above. `colmean` is the query block's mean score against
the **mean-centred pooled key centroids** -- and a more coherent key block has
less internal cancellation in its pooled centroid, so its score against a
matching query rises too. Numerator and denominator both climb with coherence.
Nothing in the formula says which climbs faster, so **the sign of the routed
density change is an empirical question per curve and per depth, not something
this page can argue.** Any future version of the old paragraph is a regression.

That is also why the answer is not a single scalar: it can differ in sign
between two curves at the same depth, and between two depths for one curve.

### Measured. The paragraph above is confirmed, not replaced

**`bench/analyze_routing.py`, landed 2026-08-16 (`51527c3`).** Ordering-effect
density -- forced-exact pairs dropped from numerator and denominator -- against
raster, on the capture, `tau` 1.3, 8 of 56 heads:

| depth | raster | `2d_frame` | `3d` | `hilbert` |
|---|---|---|---|---|
| block 24 | 14.40% | 1.157x | 1.150x | **1.171x** |
| block 49 | 13.49% | 1.087x | **0.987x** | 1.113x |

Compensating `tau` -- the value that returns each curve to raster's density --
at block 49: `2d_frame` 1.367, `3d` 1.290, `hilbert` 1.387.

**Block 49 is the counterexample, and it is why the section above says "not
derivable" rather than "the sign is reversed".** At one depth, under one
ordering `3d` routes *fewer* blocks than raster while `hilbert` routes *more*.
Opposite signs at the same depth kills any monotone coherence-to-density story
in either direction -- including the corrected one. And `3d` has the best
centroid fidelity at that depth (0.9434), so the ordering with the most coherent
blocks is the one that routes fewest. Coherence does not predict density.

> **The instance is capture-specific; the conclusion got stronger. Measured
> 2026-08-16 on the 362-frame 1024x768 three-reference capture.** "3d below 1.0
> at block 49" does **not** reproduce there. What happens instead is worse for
> any simple story: `3d` at block 49 runs 0.994x, 1.029x, 1.130x across steps
> 3 / 8 / 14, **crossing 1.0 within a single depth as a function of sigma**,
> and the sign flip has relocated to block 0, where every curve routes fewer at
> early steps and crosses above 1.0 by step 14.
>
> So quote the block-49 row against the 124-frame t2v capture it came from, and
> nothing else. **The sign is a joint function of depth and sigma, of neither
> alone, and of block coherence not at all.**
>
> **This is what makes a controlled ordering A/B unbuildable**, which is the
> load-bearing consequence. Holding routed density fixed across an ordering
> change needs a per-`(block, sigma)` `tau`; `tau_profile` is keyed per
> transformer block and has no sigma axis, and a per-block scalar cannot carry
> a term that moves 0.136 across the schedule at block 49 while moving 0.022 at
> block 24. A tau-matched pair varies two things; a density-matched pair needs
> a knob that does not exist. The only runnable comparison is **matched wall
> clock** -- set the morton arm's `tau` so both arms cost the same and judge
> quality, which is the arm proposed further down this page. It answers a real
> question and it is not a mechanism experiment; do not report it as one.

**Two densities, and they answer different questions.** The table above is
ordering effect. `analyze_routing.py` also emits kernel density, which counts
the forced-exact pairs the kernel routes regardless, and that is the number for
anything about cost or for sizing `routed_cap_percent`. Do not substitute one
for the other; an uncontrolled prototype reported the first while labelled as
the second, which is the mistake this split exists to prevent.

> **Does not transfer to 362 frames.** `kcvar` is a variance over the whole
> pooled population and the capture has **591 blocks against roughly 1,700** at
> length, so the threshold is derived from a different population. The script
> prints this warning itself below 60k tokens. Also still the float routing
> rule -- the kernel INT8-quantises pooled keys and query centroids and this
> does not -- 8 of 56 heads, three depths, one capture, one step, one prompt,
> one canvas.

Two consequences, and the first is a correction:

- **"Morton at fixed tau measures the cost and none of the payoff" is too
  clean.** This page says a version of that three times. If the threshold moves,
  a fixed-`tau` Morton arm is not the same operating point with a permutation
  added -- it is already somewhere else on the speed-quality curve, in an
  unknown direction and by an unknown amount. Every past Morton A/B here
  varied two things while believing it varied one.
- **It is cheap to settle and needs no render.** The captures at blocks 0, 24
  and 49 are on disk. Counting how many key blocks clear
  `tau * sqrt(sum_d c_d^2 * kcvar_d)` under raster and under each curve is
  arithmetic on tensors we already have. That converts "membership changes"
  into "routed density changes by X% at fixed tau", which is a number.
  **Score it against the eager reference, not a fresh implementation of the
  formula** -- `backends/eager/sol_attn.py` is upstream's own statement of the
  algorithm ("Defines the algorithm, not the CUDA kernel's arithmetic", its
  docstring) and using it keeps a reimplementation risk out from between the
  measurement and the claim. **Done 2026-08-16** -- `bench/analyze_routing.py`
  imports `_pool` from that module rather than transcribing it, and
  cross-checks its own threshold against a deliberately naive transcription in
  the same file. Numbers above.

**The axis this does not cover, and it may matter more than depth.** Routed
density is measured at whatever step the capture was taken at -- step 1 of 6 for
the capture on disk. Sol runs 11 of 16 steps at the shipped sigma window, and
nothing says the divergence is constant across them. This is load-bearing for
any compensation scheme: `tau_profile` is keyed **per transformer block**
(`parse_tau_profile` -> `{block: tau}`, read from
`transformer_options["sol_block"]`, `vendor/sol_attn_minimax.py:55-76`,
`:524-529`), so it can express a depth-dependent correction and **cannot express
a sigma-dependent one at all.** Measure a late-step capture before building
anything on top of the depth numbers.

**One precision, because the short form of this claim gets repeated.** "Token
order is the only thing that changes which 64 tokens share a block" is true of
*Morton*, and not quite true in general: the blocks are cut every 64 rows from
absolute row 0 of the packed sequence, so **where the video span starts moves
tokens between blocks as well**. That is not a reordering, it is an offset, and
it is exactly what the `(-video_start) % 64` rotation in `_perm_for` exists to
cancel. So the full statement is that block membership is a function of the
permutation *and* `video_start mod 64`, and Morton controls the first while
`_perm_for` neutralises the second. See "Morton still works with references".

### Neutral in exact arithmetic, not in floating point

Measured 2026-08-16, and it changes how to read every Morton A/B on this page.

Floating-point addition is not associative, so permuting the keys changes the
order the softmax denominator and the value-weighted sum accumulate in.
PyTorch SDPA, bf16, `T=4096`, permute q/k/v together and apply the inverse:

```
bitwise identical:      False
elements differing:     44%
max abs diff:           9.77e-04
cosine:                 0.9999964
```

Sage is a different kernel, but nothing about the argument is kernel-specific --
any implementation that reduces over keys in a different order lands somewhere
different in the last bits.

Two consequences, and the second is the useful one:

- **"Exactly neutral" is a statement about mathematics, not about output.** A
  dense Morton render and a dense non-Morton render at the same seed are not the
  same file.
- **There is a non-Morton explanation for "it feels like a different seed".**
  Over 16 steps of a flow-matching ODE a 1e-3 perturbation at step 0 is
  amplified, and `er_sde` injects noise every step on top. So the reseed-like
  impression the owner reported is exactly what a bit-level perturbation
  produces, with no block-membership effect required. That does not mean Morton
  has no effect on output; it means **this class of observation cannot
  distinguish one from the other**, and a Morton A/B judged by eye is measuring
  both at once.

The cheapest available check on real output: the ordering sweep below rendered
both dense arms (`2d_frame` at 861.6 s and Morton off at 860.8 s). Nobody has
compared those two clips' pixels, and doing so costs no GPU.

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

### Most of the damage is misalignment, not edge clipping

Worth separating, because this page attributed it to clipping first and that is
the smaller effect. Two things go wrong on a grid that is not a multiple of 8:
the Z-order cells get clipped at the frame edge, and 1008 tokens per frame is
not a multiple of 64 so blocks slide relative to the frame. Measured at
1344x768, counting blocks that form a single connected region:

| block alignment | connected | mean radius |
|---|---|---|
| aligned to the frame start (hypothetical) | 93% | 3.41 |
| real, cut every 64 rows from row 0 | 60% | 4.89 |

Frame-aligned Morton on the ragged canvas is nearly as good as on a clean one.
The sliding is what costs most of it. At 768 height the two conditions are
mathematically equivalent -- both latent dims divisible by 8 holds exactly when
tokens per frame is divisible by 64 -- so the canvas rule is unchanged. The
mechanism behind it is not what was written.

### A Hilbert curve fixes most of it, and looks like a drop-in

**It is a `2d_frame`-class curve, and that is the first thing to know about
it.** `sol_curves.py:36`: "2D Hilbert within each latent frame, frames left in
original order". So it never mixes frames, exactly like `2d_frame`, and it is a
**replacement for `2d_frame` rather than a competitor to `3d`**. The three
curves are two families:

| family | curves | block shape | canvas-sensitive |
|---|---|---|---|
| per-frame 2D | `2d_frame`, `hilbert` | 8x8 tile inside one frame | `2d_frame` yes, `hilbert` much less |
| 3D | `3d` | 4x4x4 brick over 4 latent frames | barely |

That matters when reading the capture table above, which lists all three side
by side without saying so: `3d` buys its scores by mixing frames, and the other
two decline to. Comparing `hilbert` against `3d` is comparing across that
choice, not within it.

Z-order's weakness is that it jumps: consecutive points on the curve are often
far apart in space, because the curve crosses quadrant boundaries. Measured, at
side 64 that is 2047 of 4095 consecutive steps. A Hilbert curve never jumps;
adjacency of consecutive points is its defining property, verified here at
sides 8, 16 and 64 before anything was measured with it.

**"Never jumps" is a property of the square, and the square is not what runs.**
Corrected 2026-08-16, and it is the difference between the check that exists and
the check that would have caught something. `hilbert_perm` computes on the next
power of two and drops the out-of-range points, so a 24x42 grid is a rectangle
clipped out of 64x64. Dropping points splices the curve across each gap.
Measured on the shipped path at 24x42: **6 non-adjacent steps of 1007 within a
frame**, against 0 of 4095 on the 64x64 square that `verify_adjacency` actually
tests. Still two orders of magnitude better than Z-order's ~50%, and the
argument below survives -- but the defining property does not hold on any canvas
this repo renders, and the check cannot see that. See `docs/checks.md`.

That matters exactly where Morton is failing. A run of 64 consecutive points
along a curve with few jumps is very nearly a connected region whatever the grid
shape, so it does not need the grid to factor.

Measured at real block alignment, **over single-frame blocks only** -- read the
restriction below before quoting any of it:

| canvas | ordering | blocks connected | mean radius |
|---|---|---|---|
| 1344x768 | z-order | 60% | 4.89 |
| 1344x768 | **hilbert** | **90%** | **4.49** |
| 1344x768 | raster | 100% | 11.90 |
| 1280x768 | z-order | 100% | 3.24 |
| 1280x768 | hilbert | 100% | 3.24 |

Raster is 100% connected because a run along a row is trivially connected,
which is why radius has to be read alongside it.

> ### The restriction this table did not state, and what it hides
>
> Added 2026-08-16 after the numbers were re-derived independently. They
> reproduce exactly -- 59.9% / 90.0% connected, 4.88 / 4.49 / 11.90 radius -- but
> **only when blocks spanning more than one latent frame are excluded.** Over
> every full block the same measurement gives:
>
> | ordering | connected, single-frame blocks | connected, all blocks |
> |---|---|---|
> | z-order `2d_frame` | 59.9% | 57.1% |
> | `hilbert` | 90.0% | 85.8% |
>
> Two consequences, and the second is the one that bites:
>
> - The excluded blocks are **exactly the frame-straddling ones**, ~4.7% at this
>   grid. So the headline metric behind this whole section is structurally blind
>   to frame sliding -- the failure mode the section names as the larger of the
>   two. A curve change aimed at sliding cannot show up here at all.
> - **The metric is undefined for `3d`.** Every `3d` block spans four latent
>   frames by construction, so the single-frame population is empty and the
>   computation divides by zero. This table therefore cannot rank `3d` against
>   `hilbert` even in principle, and must never be read as if it does.
>
> **No committed script computes connectivity.** `bench/analyze_morton.py` has no
> notion of it; it reports radius, fill and neighbour retention. The 60%/90%
> figures -- quoted in `sol_curves.py`'s docstring and in the node's UI tooltip --
> are the only numbers on this page with no instrument behind them. They are
> sound; they are just not reproducible from this repo, and that is a gap in
> `docs/checks.md`, not a reason to distrust them.

So Hilbert recovers most of the gap on the awkward canvases and changes nothing
on the clean ones. As a change it is small: same interface, a permutation and
its inverse, cached once, and the same start-offset rotation applies unchanged.
`morton_curve` already has two options and this would be a third.

### Frame sliding: "no curve fixes it" was wrong

**Corrected 2026-08-16.** This section said the residue after Hilbert is the
frame-sliding above, "which no curve fixes". A per-frame *phase* change fixes
most of it, and the reason the claim survived is that the metric above cannot
see the blocks it acts on.

The distinction that makes it work, because the obvious version of the idea is
false: **no permutation can put a block boundary on a frame boundary.** Blocks
are cut at absolute row multiples of 64, 1008 % 64 = 48, and the straddling-block
fraction is therefore identical for every ordering -- measured, 6.2% at
(37,24,42), the same for raster, `2d_frame`, `3d` and `hilbert`. What a
per-frame rotation by `(frame_index * area) % 64` changes is the **cut phase**:
every frame gets sliced at the same offset along its own curve, so every frame's
blocks take the same shape. That is the "aligned to the frame start
(hypothetical) -- 93%" row of the table above, made real without padding.

Measured with `bench/analyze_morton.py`'s own `block_stats`, all blocks, grid
(37,24,42), `video_start` 530:

| ordering | connected | radius | fill | nbr |
|---|---|---|---|---|
| raster | 95.4% | 12.09 | 0.62 | 44.4% |
| `2d_frame` | 58.1% | 5.49 | 0.61 | 57.6% |
| `3d` | 97.9% | 1.68 | 0.99 | 76.6% |
| `hilbert` (shipped) | 86.1% | 5.02 | 0.71 | 58.2% |
| `hilbert` + per-frame rotation | 93.8% | 4.10 | 0.89 | 59.6% |
| `hilbert` + serpentine | 93.8% | 4.09 | 0.84 | 59.4% |

**Two things stop this from being a free win, and both are why it is not
shipped.**

> **The first version of this sweep used four canvases H3 cannot render.**
> Corrected 2026-08-16, hours after it was committed. 1152x640, 1024x576,
> 832x480 and 1216x704 are not in `adapt_canvas()`'s output and appear nowhere
> in `docs/h3_resolutions.md`; they were picked from general video-model habit
> and asserted to be shipped without checking the one page in this repo whose
> whole job is to answer that. One of them carried a conclusion
> ("`3d` collapses at 832x480, and 832x480 is a shipped one") that was wrong
> twice over. **Take canvases from `docs/h3_resolutions.md`, always** -- the
> legal set is 48 landscape/square resolutions plus their portrait mirrors, and
> it does not contain the obvious ones.

*It is not universal.* Swept over **all 48 legal landscape/square canvases**
(portrait mirrors are identical -- tokens per frame is `(W/32)x(H/32)`, which is
symmetric), `video_start` 530, all blocks, **at latent_t 107 (362 frames, the
shipped length)**:

| | min | mean | max |
|---|---|---|---|
| `hilbert` (shipped) | 77.1% | 85.8% | 100% |
| + per-frame rotation | **74.3%** | 90.5% | 100% |
| + serpentine | 83.8% | **92.4%** | 100% |
| `3d` | **51.5%** | 87.1% | 100% |

**Quote this table at the shipped length, not the capture's.** The per-frame
rows are the same to within half a point at latent_t 37, but `3d`'s floor is
67.2% there and 51.5% here — it mixes frames, so frame count is one of its
inputs. The first version of this table was taken at the capture's 124 frames
and understated `3d`'s spread by 15 points.

- **Serpentine is never worse than plain `hilbert`. 0 of 48.**
- **Rotation is worse on 5 of 48**: 1536x672, 1440x736, 1408x736, 1376x736,
  896x768. It also has the worst floor of any ordering here.
- Neither dominates the other: serpentine wins 29, rotation wins 12, 7 tie.

**Only 3 of the 48 have tokens per frame divisible by 64**, so the phase problem
this fixes is the normal case, not a corner case. 1344x768 is not special for
having it; it is special only for being the canvas everything here was measured
on.

*Rotation is the wrong form of the idea.* **A Hilbert curve is an open path, not
a cycle** -- at side 64 it runs (0,0) to (63,0), Manhattan distance 63. Rotating
the start splices those two ends together, putting one 63-cell jump per frame
*inside* a block, and those five regressions are where that splice costs more
than the phase alignment buys. Reversing the curve on alternate frames
(serpentine) achieves the same alignment with no splice and never regresses.
**If this is ever built, build the serpentine form.** Anyone proposing rotation
on the grounds that "Hilbert is a closed loop so rotating preserves adjacency"
has the premise backwards; that argument has been made once and it is wrong.

Also visible only once the sweep covers legal canvases: **`3d` is the most
canvas-sensitive ordering of the four, not the most robust.** Its floor is
51.5% at the shipped length, well below plain `hilbert`'s 77.1%, and it is
worse than plain `hilbert` on **14 of the 48**. Its four worst are all legal
shipped-set canvases: 1952x544 (51.5%), 1888x544 (52.5%), 1568x672 (52.7%),
1440x736 (53.7%). It reaches 97.9% at 1344x768, which is the canvas
this page happens to measure on, and that single number is where its reputation
here comes from.

So the canvas rule this page narrowed to a `2d_frame`-only rule is not quite
that either.

### The `3d` pin was selected at `3d`'s best canvas

**Open question, raised 2026-08-16, and the most consequential thing on this
page that nobody can currently settle.** `SOL_RECOMMENDED_CUDA` pins
`morton_curve="3d"` on centroid fidelity measured at 1344x768 — where `3d` is
97.9% connected, the best of the four. The 48-canvas sweep says that is `3d`'s
best canvas and that it is the most variable ordering in the legal set. **A
default chosen where a curve looks best, deployed across a set where it is the
least predictable.**

Two readings, and nothing distinguishes them today:

- **(a) The activation advantage is canvas-independent.** Geometry and
  activations have already been shown to rank orderings differently (see
  "Geometry does not rank orderings"), so a curve can be geometrically variable
  and still summarise well everywhere. Under this reading the pin is correct and
  the sweep is a curiosity.
- **(b) The advantage tracks the geometry.** Then the pin is right for 1344x768
  and wrong for much of the legal set, and the ordering axis needs a
  canvas-dependent default rather than a fourth curve.

**(a) is not the safe assumption merely because it keeps the pin.** The evidence
for geometry-activation decoupling is 0.002-0.007 differences between arms *at
one canvas*; it says geometry does not rank orderings at a fixed canvas, which
is a different claim from "activations are canvas-invariant". Nothing here
supports the second.

**What separates them, and it is cheap.** Run `bench/analyze_capture.py`'s
centroid-fidelity arm on a capture taken at a *second* canvas and compare `3d`'s
margin over `hilbert` and `2d_frame` against the 1344x768 margins. If the margin
survives, (a). If it collapses where the geometry collapses, (b).

One caution on the obvious candidate. The 2026-08-17 reference capture is at
1024x768, whose 768 tokens per frame **is** divisible by 64 — one of only 3 such
canvases in the legal set, against 1344x768's 1008 which is not. So it moves
canvas *and* phase-alignment together, and a difference could be either. It
still beats one canvas and it costs nothing extra to score, but it is not the
clean two-point line this question wants; a ragged second canvas would be
better.

Until then: geometry only, and **no activation measurement exists at any canvas
except 1344x768.** Nothing is at risk while `morton=False` ships — the exposure
is the next person who turns it on.

**Still link 5.** Whether 90% connected beats 60% connected in the output is
unmeasured, exactly as with everything else on this page.

### What the captured activations say

**First run of `h3_capture.py`, 2026-08-15.** One dense render, Sol-Attn
bypassed so sage took every call: 124 frames, 1344x768, t2v, 6 steps. Three
captures at blocks 0, 24 and 49, step 1, each `[1, 56, 37826, 128]` bf16, taken
after the fused RMSNorm+RoPE. Sequence 37,826 = 530 conditioning rows + 37,296
video rows, grid (37, 24, 42), 591 blocks. Analysed by
`bench/analyze_capture.py`. Files and provenance kept outside the repo; see
that script's docstring.

Because the capture is dense, attention is permutation-equivariant, so a Morton
arm's q/k is *exactly* the permuted capture at every block. One render gives
every ordering with no second render and no approximation.

**Centroid fidelity.** Mean cosine of each key to its own block's centroid,
over all 56 heads and all 582 whole video blocks:

| block | raster | morton `2d_frame` | morton `3d` | `hilbert` |
|---|---|---|---|---|
| 0 | 0.7523 | 0.7565 (+0.6%) | 0.7830 (+4.1%) | not run |
| 24 | 0.7442 | 0.7665 (+3.0%) | **0.7915 (+6.4%)** | 0.7748 (+4.1%) |
| 49 | 0.8678 | 0.8804 (+1.5%) | **0.9434 (+8.7%)** | 0.8978 (+3.5%) |

**Mass concentration.** Key blocks needed to hold 90% of a query's attention
mass, of 591. Sampled: 4 heads of 56, 48 video queries each. The attention
weights are identical under every ordering, so this measures regrouping alone.

| block | raster | morton `2d_frame` | morton `3d` | `hilbert` |
|---|---|---|---|---|
| 0 | 393.7 | 386.5 (-1.8%) | 365.8 (-7.1%) | not run |
| 24 | 178.0 | 149.3 (-16.1%) | 144.7 (-18.7%) | **142.1 (-20.2%)** |
| 49 | 238.1 | 228.9 (-3.9%) | 236.3 (-0.8%) | **226.8 (-4.7%)** |

Five things follow.

**Link 5 is true.** Morton raises centroid fidelity on real activations at
every depth measured. The canvas result is therefore about something real.

**Geometry does not rank orderings.** Added 2026-08-16. This is a statement
about the *instrument*, not about how much headroom is left, and the two get
confused constantly -- so they are separated here deliberately, and the second
one is a judgement call while this one is not.

Four arms, same capture, same canvas, all blocks:

| arm | radius | connected | b0 cos | b24 cos | b49 cos |
|---|---|---|---|---|---|
| `hilbert` (shipped) | 5.02 | 86.1% | 0.7652 | 0.7748 | 0.8978 |
| + serpentine | 4.09 | 93.8% | 0.7694 | 0.7796 | 0.9023 |
| + rotation | 4.10 | 93.8% | **0.7706** | **0.7820** | **0.9070** |
| gilbert + serpentine | **3.71** | **100.0%** | 0.7694 | 0.7774 | 0.9014 |

Two readings, and neither is "diminishing returns":

- **Identical geometry, consistently different activations.** Serpentine and
  rotation are geometrically indistinguishable -- 4.09 against 4.10 radius, both
  93.8% connected -- and rotation is ahead on centroid fidelity at *all three
  depths*. Geometry cannot see whatever separates them.
- **The ranking inverts.** `gilbert + serpentine` is geometrically the best of
  the four by a wide margin, and has the *smallest* activation gain of the three
  improvements (+0.34% at block 24, against rotation's +0.93%).

So spatial compactness is not a saturated proxy for centroid fidelity. It is the
**wrong objective**, and an earlier version of this section said "saturated",
which invites "geometry still buys something, just less". It does not reliably
buy anything: it ranks these four arms differently from how the activations do.

**What that forecloses, and it is most of this page's method.** `radius`, `fill`
and `nbr` in `bench/analyze_morton.py` are what nearly every curve argument here
has been run on, including the canvas sweep above and the case for serpentine.
They are still the right tool for *mechanism* questions -- does a block hold one
region or three -- and they are not a basis for choosing between orderings.
Choose on the capture.

**A second leg, from the legal-canvas sweep above:** geometric rankings are not
even stable across canvases. `3d` runs 51.5% to 100% connected over the legal
set and swaps places with every other ordering along the way. So tuning against
radius or connectivity does not just optimise the wrong thing -- it picks a
different winner at each resolution.

> **Caveat, and it belongs inside the claim.** Both readings rest on differences
> of 0.002 to 0.007 in mean cosine, from one capture, one canvas, one step, at
> **124 frames -- below Sol's ~60k token floor** (see the box at the top of this
> page for what that does and does not invalidate). They are stated because they
> are *consistent across three depths*; a second capture could erase them. The
> activation numbers are **reported from the unlanded 2026-08-16 prototype**,
> same status as the routed-density result above. The geometry columns are
> re-derived here, and geometry is the one part measured to be length-insensitive.

**Separately, the priority call: a fourth curve is not where the remaining
quality is.** This is a judgement and does not follow from the finding above --
someone may later argue there is headroom in orderings, and they must not be
able to cite "geometry does not rank orderings" as if it settled that. The case
rests on four other things: the whole spread across every arm ever measured is
0.3-0.9% centroid fidelity; all of them are speed-identical (452.8-454.8 s); `3d`
beats every 2D arm by mixing frames, which is a different axis entirely
(0.9434 at block 49 against `gilbert + serpentine`'s 0.9014); and link 6 is
untouched, so none of it is known to reach the output at all.

**Both added curves beat the shipped `2d_frame` on both metrics at every block
measured.** They do not beat each other consistently: `3d` leads centroid
fidelity, `hilbert` leads mass concentration, and `hilbert` has the higher floor
at block 49 (min 0.8740 against `3d`'s 0.8636). The two metrics disagree about
the ranking, which is why both are bench arms and neither is a new default.
`hilbert` comes from `sol_curves.py` and needs the `MiniMaxH3SolAttnCurve`
node; `3d` needs nothing, it is already on Sol-Attn's own combo.

Taking `3d` alone against the shipped curve, it scores higher on centroid
fidelity at every depth sampled. State that as the measurement it is: it says
nothing yet about output, and no render has been made with `3d` on a long clip.

It does sit awkwardly against the reasoning behind the default. `2d_frame` was
chosen because H3's `FRAME_PER_TOKEN` is `(1, 4, 4, 4, 4)`, so index-adjacent
latent frames are 1 or 4 real frames apart and a 3D curve groups temporally
distant tokens. That argument is mechanically correct. These numbers do not
refute it; they show that whatever it costs, `3d` still summarises better on
this capture. Why is not established. One candidate, untested: video latents
may be more temporally redundant than the argument assumes.

The next step this implies is a `3d` render, not a config change.

**The geometry predicted the tail and the activations confirmed it
independently.** `2d_frame`'s p10 centroid fidelity is *below* raster at blocks
0 and 49 (0.7291 against 0.7483; 0.8455 against 0.8618) while its mean is above.
That is exactly what the geometry said: 4.7% of `2d_frame` blocks are looser
than raster's worst. Two unrelated measurements, same shape. `3d` improves the
mean and the tail together.

**Attention is not very sparse on this workload.** At its most concentrated a
query still needs 178 of 591 blocks for 90% of its mass, and 394 at block 0.
That bounds what any block-sparse method can save here, and it is an argument
against pushing `tau` far.

Caveats, and they are not small: one render, one step, one prompt, t2v, 124
frames, 6 steps. Test 2 samples 4 heads of 56. Dense activations, so a real
sparse run's trajectory would differ. And none of this is output quality.

**One more, added 2026-08-16, and it bears on the config change this
measurement caused.** The capture is at **1344x768**, which is a ragged canvas
and therefore **`2d_frame`'s worst case**. Geometry at that canvas puts
`2d_frame` at radius 5.54 / fill 0.60 against 3.24 / 1.00 on a clean one, so
the arm that lost was measured where it is weakest. On a clean canvas the gap
between `2d_frame` and `3d` would be expected to narrow; whether it narrows to
nothing, or reverses, is unmeasured.

That does not undo the result -- `3d` also beat raster, and beat `2d_frame` on
the tail as well as the mean -- but it does mean **`morton_curve="3d"` is
pinned on a comparison taken at one canvas that disadvantaged the alternative.**
The fix is cheap and the harness exists: capture once at 1280x768 and re-run
`analyze_capture.py`. One render, no new code.

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

### Morton still works with references

References change what row the video span starts on, and the kernel counts its
64-row blocks from row 0 of the whole sequence. So once references are present
the tiles stop lining up with the blocks, and every tile would be split in
half. `_perm_for` rotates the permutation by that offset to realign them.

**This one is exhaustive rather than sampled, and it is the answer to "do my
reference sizes matter".** They do not. **Block shape depends only on
`video_start mod 64`**, so there are exactly 64 distinct cases and all 64 were
run. At 1024x768 every block is a solid 8x8 square in every one of them, radius
3.240371 at all 64, spread exactly zero. At the ragged 1344x768 the spread is
1.7e-3, which is nil.

Why the count and the resolutions of the references drop out: they change
**how many rows precede the video**, and nothing else Morton can see. Four
references at four different sizes produce one `video_start`; that number mod
64 selects one of 64 cases; all 64 are verified identical. So a canvas that
tiles cleanly tiles cleanly at any reference load, and one that does not is not
made worse by adding references.

**Confirmed on `3d` as well, 2026-08-16**, since `3d` is now the shipped curve
and the exhaustive run above predates it. At 1024x768, `video_start` 0 / 33 /
63: `3d` holds radius 1.61-1.62 and fill 0.98 throughout, and `2d_frame` holds
3.24 / 1.00. The `_noroll` control moves in both cases -- `2d_frame` to radius
6.82 and fill 0.41 at offset 33, `3d` to 3.18 / 0.35 -- so the rotation is
doing real work for both curves, not just the one it was written for.

Remove the rotation and blocks degrade to radius 3.24-6.84 at 1024x768,
depending on the offset, roughly 2x looser at worst.

**Retracted, 2026-08-15: "without the rotation you would be better off with
Morton off."** Not true. Raster order at 1024x768 is radius 9.25, and
un-rotated Morton never exceeds 6.84, so it beats raster at 0 of 64 offsets.
The original and correct comparison was against the *ragged canvas*, fill 0.42
against 0.60; that got upgraded to "worse than raster" in a later edit and
nothing caught it until the exhaustive run. The rotation earns its place on the
2x, not on rescuing Morton from being harmful.

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

### Rendered, 2026-08-16: the permutation is free and the curves are indistinguishable on speed

Six arms at 1344x768, 294 frames, 16 steps, one run each, same seed. The last
two are the isolation control: all 50 blocks forced dense, so Sol-Attn does
nothing and the permutation is the only difference between them.

| arm | sampler |
|---|---|
| sparse, morton off (shipped) | 454.0 s |
| sparse, morton `2d_frame` | 452.8 s |
| sparse, morton `3d` | 454.8 s |
| sparse, `hilbert` | 453.2 s |
| dense, morton `2d_frame` | 861.6 s |
| **dense, morton off** | **860.8 s** |

The "vs its pair" ratio column this table used to carry is gone deliberately.
Every within-pair difference here is under the noise floor, and a ratio printed
to four decimal places reads as a measurement whatever caveat sits beside it.

**The permutation is free, and "free" is the strongest claim these arms
support.** Both pairs, with their signs:

| pair | morton off | morton on | delta |
|---|---|---|---|
| dense (the isolation control) | 860.8 s | 861.6 s | **+0.8 s**, 0.093% |
| sparse | 454.0 s | 452.8 s | **-1.2 s**, 0.264% the other way |

**The two pairs disagree in sign**, and the sparse one says the permutation made
the render faster, which it cannot have. One run per arm, against a bench whose
measured run-to-run spread was 0.1% and 0.12% at 362 frames. So both deltas sit
at or under what this experiment can resolve, and the honest reading is that the
permutation costs nothing measurable -- not that it costs 0.8 s.

**This page said "the permutation costs 0.8 s of 861, or 1.0009x. That is the
isolated number" for most of 2026-08-16.** It was one arm of a two-arm result, quoted
as though the other arm did not exist, three paragraphs above a passage that
kills a VRAM finding for having exactly this property. **Opposite signs across a
control pair means "no effect", and it means that for time the same way it means
it for memory.** Caught by a second reader; not by the page that contains both
paragraphs.

What survives unchanged is what it retracts: `h3_config.py`'s "worth 1.16x
alone, 94% GPU utilisation" does not describe this backend. That figure was
Triton, 362 frames, stacked on int8. It is not wrong about what it measured, and
a real 1.16x cost would have been far outside this noise floor.

**The three curves are indistinguishable on speed**, 452.8 to 454.8 s across a
2 s spread on single runs. So there is no speed argument for or against any of
them, and the choice rests entirely on the activation measurements above.

**The dense pair also priced Sol-Attn itself**, since it is the same config with
only `dense_blocks` changed. That is a Sol-Attn number rather than a Morton one,
so it lives in [`docs/SOLATTN.md`](SOLATTN.md) with the arm it replaces.
Canonical there; not restated here.

**Do not quote peak VRAM from this run.** It looked at first like Morton saved
3.7 GB, consistently across all three curves. The dense control killed it:
Morton saves 3,706 MiB in the sparse arm and *costs* 2,144 MiB in the dense
one. Opposite signs, so it is not a Morton effect, and it matches
`h3_config.py`'s standing warning that process peak here tracks ComfyUI's
allocator rather than the arm. Had the control not been run, a 3.7 GB saving
would have been reported as a finding.

**The `hilbert` arm completing is the node's live verification.**
`MiniMaxH3SolAttnCurve` raises when the permutation patch does not install, so
a clean run is proof the curve engaged rather than an assumption.

Clips, in arm order, for whoever watches them: `h3_sage_ab_00093` (morton off),
`00094` (`2d_frame`), `00095` (`3d`), `00096` (`hilbert`), `00097`
(reorder-only, dense). **Nobody has watched them. No quality claim follows from
this section.**

## Should you turn it on? The short answer for someone who just wants to know

Put here so the answer stops being re-derived, differently, in every
conversation. Nothing in this section is new; it is the rest of the page
compressed to the question people actually ask.

**Default: off.** Not because it is known to hurt, but because **nobody has
measured whether it helps -- not us, not upstream, not either third-party
pack.** It is off in the node, off in every shipped graph here, absent from
NVLabs' H3 profiles, and the author's own words are that it "may or may not
increase quality, that's something to test". Everything anyone has tuned was
tuned with it off.

**It costs nothing, so the only real cost is that your outputs change.** The
permutation is free within measurement error. But it is **not bit-neutral**, so
the same seed gives a different render. Treat turning it on as a reseed, not an
upgrade.

**Turn it on only if you are willing to test rather than assume**, and if so:

- **Check more than one seed.** The single quality finding this repo ever had
  was retracted because it vanished at the second seed, within an hour.
- **Get the curve right, because it is a second widget.** From a graph built
  here you get `3d` and want height 768, 640 or 512, any width. From a fresh
  node you get `2d_frame` and want 768x768, 1024x768 or 1280x768. Turning
  Morton on without checking the curve is the most likely way to conclude it
  does nothing.
- **Raise `tau` at the same time.** The only stated payoff anywhere is *same
  quality at higher sparsity*. At fixed `tau` you get the perturbation and none
  of the payoff -- and, because the threshold is derived from the block
  centroids, fixed `tau` is not even a fixed operating point.

**Leave it off when:**

- You have seeds you like. It changes all of them.
- You are A/B-ing anything else. It is a confound.
- You are on `2d_frame` and a canvas that is not one of the three. That is the
  one configuration measured to make a minority of blocks *worse* than
  anything raster produces -- 4.7% at 1344x768. `3d` does this at 0%.
- The clip is short or small. Below roughly 60k tokens nothing here is
  measurable either way.

**What would change this answer:** one person watching a Morton-on and a
Morton-off clip end to end, at matched wall clock rather than matched `tau`,
on a canvas that suits the curve. That is a couple of hours of GPU time and
nobody has spent it.

## What upstream says the payoff is, and why our arms could not find it

From the Morton docstring in the Triton pack, `_morton.py:1-11`, read in
source on 2026-08-15:

> Z-ordering makes each block a roughly 4 x 4 x 4 neighbourhood, which
> concentrates the mass into fewer blocks and lets the same quality be reached
> at higher sparsity.

**Check which file that is before leaning on it.** Line 1 of the same docstring
opens "Morton (Z-order) token reordering **for Wan**". Three facts, all from
greps on 2026-08-15:

- The sentence appears exactly once in the whole Triton pack, in the Wan file.
- `_morton_h3.py`, the H3 variant, makes **no** quality or sparsity claim. Its
  docstring is entirely mechanical: which segment gets reordered, why the span
  is registered from `PackedLayout.__init__`, which three pieces are gated.
- The CUDA node contains the word "sparsity" **zero** times. Its `morton`
  tooltip says only "compact 3D neighbourhood" and "exactly neutral for dense
  attention". Our `vendor/` copy is byte-identical to upstream's POC, so that
  absence is upstream's and not ours.

So the honest version is narrower than "upstream says Morton pays off at higher
sparsity on H3". It is: upstream states that rationale for Wan, states nothing
either way for H3, and told us directly on 2026-08-14 that Morton "may or may
not increase quality, that's something to test", which is consistent with the
H3 file being silent. Whether the Wan rationale carries to H3 is a question to
ask, not an argument to quote.

Taking the mechanism at face value anyway, because it is the only stated one:
the payoff is at higher sparsity. At fixed `tau`, Morton can
only add work: it costs a permutation and some non-tensor-core time, which is
what this project measured when it recorded the Morton arm running at 94% GPU
utilization where every other arm hit 99%.

Every Morton arm run here has held `tau` at its shipped value and toggled
Morton alone. Under the mechanism as its author states it, that arm cannot
find the benefit. It measures the cost and none of the payoff.

**That sentence is the tidy version and it is not quite right.** Holding `tau`
fixed does not hold sparsity fixed, because the threshold is derived from the
variance across block centroids and Morton changes those centroids -- see
"Holding `tau` fixed does not hold sparsity fixed" above. So a fixed-`tau` arm
is not "the same operating point plus a permutation" with the payoff withheld;
it is an unmeasured distance away along the same curve the payoff lives on. The
conclusion survives -- that arm cannot cleanly attribute anything -- but the
reason is worse than the one stated here, and the paragraph above (`94% GPU
utilization`) is quoting a retracted Triton figure besides.

The author has also said directly, on 2026-08-14, that Morton "may or may not
increase quality, that's something to test". That is good evidence the author
has not tested it, and no evidence at all about anyone else. Here, the speed
result is settled and the quality question is untouched.

## Sol-Engine says H3 needs no reordering. Our capture disagrees

NVLabs ship Morton for Wan, on by default, and leave it out of H3 entirely.
Their stated reason is quoted in
[`docs/sol_upstream.md`](sol_upstream.md#morton-in-sol-engine-which-does-exist):
the packed video tail "is already a contiguous grid-ordered block, and the
routing works on it directly".

**That reasoning does not obviously distinguish H3 from Wan**, and it is worth
saying so. "Already a contiguous grid-ordered block" is true of raster order,
which is exactly the layout Morton is applied to *fix* on Wan. Wan's 720p grid
is 45x80, so a 64-token raster block is under one row wide; H3 at 1344x768 is
24x42, so a block is about 1.5 rows. If anything H3's raster blocks are the
less degenerate of the two.

**And the capture measurement disagrees with the assertion.** On real H3 q/k,
Morton3D raises per-block centroid fidelity by 4.1% / 6.4% / 8.7% at blocks
0 / 24 / 49. That is not an output-quality result and it does not make them
wrong -- their claim may be "the benefit does not justify the cost", which is a
different statement, and they run Ulysses sequence parallelism where a global
permutation interacts with sharding in ways a single GPU does not face. But the
disagreement is real, and with the paper now read it is the sharpest open
question on this page: the only party who has measured whether reordering helps
H3 is us, and we measured the summary, not the picture.

### Their ablation control, which we have now half-run

`config/wan21_t2v_14b/reorder_only.toml` is a "reorder-only control: global
Morton3D order with every layer forced dense through the Sol-Attn adapter" --
reordering on, every layer dense. Since the permutation is output-neutral under
dense attention, that arm isolates what the permutation *costs* from what it
buys.

We copied it on 2026-08-16 and it was half a control. Forcing every block dense
disables Sol-Attn entirely, so the arm measures "no sparse attention plus a
permutation", not "a permutation". The missing arm was dense-without-Morton, it
cost one 15-minute render, and it is what turned 861.6 s into the 0.8 s isolated
figure below. It also killed a 3.7 GB VRAM "saving" that all three sparse curves
agreed on.

**Structural version, worth more than the number: three arms agreeing is not a
control; a fourth arm that isolates the variable is.** The three agreed because
all three had Morton on, which was the thing under test.

## What we know and what we do not

Known, by measurement on this machine:

- Morton is exactly neutral for dense attention, by construction.
- Under `2d_frame` it produces whole 8x8 tiles on 1280x768, 1024x768 and
  768x768, and ragged fragments everywhere else, 1344x768 included. Under `3d`
  the canvas barely matters: radius ~1.6 and fill 0.98 on every canvas tested.
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
  measured; which blocks the router then selects is not. **The mechanism is now
  read from source** -- Morton moves the threshold, not just the membership, so
  fixed-`tau` arms do not hold sparsity fixed -- but the size and even the sign
  of that shift are unmeasured. See "Holding `tau` fixed does not hold sparsity
  fixed". This is the cheapest unrun item on the page and needs no GPU.
- How a fixed curve compares to the content-based clustering that SVG2 uses.
- Whether a dense Morton render differs from a dense non-Morton one by more
  than the floating-point noise measured above. Two rendered clips already
  exist for this and nobody has looked at them.

## Tests run and not run

Run:

- `bench/analyze_morton.py`, all figures on this page. No GPU, no model, about
  a second. Cross-checked against an independent implementation of the
  permutation on both curves before printing.
- The full check suite, 17 scripts, green at commit `9ffe33e`.
- `bench/smoke_h3.py` on `h3_probe_sol_on_api.json`, green, with all three
  chain lines and the CUDA kernel tag confirmed in the log.

Also run, 2026-08-16:

- **Read arXiv 2607.24027.** It was the cheapest thing on this list and it
  answered one row: the paper says nothing about token order at all. See
  [`docs/sol_upstream.md`](sol_upstream.md).
- **The capture and the centroid analysis.** `h3_capture.py` ran for the first
  time; `bench/analyze_capture.py` graded three depths. This is what settled
  link 5.
- **The ordering sweep, six arms**, including the dense pair that isolated the
  permutation cost.

Not run, in the order they would answer the most:

- A `tau` by Morton grid at equal wall clock. Nothing blocks it but GPU time.
- **The routing simulation proper, and it is now the top of the list.** The
  capture answered which tokens share a block and how well the centroid
  represents them. It did **not** compute the selected-block masks, which is the
  step that turns "membership changed" into "this fraction of routed blocks
  changed, in these places". One capture is enough for both orderings without a
  second render. Promoted 2026-08-16 on the threshold finding: because `kcvar`
  is computed over the block centroids, Morton moves the routing threshold as
  well as the membership, so this is no longer a nice-to-have number -- it is
  what decides whether any fixed-`tau` Morton arm is a controlled comparison.
- **A second capture at 1280x768.** The existing one is at 1344x768, which is
  `2d_frame`'s worst case, and it is what moved `SOL_RECOMMENDED_CUDA` to `3d`.
  One render re-runs the whole comparison on ground that does not disadvantage
  the loser. Cheapest way to check a shipped default that rests on one canvas.
- A clean canvas against a ragged canvas at matched settings.
- `morton_curve="3d"` on a long clip. Not rendered here, and it is now the
  shipped curve if Morton is ever turned on.
- Watch the five clips from the ordering sweep. Every conclusion from that run
  is about time and memory, not picture.

Prior commits for context: `3b86b21` records the Morton observation that was
sent upstream, `440eea9` retracts it, and `bd392c2` adds the Sol-enabled probe
graphs the reference arms use.

## What to try next, in order

1. **Count the routed blocks under each ordering, off the existing captures.**
   Zero GPU, no render, tensors already on disk. It settles whether a
   fixed-`tau` Morton arm is even a controlled comparison, which every other
   item on this list assumes. If routed density moves, the tau-by-Morton grid
   below has to be designed around it rather than on top of it.
   This absorbs what this list used to carry separately as "run the routing
   simulation" -- same capture, same arithmetic, and the threshold finding is
   the reason it moved from third place to first.
2. Compare the two dense clips already on disk. Zero GPU, and it answers
   whether the floating-point divergence above is visible. If it is, every
   past Morton impression is explained without reference to block membership,
   and the bar for a Morton quality claim goes up sharply.
3. Run the `tau` by Morton grid. Pick a Morton-on `tau` that matches Morton-off
   at the shipped `tau` on wall clock, so the comparison is quality at equal
   speed rather than quality at equal `tau`. **Design it after item 1**, not
   before: if Morton already moves routed density at fixed `tau`, "equal wall
   clock" and "equal sparsity" are different axes and the grid has to say which
   one it holds. Note what the paper adds here: the correction's advantage over
   drop-the-block **widens as sparsity rises**, so pushing `tau` may cost less
   than a keep-or-drop intuition suggests. That is their ablation on their
   models, not ours.
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
