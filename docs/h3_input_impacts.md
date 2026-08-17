# What your inputs actually cost: canvas, length, and Sol-Attn together

Last updated: 2026-08-17.

Three docs already own the pieces. This one owns the **interaction**, because
the pieces are chosen together and each of those pages answers only its own
axis.

Following [`SOLATTN.md`](SOLATTN.md)'s rule -- a number is stated once, in the
page that owns it, and everywhere else is a verdict plus a `Canonical:` link:

| page | owns | this page |
|---|---|---|
| [`h3_resolutions.md`](h3_resolutions.md) | the legal set, `adapt_canvas`, the 32, the frame grid, the int32 threshold | restates its ratios for context only |
| [`SOLATTN.md`](SOLATTN.md) | every Sol-Attn number measured on this box, and every knob | states no Sol timing or accuracy figure at all |
| [`morton.md`](morton.md) | what token ordering does to Sol's blocks, the curves, the capture analysis, the assumption chain | borrows two figures, both marked `Canonical:` |

**What this page owns**, because it is measured here and appears nowhere else:
the per-canvas ranking of Morton `3d` across all 48 legal canvases, the
`h/32 % 4` by `w/32 % 4` grouping, the `latent_t % 4` length effect, and the
block maps. Added 2026-08-17.

If a number here disagrees with the page that owns it, that page wins.

**Read this when** you are choosing a canvas and a frame count together, or
when you want to see what a 64-token block actually looks like rather than
what its statistics are.

---

## Three grids, and they get confused

The word "latent" names two different grids in this repo, which is most of the
confusion when comparing pages.

```
  pixels          1344 x 768        what you type
     |  VAE compresses space by 16
  VAE latent        84 x 48         h3_resolutions.md's "Latent" column
     |  DiT patchifies (1, 2, 2)
  token grid        42 x 24         what attention and Sol-Attn see
```

Everything on this page is the **token grid**, `(W/32) x (H/32)`. One token is
one 32x32 pixel square of one latent frame. `patchify_video` is
`comfy/ldm/minimax/model.py:42`.

The time axis has its own compression. Frame counts snap to `17n + 5`, and the
latent frame count is `5n + 2`:

| frames | latent frames | frames | latent frames |
|---|---|---|---|
| 124 | 37 | 311 | 92 |
| 243 | 72 | 328 | 97 |
| 260 | 77 | 345 | 102 |
| 277 | 82 | 362 | 107 |
| 294 | 87 | | |

So a 362-frame clip at 1344x768 is a `107 x 24 x 42` box of tokens, which is
107,856 of them.

---

## What a block is

Sol-Attn does not look at tokens one at a time. It cuts the sequence into
groups of 64, builds one averaged summary per group, and uses those summaries
to decide which pairs of groups are worth computing exactly and which can be
skipped. A group of 64 is a **block**.

Which 64 tokens land together depends entirely on what order the tokens are in.
That is the whole reason token ordering exists as a setting.

Below, each letter is one block. These are real maps from
`bench/analyze_morton.py --map`, one latent frame, rows 1-12 of 24.

**Default order (raster), 768x768.** Reading order, like text on a page:

```
iiiiiiiiiiiiiiiiiiiiiiii
iiiiiiiiiiiiiiiiiiiiiiii
iiiiiiiiiiiiiiiijjjjjjjj
jjjjjjjjjjjjjjjjjjjjjjjj
jjjjjjjjjjjjjjjjjjjjjjjj
jjjjjjjjkkkkkkkkkkkkkkkk
kkkkkkkkkkkkkkkkkkkkkkkk
kkkkkkkkkkkkkkkkkkkkkkkk
llllllllllllllllllllllll
llllllllllllllllllllllll
llllllllllllllllmmmmmmmm
mmmmmmmmmmmmmmmmmmmmmmmm
```

A block is a wide, thin band: 2.67 rows tall and the full width of the frame.
One summary has to stand for whatever is on the left of the picture and
whatever is on the right of it at the same time.

**Morton `3d`, same canvas:**

```
000011118888999922223333
000011118888999922223333
000011118888999922223333
000011118888999922223333
22223333aaaabbbb44445555
22223333aaaabbbb44445555
22223333aaaabbbb44445555
22223333aaaabbbb44445555
gggghhhhooooppppaaaabbbb
gggghhhhooooppppaaaabbbb
gggghhhhooooppppaaaabbbb
gggghhhhooooppppaaaabbbb
```

A block is a 4x4 patch here and runs 4 latent frames deep, so 4x4x4 = 64. The
summary describes one small region of one moment.

That is the entire idea. Everything below is about when the bricks come out
solid and when they come out broken.

---

## Reading the three numbers

`bench/analyze_morton.py` reports these per canvas, and they are what the tables
on this page and in [`morton.md`](morton.md) are made of.

| column | plain reading | raster at 1344x768 | Morton `3d` there |
|---|---|---|---|
| **radius** | how far the 64 tokens sit from their own centre, in tile widths. Lower is a tighter region for one summary to describe | 12.10 | 1.65 |
| **fill** | 64 divided by the volume of the box the block fits in. 1.00 is a solid brick with no holes | 0.61 | 0.98 |
| **connected** | share of blocks whose 64 tokens form one clump rather than scattered pieces | 95.4% | 98.3% |

`Canonical: docs/morton.md` for the 1344x768 column -- that is the canvas that
page is written on, and it states these first.

None of these is a quality measurement. They describe the *summary* the router
reads, not the picture that comes out. See "What none of this establishes".

---

## Where the bricks break

**Morton `3d` at 1344x768**, whose token grid is 42 wide:

```
000011118888999922223333aaaabbbbccccddddkk
000011118888999922223333aaaabbbbccccddddkk
000011118888999922223333aaaabbbbccccddddkk
000011118888999922223333aaaabbbbccccddddkk
22223333aaaabbbb44445555ccccddddeeeeffffkk
22223333aaaabbbb44445555ccccddddeeeeffffkk
22223333aaaabbbb44445555ccccddddeeeeffffkk
22223333aaaabbbb44445555ccccddddeeeeffffkk
gggghhhhooooppppiiiijjjjqqqqrrrrmmmmnnnnuu
gggghhhhooooppppiiiijjjjqqqqrrrrmmmmnnnnuu
gggghhhhooooppppiiiijjjjqqqqrrrrmmmmnnnnuu
gggghhhhooooppppiiiijjjjqqqqrrrrmmmmnnnnuu
```

42 is not a multiple of 4. Ten clean 4-wide columns of bricks fit, then a
2-wide strip is left over on the right (`kk`, `uu`), and those leftover blocks
have to find their other 32 tokens somewhere else.

**Morton `2d_frame` at the same canvas**, the node's own default, which never
mixes frames and so has to close a 64-token run inside one frame:

```
vvvvvvvvwwwwwwwwzzzzzzzzAAAAAAAAHHHHHHHHII
vvvvvvvvwwwwwwwwzzzzzzzzAAAAAAAAHHHHHHHHII
vvvvvvvvwwwwwwwwzzzzzzzzAAAAAAAAHHHHHHHHII
vvvvvvvvwwwwwwwwzzzzzzzzAAAAAAAAHHHHHHHHII
wwwwwwwwxxxxxxxxAAAAAAAABBBBBBBBIIIIIIIIII
wwwwwwwwxxxxxxxxAAAAAAAABBBBBBBBIIIIIIIIII
wwwwwwwwxxxxxxxxAAAAAAAABBBBBBBBIIIIIIIIII
wwwwwwwwxxxxxxxxAAAAAAAABBBBBBBBIIIIIIIIII
```

Look at `w`. It occupies rows 1-4 at columns 8-15 **and** rows 5-8 at columns
0-7. One block, two pieces, not touching. That is what "58.1% connected" means
in [`morton.md`](morton.md)'s tables, made visible.

---

## The canvas rule for `3d`

Swept over all 48 legal landscape and square canvases at 362 frames, using the
vendored node's own `morton_perm`. The pattern is entirely in whether each
token axis divides by 4:

| `h/32 % 4` | `w/32 % 4` | canvases | radius | connected |
|---|---|---|---|---|
| **0** | **0** | **6** | **1.61 - 1.61** | **99.1% - 99.3%** |
| 0 | 1, 2, 3 | 21 | 1.63 - 1.80 | 86.6% - 99.0% |
| 1, 2, 3 | 0 | 5 | 1.70 - 1.86 | 84.3% - 98.8% |
| 1, 2, 3 | 1, 2, 3 | 16 | 1.73 - 2.44 | 51.5% - 97.8% |

In pixels the top row reads: **width and height both divisible by 128.** Six
canvases in the legal landscape set qualify, and every one has a portrait
mirror at identical geometry (checked directly, not assumed: 1024x768 and
768x1024 both score 1.609 / 0.988).

| canvas | token grid | radius | fill | connected | tok/frame | attention vs 16:9 |
|---|---|---|---|---|---|---|
| 1664x640 | 52x20 | 1.607 | 0.989 | 99.3% | 1040 | 1.06x |
| 1152x768 | 36x24 | 1.608 | 0.988 | 99.2% | 864 | 0.73x |
| 1280x768 | 40x24 | 1.609 | 0.988 | 99.1% | 960 | 0.91x |
| 1024x768 | 32x24 | 1.609 | 0.988 | 99.1% | 768 | 0.58x |
| 896x768 | 28x24 | 1.610 | 0.988 | 99.1% | 672 | 0.44x |
| 768x768 | 24x24 | 1.609 | 0.988 | 99.1% | 576 | 0.33x |

Portrait mirrors: 640x1664, 768x1152, 768x1280, 768x1024, 768x896, and 768x768
which is its own mirror.

The shipped default 1344x768 is not in this set. It scores 1.654 / 0.983 /
98.3%, which is 13th of 48 by radius. It is fine, not optimal.

The four worst in the legal set, all of them shipped-legal canvases: 1952x544
(51.5% connected), 1888x544 (52.5%), 1568x672 (52.7%), 1440x736 (53.7%).
`Canonical: docs/morton.md`. This page's independently written connectivity
pass reproduces all four to the decimal, which is the cross-check that the
sweep behind the tables above measures the same thing that page does.

**One caution on the mechanism story.** "Divisible by 4 gives 4x4x4 bricks" is
the ideal case, not what always happens. At 1152x768 the map shows mostly 4x4
patches but some 8-wide ones, which are 8x4x2 bricks. They still score fill
1.000 and 100% connected, so the measured claim is *solid connected bricks*,
not literally 4x4x4 everywhere.

Reproduce any row: `python bench/analyze_morton.py --canvas WxH --length 362`.

---

## The length rule for `3d`

`3d` bricks run through time as well, so the latent frame count is a third
axis. `5n + 2` is divisible by 4 at **175, 243 and 311** frames.

Every on-grid length on 768x768, regenerated with
`analyze_canvas_geometry.py --lengths 768x768`:

| frames | latent frames | `% 4` | radius | fill | connected |
|---|---|---|---|---|---|
| 124 | 37 | 1 | 1.626 | 1.000 | 100.0% |
| 141 | 42 | 2 | 1.627 | 1.000 | 100.0% |
| 158 | 47 | 3 | 1.645 | 0.972 | 97.9% |
| **175** | **52** | **0** | **1.581** | **1.000** | **100.0%** |
| 192 | 57 | 1 | 1.610 | 1.000 | 100.0% |
| 209 | 62 | 2 | 1.612 | 1.000 | 100.0% |
| 226 | 67 | 3 | 1.626 | 0.980 | 98.5% |
| **243** | **72** | **0** | **1.581** | **1.000** | **100.0%** |
| 260 | 77 | 1 | 1.603 | 1.000 | 100.0% |
| 277 | 82 | 2 | 1.605 | 1.000 | 100.0% |
| 294 | 87 | 3 | 1.616 | 0.985 | 98.9% |
| **311** | **92** | **0** | **1.581** | **1.000** | **100.0%** |
| 328 | 97 | 1 | 1.598 | 1.000 | 100.0% |
| 345 | 102 | 2 | 1.600 | 1.000 | 100.0% |
| 362 | 107 | 3 | 1.609 | 0.988 | 99.1% |

**Read the two columns separately, because they say different things.**
Radius is the one that tracks `% 4` cleanly: 1.581 at every aligned length and
1.598 to 1.645 everywhere else. Fill and connectivity do not -- they are
perfect at `% 4` of 0, 1 and 2 alike, and dip only at `% 4 == 3`. So the useful
statement is that **`latent_t % 4 == 3` is the case to avoid** (158, 226, 294
and the shipped 362), and alignment is a smaller further gain on top of that.

An earlier draft of this section said alignment is what reaches 100% connected.
It is not: 328 and 345 reach it too. Alignment buys the radius, not the
connectivity.

**A single-frame map cannot show this**, and it is worth stating because the
obvious check fails silently: the in-frame maps for 1344x768 at 311 and at 362
are byte-identical. The length effect lives in the time direction, which one
frame does not contain. Trust the table, not the picture, on this axis.

**What alignment does not fix is the floor.** At 243 frames, which is perfectly
aligned, the worst canvas in the set scores 46.1% connected, worse than 362's
51.5%. `morton.md`'s observation that `3d`'s floor degrades with clip length
stands and alignment does not rescue it. Two separate effects: the top of the
range is modular in `latent_t % 4`, the floor is not.

| frames | latent frames | `% 4` | worst canvas of 48 |
|---|---|---|---|
| 124 | 37 | 1 | 67.1% |
| 243 | 72 | 0 | 46.1% |
| 311 | 92 | 0 | 51.6% |
| 362 | 107 | 3 | 51.5% |

---

## Length as a cost lever

Independent of any ordering. Attention goes as the square of the sequence, so
length is the second-largest lever after canvas. The long end of the grid at
1344x768; the full grid runs down to 124 frames and is in the length table
above:

| frames | latent frames | aligned for `3d` | video tokens | attention vs 362 | duration |
|---|---|---|---|---|---|
| 243 | 72 | yes | 72,576 | 0.45x | 10.13s |
| 260 | 77 | no | 77,616 | 0.52x | 10.83s |
| 277 | 82 | no | 82,656 | 0.59x | 11.54s |
| 294 | 87 | no | 87,696 | 0.66x | 12.25s |
| 311 | 92 | yes | 92,736 | 0.74x | 12.96s |
| 328 | 97 | no | 97,776 | 0.82x | 13.67s |
| 345 | 102 | no | 102,816 | 0.91x | 14.38s |
| 362 | 107 | no | 107,856 | 1.00x | 15.08s |

**311 is the value point.** 26% less attention than 362 for 2.1 seconds less
clip, aligned for `3d`, and still far above the Sol-Attn floor below.

**345 only for diffusers portability**, whose hard-coded 15.0s `max_duration`
refuses 362. Ask `h3_rules.reference_would_emit()` rather than assuming.

**362 for maximum length, or for comparability**: every bench figure in this
repo was taken at 362, and `workflows/h3_config.py` warns that moving the
length breaks the comparison.

**Skip 328.** It buys 0.7 seconds over 311 for 11% more attention and loses the
alignment.

### There is no 100,000-token budget

This gets asked, and the number is real but it is neither a limit nor the
model's. 99,864 is where a signed int32 byte offset overflows inside the Triton
quantization kernels, given H3's fused qkv stride of `3 x 56 x 128 = 21504`.
`preflight.py:28` states it, and states that the crossing is already handled in
every sage build able to run this repo's attention node. Every shipped graph is
already past it. The next ceiling is a uint32 wrap near 199,728 tokens, roughly
660 frames, against a 362 maximum.

Three places in this repo call it "the model's ~100k ceiling"
(`docs/SOLATTN.md:271`, `docs/bench_plan.md:23`,
`bench/bench_e2e_h3.py:932`), which reads as a property of the checkpoint. It
is not. `preflight.py` and [`h3_resolutions.md`](h3_resolutions.md) state it
correctly. No independent upstream claim of a 100k model ceiling was found in
`docs/sol_upstream.md` or in the Sol-Engine tree.

---

## Where this meets Sol-Attn

**The floor is 60,000 video tokens.** `bench/bench_e2e_h3.py:439` sets it, and
below it a Sol-Attn arm returns a null that reads as "this knob does nothing".
Canvas and length both count, so cheap choices can push you under it:

| canvas | 243 frames | 311 frames | 362 frames |
|---|---|---|---|
| 1344x768 | 72,576 | 92,736 | 107,856 |
| 1280x768 | 69,120 | 88,320 | 102,720 |
| 1152x768 | 62,208 | 79,488 | 92,448 |
| 1024x768 | **55,296** | 70,656 | 82,176 |
| 896x768 | **48,384** | 61,824 | 71,904 |
| 768x768 | **41,472** | **52,992** | 61,632 |

Bold is under the floor. Two consequences. The cheapest member of the `3d` top
tier, 768x768, is under the floor at every length except 362, so it is the
wrong canvas to measure Sol-Attn on. And 896x768 at 311 frames clears the floor
by only 3%, which is close enough that a reference-laden graph shifting the
segment balance should not be assumed to stay above it. **1024x768 at 311
frames is the cheapest combination that is aligned, comfortably above the
floor, and 0.58x the attention of the default.**

**Morton ships off.** `SOL_RECOMMENDED_CUDA` carries `morton=False`, so none of
the ordering geometry on this page is running in any shipped graph today. It
also carries `morton_curve="3d"` (`workflows/h3_config.py:392`), which decides
what you get *if* you turn it on.

**A hand-built graph gets a different curve.** Dropping a fresh
`SolAttnMiniMax` into your own graph gives `morton_curve="2d_frame"`, the node
default, whose canvas rule is much stricter: both dimensions divisible by 256,
which is only 768x768, 1024x768, 1280x768 and the portrait 768x1024. Graphs
from `build_workflows.py` bake `3d`.

---

## What none of this establishes

**Block geometry is not output quality.** Whether a tighter block reaches the
screen is link 6 in [`morton.md`](morton.md)'s chain and is unverified at every
canvas. The repo's own conversion rate is that a 26% better radius plus 14
points of connectivity bought roughly 0.3% centroid fidelity. The whole spread
ranked on this page is a few times that, which extrapolates to on the order of
1% centroid fidelity, and that extrapolation is not a measurement.

**The `3d` pin rests on one canvas.** It was selected on activations measured
only at 1344x768, where `3d` happens to be the best of the four orderings, and
`3d` is the most canvas-variable ordering in the legal set. No activation
measurement exists at any other canvas. `morton.md` names the experiment that
would separate the two readings.

**Do not read the canvas tables as a quality ranking of canvases.** They rank
how compact the router's per-block summaries are, and nothing more.

### Evidence grades

| claim | grade |
|---|---|
| the three grids, and the 32 | source read, `comfy/ldm/minimax/model.py:42` |
| every block map and every radius / fill / connected figure | measured, with the vendored `morton_perm` the node installs |
| the four worst canvases | measured here, and reproduces `morton.md`'s published figures to the decimal |
| the `% 4` canvas rule and the `latent_t % 4` length rule | measured, 48 canvases at four lengths |
| video-token counts and attention ratios | arithmetic over counted rows |
| the 60,000-token floor | a constant in the bench, sourced from upstream's report of nothing measurable under ~250-300 frames at 1344x768 |
| any of it changing the picture | **not measured, at any canvas** |

---

## Reproducing everything here

Every table on this page regenerates. `PYTHONPATH` must reach ComfyUI; none of
these needs CUDA, a model, or a server.

```bash
# the canvas tables above, as markdown, ready to paste back in
PYTHONPATH=/path/to/ComfyUI python bench/analyze_canvas_geometry.py --markdown

# the full 48-canvas ranking, and the %4 grouping
PYTHONPATH=/path/to/ComfyUI python bench/analyze_canvas_geometry.py

# the length table above
PYTHONPATH=/path/to/ComfyUI python bench/analyze_canvas_geometry.py --lengths 768x768

# block maps and single-canvas detail
PYTHONPATH=/path/to/ComfyUI python bench/analyze_morton.py --canvas 768x768 --length 311 --map

# exact packed rows, from the layout the model builds
PYTHONPATH=/path/to/ComfyUI python bench/count_packed_rows.py --length 311 --canvas 1152x768
```

`analyze_canvas_geometry.py` runs two controls before it prints anything: the
vendored permutation against an independent implementation, and its
connectivity figures against [`morton.md`](morton.md)'s published four worst.
Both have been shown red. If either fails, nothing on this page should be
quoted until it is resolved.

The canvas set is enumerated from `adapt_canvas` rather than listed, so if the
area cap or the rounding ever moves, these tables move with it instead of
quietly describing a set that no longer exists.
