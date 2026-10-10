# The frozen video cache's cached step, timed by stage on the refine pass (2026-10-06)

lane: masked
verdict: a cached step with almost no live rows costs 6.87 s against a 23.0 s stock step at 104,515 rows, and no one stage owns it: the store, qkv over every row, the attention call and the time between blocks are each a fifth to a quarter; that fixed cost is a fifth lower than the figure derived for the masked lane's cost model, so break-even moves from a live share near one half to a little above it

**What was asked.** Extending `MiniMaxH3FrozenVideoCache` to a masked
video-to-video window (board card `use-frozen-row-cache`) rests on one number
nobody had timed: what a cached step costs when it has almost nothing live.
That fixed cost is paid whatever the mask is, so it sets the live share above
which the cache cannot win. The phase 1 cost model took it from
[`2026-09-25_frozen_cache_s1.md`](2026-09-25_frozen_cache_s1.md) by
subtraction. The audio-refine pass is the case itself: its live rows are the
text and the audio, well under a fiftieth of the sequence. Since 578e16ba
`verify` splits each cached step's seconds by stage
(`frozen_video_cache.STAGES`), and this is the first card run with it.

**How to read it.** The three arms of `bench/frozen_cache_arms.json`, one
seed, one scene, one render per arm, the card to itself. Seconds are wall
clock. Nothing here was listened to or looked at: the refine pass's quality
was judged on 2026-09-26 and is not re-judged. Every figure is in
[`2026-10-06_frozen_cache_stage_split.json`](../../data/results/2026-10-06_frozen_cache_stage_split.json)
(the cache's log lines and every progress-bar state, parsed from the server's
log) or in the rows,
[`2026-10-06_frozen_cache_stage_split.jsonl`](../../data/results/2026-10-06_frozen_cache_stage_split.jsonl).

## What ran

- **Arms**, all FlashGen 4-step with the audio-refine pass, diner scene, seed
  730451892, 1344x768, 345 frames: `refine` (the uncached graph),
  `refine_cached_verify` (the shipped graph with `verify` on),
  `refine_cached` (the shipped graph).
- **Command:** the manifest's own `run` line with this record's `--out`.
- **Server:** restarted for this by the session that owns the card today,
  unarmed, at c0d1e520, which contains 578e16ba. The first arm was the first
  prompt after the start, so it holds every cold load.
- **Cache state:** the first arm rendered the whole graph. The two cached
  arms took its first pass from ComfyUI's node cache (their rows have no
  first-pass node), so their rows time the refine sampler alone.
- **Sequence:** 104,515 packed rows (Sol-Attn's line and the cache's build
  line agree); the video is 102,816 of them, so 1,699 rows are live.
- **The cache engaged** in both cached arms: one build each, int4, 13.49 GiB
  in RAM, and five cached steps.

## The step, three ways

From the sampler's progress bar (its rate, to a hundredth of a second) and
the refine sampler node's own seconds in the rows:

| | per step | refine sampler node |
|---|--:|--:|
| stock refine step (`refine`, six steps) | 23.0 s | 138.5 s |
| the build (`refine_cached`, step 1) | 27.79 s | |
| a plain cached step (`refine_cached`, steps 2 to 6) | 6.87 s | 62.9 s |
| a verified cached step: the stock pass, then the timed cached pass | 31.3 s | 185.4 s |

The bars add up to the nodes: 27.79 plus five steps at 6.87 is 62.1 of the
62.9 s, and 28.12 plus five at 31.3 is 184.6 of the 185.4 s. The bar's
elapsed column runs backwards once in the plain cached arm (the JSON has every
state), so the step is read from the rate and checked against the node.

**So a cached step is 0.30 of a stock step, and a build is a stock step
plus 4.8 s.** The refine sampler is 2.2 times shorter with the cache. On
2026-09-25 the same node took 143.4 and 72.0 s; the stack has moved since
(that record's graphs loaded a merged rank-13 LoRA, these load rank 64 at
the call), so the pair is context, not a comparison.

## The split

The verified arm's five timed cached passes, seconds by stage:

| video sigma | between blocks | live rows | store to card | qkv, every row | attention | the pass |
|---|--:|--:|--:|--:|--:|--:|
| 0.8957 | 1.50 | 0.15 | 2.30 | 2.31 | 2.06 | 8.32 |
| 0.8575 | 1.50 | 0.14 | 2.29 | 2.32 | 2.07 | 8.32 |
| 0.8000 | 1.50 | 0.14 | 2.29 | 2.31 | 2.05 | 8.29 |
| 0.7064 | 1.50 | 0.14 | 2.29 | 2.31 | 2.06 | 8.30 |
| 0.5239 | 1.50 | 0.14 | 2.29 | 2.32 | 2.05 | 8.30 |

- **The stages add up to the pass**, to the hundredth, on every step, and
  repeat to within two hundredths from step to step.
- **A timed pass is 8.30 s where a plain one is 6.87 s.** Each lap waits for
  the card, five times a block, so work that overlaps in a plain step is laid
  end to end in a timed one. Read the table as shares of the fixed cost, not
  as its seconds: store 0.28, qkv 0.28, attention 0.25, between blocks 0.18,
  the live rows' own work 0.02.
- **No stage owns the floor.** `store to card` is the kept states leaving
  RAM, being dequantized on the card and having the live rows written in.
  `qkv, every row` is the projection and rope over the whole sequence, which
  rebuilds K and V from those states. `between blocks` is everything outside
  this module's block: core's loop and its prefetch, the embedding before
  block 0, the final layer.
- **The attention call has a cost that does not follow the query count.**
  1,699 queries against 104,515 keys is under a fiftieth of the dense square.
  A dense step's attention at this length, scaled from the stock steps in
  [`2026-10-06_masked_render_time_breakdown.md`](2026-10-06_masked_render_time_breakdown.md),
  is about 37 s, so the rectangle alone would be about 0.6 s; the stage reads
  2.06 s. The rest, about 30 ms a call, is paid per call over all the keys.
  That is arithmetic on two records, and nobody has profiled the call.

## What `verify` read

Audio velocity on each cached step against the same step run stock:

| video sigma | cosine | relative L2 |
|---|--:|--:|
| 0.8957 | 0.999668 | 0.02578 |
| 0.8575 | 0.999718 | 0.02376 |
| 0.8000 | 0.999732 | 0.02315 |
| 0.7064 | 0.999769 | 0.02148 |
| 0.5239 | 0.999720 | 0.02367 |

Closer than the 2026-09-25 run's 0.028 to 0.059, on a stack that differs as
said above. These are per-step comparisons from the same state; they do not
compound the way a trajectory does.

## What it says to do about the floor

- **Storing only the kept rows** (the masked design's deferred item) saves
  the live share of the store stage: about a quarter of 0.28 of the fixed
  cost on a window like the band's. Worth a few percent of a cached step.
- **Keeping K and V in place of the hidden state** would remove the qkv
  stage and multiply the store by the ratio of their widths, 14,336 to
  5,376 a row. On these shares that is a loss.
- **Keeping the store on the card** would remove most of the store stage and
  needs the store's own size in VRAM, which this card does not have beside
  the model.
- **The attention call's per-call cost and the time between blocks** are not
  this module's code. They are where a second look would start, since they
  are 0.43 of the fixed cost together.

None of these was tried. The floor is spread thin enough that the masked
design is better judged on its first masked run than tuned here.

## What it means for a masked window

Arithmetic from this record and the breakdown record, not a measurement of a
masked step. The model: a cached step costs a fixed part per row plus the
live share times a dense step's attention and the live rows' own linears. The
fixed part is now measured: 6.87 s less what 1,699 live rows cost by that same
model leaves 6.16 s at 104,515 rows, 59 microseconds a row. The phase 1 model
had derived 75. The build's extra, 4.8 s here, was derived as 5.1.

With one build per window and twelve steps (three dense, nine with Sol-Attn),
stock steps as measured on each window today, live shares as the song node
printed them plus the text rows:

| window | rows | live share | break-even share | cached step | sampling saved, one build | rebuilt every 3 cached steps |
|---|--:|--:|--:|--:|--:|--:|
| band window, 1344x768 | 113,072 | 0.26 | 0.53 | 19.7 s against 33.6 s | 40% | 32% |
| render A, window 1, 1024x768 | 87,692 | 0.50 | 0.55 | 21.1 s against 23.0 s | 13% | 9% |
| render A, window 2 | 87,692 | 0.33 | 0.55 | 15.7 s against 23.0 s | 32% | 25% |
| render A, window 3, 141 frames | 41,272 | 0.26 | 0.65 | 4.5 s against 7.7 s | 37% | 26% |
| render B, window 1, 1344x768 | 113,072 | 0.39 | 0.53 | 26.5 s against 33.3 s | 24% | 19% |
| render B, window 2 | 113,072 | 0.26 | 0.53 | 19.9 s against 33.3 s | 40% | 32% |
| render B, window 3, 141 frames | 52,252 | 0.22 | 0.61 | 5.8 s against 10.4 s | 41% | 30% |

Over a whole render that is 24% of render A's sampling and 33% of render
B's with one build per window, 18% and 26% with a rebuild every three cached
steps. Renders A and B are the two in the breakdown record. The band
window's stock steps and row count are from a render of the shipped motion
graph the same morning, read from that session's server log, and its live
share is from the clip's kept mask, 23,679 of 102,816 video rows.

**What is still modelled in that table:** that the dense kernel's time
follows the number of query rows once its per-call cost is paid (here it was
seen at 1,699 queries; a masked window has tens of thousands), that the
fixed part scales with the row count, and the cadence. Whether a masked step
can go unrefreshed at all is `verify`'s to say on a masked run, on the
regenerated rows next to kept ones first.

## What this does and does not show

- One scene, one seed, one length, one card, one run per arm.
- The refine pass only. No masked window has run through the cache.
- The split is of a timed pass, which is slower than a plain one.
- Not looked at: memory on the card during a cached step, and whether
  `verify`'s extra pass disturbs anything stateful under it. The three arms
  rendered with no error row.

## Files

- `2026-10-06_frozen_cache_stage_split.jsonl`: one row per arm from
  `bench/run_graph_arms.py`.
- `2026-10-06_frozen_cache_stage_split.json`: the build, `verify` and stage
  lines and every progress-bar state, parsed from the server's log. The log
  itself is kept untracked by the session that ran the arms.
