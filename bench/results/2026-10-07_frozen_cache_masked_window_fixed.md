# The frozen video cache on a masked window, with its cached step on the kitchen kernel (2026-10-07)

lane: masked
verdict: yesterday's pair repeated with one function changed, so that a cached step's attention runs on the kitchen backend the graph installs and no longer on torch's own kernel: sampling takes 267.1 s with the cache against 462.6 s without, on the same window and seed and in the same warm state, a cached step 18.8 s against a Sol-Attn stock step's 33.5 s, one build and no rebuild; both stock arms are byte-identical to yesterday's; whether the cached render is an acceptable sample of the region is the owner's eye and not said here; the masked use stays retired in the tree until the owner says otherwise

**What was asked.** The owner retired the cache's masked use on
[`2026-10-06_frozen_cache_masked_window.md`](2026-10-06_frozen_cache_masked_window.md),
where it saved no time. The next morning's
[`2026-10-07_frozen_cache_rectangle_kernel.md`](2026-10-07_frozen_cache_rectangle_kernel.md)
showed why: the cached block removed the attention override that carries the
kitchen backend, so its attention ran on torch's kernel. The owner, on the
board that day: "if you see a path here with caching, go for it. Dont spin
wheels for hours". This is the one run that tests the cache as designed.

**Corrected the same day, after the run.** "What ran" below says the changed
function "replaces [the override] with the override at the bottom of the
chain" and that on chains built the way the graphs build them "it returns
the backend under one Sol, under two". That was tested on a stand-in for
Sol's override, and it is wrong for the real one: Sol-Attn's override does
not close over the override below it directly, so the walk stopped at once
and handed back Sol's own override. What the cached arm ran, then, is the
override left in place: Sol-Attn receives the cached step's call, declines
it (fewer queries than keys) and hands it to the kitchen backend it was
installed on. The times, the files and the comparison are unaffected; only
the description of the route was wrong. The live module now leaves the
override in place and says so, and `bench/check_frozen_video_cache.py`
holds it with the real `make_override`, red against the module as retired.
The last bullet of "What this does and does not show" is also out of date:
the masked use is back in the tree as of that commit.

**How to read it.** One window, one seed, one render per arm. The comparison
is `cache_plain` against `baseline_again`, which ran back to back with the
models resident and the conditioning kept. Seconds are wall clock from the
song node's own stage clock; a step's rate is from the sampler's bar.
**Nothing here was looked at**: no arm's picture was opened for this record.
Every figure is in
[`2026-10-07_frozen_cache_masked_window_fixed.json`](2026-10-07_frozen_cache_masked_window_fixed.json).

## What ran

- **Window, graphs and patches:** as on 2026-10-06 (`band_112s`, 1344x768,
  345 frames, the graph's own seed; the shipped ref2va motion graph for the
  stock arms and the probe graph with the cache on the song node's model
  after Sol-Attn for the cached arm, int4, no refresh, no halo), with an
  output prefix of this run's own. The json has each arm's patches.
- **The code:** the pack as it stood at 25103f0e, the commit before the
  retirement, with `frozen_video_cache._dense_options` changed and nothing
  else: where it removed `optimized_attention_override`, it now replaces it
  with the override at the bottom of the chain, the dense backend core's
  Model Attention Backend node installs under Sol-Attn
  (`archive/frozen_cache_masked/dense_options_fix_2026-10-07.diff`). The
  archived check passes on that copy. On override chains built the way the
  graphs build them it returns the backend under one Sol, under two, the
  backend alone, and nothing for Sol alone.
- **Where:** a second server on its own port, started with the main
  server's launch script and flags from a scratch base directory (every
  installed pack linked in, this pack replaced by that copy). The tracked
  tree was not touched and the main server served nothing meanwhile.
- **Order:** a preview (nothing sampled), `baseline`, `cache_plain`,
  `baseline_again`.

| arm | models | source latent and conditioning |
|---|---|---|
| `baseline` | cold: the first sampling prompt of the process | encoded |
| `cache_plain` | resident | kept in the process |
| `baseline_again` | resident | kept |

## The time

| arm | sampling | the whole render | steps, from the bar's rate |
|---|--:|--:|---|
| `baseline` (cold) | 466.8 s | 577 s | three near 53.4, then nine near 33.5 |
| `cache_plain` | 267.1 s | 317 s | the build 58.9, then eleven at 18.8 |
| `baseline_again` | 462.6 s | 512 s | three near 53.4, then nine near 33.6 |

**With the cache, sampling is 195.5 s shorter than the stock arm beside it,
42% of sampling and 38% of the whole render.** The band fixed before
yesterday's run called 35 to 45% "the model holds". Yesterday's cached step
was 36.75 s; the arithmetic in the kernel record put the corrected step near
18.9 s and sampling near 267 s.

The cache's own lines: one build ("no cache yet", 28,689 of 113,072 rows
live, int4, 14.60 GiB in RAM), then eleven cached steps; no rebuild, no
decline.

## What the files say

- `baseline` and `baseline_again` are the same bytes, and the same bytes as
  yesterday's `baseline`: the scratch server reproduces yesterday's stock
  render exactly, so the pair is like for like with it.
- `cache_plain` is a different file from yesterday's `cache_plain`, as it
  must be: its cached steps ran on a different kernel.
- The composite's own line: it keeps 91% of the regenerated pixels in the
  cached arm against 88% in the stock arms. That is a count of pixels that
  differ from the source, not a judgement.

## What this does and does not show

- One window, one seed, one pair, one card. At 23% of the video tokens
  regenerating it is among the lane's most favourable windows; a window that
  regenerates more saves less, and none was run.
- Not looked at: any picture. The pair is on the output share, stacked and
  labelled, for the owner
  (`Video/compare_stacked/band_cache_fixed_baseline_vs_cached_over_mask.mp4`).
  The cached render is a different sample of the region; whether it is as
  good is theirs to say.
- Not measured: how far a cached step departs from a stock step on this
  kernel. Yesterday's `verify` table compared exact attention with Sol's
  sparse attention as well as cached rows with live ones, and was not run
  again here.
- Not measured: the card's memory during a cached step. The store was 14.60
  GiB of host memory, as yesterday.
- A two-thread CPU job of another session's ran for under two minutes during
  the pair. `baseline_again`'s sampling is within half a second of
  yesterday's stock arm, so it did not move the comparison.
- Sol-Attn is not used on a cached step: the cached step computes the live
  rows against every row densely on the kitchen kernel. Letting Sol's kernel
  take a subset of queries is the larger lever the earlier notes name and is
  untouched.
- The fix is not in the tree. The masked use is retired by the owner's
  decision of 2026-10-06, and the live module still sends a masked call
  stock. The audio-refine pass's cached steps still run torch's kernel
  through the same function.

## Files

- `2026-10-07_frozen_cache_masked_window_fixed.json`: each arm's graph,
  patches, stage seconds, report lines and output hash, the cache's log
  lines, the sampler's rates, and yesterday's figures used above.
- `archive/frozen_cache_masked/dense_options_fix_2026-10-07.diff`: the one
  change, against the module as archived.
