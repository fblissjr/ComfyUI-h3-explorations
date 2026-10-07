# The frozen video cache on the upper-body masked recipe, one window (2026-10-07)

lane: masked
verdict: on the window and recipe the owner called solid, with a third of the video tokens regenerating, sampling takes 218.5 s with the cache against 313.4 s without in the same warm state, a cached step 16.2 s against a Sol-Attn stock step's 23.2 s and a dense one's 34.5 s, one build and no rebuild; that is a smaller share saved than on the first window, where under a quarter of the tokens regenerate; both stock arms are the same bytes; whether the cached render is an acceptable sample of the region is the owner's eye and not said here

**What was asked.** The owner, on the board, 2026-10-07: "if you see a path
here with caching, go for it. Dont spin wheels for hours". The first
corrected pair
([`2026-10-07_frozen_cache_masked_window_fixed.md`](2026-10-07_frozen_cache_masked_window_fixed.md))
was on the lane's most favourable window. This is the second: a window that
regenerates more of the frame, on the recipe the owner uses.

**How to read it.** One window, one seed, one render per arm. The comparison
is `cache_plain` against `baseline_again`, which ran back to back with the
models resident and the conditioning kept. Seconds are from the song node's
own stage clock; a step's rate is from the sampler's bar. **Nothing here was
looked at**: no arm's picture was opened for this record. Every figure is in
[`2026-10-07_frozen_cache_masked_upper_window.json`](2026-10-07_frozen_cache_masked_upper_window.json).

## What ran

- **Window and patches:** `thrill_2160.mkv` from 67.0 s, 345 frames on the
  1024x768 canvas, the load with three cuts in it and the two typed
  corrections that assign the later shots: the `body_345_p7` arm of
  [`2026-10-06_masked_v2v_body_window_arms.md`](2026-10-06_masked_v2v_body_window_arms.md)
  on the prompt node's text. The json has each arm's patches; the typed
  description of the subject is left out of it.
- **Graphs:** the shipped upper-body graph
  (`h3_video_to_video_masked_upper_song_ref2va_motion_api.json`) for the
  stock arms, and
  `h3_probe_v2v_masked_upper_song_ref2va_motion_cache_api.json` for the
  cached arm, which differs by the cache node on the song node's model and
  the output prefix, at `h3_config.FROZEN_VIDEO_CACHE`.
- **Where:** the main server, restarted on the commit that brought the
  cache's masked use back, with its usual launch script and flags.
- **Order:** a preview (nothing sampled), `baseline`, `cache_plain`,
  `baseline_again`.

## The time

| arm | models | sampling | the whole render | steps, from the bar's rate |
|---|---|--:|--:|---|
| `baseline` | cold | 317.6 s | 403 s | three near 34.5, then nine near 23.2 |
| `cache_plain` | resident | 218.5 s | 256 s | the build 39.1, then eleven at 16.2 |
| `baseline_again` | resident | 313.4 s | 351 s | three near 34.5, then nine near 23.2 |

**With the cache, sampling is 94.9 s shorter than the stock arm beside it:
30% of sampling and 27% of the whole render.** On the first window the same
comparison gave 42% and 38%.

| | first window | this window |
|---|--:|--:|
| video tokens that regenerate | 23% | 35.2% |
| rows live, of all rows | 28,689 of 113,072 | 31,743 of 87,721 |
| a cached step against a Sol-Attn stock step | 18.8 s against 33.5 s | 16.2 s against 23.2 s |
| sampling saved | 42% | 30% |

The saving falls as the regenerated share rises, as the design says it must:
a cached step computes the live rows against every row, and here over a
third of the rows are live. Most of what is saved on this window is the
three dense steps a stock run opens with, which the cached run replaces
after its one build.

The cache's own lines: one build ("no cache yet", int4, 11.32 GiB in RAM),
eleven cached steps, no rebuild, no decline.

## What the files say

- `baseline` and `baseline_again` are the same bytes.
- `cache_plain` is a different file, as it must be.
- The composite's own line: it keeps 85% of the regenerated pixels in the
  cached arm against 83% in the stock arms. A count of pixels that differ
  from the source, not a judgement.

## What this does and does not show

- Two windows now, one seed each, one card. Both are single windows; a clip
  of several windows, where the cache builds once per window, has not run.
- Not looked at: any picture. The pair is on the output share, stacked and
  labelled, for the owner
  (`Video/compare_stacked/upper_cache_baseline_vs_cached_over_mask.mp4`).
- Not measured: how far a cached step departs from a stock step (`verify`),
  and the card's memory during a cached step. The store is host memory.
- The window crosses three cuts and the mask moves with the subject through
  them; the cache built once and did not rebuild, so a region that changes
  from frame to frame inside one window is one build.
- A window regenerating more than `frozen_video_cache.LIVE_SHARE_LIMIT` of
  its rows runs stock; this one is under it.

## Files

- `2026-10-07_frozen_cache_masked_upper_window.json`: each arm's graph,
  patches, stage seconds, sampler rates and output hash, the cache's log
  lines, and the first window's figures used above.
