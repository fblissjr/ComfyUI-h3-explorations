# The window keep on the card: one window rendered twice, the second on a kept latent and conditioning (2026-10-06)

lane: masked
verdict: accepted: the second run's stored window latent equals the first's on both streams and its videos are the same bytes; the source encode and the conditioning are gone from its stage seconds; one short window, one seed, one stretch

**What was asked.** `window_keep.py` (c0d1e520) keeps a song window's source
latent and its conditioning in the server's memory, and the song node reads
them when a window renders again under `reuse_windows`. Its check holds the
key on the CPU; nothing had shown a hit inside a real run. The acceptance was
written on the board card `build-keep-across-test-runs` before the pair ran:
the window latents equal, and the two stages gone from the stage seconds.

**What ran.** The shipped daily ref2va motion graph
(`workflows/daily/h3_mask_ref2va_motion_api.json` at c0d1e520), patched as the
breakdown's render A was
([`2026-10-06_masked_render_time_breakdown.md`](2026-10-06_masked_render_time_breakdown.md))
but cut to one window of 141 frames at 1024x768: `thrill_2160.mkv` from
56.0 s, twelve steps, the graph's seed, `reuse_windows` on. Run twice on one
server with two fresh filename prefixes and nothing else changed, so no stored
window matched and the window rendered both times. The seed is the same on
purpose: the question is whether the kept values give the same window, which
only an unchanged seed can show.

**Result.**

| | run 1 | run 2 |
|---|---|---|
| the song node | 131.9 s | 109.4 s |
| source encode | 14.4 s | 0.5 s |
| motion reference and conditioning | 1.0 and 6.1 s | 0.0 s |
| sampling | 98.4 s | 97.2 s |
| the report says | 1 conditioning encoded | "source latent kept from an earlier run", "conditioning kept from an earlier run" |

- **The same window.** The two stored window latents are `torch.equal` on the
  video stream and on the audio stream, and each run's window video and joined
  video are the same bytes as the other's.
- **The two stages are gone.** What is left of the source encode in run 2 is
  cutting and fitting the window's frames, which the composite needs anyway,
  and the keep's own digest.
- **The song node is 22.5 s shorter, 17% of run 1's.** The breakdown's rule
  put 16.3% of run 1 in stages that do not depend on the seed; the rest of the
  difference is sampling's own second or so between runs.
- **The write guard stayed quiet**: no `[h3-keep]` warning in the server's
  log, so nothing wrote into the kept conditioning or latent while run 1 or
  run 2 sampled.

Numbers: [`2026-10-06_window_keep_matched_pair.json`](../../data/results/2026-10-06_window_keep_matched_pair.json),
the stage tables from `bench/masked_render_time_breakdown.py` and the
comparison under `matched_pair`.

## Run 1

147.7 s in all. Cache state: a server started at 12:48:33 on c0d1e520, not armed for telemetry; models warm from earlier prompts on other windows; this window's frames and conditioning new to the server, so it tracked, encoded and stored.

| node | class | seconds | share of the run |
|---|---|---|---|
| 74 | `MiniMaxH3AudioFreezeSong` | 131.9 | 89.3% |
| 105 | `MiniMaxH3SubjectTrack` | 11.0 | 7.4% |
| 28 | `VHS_LoadVideoFFmpeg` | 3.6 | 2.4% |
| 104 | `MiniMaxH3MaskedSource` | 0.9 | 0.6% |
| 75 | `PreviewAny` | 0.2 | 0.2% |
| | outside any node | 0.1 | 0.1% |

Inside the song node:

| stage | seconds | share of the run | window 1 (141 frames) | same on a seed or step change |
|---|---|---|---|---|
| sampling | 98.4 | 66.6% | 98.4 | no |
| source encode | 14.4 | 9.7% | 14.4 | yes |
| conditioning | 6.1 | 4.1% |  | yes |
| decode | 5.4 | 3.7% | 5.4 | no |
| composite | 4.6 | 3.1% | 4.6 | no |
| write | 1.4 | 0.9% | 1.4 | no |
| motion reference | 1.0 | 0.7% |  | yes |
| join and mux | 0.2 | 0.1% |  | no |
| track encode | 0.1 | 0.1% |  | yes |
| window setup | 0.0 | 0.0% | 0.0 | no |
| not on the node's clock | 0.3 | 0.2% |  | |

Conditioning, by window: [1] motion reference 1.0, conditioning 6.1.

Video tokens regenerated, by window: [1] 58.6%.

Sampling, by step (elapsed seconds as the progress bar prints them, whole seconds); Sol-Attn's first sparse line is not in this render's log (the shape was logged earlier):

| sampled window | seconds per step | dense step, mean | Sol step, mean | sampler |
|---|---|---|---|---|
| 1 | 10 8 9 8 8 7 8 8 7 8 8 7 | - | - | 96 |

## Run 2

109.7 s in all. Cache state: the next prompt on that server: only the filename prefix changed, so everything upstream of the song node came from core's cache, no stored window matched, and the window rendered again.

| node | class | seconds | share of the run |
|---|---|---|---|
| 74 | `MiniMaxH3AudioFreezeSong` | 109.4 | 99.8% |
| 75 | `PreviewAny` | 0.2 | 0.2% |
| | outside any node | 0.0 | 0.0% |

Inside the song node:

| stage | seconds | share of the run | window 1 (141 frames) | same on a seed or step change |
|---|---|---|---|---|
| sampling | 97.2 | 88.6% | 97.2 | no |
| decode | 5.4 | 4.9% | 5.4 | no |
| composite | 4.4 | 4.0% | 4.4 | no |
| write | 1.3 | 1.2% | 1.3 | no |
| source encode | 0.5 | 0.5% | 0.5 | yes |
| join and mux | 0.2 | 0.2% |  | no |
| track encode | 0.1 | 0.1% |  | yes |
| conditioning | 0.0 | 0.0% |  | yes |
| window setup | 0.0 | 0.0% | 0.0 | no |
| not on the node's clock | 0.3 | 0.3% |  | |

Video tokens regenerated, by window: [1] 58.6%.

Sampling, by step (elapsed seconds as the progress bar prints them, whole seconds); Sol-Attn's first sparse line is not in this render's log (the shape was logged earlier):

| sampled window | seconds per step | dense step, mean | Sol step, mean | sampler |
|---|---|---|---|---|
| 1 | 9 9 9 8 8 7 8 8 7 8 8 7 | - | - | 96 |

## What this does and does not show

It shows a hit is the same render and costs nothing visible: the acceptance
the card set. It does not time a run with a changed seed, which is the case
the keep is for; the stages it skips do not read the seed, so the saving
should be the same, and that sentence is reasoning, not a run. One window of
141 frames, where the plate's encode and the conditioning are a larger share
than on a full window; the breakdown record has the shares for full windows
(8.7 to 9.0% and 2.9 to 3.6% of a run). Not shown: a conditioning hit with
the text encoder cold or displaced by other work, more than one window, the
Masked Source executing again between runs (the module says that is a miss
for the conditioning), or a graph without a motion reference. Nobody looked
at either video for this record; they are the same bytes.

The rows, saved reports and the server's log are in a session's working
folder and are not tracked.
