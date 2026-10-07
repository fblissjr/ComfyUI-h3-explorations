# The frozen video cache on a masked window: the first run, and it saves no time (2026-10-06)

lane: masked
verdict: on one 345-frame masked window of the ref2va motion graph the cache engages cleanly (one build, eleven cached steps, no rebuild) and saves nothing: sampling takes 463.8 s with it against 462.9 s without, where the model on record predicted 40% saved; a cached step costs 36.75 s against a 33.6 s Sol-Attn stock step, because the attention call for the live rows against every row takes more than twice what the model allowed; each cached step departs from the stock step by a relative L2 of 0.10 to 0.28 on the regenerated rows; the owner retires the cache's masked use and keeps it for the audio-refine pass

**What was asked.** A masked video-to-video window keeps most of its rows,
and every step still runs all of them. `MiniMaxH3FrozenVideoCache`
(`frozen_video_cache.py`) runs the first step stock, keeps each block's
attention input, and on later steps computes only the regenerated rows and
the text. It took a masked window since 1d2d776d and had never run on one.
Its saving was a model built on the refine pass's measured floor
([`2026-10-06_frozen_cache_stage_split.md`](2026-10-06_frozen_cache_stage_split.md)),
which put this window at 40% of sampling. Board card
`use-frozen-row-cache` asked for the first masked run: an uncached baseline,
the cache on, and the cache on with `verify`, same seed, with the step times
and the stored latents compared. That run also decided half of
`q-keep-the-fast-masked-graphs`.

**How to read it.** One window, one seed, one render per arm, the card and
the processors to the run for the timed arms. Seconds are wall clock: a
stage from the song node's own clock, a step from the sampler's progress-bar
rate. **Nothing here was looked at**: no arm's picture was opened for this
record, and whether the cached render is worse, better or only different is
not said. The reading was fixed before the run, with the lane's lead:
against the predicted 40%, a saving of 35 to 45% of the baseline's sampling
meant the model holds; 15 to 35%, the cache works and the model is off;
under 15%, not worth its approximation on what is the lane's most favourable
window by live share; and more than one build, a decline, or the plain and
verify latents differing would void the run. Every figure is in
[`2026-10-06_frozen_cache_masked_window.json`](2026-10-06_frozen_cache_masked_window.json).

## What ran

- **Window:** `band_112s` of `bench/turn_metric_eye_verdicts.json`, 1344x768,
  345 frames, the graph's own seed, with the patches of `man_text_s1` as its
  row in [`2026-10-06_masked_v2v_person_text.jsonl`](2026-10-06_masked_v2v_person_text.jsonl)
  has them and an output prefix of its own.
- **Graphs:** the shipped `workflows/daily/h3_mask_ref2va_motion_api.json`
  for the baselines, and
  `workflows/h3_probe_v2v_masked_song_ref2va_motion_cache_api.json` for the
  cached arms: the same graph with the cache on the song node's model after
  Sol-Attn, at `h3_config.FROZEN_VIDEO_CACHE` (int4, no refresh, a halo of no
  tokens). Twelve steps, the first three before Sol-Attn's start.
- **Arms, in the order they ran**, through `bench/run_graph_arms.py`:
  a preview (nothing sampled), `cache_verify` (the probe graph with `verify`
  on), then `baseline`, `cache_plain`, `baseline_again`. The verify arm went
  first so that a wiring fault would cost one arm. The second baseline was
  added so the cached arm is compared with a stock arm in the same state.
- **Server:** one process for all five, started 17:05:26, default mode,
  unarmed, on the tree as it stood then, which is commit f80d1fc8 for every
  file the server loaded (that commit landed a few minutes later,
  unchanged; checked by the files' times against the server's start).
- **Cache state of each arm:**

| arm | models | source latent and conditioning | mask | stored window |
|---|---|---|---|---|
| `cache_verify` | cold: the first sampling prompt after a POST `/free` | encoded | from the preview before it, through core's cache | none reused |
| `baseline` | cold: the first prompt after another POST `/free`, made by the card's owner between the two stages, with five tracker processes of theirs on the card after it | encoded | the mask the preview kept on disk | none reused |
| `cache_plain` | resident | kept in the process from `baseline` | core's cache | none reused |
| `baseline_again` | resident | kept | core's cache | none reused |

So `cache_plain` and `baseline_again` are the like-for-like pair.

## It ran as a cached run

- Every arm's report: 23.0% of the window's video tokens regenerate, 345
  frames written, no stored window reused. The regenerated grid rebuilt
  offline from the kept mask gives the same share, 23,679 of 102,816 video
  rows.
- Both cached arms: one build, for "no cache yet", with 28,689 of 113,072
  packed rows live (the regenerated video rows and the text), int4, 14.60
  GiB in RAM; then eleven cached steps. No rebuild, no decline, no warning
  but the expected one about the malloc graph. The content guard, which
  rebuilds when a kept latent is not bit-identical between a sampler's
  steps, did not fire: that was the open risk.
- The baselines: no cache line.

## The time

| arm | sampling stage | steps, from the bar's rate |
|---|--:|---|
| `baseline` (cold) | 466.8 s | 57.0, 53.3, 53.4, then nine at 33.5 to 33.6 |
| `cache_plain` | 463.8 s | the build 58.9, then eleven at 36.7 to 36.8 |
| `baseline_again` | 462.9 s | 53.5, 53.4, 53.4, then nine at 33.6 |

**The cache is 0.9 s slower over the window than the stock arm beside it,
0.2% of sampling.** Step by step: a cached step is 36.75 s. Where stock runs
dense, on steps 2 and 3, that is 16.7 s saved each; where stock runs
Sol-Attn, on the other nine, it is 3.2 s lost each; and the build costs 5.4 s
over a stock step. The three cancel.

The two baselines differ by 3.9 s of sampling, nearly all of it in the first
step, which in the cold arm holds the model's load.

**Do not read the bar's elapsed column in a cached arm.** It prints 441 s at
the last step of `cache_plain`, which would read as a twentieth saved. The
column runs backwards after the build (the refine record noted the same).
The rates add up to the song node's clock, 58.9 plus eleven at 36.75 is
463.2 of 463.8 s, and the elapsed column does not.

## Why: the attention call

`verify` splits each cached step's time by stage. A timed pass is slower
than a plain one, since every lap waits for the card, so read the shares:

| stage | seconds of a timed pass |
|---|--:|
| attention | 29.2 |
| qkv, every row | 2.6 |
| store to card | 2.5 |
| live rows | 2.4 |
| between blocks | 1.7 |
| the pass | 38.4 |

The same on every one of the eleven steps, to a tenth. The model on record
priced the dense rectangle of live queries against every key by assuming the
dense kernel's time follows the number of query rows, and got a cached step
of 19.7 s. Three quarters of the measured step is that one call. With 28,689
queries against 113,072 keys, a quarter of the square, it is between two and
a half and three times a quarter of what a dense stock step's attention
costs by that record's own figure scaled to this length. The call
goes to the kitchen backend's int8 attention, the kernel the dense stock
steps use. Why a rectangle costs that much on it was not looked at; the
refine record's note on a per-call cost that does not follow the query count
points the same way.

**Corrected 2026-10-07: the call did not go to the kitchen kernel.** This
section said it did, and nobody had checked. The cached block removes the
attention override that carries the kitchen backend, the checkpoint names
none of its own, and the call falls to torch's kernel; timed alone, torch's
kernel reproduces the 29.2 s above and the kitchen kernel on the same
rectangle is well under half of it
([`2026-10-07_frozen_cache_rectangle_kernel.md`](2026-10-07_frozen_cache_rectangle_kernel.md)).
So the times below are of a cache on the wrong kernel, the sentence after
this note and "The band is the lowest one" describe that run and not the
design, and `verify`'s table compares exact attention with Sol's sparse
int8 attention as well as cached rows with live ones.

A stock Sol-Attn step is sparse, and at this length it is cheaper than the
cached step's dense rectangle. That is the whole result: the cache replaces
a step that Sol-Attn had already made cheap.

## What `verify` read

Each cached step against the same step run stock from the same state, on
the regenerated video rows: relative L2 over all of them, over the ring
(within one token of a kept row) and the interior, and once more with the
text rows cached too.

| step | video sigma | cosine | relative L2 | ring | interior | text cached too |
|--:|--:|--:|--:|--:|--:|--:|
| 2 | 0.9925 | 0.9628 | 0.280 | 0.185 | 0.311 | 0.300 |
| 3 | 0.9837 | 0.9653 | 0.265 | 0.183 | 0.292 | 0.302 |
| 4 | 0.9730 | 0.9686 | 0.251 | 0.180 | 0.276 | 0.272 |
| 5 | 0.9601 | 0.9767 | 0.216 | 0.167 | 0.235 | 0.228 |
| 6 | 0.9440 | 0.9792 | 0.205 | 0.168 | 0.219 | 0.206 |
| 7 | 0.9231 | 0.9834 | 0.183 | 0.158 | 0.193 | 0.189 |
| 8 | 0.8957 | 0.9874 | 0.159 | 0.146 | 0.165 | 0.167 |
| 9 | 0.8575 | 0.9894 | 0.146 | 0.141 | 0.148 | 0.152 |
| 10 | 0.8000 | 0.9918 | 0.129 | 0.134 | 0.126 | 0.131 |
| 11 | 0.7064 | 0.9934 | 0.115 | 0.131 | 0.107 | 0.116 |
| 12 | 0.5239 | 0.9947 | 0.104 | 0.132 | 0.088 | 0.107 |

The ring is 7,062 rows and the interior 16,617.

- **It falls as the run goes on.** Nothing here asks for a refresh: the
  cache does not age badly, it starts far and comes closer.
- **The interior is further off than the ring until step 9**, the reverse of
  what the design expected (rows drawn against stale neighbours first). The
  ring settles near 0.13.
- **Live text buys two points at step 2 and nothing later.**
- These are much larger than the refine pass's audio (0.02 to 0.03). They
  are per-step departures from one state, not a trajectory, and no bar was
  set on them: none exists for video.

## The stored latents

Compared on the CPU, on the grid above:

- `baseline_again` equals `baseline` bit for bit, video and audio. The stock
  render is deterministic across a cold and a warm run, an encoded and a
  kept source latent.
- `cache_plain` equals `cache_verify` bit for bit. `verify` hands the cached
  output on and disturbs nothing, so its lines describe the plain arm.
- The kept video cells and the whole audio stream are bit-equal in all four
  arms. The cache moves no kept row.
- `cache_plain` against `baseline` on the regenerated cells: relative L2
  0.379, cosine 0.928; ring 0.264, interior 0.423. Two trajectories, which
  part by themselves: this says the cached render is a different sample of
  the region, not how it looks.

The decoded frames agree with that, and reach back a day's worth of commits:
both baselines and the morning's render of the same arm
(`man_text_s1`) decode to the same video, one hash over every frame, and the
two cached arms decode to another.

## What it means

- **The band is the lowest one**: no time saved, with an approximation of a
  tenth to a quarter per step.
- **Decided by the owner, 2026-10-06, on this result, as relayed by the
  lane's lead:** the cache's masked use is retired, with this record as the reason and its code kept under
  `archive/`, where nothing imports it, registers it, walks it or wires it
  into a graph; the node stays for the audio-refine pass, where it is
  measured to pay ([`2026-09-25_frozen_cache_s1.md`](2026-09-25_frozen_cache_s1.md),
  [`2026-10-06_frozen_cache_stage_split.md`](2026-10-06_frozen_cache_stage_split.md)).
  The lane's lead had called it the same way an hour earlier: not a speed
  lever on the Sol-Attn graphs at this live share. The retirement is a later
  commit; at this one the masked gate and its probe graph are still in the
  tree. The one lever this run names, for anyone who reopens it: the kernel
  the rectangular attention call runs on.
- The other rows of the model's table (the test clip's windows, at a quarter
  to two fifths saved) rest on the same assumption and fall with it.
- Where it could still pay, by this run's own steps and not tested: a graph
  whose stock step is dense. On steps 2 and 3 here the cached step is 16.7 s
  under the stock one.
- For `q-keep-the-fast-masked-graphs`: the cache takes nothing off the base
  chain's sampling, so that question is decided by its other test. The
  cache is not the base chain's alone in any case: the fast graph keeps the
  same rows and would meet the same step.
- `frozen_video_cache.LIVE_SHARE_LIMIT`'s provenance said the first masked
  run's step times would replace the model. They have, and at a live share
  of a quarter, well under the limit, the cached step already loses to a
  Sol-Attn stock step.

## What this does and does not show

- One window, one seed, one render per arm, one card. The like-for-like
  timing is one pair.
- Not looked at: any picture. The renders over their regenerated regions
  are on the share for the owner
  (`Video/mrhand/cache/band_baseline_00001_with_mask.mp4`,
  `band_cache_plain_00001_with_mask.mp4`, and a side-by-side of the two
  under `Video/compare_stacked/`).
- Not tried: a halo, a refresh, another precision, a graph without Sol-Attn,
  the fast graph, more than one window (so no previous window's tail in the
  kept rows).
- The split is of a timed pass. A plain cached step is 36.75 s where the
  timed one is 38.4 s.
- Memory: the store was 14.60 GiB in RAM and the machine had the room; the
  card's memory during a cached step was not recorded.
- The server ran the tree of 17:05, which equals f80d1fc8 for everything
  the server loaded. The cache's module and both graphs were as committed.

## Files

- `2026-10-06_frozen_cache_masked_window.json`: each arm's row, its report's
  stage seconds and state lines, the cache's build, verify and stage lines
  and the sampler's bar parsed from the server's log, the latent comparison,
  the decoded-frame hashes, and the prediction and bands as fixed before the
  run. The server's log span and each arm's history are kept untracked by
  the session that ran it.
