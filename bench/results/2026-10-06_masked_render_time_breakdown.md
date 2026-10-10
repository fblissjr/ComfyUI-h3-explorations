# Where a masked song render's time goes: two renders of one stretch, by node and by stage (2026-10-06)

lane: masked
verdict: sampling is four fifths of a render at both canvases; the plate's VAE encode is the largest stage after it and with the conditioning makes about an eighth of what a repeat run pays without needing to; the loader's working copy and the still's span across windows cannot win; the composite runs on the processors

**What was asked.** The speed plan for the masked lane (board card
`plan-speed`) ranked its cards by argument, because a render record held one
figure for the whole song node. `MiniMaxH3AudioFreezeSong` has timed its own
stages since bbe80949, and the pipeline telemetry covers what the node cannot
see. This record is the first breakdown from both, and it re-ranks the plan.

**How to read it.** Two renders of the same thirty seconds of one source on
the shipped ref2va motion graph, twelve steps, three windows each (345, 345
and 141 frames with a 39-frame context), each with a preview prompt before it
that tracked and sampled nothing. **Render A and render B differ in three
things at once**: the file (`thrill_2160.mkv` against
`thrill_2160_pad1344.mkv`, the same stretch already at canvas size), the
canvas (1024x768 against the shipped 1344x768) and cold against warm models.
So a stage is compared with the same stage, never a total with a total. One
seed, one stretch, one card. Nothing here says what either render looks like:
nobody graded them for this record.

Every number below is in
[`2026-10-06_masked_render_time_breakdown.json`](../../data/results/2026-10-06_masked_render_time_breakdown.json),
written by `bench/masked_render_time_breakdown.py`; the tables are its
output. Seconds are wall clock: per node from the runner's websocket feed,
per stage from the node's own clock at stage ends, per step from the
sampler's progress bar in the server's log (whole seconds).

## The verdicts, card by card

| card | verdict | on what |
|---|---|---|
| `use-frozen-row-cache` | stays first: the only lever inside the four fifths | sampling is 79.6% of A and 80.8% of B |
| `build-keep-across-test-runs` | supported: source latents first, conditioning second, no keep for the track encode | 12.7% of a repeat of A and 11.6% of B is in stages that do not depend on the seed or the steps |
| `speed-conditioning-per-window` | closed across windows; its mechanism moves to the card above | conditioning is 3.6% and 2.9% of a run for all three windows, and the still is the short part of the sequence |
| `speed-loader-working-copy` | ruled out | the canvas-size copy loaded no faster on either of two servers; the loader is about a hundredth of a first run |
| `speed-decode-and-composite` | decode stays; the composite is worth a small card | the composite is 2.8% and 2.4% of a run with the card idle |
| `speed-loader-frame-cap` (opened from this) | open | the tracker is the largest cost outside sampling on a first run, and it worked on frames the plan never reads |

## Render A: the original at 1024x768

910.1 s in all. Cache state: render A: the first sampling prompt on that server, so the encoder, the DiT and the VAEs load cold; the loader, the tracker and the Masked Source come from core's cache and the mask the preview kept; no window reused from disk.

| node | class | seconds | share of the run |
|---|---|---|---|
| 74 | `MiniMaxH3AudioFreezeSong` | 908.8 | 99.9% |
| 2 | `MiniMaxH3EncoderLoader` | 0.5 | 0.1% |
| 3 | `VAELoader` | 0.3 | 0.0% |
| 75 | `PreviewAny` | 0.2 | 0.0% |
| | outside any node | 0.1 | 0.0% |

Inside the song node:

| stage | seconds | share of the run | window 1 (345 frames) | window 2 (345 frames) | window 3 (141 frames) | same on a seed or step change |
|---|---|---|---|---|---|---|
| sampling | 724.8 | 79.6% | 315.7 | 311.8 | 97.4 | no |
| source encode | 81.6 | 9.0% | 33.5 | 33.5 | 14.6 | yes |
| conditioning | 33.2 | 3.6% |  |  |  | yes |
| decode | 31.0 | 3.4% | 12.9 | 12.8 | 5.3 | no |
| composite | 25.2 | 2.8% | 10.5 | 10.3 | 4.4 | no |
| write | 11.3 | 1.2% | 7.3 | 2.9 | 1.1 | no |
| join and mux | 0.8 | 0.1% |  |  |  | no |
| track encode | 0.4 | 0.0% |  |  |  | yes |
| window setup | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 | no |
| not on the node's clock | 0.5 | 0.1% |  |  |  | |

Video tokens regenerated, by window: [1] 49.4%, [2] 34.3%, [3] 25.6%.

Sampling, by step (elapsed seconds as the progress bar prints them, whole seconds); 3 step(s) ran dense before Sol-Attn's first sparse call, on a sequence of 87692 rows:

| sampled window | seconds per step | dense step, mean | Sol step, mean | sampler |
|---|---|---|---|---|
| 1 | 37 31 35 23 23 23 23 23 23 24 23 23 | 34.3 | 23.11 | 311 |
| 2 | 34 34 35 23 23 23 23 23 23 23 23 23 | 34.3 | 23.00 | 310 |
| 3 | 9 9 9 8 8 7 8 8 7 8 8 7 | 9.0 | 7.67 | 96 |

A repeat of this run (by rule: the song node and the preview of its report run again; every node upstream is served from core's cache) pays 909.1 s again, 99.9% of this run; 115.2 s of that (12.7%) is in stages that do not depend on the seed or the step count.


What the card and the processors did in each stage of a second or more (the server process's CPU seconds; other processes are not in it):

| stage | window | seconds | GPU utilisation, mean | CPU seconds | CPU seconds per second |
|---|---|---|---|---|---|
| conditioning |  | 33.2 | 73.2 | 123.7 | 3.7 |
| source encode | 1 | 33.5 | 93.6 | 51.5 | 1.5 |
| sampling | 1 | 315.7 | 98.8 | 334.9 | 1.1 |
| decode | 1 | 12.9 | 91.3 | 17.3 | 1.3 |
| composite | 1 | 10.5 | 5.6 | 126.3 | 12.0 |
| write | 1 | 7.3 | 0.0 | 16.0 | 2.2 |
| source encode | 2 | 33.5 | 94.7 | 51.0 | 1.5 |
| sampling | 2 | 311.8 | 99.7 | 318.1 | 1.0 |
| decode | 2 | 12.8 | 92.7 | 17.1 | 1.3 |
| composite | 2 | 10.3 | 2.9 | 124.3 | 12.1 |
| write | 2 | 2.9 | 0.0 | 14.4 | 5.0 |
| source encode | 3 | 14.6 | 91.8 | 21.4 | 1.5 |
| sampling | 3 | 97.4 | 99.5 | 101.1 | 1.0 |
| decode | 3 | 5.3 | 95.1 | 6.9 | 1.3 |
| composite | 3 | 4.4 | 12.4 | 50.0 | 11.4 |
| write | 3 | 1.1 | 0.0 | 3.9 | 3.5 |

Nodes with no boundary in the telemetry (served from cache, unused, or uncacheable): 27 MiniMaxH3Resolution, 28 VHS_LoadVideoFFmpeg, 100 CheckpointLoaderSimple, 104 MiniMaxH3MaskedSource, 105 MiniMaxH3SubjectTrack, 106 MiniMaxH3MaskedPrompt.

## Render B: the canvas-size copy at 1344x768

1311.1 s in all. Cache state: render B: the second sampling prompt on that server, the models as render A left them with a tracker-only prompt between; everything upstream of the song node from core's cache; no window reused from disk.

| node | class | seconds | share of the run |
|---|---|---|---|
| 74 | `MiniMaxH3AudioFreezeSong` | 1310.8 | 100.0% |
| 75 | `PreviewAny` | 0.3 | 0.0% |
| | outside any node | 0.1 | 0.0% |

Inside the song node:

| stage | seconds | share of the run | window 1 (345 frames) | window 2 (345 frames) | window 3 (141 frames) | same on a seed or step change |
|---|---|---|---|---|---|---|
| sampling | 1058.9 | 80.8% | 462.6 | 461.5 | 134.8 | no |
| source encode | 113.9 | 8.7% | 46.6 | 46.9 | 20.4 | yes |
| decode | 42.8 | 3.3% | 17.7 | 17.7 | 7.3 | no |
| conditioning | 37.7 | 2.9% |  |  |  | yes |
| composite | 32.1 | 2.4% | 13.2 | 13.1 | 5.9 | no |
| write | 21.5 | 1.6% | 16.7 | 3.5 | 1.4 | no |
| join and mux | 2.7 | 0.2% |  |  |  | no |
| track encode | 0.4 | 0.0% |  |  |  | yes |
| window setup | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 | no |
| not on the node's clock | 0.8 | 0.1% |  |  |  | |

Video tokens regenerated, by window: [1] 37.7%, [2] 26.1%, [3] 19.5%.

Sampling, by step (elapsed seconds as the progress bar prints them, whole seconds); 3 step(s) ran dense before Sol-Attn's first sparse call, on a sequence of 113072 rows:

| sampled window | seconds per step | dense step, mean | Sol step, mean | sampler |
|---|---|---|---|---|
| 1 | 54 52 54 33 34 33 33 34 33 34 33 33 | 53.3 | 33.33 | 460 |
| 2 | 54 52 54 33 33 34 33 33 34 33 33 34 | 53.3 | 33.33 | 460 |
| 3 | 13 13 14 10 11 10 11 10 10 11 10 11 | 13.3 | 10.44 | 134 |

A repeat of this run (by rule: the song node and the preview of its report run again; every node upstream is served from core's cache) pays 1311.1 s again, 100.0% of this run; 152.0 s of that (11.6%) is in stages that do not depend on the seed or the step count.


What the card and the processors did in each stage of a second or more (the server process's CPU seconds; other processes are not in it):

| stage | window | seconds | GPU utilisation, mean | CPU seconds | CPU seconds per second |
|---|---|---|---|---|---|
| conditioning |  | 37.7 | 73.8 | 138.1 | 3.7 |
| source encode | 1 | 46.6 | 94.9 | 66.7 | 1.4 |
| sampling | 1 | 462.6 | 99.8 | 471.3 | 1.0 |
| decode | 1 | 17.7 | 95.8 | 23.1 | 1.3 |
| composite | 1 | 13.2 | 2.2 | 158.5 | 12.0 |
| write | 1 | 16.7 | 0.0 | 19.6 | 1.2 |
| source encode | 2 | 46.9 | 93.7 | 69.5 | 1.5 |
| sampling | 2 | 461.5 | 99.7 | 466.7 | 1.0 |
| decode | 2 | 17.7 | 92.6 | 22.6 | 1.3 |
| composite | 2 | 13.1 | 4.4 | 159.0 | 12.1 |
| write | 2 | 3.5 | 0.0 | 18.0 | 5.1 |
| source encode | 3 | 20.4 | 91.3 | 30.6 | 1.5 |
| sampling | 3 | 134.8 | 100.0 | 139.4 | 1.0 |
| decode | 3 | 7.3 | 95.7 | 9.6 | 1.3 |
| composite | 3 | 5.9 | 5.0 | 65.4 | 11.1 |
| write | 3 | 1.4 | 0.0 | 6.2 | 4.4 |
| join and mux |  | 2.7 | 0.0 | 0.0 | 0.0 |

Nodes with no boundary in the telemetry (served from cache, unused, or uncacheable): 1 UNETLoader, 2 MiniMaxH3EncoderLoader, 3 VAELoader, 4 VAELoader, 7 KSamplerSelect, 8 BasicScheduler, 15 LoadImage, 19 MiniMaxH3SigmaShift, 21 MiniMaxH3Sol, 27 MiniMaxH3Resolution, 28 VHS_LoadVideoFFmpeg, 50 MiniMaxH3AppendRefImage, 58 ModelAttentionBackend, 100 CheckpointLoaderSimple, 104 MiniMaxH3MaskedSource, 105 MiniMaxH3SubjectTrack, 106 MiniMaxH3MaskedPrompt.

## Stage against stage

The second window of each render, which is a full window with warm models and
no first-window effects:

| stage | A, window 2 | B, window 2 | A over B |
|---|---|---|---|
| sampling | 311.8 | 461.5 | 0.68 |
| source encode | 33.5 | 46.9 | 0.71 |
| decode | 12.8 | 17.7 | 0.72 |
| composite | 10.3 | 13.1 | 0.79 |
| write | 2.9 | 3.5 | 0.83 |

For scale, A has 0.76 of B's pixels and, by Sol-Attn's own line, 0.78 of its
rows (87692 against 113072). Sampling falls faster than the rows do.

**Per sampling step.** Three steps run dense before Sol-Attn's first sparse
call at both canvases. A full window: 34.3 s dense and 23.0 to 23.1 s with
Sol at 1024x768; 53.3 s and 33.3 s at 1344x768. The band window at 1344x768
this morning (`person_text_s1` below, another clip, the same row count) reads
53.0 s and 33.6 s, so the step time follows the sequence and not the clip.

**Cold against warm shows only at the edges.** Window 1 against window 2 in
sampling is 315.7 against 311.8 s on A (cold) and 462.6 against 461.5 s on B
(warm); the first step of A's first window took 37 s where its second
window's took 34. The whole of the song node on the band window was 561.6 s on the
first render and 559.5 s on the second.

**The first window's write is slower on both renders, and it is waiting, not
working.** 7.3 against 2.9 s on A and 16.7 against 3.5 s on B, with about the
same CPU seconds as the second window's. B's first write had no other
session's load in it. The lap also holds saving the window's latent and
core's cache emptying, and the output folder is a mount; which of those waits
is not established. It is one percent of a run or less.

## What the telemetry says of two stages

**The source encode is the video VAE on the card.** Over that stage the card
is 91 to 95% busy and the server process uses 1.4 to 1.5 CPU-seconds per
second, on every window of both renders; windows 1 and 2 agree to a few
tenths, and it scales with frames. `video_mask.window`'s fit is therefore
not the cost. It is 2.6 times the decode of the same window at both canvases.
What the masked graphs run, as facts and no further: they wire a plain
`VAELoader` on `h3_config.MODELS["video_vae"]` with no `MiniMaxH3VAEPrecision`
node; the server log's VAE line for that file reads float16; by the file's
header its encoder is 33 F16 tensors, unquantized, and its decoder 144 int8
tensors; and core's `MiniMaxH3VideoVAE` is a convolutional encoder
(`EncoderFCN3D`) with a transformer decoder (`ViT3DDecoder`). Two different
networks, nothing promoting the encoder. No cause is claimed for the ratio and
no change is proposed: the encoder's precision is a quality decision.

**The composite is the processors.** The card is 2 to 12% busy and the
server uses 11 to 12 CPU-seconds per second. `video_mask.changed_alpha` runs
every pool on host tensors at full frame size: the box over each frame's
difference, two grows, the feather's box and `pixel_alpha` twice. Its time is
an accident of where the tensors sit, and it is paid on every window of every
run, first or repeat.

## Conditioning, split by the load lines

The node's clock, as these two renders ran it, gives one figure for all three
windows: 33.2 s on A with the encoder cold and 37.7 s on B. Core logs a line
each time a model is prepared for use, and the conditioning loop prepares the
video VAE (the still's copy) and then the encoder once per window, so those
lines put the window boundaries here, from the node's start:

| | window 1 (345 frames) | window 2 (345) | window 3 (141) | VAE line to encoder line |
|---|---|---|---|---|
| A | 16.4 | 11.2 | 5.6 | 1.2, 1.2, 0.9 |
| B | 18.8 | 13.2 | 5.7 | 1.2, 1.3, 0.8 |

**Soft**: these are core's "prepared" lines, not marks at the encode, and the
last column also holds building the motion reference from the source. The
node's report carries a per-window conditioning line since 68d5f9ae, for
renders after these two; the tool reads it.

The first window costs five or six seconds more than the second whether the
encoder comes from disk (A) or was displaced by the DiT (B). By the sequence
arithmetic the still is about a twentieth of what the encoder reads and the
motion reference about three fifths: Sol's row counts less the video, the
still's 4096 rows and the 575 audio rows leave 4685 text rows on A and 5585 on
B, and the difference of 900 is the reference's fifteen vision blocks at 60
fewer tokens each for the narrower shape. That remainder is arithmetic on
three log lines and core's latent shape, not a count the encoder printed.

**So keeping the still's span across windows is worth under a second or two
per window**, against windows of five to eight minutes. The switch that would
do it exists (`keep_references` on `MiniMaxH3ReferenceConditioning`; the song
node does not pass it), and the split pass is held to core on a span shaped
like a video reference on the CPU (`bench/check_reference_encode.py`, since
0.202.3). Its values on the shipped encoder at a real video span's length
have not been compared.

## The first run of a stretch

Each render had a preview prompt before it: the same graph and inputs with
the song node on preview, so the loader and the tracker ran and nothing
sampled.

### thrill56_1024_preview

207.3 s in all. Cache state: the first prompt after the server started at 11:30:51; the song node on preview, nothing sampled; the loader and the tracker ran and the mask was kept.

| node | class | seconds | share of the run |
|---|---|---|---|
| 105 | `MiniMaxH3SubjectTrack` | 188.7 | 91.0% |
| 28 | `VHS_LoadVideoFFmpeg` | 13.2 | 6.3% |
| 104 | `MiniMaxH3MaskedSource` | 5.0 | 2.4% |
| 75 | `PreviewAny` | 0.2 | 0.1% |
| 100 | `CheckpointLoaderSimple` | 0.1 | 0.0% |
| | outside any node | 0.1 | 0.0% |

No stages: no output text carries a `seconds by stage` line.

### thrill56_pad1344_preview

209.5 s in all. Cache state: the third prompt on that server; the song node on preview, nothing sampled; the loader and the tracker ran on the padded file and the mask was kept.

| node | class | seconds | share of the run |
|---|---|---|---|
| 105 | `MiniMaxH3SubjectTrack` | 187.4 | 89.4% |
| 28 | `VHS_LoadVideoFFmpeg` | 15.5 | 7.4% |
| 104 | `MiniMaxH3MaskedSource` | 6.3 | 3.0% |
| 75 | `PreviewAny` | 0.3 | 0.1% |
| | outside any node | 0.1 | 0.0% |

No stages: no output text carries a `seconds by stage` line.

Against preview plus render, the tracker is 16.9% of a first run at 1024x768
and 12.3% at 1344x768, the loader 1.2% and 1.0%, and sampling 64.9% and
69.6%.

**The loader, four rows.** With the two review passes on the previous server
process (`thrill56_1024_review`, `thrill56_pad_review` in the JSON): the 2160
original scaled to a 1024-wide canvas loaded in 13.2 and 13.7 s, the
canvas-size copy at 1344 wide in 15.5 and 15.8 s. The two differ in how many
pixels come out; per megapixel out the copy is the cheaper by about an
eighth, which bounds what decoding the larger file costs at a second or two.

**No card named this: the tracker works on frames the plan never reads.**
`workflows/build_workflows.py` sets the loader's `frame_load_cap` to the
extent plus one whole window, 1065 frames for the shipped thirty seconds; the
song node's plan for that extent covers 753. The tracker's report says it
worked on all 1065, and the loader holds them all. What the extra frames cost
the tracker is not measured, since its time need not be linear in frames (the
padded file's review pass on the earlier server took 66.4 s where its preview
here took 187.4 s, for reasons these rows do not hold). Board card
`speed-loader-frame-cap`.

## A repeat run

A second run of the same stretch on the same server pays the song node again
and nothing upstream of it. That is the tool's rule (`SONG_ONLY_ON_REPEAT`)
and it was seen once this morning on the band window: `man_text_s1` changed
only the prompt node's subject, three nodes executed, and the song node took
as long as on the render before it.

### person_text_s1

602.6 s in all. Cache state: an earlier server process, not armed for telemetry and before the node timed its stages; not the first prompt on it; tracked fresh.

| node | class | seconds | share of the run |
|---|---|---|---|
| 74 | `MiniMaxH3AudioFreezeSong` | 561.6 | 93.2% |
| 105 | `MiniMaxH3SubjectTrack` | 32.6 | 5.4% |
| 28 | `VHS_LoadVideoFFmpeg` | 5.2 | 0.9% |
| 104 | `MiniMaxH3MaskedSource` | 2.1 | 0.4% |
| 1 | `UNETLoader` | 0.3 | 0.0% |
| 2 | `MiniMaxH3EncoderLoader` | 0.3 | 0.0% |
| 75 | `PreviewAny` | 0.2 | 0.0% |
| 3 | `VAELoader` | 0.1 | 0.0% |
| 15 | `LoadImage` | 0.1 | 0.0% |
| | outside any node | 0.1 | 0.0% |

No stages: no output text carries a `seconds by stage` line.

Sampling, by step (elapsed seconds as the progress bar prints them, whole seconds); 3 step(s) ran dense before Sol-Attn's first sparse call, on a sequence of 113072 rows:

| sampled window | seconds per step | dense step, mean | Sol step, mean | sampler |
|---|---|---|---|---|
| 1 | 55 51 53 34 34 33 34 33 34 33 34 33 | 53.0 | 33.56 | 461 |

### man_text_s1

559.8 s in all. Cache state: the next prompt on that server: only the prompt node's subject changed, so everything upstream of the song node came from core's cache.

| node | class | seconds | share of the run |
|---|---|---|---|
| 74 | `MiniMaxH3AudioFreezeSong` | 559.5 | 99.9% |
| 75 | `PreviewAny` | 0.3 | 0.0% |
| | outside any node | 0.1 | 0.0% |

No stages: no output text carries a `seconds by stage` line.

Sampling, by step (elapsed seconds as the progress bar prints them, whole seconds); Sol-Attn's first sparse line is not in this render's log (the shape was logged earlier):

| sampled window | seconds per step | dense step, mean | Sol step, mean | sampler |
|---|---|---|---|---|
| 1 | 53 53 53 34 33 34 34 33 34 33 34 33 | - | - | 461 |

Of what a repeat pays, the stages that do not depend on the seed, the sampler
or the schedule are the source encode, the conditioning and the track encode
(`REPEAT_KEEPS`, read from the node's code): 115.2 s of 909.1 on A, 12.7%,
and 152.0 s of 1311.1 on B, 11.6%. The source latents are 9.0 and 8.7 points
of that, the conditioning 3.6 and 2.9, the track encode 0.4 s. A change of
the subject's words keeps the latents and not the conditioning. **This is by
rule, not timed**: no run with those stages kept exists yet.

## What ran beside the renders

Self-reported by each session, and all inside the first two windows of A or
the first window of B:

- A, window 1 (it ended 11:41:49): an ffmpeg decode and a small torch job
  from about 11:37:30 to 11:38:24; a full check sweep from 11:37:24 to about
  11:39:40; two runs of `bench/check_reference_encode.py`, the changelog and
  index checks and a commit between about 11:36 and 11:38:32; two
  few-second torch jobs around 11:36.
- A, window 2: about a dozen three-second CPU checks from 11:42 to 11:44 and
  a five-second sweep start at 11:44:25.
- B, window 1's sampling: `bench/check_window_keep.py` twice and six copies
  of it against scratch mutations, between about 11:58:30 and 11:59:40.
- Throughout: the breakdown tool itself, a dozen times, under a second each.

Sampling is on the card and windows 1 and 2 agree to within 1.3% on A and
0.3% on B, so none of it is visible there. The stages that run on the
processors (composite, write) are the ones it could reach; the composite
agrees between windows 1 and 2 to within two tenths on both renders.

## Not established

- What a run with the latents and the conditioning kept actually takes: the
  repeat figures are sums of stages, not a timed run. (Later the same day one
  short window was run twice with both kept:
  [`2026-10-06_window_keep_matched_pair.md`](2026-10-06_window_keep_matched_pair.md).
  A full three-window run on kept values is still not timed.)
- Why the VAE's encode is 2.6 times its decode.
- What the first window's write waits on.
- What the frame cap's extra frames cost the tracker.
- Anything on another clip, another seed, a longer run, the PDD8 chain or a
  graph without a motion reference; the regenerated share alone moves from
  49% to 20% between these six windows.
- Anything about how either render looks.

## Running it again

    python3 bench/masked_render_time_breakdown.py \
        --rows <runner rows>.jsonl --history-dir <saved /history outputs, one <label>.json each> \
        --telemetry-dir <the H3_TELEMETRY directory> --server-log <the server's log> \
        --label <label> --state <label>="<the cache state, in words>" --out <record>.json

The rows, the saved outputs and the logs for this record are in the
session's working folder and the telemetry records are under the capture
root; neither is tracked. The JSON beside this file is what was read out of
them.
