# The conditioning sink: which rows Sol keeps exact, and what references cost

Detail page of [`sol_attn.md`](sol_attn.md). The shipped mode is `h3_config.SOL_SINK_DEFAULT`; the modes are
`sol_attn_h3.py::SINK_CONDITIONING_MODES`.

## What the sink is

H3 packs one sequence per call: text, then any keyframe and reference segments (`cond`, `cond_audio`, `ref_img`,
`ref_audio`), then the target audio, then the target video, always in that order with target audio and video last.
Everything before `video_start` is conditioning. The node reads the layout core publishes
(`transformer_options["minimax_h3_layout"]`) and passes the kernel two ranges, in 64-row blocks:

- `sink_blocks`: the key blocks every query attends exactly.
- `sink_q`: one contiguous range of query blocks that run fully exact.

| mode | `sink_blocks` | `sink_q` |
|---|---|---|
| `exact_kv` | `[0, video_start)` | none |
| `exact_kv_and_rows` | the same | the target-audio rows only: `(audio_start // 64, video_start // 64)` |
| `exact_kv_and_all_rows` (shipped since 2026-10-02) | the same | every conditioning row: `sink_q = sink_blocks` |
| `off` | none | none |

The kernel's `sink_q` is one range, so "text and audio exact, references routed" is not expressible when reference
rows sit between them; `exact_kv_and_all_rows` is the range that covers both. The `rows` input, `per segment`, sets
text, reference and target-audio query rows apart through the kernel's `tau_map` instead
([`sol_instruments.md`](sol_instruments.md)); where the exact rows form one run they still go as `sink_q`. Which keys
every row attends exactly stays with `sink_conditioning`. Nothing has been judged with `rows` set.

**Why two kinds of row in one sink.** `final_layer` reads out only the target video and target audio. So reference and
conditioning rows matter only as keys and values for later layers, and running their queries exact is second-order;
the target audio is decoded output, first-order, and is why `exact_kv_and_rows` existed. `exact_kv_and_all_rows` was
added on 2026-09-04 because the standoff probe record
([`data/results/2026-09-03_sol_probe_base16_standoff.json`](../../data/results/2026-09-03_sol_probe_base16_standoff.json))
had the text segment disagreeing most in every block and audio least, audio being least because its queries already
ran exact. On the 2026-10-02 output panel it was the one Sol arm the owner rated fine on dialogue, where every
`exact_kv_and_rows` arm came out louder than dense, and it lost on no scene
(`bench/results/2026-10-02_sol_dense_blocks_panel.md`). Its cost is the reference rows: on a video reference that is
most of Sol's saving.

## The segments on a shipped graph, measured

`h3_probe_sol_on_api.json` as shipped on 2026-08-14 (345 frames, 1344x768, its own 216-word prompt):

```
sequence length 104,277
  video   102,816   98.6%
  audio     1,150    1.1%
  text        311    0.3%
```

`video_start = 1,461`, so the conditioning region is 23 blocks of 64, and the logged range was
`conditioning sink: KV blocks (0, 23) exact, dense query blocks (4, 23)` under `exact_kv_and_rows`, exactly as
predicted (`311 // 64 = 4`). On t2v the conditioning region is about 1.4% of the sequence.

A reference-heavy capture (362 frames, 1024x768, three image references), counted by `bench/count_packed_rows.py`
from the `PackedLayout` the model builds, reproduces the capture's 98,524 rows with zero residual:

| segment | rows | span |
|---|---|---|
| text | 7,737 (the residual; needs the encoder) | [0..7,737) |
| ref_img | 777 | [7,737..8,514) |
| ref_img | 2,500 | [8,514..11,014) |
| ref_img | 4,128 | [11,014..15,142) |
| audio | 1,206 | [15,142..16,348) |
| video | 82,176 | [16,348..98,524) |

That is 256 sink blocks of 1,539, **16.6% of all (query block, key block) pairs at every depth**, a floor `tau` cannot
reach ([`sol_error.md`](sol_error.md), "Where the exact work goes").

## References are pinned, and a video reference dominates

The pin is a per-row timestep, not a mask: `comfy/ldm/minimax/model.py` sets `VISUAL_COND_TIMESTEP = 0.999` and
`AUDIO_COND_TIMESTEP = 1.0`, applied as `max(t, aug)` over the visual and audio conditioning segments, so visual
reference rows carry 0.001 of noise augmentation and audio rows are clean. DiffSynth's H3 pipeline uses the same two
constants (`coderef/DiffSynth-Studio/diffsynth/pipelines/minimax_h3_audio_video.py:35-36`). Either way the rows are
unsparsifiable as keys.

Row counts differ by two orders of magnitude between reference types (measured in
[`../h3/references.md`](../h3/references.md), at 1344x768):

| reference | DiT rows | also adds to text |
|---|---|---|
| audio, per second | 80 | none |
| image at `match` | about 1,008 | none |
| image at `max`, 1024x1024 | 4,096 | +4,096 |
| image at `max`, 1280x720 | 7,296 | +7,296 |
| video, 960x544, 124 frames | 18,870 | about +1,700 |
| video, 960x544, 345 frames | 52,020 | +4,667 |

Images and videos pay in two segments, and the text segment is in the sink too, so a reference grows the exact
region twice.

**Share of attention forced exact** at a 362-frame 1344x768 target, by arithmetic over those counts (`S` sink rows,
`T = S + V`, `A` target-audio rows = 1,206 from `temporal_shape(362)`; exact work is `T*S` for `exact_kv` and
`T*S + A*T - A*S` for `exact_kv_and_rows`):

| configuration | `exact_kv` | `exact_kv_and_rows` | `exact_kv_and_all_rows` (shipped) |
|---|---|---|---|
| t2v, no references | 1.5% | 2.6% | 2.9% |
| 3 images at `match` | 4.1% | 5.1% | 8.1% |
| 3 images at `max` 1280x720 | 29.6% | 30.2% | 50.5% |
| 1 video ref, 124 frames | 17.1% | 17.9% | 31.2% |
| 1 video ref, 345 frames | 35.1% | 35.6% | 57.9% |

`exact_kv_and_all_rows` runs every sink row's query exact, so its exact work is `2*T*S - S²`. That column is the one
these pages called the "v1" formula and withdrew on 2026-08-16, when the node ran only the target-audio queries
exact; since 2026-10-02 it prices the shipped mode. The `exact_kv` column is derived,
not measured: `S` could not be reconstructed from committed counts for the video-reference rows (the text estimate at
heavy reference load needs the encoder), so these are arithmetic. **The conclusion does not depend on it**: the
difference between `exact_kv` and `exact_kv_and_rows` is `(A/V)(1-p)²` with `A/V` about 0.011, under a point at any
plausible `p`. Between those two modes the sink is a quality knob with a rounding-error price. `exact_kv_and_all_rows` is not: on
heavy reference load it roughly doubles the forced-exact share against `exact_kv_and_rows`, because the reference
rows' own queries are then exact as well.

## The fallback that stays

When the layout publishes no target-audio span, `exact_kv_and_rows` falls back to `sink_q = sink_blocks`. No shipped
graph reaches it: the packed layout appends the target-audio segment unconditionally, and at `audio_t = 0` it is a
zero-row segment, which collapses `exact_kv_and_rows` onto `exact_kv` rather than onto the fallback. It stays as a
guard against an upstream contract change, the class of breakage that hit this repo on 2026-08-13 when upstream
dropped `frame_count` from the same class.
