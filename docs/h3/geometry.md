# Canvas and length: what to type, what is legal, what it costs

The one home for H3's canvas, its frame grid and its length rule. The values are code; this page says what they
mean and points at the constant. Where it and the code disagree, the code is right. The cost tables (every canvas,
every length, the Sol-Attn floor, the int32 crossing) are on [`geometry_cost.md`](geometry_cost.md).

## The answer, in one screen

**What to type: a shape, not a resolution.** `MiniMaxH3Resolution` (`resolution.py`) lists every trained canvas by
shape band, each with its tokens per frame and its attention cost against 1344x768, and `custom` for anything
else. The generator does not ask at all: `workflows/h3_config.py::CANVAS_TIER` picks one of `CANVAS_TIERS`, so
"render this cheaper" is one edit and a rebuild, and the comment above it says which tier is for what. Keyframe
graphs take the canvas from the keyframe (`MiniMaxH3Conditioning`, `canvas=from_keyframe`; "A canvas from a
keyframe" below).

**The legal set, and why.** Core's `adapt_canvas` (`comfy_extras/nodes_minimax_h3.py`) turns an aspect ratio into a
canvas by three rules: the short edge starts at 768, the area is capped at 768 x 1344, and each axis rounds to a
multiple of 32 (`h3_rules.CANVAS_MULTIPLE`). Over the trained aspect range (`h3_rules.MIN_ASPECT_RATIO` to
`MAX_ASPECT_RATIO`) that gives exactly 95 canvases, which `resolution.py::_all_resolutions` sweeps rather than
lists. Only the 32 is a hard rule of the architecture; the area cap is the vendor's and unenforced here; the 768 is a
published recipe ("Why there is no resolution setting").

**The frame grid.** A frame count rounds **up** to the next `17n + 5` (`h3_rules.snap_length`, `FRAME_FACTOR`,
`FRAME_REMAINDER`): the video VAE encodes 17 frames a chunk and keeps 5 latents. Its latent frame count is `5n + 2`.
H3 runs at `h3_rules.FPS`. One frame is the single exception to the grid, the parked image path
([`image_editing.md`](image_editing.md)).

**The length rule.** The ceiling is `h3_rules.MAX_LENGTH`, the longest length H3 was trained on, by the owner's
decision of 2026-10-15, and there is no other cap. **Longest is not best**: attention grows with the square of the
sequence, so a window is as long as its shot needs and no longer ("The length axis").

**What it costs.** Tokens per frame are `(W/32) x (H/32)`; video tokens are that times the latent frame count;
attention goes as the square of the sequence, and video tokens dominate it. Canvas is the largest single cost lever
in this repo, larger than any kernel or sparsity setting, and length is the second. A square canvas costs a third of
1344x768 at the same length; portrait and landscape of one ratio cost exactly the same. Exact rows for a graph:
`bench/count_packed_rows.py --length N --canvas WxH`, or `MiniMaxH3Preflight` on the graph itself.

## Three grids, and the 32

"Latent" names two grids in this repo, and confusing them is a recurring mistake.

```
  pixels        1344 x 768    what you type
     |  the video VAE compresses space by 16
  VAE latent      84 x 48     what the VAE emits
     |  the DiT patchifies (1, 2, 2)
  token grid      42 x 24     what attention and Sol-Attn see: 1008 tokens per latent frame
```

16 x 2 = 32. Divisibility by 16 alone leaves an odd latent axis, and the 2x2 patchify cannot tile it, which is why
`MiniMaxH3Resolution` refuses a canvas that is not a multiple of 32. Tokens per frame are the product of the
**token** grid. The patchify is `patchify_video` in `comfy/ldm/minimax/model.py`; the packed sequence is that file's
`PackedLayout`.

Qwen3-VL's vision tower also lands on 32 (`patch_size=16`, `merge_size=2`), on its own grid after its own resize.
That governs a still entering the text encoder and places no constraint on the video canvas. The two meet in one
graph because a keyframe or reference still is read twice, by the VAE and by the encoder; only the VAE's copy is
tied to the video's canvas ([`conditioning.md`](conditioning.md), section 1b).

## Why there is no resolution setting

`adapt_canvas` gives no say in how many pixels you get: asking for 4K returns the canvas 720p returns at the same
ratio. There is no quality dial, only a shape choice. The three rules are not the same kind of thing:

| rule | where it holds | what enforces it |
|---|---|---|
| each axis a multiple of 32 | the architecture | the VAE and the patchify; `MiniMaxH3Resolution` refuses otherwise |
| area at most 768 x 1344 | sglang scales any larger canvas down by `sqrt(MAX_PIXELS / area)`, no override (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/resolved_plan.py:159-169`) | nothing in ComfyUI |
| short edge 768 | sglang warns on another value and renders it (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py:41-53`); its own warning calls 768 the only short edge MiniMax publishes recipes and reference outputs for | nothing anywhere; a coverage claim, not a training-data one |

No config shipped with the weights carries a canvas geometry. diffusers applies its canvas rule only when height and
width are omitted and checks an explicit pair for 32-divisibility alone
(`coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3/before_encoder.py:391-397,424-427`). Because the area cap re-imposes itself,
asking for a bigger short edge at 16:9 still returns 768; nearer square it does not.

**Core's conditioning nodes never call `adapt_canvas` on the target canvas.** `MiniMaxH3ImageToVideo`,
`MiniMaxH3ReferenceToVideo` and `EmptyMiniMaxH3LatentAV` take `width` and `height` as plain integers at `step=32`,
so whatever multiple of 32 is typed is what renders, in the trained family or not (1024x1024 is legal and outside
it). Core applies `adapt_canvas` only to size reference videos. This pack's `MiniMaxH3Conditioning` does the same
under `canvas=explicit` and derives the canvas under `from_keyframe`. `MiniMaxH3Resolution` and
`MiniMaxH3Preflight` say which side of the family a canvas is on rather than refusing it. Core's own
`ResolutionSelector` works from a megapixel target, which is not how H3 sizes a canvas; do not use it here.

### Two surprises

- **The short edge is not always 768.** It is 768 only while the area cap does not bind, from roughly 3:4 to 7:4.
  Wider or taller, the cap takes over: 21:9 returns 1536x672.
- **1344x768 is not the cost ceiling.** Rounding each axis to 32 can land above its token count; the worst canvas
  in the set is 7.3% over it, for no extra pixels. The fourteen common ratios
  ([`geometry_cost.md`](geometry_cost.md), "The common ratios") avoid this.

### 1344x768 is 7:4, not 16:9

1344/768 = 1.75; 16/9 = 1.778. Round-to-32 on both axes means no H3 canvas is exactly 16:9, and `adapt_canvas(16, 9)`
returns 1344x768. A genuine 16:9 source takes a 1.6% squeeze on that canvas, not a no-op. This is the model's canvas
rule, not a ComfyUI choice. A 16:9-against-9:16 difference is the training distribution, never speed: test those for
quality only.

## A canvas from a keyframe

Core's `MiniMaxH3ImageToVideo` takes `width`/`height` as required inputs defaulting to 1344x768 and stretches the
first keyframe onto them, non-uniformly and silently, and every frame of the clip inherits the stretch. That stretch
is faithful to the reference pipeline, which also stretches its geometry anchor; what ComfyUI lacks is the default
that makes it a no-op: the reference derives the canvas from the first keyframe when no size is given
(`coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3/before_encoder.py::MiniMaxH3ResizeStep` ->
`resolve_canvas_size`).

`MiniMaxH3Conditioning` owns the canvas on the text-to-video and keyframe graphs. Its `canvas` input:

- **`from_keyframe`** runs `adapt_canvas` on the anchor and fits the keyframes onto the result inside the one node
  (`keyframe_canvas.py::resolve_keyframe_geometry`, the one implementation). The anchor is the first frame, or the
  last frame when it is the only one, so a lone last frame keeps its whole picture. With two keyframes the first is
  stretched and the follower cover-cropped, as in the reference. The output follows the input's aspect: a 9:16 still
  renders 768x1344. The generator sets this on the keyframe graphs; use it for anything compared against diffusers.
- **`explicit`** uses `width`/`height` and cover-crops **both** keyframes, a deliberate divergence. The generator
  sets it on text-to-video, where `MiniMaxH3Resolution` owns the geometry. On an fl2va graph under `from_keyframe`
  the carried `width`/`height` are inert.

The stretch core applies at 1344x768, measured by `bench/check_keyframe_canvas.py`:

| source | ratio | stretch |
|---|---|---|
| 768x1024 | 0.750 | 2.33x |
| 1024x1024 | 1.000 | 1.75x |
| 2560x1080 | 2.370 | 1.35x |
| 1000x700 | 1.429 | 1.23x |
| 1920x1080 | 1.778 (true 16:9) | 1.016x |
| 1344x768 | 1.750 | 1.00x |

## The length axis

Any length on the `17n + 5` grid from the shortest up to `h3_rules.MAX_LENGTH` is valid. Ask for 200 and you get
209; ask for one past the ceiling and the snap lands a whole grid step over it, which is why every check runs on the
snapped count, never the request (`h3_rules.duration_in_range`). `h3_rules.describe_length` prints the snap.

**The ceiling.** `h3_rules.MAX_LENGTH` (read through `h3_rules.max_legal_length`, never derived from seconds) is the
longest length H3 was trained on: the owner's decision of 2026-08-16 and again of 2026-10-15, on thin evidence that
the `h3_rules` docstring spells out. `workflows/h3_config.py` reads it as `MAX_LENGTH`, and a long render is
`LONG_LENGTH`, the ceiling. The duration floor is `h3_rules.MIN_DURATION` (`min_legal_length`). Core's node accepts
far longer lengths and enforces only the grid, by snapping.

**Longest is not best.** Attention grows with the square of the sequence (the "attention" column of the length table
in [`geometry_cost.md`](geometry_cost.md)), so a window takes the frames its shot or its text needs and no more;
reaching for the ceiling without a reason only buys time on the card. The latent frame count, `5n + 2`, is what
prices the sequence, and it does not track duration linearly. Late-clip identity softening near the ceiling is the
ordinary long-clip failure at the edge of the trained range, and a shorter length reduces it as well as the cost.
The attention share of a step rises with length; the only measured figure for it is a floor over sampler time at one
geometry, [`../../data/results/2026-09-08_attention_share_bound.json`](../../data/results/2026-09-08_attention_share_bound.json).

**diffusers stops one grid step earlier.** Its `max_duration` is a hard-coded 15.0 s applied after the rounding, and
no on-grid count lands on exactly 15.0 s, so it refuses the ceiling. That is a fact about diffusers, not the model:
`h3_rules.reference_would_emit` answers whether a graph also runs there, and its answer is portability, not
legality.

**A chained shot is longer in sequence than its frame count says.** The packs that pin context from a previous clip
add conditioning rows, so a chained window runs longer than the same clip alone at the same frame count.

### Whole on the audio clock: a best practice, not a rule

H3 lays its audio at `h3_rules.AUDIO_LATENT_RATE` latents a second against `h3_rules.FPS` frames, and core gives a
clip `round(frames / FPS * AUDIO_LATENT_RATE)` audio latents (`comfy_extras/nodes_minimax_h3.py::temporal_shape`).
The two clocks agree exactly only when `frames * AUDIO_LATENT_RATE` is a multiple of `FPS`; on the video grid that is
every third length, 39 the first and then every 51 frames (`h3_rules.on_both_clocks`). Any other grid length, the
ceiling among them, ends with the sound off its last frame by a third of one audio latent.

Every grid length up to the ceiling stays valid, and a window that holds its whole shot takes any of them
(`loop_plan.stands_alone`). Where the clock is enforced is a seam: two chained windows meet only on a length whole on
both clocks (`h3_rules.chain_lengths`, from `SHORTEST_CHAINED`), because there the next window's sound starts where
the last one's ended (reasoned, not measured). The longest such window is `h3_config.LONG_WINDOW`, one grid step under
the ceiling, worked out and never typed. So: prefer a length on both clocks when the shot leaves the choice free; do
not pad or cut a shot to reach one. Whether the remainder can be heard or seen on a mouth is not measured. The rule
came from a community report of 2026-10-17, checked that day against `temporal_shape`.

## Where the values live

| what | where |
|---|---|
| short edge, area cap, `adapt_canvas`, `align_frame_count`, `temporal_shape` | core, `comfy_extras/nodes_minimax_h3.py` |
| patchify, packed layout | core, `comfy/ldm/minimax/model.py` (`patchify_video`, `PackedLayout`) |
| aspect range, grid, `MAX_LENGTH`, duration floor, audio clock, chain lengths, diffusers portability | `h3_rules.py`, which cites the reference for each |
| the trained-family dropdown and its cost labels | `resolution.py` (`MiniMaxH3Resolution`) |
| the shipped canvas and lengths | `workflows/h3_config.py` (`CANVAS_TIER`, `CANVAS_TIERS`, `ASPECTS`, `LENGTH`, `LONG_LENGTH`, `LONG_WINDOW`) |
| keyframe canvas | `keyframe_canvas.py::resolve_keyframe_geometry`, called by `conditioning.py` |
| what a graph will cost before it runs | `MiniMaxH3Preflight` (`preflight.py`), `bench/preflight_graph.py`, `bench/count_packed_rows.py` |
