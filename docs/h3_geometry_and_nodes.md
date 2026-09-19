# MiniMax H3: valid geometry, and which nodes to use

Last updated: 2026-09-14.

Everything here is read out of ComfyUI's own `comfy_extras/nodes_minimax_h3.py`
or measured on a 4090, not inferred from community lore.

## Resolution is an aspect-ratio choice, not a quality dial

`adapt_canvas()` gives you no say in how many pixels you get. It sets the
short edge to **768**, caps the area at **768 x 1344 = 1,032,192 px**, and
rounds each axis to a multiple of **32**. There is no higher resolution to
select — asking for 4K returns the same canvas as asking for 720p at the
same aspect.

That rule describes the family the checkpoint was trained on; nothing applies
it to what you type. Core's H3 conditioning nodes take `width` and `height` as
plain ints and never call `adapt_canvas` on the target canvas
(`comfy_extras/nodes_minimax_h3.py` calls it only to size reference videos),
so what is typed is what renders, in the family or not. This pack's
`MiniMaxH3Conditioning` does the same under `canvas=explicit`, and under
`from_keyframe` derives the canvas from the keyframe (below).
`MiniMaxH3Preflight` (`preflight.py`) is what reports whether a canvas sits
inside the trained family.

The cap only binds on wide or tall ratios, so a square never reaches it.
That is the whole reason aspect ratio is a cost decision:

| aspect | canvas | rows/frame | packed rows at 362f | attention cost |
|---|---|---|---|---|
| 21:9 | 1536x672 | 1008 | 91,728 | 1.00x |
| **16:9** | **1344x768** | 1008 | 91,728 | 1.00x |
| 3:2 | 1152x768 | 864 | 78,624 | 0.73x |
| 4:3 | 1024x768 | 768 | 69,888 | 0.58x |
| 5:4 | 960x768 | 720 | 65,520 | 0.51x |
| **1:1** | **768x768** | 576 | 52,416 | **0.33x** |
| 4:5 | 768x960 | 720 | 65,520 | 0.51x |
| 3:4 | 768x1024 | 768 | 69,888 | 0.58x |
| 2:3 | 768x1152 | 864 | 78,624 | 0.73x |
| 9:16 | 768x1344 | 1008 | 91,728 | 1.00x |

**1:1 costs a third of 16:9 at the same frame count.** Attention is O(S²)
and dominates the step at long clip lengths, so that is the largest single
lever available anywhere — larger than any kernel or sparsity setting.

**Portrait and landscape of the same ratio cost exactly the same.** Packed
rows are `(h//32) * (w//32)`, which is symmetric, so 1344x768 and 768x1344
both pack 1008 rows per frame. Any 16:9-vs-9:16 difference is the model's
training distribution, not geometry — test those for quality, never for
speed.

Rows per frame come from the VAE's 16x spatial downsample followed by the
model's `(1, 2, 2)` patchify, i.e. `(h//16//2) * (w//16//2)`.

## Frame counts snap to a 17k+5 grid

`align_frame_count()` rounds **up** to the next `n % 17 == 5`. Ask for 200
and you get 209; ask for 300 and you get 311. The node's own tooltip puts
the trained range at **~124 to 362**, and says longer is untested.

Valid counts near the top: **… 311, 328, 345, 362**.

Duration is `frames / 24`, so 362 frames is 15.08 s and 124 is ~5.2 s.

**Two things worth knowing at the top of the range.** Attention grows as S²
while everything else grows linearly, so the attention share rises with clip
length, and long clips are where kernel and sparsity work pays off most. The
only measured figure this repo has for that share is
[`../bench/results/2026-09-08_attention_share_bound.json`](../bench/results/2026-09-08_attention_share_bound.json),
and it is a floor over sampler time at one geometry rather than a share of a
step. Two percentages that used to sit in this sentence were withdrawn on
2026-09-08 as having no findable origin.

And late-clip identity softening at 362 is the ordinary long-clip DiT failure
at the edge of the trained range; stepping down to 328 or 345 costs
proportionally less attention *and* reduces it. **The shipped graphs render at
`h3_config.LONG_LENGTH`, which is 345, not 362** — this said "362 is the
shipped default" until 2026-09-08, which was wrong in the direction that
matters, since it implied you inherit the most expensive legal length rather
than choosing it.

## Three modality tags, not two

Recorded 2026-08-21 because no doc here stated it and two outside reviews of
this pipeline described it as a two-way split. Every row of the packed sequence
carries a modality tag, and `comfy/ldm/minimax/model.py:615` assigns three:

```python
seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0,
           "cond_audio": 2, "ref_audio": 2}
```

The tag is not diagnostic metadata. It picks the AdaLN modulation row --
`timestep_index * 3 + tag`, which is why `AdalnProj` emits three rows per
timestep and why `:719` and `:723` divide by 3. Keyframes, reference images and
target video all share the video tag; both target and reference audio share the
audio tag; only prose is text.

Two consequences worth carrying. A vision block's flanking
`<|vision_start|>` / `<|vision_end|>` are tagged **video**, not text --
`comfy/text_encoders/minimax.py` widens each embedding span by one on each side
deliberately, so the markers modulate with the pixels they wrap. And the
presentation text span is not uniform: it mixes tags, and the packing splits it
into runs rather than tagging it wholesale.

## Which nodes to use

### Required, all ComfyUI core

| node | notes |
|---|---|
| Load Diffusion Model (`UNETLoader`) | `fl2va` checkpoint for t2v/i2v, and it takes reference images too (owner, 2026-09-14); `ref2va` for reference-to-video |
| `CLIPLoader` | Qwen3-VL-32B text encoder, type `minimax`, for a native Comfy H3-format artifact. This pack's generated graphs wire `MiniMaxH3EncoderLoader` instead (below), which is the same load plus the guards core lacks; `h3_config.MODELS["clip"]` names the file |
| `VAELoader` x2 | video VAE and audio VAE are separate loaders |
| `MiniMaxH3ImageToVideo` | t2v **and** i2v — `first_frame`/`last_frame` are optional, so no image wired is text-to-video. This pack's generated graphs wire `MiniMaxH3Conditioning` instead (`conditioning.py`), which replaces it on the t2va and keyframe paths |
| `MiniMaxH3ReferenceToVideo` | reference images / video / audio → conditioning |
| `RandomNoise` → `KSamplerSelect` → `BasicScheduler` → `BasicGuider` → `SamplerCustomAdvanced` | standard custom-sampler stack |
| `VAEDecode` + `VAEDecodeAudio` → `CreateVideo` → `SaveVideo` | video and audio decode separately |

The default pair is `er_sde` / `simple`, one model eval per step, so sampler
choice here is a quality decision and not a speed one. That is not true of the
whole sampler list: `heun`, `dpm_2`, and the `2s`/`3s`/`res_Ns` families are
2-6 evals per step, and picking one silently multiplies whatever the sampler
occupies — which is nearly all of a render, decode being the small remainder
(derive it from the `*_outputs.json` records with the command in entry 28 of
[`open_experiments.md`](open_experiments.md)). `SAMPLING` in `workflows/h3_config.py` is the only
place the defaults live, and it carries the reasoning.

### Use from this repo

**`MiniMaxH3EncoderLoader`** (`h3_encoder_loader.py`) is the loader every
generated graph wires: core's own `CLIPLoader` load of the INT8 file
(`h3_config.MODELS["clip"]`), then a refusal if the load left parameters
unpopulated or the tokenizer did not realise the release's marker ids, and a
declaration of what core's preprocessing will do, read out of core and
stamped on the CLIP for the reference report and preflight. It adapts no
format and touches no weight. Its `device` input is core's `CLIPLoader`'s
(`h3_encoder_loader.DEVICES`), kept through a rebuild; for a specific GPU,
follow it with core's `SelectCLIPDevice`. The W4A16 adapter that used to stand here
(`MiniMaxH3AWQEncoderLoader`, with its `config/` snapshots and
`h3_awq_encoder.md`) was deleted on 2026-09-13 with the closed AWQ lane
(`docs/wiki/decisions.md`); `bench/check_h3_encoder_loader.py` is the
current loader's control.

**`MiniMaxH3ReferenceReport`** (`reference_report.py`) prices an ordered
reference list before anything is encoded: wire it the same references,
prompt, canvas and policies you give `MiniMaxH3ReferenceConditioning` and it
returns a picture (what the video model sees and what the text encoder sees,
per reference, then the packed sequence and the prompt's share of the text
segment) and the same as text. No VAE or encoder forward; only the tokenizer.
Every still is read twice: the video VAE's copy becomes DiT reference rows
attended on every step (`size_policy` sizes it), Qwen3-VL's copy becomes
vision tokens ahead of the prompt (`qwen_view` sizes it). The conditioner
shows the same text as an on-node preview after it runs.

**`MiniMax H3 SageAttention`** — the attention node. Replaces all 50 DiT
attention forwards with SageAttention's INT8-QK / FP8-PV kernel, and *also*
registers an `optimized_attention_override`. That second registration is
what lets Sol-Attn compose rather than silently bypassing sage. Defaults are
the intended config.

**`MiniMax H3 Conditioning`** (`MiniMaxH3Conditioning`, `conditioning.py`)
owns the canvas on the t2v and keyframe graphs, in place of core's
`MiniMaxH3ImageToVideo`. Its `canvas` combo is the choice this entry is about:
`from_keyframe` derives the canvas from the anchor keyframe, as the release
does, and ignores `width`/`height`; `explicit` uses them and cover-crops the
anchor to fit. `workflows/build_workflows.py::build_api` sets `from_keyframe`
on the keyframe graphs and `explicit` on text-to-video, where
`MiniMaxH3Resolution` owns the geometry.

Until 2026-09-14 this entry said to wire `MiniMax H3 Keyframe Resolution`
(`MiniMaxH3KeyframeCanvas`, `keyframe_canvas.py`) on every i2v and fl2v graph.
No generated graph wires it (walk `h3_config.graph_paths`): it left geometry
owned by two nodes in series, and it requires a `first_frame`, so it cannot
reach the last-frame-only signature (`conditioning.py`, module docstring). The
node is still installed, and its `resolve_keyframe_geometry` is the function
the conditioning node calls.

Why the choice matters: core's `MiniMaxH3ImageToVideo` takes `width`/`height`
as required inputs defaulting to 1344x768, and non-uniformly stretches the
first keyframe onto them. That
stretch is faithful to the reference pipeline, which also stretches the
geometry anchor and cover-crops any follower. What ComfyUI lacks is the
*default* that normally makes it a no-op: the reference derives the canvas
from the first keyframe when no size is given
(`coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3/before_encoder.py::MiniMaxH3ResizeStep` ->
`resolve_canvas_size`) and then skips the resize once the keyframe matches.
The reference's deliberate-override branch is ComfyUI's default branch.

Measured distortion at 1344x768, from
`bench/check_keyframe_canvas.py`:

| source | ratio | stretch |
|---|---|---|
| 768x1024 | 0.750 | **2.33x** |
| 1024x1024 | 1.000 | **1.75x** |
| 2560x1080 | 2.370 | 1.35x |
| 1000x700 | 1.429 | 1.23x |
| 1920x1080 | 1.778 (true 16:9) | 1.016x |
| 1344x768 | 1.750 | 1.00x |

**1344x768 is 7:4, not 16:9.** 1344/768 = 1.7500; 16/9 = 1.7778. A
genuine 16:9 source takes a 1.6% squeeze, not a no-op — small, but do not read
the table as "16:9 is safe". Round-to-32 on both axes means no H3 canvas is
exactly 16:9: `adapt_canvas(16, 9)` returns 1344x768. That is the model's canvas
rule, not a ComfyUI choice, and both nodes inherit it.

Core's stretch is silent, and every frame of the clip inherits it. Under
`from_keyframe` the conditioning node runs `adapt_canvas` — ComfyUI's own port
of `resolve_canvas_size`, sitting unused on core's keyframe path — on the
anchor and fits the keyframes onto the result inside the one node, so no size
is handed to a second node that resizes again. A lone last frame anchors the
canvas itself rather than being cropped into one chosen elsewhere.

**Only if you wire the standalone `MiniMaxH3KeyframeCanvas` by hand:** wire its
`last_frame` output only if you connected a `last_frame` input.
With no last frame, that output slot returns the *same tensor* as
`first_frame` (`keyframe_canvas.py`), because an IMAGE output cannot be
null. Wiring it anyway turns a one-anchor render into fl2va with
`last == first`: the model is anchored to return to its opening frame at
`frame_count - 1`, a spurious `<Picture 2>` enters the presentation, and a
second block of cond rows enters the packed sequence. Nothing errors.

With two keyframes the canvas comes from the first. `from_keyframe` maps to
the `match_keyframe` mode and `explicit` to `fit_to_canvas`
(`conditioning.py`, in `MiniMaxH3Conditioning.execute`). In `match_keyframe`
the first is stretched and the follower cover-cropped, as in the reference; in
`fit_to_canvas` **both** are cover-cropped, which is a deliberate divergence.
Use `from_keyframe` for anything being compared against diffusers.

Cost: output resolution now follows the input's aspect. A 9:16 still renders
768x1344, the slowest canvas on the area cap. That is the reference's own
behaviour, not an extra.

**`MiniMax H3 Provenance Stamp (bench)`** — **bench graphs only, keep it out
of shipped workflows.** Writes a JSON sidecar to `output/provenance/`
recording what a render's settings *resolved to*. It deliberately records
nothing you typed: `/history/{prompt_id}` already carries the whole graph with
every widget value and the output filenames. It records only what `/history`
structurally cannot know — the resolved sigmas, the eleven Sol closure values
(what actually ran, if anything replaced the override), the node-pack HEADs and
sage build, and the snapped frame count and canvas.

The field it exists for is `n_sparse`. That is not a setting anywhere: it is the
sigma window intersected with the sampler's schedule, so two schedulers with
identical `sol_compose` bounds can run a different number of sparse steps and
nothing in the graph, the logs or `/history` says so. Wire `SIGMAS` from
`BasicScheduler` or the field cannot be computed, and pass the sampler's
`LATENT` through it — ComfyUI orders by dependency, not graph position, so
without a real data dependency it can legally run *before* sampling.

Three states, all visible in the record rather than only in the log:
`sol: absent` (nothing installed, fine), `present` (values recorded), and
`broken` (override installed but its closure unreadable), which also raises —
most likely meaning the pack renamed parameters and `SOL_CLOSURE_KEYS` needs
updating. Joins to `/history` on `graph_sha256`, since ComfyUI does not expose
`prompt_id` to nodes.

Two cautions are in the module docstring and worth repeating: a
well-provenanced number is not a verified one, and a stamp makes invented
mechanisms *more* dangerous rather than less, because a number with a full
provenance record beside it reads as more trustworthy while carrying a wrong
causal story just as well. It records what settings resolved to, never why a
number came out the way it did.

### Sol-Attn, also from this repo

**`MiniMax H3 Sol-Attn`** (`MiniMaxH3SolAttn`, `sol_attn_h3.py`) — block-sparse
attention on the CUDA kernel. **Must come after** the sage node; it composes
with the attention patch it finds, and reversed it overwrites ours and you
silently get sage only. Settings live in `workflows/h3_config.py`. This entry
named `SolAttnPatch` from `ComfyUI-SolAttn_triton` until 2026-09-14; that pack
was deleted on 2026-08-16 (`docs/SOLATTN.md`, "The Triton node").

**Per-block Sol error** — the Triton pack's `SolAttnBlockProbe` went with it.
Its successor is `sol_block_probe.py`, armed by the `H3_SOL_PROBE` environment
variable rather than wired as a node and read by
`bench/check_sol_probe.py --record`. Every Sol call on an armed render also
runs the fallback, so its timings are void.

### Use from KJNodes

**`ModelPreviewOverrideKJ`** — taeh3 preview during sampling. Not in any
generated graph since 2026-09-14 (owner: its decodes cost GPU time on the
render it previews); add it by hand if you want to watch one.

### Skip, with reasons

**`MiniMaxH3MemoryEfficientSageAttentionPatch`** (KJNodes) — does the same
job as our node and patches the same key, so they conflict and the last one
applied wins. Ours additionally registers the attention override. Pick one;
there is no benefit to both.

**`MiniMaxLowVRAMAttention`** (KJNodes) — head chunking. Shrinks the kernel's
internal transients by the chunk count, and the measured figure is several
times what this doc and the shipped graph notes used to carry
(`workflows/h3_config.py`, the head-chunk arms beside `SAGE_NODE`). It
multiplies the attention call count by the same factor. Freed VRAM converts to
wall-clock at a low ceiling on this card — the value is in the same config
note — because weight streaming is already hidden behind compute, so it is
buying headroom you cannot spend. Take it
only if you are actually hitting OOM. As of KJNodes `35e5956` it composes
with an existing attention patch rather than conflicting.

**`MiniMaxChunkFeedForward`** (KJNodes) — at the top of the frame range the
attention peak is well above the FFN's, so chunking the FFN lowers a peak that
is not the binding one. The arms are in `workflows/h3_config.py` beside the
head-chunk ones. It is a short-clip feature; at short lengths
the two peaks are close.

**`PathchSageAttentionKJ`** — the global sage switch. It sages every
attention call in the process with no per-model guard. Prefer the
per-workflow node.

**Untested here**, not a recommendation either way: `EasyCache`,
`MiniMaxH3TurboLoRA`. (`MiniMaxH3Cache` is not installed on this machine, so
it is absent from `/object_info` rather than untested.)

`MiniMaxH3SigmaShift` was listed here as an untested third-party node until
2026-08-13. It is **core ComfyUI** (`comfy_extras/nodes_minimax_h3.py`,
picker name `ModelSamplingMiniMaxH3`). The generator wires it into every graph
except a PDD graph at the default shift, where the PDD node builds its
schedule from its own fused shift and the node would only be a knob that
breaks it (`workflows/build_workflows.py::build_api`, the
`pdd and sh == SIGMA_SHIFT` test); `bench/check_distill_settings.py`
validates its value against the LoRA each graph loads. This said it sits in
all eight shipped graphs until 2026-09-14.

### On `ResolutionSelector`

Core ComfyUI's resolution helper works from a **megapixel target**, which is
not how H3 sizes a canvas. `adapt_canvas` ignores your pixel budget entirely
and applies the short-edge and area-cap rules above. Use the table here and
type the numbers, or let the conditioning node's defaults stand.

## Node order

```
Load Diffusion Model (UNETLoader)
  -> LoRA / PDD loader               (distilled arms only; before the patches)
  -> ModelSamplingMiniMaxH3          (core's MiniMaxH3SigmaShift; anywhere
                                      before the fork; absent on a PDD graph
                                      at the default shift)
  -> Model Attention Backend         (core's, "comfy kitchen attention": the
                                      dense kernel; the sage arms wire
                                      MiniMax H3 SageAttention here instead)
  -> MiniMax H3 Sol-Attn             (ours, MiniMaxH3SolAttn; must be after
                                      the dense node)
  -> BasicScheduler / BasicGuider    (MODEL forks to both -- rewire both)
```

`workflows/build_workflows.py::build_api` is the order; read it over this
block. Until 2026-09-14 the block named `SolAttnPatch` and left out the shift
node and the assert. Until 2026-09-17 it showed our sage node as the dense
node, two days after the default moved to core's backend node, and
`SageChainAssert` after Sol, which left every generated graph that day.

MODEL forks to **two** consumers, `BasicScheduler.model` and
`BasicGuider.model`. Rewiring only the guider leaves the scheduler reading
sigmas off the unpatched model, and the render still succeeds — which is why
that mistake survives. The generated workflows drive both from one variable.
A PDD graph that is not split has no `BasicScheduler`: the PDD node emits the
schedule, and only the guider reads MODEL.
