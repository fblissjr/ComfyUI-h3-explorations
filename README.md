# ComfyUI-h3-explorations

Tinkering and research hub for the MiniMax H3 ecosystem in ComfyUI: attention
kernels, keyframe and provenance nodes, benchmarks, and workflows. The
flagship piece today is SageAttention kernels for consumer video DiTs on Ada
(RTX 40xx / sm89), built against the
[SageAttention-ada fork](https://github.com/fblissjr/SageAttention-ada/).

Five nodes ship today: MiniMax H3 SageAttention (below), a keyframe canvas
node, a reference-image fit node, a bench-only provenance stamp, and an
attention-chain assert. See [Layout](#layout) for the full list.

A second thread runs through several of them. ComfyUI's H3 implementation
diverges from the released reference pipeline in places where the difference
is invisible: it accepts aspect ratios and clip durations the checkpoint was
never trained on, and it never upscales a reference image where the reference
does. Those limits now live in `h3_rules.py` with a citation to the reference
source for each, and the nodes enforce them rather than leaving a render that
succeeds and is quietly wrong.

## What it does

MiniMax H3 runs one unmasked self-attention per DiT block over the whole
packed `[text | cond | audio | video]` sequence: 56 heads, head_dim 128, 50
blocks. At the default canvas that sequence is about 42k rows, and attention
is roughly 61% of the model's forward FLOPs, a good fit for SageAttention's
INT8-QK / FP8-PV kernel.

The node replaces each block's attention `forward`. Compared to going
through ComfyUI's generic attention dispatch, that also lets q/k/v stay in
the layout the fused QKV projection already produces.

An earlier version of this README credited part of the win to releasing the
float q/k/v as soon as their quantized forms exist. That claim does not hold
here: `sageattn_consume` saves 0 MiB when q/k/v are three views of one fused
QKV buffer, which is exactly what `qkv_proj(x).split(...)` produces in every
H3 block (SageAttention-ada v0.7.3 measured all four arms). The peak
reduction below is real and reproduces, but what accounts for it is not
isolated yet, so treat it as a measured number without a mechanism. This
repo's bench discipline requires both arms measured before a claim ships.

## Measured

One `Attention` module at H3's config, packed sequence 41822 (fl2va at the
default 1344x768 canvas, 124 frames), RTX 4090, bf16. One arm per process.
Reproduce with `bench/bench_minimax_attn.py`:

| | per module call | peak allocation |
|---|---|---|
| stock ComfyUI attention | 389.15 ms | 3886 MiB |
| this node | 183.23 ms | 3451 MiB |
| | 2.12x faster | 435 MiB lower |

That is the whole module: QKV projection, norms, RoPE, attention, and output
projection, not just the attention kernel. So it is the number that actually
applies per block. The attention kernel alone runs about 2.7x faster.

Accuracy, measured with `bench/check_correctness.py`: mean relative error
0.0732 against the stock forward, on both the eager norm path and the fused
RMSNorm+RoPE path. SageAttention quantizes Q/K to INT8 and V to FP8, so some
divergence is by design. This result sits at the kernel's known level.

### On a real render

The above is one module in isolation. This is a full render through a
running ComfyUI at the bundled i2v template's settings (1344x768, length 73,
20 steps, `res_multistep`/`simple`, `int8_convrot` weights), warmup
discarded, arms alternating, two paired runs. Reproduce with
`bench/bench_e2e_h3.py`:

| | sampler | total render |
|---|---|---|
| sage off | 141.2 s | 151.9 s |
| sage on | 82.9 s | 93.6 s |
| | 1.70x | 1.62x |

The two paired runs agreed to within 0.3 s on every figure. The gap between
the columns is text encode plus VAE decode, which attention cannot touch.
The sampler is 93% of total at these settings, so that gap is small. Expect
a smaller end-to-end ratio at short durations, where the packed sequence is
short enough that attention stops dominating: at length 5 the same A/B
measures 1.02x, in effect nothing.

Peak VRAM during the render was ~20.6 GB of 24 GB, so this fits with room to
spare on a 4090. Longer durations or higher resolutions will close that
margin.

## Requirements

- RTX 40xx / Ada (sm89). Other architectures fall back to whatever
  SageAttention's dispatcher picks for them and are untested here.
- CUDA 12.8 or newer.
- A ComfyUI recent enough to have `comfy.ldm.minimax`.
- [SageAttention-ada](https://github.com/fblissjr/SageAttention-ada), built
  from source, new enough to provide `sageattn_consume` (v0.7.0 or newer;
  see its
  [CHANGELOG](https://github.com/fblissjr/SageAttention-ada/blob/main/CHANGELOG.md)).
  A stock `pip install sageattention` will not work. The node checks and
  tells you so.

## Use

Drop MiniMax H3 SageAttention between the model loader and the sampler. The
defaults are the intended configuration: you do not need to change anything.

Inputs:

- `model`: an H3 model. The node refuses anything else rather than silently
  doing nothing.
- `mode` (default `auto`): `auto` lets SageAttention's dispatcher choose,
  which resolves to fp8++ on a 4090, so picking `fp8++` explicitly changes
  nothing. The explicit modes exist for bisecting a suspected accuracy
  problem. `fp16 (most accurate)` is the slowest and least lossy (mean
  relative error 0.010 vs 0.069 for the fp8 modes), and it is the one mode
  that gives up the per-call memory saving, because there is no consuming
  entry point for that kernel.
- `patch_token_refiner` (default off): also patches the 2 text
  token-refiner blocks. They run over the text span only (~2k rows vs
  ~42k), so this is worth well under 1% of attention time.

If a sage call raises at runtime, the node logs once and falls back to
ComfyUI's attention for the rest of the run. The render continues.

### Longer clips are the better case

Longer clips are the better case, not the worse one. Attention grows as S^2
while everything else in the block grows as S, so the longer the clip, the
larger the share of the step sage is attacking. At length 73 the sampler
speedup is 1.70x. At 124 it is 1.91x.

Measured per-call at the 1344x768 canvas, sage `fp8++` against torch's flash
backend, with q/k/v as the three views of one fused QKV buffer that a DiT
block actually produces.

Worth being explicit about whose numbers these are: the speed below is
upstream SageAttention's kernel, the sm89 INT8-QK / FP8-PV design from
[thu-ml](https://github.com/thu-ml/SageAttention) via
[woct0rdho](https://github.com/woct0rdho/SageAttention), which the Ada fork
ships unmodified. The fork's contribution at these lengths is that it builds
for sm89 at all and stays correct past ~99,864 rows (below), not that it
runs faster.

| frames | packed rows S | sage | flash | ratio | attention share of step |
|---|---|---|---|---|---|
| 124 | 37,774 | 90.1 ms | 253.5 ms | 2.81x | ~50% |
| 209 | 63,256 | 256.0 ms | 708.2 ms | 2.77x | n/a |
| 311 | 93,836 | 556.5 ms | 1560.3 ms | 2.80x | n/a |
| 362 | 109,126 | 757.7 ms | 2107.9 ms | 2.78x | ~76% |

The kernel ratio is flat, 2.77x to 2.81x across a 2.9x span of sequence
length, and per-call accuracy is flat with it (mean rtol 0.0978-0.0985), so
nothing is traded for the extra length.

What changes is leverage, not the multiplier. A 362-frame render logs 49.66
s/it at 20 steps. Fifty blocks at 757.7 ms each account for 37.9 s of that,
so three quarters of the clock is attention. The same step on flash would
run about 118 s/it: 39.5 minutes against 16.6 minutes for the render.

Past 328 frames the addressing changes underneath you, and it is worth
knowing why that stays safe here. Above S=99,864 rows the element offsets
in the fused-QKV layout exceed int32, which in this layout silently zeroes
the tail of the output instead of raising an error.

This is an upstream defect, not one the Ada fork introduced, and it is not
obscure. Kijai independently hit the same int32 wrap in Sol-Attn's own
Triton kernels and patched it the same day, from a different direction.
Anything built on the unpatched quant kernels will hit it somewhere past
100k tokens.

Two independent fixes exist. SageAttention-ada v0.7.0 selects an int64
specialization per launch, so ordinary shapes keep int32 addressing. v0.7.1
verified it at 362 frames, where it engages and costs 0.07%, inside noise.
KJNodes vendors its own always-int64 copy of the quant kernels, which is why
its H3 patch is safe against stock SageAttention too. Either fix covers
you. What you should not do is run long clips on unpatched stock through a
wrapper that does neither.

## Stacking with other attention patches

The node registers two things: a replacement `forward` on each of the 50
DiT attention modules, and an `optimized_attention_override`.

The forward patch is the fast path and handles every call on its own. The
override exists for patches that run ComfyUI's stock forward in order to
reach their own override. That is how Kijai's
[Sol-Attn](https://github.com/kijai/ComfyUI-SolAttn_triton) works. Without
an override of ours registered, everything Sol-Attn declines after that
point (a mask, its kernel returning `None`, a kernel error) would land on
ComfyUI's default attention instead of sage. Ours chains onto any override
already present and stays in place for a later one to chain onto, so
layering is order-independent in that direction.

Apply this node before Sol-Attn, so Sol-Attn sees it and composes:

```
Load Diffusion Model → MiniMax H3 SageAttention → SolAttnPatch → BasicGuider
```

The two do not stack per call, they alternate. Inside Sol-Attn's sigma
window it runs sparse and sage is bypassed. Outside it, sage runs dense. At
H3's defaults that is 14 of 20 steps sparse.

Do not also enable KJNodes' "MiniMax H3 Mem Eff Sage Attention Patch." It
patches the same keys, so whichever node runs last silently wins, and you
will not know which kernel is running.

Measurements, settings worth using, and what did not hold up:
[Experiments on sage + Sol-Attn](./docs/SOLATTN.md). Short version: about
1.15x on top of sage with `int8_qk=True` and `morton=False`, no quality
difference that survived replication, and Sol-Attn renders measure
consistently louder on audio for reasons not yet established.

## The nodes, and when to use which

Three of these decide size, and which one you want depends only on what you
are generating from. You never need more than two.

| node | use it when | what it decides |
|---|---|---|
| MiniMax H3 Resolution | text-to-video, or reference-to-video | the video's resolution and length, chosen by shape, with the cost shown before you commit |
| MiniMax H3 Keyframe Canvas | first-frame-to-video | the same, but derived from your keyframe, because a keyframe is patchified on the video's own latent grid and the two must match |
| MiniMax H3 Reference Fit | any graph with reference images | how large each reference arrives, which sets the vision tokens it contributes. Independent of the video's resolution |

The rule that separates the last two, since both resize an image:

- A keyframe is patchified on the video's own latent grid, so its resolution
  must equal the video's. That is why the keyframe node outputs width and
  height and the reference node does not.
- A reference is patchified on its own grid, so its resolution decides how
  many vision tokens it contributes and nothing else.

The remaining two are not sizing nodes:

| node | use it when |
|---|---|
| MiniMax H3 SageAttention | always. Between the model loader and the sampler |
| Assert Sage Attention Chain | always, last in the model chain. Turns a silently mis-composed attention stack into a refused render |

Resolution divisibility is the one hard rule and it is architectural: the VAE
compresses space by 16 and the DiT patchifies that latent 2x2, so every
dimension must be a multiple of 32. The 768 short edge and the 768x1344 area
cap are different in kind -- they describe the family the model was trained
on, and core's conditioning nodes do not enforce them. 1024x1024 renders
fine and is outside that family. `docs/h3_resolutions.md` has all 95 trained
resolutions and the derivation.

## Layout

```
attention.py       kernel selection, the replacement Attention.forward
                    (with optional head chunking), and the
                    optimized_attention_override
nodes.py            MiniMax H3 SageAttention, the flagship node
assert_chain.py     Assert Sage Attention Chain -- fails the render if the
                    attention chain did not compose as intended
resolution.py       MiniMax H3 Resolution -- pick a resolution by shape and
                    see its token cost in the dropdown you pick it from;
                    says whether you are inside the trained family
keyframe_canvas.py  MiniMax H3 Keyframe Canvas -- derives the video
                    resolution from a keyframe instead of silently
                    distorting it, and enforces the trained aspect and
                    duration limits
reference_fit.py    MiniMax H3 Reference Fit -- sizes a reference image the
                    way the reference pipeline does, including upscaling
provenance.py       MiniMax H3 Provenance Stamp (bench) -- records what a
                    render's settings actually resolved to
h3_rules.py         the reference pipeline's input limits in one place:
                    aspect range, duration window, the 17n+5 frame grid
docs/               geometry/node notes and the Sol-Attn interop writeup
coderef/            gitignored symlinks to the reference implementations
                    (diffusers, DiffSynth-Studio, comfy-kitchen), which the
                    rules above and several checks cite by file and line
workflows/          generated example graphs; see build_workflows.py for
                    how they're built and h3_config.py for shared settings
bench/
  bench_minimax_attn.py      per-module speed + peak VRAM
  bench_e2e_h3.py            full render A/B against a running ComfyUI,
                             with arm, canvas, VAE and VRAM-knob axes
                             (--arms, --canvases, --video-vae, --vram-arms)
                             and a sampled peak-VRAM column
  check_correctness.py       patched forward vs the stock one
  check_override_routing.py  which calls the override sends to sage
  check_lowvram_handoff.py   the forward survives KJNodes' block hand-off,
                             and head chunking reassembles identically
  check_keyframe_canvas.py   canvas derivation, plus the aspect and
                             duration limits ComfyUI does not enforce
  check_reference_fit.py     reference sizing against both upstream rules
  check_solattn_correctness.py  Sol-Attn's Triton kernels against the
                             algorithm author's own eager implementation
  check_workflow_schema.py   saved UI graphs against a live /object_info
```

Everything from `check_override_routing.py` down runs without CUDA except
`check_solattn_correctness.py`, which needs a GPU and Triton.

Run bench arms one per process. Peak VRAM is biased by a prior arm training
the caching allocator, and `bench_e2e_h3.py` varies the seed per iteration
because ComfyUI serves an identical graph from cache and would otherwise
report an enormous fake speedup for a render that never ran.
