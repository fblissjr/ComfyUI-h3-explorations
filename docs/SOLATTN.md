# Stacking with Sol-Attn on MiniMax H3

Measured evaluation of [Sol-Attn](https://github.com/kijai/ComfyUI-SolAttn_triton)
layered on this node, on a 4090 at H3's 1344x768 / length 124 / 20-step
configuration. Sparsity and quantization attack different things, so the
question is whether they compose usefully. They do, modestly.

Everything below is a single-machine, single-workload result. Sol-Attn is
tuned for more than H3 and the numbers here say nothing about it elsewhere.

The sage baseline is [SageAttention-ada](https://github.com/fblissjr/SageAttention-ada),
not stock SageAttention -- its
[CHANGELOG](https://github.com/fblissjr/SageAttention-ada/blob/main/CHANGELOG.md)
lists what differs. It has not been measured against stock, so these ratios
mean "on this fork, this model, this box".

## Status, 2026-08-06

Timing has since been measured at 362 frames and is reported below. **The
quality verdict is reopened.** The evaluation on this page judged quality by
comparing still frames, which cannot see a temporal artifact, and a failure
mode has since turned up that only shows in motion: at higher `tau`, a small
persistent object can dissolve partway through a clip and be replaced by
something else, with no recovery. Sections marked RETRACTED or REOPENED
below have not been re-measured yet.

Working configuration lives in `workflows/h3_config.py` as
`SOL_RECOMMENDED`, with the reasoning for each knob in the comment above it.
The generated graphs are in `workflows/`.

## The frontier

Sampler time, same seed, warmup discarded, arms alternating:

| config | sampler | vs nothing | vs sage | quality |
|---|---|---|---|---|
| no sage | 342.1 s | 1.00x | — | reference |
| **sage** | **178.7 s** | **1.91x** | 1.00x | indistinguishable by eye |
| sage + sol (`int8_qk`) | 155.0 s | **2.21x** | 1.15x | no difference held up |
| sage + sol, widened window | ~150 s (projected) | ~2.3x | ~1.2x | unmeasured, expect degradation |

**Sage is the easy call**: a large win with no perceptual cost and one
node. **Sol-Attn is a judgement call**: another 15% for a second node, a
sigma window to tune, and a sample that differs from the same seed without
being better. Below about 20k packed rows it will do less, because
attention stops dominating.

### The 15% is a floor, and the length it was measured at is why

This table is length 124, where attention is ~50% of the step. Sol-Attn
only attacks attention, so its ceiling is that share — and the share climbs
with sequence length, because attention is quadratic in S and the rest of
the block is linear:

| frames | packed rows S | attention share of step |
|---|---|---|
| 124 | 37,774 | ~50% |
| 362 | 109,126 | ~76% |

Backing the 1.15x out through Amdahl at a 50% share implied Sol-Attn made
attention itself about 1.35x faster, which projected **~1.25x** at 362
frames.

**That projection was low.** Measured at 362 frames, 1344x768, 20 steps,
same seed, warmup discarded:

| arm | sampler | vs sage |
|---|---|---|
| sage only | 991.0 s | 1.00x |
| sol, no int8, tau 1.2 | 827.9 s | 1.20x |
| sol + `int8_qk` + `int8_pv`, tau 1.2 | 714.9 s | 1.39x |
| + tau 1.6 | 648.2 s | 1.53x |
| + tau 2.0 | 602.9 s | 1.64x |

Sparsity and int8 are separate levers of comparable size (1.20x, then
another 1.16x). The gain grows with length exactly as the attention share
predicts, so the 1.15x at 124 frames really was a floor.

**Do not read the tau 1.6 and 2.0 rows as recommendations.** They are timing
only, and both sit above the point where the artifact described under
Quality appears. The rows are kept because the cost of each quality knob is
only legible against them.

These arms ran against a Sol-Attn build from 2026-08-05 whose exact commit
was not recorded, which given how fast that node moves is a real gap -- see
the build note below.

### Three Sol-Attn changes postdate this evaluation

Measured Aug 4; Sol-Attn shipped these Aug 4-6, so nothing below is
reflected in the table above.

- **`int8_pv` (new, defaults on).** Runs the exact branch's P@V in INT8 as
  well as QK. Upstream's note is that PV and QK cost the same, so this is
  "the other half of the int8 win" — our 1.15x was `int8_qk` only.
- **`SolAttnBlockProbe` + `dense_blocks`.** The probe computes every
  attention call both sparse and dense and logs per-block relative error
  worst-first; the worst blocks then go in `dense_blocks` to stay exact.
  This is the direct instrument for the audio question left open below —
  it answers per block which approximations actually reach the output,
  instead of inferring it from listening.
- **`morton_curve="2d_frame"` (new default).** Z-orders within each frame
  and leaves frame order alone, motivated by H3's frame spacing being
  non-uniform. Our evaluation ran morton off, and the failure mode it
  addresses is length-dependent, so this is worth revisiting at 362 frames
  specifically.

Upstream also fixed an int32 overflow in Sol-Attn's own Triton kernels on
Aug 4 (`9cab9a0`) and a Morton corruption at certain sizes (`e353f6d`).
Both are in the size range long clips reach, so any long-sequence
Sol-Attn measurement needs a build at or after those commits to mean
anything.

## The CUDA node, 2026-08-14

A second implementation now exists: CUDA kernels on kijai/comfy-kitchen's
unmerged `sol_attn` branch (`c04ef20`), driven by a single-file node in
`internal/refs/sol_attn_minimax.py` (`node_id="SolAttnMiniMax"`). Everything
on this page above was measured against the Triton pack and none of it
transfers automatically.

**Nothing in this section is a render measurement.** It is what the options
do, from the node source, the kernel signature, and the eager reference, plus
three things measured on a 4090 by `bench/check_solattn_correctness.py` and
one ad-hoc kernel smoke. The quality and timing questions this page exists to
answer are all still open for the CUDA path.

### Installing it is not a drop-in

`comfy_kitchen.sol_attn` ships on no wheel. The branch build declares version
`0.2.31`, byte-identical to the PyPI version ComfyUI pins in
`requirements.txt`, so the fork build and the stock wheel are
indistinguishable to `pip list` and a `--force-reinstall` swaps the kernel out
with no error and no log line -- the node then falls back to dense on every
call and the render merely gets slower. The local build here is tagged
`0.2.31+sol.c04ef20` for that reason, and `bench/check_sol_kernel.py` is what
notices if it is lost.

`node_id="SolAttnMiniMax"` is provisional. Upstream's position is that a
proper node waits on global attention timestep scheduling landing in core, so
**do not bake this node id into a shipped graph** -- see the one rule in
CLAUDE.md. A bench arm is code we can rename; a saved graph is not.

### What each option does

The four marked NEW have no counterpart on the Triton node. Three Triton
options are gone: `int8_qk`, `int8_pv` and `use_tma`. The CUDA kernel routes
in INT8 unconditionally, so there is no quantization choice left to make.

| option | default | what it does |
|---|---|---|
| `tau` | 1.3 | Routing threshold, in sigmas of the proxy row. A key block is exact when its mean score over the query block clears `tau * sqrt(var)`; everything else contributes one pooled term. Higher is sparser and cheaper. Upstream's densities: 1.0 keeps ~16% of blocks exact, 1.5 ~7%, 2.0 ~2.7%. |
| `start_percent` | 0.2 | Dense before this point in sampling. **The one knob here with no measured rationale anywhere in this repo** -- 0.2 is the paper's number, carried through. Upstream reports a later start affects motion least, which makes it the first thing to raise when quality needs clawing back. Reported, not measured here. |
| `end_percent` | 0.9 | Dense after this point. |
| `min_tokens` | 12288 | Shorter sequences stay dense. Note `SOL_RECOMMENDED` pins **4096**, and the node's own description says dense is usually faster below ~12k. The two disagree and the disagreement is unmeasured. |
| `sink_conditioning` | `exact_kv_and_rows` | H3 packs `[text][cond][ref][audio][video]` into one sequence. `exact_kv`: every query sees the conditioning rows exactly (~3% cost). `exact_kv_and_rows`: those query rows also run dense (~17%). `off`: no protection. Those rows are ~250-400 of a ~38k sequence -- thin enough to be exactly what a block-sparse router drops first. |
| `morton` | False | Reorder video tokens into Z-order so each 64-token block is a compact 3D neighbourhood. Exactly neutral for dense attention. |
| `morton_curve` | `2d_frame` | `2d_frame` Z-orders within each frame and leaves frame order alone -- correct for H3, whose `FRAME_PER_TOKEN` is `(1,4,4,4,4)`, so index-adjacent frames are 1 or 4 real frames apart and a 3D curve groups temporally distant tokens. |
| `centroid_tail` NEW | True | Evaluate the pooled branch once per 64-token query block at its centroid instead of per row, for 64x less routing work. Upstream: ~1.4x on the **operation**, **~5-10% end to end**, ~5e-4 cosine cost. Do not confuse the two figures -- see the unit trap below. |
| `routed_cap_percent` NEW | 0 | Cap the routed-block list at this percent of the sequence; 0 is uncapped. Bounds the only workspace term that grows with T². Upstream reports ~30 as 3x headroom at tau 1.4 and lossless. Below the actual density it silently degrades routed blocks to their pooled term, so it trades quality for memory, not for free. |
| `reuse_qkv_memory` NEW | False | Write the attention output into H3's fused qkv buffer instead of allocating. Upstream: cuts peak VRAM by one output-sized tensor, ~1.2 GB at 80k tokens, enough to put attention's peak below the FFN's. Safe for H3 specifically, which discards that buffer after attention; leave off for other models. |
| `tau_profile` NEW | (unset) | Per-block tau overriding the base, as `blocks=tau` entries separated by `;` or newlines. A `force_input` string, so it needs a node wired to it. |
| `dense_blocks` | `""` | Blocks kept fully dense, e.g. `0-2,-1`. First and last blocks are the most approximation-sensitive. |
| `verbose` | False | Per-shape dispatch logging, logged once per distinct shape. |

### The three things actually measured here

On a 4090, `B=1 T=512 H=4 D=128` bf16 at tau 1.3, against the eager reference
the algorithm's author wrote:

- **The two kernels' arithmetic is equivalent.** CUDA sits at cos 0.999919
  from the reference, Triton INT8 at 0.999885, Triton bf16 at 0.999995. There
  is no accuracy gap between the backends at this shape.
- **They run different tail modes**: CUDA defaults `centroid_tail=True`, the
  Triton kernel is per-row and has no such parameter. Measured by grading each
  against both reference modes, not read off the source. This is a real
  behavioural difference between the backends but it is **not** where the
  speed comes from -- see the unit trap below.
- **The `reuse_qkv_memory` path is numerically identical** to the normal entry
  (cos 0.999810 against dense SDPA, both, to six digits, on a fully-exact
  control). It changes where the output lands, not what it is.

### The unit trap: kernel speed is not end-to-end speed

Upstream reports **CUDA at 1.4x over Triton at the same tau, end to end**, and
separately that **`centroid_tail` is worth ~5-10% end to end**. Both figures
are the author's, on his hardware, reported not reproduced here.

The `centroid_tail` tooltip's "~1.4x faster" is the *operation*, not the
render. Pairing it with the 1.4x e2e figure -- as this document did briefly on
2026-08-14 -- is numerology across two different denominators, and it is wrong
in a way that is easy to miss because the digits match. The arithmetic that
kills it: when only the kernel changes, e2e speedup can never exceed kernel
speedup, so a 1.4x *e2e* win demands a kernel-level gap considerably larger
than 1.4x. A knob worth 1.4x on the op cannot produce it, and upstream's own
5-10% figure confirms it does not. The backend gap is the kernel
implementation; `centroid_tail` is a slice of it.

So: **never quote a number from `bench_minimax_attn.py` against one from
`bench_e2e_h3.py`.** They have different denominators, attention is only part
of a step, and Sol is only active inside the `start_percent`/`end_percent`
window on top of that. Two instruments, two units.

### Where the gains actually live

Upstream is explicit that the win shows up only at high token counts -- large
canvas, long duration, or many references -- and that the relative gain grows
with size. The floor he gives is blunt: **nothing is visible below roughly
250-300 frames**, and his 1.4x figure was taken around 500. Reported, not
reproduced here.

That invalidates the regime this page's own frontier table was measured in.
**Length 124 is below the floor** -- less than half of it. So is the bench's
default `--length 73`. Everything in the table above was measured where
upstream says there is nothing to see, which reframes it: those numbers are
not a weak version of the result, they are a measurement of the wrong regime.
The "15% is a floor, and the length it was measured at is why" section below
was reaching for this and stopped short of the number.

It is also why `SOL_RECOMMENDED` pinning `min_tokens=4096` against the node's
own 12288 is a live question rather than a detail: below the crossover
Sol-Attn engages and costs time, and 4096 engages it well inside the regime
upstream says is a loss.

**The floor is a token count, not a frame count.** Video tokens are
`latent_t * (w/32) * (h/32)` with `latent_t = ((n - 5) // 17) * 5 + 2`, so the
canvas is half the quantity and "250 frames" means different things on
different shapes:

| canvas | 73 frames | 124 | 250 | 300 | 362 |
|---|---|---|---|---|---|
| 1344x768 (1008/frame) | 22,176 | 37,296 | 72,576 | 87,696 | **107,856** |
| 832x768 (624/frame) | 13,728 | 23,088 | 44,928 | 54,288 | 66,768 |

Upstream's ~100k ceiling is the 1344x768 / 362-frame / 15s configuration, and
the table puts it at 107,856 video tokens -- so "100k" is that corner of the
space, and the packed sequence adds text, audio and reference rows on top.
Upstream separately reports quality holding "surprisingly well on only 60k
tokens", which is a different claim from the speed floor and should not be
merged with it.

Two consequences:

- **A frame-count guard is wrong.** 250 frames at 832x768 is 44,928 tokens --
  below the floor, while 250 frames at 1344x768 is comfortably above it.
  `bench_e2e_h3.py` warns on tokens for this reason; the first version of that
  guard counted frames and would have passed the small-canvas run and let it
  read as a null result.
- **`min_tokens=4096` in `SOL_RECOMMENDED` is the outlier that needs
  justifying**, not the node's 12288. 4096 is a third of the node's own stated
  dense/sparse crossover and two orders below where the gains appear. Those
  are not contradictory numbers -- ~12k is roughly where Sol-Attn stops
  *losing* and ~60k+ is where it *wins* -- but 4096 sits below both, which
  means the shipped config engages Sol-Attn in the regime upstream describes
  as a loss. That is measurable and unmeasured.

Length and reference count belong on the sweep axis, not fixed at the cheap
end. A bench that cannot produce a signal is the timing equivalent of a check
that cannot go red.

### References are pinned exact, and a video reference dominates the sequence

`_sink_blocks` covers rows `[0, video_start)`. From `PackedLayout` that is
text, keyframe `cond`, keyframe `cond_audio`, `ref_img`, `ref_audio` **and the
target audio segment** -- every reference type, whatever it is. `exact_kv`
makes all of it exact KV for every query; `exact_kv_and_rows` additionally
runs those query rows dense.

Row counts differ by two orders of magnitude between types (measured, see
`docs/h3_references.md`, at 1344x768):

| reference | DiT rows | also adds to text |
|---|---|---|
| audio, per second | 80 | - |
| image at `match` | ~1,008 | - |
| image at `max`, 1024x1024 | 4,096 | +4,096 |
| image at `max`, 1280x720 | 7,296 | +7,296 |
| video, 960x544, 124 frames | 18,870 | ~+1,700 |
| video, 960x544, 345 frames | **52,020** | +4,667 |

Images and videos pay in **two** segments, and the text segment is inside the
sink as well, so a reference grows the exact region twice.

Share of attention forced exact, at a 362-frame 1344x768 target (arithmetic
over the row counts above; `S` = sink rows, `T = S + V`, exact work is `T*S`
for `exact_kv` and `2*T*S - S^2` for `exact_kv_and_rows`):

| configuration | `exact_kv` | `exact_kv_and_rows` |
|---|---|---|
| t2v, no references | 1.5% | 2.9% |
| 3 images at `match` | 4.1% | 8.1% |
| 3 images at `max` 1280x720 | 29.6% | 50.5% |
| 1 video ref, 124 frames | 17.1% | 31.2% |
| 1 video ref, 345 frames | **35.1%** | **57.9%** |

So at the shipped `exact_kv_and_rows`, one long video reference forces 58% of
the attention exact and leaves Sol-Attn only 42% to work on, at any tau.
**"More context" is not the same as "more sparsity"**: a video reference is
context this configuration deliberately pins dense, and it is exactly the
workload with the most reason to want Sol.

### The sink conflates two things worth separating

`final_layer(h, t_emb, video_seg, audio_seg)` reads out only the target video
and target audio segments. So the rows in the sink split into:

- **reference and conditioning rows** -- outputs discarded. They matter only
  as keys and values for later layers, so running *their queries* dense is a
  second-order effect.
- **the target audio segment** -- decoded output, first-order, and the stated
  reason `exact_kv_and_rows` is on at all ("what keeps the generated audio
  intact", `h3_config.py`).

`exact_kv_and_rows` treats both identically. With a 345-frame video reference
that means paying ~23 points of extra forced-exact work on reference rows in
order to get the audio rows dense.

They are separable, and the kernel already allows it: target audio is a single
contiguous segment immediately before video (`PackedLayout` appends target
audio then target video, "always the last two segments"), so
`sink_q = [audio_start // BLOCK, video_start // BLOCK]` would run only the
audio queries dense. The node hardcodes `sink_q = sink_blocks` instead. This
is a proposal, not a measurement, and it is upstream's call.

### `centroid_tail` may stop being a toggle

Upstream is considering making it unconditional -- "probably should even if
it's technically a bit more lossy", on the grounds that there are too many
options for an average user, with model-specific defaults in the model config
as the alternative. Reported from conversation, not a shipped change.

If that lands, `centroid_tail=False` disappears and the A/B that separates the
toggle from the kernel becomes unrunnable. So that experiment has a clock on
it. `SOL_CUDA_DEFAULTS` pins the value rather than inheriting it, which is the
usual defence, but pinning cannot survive an input being removed -- passing a
key the node no longer declares is an error, not a silent no-op.

Grading each kernel in the other's tail mode costs about 1.2e-3 of apparent
accuracy -- larger than the gap between the kernels. Done naively it invents a
CUDA win that is not there. It did, once, before the modes were measured.

## Quality

**REOPENED.** Everything in this section was judged from still frames, and
the failure mode that matters is temporal. Read the next subsection first.

### The artifact stills cannot show

At `tau` above roughly 1.5, a small persistent object can dissolve partway
through a clip and be replaced by something else. In frames examined at
24 fps the transition took about four frames -- solid, coming apart,
gone -- and it never recovered. Not warping or blurring: the object loses
its identity and the model generates a locally plausible substitute.

Two things follow.

**A grid of stills at sampled shot-times cannot catch this**, which is what
the judgments below used. Pick four moments in an eight-second clip and the
odds of landing on both sides of a four-frame transition, on the one small
object that failed, are poor. The instrument was fine for per-frame
fidelity and blind to identity drift. A gate for this tracks one small
persistent object -- a hair ornament, jewellery, on-screen text -- frame by
frame, which is still a still-frame method, just sampled densely in time
instead of sparsely.

**The fix is to force something dense.** `dense_blocks` exempts the most
approximation-sensitive transformer blocks; `sink_conditioning` does the
same job for the packed conditioning rows. Both are cheaper than backing
`tau` off far enough to avoid the problem globally.

`SolAttnBlockProbe` is the instrument for choosing the block list: it runs
every attention call both sparse and dense and logs per-block relative
error worst-first. Run it at the `tau` you actually intend to use, since a
profile taken at a gentler setting is measured where the failure does not
occur. `SOL_RECOMMENDED` ships `dense_blocks=""` -- no blocks forced dense.
A starting set exists as `SOL_ARTIFACT_INSURANCE` in `h3_config.py`, which is
deliberately NOT what the graphs wire, pending our own probe run.

### The earlier judgments, at length 124

Treat Sol-Attn as a speed knob. No quality difference held up.

| seed | arm | blind | verdict |
|---|---|---|---|
| 801 | sol+morton+int8qk | no | Sol-Attn better |
| 701 | plain sol | yes | different, neither better |
| 702 | plain sol | yes | nearly identical |
| 1001 | sol+morton+int8qk | no | same |

One positive in four, and it was the first pair looked at -- before there was
any sense of how much these samples vary seed to seed. A later pair on the same
arm at a different seed came back "same", unblinded, so an expectation effect
had its chance and did not appear.

Limits of this, which are severe:

- One observer, one prompt, four pairs.
- Three of the four judgments came late in a long session. Fatigue pushes
  toward "these look the same", which is the direction the nulls point, so
  "no difference" and "stopped discriminating" are not separable here.
- The prompt is slow-camera, diffuse fog, ambient audio. A block-sparse artifact
  would more likely surface in fast motion, fine repeated detail, or on-screen
  text -- none of which this scene has.

For scale on the noise floor: the same observer called one plain-sage render
"dramatically more interesting" than two others differing only by seed.

The renders are kept, so re-judging cold is the cheap way to firm this up.

### Audio

**RETRACTED: "Sol-Attn renders are consistently louder."** The original
finding is below, and it does not survive 362 frames. There the two mildest
configurations measure *quieter* than sage, and what actually tracks
loudness is how aggressive the sparsity is, not whether Sol-Attn is on. The
Aug 4 result was a config effect read as a Sol-Attn effect, from three pairs
at one setting.

It may also be contaminated. Upstream fixed a Morton corruption at certain
sizes on Aug 4 (`e353f6d`), the arms below ran with morton on, and the
commit these were measured against was not recorded. A larger unreproducible
audio jump was seen on one build and could not be reproduced the next day,
which fits a size-dependent bug better than it fits anything about sparsity.

What replaces it: audio deviation tracks `tau` (about +1.6 dB mean at 2.0),
which is a real change to the signal that nobody has been able to hear. And
H3's audio rows are ~250-400 in a ~38k sequence -- thin enough to be exactly
what a block-sparse router drops first, the same shape as the object-dissolve
artifact above. Forcing them dense is the fix, which is why
`sink_conditioning="exact_kv_and_rows"` is now the default here and upstream.

The original measurement, kept for the record:

| pair | seed | mean dB (sage -> sol) | peak dB (sage -> sol) |
|---|---|---|---|
| 00031/00032 | 1001 | -40.2 -> -39.6 (+0.6) | -25.6 -> -24.6 (+1.0) |
| 00033/00034 | 1002 | -39.6 -> -38.9 (+0.7) | -24.8 -> -23.5 (+1.3) |
| 00035/00036 | 1003 | -36.9 -> -35.2 (+1.7) | -20.8 -> -14.8 (+6.0) |

Louder in 3 of 3, on both mean and peak. This does not mean *better* -- it means
Sol-Attn measurably changes the audio path, which nothing else in this session
established.

Plausible mechanism, untested: attention output is a weighted average, and
sparse attention drops low-weight blocks then renormalizes over what remains, so
the result leans harder on the strongest matches. Less averaging reads as more
dynamic range. The +6.0 dB peak against a +1.7 dB mean has that shape --
transients sharpening rather than everything rising uniformly. If that is what
is happening, "sounds better" and "is less faithful" would both be true at once:
punchier is more pleasing and less accurate.

That mechanism was written to explain the loudness, and the loudness is the
part that did not replicate, so treat it as an unsupported story rather than
a finding.

The paradox recorded here -- sparsity should *hurt* thin audio rows, yet
forcing them dense also sounded better, so both cannot be right -- resolves
once the loudness result is dropped. Only one of the two was real: thin rows
get routed out, and forcing them dense helps. There was never a second
effect to reconcile.

Still never checked: accuracy against the sage output rather than loudness.
The files exist, so that is a listening test, not a render.

## Where the time actually goes

Profiled one forward, 50 DiT blocks, device time only:

| | sage dense | sol sparse (`int8_qk`) |
|---|---|---|
| attention kernel, all 50 blocks | 4296 ms | **2668 ms** |
| per block | 85.9 ms | **53.4 ms** |
| sol routing + quant prep | — | 161 ms (2.1%) |
| everything else | 4944 ms | 4935 ms |
| whole forward | 9240 ms | 7603 ms |

Two things worth reading off that table.

**Sol-Attn's sparse kernel is 1.61x faster than sage's dense kernel.**
That is a real, large win at the kernel level.

**Non-attention time matches to 0.2%** (4944 vs 4935 ms), which is the
control: only attention changed, so the patching surface is clean.

The end-to-end result is small because of Amdahl applied twice: attention
is only ~46% of a forward, and only 14 of 20 steps fall inside Sol-Attn's
default sigma window. All 20 steps sparse would give ~1.22x on the
sampler; the window alone accounts for roughly half the gap between that
and the 1.15x measured.

## Configuration findings

- **`int8_qk` and `int8_pv` are both worth setting.** Sol-Attn's default
  kernel is **bf16**; sage runs INT8 QK + FP8 PV. At 362 frames the pair is
  worth 1.16x on top of plain sparsity (827.9 -> 714.9 s). Upstream's
  tooltip says int8 helps at `tau<=1.5` and turns into a net loss at
  `tau>=2.0`, where the quantize pass outweighs a shrinking exact branch --
  untested here, but consistent with running int8 at the lower taus this
  page now recommends.
- **`tau`: stay at or below ~1.3.** Above roughly 1.5 the object-dissolve
  artifact under Quality appears. The timing table shows what the higher
  settings would have bought; they are not worth it.
- **Morton off.** At 124 frames it measured ~2 s slower than plain Sol-Attn
  once warm. At 362 frames the picture is sharper: worth 1.16x on its own,
  but stacked on int8 it *costs* about 3.6% (1.34x against 1.39x for int8
  alone). Its arm also ran at 94% GPU utilisation where every other arm hit
  99%, i.e. the permutation adds non-tensor-core work that stops paying once
  int8 has shrunk the exact branch it routes. Upstream now defaults it off
  too.
- **`sink_conditioning="exact_kv_and_rows"`, on.** It runs H3's conditioning
  query rows dense, and those rows -- ~250-400 in a ~38k sequence -- are thin
  enough to be exactly what a block-sparse router drops first. Cost is
  unsettled: measured ~4% at 124 frames against upstream's "~20%" tooltip.
  It should get cheaper in relative terms on longer clips, since the
  conditioning row count is fixed while attention grows as S^2.
- **`dense_blocks`**, set from a `SolAttnBlockProbe` run at your own `tau`.
  Seven of fifty blocks costs roughly 54 s on a 362-frame render, which is
  less than backing `tau` down far enough to avoid the artifact globally.
- The conditioning sink forces only **9 of ~591 KV blocks** exact (1.5%),
  so it is not meaningfully inflating density.

### Record the Sol-Attn commit with every measurement

Not a configuration finding, a process one. Sol-Attn shipped five commits on
Aug 4 alone, at least two of them behaviour-changing at long-clip sizes. The
timing arms and audio numbers on this page were taken without recording
which build produced them, which puts an asterisk on all of them. Snapshot
the node's commit alongside torch and triton versions, or the numbers cannot
be defended later.

## Ordering

```
Load Diffusion Model -> MiniMax H3 SageAttention -> SolAttnPatch -> BasicGuider
```

Sol-Attn must come second: it walks the model's existing object patches
and composes with the attention forwards it finds. Reversed, this node
overwrites its patch and you silently get sage only.

They **alternate rather than stack**. Inside the sigma window Sol-Attn
runs sparse and sage is bypassed entirely; outside it, sage runs dense.
Confirmed engaged, rather than assumed, from its own verbose logging:

```
[sol_attn] composed with 50 object-patched attention forward(s)
[sol_attn] sparse (1, 37826, 56, 128) tau=1.3 int8 pointer
```

with no `dense: <reason>` lines and zero kernel failures.

## Measurement traps hit while producing this

Recorded because each one produced a confident, wrong number first.

- **ComfyUI caches node outputs.** Re-submitting an identical graph
  executes nothing and returns in milliseconds. The first version of the
  e2e bench reused one seed and reported a **405x speedup** for a render
  that never ran. Vary the seed per iteration and hard-fail when the timed
  node does not execute.
- **Run 1 of any Sol-Attn arm pays Triton autotune.** Comparing a warm arm
  against a cold one made Morton look like it helped by 3 points. It does
  not; it is slightly negative. Warm every arm or compare run 2 only.
- **Reasoning from wall-clock is not profiling.** From per-step arithmetic
  it looked like routing overhead was eating the sparsity benefit. The
  profile put routing at 2.1% and the real answer at Amdahl. The
  arithmetic was arithmetically fine and the conclusion was wrong.
- **Comparing finished renders numerically measures chaos, not quality.**
  At 20 steps of a flow-matching ODE any perturbation diverges the
  trajectory, so same-seed sage-on/sage-off latents differ substantially
  while both look fine. The honest instruments are fixed-input kernel
  divergence and human judgement.
- **Synthetic inputs cannot answer questions about input distribution.**
  A `torch.randn` sweep reported `smooth_k` as having no effect, which it
  would have reported whether or not the effect were real, since
  `torch.randn` has zero mean and `smooth_k` removes a channel offset.

## Reproducing

```
python bench/bench_e2e_h3.py --arms sage,sage+sol+morton+int8qk --runs 2 --length 124
```

Arms are defined in `bench/bench_e2e_h3.py`. Add a `verbose` arm to
confirm Sol-Attn engaged; its routing decisions land in the ComfyUI log
under `[sol_attn]`.
