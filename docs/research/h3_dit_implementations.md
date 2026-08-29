# The H3 DiT: five implementations of one model, compared

last updated: 2026-08-29

**What this file owns.** The *numerical* comparison of the MiniMax H3 DiT
across every implementation available here: module tree, packed sequence,
rotary embedding, attention, modulation, forward pass, and the sampler that
drives it — for a text-to-video+audio (`t2va`) generation.

**What it does not own, and must not restate.**
[`sglang_comparison.md`](sglang_comparison.md) owns the *runtime and
optimization* comparison against sglang. [`../h3_references.md`](../h3_references.md)
owns reference-image conditioning. [`../h3_pdd.md`](../h3_pdd.md) owns PDD.
Where this file touches those, it points and stops.

**Evidence labels**, used inline throughout:

- *read* — the code at the cited path, at the revision in §1.
- *measured* — computed for this document from an artifact on disk.
- *inference* — a conclusion, with the mechanism named so it can be refuted.

**Different is not worse.** Where an implementation does something the others
do not, that is recorded as a difference and a consequence. A verdict appears
only where there is evidence for one.

---

## 1. The implementations

| short name | what it is | revision |
|---|---|---|
| **ComfyUI** | `comfy/ldm/minimax/model.py` — what this box runs | ComfyUI `e7051b03` |
| **diffusers** | `coderef/diffusers` `MiniMaxH3Transformer3DModel` + its modular pipeline | `c1bf18c92` |
| **sglang** | `coderef/sglang` `multimodal_gen` H3 pipeline | `97781eb7f3` |
| **DiffSynth** | `coderef/DiffSynth-Studio` `minimax_h3_dit.py` (+ a Comfy variant) | `102fe99` |
| **vllm-omni** | `coderef/vllm-omni` H3 diffusion path | `72a1ce48` |

One note on vllm-omni's **VAE**, since it was read alongside: it is an
*adapter*, resolving the real conv stack, tiling and vocoder from the
checkpoint's own `trust_remote_code` module rather than implementing them. It
applies the same `(latent - mean) / std` per-channel normalization read from
the component `config.json`, and carries **no `scaling_factor` and no
`shift_factor` anywhere** — so an engine doing the diffusers-style
`latents / scaling_factor` disagrees with it (*read*).

`coderef/` checkouts are gitignored symlinks, so citations into them are
advisory to `bench/check_doc_links.py` and may drift. Re-read before quoting an
older section as current — the standing rule in
[`../config_drift.md`](../config_drift.md).

---

## 2. Ground truth, and the thing that surprised me

**The release ships no transformer modeling code.** `coderef/MiniMax-H3/`
carries `transformer/config.json` and the weight index, plus full Python for
both VAEs — and nothing for the DiT. Its `model_index.json` names
`MiniMaxH3ModularPipeline` and the transformer class
`MiniMaxH3Transformer3DModel`, which lives in **diffusers**, not in the release.

So every one of them is a reimplementation against the weights, and diffusers
is the one the vendor's own manifest points at. That makes it the reference of record
here — not because it is more correct, but because it is the one the release
names. *(read: `coderef/MiniMax-H3/model_index.json`, and the absence of any
`transformer/*.py` in that tree.)*

What the release does declare, verbatim (*read*):

| declared | value | where |
|---|---|---|
| architecture | 56 heads x 128, hidden 5376, 50 layers, 2 refiner layers, ffn 14336, in 24 / audio 32, patch (1,2,2), text_dim 5120, freq_dim 256, time_embed 5376 -> 2688 | `coderef/MiniMax-H3/transformer/config.json` |
| rope | `rope_theta: 10000.0`, `rope_freq_dim: 16` | same |
| eps | `norm_eps`, `qk_norm_eps`, `final_norm_eps` all `1e-05` | same |
| sigma shift | video `12.0`, audio `3.0` | `scheduler/` and `audio_scheduler/scheduler_config.json`; also `_minimax_h3.sigma_shift_scales` in both vendored `*_model_index.json` |
| tasks | `t2va` and `fl2va` on the fl2va partition; `ref2va` on its own | `vendor_config/{fl2va,ref2va}_model_index.json` |

`transformer/config.json` and `transformer_ref/config.json` are byte-identical,
so the two partitions differ only in weights (*measured*: `diff` over the two
files).

**t2v is `t2va` on the fl2va partition with no keyframes.** There is no
separate checkpoint and no separate code path — the packed sequence simply has
no condition rows.

---

## 3. The invariant core

Everything in this section is identical across all five, and each row says how
that was established. Where a claim could have been inherited by all five from
one source, it was checked against the artifact instead.

### 3.1 Verified against the weights, not against agreement

Three facts every implementation asserts, re-derived here from
`minimax_h3_fl2va_pruned_int8_convrot.safetensors` — the file
`h3_config.MODELS["unet_fl2va"]` resolves to:

**Rotary frequencies.** `rope.inv_freq` is a 16-element fp32 tensor equal to
`10000^(-i/16)` (*measured*). That is exactly `1/(theta**(arange(0,32,2)/32))`
with `theta = 10000` — the release's declared `rope_theta`, and per-axis rotary
width 32. diffusers' converter states independently that its recomputed buffer
is bitwise equal to the shipped one, which is a second confirmation by a
different route.

**AdaLN packing and chunk order.** Every implementation claims the modulation
table is modality-major — row `= timestep_index * 3 + tag`, tags `0` video,
`1` text, `2` audio — with chunks ordered
`shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`. Split
`blocks.N.adaln_proj.linear.bias` into its eighteen `hidden`-wide slabs and
exactly six carry a materially non-zero mean, at slabs 1, 4, 7, 10, 13, 16
(*measured*). Under modality-major those are *the two scale chunks, once per
modality* — which is what you would expect, since `scale` enters as
`1 + scale` and is the only one of the three needing a non-zero baseline.
Under the competing chunk-major reading the same six slabs would be *every
chunk of the text modality and nothing for video or audio*, i.e. only text
modulated. The layout and the chunk order are therefore fixed by the weights.

**Module inventory.** The 932 keys resolve to exactly the tree all five
describe: 50 blocks of `{norm1, norm2, attn{q_norm,k_norm,qkv_proj,out_proj},
mlp{fc1,fc2}, adaln_proj}`, a 2-block `token_refiner` with `final_norm`, a
`final_layer` with two output heads, the three input projections, and
`rope.inv_freq` (*measured*). Shapes confirm the derived widths: `qkv_proj`
`21504 = 3*56*128`, `fc1` `28672 = 2*14336`, `video_out` `96 = 24*1*2*2`,
`adaln_proj` `96768 = 6*5376*3`, `final_layer.adaln_proj` `10752 = 2*5376*1`.

### 3.2 Agreed by all five, read in each

- **One packed sequence, full bidirectional self-attention, no mask.**
  `[text | conditioning | audio | video]`, target audio always immediately
  before target video. Video, audio and text all attend to each other in every
  block. **There is no cross-attention anywhere in H3** — text conditioning is
  a prefix of the same sequence, and it is rewritten by every block rather than
  held frozen.
- **Attention inner dim exceeds the residual stream.** `56 * 128 = 7168`
  against `hidden_size = 5376`. Deliberate; every implementation comments on it.
- **QK-norm** is RMSNorm over `head_dim`, learnable, applied per head *before*
  rope. V is never normed and never rotated.
- **Partial rotary.** Three axes x 16 frequencies = 48, concatenated with
  itself to 96, so **96 of 128 head dims rotate and the last 32 pass through**.
  Convention is halves / rotate-half (`chunk(2, -1)`), never interleaved.
  ComfyUI builds an explicit rotation table from the duplicated halves and
  fuses norm+rope into one kernel; the arithmetic is the same.
- **Rope is applied to the text prefix**, on the t axis only, at
  `(row_index, 0, 0)` — and the video time axis *continues from where the text
  axis ends*. Prompt length therefore shifts every video and audio position id.
- **Position grid**: spatial axes normalized by `sqrt(latent_h * latent_w)`,
  `linspace(..., endpoint=False)`, scaled by 32, built in float64. Temporal
  spans are non-uniform, `5/3 * (1,4,4,4,4)` with an exclusive cumsum.
- **One clock for both streams.** A video pixel frame is `5/3` units and an
  audio latent frame is `1.0`; since `24 fps * 5/3 = 40 Hz`, one unit is
  `1/40 s` in both. Audio rows carry `h = 0` and are pinned to the two extremes
  of the video's w axis, one per stereo channel.
- **Timestep is `t = 1 - sigma` in `[0,1]`, unscaled**, and the sinusoidal
  embedding puts **cos before sin**. Both invert the more common convention.
- **Conditioning anchors** sit at `t = 0.999` visual, `t = 1.0` audio, in every
  implementation that has them.
- **Two sigma schedules per request**, video shift 12.0 and audio shift 3.0,
  through the same map `sigma' = s*b / (1 + (s-1)*b)`.
- **Text conditioning is Qwen3-VL hidden state 50, un-normed**, from a
  verbatim prompt with no chat template and no special tokens for `t2va`.
  ComfyUI truncates the encoder to 50 layers and sets
  `layer_norm_hidden_state=False`; diffusers reads `hidden_states[50]` off a
  full stack and warns explicitly that a stack truncated to exactly 50 layers
  would be post-norm and wrong. Same tensor, two routes.
- **The checkpoint is guidance-distilled.** diffusers says so and runs no
  unconditional pass; sglang freezes `guidance_scale` at 1.0 with a single
  denoise branch; DiffSynth defaults `cfg_scale=1.0`, and its training loss
  divides CFG back out with a comment saying the checkpoint has it distilled in;
  vllm-omni *raises* on a CFG request — "MiniMax-H3 is CFG-distilled and has no
  negative branch". Four independent statements of the same fact.

---

## 4. Divergences that change the sample

Ranked by how much they move the output.

### 4.1 The sampler: stochastic here, deterministic everywhere else

**Every reference implementation integrates with deterministic Euler and
injects no noise.** diffusers says it outright — the class is named after
"euler ancestral" but `eta` is 0 and no noise is ever re-injected. sglang's
adapter is literally `MiniMaxH3EulerAncestralEta0SchedulerAdapter`. DiffSynth
steps with `x + v * (sigma_next - sigma)`. vllm-omni's file is *also* named
`scheduling_minimax_h3_euler_ancestral`, and also has no ancestral term: no
`eta`, no `s_noise`, no `s_churn`, and no `randn` anywhere after the initial
latent. **Three of the four inherited a misleading name from the vendor and
none of them inherited the behaviour** — good evidence that eta=0 is the
release's intent rather than four independent simplifications.

The shipped graphs here split between `er_sde` and `euler`
(*measured*, over `h3_config.graph_paths`). `er_sde` is a third-order multistep
solver that **adds fresh noise every step** —
`x += alpha_t * noise_sampler(...) * s_noise * sqrt(...)` with `s_noise = 1.0`
(*read*: `comfy/k_diffusion/sampling.py`, `sample_er_sde`). So the arms that use
it differ from the vendor's integrator in two independent ways: solver order,
and stochasticity.

This is not news as a *choice* — `workflows/h3_config.py:178-197` records
`er_sde` as the owner's call since 2026-08-15. What is new is the comparison:
nothing in the repo recorded that **all four reference engines run eta=0
deterministic Euler**, so the divergence had never been priced against them.
It also compounds the standing rule in `CLAUDE.md` that a rendered clip cannot
A/B a numerical change — with `er_sde`, two runs of the *same* arm differ too.

**Not a defect, and no verdict here.** `er_sde` may well be the better solver;
this repo has no measurement either way, and getting one needs a distribution
under [`../eval_comparison.md`](../eval_comparison.md) section 3, not a pair.

### 4.2 Step-count semantics: "N steps" means different things

All implementations build the sigma grid with the same shift map over `[1, 0]`.
They differ in how many points they place and whether `0` is one of them:

| | grid | evaluations for N=16 |
|---|---|---|
| sglang, diffusers, vllm-omni | `linspace(1,0,N)`, terminal `0` included | **N-1** |
| DiffSynth | `linspace(1,0,N+1)[:-1]`, `0` forced at the last step | N |
| ComfyUI `simple` | N+1 sigmas off the 1000-entry table, `0` appended | N |

So a "16-step" request is 15 forwards in sglang, diffusers and vllm-omni, 16 in
ComfyUI and DiffSynth (*measured* for the first three; the vllm-omni row is
*read*, and its own Turbo validator says so out loud — "five sigma points
produce four denoiser evaluations").

**The same parameter changes meaning inside one engine.** On vllm-omni's
distilled path `num_inference_steps` is `len(base_schedule) - 1`, i.e. the
evaluation count, where on its uniform path it is the point count. An
implementation that ports one path's convention to the other is off by a whole
step (*read*).

**The shipped `simple` scheduler tracks the vendor grid closely.** At the
shipped step count the final Euler step spans 44.65% of the sigma range under
ComfyUI `simple`, 44.44% under DiffSynth and 46.15% under the vendor
construction; `normal` and `beta` diverge far more (*measured*). Tail
coarseness is what governs quality here — the PDD lane established that, and
`CLAUDE.md` carries it — so `simple` being the close one matters.

That corroborates the `simple` choice already argued at
`workflows/h3_config.py:199-218`, but by a different route: that note argues
from a distilled LoRA's published grid and is controlled by
`bench/check_distill_grid.py`; this argues from the vendor's own *inference
engine* construction. Two independent reasons, same answer.

### 4.3 The audio stream is carried differently

diffusers, sglang, DiffSynth and vllm-omni hold **two scheduler instances** and
step the two streams separately from one joint forward.

ComfyUI instead carries the audio latent *scaled onto the video schedule* by
`audio_scale = shift_video / shift_audio`, making the pack an ordinary
single-schedule flow latent, and undoes the scaling inside `forward()` before
the network sees it (`comfy/ldm/minimax/model.py`, `MiniMaxH3Model.forward`;
`comfy/model_sampling.py`, `ModelSamplingAV`). *Inference*: this is an exact
change of variables, not an approximation — the wrapper converts both the
latent and the returned velocity — so it should be numerically equivalent to
holding two schedules. It is nonetheless a different arrangement, and anything
comparing raw audio latent magnitudes across engines has to divide it out.

### 4.4 Sequence padding

The vendor's reference pads the packed sequence to a multiple of 64 and splits
the tail off with `cu_seqlens = [0, used, S]` — diffusers says so explicitly
while declining to copy it. sglang, DiffSynth and vllm-omni all pad, with that
exact constant. **diffusers and ComfyUI do not pad at all.**

*Inference*: numerically inert on the real rows, since the padding is a
separate attention document that real tokens cannot see. It costs compute
proportional to the padding, and it is the reason the padded implementations
need a `cu_seqlens` at all.

---

## 5. Divergences that cost, but do not change numbers

- **Output heads over every row.** diffusers, sglang, DiffSynth and vllm-omni
  run both heads across the whole sequence — padding included — and select
  afterwards; sglang comments that this is deliberate, to keep the GEMM shape
  and defer collectives. ComfyUI slices the two target segments first and runs
  the heads only there. Same numbers for the rows that survive. It is also why
  the padded implementations zero their condition rows with a multiply rather
  than a slice.
- **Where the packed layout is built.** ComfyUI builds it inside the DiT and
  caches it per shape signature; diffusers requires the caller to supply
  positions, tags and indices; sglang and DiffSynth build it in a pipeline
  stage. A consequence, not a defect: the diffusers transformer is reusable
  with any layout, and ComfyUI's cannot be driven with one it did not build.
- **`rope.inv_freq` loaded vs computed.** ComfyUI, sglang and vllm-omni read
  the buffer from the checkpoint — none of the three contains a theta literal
  at all. diffusers recomputes it from `rope_theta` and drops the key.
  DiffSynth allocates the parameter and then *never reads it*, rebuilding the
  frequencies each forward — so a checkpoint shipping a non-default `inv_freq`
  is silently ignored there, and would be ignored by diffusers too.
- **The token refiner** runs once per sampling run in ComfyUI and sglang, and
  per forward in the others. It takes no rope and no AdaLN everywhere.
- **vllm-omni's RainFusion** is block-sparse attention over the video segment
  only. It is **NPU-only and opt-in** — `forward_cuda` raises — so it is not on
  any default path, but it is *lossy by construction* (pooled-relevance block
  selection), unlike the rest of this section. Named here so it is not mistaken
  for a free optimization.

---

## 6. Three traps that look like bugs and are not

Each of these reads as a defect until you find the compensating half. All three
have cost somebody a session somewhere.

**The SwiGLU halves are swapped between conventions.** ComfyUI applies SiLU to
the *first* half of the fused `fc1` output; diffusers applies it to the
*second*. The release stores `[gate; value]`, ComfyUI and sglang and DiffSynth
read it as stored, and diffusers' converter swaps the halves at conversion
time because its own `SwiGLU` reads `[value; gate]` (*read*: the converter's
own header comment). Both are correct against their own storage. sglang's
Diffusers-layout loader performs the same swap in reverse.

**The DiT output is negated in ComfyUI and not in diffusers.** H3 predicts a
*data-ward* velocity, `x0 = x_t + sigma * v`. ComfyUI's flow convention is
`x0 = x - sigma * v`, so returning `-v` reproduces the vendor rule exactly
(*read*: `CONST.calculate_denoised` in `comfy/model_sampling.py`, against
diffusers' scheduler docstring which states the sign reversal as its first
reason for being a separate class). DiffSynth negates for the same reason.
vllm-omni does not negate either, and puts the `+` in its scheduler.

The rule is that **the negation and the scheduler sign are one decision made in
two places**. Compose them and every implementation agrees; port half of one
into the other and the ODE runs backwards.

**Two qkv layouts are in circulation, and two implementations assume opposite
ones without checking.** The release's raw shards store qkv **per-head
interleaved**, `[q_h|k_h|v_h] x 56`; the Comfy-Org repacks store it
**contiguous**, `[q_all; k_all; v_all]`. How each handles it:

| | strategy |
|---|---|
| diffusers | converts interleaved -> contiguous offline, in its converter |
| sglang | branches on a quant-config flag, permutes on load |
| DiffSynth | ships a second DiT class, selected per checkpoint; its *only* delta is `view(S, 3, heads, D)` vs `view(S, heads, 3, D)` |
| vllm-omni | reorders **unconditionally**, assuming the release layout; the only guard is a row-count assertion |
| ComfyUI | splits contiguously **unconditionally**, assuming the repack layout |

The last two rows are the interesting ones: **neither sniffs the layout, and
they assume opposite things.** A release-native file fed to ComfyUI, or a
Comfy-Org repack fed to vllm-omni, loads without error and produces noise —
the row count is identical either way, so the one assertion vllm-omni has
cannot catch it (*read*). `docs/evidence.md:176-186` already records the
concrete instance from the ComfyUI side: a DeepBeepMeep bf16 file stores
`qkv_proj` in release order, so head 0's k rows land where ComfyUI's split
expects head 1's q.

This is already recorded at `docs/evidence.md:168-175` from an earlier session.
A spot check corroborates it and settles which layout this box holds:
splitting `blocks.0.attn.qkv_proj.weight_scale` into contiguous thirds gives
three clearly distinct means and explains ~10.6% of the total variance, while
the per-head-interleaved grouping explains ~0.03% — and the V third is the low
one, which is the physical signature, since Q and K are RMSNormed downstream
and V is not (*measured*). **The file this install loads is contiguous**, which
is what ComfyUI's `split(heads*head_dim, dim=-1)` requires.

---

## 7. Findings for this install

**1. The DiT architecture config is not vendored, though the release declares
it.** `vendor_config/` carries the tokenizer, the pixel bounds, the patch
geometry and both `model_index.json`s — but not `transformer/config.json` and
not the two `scheduler_config.json`s. So the architecture constants and the
eps triple are retyped as Python defaults in `comfy/ldm/minimax/model.py`
rather than read from the release, which is the exact failure mode
`vendor_config.py`'s header says it exists to prevent. They are correct today
(*measured*: every value matches the release). Vendoring the three files and
adding readers would close it; `bench/check_vendor_config.py` already has the
shape for it.

**2. `sigma_shift_scales` is vendored but has no reader.** Both
`vendor_config/*_model_index.json` declare `{"video": 12.0, "audio": 3.0}`, and
nothing reads them — `vendor_config.py` has no accessor, and ComfyUI carries
`12.0`/`3.0` as constructor defaults. Same class of gap as #1, already half
closed.

**3. No shipped graph runs a CFG pass, and that is correct.** Every sampler in
the shipped graphs pairs `BasicGuider` with `SamplerCustomAdvanced`; there is no
`KSampler`, no `CFGGuider`, no `DualCFGGuider`, and no `cfg` input anywhere
(*measured*). Since the checkpoint is guidance-distilled — asserted
independently by diffusers, sglang, DiffSynth and vllm-omni — this install
already agrees with all four. Recording it as a confirmed non-gap, because it
was an assumption nobody had checked.

**4. The sampler divergence in §4.1 is unpriced.** It is the one difference
here that plausibly changes output quality and has no measurement behind it.

**5. Nothing guards which qkv layout a loaded checkpoint has.** §6 shows
ComfyUI assuming contiguous unconditionally, and a wrong-layout file loading
clean and rendering noise. `bench/check_model_files.py` guards model *names*,
not their qkv order. This install only ever loads Comfy-Org repacks, so it is
not currently exposed — but the failure is silent, the discriminator is cheap
(§6's weight-scale grouping separates the two decisively), and
`docs/evidence.md:176-186` records a real file in circulation that trips it.

---

## 8. What this did not cover

- **The two VAEs.** Latent geometry only, enough to price the packed sequence.
  vllm-omni's VAE wrapper was read (§1) and no other engine's was.
- **Reference and keyframe conditioning paths.** `ref2va` and `fl2va` layouts
  were read to the extent they explain the `t2va` layout; the sizing and
  presentation rules are [`../h3_references.md`](../h3_references.md)'s.
- **Runtime and parallelism** — sequence parallelism, disaggregation,
  quantization backends, cache-DiT. [`sglang_comparison.md`](sglang_comparison.md)
  owns that axis.
- **Any rendered comparison.** Nothing here was rendered. Every claim is
  static: read from source, or measured against a file on disk.
- **Whether any divergence is visible.** Not attempted, and not answerable
  without a distribution under [`../eval_comparison.md`](../eval_comparison.md).
