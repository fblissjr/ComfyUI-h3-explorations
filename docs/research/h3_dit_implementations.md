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
[`comfyui_h3_t2va_trace.md`](comfyui_h3_t2va_trace.md) owns what ComfyUI's own
code does call by call -- loaders, quantization ops, memory, the two VAEs -- for
one t2va render; this file compares implementations, that one traces ours.
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

**Reference of record is not the same as reachable here, and the gap is
load-bearing.** diffusers cannot load the pruned curve checkpoint this install
actually renders with, and neither can vllm-omni — though the two fail
differently, and vllm-omni's is the dangerous one: it has no completeness
assertion, so it warns on the unmatched tensors, skips them, and runs with an
uninitialised `time_embedder`. That is a silent wrong render, not a refusal,
and it belongs on the §9.13 list. So for a real render on this
box the comparison set is **three**: ComfyUI, sglang and DiffSynth. Read §3.2
with that in mind: it is a comparison of the *architecture*, and where a row
describes the released bf16 path rather than the shipped artifact it now says
so. §9.7 has the detail.

**That is a limit of the shipped artifact, not of what is on disk.** The
unpruned `int8_convrot` fl2va and ref2va files are both local, and so is the
full bf16 release including all fourteen transformer shards. A five-way
comparison is therefore *reachable* on unpruned weights — it is simply not the
configuration anything here renders with. Do not read the "three" as a ceiling
on what can be compared; read it as what the shipped graph reaches. §9.8.

Naming this because the failure mode is live in this repo. The `encoder`
session found the same shape on 2026-08-29 in a different lane: an instrument
swept two snapshots and omitted the regime every shipped graph resolves to, so
a generated record answered for artifacts nobody loads and the claim about the
regime that ships had nothing measuring it. **An instrument covering the wrong
population is worse than a missing one, because it reads as coverage.**

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
`10000^(-i/16)` **recomputed in fp32** — bitwise equal there, and off by up to
3.7e-9 if you recompute in float64 and cast down, which is the obvious way to
check it and does not match (*measured*). That is exactly `1/(theta**(arange(0,32,2)/32))`
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

**Module inventory.** The 932 keys of the *shipped pruned* file resolve to the
tree — with the caveat that this artifact is not the one all five implement:
it carries `adaln_t_table` and 100 int8 sidecars and has no `time_embedder`
(§10.2). Read the shapes below as confirming the architecture, not as five-way
agreement about this file. It resolves to: 50 blocks of `{norm1, norm2, attn{q_norm,k_norm,qkv_proj,out_proj},
mlp{fc1,fc2}, adaln_proj}`, a 2-block `token_refiner` with `final_norm`, a
`final_layer` with two output heads, the three input projections, and
`rope.inv_freq` (*measured*). Shapes confirm the derived widths: `qkv_proj`
`21504 = 3*56*128`, `fc1` `28672 = 2*14336`, `video_out` `96 = 24*1*2*2`,
`adaln_proj` `96768 = 6*5376*3`, `final_layer.adaln_proj` `10752 = 2*5376*1`.

### 3.2 Agreed by all five, read in each

- **One packed sequence, full bidirectional self-attention, no mask on every
  default path.** Not unconditional: sglang passes a
  `first_segment_sparse_query_block_mask` when its `SUBBLOCK_SPARSE_ATTN`
  backend is selected, and raises if it is missing. Dense-and-maskless is the
  default everywhere, not the only reachable path.
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
  **The sinusoid half of that describes the released bf16 path, which is not
  what this box runs** — on the pruned curve checkpoint the sinusoid and its
  MLP do not exist at all, and two of the five have no code path for it. §9.7.
- **Conditioning anchors** sit at `t = 0.999` visual, `t = 1.0` audio, in every
  implementation that has them.
- **Two sigma schedules per request**, through the same map
  `sigma' = s*b / (1 + (s-1)*b)`. **The 12.0 / 3.0 pair is the release's
  declared value and every engine's call-site default — it is not an invariant
  of the code.** DiffSynth's own scheduler signature defaults to `shift=2.22`
  and is overridden by its pipeline (*read*); vllm-omni reads the pair from the
  manifest as a fallback and its Turbo path overrides the video shift
  (*relayed, not verified here*). An earlier version of this row stated the two
  numbers as if the code fixed them.
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

**Partially priced 2026-08-29.** A three-arm record on one scene
([`../../bench/results/2026-08-29_market_scene_arms.json`](../../bench/results/2026-08-29_market_scene_arms.json))
includes a pair differing **only** in `sampler_name`, `dpmpp_2m_sde_gpu`
against `euler`, at a fixed seed with PDD-emitted sigmas. Both met the brief.
`euler` with its default `s_churn=0` is exactly the reference integrator --
deterministic, eta=0, first order -- so the vendor's own integrator is viable
here. That is one data point and says nothing about which is better; what it
does buy is **repeatability**, since a same-seed repeat under `euler` is a true
repeat and under either SDE sampler it is not.

One scope limit on that record: both arms load the **AWQ v2** encoder, not what
`h3_config.MODELS["clip"]` resolves to. The sampler axis is clean because both
arms share it, but "viable here" was not measured on the shipped encoder.

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

**The shipped `simple` scheduler tracks the vendor grid closely.** The shipped
step count is **16** — the dominant `BasicScheduler` setting across the graphs,
with 8, 6 and 4 also present (*measured*). At N=16 the final Euler step spans
44.65% of the sigma range under ComfyUI `simple`, 44.44% under DiffSynth and
46.15% under the vendor construction, against 12.60% for `normal` and 18.86%
for `beta` (*measured*; naming N makes all five reproducible, which the earlier
wording did not).

**One graph does not use `simple`.** `h3_probe_turbo_768p_owner_api.json` ships
`beta` at 6 steps (*measured*) — the scheduler that diverges most from the
vendor grid. Named because this paragraph's argument is that the shipped choice
tracks the vendor, and one shipped graph is the exception. Tail
coarseness is what governs quality here — the PDD lane established that, and
`CLAUDE.md` carries it — so `simple` being the close one matters.

That corroborates the `simple` choice already argued at
`workflows/h3_config.py:199-218`, but by a different route: that note argues
from a distilled LoRA's published grid and is controlled by
`bench/check_distill_grid.py`; this argues from the vendor's own *inference
engine* construction. Two independent reasons, same answer.

### 4.3 The audio stream is carried differently

diffusers, sglang, DiffSynth and vllm-omni run **two sigma schedules** and step
the two streams separately from one joint forward.

**Corrected 2026-08-29:** this previously said "two scheduler *instances*",
which is an object-structure claim and is wrong for two of them. sglang holds
none — `minimax_h3_pipeline.py` states "scheduler intentionally absent:
model_index carries scheduler=null", generates per-modality sigmas in a
timestep stage, and its stages accept `scheduler=None`. vllm-omni is the same
shape. Two *schedules*, from one function called twice; the consequence below
is unaffected.

ComfyUI instead carries the audio latent *scaled onto the video schedule* by
`audio_scale = shift_video / shift_audio`, making the pack an ordinary
single-schedule flow latent, and undoes the scaling inside `forward()` before
the network sees it (`comfy/ldm/minimax/model.py`, `MiniMaxH3Model.forward`;
`comfy/model_sampling.py`, `ModelSamplingAV`). **This is a derivation, not an
inference, and the earlier label undersold it.** With `s = sigma_v`,
`a = time_shift_sigma(s, 12, 3)` and `scale = shift_v/shift_a`, the identity
`s*(1-a)/a == (1-s)*scale` holds exactly, so the pack is an ordinary rectified
flow in the video sigma and `forward()`'s output line collapses exactly to
`n - scale*x0_a`. **Exact per forward.**

**It is not exact under a PDD fused head**, and that qualification is
[`pdd/audio_under_pdd.md`](pdd/audio_under_pdd.md)'s to own: the transform's
coefficients are frozen at the step's own sigma while a fused head returns the
block's mean velocity. §9's worked shape is such a render — the record it cites
carries the per-block drift on that exact schedule. It is nonetheless a different arrangement, and anything
comparing raw audio latent magnitudes across engines has to divide it out.

### 4.4 Sequence padding

The vendor's reference pads the packed sequence to a multiple of 64 and splits
the tail off with `cu_seqlens = [0, used, S]` — diffusers says so explicitly
while declining to copy it. sglang, DiffSynth and vllm-omni all pad, with that
exact constant. **diffusers and ComfyUI do not pad at all.**

**Corrected 2026-08-29. This section previously called padding "numerically
inert" without qualification. That is true only GIVEN segmentation, and the
unqualified form is dangerously wrong** -- it reads as a licence to add padding
to ComfyUI, which has no `cu_seqlens` and no mask.

Padding is inert in the three engines that pad *because* `cu_seqlens` isolates
it into its own attention document. Add pad rows to an implementation that
passes `mask=None` and they are not isolated.

**The mechanism stated here on 2026-08-29 was wrong and is corrected.** It
claimed a pad row's q/k/v are zero, so its key scores 0 and `exp(0) = 1` takes
softmax mass. That skips AdaLN. `DiTBlock.forward` modulates *between* `norm1`
and the attention projection, and `_mod_scale_shift` is
`h.mul_(1 + scale).add_(shift)` — so a zero row leaves `norm1` as zero and
arrives at attention as **`shift_msa`**, not zero (*measured*: nonzero, order
of a real row's magnitude). A pad row therefore carries a real query, a real
key and a real **value**, and injects its own `v` into every real row's output.

*Measured*: appending 64 **zeroed-qkv** pad rows to a 512-row sequence under
`mask=None` moves the real rows by 6.99% relative L2. Given the correction
above that is a **floor**, not the effect — the real perturbation is larger,
because the pad rows are not zero at the kernel. The conclusion strengthens.

**So: do not add padding here without adding segmentation, and there is no
reason to add either.**

---

## 5. Divergences that cost, but do not change numbers

- **Output heads over every row.** diffusers, sglang, DiffSynth and vllm-omni
  run both heads across the whole sequence — padding included — and select
  afterwards; sglang comments that this is deliberate, to keep the GEMM shape
  and defer collectives. ComfyUI slices the two target segments first and runs
  the heads only there. Same numbers for the rows that survive. It is also why
  the padded implementations zero their condition rows with a multiply rather
  than a slice.
- **Where the packed layout is built.** ComfyUI builds it in `extra_conds`,
  once per sampling run, and the DiT rebuilds only on a signature miss; diffusers requires the caller to supply
  positions, tags and indices; sglang and DiffSynth build it in a pipeline
  stage. A consequence, not a defect: the diffusers transformer is reusable
  with any layout, and ComfyUI's cannot be driven with one it did not build.
- **`rope.inv_freq` loaded vs computed.** ComfyUI, sglang and vllm-omni read
  the buffer from the checkpoint — none of the three contains a **rope** theta
  literal. All three *do* carry a `10000.0`, for the **timestep sinusoid**
  (`comfy/ldm/minimax/model.py:143`); reading that as the rope base is the two-stages
  confusion `CLAUDE.md` warns about, and this line previously invited it by
  saying "no theta literal at all". diffusers recomputes it from `rope_theta` and drops the key.
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
ones without checking.** The release's native-format shards store qkv **per-head
interleaved**, `[q_h|k_h|v_h] x 56`; the Comfy-Org repacks store it
**contiguous**, `[q_all; k_all; v_all]`. **A "correction" on 2026-08-29 claiming
the release ships no fused qkv is withdrawn** — it read only
`coderef/MiniMax-H3/transformer/`, the diffusers-format copy. The release ships
both: that one splits into `to_q`/`to_k`/`to_v` (638 keys), and
`FL2VA/transformer/` + `Ref2VA/transformer/` are the native
`MiniMaxH3DiTModel` format (535 keys) whose key names are **identical to
ComfyUI's**, `blocks.N.attn.qkv_proj` included. How each implementation handles
it:

| | strategy |
|---|---|
| diffusers | converts interleaved -> contiguous offline, in its converter |
| sglang | branches on a quant-config flag, permutes on load |
| DiffSynth | ships **two** Comfy subclasses. `MiniMaxH3DiTComfy`'s only delta is the qkv view, `view(S, 3, heads, D)` vs `view(S, heads, 3, D)`; `MiniMaxH3DiTComfyPruned` extends it with `time_embed_dim=8`, the curve lerp and a no-SiLU AdaLN proj (§9.7) |
| vllm-omni | reorders **unconditionally**, assuming the release layout; its only guard is a row-count check (a `raise ValueError`, not an assert) against `heads * 3 * head_dim` |
| ComfyUI | splits contiguously **unconditionally**, assuming the repack layout |

The last two rows are the interesting ones: **neither sniffs the layout, and
they assume opposite things.** A release-native file fed to ComfyUI, or a
Comfy-Org repack fed to vllm-omni, loads without error and produces noise —
the row count is identical either way — `56 * 3 * 128 = 21504` under both
layouts — so vllm-omni's guard is **structurally incapable** of discriminating
them, not merely insufficient (*read*). §7 finding 5 leans on this. `docs/evidence.md:176-186` already records the
concrete instance from the ComfyUI side: a DeepBeepMeep bf16 file stores
`qkv_proj` in release order, so head 0's k rows land where ComfyUI's split
expects head 1's q.

This is already recorded at `docs/evidence.md:168-175` from an earlier session.
A spot check corroborates it and settles which layout this box holds:
splitting `blocks.0.attn.qkv_proj.weight_scale` into contiguous thirds gives
three clearly distinct means and explains ~10.6% of the total variance, while
the per-head-interleaved grouping explains ~0.03% (*measured*). **The file this
install loads is contiguous**, which is what ComfyUI's
`split(heads*head_dim, dim=-1)` requires.

**The discriminator generalises; the story about it did not.** Contiguous beats
interleaved at every block sampled — 0.106/0.0003 at block 0, 0.038/0.0005 at
25, 0.293/0.0056 at 49 — so the test is sound anywhere. But an earlier version
added "the V third is the low one, which is the physical signature, since Q and
K are RMSNormed downstream and V is not". That is true **only at block 0**: V is
the *highest* third at blocks 25 and 49 (*measured*). The mechanism sentence was
a one-block observation generalised, and is withdrawn.

---

## 7. Findings for this install

**1. The DiT architecture config is not vendored, though the release declares
it.** `vendor_config/` carries the tokenizer, the pixel bounds, the patch
geometry and both `model_index.json`s — but not `transformer/config.json` and
not the two `scheduler_config.json`s. So the architecture constants and the
eps triple are retyped as Python defaults in `comfy/ldm/minimax/model.py`
rather than read from the release. They are correct today (*measured*: every
value matches the release).

**Corrected 2026-08-29: the fix this finding originally proposed cannot reach
the code it is about.** It said vendoring the three files and adding readers
"would close it". `comfy/ldm/minimax/model.py` is **ComfyUI core, outside this
repo**, and `CLAUDE.md` records that nothing here patches core — so no amount
of vendoring changes core's constructor defaults. The overreach was also in
citing `vendor_config.py`'s header, which scopes itself to literals *in our
code*; core's are not that. What is actually available is a **check** that
grades core's defaults against the release and goes red when upstream changes
one, which is detection rather than closure. The factual half stands:
`vendor_config/` carries neither `transformer/config.json` nor either
`scheduler_config.json`.

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

**4. The sampler divergence in §4.1 is now half priced.** The reference
integrator was shown viable on one scene (§4.1). What is still unmeasured is
whether either is *better* -- that needs a distribution, not the pair. The
practical consequence is already actionable though: only the deterministic arm
gives this install a repeatable baseline.

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

---

## 9. The forward pass, stage by stage, all five

Written 2026-08-29 at the granularity where an implementation difference could
hide. Sol and sage are **off** throughout — this is stock ComfyUI native code
against the four references. Where a row says "same", the arithmetic is the
same and only the spelling differs.

The worked shape is one `t2va` render: 345 frames at 1152x768, giving
`latent_t 102`, `lat_h 48`, `lat_w 72`, 864 rows per latent frame, **88,128
video rows + 1,150 audio rows + text**, run for 8 evaluations.

### 9.1 Text encode

| | |
|---|---|
| **ComfyUI** | Qwen3-VL config is built with `num_hidden_layers = 50`, `final_norm = False`, `lm_head = False`. The stack is physically 50 layers, so "layer 50's output" is the last hidden state and no norm follows it. `enable_attention_masks = False`. |
| **diffusers** | Loads the full 64-layer stack, reads `outputs.hidden_states[50]`, and raises if the stack has 50 or fewer layers — because the last hidden state of a 50-layer stack *would be post-norm* and wrong. |
| **sglang** | **Truncates, like ComfyUI and DiffSynth.** Sets both `arch.num_hidden_layers` and `arch.text_config.num_hidden_layers` to `MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER = 50`, refuses `lm_head.weight` and `model.language_model.norm.*` on load, and sets `language_model.norm = nn.Identity()`. |
| **DiffSynth** | Truncates like ComfyUI: `num_hidden_layers = 50` and `language_model.norm = torch.nn.Identity()`. Its converter drops layers >= 50 and `lm_head`. |
| **vllm-omni** | Full stack, layer 50. |

**The one that matters:** diffusers' guard names the exact trap the **three**
truncating implementations have to avoid, and all three avoid it — ComfyUI with
`final_norm = False`, DiffSynth and sglang with an `Identity()`. Three truncate,
two read layer 50 off a full stack. Same tensor, four spellings, one of which
raises if you get it wrong.

### 9.2 Text projection and the token refiner

`condition_proj` / `context_embedder` is `Linear(5120 -> 5376, bias=True)`
everywhere. The refiner is 2 pre-norm blocks, **no RoPE and no AdaLN**, plus a
final RMSNorm, everywhere.

| | when it runs |
|---|---|
| **ComfyUI** | once per sampling run, inside `extra_conds`, result cached as `c_crossattn` |
| **sglang** | once per request, result passed back in as pre-refined `prompt_embeds` |
| **diffusers, DiffSynth, vllm-omni** | every forward |

Identical output; ComfyUI and sglang pay it once per render instead of 8 times.

### 9.3 Patchify

Every implementation reshapes `[B,C,T,H,W] -> (b,c,t,pt,h,ph,w,pw)`, permutes
to `nthwcrpq`, and flattens. ComfyUI and sglang write it as `torch.einsum`,
diffusers and vllm-omni as `.permute(0,2,4,6,1,3,5,7)`. **Same permutation.**
Row order is `(t, h, w)` with w fastest; the feature index inside the 96-wide
row is `4*c + 2*p + q`, channel-major.

Audio packing is channel-major in all five: rows `[0, audio_t)` are channel 0.

### 9.4 Packed sequence

Order is `[text | cond | audio | video]` in all five, target audio always
immediately before target video.

| | who builds it | padding |
|---|---|---|
| **ComfyUI** | `MiniMaxH3.extra_conds` in `comfy/model_base.py:2208`, once per sampling run; the DiT's `_forward` rebuilds only when the shape signature misses | none |
| **diffusers** | the caller must supply positions, tags, timestep indices and three index tensors | none |
| **sglang** | a pipeline stage | to 64, `cu_seqlens [0, used, S]` |
| **DiffSynth** | a pipeline stage | to 64, same |
| **vllm-omni** | a pipeline stage | to 64, same |

See §4.4 for why the padding is not a free thing to copy.

### 9.5 Position ids

All five: text at `(row_index, 0, 0)`; video time continuing from `float(text_len)`;
spatial axes `linspace((1-r)/2, (1+r)/2, dim//2, endpoint=False) * 32` with
`r = dim / sqrt(lat_h*lat_w)`; temporal spans `5/3 * (1,4,4,4,4)` by exclusive
cumsum; audio at `h = 0` with `w` pinned to the two extremes of the width grid,
one per stereo channel. Built in **float64** in all five.

sglang and DiffSynth both keep *two* deliberately non-unified summation orders
for the temporal span, because numpy pairwise and sequential Python summation
diverge in the last ulp from n=16. ComfyUI has one order. This is a
sub-ulp difference and is named only so nobody "fixes" it.

### 9.6 RoPE

Frequencies: `inv_freq[i] = 10000^(-i/16)`, 16 per axis, three axes to 48,
duplicated to 96. **96 of 128 head dims rotate; dims 96..127 pass through.**
Convention is halves / rotate-half everywhere — never interleaved.

| | how it is applied |
|---|---|
| **ComfyUI** | builds a rotation table `[1, S, 1, 48, 2, 2]` once per forward, then `rms_rope_split_half_` does **RMSNorm and rope in one in-place kernel** on the qkv buffer |
| **diffusers** | `cos`/`sin` tensors, then `x*cos + rotate_half(x)*sin`, after a separate `norm_q`/`norm_k` |
| **sglang** | half-width `[cos(48) | sin(48)]` cache in bf16; CUDA path calls `sgl_kernel.rotary_embedding(..., is_neox=True)`, or a fused `fused_inplace_qknorm_rope` with `round_norm_before_rope=True` |
| **DiffSynth** | `cos`/`sin` computed fp32, cast to bf16 after the trig, then rotate-half |
| **vllm-omni** | fused Triton `qk_norm_rope`, `is_neox_style=True` |

**Where a real difference could hide:** the order is norm-then-rope in all
five, and sglang's `round_norm_before_rope=True` exists specifically to make
its fused kernel match the eager order. ComfyUI's fused kernel does the same.
vllm-omni's fused Triton kernel round-trips through bf16 mid-kernel, which is
real. **An earlier version of this line added that vllm-omni "notes it is not
bit-identical to its own eager path". That attribution was wrong** — the only
bit-exactness statement in that tree is about the **video VAE**, not the DiT.
Carrying a VAE claim onto the DiT is precisely the stage-crossing trap
`CLAUDE.md` names, and it had landed under this file's headline unchecked risk.
The round-trip stands; the vendor statement does not.

### 9.7 Timestep embedding

`t = 1 - sigma` in `[0,1]`, **unscaled** — no `*1000` — and the sinusoid is
`cat([cos, sin])`, cos first, base 10000, half-width 128. All five.

**On the pruned/curve checkpoint this stage does not exist.** ComfyUI, sglang
and DiffSynth all replace it with a lerp into a sampled curve table and **drop
the SiLU before the AdaLN linear**, because the table stores the post-SiLU
curve. diffusers and vllm-omni have no curve branch at all and cannot load such
a file. See §9.8.

### 9.8 AdaLN

Chunk order `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`,
modality-major, row `= timestep_index * 3 + tag`, tags `0` video `1` text `2`
audio. Verified against the weights in §3.1. Final layer is `(shift, scale)`,
one modality, indexed by timestep alone.

| checkpoint | `time_embed_dim` | activation before `adaln_proj.linear` |
|---|---|---|
| released bf16 | 2688 | `SiLU(t_emb)` |
| **pruned / curve (what this box runs)** | **8** | **none** |

**Why the pruned form exists, and what un-pruning costs** (*measured*, over 50
blocks):

| `adaln_proj.linear` | size |
|---|---|
| pruned, `T=8` F16 (what ships here) | **0.07 GiB** |
| unpruned, `T=2688` I8 | **12.11 GiB** |
| unpruned, `T=2688` BF16 | **24.22 GiB** |

Both artifacts are on this box and the file sizes confirm the arithmetic: the
pruned fl2va int8 file is 19.53 GiB with 932 keys, an `adaln_t_table` and no
`time_embedder`; the unpruned one is 31.70 GiB with 1035 keys, `time_embedder`
present and `adaln_proj` at `(96768, 2688)` I8. The 12.2 GiB between them is
the table above.

So **the AdaLN projection is the largest tensor family in H3**, and factorising
it is what puts the model on a 24 GiB card — un-pruning costs more than the
whole rest of the DiT combined, and the unpruned BF16 projection alone exceeds
such a card. Pruning is not a convenience someone chose; it is the thing that
makes consumer inference possible.

There is a third option that is neither approximate nor huge: precompute the
modulation for exactly the timesteps a schedule visits and drop the projection.
[`sglang_comparison.md`](sglang_comparison.md) owns that mechanism and its
trade. The shape is worth knowing here because it belongs to this stage — a
cache over 16 distinct timesteps is roughly 148 MiB across all 50 blocks,
against 12.11 GiB of unpruned int8 weights. Its precondition is a schedule
known ahead of time, which a PDD-emitted grid satisfies and a percent-band
scheduler does not.

**How the modulation is applied is the largest granular divergence in the
whole forward:**

```
ComfyUI          for a, b, row in segments:            # 3 slices for t2v
                     h[a:b].mul_(1 + scale[row]).add_(shift[row])
                     x[a:b].addcmul_(other[a:b], gate[row])

all four others  scale.index_select(0, adaln_indices)  # gather over ALL rows
                 hidden * (1 + scale_g) + shift_g      # new tensors
```

ComfyUI exploits that the sequence is uniform per segment, so it does three
in-place slice ops. The references gather 89,278 rows for each of six chunks,
every block, every step. **Same arithmetic; roughly 2,400 full-sequence
gathers avoided per render, and no intermediate allocations.** On a 24 GiB card
this is not a micro-optimization.

### 9.9 The block

Identical in all five:

```
h = norm1(x);  h = h*(1+scale_msa) + shift_msa;  x = x + gate_msa * attn(h)
h = norm2(x);  h = h*(1+scale_mlp) + shift_mlp;  x = x + gate_mlp * mlp(h)
```

`scale` enters as `1 + scale`; **`gate` is applied raw** — no `1 +`, no `tanh`
— in all five. Norms are RMSNorm with learnable weight, eps `1e-5`. MLP is
SwiGLU over a fused `fc1` of width `2*ffn`; the release stores `[gate; value]`
and only diffusers swaps it (§6).

ComfyUI accumulates residuals **in place** (`add_`, `addcmul_`); the others
build new tensors. Same values.

### 9.10 Attention

One packed document, **full bidirectional, non-causal, no mask** on the default
paths (§3.2), softmax scale `128 ** -0.5`, MHA with no GQA, no bias, no sink,
no softcap. All five — but ComfyUI passes **no** `scale` argument, so its value
is SDPA's default rather than an explicit choice, where DiffSynth and sglang
pass it explicitly (*read*).

| | head layout into the kernel | terminal call |
|---|---|---|
| **ComfyUI** | `[S,H,D] -> [1,H,S,D]` (HND) | `optimized_attention(...)`, a module-level backend selection resolving to `attention_pytorch` here, which calls `comfy.ops.scaled_dot_product_attention` — a wrapper that runs inside an `sdpa_kernel` priority context above 128k elements, not raw `F.sdpa` |
| **diffusers** | `unflatten(-1,(heads,-1))` | `dispatch_attention_fn(..., attn_mask=None, is_causal=False)` |
| **sglang** | packed `thd`, no batch dim | `flash_attn_varlen_func(..., cu_seqlens, causal=False)` |
| **DiffSynth** | per-segment `[1,H,L,D]` | a Python loop over `cu_seqlens` calling SDPA per segment |
| **vllm-omni** | `split` then `view(S,heads,128)` | `flash_attn_varlen_func(..., cu_seqlens, causal=False)` |

**The qkv split differs and is the classic trap** (§6): the release stores
per-head interleaved, the Comfy repacks store contiguous. ComfyUI splits
contiguously and vllm-omni reorders unconditionally — opposite assumptions,
neither sniffs.

### 9.11 Final layer and output

| | heads run over |
|---|---|
| **ComfyUI** | the two target segments only, sliced first |
| **all four others** | every row including padding, then `index_select` |

Both output heads are **fp32 in four of the five**. DiffSynth's are plain
`nn.Linear` with no `dtype=`, so they take the module dtype (bf16), and its
`_modulate_scale_shift` ends `.to(x.dtype)` — the final layer casts *down*
rather than up.

**Sign:** ComfyUI and DiffSynth negate the DiT output so their sampler's
`x0 = x - sigma*v` reproduces H3's data-ward `x0 = x + sigma*v`. diffusers,
sglang and vllm-omni do not negate and put the `+` in the scheduler. The
negation and the scheduler sign are **one decision made in two places**.

### 9.12 The sampler step

| | |
|---|---|
| **ComfyUI** | k-diffusion `sample_euler`, `s_churn=0` so no noise; `d = (x - denoised)/sigma`, `x += d*dt`. Audio rides the video schedule scaled by `shift_v/shift_a`, converted in and out inside `forward()`. |
| **diffusers, sglang, DiffSynth, vllm-omni** | Euler eta=0 on **two sigma schedules**, one per stream (§4.3 — not two scheduler objects). |

Same integrator; ComfyUI reaches it through a graph node and one schedule
rather than a hardcoded loop and two.

### 9.13 Where a tiny difference could still hide

Ranked by how silently it would fail, and none of these is currently known to
be wrong here — this is the list worth checking first if output ever looks off:

1. **qkv layout** — loads clean, renders noise, row count identical either way.
2. **The curve checkpoint's dropped SiLU** — keep it and modulation is garbage.
3. **SwiGLU half order** — plausible-looking output, wrong.
4. **The output sign composed with the scheduler sign** — port half and the ODE runs backwards.
5. **RoPE halves vs interleaved**, and the 96/128 split — both easy to get subtly wrong.
6. **Fused norm+rope kernels** — vllm-omni states its fast path is not bit-identical to its eager path; ComfyUI's and sglang's fused kernels are the same class of risk and neither has been graded here against an eager reference on this model.

**Items 1 and 6 are both unchecked**, and §7 finding 5 already says so for
item 1: `bench/check_model_files.py` guards model *names*, and a grep of it for
`qkv|interleav|contiguous` returns zero (*measured*). An earlier version of this
line claimed item 6 was the only unchecked one, contradicting §7 in the same
document.

---

## 10. Pruned against unpruned: what each path executes, and what you gain

Written 2026-08-29 because "unpruned" reads as "more correct" and, for the
artifacts on this box, **it is not**. Both fl2va and ref2va exist here in both
forms; this section traces what each actually runs and measures the difference.

### 10.1 Which path a checkpoint selects, and where

Selection is by **observable, not filename**. `comfy/model_detection.py` looks
for an `adaln_t_table` key and, finding one, sets `dit_config["adaln_curve_grid"]`
to its row count. The DiT then branches once at construction:

```
use_adaln_curves = adaln_curve_grid is not None

pruned    register_buffer("adaln_t_table", [grid, 8] fp32)
          AdalnProj(apply_silu=False, adaln_dtype=fp32)

unpruned  time_embedder = TimeEmbedder(256 -> 5376 -> 2688), fp32
          AdalnProj(apply_silu=True,  adaln_dtype=model dtype)
```

and once per forward, on the distinct timesteps only:

```
pruned    pos = clamp(t,0,1) * (grid-1)
          i0  = floor(pos).clamp(max=grid-2)
          t_emb = lerp(table[i0], table[i0+1], pos-i0)        # [M, 8]
          -> AdalnProj: linear_8(t_emb)                        # NO SiLU

unpruned  t_emb = time_embedder(t)                             # [M, 2688]
          -> AdalnProj: linear_2688(SiLU(t_emb))
```

The dropped `SiLU` is not an optimisation: the table stores the **post-SiLU**
curve, so applying it again would square the nonlinearity. That is trap #2 in
§9.13.

### 10.2 What each file actually stores

*measured*, fl2va:

| | pruned | unpruned |
|---|---|---|
| file | 19.53 GiB, 932 keys | 31.70 GiB, 1035 keys |
| `time_embedder` | absent | present, **F32** |
| `adaln_t_table` | `[1025, 8]` F32 | absent |
| `blocks.N.adaln_proj.linear.weight` | `(96768, 8)` **F16** | `(96768, 2688)` **I8** |

Note the dtype swap, which is the whole point of this section: pruning does not
merely shrink the projection, it **moves it out of the quantised set**. Only
five weight families are int8 in the unpruned file, and `adaln_proj` is one of
them; in the pruned file it is four, and `adaln_proj` is not.

### 10.3 The measurement

Modulation output for the fifteen distinct timesteps the shipped 8-step PDD
schedule visits, against the **bf16 release** evaluated in float64 as ground
truth (*measured*):

| adaln representation | modulation rel-L2, median over the timesteps |
|---|---|
| **pruned, F16 rank-8** | **1.9e-4** (blocks 0/25/49: 1.92, 2.11, 2.15 e-4) |
| **unpruned, I8 full-rank** | **9.3e-4** |

**The pruned checkpoint's modulation is about five times more accurate than the
unpruned one's.** The rank-8 basis is a very good fit — its residual sits below
the F16 storage it lives in — while int8 costs more than the truncation saves.

**Caveat on the second row, stated because it is not tight.** That number comes
from reproducing ComfyUI's per-output-row int8 on the release tensor. It does
**not** include convrot's rotation, which exists to suppress outliers before
quantising and should therefore make the shipped file somewhat better than this
estimate. The 5x gap is wide enough that the ordering is unlikely to flip, but
the margin is uncertain and a dequantisation of the shipped tensor would settle
it. Nothing here has done that.

**This does not contradict `evidence.md` #22**, which measured first-step
*velocity* moving 5.6-9.4% between the two checkpoints. That is the model's
output after 50 blocks, not the modulation entering them, and it compares two
files differing in pruning *and* in adaln quantisation at once. The two numbers
are consistent: a 1e-4 modulation perturbation amplifying to a percent-level
velocity change across 50 blocks is exactly why a rendered clip cannot A/B a
numerical knob.

### 10.4 What unpruned actually buys

Not modulation accuracy. Three other things:

1. **PDD's adaln update becomes an ordinary weight patch.** On a pruned base it
   must either be pre-solved into the rank-8 basis (`--pruned`, residual 1.2e-5
   to 6.1e-5) or re-injected at run time through 50 forward patches. On an
   unpruned base `pdd_lora.py` renames the sidecar keys into
   `diffusion_model.blocks.N.adaln_proj.linear.lora_*` and lets
   `comfy.lora.load_lora` apply them, with the `applied != loaded` count as its
   own guard. Simpler path, and the bake residual disappears.
2. **diffusers and vllm-omni become loadable**, since neither has a curve
   branch. That is what makes a five-way comparison reachable (§2).
3. **The time embedder is evaluated exactly** rather than interpolated between
   two of 1025 grid rows. That gain is already inside the 1.9e-4 above.

### 10.5 What it costs, and the conclusion

12.2 GiB, and — on the evidence above — a 5x worse modulation. ComfyUI's
dynamic VRAM-to-RAM offload makes 31.70 GiB *runnable* on a 24 GiB card, but
the adaln projection is touched once per block per step, so it is streamed
50x8 times per render rather than held.

**So the unpruned file is not the "most correct" DiT.**

**Corrected 2026-08-31, and the withdrawn half mattered.** This paragraph used
to continue: *"and building a better quantised one does not exist as an
option — §7 of this file and the encoder lane's headroom record together
establish that convrot is deterministic and data-free with no calibration to
improve, and that the int8 files are structurally complete at 534/534 release
modules with the release's fp32 islands preserved."* Three defects in one
sentence. It carried an ENCODER record
(`bench/results/2026-08-29_int8_convrot_headroom.json`, subject
`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`) to a DiT conclusion, which
is the two-models crossing `CLAUDE.md` names. §7 establishes none of what is
attributed to it — read it. And "534/534 release modules" appears nowhere else
in this repo and is sourced to nothing.

What survives is the mechanism, and only for the weights: convrot IS a
deterministic, data-free transform, so there is no calibration population to
improve, and that is as true of the DiT as of the encoder. What does NOT
survive is the conclusion, because **the lane it rests on measures one of the
two roundings.** `int8_convrot` is W8A8: §1.7-1.8 of
[`comfyui_h3_t2va_trace.md`](comfyui_h3_t2va_trace.md) traces `int8_linear`
rotating the activation online and quantising it **per token** before an int8
GEMM whose accumulation is exact — "all the error is in the two roundings".
Every quant-quality record here
(`2026-08-21_quant_delta_*`, `2026-08-28_quant_hotspots_ref2va`,
`2026-08-29_int8_convrot_headroom`) is a stored-WEIGHT distance and is blind to
the activation rounding by construction. `docs/evidence.md` carries that
caveat on the source measurement; this paragraph dropped it and then reasoned
past it.

Two in-format levers therefore remain open rather than closed:

- **`convrot_groupsize` per module.** It is per-tensor metadata (the
  `comfy_quant` blob), currently 256 on every linear.
  `_build_hadamard` demands a power of **4** that divides `in_features`, and
  the DiT's dimensions are not uniform: `attn.qkv_proj` and `mlp.fc1` take
  5376 = 2^8·21 and are **capped at 256**, while `attn.out_proj` (7168 = 2^10·7)
  and `mlp.fc2` (14336 = 2^11·7) admit **1024**. The kind with the worst
  stored-weight error is one of the two that can take a wider group, and the
  rotation's purpose is to spread outliers before rounding — so this is
  precisely the knob a weight-only sweep could not evaluate.
- **Per-module scheme.** `int8_linear` rejects any `weight_scale` that is not
  scalar or per-output-channel, so a per-GROUP weight scale is closed without a
  core change; but precision escalation on one kind is expressible. It is
  priced out rather than impossible: out_proj at bf16 is +1.8 GiB across 50
  blocks, and fp8 is strictly worse than int8 here (0.0265 against 0.0104).

Neither is worth turning until the runtime decomposition exists —
`docs/open_experiments.md` #23.

If the goal is exact modulation, neither file is the answer. The exact AdaLN
cache in §9.8 is: bf16-exact, ~148 MiB for this schedule, built from the
release shards that are already on disk. Convert PDD without `--pruned` first —
that emits the 2688-dim adaln pairs a cache builder needs, and it exercises a
branch that has never run.
