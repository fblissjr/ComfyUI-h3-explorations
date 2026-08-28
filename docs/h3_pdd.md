# Parallel Decoding Distillation on MiniMax H3

**The acceleration LoRA that is not a step distillation, and the three
mechanisms it needs that a LoRA loader does not have.**

alibaba-pai published `MiniMax-H3-FL2VA-Acc-8Step` and
`MiniMax-H3-Ref2VA-Acc-8Step` under Parallel Decoding Distillation (PDD,
arXiv 2607.26004). Their inference adapter, `minimax_h3_pdd.py`, ships beside
the weights and is the authority for the mechanism; everything below that
describes *their* design is read from that file, not inferred.

Written 2026-08-26 against the published weights, ComfyUI's
`comfy/ldm/minimax/model.py`, and the checkpoints in
`models/diffusion_models/`. Substantially revised 2026-08-27, when the node
stopped asking for a step count and started reading it off the sampler; where a
section reads as a correction of an earlier design, that is the one.

Arms have rendered at 1344x768 at both 8 and 4 evaluations; see
`bench/check_pdd_head_selection.py` for the defect the first four exposed.

---

## What it is

Not "the same trajectory in fewer steps". The trajectory stays a 32-point
grid. What changes is the **final output head**: `proj_out` and
`audio_proj_out` are each replicated once per interval of that grid, and one
sampling step fuses a contiguous block of those heads into a single effective
linear whose output is the block's mean velocity. Eight evaluations at block
width 4 cover the 32-interval trajectory; four at width 8 do too, from the same
weights.

The fusion is on **weights**, not outputs — `MiniMaxH3ParallelHead.forward`
builds one matrix per step and calls `F.linear` once — and its plan is a
function of `(shift, num_steps, start, stop)` alone: no hidden state, no
dependence on what the previous step produced. That is what makes a fused head
a thing you can compute once and reuse, and it is why `pdd_math.fuse_block` is
exact rather than an approximation of the reference's per-forward einsum.

**What it is NOT a function of is the step count**, and that is the whole of
why this document changed on 2026-08-27. A block is a span between two grid
points, and which grid points a render visits is named by the sampler's
schedule — which does not exist at patch time. So the converter ships the
32-head bank verbatim (`h3_pdd.bank.{video,audio}`) and the node fuses each
span the first time a step asks for it. An earlier design collapsed the stack
to `nfe` heads inside the converter; that pinned a step count into the
artifact, and it is gone.

One published file therefore carries three mechanisms that reach the model on
three different surfaces:

| | what | surface here |
|---|---|---|
| backbone | attn + MLP LoRA, 50 blocks and 2 refiner blocks | weight patch |
| adaln | `adaln_proj.linear` LoRA, per block | weight patch either way: in the 2688-dim time space on an unpruned base, pre-solved into the curve basis on a pruned one |
| heads | the per-interval output heads | per-step swap of two `final_layer` linears |

---

## The property that makes it cheap to wire

**The block boundaries are the plain 8-step shifted schedule, bit for bit.**
Not close to it. `linspace(1, 0, 33)[::4]` *is* `linspace(1, 0, 9)`, and
`shifted_sigma` is pointwise, so subsampling the 32-point grid at the block
size lands exactly on the schedule a normal 8-step render already uses.
Verified to `torch.equal` at 32/4 and shift 12, and again for audio at shift 3.

The release's own scheduler configs declare shift 12.0 video and 3.0 audio,
which is this repo's `SIGMA_SHIFT` and the base checkpoint's own training
shift. So a PDD arm moves **the step count and nothing else** — no
`MiniMaxH3SigmaShift` change, no scheduler change. Contrast the 768p turbo
LoRA, which was distilled at 6/3 and therefore changes two things at once,
which `docs/h3_ref2v_distillation.md` records as the reason it is harder to
attribute.

`pdd_math.pdd_time_grid` is the one implementation of the grid, and
`fusion_plan` the one implementation of a block's weights. Both consumers go
through them, so the converter and the node cannot disagree about what a block
means. `block_bounds` is the closed form for the uniform case and is now used
only by the checks — the node reaches the same boundaries through
`schedule_knots`, and `bench/check_pdd_head_selection.py` asserts the two agree
to `torch.equal` at every divisor, which is what keeps the closed form honest
as a reference rather than leaving two answers in the tree.

---

## What we built, and why not just a LoRA file

**The published files load nowhere in ComfyUI.** Their module names are
diffusers-side (`transformer_blocks.N.attn.to_q`) and their suffixes are bare
`lora_down` / `lora_up`, which matches none of the six patterns
`comfy/weight_adapter/lora.py` accepts. Every tensor is skipped with a log
line and the render completes as an undistilled 8-step pass — which looks like
a bad LoRA rather than an unapplied one.

### `bench/convert_pdd_lora.py`

Everything that can be decided once, offline. Emits one file holding the
backbone in ComfyUI generic-LoRA naming, the adaln pairs in a neutral
namespace, the published per-interval head bank, and a fingerprint.

**One file carries one adaln form.** `--pruned` names the base it is for, so it
also decides which form to emit: the baked curve-basis patch for a pruned
checkpoint, or the 2688-dim pairs plus the `silu(t_emb)` grid for a full-width
one. Shipping both was around 40% of the file dead in the only configuration
this repo renders, and two representations of one update with nothing saying
which the node used.

Four backbone transforms, each verified numerically against the release
weights before the script existed (2026-08-26, block 0 of the fl2va partition,
ComfyUI's own `TensorWiseINT8Layout.dequantize` against
`MiniMaxAI/MiniMax-H3`):

- q/k/v fuse into `attn.qkv_proj` — A concatenated, B block-diagonal, rank
  tripled and alpha tripled with it so the applied `alpha / rank` is held.
- `attn.to_out.0` → `attn.out_proj` and `ff.net.2` → `mlp.fc2`, renames.
- `ff.net.0.proj` → `mlp.fc1` **with the output halves swapped**. The release
  stores SwiGLU as `[value; gate]`, ComfyUI as `[gate; value]`. Measured
  unswapped against swapped: 1.41 relative against 0.009.
- `adaln_proj.linear` maps by the same name with no permutation.

The residual on each of those comparisons is int8 quantisation noise, which is
what makes the mapping claim a measurement rather than a code reading.

### `pdd_math.py`

The schedule arithmetic, with no ComfyUI import, so the part worth testing can
be tested without a server. One copy: a drift between the converter's fusion
and the node's selection is a silent wrong-head, and sharing the module is what
makes it impossible rather than merely unlikely.

Four functions carry the design. `pdd_time_grid` is the grid. `base_sigma`
inverts the flow shift, which is what makes a grid position readable from a
sampler's sigma at all — the grid is uniform in BASE sigma, not in `t` and not
in the shifted sigma, so undoing the shift is the only way back to an index.
`schedule_knots` turns a sigma schedule into the grid points it lands on.
`fusion_plan` is the paper's `D_k` over a span, and `fuse_block` applies it to
one block; `fuse_heads` is the uniform case built on top, kept because the
checks want the whole stack at once.

`fusion_plan` takes an END index rather than a width as of 2026-08-27, for the
reason the rest of this section keeps arriving at: a schedule-derived block is
a span between two knots and is not always as wide as its neighbours.

### `MiniMaxH3PDDLoRA` (`pdd_lora.py`)

The runtime surfaces. Goes where a LoRA loader goes — before
`MiniMaxH3SigmaShift`, before the attention nodes — for the reason
`workflows/build_workflows.py` already states about the turbo loaders: it
clones the ModelPatcher, and that clone belongs upstream of the sage-then-Sol
adjacency rather than inserted into it.

It installs four object patches when the heads are on, and they are not four
copies of one idea:

| patch | what it is for |
|---|---|
| `diffusion_model.forward` | observe `sample_sigmas` and delegate. The only patch that is not about heads |
| `final_layer.forward` | bookkeeping: pick this step's block, then call the stock forward |
| `final_layer.video_out.forward` | the swap |
| `final_layer.audio_out.forward` | the swap, on the other stream's shift |

Plus the weight patches, which go through `comfy.lora` and need no patch point
at all. **`patch_heads=False` installs none of the four** — the control arm
runs the backbone and adaln updates against the checkpoint's own heads, and
does not need the schedule, so it does not observe it either.

---

## Where the step index comes from, and why not theirs

The vendor arms its heads from a `register_forward_hook` that increments a
counter once per forward and wraps at `nfe`. Nothing ties that counter to the
schedule. One extra evaluation — a CFG uncond pass, a warmup, a shape probe,
`torch.compile` tracing, an offload dry run — desyncs it for the rest of the
render, and the wrap-around hides the desync instead of raising it.

We derive the step from `t_emb`, which is what the model was actually called
with, so it cannot desync. `FinalLayer.forward` is handed **separate rows for
the video and audio streams**, and PDD runs those on separate schedules, so the
per-stream split falls out of the model's own signature instead of being
threaded through.

Selection matches `t_emb` against the **block-boundary embeddings**, built from
the model's own arithmetic. The nearest boundary is the block, directly.

Those embeddings are built at EVERY point of the 32-interval grid, not at the
boundaries of one step count, and the tracker indexes that array by the knots
it derives. It has to be that way round: at load the node knows the grid,
because the grid is in the file, and does not know which subset of it are
boundaries, because that is the schedule's business. Building all 33 costs one
`cdist` table and removes any need to rebuild when the schedule arrives — or
when it changes mid-graph, which `split_at` two-pass sampling does.

### The step COUNT is a different question, and it used to be a widget

Selecting a head needs two things and only one of them is in `t_emb`: where the
step starts, and how far it goes. The extent is a property of the sampler's
schedule, and at patch time that schedule does not exist — `BasicScheduler` is
downstream of every model-patch node in the graph.

Until 2026-08-27 the node closed that gap with an `nfe` widget the person had to
keep equal to `BasicScheduler.steps` by hand, and the only thing standing behind
that requirement was a warning that fires after sampling has already started.
A requirement with a warning behind it is not a control, and this one had never
fired in a real render.

It now reads `transformer_options["sample_sigmas"]` — put there by
`comfy/samplers.py` — and maps each sigma back through `pdd_math.base_sigma` to
a grid index. Those knots are the block boundaries:

| schedule | knots derived |
|---|---|
| any count dividing the grid | exactly `block_bounds`, verified to `torch.equal` |
| a count that does not divide it | uneven and reported, e.g. 5 steps → `[0, 6, 13, 19, 26, 32]` |
| `denoise < 1.0` | starts partway down, which a count could not express at all |

Reaching that dict costs one more patch point than the head swap does, so
`diffusion_model.forward` is patched to observe it and delegate. Sol-Attn
composes with `.forward` patches whose owner segment contains `attn`
(`vendor/sol_attn_minimax.py`), so it leaves this one and the `final_layer` one
alone rather than gating them behind its sigma window. The patch chains onto
whatever forward is already installed.

The heads follow: rather than fusing `nfe` of them at load, the node fuses each
`(start, stop)` span the first time a step asks for it and caches it. A render
visits at most `nfe` distinct spans, so this is still the paper's "one fused
linear per block" and not a per-forward einsum — it just cannot know which
blocks until the sampler names them.

`nfe` survives as an override that forces uniform blocks and ignores the
schedule, for deliberately decoding one partition while stepping another. Every
shipped graph carries 0.

### Reading the schedule was still the wrong direction, and 0.83.0 inverted it

Everything above is about *recovering* the schedule correctly, and it does.
What it cannot do is stop the schedule being wrong in the first place, because
the knobs that decide it -- `scheduler`, `steps` -- live on `BasicScheduler`,
downstream of this node. Three ways to be off the grid followed, and each was
caught, if at all, by a static check over the SHIPPED graphs, so a hand-edited
or hand-built graph had nothing at all.

The node now also **emits** the schedule. `MiniMaxH3PDDLoRA` has a `SIGMAS`
output, and every shipped non-split PDD graph wires it straight into
`SamplerCustomAdvanced` with no `BasicScheduler` in the graph. The sampler
steps at the boundaries the heads were fused for; there is no scheduler widget
left to set wrong, and off-grid is not expressible.

**It moves no render, and that is checked rather than asserted.** The output is
`1 - pdd_time_grid`, which is `shifted_sigma` over `linspace(1, 0, nfe + 1)` --
the plain shifted schedule for the block count. Against ComfyUI's own
`calculate_sigmas` over `ModelSamplingAV`:

| steps | shift 12 and shift 6 |
|---|---|
| 2, 4, 8 | bit-identical to `simple` (`torch.equal`) |
| 16 | ~2e-3 apart; `simple` quantises against its 1,000-entry table because `1000 % 16 != 0`, and the closed form is the more correct of the two |

No PDD graph runs 16. `bench/check_pdd_sigmas.py` grades all of it, including
that the graphs actually consume the output -- perfect sigmas nothing reads
would be worth nothing.

The new `steps` input is **inert at its default of 0**, which is what keeps a
deliberately off-grid arm working: 0 means the file's own count and never
refuses, so a graph driving `BasicScheduler` at a count that does not tile the
grid is untouched and still reports itself at run time. A non-zero request must
tile the grid and raises otherwise -- at such a count no on-grid schedule
exists, so there is nothing honest to emit. The first version raised
unconditionally and would have refused a 6-step render in flight while it was
written, which is why the asymmetry is there.

What this does **not** reach: `sampler_name` -- the euler requirement is still
enforced by nothing -- and `strength`. Sol's `end_percent` is still derived from
the step count at build time, so hand-editing steps still leaves it stale.

#### The shift is now a second place the schedule is decided, and it is asserted rather than removed

The PDD node sits UPSTREAM of `MiniMaxH3SigmaShift`, so the schedule it emits is
built from the shift recorded in its own file, not from the graph's shift
widget. While `BasicScheduler` owned the schedule it read the shift off the
patched model and followed the widget. Those agree on every shipped graph and
would diverge the moment a PDD graph was set to another shift -- the sampler
stepping one curve while the model integrates another.

**Removing the widget from PDD graphs was measured and is inert**: with
`MiniMaxH3SigmaShift` deleted outright, two runs came back pixel-identical to
the settled group, and `comfy/supported_models.py`'s H3 entry declares
`shift 12.0 / audio_shift 3.0` as the default for the model class, so this is a
property of every H3 checkpoint and not of the one that was rendered.

**It was not removed anyway, and the reason is worth stating because it argues
against the obvious move.** `check_distill_grid.py` and
`check_distill_settings.py` both read the graph's shift off that node, and both
are cheap static checks that touch no model file. Delete the node and the shift
has exactly one authority left -- the PDD file's metadata -- so either those two
checks start opening safetensors, or `h3_config` grows a constant that is a
second copy of what the file already says. Removing one duplication would
create another, in a worse place.

**Closed at run time, 2026-08-28, and without moving anything.**
`MiniMaxH3SigmaShift` writes its value into `transformer_options`
(`comfy_extras/nodes_minimax_h3.py`), and this node already patches
`diffusion_model.forward` to read that dict for `sample_sigmas`. So the
mismatch is observable at the first forward with no node reordering, no
generator change and no check surgery. `_StepTracker.check_shift` RAISES,
naming both numbers -- there is no legitimate arm on the other side, because
the fused heads are themselves a function of the shift. The static assertion
stays as the build-time half; this is the half that reaches a hand-edited
graph.

**The first version of that guard was silent on exactly the render it was
written for**, and the reason is worth keeping. It latched on a one-shot flag,
and the tracker lives in the ModelPatcher, which ComfyUI's execution cache
keeps across prompts in a session. A passing shift-12 render set the flag; the
shift-6 graph queued next reused the cached patcher and skipped the check
entirely, rendering to completion. Verified both ways on the card: with the
latch, a 6.0 graph rendered after a 12.0 one; without it, the same ordering
raises. That is `queued_arms.md`'s "a render is not a pure function of its
graph", met inside a guard written to catch a different silence -- so the check
is per-forward and unlatched.

**This replaced a selector that recovered a `t` and bucketed it, and the
replacement is why that bug class is gone rather than guarded.** The old one
read the nearest row of the 1025-row curve table, which quantises `t` to about
1e-3, so a `t` sitting exactly ON a boundary came back a fraction below it and
membership returned the previous block — wrong at two of eight steps, silent,
and it shipped four renders before a deliberate drive against real inputs found
it. The first fix was a snap tolerance, which then had to be justified against
the table's own quantisation. Matching the boundaries removes the question:
there is no intermediate `t` to quantise and exactly `nfe` answers to choose
between, so **selection needs no tolerance at all**. The tolerance that remains
guards a different question — whether this render is on the fused schedule.

**This is the only reason the node patches `final_layer.forward` at all.** That
patch is pure bookkeeping and delegates to the stock forward; the actual head
swap is two separate patches on the output linears. So the modulation maths
stays upstream's and cannot silently diverge from it.

---

## Replicating the reference

The vendor ships `predict_ref2v.py` and a scheduler, and three things about
how they consume the fused heads are worth matching rather than approximating.

**Euler, not `er_sde`.** `coderef/diffusers/src/diffusers/schedulers/scheduling_minimax_h3.py::step` says it takes "one
Euler (`eta = 0`) step", and their adapter defines the fused head as "the mean
velocity of one block, which an Euler step over the block boundaries consumes".
`er_sde` injects noise and uses a different update rule, so the heads would be
consumed by something they were never distilled against.

**This is a requirement, not a preference, and the distinction matters because
a preference could be traded away.** A fused head is not "a velocity that an
Euler step happens to suit" — it is *defined* as the block's mean velocity,
which is the quantity a single Euler step across that block integrates exactly.
Consume it with anything that re-noises between boundaries and the head is
answering a question the sampler did not ask. The `p` axis below is the same
point from the other side: sub-stepping within a block is provably redundant
under a first-order solver, and stops being redundant only under a stochastic
one, which is off-distribution.

`workflows/h3_config.py` made euler/simple the default for every distilled arm
on 2026-08-27, which is a separate and weaker argument about distilled models in
general. PDD required euler before that policy existed and would require it if
the policy were reversed.

**The sigma grid already matches, and is now graded.** `simple` is EXACT at 4
and 8 steps -- it reads the discrete 1,000-entry table and both divide 1,000,
measured in `bench/check_distill_grid.py`. That check skipped every PDD graph
until 2026-08-26 because `is_turbo` is false for a PDD filename; it now grades
them against `pdd_math.block_bounds`, which is analytic ground truth rather
than a vendor table, on both the video and audio streams.

**Corrected 2026-08-27**: it took the evaluation count from the converted
file's `pdd_nfe`, which was right while the graph carried an `nfe` widget and
went red on every correct 4-step arm the moment the widget stopped carrying
one. It now takes the sampler's own `steps`, treats a non-zero `nfe` as an
override, and requires the count to divide the grid for a shipped arm.
`bench/check_distill_settings.py` had the same premise and the same correction.

**Dense DiT attention is the reference configuration**, not a handicap. Their
pipeline is Diffusers' `ModularPipeline` on stock SDPA, and running dense costs
about 2.4x on this workload -- 70.3 s/it against 28.7 with sage+Sol -- because at ~90k packed
tokens attention is quadratic and dominates everything else. Worth stating
plainly: **that makes a dense 8-step PDD render slower than a sage+Sol 16-step
base render.** The step count was never the expensive part at this sequence
length. These arms pay it to be comparable to the reference; it is not the
configuration to render production clips in.

That is a DiT-side statement throughout. The Qwen3-VL encoder resolves its own
attention inside its decoder forward and is untouched either way -- sage, Sol
and the SLA router all take a `MODEL` input and cannot reach it.

### The `p` axis they ship and never use

`MiniMaxH3ParallelHead.forward` builds `einsum("pn,noi->poi", plan, W)` and
flattens, so one forward can emit `p` separate velocity fields, and `set_plan`
accepts any `p`. `pdd_sampling_plan` only ever builds `p=1`.

That is correct, and the reason is worth recording so nobody "improves" it:
every head in a block reads the SAME hidden state, so the block has no internal
feedback and `sum(dt_n * v_n)` is identically `sum(dt) * weighted_mean(v)`.
Measured 2026-08-26 at 3.7e-16 relative -- floating-point noise. Taking four
sub-steps buys exactly nothing over one mean-velocity step.

It stops being redundant only if something nonlinear happens between sub-steps,
which is what a stochastic sampler does. So the latent capability is a 32-step
stochastic trajectory at 8 NFE. That is off-distribution from how the heads
were distilled and nothing here has tried it.

## Validated against the paper and an independent conversion

Three references, and they fail differently, which is why all three were used.
`bench/compare_pdd_conversions.py` re-runs the numeric half;
`bench/results/2026-08-26_pdd_conversion_*.json` records it.

**The paper.** Section 3.1 gives the fused layer as
`W_{n:n+L} = sum_k D_k W_k` with `D_k = (t_{k+1} - t_k) / (t_{n+L} - t_n)`,
which is `pdd_math.fusion_plan` term for term, and Algorithm 1 gives the
consumer: `u = student(x_n, t[n])` then `x_n += einsum('k,k...', h_n, u_n)` --
evaluated at the block-START time, deterministic, no noise. That is what the
`euler` sampler and the boundary-matched head selection implement.

It also settles a design question rather than leaving it to taste: *"during
inference we can avoid the extra compute of an enlarged final layer and we only
need to hold one fused linear layer per block in memory."* Holding one fused
linear per block is the paper's own recommendation, not our optimisation of it,
and the shipped adapter's per-forward einsum over all 32 heads is the form the
paper says can be avoided.

Note what the recommendation does and does not pin down. It says *hold* one per
block, not *precompute at conversion time* -- so `_FusedHeads`, which fuses a
span on first use and keeps it, satisfies it exactly. That is what let the step
count move out of the artifact without giving anything up: a render visits at
most `nfe` blocks, each is fused once, and the sampling loop is a dict lookup
from the second pass onward.

**Kijai's converted files**, an independent conversion of the same weights
arrived at without reference to ours. **They are re-uploaded in place and have
changed encoding once**, so every figure below is a dated record rather than a
property: `bench/results/2026-08-27_pdd_conversion_{fl2va,ref2va}.json`, which
now carries a sha256 of each input so a later reader can tell whether it is
reading about the same artifact. Re-run with `bench/compare_pdd_conversions.py`.

Read as of 2026-08-27, ref2va, both streams:

| | ours against his |
|---|---|
| `attn.qkv_proj`, `out_proj`, `mlp.fc1`, `mlp.fc2` | 0.0, except 6.7e-20 on qkv |
| his head bank against the published 32-stack | ~1e-10, video and audio |
| our bank against the published 32-stack | 0.0, weight and bias, both streams |
| fusing either bank, at 8 / 4 / 2 evaluations | ~1e-11 |

Bit-identical on every backbone transform, including the two that are easy to
get wrong -- the block-diagonal qkv fusion and the SwiGLU half-swap. The head
rows were exact until 2026-08-27 and are now ~1e-10 because his repackaging
factors the bank through a bf16 matrix; ours is stored verbatim at the
published bf16 and stays exact.

### His head bank changed encoding on 2026-08-27

Not the weights -- the packaging, and it is the more interesting change.

| | until 2026-08-27 | now |
|---|---|---|
| key | `final_layer.{stream}_out.set_weight` | `lora_up` / `lora_down` / `reshape_weight`, and the same for `.bias` |
| what core must learn | a `set_weight` / `set_bias` path that does not exist | nothing; `comfy/weight_adapter/lora.py` already has `reshape_weight` |

Under the new encoding the tensor he ships **is not the bank**. Core applies
that path as `pad_tensor_to_shape(weight, reshape) + up @ down`, padding with
ZEROS, so what is stored is the bank minus the padded base head: the first
`out` rows are `head_0 - base` and the rest are heads 1..31 verbatim, and `up`
is a full-rank square factor because an arbitrary matrix has to be expressed
through a path that only ever multiplies two. Reconstructing it is what
`bench/compare_pdd_conversions.py::kijai_bank` does, and the row above is that
reconstruction against the published stack.

> **In question as of 2026-08-28, reported and not verified here.** A peer
> session reading `Comfy-Org/ComfyUI#15908` reports that after commit
> `bd016b75ff9b` its head formula is only correct if the stored rows are
> DELTAS from head 0, that the alibaba-pai copies on this box are NOT deltas
> (verbatim heads, exact-zero difference against the published values), and
> that the HF repo's `lastModified` is two minutes after that commit -- so
> upstream appears to have re-uploaded and our local copies are stale. If that
> holds, he has adopted our semantics and the paragraph below is no longer
> true of his current code. **Nothing here has re-fetched the artifact or
> re-run `bench/compare_pdd_conversions.py` against it**, which is what would
> settle it; the paragraph is left standing rather than edited, because the
> evidence for the change is a source read on a moving target and the evidence
> for the paragraph was a measurement. `docs/research/pdd/pdd_implementations.md`
> carries the peer's fuller account. Re-fetch before any further cross-check.

**That padding also makes his `strength` mean something different from ours
below 1.0.** Heads 1..31 scale from zero rather than from the checkpoint's own
head, so a half-strength render decodes every block after the first with a
half-magnitude head. Ours interpolates each head toward the base head, so 0.0
is exactly the base model. Both are correct at the vendor's default of 1.0.
`ComfyUI-MiniMaxH3-PDD-Mamad8` reaches our conclusion independently and says
why in `blend_with_native`: the exported projections are complete block
velocities, not additive residuals, so scaling them directly scales the whole
Euler displacement.

**His pruned build projects the adaln update into the curve basis too**, which
is the part this repo worked out alone and most wanted a second opinion on.
Two independent solutions of the same projection, each scored against ground
truth rather than against each other. ref2va, read 2026-08-27:

| block | ours | his |
|---|---|---|
| 0 | 7.8e-05 | 8.3e-05 |
| 25 | 8.1e-05 | 7.8e-05 |
| 49 | 2.8e-05 | 3.7e-05 |

Same method, both far below bf16 resolution. The gap is storage precision and
nothing else: he keeps bf16 factors, we store the fp32 product, which is also
the smaller of the two because the projected rank is 8 and the factored rank is
64.

**The `ours` column above was wrong until 2026-08-27**, and the way it went
wrong is worth keeping. It read 2.3e-05 / 6.1e-05 / 1.2e-05, which matched no
record on disk -- the `his` column matched
`bench/results/2026-08-26_pdd_conversion_ref2va.json` exactly while the `ours`
column came from an earlier run, against a converted file that was rebuilt
later the same evening. The fl2va record has the same shape and is worse: it
was written at 18:12 and committed at 18:17, and its subject was reconverted at
18:28, so **that record describes a file that no longer exists**. Same failure
as `build_workflows.py`'s -- nothing is true of an artifact until it is rebuilt,
and a measurement taken before the rebuild is a measurement of something else.
The sha256 rows added on 2026-08-27 are what makes the next instance visible
instead of silent.

## Core is learning this, and what that costs us

Comfy-Org/ComfyUI#15908, "MiniMax-H3: Support PDD LoRA", by the same author as
the converted files above. Read 2026-08-27: **open**, and its diff is
`comfy/ldm/minimax/model.py` alone. The description also names a
`comfy/lora.py` change adding `set_bias` beside `set_weight`; that is not in
the diff any more, which is consistent with the encoding move above -- the
`reshape_weight` path he switched to needs no LoRA change at all.

What it does: `FinalLayer.forward` computes
`n = video_out.weight.shape[0] // out_features`, takes the original path when
`n == 1`, and otherwise reads `transformer_options["sample_sigmas"]`, finds the
current step by `argmin` against it, takes `sigma_next`, maps both back through
`time_shift_sigma(s, shift_v, 1.0)` to base-grid indices, and fuses the spanned
heads with an `einsum` inside the forward.

Two design differences from ours, and neither is a defect in his:

- **He derives the block from the sampler's schedule; we derive it from
  `t_emb`.** His inverts the video shift to recover the base grid, which is
  correct -- the grid is uniform in base sigma and the position is
  shift-invariant, which is also why one index serves both streams while the
  `dt` weights differ per stream. It does mean the selection depends on
  `sample_sigmas` being present and on the sampler evaluating only at scheduled
  sigmas. Ours reads what the model was called with and needs nothing threaded
  in.
- **He fuses inside every forward; we fuse each block once and cache it.** The
  paper's section 3.1 asks for one fused linear per block held in memory rather
  than an enlarged final layer evaluated per step, which is what the cache is.
  At this sequence length the arithmetic is not measurable either way — the
  reason to prefer the cache is that there are only `nfe` distinct answers and
  computing them repeatedly is a place for them to differ.
- **Both of us accept any step count.** This bullet said we refuse one that
  does not divide 32; that stopped being true on 2026-08-27. The node takes
  whatever spans the schedule names, reports uneven widths in the log, and
  refuses only a non-dividing `nfe` OVERRIDE — because an override forces
  uniform blocks by definition and so has to tile. Neither implementation
  declines the arm; ours says out loud that it is off-distribution.

**If it merges, our node breaks — and it broke loudly, which is the good
case.** The PR widens `FinalLayer.forward` to
`(x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts)`. We
object-patch that method, which replaces it outright, so a four-parameter
replacement drops three arguments the stock forward now requires:
`TypeError` on the first sampling step. Fixed 2026-08-27 by making the patch
arity-transparent, and `bench/check_pdd_head_selection.py` section 3 asserts it
against both signatures. That case is graded: pinning the patch back to four
parameters turns it red, and running that violation is what showed the case
raised `TypeError` past `check()`'s `AssertionError` handler and aborted the
run instead of reporting a named failure.

Our converted file leaves `video_out.weight` its original size, so a merged
core takes its `n == 1` path and our two output-linear patches still own the
head swap. That is correct and it is also two implementations of one mechanism
in one process. **The question that becomes live on merge is whether this node
should keep the head half at all**, or narrow to the conversion plus the
partition guard and let core do the rest. Not decided.

## A third implementation, and it is ahead of us on guards

`silveroxides/ComfyUI-UtilsCollection` ships `UC_MiniMaxH3PDDAcc`. Read at HEAD
`5bac35be3d61` on 2026-08-27; PDD landed there `23ab5f2dd4f6` (2026-08-26) and
was last touched `2ede53355074` the following day, "remove PDD filename
validation" -- they moved off a filename branch, the same correction CLAUDE.md's
2026-08-22 rule describes. Diagrams and the full comparison:
[`docs/research/pdd/README.md`](research/pdd/README.md).

It is a complete implementation, not generic plumbing: grid arithmetic, an
in-node converter for the published layout, head fusion, the adaln rebase for
pruned checkpoints, block selection, patch installation. **Third independent
agreement on the arithmetic** -- the same four backbone transforms including the
SwiGLU half-swap and the block-diagonal qkv fusion, explicit `.alpha` tensors,
and the same affine adaln solve.

**Where they are ahead**, and it is the guards rather than the maths:

| they enforce | us, as of 2026-08-27 |
|---|---|
| a partial patch-key match raises | **adopted.** `add_patches` returns the keys it matched; a shortfall against what `load_lora` resolved raises and names the first unmatched |
| refuses to stack on an existing `final_layer` object patch | **adopted.** `head_patch_clash` refuses when any of the three head keys is taken |
| head shapes checked against the live model | **partly.** The partition check tests shape before distance, which catches an enlarged bank; a genuinely mismatched one still surfaces as a torch broadcast error |
| unconsumed keys in the published file are an error | open, converter-side |
| an off-grid sigma RAISES by default, `clamp` opt-in | open, and **not** obviously worth taking -- see below |

Two adopted, one partly. The unconsumed-keys row is worth taking and is cheap.

The off-grid row is the one to leave. Raising mid-render costs a whole render,
and since 2026-08-27 the case it guards is narrower than theirs: our blocks come
FROM the sampler's schedule, so "off grid" no longer means a step count
mismatch — it means a sampler evaluating at a time its own schedule does not
contain. Ours has never fired in a real render. A guard that has never fired,
whose failure mode is a warning, is not obviously improved by making it fatal.

**Their interface solves the step count from the other side.** The node has a
`SIGMAS` output: it emits the block boundaries and you wire them into the
sampler, so the count exists in one place by construction. We reached one source
of truth from the opposite direction, by reading the sampler's schedule at run
time. Theirs is the simpler graph; ours needs no rewiring and composes with a
scheduler the user already has. Both beat a widget.

**Two things they do that we cannot.** An uneven partition by construction --
`nfe=6` as `(8,8,4,4,4,4)` -- and a hard restriction of block sizes to the
trained width and twice it. That restriction is a claim we have not tested and
which our own check should not be read as refuting; see the last row of
**Enforced by nothing**.

**Where we still lead:** the partition fingerprint. Ours remains the only one of
the three that notices a Ref2VA file loaded onto an fl2va checkpoint, which is
silent in both of the others because the key sets are identical.

**They will break twice on #15908**, harder than we would have: their
`pdd_final_forward` is installed as a four-argument `MethodType` *and* copies
core's modulation body, importing the private `_mod_row`. The copy would
silently diverge rather than fail. Ours delegates to the stock forward for
exactly that reason.

Reported, not verified: their basis match is `torch.allclose(atol=1e-6)` against
the model's `adaln_t_table`, three orders tighter than our `TABLE_TOLERANCE`.
Our own note records shipping 1e-3 there and finding it *below* the 1.6e-3 a
bf16 cast costs. Whether theirs bites depends on their target checkpoints, and
nobody here has run it.

## Three traps that are silent in both directions

### Two things cannot own the output heads

`add_object_patch` is last-writer-wins per key, and the head swap lives on
`final_layer.video_out.forward` and `.audio_out.forward` while the bookkeeping
lives on `final_layer.forward`. So a second implementation installing its own
swap does not collide loudly — it wins, silently, and the render looks entirely
normal with one implementation's bookkeeping driving the other's heads.

**The node refuses rather than chaining, and the choice is not arbitrary.**
Chaining works for the capture patch on `diffusion_model.forward`, which only
observes and delegates, so stacking observers is harmless and it does chain
there. It cannot work for the heads: `video_out` produces one tensor, and two
things claiming to produce it means one of them is not. There is no compose
that makes both right, so the honest move is to decline and say which key is
taken.

The case that needs no other pack installed is two of this node in one chain.
Beyond that, at least two other ComfyUI implementations patch the same
attribute for their own PDD artifact families, so the collision is a property
of the patch point rather than of what happens to be in `custom_nodes/` on a
given day. `head_patch_clash` is a free function taking the patch mapping, so
`bench/check_pdd_head_selection.py` grades the predicate without a loaded H3 —
including that an unrelated block-attention patch does NOT trip it, which
matters because sage and Sol patch those on every shipped graph and a sloppy
predicate would refuse every render.

Guard adopted from `silveroxides/ComfyUI-UtilsCollection`.

### The partitions have identical key sets

`docs/h3_ref2v_distillation.md` records it: fl2va and ref2va share every tensor
name, so a Ref2VA LoRA loads onto an fl2va checkpoint with zero unmatched keys
and renders. Nothing errors and nothing logs.

The converter stores `final_layer.video_out.weight` from the checkpoint it was
converted against, as the tensor `h3_pdd.base_video_out`, and the node refuses
a load whose live tensor sits further than `PARTITION_TOLERANCE` away by
relative Frobenius distance. That tensor is fp32-unquantised and **bit-identical
across pruned/unpruned and across `int8_convrot`/`fp8_scaled`**, verified
2026-08-26 over the six H3 checkpoints in `models/diffusion_models/`, so one
value names the partition for every variant we ship. The two partitions sit
about 0.05 apart.

**Compared by distance, not by hash, and this document said "sha256" until
2026-08-27.** The first version did hash it, and that version fired on the
first real render against the CORRECT checkpoint: ComfyUI casts on load, and a
cast changes every bit while moving the value a few thousandths. An exact test
against a value the loader is allowed to transform cannot separate "wrong
partition" from "loaded normally" -- a control reporting red on correct state,
which CLAUDE.md calls worse than no control. A distance separates a cast from a
partition swap by an order of magnitude and can say how far off it was.
`bench/check_pdd_head_selection.py` pins the tolerance between the two.
A sha256 of that tensor survives in the file's metadata as a label; it is not
what the node checks.

Either way it is a branch on an observable, not on a filename — the rule
CLAUDE.md adopted 2026-08-22 after the tokenizer-constant escape.

**Shape is tested before distance, and that order was earned.** On 2026-08-27,
running Comfy-Org/ComfyUI#15908 locally, a graph that loaded Kijai's PDD file
left `final_layer.video_out.weight` resident at `[32*out, in]` on the cached
model. The next graph through this node read the enlarged tensor and died
inside the subtraction with `size of tensor a (3072) must match tensor b (96)`
— a broadcast error naming two numbers and explaining nothing. The check now
tests shape first and says what an enlarged head means and how to clear it.

The reason that happens at all is the difference between the two designs, and
it is the strongest practical argument for ours. An approach that ENLARGES the
weight has changed the module, and the change outlives the graph that asked for
it, because ComfyUI caches the patched model. An approach that patches the
projection's `forward` leaves the weight alone, so nothing it does can be
inherited by the next graph. We patch forwards; that is not a stylistic
preference.

### The pruned base has nowhere to put the adaln delta

Our default checkpoints are pruned, and a pruned checkpoint replaces the
2688-dim time embedding with an 8-column curve basis plus `adaln_t_table`. The
adaln LoRA's input space does not exist there, so `load_lora` would drop all 50
modules with a log line and apply the other 208.

**It turns out the delta fits that basis.** `convert_pdd_lora.py --pruned`
pre-solves it there -- an affine fit, basis plus a constant column, because the
pruned form is an SVD of the CENTRED time curve and the mean lives in the bias
-- so the update becomes an ordinary `diff` / `diff_b` weight patch. Measured
per block over the 1025-row grid: **worst 1.1e-4 relative**, roughly thirty
times below bf16's own resolution, and refused at conversion time above 1e-3.
That removes 50 forward patches, a `cdist` per forward, the per-call casts, and
lets strength compose through ComfyUI's own patch path.

The first version of that measurement omitted the centring and reported
0.93-0.99, i.e. "cannot be baked". The positive control caught it: the BASE
adaln curve scored just as badly, which cannot be true of a basis fitted to it.

**The bake is basis-specific, and a fit residual cannot tell you so.** Baking
ref2va's delta against fl2va's table fits just as well and writes without
complaint -- both bases are SVDs of very similar smooth curves, so they span
nearly the same subspace and differ in their COORDINATES. That bake is 0.0205
wrong at runtime against 0.0001 for the right one. The guard is the node
comparing the stored table against the loaded checkpoint's own, at a tolerance
that has to sit above a dtype cast (0.00164) and below the partition gap
(0.01835).

A file converted WITHOUT `--pruned` carries the 2688-dim pairs and the
`silu(t_emb)` grid instead, for a full-width base, and the node injects at run
time. That grid is partition-specific for the same reason the bake is: the
fl2va and ref2va time curves differ by 7.8% relative, so it is derived from the
same checkpoint that supplies the fingerprint rather than bundled once and
reused.

---

## What was borrowed, and from where

Both are credited at the point of use in the source, not only here.

- **The runtime adaln injection** reimplements the approach in
  `ComfyUI-MiniMax-H3-Turbo` (`_inject_adaln_egrid`, `_make_adaln_forward`),
  which solved the same pruned-checkpoint problem for the v4 turbo pack —
  including the diagnosis of *why* it must patch a `.forward` attribute rather
  than wrap the module: a wrapper injects `.base.linear.weight` into the
  parameter tree, ComfyUI's streaming loader records that path in its backup,
  and by unload the object patch has reverted the module so the path no longer
  resolves. We do not import it. Its bundled grid is fl2va-only, and ref2va is
  the arm this exists for.
- **The per-device head cache** is
  `ComfyUI-MiniMaxH3-PDD-Mamad8::PDDHeads.for_device` — an independent
  ComfyUI implementation of the same two-projection swap, for a different PDD
  artifact family (displacement heads over a 256-interval bank, with a student
  LoRA loaded separately through the normal path). It selects its head from the
  sampler's sigma through a diffusion wrapper and reads
  `transformer_options["sample_sigmas"]`, where we recover the time from
  `t_emb`. Theirs gets an exact sigma and the full step schedule; ours is
  self-contained in one patch point and needs no wrapper. **Neither has been
  graded against the other here.**

One independent cross-check did land: our derived fl2va `silu(t_emb)` grid
agrees with the grid `ComfyUI-MiniMax-H3-Turbo` ships to 0.0017 relative,
which is bf16 storage noise. Two implementations, one number.

---

## Measured, 2026-08-26

All against the published PDD files and the release weights, on this box, no
render involved.

| quantity | value |
|---|---|
| backbone perturbation, `\|\|BA\|\|/\|\|W\|\|` per module type | 0.004 to 0.015 |
| fused head against the checkpoint's own head | 0.005 early, 0.015 at the last step |
| step-to-step change in the fused head | 0.004, rising to 0.019 at the final transition |
| mean of the 32 heads against the base head | 0.004 video, 0.002 audio |
| fl2va against ref2va, same PDD file position | no tensor identical, 728 of 728 |

For scale, `docs/h3_ref2v_distillation.md` measured the 8-step turbo LoRA at
0.00036 and the fl2va→ref2va checkpoint gap at 0.042 — on dequantised pruned
fp8 files, where these are on the release originals, so those are comparable in
order of magnitude and not in digits.

Two readings follow, and both are predictions rather than results:

- PDD perturbs the backbone far harder than the official turbo LoRAs, and it
  moves the modulation path they leave alone entirely.
- Dropping the heads entirely — running the backbone LoRA alone — is a small
  error concentrated in the last two steps, where this schedule takes its
  biggest jumps. That is a cheap arm and the control for the head machinery.

---

## Enforced by nothing

Doc-local, and deliberately not merged into `docs/checks.md`'s standing audit:
that table sweeps the imperatives in `CLAUDE.md` and says so, and widening its
scope silently is how a table stops meaning what its header claims.

- **The `strength` semantics are asserted nowhere.** The node scales all three
  mechanisms and 0.0 installs nothing at all, so it is exactly the base model.
  Nothing checks that, and a check would need a loaded model to be worth
  anything -- asserting it against a stub would grade the stub.
- **The runtime adaln injection cannot run on either file this repo ships, and
  the node says the opposite on its way into it.** `--pruned` deliberately pops
  the raw `h3_pdd.adaln.*` tensors and `h3_pdd.silu_temb_grid` -- about 40% of
  the file, dead on a pruned base -- so both shipped files carry ONLY
  `h3_pdd.adaln_baked.*`. Of the three adaln paths `MiniMaxH3PDDLoRA`
  implements, exactly one is reachable with them: the unpruned path installs 0
  modules and trips the declared-vs-installed guard, and the table-mismatch
  path logs "falling back to the runtime adaln injection, which is correct on
  any pruned base" and then raises `KeyError` on `h3_pdd.silu_temb_grid`.
  Unreachable today, because the head guard refuses a cross-partition file
  first. It matters because
  [`research/pdd/2026-08-27_handoff.md`](research/pdd/2026-08-27_handoff.md)
  used that same claim to argue the adaln "already takes care of itself" if the
  head guard were relaxed -- **it does not**, and relaxing the guard exposes
  the `KeyError` rather than a slow path. Found 2026-08-28 by diffing the two
  shipped files' key inventories, which are otherwise identical: 0 keys unique
  to either side, the silent trap this document already records, confirmed live.
- ~~**Nothing verifies the converted file against the model it will be applied
  to** beyond the partition fingerprint. A checkpoint layout change would be
  caught by `load_lora` matching nothing, which the node raises on, but a
  *partial* match would not be.~~ **Closed 2026-08-27.** The node now compares
  what `add_patches` returned against what `load_lora` resolved and raises on a
  shortfall, naming the first unmatched keys. Adopted from
  `silveroxides/ComfyUI-UtilsCollection`, which had this guard while we had the
  row admitting we did not.
- **The off-schedule warning has never fired in a real render.** It is
  exercised only by `bench/check_pdd_head_selection.py`, on a synthetic drive.
  Its threshold is reasoned, not calibrated. Since 2026-08-27 it guards a
  narrower question than it used to: not "is the step count right", which is now
  taken from the sampler and correct by construction, but "is the model being
  evaluated at a time this schedule contains". The head-selection defect above
  is still the reminder that a silent warning is not evidence the selection is
  right.
- **Nothing grades whether a legal step count is a SENSIBLE one.**
  `check_pdd_head_selection.py` asserts that 16, 8, 4 and 2 evaluations each
  select their own blocks, and they do -- but that is a statement about the
  selector, not about the arm. `silveroxides/ComfyUI-UtilsCollection` restricts
  block sizes to the trained width and twice it, on the reasoning that a block
  of 16 heads averaged into one velocity is a long way from what the block-4
  distillation was trained to produce. Nothing here has measured whether they
  are right, and our check should not be read as endorsing 2 NFE.

---

## What the renders established, 2026-08-26

Four arms at 1344x768 -- ref2va PDD at 243 frames, its head-free control, and
243/345/192-frame lengths. All completed; boundary-residual warnings silent
throughout. Three md5 comparisons across the pre-fix and post-fix runs settle
what the logs could not, because every arm is seeded and the pipeline is
deterministic:

| comparison | result | what it establishes |
|---|---|---|
| head-free arm, before and after the boundary fix | **bit-identical** | the fix is confined to the head-selection path; it moved nothing else |
| full arm, before and after the fix | differs | the wrong-head defect was live in a render, not only in the offline drive that found it |
| full against head-free, after the fix | differs | the head patch reaches the output at all |

The first row is the one worth keeping. A fix that changes the arm it targets
and leaves the arm that does not use that path untouched has demonstrated its
own scope, which no amount of reading the diff can.

**The instrument was weaker than two of those claims, noted 2026-08-27.** Those
md5s are of the `.mp4` container, and these graphs carry `save_metadata: True`,
so the workflow JSON is embedded in the file. Two payloads differing by nothing
but a `filename_prefix` produce different containers with identical frames. The
implication only runs one way: **identical container implies identical frames,
so row one stands unharmed.** Rows two and three argue from "differs", which a
container hash cannot establish — those conclusions are very likely right for
other reasons, but the evidence cited does not reach them. Nothing was re-run;
the rows are kept as the record of what was done. Compare decoded frames
(`ffmpeg -f rawvideo | md5sum`) if this question ever needs answering again.

## What the renders established, 2026-08-28: the SIGMAS rewiring is inert

Verified on the card, at 1344x768 x 39 frames, t2v PDD 4-step, one seed. Arms
differ ONLY in where the sampler's sigmas come from. Compared on **decoded
pixels**, never on file bytes -- see the methodology note below, which this
session re-learned the hard way.

| group | arms | what they share |
|---|---|---|
| settled 4-step | 4 old-wiring runs + 4 new-wiring runs | **pixel-identical** |
| 2-step | node emitting `steps=2`, and `BasicScheduler(simple, 2)` | **pixel-identical** to each other |
| 4-step vs 2-step | -- | differ |

Three things follow, and the third is the one that could have embarrassed the
change:

- **The rewiring moves no render.** Old wiring and new wiring agree exactly, at
  two step counts, on eight settled runs. This is the end-to-end form of the
  `torch.equal` result `bench/check_pdd_sigmas.py` proves offline.
- **The output is genuinely consumed.** Changing the node's `steps` moves the
  pixels. Had it not, the SIGMAS output would be decorative and the whole
  change a no-op wearing a fix's clothes.
- **A non-dividing `steps` is refused before sampling.** `steps=6` failed at
  the PDD node with the divisor message; the executed-node list contains the
  loaders and no sampler. The trained-envelope warning also fired in a real
  render at `steps=2` -- the first warning this node has ever emitted outside a
  synthetic drive.
- **The preserved path still works, which is the point of preserving it.** A
  graph left on `BasicScheduler` at `denoise=0.5` with SIGMAS unwired renders,
  the node's `steps` stays inert at 0, and the tracker derives **width 4**
  blocks -- four evaluations over half the trajectory, not the width 8 a
  full-trajectory 4-step run gets. That is the case the SIGMAS output cannot
  express and the reason the observe path was not retired with the rewiring.

### Two methodology traps, both of which caught this session

**A render here has a warm-up transient, so a matched pair is not an
instrument.** The first render or two after a state change differ from the
value the same configuration settles on, and it happens to BOTH wirings -- one
old-wiring run and one new-wiring run were each a one-off before both settled
onto the shared value. Read against a single pair, this looks exactly like "the
change moved the render", and it was read that way here for several minutes.
Only repeated, interleaved runs separate the two. **The paragraph above the
2026-08-26 table calls the pipeline "deterministic"; that is true of the
settled state and not of the first run after a restart or a batch boundary.**

**File bytes are not pixels, and this doc already said so.** The comparison was
first run over `.png` bytes from `SaveImage`, which embeds the prompt JSON --
and the arms differ in `filename_prefix`, so every file differed while the
frames were identical. That is the same trap the 2026-08-27 note below records
for `.mp4` containers, met again in a different format one day later. Decode
first, hash second; the note's advice was right and reaching for a different
file format did not escape it.

## Not measured

- **Whether the fused heads change the output in a way anyone can see.** They
  reach it -- the table above -- but reachability is not quality, and
  CLAUDE.md's rule stands: two arms differing in a numerical knob are
  different samples, not a degraded version of one sample, so a pair cannot
  answer "better". That needs the blind multi-seed process in
  `docs/eval_comparison.md`, and nobody has run it.
- Whether the head-free arm is worth running, which is the first thing to find
  out and the reason the head deltas above were measured.
- Whether PDD transfers the ref2v difficulty `docs/h3_ref2v_distillation.md`
  describes. Its Fact C — no ref2v training path exists — now has a second
  independent counterexample, and unlike the lightx2v turbo this one is
  distilled on `transformer_ref` itself, so Fact B does not bite it. Facts A
  and B are otherwise untouched by this document.

---

## See also

- [`docs/h3_ref2v_distillation.md`](h3_ref2v_distillation.md) — why ref2v
  resists step distillation, and the measurements this table is scaled against
- [`docs/checks.md`](checks.md) — the check index and the uncontrolled-requirement audit
- [`workflows/h3_config.py`](../workflows/h3_config.py) — `PDD_FL2VA_LORA`,
  `PDD_REF2VA_LORA`, `PDD_STEPS`, `PDD_SHIFT`
- [`docs/research/pdd/README.md`](research/pdd/README.md) — this page's
  mechanisms drawn against Kijai's, through the four shipped PDD graphs. A
  teaching surface that restates numbers **this** document owns, generated by
  nothing and read by no check, so it goes stale silently
