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
`models/diffusion_models/`. Four arms have rendered at 1344x768; see
`bench/check_pdd_head_selection.py` for the defect the first four exposed.

---

## What it is

Not "the same trajectory in fewer steps". The trajectory stays a 32-point
grid. What changes is the **final output head**: `proj_out` and
`audio_proj_out` are each replicated once per interval of that grid, and one
sampling step fuses a contiguous block of those heads into a single effective
linear whose output is the block's mean velocity. So `nfe = 32 / 4 = 8`
transformer evaluations cover a 32-step trajectory.

The fusion is on **weights**, not outputs — `MiniMaxH3ParallelHead.forward`
builds one matrix per step and calls `F.linear` once — and its plan depends
only on `(shift, num_steps, block_size, step)`. All four are fixed before a
render starts, so there are only ever `nfe` distinct fused heads. That is what
lets `bench/convert_pdd_lora.py` collapse the 32-head stack offline without
approximating anything.

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

`pdd_math.block_bounds` is the one implementation of this, shared by the
converter and the node so they cannot disagree about which head a step wants.

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

### `MiniMaxH3PDDLoRA` (`pdd_lora.py`)

The three runtime surfaces. Goes where a LoRA loader goes — before
`MiniMaxH3SigmaShift`, before the attention nodes — for the reason
`workflows/build_workflows.py` already states about the turbo loaders: it
clones the ModelPatcher, and that clone belongs upstream of the sage-then-Sol
adjacency rather than inserted into it.

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

Selection matches `t_emb` against the **`nfe + 1` block-boundary embeddings**,
built once at load from the model's own arithmetic. The nearest boundary is the
block, directly. A run at a step count the file was not fused for lands between
boundaries, takes the closest available head, and is reported once in the log.

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
consumed by something they were never distilled against. The PDD arms carry
`euler` for that reason and no other.

**The sigma grid already matches, and is now graded.** `simple` is EXACT at 4
and 8 steps -- it reads the discrete 1,000-entry table and both divide 1,000,
measured in `bench/check_distill_grid.py`. That check skipped every PDD graph
until 2026-08-26 because `is_turbo` is false for a PDD filename; it now grades
them against `pdd_math.block_bounds`, which is analytic ground truth rather
than a vendor table, on both the video and audio streams.

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
need to hold one fused linear layer per block in memory."* Precomputing the
fused heads is the paper's own recommendation, not our optimisation of it --
and the shipped adapter's per-forward einsum over all 32 heads is the form the
paper says can be avoided.

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
- **He fuses per forward; we fuse at load.** The paper's section 3.1 recommends
  the latter. At this sequence length neither is measurable.
- **He accepts any step count**, blending whatever heads the step spans, where
  we refuse a count that does not divide 32. His generalises off-distribution
  silently; ours declines.

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

## Two traps that are silent in both directions

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
- **Nothing verifies the converted file against the model it will be applied
  to** beyond the partition fingerprint. A checkpoint layout change would be
  caught by `load_lora` matching nothing, which the node raises on, but a
  *partial* match would not be.
- **The off-schedule warning has never fired in a real render.** It is
  exercised only by `bench/check_pdd_head_selection.py`, on a synthetic drive.
  Its threshold is reasoned, not calibrated. It now guards only "is this render
  on the fused schedule" -- selection itself no longer depends on a tolerance --
  but the head-selection defect above is still the reminder that a silent
  warning is not evidence the selection is right.

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
