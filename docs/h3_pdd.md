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
only by the checks — **corrected 2026-08-29: the node is a consumer too**,
through `emit_sigmas`, which is `1.0 - block_bounds(...)`. The node also reaches
the same boundaries through
`schedule_knots`, and `bench/check_pdd_head_selection.py` asserts the two agree
to `torch.equal` at every divisor, which is what keeps the closed form honest
as a reference rather than leaving two answers in the tree.

---

## The two conversion forms, and which file is which

One PDD source converts two ways, and the adaln encoding is the whole
difference. Both carry the same backbone and the same 32-head bank.

| file | adaln form | size | loads on |
|---|---|---|---|
| `minimax_h3_{fl2va,ref2va}_pdd_8step_comfy.safetensors` | baked into the checkpoint's rank-8 curve basis | 1069 MiB | the **pruned** base only |
| `minimax_h3_fl2va_pdd_8step_adaln2688_comfy.safetensors` | the 2688-dim pairs, plus `silu_temb_grid` | 1594 MiB | **either** base |

The name says the difference: `adaln2688` carries the modulation update in the
full time space rather than pre-solved into a curve basis. That is what makes
it the portable one -- on an unpruned checkpoint the node applies it as an
ordinary weight patch, and on a pruned one it takes the runtime injection. The
baked file is smaller and cheaper and remains the default, but it is
basis-specific: it fits only the checkpoint whose `adaln_t_table` it was solved
against, which is what the table guard checks.

**Built 2026-08-29, and it was the first execution of that path.** Every file
shipped before it was a `--pruned` conversion, so the converter's no-`--pruned`
branch and the node's unpruned branch had both never run. The conversion
succeeded first time: 208 backbone modules, 50 adaln modules, 728 source
tensors all consumed, bank rows verified verbatim, 780 tensors out.

It is also the first file carrying `h3_pdd.adaln.blocks.N.alpha`. Nothing
emitted one until this path was exercised, and without it those 50 modules take
ComfyUI's fallback scale of 1.0 -- right only while alpha/rank is 1.0, which is
the coincidence the explicit backbone alphas exist to refuse.

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
at all. **Corrected 2026-08-29: `patch_heads=False` installs ONE of the four.**
This previously said "none of the four … and does not need the schedule, so it
does not observe it either". The `diffusion_model.forward` capture patch is
installed *unconditionally*, outside the gate — moved there on 2026-08-28
precisely because `h3_probe_ref2v_pdd_headfree` ships `patch_heads: false` and
was the one arm missing the shift guard. The three head patches are gated; the
observer is not, and the shift-guard section below depends on that. — the control arm
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
output, and every shipped non-split PDD graph but one wires it straight into
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

The `steps` input is **inert at 0**, which is what keeps a
deliberately off-grid arm working: 0 means the file's own count and never
refuses, so a graph driving `BasicScheduler` at a count that does not tile the
grid is untouched and still reports itself at run time. A non-zero request must
tile the grid and raises otherwise -- at such a count no on-grid schedule
exists, so there is nothing honest to emit. The first version raised
unconditionally and would have refused a 6-step render in flight while it was
written, which is why the asymmetry is there.

#### It composes with the sigma nodes, and `denoise < 1.0` needs no fallback

**An earlier version of this section was wrong about that**, and said to leave
SIGMAS unwired and use `BasicScheduler` for a partial trajectory. Not needed:

| path | result |
|---|---|
| `SIGMAS(8)` -> `SplitSigmasDenoise(0.5)` -> `low_sigmas` | `[0.9231, 0.878, 0.8, 0.6316, 0.0]` |
| `BasicScheduler(simple, 4, denoise=0.5)` | the same vector, `torch.equal` |
| both rendered at a matched seed | **pixel-identical** |

And the composed form is the better of the two, for a reason the arithmetic
agreement hides: **every entry of the emitted vector IS a block boundary**, so
an index split lands on the grid by construction rather than by
`BasicScheduler`'s `int(steps/denoise)` happening to come out even.
`SplitSigmas` composes the same way, which is what a two-stage PDD arm would
use -- none ships, so the split graphs keep `BasicScheduler` and this is
untested there.

#### The boundary warning was firing once per SESSION, not once per schedule

Found while measuring the above, and it invalidated a measurement in the middle
of taking it. Two arms with a bit-identical truncated sigma vector, queued back
to back: the first warned, the second was silent. `self.warned` is a latch set
in `__init__` and the tracker outlives the render -- ComfyUI's execution cache
keeps the ModelPatcher across prompts -- so the second graph to go off-boundary
in a session got nothing. `_adopt` now resets it, so the budget is one warning
per schedule.

**Corrected: two instances, not three.** An earlier version of this paragraph
lumped the 2026-08-26 head-selector defect in with these, and it does not
belong -- that was `t` recovered by a quantising table lookup, a numerical
precision bug, and it produced a WRONG PICTURE. These two produced no wrong
picture at all. They produced **silence**: a guard that had stopped reporting,
read as "all clear". That is the more expensive shape by this repo's own
standard, and it is a different defect from the first.

The class these two share is exact and structural. `_StepTracker` is a mutable
object held by the ModelPatcher; ComfyUI's execution cache keeps that across
prompts in a session; so any attribute carrying per-render state survives into
the next render.

**That is now an assertion rather than a thing to remember.** All sixteen
attributes were enumerated: eight are immutable load-time configuration and
are correct to persist, seven are per-render and every one is reset by
`_adopt`, and `_key` must survive because it is the schedule identity `observe`
compares against. `bench/check_pdd_head_selection.py::no state outlives its
schedule` parses the class and asserts the mechanical form of that -- assigned
outside `__init__` implies assigned in `_adopt` -- with `_key` as a named
exemption graded for necessity. Static rather than a runtime probe on purpose:
driving a live tracker grades the fields that exist today, and the field that
will bite is the one somebody adds later. Shown red by reintroducing the exact
unreset latch, which it names.

With the latch fixed, a clean session shows **neither** denoise path warning --
so a truncated PDD schedule is on-grid both ways, and the one warning that
started this was a render queued after a RAISED render in the same session,
which is contaminated state rather than an off-grid schedule.

What this does **not** reach: `sampler_name` -- and that gap is worse than
"enforced by nothing", see below --  -- and `strength`. Sol's `end_percent` is still derived from
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

## What the authors ship, and what a step count actually changes

**They ship 8 evaluations and nothing else, and it is not a recommendation --
there is no step knob in their API.** Read from
`coderef/alibaba-pai_MiniMax-H3-Acc-LoRAs/` rather than inferred:

| where | what it says |
|---|---|
| `coderef/alibaba-pai_MiniMax-H3-Acc-LoRAs/minimax_h3_pdd.py::apply_pdd_lora` | `nfe = num_steps // block_size`, derived from the file's own config and returned to the caller. Not a parameter |
| its `DEFAULT_PDD_CONFIG` | `pdd_num_steps: 32`, `pdd_block_size: 4` -- so `nfe` is 8 |
| `predict_ref2v.py` | calls the pipeline with `num_inference_steps=nfe + 1`, i.e. whatever the file dictates |
| `README.md` | "Official **8 Step** Acc LoRA", and its comparison grid puts the 8-step Acc LoRA against LightX2V's **4-step turbo** |

All of that is about what alibaba-pai's *inference script* exposes, and an
earlier version of this section drew the wrong conclusion from it -- that 4
evaluations were "our own extrapolation". **The paper says otherwise about the
method**, and it is saved beside this file
([`arxiv_2607.26004v1`](research/pdd/arxiv_2607.26004v1_Parallel_Decoding_Distillationfor_Fast_Image_and_Video_Generation.md)):

- The **abstract**: *"By varying the block size during training, PDD supports
  sampling with different number of function evaluations (NFEs) during
  inference."* Variable NFE is the designed feature.
- **Table 1** lists PDD's NFE as *Variable*, against Pi-Flow's *Fixed*.
- §5: `L_min` and `L_max` *"determine the set of available NFEs at inference
  time"*, and for **LTX-2.3 -- the 22B joint video+audio model, the closest
  analogue to H3 in the whole paper** -- they are chosen so the available NFEs
  are **4 and 8**.

The H3 files declare `pdd_num_steps 32` and `pdd_block_size 4`, giving
`nfe = 8`, which is exactly consistent with `L_min = 4, L_max = 8` -- the same
{4, 8} set the paper chose for the multimodal model. **So 4 evaluations is most
likely a TRAINED configuration for these artifacts.**

What stops that being settled: `L_max` appears nowhere in the released
metadata, and the vendor script derives `nfe = num_steps // block_size` and
offers no choice -- an inference-script limitation rather than evidence about
the weights, but it does leave the trained ceiling unpublished. **Status: very
likely trained, certainly not shown worse, not confirmable from what was
released.** Recovering `L_max` is the one thing that would settle it.

### The sweep, 1 to 10 evaluations

Generated against `pdd_math` and `h3_config`, not recalled:

| steps | tiles the grid | block width | vs trained width 4 | the node | Sol `end_percent` |
|---|---|---|---|---|---|
| 1 | yes | 32 | past 2x | emits | 0.9 (default) |
| 2 | yes | 16 | past 2x | emits | 0.9 (default) |
| 3 | **no** | -- | -- | **raises** | 0.9 (default) |
| 4 | yes | 8 | **2x, the edge** | emits | 0.74 |
| 5 | **no** | -- | -- | **raises** | 0.9 (default) |
| 6 | **no** | -- | -- | **raises** | 0.83 |
| 7 | **no** | -- | -- | **raises** | 0.9 (default) |
| 8 | yes | 4 | **trained** | emits | 0.87 |
| 9 | **no** | -- | -- | **raises** | 0.9 (default) |
| 10 | **no** | -- | -- | **raises** | 0.9 (default) |

Emitted sigmas at shift 12, where a schedule exists at all:

    1 step   1.0000, 0.0000
    2 steps  1.0000, 0.9231, 0.0000
    4 steps  1.0000, 0.9730, 0.9231, 0.8000, 0.0000
    8 steps  1.0000, 0.9882, 0.9730, 0.9524, 0.9231, 0.8780, 0.8000, 0.6316, 0.0000

### Three things move when you change the step count, and two are invisible

- **The block width**, which is the obvious one. 8 -> 4 steps doubles it: each
  evaluation now averages eight of the 32 distilled heads into one velocity
  instead of four.
- **`nfe` does NOT react.** It is an independent override and stays whatever it
  was. The two are allowed to disagree and that disagreement is `nfe`'s whole
  purpose; see its tooltip. Every shipped graph carries 0.
- **Sol's `end_percent`**, which is a BUILD-TIME lookup in
  `h3_config.SOL_END_PERCENT_BY_STEPS` and reacts only if the generator runs.
  Change `steps` on a loaded graph by hand and this goes stale silently -- the
  edge `h3_config` already documents. Note the table has entries for 4, 6 and 8
  only; any other count silently takes `SOL_RECOMMENDED_CUDA`'s 0.9, which is
  the value whose wrongness at 8 steps created the table in the first place.
- **Whether a schedule exists at all.** Only divisors of 32 have one. The sweep
  is mostly `raises`, and that is correct rather than restrictive: at 5 steps
  there is no on-grid schedule to emit, so there is nothing honest to hand the
  sampler.

### The uniform partition is not the only legal one, and it may not be the best

Asked 2026-08-28: can an optimal block partition be *derived* rather than
chosen? For `nfe` the answer is trivial -- its optimum is `nfe=0`, which takes
the blocks from the schedule. The interesting question is whether the SCHEDULE
should be uniform, and the answer is measured and surprising.

**Every contiguous span of the 32-point grid has a well-defined fused head** --
that is the whole point of shipping the bank rather than `nfe` pre-fused heads.
So a partition does not have to be uniform; it only has to be a subset of the
grid knots. Minimising the worst per-block fusion loss over ALL partitions
(dynamic programming over the 528 spans, `dt`-weighted RMS of each block's heads
against their own fused head):

| nfe | uniform widths | worst | optimal widths | worst | |
|---|---|---|---|---|---|
| 4 | `[8,8,8,8]` | 0.0309 | `[28,2,1,1]` | 0.0105 | 2.95x |
| 8 | `[4]*8` | 0.0335 | `[1,1,2,1,23,2,1,1]` | 0.0098 | 3.40x |

Audio behaves the same way, 1.80x at nfe=4 and 3.10x at nfe=8.

`[28,2,1,1]` reads as degenerate and it is not -- **the uniform partition is the
lopsided one.** Under shift 12 the time grid is heavily back-loaded, so as a
share of the trajectory each Euler step integrates:

    uniform [8,8,8,8]    2.7%   5.0%  12.3%  80.0%
    optimal [28,2,1,1]  36.8%  18.7%  16.5%  27.9%

At four evaluations the standard schedule spends one step on 80% of the
trajectory and the other three on 20% between them. So on both axes measured --
fusion loss, and arc integrated per step -- the non-uniform partition looks
better, and both are legal subsets of the same grid.

> **MEASURED 2026-08-28, AND THE PREDICTION BELOW IS REFUTED.** Graded against
> the 32-point trajectory itself (`steps=32` is block width 1 -- every published
> head on its own interval, no fusion -- so it IS what a coarser partition
> approximates, and the comparison needs no human). Same seed, 39 frames.
> **The noise floor is exactly 0.00000 on both streams across three runs of one
> arm**, so every number here is real (reproduced across a server restart):
>
> | partition | video rel L2 vs the 32-step trajectory | audio |
> |---|---|---|
> | uniform `[4]*8`, 8 evals | 0.458 | 0.984 |
> | uniform `[8,8,8,8]`, 4 evals | 0.531 | 1.103 |
> | `[28,2,1,1]`, 4 evals | **0.552** | 1.014 |
>
> The fusion-loss optimum is **further** from the trajectory than uniform, not
> closer. So minimising `dt`-weighted head-to-mean deviation does not predict
> fidelity, and the weight-space metric was simply the wrong objective -- which
> was the first read here, then talked out of on the arc-length argument, and
> the measurement sides with the first read. The arc-length reasoning below is
> left standing because it is correct about the schedule being back-loaded; it
> just does not follow that rebalancing it helps.
>
>> **SUPERSEDED 2026-08-28, and know which half.** These were taken at
>> `length=39`. That puts the packed sequence at 12,226 rows — **62 below
>> `SolAttnMiniMax`'s `min_tokens` of 12,288 — so Sol was INERT** and no arm ran
>> the production attention path. The graph also carries the market prompt that
>> [`prompt_audit.md`](prompt_audit.md) verdicts `rewrite` and the owner
>> disqualified as a sample.
>
>> **Survives:** the ordering and the exactly-zero same-arm floor. Every arm
>> shares prompt, length, canvas, seed and Sol state, so a shared confound does
>> not flip an ordering — the `[28,2,1,1]` refutation stands.
>
>> **Does not:** every magnitude as a production number, and specifically the
>> audio-versus-video ratio, which is untested with Sol active and 9x the video
>> rows. The audio *share* is not the problem: 1.06% at 39 frames against 1.11%
>> at 362. `bench/grade_pdd_partitions.py` now defaults to 362.
>
> `bench/grade_pdd_partitions.py`,
> [`bench/results/2026-08-28_pdd_partition_fidelity.json`](../bench/results/2026-08-28_pdd_partition_fidelity.json),
> which also carries what these magnitudes do NOT license.

> **RE-RUN AT 362 FRAMES, 2026-08-28, AND THE ORDERING INVERTED.**
> `bench/results/2026-08-28_pdd_partition_fidelity_362.json`, on the `fast`
> tier (1152x768) with Sol active and the rewritten market prompt:
>
> | arm | video rel L2 | at length 39 |
> |---|---|---|
> | opt4 `[28,2,1,1]` | **0.52253** | 0.552, the worst |
> | mix6 `[4,4,4,4,8,8]` | 0.53634 | not run |
> | u4 / u4b / u4c | 0.53709 | 0.531 |
> | u8 | **0.54198** | 0.458, the best |
>
> The best and worst arms swapped ends, and the spread collapsed from about
> 0.094 to 0.019. The same-arm floor is again **exactly 0.00000** on three
> byte-identical repeats, so this is not noise.
>
> **So the caveat above is itself withdrawn.** It said the length-39 ORDERING
> survived because every arm shared prompt, length, canvas, seed and Sol state.
> That reasoning was wrong: a shared confound cannot flip an ordering *within* a
> run, but an ordering is not a property of the partitions alone -- it is a
> property of the partitions at a length, a canvas and a prompt, and it did not
> survive changing them. **Quote neither ordering.**
>
> What the re-run supports is weaker and more useful: at production settings
> every arm sits 0.52-0.54 from the reference with `max|d|` = 1.0, which is the
> different-sample regime rather than a degraded-version-of-one-sample regime.
> **This metric does not separate these partitions at 362 frames.** Read that as
> a null result, not as a new winner -- 4% on one seed is not a ranking. The
> `[28,2,1,1]` refutation recorded earlier the same day is withdrawn with the
> rest; it was a length-39 result.


**The paper explains the refutation, which the measurement alone could not.**
§3's training rule is the constraint the DP search ignored: *"during
training, we consider multiples of `L_min` for indices `n` of initial states
and sample `k` in `{n, ..., n + L_max - 1}` inside each block."* So a legal
partition is not any subset of the knots. It needs **every block to START at a
multiple of `L_min`** and **no block wider than `L_max`**.

`[28,2,1,1]` violates both, on nearly every block:

| block | start | multiple of 4? | width | within `L_max`? |
|---|---|---|---|---|
| 0 | 0 | yes | 28 | no, by a mile |
| 1 | 28 | yes | 2 | yes |
| 2 | 30 | **no** | 1 | yes |
| 3 | 31 | **no** | 1 | yes |

A width-28 block asks the decoder for head offsets 0..27 from one hidden state,
where training only ever sampled offsets below `L_max`. So the measured result
is not a surprise about the metric alone -- the partition was outside the
training distribution in a way the fusion loss cannot see, because fusion loss
is computed from the head weights and knows nothing about which spans were
trained.

**Our legality rule is not the paper's, and it is wrong in BOTH directions.**
`resolve_emit_steps` requires the count to divide 32 uniformly. The paper
requires only that blocks start at multiples of `L_min` and stay within
`L_max` -- which permits mixed widths. Taking the inferred `L_min = 4,
L_max = 8`:

| count | widths | paper | us |
|---|---|---|---|
| 8 | `[4]x8` | legal | legal |
| **6** | `[4,4,4,4,8,8]` | **legal** | **REFUSED** -- does not divide 32 |
| 4 | `[8]x4` | legal | legal |
| **2** | `[16,16]` | **outside `L_max`** | **allowed**, with only a warning |
| **1** | `[32]` | **far outside `L_max`** | **allowed**, with only a warning |

So we are stricter than the paper at 6 and looser at 1 and 2. Worth noting the
2x-envelope warning added here lands on block width 8 -- exactly the inferred
`L_max` -- reached from `ComfyUI-UtilsCollection`'s reasoning rather than from
the paper, which is a nice convergence but means the warning is doing the
paper's job by accident rather than by construction.

> **RUN AND SUPERSEDED, 2026-08-28. The arm named here — `[4,4,4,4,8,8]`,
> knots `[0,4,8,12,16,24,32]` — was the wrong six-evaluation partition.** It is
> legal under the envelope, and it spends its extra evaluations at the FRONT
> where the trajectory is nearly flat, so it keeps the same 80% final Euler step
> as the uniform four-evaluation arm it was meant to improve on. Legality was
> necessary and not sufficient.
>
> `[8,8,4,4,4,4]` is the one that matters: same evaluation count, same width
> multiset, coarse blocks at the FRONT, and a **63.2%** final step — the same
> tail the vendor's own eight-evaluation schedule has. It now ships as
> `workflows/h3_text_to_video_pdd_manual_sigmas.json`.
>
> **And four evaluations cannot be improved at all.** Enumerated: `[8,8,8,8]` is
> the ONLY partition of the 32-point grid into four blocks that starts every
> block on a multiple of `L_min` and keeps every width within `L_max`. Its 80%
> final step is forced, not chosen, so there is nothing to tune toward — the
> operating point itself is the problem.
>
> [`research/pdd/audio_under_pdd.md`](research/pdd/audio_under_pdd.md) has the
> matched-pair render behind that, and the reason no partition experiment can
> attribute the effect to either stream: every coarseness statistic ranks the
> arms identically whether computed in video time or through the audio
> transform.

**The six-evaluation arm is reachable today** and was, before this section was
written, the most paper-grounded untried thing in the lane: the run-time path
takes an uneven partition happily (`schedule_knots` derives and reports it), so
`ManualSigmas` runs one with no code change.

**Stated as a prediction, not a result.** Fusion loss is a weight-space proxy:
it measures how far a block's heads sit from their own mean, and it does NOT
measure the integration error of freezing the hidden state across the span, nor
whether a non-standard sigma set is off-distribution for the DiT. The uniform
partition is the schedule these models are normally run at, and deviating from
it is untested here.

**It costs nothing to test.** Both are just sigma vectors, and neither needs a
code change -- `ManualSigmas` or `FloatToSigmas` into `SamplerCustomAdvanced`
does it, and the tracker derives blocks from whatever it is handed and reports
an uneven partition rather than refusing:

    uniform  sigmas [1.0, 0.972973, 0.923077, 0.8,      0.0]         80.0% tail
    optimal  sigmas [1.0, 0.631579, 0.444444, 0.27907,  0.0]         refuted
    SHIPPED  sigmas [1.0, 0.972973, 0.923077, 0.878049,
                     0.8, 0.631579, 0.0]                             63.2% tail

**A `ManualSigmas` arm is now a shipped shape, not just a possibility**, and two
checks learned it: `h3_config.graph_schedule` reads the vector and reports the
scheduler as `manual` rather than `simple` (calling it simple would assert an
equality with `calculate_sigmas` that is false for a non-uniform partition), and
`check_distill_settings` grades such an arm's knots against the grid and its
widths against the envelope instead of demanding the count divide 32.
`check_pdd_sigmas` skips them and says which check covers them instead.

Matched seed, `C2`-style with Sol removed from both sides, is the arm.

### Audio is the first thing a low step count costs, and there is a known fix

Two more packs on this box speak to the question this section opened with, and
they agree with each other without agreeing with us.

**`coderef/comfyui-minimax-h3-audio-T8/pdd_advanced.py` is a sixth PDD
implementation and the strictest of them.** Its `validate_pdd_sigmas` RAISES on
anything that is not the exact schedule: *"MiniMax H3 PDD requires exactly 8
model evaluations and a terminal sigma (9 values)"*, and separately *"requires
the official Euler/simple 8-step sigma schedule with video shift 12"*, to
`5e-6`. Its shift check is independent convergence on the run-time shift guard added
here the same day.

**But read its strictness for what it is.** T8's own module docstring gives the
reason, and it is not a measurement: *"The math follows the Apache-2.0
reference implementation published with `alibaba-pai/MiniMax-H3-Acc-LoRAs` at
revision 78db175..."*. That is a **fidelity policy** -- reproduce the reference
exactly, refuse everything else -- and it is a perfectly good goal for a pack
that wants vendor-equivalent output. It is not evidence that four evaluations
are worse.

**And the tally does not say what it first looks like it says:**

| implementation | 4 steps (block width 8) |
|---|---|
| alibaba-pai, the authors | no step knob exists; 4 is *unvalidated*, not refused |
| T8 | refuses, on a stated fidelity policy |
| `UtilsCollection` | **allows** -- its cap is `(trained, 2 x trained)`, and width 8 is exactly 2x |
| Mamad8 | allows, exposes `steps` |
| ours | allows any divisor, warns past 2x |

So three of five allow it, one refuses on policy, and the authors simply never
exposed the choice. **Nothing in that table is a rendered comparison.** Nor is
the weight-space evidence against 4: the step-size-weighted fusion loss measured
here does NOT grow from width 4 to width 8 -- the last block comes out 0.0309
against 0.0335 -- because the `dt` weighting concentrates on the late heads and
widening only adds low-weight ones.

**The honest position: 4-step PDD is unvalidated upstream, not shown worse.**
The one clean rendered comparison this repo owns -- `C2_pdd4_nosol` against
`C2_pdd8_nosol`, three seeds a side with Sol removed from both -- is still
unjudged, and it is the only thing here that can settle it.

**`coderef/ComfyUI-H3-AudioRefine` states the symptom outright and ships the
fix.** Its README: *"4-step video is acceptable but 4-step audio is not"* --
written about turbo, and the mechanism is not turbo-specific. The reason is
structural rather than incidental: on a 1344x768 124-frame clip the audio is
about **400 rows against video's ~37,000**, roughly 1% of the packed sequence,
so whatever a distillation gives up it gives up on audio first and least
visibly.

Its approach, and the part that makes it applicable here:

- Freeze the video with a per-stream `denoise_mask` (video 0, audio 1). ComfyUI
  supports that natively -- for H3 `scale_latent_inpaint` returns the latent
  unmodified and the frozen rows are fed at `VISUAL_COND_TIMESTEP`, the same
  treatment keyframe conditioning gets, so the returned video is bit-identical.
- **Take the MODEL from before the LoRA for the refinement pass.** Pass one runs
  distilled; the audio-only pass runs undistilled. Their sentence for why is the
  whole argument: *"the audio quality you were missing is exactly what the turbo
  LoRA took away."*
- Freezing alone saves almost nothing -- attention is one fused sequence, so the
  37,000 frozen rows still pay qkv, attention and MLP at all 50 blocks (measured
  20.7s against 23.0s). The pack's `H3FrozenVideoCache` is what buys the time
  back, caching each block's attention input; it hooks `set_model_patch_replace`
  and a `WrappersMP.DIFFUSION_MODEL` wrapper, both stock, no monkeypatching.
  Its stated approximation is that the video->audio attention edge is severed
  between rebuilds.

**Why this is worth a look here specifically.** Because the refinement runs on
the base model, it needs nothing from the PDD head schedule -- so it sidesteps
every constraint this document spends its length on. A 4-step PDD video pass
followed by an audio-only pass on the undistilled weights is expressible today,
costs roughly 8-10 full steps of arithmetic, and targets exactly the stream that
a 2x-wide block hurts most.

**Nothing here has run it**, and the interaction that would need checking first
is whether the PDD node's patches and the pack's block replacement compose --
they sit on different surfaces (`final_layer.forward` and
`diffusion_model.forward` against `("dit", "double_block", i)` plus a model
wrapper), which is a reason to expect they do and not evidence that they do.

**The 6 in the Sol table is a turbo count, not a PDD one** -- worth stating
because reading that table alone suggests otherwise. Every 6-step arm this repo
ships is a 768p turbo graph. **Corrected 2026-08-29: this previously said "no
PDD graph can run 6, because 6 does not divide 32 and the node refuses".** Six
*evaluations* ship — `workflows/h3_text_to_video_pdd_manual_sigmas_api.json`
runs the uneven `[8,8,4,4,4,4]` partition this document introduces below. What
the node refuses is a non-dividing `steps` REQUEST; an explicit knot list that
tiles the grid unevenly is legal and is the point of that arm.

---

## Replicating the reference

The vendor ships `predict_ref2v.py` and a scheduler, and three things about
how they consume the fused heads are worth matching rather than approximating.

**Euler, not `er_sde`.** `coderef/diffusers/src/diffusers/schedulers/scheduling_minimax_h3.py::step` says it takes "one
Euler (`eta = 0`) step", and their adapter defines the fused head as "the mean
velocity of one block, which an Euler step over the block boundaries consumes".
`er_sde` injects noise and uses a different update rule, so the heads would be
consumed by something they were never distilled against.

**Whose requirement this is, stated precisely, because the loose version gets
quoted.** The MECHANISM is the paper's: eq 4, eq 10, eq 14 and Algorithm 1
define a first-order update with no noise term, evaluated at the block start.
The PROHIBITION is ours. The paper *defines* its sampler rather than choosing
among them, so it never contemplates `er_sde` and therefore does not forbid it.
"The paper requires euler" is not quite true; "the paper's sampler is euler,
and anything else is outside what it defines" is.

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

**The conversion reproduces its CONTENT but not its FILE, measured 2026-08-28**
([`bench/results/2026-08-28_pdd_conversion_reproducibility.json`](../bench/results/2026-08-28_pdd_conversion_reproducibility.json)).
Both partitions were re-converted from the identical source, base and pruned
inputs -- the source sha256 is unchanged since the 2026-08-27 record -- and the
result is:

| | old against new |
|---|---|
| all 730 tensors | **bit-identical**, both partitions |
| every metadata VALUE | identical |
| tensor key set, key order, `data_offsets` | identical |
| the file's sha256 | **differs** |

The whole difference is the ORDER of metadata keys inside the header JSON.
`safetensors` does not preserve the order of the dict the converter hands it and
picks a different one per process, so the header is the same length, holds the
same pairs, and serialises to different bytes.

**So a converted artifact's file hash is not a stable identity, and a mismatch
does not imply the content changed.** That matters here specifically:
`2026-08-27_pdd_conversion_*.json` records the file sha256 as `ours`, and it
will not reproduce. The 2026-08-28 record carries a layout-independent content
hash instead -- sorted metadata pairs, then every tensor's bytes in sorted key
order -- which is the thing to compare when the question is "did this artifact
change".

Note what the re-run did NOT establish. The source files are byte-identical to
what the 2026-08-27 record hashed, so it says nothing about whether upstream
has re-uploaded; see the lead recorded above. Only a re-fetch settles that, and
**the converter would not notice a re-encoded bank**: it copies `proj_out` into
`h3_pdd.bank.*` verbatim with no check of whether the rows are absolute heads
or deltas from head 0. The cheap observable, if a detector is ever wanted, is
the row norms -- on the artifacts here every row sits within 1% of every other
(min/row0 = 0.9906, all ~57), where a delta encoding would put rows 1..31 far
below row 0.

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
| ~~unconsumed keys in the published file are an error~~ | **closed since the first PDD commit** — `bench/convert_pdd_lora.py` raises `SystemExit` naming the leftovers. This row was never true |
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

**One thing they do that we cannot** — **corrected 2026-08-29: this said
"two", and listed an uneven partition first. We ship that partition**, as
`h3_config.PDD_MANUAL_SIGMAS`, exactly `(8,8,4,4,4,4)`. What remains is a hard
restriction of block sizes to the
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

> **One item left this list on 2026-08-28.** The converter copied
> `proj_out.weight` into `h3_pdd.bank.*` with no encoding check, and nothing
> downstream could have caught a delta-encoded re-upload -- the partition guard
> compares the base checkpoint's head, not the bank, and
> `compare_pdd_conversions.py` grades our bank against the file it came from.
> `assert_bank_verbatim` now refuses one at conversion time, on the values
> rather than on a revision, with `bench/check_pdd_bank_encoding.py` as its
> red proof and `bench/results/2026-08-28_pdd_bank_encoding.json` as the
> baseline a re-fetch is compared against.

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
- **LoRA, adaLN and pruning appear nowhere in the paper.** Those three surfaces
  are alibaba-pai's packaging of the method, neither prescribed nor forbidden by
  it. So `--pruned` is an approximation with **no paper counterpart** rather
  than something the paper sanctions -- the residual is ~50x below bf16 so the
  exposure is nil, but the label matters when someone cites the paper as
  authority for it. Reported by a peer's read; not independently re-derived
  here beyond confirming the terms are absent.

- **A second-order or stochastic sampler is not merely unguarded -- it is
  STRUCTURALLY INVISIBLE to `BOUNDARY_TOLERANCE`, and no tightening will ever
  reach it.** Reported by a peer reading the paper against our code, then
  verified here by driving the real `_StepTracker` with a real
  `adaln_t_table` over heun's actual evaluation pattern.

  `comfy/k_diffusion/sampling.py:296` evaluates heun's corrector at
  `sigmas[i + 1]` -- which **is** the next block boundary. The boundary check
  measures distance-to-nearest-boundary, so it sees ~0, which is precisely the
  state it is built to call healthy. Driven at 4 steps, knots
  `[0, 8, 16, 24, 32]`:

      step  evaluation          sigma   block   warned
         0  euler predictor   1.00000       0    False
         0  HEUN corrector    0.97297       8    False
         1  euler predictor   0.97297       8    False
         1  HEUN corrector    0.92308      16    False

  **Every corrector selects the NEXT block's fused head while integrating the
  current step**, and `warned` never flips. Half the model evaluations use the
  wrong head and nothing says so. `er_sde` is invisible the same way.

  So the honest statement is not "the euler requirement is enforced by
  nothing" -- it is that **the guard which looks closest to covering it cannot,
  by construction.** If it is ever worth closing, the observable is the number
  of model calls per step, or the sampler's identity, and NOT the distance to a
  boundary. Do not spend time tightening the tolerance.

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

## Why audio suffers more than video, and why it worsens with block width

The owner's question was why audio is always the thing that is off in these
distilled arms, and whether it is a latent ComfyUI bug. **No bug was found. What
was found is a structural interaction between how ComfyUI samples H3's audio and
what a PDD head returns**, and it predicts the size of the effect.

### The divergence, confirmed from source

The vendor's own pipeline steps audio on **its own scheduler**:
`coderef/alibaba-pai_MiniMax-H3-Acc-LoRAs/predict_t2v.py:36` passes
`pipeline.scheduler.shift` AND `pipeline.audio_scheduler.shift` — two
schedulers, audio integrated at shift 3 in its own time.

ComfyUI does not. `comfy/model_sampling.py:328` carries the audio latent
**scaled onto the video schedule** so the pack is "an ordinary single-schedule
flow latent", and `comfy/ldm/minimax/model.py:553-577` undoes and redoes that
carry around every forward:

    carry  = sigma_a / sigma_v                       # before _forward, audio seen clean
    out[1] = (1 - s) * (audio_src * carry)           # after _forward
             + (1 + (s - 1) * sigma_a) * out[1]      # s = shift_v/shift_a = 4

That second line is a **change of variable on an instantaneous derivative**,
evaluated at this step's `sigma_a`.

### Where PDD lands in it

`MiniMaxH3PDDLoRA` patches `final_layer.{video,audio}_out.forward`, which is
inside `_forward` — so the fused head runs in the clean-audio domain and the
transform is applied to its output. That placement is correct.

**But a fused head does not return an instantaneous velocity.** It returns the
block's MEAN velocity over `[t_n, t_{n+L}]` — that is the whole idea. So an
instantaneous change of variable, evaluated at the block's starting sigma, is
applied to a quantity averaged over the block. That is exact only as the block
narrows, and **video has no such transform at all**, because video is the
reference stream the carry is defined against.

So the error is audio-only and should grow with block width. **This is an
inference from source, not a measurement** — but it makes a prediction the
existing data can score.

### The prediction, scored

`bench/results/2026-08-28_pdd_partition_fidelity_362.json`, four arms:

| arm | widest block | video rel L2 | audio rel L2 | audio/video |
|---|---|---|---|---|
| u8 | 4 | 0.542 | 0.858 | 1.58 |
| mix6 | 8 | 0.536 | 0.899 | 1.68 |
| u4 | 8 | 0.537 | 0.923 | 1.72 |
| opt4 | 28 | 0.522 | 0.992 | 1.90 |

**Video is flat across a 7x range of block width — it even improves slightly.
Audio rises monotonically, and so does the ratio.** That is the signature the
mechanism predicts, on the one variable video is insensitive to.

### What this does and does not license

It does NOT say ComfyUI is wrong. Carrying audio on one schedule is a
deliberate design that makes the pack a single-schedule latent, and it is
presumably fine for undistilled sampling where every step returns an
instantaneous velocity. The interaction is with PDD specifically.

It is also **one seed**, and audio rel L2 is raw-waveform and phase-sensitive —
a poor absolute metric. What carries the argument is the ORDERING across four
arms with video flat, which phase noise does not explain.

**What would settle it:** render one arm with `shift_audio` set equal to
`shift_video`, making `s = 1` and the transform the identity. If the audio
penalty collapses, the mechanism is confirmed. That changes what the model is
asked for, so it is a diagnostic and not a shipping option. Unrun.

**Practical consequence meanwhile:** prefer partitions whose blocks are narrow
where it matters, which is the same conclusion the tail5/tail6 enumeration
reached from the video side for a different reason.

## What the artifact costs at run time, and whether it has to

Raised by the owner on 2026-08-28: the file adds about a gigabyte at run time —
is there a way to do that more efficiently without losing quality? Measured
composition first, because the intuitive answer is wrong.

### Where the gigabyte actually is

By tensor group, on `minimax_h3_ref2va_pdd_8step_comfy.safetensors`:

| group | MiB | share |
|---|---|---|
| **backbone rank-64 LoRA A/B pairs** | **933** | **88%** |
| adaln baked delta | 92.3 | 8.6% |
| head bank, 32 replicated heads | 42 | 4.0% |
| grid / base head / alphas | ~1 | — |

**The head bank is not the cost, and saying it was is a mistake this repo made
out loud.** It is the part everyone reaches for because it is the unusual thing
about PDD, and it is 4%. **Corrected 2026-08-29: the node does not fuse at
load.** `_FusedHeads.get` fuses a span on first use and caches it — which is
what let the step count move out of the artifact, as this document says in
three other places. The sentence below described the retired design. The node
formerly fused the bank down to the requested
evaluation count at load, so keeping only the fused heads would save about
31 MiB. Worth doing as tidiness; it is not an answer to anything.

### Why the backbone is dynamic, and whether it still needs to be

The backbone installs through ComfyUI's native `add_patches`, so the pairs are
held and the patched weights materialised. The argument for keeping A/B pairs
separable is `strength`. **Across every shipped PDD node, `strength` is 1.0** —
nothing uses the scaling the dynamic form exists to provide.

There is also a precedent in this file: the adaln update WAS a runtime
injection and is now a baked weight patch, which removed 50 forward patches, a
`cdist` per forward and the per-call casts, at a measured worst 1.1e-4 relative
— about thirty times below bf16's own resolution — with the converter refusing
above 1e-3. So one of the three mechanisms has already had exactly this
treatment, and the tooling and the discipline for grading it exist.

### The blocker, which is real and is measurable

Baking the backbone is NOT the same operation as baking the adaln. The adaln
bake was a change of *representation* into a basis already present in the
pruned checkpoint. Merging the backbone means folding a bf16 residual into an
**INT8 ConvRot** base — dequantise, add, requantise — which is lossy in a way
the adaln bake was not. `coderef/comfyui-minimax-h3-audio-T8` declines this step
for that stated reason.

Lossy is a measurement, not a verdict, and two instruments for it already exist:

- `MiniMaxH3PDDLoRA`'s `strength=0.01` is documented as the way to price the
  backbone's dequantise/add/requantise cost against `0.0`, which installs
  nothing.
- `bench/grade_sage_on_capture.py`'s method — grade against an exact reference
  on captured activations — is the controlled comparison, and needs no render,
  so it does not run into the different-sample rule.

**What would settle it:** bake one partition, then grade merged against patched
on captured activations with a bf16 reference. If the residual lands where the
adaln bake did, ship a pre-merged checkpoint per partition and delete 933 MiB of
run-time patching — which is the shape this repo already ships for everything
else, since `MODELS` names four pre-converted checkpoints. If it lands near the
INT8 quantisation floor, the current design was right and now for a stated
reason.

**Unrun, and DEPRIORITISED by the owner on 2026-08-28 — do not pick this up as
the next thing.** It is written down because the analysis was done and would
otherwise be re-derived, not because it is queued. Nothing depends on it: the
current design works, the memory failure it might have been justified by turned
out to be a session artifact, and the whole section exists to say the intuitive
version of this optimisation aims at the wrong 4% anyway. If it is ever run,
run it because the run-time patching is a moving part worth deleting.

### Do not justify it with the OOM

The 2026-08-28 ref2va memory failure is recorded in
`bench/results/2026-08-28_pdd_ref2va_memory_marginality.json`, and it does not
support this work. The shortfall was **17.5 MiB**, not gigabytes; the same graph
succeeded on retry; and the 31 MiB head-bank fuse alone might have cleared it.
The case for baking the backbone is that it is *correct* — fewer moving parts,
no transition peak, strength composing natively — not that memory demands it.
Reaching for a 933 MiB redesign to solve a 17.5 MiB shortfall is the wrong size
of fix, and the record says so.

## Distilled motion looks WRONG, not necessarily greater — and this repo had no record of it

**Owner observation, 2026-08-28, and it is not new to him.** Distilled arms
render camera movement worse — "cameras get shakier" — seen "a ton", first with
the **LightX2V 4/8-step turbo LoRAs, before PDD existed**, and again in the PDD
clips of 2026-08-27 and 2026-08-28. Reported as community-known for step
distillation generally.

> **REFINED by the owner on closer viewing, and the refinement matters.** *"Both
> the base and the distills and pdds shake, but I guess it's less natural in the
> pdd one and distills."* So it is **not** that distillation invents shake where
> there was none — the base shakes too. The difference is in the CHARACTER of
> the movement, not obviously its amount. That is a weaker and more specific
> claim than the one this section originally made, and the numbers below should
> be read through it.

**Nothing in this repo recorded it before today.** Grepped: no mention of
distills and motion, shake, or temporal instability anywhere in `docs/`. That is
the notable part — a repeatedly observed, cross-implementation behaviour of the
exact technique this file is about, with no entry. It is not a PDD finding; PDD
is where we happen to be looking at it.

### The one measurement here, and what confounds it

Dialogue scene, base 16-step against PDD 4-step, same prompt and seed, shot cuts
masked, 336x192 greyscale:

| | median motion | p90 | jitter | jitter/motion |
|---|---|---|---|---|
| base 16-step | 0.0047 | 0.0099 | 0.00085 | 0.179 |
| PDD 4-step | 0.0082 | 0.0180 | 0.00118 | 0.145 |

**1.72x the motion, 1.39x the jitter**, with the jitter-to-motion RATIO falling.
**But read that against the refinement above: both arms shake, so this is very
likely the SAME prompted camera movement rendered differently rather than extra
movement invented.** A magnitude metric cannot tell those apart.

**An attempt to measure "less natural" did not work, and is recorded because the
negative is useful.** Natural handheld shake is aperiodic and broadband, so a
peakier motion spectrum should read as less natural. Two statistics on the same
signal disagree in direction:

    spectral flatness   base 0.0832   PDD 0.0646   (PDD peakier -> less natural)
    peak / mean         base 76.1     PDD 61.2     (base peakier -> other way)

**So this does not capture it.** One scene, one seed, and two measures pointing
opposite ways is not a result; picking the agreeing one would be choosing a
statistic for its answer. Whatever "less natural" is, frame-differenced motion
magnitude and its spectrum do not see it.

**The confound, and it is in the prompt.** That scene's own text says *"the
camera shakes slightly with the operator's breathing"*. Both arms share it, so
the comparison is controlled — but "the distilled arm over-expresses a PROMPTED
shake" is a weaker and different claim from "distillation invents motion". **The
market prompt asks for no camera shake and is the clean substrate for this
question. It is unmeasured for it.**

### Separate from the partition effect, and they compound

The partition axis shows the opposite sign: the coarse `[8,8,8,8]` arm has
**0.79x** the motion of `[8,8,4,4,4,4]` and 1.01x the jitter — more sluggish and
smeared, not shakier. So:

  * **distillation** adds motion (base -> distilled), independent of partition;
  * **coarse partitions** degrade under motion (+0.676 within-clip correlation).

**A 4-evaluation arm is therefore bad twice over: it generates more of the thing
it is least able to render.** Narrowing the tail addresses the second and
nothing here addresses the first.

### Why it matters beyond quality

If distillation systematically adds motion, then a distilled arm is not a faster
version of the same sample — it is a different one in a way that has a
direction. Any comparison that treats a distilled clip as "the base, cheaper" is
assuming something this observation contradicts.

**Unmeasured and worth having:** whether it reproduces on the market scene
(no prompted shake, both arms already rendered), and whether the turbo LoRAs
show the same ratio, which would make it a property of step distillation rather
than of PDD.

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

- [`research/pdd/audio_under_pdd.md`](research/pdd/audio_under_pdd.md) — **why a
  4-evaluation arm renders jagged video with scratchy audio, and why 4 cannot be
  fixed.** Carries the matched-pair render, the enumeration showing `[8,8,8,8]`
  is the only legal 4-block partition, the energy-collapse measurement, and the
  reason no partition experiment can attribute the effect to one stream. Its
  central MECHANISM claim is superseded; its causal result is not, and the file
  says which is which at the top.
- [`research/pdd/2026-08-28_audio_plan.md`](research/pdd/2026-08-28_audio_plan.md)
  — the execution plan for that lane, written for a session with no context.

- [`docs/h3_ref2v_distillation.md`](h3_ref2v_distillation.md) — why ref2v
  resists step distillation, and the measurements this table is scaled against
- [`docs/checks.md`](checks.md) — the check index and the uncontrolled-requirement audit
- [`workflows/h3_config.py`](../workflows/h3_config.py) — `PDD_FL2VA_LORA`,
  `PDD_REF2VA_LORA`, `PDD_STEPS`, `PDD_SHIFT`
- [`docs/research/pdd/README.md`](research/pdd/README.md) — this page's
  mechanisms drawn against Kijai's, through the four shipped PDD graphs. A
  teaching surface that restates numbers **this** document owns, generated by
  nothing and read by no check, so it goes stale silently
