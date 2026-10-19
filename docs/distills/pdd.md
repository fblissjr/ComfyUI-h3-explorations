# Parallel Decoding Distillation on MiniMax H3

PDD is alibaba-pai's acceleration LoRA for H3 (`MiniMax-H3-FL2VA-Acc-8Step`, `MiniMax-H3-Ref2VA-Acc-8Step`; paper
arXiv 2607.26004, saved as
[`arxiv_2607.26004v1`](../research/pdd/arxiv_2607.26004v1_Parallel_Decoding_Distillationfor_Fast_Image_and_Video_Generation.md)).
This page is the overview and the contract: what it is, what ships on which checkpoint, the rules a PDD graph has to
keep, the traps that render normally, and what is measured. The node and converter internals, and how they were
checked against the paper and the other implementations, are [`pdd_node.md`](pdd_node.md). Which file is which, with
its metadata, is [`pdd_artifacts.md`](pdd_artifacts.md). Which distill suits which shot is
[`which_distill.md`](which_distill.md). The dated studies behind this page are under
[`../research/pdd/`](../research/pdd/README.md), whose README orders the handoffs.

The vendor's inference adapter, `minimax_h3_pdd.py`, ships beside the weights
(`coderef/alibaba-pai_MiniMax-H3-Acc-LoRAs/`) and is the authority for the mechanism; what this page says of their
design is read from it.

## What it is

Not "the same trajectory in fewer steps". The trajectory stays a 32-point grid. What changes is the final output
head: `proj_out` and `audio_proj_out` are replicated once per interval of that grid, and one sampling step fuses a
contiguous block of those heads into one linear whose output is the block's `dt`-weighted mean velocity. Eight
evaluations at block width 4 cover the 32 intervals; four at width 8 do too, from the same weights.

The fusion is on weights, not outputs, and its plan is a function of `(shift, num_steps, start, stop)` alone, so a
fused head can be computed once per block and reused (`pdd_math.fuse_block` is exact, not an approximation). It is
not a function of the step count: a block is a span between two grid points, and which points a render visits is
named by the sampler's schedule. So the converted file carries the 32-head bank verbatim and the node fuses each
span the first time a step asks for it.

One published file carries three mechanisms, on three surfaces:

| | what | surface here |
|---|---|---|
| backbone | attn + MLP LoRA on the 50 blocks and the 2 token-refiner blocks | applied at the call (`backbone_apply`, below) |
| adaln | `adaln_proj.linear` LoRA, per block | weight patch: in the 2688-dim time space on an unpruned base, pre-solved into the curve basis on a pruned one |
| heads | the per-interval output heads | per-step swap of two `final_layer` linears |

**The block boundaries are the plain shifted schedule, bit for bit.** `linspace(1, 0, 33)[::4]` is
`linspace(1, 0, 9)`, and the shift is pointwise, so the 8-step boundaries are exactly the 8-step schedule (verified
to `torch.equal` at shift 12 video and shift 3 audio). The release's scheduler configs declare 12.0 video and 3.0
audio, which is the base checkpoint's own shift. A PDD arm therefore moves the step count and nothing else.

## What ships, and on which checkpoint

| constant (`workflows/h3_config.py`) | value |
|---|---|
| `PDD_FL2VA_LORA`, `PDD_REF2VA_LORA` | the adaln-baked `_comfy` sidecars, for the pruned int8_convrot bases |
| `MODELS["unet_fl2va_pdd8_baked"]` with `PDD_FL2VA_STRIPPED_LORA` | the fl2va backbone bake and the stripped sidecar cut against it, a pair the node enforces |
| `PDD_STEPS`, `PDD_STEPS_FAST` | 8 and 4 |
| `PDD_MANUAL_SIGMAS`, `PDD_MANUAL_EVALS` | the six-evaluation `[8,8,4,4,4,4]` partition (below) |
| `PDD_STRENGTH`, `PDD_SHIFT` | 1.0; 12/3 |

Which graph loads which pair is a column of [`workflows/INDEX.md`](../../workflows/INDEX.md), read off the graphs:
as of this writing the masked-lane and whole-song graphs load the baked checkpoint with the stripped sidecar, and
the other PDD graphs (the daily PDD8-then-FlashGen finishes among them, `h3_config.DAILY_GRAPHS`) load the pruned
base with the `_comfy` sidecar. **The published files load nowhere in ComfyUI as shipped**: their keys are
diffusers-side with bare `lora_down` / `lora_up` suffixes, every tensor is skipped with a log line, and the render
comes out as an undistilled pass that looks like a bad LoRA. They are converted by `bench/convert_pdd_lora.py` and
loaded by `MiniMaxH3PDDLoRA` (`pdd_lora.py`); `LoraLoaderModelOnly` cannot carry two of the three mechanisms.

**Which sidecar fits which checkpoint**, and what the node does on each wrong pairing, is
[`pdd_artifacts.md`](pdd_artifacts.md) section 2. In one line: the observable is
`blocks.0.adaln_proj.linear.weight`'s shape (`[96768, 8]` pruned, `[96768, 2688]` unpruned); the baked `_comfy`
file fits only the pruned base it was solved against, `_adaln2688` fits either, and the node branches on the loaded
model (`use_adaln_curves`, the live table, the output head, the probe), never on a filename. Do not answer the
question from `h3_pdd_base` metadata: it records the converter's `--base`, which for a baked file is the unpruned
checkpoint it refuses.

## The contract

### Sampler: Euler, and only Euler

A fused head is defined as the block's mean velocity, which a single Euler step across the block integrates exactly;
the paper's Algorithm 1 is a first-order, noise-free update evaluated at the block start. The mechanism is the
paper's; the prohibition is ours, since the paper defines its sampler rather than choosing among them. Anything that
re-noises between boundaries (`er_sde`) or evaluates mid-block (heun's corrector) consumes the heads with an update
they were never distilled against. `h3_config.DISTILL_SAMPLING` (euler, simple) applies to every distilled arm. **No
guard can catch a wrong sampler** (see "Traps").

### Schedule: the node emits it

`MiniMaxH3PDDLoRA` has a `SIGMAS` output, `1 - pdd_time_grid` for the step count, and the shipped graphs wire it
into `SamplerCustomAdvanced` with no `BasicScheduler`, so the sampler steps at the boundaries the heads were fused
for. Against ComfyUI's `calculate_sigmas` over `ModelSamplingAV`, the output is bit-identical to `simple` at 2, 4
and 8 steps and about 2e-3 off at 16, where `simple` quantises against its 1,000-entry table and the closed form is
the more correct of the two. `bench/check_pdd_sigmas.py` grades this, including that each graph consumes the output.

The node's `steps` input sets the count. Legal counts, generated from `pdd_lora.resolve_emit_steps` and
`envelope_partition` (drive the functions rather than trusting this table):

| steps | partition | vs trained width 4 | the node |
|---|---|---|---|
| 1 | `[32]` | past 2x | emits |
| 2 | `[16,16]` | past 2x | emits |
| 3 | -- | -- | raises |
| 4 | `[8,8,8,8]` | 2x, the edge | emits |
| 5 | `[8,8,8,4,4]` | 2x, the edge | emits |
| 6 | `[8,8,4,4,4,4]` | 2x, the edge | emits |
| 7 | `[8,4,4,4,4,4,4]` | 2x, the edge | emits |
| 8 | `[4]*8` | trained | emits |
| 9, 10 | -- | -- | raises |

A count that divides 32 tiles it uniformly; 5, 6 and 7 tile it unevenly inside the trained envelope. Emitted sigmas
at shift 12:

    1 step   1.0000, 0.0000
    2 steps  1.0000, 0.9231, 0.0000
    4 steps  1.0000, 0.9730, 0.9231, 0.8000, 0.0000
    8 steps  1.0000, 0.9882, 0.9730, 0.9524, 0.9231, 0.8780, 0.8000, 0.6316, 0.0000

- `steps` 0 means the file's own count and never refuses; it is for a graph where `SIGMAS` is unwired and something
  else (`ManualSigmas`, `BasicScheduler`) supplies the schedule. The node also observes
  `transformer_options["sample_sigmas"]` on every forward and derives the block knots from whatever schedule it is
  handed, reporting an uneven partition rather than refusing it, so those graphs still select the right heads.
- `nfe` is an override that forces uniform blocks and ignores the schedule, for decoding one partition while
  stepping another. Every shipped graph carries 0.
- For `denoise < 1.0`, send `SIGMAS` through `SplitSigmasDenoise` and take `low_sigmas`: `SIGMAS(8)` split at 0.5
  equals `BasicScheduler(simple, 4, denoise=0.5)` (`torch.equal`, and pixel-identical at a matched seed), and every
  entry is a block boundary by construction.

**Which partition is legal, per the paper.** Training samples blocks that start at multiples of `L_min` and are no
wider than `L_max` (paper section 3). The H3 files declare `pdd_num_steps 32` and `pdd_block_size 4`, consistent
with `L_min = 4, L_max = 8`, the `{4, 8}` NFE set the paper chose for LTX-2.3, its joint video and audio model. So 4
evaluations is very likely a trained configuration; `L_max` is in no released metadata and the vendor script exposes
no step knob, so it is not confirmable. Under that rule `[8,8,8,8]` is the only legal four-block partition, and its
80% final Euler step (at shift 12) is forced. The six-evaluation `[8,8,4,4,4,4]` puts the coarse blocks first and
keeps a 63.2% final step, the same tail the vendor's 8-step schedule has; it ships as
`workflows/h3_text_to_video_pdd_manual_sigmas_api.json` through `ManualSigmas`. Counts 1 and 2 exceed `L_max` and the
node only warns.

### Shift: the model's own

`MiniMaxH3SigmaShift` is absent from PDD graphs at the base shift (the generator condition is
`shift == SIGMA_SHIFT`, so a PDD arm fused at another shift gets the node back). The fused heads are a function of
the shift, so `_StepTracker.check_shift` raises at the first forward, naming both numbers, when a graph's shift
differs from the file's; it is per-forward and unlatched. The checks read the default shift from ComfyUI's model
config (`bench/check_pdd_sigmas.py::_checkpoint_default_shift`, from
`comfy.supported_models.MiniMaxH3.sampling_settings`), not from a repo literal.

### Strength and the surfaces

- `strength` scales all three surfaces; 1.0 is the vendor default. 0.0 installs nothing at all, so it is exactly
  the base model. Below 1.0 the node interpolates each head toward the checkpoint's own head (Kijai's conversion
  scales from zero instead; [`pdd_node.md`](pdd_node.md)).
- `head_strength` (-1 follows `strength`) scales the heads alone; 0.0 there, or `patch_heads` off, runs the
  checkpoint's own heads with the backbone and adaln updates, which is the head-free control arm.
- `backbone_apply` defaults to `exact branch` (owner, 2026-09-26): the backbone and refiner LoRA is applied at the
  call through `lora_branch.py`, `y = W x + B (A x)`, on every module including `mlp.fc2`, and the int8 weight is
  left as shipped. `merge` is the control arm, where ComfyUI dequantises, adds and requantises each int8 weight. The
  policy for every LoRA on an int8 checkpoint is [`../attention/quant_policy.md`](../attention/quant_policy.md).
- `unmerged_blocks` (with `unmerged_strength` and `unmerged_window`) lifts named blocks out of the merge; it works
  only under `merge`.
- With a stripped sidecar the node refuses any `strength` other than the bake's, and refuses `unmerged_blocks`.

### Attention

PDD graphs take `h3_config.SOL_PDD_CUDA`, which is the base recipe with no overrides (`SOL_PDD_OVERRIDES` is empty),
resolved by `h3_config.sol_for_graph`; Sol's settings do not move with the step count. What Sol costs on a PDD arm
is [`../attention/sol_attn.md`](../attention/sol_attn.md). **Dense DiT attention is the reference configuration**
(the vendor runs Diffusers on stock SDPA), and dense costs about 2.4x on this workload, 70.3 s/it against 28.7 with
sage and Sol at about 90k packed tokens, so a dense 8-step PDD render is slower than a sage-and-Sol 16-step base
render. The step count was never the expensive part at this length.

## Traps that render normally

Each of these produces a plausible clip, which is why each has a guard or a warning here.

| trap | what happens | what catches it |
|---|---|---|
| a published (unconverted) file | every tensor skipped with a log line; an undistilled render | nothing at load; convert first |
| fl2va file on a ref2va base, or the reverse | the partitions share every tensor name, so it loads with zero unmatched keys (`ref2v_distillation.md`) | `h3_pdd.base_video_out` compared by relative Frobenius distance against the live head, `PARTITION_TOLERANCE`; the partitions sit about 0.05 apart. Shape is tested first, so an enlarged head left by another implementation names itself |
| a second implementation patching the same output heads | `add_object_patch` is last-writer-wins, so one implementation's bookkeeping drives the other's heads | `head_patch_clash` refuses when a head key is taken (adopted from `silveroxides/ComfyUI-UtilsCollection`); an unrelated block-attention patch does not trip it |
| a baked `_comfy` adaln file on an unpruned base | its 50 adaln modules install nowhere | the `declared_adaln` guard refuses and names the `_adaln2688` file |
| an adaln bake solved against another partition's table | fits its basis just as well and writes without complaint; 0.0205 wrong at run time against 0.0001 | the stored `adaln_table` against the live one, `TABLE_TOLERANCE`, which sits above a dtype cast (0.00164) and below the partition gap (0.01835) |
| a full sidecar on a baked checkpoint | the backbone goes on twice, at effectively double strength with correct heads | the backbone probe (below) |
| a stripped sidecar on the base, or on another bake | the backbone goes on never, or wrongly | the backbone probe |
| a partial patch-key match after a layout change | the rest applies and the render is plausible | the node compares what `add_patches` matched against what `load_lora` resolved and raises on a shortfall |
| a second-order or stochastic sampler | heun's corrector is evaluated at `sigmas[i + 1]`, which is the next block boundary, so every corrector selects the next block's head while integrating the current step; `er_sde` is invisible the same way | nothing. `BOUNDARY_TOLERANCE` measures distance to the nearest boundary, so it reads these as healthy by construction. If this is ever closed, the observable is model calls per step or the sampler's identity; do not tighten the tolerance |
| per-render state on the step tracker | ComfyUI's execution cache keeps the ModelPatcher across prompts, so a latch set by one render silences the next | every per-render attribute is reset by `_adopt`; `bench/check_pdd_head_selection.py` ("no state outlives its schedule") asserts the rule statically |

**Comparing two renders by file hash is a trap of its own.** Clips rendered before 2026-09-23 carry the API prompt
in `format.tags.comment` (so any two arms differ) and a wall-clock `creation_time` (so any two runs differ); PNGs
from `SaveImage` embed the prompt too. Identical containers imply identical frames, never the reverse. Hash decoded
frames (`bench/verify_vsa_render.py`, or `ffmpeg -f rawvideo | md5sum`). And the first render or two after a restart
or a state change can differ from the value the same configuration settles on, so a single matched pair is not an
instrument; interleave repeats.

## What a backbone bake pins

A bake folds the backbone LoRA into the checkpoint offline, once, from the bf16 release
(`bench/bake_pdd_checkpoint.py`; plan and state
[`../research/pdd/2026-09-05_bake_plan.md`](../research/pdd/2026-09-05_bake_plan.md)). Traced in source on
2026-08-31:

- **Only 200 modules requantise, all backbone linears.** Every int8 `.weight` in the pruned int8_convrot checkpoint
  is a `blocks.*` linear (50 blocks by 4 kinds); `token_refiner.*` is bf16 and `adaln_proj.linear` is f16, so the 8
  refiner modules and the 50 adaln updates are ordinary float patches. Read it off the checkpoint (`safe_open`, dtype
  per key).
- **The partition never reaches the backbone.** `resolve_emit_steps` and `envelope_partition` feed the `SIGMAS`
  output, `_StepTracker` and `_FusedHeads` only; the backbone dict is built from the file's `diffusion_model.*` keys
  before any of them exists. So **a bake pins `strength` and nothing else**, and one baked artifact serves every step
  count. After a bake the node still owns the refiner, the adaln patches, the heads, the `SIGMAS` emission and the
  tracker.
- **The pair is a stripped sidecar plus its one bake.** `bench/convert_pdd_lora.py --omit-backbone --baked <ckpt>`
  drops every backbone tensor and keeps the `h3_pdd.*` payload and the refiner keys; the sidecar declares the
  strength its bake used (`h3_pdd_backbone_strength_baked`). Detecting a bake in the node instead was rejected
  because it fails open if the marker is ever absent.
- **Pairing is by identity.** Every converter-version-3 file carries `h3_pdd.backbone_probe` and
  `h3_pdd.backbone_probe_scale`: leading rows of `blocks.49.mlp.fc2.weight`'s int8 codes and fp32 row scales, cut
  from the one checkpoint the file pairs with (the base for a full sidecar, the bake for a stripped one; `fc2` of
  block 49 carries the largest update in the file). The node requires the loaded module to equal both before
  patching (`probe_match`, `check_backbone_identity`), asserts after `load_lora` that no backbone target resolved
  and the refiner did (`check_stripped_targets`), and asserts populations against the release's declared depth
  (`vendor_config.transformer_depth()`) as sets of `(block, kind)`. Count keys off the file, not the metadata:
  `backbone_modules: 208` includes the 8 refiner modules, and under `diffusion_model.blocks.*` the file carries 200
  modules as 600 tensors. `bench/check_pdd_sidecar_contract.py` grades every refusal, on synthetic inputs and on the
  real artifacts.

The fl2va bake at strength 1 and its stripped sidecar exist (the inventory in `pdd_artifacts.md`); the ref2va bake
does not.

## What is measured

All stored-weight or offline unless it says rendered. A rendered clip cannot A/B a numerical change, so nothing
below says a difference is visible.

**The weight deltas** (2026-08-26, the published files against the release weights):

| quantity | value |
|---|---|
| backbone perturbation, `\|\|BA\|\|/\|\|W\|\|` per module type | 0.004 to 0.015 |
| fused head against the checkpoint's own head | 0.005 early, 0.015 at the last step |
| step-to-step change in the fused head | 0.004, rising to 0.019 at the final transition |
| mean of the 32 heads against the base head | 0.004 video, 0.002 audio |
| fl2va against ref2va, same PDD file position | no tensor identical, 728 of 728 |

For scale, `ref2v_distillation.md` measured the 8-step turbo LoRA at 0.00036 and the fl2va-to-ref2va checkpoint gap
at 0.042, on dequantised pruned fp8 files, so those compare in order of magnitude and not in digits. PDD perturbs the
backbone far harder than the turbo LoRAs and moves the modulation path they leave alone. Dropping the heads is a small
error concentrated in the last two steps, which is what the head-free arm controls for.

**Where the file's run-time footprint is**, on `minimax_h3_ref2va_pdd_8step_comfy.safetensors`:

| group | MiB | share |
|---|---|---|
| backbone rank-64 LoRA A/B pairs | 933 | 88% |
| adaln baked delta | 92.3 | 8.6% |
| head bank, 32 replicated heads | 42 | 4.0% |
| grid / base head / alphas | ~1 | -- |

The head bank is not the cost. A 2026-08-28 ref2va memory failure was 17.5 MiB short and succeeded on retry
(`data/results/2026-08-28_pdd_ref2va_memory_marginality.json`), so it argues for nothing this size.

**The adaln bake** lands the update in the pruned checkpoint's curve basis by an affine fit (basis plus a constant
column, because the pruned form is an SVD of the centred time curve): worst 1.1e-4 relative per block over the
1025-row grid, about thirty times below bf16's own resolution, and the converter refuses a block above 1e-3.

**Folding a delta into an already-quantised weight is the lossy step**, measured over the backbone against the bf16
release (`bench/measure_pdd_quant_interaction.py`, `bench/results/2026-08-30_pdd_quant_interaction.json`), as
stored-weight relative error against `W_ref + s*BA`:

| strength | 0.0 | 0.5 | 1.0 |
|---|---|---|---|
| merged into the shipped int8 at run time (`backbone_apply: merge`) | 0.00942 | 0.00990 | 0.01058 |
| baked once from the release | 0.00942 | 0.00942 | 0.00942 |

The merge quantises twice (the shipped int8 is already `Q(W_ref)`, and `scale="recalculate"` moves the grid), so its
rise tracks the module's own `||BA||/||W||` (correlation 0.78 over 28 modules, 0.751 over all 200). Over all 200
modules ([`data/results/2026-08-31_pdd_quant_interaction_all_blocks.json`](../../data/results/2026-08-31_pdd_quant_interaction_all_blocks.json),
aggregated by `bench/analyze_pdd_unmerge_curve.py` into
[`data/results/2026-08-31_pdd_unmerge_recovery.json`](../../data/results/2026-08-31_pdd_unmerge_recovery.json)) the
mean is 1.1164x the merge-free error and there is no hotspot: block inflation spans 1.131x, int8-vs-bf16 error spans
1.086x across blocks (`data/results/2026-08-30_pdd_block_magnitude.json`), and what spread exists is by module kind.
On the three kinds `unmerged_blocks` can reach, under the shipped seeded stochastic merge, worst-first recovers
12.6% of the reachable gap with 5 blocks and 59.9% with 25, and all 50 recover 76.2% (`mlp.fc2` stays merged); the
first ten are `7, 20, 15, 11, 8, 10, 9, 6, 14, 23`. So a subset is a compute optimisation, not a fidelity one.
Un-merging is not a memory trade: both arms of the 2026-08-30 smoke staged 19995 MB, because `add_patches` holds
the pairs either way; the cost is about +2.4% FLOPs on an un-merged block's linears, from the shapes and not timed.
`attn.qkv_proj` has the highest inflation (1.164x) and the lowest error before PDD (0.00883); `attn.out_proj` the
highest error before PDD (0.01041) and nearly the lowest inflation. Turning PDD down where quantisation hurts is
the same instruction as turning it down where the distillation works hardest, and anti-aligned with block
propagation (`data/results/2026-08-29_block_propagation.json`). `bench/check_pdd_unmerged.py` grades the un-merged
identity, including a stub with `alpha != rank`, since PDD's `alpha/rank` is exactly 1.0 and a scale-dropping
implementation would be bit-identical to a correct one on every shipped artifact. The `exact branch` default now
keeps every backbone delta off the int8 grid; the activation rounding is uncounted by any of this
(`docs/state/open_experiments.md` #23).

**The step count, rendered and offline.**

- PDD quality follows the schedule's tail, not the evaluation count (`docs/method/evidence.md`, "Settled about H3").
  On a matched pair the uniform 4-evaluation arm rendered jagged lines and scratchy audio where the 6-evaluation
  `[8,8,4,4,4,4]` arm was acceptable in both, by the owner's look
  ([`docs/research/pdd/2026-08-28_audio_under_pdd.md`](../research/pdd/2026-08-28_audio_under_pdd.md), which also says why no partition
  experiment can attribute the effect to one stream).
- Graded against the 32-step trajectory (`bench/grade_pdd_partitions.py`) at 362 frames with Sol active, every
  partition sits 0.52 to 0.54 relative L2 from the reference with the same-arm floor exactly zero: the metric does
  not separate partitions at production settings
  (`data/results/2026-08-28_pdd_partition_fidelity_362.json`). The 39-frame run that seemed to rank them
  (`data/results/2026-08-28_pdd_partition_fidelity.json`) had Sol inert, and its ordering inverted at 362; quote
  neither ordering. A fusion-loss-optimal partition (`[28,2,1,1]`) breaks the training rule above on most of its
  blocks, which is why minimising weight-space fusion loss was the wrong objective.
- The audio loss at coarse partitions is PDD's own. Core's single-schedule audio carry reproduces the vendor's
  two-schedule audio Euler step exactly at every block width
  ([`../research/pdd/2026-09-25_upstream_pdd_comparison.md`](../research/pdd/2026-09-25_upstream_pdd_comparison.md),
  finding 1), which refutes the change-of-variable mechanism once derived here. The vendor steps audio on its own
  scheduler (`coderef/alibaba-pai_MiniMax-H3-Acc-LoRAs/predict_t2v.py:36`). An audio-only refine pass on the
  undistilled weights is the known fix shape (`coderef/ComfyUI-H3-AudioRefine`; this repo's version is the
  audio-refine constants in `h3_config.py` and `docs/research/pdd/2026-08-28_audio_under_pdd.md`).
- Artifact severity under PDD tracks inter-frame delta (+0.676 within clip; `bench/measure_clip_delta.py`). One
  dialogue scene, base 16-step against PDD 4-step, measured 1.72x the motion and 1.39x the jitter, but both arms
  shake (the prompt asks for it) and the owner's reading is that the distilled movement is less natural rather than
  larger; two spectral statistics disagreed in direction, so nothing here captures "less natural". The coarse
  `[8,8,8,8]` arm had 0.79x the motion of `[8,8,4,4,4,4]`. The market scene, which asks for no shake, is the clean
  substrate and is unmeasured for it.
- The 2026-09-05 bake pair (merged against baked, five scenes, one seed) read the same or within one sample's
  variation; the PDD8 defects attribute to the eight-step schedule, not to the requantised merge, at that seed
  (`docs/method/evidence.md` carries the bound).

## Not enforced, and not measured

- **The `strength` semantics are asserted nowhere** (0.0 installs nothing; a check would need a loaded model).
- **The runtime adaln injection cannot run on a `_comfy` file.** `--pruned` drops the raw `h3_pdd.adaln.*` and
  `h3_pdd.silu_temb_grid`, so the table-mismatch path's "falling back to the runtime adaln injection" would raise
  `KeyError` on the grid. Unreachable today because the partition guard refuses a cross-partition file first;
  relaxing that guard exposes the `KeyError`, not a slow path.
- **The off-schedule warning has fired only on synthetic drives** (`bench/check_pdd_head_selection.py`); its
  threshold is reasoned. The trained-envelope warning has fired in a real render (at 2 steps).
- **Nothing grades whether a legal count is sensible.** The selector check asserts 16, 8, 4 and 2 evaluations each
  select their own blocks, which is a statement about the selector, not an endorsement of 2.
- **LoRA, adaln and pruning appear nowhere in the paper.** They are alibaba-pai's packaging; `--pruned` has no paper
  counterpart, though its residual is far below bf16.
- **Whether the fused heads change the output visibly** needs the blind multi-seed process
  (`docs/method/eval_comparison.md`). The `C2_pdd4_nosol` / `C2_pdd8_nosol` pair (three seeds a side, Sol removed
  from both) is the one clean rendered comparison of 4 against 8, and it is unjudged
  ([`docs/research/pdd/2026-08-27_queued_arms.md`](../research/pdd/2026-08-27_queued_arms.md)).
- **Whether PDD transfers the ref2v difficulty** of `ref2v_distillation.md`: the Ref2VA file is distilled on
  `transformer_ref` itself, so that page's Fact B does not bite it, and its Fact C now has a second counterexample.
- **Whether this node should keep the head half** now that core's PDD support (Comfy-Org/ComfyUI#15908, merged
  2026-08-29 as `2504e68d`) fuses heads in `FinalLayer.forward`. Our files leave `video_out.weight` its original
  size, so core takes its `n == 1` path and our patches own the swap: two implementations of one mechanism in one
  process. Undecided; the evidence is
  [`../research/pdd/2026-09-25_upstream_pdd_comparison.md`](../research/pdd/2026-09-25_upstream_pdd_comparison.md).

## See also

- [`pdd_node.md`](pdd_node.md): how the node selects heads, its patches, the converter's transforms, and the checks
  against the paper, Kijai's conversion, core and the other implementations.
- [`pdd_artifacts.md`](pdd_artifacts.md): every PDD file, its sidecar tensors and the converter and artifact
  changelogs.
- [`../research/pdd/README.md`](../research/pdd/README.md): the handoffs, the audio lane, depth and the axes, and the
  diagrams; [`docs/research/pdd/2026-08-28_pdd_implementations.md`](../research/pdd/2026-08-28_pdd_implementations.md) compares the
  implementations.
- [`ref2v_distillation.md`](ref2v_distillation.md): why ref2v resists step distillation, and the measurements the
  delta table above is scaled against.
- [`../method/checks.md`](../method/checks.md): the check index.
