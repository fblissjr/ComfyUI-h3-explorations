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
`models/diffusion_models/`.

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
| adaln | `adaln_proj.linear` LoRA, per block | weight patch when unpruned; runtime injection when pruned |
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
namespace, the fused heads, a partition-matched time grid, and a fingerprint.

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

Selection is by **interval membership**, not nearest boundary, so a run at a
step count or shift the file was not fused for degrades to the closest
available head rather than an arbitrary one. `boundary_residual` then says so
in the log, once — on the schedule the heads were fused for it is ~1e-9, and
the tightest boundary gap at 32/4 shift 12 is 0.0118, so the tolerance sits
well inside "wrong schedule" and well outside float noise.

**This is the only reason the node patches `final_layer.forward` at all.** That
patch is pure bookkeeping and delegates to the stock forward; the actual head
swap is two separate patches on the output linears. So the modulation maths
stays upstream's and cannot silently diverge from it.

---

## Two traps that are silent in both directions

### The partitions have identical key sets

`docs/h3_ref2v_distillation.md` records it: fl2va and ref2va share every tensor
name, so a Ref2VA LoRA loads onto an fl2va checkpoint with zero unmatched keys
and renders. Nothing errors and nothing logs.

The converter records a sha256 of `final_layer.video_out.weight` from the
checkpoint it was converted against, and the node refuses a mismatch. That
tensor is fp32-unquantised and **bit-identical across pruned/unpruned and
across `int8_convrot`/`fp8_scaled`**, verified 2026-08-26 over the six H3
checkpoints in `models/diffusion_models/`, so one value names the partition for
every variant we ship. The two partitions' values differ.

That is a branch on an observable, not on a filename — the rule CLAUDE.md
adopted 2026-08-22 after the tokenizer-constant escape.

### The pruned base has nowhere to put the adaln delta

Our default checkpoints are pruned, and a pruned checkpoint replaces the
2688-dim time embedding with an 8-column curve basis plus `adaln_t_table`. The
adaln LoRA's input space does not exist there, so `load_lora` would drop all 50
modules with a log line and apply the other 208.

The node re-injects them at run time from a grid of `silu(t_emb)`, recovering
the row per timestep from the model's own table. **The grid is
partition-specific**: the fl2va and ref2va time curves differ by 7.8% relative
(measured 2026-08-26 against both release partitions), so ours is derived at
conversion time from the same checkpoint that supplies the fingerprint, rather
than bundled once and reused.

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

- **No graph ships a PDD arm yet**, so `bench/check_distill_settings.py` has no
  PDD row. It fails unrecognised LoRA filenames by design, so the row has to
  land *before* the first graph does, not after.
- **The `strength` semantics are asserted nowhere.** The node interpolates all
  three mechanisms together and 0.0 returns the base model exactly, including
  the heads. Nothing checks that claim.
- **Nothing verifies the converted file against the model it will be applied
  to** beyond the partition fingerprint. A checkpoint layout change would be
  caught by `load_lora` matching nothing, which the node raises on, but a
  *partial* match would not be.
- **The boundary-residual warning has never fired in a real render**, because
  no real render has happened. Its threshold is reasoned, not calibrated.

---

## Not measured

- Whether any of this renders. Nothing here has been through a sampler.
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
