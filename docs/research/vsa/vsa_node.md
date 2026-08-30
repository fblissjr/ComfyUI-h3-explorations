# Running VSA on H3: what the node does and what still blocks it

last updated: 2026-08-30

**This file owns `MiniMaxH3VSAAttention` -- what it does, what it refuses, and
what is verified about it.** It does not own the checkpoint
([`fastvideo_vsa_checkpoint.md`](fastvideo_vsa_checkpoint.md)) or the kernel
([`../../SOLATTN.md`](../../SOLATTN.md)), and asserts nothing against either.

**Nothing here has rendered.** Every claim below is either static, or executed
against a stub, or read from a source. The one thing that would settle whether
this works is a render, and it cannot happen on this install yet -- see the
blocker.

## VSA is a fourth regime, not a knob

The three arguments the merged Sol kernel gained read like three knobs. They
are one feature. Upstream's own tests group them under "VSA-style pieces:
padded tiles, no tail, gated coarse branch", and the T8 pack's call site uses
all three together with `topk_ratio`.

| | approximates | needs training | reachable on H3 |
|---|---|---|---|
| sage | the arithmetic: quantised dense attention | no | shipped |
| Sol-Attn | the algorithm: route a subset exact, pooled term for the rest | no | shipped |
| SLA | the algorithm: route, no pooled term | yes, and the Turbo-SLA LoRA exists | `MiniMaxH3SolAttn` with `pooled_tail` off |
| VSA | route, no pooled term, plus a gated coarse branch | yes, the gate is a learned projection | this node, once the blocker clears |
| PDD | the sampler: fewer evaluations | yes, the Acc LoRAs | shipped |

**VSA competes with PDD rather than complementing it.** The published
checkpoint's filename says `4step`, so it is a distillation that cuts
evaluations, which is what a PDD arm does by other means. Note that nothing in
the artifact supports the 4-step claim -- it carries no metadata at all -- so
that is a reading of a filename, not a property anyone here has verified.

## Why it cannot be a widget on the Sol node

`MiniMaxH3SolAttn` installs an `optimized_attention_override`. That hook is
handed Q, K and V **already built**. VSA needs two things upstream of that:

- the gate is `to_gate_compress(x)`, a projection of the BLOCK INPUT, taken
  before `qkv_proj`;
- the cube tiling reorders and pads the sequence, so the projection itself has
  to run on the padded rows.

So the block forward is replaced, through `patches_replace["dit"]`, on the 50
main blocks. The 2 token-refiner blocks carry no gate and are left alone -- an
upstream sage node still handles those, which is why this node warns about an
existing attention override rather than refusing one.

## The geometry, and what it costs

Video tokens are grouped into 4x4x4 cubes, one cube per 64-row kernel block.
`4*4*4 = 64` is not a coincidence and not a tunable: it is why `block_len`
exists, because a cube at the edge of the grid holds fewer than 64 real rows
and the kernel must be told which rows are live or it folds zeros into the
block means.

Prefix segments -- everything before video -- are chunked 64 rows at a time in
their existing order and declared as both `sink_blocks` and `sink_q`, so the
conditioning stays exact on both sides.

**This node accepts any prefix; the T8 pack accepts only plain text/audio/video
and runs dense otherwise.** What the geometry actually requires is that VIDEO
IS LAST, which core guarantees ("target audio then target video, always the
last two segments") and which this node asserts rather than trusts. So
reference graphs are in scope here and are not there.

The padding is a real cost and is measured, not estimated:
`bench/check_vsa_geometry.py` reports it per shape. At a shipped canvas the
cube walk stages about 3% more rows than the sequence has. The kernel skips
them as keys but still stages them.

## What is verified, and how

`bench/check_vsa_geometry.py`, no CUDA and no model needed:

- the reorder is a bijection, every row lands inside its block's live rows, the
  prefix occupies exactly the sink blocks, and scatter-then-gather is the
  identity -- over five shapes chosen to be ragged in every axis at once,
  because a cube walk is the kind of code that is correct whenever the grid
  divides by four;
- **cube membership** is re-derived from the source index rather than from the
  walk that built it, so it is a second derivation and not a restatement;
- a **red control** corrupts the permutation by one block and confirms the
  invariants catch it. Without that, the cases prove only that they agree with
  themselves.

None of this touches the kernel call, the gate projection or the output
ordering under a real forward. Those are unexercised.

## The blocker, and it has two halves

**Half one: core cannot load the gate.** ComfyUI master carries no
`gate_compress` in `comfy/ldm/minimax/model.py` and no detection for it in
`comfy/model_detection.py`. Comfy-Org/ComfyUI#15958 is a draft that adds both,
in twelve lines. Without it the checkpoint's 150 gate keys have no slot on the
constructed model and are dropped on load with a warning -- **and the render
then succeeds, giving you the dense base checkpoint.** That is why
`_gate_modules` refuses by name: a silent dense render that the user believes
is VSA is worse than an error.

**Half two: core would not use it anyway.** That PR's own comment says the
weight is "unused by the dense forward; consumed by sparse attention patches".
So core loads it and something else has to compute the gate and pass it to
`sol_attn` as `coarse_gate`. This node is that something. As far as searching
found on 2026-08-30, the only other one is
`coderef/comfyui-minimax-h3-audio-T8/fast_h3_vsa_advanced.py`.

The kernel half is already here: the installed `comfy_kitchen` exposes
`coarse_gate`, `tail` and `block_len`, and
`bench/check_solattn_correctness.py` grades all three against the algorithm's
own eager reference.

## What would settle it

A render, and nothing short of one. The order:

1. Apply #15958 to the ComfyUI checkout. It is additive and touches two files.
2. Load the checkpoint and confirm the gate keys are no longer dropped.
3. Wire this node with no Sol-Attn node, `keep_percent` at the distillation's
   own sparsity, `pooled_tail` off.
4. Compare against the same checkpoint run dense -- which is what it does today
   without the PR, so that arm already exists.

Step 4 is a weight-level comparison and answers "does each arm satisfy the
brief", never "which clip is better": a rendered pair cannot A/B a numerical
change, and `docs/eval_comparison.md` section 3 is the process for anything
that will be quoted.
