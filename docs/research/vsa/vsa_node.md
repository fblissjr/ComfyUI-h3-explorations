# Running VSA on H3: what the node does and what still blocks it

last updated: 2026-08-30

**This file owns `MiniMaxH3VSAAttention` -- what it does, what it refuses, and
what is verified about it.** It does not own the checkpoint
([`fastvideo_vsa_checkpoint.md`](fastvideo_vsa_checkpoint.md)) or the kernel
([`../../SOLATTN.md`](../../SOLATTN.md)), and asserts nothing against either.

**It has now rendered, once, and that answers a mechanical question only.**
2026-08-30: VSA runs, its gate is consumed, and it reproduces itself at a fixed
seed. Nothing about output QUALITY is established and nothing here should be
read as a recommendation. See "The first render" below.

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

## The blocker: half cleared 2026-08-30, half remaining

**Half one, core cannot load the gate: CLEARED on this box, and only on this
box.** ComfyUI master carries no `gate_compress`;
`github.com/comfyanonymous/ComfyUI` PR #15958 adds it in twelve lines across
two files, and **it is a DRAFT**. Applied here on 2026-08-30 from head
`10febb01` as an UNCOMMITTED working-tree change on master rather than a merge,
so `git checkout --` on the two files reverts it and a later `git pull` refuses
loudly instead of quietly merging a draft.

`bench/check_vsa_core_patch.py` is the provenance record. It reports absence
rather than failing on it -- a machine without the patch is the normal state,
and failing on it would train a reader to ignore red. What it does fail on is a
HALF-applied patch, because the two halves fail in opposite directions and one
of them is silent: with only the model change, every H3 model takes a
`gate_compress` parameter that detection never sets, so it stays False and
behaves exactly like stock while `grep` says the support is there.

**Verified against the artifact, not just the source.** Detection sets
`gate_compress` from the checkpoint's own keys, 50 `to_gate_compress` modules
are constructed, all 50 gate weights find a slot and no weight key is left
without one -- executed on meta tensors, so nothing was allocated. Before the
patch all 50 were dropped on load and the render succeeded as the dense base.

**Consequence worth stating in any measurement taken here:** the H3 model this
box builds is not the one stock ComfyUI builds.

**Half two, core would not use it anyway: REMAINS.** That PR's own comment says
the weight is "unused by the dense forward; consumed by sparse attention
patches". So core now loads the gate and something else still has to compute it
and pass it to `sol_attn` as `coarse_gate`. This node is that something. As far
as searching found on 2026-08-30, the only other one is
`coderef/comfyui-minimax-h3-audio-T8/fast_h3_vsa_advanced.py`.

The kernel half was never blocked: the installed `comfy_kitchen` exposes
`coarse_gate`, `tail` and `block_len`, and
`bench/check_solattn_correctness.py` grades all three against the algorithm's
own eager reference.

## The first render, 2026-08-30

Record: `bench/results/2026-08-30_vsa_first_render.json`. Arms are
`workflows/h3_probe_vsa.json` and `workflows/h3_probe_vsa_dense.json`, matched
at seed 730451892, 4 steps, 768x768, 124 frames, 22,121 packed rows.

**What it establishes.** VSA runs to completion. The node logs `VSA on 50
blocks` with no fallback warning, and -- the part that actually settles it --
its output differs from the dense control on the same checkpoint at the same
seed, so the gate is genuinely consumed rather than the replacement quietly
falling through to the original block. Two VSA runs at the same seed produce
identical pixels, so the difference is attributable to the regime rather than
to noise.

**A trap worth carrying forward, and it has bitten two sessions
independently.** The first comparison was done on `md5sum` of the mp4 files and
was WRONG in a way that looked right. Two container tags cause it, and only one
explains the same-arm case: `format.tags.comment` carries the whole API prompt
under VHS's `save_metadata`, so any two ARMS differ by construction; and
`format.tags.creation_time` is a wall clock, so any two RUNS differ, including
two runs of one arm. The muxer and the codec are not the cause -- remuxing a
file twice, and re-encoding it twice at the same settings, each give identical
bytes. `docs/h3_pdd.md` has recorded the first mechanism since 2026-08-27;
rediscovering it says the note was not reachable from where people look.

The tell was that the two VSA runs hashed differently while their file sizes
matched to the byte. `bench/verify_vsa_render.py` compares the DECODED RGB
stream, and exists so nobody repeats it. **The one-way implication survives
both mechanisms**: identical container still implies identical frames, so
nothing concluded from a matching hash is withdrawn.

**And the filenames carry no arm information.** `bench/smoke_h3.py` hard-codes
one `_smoketest` prefix, so every session rendering on this box shares one
output counter and consecutive files may belong to different sessions -- a peer
session went looking for its own pair, found a consecutive one, and it was
this session's. Arm identity lives only in the embedded comment tag, which is
the same trap wearing a different hat. So the verifier identifies the arms from
the graph inside each file and fails on a mismatched pair; without that case, a
wrong pair passes the pixel comparison and reads as a result.

**What it does not establish, and the list is longer than what it does.** No
quality claim: a rendered pair cannot A/B a numerical change, and this pair
changes the attention regime outright. Not that `keep_percent` 10.0 is right --
it is the distillation's published sparsity and is unmeasured here. And nothing
at the lengths this repo actually renders: 22,121 rows is far below the
31k-128k of the shipped graphs, and sparse attention's advantage grows with
length, so this shape is close to the least favourable one available.

**The timing is NOT MEASURED, in either direction.** Warm, VSA 22.68 s against
the dense control 24.18 s, one run each. One run per arm cannot separate a 6%
difference from run-to-run excursion, so this is neither evidence of a speedup
nor evidence of its absence. An earlier wording here said "no speedup", which
is a claim this sample cannot support either. The two warm numbers share a
cached text encode, which is what makes them comparable to each other at all;
the cold run at 43.43 s carries the model load and is comparable to neither.

## What would settle the rest

1. ~~Apply #15958 to the ComfyUI checkout.~~ Done 2026-08-30, working tree.
2. ~~Confirm the gate keys are no longer dropped.~~ Done, all 50 placed.
3. ~~Run it, with a dense control.~~ Done; see above.
4. **A length where sparse attention is supposed to win.** The shipped canvas
   at a shipped frame count, which is 31k-128k rows against this run's 22k.
5. **A sampler recipe.** The checkpoint's "4step" is a filename, not a
   property; the artifact carries no schedule. Whether 4 steps and this repo's
   default sampler are what it was distilled for is unknown, and a bad recipe
   would look exactly like a bad regime.
6. **Anything perceptual**, which needs `docs/eval_comparison.md` section 3 --
   many seeds per arm, judged blind, recorded as a distribution.

Step 4 is a weight-level comparison and answers "does each arm satisfy the
brief", never "which clip is better": a rendered pair cannot A/B a numerical
change, and `docs/eval_comparison.md` section 3 is the process for anything
that will be quoted.
