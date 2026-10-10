# Running VSA on H3: what the node does and what still blocks it

last updated: 2026-09-27 (core's gate support and the park only)

**This file owns `MiniMaxH3VSAAttention` -- what it does, what it refuses, and
what is verified about it.** It does not own the checkpoint
([`fastvideo_vsa_checkpoint.md`](fastvideo_vsa_checkpoint.md)) or the kernel
([`../../SOLATTN.md`](../../SOLATTN.md)), and asserts nothing against either.

**Parked since 2026-09-27 (pack 0.157.1).** The node refuses at execute
while `vsa_attention.PARK_OVERRIDE` is False, because `_publish_layout`
mutates the shared model outside `ModelPatcher` and would leak into every
later render on the server; the module docstring owns the defect and the fix.
For FastH3, use core's `BlockSparseAttention` with selection "vsa"
(`h3_config.FASTH3_CONTRACT_VSA`).

**It has rendered, once, and that answers a mechanical question only.**
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
| SLA | the algorithm: route, no pooled term | yes, and the Turbo-SLA LoRA existed | none since 2026-09-27: the SLA lane closed and `MiniMaxH3SolAttn` (which ran it with `pooled_tail` off) is deleted |
| VSA | route, no pooled term, plus a gated coarse branch | yes, the gate is a learned projection | core's `BlockSparseAttention` (selection "vsa"); this node is parked |
| PDD | the sampler: fewer evaluations | yes, the Acc LoRAs | shipped |

**VSA competes with PDD rather than complementing it.** The published
checkpoint's filename says `4step`, so it is a distillation that cuts
evaluations, which is what a PDD arm does by other means. Note that nothing in
the artifact supports the 4-step claim -- it carries no metadata at all -- so
that is a reading of a filename, not a property anyone here has verified.
**Since 2026-09-04 the reading has a source**: FastVideo's own card calls it
a four-step DMD2 distillation, and two serving engines pin the schedule as
code; [`fastvideo_vsa_checkpoint.md`](fastvideo_vsa_checkpoint.md) section 6
holds the pointers. Still unverified here, but no longer a guess.

## Why it is a separate node, which is not the same as "why it must be"

**Corrected 2026-08-30.** This section said VSA *cannot* be a widget on the Sol
node because an `optimized_attention_override` is handed Q, K and V already
built. That is true of the hook and false as a conclusion: a forward pre-hook
on `Attention` can stash the block input into `transformer_options`, which the
override receives, and `MiniMaxH3SolAttn` used exactly that route to publish
the block index (until 2026-09-27; `MiniMaxH3Sol` reads core's `block_index`). Verified by executing the pattern rather than reading
it. So the gate is reachable from there.

The real reasons are weaker and worth stating as what they are:

- VSA needs the gate, the cube reorder AND the padding together, and putting a
  second reordering inside a hook that already owns the Morton one is a
  collision, not an impossibility;
- the two regimes are **mutually exclusive** at the same 50 blocks, so sharing
  a node would mean one silently winning;
- the T8 pack's implementation, which this node follows, replaces the block
  forward too.

So the block forward is replaced, through `patches_replace["dit"]`, on the 50
main blocks. The 2 token-refiner blocks carry no gate and are left alone -- an
upstream attention node (the kitchen backend or sage) still handles those, which is why this node warns about an
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

## Core support: in stock core since e308cc73

Stock ComfyUI core builds `to_gate_compress` since core commit e308cc73
("Add Sparse Attention node", #16072): `comfy/model_detection.py` sets
`gate_compress` from the checkpoint's own keys and `comfy/ldm/minimax/model.py`
creates the slot. So the blocker this section used to describe is gone, and
with it the refusal that kept `_publish_layout` from running; the node is
parked for that reason (see the top of this page).

*2026-09-27: until today this section said core here was stock with no
`gate_compress`, and that the blocker stood until the draft PR #15958 merged.
That draft was applied as an uncommitted change on 2026-08-30 and lost on
2026-08-31; `bench/check_vsa_core_patch.py` holds that history.*

`bench/check_vsa_core_patch.py` checks that the gate support is consistent
across the two core files (a half-present change is silent in one direction)
and, when the VSA checkpoint is on disk, that all its gate keys find a slot.
It cannot tell stock support from a local edit.

**Half two, core would not use it by itself: still true of the model code.**
Core's comment on the slot says the weight is "unused by the dense forward;
consumed by sparse attention patches". Core's own `BlockSparseAttention`
(`comfy_extras/nodes_sparse_attention.py`, selection "vsa") is such a patch
and passes the gate to `sol_attn_chunked` as `coarse_gate`; this node is
another, and `coderef/comfyui-minimax-h3-audio-T8/fast_h3_vsa_advanced.py` a
third.
`bench/results/2026-09-27_attention_parity.md` compares this node, core's and
FastVideo's.

The kernel half was never blocked: the installed `comfy_kitchen` exposes
`coarse_gate`, `tail` and `block_len`, and
`bench/check_solattn_correctness.py` grades all three against the algorithm's
own eager reference.

## The first render, 2026-08-30

Record: `data/results/2026-08-30_vsa_first_render.json`. Arms are
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
this session's. Arm identity lives only in the embedded graph, which is
the same trap wearing a different hat. So the verifier identifies the arms from
the graph each file was rendered from and fails on a mismatched pair; without
that case, a wrong pair passes the pixel comparison and reads as a result.
(2026-09-23: videos carry no container tags any more, and the graph is read
from the first-frame PNG beside each clip; `bench/diff_clip_graphs.py::graph_of`
says how.)

**What it does not establish, and the list is longer than what it does.** No
quality claim: a rendered pair cannot A/B a numerical change, and this pair
changes the attention regime outright. Not that `keep_percent` 10.0 is right --
it is the distillation's published sparsity and is unmeasured here. And nothing
at the lengths this repo actually renders: 22,121 rows is far below the
31k-128k of the shipped graphs, and sparse attention's advantage grows with
length, so this shape is close to the least favourable one available.

**The timing at THIS length is not measured, in either direction.** Warm, VSA
22.68 s against the dense control 24.18 s, one run each. One run per arm cannot
separate 6% from run-to-run excursion, so it is neither evidence of a speedup
nor of its absence. An earlier wording said "no speedup", which this sample
cannot support either. **A longer shape settles it -- see below.**

**The shape was the least favourable available, and that is now measured
rather than hedged.** `bench/preflight_graph.py` prices any graph statically,
without touching the card: this arm packs 22,121 rows, against 109,457 for the
shipped t2v graphs and 63,233 for the cheapest square-canvas probe. So it sat
below every shipped graph, by 3x against the cheapest and 5x against the
common ones -- and sparse attention's advantage grows with length.

## Length scaling, 2026-08-30

Record: `data/results/2026-08-30_vsa_length_scaling.json`. Same two arms, same
seed and sampler, 768x768, at 124 frames and again at 362.

| packed rows | VSA | dense (sage) | verdict |
|---|---|---|---|
| 22,121 | 22.68 s | 24.18 s | not measured, one run each, 1.5 s apart |
| 63,233 | 71.55, 68.83 s | 102.43, 102.50 s | **attributable**, two runs each |

At the longer shape the between-arm gap is about 32 s while the largest
within-arm spread is 2.72 s -- an order of magnitude smaller, which is what
makes it attributable where the first pair was not. VSA is about **1.46x**
faster than sage on the same checkpoint there.

**Two points is a direction, not a curve.** What it shows is that the
advantage is length-dependent and appears where sparse attention predicts it
should, which is the thing the first render could not see.

**And it is still not a quality result.** Speed says nothing about output. The
control is SAGE, not Sol-Attn, so this says nothing about VSA against the
sparse attention this repo actually ships. 63,233 rows is still below every
shipped t2v graph. And at both lengths the correctness side holds
independently: VSA differs from its control and each arm reproduces itself at
the same seed, on decoded pixels, with the arms identified from the graph
embedded in each file.

## This capture cannot be re-asked, and that is a defect in it

Recorded against CLAUDE.md's `capture broadly first` rule (owner, 2026-08-30).
The two length arms recorded **two numbers**: total wall time, and a hash of
the decoded pixels. So the 1.46x cannot be split between attention and
everything else -- and attention is the only part VSA touches. No latents were
kept, so no fidelity question can be scored offline at all. VRAM, one of the
two things a sparse kernel is for, was not recorded.

The next VSA measurement should record the output LATENT rather than the
encoded video, per-step time and peak VRAM, and `PackedLayout.segments` --
which no capture currently carries and which is what blocks the
segment-boundary question for sage as well as for VSA. One field, two lanes.

The 1.46x is not withdrawn; two samples per arm against an order-of-magnitude
smaller spread is a sound wall-time observation. What is recorded is that it is
the only question those two renders can answer.

## What would settle the rest

1. ~~Core support.~~ In stock core since e308cc73 (#16072); see above.
   What blocks this node now is the park, which lifts only with the
   `_publish_layout` fix.
2. ~~Confirm the gate keys are no longer dropped.~~ Done 2026-08-30 on the
   draft-patched tree, all 50 placed. `bench/check_vsa_core_patch.py` re-runs
   it on current core when the checkpoint is on disk.
3. ~~Run it, with a dense control.~~ Done 2026-08-30; see above. Not
   re-run on stock core, and the park stops this node running at all.
4. **A length where sparse attention is supposed to win.** The shipped canvas
   at a shipped frame count, which is 31k-128k rows against this run's 22k.
5. **A sampler recipe.** The checkpoint's "4step" is a filename, not a
   property; the artifact carries no schedule. Whether 4 steps and this repo's
   default sampler are what it was distilled for is unknown, and a bad recipe
   would look exactly like a bad regime. **Narrowed 2026-09-04**: the recipe
   is now specified outside the artifact (five sigma grid points, t2va only;
   pointers in [`fastvideo_vsa_checkpoint.md`](fastvideo_vsa_checkpoint.md)
   section 6). What remains open is whether this repo's sampler lands on
   those points, which is a check, not a search.
6. **Anything perceptual**, which needs `docs/eval_comparison.md` section 3 --
   many seeds per arm, judged blind, recorded as a distribution.

Step 4 is a weight-level comparison and answers "does each arm satisfy the
brief", never "which clip is better": a rendered pair cannot A/B a numerical
change, and `docs/eval_comparison.md` section 3 is the process for anything
that will be quoted.
