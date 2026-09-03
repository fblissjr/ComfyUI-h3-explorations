# Sol exact-P-quantization and the Base16 capture

**Dated research record, 2026-09-03.** This records an audit and the resulting
experiment design. It is not the authority for shipped Sol settings;
[`../SOLATTN.md`](../SOLATTN.md) owns those. No workflow, default, model,
kernel, or installed wheel was changed by this analysis.

The target models throughout are the checkpoints this repository actually
wires: the **pruned INT8 ConvRot FL2VA and Ref2VA DiTs**. Results from BF16
release weights, an unpruned DiT, or one partition must not be silently carried
to these targets or to the other partition.

## Conclusions

1. Kijai's `sol_exact_pquant` branch contains the same full-range per-key-block
   probability quantization already present in our installed
   `0.2.31+sol.d25f2e8` wheel. Its additional FP16 support is inert for H3
   today because [`sol_attn_h3.py`](../../sol_attn_h3.py) admits BF16 only.
   Rebuilding the branch together with our count-output work should therefore
   leave H3's BF16 Sol numerics unchanged from the installed wheel.
2. Rebase the **direct `sol_attn`** `blk_cnt` work onto comfy-kitchen v0.2.32
   and offer that small contract upstream. Do not replace the installed wheel
   with plain PR #150: it lacks `blk_cnt`, which the route observer requires.
3. A real-activation measurement is warranted, but it should be folded into
   the capture already needed for H3 rather than spend a render solely to
   repeat PR #150's result.
4. A no-Sol Base16 capture is a useful **dense-trajectory control** for an
   old-versus-new kernel replay. It is not a canonical production-trajectory
   capture, because later-block activations have not inherited earlier Sol
   outputs.
5. Neither that capture nor the route-count record can choose a dense-block
   set across all 50 blocks. Counts measure cost; dense-block selection needs
   output sensitivity on the same block inputs and set-level validation.
6. Base16/Base20 results do not establish PDD5-8 behavior, and FL2VA results
   do not establish Ref2VA behavior. Those paths need separately attributed
   evidence.

## 1. What is in the upstream branches

### PR #150: `sol_exact_pquant`

[Comfy-Org/comfy-kitchen PR #150](https://github.com/Comfy-Org/comfy-kitchen/pull/150)
is three commits on v0.2.32 as observed on 2026-09-03:

- quantize the exact branch's probability matrix against each key block's own
  maximum, floored `2^-20` below the running maximum;
- expose `sol_attn_chunked` at package top level; and
- accept FP16 Q/K/V and output in addition to BF16.

Comparing its `sol_attn_exact.cu` against Kijai's earlier
`sol_attn_continued` commit `402fc90` shows no change to the BF16 exact-branch
algorithm. The remaining changes in that file are comments and output-type
templating. The installed `d25f2e8` file is bit-for-bit the `402fc90` exact
kernel, so PR #150 is numerically new against upstream v0.2.32 but not against
what this box already runs.

The H3 wrapper still refuses `q.dtype != torch.bfloat16` at
`sol_attn_h3.py::_ineligible`. FP16 support changes capability in
comfy-kitchen, not this repository's current H3 route. If the wrapper ever
admits FP16, its current “kernel is bf16-only” refusal text must change with
the eligibility rule.

The public chunked export is also behaviorally inert for this pack today.
[`sol_chunked_h3.py`](../../sol_chunked_h3.py) deliberately calls the CUDA
backend's `sol_attn_chunked` directly, and no generated workflow currently
wires `MiniMaxH3SolChunked`.

### `minimax_vae`

Kijai's `minimax_vae` commit adds fused operations shaped for ComfyUI's H3
video VAE: group-norm/SiLU/padding plus Conv3D, FP16 linear/Conv3D paths, and
residual or RMS-normalized INT8 linear variants. As observed on 2026-09-03,
neither ComfyUI core nor this pack calls those operations. It is groundwork,
not a current H3 execution path, and should not be merged or benchmarked here
until a consumer exists.

### PR #146

[Comfy-Org/comfy-kitchen PR #146](https://github.com/Comfy-Org/comfy-kitchen/pull/146)
changes the `tail=False` long-call exact consumer. The shipped H3 graphs use
`tail=True`, so it does not change this experiment or the current production
route. It is a separate integration question.

## 2. What our count-output branch contains

The clean `sol-blk-cnt` branch consists of:

- `cc81f55`: optional count output on direct `sol_attn`;
- `24908e1`: schema default and corrected tie wording; and
- `b292c74`: a batch-slice control proving a batched count tensor matches the
  corresponding individual calls.

It is based on upstream `c1c6751`, not v0.2.32, so it needs rebasing before an
upstream PR. The direct API is the smallest coherent first contribution:
`blk_cnt=None` changes nothing, while a supplied buffer receives the route
stage's existing `(B,H,NQ)` count slice from the same invocation that produced
the output.

The installed `d25f2e8` branch additionally exposes counts from
`sol_attn_chunked`. That second surface is useful to the experimental chunked
node but is absent from every generated workflow. It should either be a
follow-up or be explicitly made a cross-backend public contract; it should not
silently enlarge the direct-Sol PR merely because both commits exist locally.

If PR #150 merges before the count PR is submitted, rebase onto its merge
commit. Until then, target v0.2.32 and state that the count commits were also
applied cleanly to the PR #150 head. Do not install either upstream branch
alone over `d25f2e8`, because doing so removes the observer's count surface.

## 3. What has and has not been measured

[`measure_sol_exact_variants.py`](../../bench/measure_sol_exact_variants.py)
ran the two exact algorithms on the same seeded random BF16 inputs:

| all blocks routed, T=4096 H=8 | rel L2 against FP32 dense |
|---|---:|
| upstream exact + `blk_cnt`, `24908e1` | 0.9225% |
| per-block-P exact, `7cb749c` | 0.9093% |

At the shipped `tau=1.0`, both were approximately 1.145% from the eager Sol
reference and had indistinguishable isolated kernel timing. This is a random
input result, not an H3 estimate. The records are
[`2026-09-01_sol_exact_main_24908e1.json`](../../bench/results/2026-09-01_sol_exact_main_24908e1.json)
and
[`2026-09-01_sol_exact_continued_7cb749c.json`](../../bench/results/2026-09-01_sol_exact_continued_7cb749c.json).

PR #150 reports a larger gain on one real H3 capture: all-block relative L2
against FP32 falls from 1.96% to 1.40%, and at 10% keep the mean/worst per-head
cosine moves from 0.9830/0.9447 to 0.9853/0.9521, at no more than 3% per-call
cost. That result is upstream's measurement, not independently reproduced
here. The two records do not disagree: their inputs differ, and H3's score
distribution is exactly the variable random inputs erase.

The previous Q/K/V captures were deliberately deleted after their inventory
was written. [`2026-08-30_capture_inventory.json`](../../bench/results/2026-08-30_capture_inventory.json)
preserves what they contained, not the tensors needed to replay the new branch.

## 4. Base16/Base20 changes the exposure, not the kernel contract

The kernel's eligibility, workspace per call, head shape, and count-output API
do not depend on the sampler's number of evaluations. The number and
distribution of calls which reach it do.

The following was executed through ComfyUI's real `calculate_sigmas` path at
video shift 12, then classified through the current Sol gate. Indices are
zero-based model evaluations and exclude the terminal zero sigma.

| workload | window | Sol evaluations | dense evaluations |
|---|---|---|---|
| PDD8 | `.20-.74` | `2-5` = 4/8 | `0-1,6-7` = 4/8 |
| Base16 | `.20-.90` | `4-14` = 11/16 | `0-3,15` = 5/16 |
| Base20 | `.20-.90` | `5-18` = 14/20 | `0-4,19` = 6/20 |

The non-PDD resolver deliberately leaves both 16 and 20 steps at the inherited
`.90` end; its smaller-step table does not name them. At 20 steps, evaluation
4 lies on the floating boundary and the executed comparison leaves it dense.

Consequences:

- per-block-P quantization runs during a much larger share of the base
  trajectory than PDD8, so its cumulative importance may be larger even if
  its per-call improvement is unchanged;
- a base capture removes the PDD LoRA's merged-weight and replicated-head
  behavior, making it the cleaner independent kernel-arithmetic test;
- its activations are nevertheless a different population from PDD, so its
  dense-block ranking cannot become a PDD default; and
- 16 and 20 steps should be compared at similar trajectory percentages, not
  by blindly reusing the same integer indices.

The canonical base graph remains 16 steps. A 20-step record is a separate
experimental workload, not a replacement authority for Base16.

## 5. Capture designs and their claims

Every production-scale capture below means 1344x768, 345 frames, a fixed
prompt and seed, and the appropriate pruned INT8 ConvRot DiT. Timing from an
instrumented render is not quotable.

### Dense-trajectory Base16 control

The presently proposed no-Sol graph is valid for isolating kernel arithmetic:

```text
model   pruned INT8 ConvRot FL2VA base
steps   16, simple, video/audio shift 12/3
blocks  0,24,32,40,49
steps   4,8,12,14,15
Sol     absent; Sage supplies the live trajectory
```

Evaluations 4, 8, 12, and 14 are early/middle/late points that the production
Sol window would route. Evaluation 15 is the dense-tail control. With Sol
absent, all requested cells reach the capture seam, and the saved Q/K/V can be
replayed through both exact kernels without either kernel having generated the
inputs.

This must be labelled **dense-trajectory control**. It does not reproduce the
canonical production distribution at deeper blocks or later evaluations,
because none of its earlier blocks ran through Sol.

At the observed 345-frame packed length `S=104,361`, one BF16 Q/K/V cell is
about 4.18 GiB. Five blocks times five evaluations is about **104.5 GiB** before
manifest overhead. Print and verify the disk budget before arming.

### Canonical production-trajectory Base16 capture

Keep the same model, geometry, prompt, seed, blocks, and evaluations, but run
the canonical Sol/Sage composition with `dense_blocks=""`. This answers what
Q/K/V the shipped trajectory actually presents after earlier Sol calls. Its
records must retain the actual per-cell route.

The dense and canonical captures are complementary:

- dense trajectory is the cleaner independent numerical control;
- canonical trajectory is the relevant population for a shipped decision.

If only the dense capture runs, it is enough to grade PR #150's arithmetic and
not enough to set production policy. Preserve its exact prompt and seed so a
matched canonical capture can follow.

### Base20 equivalents

If 20 steps becomes a target, comparable points are:

```text
Sol points: 5,10,15,18
dense-tail control: 19
```

This remains its own workload. Do not relabel a Base20 result as Base16 merely
because the selected percentages are similar.

### Ref2VA and PDD

Repeat only after the first machinery and offline replay are graded:

- Ref2VA needs its own reference-conditioned capture on the pruned INT8
  ConvRot Ref2VA DiT; and
- PDD5-8 needs its own legal PDD schedule, LoRA, fused heads, and capture cells.

One capture can validate an instrument. At least two scenes/seeds are needed
before generalizing a routing policy, and partition-specific conclusions stay
partition-specific.

## 6. Why this still does not choose `dense_blocks`

The Sol route record reports how many key blocks were consumed. That is a cost
measurement. It contains no dense output and therefore no measure of what was
lost by routing sparsely.

Likewise, five captured transformer blocks can rank those five but say nothing
about the other 45. A production dense-block decision requires all of:

1. an all-50-block sensitivity statistic on the actual target
   model/schedule/partition, preferably Sol versus the real fallback on the
   same live input while retaining only reductions;
2. a proposed **set**, because independently good blocks can interact through
   later activations;
3. a measured timing cost for that set in a clean, uninstrumented process; and
4. multi-scene validation before it enters shared workflow defaults.

Until that exists, both shared configurations correctly keep
`dense_blocks=""`. The earlier `0-2,32` list remains an experiment, not a
default recovered by this capture.

## 7. Recommended order

Within the comfy-kitchen lane:

1. rebase the direct `blk_cnt` commits onto v0.2.32 and submit the minimal
   upstream PR;
2. grade the currently planned dense Base16 capture with an offline harness
   that runs both exact-kernel variants on the identical saved tensors;
3. run the matched canonical-Sol Base16 capture if the result will inform a
   shipped default;
4. add all-block sensitivity summaries before proposing any protected-block
   list; and
5. defer `minimax_vae` until ComfyUI has a real consumer.

Outside this lane, commit `e38655d` landed after this analysis began and before
this record was committed. It closes the two PDD contract defects the audit had
identified: exact checkpoint/sidecar identity now includes both INT8 codes and
FP32 scales, and the population is asserted against the release's 50-block,
208-module layout. The baked checkpoint and its paired stripped sidecar still
do not exist; producing and grading that pair is the remaining bake work. See
[`../h3_pdd.md`](../h3_pdd.md), which owns the current contract.

## Addendum, later on 2026-09-03: the dense-trajectory control was run and graded

The capture in section 5 rendered as specified (25 cells, retained outside
the repo; inventoried and given a manifest later the same day,
`bench/results/2026-09-03_capture_{inventory,manifest}_base16.json`) and was
replayed through both exact-branch variants with
`bench/measure_sol_exact_variants.py --capture`. Records:
[`2026-09-03_sol_exact_base16_capture_d25f2e8.json`](../../bench/results/2026-09-03_sol_exact_base16_capture_d25f2e8.json)
(installed, block-max P) and
[`2026-09-03_sol_exact_base16_capture_24908e1.json`](../../bench/results/2026-09-03_sol_exact_base16_capture_24908e1.json)
(upstream running-max P). Direction: with every block routed the block-max
build is several times closer to fp32 on every cell, a larger gap than PR
#150's own capture reports; at tau 1.0 WITHOUT sink ranges (the
`tau_1.0_no_sinks` arm, an unsunk diagnostic -- this sentence first said
"the shipped tau", withdrawn below) the gap is small because routing
dominates. ~~Block 49's all-routed error is far above every other block's on
both builds, so it is quantisation rather than sparsity.~~ **Withdrawn the
same day, in place** (Codex's review): true only on the block-max build and
only under the whole-tensor metric, and the reasons are below. The Sage
modes were graded on the same cells with `bench/grade_sage_on_capture.py`
([`2026-09-03_sage_on_base16_capture.json`](../../bench/results/2026-09-03_sage_on_base16_capture.json)),
which is the Sol-versus-fallback comparison section 6 asks for, on five
blocks and one scene -- **but not on matched footing**: that record scored
512 sampled rows against float64 while the Sol records scored every row
against fp32 (Codex's review). The Sol records were regenerated the same
day with `sage_<mode>` arms on the same q/k/v, every row, the same fp32
reference; read the numbers from their `aggregate` block. The direction on
matched footing: the sage arms are identical across the two Sol records
(the consistency check the rerun carried), Sol with every block routed on
the block-max build is closer to exact than the shipped sage `auto`
fallback under both the whole-tensor and the per-row metric and sits near
sage's fp16 mode, the old running-max build is further from exact than
the fallback, and the unsunk tau arm is far further than either. So on
this one scene and five blocks, the exact branch's arithmetic is not what
a dense block would buy back; routing is. Two further
corrections from the same review: the `tau_1.0` arm passes no sink ranges
(the capture has no segment table; Sol's rope hook publishes it and Sol was
absent), so it is renamed `tau_1.0_no_sinks`, an unsunk diagnostic rather
than a bound on the shipped call (errors can cancel, so relative L2 and
cosine are not monotone in the sink ranges); and block 49's "four times" holds only on the new build and
only under the whole-tensor, norm-weighted metric -- per row it is barely
above block 40, on the old build block 40 is worse, and a `dense_blocks`
entry does bypass Sol's exact stage, so the earlier "no list touches it"
was wrong. This five-block record does not say whether a dense block 49
is worthwhile. The first "not established" item below is therefore
established; the rest stand. The scene is the covered-market prompt, so
every magnitude here is one scene's.

## Not established

- ~~that PR #150's H3 number reproduces on this box~~ -- established by the
  addendum above, on one scene;
- that a smaller kernel error produces a perceptible render difference;
- that Base16, Base20, PDD, FL2VA, and Ref2VA share a block ranking;
- that any non-empty dense-block set improves quality enough to pay its cost;
- that the new VAE operations improve this pipeline; or
- any performance number from an instrumented capture render.
