# What upstream says: the paper, Sol-Engine, and the other packs

Last updated: 2026-08-16. Renamed from `sol_engine_reference.md` the same day,
when the paper was added and the scope widened past one vendor's framework.

**This page states what other people do and claim. It asserts none of our
numbers.** When a comparison against what we run is needed, it lives in the
doc that owns our side of it: [`docs/SOLATTN.md`](SOLATTN.md) for the knobs and
the measurements, [`docs/morton.md`](morton.md) for token order. If this page
and one of those disagree about our configuration, they are right.

Three sources, read three different ways, and the depth differs:

| source | how it was read | when |
|---|---|---|
| arXiv 2607.24027, the Sol-Attn paper | fetched: abstract, ablation summary, HTML v1. **Not a full end-to-end read** | 2026-08-16 |
| Sol-Engine | source, `coderef/Sana` at `origin/sol-engine`, HEAD `6fb7eb1` | 2026-08-15, 2026-08-16 |
| two third-party ComfyUI packs | their READMEs only | 2026-08-16 |

---

## The paper

**"Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention
Sparsification."** Read 2026-08-16, after this repo spent two weeks treating it
as unread. Scope qualifier that matters: the abstract, the ablation summary and
the HTML were read, not the whole PDF. Anything below is at that depth.

**The contribution is the correction, not the sparsity.** The framing is that
existing block-sparse methods fail two ways: routing is rigid and costly to
materialize, and unselected blocks are dropped outright, which degrades
accuracy under aggressive sparsity. Sol-Attn thresholds during online softmax
and **reuses the proxy scores of unselected blocks to approximate their
contribution** instead of discarding them. The ablation that matters compares
exact-only against exact-or-approx, and the correction's advantage **widens as
sparsity rises**.

Five things follow that bear directly on this repo:

- **`centroid_tail` is the method, not a side knob.** We ship it on and
  measured it at 2.5% of render time, which reads as a minor optimisation. It
  is the paper's core claim. Sol-Engine's own public API has no toggle for it
  at all.
- **`tau` is the paper's beta, and it is never swept.** The threshold is
  `t_i = mu_i + beta * sigma_i` over the row-wise mean and standard deviation,
  with beta described as a shared standardized cutoff. That confirms what the
  CUDA source says our `tau` means, and it means the paper does not settle
  1.0 against our 1.3. Sol-Engine defaults 1.0.
- **H3 is not evaluated anywhere in it.** Tested models are Wan 2.1-14B
  (720p/81f), HunyuanVideo-13B (720p/129f), LTX 2.3-22B (1080p, up to 721f),
  Bernini-14B for editing, a SANA-WM refiner, Ideogram 4 for 2K text-to-image,
  and Wan 2.1-1.3B on a 5090. Headline numbers are 2.1x end to end on video
  generation and 2.3x on editing, against dense.
- **No token reordering appears in it.** No Morton, no Z-order, no permutation,
  no spatial block layout. See [`docs/morton.md`](morton.md), which owns what
  that does and does not license.
- **sm89 was not an official target** when the paper and its docs were written.
  That changed on 2026-08-15; see the kernel table below.

---

## Sol-Engine's H3 recipe

Sol-Engine **supports MiniMax-H3 as a first-class model** with validated,
pinned configurations -- worth knowing, because until 2026-08-15 this repo
assumed H3 support was entirely kijai's port on top of a paper that does not
mention H3. The paper does not. The framework does.

Every H3 config runs at **1344x768, 124 frames** -- our canvas, and by
coincidence the exact geometry of the capture in `docs/morton.md`.

**The recipe is per-hardware-profile, not one policy.** This page said "their
H3 policy" until 2026-08-16 and that was wrong in a way that mattered: it
generalised the 4xA100 cell to H3 as a whole.

| profile | attention | thresholding | cache |
|---|---|---|---|
| 4xA100 (sm80) | **Triton reference**, tau 1.0 | `exact` | FirstBlockCache 0.08 |
| 4xH100 (sm90) | CuTe, tau 1.0 | `exact` | FirstBlockCache 0.08 |
| 8xGB200 (sm100) | CuTe, tau 1.0 | `diag` | FirstBlockCache 0.08 |
| **1x RTX 5090 (sm120)** | CuTe, tau 1.0 | `diag` | TeaCache 0.10 |

Sources: `coderef/Sana/models/minimax_h3/A100/adapter.py:550` and the H100 cell's same line,
`coderef/Sana/models/minimax_h3/GB200/gpu_infer.py:144`,
`coderef/Sana/models/minimax_h3/RTX5090/adapter.py:477`, and each cell's README.

**A100 is the Triton reference path**, because sm80 has no CuTe kernel. That is
the cell this page used to quote as "their validated H3 policy", while we
compare against a CuTe-class kernel. Not the same thing.

**Every profile runs the first two blocks and the first 10 of 50 steps dense.**
`SOL_ATTN_FIRST_DENSE_LAYERS=2` (RTX5090), `H3_SOL_DENSE_LAYERS=2` (GB200),
`dense_blocks: int = 2` (`coderef/Sana/models/minimax_h3/GB10/fusion_install.py:84`,
whose comment calls it "the released policy's first-two-blocks-dense rule").

**None of them reorders tokens.** Stated explicitly in the A100 README:
"pinned SGLang BF16 execution without `torch.compile` or token reordering".

### The single-card profile, which is the one to watch

`models/minimax_h3/RTX5090/` runs H3 on **one consumer GPU** with SGLang
layerwise component offload, and it ships an isolated Sol-only arm:
`config/minimax_h3/rtx5090_sol.toml` against `config/minimax_h3/rtx5090_dense.toml`,
at 1344x768, 124 frames, 50 steps, seed 0, tau 1.0, `diag`, first 10 steps and
first two blocks dense.

That is the closest thing upstream has to our own Sol-versus-dense arm: same
canvas, one card, a dense control rather than a different attention kernel. The
differences that remain are the card (sm120 against our sm89), the step count
(50 against our 16) and the baseline (BF16 dense against our sage).

**No number is published for that arm.** Only the full-opt 4.52x is, and
full-opt includes TeaCache and `torch.compile`.

---

## Side by side with `SOL_RECOMMENDED_CUDA`

Our column is a pointer, not a claim -- [`docs/SOLATTN.md`](SOLATTN.md) owns
every value in it and the evidence behind it.

| | Sol-Engine H3 | ours | note |
|---|---|---|---|
| `tau` | 1.0 | 1.3 | theirs is the paper default; ours was tuned locally. A third-party H3 pack landed on 1.3 independently |
| threshold mode | `exact` on A100/H100, **`diag` on GB200 and RTX 5090** | `diag`, unconditionally | **not the gap this page used to claim.** See below |
| KV sink | full prefix, dense prefix queries | `exact_kv_and_rows` | **the same choice.** Both sink the whole prefix and run its query rows dense |
| dense steps | first 10 of 50 (20%) | `start_percent` 0.2 -> 5 of 16 (31%) | same idea, different spelling; ours is the larger fraction |
| dense blocks | **first two, on every profile** | **none** | the clearest gap, and the cheapest to close. `SOL_ARTIFACT_INSURANCE` exists here and is not shipped |
| token reordering | **none, stated explicitly** | `morton=False` | **agrees with us.** [`docs/morton.md`](morton.md) owns why |
| cache | FirstBlockCache 0.08, or TeaCache 0.10 on the 5090 | none | we have no cache of any kind |
| steps | 50 | 16 | their baseline is 50-step Diffusers |

---

## What Sol-Engine has that we do not, and what blocks each

Re-derived from source on 2026-08-16. Two of these were miscategorised on this
page for a day: they were written as knobs our node fails to expose, and they
are not that.

### `thresh_type` -- the blocker is the kernel, not the node

Their two modes, in `coderef/Sana/techniques/sparse_backends/sol_attn/preprocess.py`:

- **`diag`** (`:178-185`): variance estimated as `sum_d q_d^2 * var(kc_d)` from
  the per-dimension variance of the key-block centroids, which assumes
  independence across the 128 dims.
- **`exact`** (`:255-271`): the real score row's `E[s^2] - E[s]^2` over all key
  blocks for that query.

comfy-kitchen's CUDA kernel computes `kcvar[d]` as the per-dimension variance of
the key-block centroids (`coderef/comfy-kitchen-sol/comfy_kitchen/backends/cuda/ops/sol_attn_preprocess.cu:123`), reduces
`sred[d] = c_d^2 * kcvar[d]` (`:192`), and thresholds at
`tau * sqrt(sum + 1e-6)` (`:199`), against mean-centred key centroids so the
mean term is already absorbed. **That is Sol-Engine's `diag` formula.**

So `exact` is not an unexposed input. It is a threshold kernel nobody has
written for this backend, and adding a widget would not produce it. What it
would buy is the true per-query score-row variance instead of a
diagonal-covariance estimate; what it costs is a second pass over the same
`[N x N]` centroid scores before routing can start.

**And the profile closest to our hardware already uses `diag`.** NVLabs' own
single-consumer-card H3 cell runs the mode we run.

### `kv_splits` -- there is nothing to have on this card

`coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:98-100` raises
`"kv_splits=2/4 is currently available on SM90 only"`, and their own docs say
B200, RTX 4090 and RTX 5090 use `kv_splits=1`. A 4090 gets 1 in Sol-Engine too.
This row was a gap on paper only.

### Dense layer guards -- nothing blocks this at all

We have the mechanism (`dense_blocks`) and ship it empty. Every H3 profile they
publish ships first-two-dense. Adopting it is a string in
`SOL_RECOMMENDED_CUDA`, not a build.

What we actually lost when `SolAttnBlockProbe` went with the Triton pack is the
ability to choose **our own** list from measurement. Copying a validated `0-1`
needs no instrument. The cost estimate and the decision live in
[`docs/roadmap.md`](roadmap.md).

### The SM89 CuTe kernel -- one dependency and one seam

PR #464, merged 2026-08-15: "a general BF16 SM89 CuTe DSL Sol-Attn forward
kernel using M64/N64 tiles, cp.async, and warp MMA". That is the 4090, and it
makes NVLabs' own kernel a **second independent implementation** beside the
comfy-kitchen `sol_attn` branch we build locally.

The code is already in our tree at
`coderef/Sana/techniques/sparse_backends/sol_attn/sm89/`, with dispatch at
`coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:11` (`(8, 9): "cute_sm89"`).

Their stated requirements against this box, checked 2026-08-16:

| requirement | here | |
|---|---|---|
| PyTorch >= 2.10 | 2.13.0+cu132 | ok |
| CUDA >= 12.8 | 13.2 | ok |
| Triton >= 3.6 | 3.7.1 | ok |
| `cuda-python` | present | ok |
| CuTe DSL / CUTLASS Python | **absent** | the only unmet one |

Their docs warn that a DSL version mismatch **fails at compile time rather than
falling back**, "so that a dense run is never reported as a sparse one" -- the
same failure discipline `check_sol_kernel.py` exists to enforce here.

The second blocker is API surface. Their public entry point
(`coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:397-408`) is
`sol_attn(q, k, v, *, scale, tau, thresh_type, kv_splits, sink_tokens, sink_start)`.
There is **no `sink_q`**, no `dense_blocks`, no `centroid_tail`, no
`max_blocks`, no `key_bias`. Our `exact_kv_and_rows` is comfy-kitchen's
invention; its query half would have to be done at the integration layer, which
is what their docs mean by "valid text-query rows still use dense attention".

So a head-to-head is "install one package, write a seam", not blocked.

### Token pruning and NVFP4 -- one is hardware, one is plumbing

Two of Sol-Engine's five methods, and **not applied to their H3 stack** either.

- **NVFP4 is hardware-blocked here.** `AGENTS.md:172`: needs Blackwell
  (sm_100+) plus `transformer_engine`. A 4090 is sm89. Not a choice they made
  and not one we can make.
- **Token pruning is not hardware-blocked**, but the policy in
  `techniques/methods/token_prune.py` is model-agnostic scaffolding whose
  runtime lives in SGLang rather than in that repo, and it needs a `ModelSpec`
  seam H3 does not have there. It ships on LTX-2.3, not H3.

---

## Morton in Sol-Engine, which does exist

Correcting a claim this repo made three times on 2026-08-15: **Sol-Engine does
ship Morton.** It is absent from the paper, absent from the published docs, and
present and on by default for Wan.

- `techniques/sparse_backends/sol_attn_backend.py` has `_morton3d_perm`
  ("canonical x/y/z-interleaved") and `install_wan_morton_forward`.
- Every Wan config description reads "released Sol-Attn at tau 1.0 with
  **global Morton3D order** and dense guards".
- HunyuanVideo has it **off**: `HUNYUAN_SOL_MORTON = "0"`.
- H3 has it absent entirely.
- **They ship only the 3D curve.** There is no `2d_frame` variant anywhere in
  Sol-Engine; that is kijai's addition for H3's non-uniform temporal axis.

Their stated reason for H3 needing none, from
`models/minimax_h3/GB200/sol_attn_h3.py`:

> **Morton reordering per attention call.** The released
> `install_wan_morton_forward` docstring says why not: only self-attention is
> order-sensitive, so the permutation belongs once at the block stack, and
> doing it per call cost more than the kernel saved. What the H3 integration
> then shows is that H3 needs *no* reordering at all -- the packed video tail
> is already a contiguous grid-ordered block, and the routing works on it
> directly.

And the ablation control they ship for Wan, `config/wan21_t2v_14b/reorder_only.toml`:

> "reorder-only control: global Morton3D order with every layer forced dense
> through the Sol-Attn adapter."

`WAN22_SOL_REORDER=1` with `WAN22_SOL_DENSE_LAYERS="0-39"`.

**Whether any of that transfers to H3 is argued in
[`docs/morton.md`](morton.md), not here.** That page holds the counter-argument,
the measurement that disagrees with the assertion above, and what happened when
the reorder-only control was run on this box.

---

## FirstBlockCache

`models/minimax_h3/A100/first_block_cache.py`, about 30 lines of policy around a
threshold. Gated on `H3_FIRSTBLOCKCACHE=1`, default threshold `0.08` from
`H3_CACHE_THRESHOLD`, with decisions synchronized across ranks so a multi-GPU
run cannot diverge. The RTX 5090 cell uses TeaCache at 0.10 instead, with five
retained steps and one cooldown step.

The idea, which is not ours and not new: run the first transformer block, look
at how much its output changed against the previous step, and if the change is
below the threshold, reuse the whole step's residual instead of running the
remaining blocks. A step-skipping cache, orthogonal to attention entirely.

Notes for whoever picks it up:

- It is a **different quality tradeoff from sparse attention**, not a
  complementary one. Sparse attention approximates within a step; a cache skips
  steps outright. The `h3-turbo-eval` prior art measured Spectrum, a forecasting
  variant of the same family, at 0.789 frame correlation against stock, i.e.
  "closer to changing the seed than to accelerating the same render".
- Their thresholds are tuned at **50 steps**. We run 16. A cache that skips
  steps is far more aggressive when there are only 16 of them, and the
  documented failure mode of too-few-steps on H3 is prompt adherence collapse,
  not blur (`h3_config.py`: at 12 steps the third scripted shot never happens).
  So 0.08 should not be carried over.
- The rank synchronization is irrelevant on one GPU.
- ComfyUI already has TeaCache-style nodes in the wild; checking whether one
  works on H3 is cheaper than porting this.

---

## Other ComfyUI Sol-Attn packs

Found 2026-08-16, READMEs only. They matter mostly as independent readings of
the same knobs.

**[Saganaki22/ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn)**
is H3-specific: zero-copy MiniMax H3 nodes, published benchmarks, and
**scheduled tau** -- ramping the threshold across sampling steps, sparse early
to dense late, with selectable interpolation curves. Ours is a binary sigma
window; that is the continuous version of the same idea.

Two details worth carrying:

- They default **tau to 1.3, "tuned locally", against the paper's 1.0**.
  Independent convergence on the value we ship, arrived at separately. It is
  not confirmation that 1.3 is right -- two people tuning by eye on different
  cards can land in the same place for the same wrong reason -- but it is worth
  more than one.
- Their quality evaluation is **correctness against a reference** (L2 against
  SDPA, bit-identical strided views), not perceptual.

Their speed numbers are on a 5090 with `torch.randn` tensors, which is both the
wrong card and the synthetic-input trap `docs/SOLATTN.md` already carries. Not
comparable to anything measured here.

**[sumeetprashant/ComfyUI-SolAttn](https://github.com/sumeetprashant/ComfyUI-SolAttn)**
exists and was not read past its README.

**Neither pack reorders tokens.** With the paper and Sol-Engine, that makes four
searched sources with no token reordering outside kijai's packs, and still zero
measurements of Morton's effect on output anywhere.

---

## Numbers, and why not to quote them against ours

Sol-Engine's published H3 speedups: **4.52x on 1x RTX 5090** (1045.4 s ->
231.2 s), 3.95x on 8xGB200, 3.92x on DGX Spark GB10, 3.56x on 4xH100, 3.55x on
4xA100 (217.32 s -> 61.28 s).

**None of these is a Sol-Attn number.** Full-opt for H3 is context parallel plus
kernel fusion plus Sol-Attn plus a cache, against a 50-step dense baseline. Ours
is Sol-Attn against sage at 16 steps on one 4090. Different stack, different
baseline, different hardware, different step count.

This is `docs/SOLATTN.md`'s unit trap at a larger scale, and the numbers are not
comparable in either direction. The one arm that *would* be comparable --
`rtx5090_sol` against `rtx5090_dense` -- has no published result.
