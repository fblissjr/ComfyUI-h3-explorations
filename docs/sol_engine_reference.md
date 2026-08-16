# Sol-Engine's own MiniMax-H3 recipe, and how ours differs

Last updated: 2026-08-15. Read from `coderef/Sana` at `origin/sol-engine`
(HEAD `6fb7eb1`), not from the web.

NVLabs ships Sol-Attn inside **Sol-Engine**, a branch of the Sana repo, and
Sol-Engine **supports MiniMax-H3 as a first-class model** with a validated,
pinned configuration. That is worth knowing because until 2026-08-15 this repo
assumed H3 support was entirely kijai's port on top of a paper that does not
mention H3. The paper does not. The framework does.

Their H3 config runs at **1344x768, 124 frames** -- our canvas, and by
coincidence the exact geometry of the capture in `docs/morton.md`.

## Their validated H3 policy, verbatim from the source

`models/minimax_h3/A100/README.md`:

> **Attention:** Triton Sol-Attn with `tau=1.0`, exact thresholding, a
> full-prefix KV sink, dense prefix queries, and the first 10 steps and first
> two blocks dense.
> **Cache:** FirstBlockCache with threshold `0.08` and synchronized decisions
> across ranks.
> **Runtime:** pinned SGLang BF16 execution without `torch.compile` or token
> reordering.

## Side by side with `SOL_RECOMMENDED_CUDA`

| | Sol-Engine H3 | ours | note |
|---|---|---|---|
| `tau` | 1.0 | 1.3 | theirs is the paper default; ours was tuned locally. A third-party H3 pack also landed on 1.3 independently |
| threshold mode | `exact` | not exposed | our node has no `thresh_type`. Wan configs use `diag` |
| KV sink | full prefix | `exact_kv_and_rows` | **the same choice.** Both sink the whole prefix and run its query rows dense |
| dense steps | first 10 of 50 (20%) | `start_percent` 0.2 -> 5 of 16 (31%) | same idea, different spelling; ours is the larger fraction |
| dense blocks | **first two** | **none** | the clearest gap. `SOL_ARTIFACT_INSURANCE` exists here and is not shipped |
| token reordering | **none, stated explicitly** | `morton=False` | **agrees with us** |
| cache | FirstBlockCache 0.08 | none | we have no cache of any kind |
| steps | 50 | 16 | their baseline is 50-step Diffusers |

Two things stand out. **Our sink choice and our Morton choice both match their
validated policy**, arrived at independently. And **they run the first two
blocks dense where we run none**, which is the one place their tested recipe is
strictly more conservative than ours.

## Why they say H3 needs no reordering

From `models/minimax_h3/GB200/sol_attn_h3.py`, a rewrite against the released
kernel that records three things its own earlier version got wrong:

> **Morton reordering per attention call.** The released
> `install_wan_morton_forward` docstring says why not: only self-attention is
> order-sensitive, so the permutation belongs once at the block stack, and
> doing it per call cost more than the kernel saved. What the H3 integration
> then shows is that H3 needs *no* reordering at all -- the packed video tail
> is already a contiguous grid-ordered block, and the routing works on it
> directly.

**That reasoning does not obviously distinguish H3 from Wan**, and it is worth
saying so. "Already a contiguous grid-ordered block" is true of raster order,
which is exactly the layout Morton is applied to *fix* on Wan. Wan's 720p grid
is 45x80, so a 64-token raster block is under one row wide; H3 at 1344x768 is
24x42, so a block is about 1.5 rows. If anything H3's raster blocks are the
less degenerate of the two.

**And our own capture measurement disagrees with the assertion.** On real H3
q/k, Morton3D raises per-block centroid fidelity by 4.1% / 6.4% / 8.7% at
blocks 0 / 24 / 49 (`docs/morton.md`). That is not an output-quality result and
it does not make them wrong -- their claim may be "the benefit does not justify
the cost", which is a different statement, and they run Ulysses sequence
parallelism where a global permutation interacts with sharding in ways a single
GPU does not face. But the disagreement is real and it is the sharpest open
question this repo has on Morton.

## Morton in Sol-Engine, which does exist

Correcting a claim this repo made three times on 2026-08-15: **Sol-Engine does
ship Morton.** It is absent from the paper and from the H3 path, and present
and on by default for Wan.

- `techniques/sparse_backends/sol_attn_backend.py` has `_morton3d_perm`
  ("canonical x/y/z-interleaved") and `install_wan_morton_forward`.
- Every Wan config description reads "released Sol-Attn at tau 1.0 with
  **global Morton3D order** and dense guards".
- HunyuanVideo has it **off**: `HUNYUAN_SOL_MORTON = "0"`.
- H3 has it absent entirely.

**They ship only the 3D curve.** There is no `2d_frame` variant anywhere in
Sol-Engine; that is kijai's addition for H3's non-uniform temporal axis. Our
capture measurement found 3D scoring higher than `2d_frame` at every block,
which points the same way as their choice without confirming their reasoning.

`install_wan_morton_forward` is structurally identical to kijai's H3 port:
permute at the block-0 pre-hook, permute the rope, invert at the last block's
post-hook. kijai's version adds the video-span restriction and the
`(-video_start) % 64` rotation, neither of which Wan needs because Wan's whole
sequence is video.

### The ablation control worth stealing

`config/wan21_t2v_14b/reorder_only.toml`:

> "reorder-only control: global Morton3D order with every layer forced dense
> through the Sol-Attn adapter."

`WAN22_SOL_REORDER=1` with `WAN22_SOL_DENSE_LAYERS="0-39"`. Reordering on,
every layer dense. Since the permutation is output-neutral under dense
attention, this arm isolates **what the permutation costs** from what it buys,
and doubles as a check that it really is neutral. This repo has no equivalent
and it is cheap to build.

## FirstBlockCache

`models/minimax_h3/A100/first_block_cache.py`, ~30 lines of policy around a
threshold. Gated on `H3_FIRSTBLOCKCACHE=1`, default threshold `0.08` from
`H3_CACHE_THRESHOLD`, with decisions synchronized across ranks so a
multi-GPU run cannot diverge.

The idea, which is not ours and not new: run the first transformer block, look
at how much its output changed against the previous step, and if the change is
below the threshold, reuse the whole step's residual instead of running the
remaining blocks. It is a step-skipping cache, orthogonal to attention entirely.

**Worth trying here, and cheap in the sense that costs nothing to reason
about, expensive in that it needs building.** Notes for whoever picks it up:

- It is a **different quality tradeoff from sparse attention**, not a
  complementary one. Sparse attention approximates within a step; a cache skips
  steps outright. The `h3-turbo-eval` prior art measured Spectrum, a forecasting
  variant of the same family, at 0.789 frame correlation against stock, i.e.
  "closer to changing the seed than to accelerating the same render".
- Their threshold 0.08 is tuned at **50 steps**. We run 16. A cache that skips
  steps is far more aggressive when there are only 16 of them, and the
  documented failure mode of too-few-steps on H3 is prompt adherence collapse,
  not blur (`h3_config.py`: at 12 steps the third scripted shot never happens).
  So 0.08 should not be carried over.
- The rank synchronization is irrelevant on one GPU.
- ComfyUI already has TeaCache-style nodes in the wild; checking whether one
  works on H3 is cheaper than porting this.

## Other things in Sol-Engine we do not have

- **`thresh_type`.** `exact` for H3, `diag` for Wan. Our node exposes neither.
  Their H3 policy uses `exact`, which the docs describe as using the full
  covariance. We cannot currently select it.
- **`kv_splits`.** Split-KV execution, `auto` in the Wan configs.
- **Dense layer guards as a shipped default.** They run the first two H3 blocks
  dense. We have the mechanism (`dense_blocks`) and ship it empty.
- **An SM89 CuTe kernel.** PR #464, merged 2026-08-15, "a general BF16 SM89
  CuTe DSL Sol-Attn forward kernel using M64/N64 tiles, cp.async, and warp MMA",
  enabling RTX 4090 dispatch through the public API. This is a **second
  independent 4090 implementation** beside the comfy-kitchen `sol_attn` branch
  we build locally. Which is faster or more accurate on this card is unmeasured
  and is now answerable.
- **Token pruning and NVFP4 quantization** are two of Sol-Engine's five methods
  and are **not** applied to their H3 stack.

## Numbers, and why not to quote them against ours

Sol-Engine's published H3 speedups: 4.52x on RTX 5090, 3.95x on 8xGB200, 3.92x
on DGX Spark GB10, 3.56x on 4xH100, 3.55x on 4xA100.

**None of these is a Sol-Attn number.** "Full-opt" for H3 is context parallel +
kernel fusion + Sol-Attn + FirstBlockCache, against a 50-step Diffusers
baseline. Ours is Sol-Attn against sage at 16 steps on one 4090. Different
stack, different baseline, different hardware, different step count. This is
`docs/SOLATTN.md`'s unit trap at a larger scale, and the numbers are not
comparable in either direction.
