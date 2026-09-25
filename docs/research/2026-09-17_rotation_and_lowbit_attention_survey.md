# Rotation and low-bit attention for MiniMax H3 on Ada sm89 — research survey

> **Provenance, 2026-09-17.** Written by a research subagent (Claude Opus) on
> the owner's request, read-only, no GPU, from the two `coderef/` trees and
> the web. Reviewed by the session that commissioned it, which re-checked two
> of its code claims against the kitchen fork (the warp-shuffle Hadamard in
> `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/quant_qk_int8.cu` and the sign words it shares with `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh`) and
> found them correct; the rest is as the agent reported it, with its own
> verified / inference / unverified labels. Numbers quoted from papers and
> other repos are THEIR measurements on THEIR setups. Where it says "this
> repo's CHANGELOG" it means the sage fork's. The briefing it was given said
> sage's quantizer was CUDA; it is Triton, and section 0b corrects that.

Date: 2026-09-17. Read-only survey. Nothing was built, run, benchmarked, or
measured for this document; no GPU was touched. Every number below is either
(a) quoted from a paper, labelled with whose setup it was measured on, or
(b) quoted from a dated record already in this repo or a checkout, or
(c) an arithmetic estimate I derived from shapes, explicitly marked as such.

Target throughout: MiniMax H3, one RTX 4090 (Ada, sm89), head_dim 128, 50 DiT
blocks, 56 heads, one self-attention call per block over a packed
`[text | audio | video]` sequence of ~104k tokens, training-free.

Evidence labels used in every section:
- **[V]** verified from code I read or paper text I read
- **[I]** my inference / derivation (marked as such, never presented as a finding)
- **[?]** could not verify; says what was searched

---

## 0. What to do, ranked

Ranked for this project, not in general. Each item: payoff, cost to try, evidence.

**1. Add three arms to the rotation spike before writing any kernel.**
`tests/spikes/spike_h3_qk_rotation.py` already rotates captured q/k in PyTorch
and calls the shipped kernel, so a new rotation form is a change to one
function. In priority order:
(a) **kitchen's exact matrix** — `diag(sign_words) @ H128 / sqrt(128)`, using
the same four hard-coded sign words shared by
`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/quant_qk_int8.cu:90-108` and `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh:74-78`. The spike currently uses
its own draw (`SEED = 0x51A6E`); adopting kitchen's gives bit-level agreement
across sage, `int8_attention` and Sol's `rotate`, and the matrix is already
validated upstream.
(b) a RotateAttention-style **half rotation**: a Givens pair (c, c+8) inside
each 16-wide RoPE axis block, applied post-RoPE.
(c) a **targeted** block rotation built offline from `k_norm.weight` — a
Hadamard-8 over each loud channel plus seven quiet partners.
Run at least one block-0 capture alongside block 49 so the neutrality claim is
tested, not assumed.
*Payoff:* discriminates every rotation candidate in this report on real
activations at zero kernel cost, and settles the design input for the kernel
work. *Cost:* roughly ten lines in `rotation()` plus a GPU run.
*Evidence:* RotateAttention's Table 1 is the only head-to-head rotation
ablation in the literature and it does **not** favour the full Hadamard
(§A.6); their cheap half rotation is claimed at ~3% of full-rotation cost.
Our own block-49 channel data (CHANGELOG 2026-09-14) says a targeted form is
available because the loud channels are known from the weights alone.

**2. Do not design the kernel. The reference implementation is already in
comfy-kitchen, and sglang wrote the same thing independently.**
`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/quant_qk_int8.cu:135-156`
is a 128-wide Hadamard done as an in-lane H4 plus five `__shfl_xor_sync`
butterfly stages, one warp per 128-channel row, four contiguous channels per
lane, **no shared memory and no `__syncthreads` anywhere in the file**, with
the INT8 absmax recomputed over the rotated registers so rotation and scale
selection are one pass. sglang's DeepSeek-V4 indexer kernel
(`coderef/sglang/python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh:570-602`)
is the same design — one warp per (token, head), `kHeadDim=128`,
`kVecSize=4`, two in-register stages plus five shuffle stages — fused with
RoPE before it and FP8 quantization after it.
*Payoff:* the "make it nearly free" question is answered by existing code, and
adopting kitchen's exact matrix (the same four hard-coded sign words) buys
bit-level agreement with `int8_attention` and Sol's `rotate`.
*Cost:* reading two files. *Evidence:* §B.3, §B.4.

**3. Sol's `rotate` costs ~15% for an avoidable reason, and the fix is 200
lines away in the same repo.**
Sol's rotation uses `rot128_serial`
(`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh:81-95`),
in which **one thread owns a whole 128-float row in local memory** and runs
seven butterfly stages serially, called from the one-thread-per-token
`quant_q_rows` / `quant_k_rows` (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh:218-239`, `:299-320`). The
pooled and centroid paths use `rot128_block` (`:98-110`), 128 threads through
shared memory with two `__syncthreads` per stage. The warp-shuffle form in
`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/quant_qk_int8.cu` computes the identical matrix with neither.
*Payoff:* a known ~15% of the Sol call, with a mechanism and a named
alternative. *Cost:* restructuring those two producers from one-thread-per-token
to one-warp-per-token. *Evidence:* §B.4. This is a comfy-kitchen change, not a
sage one.

**4. Stop looking for a mergeable, zero-runtime-cost rotation. It provably
does not exist for H3's problem.**
The complete set of rotations that commute with H3's 3D RoPE is: an
independent 2x2 rotation inside each RoPE pair (i, i+48), times an arbitrary
O(32) on the un-roped dims 96..127 (§A.2, derived). Of those, only the
per-pair part has a free slot (the RoPE table build). H3's loud channels at
block 49 are 34, 82, 67, 19 — that is exactly the two *complete* pairs
(34,82) and (19,67), holding 93.2% of K energy. A rotation inside a pair
preserves that pair's total energy, so the one free rotation is provably
useless here. Any rotation that mixes across pairs must be applied after RoPE
and therefore costs runtime.
*Payoff:* removes a whole branch from the design space.
*Cost:* zero — already done. *Evidence:* §A.2 algebra, corroborated by
QuaRot ("Post-RoPE caching", R3 cannot merge) and RotateAttention Eq. 7;
channel indices from this repo's CHANGELOG entry dated 2026-09-14.

**5. Mixed precision by *block*, chosen from `k_norm.weight`, is the cheapest
accuracy lever on the board and needs no kernel.**
The checkpoint ranks all 50 blocks with no capture: top-4 channel energy
share 70% at block 49, 31% at 45, 25% at 48, 4-6% everywhere else
(this repo's CHANGELOG, 2026-09-14). Route those three blocks to the most
accurate kernel available (kitchen rotated INT8 dense, or bf16) and leave the
other 47 on the fast path.
*Payoff:* targets the three blocks that carry the error, at 3/50 of the cost
of a global precision change. *Cost:* consumer-side dispatch only.
*Evidence:* our own weights scan; RotateAttention keeps the first and last
two DiT blocks in FP16 as a static policy (§C.5); LightX2V ships
`dense_layers: [0]` for H3 (§D.2).

**6. On the dense/early steps: three shipped levers, and a step cache is worth
more than sparse attention on this exact GPU and model.**
In order of measured payoff: (a) a TeaCache-class step cache — the Sana
single-RTX-4090 record on H3 FL2VA at 1344x768/5s/50 steps attributes **3.18x**
to caching and **1.22x** incremental to Sol-Attn, in a controlled same-runtime
sequence (§D.5); (b) run the dense prefix on a quantized kernel — sglang
documents `dense_backend=sage_attn` as the "Sage then Sol hybrid" recipe and
LightX2V makes `"dense_backend": "sage_attn2"` the default in every shipped H3
config, which is where a better INT8 q/k path pays twice; (c) drop
`dense_layers` to 0 — sglang measured the 2-to-0 change as inside its
run-to-run noise floor and worth about 1% (§D.6). Keep the dense *step*
fraction near the 20-25% every shipped config converges on; sglang's sweep
found halving it from 10 to 5 of 50 halves cosine-versus-dense and visibly
re-frames the shot.
*Payoff:* directly on the dominant term, with one arm measured on our card.
*Cost:* config sweep plus the quality grade for (b) and (c); real work for (a).
*Evidence:* §D.5, §D.6.

**7. Value-path smoothing (VC-Attention) is the one 2026 idea that is both
sm89-implementable and measured on MiniMax H3.**
Per-block-of-128-rows mean subtraction of V after an online k-means
permutation of tokens, with the means restored through the row sums the
online softmax already keeps. On their B200 run at 1344x768 on MiniMax H3,
V-smoothing alone scored higher PSNR than the full method and than
SageAttention2 (§C.3). It is a V-path change, so it is orthogonal to
everything in §A.
*Payoff:* the largest H3-specific quality delta any 2026 paper reports.
*Cost:* real kernel work (a permutation plus block means plus an online-softmax
change). *Evidence:* §C.3. Not measured on Ada by anyone.

**8. Consider FP8 E4M3 for Q and K instead of INT8 on Ada.**
sm89 has FP8 tensor cores and sage already uses them for the PV product. E4M3
carries ~4 bits of *relative* precision on every channel regardless of
magnitude, which is exactly the failure mode a shared per-tile INT8 scale has
at block 49. VC-Attention's 8-bit configuration is per-channel E4M3 for Q, K
and V.
*Payoff:* could make the loud-channel problem structural rather than
patched. *Cost:* a new kernel variant — the largest item on this list.
*Evidence:* §C.4. No one in the surveyed literature reports FP8-QK on Ada;
treat throughput parity as unverified.

**9. The token axis of the scale tile is the other half of the error, and two
cheap checks live there.**
Sage's K scale covers 16 tokens x 128 channels and Q's covers 4 x 128
(verified in `quant_per_thread.py`; kitchen's `int8_attention` uses the same
4-and-16 split). A rotation flattens the channel axis; it does nothing about
one spiky token setting the scale for its 15 neighbours, which is where the
CHANGELOG says the remaining four-fifths of block-49 error lives. Two probes,
both cheap: (a) `tests/spikes/spike_h3_qk_quant_gran.py` already exists for
the granularity question; (b) the Jensen-bias correction from
`arXiv 2605.26266` is a per-key-group logit shift `~ (1/24d) * sum_j
Delta_{i,j}^2 ||q_j||^2` that does **not** cancel under softmax when the
per-group step sizes differ — testable in the existing CPU simulation with no
kernel. The published regime is INT2, so expect a small term at INT8.
*Payoff:* (a) potentially large, (b) probably small. *Cost:* low for both.
*Evidence:* §C.1, §C.6; this repo's CHANGELOG, 2026-09-14.

**10. Serving-stack engineering with a plausible payoff on one 24 GB card.**
In order: the H3 adaLN cache (a very wide fp32 projection removed entirely,
and mandatory under block offload in LightX2V); block double-buffering with a
contiguous pinned slab (2 resident blocks instead of 50); regional
`torch.compile` per repeated block with `mode="default"` — *not*
`reduce-overhead`, which corrupts modulation tensors under CUDA graphs.
*Evidence:* §E. Skip anything CFG-related: H3 ships CFG-distilled and both
serving stacks disable CFG outright.

---

## 0b. Four corrections to the briefing premises

1. **Sage's q/k quantizer is Triton, not CUDA.** `coderef/sage-fork/sageattention/core.py:1198`
   routes the shipped sm89 fp8++ path to `per_thread_int8_triton`;
   `grep -rn "balance" csrc/` returns nothing, and `coderef/sage-fork/sageattention/core.py:389` records that
   the CUDA `per_warp` path "measured slower and less accurate on sm89". So
   "a CUDA implementation inside sage's quantizer" is not a small change — it
   is either a quantizer port or a Triton butterfly. §B.7 says the Triton
   route is viable and names a working reference at the same width. **[V]**
2. **Sol's INT8 is comfy-kitchen's engineering choice, not the paper's
   design.** arXiv 2607.24027 specifies mean-pooled proxies with no
   quantization and does not state the exact stage's precision; the INT8
   Q/K/V/V^T and full-range P quantization are kitchen's
   (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn.cu:18-26`), and sglang documents a third lineage with an
   `int8_qk` kwarg the NVlabs API does not have. Cite the kernel, not the
   paper, for any Sol INT8 error term. **[V, §D.4]**
3. **kitchen's `int8_attention` rotation is not randomized at runtime.** The
   sign diagonal is four hard-coded compile-time 32-bit words with no RNG
   anywhere, identical for q and k (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/quant_qk_int8.cu:90-108`). A fixed draw,
   not a per-call randomization. **[V]**
4. **`draft_attention` / comfy-kitchen PR 179 is not present**, in either
   kitchen checkout, by commit-message search and content grep. **[V, §D.8]**

A refinement rather than a correction: the four loud channels at block 49 are
four *activation* channels arising from **two** weight channels (82 and 19),
with RoPE spreading each into its pair mate. That is what makes §A.3's
conclusion airtight — post-RoPE, both members of each loud pair are loud, so
the one mergeable rotation has nothing left to redistribute.

---

## A. Rotation for attention Q/K

### A.1 What the field actually uses, and what each one costs

| form | where used | runtime cost | mergeable offline? | evidence it beats the simpler option |
|---|---|---|---|---|
| Full randomized Hadamard on head_dim | QuaRot R3 (q/k, post-RoPE), VC-Attention (q/k, fused in quantizer), kitchen `int8_attention` | log2(d) butterfly stages per row; QuaRot reports <=7% forward overhead (their LLM setting, not ours) | **No** for q/k after RoPE | RotateAttention Table 1: best PSNR on Wan2.2-I2V but *not* best on cosine or on T2V (§A.6) |
| Block Hadamard (block < head_dim) | DuQuant (block rotation), our `spike_h3_qk_rotation.py` `block32` arm | log2(block) stages, fits in registers | No | untested on H3; DuQuant's evidence is LLM weight/activation quantization |
| Block rotation + channel permutation | DuQuant's zigzag permutation; our `block32+perm` arm | permutation is free if folded into the block's index math | permutation yes, rotation no | DuQuant's own ablation (LLM setting) |
| Randomized sign diagonal | prefix of every Hadamard scheme above, including our spike's `rotation()` | one multiply per element, or free if folded into `k_norm.weight` | **Yes** (diagonal; see A.4) | standard: prevents the worst case where a row aligns with a Hadamard row |
| Per-RoPE-pair 2x2 rotation ("interleaved") | RotateAttention's mergeable strategy | **zero** — folds into the RoPE table | **Yes** | RotateAttention Table 1: *worse than no rotation* on Wan2.2-I2V; better on T2V cosine |
| Cross-pair Givens pair ("half rotation") | RotateAttention's online strategy | claimed ~3% of a full rotation; 2 non-zeros per row makes it element-wise like RoPE itself | No | RotateAttention's recommended default, "due to its consistent performance" |
| Learned / Cayley rotations | SpinQuant | same as a fixed rotation at inference | yes where QuaRot's are | needs a training/optimization budget — out of scope here |
| Kronecker / affine transforms | FlatQuant; QuaRot's `H_{2^n} (x) H_m` for non-power-of-two dims | two small matmuls instead of one large | partially | not applicable: head_dim 128 is already a power of two |
| Per-channel scaling (SmoothQuant form) | sage's `qk_balance`; VC-Attention's channel-mean centering | one multiply, or **free** folded into `k_norm.weight` | **Yes** | our own measurement: -19.4% rel L2 at block 49, +0.8..+3.8% elsewhere (CHANGELOG 2026-09-14) |

**[V]** QuaRot rotation inventory and the R3 statement, read from
`arxiv.org/html/2404.00456v2` (HTML, method sections): R1 global input
rotation merged into weights by computational invariance; R2 online before
`W_down`, fused into `W_down`; **R3 a head-dimension Hadamard applied online
to queries and keys, after RoPE** — "Post-RoPE caching ... helps us to apply
a Hadamard transformation on a single token at each decoding step" — and it
cannot be merged because RoPE sits between the weight matrices and the
quantization point; R4 `(H_{n_h} (x) I)` online before `W_out`. Non-power-of-two
dims handled by `H_d = H_{2^n} (x) H_m`. Overhead quoted as "at most 7%"
measured on the `W_down` layer at sequence length 2048 — that is **their**
LLM setting and is not a measurement of anything on H3 or on Ada.

### A.2 The algebra: what can actually be folded in H3 [V for the code, I for the derivation]

H3's q/k path, from `comfy/ldm/minimax/model.py:173-200` (ComfyUI tree,
read 2026-09-17):

```
x -> qkv_proj (Linear, no bias) -> q_raw          # model.py:175
  -> RMSNorm over head_dim with per-channel weight w   # model.py:165-166, 193-194
  -> 3D RoPE, split-half, rot_dim = 96                 # model.py:183-189
  -> optimized_attention(..., mask=None)               # model.py:199
```

The RoPE geometry, from `comfy/ldm/minimax/model.py:523-530` and
`:149-155`: `inv_freq` has 16 entries; `per_axis = pos[S,3,1] * inv[1,1,16]`
gives `[S,3,16]`; the three axis blocks are concatenated to `[S,48]` and then
duplicated to `[S,96]`. So channel `c` in `[0,96)` carries axis `c // 16`
(0 = t, 1 = h, 2 = w) and frequency index `c % 16`, and the rotate-half
pairing is `(c, c+48)`. Dims 96..127 are untouched by RoPE. **[V]**

Write `R_rope(p)` for the block-diagonal orthogonal matrix that RoPE applies
at position `p`: a 2x2 rotation on each pair, identity on dims 96..127.

**Claim 1 (what a rotation must satisfy to sit before RoPE).** If `R` is
applied to both q and k *before* RoPE, the score becomes
`q'^T R^T R_rope(p_j - p_i) R k'`. This equals the original score for all
position pairs iff `R` commutes with every `R_rope(Δ)`. **[I]**

**Claim 2 (the commutant).** The matrices commuting with the whole family
`{R_rope(Δ)}` are exactly: an independent 2x2 rotation-and-scale inside each
RoPE pair `(i, i+48)`, times an arbitrary linear map on the un-roped block
96..127. Restricted to orthogonal `R`: an independent 2D rotation per pair,
times an arbitrary O(32) on dims 96..127. **[I]**

*Why, without needing to know `inv_freq`.* Mixing channels from two different
pairs would require the two pairs' angle functions to coincide identically
across all tokens. The t, h and w coordinates vary independently token to
token, so two pairs on different axes can never have equal angles for all
tokens, whatever the `inv_freq` values are; two pairs on the same axis would
need two exactly equal `inv_freq` entries. So no cross-pair mixing survives,
and mixing between a roped and an un-roped channel is excluded for the same
reason. Within a pair, and within the un-roped block, everything commutes.
This conclusion is therefore robust to the actual contents of the
`rope.inv_freq` buffer, which I did not read.

**Claim 3 (the free slots).** H3's q/k path has exactly two places that can
absorb a transform at zero runtime cost. **[I]**

1. **The diagonal**, in `q_norm.weight` / `k_norm.weight`. RMSNorm computes
   `w (*) q_raw * sqrt(D)/||q_raw||`; the normalizer depends on `q_raw`, which
   a change to `w` does not touch. So replacing `w` by `w (*) s` yields exactly
   `s (*) q_n`. For that diagonal to commute with RoPE it must satisfy
   `s[i] == s[i+48]` on dims 0..95; it is unconstrained on 96..127. This is
   precisely the slot the existing `MiniMaxH3ChannelBalance` node uses, and
   the algebra matches this repo's CHANGELOG entry of 2026-09-14. **[V for the
   slot's existence — it is already shipped; I for the derivation.]**
2. **The RoPE table**, built once per forward at model.py:746
   (`rope_rotation_table`, model.py:149-155). Composing each pair's 2x2
   matrix with a fixed extra rotation by `phi_i` is free: the table is shared
   by q and k, so the score is preserved. This is RotateAttention's
   "interleaved / mergeable" rotation, in the split-half rather than
   interleaved convention. **[I]**

**Claim 4 (nothing else folds).** A rotation `R` cannot be folded into
`qkv_proj`. Folding requires `A` with `A diag(w) = diag(w) R`, i.e.
`A = diag(w) R diag(w)^{-1}`; and `A` must be orthogonal, or it changes the
RMSNorm denominator `||q_raw||` and the model with it. `diag(w) R diag(w)^{-1}`
is orthogonal only when `R` is a signed permutation among channels of equal
`|w|`, or when `w` is constant on the block `R` acts on. Since the
non-uniformity of `w` *is* the problem here, essentially nothing folds. In
particular the O(32) on the un-roped dims — which Claim 2 permits — has **no**
free slot either. **[I]**

Contrast with QuaRot, where the RMSNorm sits *before* the projection, so its
diagonal folds forward into `W_k` (`W_k <- Q^T diag(alpha) W_k`) and the
rotation folds with it. H3's QK-norm sits *after* the projection with no
weight matrix following it, which is the structurally harder position. **[V]**

**Claim 5 (so any useful rotation for H3 is a runtime op).** Any rotation
that mixes across RoPE pairs must be applied after RoPE. There is no weight
between RoPE and the attention dot product. Therefore it is a kernel cost.
**[I]** This is the same conclusion QuaRot reached for R3 and RotateAttention
encoded in their split between a mergeable and an online rotation. **[V]**

### A.3 Does the one free rotation buy anything on H3? No.

From this repo's CHANGELOG entry dated 2026-09-14 (measured on captured
activations from the 2026-09-03 base16 t2v set, RTX 4090, build `069becf`):
at block 49 step 15 the top-4 K channels by energy are **34, 82, 67, 19**,
carrying **93.2%** of K energy; `34+48=82` and `19+48=67`, so those four
channels are exactly **two complete RoPE pairs**. **[V — quoted from the
repo's own dated record.]**

A 2D rotation inside a pair preserves that pair's total energy. Both channels
of both loud pairs are loud. Therefore the free per-pair rotation cannot
reduce the peak channel amplitude that sets the INT8 scale. **[I]** The same
record notes the *weights* carry the loudness in only two channels (82 and
19) and that RoPE spreads each into its mate — which is exactly why the
post-RoPE pair is uniformly loud and the pre-RoPE, pair-preserving fix cannot
reach it.

Mapping the loud channels onto the RoPE geometry of model.py:523-530:
channel 19 is axis 1 (h), frequency index 3; channel 34 is axis 2 (w),
frequency index 2. The temporal axis carries no loud channel at block 49.
**[I — arithmetic on the indices; the index-to-axis mapping is [V].]**

### A.4 What the free diagonal already buys, for comparison

Same record, `tests/spikes/spike_h3_k_channel_balance.py`, mean rel L2 over
8-head chunks against fp32 SDPA, sage fp8++ as served:

| cell | plain | pair-equal, alpha 0.5 |
|---|---|---|
| block 0, step 15 | 0.0085 | +3.5% |
| block 32, step 15 | 0.0293 | +0.8% |
| block 40, step 15 | 0.0426 | +1.8% |
| block 49, step 15 | 0.0472 | **-19.4%** |

**[V — this repo's CHANGELOG, 2026-09-14; RTX 4090, MiniMax H3 captures.]**
Note the sign: the free diagonal *costs* a few percent wherever no channel is
loud, which is why it ships gated. That cost is the bar a rotation has to
beat.

### A.5 Why a rotation might beat the diagonal — and the honest limit of that argument

A per-channel rebalance moves scale from K onto Q, which spends Q's own INT8
resolution; the repo's alpha sweep shows the undiluted version of that cost
(`pair a=1.0` at block 0: +135%). An orthogonal rotation spends nothing on
either side: it preserves each row's L2 norm exactly, and it needs no
statistics, no alpha and no gate, so it can be always-on. **[I]**

The limit: it is **not** true that a rotation can never increase a row's peak
channel. `max|x_c| >= ||x||/sqrt(128)` always, and on a row that is already
flat a Hadamard is roughly neutral, with a worst case of `sqrt(n)` if the row
happens to align with a Hadamard row. The randomized sign diagonal is what
makes the worst case improbable rather than impossible — which is exactly why
every scheme in §A.1 prefixes one, and why our spike's `rotation()` does too
(`SEED = 0x51A6E`). The honest claim is *neutral-to-positive with no tuning
knob*, and the block-0 arm of the spike is what tests the "neutral" half.
**[I]**

The scale in sage is per **tile**, not per row: Q's scale covers 4 rows x 128
channels and K's covers 16 rows x 128 channels
(`sageattention/triton/quant_per_thread.py`, verified below in §B). A
rotation flattens the channel axis of that tile; it does nothing about one
spiky token setting the scale for its 15 neighbours. Translate the 93.2%
concentration into recoverable bits with that in mind. **[V for the tile
shapes; I for the consequence.]**

### A.6 The one head-to-head ablation in the literature, and it is not a clean win

RotateAttention (arXiv 2607.02584, read as HTML `arxiv.org/html/2607.02584v2`,
method + ablation sections) Table 1, on Wan2.2-I2V:

| rotation | cosine | MSE | SSIM | PSNR |
|---|---|---|---|---|
| none | 0.9809 | 519.30 | 0.7752 | 22.828 |
| half (cross-pair Givens, online) | 0.9824 | 493.35 | 0.7767 | 22.903 |
| interleave (per-pair, mergeable) | 0.9792 | 583.88 | 0.7502 | 22.057 |
| full Hadamard | 0.9807 | 515.34 | 0.7808 | 23.176 |

**[V — their numbers, their models (Wan2.2-I2V/T2V, HunyuanVideo T2V, 480P,
81-129 frames), their hardware (kernel on A10, pipeline accuracy on H20), at
INT4. Not a measurement on H3 and not on Ada.]** On Wan2.2-T2V the ordering
changes again: interleave 0.9562 cosine, half 0.9492, none 0.9458. They
recommend half rotation as the default "due to its consistent performance",
and claim it costs ~3% of a full rotation because the combined
permutation-rotation matrix has two non-zeros per row and lowers to an
element-wise op like RoPE itself.

**Reconciling this against the recommendation in §0.** Their outlier
mechanism is different from ours. They report outliers partitioned by 3D-RoPE
segment with "low-frequency channel pairs accumulat[ing] larger angular
spread over long sequences and thus higher variance" — a *geometry* property
that grows with sequence length. H3's loud channels are a *learned weight*
property: `k_norm.weight` peaks, identical across every full H3 checkpoint on
the box, and far more extreme (93.2% of K energy in four channels, versus a
distribution spread over half of each segment in their analysis). A rotation
sized to spread four channels over 128 is a different problem from one sized
to flatten a half-segment. So their Table 1 is a genuine warning that the
full Hadamard is not automatically the answer — and it is the reason item 1
in §0 is *run the spike*, not *build the kernel*.

Their range-rectified INT4 softmax, for completeness:
`P~ = (P / max(P) * 15) - 8`, exploiting that `P = exp(QK - max)` is
non-negative so symmetric INT4 wastes half the codes. **[V]** Not directly
transferable — sage's P is FP8 E4M3, not INT4, and E4M3 has no equivalent
sign-half waste.

### A.7 What a targeted rotation would look like here [I — a design proposal, not a finding]

Because the loud channels are static, known from `k_norm.weight` with no
capture, and only two per block in weight space (four after RoPE), the
rotation does not have to be dense. A signed permutation composed with a
block-diagonal Hadamard-8, where each block contains one loud channel and
seven quiet partners, is orthogonal, exact-preserving, and touches 32 of 128
channels with 3 butterfly stages each — roughly one tenth of the work of a
full 128-wide butterfly.

Two caveats I want on the record before anyone builds it:
- The cost saving is probably irrelevant. The quantizer pass is memory-bound
  by a wide margin (§B.1), so the full butterfly's arithmetic is hidden
  anyway. If that holds, the targeted form's advantage over a full Hadamard is
  expressibility, not speed.
- It is a per-block rotation matrix, so it needs a per-block constant in the
  kernel or a small lookup — more plumbing than one global Hadamard.

The one thing it is unambiguously good for is the **spike**: it is the
cheapest way to find out how much of the available gain comes from spreading
four known channels versus from flattening everything.

---

## B. How well-optimized projects implement the Hadamard / rotation kernel

*(filled from the code survey; see §B.0 for the scope of what was searched)*

### B.1 The cost model for our case, first, because it reframes the question [I]

Per attention call at the shipped clip length: q and k are each
`[1, 56, ~104k, 128]`. That is `56 * 104000 * 128 ~ 7.5e8` elements per
tensor, `1.5e9` for both — about 3 GB of bf16 read and 0.75 GB of INT8
written by the existing quantizer pass. On a card with roughly 1 TB/s of
bandwidth that pass is memory-bound at the several-millisecond scale.

A 128-wide butterfly is `log2(128) = 7` stages of one add/sub per element:
about `1.0e10` add/sub for the pair of tensors. A 128x128 dense matmul form
is `2 * 128 = 256` flops per element, about `3.8e11` flops.

The arithmetic ratio is what matters: **the butterfly form is roughly an
order of magnitude below the memory-bound floor of a pass we already run,
and the matmul form is not.** So the design question is not "can we afford a
Hadamard" — it is "can the chosen home emit a butterfly without paying a
cross-lane data-layout conversion that turns it back into something
matmul-shaped".

These are my estimates from tensor shapes, not measurements. The bandwidth
and flop figures are the constants the estimate depends on; nothing here was
timed.

### B.2 The two candidate homes *for us*, characterized [V]

(A third kernel — comfy-kitchen's own `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/quant_qk_int8.cu` — already implements
the rotation we want, but it is kitchen's `int8_attention` quantizer, not a
home for sage's. It is the reference implementation, detailed in §B.4.)

**Home A — sage's quantizer.** `sageattention/triton/quant_per_thread.py`.
The shipped fp8++ path calls `per_thread_int8_triton(q, k, km, tensor_layout,
BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)` from `coderef/sage-fork/sageattention/core.py:1198`.
Inside:
- `quant_query_per_thread_int8_kernel` grid is
  `((qo_len + BLKQ-1)//BLKQ * (BLKQ//WARPQ) * 8, h_qo, b)`; each program owns
  a `[WARPQ//8, C] = [4, 128]` tile with a **single** INT8 scale
  (`scale = tl.max(tl.abs(x)) / 127.`).
- `quant_key_per_thread_int8_kernel` owns two strided `[BLKK//8, C] = [8,128]`
  half-tiles, i.e. 16 rows x 128 channels under one scale.
- `qk_balance` is already wired here as a per-`(batch, kv-head, channel)`
  multiply applied to the fp32 tile before the max — the exact point a
  rotation would go, one line later.
- The quantizer is a **separate pre-pass** that writes INT8 q/k and the scale
  arrays to global memory; the CUDA attention kernel consumes them. So a
  rotation fused here adds **zero** extra memory traffic.

*Risk:* the tiles are small (512 elements for Q) and Triton's tile layout on
the 128-wide contiguous axis is what decides whether a butterfly lowers to
register shuffles or to a full layout conversion. I could not determine this
from source reading and did not run anything. **[?]**

**Home B — comfy-kitchen's fused RMSNorm+RoPE kernel.**
`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/ops/rms_rope.cu`
(comfy-kitchen at `b532e28`, 2026-09-15). H3 calls it on the shipped path at
`comfy/ldm/minimax/model.py:185-189`
(`comfy.quant_ops.ck.rms_rope_split_half_`). Structure read from source:
- `rope_kernel` at `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/ops/rms_rope.cu:36-50`, `kWarpsPerBlock = 4`,
  `kThreads = 4 * 32` (`:33-34`).
- **One warp per (token, head) row**: `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/ops/rms_rope.cu:54-56`
  (`lane = threadIdx.x & 31`, `warp = threadIdx.x >> 5`,
  `row = blockIdx.x * kWarpsPerBlock + warp`).
- The RMS reduction already spans the full 128-wide row across the warp
  (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/ops/rms_rope.cu:79-88`, comment at `:94-95`: "the RMS reduction above always
  spans the full head_dim").
- The RoPE loop at `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/ops/rms_rope.cu:99-100` gives each lane
  `kPairsPerLane = 2` pairs (`:97`), vectorized as `rope::Pair` loads
  (`:104-122`), and the un-roped tail `96..127` is handled one channel per
  lane at `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/ops/rms_rope.cu:211`.
- So after RoPE the full 128-channel row is live in registers, distributed
  across 32 lanes of one warp, in a kernel that already reads and writes q
  and k (in place, for the H3 inference path).

*Why this is attractive:* a warp-level Hadamard-128 is the textbook case —
4 channels per lane means 2 stages purely in registers and 5 stages by
`__shfl_xor_sync`. It is already CUDA, already post-RoPE, and it sits
**upstream of all three attention kernels**, so one implementation serves
sage, kitchen `int8_attention` and Sol.
*Risk:* the current per-lane channel assignment is uneven (lanes 0..23 hold
five channels each, lanes 24..31 hold one), so a butterfly needs a re-layout
— a 4-row x 128-channel shared-memory staging buffer per warp is 2 KB, which
is nothing, but it is work. And the in-place variant writes bf16 back, so the
rotated values take a bf16 rounding that a rotation fused into sage's fp32
quantizer would not. The spike's `rerounding` arm already measures the size
of that penalty. **[V for the structure; I for the assessment.]**

### B.3 The four implementation designs found in the wild [V]

| design | where | mechanism | matrix materialized? |
|---|---|---|---|
| **warp-shuffle butterfly, in-register** | comfy-kitchen `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/quant_qk_int8.cu`; sglang's general FHT and its fused DSV4 indexer kernels | `__shfl_xor_sync` across lanes, 4 or 8 elements per thread; shared memory only for cross-warp stages | no |
| **tensor-core mma** | vLLM's hadacore | `mma.sync.aligned.m16n8k16`; the 16x16 Hadamard fragment built from packed sign bitmasks in registers | no |
| **materialized H + one GEMM** | vLLM TurboQuant; vLLM's INT4 KV Triton tier; compressed-tensors dense fallback; vllm-omni NPU | one `tl.dot` / cuBLAS GEMM against a cached D x D matrix | yes, D^2 bytes |
| **Triton reshape/split/join butterfly** | vLLM glm5next `kpool_compress.py` | `tl.trans` / `tl.split` / `tl.join`, seven hand-unrolled stages, no `tl.dot`, no H | no |

Both sides of the design argument are written down in vLLM itself.
TurboQuant argues for the GEMM —
`coderef/vllm/vllm/v1/attention/backends/turboquant_attn.py:107-109`: "single
cuBLAS GEMM instead of log2(D) butterfly kernel launches". glm5next argues
for the in-register butterfly —
`coderef/vllm/vllm/models/glm5next/common/attention.py:363-365`: "Fusing the
fp32 FWHT and quantization avoids an intermediate HBM round-trip and bf16
matrix-rounding bias." Our §B.1 estimate sides with the butterfly, because
the launch-count argument does not apply when the transform is fused into a
pass we already run.

### B.4 The reference implementations, in detail

**comfy-kitchen, `int8_attention`'s q/k quantizer — the closest thing to what
we would build, already in a repo we depend on.** All paths
`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/quant_qk_int8.cu`,
comfy-kitchen at `b532e28`, 2026-09-15:
- `convrot4()` at `:71-84` — an in-lane H4 over the four channels one lane
  owns, normalized by 1/2.
- `apply_convrot_sign128()` at `:90-108` — the random sign diagonal, as four
  **hard-coded compile-time 32-bit words**, applied by XOR-ing the IEEE sign
  bit rather than an FP multiply (comment at `:89`: "exact and avoids an FP
  multiply"). There is no RNG anywhere: the signs are not seeded per call,
  per head or per model. The comment at `:86-88` states the invariant we rely
  on: "Q and K use the same signs, so their exact dot product is unchanged."
- `convrot128_plain()` at `:135-151` — the in-lane H4 plus **five
  `__shfl_xor_sync` butterfly stages across 32 lanes**, scaled by
  `0.1767766952966369` (= 1/sqrt(32), which with `convrot4`'s x0.5 composes to
  1/sqrt(128)). `convrot128()` at `:153-156` is the sign diagonal then that.
- **Thread mapping: one warp per 128-channel row, four contiguous channels
  per lane** (`ch = tile*128 + (lane<<2)` at `:186`, `:249`, `:308`, `:333`),
  loaded one `float4` per lane. **No shared memory and no `__syncthreads`
  anywhere in the file**, stated at `:12`.
- **Fused with the scale:** a pre-rotation absmax is accumulated, then
  discarded and **recomputed over the rotated registers** (`:234-238` for Q,
  `:391-395` for K) before the warp reduction and the INT8 store. No rotated
  tensor ever reaches global memory.
- Size dispatch at `:811-853`: head_dim 64 gets H64 (half-warp) above
  `Lk > 256` and H4 below; head_dim 128 gets `convrot128` with signs; head_dim
  256 gets the sign-free variant applied to each of two 128-channel tiles. The
  source gives no stated reason for either carve-out.
- Alongside it, an **anchor-key centering** pass: `detect_k_anchor` at
  `:459-605` samples nine evenly spaced keys per (batch, kv-head), picks the
  one closest in L2 to the sample mean, and accepts it only if it lowers
  energy without raising the absmax by more than 12.5% (`:597-599`); the
  chosen key is subtracted from every key before absmax, before rotation and
  before quantization (`:344-346`, `:360-367`), which the header at `:286-288`
  notes is "exactly softmax-invariant". **Q is not centered.**
- Scale granularity is the same 4-and-16 split sage uses: `q_sc_per_h =
  q_oblk * 8`, `k_sc_per_h = k_oblk * 4` at `:738-739`.

**Provenance, which matters for what counts as upstream:** commit `48e4b28`
(2026-08-10, "Implement optimized int8 attention", #103) is in
`origin/main` and already contained `apply_convrot_sign128`, `convrot128` and
`detect_k_anchor`. So the `int8_attention` rotation and the anchor centering
are **upstream and unconditional**. The Sol `rotate` and `qk_balance` commits
(`bbd84c7`, `cebba8f`, `b532e28`, `b36ed9a`, all 2026-09-15) are **not** in
`origin/main`; they are local branch work, default off, CUDA-only.

**Sol's rotation, and why it costs what it costs.**
`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh`:
- `ROT_NORM = 0.08838834764831845f` (= 1/sqrt(128)) at `:73`; `rot_sign(d)` at
  `:74-78` uses **the same four sign words** as `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/quant_qk_int8.cu`.
- `rot128_serial(float* v)` at `:81-95` — **one thread owns a whole 128-float
  row in local memory**, seven butterfly stages, serial loop. Called from
  `quant_q_rows` (`:218-239`) and `quant_k_rows` (`:299-320`), which are
  one-thread-per-token.
- `rot128_block(float x, float* s)` at `:98-110` — 128 threads, one channel
  each, through shared memory, seven stages with **two `__syncthreads` per
  stage**. Used by the pooled and centroid producers
  (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_preprocess.cu:130`, `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_layout.cuh:273`, `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu:84`).
- Neither is the warp-shuffle form. The comment at `:97` confirms the two Sol
  variants compute the same matrix as each other; `:60-71` confirms it is the
  same matrix `int8_attention` applies.
- A deliberate carve-out worth knowing: the **routing threshold and the
  coarse/pooled branch are computed unrotated**, because the diagonal variance
  approximation is invariant under rescale but not under mixing
  (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_preprocess.cu:271-288`, `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn_token.cu:82-84`).

**sglang's DeepSeek-V4 indexer — the same design, arrived at independently,
and the fullest fusion chain in the survey.** sglang at `6c73368c32`,
2026-09-18:
- `coderef/sglang/python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh:493-622`
  (`fused_q_indexer_rope_hadamard_quant`), AOT twin at
  `coderef/sglang/python/sglang/kernels/aot/csrc/elementwise/dsv4_norm_rope.cu:436-520`.
- One warp per (token, head); `kHeadDim = 128`, `kVecSize = 4`, with a
  `static_assert(kHeadDim == kWarpThreads * kVecSize)` at `:497-501` — **32
  lanes x 4 elements, entirely in registers, zero shared memory**.
- Order: load (`:535-550`), **RoPE** (`:552-566`), **Hadamard** (`:570-602`,
  described in the source as "2 local stages + 5 cross-lane shfl_xor stages"),
  **FP8 quant and store** (`:604-621`). The `rsqrt(128)` scale is applied
  in-register at `:598-601`, i.e. **before** the quantization absmax.
- The butterfly itself, at `:591-597`:
  ```
  for (uint32_t mask = 1; mask < kWarpThreads; mask <<= 1) {
    for (int i = 0; i < kVecSize; ++i) {
      const float other = __shfl_xor_sync(0xFFFFFFFFu, data[i], mask, kWarpThreads);
      data[i] = (lane_id & mask) ? (other - data[i]) : (data[i] + other);
    }
  }
  ```
- The key-side twin,
  `coderef/sglang/python/sglang/kernels/jit/csrc/deepseek_v4/fused_norm_rope_v2.cuh:115-195`,
  does **RMSNorm, then RoPE, then Hadamard, then per-warp FP8 quant, then the
  paged cache write, in one kernel**. That is exactly the Home-B shape
  proposed in §B.2, shipping.
- And the invariant, stated in sglang's own comment at
  `coderef/sglang/python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh:572-574`: "V3.2 omits the rotation (kHadamard=false): it
  is logit-preserving (H orthonormal, applied to both q and k), so dropping it
  only trades fp8 quant accuracy."

**Triton can express the butterfly without `tl.dot` — this closes the main
open risk on Home A.** `coderef/vllm/vllm/models/glm5next/nvidia/ops/kpool_compress.py`:
- one butterfly stage at `:27-34`, pure `tl.reshape` / `tl.trans` / `tl.split`
  / `tl.join`:
  ```
  x3 = tl.reshape(x, (GROUPS, 2, STRIDE)); x3 = tl.trans(x3, 0, 2, 1)
  a, b = tl.split(x3); x3 = tl.join(a + b, a - b)
  x3 = tl.trans(x3, 0, 2, 1); return tl.reshape(x3, (128,))
  ```
- `_hadamard128` at `:37-45` — seven hand-unrolled stages
  `(64,1),(32,2),(16,4),(8,8),(4,16),(2,32),(1,64)`, then a hardcoded
  `* 0.08838834764831845`.
- Fused with FP8 quantization in `_fwht_quant_kernel` at `:69-101`
  (BLOCK_R=32, num_warps=2), and on the key side fused with softmax pooling
  and a paged cache write at `:139` onward.
- RoPE ordering verified in the caller:
  `coderef/vllm/vllm/models/glm5next/common/attention.py:347` applies
  `rotary_emb`, re-concatenates at `:355-357`, and only then calls
  `fwht128_quant_fp8(q)` at `:369`.

**VC-Attention, for contrast.** Read as HTML `arxiv.org/html/2609.15810v1`,
method plus Appendix B: **Q** "fuses RoPE, Hadamard rotation, and
quantization"; **K** "fuses the centering statistics, RoPE, the gather by pi,
Hadamard rotation, mean subtraction, and quantization"; **V** "fuses gather,
block demeaning, quantization". Cost claim: "The QK Hadamard is free in time"
(their Section 4.3). The paper does **not** state the Hadamard size, thread
mapping, register-versus-shared-memory strategy, or a measured overhead — it
says only that the passes are "hand-written in CuTe/CUDA rather than
compiler-generated". Hardware B200/B300/H200/RTX PRO 6000/RTX 5090; **no Ada**.

### B.5 Sizes, and where the 1/sqrt(n) lands [V]

**Non-power-of-two support exists in exactly two places in the whole survey**,
and neither matters at head_dim 128: sglang's Kronecker kernels
`12N / 20N / 28N / 40N`, whose base matrices are generated by Paley,
Williamson and Turyn constructions
(`coderef/sglang/python/sglang/kernels/jit/csrc/fast-hadamard-transform/code_gen.py:7,39,63,95`,
emitted into `fast_hadamard_transform_special.h:12,32,80,172`); and
compressed-tensors' `hadamards.safetensors` divisor table (66 sizes, max 256)
plus Sylvester doubling. Everything else is power-of-two only: hadacore
2^1..2^15; vLLM's INT4 path asserts `d & (d-1) == 0`; glm5next and sglang's
fused indexer hard-code 128. **128 is the universal special case** —
`kHeadDim=128`, `_hadamard128`, `_TRITON_HADAMARD_MAX_D=128`,
`transform_block_size=128`.

**A real hazard: the 1/sqrt(n) factor lands in four different places across
these stacks, and one of them actively undoes another's choice.** sglang's
general kernel applies a caller-supplied scale in the store epilogue, default
**1.0** — unnormalized unless the caller passes `hidden_size**-0.5`, which
`coderef/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py:205` does. sglang's fused kernels hardcode `rsqrt(kHeadDim)`
before the quant absmax. hadacore normalizes internally, and vLLM's INT4 path
then multiplies it back out (`rescale = d**0.5`,
`coderef/vllm/vllm/v1/attention/ops/int4_per_token_head.py:1090-1092`) because
its scale math is calibrated to the unnormalized convention, compensating with
two separate `/head_size` divisions (`:918`, `:948`). compressed-tensors and
TurboQuant fold it into stored data at load. If we adopt kitchen's matrix, the
factor is already split across `convrot4` (x0.5) and `convrot128_plain`
(1/sqrt(32)); do not add a third.

### B.6 Two sharp negatives [V]

- **flashinfer has no Hadamard at all.** Searched `hadamard`, `hadacore`,
  `walsh`, `quarot`, `spinquant`, `rotate_activation`, `rotation`, `butterfly`
  and `orthogonal` across `flashinfer/`, `include/` and `csrc/`; every hit is
  an unrelated bit-manipulation or logging use. It does ship fused
  RMSNorm+FP8-quant kernels (`csrc/norm.cu`), i.e. the infrastructure a
  Hadamard would slot into, with no Hadamard stage. Same result for the triton
  checkout: **zero** matches for `hadamard`, `walsh`, `quarot`, `spinquant`
  anywhere — every Triton Hadamard in this survey lives in a downstream repo.
- **llm-compressor declares R3 (the online q/k rotation) and does not
  implement it.** `coderef/llm-compressor/src/llmcompressor/modifiers/transform/spinquant/base.py:233-246`
  defines R3 targeting `q_attn` and `k_cache`, but the applying machinery in
  compressed-tensors raises `NotImplementedError` for those locations, and
  `coderef/llm-compressor/examples/transform/spinquant_example.py:14-15` says so plainly: "currently
  only rotations R1, R2, and R4 are available." So the LLM PTQ tooling has the
  *offline* rotations working and the *attention* rotation on paper only. That
  is the gap our work sits in.

### B.7 Is the INT8 per-group quantizer the right home? [I]

Yes, with a caveat, and the answer is now better supported than it was before
the code survey:

- The §B.1 risk that killed Home A — Triton not being able to emit a butterfly
  — is **answered**: vLLM's glm5next does exactly that, fused with FP8
  quantization, at the same width 128. So sage's Triton quantizer is a viable
  home and has a working reference to copy the stage structure from.
- Home B (kitchen's `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/ops/rms_rope.cu`) is what sglang independently arrived at for
  a production serving stack, and it serves three attention kernels instead of
  one. It costs a cross-repo change and a bf16 round trip.
- The overheads reported anywhere in these codebases are **essentially
  undocumented**. The only in-tree speedup number found is sglang's fused
  Hadamard + per-row quant + IMMA GEMM path reaching 2.49x BF16 where
  `torch._int_mm` reaches 0.46-0.90x
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/quantization/kitchen_int8.py:5-13`)
  — a different op on unnamed hardware, quoted because it is the only one that
  exists. Both sglang's `bench_hadamard.py` and vLLM's qutlass benchmarks are
  harnesses with **no committed results**.

What is not defensible is building either home before the spike in §0.1 says
which rotation form is worth building.

---

## C. Beyond rotation: low-bit attention on non-FP4 hardware in 2026

### C.1 Scale granularity

| granularity | where | needs a new kernel? | sm89? |
|---|---|---|---|
| per-tensor | baseline everywhere | no | yes |
| per-token (per-row) | RotateAttention's Q/K INT4 | yes | yes |
| per-thread-group tile | **sage as shipped**: Q 4 rows x 128 ch, K 16 rows x 128 ch, one scale each | already have it | yes |
| per-channel | VC-Attention's Q/K/V at 8 bits (E4M3) | yes | yes for FP8 |
| per-block-of-64 keys | kitchen `int8_attention` / Sol | already have it | yes |
| micro-scaled (per-16 with E4M3 scale) | VC-Attention's 4-bit path; SageAttention3 | yes, and FP4 hardware | **no** |

**[V for the sage row — read from `quant_per_thread.py`. [V] for the others
from the cited papers.]**

The relevant asymmetry for us: sage's K scale is the coarsest thing in the
pipeline (16 tokens x 128 channels under one number) and K's rounding is
where block 49's error lives — this repo's CPU decomposition puts
K-INT8-alone at 0.0645 versus Q-INT8-alone at 0.0159 at block 49, against
0.0031 / 0.0052 at block 0 (CHANGELOG 2026-09-14). **[V]** A rotation
addresses the 128-channel half of that tile. Nothing in §A addresses the
16-token half.

### C.2 Asymmetric / range-rectified quantization of the softmax output

RotateAttention's `P~ = (P/max(P))*15 - 8` for INT4 P. **[V]** Rationale:
`P >= 0`, so a symmetric signed format wastes half its codes. Applicability
to us: **none directly** — sage's P is FP8 E4M3, which has no unused sign
half in the same sense. The transferable idea is the general one: whenever a
tensor's sign is known, an asymmetric mapping recovers a bit.

VC-Attention's ExpCast-FP8 is the 2026 version of the same instinct applied
to the *cost* rather than the range:
`c(u) = round( 8(u+8) + 56 + beta )`, clipped to the range 0..120, with `beta = -0.35` and
`u = (s - m_i) log2(e) <= 0`, producing an E4M3 byte with a single FMA and
skipping both the FP32 exponential and the FP32-to-E4M3 cast. Their
Proposition 3.1 bounds the total variation at <3.64% for normal-range
entries. **[V — read from the HTML.]** Applicable on sm89 in principle (it is
bit manipulation on an FP8 code, not a tensor-core feature), but the
motivation is that on B200 the MUFU exponential is twice the cost of the FP8
tensor-core op — a ratio that is **not** established on Ada. **[I]** Their
own ablation is also a caution: on MiniMax H3 at 1344x768, V-smoothing alone
scored *higher* PSNR (21.0 dB) than V-smoothing plus ExpCast (20.2 dB).

### C.3 Value-path smoothing

VC-Attention's value smoothing, read from the HTML: online k-means over the
value tokens produces labels; sorting by label gives a permutation `pi`
applied to **K and V together** while queries stay in place, which preserves
the output because `P'V' = PV`. Values are then partitioned into hardware
blocks of `B_v = 128` rows; per block `mu_j = (1/B_v) V_j'^T 1` and
`R_j = V_j' - 1 mu_j^T`; only the residual is quantized, and the mean is
restored inside the online softmax through the outer product `r_ij mu_j^T`
where `r_ij` is the probability row sum the softmax already maintains. **[V]**

Cost, in their words: "grouping costs more than the stage it joins" — about
30% overhead on grouped steps — amortized by running grouping and demeaning
only on "the first 25% of the denoising steps" with 4-step reuse windows, for
an average of "3-4% of attention time". **[V — their measurement, their
GPUs.]**

Their MiniMax-H3 numbers, 8-bit, 1344x768, PSNR against a BF16
FlashAttention-4 reference, **on B200**: SageAttention2 19.9 dB,
VC-Attention (V-Smooth + ExpCast) 20.2 dB, VC-Attention (V-Smooth only)
21.0 dB. **[V — their numbers on their hardware. Not a measurement of our
fork, not on Ada, and their SageAttention2 arm is upstream's kernel, not
this fork's fp8++ path with `qk_balance`.]** Their speed claims against
SageAttention2 on B200 (they measure it slower than the BF16 baseline there)
say nothing about Ada, where FlashAttention has no FP8/FP4 tensor-core path
to fall back on.

Relation to what we already have: sage's `smooth_v` subtracts a per-**channel**
mean of V; VC's version subtracts a per-**block-of-128-tokens** mean after a
semantic permutation. Different axis, and the permutation is the expensive
part.

### C.4 FP8 instead of INT8 for Q and K on Ada [I, flagged]

VC-Attention's 8-bit configuration is per-channel E4M3 for Q, K and V.
**[V]** sm89 has FP8 tensor cores and sage already uses them for the PV
product, so the hardware path exists. The argument for it here is structural:
E4M3 carries roughly four bits of *relative* precision on every channel
independent of magnitude, whereas a shared INT8 tile scale gives the loud
channels ~7 bits and the quiet ones far fewer — which is the block-49 failure
mode exactly. The argument against: it is a new kernel variant (the largest
item in §0), and I found **no** source that reports FP8-QK throughput or
accuracy on Ada. Dense INT8 and FP8 tensor-core rates on Ada are
conventionally quoted as comparable, but I did not verify that from a
primary source and it is not a claim this report makes. **[?]**

### C.5 Mixed precision by layer and by step

- **RotateAttention** uses a *static, heuristic* policy, not a calibrated
  one: Wan2.2-I2V keeps "FP16 precision for the first 6 and last 4 sampling
  steps, as well as the first and last 2 DiT blocks" (~32% FP16); Wan2.2-T2V
  first and last 2 of each (~19%); HunyuanVideo-T2V first 4 and last 2 steps,
  first and last 2 blocks (~24%). No ablation on the step selection is given.
  **[V]**
- **VC-Attention** does the opposite: no per-layer precision change at all;
  uniform 8-bit or 4-bit, training-free and model-agnostic, with the *value
  grouping* — not the precision — restricted to the first 25% of steps. **[V]**
- **This repo** already has the better selection signal for H3: the
  checkpoint's `k_norm.weight` ranks all 50 blocks with no capture and no
  calibration (top-4 channel energy share 70% at block 49, 31% at 45, 25% at
  48, 4-6% elsewhere; CHANGELOG 2026-09-14). **[V]** That is a
  calibration-free, weights-only criterion of the kind neither paper has.

### C.6 Bias correction for quantized keys

"Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in
Video Diffusion" (arXiv 2605.26266, Tuncer, Becker, Pfeil; ICML 2026
workshops SCALE and F2S; read as HTML `arxiv.org/html/2605.26266v1`).

Mechanism: softmax's exponential is convex, so zero-mean quantization noise
on a key's logit *inflates* that key's contribution in expectation — the
"Jensen bias". The exact correction (their Eq. 12) is
`b_i = sum_c log( sinh(q_c Delta_{i,c} / (2 sqrt(d))) / (q_c Delta_{i,c} / (2 sqrt(d))) )`,
with the second-order form (Eq. 13)
`b_i ~ (1/(24 d)) sum_c q_c^2 Delta_{i,c}^2`, and for grouped quantization
`b_i ~ (1/(24 d)) sum_j Delta_{i,j}^2 ||q_j||^2`. It is subtracted from the
score before the exponential. They report ~5% end-to-end latency overhead on
MAGI-1 and zero extra storage. **[V]**

Their regime: INT2 (2.75 effective bits, group size 32) with INT4 ablations,
on MAGI-1 4.5B (NVIDIA L4), SkyReels-V2 1.3B and HY-WorldPlay 8B (A100).
Training-free. Headline: INT2 QuaRot+RTN on MAGI-1, PSNR 17.10 -> 22.97,
VBench 70.24 -> 78.02 against a BF16 baseline of 78.27. **[V — their numbers,
their models, their hardware.]**

**Applicability here, carefully.** The paper's framing requires *mixed*
quantization — quantized cached keys competing with an unquantized current
chunk — and H3 has no KV cache and quantizes every key. **[V — the paper
states the mixed-quantization condition.]** But Eq. 13 is written per key
group, and sage's K scale varies between 16-token groups, so the bias does
not cancel exactly under softmax normalization: groups with a larger step
size get relatively inflated. **[I]** At INT8, with `Delta ~ max/127`, the
term is small. Ranked in §0 as a near-zero-cost check in the existing CPU
simulation (`tests/spikes/spike_h3_block49_error_anatomy.py`), not as a
kernel change.

### C.7 Calibration-free versus calibrated

Everything ranked in §0 is calibration-free. The two 2026 video-attention
papers are both explicitly training-free (VC-Attention: "training-free" and
"model-agnostic"; RotateAttention: post-training). SpinQuant's learned
rotations and FlatQuant's learned affine transforms both need an
optimization budget and are out of scope under the stated no-training-budget
constraint. H3 has an unusually good calibration-free signal in
`k_norm.weight`, which is why §0 item 4 is cheap.

---

## D. Sparse and block-sparse video attention

*(filled from the code survey)*

### D.1 Sol-Attn, from the paper [V]

"Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention
Sparsification", Li, Li, Chen, Ye, Liu, Yu, Wang, Zhang, Xie, Xie, Han;
arXiv 2607.24027, submitted 2026-07-27; read as HTML
`arxiv.org/html/2607.24027v1` (method + experiments).

- Blocks: 64x64 for query and key/value.
- Routing signal: `s_ij = Qbar_i Kbar_j^T`, mean pooling along the token
  dimension. **The paper does not quantize the proxy** and does not describe
  an INT8 routing path.
- Threshold: per query block, `tau_i = mu_i + beta * sigma_i` where `mu_i`
  and `sigma_i` are the mean and standard deviation of that query block's
  proxy scores; a single standardized `beta` controls sparsity model-wide.
  No per-head or per-layer budget.
- Tail term: a Taylor expansion of the exponential around the row-wise mean
  score inside each unselected key block, keeping only the zeroth-order term,
  contributing to both the softmax numerator and denominator.
- Dense warm-up: dense "during the first 20% of denoising steps and in the
  first layer".
- Models/hardware: Wan2.1-14B, HunyuanVideo-13B, LTX 2.3-22B, Bernini-14B,
  Ideogram 4, on **H100**. 80-90% sparsity, "2.1x-3.0x end-to-end speedups",
  up to 5x with their Sol-Engine.
- The paper does **not** specify the precision of the exact stage and does
  **not** discuss composition with quantized attention kernels.

**Consequence for us [I]:** the INT8 centroids and INT8 exact stage in the
comfy-kitchen implementation are that project's engineering choices, not the
paper's design. Any reasoning about Sol's INT8 error term is reasoning about
kitchen's kernel, and should cite kitchen's source rather than 2607.24027.

### D.2 What actually runs on sm89 [V]

Most of the best-measured 2026 work is gated to Hopper or Blackwell and fails
at **startup**, not silently. The survivors on an Ada card, from the local
checkouts:

| method | arch gate, as written in the code | packed varlen? | quantizes inside? | training? |
|---|---|---|---|---|
| **Sol-Attn** (three lineages, §D.4) | sm80+; NVlabs upstream ships a `cute_sm89` kernel and lists "NVIDIA RTX 4090 / SM89 / CuTe DSL"; comfy-kitchen's `coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn.cu:25` says "sm_80+ (cp.async, INT8 MMA), tuned for sm_120" | yes, flat 64-blocks, `sink_start`/`sink_tokens` cover our 1550-token prefix exactly | kitchen lineage: yes, INT8 Q/K/V | training-free |
| **SpargeAttn** | `{(8,0),(8,6),(8,7),(8,9),(9,0)}` in sglang's platforms/cuda.py:208-218 | square self-attn >=128 only, else dense fallback | yes, INT8 QK / FP8 PV | training-free as wired |
| **SLA / SageSLA** | runs on Ada; TurboDiffusion ships 4090 instructions | yes | Sage variant: INT8 QK / FP8 PV | **needs fine-tuned weights** (`proj_l`); LightX2V's H3 SLA path is a distinct LoRA |
| **hybrid_window_attn_h3 (VDN-H3)** | `{(8,0),(8,6),(8,9),(9,0),...}`; on sm89 it runs FA3's Sm80 mainloop at FA2-class throughput | yes, built for H3's packed layout | no | **needs a 4.3 GB linear branch + an 8-step LoRA** |

Ruled out on this card or by geometry: **SubBlock** (`supported_capabilities
= {(9,0),(10,0),(12,0)}`, and sglang's own README says "On an unsupported GPU
it is not a fallback, it is an error at startup"), VSA-H3
(`{(9,0),(10,0),(10,3)}`), cube, SVG2, sliding-tile/STA, radial (Wan-only in
LightX2V), vmoba, and flashinfer's block-sparse kernels (sm100/sm120 only).
**SubBlock is the best-measured method in the survey and it cannot run on our
card.**

### D.3 Routing signals, compared [V]

- **Sol-Attn**: mean-pooled query-block centroid against pooled-key variance,
  threshold `tau_i = mu_i + beta*sigma_i`; unrouted blocks keep a zeroth-order
  Taylor tail term. 64-token blocks both sides, per-head.
- **SubBlock** (sglang): log-sum-exp over **sub-block pairs**,
  `score(i,j) = log sum_{a,b} exp(mean(Q_{i,a}) . mean(K_{j,b}) * scale)`, then
  top-k. Q64 x K64 with `n_q = n_k = 4` sub-blocks, i.e. 16-token pooling cells.
  Per-head selection, **uniform budget**.
- **VSA-H3**: top-k over pooled fp32 tile means, tiles = 4x4x4 video cubes.
- **SVG2**: k-means centroids of q and k plus a semantic permutation.
- **SpargeAttn**: mean-similarity plus CDF/top-k over smoothed K.
- **SLA**: top-k over mean-pooled blocks **plus a linear-attention branch**
  covering the complement — the one design that does not simply drop the tail.
- **Static patterns**: radial (window decays with frame distance), sliding
  tile, VDN's frame window plus first/last-frame anchors.
- **LLM-side (DeepSeek sparse attention, glm5next)**: a **learned** indexer —
  `wq_b` and `weights_proj` ship with the checkpoint — scoring individual
  tokens over a persistent KV cache in FP8/FP4. What transfers is only the
  *estimator shape* (cheap low-precision scoring, top-k, exact compute on
  survivors), which is the family Sol already belongs to; the trained
  projections and the causal decode setting do not.

### D.4 Sol-Attn is three different implementations [V]

They share the paper and the 64-block geometry and diverge in kernel
arithmetic. Which one runs is decided by which package imports.

| lineage | quantization inside the kernel | extra features |
|---|---|---|
| **NVlabs/Sana upstream** (`coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:9-14`) | bf16 API, per-arch CuTe kernels including `cute_sm89`; **no `int8_qk` kwarg** | `kv_splits` (H100 only), `sink_start`/`sink_tokens`, `thresh_type` diag/exact, MPS backend |
| **comfy-kitchen** (`coderef/comfy-kitchen/comfy_kitchen/backends/cuda/sage_attention/sol_attn.cu:18-26`) | **INT8 Q/K/V, INT8 V^T for PV, full-range P quantization in the exact branch** | token routing, `qk_balance`, `rotate`, `blk_cnt`, `sol_attn_chunked`, `coarse_gate`, `topk_ratio` |
| **"Wan2GP Ada port"** (referenced by sglang only) | exposes an `int8_qk` kwarg, enabled at `>= (8,9)` | — |

The split is documented in
`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/sol_attn.py:229-233`:
"Wan2GP Ada port: INT8-QK Triton; official NVlabs API has no int8_qk."
**So the INT8 in Sol's kernel on this box is comfy-kitchen's engineering
choice, not the paper's design** (§D.1 confirms the paper describes no
quantization of the proxy and does not specify the exact stage's precision).
Reasoning about "Sol's INT8 error term" is reasoning about kitchen's kernel.

### D.5 Reported numbers — read the sequence length before the speedup [V]

**Every H3 sparse-attention measurement in these checkouts is at
S ~ 37.7k-38.2k tokens, which is 1344x768 at 5 s / 124 frames. The shipped
clip length here is the 362-frame / 15-second configuration at ~104k tokens.**

The most relevant record is on our exact GPU:
`coderef/Sana/models/minimax_h3/RTX4090/README.md:11-40`, H3 FL2VA BF16,
1344x768 / 5 s / 50 steps, single RTX 4090, layerwise offload, one warm
sample — a controlled same-runtime attribution:

| arm | wall time | factor |
|---|---|---|
| baseline | 2239.22 s | — |
| lossless optimizations | 1951.27 s | 1.00x (reference) |
| + TeaCache | 613.71 s | **3.18x** |
| + TeaCache + Sol-Attn | 504.33 s | **1.22x incremental**, 3.87x cumulative |

**[V — Sana's numbers, on an RTX 4090, on MiniMax H3, at 5 s / 124 frames.
Not our clip length and not our software stack.]** Validation in the same
record: "First sparse forward: `[1, 38247, 56, 128]` at 19.65% effective block
density". The reading that matters: **on this card and this model, the step
cache was worth 2.6x more than the sparse attention.**

Sequence-length scaling, the best evidence for extrapolating to 104k —
`coderef/sglang/.../backends/subblock_sparse/README.md:169-172`: "The speedup
is bounded by sequence length, not by the method. The same config measured
1.13x at 37.7k tokens, 1.20x at 52k and 1.47x at 96k... Treat 1.2x as the
768p/5 s number, not the ceiling." **[V — sglang's numbers, on 8x B200, with a
kernel that cannot run on our card. Directional only.]**

Estimator quality, from the same README's router notes
(`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:24-46`): 567 samples of H3 DiT attention, mean
recall of retained softmax mass at 0.9 sparsity — `n_q=1,n_k=1` 0.6513;
`n_q=1,n_k=4` 0.6655; `n_q=8,n_k=8` 0.6793; oracle 0.7355.

### D.6 The dense warm-up question, which matters most here

**Every shipped mechanism is a fixed integer constant. No sigma rule exists
anywhere in these checkouts.** **[V]**

| repo | knob | value | source |
|---|---|---|---|
| sglang `sol_attn` | `dense_steps` / `dense_layers` | **10** of 50 / `"0,1"` | `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/sol_attn.py:79-80` |
| sglang SubBlock | `SKIP_FIRST_STEPS` / `SKIP_FIRST_LAYERS` | **10** of 50 / **0** | `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py:79-81` |
| sglang VSA-H3 | `vsa_dense_first_n_steps` | **0** (off) | `backends/video_sparse_attn_h3.py:188,197` |
| Sana RTX 4090 | `SOL_ATTN_FIRST_DENSE_STEPS` / `_LAYERS` | **10** of 50 / **2** | `coderef/Sana/config/minimax_h3/rtx4090_fullopt.toml:43-44` |
| Sana Sol-H3 (4-forward distilled) | `dense_steps` / `dense_layers` | **1** / 2 | `coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py:116-123` |
| LightX2V H3, 29-step | `dense_steps` / `dense_layers` | **6** of 29 / `[0]` | `configs/minimax_h3/minimax_h3_sol_block_offload.json` |
| LightX2V H3, 4-step DMD | `dense_steps` / `dense_layers` | **1** of 4 / `[0]` | `configs/minimax_h3/dmd/minimax_h3_bf16_4step_sol.json` |
| Sol paper | dense for the first 20% of steps plus the first layer | — | arXiv 2607.24027 |

10/50 = 20%, 6/29 = 21%, 1/4 = 25%, 20% in the paper. **Nobody documents a
rule, but every shipped config lands at 20-25% of the denoise steps run
dense.** If our step count differs, that fraction is the only portable
guidance these repos contain. **[V for the constants; I for the reading.]**

**The one measured justification for the constant 10**, from
`coderef/sglang/.../backends/subblock_sparse_attn.py:69-77`, swept on H3 t2va
1344x768 / 5 s / 50 steps / n_k=4 / sparsity 0.75: "Lowering the step cutoff
from 10 to 5 halves cosine-vs-dense (0.558 -> 0.310 on two clips) and visibly
re-frames the shot; going to 0 leaves the sample essentially uncorrelated with
dense for 1.20x -> 1.30x." **[V — sglang's sweep, their hardware, at 38k
tokens.]**

**The dense-*layers* knob is measured as nearly free to drop**, same file at
`:74-77`: "Lowering the layer cutoff from 2 to 0 costs 0.0013 of that cosine —
inside the 0.02 run-to-run noise floor — and is worth ~1%, so the first DiT
blocks get no special treatment." **[V]**

**Three shipped levers for making the early steps cheaper:**

1. **Swap the warm-up kernel to INT8.** sglang has a first-class
   `dense_backend` selector (`_DENSE_BACKENDS = {"fa", "sage_attn"}`,
   `coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/sol_attn.py:23`) and documents it as a named recipe in
   `coderef/sglang/docs/docs/sglang-diffusion/attention_backends.mdx:938-953`: "Set
   `dense_backend=sage_attn` to run that prefix on SageAttention and the tail
   on Sol sparse attention." LightX2V makes it the **default in every shipped
   H3 config**. One caveat to measure: sglang's packed-varlen Sage path loops
   over `cu_seqlens` segments **in Python** (sol_attn.py:189-202), one
   `sageattn` call per document — three launches per layer per dense step for
   our 3-segment sequence, which the single-call `fa` path does not pay.
2. **Cache the early steps away.** §D.5's 3.18x. Sana's TeaCache
   (`models/minimax_h3/RTX4090/teacache.py:36-47,126-160`) uses
   `threshold=0.10` on an accumulator of rescaled relative L1 between
   consecutive modulated inputs, `retain_steps=5` forcing the first five steps
   to compute and `cooldown_steps=1` forcing the last; measured 14 computes
   and 35 reuses over 49 decisions. Note it does **not** remove the warm-up —
   the first five steps still compute in full — it removes most of everything
   after.
3. **Fewer total steps, with the fraction held.** Both turbo repos resolve the
   conflict between distillation and dense warm-up by shrinking `dense_steps`
   proportionally rather than dropping it. Distilled weights are already on the
   box (`coderef/Minimax-H3-Turbo`, `coderef/alibaba-pai_MiniMax-H3-Acc-LoRAs`).

**A negative worth recording:** LightX2V's `warmup` / `warmup_steps` keys and
sglang's `server_args.warmup_steps` are **server and compile warmup, not dense
denoise steps**; do not count them. And no checkout chooses the dense-step
count adaptively — the only adaptive early-step machinery is TeaCache's L1
accumulator, which gates step skipping, not attention density. **[V]**

### D.7 Two levers nobody has exploited [V — quoted from sglang's router notes]

- **Per-head budget.** `coderef/sglang/.../backends/subblock_sparse/router.py:63-68`,
  filed under "Worth trying, not yet exposed": "A per-head budget beats any
  estimator upgrade measured here: at a fixed mean sparsity, spending more
  blocks on diffuse heads and fewer on peaked ones lifts 5th-percentile mass
  recall from .52 to .90. It needs a rule for setting the per-head split,
  which nothing in the pipeline currently produces." Every implementation
  surveyed selects per-head but spends a **uniform** budget per head. With 56
  heads and one call site, a cheap per-head statistic from routing scores we
  already compute would set that split. Orthogonal to the kernel.
- **A methodological warning that matches this repo's own testing rules.**
  Same file, `:48-61`: normalizing each query sub-block into a distribution
  measured *better* on both offline proxies (block-mass recall .6741 -> .6779,
  rel-L2 .2043 -> .1982, paired t = +8.0 / -5.7) and was **worse in the pixels
  on 0 of 15 prompts**, by 0.107 cosine (t = -6.4): "Single-layer output error,
  even measured directly, does not order these estimators the way 40 denoise
  steps through 50 layers do." If routing is tuned on the 104k sequence, budget
  for end-to-end renders; kernel-level L2 will mislead.

### D.8 A negative on `draft_attention` / comfy-kitchen PR 179 [V]

Searched for it and it is **not present**: `git log --all --oneline -i
--grep='draft'` and `--grep='anemoi'` both return empty on the comfy-kitchen
worktree and on comfy-kitchen-kijai, and a content grep for
`draft_attn|draft_attention|anemoi` across both worktrees returns empty.
Strengthening the negative rather than leaving it as an unfetched ref: local
branches include `pr184` and `sol-blk-cnt-pr`, and `upstream/*` was fetched as
recently as 2026-09-16, so PR-numbered branches do get fetched here.
comfy-kitchen's only sparse attention is Sol-Attn; `VSA`, `SLA`, `SpargeAttn`,
`radial`, `sliding_tile` and `STA` are all absent.

### D.9 The 2026 landscape [V — from the Awesome-Sparse-Attention-Video-Diffusion index]

*Corrected 2026-09-19 (caught by the reference-code session; verified against the papers in
[`2026-09-19_video_sparse_attention_2026.md`](2026-09-19_video_sparse_attention_2026.md)): this list credited DFSAttn
with layer-wise sparsity profiling and bidirectional co-clustering. That mechanism is SVOO's, the next entry (arXiv
2603.18636, "Attention Sparsity is Input-Stable"). DFSAttn (arXiv 2605.23445) is Hilbert reordering with mask caching.
HASTE below was retitled HEART in its second version.*

Training-free: DFSAttn (ICML 2026, Hilbert reordering and mask caching);
SVOO, Training-Free Sparse Attention via Offline
Layer-Wise Sparsity Profiling and Online Bidirectional Co-Clustering (ICML
2026); QuantSparse (ICLR 2026, **joint quantization + sparsification**);
Training-free and Adaptive Sparse Attention for Efficient Long Video
Generation (CVPR 2026); Sparse-vDiT (AAAI 2026); HyperVAttention
(spatio-temporal clustering); Sparse VideoGen2 (semantic-aware permutation);
LiteAttention (temporal sparsity); HASTE (head-wise adaptive).

Training-based: VSA (trainable, coarse tile pooling then fine token-level);
SLA (ICLR 2026, sparse + linear hybrid, fine-tunable); SLA2 (learnable
routing + QAT, **sparsity + quantization**); Veda (distilled); Light Forcing;
Improving Video Sparse Attention with Fine-grained Router and Sparse
Rebasing (ICML 2026); SALAD (~20.6 GPU hours of adaptation).

**Relevance filter for this project:** anything requiring training,
distillation or a fine-tune is out under the stated constraint. That removes
VSA, SLA, SLA2, Veda, SALAD and Light Forcing. Of the training-free set, the
two that share a mechanism with something already on the box are Sparse
VideoGen2 (semantic permutation — the same idea VC-Attention uses for its
value smoothing) and QuantSparse (the only one explicitly co-designing
quantization with sparsity, which is the regime the kitchen Sol kernel is
already in). I read only the index entries and abstracts for all of these.

---

## E. Transferable engineering patterns

Only items with a plausible payoff on one 24 GB Ada card running one
104k-token DiT. Every threshold and coefficient below is a **code constant**
from the cited checkout, not a measurement; none of these repos publishes a
4090/sm89 benchmark for H3.

**E.1 CFG: nothing to do.** `coderef/vllm-omni` states "MiniMax H3 only
supports cfg-distilled checkpoints"
(`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/denoise_loop.py:289`, vllm-omni at
`97912de8d`, 2026-09-18) and every LightX2V H3 config sets
`"enable_cfg": false`. Two independent serving stacks converged on CFG-free
H3. Spend nothing on CFG caching, CFG-zero-star, or batched-versus-sequential
CFG. **[V]**

**E.2 The adaLN cache is the biggest H3-specific memory idea in these
repos.** LightX2V hard-errors if `cpu_offload=true` without
`use_adaln_cache=true`
(`coderef/LightX2V/lightx2v/models/networks/minimax_h3/model.py:61-62`). With
it enabled, `time_embedder.linear_1/2` and `norm_out.linear` are never
constructed at all; their outputs are precomputed offline and loaded from
safetensors (weights/pre_weights.py:87-92, weights/post_weights.py:16-24).
The reason it pays: H3's adaLN projection is `18 * 5376 = 96768` outputs from
a 2688-wide timestep embedding
(`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py:113`),
a very wide fp32 matmul whose input depends only on the timestep — and the
timestep schedule is a fixed small set. **Constraint:** the cache is keyed by
bit-exact fp32 timesteps
(.../minimax_h3/adaln_cache.py:52-58), so changing `infer_steps` or the flow
shift invalidates it. **[V]**

**E.3 Block double-buffering plus a contiguous pinned slab.**
`coderef/LightX2V/lightx2v/common/offload/manager.py` keeps two device
buffers and prefetches the next block on a dedicated load stream (`:78-83`),
then swaps (`:102-112`), with version-aware stream priorities (`:17-27`). For
50 identical H3 blocks the resident weight cost is 2 blocks, not 50.
`coderef/LightX2V/lightx2v/common/offload/block_slab.py:111-120` packs each
block's tensors into one 16-byte-aligned pinned `uint8` buffer with named
views — one large H2D copy per block instead of dozens of small ones, which
is the transfer-efficiency and host-fragmentation fix in one. **[V]**

**E.4 Step caches: budget the VRAM before you add one.** TeaCache in
vllm-omni holds `previous_modulated_input` and `previous_residual`, both full
`[seq_len, hidden]` (`vllm_omni/diffusion/cache/teacache/state.py`), and the
slow path clones the hidden states as well (hook.py:165). At 104k tokens
and hidden 5376 in bf16 that is roughly a gigabyte per tensor — **[I,
arithmetic from the shapes, not a measurement]**. LightX2V's answer is to
evict the cached residual to CPU under the same flag as weight offload
(lightx2v/models/networks/wan/infer/feature_caching/transformer_infer.py:132-176),
pulling it back with an in-place `add_` on a hit.

Provenance warning on the H3 TeaCache constants: vllm-omni carries
MiniMax-H3 polynomial coefficients and an H3-specific `rel_l1` threshold of
0.17 (`coderef/vllm-omni/vllm_omni/diffusion/cache/teacache/config.py:86-99`) documented only as "MiniMax-H3 FL2VA
coefficients", with no fitting procedure — while a neighbouring model's entry
in the same file documents its calibration explicitly. Treat them as a
starting point, not a measurement. **[V]**

*2026-09-25: the LightX2V half of E.5 is stale. `8652c6f1` (#1557) makes
`MiniMaxH3Model._init_infer_class` accept DPCache and ships it combined with
Sol (`coderef/LightX2V/configs/minimax_h3/decache/`), so LightX2V now does
both at once. [`../wiki/references.md`](../wiki/references.md), "What moved by
2026-09-25".*

**E.5 A cross-repo contradiction worth knowing.** LightX2V **refuses** to run
any feature cache on H3
(`coderef/LightX2V/lightx2v/models/networks/minimax_h3/model.py:500-501` raises
`NotImplementedError`), and every shipped H3 config sets
`"feature_caching": "NoCaching"` — while shipping TeaCache/MagCache/
TaylorSeer/AdaCache/FirstBlock for other models. vllm-omni, by contrast, has
a working Cache-DiT path for H3 with `Fn_compute_blocks=1` (first-block
cache), `max_warmup_steps=4`, `residual_diff_threshold=0.04` and
`max_continuous_cached_steps=1` for its high-quality profile
(`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/quality_policy.py:24-33`) — six times
stricter than its own 0.24 default. So LightX2V ships sol_attn and
step-caching as **alternatives** for H3, never combined. **[V]** That is a
meaningful signal about the sparse-plus-cache interaction, though neither
repo states a reason.

**And a third position that settles it empirically:** Sana's single-RTX-4090
H3 record (§D.5) runs TeaCache **and** Sol-Attn together and reports the
incremental factor of each, so the combination is not merely allowed but
measured — on our card, on this model, at 5 s / 124 frames. Whatever LightX2V's
`NotImplementedError` is about, it is not evidence that the combination does
not work. **[V]**

**E.6 Regional compile, and the CUDA-graph trap.** vllm-omni compiles each
*repeated block* separately rather than the whole model
(`coderef/vllm-omni/vllm_omni/diffusion/compile.py:31-105`), building all compiled callables
before mutating anything so a failure leaves the model usable, and rewriting
hook references so a step-cache hook survives compilation. Separately,
`coderef/vllm-omni/vllm_omni/diffusion/models/dreamzero/pipeline_dreamzero.py:671-675` records
the precise reason `mode="reduce-overhead"` is unsafe on DiT blocks: CUDA
graphs reuse a static output buffer and the modulation tensors get
overwritten across steps. Encoders are safe; DiT blocks are not. **[V]**

**E.7 Clean negative: no serving stack in these checkouts pins GPU clocks.**
Searched vllm, sglang and LightX2V for `lock.?gpu.?clock`, `-lgc`,
`persistence_mode` and in-process `nvidia-smi` invocation; every hit is
environment collection, build-time CUDA detection, or CI device visibility.
**[V — an empty search, reported as such.]**

---

## Bibliography, with access notes

| work | id / URL | what I read |
|---|---|---|
| RotateAttention: RoPE-Aware Rotation and Range Rectification for INT4 Quantized Attention in Video Generation. Liu, Lan, Li, Yuan, Yang. Submitted 2026-07-01, revised 2026-07-11. | arxiv.org/abs/2607.02584 ; arxiv.org/html/2607.02584v2 | **HTML full text**, method + ablation + mixed-precision sections |
| VC-Attention: Value Smoothing and Softmax Casting for Low-bit Attention. Li, Zou, Zhang, Chen, Xi, Zhang, Zhu, Han, Zhang, Lin, Li. Submitted 2026-09-14. | arxiv.org/abs/2609.15810 ; arxiv.org/html/2609.15810v1 | **HTML full text**, method + experiments + Appendix B |
| Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification. Li, Li, Chen, Ye, Liu, Yu, Wang, Zhang, Xie, Xie, Han. Submitted 2026-07-27. | arxiv.org/abs/2607.24027 ; arxiv.org/html/2607.24027v1 | **HTML full text**, method + experiments |
| Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion. Tuncer, Becker, Pfeil. ICML 2026 workshops SCALE and F2S. | arxiv.org/abs/2605.26266 ; arxiv.org/html/2605.26266v1 | **HTML full text**, method + experiments |
| QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. | arxiv.org/abs/2404.00456 ; arxiv.org/html/2404.00456v2 | **HTML full text**, rotation inventory + R3 + norm folding |
| SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization. | arxiv.org/abs/2411.10958 | **abstract and search-result synthesis only** |
| SageAttention3 (microscaling FP4, NeurIPS 2025 Spotlight), SageAttention2++ | thu-ml/SageAttention repo | **abstract / repo README level only** |
| DuQuant (block rotation + zigzag permutation, NeurIPS 2024) | — | **not fetched this session**; referenced from prior knowledge and from the design comment in `tests/spikes/spike_h3_qk_rotation.py` |
| SpinQuant (learned Cayley rotations), FlatQuant (Kronecker affine transforms), ConvRot (arXiv 2512.03673) | — | **not fetched this session**; listed for completeness, not cited for any claim |
| Awesome-Sparse-Attention-Video-Diffusion index | github.com/Mutual-Luo/Awesome-Sparse-Attention-Video-Diffusion | **index page only**; entry titles, venues and one-line mechanisms |
| VSA (2505.13389), SLA (2509.24006), SALAD (2601.16515), Light Forcing (2602.04789), Sparse-vDiT (2506.03065), Sparse VideoGen2 (2505.18875), LiteAttention (2511.11062), HASTE (2605.14513), DFSAttn (2605.23445), QuantSparse (2509.23681), HyperVAttention (2607.03012), PISA (2602.01077) | arXiv ids as listed | **search-result titles and abstracts only**; none read in full |

---

## Checkouts, commits and dates

Recorded 2026-09-17. All read-only; nothing was modified or built.

| checkout | HEAD | date |
|---|---|---|
| comfy-kitchen | b532e28 | 2026-09-15 |
| comfy-kitchen-kijai | 402fc90 | 2026-09-01 |
| ComfyUI-H3-Quant | dfb49a2 | 2026-09-15 |
| ComfyUI-UtilsCollection | 09160b2 | 2026-09-17 |
| DiffSynth-Studio | c458cb4 | 2026-09-14 |
| diffusers | 7221eef45 | 2026-09-18 |
| flashinfer | 7c194873 | 2026-09-17 |
| h3-the-transformation-engine | 0ad30a8 | 2026-09-10 |
| LightX2V | 69018c92 | 2026-09-17 |
| llm-compressor | a3109f85d | 2026-09-15 |
| MiniMax-H3 | d21241f | 2026-08-15 |
| Minimax-H3-Turbo | 02e26d5 | 2026-08-27 |
| Sana | bb60499 | 2026-09-16 |
| sage-fork (coderef copy) | 9adb47d | 2026-09-15 |
| sglang | 6c73368c32 | 2026-09-18 |
| TaoMate-H3 | b933d8e | 2026-09-17 |
| TaoMate-LTX | 136d890 | 2026-09-08 |
| transformers | 6c6bac29f5 | 2026-09-17 |
| triton | b63e34c521 | 2026-09-17 |
| TurboDiffusion | e3d6136 | 2026-08-27 |
| vllm | a9a7e45f31 | 2026-09-17 |
| vllm-omni | 97912de8d | 2026-09-18 |
| flash-attention | 98eb7998 | 2026-09-11 |
| sageattention-autotune | cde5993 | 2026-08-09 |
| comfyui_dagthomas | 469fdd6 | 2026-08-30 |
| ComfyUI-KJNodes | d3cfe21 | 2026-09-13 |
| ComfyUI-AudioLoopHelper | ad5a974 | 2026-07-05 |
| ComfyUI-h3-explorations | 140dbf3 | 2026-09-17 |
| alibaba-pai_MiniMax-H3-Acc-LoRAs | not a git checkout | — |

Two checkouts cited above sit outside both `coderef/` trees and are named
here so the citations resolve: `compressed-tensors` at `c649159` (2025-09-12),
which is where llm-compressor's transform math and the R3
`NotImplementedError` actually live; and ComfyUI itself, whose H3 model
definition is cited ComfyUI-relative as `comfy/ldm/minimax/model.py`.

Working tree of this repo at the time of the survey: `main` at `9adb47d`,
with `tests/spikes/spike_h3_qk_rotation.py` untracked. Sana is on branch
`sol-engine`, comfy-kitchen on `h3-build` (its `origin/main` at `21003fa`,
2026-09-07), comfy-kitchen-kijai on `sol_attn_continued`.
