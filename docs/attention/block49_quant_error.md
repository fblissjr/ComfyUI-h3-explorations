# Block 49: why unrotated INT8 attention loses accuracy on H3's last blocks

**The answer.** Block 49's `k_norm.weight` puts an order of magnitude more gain on a few channels than on the rest, so
after RMSNorm four K channels carry about 93% of the block's K energy. An INT8 kernel that quantizes K with one scale
across all 128 channels (sage per 64-token block, Sol per key row) lets those four set the scale, and the other 124
keep about three bits. Block 49's attention is also the peakiest in the model, so a small relative error in K becomes
a large logit error and the softmax flips. The product is a block whose INT8 error is five times block 0's, almost all
of it on the K side. Blocks 45 and 48 carry the same defect more weakly.

**What ships because of it.** A Hadamard rotation of q and k before INT8 removes most of it: kitchen's `int8_attention`
has always done this and was never affected, and Sol's own `quantizer="rotated"` does it on Sol's steps. The default
chain since 2026-09-27 is rotated INT8 everywhere ([`quant_policy.md`](quant_policy.md)). The per-channel rebalancing
levers built first (sections 4 and 8) remain as options. bf16 attention at 45/48/49 is the best arm measured and keeps
a small edge over every INT8 lever on the sage chain (section 6), not re-judged since rotation shipped.

Model throughout: **MiniMax H3, the pruned int8 convrot fl2va checkpoint** (`h3_config.MODELS["unet_fl2va"]`). The
weights finding holds for every H3 DiT checkpoint on disk: the loud channels are identical across all fourteen full
DiT files here, and the turbo, SLA and PDD LoRAs carry no norm weights. Capture: the 2026-09-03 base16 t2v set at
1344x768, S=104,361 (395 text, 1,150 audio, 102,816 video rows), kept to 2026-10-31.

## What "error" means here

Not the video. For one block and one step, run the captured q, k, v through the INT8 kernel and through exact fp32
attention, and divide the size of the difference by the size of the exact output: relative L2. 0.01 means the
kernel's output is off by one percent of its own magnitude. `h3_capture.py` saves the q, k, v a kernel received during
a real render (about 4.5 GB a cell at 345 frames; the set covers blocks 0, 24, 32, 40 and 49 of one t2v render), and
the grade scripts print every table below. A change visible in a clip cannot be traced to a block or kernel by looking
at the clip; this can. The clips come back at the end, to ask whether a difference that size shows.

| what | how it is measured | referent | tool | needs |
|---|---|---|---|---|
| which blocks have loud K channels | energy share of the four largest `k_norm.weight` entries per block | none: a property of the weights | `bench/check_channel_balance.py` | the checkpoint, no GPU |
| whether the activations agree | per-head channel energy of captured q/k after norm and RoPE | none | `bench/analyze_head_magnitudes.py` | a capture |
| how much each INT8 kernel loses | relative L2 against fp32 attention on the same bf16 q/k/v | exact attention on identical inputs | `bench/grade_channel_balance.py`, `bench/grade_ck_int8_on_capture.py`, `bench/analyze_sol_error.py` | a capture and the kernels |
| what a fix recovers | the same, with the fix on | as above | the same | the same |
| that a fix is exact for the math | the fp32 eager reference with and without the rescale | the reference against itself | kitchen `tests/test_sol_attn.py -k qk_balance`, sage `tests/test_qk_balance.py` | any CUDA card |
| what it costs | wall time from the server's own history | the unmodified render on the same seed | server history | a render |
| whether people can see it | labelled originals in a fixed order, one question, words filed verbatim | none: opinion | `bench/results/2026-09-15_block49_*` | eyes |

## Who it reaches

Unrotated INT8 attention with a shared per-row or per-block K scale: stock comfy-kitchen `sol_attn` (Comfy-Org's and
kijai's, which core's block-sparse node calls), SageAttention and every fork in every mode including the fp16 one (all
quantize QK to INT8), and by the same mechanism NVLabs' INT8 Sol kernels (unmeasured here). **Not** kitchen's
`int8_attention`, the kernel behind core's Model Attention Backend on "comfy kitchen attention" and
`--use-ck-attention`: it rotates q/k with a block-Hadamard first, and on the block-49 capture its error is a third of
Sol's and of sage fp8++'s with nothing done; the weights fold moves it about one percent
([`data/results/2026-09-15_ck_int8_attention_block49.json`](../../data/results/2026-09-15_ck_int8_attention_block49.json)).
ComfyUI's default (pytorch SDPA in bf16, read from `comfy/ldm/modules/attention.py` on 2026-09-15) has no scale to
share. So the population is everyone on `--use-sage-attention` or the block-sparse node, which is many of the people
rendering long clips and none of the people on defaults.

It is a quality effect, not a correctness one: renders complete and are plausible, and the error lands on the last
block's sharp read of the text rows (section 5).

An outside runtime treats the tail as sensitive too: TaoMate-H3 (`coderef/TaoMate-H3`, deprecated as of 2026-09-27,
history only) keeps blocks 0, 1, 47, 48 and 49 in bf16 on its W8A8 linear path and quantizes only `qkv_proj` and
`fc1` of the interior. That is a different surface (attention there is bf16 FA3) and corroborates the tail's
sensitivity, not this cause; its set and the lopsided set (45, 48, 49) overlap on 48 and 49 only.

## 1. It is the weights, and it is not only block 49

Ranking all fifty blocks by the energy share of the four loudest `k_norm.weight` channels
(`bench/check_channel_balance.py`, no card):

| block | top-4 K-norm energy share | max / median weight |
|---|---|---|
| 49 | 70% | 16x |
| 45 | 31% | 7x |
| 48 | 25% | 6x |
| every other block | 4-6% | 1.0-1.6x |

`q_norm.weight` peaks at the same channels at the same blocks (69% at 49). The weight peaks at block 49 sit at channels
82 and 19; H3's RoPE is split-half over channels 0-95 and rotates (i, i+48) together, so each peak spreads into its
mate and the activations show the pairs 34/82 and 19/67. That is the four-channel set the 2026-08-20 head-magnitude
analysis found ([`../state/roadmap.md`](../state/roadmap.md), "block 49 attributed, at the input level";
`bench/results/2026-08-20_head_magnitudes*.json`), and the sage fork's capture reproduced it exactly (share 93.2% at
block 49, 5-11% at blocks 0, 32, 40). The checkpoint scan and attention-target record is
[`data/results/2026-09-14_block49_checkpoint_scan_and_targets.txt`](../../data/results/2026-09-14_block49_checkpoint_scan_and_targets.txt).

**Blocks 45 and 48, captured 2026-10-02.** Full 50-block telemetry on Ref2VA (S = 119k) under `quantizer="rotated"`:
block 48's error is 7.14% (max 9.41%) and block 45's 9.55%, against 13.73% at block 49, so rotation resolved the
outlier defect at 45 and 48 and left them cleaner than the interior, where routing error peaks at blocks 38 to 43
([`../research/sparse/sol_dense_blocks_reanalysis.md`](../research/sparse/sol_dense_blocks_reanalysis.md)).

## 2. It is K's rounding, not Q's, not the PV side

A CPU decomposition on the captured block-49 and block-0 cells: fp32 exact attention on 1,024 sampled query rows
(256 text, 256 audio, 512 video) over all keys and all 56 heads, with each of sage's quantization steps simulated
alone (`tests/spikes/spike_h3_block49_error_anatomy.py` in the sage fork; K per 64-token block, Q per token):

| arm | block 49 | block 0 | 49 / 0 |
|---|---|---|---|
| K int8 alone | 0.0645 | 0.0031 | 21x |
| Q int8 alone | 0.0159 | 0.0052 | 3x |
| Q and K int8 | 0.0666 | 0.0062 | 11x |

Q's error is small because its scale is per token and no channel dominates a token the way the loud channels dominate
a 64-token block of K. The fp8 PV side adds about a seventh on top (fp8++ 0.0472 against the fp16 kernel's 0.0409 at
block 49, `spike_h3_real_activations.py`): real, and not the mechanism.

## 3. It is amplified by how block 49 attends

The exact reference's own shape, median over heads and sampled rows:

| | row max p | effective keys | logit range (max minus mean) |
|---|---|---|---|
| block 0, text queries | 0.002 | 25,602 | 7.0 |
| block 0, video queries | 0.003 | 10,653 | 8.9 |
| block 49, text queries | 0.076 | 340 | 15.6 |
| block 49, video queries | 0.095 | 165 | 17.5 |
| block 49, worst-4 heads | 0.453 | 5 | 145.5 |

An INT8 error in K is a relative error in the logit; the same relative error on a logit of 145 moves the softmax by
orders of magnitude more than on a logit of 7. The four worst heads by K-rounding error are 17, 9, 11 and 22, each five
to seven times the block's median. Head 11 is the head that loses under token routing at every step
([`../research/2026-09-04_sol_token_aug_grade.md`](../research/2026-09-04_sol_token_aug_grade.md)); a head attending to
five keys over a hundred-logit range is a candidate mechanism, not measured. Text query rows carry the most error at
block 49 (0.11 against 0.04 for video rows).

## 4. What can be done: the rescale, measured

`q . k == (q * f) . (k / f)` for any per-(head, channel) factor, so K's loud channels can be rebalanced against Q
before quantization at no cost to the math. Sol's routing threshold is invariant under the same rescale in exact
arithmetic (`c_d` scales by `s_d`, `kcvar_d` by `1/s_d^2`), so it changes which keys are quantized coarsely, not which
blocks Sol routes. With `f = rms_k^a / rms_q^(1-a)` (SmoothQuant's form), sage fp8++ mean rtol against fp32:

| factor form | block 49 | block 0 | where it lives |
|---|---|---|---|
| none | 0.0472 | 0.0085 | |
| per head, from the capture, RoPE-pair-equal (a=0.5) | 0.0381 (-19%) | 0.0088 (+3.5%) | superseded by the in-quantizer form (section 8) |
| per channel from the checkpoint's norm weights, pair-equal (a=0.5), CPU simulation of the QK side | 0.0587 against 0.0666 plain (-12%) | 0.0062 against 0.0062 | `MiniMaxH3ChannelBalance`, folded into the norm weights |

Alpha 0.5 is the optimum on both captured block-49 steps (0.35 and 0.65 give -12% to -18%; 0.15 hurts; a=1.0 is bad
everywhere because Q then carries the whole scale). The per-channel form gets about half the per-head gain because a
[128] weight cannot express per-head strength; it is exactly neutral at a flat block. Because the loud channels are two
RoPE pairs, a factor equal within each pair commutes with RoPE and folds into `q_norm.weight` and `k_norm.weight` at
load for zero runtime cost; the pair-equal form measured within a point of the free form (-19.4% against -20.4%).

**Both kernels share the mechanism, at different strengths.** Sol quantizes K per key row after subtracting the
per-channel key mean, with one INT8 scale across the row's 128 channels (`sol_layout.cuh`, `quant_k_rows`); sage
quantizes K per 64-token block. Sol's is finer and mean-centred, so its penalty is smaller, but a row's scale is still
set by its loudest channel.

| lever | where | reaches | block 49 INT8 term (Sol's total error moves about a third as much, since the routing term stays) | cost |
|---|---|---|---|---|
| `MiniMaxH3ChannelBalance` (`channel_balance.py`; combo `balance` off by default, "loud blocks (from weights)" or "named blocks") | per-channel factor from the norm weights, folded at load, head-shared, RoPE-pair-equal | any INT8 kernel | Sol -13%, sage -7% (8 heads) | none at render time |
| `qk_balance` in the sage fork (v0.7.19; the sage node's `fp8++ balanced`) | per-head factor from per-call channel norms, inside the per-thread quantizer, gated per head | sage steps | sage -23% (all heads) | +0.7% a call, no memory |
| `qk_balance` in Sol (kitchen fork; `quantizer="balanced"`) | the same per-head factor in Sol's preprocess, in its pooled, Q and K quantizers, threshold unbalanced | Sol's routed steps | Sol -27% (8 heads) | one extra read of q a call |
| `rotate` in Sol (kitchen fork; `quantizer="rotated"`, shipped) | sign-diagonal plus Hadamard rotation of every q/k row before INT8 | Sol's routed steps | plain 0.0265, balanced 0.0193, rotated 0.0140, both 0.0126 | about 15% of a Sol call as built |

The weights fold, graded through Sol's own decomposition (tau 1.0, first 8 heads,
[`bench/results/2026-09-14_channel_balance_{b49_s15,b0_s15}.json`](../../bench/results/2026-09-14_channel_balance_b49_s15.json)):

| cell | arm | Sol sparsity_l2 | Sol quant_l2 | Sol total_l2 | sage fp8++ l2 |
|---|---|---|---|---|---|
| block 49, step 15 | plain | 0.0324 | 0.0265 | 0.0415 | 0.0487 |
| block 49, step 15 | balanced | 0.0323 | 0.0231 (-12.9%) | 0.0393 | 0.0453 |
| block 0, step 15 | plain | 0.1247 | 0.0024 | 0.1244 | 0.0036 |
| block 0, step 15 | balanced | 0.1248 | 0.0024 (+0.9%) | 0.1244 | 0.0036 |

Routing is unchanged: the eager Sol reference moves by exactly as much as exact attention under the bf16 re-rounding
of the balanced inputs (1.69e-2 against 1.67e-2 at block 49, 5.4e-4 against 5.4e-4 at block 0). Sol's in-quantizer
factor removes about twice what the fold removes and is neutral at blocks 0 and 32
([`bench/results/2026-09-15_channel_balance_kernel_b{49,0,32}_s15.json`](../../bench/results/2026-09-15_channel_balance_kernel_b49_s15.json);
also `bench/results/2026-09-15_channel_balance_kernel_b{0,32}_s15.json` and the `bench/results/2026-09-15_channel_balance_kernel_*.json`
set; install record
[`data/results/2026-09-15_kitchen_0234_qk_balance_install.json`](../../data/results/2026-09-15_kitchen_0234_qk_balance_install.json),
off path bit-identical to the previous wheel). The rotation grade is
`bench/results/2026-09-15_sol_rotate_b{49,0}_s15.json`; its install record is
[`data/results/2026-09-15_kitchen_0234_rotate_install.json`](../../data/results/2026-09-15_kitchen_0234_rotate_install.json).

With every lever on, block 49 still sits at several times block 0: the attention shape amplifies whatever rounding
remains, and a per-channel factor cannot fix a spike in one token's row. Rotation is the lever that reaches it.

## 5. What the peaky heads attend to

768 sampled query rows per cell over all keys, all 56 heads, exact fp32 softmax (the checkpoint scan record above):

| | block 0 | block 49 |
|---|---|---|
| attention mass on the 395 text keys, median head | 0.4% | 12.2% |
| attention mass on the 1,150 audio keys, median head | 17.7% | 16.2% |
| queries whose top-1 key is a text key, median head | 4% | 20% |
| queries sharing one top-1 key (sink signature), median / max head | 2% / 67% | 7% / 34% |
| loudest key row's norm over the median, worst heads | 1.0-1.1x | 1.0-1.4x |

The four worst heads put 6-38% of their mass on text keys, with text tokens (rows 0 and 114) as top-1 for a tenth to a
fifth of queries and no key-norm outlier. Head 11 is the heaviest text reader at 38%. The block-49 error is
concentrated on the prompt read, not on a sink token and not on video-to-video attention.

## 6. Visibility, 2026-09-15: three scenes, labelled, not blind

"Shipped" or `default` in these records means the owner's t2v graph of that date: sage `fp8++` plus Sol at the
recipe of the day, INT8 attention everywhere.

| look | arms | what was seen | record |
|---|---|---|---|
| 6a. market scene, seed 730451892, two viewers | shipped; balanced (balance node + sage `fp8++ balanced`); exact tail (45, 48, 49 on bf16) | shipped: the porter and crate morph as he turns, this scene's known failure. Balanced: no morph, "way better". Exact tail: best "in subtle ways" (sets the crate down and lifts it, steps around shoppers, a more natural weight shift). The woman's identity drifts in all three | `bench/results/2026-09-15_block49_market_feedback.md` |
| 6b. diner, seeds 730451892 and 20260915 | the same three | the default last (a chef appears from nothing; the door sign reads "TUE SATR DINER" twice), rebalanced second (sign legible), bf16 tail best. Rebalanced within a second of the default's wall time; the bf16 tail about a sixth more | `bench/results/2026-09-15_block49_diner_batch.md` |
| 6c. every lever (`h3_probe_t2v_levers`) against the same plus bf16 at 45/48/49 (`h3_probe_t2v_policy`), market and restaurant kitchen | the pair that decides the tail | market: the owner could not tell them apart; one viewer preferred the bf16 tail on hands and coins (checked on frames: a legibility difference, not an undecided object); a third "super close, maybe faces". Loudness differences were the take, not the tail (the six diner clips sit within two LU) | `bench/results/2026-09-15_block49_kitchen_batch.md` |

The ranking follows the error sizes: unrebalanced worst, the free levers in the middle, no INT8 error at those blocks
best. The owner's reading of the morph: the worse outputs never decide whether a crate is one deep box or two shallow
trays, and the morph is that indecision over time, which is what a less noisy read of the prompt at the last block
would fix. `bench/compare_clip_pixels.py` confirms the clips are different takes from frame 0, so any single
comparison can still be luck; the prior (the porter usually morphs) is what makes it more than a coin flip. Captioned
three-band stacks exist beside the singles (`bench/stack_labeled_clips.py`).

## 7. What this does not establish

- **A blind verdict.** Every look was on labelled originals in a known order.
- **fp8 q/k attention.** Each element has its own exponent, so the starving cannot happen the same way, but fp8 still
  places a per-block range with three mantissa bits: expected much smaller, unmeasured. bf16 at the loud blocks is the
  best arm measured, not a measured zero; nothing compares it to fp32.
- **Anything outside attention.** The loud channels are created by the k_norm gain after the qkv projection, so the
  convrot INT8 linears never see them (read from the model code).

## 8. The in-quantizer factor, built 2026-09-15

The weights fold is capped by the architecture: the only per-channel weights after the projection are the q/k RMSNorm
gains, shared across heads, so a per-head factor (twice the gain in simulation) had nowhere to live there. The
quantizers stream q and k once, so the multiply is free inside them. Built first in the sage fork as `qk_balance`
(v0.7.19; its `CHANGELOG.md` and `tests/test_qk_balance.py`), then the same factor in Sol's preprocess on the kitchen
fork:

- Q is multiplied by `f` and K by `1/f` right after the load, before the absmax. `f = rms_k^0.5 / rms_q^0.5` per
  (batch, kv head, channel), geometric mean one per head, from `torch.linalg.vector_norm` with fp32 accumulation, so no
  copy of q or k is made (the difference from `smooth_k`, rejected for materializing a K copy at the frame ceiling).
- Gated per head on K's top-4 channel energy share; below the gate a head's codes are bit-identical to the plain path.
  Under GQA the factor is per kv head.
- No calibration, no capture, no RoPE-pair constraint (it acts after RoPE), no per-block list: blocks 45, 48 and 49
  open on their own. Not only those: some heads open on every captured block, block 0 most of all
  ([`data/results/2026-09-27_qk_balance_gate_on_capture.json`](../../data/results/2026-09-27_qk_balance_gate_on_capture.json)),
  so `balanced` was never inert behind the dense tail.

### What it measured

Sage fp8++ as served, the same cells and reference:

| cell | plain | `qk_balance` (gate 0.2, the default) | gate 0.5 |
|---|---|---|---|
| block 49, step 15 | 0.0472 | 0.0364 (-22.9%) | 0.0412 (-12.8%) |
| block 40, step 15 | 0.0426 | 0.0426 (+0.1%) | (no head opens) |
| block 0, step 15 | 0.0085 | 0.0085 (-0.2%) | 0.0085 (+0.0%) |

In the CPU simulation block 49 loses a third of its QK error at the default gate and head 17 three-quarters of its
own; block 0's simulated 3% cost dilutes to nothing in the real kernel, which is why the default was chosen on kernel
rows. Cost: +3.3 ms on the quant step at 104k rows, +0.7% on the call, no change in peak memory. The balanced fp8++
call at block 49 (0.0364) is below the fp16 kernel's unbalanced error on the same cell (0.0409). Prior art: the rescale
is SmoothQuant's migration pointed at the attention product
([`../research/smoothquant_for_attention_qk.md`](../research/smoothquant_for_attention_qk.md)). The Sol form is offered
upstream as [`sol_upstream_qk_balance_pr.md`](sol_upstream_qk_balance_pr.md), not yet sent.

## The defaults, re-read against the sage-side error records, 2026-09-14

Written from the sage fork's session after the fork graded `smooth_k` across ten cells of the 2026-09-03 capture. The
sage-side instrument measures sage's dense INT8/FP8 kernel against fp32 attention and never runs Sol, so it informs a
Sol default only where Sol's error shares a cause with sage's: the INT8 term. Sources: the error split in
[`sol_error.md`](sol_error.md), the 2026-08-20 attribution in the roadmap, and the sage fork's `CHANGELOG.md` entries
"sm89 q/k quantization", "`smooth_k` on H3: graded across the trajectory" and "MiniMax H3, block 49: per-channel K
balancing" (harnesses `tests/spikes/spike_h3_real_activations.py` and `tests/spikes/spike_h3_k_channel_balance.py`
there).

**Quantization error is a function of block depth and nothing else.** At a given block the captured steps agree to a
fraction of a percent, and mean rtol rises about fivefold from block 0 to block 49; Sol's `1 - quant_cos` shows the
same shape. Two instruments on two kernels and two captures agree the INT8 term has no step axis.

What that implies per knob, as re-read then and updated since:

| knob | reading |
|---|---|
| `tau`, `start_percent`, `end_percent` | no information from the sage side: nothing quantization-shaped varies with step or sigma, so their cost and benefit are the sparsity term's |
| `min_tokens` | no information; below it the call goes to the dense kernel, which the sage records grade and nothing else |
| `dense_blocks` | a dense block removes Sol's sparsity term and leaves the dense kernel's INT8 term; on the sage fallback that was half the error at block 49. Superseded by the rotated kitchen fallback and the 2026-10-02 list ([`sol_dense_blocks.md`](sol_dense_blocks.md)) |
| the block-49 fold | the experiment ran in four steps: rank all 50 blocks from the weights (done, section 1); build the fold (`MiniMaxH3ChannelBalance`); grade it through Sol (done, section 4); then a blind multi-scene comparison, which was overtaken by the in-quantizer forms and rotation |
| `token_routing` at block 49 | token routing hurt block 49, head 11 at every step; the loud channels are a candidate mechanism. With balance and rotation on it gave the best Sol number on block 49 ([`quant_policy.md`](quant_policy.md)); no render judged |
| `smooth_k` on the sage fallback | leave it off: slightly harmful at block 0, a few percent helpful at 32-49, a full bf16 K copy at the frame ceiling, and it inverts this pack's clone-v choice. The fold does what it reached for at block 49 four times over, for free |
| per-thread q/k on sm89 | leave it: per-warp CUDA quantization measured slower for the whole call at every length and worse on all ten cells |

Morton's one cross-link: `kcvar` at block 49 is dominated by the same four dims, so the routing threshold there is set
by the channels that set the INT8 scale, which may be why block 49 was the counterexample in every ordering table
([`morton.md`](morton.md)). A hypothesis consistent with three records, not measured.

## Records

- Sage fork `CHANGELOG.md` (the three entries above) and its `tests/spikes/` harnesses.
- This repo: [`../state/roadmap.md`](../state/roadmap.md), "block 49 attributed, at the input level" (2026-08-20), and
  `bench/results/2026-08-20_head_magnitudes*.json`; `bench/results/2026-09-14_channel_balance_*.json`; the 2026-09-15
  kernel grades, install records and viewer records named above; [`quant_policy.md`](quant_policy.md).
- The node on its own: [ComfyUI-H3-Quant](https://github.com/fblissjr/ComfyUI-H3-Quant), published 2026-09-15 from
  `standalone/h3_quant`; this pack defers to it when installed.
