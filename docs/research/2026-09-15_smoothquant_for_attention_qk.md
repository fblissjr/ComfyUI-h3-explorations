# SmoothQuant's trick, pointed at attention's q and k

**In short (as of 2026-09-15).** The channel balance this repo applies to attention's q and k (`MiniMaxH3ChannelBalance`, sage's and Sol's `qk_balance`) is SmoothQuant's migration pointed at the attention product instead of a linear, computed per head per call without calibration, and gated to heads with loud channels. The rotation form of the same fix already exists in an attention kernel (kitchen's `int8_attention`), and it is both the stronger answer and prior art, so the novelty claim narrows to the migration form. Sol has run `quantizer="rotated"` by default since 2026-09-27 ([`docs/attention/quant_policy.md`](../attention/quant_policy.md)).

last updated: 2026-09-15

**This file owns the prior-art framing of the channel balance** (the
`MiniMaxH3ChannelBalance` node, sage's `qk_balance`, Sol's `qk_balance`):
where the idea comes from, who already uses it in DiT quantization, and
which part of what this repo does is, as far as we can tell, not in the
literature. The mechanism, the measurements and the visible results live in
[`docs/attention/block49_quant_error.md`](../attention/block49_quant_error.md); the plan
in [`docs/attention/quant_policy.md`](../attention/quant_policy.md). Prior-art claims
below are dated and hedged on purpose: "as far as we know on 2026-09-15"
is the strongest thing said here.

## The idea, in one line

For any per-channel factor `f`, `q . k == (q * f) . (k / f)`. If a few
channels of `k` are loud and the quantizer shares one scale across all
channels, divide `k` by a factor that flattens them and multiply `q` by the
same factor. Every score is unchanged; the rounding is spread across both
sides instead of starving one.

## Where it comes from: SmoothQuant (LLMs, 2022)

SmoothQuant (Xiao et al., MIT Han lab, 2022) quantizes LLM linear layers
to W8A8. Its problem is the same shape as ours: activations have a few
outlier channels, per-tensor or per-token INT8 gives every other channel a
handful of levels. Its move: per input channel, divide the activation by
`s = max|X|^a / max|W|^(1-a)` and multiply the weight's matching column by
`s`, offline, so the matmul is unchanged and the outliers migrate into the
weights, which quantize per output channel and can absorb them. Its
migration strength `a` is 0.5 by default, with the paper's own sweep
showing the extremes (all onto one side) losing.

The mapping to what this repo does is direct: SmoothQuant's activation is
our `k`, its weight is our `q`, its `s` is our `f` (rms instead of absmax,
the same exponent), and its offline weight absorption is the balance node's
fold into `q_norm.weight` / `k_norm.weight`. The 0.5 exponent was measured
here independently before the connection was made (sweep in the sage
fork's `CHANGELOG.md`, workload intel "MiniMax H3, block 49"; peak at 0.5,
0.35 and 0.65 a few points behind, the extremes worse everywhere), which
is a small piece of evidence that the result is the method's and not a
tuning accident.

## Who already uses it on DiTs

All of these apply the migration to the LINEAR layers (W8A8 or W4A8), not
to attention:

- **SVDQuant / Nunchaku** (same lab, 2024): smoothing is its first stage for
  Flux and SDXL; a low-rank branch then absorbs what smoothing leaves. The
  most widely run DiT quantization in ComfyUI, so a lot of people already
  run SmoothQuant's idea without naming it.
- **PTQ4DiT** (2024): "channel-wise salience balancing", the same
  migration under another name, for DiT linears.
- **ViDiT-Q** (2024): the same for video DiTs, with the factor tracked
  across timesteps.
- **Q-DiT** (2024): group-wise scaling against the same outlier channels.

Read against these, the channel imbalance in H3's last blocks is not
exotic; it is the standard outlier-channel problem, showing up in a place
the DiT quantization papers do not look.

## What is different here, as far as we know

1. **The target is the attention product, not a linear.** The INT8
   attention kernels this repo runs (SageAttention, comfy-kitchen's
   `sol_attn`) quantize `q` and `k` activations per token row (or per
   token block) with one scale across the 128 head channels. That is
   SmoothQuant's shared-scale problem inside the attention kernel, and the
   fix transfers unchanged because `q . k` has the same bilinear form as
   `X . W`. We have not found a paper or kernel that applies the migration
   there. SageAttention's own "smooth K" is a different operation (subtract
   the per-channel mean of K, which removes a bias, not a spread); the two
   compose.

   **But the rotation form of the same fix already exists in an attention
   kernel** (found 2026-09-15, after this note was first written):
   comfy-kitchen's `int8_attention` applies a randomized block-Hadamard
   rotation to q and k before quantizing, the QuaRot / SpinQuant / convrot
   move (rotate so outliers spread across channels, quantize, and let the
   orthogonality keep the product exact). On the block-49 capture it is
   immune to the loud channels and more accurate than the rebalanced Sol
   row (`data/results/2026-09-15_ck_int8_attention_block49.json`). So the
   honest novelty claim narrows to: the migration form, per head, per
   call, gated, in kernels that quantize unrotated. The rotation form is
   the stronger answer and is prior art.
2. **No calibration set.** SmoothQuant and its DiT descendants derive `s`
   from calibration activations offline. The kernel forms here compute `f`
   from the call's own `q` and `k` (their channel rms over the sequence),
   per head, per call. The offline form exists too (the node folds a
   weights-derived factor, no captures needed), and it recovers about half
   of what the per-head dynamic form does on the block that matters
   (`data/results/2026-09-15_channel_balance_kernel_b49_s15.json`).
3. **Gated per head.** Where no channel is loud, migrating resolution onto
   `q` costs a few percent for nothing. The kernels apply the factor only
   on heads whose four loudest K channels carry at least a fifth of K's
   energy; the rest are untouched, which is what makes it safe to leave on
   everywhere.
4. **RoPE in the way.** The fold into the norm weights has to commute with
   the rotation that follows the norm, so the folded factor is forced equal
   within each RoPE pair. The per-head kernel form has no such constraint,
   which is part of why it does better.

## What is the same limit

SmoothQuant cannot fix an outlier that lives in one token rather than one
channel, and neither can this: the factor must be the same for every token
or the scores change. That residual is the gap that remains between the
balanced arms and bf16 attention on the last three blocks. The literature's
answers are finer granularity (per-group scales, SVDQuant's low-rank
residue) or rotation (QuaRot, SpinQuant), and rotation handles the
per-token case too, since it flattens whatever a row's outlier is. That
is what kitchen's `int8_attention` does, and it is the shape Tier 2 in
[`docs/attention/quant_policy.md`](../attention/quant_policy.md) should take in sage's
and Sol's quantizers rather than a second scale group.

## What to cite if this gets written up

- The mechanism and measurements: `docs/attention/block49_quant_error.md`
  sections 1 through 4 and 8.
- The checkpoint one-liner (the weights fact anyone can check on any H3
  checkpoint): `bench/check_channel_balance.py`, or the inline version in
  `docs/attention/block49_quant_error.md`.
- The grade records: `bench/results/2026-09-14_channel_balance_*.json`
  and `bench/results/2026-09-15_channel_balance_kernel_*.json`.
- The visible results: `bench/results/2026-09-15_block49_market_feedback.md`,
  `bench/results/2026-09-15_block49_diner_batch.md`.
- The kernels: the sage fork (v0.7.19, `qk_balance`) and the comfy-kitchen
  fork (`h3-build`, `sol_attn(qk_balance=True)`).
