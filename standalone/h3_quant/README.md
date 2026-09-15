# ComfyUI-H3-Quant

Findings about low-precision attention on MiniMax H3 in ComfyUI, and one
node that acts on them. Measured on one machine (an RTX 4090, one set of
captured activations, three scenes watched by a few people on labelled
clips), so nothing here is established beyond that. Every claim below links
to the measurement behind it in
[ComfyUI-h3-explorations](https://github.com/fblissjr/ComfyUI-h3-explorations);
the point of publishing is for other people to check it on their own card.

## The finding

Long H3 renders run attention in INT8 for speed: SageAttention on the dense
steps for some, the Sol / Block Sparse Attention kernel on the routed steps
for most. Both round q and k with one scale per row across a head's 128
channels. H3's `k_norm.weight` on blocks 45, 48 and 49 puts most of its
energy into four channels, on every checkpoint checked
([scan](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/bench/results/2026-09-14_block49_checkpoint_scan_and_targets.txt)),
so on those blocks the loud channels set the scale and the rest keep a
couple of levels. Block 49 also reads the prompt most sharply, and on our
clips that showed as morphs and garbled text
([mechanism and evidence](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/docs/h3_block49_quant_error.md)).

Check the weights fact on your own checkpoint, no GPU needed:

    python check_h3_loud_blocks.py /path/to/minimax_h3_*.safetensors

## What "error" means, and what was measured

Not the video: the same captured q, k, v through the INT8 kernel and through
fp32 attention, outputs subtracted, divided by the size of the exact output
([how the captures and grades work](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/docs/h3_block49_quant_error.md#what-error-means-on-this-page)).
On block 49, first 8 heads: Sol 0.027, SageAttention fp8++ 0.049, and
kitchen's dense `int8_attention` 0.017 because it rotates q/k before
rounding and does not have this problem
([records](https://github.com/fblissjr/ComfyUI-h3-explorations/tree/main/bench/results)).
The weights fold in this node takes about 13 percent off Sol's term
([grade](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/bench/results/2026-09-14_channel_balance_b49_s15.json));
per-head factors and a Hadamard rotation inside the kernels do two to four
times that and are headed upstream as comfy-kitchen PRs
([policy and status](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/docs/h3_quant_policy.md)).

## What ships here

`MiniMax H3 Channel Balance`: since `q . k == (q * f) . (k / f)`, it folds
`k_norm.weight / f` and `q_norm.weight * f` into the model at load on the
lopsided blocks, with f from the checkpoint's own norm weights, equal within
each RoPE pair. Exact for every score; only the INT8 rounding changes. No
kernels, no forks, zero render-time cost, off by default. It does nothing
useful on plain pytorch attention or on the kitchen dense backend without
Sol, because neither has the problem.

## Try it

Put this folder in `custom_nodes/`, run the scan, add the node between the
model loader and your attention nodes, render one prompt with `balance` off
and again with "loud blocks (from weights)" at the same seed, and watch
both. Same seed is not the same take once numerics change: judge whether
objects stay one thing and text stays text, not frame matches. An issue
with your card, chain (Sage, Block Sparse, Model Attention Backend, plain),
checkpoint and what changed, or did not, is the result this repo wants.

## What is uncertain

One machine, one capture set, unblinded viewers; fp8 attention unmeasured;
the fold is not the ceiling (bf16 on the three blocks still looked a little
better); nothing here concerns the quantized linears. The full list, with
what would settle each item, is in the
[open questions](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/docs/h3_block49_quant_error.md#7-what-this-does-not-establish);
prior art (this is SmoothQuant's migration aimed at attention) is
[here](https://github.com/fblissjr/ComfyUI-h3-explorations/blob/main/docs/research/smoothquant_for_attention_qk.md).
