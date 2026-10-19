# Attention: Sol-Attn and quantization

| page | what it is for |
|---|---|
| [`sol_attn.md`](sol_attn.md) | **the Sol-Attn authority**: the published claim, the knobs, the measured arms, ordering, its own do-not-rely table |
| [`morton.md`](morton.md) | Morton token ordering: what it does, what was measured, should you turn it on |
| [`sol_upstream.md`](sol_upstream.md) | what upstream says: the paper, Sol-Engine, Sol-H3, comfy-kitchen and core's own Sol node, read on dated passes |
| [`sol_upstream_qk_balance_pr.md`](sol_upstream_qk_balance_pr.md) | the `qk_balance` change as prepared for upstream |
| [`quant_policy.md`](quant_policy.md) | the per-block precision policy for attention and linears, each row with status and evidence, and the tiered plan |
| [`block49_quant_error.md`](block49_quant_error.md) | why the last block's INT8 attention error is five times block 0's on unrotated INT8, and what was done about it |

The inputs-cost interaction (canvas and length against Sol's blocks) is in [`../h3/input_impacts.md`](../h3/input_impacts.md).

Studies under [`../research/`](../research/index.md): the Sol node redesign of 2026-09-27, the sparse folder (dense
blocks, the hyperparameter campaign), token routing, rotation and low-bit attention, SmoothQuant on q/k, the 2026
sparse-attention and scorer sweeps, the quantization levers.
