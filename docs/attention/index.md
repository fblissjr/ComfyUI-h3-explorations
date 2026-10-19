# Attention: Sol-Attn and quantization

Start at [`sol_attn.md`](sol_attn.md). Pages run from the overview down to the detail.

| page | what it is for |
|---|---|
| [`sol_attn.md`](sol_attn.md) | **the Sol-Attn authority**: what it is, what ships and where the values live, every knob and what is measured about it, what not to rely on, what is open |
| [`sol_kernel.md`](sol_kernel.md) | the kernel: which build runs, installing it, verifying it ran, what it exposes, the node's and the Triton backend's history |
| [`sol_wiring.md`](sol_wiring.md) | where the node sits in a graph, what declines to the dense kernel, and the composition traps that stop Sol silently |
| [`sol_sink.md`](sol_sink.md) | the conditioning sink: the modes, the segments on a real graph, what references cost in forced-exact attention |
| [`sol_dense_blocks.md`](sol_dense_blocks.md) | which blocks to keep off Sol: the 2026-10-02 decision and every ranking measured before it |
| [`sol_error.md`](sol_error.md) | where Sol's error comes from: sparsity against INT8 by depth, `tau` 1.0 against 1.3, where the exact work goes |
| [`sol_measured_arms.md`](sol_measured_arms.md) | the timed arms and quality judgements, the sigma-window table, the unit trap |
| [`sol_instruments.md`](sol_instruments.md) | the route observer, call timer, tau sweep and tau table, the reproduce commands, the measurement traps |
| [`sol_upstream.md`](sol_upstream.md) | what upstream ships, once per upstream at the revision read: the paper, Sol-Engine, Sol-H3, core, comfy-kitchen, sglang, other packs; and what we would adopt |
| [`sol_upstream_reads.md`](sol_upstream_reads.md) | the dated passes over comfy-kitchen and ComfyUI core, newest first: history, not guidance |
| [`sol_upstream_qk_balance_pr.md`](sol_upstream_qk_balance_pr.md) | the `qk_balance` change as prepared for upstream, not yet sent |
| [`morton.md`](morton.md) | Morton token ordering: closed 2026-09-27, not refuted; what was measured and why it went |
| [`quant_policy.md`](quant_policy.md) | the per-block precision policy for attention and linears, each row with status and evidence, and the tiered plan |
| [`block49_quant_error.md`](block49_quant_error.md) | why unrotated INT8 attention loses accuracy on H3's last blocks, the levers built, and what rotation fixed |

Canvas and length against Sol's blocks are in [`../h3/geometry.md`](../h3/geometry.md) and its tables page [`../h3/geometry_cost.md`](../h3/geometry_cost.md).

Studies under [`../research/`](../research/index.md): the Sol node redesign of 2026-09-27, the sparse folder (dense
blocks, the hyperparameter campaign), token routing, rotation and low-bit attention, SmoothQuant on q/k, the 2026
sparse-attention and scorer sweeps, the quantization levers.
