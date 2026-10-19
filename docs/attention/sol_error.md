# Where Sol's error comes from

Detail page of [`sol_attn.md`](sol_attn.md).

**The answer.** Through the trunk, Sol's sparsity approximation costs three to five times what its kernel's INT8
arithmetic costs. At block 49 the two are equal, on unrotated INT8. Lower `tau` lowers the sparsity term; 1.0, the
shipped value, is the lowest value graded and errors were still falling there. These figures are from captures taken
before the `rotated` quantizer shipped, which lowers the INT8 term ([`block49_quant_error.md`](block49_quant_error.md));
the sparsity term is unchanged by it.

## How the two terms are separated

`bench/analyze_sol_error.py::decompose_single` builds three outputs from the same captured q, k, v:

    out_dense = dense_reference(q, k, v)           fp, full attention
    out_eager = eager_sol_reference(q, k, v, tau)  fp, Sol's sparsity
    out_cuda  = cuda_sol_kernel(q, k, v, tau)      Sol's CUDA kernel, INT8 inside

    sparsity_l2 = eager vs dense      the sparsity approximation
    quant_l2    = cuda  vs eager      the kernel's INT8 arithmetic

The weights are applied before q, k, v exist, so whatever the INT8 checkpoint or the encoder contributes is in all
three and cancels. This instrument says nothing about weight quantization in either direction.

## The split, one capture, tau 1.0

Fourteen (block, step) rows of a ref2va capture
([`data/results/2026-08-19_sol_error_per_head_tau1.0.json`](../../data/results/2026-08-19_sol_error_per_head_tau1.0.json)):

| block | sparsity_l2 | quant_l2 | ratio |
|---|---|---|---|
| 0, 8, 16, 24, 32, 40 | 0.089 - 0.224 | 0.024 - 0.049 | 2.90 - 4.79 |
| 49, step 3 | 0.1427 | 0.1354 | 1.05 |
| 49, step 11 | 0.1647 | 0.1353 | 1.22 |

Block 49 is the last of the 50 (`transformer/config.json`). Its INT8 error is about 10x the trunk's on a
scale-invariant measure (`1 - quant_cos` 2.1e-4 to 8.3e-4 at blocks 0-40, 7.5e-3 and 7.7e-3 at block 49), while its
sparsity gets easier (`sparsity_cos` 0.9915 against 0.9784 at block 40). Why the INT8 term breaks there is
[`block49_quant_error.md`](block49_quant_error.md). The propagation measurement put blocks 45, 48 and 49 among the
blocks whose error moves the output least ([`sol_dense_blocks.md`](sol_dense_blocks.md)), so "nothing after block 49
absorbs its error" is reasoning, not a measurement.

Limits: one capture, one tau, seven blocks at two steps, an L2 against an oracle and nothing perceptual. None of it
says Sol is not worth its speed.

## `tau`: 1.0 against 1.3

A matched pair on the same capture and the same fourteen cells
([`data/results/2026-08-19_sol_error_per_head.json`](../../data/results/2026-08-19_sol_error_per_head.json) at 1.3,
`..._per_head_tau1.0.json` at 1.0):

| | tau 1.3 | tau 1.0 (ships) |
|---|---|---|
| sparsity_l2, median | 0.168 | 0.143 |
| quant_l2, median | 0.035 | 0.037 |
| total_l2, median | 0.182 | 0.156 (0.86x) |
| sparsity_cos, min | 0.969 | 0.978 |

1.0 is better than the value it replaced and sits at the edge of the tested range, in the direction that was still
improving. Nothing here says it is optimal. The sweep that answers that is the tau-sweep instrument
([`sol_instruments.md`](sol_instruments.md)).

**The trap in the control file.** `2026-08-19_sol_error_control.json` carries `tau: 1.3` in its metadata and an empty
`rows`. Its `controls` hold the dense limit (`tau=-1e9`) and the sparse limit (`tau=26`), the bounds rather than a tau
arm (`bench/analyze_sol_error.py`, the controls block). Read without the producer it looks like a tau comparison with
a sparsity error near zero.

## Where the exact work goes, by depth

On the 362-frame 1024x768 three-reference capture, tau 1.3, 4 of 56 heads (`bench/analyze_routing.py`, 2026-08-16):

| block | diagonal | sink | routed | total exact | `tau` can address |
|---|---|---|---|---|---|
| 0 | 0.2% | 16.6% | 29.8% | 46.6% | 64% |
| 8 | 0.2% | 16.6% | 12.3% | 29.1% | 42% |
| 24 | 0.2% | 16.6% | 11.3% | 28.1% | 40% |
| 49 | 0.2% | 16.6% | 10.9% | 27.7% | 39% |

- **The sink is a floor `tau` cannot reach**: 16.6% of all pairs at every depth on reference-heavy work
  ([`sol_sink.md`](sol_sink.md)), so raising `tau` has a ceiling set by reference load.
- **Block 0 routes 1.7x more than any deeper block**, consistent with early attention being near-uniform. Keeping it
  dense would cost about 1% of compute by arithmetic.
- **Attention is not very sparse on this workload.** On a 124-frame t2v capture a query still needs 178 of 591 key
  blocks for 90% of its mass at its most concentrated, and 394 at block 0 ([`morton.md`](morton.md)), which bounds
  what any block-sparse method saves and argues against pushing `tau` far.

Routed density from these offline tables re-derives the router in float arithmetic the kernel does not use. The live
count from the kernel is the route observer ([`sol_instruments.md`](sol_instruments.md)); its densities join this
table through `ordering_effect_density`.

## The route itself under INT8

Against the fp32 route recomputed with the eager rule, the INT8 kernel's routed-block count per query block is off by
about two blocks at block 49 and about one at block 0 (counts only; a swap at equal count reads as agreement).
`qk_balance` takes a fifth off that at block 49, rotation a quarter, both two fifths; at block 0 rotation takes a
third. Density is unchanged in every arm, so the error is scatter, not bias
([`bench/results/2026-09-15_sol_route_b{49,0}_s15.json`](../../bench/results/2026-09-15_sol_route_b49_s15.json),
`bench/grade_sol_route_on_capture.py`).
