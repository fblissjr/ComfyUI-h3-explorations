# sol_attn: optional `qk_balance`, per-head q/k channel rebalancing inside the INT8 quantizers

## What

`sol_attn(..., qk_balance=True)` rescales q by a per-(batch, head, channel)
factor f and k by 1/f before the INT8 quantizers, so `q . k` is unchanged
in exact arithmetic and the rounding is spread across both operands. Off
by default; the off path is bit-identical to today's kernel (a null factor
skips every new pass, and the quantizers take no multiply at all).

The factor is `rms_k^0.5 / rms_q^0.5` over the live rows, geometric mean
one per head, computed in the preprocess from the call's own q and k: K's
sums of squares ride along in pass 1, Q's take one extra read of q, two
fixed-order per-(b, h) reductions produce f. It is applied in the pooled,
Q and K quantizers; the routing threshold, kmean, kcvar and the coarse
branch stay in the unbalanced space, so the route is invariant up to the
changed quantization. A head whose four loudest K channels carry under a
fifth of its K energy keeps f = 1 (gate), so ordinary heads are untouched.
Constants live in `sol_layout.cuh` and are mirrored in the eager reference.

CUDA only; HIP accepts the argument and refuses True (its preprocess is a
separate source). `sol_attn_chunked` does not take it: its q and k arrive
chunk by chunk, before the sequence's rms exists.

## Why

MiniMax H3's last transformer blocks (45, 48, 49 on every released
checkpoint) put most of K's energy into four channels through their
`k_norm.weight`. `quant_k_rows` uses one INT8 scale per key row across all
128 channels, so on those blocks four channels set the scale and the other
124 keep a handful of levels, on the block that reads the prompt most
sharply. Measured on captured activations (S=104k, first 8 heads, relative
L2 against fp32 attention on the same inputs), Sol's INT8 term at block 49
goes from 0.0265 to 0.0193 with the option on; blocks 0 and 32 are
unchanged (gate shut). A weights-side fold of the same idea gives 0.0231.
`int8_attention` does not have this problem because it rotates q/k first;
`sol_attn` quantizes unrotated, which is what this addresses without
changing its layout.

This is SmoothQuant's migration (rms^0.5 on each side) applied to the
attention product instead of a linear layer, per head, per call, with no
calibration.

## Tests

Ten cases in `tests/test_sol_attn.py`: the reference factor's algebra
(reciprocals, geometric mean one, gate open on a loud head and shut on a
flat one); the eager identity (the rescale is a no-op in fp32 up to route
flips on threshold ties); a shut gate reproduces the plain call's bytes;
one flat head among loud ones stays bit-identical while the loud ones
move; the INT8 error against dense attention falls on loud-channel inputs
with every block routed; the balanced call tracks the reference at least
as closely as the plain one; the route moves only within what a flipped
block is worth; dead `block_len` rows count for nothing; signature parity
across the entries; HIP refuses.

Verified on sm_89 (RTX 4090), CUDA 13.2/13.3, torch 2.14: the full
`test_sol_attn.py` on this branch, and plain `sol_attn` bit-identical to
the previous build on twelve shape/option cases.

## Cost

One kc-sized scratch array for the partials (carved whether or not the
option is on: the plan is sized before the call's options are known, and
the T^2 index array sits beside it), one extra read of q when on, no
measurable wall-time change on a 345-frame H3 render.
