# What can be changed about H3's quantisation, and what each lever needs

**Start here before proposing a quantisation change.** This file owns the
lever inventory: what is available, what it costs, what it needs, and which
ones are closed. Nothing owned this before 2026-08-31, which is how
[`h3_dit_implementations.md`](h3_dit_implementations.md) §10.5 came to carry a
withdrawn claim that a better quantised DiT "does not exist as an option".

**Read the scope line before any number here.** `int8_convrot` is **W8A8**.
`comfy_kitchen`'s `int8_linear` rotates the activation online with the same
Hadamard, quantises it **per token**, runs an int8 GEMM whose int32
accumulation is exact, and scales in fp32 —
[`comfyui_h3_t2va_trace.md`](comfyui_h3_t2va_trace.md) §1.7-1.8 puts it as
*"all the error is in the two roundings"*. **Every measurement in this file is
the WEIGHT rounding.** The activation rounding has never been measured, and
it is the side with the outliers. That is
[`open_experiments.md`](../open_experiments.md) #23.

Scope is the DiT's 200 backbone linears. `adaln_proj.linear` is F16 and the
output heads are F32, so all quantisation loss lives there. The encoder is out
of scope: its weight-side question is closed
(`bench/results/2026-08-29_int8_convrot_headroom.json`), and the owner runs
bf16 Qwen3-VL.

---

## The inventory

| lever | what it changes | status | needs |
|---|---|---|---|
| **`unmerged_blocks`** | applies a LoRA at the call so the base weight is never requantised | **ships today**, off by default | a decision on whether it matters at the output |
| **deterministic merge** | replaces the merge path's stochastic rounding with round-to-nearest | **arithmetic, not a hypothesis** — see below | our node merging itself rather than via `add_patches` |
| **`convrot_groupsize` 1024** | a wider Hadamard, spreading outliers further before rounding | **measured on the weight side**, and it is one module kind | a CPU requant of 50 tensors, and a timing arm |
| **`full_precision_matrix_mult`** | per-module escape to a bf16 matmul, killing **both** roundings there | available, unmeasured | editing the module's `comfy_quant` blob — `comfy/ops.py:1150` reads it before the format branch, so **no core patch** |
| **SmoothQuant via `pre_quant_scale`** | migrates activation outliers into the weight | reachable, gated on #23 | the scale as a module attribute (`add_object_patch` → `comfy.utils.set_attr`, a plain `setattr` that backs up the old value) **plus** a rebaked weight |
| **GPTQ rounding** | picks int8 values minimising `‖(W−Ŵ)X‖` rather than `‖W−Ŵ‖` | not started | Hessians, and a reason to believe it transfers from LLMs to a DiT |
| channel permutation | permuting `fc1`'s output rows and `fc2`'s input columns identically — an exact-equivalence transform that changes which channels share a rotation group | **cannot be assessed from what is recorded** | a capture that keeps per-channel structure instead of a median |

**Closed. Do not revisit:**

- **Per-GROUP weight scale.** `int8_linear` raises unless `weight_scale` is
  scalar or per-output-channel.
- **fp16 outlier channels** (LLM.int8-style decomposition). Needs kernel
  support.
- **Bias correction.** The backbone linears carry no bias — there is no slot.

---

## What Tier 0 measured, 2026-08-31

`bench/analyze_weight_outliers.py` →
[`bench/results/2026-08-31_dit_weight_outliers.json`](../../bench/results/2026-08-31_dit_weight_outliers.json).
All 200 modules of `minimax_h3_fl2va_pruned_int8_convrot` against the BF16
release. CPU only.

### 1. The DiT build is at its format floor

`e_shipped` **0.0093617314051** against a deterministic reproduction's
**0.0093617314403** — equal to ten significant figures, on all 200 modules.
The shipped bytes are exactly what stock `TensorWiseINT8Layout.quantize` emits
at gs 256 from the release BF16.

**This is the DiT record that did not exist**, and its absence is what let
§10.5 reason from the encoder's. Note the DiT and the encoder differ in source
precision: the encoder needed **fp32** to reproduce (bf16 gave 7.9% differing
int8 values), the DiT reproduces from its own **bf16**, which is the only
precision the release ships for it.

### 2. Stochastic rounding costs exactly √2, and it is on the shipped path

```
e_stochastic / e_deterministic = 1.4142150      over 200 modules
                        sqrt(2) = 1.4142136      range [1.41388, 1.41453]
```

Seven significant figures. This is not a fit — it is the prediction from the
grid geometry: at fractional offset `p` in a cell, round-to-nearest has squared
error `min(p,1-p)^2` and stochastic rounding has expected squared error
`p(1-p)`; over `p` uniform that is `1/12` against `1/6`, exactly twice the MSE.

**It is on the shipped merge path.** `comfy/model_patcher.py:928` passes
`seed=comfy.utils.string_to_seed(key)` into `set_weight`, which calls
`requantize_from_float(..., stochastic_rounding=seed)`, reaching
`_round_int8`'s stochastic branch. So **any LoRA merged onto an int8_convrot
module carries √2 the requantisation error a deterministic bake would** — and
every stored-weight number this repo has published for the merge path was
measured deterministically, so they are the optimistic case.

Two consequences beyond this lane. `unmerged_blocks` avoids it entirely, which
makes the knob worth more than its measured 11.6%. And a patched model and an
unpatched one differ in their **weights** before a single sampler step runs,
which is upstream of `CLAUDE.md`'s different-sample rule rather than an
instance of it.

### 3. The rotation is worth 27%, and it does not finish on one module kind

`e_no_rotation / e_shipped` = **1.269** pooled.

But the rotation's job is to flatten each group before a per-output-row amax
sets the step, and it completes that job on three kinds and not the fourth.
Median row kurtosis after the shipped 256-wide rotation:

| kind | in_features | kurtosis after gs 256 | legal group sizes |
|---|---|---|---|
| attn.qkv_proj | 5376 = 2^8·21 | ≈ 0.00 | 4, 16, 64, **256 — capped** |
| mlp.fc1 | 5376 | ≈ 0.00 | 4, 16, 64, **256 — capped** |
| mlp.fc2 | 14336 = 2^11·7 | ≈ 0.00 | …, 256, **1024** |
| **attn.out_proj** | 7168 = 2^10·7 | **0.16-0.68 by depth** | …, 256, **1024** |

`_build_hadamard` demands a power of **4** dividing `in_features`, which is why
qkv_proj and fc1 cannot go wider: 5376 admits nothing above 256.

**out_proj's excess is now explained.**
`bench/results/2026-08-28_quant_hotspots_ref2va.json` listed under
`not_measured`: *"why the INT8 scheme treats out_proj worse. Outlier structure
the rotation does not spread is the obvious candidate and is untested."* It is
now tested. out_proj's outliers span wider than 256 channels; a 1024-wide
rotation reaches them.

### 4. `convrot_groupsize` 1024: one kind, ~10%, and it is a trade

| kind | gs1024 / gs256 |
|---|---|
| **attn.out_proj** | **0.898** (−10.2%) |
| mlp.fc2 | 0.998 (−0.17%) |

fc2 is already flat at 256 and gains nothing. out_proj's win is **strongest
shallow and never absent**:

| blocks | gs1024 win | e at gs 256 |
|---|---|---|
| 0-9 | 15.4% | 0.011436 |
| 10-19 | 10.4% | 0.010423 |
| 20-29 | 7.8% | 0.010032 |
| 30-39 | 7.2% | 0.009922 |
| 40-49 | 9.1% | 0.010258 |

Best block 3 at 20.8%, worst block 33 at 4.4%, and **49 of 50 blocks are at or
above 5%**. Block 0 alone said 20.8%, which is why a one-block read would have
overstated it twofold.

**The cost, and it is not free.** gs 1024 is supported —
`quantize_int8_convrot_weight` "falls back to explicit rotation plus row-wise
quantize" — but `_should_use_convrot_fused_kernel` and
`_should_use_convrot_dequant_kernel` both require `group_size == 256`. That
fused path combines the activation rotation, the row-wise quantisation and the
SwiGLU into one kernel, **per forward**. So this is a speed/accuracy trade on
one module kind and **any arm must be timed, not only scored**.

### 5. Two invariant violations, both named

Neither undermines the result, and both are reported rather than smoothed:

- **`gs 64` beats `gs 256` on exactly one module** — block 8 `mlp.fc1`, by
  3.8e-06 relative (0.04%). A near-tie, not a mechanism.
- **Rotation raised `group_disagreement` on two modules** — block 3 and block
  49 `attn.out_proj`, by ~0.5% and ~1.6%. This is a property of that metric
  rather than of the rotation: the Hadamard minimises spread **within** a
  group, and `group_disagreement` is a ratio **across** groups, which it does
  not directly optimise. The error improves on both.

### 6. What Tier 0 could not answer, and why

The record collapses per-group and per-channel structure to a single median
ratio per module. That is sufficient for the groupsize question, which the
error numbers answer directly, and **insufficient for the permutation lever**,
which needs the per-channel amax vector per block. It also keeps only
`rel_delta` of the seven quantities `analyze_quant_delta.stats` computes,
discarding `row_rel_p95` and `row_rel_max` — the statistics that characterised
out_proj in the first place — so it cannot say whether gs 1024 fixes the median
or the tail.

Both are `CLAUDE.md`'s *"a capture that reduces a grouping it did not record
has also lost its shape"*, committed here. A second pass should record the
vectors; it is cheap (in_features floats per module) and the shapes match what
#23 wants for the activation side.

---

## What follows

1. **#23 first, before any rebake.** Everything above is one of two roundings.
   A 10% weight-side win on out_proj may or may not survive to the output, and
   the fused-kernel cost is on the activation side where nothing is measured.
2. **The deterministic merge needs no further evidence.** √2 is arithmetic,
   confirmed on 200 real modules to seven figures.
3. **The permutation lever needs a capture that keeps its shape.**

## See also

- [`h3_dit_implementations.md`](h3_dit_implementations.md) §10.5 — where the
  withdrawn claim was, and the correction.
- [`comfyui_h3_t2va_trace.md`](comfyui_h3_t2va_trace.md) §1.7-1.8 — the two
  roundings, traced.
- [`../h3_pdd.md`](../h3_pdd.md) — the merge/un-merge trade and the all-blocks
  PDD record.
- [`../open_experiments.md`](../open_experiments.md) #23 — the runtime
  decomposition.
