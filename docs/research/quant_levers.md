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
| ~~deterministic merge~~ | ~~replaces the merge path's stochastic rounding with round-to-nearest~~ | **WITHDRAWN 2026-08-31, hours after it was written** — it makes things worse, see §2 | — |
| **`convrot_groupsize` 1024** | a wider Hadamard, spreading outliers further before rounding | **measured on the weight side**, and it is one module kind | a CPU requant of 50 tensors, and a timing arm |
| ~~`full_precision_matrix_mult`~~ | ~~per-module escape to a bf16 matmul~~ | **WITHDRAWN 2026-08-31 — INERT on H3, and it never killed both roundings.** See §3 | — |
| **SmoothQuant via `pre_quant_scale`** | migrates activation outliers into the weight | **reachable on ordinary linears, INERT on `mlp.fc2`** (§3), gated on #23 | the scale as a module attribute (`add_object_patch` → `comfy.utils.set_attr`) **plus** a rebaked weight, **plus** a route into the fused fc2 helper |
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
The shipped file reproduces from stock `TensorWiseINT8Layout.quantize` at
gs 256 from the release BF16, **to within a handful of ties** — not
byte-identical, and the distinction is kept deliberately: see the table below,
where a few values per module still differ at max abs 1, roughly one in twenty
million.

**This is the DiT record that did not exist**, and its absence is what let
§10.5 reason from the encoder's.

**Verified at the byte level 2026-08-31, after a peer session could not
reproduce it.** Comparing error norms is not comparing bytes, and the original
version of this section claimed the stronger thing from the weaker
measurement. The direct test — shipped stored int8 against a deterministic
reproduction, integer for integer:

| module | int8 values differing | max abs |
|---|---|---|
| `blocks.0.attn.qkv_proj` | 6 of 115,605,504 | 1 |
| `blocks.25.mlp.fc1` | 13 of 154,140,672 | 1 |
| `blocks.49.attn.out_proj` | 5 of 38,535,168 | 1 |

Six, thirteen and five differing values is not zero. They are almost certainly
genuine ties resolving differently and they are not evidence of a different
recipe — but "exactly" would be a second unsupported byte claim in the
paragraph that exists to withdraw the first one, and it would not survive
someone re-running this on another box. **Reproduces to within a handful of
ties** is the claim.

**The reproduction must be computed in fp32, and that is the whole of the
disagreement.** Quantising the same bf16 release tensor without casting up
gives 8.663% / 8.727% / 6.734% differing on those three modules — the peer's
figures, reproduced to three decimals. The Hadamard rotation in bf16 loses
enough precision to change rounding decisions near ties. It is a **compute**
dtype effect, not a source dtype one.

**And that corrects a claim this section used to make.** It said the encoder
"needed fp32 to reproduce" while the DiT "reproduces from its own bf16",
presenting the two models as differing in source precision. They do not differ,
and the encoder record does not say they do:
`measure_int8_convrot_headroom.py` builds both arms from the **same** bf16
tensor — `weight.float().cuda()` against `weight.clone().cuda()` — so its
`reproduced_from_bf16_int8_values_differing_pct: 7.87` is the identical
compute-precision artifact under a field name that invites the misreading.
There was never an encoder/DiT divergence; it was invented from a variable
name. **So: both models' int8 builds reproduce exactly, in fp32, and neither
measurement says anything about the precision the vendor quantised from.**

### 3. Two levers were reachable only on paper, and an executable probe killed one

**Added 2026-08-31 after an external reviewer ran the code instead of reading
it.** Both rows below were entered in this file's inventory from a source
trace. Both were wrong, and the probe is four lines:

```
full_precision_mm changes the output?      False   max|d| 0.0
flag arm vs explicit dequant BF16 matmul:   rel L2 0.0086
fused fc2 helper honours the flag?          False
pre_quant_scale works on Linear.forward?    True (bitwise == scaling the input 2x)
fused fc2 helper honours pre_quant_scale?   False
```

**`full_precision_matrix_mult` is inert on H3, and it was mis-described
besides.** Two separate errors:

- Even working, it would not "kill **both** roundings". It bypasses the online
  activation quantisation and runs a BF16 matmul against the **already-rounded**
  int8 weight, dequantised. The weight rounding stays. That sentence was in the
  inventory and is withdrawn.
- It does not work here at all. Toggling it produces **bit-identical output**,
  and the arm sits 0.86% from an explicit dequantised BF16 matmul — i.e. it is
  still running the int8 kernel. The cause is that the quantised tensor's
  **logical** dtype is already `bfloat16`, so `cast_bias_weight` sees no dtype
  mismatch, hands the weight through unchanged, and `F.linear`'s
  quantized-tensor dispatch re-enters the int8 path. `comfy/ops.py:1150` does
  read the flag from the `comfy_quant` blob; reaching the flag is not the same
  as the flag doing anything.

**`pre_quant_scale` is live on ordinary linears and dead on `mlp.fc2`.** On a
normal `Linear.forward` it is bit-identical to scaling the input, so the
SmoothQuant runtime half is real. But `mlp.fc2.forward` **is not called on the
shipped INT8 path at all** — `comfy.ops.linear_input_act` owns it for the
SwiGLU fusion, and that helper ignores both `pre_quant_scale` and
`_full_precision_mm`. So SmoothQuant reaches three of the four kinds, and the
fourth needs a route into the fused helper.

**The fc2 fused path is a general trap, and this repo has already paid for it
once**: `unmerged_blocks` silently dropped `mlp.fc2` on 2026-08-30 for exactly
this reason — the object patch on `mlp.fc2.forward` never fired while the LoRA
keys had already left the weight patch. Anything patching a linear's `forward`
on this model must account for fc2 separately. **That includes
`docs/open_experiments.md` #23's observer**, whose planned 200 object patches
would have silently recorded 150.

Source reading put both rows in the table; running the code took one out and
halved the other. Third lever withdrawn today, and the only one that was never
live at all.

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

### 2b. And the lever that followed from it is WITHDRAWN

**This file said, hours earlier, that switching the merge to deterministic
rounding was "arithmetic, not a hypothesis" and needed no further evidence.
That is wrong, and it is wrong in the direction that matters.** The √2 stands —
it is a statement about **rounding a fixed tensor**. Merging is a different
problem, and carrying the constant across was the error.

The delta being merged is **smaller than one int8 step** (a peer session
measured `delta_rms / step` at median 0.0805 for PDD and 0.0050 for the turbo
LoRA, whose `alpha/rank` is 0.0625 against PDD's 1.0 — a 16x scale difference
that explains the gap). Round-to-nearest is a **biased** map: for a sub-step
delta most weights never cross a midpoint, so the update is simply thrown away.
Stochastic rounding is **unbiased** by construction, `E[Q_s(x)] = x`, so the
update lands in expectation.

Measured, `bench/measure_merge_realisation.py`, 20 modules per LoRA across
blocks 0/12/25/37/49
([PDD](../../bench/results/2026-08-31_merge_realisation_pdd.json),
[turbo](../../bench/results/2026-08-31_merge_realisation_turbo.json)).
`realised_along_d` is `<Q(W+d) − Q(W), d> / <d, d>`, the fraction of the delta
that landed; `noise_over_delta` is `‖Q(W+d) − (W+d)‖ / ‖d‖`, the error the
merge injects measured against the **update** rather than the weight:

| LoRA | arm | realised | noise / ‖d‖ |
|---|---|---|---|
| PDD, **all 200 modules** | deterministic | 0.341 mean, **0.0043 worst** | 0.86 |
| PDD, **all 200 modules** | stochastic *(ships)* | 1.0000 | **median 1.981**, max 3.121 |
| turbo, 20 modules | deterministic | **0.025 mean, 0.0001 worst** | 1.03 |
| turbo, 20 modules | stochastic *(ships)* | 0.9999 | **11.87 mean, 26.6 max** |

**On PDD, 194 of 200 modules carry more noise than update** (97%), and the
median is almost exactly 2x. By kind: `mlp.fc2` 2.222, `mlp.fc1` 2.038,
`attn.out_proj` 1.978, `attn.qkv_proj` 1.424.

**A 20-module sample was optimistic and a 5-block one more so.** This table
first carried 1.85 mean and 0.395 realised from 20 modules across 5 blocks; the
full sweep gives 1.981 median and 0.341 realised, and the worst realised module
moves from 0.020 to **0.0043**. A peer session running an independently written
statistic over a different 100-module subset got median **1.979** against this
file's 1.981 — the agreement is what makes either quotable.

**The two metrics rank the arms oppositely.** RTN has the lower stored-weight
distance *because it barely applies the LoRA*. Dose-responsive throughout:
`blocks.49.mlp.fc2` at 4.4% of the weight realises 0.995 under RTN,
`blocks.25.mlp.fc2` at 0.33% realises 0.020, and the turbo LoRA — whose deltas
average **0.038%** of the weight, `alpha/rank` 0.0625 against PDD's 1.0 —
realises 0.025.

So this is **not a turbo-only effect and not a lightx2v property**. The
governing variable is delta magnitude against the quantisation step, and it is
one continuous relationship with turbo at the low-delta end (median delta/step
~0.005, noise ~12x) and PDD further along it (~0.08, noise ~2x) — not two
phenomena.

**And `realised_along_d` alone still concludes too early.** It says the shipped
stochastic arm is fine, because it is: 1.0000. But a sub-step delta is realised
as a sparse set of **full-step jumps**, so the direction is right and the
per-weight representation is not — on turbo the shipped merge applies the
update and injects **twelve times its magnitude in noise**, 26.6x on the worst
module. **Both merge arms are bad on turbo, for opposite reasons**, and each
metric here is blind to one of them.

The PDD arm was replicated independently by a peer session, written fresh
rather than importing this statistic: their `deterministic_min` is 0.0199
against this file's 0.01988, on a different subset. Their record is
`bench/results/2026-08-31_stochastic_rounding.json`, retracted and re-issued
because its original conclusion was the same wrong inference this file made.

**So a stored-weight metric rewards the arm that does nothing**, and this file
recommended that arm on the strength of one. ComfyUI's choice of stochastic
rounding in `set_weight` now looks deliberate and correct, and the proposal was
to revert it. This is the same failure the file's own scope line warns about,
committed inside the file that carries the warning — the tell was available
and not taken: a √2 in the weight domain licenses nothing about a merge until
you ask what the weight domain cannot see.

Credit where due: the sub-step observation came from a peer session's
independent measurement, not from re-reading this file.

### 2c. What survives

`unmerged_blocks` **avoids the whole question** — it never requantises, so
neither the √2, the discard, nor the full-step noise applies, and the delta
stays exact in bf16.

**It is now the main lever in this inventory rather than one of several**, and
that is a change of standing rather than of number: of the six levers listed
above, two have been withdrawn today and one is inert on a quarter of the
modules. Its value is far more than the 11.6% stored-weight figure implies and
is **per-LoRA** — on PDD the merge carries ~2x the update in noise on 97% of
modules, on turbo ~12x.

**Also relevant to what it is not needed for**: the shipped VSA artifact
(`minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot`) carries **0
`lora_*` keys**. It is a fully baked checkpoint, never merged and never
requantised at load, so it inherits none of this as shipped.

Its weight counts, stated unambiguously because a bare "block weights" figure
has already been read two ways: **500** `blocks.*.weight` (the DiT), **16**
`token_refiner.blocks.*.weight`, **524** `.weight` keys in total, plus 150 gate
keys and 250 `comfy_quant` blobs. A "516" in circulation is the two block
families summed and is not wrong, only ambiguous against the 500. A future VSA shipping as a LoRA would inherit
all of it.

**What it is NOT: the shipped turbo graphs are not rendering with a discarded
LoRA.** Discarding is the RTN failure, and RTN does not ship —
`comfy.utils.string_to_seed` returns a non-zero seed on all 200 module keys
(checked, minimum 12054335), so `_round_int8` takes the stochastic branch on
every module and the update is realised. The shipped defect is noise, not
absence, and it is the quieter of the two.

A patched model and an unpatched one also differ in their **weights** before a
single sampler step runs, which is upstream of `CLAUDE.md`'s different-sample
rule rather than an instance of it.

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
2. **The deterministic merge is withdrawn** (§2b). The √2 is real and the
   inference from it was not. What replaces it as the cheap lever is
   `unmerged_blocks`, which sidesteps the rounding entirely — and the open
   question is which LoRAs it helps most, since the loss scales with how small
   the delta is against one int8 step.
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
