# What happens to a LoRA when it is merged onto an int8 module

**A data and provenance record, not a lever inventory.** What was traced in
code, what was executed, what the numbers are, and what is not established.
[`quant_levers.md`](quant_levers.md) owns what to DO about any of it and is the
authority for the lever table; this file owns where the observations came from.
Where the two disagree on a number, re-run both producers rather than picking.

**Nothing here is a quality finding.** Every measurement is on STORED WEIGHTS.
No render was made, no output was compared, and the PDD LoRAs demonstrably
produce working video. This is a cost of known magnitude and unknown effect.

---

## 1. Traced: the path a merged LoRA actually takes

Read at the source and then confirmed by execution where noted. Line numbers are as of 2026-08-31. The `comfy_kitchen` rows name
MODULES rather than paths deliberately: it is an installed wheel, not a
checkout, and `import comfy_kitchen` is how to reach it — CLAUDE.md's rule
about not importing from `coderef/` applies.

| step | where | what |
|---|---|---|
| our node | `pdd_lora.py:1881` | `m.add_patches(loaded, strength)` — the backbone LoRA goes through ComfyUI's NATIVE weight-patch path |
| our node | `pdd_lora.py:1933, 2074, 2114, 2131` | the output heads go through `add_object_patch` — applied at the call, **never requantised** |
| core | `comfy/model_patcher.py:928` (and `:1669`) | `set_func(..., seed=comfy.utils.string_to_seed(key))` |
| core | `comfy/ops.py:1434` | `set_weight` → `requantize_from_float(..., stochastic_rounding=seed)` |
| kernel | `comfy_kitchen.tensor.base`, line 302 | `requantize_from_float` → `from_float`, preserving convrot options |
| kernel | `comfy_kitchen.tensor.int8`, line 128 | → `quantize_int8_convrot_weight(..., stochastic_rounding=...)` |
| kernel | `comfy_kitchen.backends.eager.quantization`, line 822 | `_round_int8`: `if stochastic_rounding is not None and > 0` → add RNG, floor; else round-to-nearest |

**So the backbone merge is dequantise → add → requantise, and the heads are
not.** That split is the single most important fact here and it is a property
of our node, not of the format.

**Round-to-nearest is unreachable on the shipped path.** Executed, not
inferred: over 466 module weight keys, `string_to_seed` returns a minimum of
**12,054,335** and **zero** keys yield 0, so every module takes the stochastic
branch. Any statement about RTN below is about a hypothetical arm.

---

## 2. Measured: three statistics, each blind to the next

`bench/measure_merge_noise.py` → `bench/results/2026-08-31_merge_noise.json`.
CPU only, fixed seed, `int8_convrot` at the layout's own group size.

  delta / step        `rms(d)` against ONE quantisation step, `2 * mean(scale)`
  realised            `<Q(W+d) − Q(W), d> / <d, d>` — the fraction of the
                      update landing along its own direction
  noise / |d|         `‖Q(W+d) − Q(W) − d‖ / ‖d‖` — what rides along

**The three rank the arms differently and that is the finding**, recorded as a
rule in [`../checks.md`](../checks.md): stored-weight distance prefers the arm
that does nothing, realisation prefers the arm that adds noise, and only
`noise/|d|` sees the third failure.

| arm | delta/step | realised (stochastic) | realised (RTN, hypothetical) | noise/&#124;d&#124; |
|---|---|---|---|---|
| PDD fl2va 8step | 0.0735 | 1.0000 | 0.467 mean, 0.020 worst | ~2.0 |
| turbo fl2v 8step | 0.0025 | 0.9991 | 0.025 mean, 0.0001 worst | ~12 |

A separate 100-module sweep of PDD fl2va across all 50 blocks put
`noise/|d|` above 1 on **99 of 100 modules**, median 1.979, range
0.877–3.121, and ordered the kinds `mlp.fc2` 2.207 > `mlp.fc1` 1.924 >
`attn.out_proj` 1.833 > `attn.qkv_proj` 1.534.

**The driving variable is delta magnitude against one step, not the LoRA's
provenance.** Across 72 modules of both artifacts the relationship is monotone
and continuous — one curve with the turbo LoRA at the low-delta end, not two
phenomena. The 16x gap between the two is their `alpha/rank`: PDD 192/192 =
1.0, turbo 24/384 = 0.0625.

---

## 3. Measured: what does NOT change the answer

Each varies exactly one thing against the shipped PDD arm. All identical to
four decimal places on the same modules, which is the point — a difference
would have been attributable.

| varied | result |
|---|---|
| base pruned vs unpruned | **no difference.** Pruning touches AdaLN; only the backbone merges |
| PDD variant `adaln2688` vs shipped | **no difference.** The variants differ in the adaln/head payload, which does not merge |

So "which PDD file" and "which base" do not change the merge cost. **Which
node does**, because that decides whether a weight patch happens at all.

---

## 4. Out of scope, with the check that says so

**VSA, as shipped.** `minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot`
carries **0 `lora_*` keys**, 516 block weights and 150 `to_gate_compress`
keys — a full baked checkpoint. It is never merged and never requantised at
load, so nothing in this file applies to it. **A future VSA shipping as a LoRA
would inherit all of it**, which is worth knowing before that choice is made
rather than after.

**PDD's output heads.** `add_object_patch`, applied at the call. Unaffected.

**`unmerged_blocks`.** Moves a backbone LoRA from a weight patch to a forward
patch, so it never requantises and sidesteps this entirely. Its value is
therefore per-LoRA and larger where the delta is smaller.

---

## 5. Not established

- That any of this is visible in a render. Nothing here was rendered.
- Any effect on the ACTIVATION rounding. `int8_convrot` is W8A8 and every
  number here is the weight side; `../open_experiments.md` #23 owns the other.
- Whether the `ref2va` PDD pairing behaves like `fl2va` beyond a shape check.
- Whether a LoRA applied through `unmerged_blocks` produces a different output,
  as opposed to different weights.

---

## 6. How the numbers here were corrected, twice each way

Kept because the corrections are the provenance, and because each was found the
same way.

- The first reading of the merged arm ranked the rounding modes **backwards**,
  from stored-weight error alone — a metric that rewards the arm that discards
  the update. Withdrawn the same day.
- The lever that followed from it — switch the merge to deterministic rounding
  — was withdrawn on the same evidence.
- `realised` then hid the third failure, and `noise/|d|` was added.
- A 10-module sample understated PDD's `noise/|d|` because it omitted the two
  worst module kinds; the 100-module sweep corrected it.

**Every one of those was caught by a second IMPLEMENTATION, never by a second
reading.** Two lanes wrote the statistics independently and agreed on the turbo
worst case to four figures (26.606 against 26.6) and on PDD's RTN worst case to
four (0.0199 against 0.01988). That agreement is what makes the numbers
quotable, and it is recorded as a rule in [`../checks.md`](../checks.md).
