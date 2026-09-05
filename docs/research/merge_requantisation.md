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

> **A fourth was added 2026-08-31** and it is not in the table below, because
> it is measured against a different reference and belongs to
> [`quant_levers.md`](quant_levers.md) §7: `e` against `W_release + d`, the
> weight an unquantised run would use, under BOTH rounding regimes. It is the
> one that shows the shipped merge costs **+40%** rather than the +11.6% quoted
> from a round-to-nearest reproduction. Everything below is unaffected — it is
> relative to the dequantised shipped weight, as the next paragraph but one
> says.


`bench/measure_merge_noise.py` → `bench/results/2026-08-31_merge_noise.json`.
CPU only, fixed seed, `int8_convrot` at the layout's own group size.

**Read `blocks_sampled` in that record before quoting it.** It is written by
every invocation, including a `--stride` shape check, so the file name does not
tell you the coverage and the record is the only thing that does. The figures
in the table below are corroborated by two fuller sweeps named beneath it.

**And note what `check_doc_links.py` does not do**: it resolves a citation
against the FILESYSTEM, not against git, deliberately, so that new untracked
work is not a false red. A committed document citing an UNTRACKED file
therefore passes green and is broken for anyone who clones. That happened to
this file on 2026-08-31 and was caught by a human reader, not by a check.

  delta / step        `rms(d)` against ONE quantisation step, `2 * mean(scale)`
  realised            `<Q(W+d) − Q(W), d> / <d, d>` — the fraction of the
                      update landing along its own direction
  noise / |d|         `‖Q(W+d) − Q(W) − d‖ / ‖d‖` — what rides along

**`W` here is the DEQUANTISED SHIPPED WEIGHT, not the BF16 release.** Every
figure below is relative to what the int8 file already holds, so the base's own
quantisation error is outside the frame by construction. The spelling does not
make that obvious and it is the first thing to check before comparing these
numbers against anything measured against the release.

**Two spellings are in use across the two lanes and they agree.** This file
uses `‖Q(W+d) − Q(W) − d‖`; `quant_levers.md` uses `‖Q(W+d) − (W+d)‖`. They
differ by `‖Q(W) − W‖`, the base's requantisation residual, which is NOT zero —
measured at 1.2e-04, 6.1e-05 and 8.4e-06 of `‖d‖` on three modules, moving the
final statistic by at most 1.2e-06. So the agreement is real and the reason
sometimes given for it — that `Q(W) == W` because `W` is already dequantised —
is very nearly true rather than true.

**What the residual is, and the threshold it sets.** The requantisation IS
idempotent at the INTEGER level: re-quantising `W` deterministically reproduces
the shipped int8 exactly, **0 differing values** on every module tested. So the
residual is not a rounding disagreement — it is the fp32 un-rotate/re-rotate
round trip, and it is near-constant as a fraction of the weight,
`‖Q(W) − W‖ / ‖W‖` measured at 3.71e-07, 3.74e-07, 3.74e-07 and 3.91e-07 across
four modules spanning depth and all four kinds. That turns the scaling into a
number:

    contamination  ≈  3.7e-07 / r        where  r = ‖d‖ / ‖W‖

    PDD    r ≈ 0.0074   ->  ~5e-05     predicted 1.18e-04 vs measured 1.19e-04
    turbo  r ≈ 0.00038  ->  ~1e-03     (the four-module fit is within ~5%)
    r ≈ 3.7e-05         ->  ~1e-02     one percent of the statistic

**A LoRA whose delta is below roughly 4e-05 of the weight puts one percent into
this statistic**, and below that the two spellings stop agreeing usefully. Both
shipped LoRAs are two to three orders of magnitude clear of it. Check `r`
rather than inherit the equivalence. Mechanism and constant from the PDD lane,
verified here on a fourth module it did not test.

**The three rank the arms differently and that is the finding**, recorded as a
rule in [`../checks.md`](../checks.md): stored-weight distance prefers the arm
that does nothing, realisation prefers the arm that adds noise, and only
`noise/|d|` sees the third failure.

**All 200 modules per arm — 50 blocks x 4 kinds, no subsampling.**

| arm | delta/step | realised | noise/&#124;d&#124; median | min | max | noise > update |
|---|---|---|---|---|---|---|
| PDD fl2va (ships) | 0.07557 | 1.0000 | 1.981 | 0.274 | 3.121 | 194/200 (97%) |
| PDD ref2va (ships) | 0.08284 | 1.0000 | 1.848 | 0.239 | 3.209 | 193/200 (96%) |
| turbo fl2v | 0.00232 | 0.9991 | **12.156** | 3.107 | **27.313** | **200/200** |

Per kind, and **the ordering is partition-dependent**, which no subsample
showed: fl2va runs `fc2` 2.222 > `fc1` 2.038 > `out_proj` 1.978 > `qkv` 1.424,
while ref2va runs `out_proj` 2.066 > `fc2` 2.051 > `fc1` 1.824 > `qkv` 1.430.
Only `qkv_proj` being the least affected is common to both.

**The ranking is denominator-dependent, and the two denominators disagree.**
Everything above is the perturbation against the DELTA. Against the WEIGHT, on
the same 200 modules per arm:

| arm | &#124;d&#124;/&#124;W&#124; | noise/&#124;d&#124; | **noise/&#124;W&#124;** | worst module |
|---|---|---|---|---|
| PDD fl2va | 0.00476 | 1.981 | **0.921%** | 1.240% |
| PDD ref2va | 0.00511 | 1.848 | **0.947%** | 1.285% |
| turbo fl2v | 0.00014 | 12.156 | **0.172%** | 0.495% |

**Turbo looks 6x worse than PDD against the delta and is 5x better against the
weight.** Both are true. Turbo's delta is 34x smaller relative to the weight,
so twelve times a very small thing is still small, while twice PDD's larger
delta is not. The alarming 12x is substantially an artifact of dividing by a
tiny denominator.

**The reference point that makes these interpretable** — the PDD lane's, and
the best framing produced today — is what the checkpoint ALREADY carries.
Base int8 error against the BF16 release, verified here over the same 200
modules: median **0.910%**, range 0.881–1.256%. So:

    PDD merge noise      0.921% of the weight
    base int8 error      0.910% of the weight   <- already there, before any LoRA

**The merge injects an error term about the size of the entire int8
quantisation.** Against the release that is the **+11.6%** on total
stored-weight error the lever inventory already reported (0.936% -> 1.045%).

Three true sentences about one fact — "twice the update", "0.92% of the
weight", "+11.6% on the total" — reading as alarming, negligible and modest.
**Quote the +11.6% against the release**: it answers "how much worse is the
merged model", which is what a reader is asking. The 2x answers "how faithfully
is the update delivered", which is narrower and is what both lanes led with all
day.

Which denominator matters depends on the question, and **the output impact,
which would settle it, is unmeasured.** So neither is "the" answer and neither
should be quoted alone. This is the third
metric in this file to rank these arms differently, after stored-weight
distance and `realised`; `../checks.md` carries the rule.

Round-to-nearest, which does not ship, realises 0.467 mean / 0.020 worst on PDD
and 0.025 mean / 0.0001 worst on turbo — it discards a sub-step update rather
than adding noise to it.

**What subsampling cost, measured.** An earlier 100-module half of the PDD
fl2va arm gave median 1.979 against the full 1.981 — the centre to three
decimals — but a minimum of 0.877 against the true **0.274**, a 3.2x error on
the low tail, and 99% against 97% above 1.0. The turbo maximum moved 24.889 ->
27.313 the same way. **Subsets preserved the centre and missed the tail**, in
both directions and on every arm, which is why this table is not subsampled and
why the smaller samples earlier in the day were optimistic in a structured
rather than a random way.

A separate 100-module sweep of PDD fl2va across all 50 blocks put
`noise/|d|` above 1 on **99 of 100 modules**, median 1.979, range
0.877–3.121, and ordered the kinds `mlp.fc2` 2.207 > `mlp.fc1` 1.924 >
`attn.out_proj` 1.833 > `attn.qkv_proj` 1.534.

**A separate implementation in the PDD lane covered all 200 modules** and
agrees in the third decimal: median 1.981, mean 2.002, above 1.0 on 194 of 200
(97%), same kind ordering (`fc2` 2.222 > `fc1` 2.038 > `out_proj` 1.978 > `qkv`
1.424), and RTN realised 0.341 mean with a worst module of 0.0043.
`quant_levers.md` and `bench/results/2026-08-31_merge_realisation_pdd_all_blocks.json`
are that record. Two implementations, two subsets, agreement to the third
decimal — which is what makes these quotable at all. **Both lanes' small
samples were optimistic in the same direction**, because the two module kinds
easiest to sample are the two least affected.

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

Across all 200 modules, not a sample — **every summary statistic is identical
to the digit**, medians, minima, maxima, counts and all four per-kind medians.

| varied | result |
|---|---|
| base pruned vs unpruned | **no difference.** median 1.981, min 0.274, max 3.121, 194/200 on both. Pruning touches AdaLN; only the backbone merges |
| PDD variant `adaln2688` vs shipped | **no difference**, same five figures again. The variants differ in the adaln/head payload, which does not merge |
| fl2va vs ref2va partition | **a real but small difference** — 1.981 against 1.848 — and a DIFFERENT per-kind ordering. The one thing here that is not invariant |

So "which PDD file" and "which base" do not change the merge cost. **Which
node does**, because that decides whether a weight patch happens at all.

---

## 4. Out of scope, with the check that says so

**VSA, as shipped.** `minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot`
carries **0 `lora_*` keys** and 150 `to_gate_compress` keys — a full baked
checkpoint. **Corrected 2026-08-31:** this said "516 block weights", which
conflated two populations and read as a contradiction beside other counts. The
disambiguated figures, verified independently in two lanes: **500
`blocks.*.weight` (the DiT) plus 16 `token_refiner.blocks.*.weight` = 516**,
against 524 `.weight` keys in total. It is never merged and never requantised at
load, so nothing in this file applies to it. **A future VSA shipping as a LoRA
would inherit all of it**, which is worth knowing before that choice is made
rather than after.

**PDD's output heads.** `add_object_patch`, applied at the call. Unaffected.

**`mlp.fc2.forward` is never called on the shipped INT8 path.**
`comfy.ops.linear_input_act` owns it for the SwiGLU fusion, and that helper
ignores both `pre_quant_scale` and `_full_precision_mm`. So anything that
patches a LINEAR's forward silently misses fc2 and looks clean across the other
three kinds — the same reachability fact behind the 2026-08-30 `unmerged_blocks`
defect that dropped fc2. **Checked for this pack: nothing here patches a
linear's forward.** `exact_blocks.py:155` patches `blocks.{i}.attn.forward`,
and `sol_attn_h3.py:870` only COMPOSES with patches whose owner segment already
contains `attn`; the VSA node replaces a whole DiT block, which reaches fc2
through the ordinary call. Recorded because the trap is one lane over and the
next person to add an object patch here will not know it.

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
- **Whether the per-module stochastic values anywhere in this lane are the ones
  a GPU load produces.** They are not, and this was checked rather than
  assumed on 2026-08-31: `comfy_kitchen`'s registry resolves
  `quantize_int8_convrot_weight` to the eager implementation on CPU and to a
  CUDA one on GPU, and on the SAME seed the two draw different noise — about a
  third of the int8 codes differ by one step. They agree on magnitude to
  within 0.01% and both sit at exactly √2 of round-to-nearest
  (`cross_backend` in
  [`../../bench/results/2026-08-31_merge_rounding_regimes.json`](../../bench/results/2026-08-31_merge_rounding_regimes.json)).
  **So every stochastic MEAN in this lane is the shipped path's and every
  per-module stochastic figure is one draw from it.** Nothing quoted here
  changes; what changes is what a single row may be said to be.

---

## 7. The live server's path, read 2026-09-05

Sections 1 to 3 measure the stored weight after `patch_weight_to_device`. The
live server does not take that path for most modules, and the difference
matters for what the numbers above mean at render time. Read from ComfyUI
at the commit the launcher runs, not measured:

- Under dynamic VRAM loading (the server log's "prepared for dynamic VRAM
  loading ... N patches attached"), `comfy/model_patcher.py::ModelPatcher.load`
  attaches a `LowVramPatch` weight function to every patched weight and calls
  `patch_weight_to_device` only when `force_patch_weights` is set or the
  patch changes the weight's shape. So the LoRA is realised at cast time, per
  call, and the stored int8 weight is never rewritten.
- At cast time on a quantized weight, `comfy/ops.py::resolve_cast_module_with_vbar`
  dequantises, applies the patch functions in the compute dtype, and then,
  because the quantized linear op asks for `want_requant=True`
  (`comfy/ops.py`, the `CastBiasWeightContext` call in the quantized op's
  forward), calls `requantize_from_float(scale="recalculate",
  stochastic_rounding=seed)`: the delta goes in exactly, then the sum is
  rounded back into a recalculated int8 grid so the fused int8 kernel can
  run. The rounding is seeded from the module key, so it is deterministic per
  module and per render.
- The fp32-preserving path that `comfy/ops.py::cast_bias_weight` takes for a
  weight with functions (dequantise, apply, return the bf16 weight, no
  requantisation) is what a non-quantized op runs. On `int8_convrot` it does
  not apply.

Consequences. The mechanism measured in section 2 (a merged delta moves the
quantisation grid and the stored-weight error rises with strength) is the
same mechanism the live server exercises, transiently and per cast, so the
record's direction transfers; its magnitudes were taken on the stored merge
and are not the live per-cast values, which this document does not measure.
A sister node's claim that the cast-time path "preserves the delta in fp32
without requantising" (`ComfyUI-MiniMax-H3-Turbo`, its `_int8_fused_fc2`
docstring) holds on bf16 and fp8 bases and not on this checkpoint. And the
one module class no runtime path rescues on int8 is `mlp.fc2`: our
`unmerged_blocks` cannot reach it (`pdd_lora.py`, `BACKBONE_KINDS`), and a
bypass forward hook on it never fires because the fused int8 path calls the
kernel on the weight directly (the same finding, made independently by that
node). The bake fixes fc2 for every LoRA at once; the only runtime
alternative is an fc2 forward that runs dequantised in bf16 at a per-call
cost, which nothing here has built or priced.

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
- The headline was then quoted from a **round-to-nearest** reproduction while
  the shipped path rounds stochastically, on the reasoning that RTN is "its
  expectation". RTN is the expectation of the WEIGHT, not of the ERROR, and a
  distance statistic cannot borrow that identity — corrected to +40% the same
  evening ([`quant_levers.md`](quant_levers.md) §7). **The arithmetic was never
  wrong**: a second implementation reproduces all three published RTN means
  bit-identically. The REGIME LABEL was, and it sat in the record's own
  `rounding` field where it read as a disclosure rather than a claim.

**Every one of those was caught by a second IMPLEMENTATION, never by a second
reading.** Two lanes wrote the statistics independently and agreed on the turbo
worst case to four figures (26.606 against 26.6) and on PDD's RTN worst case to
four (0.0199 against 0.01988). That agreement is what makes the numbers
quotable, and it is recorded as a rule in [`../checks.md`](../checks.md).
