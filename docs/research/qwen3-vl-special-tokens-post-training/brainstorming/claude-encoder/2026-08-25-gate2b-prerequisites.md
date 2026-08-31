# Gate 2B prerequisites: the storage axis and the kernel axis, measured

**Date:** 2026-08-25
**Status:** Deliverable for Codex acceptance. Not authority; not a launch
request; Gate 2B not started. No recipe, no modifier, no candidate directory,
no deployment change.
**Scope:** the two prerequisites Codex set after accepting Gate 2A
([`2026-08-25-gate2a-corrected-floor.md`](2026-08-25-gate2a-corrected-floor.md)):
BF16 inactive storage with FP32 active compute, and the four-step kernel
matrix for expanded-KV attention, each tested as its own axis before the two
were composed on the long rows.

## Headline

**MEASURED, storage axis.** `comfy_exact_bf16_store`, which keeps the weights
BF16 where they are stored and offloaded and computes every parameterised op
with a transient FP32 copy of its weight, is **bit-identical** to FP32-stored
`comfy_exact` at the raw layer-49 state on all four released-weight Gate 1B
fixtures: relative L2 0.0 on all, text and vision rows, identical hidden-state
hashes. Through the bridge it loads in 0.2 s, holds 62 GiB on CPU with nothing
on disk, and leaves `MemAvailable` at 118 to 120 GiB across the run, where the
FP32-resident arrangement left 5 GiB. This is the arrangement that can host a
modifier.

**MEASURED, kernel axis.** The KV expansion is not numerically free, as Codex
required it not to be assumed. On the two fixtures whose accepted
`comfy_exact` residual is small, expansion and kernel arithmetic each move the
layer-49 vision state by the same few 1e-4 to 1e-3 as that residual. On the
two fixtures Gate 1B already showed to amplify vision-side perturbations,
every kernel arm scatters by percent around deployed ComfyUI; an early-tap
control shows those arms at ~1e-3 from ComfyUI at layer 24 and their ordering
flipping by layer 49, so the layer-49 spread there is compounding rather than
a kernel ranking. At layer 24, expanded-KV efficient attention matches
grouped-query math against ComfyUI on both sensitive fixtures.

**MEASURED, composed.** Storage plus expanded-KV efficient attention runs the
**entire five-row primary population** (25,250 tokens, longest row 10,358)
through the real sequential path at **6.4 GiB allocated / 7.7 GiB reserved**,
and the 8,981-token stress row at 5.7 GiB, where grouped-query math failed the
fourth row at 22.5 GiB. Every step clean; mask omission proven in every step.

## Provenance

| commit | what |
|---|---|
| `471ab2d` | `comfy_exact_bf16_store` policy, `storage_dtype` / `storage_policy`, loaders switched, check arm |
| `456627f` | gate moved to the parameterised modules after the first bridge run failed; comparator `--field-under-test` |
| `1c8fe43` | storage-axis results |
| `998619c` | `h3_attention_kernel.py` switch, its check, `--attention` on the comparator and pilot |
| `7e13ce4` | kernel-axis and composed-path results; TF32 note |
| `d5d58d8` | early-tap control results |

Producers: [`h3_calibration_precision.py`](../../../../../bench/h3_calibration_precision.py),
[`check_calibration_precision_policy.py`](../../../../../bench/check_calibration_precision_policy.py),
[`h3_attention_kernel.py`](../../../../../bench/h3_attention_kernel.py),
[`check_attention_kernel.py`](../../../../../bench/check_attention_kernel.py),
[`compare_transformers_comfy_layer50.py`](../../../../../bench/compare_transformers_comfy_layer50.py),
[`pilot_sequential_feasibility.py`](../../../../../bench/pilot_sequential_feasibility.py).
Every report names the commit that wrote it. Fixtures are the four Gate 1B
rows, rebuilt from the accepted pool on the current tree; the long-row
population is the Gate 2A primary and stress bundles.

## 1. Storage axis

### The arrangement

Weights load and offload at BF16. `torch.nn.functional.linear`, `embedding`,
`layer_norm` and `conv3d` are patched, process-wide but gated by hooks on this
model's own Linear / Embedding / LayerNorm / Conv3d modules on the entering
thread, to compute with `weight.float()`; the embedding gathers BF16 rows and
upcasts the result rather than the 151,936-row table. The patch-embed conv is
kept FP32 at load (`_keep_in_fp32_modules_strict`, via `storage_policy`)
because transformers downcasts the vision input to that weight's dtype in two
places before any hook can see it; the policy refuses to run if it finds that
module BF16 or `visual.dtype` reporting anything but float32, and refuses at
the first op if any activation below FP32 reaches it (`PrecisionLeak`).

Why the functional layer, and why the modules rather than the root: Accelerate's
device hooks and compressed-tensors' offload wrappers both capture the original
`forward` when they wrap a module, so an instance- or class-level override is
never reached under one loader or the other; and the sequential pipeline never
calls the root forward, it executes traced subgraphs that call submodules
directly. The second point was found by the first bridge run under the policy
failing at the first linear, not by design.

### What the check holds it to

[`check_calibration_precision_policy.py`](../../../../../bench/check_calibration_precision_policy.py),
on a tiny full Qwen3-VL at the released shape ratios through
`from_pretrained`: load path applies the keep-FP32 set and nothing else;
tower-only use refused; bit identity to the FP32-stored arm at the last layer;
a directly called decoder layer computes in FP32 without the root forward;
functional patches and hooks restored, including after a raising forward; a
BF16 patch embed refused at entry; a BF16 activation injected before layer 0
refused at the first linear; the un-policied kept-FP32 model fails with the
library's own mixed-dtype error, which is the proof the patches are gone.

### Released weights

**MEASURED.** Layer 49, FP32-stored `comfy_exact` as reference, grouped-query
math on both sides:

| fixture | rows | relative L2 all / text / vision | hidden-state hash equal |
|---|---:|---|---|
| single-image 44x40 | 1,100 | 0.0 / 0.0 / 0.0 | yes |
| multi-image 18x18 x2 | 951 | 0.0 / 0.0 / 0.0 | yes |
| keyframe 48x84 | 2,007 | 0.0 / 0.0 / 0.0 | yes |
| mixed 84x48 + 22x22 | 1,746 | 0.0 / 0.0 / 0.0 | yes |

- [`2026-08-25_storage_axis_layer49_single_image.json`](../../../../../bench/results/archive/v2_encoder/2026-08-25_storage_axis_layer49_single_image.json)
- [`2026-08-25_storage_axis_layer49_multi_image.json`](../../../../../bench/results/archive/v2_encoder/2026-08-25_storage_axis_layer49_multi_image.json)
- [`2026-08-25_storage_axis_layer49_keyframe_only.json`](../../../../../bench/results/archive/v2_encoder/2026-08-25_storage_axis_layer49_keyframe_only.json)
- [`2026-08-25_storage_axis_layer49_mixed_keyframe_reference.json`](../../../../../bench/results/archive/v2_encoder/2026-08-25_storage_axis_layer49_mixed_keyframe_reference.json)

Because the arm is bit-identical to `comfy_exact`, it inherits the accepted
Gate 1B residuals against deployed ComfyUI exactly; the kernel-axis files below
reproduce them (5.9e-2, 4.7e-4, 2.9e-3, 5.1e-2 on vision rows).

### Through the bridge

**MEASURED**, primary escalation, grouped-query math held fixed
([`2026-08-25_gate2a_primary_bf16_store.json`](../../../../../bench/results/archive/v2_encoder/2026-08-25_gate2a_primary_bf16_store.json)),
beside the FP32-resident record:

| | FP32 storage | BF16 storage, FP32 compute |
|---|---:|---:|
| load | 15 s; 114 GiB CPU + 10 GiB disk (symlinked shards) | 0.2 s; 62.1 GiB CPU, nothing on disk |
| host RSS after load / peak during load | 108 / 122.6 GiB | 0.96 / 0.96 GiB (pages mapped on first touch) |
| `MemAvailable` after three rows | 5.8 GiB | 119.8 GiB |
| CUDA, 1 row | 5.32 / 5.38 GiB | 2.80 / 2.84 GiB |
| CUDA, 3 rows | 22.11 / 23.00 GiB | 21.20 / 22.82 GiB |
| 4 rows | OOM, call 12, 6,189-token row | OOM, call 12, 6,189-token row |
| 1-row step time | 62 to 213 s | 20 s |

The CUDA boundary did not move because the kernel did not: the 19.4 GiB warm
transient is the math backend's logit tensor either way. What moved is
everything the storage arrangement owns: the host, the load, and the
wall-clock. RSS reads 64 GiB during the run because the mapped BF16 pages
count as resident, and `MemAvailable` stays at 119 GiB because the kernel can
drop them; both figures are in the report and neither stands in for the other.

**TF32, for the precision record.** `torch.backends.cudnn.allow_tf32` is True
by default in both virtualenvs and ComfyUI's `ops.py` passes `allow_tf32=True`
to `cudnn_convolution` explicitly, so the patch-embed conv is not literally
FP32 under either implementation; matmul TF32 is off on both
(`get_float32_matmul_precision() == "highest"`). Consistent between the stacks,
therefore not a divergence, and now written beside the policy that patches
`conv3d`. Read from source; raised by the peer session's levers note.

## 2. Kernel axis

### The switch

[`h3_attention_kernel.py`](../../../../../bench/h3_attention_kernel.py)
fixes what transformers' `sdpa_attention_forward` sends to SDPA:
`grouped_query` leaves the library's `enable_gqa=True` decision alone;
`expanded_kv` forces `repeat_kv` so SDPA sees 64 KV heads and no `enable_gqa`.
Patched at the module-level helper, gated by this model's attention modules
on the entering thread, source-guarded on the two lines it depends on,
counting the calls it governed. Its check observes what actually arrives at
`scaled_dot_product_attention` under each kind, scopes to one model, restores,
and refuses a model without grouped-query attention.

### The matrix, in Codex's order

**MEASURED.** Storage held at `comfy_exact_bf16_store`; relative L2 at layer
49, vision rows / text rows. A = grouped-query math, B = expanded-KV forced
math, C = expanded-KV forced efficient, D = deployed ComfyUI.

| fixture | 1. A vs B, expansion (math fixed) | 2. B vs C, kernel (expanded fixed) | 3. D vs A | D vs B | D vs C |
|---|---:|---:|---:|---:|---:|
| single-image 44x40 | 8.1e-2 / 3.6e-4 | 1.1e-1 / 3.9e-4 | **5.9e-2** / 3.0e-4 | 2.2e-2 / 3.0e-4 | 9.0e-2 / 2.5e-4 |
| multi-image 18x18 x2 | 4.4e-4 / 3.4e-4 | 5.4e-4 / 3.6e-4 | **4.7e-4** / 3.8e-4 | 4.4e-4 / 4.1e-4 | 3.5e-4 / 1.7e-4 |
| keyframe 48x84 | 3.4e-3 / 2.9e-4 | 3.1e-3 / 2.9e-4 | **2.9e-3** / 2.6e-4 | 3.4e-3 / 2.8e-4 | 2.9e-3 / 1.5e-4 |
| mixed 84x48 + 22x22 | 1.0e-1 / 7.4e-4 | 1.2e-1 / 9.2e-4 | **5.1e-2** / 7.2e-4 | 1.1e-1 / 9.5e-4 | 5.8e-2 / 5.6e-4 |

The bold column reproduces the accepted Gate 1B `comfy_exact` residuals
exactly, which is the storage-axis identity showing up where it should.

Files, per fixture: `2026-08-25_kernel_axis_expansion_*.json`,
`_kernel_*.json`, `_vs_comfy_*.json` under
[`bench/results/`](../../../../../bench/results/).

Read across the rows:

- **Text rows are stable everywhere**, 1e-4 to 1e-3, under every arm.
- **On the two stable fixtures** (multi-image, keyframe), expansion alone and
  kernel arithmetic alone each move the vision state by the same few 1e-4 to
  1e-3 as the accepted residual, and C is as close to ComfyUI as A or closer.
- **On the two sensitive fixtures** (single-image, mixed), the arms scatter by
  2 to 12 percent, and not consistently: B is closest to ComfyUI on one and
  farthest on the other. Gate 1B had already recorded these rows amplifying
  a BF16 reduction-order change from 0.61 to 0.06.

### Early-tap control: compounding, measured

The same four arms at decoder layers 0 and 24 on the two sensitive fixtures
([`2026-08-25_kernel_axis_tap24_single_image.json`](../../../../../bench/results/archive/v2_encoder/2026-08-25_kernel_axis_tap24_single_image.json),
[`2026-08-25_kernel_axis_tap24_mixed_keyframe_reference.json`](../../../../../bench/results/archive/v2_encoder/2026-08-25_kernel_axis_tap24_mixed_keyframe_reference.json),
and the `tap0` pair).

**MEASURED**, relative L2 on vision rows against ComfyUI:

| fixture | tap | A grouped math | B expanded math | C expanded efficient |
|---|---:|---:|---:|---:|
| single-image | 24 | 9.6e-4 | 1.35e-3 | 9.5e-4 |
| single-image | 49 | 5.9e-2 | 2.2e-2 | 9.0e-2 |
| mixed | 24 | 1.24e-3 | 2.01e-3 | 1.23e-3 |
| mixed | 49 | 5.1e-2 | 1.1e-1 | 5.8e-2 |

At layer 24 all three arms sit at ~1e-3, with C and A indistinguishable; by
layer 49 the same arms are at percent level and their order has changed. That
is compounding through the last 25 layers, and it means the layer-49 vision
rows on these two fixtures cannot rank kernels. The text rows at layer 24 are
5e-5 to 8e-5.

**Tap 0 is not a valid comparison point for vision rows, and the record says
so.** All three arms differ from ComfyUI identically there (0.40 on vision
rows, 1.7e-4 on text). Transformers' hook on `layers[0]` reads the residual
before the DeepStack injection that follows decoder layers 0, 1 and 2
(`canonical/2026-08-24_calibration_input_seam.md` section 4), while the
ComfyUI arm's truncated stack returns it after; text rows, which DeepStack
never touches, agree. Any wrong-layer or early-tap comparison on this
instrument must tap past layer 2. The `tap0` files are kept as the record of
that, not as a divergence.

### What the kernel axis establishes, and what it does not

- Expanded-KV attention changes the arithmetic by the same order as the
  accepted `comfy_exact` residual on stable fixtures, and is not separable
  from grouped-query math against ComfyUI at mid-stack on the sensitive ones.
- It does **not** pass or fail a Gate 1B-style rule as written, because that
  rule compares arms at layer 49 and two of the four fixtures cannot rank
  kernels there. Whether that rule is applied at layer 49, at a mid-stack tap,
  or on stable fixtures only is Codex's call; the numbers for each reading are
  above.
- Nothing here bears on whether AWQ's per-channel statistics, which aggregate
  over rows and positions, are sensitive at this level. INFERENCE, not
  measured: the compounding that separates the arms at layer 49 is a property
  of the two rows, and calibration statistics are not a single row's layer-49
  state.

## 3. Composed path on the long rows

**MEASURED**, `comfy_exact_bf16_store` + `expanded_kv`, 64 layers through the
bridge, kernel selection left to `auto` (which the probe established selects
efficient for the expanded shape):

| population | rows | tokens | longest row | outcome | CUDA alloc / reserved | warm entry / transient | host `MemAvailable` | forwards | seconds |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| primary | 1 | 1,100 | 1,100 | completed | 2.80 / 2.84 | 2.63 / 0.74 | 119.4 | 130 | 10.2 |
| primary | 3 | 8,703 | 5,857 | completed | 4.53 / 5.54 | 3.18 / 1.90 | 119.4 | 390 | 41.4 |
| primary | 4 | 14,892 | 6,189 | **completed** | 4.99 / 5.97 | 3.61 / 2.01 | 119.0 | 520 | 65.4 |
| primary | 5 | 25,250 | 10,358 | **completed** | 6.44 / 7.73 | 3.94 / 3.36 | 118.2 | 650 | 116.1 |
| stress | 2 | 10,081 | 8,981 | **completed** | 5.68 / 6.77 | 3.46 / 2.91 | 119.0 | 260 | 53.6 |

- [`2026-08-25_gate2a_primary_bf16_store_expanded_kv.json`](../../../../../bench/results/archive/v2_encoder/2026-08-25_gate2a_primary_bf16_store_expanded_kv.json)
- [`2026-08-25_gate2a_stress_bf16_store_expanded_kv.json`](../../../../../bench/results/archive/v2_encoder/2026-08-25_gate2a_stress_bf16_store_expanded_kv.json)

The 10,358-token row's warm transient is 3.4 GiB where the math path would
allocate a 25.6 GiB logit tensor; the expansion's own cost (eight KV heads
becoming 64, nominally 0.6 GiB of extra K/V at that length) is inside that
figure. The switch governed 1,664 attention decisions over the run and
expanded every one; the cast counted 14,664 linears. Cache on CPU at 46 to
69 KB per token, mask omission proven in every step, CUDA back to 0.009 GiB
after every step, nothing staged.

**This is a floor, not a budget**: no modifier ran. It is the floor of the
arrangement Gate 2B would run on, which the FP32-resident floor was not.

## What this does not establish

- Acceptance of expanded-KV attention as the calibration arithmetic. The
  matrix is measured; the reading of it against a rule is Codex's.
- Anything about AWQ's increment in memory, time, or host state.
- Any render, fidelity or quality claim.
- The 2048-upscale row's standing: it now runs, and it remains evaluation-only
  by the accepted decision.

## Operational notes

- ComfyUI was stopped by its port owner before the bridge loads (owner's
  instruction of 2026-08-25) and restarted from `start.sh` in `default` mode
  afterwards; its queue was empty on both sides.
- The ComfyUI arm of the layer-49 instrument hit a CUDA OOM once on the
  tap-24 mixed fixture with the card free and completed with
  `--reserve-vram-gib 4`, the same remedy Gate 1B recorded for large
  fixtures.
- An untracked `transformers` symlink to a sister checkout appeared at the
  repo root during the session; it is not this lane's and was left alone.

## For Codex

1. **Storage axis: accept.** Bit-identical on four fixtures, the host freed,
   the load two orders of magnitude faster, the check watched failing.
2. **Kernel axis: decide the reading.** Options, all with numbers above: apply
   the Gate 1B rule at layer 49 (C fails on the two sensitive fixtures by a
   factor of 1.1 to 1.5 against A); apply it at a mid-stack tap past the
   DeepStack injections (C ties A on both); or apply it on the two stable
   fixtures only (C passes). The compounding evidence favours the second or
   third, and the choice is a rule, not a measurement.
3. **If the kernel is accepted, Gate 2B has its arrangement**: BF16 storage,
   FP32 compute, expanded-KV efficient attention, all 64 layers through the
   bridge, with the whole primary population measured at under 8 GiB on the
   card and 118 GiB of host available for the modifier's own state.
