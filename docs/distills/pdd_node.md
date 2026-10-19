# PDD: the node, the converter, and how they were checked

The detail under [`pdd.md`](pdd.md), which owns the contract and the traps. This page says how `MiniMaxH3PDDLoRA`
(`pdd_lora.py`), `pdd_math.py` and `bench/convert_pdd_lora.py` work, and what each was checked against. The dated
story of how each piece arrived is in [`../research/pdd/`](../research/pdd/README.md) and in
[`pdd_artifacts.md`](pdd_artifacts.md)'s changelogs; [`docs/research/pdd/2026-08-28_pdd_implementations.md`](../research/pdd/2026-08-28_pdd_implementations.md)
compares the implementations at length.

## What the authors ship

They ship 8 evaluations and no step knob. Read from `coderef/alibaba-pai_MiniMax-H3-Acc-LoRAs/`:

| where | what it says |
|---|---|
| `coderef/alibaba-pai_MiniMax-H3-Acc-LoRAs/minimax_h3_pdd.py::apply_pdd_lora` | `nfe = num_steps // block_size`, derived from the file's own config and returned to the caller; not a parameter |
| its `DEFAULT_PDD_CONFIG` | `pdd_num_steps: 32`, `pdd_block_size: 4`, so `nfe` is 8 |
| `predict_ref2v.py` | calls the pipeline with `num_inference_steps=nfe + 1` |
| `README.md` | "Official 8 Step Acc LoRA", compared against LightX2V's 4-step turbo |

That is a limit of the inference script, not evidence about the weights: the paper's abstract, its Table 1 and its
section 5 make variable NFE the designed feature ([`pdd.md`](pdd.md), "Which partition is legal").

Their reference consumer is Euler: `coderef/diffusers/src/diffusers/schedulers/scheduling_minimax_h3.py::step`
takes "one Euler (`eta = 0`) step", and the adapter defines the fused head as "the mean velocity of one block, which
an Euler step over the block boundaries consumes". `simple` is exact at 4 and 8 steps (it reads the 1,000-entry table
and both divide 1,000); `bench/check_distill_grid.py` grades PDD graphs against `pdd_math.block_bounds` on both
streams, taking the count from the sampler's own `steps` and treating a non-zero `nfe` as an override.

## `bench/convert_pdd_lora.py`

Everything decided once, offline. One file out per source: the backbone in ComfyUI generic-LoRA naming, the adaln
update in one form, the published head bank, the partition fingerprint and the probe. `--pruned` names the base and
so picks the adaln form (the curve-basis bake, or the 2688-dim pairs plus the `silu(t_emb)` grid); shipping both was
about 40% of the file dead in the configuration this repo renders.

Four backbone transforms, each verified numerically against the release weights (block 0 of the fl2va partition,
ComfyUI's own `TensorWiseINT8Layout.dequantize` against `MiniMaxAI/MiniMax-H3`), where the residual is int8
quantisation noise:

- q/k/v fuse into `attn.qkv_proj`: A concatenated, B block-diagonal, rank and alpha tripled together so the applied
  `alpha / rank` holds.
- `attn.to_out.0` to `attn.out_proj` and `ff.net.2` to `mlp.fc2`, renames.
- `ff.net.0.proj` to `mlp.fc1` **with the output halves swapped**: the release stores SwiGLU as `[value; gate]`,
  ComfyUI as `[gate; value]`. Unswapped measured 1.41 relative, swapped 0.009.
- `adaln_proj.linear` maps by name with no permutation.

The converter raises on any unconsumed source key, asserts the bank verbatim (`assert_bank_verbatim`, refusing a
delta-encoded re-upload by its values; `bench/check_pdd_bank_encoding.py` is the red proof and
`bench/results/2026-08-28_pdd_bank_encoding.json` the baseline), and asserts populations against the release's depth.
The adaln bake is solved against the base's own `adaln_t_table` and the `silu(t_emb)` grid is derived from the same
`--base` that supplies the fingerprint, because both are partition-specific. Which metadata fields each converter
version writes is [`pdd_artifacts.md`](pdd_artifacts.md) section 4.

## `pdd_math.py`

The schedule arithmetic, with no ComfyUI import, so it is testable without a server, and one copy, so the converter
and the node cannot disagree about a block. `pdd_time_grid` is the grid. `base_sigma` inverts the flow shift, which is
the only way from a sampler's sigma back to a grid index, since the grid is uniform in base sigma. `schedule_knots`
turns a sigma schedule into the grid points it lands on. `fusion_plan` is the paper's `D_k` over a span (it takes an
end index, because a schedule-derived block need not match its neighbours' width), `fuse_block` applies it, and
`fuse_heads` is the uniform case. `block_bounds` is the closed form for uniform blocks; `pdd_lora.emit_sigmas` is
`1.0 - block_bounds(...)`, and `bench/check_pdd_head_selection.py` asserts it agrees with `schedule_knots` to
`torch.equal` at every divisor.

## `MiniMaxH3PDDLoRA`

It goes where a LoRA loader goes, before the attention nodes, because it clones the ModelPatcher. With the heads on
it installs four object patches:

| patch | what it is for |
|---|---|
| `diffusion_model.forward` | observe `sample_sigmas` and the graph's shift, and delegate; installed unconditionally, so the head-free arm keeps the shift guard |
| `final_layer.forward` | bookkeeping: pick this step's block, then call the stock forward |
| `final_layer.video_out.forward` | the swap |
| `final_layer.audio_out.forward` | the swap, on the audio stream's shift |

`patch_heads=False` installs only the first. The swap patches the projections' `forward` rather than enlarging the
weight, so nothing it does outlives the graph in ComfyUI's model cache, and the modulation maths stays upstream's.
The `final_layer.forward` patch is arity-transparent, so it survives core's widened
`(x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts)` signature; `bench/check_pdd_head_selection.py`
section 3 asserts it against both. Sol composes with `.forward` patches whose owner segment contains `attn`
(`vendor/sol_attn_minimax.py`), so it leaves these alone.

**Where the step index comes from.** The vendor arms its heads from a forward hook that counts calls and wraps at
`nfe`; one extra evaluation (a CFG pass, a warmup, a compile trace) desyncs it silently. The node instead matches
`t_emb`, what the model was called with, against block-boundary embeddings built from the model's own arithmetic at
all 33 points of the grid, and indexes them by the knots it derives from the sampler's schedule, so a schedule that
changes mid-graph (`split_at` two-pass sampling) needs no rebuild. `FinalLayer.forward` receives separate rows for
video and audio, so the per-stream split falls out of the signature. Selection needs no tolerance; the tolerance that
remains asks only whether the render is on the fused schedule. An earlier selector recovered `t` from the 1025-row
curve table, which quantises to about 1e-3, and picked the previous block at two of eight steps.

**Fusing.** `_FusedHeads` fuses a span on first use and caches it: fp32 masters on CPU keyed by block, and one cast
copy per `(block, device, dtype)`, held in a closure the streaming loader never sees. A render visits at most `nfe`
spans, so this is the paper's "one fused linear per block held in memory", and the sampling loop is a dict lookup
after the first pass. The bank tensors are read at fuse time, so offload and reload are transparent.

**Sub-steps are redundant.** `MiniMaxH3ParallelHead.forward` can emit `p` velocity fields per forward and the vendor
only builds `p=1`. That is correct: every head in a block reads the same hidden state, so `sum(dt_n * v_n)` is
`sum(dt) * weighted_mean(v)`, measured at 3.7e-16 relative. Sub-steps would matter only under a stochastic sampler,
which is off-distribution.

**The partition fingerprint** (`h3_pdd.base_video_out`) is fp32 and bit-identical across pruned and unpruned and
across `int8_convrot` and `fp8_scaled`, over the six H3 checkpoints checked on 2026-08-26, so one value names the
partition for every variant. It is compared by distance, not hash, because ComfyUI casts on load; a sha256 of it
stays in the metadata as a label.

## Checked against

**The paper.** Section 3.1's fused layer `W_{n:n+L} = sum_k D_k W_k` with
`D_k = (t_{k+1} - t_k) / (t_{n+L} - t_n)` is `fusion_plan` term for term; Algorithm 1's consumer is evaluated at the
block-start time with no noise, which is what Euler plus boundary-matched selection implement. The paper recommends
holding one fused linear per block rather than an enlarged final layer, which `_FusedHeads` does.

**Conversion reproducibility** (2026-08-28,
[`data/results/2026-08-28_pdd_conversion_reproducibility.json`](../../data/results/2026-08-28_pdd_conversion_reproducibility.json)).
Re-converting from identical inputs gives all 730 tensors bit-identical, every metadata value, key order and
`data_offsets` identical, and a different file sha256: `safetensors` orders the header's metadata keys differently
per process. **A file hash is not a stable identity**; compare the content hash (sorted metadata pairs, then every
tensor's bytes in sorted key order), which the inventory records. The `2026-08-27_pdd_conversion_*.json` records
carry file hashes and will not reproduce. A delta-encoded bank would show in the row norms: on the artifacts here
every row sits within 1% of the others (min/row0 = 0.9906, all about 57).

**Kijai's conversions**, an independent conversion of the same weights (re-run with
`bench/compare_pdd_conversions.py`; records `bench/results/2026-08-26_pdd_conversion_*.json` and
`bench/results/2026-08-27_pdd_conversion_{fl2va,ref2va}.json`, which carry a sha256 of each input). Read 2026-08-27,
ref2va:

| | ours against his |
|---|---|
| `attn.qkv_proj`, `out_proj`, `mlp.fc1`, `mlp.fc2` | 0.0, except 6.7e-20 on qkv |
| his head bank against the published 32-stack | about 1e-10, video and audio |
| our bank against the published 32-stack | 0.0, weight and bias, both streams |
| fusing either bank, at 8 / 4 / 2 evaluations | about 1e-11 |

His pruned build projects the adaln update into the curve basis too, and the two solutions agree, each scored
against ground truth (ref2va, blocks 0 / 25 / 49: ours 7.8e-05 / 8.1e-05 / 2.8e-05, his 8.3e-05 / 7.8e-05 / 3.7e-05;
his column matches `data/results/2026-08-26_pdd_conversion_ref2va.json`). The gap is storage precision: he keeps
bf16 factors, we store the fp32 product.

His packaging changed on 2026-08-27: the head bank now ships as `lora_up` / `lora_down` / `reshape_weight`, which core
applies as `pad_tensor_to_shape(weight, reshape) + up @ down` with zero padding, so what is stored is the bank minus
the padded base head (`bench/compare_pdd_conversions.py::kijai_bank` reconstructs it). Under that encoding his
`strength` below 1.0 scales heads 1..31 from zero rather than from the base head; ours interpolates toward the base
head, and `ComfyUI-MiniMaxH3-PDD-Mamad8::blend_with_native` reaches our reading independently. Kijai re-uploaded all
four files at 2026-08-27T21:25Z (HF commit `f94b1bcc94`) with delta-encoded banks, which is what merged core expects;
the copies symlinked into this install predate that upload, so merged core decodes them to a roughly doubled head
([`../research/pdd/2026-09-25_upstream_pdd_comparison.md`](../research/pdd/2026-09-25_upstream_pdd_comparison.md)
section 1). No shipped graph names them.

**Core** (Comfy-Org/ComfyUI#15908, by the same author, merged 2026-08-29 as `2504e68d`). `FinalLayer.forward` takes
`n = video_out.weight.shape[0] // out_features`, keeps the original path at `n == 1`, and otherwise finds the step by
`argmin` against `sample_sigmas`, maps both ends back through `time_shift_sigma` to base-grid indices and fuses the
spanned heads with an `einsum` in every forward. Two differences from ours, neither a defect: it derives the block
from the sampler's schedule where we use `t_emb`, so it depends on `sample_sigmas` being present and on evaluations
only at scheduled sigmas; and it fuses per forward where we fuse each block once. Our files keep `video_out` its
original size, so core takes `n == 1` and our patches own the swap ([`pdd.md`](pdd.md), "Not enforced, and not
measured").

**`silveroxides/ComfyUI-UtilsCollection` (`UC_MiniMaxH3PDDAcc`)**, read at `5bac35be3d61`: a complete third
implementation, with the same four backbone transforms, explicit `.alpha` tensors and the same affine adaln solve.
Its guards that we adopted: a partial patch-key match raises, and a head-patch clash refuses. Its `SIGMAS` output is
the interface ours took on. It refuses an off-grid sigma by default; ours does not, because our blocks come from the
schedule itself. It restricts block sizes to the trained width and twice it, which nothing here has tested. Its
basis match is `torch.allclose(atol=1e-6)` against `adaln_t_table`, reported and not run here; our
`TABLE_TOLERANCE` must sit above the 1.6e-3 a bf16 cast costs. Ours is the only one of these that notices a Ref2VA file
on an fl2va checkpoint.

**`coderef/comfyui-minimax-h3-audio-T8/pdd_advanced.py`** is the strictest: `validate_pdd_sigmas` raises on anything
but exactly 8 evaluations on the official Euler/simple shift-12 schedule, to 5e-6. Its docstring gives the reason as
reproducing the Apache-2.0 reference exactly, a fidelity policy and not a measurement. On 4 steps: the authors expose
no knob, T8 refuses on that policy, UtilsCollection and Mamad8 allow it, and ours allows it inside the envelope. No row
is a rendered comparison. `coderef/comfyui-minimax-h3-audio-T8` also declines a backbone bake on fidelity grounds,
which the bake measurement in [`pdd.md`](pdd.md) answers: a bake from bf16 lands on the base checkpoint's own error.

## Borrowed, and from where

Credited at the point of use in the source as well.

- **The runtime adaln injection** reimplements `ComfyUI-MiniMax-H3-Turbo`'s `_inject_adaln_egrid` /
  `_make_adaln_forward`, including why it patches a `.forward` attribute rather than wrapping the module (a wrapper
  puts `.base.linear.weight` into the parameter tree, the streaming loader records that path, and by unload the path
  no longer resolves). Its bundled grid is fl2va-only; ours derives the grid per checkpoint, and our derived fl2va grid
  agrees with its bundled one to 0.0017 relative, bf16 storage noise.
- **The per-device head cache** is `ComfyUI-MiniMaxH3-PDD-Mamad8::PDDHeads.for_device`, an independent implementation
  of the same two-projection swap for a different PDD artifact family (displacement heads over a 256-interval bank).
  It selects its head from the sampler's sigma through a diffusion wrapper. Neither has been graded against the
  other here.

What that Mamad8 node teaches about a distillation LoRA on this checkpoint, read 2026-09-05:

- It applies the backbone at the call through ComfyUI's bypass injection (`BypassInjectionManager` with a
  `WeightAdapter`), with an in-place low-rank accumulate; this repo's equivalent is `lora_branch.py`, which also
  reaches `mlp.fc2` by patching the MLP's forward, since the fused int8 path never calls the `fc2` module.
- On the pruned checkpoint its current adaln injector looks each row up in the model's own `adaln_t_table`, as ours
  does, but it ships one fl2va grid for every base, which is wrong on ref2va. Derive, never bundle.
- PDD is schedule-pinned (exact evaluations, per-interval heads, a fixed partition, a shift check), where a turbo
  LoRA is a plain file. A general distillation-LoRA node is the injection layer plus the adaln reinjection, with the
  schedule pin and the heads as PDD's extras.
- Its debug wrapper reads one projection's forward owner on the first calls to say whether the LoRA is live, the
  same idea as `SageChainAssert`; a canary that cheap belongs on any injection path.
- Its user guidance (four to eight steps, strength one, over eight over-sharpens) is a prior, not a measurement here.
