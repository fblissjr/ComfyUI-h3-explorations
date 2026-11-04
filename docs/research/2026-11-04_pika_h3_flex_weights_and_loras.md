# Pika H3 Flex: what the weights are, how the ComfyUI LoRAs were made, and which to run on int8

2026-11-04. The owner asked four things: whether the early-access download differs from the public
`Pika-Labs/Pika-H3-Flex` repo, whether the public repo's base components differ from the local
`MiniMaxAI/MiniMax-H3` download, how kijai's Pika LoRAs were built and whether we could do better,
and what changes given that we run the pruned int8 convrot checkpoints. A dated record: what was
true on this day, not standing guidance. No render was made; every figure is on stored weights or
a microbenchmark.

Local folders are named by their download, under the owner's model storage: `Pika-H3-Flex-Early-Access-weights`,
`MiniMaxAI_MiniMax-H3`, `Kijai_MiniMax-H3/loras/Pika`, `Comfy-Org_MiniMax-H3`. Hub revisions read on
this day: `Pika-Labs/Pika-H3-Flex` at `3ff16644`, `MiniMaxAI/MiniMax-H3` at `42ed227e` (still its
`main`).

## Short answer

- **Early access and public: the same adapters.** The four weight files hash identically.
- **Pika's base components and MiniMax-H3: the same tensors.** They are resharded, with new metadata, two
  scheduler flags and seven appended special tokens.
- **Kijai's LoRAs** are Pika's adapters in ComfyUI layout, then a kohya-style dynamic SVD resize at `sv_fro`
  0.99 from rank 384. That keeps 0.99 of each module's norm, which leaves about 14% of each delta out.
- **Pika's own `comfy/loras/*_comfy_bf16` files** are exact up to bf16 rounding. Comfy-Org's
  `pika_h3_flex_*_v1_bf16` (the file its workflow templates load) is the same tensors.
- **On the int8 convrot checkpoints**, Pika's delta is a fraction of one int8 step, so a merge
  (`LoraLoaderModelOnly`, which Comfy's templates use) keeps little of it. `MiniMaxH3LoRABranch` keeps all of
  it. On the branch, rank costs little per step, so the full-rank file beats kijai's resize.

## 1. Early access against the public repo

`sha256sum` of the local files against the Hub API's LFS sha256 and the public `SHA256SUMS`:
identical for `fl2va/adapter.safetensors`, `ref2va/adapter.safetensors` and both
`comfy/loras/pika_h3_flex_{fl2va,ref2va}_comfy_bf16.safetensors`. Both `manifest.json` files and
`comfy/SHA256SUMS` are byte-identical. Both manifests pin `MiniMaxAI/MiniMax-H3` at `42ed227e`.

The public repo adds Diffusers-format base components (`transformer/`, `transformer_ref/`,
`text_encoder/`, `vae/`, `audio_vae/`, tokenizer, processor, schedulers,
`modular_model_index.json`), `examples/` for Diffusers, `LICENSE-CODE` and a Qwen3-VL licence. Its
README links ComfyUI's tutorial and three workflow templates, and Pika API extensions (hosted
1080p to 4K, audio stems, prompt rewriting). `comfy/README.md` drops the private-repo token wording
and the pinned revision `6f3072e1`, deferring to the inference repo's `release.json`.

## 2. Pika's base components against MiniMax-H3

The local `MiniMaxAI_MiniMax-H3` folder is revision `42ed227e`: all 279 of its
`.cache/huggingface/download/*.metadata` etags match the Hub at that revision.

| component | shards (MiniMax, Pika) | tensor names, dtypes, shapes | bytes |
|---|---|---|---|
| `transformer` | 14, 7 | identical | 336 tensors sampled, identical |
| `transformer_ref` | 14, 7 | identical | 335 sampled, identical |
| `text_encoder` | 14, 2 | identical | 116 sampled, identical |
| `vae` | 3, 2 | identical | 51 sampled, identical |
| `audio_vae` | 1, 1 | identical | same sha256 for the whole file |

Headers were read whole from both sides. Pika's side was fetched by HTTP range request, never downloaded
whole. Tensors up to 32 MB were compared whole, larger ones at three 1 MiB windows (start, middle, end), so
the byte column is a sample. Pika's own `text_encoder/config.json` says the same: "tensor payload unchanged
from the contributed Diffusers-format checkpoint".

What does differ:

- **Configs**: Pika's modification notice, `_diffusers_version` 0.42, and the text encoder's config rewritten
  for `transformers` 5.x (`rope_scaling` becomes `rope_parameters`).
- **Schedulers**: `scheduler/` and `audio_scheduler/` gain `shift_provided_sigmas: true` and
  `stochastic_sampling: true`. The shifts are unchanged, 12.0 for video and 3.0 for audio.
- **Tokenizer**: Pika appends seven tokens at ids 151669-151675: `<d>`, `</d>`, `<|cutoff|>`,
  `<|lyrics_start|>`, `<|lyrics_end|>`, `<|caption_start|>` and `<|caption_end|>`. Every shared token keeps
  its id, and ordinary text, including chat markers, Chinese and emoji, encodes to the same ids. Only those
  literal strings tokenize differently.
- **Coverage**: the local MiniMax folder also holds the native `FL2VA/` and `Ref2VA/` trees, `docs/` and
  `scripts/`, which Pika's repo does not.

## 3. How kijai's Pika LoRAs were made

`Kijai/MiniMax-H3` (Hub commit `587e806e`), `loras/Pika/pika_h3_flex_fl2va_v1_avg_rank_56_bf16` and
`pika_h3_flex_ref2va_v1_avg_rank_55_bf16`. Their `__metadata__` states the method:

- `conversion`: "qkv block-diag fused (contiguous q|k|v); mlp.fc1 swiglu halves swapped; alpha
  folded into lora_B; scale 1.0 baked as a flat multiplier". This is the same layout as Pika's
  `comfy/` files.
- `ss_training_comment`: "Dynamic resize with sv_fro: 0.99 from 384". This is kohya's per-module SVD resize:
  keep singular values until they hold 0.99 of the Frobenius norm. The ranks vary per module, from 2 to 180,
  each module with `alpha` equal to its rank.

The catch is in the definition: keeping 0.99 of the norm leaves sqrt(1 - 0.99^2) of it, about 0.14, out of
every module. A one-off script measured each Comfy module's delta `B A` against the delta rebuilt from Pika's
fp32 adapter. It computed the error from the small Gram products, never forming the full matrices, over all
208 Comfy modules, which cover all 312 original ones including the two token-refiner blocks. The result: about
0.14 relative error on every module of kijai's files, for both fl2va and ref2va. Pika's Comfy files came out at
bf16 rounding only, a maximum of 2.4e-3. The script was not kept. The 0.14 follows from the `sv_fro` value in
kijai's own metadata, so it can be re-derived without it.

The same tensor-level check with the tracked tool, for the record:

- `bench/compare_lora_files.py --a <Pika fp32 adapter> --b <Pika comfy bf16>` gives a bf16 cast on every sampled
  tensor, with the q|k|v bands placed as fused (`data/results/2026-11-04_pika_fp32_vs_comfy_fl2va.json`).
- `bench/compare_lora_files.py --a <Pika comfy bf16> --b <Comfy-Org pika_h3_flex_fl2va_v1_bf16 by URL>` finds all
  48 sampled tensors identical (`data/results/2026-11-04_pika_comfy_vs_comfyorg_fl2va.json`). Comfy-Org's file
  is 128 bytes larger, all in the header.

This matches what the repo found for kijai's FlashGen and PDMD resizes (`workflows/h3_config.py`,
`FLASHGEN_LORA` and `FLASHGEN_R64_LORA`; `CHANGELOG.md`, the PDMD conversion entry).

## 4. On the pruned int8 convrot checkpoints

### The layout holds on these checkpoints

`minimax_h3_{fl2va,ref2va}_pruned_int8_convrot` hold all four modules Pika targets (`attn.qkv_proj`,
`attn.out_proj`, `mlp.fc1`, `mlp.fc2`) in all 50 blocks and both token-refiner blocks. Pruning replaced the
AdaLN with an 8-column time table (`adaln_t_table`). Pika's adapters do not touch AdaLN, so FlashGen's
basis re-expression (`bench/convert_flashgen_lora.py`) is not needed here.

The row order matches the fused layout:

- **Token refiner.** It is stored bf16 in these checkpoints, so it compares directly. `qkv_proj` equals
  `cat[to_q; to_k; to_v]` exactly, and `fc1` equals `ff.net.0.proj` with its halves swapped.
- **The 50 DiT blocks.** They are int8 with `comfy_quant` `{"format": "int8_tensorwise", "convrot": true,
  "convrot_groupsize": 256}`, which rotates the input columns, so a direct comparison cannot match. Row norms
  survive that rotation. On blocks 0, 25 and 49 they match contiguous `cat[q;k;v]` to 1e-4 and miss per-head
  interleave by 0.2 to 0.5. `fc1` matches swapped to 1e-4 and misses unswapped by 0.08 to 0.38.
- **Which partition each checkpoint pairs with.** fl2va was compared against `transformer/`. The ref2va
  checkpoint's block 25 sits closer to `transformer_ref/` than to `transformer/` (row-norm error 0.00013
  against 0.00055). The partitions are close (`docs/research/h3_partition_distance.md`), so this is the weaker
  of the tests.

These were also one-off scripts. Making them a check is listed under "Not done".

### A merge keeps little of Pika's delta

`bench/probe_int8_lora_requant.py` on the fl2va checkpoint, blocks 0, 24 and 49, four modules each,
for Pika's Comfy file and kijai's (`data/results/2026-11-04_pika_flex_int8_lora_requant.json`):

- **The delta is small.** Its norm is 0.2% to 1.1% of the weight's, and its mean magnitude is 0.05 to 0.26 of
  the row's int8 step.
- **ComfyUI's merge keeps little of it.** That path is dequantize, add, then requantize with stochastic
  rounding. The surviving delta's cosine to the requested one is 0.14 to 0.67, with relative error 1.1 to 5.8.
- **An offline round-to-nearest bake does better but still loses most of it.** Cosine is 0.04 to 0.86, with
  relative error 0.56 to 1.85. So baking into a new int8 checkpoint (`bench/bake_pdd_checkpoint.py`) is not
  the way either.

`docs/research/merge_requantisation.md` is the background: a cost of known magnitude and unknown
effect on the picture. Comfy-Org's three Pika templates (`video_pika_h3_flex_{fl2va,ref2va,t2v}.json`, read
on this day) load these same int8 checkpoints through `LoraLoaderModelOnly`, so they are in this regime.
Pika's `comfy/README.md` says no end-to-end ComfyUI render was verified for the upload.

`MiniMaxH3LoRABranch` (`lora_branch.py`) leaves the int8 weight alone and adds `s * B (A x)` at the call.
That keeps the whole delta, in the compute dtype.

### On the branch, rank is cheap

`bench/profile_lora_branch.py`, blocks 0, 25 and 49, 32768 rows, on the idle 4090
(`data/results/2026-11-04_pika_branch_profile_*.json`). Pika's file was registered through a
wrapper that adds its folder to the `loras` path for the run, so nothing went into the models
folder.

| file | s/forward, branch | of which the host-to-device copy | with the factors resident |
|---|---|---|---|
| Pika comfy, full rank (qkv 384, others 128) | 1.04-1.05 | 0.16 | 0.88-0.89 |
| kijai, average rank 56 | 0.85-0.88 | 0.02-0.05 | 0.83-0.84 |

The floor does not depend on rank. qkv at rank 4 costs nearly what it costs at rank 81. The cost is the
output-sized `addmm_` and `fc2`'s recomputed swiglu (0.25-0.26 s/forward). Full rank adds about 0.2 s per
forward at this row count, most of it the copy. The tool scales linearly with rows by assumption, which it
states and does not check. A render's row count and the base forward's own time were not measured here, so
the share of a render is not stated.

## 5. Could we do better

- **Run Pika's own Comfy files (or Comfy-Org's, the same tensors) through `MiniMaxH3LoRABranch`.**
  That is the whole delta at bf16 for about 0.2 s per forward more than kijai's file. Nothing needs
  converting.
- **An fp32 conversion gains nothing on this path.** The branch casts the factors to the compute dtype
  (`lora_branch.py`, `_Branch._dev`).
- **Keeping the factors resident saves the copy, about 0.16 s per forward.** It holds about 2 GB of
  factors on a 24 GB card that ComfyUI does not account for, which is the trade `lora_branch.py` already
  declined.
- **The real cost is the rank-independent floor**, and it is shared by every LoRA on the branch. Fusing
  `fc2`'s branch into the int8 swiglu path would remove the recompute. That is a `lora_branch.py` question,
  not a Pika one.
- **A smaller file, if one is ever wanted:** a resize weighted by H3's own activations, graded on captured
  activations (`bench/grade_sage_on_capture.py`'s rule), would spend rank where the model uses it. Kohya's
  `sv_fro` spends it by weight norm alone.

## 6. The sampling contract

Pika's manifests: knots `1000/750/500/250/100/40/0`, video shift 12, audio shift 3, method
"renoise", fresh noise at every step. Comfy-Org's templates use `KSamplerSelect` `lcm` and
`ManualSigmas` `1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0`. Those are the knots through
`shift * t / (1 + (shift - 1) * t)` at shift 12. Core's `sample_lcm`
(`comfy/k_diffusion/sampling.py`) sets `x` to the denoised estimate and re-noises it to the next
sigma with fresh noise, which is the renoise method as Pika describes it. The templates pass one
sigma list, so audio gets the shift-12 sigmas, not shift 3 (0.9, 0.75, 0.5, 0.25, 0.111).
Whether that matters, and whether this pack's sigma-shift node covers it, was not checked.

## Not done

- No render. Every figure is on stored weights or a microbenchmark.
- The tensor-level comparisons in sections 2 and 3, the row-norm layout check in section 4 and the template
  read in section 6 were one-off scripts, not kept. If the Pika LoRAs are adopted, the layout check belongs in
  a tracked tool. `bench/compare_lora_files.py` could gain a delta-error mode, so a resize's loss is measured
  rather than inferred.
- No graph in this repo loads a Pika LoRA. A graph would apply it with `MiniMaxH3LoRABranch`, on `lcm` with
  the manual sigmas above, and would be generated by `workflows/build_workflows.py`.
