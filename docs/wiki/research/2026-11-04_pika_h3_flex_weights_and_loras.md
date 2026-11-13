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
  0.99 from rank 384. That keeps 0.99 of each Comfy module's combined Frobenius norm (about 14% combined error),
  but because `to_q`, `to_k` and `to_v` were fused into `qkv_proj` before the SVD, the larger pre-RMSNorm `Q` and
  `K` updates starve the unnormalized `V` slice in late blocks (`blocks.49` truncates `qkv_proj` from rank 384 to
  rank 4 on `fl2va` and rank 5 on `ref2va`, losing 94.8% of `to_v` on `fl2va` and 72.2% on `ref2va`;
  `data/results/2026-11-04_pika_kijai_component_errors.json`).
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
  folded into lora_B; scale 1.0 baked as a flat multiplier", along with `converted_layout: "comfyui_minimax_h3"`
  and `baked_scale: "1.0"`. Those exact strings come from Comfy-Org's `pika_h3_flex_fl2va_v1_bf16.safetensors`
  header (`data/results/2026-11-04_pika_comfy_vs_comfyorg_fl2va.json`), whereas Pika's own `comfy/loras/` header
  has `converted_for: "comfyui-minimax-h3"` and no `baked_scale` key (passing `bench/check_lora_alpha.py` because
  no `.alpha` tensors or metadata scale exist). Kijai therefore resized Comfy-Org's converted file rather than
  Pika's `adapter.safetensors`.
- `ss_training_comment`: "Dynamic resize with sv_fro: 0.99 from 384". This is kohya's per-module SVD resize:
  keep singular values until they hold 0.99 of the Frobenius norm. The ranks vary per module, from 2 to 180
  (`qkv_proj` spans 4 to 180, average 81.1 across all 52 blocks on `fl2va`; 5 to 179, average 80.6 on `ref2va`),
  each module with `alpha` equal to its rank.

The catch is in the definition and in running SVD after fusing `qkv_proj` (`data/results/2026-11-04_pika_kijai_component_errors.json`):

- **Combined Comfy modules (`qkv_proj`, `out_proj`, `fc1`, `fc2`)**: keeping 0.99 of a module's Frobenius norm leaves
  $\sqrt{1 - 0.99^2} \approx 0.141$ of its delta out. Measured across all 208 Comfy modules against Pika's fp32 adapter,
  the combined module error is 0.134–0.139 mean (0.140 max) on both `fl2va` and `ref2va`, while Pika's Comfy files sit at
  bf16 rounding only (max 2.4e-3).
- **Per-slice `to_q`, `to_k`, `to_v` inside `qkv_proj`**: in `comfy/ldm/minimax/model.py:165-178`, `q` and `k` pass through
  per-head RMSNorm (`q_norm`, `k_norm`) while `v` has no normalization. In late blocks, the pre-RMSNorm `to_q` and `to_k`
  deltas have much larger Frobenius norm than `to_v` (in `blocks.49` on `fl2va`, $\lVert\Delta W_Q\rVert_F = 7.02$ and
  $\lVert\Delta W_K\rVert_F = 12.42$ against $\lVert\Delta W_V\rVert_F = 1.33$, so `to_v` is only 0.85% of the squared Frobenius norm of
  `qkv_proj`). Because kijai fused `to_q`, `to_k`, `to_v` before running `sv_fro: 0.99`, `blocks.49.attn.qkv_proj` is
  truncated from rank 384 to **rank 4 on `fl2va` (rank 5 on `ref2va`)**, which meets the 0.136 combined budget on `qkv_proj`
  while losing **94.84% of `to_v` on `fl2va` (72.18% on `ref2va`)**. Across all 52 blocks of `fl2va`, `to_v` relative error
  averages 18.44% (max 94.84%), `to_k` averages 15.44% (max 24.69% at `blocks.23`), and `to_q` averages 12.47% (max 21.26%
  at `blocks.22`).
- **Per-component QR-SVD before fusion (`bench/convert_pika_lora.py`)**: running FP64 QR-SVD on `to_q`, `to_k` and `to_v`
  individually before block-diagonal `qkv_proj` fusion bounds every component (`to_q`, `to_k`, `to_v`, `out_proj`, `fc1`, `fc2`)
  across all 52 blocks to $\le 14.11$% max error at `sv_fro=0.99` (`pika_h3_flex_{fl2va,ref2va}_svd99_bf16.safetensors`,
  590 MB / 583 MB, average rank 58.62 / 57.89) and $\le 9.99$% max error at `sv_fro=0.995`
  (`pika_h3_flex_{fl2va,ref2va}_svd995_bf16.safetensors`, 898 MB / 891 MB, average rank 89.34 / 88.55).

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
  against 0.00055). The partitions are close (`docs/wiki/research/h3_partition_distance.md`), so this is the weaker
  of the tests.

These were also one-off scripts. Making them a check is listed under "Not done".

### A merge adds quantization noise larger than Pika's delta

`bench/probe_int8_lora_requant.py` on the fl2va checkpoint, blocks 0, 24 and 49, four modules each,
for Pika's Comfy file and kijai's (`data/results/2026-11-04_pika_flex_int8_lora_requant.json`):

- **The delta is small.** Its norm is 0.2% to 1.1% of the weight's, and its mean magnitude is 0.05 to 0.26 of
  the row's int8 step.
- **ComfyUI's merge (`LoraLoaderModelOnly`) preserves the delta in expectation but buries it in quantization noise.**
  That path is dequantize, add, then requantize with stochastic rounding (`lora_branch.py:11-16`). Stochastic
  rounding is unbiased in expectation ($\mathbb{E}[\operatorname{round}_{\text{stoch}}(W + \Delta W) - W] = \Delta W$), but because
  $|\Delta W|$ is only 0.05 to 0.26 of an int8 step, each merged weight receives unbiased quantization noise of RMS
  roughly 0.8x to 3.5x the delta's size. In a single realization, the surviving delta's cosine to the requested one is
  0.14 to 0.67, with relative error 1.1 to 5.8.
- **An offline round-to-nearest bake does better on cosine but loses most sub-half-step entries.** Cosine is 0.04 to 0.86,
  with relative error 0.56 to 1.85. So baking into a new int8 checkpoint (`bench/bake_pdd_checkpoint.py`) is not
  the way either.

`docs/wiki/research/merge_requantisation.md` is the background: a cost of known magnitude and unknown
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
(`comfy/k_diffusion/sampling.py:1083-1113`) sets `x` to the denoised estimate and re-noises it to the next
sigma with fresh noise, which is the renoise method as Pika describes it. Although `ManualSigmas` passes a
single shift-12 sigma list, `MiniMaxH3` (`ModelType.FLOW_AV` in `comfy/model_base.py:2195-2222` and
`ModelSamplingAV` in `comfy/model_sampling.py:338-358`, with `shift=12.0, audio_shift=3.0, audio_scale=4.0`)
carries the audio latent scaled by $y_a = (\sigma_v / \sigma_a) x_a$. Inside `MiniMaxH3Model.forward`
(`comfy/ldm/minimax/model.py:607-618, 643-647, 672-676`), the model un-carries audio via
$\text{carry} = \sigma_a / \sigma_v$ where `time_shift_sigma(sigma_v, 12.0, 3.0)` gives $\sigma_a = [1.0, 0.9, 0.75, 0.5, 0.25, 0.111111, 0.0]$,
feeds the shift-3 audio timestep $t_a = 1 - \sigma_a$ to the network, and transforms the output audio velocity so
that `CONST.calculate_denoised(sigma_v, out[1], y_a)` equals $\text{scale} \cdot \hat{x}_{0,a}$. When `sample_lcm` runs `noise_scaling(sigma_v_next, noise, denoised)`, the next step's $\text{carry}_{\text{next}} = \sigma_{a,\text{next}} / \sigma_{v,\text{next}}$
reduces via the Möbius identity $\frac{\sigma_a}{\sigma_v}\text{scale}(1 - \sigma_v) \equiv 1 - \sigma_a$ to
$x_{a,\text{next}} = (1 - \sigma_{a,\text{next}})\hat{x}_{0,a} + \sigma_{a,\text{next}}\epsilon_a$, driving video at shift 12
and audio at shift 3 simultaneously.

Pika's `numerics_policy` (`fp32_conditioning_v1` for `ref2va`, `bf16_conditioning_v1` for `fl2va`;
`coderef/Pika-H3-Flex/src/pika_h3/core/numerics.py:12-69`) does not change `prompt_embeds`, `condition_proj`,
`token_refiner`, or `AdaLN`—`prepare_conditioning` (`coderef/Pika-H3-Flex/src/pika_h3/core/numerics.py:60-69`) casts
`prompt_embeds` to `torch.bfloat16` under both policies. Instead, `_condition_entry_to`
(`coderef/Pika-H3-Flex/src/pika_h3/core/numerics.py:45-57`) controls only whether `LATENT_FIELDS` (`latents`,
`video_latents`, `audio_latents`) inside `keyframes` and `references` are kept in `torch.float32` (`fp32_conditioning_v1`)
or cast to `torch.bfloat16` (`bf16_conditioning_v1`) before anchor noise augmentation (`coderef/Pika-H3-Flex/src/pika_h3/core/wrapper.py:80-94`).
In ComfyUI (`comfy/ldm/minimax/model.py:499-500, 534-578`), `_cond_video_rows` and `_cond_audio_rows` always patchify,
noise-augment, and project reference/keyframe latents through `video_patch_proj` and `audio_patch_proj` in `torch.float32`,
which matches `fp32_conditioning_v1` on `ref2va`.

## 7. Where Pika's change lives: switches, tail, and `fl2va` vs `ref2va`

Reproduced by `bench/convert_pika_lora.py --analyze` and recorded in
[`data/results/2026-11-04_pika_lora_spectrum_and_adaln.json`](../../../data/results/2026-11-04_pika_lora_spectrum_and_adaln.json).
Across all 50 blocks, each module's rank-128 delta splits into two spectral regimes:

- **Top-3 singular directions ("switches", `r = 1..3`)**: hold 74% to 91% of each component's squared Frobenius energy
  (85.8% to 95.2% of its norm; stable rank $\lVert\Delta W\rVert_F^2 / \sigma_1^2 = 1.4\text{ to }2.7$). In `blocks.48-49`,
  `to_q` and `to_k` hold 94% to 98% of their energy in a single direction (`r = 1`), and `out_proj` and `fc2` reach 99%
  norm at rank 3 to 4.
- **Remaining 125 singular directions ("body/tail", `r = 4..128`)**: hold 9% to 25% of the squared energy (28% to 47% of
  the norm), spread across 60 to 82 directions in middle blocks (`blocks.20-39`) and in late-block `to_v` (`blocks.49`
  needs rank 67 for 99% norm). `svd995` keeps 79% to 90% of this tail energy.
- **`fl2va` and `ref2va` similarity and task difference**: the two adapters have cosine 0.951 to 0.998 (mean 0.990 to
  0.992) across every module. Inside the top-3 subspace (`r = 1..3`), `fl2va` and `ref2va` differ by only 2.3% to 2.9%
  of $\lVert\Delta W\rVert_F$; in the `r = 4..128` tail they differ by 12.0% to 13.8% (peaking at 20% to 31% in `blocks.47-49`).
  Truncating `ref2va` with `svd99` removes only a median 1.8% (max 6.6%) projection along $D = \Delta W_{\text{ref}} - \Delta W_{\text{fl}}$
  and pulls the truncated adapter slightly toward base MiniMax-H3 (`0`) rather than toward `fl2va` ($\cos(\widehat{W}_{\text{ref}}, W_{\text{fl}})$
  drops from 0.9924 to 0.9824, and $\lVert\widehat{W}_{\text{ref}} - W_{\text{fl}}\rVert_F / \lVert D\rVert_F$ grows to 1.49x; naive
  comparisons of $\lVert\widehat{W}_{\text{ref}} - \widehat{W}_{\text{fl}} - D\rVert_F / \lVert D\rVert_F$ are inflated in 151 of 300
  modules where independent `sv_fro` thresholds pick $k_{\text{ref}} \ne k_{\text{fl}}$).
- **Partial mechanism—alignment with `adaln_proj`**: in `comfy/ldm/minimax/model.py:224-255, 300-305`, `AdalnProj`
  produces separate `shift`, `scale` and `gate` vectors for 3 modalities (`video=0`, `text=1`, `audio=2`) from the
  8-column `adaln_t_table`, so each block's `shift(t, tag)` spans at most 27 dimensions out of 5376 (0.502% of
  $\mathbb{R}^{5376}$). Projecting the input singular vectors of `to_q`, `to_k`, `to_v` and `fc1` onto that 27-D
  `shift` subspace shows that $v_1$ carries 11.0% to 18.6% of its energy there on average (max 65.5%, 22x to 37x
  random chance), tapering across $v_2$ (13x to 25x), $v_3$ (9x to 16x), and $v_4..v_5$ (5x to 11x), while $v_6..v_{128}$
  sit near chance (1.14x to 1.22x). Similarly, in `blocks.48-49`, $v_1$ of `out_proj` and $v_1(\text{fc2})$ align at
  18x to 45x chance with the propagated subspaces $W_V(\text{shift}_{\text{msa}})$ and $W_{\text{fc1}}(\text{shift}_{\text{mlp}})$
  (tails at 1.14x); for `out_proj` this propagation is exact because row-stochastic attention weights preserve a
  per-modality constant offset in $V$, whereas for `fc2` it is a linearized correlation because SwiGLU multiplies the
  `up` half by a token-varying gate. That explains 11% to 19% (up to 65%) of each switch as a virtual per-modality,
  per-step bias read from `shift(t, tag)`. Whether the remaining 81% to 89% of $v_1$ reads a common token DC component in
  the RMSNorm output or token-varying features can be tested in the same single-pass subspace capture by recording the
  per-modality, per-step mean $\bar{z}_{\text{tag}} \in \mathbb{R}^{128}$ of $Z = X A_0^\top$ alongside the $128 \times 128$
  Gram matrices $G = Z^\top Z$, and only attribution renders can separate 6-step convergence from Pika's visual look.
- **Converter support for attribution**: `bench/convert_pika_lora.py` exposes `--switch-rank`, `--switch-scale`,
  `--body-scale`, `--peer-dir` and `--task-boost` so spike-only (`--switch-scale 1 --body-scale 0`), tail-only
  (`--switch-scale 0 --body-scale 1`), independent switch/body scaling, and task-difference boosting
  ($\Delta W + \beta(\Delta W - \Delta W_{\text{peer}})$) can be emitted without modifying runtime code.

## Not done

- No render. Every figure is on stored weights or a microbenchmark.
- The tensor-level comparisons in sections 2 and 3, the row-norm layout check in section 4 and the template
  read in section 6 were one-off scripts, not kept. If the Pika LoRAs are adopted, the layout check belongs in
  a tracked tool. `bench/compare_lora_files.py` could gain a delta-error mode, so a resize's loss is measured
  rather than inferred.
- No graph in this repo loads a Pika LoRA. A graph would apply it with `MiniMaxH3LoRABranch`, on `lcm` with
  the manual sigmas above, and would be generated by `workflows/build_workflows.py`.
