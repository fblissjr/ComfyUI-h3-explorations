# What upstream says: the paper, Sol-Engine, Sol-H3, and the other packs

Last updated: 2026-09-25 (section "comfy-kitchen and core, 2026-09-25", with
ComfyUI's workflow templates; a dated note under sglang's SubBlock router);
2026-09-22 (section "comfy-kitchen, 2026-09-22": kitchen only,
core not re-read); 2026-09-19 (section "comfy-kitchen and core, 2026-09-19", with
dated notes on the PRs it found closed or merged); 2026-09-15 (the
comfy-kitchen section); before that 2026-09-11, when sglang's own Sol-Attn backend was added, the
comfy-kitchen snapshot moved forward, an
open core PR against core's Sol node was read, and a third ComfyUI Sol pack
was added from its README. Before that, 2026-09-10, when Sana's `sol-engine`
branch was re-read at `757d902` (the Sol-H3, Sol-H3-Spark,
`super_acceleration` and `RTX4090` packages), ComfyUI core's own Sol node was
added, two sglang SubBlock notes were recorded, and the comfy-kitchen snapshot
moved forward. The paper and the two original third-party packs are as read
on 2026-08-16. Renamed from
`sol_engine_reference.md` on 2026-08-19, when the paper was added and the scope
widened past one vendor's framework.

**This page states what other people do and claim. It asserts none of our
numbers, and it takes no decision.** When a comparison against what we run is
needed, it lives in the doc that owns our side of it: [`docs/SOLATTN.md`](SOLATTN.md)
for the knobs, the measurements, and what we would adopt from upstream (its
section "What Sana's newer H3 packages offer this card"), and
[`docs/morton.md`](morton.md) for token order. If this page and one of those
disagree about our configuration, they are right.

| source | how it was read | when |
|---|---|---|
| arXiv 2607.24027, the Sol-Attn paper | fetched: abstract, ablation summary, HTML v1. **Not a full end-to-end read** | 2026-08-16 |
| Sol-Engine's per-hardware H3 cells | source, `coderef/Sana` on `sol-engine`; first read at `6fb7eb1`, pointers re-verified at `757d902` | 2026-08-15, 2026-08-16, 2026-09-10 |
| Sol-H3, Sol-H3-Spark, `super_acceleration`, `RTX4090` | source at `757d902`: added by `9791888` (RTX4090, 2026-08-17), `2936c47` (Sol-H3, 2026-09-08), `8249b28` (its MXFP8, 2026-09-09), `757d902` (Spark) | 2026-09-10 |
| [`Efficient-Large-Model/H3-to-LTX-Latent-Adapter`](https://huggingface.co/Efficient-Large-Model/H3-to-LTX-Latent-Adapter) | the Hugging Face model card only | 2026-09-10 |
| ComfyUI core's `BlockSparseAttention` | source in the ComfyUI checkout at `1f641fd9` | 2026-09-10 |
| sglang's SubBlock router | source at `ffe98a4279`; `sage_fp8`'s arch set re-read at `2f5c9ac43d` | 2026-09-10, 2026-09-25 |
| sglang's Sol-Attn backend | source at `593c7a900d`, with its attention-backend doc and H3 cookbook page | 2026-09-11 |
| two third-party ComfyUI packs | their READMEs only | 2026-08-16 |
| Comfy-Org/workflow_templates | `gh api`: commits since 2026-09-19 touching H3, and the diff of `fc427f00` | 2026-09-25 |
| xmarre's ComfyUI-Sol-H3 | its README and the body of its PR 9, via `gh` | 2026-09-11 |
| Comfy-Org/ComfyUI PR 16239 (closed unmerged 2026-09-16) | its diff via `gh`: `nodes_sparse_attention.py` and two helpers in its new module | 2026-09-11 |
| Comfy-Org/ComfyUI PRs touching H3 or core's sparse node, and kitchen's open PRs | `gh`: lists, bodies and threads; diffs for 16388, 16344, 16404, 16245, 16362, 16378 (2026-09-19); diffs for 16460, 16508, 16497, 16548, 16476, 16542, 16156, 16483, 16391 (2026-09-25) | 2026-09-19, 2026-09-25 |
| Comfy-Org/comfy-kitchen | fetch and `gh`; dated sections below | 2026-09-04, 2026-09-08, 2026-09-10, 2026-09-11, 2026-09-15, 2026-09-19, 2026-09-22, 2026-09-25 |

Every `coderef/Sana/...` pointer below resolves against a checkout at
`757d902`. The branch `release/sol-h3-spark` has the same tree as that tip.

---

## comfy-kitchen and core, 2026-09-25

Read with the kitchen clone's `upstream` and kijai's fork fetched, `gh` PR
lists, bodies and diffs, and the ComfyUI checkout at `88ab4a06` (pulled
2026-09-25; `git reflog` in it dates each pull). Core commits are read against
`3c80da7f`, the 2026-09-19 read. No ComfyUI server was running at the time of
the read, so the next start runs this checkout.

**Kitchen: still current, nothing to rebuild.** `vendor/rebuild_kernel.sh
--check` reports the source current against ComfyUI's pin and the venv
holding the `h3-build` tip. Upstream `main` is past `v0.2.35` and untagged,
so by policy it is not built. What it holds, for the next rebase:

- **`ef40891` (#192, kijai) will reach H3's video VAE encode on the next
  tag.** Its `zero_pad` and `out=` work is for the SeedVR2 VAE, which closes
  the 2026-09-19 note that the fork commit named no model. The same PR also
  raises the depth gate on `fp16_conv3d`'s fp16 accumulation. Core's H3 VAE
  calls that kernel when fp16 accumulation is on (`comfy/ldm/minimax/vae.py`,
  the `fp16_conv3d` helper, gated on `comfy/ops.py::_fp16_linear_wanted`).
  This install's launcher turns fp16 accumulation on (`start.sh`,
  `--fast fp16_accumulation`). The reader counted the encoder's 512-channel
  3x3x3 convolutions that fall between the old gate and the new one; the
  decoder has none. So after that rebase, reference and keyframe encodes on
  those stages move to fp16 accumulation. The PR body reports the fidelity
  cost for SeedVR2's decoder. Nothing here has measured it for H3. The
  rebuild record for that tag should say so.
- `f61028a` makes kitchen's `is_available` functions return False on a CPU
  device, a semantics change to carry through the next rebase. #186 adds head
  dim 256 to the Flash kernel, which is not H3's. #191 and #193 add W6A8 (no
  model in `workflows/h3_config.py::MODELS` uses it: every DiT and encoder
  header is `int8_tensorwise`). #194 is HIP INT8 attention; the rest is
  Ascend.
- **Open:** 187 and 189 unchanged; 195, 197, 199, 200, 201 and 202 are HIP,
  Ascend, or Triton INT8 dequantisation. **168 (ours)** is still open with no
  activity since 2026-09-11.

**kijai's fork.** `zero-pad-mode` moved to `624fdaf` (2026-09-25), which adds
an `fp32_accumulate` option to `fp16_conv3d`. A `w6a8` branch holds #191's
follow-ups.

**Core, merged since 2026-09-19 and in this checkout:**

- **`fc584aaa` (#16436, kijai) changes every H3 video decode's pixels at tile
  seams.** Tiles now blend against their neighbours as already blended.
  Tiling is on by default in core's H3 VAE (`MiniMaxH3VideoVAE.__init__`,
  `tiling=True`), so every decode at a trained canvas takes that path. It
  reached this checkout in the 2026-09-22 pull. **A clip rendered before that
  pull is not bit-comparable with one rendered after it**, which bears on any
  record that grades a rebuild by a bit-identical render against an older
  reference clip, such as `bench/results/2026-09-18_kitchen_0.2.35_rebuild.md`.
- `912fca4f` (#16485) moves the VAE's `qk_norm_scale` to the query's device
  before kitchen's fused `rms_rope`, for offloaded VAEs. It supersedes open
  16391.
- **`c194dd00` (#16419): a model file can name a block's attention.** A
  `<module>.comfy_attention.config` tensor holding JSON selects kitchen INT8
  attention for that block, and only that method is supported. An
  `optimized_attention_override` keeps priority, so Sol's override still
  wins. **No file in `workflows/h3_config.py::MODELS` carries the key**
  (every file scanned with `safe_open`, 2026-09-25), so no shipped graph
  moves. Two copies of core's forward in this pack would react differently
  to a file that did carry it. `exact_blocks.py::_exact_forward` strips the
  override and calls core's `Attention.forward`, so its "exact" blocks would
  run kitchen INT8. The sage node's forward
  (`attention.py::make_minimax_attn_forward`) does not pass
  `preferred_attention`. Both are inert today.
- **`b16023b0` (#16457) removes torchaudio from core's requirements.** Core's
  H3 reference-audio resample now calls `comfy/audio.py::resample`. This pack
  still imports torchaudio lazily at three resample sites
  (`audio_freeze.py`, `reference_conditioning.py`), and `pyproject.toml`
  declares no dependency. So a fresh install without torchaudio fails only
  when a clip's sample rate differs from the audio VAE's. This box still has
  torchaudio. `comfy.audio.resample` returned tensors `torch.equal` to
  `torchaudio.functional.resample` on CPU fp32 at four common source rates
  into the VAE's rate, checked in the 2026-09-25 session. That makes it a
  drop-in replacement; it is not yet made.
- `95539f56` (#16471) Fun-ControlNet Union 2.0 for H3; nothing here wires a
  ControlNet. ComfyUI `v0.37.0` was tagged 2026-09-20. The rest is other
  models, assets, partner nodes and NPU/ROCm.

**ComfyUI's workflow templates switched H3's video VAE to INT8.**
Comfy-Org/workflow_templates `fc427f00` (#1280, 2026-09-22) points every H3
template, the FastH3 ones included, at
`minimax_h3_video_vae_int8_convrot.safetensors`. That is the file the owner
removed from this repo on 2026-08-21 (the comment at
`workflows/h3_config.py::MODELS`, `video_vae`), and
`bench/check_model_files.py` goes red on any graph naming it. **It does not
trigger the adopt-upstream rule.** Templates are not one of the rule's three
upstreams. sglang runs the VAE more precisely, not less
([`wiki/references.md`](wiki/references.md), "What the 2026-08-28 pass
established"). The owner's decision stands unless the owner reopens it. The
core decoder has an INT8 attention branch for such a file
(`comfy/ldm/minimax/vae.py`, `Attention.forward`, the `QuantizedTensor`
test).

**Open core PRs, new or moved since 2026-09-19.** Found by keyword search
(`gh pr list --search`) on H3, MiniMax, sparse attention, Qwen3, kitchen and
the new attention key, with no file filter, so a PR that touches an H3 path
under an unrelated title can be missed.

- **16508 (fp16 inference for H3) would move this install's DiT to fp16 if
  it merged.** It adds `torch.float16` to H3's `supported_inference_dtypes`.
  `start.sh`'s `--fast fp16_accumulation` sets `PRIORITIZE_FP16`
  (`comfy/model_management.py`), which `unet_dtype()` consults before its
  dtype loop. The quantized-checkpoint exemption in the PR body covers
  fp16-stored weights, not that branch, and every shipped graph loads the DiT
  with `weight_dtype` at default. With fp16 activations, Sol's eligibility
  gate refuses the call (`sol_attn_h3.py::_ineligible`, "kernel is
  bf16-only"), so every Sol graph would run dense, and the reason would show
  only in the route record. This pack's own attention forwards also lack the
  PR's fp16 `out_proj` rescale. All of this is reasoned from the code, not
  run. It is the one open PR here worth watching.
- **16460 (packed-row memory estimate) and 16542 (dynamic-VRAM headroom)**
  would together change what gets paged out when sampling is admitted: the
  first changes the estimate, and the second makes `free_memory` partially
  unload dynamic models to meet it. On long runs that means the text encoder
  and the DiT's own resident pages. Neither is measured here.
- **16156 (`percent_to_sigma` rounding and half-open windows)** reaches
  `sol_attn_h3.py`, `pdd_lora.py` and `audio_freeze.py`, which all call
  `percent_to_sigma`. The reader simulated the schedulers, step counts and
  windows our graphs use, and none puts a step exactly on a boundary, so no
  shipped graph's step set changes. That was simulated, not rendered. A
  window whose percent times the step count is a whole number would move by
  one step.
- 16497 (save and load nested conditioning), 16548 (RGBA into Qwen VL
  preprocessing), 16476 (a stale KV cache in `generate()`) and 16483 (W6A8):
  no reach. We save no conditioning, every image path here is already RGB,
  H3 conditioning never calls `generate()`, and no model here is W6A8.
- Unchanged and still open from the 2026-09-19 list: 16388, 16344, 16404,
  16245, 16116, 15735, 16283, 16221, 16228, 16301, 16076, 15135, 16401,
  16402, 16362. 16391 is superseded by 16485 above.

## comfy-kitchen, 2026-09-22

Read with `upstream` and `origin` fetched in the clone, `gh` PR lists and
bodies, and kijai's fork through the GitHub API (the clone no longer has a
`kijai` remote). Core was not re-read.

**Still current, nothing to carry.** `vendor/rebuild_kernel.sh --check` is the
observable: ComfyUI's pin has not moved from the tag the 2026-09-19 read
found, upstream `main` has nothing past it, and the venv holds the `h3-build`
tip. The clone's local `main` and the fork's `origin/main` mirror lag
upstream; neither is built.

**Kitchen PRs new or changed since 2026-09-19:**

- **187, Triton `rms_rope` grid overflow at 65,536 or more rows** (issue
  169). H3's packed sequence is past that, but the kernel is kitchen's Triton
  backend, which core disables unless `--enable-triton-backend` is passed
  (`comfy/quant_ops.py`), and this install's launcher does not pass it. The
  failure is also loud (a CUDA invalid-argument error), not silent. Not this
  card, the same reasoning as 172.
- **189, CUDA `dequantize_mxfp8`** (issue 190), replacing an eager fallback.
  No model in `workflows/h3_config.py::MODELS` is MXFP8. Not this card.
- 185 (HIP arch support), and 184 updated (HIP Sol exact): not this card.
- **168 (ours)**: open, still no activity since 2026-09-11. Its last comment
  names PR 171's `*, key_bias` as the thing it folds with; 171 closed
  unmerged on 2026-09-16, so that note no longer binds anything.

**kijai's fork:** no branch has moved since the last read; the newest head
is still `zero-pad-mode` at `16651db`.

## comfy-kitchen and core, 2026-09-19

Read with the fork's remotes fetched and `gh` (PR lists, bodies, threads, and
the diffs named below). ComfyUI's pin is `comfy-kitchen==0.2.35`
(`requirements.txt` in the ComfyUI checkout); upstream `main` has nothing past
the `v0.2.35` tag, and `h3-build` is rebased onto it (CHANGELOG 0.123.1).

**Closed since the last read, and what that retires here:**

- **Kitchen PR 171 (chunked `key_bias`) and core PR 16239 (attention key
  measure) were both closed unmerged by their author on 2026-09-16.** The
  author's closing comments say the mixed-resolution path they served was
  retired from the author's own pipeline, and that the branches are kept on
  the author's fork. The `*, key_bias` rebase hazard in the 2026-09-11 section
  and the "would rewrite this node" note under core's Sol node no longer
  apply. Neither came back under a new number: the same author's 2026-09-18
  core PRs 16401 and 16402 are for "MiniMax H3 Keyless", a checkpoint variant
  whose main blocks carry a packed `qv_proj` and no K projection (core's
  sparse node would give it only the generic override and refuse VSA; generic
  LoRA loading fails closed). This repo loads no such checkpoint.
- **Kitchen PR 176 (W4A8 decode GEMV) merged into `v0.2.35`**, with its HIP
  port 182, so the 2026-09-15 table's "not carried" row is now carried code.
  What the tag brings is in CHANGELOG 0.123.1: nothing on the H3 attention
  path. Its core half, PR 15623 (Qwen3 cudagraphs and W4A8 GEMV), merged
  2026-09-18.

**Open kitchen PRs new since 2026-09-15:**

- **179 (another developer), `draft_attention`, with its core half 16362.**
  A DraftMap block-sparse self-attention for packed video, adapted from the
  anemoi project, with a native Ada executor: its body lists SM89 among the
  served capabilities, so **it is the one open kernel that would run on this
  card.** The core half adds it as a `draft` method in core's Sparse
  Attention node, H3 only, prefix exact, sparse routing on the video rows.
  The PR body's end-to-end table includes an RTX 4090. kijai's reply on
  2026-09-16 graded it against kitchen's `sol_attn` on real H3 captures (two
  blocks, relative L2 against dense, on a 5090) and found it no better than
  Sol at matched speed, with a much larger workspace; the numbers are in that
  thread and not copied here. If it merges and is ever worth a look, the
  instrument is a capture grader (`bench/grade_sol_route_on_capture.py` and
  its siblings), not a render.
- **186 (comfyanonymous): head dim 256 in the Flash kernel.** H3's DiT head
  dim is `attention_head_dim` in `vendor_config/fl2va_transformer_config.json`,
  which is not 256. Not H3.
- 183, 184 (HIP Sol exact attention), 174, 180, 185 (HIP), 151 (Ascend), 181
  (agent docs): not this card.
- **168 (ours, `blk_cnt`)** is open, no activity since 2026-09-11. The
  `qk_balance` branch (`sol-qk-balance-pr`, section "2026-09-15, evening"
  below) has no PR.

**kijai's fork.** `zero-pad-mode` (`16651db`, 2026-09-19, not a PR yet) adds a
zero-padding mode to the fused VAE group-norm, SiLU and pad3d kernel from PR
167, and touches the fp16 conv3d path. The commit does not name the model it
is for.

**Core, merged since 2026-09-11 and running in this install** (the ComfyUI
checkout is at `3c80da7f`; `git reflog` in it dates each pull):

- **16187 and 16332: the H3 VAE kernels are live.** With kitchen PR 167 in
  `v0.2.34`, both halves of [`open_experiments.md`](open_experiments.md) #28
  have merged; its status line carries what that means.
- **16326, 16351, 16389 and 15623 reach H3's text encoder, not the DiT.**
  Core's Qwen3-VL encoder is `Llama2_` from `comfy/text_encoders/llama.py`
  (`comfy/text_encoders/qwen3vl.py` imports it), and 16326 (`6cff1e97`) makes
  that file's `apply_rope` call kitchen's `apply_rope_split_half` in place of
  core's own torch code. The 2026-09-18 rebuild record's bit-identical render
  (`bench/results/2026-09-18_kitchen_0.2.35_rebuild.md`) spans core `36da3ff7`
  to `a8686f2b`, which covers 16351, 16389 and 15623 on that graph and scene:
  none of the three reached this checkout before the 2026-09-16 pull, and the
  reference clip was rendered on 2026-09-15. It does not establish anything
  about 16326: the checkout had `6cff1e97` from the 2026-09-15 morning pull,
  and no record says whether the server that rendered the reference clip was
  started before or after it.

  **What of 15623 reaches H3's encode path, read 2026-09-19 at `3c80da7f`:
  one change, and it is a load-order change.** Most of the PR is decode-time
  work (fixed KV caches, CUDA-graph decode, speculative verify, layer
  prefetch), and H3 conditions with one prefill and a layer tap, never a
  decode loop: in `comfy/text_encoders/llama.py::Llama2_.forward`, graph
  capture needs a decode step, and layer prefetch is gated on
  `past_key_values is not None`. `Qwen3VL_32BConfig` inherits the new
  `fixed_kv`, `graph_dynamic_vbar_blocks` and `prefetch_dynamic_vbars`
  flags from `Qwen3VL_8BConfig`, but only the second reaches an encode: the
  PR moved `get_dynamic_vram__units` onto `comfy/sd1_clip.py::SD1ClipModel`,
  which `MiniMaxH3TEModel` extends, so `comfy/model_patcher.py`'s dynamic-VRAM
  load now groups the encoder's weights one unit per decoder layer, where
  before the H3 TE model offered no units. Checked by building
  `comfy/text_encoders/minimax.py::MiniMaxH3TEModel` on the meta device and
  calling it.

  **Its multi-token prediction (MTP) half cannot apply to H3, three ways.**
  Wiring: `comfy/sd.py` passes `mtp=` only in the `QWEN35_*` branch, and
  `MTPHead` lives in `comfy/text_encoders/qwen35.py`; H3 loads through the
  `CLIPType.MINIMAX` branch, and
  `comfy/text_encoders/qwen3vl.py::Qwen3VLClipModel.generate`
  accepts `mtp` and ignores it. Weights: core arms MTP on an `mtp.fc.weight`
  tensor, and the shipped encoder's header (`h3_config.ENCODER_INT8`) has no
  `mtp`, `lm_head` or final-norm tensors, matching the config's
  `lm_head=False, final_norm=False`, so it cannot produce logits at all. Call
  path: MTP is speculative decoding inside `generate()`, and nothing in this
  pack calls `generate()`; conditioning goes through
  `encode_from_tokens_scheduled` (`conditioning.py`).

  **`MiniMaxH3EncoderLoader` gets all of it, as `CLIPLoader` does.**
  `h3_encoder_loader.py::load_guarded_clip` is core's `comfy.sd.load_clip`
  with `CLIPType.MINIMAX`, guards run after construction, and the
  `cached_patcher_init` it registers rebuilds through the same call. The
  flags live on the config class and the units method on the base class, so
  nothing here has to opt in. What the unit grouping does to encode wall time
  or peak VRAM is not measured, and a figure for it would be a statement
  about cache state.
- 16285 (`linear_input_act` respects `_full_precision_mm`): acts only where a
  format is disabled on the device; INT8 is supported on this card
  (reasoned). 16240: the Fun ControlNet under the memory compiler.

**Open core PRs that touch H3, read against their diffs:**

- **16388** makes core's `parse_block_list` (the `dense_blocks` of core's
  sparse node) refuse a negative entry that it used to read as positive. Ours
  is not affected: `block_spec.py::parse_blocks` resolves negatives from the
  end and refuses anything else, and no graph uses core's node (a walk of
  `h3_config.graph_paths(..., include_bench=True)` finds only
  `MiniMaxH3SolAttn`).
- **16344**: core's sparse node replaced any earlier `double_block` patch (a
  Fun ControlNet applied before it silently stopped acting), and the ControlNet
  tower inherited the sparse override. Our VSA node already refuses an earlier
  block patch (`vsa_attention.py`), and our Sol node composes with a previous
  override. The second half applies to our Sol override too: a ControlNet
  wired under it would run its tower through Sol. Nothing here wires one.
- **16404** (supersedes the closed 16378): keeps one INT8 QKV cast alive
  across core's chunked producer when VBAR falls back to temporary casts;
  measured on AMD. Our `MiniMaxH3SolChunked` projects per chunk the same way
  (`sol_chunked_h3.py::make_chunked_forward`), and no shipped graph uses it.
  kijai's reply on 16378 disputes its chunk-size claim on CUDA.
- **16245**: a per-model `disable_comfy_compiler` key in `transformer_options`
  for H3. If it merges, it is the per-graph alternative to a node refusing
  the memory compiler (the reorder's history: CHANGELOG 0.123.0).
- 16116 (attention memory estimate; its body says NVIDIA with PyTorch
  attention on is unaffected), 15735 (an H3 AV latent builder for two-pass
  upscales), 16283 and 16221 (Fun ControlNet), 16391 (XPU VAE), 16228 (AMD
  encoder fallback), 16301 (startup guard): title and body only, none reaches
  a shipped graph on this card.

**Open core PRs on H3's text encoder, 2026-09-19, title and body only via
`gh` (no diffs read), against core at `3c80da7f`:**

- **16076, per-image text-encoder-only references.** A marker node that
  sends one still to Qwen3-VL alone while its neighbours keep their DiT
  rows. Ours is all or nothing per node: leave the VAE unwired and every
  reference goes encoder-only (`docs/h3_references.md`, "Encoder-only
  references"). Per-image mixing is the capability open experiment 26 would
  want if it closes toward the cheaper arm. If it merges, it is a port
  candidate for `MiniMaxH3ReferenceConditioning`; nothing to do before.
- **15135, masked grouped-query attention falls back to SDPA's math
  backend.** Probably covered in core already, by `a1c42199` (PR 15190,
  2026-07-31), and not yet observed. H3's encoder is that case: fp32
  activations, a causal float mask, fewer KV heads than query heads. On
  NVIDIA, `comfy/ops.py::scaled_dot_product_attention` expands K and V only
  when `SDPAParams` says no fused backend takes native grouped-query
  attention under the mask; elsewhere it expands unconditionally. The
  2026-08-25 record found flash, cuDNN and efficient all unavailable for
  that call at fp32 (torch 2.13, the closed calibration lane's transformers
  harness, not core's path:
  `docs/research/qwen3-vl-special-tokens-post-training/brainstorming/claude-encoder/2026-08-25-gate2a-corrected-floor.md`,
  "Backend selection, measured"). If torch 2.14 answers the same, core
  expands and the efficient kernel runs. What closes it is the dispatched
  `aten::_scaled_dot_product_*` op under the profiler, on core's call with
  the encoder's geometry, on a free card; `bench/probe_sdpa_backend_selection.py`
  is the pattern, though it calls torch directly rather than `comfy.ops`.
- **15316**, reserve the encoder's memory for image encodes: its body
  describes the stall with `--disable-dynamic-vram`, which this server does
  not pass. **15552**, an INT8 embedding crash under dynamic VRAM: the
  encoder this install loads keeps `model.embed_tokens.weight` in BF16 (the
  header of `h3_config.ENCODER_INT8`). **16277, 15638, 16262**: `generate()`
  only; H3's encoder has no `lm_head` and is never asked to generate.
  **15983** is the DiT's memory estimate, not the encoder's.

**Merged, on the encoder's placement:** 16374 (`d39cdfdb`) makes
`text_encoder_device()` the GPU whenever dynamic VRAM is on. Before, that
branch still asked `should_use_fp16`, which this card answers yes, so no
change is expected here (reasoned, not checked). A `device` of cpu on
`MiniMaxH3EncoderLoader` overrides it either way (CHANGELOG 0.131.0).

---

## comfy-kitchen, 2026-09-15: the pin moved to v0.2.34, and what was and was not carried

Read with the fork's remotes fetched (Comfy-Org as `upstream`, kijai as
`kijai`) and the GitHub PR lists of both, the day ComfyUI's
`requirements.txt` moved to `comfy-kitchen==0.2.34`.

**Carried, by rebasing `h3-build` onto `v0.2.34`** (everything the tag holds
past `v0.2.33`): PR 167 and its HIP port 175, the MiniMax H3 VAE kernels
(fused encoder pad/norm, fp16-accumulate conv3d and GEMM, int8 residual
epilogue); PR 162, persistent RoPE allocations behind
`set_allocation_context`, which core now calls; PR 165, `compress-mode=size`
for the CUDA build; an eager `apply_rope_split_half1` optimization. Our six
`blk_cnt` commits (PR 168, still open) reapplied without conflict.

**Kijai's branches: nothing left to pull.** Comfy-Org squash-merges, so
commit counts against upstream overstate what is unmerged; by content, on
`comfy_kitchen/backends/cuda/sage_attention` and the eager Sol reference,
`kijai/sol_exact_pquant` (#150), `kijai/sol_token_aug_main` (#156) and
`kijai/minimax_vae` (#167) are identical to `v0.2.34`. Their remaining
differences are an older base (`v0.2.32`): missing HIP ports and tests, and
older Triton rope/quantization helpers. `kijai/w4a8_gemv` is PR 176 below.
A local branch named `sol_fp16_pv` in the kijai mirror is kijai's
2026-08-14 `sol_attn` head, not a 16-bit PV variant; the name misleads.

**Open PRs assessed and not carried:**

| PR | what | why not |
|---|---|---|
| 176, kijai, `w4a8_gemv` (*merged into `v0.2.35` on 2026-09-16 and carried since the 0.123.1 rebuild; 2026-09-19*) | W4A8 codebook GEMV for M <= 8, int8 decode GEMV, fused GatedDeltaNet decode; for an LLM text-encoder *decode* loop, paired with a core PR | H3's encoder is prefill only here; no decode path runs. Revisit if ComfyUI's pin moves to a tag that needs it, which the rebuild gate will say |
| 171, xmarre, chunked `key_bias` (*closed unmerged 2026-09-16; 2026-09-19*) | per-key logit bias on the fused-QKV `sol_attn_chunked` producers, CUDA and HIP | `sol_attn_chunked` is registered and unwired here, and `key_bias` is left at its default on the direct path too |
| 172, neuregex, Triton INT8 GEMM int64 offsets | the same overflow class as the sage fork's v0.7.0 and v0.7.17 fixes, in kitchen's Triton INT8 GEMM output offset | this card runs the CUDA INT8 linear, not the Triton one; a correct fix for a path not on ours. Worth taking if that changes |
| 174 (HIP WMMA), 151/153/154/155/161 (Ascend), 157 (stochastic fp8, Triton), 160 (dead import) | other backends, or cosmetic | not this card |

**What would make this stale:** ComfyUI moving its pin again (the gate in
`vendor/rebuild_kernel.sh` reports it), PR 168 merging (drop the carried
commits on the next rebase), or a kitchen change to `quant_k_rows` /
`quant_q_rows`, which is where the sage fork's `qk_balance` factor would go
on the Sol side (`docs/h3_block49_quant_error.md`, section 7).

## The paper

**"Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention
Sparsification."** Read 2026-08-16, after this repo spent two weeks treating it
as unread. Scope qualifier that matters: the abstract, the ablation summary and
the HTML were read, not the whole PDF. Anything below is at that depth.

**The contribution is the correction, not the sparsity.** The framing is that
existing block-sparse methods fail two ways: routing is rigid and costly to
materialize, and unselected blocks are dropped outright, which degrades
accuracy under aggressive sparsity. Sol-Attn thresholds during online softmax
and **reuses the proxy scores of unselected blocks to approximate their
contribution** instead of discarding them. The ablation that matters compares
exact-only against exact-or-approx, and the correction's advantage **widens as
sparsity rises**.

What follows from it:

- **The pooled tail is the method, not a side knob.** Sol-Engine's public API
  has no toggle for it at all, and neither does the copy Sol-H3 vendors (both
  signatures are cited below). Our side of it is `pooled_tail` in
  [`docs/SOLATTN.md`](SOLATTN.md).
- **`tau` is the paper's beta, and it is never swept.** The threshold is
  `t_i = mu_i + beta * sigma_i` over the row-wise mean and standard deviation,
  with beta described as a shared standardized cutoff. Sol-Engine defaults 1.0.
  *Corrected 2026-09-10:* this bullet said "the paper does not settle 1.0
  against our 1.3"; ours has been 1.0 since 2026-08-20
  (`workflows/h3_config.py::SOL_RECOMMENDED_CUDA`), and this page does not
  carry our values.
- **H3 is not evaluated anywhere in it.** Tested models are Wan 2.1-14B,
  HunyuanVideo-13B, LTX 2.3-22B, Bernini-14B for editing, a SANA-WM refiner,
  Ideogram 4 for 2K text-to-image, and Wan 2.1-1.3B on a 5090. The headline
  speedups are in the abstract, against dense.
- **No token reordering appears in it.** No Morton, no Z-order, no permutation,
  no spatial block layout. See [`docs/morton.md`](morton.md), which owns what
  that does and does not license.
- **sm89 was not an official target** when the paper and its docs were written.
  That changed on 2026-08-15; see the kernel section below.

---

## Sol-Engine's per-hardware H3 cells

Sol-Engine **supports MiniMax-H3 as a first-class model** with validated,
pinned configurations. The paper does not mention H3. The framework does.

The per-hardware cells run at **1344x768, 124 frames**, 50 steps, on the
released BF16 weights. *Corrected 2026-09-10:* this said "every H3 config runs
at 1344x768, 124 frames"; the packages added since run other geometries
(Sol-H3 at 124, 243 or 362 frames, Spark's draft at 672x384, the
`super_acceleration` first stage at 896x512), each cited in its own section.

**The recipe is per-hardware-profile, not one policy.**

| cell | attention | thresholding | cache |
|---|---|---|---|
| 4xA100 (sm80) | **Triton reference**, tau 1.0 | `exact` | FirstBlockCache |
| 4xH100 (sm90) | CuTe, tau 1.0 | `exact` | FirstBlockCache |
| 8xGB200 (sm100) | CuTe, tau 1.0 | `diag` | FirstBlockCache |
| 1x RTX 5090 (sm120) | CuTe, tau 1.0 | `diag` | TeaCache |
| **1x RTX 4090 (sm89)**, added 2026-08-17 | SM89 Sol-Attn, tau 1.0 | `diag` | TeaCache |

Sources: `coderef/Sana/models/minimax_h3/A100/adapter.py:550` and the H100
cell's same line, `coderef/Sana/models/minimax_h3/GB200/gpu_infer.py:144`,
`coderef/Sana/models/minimax_h3/RTX5090/adapter.py:477`,
`coderef/Sana/models/minimax_h3/RTX4090/adapter.py:476-477`, and each cell's
README.

**A100 is the Triton reference path**, because sm80 has no CuTe kernel.

**Every per-hardware cell runs the first two blocks and the first 10 of 50
steps dense.** `SOL_ATTN_FIRST_DENSE_STEPS` and `SOL_ATTN_FIRST_DENSE_LAYERS`
(`coderef/Sana/models/minimax_h3/RTX5090/adapter.py:457-460`, the same
function in the RTX4090 cell at `:457-461`), `H3_SOL_DENSE_LAYERS`
(`coderef/Sana/models/minimax_h3/GB200/gpu_infer.py:146`), `dense_blocks: int = 2`
(`coderef/Sana/models/minimax_h3/GB10/fusion_install.py:84`). *Corrected
2026-09-10:* this said "every profile", and read as NVLabs' H3 policy. The
packages added since do not follow it: Sol-H3 runs Ref2VA with no dense steps
or layers, and Spark's Ref2VA draft keeps only step 0 and layer 0 dense, both
cited below.

**None of the per-hardware cells reorders tokens.** Stated explicitly in the
A100 README: "pinned SGLang BF16 execution without `torch.compile` or token
reordering" (`coderef/Sana/models/minimax_h3/A100/README.md:24`).

### The single-card cells: RTX 5090 and RTX 4090

`models/minimax_h3/RTX5090/` runs H3 on **one consumer GPU** with SGLang
layerwise component offload, and it ships an isolated Sol-only arm,
`config/minimax_h3/rtx5090_sol.toml` against `config/minimax_h3/rtx5090_dense.toml`,
with tau 1.0, `diag`, and the dense steps and blocks above. **No number is
published for that arm**; the RTX 5090 README publishes only the full-opt
figure (`coderef/Sana/models/minimax_h3/RTX5090/README.md:13`), and full-opt
includes TeaCache and `torch.compile`.

`models/minimax_h3/RTX4090/` is the same stack on **this repo's card class**:
the released BF16 FL2VA checkpoint on one 24 GiB RTX 4090, the 33B DiT, the
Qwen3-VL conditioner and the VAEs under SGLang layerwise component offload
(`coderef/Sana/models/minimax_h3/RTX4090/README.md:5-7`). Its full-opt recipe
(`:33-40`): SM89 Sol-Attn at tau 1.0 with `diag`, an exact prefix KV sink and
dense prefix queries, the first 10 steps and first two blocks dense, TeaCache
(`coderef/Sana/models/minimax_h3/RTX4090/teacache.py`), and regional
`torch.compile`. It imports the `sol_attn` package
(`coderef/Sana/models/minimax_h3/RTX4090/adapter.py:337`, `:474`), whose
dispatch table in Sana maps compute capability 8.9 to `cute_sm89`
(`coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:11`).

**It publishes a Sol-increment arm, and that arm sits on top of TeaCache, not
against dense.** The controlled attribution table
(`coderef/Sana/models/minimax_h3/RTX4090/README.md:20-31`) goes lossless opt,
then plus TeaCache, then plus TeaCache and Sol-Attn, from one warm timing
sample per arm with no perceptual or embedding similarity metric, which the
README says itself. Its correctness evidence is a real-QKV gate against SDPA
(`:83-85`, the gate at `coderef/Sana/models/minimax_h3/RTX4090/adapter.py:347-362`,
`tau=-1000` so every block is routed). The README records the evidence as
collected at Sana `6fb7eb1` before the cell was split into its own directory
(`:81`, `:88-91`).

---

## Sol-H3: the multi-GPU production package

Added by `2936c47` (2026-09-08). A production inference package for text-to-video,
first-frame image-to-video and Ref2VA at 1344x768 and 24 FPS, for 1, 2, 4 or 8
GPUs, **validated on 8x NVIDIA B300** (`coderef/Sana/models/minimax_h3/Sol-H3/README.md:5-12`).
Kernel validation is limited to SM103; the SM90 and SM120 paths are declared
unvalidated (`:32-34`). One GPU runs dense by contract: the engine refuses SOL
attention below two processes
(`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/engine.py:147-148`).

It runs four DiT forwards (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/engine.py:22`: five scheduler points), at shift
12 for video and 3 for audio (`:214-215`), with the FastH3 4-step preview
adapter for T2V and I2V and the LightX2V ref2v Turbo 4-step LoRA for Ref2VA
(`coderef/Sana/models/minimax_h3/Sol-H3/README.md:115`, `:140`). Attention
backends are `dense`, `sol` and `sol_bsa` (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/engine.py:24`); Ref2VA accepts
only `dense` or `sol_bsa` (`:149-150`), and `sol_bsa` is the default
(`coderef/Sana/models/minimax_h3/Sol-H3/README.md:148`).

### Policy per task

One call sets it (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/engine.py:265-272`):

| task | tau | dense steps | dense layers | sink mode |
|---|---|---|---|---|
| T2V, I2V | 1.0 | 1 (the first of four forwards) | 2 | `prefix` |
| Ref2VA | 1.0 | 0 | 0 | `text_audio` |

- **No dense tail.** The only step gate is `step < dense_steps`
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py:345`);
  the layer gate is beside it (`:349`).
- **The diagonal band is forced exact**, one block either side
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/bsa_metadata.py:84`).
  comfy-kitchen forces the same band (the `<= 1` line in
  `coderef/comfy-kitchen/comfy_kitchen/backends/eager/sol_attn.py`).
- **No top-k list, no per-head or per-block tau.** tau is one value per call.

### How the sink is built

- **`prefix`** marks everything before the target-video tail as an exact KV
  range (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py:296`) and then recomputes every prefix query row
  with dense attention (`:418-420`). The module docstring gives the reason: the
  audio rows are generated, not only conditioned on, and one prompt's dialogue
  fell apart under the text-only policy (`:26-38`).
- **`text_audio`** leaves the Sol-kernel sink empty (`:287-288`) and instead
  builds a per-block mask from the text and audio indices, **excluding Ref2VA
  visual references** (`:298-318`); the BSA route then forces those key blocks
  and those query blocks (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/bsa_metadata.py:85-88`). So text and audio are exact
  both ways while reference rows stay sparse. It does this through cuDNN
  block-sparse attention's per-block selection, not through the Sol kernel's
  contiguous sink.

### The kernel it carries

- **`sol_bsa`** computes the routed blocks with cuDNN block-sparse attention
  and merges the pooled term for the omitted blocks through a log-sum-exp in a
  Triton kernel (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sol_residual.py:146-160`).
- **`relayout.py`** is the Ulysses all-to-all layout pair, with an optional
  int8 wire path after QK-norm and RoPE; it moves heads between GPUs and does
  not reorder tokens (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/relayout.py:1-27`).
- **The vendored `sol_attn`** is pinned to Sana `cee25847a` as "released Sol-Attn
  integration contract" (`coderef/Sana/models/minimax_h3/Sol-H3/SOURCE_SNAPSHOT.json:14-16`).
  Same algorithm as comfy-kitchen's: 64-token blocks, threshold `mean + tau*std`
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn/preprocess.py:232-240`),
  and the pooled tail unconditionally on in the Triton reference
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn/triton_ref/fwd.py:108-131`). Differences: it is
  BF16 throughout (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn/sm120/mainloop.py:727-729`), and its
  signature has **no top-k, no `sink_q`, no tail toggle**
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn/interface.py:421-433`).
  **Its architecture table omits sm89**: it lists (9,0), (10,0), (10,3) and
  (12,0) (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn/interface.py:10-19`), and anything else selects the Triton
  reference (`:95`).
- **Its arithmetic gate** runs block-sparse attention with every block selected
  against dense SDPA on the real Q/K/V, once per shape
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py:470-500`).
- The module docstring repeats the "H3 needs no reordering" statement word for
  word from the GB200 cell (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py:8-12`); see the Morton section.

### MXFP8 and the transport quantization

- **MXFP8 is linears only, and compute capability 10 only.** It replaces the
  fused QKV, the output projection and the FFN up and down projections in
  blocks 2 to 46 (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/compute_quant.py:11-12`,
  `:96-130`), with E4M3 values and one E8M0 scale per 32 along K
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/mxfp8.py:3-4`), and
  refuses any device whose major capability is not 10 (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/compute_quant.py:27-35`).
  Attention's QK and PV products are not touched.
- **The multi-GPU exchange is quantized separately**: group-scaled int8 for the
  Q/K/V packet and FP8 for the returned output
  (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/comm_quant.py:1`, `:14-24`,
  `:130-139`). This exists only when there is an exchange.

### What it validates

The README's latency tables and its own caveat that the Sol-H3 rows are a
four-forward distilled adapter plus approximate attention, so "the difference
from either 49-forward baseline is not a runtime-only speedup"
(`coderef/Sana/models/minimax_h3/Sol-H3/README.md:36-65`). Its one quality
comparison is MXFP8 against BF16 at one prompt and seed, as decoded-RGB PSNR
and SSIM (`:67-83`).

---

## Sol-H3-Spark: one DGX Spark, two stages

Added by `757d902`. A two-stage video-and-audio pipeline for **one GB10
(SM121)**; every stage refuses any other capability
(`coderef/Sana/models/minimax_h3/Sol-H3-Spark/README.md:1-5`, `:71`;
`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage2.py:181-182`;
`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage1_ops/sol.py:101-102`).
The recipe is a frozen record whose hash is checked at load
(`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/config.py:22-27`), so the
config file is the authority (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/configs/default.json:4-42`):

1. **Stage 1, an H3 draft** at 672x384x124: FastH3 VSA DataFree LoRA, W8A8 FP8
   DiT, four updates, VSA at 90% video sparsity on 64-token tiles through cuDNN
   BSA, no video decode (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/configs/default.json:4-14`). The VSA profile keeps the prefix
   exempt and has no dense layers or steps
   (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage1_ops/vsa.py:19-40`).
2. **Latent transfer**: a learned x2 H3 latent upscaler, then the H3-to-LTX
   latent adapter, which **replaces an H3 VAE decode followed by an LTX VAE
   encode** (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/README.md:30`, `:47-51`; `coderef/Sana/models/minimax_h3/Sol-H3-Spark/configs/default.json:15-22`). The adapter is a
   small Conv3D residual network
   (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage2_ops/h3_ltx_adapter/model.py:27-42`).
   Its [model card](https://huggingface.co/Efficient-Large-Model/H3-to-LTX-Latent-Adapter)
   gives the boundary as a normalized H3 latent `[B, 24, T, H/16, W/16]` in and
   a normalized LTX-2.5 latent `[B, 128, T, H/32, W/32]` out, and says the
   checkpoint holds neither model's weights and generates nothing alone.
3. **Stage 2, an LTX-2.5 refiner**: BF16 LTX-2.5 dev with a distilled LoRA,
   three joint audio-video updates, Triton Sol with
   `"taus": [1.0, 1.25, 1.5]`, `diag`, layer 0 dense (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/configs/default.json:23-37`,
   `coderef/Sana/models/minimax_h3/Sol-H3-Spark/README.md:33`). **This tau ramp runs on the LTX refiner, not on H3.**
4. **Output**: LTX's own Conv VideoVAE at 1344x768, 121 frames, with H3's
   generated audio muxed in (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/configs/default.json:38-42`).

### Ref2VA's opt-in Sol draft: per-step tau and a permuted sink

Ref2VA swaps in its own partition and the LightX2V Ref2VA 4-step LoRA, and
its draft attention is **dense FA4 by default, Sol only when selected**
(`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/config.py:12`, `:36-51`).
When Sol is selected the policy is (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/config.py:52-59`):

- dense for step 0 and for body layer 0 of the later steps;
- `tau_by_step` `[None, 1.0, 1.25, 1.5]`, i.e. **sparser as denoising
  proceeds** (higher tau keeps fewer blocks); the same tuple is `TAUS` in
  `coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage1_ops/sol.py:12`,
  routed by `route()` (`:15-18`);
- text and audio tokens are the sink and their queries run dense; reference
  image and Qwen-visual tokens are **not** in the sink.

This is the only place in the tree where **H3** runs a per-step tau.

**How the Sol kernel is made to express "text and audio exact, references
sparse".** The kernel's sink is one contiguous range, and H3 packs reference
rows between text and audio. `sink_plan` builds a permutation that puts every
visual row first and every text or audio row last, each part in its native
order, and records `"sparse_block_grouping_changed": True` in its receipt
(`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage1_ops/sol.py:21-47`). Q, K and V are permuted after RoPE, the kernel runs with the
sink over the trailing text-audio range, the sink query rows are recomputed
dense, and the output is permuted back (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage1_ops/sol.py:134-148`). The route counts per
request are asserted, not assumed (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/stage1_ops/sol.py:83-92`). Its own validation notes
describe the same policy (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/docs/validation.md:92-98`).

### Reference matching

"Reference matching" is a **per-image pixel budget**, taken from the draft
canvas or the final canvas (`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/config.py:43-50`), applied with each image's own
aspect ratio and nearest-32 alignment, Lanczos resampling
(`coderef/Sana/models/minimax_h3/Sol-H3-Spark/runtime/qwen_ops/media.py:46-73`;
`coderef/Sana/models/minimax_h3/Sol-H3-Spark/docs/validation.md:87-90`). It is not an identity matcher.

### What it validates

Functional coverage, stated as such: bitwise context and latent checks, call
counts, and final media produced, for T2VA, three FL2VA endpoint cases, and a
small set of Ref2VA image and image-plus-audio cases
(`coderef/Sana/models/minimax_h3/Sol-H3-Spark/docs/validation.md:25-40`, `:49-106`). The same file says it gives no
identity, accessory or voice fidelity guarantee and does not establish that
reducing reference detail preserves quality (`:104-106`).

---

## `super_acceleration`: the same split across two GB200s

H3 FL2VA as stage 1 on one GB200 at 896x512 with the LightX2V four-step LoRA,
then a tensor handoff to an LTX-2.5 stage 2 on a second GB200
(`coderef/Sana/models/minimax_h3/super_acceleration/README.md:14-31`). Stage 2
keeps LTX layer 0 dense and runs layers 1-47 on Sol with taus 1.0, 1.25, 1.5
(`coderef/Sana/models/minimax_h3/super_acceleration/stage2/sol_attention.py:18`,
`STAGE2_TAUS`). **That ramp is on the LTX transformer, not H3.** The README
makes no speedup claim and says its validation "must not be described as a
quality pass" (`coderef/Sana/models/minimax_h3/super_acceleration/README.md:52-56`). Spark runs the same H3-draft-then-LTX-refiner
shape on one box.

---

## ComfyUI core's own Sol node, `BlockSparseAttention`

Merged 2026-09-06 (Comfy-Org/ComfyUI #16072, kijai; core commit `e308cc73`),
display name "Model Sparse Attention", experimental
(`comfy_extras/nodes_sparse_attention.py:353-365`). Its defaults are read from
`BlockSparseAttention.define_schema` (`:355-411`), not restated here.

- **Three methods**: `sol-attn` (adaptive tau), `sla` (a fixed keep percent,
  for LoRAs distilled against that pattern) and `vsa` (FastVideo's cube tiling
  with the learned coarse branch, for FastH3 weights) (`:368-389`).
- **Token routing is on for every block by default.** `extra_tokens`
  (`:398-401`) is passed as kitchen's `token_aug` on both paths (`:208`,
  `:296`), and is ignored under VSA (`:326-329`). The Python `execute`
  signature defaults it to 0 (`:414-415`); the schema default is what a graph
  built in the UI carries.
- **The sigma window** is `start_percent` / `end_percent` (`:390-393`), turned
  into sigmas at patch time (`:331-332`); with its default end the window runs
  to the last step. `dense_blocks` and `min_tokens` gate as their names say
  (`:68-82`).
- **Sink modes** `exact_kv`, `exact_kv_and_rows` (the default) and `off`
  (`:402-406`): the packed prefix is an exact KV range for every query, and the
  second mode also runs the target-audio query rows dense (`sinks()`, `:84-100`).
- **Two execution paths.** Generic models go through the attention override,
  which calls `ck.sol_attn` on full Q/K/V and falls back to whatever override
  was on the hook before it (`:171-226`). **H3 goes through block patches**
  (`set_model_patch_replace(..., "dit", "double_block", i)`, `:343-345`) into
  kitchen's chunked producer: Q/K/V are projected in 4096-row slices straight
  into the kernel's int8 carriers and never built in full (`:248-307`), and
  each block's pooled `kmean` / `vscale` statistics are **carried over from the
  previous step**, per conditioning branch (`:263-270`, `:292-301`). The CUDA
  backend's `sol_attn_chunked` docstring at comfy-kitchen tag `v0.2.33`
  describes those two tensors as the last step's statistics, computed fresh
  when absent.
- **What core's model gained for it**: a `to_gate_compress` layer created when
  the checkpoint carries one (`comfy/ldm/minimax/model.py:168-171`, detected at
  `comfy/model_detection.py:415`, i.e. FastH3 VSA weights), and two
  `transformer_options` keys for attention patches, `minimax_h3_layout`
  (`comfy/ldm/minimax/model.py:623`) and `block_index` (`:754`).

*2026-09-19: PR 16239 was closed unmerged on 2026-09-16 and this paragraph no
longer describes a pending change; section "comfy-kitchen and core,
2026-09-19".*

**An open PR would rewrite this node (read 2026-09-11).** Comfy-Org/ComfyUI
PR 16239 (another developer; draft, unreviewed) adds a per-key "attention
measure" to both paths for a mixed-resolution pipeline, handed to kitchen as
`key_bias` (kitchen PR 171, below). Read against its diff: with no measure in
`transformer_options`, the non-VSA paths make the same `ck.sol_attn` and
`sol_attn_chunked` calls with the same arguments, and the pooled-statistics
key keeps its three fields (`pool_key` in the PR's new
`comfy_extras/sparse_attention_measure.py`). The VSA branch is restructured,
and every install registers a capability object into `transformer_options`
whether or not a measure is requested. If it merges, a record comparing
against core's node names the core commit it ran on.

---

## sglang's SubBlock router: two notes, read 2026-09-10

[`docs/research/sglang_h3_pipeline.md`](research/sglang_h3_pipeline.md) §11
owns SubBlock. Two things in the source it does not yet record:

- **A reserved sink block and a forced diagonal were tried and left out.** The
  router's docstring says both were measured on real H3 attention cells, that
  the diagonal barely moved the error and the sink helped only in a middle band
  of DiT layers that did not survive to the pixels
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse/router.py:212-215`).
- **The step and layer cutoffs**, already recorded at
  `docs/research/sglang_h3_pipeline.md:722-729`, now sit beside a second
  compute mode: `sage_fp8`, which on SM90 quantizes Q/K to INT8 and V and the
  softmax probabilities to E4M3
  (`coderef/sglang/python/sglang/kernels/ops/attention/subblock_sage_fp8_sm90.py:2-8`),
  added by `ffe98a4279` (2026-09-09) and not the default
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/subblock_sparse_attn.py:106`;
  the cutoff defaults and their stated sweep are at `:76-85` of the same file).
  *2026-09-25: `2a0cb2f04e` (#40116) extends `sage_fp8` to SM120 through a
  FlashInfer CuTe-DSL kernel and makes its default key sub-block count per
  arch. `DEFAULT_COMPUTE_MODE` is still `bf16`, and it now sits at `:105`.
  SubBlock still accepts only SM90, SM100 and SM120, so none of it runs on
  this card.*

---

## sglang's own Sol-Attn backend, read 2026-09-11

sglang has had a `sol_attn` DiT attention backend since `51470b376f` (#33702,
2026-08-09). It wraps NVLabs' `sol_attn` package from Sana's `sol-engine`
branch and is opt-in: sglang's default DiT backend is `fa`.
[`docs/research/sglang_h3_pipeline.md`](research/sglang_h3_pipeline.md)
section 14.9 walks the source with line citations, and
[`docs/research/sglang_comparison.md`](research/sglang_comparison.md) sets it
beside core's node and ours.

- **Defaults**
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/layers/attention/backends/sol_attn.py:59-83`):
  tau 1.0, `thresh_type` `diag`, `kv_splits` `auto`, `sink_tokens` 0,
  `sink_start` 0, `dense_steps` 10, `dense_layers` `"0,1"`, dense backend
  `fa` or `sage_attn`.
- **Step policy.** Dense for the first `dense_steps` step indices, sparse
  from there through the last step, with no end gate (`:128-140` of the same
  file).
  Tau, threshold, dense prefix and dense layers take the shape of the
  per-hardware cells' policy in `docs/SOLATTN.md`'s "Upstream H3 policies
  beside ours"; the sink does not.
- **Sink.** A fixed `sink_start`/`sink_tokens` range from the config.
  Nothing in sglang's H3 pipeline derives it from the packed layout, so by
  default no conditioning rows are exact.
- **Absent:** token routing, and a minimum sequence length.
- **INT8 QK** is passed only when the installed package accepts `int8_qk`
  and the device is SM89 or newer; the comment beside it says NVLabs'
  official API has no such argument (`:229-233` of the same file).
- **The documented H3 recipe** is `dense_backend=sage_attn,dense_steps=10`,
  with the text encoder on `torch_sdpa`
  (`coderef/sglang/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx:1116-1119`).

---

## Sol-Engine's shared kernel package, and what separates it from comfy-kitchen's

Re-derived from source on 2026-08-16, pointers re-verified 2026-09-10.

### `thresh_type`: two modes, and comfy-kitchen implements one

Their two modes, in `coderef/Sana/techniques/sparse_backends/sol_attn/preprocess.py`:

- **`diag`** (`_compute_diag_threshold`): variance estimated as
  `sum_d q_d^2 * var(kc_d)` from the per-dimension variance of the key-block
  centroids, which assumes independence across the 128 dims.
- **`exact`** (`_compute_exact_threshold`): the real score row's
  `E[s^2] - E[s]^2` over all key blocks for that query.

comfy-kitchen's CUDA kernel computes `kcvar[d]` as the per-dimension variance of
the key-block centroids
(`coderef/comfy-kitchen-kijai/comfy_kitchen/backends/cuda/sage_attention/sol_attn_preprocess.cu:106`),
reduces `c_d^2 * kcvar[d]` (`:148`), and thresholds at
`tau * sqrt(var + 1e-6)` (`:54`), against mean-centred key centroids so the
mean term is already absorbed. **That is Sol-Engine's `diag` formula.** `exact`
is therefore a threshold kernel comfy-kitchen does not have, not an input it
leaves unexposed. *Pointers updated 2026-09-10*: they were `:122`, `:191` and
`:198`, which no longer land in that checkout.

The single-consumer-card cells (RTX 5090, RTX 4090) both run `diag`.

### `kv_splits`: SM90 only

`coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:116` raises
`"kv_splits=2/4 is currently available on SM90 only"`. A 4090 gets 1 in
Sol-Engine too. *Corrected 2026-09-10:* this also said "their own docs say
B200, RTX 4090 and RTX 5090 use `kv_splits=1`"; no such document was found in
the tree at `757d902`, so only the code claim stands.

### The SM89 CuTe kernel

PR #464, merged 2026-08-15 (the dispatch commit is `9cfdd07`): "a general BF16
SM89 CuTe DSL Sol-Attn forward kernel using M64/N64 tiles, cp.async, and warp
MMA". That is the 4090, and it makes NVLabs' own kernel a **second independent
implementation** beside comfy-kitchen's.

The code is at `coderef/Sana/techniques/sparse_backends/sol_attn/sm89/`, with
dispatch at `coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:11`
(`(8, 9): "cute_sm89"`). The RTX 4090 cell above is its first published H3
use. Sol-H3's vendored copy does not carry it (see "The kernel it carries").

**It is installed and it runs on this box.** `vendor/build_sana_sol_sm89.sh`
installs the runtime, builds the `sol-attn` wheel out of the checkout, and then
compiles and exercises the kernel on this card; run it for the versions and the
deviations, which are printed rather than written down here.

Two things that section of their docs does not say.

**The requirements list is incomplete.** It names PyTorch, CUDA, Triton,
`cuda-python` and the CuTe DSL. Installing the DSL is not sufficient:
`apache-tvm-ffi` is named in no Sol-Attn requirements list and is not a
dependency of the DSL wheel either, yet `sol_attn/common/runtime.py` passes
`enable_tvm_ffi=True` on every tensor and `_compile_sm89` compiles with
`--enable-tvm-ffi`, so the first SM89 call dies on
`ModuleNotFoundError: No module named 'tvm_ffi'`. Found by running it,
2026-08-19.

**The compile-time failure discipline does not cover the import.** Their docs
warn that a DSL *version* mismatch fails at compile time rather than falling
back, "so that a dense run is never reported as a sparse one". An *absent*
runtime is a different path: backend selection returns `"triton"` whenever the
CuTe runtime is unavailable
(`coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:84-91`). That is
why the build script asserts `get_sol_attn_backend() == "cute_sm89"` before it
measures anything.

**The API surface.** Their public entry point
(`coderef/Sana/techniques/sparse_backends/sol_attn/interface.py:413-424`) is
`sol_attn(q, k, v, *, scale, tau, thresh_type, kv_splits, sink_tokens, sink_start)`.
There is **no `sink_q`**, no `dense_blocks`, no tail toggle, no `max_blocks`,
no `key_bias`. Their integration contract puts dense query rows at the caller:
"an MMDiT integration should still compute valid text query rows with dense
attention" (quoted in
`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py:21-24`).

### Token pruning and NVFP4

Two of Sol-Engine's five methods, and **not applied to their H3 stacks**.

- **NVFP4 needs Blackwell** (sm_100+) plus `transformer_engine`.
- **Token pruning** is model-agnostic scaffolding in
  `coderef/Sana/techniques/methods/token_prune.py` whose runtime lives in SGLang,
  and it needs a `ModelSpec` seam H3 does not have there. It ships on LTX-2.3,
  not H3.

---

## Morton in Sol-Engine, which does exist

Correcting a claim this repo made three times on 2026-08-15: **Sol-Engine does
ship Morton.** It is absent from the paper, absent from the published docs, and
present and on by default for Wan.

- `coderef/Sana/techniques/sparse_backends/sol_attn_backend.py` has
  `_morton3d_perm` (`:176`, "canonical x/y/z-interleaved") and
  `install_wan_morton_forward` (`:209`).
- The Wan Sol configs describe themselves as Sol-Attn "at tau 1.0 with
  **global Morton3D order** and dense guards"
  (`coderef/Sana/config/wan21_t2v_1_3b/wan21_sol_only.toml:4`, and the same
  phrase in the other Wan fullstack and Sol-only configs).
- HunyuanVideo has it **off**: `HUNYUAN_SOL_MORTON = "0"`
  (`coderef/Sana/config/hunyuan_video/full.toml:20`).
- H3 has it absent entirely.
- **They ship only the 3D curve.** There is no `2d_frame` variant anywhere in
  Sol-Engine; that is kijai's addition for H3's non-uniform temporal axis.

Their stated reason for H3 needing none, from
`coderef/Sana/models/minimax_h3/GB200/sol_attn_h3.py:8`, repeated word for word
in Sol-H3 (`coderef/Sana/models/minimax_h3/Sol-H3/h3_runtime/sparse_attention.py:8-12`):

> **Morton reordering per attention call.** The released
> `install_wan_morton_forward` docstring says why not: only self-attention is
> order-sensitive, so the permutation belongs once at the block stack, and
> doing it per call cost more than the kernel saved. What the H3 integration
> then shows is that H3 needs *no* reordering at all -- the packed video tail
> is already a contiguous grid-ordered block, and the routing works on it
> directly.

And the ablation control they ship for Wan, `config/wan21_t2v_14b/reorder_only.toml`:

> "reorder-only control: global Morton3D order with every layer forced dense
> through the Sol-Attn adapter."

**One H3 permutation now exists, and it is not a spatial curve.** Spark's
Ref2VA Sol draft permutes Q, K and V so visual rows come first and text-audio
rows last, keeping native order inside each part, to make the sink contiguous;
it records that block grouping changes (see the Spark section). *Corrected
2026-09-10:* this page said no Sol-Engine H3 path reorders tokens. That still
holds for spatial reordering; it no longer holds for token order as such.

**Whether any of that transfers to H3 is argued in
[`docs/morton.md`](morton.md), not here.**

---

## FirstBlockCache and TeaCache

`coderef/Sana/models/minimax_h3/A100/first_block_cache.py`: about 30 lines of
policy around a threshold, gated on `H3_FIRSTBLOCKCACHE=1`, threshold from
`H3_CACHE_THRESHOLD`, with decisions synchronized across ranks so a multi-GPU
run cannot diverge. The RTX 5090 and RTX 4090 cells use TeaCache instead
(`coderef/Sana/models/minimax_h3/RTX4090/teacache.py`; the retained and
cooldown steps in `coderef/Sana/models/minimax_h3/RTX4090/README.md:39`).

The idea, which is not theirs alone and not new: run the first transformer
block, look at how much its output changed against the previous step, and if
the change is below the threshold, reuse the step's residual instead of running
the remaining blocks. A step-skipping cache, orthogonal to attention.

Their thresholds are tuned at **50 steps**, and every cache cell above runs 50.
*Corrected 2026-09-10:* this section carried a frame-correlation figure for a
forecasting cache attributed to "the `h3-turbo-eval` prior art", with no
findable record; withdrawn rather than re-pointed.

---

## Other ComfyUI Sol-Attn packs

Found 2026-08-16, READMEs only.

**[Saganaki22/ComfyUI-sol-attn](https://github.com/Saganaki22/ComfyUI-sol-attn)**
is H3-specific: zero-copy MiniMax H3 nodes, published benchmarks, and
**scheduled tau** -- ramping the threshold across sampling steps, sparse early
to dense late, with selectable interpolation curves. Spark's Ref2VA draft
ramps the other way, dense first and sparser late.

- They default **tau to 1.3, "tuned locally", against the paper's 1.0**.
  *Corrected 2026-09-10:* this called it independent convergence "on the value
  we ship"; that was our value until 2026-08-20 and is not now.
- Their quality evaluation is **correctness against a reference** (L2 against
  SDPA, bit-identical strided views), not perceptual.
- Their speed numbers are on a 5090 with `torch.randn` tensors, which is both a
  different card and the synthetic-input trap `docs/SOLATTN.md` already carries.

**[sumeetprashant/ComfyUI-SolAttn](https://github.com/sumeetprashant/ComfyUI-SolAttn)**
exists and was not read past its README.

**[xmarre/ComfyUI-Sol-H3](https://github.com/xmarre/ComfyUI-Sol-H3)**, added
2026-09-11 from its README and the body of its PR 9 only, packages the Sol-H3
CuTe kernel from the author's own Sana fork and does not substitute
comfy-kitchen's `sol_attn` for it. Its supported kernel target is SM120 on
Linux or WSL2, so its sparse route does not run on this card. It composes
with a family of the same author's H3 packs (VDN, Spectrum,
Flow-Aligned-Regenerate and others), and PR 9 is the provider half of the
mixed-resolution work behind kitchen PR 171.

**Neither of the two 2026-08-16 packs reorders tokens.** With the paper and the Sol-Engine cells, no
searched source outside kijai's packs applies a spatial token order to H3.

---

## Comfy-Org/comfy-kitchen, as of 2026-09-11

Fetch and `gh` only. Nothing merged since the 2026-09-10 snapshot: `main` is
still `21003fa`, and there is no tag after `v0.2.33`.

- *2026-09-19: PR 171 was closed unmerged on 2026-09-16, so the rebase
  hazard below no longer applies; section "comfy-kitchen and core,
  2026-09-19".*
- **PR 171** (another developer; draft, no maintainer review, and its body
  says a compiled GPU run is still owed) gives `sol_attn_chunked` the
  `key_bias` the direct `sol_attn` already takes: a natural-log per-key score
  bias, folded into K-row quantisation so only exact blocks see it, with the
  pooled statistics left unweighted. Biased blocks must be sinks, and neither
  path checks it; the contract lives in comments. **Nothing here consumes
  it**: `sol_attn_h3.py`'s module docstring says why `key_bias` is not
  offered. The one contact is the fork. The PR puts `*, key_bias` exactly
  where the fork's chunked `blk_cnt` sits as a positional parameter, and its
  AST test requires the positional parameters to end at `token_aug`. On a
  rebase onto a tag that contains it, `blk_cnt` moves behind the `*`; the one
  caller here that passes it already does so by keyword (the
  `sol_attn_chunked` call in `sol_chunked_h3.py`).
- **PR 168 carries only the two `sol_attn` commits.** The chunked `blk_cnt`
  commit stays a fork delta even if 168 merges.
- **PR 172** (issue 136) moves the Triton INT8 GEMM's output offsets to
  int64. The int32 form faults once rows times output width passes the int32
  range, which H3's widest projection reaches at the shipped long 16:9 length
  (`workflows/h3_config.py::LONG_LENGTH`; rows per frame in
  [`h3_geometry_and_nodes.md`](h3_geometry_and_nodes.md); the width is
  `ffn_hidden_size` in `vendor_config/fl2va_transformer_config.json`, doubled
  by core's fused `fc1`). **It cannot reach this install, and the reason is a
  launcher default rather than our code**: core disables kitchen's Triton
  backend unless `--enable-triton-backend` is passed
  (`comfy/quant_ops.py:33-41`), this install's launcher does not pass it, the
  installed build has cuBLASLt so the CUDA backend takes `int8_linear`, and
  that path's quantise and dequantise kernels index with 64-bit offsets
  (`comfy_kitchen/backends/cuda/ops/int8_linear.cu` in the kitchen source).
- **PR 167** has not moved since `a63ca28`, and it is not VAE-only: it
  changes the launch block size of `quantize_int8_rowwise_convrot64_kernel`,
  the fused ConvRot quantiser `int8_linear` calls when `convrot` is on and the
  row width qualifies, and adds opt-in residual and RMSNorm arguments to
  `int8_linear`. [`open_experiments.md`](open_experiments.md) #28 carries what
  that means.
- Nothing else open touches H3 on NVIDIA. PR 134 changed on 2026-09-11 and is
  HIP-only; 124, 146 and 142 have not moved, and 142 tunes sm86 only.

## Comfy-Org/comfy-kitchen, as of 2026-09-10

Fetch and `gh` only. The branches are the authority; this is a dated snapshot.

- **No tag after `v0.2.33`.** `main` has two commits past it that matter here:
  PR 162 (`e3d714b`, persistent RoPE allocations) demotes the kernel's cached
  RoPE reference to a weakref and adds `set_allocation_context`
  (`comfy_kitchen/allocation.py` on `main`), a hook through which ComfyUI can
  hand kitchen a context for allocations that outlive a call. Its PR body says
  this takes ComfyUI's memory compiler to zero graph breaks in every sparse
  attention mode on H3. PR 165 adds a compile flag. ComfyUI core calls the hook
  only when kitchen exports it (`comfy/model_prefetch.py:63-64`).
- **PR 168** (`blk_cnt`, opened 2026-09-08 from the owner's fork) is open.
  **PR 167** (kijai's H3 VAE kernels) is open, and kijai's `minimax_vae`
  branch moved on 2026-09-09.

## Comfy-Org/comfy-kitchen, as of 2026-09-08

Checked in a working checkout on 2026-09-08, after ComfyUI moved its pin to
`comfy-kitchen==0.2.33` (core `18ebc2af`) and the stock wheel replaced our
build. **This section stood at 2026-09-04 and said PR 156 was open and that
the installed wheel predated the 150 merge; both are stale.**

**The fork's delta is four commits.** Everything we carried from kijai's
`sol_attn_continued` is upstream: PR 117 (int8 Sol-Attn), 143 (HIP port), 150
(`4950f16`, full-range P quantisation in the exact branch and a public
`sol_attn_chunked`), 156 (`b678fdf`, token routing) and 158 (its HIP port).
What is left that is ours is `blk_cnt`, the routed-count out-parameter the
route observer reads, on `sol_attn` and on `sol_attn_chunked`. It is a
Python-only change: the route stage already writes the count into the
workspace and the public API discarded it.

**The rebase target did not need re-grading.** The 2026-09-04 token routing
grade was taken against kijai's PR head `1128df6`, and the plan said a moved
head means the grade is redone. It did not move, it merged: `b678fdf` has the
identical tree to `1128df6`. Recorded, with the control that shows the diff
command can print something, in
`bench/results/2026-09-08_kitchen_0233_blk_cnt_rebase.json`.

**Which build is installed** is read, never written down:
`bench/check_sol_kernel.py` reports the local segment, and
`comfy_kitchen_build.json` beside the venv names the branch it came from, built
by `vendor/rebuild_kernel.sh`. *Corrected 2026-09-10:* this paragraph named the
installed build and branch as of 2026-09-08; the build has been rebuilt since,
which is why it now points instead. The clone's checked-out branch need not be
the built sha: the build record is what says which wheel runs.

**Carried for good, current by rule (owner, 2026-09-11).** The `blk_cnt`
commits stay in the fork whether or not PR 168 merges (and so, from
2026-09-15, do the `qk_balance` commits: the per-head q/k channel rebalancing
inside Sol's INT8 quantizers, `docs/h3_block49_quant_error.md` section 8,
not offered upstream until it has a blind pair behind it), and everything
else tracks upstream: the carried commits sit on the tag ComfyUI's requirements
pin, because a build on any other version stops satisfying that pin and a
requirements install puts the stock wheel back. `vendor/rebuild_kernel.sh`
refuses any other source, and `vendor/rebuild_kernel.sh --check` reports
whether the source is current without building.

**Upstream's own Sol suite does not pass on this card**, on the stock PyPI
wheel or on a local build of the same commit: three groups of nanobind
binding-validation cases whose expected error is raised only by the HIP
bindings, and `test_topk_ties_over_select` against its own cosine bar. One of
the binding cases leaves a sticky CUDA error, so a single gap reads as dozens
of failures in the same process. Attribution, the stock control and the
numbers are in the record named above. Not established there: whether these
fail on upstream's CI, which builds elsewhere with its own flags.

## Comfy-Org/comfy-kitchen, as it stood on 2026-09-04

Checked by fetch and `gh` on 2026-09-04; nothing was pulled into a working
checkout. This section is a dated snapshot and the branches are the authority.

**In `main`.** PR 117 (int8 Sol-Attn, 2026-08-29), PR 143 (the HIP port,
2026-08-31), and PR 150 (kijai, merged 2026-09-03 as `4950f16`): full-range
P quantisation in the exact branch, fp16 q/k/v/out through an `elem` code,
and a public `sol_attn_chunked`. The installed wheel then predated that merge
but was built from kijai's `sol_attn_continued`, which carried the same
exact-branch algorithm, so the merge moved nothing numerically on this box.

**Open then.** [PR 156](https://github.com/Comfy-Org/comfy-kitchen/pull/156)
(kijai, opened 2026-09-04, head `sol_token_aug_main`): a `token_aug` knob,
zero or a multiple of 64 up to 256. Each query block additionally attends up
to that many top-scoring individual tokens outside its routed blocks, and the
remaining tail is made exact for the block centroid; a new `sol_attn_token.cu`
does the selection, the eager reference ignores the knob, HIP warns and runs
without it. The PR body carries his numbers on an H3 capture and names the
defect it targets as a DC bias that shows as brightness pulsing on a
five-latent-frame period. **Our `sol-blk-cnt` commits did not cherry-pick onto
that branch**: conflicts in `comfy_kitchen/__init__.py`, both backend
`__init__.py` files and `backends/eager/sol_attn.py`; resolved by the
2026-09-08 rebase above. [PR 146](https://github.com/Comfy-Org/comfy-kitchen/pull/146)
(Sage INT8 consumer for the Sol exact branch, `tail=False` only) was open and
idle since 2026-09-01. [PR 124](https://github.com/Comfy-Org/comfy-kitchen/pull/124)
is not Sol but touches every int8 H3 render: it rounds the fused SwiGLU product
to the storage dtype before ConvRot quantisation, and
`docs/research/quant_levers.md` owns why that path is reached on the shipped
model.

---

## Published numbers, and why not to quote them against ours

Where upstream's H3 figures live, each with the conditions beside it:

- the per-hardware cells: each cell's README, e.g.
  `coderef/Sana/models/minimax_h3/RTX5090/README.md:13`;
- the RTX 4090 cell, including its TeaCache-then-Sol attribution:
  `coderef/Sana/models/minimax_h3/RTX4090/README.md:9-31`;
- Sol-H3 on B300: `coderef/Sana/models/minimax_h3/Sol-H3/README.md:36-65`;
- `super_acceleration` on GB200: `coderef/Sana/models/minimax_h3/super_acceleration/README.md:40-66`.

**None of these is a Sol-Attn-only number.** Full-opt stacks a cache,
compilation and, on the multi-GPU cells, context parallelism on top of Sol,
against a 50-step dense baseline; Sol-H3 compares a four-forward distilled
adapter against 49-forward baselines and says so; `super_acceleration`
publishes latency with no baseline. Ours is Sol-Attn against sage at our own
step count on one 4090. Different stack, different baseline, different step
count, and on all but one cell different hardware.

This is `docs/SOLATTN.md`'s unit trap at a larger scale. The RTX 4090 cell's
Sol increment is the closest thing to comparable, and it is an increment over
TeaCache from one sample. The one arm that *would* be comparable,
`rtx5090_sol` against `rtx5090_dense`, still has no published result.

## 2026-09-15, evening: `qk_balance` prepared for upstream

Branch `sol-qk-balance-pr` in the fork clone (`coderef/comfy-kitchen`): the
two `qk_balance` commits rebased by hand onto upstream `main`
(`e5e0d02`, the v0.2.34 tag), with the blk_cnt-adjacent lines dropped and
the two tests that read the route through `blk_cnt` restated on the
output. Built against that base (sm_89) and its Sol suite run from the
wheel: the ten new cases pass, the whole suite fails only the pre-existing
`test_topk_ties_over_select`. Not pushed, no PR opened; the PR text is
`docs/sol_upstream_qk_balance_pr.md`. The owner pushes and opens it, per
the standing rule; `gh pr create` needs `--repo Comfy-Org/comfy-kitchen
--base main --head <fork>:sol-qk-balance-pr` spelled out.
