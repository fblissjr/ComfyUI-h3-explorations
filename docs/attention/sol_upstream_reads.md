# Dated reads of comfy-kitchen and ComfyUI core

Detail page of [`sol_upstream.md`](sol_upstream.md), which holds the current state of each upstream. This page logs
each pass over comfy-kitchen and core, newest first: what moved, and what it meant for this repo on that date. It is
history, not guidance; where a later pass or the code says otherwise, they win. The other clones' passes are in
[`../working/coderef.md`](../working/coderef.md), one section per checkout, and sglang's in
[`../research/sglang_comparison.md`](../research/sglang_comparison.md).

**What still bears on this repo:**

| item | read |
|---|---|
| core can run kitchen's Sol-Attn with no node in the graph, on a model file's or a LoRA's say-so | 2026-10-09 |
| core PR 16681 would have broken `MiniMaxH3LoRABranch` on fc2; this pack follows it since 0.184.10 | 2026-10-02 |
| a clip rendered before the 2026-09-22 core pull is not bit-comparable with one after it (VAE tile blending, `fc584aaa`) | 2026-09-25 |
| core PR 16508 (fp16 H3) is the one open core PR worth watching | 2026-09-25 |
| a model file can name a block's attention (`c194dd00`); no shipped model carries the key | 2026-09-25 |
| kitchen PR 179 (`draft_attention`) is the one open kernel that would run on this card | 2026-09-19 |
| kitchen PR 168 (ours, `blk_cnt`) is open with no activity since 2026-09-11 | every read |

## comfy-kitchen and core, 2026-10-09

Read with the ComfyUI checkout at `08ff3c11`, against `65787d66`, and with
`gh` for kitchen. The account is in
[`docs/working/coderef.md`](../working/coderef.md), "ComfyUI core" (its current state; the dated read of
2026-10-09 is in git at `901bee1`). The one thing a reader of this
page needs from it: **core can now run kitchen's Sol-Attn with no node in
the graph**, on a model file's or a LoRA's say-so (`b26625f2`, #16831;
`f49c531e`, #16880), at kitchen's defaults but for the threshold. "ComfyUI core's own Sol node" in [`sol_upstream.md`](sol_upstream.md) describes `BlockSparseAttention`, which is a
different door to the same kernel and did not change.

## comfy-kitchen and core, 2026-10-02

Read with `gh` (compare, PR lists, bodies and diffs) and the ComfyUI checkout
at `65787d66` (pulled 2026-10-02; `git reflog` in it dates each pull),
against `88ab4a06`, the 2026-09-25 read. The next server start runs this
checkout.

**Nothing below changes a default this repo ships, and nothing triggers the
adopt-upstream rule.** No H3 template moved a widget value, and sglang moved
no default ([`docs/research/sglang_comparison.md`](../research/sglang_comparison.md),
"Eighth read").

**Kitchen: current on that day.** `vendor/rebuild_kernel.sh --check` reported the build
installed and current against ComfyUI's pin, which core moved to `0.2.36`
(`6f6a2742`, #16635). The build's contents are recorded in
`bench/results/2026-10-01_kitchen_merge_aade8d5.md`. Both have moved since; read the
installed build as [`sol_kernel.md`](sol_kernel.md) says. Between that build and the one
installed on 2026-10-03 the fork's log shows upstream's two Ascend commits and a version
bump, and no change to what we carry (`git log aade8d5..6b42fab` in the clone). Upstream main's two
commits past the build, `5eeefcf` (#201) and `d0da34b` (#199), are Ascend.
An open stack of H3-targeted kitchen PRs splits two ways. 217, 218, 222, 227
and 231 dispatch only on SM120, which is not this card. 219, 220, 221 and 223
add opt-in indexed-gate, fused-norm and BSHD-output APIs that core does not
call yet; core's 16681 and 16678 below are their consumers. 224 fixes INT8
attention for a nonpositive scale, which H3 never passes. **168 (ours)** is
still open with no activity since 2026-09-11.

**Core, merged since 2026-09-25:**

- **`2d6b7328` (#16677) is an extract-method refactor of H3's forward.** The
  patchify, cond-row scatter, patch projections, condition projection and
  token refiner move from `MiniMaxH3Model._forward` into
  `comfy/ldm/minimax/model.py::MiniMaxH3Model._embed_and_pack`, so their
  temporaries are freed before the blocks run. Statement order is unchanged
  and `Attention.forward` did not move. No forward in this pack copies that
  span, so nothing here follows it. A DiT peak recorded before this pull
  describes the older core. The line citations in "ComfyUI core's own Sol node" in `sol_upstream.md` and in `h3_capture.py` were re-read against it.
- **`83071e1a` (#16698) fixes a crash decoding H3 video with the VAE
  offloaded.** `FeedForward.forward` and `Attention.forward` in
  `comfy/ldm/minimax/vae.py` passed the pre-norm weight uncast into
  `linear_input_act`'s fused RMSNorm, so an offloaded weight sat on the CPU.
  It now goes through `CastBiasWeightContext`. The shipped INT8 decoder
  (`h3_config.MODELS["video_vae"]`) takes that path, and `start.sh`'s `safe`
  mode offloads. The failure was loud, so no clip that decoded is affected.
  Whether the cast is a bit-level no-op for a resident VAE was not checked.
- **`986c4d15` (#16657) loads H3 video VAEs with a shorter decoder.**
  `comfy/sd.py` counts the decoder's transformer blocks from the file and
  passes the count to `MiniMaxH3VideoVAE`. Its first user is LynnReal's
  "light" VAE: the stock encoder with a distilled decoder of fewer blocks,
  whose model card says the latent space is unchanged and the RGB output is
  not. It is a full decoder, not a preview one, and stock `VAELoader` loads
  it. The shipped files load as before. Nothing here has measured it.
- `8d534945` (#16595) adds `comfy_attention` to other models; H3's
  `Attention` already passed `preferred_attention`. W6A8 (`491bf3b9`,
  `9671d89f`), the RGBA fix (`39cb5a8b`, the 16548 of the last read) and a
  `FixedKVBias` write in `generate()` (`a7169322`) have no reach. ComfyUI
  `v0.38.0` was tagged 2026-09-29. The rest is other models, assets, partner
  nodes and EXR.

**Templates.** Since 2026-09-25 the H3 files in Comfy-Org/workflow_templates
changed only their download notes (`98fd32ca`, `e7cd011d`); a widget-value
diff of the H3 and FastH3 t2v templates before and after shows no change.
`e7849852` adds API copies of those two graphs under `benchmarks/`, and
`9b912856` removes the site's H3 demo page. Core bumped the package through
`v0.11.74` (`fa98a189`).

**Open core PRs, new or moved since 2026-09-25.** Keyword search as before,
so a PR touching an H3 path under an unrelated title can be missed.

- **16681 would break `MiniMaxH3LoRABranch` on any LoRA that touches fc2.**
  It changes `MLP.forward(x)` to `forward(x, residual=None, gate=None,
  segments=None)`, and `DiTBlock.forward` then calls
  `self.mlp(h, residual=x, gate=gate_mlp, segments=mod_segments)` unless the
  MLP carries a hook or core is training. `lora_branch.py::_mlp_forward`
  installs a one-argument `forward(x)` as an object patch, which is not a
  hook, so the call would raise a TypeError. The same copy is in
  `standalone/h3_mutant_distill/exact_lora.py`. Read from the diff and
  reasoned; nothing has run under the PR. The fix is to accept the three
  keywords and apply core's `_mod_gate` when `residual` is given;
  [`docs/state/next_steps.md`](../state/next_steps.md) carries it. *Made the same day
  in 0.184.10.*
- 16712 and 16713 rework `_embed_and_pack` again and move reference-row
  preparation into `extra_conds`; their bodies report identical latents.
  Nothing here mutates `cond_*_latents`. `bench/probe_ref_rows_16604.py`
  calls `_cond_video_rows` and would need to follow 16713. 16678 waits on
  kitchen 220. 16688 duplicates `83071e1a`. 16619 fixes async-offload cast
  aliasing, including the H3 audio VAE, and reaches only `--async-offload`.
- 16508, 16460 and 16542 have not moved. 16156 reworded its 2026-09-27
  commit and keeps the exclusive end. 16391 is still open though superseded.


## comfy-kitchen and core, 2026-09-25

Read with the kitchen clone's `upstream` and kijai's fork fetched, `gh` PR
lists, bodies and diffs, and the ComfyUI checkout at `88ab4a06` (pulled
2026-09-25; `git reflog` in it dates each pull). Core commits are read against
`3c80da7f`, the 2026-09-19 read. No ComfyUI server was running at the time of
the read, so the next start runs this checkout.

**Kitchen.** Upstream `main` was past `v0.2.35` and untagged, which the build
policy of that day did not build. Since 2026-09-27 the build tracks upstream main
(`h3-frontier`, `vendor/rebuild_kernel.sh` "Track upstream main"), and everything
below is in it from `0.2.35+sol.863e953.up.c8c7825`:

- **`ef40891` (#192, kijai) will reach H3's video VAE encode on the next
  tag.** Its `zero_pad` and `out=` work is for the SeedVR2 VAE, which closes
  the 2026-09-19 note that the fork commit named no model. The same PR also
  raises the depth gate on `fp16_conv3d`'s fp16 accumulation. Core's H3 VAE
  calls that kernel when fp16 accumulation is on (`comfy/ldm/minimax/vae.py`,
  the `fp16_conv3d` helper, gated on `comfy/ops.py::_fp16_linear_wanted`).
  This install's launcher turned fp16 accumulation on (`start.sh`,
  `--fast fp16_accumulation`) until 2026-09-28, and passes no `--fast`
  since. The reader counted the encoder's 512-channel
  3x3x3 convolutions that fall between the old gate and the new one; the
  decoder has none. So after that rebase, reference and keyframe encodes on
  those stages move to fp16 accumulation. The PR body reports the fidelity
  cost for SeedVR2's decoder. Nothing here has measured it for H3. The
  rebuild record for that tag should say so.
- `f61028a` makes kitchen's `is_available` functions return False on a CPU
  device, a semantics change to carry through the next rebase. #186 adds head
  dim 256 to the Flash kernel, which is not H3's. #191 and #193 add W6A8 (no
  model in `workflows/h3_config.py::MODELS` uses it: every DiT and encoder
  header is `int8_tensorwise`, `bench/results/2026-09-25_upstream_survey_checks.md` check 3). #194 is HIP INT8 attention; the rest is
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
  (`bench/results/2026-09-25_upstream_survey_checks.md` check 2), so no shipped graph
  moves. Two copies of core's forward in this pack would react differently
  to a file that did carry it. `exact_blocks.py::_exact_forward` strips the
  override and calls core's `Attention.forward`, so its "exact" blocks would
  run kitchen INT8. The sage node's forward
  (`attention.py::make_minimax_attn_forward`) does not pass
  `preferred_attention`. Both are inert today.
- **`b16023b0` (#16457) removes torchaudio from core's requirements.** Core's
  H3 reference-audio resample now calls `comfy/audio.py::resample`. This pack
  still imports torchaudio lazily at its resample sites
  (`grep -n torchaudio audio_freeze.py reference_conditioning.py`), and `pyproject.toml`
  declares no dependency. So a fresh install without torchaudio fails only
  when a clip's sample rate differs from the audio VAE's. This box still has
  torchaudio. `comfy.audio.resample` returned tensors `torch.equal` to
  `torchaudio.functional.resample` on CPU in fp32 (`bench/results/2026-09-25_upstream_survey_checks.md` check 1,
  with the conditions it did not cover). That makes it a drop-in
  replacement there. *Made the same day in 0.139.0 (`audio_resample.py`),
  after the check was extended to CUDA, fp16 and bf16.*
- `95539f56` (#16471) Fun-ControlNet Union 2.0 for H3; nothing here wires a
  ControlNet. ComfyUI `v0.37.0` was tagged 2026-09-20. The rest is other
  models, assets, partner nodes and NPU/ROCm.

**ComfyUI's workflow templates switched H3's video VAE to INT8, and so, later, did this repo.**
Comfy-Org/workflow_templates `fc427f00` (#1280, 2026-09-22) points every H3
template, the FastH3 ones included, at
`minimax_h3_video_vae_int8_convrot.safetensors`. The owner had removed that file
from this repo on 2026-08-21 and reopened the lane on 2026-09-26;
`workflows/h3_config.py::MODELS` names it as `video_vae` again, and the comment beside
it says why. Templates are not one of the adopt-upstream rule's three upstreams. The
core decoder has an INT8 attention branch for such a file
(`comfy/ldm/minimax/vae.py`, `Attention.forward`, the `QuantizedTensor` test).

**Open core PRs, new or moved since 2026-09-19.** Found by keyword search
(`gh pr list --search`) on H3, MiniMax, sparse attention, Qwen3, kitchen and
the new attention key, with no file filter, so a PR that touches an H3 path
under an unrelated title can be missed.

- **16508 (fp16 inference for H3) moves the DiT to fp16 only under
  `--fast fp16_accumulation`.** It adds `torch.float16` to H3's
  `supported_inference_dtypes`; that flag sets `PRIORITIZE_FP16`
  (`comfy/model_management.py`), and with it and the PR's list `unet_dtype` and
  `unet_manual_cast` both choose fp16 for a quantized checkpoint, which the PR body's
  quantized-checkpoint exemption does not cover; every graph loads the DiT with
  `weight_dtype` at default. Both were called in-process
  (`bench/results/2026-09-25_upstream_survey_checks.md` check 4). This install's
  `start.sh` passes no `--fast` since 2026-09-28, so both calls choose bf16 with or
  without the PR (check 4's other three cases). Since 2026-09-27 `MiniMaxH3Sol` takes
  fp16 (the direct kernel entry accepts it; `sol_attn_h3.py::_apply_sol`), so an fp16
  DiT under this PR would run Sol. This pack's own attention forwards lack the PR's
  fp16 `out_proj` rescale (reasoned from the code, no render under the PR). It is the
  one open PR here worth watching.
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
  the author's fork. The `*, key_bias` rebase hazard and the reading that 16239 would rewrite core's Sol node no longer apply. Neither came back under a new number: the same author's 2026-09-18
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
  `qk_balance` branch (`sol-qk-balance-pr`, [`sol_upstream.md`](sol_upstream.md), "comfy-kitchen") has no PR.

**kijai's fork.** `zero-pad-mode` (`16651db`, 2026-09-19, not a PR yet) adds a
zero-padding mode to the fused VAE group-norm, SiLU and pad3d kernel from PR
167, and touches the fp16 conv3d path. The commit does not name the model it
is for.

**Core, merged since 2026-09-11 and running in this install** (the ComfyUI
checkout is at `3c80da7f`; `git reflog` in it dates each pull):

- **16187 and 16332: the H3 VAE kernels are live.** With kitchen PR 167 in
  `v0.2.34`, both halves of [`docs/state/open_experiments.md`](../state/open_experiments.md) #28
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
  `h3_config.graph_paths(..., include_bench=True)` found only `MiniMaxH3SolAttn`, the node of that date).
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
  reference goes encoder-only (`docs/h3/references.md`, "Encoder-only
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
`quant_q_rows`, where the carried `qk_balance` and `rotate` commits live ([`block49_quant_error.md`](block49_quant_error.md)).
