# The sister checkouts: what each one is good for

`coderef/` holds the reference implementations; `ls -l coderef/` is what is on
disk (some symlinks, some real clones). This page says what each is *for*: what
it implements, what has been compared against it, at which revision, and what it
is not evidence of. Each checkout's section is its current state at the revision
it was last read; earlier reads are in git history.

**Where things stand (last full read 2026-10-09).** No upstream moved a default
this pack ships, and nothing triggers the adopt-upstream rule (root `AGENTS.md`,
"A tinkering repo": only sglang, core's node and comfy-kitchen can). The open
items from the reads, in order of what to act on:

1. A model file or a LoRA can switch core's own Sol-Attn on with no node in the
   graph ("ComfyUI core"); `check_model_contents.py` watches the files this pack names.
2. FastVideo published a single-4090 path for FastH3 that is neither the contract
   nor a public checkpoint ("FastVideo").
3. Two masking ideas the masked lane has not tried ("Masks and edited video
   upstream").

**Rules for using them.**

- **A revision is an observation point, not a pin.** A sister checkout moves under
  you; re-read before quoting. Two H3 engines moved during the 2026-08-28 pass itself.
- **Do not import Python from `coderef/`** (root `AGENTS.md`, "The lanes"):
  requiring the clone and prepending it to `sys.path` made a bench script
  unrunnable on a box that had the wheel and no checkout. Use the clone for
  sources you cannot import; import the rest.
- **Search with `find -L` and `grep -r --dereference-recursive`.** More than half
  of `coderef/` is symlinks, so a plain search answers about a minority of it.
- **One reference lives in this repo instead**: `bench/_sol_attn_reference.py`,
  the Sol-Attn reference for what can be imported. (`vendor/sol_attn_minimax.py`,
  the old read-only reference node, is no longer in `vendor/`.)
- **How a read is done.** Fetch each clone's upstream branch; the commit list is
  `git log <recorded>..HEAD` inside it (Sana: `origin/sol-engine`). Diffs are read
  for the commits named below and titles for the rest. Sol-side movement (core's
  Sol node, comfy-kitchen, kijai's fork, core's PRs, ComfyUI's workflow templates)
  is in [`sol_upstream.md`](../attention/sol_upstream.md); sglang's numbered
  reads are in [`docs/h3/sglang_comparison.md`](../h3/sglang_comparison.md).

## A distill's reference is its trainer's contract

**A distilled checkpoint or LoRA runs the way its trainer says**: the sampling
contract the trainer ships outranks any downstream template, ComfyUI's included
(the owner, 2026-09-26). When they differ, the contract is the default and the
template is an arm. Root `AGENTS.md`'s adopt-upstream rule covers defaults sglang
and ComfyUI agree on; this rule covers a distill's sampling, where the trainer is
the one source with the answer.

What counts as the contract:

- a settings file in the release (FastVideo's `fastvideo_inference.json`);
- a release's merge script and schedule (FlashGen's `merge_lora_ckpt.py` and
  `base_schedule`);
- a serving engine's validator that pins those values
  (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/fasth3_checkpoint.py`
  checks FastH3 V2 against its file);
- a trainer's own inference script and the scheduler it pins (the one instance was
  PDMD's `worker/run_a10.py`, retired with its lane on 2026-10-05).

The instance that earned it: FastH3 V2 was first built to ComfyUI's template, which
departs from FastVideo's contract on the sampler, the VSA kept fraction and a dense
warm-up. The contract arms rendered a different and more complete take
(`../../bench/results/2026-09-26_fasth3_contract_s1.md`). FlashGen was already on
its contract's schedule (`docs/research/2026-09-26_flashgen.md`).

---

## The H3 implementations

These implement MiniMax H3 end to end. The first four were compared against our
node chain on 2026-08-28; the findings are
[`custom_node_gaps.md`](../h3/custom_node_gaps.md), which this page routes to
rather than restating.

| checkout | last read | what it is | reach for it when |
|---|---|---|---|
| `sglang` | `f48ed2127e` (2026-10-09) | **the vendor's own serving path**, the closest thing to ground truth for what MiniMax intended | you need to know what the release does at a stage |
| `vllm-omni` | `4c5541cfc` (2026-10-09) | a serving engine with the most H3 activity of any clone | continuation, latent-mask editing, a second engine's contract validator |
| `LightX2V` | `b6d38283` (2026-10-09) | inference engine; origin of the SLA work and the Turbo LoRAs (both lanes closed: turbo 2026-09-26, SLA 2026-09-27, [`roadmap.md`](../state/roadmap.md), "Closed lanes"; `bench/check_distill_settings.py` fails a graph that loads a Turbo LoRA) | SLA, DMD step distillation, offload, a causal streaming runtime |
| `FastVideo` | `2b164405` (2026-10-09) | FastH3's trainer | the FastH3 V2 contract (`workflows/h3_config.py::FASTH3_*`) |
| `DiffSynth-Studio` | `acf2ad2` (2026-10-09) | model library with a native H3 pipeline, its own converters and a LoRA path | a second opinion on a state-dict namespace or a converter |
| `diffusers` | `1d5d056ec` (2026-10-09) | model library with a native H3 pipeline, a named H3 scheduler and a conversion script | the canonical tensor namespace, or a clean statement of the sampler |

For sglang, [`docs/research/2026-08-25_sglang_h3_pipeline.md`](../research/2026-08-25_sglang_h3_pipeline.md) (what it
does stage by stage) and [`docs/h3/sglang_comparison.md`](../h3/sglang_comparison.md)
(what its serving path does that we do not) are the authority over anything here.

**What the 2026-08-28 pass established about the references themselves:**

- **Three-against-one is a real signal, and it fired twice.** sglang, DiffSynth
  and diffusers agree on feeding one prepared reference tensor to both towers, and
  on running the video VAE more precisely than we do.
- **Agreement is broad.** The reference label rules, the encoder layer, the
  absence of a chat template and the VAE normalisation statistics all agree. When
  four implementations agree, a fifth reading is not the cheapest next step.
- **Neither model library is independent evidence about the seven markers.** Both
  inherit the release tokenizer without touching the ids in code.
- **PDD**: sglang implements it since `973fb44471` (#40568, 2026-09-23); diffusers,
  LightX2V and DiffSynth do not
  ([`docs/research/pdd/2026-08-28_pdd_implementations.md`](../research/pdd/2026-08-28_pdd_implementations.md)).

### sglang

At `f48ed2127e`. No base-H3 default moved
(`coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::MiniMaxH3SamplingParams`).

- **PDD** (`973fb44471`): the same dt-weighted head fusion as ours
  (`pdd_math.py::fusion_plan`), at `h3_config.PDD_SHIFT` and `h3_config.PDD_STEPS`
  evaluations, over a fixed uniform partition fused offline. The seventh read has
  the comparison and a probable gate/value swap in its offline fc1 merge.
- **FastH3 is the 8-step V2 checkpoint on the trainer's contract** (`4f320fb85a`,
  #37662): grid points, trained rungs and shifts from the checkpoint's metadata;
  Euler, kept share and tile pinned in
  `coderef/sglang/python/sglang/multimodal_gen/configs/sample/minimax_h3.py::FastH3SamplingParams`
  and
  `coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py::FastH3PipelineConfig`;
  t2va only, no dense opening steps. A second engine on the values
  `workflows/h3_config.py` holds as `FASTH3_CONTRACT_POSITIONS`, `FASTH3_SHIFT`,
  `FASTH3_CONTRACT_SAMPLER` and `FASTH3_CONTRACT_VSA`. Its sparse backend refuses
  this card
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/platforms/cuda.py::_VideoSparseAttentionH3BackendResolver`).
- **Its default quality tier changed meaning and kept its name** (`7a4d6dfd2f`,
  #42370): the reference path is `exact`, and `lossless`, still the default, allows
  fused kernels
  (`coderef/sglang/python/sglang/multimodal_gen/configs/sample/sampling_params.py::QUALITY_LEVELS`).
  For H3 the denoise is the same; the difference is the video VAE's fused decode
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/decoding.py`,
  the `quality_allows` call). A sentence written before 2026-10-09 that calls
  sglang's default "lossless" names the older, stricter tier.
- **AdaLN rounding: sglang and core agree.** Its bf16 block kernel
  (`coderef/sglang/python/sglang/kernels/ops/diffusion/modulate/indexed_modulation_triton.py::_indexed_scale_shift_bf16_kernel`)
  rounds to bf16 at `1+scale`, at the product and at the stored sum, the same three
  points as core's three in-place bf16 ops
  (`comfy/ldm/minimax/model.py::_mod_scale_shift`). vllm-omni's fp32 path is the
  outlier ("Eighth read").
- **It runs the H3 DiT under a ComfyUI graph** (`f1e62e3a2e`, #35990): core builds
  the model and runs the encoder, VAEs, conditioning and sampler, and an sglang
  worker runs each DiT step. None of our DiT-side patches would reach a DiT run that
  way. `fbce0d9478` lets that worker read serialized INT8 files.
- **Kohya LoRA keys** (`f048d5aa4b`, #35857):
  `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/lora/format_adapter.py`
  rewrites `lora_unet_blocks_N_*` names; core's loader already maps them
  (`comfy/lora.py::model_lora_keys_unet`), and `MiniMaxH3LoRABranch` reads them
  through core's table since 2026-10-05 (`lora_branch.py::native_keys`). No
  Kohya-trained file has been run here.
- **Things core already does**: the VAE decoder's fused RMSNorm and QK RoPE
  (`284cda01fb`; core through kitchen in `comfy/ldm/minimax/vae.py`,
  `ck.rms_rope_split_half_` and `linear_input_act(..., "rms_norm")`), no full-video
  clone after decode (`8663e3b670`), streaming an oversized DiT (`a977e3b9d5`).
- **Noted, not proposed**: sparse routing in head chunks (`1093c501df`, #41985),
  which would lower peak VRAM in our dense-QKV `sol_attn` path on long clips, not
  speed it (our chunked producer chunks rows); an explicit QKV layout override in
  its loader (`efb62ce269`), which matches the row-order question in
  [`custom_node_gaps.md`](../h3/custom_node_gaps.md) without closing it; an opt-in
  Spectrum skip-step, `kitchen_int8` renamed `convrot_int8`, an exact conditioning
  cache, t2va RL rollout ("Eighth read").
- **VDN-H3** is OpenVDN's own checkpoint, t2va and fl2va, refused on base weights
  ("Sixth read"); the cookbook lists Ampere and Ada as enabled and not benchmarked.
- **Reference stills** go to its short edge with upscaling on
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py`),
  which is what `MiniMaxH3AppendRefImage`'s parity defaults follow.
- It has no mask path ("Ninth read", 2026-10-08).

### vllm-omni

At `4c5541cfc`. Not one of the adopt-upstream rule's sources, so any difference is
an ordinary judgement call.

- **Its ComfyUI pack does not run H3 in ComfyUI.**
  `coderef/vllm-omni/apps/ComfyUI-vLLM-Omni` sends requests to a vllm-omni server
  (`coderef/vllm-omni/apps/ComfyUI-vLLM-Omni/docs/minimax-h3-t2v.md`). Its example
  graphs (`coderef/vllm-omni/apps/ComfyUI-vLLM-Omni/example_workflows/`) are templates for that server's sampler:
  deterministic Euler on a shifted linear schedule with no CFG
  (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/time_request.py`).
  They differ from ours on base step count, sampler, frame count, the canvas used
  with a reference video, and the 768p Turbo LoRA's version, step count and
  strength; they agree on shift, fps, the base canvas and no CFG.
- **Continuation from the latent tail, as guide rows** (`139a47a57`, #7838). The
  previous window's sampled video and audio tail goes in as extra conditioning rows
  (a `latent_guide` ref block) sharing the new window's time origin; the overlap is
  regenerated and discarded; every window's positions are offset onto one global
  clock; audio boundaries are rounded on the cumulative timeline. Nothing is
  decoded. `audio_mode=lock_source` pins one encoded track in the target audio rows
  at timestep 1.0 every step, which is our mask freeze at `audio_mask` 0 with
  `track_latent`. This is the guide arm of
  [`audio_freeze.md`](../h3/audio_freeze.md) section 5 idea 5. Core can express the
  guide half (its packed layout reads a latent from `minimax_keyframes`); it has no
  temporal offset, but a pack can add one without editing core. The docstring
  credits the algorithm to `ttulttul/ComfyUI-Minimax-H3-Continuation`; the recipe
  credits `vizart-vj/ComfyUI-MiniMax-H3-LongMedia` for the global offset only. All
  three packs are compared in
  [`2026-09-25_continuation_guide_rows.md`](../research/2026-09-25_continuation_guide_rows.md)
  (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/continuation.py::diffuse_continuation`,
  `::resolve_continuation`; `coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/packed_sequence.py`;
  `coderef/vllm-omni/recipes/MiniMaxAI/MiniMax-H3.md`, "Long video and driving audio").
- **AdaLN in fp32 on every arch** (`84977d954`, #7913, which reversed #7693's
  SM90-only bf16 rounding;
  `coderef/vllm-omni/vllm_omni/diffusion/layers/indexed_modulation.py`). See
  "sglang": sglang and core agree, this is the outlier. The same commit turns TF32
  off for the keyframe encode on SM90 only, so
  [`open_experiments.md`](../state/open_experiments.md) #30's `allow_tf32=True`
  quote is arch-conditional; #30 stays closed. `30d6a0b4e` (#7191) pins cuDNN flags
  around the keyframe encode (#30).
- **FastH3 V2** loads `FastVideo/FastVideo-FastH3-8-Step-V2` in Diffusers layout
  on its own ladder and shift (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/fasth3_checkpoint.py`, the contract
  validator above). The Ascend four-step path (`817f5d0e4`, #7149) writes the same
  ladder (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/fasth3.py::FASTH3_BASE_SCHEDULE`); `a43cdcdee` adds
  capability-12 FlashInfer kernels for its tile.
- **Its H3 VSA is in the model** (`507cb1d83`, #7535): the same cube tiling and
  gated coarse branch as `vsa_attention.py`, keeping a fixed count of video tiles
  where ours keeps a fraction (`h3_config.VSA_KEEP_PERCENT`).
- **Steps count evaluations** (`67aa30c96`, #7219), as LightX2V and ComfyUI's
  `steps` do; its own client templates still describe sigma-point counts, which its
  server rejects for the Turbo files it validates (read, not run). `af74a5a15`
  (#7062) derives a Turbo file's sampler contract from its filename and scales by
  declared alpha over rank, matching ComfyUI.
- **Reference stills keep their own size** (`7266fc613`, #8253), rounded per axis
  to 32 inside validation bounds
  (`coderef/vllm-omni/vllm_omni/model_executor/models/minimax_h3/preprocessing.py::resolve_minimax_h3_reference_image_shape`),
  where core's `max` mode shrinks at its short edge. A reference video goes on the
  canvas only when it has at least the canvas's pixels
  (`coderef/vllm-omni/vllm_omni/model_executor/models/minimax_h3/reference_video.py::_reference_video_shape`), core's rule. The
  PR's evidence is wall time, not quality. One more data point for the unrendered
  `allow_upscale` arm in `bench/refview2_arms.json`, not a verdict; the append
  node's parity default stands ([`evidence.md`](../method/evidence.md), "Settled
  about H3").
- **Latent super-resolution and a hi-res refine pass** (`a038b3817`, #8322): an
  optional community 3D-conv latent upscaler (`LBH-123-AI/Minimax_h3_latent_Upscaler`;
  `coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/latent_upscaler.py::MiniMaxH3LatentResizer3D`), and
  `latent_refine`, which re-noises to shifted sigmas and reruns the tail at the
  larger size (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py::_refined_latents`). Both off
  unless a checkpoint is set; the refine runs the DiT above the trained canvas
  ([`docs/h3/geometry.md`](../h3/geometry.md)). Whether core's
  `LatentUpscaleModelLoader` loads the file is unchecked.
- **Exact AdaLN projection cache** (`535bd35b4`, #7987), on by default, keyed on a
  digest of the whole `t_emb` plus settings, so hits come across requests with the
  same schedule, not across steps; the second engine after LightX2V's `95e9b86b`.
  Core recomputes.
- **Latent-mask editing** (`5a93ec1b4`, #7465; frame-space masks `25fb25766`, #7947;
  client `bad88bca2`, #7575): "Masks and edited video upstream".
- **VDN-H3** (`9cc105d1d`, #8439) as `VDNH3_ATTN`: the same OpenVDN checkpoint as
  sglang's, as two LoRAs fused over fl2va plus the linear branch
  (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/vdnh3.py::VDN_TASKS`), refused on base weights; no
  compute-capability gate, Hopper documented.
- **Portable prompt skills** (`f8a00b149`, #7923) keep shot-header timestamps,
  which the house rule forbids, and bind `<Picture 1>`/`<Picture 2>` for FL2VA where
  the vendor guide's alignment line is bare. The vendor guides win.
- `dbd6a35dd` (#8008) refuses a LoRA whose modules do not all bind; core's
  `comfy/lora.py` warns and continues. `3d952d133` (#7281) gives a reference
  video's soundtrack and standalone reference audio separate duration budgets.
  `43b8de9b0` (#7167) fuses q/k RMSNorm and RoPE in the video VAE on SM90+.

### LightX2V

At `b6d38283`. Nothing under `configs/minimax_h3` changed in the last three reads.

- **DPCache** (`8652c6f1`, #1557): on unselected steps it skips the block stack
  and extrapolates its output by a Taylor series, steps chosen by a dynamic program
  over a calibration run keyed on the Sol settings. Skipped steps are predictions,
  not reuse. Ships with Sol, on ref2av at the base step count, multi-GPU
  (`coderef/LightX2V/configs/minimax_h3/decache/`). This reverses the rotation
  survey's E.5 claim that LightX2V refuses feature caching on H3. A MagCache class
  cannot be selected
  (`coderef/LightX2V/lightx2v/models/networks/minimax_h3/model.py::MiniMaxH3Model._init_infer_class`).
- **Causal streaming RefA2V** (`d43f15f7`, #1539; `40744764`, #1542; prompt travel
  `d46ab933`, #1568). A second KV-cached causal H3 runtime beside TaoMate: its own
  full checkpoint of unstated provenance; a rolling K/V cache per denoising step
  filled by the noisy forward; text cached by prefill; cached keys re-roped into
  bounded slots; driving audio clean in the target rows at timestep 1.0, which
  implies weights trained that way ([`audio_freeze.md`](../h3/audio_freeze.md)
  section 5 item 8; reasoned from the code). Prompt travel pads each re-encoded
  prompt to one fixed token length so the text prefix keeps its row count
  (`coderef/LightX2V/lightx2v/models/runners/minimax_h3_causal/action_prompt_travel.py::ActionPromptTravel`),
  the pattern to know if anything here swaps prompts per window inside one
  sequence. Needs tensor parallelism and FlashAttention-3
  (`coderef/LightX2V/lightx2v/models/networks/minimax_h3_causal/streaming.py::MiniMaxH3StreamingPlan`,
  `coderef/LightX2V/lightx2v/common/kvcache/rolling.py::StepRollingKVCachePool`).
- **A persistent AdaLN cache** (`95e9b86b`, #1503), keyed on the float32 bits of
  each timestep, on across its H3 DMD configs.
- `35d4aaa8` wraps kitchen's `int8_linear` in a `torch.library.custom_op` for
  torch.compile; core calls it unwrapped (`comfy/ops.py`) and this repo never
  compiles. `fabad304` redefined `infer_steps` as evaluations. `b6d38283` counts
  only standalone audio references against its cap. `913a974d` adds an optional
  fp32 LoRA merge. The rest is XPU, MPS (`190f2ef0`, a four-step smoke config with
  no LoRA and no shift), offload and multi-GPU.

### FastVideo

At `2b164405`. No value of the FastH3 V2 contract moved.

- **A single 4090 is measured, not on the contract as shipped** (`8b5138046`,
  #1919; `coderef/FastVideo/scripts/benchmarks/minimax_h3_4090/README.md`): a
  private pruned FP8 checkpoint, the contract's kept share and tile, the DiT
  streamed layer by layer, and in the headline rows an INT8 Q/K kernel its source
  calls experimental (`coderef/FastVideo/fastvideo/envs.py::FASTVIDEO_H3_VSA_SM89_KERNEL`).
  No recipe or table for the public checkpoint. V2's tile has a Triton route.
- **It refuses base H3 under its sparse backend**
  (`coderef/FastVideo/fastvideo/pipelines/basic/minimax_h3/vsa_guard.py::refuse_zero_initialized_h3_vsa`):
  training-free sparse attention on the base model is not something the trainer's
  own code runs.
- **A FastH3 Ref2VA PDD student** (`8444c089`, #1907; repo named only in
  `coderef/FastVideo/scripts/benchmarks/fasth3_omniref_b200/README.md`) with a
  reference-video VSA policy: each reference video its own sparse region, every
  video query keeping a set share of each reference's tiles
  (`coderef/FastVideo/docs/inference/fasth3-distilled.md`,
  `coderef/FastVideo/fastvideo/attention/backends/video_sparse_attn_h3.py`). Its
  tile has one route, sm_100a/sm_103a, no fallback. What carries to SM89 is the
  idea, already an open row: "Text and audio exact, references sparse, by
  permutation" in [`sol_attn.md`](../attention/sol_attn.md). FastVideo keeps
  conditioning dense for every other checkpoint, so upstream treats the policy as
  trained; a training-free version needs a reference capture graded first, and
  per-reference budgets would need a per-region threshold in kitchen's route kernel.
- **"FastH3 Trim"** (fewer blocks, low-rank AdaLN, V2's schedule; Spark and MLX
  recipes) and **"CompactH3"** (a dense prune on a shorter schedule, for the 5090)
  are separate checkpoints, not profiles of V2 (`e235b6a33`, `02a027c49`).
- **Ref2VA training shows references the way inference does** (`d5287f835`, #1757;
  `coderef/FastVideo/fastvideo/pipelines/basic/minimax_h3/reference.py`): stills at
  `MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE`, a reference video at the canvas and frame
  rate, the encoder's copy at `MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS`, visual anchors
  noised once at the keyframe level, reference audio clean. Agrees with
  [`references.md`](../h3/references.md) and with `reference_noise.py`'s default.
- Its PDD layer is the algorithm in `pdd_math.py`. FastH3's own scheduler takes an
  Euler step (`coderef/FastVideo/fastvideo/models/schedulers/scheduling_minimax_h3.py`).
  `8348e83e8` imports Comfy-Org's NVFP4 text encoder and dequantizes it per layer
  below Blackwell.

### DiffSynth-Studio

At `acf2ad2`.

- **Fun ControlNet Union loader** (`013296e`, #1659): a second stack of DiT blocks
  added to the main stream after a fixed block set
  (`diffsynth/models/minimax_h3_controlnet.py`, `control_hints` in
  `diffsynth/models/minimax_h3_dit.py`, `diffsynth/configs/model_configs.py`). Core
  has its own loader (`comfy/ldm/minimax/controlnet.py`, from
  `comfy_extras/nodes_model_patch.py`). Nothing here wires a ControlNet.
- **Singularity ref2va v1.3 INT8 files** (`f7b7db9`, #1684) load as ComfyUI-format
  checkpoints; its converter only strips `model.diffusion_model.`
  (`diffsynth/utils/state_dict_converters/minimax_h3_dit.py`), which corroborates
  that such a file loads in ComfyUI as it is. No graph here loads it.
- **EntroPack** (`d393669`, #1712): pre-quantized H3 DiT, encoder and VAE in its own
  `CompressedLinear` format
  (`coderef/DiffSynth-Studio/examples/minimax_h3/model_inference/MiniMax-H3-EntroPack.py`).
  The encoder packages touch the closed lane on quantising our encoder.
- Training only: a DeCFG training adapter (`ce9f454`, #1678; it agrees with
  [`prompting.md`](../prompting/prompting.md) that guidance is CFG-distilled) and
  `--audio_loss_weight` (`84f93fc`). `50e5efb` announces
  [DiffSynth-ComfyUI](https://github.com/modelscope/DiffSynth-ComfyUI), with H3
  fl2va, ref2va and pruned-NF4 graphs; not read past its file list.

### diffusers

At `1d5d056ec`.

- **The H3 VAE decoder runs in the pipeline dtype** (`51a454be9`, #14754); only the
  encoder, `quant_conv`, the norms and the LayerScale scales stay fp32
  (`coderef/diffusers/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py::AutoencoderKLMiniMaxH3._keep_in_fp32_modules`).
  On decode, diffusers moved toward core; core still keeps the scales and residual
  stream in the activation dtype (`comfy/ldm/minimax/vae.py::TransformerBlock._residual_scale`),
  which only a capture could size ([`custom_node_gaps.md`](../h3/custom_node_gaps.md)
  item 3).
- **Single-file loading for ComfyUI-format files** (`4de185d6e`, #14839) records
  three namespace facts: Comfy-Org files stack QKV as `[q;k;v]` where the release
  interleaves per head, the fused FF is `[gate;value]`, and pruned files
  (`adaln_t_table` in place of `time_embedder`) are refused
  (`coderef/diffusers/src/diffusers/loaders/single_file_utils.py::convert_minimax_h3_transformer_checkpoint_to_diffusers`).
- `e0abab83b` downcasts `position_ids` only without fp64; `fef717ffb` adds an
  SM120-only Sage backend; `80c7ed262` fixes flash and sage varlen prep under
  `torch.compile`.

### ComfyUI core

Read at `08ff3c11` (against `65787d66`, 2026-10-09). The kitchen pin is `0.2.37`
(`2472a20b`). No H3 node or template moved a widget value.

- **A checkpoint can ask for kitchen's Sol-Attn per layer** (`b26625f2`, #16831),
  and **a LoRA can set or replace that choice** (`f49c531e`, #16880, which names
  turbo LoRAs). A `comfy_attention.config` entry is a preference list with
  `comfy_kitchen_sol` and a `tau` as one method
  (`comfy/ldm/modules/attention.py::ComfyAttention`,
  `comfy/ldm/modules/attention.py::attention_comfy_kitchen_sol`; core's
  `QUANTIZATION.md`, "Diffusion attention preferences"). A LoRA key ending in
  `.config` becomes an object patch (`comfy/model_patcher.py::ModelPatcher.add_patches`).
  - Core's call passes the threshold and the scale and nothing else: no exact sink
    for text, reference or audio rows, no dense opening steps, no short-call gate.
    `MiniMaxH3Sol`'s schema (`sol_attn_h3.py`) is what ours sets on top.
  - **With one of our attention nodes wired, the file's choice is not consulted**,
    on calls the node takes or declines: an override is tried first and handed the
    stock function (`comfy/ldm/modules/attention.py::wrap_attn`).
    `MiniMaxH3ExactBlocks` sets a preference aside for its own call (`exact_blocks.py`).
  - **With no override, the file decides, silently**: every graph on stock
    attention, the baseline among them (a `low_precision_attention` of false does
    not switch it off). `check_model_contents.py` reads every named file's header
    for the key ([`checks.md`](../method/checks.md)). `MiniMaxH3LoRABranch` refuses
    a LoRA that carries one (`lora_branch.py::parse_lora`); core's loader applies it.
- **The Linear call was reworked twice with no net diff in H3's DiT** (`0752bcb2`,
  #16816; `62c49c4d`, #16861). The video VAE's fused norm now casts its weight
  inside `comfy/ops.py`. `d91ed5f5` (#16862) adds `comfy_force_forward` so a bypass
  LoRA's `forward` wins over the fused path; `MiniMaxH3LoRABranch` covers `fc2` by
  replacing the MLP's forward (`lora_branch.py::install`), so it is not caught.
- `3d9b2d55` (#16167) adds signed-policy governance of custom nodes, off in a
  stock checkout (`app/governance.py`, `GOVERNANCE_REQUIRED`). `926d828e` adds LoRA
  stack loaders.
- **Latent-mask inpaint and mask pooling** (`comfy/model_base.py::MiniMaxH3.scale_latent_inpaint`,
  `comfy/ldm/minimax/model.py::mask_row_values`): "Masks and edited video upstream".
- **No KV chunking**: core's KV-cached causal sampler
  (`comfy/k_diffusion/sampling.py::sample_ar_video`) needs a model exposing
  `init_kv_caches`, which only Causal-Wan does, and core's H3 runs one unmasked
  attention over the whole packed sequence
  ([`docs/h3/technique_transfer.md`](../h3/technique_transfer.md), fact 2). The
  "attention chunking" nodes in third-party H3 packs slice queries to cut peak VRAM
  and carry no cache.

---

## The ComfyUI-side references

| checkout | last read | what it is |
|---|---|---|
| `comfy-kitchen` | `7490d87`; upstream main unmoved past the installed base on 2026-10-09 | **the build source**: the owner's fork. We build and install `h3-frontier` (upstream main plus our Sol commits) from the clone itself, so this checkout is what is installed. `h3-build`, the retired tag-based line every record before 2026-09-27 cites, stays as a branch. `vendor/rebuild_kernel.sh`'s "The build branch" header owns the layout; `--check` prints the upstream base, how far main has moved, our commits and the submodule pins. Import the installed Python rather than requiring the clone. Open and not carried: kitchen #234 (kijai, draft; tunes INT8 and Sol for capability 8.0 to 8.8, leaves Ada, would conflict with `comfy_kitchen/backends/cuda/__init__.py`), #224 ([`sol_upstream.md`](../attention/sol_upstream.md)), #168 (ours) |
| `comfy-kitchen-kijai` | `bd3fc78` | kijai's fork, **read-only**, renamed from `comfy-kitchen-sol` on 2026-09-08. Its `.cu` files ship in no wheel, so [`morton.md`](../attention/morton.md) quotes it by path |
| `ComfyUI-UtilsCollection` | `6fb9163` (2026-10-09) | a third-party pack with its own PDD path; two of our guards were **adopted from it**. Not installed; no shipped graph wires a `UC_*` node. Findings below |
| `Minimax-H3-Turbo` | `02e26d5` (unmoved) | the vendor README that publishes the distilled sigma grid `bench/check_distill_grid.py` grades against. Rows for the v1.0 768p 4-step and 8-step files only |
| `pdmd` | `041b70b` (2026-10-05) | **retired with its lane on 2026-10-05**; nothing reads it, and the checkout is the owner's to delete. Its scripts moved the 2-step audio shift (`AUDIO_SHIFT_2NFE`; `coderef/pdmd/worker/run_a10.py`, `run_a100.py`), which our 2-step probe never followed |
| `ComfyUI-H3-AudioRefine` | `d78d34f` (unmoved) | freezes the video stream of a sampled H3 latent and re-denoises only audio through core's per-stream `noise_mask` on the undistilled model, with a K/V cache over the frozen video rows (`coderef/ComfyUI-H3-AudioRefine/README.md`, `TECHNICAL.md`). Read for the audio-refine regime ([`audio_freeze.md`](../h3/audio_freeze.md), [`pdd.md`](../distills/pdd.md)); not installed |
| `MaskVidExperiments` | `c32ed8c` (unmoved since 2026-09-11) | video masking tools: a mask reduced to the latent grid by the VAE's frame cycle and H3's token grid, speck cleanup, frame-range and audio time-range masks, a soft Differential Diffusion, a stable subject crop sampled alone and pasted back (`coderef/MaskVidExperiments/README.md`). Read for the masked lane; not installed |
| `ComfyUI-NKD-Basic-Tools` | `4f487a5` | a crop and stitch pair that cuts a masked area at the model's resolution, samples it and puts it back with a colour match (`coderef/ComfyUI-NKD-Basic-Tools/docs/inpaint-crop-stitch.md`). For stills; nothing H3-specific; not installed |
| `ComfyUI-Minimax-H3-Continuation`, `ComfyUI-MiniMax-H3-LongMedia`, `comfyui-minimax-h3-audio-T8` | `e1768d5`, `409e4cb`, `70fb30f` (2026-10-02) | the continuation packs vllm-omni credits; compared in [`2026-09-25_continuation_guide_rows.md`](../research/2026-09-25_continuation_guide_rows.md). Not under `coderef/` at the 2026-10-05 read |
| `comfyui_dagthomas`, `h3-the-transformation-engine`, `ComfyUI-H3-Quant`, `ComfyUI-H3-Motion-Context-MultiRef` | unmoved | a third-party prompt-writer pack, the sister prompt compiler, the owner's pack published from `standalone/h3_quant`, and a multi-reference pack. None is an H3 implementation to compare against |
| `sage-fork` | `56a5be4` | our SageAttention fork |
| `SLA` | `7db4039` | the sparse top-k attention reference |
| `TurboDiffusion` | `e3d6136` (unmoved) | step-distillation reference |

### ComfyUI-UtilsCollection

- **Clip continuation, two ways.** A loop sampler (`UC_H3LoopSampler`,
  `plan_h3_schedule`) windows one whole-clip latent and carries the previous
  window's sampled tail through `noise_mask`, a prompt per window, positions
  restarted per window: the same arm as ours, but its seams are not held to the
  audio grid where ours require `39+51k`. A reader found defects by reading, none
  run: unused accepted inputs, a `preserve_input` mode that yields silent audio when
  no source is wired, an audio mask handed to the video stream as a plain tensor
  (`coderef/ComfyUI-UtilsCollection/helpers/sampling_helpers.py::start_sampling_loop`).
  The other path decodes and re-encodes: it saves the last decoded frames and audio
  (`coderef/ComfyUI-UtilsCollection/helpers/model_helpers.py::save_minimax_h3_clip_continuation_media`), writes
  core's `minimax_keyframes` (first and last tail frames as single-frame keyframes,
  tail audio as guide-audio rows; first tail frame as a Picture and the rest as a
  Video block on the encoder side,
  `coderef/ComfyUI-UtilsCollection/helpers/encoder_helpers.py::execute_advanced_minimax_h3_image_to_video`), and
  cuts at the best seam by similarity. That is the decode-and-re-encode arm
  [`audio_freeze.md`](../h3/audio_freeze.md) section 5 lists and never ran; ours
  carries the sampled latent tail (`audio_freeze.py::MiniMaxH3FreezeAudioWindow`).
  Theirs carries generated audio forward, which ours has no path for. It departs
  from the release's keyframe rules twice: a keyframe inside the clip, and one that
  carries audio. `0251732` sends the tail to Qwen at 2 fps from index 0, as core does.
- **References.** First and last frames become core-style keyframes and also go to
  Qwen as pictures (`f767043`). Native image references go to the VAE at source size
  floored to 16, with a separate smaller Qwen view
  (`coderef/ComfyUI-UtilsCollection/helpers/minimax_h3_reference_media_helpers.py`): two stills for the two
  towers departs from the vendor parity `MiniMaxH3AppendRefImage` defaults to
  ([`references.md`](../h3/references.md)). Its reference-video path truncates,
  snaps the frame count and applies the canvas rule as core does (`ed4716e`;
  `::prepare_minimax_h3_reference_video`), but orders labels images, videos, then
  all audio, turning a video's soundtrack into a standalone audio reference, which
  departs from the label rules all four implementations agree on. "Pooled" and
  "refined" reference-video modes (`6552e04`) pool the VAE latent and fit it by
  gradient steps (`coderef/ComfyUI-UtilsCollection/helpers/model_helpers.py::_pool_minimax_h3_visual_latent`,
  `::_refine_minimax_h3_visual_latent`); a pooled latent is not the encoding of a
  smaller clip. The default is the full video. `7605f6b` and `e940d37` send
  standalone reference audio to `minimax_refs` only, matching core's
  `comfy_extras/nodes_minimax_h3.py::MiniMaxH3ReferenceToVideo` (verified).
- **Encoder.** `e25109c` patches the Qwen3-VL encoder's still-image pixel bounds to
  the release's (`coderef/ComfyUI-UtilsCollection/helpers/minimax_h3_preprocessing_helpers.py`), a runtime
  workaround for [`comfyui_vendor_gaps.md`](../h3/comfyui_vendor_gaps.md) gaps 3 and
  4, stills only. An encode cache keyed on the encoder object, its patches and the
  token bytes lives in ComfyUI's temp directory and does not outlast a restart;
  `d1921ae` reverted per-section encoding after reported distortion. A "VLM guide"
  splices a separately encoded Qwen forward in front of the prompt text.
- **Prompts.** Its VLM presets keep timed timelines, the same departure from
  [`prompting.md`](../prompting/prompting.md) as vllm-omni's prompt skills (`8a61115`
  moved them closer: a shot header only at a real cut, three base fields for T2VA).
- **Not port candidates**: zero-sentinel inputs (megapixels 0, "0 preserves
  original"), which this pack refuses (`bench/check_literal_widgets.py`); a "DMAD
  re-noise" sampler (`0602581`; `coderef/ComfyUI-UtilsCollection/helpers/sampling_helpers.py::sample_dmad_renoise`),
  whose update is core's `lcm` and off every distill's contract; a megapixel
  resolution picker (`10d28c3`) that leaves the trained canvases
  ([`docs/h3/geometry.md`](../h3/geometry.md)), though its length-from-a-clip input is
  an idea we do not have; `84c0182`'s object patches on the patcher clone, including
  its own copy of core's H3 `_forward` (it edits no core file); a landmark-warped
  face composite in pixel space.

---

## The upstream and infrastructure clones

Not H3 implementations. Listed so nobody mistakes one for a comparison target.

| checkout | what it is | what it is not, or what it showed |
|---|---|---|
| `MiniMax-H3` (`d21241f`, unmoved) | the release repository | not a runnable pipeline for our purposes |
| `MiniMax-Music3` | a different model | not H3 |
| `transformers`, `vllm`, `llm-compressor` | encoder-side and quantisation infrastructure | say nothing about the DiT. A commit-message search for Qwen3-VL, MiniMax and H3 at each read through 2026-10-09 found MiniMax-M3 work and minor Qwen3-VL fixes, nothing that reaches SM89 or our encoder |
| `Model-Optimizer` (`5a7bb6764e`, 2026-10-09) | NVIDIA's PTQ/QAT toolkit and recipe catalogue (`modelopt_recipes/ptq.md`). Its `model_type/qwen3_vl/ptq` pair puts FP8 on the Qwen3-VL vision branch's Linear layers, deepstack mergers included, and leaves the patch embedding and vision attention BMM operands high precision: a scope precedent for the vision tower our encoder keeps in BF16 (the header of `h3_config.ENCODER_INT8`). Its calibration levers (MSE weight search, local Hessian, NVFP4 activation headroom, SmoothQuant alpha; its Gemma override picks the same 0.5 `channel_balance.py` defaults to). `9ceca982a6` adds a Parallel Decoding Distillation trainer with a Qwen-Image example, the first public PDD training code here (the PDD lanes are closed) | not something to run: it exports TensorRT-LLM checkpoints, its NVFP4 schemes need Blackwell, and calibrating our own encoder is a closed lane ([`roadmap.md`](../state/roadmap.md), "Closed lanes", 2026-08-27) |
| `triton` (main, past `v3.8.0`) | kernel infrastructure | main still pins the legacy `ptxas` below architecture 90 (`coderef/triton/cmake/nvidia-toolchain-version.json`, `coderef/triton/third_party/nvidia/backend/compiler.py::get_ptxas`), and `140c33fc3c` turns implicit floating-point fusion off by default; no release tag holds it as of 2026-10-09. The sage fork's `per_thread_int8`, the one Triton kernel in our render path, was bit-identical under the bundled and the system CUDA 13.2 Update 2 `ptxas` (a probe of 2026-10-05; script in the session's `internal/` notes, no committed record). Rerun that comparison when a release carries the fusion change |
| `pytorch` | trunk, not what is installed | read the installed release with `git show <tag>:<path>`; `torch.__version__` names it |
| `flashinfer`, `nanobind` | kernel infrastructure | flashinfer's H3 kernels (MXFP8 and BF16 pre-attention, Sage block-sparse) are for capability 10, 12 and SM90+ only; none runs on this card |
| `Sana` (`670482d`, `sol-engine`) | adjacent research: Sol-H3 recipes ([`sol_upstream.md`](../attention/sol_upstream.md)), HyperFlow (an 8-step adapter that swaps the DiT's time embedder for a two-time one, `coderef/Sana/models/minimax_h3/HyperFlow/hyperflow_h3/embedder.py`, eight GPUs), SoL-Refiner's H3 version (a one-step LTX-2.5 refiner over a decoded H3 clip, no audio, a full Diffusers pipeline, tested on an H100, `coderef/Sana/models/sol-refiner/MiniMax-H3/sol_refiner_h3/pipeline.py::SoLRefinerH3Pipeline`), Spark's two-stage recipe for one RTX 5090 (`1fb0648`; SM120 only, `coderef/Sana/models/minimax_h3/Sol-H3-RTX5090/runtime/offload.py`) | no ComfyUI H3 model has HyperFlow's embedder; SoL-Refiner uses no Sol on H3 and has no ComfyUI path. Its shared Sol backend table maps SM89 to `cute_sm89` |
| `h3-turbo-eval` | adjacent research | |
| `sam-audio`, `sam-3d-body`, `sapiens2`, `sam3` | Meta's perception models beside the masked lane: sound separation; a body mesh from one image; human parts and keypoints; the tracker ([`perception_models.md`](../masked_v2v/perception_models.md), [`sam31.md`](../masked_v2v/sam31.md)) | say nothing about H3; none is imported, and a check may run one as a one-off reference |
| Hugging Face, not cloned | community FastH3 conversions (NVFP4 rotated, GGUF, a dense-datafree ComfyUI file); `FastVideo/FastVideo-FastH3-Comfy`'s `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors`, which is on disk and loaded by `h3_probe_t2v_fasth3_8step` (template settings) and `h3_probe_t2v_fasth3_8step_contract`; `alibaba-pai_MiniMax-H3-Acc-LoRAs` (newest commit 2026-08-27) | `junchaoh-cs/SolarWM-H3-33B` is gated and was not read |

---

## Masks and edited video upstream, beside the masked lane

**The lane agrees with upstream on what the kept region is shown and on which
frames a latent covers; it differs deliberately on token granularity; and two
upstream ideas are untried here.** Sources (2026-10-09, the owner's ask):
vllm-omni's latent-mask editing, core, `MaskVidExperiments` and FastVideo's
reference preparation. sglang has no mask path ("Ninth read"); DiffSynth,
diffusers, LightX2V and FastVideo have no H3 mask or edit path (a search of their
H3 code). The lane's own map is [`overview.md`](../masked_v2v/overview.md).

**Where the lane and upstream agree**

- **What the model is shown in the kept region.** Core mixes the source's clean
  latent with noise at the reference level and labels those rows at that level on
  every step (`comfy/model_base.py::MiniMaxH3.scale_latent_inpaint`,
  `VISUAL_COND_TIMESTEP`), then writes the clean latent back. vllm-omni does the
  same and says it matches core
  (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/latent_mask.py::_quantize`).
  The lane samples through core. The level is the constant, not the payload's:
  `MiniMaxH3ReferenceNoise` moves how clean references are shown and does not reach
  the kept plate.
- **Which frames a latent covers.** vllm-omni pools a per-frame mask over the frame
  cycle (`coderef/vllm-omni/vllm_omni/model_executor/models/minimax_h3/encoder_processing.py::_temporal_group_max_pool`)
  that `video_mask.py::run_lengths` takes from core's `FRAME_PER_TOKEN`, with the
  maximum, as `video_mask.py::token_mask` does: one regenerated frame regenerates
  its whole latent. `MaskVidExperiments`' Mask To Latent Space is the same
  reduction, and its README shows on LTX what ComfyUI's default resize does to a
  mask that skips it.

**Where they differ**

| axis | the lane | upstream |
|---|---|---|
| granularity | a token is kept or regenerated whole (`token_mask` thresholds, then makes every cell of a token alike) | core and vllm-omni pool to the token for the label and keep the mask per latent cell for the restore, so half a token can be source; vllm-omni's frame-space intake resizes by area, so an edge token is labelled part-way (`coderef/vllm-omni/vllm_omni/model_executor/models/minimax_h3/encoder_processing.py::_resize_video_edit_mask`). Reasoned in `docs/research/masking/2026-10-05_mryellow.md` (no longer in the tree since 2026-10-11), section 1; never rendered side by side |
| task and references | ref2va with a still, and on the motion graphs the subject's own frames as a reference video | vllm-omni's editing graph runs t2va on fl2va weights with no reference; its server does not refuse references with a mask, but nothing read sends both |
| source shape | `video_mask.py::fit_frames` centre-crops, as core's nodes do | vllm-omni stretches to the canvas (`coderef/vllm-omni/vllm_omni/model_executor/models/minimax_h3/reference_video.py::prepare_edit_video`) |
| margin | `grow_pixels` or a share of the subject's size (`video_mask.py::margins`) | one token's width round the object, as advice |
| audio | the whole track frozen ([`audio_freeze.md`](../h3/audio_freeze.md)) | a fractional audio mask in some vllm-omni cases; a time-range audio mask in `MaskVidExperiments` |
| specks | `subject_tracks.py::drop_specks` judges each frame alone, by size and distance from the largest piece | `MaskVidExperiments` labels blobs across time, so a small brief blob goes and a small persistent one stays |

**What upstream does that the lane has not tried**

- **A crop round the subject, sampled alone and pasted back** (`MaskVidExperiments`'
  Subject Crop and Uncrop; NKD's crop and stitch for stills). Planned over the whole
  clip so it holds still under mask jitter and moves only with sustained motion, and
  can be enlarged to the model's resolution: the subject gets the canvas's tokens,
  where the lane gives a small subject a small share. Cost on H3 unknown: the plate
  inside is an enlargement, the room outside is unseen, and a moving crop is camera
  motion to the model. Not proposed as a build before the owner has seen it on one
  clip. (The lane's zoom, `render_region`, is its own answer since; see the runbook's
  remake table.)
- **Keeping a stretch of time and regenerating the rest**: vllm-omni's temporal mask
  node (kept length snapped to the frame rule) and `MaskVidExperiments`' frame-range
  mask. `bench/patch_render_window.py` is the lane's use of that axis, for a hole in
  the middle.
- **A soft edge during sampling**: `MaskVidExperiments`' soft Differential Diffusion
  keeps a feathered edge at every step. The lane's edge is hard at the token and
  softened only in the pixel composite; whether core's per-token label and a
  per-step threshold compose on H3 is not known.

Three questions the whole-frame lane asked the same day (the first sigmas of a late
start, the last latent frame of a window, one fraction as the mask over the whole
frame) are answered from code in
[`2026-10-09_mrfetch.md`](../research/masking/2026-10-09_mrfetch.md).

---

## The streaming references: TaoMate

Read 2026-09-15 (`TaoMate-H3` later moved to `6b2f998` in README and prompt text
only; `TaoMate-LTX` unmoved). The lane that used them was deprecated by the owner
on 2026-09-27; the checkouts stay as references.

| checkout | revision read | what it is | reach for it when |
|---|---|---|---|
| `TaoMate-H3` | `ccc1a70` | TaoLiveAIGC's streaming runtime for H3. A 3-step LoRA on the FL2VA partition (now named T2AV, FL2AV and Ref2AV listed as coming), run over each 5-second request in causal chunks: the chunk's video attends to the prompt, to a clean K/V cache and to itself, and a sigma-zero forward after each chunk commits its K/V. The cache keeps the first chunk's video as a sink and the two most recent chunks (`coderef/TaoMate-H3/src/taomate_h3/streaming/cache.py::CleanAVKVCache.retain_sink_and_recent_commits`). It accepts only 4 or 8 GPUs under TP2 with Ulysses and requires FlashAttention-3 (`coderef/TaoMate-H3/src/taomate_h3/config.py::DirectRunConfig`, `coderef/TaoMate-H3/src/taomate_h3/streaming/attention_hook.py`), so it does not run on this box. This pack's port (a converted LoRA, probe graphs, a streaming sampler node) was removed on 2026-09-27 ([`decisions.md`](../state/decisions.md)); its records stay in `bench/results/` (`2026-09-15_taomate_*`) | the adapter's distilled sigma grid (`coderef/TaoMate-H3/src/taomate_h3/model/pipeline.py`, `DISTILLED_STATE_INDICES` at its two shifts), how a KV-cached causal H3 is wired, or an H3 team's own audio-freeze regime |
| `TaoMate-LTX` | `136d890` | the same group's LTX 2.3 system and the code for their paper (arXiv 2607.24359): learned persistent memory, reference-aware FiLM, a pyramid K/V retention policy, stage-parallel inference | the paper's mechanisms. Not evidence about H3 |

What the H3 checkout is not evidence of:

- **The paper's memory.** The H3 adapter holds LoRA factors on the existing
  attention and MLP linears and nothing else (the removed converter refused any
  other tensor; its record is `data/results/2026-09-15_taomate_lora_conversion.json`).
  The runtime's only appearance mechanism is an untrained per-channel
  renormalisation of each chunk to the first chunk's statistics
  (`coderef/TaoMate-H3/src/taomate_h3/streaming/runtime.py::H3StreamingRuntime._renorm_clean_video_rows`).
- **Streamed or distilled audio.** `coderef/TaoMate-H3/src/taomate_h3/teacher.py`
  runs the base model with no adapter over each request, and the streaming loop
  overwrites the adapter's audio with those states after every step, then asserts
  the published audio equals the base result
  (`coderef/TaoMate-H3/src/taomate_h3/streaming/runtime.py::_base_audio_teacher_step_callback`
  and the guard after the phase loop). The README's speed table excludes that pass
  by its own note. This is the audio-freeze regime
  ([`audio_freeze.md`](../h3/audio_freeze.md)).
- **Anything about a single-card graph.** The README's timings are its own base
  runtime on its own node.

The community ComfyUI copy, kijai's `minimax_h3_taomate_3step_lora_avg_rank_19_bf16`,
is a per-module truncated SVD of this adapter in the same qkv and SwiGLU layout. How
much of each delta it keeps is in the conversion record's `comparison`.

---

## The trap this page exists to prevent

**"Attention" and "capture" each name something at the DiT and at the Qwen3-VL
encoder; a fact about one is not a fact about the other** (root `AGENTS.md`,
"Settled about H3"). Three instances of carrying a DiT-side fact to an encoder-side
conclusion happened in a single day. The tell is a type or a module prefix, never
the vocabulary of the claim. A clone makes this worse: it is a confident,
well-written source for the wrong stage.
